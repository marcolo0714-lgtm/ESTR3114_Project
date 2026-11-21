"""Train one matched condition from independent prepared assets."""
import argparse
from dataclasses import asdict
import json
import math
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from src.data import Blocks, sha256
from src.evaluation import evaluate
from src.model import ConceptModel, ModelConfig
from src.optimization import assign_gradients
from src.results import METHODS, read_json, write_json, number, run_name

ROOT = Path(__file__).resolve().parent


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method', choices=METHODS, default='joint')
    p.add_argument('--weight', '--lambda', type=float, default=1.)
    p.add_argument('--beta', type=float, default=0.1)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--steps', type=int, default=6000)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--eval-batch-size', type=int, default=8)
    p.add_argument('--learning-rate', type=float, default=2e-5)
    p.add_argument('--weight-decay', type=float, default=.01)
    p.add_argument('--warmup-ratio', type=float, default=.1)
    p.add_argument('--max-grad-norm', type=float, default=1.)
    p.add_argument('--eval-every', type=int, default=100)
    p.add_argument('--diagnostics-every', type=int, default=25)
    p.add_argument('--resources', type=Path, default=ROOT/'resources')
    p.add_argument('--output', type=Path, help='Exact run directory, not experiment root')
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    p.add_argument('--test', action='store_true', help='Evaluate held-out test once at the final step')
    p.add_argument('--save-model', action='store_true', help='Save best-validation and final checkpoints')
    return p


def train(a):
    for key in ('steps', 'batch_size', 'eval_batch_size', 'eval_every', 'diagnostics_every'):
        if getattr(a, key) <= 0:
            raise ValueError(f'{key} must be positive')
    for key in ('learning_rate', 'max_grad_norm'):
        if not math.isfinite(getattr(a, key)) or getattr(a, key) <= 0:
            raise ValueError(f'{key} must be finite and positive')
    number(a.weight); number(a.beta); number(a.weight_decay)
    if not 0 <= a.warmup_ratio < 1:
        raise ValueError('warmup-ratio must be in [0, 1)')
    multitask = a.method in ('joint', 'pcgrad')
    if multitask and (a.weight <= 0 or a.beta <= 0):
        raise ValueError('Full joint/PCGrad conditions require positive weight and beta')
    if not multitask:
        a.weight = a.beta = 0.
    root = a.resources.resolve()
    manifest = read_json(root/'manifest.json')
    if sha256(root/'initial_model.pt') != manifest['initial_model_sha256']:
        raise ValueError('Initial model checksum differs from manifest')
    for split in ('train', 'validation', 'test'):
        if split == 'test' and not a.test:
            continue
        if sha256(root/'data'/f'{split}.npz') != manifest['splits'][split]['sha256']:
            raise ValueError(f'{split} data checksum differs from manifest')
    config = {k:v for k,v in vars(a).items() if k not in ('resources', 'output')}
    config.update(assets=manifest, torch_version=torch.__version__)
    config['implementation_sha256'] = {str(path.relative_to(ROOT)):sha256(path)
             for path in [ROOT/'2_train.py', *sorted((ROOT/'src').glob('*.py'))]}
    device = 'cuda' if a.device == 'auto' and torch.cuda.is_available() else 'cpu' if a.device == 'auto' else a.device
    config['device_used'] = device
    outdir = a.output or ROOT/'results'/run_name(a.method, a.weight, a.beta, a.steps, a.seed)
    if outdir.exists() and any(outdir.iterdir()):
        if (outdir/'summary.json').exists() and read_json(outdir/'config.json') == config:
            old = read_json(outdir/'summary.json')
            if old.get('complete'):
                print(f'Reusing completed run: {outdir}')
                return old
        raise FileExistsError(f'Nonempty or incompatible run directory: {outdir}')
    outdir.mkdir(parents=True, exist_ok=True)
    write_json(outdir/'config.json', config)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(a.seed)
        torch.backends.cudnn.benchmark = False
        torch.cuda.reset_peak_memory_stats()
    # Loading identical token weights, then reinitialize only concept modules
    # with the run seed. All four conditions use the same batch-order generator.
    initial = torch.load(root/'initial_model.pt', map_location='cpu', weights_only=True)
    model = ConceptModel(ModelConfig(**initial['model_config']))
    model.load_state_dict(initial['state_dict'])
    for module in (model.concept_blocks, model.concept_norm, model.concept_head, model.quantizer):
        def reset(m):
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()
        module.apply(reset)
    with torch.no_grad():
        model.quantizer.codes.normal_(std=.02)
    del initial
    model.to(device)
    use_concepts = a.method != 'token_baseline'
    if not multitask:
        for p in model.quantizer.parameters():
            p.requires_grad_(False)
    if not use_concepts:
        for module in (model.concept_blocks, model.concept_norm, model.concept_head):
            module.requires_grad_(False)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=a.learning_rate, weight_decay=a.weight_decay)
    warmup = int(a.steps*a.warmup_ratio)
    def lr_factor(step):
        if warmup and step < warmup:
            return (step+1)/warmup
        return max(0., (a.steps-step)/max(1, a.steps-warmup))
    trainset = Blocks(root/'data'/'train.npz')
    trainloader = DataLoader(trainset, batch_size=a.batch_size, shuffle=True,
                            generator=torch.Generator().manual_seed(a.seed), num_workers=0)
    val_loader = DataLoader(Blocks(root/'data'/'validation.npz'), batch_size=a.eval_batch_size)
    iterator = iter(trainloader)
    best_loss, best_step = float('inf'), 0
    final_val = None
    model.train()
    def sync():
        if device == 'cuda':
            torch.cuda.synchronize()
    def checkpoint(path, step):
        torch.save(dict(model_config=asdict(model.config), state_dict=model.state_dict(),
                        method=a.method, step=step, assets=manifest), path)
    sync()
    start = time.perf_counter()
    with open(outdir/'history.jsonl', 'w', encoding='utf-8') as history, open(outdir/'validation.jsonl', 'w', encoding='utf-8') as validations:
        for step in range(1, a.steps+1):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(trainloader)
                batch = next(iterator)
            batch = {k:v.to(device) for k,v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            output = model(**batch, use_concepts=use_concepts)
            for key in ('token_loss', 'concept_loss', 'vq_loss'):
                if not torch.isfinite(output[key]):
                    raise FloatingPointError(f'Nonfinite {key} at step {step}')
            measure = multitask and step % a.diagnostics_every == 0
            if multitask:
                diagnostics = assign_gradients(model, output, a.weight, a.beta,
                                               project=a.method == 'pcgrad', measure=measure)
            else:
                output['token_loss'].backward()
                diagnostics = {}
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), a.max_grad_norm, error_if_nonfinite=True)
            lr = a.learning_rate*lr_factor(step-1)
            for group in optimizer.param_groups:
                group['lr'] = lr
            optimizer.step()
            row = dict(step=step, learning_rate=lr, unclipped_gradient_norm=float(grad_norm),
                       **{k:float(output[k].detach()) for k in ('token_loss', 'concept_loss', 'vq_loss')})
            if measure:
                row.update(diagnostics)
            history.write(json.dumps(row, allow_nan=False)+'\n')
            history.flush()
            del output
            if step % a.eval_every == 0 or step == a.steps:
                final_val = evaluate(model, val_loader, device, use_concepts)
                validations.write(json.dumps(dict(step=step, **final_val), allow_nan=False)+'\n')
                validations.flush()
                if final_val['token_loss'] < best_loss:
                    best_loss, best_step = final_val['token_loss'], step
                    if a.save_model:
                        (outdir/'checkpoints').mkdir(exist_ok=True)
                        checkpoint(outdir/'checkpoints'/'best_validation.pt', step)
                print(f'{a.method} step={step} val_ppl={final_val["perplexity"]:.3f}', flush=True)
    if a.save_model:
        checkpoint(outdir/'checkpoints'/'final.pt', a.steps)
    sync()
    elapsed = time.perf_counter()-start
    summary = dict(complete=True, validation=final_val, best_validation_step=best_step,
                   best_validation_token_loss=best_loss, elapsed_seconds=elapsed,
                   seconds_per_step=elapsed/a.steps,
                   peak_gpu_memory_bytes=torch.cuda.max_memory_allocated() if device == 'cuda' else 0,
                   trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))
    if a.test:
        testloader = DataLoader(Blocks(root/'data'/'test.npz'), batch_size=a.eval_batch_size)
        summary['test'] = evaluate(model, testloader, device, use_concepts)
    write_json(outdir/'summary.json', summary)
    return summary


if __name__ == '__main__':
    train(parser().parse_args())
