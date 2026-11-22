"""Train one 20-task joint/PCGrad condition and print evaluations every 100 steps."""
import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import platform
import time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
from cifar_mtl.data import Data,BalancedBatches,verify_assets
from cifar_mtl.model import ModelConfig,MultitaskCNN
from cifar_mtl.gradients import assign
from cifar_mtl.evaluation import evaluate
from cifar_mtl.results import read_json,write_json,sha256,index_runs

ROOT=Path(__file__).resolve().parent


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method',choices=['joint','pcgrad'],default='joint')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--steps',type=int,default=10000)
    p.add_argument('--batch-per-task',type=int,default=8)
    p.add_argument('--eval-batch-size',type=int,default=128)
    p.add_argument('--eval-every',type=int,default=100)
    p.add_argument('--diagnostics-every',type=int,default=100)
    p.add_argument('--learning-rate',type=float,default=.01)
    p.add_argument('--momentum',type=float,default=.9)
    p.add_argument('--weight-decay',type=float,default=.0005)
    p.add_argument('--max-grad-norm',type=float,default=0.,help='0 disables gradient clipping')
    p.add_argument('--channels',type=int,default=160)
    p.add_argument('--hidden',type=int,default=320)
    p.add_argument('--task-weights',type=float,nargs='+',help='Optional fixed positive weights, normalized to sum 1')
    p.add_argument('--no-augmentation',action='store_true')
    p.add_argument('--resources',type=Path,default=ROOT/'resources')
    p.add_argument('--output',type=Path,help='Exact run folder, not the sweep root')
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    p.add_argument('--save-model',action='store_true',help='Save final, best-validation, and resumable last checkpoints')
    p.add_argument('--resume',action='store_true',help='Resume this output folder from checkpoints/last.pt')
    p.add_argument('--test',action='store_true',help='Evaluate final and, when saved, best-validation states on test')
    return p


def save_checkpoint(path,state):
    temporary=path.with_suffix('.tmp');torch.save(state,temporary);temporary.replace(path)


def train(a):
    for key in ('steps','batch_per_task','eval_batch_size','eval_every','diagnostics_every','channels','hidden'):
        if getattr(a,key)<=0:raise ValueError(f'{key} must be positive')
    for key in ('learning_rate','momentum','weight_decay','max_grad_norm'):
        if not math.isfinite(getattr(a,key)) or getattr(a,key)<0:raise ValueError(f'{key} must be finite and nonnegative')
    if a.learning_rate==0 or a.momentum>=1:raise ValueError('learning-rate must be >0 and momentum <1')
    if a.resume and not a.save_model:raise ValueError('--resume requires --save-model')
    print(f'Preparing {a.method}, seed={a.seed}, steps={a.steps}: verifying local assets...',flush=True)
    assets=verify_assets(a.resources)
    n=assets['tasks']
    raw_weights=a.task_weights if a.task_weights is not None else [1.]*n
    if len(raw_weights)!=n or any(not math.isfinite(w) or w<=0 for w in raw_weights):
        raise ValueError(f'Provide exactly {n} finite positive task weights')
    weights=[w/sum(raw_weights) for w in raw_weights]
    device='cuda' if a.device=='auto' and torch.cuda.is_available() else 'cpu' if a.device=='auto' else a.device
    if device=='cuda' and not torch.cuda.is_available():raise ValueError('CUDA is unavailable')
    config={k:v for k,v in vars(a).items() if k not in ('output','resources','resume','task_weights')}
    config.update(task_weights=weights,assets=assets,device_used=device,
       device_name=torch.cuda.get_device_name(0) if device=='cuda' else 'cpu',
       torch_version=str(torch.__version__),numpy_version=np.__version__,python_version=platform.python_version(),
       implementation_sha256={p.relative_to(ROOT).as_posix():sha256(p)
                  for p in [ROOT/'2_train.py',*sorted((ROOT/'cifar_mtl').glob('*.py'))]})
    out=a.output or ROOT/'results'/f'{a.method}_step{a.steps}_seed{a.seed}'
    existing=out.exists() and any(out.iterdir())
    if existing:
        if not (out/'config.json').exists() or read_json(out/'config.json')!=config:
            raise FileExistsError(f'Incompatible output settings/code/assets: {out}; choose a new results root')
        if (out/'summary.json').is_file() and read_json(out/'summary.json').get('complete'):
            print(f'Reusing completed run: {out}',flush=True);return read_json(out/'summary.json')
        if not a.resume:raise FileExistsError(f'Incomplete run: {out}; use --resume with --save-model or choose a new root')
    out.mkdir(parents=True,exist_ok=True)
    if not existing:write_json(out/'config.json',config)
    torch.manual_seed(a.seed)
    torch.use_deterministic_algorithms(True)
    if device=='cuda':
        torch.cuda.manual_seed_all(a.seed);torch.backends.cudnn.benchmark=False
        torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False;torch.cuda.reset_peak_memory_stats()
    print(f'Loading data/model on {device}; output={out}',flush=True)
    data=Data(a.resources,assets,include_test=a.test)
    model=MultitaskCNN(ModelConfig(n,assets['classes_per_task'],a.channels,a.hidden)).to(device)
    optimizer=torch.optim.SGD(model.parameters(),lr=a.learning_rate,momentum=a.momentum,weight_decay=a.weight_decay)
    sampler=BalancedBatches(data,a.batch_per_task,a.seed,augment=not a.no_augmentation)
    projection_rng=torch.Generator().manual_seed(a.seed+30001)
    task_weights=torch.tensor(weights,device=device)
    start_step=0;best_accuracy=-1.;best_loss=float('inf');best_step=0
    best_metrics=None;previous_elapsed=0.;training_seconds=0.
    previous_peak_memory=0
    def sync():
        if device=='cuda':torch.cuda.synchronize()
    if a.resume and existing:
        saved=torch.load(out/'checkpoints/last.pt',map_location='cpu',weights_only=True)
        if saved['config']!=config:raise ValueError('Checkpoint settings differ')
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
        sampler.load_state_dict(saved['sampler']);projection_rng.set_state(saved['projection_rng'])
        torch.set_rng_state(saved['torch_rng'])
        if device=='cuda':torch.cuda.set_rng_state_all(saved['cuda_rng'])
        start_step=saved['step'];best_accuracy=saved['best_accuracy'];best_loss=saved['best_loss']
        best_step=saved['best_step'];best_metrics=saved['best_metrics']
        previous_elapsed=saved['elapsed_seconds'];training_seconds=saved['training_seconds']
        previous_peak_memory=saved.get('peak_gpu_memory_bytes',0)
        # Discard only records newer than the committed last checkpoint.
        for filename in ('history.jsonl','validation.jsonl'):
            path=out/filename
            rows=[r for r in read_jsonl(path) if r['step']<=start_step]
            path.write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in rows),encoding='utf-8')
        print(f'Resuming after step {start_step}; saved sampler/augmentation/projection RNG restored.',flush=True)
    if a.save_model:(out/'checkpoints').mkdir(exist_ok=True)
    sync();start=time.perf_counter();model.train()
    def elapsed():return previous_elapsed+time.perf_counter()-start
    def state(step):
        return dict(format_version=1,model=model.state_dict(),model_config=asdict(model.config),
            optimizer=optimizer.state_dict(),step=step,config=config,mapping=data.mapping,
            sampler=sampler.state_dict(),projection_rng=projection_rng.get_state(),
            torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if device=='cuda' else [],
            best_accuracy=best_accuracy,best_loss=best_loss,best_step=best_step,best_metrics=best_metrics,
            elapsed_seconds=elapsed(),training_seconds=training_seconds,
            peak_gpu_memory_bytes=max(previous_peak_memory,torch.cuda.max_memory_allocated() if device=='cuda' else 0))
    final_metrics=None
    print(f'{a.method}: {n} tasks x {a.batch_per_task} images/update; metrics every {a.eval_every} steps',flush=True)
    with open(out/'history.jsonl','a' if existing else 'w',encoding='utf-8') as history, open(out/'validation.jsonl','a' if existing else 'w',encoding='utf-8') as validations:
        for step in range(start_step+1,a.steps+1):
            sync();train_start=time.perf_counter()
            x,y,batch_hash=sampler.next();x=x.to(device);y=y.to(device)
            optimizer.zero_grad(set_to_none=True)
            losses,accuracies=model.task_losses(x,y)
            if not torch.isfinite(losses).all():raise FloatingPointError(f'Nonfinite loss at step {step}')
            measure=step%a.diagnostics_every==0 or step==a.steps
            diagnostic=assign(model,losses,task_weights,a.method=='pcgrad',measure,projection_rng)
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),a.max_grad_norm if a.max_grad_norm else float('inf'),error_if_nonfinite=True)
            lr=a.learning_rate*.5*(1+math.cos(math.pi*(step-1)/a.steps))
            for group in optimizer.param_groups:group['lr']=lr
            optimizer.step();sync();training_seconds+=time.perf_counter()-train_start
            row=dict(step=step,learning_rate=lr,loss=float((losses.detach()*task_weights).sum()),
                     task_losses=losses.detach().cpu().tolist(),task_accuracies=accuracies.detach().cpu().tolist(),
                     gradient_norm=float(norm),batch_sha256=batch_hash,
                     images_processed=step*n*a.batch_per_task,
                     equivalent_epochs=step*n*a.batch_per_task/assets['split_counts']['train'],
                     elapsed_seconds=elapsed(),**diagnostic)
            history.write(json.dumps(row,allow_nan=False)+'\n');history.flush()
            if step%a.eval_every==0 or step==a.steps:
                print(f'{a.method} seed={a.seed} step={step}/{a.steps}: evaluating validation...',flush=True)
                final_metrics=evaluate(model,data,'validation',a.eval_batch_size,device)
                validations.write(json.dumps(dict(step=step,**final_metrics),allow_nan=False)+'\n');validations.flush()
                better=(final_metrics['accuracy']>best_accuracy or
                        final_metrics['accuracy']==best_accuracy and final_metrics['loss']<best_loss)
                if better:
                    best_accuracy,best_loss,best_step=final_metrics['accuracy'],final_metrics['loss'],step
                    best_metrics=final_metrics
                    if a.save_model:save_checkpoint(out/'checkpoints/best_validation.pt',state(step))
                print(f'{out.name}: {step}/{a.steps} val_accuracy={final_metrics["accuracy"]:.4f} '
                      f'val_macro_f1={final_metrics["macro_f1"]:.4f} val_loss={final_metrics["loss"]:.4f} '
                      f'worst_task_accuracy={final_metrics["worst_task_accuracy"]:.4f} '
                      f'lr={lr:.3e} elapsed={elapsed():.1f}s',flush=True)
                if a.save_model:save_checkpoint(out/'checkpoints/last.pt',state(step))
    # A resume from a final-step last checkpoint still needs final reporting.
    if final_metrics is None:final_metrics=evaluate(model,data,'validation',a.eval_batch_size,device)
    if a.save_model:save_checkpoint(out/'checkpoints/final.pt',state(a.steps))
    sync();total_elapsed=elapsed()
    summary=dict(complete=True,validation_final=final_metrics,validation_best=best_metrics,
        best_validation_step=best_step,selection_metric='mean task validation accuracy; ties: lower mean task loss',
        elapsed_seconds=total_elapsed,training_seconds=training_seconds,seconds_per_step=total_elapsed/a.steps,
        images_processed=a.steps*n*a.batch_per_task,
        equivalent_epochs=a.steps*n*a.batch_per_task/assets['split_counts']['train'],
        parameters=sum(p.numel() for p in model.parameters()))
    if a.test:
        for label in ('final','best_validation') if a.save_model else ('final',):
            if label=='best_validation':
                saved=torch.load(out/'checkpoints/best_validation.pt',map_location=device,weights_only=True)
                model.load_state_dict(saved['model'])
            print(f'{a.method}: evaluating {label} state on test (not used for selection)...',flush=True)
            metrics,predictions=evaluate(model,data,'test',a.eval_batch_size,device,True)
            summary['test_'+label]=metrics
            write_json(out/f'predictions_test_{label}.json',predictions)
    summary['peak_gpu_memory_bytes']=max(previous_peak_memory,torch.cuda.max_memory_allocated() if device=='cuda' else 0)
    write_json(out/'summary.json',summary);index_runs(out.parent)
    print(f'Completed {a.method}; summary saved to {out/"summary.json"}',flush=True)
    return summary


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


if __name__=='__main__':train(parser().parse_args())
