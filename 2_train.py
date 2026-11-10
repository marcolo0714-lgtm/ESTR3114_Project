"""Train one condition; save JSONL curves and reproducibility metadata.

Example: python 2_train.py --method pcgrad --weight 0.3 --seed 42
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data import TweetDataset, make_collator
from src.evaluation import evaluate
from src.gradients import assign_joint_gradients
from src.model import JointGPT2, task_losses


ROOT = Path(__file__).resolve().parent


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["token", "sentiment", "joint", "pcgrad"], default="joint")
    parser.add_argument("--weight", type=float, default=1.0, help="Sentiment weight lambda for joint methods")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--diagnostics-every", type=int, default=25)
    parser.add_argument("--train-limit", type=int, default=0, help="First N examples; 0 means all")
    parser.add_argument("--validation-limit", type=int, default=0)
    parser.add_argument("--test", action="store_true", help="Evaluate test split only at end")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--save-model", action="store_true", help="Save final weights (large)")
    return parser.parse_args()


def run(args):
    if args.steps < 1 or args.eval_every < 1 or args.diagnostics_every < 1:
        raise ValueError("steps, eval-every and diagnostics-every must be positive")
    if args.max_length < 2 or args.batch_size < 1 or args.eval_batch_size < 1:
        raise ValueError("max-length >= 2 and batch sizes >= 1 are required")
    if args.method in ("joint", "pcgrad") and args.weight <= 0:
        raise ValueError("joint methods require --weight > 0")
    if args.learning_rate <= 0 or not 0 <= args.warmup_ratio < 1:
        raise ValueError("learning-rate > 0 and warmup-ratio in [0,1) are required")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          "cpu" if args.device == "auto" else args.device)
    model_dir = ROOT / "resources" / "model"
    data_dir = ROOT / "resources" / "data"
    for path in (model_dir / "config.json", data_dir / "train.jsonl", data_dir / "validation.jsonl"):
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}; run python 1_prepare_assets.py first")

    from transformers import AutoTokenizer, GPT2Model

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    collator = make_collator(tokenizer.pad_token_id)
    train_set = TweetDataset(data_dir / "train.jsonl", tokenizer, args.max_length, args.train_limit)
    val_set = TweetDataset(data_dir / "validation.jsonl", tokenizer, args.max_length, args.validation_limit)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              generator=torch.Generator().manual_seed(args.seed), collate_fn=collator)
    val_loader = DataLoader(val_set, batch_size=args.eval_batch_size, collate_fn=collator)
    backbone = GPT2Model.from_pretrained(model_dir, local_files_only=True)
    model = JointGPT2(backbone).to(device)
    model.train()
    # AdamW excludes biases and layer norms from weight decay.
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        (no_decay if name.endswith("bias") or "ln_" in name else decay).append(param)
    optimizer = torch.optim.AdamW([
        {"params": decay, "weight_decay": args.weight_decay},
        {"params": no_decay, "weight_decay": 0},
    ], lr=args.learning_rate)
    warmup = int(args.steps * args.warmup_ratio)
    def rate(step):
        if step < warmup:
            return (step + 1) / max(1, warmup)
        return max(0.0, (args.steps - step) / max(1, args.steps - warmup))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, rate)

    output = args.output or ROOT / "results" / f"{args.method}_w{args.weight:g}_seed{args.seed}"
    output.mkdir(parents=True, exist_ok=True)
    metadata = vars(args).copy()
    metadata["output"] = str(output)
    metadata["device_used"] = str(device)
    metadata["torch_version"] = torch.__version__
    metadata["numpy_version"] = np.__version__
    manifest = ROOT / "resources" / "manifest.json"
    metadata["assets"] = json.loads(manifest.read_text(encoding="utf-8")) if manifest.exists() else {}
    (output / "config.json").write_text(json.dumps(metadata, indent=2, default=str) + "\n", encoding="utf-8")

    start = time.perf_counter()
    iterator = iter(train_loader)
    with (output / "history.jsonl").open("w", encoding="utf-8") as history:
        for step in range(1, args.steps + 1):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                batch = next(iterator)
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            token_logits, sentiment_logits = model(batch["input_ids"], batch["attention_mask"])
            tok_loss, sent_loss, _, _, _ = task_losses(token_logits, sentiment_logits, batch)
            diagnostics = None
            if args.method in ("joint", "pcgrad"):
                # The joint baseline extracts task gradients only at sampled
                # diagnostic steps; PCGrad extracts them at every update.
                if args.method == "pcgrad" or step % args.diagnostics_every == 0:
                    diagnostics = assign_joint_gradients(
                        model, tok_loss, sent_loss, args.weight,
                        args.method == "pcgrad", measure=step % args.diagnostics_every == 0,
                    )
                else:
                    (tok_loss + args.weight * sent_loss).backward()
            elif args.method == "token":
                tok_loss.backward()
            else:
                sent_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            record = {
                "step": step, "train_token_loss": tok_loss.item(),
                "train_sentiment_loss": sent_loss.item(),
                "elapsed_seconds": time.perf_counter() - start,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
            if diagnostics is not None and step % args.diagnostics_every == 0:
                record["gradients"] = diagnostics
            if step % args.eval_every == 0 or step == args.steps:
                record["validation"] = evaluate(model, val_loader, device)
                print(f"{output.name}: {step}/{args.steps} "
                      f"val ppl={record['validation']['perplexity']:.2f} "
                      f"macro-F1={record['validation']['macro_f1']:.3f}", flush=True)
            history.write(json.dumps(record) + "\n")
            history.flush() if "validation" in record else None

    summary = {"method": args.method, "weight": args.weight, "seed": args.seed,
               "steps": args.steps, "elapsed_seconds": time.perf_counter() - start,
               "validation": record["validation"]}
    if args.test:
        test_set = TweetDataset(data_dir / "test.jsonl", tokenizer, args.max_length)
        test_loader = DataLoader(test_set, batch_size=args.eval_batch_size, collate_fn=collator)
        summary["test"] = evaluate(model, test_loader, device)
    if args.save_model:
        torch.save(model.state_dict(), output / "final_model.pt")
        tokenizer.save_pretrained(output / "tokenizer")
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(run(arguments()), indent=2))
