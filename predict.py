"""Inspect sentiment probabilities and token perplexity from saved final weights.

Example: python predict.py --checkpoint results/pcgrad_w0.3_seed42/final_model.pt --text "I love this!"
"""

import argparse
import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F

from src.model import JointGPT2


ROOT = Path(__file__).resolve().parent
LABELS = ("negative", "neutral", "positive")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Final weights saved by train.py --save-model")
    parser.add_argument("--text", required=True)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    if args.max_length < 2:
        raise ValueError("max-length must be at least 2")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else (
        "cpu" if args.device == "auto" else args.device
    )
    from transformers import AutoTokenizer, GPT2Model

    source = ROOT / "resources" / "model"
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
    ids = tokenizer.encode(args.text, add_special_tokens=False)[:args.max_length - 1]
    ids = (ids or [tokenizer.eos_token_id]) + [tokenizer.eos_token_id]
    input_ids = torch.tensor([ids], device=device)
    attention_mask = torch.ones_like(input_ids)
    model = JointGPT2(GPT2Model.from_pretrained(source, local_files_only=True)).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    model.eval()
    with torch.no_grad():
        token_logits, sentiment_logits = model(input_ids, attention_mask)
        log_probs = F.log_softmax(token_logits[:, :-1], dim=-1)
        target_log_probs = log_probs.gather(-1, input_ids[:, 1:, None]).squeeze(-1)
        token_loss = -target_log_probs.mean().item()
        probabilities = sentiment_logits.softmax(dim=-1)[0].cpu().tolist()
    print(json.dumps({
        "label": LABELS[max(range(3), key=probabilities.__getitem__)],
        "probabilities": dict(zip(LABELS, probabilities)),
        "token_loss": token_loss,
        "perplexity": math.exp(min(token_loss, 80)),
    }, indent=2))


if __name__ == "__main__":
    main()
