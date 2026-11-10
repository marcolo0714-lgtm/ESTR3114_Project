"""Download exact inputs into resources/ for offline training afterwards.

Run once with network access: python 1_prepare_assets.py
"""

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "resources"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="openai-community/gpt2")
    parser.add_argument("--model-revision", default="main")
    parser.add_argument("--dataset-revision", default="main")
    args = parser.parse_args()

    # Import here so the rest of the repository can be tested without downloads.
    from datasets import load_dataset
    from huggingface_hub import HfApi, snapshot_download

    ASSETS.mkdir(exist_ok=True)
    api = HfApi()
    model_sha = api.model_info(args.model, revision=args.model_revision).sha
    dataset_sha = api.dataset_info("cardiffnlp/tweet_eval", revision=args.dataset_revision).sha
    model_dir = ASSETS / "model"
    model_dir.mkdir(exist_ok=True)
    model_snapshot = snapshot_download(
        repo_id=args.model,
        revision=model_sha,
        local_dir=model_dir,
        allow_patterns=["*.json", "*.txt", "*.safetensors", "*.model"],
    )
    # The dataset cache is also kept inside this project folder.
    data = load_dataset(
        "cardiffnlp/tweet_eval", "sentiment",
        revision=dataset_sha,
        cache_dir=str(ASSETS / "hf_cache"),
    )
    expected_labels = ["negative", "neutral", "positive"]
    actual_labels = data["train"].features["label"].names
    if actual_labels != expected_labels:
        raise ValueError(f"Unexpected sentiment labels: {actual_labels}")
    output = ASSETS / "data"
    output.mkdir(exist_ok=True)
    counts = {}
    for split in ("train", "validation", "test"):
        path = output / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as stream:
            for row in data[split]:
                stream.write(json.dumps(
                    {"text": row["text"], "label": int(row["label"])},
                    ensure_ascii=False,
                ) + "\n")
        counts[split] = len(data[split])

    manifest = {
        "model": args.model,
        "model_revision_requested": args.model_revision,
        "model_revision_resolved": model_sha,
        "model_snapshot": str(model_snapshot),
        "dataset": "cardiffnlp/tweet_eval/sentiment",
        "dataset_revision_requested": args.dataset_revision,
        "dataset_revision_resolved": dataset_sha,
        "dataset_fingerprints": {split: data[split]._fingerprint for split in counts},
        "counts": counts,
        "label_names": expected_labels,
    }
    (ASSETS / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print("Prepared:", json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
