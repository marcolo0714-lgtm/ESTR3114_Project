"""Download this experiment's OWN GPT-2 and corpus; no parent assets required."""
import argparse
from dataclasses import asdict
from pathlib import Path
import torch
from src.data import read_documents, split_documents, write_blocks, sha256
from src.model import ConceptModel, ModelConfig
from src.results import write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--resources', type=Path, default=Path(__file__).parent/'resources')
    p.add_argument('--model', default='openai-community/gpt2')
    p.add_argument('--model-revision', default='607a30d783dfa663caf39e06633721c8d4cfcd7e')
    p.add_argument('--dataset', default='Salesforce/wikitext')
    p.add_argument('--dataset-config', default='wikitext-2-raw-v1')
    p.add_argument('--dataset-revision', default='main')
    p.add_argument('--local-text', type=Path, help='Optional own JSONL/TXT corpus')
    p.add_argument('--sequence-length', type=int, default=256)
    p.add_argument('--seed', type=int, default=123)
    a = p.parse_args()
    if a.sequence_length < 8 or a.sequence_length % 4:
        p.error('sequence-length must be >=8 and divisible by 4')
    root = a.resources
    if root.exists() and any(root.iterdir()):
        p.error('resources directory is nonempty; reuse it or select a new directory')
    root.mkdir(parents=True, exist_ok=True)
    from transformers import AutoTokenizer, GPT2LMHeadModel
    from huggingface_hub import HfApi
    torch.manual_seed(a.seed)
    # Resolve mutable revisions before downloading and record immutable commits.
    api = HfApi()
    model_revision = api.model_info(a.model, revision=a.model_revision).sha
    tokenizer = AutoTokenizer.from_pretrained(a.model, revision=model_revision, cache_dir=root/'cache')
    hf = GPT2LMHeadModel.from_pretrained(a.model, revision=model_revision, cache_dir=root/'cache')
    hc = hf.config
    config = ModelConfig(vocab_size=hc.vocab_size, hidden_size=hc.n_embd,
                         token_layers=hc.n_layer, heads=hc.n_head, segments=hc.n_head,
                         max_positions=hc.n_positions, layer_norm_epsilon=hc.layer_norm_epsilon)
    if a.sequence_length > config.max_positions:
        p.error('sequence-length exceeds model positional capacity')
    model = ConceptModel(config)
    model.load_gpt2(hf)
    del hf
    tokenizer.save_pretrained(root/'tokenizer')
    torch.save({'model_config': asdict(config), 'state_dict': model.state_dict()}, root/'initial_model.pt')
    del model
    dataset_revision = None
    if a.local_text:
        splits = split_documents(read_documents(a.local_text), a.seed)
        source = dict(local_filename=a.local_text.name, sha256=sha256(a.local_text))
    else:
        from datasets import load_dataset
        dataset_revision = api.dataset_info(a.dataset, revision=a.dataset_revision).sha
        ds = load_dataset(a.dataset, a.dataset_config, revision=dataset_revision, cache_dir=str(root/'cache'))
        if not all(k in ds for k in ('train', 'validation', 'test')):
            raise ValueError('Published dataset must provide train/validation/test splits; alternatively use --local-text')
        # WikiText stores lines; assemble articles using its top-level headings.
        def documents(lines):
            docs, buffer = [], []
            for line in lines:
                s = line.strip()
                if s.startswith('= ') and s.endswith(' =') and not s.startswith('= ='):
                    if buffer:
                        docs.append('\n'.join(buffer))
                    buffer = []
                if s:
                    buffer.append(s)
            if buffer:
                docs.append('\n'.join(buffer))
            return docs
        splits, seen = {}, set()
        for split in ('train', 'validation', 'test'):
            docs = documents(ds[split]['text']) if 'wikitext' in a.dataset.lower() else ds[split]['text']
            splits[split] = []
            for doc in docs:
                if doc.strip() and doc not in seen:
                    splits[split].append(doc)
                    seen.add(doc)
        source = dict(dataset=a.dataset, config=a.dataset_config, revision=dataset_revision)
    (root/'data').mkdir()
    stats = {}
    for split, docs in splits.items():
        with open(root/'data'/f'{split}.jsonl', 'w', encoding='utf-8') as f:
            import json
            for doc in docs:
                f.write(json.dumps({'text': doc}, ensure_ascii=False)+'\n')
        stats[split] = write_blocks(docs, tokenizer, root/'data'/f'{split}.npz', a.sequence_length)
    write_json(root/'manifest.json', dict(model=a.model, model_revision=model_revision,
               source=source, seed=a.seed, sequence_length=a.sequence_length,
               initial_model_sha256=sha256(root/'initial_model.pt'), splits=stats))
    print(f'Prepared independent assets at {root}; training has not started.')


if __name__ == '__main__':
    main()
