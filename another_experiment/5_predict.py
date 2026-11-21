"""Inspect a saved checkpoint and optionally generate tokens causally."""
import argparse
import json
from pathlib import Path
import torch
from src.model import ConceptModel, ModelConfig


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--resources', type=Path, default=Path(__file__).parent/'resources')
    p.add_argument('--text', required=True)
    p.add_argument('--max-new-tokens', type=int, default=0)
    p.add_argument('--device', default='cpu', choices=['cpu','cuda'])
    a = p.parse_args()
    if a.max_new_tokens < 0:
        p.error('max-new-tokens must be nonnegative')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(a.resources/'tokenizer',local_files_only=True)
    saved = torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    model = ConceptModel(ModelConfig(**saved['model_config']))
    model.load_state_dict(saved['state_dict']); model.to(a.device).eval()
    tokens = tokenizer.encode(a.text,add_special_tokens=False)
    if not tokens:
        p.error('text must contain at least one token')
    c = model.config
    def forward(ids):
        # Masked right padding supplies complete tensor shapes, never future input.
        length = max(2*c.chunk_size, ((len(ids)+c.chunk_size-1)//c.chunk_size)*c.chunk_size)
        if length > c.max_positions:
            raise ValueError('Prompt plus generation exceeds model positional capacity')
        x = torch.full((1,length),tokenizer.eos_token_id,device=a.device,dtype=torch.long)
        mask = torch.zeros_like(x,dtype=torch.bool)
        x[0,:len(ids)] = torch.tensor(ids,device=a.device); mask[0,:len(ids)] = True
        return model(x,mask,use_concepts=saved['method']!='token_baseline')
    with torch.no_grad():
        output = forward(tokens)
        pnext = output['logits'][0,len(tokens)-1].softmax(-1)
        probs, ids = pnext.topk(5)
        result = dict(token_loss=float(output['token_loss']),
                      predicted_token_count=int(output['token_count']),
                      concept_loss=float(output['concept_loss']),
                      concept_count=int(output['concept_count']),
                      next_tokens=[dict(text=tokenizer.decode([int(i)]), probability=float(v)) for i,v in zip(ids,probs)])
        generated = list(tokens)
        for _ in range(a.max_new_tokens):
            output = forward(generated)
            token = int(output['logits'][0,len(generated)-1].argmax())
            generated.append(token)
            if token == tokenizer.eos_token_id:
                break
        result['generated_text'] = tokenizer.decode(generated)
        result['note'] = 'Concept error requires two complete chunks; zero count means unavailable. Generation is greedy.'
    print(json.dumps(result,indent=2,ensure_ascii=False))


if __name__ == '__main__':
    main()
