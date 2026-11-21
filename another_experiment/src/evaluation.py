"""Count-weighted evaluation; code usage aggregated across the complete split."""
import math
import torch


@torch.no_grad()
def evaluate(model, loader, device, use_concepts):
    was_training = model.training
    model.eval()
    token_sum = concept_sum = vq_sum = variance_sum = 0.
    token_count = concept_count = batches = 0
    counts = None
    for batch in loader:
        out = model(**{k: v.to(device) for k,v in batch.items()}, use_concepts=use_concepts)
        nt, nc = int(out['token_count']), int(out['concept_count'])
        token_sum += float(out['token_loss'])*nt
        concept_sum += float(out['concept_loss'])*nc
        # VQ uses all complete chunks, not just next-concept pairs.
        k = model.config.chunk_size
        nv = int(batch['attention_mask'].reshape(len(batch['input_ids']), -1, k).all(-1).sum()) if use_concepts else 0
        vq_sum += float(out['vq_loss'])*nv
        variance_sum += float(out.get('representation_variance', 0))
        batches += 1
        token_count += nt
        concept_count += nc
        if use_concepts:
            current = out['code_counts'].cpu()
            counts = current if counts is None else counts+current
        if batches == 1:
            vq_count = nv
        else:
            vq_count += nv
    model.train(was_training)
    if not token_count:
        raise ValueError('Evaluation has no valid predicted tokens')
    loss = token_sum/token_count
    result = dict(token_loss=loss, perplexity=math.exp(min(loss, 700)),
                  token_count=token_count, concept_loss=concept_sum/max(concept_count, 1),
                  concept_count=concept_count, vq_loss=vq_sum/max(vq_count, 1),
                  batch_mean_representation_variance=variance_sum/max(batches, 1))
    if counts is not None:
        p = counts.float()/counts.sum(-1, keepdim=True).clamp_min(1)
        result.update(codebook_perplexity=float((-(p*p.clamp_min(1e-12).log()).sum(-1)).exp().mean()),
                      codebook_usage_fraction=float((counts>0).float().mean()), code_counts=counts.tolist())
    return result
