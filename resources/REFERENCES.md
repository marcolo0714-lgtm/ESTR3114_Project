# Project resources

The proposal is in `../estr3114_proposal.pdf`. `prepare_assets.py` downloads
the actual model and dataset into this directory; no extra external cache is
needed for a run after preparation.

- GPT-2 Small model and tokenizer: https://huggingface.co/openai-community/gpt2
- TweetEval sentiment dataset: https://huggingface.co/datasets/cardiffnlp/tweet_eval
- Yu et al., *Gradient Surgery for Multi-Task Learning* (PCGrad): https://arxiv.org/abs/2001.06782
- Sener and Koltun, *Multi-Task Learning as Multi-Objective Optimization*: https://arxiv.org/abs/1810.04650
- Chen et al., *GradNorm: Gradient Normalization for Adaptive Loss Balancing*: https://arxiv.org/abs/1711.02257

TweetEval's sentiment labels are 0=negative, 1=neutral, 2=positive. Retain
the dataset's original train/validation/test split. Check upstream model and
dataset terms before redistributing downloaded weights or examples.
