# Evaluation of the Colab experiment

## Files and integrity

The original ZIP was copied to `results/comparison-20261119T165714Z-1-001.zip` and extracted to `results/comparison/`. The original download and existing smoke-run results were preserved. ZIP CRC validation passed; source/copy SHA-256 values match.

Archive SHA-256: `2e8a34bc82ed972641007bd189417a103ba78d543899ec825604c0454fb58164`

All four pilot runs contain 100 consecutive steps. All twelve main runs contain 1,000 consecutive steps, finite training losses, final validation metrics for 2,000 examples, and test metrics for 12,284 examples. Final validation histories match summaries; CSV metrics match per-run summaries. The archive includes four plots, aggregate analysis, and one final PCGrad checkpoint (seed 42, 652,214,341 bytes). Checkpoint presence was verified; no inference or retraining was performed.

## Experimental setup

GPT-2 Small with an untied pretrained token head and a randomly initialized three-class sentiment head. Four methods, seeds 42/43/44, batch size 8, maximum sequence length 96, learning rate 2e-5, 1,000 main steps, validation every 100 steps, gradient diagnostics every 25 steps. Configurations record CUDA and PyTorch 2.14.1+cu130. Each main run processes 8,000 training examples, approximately 17.54% of the 45,615-example training set. This is a short fine-tuning comparison, not a full-epoch training study.

## Pilot selection

| Condition | Validation token loss | Validation macro-F1 |
|---|---:|---:|
| Token control | 4.620763 | 0.203704 |
| Joint, weight 0.1 | 4.621909 | 0.234258 |
| Joint, weight 0.3 | 4.625885 | 0.306062 |
| Joint, weight 1.0 | 4.650799 | 0.281290 |

All three candidates satisfy the token-loss threshold 4.620763 x 1.05 = 4.851801. Weight 0.3 is correctly selected for its highest validation macro-F1. Its token loss is only 0.111% above the pilot control. Pilot validation uses 500 examples and seed 42; it is a noisy, short-horizon selection. The weight is tuned for joint training and reused for PCGrad, so the result compares methods at a common weight, not each method at its own optimum.

## Final held-out test results

Means +/- sample standard deviations across three seeds. Accuracy and F1 below are percentages; token loss and perplexity are lower-is-better.

| Method | Token loss | Perplexity | Accuracy (%) | Macro-F1 (%) | Recorded seconds/run |
|---|---:|---:|---:|---:|---:|
| token | 4.7605 +/- 0.0025 | 116.80 +/- 0.29 | 40.79 +/- 13.07 | 20.26 +/- 2.53 | 231.17 +/- 0.46 |
| sentiment | 6.0179 +/- 0.3903 | 433.22 +/- 180.54 | 60.56 +/- 8.95 | 55.33 +/- 14.87 | 200.35 +/- 0.13 |
| joint | 4.7635 +/- 0.0015 | 117.16 +/- 0.17 | 49.31 +/- 2.48 | 36.02 +/- 1.86 | 236.62 +/- 0.28 |
| pcgrad | 4.7636 +/- 0.0014 | 117.17 +/- 0.16 | 49.45 +/- 2.55 | 36.11 +/- 1.89 | 316.77 +/- 0.27 |

The token-only run is the language-model control; its sentiment head is untrained and its sentiment score is not a competitive classifier baseline. Sentiment-only achieves the highest F1 but its shared-backbone updates degrade compatibility with the frozen pretrained language head, giving much higher perplexity. Its seed-44 test F1 (0.381613) is far below seeds 42/43 (0.637869/0.640437), showing substantial sensitivity to initialization and early training.

Joint and PCGrad preserve token perplexity near the token-only control (about 0.30% and 0.32% higher respectively), while obtaining test F1 near 0.36. Compared with sentiment-only, their mean test F1 is about 19.3 percentage points lower. This is evidence of a task trade-off in this configuration, not of equal competence on both tasks.

## PCGrad versus joint: paired comparison

| Seed | Joint test F1 | PCGrad test F1 | PCGrad - joint |
|---|---:|---:|---:|
| 42 | 0.379330 | 0.380426 | +0.001096 |
| 43 | 0.359059 | 0.360191 | +0.001132 |
| 44 | 0.342213 | 0.342718 | +0.000505 |

PCGrad improves test F1 in all three matched seeds, but the mean gain is only 0.000911 (0.0911 percentage points). Mean accuracy rises by 0.1357 percentage points. Mean token loss increases by 0.000117 and perplexity by 0.01368; seed 42 improves token loss slightly and seeds 43/44 worsen it slightly. PCGrad therefore does not dominate joint training across both tasks.

Validation macro-F1 is marginally lower for PCGrad (0.395508 versus 0.396066). The small test advantage is not mirrored by the mean validation result. Three seed pairs are insufficient for a broad superiority claim: even three wins out of three have two-sided exact sign-test p=0.25. This is a descriptive result, not evidence that either method is universally better.

## Gradient conflict

There are 40 diagnostic samples per main joint/PCGrad run and 120 per method. Joint has negative global cosine in 54/120 samples (45.0%); PCGrad in 52/120 (43.33%). These are pre-projection measurements along different training trajectories, not before/after measurements on identical gradients. The sampling rate does not establish the fraction of all 1,000 updates projected.

Conflict is frequent but typically weak: mean absolute global cosine is 0.031783 for joint and 0.031872 for PCGrad; median absolute cosine is about 0.0224/0.0226. The task gradients are usually close to perpendicular. For a negative pair, the relative length of the component removed from either task gradient equals the absolute cosine. Thus the sampled conflicts generally imply small geometric corrections, a plausible explanation for nearly overlapping learning curves; this is an interpretation, not a demonstrated causal mechanism.

The mean raw token/sentiment gradient-norm ratio is 0.8425 for joint and 0.8368 for PCGrad. These are unweighted ratios: weight 0.3 reduces sentiment contribution. The average ratio itself does not establish effective optimization influence because PCGrad, clipping and AdamW further transform the update.

Layer conflict rates are roughly 42-57%, with no decisive concentration at one layer. PCGrad projects the complete shared gradient, not each layer independently. Layer-wise conflict can remain even when the complete projected pair has nonnegative cosine.

The logged post_projection_global_cosine is computed as abs(pre_projection_cosine) by the implementation, using the two-task projection identity; it is not an independently measured verification of projected gradients or the actual AdamW parameter update.

## Learning curves and generalization

Validation token losses fall smoothly from about 4.62 at step 100 to about 4.32 at step 1,000 for token/joint/PCGrad. Joint and PCGrad sentiment F1 rises toward roughly 0.39-0.40 and then fluctuates; their curves nearly overlap. For seeds 42/43 the best sampled joint F1 is at steps 900/800, while PCGrad peaks at step 900 for both; seed 44 peaks at step 1,000. Final-step evaluation is consistent with the fixed-budget design. These small fluctuations do not establish overfitting or the optimal training budget.

The validation/test gap is real: joint F1 0.3961/0.3602 and PCGrad 0.3955/0.3611, with token loss about 4.32/4.76. Local dataset label counts show a shift: train negative/neutral/positive = 7,093/20,673/17,849; validation = 312/869/819; test = 3,972/5,937/2,375. In particular, positive share falls from about 39% in training to 19% in test, and negative share rises from about 16% to 32%. This may contribute to generalization differences, but aggregate metrics do not isolate the cause.

An always-neutral predictor has test accuracy 48.331% and macro-F1 about 0.217222. Joint (49.313%) and PCGrad (49.449%) only modestly exceed that accuracy baseline, although both improve macro-F1 substantially. Accuracy alone would overstate their usefulness. Saved logs contain no predictions, per-class precision/recall/F1, or confusion matrices, so minority-class behavior and statistical uncertainty over test examples cannot be recovered from these summaries alone.

## Runtime and fairness

Mean recorded runtime is 236.619 seconds for joint and 316.768 seconds for PCGrad: 80.149 seconds extra per run, or 33.87% overhead. These are training-loop times including periodic and final validation. Source inspection confirms the timer stops before held-out test evaluation and checkpoint saving, and starts after data/model setup. They are not complete end-to-end job durations or pure optimizer-step times. The one saved checkpoint therefore does not explain the measured PCGrad overhead.

Main recorded times total 49.25 minutes; pilot recorded times total about 1.10 minutes. Actual notebook time is longer because setup, test evaluation and checkpoint I/O are omitted. No GPU model is recorded in configs, limiting cross-machine runtime reproducibility.

## Conclusion and next experiments

The experiment completed and supports a narrow conclusion: at weight 0.3 and a 1,000-step budget, PCGrad gives a tiny test-sentiment gain over weighted joint AdamW, almost unchanged language modeling, and appreciably higher runtime. It provides no convincing practical advantage in this run. Ordinary joint training is the simpler reference for this setting, while sentiment-only remains the strongest sentiment model at the cost of language-model performance.

Recommended follow-up order: (1) obtain confusion matrices and per-class F1 for trained joint/PCGrad models; only PCGrad seed42 is currently saved, so other models require retraining, (2) extend the training budget under a predeclared validation-based policy, (3) sweep sentiment weights for both methods with matched budgets, (4) add seeds and evaluate paired differences, and (5) measure train-only and end-to-end runtime separately and log the GPU model. Keep the current test results as reported; use validation or a fresh holdout to select future settings, rather than optimizing against these test scores.

## Checkpoint for prediction

`results/comparison/main/weight0.3_pcgrad_step1000_seed42/final_model.pt` exists alongside its tokenizer. It is final weights, not a resumable optimizer checkpoint. Prediction still requires local assets under resources/model.

```powershell
python 5_predict.py --checkpoint results/comparison/main/weight0.3_pcgrad_step1000_seed42/final_model.pt --text "I really enjoyed this movie!"
```
