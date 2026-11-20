# Evaluation of the sentiment-weight sweep

## Import and completeness

The source ZIP was copied to results/weight_sweep-20261120T112419Z-1-001.zip and extracted under results/weight_sweep/. Original downloads and prior comparison results were preserved. ZIP CRC validation and source/copy SHA-256 comparison passed.

Archive SHA-256: `ab1747defe26b6805ded5072e2415d063027c0f688cc4fd99bf534ba32e6283b`

The archive contains 73 files: three files per run (config.json, history.jsonl, summary.json) for 24 conditions, plus main_summary.csv. All 24 runs have 1,000 consecutive steps and finite training losses; summaries agree with final histories and CSV metrics. Each run has 2,000 validation and 12,284 test examples. The archive contains no model checkpoints; every config records save_model=false. Four standard analysis figures and analysis_summary.json were generated locally, plus figures/weight_effects.png and this report.

## Matched setup

Weights 0.1, 1.0, 3.0; seeds 42, 43, 44. Joint and PCGrad run for every weight/seed, while token-only and sentiment-only each run once per seed. All configurations record batch size 8, max length 96, learning rate 2e-5, 10% warmup, weight decay 0.01, evaluation every 100 steps, diagnostics every 25 steps, and CUDA. They use the same asset revisions and split fingerprints. GPU model is not recorded. Each run processes 8,000 examples (about 17.54% of a 45,615-example epoch); the learning rate decays to zero at step 1,000.

## Final test metrics

Mean +/- sample standard deviation across three seeds. Accuracy and F1 are percentages. Perplexity is averaged across individual run perplexities, not computed by exponentiating the mean loss.

| Condition | Token loss (lower) | Perplexity (lower) | Accuracy (%) | Macro-F1 (%) |
|---|---:|---:|---:|---:|
| token | 4.7605 +/- 0.0025 | 116.80 +/- 0.29 | 40.79 +/- 13.07 | 20.26 +/- 2.53 |
| sentiment | 6.0179 +/- 0.3903 | 433.22 +/- 180.54 | 60.56 +/- 8.95 | 55.33 +/- 14.87 |
| joint_w0.1 | 4.7581 +/- 0.0026 | 116.52 +/- 0.30 | 45.75 +/- 3.86 | 32.34 +/- 1.61 |
| pcgrad_w0.1 | 4.7581 +/- 0.0024 | 116.52 +/- 0.28 | 46.23 +/- 3.66 | 32.33 +/- 1.08 |
| joint_w1.0 | 4.7811 +/- 0.0070 | 119.24 +/- 0.84 | 52.88 +/- 4.13 | 41.28 +/- 5.88 |
| pcgrad_w1.0 | 4.7804 +/- 0.0075 | 119.16 +/- 0.89 | 52.95 +/- 4.14 | 41.46 +/- 6.12 |
| joint_w3.0 | 4.8293 +/- 0.0120 | 125.13 +/- 1.50 | 59.55 +/- 8.44 | 53.39 +/- 14.23 |
| pcgrad_w3.0 | 4.8249 +/- 0.0119 | 124.58 +/- 1.48 | 59.23 +/- 8.23 | 52.86 +/- 13.86 |

## Effect of the sentiment weight

Increasing weight from 0.1 to 3.0 increases joint test macro-F1 from 0.323351 to 0.533878 (+21.05 percentage points) and PCGrad from 0.323346 to 0.528624 (+20.53 points). Joint accuracy rises from 45.75% to 59.55%. The effect of the weight is substantially larger than the joint-versus-PCGrad difference at any tested weight. Mean validation F1 also rises monotonically: joint 0.344916 -> 0.451983 -> 0.540718; PCGrad 0.343183 -> 0.451524 -> 0.535691. Weight 3.0 is the strongest tested multitask weight for validation sentiment F1, but the sweep does not identify an optimal weight beyond this grid.

The gain has a language-model cost. Relative to the token-only mean perplexity 116.80, joint perplexity is 116.52 at weight 0.1 (-0.24%), 119.24 at weight 1.0 (+2.09%), and 125.13 at weight 3.0 (+7.13%). PCGrad is 124.58 at weight 3.0 (+6.66%). Thus weight 3.0 largely closes the sentiment-only F1 gap while preserving much better language modeling than sentiment-only perplexity 433.22. It does not strictly dominate sentiment-only, whose mean sentiment F1 remains higher.

There is a small observed average improvement over token-only language modeling at weight 0.1, alongside improved F1. This is not evidence of general Pareto dominance: token differences are small, only three seeds were tested, and the token-only sentiment head is untrained.

## PCGrad versus joint at matched weights

Differences below are PCGrad minus joint, paired by seed; positive F1 is better and negative perplexity is better.

| Weight | Mean test F1 difference (percentage points) | Test F1 wins / 3 | Mean perplexity difference | Runtime overhead |
|---|---:|---:|---:|---:|
| 0.1 | -0.0005 | 2/3 | +0.0002 | 34.31% |
| 1.0 | +0.1882 | 3/3 | -0.0804 | 34.08% |
| 3.0 | -0.5255 | 0/3 | -0.5536 | 34.26% |

At weight 0.1 the F1 difference is effectively zero. PCGrad improves accuracy by 0.48 percentage points but not mean macro-F1, illustrating why accuracy alone is insufficient. At weight 1.0 PCGrad has a small test F1 gain (+0.1882 percentage points), with a slightly lower mean perplexity, but lower mean validation F1. At weight 3.0 PCGrad lowers perplexity in every seed while lowering sentiment F1 and accuracy in every seed: it moves the trade-off toward token prediction rather than improving both tasks.

PCGrad mean validation F1 is lower than joint at every tested weight (differences -0.001733, -0.000459, -0.005028). There is no consistent held-out benefit large enough to establish practical superiority over ordinary joint training. Only three paired seeds are available: even a three-out-of-three directional win has two-sided exact sign-test p=0.25. No broad significance or universal superiority claim is justified.

### Per-seed sentiment performance

| Seed | Sentiment-only test F1 | Joint w=3 test F1 | PCGrad w=3 test F1 |
|---|---:|---:|---:|
| 42 | 0.637869 | 0.656150 | 0.648406 |
| 43 | 0.640437 | 0.567731 | 0.560685 |
| 44 | 0.381613 | 0.377754 | 0.376779 |

Seed 44 remains the weakest across sentiment-only and high-weight multitask training. Joint weight 3.0 beats sentiment-only at seed 42, but not seeds 43/44. Mean sentiment-only F1 is 0.553306 versus 0.533878 for joint weight 3.0 (a 1.94-point gap), with substantial seed variability. The short budget and random head initialization are plausible contributors; these aggregates do not identify the cause.

## Gradient conflict and magnitude

Each multitask condition has 120 diagnostic samples (40 per seed). Rates below refer to raw, pre-projection shared-backbone gradients.

| Condition | Negative cosine frequency (%) | Mean absolute cosine | Mean raw token/sentiment norm ratio |
|---|---:|---:|---:|
| joint_w0.1 | 46.67 | 0.03448 | 0.7568 |
| pcgrad_w0.1 | 45.83 | 0.03426 | 0.7595 |
| joint_w1.0 | 45.83 | 0.03503 | 0.7975 |
| pcgrad_w1.0 | 45.83 | 0.03489 | 0.7901 |
| joint_w3.0 | 41.67 | 0.04251 | 0.7979 |
| pcgrad_w3.0 | 44.17 | 0.04239 | 0.7921 |

Negative global cosines occur in roughly 42-47% of diagnostic samples, but mean absolute cosines are only 0.034-0.043: gradients are generally nearly perpendicular. Frequent conflict therefore does not mean large opposing components. Layer conflict frequencies span roughly 37.5-55.8%, without a decisive single-layer concentration. Increasing the weight changes the optimization trajectory, so frequencies across runs are not controlled before/after projection measurements.

The mean raw token/sentiment norm ratio remains about 0.76-0.80. As an indicative scale calculation, dividing by weight gives approximately 7.6, 0.8 and 0.27 for the token-to-weighted-sentiment ratio at weights 0.1, 1 and 3. This is consistent with sentiment becoming more influential at high weight. It is a ratio of sampled norms before combination, clipping and AdamW; it is not the effective parameter-update ratio.

PCGrad uses unweighted task gradients for projection and then weights the projected sentiment gradient. Positive weight scaling does not itself change the sign of a fixed gradient dot product. Post-projection cosine in the logs is derived analytically as abs(raw cosine), not independently measured. Projection is global; individual layers can still conflict.

## Learning curves

Joint/PCGrad token loss declines throughout training. Validation sentiment F1 stays relatively low at weight 0.1, is higher at weight 1, and continues improving on average toward the end at weight 3. This supports a larger role for sentiment weighting, but does not establish convergence. Mean validation F1 at selected steps:

| Condition | Step 100 | Step 300 | Step 500 | Step 800 | Step 1000 |
|---|---:|---:|---:|---:|---:|
| token | 0.1646 | 0.1646 | 0.1646 | 0.1782 | 0.1790 |
| sentiment | 0.2452 | 0.3985 | 0.4709 | 0.5416 | 0.5631 |
| joint_w0.1 | 0.2229 | 0.3130 | 0.3059 | 0.3555 | 0.3449 |
| pcgrad_w0.1 | 0.2213 | 0.3167 | 0.3192 | 0.3596 | 0.3432 |
| joint_w1.0 | 0.2614 | 0.3956 | 0.4110 | 0.4468 | 0.4520 |
| pcgrad_w1.0 | 0.2559 | 0.3973 | 0.3967 | 0.4445 | 0.4515 |
| joint_w3.0 | 0.2427 | 0.4105 | 0.4542 | 0.5166 | 0.5407 |
| pcgrad_w3.0 | 0.2434 | 0.4108 | 0.4530 | 0.5157 | 0.5357 |

Best sampled validation-F1 checkpoints vary: high-weight joint and PCGrad peak at steps 800/1000/1000 for seeds 42/43/44. Final summaries consistently use the predetermined last step, not per-seed best validation. Logs permit locating earlier peaks but do not contain the corresponding model weights or test predictions.

## Accuracy, class balance, and test generalization

Always predicting neutral gives test accuracy 48.33% and macro-F1 about 0.2172. Both weight-0.1 multitask methods average below that accuracy baseline while exceeding its macro-F1; increased weight substantially improves both metrics. The token-only model has an untrained sentiment head and should not be interpreted as a trained classifier.

The original split label counts are train negative/neutral/positive 7,093/20,673/17,849; validation 312/869/819; test 3,972/5,937/2,375. The change in class proportions may contribute to validation/test differences. Confusion matrices, classwise precision/recall, and per-example probabilities were not saved, so the summaries do not establish which classes explain poor runs or permit example-level uncertainty estimates.

## Runtime and reproducibility

Mean recorded runtime is approximately 237 seconds per joint run and 318 seconds per PCGrad run, with 34.1-34.3% overhead. Total recorded time across the 24 runs is 6,296.38 seconds (104.94 minutes). Source inspection shows this timer includes the training loop and validation but excludes setup, final test evaluation, and checkpoint saving. Runtime is not pure optimizer cost or end-to-end Colab duration.

Single-task validation and test metrics exactly match the same seeds in the previous weight-0.3 experiment, confirming that placeholder weight changes do not alter their updates. These identical baseline results are not additional independent seed replications. The earlier weight-0.3 result (joint/PCGrad F1 0.3602/0.3611) fits between weights 0.1 and 1.0; it is historical context, not a fourth weight in this archive.

## Conclusions and next steps

The dominant finding is a substantial sentiment-versus-language-model trade-off controlled by weight. Within this grid and training budget, weight 3.0 is the most promising multitask setting for sentiment, retaining token perplexity near 125 rather than sentiment-only 433. Weight 1.0 offers lower perplexity near 119 but lower sentiment F1. Weight 0.1 strongly favors token prediction. The choice requires a stated validation-based trade-off criterion.

PCGrad has no consistent practical advantage: almost no F1 change at weight 0.1, a tiny test improvement at weight 1, and lower F1 but better token prediction at weight 3, with about 34% more recorded time. The comparison supports conditional findings at these weights, not a claim that PCGrad cannot help elsewhere.

Next, extend the matched training budget with validation-based selection, record per-class metrics and confusion matrices, save at least selected final/best-validation checkpoints, and add seeds if small method differences remain of interest. Do not choose future weights or training lengths by repeatedly optimizing these already-inspected test scores; use validation or a fresh holdout. All current runs are under one fifth of an epoch and use a newly initialized sentiment head.

## Generated files

- figures/convergence.png
- figures/tradeoff.png
- figures/layer_conflicts.png
- figures/runtime.png
- figures/weight_effects.png
- analysis_summary.json
- evaluation_statistics.json
