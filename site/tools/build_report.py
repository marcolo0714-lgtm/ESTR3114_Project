"""Rebuild the report from data already copied into this standalone folder."""
import csv
import json
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[1]
NAMES = {'token': 'Token-only', 'sentiment': 'Sentiment-only', 'joint': 'Joint', 'pcgrad': 'PCGrad'}
METRICS = [('token_loss', 1, 4), ('perplexity', 1, 2), ('accuracy', 100, 2), ('macro_f1', 100, 2)]


def main():
    with (ROOT/'experiment_data/metrics.csv').open(encoding='utf-8') as stream:
        source = list(csv.DictReader(stream))
    report = dict(settings=[
        ['Tasks', 'Next-token prediction and three-class sentiment: negative, neutral, positive.'],
        ['Model', 'Pretrained GPT-2 Small (12 layers, hidden size 768), with a shared backbone, an untied pretrained token head, and a randomly initialized sentiment head.'],
        ['Dataset', 'TweetEval sentiment: 45,615 training, 2,000 validation, and 12,284 test tweets.'],
        ['Budgets and seeds', '1,000 and 6,000 optimizer steps; seeds 42, 43, and 44. Batch size 8; evaluation batch size 16.'],
        ['Sequence length', 'Maximum 96 GPT-2 tokens, including EOS: at most 95 text tokens plus one EOS. Longer texts are truncated; padding is excluded from token loss.'],
        ['Optimizer', 'AdamW; learning rate 0.00002; weight decay 0.01; gradient norm clipped at 1.'],
        ['Schedule', '10% warmup followed by linear decay to zero at the chosen final step. Fresh initialization from the pretrained model for each run.'],
        ['Weights', 'Sentiment weights 0.1, 0.3, 1.0, and 3.0 at 1,000 steps. Only 3.0 at 6,000 steps. Single-task controls ignore the weight.'],
        ['Evaluation', 'Validation every 100 steps; test evaluated at the final step. Gradient diagnostics every 25 steps. Training on CUDA in Google Colab.'],
        ['Exposure', '1,000 × 8 = 8,000 training examples (~0.175 epochs). 6,000 × 8 = 48,000 (~1.05 epochs).'],
        ['Interactive demonstration', 'Four saved seed-44 models, each trained for 6,000 steps. The joint and PCGrad models use sentiment weight 3.0.'],
    ], conditions=[], demo_rows=[])
    for steps, weight in [(1000,.1),(1000,.3),(1000,1.),(1000,3.),(6000,3.)]:
        rows = []
        for method in NAMES:
            selected = [r for r in source if r['method']==method and int(r['steps'])==steps
                        and (method in ('token','sentiment') or float(r['weight'])==weight)]
            assert sorted(int(r['seed']) for r in selected) == [42,43,44]
            cells=[]
            for metric, scale, digits in METRICS:
                values=[float(r['test_'+metric])*scale for r in selected]
                suffix='%' if scale==100 else ''
                cells.append(dict(mean=mean(values), digits=digits,
                                  text=f'{mean(values):.{digits}f}{suffix} ± {stdev(values):.{digits}f}'))
            rows.append(dict(name=NAMES[method], cells=cells))
        for col in range(4):
            displayed=[round(row['cells'][col]['mean'],row['cells'][col]['digits']) for row in rows]
            best=(min if col<2 else max)(displayed)
            for row, value in zip(rows,displayed): row['cells'][col]['best']=value==best
        report['conditions'].append(dict(steps=steps,weight=f'{weight:.1f}',rows=rows))
    for method in NAMES:
        test=json.loads((ROOT/f'models/text/{method}/summary.json').read_text())['test']
        cells=[f'{test[m]*scale:.{digits}f}'+('%' if scale==100 else '') for m,scale,digits in METRICS]
        report['demo_rows'].append(dict(name=NAMES[method],cells=cells))
    report['figures']=[
        dict(file='summary_convergence.png',title='Validation learning curves',
             alt='Validation token loss and macro-F1 across 6,000 steps for the four methods.',
             caption='Thin lines are individual seeds; bold lines are seed means. The orange PCGrad curve largely covers the blue joint curve because they nearly coincide.'),
        dict(file='summary_convergence_zoomed.png',title='Joint and PCGrad on enlarged scales',
             alt='Zoomed mean validation curves and PCGrad-minus-joint differences.',
             caption='Top: steps 4,000–6,000 on tight vertical scales. Bottom: PCGrad minus joint over the full run. Values are not shifted. Zoom magnifies small differences.'),
        dict(file='summary_convergence_zoomed_seed44.png',title='Seed 44: joint and PCGrad',
             alt='Seed-44 validation curves on enlarged scales and paired differences.',
             caption='Only the original seed-44 evaluation runs are shown here; no averaging across seeds. These curves are distinct from the separately saved demo reruns.'),
        dict(file='summary_tradeoff.png',title='Validation trade-off between the tasks',
             alt='Final validation token loss versus sentiment macro-F1 for all four methods.',
             caption='Lower token loss and higher macro-F1 are preferred. Points show seeds; error bars show sample standard deviation.'),
        dict(file='summary_layer_conflicts.png',title='Gradient conflict across layers',
             alt='Line plot of raw gradient-conflict frequencies across GPT-2 layers.',
             caption='Solid: joint. Dashed: PCGrad. Each method has 240 sampled states per seed, 720 total. These frequencies are measured before projection.'),
        dict(file='summary_runtime.png',title='Recorded runtime',
             alt='Recorded training-plus-validation seconds per step for the four methods.',
             caption='Training and validation time, averaged across three seeds. Setup, final test evaluation, and saving checkpoints are excluded.'),
        dict(file='summary_weight_effects.png',title='Per-seed test performance and compute cost',
             alt='Per-seed test macro-F1, perplexity, paired sentiment differences, and runtime.',
             caption='The overview uses the single evaluated 6,000-step weight, λ = 3.0. Token-only sentiment predictions are omitted from the classification panel because that head is untrained.'),
    ]
    (ROOT/'experiment_data/text_results.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print('Generated five condition tables and seven figures in the report index.')


if __name__ == '__main__': main()
