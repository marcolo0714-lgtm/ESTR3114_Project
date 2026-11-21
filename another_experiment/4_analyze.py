"""Aggregate matched runs, plot curves/trade-offs and report paired differences."""
import argparse
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path
from src.results import completed_runs, index_runs, write_json

ROOT = Path(__file__).resolve().parent


def records(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def mean_sd(values):
    return dict(mean=statistics.mean(values), sample_sd=statistics.stdev(values) if len(values)>1 else None,
                n=len(values))


def signature(config):
    settings = {k:v for k,v in config.items() if k not in ('method','seed','weight','beta')}
    return hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:12]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results', type=Path, default=ROOT/'results')
    p.add_argument('--output', type=Path, default=ROOT/'analysis')
    p.add_argument('--steps', type=int)
    a = p.parse_args()
    runs = [(d,c,s) for d,c,s in completed_runs(a.results) if a.steps is None or c['steps']==a.steps]
    if not runs:
        p.error('No completed matching runs')
    a.output.mkdir(parents=True, exist_ok=True)
    index_runs(a.results)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    groups = defaultdict(list)
    for d,c,s in runs:
        groups[(signature(c), c['method'],c['weight'],c['beta'])].append((d,c,s))
    summary, paired = {}, []
    fig, axes = plt.subplots(1,2,figsize=(13,5))
    trade, tax = plt.subplots(figsize=(9,6))
    runtime, rax = plt.subplots(figsize=(10,6))
    labels, seconds = [], []
    for key, entries in groups.items():
        sig,method,w,b = key
        seeds = [c['seed'] for _,c,_ in entries]
        if len(seeds) != len(set(seeds)):
            raise ValueError(f'Duplicate seed in group {key}; use separate results directories')
        label = f'{method} lambda={w:g} beta={b:g} steps={entries[0][1]["steps"]} [{sig}]'
        stats = dict(seeds=seeds, settings_signature=sig)
        for split in ('validation','test'):
            selected = [s[split] for _,_,s in entries if split in s]
            if selected:
                stats[split] = {metric:mean_sd([v[metric] for v in selected])
                               for metric in ('token_loss','perplexity','concept_loss','vq_loss')}
        stats['seconds_per_step'] = mean_sd([s['seconds_per_step'] for _,_,s in entries])
        diagnostics = [r for d,_,_ in entries for r in records(d/'history.jsonl') if 'gradient_cosine' in r]
        if diagnostics:
            stats['sampled_conflict_frequency'] = statistics.mean(float(r['gradient_conflict']) for r in diagnostics)
            stats['sampled_mean_absolute_cosine'] = statistics.mean(abs(r['gradient_cosine']) for r in diagnostics)
            stats['sampled_projection_relative_norm'] = statistics.mean(r['projection_relative_norm'] for r in diagnostics)
        summary[label] = stats
        curves = defaultdict(list)
        for d,_,_ in entries:
            for row in records(d/'validation.jsonl'):
                curves[row['step']].append(row)
        xs = sorted(curves)
        axes[0].plot(xs,[statistics.mean(r['token_loss'] for r in curves[x]) for x in xs],label=label)
        if method != 'token_baseline':
            axes[1].plot(xs,[statistics.mean(r['concept_loss'] for r in curves[x]) for x in xs],label=label)
            vals = stats['validation']
            tax.errorbar(vals['token_loss']['mean'],vals['concept_loss']['mean'],
                         xerr=vals['token_loss']['sample_sd'] or 0,
                         yerr=vals['concept_loss']['sample_sd'] or 0,fmt='o',label=label)
        labels.append(label); seconds.append(stats['seconds_per_step']['mean'])
        if method == 'pcgrad':
            other = groups.get((sig,'joint',w,b),[])
            joint = {c['seed']:s for _,c,s in other}
            for _,c,s in entries:
                if c['seed'] not in joint:
                    continue
                j = joint[c['seed']]
                row = dict(seed=c['seed'],weight=w,beta=b,settings_signature=sig,
                           steps=c['steps'],runtime_ratio=s['elapsed_seconds']/j['elapsed_seconds'])
                for split in ('validation','test'):
                    if split in s and split in j:
                        for metric in ('token_loss','perplexity'):
                            row[f'{split}_{metric}_pcgrad_minus_joint'] = s[split][metric]-j[split][metric]
                paired.append(row)
    axes[0].set(title='Validation token loss',xlabel='Step',ylabel='NLL (lower is better)')
    axes[1].set(title='Validation concept error (moving target)',xlabel='Step',ylabel='MSE (diagnostic only)')
    for ax in axes:
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=6)
    fig.tight_layout(); fig.savefig(a.output/'summary_convergence.png',dpi=160); plt.close(fig)
    tax.set(xlabel='Validation token loss',ylabel='Concept MSE (spaces differ between runs)',title='Validation trade-off diagnostics')
    tax.legend(fontsize=6); trade.tight_layout(); trade.savefig(a.output/'summary_tradeoff.png',dpi=160); plt.close(trade)
    rax.barh(labels,seconds); rax.set(xlabel='Seconds per step (training + validation + checkpoint I/O)',title='Recorded runtime')
    rax.tick_params(axis='y',labelsize=6); runtime.tight_layout(); runtime.savefig(a.output/'summary_runtime.png',dpi=160); plt.close(runtime)
    write_json(a.output/'analysis_summary.json',summary)
    write_json(a.output/'paired_differences.json',paired)
    lines = ['# Experiment results','', 'Means and sample SD across seeds. Negative paired token-loss differences favor PCGrad.', '',
             '| Condition | Seeds | Validation PPL | Test PPL | Seconds/step |','|---|---:|---:|---:|---:|']
    def display(v):
        return f'{v["mean"]:.4f} +/- {v["sample_sd"]:.4f}' if v['sample_sd'] is not None else f'{v["mean"]:.4f} (one seed)'
    for label,s in summary.items():
        test = display(s['test']['perplexity']) if 'test' in s else 'Not evaluated'
        lines.append(f'| {label} | {len(s["seeds"])} | {display(s["validation"]["perplexity"])} | {test} | {display(s["seconds_per_step"])} |')
    lines += ['', 'Concept MSE is not a common fixed target across independently trained models. Use held-out token loss as the primary comparison.',
              'Only identical training settings and common seeds enter paired differences. Single-seed results do not estimate seed variability.',
              '', '## Paired PCGrad minus joint', '']
    for row in paired:
        lines.append('- '+json.dumps(row,sort_keys=True))
    (a.output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(f'Analyzed {len(runs)} runs at {a.output}')


if __name__ == '__main__':
    main()
