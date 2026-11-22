"""Aggregate matched runs, class metrics, confusion counts, and paired differences."""
import argparse
import csv
from collections import defaultdict
from pathlib import Path
import statistics
import numpy as np
from cifar_mtl.results import completed_runs,index_runs,read_json,read_jsonl,settings_signature,write_json,sha256

ROOT=Path(__file__).resolve().parent


def mean_sd(values):
    return dict(mean=statistics.mean(values),sample_sd=statistics.stdev(values) if len(values)>1 else None,n=len(values))


def csv_rows(path,rows):
    if not rows:return
    with open(path,'w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader();writer.writerows(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,default=ROOT/'results')
    p.add_argument('--output',type=Path,default=ROOT/'analysis')
    p.add_argument('--steps',type=int)
    a=p.parse_args()
    runs=[r for r in completed_runs(a.results) if a.steps is None or r[1]['steps']==a.steps]
    for directory,config,summary in runs:
        for label in ('final','best_validation'):
            path=directory/f'evaluation_test_{label}.json'
            if path.is_file():
                saved=read_json(path)
                if saved['split']!='test' or saved['checkpoint_sha256']!=sha256(directory/'checkpoints'/f'{label}.pt'):
                    raise ValueError(f'Evaluation provenance mismatch: {path}')
                summary['test_'+label]=saved['metrics']
    if not runs:p.error('No completed matching runs')
    a.output.mkdir(parents=True,exist_ok=True);index_runs(a.results)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    groups=defaultdict(list)
    for path,c,s in runs:groups[(settings_signature(c),c['method'])].append((path,c,s))
    summary={};paired=[];class_rows=[];task_rows=[]
    fig,axes=plt.subplots(1,2,figsize=(12,5));runtime,rax=plt.subplots(figsize=(10,5))
    runtime_labels=[];runtime_values=[]
    taskfig,tax=plt.subplots(figsize=(13,5))
    for (signature,method),entries in groups.items():
        seeds=[c['seed'] for _,c,_ in entries]
        if len(set(seeds))!=len(seeds):raise ValueError('Duplicate seeds within a settings group')
        label=f'{method} steps={entries[0][1]["steps"]} [{signature}]'
        result=dict(seeds=seeds,settings_signature=signature)
        for key in ('validation_final','validation_best','test_final','test_best_validation'):
            available=[s[key] for _,_,s in entries if key in s]
            if not available:continue
            result[key]={metric:mean_sd([s[metric] for s in available]) for metric in
                         ('accuracy','macro_f1','loss','worst_task_accuracy')}
            for path,c,s in entries:
                if key not in s:continue
                for t in s[key]['tasks']:
                    task_rows.append(dict(method=method,signature=signature,seed=c['seed'],split_checkpoint=key,
                         task=t['task'],name=t['name'],accuracy=t['accuracy'],macro_f1=t['macro_f1'],loss=t['loss']))
                    for i,name in enumerate(t['class_names']):
                        class_rows.append(dict(method=method,signature=signature,seed=c['seed'],split_checkpoint=key,
                            task=t['task'],task_name=t['name'],class_name=name,precision=t['precision'][i],
                            recall=t['recall'][i],f1=t['f1'][i],support=t['support'][i]))
        result['seconds_per_step']=mean_sd([s['seconds_per_step'] for _,_,s in entries])
        result['peak_gpu_memory_bytes']=mean_sd([s['peak_gpu_memory_bytes'] for _,_,s in entries])
        diagnostics=[row for path,_,_ in entries for row in read_jsonl(path/'history.jsonl') if 'pairwise_cosines' in row]
        if diagnostics:
            for key in ('conflict_fraction','mean_absolute_cosine','projection_relative_norm'):
                result['sampled_'+key]=statistics.mean(r[key] for r in diagnostics)
            cos=np.mean([r['pairwise_cosines'] for r in diagnostics],axis=0)
            heat,hax=plt.subplots(figsize=(7,6));im=hax.imshow(cos,vmin=-1,vmax=1,cmap='coolwarm')
            hax.set(title=label+' raw gradient cosines',xlabel='Task ID',ylabel='Task ID')
            heat.colorbar(im,ax=hax);heat.tight_layout();heat.savefig(a.output/f'gradient_cosines_{method}_{signature}.png',dpi=150);plt.close(heat)
            write_json(a.output/f'gradient_cosines_{method}_{signature}.json',cos.tolist())
        curves=defaultdict(list)
        for path,_,_ in entries:
            for row in read_jsonl(path/'validation.jsonl'):curves[row['step']].append(row)
        xs=sorted(curves)
        for axis,metric in zip(axes,('accuracy','loss')):
            ys=[statistics.mean(r[metric] for r in curves[x]) for x in xs]
            axis.plot(xs,ys,label=label)
            if len(entries)>1:
                sd=[statistics.stdev(r[metric] for r in curves[x]) if len(curves[x])>1 else 0 for x in xs]
                axis.fill_between(xs,np.array(ys)-sd,np.array(ys)+sd,alpha=.12)
        task_accuracy=np.mean([[t['accuracy'] for t in s['validation_final']['tasks']] for _,_,s in entries],axis=0)
        tax.plot(range(len(task_accuracy)),task_accuracy,marker='o',label=label)
        runtime_labels.append(label);runtime_values.append(result['seconds_per_step']['mean'])
        # Aggregate counts across seeds; these are not averaged F1 statistics.
        for key in ('validation_final','test_final'):
            available=[s[key] for _,_,s in entries if key in s]
            if not available:continue
            matrices=np.sum([[t['confusion_matrix'] for t in s['tasks']] for s in available],axis=0)
            write_json(a.output/f'confusions_{method}_{signature}_{key}.json',dict(
                aggregation='Sum of raw counts across available seeds',
                tasks=[dict(task=t['task'],name=t['name'],class_names=t['class_names'],confusion_matrix=m.tolist())
                       for t,m in zip(available[0]['tasks'],matrices)]))
        summary[label]=result
        if method=='pcgrad':
            joint={c['seed']:(path,s) for path,c,s in groups.get((signature,'joint'),[])}
            for path,c,s in entries:
                if c['seed'] not in joint:continue
                jp,j=joint[c['seed']]
                jh=read_jsonl(jp/'history.jsonl');ph=read_jsonl(path/'history.jsonl')
                if [r['batch_sha256'] for r in jh]!=[r['batch_sha256'] for r in ph]:
                    raise ValueError('Matched methods did not see identical sampling/augmentation')
                row=dict(signature=signature,seed=c['seed'],steps=c['steps'],
                         runtime_ratio=s['elapsed_seconds']/j['elapsed_seconds'])
                for key in ('validation_final','validation_best','test_final','test_best_validation'):
                    if key in j and key in s:
                        for metric in ('accuracy','macro_f1','loss','worst_task_accuracy'):
                            row[f'{key}_{metric}_pcgrad_minus_joint']=s[key][metric]-j[key][metric]
                paired.append(row)
    axes[0].set(title='Mean validation task accuracy',xlabel='Step',ylabel='Accuracy (0..1)')
    axes[1].set(title='Mean validation task loss',xlabel='Step',ylabel='Cross-entropy')
    axes[0].legend(fontsize=7)
    for ax in axes:ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(a.output/'summary_convergence.png',dpi=160);plt.close(fig)
    tax.set(xlabel='Task ID (see task_mapping.json)',ylabel='Validation accuracy',title='Per-task performance')
    tax.legend(fontsize=7);tax.grid(alpha=.2);taskfig.tight_layout();taskfig.savefig(a.output/'summary_tasks.png',dpi=160);plt.close(taskfig)
    rax.barh(runtime_labels,runtime_values);rax.set(xlabel='Recorded seconds per optimizer step',title='Training + validation + checkpoint I/O')
    rax.tick_params(axis='y',labelsize=7);runtime.tight_layout();runtime.savefig(a.output/'summary_runtime.png',dpi=160);plt.close(runtime)
    write_json(a.output/'analysis_summary.json',summary)
    csv_rows(a.output/'paired_differences.csv',paired);csv_rows(a.output/'per_task_metrics.csv',task_rows)
    csv_rows(a.output/'per_class_metrics.csv',class_rows)
    report=['# Multitask CIFAR-100 results','',
        'Conditional five-class classification with supplied superclass; not ordinary 100-class accuracy.',
        'Accuracy and F1 below are percentages. SD is sample SD across seeds. One-seed SD is unavailable.', '',
        '| Condition | Seeds | Final validation accuracy (%) | Final validation macro-F1 (%) | Final test accuracy (%) |',
        '|---|---|---:|---:|---:|']
    def display(metric):
        return f'{100*metric["mean"]:.2f} +/- {100*metric["sample_sd"]:.2f}' if metric['sample_sd'] is not None else f'{100*metric["mean"]:.2f} (one seed)'
    for label,s in summary.items():
        test=display(s['test_final']['accuracy']) if 'test_final' in s else 'Not evaluated'
        report.append(f'| {label} | {s["seeds"]} | {display(s["validation_final"]["accuracy"])} | {display(s["validation_final"]["macro_f1"])} | {test} |')
    report+=['','Positive paired accuracy/F1 differences favor PCGrad; negative paired loss differences favor PCGrad.',
             'Paired rows require identical settings and verified batch/augmentation hashes. Do not pool different signatures.',
             'Best checkpoint selection uses mean validation accuracy, breaking ties with lower validation loss.',
             'Per-class precision/recall/F1 and raw confusion counts are retained for all evaluation checkpoints in validation.jsonl.',
             'Gradient metrics are sampled pre-projection diagnostics; post-projection pairwise cosines need not all be nonnegative.',
             'Runtime includes validation and checkpoint writes, excluding asset loading and final test evaluation. No significance is inferred.',
             'Use analysis_summary.json for separate final/best-validation metrics and test results when available.']
    (a.output/'report.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    print(f'Analyzed {len(runs)} completed runs; output={a.output}',flush=True)


if __name__=='__main__':main()
