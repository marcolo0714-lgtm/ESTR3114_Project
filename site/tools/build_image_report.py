"""Rebuild experiment 2 report data from copies bundled inside site (no GPU/data)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    evidence = {method:json.loads((ROOT/f'experiment_data/images/{method}.json').read_text(encoding='utf-8'))
                for method in ('joint','pcgrad')}
    registry = json.loads((ROOT/'models/images/registry.json').read_text(encoding='utf-8'))
    j,p = [evidence[m]['summary'] for m in ('joint','pcgrad')]
    report = dict(tasks=registry['tasks'], seed=42, training_steps=10000,
                  settings=[
                      ['Tasks and input', '20 supplied superclasses; 5 fine classes per task; 32 x 32 RGB images.'],
                      ['Data split', '45,000 training / 5,000 validation / 10,000 held-out test images. Split seed 123; 50 validation images per fine class.'],
                      ['Shared network', 'Three convolution/ReLU/max-pool blocks (160 channels, 3 x 3 filters), followed by two 320-unit fully connected/ReLU layers. Twenty private five-class linear heads.'],
                      ['Initialization', 'Training from scratch; no pretraining, dropout or batch normalization.'],
                      ['Preprocessing', 'Training: four-pixel padding, random crop, horizontal flips. All splits: normalization using training-only channel statistics.'],
                      ['Updates and seeds', '10,000 optimizer steps; seed 42 only. 8 images per task, 160 per update; about 35.56 training epochs.'],
                      ['Optimizer and schedule', 'SGD; initial learning rate 0.01; momentum 0.9; weight decay 0.0005. Cosine decay over 10,000 steps; clipping disabled.'],
                      ['Task weights', 'Equal fixed weights, 1/20 per task, for both methods.'],
                      ['Matching', 'Identical initialization, sampling and augmentation within the seed; all 10,000 batch hashes match. Separate projection RNG.'],
                      ['Evaluation', 'Every 100 steps; evaluation batch size 128. Best checkpoint: highest mean validation accuracy, breaking ties with lower loss.'],
                      ['Hardware and precision', 'Tesla T4; float32; deterministic PyTorch operations enabled.'],
                      ['Study scope', 'Completed single-seed comparison, with held-out test evaluation of each best-validation checkpoint.']
                  ], results={}, runtime=[], task_rows=[], figures=[], confusions=[])
    for split in ('validation_final','validation_best'):
        result=[]
        for method,name in [('joint','Joint'),('pcgrad','PCGrad')]:
            s=evidence[method]['summary'];m=s[split]
            result.append(dict(id=method,name=name,step=10000 if split=='validation_final' else s['best_validation_step'],
                accuracy=f'{100*m["accuracy"]:.2f}',f1=f'{100*m["macro_f1"]:.2f}',loss=f'{m["loss"]:.4f}',
                worst=f'{100*m["worst_task_accuracy"]:.2f}'))
        report['results'][split]=result
    for method,name in [('joint','Joint'),('pcgrad','PCGrad')]:
        s=evidence[method]['summary'];d=evidence[method]['diagnostics']
        report['runtime'].append(dict(name=name,minutes=f'{s["elapsed_seconds"]/60:.2f}',
            train_minutes=f'{s["training_seconds"]/60:.2f}',per_step=f'{s["seconds_per_step"]:.4f}',
            memory=f'{s["peak_gpu_memory_bytes"]/1024**2:.1f}',
            raw=f'{d["conflict_fraction"]*100:.2f}',post=f'{d["post_conflict_fraction"]*100:.2f}',
            cosine=f'{d["mean_absolute_cosine"]:.4f}',relative=f'{d["projection_relative_norm"]*100:.2f}'))
    for a,b in zip(j['validation_final']['tasks'],p['validation_final']['tasks']):
        report['task_rows'].append(dict(name=a['name'].replace('_',' '),joint=f'{a["accuracy"]*100:.1f}',
            pcgrad=f'{b["accuracy"]*100:.1f}',delta=f'{(b["accuracy"]-a["accuracy"])*100:+.1f}'))
        item=dict(id=a['task'],name=a['name'].replace('_',' '),methods=[])
        for name,t in [('Joint',a),('PCGrad',b)]:
            item['methods'].append(dict(name=name,labels=t['class_names'],
                matrix=[dict(label=label.replace('_',' '),counts=counts)
                        for label,counts in zip(t['class_names'],t['confusion_matrix'])],
                classes=[dict(label=label.replace('_',' '),precision=f'{t["precision"][i]*100:.1f}',
                    recall=f'{t["recall"][i]*100:.1f}',f1=f'{t["f1"][i]*100:.1f}',support=t['support'][i])
                         for i,label in enumerate(t['class_names'])]))
        report['confusions'].append(item)
    report['runtime_ratio']=f'{p["elapsed_seconds"]/j["elapsed_seconds"]:.2f}'
    report['gain']=f'{100*(p["validation_final"]["accuracy"]-j["validation_final"]["accuracy"]):.2f}'
    test = json.loads((ROOT/'experiment_data/images/test_results.json').read_text(encoding='utf-8'))
    report['test'] = dict(rows=[], task_rows=[], timing=test['execution'])
    for method,name in [('joint','Joint'),('pcgrad','PCGrad')]:
        saved=test['methods'][method];m=saved['metrics']
        report['test']['rows'].append(dict(name=name,step=saved['step'],
            accuracy=f'{m["accuracy"]*100:.2f}',f1=f'{m["macro_f1"]*100:.2f}',
            loss=f'{m["loss"]:.4f}',worst=f'{m["worst_task_accuracy"]*100:.2f}',
            correct=round(m['accuracy']*m['example_count'])))
    jt,pt=[test['methods'][m]['metrics'] for m in ('joint','pcgrad')]
    report['test']['gain']=f'{(pt["accuracy"]-jt["accuracy"])*100:.2f}'
    report['test']['extra_correct']=round((pt['accuracy']-jt['accuracy'])*10000)
    report['test']['improved']=sum(b['accuracy']>a['accuracy'] for a,b in zip(jt['tasks'],pt['tasks']))
    for a,b in zip(jt['tasks'],pt['tasks']):
        report['test']['task_rows'].append(dict(name=a['name'].replace('_',' '),
            joint=f'{a["accuracy"]*100:.2f}',pcgrad=f'{b["accuracy"]*100:.2f}',
            delta=f'{(b["accuracy"]-a["accuracy"])*100:+.2f}'))
    report['figures']=[
        dict(id='learning',title='Learning curves, time and paired accuracy gap',file='comparison_10000_seed42.png',
             caption='Seed 42 only. Stars mark best-validation accuracy checkpoints. Elapsed times are recorded before the corresponding validation evaluation; timing includes prior validation and checkpoint I/O.'),
        dict(id='tasks',title='Accuracy differences across the twenty tasks',file='task_differences_10000_seed42.png',
             caption='Final-step validation accuracy: PCGrad minus joint. Positive values favor PCGrad. Each task has 250 validation images.'),
        dict(id='joint-cosines',title='Joint: mean raw task-gradient cosines',file='gradient_cosines_joint_6bd2e268b095.png',
             caption='Mean pre-projection shared-gradient cosine over 100 diagnostic samples. Task IDs follow the numbered superclass list below.'),
        dict(id='pcgrad-cosines',title='PCGrad: mean raw task-gradient cosines',file='gradient_cosines_pcgrad_6bd2e268b095.png',
             caption='These are raw gradients along the PCGrad trajectory, before surgery. They are not post-projection cosines.')]
    (ROOT/'experiment_data/image_results.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print('Built experiment-2 report from bundled evidence.')


if __name__ == '__main__': main()
