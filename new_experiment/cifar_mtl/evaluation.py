"""Per-class F1 and raw confusion counts for twenty conditional classifiers."""
import numpy as np
import torch
from torch.nn import functional as F


def confusion_metrics(matrix):
    m=np.asarray(matrix,dtype=np.int64)
    tp=np.diag(m).astype(float);support=m.sum(1);predicted=m.sum(0)
    precision=np.divide(tp,predicted,out=np.zeros_like(tp),where=predicted!=0)
    recall=np.divide(tp,support,out=np.zeros_like(tp),where=support!=0)
    f1=np.divide(2*precision*recall,precision+recall,out=np.zeros_like(tp),where=precision+recall!=0)
    return dict(accuracy=float(tp.sum()/max(m.sum(),1)),macro_f1=float(f1.mean()),
                precision=precision.tolist(),recall=recall.tolist(),f1=f1.tolist(),support=support.tolist())


@torch.no_grad()
def evaluate(model,data,split,batch_size,device,collect_predictions=False):
    was_training=model.training;model.eval()
    images,coarse,local,ids=data.splits[split]
    tasks=[];predictions=[]
    for task in range(data.tasks):
        selected=ids[coarse[ids]==task];matrix=np.zeros((data.classes,data.classes),dtype=np.int64)
        total_loss=0.
        if not len(selected):raise ValueError(f'Empty evaluation task {task}')
        for start in range(0,len(selected),batch_size):
            indices=selected[start:start+batch_size]
            logits=model(data.normalize(images[indices]).to(device),task)
            y=local[indices].to(device);probabilities=logits.softmax(-1);pred=logits.argmax(-1)
            total_loss+=float(F.cross_entropy(logits,y,reduction='sum'))
            truth=y.cpu().numpy();guesses=pred.cpu().numpy()
            matrix+=np.bincount(truth*data.classes+guesses,minlength=data.classes**2).reshape(data.classes,data.classes)
            if collect_predictions:
                predictions.append(dict(task=task,indices=indices.tolist(),targets=truth.tolist(),
                    predictions=guesses.tolist(),probabilities=probabilities.cpu().tolist()))
        metrics=confusion_metrics(matrix)
        tasks.append(dict(task=task,name=data.mapping['tasks'][task]['name'],
                          class_names=data.mapping['tasks'][task]['class_names'],
                          loss=total_loss/len(selected),confusion_matrix=matrix.tolist(),**metrics))
    model.train(was_training)
    result=dict(loss=float(np.mean([t['loss'] for t in tasks])),
                accuracy=float(np.mean([t['accuracy'] for t in tasks])),
                macro_f1=float(np.mean([t['macro_f1'] for t in tasks])),
                worst_task_accuracy=min(t['accuracy'] for t in tasks),
                example_count=sum(sum(t['support']) for t in tasks),tasks=tasks)
    return (result,predictions) if collect_predictions else result
