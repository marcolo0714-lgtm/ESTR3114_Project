"""Predict within a known CIFAR-100 superclass from a self-contained checkpoint."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from cifar_mtl.model import ModelConfig,MultitaskCNN
from cifar_mtl.data import Data,verify_assets
from cifar_mtl.evaluation import evaluate
from cifar_mtl.results import write_json,sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--image',type=Path)
    mode.add_argument('--evaluate-split',choices=['validation','test'])
    p.add_argument('--task',help='Superclass ID/name, required with --image')
    p.add_argument('--resources',type=Path,default=Path(__file__).parent/'resources')
    p.add_argument('--output',type=Path)
    p.add_argument('--batch-size',type=int,default=128)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    a=p.parse_args()
    saved=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    mapping=saved['mapping'];tasks=mapping['tasks']
    if a.batch_size<=0:p.error('batch-size must be positive')
    if a.evaluate_split:
        manifest=verify_assets(a.resources)
        if manifest!=saved['config']['assets']:raise ValueError('Checkpoint and evaluation assets differ')
        model=MultitaskCNN(ModelConfig(**saved['model_config']))
        model.load_state_dict(saved['model']);model.to(a.device).eval()
        data=Data(a.resources,manifest,include_test=a.evaluate_split=='test')
        metrics,predictions=evaluate(model,data,a.evaluate_split,a.batch_size,a.device,True)
        result=dict(step=saved['step'],method=saved['config']['method'],split=a.evaluate_split,
                    checkpoint_sha256=sha256(a.checkpoint),metrics=metrics,predictions=predictions)
        if a.output:write_json(a.output,result)
        print(json.dumps({k:metrics[k] for k in ('accuracy','macro_f1','loss','worst_task_accuracy')},indent=2))
        return
    if a.task is None:p.error('--image requires --task')
    matches=[t['id'] for t in tasks if str(t['id'])==a.task or t['name']==a.task]
    if len(matches)!=1:p.error('Unknown task. Choose: '+', '.join(f'{t["id"]}:{t["name"]}' for t in tasks))
    task=matches[0]
    model=MultitaskCNN(ModelConfig(**saved['model_config']))
    model.load_state_dict(saved['model']);model.to(a.device).eval()
    image=Image.open(a.image).convert('RGB').resize((32,32),Image.Resampling.BILINEAR)
    x=torch.from_numpy(np.asarray(image).copy()).permute(2,0,1).float().unsqueeze(0)/255
    stats=saved['config']['assets']['normalization']
    x=(x-torch.tensor(stats['mean']).reshape(1,3,1,1))/torch.tensor(stats['std']).reshape(1,3,1,1)
    with torch.no_grad():probabilities=model(x.to(a.device),task).softmax(-1)[0].cpu().tolist()
    best=int(np.argmax(probabilities))
    print(json.dumps(dict(task=tasks[task]['name'],step=saved['step'],
        predicted_class=tasks[task]['class_names'][best],
        probabilities=dict(zip(tasks[task]['class_names'],probabilities)),
        note='Conditional five-class prediction; the superclass was supplied, not predicted.'),indent=2))


if __name__=='__main__':main()
