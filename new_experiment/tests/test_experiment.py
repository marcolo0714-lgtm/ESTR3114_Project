"""Offline checks; no CIFAR download or full-size training is needed."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pytest
import torch
from cifar_mtl.data import Data,BalancedBatches,make_mapping,make_splits
from cifar_mtl.model import ModelConfig,MultitaskCNN
from cifar_mtl.gradients import assign,project_coefficients
from cifar_mtl.evaluation import confusion_metrics
from cifar_mtl.results import write_json,sha256,read_json,read_jsonl

ROOT=Path(__file__).resolve().parents[1]


def load_script(filename,name):
    spec=importlib.util.spec_from_file_location(name,ROOT/filename)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def fake_assets(root):
    root.mkdir()
    # All 20 tasks and 100 fine classes, tiny images/splits.
    fine=np.repeat(np.arange(100),3);coarse=fine//5
    test_fine=np.arange(100);test_coarse=test_fine//5
    images=np.random.default_rng(8).integers(0,256,(300,3,32,32),dtype=np.uint8)
    mapping=make_mapping(fine,coarse,[f'class{i}' for i in range(100)],[f'task{i}' for i in range(20)])
    train,val=make_splits(fine,1,123)
    np.savez(root/'dataset.npz',train_images=images,train_fine=fine,train_coarse=coarse,
             test_images=images[:100],test_fine=test_fine,test_coarse=test_coarse)
    np.savez(root/'split_indices.npz',train=train,validation=val)
    write_json(root/'task_mapping.json',mapping)
    manifest=dict(tasks=20,classes_per_task=5,split_counts=dict(train=200,validation=100,test=100),
                  normalization=dict(mean=[.5]*3,std=[.25]*3),
                  files={n:sha256(root/n) for n in ('dataset.npz','split_indices.npz','task_mapping.json')})
    write_json(root/'manifest.json',manifest)
    return manifest


def test_stratification_mapping_and_matched_sampling(tmp_path):
    root=tmp_path/'assets';manifest=fake_assets(root)
    data=Data(root,manifest)
    with np.load(root/'split_indices.npz') as s:
        assert not set(s['train'])&set(s['validation'])
    a=BalancedBatches(data,3,42);b=BalancedBatches(data,3,42)
    for _ in range(5):
        x,y,digest=a.next();xx,yy,other=b.next()
        assert x.shape==(20,3,3,32,32) and y.min()>=0 and y.max()<5
        torch.testing.assert_close(x,xx);assert digest==other
    state=a.state_dict();x,y,digest=a.next()
    b.load_state_dict(state);xx,yy,other=b.next()
    torch.testing.assert_close(x,xx);assert digest==other


def test_confusion_f1():
    result=confusion_metrics([[2,1],[0,1]])
    assert result['accuracy']==.75
    assert result['precision']==[1.,.5]
    assert result['recall']==[2/3,1.]
    assert result['macro_f1']==pytest.approx((.8+2/3)/2)


def test_gram_pcgrad_matches_direct_random_order():
    generator=torch.Generator().manual_seed(1)
    gradients=torch.randn(20,29,generator=generator);gradients[2].zero_()
    coeff,gram,orders=project_coefficients(gradients,generator)
    expected=gradients.clone()
    for i,order in enumerate(orders):
        assert sorted(order)==[j for j in range(20) if j!=i]
        for j in order:
            dot=expected[i]@gradients[j];norm=gradients[j].square().sum()
            if dot<0 and norm>0:expected[i]-=dot/norm*gradients[j]
    torch.testing.assert_close(coeff@gradients,expected,rtol=2e-5,atol=2e-6)


def test_joint_gradient_equivalence_and_private_heads():
    torch.manual_seed(5)
    model=MultitaskCNN(ModelConfig(tasks=3,channels=2,hidden=4))
    x=torch.randn(3,2,3,32,32);y=torch.randint(0,5,(3,2));weights=torch.tensor([.2,.3,.5])
    losses,_=model.task_losses(x,y);(losses*weights).sum().backward()
    expected={n:p.grad.clone() for n,p in model.named_parameters()}
    model.zero_grad(set_to_none=True);losses,_=model.task_losses(x,y)
    assign(model,losses,weights,False,True)
    for n,p in model.named_parameters():torch.testing.assert_close(p.grad,expected[n],atol=1e-6,rtol=1e-4)
    model.zero_grad(set_to_none=True);losses,_=model.task_losses(x,y)
    assign(model,losses,weights,True,True,torch.Generator().manual_seed(10))
    for n,p in model.named_parameters():
        if n.startswith('heads.'):torch.testing.assert_close(p.grad,expected[n],atol=1e-6,rtol=1e-4)


def test_training_resume_evaluation_analysis_and_prediction(tmp_path):
    root=tmp_path/'resources';fake_assets(root)
    train=load_script('2_train.py','train_test')
    common=['--resources',str(root),'--steps','3','--batch-per-task','1','--channels','2','--hidden','4',
            '--eval-every','2','--diagnostics-every','2','--eval-batch-size','16','--device','cpu','--save-model','--test']
    for method in ('joint','pcgrad'):
        folder=tmp_path/'results'/method
        args=train.parser().parse_args(common+['--method',method,'--output',str(folder)])
        capture=io.StringIO()
        with contextlib.redirect_stdout(capture):summary=train.train(args)
        lines=capture.getvalue().splitlines()
        assert len([line for line in lines if 'val_accuracy=' in line])==2
        assert not any('1/3 val_accuracy=' in line for line in lines)
        assert summary['complete'] and 'test_best_validation' in summary
        assert train.train(args)==summary
        baseline=torch.load(folder/'checkpoints/final.pt',weights_only=True,map_location='cpu')
        # Simulate interruption immediately before the final training update.
        # Preserve the actual step-2 state, reconstructed by interrupting a fresh run.
        interrupted=tmp_path/'interrupted'/method
        interrupted_args=train.parser().parse_args([*common,'--method',method,'--output',str(interrupted)])
        original_save=train.save_checkpoint
        def interrupt_after_checkpoint(path,state):
            original_save(path,state)
            if path.name=='last.pt' and state['step']==2:
                raise RuntimeError('Simulated interruption')
        with patch.object(train,'save_checkpoint',side_effect=interrupt_after_checkpoint):
            with pytest.raises(RuntimeError,match='Simulated interruption'):
                train.train(interrupted_args)
        interrupted_args.resume=True
        resumed=train.train(interrupted_args)
        restored=torch.load(interrupted/'checkpoints/final.pt',weights_only=True,map_location='cpu')
        for key,value in baseline['model'].items():torch.testing.assert_close(value,restored['model'][key],atol=0,rtol=0)
        assert resumed['validation_final']==summary['validation_final']
        assert [r['batch_sha256'] for r in read_jsonl(folder/'history.jsonl')]==[r['batch_sha256'] for r in read_jsonl(interrupted/'history.jsonl')]
        args.learning_rate*=2
        with pytest.raises(FileExistsError):train.train(args)
    analyze=load_script('4_analyze.py','analyze_test')
    import sys
    with patch.object(sys,'argv',['4_analyze.py','--results',str(tmp_path/'results'),'--output',str(tmp_path/'analysis')]):
        analyze.main()
    assert (tmp_path/'analysis/per_class_metrics.csv').exists()
    assert (tmp_path/'analysis/summary_convergence.png').exists()
    from PIL import Image
    image=tmp_path/'example.png';Image.fromarray(np.zeros((32,32,3),dtype=np.uint8)).save(image)
    predict=load_script('5_predict.py','predict_test')
    with patch.object(sys,'argv',['5_predict.py','--checkpoint',str(tmp_path/'results/joint/checkpoints/final.pt'),
                                  '--image',str(image),'--task','0']):predict.main()
    output=tmp_path/'results/joint/evaluation_test_final.json'
    with patch.object(sys,'argv',['5_predict.py','--checkpoint',str(tmp_path/'results/joint/checkpoints/final.pt'),
                                  '--resources',str(root),'--evaluate-split','test','--output',str(output)]):predict.main()
    assert read_json(output)['metrics']==read_json(tmp_path/'results/joint/summary.json')['test_final']


def test_sweep_launches_unbuffered_matched_conditions(tmp_path):
    import sys
    sweep=load_script('3_run_experiments.py','sweep_test')
    commands=[]
    with patch.object(sys,'argv',['3_run_experiments.py','--seeds','42','43','42','--steps','1000','--output',str(tmp_path)]), \
         patch.object(sweep.subprocess,'run',side_effect=lambda command,**kw:commands.append(command)):
        sweep.main()
    assert len(commands)==4
    train=load_script('2_train.py','sweep_train_test')
    for command in commands:
        assert command[1]=='-u'
        args=train.parser().parse_args(command[3:])
        assert args.eval_every==100 and args.diagnostics_every==100
        assert args.output.parent==tmp_path and args.steps==1000
