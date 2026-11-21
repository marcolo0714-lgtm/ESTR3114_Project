import importlib.util
from pathlib import Path
import numpy as np
import pytest
import torch
from src.data import sha256, split_documents
from src.model import ConceptModel, ModelConfig
from src.optimization import combine, assign_gradients
from src.results import write_json, run_name

ROOT = Path(__file__).resolve().parents[1]


def tiny():
    return ConceptModel(ModelConfig(vocab_size=23,hidden_size=12,token_layers=2,
                         heads=3,segments=3,max_positions=16,concept_layers=1,dropout=0.))


def test_future_tokens_cannot_change_past_predictions():
    torch.manual_seed(1)
    model = tiny().eval()
    ids = torch.randint(0,23,(2,16))
    original = model(ids)['logits']
    for cut in range(1,16):
        changed = ids.clone(); changed[:,cut:] = (changed[:,cut:]+1)%23
        torch.testing.assert_close(original[:,:cut],model(changed)['logits'][:,:cut],atol=1e-6,rtol=1e-5)


def test_padding_does_not_change_loss_or_counts():
    model = tiny().eval()
    ids = torch.randint(0,23,(2,16)); mask = torch.ones_like(ids,dtype=torch.bool)
    mask[:,10:] = False
    a = model(ids,mask)
    ids[:,10:] = (ids[:,10:]+2)%23
    b = model(ids,mask)
    for k in ('token_loss','concept_loss','vq_loss'):
        torch.testing.assert_close(a[k],b[k])
    assert int(a['token_count']) == 18
    assert int(a['concept_count']) == 2


def test_quantizer_is_private_and_targets_detached():
    model = tiny()
    out = model(torch.randint(0,23,(2,16)))
    entries = list(model.quantizer.parameters())
    assert all(g is None for g in torch.autograd.grad(out['token_loss'],entries,allow_unused=True,retain_graph=True))
    assert all(g is None for g in torch.autograd.grad(out['concept_loss'],entries,allow_unused=True,retain_graph=True))
    encoder = list(model.h[0].parameters())
    assert all(g is None for g in torch.autograd.grad(out['vq_loss'],encoder,allow_unused=True,retain_graph=True))
    assert any(g is not None for g in torch.autograd.grad(out['vq_loss'],entries,allow_unused=True))


def test_pcgrad_projection_and_private_gradients():
    a,b = torch.tensor([1.,0.]),torch.tensor([-1.,1.])
    result, stats = combine([a,torch.tensor([3.]),None], [b,None,torch.tensor([4.])],2.,True)
    torch.testing.assert_close(result[0],torch.tensor([.5,2.5]))
    torch.testing.assert_close(result[1],torch.tensor([3.]))
    torch.testing.assert_close(result[2],torch.tensor([8.]))
    assert stats['gradient_conflict'] and stats['projection_relative_norm']>0


def test_joint_autograd_matches_manual_gradient_routing():
    model = tiny()
    ids = torch.randint(0,23,(2,16))
    out = model(ids)
    (out['token_loss']+.7*out['concept_loss']+.2*out['vq_loss']).backward()
    expected = [None if p.grad is None else p.grad.clone() for p in model.parameters()]
    model.zero_grad(set_to_none=True)
    assign_gradients(model,model(ids),.7,.2,False,True)
    for p,grad in zip(model.parameters(),expected):
        if grad is None:
            assert p.grad is None
        else:
            torch.testing.assert_close(p.grad,grad,atol=1e-6,rtol=1e-4)


def test_split_deduplicates_before_assignment():
    splits = split_documents([str(i) for i in range(100)]*2,1)
    sets = [set(s) for s in splits.values()]
    assert sum(map(len,sets)) == 100
    assert not sets[0]&sets[1] and not sets[0]&sets[2] and not sets[1]&sets[2]
    assert 'beta0.1' in run_name('joint',1.,.1,10,42)


def test_pcgrad_leaves_quantizer_and_decoder_updates_unchanged():
    torch.manual_seed(3)
    model = tiny()
    ids = torch.randint(0,23,(2,16))
    assign_gradients(model,model(ids),.7,.2,False,True)
    private = {n:p.grad.clone() for n,p in model.named_parameters()
               if n.startswith(('quantizer.', 'h.1.', 'ln_f.')) and p.grad is not None}
    model.zero_grad(set_to_none=True)
    assign_gradients(model,model(ids),.7,.2,True,True)
    for n,p in model.named_parameters():
        if n in private:
            torch.testing.assert_close(p.grad,private[n])


def load_script(filename,name):
    spec = importlib.util.spec_from_file_location(name,ROOT/filename)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def test_all_conditions_train_checkpoint_and_reuse(tmp_path):
    torch.set_num_threads(1)
    root = tmp_path/'resources'; (root/'data').mkdir(parents=True)
    model = tiny()
    torch.save(dict(model_config=model.save_config(),state_dict=model.state_dict()),root/'initial_model.pt')
    splits = {}
    for split in ('train','validation','test'):
        path = root/'data'/f'{split}.npz'
        np.savez(path,input_ids=np.random.default_rng(0).integers(0,23,(4,16),dtype=np.int32),
                 attention_mask=np.ones((4,16),dtype=np.uint8))
        splits[split] = dict(sha256=sha256(path))
    write_json(root/'manifest.json',dict(initial_model_sha256=sha256(root/'initial_model.pt'),splits=splits))
    trainer = load_script('2_train.py','train_test')
    for method in ('token_baseline','concept_token_only','joint','pcgrad'):
        output = tmp_path/'results'/method
        args = trainer.parser().parse_args(['--resources',str(root),'--output',str(output),
                      '--method',method,'--steps','2','--batch-size','2','--eval-every','1',
                      '--diagnostics-every','1','--device','cpu','--test','--save-model'])
        summary = trainer.train(args)
        assert summary['complete'] and np.isfinite(summary['test']['perplexity'])
        assert trainer.train(args) == summary
        saved = torch.load(output/'checkpoints'/'final.pt',weights_only=True)
        restored = ConceptModel(ModelConfig(**saved['model_config']))
        restored.load_state_dict(saved['state_dict'])
        restored.eval()
        assert torch.isfinite(restored(torch.ones((1,16),dtype=torch.long))['token_loss'])
        args.learning_rate *= 2
        with pytest.raises(FileExistsError):
            trainer.train(args)
    try:
        import matplotlib
    except ImportError:
        return
    import sys
    analysis = load_script('4_analyze.py','analysis_test')
    old = sys.argv
    try:
        sys.argv = ['4_analyze.py','--results',str(tmp_path/'results'),'--output',str(tmp_path/'analysis')]
        analysis.main()
    finally:
        sys.argv = old
    assert (tmp_path/'analysis'/'summary_convergence.png').exists()
    import json
    paired = json.loads((tmp_path/'analysis'/'paired_differences.json').read_text())
    assert len(paired)==1


def test_hf_gpt2_conversion_when_transformers_available():
    transformers = pytest.importorskip('transformers')
    hf = transformers.GPT2LMHeadModel(transformers.GPT2Config(vocab_size=23,n_embd=12,
          n_layer=2,n_head=3,n_positions=16,resid_pdrop=0,embd_pdrop=0,attn_pdrop=0)).eval()
    model = tiny().eval(); model.load_gpt2(hf)
    ids = torch.randint(0,23,(2,16))
    torch.testing.assert_close(model(ids,use_concepts=False)['logits'],hf(ids).logits,atol=2e-6,rtol=1e-4)
