"""Contract tests plus an actual tiny-GPT-2 check of token/EOS positions."""
import json
from pathlib import Path
import sys

import pytest
import torch
from fastapi.testclient import TestClient
from transformers import GPT2Config, GPT2Model

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app import create_app
from inference.model import JointGPT2
from inference.text import prepare_input, ModelStore


class Tokenizer:
    eos_token_id = 999

    def encode(self, text, add_special_tokens=False):
        return list(range(len(text.split())))


class StubStore:
    def __init__(self): self.calls = []
    def status(self): return []
    def predict(self, text, methods):
        self.calls.append((text, methods))
        return dict(results=[], total_ms=0)


@pytest.fixture
def client():
    return TestClient(create_app(store=StubStore()))


def test_four_pages_and_image_pages_are_implemented(client):
    for path, heading in [('/text','One sentence. Four methods.'),('/images','Try image models'),
                          ('/experiment-1','Do the two tasks help each other?'),('/experiment-2','Experiment 2')]:
        response = client.get(path)
        assert response.status_code == 200
        assert heading in response.text
        assert response.text.count('aria-current="page"') == 1
        assert 'reserved-space' not in response.text
    assert 'id="image-form"' in client.get('/images').text
    assert 'Validation performance' in client.get('/experiment-2').text
    assert client.post('/api/predict-image').status_code == 422


def test_report_has_all_conditions_and_local_figures(client):
    html = client.get('/experiment-1').text
    for text in ('λ = 0.1','λ = 0.3','λ = 1.0','λ = 3.0','6,000 steps','Seed 44 checkpoints'):
        assert text in html
    report = json.loads((ROOT/'experiment_data/text_results.json').read_text(encoding='utf-8'))
    assert len(report['conditions']) == 5
    for figure in report['figures']:
        assert client.get('/static/figures/'+figure['file']).status_code == 200
    assert 'https://' not in html


@pytest.mark.parametrize('body', [dict(text='   '),dict(text='x'*12001),
                                  dict(text='hello',methods=[]),dict(text='hello',methods=['unknown'])])
def test_invalid_prediction_requests_do_not_run_models(client, body):
    assert client.post('/api/predict-text', json=body).status_code == 422
    assert client.app.state.store.calls == []


def test_input_preserved_and_duplicate_methods_removed(client):
    body=dict(text='  Hello world ',methods=['joint','joint','pcgrad'])
    assert client.post('/api/predict-text',json=body).status_code == 200
    assert client.app.state.store.calls == [('  Hello world ',['joint','pcgrad'])]


def test_truncation_leaves_room_for_eos():
    ids,count,truncated = prepare_input(Tokenizer(),' '.join(['word']*110))
    assert len(ids)==96 and ids[-1]==999 and count==110 and truncated
    ids,count,truncated = prepare_input(Tokenizer(),'hello world')
    assert ids==[0,1,999] and count==2 and not truncated


def test_next_token_is_predicted_before_eos_and_sentiment_at_eos():
    torch.manual_seed(42)
    model=JointGPT2(GPT2Model(GPT2Config(n_embd=16,n_layer=1,n_head=2,vocab_size=23,n_positions=16))).eval()
    ids=torch.tensor([[4,7,9,22]])
    mask=torch.ones_like(ids)
    with torch.inference_mode():
        all_tokens,sentiment=model(ids,mask)
        next_token,pooled_sentiment=model.predict_positions(ids,mask)
    torch.testing.assert_close(next_token,all_tokens[:,-2])
    torch.testing.assert_close(pooled_sentiment,sentiment)
    assert not torch.allclose(next_token,all_tokens[:,-1])


def test_registry_paths_stay_inside_site_models():
    store=ModelStore(ROOT/'models')
    assert all(entry['available'] for entry in store.status())
    assert store.loaded == {}
    with pytest.raises(ValueError): store.local_path('../../resources/model')


def test_checkpoint_files_are_not_web_accessible(client):
    assert client.get('/models/text/token/final_model.pt').status_code==404
    assert client.get('/api/health').headers['cache-control']=='no-store'


def test_missing_checkpoint_returns_clear_service_error():
    class Missing(StubStore):
        def predict(self, text, methods): raise FileNotFoundError('private-path')
    response=TestClient(create_app(store=Missing())).post('/api/predict-text',json=dict(text='Hello'))
    assert response.status_code==503
    assert 'private-path' not in response.text
