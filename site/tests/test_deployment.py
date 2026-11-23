"""Vercel export, no-ML frontend startup, and real backend proxy round-trips."""
import base64
from io import BytesIO
import json
import os
from pathlib import Path
import subprocess
import sys

from fastapi import FastAPI
from fastapi.testclient import TestClient
import httpx
from PIL import Image
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app import app,create_app
from deployment import FRONTEND_IMAGE_LIMIT


def test_top_level_export_and_frontend_without_backend():
    assert isinstance(app,FastAPI)
    client=TestClient(create_app(frontend_only=True,inference_url=''))
    for route in ['/text','/images','/experiment-1','/experiment-2']:
        assert client.get(route).status_code==200
    assert 'up to 4 MB' in client.get('/images').text
    assert client.get('/api/health').json()['mode']=='frontend'
    for route in ['/api/models','/api/image-models']:
        data=client.get(route).json()
        assert data['location']=='remote' and 'model server' in data['message']
    assert not any(m['available'] for m in client.get('/api/models').json()['models'])
    assert client.post('/api/predict-text',json={'text':'hello'}).status_code==503
    assert client.post('/api/predict-image?task=1',content=b'image').status_code==503


def test_frontend_imports_and_serves_pages_without_model_dependencies():
    code=r'''
import importlib.abc,os,sys
class NoML(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in {'torch','numpy','PIL','transformers'}:
            raise AssertionError('Unexpected heavy dependency: '+fullname)
sys.meta_path.insert(0,NoML())
os.environ['VERCEL']='1'
os.environ.pop('SITE_MODE',None)
os.environ.pop('INFERENCE_API_URL',None)
from app import app
from fastapi.testclient import TestClient
c=TestClient(app)
for path in ['/text','/images','/experiment-1','/experiment-2','/api/models','/api/image-models']:
    assert c.get(path).status_code==200
assert app.state.frontend_only
assert not any(x in sys.modules for x in ['torch','numpy','PIL','transformers'])
print('Frontend import and all six routes passed without ML dependencies.')
'''
    run=subprocess.run([sys.executable,'-c',code],cwd=ROOT,capture_output=True,text=True,timeout=60)
    assert run.returncode==0,run.stderr


def test_proxy_preserves_text_payload_queries_and_remote_errors():
    seen=[]
    async def remote(request):
        seen.append((request.url.path,request.url.query,json.loads(request.content)))
        return httpx.Response(200,json=dict(results=[],device='cpu',total_ms=0))
    front=create_app(frontend_only=True,inference_url='https://models.example',
                     proxy_transport=httpx.MockTransport(remote))
    response=TestClient(front).post('/api/predict-text',json=dict(text='hello',methods=['joint']))
    assert response.status_code==200 and seen[0][2]['text']=='hello'
    async def fail(request):return httpx.Response(503,json={'error':'Selected checkpoint is missing.'})
    front=create_app(frontend_only=True,inference_url='https://models.example',
                     proxy_transport=httpx.MockTransport(fail))
    assert TestClient(front).post('/api/predict-text',json=dict(text='x')).status_code==503


def test_real_image_inference_through_authenticated_frontend(monkeypatch):
    torch.set_num_threads(4)
    monkeypatch.setenv('MODEL_SERVER_TOKEN','test-service-secret')
    monkeypatch.setenv('INFERENCE_API_TOKEN','test-service-secret')
    backend=create_app(frontend_only=False)
    assert TestClient(backend).get('/api/image-models').status_code==401
    frontend=create_app(frontend_only=True,inference_url='https://models.example',
                        proxy_transport=httpx.ASGITransport(app=backend))
    client=TestClient(frontend)
    metadata=client.get('/api/image-models').json()
    assert metadata['location']=='remote' and metadata['max_image_bytes']==FRONTEND_IMAGE_LIMIT
    assert all(x['checkpoints']['best_validation']['available'] for x in metadata['models'])
    buffer=BytesIO();Image.new('RGB',(23,17),(130,40,70)).save(buffer,format='PNG')
    response=client.post('/api/predict-image?task=1&methods=joint&methods=pcgrad&checkpoint=final',content=buffer.getvalue())
    assert response.status_code==200,response.text
    data=response.json()
    assert len(data['results'])==2 and all(x['step']==10000 for x in data['results'])
    assert data['task']['name']=='fish'
    assert 'test-service-secret' not in response.text


def test_proxy_upload_bound_and_self_proxy_rejected():
    async def unused(request):raise AssertionError('Oversize body reached backend')
    front=create_app(frontend_only=True,inference_url='https://models.example',
                     proxy_transport=httpx.MockTransport(unused))
    assert TestClient(front).post('/api/predict-image?task=1',content=b'x'*(FRONTEND_IMAGE_LIMIT+1)).status_code==413
    loop=create_app(frontend_only=True,inference_url='http://testserver')
    assert TestClient(loop).get('/api/models').status_code==503


def test_proxy_network_and_non_json_errors_are_clear():
    async def disconnected(request):raise httpx.ConnectError('private server details',request=request)
    for handler in [disconnected,lambda r:httpx.Response(200,text='<html>Error</html>'),
                    lambda r:httpx.Response(200,json=['wrong shape'])]:
        front=create_app(frontend_only=True,inference_url='https://models.example',
                         proxy_transport=httpx.MockTransport(handler))
        response=TestClient(front).get('/api/models')
        assert response.status_code==502
        assert 'private server details' not in response.text


def test_three_gradient_example_and_graph():
    client=TestClient(create_app())
    html=client.get('/experiment-2').text
    for text in ['three-gradient-example','(3/5, 6/5)','(1/5, 16/15)','(2/15, 14/15)','original peers']:
        assert text in html
    assert client.get('/static/figures/cifar/pcgrad_three_gradients.svg').status_code==200
