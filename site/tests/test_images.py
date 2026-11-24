"""Actual bundled CNN inference, preprocessing parity, and image API contracts."""
from io import BytesIO
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest
import torch
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app import create_app
from inference.images import ImageModelStore, decode_image, MAX_IMAGE_BYTES
from inference.image_model import ModelConfig, MultitaskCNN


@pytest.fixture(scope='module')
def client():
    torch.set_num_threads(4)
    return TestClient(create_app())


@pytest.fixture
def image_bytes():
    # Deliberately non-square and non-uniform to expose normalization/resize errors.
    pixels = np.arange(19*31*3, dtype=np.uint8).reshape(19,31,3)
    image = Image.fromarray(pixels)
    buffer = BytesIO(); image.save(buffer,format='PNG')
    return buffer.getvalue()


def test_image_metadata_and_report_are_complete(client):
    metadata = client.get('/api/image-models').json()
    assert len(metadata['tasks']) == 20
    assert all(len(t['class_names']) == 5 for t in metadata['tasks'])
    assert [m['id'] for m in metadata['models']] == ['joint','pcgrad']
    for m in metadata['models']:
        assert all(v['available'] for v in m['checkpoints'].values())
    html = client.get('/experiment-2').text
    for text in ('67.12','69.12','67.34','69.38','9,500','9,200',
                 'one seed','Held-out test performance','67.07','68.80','18 tasks'):
        assert text in html
    assert 'In progress' not in html
    assert 'not a measured training' not in html
    assert 'test set has not been evaluated' not in html
    report = json.loads((ROOT/'experiment_data/image_results.json').read_text())
    assert len(report['task_rows']) == len(report['confusions']) == 20
    for figure in report['figures']:
        assert client.get('/static/figures/cifar/'+figure['file']).status_code == 200


@pytest.mark.parametrize('kind', ['best_validation','final'])
def test_real_predictions_match_independent_preprocessing(client,image_bytes,kind):
    response = client.post('/api/predict-image',params=dict(task=1,checkpoint=kind),content=image_bytes)
    assert response.status_code == 200, response.text
    result = response.json()
    assert result['task']['name'] == 'fish'
    assert result['image']['width'] == 31 and result['image']['height'] == 19
    assert result['image']['preview'].startswith('data:image/png;base64,')
    with Image.open(BytesIO(image_bytes)) as source:
        pixels=np.asarray(source.convert('RGB').resize((32,32),Image.Resampling.BILINEAR)).copy()
    for row in result['results']:
        saved=torch.load(ROOT/f'models/images/{row["id"]}/{kind}.pt',map_location='cpu',weights_only=True)
        model=MultitaskCNN(ModelConfig(**saved['model_config']))
        model.load_state_dict(saved['model'],strict=True);model.eval()
        x=torch.from_numpy(pixels).permute(2,0,1).float()[None]/255
        n=saved['normalization']
        x=(x-torch.tensor(n['mean'])[None,:,None,None])/torch.tensor(n['std'])[None,:,None,None]
        with torch.inference_mode(): expected=model(x,1).softmax(-1)[0].tolist()
        actual={v['label']:v['probability'] for v in row['probabilities']}
        assert list(actual)==[v['label'] for v in sorted(row['probabilities'],key=lambda v:-v['probability'])]
        assert sum(actual.values()) == pytest.approx(1,abs=1e-6)
        for label,value in zip(saved['mapping']['tasks'][1]['class_names'],expected):
            assert actual[label] == pytest.approx(value,abs=1e-7)
        assert row['predicted_class']==max(actual,key=actual.get)
        assert row['step']==saved['step']


def test_method_selection_cache_and_all_task_heads(client,image_bytes):
    for task in range(20):
        response=client.post('/api/predict-image',params=[('task',task),('methods','joint')],content=image_bytes)
        assert response.status_code==200
        row=response.json()['results'][0]
        assert row['id']=='joint' and len(row['probabilities'])==5
        assert {x['label'] for x in row['probabilities']}==set(response.json()['task']['class_names'])
    warm=client.post('/api/predict-image?task=1&methods=joint&methods=joint',content=image_bytes).json()
    assert len(warm['results'])==1 and not warm['results'][0]['first_load']
    assert client.get('/api/image-models').json()['models'][0]['checkpoints']['best_validation']['loaded']


@pytest.mark.parametrize('query', ['', '?task=-1','?task=20','?task=fish','?task=1&methods=token',
                                  '?task=1&checkpoint=unknown'])
def test_invalid_image_options(client,image_bytes,query):
    assert client.post('/api/predict-image'+query,content=image_bytes).status_code==422


@pytest.mark.parametrize('content', [b'',b'not an image',b'<svg><script>alert(1)</script></svg>'])
def test_invalid_image_bytes(client,content):
    response=client.post('/api/predict-image?task=1',content=content)
    assert response.status_code==400
    assert 'error' in response.json()


def test_upload_size_pixel_bounds_and_private_files(client,image_bytes,monkeypatch):
    response=client.post('/api/predict-image?task=1',content=b'x'*(MAX_IMAGE_BYTES+1))
    assert response.status_code==413
    import inference.images as images
    monkeypatch.setattr(images,'MAX_IMAGE_PIXELS',4)
    assert client.post('/api/predict-image?task=1',content=image_bytes).status_code==400
    assert client.get('/models/images/joint/best_validation.pt').status_code==404
    assert client.get('/api/image-models').headers['cache-control']=='no-store'


def test_image_paths_missing_and_corrupt_checkpoint(client,image_bytes,tmp_path):
    store=ImageModelStore(ROOT/'models/images')
    with pytest.raises(ValueError):store.local_path('../../app.py')
    registry=json.loads((ROOT/'models/images/registry.json').read_text())
    (tmp_path/'registry.json').write_text(json.dumps(registry))
    missing=TestClient(create_app(image_store=ImageModelStore(tmp_path)))
    response=missing.post('/api/predict-image?task=1&methods=joint',content=image_bytes)
    assert response.status_code==503 and str(tmp_path) not in response.text
    path=tmp_path/'joint/best_validation.pt';path.parent.mkdir();path.write_bytes(b'corrupted')
    assert missing.post('/api/predict-image?task=1&methods=joint',content=image_bytes).status_code==500


def test_site_is_independent_of_current_working_directory(client,image_bytes,monkeypatch,tmp_path):
    monkeypatch.chdir(tmp_path)
    assert client.get('/experiment-2').status_code==200
    assert client.post('/api/predict-image?task=0&methods=pcgrad',content=image_bytes).status_code==200
