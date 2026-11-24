"""Public QR entrance guidance and local demo separation."""
import json
from pathlib import Path
import sys

from fastapi.testclient import TestClient
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import app as site_app
from deployment import load_demo_url


def test_demo_link_configuration(tmp_path):
    config = tmp_path/'demo_config.json'
    for value,expected in [('', ''), (' https://example.trycloudflare.com/ ', 'https://example.trycloudflare.com')]:
        config.write_text(json.dumps({'url':value}),encoding='utf-8-sig')
        assert load_demo_url(tmp_path) == expected
    for value in ('javascript:alert(1)', 'http://example.com', 'https://example.com/text',
                  'https://user:secret@example.com', 'https://example.com?q=1', 'https://example.com/#fragment'):
        config.write_text(json.dumps({'url':value}),encoding='utf-8')
        with pytest.raises(ValueError): load_demo_url(tmp_path)


def test_frontend_guide_links_and_local_forms(monkeypatch):
    url = 'https://presentation-demo.trycloudflare.com'
    monkeypatch.setattr(site_app, 'load_demo_url', lambda root: url)
    front = TestClient(site_app.create_app(frontend_only=True,inference_url=''))
    for route,form,script in [('text','prediction-form','app.js'),('images','image-form','images.js')]:
        html=front.get('/'+route).text
        assert f'href="{url}/{route}"' in html
        assert 'id="live-demo-dialog"' in html
        assert f'id="{form}"' not in html
        assert f'src="/static/{script}"' not in html
    for route in ('experiment-1','experiment-2'):
        html=front.get('/'+route).text
        assert 'id="live-demo-dialog"' in html
        assert 'id="report-sidebar"' in html
    local = TestClient(site_app.create_app(frontend_only=False))
    for route,form in [('text','prediction-form'),('images','image-form')]:
        html=local.get('/'+route).text
        assert f'id="{form}"' in html
        assert 'id="live-demo-dialog"' not in html
        assert url not in html


def test_pending_link_has_no_fake_demo_destination(monkeypatch):
    monkeypatch.setattr(site_app,'load_demo_url',lambda root:'')
    client=TestClient(site_app.create_app(frontend_only=True,inference_url=''))
    html=client.get('/text').text
    assert 'live demo link has not been published yet' in html
    assert 'href="https://' not in html
    assert 'id="prediction-form"' not in html
