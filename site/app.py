"""Run the standalone four-page local website: python app.py."""
import argparse
import json
from pathlib import Path
from threading import Lock
from typing import Literal

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field, field_validator

ROOT = Path(__file__).resolve().parent
Method = Literal['token', 'sentiment', 'joint', 'pcgrad']


class PredictionRequest(BaseModel):
    text: str = Field(min_length=1, max_length=12000)
    methods: list[Method] = Field(default_factory=lambda: ['token', 'sentiment', 'joint', 'pcgrad'],
                                  min_length=1, max_length=4)

    @field_validator('text')
    @classmethod
    def meaningful_text(cls, value):
        if not value.strip():
            raise ValueError('Enter a sentence containing some text')
        return value

    @field_validator('methods')
    @classmethod
    def unique_methods(cls, value):
        return list(dict.fromkeys(value))


def create_app(store=None, device='cpu'):
    app = FastAPI(title='ESTR3114 · Model comparison', docs_url=None, redoc_url=None)
    app.mount('/static', StaticFiles(directory=ROOT/'static'), name='static')
    templates = Jinja2Templates(directory=ROOT/'templates')
    report = json.loads((ROOT/'experiment_data/text_results.json').read_text(encoding='utf-8'))
    app.state.store = store
    app.state.device = device
    store_lock = Lock()

    def model_store():
        with store_lock:
            if app.state.store is None:
                from inference.text import ModelStore
                app.state.store = ModelStore(ROOT/'models', device=app.state.device)
        return app.state.store

    def page(request, template, active, **context):
        return templates.TemplateResponse(request=request, name=template,
                                           context=dict(active=active, **context))

    @app.get('/', include_in_schema=False)
    def home():
        return RedirectResponse('/text', status_code=307)

    @app.get('/text')
    def text_page(request: Request):
        return page(request, 'text.html', 'text')

    @app.get('/images')
    def image_page(request: Request):
        return page(request, 'placeholder.html', 'images', title='Try image models',
                    description='Image prediction will be available after experiment 2 is complete.')

    @app.get('/experiment-1')
    def experiment_one(request: Request):
        return page(request, 'experiment.html', 'experiment-1', report=report)

    @app.get('/experiment-2')
    def experiment_two(request: Request):
        return page(request, 'placeholder.html', 'experiment-2', title='Experiment 2',
                    description='Settings and results will be added when this experiment is complete.')

    @app.get('/api/models')
    def models():
        return dict(models=model_store().status())

    @app.post('/api/predict-text')
    def predict(body: PredictionRequest):
        try:
            return model_store().predict(body.text, body.methods)
        except FileNotFoundError:
            return JSONResponse(status_code=503, content=dict(error='A selected checkpoint is missing. Check the local model installation.'))
        except Exception:
            # Keep paths and submitted text out of errors and logs.
            return JSONResponse(status_code=500, content=dict(error='Prediction failed. Check model compatibility and available memory, then restart the server.'))

    @app.get('/api/health')
    def health():
        return dict(status='ok')

    @app.middleware('http')
    async def headers(request, call_next):
        response = await call_next(request)
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['Referrer-Policy'] = 'same-origin'
        response.headers['Content-Security-Policy'] = "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; object-src 'none'; base-uri 'self'; frame-ancestors 'none'"
        if request.url.path.startswith('/api/'):
            response.headers['Cache-Control'] = 'no-store'
        return response

    return app


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', default=8000, type=int)
    parser.add_argument('--device', choices=('cpu', 'cuda', 'auto'), default='cpu')
    parser.add_argument('--threads', default=4, type=int, help='CPU threads per inference request')
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('--threads must be positive')
    import torch
    import uvicorn
    torch.set_num_threads(args.threads)
    uvicorn.run(create_app(device=args.device), host=args.host, port=args.port,
                access_log=False)
