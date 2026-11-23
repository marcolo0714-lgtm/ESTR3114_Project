"""Lightweight same-origin proxy for Vercel; imports no model dependencies."""
import json
import os
from pathlib import Path
from urllib.parse import urlsplit
import httpx
from fastapi.responses import JSONResponse, Response

API_PATHS = frozenset(('/api/models','/api/image-models','/api/predict-text','/api/predict-image'))
FRONTEND_IMAGE_LIMIT = 4 * 1024 * 1024
LOCAL_IMAGE_LIMIT = 8 * 1024 * 1024


def frontend_mode():
    return os.environ.get('SITE_MODE', 'frontend' if os.environ.get('VERCEL') else 'local') == 'frontend'


class InferenceProxy:
    def __init__(self, root, url='', token='', transport=None):
        self.root = Path(root)
        self.url = url.rstrip('/')
        self.token = token
        self.transport = transport
        if self.url:
            parsed = urlsplit(self.url)
            if parsed.scheme not in ('http','https') or not parsed.netloc or parsed.username or parsed.password or parsed.query or parsed.fragment:
                raise ValueError('INFERENCE_API_URL must be an HTTP(S) base URL without credentials or query parameters')

    def not_configured(self, path):
        message = 'Predictions are not available yet. The site owner needs to connect a model server.'
        if path == '/api/models':
            registry = json.loads((self.root/'models/registry.json').read_text(encoding='utf-8'))
            models = [{k:e[k] for k in ('id','name','seed','steps','weight','sentiment_trained')}
                      | dict(available=False,loaded=False) for e in registry['models']]
            return JSONResponse(dict(models=models,message=message,location='remote'))
        if path == '/api/image-models':
            registry = json.loads((self.root/'models/images/registry.json').read_text(encoding='utf-8'))
            models = [dict(id=e['id'],name=e['name'],seed=e['seed'],training_steps=e['training_steps'],
                           checkpoints={k:dict(step=c['step'],available=False,loaded=False)
                                        for k,c in e['checkpoints'].items()}) for e in registry['models']]
            return JSONResponse(dict(models=models,tasks=registry['tasks'],message=message,
                                     location='remote',max_image_bytes=FRONTEND_IMAGE_LIMIT))
        return JSONResponse(status_code=503,content=dict(error=message))

    async def forward(self, request):
        if not self.url:
            return self.not_configured(request.url.path)
        if request.method not in ('GET','POST'):
            return JSONResponse(status_code=405,content=dict(error='Method not allowed'))
        if urlsplit(self.url).netloc == request.url.netloc:
            return JSONResponse(status_code=503,content=dict(error='The inference URL must point to the model server, not this website.'))
        body = bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body) > FRONTEND_IMAGE_LIMIT:
                return JSONResponse(status_code=413,content=dict(error='Choose an image no larger than 4 MB for the deployed website.'))
        headers = {'Content-Type':request.headers.get('content-type','application/json')}
        if self.token: headers['Authorization'] = 'Bearer '+self.token
        target = self.url+request.url.path
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(180,connect=10),
                                         follow_redirects=False,transport=self.transport) as client:
                response = await client.request(request.method,target,params=request.query_params.multi_items(),
                                                content=bytes(body),headers=headers)
            if 'application/json' not in response.headers.get('content-type',''):
                return JSONResponse(status_code=502,content=dict(error='The model server did not return a valid API response. Check its URL and service status.'))
            payload = response.json()
            if not isinstance(payload,dict):
                return JSONResponse(status_code=502,content=dict(error='The model server returned an unexpected API response.'))
            if response.is_success and request.url.path in ('/api/models','/api/image-models'):
                payload['location'] = 'remote'
                if request.url.path == '/api/image-models': payload['max_image_bytes'] = FRONTEND_IMAGE_LIMIT
            return JSONResponse(status_code=response.status_code,content=payload)
        except (httpx.RequestError,ValueError):
            return JSONResponse(status_code=502,content=dict(error='Cannot reach the model server. Check its deployment and INFERENCE_API_URL.'))
