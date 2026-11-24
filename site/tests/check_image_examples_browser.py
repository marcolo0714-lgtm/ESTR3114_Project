"""Verify one-click example selection and real prediction input on phone/desktop."""
import json
from pathlib import Path
import sys
from urllib.parse import urlsplit,parse_qs

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from fastapi.testclient import TestClient
from playwright.sync_api import sync_playwright
from app import create_app

torch.set_num_threads(4)
client=TestClient(create_app(frontend_only=False))
examples=json.loads((ROOT/'experiment_data/image_examples.json').read_text())
errors=[]
submitted=[]
fail_example=False
out=ROOT/'.qa/image-examples'
out.mkdir(parents=True,exist_ok=True)
with sync_playwright() as p:
    browser=p.chromium.launch(executable_path=r'C:\Program Files\Google\Chrome\Application\chrome.exe',headless=True)
    context=browser.new_context(viewport=dict(width=1440,height=1000))
    def route_request(route):
        request=route.request
        url=urlsplit(request.url)
        if fail_example and url.path.startswith('/static/examples/'):
            route.fulfill(status=503,body='Unavailable');return
        if url.path=='/api/predict-image':
            submitted.append((parse_qs(url.query),request.post_data_buffer))
        response=client.request(request.method,url.path+('?' + url.query if url.query else ''),
            content=request.post_data_buffer,headers={'Content-Type':request.headers.get('content-type','application/json')})
        headers={k:v for k,v in response.headers.items() if k.lower() not in ('content-length','content-encoding')}
        route.fulfill(status=response.status_code,headers=headers,body=response.content)
    context.route('http://examples.test/**',route_request)
    page=context.new_page()
    page.on('pageerror',lambda error:errors.append(str(error)))
    page.goto('http://examples.test/images')
    page.get_by_text('2 of 2 image models available locally').wait_for()
    assert page.locator('[data-image-example]').count()==3
    for example in examples:
        button=page.locator(f'[data-task="{example["task"]}"]')
        button.click()
        page.get_by_text(f'{example["label"]} example ready. Superclass selected automatically.',exact=True).wait_for()
        assert page.locator('#image-task').input_value()==str(example['task'])
        assert button.get_attribute('aria-pressed')=='true'
        assert page.locator('[data-image-example][aria-pressed="true"]').count()==1
        page.locator('#upload-preview').wait_for(state='visible')
        page.get_by_role('button',name='Predict',exact=True).click()
        page.locator('.image-result-card').nth(1).wait_for()
        query,body=submitted[-1]
        assert query['task']==[str(example['task'])]
        assert body==(ROOT/example['url'].lstrip('/')).read_bytes()
        assert not page.locator('#image-error').is_visible()
    page.screenshot(path=str(out/'desktop.png'),full_page=True)
    page.get_by_role('button',name='Clear',exact=True).click()
    assert not page.locator('[aria-pressed="true"]').count()
    assert page.locator('#image-task').input_value()==''
    assert not page.locator('#upload-preview').is_visible()
    page.set_viewport_size(dict(width=390,height=844))
    page.locator('[data-task="2"]').click()
    page.get_by_text('sunflower example ready. Superclass selected automatically.',exact=True).wait_for()
    assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
    page.screenshot(path=str(out/'phone.png'),full_page=True)
    # Upload remains usable after a template and resets its selected state.
    page.locator('#image-file').set_input_files(str(ROOT/'static/examples/aquarium_fish.png'))
    assert not page.locator('[aria-pressed="true"]').count()
    fail_example=True
    page.locator('[data-task="18"]').click()
    page.get_by_text('The example image could not be loaded.',exact=False).wait_for()
    assert page.locator('#image-predict').is_enabled()
    assert not page.locator('#upload-preview').is_visible()
    assert not errors,errors
    browser.close()
print('Passed: all 3 examples select correct tasks and bytes for real CNN predictions; clear, upload, mobile and fetch failure work.')
