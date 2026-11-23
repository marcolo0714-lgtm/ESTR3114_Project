"""Manual browser QA using real CNNs through local ASGI request routing.

Run from site: .venv/Scripts/python.exe tests/check_image_browser.py
Playwright/Chrome are needed only for this development check, not the website.
"""
from io import BytesIO
from pathlib import Path
import sys
from urllib.parse import urlsplit
import json
import base64

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from PIL import Image
from fastapi.testclient import TestClient
from playwright.sync_api import sync_playwright
from app import create_app

torch.set_num_threads(4)
client=TestClient(create_app())
out=ROOT/'.qa/experiment2'
out.mkdir(parents=True,exist_ok=True)
buffer=BytesIO()
image=Image.new('RGB',(180,100))
image.putdata([(x%256,y%256,(x+y)%256) for y in range(100) for x in range(180)])
image.save(buffer,format='PNG')
upload=dict(name='example.png',mimeType='image/png',buffer=buffer.getvalue())
errors=[]
with sync_playwright() as p:
    browser=p.chromium.launch(executable_path=r'C:\Program Files\Google\Chrome\Application\chrome.exe',headless=True)
    context=browser.new_context(viewport=dict(width=1440,height=1000),device_scale_factor=1)
    def route_request(route):
        req=route.request
        url=urlsplit(req.url)
        if url.netloc!='local.test':
            errors.append('Unexpected external request: '+url.netloc);route.abort();return
        path=url.path+('?' + url.query if url.query else '')
        response=client.request(req.method,path,content=req.post_data_buffer,
                                headers={'Content-Type':req.headers.get('content-type','application/json')})
        headers={k:v for k,v in response.headers.items() if k.lower() not in ('content-length','content-encoding')}
        route.fulfill(status=response.status_code,headers=headers,body=response.content)
    context.route('**/*',route_request)
    page=context.new_page()
    page.on('pageerror',lambda error:errors.append(str(error)))
    page.goto('http://local.test/images')
    page.get_by_text('2 of 2 image models available locally').wait_for(timeout=30000)
    page.locator('#image-file').set_input_files(upload)
    page.locator('#image-task').select_option('1')
    assert 'shark' in page.locator('#task-classes').inner_text()
    page.get_by_role('button',name='Predict',exact=True).click()
    page.locator('.image-result-card').nth(1).wait_for(timeout=30000)
    assert page.locator('.image-result-card tbody tr').count()==10
    assert not page.locator('#image-error').is_visible()
    assert '9,500' in page.locator('.image-result-card').first.inner_text()
    assert '9,200' in page.locator('.image-result-card').nth(1).inner_text()
    assert page.locator('#model-preview').evaluate('(img)=>img.naturalWidth')==32
    assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
    page.screenshot(path=str(out/'images-desktop.png'),full_page=True)
    page.locator('#image-checkpoint').select_option('final')
    page.get_by_role('button',name='Predict',exact=True).click()
    page.locator('.image-result-card').nth(1).wait_for(timeout=30000)
    assert all('10,000' in x.inner_text() for x in page.locator('.image-result-card').all())
    page.get_by_role('button',name='Clear',exact=True).click()
    assert page.locator('.image-result-card').count()==0
    assert page.locator('#image-task').input_value()==''
    assert not page.locator('#upload-preview').is_visible()
    # Supply browser clipboard File objects through a paste event, keeping the
    # user's operating-system clipboard untouched during this automated check.
    assert not page.locator('#image-file').evaluate('(input)=>input.required')
    page.locator('#image-paste-area').focus()
    pasted = page.evaluate('''(base64) => {
        const bytes=Uint8Array.from(atob(base64), c=>c.charCodeAt(0));
        const clipboard=new DataTransfer();
        clipboard.items.add(new File([bytes], 'pasted.png', {type:'image/png'}));
        const event=new ClipboardEvent('paste', {clipboardData:clipboard, bubbles:true, cancelable:true});
        document.activeElement.dispatchEvent(event);
        return event.defaultPrevented;
    }''',base64.b64encode(buffer.getvalue()).decode('ascii'))
    assert pasted
    page.get_by_text('Pasted image ready',exact=False).wait_for()
    page.locator('#upload-preview').wait_for(state='visible')
    page.locator('#image-task').select_option('1')
    page.get_by_role('button',name='Predict',exact=True).click()
    page.locator('.image-result-card').nth(1).wait_for(timeout=30000)
    assert page.locator('.image-result-card tbody tr').count()==10
    assert not page.locator('#image-error').is_visible()
    # Text paste is not intercepted, and cannot silently replace the image.
    assert not page.evaluate('''() => {
        const clipboard=new DataTransfer(); clipboard.setData('text/plain','ordinary text');
        const event=new ClipboardEvent('paste',{clipboardData:clipboard,bubbles:true,cancelable:true});
        document.dispatchEvent(event); return event.defaultPrevented;
    }''')
    assert 'Pasted image ready' in page.locator('#image-source').inner_text()
    page.screenshot(path=str(out/'images-pasted.png'),full_page=True)
    page.get_by_role('button',name='Clear',exact=True).click()
    assert page.locator('#image-source').inner_text()=='No image selected.'
    assert not page.locator('#upload-preview').is_visible()
    assert page.locator('#image-file').input_value()==''
    for width,height,label in [(1440,1000,'desktop'),(390,844,'mobile')]:
        page.set_viewport_size(dict(width=width,height=height))
        page.goto('http://local.test/experiment-2')
        assert page.get_by_role('navigation',name='Main navigation').locator('[aria-current]').count()==1
        assert page.locator('.report-sidebar').is_visible()
        assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
        page.screenshot(path=str(out/f'report-{label}.png'))
        page.locator('#validation-results').scroll_into_view_if_needed()
        page.screenshot(path=str(out/f'results-{label}.png'))
        page.locator('#errors details').first.locator('summary').click()
        assert page.locator('#errors details').first.locator('table').count()==4
        assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
        page.goto('http://local.test/images')
        page.get_by_text('2 of 2 image models available locally').wait_for()
        page.locator('#image-file').set_input_files(upload)
        page.locator('#image-task').select_option('14')
        page.get_by_role('button',name='Predict',exact=True).click()
        page.locator('.image-result-card').nth(1).wait_for()
        assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
        page.screenshot(path=str(out/f'images-{label}.png'),full_page=True)
    assert not errors,errors
    browser.close()
(out/'verification.json').write_text(json.dumps(dict(real_checkpoints=True,desktop=True,mobile=True,
    image_paste=True,text_paste_preserved=True,clear_pasted_image=True,
    two_checkpoint_modes=True,confusion_tables=True,javascript_errors=errors),indent=2))
print('Browser QA passed: image paste, clear, file selection, real predictions and desktop/mobile pages.')
