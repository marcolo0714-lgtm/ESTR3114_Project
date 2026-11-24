"""Check the QR entrance popup, session dismissal and destination links."""
from pathlib import Path
import shutil
import sys
from urllib.parse import urlsplit

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from fastapi.testclient import TestClient
from playwright.sync_api import sync_playwright
import app as site_app

demo_url='https://presentation-demo.trycloudflare.com'
site_app.load_demo_url=lambda root:demo_url
client=TestClient(site_app.create_app(frontend_only=True,inference_url=''))
out=ROOT/'.qa/demo-guide'
out.mkdir(parents=True,exist_ok=True)
errors=[]

with sync_playwright() as p:
    executable=shutil.which('chromium') or shutil.which('google-chrome') or r'C:\Program Files\Google\Chrome\Application\chrome.exe'
    browser=p.chromium.launch(executable_path=executable,headless=True)
    for width,height in ((390,844),(1440,1000)):
        context=browser.new_context(viewport=dict(width=width,height=height))
        def handle(route):
            response=client.get(urlsplit(route.request.url).path)
            headers={k:v for k,v in response.headers.items() if k.lower() not in ('content-length','content-encoding')}
            route.fulfill(status=response.status_code,headers=headers,body=response.content)
        context.route('http://demo.test/**',handle)
        page=context.new_page()
        page.on('pageerror',lambda error:errors.append(str(error)))
        page.goto('http://demo.test/text')
        dialog=page.locator('#live-demo-dialog')
        assert dialog.is_visible()
        assert dialog.locator('a.primary').get_attribute('href')==demo_url+'/text'
        assert not page.locator('#prediction-form').count()
        assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
        box=dialog.bounding_box()
        assert box['x']>=0 and box['y']>=0 and box['width']<=width and box['height']<=height
        page.screenshot(path=str(out/f'popup-{width}.png'))
        page.keyboard.press('Escape')
        assert not dialog.is_visible()
        page.reload()
        assert not dialog.is_visible()
        page.get_by_role('link',name='Try image models',exact=True).first.click()
        assert not dialog.is_visible()
        assert not page.locator('#image-form').count()
        assert page.locator('.demo-guide a.primary').get_attribute('href')==demo_url+'/images'
        page.screenshot(path=str(out/f'image-guide-{width}.png'))
        page.get_by_role('link',name='Experiment 2',exact=True).first.click()
        assert not dialog.is_visible()
        assert page.locator('#test-results').count()==1
        context.close()
    # The same tab sees a fresh announcement after tomorrow's URL changes.
    context=browser.new_context()
    context.route('http://demo.test/**',handle)
    page=context.new_page()
    page.goto('http://demo.test/images')
    page.get_by_role('button',name='Stay here and read the reports').click()
    demo_url='https://updated-demo.trycloudflare.com'
    client=TestClient(site_app.create_app(frontend_only=True,inference_url=''))
    page.reload()
    assert page.locator('#live-demo-dialog').is_visible()
    assert page.locator('#live-demo-dialog a.primary').get_attribute('href')==demo_url+'/images'
    # An unpublished URL has guidance but no outgoing demo buttons.
    demo_url=''
    client=TestClient(site_app.create_app(frontend_only=True,inference_url=''))
    page.reload()
    assert page.locator('#live-demo-dialog').is_visible()
    assert not page.locator('#live-demo-dialog a').count()
    assert 'not been published yet' in page.locator('#live-demo-dialog').inner_text()
    assert not errors,errors
    browser.close()
print('Passed: phone/desktop popup, Escape, session persistence, text/image links, URL changes and pending state.')
