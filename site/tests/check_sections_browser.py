"""Verify mobile report toggles and unchanged persistent desktop navigation."""
import os
from pathlib import Path
import shutil
import sys
from urllib.parse import urlsplit

os.environ['SITE_MODE'] = 'frontend'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from fastapi.testclient import TestClient
from playwright.sync_api import sync_playwright
from app import create_app

client = TestClient(create_app(frontend_only=True))
out = ROOT/'.qa/sections'
out.mkdir(parents=True,exist_ok=True)
errors = []
with sync_playwright() as p:
    executable = shutil.which('chromium') or shutil.which('google-chrome')
    windows_chrome = Path(r'C:\Program Files\Google\Chrome\Application\chrome.exe')
    if not executable and windows_chrome.exists(): executable = str(windows_chrome)
    browser = p.chromium.launch(executable_path=executable,headless=True)
    context = browser.new_context(viewport=dict(width=390,height=844))
    def handle(route):
        url = urlsplit(route.request.url)
        response = client.get(url.path)
        headers = {k:v for k,v in response.headers.items() if k.lower() not in ('content-length','content-encoding')}
        route.fulfill(status=response.status_code,headers=headers,body=response.content)
    context.route('http://sections.test/**',handle)
    page = context.new_page()
    page.on('pageerror',lambda error:errors.append(str(error)))
    for report in ('experiment-1','experiment-2'):
        for width,height in ((390,844),(768,1024)):
            page.set_viewport_size(dict(width=width,height=height))
            page.goto('http://sections.test/'+report)
            toggle = page.locator('#report-sections-toggle')
            sidebar = page.locator('#report-sidebar')
            assert toggle.is_visible() and not sidebar.is_visible()
            assert toggle.get_attribute('aria-expanded') == 'false'
            assert toggle.inner_text() == 'Show sections'
            box = toggle.bounding_box()
            assert box['x'] < 30 and box['y'] < 20
            assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
            page.screenshot(path=str(out/f'{report}-{width}-closed.png'))
            toggle.click()
            assert sidebar.is_visible() and toggle.get_attribute('aria-expanded') == 'true'
            page.screenshot(path=str(out/f'{report}-{width}-open.png'))
            sidebar.locator('a[href="#settings"]').click()
            assert not sidebar.is_visible()
            assert page.url.endswith('#settings')
            toggle.click()
            page.keyboard.press('Escape')
            assert not sidebar.is_visible() and toggle.get_attribute('aria-expanded') == 'false'
            toggle.click()
            page.mouse.click(width-5,height-5)
            assert not sidebar.is_visible()
            page.locator('header .brand').click()
            assert not page.locator('#report-sections-toggle').count()
            page.goto('http://sections.test/'+report)
            page.get_by_role('button',name='Show sections',exact=True).click()
            page.set_viewport_size(dict(width=1440,height=1000))
            assert page.locator('#report-sidebar').is_visible()
            assert not page.locator('#report-sections-toggle').is_visible()
            assert page.locator('#report-sidebar').bounding_box()['x'] == 0
            page.screenshot(path=str(out/f'{report}-desktop.png'))
            page.set_viewport_size(dict(width=390,height=844))
            assert not page.locator('#report-sidebar').is_visible()
            assert page.get_by_role('button',name='Show sections',exact=True).is_visible()
    assert not errors,errors
    browser.close()
print('Passed: both reports on mobile/tablet, toggles, anchors, Escape, resizing and persistent desktop sidebars.')
