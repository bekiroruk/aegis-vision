"""Real HTTP routes + standalone Redis + separate native YOLO process.

The lightweight HTTP fixture mounts exactly the same optional routes as LocalService;
it does not replace the full-service regression tests or claim GPU coverage.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import uuid
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def main():
    p = argparse.ArgumentParser()
    p.add_argument('fixture', type=Path)
    p.add_argument('worker', type=Path)
    p.add_argument('model', type=Path)
    p.add_argument('media', type=Path)
    p.add_argument('--port', type=int, default=16379)
    p.add_argument('--output', type=Path)
    p.add_argument('--browser', action='store_true', help='Also exercise the shipped panel in headless Chromium')
    args = p.parse_args()
    namespace = 'http-' + uuid.uuid4().hex
    # Test modified files only in our own temporary media root.
    with tempfile.TemporaryDirectory(prefix='aegis-http-image-') as tmp:
        media = Path(tmp)
        shutil.copyfile(args.media / 'bus.jpg', media / 'bus.jpg')
        process = subprocess.Popen([str(args.fixture.resolve()), str(args.model.resolve()), str(media),
                                    str(args.port), namespace, str(Path('web').resolve())],
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            line = process.stdout.readline().strip()
            if not line.isdigit():
                raise AssertionError('HTTP fixture startup failed: ' + process.stderr.read())
            base = 'http://127.0.0.1:' + line

            def request(path, body=None, expected=200, headers=None):
                h = {'Content-Type': 'application/json', **(headers or {})}
                req = Request(base + path, data=None if body is None else json.dumps(body).encode(), headers=h)
                try:
                    response = urlopen(req, timeout=15)
                except HTTPError as e:
                    response = e
                with response:
                    data = response.read()
                    assert response.status == expected, (path, response.status, data)
                    return json.loads(data) if 'application/json' in response.headers.get('Content-Type', '') else data

            status = request('/api/redis-images')
            assert status['enabled'] and status['images'] == ['bus.jpg']
            assert status['worker_status'] == 'no_recent_heartbeat' and status['worker_count'] == 0
            assert b'redis-images.js' in request('/redis.html')
            assert b'Foto' in request('/redis-images.js')
            request('/api/redis-images', headers={'Origin': 'http://evil.test'}, expected=403)
            request('/api/redis-images', {'id': 'x', 'image': 'bus.jpg'}, headers={'Origin': 'http://evil.test'}, expected=403)
            request('/api/redis-images', {'id': '../x', 'image': 'bus.jpg'}, expected=400)
            request('/api/redis-images', {'id': 'bad', 'image': '../bus.jpg'}, expected=400)
            request('/api/redis-images/missing', expected=404)
            submitted = request('/api/redis-images', {'id': 'real', 'image': 'bus.jpg'}, expected=202)
            assert submitted['state'] == 'queued' and 'token' not in submitted and 'owner' not in submitted
            again = request('/api/redis-images', {'id': 'real', 'image': 'bus.jpg'}, expected=202)
            assert again == submitted
            request('/api/redis-images/real/preview.jpg', expected=409)
            native = subprocess.run([str(args.worker.resolve()), 'run', str(args.port), namespace,
                                     'http-worker', str(args.model.resolve()), str(media), '--once', '3000'],
                                    capture_output=True, text=True, timeout=120)
            assert native.returncode == 0, native.stderr
            result = request('/api/redis-images/real')
            assert result['state'] == 'succeeded' and result['attempts'] == 1
            labels = [d['label'] for d in result['result']['detections']]
            assert labels.count('person') >= 3 and 'bus' in labels
            preview = request('/api/redis-images/real/preview.jpg')
            assert preview[:2] == b'\xff\xd8' and len(preview) <= 1024 * 1024
            request('/api/redis-images', {'id': 'cancel', 'image': 'bus.jpg'}, expected=202)
            assert request('/api/redis-images/cancel/cancel', {})['state'] == 'cancelled'
            # Do not render old boxes on a different image; never silently overwrite an existing ID.
            with (media / 'bus.jpg').open('ab') as f:
                f.write(b'changed')
            request('/api/redis-images/real/preview.jpg', expected=409)
            request('/api/redis-images', {'id': 'real', 'image': 'bus.jpg'}, expected=409)
            evidence = {'passed': True, 'http_redis_yolo_verified': True, 'labels': labels,
                        'preview_bytes': len(preview), 'job': result}
            # A persistent idle worker keeps its own liveness window refreshed.
            # Then prove actual inference and expiry after that process exits.
            shutil.copyfile(args.media / 'bus.jpg', media / 'bus.jpg')
            worker = subprocess.Popen([str(args.worker.resolve()), 'run', str(args.port), namespace,
                                       'presence-worker', str(args.model.resolve()), str(media)],
                                      stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            def wait_for(check, seconds=15):
                deadline = time.monotonic() + seconds
                while time.monotonic() < deadline:
                    if check(): return
                    time.sleep(.1)
                raise AssertionError('Timed out waiting for worker presence/result')
            try:
                time.sleep(5.5)  # Earlier --once heartbeat must expire; idle worker must remain.
                assert worker.poll() is None
                presence = request('/api/redis-images')
                assert presence['worker_count'] == 1 and presence['worker_status'] == 'recent_heartbeat'
                assert presence['heartbeat_ttl_ms'] == 5000
                request('/api/redis-images', {'id': 'presence-job', 'image': 'bus.jpg'}, expected=202)
                wait_for(lambda: request('/api/redis-images/presence-job')['state'] == 'succeeded', 120)
            finally:
                worker.terminate()
                try: worker.communicate(timeout=5)
                except subprocess.TimeoutExpired:
                    worker.kill()
                    worker.communicate(timeout=5)
            wait_for(lambda: request('/api/redis-images')['worker_count'] == 0)
            assert request('/api/redis-images')['worker_status'] == 'no_recent_heartbeat'
            evidence['worker_presence_verified'] = True
            if args.browser:
                from playwright.sync_api import sync_playwright, expect
                shutil.copyfile(args.media / 'bus.jpg', media / 'bus.jpg')
                with sync_playwright() as playwright:
                    browser = playwright.chromium.launch()
                    try:
                        page = browser.new_page(viewport={'width': 1280, 'height': 1000})
                        errors = []
                        page.on('pageerror', lambda error: errors.append(str(error)))
                        page.goto(base + '/redis.html')
                        expect(page.locator('#ri-connection')).to_contain_text('İşçi sinyali yok')
                        expect(page.locator('#ri-submit')).to_be_enabled()
                        page.locator('#ri-submit').click()
                        expect(page.locator('#ri-status')).to_contain_text('Sırada')
                        saved = page.evaluate("JSON.parse(localStorage.getItem('aegis-redis-image-v1'))")
                        page.reload()
                        expect(page.locator('#ri-status')).to_contain_text(saved['id'])
                        expect(page.locator('#ri-submit')).to_be_disabled()
                        native = subprocess.run([str(args.worker.resolve()), 'run', str(args.port), namespace,
                                                 'browser-worker', str(args.model.resolve()), str(media), '--once'],
                                                capture_output=True, text=True, timeout=120)
                        assert native.returncode == 0, native.stderr
                        expect(page.locator('#ri-preview')).to_be_visible(timeout=30000)
                        expect(page.locator('#ri-results li')).to_have_count(len(labels))
                        assert request('/api/redis-images/' + saved['id'])['state'] == 'succeeded'
                        if args.output:
                            args.output.parent.mkdir(parents=True, exist_ok=True)
                            page.screenshot(path=str(args.output.with_suffix('.png')), full_page=True)
                        page.locator('#ri-new').click()
                        page.locator('#ri-submit').click()
                        expect(page.locator('#ri-status')).to_contain_text('Sırada')
                        page.locator('#ri-cancel').click()
                        expect(page.locator('#ri-status')).to_contain_text('İptal edildi')
                        assert not errors, errors
                        evidence['browser_verified'] = True
                    finally:
                        browser.close()
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(evidence, indent=2), encoding='utf-8')
                args.output.with_suffix('.jpg').write_bytes(preview)
            print(json.dumps({key: value for key, value in evidence.items() if key != 'job'}))
        finally:
            process.terminate()
            try:
                process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate(timeout=5)


if __name__ == '__main__':
    main()
