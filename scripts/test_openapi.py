"""OpenAPI format, request bounds, route coverage, and optional read-only HTTP validation."""
import argparse
import json
import re
import unittest
from urllib.parse import urlsplit
from urllib.request import urlopen

from jsonschema import Draft202012Validator, ValidationError
from openapi_spec_validator import validate
from openapi_contract import ROOT, TARGET, document, ref, render

SPEC = document()


def check(value, schema):
    Draft202012Validator({'allOf': [schema], 'components': SPEC['components']}).validate(value)


class ContractTests(unittest.TestCase):
    def test_valid_openapi_and_reproducible_artifact(self):
        validate(SPEC)
        self.assertEqual(TARGET.read_text(encoding='utf-8'), render())

    def test_all_cpp_routes_documented(self):
        source = '\n'.join((ROOT / path).read_text(encoding='utf-8') for path in
                           ['src/service.cpp', 'src/service_video.cpp', 'src/image_queue_http.cpp'])
        actual = {(method.lower(), literal or raw) for method, literal, raw in
                  re.findall(r'http_\.(Get|Post)\((?:"([^"\n]+)"|R"\((.*?)\)")', source)}
        listed = {(method, operation['x-cpp-route']) for path in SPEC['paths'].values()
                  for method, operation in path.items()}
        self.assertGreaterEqual(len(actual), 16)
        self.assertEqual(actual, listed, 'New/changed HTTP route lacks matching OpenAPI operation')

    def test_distinct_operation_ids_and_body_limits(self):
        ids = []
        for path in SPEC['paths'].values():
            for method, operation in path.items():
                ids.append(operation['operationId'])
                if method == 'post':
                    self.assertIn('415', operation['responses'])
                    self.assertIn('413', operation['responses'])
        self.assertEqual(len(ids), len(set(ids)))
        source = (ROOT / 'src/service.cpp').read_text(encoding='utf-8')
        self.assertIn('set_payload_max_length(8192)', source)

    def test_client_job_requests(self):
        for request in [
            {'type': 'index_video', 'path': 'pedestrians.mp4'},
            {'type': 'segment_video', 'path': 'x.mp4', 'max_frames': 1000},
            *[{'type': kind, 'path': 'x.mp4', 'frame_index': 10000}
              for kind in ['segment_frame', 'ocr_frame', 'scene_frame']],
            {'type': 'search', 'query': 'person', 'limit': 20, 'scope': 'live'},
        ]:
            with self.subTest(request=request):
                check(request, ref('JobRequest'))
        for request in [
            {'type': 'index_live_archive', 'path': 'x.mp4'},
            {'type': 'search', 'query': '   \t'}, {'type': 'search', 'query': 'a' * 1025},
            {'type': 'search', 'query': 'person', 'scope': 'camera'},
            {'type': 'search', 'query': 'person', 'limit': 21},
            {'type': 'index_video', 'path': 'x.mp4', 'source_bytes': 1},
            {'type': 'index_video', 'path': 'x.mp4', 'stride': True},
            {'type': 'index_video', 'path': 'x.mp4', 'stride': 0},
            {'type': 'segment_video', 'path': 'x.mp4', 'max_frames': 1001},
            {'type': 'ocr_frame', 'path': 'x.mp4', 'frame_index': -1},
        ]:
            with self.subTest(request=request), self.assertRaises(ValidationError):
                check(request, ref('JobRequest'))

    def test_integer_bounds_match_cpp(self):
        source = (ROOT / 'src/service.cpp').read_text(encoding='utf-8')
        actual = {(key, int(default), int(low), int(high)) for key, default, low, high in
                  re.findall(r'integer\(request,\s*"(\w+)",\s*(\d+),\s*(\d+),\s*(\d+)\)', source)}
        expected = set()
        for name in ['IndexVideo', 'SegmentVideo', 'FrameAnalysis', 'Search']:
            for key, schema in SPEC['components']['schemas'][name]['properties'].items():
                if schema.get('type') == 'integer':
                    expected.add((key, schema['default'], schema['minimum'], schema['maximum']))
        self.assertEqual(actual, expected)

    def test_live_inputs_and_provenance(self):
        check({'source_id': 'local-replay', 'archive': False}, ref('LiveStart'))
        check({'session_id': 'live-123-1', 'segment_index': 8}, ref('ArchiveIndex'))
        for value, name in [({'source_id': 'test', 'url': 'rtsp://evil'}, 'LiveStart'),
                            ({'source_id': 'test', 'archive': 1}, 'LiveStart'),
                            ({'session_id': 'live-123-1', 'segment_index': 9}, 'ArchiveIndex'),
                            ({'unexpected': True}, 'Empty')]:
            with self.subTest(name=name), self.assertRaises(ValidationError):
                check(value, ref(name))
        self.assertEqual(SPEC['components']['schemas']['Sources']['properties']['source_kind']['enum'],
                         ['rtsp', 'file_replay'])
        self.assertNotIn('content', SPEC['paths']['/api/live/{id}/preview.jpg']['get']['responses']['204'])

    def test_normalized_request_is_not_client_input(self):
        normalized = {'type': 'index_video', 'path': 'x.mp4', 'stride': 15,
                      'max_frames': 0, 'source_bytes': 500, 'source_modified': '123'}
        with self.assertRaises(ValidationError):
            check(normalized, ref('JobRequest'))
        accepted = {'id': '100-1', 'state': 'queued', 'request': normalized, 'progress': {},
                    'created_ms': 100, 'updated_ms': 100, 'attempts': 0, 'recoveries': 0,
                    'cancel_requested': False, 'result': None, 'error': None}
        check(accepted, ref('Job'))
        accepted['state'] = 'succeeded'
        with self.assertRaises(ValidationError):
            check(accepted, ref('Job'))  # Success must carry the type-specific result, not null.


def verify_live(base):
    parts = urlsplit(base)
    if parts.scheme != 'http' or parts.hostname not in ('127.0.0.1', 'localhost') or \
            parts.username or parts.password or parts.path not in ('', '/') or parts.query or parts.fragment:
        raise ValueError('Only local HTTP service origins are accepted')
    base = base.rstrip('/')
    checked = 0
    for path in ['/api/health', '/api/media', '/api/jobs', '/api/live/sources', '/api/live', '/api/live/archive']:
        with urlopen(base + path, timeout=10) as response:
            value = json.load(response)
        check(value, SPEC['paths'][path]['get']['responses']['200']['content']['application/json']['schema'])
        checked += 1
        if path == '/api/jobs':
            result_types = {'index_video': 'IndexResult', 'search': 'SearchResult',
                'segment_frame': 'SegmentResult', 'ocr_frame': 'OcrResult',
                'scene_frame': 'SceneResult', 'segment_video': 'VideoResult'}
            for job in value['jobs']:
                if job['state'] == 'succeeded' and job['request']['type'] in result_types:
                    check(job['result'], ref(result_types[job['request']['type']]))
                    checked += 1
    with urlopen(base + '/openapi.json', timeout=10) as response:
        served = json.load(response)
    if served != SPEC:
        raise AssertionError('Running service does not serve the current OpenAPI artifact')
    print(f'PASS: {checked} real response/result schemas and served OpenAPI; GET only, no jobs or live sessions created.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', help='Optional existing local server, read-only checks')
    args = parser.parse_args()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ContractTests))
    if not result.wasSuccessful():
        raise SystemExit(1)
    if args.base_url:
        verify_live(args.base_url)
