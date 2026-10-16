"""Source for the checked-in OpenAPI document; no runtime dependency in C++.

Run --write after intentional edits, --check in CI. Generated JSON is served
by the existing static web mount; it never executes requests or loads a CDN.
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / 'web/openapi.json'


def ref(name):
    return {'$ref': '#/components/schemas/' + name}


def obj(properties, required=None, closed=False):
    return {'type': 'object', 'properties': properties,
            'required': list(properties) if required is None else required,
            'additionalProperties': not closed}


def arr(items):
    return {'type': 'array', 'items': items}


def integer(low=0, high=None, default=None):
    result = {'type': 'integer', 'minimum': low}
    if high is not None:
        result['maximum'] = high
    if default is not None:
        result['default'] = default
    return result


TEXT = {'type': 'string'}
BOOL = {'type': 'boolean'}
NUMBER = {'type': 'number'}
NULL_TEXT = {'type': ['string', 'null']}
JOB_ID = {'type': 'string', 'pattern': '^[0-9]+-[0-9]+$'}
LIVE_ID = {'type': 'string', 'pattern': '^live-[0-9]+-[0-9]+$'}
JOB_STATES = ['queued', 'running', 'succeeded', 'failed', 'cancelled']


def document():
    schemas = {
        'Error': obj({'error': TEXT}),
        'Backend': obj({'engine': TEXT, 'device': TEXT, 'precision': TEXT,
                        'gpu': TEXT, 'engine_sha256': TEXT}, ['engine', 'device', 'precision']),
        'Empty': obj({}, closed=True),
        'IndexVideo': obj({'type': {'const': 'index_video'}, 'path': TEXT,
                           'stride': integer(1, 10000, 15), 'max_frames': integer(0, 1000000, 0)},
                          ['type', 'path'], True),
        'SegmentVideo': obj({'type': {'const': 'segment_video'}, 'path': TEXT,
                             'max_frames': integer(0, 1000, 0)}, ['type', 'path'], True),
        'FrameAnalysis': obj({'type': {'enum': ['segment_frame', 'ocr_frame', 'scene_frame']},
                              'path': TEXT, 'frame_index': integer(0, 10000, 0)}, ['type', 'path'], True),
        'Search': obj({'type': {'const': 'search'}, 'query': {
            'type': 'string', 'minLength': 1, 'maxLength': 1024, 'pattern': '[^ \\r\\n\\t]',
            'description': 'Server limit is 1024 UTF-8 BYTES, not characters. Whitespace-only rejected.',
            'x-max-utf8-bytes': 1024}, 'limit': integer(1, 20, 8),
            'scope': {'enum': ['all', 'live'], 'default': 'all'}}, ['type', 'query'], True),
        'JobRequest': {'oneOf': [ref(n) for n in ['IndexVideo', 'SegmentVideo', 'FrameAnalysis', 'Search']]},
        # Accepted requests include server-owned fingerprints and internal archive jobs.
        # They intentionally are NOT constrained by the client-only JobRequest schema.
        'Job': obj({'id': JOB_ID, 'state': {'enum': JOB_STATES},
                    'request': obj({'type': TEXT}, ['type']), 'progress': {'type': 'object'},
                    'created_ms': integer(), 'updated_ms': integer(), 'cancel_requested': BOOL,
                    'attempts': integer(), 'recoveries': integer(),
                    'result': {'type': ['object', 'null'], 'description': 'Type-specific result; see result schemas.'},
                    'error': NULL_TEXT}),
        'Jobs': obj({'jobs': arr(ref('Job'))}),
        'Health': obj({'status': {'enum': ['ready', 'storage_error']}, 'worker_count': {'const': 1},
                       'queue_capacity': integer(1), 'history_capacity': integer(1),
                       'history_persistent': BOOL, 'detector_backend': ref('Backend'),
                       'segmentation_enabled': BOOL, 'video_analysis_enabled': BOOL,
                       'video_analysis_max_frames': {'const': 1000}, 'ocr_enabled': BOOL,
                       'ocr_unicode': BOOL, 'vlm_enabled': BOOL, 'storage_error': TEXT}),
        'Media': obj({'videos': arr(obj({'path': TEXT}))}),
        'ArchiveConfig': obj({'segment_seconds': NUMBER, 'max_segments_per_session': integer(),
                              'max_total_segments': integer(), 'max_bytes': integer()}),
        'Sources': obj({'sources': arr(obj({'id': TEXT, 'label': TEXT})), 'max_active': {'const': 1},
                        'duration_seconds': integer(1), 'backend': ref('Backend'),
                        'source_kind': {'enum': ['rtsp', 'file_replay']},
                        'analysis_mode': {'enum': ['detection', 'segmentation']}, 'tracker': TEXT,
                        'preview_max_age_ms': integer(), 'persistent': {'const': False},
                        'archive_available': BOOL, 'archive_config': ref('ArchiveConfig')}),
        'LiveStart': obj({'source_id': TEXT, 'archive': {'type': 'boolean', 'default': False}}, ['source_id'], True),
        'ArchiveIndex': obj({'session_id': LIVE_ID, 'segment_index': integer(1, 8)}, closed=True),
        'LiveSession': obj({'id': LIVE_ID, 'source_id': TEXT,
            'source_kind': {'enum': ['rtsp', 'file_replay']},
            'state': {'enum': ['starting', 'running', 'stopping', 'stopped', 'completed', 'failed']},
            'active': BOOL, 'backend': ref('Backend'), 'analysis_mode': {'enum': ['detection', 'segmentation']},
            'connection_state': {'enum': ['connecting', 'live', 'reconnecting', 'stopped']},
            'cancel_requested': BOOL, 'error': TEXT, 'has_preview': BOOL, 'preview_sequence': integer(),
            'decode_age_ms': {'type': ['number', 'null']}, 'processed_frames': integer(),
            'decoded_frames': integer(), 'sessions': integer(), 'connection_attempts': integer(),
            'read_failures': integer(), 'dropped_frames': integer(), 'queue_high_watermark': integer(),
            'queue_capacity': integer(1), 'tracking_epochs': integer(), 'mean_analysis_ms': NUMBER,
            'stop_reason': TEXT, 'duration_seconds': integer(1), 'elapsed_ms': integer(),
            'archive': {'type': 'object'}, 'timestamp_basis': TEXT}),
        'LiveStatus': obj({'session': {'anyOf': [ref('LiveSession'), {'type': 'null'}]}}),
        'ArchiveCatalog': obj({'enabled': BOOL, 'config': ref('ArchiveConfig'),
            'segments': arr(obj({'session_id': LIVE_ID, 'segment_index': integer(1, 8),
                'source_id': TEXT, 'media_path': NULL_TEXT, 'index_state': {'enum': JOB_STATES + ['pending']},
                'job_id': NULL_TEXT, 'error': NULL_TEXT}, ['session_id', 'segment_index', 'index_state', 'error']))}),
        'IndexResult': obj({'source_id': TEXT, 'decoded_frames': integer(), 'sampled_frames': integer(),
            'indexed_items': integer(), 'source_fps': NUMBER, 'used_fallback_fps': BOOL, 'stop_reason': TEXT}),
        'SearchResult': obj({'query': TEXT, 'scope': {'enum': ['all', 'live']},
            'results': arr(obj({'id': TEXT, 'score': NUMBER, 'metadata': {'type': 'object', 'additionalProperties': TEXT},
                               'media_path': TEXT}, ['id', 'score', 'metadata']))}),
        'FrameResult': obj({'path': TEXT, 'frame_index': integer(), 'width': integer(1), 'height': integer(1),
                            'model_signature': TEXT, 'preview_data_url': {'type': 'string', 'pattern': '^data:image/jpeg;base64,'}}),
        'SegmentResult': {'allOf': [ref('FrameResult'), obj({'tracking': {'const': False},
            'mask_format': {'const': 'COCO uncompressed column-major RLE'}, 'instances': arr(obj({
                'label': TEXT, 'score': NUMBER, 'bbox': {'type': 'array', 'items': NUMBER, 'minItems': 4, 'maxItems': 4},
                'mask_pixels': integer(), 'segmentation': obj({'size': {'type': 'array', 'items': integer(1), 'minItems': 2, 'maxItems': 2},
                                                              'counts': arr(integer())})}))})]},
        'OcrResult': {'allOf': [ref('FrameResult'), obj({'regions': arr(obj({'text': TEXT,
            'polygon': {'type': 'array', 'minItems': 4, 'maxItems': 4,
                        'items': {'type': 'array', 'minItems': 2, 'maxItems': 2, 'items': NUMBER}},
            'detection_confidence': NUMBER, 'recognition_confidence': NUMBER})),
            'charset': TEXT, 'polygon_order': {'const': 'BL,TL,TR,BR'}, 'recognition_confidence_kind': TEXT})]},
        'SceneResult': {'allOf': [ref('FrameResult'), obj({'summary': TEXT, 'verified': {'const': False}, 'warning': TEXT})]},
        'VideoResult': obj({'path': TEXT, 'export_id': TEXT, 'frames': integer(1), 'fps': NUMBER,
            'duration_seconds': NUMBER, 'width': integer(1), 'height': integer(1), 'bytes': integer(1),
            'instances': integer(), 'track_ids': integer(), 'tracking': {'const': True},
            'tracker': TEXT, 'audio': {'const': False}, 'stop_reason': TEXT, 'model_signature': TEXT}),
    }
    schemas['Job']['allOf'] = [
        {'if': obj({'state': {'const': 'succeeded'}, 'request': obj({'type': {'const': kind}})}),
         'then': obj({'result': ref(result)})}
        for kind, result in [('index_video', 'IndexResult'), ('index_live_archive', 'IndexResult'),
                             ('search', 'SearchResult'), ('segment_frame', 'SegmentResult'),
                             ('ocr_frame', 'OcrResult'), ('scene_frame', 'SceneResult'),
                             ('segment_video', 'VideoResult')]]
    json_response = lambda name: {'description': name, 'content': {'application/json': {'schema': ref(name)}}}
    errors = {str(code): json_response('Error') for code in [400, 403, 404, 409, 415, 429, 500, 503]}
    image_id = {'type':'string', 'pattern':'^[A-Za-z0-9_-]{1,64}$'}
    image_path = {'type':'string', 'minLength':1, 'maxLength':512, 'pattern':r'^[^\u0000]+$'}
    sha = {'type':'string', 'pattern':'^[0-9a-f]{64}$'}
    schemas['RedisImageRequest'] = obj({'id':image_id, 'image':image_path}, closed=True)
    schemas['RedisImageStatus'] = obj({'enabled':BOOL,'queue':TEXT,'context':TEXT,
        'backend':{'const':'opencv-dnn-cpu'}, 'worker_status':{'const':'unknown'},
        'images':{'type':'array','items':image_path,'maxItems':128}}, ['enabled'], closed=True)
    schemas['RedisImageResult'] = obj({'type':{'const':'detect_image'},'context':TEXT,
        'model_sha256':sha,'image_sha256':sha,'image':image_path,'width':integer(1),'height':integer(1),
        'inference_ms':{'type':'number','minimum':0},'detections':{'type':'array','maxItems':100,
        'items':obj({'label':{'type':'string','maxLength':64},'confidence':{'type':'number','minimum':0,'maximum':1},
        'class_id':integer(0,79),'xyxy':{'type':'array','items':NUMBER,'minItems':4,'maxItems':4}},closed=True)}},closed=True)
    schemas['RedisImageJob'] = obj({'id':image_id,'state':{'enum':['queued','running','succeeded','failed','cancelled']},
        'attempts':integer(0,3),'payload':obj({'type':{'const':'detect_image'},'image':image_path,'sha256':sha},closed=True),
        'result':{'oneOf':[{'type':'null'},ref('RedisImageResult'),obj({'error':TEXT},closed=True)]}},closed=True)
    paths = {}

    def add(path, method, op_id, summary, schema, cpp, codes=(), body=None, parameters=None, status=200, description=''):
        responses = {str(status): json_response(schema), '403': errors['403'], '500': errors['500']}
        responses.update({str(code): errors[str(code)] for code in codes})
        op = {'operationId': op_id, 'summary': summary, 'responses': responses, 'x-cpp-route': cpp}
        if description:
            op['description'] = description
        if parameters:
            op['parameters'] = parameters
        if method == 'post':
            responses['415'] = errors['415']
            responses['413'] = {'description': 'Request exceeds 8192 bytes; framework response, not necessarily JSON.'}
        if body:
            op['requestBody'] = {'required': True, 'content': {'application/json': {'schema': ref(body)}}}
        paths.setdefault(path, {})[method] = op
        return op

    def param(name, schema, where='path'):
        return {'name': name, 'in': where, 'required': where == 'path', 'schema': schema}

    add('/api/health', 'get', 'health', 'Read capabilities and queue storage health', 'Health', '/api/health',
        description='Not a fresh Qdrant, GPU, decoder or VLM readiness probe.')['responses']['503'] = json_response('Health')
    add('/api/media', 'get', 'listMedia', 'List top-level local video files', 'Media', '/api/media')
    add('/api/jobs', 'get', 'listJobs', 'List retained jobs, newest first', 'Jobs', '/api/jobs')
    op = add('/api/jobs', 'post', 'submitJob', 'Enqueue finite analysis or search', 'Job', '/api/jobs',
        [400, 429, 503], 'JobRequest', status=202,
        description='202 is acceptance, not completion. Requires enabled model. Paths stay inside media_root; no uploads or URLs. Poll job state. max_frames=0 means all subject to job-specific limits.')
    op['responses']['202']['headers'] = {'Location': {'schema': TEXT, 'description': 'Job status URL'}}
    op['responses']['429']['headers'] = {'Retry-After': {'schema': TEXT, 'description': '2 seconds'}}
    add('/api/jobs/{id}', 'get', 'getJob', 'Read job status/result', 'Job', r'/api/jobs/([0-9]+-[0-9]+)', [404], parameters=[param('id', JOB_ID)])
    add('/api/jobs/{id}/cancel', 'post', 'cancelJob', 'Request cooperative cancellation', 'Job',
        r'/api/jobs/([0-9]+-[0-9]+)/cancel', [404, 503], parameters=[param('id', JOB_ID)],
        description='Send {} with application/json. Body is ignored by this endpoint. Completed writes are not rolled back; terminal jobs remain terminal.')
    # Body is optional because the implementation ignores it, but Content-Type remains required.
    paths['/api/jobs/{id}/cancel']['post']['requestBody'] = {'content': {'application/json': {'schema': {}}}}
    add('/api/live/sources', 'get', 'liveSources', 'Read configured presets and source provenance', 'Sources', '/api/live/sources')
    add('/api/live', 'get', 'liveStatus', 'Read active or last session', 'LiveStatus', '/api/live')
    add('/api/live/start', 'post', 'startLive', 'Start one nonpersistent analysis session', 'LiveSession',
        '/api/live/start', [400, 409], 'LiveStart', status=202,
        description='Select server preset ID only. source_kind=file_replay is a local demo, not RTSP evidence. No source URL/path in request.')
    add('/api/live/{id}/stop', 'post', 'stopLive', 'Request session stop', 'LiveSession',
        r'/api/live/(live-[0-9]+-[0-9]+)/stop', [400, 404], 'Empty', [param('id', LIVE_ID)], 202)
    add('/api/live/archive', 'get', 'listArchive', 'List sealed analysis-frame clips', 'ArchiveCatalog', '/api/live/archive')
    add('/api/live/archive/index', 'post', 'indexArchive', 'Retry indexing one sealed archive segment', 'Job',
        '/api/live/archive/index', [400, 409, 429, 503], 'ArchiveIndex', status=202)

    def binary(path, op_id, summary, cpp, mime, parameters, ranges=False):
        op = add(path, 'get', op_id, summary, 'Error', cpp, [404], parameters=parameters)
        op['responses']['200'] = {'description': 'Binary or CSV content', 'content': {mime: {}}}
        if ranges:
            op['responses']['206'] = {'description': 'Byte range content; multiple ranges may use multipart/byteranges.', 'content': {mime: {}, 'multipart/byteranges': {}}}
            op['responses']['416'] = {'description': 'Unsatisfiable range; framework response.'}
            op['parameters'].append(param('Range', TEXT, 'header'))
        return op

    preview = binary('/api/live/{id}/preview.jpg', 'livePreview', 'Read current shared JPEG',
        r'/api/live/(live-[0-9]+-[0-9]+)/preview.jpg', 'image/jpeg', [param('id', LIVE_ID)])
    preview['responses']['204'] = {'description': 'No current frame, stopped, disconnected or decode-arrival age exceeds 2 seconds. Empty body; clear stale UI.'}
    preview['responses']['200']['headers'] = {name: {'schema': TEXT, 'description': name} for name in
        ['X-Live-Sequence', 'X-Live-Source-Session', 'X-Live-Epoch', 'X-Decode-Age-Ms']}
    binary('/api/preview/{id}/{index}.jpg', 'searchPreview', 'Render a completed search result',
        r'/api/preview/([0-9]+-[0-9]+)/([0-9]+)\.jpg', 'image/jpeg', [param('id', JOB_ID), param('index', integer())])
    binary('/media/{path}', 'mediaVideo', 'Read a permitted local video', '/media/(.+)', 'video/*',
        [param('path', {'type': 'string', 'description': 'Relative media path; encode each segment. Nested archive clip paths supported; no traversal.'})], True)
    for file, mime in [('video.mp4', 'video/mp4'), ('tracks.csv', 'text/csv'), ('preview.jpg', 'image/jpeg')]:
        binary('/api/jobs/{id}/' + file, 'analysis_' + file.replace('.', '_'), 'Read completed segment_video output: ' + file,
            r'/api/jobs/([0-9]+-[0-9]+)/(video\.mp4|tracks\.csv|preview\.jpg)', mime,
            [param('id', JOB_ID), param('download', {'type': 'string', 'description': 'Presence forces attachment. CSV always downloads.'}, 'query')], True)

    add('/api/redis-images','get','redisImageStatus','Read optional Redis queue and local image catalog',
        'RedisImageStatus','/api/redis-images',[503],description='enabled=false when not configured. Reachable Redis does not prove a worker is alive. Top-level JPG/PNG catalog only, bounded scan.')
    add('/api/redis-images','post','redisImageSubmit','Submit one image with a stable client-generated ID',
        'RedisImageJob','/api/redis-images',[400,409,429,503],'RedisImageRequest',status=202,
        description='Separate CPU worker, not SQLite. Retry uncertain submission with the SAME id/image. Server hashes local bytes. No upload, external URL, model selection or raw queue commands.')
    add('/api/redis-images/{id}','get','redisImageGet','Read Redis image state and detections',
        'RedisImageJob',r'/api/redis-images/([A-Za-z0-9_-]{1,64})',[404,503],parameters=[param('id',image_id)])
    add('/api/redis-images/{id}/cancel','post','redisImageCancel','Fence a Redis image job result',
        'RedisImageJob',r'/api/redis-images/([A-Za-z0-9_-]{1,64})/cancel',[400,404,503],'Empty',[param('id',image_id)],
        description='Cancellation blocks result acceptance, not an in-flight OpenCV call. Terminal jobs are unchanged.')
    preview=binary('/api/redis-images/{id}/preview.jpg','redisImagePreview','Render verified image and detection boxes',
        r'/api/redis-images/([A-Za-z0-9_-]{1,64})/preview\.jpg','image/jpeg',[param('id',image_id)])
    for status in ['409','503']: preview['responses'][status]={'description':'Incomplete, changed media or unavailable queue',
        'content':{'application/json':{'schema':ref('Error')}}}

    return {'openapi': '3.1.1', 'info': {'title': 'AegisVision local C++ API', 'version': '0.1.0',
        'description': 'Local-only service, no authentication/TLS. Do not expose publicly. Host must be localhost:<port> or 127.0.0.1:<port>; Origin, if present, must equal http://Host. POST requires application/json, maximum 8192 bytes. One finite-job worker; live sessions are separate and nonpersistent. Models/features vary: inspect /api/health. Schemas describe the contract, not runtime schema enforcement. No file-upload endpoint.'},
        'servers': [{'url': '/', 'description': 'Same origin as this document (8090 CPU or configured port)'}],
        'paths': paths, 'components': {'schemas': schemas},
        'externalDocs': {'url': 'https://github.com/bekiroruk/aegis-vision/blob/main/docs/api.md'}}


def render():
    return json.dumps(document(), ensure_ascii=False, indent=2) + '\n'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--write', action='store_true')
    mode.add_argument('--check', action='store_true')
    args = parser.parse_args()
    if args.write:
        TARGET.write_text(render(), encoding='utf-8', newline='\n')
    elif not TARGET.exists() or TARGET.read_text(encoding='utf-8') != render():
        raise SystemExit('web/openapi.json is stale; run scripts/openapi_contract.py --write')
