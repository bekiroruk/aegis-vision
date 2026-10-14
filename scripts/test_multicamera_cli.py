"""Synthetic CLI contract only: no real camera/model accuracy claim or downloads."""
import copy
import argparse
import json
from pathlib import Path
import subprocess
import tempfile


def fixture():
    space = 'synthetic-person-space-not-a-model'
    def track(camera, local_id, start, end, dimension):
        vector = [0.] * 512
        vector[dimension] = 1.
        return dict(camera=camera, epoch=1, local_id=local_id, start_ms=start, end_ms=end,
                    embedding_space=space, embedding=vector)
    return dict(version=1, time_basis='synchronized_ms', config=dict(
        cameras=['entrance', 'corridor'], embedding_space=space, min_similarity=.85, ambiguity_margin=.05,
        transitions=[{'from': 'entrance', 'to': 'corridor', 'min_gap_ms': 100, 'max_gap_ms': 2000}]),
        tracklets=[track('entrance', 1, 0, 500, 0), track('corridor', 1, 800, 1200, 0),
                   track('entrance', 2, 0, 500, 1), track('corridor', 2, 800, 1200, 1)])


def run(exe):
    data = fixture()
    # All writes/deletion are limited to this newly created temporary test directory.
    with tempfile.TemporaryDirectory(prefix='aegis-multicamera-') as directory:
        root = Path(directory)
        source = root / 'input.json'
        def invoke(value, name, success):
            source.write_text(json.dumps(value), encoding='utf-8')
            result = subprocess.run([str(exe), str(source), str(root / name)], capture_output=True, text=True, timeout=15)
            assert (result.returncode == 0) == success, result.stdout + result.stderr
            return root / name
        out = invoke(data, 'valid', True)
        report = json.loads((out / 'report.json').read_text())
        assert report['accepted_handoffs'] == 2 and report['identity_hypotheses'] == 2
        assert report['verified_person_identities'] is False
        assert 'embedding' not in report['tracklets'][0]
        original = (out / 'report.json').read_bytes()
        invoke(data, 'valid', False)
        assert (out / 'report.json').read_bytes() == original
        for index, change in enumerate([
            lambda x: x.update(time_basis='camera_pts'),
            lambda x: x['tracklets'][0].update(local_id=1.2),
            lambda x: x['tracklets'][0].update(epoch=-1),
            lambda x: x['tracklets'][0].update(embedding_space='different'),
            lambda x: x['tracklets'][0].update(extra=True),
            lambda x: x['tracklets'][0].update(embedding=[0.] * 512),
        ]):
            bad = copy.deepcopy(data)
            change(bad)
            output = invoke(bad, f'invalid-{index}', False)
            assert not output.exists(), 'Invalid input created output'
        ambiguous = copy.deepcopy(data)
        ambiguous['tracklets'][2]['embedding'] = ambiguous['tracklets'][0]['embedding']
        ambiguous['tracklets'][3]['embedding'] = ambiguous['tracklets'][1]['embedding']
        out = invoke(ambiguous, 'ambiguous', True)
        assert json.loads((out / 'report.json').read_text())['accepted_handoffs'] == 0
        print('PASS: synthetic CLI two handoffs, ambiguity rejection, strict input, no overwrite. No real-camera accuracy measured.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('executable', type=Path)
    parser.add_argument('--example-directory', type=Path, help='Also retain a clearly synthetic example in a NEW directory')
    args = parser.parse_args()
    exe = args.executable.resolve()
    run(exe)
    if args.example_directory:
        destination = args.example_directory.resolve()
        destination.mkdir(exist_ok=False)
        source = destination / 'synthetic-input.json'
        source.write_text(json.dumps(fixture(), indent=2), encoding='utf-8')
        subprocess.run([str(exe), str(source), str(destination / 'synthetic-result')], check=True, timeout=15)
        print('Retained synthetic example:', destination)
