"""Malformed-input CLI regressions, no model/network/real image required."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

EXE = Path(sys.argv.pop(1)).resolve()


class InputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="aegis-crossview-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        data = b"placeholder; must never reach inference"
        (self.root / "fake.png").write_bytes(data)
        views = [dict(camera=camera, image="fake.png", sha256=hashlib.sha256(data).hexdigest(),
                      persons=[dict(identity=1, bbox=[0, 0, 40, 100])]) for camera in ("C1", "C2")]
        self.job = dict(version=1, protocol="wildtrack-c1-c2-same-frame-gt-crops-v1",
            cameras=["C1", "C2"], min_similarity=.85, ambiguity_margin=.05,
            frames=[dict(frame=i, views=copy.deepcopy(views)) for i in range(0, 2000, 125)])

    def rejects(self, expected):
        path = self.root / "input.json"
        path.write_text(json.dumps(self.job), encoding="utf-8")
        result = subprocess.run([str(EXE), str(path), str(self.root / "absent-model"),
                                 str(self.root / "result")], capture_output=True, text=True, timeout=10)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(expected, result.stderr)
        self.assertFalse((self.root / "result").exists())

    def test_version(self):
        self.job["version"] = 1.5
        self.rejects("integer")

    def test_changed_threshold(self):
        self.job["min_similarity"] = .4
        self.rejects("frozen thresholds")

    def test_missing_frames(self):
        self.job["frames"].pop()
        self.rejects("16-frame")

    def test_cherry_picked_frame(self):
        self.job["frames"][0]["frame"] = 5
        self.rejects("subset frame")

    def test_same_camera(self):
        self.job["frames"][0]["views"][1]["camera"] = "C1"
        self.rejects("C1 and C2")

    def test_duplicate_label(self):
        self.job["frames"][0]["views"][0]["persons"] *= 2
        self.rejects("Duplicate person")

    def test_small_crop(self):
        self.job["frames"][0]["views"][0]["persons"][0]["bbox"] = [0, 0, 19, 100]
        self.rejects("size filter")

    def test_corrupt_image(self):
        self.job["frames"][0]["views"][0]["sha256"] = "0" * 64
        self.rejects("SHA256 mismatch")

    def test_path_escape(self):
        self.job["frames"][0]["views"][0]["image"] = "../escape.png"
        self.rejects("escapes dataset")

    def test_no_overwrite(self):
        result_dir = self.root / "result"
        result_dir.mkdir()
        (result_dir / "existing.txt").write_text("keep", encoding="utf-8")
        result = subprocess.run([str(EXE), "missing.json", "missing-model", str(result_dir)], capture_output=True, timeout=10)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual((result_dir / "existing.txt").read_text(), "keep")

    def development(self):
        self.job.update(protocol="wildtrack-flip-development-v1", cameras=["C3", "C4"])
        views = self.job["frames"][0]["views"]
        for camera, view in zip(("C3", "C4"), views):
            view["camera"] = camera
            view["persons"][0]["identity"] = 2
        self.job["frames"] = [dict(frame=i, views=copy.deepcopy(views)) for i in range(60, 2000, 50)]

    def test_identity_split(self):
        self.development()
        self.job["frames"][0]["views"][0]["persons"][0]["identity"] = 3
        self.rejects("Identity crossed experiment split")

    def test_development_frame_selection(self):
        self.development()
        self.job["frames"][1]["frame"] = 125
        self.rejects("subset frame")


if __name__ == "__main__":
    unittest.main()
