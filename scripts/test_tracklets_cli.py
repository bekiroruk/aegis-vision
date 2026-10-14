"""No weights/downloads. Synthetic export-to-association contract, not accuracy."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

EXE = Path(sys.argv.pop(1)).resolve()
ASSOCIATE = Path(sys.argv.pop(1)).resolve()


class BridgeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="aegis-tracklets-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.exports = []
        for camera in ("a", "b"):
            track = dict(camera=camera, epoch=1, local_id=1, start_ms=0,
                         end_ms=100, embedding_space="synthetic-space",
                         embedding=[1.] + [0.] * 511, observations=3,
                         closure="missing_timeout", eligible=True)
            self.exports.append(dict(version=1, complete=True,
                time_basis="relative_cfr_estimate_ms", embedding_space="synthetic-space",
                config=dict(camera=camera, epoch=1, min_observations=3), tracklets=[track]))
        self.mapping = dict(version=1, time_basis="synchronized_ms",
            clock_evidence="Synthetic test clock, not a camera measurement",
            sources=[dict(file="a.json", origin_ms=0), dict(file="b.json", origin_ms=200)],
            cameras=["a", "b"], transitions=[dict(from_="a", to="b", min_gap_ms=100, max_gap_ms=200)],
            min_similarity=.85, ambiguity_margin=.05)
        self.mapping["transitions"][0]["from"] = self.mapping["transitions"][0].pop("from_")

    def run_prepare(self, success=True):
        for name, value in zip(("a.json", "b.json"), self.exports):
            (self.root / name).write_text(json.dumps(value), encoding="utf-8")
        (self.root / "map.json").write_text(json.dumps(self.mapping), encoding="utf-8")
        result = subprocess.run([str(EXE), "--prepare-association", str(self.root / "map.json"),
                                 str(self.root / "prepared")], capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode == 0, success, result.stdout + result.stderr)
        if not success:
            self.assertFalse((self.root / "prepared").exists())
        return self.root / "prepared"

    def test_real_cli_connection(self):
        prepared = self.run_prepare()
        payload = json.loads((prepared / "association-input.json").read_text())
        self.assertEqual(payload["tracklets"][1]["start_ms"], 200)
        self.assertEqual(payload["tracklets"][0]["end_ms"], 100)
        subprocess.run([str(ASSOCIATE), str(prepared / "association-input.json"),
                        str(self.root / "result")], check=True, capture_output=True, timeout=20)
        report = json.loads((self.root / "result/report.json").read_text())
        self.assertEqual(report["accepted_handoffs"], 1)
        self.assertFalse(json.loads((prepared / "provenance.json").read_text())["clock_verified_by_software"])
        before = (prepared / "association-input.json").read_bytes()
        result = subprocess.run([str(EXE), "--prepare-association", str(self.root / "map.json"), str(prepared)],
                                capture_output=True, timeout=20)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(before, (prepared / "association-input.json").read_bytes())

    def test_censored_excluded(self):
        self.exports[1]["tracklets"][0].update(closure="frame_limit", eligible=False)
        prepared = self.run_prepare()
        self.assertEqual(len(json.loads((prepared / "association-input.json").read_text())["tracklets"]), 1)

    def test_clock_required(self):
        self.mapping["clock_evidence"] = ""
        self.run_prepare(False)

    def test_local_clock_rejected(self):
        self.mapping["time_basis"] = "relative_cfr_estimate_ms"
        self.run_prepare(False)

    def test_overflow_rejected(self):
        self.mapping["sources"][1]["origin_ms"] = 2**63 - 1
        self.run_prepare(False)

    def test_mixed_model_rejected(self):
        self.exports[1]["embedding_space"] = "another-model"
        self.run_prepare(False)

    def test_false_eligibility_rejected(self):
        self.exports[1]["tracklets"][0]["closure"] = "end_of_input"
        self.run_prepare(False)

    def test_incomplete_export_rejected(self):
        self.exports[1]["complete"] = False
        self.run_prepare(False)

    def test_duplicate_camera_rejected(self):
        self.exports[1] = copy.deepcopy(self.exports[0])
        self.run_prepare(False)

    def test_fractional_offset_rejected(self):
        self.mapping["sources"][1]["origin_ms"] = .5
        self.run_prepare(False)

    def test_insufficient_observations_rejected(self):
        self.exports[1]["tracklets"][0]["observations"] = 1
        self.run_prepare(False)

    def test_bad_embedding_rejected(self):
        self.exports[1]["tracklets"][0]["embedding"] = [1]
        self.run_prepare(False)

    def test_epoch_mismatch_rejected(self):
        self.exports[1]["tracklets"][0]["epoch"] = 2
        self.run_prepare(False)


if __name__ == "__main__":
    unittest.main()
