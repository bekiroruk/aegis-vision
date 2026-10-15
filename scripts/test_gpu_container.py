"""No GPU/daemon required: container preflight and startup regressions."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("gpu_entry", ROOT / "deploy/gpu/entrypoint.py")
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)


class ContainerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="aegis-gpu-preflight-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.files = ["models/yolov8n.onnx", "models/clip-vit-b32/manifest.json",
            "models/clip-vit-b32/vision.onnx", "models/clip-vit-b32/text.onnx",
            "models/clip-vit-b32/tokenizer.json", "engines/linux-fp16/model.engine",
            "opt/aegis/build/aegisvision_server", "opt/aegis/web/index.html"]
        for name in self.files:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"fixture")
        self.manifest = self.root / "engines/linux-fp16/manifest.json"
        self.manifest.write_text(json.dumps({"precision": "fp16-mixed"}), encoding="utf-8")
        for name in ("media", "state"):
            (self.root / name).mkdir()

    def test_complete_mounts_no_writes(self):
        before = {p.relative_to(self.root): p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        entry.preflight(self.root)
        self.assertEqual(before, {p.relative_to(self.root): p.read_bytes()
                                  for p in self.root.rglob("*") if p.is_file()})

    def test_each_empty_required_file_rejected(self):
        for name in self.files:
            path = self.root / name
            path.write_bytes(b"")
            with self.subTest(name=name), self.assertRaises(ValueError): entry.preflight(self.root)
            path.write_bytes(b"fixture")

    def test_missing_mount_does_not_get_created(self):
        (self.root / "state").rmdir()
        with self.assertRaisesRegex(ValueError, "state"): entry.preflight(self.root)
        self.assertFalse((self.root / "state").exists())

    def test_unwritable_mount_rejected(self):
        with patch.object(entry.os, "access", return_value=False):
            with self.assertRaisesRegex(ValueError, "unwritable"): entry.preflight(self.root)

    def test_unapproved_or_malformed_engine(self):
        for value in ('{}', '[]', '{', '{"precision":"int8"}', ' ' * 65537):
            self.manifest.write_text(value, encoding="utf-8")
            with self.subTest(value=value[:30]), self.assertRaises(ValueError): entry.preflight(self.root)

    def test_health_requires_actual_fp16_backend(self):
        good = {"status": "ready", "detector_backend": {
            "engine": "tensorrt", "precision": "fp16-mixed", "device": "cuda:0"}}
        self.assertTrue(entry.healthy(good))
        for value in (None, [], {}, {"status": "ready", "detector_backend": "tensorrt"},
                      dict(good, status="storage_error")):
            self.assertFalse(entry.healthy(value))
        for key, value in (("engine", "opencv"), ("precision", "int8"), ("device", "cpu")):
            bad = dict(good, detector_backend=dict(good["detector_backend"], **{key: value}))
            self.assertFalse(entry.healthy(bad))

    def test_check_does_not_contact_or_start_service(self):
        with patch.object(entry.sys, "argv", ["entry", "--check"]), \
             patch.object(entry, "preflight") as preflight, \
             patch.object(entry, "fetch_json") as fetch, patch.object(entry.os, "execv") as run:
            entry.main()
            preflight.assert_called_once()
            fetch.assert_not_called()
            run.assert_not_called()

    def test_start_execs_cpp_with_mandatory_gpu_and_isolated_state(self):
        with patch.object(entry.sys, "argv", ["entry"]), patch.object(entry, "preflight"), \
             patch.object(entry, "fetch_json", return_value={}), patch.object(entry.os, "execv") as run:
            entry.main()
            executable, argv = run.call_args.args
            self.assertEqual(executable, argv[0])
            self.assertEqual(argv[-2:], ["--trt-bundle", "/engines/linux-fp16"])
            self.assertIn("8093", argv)
            self.assertIn("/state/jobs.sqlite", argv)

    def test_dependency_failure_never_launches_cpu_fallback(self):
        with patch.object(entry.sys, "argv", ["entry"]), patch.object(entry, "preflight"), \
             patch.object(entry, "fetch_json", side_effect=OSError("offline")) as fetch, \
             patch.object(entry.time, "sleep"), patch.object(entry.os, "execv") as run:
            with self.assertRaises(RuntimeError): entry.main()
            self.assertEqual(fetch.call_count, 15)
            run.assert_not_called()


if __name__ == "__main__": unittest.main()
