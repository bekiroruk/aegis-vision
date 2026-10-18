"""No Docker/GPU: assertions used by the opt-in runtime acceptance test."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("runtime", Path(__file__).with_name("test_gpu_container_runtime.py"))
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)


class RuntimeContractTests(unittest.TestCase):
    def health(self):
        return {"status": "ready", "storage_error": "", "history_persistent": True,
                "detector_backend": {"engine": "tensorrt", "precision": "fp16-mixed",
                                     "device": "cuda:0", "engine_sha256": "a" * 64}}

    def test_gpu_contract_rejects_cpu_int8_and_missing_identity(self):
        runtime.require_gpu(self.health())
        for key, value in (("engine", "opencv-dnn"), ("precision", "int8"),
                           ("device", "cpu"), ("engine_sha256", "")):
            data = self.health()
            data["detector_backend"][key] = value
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                runtime.require_gpu(data)

    def test_persistence_and_storage_health_are_required(self):
        for key, value in (("history_persistent", False), ("storage_error", "disk full"), ("status", "starting")):
            data = self.health()
            data[key] = value
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                runtime.require_gpu(data)

    def test_unrelated_pending_work_prevents_restart(self):
        for state in ("running", "queued"):
            with patch.object(runtime, "request", return_value={"jobs": [{"id": "other", "state": state}]}):
                with self.assertRaises(RuntimeError):
                    runtime.require_idle(except_id="mine")
        with patch.object(runtime, "request", return_value={"jobs": [{"id": "mine", "state": "running"}]}):
            runtime.require_idle(except_id="mine")

    def test_only_running_job_with_progress_counts_as_interruption_test(self):
        done = {"state": "succeeded", "id": "1-1"}
        with patch.object(runtime, "request", return_value=done):
            self.assertEqual(runtime.job_until("1-1"), done)
            with self.assertRaises(RuntimeError):
                runtime.job_until("1-1", running=True)
        active = {"state": "running", "progress": {"indexed_items": 2}}
        with patch.object(runtime, "request", return_value=active):
            self.assertEqual(runtime.job_until("1-1", running=True), active)

    def test_failed_job_is_not_reported_as_success(self):
        for state in ("failed", "cancelled"):
            with patch.object(runtime, "request", return_value={"state": state}):
                with self.assertRaises(RuntimeError):
                    runtime.job_until("1-1")


if __name__ == "__main__":
    unittest.main()
