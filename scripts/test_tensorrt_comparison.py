import copy
import unittest
from compare_tensorrt import compare


def fixtures():
    cpu = {"kind": "images", "opencv_threads": 1, "opencv_version": "4.12", "manifest_sha256": "same",
        "data_provenance": {"images": [1, 2]}, "provenance": {"engine": "opencv-dnn", "precision": "fp32",
        "model_sha256": "model", "config_sha256": "config", "confidence_threshold": .001,
        "nms_iou_threshold": .45, "input_size": 640, "max_detections": 300},
        "performance": {"warmup_iterations": 5, "detector": {"samples": 2, "mean_ms": 100., "p50_ms": 100., "p95_ms": 120.}},
        "detection": {"ap50": .8, "ap50_95": .6, "per_class": [{"label": "cat", "ap50_95": .6}], "operating_point": {}}}
    fp32, fp16 = copy.deepcopy(cpu), copy.deepcopy(cpu)
    for report, precision in ((fp32, "fp32"), (fp16, "fp16-mixed")):
        report["provenance"].update(engine="tensorrt", precision=precision)
        report["performance"]["detector"].update(mean_ms=10., p50_ms=10., p95_ms=12.)
    return cpu, fp32, fp16


class ComparisonTests(unittest.TestCase):
    def test_pass_is_not_deployment(self):
        result = compare(*fixtures())
        self.assertTrue(result["fp16_quality_and_speed_gate_passed"])
        self.assertFalse(result["deployment_approved"])
        self.assertEqual(result["panels"]["trt_fp16"]["p50_speedup_vs_cpu"], 10.)

    def test_quality_regression(self):
        cpu, fp32, fp16 = fixtures()
        fp16["detection"]["ap50_95"] = .59
        self.assertFalse(compare(cpu, fp32, fp16)["fp16_quality_and_speed_gate_passed"])

    def test_class_regression(self):
        cpu, fp32, fp16 = fixtures()
        fp16["detection"]["per_class"][0]["ap50_95"] = .57
        self.assertFalse(compare(cpu, fp32, fp16)["fp16_quality_and_speed_gate_passed"])

    def test_tail_regression(self):
        cpu, fp32, fp16 = fixtures()
        fp16["performance"]["detector"]["p95_ms"] = 150
        self.assertFalse(compare(cpu, fp32, fp16)["fp16_quality_and_speed_gate_passed"])

    def test_invalid_protocol(self):
        changes = [("opencv_threads", 8), ("manifest_sha256", "different"), ("opencv_version", "other")]
        for key, value in changes:
            cpu, fp32, fp16 = fixtures()
            fp16[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                compare(cpu, fp32, fp16)

    def test_wrong_weights_precision_or_threshold(self):
        for key, value in (("model_sha256", "different"), ("precision", "fp32"), ("confidence_threshold", .35)):
            cpu, fp32, fp16 = fixtures()
            fp16["provenance"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                compare(cpu, fp32, fp16)

    def test_invalid_values(self):
        for value in (0, -1, float("nan"), float("inf")):
            cpu, fp32, fp16 = fixtures()
            fp16["performance"]["detector"]["p50_ms"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                compare(cpu, fp32, fp16)

    def test_missing_samples_and_warmup(self):
        cpu, fp32, fp16 = fixtures()
        fp16["performance"]["detector"]["samples"] = 1
        with self.assertRaises(ValueError):
            compare(cpu, fp32, fp16)
        fp16["performance"]["detector"]["samples"] = 2
        fp16["performance"]["warmup_iterations"] = 0
        with self.assertRaises(ValueError):
            compare(cpu, fp32, fp16)


if __name__ == "__main__":
    unittest.main()
