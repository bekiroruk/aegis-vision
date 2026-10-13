import copy
import unittest
from prepare_int8_calibration import select_images
from compare_int8 import compare, inspect_int8
from test_tensorrt_comparison import fixtures


def inputs():
    _, _, fp16 = fixtures()
    evaluation = [{"coco_image_id": i, "sha256": f"eval-{i}"} for i in range(64)]
    fp16["data_provenance"]["images"] = evaluation
    fp16["performance"]["detector"]["samples"] = 64
    fp16["provenance"]["tensorrt"] = {"gpu": "test", "compute_major": 8, "compute_minor": 6,
        "cuda_runtime": 13040, "tensorrt": 101303}
    int8 = copy.deepcopy(fp16)
    int8["provenance"]["precision"] = "int8-mixed"
    int8["performance"]["detector"].update(mean_ms=8., p50_ms=8., p95_ms=9.)
    proof = {"images": 128, "algorithm": "entropy2", "batch_size": 1, "evaluation_manifest_sha256": "same"}
    int8["provenance"]["tensorrt"]["calibration"] = proof
    cal = {"version": 1, "kind": "int8-calibration", "algorithm": "entropy2", "batch_size": 1,
        "model_sha256": "model", "evaluation_manifest_sha256": "same", "excluded_images": evaluation,
        "images": [{"coco_image_id": i+64, "sha256": f"cal-{i}", "decoded_sha256": f"pixel-{i}"} for i in range(128)],
        "limitations": []}
    return fp16, int8, cal


class Int8Tests(unittest.TestCase):
    def test_actual_int8_evidence_required(self):
        layers = {"Layers": [{"Outputs": [{"Format/Datatype": "Half"}]}]}
        with self.assertRaises(ValueError):
            inspect_int8(layers)
        layers["Layers"][0]["Outputs"].append({"Format/Datatype": "Int8"})
        self.assertEqual(inspect_int8(layers), {"Half": 1, "Int8": 1})

    def test_gate_never_deploys(self):
        result = compare(*inputs())
        self.assertTrue(result["quality_and_speed_gate_passed"])
        self.assertFalse(result["deployment_approved"])

    def test_quality_or_class_loss(self):
        for key in ("ap50", "ap50_95", "class"):
            base, candidate, cal = inputs()
            if key == "class":
                candidate["detection"]["per_class"][0]["ap50_95"] -= .03
            else:
                candidate["detection"][key] -= .006
            self.assertFalse(compare(base, candidate, cal)["quality_gate_passed"])

    def test_speed_or_tail_loss(self):
        for key, value in (("p50_ms", 9.9), ("p95_ms", 13.)):
            base, candidate, cal = inputs()
            candidate["performance"]["detector"][key] = value
            self.assertFalse(compare(base, candidate, cal)["speed_gate_passed"])

    def test_overlap_and_duplicate(self):
        for key in ("coco_image_id", "sha256", "decoded_sha256"):
            base, candidate, cal = inputs()
            cal["images"][0][key] = cal["images"][1][key]
            with self.assertRaises(ValueError):
                compare(base, candidate, cal)
            if key != "decoded_sha256":
                base, candidate, cal = inputs()
                cal["images"][0][key] = cal["excluded_images"][0][key]
                with self.assertRaises(ValueError):
                    compare(base, candidate, cal)

    def test_wrong_proof(self):
        for key, value in (("model_sha256", "wrong"), ("evaluation_manifest_sha256", "wrong"), ("batch_size", 2)):
            base, candidate, cal = inputs()
            cal[key] = value
            with self.assertRaises(ValueError):
                compare(base, candidate, cal)

    def test_wrong_hardware_or_precision(self):
        for key, value in (("gpu", "other"), ("tensorrt", 110000)):
            base, candidate, cal = inputs()
            candidate["provenance"]["tensorrt"][key] = value
            with self.assertRaises(ValueError):
                compare(base, candidate, cal)
        base, candidate, cal = inputs()
        candidate["provenance"]["precision"] = "fp16-mixed"
        with self.assertRaises(ValueError):
            compare(base, candidate, cal)

    def test_nonfinite_or_negative_latency(self):
        for value in (float("nan"), float("inf"), -1, 0):
            base, candidate, cal = inputs()
            candidate["performance"]["detector"]["p50_ms"] = value
            with self.assertRaises(ValueError):
                compare(base, candidate, cal)

    def test_missing_image_or_timing(self):
        base, candidate, cal = inputs()
        cal["images"].pop()
        with self.assertRaises(ValueError):
            compare(base, candidate, cal)
        base, candidate, cal = inputs()
        candidate["performance"]["detector"]["samples"] = 63
        with self.assertRaises(ValueError):
            compare(base, candidate, cal)

    def test_selection_frozen_disjoint_balanced(self):
        evaluation = {"images": [{"coco_image_id": i} for i in range(64)],
            "classes": [{"category_id": c} for c in range(8)]}
        annotations = {"annotations": [{"image_id": c*100+i, "category_id": c, "iscrowd": 0}
            for c in range(8) for i in range(100)]}
        selected, assigned = select_images(annotations, evaluation)
        self.assertEqual(len(selected), 128)
        self.assertFalse(set(selected) & set(range(64)))
        for c in range(8):
            self.assertEqual(list(assigned.values()).count(c), 16)
        annotations["annotations"].reverse()
        self.assertEqual((selected, assigned), select_images(annotations, evaluation))

    def test_insufficient_calibration_rejected(self):
        evaluation = {"images": [{"coco_image_id": i} for i in range(64)],
            "classes": [{"category_id": c} for c in range(8)]}
        with self.assertRaises(ValueError):
            select_images({"annotations": []}, evaluation)


if __name__ == "__main__":
    unittest.main()
