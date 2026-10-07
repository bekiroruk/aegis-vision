"""Network-free protocol tests; official metric dependencies are not needed."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest

from verify_quality_reference import (base_inputs, checked_objects, compare_count,
    compare_number, finite, integer, manifest_file, read_json, tracking_frame_inputs,
    write_success, compare_hota)


def f32(value):
    return struct.unpack("!f", struct.pack("!f", value))[0]


class QualityReferenceTests(unittest.TestCase):
    def test_hota_all_thresholds_counts_means_and_grid(self):
        reference = {name: [value] * 19 for name, value in {
            "HOTA_TP": 2, "HOTA_FP": 0, "HOTA_FN": 0,
            "HOTA": 1, "DetA": 1, "AssA": 1, "LocA": 1}.items()}
        cpp = {"mean": 1, "det_a": 1, "ass_a": 1, "loc_a": 1, "thresholds": [
            {"alpha": .05 + .05 * i, "tp": 2, "fp": 0, "fn": 0,
             "hota": 1, "det_a": 1, "ass_a": 1, "loc_a": 1} for i in range(19)]}
        self.assertEqual(len(compare_hota(cpp, reference, True)["thresholds"]), 19)
        for change in ("grid", "count", "score", "mean", "missing"):
            bad = deepcopy(cpp)
            if change == "grid": bad["thresholds"][3]["alpha"] = .5
            elif change == "count": bad["thresholds"][5]["fp"] = 1
            elif change == "score": bad["thresholds"][8]["ass_a"] = .99
            elif change == "mean": bad["mean"] = .8
            else: bad["thresholds"].pop()
            with self.subTest(change=change), self.assertRaises(ValueError):
                compare_hota(bad, reference, True)
        for key in ("mean", "det_a", "ass_a", "loc_a"): cpp[key] = None
        self.assertIsNone(compare_hota(cpp, reference, False)["means"]["mean"]["cpp"])
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)

    def fixture(self, motion=False, center=False, reid=False):
        source_box = [-1.0, 2.0, 10.123456789, 22.123456789]
        manifest = {"frames": 2, "classes": ["person"], "ground_truth": [
            {"frame_index": 1, "id": 5, "bbox": source_box}]}
        names = (["iou", "two_stage"] + (["kalman"] if motion else [])
                 + (["kalman_center"] if center else []) + (["kalman_reid"] if reid else []))
        report = {"tracking": {name: {"idf1": None} for name in names}}
        report["tracking"].update({"parameters": {"match_iou": .3}, "protocol": "fixture"})
        frames = []
        for index in (1, 2):
            frame = {"frame_index": index, "ground_truth": (
                [{"id": 5, "bbox": list(map(f32, source_box))}] if index == 1 else [])}
            frame.update({name: [] for name in names})
            frames.append(frame)
        return manifest, report, frames

    def test_legacy_two_tracker_report_remains_valid(self):
        inputs = self.fixture()
        before = deepcopy(inputs)
        self.assertEqual(tracking_frame_inputs(*inputs), ("iou", "two_stage"))
        self.assertEqual(inputs, before)

    def test_active_ablation_sixth_tracker_is_checked(self):
        inputs = self.fixture(motion=True, center=True, reid=True)
        inputs[1]["tracking"]["kalman_reid_active"] = {"idf1": None}
        for frame in inputs[2]: frame["kalman_reid_active"] = []
        self.assertEqual(tracking_frame_inputs(*inputs),
            ("iou", "two_stage", "kalman", "kalman_center", "kalman_reid", "kalman_reid_active"))
        del inputs[2][1]["kalman_reid_active"]
        with self.assertRaises(ValueError): tracking_frame_inputs(*inputs)

    def test_guarded_seventh_tracker_is_checked(self):
        inputs = self.fixture(motion=True, center=True, reid=True)
        for name in ("kalman_reid_active", "kalman_reid_guarded"):
            inputs[1]["tracking"][name] = {"idf1": None}
            for frame in inputs[2]: frame[name] = []
        self.assertEqual(len(tracking_frame_inputs(*inputs)), 7)
        del inputs[2][0]["kalman_reid_guarded"]
        with self.assertRaises(ValueError): tracking_frame_inputs(*inputs)

    def test_reid_optional_and_five_tracker_report_unchanged(self):
        for motion, center in ((False, False), (True, True)):
            inputs = self.fixture(motion=motion, center=center, reid=True)
            before = deepcopy(inputs)
            expected = ("iou", "two_stage", "kalman", "kalman_center", "kalman_reid") if motion else (
                "iou", "two_stage", "kalman_reid")
            self.assertEqual(tracking_frame_inputs(*inputs), expected)
            self.assertEqual(inputs, before)

    def test_reid_report_and_each_frame_must_agree(self):
        for side in ("report", "first", "last"):
            inputs = self.fixture(motion=True, center=True, reid=True)
            if side == "report":
                del inputs[1]["tracking"]["kalman_reid"]
            else:
                del inputs[2][0 if side == "first" else 1]["kalman_reid"]
            with self.assertRaisesRegex(ValueError, "Tracker set differs"):
                tracking_frame_inputs(*inputs)

    def test_reid_objects_and_names_are_strictly_validated(self):
        inputs = self.fixture(reid=True)
        inputs[2][0]["kalman_reid"] = [{"id": -1, "bbox": [1, 2, 3, 4]}]
        with self.assertRaises(ValueError):
            tracking_frame_inputs(*inputs)
        inputs = self.fixture(reid=True)
        inputs[1]["tracking"]["kalman-reid"] = inputs[1]["tracking"].pop("kalman_reid")
        with self.assertRaisesRegex(ValueError, "Unknown tracking report keys"):
            tracking_frame_inputs(*inputs)

    def test_three_tracker_report_and_raw_detection_provenance(self):
        inputs = self.fixture(motion=True)
        inputs[2][0]["raw_detections"] = [{"bbox": [1, 2, 3, 4], "label": "person", "score": .5}]
        self.assertEqual(tracking_frame_inputs(*inputs), ("iou", "two_stage", "kalman"))
        # Raw detections do not become reference predictions or mutate them.
        self.assertEqual(inputs[2][0]["kalman"], [])

    def test_center_tracker_is_optional_independently_of_legacy_kalman(self):
        for motion in (False, True):
            with self.subTest(legacy_kalman=motion):
                inputs = self.fixture(motion=motion, center=True)
                inputs[1]["tracking"]["parameters"]["kalman_center"] = {
                    "association": "active-high, active-low, lost-high"}
                before = deepcopy(inputs)
                expected = (("iou", "two_stage", "kalman", "kalman_center")
                            if motion else ("iou", "two_stage", "kalman_center"))
                self.assertEqual(tracking_frame_inputs(*inputs), expected)
                self.assertEqual(inputs, before)

    def test_center_tracker_report_and_every_frame_must_agree(self):
        for motion in (False, True):
            for change in ("missing_report", "missing_first_frame", "missing_last_frame",
                           "extra_frame", "extra_report"):
                with self.subTest(legacy_kalman=motion, change=change):
                    inputs = self.fixture(motion=motion, center=True)
                    if change == "missing_report":
                        del inputs[1]["tracking"]["kalman_center"]
                    elif change == "missing_first_frame":
                        del inputs[2][0]["kalman_center"]
                    elif change == "missing_last_frame":
                        del inputs[2][1]["kalman_center"]
                    elif change == "extra_frame":
                        inputs = self.fixture(motion=motion)
                        inputs[2][1]["kalman_center"] = []
                    else:
                        inputs = self.fixture(motion=motion)
                        inputs[1]["tracking"]["kalman_center"] = {}
                    with self.assertRaisesRegex(ValueError, "Tracker set differs"):
                        tracking_frame_inputs(*inputs)

    def test_center_tracker_misnaming_is_not_silently_ignored(self):
        for key in ("kalman_centers", "center_kalman", "kalman-center"):
            for side in ("report", "frame"):
                with self.subTest(key=key, side=side):
                    inputs = self.fixture(motion=True, center=True)
                    if side == "report":
                        inputs[1]["tracking"][key] = inputs[1]["tracking"].pop("kalman_center")
                        error = "Unknown tracking report keys"
                    else:
                        inputs[2][0][key] = inputs[2][0].pop("kalman_center")
                        error = "Unknown frame 1 keys"
                    with self.assertRaisesRegex(ValueError, error):
                        tracking_frame_inputs(*inputs)

    def test_center_tracker_cannot_hide_in_metadata_or_bypass_object_validation(self):
        inputs = self.fixture(center=True)
        del inputs[1]["tracking"]["kalman_center"]
        inputs[1]["tracking"]["parameters"]["kalman_center"] = {}
        with self.assertRaisesRegex(ValueError, "Tracker set differs"):
            tracking_frame_inputs(*inputs)
        inputs = self.fixture(center=True)
        inputs[1]["tracking"]["kalman_center"] = []
        with self.assertRaisesRegex(ValueError, "metric report objects"):
            tracking_frame_inputs(*inputs)
        inputs = self.fixture(center=True)
        entry = {"id": 9, "bbox": [1, 2, 3, 4]}
        inputs[2][1]["kalman_center"] = [entry, entry]
        with self.assertRaisesRegex(ValueError, "Duplicate per-frame"):
            tracking_frame_inputs(*inputs)

    def test_report_and_frame_tracker_presence_must_agree(self):
        inputs = self.fixture()
        inputs[2][0]["kalman"] = []
        with self.assertRaisesRegex(ValueError, "Tracker set differs"):
            tracking_frame_inputs(*inputs)
        inputs = self.fixture(motion=True)
        del inputs[2][1]["kalman"]
        with self.assertRaisesRegex(ValueError, "Tracker set differs"):
            tracking_frame_inputs(*inputs)

    def test_required_baselines_cannot_be_omitted(self):
        inputs = self.fixture(motion=True)
        del inputs[1]["tracking"]["iou"]
        with self.assertRaisesRegex(ValueError, "must contain iou and two_stage"):
            tracking_frame_inputs(*inputs)

    def test_unknown_report_or_frame_tracker_is_not_silently_ignored(self):
        inputs = self.fixture()
        inputs[1]["tracking"]["unknown_tracker"] = {}
        with self.assertRaisesRegex(ValueError, "Unknown tracking report keys"):
            tracking_frame_inputs(*inputs)
        inputs = self.fixture()
        inputs[2][0]["unknown_tracker"] = []
        with self.assertRaisesRegex(ValueError, "Unknown frame 1 keys"):
            tracking_frame_inputs(*inputs)

    def test_report_trackers_must_be_metric_objects(self):
        inputs = self.fixture()
        inputs[1]["tracking"]["two_stage"] = []
        with self.assertRaisesRegex(ValueError, "metric report objects"):
            tracking_frame_inputs(*inputs)

    def test_all_frames_are_required_and_contiguous(self):
        inputs = self.fixture()
        inputs[2].pop()
        with self.assertRaisesRegex(ValueError, "every frame"):
            tracking_frame_inputs(*inputs)
        inputs = self.fixture()
        inputs[2][1]["frame_index"] = 3
        with self.assertRaisesRegex(ValueError, "contiguous"):
            tracking_frame_inputs(*inputs)
        inputs = self.fixture()
        inputs[2][0]["frame_index"] = True
        with self.assertRaisesRegex(ValueError, "integer"):
            tracking_frame_inputs(*inputs)

    def test_source_gt_requires_exact_float32_widening(self):
        inputs = self.fixture()
        inputs[2][0]["ground_truth"][0]["bbox"][2] += 1e-10
        with self.assertRaisesRegex(ValueError, "GT box differs"):
            tracking_frame_inputs(*inputs)
        inputs = self.fixture()
        inputs[2][0]["ground_truth"][0]["id"] = 6
        with self.assertRaisesRegex(ValueError, "GT identities differ"):
            tracking_frame_inputs(*inputs)

    def test_duplicate_manifest_or_tracker_ids_are_rejected(self):
        inputs = self.fixture()
        inputs[0]["ground_truth"].append(deepcopy(inputs[0]["ground_truth"][0]))
        with self.assertRaisesRegex(ValueError, "Duplicate per-frame"):
            tracking_frame_inputs(*inputs)
        inputs = self.fixture(motion=True)
        entry = {"id": 9, "bbox": [1, 2, 3, 4]}
        inputs[2][1]["kalman"] = [entry, entry]
        with self.assertRaisesRegex(ValueError, "Duplicate per-frame"):
            tracking_frame_inputs(*inputs)

    def test_bounds_and_single_class_protocol(self):
        inputs = self.fixture()
        inputs[0]["classes"] = ["person", "car"]
        with self.assertRaisesRegex(ValueError, "single-class"):
            tracking_frame_inputs(*inputs)
        inputs = self.fixture()
        inputs[2][0]["raw_detections"] = [{}] * 501
        with self.assertRaisesRegex(ValueError, "Invalid raw detections"):
            tracking_frame_inputs(*inputs)
        inputs = self.fixture()
        inputs[0]["ground_truth"][0]["frame_index"] = 3
        with self.assertRaisesRegex(ValueError, "exceeds sequence"):
            tracking_frame_inputs(*inputs)
        with self.assertRaisesRegex(ValueError, "Invalid object list"):
            checked_objects([{}] * 501, "fixture")

    def test_object_boxes_and_numbers_are_strict(self):
        checked_objects([{"id": 0, "bbox": [-3, -2, 1, 4]}], "negative coordinates are valid")
        for box in ([1, 2, 1, 4], [1, 2, 3], [1, 2, float("nan"), 4], [1, 2, True, 4]):
            with self.subTest(box=box), self.assertRaises(ValueError):
                checked_objects([{"id": 0, "bbox": box}], "fixture")
        for value in (True, 1.0, -1):
            with self.subTest(value=value), self.assertRaises(ValueError):
                integer(value, "fixture")
        for value in (True, float("inf"), "1"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                finite(value, "fixture")

    def test_metric_comparison_counts_nulls_and_tolerance(self):
        self.assertEqual(compare_count(3, 3, "count"), {"cpp": 3, "reference": 3})
        self.assertIsNone(compare_number(None, None, "empty GT")["cpp"])
        self.assertLess(compare_number(.5 + 5e-7, .5, "score")["absolute_error"], 1e-6)
        for actual, expected in ((None, .5), (.5, None), (.501, .5), (float("nan"), .5)):
            with self.subTest(actual=actual), self.assertRaises(ValueError):
                compare_number(actual, expected, "score")
        with self.assertRaises(ValueError):
            compare_count(2, 3, "count")

    def test_manifest_paths_are_local_existing_files(self):
        manifest = self.base / "manifest.json"
        existing = self.base / "video.mp4"
        existing.write_bytes(b"fixture")
        self.assertEqual(manifest_file(manifest, "video.mp4"), existing.resolve())
        for name in ("../outside.mp4", str(existing.resolve()), "missing.mp4"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                manifest_file(manifest, name)

    def test_json_rejects_nonfinite_and_oversized_input(self):
        source = self.base / "source.json"
        source.write_text('{"value": NaN}', encoding="utf-8")
        with self.assertRaises(ValueError):
            read_json(source)
        source.write_text("{}", encoding="utf-8")
        with self.assertRaises(ValueError):
            read_json(source, limit=1)

    def test_report_must_describe_exact_manifest_and_dataset(self):
        manifest_path = self.base / "manifest.json"
        manifest = {"version": 1, "kind": "video", "dataset": "fixture"}
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        report = {"version": 1, "kind": "video", "dataset": "fixture",
            "manifest_sha256": digest, "data_provenance": manifest}
        (self.base / "report.json").write_text(json.dumps(report), encoding="utf-8")
        self.assertEqual(base_inputs(manifest_path, self.base, "video")[0], manifest)
        report["dataset"] = "different"
        (self.base / "report.json").write_text(json.dumps(report), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "dataset differs"):
            base_inputs(manifest_path, self.base, "video")
        report["dataset"] = "fixture"
        report["manifest_sha256"] = "0" * 64
        (self.base / "report.json").write_text(json.dumps(report), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "exact manifest"):
            base_inputs(manifest_path, self.base, "video")

    def test_changed_input_does_not_emit_or_replace_success(self):
        source = self.base / "input.json"
        source.write_bytes(b"first")
        hashes = {str(source): hashlib.sha256(source.read_bytes()).hexdigest()}
        success = self.base / "reference-validation.json"
        success.write_bytes(b"prior successful run")
        source.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "Input changed"):
            write_success(self.base, {"checks": {}}, hashes)
        self.assertEqual(success.read_bytes(), b"prior successful run")


if __name__ == "__main__":
    unittest.main()
