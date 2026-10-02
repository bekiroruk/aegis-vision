"""Preparation-only tests; no network, dataset, inference, or model required."""

import hashlib
import json
import io
import zipfile
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from prepare_quality_data import (bounded_bytes, download_fixed, local_path,
    parse_mot_gt, prepare_coco, prepare_mot, save_new_or_identical, xywh_box)


class QualityDataTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)

    def coco_fixture(self):
        (self.base / "images").mkdir()
        (self.base / "images/first.jpg").write_bytes(b"\xff\xd8fixture\xff\xd9")
        annotation = {"categories": [{"id": 17, "name": "cat"}, {"id": 18, "name": "dog"}],
            "images": [{"id": 12, "file_name": "first.jpg", "width": 100, "height": 80}],
            "annotations": [
                {"id": 9, "image_id": 12, "category_id": 17, "bbox": [2, 3, 20, 10], "iscrowd": 0},
                {"id": 10, "image_id": 12, "category_id": 17, "bbox": [-1, 1, 2, 2], "iscrowd": 0},
                {"id": 11, "image_id": 12, "category_id": 17, "bbox": [1, 1, 10, 10], "iscrowd": 1},
                {"id": 13, "image_id": 12, "category_id": 18, "bbox": [1, 1, 10, 10], "iscrowd": 0}]}
        raw = json.dumps(annotation).encode()
        annotations = self.base / "annotations.json"
        annotations.write_bytes(raw)
        crops = self.base / "crops.json"
        crops.write_text(json.dumps({"version": 1, "dataset": "fixture", "selection": "fixture",
            "annotation_sha256": hashlib.sha256(raw).hexdigest(),
            "items": [{"id": "coco-val2017-9", "image": "images/first.jpg", "label": "cat"}]}))
        return annotations, crops, self.base / "quality.json"

    def test_coco_full_selected_class_ground_truth_includes_small_and_crowd(self):
        paths = self.coco_fixture()
        manifest = prepare_coco(*paths)
        self.assertEqual(manifest["classes"], [{"label": "cat", "category_id": 17}])
        image = manifest["images"][0]
        self.assertEqual(image["coco_image_id"], 12)
        self.assertEqual(image["ground_truth"][0]["bbox"], [2, 3, 22, 13])
        self.assertEqual(image["ground_truth"][1]["bbox"], [-1, 1, 1, 3])
        self.assertEqual(len(image["ground_truth"]), 3)
        self.assertTrue(image["ground_truth"][2]["crowd"])
        self.assertEqual(image["sha256"], hashlib.sha256((self.base / image["path"]).read_bytes()).hexdigest())
        first = paths[2].read_bytes()
        self.assertEqual(prepare_coco(*paths), manifest)
        self.assertEqual(paths[2].read_bytes(), first)

    def test_coco_annotation_hash_mismatch(self):
        paths = self.coco_fixture()
        paths[0].write_bytes(paths[0].read_bytes() + b" ")
        with self.assertRaisesRegex(ValueError, "exact annotations"):
            prepare_coco(*paths)
        self.assertFalse(paths[2].exists())

    def test_coco_escaped_source_rejected(self):
        paths = self.coco_fixture()
        crops = json.loads(paths[1].read_bytes())
        crops["items"][0]["image"] = "../outside.jpg"
        paths[1].write_text(json.dumps(crops))
        with self.assertRaisesRegex(ValueError, "escaped"):
            prepare_coco(*paths)

    def test_coco_unknown_class_rejected(self):
        paths = self.coco_fixture()
        crops = json.loads(paths[1].read_bytes())
        crops["items"][0]["label"] = "unknown"
        paths[1].write_text(json.dumps(crops))
        with self.assertRaisesRegex(ValueError, "Unknown selected class"):
            prepare_coco(*paths)

    def test_mot15_flag_and_world_fields_are_not_classes(self):
        raw = (b"1,2,1,1,10,20,1,-1,-1,-1\n"
               b"1,3,5,5,10,10,0,1,0.8,0\n"
               b"2,2,-3,2,10,20,1,999,0,100\n")
        gt, ignored = parse_mot_gt(raw, 2)
        self.assertEqual(ignored, 1)
        self.assertEqual(len(gt), 2)
        self.assertEqual(gt[0], {"frame_index": 1, "id": 2, "bbox": [0, 0, 10, 20]})
        self.assertEqual(gt[1]["bbox"], [-4, 1, 6, 21])

    def test_campus_sequence_uses_its_own_metadata_and_pins(self):
        video = b"campus-video-fixture"
        (self.base / "TUD-Campus-raw.mp4").write_bytes(video)
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("MOT15Labels/train/TUD-Campus/gt/gt.txt", "71,2,1,1,10,20,1,-1,-1,-1\n")
            archive.writestr("MOT15Labels/train/TUD-Campus/seqinfo.ini",
                "[Sequence]\nname=TUD-Campus\nseqLength=71\nimWidth=640\nimHeight=480\nframeRate=25\n")
        labels = buffer.getvalue()
        (self.base / "MOT15Labels.zip").write_bytes(labels)
        with patch("prepare_quality_data.MOT_CAMPUS_VIDEO_SIZE",len(video)), \
             patch("prepare_quality_data.MOT_CAMPUS_VIDEO_SHA256",hashlib.sha256(video).hexdigest()), \
             patch("prepare_quality_data.MOT_LABELS_SIZE",len(labels)), \
             patch("prepare_quality_data.MOT_LABELS_SHA256",hashlib.sha256(labels).hexdigest()):
            result = prepare_mot(self.base,sequence="TUD-Campus")
        self.assertEqual(result["frames"],71)
        self.assertEqual(result["ground_truth"][0]["frame_index"],71)
        self.assertEqual(result["video"],"TUD-Campus-raw.mp4")
        self.assertEqual(result["dataset"],"mot15-tud-campus-reencoded-raw-v1")

    def test_unknown_mot_sequence_rejected_before_filesystem_writes(self):
        output = self.base / "uncreated"
        with self.assertRaisesRegex(ValueError,"pinned"):
            prepare_mot(output,sequence="../unexpected")
        self.assertFalse(output.exists())

    def test_mot15_invalid_rows_rejected(self):
        for raw in (b"1,2,1,1,10,20,1,-1,-1\n", b"0,2,1,1,10,20,1,-1,-1,-1\n",
                    b"1,2.5,1,1,10,20,1,-1,-1,-1\n", b"1,2,1,1,0,20,1,-1,-1,-1\n",
                    b"1,2,nan,1,10,20,1,-1,-1,-1\n",
                    b"1,2,1,1,10,20,1,-1,-1,-1\n1,2,1,1,10,20,1,-1,-1,-1\n"):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                parse_mot_gt(raw, 1)

    def test_boxes_must_be_numeric_finite_and_non_degenerate(self):
        for values in ([0, 0, 0, 1], [0, 0, 1, float("inf")], [0, 0, True, 1], [1, 2], ["0", 0, 1, 1]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                xywh_box(values)

    def test_existing_file_preserved_on_conflict(self):
        path = self.base / "existing.txt"
        save_new_or_identical(path, b"first")
        save_new_or_identical(path, b"first")
        with self.assertRaises(ValueError):
            save_new_or_identical(path, b"second")
        self.assertEqual(path.read_bytes(), b"first")

    def test_file_and_relative_path_limits(self):
        path = self.base / "data.txt"
        path.write_bytes(b"1234")
        self.assertEqual(bounded_bytes(path, 4), b"1234")
        with self.assertRaises(ValueError):
            bounded_bytes(path, 3)
        with self.assertRaises(ValueError):
            local_path(self.base, str(path.resolve()))

    def test_download_cache_requires_pinned_size_and_preserves_conflict(self):
        path = self.base / "cached.bin"
        path.write_bytes(b"1234")
        with patch("prepare_quality_data.urlopen") as remote:
            download_fixed("https://example.invalid/data", path, 4)
            remote.assert_not_called()
            with self.assertRaises(ValueError):
                download_fixed("https://example.invalid/data", path, 5)
        self.assertEqual(path.read_bytes(), b"1234")

    def test_download_cache_sha256_mismatch_preserves_same_size_asset(self):
        path = self.base / "cached.bin"
        path.write_bytes(b"1234")
        with patch("prepare_quality_data.urlopen") as remote:
            download_fixed("https://example.invalid/data", path, 4, hashlib.sha256(b"1234").hexdigest())
            with self.assertRaisesRegex(ValueError, "SHA256"):
                download_fixed("https://example.invalid/data", path, 4, hashlib.sha256(b"4321").hexdigest())
            remote.assert_not_called()
        self.assertEqual(path.read_bytes(), b"1234")

    def test_download_pinned_checksum_verified_before_persist(self):
        path = self.base / "new.bin"
        with patch("prepare_quality_data.urlopen") as remote:
            response = remote.return_value.__enter__.return_value
            response.geturl.return_value = "https://example.invalid/data"
            response.headers.get.return_value = "4"
            response.read.return_value = b"1234"
            with self.assertRaisesRegex(ValueError, "SHA256"):
                download_fixed("https://example.invalid/data", path, 4, hashlib.sha256(b"4321").hexdigest())
            self.assertFalse(path.exists())
            download_fixed("https://example.invalid/data", path, 4, hashlib.sha256(b"1234").hexdigest())
        self.assertEqual(path.read_bytes(), b"1234")


if __name__ == "__main__":
    unittest.main()
