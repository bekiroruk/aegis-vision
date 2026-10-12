"""Offline contracts for the frozen segmentation training experiment (no GPU)."""
import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

import numpy as np
from PIL import Image
import prepare_segmentation_training as prep
from train_segmentation import audit_data


def names():
    return [f"PennPed{i:05d}" for i in range(1, 97)] + [f"FudanPed{i:05d}" for i in range(1, 75)]


class GeometryTests(unittest.TestCase):
    def test_frozen_split(self):
        split = prep.split_names(names())
        self.assertEqual(split, prep.split_names(list(reversed(names()))))
        self.assertEqual({k: len(v) for k, v in split.items()}, {"train": 80, "val": 16, "test": 74})
        self.assertTrue(all(n.startswith("Fudan") for n in split["test"]))
        self.assertEqual(len(set(sum(split.values(), []))), 170)

    def test_bad_split_names(self):
        for bad in (names()[:-1], names() + [names()[0]], ["../escape"] + names()[1:]):
            with self.assertRaises(ValueError):
                prep.split_names(bad)

    def test_rle_exact_roundtrip(self):
        for mask in (np.zeros((7, 9), bool), np.ones((7, 9), bool), np.eye(7, 9, dtype=bool)):
            encoded = prep.rle(mask)
            decoded = np.repeat(np.arange(len(encoded["counts"])) % 2, encoded["counts"])
            np.testing.assert_array_equal(decoded.reshape(mask.shape, order="F"), mask)

    def test_polygon_approximation_is_measured(self):
        mask = np.zeros((20, 20), np.uint8)
        mask[2:10, 2:10] = 1
        text, overlap = prep.polygon(mask)
        self.assertEqual(overlap, 1)
        self.assertEqual(text.split()[0], "0")
        self.assertTrue(all(0 <= float(v) <= 1 for v in text.split()[1:]))
        mask[4:6, 4:6] = 0  # exterior polygon fills this hole
        mask[15:18, 15:18] = 1  # smaller disconnected part omitted
        self.assertLess(prep.polygon(mask)[1], 1)
        with self.assertRaises(ValueError):
            prep.polygon(np.zeros((5, 5), np.uint8))


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.archive = self.root/"fixture.zip"
        self.categories = self.root/"categories.json"
        prep.save(self.categories, {"categories": [{"id": i+1, "name": "person" if i == 0 else f"class{i}"} for i in range(80)]})
        with zipfile.ZipFile(self.archive, "w") as z:
            for i, name in enumerate(names()):
                rgb = np.full((12, 10, 3), i, np.uint8)
                mask = np.zeros((12, 10), np.uint8)
                mask[2:10, 2:8] = 3  # noncontiguous instance ID, not RGB palette value
                for subdir, suffix, pixels in (("PNGImages", "", rgb), ("PedMasks", "_mask", mask)):
                    stream = io.BytesIO()
                    Image.fromarray(pixels).save(stream, format="PNG")
                    z.writestr(f"PennFudanPed/{subdir}/{name}{suffix}.png", stream.getvalue())
        self.data = self.root/"prepared"
        prep.prepare(self.archive, self.data, self.categories, prep.digest(self.archive))

    def manifest(self, split="train"):
        return json.loads((self.data/f"{split}-manifest.json").read_text())

    def rewrite(self, path, obj):
        path.write_text(json.dumps(obj), encoding="utf-8")

    def test_prepare_and_audit(self):
        self.assertEqual(set(audit_data(self.data)), {"train", "val", "test", "dataset_yaml"})
        annotations = json.loads((self.data/"annotations.json").read_text())
        self.assertEqual(len(annotations["annotations"]), 170)
        self.assertTrue(all(a["area"] == 48 for a in annotations["annotations"]))
        with self.assertRaises(FileExistsError):
            prep.prepare(self.archive, self.data, self.categories, prep.digest(self.archive))

    def test_wrong_archive_hash(self):
        with self.assertRaises(ValueError):
            prep.prepare(self.archive, self.root/"bad", self.categories, "0"*64)
        self.assertFalse((self.root/"bad").exists())

    def test_image_tampering(self):
        image = self.data/self.manifest()["images"][0]["path"]
        image.write_bytes(b"tampered")
        with self.assertRaises(ValueError):
            audit_data(self.data)

    def test_label_tampering(self):
        name = Path(self.manifest()["images"][0]["path"]).stem
        (self.data/"labels/train"/f"{name}.txt").write_text("0 0 0 1 1 0 1")
        with self.assertRaises(ValueError):
            audit_data(self.data)

    def test_unmanifested_image(self):
        (self.data/"images/train/extra.png").write_bytes(b"extra")
        with self.assertRaises(ValueError):
            audit_data(self.data)

    def test_annotation_tampering(self):
        with (self.data/"annotations.json").open("a") as f:
            f.write(" ")
        with self.assertRaises(ValueError):
            audit_data(self.data)

    def test_path_escape(self):
        manifest = self.manifest()
        name = Path(manifest["images"][0]["path"]).name
        manifest["images"][0]["path"] = f"../{name}"
        self.rewrite(self.data/"train-manifest.json", manifest)
        with self.assertRaises(ValueError):
            audit_data(self.data)

    def test_config_class_order(self):
        path = self.data/"dataset.yaml"
        config = json.loads(path.read_text())
        config["names"]["0"] = "car"
        self.rewrite(path, config)
        with self.assertRaises(ValueError):
            audit_data(self.data)

    def test_duplicate_identity(self):
        manifest = self.manifest()
        manifest["images"][0]["coco_image_id"] = manifest["images"][1]["coco_image_id"]
        self.rewrite(self.data/"train-manifest.json", manifest)
        with self.assertRaises(ValueError):
            audit_data(self.data)

    def test_changed_frozen_split(self):
        train, val = self.manifest(), self.manifest("val")
        train["images"][0], val["images"][0] = val["images"][0], train["images"][0]
        self.rewrite(self.data/"train-manifest.json", train)
        self.rewrite(self.data/"val-manifest.json", val)
        with self.assertRaises(ValueError):
            audit_data(self.data)


if __name__ == "__main__":
    unittest.main()
