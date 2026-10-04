"""Network-free model preparation checks; ML dependencies are optional."""

import hashlib
import importlib.util
import io
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from export_reid import (ARCHITECTURE_REVISION, ARCHITECTURE_SHA256, CHECKPOINT_FILENAME,
    CHECKPOINT_SHA256, CHECKPOINT_SIZE, HF_REVISION, architecture_tree, checkpoint_state,
    download_pinned, load_reid_weights, manifest, normalized_feature, preprocessing, restricted_checkpoint,
    verify_file, write_json)


class Tensor:
    def __init__(self, shape, dtype="float32", finite=True, device="cpu"):
        self.shape = shape
        self.dtype = dtype
        self.device = SimpleNamespace(type=device)
        self.finite = finite

    def numel(self):
        return math.prod(self.shape)

    def is_floating_point(self):
        return self.dtype.startswith("float")


class Torch:
    @staticmethod
    def is_tensor(value):
        return isinstance(value, Tensor)

    @staticmethod
    def isfinite(value):
        return SimpleNamespace(all=lambda: value.finite)


class Model:
    def __init__(self):
        self.expected = {
            "conv.weight": Tensor((16, 3, 7, 7)),
            "bn.num_batches_tracked": Tensor((), "int64"),
            "classifier.weight": Tensor((1, 512)),
            "classifier.bias": Tensor((1,)),
        }
        self.loaded = None

    def state_dict(self):
        return self.expected

    def load_state_dict(self, value, strict):
        if strict is not True:
            raise ValueError("Strict state load required")
        self.loaded = value


def checkpoint():
    return {
        "conv.weight": Tensor((16, 3, 7, 7)),
        "bn.num_batches_tracked": Tensor((), "int64"),
        "classifier.weight": Tensor((4101, 512)),
        "classifier.bias": Tensor((4101,)),
    }


class Response(io.BytesIO):
    def __init__(self, data, url="https://fixed.example/model", length=None):
        super().__init__(data)
        self.url = url
        self.headers = {} if length is None else {"Content-Length": str(length)}


class ExportProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)

    def test_pins_select_person_reid_not_imagenet_checkpoint(self):
        self.assertIn("_msmt17_combineall_", CHECKPOINT_FILENAME)
        self.assertNotIn("imagenet", CHECKPOINT_FILENAME)
        self.assertEqual(CHECKPOINT_SIZE, 9336983)
        for revision in (HF_REVISION, ARCHITECTURE_REVISION):
            self.assertRegex(revision, r"^[0-9a-f]{40}$")
        for digest in (CHECKPOINT_SHA256, ARCHITECTURE_SHA256):
            self.assertRegex(digest, r"^[0-9a-f]{64}$")

    def test_unpinned_architecture_is_not_executable(self):
        with self.assertRaisesRegex(ValueError, "pinned SHA256"):
            architecture_tree("import os\nos.system('untrusted')")

    def test_checkpoint_loader_is_restricted_and_never_retries_unsafely(self):
        path = self.base / "pinned.pth"
        value = object()
        torch_module = SimpleNamespace(load=Mock(return_value=value))
        self.assertIs(restricted_checkpoint(path, torch_module), value)
        torch_module.load.assert_called_once_with(path, weights_only=True, map_location="cpu")
        for error in (RuntimeError("untrusted pickle rejected"), TypeError("old unsupported torch")):
            with self.subTest(error=type(error)):
                torch_module = SimpleNamespace(load=Mock(side_effect=error))
                with self.assertRaises(type(error)):
                    restricted_checkpoint(path, torch_module)
                torch_module.load.assert_called_once_with(path, weights_only=True, map_location="cpu")

    def test_checkpoint_prefix_normalization_and_wrapper(self):
        state = checkpoint()
        wrapped = {"state_dict": {"module." + key: value for key, value in state.items()}, "epoch": 150}
        self.assertEqual(checkpoint_state(wrapped, Torch), state)
        state["module.conv.weight"] = state["conv.weight"]
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            checkpoint_state(state, Torch)

    def test_only_known_training_classifier_head_is_ignored(self):
        model, source = Model(), checkpoint()
        self.assertEqual(load_reid_weights(model, source, Torch), 2)
        self.assertIs(model.loaded["conv.weight"], source["conv.weight"])
        self.assertIs(model.loaded["classifier.weight"], model.expected["classifier.weight"])
        self.assertIs(model.loaded["classifier.bias"], model.expected["classifier.bias"])

    def test_missing_unexpected_or_misshaped_backbone_fails(self):
        for case in ("missing", "unexpected", "shape", "dtype"):
            with self.subTest(case=case):
                source = checkpoint()
                if case == "missing":
                    del source["conv.weight"]
                elif case == "unexpected":
                    source["classifier.unapproved"] = Tensor((1,))
                elif case == "shape":
                    source["conv.weight"] = Tensor((8, 3, 7, 7))
                else:
                    source["conv.weight"] = Tensor((16, 3, 7, 7), "float16")
                model = Model()
                with self.assertRaises(ValueError):
                    load_reid_weights(model, source, Torch)
                self.assertIsNone(model.loaded)

    def test_checkpoint_requires_the_pinned_training_head(self):
        for case in ("missing", "imagenet", "wrong_bias"):
            with self.subTest(case=case):
                source = checkpoint()
                if case == "missing":
                    del source["classifier.weight"]
                elif case == "imagenet":
                    source["classifier.weight"] = Tensor((1000, 512))
                else:
                    source["classifier.bias"] = Tensor((1,))
                with self.assertRaises(ValueError):
                    load_reid_weights(Model(), source, Torch)

    def test_checkpoint_state_is_cpu_finite_bounded_tensors_only(self):
        for value in (False, Tensor((1,), finite=False), Tensor((1,), device="cuda"), Tensor((3_000_001,))):
            with self.subTest(value=value), self.assertRaises(ValueError):
                checkpoint_state({"weight": value}, Torch)
        for source in ([], {}, {"state_dict": []}, {"": Tensor((1,))}):
            with self.subTest(source=source), self.assertRaises(ValueError):
                checkpoint_state(source, Torch)

    def test_download_verified_by_size_and_digest_before_reuse(self):
        payload = b"verified model"
        digest = hashlib.sha256(payload).hexdigest()
        target = self.base / "model.pth"
        with patch("export_reid.urllib.request.urlopen", return_value=Response(payload, length=len(payload))) as fetch:
            download_pinned("https://fixed.example/model", target, len(payload), digest)
            fetch.assert_called_once()
        with patch("export_reid.urllib.request.urlopen") as fetch:
            download_pinned("https://fixed.example/model", target, len(payload), digest)
            fetch.assert_not_called()
        self.assertEqual(target.read_bytes(), payload)
        with self.assertRaisesRegex(ValueError, "SHA256"):
            verify_file(target, len(payload), "0" * 64)
        with self.assertRaisesRegex(ValueError, "sized"):
            verify_file(target, len(payload) + 1, digest)

    def test_download_short_extra_corrupt_and_downgrade_are_rejected(self):
        digest = hashlib.sha256(b"1234").hexdigest()
        for name, response in (("short", Response(b"123")), ("extra", Response(b"12345")),
                               ("corrupt", Response(b"4321")),
                               ("downgrade", Response(b"1234", url="http://fixed.example/model")),
                               ("length", Response(b"1234", length=5))):
            with self.subTest(case=name), patch("export_reid.urllib.request.urlopen", return_value=response):
                with self.assertRaises(ValueError):
                    download_pinned("https://fixed.example/model", self.base / name, 4, digest)
        with patch("export_reid.urllib.request.urlopen") as fetch:
            for url, size in (("http://fixed.example/model", 4), ("https://fixed.example/model", 20_000_000)):
                with self.assertRaises(ValueError):
                    download_pinned(url, self.base / "invalid", size, digest)
            fetch.assert_not_called()

    def test_manifest_contract_and_exclusive_json_publication(self):
        model = self.base / "model.onnx"
        model.write_bytes(b"fixture")
        data = manifest(model)
        self.assertEqual(len(data), 17)
        self.assertEqual(len(data["source"]), 7)
        self.assertEqual(data["architecture"], "osnet_x0_25")
        self.assertEqual(data["training_dataset"], "MSMT17-combineall")
        self.assertEqual(data["input_shape"], [1, 3, 256, 128])
        self.assertEqual(data["output_shape"], [1, 512])
        self.assertEqual(data["input_name"], "images")
        self.assertEqual(data["output_name"], "features")
        self.assertTrue(data["normalize_output"])
        self.assertEqual(data["model_size_bytes"], 7)
        result = self.base / "manifest.json"
        write_json(result, data)
        before = result.read_bytes()
        with self.assertRaises(FileExistsError):
            write_json(result, data)
        self.assertEqual(result.read_bytes(), before)


@unittest.skipUnless(importlib.util.find_spec("numpy") and importlib.util.find_spec("cv2"),
                     "OpenCV/numpy preparation dependencies unavailable")
class PreprocessingTests(unittest.TestCase):
    def test_rgb_order_means_std_and_nchw_layout(self):
        import numpy as np
        bgr = np.zeros((3, 7, 3), dtype=np.uint8)
        bgr[:] = [10, 20, 240]
        blob = preprocessing(bgr)
        self.assertEqual(blob.shape, (1, 3, 256, 128))
        self.assertEqual(blob.dtype, np.float32)
        self.assertTrue(blob.flags.c_contiguous)
        expected = (np.array([240, 20, 10], np.float32) / np.float32(255)
                    - np.array([.485, .456, .406], np.float32)) / np.array([.229, .224, .225], np.float32)
        np.testing.assert_allclose(blob[0, :, 100, 42], expected, rtol=0, atol=1e-7)
        self.assertEqual(len(blob.astype("<f4").tobytes()), 393216)

    def test_invalid_crop_is_rejected(self):
        import numpy as np
        for crop in (None, np.zeros((0, 2, 3), np.uint8), np.zeros((2, 3), np.uint8),
                     np.zeros((2, 3, 4), np.uint8), np.zeros((2, 3, 3), np.float32),
                     np.zeros((1, 4097, 3), np.uint8)):
            with self.subTest(shape=getattr(crop, "shape", None)), self.assertRaises(ValueError):
                preprocessing(crop)

    def test_l2_output_is_finite_nonzero_and_exact_dimension(self):
        import numpy as np
        result = normalized_feature(np.ones((1, 512), np.float32))
        self.assertEqual(result.shape, (512,))
        self.assertAlmostEqual(float(np.linalg.norm(result)), 1.0, places=6)
        for raw in (np.zeros((1, 512)), np.ones((512,)), np.ones((1, 256)),
                    np.full((1, 512), np.nan), np.full((1, 512), np.inf)):
            with self.subTest(shape=raw.shape), self.assertRaises(ValueError):
                normalized_feature(raw)


if __name__ == "__main__":
    unittest.main()
