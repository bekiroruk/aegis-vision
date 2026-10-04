#!/usr/bin/env python3
"""Export pinned, person-ReID-trained OSNet for native C++/OpenCV inference.

Development-only preparation. The complete Torchreid package is not installed or
executed. A SHA-pinned, inspected standalone MIT architecture supplies only the
network classes; its download/unrestricted checkpoint-loader code is excluded.
Run with the existing local torch/onnx/OpenCV preparation environment.
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import math
from pathlib import Path
import urllib.request


HF_REPOSITORY = "kaiyangzhou/osnet"
HF_REVISION = "a5c5cc037c24235cda3b21085b93ad77c9616224"
CHECKPOINT_FILENAME = (
    "osnet_x0_25_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_"
    "b64_fb10_softmax_labelsmooth_flip_jitter.pth"
)
CHECKPOINT_SHA256 = "cf55163d78fc44c62c82f85ab62d39f10438679b5abe8c698ae08cfa84aa6e18"
CHECKPOINT_SIZE = 9336983
ARCHITECTURE_REVISION = "f8cd150fdf77e8d9e1ed143b7f308c2c609ded50"
ARCHITECTURE_SHA256 = "c7c1c29187d6330f859c91da229271531920464c7011aec13842a086b2263cae"
ARCHITECTURE_SIZE = 17037
LICENSE_SHA256 = "3ac8ce2a83d170cb1c7c84152e0c1faca1f187794303514383960d2441716247"
LICENSE_SIZE = 1069
ARCHITECTURE_URL = (
    "https://raw.githubusercontent.com/KaiyangZhou/deep-person-reid/"
    f"{ARCHITECTURE_REVISION}/torchreid/models/osnet.py"
)
CHECKPOINT_URL = f"https://huggingface.co/{HF_REPOSITORY}/resolve/{HF_REVISION}/{CHECKPOINT_FILENAME}"
LICENSE_URL = (
    "https://raw.githubusercontent.com/KaiyangZhou/deep-person-reid/"
    f"{ARCHITECTURE_REVISION}/LICENSE"
)
MODEL_LIMIT = 32 * 1024 * 1024
INPUT_SHAPE = [1, 3, 256, 128]
OUTPUT_SHAPE = [1, 512]
MEAN = [.485, .456, .406]
STD = [.229, .224, .225]
CLASSIFIER_KEYS = frozenset(("classifier.weight", "classifier.bias"))
ARCHITECTURE_CLASSES = frozenset((
    "ConvLayer", "Conv1x1", "Conv1x1Linear", "Conv3x3", "LightConv3x3",
    "ChannelGate", "OSBlock", "OSNet",
))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(path: Path, expected_size: int, expected_sha256: str) -> None:
    if not path.is_file() or path.stat().st_size != expected_size:
        raise ValueError(f"Missing or incorrectly sized pinned file: {path}")
    if file_sha256(path) != expected_sha256:
        raise ValueError(f"Pinned SHA256 differs: {path}")


def download_pinned(url: str, output: Path, size: int, digest: str) -> None:
    """Bound every fixed-source download, then verify bytes before consumption."""
    if output.exists():
        verify_file(output, size, digest)
        return
    if not url.startswith("https://") or size <= 0 or size > 16 * 1024 * 1024:
        raise ValueError("Invalid fixed download URL or bound")
    request = urllib.request.Request(url, headers={"User-Agent": "AegisVision-ReID-preparation/1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        if not response.url.startswith("https://"):
            raise ValueError("Download redirect downgraded HTTPS")
        content_length = response.headers.get("Content-Length")
        if content_length is not None and int(content_length) != size:
            raise ValueError("Pinned download Content-Length differs")
        with output.open("xb") as stream:
            total = 0
            while True:
                chunk = response.read(min(65536, size - total + 1))
                if not chunk:
                    break
                total += len(chunk)
                if total > size:
                    raise ValueError("Pinned download exceeds exact byte limit")
                stream.write(chunk)
    verify_file(output, size, digest)


def architecture_tree(source: str) -> ast.Module:
    """Only inspected tensor-layer definitions; no package/loader/downloader."""
    if hashlib.sha256(source.encode("utf-8")).hexdigest() != ARCHITECTURE_SHA256:
        raise ValueError("Standalone architecture source must match pinned SHA256")
    original = ast.parse(source, filename="osnet-pinned.py")
    selected = []
    classes = set()
    factories = 0
    for node in original.body:
        if isinstance(node, ast.Import) and [alias.name for alias in node.names] == ["torch"]:
            selected.append(node)
        elif isinstance(node, ast.ImportFrom) and node.module in ("torch", "torch.nn"):
            selected.append(node)
        elif isinstance(node, ast.ClassDef) and node.name in ARCHITECTURE_CLASSES:
            selected.append(node)
            classes.add(node.name)
        elif isinstance(node, ast.FunctionDef) and node.name == "osnet_x0_25":
            factory = copy.deepcopy(node)
            # Calling the original factory with pretrained=False would already
            # avoid its loader, but remove that branch entirely as a second guard.
            factory.body = [item for item in factory.body if not (
                isinstance(item, ast.If) and isinstance(item.test, ast.Name)
                and item.test.id == "pretrained")]
            selected.append(factory)
            factories += 1
    if classes != ARCHITECTURE_CLASSES or factories != 1:
        raise ValueError("Pinned source has an unexpected architecture topology")
    tree = ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[]))
    text = ast.unparse(tree)
    if any(name in text for name in ("torch.load", "gdown", "init_pretrained_weights", "pretrained_urls")):
        raise ValueError("Unsafe loader/downloader leaked into architecture module")
    return tree


def checkpoint_state(checkpoint: object, torch_module: object) -> dict:
    """Permit only named CPU tensor state, with no silent dropped backbone keys."""
    if not isinstance(checkpoint, dict):
        raise ValueError("Expected tensor state dictionary checkpoint")
    state = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state, dict) or not 1 <= len(state) <= 2000:
        raise ValueError("Invalid or oversized checkpoint state dictionary")
    result = {}
    for key, value in state.items():
        if not isinstance(key, str) or not key or len(key) > 256:
            raise ValueError("Invalid checkpoint tensor key")
        name = key[7:] if key.startswith("module.") else key
        if name in result or not torch_module.is_tensor(value):
            raise ValueError("Duplicate normalized checkpoint key or non-tensor state")
        if value.device.type != "cpu" or value.numel() > 3_000_000:
            raise ValueError("Unexpected tensor device or allocation size")
        if value.is_floating_point() and not bool(torch_module.isfinite(value).all()):
            raise ValueError("Checkpoint contains a non-finite tensor")
        result[name] = value
    return result


def restricted_checkpoint(path: Path, torch_module: object) -> object:
    # Unsupported or rejected restricted loads are terminal. In particular,
    # never retry with weights_only=False or add unsafe globals to a safe list.
    return torch_module.load(path, weights_only=True, map_location="cpu")


def load_reid_weights(model: object, checkpoint: object, torch_module: object) -> int:
    state = checkpoint_state(checkpoint, torch_module)
    expected = model.state_dict()
    # The official MSMT17 combineall checkpoint classifies 4,101 identities.
    # Ignore exactly that known training head, never arbitrary unexpected keys.
    if not CLASSIFIER_KEYS.issubset(state):
        raise ValueError("Pinned ReID checkpoint is missing the known classifier head")
    if tuple(state["classifier.weight"].shape) != (4101, 512) or tuple(state["classifier.bias"].shape) != (4101,):
        raise ValueError("Checkpoint is not the expected MSMT17 combineall training head")
    actual_backbone = set(state) - CLASSIFIER_KEYS
    expected_backbone = set(expected) - CLASSIFIER_KEYS
    if actual_backbone != expected_backbone:
        raise ValueError(f"Backbone checkpoint keys differ: missing={sorted(expected_backbone-actual_backbone)}, "
                         f"unexpected={sorted(actual_backbone-expected_backbone)}")
    for name in expected_backbone:
        if state[name].shape != expected[name].shape or state[name].dtype != expected[name].dtype:
            raise ValueError(f"Backbone tensor shape/dtype differs: {name}")
    complete = dict(expected)
    complete.update({name: state[name] for name in expected_backbone})
    model.load_state_dict(complete, strict=True)
    return len(expected_backbone)


def preprocessing(bgr: object) -> object:
    import cv2
    import numpy as np
    if (not isinstance(bgr, np.ndarray) or bgr.dtype != np.uint8 or bgr.ndim != 3
            or bgr.shape[2] != 3 or min(bgr.shape[:2]) < 1 or max(bgr.shape[:2]) > 4096):
        raise ValueError("Expected bounded uint8 BGR crop")
    resized = cv2.resize(bgr, (128, 256), interpolation=cv2.INTER_LINEAR)
    rgb = resized[:, :, ::-1].astype(np.float32) / np.float32(255)
    normalized = ((rgb - np.array(MEAN, np.float32)) / np.array(STD, np.float32))
    return normalized.transpose(2, 0, 1).copy()[None]


def normalized_feature(raw: object) -> object:
    import numpy as np
    feature = np.asarray(raw, dtype=np.float32)
    if feature.shape != (1, 512) or not np.all(np.isfinite(feature)):
        raise ValueError("Expected finite raw OSNet feature tensor [1,512]")
    norm = np.sqrt(np.sum(feature.astype(np.float64) ** 2))
    if not math.isfinite(norm) or norm <= 1e-12:
        raise ValueError("OSNet feature has no finite positive norm")
    return (feature[0].astype(np.float64) / norm).astype(np.float32)


def manifest(model_path: Path) -> dict:
    size = model_path.stat().st_size
    if not 1 <= size <= MODEL_LIMIT:
        raise ValueError("Exported model exceeds native encoder limit")
    return {
        "version": 1, "architecture": "osnet_x0_25", "training_dataset": "MSMT17-combineall",
        "model_file": "model.onnx", "model_sha256": file_sha256(model_path), "model_size_bytes": size,
        "input_name": "images", "input_shape": INPUT_SHAPE, "output_name": "features",
        "output_shape": OUTPUT_SHAPE, "dimension": 512, "preprocessing": "opencv-linear-rgb-imagenet-v1",
        "mean": MEAN, "std": STD, "normalize_output": True, "license": "MIT",
        "source": {"hf_repository": HF_REPOSITORY, "hf_revision": HF_REVISION,
                   "checkpoint_filename": CHECKPOINT_FILENAME, "checkpoint_sha256": CHECKPOINT_SHA256,
                   "checkpoint_size_bytes": CHECKPOINT_SIZE, "architecture_revision": ARCHITECTURE_REVISION,
                   "source_sha256": ARCHITECTURE_SHA256},
    }


def reference_crops(video: Path, manifest_path: Path) -> list:
    import cv2
    import numpy as np
    if not manifest_path.is_file() or manifest_path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError("Missing or oversized MOT reference manifest")
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if data.get("version") != 1 or data.get("kind") != "video" or data.get("classes") != ["person"]:
        raise ValueError("Expected normalized single-person MOT reference manifest")
    if not video.is_file() or video.stat().st_size > 32 * 1024 * 1024 or file_sha256(video) != data["video_sha256"]:
        raise ValueError("Reference video differs from pinned dataset manifest")
    capture = cv2.VideoCapture(str(video))
    try:
        success, image = capture.read()
    finally:
        capture.release()
    if not success or image is None or list(image.shape[:2]) != [data["height"], data["width"]]:
        raise ValueError("Reference frame decoding/dimensions failed")
    rows = sorted((row for row in data["ground_truth"] if row["frame_index"] == 1), key=lambda row: row["id"])
    if len(rows) < 2:
        raise ValueError("Need two real first-frame person crops for golden fixtures")
    result = []
    for row in rows[:2]:
        values = row["bbox"]
        if len(values) != 4 or any(not math.isfinite(value) for value in values):
            raise ValueError("Invalid reference person box")
        x1, y1 = max(0, math.floor(values[0])), max(0, math.floor(values[1]))
        x2, y2 = min(image.shape[1], math.ceil(values[2])), min(image.shape[0], math.ceil(values[3]))
        if x2 <= x1 or y2 <= y1:
            raise ValueError("Reference person box has no visible crop")
        result.append(np.ascontiguousarray(image[y1:y2, x1:x2]))
    return result


def export_bundle(output: Path, video: Path, reference_manifest: Path) -> None:
    import cv2
    import numpy as np
    import onnx
    import torch
    if output.exists():
        raise ValueError("Output already exists; choose a new bundle directory")
    output.mkdir(parents=True)
    download_pinned(ARCHITECTURE_URL, output / "osnet-source.py", ARCHITECTURE_SIZE, ARCHITECTURE_SHA256)
    download_pinned(LICENSE_URL, output / "LICENSE-MIT.txt", LICENSE_SIZE, LICENSE_SHA256)
    checkpoint_path = output / CHECKPOINT_FILENAME
    download_pinned(CHECKPOINT_URL, checkpoint_path, CHECKPOINT_SIZE, CHECKPOINT_SHA256)
    source = (output / "osnet-source.py").read_text(encoding="utf-8")
    tree = architecture_tree(source)
    namespace = {"__name__": "aegisvision_pinned_osnet"}
    exec(compile(tree, "osnet-pinned.py", "exec"), namespace)
    torch.set_num_threads(2)
    torch.manual_seed(0)
    model = namespace["osnet_x0_25"](num_classes=1, pretrained=False).eval()
    # weights_only is mandatory; never retry using unrestricted pickle.
    checkpoint = restricted_checkpoint(checkpoint_path, torch)
    loaded = load_reid_weights(model, checkpoint, torch)
    model_path = output / "model.onnx"
    with torch.no_grad():
        torch.onnx.export(model, (torch.zeros(INPUT_SHAPE),), model_path,
                          input_names=["images"], output_names=["features"], opset_version=13,
                          dynamo=False, do_constant_folding=True, export_params=True)
    graph = onnx.load(model_path)
    onnx.checker.check_model(graph)
    if ([value.name for value in graph.graph.input] != ["images"]
            or [value.name for value in graph.graph.output] != ["features"]
            or any(tensor.data_location == onnx.TensorProto.EXTERNAL for tensor in graph.graph.initializer)):
        raise ValueError("Unexpected exported ONNX tensor names or external weights")
    def shape(value):
        return [dimension.dim_value for dimension in value.type.tensor_type.shape.dim]
    if shape(graph.graph.input[0]) != INPUT_SHAPE or shape(graph.graph.output[0]) != OUTPUT_SHAPE:
        raise ValueError("ONNX does not have the required static input/output shape")
    net = cv2.dnn.readNetFromONNX(str(model_path))
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    cv2.setNumThreads(1)
    # Asymmetric RGB/BGR regions and a non-target-size image catch channel order,
    # normalization, resize and layout mistakes in the later C++ golden test.
    yy, xx = np.indices((193, 89))
    synthetic = np.stack(((xx * 3 + yy) % 256, (yy * 5) % 256, (xx * 7 + yy * 2) % 256), axis=-1).astype(np.uint8)
    crops = [synthetic, *reference_crops(video, reference_manifest)]
    fixtures = []
    errors = []
    with torch.no_grad():
        for index, image in enumerate(crops):
            name = f"reference-{index}.png"
            success, encoded = cv2.imencode(".png", image)
            if not success:
                raise ValueError("PNG fixture encoding failed")
            (output / name).write_bytes(encoded.tobytes())
            blob = preprocessing(image)
            blob_name = f"input-{index}.f32"
            (output / blob_name).write_bytes(blob.astype("<f4").tobytes())
            pytorch = normalized_feature(model(torch.from_numpy(blob)).detach().cpu().numpy())
            net.setInput(blob, "images")
            opencv = normalized_feature(net.forward("features"))
            maximum = float(np.max(np.abs(pytorch - opencv)))
            cosine = float(np.dot(pytorch.astype(np.float64), opencv.astype(np.float64)))
            if maximum > 1e-4 or abs(1.0 - cosine) > 1e-5:
                raise ValueError(f"PyTorch/OpenCV reference mismatch: max={maximum}, cosine={cosine}")
            fixtures.append({"image_file": name, "input_blob_file": blob_name,
                             "pytorch_embedding": pytorch.tolist(), "opencv_embedding": opencv.tolist(),
                             "image_sha256": file_sha256(output / name),
                             "input_blob_sha256": file_sha256(output / blob_name)})
            errors.append({"fixture": index, "max_abs_error": maximum, "cosine": cosine})
    write_json(output / "reference.json", {"version": 1, "fixtures": fixtures})
    write_json(output / "export-verification.json", {
        "version": 1, "status": "passed", "torch": torch.__version__, "opencv": cv2.__version__,
        "numpy": np.__version__, "onnx": onnx.__version__, "opset": 13,
        "loaded_backbone_tensors": loaded, "checkpoint_loader": "torch.load(weights_only=True,map_location=cpu)",
        "architecture_selection": "pinned standalone tensor classes and x0.25 factory; no package/loaders",
        "model_sha256": file_sha256(model_path), "fixtures": errors,
        "reference_video_sha256": file_sha256(video), "reference_manifest_sha256": file_sha256(reference_manifest),
        "reference_crop_policy": "first two distinct GT identities at frame 1; floor/ceil and clip to image",
        "limitations": ["Fixtures verify implementation parity, not ReID accuracy or generalization.",
                        "OpenCV INTER_LINEAR preprocessing is this project contract; not claimed bit-identical to upstream PIL transforms.",
                        "MSMT17 combineall checkpoint is cross-domain on MOT15; no MOT fine-tuning or threshold calibration."],
    })
    # Publish the successful runtime contract last; failed partial bundles have
    # no manifest and cannot be mistaken for verified native encoder artifacts.
    write_json(output / "manifest.json", manifest(model_path))
    print(f"Prepared {output}; static FP32 OSNet, 3 golden inputs and PyTorch/OpenCV parity passed.")
    print(f"ONNX SHA256 {file_sha256(model_path)}; {model_path.stat().st_size} bytes")


def write_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--reference-video", required=True, type=Path)
    parser.add_argument("--reference-manifest", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        export_bundle(args.output, args.reference_video, args.reference_manifest)
    except (OSError, ValueError, RuntimeError, ImportError, KeyError, TypeError) as error:
        parser.exit(1, f"ReID preparation failed: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
