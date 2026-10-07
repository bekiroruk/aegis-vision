#!/usr/bin/env python3
"""Offline development check against official COCOeval and TrackEval metrics.

The production benchmark remains C++; this script does not run a model or tune
parameters. Install reference dependencies separately, then validate a completed
result directory. No network requests or edits to input data are performed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
from typing import Any


TOLERANCE = 1e-6
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCE_ROOT = PROJECT_ROOT / "artifacts/deps/quality-reference"
TRACKERS = ("iou", "two_stage", "kalman", "kalman_center", "kalman_reid", "kalman_reid_active", "kalman_reid_guarded")
REQUIRED_TRACKERS = frozenset(("iou", "two_stage"))
TRACKING_METADATA = frozenset(("parameters", "protocol"))
FRAME_METADATA = frozenset(("frame_index", "ground_truth", "raw_detections"))


def reject_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON constant: {value}")


def read_json(path: Path, limit: int = 32 * 1024 * 1024) -> Any:
    if not path.is_file() or path.stat().st_size > limit:
        raise ValueError(f"Missing or oversized input: {path}")
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream, parse_constant=reject_constant)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def integer(value: Any, context: str, low: int = 0) -> int:
    if type(value) is not int or value < low:
        raise ValueError(f"Expected integer >= {low}: {context}")
    return value


def finite(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"Expected finite number: {context}")
    return float(value)


def compare_number(actual: Any, expected: float | None, context: str) -> dict[str, Any]:
    if expected is None:
        if actual is not None:
            raise ValueError(f"{context}: undefined score must be null, got {actual}")
        return {"cpp": None, "reference": None, "absolute_error": None}
    actual_value = finite(actual, context)
    error = abs(actual_value - expected)
    if error > TOLERANCE:
        raise ValueError(f"{context}: C++={actual_value:.12g}, reference={expected:.12g}, error={error:.12g}")
    return {"cpp": actual_value, "reference": expected, "absolute_error": error}


def compare_count(actual: Any, expected: Any, context: str) -> dict[str, int]:
    value = integer(actual, context)
    expected_value = int(expected)
    if value != expected_value:
        raise ValueError(f"{context}: C++={value}, reference={expected_value}")
    return {"cpp": value, "reference": expected_value}


def base_inputs(manifest_path: Path, result_dir: Path, kind: str) -> tuple[dict, dict, dict[str, str]]:
    manifest = read_json(manifest_path, 16 * 1024 * 1024)
    report_path = result_dir / "report.json"
    report = read_json(report_path)
    if manifest.get("version") != 1 or report.get("version") != 1:
        raise ValueError("Expected manifest and report version 1")
    if manifest.get("kind") != kind or report.get("kind") != kind:
        raise ValueError(f"Expected {kind} manifest and report")
    digest = sha256(manifest_path)
    if report.get("manifest_sha256") != digest or report.get("data_provenance") != manifest:
        raise ValueError("Report does not describe this exact manifest")
    if report.get("dataset") != manifest.get("dataset"):
        raise ValueError("Report dataset differs from manifest")
    hashes = {str(manifest_path.resolve()): digest, str(report_path.resolve()): sha256(report_path)}
    return manifest, report, hashes


def coco_validation(manifest_path: Path, result_dir: Path, annotations_path: Path, reference_root: Path) -> tuple[dict, dict]:
    import numpy as np
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    manifest, report, hashes = base_inputs(manifest_path, result_dir, "images")
    annotation_digest = sha256(annotations_path)
    if annotation_digest != manifest.get("annotation_sha256"):
        raise ValueError("Original COCO annotation SHA256 differs from manifest")
    hashes[str(annotations_path.resolve())] = annotation_digest
    original = read_json(annotations_path, 256 * 1024 * 1024)
    predictions_path = result_dir / "predictions.json"
    predictions = read_json(predictions_path)
    hashes[str(predictions_path.resolve())] = sha256(predictions_path)
    if not isinstance(predictions, list) or len(predictions) > 1000000:
        raise ValueError("Expected bounded COCO prediction list")
    classes = manifest["classes"]
    if not isinstance(classes, list) or not 1 <= len(classes) <= 80:
        raise ValueError("Invalid COCO class list")
    label_to_category = {row["label"]: integer(row["category_id"], "category_id", 1) for row in classes}
    category_ids = sorted(label_to_category.values())
    if len(set(category_ids)) != len(classes) or len(label_to_category) != len(classes):
        raise ValueError("Duplicate COCO class label or category ID")
    original_categories = {row["id"]: row["name"] for row in original["categories"]}
    if any(original_categories.get(category) != label for label, category in label_to_category.items()):
        raise ValueError("Manifest label/category mapping differs from original COCO taxonomy")
    images = manifest["images"]
    if not isinstance(images, list) or not 1 <= len(images) <= 1000:
        raise ValueError("Invalid COCO image list")
    image_ids = sorted(integer(row["coco_image_id"], "coco_image_id", 1) for row in images)
    if len(set(image_ids)) != len(images):
        raise ValueError("Duplicate COCO image ID")
    original_images = {row["id"]: row for row in original["images"]}
    annotations_by_image = {image_id: [] for image_id in image_ids}
    for annotation in original["annotations"]:
        if annotation["image_id"] in annotations_by_image and annotation["category_id"] in category_ids:
            annotations_by_image[annotation["image_id"]].append(annotation)
    # Validate the complete selected-image annotation set, including small and
    # crowd objects. A crop-selection manifest must not silently drop their GT.
    for image in images:
        source_image = original_images.get(image["coco_image_id"])
        if source_image is None or any(image[key] != source_image[key] for key in ("width", "height")):
            raise ValueError("Manifest image dimensions differ from original COCO annotation")
        expected = {row["id"]: row for row in annotations_by_image[image["coco_image_id"]]}
        actual = {row["annotation_id"]: row for row in image["ground_truth"]}
        if len(actual) != len(image["ground_truth"]) or actual.keys() != expected.keys():
            raise ValueError("COCO manifest GT is not the complete selected-category annotation set")
        for annotation_id, row in actual.items():
            source = expected[annotation_id]
            x, y, width, height = source["bbox"]
            source_box = [x, y, x + width, y + height]
            if (row["label"] != original_categories[source["category_id"]]
                    or type(row["crowd"]) is not bool or row["crowd"] != bool(source.get("iscrowd", 0))
                    or len(row["bbox"]) != 4
                    or any(abs(finite(a, "COCO GT box") - b) > 1e-9 for a, b in zip(row["bbox"], source_box))):
                raise ValueError(f"COCO manifest annotation {annotation_id} differs from original source")
    per_category_predictions = {category: 0 for category in category_ids}
    for prediction in predictions:
        if prediction["image_id"] not in image_ids or prediction["category_id"] not in category_ids:
            raise ValueError("COCO prediction lies outside selected images/classes")
        score = finite(prediction["score"], "prediction score")
        bbox = prediction["bbox"]
        if not 0 <= score <= 1 or len(bbox) != 4 or any(not math.isfinite(finite(v, "prediction bbox")) for v in bbox):
            raise ValueError("Invalid COCO prediction")
        if bbox[2] <= 0 or bbox[3] <= 0:
            raise ValueError("Degenerate COCO prediction")
        per_category_predictions[prediction["category_id"]] += 1
    coco_gt = COCO(str(annotations_path))
    if predictions:
        coco_dt = coco_gt.loadRes(predictions)
    else:
        # Official loadRes assumes a nonempty list. An empty COCO dataset uses
        # its public index API; metric computation is still official COCOeval.
        coco_dt = COCO()
        coco_dt.dataset = {"images": original["images"], "categories": original["categories"], "annotations": []}
        coco_dt.createIndex()
    evaluator = COCOeval(coco_gt, coco_dt, "bbox")
    evaluator.params.imgIds = image_ids
    evaluator.params.catIds = category_ids
    evaluator.params.maxDets = [1, 10, 100]
    evaluator.evaluate()
    evaluator.accumulate()
    # [IoU, recall, category, area, maxDets]; use only all-area/maxDets100.
    precision = evaluator.eval["precision"][:, :, :, 0, 2]

    def mean_valid(values: Any) -> float | None:
        valid = values[values >= 0]
        return float(np.mean(valid)) if valid.size else None

    detection = report["detection"]
    checks: dict[str, Any] = {
        "images": compare_count(detection["images"], len(image_ids), "COCO images"),
        "predictions": compare_count(detection["predictions"], len(predictions), "COCO predictions"),
        "ap50": compare_number(detection["ap50"], mean_valid(precision[0]), "COCO AP50"),
        "ap50_95": compare_number(detection["ap50_95"], mean_valid(precision), "COCO AP50:95"),
        "per_class": {},
    }
    cpp_classes = {row["label"]: row for row in detection["per_class"]}
    if len(cpp_classes) != len(detection["per_class"]) or cpp_classes.keys() != label_to_category.keys():
        raise ValueError("Report per-class entries do not match manifest")
    crowd_count = sum(bool(row.get("iscrowd", 0)) for rows in annotations_by_image.values() for row in rows)
    gt_count = sum(len(rows) for rows in annotations_by_image.values()) - crowd_count
    checks["ground_truth"] = compare_count(detection["ground_truth"], gt_count, "COCO regular GT")
    checks["crowd_ground_truth"] = compare_count(detection["crowd_ground_truth"], crowd_count, "COCO crowd GT")
    for label, category in label_to_category.items():
        index = evaluator.params.catIds.index(category)
        rows = [row for group in annotations_by_image.values() for row in group if row["category_id"] == category]
        crowd = sum(bool(row.get("iscrowd", 0)) for row in rows)
        cpp = cpp_classes[label]
        checks["per_class"][label] = {
            "category_id": category,
            "gt": compare_count(cpp["gt"], len(rows) - crowd, f"{label} GT"),
            "crowd_gt": compare_count(cpp["crowd_gt"], crowd, f"{label} crowd GT"),
            "predictions": compare_count(cpp["predictions"], per_category_predictions[category], f"{label} predictions"),
            "ap50": compare_number(cpp["ap50"], mean_valid(precision[0, :, index]), f"{label} AP50"),
            "ap50_95": compare_number(cpp["ap50_95"], mean_valid(precision[:, :, index]), f"{label} AP50:95"),
        }
    return {"reference": "official pycocotools COCOeval", "pycocotools_version": importlib.metadata.version("pycocotools"),
            "protocol": "bbox; selected imgIds/catIds; all area; maxDets 100; default .50:.05:.95 IoUs and 101 recalls",
            "checks": checks}, hashes


def checked_objects(rows: Any, context: str) -> list[dict]:
    if not isinstance(rows, list) or len(rows) > 500:
        raise ValueError(f"Invalid object list: {context}")
    ids = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"Expected object entry: {context}")
        object_id = integer(row["id"], f"{context} ID")
        if object_id in ids:
            raise ValueError(f"Duplicate per-frame identity: {context}")
        ids.add(object_id)
        box = row["bbox"]
        if not isinstance(box, list) or len(box) != 4:
            raise ValueError(f"Expected xyxy box: {context}")
        values = [finite(value, f"{context} bbox") for value in box]
        if values[2] <= values[0] or values[3] <= values[1]:
            raise ValueError(f"Degenerate xyxy box: {context}")
    return rows


def tracking_frame_inputs(manifest: dict, report: dict, frames: Any) -> tuple[str, ...]:
    """Validate the complete emitted sequence without reference dependencies.

    The report may contain protocol/parameter metadata, but never an unknown
    tracker. Legacy two-tracker reports remain valid. Every frame and report
    must agree exactly on which optional motion trackers are present. Detector rows are
    bounded, optional provenance only; they never feed reference metric inputs.
    """
    count = integer(manifest["frames"], "manifest frames", 1)
    if (not isinstance(frames, list) or len(frames) != count or count > 1000
            or manifest["classes"] != ["person"]):
        raise ValueError("Expected every frame of the single-class tracking sequence")
    tracking = report.get("tracking")
    if not isinstance(tracking, dict):
        raise ValueError("Expected tracking report object")
    unknown = tracking.keys() - set(TRACKERS) - TRACKING_METADATA
    if unknown:
        raise ValueError(f"Unknown tracking report keys: {sorted(unknown)}")
    trackers = tuple(name for name in TRACKERS if name in tracking)
    if not REQUIRED_TRACKERS.issubset(trackers):
        raise ValueError("Tracking report must contain iou and two_stage")
    if any(not isinstance(tracking[name], dict) for name in trackers):
        raise ValueError("Expected per-tracker metric report objects")

    rows = manifest["ground_truth"]
    if not isinstance(rows, list) or len(rows) > 500000:
        raise ValueError("Invalid or oversized manifest tracking GT")
    expected_gt: list[list[dict]] = [[] for _ in range(count)]
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Expected manifest GT entry")
        frame_index = integer(row["frame_index"], "GT frame index", 1)
        if frame_index > count:
            raise ValueError("GT frame index exceeds sequence")
        expected_gt[frame_index - 1].append({"id": row["id"], "bbox": row["bbox"]})

    total_pair_cells = 0
    for index, frame in enumerate(frames):
        if not isinstance(frame, dict) or integer(frame["frame_index"], "frame index", 1) != index + 1:
            raise ValueError("Tracking frames must be contiguous and one-based")
        unknown = frame.keys() - set(TRACKERS) - FRAME_METADATA
        if unknown:
            raise ValueError(f"Unknown frame {index + 1} keys: {sorted(unknown)}")
        emitted = tuple(name for name in TRACKERS if name in frame)
        if emitted != trackers:
            raise ValueError(f"Tracker set differs between report and frame {index + 1}")
        for side in ("ground_truth", *trackers):
            checked_objects(frame[side], f"frame {index + 1} {side}")
        if "raw_detections" in frame:
            # A sealed provenance field, not another tracker or a source of GT.
            raw = frame["raw_detections"]
            if not isinstance(raw, list) or len(raw) > 500:
                raise ValueError(f"Invalid raw detections at frame {index + 1}")
        expected = {row["id"]: row for row in checked_objects(expected_gt[index], "manifest GT")}
        actual = {row["id"]: row for row in frame["ground_truth"]}
        if expected.keys() != actual.keys():
            raise ValueError(f"Emitted GT identities differ from manifest at frame {index + 1}")
        for object_id, row in expected.items():
            # BoundingBox owns IEEE float32; widening it to JSON doubles must
            # match exactly, not merely overlap the source box geometrically.
            canonical = [struct.unpack("!f", struct.pack("!f", value))[0] for value in row["bbox"]]
            if actual[object_id]["bbox"] != canonical:
                raise ValueError(f"Emitted GT box differs from manifest at frame {index + 1}, ID {object_id}")
        total_pair_cells += len(actual) * sum(len(frame[name]) for name in trackers)
    if total_pair_cells > 50000000:
        raise ValueError("Tracking reference pair-comparison limit exceeded")
    return trackers


def manifest_file(manifest_path: Path, name: str) -> Path:
    base = manifest_path.resolve().parent
    relative = Path(name)
    if relative.is_absolute() or relative.drive:
        raise ValueError("Manifest paths must be relative")
    result = (base / relative).resolve()
    if not result.is_relative_to(base) or not result.is_file():
        raise ValueError("Missing or escaped manifest input")
    return result


def check_original_mot_gt(manifest_path: Path, manifest: dict, hashes: dict[str, str]) -> None:
    ground_truth = manifest_file(manifest_path, manifest["ground_truth_file"])
    digest = sha256(ground_truth)
    if digest != manifest["ground_truth_sha256"]:
        raise ValueError("Original MOT GT SHA256 differs from manifest")
    if ground_truth.stat().st_size > 16 * 1024 * 1024:
        raise ValueError("Original MOT GT exceeds size limit")
    hashes[str(ground_truth)] = digest
    expected = {}
    with ground_truth.open("r", encoding="utf-8-sig", newline="") as stream:
        for index, row in enumerate(csv.reader(stream)):
            if not row:
                continue
            if index >= 100000 or len(row) != 10:
                raise ValueError("Invalid or oversized original MOT15 GT")
            values = [finite(float(value), "original MOT GT") for value in row]
            if values[6] == 0:  # MOT15 consider flag, not a MOT16 class column.
                continue
            frame, object_id = values[:2]
            if not frame.is_integer() or not object_id.is_integer() or not 1 <= frame <= manifest["frames"] or object_id < 0:
                raise ValueError("Invalid original MOT GT frame or ID")
            key = (int(frame), int(object_id))
            if key in expected:
                raise ValueError("Duplicate original MOT GT frame/ID")
            x, y, width, height = values[2:6]
            if width <= 0 or height <= 0:
                raise ValueError("Degenerate original MOT GT box")
            expected[key] = [x - 1, y - 1, x - 1 + width, y - 1 + height]
    actual = {(row["frame_index"], row["id"]): row["bbox"] for row in manifest["ground_truth"]}
    if len(actual) != len(manifest["ground_truth"]) or actual.keys() != expected.keys():
        raise ValueError("MOT manifest is not the complete consider-flag-filtered original GT")
    for key, source_box in expected.items():
        box = actual[key]
        if len(box) != 4 or any(abs(finite(a, "manifest MOT GT") - b) > 1e-9 for a, b in zip(box, source_box)):
            raise ValueError(f"MOT manifest box differs from original GT at frame/ID {key}")
    video = manifest_file(manifest_path, manifest["video"])
    video_digest = sha256(video)
    if video_digest != manifest["video_sha256"]:
        raise ValueError("MOT source video SHA256 differs from manifest")
    hashes[str(video)] = video_digest


def tracking_data(frames: list[dict], tracker: str, np: Any) -> dict:
    gt_ids = sorted({row["id"] for frame in frames for row in frame["ground_truth"]})
    pred_ids = sorted({row["id"] for frame in frames for row in frame[tracker]})
    if len(gt_ids) > 1000 or len(pred_ids) > 1000:
        raise ValueError("Tracking identity limit exceeded")
    gt_map = {object_id: index for index, object_id in enumerate(gt_ids)}
    pred_map = {object_id: index for index, object_id in enumerate(pred_ids)}
    data: dict[str, Any] = {
        "num_timesteps": len(frames), "num_gt_ids": len(gt_ids), "num_tracker_ids": len(pred_ids),
        "num_gt_dets": sum(len(frame["ground_truth"]) for frame in frames),
        "num_tracker_dets": sum(len(frame[tracker]) for frame in frames),
        "gt_ids": [], "tracker_ids": [], "similarity_scores": [],
    }
    for frame in frames:
        ground_truth, predictions = frame["ground_truth"], frame[tracker]
        data["gt_ids"].append(np.array([gt_map[row["id"]] for row in ground_truth], dtype=np.int64))
        data["tracker_ids"].append(np.array([pred_map[row["id"]] for row in predictions], dtype=np.int64))
        a = np.array([row["bbox"] for row in ground_truth], dtype=np.float64).reshape((-1, 4))
        b = np.array([row["bbox"] for row in predictions], dtype=np.float64).reshape((-1, 4))
        intersection_size = np.maximum(0.0, np.minimum(a[:, None, 2:], b[None, :, 2:]) - np.maximum(a[:, None, :2], b[None, :, :2]))
        intersection = intersection_size[:, :, 0] * intersection_size[:, :, 1]
        a_area = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
        b_area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        union = a_area[:, None] + b_area[None, :] - intersection
        data["similarity_scores"].append(np.clip(intersection / union, 0.0, 1.0))
    return data


def compare_hota(cpp: dict, reference: dict, has_gt: bool) -> dict:
    if not isinstance(cpp, dict) or not isinstance(cpp.get("thresholds"), list) or len(cpp["thresholds"]) != 19:
        raise ValueError("HOTA requires all 19 alpha levels")
    metrics = {"hota": "HOTA", "det_a": "DetA", "ass_a": "AssA", "loc_a": "LocA"}
    levels = []
    for index, row in enumerate(cpp["thresholds"]):
        expected_alpha = .05 + .05 * index
        if abs(finite(row["alpha"], "HOTA alpha") - expected_alpha) > 1e-12:
            raise ValueError("HOTA alpha grid differs")
        checks = {name: compare_count(row[name], reference[field][index], f"HOTA {index} {name}")
                  for name, field in (("tp", "HOTA_TP"), ("fp", "HOTA_FP"), ("fn", "HOTA_FN"))}
        checks.update({name: compare_number(row[name], float(reference[field][index]), f"HOTA {index} {name}")
                       for name, field in metrics.items()})
        levels.append({"alpha": expected_alpha, "checks": checks})
    means = {"mean" if name == "hota" else name:
             compare_number(cpp["mean" if name == "hota" else name],
                            sum(map(float, reference[field])) / 19 if has_gt else None,
                            f"HOTA mean {name}") for name, field in metrics.items()}
    return {"means": means, "thresholds": levels}


def mot_validation(manifest_path: Path, result_dir: Path, reference_root: Path) -> tuple[dict, dict]:
    import numpy as np

    compatibility = []
    if "int" not in np.__dict__:
        # Upstream Identity uses the removed np.int spelling solely in astype.
        # Restoring its original Python-int alias does not change metric logic.
        np.int = int
        compatibility.append("np.int = int: upstream Identity astype compatibility with NumPy >= 1.24")
    from trackeval.metrics import CLEAR, Identity, HOTA

    manifest, report, hashes = base_inputs(manifest_path, result_dir, "video")
    frames_path = result_dir / "tracking-frames.json"
    frames = read_json(frames_path)
    hashes[str(frames_path.resolve())] = sha256(frames_path)
    trackers = tracking_frame_inputs(manifest, report, frames)
    has_hota = any("hota" in report["tracking"][tracker] for tracker in trackers)
    if has_hota and not all("hota" in report["tracking"][tracker] for tracker in trackers):
        raise ValueError("HOTA must be reported for every selected tracker")
    if has_hota and "float" not in np.__dict__:
        np.float = float
        compatibility.append("np.float = float: upstream HOTA dtype compatibility with NumPy >= 1.24")
    check_original_mot_gt(manifest_path, manifest, hashes)
    checks: dict[str, Any] = {}
    for tracker in trackers:
        cpp = report["tracking"][tracker]
        if cpp["iou_threshold"] != 0.5:
            raise ValueError("Reference protocol requires IoU threshold .5")
        data = tracking_data(frames, tracker, np)
        clear = CLEAR({"THRESHOLD": 0.5, "PRINT_CONFIG": False}).eval_sequence(data)
        identity = Identity({"THRESHOLD": 0.5, "PRINT_CONFIG": False}).eval_sequence(data)
        counts = {"frames": data["num_timesteps"], "gt": data["num_gt_dets"], "predictions": data["num_tracker_dets"],
                  "gt_identities": data["num_gt_ids"], "predicted_identities": data["num_tracker_ids"],
                  "tp": clear["CLR_TP"], "fp": clear["CLR_FP"], "fn": clear["CLR_FN"], "id_switches": clear["IDSW"],
                  "idtp": identity["IDTP"], "idfp": identity["IDFP"], "idfn": identity["IDFN"]}
        tracker_checks = {key: compare_count(cpp[key], value, f"{tracker} {key}") for key, value in counts.items()}
        has_gt, has_predictions, has_tp = bool(data["num_gt_dets"]), bool(data["num_tracker_dets"]), bool(clear["CLR_TP"])
        scores = {"precision": float(clear["CLR_Pr"]) if has_gt and has_predictions else None,
                  "recall": float(clear["CLR_Re"]) if has_gt else None,
                  "mota": float(clear["MOTA"]) if has_gt else None,
                  "motp": float(clear["MOTP"]) if has_tp else None,
                  "idf1": float(identity["IDF1"]) if has_gt else None}
        tracker_checks.update({key: compare_number(cpp[key], value, f"{tracker} {key}") for key, value in scores.items()})
        if has_hota:
            tracker_checks["hota"] = compare_hota(cpp["hota"], HOTA().eval_sequence(data), has_gt)
        checks[tracker] = tracker_checks
    repo = reference_root / "TrackEval"
    commit = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    changed = subprocess.run(["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"], check=True,
                             capture_output=True, text=True).stdout.strip()
    if changed:
        raise ValueError("Reference TrackEval checkout has tracked changes; cannot attest official source")
    return {"reference": "official TrackEval CLEAR and Identity" + (" and HOTA" if has_hota else ""), "trackeval_commit": commit,
            "protocol": "CLEAR/Identity IoU .5; HOTA alpha .05:.05:.95 when emitted; unmodified algorithms; complete normalized MOT15 GT; contiguous ID remapping",
            "numpy_compatibility": compatibility, "numpy_version": np.__version__, "checks": checks}, hashes


def write_success(result_dir: Path, result: dict, hashes: dict[str, str]) -> None:
    # A successful record is tied to exact inputs/outputs and never emitted if
    # any comparison fails or an input changes while reference metrics run.
    for filename, digest in hashes.items():
        if sha256(Path(filename)) != digest:
            raise ValueError(f"Input changed during reference validation: {filename}")
    payload = {"version": 1, "status": "passed", "absolute_tolerance": TOLERANCE, "input_sha256": hashes, **result}
    output = result_dir / "reference-validation.json"
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=result_dir, prefix="reference-validation-", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    print(f"PASS: official reference agrees within {TOLERANCE:g}; {output}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("coco", "mot"):
        command = commands.add_parser(name)
        command.add_argument("manifest", type=Path)
        command.add_argument("result_dir", type=Path)
        if name == "coco":
            command.add_argument("annotations", type=Path)
        command.add_argument("--reference-root", type=Path, default=DEFAULT_REFERENCE_ROOT)
    args = parser.parse_args(argv)
    reference_root = args.reference_root.resolve()
    if not reference_root.is_dir():
        parser.error("Reference root is missing; install official pycocotools/TrackEval separately")
    sys.path[:0] = [str(reference_root), str(reference_root / "TrackEval")]
    try:
        if args.command == "coco":
            result, hashes = coco_validation(args.manifest, args.result_dir, args.annotations, reference_root)
        else:
            result, hashes = mot_validation(args.manifest, args.result_dir, reference_root)
        write_success(args.result_dir, result, hashes)
    except (OSError, ValueError, KeyError, IndexError, TypeError, OverflowError, ImportError, subprocess.CalledProcessError) as error:
        print(f"Reference validation FAILED: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
