"""Offline official COCO segmentation evaluation of native C++ mask exports.

No inference, downloading, threshold tuning or input mutation. Reuses the fixed
64-image development selection; not a fresh holdout or full COCO leaderboard.
"""
import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import sys
import struct

# Match the native float32 threshold exactly, including its JSON double expansion.
NATIVE_CONFIDENCE = struct.unpack("f", struct.pack("f", .35))[0]

def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

def checked_rle(value, height, width):
    if not isinstance(value, dict) or value.get("size") != [height, width]:
        raise ValueError("Mask size differs from source image")
    counts = value.get("counts")
    if not isinstance(counts, list) or not 1 <= len(counts) <= height * width + 1:
        raise ValueError("Invalid RLE length")
    if any(type(x) is not int or x < 0 or x > height * width for x in counts):
        raise ValueError("Invalid RLE counts")
    if sum(counts) != height * width:
        raise ValueError("RLE does not cover the source image")
    return value

def validate_export(manifest, report, categories):
    if (report.get("version") != 1 or report.get("tracking") is not False or
        report.get("confidence") != .35 or report.get("mask_threshold") != .5 or
        report.get("nms_iou") != .45 or report.get("max_instances") != 100):
        raise ValueError("Unexpected native export protocol")
    expected = {x["coco_image_id"]: x for x in manifest["images"]}
    if not expected or len(expected) != len(manifest["images"]):
        raise ValueError("Invalid manifest image IDs")
    received = report["images"]
    if len(received) != len(expected) or len({x["id"] for x in received}) != len(expected):
        raise ValueError("Missing or duplicated exported images")
    for image in received:
        source = expected[image["id"]]
        if any(image[k] != source[k] for k in ("width", "height", "sha256")):
            raise ValueError("Export image provenance mismatch")
    selected = {x["label"]: x["category_id"] for x in manifest["classes"]}
    if any(categories.get(label) != cat for label, cat in selected.items()):
        raise ValueError("COCO class mapping mismatch")
    per_image = dict.fromkeys(expected, 0)
    predictions = []
    for item in report["predictions"]:
        source = expected[item["image_id"]]
        if item["label"] not in categories:
            raise ValueError("Unknown predicted class")
        score = item["score"]
        if type(score) not in (int, float) or not math.isfinite(score) or not NATIVE_CONFIDENCE <= score <= 1:
            raise ValueError("Invalid score")
        checked_rle(item["segmentation"], source["height"], source["width"])
        per_image[item["image_id"]] += 1
        if per_image[item["image_id"]] > 100:
            raise ValueError("Native instance limit exceeded")
        if item["label"] in selected:
            predictions.append({"image_id": item["image_id"], "category_id": selected[item["label"]],
                                "score": score, "segmentation": item["segmentation"]})
    return expected, selected, predictions

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "annotations", "model", "predictions", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--reference-root", type=Path,
                        default=Path(__file__).resolve().parents[1]/"artifacts/deps/quality-reference")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output must not exist")
    sys.path.insert(0, str(args.reference_root.resolve()))
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    from pycocotools import mask as masks
    hashes = {name: digest(getattr(args, name)) for name in ("manifest", "annotations", "model", "predictions")}
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    report = json.loads(args.predictions.read_text(encoding="utf-8"))
    if (manifest["annotation_sha256"] != hashes["annotations"] or
        report["manifest_sha256"] != hashes["manifest"] or report["model_sha256"] != hashes["model"]):
        raise ValueError("Evaluation input hashes do not match")
    gt = COCO(str(args.annotations))
    expected, selected, predictions = validate_export(manifest, report, {c["name"]: c["id"] for c in gt.cats.values()})
    for image_id, source in expected.items():
        if any(gt.imgs[image_id][k] != source[k] for k in ("width", "height")):
            raise ValueError("Annotation/image size mismatch")
    for item in predictions:
        rle = item["segmentation"]
        item["segmentation"] = masks.frPyObjects(rle, *rle["size"])
    if predictions:
        dt = gt.loadRes(predictions)  # No bbox field: area derives from MASK, not detection box.
    else:
        dt = COCO()
        dt.dataset = {"images": list(gt.imgs.values()), "categories": list(gt.cats.values()), "annotations": []}
        dt.createIndex()
    evaluator = COCOeval(gt, dt, "segm")
    evaluator.params.imgIds = sorted(expected)
    evaluator.params.catIds = sorted(selected.values())
    evaluator.evaluate(); evaluator.accumulate(); evaluator.summarize()
    matched_ious = []
    tp = fp = fn = ignored = 0
    for entry in evaluator.evalImgs:
        if entry is None or entry["aRng"] != evaluator.params.areaRng[0]:
            continue
        for index, prediction_id in enumerate(entry["dtIds"]):
            if entry["dtIgnore"][0, index]:
                ignored += 1
                continue
            match = int(entry["dtMatches"][0, index])
            if match:
                tp += 1
                matched_ious.append(float(masks.iou([dt.anns[prediction_id]["segmentation"]],
                                                    [gt.anns[match]["segmentation"]], [0])[0, 0]))
            else:
                fp += 1
        fn += sum(not bool(ignore) and not bool(match) for ignore, match in zip(entry["gtIgnore"],entry["gtMatches"][0]))
    names = ("AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large",
             "AR1", "AR10", "AR100", "AR_small", "AR_medium", "AR_large")
    result = {"protocol": "official COCOeval segm, IoU .50:.05:.95, maxDets [1,10,100]; fixed confidence .35",
              "images": len(expected), "classes": selected, "predictions_in_selected_classes": len(predictions),
              "hashes": hashes, "pycocotools": importlib.metadata.version("pycocotools"),
              "metrics": {name: float(value) if value >= 0 else None for name,value in zip(names,evaluator.stats)},
              "at_mask_iou_50": {"tp": int(tp), "fp": int(fp), "fn": int(fn), "ignored_detections": ignored,
                                 "matched_mean_iou": sum(matched_ious)/len(matched_ious) if matched_ious else None},
              "limitations": ["Previously observed 64-image/8-class development selection, not fresh holdout.",
                              "AP is at deployed confidence .35, not low-threshold/full COCO model benchmark.",
                              "Matched mean IoU excludes false positives and misses; read TP/FP/FN and AP together."]}
    if hashes != {name: digest(getattr(args,name)) for name in hashes}:
        raise ValueError("Input changed during evaluation")
    with args.output.open("x",encoding="utf-8") as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
    print(args.output)

if __name__ == "__main__":
    main()
