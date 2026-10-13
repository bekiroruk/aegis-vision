"""Freeze 128 COCO calibration images, excluding all 64 evaluation images.

Selection uses annotations only to balance eight categories, never model outputs.
Images are downloaded only from the official COCO image host. No training.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
from urllib.request import urlopen


def digest(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def select_images(annotations, evaluation, per_class=16):
    excluded = {i["coco_image_id"] for i in evaluation["images"]}
    categories = sorted(c["category_id"] for c in evaluation["classes"])
    if len(categories) != 8 or len(excluded) != 64:
        raise ValueError("Expected frozen 64-image/eight-class evaluation selection")
    candidates = {c: set() for c in categories}
    for a in annotations["annotations"]:
        if a["category_id"] in candidates and not a["iscrowd"] and a["image_id"] not in excluded:
            candidates[a["category_id"]].add(a["image_id"])
    selected, assigned = set(), {}
    for category in categories:
        ordered = sorted(candidates[category] - selected,
            key=lambda i: hashlib.sha256(f"aegis-int8-calibration-v1:{category}:{i}".encode()).hexdigest())
        if len(ordered) < per_class:
            raise ValueError("Insufficient disjoint calibration candidates")
        for image_id in ordered[:per_class]:
            selected.add(image_id); assigned[image_id] = category
    return sorted(selected, key=lambda i: hashlib.sha256(f"aegis-int8-order-v1:{i}".encode()).hexdigest()), assigned


def decoded(raw):
    import cv2
    import numpy as np
    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if image is None or max(image.shape[:2]) > 1920:
        raise ValueError("Invalid image dimensions")
    fingerprint = hashlib.sha256(str(image.shape).encode()+image.tobytes()).hexdigest()
    return image, fingerprint


def prepare(annotation_path, evaluation_path, model, output):
    annotation_hash, evaluation_hash, model_hash = map(digest, (annotation_path, evaluation_path, model))
    annotations = json.loads(annotation_path.read_text(encoding="utf-8"))
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    if evaluation["annotation_sha256"] != annotation_hash:
        raise ValueError("Annotation hash mismatch")
    selected, assigned = select_images(annotations, evaluation)
    metadata = {i["id"]: i for i in annotations["images"]}
    excluded_hashes, excluded_pixels = set(), set()
    for entry in evaluation["images"]:
        path = (evaluation_path.parent/entry["path"]).resolve()
        if not path.is_relative_to(evaluation_path.parent.resolve()) or digest(path) != entry["sha256"]:
            raise ValueError("Evaluation image provenance mismatch")
        excluded_hashes.add(entry["sha256"])
        excluded_pixels.add(decoded(path.read_bytes())[1])
    output.mkdir(parents=True, exist_ok=False)
    (output/"images").mkdir()
    def download(image_id):
        item = metadata[image_id]
        name = item["file_name"]
        if not re.fullmatch(r"\d{12}\.jpg", name) or name != f"{image_id:012d}.jpg":
            raise ValueError("Unexpected COCO filename")
        url = f"https://s3.amazonaws.com/images.cocodataset.org/val2017/{name}"
        with urlopen(url, timeout=60) as response:
            if response.geturl() != url:
                raise ValueError("Unexpected download redirect")
            raw = response.read(5*1024*1024+1)
        if not 0 < len(raw) <= 5*1024*1024:
            raise ValueError("Image byte limit")
        image, pixel_hash = decoded(raw)
        byte_hash = hashlib.sha256(raw).hexdigest()
        if image.shape[:2] != (item["height"], item["width"]):
            raise ValueError("COCO image dimension mismatch")
        if byte_hash in excluded_hashes or pixel_hash in excluded_pixels:
            raise ValueError("Calibration/evaluation content overlap")
        with (output/"images"/name).open("xb") as f:
            f.write(raw)
        return {"coco_image_id": image_id, "path": f"images/{name}", "sha256": byte_hash,
            "decoded_sha256": pixel_hash, "width": item["width"], "height": item["height"],
            "assigned_category_id": assigned[image_id], "source": url}
    with ThreadPoolExecutor(max_workers=4) as pool:
        images = list(pool.map(download, selected))  # frozen order, not completion order
    if len({i["sha256"] for i in images}) != 128 or len({i["decoded_sha256"] for i in images}) != 128:
        raise ValueError("Duplicated calibration images")
    if (annotation_hash, evaluation_hash, model_hash) != tuple(map(digest, (annotation_path, evaluation_path, model))):
        raise ValueError("Inputs changed during preparation")
    manifest = {"version": 1, "kind": "int8-calibration", "algorithm": "entropy2", "batch_size": 1,
        "preprocessing": "prepare_yolo-v1-640-rgb-nchw-0to1-pad114", "model_sha256": model_hash,
        "annotation_sha256": annotation_hash, "evaluation_manifest_sha256": evaluation_hash,
        "excluded_images": [{"coco_image_id": i["coco_image_id"], "sha256": i["sha256"]} for i in evaluation["images"]],
        "selection": "16 unique noncrowd scenes per selected category; SHA256 deterministic selection/order; no inference",
        "limitations": ["Calibration and evaluation images disjoint by ID, byte hash and decoded pixels.",
            "COCO val2017 subset; pretrained-model independence not claimed. Evaluation set was previously observed.",
            "Eight-category scene coverage, not representative of all 80 classes or production cameras."],
        "images": images}
    with (output/"manifest.json").open("x", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Frozen calibration: {len(images)} images; excluded {len(excluded_hashes)} evaluation images")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("annotations", "evaluation", "model", "output"):
        parser.add_argument(key, type=Path)
    args = parser.parse_args()
    prepare(args.annotations, args.evaluation, args.model, args.output)
