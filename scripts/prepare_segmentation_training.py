"""Build a frozen Penn->Fudan transfer experiment; no model inference or training."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import zipfile

import cv2
import numpy as np

SOURCE = "https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip"


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def split_names(names):
    if len(names) != len(set(names)) or any(not re.fullmatch(r"(?:Penn|Fudan)Ped\d{5}", n) for n in names):
        raise ValueError("Invalid or duplicated image names")
    penn = sorted((n for n in names if n.startswith("Penn")),
                  key=lambda n: hashlib.sha256(("aegis-person-finetune-v1:" + n).encode()).hexdigest())
    fudan = sorted(n for n in names if n.startswith("Fudan"))
    if len(penn) != 96 or len(fudan) != 74:
        raise ValueError("Expected 96 Penn and 74 Fudan images")
    return {"train": sorted(penn[16:]), "val": sorted(penn[:16]), "test": fudan}


def rle(mask):
    flat = mask.ravel(order="F").astype(np.uint8)
    edges = np.flatnonzero(np.diff(flat)) + 1
    counts = np.diff(np.r_[0, edges, flat.size]).tolist()
    if flat[0]:
        counts.insert(0, 0)
    return {"size": list(mask.shape), "counts": counts}


def polygon(mask):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("Empty instance mask")
    contour = max(contours, key=cv2.contourArea).reshape(-1, 2)
    if len(contour) < 3 or cv2.contourArea(contour) <= 0:
        raise ValueError("Degenerate instance polygon")
    reconstructed = np.zeros(mask.shape, np.uint8)
    cv2.fillPoly(reconstructed, [contour], 1)
    overlap = float(np.logical_and(reconstructed, mask).sum() / np.logical_or(reconstructed, mask).sum())
    height, width = mask.shape
    points = contour.astype(np.float64) / [width, height]
    return "0 " + " ".join(f"{x:.9f}" for x in points.ravel()), overlap


def save(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def prepare(archive, output, categories_path, expected_sha):
    if digest(archive) != expected_sha:
        raise ValueError("Dataset archive SHA256 differs")
    # Preserve the pretrained 80-class head/order for the existing C++ decoder.
    categories = sorted(json.loads(categories_path.read_text(encoding="utf-8"))["categories"], key=lambda x: x["id"])
    if len(categories) != 80 or categories[0]["name"] != "person" or categories[0]["id"] != 1:
        raise ValueError("Expected original COCO 80-class mapping")
    output.mkdir(parents=True, exist_ok=False)
    coco = {"info": {"description": "Penn-Fudan original pixel masks, COCO-shaped annotations"},
            "images": [], "annotations": [], "categories": categories}
    entries = {s: [] for s in ("train", "val", "test")}
    overlaps = []
    with zipfile.ZipFile(archive) as z:
        if len(z.infolist()) != len(set(z.namelist())) or sum(i.file_size for i in z.infolist()) > 100 * 1024 * 1024:
            raise ValueError("Archive duplicate entries or size limit")
        names = [Path(n).stem for n in z.namelist() if re.fullmatch(r"PennFudanPed/PNGImages/(?:Penn|Fudan)Ped\d{5}\.png", n)]
        splits = split_names(names)
        identities = {name: index + 1 for index, name in enumerate(sorted(names))}
        fingerprints = set()
        for split, selected in splits.items():
            (output / "images" / split).mkdir(parents=True)
            (output / "labels" / split).mkdir(parents=True)
            for name in selected:
                image_bytes = z.read(f"PennFudanPed/PNGImages/{name}.png")
                mask_bytes = z.read(f"PennFudanPed/PedMasks/{name}_mask.png")
                if max(len(image_bytes), len(mask_bytes)) > 5 * 1024 * 1024:
                    raise ValueError("Image byte limit")
                image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
                # Dataset masks are indexed PNGs; PIL preserves the instance palette indices.
                from PIL import Image
                import io
                mask = np.asarray(Image.open(io.BytesIO(mask_bytes)))
                if image is None or mask.ndim != 2 or image.shape[:2] != mask.shape or max(mask.shape) > 1920:
                    raise ValueError("Image/mask dimensions invalid")
                fingerprint = hashlib.sha256(image.tobytes()).hexdigest()
                if fingerprint in fingerprints:
                    raise ValueError("Duplicate decoded image across experiment splits")
                fingerprints.add(fingerprint)
                height, width = mask.shape
                image_id = identities[name]
                relative = f"images/{split}/{name}.png"
                with (output / relative).open("xb") as stream:
                    stream.write(image_bytes)
                coco["images"].append({"id": image_id, "width": width, "height": height, "file_name": relative})
                labels = []
                for instance in sorted(set(np.unique(mask)) - {0}):
                    binary = mask == instance
                    text, overlap = polygon(binary)
                    labels.append(text); overlaps.append(overlap)
                    ys, xs = np.nonzero(binary)
                    coco["annotations"].append({"id": len(coco["annotations"]) + 1, "image_id": image_id,
                        "category_id": 1, "iscrowd": 0, "area": int(binary.sum()),
                        "bbox": [int(xs.min()), int(ys.min()), int(xs.max()-xs.min()+1), int(ys.max()-ys.min()+1)],
                        "segmentation": rle(binary)})
                label_path = output / "labels" / split / f"{name}.txt"
                with label_path.open("x", encoding="utf-8") as stream:
                    stream.write("\n".join(labels) + "\n")
                entries[split].append({"path": relative, "coco_image_id": image_id, "width": width, "height": height,
                    "sha256": hashlib.sha256(image_bytes).hexdigest(), "label_sha256": digest(label_path),
                    "source_mask_sha256": hashlib.sha256(mask_bytes).hexdigest(), "instances": len(labels)})
    save(output / "annotations.json", coco)
    limitations = ["Small person-only domain transfer, not a general 80-class quality guarantee.",
        "Penn train/validation scenes may share backgrounds; Fudan location is held out from fine-tuning.",
        "No claim that upstream pretrained weights never encountered similar images.",
        "Training uses largest external polygon per person; evaluation uses complete original pixel masks.",
        "Dataset contains upright, relatively large pedestrians; no negative-only scenes."]
    for split in entries:
        save(output / f"{split}-manifest.json", {"version": 1, "kind": "images", "dataset": f"penn-fudan-transfer-v1-{split}",
            "source": SOURCE, "archive_sha256": expected_sha, "annotation_sha256": digest(output / "annotations.json"),
            "classes": [{"label": "person", "category_id": 1}], "limitations": limitations, "images": entries[split]})
    save(output / "dataset.yaml", {"path": str(output.resolve()), "train": "images/train", "val": "images/val",
        "test": "images/test", "names": {i: c["name"] for i, c in enumerate(categories)}})
    save(output / "split-summary.json", {"archive_sha256": expected_sha, "splits": {s: len(v) for s, v in entries.items()},
        "instances": len(overlaps), "training_polygon_iou_min": min(overlaps),
        "training_polygon_iou_mean": sum(overlaps)/len(overlaps), "limitations": limitations})
    print(json.dumps({s: len(v) for s, v in entries.items()}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path); parser.add_argument("output", type=Path)
    parser.add_argument("coco_categories", type=Path); parser.add_argument("--sha256", required=True)
    args = parser.parse_args()
    prepare(args.archive, args.output, args.coco_categories, args.sha256)
