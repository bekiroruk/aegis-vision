"""Create a deterministic, local-only COCO 2017 crop-search benchmark manifest.

Download official instances_val2017.json separately; this script downloads only
the selected JPEGs from the official COCO image host when --download is supplied.
No images or annotations are committed to Git.
"""

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen


CLASSES = ("bicycle", "motorcycle", "bus", "train", "cat", "dog", "zebra", "giraffe")
PROMPT = "a photo of a {}"
SOURCE = "https://s3.amazonaws.com/images.cocodataset.org/val2017/"


def select(annotation: dict, per_class: int) -> list[tuple[dict, dict, str]]:
    names = {category["id"]: category["name"] for category in annotation["categories"]}
    images = {image["id"]: image for image in annotation["images"]}
    candidates: dict[str, list[tuple[dict, dict, str]]] = {name: [] for name in CLASSES}
    for entry in annotation["annotations"]:
        label = names.get(entry["category_id"])
        if label not in candidates or entry["iscrowd"]:
            continue
        image = images[entry["image_id"]]
        x, y, width, height = entry["bbox"]
        if width < 96 or height < 96 or width * height < 0.05 * image["width"] * image["height"]:
            continue
        if x < 0 or y < 0 or x + width > image["width"] + 1 or y + height > image["height"] + 1:
            continue
        candidates[label].append((entry, image, label))
    chosen = []
    used_images = set()
    for label in CLASSES:
        # Stable pseudo-random order avoids systematically selecting low COCO IDs.
        ordered = sorted(candidates[label], key=lambda row: hashlib.sha256(
            f"aegis-coco-search-v1:{row[0]['id']}".encode()).hexdigest())
        for entry, image, _ in ordered:
            if image["id"] not in used_images:
                chosen.append((entry, image, label))
                used_images.add(image["id"])
            if sum(row[2] == label for row in chosen) == per_class:
                break
        if sum(row[2] == label for row in chosen) != per_class:
            raise ValueError(f"Insufficient eligible COCO annotations for {label}")
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("annotations", type=Path, help="Official instances_val2017.json")
    parser.add_argument("output", type=Path, help="Manifest JSON path, e.g. artifacts/datasets/coco-search/manifest.json")
    parser.add_argument("--per-class", type=int, default=8)
    parser.add_argument("--download", action="store_true", help="Download selected JPEGs over verified HTTPS")
    parser.add_argument("--image-base-url", default=SOURCE,
                        help="Official COCO val2017 image URL prefix (HTTPS by default)")
    args = parser.parse_args()
    if not 2 <= args.per_class <= 100:
        parser.error("--per-class must be 2..100")
    annotation_bytes = args.annotations.read_bytes()
    annotation = json.loads(annotation_bytes)
    chosen = select(annotation, args.per_class)
    image_dir = args.output.parent / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    items = []
    for entry, image, label in chosen:
        filename = image["file_name"]
        if Path(filename).name != filename or not filename.endswith(".jpg"):
            raise ValueError("Unexpected COCO image filename")
        target = image_dir / filename
        if not target.is_file() and args.download:
            with urlopen(args.image_base_url + filename, timeout=30) as stream:
                content = stream.read()
            if not content.startswith(b"\xff\xd8") or not content.endswith(b"\xff\xd9"):
                raise ValueError(f"Invalid JPEG download: {filename}")
            target.write_bytes(content)
        if not target.is_file():
            raise FileNotFoundError(f"Missing {target}; add --download or provide the COCO val2017 image")
        x, y, width, height = entry["bbox"]
        items.append({"id": f"coco-val2017-{entry['id']}", "image": f"images/{filename}",
                      "label": label, "bbox": [x, y, min(x + width, image["width"]),
                                              min(y + height, image["height"])]})
    manifest = {"version": 1, "dataset": f"coco-val2017-crops-8x{args.per_class}-v1",
                "source": "https://cocodataset.org/#download",
                "image_base_url": args.image_base_url,
                "annotation_sha256": hashlib.sha256(annotation_bytes).hexdigest(),
                "selection": "sha256(aegis-coco-search-v1:annotation_id); >=96px and >=5% image area",
                "items": items,
                "queries": [{"text": PROMPT.format(label), "label": label} for label in CLASSES]}
    args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Prepared {len(items)} crop items and {len(CLASSES)} queries: {args.output}")


if __name__ == "__main__":
    main()
