"""Prepare auditable local inputs for the native C++ quality benchmark.

No inference or model evaluation takes place in Python. COCO mode reuses the
already downloaded full scenes; MOT mode downloads only two small official
assets when --download is supplied. Data stay under ignored artifacts/.
"""

import argparse
import configparser
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from urllib.request import Request, urlopen
import zipfile


MOT_VIDEO_URL = "https://motchallenge.net/sequenceVideos/TUD-Stadtmitte-raw.mp4"
MOT_LABELS_URL = "https://motchallenge.net/data/MOT15Labels.zip"
MOT_VIDEO_SIZE = 1_870_025
MOT_LABELS_SIZE = 1_534_073
# Byte identities observed from the two official HTTPS assets, 2026-10-02.
MOT_VIDEO_SHA256 = "057efff329eb73f3434649f9b21b37d0d3ca7de8f194524140161e2d13a6ae33"
MOT_LABELS_SHA256 = "b48ee31a720a3dae558df4303b82924bd862067138bc5376e1c86aadb82f782a"
MOT_GT_ENTRY = "MOT15Labels/train/TUD-Stadtmitte/gt/gt.txt"
MOT_INFO_ENTRY = "MOT15Labels/train/TUD-Stadtmitte/seqinfo.ini"
MOT_CAMPUS_VIDEO_SIZE = 936_958
MOT_CAMPUS_VIDEO_SHA256 = "95590324a7fcd27c6a5babf7e69f763be5709f788ac1a32c986a4431aee14eae"
# Fixed before inference, 2026-10-05. Full sequences, no selected subclips.
TRANSFER_SEQUENCES = {
    "ETH-Sunnyday": (8750726, "1323d5c68c19a4f8acebcce9b6dcce58467d8a84bcec59e9c376930150499a17", 354, 640, 480, 14, "validation"),
    "PETS09-S2L1": (21357352, "e03a8d3ae953c640a2eb26bdeff8502111e8c5ac11e529f4dc67931dd548557e", 795, 768, 576, 7, "test"),
}
JSON_LIMIT = 64 * 1024 * 1024


def sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def bounded_bytes(path: Path, limit: int) -> bytes:
    if not path.is_file() or not 0 < path.stat().st_size <= limit:
        raise ValueError(f"Missing, empty, or oversized file: {path}")
    with path.open("rb") as stream:
        content = stream.read(limit + 1)
    if not 0 < len(content) <= limit:
        raise ValueError(f"File size changed or exceeds limit: {path}")
    return content


def json_bytes(path: Path) -> tuple[dict, bytes]:
    raw = bounded_bytes(path, JSON_LIMIT)
    result = json.loads(raw)
    if not isinstance(result, dict):
        raise ValueError("Expected a JSON object")
    return result, raw


def local_path(base: Path, name: str) -> Path:
    relative = Path(name)
    if not name or relative.is_absolute() or relative.drive:
        raise ValueError("Data paths must be relative")
    result = (base.resolve() / relative).resolve()
    if not result.is_relative_to(base.resolve()) or not result.is_file():
        raise ValueError(f"Missing or escaped local data path: {name}")
    return result


def positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def xywh_box(values: list) -> list[float]:
    if not isinstance(values, list) or len(values) != 4:
        raise ValueError("Expected four xywh coordinates")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in values):
        raise ValueError("Box coordinates must be numeric")
    x, y, width, height = map(float, values)
    if not all(math.isfinite(v) for v in (x, y, width, height)) or width <= 0 or height <= 0:
        raise ValueError("Non-finite or degenerate box")
    return [x, y, x + width, y + height]


def save_new_or_identical(path: Path, content: bytes) -> None:
    """Preserve any conflicting existing file rather than silently replacing it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not path.is_file() or path.read_bytes() != content:
            raise ValueError(f"Existing output differs; choose a new output path: {path}")
        return
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".quality-", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        # Exclusive destination creation also protects against a concurrent writer.
        with path.open("xb") as destination, temporary.open("rb") as source:
            while chunk := source.read(65536):
                destination.write(chunk)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def save_manifest(path: Path, manifest: dict) -> None:
    save_new_or_identical(path, (json.dumps(manifest, indent=2, allow_nan=False) + "\n").encode("utf-8"))


def prepare_coco(annotations: Path, crop_manifest: Path, output: Path) -> dict:
    annotation, annotation_raw = json_bytes(annotations)
    crops, crop_raw = json_bytes(crop_manifest)
    if crops.get("version") != 1 or crops.get("annotation_sha256") != sha256(annotation_raw):
        raise ValueError("Crop selection was not created from these exact annotations")
    if not isinstance(crops.get("items"), list) or not 1 <= len(crops["items"]) <= 1000:
        raise ValueError("Require 1..1000 selected crop records")
    categories = {}
    for category in annotation["categories"]:
        label = category["name"]
        if not isinstance(label, str) or not label or label in categories:
            raise ValueError("Invalid or duplicate category name")
        categories[label] = positive_int(category["id"], "category id")
    selected_labels = sorted({item["label"] for item in crops["items"]})
    if not 1 <= len(selected_labels) <= 80 or any(label not in categories for label in selected_labels):
        raise ValueError("Unknown selected class")
    metadata = {image["file_name"]: image for image in annotation["images"]}
    if len(metadata) != len(annotation["images"]):
        raise ValueError("Duplicate COCO image filename")
    scene_paths = {}
    for item in crops["items"]:
        source = local_path(crop_manifest.parent, item["image"])
        image = metadata.get(source.name)
        if image is None:
            raise ValueError("Selected image is absent from annotations")
        if source.name != image["file_name"]:
            raise ValueError("COCO filename mismatch")
        if not source.is_relative_to(output.parent.resolve()):
            raise ValueError("Output manifest must contain selected images within its directory")
        scene_paths[positive_int(image["id"], "image id")] = (source, image)
    category_labels = {categories[label]: label for label in selected_labels}
    scene_gt = {image_id: [] for image_id in scene_paths}
    annotation_ids = set()
    for entry in annotation["annotations"]:
        if entry["image_id"] not in scene_gt or entry["category_id"] not in category_labels:
            continue
        annotation_id = positive_int(entry["id"], "annotation id")
        if annotation_id in annotation_ids:
            raise ValueError("Duplicate selected annotation id")
        annotation_ids.add(annotation_id)
        crowd = entry.get("iscrowd", 0)
        if crowd not in (0, 1) or isinstance(crowd, float):
            raise ValueError("iscrowd must be 0 or 1")
        if entry.get("ignore", 0):
            raise ValueError("Additional COCO ignore annotations are not supported")
        scene_gt[entry["image_id"]].append({"bbox": xywh_box(entry["bbox"]),
            "label": category_labels[entry["category_id"]], "crowd": bool(crowd), "annotation_id": annotation_id})
    images = []
    for image_id, (source, image) in sorted(scene_paths.items()):
        raw = bounded_bytes(source, 16 * 1024 * 1024)
        if not raw.startswith(b"\xff\xd8") or not raw.endswith(b"\xff\xd9"):
            raise ValueError(f"Not a complete JPEG: {source}")
        images.append({"id": f"coco-val2017-{image_id}", "coco_image_id": image_id,
            "path": source.relative_to(output.parent.resolve()).as_posix(), "sha256": sha256(raw),
            "width": positive_int(image["width"], "width"), "height": positive_int(image["height"], "height"),
            "ground_truth": sorted(scene_gt[image_id], key=lambda g: g["annotation_id"])})
    manifest = {"version": 1, "kind": "images",
        "dataset": f"coco-val2017-selected-scenes-{len(images)}-v1",
        "source": "https://cocodataset.org/#download", "annotation_sha256": sha256(annotation_raw),
        "crop_manifest_sha256": sha256(crop_raw), "crop_dataset": crops.get("dataset"),
        "selection": crops.get("selection"),
        "limitations": ["Selection reuses large-object crop search scenes; the scene selection is biased.",
            "All annotations of selected categories are retained, including small and crowd objects.",
            "Only selected categories are evaluated; not an official 80-category COCO score.",
            "Validation scenes are an exploratory baseline, not a held-out test split or training dataset."],
        "classes": [{"label": label, "category_id": categories[label]} for label in selected_labels], "images": images}
    save_manifest(output, manifest)
    return manifest


def download_fixed(url: str, target: Path, exact_size: int, expected_sha256: str | None = None) -> None:
    if target.exists():
        if not target.is_file() or target.stat().st_size != exact_size:
            raise ValueError(f"Existing asset has unexpected size; preserving it: {target}")
        if expected_sha256 and sha256(bounded_bytes(target, exact_size)) != expected_sha256:
            raise ValueError(f"Existing asset has unexpected SHA256; preserving it: {target}")
        return
    request = Request(url, headers={"User-Agent": "AegisVision-quality-preparer/1.0"})
    with urlopen(request, timeout=30) as response:
        if response.geturl().split(":", 1)[0] != "https":
            raise ValueError("Refusing non-HTTPS asset redirect")
        declared = response.headers.get("Content-Length")
        if declared is not None and int(declared) != exact_size:
            raise ValueError("Official asset size differs from the pinned download")
        content = response.read(exact_size + 1)
        if len(content) != exact_size:
            raise ValueError("Official asset is incomplete or exceeds its pinned size")
        if expected_sha256 and sha256(content) != expected_sha256:
            raise ValueError("Official asset content differs from its pinned SHA256")
    save_new_or_identical(target, content)


def parse_mot_gt(raw: bytes, frame_count: int) -> tuple[list[dict], int]:
    ground_truth = []
    seen = set()
    ignored = 0
    for number, line in enumerate(raw.decode("utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        cells = line.strip().split(",")
        if len(cells) != 10:
            raise ValueError(f"MOT15 row {number} must have exactly 10 fields")
        values = [float(cell) for cell in cells]
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"MOT15 row {number} is non-finite")
        frame, identity = values[:2]
        if not frame.is_integer() or not 1 <= frame <= frame_count or not identity.is_integer() or identity < 0:
            raise ValueError(f"Invalid MOT frame or identity at row {number}")
        box = xywh_box(values[2:6])
        key = (int(frame), int(identity))
        if key in seen:
            raise ValueError(f"Duplicate frame/identity in MOT15 row {number}")
        seen.add(key)
        # MOT15 column 7 is a consideration flag; columns 8..10 are world xyz.
        # They are not the class/visibility fields of MOT16/MOT17.
        if values[6] == 0:
            ignored += 1
            continue
        # Official MOT uses (1,1) for the top-left; do not clip off-screen GT.
        ground_truth.append({"frame_index": int(frame), "id": int(identity), "bbox": [v - 1 for v in box]})
    return sorted(ground_truth, key=lambda g: (g["frame_index"], g["id"])), ignored


def prepare_mot(output_dir: Path, download: bool = False, sequence: str = "TUD-Stadtmitte") -> dict:
    if sequence not in ("TUD-Stadtmitte", "TUD-Campus", *TRANSFER_SEQUENCES):
        raise ValueError("Only pinned MOT15 sequences are supported")
    campus = sequence == "TUD-Campus"
    video_url = f"https://motchallenge.net/sequenceVideos/{sequence}-raw.mp4"
    video_size = MOT_CAMPUS_VIDEO_SIZE if campus else MOT_VIDEO_SIZE
    video_sha = MOT_CAMPUS_VIDEO_SHA256 if campus else MOT_VIDEO_SHA256
    frame_count = 71 if campus else 179
    width, height, fps = 640, 480, 25
    if sequence in TRANSFER_SEQUENCES:
        video_size, video_sha, frame_count, width, height, fps, split = TRANSFER_SEQUENCES[sequence]
    gt_entry = f"MOT15Labels/train/{sequence}/gt/gt.txt"
    info_entry = f"MOT15Labels/train/{sequence}/seqinfo.ini"
    output_dir.mkdir(parents=True, exist_ok=True)
    video = output_dir / f"{sequence}-raw.mp4"
    archive = output_dir / "MOT15Labels.zip"
    if download:
        download_fixed(video_url, video, video_size, video_sha)
        download_fixed(MOT_LABELS_URL, archive, MOT_LABELS_SIZE, MOT_LABELS_SHA256)
    video_raw = bounded_bytes(video, video_size)
    archive_raw = bounded_bytes(archive, MOT_LABELS_SIZE)
    if len(video_raw) != video_size or len(archive_raw) != MOT_LABELS_SIZE:
        raise ValueError("MOT assets do not have the pinned official sizes")
    if sha256(video_raw) != video_sha or sha256(archive_raw) != MOT_LABELS_SHA256:
        raise ValueError("MOT assets do not match the pinned official SHA256 identities")
    with zipfile.ZipFile(archive) as zipped:
        if len(zipped.namelist()) != len(set(zipped.namelist())):
            raise ValueError("Duplicate ZIP entry names")
        info = zipped.getinfo(gt_entry)
        seq = zipped.getinfo(info_entry)
        if not 0 < info.file_size <= 512_000 or not 0 < seq.file_size <= 4096:
            raise ValueError("Unexpected MOT15 label entry size")
        gt_raw, seq_raw = zipped.read(info), zipped.read(seq)
    metadata = configparser.ConfigParser()
    metadata.read_string(seq_raw.decode("utf-8-sig"))
    section = metadata["Sequence"]
    if (section.get("name"), section.getint("seqLength"), section.getint("imWidth"),
            section.getint("imHeight"), section.getint("frameRate")) != (sequence, frame_count, width, height, fps):
        raise ValueError("MOT15 sequence metadata differs from the pinned sequence")
    gt, ignored = parse_mot_gt(gt_raw, frame_count)
    if not gt:
        raise ValueError("MOT15 sequence has no considered GT boxes")
    save_new_or_identical(output_dir / "gt.txt", gt_raw)
    save_new_or_identical(output_dir / "seqinfo.ini", seq_raw)
    manifest = {"version": 1, "kind": "video", "dataset": f"mot15-{sequence.lower()}-reencoded-raw-v1",
        "source": "https://motchallenge.net/data/MOT15/", "video_source": video_url,
        "labels_source": MOT_LABELS_URL, "archive_gt_entry": gt_entry, "archive_seqinfo_entry": info_entry,
        "video": video.name, "video_sha256": sha256(video_raw), "labels_zip_sha256": sha256(archive_raw),
        "ground_truth_file": "gt.txt", "ground_truth_sha256": sha256(gt_raw), "seqinfo_sha256": sha256(seq_raw),
        "coordinate_basis": "0-based xyxy, converted from MOT15 1-based xywh by subtracting 1 from x/y only; no clipping",
        "source_fps": fps, "width": width, "height": height, "frames": frame_count, "classes": ["person"],
        "ignored_gt_rows": ignored, "ground_truth": gt,
        "limitations": ["Official reencoded preview MP4, not the original challenge JPEG sequence.",
            "MOT15 training sequence used only for an exploratory baseline; not held-out test evaluation.",
            "Column 7 controls GT inclusion; world coordinates in columns 8..10 are not class or visibility.",
            "MOT materials historically publish CC BY-NC-SA 3.0; confirm current terms and attribution before reuse."]}
    if sequence in TRANSFER_SEQUENCES:
        protocol_path = Path(__file__).resolve().parent.parent / "configs/tracking-transfer-v1.json"
        protocol_raw = bounded_bytes(protocol_path, 64 * 1024)
        protocol = json.loads(protocol_raw)
        if (protocol.get("version") != 1 or protocol.get("protocol") != "tracking-transfer-v1"
                or sequence not in protocol.get(f"{split}_sequences", [])):
            raise ValueError("Sequence split disagrees with the frozen transfer protocol")
        manifest["project_split"] = split
        manifest["evaluation_protocol"] = protocol
        manifest["evaluation_protocol_sha256"] = sha256(protocol_raw)
        manifest["limitations"][1] = "Project-level unseen sequence; public MOT15 training data, not the official hidden test. No guarantee against pretrained data overlap."
    save_manifest(output_dir / "quality-manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    coco = commands.add_parser("coco", help="Reuse full COCO scenes selected by the crop-search manifest")
    coco.add_argument("annotations", type=Path)
    coco.add_argument("crop_manifest", type=Path)
    coco.add_argument("output", type=Path)
    mot = commands.add_parser("mot", help="Prepare a bounded, checksum-pinned MOT15 sequence")
    mot.add_argument("output_dir", type=Path)
    mot.add_argument("--download", action="store_true", help="Download the two pinned official HTTPS assets")
    mot.add_argument("--sequence", choices=("TUD-Stadtmitte", "TUD-Campus", *TRANSFER_SEQUENCES), default="TUD-Stadtmitte")
    args = parser.parse_args()
    if args.command == "coco":
        manifest = prepare_coco(args.annotations, args.crop_manifest, args.output)
        print(f"Prepared {len(manifest['images'])} scenes / {len(manifest['classes'])} classes / "
              f"{sum(len(image['ground_truth']) for image in manifest['images'])} GT boxes: {args.output}")
    else:
        manifest = prepare_mot(args.output_dir, args.download, args.sequence)
        identities = {box['id'] for box in manifest['ground_truth']}
        print(f"Prepared {manifest['frames']} frames / {len(identities)} identities / "
              f"{len(manifest['ground_truth'])} GT boxes / {manifest['ignored_gt_rows']} ignored rows: "
              f"{args.output_dir / 'quality-manifest.json'}")


if __name__ == "__main__":
    main()
