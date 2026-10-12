"""Pinned GPU fine-tuning preparation for C++ ONNX inference. Never deploys a model."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
from prepare_segmentation_training import split_names

ROOT = Path(__file__).resolve().parents[1]
os.environ["YOLO_CONFIG_DIR"] = str(ROOT / "work/ultralytics-config")
os.environ["YOLO_AUTOINSTALL"] = "false"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
WEIGHTS_SHA = "a7cd8f929e1903d78a12a48efecab430209f18dc46cb96c3599a5980c63c423c"


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def audit_data(root):
    """Fail before training on any label/image mutation, split leakage, or path escape."""
    root = root.resolve()
    manifests = {s: json.loads((root/f"{s}-manifest.json").read_text(encoding="utf-8"))
                 for s in ("train", "val", "test")}
    frozen = split_names([Path(e["path"]).stem for m in manifests.values() for e in m["images"]])
    fingerprints, identities, hashes = set(), set(), {}
    for split, count in {"train": 80, "val": 16, "test": 74}.items():
        manifest_path = root / f"{split}-manifest.json"
        manifest = manifests[split]
        if sorted(Path(e["path"]).stem for e in manifest["images"]) != frozen[split]:
            raise ValueError("Frozen split membership changed")
        hashes[split] = digest(manifest_path)
        if manifest["annotation_sha256"] != digest(root / "annotations.json") or len(manifest["images"]) != count:
            raise ValueError("Dataset size or annotation hash changed")
        for image in manifest["images"]:
            path = (root / image["path"]).resolve()
            if path.parent != root / "images" / split or path.suffix != ".png":
                raise ValueError("Unexpected image path or split")
            if split == "test" and not path.stem.startswith("Fudan") or split != "test" and not path.stem.startswith("Penn"):
                raise ValueError("Location split changed")
            label = (root / "labels" / split / f"{path.stem}.txt").resolve()
            if label.parent != root / "labels" / split:
                raise ValueError("Label path escapes split")
            if digest(path) != image["sha256"] or digest(label) != image["label_sha256"]:
                raise ValueError("Training image or label changed")
            if image["sha256"] in fingerprints or image["coco_image_id"] in identities:
                raise ValueError("Image leakage between splits")
            fingerprints.add(image["sha256"]); identities.add(image["coco_image_id"])
        actual_images = {p.name for p in (root / "images" / split).iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}}
        if actual_images != {Path(e["path"]).name for e in manifest["images"]}:
            raise ValueError("Unmanifested training images")
    yaml = json.loads((root / "dataset.yaml").read_text(encoding="utf-8"))
    if Path(yaml["path"]).resolve() != root or any(yaml[s] != f"images/{s}" for s in ("train", "val", "test")):
        raise ValueError("Dataset config changed")
    categories = sorted(json.loads((root/"annotations.json").read_text(encoding="utf-8"))["categories"], key=lambda c: c["id"])
    if yaml["names"] != {str(i): c["name"] for i, c in enumerate(categories)} or len(categories) != 80:
        raise ValueError("Class mapping changed")
    hashes["dataset_yaml"] = digest(root / "dataset.yaml")
    return hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path); parser.add_argument("output", type=Path)
    parser.add_argument("--weights", type=Path, default=ROOT/"artifacts/models/yolov8n-seg/yolov8n-seg.pt")
    args = parser.parse_args()
    data = args.dataset.resolve(); hashes = audit_data(data)
    if digest(args.weights) != WEIGHTS_SHA:
        raise ValueError("Base checkpoint must match the pinned YOLOv8n-seg weights")
    import torch
    import ultralytics
    import onnx
    from ultralytics import YOLO
    if ultralytics.__version__ != "8.3.252" or torch.__version__ != "2.9.1+cu130":
        raise RuntimeError("Use ultralytics 8.3.252 and torch 2.9.1+cu130 in the existing export environment")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for this fixed experiment")
    args.output.mkdir(parents=True, exist_ok=False)
    options = dict(data=str(data/"dataset.yaml"), epochs=10, batch=4, imgsz=640, device=0,
        workers=0, seed=20261012, deterministic=True, optimizer="AdamW", lr0=.0001, lrf=.1,
        weight_decay=.0005, freeze=10, amp=False, mosaic=0., mixup=0., copy_paste=0.,
        scale=.1, translate=.05, close_mosaic=0, patience=10, cache=False, plots=True,
        project=str(args.output.resolve()), name="training", exist_ok=False, save=True, val=True)
    protocol = {"experiment": "penn-fudan-person-finetune-v1", "options": options,
        "selection": "best.pt chosen only by the 16-image Penn validation split; Fudan test never selects weights",
        "software": {"torch": torch.__version__, "ultralytics": ultralytics.__version__, "onnx": onnx.__version__},
        "gpu": torch.cuda.get_device_name(0), "weights_sha256": WEIGHTS_SHA, "data_hashes": hashes,
        "deployment": "Candidate only. Do not replace the general 80-class production model based on person-only results."}
    (args.output/"protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    model = YOLO(str(args.weights))
    model.train(**options)
    best = args.output/"training/weights/best.pt"
    candidate = YOLO(str(best))
    exported = Path(candidate.export(format="onnx", imgsz=640, batch=1, opset=12,
        dynamic=False, simplify=False, half=False, nms=False, device="cpu"))
    onnx.checker.check_model(onnx.load(exported))
    if audit_data(data) != hashes or digest(args.weights) != WEIGHTS_SHA:
        raise ValueError("Inputs changed during training")
    target = args.output/"candidate.onnx"
    shutil.copyfile(exported, target)
    (args.output/"candidate-manifest.json").write_text(json.dumps({
        "task": "person-domain instance segmentation", "weights_sha256": digest(best),
        "onnx_sha256": digest(target), "protocol_sha256": digest(args.output/"protocol.json"),
        "input": [1,3,640,640], "class_head": "COCO 80 classes preserved; only person fine-tuned",
        "deployment_approved": False}, indent=2), encoding="utf-8")
    print(f"Candidate for C++ evaluation: {target}")


if __name__ == "__main__":
    main()
