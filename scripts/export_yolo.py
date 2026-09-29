"""One-time model preparation. Runtime inference is native C++/OpenCV.

Use an isolated environment with ultralytics==8.3.252 and onnx==1.19.1.
The upstream pretrained YOLOv8n weights are downloaded by Ultralytics if absent.
"""
import hashlib
import json
import os
from pathlib import Path

# Keep helper-generated settings inside the local scratch area.
config_dir = Path(__file__).resolve().parents[1] / "work" / "ultralytics-config"
config_dir.mkdir(parents=True, exist_ok=True)
os.environ["YOLO_CONFIG_DIR"] = str(config_dir)
os.environ["YOLO_AUTOINSTALL"] = "false"

import onnx
import torch
import ultralytics
from ultralytics import YOLO


def main():
    directory = Path(__file__).resolve().parents[1] / "artifacts" / "models"
    directory.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(directory / "yolov8n.pt"))
    exported = Path(model.export(format="onnx", imgsz=640, batch=1, opset=12,
                                 dynamic=False, simplify=False, half=False, nms=False, device="cpu"))
    onnx.checker.check_model(onnx.load(exported))
    metadata = {
        "model": "YOLOv8n COCO detection",
        "source": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt",
        "license_information": "https://www.ultralytics.com/license",
        "ultralytics": ultralytics.__version__, "torch": torch.__version__, "onnx": onnx.__version__,
        "export": {"imgsz": 640, "batch": 1, "opset": 12, "dynamic": False,
                   "simplify": False, "half": False, "nms": False},
        "onnx_sha256": hashlib.sha256(exported.read_bytes()).hexdigest(),
        "weights_sha256": hashlib.sha256((directory / "yolov8n.pt").read_bytes()).hexdigest(),
        "classes": model.names,
    }
    (directory / "model-manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Validated model: {exported}")


if __name__ == "__main__":
    main()
