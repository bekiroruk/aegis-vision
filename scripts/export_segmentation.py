"""Pinned one-time preparation; inference runs in C++, not Python."""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ["YOLO_CONFIG_DIR"] = str(ROOT / "work/ultralytics-config")
os.environ["YOLO_AUTOINSTALL"] = "false"

def main():
    import onnx
    import ultralytics
    from ultralytics import YOLO
    if ultralytics.__version__ != "8.3.252":
        raise RuntimeError("Use ultralytics==8.3.252 from the existing export environment")
    target = ROOT / "artifacts/models/yolov8n-seg"
    target.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(target / "yolov8n-seg.pt"))
    path = Path(model.export(format="onnx", imgsz=640, batch=1, opset=12,
                            dynamic=False, simplify=False, half=False, nms=False, device="cpu"))
    onnx.checker.check_model(onnx.load(path))
    manifest = {"task": "instance-segmentation", "ultralytics": ultralytics.__version__,
                "source": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n-seg.pt",
                "license": "https://www.ultralytics.com/license", "input": [1,3,640,640],
                "onnx_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "weights_sha256": hashlib.sha256((target/"yolov8n-seg.pt").read_bytes()).hexdigest()}
    (target/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print(path)

if __name__ == "__main__":
    main()
