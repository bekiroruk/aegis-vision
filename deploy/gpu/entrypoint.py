"""Container-only preflight and exec; model validation stays in the C++ runtime."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import urllib.request


def preflight(root=Path("/")):
    required = (
        "models/yolov8n.onnx", "models/clip-vit-b32/manifest.json",
        "models/clip-vit-b32/vision.onnx", "models/clip-vit-b32/text.onnx",
        "models/clip-vit-b32/tokenizer.json",
        "engines/linux-fp16/manifest.json", "engines/linux-fp16/model.engine",
        "opt/aegis/build/aegisvision_server", "opt/aegis/web/index.html",
    )
    errors = [f"Missing/empty file: /{p}" for p in required
              if not (root / p).is_file() or (root / p).stat().st_size == 0]
    for name in ("media", "state"):
        path = root / name
        if not path.is_dir() or not os.access(path, os.W_OK | os.X_OK):
            errors.append(f"Missing/unwritable mount: /{name}")
    manifest = root / "engines/linux-fp16/manifest.json"
    if manifest.is_file():
        try:
            if manifest.stat().st_size > 65536:
                raise ValueError("Oversized engine manifest")
            data = json.loads(manifest.read_text(encoding="utf-8"))
            if not isinstance(data, dict) or data.get("precision") != "fp16-mixed":
                raise ValueError("Only FP16-mixed is approved for this service")
        except (ValueError, OSError) as exc:
            errors.append(f"Engine manifest: {exc}")
    if errors:
        raise ValueError("\n".join(errors))


def fetch_json(url):
    # Loopback checks must not go through HTTP_PROXY inherited by a container.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(url, timeout=3) as response:
        return json.loads(response.read(65536))


def healthy(data):
    if not isinstance(data, dict):
        return False
    backend = data.get("detector_backend")
    return (data.get("status") == "ready" and isinstance(backend, dict)
            and backend.get("engine") == "tensorrt" and backend.get("precision") == "fp16-mixed"
            and backend.get("device") == "cuda:0")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--health", action="store_true")
    args = parser.parse_args()
    if args.health:
        if not healthy(fetch_json("http://127.0.0.1:8093/api/health")):
            raise ValueError("GPU service is not ready")
        return
    preflight()
    if args.check:
        print("Mount preflight passed; C++ model/GPU validation still required")
        return
    for attempt in range(15):
        try:
            fetch_json("http://127.0.0.1:16333/collections")
            break
        except (OSError, ValueError):
            if attempt == 14:
                raise RuntimeError("Qdrant not ready on loopback port 16333")
            time.sleep(2)
    executable = "/opt/aegis/build/aegisvision_server"
    os.execv(executable, [executable, "/opt/aegis/deploy/search.toml",
        "/opt/aegis/deploy/detector.toml", "/media", "/opt/aegis/web", "8093",
        "/state/jobs.sqlite", "--trt-bundle", "/engines/linux-fp16"])


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError) as error:
        print(f"GPU container: {error}", file=sys.stderr)
        sys.exit(1)
