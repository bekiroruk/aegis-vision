"""Real native YOLO inference; optional end-to-end Redis submission/worker/result.

Uses the upstream bus.jpg reference (not a quality benchmark). No mock model.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import uuid


def run(args, expected=0):
    result = subprocess.run([str(x) for x in args], capture_output=True, text=True,
                            encoding="utf-8", timeout=120)
    if result.returncode != expected:
        raise AssertionError(f"Exit {result.returncode}, expected {expected}: {result.stderr}")
    return json.loads(result.stdout) if result.stdout.strip() else None


def check_detection(result, request):
    assert result["context"] == request["context"]
    assert result["image_sha256"] == request["payload"]["sha256"]
    assert result["width"] == 810 and result["height"] == 1080
    labels = [d["label"] for d in result["detections"]]
    assert labels.count("person") >= 3 and "bus" in labels, labels
    for d in result["detections"]:
        x1, y1, x2, y2 = d["xyxy"]
        assert 0 <= x1 < x2 <= result["width"] and 0 <= y1 < y2 <= result["height"]
        assert .35 <= d["confidence"] <= 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("worker", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument("media", type=Path)
    parser.add_argument("--queue", type=Path)
    parser.add_argument("--port", default=16379, type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    worker, model, media = args.worker.resolve(), args.model.resolve(), args.media.resolve()
    request = run([worker, "request", model, media, "bus.jpg"])
    assert request["payload"]["sha256"] == hashlib.sha256((media / "bus.jpg").read_bytes()).hexdigest()
    direct = run([worker, "analyze", model, media, "bus.jpg"])
    check_detection(direct, request)
    assert direct["model_sha256"] == hashlib.sha256(model.read_bytes()).hexdigest()
    for bad in ["../bus.jpg", "/bus.jpg", "C:/bus.jpg", "a\\bus.jpg", "bus.jpg:secret", "missing.jpg"]:
        run([worker, "request", model, media, bad], expected=1)
    # Canonical containment rejects a symlink pointing outside the media root.
    with tempfile.TemporaryDirectory(prefix="aegis-worker-test-") as tmp:
        root = Path(tmp)
        try:
            (root / "escape.jpg").symlink_to(media / "bus.jpg")
        except OSError:
            pass  # Windows may require symlink privilege; Linux CI exercises this.
        else:
            run([worker, "request", model, root, "escape.jpg"], expected=1)
        assert run([worker, "request", model, media, "bus.jpg"])["context"] == request["context"]
    evidence = {"direct_inference": direct, "redis_verified": False}
    if args.queue:
        queue = args.queue.resolve()
        namespace = "image-" + uuid.uuid4().hex

        def call(command):
            p = subprocess.run([str(queue), str(args.port), namespace, request["context"]],
                               input=json.dumps(command), capture_output=True, text=True,
                               encoding="utf-8", timeout=15, check=True)
            return json.loads(p.stdout)

        def once(expected=0):
            return run([worker, "run", args.port, namespace, "image-worker", model, media,
                        "--once", "3000"], expected)

        call({"op": "submit", "id": "real-image", "payload": request["payload"]})
        assert once()["outcome"] == "completed"
        done = call({"op": "get", "id": "real-image"})
        assert done["state"] == "succeeded" and done["attempts"] == 1
        check_detection(done["result"], request)
        assert once()["outcome"] == "empty"  # No duplicate terminal execution.
        # Changed content, path traversal and unsupported jobs fail closed.
        for i, payload in enumerate([
            {**request["payload"], "sha256": "0" * 64},
            {**request["payload"], "image": "../bus.jpg"},
            {"type": "unsupported"},
        ]):
            job_id = f"invalid-{i}"
            call({"op": "submit", "id": job_id, "payload": payload})
            assert once(expected=1)["outcome"] == "failed"
            failed = call({"op": "get", "id": job_id})
            assert failed["state"] == "failed" and failed["result"]["error"] == "analysis_failed"
        call({"op": "submit", "id": "cancelled", "payload": request["payload"]})
        call({"op": "cancel", "id": "cancelled"})
        assert once()["outcome"] == "empty"
        assert call({"op": "get", "id": "cancelled"})["state"] == "cancelled"
        evidence.update(redis_verified=True, redis_job=done)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    print(json.dumps({"passed": True, "redis_verified": evidence["redis_verified"],
                      "detections": len(direct["detections"]), "labels": [d["label"] for d in direct["detections"]]}))


if __name__ == "__main__":
    main()
