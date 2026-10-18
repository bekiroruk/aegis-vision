"""Opt-in real GPU test of the already-running isolated Compose stack.

Creates indexing/search jobs. --restart-test stops ONLY this profile's service
while its own job runs, then starts it again. No build/download/delete/reset.
"""
import argparse
import base64
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
COMPOSE = ["docker", "compose", "-f", "compose.gpu.yaml", "-f", "compose.gpu.isolated.yaml"]
HTTP = """
import base64,json,sys,urllib.request
args=json.load(sys.stdin)
data=None if args['body'] is None else json.dumps(args['body']).encode()
request=urllib.request.Request('http://127.0.0.1:8093'+args['path'],data=data,
    headers={'Content-Type':'application/json'},method=args['method'])
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
with opener.open(request,timeout=10) as response:
    raw=response.read(4*1024*1024+1)
    if len(raw)>4*1024*1024: raise ValueError('Response too large')
    print(json.dumps({'base64':base64.b64encode(raw).decode()} if args['binary'] else json.loads(raw)))
"""


def command(args, *, data=None, timeout=60):
    result = subprocess.run(args, cwd=ROOT, input=data, text=True, encoding="utf-8",
                            capture_output=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f"Command failed ({result.returncode}): {result.stderr[-4000:]}")
    return result.stdout.strip()


def request(path, body=None, binary=False):
    return json.loads(command(COMPOSE + ["exec", "-T", "service", "python3", "-c", HTTP],
        data=json.dumps(dict(path=path, body=body, binary=binary,
                             method="GET" if body is None else "POST"))))


def require_gpu(data):
    backend = data.get("detector_backend", {})
    if (data.get("status") != "ready" or data.get("storage_error") or
            not data.get("history_persistent") or
            any(backend.get(k) != v for k, v in
                {"engine": "tensorrt", "precision": "fp16-mixed", "device": "cuda:0"}.items())):
        raise RuntimeError("Expected healthy persistent native FP16 GPU service")
    if len(backend.get("engine_sha256", "")) != 64:
        raise RuntimeError("Missing engine identity")


def ready():
    deadline = time.monotonic() + 180
    while True:
        try:
            health = request("/api/health")
            require_gpu(health)
            return health
        except (subprocess.SubprocessError, ValueError, RuntimeError):
            if time.monotonic() >= deadline:
                raise
            time.sleep(2)


def job_until(job_id, *, running=False):
    deadline = time.monotonic() + 240
    while True:
        job = request("/api/jobs/" + job_id)
        if running and job["state"] == "running" and job.get("progress", {}).get("indexed_items", 0) > 0:
            return job
        if job["state"] == "succeeded" and not running:
            return job
        if job["state"] in ("failed", "cancelled", "succeeded"):
            raise RuntimeError("Unexpected job state: " + json.dumps(job))
        if time.monotonic() >= deadline:
            raise TimeoutError("Job timed out: " + job_id)
        time.sleep(1)


def require_idle(except_id=None):
    jobs = request("/api/jobs")["jobs"]
    if any(j["state"] in ("queued", "running") and j["id"] != except_id for j in jobs):
        raise RuntimeError("Unrelated work exists; refusing to interrupt it")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--restart-test", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "profile": "aegis-gpu-isolated"}

    def save():
        (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    try:
        container_id = command(COMPOSE + ["ps", "-q", "service"])
        if not container_id or "\n" in container_id:
            raise RuntimeError("Exactly one isolated service must already be running")
        info = json.loads(command(["docker", "inspect", container_id]))[0]
        if info["Config"]["Labels"].get("com.docker.compose.project") != "aegis-gpu-isolated":
            raise RuntimeError("Wrong Compose project")
        if not info["HostConfig"]["ReadonlyRootfs"]:
            raise RuntimeError("Service root filesystem must be read-only")
        uid = command(COMPOSE + ["exec", "-T", "service", "id", "-u"])
        if uid == "0":
            raise RuntimeError("Service must not run as root")
        mounts = {m["Destination"]: m for m in info["Mounts"]}
        if any(mounts[p]["RW"] for p in ("/models", "/engines")):
            raise RuntimeError("Models and engines must be mounted read-only")
        report.update(image=info["Image"], uid=uid, health=ready())
        require_idle()
        accepted = request("/api/jobs", {"type": "index_video", "path": "pedestrians.mp4",
                                         "stride": 10, "max_frames": 30})
        report["index_accepted"] = accepted
        save()
        indexed = job_until(accepted["id"])
        if indexed["result"]["indexed_items"] < 1 or indexed["result"]["decoded_frames"] != 30:
            raise RuntimeError("Expected GPU video indexing to produce objects from 30 frames")
        report["index"] = indexed
        accepted = request("/api/jobs", {"type": "search", "query": "a person walking on the street", "limit": 4})
        report["search_accepted"] = accepted
        save()
        found = job_until(accepted["id"])
        hits = found["result"]["results"]
        signature = "tensorrt-fp16-v1:" + report["health"]["detector_backend"]["engine_sha256"] + ":"
        if not hits or any(not h["metadata"]["detector_signature"].startswith(signature) for h in hits):
            raise RuntimeError("Search must return objects from this exact GPU engine")
        report["search"] = found
        preview = base64.b64decode(request("/api/preview/" + found["id"] + "/0.jpg", binary=True)["base64"])
        if len(preview) < 100 or not preview.startswith(b"\xff\xd8"):
            raise RuntimeError("Invalid search preview")
        (args.output / "preview.jpg").write_bytes(preview)
        save()
        if args.restart_test:
            require_idle()
            accepted = request("/api/jobs", {"type": "index_video", "path": "pedestrians.mp4",
                                             "stride": 3, "max_frames": 90})
            report["restart_accepted"] = accepted
            save()
            report["before_stop"] = job_until(accepted["id"], running=True)
            require_idle(except_id=accepted["id"])
            started = time.monotonic()
            command(COMPOSE + ["stop", "service"], timeout=60)
            report["stop_seconds"] = round(time.monotonic() - started, 3)
            state = json.loads(command(["docker", "inspect", container_id]))[0]["State"]
            report["stopped_state"] = state
            save()
            if state["Running"] or state["ExitCode"] != 0 or state["OOMKilled"]:
                raise RuntimeError("Service did not exit cleanly; no graceful-shutdown claim")
            command(COMPOSE + ["start", "service"])
            report["health_after_restart"] = ready()
            recovered = job_until(accepted["id"])
            if recovered["attempts"] < 2 or recovered["result"]["indexed_items"] < 1:
                raise RuntimeError("Active job was not replayed under the same ID")
            report["recovered"] = recovered
            retained = request("/api/jobs/" + found["id"])
            if retained["state"] != "succeeded" or retained["result"] != found["result"]:
                raise RuntimeError("Completed search result changed across restart")
        report["status"] = "passed"
        print("PASS: native Linux GPU indexing, CLIP/Qdrant search, preview" +
              (", graceful stop and same-ID replay" if args.restart_test else ""), flush=True)
    except Exception as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
