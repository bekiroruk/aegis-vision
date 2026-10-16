"""Before/after Redis restart evidence via native CLI; fixture namespace only."""
import json
from pathlib import Path
import subprocess
import sys

exe, port, queue, mode = sys.argv[1:]
if not queue.startswith("restart-test-") or mode not in ("write", "read"):
    raise SystemExit("Use a dedicated restart-test- namespace and write/read mode")

def call(op, **fields):
    r = subprocess.run([str(Path(exe).resolve(strict=True)), port, queue, "restart-fixture"],
        input=json.dumps(dict(op=op, **fields)), text=True, capture_output=True, timeout=5, check=True)
    return json.loads(r.stdout)

if mode == "write":
    call("submit", id="done", payload={"type": "fixture"})
    done = call("claim", owner="finisher", lease_ms=10000)
    call("complete", id="done", owner="finisher", token=done["token"], result={"ok": True})
    for id in ("active", "waiting", "cancelled"):
        call("submit", id=id, payload={"type": "fixture"})
    old = call("claim", owner="old", lease_ms=100)
    if old["id"] != "active": raise RuntimeError("Unexpected fixture order")
    call("cancel", id="cancelled")
    # Waiting remains queued; active owner disappears with the short-lived process.
    print("Durable active, queued and cancelled fixtures written")
else:
    if call("get", id="cancelled")["state"] != "cancelled": raise RuntimeError("Cancellation lost")
    done = call("get", id="done")
    if done["state"] != "succeeded" or done["attempts"] != 1: raise RuntimeError("Completed result lost or replayed")
    claimed = []
    for _ in range(2):
        job = call("claim", owner="new", lease_ms=10000)
        if job is None: raise RuntimeError("Restart lost work")
        claimed.append(job["id"])
        call("complete", id=job["id"], owner="new", token=job["token"], result={"ok": True})
    if sorted(claimed) != ["active", "waiting"]: raise RuntimeError("Wrong recovered population")
    if call("get", id="active")["attempts"] != 2: raise RuntimeError("Active lease did not recover")
    if call("claim", owner="new", lease_ms=10000) is not None: raise RuntimeError("Duplicate replay")
    print("Redis restart preserved jobs, expired-owner recovery and cancellation")
