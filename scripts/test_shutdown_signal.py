"""POSIX parent-to-child SIGTERM integration; no models/GPU/downloads."""
import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys

if os.name != "posix":
    raise SystemExit("This test requires POSIX signals; Windows uses native raise-based tests")
exe = Path(sys.argv[1]).resolve(strict=True)
with subprocess.Popen([str(exe), "--external"], stdout=subprocess.PIPE,
                      stderr=subprocess.STDOUT) as child:
    try:
        with selectors.DefaultSelector() as events:
            events.register(child.stdout, selectors.EVENT_READ)
            if not events.select(timeout=10):
                raise RuntimeError("Child never reached HTTP-ready active-work state")
            line = child.stdout.readline()
            if not line.startswith(b"READY "):
                raise RuntimeError(f"Unexpected child state: {line!r}")
        child.send_signal(signal.SIGTERM)
        output = child.communicate(timeout=10)[0].decode(errors="replace")
        print(output)
        if child.returncode != 0 or "same-ID replay" not in output:
            raise RuntimeError(f"Signal shutdown/replay failed: exit={child.returncode}")
    finally:
        if child.poll() is None:
            child.kill()  # Only the fixture child created above.
            child.wait(timeout=5)
