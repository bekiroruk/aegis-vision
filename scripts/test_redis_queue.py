"""Real Redis integration through separate native CLI processes; no fake server."""
import concurrent.futures
import json
from pathlib import Path
import subprocess
import sys
import time
import unittest
import uuid

EXE = str(Path(sys.argv.pop(1)).resolve(strict=True))
PORT = sys.argv.pop(1)


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.queue = "test-" + uuid.uuid4().hex

    def call(self, op, context="fixture-v1", **fields):
        result = subprocess.run([EXE, PORT, self.queue, context],
            input=json.dumps(dict(op=op, **fields)), text=True, capture_output=True, timeout=5)
        if result.returncode:
            raise RuntimeError(result.stderr)
        return json.loads(result.stdout)

    def submit(self, id="j"):
        return self.call("submit", id=id, payload={"type": "fixture", "unicode": "Türkçe"})

    def claim(self, owner="a", lease=10000):
        return self.call("claim", owner=owner, lease_ms=lease)

    def finish(self, job, **override):
        fields = dict(id=job["id"], owner=job["owner"], token=job["token"], result={"done": True})
        fields.update(override)
        return self.call("complete", **fields)

    def test_separate_workers_only_one_claim(self):
        self.submit()
        with concurrent.futures.ThreadPoolExecutor(8) as pool:
            claims = list(pool.map(lambda i: self.claim(str(i)), range(8)))
        winners = [x for x in claims if x is not None]
        self.assertEqual(len(winners), 1)
        self.assertEqual(self.finish(winners[0])["state"], "succeeded")
        self.assertIsNone(self.finish(winners[0]))

    def test_idempotent_submit_conflict_and_context(self):
        self.assertEqual(self.submit(), self.submit())
        with self.assertRaisesRegex(RuntimeError, "ID_CONFLICT"):
            self.call("submit", id="j", payload={"different": True})
        with self.assertRaisesRegex(RuntimeError, "CONTEXT_MISMATCH"):
            self.call("get", context="changed-model", id="j")
        self.assertEqual(self.call("get", id="j")["state"], "queued")

    def test_expired_worker_cannot_publish_or_renew(self):
        self.submit()
        old = self.claim(lease=150)
        time.sleep(.25)
        self.assertIsNone(self.finish(old))
        self.assertIsNone(self.call("renew", id="j", owner="a", token=old["token"], lease_ms=10000))
        new = self.claim("b")
        self.assertGreater(int(new["token"]), int(old["token"]))
        self.assertEqual(new["attempts"], 2)
        self.assertIsNone(self.finish(old))
        self.assertEqual(self.finish(new)["state"], "succeeded")

    def test_heartbeat_owner_token_and_cancellation(self):
        self.submit()
        job = self.claim()
        self.assertIsNone(self.call("renew", id="j", owner="b", token=job["token"], lease_ms=10000))
        self.assertIsNone(self.finish(job, token="0"))
        renewed = self.call("renew", id="j", owner="a", token=job["token"], lease_ms=20000)
        self.assertGreaterEqual(renewed["deadline_ms"], job["deadline_ms"])
        self.assertEqual(self.call("cancel", id="j")["state"], "cancelled")
        self.assertIsNone(self.finish(job))
        self.assertIsNone(self.claim("b"))

    def test_queued_cancel_and_failed_result(self):
        self.submit("a"); self.submit("b")
        self.call("cancel", id="a")
        job = self.claim()
        self.assertEqual(job["id"], "b")
        result = self.call("fail", id="b", owner="a", token=job["token"], result={"error": "decode"})
        self.assertEqual(result["state"], "failed")
        self.assertEqual(self.call("cancel", id="b")["state"], "failed")

    def test_bounded_retry_and_empty_queue(self):
        self.assertIsNone(self.claim())
        self.assertIsNone(self.call("get", id="absent"))
        self.submit()
        for attempt in range(1, 4):
            self.assertEqual(self.claim(lease=100)["attempts"], attempt)
            time.sleep(.2)
        self.assertIsNone(self.claim())
        self.assertEqual(self.call("get", id="j")["state"], "failed")

    def test_capacity_includes_retained_terminal_jobs(self):
        for i in range(128): self.submit(str(i))
        with self.assertRaisesRegex(RuntimeError, "QUEUE_FULL"): self.submit("overflow")
        self.call("cancel", id="0")
        with self.assertRaisesRegex(RuntimeError, "QUEUE_FULL"): self.submit("still-full")
        self.assertEqual(self.submit("1")["state"], "queued")


if __name__ == "__main__": unittest.main()
