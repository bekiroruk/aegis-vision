"""Validate resolved deployment profiles without starting a Docker daemon."""
import json
from pathlib import Path
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]


def config(isolated=False):
    command = ["docker", "compose", "-f", "compose.gpu.yaml"]
    if isolated:
        command += ["-f", "compose.gpu.isolated.yaml"]
    result = subprocess.run(command + ["--profile", "tools", "config", "--format", "json"],
                            cwd=ROOT, check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


class ComposeTests(unittest.TestCase):
    def test_native_profile_keeps_loopback_host_network(self):
        data = config()
        for name in ("service", "qdrant-gpu"):
            self.assertEqual(data["services"][name]["network_mode"], "host")

    def test_isolated_profile_shares_private_loopback_without_ports(self):
        data = config(True)
        self.assertEqual(data["name"], "aegis-gpu-isolated")
        service, qdrant = (data["services"][n] for n in ("service", "qdrant-gpu"))
        self.assertEqual(service["network_mode"], "service:qdrant-gpu")
        self.assertNotIn("network_mode", qdrant)
        self.assertTrue(data["networks"]["gpu-test"]["internal"])
        self.assertEqual(qdrant["environment"]["QDRANT__SERVICE__HOST"], "127.0.0.1")
        for item in data["services"].values():
            self.assertFalse(item.get("ports"))
        mounts = {item["target"]: item for item in service["volumes"]}
        self.assertTrue(mounts["/state"]["source"].endswith("isolated-state"))
        self.assertTrue(mounts["/media"]["source"].endswith("isolated-media"))

    def test_both_profiles_keep_gpu_and_runtime_restrictions(self):
        for isolated in (False, True):
            with self.subTest(isolated=isolated):
                data = config(isolated)
                service = data["services"]["service"]
                self.assertTrue(service["read_only"])
                self.assertTrue(service["init"])
                self.assertNotEqual(service["user"].split(":")[0], "0")
                self.assertIn("ALL", service["cap_drop"])
                self.assertIn("no-new-privileges:true", service["security_opt"])
                devices = service["deploy"]["resources"]["reservations"]["devices"]
                self.assertEqual(devices[0]["driver"], "nvidia")
                self.assertIn("gpu", devices[0]["capabilities"])
                mounts = {item["target"]: item for item in service["volumes"]}
                for target in ("/models", "/engines"):
                    self.assertTrue(mounts[target]["read_only"])
                for mount in mounts.values():
                    self.assertFalse(mount.get("bind", {}).get("create_host_path", False))
                self.assertEqual(data["services"]["engine-build"]["network_mode"], "none")


if __name__ == "__main__":
    unittest.main()
