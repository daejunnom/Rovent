"""Synthetic identities only; no existing owner or credentials are read."""
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from storage_consumer_broker_identity import BrokerIdentityError, OriginalSupervisorIdentity
from managed_storage import SCHEMA


class SyntheticRoot:
    def __truediv__(self, child):
        return Path("/synthetic") / child

    def stat(self):
        return SimpleNamespace(st_dev=12, st_ino=34)


class SyntheticRegistry:
    def __init__(self):
        self.valid = True

    def _check_identity(self):
        if not self.valid:
            raise RuntimeError("registry replaced")


class SupervisorIdentityTests(unittest.TestCase):
    def setUp(self):
        self.identity = OriginalSupervisorIdentity.__new__(OriginalSupervisorIdentity)
        self.identity.root = SyntheticRoot()
        self.identity.source = Path("/synthetic/source")
        self.identity.token = "synthetic-token"
        self.identity.parent_pid = 123
        self.identity.uid, self.identity.gid = 456, 789
        self.identity.host, self.identity.boot, self.identity.birth = "host", "boot", "100"
        self.identity.registry = SyntheticRegistry()
        self.record = {"schema": SCHEMA, "pid": 123, "token": "synthetic-token",
                       "source": str(self.identity.source), "root_identity": [12, 34],
                       "lifetime": {"platform": "linux", "host": "host", "boot": "boot",
                                    "process_birth": "100"}}
        self.parent = patch("storage_consumer_broker_identity.os.getppid", return_value=123)
        self.birth = patch("storage_consumer_broker_identity._process_birth", return_value="100")
        self.record_read = patch("managed_storage.read_record", side_effect=lambda _: dict(self.record))
        self.parent.start()
        self.birth.start()
        self.record_read.start()
        self.addCleanup(self.parent.stop)
        self.addCleanup(self.birth.stop)
        self.addCleanup(self.record_read.stop)

    def test_original_peer_and_owner_match(self):
        self.identity.check_peer(123, 456, 789)

    def test_wrong_peer_or_replaced_owner_fails(self):
        with self.assertRaises(BrokerIdentityError):
            self.identity.check_peer(124, 456, 789)
        self.record["root_identity"] = [12, 35]
        with self.assertRaises(BrokerIdentityError):
            self.identity.check_peer(123, 456, 789)

    def test_stale_birth_and_replaced_registry_fail(self):
        self.record["lifetime"]["process_birth"] = "different"
        with self.assertRaises(BrokerIdentityError):
            self.identity.check_peer(123, 456, 789)
        self.record["lifetime"]["process_birth"] = "100"
        self.identity.registry.valid = False
        with self.assertRaises(RuntimeError):
            self.identity.check_peer(123, 456, 789)


if __name__ == "__main__":
    unittest.main()
