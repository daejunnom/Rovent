"""Synthetic source observations; not evidence that a reader is safe."""
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from managed_storage import StorageError
from storage_consumer_broker_profile import ObservedReaderProfile, ReviewedReaderContract


@unittest.skipUnless(sys.platform == "linux", "Linux source descriptor required")
class ObservedReaderProfileTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "scripts").mkdir()
        self.script = self.root / "scripts" / "rovent_v3_parent.py"
        self.script.write_text("# synthetic only\n", encoding="utf-8")
        self.manager = SimpleNamespace(source=self.root, _assert_owner=lambda: None)

    def test_dynamic_source_observation_and_drift(self):
        command = [sys.executable, str(self.script), "--synthetic"]
        profile = ObservedReaderProfile(self.manager, command)
        self.addCleanup(profile.close)
        command.append("--later")
        self.assertEqual(profile.command, (sys.executable, str(self.script), "--synthetic"))
        self.assertTrue(profile.unchanged())
        self.assertEqual(profile.observation()["scope"], "observed_script_only")
        self.assertEqual(profile.observation()["imports"], "unknown")
        self.script.write_text("# changed synthetic content\n", encoding="utf-8")
        self.assertFalse(profile.unchanged())

    def test_unreviewed_direct_script_is_rejected(self):
        other = self.root / "scripts" / "other.py"
        other.write_text("# synthetic only\n", encoding="utf-8")
        with self.assertRaises(StorageError):
            ObservedReaderProfile(self.manager, [sys.executable, str(other)])

    def test_relative_interpreter_is_rejected(self):
        with self.assertRaises(StorageError):
            ObservedReaderProfile(self.manager, ["python", str(self.script)])

    def test_path_replacement_is_reported_as_drift(self):
        profile = ObservedReaderProfile(self.manager, [sys.executable, str(self.script)])
        self.addCleanup(profile.close)
        replacement = self.root / "scripts" / "replacement"
        replacement.write_bytes(self.script.read_bytes())
        os.replace(replacement, self.script)
        self.assertFalse(profile.unchanged())

    def test_contract_requires_separate_review_and_matching_observed_digest(self):
        profile = ObservedReaderProfile(self.manager, [sys.executable, str(self.script)])
        self.addCleanup(profile.close)
        digest = profile.observation()["source_sha256"]
        with self.assertRaises(StorageError):
            ReviewedReaderContract(profile, reviewed_sha256=digest,
                                   descriptor_content_only=False, derived_content_after_ack=True)
        with self.assertRaises(StorageError):
            ReviewedReaderContract(profile, reviewed_sha256="0" * 64,
                                   descriptor_content_only=True, derived_content_after_ack=True)
        contract = ReviewedReaderContract(profile, reviewed_sha256=digest,
                                          descriptor_content_only=True, derived_content_after_ack=True)
        self.assertTrue(contract.allows_unknown_abandon())
        self.script.write_text("# changed\n", encoding="utf-8")
        self.assertFalse(contract.allows_unknown_abandon())


if __name__ == "__main__":
    unittest.main()
