import json
import io
import os
from pathlib import Path
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from managed_storage import StorageError
from storage_legacy_inventory import inspect_observation, observe
from storage_maintenance import main as maintenance_main


@unittest.skipUnless(sys.platform == "linux", "Linux filesystem observation")
class LegacyInventoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.inputs = self.base / "inputs"
        self.directory = self.inputs / "original-binaries-build-151"
        self.directory.mkdir(parents=True)
        self.registry = self.base / "registrations"
        self.registry.mkdir()
        self.output = self.registry / "inventory.json"
        self.binary = self.directory / "rz-rovent-trainer"
        self.binary.write_bytes(b"\x7fELFexample")

    def tearDown(self):
        self.temp.cleanup()

    def register(self, **kwargs):
        return observe(self.inputs, self.directory, self.output,
                       max_bytes=kwargs.pop("max_bytes", 1000), **kwargs)

    def test_snapshot_does_not_adopt_or_change_source(self):
        before = self.binary.read_bytes()
        record = self.register()
        self.assertEqual(json.loads(self.output.read_text()), record)
        self.assertEqual(self.binary.read_bytes(), before)
        self.assertFalse(record["ownership_adopted"])
        self.assertFalse(record["deletion_authorized"])
        self.assertFalse(record["physical_completion_attested"])
        self.assertFalse(record["contents_hashed"])
        self.assertEqual(record["total_file_bytes"], len(before))
        self.assertEqual(record["root_identity"]["device"], self.directory.stat().st_dev)
        self.assertEqual(record["root_identity"]["inode"], self.directory.stat().st_ino)
        with self.assertRaises(FileExistsError):
            self.register()
        self.assertEqual(json.loads(self.output.read_text()), record)
        self.assertEqual(sorted(p.name for p in self.registry.iterdir()), [self.output.name])

    def test_failed_write_or_publish_cleans_temporary_without_replacing(self):
        with mock.patch("storage_legacy_inventory.os.write", side_effect=OSError("write failed")):
            with self.assertRaises(OSError):
                self.register()
        self.assertEqual(list(self.registry.iterdir()), [])
        self.assertTrue(self.binary.exists())
        with mock.patch("storage_legacy_inventory._rename_noreplace", side_effect=OSError("rename failed")):
            with self.assertRaises(OSError):
                self.register()
        self.assertEqual(list(self.registry.iterdir()), [])
        self.output.write_text("existing")
        with self.assertRaises(FileExistsError):
            self.register()
        self.assertEqual(self.output.read_text(), "existing")
        self.assertEqual(sorted(p.name for p in self.registry.iterdir()), [self.output.name])

    def test_fsync_failure_preserves_complete_publication_if_already_linked(self):
        calls = 0
        original = os.fsync

        def fail_publication_sync(fd):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("publication directory sync failed")
            return original(fd)

        with mock.patch("storage_legacy_inventory.os.fsync", side_effect=fail_publication_sync):
            with self.assertRaisesRegex(StorageError, "publication may exist; durability is unknown"):
                self.register()
        self.assertEqual(sorted(p.name for p in self.registry.iterdir()), [self.output.name])
        self.assertFalse(json.loads(self.output.read_text())["deletion_authorized"])
        self.assertTrue(self.binary.exists())

    def test_explicit_binary_directories_supported_but_runtime_rejected(self):
        for name in ("binaries", "binaries-build-421"):
            candidate = self.inputs / name
            candidate.mkdir()
            (candidate / "rz-rovent-native").write_bytes(b"\x7fELFexample")
            output = self.registry / (name + ".json")
            result = observe(self.inputs, candidate, output, max_bytes=1000)
            self.assertEqual(result["source"], str(candidate))
            self.assertFalse(result["deletion_authorized"])
        runtime = self.inputs / "runtime"
        runtime.mkdir()
        with self.assertRaises(StorageError):
            observe(self.inputs, runtime, self.registry / "runtime.json", max_bytes=1000)

    def test_maintenance_entry_registers_and_inspects_without_authority(self):
        args = ["storage_maintenance.py", "register-legacy-observation",
                "--inputs-root", str(self.inputs), "--directory", str(self.directory),
                "--output", str(self.output), "--max-bytes", "1000"]
        with mock.patch.object(sys, "argv", args), redirect_stdout(io.StringIO()):
            self.assertEqual(maintenance_main(), 0)
        summary = inspect_observation(self.output)
        self.assertFalse(summary["deletion_authorized"])
        self.assertFalse(summary["current_source_revalidated"])
        self.assertEqual(summary["root_identity"]["inode"], self.directory.stat().st_ino)
        args = ["storage_maintenance.py", "inspect-legacy-observation", "--observation", str(self.output)]
        captured = io.StringIO()
        with mock.patch.object(sys, "argv", args), redirect_stdout(captured):
            self.assertEqual(maintenance_main(), 0)
        self.assertFalse(json.loads(captured.getvalue())["deletion_authorized"])

    def test_refuses_secret_like_entry_and_links(self):
        secret = self.directory / ".env.secret"
        secret.touch()
        with self.assertRaises(StorageError):
            self.register()
        self.assertFalse(self.output.exists())
        secret.unlink()
        link = self.directory / "linked"
        link.symlink_to(self.binary)
        with self.assertRaises(StorageError):
            self.register()
        self.assertFalse(self.output.exists())
        link.unlink()
        os.link(self.binary, self.directory / "hardlink")
        with self.assertRaises(StorageError):
            self.register()
        self.assertFalse(self.output.exists())

    def test_bounds_and_scope_reject_without_publishing(self):
        with self.assertRaises(StorageError):
            self.register(max_bytes=1)
        self.assertFalse(self.output.exists())
        with self.assertRaises(StorageError):
            observe(self.inputs, self.directory, self.directory / "output.json", max_bytes=1000)
        self.assertFalse((self.directory / "output.json").exists())
        with self.assertRaises(StorageError):
            observe(self.inputs, self.inputs, self.output, max_bytes=1000)
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
