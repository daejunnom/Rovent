"""CPU-only checks for the V3 snapshot boundary, not engine/GPU evidence."""
from __future__ import annotations

import builtins
import hashlib
import importlib.util
import os
from pathlib import Path
import stat
import sys
import tempfile
import time
import unittest
from unittest import mock


SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
import rovent_v3_input_snapshot as snapshot


class SnapshotImportTests(unittest.TestCase):
    def test_module_load_does_not_import_the_research_runner(self):
        original_import = builtins.__import__

        def import_without_research(name, *args, **kwargs):
            if name == "rovent_before_native_owner":
                raise AssertionError("research runner is not a snapshot dependency")
            return original_import(name, *args, **kwargs)

        spec = importlib.util.spec_from_file_location(
            "independent_v3_snapshot_fixture", SCRIPTS / "rovent_v3_input_snapshot.py"
        )
        module = importlib.util.module_from_spec(spec)
        with mock.patch("builtins.__import__", side_effect=import_without_research):
            spec.loader.exec_module(module)
        self.assertTrue(callable(module.copy_pin))

    def test_duplicate_json_keys_and_nonfinite_values_are_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "duplicate JSON key"):
            snapshot.unique_json(b'{"a":1,"a":2}')
        with self.assertRaises(ValueError):
            snapshot.unique_json(b'{"a":NaN}')
        self.assertEqual(snapshot.unique_json(b'{"a":1}'), {"a": 1})


@unittest.skipUnless(os.name == "posix" and hasattr(os, "O_NOFOLLOW"), "POSIX no-follow required")
class SnapshotOwnershipTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.output = self.root / "output"
        self.source.mkdir()
        self.output.mkdir()
        self.source_fd = snapshot.open_directory(self.source)
        self.addCleanup(os.close, self.source_fd)

    def test_copy_uses_actual_bytes_and_never_overwrites(self):
        data = os.urandom(127)
        (self.source / "asset").write_bytes(data)
        sha = hashlib.sha256(data).hexdigest()
        item = {"path": "asset", "bytes": len(data), "sha256": sha}
        progress = [0, len(data)]
        result = snapshot.copy_pin(self.source_fd, self.output, item, time.monotonic() + 5, progress)
        self.assertEqual(result, {"path": "asset", "bytes": len(data), "sha256": sha})
        self.assertEqual((self.output / "asset").read_bytes(), data)
        self.assertEqual(progress[0], len(data))
        with self.assertRaises(FileExistsError):
            snapshot.copy_pin(self.source_fd, self.output, item, time.monotonic() + 5, [0, len(data)])
        self.assertEqual((self.output / "asset").read_bytes(), data)

    def test_wrong_size_hash_and_expired_deadline_are_distinct_failures(self):
        data = os.urandom(31)
        sha = hashlib.sha256(data).hexdigest()
        for name in ("size", "hash", "expired"):
            (self.source / name).write_bytes(data)
        with self.assertRaisesRegex(RuntimeError, "source artifact size differs"):
            snapshot.copy_pin(self.source_fd, self.output,
                              {"path": "size", "bytes": len(data) + 1, "sha256": sha},
                              time.monotonic() + 5, [0, len(data) + 1])
        different_sha = hashlib.sha256(data + b"different").hexdigest()
        with self.assertRaisesRegex(RuntimeError, "artifact bytes or source identity changed"):
            snapshot.copy_pin(self.source_fd, self.output,
                              {"path": "hash", "bytes": len(data), "sha256": different_sha},
                              time.monotonic() + 5, [0, len(data)])
        with self.assertRaisesRegex(RuntimeError, "deadline expired"):
            snapshot.copy_pin(self.source_fd, self.output,
                              {"path": "expired", "bytes": len(data), "sha256": sha},
                              time.monotonic() - 1, [0, len(data)])

    def test_source_and_ancestor_symlinks_are_not_followed(self):
        data = os.urandom(16)
        (self.source / "actual").write_bytes(data)
        (self.source / "link").symlink_to("actual")
        item = {"path": "link", "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        with self.assertRaises(OSError):
            snapshot.copy_pin(self.source_fd, self.output, item, time.monotonic() + 5, [0, len(data)])
        (self.root / "source-link").symlink_to(self.source, target_is_directory=True)
        with self.assertRaises(OSError):
            snapshot.open_directory(self.root / "source-link")

    def test_changed_source_identity_is_not_accepted_even_if_first_read_matches(self):
        data = os.urandom(64)
        path = self.source / "changing"
        path.write_bytes(data)
        item = {"path": path.name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        original_read = os.read
        changed = False

        def change_after_first_read(fd, size):
            nonlocal changed
            chunk = original_read(fd, size)
            if chunk and not changed:
                changed = True
                # Keep the already-read bytes unchanged, but make the source
                # descriptor's size observably different on every filesystem.
                # Same-size writes may have identical timestamp resolution.
                path.write_bytes(b"")
            return chunk

        with mock.patch.object(snapshot.os, "read", side_effect=change_after_first_read):
            with self.assertRaisesRegex(RuntimeError, "artifact bytes or source identity changed"):
                snapshot.copy_pin(self.source_fd, self.output, item,
                                  time.monotonic() + 5, [0, len(data)])
        self.assertTrue(changed)

    def test_output_mode_change_requires_same_lease_owner_inode(self):
        parent_fd = snapshot.open_directory(self.root)
        self.addCleanup(os.close, parent_fd)
        info = self.output.stat()
        lease = mock.Mock(record={"identity": [info.st_dev, info.st_ino]}, _created_directory=None)
        os.chmod(self.output, 0o755)
        snapshot.protect_output(lease, self.output, parent_fd)
        self.assertEqual(stat.S_IMODE(self.output.stat().st_mode), 0o700)
        os.chmod(self.output, 0o755)
        lease.record = {"identity": [info.st_dev, info.st_ino + 1]}
        with self.assertRaisesRegex(RuntimeError, "identity changed"):
            snapshot.protect_output(lease, self.output, parent_fd)
        self.assertEqual(stat.S_IMODE(self.output.stat().st_mode), 0o755)


if __name__ == "__main__":
    unittest.main()
