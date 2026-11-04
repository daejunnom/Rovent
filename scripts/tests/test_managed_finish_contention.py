"""Bounded original-owner finalization under output catalog contention."""
import errno
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from managed_storage import CatalogBusy, ManagedBuild, StorageError, read_record, write_record
from storage_guard import CatalogGuard
import managed_outputs


class FinishContentionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="rz-finish-contention-")
        self.addCleanup(temporary.cleanup)
        base = Path(temporary.name)
        self.source = base / "source"
        self.source.mkdir()
        self.root = base / "build"
        self.catalog = base / "output-catalog"
        self.catalog.mkdir()
        self.manager = ManagedBuild(self.root, self.source).acquire()
        self.addCleanup(self.manager.guard.release)
        self.patched = patch.object(managed_outputs, "canonical_catalog", return_value=self.catalog)
        self.patched.start()
        self.addCleanup(self.patched.stop)

    def test_transient_busy_retries_and_closes_original_owner(self):
        busy = CatalogBusy("held")
        with patch.object(managed_outputs, "finalize_supervised_outputs", side_effect=[busy, None]) as finish, \
             patch("managed_storage.time.sleep") as sleep:
            result = self.manager.finish(exit_code=0, tree_gone=True)
        self.assertEqual(finish.call_count, 2)
        sleep.assert_called_once()
        self.assertFalse(self.manager.lock.exists())
        self.assertTrue(result["tree_cleanup_verified"])
        self.assertTrue(read_record(self.root / "closure.json")["tree_gone"])

    def test_real_catalog_lock_released_by_other_owner(self):
        held = CatalogGuard(self.catalog / "catalog.lock").acquire()
        self.addCleanup(held.release)
        released = threading.Event()
        def release_after_delay():
            time.sleep(0.15)
            held.release()
            released.set()
        thread = threading.Thread(target=release_after_delay)
        thread.start()
        try:
            result = self.manager.finish(exit_code=0, tree_gone=True)
        finally:
            thread.join(timeout=2)
        self.assertTrue(released.is_set())
        self.assertTrue(result["tree_cleanup_verified"])
        self.assertFalse(self.manager.lock.exists())

    def test_persistent_busy_preserves_exact_owner_and_closure(self):
        owner_before = read_record(self.manager.lock)
        with patch.object(managed_outputs, "finalize_supervised_outputs", side_effect=CatalogBusy("held")) as finish, \
             patch("managed_storage.time.monotonic", side_effect=[0.0, 0.5, 2.0]), \
             patch("managed_storage.time.sleep"):
            with self.assertRaises(CatalogBusy):
                self.manager.finish(exit_code=125, tree_gone=True)
        self.assertEqual(finish.call_count, 2)
        self.assertEqual(read_record(self.manager.lock), owner_before)
        self.assertEqual(read_record(self.root / "closure.json"),
                         {"schema": owner_before["schema"], "owner": owner_before, "tree_gone": True})
        self.assertTrue(self.manager.scratch.exists())

    def test_non_contention_error_is_not_retried(self):
        original = StorageError("output metadata changed")
        with patch.object(managed_outputs, "finalize_supervised_outputs", side_effect=original) as finish:
            with self.assertRaises(StorageError) as caught:
                self.manager.finish(exit_code=125, tree_gone=True)
        self.assertIs(caught.exception, original)
        finish.assert_called_once()
        self.assertTrue(self.manager.lock.exists())
        self.assertTrue((self.root / "closure.json").exists())

    def test_owner_change_during_wait_stops_retry(self):
        owner_before = read_record(self.manager.lock)
        changed = {**owner_before, "token": "other-owner"}
        def first(*args, **kwargs):
            write_record(self.manager.lock, changed)
            raise CatalogBusy("held")
        with patch.object(managed_outputs, "finalize_supervised_outputs", side_effect=first) as finish, \
             patch("managed_storage.time.sleep"):
            with self.assertRaisesRegex(StorageError, "owner changed"):
                self.manager.finish(exit_code=125, tree_gone=True)
        finish.assert_called_once()
        self.assertEqual(read_record(self.manager.lock), changed)
        self.assertEqual(read_record(self.root / "closure.json")["owner"], owner_before)

    def test_closure_change_during_wait_stops_retry(self):
        def first(*args, **kwargs):
            closure = self.root / "closure.json"
            write_record(closure, read_record(closure))
            raise CatalogBusy("held")
        with patch.object(managed_outputs, "finalize_supervised_outputs", side_effect=first) as finish, \
             patch("managed_storage.time.sleep"):
            with self.assertRaisesRegex(StorageError, "closure changed"):
                self.manager.finish(exit_code=125, tree_gone=True)
        finish.assert_called_once()
        self.assertTrue(self.manager.lock.exists())

    def test_unknown_tree_is_never_terminal(self):
        with self.assertRaisesRegex(StorageError, "cleanup unverified"):
            self.manager.finish(exit_code=125, tree_gone="unknown")
        self.assertTrue(self.manager.lock.exists())
        self.assertFalse((self.root / "closure.json").exists())

    @unittest.skipUnless(os.name == "posix", "POSIX flock error classification")
    def test_only_actual_lock_contention_has_retryable_type(self):
        lock = self.catalog / "catalog.lock"
        with patch("fcntl.flock", side_effect=OSError(errno.EIO, "I/O error")):
            with self.assertRaises(StorageError) as caught:
                CatalogGuard(lock).acquire()
        self.assertNotIsInstance(caught.exception, CatalogBusy)
        with patch("fcntl.flock", side_effect=BlockingIOError(errno.EAGAIN, "held")):
            with self.assertRaises(CatalogBusy):
                CatalogGuard(lock).acquire()


if __name__ == "__main__":
    unittest.main()
