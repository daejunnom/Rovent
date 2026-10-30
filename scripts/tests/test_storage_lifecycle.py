"""Real temporary-file/owner boundaries; no NN, GPU, Cargo or real data cleanup."""
import hashlib
import contextlib
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import stat
from types import SimpleNamespace
import tempfile
import time
import unittest
from unittest.mock import MagicMock, Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from managed_storage import CatalogBusy, ManagedBuild, StorageError, recover_build, tree_size, write_record
from managed_outputs import (OutputLease, OutputBudgetError, publish_runtime,
                             recover_output, sample_output_budget)
from storage_guard import CatalogGuard, lifetime
from storage_maintenance import inventory


def pin(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="rovezero-lifecycle-")
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name).resolve()
        self.source = self.base / "source"
        self.source.mkdir()
        self.catalog = self.base / "outputs"
        self.build = self.base / "build"
        # Exercise the real default finalizer in this private fixture. Tests
        # with explicit catalog overrides and outer-supervisor ownership still
        # use their existing real code paths, never the user's live catalog.
        catalog_patch = patch("managed_outputs.canonical_catalog", return_value=self.catalog)
        catalog_patch.start()
        self.addCleanup(catalog_patch.stop)

    def lease(self, name="run", **kwargs):
        owner = OutputLease(self.base / name, self.source, catalog=self.catalog, **kwargs).acquire()
        self.addCleanup(owner.guard.release)
        return owner

    def test_retained_evidence_counts_after_expiry_no_automatic_deletion(self):
        first = self.lease(reserve=12000, total_limit=18000)
        evidence = first.output / "unique.pgn"
        evidence.write_bytes(b"x" * 7000)
        first.finish(status="failed_evidence_preserved", tree_gone=True)
        with patch("managed_outputs.time.time_ns", return_value=10**30):
            next_owner = self.lease("next", reserve=12000, total_limit=18000)
            next_owner.finish(status="completed", tree_gone=True)
        self.assertEqual(evidence.stat().st_size, 7000)
        self.assertTrue((self.base / "next").exists())
        self.assertGreater(sample_output_budget(self.catalog), 7000)

    def test_total_warning_does_not_erase_evidence_or_replace_per_run_limit(self):
        owner = self.lease(reserve=4096, total_limit=1024)
        payload = owner.output / "observed.bin"
        payload.write_bytes(b"x" * 1600)
        with patch("managed_outputs.warn_output_threshold") as warning:
            owner.check_budget()
            self.assertTrue(warning.called)
        owner.finish(status="completed", tree_gone=True)
        self.assertEqual(payload.stat().st_size, 1600)

        other = self.lease("larger-reservation", reserve=20 * 1024 ** 3)
        other.finish(status="completed", tree_gone=True)
        with patch("managed_outputs.tree_size", side_effect=lambda path, **_: (
                 19 * 1024 ** 3 if path == other.output else 1600)), \
             patch("managed_outputs.warn_output_threshold") as warning:
            self.assertEqual(sample_output_budget(self.catalog), 19 * 1024 ** 3 + 1600)
            self.assertTrue(warning.called)

    def test_repeated_budget_check_uses_guarded_foreign_reservations(self):
        old = self.lease("foreign", reserve=8192)
        old_file = old.output / "unique-result.pgn"
        old_file.write_bytes(b"keep")
        old.finish(status="closed", tree_gone=True)
        owner = self.lease("own", reserve=4096)
        data = owner.output / "data"
        data.write_bytes(b"x" * 10)
        with patch("managed_outputs._catalog_rows", side_effect=AssertionError("repeat foreign census")):
            with patch("managed_outputs.warn_output_threshold") as warn:
                owner.check_budget()
                owner.check_budget()
                self.assertEqual(warn.call_args.args[0], owner._last_own_usage + 8192)
            data.write_bytes(b"x" * 5000)
            with self.assertRaises(OutputBudgetError):
                owner.check_budget()
            data.unlink()
            owner.finish(status="closed", tree_gone=True)
        self.assertEqual(old_file.read_bytes(), b"keep")

    def test_repeated_budget_check_still_rejects_changed_own_marker(self):
        owner = self.lease(reserve=4096)
        marker = owner.output / ".rovezero-output.json"
        original = marker.read_bytes()
        marker.write_text("{}", encoding="utf-8")
        with self.assertRaises(StorageError):
            owner.check_budget()
        marker.write_bytes(original)
        owner.finish(status="closed", tree_gone=True)

    def test_live_and_interrupted_output_owner_never_stolen(self):
        owner = self.lease()
        with self.assertRaises(StorageError):
            self.lease("second")
        owner.guard.release()  # Simulate lost gate; not a child closure proof.
        independent = self.lease("second")
        independent.finish(status="independent-complete", tree_gone=True)
        with self.assertRaises(StorageError):
            recover_output(self.catalog, owner.key, pin(self.catalog / (owner.key + ".json")))
        self.assertTrue(owner.output.exists())
        self.assertEqual(json.loads((self.catalog / (owner.key + ".json")).read_text())["state"], "active")

    def test_failed_post_publication_admission_retains_original_without_owner_leak(self):
        for deferred in (False, True):
            with self.subTest(deferred=deferred):
                owner = OutputLease(self.base / ("failed-" + str(deferred)), self.source,
                                    catalog=self.catalog, deferred=deferred)
                with patch.object(owner, "_seal_preservation_session", side_effect=OSError("seal failed")):
                    with self.assertRaisesRegex(OSError, "seal failed"):
                        owner.acquire()
                row = json.loads((self.catalog / (owner.key + ".json")).read_text())
                self.assertEqual(row["state"], "retained")
                self.assertEqual(row["status"], "acquire_failed_before_dispatch")
                self.assertEqual(row["tree_gone"], "not_started_by_lease")
                self.assertEqual(row["physical_completion"], "not_inferred_from_output_lifecycle")
                if not deferred:
                    marker = json.loads((owner.output / ".rovezero-output.json").read_text())
                    self.assertEqual(marker, row)
                else:
                    self.assertFalse(owner.output.exists())
                independent = self.lease("next-" + str(deferred))
                independent.finish(status="independent", tree_gone=True)

    def test_failed_partial_publication_or_changed_identity_stays_unrecovered(self):
        partial = OutputLease(self.base / "partial", self.source, catalog=self.catalog)
        original_write = partial._write
        def failed_publication():
            original_write()
            raise OSError("publication outcome uncertain")
        with patch.object(partial, "_write", side_effect=failed_publication):
            with self.assertRaisesRegex(OSError, "publication outcome uncertain"):
                partial.acquire()
        partial_row = json.loads((self.catalog / (partial.key + ".json")).read_text())
        self.assertEqual(partial_row["state"], "active")

        changed = OutputLease(self.base / "changed", self.source, catalog=self.catalog)
        def change_marker_then_fail():
            marker = changed.output / ".rovezero-output.json"
            marker.write_text("{}")
            raise OSError("seal failed after identity change")
        with patch.object(changed, "_seal_preservation_session", side_effect=change_marker_then_fail):
            with self.assertRaisesRegex(OSError, "seal failed after identity change"):
                changed.acquire()
        changed_row = json.loads((self.catalog / (changed.key + ".json")).read_text())
        self.assertEqual(changed_row["state"], "active")
        self.assertEqual((changed.output / ".rovezero-output.json").read_text(), "{}")

    def test_brief_catalog_lock_contention_is_retried_without_stealing_owner(self):
        candidate = OutputLease(self.base / "retry", self.source, catalog=self.catalog)
        original_acquire = candidate.guard.acquire
        first = True
        def acquire_after_contention():
            nonlocal first
            if first:
                first = False
                raise CatalogBusy("busy")
            return original_acquire()
        with patch.object(candidate.guard, "acquire", side_effect=acquire_after_contention) as attempts, \
             patch("managed_outputs.time.sleep") as sleeper:
            candidate.acquire()
        self.addCleanup(candidate.guard.release)
        self.assertEqual(attempts.call_count, 2)
        sleeper.assert_called_once()
        candidate.finish(status="complete", tree_gone=True)

    def test_persistent_catalog_lock_contention_exits_without_creating_output(self):
        candidate = OutputLease(self.base / "busy", self.source, catalog=self.catalog)
        with patch.object(candidate.guard, "acquire", side_effect=CatalogBusy("busy")) as attempts, \
             patch("managed_outputs.time.monotonic", side_effect=[0.0, 0.1, 2.1]), \
             patch("managed_outputs.time.sleep") as sleeper:
            with self.assertRaises(CatalogBusy):
                candidate.acquire()
        self.assertEqual(attempts.call_count, 2)
        sleeper.assert_called_once()
        self.assertFalse(candidate.output.exists())

    def test_unknown_tree_is_preserved_while_independent_root_proceeds(self):
        build = ManagedBuild(self.build, self.source).acquire()
        self.addCleanup(build.guard.release)
        with patch.dict(os.environ, {"RZ_MANAGED_BUILD_ROOT": str(build.root),
                                     "RZ_MANAGED_BUILD_TOKEN": build.token}):
            owner = self.lease(outer_supervised=True)
        owner.finish(status="child_returned_outer_closure_pending", tree_gone="unknown")
        self.assertEqual(owner.record['state'], 'active')
        independent = self.lease('independent')
        independent.finish(status='independent-complete', tree_gone=True)
        self.assertEqual(json.loads((self.catalog / (owner.key + '.json')).read_text())['state'], 'active')
        with patch('managed_outputs.canonical_catalog', return_value=self.catalog):
            build.finish(exit_code=0, tree_gone=True)
        row = json.loads((self.catalog / (owner.key + '.json')).read_text())
        self.assertEqual(row['state'], 'retained')
        self.assertEqual(row['closure'], 'original_outer_supervisor')
        self.assertEqual(row['physical_completion'], 'not_inferred_from_output_lifecycle')
        later = self.lease('admitted')
        later.finish(status='completed', tree_gone=True)

    def test_last_receipt_growth_is_checked_again_by_outer_supervisor(self):
        build = ManagedBuild(self.build, self.source).acquire()
        self.addCleanup(build.guard.release)
        with patch.dict(os.environ, {"RZ_MANAGED_BUILD_ROOT": str(build.root),
                                     "RZ_MANAGED_BUILD_TOKEN": build.token}):
            owner = self.lease(reserve=4096, outer_supervised=True)
        owner.finish(status='checks_completed', tree_gone='unknown')
        (owner.output / 'last-receipt.json').write_bytes(b'x' * 5000)
        with patch('managed_outputs.canonical_catalog', return_value=self.catalog):
            with self.assertRaisesRegex(StorageError, 'exceeds reservation'):
                build.finish(exit_code=0, tree_gone=True)
        self.assertTrue(build.lock.exists())
        self.assertEqual(json.loads((self.catalog / (owner.key + '.json')).read_text())['state'], 'active')

    def test_output_overflow_preserves_result_and_records_failure(self):
        owner = self.lease(reserve=4096, total_limit=8192)
        evidence = owner.output / "raw.log"
        evidence.write_bytes(b"x" * 5000)
        with self.assertRaises(OutputBudgetError):
            owner.finish(status="failed", tree_gone=True)
        record = json.loads((self.catalog / (owner.key + ".json")).read_text())
        self.assertEqual(record["state"], "retained")
        self.assertIn("budget", record["budget_error"])
        self.assertEqual(evidence.stat().st_size, 5000)

    def test_terminal_metadata_growth_is_not_a_successful_finish(self):
        owner = self.lease(reserve=4096, total_limit=8192)
        with self.assertRaises(OutputBudgetError):
            owner.finish(status='diagnostic:' + 'x' * 2000, tree_gone=True)
        row = json.loads((self.catalog / (owner.key + '.json')).read_text())
        self.assertEqual(row['status'], 'failed_output_budget')
        self.assertIn('budget', row['budget_error'])

    def test_changed_owner_is_never_overwritten_by_finish(self):
        owner = self.lease()
        changed = {**owner.record, "token": "a" * 32}
        for path in (owner.output / ".rovezero-output.json", self.catalog / (owner.key + ".json")):
            write_record(path, changed)
        with self.assertRaisesRegex(StorageError, "owner changed"):
            owner.finish(status="completed", tree_gone=True)
        self.assertEqual(json.loads((owner.output / ".rovezero-output.json").read_text()), changed)

    def test_deferred_file_preserves_consumer_fresh_path_and_future_identity(self):
        owner = self.lease("report.json", deferred=True, reserve=4096)
        self.assertFalse(owner.output.exists())
        owner.output.write_bytes(b"actual-result")
        owner.check_budget()
        owner.finish(status="completed", tree_gone=True)
        self.assertEqual(sample_output_budget(self.catalog), len(b"actual-result"))
        other = self.lease("second")
        other.finish(status="completed", tree_gone=True)
        self.assertEqual(owner.output.read_bytes(), b"actual-result")

    def test_deferred_no_child_output_is_retained_as_absent_not_adopted_later(self):
        owner = self.lease("not-created.json", deferred=True)
        owner.finish(status="not_started", tree_gone=True)
        self.assertFalse(owner.output.exists())
        later = self.lease("later")
        later.finish(status="completed", tree_gone=True)
        original_row = (self.catalog / (owner.key + ".json")).read_bytes()
        owner.output.write_bytes(b"unrelated-later-file")
        independent = self.lease("next")
        independent.finish(status="independent-complete", tree_gone=True)
        self.assertEqual((self.catalog / (owner.key + ".json")).read_bytes(), original_row)
        self.assertEqual(owner.output.read_bytes(), b"unrelated-later-file")
        with self.assertRaisesRegex(StorageError, "custody changed"):
            self.lease("not-created.json")

    def test_historical_and_overlapping_outputs_are_rejected(self):
        historical = self.base / "historical"
        historical.mkdir()
        (historical / "only-result").write_bytes(b"keep")
        with self.assertRaisesRegex(StorageError, "fresh"):
            self.lease("historical")
        owner = self.lease()
        owner.finish(status="completed", tree_gone=True)
        with self.assertRaisesRegex(StorageError, "overlapping"):
            self.lease("run/nested")
        self.assertEqual((historical / "only-result").read_bytes(), b"keep")

    def test_replaced_catalog_gate_is_rejected(self):
        owner = self.lease()
        original = self.catalog / "catalog.lock"
        try:
            original.rename(self.catalog / "retired-gate")
        except PermissionError:
            self.skipTest("Windows denies replacement of the held file")
        original.write_bytes(b"0")
        with self.assertRaisesRegex(StorageError, "gate replaced"):
            owner.finish(status="completed", tree_gone=True)

    @unittest.skipUnless(sys.platform == 'linux', 'directory replacement while gate inode remains open')
    def test_catalog_replacement_with_same_lock_inode_is_rejected(self):
        owner = self.lease()
        saved = self.base / 'original-catalog'
        self.catalog.rename(saved)
        self.catalog.mkdir()
        (saved / 'catalog.lock').rename(self.catalog / 'catalog.lock')
        with self.assertRaisesRegex(StorageError, 'gate replaced'):
            owner.finish(status='completed', tree_gone=True)
        self.assertTrue((saved / (owner.key + '.json')).exists())

    def test_purpose_slots_count_together_and_do_not_adopt_manual_sibling(self):
        sibling = self.base / "startup-caller-diagnostic"
        sibling.mkdir()
        (sibling / "unknown").write_bytes(b"keep")
        slots = []
        for purpose in (None, "startup-caller", "another-purpose"):
            owner = ManagedBuild(self.build, self.source, purpose=purpose, slots_limit=2).acquire()
            self.addCleanup(owner.guard.release)
            slots.append(owner.slot)
            owner.finish(exit_code=0, tree_gone=True)
        self.assertFalse(slots[0].exists())
        self.assertEqual((sibling / "unknown").read_bytes(), b"keep")
        report = inventory([self.base])
        self.assertFalse(report["deletion_performed"])
        self.assertEqual(report["whole_C_drive_or_VHD_physical_usage"], "unknown")

    def test_unknown_v1_and_same_boot_lease_preserved(self):
        owner = ManagedBuild(self.build, self.source).acquire()
        self.addCleanup(owner.guard.release)
        (owner.scratch / "in-use").write_bytes(b"keep")
        owner.guard.release()
        for legacy in (False, True):
            if legacy:
                for path in (owner.lock, owner.slot / "record.json"):
                    record = json.loads(path.read_text())
                    record.pop("lifetime")
                    record.pop("root_identity")
                    write_record(path, record)
            original = owner.lock.read_bytes()
            with self.assertRaises(StorageError):
                recover_build(self.build, pin(owner.lock))
            self.assertEqual(owner.lock.read_bytes(), original)
            self.assertEqual((owner.scratch / "in-use").read_bytes(), b"keep")

    def test_mocked_reboot_recovery_exact_owner_and_root_only(self):
        owner = ManagedBuild(self.build, self.source).acquire()
        self.addCleanup(owner.guard.release)
        (owner.scratch / "input").write_bytes(b"regenerable")
        (owner.target / "cache").write_bytes(b"regenerable")
        owner.guard.release()
        original = pin(owner.lock)
        # A format change is not a reboot.  Inject a later counter in the same
        # Windows domain; Linux retains its distinct per-boot UUID contract.
        if owner.lifetime["platform"] == "win32":
            from storage_guard import WINDOWS_BOOT_COUNTER_PREFIX, _windows_boot_count
            count = _windows_boot_count(owner.lifetime["boot"])
            self.assertIsNotNone(count)
            self.assertLess(count, 0xffffffff)
            fixture_boot = WINDOWS_BOOT_COUNTER_PREFIX + str(count + 1)
        else:
            fixture_boot = "different-test-boot"
        current = {**owner.lifetime, "boot": fixture_boot}
        with patch("managed_storage.lifetime", return_value=current):
            with self.assertRaisesRegex(StorageError, "digest changed"):
                recover_build(self.build, "0" * 64)
            result = recover_build(self.build, original)
        self.assertTrue(result["recovered"])
        self.assertFalse(owner.lock.exists())
        self.assertFalse(owner.scratch.exists())
        self.assertFalse(owner.target.exists())
        reuse = ManagedBuild(self.build, self.source).acquire()
        reuse.finish(exit_code=0, tree_gone=True)

    def test_real_owner_crash_after_durable_tree_closure_is_recoverable(self):
        scripts = str(Path(__file__).resolve().parents[1])
        program = f"""import sys,os
sys.path.insert(0, {scripts!r})
from pathlib import Path
from unittest.mock import patch
from managed_storage import ManagedBuild, StorageError
owner=ManagedBuild(Path({str(self.build)!r}),Path({str(self.source)!r})).acquire()
(owner.scratch/'input').write_bytes(b'owned')
try:
    # A new Python process cannot inherit the parent's unittest.mock patch.
    with patch('managed_outputs.canonical_catalog',return_value=Path({str(self.catalog)!r})):
        with patch('managed_storage.remove_tree',side_effect=StorageError('synthetic crash after closure')):
            owner.finish(exit_code=0,tree_gone=True)
except StorageError:
    os._exit(17)
"""
        child = subprocess.run([sys.executable, "-B", "-c", program], timeout=10, capture_output=True)
        self.assertEqual(child.returncode, 17, child.stderr)
        digest = pin(self.build / "owner.json")
        with patch.object(Path, 'read_bytes', side_effect=AssertionError('second pathname byte read forbidden')):
            result = recover_build(self.build, digest)
        self.assertEqual(result["reason"], "original_terminal_closure_and_owner_dead")
        self.assertFalse((self.build / "owner.json").exists())

    def test_shared_runtime_reuse_corruption_and_quota_keep_originals(self):
        source = self.base / "source-runtime.so"
        source.write_bytes(b"synthetic runtime bytes")
        store = self.base / "shared-runtime"
        first, digest = publish_runtime(source, store, limit=10000)
        original_identity = (first.stat().st_dev, first.stat().st_ino)
        second, same_digest = publish_runtime(source, store, limit=10000)
        self.assertEqual((digest, first), (same_digest, second))
        self.assertEqual(original_identity, (second.stat().st_dev, second.stat().st_ino))
        corrupt = b"c" * source.stat().st_size
        first.write_bytes(corrupt)
        with self.assertRaisesRegex(StorageError, "digest mismatch"):
            publish_runtime(source, store, limit=10000)
        self.assertEqual(first.read_bytes(), corrupt)
        first.write_bytes(corrupt + b'oversized')
        real_open = Path.open
        def forbid_target_read(path, *args, **kwargs):
            if path == first:
                raise AssertionError('oversized target must not be opened')
            return real_open(path, *args, **kwargs)
        with patch.object(Path, 'open', forbid_target_read):
            with self.assertRaisesRegex(StorageError, 'size mismatch'):
                publish_runtime(source, store, limit=10000)
        source.write_bytes(b"new runtime")
        with self.assertRaisesRegex(StorageError, "budget exhausted"):
            publish_runtime(source, store, limit=1)
        self.assertEqual(source.read_bytes(), b"new runtime")


class CreatedDirectoryBindingTests(unittest.TestCase):
    """Windows-safe in-memory OS/catalog faults; never touch a real output tree."""
    def setUp(self):
        self.mocks = contextlib.ExitStack()
        self.addCleanup(self.mocks.close)
        self.base = Path(os.path.abspath("mock-created-output"))
        self.parent = self.base / "private"
        self.target = self.parent / "attempt"
        self.catalog = self.base / "catalog"
        self.life = {"host": "same-host", "boot": "original-boot",
                     "platform": "linux", "process_birth": "original-birth"}
        self.parent_info = self.info(20)
        self.named_parent, self.named_child = self.parent_info, None
        self.fd_info, self.events, self.size = {}, [], 0
        self.version, self.writes, self.mkdir_calls = 0, 0, 0
        self.write_error = self.sync_error = self.readback_error = None
        self.gate = MagicMock(spec=CatalogGuard)
        self.gate.__enter__.return_value = self.gate
        self.gate.assert_held.side_effect = self.assert_held
        self.held = True
        self.mocks.enter_context(patch("managed_outputs.CatalogGuard", return_value=self.gate))
        self.mocks.enter_context(patch("managed_outputs.checked", side_effect=lambda path: Path(path)))
        self.mocks.enter_context(patch("managed_outputs.sys", SimpleNamespace(platform="linux", exc_info=sys.exc_info)))
        self.fake_os = SimpleNamespace(path=os.path, O_RDONLY=os.O_RDONLY,
            O_DIRECTORY=0x10000, O_NOFOLLOW=0x20000, getpid=lambda: 123, geteuid=lambda: 1000,
            open=Mock(side_effect=self.open_fd), fstat=lambda fd: self.fd_info[fd],
            stat=self.stat_name, mkdir=self.mkdir, fsync=self.fsync,
            close=Mock(side_effect=lambda fd: self.events.append(("close", fd))),
            unlink=Mock(side_effect=AssertionError("no deletion")))
        self.mocks.enter_context(patch("managed_outputs.os", self.fake_os))
        self.mocks.enter_context(patch("managed_outputs.lifetime", side_effect=lambda: dict(self.life)))
        self.mocks.enter_context(patch.object(Path, "lstat", lambda path: self.lstat(path)))
        self.mocks.enter_context(patch.object(Path, "exists", lambda path: path == self.catalog or (path == self.target and self.named_child is not None)))
        self.mocks.enter_context(patch("managed_outputs.read_record_snapshot", side_effect=self.snapshot))
        self.mocks.enter_context(patch("managed_outputs._read", side_effect=lambda path: copy.deepcopy(self.disk)))
        self.mocks.enter_context(patch("managed_outputs.write_record", side_effect=self.write))
        self.mocks.enter_context(patch("managed_outputs.tree_size", side_effect=lambda path, **kwargs: self.size if path == self.target else 0))
        self.owner = OutputLease(self.target, self.base / "source", catalog=self.catalog,
                                 deferred=True, reserve=4096, total_limit=8192)
        self.owner.record = {"schema": "rovezero.managed-output.v1", "key": self.owner.key,
            "output": str(self.target), "source": str(self.owner.source), "identity": None,
            "token": "original-token", "deferred": True, "outer_supervisor": None,
            "pid": 123, "lifetime": dict(self.life), "state": "active",
            "reserved_bytes": 4096, "catalog_limit_bytes": 8192}
        self.disk = copy.deepcopy(self.owner.record)
        self.owner._rows = Mock(side_effect=lambda: [(self.row_path, copy.deepcopy(self.disk), self.target)])
        self.addCleanup(self.owner._close_created_directory)

    @property
    def row_path(self):
        return self.catalog / (self.owner.key + ".json")

    @staticmethod
    def info(inode, mode=None):
        return SimpleNamespace(st_dev=7, st_ino=inode, st_mode=mode or (stat.S_IFDIR | 0o700),
                               st_uid=1000, st_gid=1000, st_nlink=2)

    def assert_held(self):
        if not self.held:
            raise StorageError("catalog gate not held")

    def open_fd(self, path, flags, *, dir_fd=None):
        self.assert_held()
        self.assertTrue(flags & self.fake_os.O_DIRECTORY)
        self.assertTrue(flags & self.fake_os.O_NOFOLLOW)
        if dir_fd is None:
            self.assertEqual(path, self.parent)
            self.fd_info[101] = self.parent_info
            return 101
        self.assertEqual((path, dir_fd), ("attempt", 101))
        self.fd_info[102] = self.named_child
        return 102

    def lstat(self, path):
        if path == self.catalog.with_name(self.catalog.name + ".interrupted-unbound-custody-v1"):
            raise FileNotFoundError(path)
        self.assertEqual(path, self.parent)
        return self.named_parent

    def stat_name(self, name, *, dir_fd, follow_symlinks):
        self.assertEqual((name, dir_fd, follow_symlinks), ("attempt", 101, False))
        if self.named_child is None:
            raise FileNotFoundError(name)
        return self.named_child

    def mkdir(self, name, mode, *, dir_fd):
        self.assertEqual((name, mode, dir_fd), ("attempt", 0o700, 101))
        if self.named_child is not None:
            raise FileExistsError(name)
        self.mkdir_calls += 1
        self.named_child = self.info(30)
        self.events.append(("mkdir", name))

    def fsync(self, fd):
        self.events.append(("fsync", fd))
        if self.sync_error is not None:
            raise self.sync_error

    def snapshot(self, path, *, schema):
        self.assertEqual((path, schema), (self.row_path, "rovezero.managed-output.v1"))
        if self.writes and self.readback_error is not None:
            raise self.readback_error
        raw = (json.dumps(self.disk, sort_keys=True) + "\n").encode()
        return copy.deepcopy(self.disk), hashlib.sha256(raw).hexdigest(), (self.version,)

    def write(self, path, value):
        self.assert_held()
        self.assertEqual(path, self.row_path)
        self.writes += 1
        self.disk, self.version = copy.deepcopy(value), self.version + 1
        self.events.append(("catalog_write", self.writes))
        if self.write_error is not None:
            raise self.write_error  # Simulate replace succeeding before directory fsync fails.

    def assert_sticky_failure(self, primary):
        self.assertIs(self.owner._directory_binding_error, primary)
        record, record_before = self.owner.record, copy.deepcopy(self.owner.record)
        disk, size, count = copy.deepcopy(self.disk), self.size, self.writes
        named, named_before = self.named_child, copy.deepcopy(vars(self.named_child)) if self.named_child is not None else None
        parent, parent_before = self.named_parent, copy.deepcopy(vars(self.named_parent))
        cap = self.owner._created_directory
        fds = (cap.fd, cap.parent_fd) if cap is not None else ()
        opened = [fd for fd in fds if fd is not None]
        closed, releases = self.fake_os.close.call_count, self.gate.release.call_count
        self.gate.assert_held()
        self.assertTrue(self.held)
        self.assertTrue(all(fd in self.fd_info for fd in opened))
        with self.assertRaises(StorageError) as before_finish:
            self.owner.check_budget()
        self.assertIs(before_finish.exception.__cause__, primary)
        self.assertEqual(self.gate.release.call_count, releases)
        self.assertEqual(self.fake_os.close.call_count, closed)
        if cap is not None:
            self.assertEqual((cap.fd, cap.parent_fd), fds)
        self.assertEqual((self.owner.record, self.disk, self.size, self.writes),
                         (record_before, disk, size, count))
        with self.assertRaises(StorageError) as blocked:
            self.owner.finish(status="completed", tree_gone=True)
        self.assertIs(blocked.exception.__cause__, primary)
        self.assertIs(self.owner._directory_binding_error, primary)
        self.assertIs(self.owner.record, record)
        self.assertEqual(self.owner.record, record_before)
        self.assertEqual(self.disk, disk)
        self.assertEqual(self.size, size)
        self.assertEqual(self.writes, count)
        self.assertEqual(self.disk["state"], "active")
        self.assertIs(self.named_child, named)
        if named is not None:
            self.assertEqual(vars(self.named_child), named_before)
        self.assertIs(self.named_parent, parent)
        self.assertEqual(vars(self.named_parent), parent_before)
        self.assertIs(self.owner._created_directory, cap)
        if cap is not None:
            self.assertIsNone(cap.fd)
            self.assertIsNone(cap.parent_fd)
        self.assertCountEqual([call.args[0] for call in self.fake_os.close.call_args_list[closed:]], opened)
        self.assertEqual(self.gate.release.call_count, releases + 1)
        self.fake_os.unlink.assert_not_called()

    def test_seal_precedes_payload_and_only_identity_changes(self):
        original = copy.deepcopy(self.disk)
        self.assertEqual(self.owner.create_deferred_directory(), self.target)
        self.assertEqual(self.owner._directory_binding_stage, "sealed")
        self.assertEqual(self.disk, dict(original, identity=[7, 30]))
        self.assertEqual(self.writes, 1)
        created = self.owner._created_directory
        owned_fds = (created.fd, created.parent_fd)
        creation = self.events.index(("mkdir", "attempt"))
        publication = next(index for index, event in enumerate(self.events) if event[0] == "catalog_write")
        durability = self.events[creation + 1:publication]
        self.assertIn(("fsync", created.fd), durability)
        self.assertIn(("fsync", created.parent_fd), durability)
        self.size = 100
        self.owner.check_budget()
        self.owner.finish(status="provider_unknown", tree_gone=False)
        self.assertEqual(self.disk["identity"], [7, 30])
        self.assertEqual(self.disk["state"], "active")
        self.assertEqual(self.disk["physical_completion"], "not_inferred_from_output_lifecycle")
        self.assertCountEqual([fd for action, fd in self.events if action == "close"], owned_fds)
        self.gate.release.assert_called_once()

    def test_existing_directory_regular_and_symlink_are_never_adopted(self):
        for mode in (stat.S_IFDIR | 0o700, stat.S_IFREG | 0o600, stat.S_IFLNK | 0o777):
            with self.subTest(mode=mode):
                self.owner._directory_binding_error = None
                self.owner._created_directory = None
                self.named_child = self.info(99, mode)
                with self.assertRaisesRegex(StorageError, "exclusive fresh") as failure:
                    self.owner.create_deferred_directory()
                self.assertEqual(self.named_child.st_ino, 99)
                self.assertEqual((self.mkdir_calls, self.writes), (0, 0))
                self.assert_sticky_failure(failure.exception)
                self.owner._close_created_directory()

    def test_changed_token_process_lifetime_or_gate_cannot_bind(self):
        for change in ("token", "lifetime", "gate"):
            with self.subTest(change=change):
                self.owner._directory_binding_error = None
                if change == "token":
                    self.disk["token"] = "replacement"
                elif change == "lifetime":
                    self.disk = copy.deepcopy(self.owner.record)
                    self.life["boot"] = "other-boot"
                else:
                    self.life["boot"] = "original-boot"
                    self.held = False
                with self.assertRaises(StorageError):
                    self.owner.create_deferred_directory()
                self.assertEqual((self.mkdir_calls, self.writes), (0, 0))

    def test_directory_and_parent_replacement_block_budget_and_finish(self):
        self.owner.create_deferred_directory()
        original = copy.deepcopy(self.disk)
        cap = self.owner._created_directory
        fds = (cap.fd, cap.parent_fd)
        self.named_child = self.info(99)
        replacement = self.named_child
        with self.assertRaisesRegex(StorageError, "directory changed"):
            self.owner.check_budget()
        self.assertEqual(self.disk, original)
        self.assertEqual((cap.fd, cap.parent_fd), fds)
        self.gate.release.assert_not_called()
        self.named_parent = self.info(98)
        replacement_parent = self.named_parent
        with self.assertRaisesRegex(StorageError, "parent changed"):
            self.owner.check_budget()
        self.assertEqual(self.disk, original)
        self.assertEqual((cap.fd, cap.parent_fd), fds)
        self.gate.assert_held()
        self.gate.release.assert_not_called()
        self.fake_os.close.assert_not_called()
        primary = []
        actual_check = self.owner.check_budget
        def check():
            try:
                actual_check()
            except StorageError as error:
                primary.append(error)
                raise
        with patch.object(self.owner, "check_budget", side_effect=check):
            with self.assertRaises(StorageError) as failure:
                self.owner.finish(status="completed", tree_gone=True)
        self.assertIs(failure.exception, primary[0])
        self.assertEqual(self.disk, original)
        self.assertIs(self.named_child, replacement)
        self.assertIs(self.named_parent, replacement_parent)
        self.assertIs(self.owner._created_directory, cap)
        self.assertIsNone(cap.fd)
        self.assertIsNone(cap.parent_fd)
        self.assertCountEqual([call.args[0] for call in self.fake_os.close.call_args_list], fds)
        self.gate.release.assert_called_once()
        self.fake_os.unlink.assert_not_called()

    def test_parent_fstat_failure_keeps_fd_owned_and_close_preserves_primary(self):
        primary = OSError("parent fstat failed")
        with patch("managed_outputs.os.fstat", side_effect=primary):
            with self.assertRaises(OSError) as failure:
                self.owner.create_deferred_directory()
        self.assertIs(failure.exception, primary)
        self.assertEqual(self.owner._created_directory.parent_fd, 101)
        self.assertEqual((self.mkdir_calls, self.writes), (0, 0))
        self.assert_sticky_failure(primary)
        self.owner._close_created_directory(primary)
        self.assertEqual(self.events, [("close", 101)])

    def test_parent_identity_setup_failure_keeps_fd_owned_for_close(self):
        primary = MemoryError("parent identity setup failed")
        with patch("managed_outputs._CreatedOutputDirectory.directory_identity_of", side_effect=primary):
            with self.assertRaises(MemoryError) as failure:
                self.owner.create_deferred_directory()
        self.assertIs(failure.exception, primary)
        self.assertEqual(self.owner._created_directory.parent_fd, 101)
        self.owner._close_created_directory(primary)
        self.assertEqual(self.events, [("close", 101)])
        self.assertEqual((self.mkdir_calls, self.writes), (0, 0))

    def test_cap_initialization_failure_precedes_any_fd_open(self):
        primary = MemoryError("cap initialization failed")
        with patch("managed_outputs._CreatedOutputDirectory", side_effect=primary):
            with self.assertRaises(MemoryError) as failure:
                self.owner.create_deferred_directory()
        self.assertIs(failure.exception, primary)
        self.fake_os.open.assert_not_called()
        self.assertEqual((self.mkdir_calls, self.writes), (0, 0))
        self.assert_sticky_failure(primary)

    def test_prepublication_sync_failure_preserves_null_row_and_directory(self):
        self.sync_error = OSError("created directory sync failed")
        with self.assertRaises(OSError) as failure:
            self.owner.create_deferred_directory()
        self.assertIs(failure.exception, self.sync_error)
        self.assertEqual((self.mkdir_calls, self.writes), (1, 0))
        self.assertIsNone(self.disk["identity"])
        self.assertEqual(self.named_child.st_ino, 30)
        self.assert_sticky_failure(failure.exception)

    def test_replace_then_fsync_failure_is_not_rolled_back_or_finished(self):
        self.write_error = OSError("catalog directory sync failed")
        with self.assertRaises(OSError) as failure:
            self.owner.create_deferred_directory()
        self.assertIs(failure.exception, self.write_error)
        self.assertEqual(self.owner._directory_binding_stage, "catalog_write_pending")
        self.assertEqual(self.disk["identity"], [7, 30])
        self.assertIsNone(self.owner.record["identity"])
        self.assert_sticky_failure(failure.exception)

    def test_readback_and_postwrite_budget_errors_never_release_payload(self):
        self.readback_error = StorageError("readback unavailable")
        with self.assertRaises(StorageError) as failure:
            self.owner.create_deferred_directory()
        self.assert_sticky_failure(failure.exception)
        self.assertEqual(self.owner._directory_binding_stage, "catalog_written_readback_pending")

    def test_postwrite_budget_overflow_keeps_identity_but_blocks_finish(self):
        original_write = self.write
        def grow_after_write(path, row):
            original_write(path, row)
            self.size = 5000
        with patch("managed_outputs.write_record", side_effect=grow_after_write):
            with self.assertRaises(OutputBudgetError) as failure:
                self.owner.create_deferred_directory()
        self.assertEqual(self.disk["identity"], [7, 30])
        self.assert_sticky_failure(failure.exception)

    def test_actual_catalog_encoding_bound_prevents_write_after_create(self):
        size = len((json.dumps(dict(self.owner.record, padding=""), sort_keys=True) + "\n").encode())
        self.owner.record["padding"] = "x" * (8191 - size)
        self.disk = copy.deepcopy(self.owner.record)
        with self.assertRaisesRegex(StorageError, "record exceeds bound") as failure:
            self.owner.create_deferred_directory()
        self.assertEqual((self.mkdir_calls, self.writes), (1, 0))
        self.assertIsNone(self.disk["identity"])
        self.assert_sticky_failure(failure.exception)

    def test_legacy_deferred_regular_file_and_absent_path_remain_compatible(self):
        with patch("managed_outputs.identity", return_value=[7, 88]):
            self.named_child = self.info(88, stat.S_IFREG | 0o600)
            self.owner.finish(status="completed", tree_gone=True)
        self.assertEqual(self.disk["identity"], [7, 88])
        self.assertEqual(self.writes, 1)
        self.assertIsNone(self.owner._created_directory)
        self.assertEqual(self.events, [("catalog_write", 1)])

    def test_legacy_deferred_absent_row_rejects_unrelated_later_path(self):
        self.owner.finish(status="not_started", tree_gone=True)
        self.assertIsNone(self.disk["identity"])
        with patch.object(Path, "iterdir", return_value=iter([self.row_path])):
            self.assertEqual(len(OutputLease._rows(self.owner)), 1)
        self.named_child = self.info(99, stat.S_IFREG | 0o600)
        with patch.object(Path, "iterdir", return_value=iter([self.row_path])):
            with self.assertRaisesRegex(StorageError, "custody changed"):
                OutputLease._rows(self.owner)
        self.assertEqual(self.writes, 1)
        self.assertEqual(self.named_child.st_ino, 99)

    def test_group_or_other_writable_parent_is_rejected_before_mkdir(self):
        self.parent_info.st_mode = stat.S_IFDIR | 0o777
        with self.assertRaisesRegex(StorageError, "without group/other writers") as failure:
            self.owner.create_deferred_directory()
        self.assertEqual((self.mkdir_calls, self.writes), (0, 0))
        self.assert_sticky_failure(failure.exception)

    def test_descriptor_close_failure_preserves_primary_write_error(self):
        self.owner.create_deferred_directory()
        self.write_error = OSError("original finish write failed")
        with patch("managed_outputs.os.close", side_effect=OSError("secondary FD close failed")):
            with self.assertRaises(OSError) as failure:
                self.owner.finish(status="completed", tree_gone=True)
        self.assertIs(failure.exception, self.write_error)
        self.assertIn("close also failed", failure.exception.__notes__[0])
        self.gate.release.assert_called_once()

    def test_initial_finish_storage_failure_closes_original_resources_without_rewriting(self):
        self.owner.create_deferred_directory()
        cap = self.owner._created_directory
        opened = [cap.fd, cap.parent_fd]
        record, disk = self.owner.record, copy.deepcopy(self.disk)
        named, named_before = self.named_child, copy.deepcopy(vars(self.named_child))
        writes = self.writes
        primary = StorageError("initial preservation budget observation unavailable")
        journal, view = Mock(), Mock()
        self.owner._preservation_journal, self.owner._quarantine_view = journal, view
        with patch.object(self.owner, "check_budget", side_effect=primary):
            with self.assertRaises(StorageError) as failure:
                self.owner.finish(status="must_remain_unpublished", tree_gone=True)
        self.assertIs(failure.exception, primary)
        self.assertIs(self.owner.record, record)
        self.assertEqual(self.disk, disk)
        self.assertEqual(self.writes, writes)
        self.assertIs(self.named_child, named)
        self.assertEqual(vars(self.named_child), named_before)
        self.assertIs(self.owner._created_directory, cap)
        self.assertIsNone(cap.fd)
        self.assertIsNone(cap.parent_fd)
        self.assertCountEqual([call.args[0] for call in self.fake_os.close.call_args_list], opened)
        journal.close.assert_called_once_with(primary)
        view.close.assert_called_once_with(primary)
        self.gate.release.assert_called_once()
        self.fake_os.unlink.assert_not_called()

    def test_nondeferred_lifetime_failure_precedes_directories_rows_and_guard_acquire(self):
        target, catalog = self.base / "unused-output", self.base / "unused-catalog"
        owner = OutputLease(target, self.owner.source, catalog=catalog, deferred=False)
        owner._rows = Mock(return_value=[])
        primary = OSError("current owner lifetime unavailable")
        with patch("managed_outputs.lifetime", side_effect=primary) as observe, \
                patch.object(Path, "mkdir") as mkdir, \
                patch("managed_outputs.write_record") as write:
            with self.assertRaises(OSError) as failure:
                owner.acquire()
        self.assertIs(failure.exception, primary)
        observe.assert_called_once_with()
        mkdir.assert_not_called()
        write.assert_not_called()
        owner._rows.assert_not_called()
        self.gate.acquire.assert_not_called()
        self.gate.release.assert_not_called()
        self.assertIsNone(owner.record)
        self.assertFalse(target.exists())
        self.assertFalse(catalog.exists())

    def test_descriptor_close_failure_does_not_mask_recorded_budget_error(self):
        self.owner.create_deferred_directory()
        self.size = 5000
        with patch("managed_outputs.os.close", side_effect=OSError("secondary FD close failed")):
            with self.assertRaises(OutputBudgetError) as failure:
                self.owner.finish(status="completed", tree_gone=True)
        self.assertIn("close also failed", failure.exception.__notes__[0])
        self.assertEqual(self.disk["status"], "failed_output_budget")
        self.gate.release.assert_called_once()

    def test_normal_descriptor_close_failure_is_not_success(self):
        self.owner.create_deferred_directory()
        with patch("managed_outputs.os.close", side_effect=OSError("FD close failed")):
            with self.assertRaisesRegex(OSError, "FD close failed"):
                self.owner.finish(status="completed", tree_gone=True)
        self.gate.release.assert_called_once()

    def test_unsupported_host_does_not_create_or_publish(self):
        with patch("managed_outputs.sys.platform", "win32"):
            with self.assertRaisesRegex(StorageError, "original active Linux"):
                self.owner.create_deferred_directory()
        self.assertEqual((self.mkdir_calls, self.writes), (0, 0))

    def test_recovery_does_not_fill_historical_null_identity(self):
        self.named_child = self.info(30)
        rebooted = dict(self.life, boot="new-boot")
        with patch("managed_outputs.lifetime", return_value=rebooted):
            with self.assertRaisesRegex(StorageError, "termination not proven"):
                recover_output(self.catalog, self.owner.key, self.snapshot(self.row_path, schema="rovezero.managed-output.v1")[1])
        self.assertEqual(self.writes, 0)
        self.assertIsNone(self.disk["identity"])

    def test_reboot_recovery_uses_sealed_identity_and_same_boot_stays_rejected(self):
        self.owner.create_deferred_directory()
        digest = self.snapshot(self.row_path, schema="rovezero.managed-output.v1")[1]
        with patch("managed_outputs.identity", return_value=[7, 30]):
            with self.assertRaisesRegex(StorageError, "termination not proven"):
                recover_output(self.catalog, self.owner.key, digest)
            with patch("managed_outputs.lifetime", return_value=dict(self.life, boot="new-boot", host="other-host")):
                with self.assertRaisesRegex(StorageError, "termination not proven"):
                    recover_output(self.catalog, self.owner.key, digest)
            with patch("managed_outputs.lifetime", return_value=dict(self.life, boot="new-boot")):
                with self.assertRaisesRegex(StorageError, "digest changed"):
                    recover_output(self.catalog, self.owner.key, "0" * 64)
                with patch("managed_outputs.identity", return_value=[7, 99]):
                    with self.assertRaisesRegex(StorageError, "termination not proven"):
                        recover_output(self.catalog, self.owner.key, digest)
                result = recover_output(self.catalog, self.owner.key, digest)
        self.assertEqual(result["state"], "interrupted_retained")
        self.assertEqual(result["identity"], [7, 30])
        self.assertNotIn("physical_completion", result)

    def test_active_bound_row_blocks_overlapping_root_admission(self):
        self.owner.create_deferred_directory()
        other = OutputLease(self.target / "nested", self.owner.source, catalog=self.catalog)
        other._rows = Mock(return_value=[(self.row_path, copy.deepcopy(self.disk), self.target)])
        attempted_directories = []
        def record_mkdir(path, *args, **kwargs):
            attempted_directories.append(path)
        with patch.object(Path, "mkdir", autospec=True, side_effect=record_mkdir):
            with self.assertRaisesRegex(StorageError, "overlapping output"):
                other.acquire()
        self.assertNotIn(other.output, attempted_directories)
        self.assertTrue(all(path == self.catalog for path in attempted_directories))
        self.assertEqual(self.writes, 1)
        self.assertEqual(self.disk["state"], "active")


@unittest.skipUnless(sys.platform == "linux", "real Linux directory descriptors and private temporary filesystem")
class CreatedDirectoryLinuxFilesystemTests(unittest.TestCase):
    """Only this test's private source/catalog/output; no production assets."""
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="rovezero-created-directory-")
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name).resolve()
        self.base.chmod(0o700)
        self.source = self.base / "source"
        self.source.mkdir(mode=0o700)
        self.owners = []
        self.addCleanup(self.close_owned_descriptors)

    def close_owned_descriptors(self):
        first = None
        for owner in reversed(self.owners):
            try:
                owner._close_created_directory()
            except OSError as error:
                if first is None:
                    first = error
            finally:
                owner.guard.release()
        if first is not None:
            raise first

    def lease(self, label):
        parent, catalog = self.base / (label + "-parent"), self.base / (label + "-catalog")
        parent.mkdir(mode=0o700)
        parent.chmod(0o700)
        catalog.mkdir(mode=0o700)
        owner = OutputLease(parent / "attempt", self.source, catalog=catalog,
                            deferred=True, reserve=4096, total_limit=8192).acquire()
        self.owners.append(owner)
        return owner, parent, catalog / (owner.key + ".json")

    def test_real_create_seals_before_payload_and_finish_closes_original_fds(self):
        owner, _, row_path = self.lease("positive")
        self.assertIsNone(json.loads(row_path.read_text())["identity"])
        self.assertFalse(owner.output.exists())
        self.assertEqual(owner.create_deferred_directory(), owner.output)
        opened = os.fstat(owner._created_directory.fd)
        original_identity = [opened.st_dev, opened.st_ino]
        self.assertEqual(json.loads(row_path.read_text())["identity"], original_identity)
        self.assertEqual([owner.output.stat().st_dev, owner.output.stat().st_ino], original_identity)
        self.assertEqual(stat.S_IMODE(owner.output.stat().st_mode), 0o700)
        self.assertFalse((owner.output / ".rovezero-output.json").exists())
        payload = owner.output / "payload.txt"
        with payload.open("xb") as stream:
            stream.write(b"private finite fixture")
        owner.check_budget()
        result = owner.finish(status="temporary_fixture_written", tree_gone=True)
        self.assertEqual(result["state"], "retained")
        self.assertEqual(result["identity"], original_identity)
        self.assertEqual(result["physical_completion"], "not_inferred_from_output_lifecycle")
        self.assertEqual(payload.read_bytes(), b"private finite fixture")
        self.assertIsNone(owner._created_directory.fd)
        self.assertIsNone(owner._created_directory.parent_fd)
        self.assertIsNone(owner.guard.stream)

    def test_real_target_and_parent_rename_replacement_never_rebinds(self):
        for replace_parent in (False, True):
            with self.subTest(replace_parent=replace_parent):
                label = "parent" if replace_parent else "target"
                owner, parent, row_path = self.lease(label)
                owner.create_deferred_directory()
                (owner.output / "original.txt").write_bytes(b"keep original")
                original_identity, original_pin = list(owner.record["identity"]), pin(row_path)
                if replace_parent:
                    moved_parent = self.base / "moved-parent"
                    parent.rename(moved_parent)
                    parent.mkdir(mode=0o700)
                    original_path = moved_parent / "attempt"
                else:
                    original_path = parent / "moved-attempt"
                    owner.output.rename(original_path)
                owner.output.mkdir(mode=0o700)
                (owner.output / "replacement.txt").write_bytes(b"keep replacement")
                message = "parent changed" if replace_parent else "directory changed"
                with self.assertRaisesRegex(StorageError, message):
                    owner.check_budget()
                cap = owner._created_directory
                opened = os.fstat(cap.fd)
                self.assertEqual([opened.st_dev, opened.st_ino], original_identity)
                self.assertIsNotNone(cap.parent_fd)
                self.assertIsNotNone(owner.guard.stream)
                self.assertEqual(pin(row_path), original_pin)
                self.assertEqual((original_path / "original.txt").read_bytes(), b"keep original")
                self.assertEqual((owner.output / "replacement.txt").read_bytes(), b"keep replacement")
                with self.assertRaisesRegex(StorageError, message):
                    owner.finish(status="must_not_finish", tree_gone=True)
                self.assertEqual(pin(row_path), original_pin)
                self.assertEqual(json.loads(row_path.read_text())["state"], "active")
                self.assertIs(owner._created_directory, cap)
                self.assertIsNone(cap.fd)
                self.assertIsNone(cap.parent_fd)
                self.assertIsNone(owner.guard.stream)
                self.assertEqual((original_path / "original.txt").read_bytes(), b"keep original")
                self.assertEqual((owner.output / "replacement.txt").read_bytes(), b"keep replacement")

    def test_real_same_boot_recovery_rejects_bound_active_row(self):
        owner, _, row_path = self.lease("same-boot")
        owner.create_deferred_directory()
        (owner.output / "payload.txt").write_bytes(b"keep same boot")
        digest = pin(row_path)
        owner._close_created_directory()
        owner.guard.release()  # Releasing an OS gate is not a reboot/closure witness.
        with self.assertRaisesRegex(StorageError, "termination not proven"):
            recover_output(owner.catalog, owner.key, digest)
        self.assertEqual(pin(row_path), digest)
        self.assertEqual(json.loads(row_path.read_text())["state"], "active")
        self.assertEqual((owner.output / "payload.txt").read_bytes(), b"keep same boot")

    def test_real_existing_unsealed_directory_keeps_null_identity_and_payload(self):
        owner, _, row_path = self.lease("unsealed")
        owner.output.mkdir(mode=0o700)
        (owner.output / "unsealed.txt").write_bytes(b"keep unsealed fixture")
        digest = pin(row_path)
        with self.assertRaisesRegex(StorageError, "exclusive fresh"):
            owner.create_deferred_directory()
        self.assertIsNone(json.loads(row_path.read_text())["identity"])
        owner._close_created_directory()
        owner.guard.release()
        with self.assertRaisesRegex(StorageError, "termination not proven"):
            recover_output(owner.catalog, owner.key, digest)
        self.assertEqual(pin(row_path), digest)
        self.assertIsNone(json.loads(row_path.read_text())["identity"])
        self.assertEqual((owner.output / "unsealed.txt").read_bytes(), b"keep unsealed fixture")


class InterruptedUnboundQuarantineTests(unittest.TestCase):
    """In-memory OS/metadata only; never applies maintenance to a real catalog."""
    def setUp(self):
        import managed_outputs as outputs
        import managed_output_quarantine as custody
        self.o, self.q = outputs, custody
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.base = Path(os.path.abspath("mock-unbound-custody"))
        self.catalog, self.target = self.base / "storage" / "outputs", self.base / "private" / "retained"
        self.registry = custody.registry_path(self.catalog)
        self.now = {"host": "host", "boot": "new", "platform": "linux", "process_birth": "new-birth"}
        self.key = hashlib.sha256(os.path.normcase(str(self.target)).encode()).hexdigest()
        self.row_path = self.catalog / (self.key + ".json")
        self.old = {"schema": "rovezero.managed-output.v1", "key": self.key, "output": str(self.target),
            "source": str(self.base / "source"), "state": "active", "deferred": True, "identity": None,
            "token": "original-only", "pid": 123, "lifetime": dict(self.now, boot="old", process_birth="old-birth"),
            "reserved_bytes": 4096, "catalog_limit_bytes": 18 * 1024 ** 3, "legacy_unknown": None}
        self.files, self.infos, self.fd_paths, self.actions = {}, {}, {}, []
        for path in (self.catalog, self.catalog.parent, self.target.parent, self.target, self.base / "source"):
            self.infos[path] = self.info(len(self.infos) + 10, stat.S_IFDIR | 0o700)
        self.source_path = Path(custody.__file__).resolve()
        self.put(self.source_path, b"independently pinned implementation")
        self.reset_old()
        self.source_pin = custody._pin(self.source_path, self.files[self.source_path])
        self.gate = MagicMock(spec=CatalogGuard)
        self.gate.path = self.catalog / "catalog.lock"
        self.gate.acquire.return_value = self.gate
        self.gate.__enter__.return_value = self.gate
        test = self
        class Held:
            def __init__(self, seconds):
                custody._need(0 < seconds <= 30, "original custody deadline")
                self.until, self.blobs = time.monotonic() + seconds, []
            def tick(self):
                custody._need(time.monotonic() < self.until, "original custody deadline")
            def directory(self, path):
                path = Path(path)
                custody._need(path in test.infos, "missing directory")
                fd = len(test.fd_paths) + 100
                test.fd_paths[fd] = path
                return fd
            def read(self, pin, path, maximum):
                self.tick()
                path = Path(path)
                test.assertFalse(path == test.target or test.target in path.parents, "target payload content read")
                custody._pin_shape(pin, path, maximum)
                raw = test.files[path]
                custody._need(custody._pin(path, raw) == pin, "raw pin mismatch")
                self.blobs.append((path, raw))
                return raw
            def check(self):
                self.tick()
                custody._need(all(test.files.get(path) == raw for path, raw in self.blobs), "held metadata changed")
            def close(self, primary=None):
                self.blobs.clear()
        class Observed:
            def __init__(self, held, target):
                self.target = target
                self.parent_identity = custody._did(test.infos[target.parent])
                self.identity = custody._did(test.infos[target])
                self.mounts = {"namespace": [1, 2], "mountinfo_bytes": 10, "mountinfo_sha256": "a" * 64}
                self.snapshot = test.manifest()
            def check(self):
                custody._need(self.snapshot == test.manifest() and self.identity == custody._did(test.infos[self.target])
                              and self.parent_identity == custody._did(test.infos[self.target.parent]), "snapshot changed")
        self.stack.enter_context(patch.object(custody, "_Held", Held))
        self.stack.enter_context(patch.object(custody, "_Observed", Observed))
        self.stack.enter_context(patch.object(custody, "checked", side_effect=lambda path: Path(path)))
        self.stack.enter_context(patch.object(outputs, "checked", side_effect=lambda path: Path(path)))
        self.stack.enter_context(patch.object(custody, "lifetime", side_effect=lambda: dict(self.now)))
        self.stack.enter_context(patch.object(outputs, "lifetime", side_effect=lambda: dict(self.now)))
        self.stack.enter_context(patch.object(custody, "CatalogGuard", return_value=self.gate))
        self.stack.enter_context(patch.object(outputs, "CatalogGuard", return_value=self.gate))
        self.stack.enter_context(patch.object(Path, "lstat", lambda path: self.infos[path] if path in self.infos else self.missing(path)))
        self.stack.enter_context(patch.object(Path, "exists", lambda path: path in self.infos))
        self.stack.enter_context(patch.object(Path, "mkdir", lambda path, *args, **kwargs: self.make(path, kwargs.get("exist_ok", False))))
        self.stack.enter_context(patch.object(Path, "iterdir", lambda path: iter([item for item in self.files if item.parent == path])))
        self.stack.enter_context(patch.object(outputs, "_read", side_effect=lambda path: custody._decode(self.files[path])))
        self.stack.enter_context(patch.object(outputs, "identity", side_effect=lambda path: custody._did(self.infos[path])[:2]))
        self.stack.enter_context(patch.object(outputs, "tree_size", side_effect=self.tree_bytes))
        self.stack.enter_context(patch.object(custody, "tree_size", side_effect=self.tree_bytes))
        self.stack.enter_context(patch.object(outputs, "write_record", side_effect=lambda path, row: self.put(path, custody._encode(row))))
        self.fake_os = SimpleNamespace(geteuid=lambda: 1000, path=os.path,
            scandir=self.scandir, fstat=lambda fd: self.infos[self.fd_paths[fd]],
            stat=self.stat_at, mkdir=lambda path, *args: self.make(path, False),
            fsync=lambda fd: None, major=lambda value: 0, minor=lambda value: value)
        self.stack.enter_context(patch.object(custody, "os", self.fake_os))
        self.stack.enter_context(patch.object(custody, "_exclusive", side_effect=self.exclusive))

    @staticmethod
    def missing(path):
        raise FileNotFoundError(str(path))

    @staticmethod
    def info(inode, mode, size=0):
        return SimpleNamespace(st_dev=7, st_ino=inode, st_mode=mode, st_uid=1000, st_gid=1000,
            st_nlink=1, st_size=size, st_mtime_ns=100, st_ctime_ns=100)

    def put(self, path, raw):
        path = Path(path)
        self.files[path] = raw
        self.infos[path] = self.info(len(self.infos) + 100, stat.S_IFREG | 0o600, len(raw))

    def reset_old(self):
        self.old_raw = (json.dumps(self.old, indent=2) + chr(10)).encode()
        self.put(self.row_path, self.old_raw)
        self.old_pin = self.q._pin(self.row_path, self.old_raw)

    def make(self, path, exist_ok=False):
        path = Path(path)
        if path in self.infos:
            if not exist_ok:
                raise FileExistsError(str(path))
        else:
            self.infos[path] = self.info(len(self.infos) + 10, stat.S_IFDIR | 0o700)

    def manifest(self):
        return {"schema": self.q.SCHEMA, "kind": "snapshot", "entry_count": 1, "regular_bytes": 8,
            "entries": [{"name": ".", "stat": self.q._stamp(self.infos[self.target])}]}

    def stat_at(self, name, *, dir_fd=None, follow_symlinks=True):
        path = Path(name) if dir_fd is None else self.fd_paths[dir_fd] / name
        return self.infos[path] if path in self.infos else self.missing(path)

    def scandir(self, fd):
        root = self.fd_paths[fd]
        entries = [SimpleNamespace(name=path.name) for path in self.files if path.parent == root]
        class Scan:
            def __enter__(self):
                return iter(entries)
            def __exit__(self, *args):
                return False
        return Scan()

    def tree_bytes(self, path, **kwargs):
        if path == self.target:
            return 8
        return sum(len(raw) for item, raw in self.files.items() if item == path or path in item.parents)

    def exclusive(self, held, fd, name, raw, maximum):
        held.tick()
        path = self.fd_paths[fd] / name
        self.assertNotEqual(path, self.row_path)
        self.assertLessEqual(len(raw), maximum)
        if path in self.files:
            raise FileExistsError(str(path))
        self.actions.append(name)
        self.put(path, raw)

    def publish(self):
        result = self.q.publish_custody_quarantine(self.catalog, self.old_pin, self.source_pin,
            root_exclusive_writer=True, apply=True)
        self.assertEqual(self.files[self.row_path], self.old_raw)
        return result["receipt"]

    def view(self, pins):
        return self.q.validate_quarantines(self.catalog, self.o._catalog_rows(self.catalog), pins, self.gate)

    def test_publication_marker_last_original_bytes_and_authority_unchanged(self):
        receipt_pin = self.publish()
        self.assertEqual([name.split(".")[-2] for name in self.actions], ["original", "snapshot", "receipt"])
        receipt = self.q._decode(self.files[Path(receipt_pin["path"])])
        self.assertIsNone(self.q._decode(self.files[self.row_path])["identity"])
        self.assertFalse(receipt["original_generation_custody_reissued"])
        self.assertFalse(receipt["original_generation_identity_known"])
        self.assertEqual(receipt["physical_completion"], "unknown_unchanged")
        view = self.view([receipt_pin])
        try:
            self.assertEqual(view.keys, frozenset([self.key]))
            self.assertEqual(view.usage(self.o._catalog_rows(self.catalog)), 4096 + sum(len(raw) for path, raw in self.files.items() if path.parent == self.registry))
        finally:
            view.close()

    def test_default_admission_blocks_and_explicit_boundaries_preserve_owner(self):
        receipt = self.publish()
        with self.assertRaisesRegex(StorageError, "overlapping output"):
            OutputLease(self.target / "nested", self.base / "source", catalog=self.catalog).acquire()
        self.assertNotIn(self.target / "nested", self.infos)
        owner = OutputLease(self.base / "fresh", self.base / "source", catalog=self.catalog,
                            quarantine_receipts=[receipt], reserve=4096).acquire()
        self.addCleanup(owner._close_quarantine_view)
        owner_path = self.catalog / (owner.key + ".json")
        owner_record, owner_raw = copy.deepcopy(owner.record), self.files[owner_path]
        owner.check_budget()
        amount = sample_output_budget(self.catalog, quarantine_receipts=[receipt], quarantine_guard=owner.guard)
        self.assertGreaterEqual(amount, 4096)
        self.assertEqual(owner.record, owner_record)
        self.assertEqual(self.files[owner_path], owner_raw)
        self.assertEqual(self.files[self.row_path], self.old_raw)
        owner.finish(status="finite-mock", tree_gone=True)
        self.assertEqual(self.files[self.row_path], self.old_raw)

    def test_outer_and_receipt_selection_rejected_before_any_path_io(self):
        with patch.object(self.o, "checked", side_effect=AssertionError("no IO")) as read:
            with self.assertRaisesRegex(StorageError, "outer finalizer"):
                OutputLease(self.base / "unused", self.base / "source", outer_supervised=True, quarantine_receipts=[])
        read.assert_not_called()
        self.assertFalse(self.actions)

    def test_same_boot_and_different_host_publication_refused(self):
        for key, value in (("boot", "old"), ("host", "other")):
            previous = self.now[key]
            self.now[key] = value
            with self.assertRaisesRegex(StorageError, "reboot contract"):
                self.publish()
            self.now[key] = previous
        self.assertFalse(self.actions)
        self.assertEqual(self.files[self.row_path], self.old_raw)

    def test_no_implicit_receipt_selection_wrong_pin_and_boot_drift(self):
        receipt = self.publish()
        with self.assertRaisesRegex(StorageError, "explicit exact selection"):
            self.view([])
        with self.assertRaisesRegex(StorageError, "raw pin mismatch"):
            self.view([dict(receipt, sha256="0" * 64)])
        self.now["boot"] = "later-boot"
        with self.assertRaisesRegex(StorageError, "current boot mismatch"):
            self.view([receipt])

    def test_admission_budget_and_sampling_reject_changed_inputs_without_mutation(self):
        receipt = self.publish()
        receipt_path = Path(receipt["path"])
        changes = (("receipt_bytes", receipt_path, None),
                   ("target_mtime", self.target, "st_mtime_ns"),
                   ("target_identity", self.target, "st_ino"),
                   ("parent_metadata", self.target.parent, "st_mode"))
        for boundary in ("acquire", "check_budget", "sample_output_budget"):
            for name, path, field in changes:
                with self.subTest(boundary=boundary, changed=name):
                    owner = OutputLease(self.base / (boundary + "-" + name), self.base / "source",
                                        catalog=self.catalog, quarantine_receipts=[receipt], reserve=4096)
                    if boundary != "acquire":
                        owner.acquire()
                        self.addCleanup(owner._close_quarantine_view)
                    owner_path = self.catalog / (owner.key + ".json")
                    owner_record = copy.deepcopy(owner.record)
                    if field is None:
                        original = self.files[path]
                        self.files[path] = original + b" "
                    else:
                        original = getattr(self.infos[path], field)
                        setattr(self.infos[path], field, original + 1)
                    files_before = dict(self.files)
                    metadata_before = {key: vars(value).copy() for key, value in self.infos.items()}
                    actions_before = list(self.actions)
                    try:
                        with self.assertRaises(StorageError):
                            if boundary == "acquire":
                                owner.acquire()
                            elif boundary == "check_budget":
                                owner.check_budget()
                            else:
                                sample_output_budget(self.catalog, quarantine_receipts=[receipt],
                                                     quarantine_guard=owner.guard)
                        self.assertEqual(self.files, files_before)
                        self.assertEqual({key: vars(value) for key, value in self.infos.items()}, metadata_before)
                        self.assertEqual(self.actions, actions_before)
                        self.assertEqual(self.files[self.row_path], self.old_raw)
                        self.assertEqual(owner.record, owner_record)
                        if boundary == "acquire":
                            self.assertNotIn(owner_path, self.files)
                            self.assertNotIn(owner.output, self.infos)
                        else:
                            self.assertEqual(self.q._decode(self.files[owner_path]), owner_record)
                            self.assertEqual(owner_record["state"], "active")
                    finally:
                        if field is None:
                            self.files[path] = original
                        else:
                            setattr(self.infos[path], field, original)
                    if boundary != "acquire":
                        owner.finish(status="finite-mock", tree_gone=True)

    def test_marker_before_complete_publication_never_opens_admission(self):
        original = self.exclusive
        def fail_snapshot(held, fd, name, raw, maximum):
            if name.endswith(".snapshot.json"):
                raise OSError("snapshot sync failed")
            original(held, fd, name, raw, maximum)
        with patch.object(self.q, "_exclusive", side_effect=fail_snapshot):
            with self.assertRaisesRegex(OSError, "snapshot sync failed"):
                self.publish()
        self.assertEqual(len(self.actions), 1)
        self.assertEqual(self.files[self.row_path], self.old_raw)
        with self.assertRaisesRegex(StorageError, "partial custody"):
            self.view([])

    def test_charge_keeps_reservation_and_warns_when_metadata_exceeds_threshold(self):
        self.old["reserved_bytes"] = 18 * 1024 ** 3
        self.reset_old()
        receipt = self.publish()
        self.assertGreater(self.view([receipt]).usage(self.o._catalog_rows(self.catalog)), 18 * 1024 ** 3)
        self.assertEqual(self.files[self.row_path], self.old_raw)

    def test_old_path_ancestors_descendants_and_other_active_row_stay_blocked(self):
        receipt = self.publish()
        for target in (self.target, self.target.parent, self.target / "child"):
            with self.assertRaisesRegex(StorageError, "overlapping"):
                OutputLease(target, self.base / "source", catalog=self.catalog, quarantine_receipts=[receipt]).acquire()
        other_target = self.base / "another-active"
        other_key = hashlib.sha256(os.path.normcase(str(other_target)).encode()).hexdigest()
        other = dict(self.old, key=other_key, output=str(other_target))
        self.put(self.catalog / (other_key + ".json"), self.q._encode(other))
        with self.assertRaisesRegex(StorageError, "interrupted output"):
            OutputLease(self.base / "new-root", self.base / "source", catalog=self.catalog, quarantine_receipts=[receipt]).acquire()

    def test_registry_unknown_symlink_hardlink_and_metadata_bounds_refused(self):
        receipt = self.publish()
        receipt_path = Path(receipt["path"])
        for field, value in (("st_mode", stat.S_IFLNK | 0o777), ("st_nlink", 2), ("st_size", 8193)):
            old = getattr(self.infos[receipt_path], field)
            setattr(self.infos[receipt_path], field, value)
            with self.assertRaisesRegex(StorageError, "registry file"):
                self.view([receipt])
            setattr(self.infos[receipt_path], field, old)
        self.put(self.registry / "unknown.json", b"x")
        with self.assertRaisesRegex(StorageError, "unknown custody registry"):
            self.view([receipt])



    def test_held_guard_sampling_never_reacquires_and_busy_independent_sampling_fails(self):
        receipt = self.publish()
        with patch.object(self.o, "CatalogGuard", side_effect=AssertionError("no second live gate")):
            self.assertGreater(sample_output_budget(self.catalog, quarantine_receipts=[receipt], quarantine_guard=self.gate), 4096)
        self.gate.assert_held.assert_called()
        with patch.object(self.q, "CatalogGuard", side_effect=StorageError("catalog gate busy")):
            with self.assertRaisesRegex(StorageError, "gate busy"):
                sample_output_budget(self.catalog, quarantine_receipts=[receipt])
        with self.assertRaisesRegex(StorageError, "without explicit"):
            sample_output_budget(self.catalog, quarantine_guard=self.gate)

    def test_original_row_identity_raw_and_receipt_meaning_changes_refused(self):
        receipt = self.publish()
        row_info = self.infos[self.row_path]
        row_info.st_ino += 1
        with self.assertRaisesRegex(StorageError, "original catalog metadata"):
            self.view([receipt])
        row_info.st_ino -= 1
        raw = self.files[self.row_path]
        self.files[self.row_path] = raw + b" "
        with self.assertRaisesRegex(StorageError, "raw pin mismatch"):
            self.view([receipt])
        self.files[self.row_path] = raw
        path = Path(receipt["path"])
        original = self.files[path]
        value = self.q._decode(original)
        for key, invalid in (("unregistered_field", True), ("original_generation_custody_reissued", True),
                             ("physical_completion", "complete")):
            changed = dict(value, **{key: invalid})
            self.files[path] = self.q._encode(changed)
            self.infos[path].st_size = len(self.files[path])
            with self.assertRaises(StorageError):
                self.view([self.q._pin(path, self.files[path])])
        self.files[path] = original
        self.infos[path].st_size = len(original)
        self.files[self.source_path] = b"different publisher implementation"
        with self.assertRaisesRegex(StorageError, "raw pin mismatch"):
            self.view([receipt])

    def test_receipt_write_failure_and_registry_caps_preserve_partial_evidence(self):
        original = self.exclusive
        def fail_marker(held, fd, name, raw, maximum):
            if name.endswith(".receipt.json"):
                raise OSError("receipt fsync uncertain")
            return original(held, fd, name, raw, maximum)
        with patch.object(self.q, "_exclusive", side_effect=fail_marker):
            with self.assertRaisesRegex(OSError, "receipt fsync uncertain"):
                self.publish()
        self.assertEqual([name.split(".")[-2] for name in self.actions], ["original", "snapshot"])
        self.assertEqual(self.files[self.row_path], self.old_raw)
        with self.assertRaisesRegex(StorageError, "partial custody"):
            self.view([])
        with patch.object(self.q, "REGISTRY_LIMIT", 1):
            with self.assertRaisesRegex(StorageError, "registry bound"):
                self.view([])
        with patch.object(self.q, "MAX_RUNS", 0):
            with self.assertRaisesRegex(StorageError, "registry bound"):
                self.view([])

    def test_original_gate_release_failure_preserves_primary(self):
        primary = StorageError("original publication failure")
        self.gate.release.side_effect = OSError("secondary gate close failure")
        with self.assertRaisesRegex(StorageError, "original publication failure") as caught:
            with self.q.original_custody_gate(self.catalog):
                raise primary
        self.assertIs(caught.exception, primary)
        self.assertTrue(any("gate release also failed" in note for note in primary.__notes__))

    def test_registry_equal_parent_child_outputs_refused_before_mkdir_or_row_write(self):
        receipt = self.publish()
        before = dict(self.files)
        with patch.object(Path, "mkdir", side_effect=AssertionError("no target creation")) as create, \
                patch.object(self.o, "write_record", side_effect=AssertionError("no row write")) as write:
            for target in (self.registry, self.registry / "child", self.registry.parent):
                with self.subTest(target=target), self.assertRaisesRegex(StorageError, "overlaps custody registry"):
                    OutputLease(target, self.base / "source", catalog=self.catalog,
                                quarantine_receipts=[receipt])
        create.assert_not_called()
        write.assert_not_called()
        self.assertEqual(self.files, before)
        self.assertEqual(self.files[self.row_path], self.old_raw)

    def test_common_verifier_refuses_existing_rows_overlapping_registry(self):
        receipt = self.publish()
        rows = self.o._catalog_rows(self.catalog)
        for target in (self.registry, self.registry / "child", self.registry.parent):
            key = hashlib.sha256(os.path.normcase(str(target)).encode()).hexdigest()
            other = dict(self.old, key=key, output=str(target))
            joined = rows + [(self.catalog / (key + ".json"), other, target)]
            with self.subTest(target=target), self.assertRaisesRegex(StorageError, "row overlaps custody registry"):
                self.q.validate_quarantines(self.catalog, joined, [receipt], self.gate)
        self.assertEqual(self.files[self.row_path], self.old_raw)

    def test_legacy_recovery_refuses_final_partial_empty_or_broken_registry_before_row_read(self):
        receipt = self.publish()
        self.infos.pop(self.target)  # Fault: original target name is now absent.
        artifacts = {path: raw for path, raw in self.files.items() if path.parent == self.registry}
        cases = (("final", stat.S_IFDIR | 0o700, artifacts),
                 ("partial", stat.S_IFDIR | 0o700, {path: raw for path, raw in artifacts.items() if path.name.endswith(".snapshot.json")}),
                 ("empty", stat.S_IFDIR | 0o700, {}),
                 ("broken-symlink", stat.S_IFLNK | 0o777, {}),
                 ("regular", stat.S_IFREG | 0o600, {}))
        for label, mode, selected in cases:
            for path in artifacts:
                self.files.pop(path, None)  # In-memory fault state only; no filesystem deletion.
            self.files.update(selected)
            self.infos[self.registry].st_mode = mode
            with self.subTest(case=label), \
                    patch.object(self.o, "read_record_snapshot", side_effect=AssertionError("no original row read")) as read, \
                    patch.object(self.o, "write_record", side_effect=AssertionError("no original row write")) as write:
                with self.assertRaisesRegex(StorageError, "legacy output recovery is forbidden"):
                    recover_output(self.catalog, self.key, self.old_pin["sha256"])
                read.assert_not_called()
                write.assert_not_called()
            self.assertEqual(self.files[self.row_path], self.old_raw)

    def test_registry_absent_legacy_recovery_keeps_original_absent_deferred_behavior(self):
        self.infos.pop(self.target)
        snapshot = (copy.deepcopy(self.old), self.old_pin["sha256"], "mock-stamp")
        with patch.object(self.o, "read_record_snapshot", return_value=snapshot):
            result = recover_output(self.catalog, self.key, self.old_pin["sha256"])
        self.assertEqual(result["state"], "interrupted_retained")
        self.assertIsNone(result["identity"])
        self.assertNotIn(self.registry, self.infos)

    def assert_postwrite_refusal(self, mutation, expected, secondary_close=False):
        receipt = self.publish()
        owner = OutputLease(self.base / "late-write", self.base / "source", catalog=self.catalog,
                            deferred=True, quarantine_receipts=[receipt], reserve=4096)
        self.addCleanup(owner._close_quarantine_view)
        row_path = self.catalog / (owner.key + ".json")
        def write_then_change(path, row):
            self.put(path, self.q._encode(row))
            if path == row_path:
                mutation(owner)
        if secondary_close:
            self.gate.release.side_effect = OSError("secondary release failure")
        with patch.object(self.o, "write_record", side_effect=write_then_change):
            with self.assertRaisesRegex(StorageError, expected) as failure:
                owner.acquire()
        self.assertIn(row_path, self.files)
        self.assertEqual(self.q._decode(self.files[row_path])["state"], "active")
        self.assertFalse(owner.output.exists())
        self.assertEqual(self.files[self.row_path], self.old_raw)
        if secondary_close:
            self.assertTrue(any("gate release also failed" in note for note in failure.exception.__notes__))

    def test_postwrite_expired_observation_refuses_admission_without_rollback(self):
        self.assert_postwrite_refusal(lambda owner: setattr(owner._quarantine_view.held, "until", 0), "original custody deadline")

    def test_postwrite_boot_change_refuses_admission_without_rollback(self):
        self.assert_postwrite_refusal(lambda owner: self.now.update(boot="changed-during-write"), "current custody boot changed")

    def test_postwrite_snapshot_change_refuses_admission_without_rollback(self):
        self.assert_postwrite_refusal(lambda owner: setattr(self.infos[self.target], "st_mtime_ns", 900), "snapshot changed")

    def test_postwrite_failure_remains_primary_when_gate_release_also_fails(self):
        self.assert_postwrite_refusal(lambda owner: setattr(self.infos[self.target], "st_mtime_ns", 900),
                                     "snapshot changed", secondary_close=True)


class QuarantineMetadataBoundaryTests(unittest.TestCase):
    """Exercise production FD/metadata walkers against a closed in-memory OS."""
    def setUp(self):
        import managed_output_quarantine as custody
        self.q = custody
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.target = Path("/custody-metadata/private/old")
        self.nodes = {}
        for inode, path in enumerate((self.target.parent, self.target, self.target / "sub"), 10):
            self.nodes[path] = self.info(inode, stat.S_IFDIR | 0o700)
        self.file = self.target / "result.log"
        self.nodes[self.file] = self.info(20, stat.S_IFREG | 0o600, 8)
        self.nodes[self.target / "sub" / "more.log"] = self.info(21, stat.S_IFREG | 0o600, 9)
        self.fds, self.opened, self.closed = {}, [], []
        self.namespace = self.info(90, stat.S_IFREG | 0o400)
        self.mount_raw = b"1 0 0:7 / / rw - ext4 dev rw\n"
        real_os = os
        self.fake_os = SimpleNamespace(
            O_RDONLY=real_os.O_RDONLY, O_DIRECTORY=getattr(real_os, "O_DIRECTORY", 65536),
            O_NOFOLLOW=getattr(real_os, "O_NOFOLLOW", 131072), O_PATH=getattr(real_os, "O_PATH", 2097152),
            O_WRONLY=real_os.O_WRONLY, O_CREAT=real_os.O_CREAT, O_EXCL=real_os.O_EXCL,
            open=self.open, close=self.close, fstat=lambda fd: self.nodes[self.fds[fd]], stat=self.stat_at,
            scandir=self.scandir, geteuid=lambda: 1000, major=lambda device: 0, minor=lambda device: device,
            read=Mock(side_effect=AssertionError("payload reads forbidden")))
        self.stack.enter_context(patch.object(custody, "os", self.fake_os))
        self.stack.enter_context(patch.object(custody, "sys", SimpleNamespace(platform="linux", exc_info=sys.exc_info)))
        self.stack.enter_context(patch.object(custody, "checked", side_effect=lambda path: Path(path)))
        self.stack.enter_context(patch.object(custody, "_kernel", side_effect=lambda held, path, maximum: self.mount_raw))
        self.stack.enter_context(patch.object(Path, "lstat", lambda path: self.nodes[path]))
        self.held = custody._Held(30)
        self.addCleanup(self.held.close)

    @staticmethod
    def info(inode, mode, size=0):
        return SimpleNamespace(st_dev=7, st_ino=inode, st_mode=mode, st_uid=1000, st_gid=1000,
            st_nlink=1, st_size=size, st_atime_ns=100, st_mtime_ns=100, st_ctime_ns=100)

    def open(self, name, flags, *args, dir_fd=None):
        path = Path(name) if dir_fd is None else self.fds[dir_fd] / name
        if stat.S_ISREG(self.nodes[path].st_mode):
            self.assertTrue(flags & self.fake_os.O_PATH, "regular payload must use O_PATH only")
        self.assertTrue(flags & self.fake_os.O_NOFOLLOW)
        fd = max([99, *self.fds]) + 1
        self.fds[fd] = path
        self.opened.append((path, flags))
        return fd

    def close(self, fd):
        self.closed.append(fd)
        del self.fds[fd]

    def stat_at(self, name, *, dir_fd=None, follow_symlinks=True):
        if str(name) == "/proc/self/ns/mnt":
            return self.namespace
        path = Path(name) if dir_fd is None else self.fds[dir_fd] / name
        return self.nodes[path]

    def scandir(self, fd):
        children = [SimpleNamespace(name=path.name) for path in self.nodes if path.parent == self.fds[fd]]
        class Scan:
            def __enter__(self):
                return iter(children)
            def __exit__(self, *args):
                return False
        return Scan()

    def test_real_walker_uses_path_fds_atime_and_sibling_parent_changes_are_allowed(self):
        observed = self.q._Observed(self.held, self.target)
        self.assertEqual(observed.snapshot["regular_bytes"], 17)
        self.assertEqual(observed.snapshot["entry_count"], 4)
        self.nodes[self.file].st_atime_ns += 500
        self.nodes[self.target].st_atime_ns += 500
        parent = self.nodes[self.target.parent]
        parent.st_nlink += 1
        parent.st_size += 1
        parent.st_mtime_ns += 500
        parent.st_ctime_ns += 500
        self.nodes[self.target.parent / "fresh-sibling"] = self.info(30, stat.S_IFDIR | 0o700)
        observed.check()
        self.fake_os.read.assert_not_called()
        files = [flags for path, flags in self.opened if stat.S_ISREG(self.nodes[path].st_mode)]
        self.assertTrue(files and all(flags & self.fake_os.O_PATH for flags in files))
        self.held.close()
        self.assertFalse(self.fds)

    def test_target_roster_mtime_ctime_and_parent_or_target_identity_changes_refused(self):
        observed = self.q._Observed(self.held, self.target)
        for path, field in ((self.file, "st_mtime_ns"), (self.file, "st_ctime_ns"), (self.file, "st_size"),
                            (self.target, "st_nlink"), (self.target.parent, "st_ino"),
                            (self.target.parent, "st_mode"), (self.target, "st_ino")):
            before = getattr(self.nodes[path], field)
            setattr(self.nodes[path], field, before + 1)
            with self.subTest(path=path, field=field), self.assertRaises(StorageError):
                observed.check()
            setattr(self.nodes[path], field, before)
        added = self.target / "new.log"
        self.nodes[added] = self.info(50, stat.S_IFREG | 0o600, 1)
        with self.assertRaisesRegex(StorageError, "snapshot changed"):
            observed.check()

    def test_symlink_hardlink_special_cross_device_and_duplicate_inode_refused(self):
        fd = self.held.directory(self.target)
        for field, value in (("st_mode", stat.S_IFLNK | 0o777), ("st_mode", stat.S_IFSOCK | 0o600),
                             ("st_nlink", 2), ("st_dev", 8), ("st_uid", 1001), ("st_ino", 21)):
            before = getattr(self.nodes[self.file], field)
            setattr(self.nodes[self.file], field, value)
            with self.subTest(field=field, value=value), self.assertRaises(StorageError):
                self.q._snapshot(self.held, self.target, fd)
            setattr(self.nodes[self.file], field, before)
        self.fake_os.read.assert_not_called()

    def test_roster_entry_and_snapshot_serialization_caps_are_fail_closed(self):
        fd = self.held.directory(self.target)
        with patch.object(self.q, "MAX_ENTRIES", 2), self.assertRaisesRegex(StorageError, "bound exceeded"):
            self.q._snapshot(self.held, self.target, fd)
        with patch.object(self.q, "SNAPSHOT_LIMIT", 128), self.assertRaisesRegex(StorageError, "metadata bound"):
            self.q._snapshot(self.held, self.target, fd)

    def test_current_mount_namespace_alias_nested_stack_and_drift_are_refused(self):
        observed = self.q._Observed(self.held, self.target)
        original = self.mount_raw
        variants = (
            b"2 1 0:7 /custody-metadata/private /alias rw - ext4 dev rw\n",
            b"2 1 0:8 / /custody-metadata/private/old/sub rw - ext4 dev rw\n",
            b"2 1 0:7 / / rw - ext4 dev rw\n",
        )
        for extra in variants:
            self.mount_raw = original + extra
            with self.subTest(extra=extra), self.assertRaises(StorageError):
                self.q._mounts(self.held, self.target.parent, self.target, 7)
        self.mount_raw = original
        self.namespace.st_ino += 1
        with self.assertRaisesRegex(StorageError, "namespace/mount") as changed_namespace:
            observed.check()
        self.assertTrue(changed_namespace.exception.custody_mount_change_observation["namespace_changed"])
        self.namespace.st_ino -= 1
        self.mount_raw += b"3 1 0:8 / /unrelated rw - ext4 dev rw\n"
        with self.assertRaisesRegex(StorageError, "namespace/mount") as unrelated_mount:
            observed.check()
        diagnostic = unrelated_mount.exception.custody_mount_change_observation
        self.assertTrue(diagnostic["mountinfo_changed"])
        self.assertFalse(diagnostic["namespace_changed"])
        self.assertFalse(diagnostic["readonly_aliases_changed"])
        self.assertFalse(diagnostic["origin_changed"])
        self.assertNotIn("/unrelated", json.dumps(diagnostic))
        self.mount_raw = original.replace(b" / rw -", b" / rw,noexec -")
        with self.assertRaisesRegex(StorageError, "namespace/mount") as changed_origin:
            observed.check()
        self.assertTrue(changed_origin.exception.custody_mount_change_observation["origin_changed"])

    def test_directory_fd_cache_name_recheck_and_first_fstat_failure_close(self):
        first = self.held.directory(self.target)
        self.assertEqual(first, self.held.directory(self.target))
        self.assertEqual(len(self.held.fds), 1)
        info = self.nodes[self.target]
        self.nodes[self.target] = self.info(900, info.st_mode)
        with patch.object(self.fake_os, "fstat", return_value=info):
            with self.assertRaisesRegex(StorageError, "directory name changed"):
                self.held.directory(self.target)
        self.nodes[self.target] = info
        self.held.close()
        self.assertFalse(self.fds)
        with patch.object(self.fake_os, "fstat", side_effect=OSError("first fstat failed")):
            with self.assertRaisesRegex(OSError, "first fstat failed") as failure:
                try:
                    self.q._Observed(self.held, self.target)
                except BaseException as primary:
                    self.held.close(primary)
                    raise
        self.assertFalse(self.fds)

    def test_partial_exclusive_write_keeps_original_error_when_close_also_fails(self):
        primary = OSError("primary write failure")
        fake = SimpleNamespace(O_WRONLY=1, O_CREAT=2, O_EXCL=4, O_NOFOLLOW=8,
            open=Mock(return_value=200), write=Mock(side_effect=primary),
            close=Mock(side_effect=OSError("secondary close failure")), fsync=Mock())
        with patch.object(self.q, "os", fake):
            with self.assertRaisesRegex(OSError, "primary write failure") as caught:
                self.q._exclusive(self.held, 10, "receipt.json", b"bounded", 8192)
        self.assertIs(caught.exception, primary)
        self.assertTrue(any("close also failed" in note for note in primary.__notes__))
        fake.fsync.assert_not_called()

    def test_related_readonly_mount_alias_allows_rw_superblock_and_records_scope(self):
        self.mount_raw += b"2 1 0:7 / /readonly ro,relatime - ext4 dev rw,relatime\n"
        observed = self.q._Observed(self.held, self.target)
        aliases = observed.mounts["readonly_aliases"]
        self.assertEqual(len(aliases), 1)
        self.assertEqual(aliases[0]["mount_id"], "2")
        self.assertEqual(aliases[0]["mount_options"], "ro,relatime")
        self.assertEqual(aliases[0]["mountpoint"], str(Path("/readonly")))
        observed.check()
        self.fake_os.read.assert_not_called()

    def test_related_writable_or_unknown_mount_options_never_use_superblock_ro(self):
        original = self.mount_raw
        for mount, superblock in (("rw,relatime", "ro"), ("ro,rw", "ro"),
                                  ("relatime", "ro"), ("rofoo", "ro")):
            self.mount_raw = original + ("2 1 0:7 / /readonly " + mount + " - ext4 dev " + superblock + "\n").encode()
            with self.subTest(mount=mount, superblock=superblock), self.assertRaisesRegex(StorageError, "writable or unknown"):
                self.q._mounts(self.held, self.target.parent, self.target, 7)

    def test_readonly_alias_any_stacked_or_nested_mount_stays_forbidden(self):
        original = self.mount_raw + b"2 1 0:7 / /readonly ro - ext4 dev rw\n"
        for other in (b"3 1 0:8 / /readonly rw - other dev rw\n",
                      b"3 1 0:8 / /readonly/unrelated-child ro - other dev ro\n"):
            self.mount_raw = original + other
            with self.subTest(other=other), self.assertRaisesRegex(StorageError, "stacked readonly|nested readonly"):
                self.q._mounts(self.held, self.target.parent, self.target, 7)
        # Canonical target nesting remains forbidden even if that mount is ro.
        self.mount_raw = original + b"3 1 0:8 / /custody-metadata/private/old/sub ro - other dev ro\n"
        with self.assertRaisesRegex(StorageError, "nested custody|nested readonly"):
            self.q._mounts(self.held, self.target.parent, self.target, 7)

    def test_readonly_alias_options_drift_refuses_previously_observed_view(self):
        self.mount_raw += b"2 1 0:7 / /readonly ro,relatime - ext4 dev rw\n"
        observed = self.q._Observed(self.held, self.target)
        self.mount_raw = self.mount_raw.replace(b"/readonly ro,relatime", b"/readonly rw,relatime")
        with self.assertRaisesRegex(StorageError, "writable or unknown"):
            observed.check()


@unittest.skipUnless(sys.platform == "linux", "Linux actual FD/fsync/mount metadata only; Root executes")
class InterruptedUnboundQuarantineLinuxTests(unittest.TestCase):
    """Private fixtures, real verifier/FDs; lifetime injection is not a reboot proof."""
    def setUp(self):
        import managed_outputs as outputs
        import managed_output_quarantine as custody
        self.o, self.q = outputs, custody
        temporary = tempfile.TemporaryDirectory(prefix="rovezero-unbound-custody-")
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name).resolve()
        self.base.chmod(0o700)
        self.source, self.catalog = self.base / "source", self.base / "outputs"
        self.parent = self.base / "private"
        for path in (self.source, self.catalog, self.parent):
            path.mkdir(mode=0o700)
            path.chmod(0o700)
        self.target = self.parent / "old"
        self.target.mkdir(mode=0o700)
        (self.target / "original.log").write_bytes(b"retained fixture payload")
        nested = self.target / "sub"
        nested.mkdir(mode=0o700)
        (nested / "more.log").write_bytes(b"second fixture payload")
        # Byte-identical implementation copy gives this fixture a private source
        # owner/path. Only provenance location and lifetime are fault-injected;
        # the Linux verifier, metadata walker, mount checks and writes stay real.
        implementation = self.source / "managed_output_quarantine.py"
        implementation.write_bytes(Path(custody.__file__).read_bytes())
        implementation.chmod(0o600)
        self.source_pin = custody._pin(implementation, implementation.read_bytes())
        self.now = {"host": "fixture-only-host", "boot": "fixture-new-boot",
                    "platform": "linux", "process_birth": "fixture-process-birth"}
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(custody, "__file__", str(implementation)))
        self.stack.enter_context(patch.object(custody, "lifetime", side_effect=lambda: dict(self.now)))
        self.stack.enter_context(patch.object(outputs, "lifetime", side_effect=lambda: dict(self.now)))
        self.key = hashlib.sha256(os.path.normcase(str(self.target)).encode()).hexdigest()
        self.row_path = self.catalog / (self.key + ".json")
        self.row = {"schema": "rovezero.managed-output.v1", "key": self.key, "output": str(self.target),
            "source": str(self.source), "state": "active", "deferred": True, "identity": None,
            "token": "fixture-original-token", "pid": 123, "lifetime": dict(self.now, boot="fixture-old-boot"),
            "reserved_bytes": 1 << 20, "catalog_limit_bytes": 18 * 1024 ** 3,
            "original_result": {"failure": "preserved", "physical": "unknown"}}
        self.raw = (json.dumps(self.row, indent=2) + chr(10)).encode()
        self.row_path.write_bytes(self.raw)
        self.row_path.chmod(0o600)
        self.row_pin = custody._pin(self.row_path, self.raw)
        self.registry = custody.registry_path(self.catalog)

    def publish(self):
        result = self.q.publish_custody_quarantine(self.catalog, self.row_pin, self.source_pin,
            root_exclusive_writer=True, apply=True)
        self.assertEqual(self.row_path.read_bytes(), self.raw)
        return result["receipt"]

    def fresh_owner(self, receipt):
        owner = OutputLease(self.parent / "fresh", self.source, catalog=self.catalog,
                            reserve=1 << 20, deferred=True, quarantine_receipts=[receipt]).acquire()
        def close_descriptors_only():
            try:
                owner._close_created_directory()
            finally:
                try:
                    owner._close_quarantine_view()
                finally:
                    owner.guard.release()
        self.addCleanup(close_descriptors_only)
        return owner

    def assert_new_admission_refused(self, receipt):
        with self.assertRaises(StorageError):
            OutputLease(self.parent / "fresh", self.source, catalog=self.catalog,
                        deferred=True, quarantine_receipts=[receipt]).acquire()
        self.assertEqual(self.row_path.read_bytes(), self.raw)
        self.assertIsNone(json.loads(self.row_path.read_bytes())["identity"])
        self.assertFalse((self.parent / "fresh").exists())

    def test_actual_publication_marker_last_and_same_parent_sibling_producer(self):
        order = []
        original = self.q._exclusive
        def record(held, fd, name, raw, maximum):
            if name.endswith(".receipt.json"):
                self.assertTrue(any(self.registry.glob("*.original.json")))
                self.assertTrue(any(self.registry.glob("*.snapshot.json")))
                self.assertFalse(any(self.registry.glob("*.receipt.json")))
            original(held, fd, name, raw, maximum)
            order.append(name.split(".")[-2])
        with patch.object(self.q, "_exclusive", side_effect=record):
            receipt = self.publish()
        self.assertEqual(order, ["original", "snapshot", "receipt"])
        owner = self.fresh_owner(receipt)
        before_parent = self.parent.stat()
        owner.create_deferred_directory()
        (owner.output / "new.log").write_bytes(b"fresh fixture payload")
        self.assertTrue(owner.output.is_dir())
        self.assertEqual(self.q._did(self.parent.stat()), self.q._did(before_parent))
        owner.check_budget()
        sampled = sample_output_budget(self.catalog, quarantine_receipts=[receipt], quarantine_guard=owner.guard)
        self.assertGreaterEqual(sampled, self.row["reserved_bytes"])
        finished = owner.finish(status="fixture-no-child", tree_gone=True)
        self.assertEqual(finished["state"], "retained")
        self.assertEqual(finished["physical_completion"], "not_inferred_from_output_lifecycle")
        marker = json.loads(Path(receipt["path"]).read_bytes())
        self.assertEqual(marker["physical_completion"], "unknown_unchanged")
        self.assertFalse(marker["original_generation_custody_reissued"])
        self.assertEqual(self.row_path.read_bytes(), self.raw)

    def test_actual_target_rename_replacement_cannot_rebind_receipt(self):
        receipt = self.publish()
        self.target.rename(self.parent / "preserved-original")
        self.target.mkdir(mode=0o700)
        (self.target / "original.log").write_bytes(b"retained fixture payload")
        self.assert_new_admission_refused(receipt)

    def test_actual_parent_rename_replacement_cannot_rebind_receipt(self):
        receipt = self.publish()
        self.parent.rename(self.base / "preserved-parent")
        self.parent.mkdir(mode=0o700)
        self.target.mkdir(mode=0o700)
        (self.target / "original.log").write_bytes(b"retained fixture payload")
        self.assert_new_admission_refused(receipt)

    def test_actual_partial_receipt_write_keeps_original_and_blocks_admission(self):
        original = self.q._exclusive
        def fail_marker(held, fd, name, raw, maximum):
            if name.endswith(".receipt.json"):
                raise OSError("fixture receipt write refused")
            return original(held, fd, name, raw, maximum)
        with patch.object(self.q, "_exclusive", side_effect=fail_marker):
            with self.assertRaisesRegex(OSError, "fixture receipt write refused"):
                self.publish()
        self.assertTrue(any(self.registry.glob("*.original.json")))
        self.assertTrue(any(self.registry.glob("*.snapshot.json")))
        self.assertFalse(any(self.registry.glob("*.receipt.json")))
        self.assertEqual(self.row_path.read_bytes(), self.raw)
        with self.assertRaisesRegex(StorageError, "partial custody"):
            OutputLease(self.parent / "fresh", self.source, catalog=self.catalog,
                        deferred=True, quarantine_receipts=[]).acquire()
        self.assertFalse((self.parent / "fresh").exists())

if __name__ == "__main__":
    unittest.main()
