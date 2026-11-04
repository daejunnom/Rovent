import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import rovent_cgroup_owner as subject
from managed_process import OwnedProcessGroup


@unittest.skipUnless(sys.platform == "linux", "Linux cgroup descriptors")
class CgroupOwnerTests(unittest.TestCase):
    def make_owner(self, root):
        scope = Path(root) / "scope"
        manager = scope / "manager"
        workload = scope / "workload"
        entry = workload / "entry"
        manager.mkdir(parents=True)
        entry.mkdir(parents=True)
        (workload / "cgroup.events").write_text("populated 0\nfrozen 0\n")
        return subject.CgroupOwner(scope, manager, workload, entry)

    def test_pregrant_rejection_closes_barrier(self):
        with tempfile.TemporaryDirectory() as root:
            owner = self.make_owner(root)
            read_fd, write_fd = os.pipe()
            try:
                with self.assertRaises(subject.CgroupOwnerError):
                    owner.admit_waiting_child(-1, write_fd)
                self.assertEqual(owner.admission_outcome, "rejected_before_grant")
                self.assertEqual(os.read(read_fd, 1), b"")
            finally:
                os.close(read_fd)
                for fd in owner._fds.values():
                    os.close(fd)

    def test_borrowed_pregrant_rejection_leaves_original_barrier_with_caller(self):
        with tempfile.TemporaryDirectory() as root:
            owner = self.make_owner(root)
            read_fd, write_fd = os.pipe()
            try:
                with self.assertRaises(subject.CgroupOwnerError):
                    owner.admit_waiting_child_borrowed(-1, write_fd)
                self.assertEqual(owner.admission_outcome, "rejected_before_grant")
                os.fstat(write_fd)
                os.write(write_fd, b"x")
                self.assertEqual(os.read(read_fd, 1), b"x")
            finally:
                os.close(write_fd)
                os.close(read_fd)
                for fd in owner._fds.values():
                    os.close(fd)

    def test_close_admission_prevents_new_barrier_before_empty_observation(self):
        with tempfile.TemporaryDirectory() as root:
            owner = self.make_owner(root)
            read_fd, write_fd = os.pipe()
            try:
                group = OwnedProcessGroup()
                owner.reserve_output_group(group)
                self.assertTrue(owner.close_admission_and_observe_empty(group))
                with self.assertRaises(subject.CgroupOwnerError):
                    owner.admit_waiting_child(os.getpid(), write_fd)
                self.assertEqual(os.read(read_fd, 1), b"")
            finally:
                os.close(read_fd)
                for fd in owner._fds.values():
                    os.close(fd)

    def test_output_group_reservation_preserves_original_on_competing_request(self):
        with tempfile.TemporaryDirectory() as root:
            owner = self.make_owner(root)
            first, second = OwnedProcessGroup(), OwnedProcessGroup()
            try:
                owner.reserve_output_group(first)
                with self.assertRaises(subject.CgroupOwnerError):
                    owner.reserve_output_group(second)
                self.assertIs(owner._output_group, first)
                owner.release_unstarted_output_group(first)
                owner.reserve_output_group(second)
                self.assertIs(owner._output_group, second)
            finally:
                for fd in owner._fds.values():
                    os.close(fd)

    def test_descriptor_retry_never_closes_reused_unknown_fd(self):
        # Private descriptors only; filesystem cgroup removal is outside this
        # test. Simulate the explicit post-removal phase of the same owner.
        with tempfile.TemporaryDirectory() as root:
            owner = self.make_owner(root)
            owner._closed = True
            target_name = next(iter(owner._fds))
            target_fd = owner._fds[target_name]
            original_close = os.close
            replacement = []
            def close_then_report_unknown(fd):
                if fd == target_fd and not replacement:
                    original_close(fd)
                    replacement.append(os.open("/dev/null", os.O_RDONLY))
                    if replacement[0] != target_fd:
                        self.skipTest("kernel did not reuse the closed descriptor number")
                    raise OSError("close-result-unknown")
                original_close(fd)
            try:
                with mock.patch.object(subject.os, "close", side_effect=close_then_report_unknown):
                    with self.assertRaises(OSError):
                        owner.close()
                    with self.assertRaises(subject.CgroupOwnerError):
                        owner.retry()
                    os.fstat(replacement[0])
                    observation = owner.descriptor_close_observation()
                    self.assertEqual(observation["descriptors"][target_name], "unknown")
                    self.assertTrue(all(value in {"closed", "unknown"}
                                        for value in observation["descriptors"].values()))
            finally:
                if replacement:
                    original_close(replacement[0])
                for name, fd in tuple(owner._fds.items()):
                    if owner._fd_close_states[name] == "unattempted":
                        original_close(fd)

    def test_workload_observation_refuses_path_replacement(self):
        with tempfile.TemporaryDirectory() as root:
            owner = self.make_owner(root)
            try:
                self.assertTrue(owner.workload_empty())
                owner.workload.rename(owner.workload.with_name("old-workload"))
                owner.workload.mkdir()
                with self.assertRaises((subject.CgroupOwnerError, FileNotFoundError)):
                    owner.workload_empty()
            finally:
                for fd in owner._fds.values():
                    os.close(fd)

    def test_close_can_retry_after_entry_was_removed(self):
        with tempfile.TemporaryDirectory() as root:
            owner = self.make_owner(root)
            # Kernel control writes and self-migration are outside this unit
            # test; it exercises only retrying a partially removed leaf.
            owner._controls_disabled = True
            owner._manager_moved = True
            original_rmdir = Path.rmdir
            attempts = 0

            def rmdir(path):
                nonlocal attempts
                if path == owner.workload:
                    attempts += 1
                    if attempts == 1:
                        raise OSError("injected workload removal failure")
                    (owner.workload / "cgroup.events").unlink()
                return original_rmdir(path)

            with mock.patch.object(Path, "rmdir", rmdir), mock.patch.object(subject, "_write"):
                with self.assertRaises(OSError):
                    owner.close()
                self.assertTrue(owner._entry_removed)
                owner.close()
            self.assertTrue(owner._closed)
            self.assertEqual(attempts, 2)

    def test_empty_remains_observable_after_workload_removal_then_cleanup_error(self):
        with tempfile.TemporaryDirectory() as root:
            owner = self.make_owner(root)
            original_rmdir = Path.rmdir

            def rmdir(path):
                if path == owner.workload:
                    (owner.workload / "cgroup.events").unlink()
                return original_rmdir(path)

            with mock.patch.object(Path, "rmdir", rmdir):
                # Missing pseudo-file simulates a later controller-read error.
                with self.assertRaises(FileNotFoundError):
                    owner.close()
                self.assertTrue(owner._workload_removed)
                self.assertTrue(owner.workload_empty())
                owner._controls_disabled = True
                owner._manager_moved = True
                owner.close()
            self.assertTrue(owner._closed)

    def test_empty_reconciles_interruption_after_kernel_removed_workload(self):
        with tempfile.TemporaryDirectory() as root:
            owner = self.make_owner(root)
            original_rmdir = Path.rmdir

            def rmdir(path):
                if path == owner.workload:
                    (owner.workload / "cgroup.events").unlink()
                    original_rmdir(path)
                    raise InterruptedError("interrupted after removal")
                return original_rmdir(path)

            with mock.patch.object(Path, "rmdir", rmdir):
                with self.assertRaises(InterruptedError):
                    owner.close()
                self.assertFalse(owner._workload_removed)
                self.assertTrue(owner.workload_empty())
                self.assertTrue(owner._workload_removed)
                owner._controls_disabled = True
                owner._manager_moved = True
                owner.close()
            self.assertTrue(owner._closed)


if __name__ == "__main__":
    unittest.main()
