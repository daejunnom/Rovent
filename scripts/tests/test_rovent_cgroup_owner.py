import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import rovent_cgroup_owner as subject


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


if __name__ == "__main__":
    unittest.main()
