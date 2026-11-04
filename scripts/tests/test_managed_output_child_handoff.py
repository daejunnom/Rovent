"""Private lease/direct-child protocol; cgroup observations are test fixtures."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from managed_output_child_handoff import ManagedOutputChildHandoff
from managed_outputs import OutputLease
from managed_process import OwnedProcessGroup, ProcessNotSpawned, run_group
from managed_storage import StorageError
from rovent_cgroup_owner import CgroupOwner
from storage_guard import CatalogGuard

_PRESERVED_UNKNOWN = []


@unittest.skipUnless(sys.platform == "linux", "Linux peer credentials and inherited FDs")
class ManagedOutputChildHandoffTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="rz-child-output-")
        self.root = Path(self.tmp.name).resolve()
        self.source = self.root / "source"
        self.source.mkdir()
        self.catalog = self.root / "catalog"
        catalog_patch = patch("managed_outputs.canonical_catalog", return_value=self.catalog)
        catalog_patch.start()
        self.addCleanup(catalog_patch.stop)
        maintenance_patch = patch("rovezero_usage_snapshot.claim_total_usage_maintenance", return_value=False)
        maintenance_patch.start()
        self.addCleanup(maintenance_patch.stop)
        self.lease = OutputLease(self.root / "output", self.source, catalog=self.catalog,
                                 reserve=1024 * 1024, total_limit=2 * 1024 * 1024).acquire()
        self.group = None
        self.handoff = None
        self.finished = False
        self.addCleanup(self.cleanup_original_owners)
        self.key = self.lease.prepare_live_handoff()
        self.group = OwnedProcessGroup()
        # This fixture does not establish real kernel cgroup containment.
        self.scope = object.__new__(CgroupOwner)
        self.scope.workload = Path("/sys/fs/cgroup/rz-test-scope")
        self.scope.admission_outcome = "granted"
        self.scope._check_owner = lambda: None
        self.scope.confirms_granted_child = lambda _pid: True
        scope_state = {"group": None, "attempted": False}
        def reserve(group):
            self.assertIsNone(scope_state["group"])
            scope_state["group"] = group
        def release(group):
            self.assertIs(scope_state["group"], group)
            scope_state["group"] = None
        def mark_dispatch(group):
            self.assertIs(scope_state["group"], group)
            scope_state["attempted"] = True
        self.scope.reserve_output_group = reserve
        self.scope.owns_output_group = lambda group: scope_state["group"] is group
        self.scope.mark_output_dispatch_attempted = mark_dispatch
        self.scope.output_dispatch_attempted = lambda _group: scope_state["attempted"]
        self.scope.release_unstarted_output_group = release
        self.scope.close_admission_and_observe_empty = lambda _group: True
        self.scope.release_finished_output_group = release
        membership_patch = patch("managed_output_child_handoff._membership", return_value="/rz-test-scope/entry")
        membership_patch.start()
        self.addCleanup(membership_patch.stop)
        self.handoff = ManagedOutputChildHandoff(self.key, self.group, self.scope)

    def cleanup_original_owners(self):
        if self.group is not None and (
                (not self.group._completed and (self.group._child is not None
                 or (self.handoff is not None and self.handoff._dispatch_attempted)))
                or (self.handoff is not None and self.handoff._dispatch_attempted
                    and not self.handoff._scope_bound)):
            # Preserve the original temporary owner if physical drain is unknown.
            self.tmp._finalizer.detach()
            _PRESERVED_UNKNOWN.append((self.tmp, self.lease, self.group, self.handoff))
            return
        if not self.finished:
            if self.lease._live_suspended:
                self.lease.resume_live()
            self.lease.finish(status="test-cleanup", tree_gone=True)
            self.finished = True
        if self.handoff is not None:
            self.handoff.revoke()
        self.lease.guard.release()
        self.tmp.cleanup()

    def run_child(self, source, *extra, on_spawn=None, on_tick=None):
        child_socket, output_fd = self.handoff.child_fds
        with tempfile.TemporaryFile(mode="w+b", dir=self.root) as output:
            result = run_group(
                [str(Path(sys.executable).resolve()), "-c", source,
                 str(child_socket), str(output_fd), *extra],
                self.root, os.environ.copy(), 10, lambda: None,
                pass_fds=(child_socket, output_fd), stdout=output,
                owned_group=self.group, output_handoff=self.handoff, on_spawn=on_spawn,
                on_tick=on_tick,
            )
            output.seek(0)
            payload = output.read()
        if result[1]:
            if self.lease._live_suspended:
                self.lease.resume_live()
            self.lease.finish(status="test-complete", tree_gone=True)
            self.finished = True
        return result, payload

    def test_original_direct_child_checks_same_live_fd_twice(self):
        source = '''
import json, os, socket, stat, struct, sys
s = socket.socket(fileno=int(sys.argv[1]))
s.setsockopt(socket.SOL_SOCKET, socket.SO_PASSCRED, 1)
info = os.fstat(int(sys.argv[2]))
assert stat.S_ISDIR(info.st_mode)
for seq in (1, 2):
    s.send(json.dumps({"schema": "rz-managed-output-lease/1", "sequence": seq}).encode())
    data, ancillary, flags, _ = s.recvmsg(4096, socket.CMSG_SPACE(struct.calcsize("3i")))
    assert flags == 0
    creds = [struct.unpack("3i", value) for level, kind, value in ancillary
             if level == socket.SOL_SOCKET and kind == socket.SCM_CREDENTIALS]
    assert len(creds) == 1 and creds[0][0] == os.getppid() and creds[0][1] == os.geteuid()
    reply = json.loads(data)
    assert reply["schema"] == "rz-managed-output-lease/1"
    assert reply["sequence"] == seq
    assert (reply["device"], reply["inode"]) == (info.st_dev, info.st_ino)
print(json.dumps({"owner_pid": reply["owner_pid"], "checks": 2}))
'''
        result, payload = self.run_child(source)
        self.assertEqual(result, (0, True, None))
        self.assertEqual(json.loads(payload), {"owner_pid": os.getpid(), "checks": 2})
        self.assertEqual(self.handoff.scope_release_observation(), "released")

    def test_grandchild_cannot_claim_direct_child_lease(self):
        source = '''
import json, os, socket, sys, time
s = socket.socket(fileno=int(sys.argv[1]))
s.send(json.dumps({"schema": "rz-managed-output-lease/1", "sequence": 1}).encode())
s.recv(4096)
child = os.fork()
if child == 0:
    s.send(json.dumps({"schema": "rz-managed-output-lease/1", "sequence": 2}).encode())
    os._exit(0)
os.waitpid(child, 0)
time.sleep(2)
'''
        result, _ = self.run_child(source)
        self.assertTrue(result[1])
        self.assertNotEqual(result[0], 0)
        self.assertIn("original child", str(result[2]))

    def test_suspended_lease_rechecks_temporarily_and_rejects_early_finish(self):
        self.lease.suspend_live()
        rejected = []
        def premature_finish():
            with self.assertRaises(StorageError):
                self.lease.finish(status="not-done", tree_gone=True)
            rejected.append(True)
        source = '''
import json, socket, sys
s = socket.socket(fileno=int(sys.argv[1]))
s.setsockopt(socket.SOL_SOCKET, socket.SO_PASSCRED, 1)
s.send(json.dumps({"schema": "rz-managed-output-lease/1", "sequence": 1}).encode())
data = json.loads(s.recv(4096))
assert data["sequence"] == 1
'''
        result, _ = self.run_child(source, on_spawn=premature_finish)
        self.assertEqual(result, (0, True, None))
        self.assertEqual(rejected, [True])

    def test_preflight_rejection_keeps_original_lease_finishable(self):
        with self.assertRaises(ProcessNotSpawned):
            run_group([str(self.root / "no-executable")], self.root, os.environ.copy(),
                      10, lambda: None, pass_fds=self.handoff.child_fds,
                      owned_group=self.group, output_handoff=self.handoff,
                      verify_before_spawn=True)
        self.assertFalse(self.handoff._dispatch_attempted)
        self.lease.finish(status="test-not-started", tree_gone=True)
        self.finished = True

    def test_confirmed_grant_can_finish_when_child_requests_no_output(self):
        result, _ = self.run_child("pass")
        self.assertEqual(result, (0, True, None))
        self.assertTrue(self.handoff._scope_bound)

    def test_original_grant_can_be_reobserved_after_spawn_callback_error(self):
        def failed_after_grant():
            raise RuntimeError("after-original-grant")
        result, _ = self.run_child("import time; time.sleep(2)", on_spawn=failed_after_grant)
        self.assertTrue(result[1])
        self.assertNotEqual(result[0], 0)
        self.assertIn("after-original-grant", str(result[2]))
        self.assertTrue(self.handoff._scope_bound)

    def test_suspended_guard_contention_keeps_same_pending_request(self):
        self.lease.suspend_live()
        other = CatalogGuard(self.catalog / "catalog.lock")
        self.addCleanup(other.release)
        ticks = []
        def after_spawn():
            other.acquire()
        def release_after_contention():
            if self.handoff._pending_request == 1:
                ticks.append(True)
                if len(ticks) == 3:
                    other.release()
        source = '''
import json, socket, sys
s = socket.socket(fileno=int(sys.argv[1]))
s.setsockopt(socket.SOL_SOCKET, socket.SO_PASSCRED, 1)
s.send(json.dumps({"schema": "rz-managed-output-lease/1", "sequence": 1}).encode())
assert json.loads(s.recv(4096))["sequence"] == 1
'''
        result, _ = self.run_child(source, on_spawn=after_spawn, on_tick=release_after_contention)
        self.assertEqual(result, (0, True, None))
        self.assertGreaterEqual(len(ticks), 3)

    def test_duplicate_sequence_is_not_a_new_lease_grant(self):
        source = '''
import json, socket, sys, time
s = socket.socket(fileno=int(sys.argv[1]))
request = json.dumps({"schema": "rz-managed-output-lease/1", "sequence": 1}).encode()
s.send(request)
s.recv(4096)
s.send(request)
time.sleep(2)
'''
        result, _ = self.run_child(source)
        self.assertTrue(result[1])
        self.assertNotEqual(result[0], 0)
        self.assertIn("sequence", str(result[2]))

    def test_truncated_rights_are_rejected_without_retaining_received_fds(self):
        before = []
        def count_null_descriptors():
            null = os.stat("/dev/null")
            count = 0
            for name in os.listdir("/proc/self/fd"):
                try:
                    info = os.fstat(int(name))
                except OSError:
                    continue
                if (info.st_dev, info.st_ino, info.st_rdev) == (null.st_dev, null.st_ino, null.st_rdev):
                    count += 1
            return count
        source = '''
import array, json, os, socket, sys, time
s = socket.socket(fileno=int(sys.argv[1]))
s.send(json.dumps({"schema": "rz-managed-output-lease/1", "sequence": 1}).encode())
s.recv(4096)
fds = array.array("i", [os.open("/dev/null", os.O_RDONLY) for _ in range(32)])
packet = json.dumps({"schema": "rz-managed-output-lease/1", "sequence": 2}).encode()
s.sendmsg([packet], [(socket.SOL_SOCKET, socket.SCM_RIGHTS, fds)])
time.sleep(2)
'''
        result, _ = self.run_child(source, on_spawn=lambda: before.append(count_null_descriptors()))
        self.assertTrue(result[1])
        self.assertNotEqual(result[0], 0)
        self.assertIn("invalid managed output", str(result[2]))
        self.assertEqual(count_null_descriptors(), before[0])

    def test_oversized_packet_is_rejected(self):
        source = '''
import json, socket, sys, time
s = socket.socket(fileno=int(sys.argv[1]))
s.send(json.dumps({"schema": "rz-managed-output-lease/1", "sequence": 1}).encode())
s.recv(4096)
s.send(b"x" * 8192)
time.sleep(2)
'''
        result, _ = self.run_child(source)
        self.assertTrue(result[1])
        self.assertNotEqual(result[0], 0)
        self.assertIn("invalid managed output", str(result[2]))

    def test_empty_datagram_is_not_mistaken_for_eof(self):
        source = '''
import json, socket, sys, time
s = socket.socket(fileno=int(sys.argv[1]))
s.send(json.dumps({"schema": "rz-managed-output-lease/1", "sequence": 1}).encode())
s.recv(4096)
s.send(b"")
time.sleep(2)
'''
        result, _ = self.run_child(source)
        self.assertTrue(result[1])
        self.assertNotEqual(result[0], 0)
        self.assertIn("invalid managed output", str(result[2]))


if __name__ == "__main__":
    unittest.main()
