"""CPU-only lifecycle checks; no model, CUDA, or completion claim."""
import hashlib
import os
from pathlib import Path
import select
import selectors
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import lc0_process_admission_owner as module


@unittest.skipUnless(sys.platform == "linux", "Linux direct-child owner")
class OwnerFixtureTests(unittest.TestCase):
    def owner(self, **overrides):
        values = dict(binary=Path("/fixture/unused"), binary_bytes=1,
                      binary_sha256=hashlib.sha256(b"binary").hexdigest(),
                      native_args=("fixture",), epoch=3, ordinal=0,
                      lock_sha256=hashlib.sha256(b"lock").hexdigest(),
                      run_seconds=5, cleanup_seconds=2,
                      deadline=time.monotonic() + 5, max_stderr_bytes=8192,
                      max_control_attempts=1)
        values.update(overrides)
        return module.Lc0FirstAdmissionOwner(**values)

    def test_before_spawn_can_release_only_original_pin(self):
        owner = self.owner()
        with tempfile.TemporaryFile() as source:
            fd = os.dup(source.fileno())
            owner._deadline = time.monotonic() - 1
            with mock.patch.object(module, "_open_pinned_binary", return_value=(fd, os.fstat(fd))):
                self.assertEqual(owner.start(), "failed_before_spawn")
            os.fstat(fd)
            self.assertEqual(owner.release_before_spawn(), "pre_spawn_released")
            with self.assertRaises(OSError):
                os.fstat(fd)

    def test_unknown_spawn_retains_pin_and_refuses_pre_spawn_release(self):
        owner = self.owner()
        error = OSError("fixture spawn failure")
        with tempfile.TemporaryFile() as source:
            fd = os.dup(source.fileno())
            with mock.patch.object(module, "_open_pinned_binary", return_value=(fd, os.fstat(fd))), \
                    mock.patch.object(module.subprocess, "Popen", side_effect=error):
                self.assertEqual(owner.start(), "spawn_result_unknown")
            self.assertIs(owner.original_failure, error)
            os.fstat(fd)
            with self.assertRaises(RuntimeError):
                owner.release_before_spawn()
            # This fixture's patched Popen definitely did not start a child.
            os.close(fd)
            owner._binary_fd = None
            owner._spawn_attempted = False

    def test_partial_first_admission_object_remains_with_original_owner(self):
        owner = self.owner()
        owner._state = "first_authentication"
        handshake = mock.Mock()
        handshake.advance.return_value = "authenticated"
        owner._handshake = handshake
        marker = object()
        error = OSError("fixture transfer failure")

        class PartialAdmission:
            def __init__(self, supplied):
                self.handshake = supplied
                self.socket = marker
                raise error

        with mock.patch.object(owner, "_observe_child_image"), \
                mock.patch.object(module, "OriginalAdmission", PartialAdmission):
            self.assertEqual(owner.advance_first_admission(0), "first_admission_unknown")
        self.assertIs(owner.original_failure, error)
        self.assertIs(owner._pending_admission.handshake, handshake)
        self.assertIs(owner._pending_admission.socket, marker)

    def test_failed_control_image_observation_keeps_slot_and_capacity_is_local(self):
        owner = self.owner()
        owner._state = "admitted"
        owner._original_admission = object()
        error = OSError("fixture pidfd error")
        with mock.patch.object(owner, "_observe_child_image", side_effect=error):
            with self.assertRaises(OSError):
                owner.new_control_handshake(time.monotonic() + 1)
        self.assertIs(owner._controls[0].failure, error)
        self.assertEqual(owner.state, "admitted")
        with self.assertRaises(RuntimeError):
            owner.new_control_handshake(time.monotonic() + 1)
        self.assertTrue(owner.control_capacity_reached)
        self.assertIsNotNone(owner._original_admission)

    def test_partial_control_constructor_remains_in_reserved_slot(self):
        owner = self.owner()
        owner._state = "admitted"
        owner._original_admission = object()
        owner._process = mock.Mock(pid=123)
        marker = object()
        error = OSError("fixture control failure")

        class PartialHandshake:
            def __init__(self, *args, **kwargs):
                self.socket = marker
                raise error

        with mock.patch.object(owner, "_observe_child_image"), \
                mock.patch.object(module, "NonblockingParentHandshake", PartialHandshake):
            with self.assertRaises(OSError):
                owner.new_control_handshake(time.monotonic() + 1)
        self.assertIs(owner._controls[0].failure, error)
        self.assertIs(owner._controls[0].value.socket, marker)
        owner._process = None

    def test_partial_observation_constructor_remains_in_reserved_slot(self):
        owner = self.owner()
        owner._state = "admitted"
        owner._original_admission = object()
        marker = object()
        error = OSError("fixture observation failure")

        class FakeHandshake:
            pass

        class PartialObservation:
            def __init__(self, *args, **kwargs):
                self.socket = marker
                raise error

        control = FakeHandshake()
        slot = module._AttemptSlot()
        slot.value = control
        owner._controls.append(slot)
        with mock.patch.object(owner, "_observe_child_image"), \
                mock.patch.object(module, "NonblockingParentHandshake", FakeHandshake), \
                mock.patch.object(module, "NonblockingSessionObservation", PartialObservation):
            with self.assertRaises(OSError):
                owner.new_observation(control, sequence=1, deadline=time.monotonic() + 1)
        self.assertIs(owner._observations[0].failure, error)
        self.assertIs(owner._observations[0].value.socket, marker)


_CHILD = r'''
import json, os, signal, socket, struct, sys
signal.alarm(10)
directory, epoch, digest, ordinal = sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4])
directory_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
route = f"/proc/{os.getpid()}/fd/{directory_fd}/lc0-system-control.sock"
listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
listener.bind(route)
listener.listen(2)
listener.settimeout(5)
print(json.dumps({"schema":"rz-lc0-parent-endpoint/1", "process_id":os.getpid(),
                  "process_epoch":epoch, "lock_sha256":digest, "cuda_ordinal":ordinal,
                  "control_path":route, "native_started":False}), file=sys.stderr, flush=True)
connection, _ = listener.accept()
connection.settimeout(5)
peer = struct.unpack("3i", connection.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i")))
if peer[0] != os.getppid() or peer[1] != os.geteuid():
    raise RuntimeError("wrong parent")
data = b""
while not data.endswith(b"\n"):
    part = connection.recv(1024)
    if not part: raise EOFError("admission closed")
    data += part
request = json.loads(data)
expected = {"schema":"rz-rovent-parent-admission/1", "child_pid":os.getpid(),
            "process_epoch":epoch, "lock_sha256":digest, "cuda_ordinal":ordinal}
if request != expected: raise ValueError("wrong admission")
ack = {**expected, "schema":"rz-rovent-parent-admission-ack/1", "scope":"parent-child-connection-not-gpu-fence"}
connection.sendall(json.dumps(ack, separators=(",", ":")).encode() + b"\n")
sys.stdin.buffer.readline()
connection.close()
listener.close()
os.close(directory_fd)
'''


@unittest.skipUnless(sys.platform == "linux" and hasattr(os, "pidfd_open")
                     and hasattr(os, "P_PIDFD") and hasattr(os, "WNOWAIT"),
                     "Linux pidfd/direct-child support required")
class RealUnixChildTests(unittest.TestCase):
    def test_first_admission_comes_from_the_owned_direct_child(self):
        binary = Path(sys.executable).resolve(strict=True)
        digest = hashlib.sha256()
        with binary.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        lock = hashlib.sha256(b"independent-test-lock").hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            owner = module.Lc0FirstAdmissionOwner(
                binary=binary, binary_bytes=binary.stat().st_size, binary_sha256=digest.hexdigest(),
                native_args=("-u", "-c", _CHILD, directory, "3", lock, "0"), epoch=3,
                ordinal=0, lock_sha256=lock, run_seconds=5, cleanup_seconds=2,
                deadline=time.monotonic() + 6, max_stderr_bytes=32768, max_control_attempts=1)
            try:
                self.assertEqual(owner.start(), "awaiting_first_notice")
                until = time.monotonic() + 6
                while owner.state == "awaiting_first_notice" and time.monotonic() < until:
                    _, _, stderr_fd = owner.pipe_filenos()
                    if select.select([stderr_fd], [], [], 0.1)[0]:
                        owner.advance_stderr()
                self.assertEqual(owner.state, "first_authentication")
                while owner.state == "first_authentication" and time.monotonic() < until:
                    fd, interest, retry = owner.first_admission_poll()
                    if not interest:
                        time.sleep(0.01)
                        owner.advance_first_admission(0)
                        continue
                    reads = [fd] if interest & selectors.EVENT_READ else []
                    writes = [fd] if interest & selectors.EVENT_WRITE else []
                    readable, writable, _ = select.select(reads, writes, [], 0.1)
                    events = ((selectors.EVENT_READ if readable else 0)
                              | (selectors.EVENT_WRITE if writable else 0))
                    owner.advance_first_admission(events)
                self.assertEqual(owner.state, "admitted")
                self.assertEqual(owner._original_admission.binding[0], owner.child_pid)
                self.assertIn(b"rz-lc0-parent-endpoint/1", owner.stderr_bytes)
            finally:
                # This is the test's own known CPU-only child, never an engine.
                process = owner._process
                if process is not None:
                    try:
                        process.communicate(input=b"\n", timeout=2)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.communicate(timeout=2)
                    if owner._original_admission is not None:
                        owner._original_admission._socket.close()
                        owner._original_admission._socket = None
                    for pipe in (process.stdin, process.stdout, process.stderr):
                        if pipe is not None and not pipe.closed:
                            pipe.close()
                    owner._process = None
                for field in ("_pidfd", "_binary_fd"):
                    fd = getattr(owner, field)
                    if fd is not None:
                        os.close(fd)
                        setattr(owner, field, None)
                owner._spawn_attempted = False


if __name__ == "__main__":
    unittest.main()
