"""CPU-only lifecycle checks; no model, CUDA, or completion claim."""
import hashlib
import json
import os
from pathlib import Path
import select
import selectors
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import lc0_process_admission_owner as module


class ScriptedWriter(module.AsyncOutputSpool):
    """No worker; exercise only the owner's offer/custody protocol."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.received = []
        self.indeterminate_error = None
        self.prefix = None
        self.last_prefix = None

    def bind_prefix(self, token):
        if self.prefix is not None and self.prefix is not token:
            return False
        self.prefix = token
        return True

    def end_prefix(self, token):
        if self.prefix is token:
            self.last_prefix, self.prefix = token, None
            return True
        return self.prefix is None and self.last_prefix is token

    def offer(self, data, *, prefix_token=None):
        if self.prefix is not prefix_token:
            return "unavailable"
        self.received.append(data)
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response

    def note_offer_indeterminate(self, error):
        self.indeterminate_error = error
        return "marked"


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

    def test_split_notice_preserves_each_original_read_chunk(self):
        owner = self.owner()
        owner._process = mock.Mock(pid=123)
        owner._state = "awaiting_first_notice"
        notice = json.dumps({"schema": module.NOTICE_SCHEMA, "process_id": 123,
                             "process_epoch": owner._epoch, "lock_sha256": owner._lock_sha256,
                             "cuda_ordinal": owner._ordinal, "native_started": False,
                             "control_path": "/proc/123/fd/5/lc0-system-control.sock"}).encode()
        first = b"x" * module.FRAME_LIMIT
        second = b"\n" + notice[:17]
        third = notice[17:] + b"\nremaining diagnostics\n"

        class StubHandshake:
            def __init__(self, *args, **kwargs):
                pass

        with mock.patch.object(module.os, "read", side_effect=[first, second, third]), \
                mock.patch.object(owner, "_observe_child_image"), \
                mock.patch.object(module, "NonblockingParentHandshake", StubHandshake):
            self.assertEqual(owner.advance_stderr(), "awaiting_first_notice")
            self.assertEqual(owner.advance_stderr(), "awaiting_first_notice")
            self.assertEqual(owner.advance_stderr(), "first_authentication")
        self.assertIs(owner._stderr_chunks[0], first)
        self.assertIs(owner._stderr_chunks[1], second)
        self.assertIs(owner._stderr_chunks[2], third)
        self.assertEqual(owner.stderr_bytes, first + second + third)
        owner._process = None

    def test_maximum_notice_body_is_accepted_across_reads(self):
        owner = self.owner()
        owner._process = mock.Mock(pid=123)
        owner._state = "awaiting_first_notice"
        body = json.dumps({"schema": module.NOTICE_SCHEMA, "process_id": 123,
                           "process_epoch": owner._epoch, "lock_sha256": owner._lock_sha256,
                           "cuda_ordinal": owner._ordinal, "native_started": False,
                           "control_path": "/proc/123/fd/5/lc0-system-control.sock"}).encode()
        body += b" " * (module.FRAME_LIMIT - 1 - len(body))

        class StubHandshake:
            def __init__(self, *args, **kwargs):
                pass

        with mock.patch.object(module.os, "read", side_effect=[body[:3000], body[3000:] + b"\n"]), \
                mock.patch.object(owner, "_observe_child_image"), \
                mock.patch.object(module, "NonblockingParentHandshake", StubHandshake):
            self.assertEqual(owner.advance_stderr(), "awaiting_first_notice")
            self.assertEqual(owner.advance_stderr(), "first_authentication")
        self.assertEqual(owner.stderr_bytes, body + b"\n")
        owner._process = None

    def stderr_owner(self):
        owner = self.owner()
        owner._process = mock.Mock(pid=123)
        owner._state = "admitted"
        owner._stderr_chunks = [b"first original chunk", b"second original chunk"]
        owner._stderr_length = sum(map(len, owner._stderr_chunks))
        self.addCleanup(setattr, owner, "_process", None)
        return owner

    def test_stderr_prefix_is_offered_once_in_order_to_same_writer(self):
        owner = self.stderr_owner()
        writer = ScriptedWriter(["backpressure", "accepted", "accepted"])
        first, second = owner._stderr_chunks
        self.assertEqual(owner.handoff_stderr_prefix(writer), "backpressure")
        self.assertEqual(owner.stderr_prefix_handoff_observation()["accepted_chunks"], 0)
        self.assertEqual(owner.handoff_stderr_prefix(writer), "accepted")
        with self.assertRaises(ValueError):
            owner.handoff_stderr_prefix(ScriptedWriter([]))
        self.assertEqual(owner.handoff_stderr_prefix(writer), "accepted")
        self.assertEqual(owner.handoff_stderr_prefix(writer), "complete")
        self.assertIs(writer.received[0], first)
        self.assertIs(writer.received[1], first)
        self.assertIs(writer.received[2], second)
        self.assertEqual(owner.stderr_bytes, first + second)
        self.assertEqual(owner.stderr_prefix_handoff_observation()["accepted_chunks"], 2)
        with self.assertRaises(ValueError):
            owner.handoff_stderr_prefix(ScriptedWriter([]))

    def test_raised_offer_retains_original_chunk_and_never_retries_it(self):
        owner = self.stderr_owner()
        failure = OSError("private offer detail")
        writer = ScriptedWriter([failure, "accepted"])
        self.assertEqual(owner.handoff_stderr_prefix(writer), "indeterminate")
        self.assertIs(owner.original_stderr_handoff_error, failure)
        self.assertIs(writer.indeterminate_error, failure)
        self.assertEqual(owner.stderr_prefix_handoff_observation()["writer_indeterminate_mark"], "marked")
        self.assertIs(owner._stderr_offer.chunk, owner._stderr_chunks[0])
        self.assertEqual(owner.handoff_stderr_prefix(writer), "indeterminate")
        self.assertEqual(len(writer.received), 1)
        self.assertEqual(owner.stderr_prefix_handoff_observation()["accepted_chunks"], 0)

    def test_definitive_first_rejection_allows_another_writer(self):
        owner = self.stderr_owner()
        too_small = ScriptedWriter(["oversized"])
        suitable = ScriptedWriter(["accepted", "accepted"])
        self.assertEqual(owner.handoff_stderr_prefix(too_small), "oversized")
        self.assertEqual(owner.handoff_stderr_prefix(suitable), "accepted")
        self.assertEqual(owner.handoff_stderr_prefix(suitable), "accepted")
        self.assertEqual(owner.handoff_stderr_prefix(suitable), "complete")
        self.assertEqual(suitable.received, owner._stderr_chunks)

    def test_concurrent_handoff_does_not_offer_a_chunk_twice(self):
        owner = self.stderr_owner()
        entered, release = threading.Event(), threading.Event()

        class SlowOffer(ScriptedWriter):
            def offer(self, data, *, prefix_token=None):
                if self.prefix is not prefix_token:
                    return "unavailable"
                self.received.append(data)
                entered.set()
                if not release.wait(3):
                    raise TimeoutError("fixture release timed out")
                return "accepted"

        writer = SlowOffer([])
        result = []
        thread = threading.Thread(target=lambda: result.append(owner.handoff_stderr_prefix(writer)))
        thread.start()
        try:
            self.assertTrue(entered.wait(3))
            self.assertEqual(owner.handoff_stderr_prefix(writer), "busy")
            self.assertEqual(owner.stderr_prefix_handoff_observation(), {"state": "busy"})
            with self.assertRaises(BlockingIOError):
                _ = owner.original_stderr_handoff_error
        finally:
            release.set()
            thread.join(timeout=3)
        self.assertFalse(thread.is_alive())
        self.assertEqual(result, ["accepted"])
        self.assertEqual(len(writer.received), 1)
        self.assertIs(writer.received[0], owner._stderr_chunks[0])

    def test_previously_accepted_attempt_commits_without_second_offer(self):
        owner = self.stderr_owner()
        writer = ScriptedWriter(["accepted"])
        owner._stderr_writer = writer
        self.assertTrue(writer.bind_prefix(owner._stderr_prefix_token))
        attempt = module._StderrOffer(0, owner._stderr_chunks[0])
        attempt.result = "accepted"
        owner._stderr_offer = attempt
        self.assertEqual(owner.handoff_stderr_prefix(writer), "accepted")
        self.assertEqual(writer.received, [owner._stderr_chunks[1]])
        self.assertEqual(owner.handoff_stderr_prefix(writer), "complete")

    def test_real_async_writer_receives_original_prefix_in_order(self):
        owner = self.stderr_owner()
        expected = owner.stderr_bytes
        with tempfile.TemporaryFile() as output:
            writer = module.AsyncOutputSpool(output.fileno(), max_bytes=len(expected),
                                             max_queue_bytes=len(expected), max_queue_items=2)
            try:
                until = time.monotonic() + 3
                result = None
                while result != "complete" and time.monotonic() < until:
                    result = owner.handoff_stderr_prefix(writer)
                    if result == "backpressure":
                        time.sleep(0.01)
                    else:
                        self.assertIn(result, ("accepted", "complete"))
                self.assertEqual(result, "complete")
            finally:
                if writer.observation()["source_terminal"] == "unknown":
                    writer.observe_eof()
                writer.request_finish()
                writer._thread.join(timeout=3)
                if not writer._thread.is_alive() and writer.observation()["writer_state"] == "stopped":
                    writer.release_notification()
            self.assertFalse(writer._thread.is_alive())
            self.assertEqual(writer.observation()["writer_state"], "stopped")
            self.assertEqual(os.pread(output.fileno(), len(expected), 0), expected)
            self.assertEqual(owner.stderr_bytes, expected)

    def test_eof_and_other_offer_wait_for_prefix_commit(self):
        owner = self.stderr_owner()
        owner._stderr_chunks = owner._stderr_chunks[:1]
        entered, release = threading.Event(), threading.Event()

        class HoldAfterAcceptance(module.AsyncOutputSpool):
            def offer(self, data, *, prefix_token=None):
                result = super().offer(data, prefix_token=prefix_token)
                if result == "accepted":
                    entered.set()
                    if not release.wait(3):
                        raise TimeoutError("fixture handoff timed out")
                return result

        with tempfile.TemporaryFile() as output:
            writer = HoldAfterAcceptance(output.fileno(), max_bytes=64,
                                         max_queue_bytes=64, max_queue_items=2)
            result = []
            worker = threading.Thread(target=lambda: result.append(owner.handoff_stderr_prefix(writer)))
            worker.start()
            try:
                self.assertTrue(entered.wait(3))
                with self.assertRaises(BlockingIOError):
                    writer.observe_eof()
                with self.assertRaises(ValueError):
                    writer.request_finish()
                self.assertEqual(writer.offer(b"later"), "unavailable")
                self.assertTrue(writer.observation()["prefix_transfer_active"])
            finally:
                release.set()
                worker.join(timeout=3)
                # Fixture-only cleanup after the known mocked offer returns.
                if not worker.is_alive() and not writer.observation()["offer_accounting"] == "unknown":
                    writer.end_prefix(owner._stderr_prefix_token)
                try:
                    writer.observe_eof()
                except (ValueError, BlockingIOError):
                    pass
                try:
                    writer.request_finish()
                except ValueError:
                    pass
                writer._thread.join(3)
                if not writer._thread.is_alive() and writer.observation()["writer_state"] == "stopped":
                    writer.release_notification()
            self.assertFalse(worker.is_alive())
            self.assertEqual(result, ["accepted"])
            self.assertFalse(writer._thread.is_alive())

    def test_lost_release_result_does_not_claim_closed_writer_was_sealed(self):
        owner = self.stderr_owner()
        owner._stderr_chunks = owner._stderr_chunks[:1]
        released, proceed = threading.Event(), threading.Event()
        failure = RuntimeError("fixture lost prefix release result")

        class LostRelease(module.AsyncOutputSpool):
            def end_prefix(self, token):
                result = super().end_prefix(token)
                if result:
                    released.set()
                    if not proceed.wait(3):
                        raise TimeoutError("fixture release timed out")
                    raise failure
                return result

        with tempfile.TemporaryFile() as output:
            writer = LostRelease(output.fileno(), max_bytes=64,
                                 max_queue_bytes=64, max_queue_items=2)
            self.assertEqual(owner.handoff_stderr_prefix(writer), "accepted")
            result = []
            worker = threading.Thread(target=lambda: result.append(owner.handoff_stderr_prefix(writer)))
            worker.start()
            try:
                self.assertTrue(released.wait(3))
                writer.observe_eof()
                self.assertTrue(writer.request_finish())
                writer._thread.join(3)
                self.assertFalse(writer._thread.is_alive())
            finally:
                proceed.set()
                worker.join(3)
                if not writer._thread.is_alive() and writer.observation()["writer_state"] == "stopped":
                    writer.release_notification()
            self.assertFalse(worker.is_alive())
            self.assertEqual(result, ["indeterminate"])
            self.assertIs(owner.original_stderr_handoff_error, failure)
            self.assertEqual(owner.stderr_prefix_handoff_observation()["writer_indeterminate_mark"], "too_late")
            self.assertEqual(owner.handoff_stderr_prefix(writer), "indeterminate")


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
