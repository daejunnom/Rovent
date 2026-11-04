"""Small CPU-only checks for original input snapshots and child FD handoff."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if sys.platform == "linux":
    import fcntl
    from managed_process import ProcessNotSpawned, run_group
    from virtual_system_numeric_run import capture_process_status, require_capture_window, snapshot_file
    from virtual_custody_transport import CustodyTransport


@unittest.skipUnless(sys.platform == "linux", "Linux memfd/process-group ownership only")
class SystemNumericFdHandoffTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="rovezero-fd-handoff-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()

    def snapshot(self, data=b"original snapshot bytes\n"):
        source = self.root / "source.bin"
        source.write_bytes(data)
        owner_fd, image_fd, observed = snapshot_file(
            source, hashlib.sha256(data).hexdigest(), len(data),
            time.monotonic() + 10, content=True, exact_size=len(data),
        )
        self.addCleanup(os.close, owner_fd)
        self.addCleanup(os.close, image_fd)
        self.assertEqual(observed, data)
        return source, image_fd, data

    def test_capture_window_includes_preparation_and_cleanup(self):
        config = {"deadline_seconds": 10, "cleanup_seconds": 2}
        result = require_capture_window(config, 20_000_000_000, now_ns=8_000_000_000)
        self.assertEqual(result["required_ns"], 12_000_000_000)
        with self.assertRaises(TimeoutError):
            require_capture_window(config, 20_000_000_000, now_ns=8_000_000_001)
        self.assertTrue(result["child_startup_recheck_required"])

    def test_late_exit_observation_preserves_exit_without_claiming_deadline(self):
        self.assertEqual(capture_process_status(0, True, None, {"exit_observation_timing": "before_deadline"}),
                         "capture_process_completed")
        self.assertEqual(capture_process_status(0, True, None, {"exit_observation_timing": "at_or_after_deadline"}),
                         "capture_process_completed_deadline_unknown")
        self.assertEqual(capture_process_status(0, True, None, {}),
                         "capture_process_completed_deadline_unknown")
        self.assertEqual(capture_process_status(17, True, None, {"exit_observation_timing": "at_or_after_deadline"}),
                         "capture_process_failed_or_incomplete")

    def test_capture_window_rejects_noninteger_or_out_of_range_values(self):
        for config in ({"deadline_seconds": True, "cleanup_seconds": 1},
                       {"deadline_seconds": 1.0, "cleanup_seconds": 1},
                       {"deadline_seconds": 1, "cleanup_seconds": 0},
                       {"deadline_seconds": 1, "cleanup_seconds": 31},
                       {"deadline_seconds": 0, "cleanup_seconds": 1}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                require_capture_window(config, 100_000_000_000, now_ns=0)

    def test_custody_transport_reobserves_same_child_without_device_fence(self):
        channel = CustodyTransport()
        self.addCleanup(channel.close)
        channel.child.send(b"PENDING")
        channel.poll()
        self.assertEqual(channel.state["requests"], 0)
        channel.child.send(b"UNKNOWN")
        channel.poll()
        self.assertEqual(channel.child.recv(64), b"REOBSERVE")
        channel.child.send(b"PENDING")
        channel.poll()
        self.assertEqual(channel.child.recv(64), b"REOBSERVE")
        channel.child.send(b"SESSION_RELEASED")
        channel.poll()
        self.assertEqual(channel.state["events"], ["PENDING", "UNKNOWN", "PENDING", "SESSION_RELEASED"])
        self.assertEqual(channel.state["scope"], "same_child_process_only")
        self.assertNotIn("gpu_physical_completion", channel.state)

    def test_custody_transport_rejects_oversize_and_unrecognized_messages(self):
        for packet in (b"UNKNOWN-UNTRUSTED", b"X" * 65):
            with self.subTest(packet=packet):
                channel = CustodyTransport()
                try:
                    channel.child.send(packet)
                    channel.poll()
                    self.assertEqual(channel.state["error"], "invalid_or_excess_custody_packet")
                    self.assertEqual(channel.state["requests"], 0)
                    self.assertEqual(channel.state["status"], "unobserved")
                finally:
                    channel.close()

    def test_custody_transport_rejects_unannounced_release(self):
        channel = CustodyTransport()
        self.addCleanup(channel.close)
        channel.child.send(b"SESSION_RELEASED")
        channel.poll()
        self.assertEqual(channel.state["error"], "invalid_custody_sequence")
        self.assertEqual(channel.state["status"], "unobserved")

    def test_custody_transport_keeps_no_native_and_recovery_distinct(self):
        channel = CustodyTransport()
        self.addCleanup(channel.close)
        channel.child.send(b"NO_NATIVE_ENTRY")
        channel.poll()
        self.assertEqual(channel.state["status"], "no_native_entry")
        self.assertEqual(channel.state["requests"], 0)
        self.assertEqual(channel.state["scope"], "same_child_process_only")

    def test_sealed_snapshot_keeps_actual_bytes_when_source_changes(self):
        source, image_fd, data = self.snapshot()
        source.write_bytes(b"replaced source content")
        os.lseek(image_fd, 0, os.SEEK_SET)
        self.assertEqual(os.read(image_fd, len(data) + 1), data)
        with self.assertRaises(PermissionError):
            os.write(image_fd, b"alter")
        with self.assertRaises(PermissionError):
            os.ftruncate(image_fd, 0)

    def test_wrong_digest_size_and_expired_window_are_rejected(self):
        data = b"bounded input"
        source = self.root / "input"
        source.write_bytes(data)
        digest = hashlib.sha256(data).hexdigest()
        with self.assertRaises(ValueError):
            snapshot_file(source, hashlib.sha256(b"different").hexdigest(), len(data),
                          time.monotonic() + 10)
        with self.assertRaises(ValueError):
            snapshot_file(source, digest, len(data), time.monotonic() + 10,
                          exact_size=len(data) + 1)
        with self.assertRaises(TimeoutError):
            snapshot_file(source, digest, len(data), time.monotonic() - 1)

    def test_child_receives_only_explicit_fd_and_returns_original_stdout(self):
        _, image_fd, data = self.snapshot()
        unrelated = fcntl.fcntl(image_fd, fcntl.F_DUPFD, 100)
        self.addCleanup(os.close, unrelated)
        os.set_inheritable(unrelated, True)
        output = os.memfd_create("rovezero-fd-test-stdout", os.MFD_CLOEXEC)
        self.addCleanup(os.close, output)
        program = (
            "import json,os,sys\n"
            "fd,other=map(int,sys.argv[1:])\n"
            "data=os.read(fd,4096).decode('ascii')\n"
            "try: os.fstat(other); extra=True\n"
            "except OSError: extra=False\n"
            "print(json.dumps({'data':data,'unrelated_inherited':extra}))\n"
        )
        executable = str(Path(sys.executable).resolve())
        events = []
        code, gone, failure = run_group(
            [executable, "-c", program, str(image_fd), str(unrelated)],
            self.root, os.environ.copy(), 10, lambda: None,
            pass_fds=(image_fd,), stdout=output, on_spawn=lambda: events.append("spawned"),
            verify_before_spawn=True,
        )
        self.assertEqual((code, gone, failure), (0, True, None))
        self.assertEqual(events, ["spawned"])
        os.lseek(output, 0, os.SEEK_SET)
        result = json.loads(os.read(output, 4096))
        self.assertEqual(result, {"data": data.decode("ascii"), "unrelated_inherited": False})

    def test_failed_opt_in_preflight_does_not_call_popen_or_spawn_callback(self):
        events = []
        missing = str(self.root / "nonexistent-executable")
        with mock.patch("managed_process.subprocess.Popen") as popen:
            with self.assertRaises(ProcessNotSpawned) as raised:
                run_group([missing], self.root, os.environ.copy(), 10, lambda: None,
                          on_spawn=lambda: events.append("spawned"), verify_before_spawn=True)
        self.assertIsInstance(raised.exception.__cause__, FileNotFoundError)
        popen.assert_not_called()
        self.assertEqual(events, [])

    def test_failing_spawn_callback_drains_and_reaps_its_own_child(self):
        children = []
        original = subprocess.Popen

        def record_child(*args, **kwargs):
            child = original(*args, **kwargs)
            children.append(child)
            return child

        def fail_after_spawn():
            raise ValueError("callback-failed")

        executable = str(Path(sys.executable).resolve())
        with mock.patch("managed_process.subprocess.Popen", side_effect=record_child):
            code, gone, failure = run_group(
                [executable, "-c", "import time; time.sleep(30)"],
                self.root, os.environ.copy(), 10, lambda: None,
                on_spawn=fail_after_spawn, verify_before_spawn=True,
            )
        self.assertEqual((code, gone, failure), (125, True, "callback-failed"))
        self.assertEqual(len(children), 1)
        self.assertIsNotNone(children[0].poll(), "original child must be reaped")

    def test_default_admission_system_exit_still_propagates(self):
        # The default historical branch propagates BaseException; use a fake
        # child so this compatibility check cannot leave a real process behind.
        child = mock.Mock(pid=123456)
        with mock.patch("managed_process.subprocess.Popen", return_value=child), \
             mock.patch("managed_process.os.waitid", return_value=None), \
             mock.patch("managed_process.os.killpg") as killpg:
            with self.assertRaises(SystemExit) as raised:
                run_group([str(Path(sys.executable).resolve())], self.root,
                          os.environ.copy(), 10,
                          lambda: (_ for _ in ()).throw(SystemExit("unchanged")))
        self.assertEqual(str(raised.exception), "unchanged")
        killpg.assert_not_called()

    def test_spawn_callback_system_exit_still_drains_owned_child(self):
        children = []
        original = subprocess.Popen

        def record_child(*args, **kwargs):
            child = original(*args, **kwargs)
            children.append(child)
            return child

        def stop_after_spawn():
            raise SystemExit("callback-stop")

        with mock.patch("managed_process.subprocess.Popen", side_effect=record_child):
            code, gone, failure = run_group(
                [str(Path(sys.executable).resolve()), "-c", "import time; time.sleep(30)"],
                self.root, os.environ.copy(), 10, lambda: None,
                on_spawn=stop_after_spawn, verify_before_spawn=True,
            )
        self.assertEqual((code, gone, failure), (125, True, "SystemExit: callback-stop"))
        self.assertEqual(len(children), 1)
        self.assertIsNotNone(children[0].poll())


if __name__ == "__main__":
    unittest.main()
