"""Small CPU-only checks for original input snapshots and child FD handoff."""
import hashlib
import json
import os
from pathlib import Path
import socket
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
    import virtual_system_numeric_run as numeric_runner
    from managed_process import ProcessNotSpawned, run_group
    from virtual_system_numeric_run import (capture_cleanup_partition, capture_process_status, observe_unresolved_owned,
                                            output_lease_tree_observation,
                                            require_capture_window, snapshot_file)
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

    def test_cleanup_partition_does_not_extend_registered_deadline(self):
        cleanup, group_reserve, child_deadline = capture_cleanup_partition(1, 20_000_000_000)
        self.assertEqual((cleanup, group_reserve, child_deadline),
                         (1_000_000_000, 500_000_000, 19_500_000_000))
        self.assertEqual(child_deadline + group_reserve, 20_000_000_000)

    def test_late_exit_observation_preserves_exit_without_claiming_deadline(self):
        custody = {"status": "session_released", "native_authorization": "sent", "error": None}
        self.assertEqual(capture_process_status(0, True, None, {"exit_observation_timing": "before_deadline"}, custody),
                         "capture_process_completed")
        self.assertEqual(capture_process_status(0, True, None, {"exit_observation_timing": "at_or_after_deadline"}, custody),
                         "capture_process_completed_deadline_unknown")
        self.assertEqual(capture_process_status(0, True, None, {}, custody),
                         "capture_process_completed_deadline_unknown")
        self.assertEqual(capture_process_status(17, True, None, {"exit_observation_timing": "at_or_after_deadline"}, custody),
                         "capture_process_failed_or_incomplete")
        self.assertEqual(capture_process_status(0, True, None, {"exit_observation_timing": "before_deadline"}, {}),
                         "capture_process_completed_custody_unknown")

    def test_capture_window_rejects_noninteger_or_out_of_range_values(self):
        for config in ({"deadline_seconds": True, "cleanup_seconds": 1},
                       {"deadline_seconds": 1.0, "cleanup_seconds": 1},
                       {"deadline_seconds": 1, "cleanup_seconds": 0},
                       {"deadline_seconds": 1, "cleanup_seconds": 31},
                       {"deadline_seconds": 0, "cleanup_seconds": 1}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                require_capture_window(config, 100_000_000_000, now_ns=0)

    def test_unresolved_reobservation_keeps_original_owner_and_registered_deadline(self):
        original = mock.Mock()
        original.reobserve_and_drain.return_value = (True, 17, None)
        poll = mock.Mock()
        record = observe_unresolved_owned(original, poll, 12.0, clock=lambda: 11.0)
        original.reobserve_and_drain.assert_called_once_with(absolute_deadline=12.0, on_tick=poll)
        self.assertEqual((record["tree_gone"], record["reaped_exit_code"]), (True, 17))
        self.assertEqual(record["scope"], "original_group_and_channel_until_registered_deadline")

    def test_unresolved_reobservation_never_extends_expired_window(self):
        original = mock.Mock()
        record = observe_unresolved_owned(original, lambda: None, 12.0, clock=lambda: 12.0)
        original.reobserve_and_drain.assert_not_called()
        self.assertFalse(record["attempted"])
        self.assertEqual(record["tree_gone"], "unknown")
        self.assertEqual(record["error"], "registered_deadline_elapsed_before_reobservation")

    def test_group_completion_does_not_prove_escaped_tree_completion(self):
        self.assertEqual(output_lease_tree_observation(True, False), "unknown")
        self.assertIs(output_lease_tree_observation(False, False), True)
        self.assertIs(output_lease_tree_observation(True, True), True)

    def test_unresolved_reobservation_keeps_failure_separate_from_original_result(self):
        original = mock.Mock()
        original.reobserve_and_drain.return_value = (
            False, None, {"kind": "deadline", "error_type": "TimeoutError", "message": "elapsed"})
        record = observe_unresolved_owned(original, lambda: None, 12.0, clock=lambda: 11.0)
        self.assertIs(record["tree_gone"], False)
        self.assertIsNone(record["reaped_exit_code"])
        self.assertEqual(record["error"]["kind"], "deadline")

    def test_custody_transport_reobserves_same_child_without_device_fence(self):
        channel = CustodyTransport(allow_native=lambda: True)
        self.addCleanup(channel.close)
        channel.child.send(b"PENDING")
        channel.poll()
        self.assertEqual(channel.state["requests"], 0)
        channel.child.send(b"NATIVE_READY")
        channel.poll()
        self.assertEqual(channel.child.recv(64), b"ALLOW_NATIVE")
        channel.child.send(b"UNKNOWN")
        channel.poll()
        self.assertEqual(channel.child.recv(64), b"REOBSERVE")
        channel.child.send(b"PENDING")
        channel.poll()
        self.assertEqual(channel.child.recv(64), b"REOBSERVE")
        channel.child.send(b"SESSION_RELEASED")
        channel.poll()
        self.assertEqual(channel.state["events"], ["PENDING", "NATIVE_READY", "UNKNOWN", "PENDING", "SESSION_RELEASED"])
        self.assertEqual(channel.state["scope"], "per_spawn_private_channel_peer_pid_unverified")
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
        self.assertEqual(channel.state["scope"], "per_spawn_private_channel_peer_pid_unverified")

        recovered = CustodyTransport()
        self.addCleanup(recovered.close)
        recovered.child.send(b"PENDING")
        recovered.child.send(b"UNKNOWN")
        recovered.poll()
        self.assertEqual(recovered.child.recv(64), b"REOBSERVE")
        recovered.child.send(b"OWNER_RECOVERED")
        recovered.poll()
        self.assertEqual(recovered.state["status"], "owner_recovered")
        self.assertNotEqual(recovered.state["status"], "session_released")

    def test_native_ack_is_denied_when_supervisor_window_expired(self):
        channel = CustodyTransport(allow_native=lambda: False)
        self.addCleanup(channel.close)
        channel.child.send(b"PENDING")
        channel.child.send(b"NATIVE_READY")
        channel.poll()
        self.assertEqual(channel.state["native_authorization"], "not_sent_deadline_or_policy")
        channel.child.setblocking(False)
        with self.assertRaises(BlockingIOError):
            channel.child.recv(64)
        channel.child.send(b"NO_NATIVE_ENTRY")
        channel.poll()
        self.assertEqual(channel.state["status"], "no_native_entry")

    def test_pre_native_rejection_does_not_request_output_admission(self):
        acquire_output = mock.Mock(return_value=True)
        channel = CustodyTransport(allow_native=acquire_output)
        self.addCleanup(channel.close)
        channel.child.send(b"PENDING")
        channel.child.send(b"NO_NATIVE_ENTRY")
        channel.poll()
        acquire_output.assert_not_called()
        self.assertEqual(channel.state["status"], "no_native_entry")
        self.assertEqual(channel.state["native_authorization"], "not_requested")

    def test_supervised_pre_native_rejection_creates_no_output_lease(self):
        binary = self.root / "capture"
        binary_bytes = b"test executable bytes"
        binary.write_bytes(binary_bytes)
        plan = self.root / "plan.json"
        plan_bytes = json.dumps({"bindings": {
            "source_components": {"native": "fixture"},
            "model_semantic_sha256": "model", "encoding_sha256": "encoding",
        }}).encode()
        plan.write_bytes(plan_bytes)
        config = self.root / "config.json"
        config_bytes = json.dumps({
            "provider": "cuda-system", "output_root": str(self.root / "output"),
            "case_plan": str(plan), "case_plan_sha256": hashlib.sha256(plan_bytes).hexdigest(),
            "deadline_seconds": 1, "cleanup_seconds": 1,
        }).encode()
        config.write_bytes(config_bytes)
        receipt = self.root / "receipt.json"

        def rejected_child(*args, **kwargs):
            peer = socket.socket(fileno=os.dup(kwargs["pass_fds"][-1]))
            try:
                kwargs["on_spawn"]()
                peer.send(b"PENDING")
                peer.send(b"NO_NATIVE_ENTRY")
                kwargs["on_tick"]()
                kwargs["admission_check"]() if "admission_check" in kwargs else args[4]()
                kwargs["deadline_observation"]["exit_observation_timing"] = "before_deadline"
                return 17, True, None
            finally:
                peer.close()

        def getter(_fd, label, *_args):
            if label == "--source-pins":
                return {"native": "fixture"}
            return {"model_semantic_sha256": "model", "encoding_sha256": "encoding"}

        argv = ["--binary", str(binary), "--binary-sha256", hashlib.sha256(binary_bytes).hexdigest(),
                "--binary-bytes", str(len(binary_bytes)), "--config", str(config),
                "--config-sha256", hashlib.sha256(config_bytes).hexdigest(),
                "--receipt", str(receipt), "--timeout", "30"]
        with mock.patch.object(numeric_runner, "metadata_getter", side_effect=getter), \
                mock.patch.object(numeric_runner, "run_group", side_effect=rejected_child), \
                mock.patch.object(numeric_runner, "OutputLease") as output_lease:
            self.assertEqual(numeric_runner.main(argv), 1)
        output_lease.assert_not_called()
        record = json.loads(receipt.read_text())
        self.assertEqual(record["producer_exit_code"], 17)
        self.assertEqual(record["custody_transport"]["status"], "no_native_entry")
        self.assertFalse(record["output_lease"]["acquired"])
        self.assertFalse((self.root / "output").exists())

    def test_failed_output_admission_never_grants_native_entry(self):
        acquire_output = mock.Mock(side_effect=OSError("original admission failure"))
        channel = CustodyTransport(allow_native=acquire_output)
        self.addCleanup(channel.close)
        channel.child.send(b"PENDING")
        channel.child.send(b"NATIVE_READY")
        channel.poll()
        acquire_output.assert_called_once_with()
        self.assertEqual(channel.state["error"], "OSError")
        self.assertEqual(channel.state["native_authorization"], "not_sent_deadline_or_policy")
        channel.child.setblocking(False)
        with self.assertRaises(BlockingIOError):
            channel.child.recv(64)

    def test_stop_closes_native_admission_but_keeps_recovery_channel(self):
        channel = CustodyTransport(allow_native=lambda: True)
        self.addCleanup(channel.close)
        channel.child.send(b"PENDING")
        channel.poll()
        channel.close_native_admission()
        channel.child.send(b"NATIVE_READY")
        channel.poll()
        self.assertEqual(channel.state["native_authorization"], "not_sent_deadline_or_policy")
        channel.child.setblocking(False)
        with self.assertRaises(BlockingIOError):
            channel.child.recv(64)
        channel.child.send(b"UNKNOWN")
        channel.poll()
        self.assertEqual(channel.child.recv(64), b"REOBSERVE")

    def test_pending_reobservation_is_spread_across_cleanup_window(self):
        now = [100]
        channel = CustodyTransport(reobserve_interval_ns=25, clock_ns=lambda: now[0])
        self.addCleanup(channel.close)
        channel.child.send(b"UNKNOWN")
        channel.poll()
        self.assertEqual(channel.child.recv(64), b"REOBSERVE")
        channel.child.send(b"PENDING")
        channel.poll()
        channel.child.setblocking(False)
        with self.assertRaises(BlockingIOError):
            channel.child.recv(64)
        now[0] += 25
        channel.poll()
        self.assertEqual(channel.child.recv(64), b"REOBSERVE")
        self.assertEqual(channel.state["requests"], 2)

    def test_terminal_packet_is_drained_before_control_write_to_closed_peer(self):
        channel = CustodyTransport(allow_native=lambda: True)
        self.addCleanup(channel.close)
        channel.child.send(b"PENDING")
        channel.child.send(b"NATIVE_READY")
        channel.poll()
        self.assertEqual(channel.child.recv(64), b"ALLOW_NATIVE")
        channel.child.send(b"UNKNOWN")
        channel.child.send(b"SESSION_RELEASED")
        channel.child.close()
        channel.poll()
        self.assertEqual(channel.state["status"], "session_released")
        self.assertIsNone(channel.state["error"])
        self.assertEqual(channel.state["requests"], 0)

    def test_private_channel_child_handshake_uses_lightweight_supervisor_tick(self):
        channel = CustodyTransport(allow_native=lambda: True)
        self.addCleanup(channel.close)
        environment = os.environ.copy()
        environment["TEST_CUSTODY_FD"] = str(channel.child_fd())
        program = (
            "import os,socket\n"
            "s=socket.socket(fileno=int(os.environ['TEST_CUSTODY_FD']))\n"
            "s.settimeout(3)\n"
            "s.send(b'PENDING'); s.send(b'NATIVE_READY')\n"
            "assert s.recv(64)==b'ALLOW_NATIVE'\n"
            "s.send(b'SESSION_RELEASED')\n"
        )
        observation = {}
        deadline = time.monotonic() + 10
        code, gone, failure = run_group(
            [str(Path(sys.executable).resolve()), "-c", program], self.root, environment,
            10, lambda: None, pass_fds=(channel.child_fd(),),
            on_spawn=channel.child_started, on_tick=channel.poll,
            verify_before_spawn=True, absolute_deadline=deadline,
            deadline_observation=observation,
        )
        channel.poll()
        self.assertEqual((code, gone, failure), (0, True, None))
        self.assertEqual(channel.state["events"], ["PENDING", "NATIVE_READY", "SESSION_RELEASED"])
        self.assertEqual(channel.state["native_authorization"], "sent")
        self.assertEqual(observation["exit_observation_timing"], "before_deadline")

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
