"""CPU-only direct-parent protocol fixtures, never native/GPU evidence."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
import rovent_v3_parent as parent


FAKE_CHILD = r'''
import json, os, socket, sys, time
from pathlib import Path

def option(prefix):
    return next(item[len(prefix):] for item in sys.argv[1:] if item.startswith(prefix))

epoch = int(option("--process-epoch="))
lock = option("--test-lock=")
variant = option("--test-variant=")
pid = os.getpid()
if variant in ("admission_before", "admission_tail", "admission_unobserved"):
    # The test parent creates this marker only after its real /proc image
    # observation; do not assume an image-observation time interval.
    deadline = time.monotonic() + 5
    while variant != "admission_unobserved" and not Path("image-observed").exists():
        if time.monotonic() >= deadline:
            sys.exit(4)
        time.sleep(0.005)
    evidence = {"schema":"rz-rovent-resource-pre-native-admission/1",
        "process_id":pid, "process_generation":epoch,
        "reason":"resource_plan_provenance", "native_scope_started":False}
    print(json.dumps({"schema":"rz-rovent-native-system-resource-run-failure/3",
        "resource_state_poisoned":False, "control_panic_payload_retained":False,
        "automatic_cleanup":False, "training_executed":False, "product_finish":None,
        "session_evidence":"scope_unavailable", "pre_native_admission":evidence}),
        file=sys.stderr, flush=True)
    if variant == "admission_tail":
        sys.stderr.write('{"schema":')
        sys.stderr.flush()
    sys.exit(1)
directory = Path("fake-control")
directory.mkdir()
fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
route = f"/proc/{pid}/fd/{fd}/custody.sock"
listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
listener.bind(route)
listener.listen()
print(json.dumps({"schema":"rz-rovent-direct-custody-startup/1", "process_id":pid,
    "process_epoch":epoch, "lock_sha256":lock, "cuda_ordinal":0,
    "control_path":route, "physical_completion":"unknown_until_original_owner_confirms"}),
    file=sys.stderr, flush=True)
channel, _ = listener.accept()
request_bytes = bytearray()
while not request_bytes.endswith(b"\n"):
    chunk = channel.recv(4096)
    if not chunk:
        sys.exit(3)
    request_bytes.extend(chunk)
request = json.loads(request_bytes)
assert request == {"schema":"rz-rovent-parent-admission/1", "child_pid":pid,
    "process_epoch":epoch, "lock_sha256":lock, "cuda_ordinal":0}
ack = dict(request, schema="rz-rovent-parent-admission-ack/1",
    scope="parent-child-connection-not-gpu-fence")
channel.sendall(json.dumps(ack).encode() + b"\n")
if variant == "factory_before":
    evidence = {"schema":"rz-rovent-factory-pre-native-observation/1", "process_id":pid,
        "process_generation":epoch, "input_lock_sha256":lock, "resource_scope_started":True,
        "native_entry":"not_entered_by_original_factory", "physical_completion":"not_inferred"}
    print(json.dumps({"schema":"rz-rovent-native-system-resource-run-failure/3",
        "resource_state_poisoned":False, "control_panic_payload_retained":False,
        "automatic_cleanup":False, "training_executed":False, "product_finish":None,
        "session_evidence":"before_native", "factory_pre_native":evidence}),
        file=sys.stderr, flush=True)
    sys.exit(1)
commands = []
for command in sys.stdin:
    command = command.strip()
    commands.append(command)
    if command == "uci":
        print("uciok", flush=True)
    elif command == "isready":
        print("readyok", flush=True)
    elif command == "go nodes 64":
        assert commands[-2] == "position startpos"
        print("bestmove e2e4", flush=True)
    elif command == "quit":
        assert commands[:4] == ["uci", "isready", "position startpos", "go nodes 64"]
        if variant == "unknown":
            break
        finish = {"schema":"rz-rovent-product-finish-observation/1", "process_id":pid,
            "process_epoch":epoch, "input_lock_sha256":lock,
            "state":"confirmed_by_original_product", "device_fence":False}
        if variant == "failure":
            value = {"schema":"rz-rovent-native-system-resource-run-failure/3",
                "error":"fixture failure", "product_finish":finish}
        else:
            value = {"schema":"rz-rovent-native-system-resource-run/3", "process_id":pid,
                "process_generation":epoch, "product_finish":finish}
        print(json.dumps(value), file=sys.stderr, flush=True)
        if variant == "late_quarantine":
            print(json.dumps({"schema":"rz-rovent-direct-custody-quarantined/1",
                              "process_id":pid, "physical_completion":"unknown"}),
                  file=sys.stderr, flush=True)
        break
channel.close()
listener.close()
os.close(fd)
sys.exit(2 if variant in ("failure", "unknown") else 0)
'''


class NoticeTests(unittest.TestCase):
    def test_typed_before_native_evidence_requires_original_identity_and_scope(self):
        pid, epoch, lock = 123, 19, hashlib.sha256(os.urandom(32)).hexdigest()
        common = {"schema": "rz-rovent-native-system-resource-run-failure/3",
                  "resource_state_poisoned": False, "control_panic_payload_retained": False,
                  "automatic_cleanup": False, "training_executed": False, "product_finish": None}
        admission = {"schema": "rz-rovent-resource-pre-native-admission/1", "process_id": pid,
                     "process_generation": epoch, "reason": "resource_plan_provenance",
                     "native_scope_started": False}
        record = {**common, "session_evidence": "scope_unavailable", "pre_native_admission": admission}
        self.assertEqual(parent._pre_native_evidence(record, pid, epoch, lock), "resource_admission_before_scope")
        self.assertIsNone(parent._pre_native_evidence(record, pid + 1, epoch, lock))
        self.assertIsNone(parent._pre_native_evidence({**record, "resource_state_poisoned": True}, pid, epoch, lock))
        factory = {"schema": "rz-rovent-factory-pre-native-observation/1", "process_id": pid,
                   "process_generation": epoch, "input_lock_sha256": lock, "resource_scope_started": True,
                   "native_entry": "not_entered_by_original_factory", "physical_completion": "not_inferred"}
        record = {**common, "session_evidence": "before_native", "factory_pre_native": factory}
        self.assertEqual(parent._pre_native_evidence(record, pid, epoch, lock), "factory_before_native")
        self.assertIsNone(parent._pre_native_evidence(record, pid, epoch, "other-lock"))
        self.assertIsNone(parent._pre_native_evidence({**record, "pre_native_admission": admission}, pid, epoch, lock))

    def test_notice_requires_exact_original_identity_and_route(self):
        pid, epoch, lock = 123, 19, hashlib.sha256(os.urandom(32)).hexdigest()
        notice = {"schema": parent.NOTICE_SCHEMA, "process_id": pid, "process_epoch": epoch,
                  "lock_sha256": lock, "cuda_ordinal": 0,
                  "control_path": f"/proc/{pid}/fd/7/custody.sock",
                  "physical_completion": "unknown_until_original_owner_confirms"}
        self.assertEqual(parent._notice_route(notice, pid, epoch, lock, 0), notice["control_path"])
        for field, replacement in (("process_id", pid + 1), ("process_epoch", True),
                                   ("cuda_ordinal", 1), ("lock_sha256", "different"),
                                   ("control_path", f"/proc/{pid}/fd/7/other.sock")):
            changed = dict(notice, **{field: replacement})
            with self.subTest(field=field), self.assertRaises(parent.PreflightError):
                parent._notice_route(changed, pid, epoch, lock, 0)

    def test_failed_acquire_recognizes_only_same_durable_retained_owner(self):
        import managed_outputs
        token = hashlib.sha256(os.urandom(32)).hexdigest()
        record = {"state": "retained", "status": "acquire_failed_before_dispatch",
                  "pid": os.getpid(), "lifetime": token, "identity": [7, 11]}
        lease = mock.Mock(record=record, output=Path("/fixture/output"),
                          catalog=Path("/fixture/catalog"), key="fixture")
        lease.guard.stream = None
        with mock.patch.object(managed_outputs, "lifetime", return_value=token), \
             mock.patch.object(managed_outputs, "identity", return_value=[7, 11]), \
             mock.patch.object(managed_outputs, "_read", return_value=record):
            self.assertTrue(parent._failed_acquire_retained(lease))
            lease.guard.stream = object()
            self.assertFalse(parent._failed_acquire_retained(lease))
            lease.guard.stream = None
            with mock.patch.object(managed_outputs, "_read", return_value={**record, "state": "active"}):
                self.assertFalse(parent._failed_acquire_retained(lease))

    def test_same_lease_finish_recovery_distinguishes_intent_state(self):
        lease = mock.Mock()
        lease.live_finish_observation.return_value = {"phase": "active_before_intent"}
        lease.live_handoff_observation.return_value = {"state": "suspended"}
        self.assertTrue(parent._recover_same_live_finish(lease, "original_status"))
        lease.resume_live.assert_called_once()
        lease.finish.assert_called_once_with(status="original_status", tree_gone=True)
        lease.reset_mock()
        lease.live_finish_observation.return_value = {"phase": "terminal_pending"}
        self.assertTrue(parent._recover_same_live_finish(lease, "original_status"))
        lease.recover_live_terminal.assert_called_once_with()
        lease.finish.assert_not_called()
        lease.reset_mock()
        lease.live_finish_observation.return_value = {"phase": "unknown"}
        self.assertFalse(parent._recover_same_live_finish(lease, "original_status"))
        lease.finish.assert_not_called()
        lease.recover_live_terminal.assert_not_called()


@unittest.skipUnless(sys.platform == "linux", "requires Linux procfs and Unix peer credentials")
class ParentCpuFixtureTests(unittest.TestCase):
    def command(self, root, variant):
        output = root / "output"
        output.mkdir()
        fixture = root / "fake_child.py"
        fixture.write_text(FAKE_CHILD, encoding="utf-8")
        binary = Path(sys.executable).resolve(strict=True)
        image = binary.read_bytes()
        binary_sha = hashlib.sha256(image).hexdigest()
        lock = hashlib.sha256(os.urandom(64)).hexdigest()
        epoch = 19
        command = [sys.executable, str(SCRIPTS / "rovent_v3_parent.py"),
                       "--binary", str(binary), "--binary-bytes", str(len(image)),
                       "--binary-sha256", binary_sha, "--cwd", str(root),
                       "--output-dir", str(output), "--epoch", str(epoch),
                       "--lock-sha256", lock, "--cuda-ordinal", "0",
                       "--seconds", "8", "--cleanup-seconds", "2", "--", str(fixture),
                       "--resource-system-runtime-v3", "--custody-control",
                       f"--process-epoch={epoch}", "--input-lock=inputs/example",
                       "--resource-plan=inputs/plan", "--resource-plan-sha256=fixture",
                       "--resource-plan-bytes=1", f"--test-lock={lock}", f"--test-variant={variant}"]
        return command, output

    def exercise(self, variant):
        with tempfile.TemporaryDirectory() as temporary:
            command, output = self.command(Path(temporary), variant)
            completed = subprocess.run(command, capture_output=True, timeout=15, check=False)
            records = [json.loads(line) for line in completed.stdout.splitlines()]
            self.assertTrue(records, completed.stderr.decode(errors="replace"))
            result = records[-1]
            self.assertTrue(result["parent_authenticated"])
            self.assertTrue(result["child_image_observed"])
            self.assertEqual(result["uci"]["bestmove"], "e2e4")
            self.assertEqual(result["physical_completion"], "session_reported_by_original_product")
            self.assertEqual(result["gpu_device_fence"], "not_observed")
            self.assertEqual(result["move_legality"], "not_evaluated")
            self.assertTrue((output / "rovent-v3-stderr.log").is_file())
            return completed.returncode, result

    def test_original_parent_handshake_and_uci_sequence(self):
        code, result = self.exercise("success")
        self.assertEqual(code, 0)
        self.assertEqual(result["status"], "uci_exchange_observed")

    def test_failed_product_can_still_report_original_session_finish(self):
        code, result = self.exercise("failure")
        self.assertNotEqual(code, 0)
        self.assertEqual(result["status"], "failed_or_incomplete")
        self.assertEqual(result["error_type"], "child_resource_run_failed")

    def test_contained_typed_pre_native_failures_finish_without_fabricating_session(self):
        import multiprocessing
        for variant, expected in (("admission_before", "resource_admission_before_scope"),
                                  ("factory_before", "factory_before_native")):
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                command, output = self.command(root, variant)
                args = argparse_namespace_for_cancel(Path(command[3]), root, output)
                # Reuse the exact fixture CLI values while running the parent
                # in a bounded child so a regression cannot hang this suite.
                args.binary = Path(command[3])
                args.binary_bytes = int(command[5])
                args.binary_sha256 = command[7]
                args.epoch = int(command[13])
                args.lock_sha256 = command[15]
                args.seconds = 8
                args.cleanup_seconds = 2
                args.native_args = command[command.index("--") + 1:]
                context = multiprocessing.get_context("fork")
                result_queue = context.Queue()
                process = context.Process(target=_run_fixture_with_containment, args=(args, result_queue))
                process.start()
                process.join(15)
                try:
                    self.assertFalse(process.is_alive(), "before-native fixture retained an owner")
                    result, outcome, report = result_queue.get(timeout=2)
                    self.assertEqual(result, 1)
                    self.assertTrue(outcome["before_native"])
                    self.assertFalse(outcome["session_reported"])
                    self.assertEqual(outcome["pre_native_evidence"], expected)
                    self.assertEqual(outcome["stderr_tail"], "none")
                    self.assertEqual(outcome["output_drain"], "eof_observed")
                    self.assertEqual(report["physical_completion"], "native_not_entered_by_original_product")
                    self.assertEqual(report["gpu_device_fence"], "not_observed")
                    self.assertEqual(report["error_type"], "child_resource_run_failed")
                finally:
                    if process.is_alive():
                        process.terminate()
                        process.join(5)
                    result_queue.close()

    def test_pre_native_requires_observed_image_and_complete_stderr(self):
        import multiprocessing
        for variant in ("admission_tail", "admission_unobserved"):
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                command, output = self.command(root, variant)
                args = argparse_namespace_for_cancel(Path(command[3]), root, output)
                args.binary_bytes, args.binary_sha256 = int(command[5]), command[7]
                args.epoch, args.lock_sha256 = int(command[13]), command[15]
                args.seconds, args.cleanup_seconds = 8, 2
                args.native_args = command[command.index("--") + 1:]
                context = multiprocessing.get_context("fork")
                result_queue = context.Queue()
                process = context.Process(target=_run_fixture_with_containment, args=(args, result_queue))
                process.start()
                try:
                    kind, _, report = result_queue.get(timeout=12)
                    self.assertEqual(kind, "quarantined")
                    self.assertEqual(report["physical_completion"], "unknown")
                    self.assertTrue(process.is_alive(), "unproven owner must stay retained")
                finally:
                    if process.is_alive():
                        process.kill()  # CPU-only fixture owner, never a GPU process.
                    process.join(5)
                    result_queue.close()

    def test_unobserved_session_keeps_original_owner_after_cpu_fixture_exits(self):
        # The child here is a CPU-only Python fixture. Forced test cleanup is
        # confined to this fixture and is not a GPU completion procedure.
        for variant in ("unknown", "late_quarantine"):
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as temporary:
                command, _ = self.command(Path(temporary), variant)
                process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                try:
                    import selectors
                    selector = selectors.DefaultSelector()
                    selector.register(process.stdout, selectors.EVENT_READ)
                    self.assertTrue(selector.select(15), "fixture did not report retained owner")
                    record = json.loads(process.stdout.readline())
                    self.assertEqual(record["status"], "quarantined_original_owner_retained")
                    self.assertEqual(record["physical_completion"], "unknown")
                    self.assertIsNone(process.poll(), "owner must remain alive for unknown scope")
                    selector.close()
                finally:
                    if process.poll() is None:
                        process.kill()
                    process.communicate(timeout=5)

    def test_cancelled_before_spawn_never_creates_child(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "output"
            output.mkdir()
            binary = Path(sys.executable).resolve(strict=True)
            args = argparse_namespace_for_cancel(binary, root, output)

            def register_handler(sig, handler):
                if callable(handler):
                    handler(sig, None)
                return signal.SIG_DFL

            with mock.patch.object(parent, "_open_pinned_binary", side_effect=lambda *unused: (os.open(binary, os.O_RDONLY), os.stat(binary))), \
                 mock.patch.object(parent.signal, "signal", side_effect=register_handler), \
                 mock.patch.object(parent.subprocess, "Popen") as spawn:
                with self.assertRaises(InterruptedError):
                    parent.run(args)
                spawn.assert_not_called()


def argparse_namespace_for_cancel(binary, root, output):
    import argparse
    return argparse.Namespace(
        binary=binary, binary_bytes=1, binary_sha256="unused", cwd=root, output_dir=output,
        epoch=1, lock_sha256="unused", cuda_ordinal=0, seconds=3, cleanup_seconds=1,
        numeric_observation=False, go_nodes=64,
        native_args=["--resource-system-runtime-v3", "--custody-control", "--process-epoch=1",
                     "--input-lock=inputs/lock", "--resource-plan=inputs/plan",
                     "--resource-plan-sha256=unused", "--resource-plan-bytes=1"],
    )


def _run_fixture_with_containment(args, result_queue):
    class FixtureContainment:
        admission_outcome = "not_attempted"

        def admit_waiting_child(self, _pid, fd):
            self.admission_outcome = "granted"
            os.write(fd, b"R")
            os.close(fd)

        def workload_empty(self):
            return True

    outcome = {}
    reports = []
    def capture_report(value):
        reports.append(value)
        if value.get("status") == "quarantined_original_owner_retained":
            result_queue.put(("quarantined", {}, value))

    parent._report = capture_report
    real_stat = os.stat

    def observe_image(path, *positional, **keywords):
        if ("--test-variant=admission_unobserved" in args.native_args
                and isinstance(path, str) and path.startswith("/proc/") and path.endswith("/exe")):
            raise FileNotFoundError("fixture image was not observed")
        result = real_stat(path, *positional, **keywords)
        if isinstance(path, str) and path.startswith("/proc/") and path.endswith("/exe"):
            (Path(args.cwd) / "image-observed").touch()
        return result

    try:
        with mock.patch.object(parent.os, "stat", side_effect=observe_image):
            result = parent.run(args, containment=FixtureContainment(), outcome=outcome)
        result_queue.put((result, outcome, reports[-1]))
    except BaseException as error:
        result_queue.put(("error", {"error_type": type(error).__name__}, {}))


if __name__ == "__main__":
    unittest.main()
