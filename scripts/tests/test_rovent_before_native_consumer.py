"""Original pre-native owner input leases; CPU/mock evidence only."""
from __future__ import annotations

import os
import hashlib
import io
import json
from pathlib import Path
import signal
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock
from contextlib import redirect_stdout

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))

if sys.platform == "linux":
    import rovent_before_native_owner as owner
    from storage_consumer_registry import ConsumerRegistry, bind_catalog
    from storage_guard import CatalogGuard


@unittest.skipUnless(sys.platform == "linux", "pre-native owner requires Linux")
class BeforeNativeConsumerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / "source"
        self.templates = self.root / "templates"
        self.assets = self.root / "assets"
        self.catalog = self.root / "catalog"
        for path in (self.source, self.templates, self.assets, self.catalog):
            path.mkdir()
        self.binary = self.root / "test-elf"
        self.binary.write_bytes(b"fixture")
        self.output = self.root / "output"
        self.registry = ConsumerRegistry.initialize(self.root / "registry", scope="fixture")
        with CatalogGuard(self.catalog / "catalog.lock") as guard:
            bind_catalog(self.catalog, self.registry, guard)
        self.argv = ["owner", "--repo", str(self.source), "--output", str(self.output),
                     "--template-root", str(self.templates), "--template", "lock.json",
                     "--asset-root", str(self.assets), "--test-binary", str(self.binary),
                     "--process-epoch", "1", "--seconds", "60",
                     "--consumer-registry-root", str(self.registry.root),
                     "--consumer-registry-id", self.registry.registry_id,
                     "--consumer-input-catalog", str(self.catalog)]

    def _run_main(self, callback):
        with mock.patch.object(sys, "argv", self.argv), \
                mock.patch.dict(os.environ, {}, clear=True), \
                mock.patch.object(owner, "original_owner_supervisor_capability",
                                  return_value=owner.OriginalOwnerSupervisorCapability.CONNECTED), \
                mock.patch.object(owner, "run", side_effect=callback):
            return owner.main()

    def test_unconnected_supervisor_is_reported_before_any_input_or_dispatch(self):
        output = io.StringIO()
        with mock.patch.object(sys, "argv", self.argv), \
                mock.patch.dict(os.environ, {}, clear=True), \
                mock.patch.object(owner, "NumericInputRegistration") as registration, \
                mock.patch.object(owner, "open_directory") as open_directory, \
                mock.patch.object(owner, "OutputLease") as lease, \
                mock.patch.object(owner, "run_group") as dispatch, \
                redirect_stdout(output):
            self.assertEqual(owner.main(), 2)
        self.assertEqual(json.loads(output.getvalue()), {
            "schema": "rovent-before-native-capability/1",
            "status": "unsupported",
            "reason": "original_owner_supervisor_not_connected",
            "input_registration": "not_started",
            "test_dispatch": "not_started",
            "physical_completion": "not_observed",
        })
        registration.assert_not_called()
        open_directory.assert_not_called()
        lease.assert_not_called()
        dispatch.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_no_dispatch_closes_original_input_and_future_output_leases(self):
        def before_dispatch(args, *pinned_and_consumer):
            consumer = pinned_and_consumer[-1]
            self.assertFalse(consumer.dispatched)
            with self.registry.guard() as guard:
                self.assertTrue(guard.conflicts([self.binary])["conflicts"])
                self.assertTrue(guard.conflicts([self.output])["conflicts"])
            return 0

        self.assertEqual(self._run_main(before_dispatch), 0)
        self.assertEqual(list(self.registry.root.glob("consumer-*.json")), [])

    def test_dispatched_owner_keeps_inputs_and_output_when_fence_unknown(self):
        def dispatched(args, *pinned_and_consumer):
            pinned_and_consumer[-1].dispatched = True
            return 0

        self.assertEqual(self._run_main(dispatched), 0)
        with self.registry.guard() as guard:
            self.assertTrue(guard.conflicts([self.binary])["conflicts"])
            self.assertTrue(guard.conflicts([self.output])["conflicts"])
        from managed_storage import read_record_snapshot
        for path in self.registry.root.glob("consumer-*.json"):
            row, _, _ = read_record_snapshot(path, schema="rovezero.consumer-input-lease.v1")
            self.assertEqual(row["state"], "active")

    def test_original_group_end_retains_physical_unknown_and_protection(self):
        def dispatched_and_joined(args, *pinned_and_consumer):
            consumer = pinned_and_consumer[-1]
            consumer.dispatched = True
            consumer.original_group_gone = True
            return 0

        self.assertEqual(self._run_main(dispatched_and_joined), 0)
        from managed_storage import read_record_snapshot
        rows = list(self.registry.root.glob("consumer-*.json"))
        self.assertEqual(len(rows), 2)
        for path in rows:
            row, _, _ = read_record_snapshot(path, schema="rovezero.consumer-input-lease.v1")
            self.assertEqual(row["state"], "physical_unknown_retained")
        with self.registry.guard() as guard:
            self.assertTrue(guard.conflicts([self.binary])["conflicts"])
            self.assertTrue(guard.conflicts([self.output])["conflicts"])

    def test_system_runtime_registers_canonical_libraries_only(self):
        consumer = mock.Mock()
        consumer.registry = object()
        manifest = {"system_runtime_lock": {"runtime": {"configuration": {"libraries": [
            {"canonical_path": "/system/libort.so", "requested_path": "/alias/libort.so"},
            {"canonical_path": "/system/libort.so", "requested_path": "/other/libort.so"},
        ]}}}}
        self.assertEqual(owner.system_runtime_inputs(consumer, manifest), [Path("/system/libort.so")])

    def test_hash_checks_recorded_signal_between_chunks(self):
        state = {"signal": None}
        consumer = mock.Mock(interruption=state)

        class SignalAfterFirstChunk:
            reads = 0

            def read(self, _limit):
                self.reads += 1
                state["signal"] = signal.SIGTERM
                return b"fixture"

        source = SignalAfterFirstChunk()
        with self.assertRaises(KeyboardInterrupt):
            owner.hash_stream(source, time.monotonic() + 10, consumer)
        self.assertEqual(source.reads, 1)

    def test_open_failure_closes_never_dispatched_owner(self):
        with mock.patch.object(sys, "argv", self.argv), \
                mock.patch.dict(os.environ, {}, clear=True), \
                mock.patch.object(owner, "original_owner_supervisor_capability",
                                  return_value=owner.OriginalOwnerSupervisorCapability.CONNECTED), \
                mock.patch.object(owner, "open_directory", side_effect=OSError("fixture")):
            with self.assertRaisesRegex(OSError, "fixture"):
                owner.main()
        self.assertEqual(list(self.registry.root.glob("consumer-*.json")), [])

    def test_pre_dispatch_term_closes_original_consumer_leases(self):
        original_open = owner.open_directory
        called = False

        def stop_then_open(path):
            nonlocal called
            if not called:
                called = True
                os.kill(os.getpid(), signal.SIGTERM)
            return original_open(path)

        def require_no_stop(args, *pinned_and_consumer):
            owner.check_consumer_interruption(pinned_and_consumer[-1])
            self.fail("a pre-dispatch stop must not start the child")

        with mock.patch.object(owner, "open_directory", side_effect=stop_then_open):
            with self.assertRaises(KeyboardInterrupt):
                self._run_main(require_no_stop)
        self.assertEqual(list(self.registry.root.glob("consumer-*.json")), [])

    def test_repeated_signals_drain_actual_child_and_preserve_unknown(self):
        # A real short-lived child ignores TERM so the original process-group
        # supervisor must complete its drain while the parent gets another
        # signal. No GPU, external engine, or production asset is involved.
        self.binary.write_text(
            "#!/usr/bin/python3\nimport signal,time\n"
            "signal.signal(signal.SIGTERM, lambda *_: None)\n"
            "print('fixture-child-ready', flush=True)\n"
            "while True: time.sleep(0.02)\n", encoding="utf-8")
        self.binary.chmod(0o700)
        source_file = self.source / "crates/rz-uci/src/rovent_bootstrap/launch/v3.rs"
        source_file.parent.mkdir(parents=True)
        source_file.write_text("fixture", encoding="utf-8")

        def pin(root, path):
            target = root / path
            target.parent.mkdir(parents=True, exist_ok=True)
            data = ("fixture-" + path).encode()
            target.write_bytes(data)
            return {"path": path, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}

        paths = ("inputs/export/export.json", "inputs/model/rules-descriptor.json",
                 "inputs/export/public_memory.onnx", "inputs/export/shared_pc_if.onnx")
        provenance = [pin(self.assets, path) for path in paths]
        weights = pin(self.assets, "inputs/model/weights.pt")
        checkpoint = pin(self.assets, "inputs/model/provenance.json")
        startup = pin(self.templates, "startup.json")
        runtime = self.root / "runtime.so"
        runtime.write_bytes(b"fixture-runtime")
        manifest = {"system_runtime_lock": {"runtime": {"kind": "system_cuda",
                    "configuration": {"libraries": [{"canonical_path": str(runtime)}]}}},
                    "startup_document": startup,
                    "engine": {"pc": {"proposer": {"weights": weights, "provenance": checkpoint},
                                        "critic": {"weights": weights, "provenance": checkpoint}}},
                    "provenance": provenance}
        (self.templates / "lock.json").write_text(json.dumps({"manifest": manifest}), encoding="utf-8")
        finish = []

        class FixtureOutputLease:
            def __init__(self, output, _source, **_kwargs):
                self.output = output

            def acquire(self):
                self.output.mkdir()
                return self

            def check_budget(self):
                pass

            def finish(self, *, status, tree_gone):
                finish.append((status, tree_gone))

        signals_sent = []

        def send_when_ready():
            deadline = time.monotonic() + 8
            log = self.output / "test.stdout"
            while time.monotonic() < deadline:
                if log.exists() and "fixture-child-ready" in log.read_text(encoding="utf-8"):
                    break
                time.sleep(0.01)
            for _ in range(2):
                os.kill(os.getpid(), signal.SIGTERM)
                signals_sent.append(True)
                time.sleep(0.05)

        sender = threading.Thread(target=send_when_ready)
        with mock.patch.object(sys, "argv", self.argv), \
                mock.patch.dict(os.environ, {}, clear=True), \
                mock.patch.object(owner, "original_owner_supervisor_capability",
                                  return_value=owner.OriginalOwnerSupervisorCapability.CONNECTED), \
                mock.patch.object(owner, "OutputLease", FixtureOutputLease), \
                mock.patch.object(owner, "protect_output"):
            sender.start()
            try:
                self.assertEqual(owner.main(), 1)
            finally:
                sender.join(timeout=10)
        self.assertFalse(sender.is_alive())
        self.assertEqual(len(signals_sent), 2)
        self.assertEqual(finish, [("test_failed_or_unconfirmed", True)])
        from managed_storage import read_record_snapshot
        rows = list(self.registry.root.glob("consumer-*.json"))
        self.assertGreaterEqual(len(rows), 2)
        for path in rows:
            row, _, _ = read_record_snapshot(path, schema="rovezero.consumer-input-lease.v1")
            self.assertEqual(row["state"], "physical_unknown_retained")
        receipt = json.loads((self.output / "owner-receipt.json").read_text(encoding="utf-8"))
        self.assertEqual(receipt["physical_completion"], "not_inferred_from_process_exit")


if __name__ == "__main__":
    unittest.main()
