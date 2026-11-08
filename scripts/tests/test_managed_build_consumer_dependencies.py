"""A participating build cleanup cannot evict another consumer's inputs."""
from pathlib import Path
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from managed_storage import (ManagedBuild, SCHEMA, StorageError, read_record,
                             read_record_snapshot, recover_build, write_record)
from storage_consumer_registry import ConsumerRegistry, bind_catalog
from storage_guard import CatalogGuard
import managed_outputs
import rovezero_usage_snapshot
import run_managed
from run_managed import _consumer_registry_options


class ManagedBuildConsumerDependencyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="rovezero-build-consumer-")
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.source = self.base / "source"
        self.source.mkdir()
        self.root = self.base / "build"
        self.registry = ConsumerRegistry.initialize(self.base / "registry", scope="fixture")

    def completed_slot(self, *, bound=True):
        owner = ManagedBuild(self.root, self.source,
                             dependency_registry=self.registry if bound else None).acquire()
        payload = owner.target / "retained.bin"
        payload.write_bytes(b"a" * 2048)
        owner.finish(exit_code=0, tree_gone=True, _not_dispatched=True)
        return owner.slot, payload

    def test_prune_skips_registered_input_and_removes_after_release(self):
        old_slot, payload = self.completed_slot()
        lease = self.registry.register([payload])
        other = self.base / "other"
        other.mkdir()
        owner = ManagedBuild(self.root, other, slot_limit=256, total_limit=256,
                             dependency_registry=self.registry,
                             dependency_registry_required=True).acquire()
        self.assertTrue(payload.is_file())
        lease.close(not_started=True)
        owner._prune()
        self.assertFalse(old_slot.exists())
        owner.finish(exit_code=0, tree_gone=True, _not_dispatched=True)

    def test_required_registry_missing_preserves_candidates(self):
        old_slot, payload = self.completed_slot(bound=False)
        other = self.base / "other"
        other.mkdir()
        owner = ManagedBuild(self.root, other, slot_limit=256, total_limit=256,
                             dependency_registry_required=True).acquire()
        self.assertTrue(old_slot.exists())
        self.assertTrue(payload.exists())
        owner.finish(exit_code=0, tree_gone=True, _not_dispatched=True)

    def test_nonempty_unbound_catalog_is_not_adopted_and_execution_is_isolated(self):
        old_slot, payload = self.completed_slot(bound=False)
        other = self.base / "other"
        other.mkdir()
        owner = ManagedBuild(self.root, other, slot_limit=256, total_limit=256,
                             dependency_registry=self.registry).acquire()
        self.assertIsNone(owner.dependency_registry)
        self.assertTrue(owner.dependency_registry_required)
        self.assertTrue((self.root / "consumer-registry.required").is_file())
        self.assertFalse((self.root / "consumer-registry.binding").exists())
        self.assertTrue(owner.target.is_relative_to(owner.scratch))
        self.assertTrue(old_slot.exists())
        self.assertTrue(payload.exists())
        owner.finish(exit_code=0, tree_gone=True, _not_dispatched=True)
        later = ManagedBuild(self.root, other).acquire()
        self.assertTrue(later.dependency_registry_required)
        self.assertTrue(later.target.is_relative_to(later.scratch))
        self.assertTrue(payload.exists())
        later.finish(exit_code=0, tree_gone=True, _not_dispatched=True)

    def test_same_slot_reuse_isolates_protected_target_without_owner_leak(self):
        _, payload = self.completed_slot()
        lease = self.registry.register([payload])
        owner = ManagedBuild(self.root, self.source, slot_limit=256, total_limit=256,
                             dependency_registry=self.registry,
                             dependency_registry_required=True).acquire()
        self.assertTrue(owner.target.is_relative_to(owner.scratch))
        owner.target.joinpath("new-output").write_bytes(b"new")
        self.assertTrue(payload.exists())
        owner.finish(exit_code=0, tree_gone=True, _not_dispatched=True)
        self.assertFalse((self.root / "owner.json").exists())
        lease.close(not_started=True)

    def test_binding_is_loaded_without_arguments_for_a_later_owner(self):
        first = ManagedBuild(self.root, self.source, dependency_registry=self.registry).acquire()
        payload = first.target / "in-use.bin"
        payload.write_bytes(b"input")
        first.finish(exit_code=0, tree_gone=True, _not_dispatched=True)
        lease = self.registry.register([payload])
        second = ManagedBuild(self.root, self.source).acquire()
        self.assertEqual(second.dependency_registry.registry_id, self.registry.registry_id)
        self.assertTrue(second.target.is_relative_to(second.scratch))
        self.assertTrue(payload.exists())
        second.finish(exit_code=0, tree_gone=True, _not_dispatched=True)
        lease.close(not_started=True)

    def test_unavailable_required_registry_uses_new_scratch_across_runs(self):
        first = ManagedBuild(self.root, self.source, dependency_registry_required=True).acquire()
        first.scratch.joinpath("preserved").write_bytes(b"old")
        first_scratch = first.scratch
        self.assertFalse(first.finish(exit_code=0, tree_gone=True, _not_dispatched=True)["temporary_removed"])
        second = ManagedBuild(self.root, self.source).acquire()
        self.assertNotEqual(second.scratch, first_scratch)
        self.assertTrue(second.target.is_relative_to(second.scratch))
        self.assertTrue(first_scratch.joinpath("preserved").exists())
        second.finish(exit_code=0, tree_gone=True, _not_dispatched=True)
        self.assertFalse((self.root / "owner.json").exists())

    def test_retained_scratch_is_reclaimed_after_consumer_release(self):
        first = ManagedBuild(self.root, self.source, dependency_registry=self.registry).acquire()
        payload = first.scratch / "in-use.bin"
        payload.write_bytes(b"input")
        first_scratch = first.scratch
        lease = self.registry.register([payload])
        first.finish(exit_code=0, tree_gone=True, _not_dispatched=True)
        second = ManagedBuild(self.root, self.source).acquire()
        self.assertTrue(payload.exists())
        self.assertNotEqual(second.scratch, first_scratch)
        second.finish(exit_code=0, tree_gone=True, _not_dispatched=True)
        lease.close(not_started=True)
        third = ManagedBuild(self.root, self.source).acquire()
        self.assertFalse(first_scratch.exists())
        third.finish(exit_code=0, tree_gone=True, _not_dispatched=True)

    def test_registry_temporarily_unavailable_preserves_then_reclaims_scratch(self):
        first = ManagedBuild(self.root, self.source, dependency_registry=self.registry).acquire()
        first.scratch.joinpath("data").write_bytes(b"input")
        first_scratch = first.scratch
        moved = self.base / "registry-offline"
        self.registry.root.rename(moved)
        self.assertFalse(first.finish(exit_code=0, tree_gone=True, _not_dispatched=True)["temporary_removed"])
        second = ManagedBuild(self.root, self.source).acquire()
        self.assertNotEqual(second.scratch, first_scratch)
        second.finish(exit_code=0, tree_gone=True, _not_dispatched=True)
        moved.rename(self.registry.root)
        third = ManagedBuild(self.root, self.source).acquire()
        self.assertFalse(first_scratch.exists())
        third.finish(exit_code=0, tree_gone=True, _not_dispatched=True)

    def test_dispatched_scratch_with_unknown_physical_completion_is_not_reclaimed(self):
        first = ManagedBuild(self.root, self.source, dependency_registry=self.registry).acquire()
        payload = first.scratch / "in-use.bin"
        payload.write_bytes(b"input")
        first_scratch = first.scratch
        lease = self.registry.register([payload])
        first._dispatch_started = True
        with patch.object(managed_outputs, "finalize_supervised_outputs"):
            first.finish(exit_code=0, tree_gone=True)
        lease.close(physical_complete=True)  # Independent proof for this declared input only.
        second = ManagedBuild(self.root, self.source).acquire()
        self.assertTrue(first_scratch.exists())  # No physical proof for its producing run.
        second.finish(exit_code=0, tree_gone=True, _not_dispatched=True)

    def test_unknown_previous_target_uses_isolated_next_target_without_lease(self):
        first = ManagedBuild(self.root, self.source, dependency_registry=self.registry).acquire()
        payload = first.target / "previous.bin"
        payload.write_bytes(b"input")
        first.scratch.joinpath("scratch").write_bytes(b"input")
        first._dispatch_started = True
        with patch.object(managed_outputs, "finalize_supervised_outputs"):
            receipt = first.finish(exit_code=0, tree_gone=True)
        self.assertFalse(receipt["temporary_removed"])
        second = ManagedBuild(self.root, self.source).acquire()
        self.assertTrue(second.target.is_relative_to(second.scratch))
        self.assertTrue(payload.exists())
        second.finish(exit_code=0, tree_gone=True, _not_dispatched=True)

    def test_protected_completed_slots_can_exceed_limit_without_new_job_failure(self):
        paths = []
        with patch.object(managed_outputs, "finalize_supervised_outputs"):
            for i in range(5):
                source = self.base / ("source-" + str(i))
                source.mkdir()
                owner = ManagedBuild(self.root, source, dependency_registry=self.registry,
                                     slots_limit=4).acquire()
                payload = owner.scratch / "protected.bin"
                payload.write_bytes(b"input")
                paths.append(payload)
                owner._dispatch_started = True
                owner.finish(exit_code=0, tree_gone=True)
        self.assertTrue(all(path.exists() for path in paths))
        self.assertEqual(len(list((self.root / "slots").iterdir())), 5)
        self.assertFalse((self.root / "owner.json").exists())

    def test_finish_reports_protected_scratch_as_retained(self):
        owner = ManagedBuild(self.root, self.source, dependency_registry=self.registry,
                             dependency_registry_required=True).acquire()
        payload = owner.scratch / "in-use.bin"
        payload.write_bytes(b"input")
        lease = self.registry.register([payload])
        receipt = owner.finish(exit_code=0, tree_gone=True, _not_dispatched=True)
        self.assertFalse(receipt["temporary_removed"])
        self.assertTrue(payload.exists())
        self.assertFalse((self.root / "owner.json").exists())
        lease.close(not_started=True)

    def test_receipt_write_failure_preserves_scratch_but_closes_owner(self):
        owner = ManagedBuild(self.root, self.source, dependency_registry=self.registry).acquire()
        payload = owner.scratch / "in-use.bin"
        payload.write_bytes(b"input")
        lease = self.registry.register([payload])
        with patch.object(owner, "_record_retained_scratch", side_effect=OSError("disk error")):
            receipt = owner.finish(exit_code=0, tree_gone=True, _not_dispatched=True)
        self.assertEqual(receipt["retained_scratch_receipt"], "unavailable_preserved")
        self.assertFalse(receipt["temporary_removed"])
        self.assertTrue(payload.exists())
        self.assertFalse((self.root / "owner.json").exists())
        lease.close(not_started=True)

    def test_pre_dispatch_failure_closes_exact_active_slot_before_owner(self):
        owner = ManagedBuild(self.root, self.source, dependency_registry=self.registry)
        original_prune = owner._prune
        def fail_after_publication():
            if owner.record:
                raise StorageError("pre-dispatch injected failure")
            return original_prune()
        with patch.object(owner, "_prune", side_effect=fail_after_publication):
            with self.assertRaisesRegex(StorageError, "pre-dispatch injected failure"):
                owner.acquire()
        self.assertFalse((self.root / "owner.json").exists())
        self.assertEqual(read_record(owner.slot / "record.json")["state"], "complete")
        next_owner = ManagedBuild(self.root, self.source).acquire()
        next_owner.finish(exit_code=0, tree_gone=True, _not_dispatched=True)

    def test_runner_forwards_pinned_registry_and_reports_scope(self):
        keys = ("RZ_CONSUMER_REGISTRY_ROOT", "RZ_CONSUMER_REGISTRY_ID", "RZ_CONSUMER_REGISTRY_REQUIRED")
        clean = {k: v for k, v in os.environ.items() if k not in keys}
        with patch.dict(os.environ, clean, clear=True):
            options = _consumer_registry_options(self.registry.root, self.registry.registry_id, False)
            owner = ManagedBuild(self.root, self.source, **options).acquire()
            env = owner.environment()
            self.assertEqual(env["RZ_CONSUMER_REGISTRY_ROOT"], str(self.registry.root))
            self.assertEqual(env["RZ_CONSUMER_REGISTRY_ID"], self.registry.registry_id)
            self.assertEqual(env["RZ_CONSUMER_REGISTRY_REQUIRED"], "1")
            receipt = owner.finish(exit_code=0, tree_gone=True, _not_dispatched=True)
        self.assertEqual(receipt["consumer_dependency_coverage"], "registered_only")

    def test_runner_missing_registry_defers_removal_instead_of_failing_setup(self):
        keys = ("RZ_CONSUMER_REGISTRY_ROOT", "RZ_CONSUMER_REGISTRY_ID", "RZ_CONSUMER_REGISTRY_REQUIRED")
        clean = {k: v for k, v in os.environ.items() if k not in keys}
        with patch.dict(os.environ, clean, clear=True):
            options = _consumer_registry_options(self.base / "missing", "f" * 32, False)
            self.assertIsNone(options["dependency_registry"])
            self.assertTrue(options["dependency_registry_required"])
            owner = ManagedBuild(self.root, self.source, **options).acquire()
            owner.scratch.joinpath("keep").write_bytes(b"unconfirmed")
            receipt = owner.finish(exit_code=0, tree_gone=True, _not_dispatched=True)
        self.assertFalse(receipt["temporary_removed"])
        self.assertEqual(receipt["consumer_dependency_coverage"], "required_registry_unavailable")

    def test_runner_cli_keeps_registered_input_during_actual_cleanup(self):
        catalog = self.base / "outputs"
        catalog.mkdir()
        leases = []
        def supervised(_command, _source, env, _timeout, _check):
            registry = ConsumerRegistry(env["RZ_CONSUMER_REGISTRY_ROOT"],
                                        registry_id=env["RZ_CONSUMER_REGISTRY_ID"])
            payload = Path(env["TMPDIR"]) / "in-use.bin"
            payload.write_bytes(b"input")
            leases.append(registry.register([payload]))
            return 0, True, None
        args = ["run_managed.py", "--source-root", str(self.source),
                "--consumer-registry-root", str(self.registry.root),
                "--consumer-registry-id", self.registry.registry_id,
                "--", sys.executable]
        with patch.object(sys, "argv", args), \
             patch.object(run_managed, "canonical_root", return_value=self.root), \
             patch.object(managed_outputs, "canonical_catalog", return_value=catalog), \
             patch.object(run_managed, "sample_output_budget", return_value=0), \
             patch.object(run_managed, "run_group", side_effect=supervised), \
             patch.object(rovezero_usage_snapshot, "claim_total_usage_maintenance", return_value=False):
            self.assertEqual(run_managed.main(), 0)
        receipt_path = next((self.root / "slots").glob("*/last-run.json"))
        receipt = json.loads(receipt_path.read_text(encoding="utf8"))
        self.assertEqual(receipt["consumer_dependency_coverage"], "registered_only")
        self.assertFalse(receipt["temporary_removed"])
        self.assertEqual(len(leases), 1)
        leases[0].close(not_started=True)

    def test_supervisor_retains_input_lease_when_physical_completion_is_unknown(self):
        scripts = self.source / "scripts"
        scripts.mkdir()
        validator = scripts / "pals_model_validate.py"
        validator.write_text("# mock CPU validator\n", encoding="utf8")
        input_file = self.base / "input.bin"
        input_file.write_bytes(b"input")
        future = self.base / "runtime-store"
        participating = self.base / "participating-output-catalog"
        participating.mkdir()
        with CatalogGuard(participating / "catalog.lock") as guard:
            bind_catalog(participating, self.registry, guard)
        output = self.base / "outputs"
        output.mkdir()
        observed = {}
        def supervised(_command, _source, env, _timeout, _check):
            observed["owner"] = env["RZ_CONSUMER_LEASE_OWNER_ID"]
            self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "-1")
            with self.registry.guard() as guard:
                self.assertTrue(guard.conflicts([input_file])["conflicts"])
                self.assertTrue(guard.conflicts([future])["conflicts"])
            return 0, True, None
        args = ["run_managed.py", "--source-root", str(self.source),
                "--consumer-registry-root", str(self.registry.root),
                "--consumer-registry-id", self.registry.registry_id,
                "--consumer-input", str(input_file), "--consumer-future-input", str(future),
                "--consumer-input-catalog", str(participating), "--consumer-input-cpu-only",
                "--", sys.executable, str(validator)]
        with patch.object(sys, "argv", args), \
             patch.object(run_managed, "canonical_root", return_value=self.root), \
             patch.object(managed_outputs, "canonical_catalog", return_value=output), \
             patch.object(run_managed, "sample_output_budget", return_value=0), \
             patch.object(run_managed, "run_group", side_effect=supervised), \
             patch.object(rovezero_usage_snapshot, "claim_total_usage_maintenance", return_value=False):
            self.assertEqual(run_managed.main(), 0)
        record = read_record_snapshot(self.registry.root / ("consumer-" + observed["owner"] + ".json"),
                                      schema="rovezero.consumer-input-lease.v1")[0]
        self.assertEqual(record["state"], "physical_unknown_retained")
        self.assertTrue(record["tree_gone"])
        self.assertNotIn("completion", record)

    def test_supervisor_rejects_unbound_catalog_before_child_dispatch(self):
        scripts = self.source / "scripts"
        scripts.mkdir()
        validator = scripts / "pals_model_validate.py"
        validator.write_text("# mock CPU validator\n", encoding="utf8")
        input_file = self.base / "input.bin"
        input_file.write_bytes(b"input")
        unbound = self.base / "unbound"
        unbound.mkdir()
        args = ["run_managed.py", "--source-root", str(self.source),
                "--consumer-registry-root", str(self.registry.root),
                "--consumer-registry-id", self.registry.registry_id,
                "--consumer-input", str(input_file), "--consumer-input-catalog", str(unbound),
                "--consumer-input-cpu-only", "--", sys.executable, str(validator)]
        with patch.object(sys, "argv", args), \
             patch.object(run_managed, "canonical_root", return_value=self.root), \
             patch.object(run_managed, "run_group") as dispatch:
            self.assertEqual(run_managed.main(), 125)
        dispatch.assert_not_called()
        self.assertFalse((self.root / "owner.json").exists())

    def test_catalog_only_is_forwarded_to_non_validator_child(self):
        participating = self.base / "participating-output-catalog"
        participating.mkdir()
        with CatalogGuard(participating / "catalog.lock") as guard:
            bind_catalog(participating, self.registry, guard)
        output = self.base / "outputs"
        output.mkdir()
        observed = {}
        def supervised(_command, _source, env, _timeout, _check):
            observed["catalogs"] = json.loads(env["RZ_CONSUMER_INPUT_CATALOGS"])
            return 0, True, None
        args = ["run_managed.py", "--source-root", str(self.source),
                "--consumer-registry-root", str(self.registry.root),
                "--consumer-registry-id", self.registry.registry_id,
                "--consumer-input-catalog", str(participating), "--", sys.executable, "-c", "pass"]
        with patch.object(sys, "argv", args), \
             patch.object(run_managed, "canonical_root", return_value=self.root), \
             patch.object(managed_outputs, "canonical_catalog", return_value=output), \
             patch.object(run_managed, "sample_output_budget", return_value=0), \
             patch.object(run_managed, "run_group", side_effect=supervised), \
             patch.object(rovezero_usage_snapshot, "claim_total_usage_maintenance", return_value=False):
            self.assertEqual(run_managed.main(), 0)
        self.assertEqual(observed["catalogs"], [str(participating)])

    def test_recovery_preserves_registered_input_and_owner_until_release(self):
        owner = ManagedBuild(self.root, self.source).acquire()
        payload = owner.target / "in-use.bin"
        payload.write_bytes(b"input")
        lease = self.registry.register([payload])
        original, digest, _ = read_record_snapshot(self.root / "owner.json")
        write_record(self.root / "closure.json", {"schema": SCHEMA, "owner": original, "tree_gone": True})
        owner.guard.release()
        with patch("managed_storage.owner_proven_dead", return_value=True), \
             patch.object(managed_outputs, "finalize_supervised_outputs"):
            with self.assertRaisesRegex(StorageError, "recovery scratch required"):
                recover_build(self.root, digest, dependency_registry=self.registry,
                              dependency_registry_required=True)
            self.assertTrue(payload.exists())
            self.assertEqual(read_record(self.root / "owner.json"), original)
            lease.close(not_started=True)
            self.assertTrue((self.root / "consumer-registry.required").exists())
            self.assertFalse((self.root / "consumer-registry.binding").exists())

    def test_scoped_recovery_uses_original_owner_scratch_token(self):
        owner = ManagedBuild(self.root, self.source, dependency_registry=self.registry).acquire()
        payload = owner.scratch / "in-use.bin"
        payload.write_bytes(b"input")
        lease = self.registry.register([payload])
        original, digest, _ = read_record_snapshot(self.root / "owner.json")
        write_record(self.root / "closure.json", {"schema": SCHEMA, "owner": original, "tree_gone": True})
        owner.guard.release()
        with patch("managed_storage.owner_proven_dead", return_value=True), \
             patch.object(managed_outputs, "finalize_supervised_outputs"):
            result = recover_build(self.root, digest)
            self.assertTrue(payload.exists())
            self.assertFalse((self.root / "owner.json").exists())
            lease.close(not_started=True)
        self.assertTrue(result["recovered"])
        self.assertEqual(result["physical_completion"], "unknown_preserved")
        self.assertTrue(payload.exists())
        next_owner = ManagedBuild(self.root, self.source).acquire()
        self.assertTrue(next_owner.target.is_relative_to(next_owner.scratch))
        self.assertTrue(payload.exists())
        next_owner.finish(exit_code=0, tree_gone=True, _not_dispatched=True)


if __name__ == "__main__":
    unittest.main()
