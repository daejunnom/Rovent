"""Temporary fixture ELF only; never accesses project output or GPUs."""
from pathlib import Path
from contextlib import nullcontext
import json
import os
import sys
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from managed_storage import StorageError, write_record
import storage_explicit_elf_retirement as explicit
import test_storage_artifact_review as fixture


class _DependencyGuard:
    def __init__(self):
        self.held = False
        self.conflicting = []
        self.unknown = []

    def guard(self):
        return self

    def __enter__(self):
        self.held = True
        return self

    def __exit__(self, *_):
        self.held = False

    def conflicts(self, targets):
        if not self.held:
            raise AssertionError("dependency registry lock not held")
        return {"conflicts": self.conflicting, "unknown": self.unknown, "coverage": "registered_only"}


@unittest.skipUnless(sys.platform == "linux", "native Linux retirement only")
class ExplicitRetirementTests(unittest.TestCase):
    write_canonical = fixture.ArtifactReviewTests.write_canonical
    write_research = fixture.ArtifactReviewTests.write_research

    def setUp(self):
        fixture.ArtifactReviewTests.setUp(self)
        self.record["outer_supervisor"] = {"owner": {"pid": self.record["pid"],
                                                        "lifetime": self.record["lifetime"]}}
        write_record(self.record_path, self.record)
        write_record(self.output / ".rovezero-output.json", self.record)
        self.leaf = self.output / "rz-experiments"
        self.leaf.write_bytes(b"\x7fELF-unique-fixture-never-executed")
        file_pin = fixture.pin(self.leaf)
        self.receipt = self.output / "build-receipt.json"
        self.receipt.write_text(json.dumps({
            "schema": "rovent.managed-cpu-four-target-build.v1", "gpu_executed": False,
            "source_commit_requested_by_windows_git": "a" * 40,
            "builds": {"one": {"artifact": self.leaf.name, "returncode": 0,
                               "retained": {"bytes": file_pin["bytes"], "sha256": file_pin["sha256"]}}}}))
        self.plan = {"schema": explicit.SCHEMA, "authorization": explicit.AUTHORIZATION,
                     "catalog": str(self.catalog), "key": self.record["key"],
                     "record_sha256": fixture.pin(self.record_path)["sha256"],
                     "receipt": fixture.pin(self.receipt), "source_commit": "a" * 40,
                     "files": [{"relative": self.leaf.name, "bytes": file_pin["bytes"],
                                "sha256": file_pin["sha256"]}]}
        self.plan_path = self.base / "explicit-plan.json"
        self.dependencies = _DependencyGuard()

    def run_explicit(self):
        # Only temporary fixtures bypass the unconnected cross-catalog gate;
        # these tests do not establish that production dependencies are safe.
        # The small in-memory lock double exercises retirement sequencing,
        # while real registries below exercise the actual binding contract.
        binding = (patch.object(explicit, "require_catalog_binding")
                   if isinstance(self.dependencies, _DependencyGuard) else nullcontext())
        with patch.object(explicit, "CROSS_CATALOG_DEPENDENCY_GATE_CONNECTED", True), binding:
            return explicit.retire_explicit_cpu_elf(fixture.dump(self.plan_path, self.plan), seconds=30,
                                                    dependency_registry=self.dependencies)

    def bound_registry(self):
        from storage_consumer_registry import ConsumerRegistry, bind_catalog
        from storage_guard import CatalogGuard
        registry = ConsumerRegistry.initialize(self.base / "registry", scope="isolated_test")
        with CatalogGuard(self.catalog / "catalog.lock") as guard:
            bind_catalog(self.catalog, registry, guard)
        return registry

    def test_registered_conflict_and_unknown_preserve_original(self):
        for field in ("conflicting", "unknown"):
            setattr(self.dependencies, field, [{"reason": field}])
            with self.assertRaisesRegex(StorageError, "consumer dependency"):
                self.run_explicit()
            self.assertTrue(self.leaf.exists())
            self.assertFalse(self.dependencies.held)
            setattr(self.dependencies, field, [])

    def test_dependency_lock_is_held_at_unlink(self):
        unlink = explicit.os.unlink

        def check_lock(*args, **kwargs):
            self.assertTrue(self.dependencies.held)
            return unlink(*args, **kwargs)

        with patch.object(explicit.base, "_process_census", return_value={"scope": "test_mock"}), \
                patch.object(explicit.os, "unlink", side_effect=check_lock):
            self.run_explicit()
        self.assertFalse(self.leaf.exists())
        self.assertFalse(self.dependencies.held)

    def test_real_registered_consumer_blocks_until_explicit_close(self):
        registry = self.bound_registry()
        lease = registry.register([self.leaf])
        self.dependencies = registry
        with self.assertRaisesRegex(StorageError, "consumer dependency"):
            self.run_explicit()
        self.assertTrue(self.leaf.exists())
        lease.close(not_started=True)
        with patch.object(explicit.base, "_process_census", return_value={"scope": "test_mock"}):
            self.run_explicit()
        self.assertFalse(self.leaf.exists())

    def test_unrelated_registered_input_does_not_block(self):
        registry = self.bound_registry()
        lease = registry.register([self.source])
        self.dependencies = registry
        with patch.object(explicit.base, "_process_census", return_value={"scope": "test_mock"}):
            self.run_explicit()
        self.assertFalse(self.leaf.exists())
        lease.close(not_started=True)

    def test_unbound_different_registry_and_required_marker_preserve_leaf(self):
        from storage_consumer_registry import ConsumerRegistry, bind_catalog, mark_catalog_registry_unavailable
        from storage_guard import CatalogGuard
        registry = ConsumerRegistry.initialize(self.base / "registry", scope="isolated_test")
        other = ConsumerRegistry.initialize(self.base / "other-registry", scope="isolated_test")
        self.dependencies = registry
        with self.assertRaises(StorageError):
            self.run_explicit()
        self.assertTrue(self.leaf.exists())
        with CatalogGuard(self.catalog / "catalog.lock") as guard:
            bind_catalog(self.catalog, registry, guard)
        self.dependencies = other
        with self.assertRaises(StorageError):
            self.run_explicit()
        self.assertTrue(self.leaf.exists())
        with CatalogGuard(self.catalog / "catalog.lock") as guard:
            mark_catalog_registry_unavailable(self.catalog, guard, force_bound=True)
        self.dependencies = registry
        with self.assertRaises(StorageError):
            self.run_explicit()
        self.assertTrue(self.leaf.exists())

    def test_explicit_leaf_only_marker_and_receipt_are_preserved(self):
        before = [p.read_bytes() for p in (self.record_path, self.output / ".rovezero-output.json", self.receipt)]
        with patch.object(explicit.base, "_process_census", return_value={"scope": "test_mock"}):
            result = self.run_explicit()
        self.assertEqual(result["retired_files"], [self.leaf.name])
        self.assertFalse(self.leaf.exists())
        self.assertEqual(before, [p.read_bytes() for p in (self.record_path, self.output / ".rovezero-output.json", self.receipt)])
        self.assertEqual(result["source_reproducibility"], "not_asserted")
        self.assertEqual(result["physical_space_reclaimed"], "unknown")
        self.assertTrue(Path(result["journal"]["path"]).is_file())

    def test_unknown_census_fails_without_unlink(self):
        with patch.object(explicit.base, "_process_census", side_effect=StorageError("census unknown")):
            with self.assertRaisesRegex(StorageError, "census unknown"):
                self.run_explicit()
        self.assertTrue(self.leaf.exists())

    def test_dependency_observation_preserves_originals_and_does_not_open_retirement_gate(self):
        plan_pin = fixture.dump(self.plan_path, self.plan)
        result = explicit.observe_explicit_cpu_elf_dependencies(plan_pin, seconds=30)
        self.assertFalse(result["deletion_authorized"])
        self.assertFalse(result["retirement_preflight"])
        self.assertEqual(len(result["observations"]), 1)
        self.assertEqual(result["observations"][0]["input_completeness"], "unknown")
        self.assertTrue(self.leaf.exists())
        with self.assertRaisesRegex(StorageError, "active/unknown dependency protection"):
            explicit.retire_explicit_cpu_elf(plan_pin, seconds=30)
        self.assertTrue(self.leaf.exists())

    def test_duplicate_names_and_unapproved_names_rejected(self):
        self.plan["files"].append(dict(self.plan["files"][0]))
        with self.assertRaisesRegex(StorageError, "unique allowlisted"):
            self.run_explicit()
        self.plan["files"] = [dict(self.plan["files"][0], relative="other")]
        with self.assertRaisesRegex(StorageError, "unique allowlisted"):
            self.run_explicit()
        self.assertTrue(self.leaf.exists())

    def test_wrong_receipt_or_no_explicit_authorization_is_rejected(self):
        self.plan["source_commit"] = "b" * 40
        with self.assertRaisesRegex(StorageError, "CPU-only build receipt"):
            self.run_explicit()
        self.plan["authorization"] = "automatic"
        with self.assertRaisesRegex(StorageError, "authorization"):
            self.run_explicit()
        self.assertTrue(self.leaf.exists())

    def test_active_original_owner_row_is_not_retired(self):
        self.record["state"] = "active"
        write_record(self.record_path, self.record)
        write_record(self.output / ".rovezero-output.json", self.record)
        with self.assertRaisesRegex(StorageError, "active catalog"):
            self.run_explicit()
        self.assertTrue(self.leaf.exists())

    def test_hardlinked_candidate_is_not_retired(self):
        second = self.base / "other-name"
        os.link(self.leaf, second)
        with self.assertRaisesRegex(StorageError, "FD/type/identity"):
            self.run_explicit()
        self.assertTrue(self.leaf.exists())
        self.assertTrue(second.exists())

    @unittest.skipUnless(getattr(os, "geteuid", lambda: -1)() == 0,
                         "real /proc census requires privileged Linux")
    def test_real_proc_census_on_temporary_fixture(self):
        result = self.run_explicit()
        self.assertEqual(result["retired_files"], [self.leaf.name])
        self.assertFalse(self.leaf.exists())
        self.assertFalse(result["physical_completion_attested"])


class ExplicitRetirementUnsupportedTests(unittest.TestCase):
    def test_non_linux_rejected_before_path_access(self):
        with patch.object(explicit.sys, "platform", "win32"):
            with self.assertRaises(explicit.base.CodeCacheRetirementUnsupported):
                explicit.retire_explicit_cpu_elf({"not": "a path"})

    def test_unconnected_dependency_gate_rejects_before_path_access(self):
        with patch.object(explicit.sys, "platform", "linux"):
            with self.assertRaisesRegex(StorageError, "active/unknown dependency protection"):
                explicit.retire_explicit_cpu_elf({"not": "a path"})
