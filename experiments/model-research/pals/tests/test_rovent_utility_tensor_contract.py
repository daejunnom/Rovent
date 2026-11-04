"""Neural-blocked utility lifecycle contracts, not tensor or learning evidence.

Every owner below is a metadata fixture with no model or optimizer. Synthetic
receipt pins exercise bookkeeping only and never represent original execution.
"""
import copy
import hashlib
import importlib.abc
from pathlib import Path
import subprocess
import sys
import textwrap
import unittest

from test_rovent_utility_training import membership, registration


class NeuralImportsForbidden(importlib.abc.MetaPathFinder):
    def __init__(self):
        self.attempts = []

    def find_spec(self, fullname, path=None, target=None):
        if (fullname.split(".")[0] in {"torch", "numpy", "onnx", "onnxruntime"}
                or fullname in {"rz_pals_model.model", "rz_pals_model.training",
                    "rz_pals_model.nonzero_training", "rz_pals_model.nnuec_reference",
                    "rz_pals_model.nnuec_trainable", "rz_pals_model.nnuec_utility_trainable"}):
            self.attempts.append(fullname)
            raise AssertionError("neural import forbidden in pure utility lifecycle contracts")


_guard = NeuralImportsForbidden()
sys.meta_path.insert(0, _guard)
try:
    from rz_pals_model import rovent_utility_tensor_training as tensors
    from rz_pals_model import rovent_utility_training_run as run
    from rz_pals_model import rovent_utility_training as training
    from rz_pals_model.generation_data import digest
finally:
    sys.meta_path.remove(_guard)


def pin(label):
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def preparation_fixture():
    spec = membership()
    registered = registration(spec)
    recipe = training.make_recipe(spec["optimizer_owner"])
    return recipe, registered, spec, training.prepare_training(recipe, registered, spec)


def metadata_owner(*, completed=0, ancestors=0):
    """Construct ledger-shaped test data, never a materialized tensor owner."""
    _, registered, _, preparation = preparation_fixture()
    metadata = {"initial_training": {"kind": "untrained"},
        "base_weights_sha256": registered["frozen_base"]["weights_sha256"],
        "frozen_base_mapping_sha256": pin("fixture-frozen-base-mapping")}
    owner = tensors.UtilityTensorOwner(None, preparation, metadata, None, None,
        (), {}, {"seed": 31, "cursor": 0}, (), completed_updates=completed,
        dispatched_updates=completed, ancestor_updates=ancestors)
    owner.completion_receipts = [{"sha256": pin("fixture-current-" + str(index)), "bytes": 128}
        for index in range(completed)]
    owner.ancestor_completion_sha256 = [pin("fixture-ancestor-" + str(index))
        for index in range(ancestors)]
    return owner


class UtilityTensorPureContractTests(unittest.TestCase):
    def setUp(self):
        self.guard = NeuralImportsForbidden()
        sys.meta_path.insert(0, self.guard)

    def tearDown(self):
        sys.meta_path.remove(self.guard)
        self.assertEqual(self.guard.attempts, [], "a pure contract attempted a neural import")

    def test_fresh_isolated_import_never_attempts_a_neural_dependency(self):
        # A fresh -S interpreter prevents a previously cached Torch module from
        # bypassing the import finder during discovery of a larger test suite.
        source = Path(__file__).resolve().parents[1] / "src"
        script = textwrap.dedent("""
            import importlib.abc
            import sys

            attempts = []
            forbidden = {"torch", "numpy", "onnx", "onnxruntime"}
            neural_modules = {"rz_pals_model.model", "rz_pals_model.training",
                "rz_pals_model.nonzero_training", "rz_pals_model.nnuec_reference",
                "rz_pals_model.nnuec_trainable", "rz_pals_model.nnuec_utility_trainable"}

            class BlockNeural(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    if fullname.split(".")[0] in forbidden or fullname in neural_modules:
                        attempts.append(fullname)
                        raise AssertionError("unexpected neural dependency: " + fullname)

            sys.path.insert(0, sys.argv[1])
            sys.meta_path.insert(0, BlockNeural())
            from rz_pals_model import rovent_utility_tensor_training as tensors
            from rz_pals_model import rovent_utility_training_run as run
            assert tensors.STATE_DOMAIN == "rz-rovent-utility-derived-training-state/1"
            assert tensors.PROJECTION_DOMAIN == "rz-rovent-utility-tensor-projection/1"
            assert tensors.MAX_PROJECTION_BYTES == 16 * 1024**2
            assert run.RESULT_DOMAIN == "rz-rovent-utility-learner-window-result/1"
            assert not attempts, attempts
            assert not any(name.split(".")[0] in forbidden or name in neural_modules
                for name in sys.modules)
            print("utility lifecycle import remained neural-free")
        """)
        result = subprocess.run([sys.executable, "-I", "-S", "-B", "-c", script, str(source)],
            check=False, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "utility lifecycle import remained neural-free")

    def test_default_execution_guards_reject_every_materializing_boundary(self):
        recipe, registered, spec, _ = preparation_fixture()
        calls = {
            "create": lambda: tensors.create_utility_owner(None, recipe, registered, spec,
                sampler={"seed": 31}, sampler_validator=lambda value: None),
            "payload": lambda: tensors.utility_tensor_payload(None, sampler_validator=None),
            "projection": lambda: tensors.utility_tensor_projection(None, sampler_validator=None),
            "save": lambda: tensors.save_utility_checkpoint(None, None, sampler_validator=None,
                max_file_bytes=1),
            "resume": lambda: tensors.stage_utility_resume(None, None, None, recipe, registered, spec,
                sampler_validator=None),
            "fork": lambda: tensors.stage_utility_fork(None, recipe=recipe, registration=registered,
                membership=spec, sampler={}, source_sampler_validator=None,
                sampler_validator=None, reset_seed=31),
            "export": lambda: tensors.export_utility_owner(None, None, authority=None, kind="collection"),
        }
        for boundary, call in calls.items():
            with self.subTest(boundary=boundary):
                with self.assertRaisesRegex(ValueError, "explicit utility tensor execution required"):
                    call()

    def test_execution_flag_requires_literal_true_without_truthy_coercion(self):
        for value in (False, None, 0, 1, "true", [], {}, object()):
            with self.subTest(flag=type(value).__name__):
                with self.assertRaisesRegex(ValueError, "explicit utility tensor execution required"):
                    tensors._runtime(value)

    def test_checkpoint_reserve_precedes_original_execution_without_renewing_it(self):
        spec = {"schema": run.WINDOW_DOMAIN, "max_wall_time_ns": 100, "checkpoint_reserve_ns": 20}
        self.assertEqual(run._execution_cutoffs(spec, started_ns=10, execution_until_ns=95), (95, 75))
        self.assertEqual(run._execution_cutoffs(spec, started_ns=10, execution_until_ns=1000), (110, 90))
        # A parent deadline consumed by preparation is still the same absolute
        # deadline; the child's start never earns another full execution lease.
        self.assertEqual(run._execution_cutoffs(spec, started_ns=90, execution_until_ns=100), (100, 80))

    def test_checkpoint_reserve_is_explicit_bounded_integer_and_never_defaults(self):
        spec = {"schema": run.WINDOW_DOMAIN, "max_wall_time_ns": 100, "checkpoint_reserve_ns": 20}
        for value in (0, -1, True, 100, 101, None, "20"):
            with self.subTest(reserve=value), self.assertRaises(ValueError):
                run._execution_cutoffs({**spec, "checkpoint_reserve_ns": value}, started_ns=10, execution_until_ns=95)
        with self.assertRaises(ValueError):
            run._execution_cutoffs({"schema": run.WINDOW_DOMAIN, "max_wall_time_ns": 100},
                started_ns=10, execution_until_ns=95)
        with self.assertRaises(ValueError):
            run._execution_cutoffs({**spec, "schema": "rz-rovent-utility-train-window/1"},
                started_ns=10, execution_until_ns=95)
        with self.assertRaises(ValueError):
            run._execution_cutoffs(spec, started_ns=2**64 - 1, execution_until_ns=2**64 - 1)

    def test_explicit_false_does_not_consult_the_owner_or_checkpoint_path(self):
        class Untouchable:
            def __getattribute__(self, name):
                raise AssertionError("execution guard consulted an owner or path")

            def __fspath__(self):
                raise AssertionError("execution guard inspected a checkpoint path")

        value = Untouchable()
        for call in (
            lambda: tensors.create_utility_owner(value, value, value, value,
                sampler=value, sampler_validator=None, execute=False),
            lambda: tensors.create_cohort_utility_owner(value, value, value, value,
                sampler=value, sampler_validator=None, execute=False),
            lambda: tensors.utility_tensor_payload(value, sampler_validator=None, execute=False),
            lambda: tensors.utility_tensor_projection(value, sampler_validator=None, execute=False),
            lambda: tensors.save_utility_checkpoint(value, value, sampler_validator=None,
                max_file_bytes=1, execute=False),
            lambda: tensors.stage_utility_resume(value, value, value, value, value, value,
                sampler_validator=None, execute=False),
            lambda: tensors.stage_cohort_utility_resume(value, value, value, value, value, value,
                sampler_validator=None, authority=value, historical_checkpoint=value, execute=False),
            lambda: tensors.stage_utility_fork(value, recipe=value, registration=value, membership=value,
                sampler=value, source_sampler_validator=None, sampler_validator=None, reset_seed=value,
                execute=False),
            lambda: tensors.export_utility_owner(value, value, authority=value,
                kind="collection", execute=False),
        ):
            with self.assertRaisesRegex(ValueError, "explicit utility tensor execution required"):
                call()

    def test_initial_source_declaration_stays_separate_from_zero_update_state(self):
        owner = metadata_owner()
        before = copy.deepcopy((owner.metadata, owner.preparation))
        state = tensors.derived_training_state(owner)
        self.assertEqual(state["schema"], "rz-rovent-utility-derived-training-state/1")
        self.assertEqual(state["initial_source_declaration"], {"kind": "untrained"})
        self.assertEqual(state["current_tensor_kind"], "initial_source_only")
        self.assertEqual((state["completed_updates"], state["ancestor_updates"], state["total_updates"]),
            (0, 0, 0))
        self.assertEqual(state["completion_chain_sha256"], digest(tensors.STATE_DOMAIN, []))
        self.assertFalse(state["live_training_authority_reissued"])
        self.assertEqual((owner.metadata, owner.preparation), before)
        self.assertIsNone(owner.model)
        self.assertIsNone(owner.optimizer)

    def test_nonzero_journal_metadata_never_relabels_the_initial_source_or_issues_authority(self):
        owner = metadata_owner(completed=2, ancestors=1)
        state = tensors.derived_training_state(owner)
        self.assertEqual(state["current_tensor_kind"], "learned")
        self.assertEqual(state["initial_source_declaration"], {"kind": "untrained"})
        self.assertEqual((state["completed_updates"], state["ancestor_updates"], state["total_updates"]),
            (2, 1, 3))
        chain = [*owner.ancestor_completion_sha256,
            *(entry["sha256"] for entry in owner.completion_receipts)]
        self.assertEqual(state["completion_chain_sha256"], digest(tensors.STATE_DOMAIN, chain))
        self.assertEqual(state["recipe_sha256"], owner.preparation["recipe_sha256"])
        self.assertEqual(state["registration_sha256"], owner.preparation["registration_sha256"])
        self.assertEqual(state["frozen_base_weights_sha256"], owner.metadata["base_weights_sha256"])
        self.assertEqual(state["frozen_base_mapping_sha256"], owner.metadata["frozen_base_mapping_sha256"])
        self.assertFalse(state["live_training_authority_reissued"])
        for flag in ("tensor_membership_verified", "source_authority_verified", "optimizer_state_present",
                     "checkpoint_tensor_present", "export_tensor_present", "execution_authorized"):
            self.assertFalse(owner.preparation[flag])

    def test_ancestor_only_update_history_is_not_reset_to_initial_source(self):
        owner = metadata_owner(ancestors=2)
        state = tensors.derived_training_state(owner)
        self.assertEqual(state["completed_updates"], 0)
        self.assertEqual(state["ancestor_updates"], 2)
        self.assertEqual(state["total_updates"], 2)
        self.assertEqual(state["current_tensor_kind"], "learned")
        self.assertFalse(state["live_training_authority_reissued"])

    def test_derived_state_is_a_detached_projection_and_not_a_mutable_owner_alias(self):
        owner = metadata_owner(completed=1)
        state = tensors.derived_training_state(owner)
        state["initial_source_declaration"]["kind"] = "claimed-learned"
        state["live_training_authority_reissued"] = True
        again = tensors.derived_training_state(owner)
        self.assertEqual(owner.metadata["initial_training"], {"kind": "untrained"})
        self.assertEqual(again["initial_source_declaration"], {"kind": "untrained"})
        self.assertFalse(again["live_training_authority_reissued"])
        self.assertEqual(again["completion_chain_sha256"], state["completion_chain_sha256"])

    def test_completion_chain_preserves_ancestor_then_current_receipt_order(self):
        owner = metadata_owner(completed=2, ancestors=1)
        initial = tensors.derived_training_state(owner)["completion_chain_sha256"]
        owner.completion_receipts.reverse()
        reordered = tensors.derived_training_state(owner)["completion_chain_sha256"]
        self.assertNotEqual(initial, reordered)
        owner.completion_receipts.reverse()
        owner.ancestor_completion_sha256[0] = pin("different-fixture-ancestor")
        self.assertNotEqual(initial, tensors.derived_training_state(owner)["completion_chain_sha256"])

    def test_completed_boundary_rejects_partial_or_undispatched_update_metadata(self):
        for field, value in (("pending_microbatches", 1), ("dispatched_updates", 0)):
            owner = metadata_owner(completed=1)
            setattr(owner, field, value)
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, "completed optimizer boundary"):
                    tensors.derived_training_state(owner)
        owner = metadata_owner()
        owner.completed_updates = True
        with self.assertRaises(ValueError):
            tensors.derived_training_state(owner)

    def test_missing_duplicate_or_zero_byte_completion_receipts_are_rejected(self):
        changes = (
            lambda owner: owner.completion_receipts.pop(),
            lambda owner: owner.completion_receipts[1].update(sha256=owner.completion_receipts[0]["sha256"]),
            lambda owner: owner.completion_receipts[0].update(bytes=0),
            lambda owner: owner.completion_receipts[0].update(bytes=True),
            lambda owner: owner.completion_receipts[0].update(sha256="invalid"),
        )
        for index, change in enumerate(changes):
            owner = metadata_owner(completed=2)
            change(owner)
            with self.subTest(case=index):
                with self.assertRaises(ValueError):
                    tensors.derived_training_state(owner)

    def test_ancestor_chain_cannot_be_missing_or_reuse_current_receipts(self):
        for change in (
            lambda owner: owner.ancestor_completion_sha256.clear(),
            lambda owner: owner.ancestor_completion_sha256.__setitem__(0, owner.completion_receipts[0]["sha256"]),
            lambda owner: owner.ancestor_completion_sha256.__setitem__(1, owner.ancestor_completion_sha256[0]),
        ):
            owner = metadata_owner(completed=1, ancestors=2)
            change(owner)
            with self.assertRaises(ValueError):
                tensors.derived_training_state(owner)

    def test_ancestral_cost_is_preserved_separately_and_totals_do_not_mutate_it(self):
        owner = metadata_owner(completed=1, ancestors=1)
        owner.ancestor_cost["samples"] = 8
        owner.cost["samples"] = 3
        owner.ancestor_cost["wall_time_ns"] = 1000
        owner.cost["wall_time_ns"] = 200
        before = copy.deepcopy((owner.cost, owner.ancestor_cost))
        self.assertEqual(owner.total_cost()["samples"], 11)
        self.assertEqual(owner.total_cost()["wall_time_ns"], 1200)
        tensors.derived_training_state(owner)
        self.assertEqual((owner.cost, owner.ancestor_cost), before)
        owner.cost["wall_time_ns"] = -1
        with self.assertRaises(ValueError):
            tensors.derived_training_state(owner)

    def test_plain_derived_metadata_and_duck_typed_claims_cannot_issue_a_live_utility_authority(self):
        from rz_pals_model.rovent_utility_training_authority import (
            OriginalUtilityAuthorityRequired, require_original_utility_authority)

        class ClaimedAuthority:
            def validate_learned_export(self, owner, kind):
                raise AssertionError("a duck-typed authority must not be called")

        owner = metadata_owner(completed=1)
        state = tensors.derived_training_state(owner)
        state["live_training_authority_reissued"] = True
        for claim in (None, state, owner, ClaimedAuthority()):
            with self.subTest(claim=type(claim).__name__):
                with self.assertRaisesRegex(OriginalUtilityAuthorityRequired,
                        "original connected utility parent"):
                    require_original_utility_authority(claim, allow_unleased=True)


if __name__ == "__main__":
    unittest.main()
