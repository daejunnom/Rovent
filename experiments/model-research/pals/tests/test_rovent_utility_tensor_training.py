"""Utility lifecycle fixtures; no forward, backward or optimizer update.

Nonzero moments and receipt pins are synthetic state-machine operands. They
never represent original action data, actual updates or learned admission.
These tests exercise CPU ownership and file/state recovery only.
"""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

import torch

from rz_pals_model import artifacts, generation_export
from rz_pals_model import nnuec_utility_reference as numeric
from rz_pals_model import nnuec_utility_trainable as trainable
from rz_pals_model import nnuec_v2_reference as relation
from rz_pals_model import rovent_utility_tensor_training as tensors
from rz_pals_model import rovent_utility_training as preparation
from rz_pals_model.nonzero_training import _state_digest, state_equal
from rz_pals_model.rovent_training_bootstrap import PinnedReadBudget
from rz_pals_model.training import _clone_state, capture_rng
from test_nnuec_utility_reference import checkpoint_mapping as utility_mapping
from test_nnuec_v2_reference import checkpoint_mapping as base_mapping


def forbidden_execution(*_args, **_kwargs):
    raise AssertionError("utility lifecycle tests cannot execute forward/backward/optimizer.step")


def sampler(*, seed=31, cursor=0):
    return {"schema": "synthetic-utility-lifecycle-sampler/1", "seed": seed, "epoch": 0, "cursor": cursor}


def validate_sampler(value):
    if (type(value) is not dict or set(value) != {"schema", "seed", "epoch", "cursor"}
            or value["schema"] != "synthetic-utility-lifecycle-sampler/1"
            or any(type(value[name]) is not int for name in ("seed", "epoch", "cursor"))
            or not 0 <= value["seed"] <= 2**32 - 1 or value["epoch"] != 0
            or not 0 <= value["cursor"] <= 8):
        raise ValueError("synthetic lifecycle sampler differs")


def lifecycle_fixture():
    base = relation.NnuecV2Checkpoint.from_mapping(base_mapping())
    checkpoint = utility_mapping(base.to_mapping(), hidden=2)
    checkpoint["parameters"]["hidden_bias"][:] = [0.125, 0.25]
    model = trainable.NnuecUtilityTrainableV1.from_checkpoint(checkpoint, frozen_base=base, device="cpu")
    owner_name = "utility-lifecycle-fixture-owner"
    config = base.base.config.to_mapping()
    width = 2 * config["board_width"] + 2 * config["context_width"] + config["local_width"]
    spec = {"hidden_width": 2, "input_width": width,
        "wire_to_parameter": {name: name for name in preparation.PARAMETER_FIELDS},
        "parameters": {name: {"checkpoint_path": "parameters." + name, "shape": list(shape),
            "dtype": "float32", "planned_requires_grad": True}
            for name, shape in zip(preparation.PARAMETER_FIELDS, ((2, width), (2,), (7, 2), (7,)), strict=True)},
        "optimizer_owner": owner_name, "frozen_components": ["nnuec_base", "p", "c"]}
    registered = {"collection_source_sha256": "10" * 32, "trainer_source_sha256": "11" * 32,
        "cost_epoch_sha256": "50" * 32, "resource_profile_sha256": "40" * 32,
        "cost_scope": preparation.COST_SCOPE, "objective": preparation.COST_OBJECTIVE,
        "frozen_base": {"schema": preparation.BASE_SCHEMA, "semantics": preparation.BASE_SEMANTICS,
            "weights_sha256": base.weights_sha256, "config": config},
        "utility": {"schema": preparation.UTILITY_SCHEMA, "semantics": preparation.UTILITY_SEMANTICS,
            "initial_weights_sha256": numeric.UtilityCheckpoint.from_mapping(checkpoint, base).weights_sha256,
            "base_weights_sha256": base.weights_sha256,
            "parameter_spec_sha256": preparation.parameter_spec_sha256(spec, config, owner_name)}}
    return base, model, preparation.make_recipe(owner_name), registered, spec


def synthetic_completed_state(owner):
    """Create an exact nonzero lifecycle fixture without taking an update."""
    owner.completed_updates = owner.dispatched_updates = 1
    owner.completion_receipts = [{"sha256": "61" * 32, "bytes": 17}]
    for parameter in owner.optimizer.param_groups[0]["params"]:
        owner.optimizer.state[parameter] = {"step": torch.tensor(1.0, dtype=torch.float32),
            "exp_avg": torch.full_like(parameter, 0.125), "exp_avg_sq": torch.full_like(parameter, 0.25)}
    schedule = copy.deepcopy(owner.scheduler.state_dict())
    schedule.update(last_epoch=1, _step_count=2, _last_lr=schedule["base_lrs"])
    owner.scheduler.load_state_dict(schedule)
    owner.sampler = sampler(cursor=1)
    owner.cost["samples"] = 8
    owner.cost["wall_time_ns"] = 123


class UtilityTensorLifecycleTests(unittest.TestCase):
    def setUp(self):
        for guard in (mock.patch.object(torch.optim.AdamW, "step", forbidden_execution),
                      mock.patch.object(torch.nn.Module, "_call_impl", forbidden_execution),
                      mock.patch.object(torch.Tensor, "backward", forbidden_execution)):
            guard.start()
            self.addCleanup(guard.stop)
        temporary = tempfile.TemporaryDirectory(prefix="rz-utility-lifecycle-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.base, self.model, self.recipe, self.registration, self.membership = lifecycle_fixture()

    def owner(self, *, model=None, registration=None, membership=None, sampler_state=None):
        return tensors.create_utility_owner(model if model is not None else self.model, self.recipe,
            registration if registration is not None else self.registration,
            membership if membership is not None else self.membership,
            sampler=sampler_state if sampler_state is not None else sampler(),
            sampler_validator=validate_sampler, execute=True)

    def payload(self, owner):
        return tensors.utility_tensor_payload(owner, sampler_validator=validate_sampler, execute=True)

    def save(self, owner, *, name="utility.pt", maximum=1024**2, read_accounting=None):
        return tensors.save_utility_checkpoint(owner, self.root / name, sampler_validator=validate_sampler,
            max_file_bytes=maximum, read_accounting=read_accounting, execute=True)

    def resume(self, pin, **kwargs):
        return tensors.stage_utility_resume(pin["path"], pin["sha256"], self.model, self.recipe,
            self.registration, self.membership, sampler_validator=validate_sampler,
            execute=True, **kwargs)

    def fork(self, source, *, seed=31):
        return tensors.stage_utility_fork(source, recipe=self.recipe, registration=self.registration,
            membership=self.membership, sampler=sampler(seed=41), source_sampler_validator=validate_sampler,
            sampler_validator=validate_sampler, reset_seed=seed, execute=True)

    def write_mutated_payload(self, original, mutate, name):
        payload = _clone_state(original)
        mutate(payload)
        payload.pop("state_sha256")
        payload["state_sha256"] = _state_digest(payload)
        target = self.root / (name + ".pt")
        torch.save(payload, target)
        return {"path": str(target), "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}

    def test_only_four_private_parameters_are_owned_and_callers_base_model_rng_stay_unchanged(self):
        before_model, before_base, rng = _clone_state(self.model.state_dict()), self.base.to_mapping(), capture_rng(())
        owner = self.owner()
        named = dict(owner.model.named_parameters())
        self.assertEqual(tuple(named), preparation.PARAMETER_FIELDS)
        self.assertEqual(tuple(owner.parameter_names), preparation.PARAMETER_FIELDS)
        self.assertEqual([id(value) for value in owner.optimizer.param_groups[0]["params"]],
                         [id(named[name]) for name in preparation.PARAMETER_FIELDS])
        self.assertTrue(all(value.requires_grad and value.grad is None for value in named.values()))
        self.assertEqual(list(owner.model.named_children()), [])
        self.assertTrue(state_equal(before_model, _clone_state(self.model.state_dict())))
        self.assertEqual(before_base, self.base.to_mapping())
        self.assertTrue(state_equal(rng, capture_rng(())))
        self.assertIsNot(owner.model, self.model)
        self.assertTrue(all(id(named[name]) != id(dict(self.model.named_parameters())[name]) for name in named))
        self.assertEqual(owner.metadata["initial_training"], {"kind": "untrained"})

    def test_different_initial_weights_are_rejected_before_an_optimizer_can_be_created(self):
        registered = copy.deepcopy(self.registration)
        registered["utility"]["initial_weights_sha256"] = "62" * 32
        before, rng = _clone_state(self.model.state_dict()), capture_rng(())
        with mock.patch.object(tensors, "_fresh_optimizer", side_effect=AssertionError("optimizer must remain uncreated")):
            with self.assertRaisesRegex(ValueError, "initialization tensors differ"):
                self.owner(registration=registered)
        self.assertTrue(state_equal(before, _clone_state(self.model.state_dict())))
        self.assertTrue(state_equal(rng, capture_rng(())))

    def test_subclass_cannot_replace_the_concrete_tensor_and_frozen_mapping_observer(self):
        class ForgedObserver(trainable.NnuecUtilityTrainableV1):
            def current_weights_sha256(self):
                return self.registration["utility"]["initial_weights_sha256"]

        forged = copy.deepcopy(self.model)
        forged.__class__ = ForgedObserver
        forged.registration = copy.deepcopy(self.registration)
        before, rng = _clone_state(self.model.state_dict()), capture_rng(())
        with mock.patch.object(tensors, "_fresh_optimizer", side_effect=AssertionError("optimizer must remain uncreated")):
            with self.assertRaisesRegex(ValueError, "concrete utility tensor owner"):
                self.owner(model=forged)
        self.assertTrue(state_equal(before, _clone_state(self.model.state_dict())))
        self.assertTrue(state_equal(rng, capture_rng(())))

    def test_frozen_foreign_dtype_and_nonfinite_parameters_are_rejected(self):
        candidates = []
        frozen = copy.deepcopy(self.model)
        frozen.hidden_weights.requires_grad_(False)
        candidates.append(frozen)
        candidates.append(copy.deepcopy(self.model).double())
        extra = copy.deepcopy(self.model)
        extra.register_parameter("foreign", torch.nn.Parameter(torch.ones(1)))
        candidates.append(extra)
        for value in (float("nan"), float("inf"), 1001.0):
            candidate = copy.deepcopy(self.model)
            with torch.no_grad():
                candidate.hidden_bias[0] = value
            candidates.append(candidate)
        for candidate in candidates:
            with self.subTest(kind=len(list(candidate.parameters()))), self.assertRaises(ValueError):
                self.owner(model=candidate)

    def test_same_storage_different_slices_are_not_separate_optimizer_arrays(self):
        candidate = copy.deepcopy(self.model)
        shared = torch.zeros(9, dtype=torch.float32)
        candidate.hidden_bias = torch.nn.Parameter(shared[:2])
        candidate.output_bias = torch.nn.Parameter(shared[2:])
        with mock.patch.object(tensors, "_fresh_optimizer", side_effect=AssertionError("optimizer must remain uncreated")):
            with self.assertRaisesRegex(ValueError, "alias"):
                self.owner(model=candidate)

    def test_existing_caller_gradient_cannot_be_hidden_by_parameter_deepcopy(self):
        candidate = copy.deepcopy(self.model)
        candidate.hidden_bias.grad = torch.ones_like(candidate.hidden_bias)
        before, rng = _clone_state(candidate.state_dict()), capture_rng(())
        with mock.patch.object(tensors, "_fresh_optimizer", side_effect=AssertionError("optimizer must remain uncreated")):
            with self.assertRaisesRegex(ValueError, "cannot discard existing gradients"):
                self.owner(model=candidate)
        self.assertTrue(state_equal(before, _clone_state(candidate.state_dict())))
        self.assertTrue(torch.equal(candidate.hidden_bias.grad, torch.ones_like(candidate.hidden_bias)))
        self.assertTrue(state_equal(rng, capture_rng(())))

    def test_actual_device_and_cuda_rng_roster_must_match_without_cuda_materialization(self):
        tensors._device_roster({"one": SimpleNamespace(device=torch.device("cpu"))}, ())
        tensors._device_roster({"one": SimpleNamespace(device=torch.device("cuda:2"))}, (2,))
        for device, roster in (("cpu", (0,)), ("cuda:2", ()), ("cuda:2", (1,)), ("cuda:2", (2, 2))):
            with self.subTest(device=device, roster=roster), self.assertRaises(ValueError):
                tensors._device_roster({"one": SimpleNamespace(device=torch.device(device))}, roster)

    def test_optimizer_membership_cannot_be_replaced_with_a_callers_parameter(self):
        owner = self.owner()
        owner.optimizer.param_groups[0]["params"][0] = self.model.hidden_weights
        with self.assertRaisesRegex(ValueError, "exact private tensor ownership"):
            self.payload(owner)

    def test_full_synthetic_five_state_roundtrip_preserves_snapshot_and_exposes_unknown_tail(self):
        owner = self.owner()
        synthetic_completed_state(owner)
        expected = self.payload(owner)
        read_budget = PinnedReadBudget(2 * 1024**2, until_ns=2, clock=lambda: 1, strict_growth_credit=True)
        with mock.patch.object(read_budget, "reserve_external", wraps=read_budget.reserve_external) as reserve:
            pin = self.save(owner, read_accounting=read_budget)
        projection_pin = pin["tensor_projection"]
        raw_projection = Path(projection_pin["path"]).read_bytes()
        projection = json.loads(raw_projection)
        self.assertEqual(Path(projection_pin["path"]), Path(pin["path"]).with_name("utility.pt.projection.json"))
        self.assertEqual(projection_pin["sha256"], hashlib.sha256(raw_projection).hexdigest())
        self.assertEqual(projection_pin["bytes"], len(raw_projection))
        self.assertLessEqual(projection_pin["bytes"], tensors.MAX_PROJECTION_BYTES)
        self.assertLessEqual(pin["bytes"] + projection_pin["bytes"], 1024**2)
        self.assertEqual([call.args[0] for call in reserve.call_args_list],
                         [pin["bytes"] + 1, projection_pin["bytes"] + 1])
        self.assertEqual(read_budget.observation(), {"charged_bytes": pin["bytes"] + projection_pin["bytes"],
            "unknown": False, "maximum_bytes": 2 * 1024**2})
        self.assertEqual(projection["schema"], "rz-rovent-utility-tensor-projection/1")
        self.assertEqual(projection["payload_state_sha256"], expected["state_sha256"])
        for family in ("model", "optimizer", "scheduler", "rng", "sampler"):
            self.assertEqual(projection[family + "_state_sha256"], _state_digest(expected[family]), family)
        for family in ("preparation", "metadata", "sampler", "cuda_devices", "parameter_names",
                       "completed_updates", "dispatched_updates", "ancestor_updates", "pending_microbatches",
                       "cost", "ancestor_cost", "completion_receipts", "ancestor_completion_sha256",
                       "parent_checkpoint_sha256", "source_cost_observation", "derived_training_state"):
            self.assertEqual(projection[family], expected[family], family)
        self.assertEqual(projection["cost"], pin["snapshot_cost"])
        self.assertFalse(projection["preparation"]["execution_authorized"])
        self.assertFalse(projection["preparation"]["source_authority_verified"])
        self.assertFalse(projection["derived_training_state"]["live_training_authority_reissued"])
        self.assertNotIn("training_target_authorized", projection)
        before_model, before_base, rng = _clone_state(self.model.state_dict()), self.base.to_mapping(), capture_rng(())
        restored = self.resume(pin)
        actual = self.payload(restored)
        for family in ("model", "optimizer", "scheduler", "rng", "sampler", "metadata", "preparation",
                       "cost", "ancestor_cost", "completion_receipts", "derived_training_state"):
            self.assertTrue(state_equal(expected[family], actual[family]), family)
        self.assertEqual(restored.source_checkpoint_sha256, pin["sha256"])
        self.assertEqual(restored.source_payload_state_sha256, expected["state_sha256"])
        self.assertEqual(restored.source_cost_observation["status"], "unknown")
        self.assertEqual(restored.source_cost_observation["scope"], "raw_snapshot_only")
        self.assertEqual(restored.physical_completion, "unknown")
        self.assertFalse(actual["derived_training_state"]["live_training_authority_reissued"])
        self.assertEqual(pin["post_snapshot_cost"]["output_bytes"],
                         pin["snapshot_cost"]["output_bytes"] + pin["bytes"] + projection_pin["bytes"])
        self.assertTrue(state_equal(before_model, _clone_state(self.model.state_dict())))
        self.assertEqual(before_base, self.base.to_mapping())
        self.assertTrue(state_equal(rng, capture_rng(())))

    def test_bad_moment_model_metadata_rng_or_sampler_rejects_before_caller_mutation(self):
        owner = self.owner()
        synthetic_completed_state(owner)
        original = self.payload(owner)
        before, rng = _clone_state(self.model.state_dict()), capture_rng(())
        def wrong_moment(payload):
            first = next(iter(payload["optimizer"]["state"].values()))
            first["exp_avg"] = first["exp_avg"].double()
        cases = (("moment", wrong_moment),
            ("metadata", lambda payload: payload["metadata"].update(hidden_width=3)),
            ("model", lambda payload: payload["model"]["hidden_bias"].fill_(float("nan"))),
            ("scheduler", lambda payload: payload["scheduler"].update(last_epoch=99)),
            ("rng", lambda payload: payload["rng"].update(torch_cpu=torch.zeros(1, dtype=torch.uint8))),
            ("sampler", lambda payload: payload["sampler"].update(cursor=99)))
        for label, mutate in cases:
            with self.subTest(case=label):
                pin = self.write_mutated_payload(original, mutate, label)
                with self.assertRaises((ValueError, RuntimeError)):
                    self.resume(pin)
                self.assertTrue(state_equal(before, _clone_state(self.model.state_dict())))
                self.assertTrue(state_equal(rng, capture_rng(())))

    def test_wrong_byte_pin_and_input_cap_fail_before_deserialization(self):
        pin = self.save(self.owner())
        with mock.patch.object(torch, "load", side_effect=AssertionError("unverified bytes cannot deserialize")):
            with self.assertRaises(ValueError):
                self.resume({**pin, "sha256": "ff" * 32})
            with self.assertRaises(ValueError):
                self.resume(pin, max_file_bytes=1)

    def test_resume_deserializes_the_bounded_fd_snapshot_even_if_original_path_changes(self):
        owner = self.owner()
        expected = self.payload(owner)
        pin = self.save(owner)
        original_load = torch.load
        def load_snapshot(stream, *args, **kwargs):
            self.assertFalse(isinstance(stream, (str, bytes, Path)))
            self.assertTrue(callable(getattr(stream, "fileno", None)))
            Path(pin["path"]).write_bytes(b"changed after verified original FD snapshot")
            return original_load(stream, *args, **kwargs)
        with mock.patch.object(torch, "load", side_effect=load_snapshot):
            restored = self.resume(pin)
        self.assertTrue(state_equal(expected["model"], self.payload(restored)["model"]))
        self.assertNotEqual(hashlib.sha256(Path(pin["path"]).read_bytes()).hexdigest(), pin["sha256"])

    def test_output_cap_preserves_partial_and_quarantines_owner(self):
        owner = self.owner()
        with self.assertRaises((ValueError, RuntimeError)):
            self.save(owner, name="too-small.pt", maximum=64)
        self.assertTrue(owner.quarantined)
        self.assertEqual(owner.physical_completion, "unknown")
        self.assertFalse((self.root / "too-small.pt").exists())
        self.assertTrue(list(self.root.glob("too-small.pt.*.partial")))
        with self.assertRaisesRegex(ValueError, "quarantined"):
            self.payload(owner)
        # A .pt that fits alone must still leave credit for its mandatory JSON
        # projection; publication failure retains the already published .pt.
        baseline = self.save(self.owner(), name="baseline.pt")
        projection_limited = self.owner()
        with self.assertRaisesRegex(ValueError, "projection exceeds original remaining output credit"):
            self.save(projection_limited, name="projection-limited.pt", maximum=baseline["bytes"] + 1)
        self.assertTrue(projection_limited.quarantined)
        self.assertEqual(projection_limited.physical_completion, "unknown")
        self.assertTrue((self.root / "projection-limited.pt").exists())
        self.assertFalse((self.root / "projection-limited.pt.projection.json").exists())

    def test_publish_or_hash_failure_retains_owned_artifacts_and_never_leaves_clean_owner(self):
        for label, target, attribute in (("publish", artifacts, "publish_noreplace"),
                                          ("hash", generation_export, "_hash_file")):
            owner = self.owner()
            with self.subTest(stage=label), mock.patch.object(target, attribute, side_effect=OSError("synthetic output failure")):
                with self.assertRaises(OSError):
                    self.save(owner, name=label + ".pt")
            self.assertTrue(owner.quarantined)
            self.assertEqual(owner.physical_completion, "unknown")
            self.assertTrue((self.root / (label + ".pt")).exists() or list(self.root.glob(label + ".pt.*.partial")))

    def test_post_publication_cost_overflow_preserves_files_and_quarantines_owner(self):
        owner = self.owner()
        owner.cost["output_bytes"] = 2**64 - 1
        with self.assertRaises(ValueError):
            self.save(owner, name="cost-overflow.pt")
        self.assertTrue((self.root / "cost-overflow.pt").is_file())
        self.assertTrue((self.root / "cost-overflow.pt.projection.json").is_file())
        self.assertTrue(owner.quarantined)
        self.assertEqual(owner.physical_completion, "unknown")
        self.assertEqual(owner.cost["output_bytes"], 2**64 - 1)

    def test_no_replace_checkpoint_publication_preserves_existing_bytes(self):
        first = self.save(self.owner())
        raw = Path(first["path"]).read_bytes()
        with self.assertRaises(FileExistsError):
            self.save(self.owner())
        self.assertEqual(Path(first["path"]).read_bytes(), raw)

    def test_fork_resets_private_state_and_rng_but_preserves_exact_parent_and_ancestry(self):
        parent = self.owner()
        synthetic_completed_state(parent)
        source = self.resume(self.save(parent))
        rng = capture_rng(())
        children = [self.fork(source, seed=31), self.fork(source, seed=31)]
        self.assertTrue(state_equal(rng, capture_rng(())))
        for child in children:
            self.assertEqual(child.parent_checkpoint_sha256, source.source_checkpoint_sha256)
            self.assertEqual(child.completed_updates, 0)
            self.assertEqual(child.dispatched_updates, 0)
            self.assertEqual(child.ancestor_updates, 1)
            self.assertEqual(child.ancestor_cost, source.total_cost())
            self.assertEqual(child.ancestor_completion_sha256, ["61" * 32])
            self.assertEqual(child.completion_receipts, [])
            self.assertEqual(child.sampler, sampler(seed=41))
            self.assertEqual(child.optimizer.state_dict()["state"], {})
            self.assertEqual(child.scheduler.state_dict()["last_epoch"], 0)
            self.assertTrue(state_equal(_clone_state(source.model.state_dict()), _clone_state(child.model.state_dict())))
            self.assertEqual(child.metadata["initial_training"], {"kind": "untrained"})
            self.assertEqual(tensors.derived_training_state(child)["current_tensor_kind"], "learned")
        self.assertTrue(state_equal(children[0].rng, children[1].rng))
        self.assertIsNot(children[0].model, children[1].model)
        self.assertIsNot(children[0].optimizer, children[1].optimizer)
        self.assertTrue(state_equal(rng, capture_rng(())))

    def test_fork_rejects_changed_model_sampler_or_ledger_after_original_pin(self):
        parent = self.owner()
        synthetic_completed_state(parent)
        pin = self.save(parent)
        def change_model(source):
            with torch.no_grad():
                source.model.output_bias[0].add_(0.125)
        for label, mutate in (("model", change_model),
                              ("sampler", lambda source: source.sampler.update(cursor=2)),
                              ("cost", lambda source: source.cost.update(samples=9)),
                              ("ancestor cost", lambda source: source.ancestor_cost.update(samples=1)),
                              ("cost observation", lambda source: source.source_cost_observation.update(scope="changed"))):
            source = self.resume(pin)
            mutate(source)
            rng = capture_rng(())
            with self.subTest(case=label), self.assertRaisesRegex(ValueError, "source changed"):
                self.fork(source)
            self.assertTrue(state_equal(rng, capture_rng(())))

    def test_fork_requires_original_pin_and_exact_non_boolean_reset_seed(self):
        with self.assertRaisesRegex(ValueError, "pinned checkpoint source"):
            self.fork(self.owner())
        source = self.resume(self.save(self.owner()))
        for seed in (True, -1, 2**32):
            rng = capture_rng(())
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                self.fork(source, seed=seed)
            self.assertTrue(state_equal(rng, capture_rng(())))

    def test_partial_gradients_or_pending_microbatches_cannot_publish(self):
        for kind in ("gradient", "pending"):
            owner = self.owner()
            if kind == "gradient":
                owner.model.hidden_bias.grad = torch.zeros_like(owner.model.hidden_bias)
            else:
                owner.pending_microbatches = 1
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                self.save(owner, name=kind + ".pt")
            self.assertFalse((self.root / (kind + ".pt")).exists())

    def test_raw_nonzero_checkpoint_never_reissues_learned_export_authority(self):
        owner = self.owner()
        synthetic_completed_state(owner)
        restored = self.resume(self.save(owner))
        restored.physical_completion = "confirmed"
        with mock.patch.object(restored.model, "write_checkpoint", side_effect=forbidden_execution):
            with self.assertRaises(ValueError):
                tensors.export_utility_owner(restored, self.root / "forbidden.json", authority=object(),
                    kind="deployment", execute=True)
        self.assertFalse((self.root / "forbidden.json").exists())
        self.assertFalse(tensors.derived_training_state(restored)["live_training_authority_reissued"])


if __name__ == "__main__":
    unittest.main()
