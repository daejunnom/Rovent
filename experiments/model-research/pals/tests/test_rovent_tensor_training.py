"""Small CPU tensor lifecycle checks; no forward/backward/optimizer update.

The moment/step fixtures below are synthetic state-machine operands, never
learned weights, real data admission or training evidence. The PC export stub
checks adapter wiring only; the NNUEC JSON writer is the actual implementation.
Numerical ONNX/CUDA acceptance remains a separate Root-owned execution.
"""
import copy
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

from rz_pals_model import artifacts
from rz_pals_model import rovent_tensor_training as tensors
from rz_pals_model.nnuec_trainable import NnuecTrainableV1, NnuecTrainableV2
from rz_pals_model.training import _clone_state, capture_rng
from rz_pals_model.nonzero_training import state_equal
from test_nnuec_reference import checkpoint_mapping
from test_nnuec_v2_reference import checkpoint_mapping as v2_checkpoint_mapping
from test_rovent_training import membership_fixture, membership_v2_fixture, recipe_fixture


class TinyConfig:
    def to_dict(self):
        return {"fixture": "tensor-lifecycle-only-not-neural-acceptance"}


class TinyComposed(torch.nn.Module):
    def __init__(self, membership):
        super().__init__()
        self.config = TinyConfig()
        self.public_encoder = torch.nn.Module()
        self.public_encoder.register_parameter("record_field_embedding", torch.nn.Parameter(torch.tensor([[0.125, 0.25]])))
        self.experts = torch.nn.ModuleDict()
        for role in ("proposer", "critic", "validator"):
            if role == "validator" and not membership["validator_present"]:
                continue
            bank = torch.nn.Module()
            bank.register_parameter("initial_latent", torch.nn.Parameter(torch.tensor([[0.5, 0.75]])))
            self.experts[role] = bank
        self.nnuec = (NnuecTrainableV2.from_checkpoint(v2_checkpoint_mapping(**membership["nnuec_config"]), device="cpu")
            if membership["schema"].endswith("/2") else
            NnuecTrainableV1.from_checkpoint(checkpoint_mapping(**membership["nnuec_config"]), device="cpu"))

    def forward(self, *_args, **_kwargs):
        raise AssertionError("lifecycle tests must not run a model forward")


def samplers():
    return {"cursor": 0, "epoch": 0, "seed": 31}


def validate_sampler(value):
    if value != samplers():
        raise ValueError("fixture sampler state mismatch")


def forbidden_execution(*_args, **_kwargs):
    raise AssertionError("forward/backward/optimizer update forbidden in lifecycle test")


def synthetic_completed_parent(owner):
    """Direct synthetic AdamW tensors, without invoking any optimizer update."""
    owner.completed.update(proposer=2, critic=2)
    owner.dispatched.update(proposer=2, critic=2)
    parameters = dict(owner.model.named_parameters())
    for identity, names in owner.parameter_names.items():
        for name in names:
            count = 0 if name.startswith("experts.validator.") else 2 if name.startswith("experts.") else 4
            owner.parameter_updates[name] = count
            if count:
                parameter = parameters[name]
                owner.optimizers[identity].state[parameter] = {"step": torch.tensor(float(count)),
                    "exp_avg": torch.zeros_like(parameter), "exp_avg_sq": torch.zeros_like(parameter)}
        schedule = copy.deepcopy(owner.scheduler_configuration[identity])
        count = 0 if identity == "v" else 4
        schedule.update(last_epoch=count, _step_count=count + 1, _last_lr=schedule["base_lrs"])
        owner.schedulers[identity].load_state_dict(schedule)


class RoventTensorLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.guard = mock.patch.multiple(torch.optim.AdamW, step=forbidden_execution)
        self.guard.start()
        self.forward_guard = mock.patch.object(torch.nn.Module, "_call_impl", forbidden_execution)
        self.forward_guard.start()
        self.backward_guard = mock.patch.object(torch.Tensor, "backward", forbidden_execution)
        self.backward_guard.start()
        self.temp = tempfile.TemporaryDirectory(prefix="rz-rovent-tensor-test-")
        self.root = Path(self.temp.name).resolve()
        self.membership = membership_fixture()
        self.recipe = recipe_fixture(self.membership)
        self.model = TinyComposed(self.membership)

    def tearDown(self):
        self.backward_guard.stop()
        self.forward_guard.stop()
        self.guard.stop()
        self.temp.cleanup()

    def owner(self):
        return tensors.create_tensor_owner(self.model, self.recipe, self.membership, stage_id="common",
            samplers=samplers(), bucket=8, sampler_validator=validate_sampler, execute=True)

    def save(self, owner, name="checkpoint.pt"):
        return tensors.save_tensor_checkpoint(owner, self.root / name, sampler_validator=validate_sampler,
            max_file_bytes=1024**2, execute=True)

    def resume(self, pin, **kwargs):
        return tensors.stage_tensor_resume(pin["path"], pin["sha256"], self.model, self.recipe, self.membership,
            sampler_validator=validate_sampler, max_file_bytes=1024**2, expected_stage_id="common", execute=True, **kwargs)

    def test_v2_registry_matches_actual_trainable_parameter_spec_and_roundtrips_its_24_tensors(self):
        self.membership = membership_v2_fixture()
        self.recipe = recipe_fixture(self.membership)
        self.model = TinyComposed(self.membership)
        actual = {"nnuec." + name: list(spec["shape"]) for name, spec in self.model.nnuec.parameter_spec().items()}
        declared = {item["name"]: item["shape"] for item in self.membership["parameters"] if item["name"].startswith("nnuec.")}
        self.assertEqual(actual, declared)
        owner = self.owner()
        self.assertEqual(set(owner.parameter_names["pc"]), set(dict(owner.model.named_parameters())) - {"experts.validator.initial_latent"})
        restored = self.resume(self.save(owner))
        self.assertTrue(state_equal(tensors.tensor_payload(owner, sampler_validator=validate_sampler, execute=True),
            tensors.tensor_payload(restored, sampler_validator=validate_sampler, execute=True)))

    def test_one_pc_owner_and_actual_role_freeze_do_not_touch_the_callers_model(self):
        original = _clone_state(self.model.state_dict())
        rng = capture_rng(())
        owner = self.owner()
        self.assertTrue(state_equal(original, _clone_state(self.model.state_dict())))
        self.assertTrue(state_equal(rng, capture_rng(())))
        self.assertEqual(set(owner.optimizers), {"pc", "v"})
        self.assertFalse(any(name.startswith("experts.validator.") for name in owner.parameter_names["pc"]))
        owner.activate_role("critic")
        for name, parameter in owner.model.named_parameters():
            self.assertEqual(parameter.requires_grad, not name.startswith(("experts.proposer.", "experts.validator.")))
        with self.assertRaisesRegex(ValueError, "unknown learner role"):
            owner.activate_role("validator")
        next(owner.model.parameters()).grad = torch.zeros_like(next(owner.model.parameters()))
        with self.assertRaisesRegex(ValueError, "pending accumulation/gradients"):
            owner.activate_role("proposer")

    def test_full_tensor_roundtrip_preserves_model_moments_scheduler_rng_and_metadata(self):
        owner = self.owner()
        synthetic_completed_parent(owner)
        pin = self.save(owner)
        expected = tensors.tensor_payload(owner, sampler_validator=validate_sampler, execute=True)
        rng = capture_rng(())
        restored = self.resume(pin)
        actual = tensors.tensor_payload(restored, sampler_validator=validate_sampler, execute=True)
        self.assertTrue(state_equal(expected, actual))
        self.assertTrue(state_equal(rng, capture_rng(())))
        self.assertIsNot(owner.model, restored.model)
        self.assertIsNot(owner.optimizers["pc"], restored.optimizers["pc"])
        self.assertEqual(restored.nnuec_metadata, owner.nnuec_metadata)
        self.assertEqual(restored.model_metadata, owner.model_metadata)
        self.assertIs(pin["training_provenance_verified"], False)

    def test_malformed_pin_moment_or_metadata_cannot_mutate_caller_or_rng(self):
        owner = self.owner()
        synthetic_completed_parent(owner)
        pin = self.save(owner)
        before, rng = _clone_state(self.model.state_dict()), capture_rng(())
        with self.assertRaisesRegex(ValueError, "byte pin mismatch"):
            self.resume({**pin, "sha256": "f" * 64})
        source = torch.load(pin["path"], map_location="cpu", weights_only=True)
        def wrong_moment(value):
            first = next(iter(value["optimizers"]["pc"]["state"].values()))
            first["exp_avg"] = first["exp_avg"].to(dtype=torch.float64)
        for label, mutate, error in (
            ("moment", wrong_moment, "coerced incoming dtype"),
            ("metadata", lambda value: value["nnuec_metadata"].update(output_scale=2.0), "metadata"),
            ("model", lambda value: value["model_metadata"]["config"].update(fixture="different"), "immutable configuration"),
        ):
            with self.subTest(case=label):
                payload = copy.deepcopy(source)
                mutate(payload)
                target = self.root / (label + ".pt")
                torch.save(payload, target)
                changed = {"path": str(target), "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}
                with self.assertRaisesRegex(ValueError, error):
                    self.resume(changed)
                self.assertTrue(state_equal(before, _clone_state(self.model.state_dict())))
                self.assertTrue(state_equal(rng, capture_rng(())))

    def test_both_teacher_forks_share_one_parent_pin_but_reset_child_state_and_keep_ancestry(self):
        parent = self.owner()
        synthetic_completed_parent(parent)
        parent.cost["samples"] = 16  # Synthetic ledger operand, not actual work.
        pin = self.save(parent)
        expected = _clone_state(parent.model.state_dict())
        rng = capture_rng(())
        forks = [tensors.stage_tensor_fork(pin["path"], pin["sha256"], self.model, self.recipe, self.membership,
            source_sampler_validator=validate_sampler, target_sampler_validator=validate_sampler,
            target_stage_id=stage, reset_seed=31, reset_samplers=samplers(), reset_bucket=16,
            max_file_bytes=1024**2, execute=True) for stage in ("sf-ft", "reckless-ft")]
        for child in forks:
            self.assertEqual(child.parent_checkpoint_sha256, pin["sha256"])
            self.assertTrue(state_equal(expected, _clone_state(child.model.state_dict())))
            self.assertEqual(child.completed, dict.fromkeys(tensors.ROLES, 0))
            self.assertEqual(child.ancestor_updates, parent.completed)
            self.assertEqual(child.ancestor_cost, parent.total_cost())
            self.assertEqual(child.bucket, 16)
            self.assertTrue(all(not value.state for value in child.optimizers.values()))
        self.assertIsNot(forks[0].model, forks[1].model)
        self.assertTrue(state_equal(rng, capture_rng(())))

    def test_slot_revalidates_before_replacing_original_owner_or_global_rng(self):
        first, second = self.owner(), self.owner()
        slot = tensors.TensorOwnerSlot(first)
        second.samplers["cursor"] = 99
        rng = capture_rng(())
        with self.assertRaisesRegex(ValueError, "sampler state mismatch"):
            slot.publish(second, sampler_validator=validate_sampler, execute=True)
        self.assertIs(slot.owner, first)
        self.assertTrue(state_equal(rng, capture_rng(())))
        second.samplers = samplers()
        previous = slot.publish(second, sampler_validator=validate_sampler, execute=True)
        self.assertIs(previous, first)
        self.assertIs(slot.owner, second)

    def test_deployment_calls_export_boundary_without_v_and_actual_nnuec_writer(self):
        owner = self.owner()
        pin = self.save(owner)
        def pc_stub(_model, metadata, directory, *, include_validator, rules_profile_json, output_guard):
            self.assertIs(include_validator, False)
            output = artifacts.output_directory(directory, output_guard=output_guard)
            graphs = []
            for role in ("public", "shared_pc"):
                raw = ("contract-only-not-onnx:" + role).encode()
                output_guard(len(raw))
                path = output / (role + ".onnx")
                path.write_bytes(raw)
                graphs.append({"role": role, "file": path.name, "sha256": hashlib.sha256(raw).hexdigest()})
            result = {"roles": ["proposer", "critic"], "validator_present": False, "graphs": graphs}
            artifacts.atomic_json(output / "export.json", result, output_guard=output_guard)
            return result
        with mock.patch.object(artifacts, "export_virtual_checkpoint", pc_stub):
            receipt = tensors.export_tensor_owner(owner, self.root / "deployment", kind="deployment",
                rules_profile_json=self.root / "not-read-by-stub.json",
                initialization_declaration={"kind": "untrained", "checkpoint_sha256": pin["sha256"]},
                sampler_validator=validate_sampler, max_output_bytes=1024**2, execute=True)
        self.assertEqual(receipt["pc"]["roles"], ["proposer", "critic"])
        self.assertTrue((self.root / "deployment" / "nnuec.json").is_file())
        self.assertEqual(receipt["nnuec"]["schema"], "rz-nnuec-fp32-v1")
        self.assertIs(receipt["actual_export_written"], True)
        self.assertIs(receipt["arena_eligible"], False)
        self.assertIs(receipt["training_provenance_verified"], False)
        self.assertTrue(owner.model.training)

    def test_learned_ancestry_cannot_be_exported_as_untrained_after_a_fork_reset(self):
        owner = self.owner()
        owner.ancestor_updates["proposer"] = 1
        pin = self.save(owner)
        with self.assertRaisesRegex(ValueError, "original native training provenance"):
            tensors.export_tensor_owner(owner, self.root / "refused", kind="deployment", rules_profile_json=None,
                initialization_declaration={"kind": "untrained", "checkpoint_sha256": pin["sha256"]},
                sampler_validator=validate_sampler, max_output_bytes=1024**2, execute=True)
        self.assertFalse((self.root / "refused").exists())


if __name__ == "__main__":
    unittest.main()
