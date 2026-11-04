"""Utility-only CPU reference/gradient/lifecycle checks, no optimizer or run.

All checkpoints and inputs below are synthetic numerical fixtures. None of
these tests constructs an actual route/witness owner or training authority.
"""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn

from rz_pals_model import nnuec_utility_reference as reference
from rz_pals_model import nnuec_utility_trainable as trainable
from rz_pals_model import nnuec_v2_reference as relation
from test_nnuec_utility_reference import checkpoint_mapping, input_mapping
from test_nnuec_v2_reference import checkpoint_mapping as base_checkpoint, record, edge


def fixture():
    base = base_checkpoint()
    base["base"]["parameters"]["board_weights"][768] = 0.25
    base["parameters"]["kind_embeddings"][:] = [0.125, 0.25, 0.0, 0.0]
    base["parameters"]["pair_parent"][0] = 0.5
    base["parameters"]["pair_child"][0] = 0.25
    base["parameters"]["pair_projection"][0] = 1.0
    base["base"]["parameters"]["context_projection"][0] = 1.0
    base["base"]["parameters"]["context_bias"][0] = 0.5
    checkpoint = checkpoint_mapping(base, hidden=2)
    checkpoint["parameters"]["hidden_weights"][:] = [((index % 5) + 1) / 16.0
        for index in range(len(checkpoint["parameters"]["hidden_weights"]))]
    checkpoint["parameters"]["hidden_bias"][:] = [0.125, 0.25]
    checkpoint["parameters"]["output_weights"][:] = [(index + 1) / 32.0 for index in range(14)]
    checkpoint["parameters"]["output_bias"][:] = [(index - 3) / 16.0 for index in range(7)]
    inputs = input_mapping(records=(record(1, 0), record(2, 1)), edges=(edge(1, 2),), target=(1, 2))
    return relation.NnuecV2Checkpoint.from_mapping(base), checkpoint, inputs


def make_model(base, checkpoint):
    return trainable.NnuecUtilityTrainableV1.from_checkpoint(checkpoint, frozen_base=base, device="cpu")


def clone_state(model):
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


class RecordingReadBudget:
    """Tiny local accounting fixture; it grants no parent/read authority."""
    def __init__(self):
        self.reservations, self.charges, self.finished, self.failed = [], [], [], []

    def reserve_external(self, maximum):
        lease = object()
        self.reservations.append((lease, maximum))
        return lease

    def charge_external(self, lease, count):
        self.charges.append((lease, count))

    def finish_external(self, lease, *, expected_bytes):
        self.finished.append((lease, expected_bytes))

    def fail_external(self, lease):
        self.failed.append(lease)


class NnuecUtilityTrainableTests(unittest.TestCase):
    def test_cpu_forward_reuses_exact_base_target_features_and_scalar_order_head(self):
        base, checkpoint, inputs = fixture()
        before = copy.deepcopy((base.to_mapping(), checkpoint, inputs))
        admitted = reference.UtilityCheckpoint.from_mapping(checkpoint, base)
        expected = reference.evaluate(base, admitted, reference.ReferenceInput.from_mapping(inputs, base.base.config))
        actual = make_model(base, checkpoint)(inputs)
        for name in ("clipped_input", "global_context", "target_contribution", "hidden", "raw_outputs"):
            self.assertEqual(getattr(actual, name).detach().cpu().tolist(), expected[name], name)
        for name in ("schema", "semantics", "base_weights_sha256", "weights_sha256", "context_sha256", "target_sha256", "input_sha256"):
            self.assertEqual(getattr(actual, name), expected[name], name)
        self.assertFalse(actual.clipped_input.requires_grad)
        self.assertFalse(actual.global_context.requires_grad)
        self.assertFalse(actual.target_contribution.requires_grad)
        self.assertFalse(actual.policy_actions_authorized)
        self.assertFalse(actual.training_target_authorized)
        self.assertFalse(actual.runtime_lease_authority)
        self.assertEqual(actual.target_known_mask, (False,) * 7)
        self.assertEqual((base.to_mapping(), checkpoint, inputs), before)

    def test_only_four_bare_wire_arrays_receive_gradients_and_base_remains_immutable(self):
        base, checkpoint, inputs = fixture()
        model = make_model(base, checkpoint)
        base_before = base.to_mapping()
        parameters_before = clone_state(model)
        output = model(inputs)
        output.raw_outputs.sum().backward()
        self.assertEqual(list(dict(model.named_parameters())), ["hidden_weights", "hidden_bias", "output_weights", "output_bias"])
        self.assertEqual(list(model.named_children()), [])
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(bool(torch.isfinite(parameter.grad).all()), name)
            self.assertGreater(float(parameter.grad.abs().max()), 0.0, name)
            self.assertTrue(torch.equal(parameter.detach(), parameters_before[name]), name)
        self.assertEqual(base.to_mapping(), base_before)
        self.assertEqual(model.parameter_spec(), {"hidden_weights": {"shape": (2, 5), "dtype": "float32"},
            "hidden_bias": {"shape": (2,), "dtype": "float32"}, "output_weights": {"shape": (7, 2), "dtype": "float32"},
            "output_bias": {"shape": (7,), "dtype": "float32"}})
        self.assertFalse(model.capabilities()["base_gradients"])
        self.assertFalse(model.capabilities()["pc_loaded"])
        self.assertFalse(model.capabilities()["optimizer_present"])

    def test_construction_and_tensor_resume_are_owned_copies_without_rng_or_forward(self):
        base, checkpoint, _inputs = fixture()
        rng = torch.random.get_rng_state().clone()
        model = make_model(base, checkpoint)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), rng))
        metadata, state = model.metadata_state(), clone_state(model)
        before_metadata, before_state = copy.deepcopy(metadata), {name: value.clone() for name, value in state.items()}
        with patch.object(torch.Tensor, "tolist", side_effect=AssertionError("resume may not expand tensor JSON")), \
                patch.object(trainable.NnuecUtilityTrainableV1, "forward", side_effect=AssertionError("resume may not run forward")):
            rebuilt = trainable.NnuecUtilityTrainableV1.from_tensor_state(metadata, state, frozen_base=base)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), rng))
        self.assertEqual(rebuilt.metadata_state(), before_metadata)
        for name, parameter in rebuilt.named_parameters():
            self.assertTrue(torch.equal(parameter.detach(), before_state[name]), name)
            self.assertNotEqual(parameter.untyped_storage().data_ptr(), state[name].untyped_storage().data_ptr(), name)
            self.assertTrue(torch.equal(state[name], before_state[name]), name)
        with torch.no_grad():
            state["output_bias"].add_(0.25)
        self.assertTrue(torch.equal(rebuilt.output_bias.detach(), before_state["output_bias"]))

    def test_metadata_binds_original_semantics_base_mapping_and_initial_declaration(self):
        base, checkpoint, _inputs = fixture()
        model = make_model(base, checkpoint)
        metadata = model.metadata_state()
        self.assertEqual(set(metadata), {"schema", "model_schema", "model_semantics", "base_schema", "base_semantics",
            "base_weights_sha256", "frozen_base_mapping_sha256", "frozen_base_config", "hidden_width", "input_width", "initial_training"})
        raw_mapping = json.dumps(base.to_mapping(), sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
        self.assertEqual(metadata["frozen_base_mapping_sha256"], hashlib.sha256(raw_mapping).hexdigest())
        self.assertEqual(metadata["initial_training"], {"kind": "untrained"})
        for key, changed in (("model_semantics", "prototype"), ("base_weights_sha256", "a" * 64),
                ("frozen_base_mapping_sha256", "b" * 64), ("input_width", True)):
            candidate = copy.deepcopy(metadata)
            candidate[key] = changed
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "metadata"):
                model.restore_metadata_state(candidate)
            self.assertEqual(model.metadata_state(), metadata)
        candidate = copy.deepcopy(metadata)
        candidate["initial_training"] = {"kind": "learned", "run_id": "declared-only", "steps": 1, "dataset_sha256": "c" * 64}
        with self.assertRaisesRegex(ValueError, "initial training declaration"):
            model.restore_metadata_state(candidate)
        self.assertEqual(model.metadata_state(), metadata)
        self.assertIsNone(model.restore_metadata_state(copy.deepcopy(metadata)))

    def test_bad_tensor_state_is_rejected_before_any_caller_model_or_rng_change(self):
        base, checkpoint, _inputs = fixture()
        model = make_model(base, checkpoint)
        metadata, pristine = model.metadata_state(), clone_state(model)
        cases = (
            ("missing", lambda s: s.pop("output_bias"), "namespace/membership"),
            ("extra", lambda s: s.update(base_weights=torch.tensor([0.0])), "namespace/membership"),
            ("dtype", lambda s: s.update(output_bias=s["output_bias"].double()), "shape/dtype/layout/device"),
            ("shape", lambda s: s.update(output_bias=s["output_bias"][:6]), "shape/dtype/layout/device"),
            ("nonfinite", lambda s: s["output_bias"].__setitem__(0, float("nan")), "finite bounded"),
            ("excessive", lambda s: s["output_bias"].__setitem__(0, 1001.0), "finite bounded"),
        )
        for name, mutate, error in cases:
            candidate = {key: value.clone() for key, value in pristine.items()}
            mutate(candidate)
            rng = torch.random.get_rng_state().clone()
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, error):
                trainable.NnuecUtilityTrainableV1.from_tensor_state(metadata, candidate, frozen_base=base)
            self.assertTrue(torch.equal(torch.random.get_rng_state(), rng))
            self.assertEqual(model.metadata_state(), metadata)
            for key, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, pristine[key]), key)

    def test_shared_storage_is_rejected_in_source_and_live_parameter_membership(self):
        base, checkpoint, _inputs = fixture()
        model = make_model(base, checkpoint)
        state = clone_state(model)
        shared = torch.tensor([0.125] * 9, dtype=torch.float32)
        state["hidden_bias"], state["output_bias"] = shared[:2], shared[2:]
        with self.assertRaisesRegex(ValueError, "must not alias storage"):
            trainable.NnuecUtilityTrainableV1.from_tensor_state(model.metadata_state(), state, frozen_base=base)
        model.hidden_bias, model.output_bias = nn.Parameter(shared[:2]), nn.Parameter(shared[2:])
        with self.assertRaisesRegex(ValueError, "must not alias storage"):
            model.export_checkpoint()

    def test_export_reads_actual_current_bits_and_does_not_relabel_initial_training(self):
        base, checkpoint, _inputs = fixture()
        model = make_model(base, checkpoint)
        original = reference.UtilityCheckpoint.from_mapping(checkpoint, base)
        self.assertEqual(reference.UtilityCheckpoint.from_mapping(model.export_checkpoint(), base).weights_sha256, original.weights_sha256)
        source_initial_pin = original.weights_sha256
        self.assertEqual(model.current_weights_sha256(), source_initial_pin)
        expected = copy.deepcopy(checkpoint)
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                parameter.reshape(-1)[0].add_(0.125)
                expected["parameters"][name][0] += 0.125
        changed = model.export_checkpoint()
        expected_current = reference.UtilityCheckpoint.from_mapping(expected, base).weights_sha256
        self.assertEqual(reference.UtilityCheckpoint.from_mapping(changed, base).weights_sha256, expected_current)
        self.assertEqual(model.current_weights_sha256(), expected_current)
        self.assertNotEqual(model.current_weights_sha256(), source_initial_pin)
        mapping_bytes = json.dumps(changed, sort_keys=True, ensure_ascii=False, allow_nan=False,
            separators=(",", ":")).encode("utf-8")
        self.assertNotEqual(model.current_weights_sha256(), hashlib.sha256(mapping_bytes).hexdigest())
        declaration = {"kind": "learned", "run_id": "caller-declaration-not-proof", "steps": 1, "dataset_sha256": "a" * 64}
        exported = model.export_checkpoint(training=declaration)
        self.assertEqual(exported["training"], declaration)
        self.assertEqual(model.metadata_state()["initial_training"], {"kind": "untrained"})
        self.assertFalse(model.capabilities()["source_authority_verified"])
        self.assertFalse(model.capabilities()["training_target_authorized"])

    def test_writer_preserves_independent_utility_wire_and_never_overwrites(self):
        base, checkpoint, _inputs = fixture()
        model = make_model(base, checkpoint)
        rng = torch.random.get_rng_state().clone()
        with tempfile.TemporaryDirectory(prefix="rovezero-utility-trainable-test-") as directory:
            reads = RecordingReadBudget()
            for maximum in (1, 0, True, reference.MAX_CHECKPOINT_BYTES + 1):
                rejected = Path(directory) / ("cap-" + str(maximum) + ".json")
                with self.subTest(maximum=maximum), self.assertRaisesRegex(ValueError, "utility output byte cap"):
                    model.write_checkpoint(rejected, max_output_bytes=maximum, read_accounting=reads)
                self.assertFalse(rejected.exists())
                self.assertFalse(rejected.with_name(rejected.name + ".partial").exists())
            self.assertEqual((reads.reservations, reads.charges, reads.finished, reads.failed), ([], [], [], []))
            destination = Path(directory) / "utility.json"
            receipt = model.write_checkpoint(destination, max_output_bytes=reference.MAX_CHECKPOINT_BYTES, read_accounting=reads)
            raw = destination.read_bytes()
            decoded = reference.UtilityCheckpoint.from_mapping(json.loads(raw), base)
            self.assertEqual(receipt["sha256"], hashlib.sha256(raw).hexdigest())
            self.assertEqual(receipt["bytes"], len(raw))
            self.assertEqual(receipt["weights_sha256"], decoded.weights_sha256)
            self.assertEqual(receipt["base_weights_sha256"], base.weights_sha256)
            self.assertEqual(json.loads(raw)["schema"], "rz-nnuec-utility-fp32-v1")
            self.assertEqual(len(reads.reservations), 1)
            lease, reserved = reads.reservations[0]
            self.assertEqual(reserved, len(raw) + 1)
            self.assertEqual(sum(count for seen, count in reads.charges if seen is lease), len(raw))
            self.assertTrue(all(seen is lease for seen, _ in reads.charges))
            self.assertEqual(reads.finished, [(lease, len(raw))])
            self.assertEqual(reads.failed, [])
            with self.assertRaises(FileExistsError):
                model.write_checkpoint(destination)
            self.assertEqual(destination.read_bytes(), raw)
            preserved = Path(directory) / "preserved-failed-pin.json"
            with patch("rz_pals_model.generation_export._hash_file", return_value=("a" * 64, len(raw))):
                with self.assertRaisesRegex(ValueError, "original byte pin changed"):
                    model.write_checkpoint(preserved)
            self.assertEqual(preserved.read_bytes(), raw)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), rng))

    def test_unregistered_semantics_base_and_casts_are_rejected_without_conversion(self):
        base, checkpoint, inputs = fixture()
        bad = copy.deepcopy(checkpoint)
        bad["semantics"] = "prototype-utility"
        with self.assertRaisesRegex(ValueError, "schema or semantics"):
            make_model(base, bad)
        changed = base.to_mapping()
        changed["base"]["parameters"]["output_bias"][0] = 0.125
        with self.assertRaisesRegex(ValueError, "do not belong"):
            make_model(relation.NnuecV2Checkpoint.from_mapping(changed), checkpoint)
        with self.assertRaisesRegex(ValueError, "validated immutable V2 base"):
            trainable.NnuecUtilityTrainableV1.from_checkpoint(checkpoint, frozen_base=base.to_mapping())
        model = make_model(base, checkpoint).half()
        with self.assertRaisesRegex(ValueError, "shape/dtype/layout/device"):
            model(inputs)

    def test_crossing_or_unknown_target_still_uses_the_original_numeric_boundary(self):
        base, checkpoint, inputs = fixture()
        for ids, error in (([1], "crossing relation"), ([99], "unknown record")):
            candidate = copy.deepcopy(inputs)
            candidate["target_ids"] = ids
            before = copy.deepcopy(candidate)
            with self.subTest(ids=ids), self.assertRaisesRegex(ValueError, error):
                make_model(base, checkpoint)(candidate)
            self.assertEqual(candidate, before)


if __name__ == "__main__":
    unittest.main()
