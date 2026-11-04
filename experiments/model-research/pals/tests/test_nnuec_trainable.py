"""CPU forward/gradient/export checks; no optimizer update or training run."""

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from rz_pals_model import nnuec_reference as reference
from rz_pals_model import nnuec_v2_reference as relations
from rz_pals_model import nnuec_trainable as trainable
from test_nnuec_reference import checkpoint_mapping, input_mapping, line
from test_nnuec_v2_reference import checkpoint_mapping as v2_checkpoint, input_mapping as v2_input, record, edge


def assert_numeric(test, actual, expected):
    for name in ("board_accumulators", "context_accumulator", "local_accumulator", "hidden",
                 "raw_outputs", "value", "wdl_logits"):
        observed = getattr(actual, name).detach().cpu().reshape(-1).tolist()
        def flatten(value):
            if isinstance(value, list):
                return [item for nested in value for item in flatten(nested)]
            return [value]
        wanted = flatten(expected[name])
        test.assertEqual(len(observed), len(wanted))
        for left, right in zip(observed, wanted, strict=True):
            test.assertLessEqual(abs(left - right), 1e-4 + 1e-3 * abs(right), (name, left, right))
    for observed, wanted in zip(actual.wdl.detach().cpu().tolist(), expected["wdl"], strict=True):
        test.assertLessEqual(abs(observed - wanted), 1e-4)


class NnuecTrainableTests(unittest.TestCase):
    def test_v1_cpu_forward_matches_independent_fp32_reference(self):
        checkpoint = checkpoint_mapping(board_width=2, line_width=2, context_width=2, hidden_width=2)
        for name, values in checkpoint["parameters"].items():
            checkpoint["parameters"][name] = [((index % 7) - 3) / 64.0 for index in range(len(values))]
        checkpoint["parameters"]["token_bias"][:] = [0.5, 0.25]
        checkpoint["parameters"]["causal_bias"][:] = [0.125, 0.125]
        inputs = input_mapping(context=(line(((0, 8, 0), (1, 9, 0))),), local=line(((0, 8, 0),)))
        admitted = reference.NnuecCheckpoint.from_mapping(checkpoint)
        expected = reference.evaluate(admitted, reference.ReferenceInput.from_mapping(inputs, admitted.config))
        model = trainable.NnuecTrainableV1.from_checkpoint(checkpoint, device="cpu")
        actual = model(inputs)
        assert_numeric(self, actual, expected)
        self.assertEqual(actual.schema, reference.OUTPUT_SCHEMA)

    def test_v2_cpu_kind_relation_forward_matches_independent_reference(self):
        checkpoint = v2_checkpoint()
        checkpoint["parameters"]["kind_embeddings"][:] = [0.1, 0.2, 0.0, 0.0]
        checkpoint["parameters"]["pair_parent"][:] = [0.2, 0.4]
        checkpoint["parameters"]["pair_child"][:] = [0.3, 0.1]
        checkpoint["parameters"]["pair_bias"][:] = [0.05, 0.0]
        checkpoint["parameters"]["pair_projection"][0] = 1.0
        checkpoint["base"]["parameters"]["hidden_weights"][2] = 1.0
        checkpoint["base"]["parameters"]["output_weights"][0] = 1.0
        inputs = v2_input(records=(record(1, 0), record(2, 1)), edges=(edge(1, 2),))
        admitted = relations.NnuecV2Checkpoint.from_mapping(checkpoint)
        expected = relations.evaluate(admitted, relations.ReferenceInput.from_mapping(inputs, admitted.base.config))
        model = trainable.NnuecTrainableV2.from_checkpoint(checkpoint, device="cpu")
        actual = model(inputs)
        assert_numeric(self, actual, expected)
        self.assertEqual(actual.context_sha256, expected["context_sha256"])
        self.assertEqual(actual.schema, relations.OUTPUT_SCHEMA)

    def test_exact_accumulator_forward_and_declared_sum_gradient_surrogate(self):
        operands = torch.tensor([[1.0], [1e9], [-1e9]], dtype=torch.float32, requires_grad=True)
        result = trainable.exact_sum_fp32(operands, bits=256, max_operand=float(1 << 70),
                                         max_result=float(1 << 79), strict_result=True)
        self.assertEqual(result.detach().tolist(), [1.0])
        result[0].backward()
        self.assertTrue(torch.equal(operands.grad, torch.ones_like(operands)))
        board_terms = torch.tensor([[1000.0]] * 8 + [[2.0 ** -12]] + [[-1000.0]] * 8, dtype=torch.float32)
        board = trainable.exact_sum_fp32(board_terms, bits=192, max_operand=1000.0, max_result=float(1 << 20))
        self.assertEqual(board.tolist(), [2.0 ** -12])

    def test_gradients_reach_actual_encoder_context_and_head_without_updating_parameters(self):
        checkpoint = checkpoint_mapping()
        for name, values in checkpoint["parameters"].items():
            checkpoint["parameters"][name] = [0.05] * len(values)
        checkpoint["parameters"]["context_projection"][0] = 0.2
        checkpoint["parameters"]["local_projection"][0] = 0.2
        model = trainable.NnuecTrainableV1.from_checkpoint(checkpoint, device="cpu")
        before = {name: value.detach().clone() for name, value in model.named_parameters()}
        actual = model(input_mapping(boards=((0,), (56,)), context=(line(((0, 8, 0),)),), local=line(((0, 8, 0),))))
        actual.raw_outputs.sum().backward()
        for name in ("board_weights", "anchor_weights", "from_weights", "causal_weights",
                     "context_projection", "local_projection", "hidden_weights", "output_weights"):
            value = dict(model.named_parameters())[name]
            self.assertIsNotNone(value.grad, name)
            self.assertTrue(bool(torch.isfinite(value.grad).all()), name)
            self.assertGreater(float(value.grad.abs().max()), 0.0, name)
        for name, value in model.named_parameters():
            self.assertTrue(torch.equal(value.detach(), before[name]), name)
        self.assertEqual(float(model.board_weights.grad[799].abs().max()), 0.0)

    def test_v2_gradients_reach_kind_and_directed_pair_parameters(self):
        checkpoint = v2_checkpoint()
        checkpoint["parameters"]["kind_embeddings"][:] = [0.1, 0.2, 0.3, 0.4]
        checkpoint["parameters"]["pair_parent"][:] = [0.1, 0.2]
        checkpoint["parameters"]["pair_child"][:] = [0.2, 0.3]
        checkpoint["parameters"]["pair_bias"][:] = [0.1, 0.2]
        checkpoint["parameters"]["pair_projection"][0] = 0.2
        checkpoint["base"]["parameters"]["hidden_weights"][2] = 0.5
        checkpoint["base"]["parameters"]["output_weights"][0] = 0.5
        model = trainable.NnuecTrainableV2.from_checkpoint(checkpoint, device="cpu")
        value = model(v2_input(records=(record(1, 0), record(2, 1)), edges=(edge(1, 2),))).value
        value.backward()
        for name in relations.PARAMETER_FIELDS:
            parameter = dict(model.named_parameters())[name]
            self.assertIsNotNone(parameter.grad, name)
            self.assertGreater(float(parameter.grad.abs().max()), 0.0, name)

    def test_explicit_parameter_names_shapes_and_no_random_initializer(self):
        state = torch.random.get_rng_state().clone()
        model = trainable.NnuecTrainableV1.from_checkpoint(checkpoint_mapping(), device="cpu")
        self.assertTrue(torch.equal(torch.random.get_rng_state(), state))
        self.assertEqual(set(dict(model.named_parameters())), set(reference.PARAMETER_FIELDS))
        self.assertEqual(model.parameter_spec()["board_weights"], {"shape": (800, 1), "dtype": "float32"})
        second = trainable.NnuecTrainableV2.from_checkpoint(v2_checkpoint(), device="cpu")
        self.assertEqual(set(dict(second.named_parameters())),
                         {"base." + name for name in reference.PARAMETER_FIELDS} | set(relations.PARAMETER_FIELDS))
        self.assertEqual(second.parameter_spec()["pair_parent"], {"shape": (2, 1), "dtype": "float32"})

    def test_export_reads_actual_tensor_bits_and_preserves_explicit_metadata(self):
        checkpoint = checkpoint_mapping()
        model = trainable.NnuecTrainableV1.from_checkpoint(checkpoint, device="cpu")
        exported = model.export_checkpoint()
        self.assertEqual(reference.NnuecCheckpoint.from_mapping(exported).weights_sha256,
                         reference.NnuecCheckpoint.from_mapping(checkpoint).weights_sha256)
        metadata = model.metadata_state()
        changed = copy.deepcopy(metadata)
        changed["output_scale"] = 2.0
        before = model.metadata_state()
        with self.assertRaisesRegex(ValueError, "cannot change"):
            model.restore_metadata_state(changed)
        self.assertEqual(model.metadata_state(), before)
        with torch.no_grad():
            model.output_bias[0].add_(0.125)
        updated = model.export_checkpoint()
        self.assertEqual(updated["parameters"]["output_bias"][0], 0.125)
        self.assertNotEqual(reference.NnuecCheckpoint.from_mapping(updated).weights_sha256,
                            reference.NnuecCheckpoint.from_mapping(exported).weights_sha256)

    def test_atomic_writer_publishes_actual_rust_readable_bytes_without_overwrite(self):
        model = trainable.NnuecTrainableV2.from_checkpoint(v2_checkpoint(), device="cpu")
        with tempfile.TemporaryDirectory(prefix="rovezero-nnuec-trainable-test-") as directory:
            destination = Path(directory) / "explicit-v2.json"
            receipt = model.write_checkpoint(destination, training={"kind": "untrained"})
            raw = destination.read_bytes()
            checked = relations.NnuecV2Checkpoint.from_mapping(json.loads(raw))
            self.assertEqual(receipt["sha256"], hashlib.sha256(raw).hexdigest())
            self.assertEqual(receipt["bytes"], len(raw))
            self.assertEqual(receipt["weights_sha256"], checked.weights_sha256)
            with self.assertRaises(FileExistsError):
                model.write_checkpoint(destination)
            self.assertEqual(destination.read_bytes(), raw)

    def test_non_fp32_parameter_cast_and_invalid_exact_operands_fail_closed(self):
        model = trainable.NnuecTrainableV1.from_checkpoint(checkpoint_mapping(), device="cpu")
        model.half()
        with self.assertRaisesRegex(ValueError, "dtype"):
            model(input_mapping())
        with self.assertRaisesRegex(ValueError, "dtype"):
            model.export_checkpoint()
        with self.assertRaisesRegex(ValueError, "operand"):
            trainable.exact_sum_fp32(torch.tensor([[1001.0]], dtype=torch.float32), bits=192,
                                     max_operand=1000.0, max_result=float(1 << 20))
        capabilities = model.capabilities()
        self.assertEqual(capabilities["exact_sum_backward"], "rounding_through_mathematical_sum_surrogate")
        self.assertTrue(capabilities["host_synchronization"])
        self.assertFalse(capabilities["optimized_backend"])

    def test_tensor_state_factory_owns_copies_without_json_lists_random_or_forward(self):
        for cls, checkpoint in ((trainable.NnuecTrainableV1, checkpoint_mapping()),
                                (trainable.NnuecTrainableV2, v2_checkpoint())):
            with self.subTest(model=cls.__name__):
                original = cls.from_checkpoint(checkpoint, device="cpu")
                state = {name: value.detach().clone() for name, value in original.state_dict().items()}
                metadata = original.metadata_state()
                before_metadata = copy.deepcopy(metadata)
                before_state = {name: value.clone() for name, value in state.items()}
                rng = torch.random.get_rng_state().clone()
                with patch.object(torch.Tensor, "tolist", side_effect=AssertionError("tensor-state factory may not expand JSON lists")), \
                     patch.object(cls, "forward", side_effect=AssertionError("tensor-state factory may not run forward")):
                    rebuilt = cls.from_tensor_state(metadata, state, device="cpu")
                self.assertTrue(torch.equal(torch.random.get_rng_state(), rng))
                self.assertEqual(rebuilt.metadata_state(), before_metadata)
                self.assertEqual(rebuilt.parameter_spec(), original.parameter_spec())
                for name, value in rebuilt.named_parameters():
                    self.assertTrue(torch.equal(value.detach(), before_state[name]), name)
                    self.assertNotEqual(value.untyped_storage().data_ptr(), state[name].untyped_storage().data_ptr(), name)
                    self.assertTrue(torch.equal(state[name], before_state[name]), name)
                first = next(iter(state))
                with torch.no_grad():
                    state[first].add_(0.125)
                self.assertTrue(torch.equal(dict(rebuilt.named_parameters())[first].detach(), before_state[first]))

    def test_tensor_state_preflight_rejects_bad_metadata_names_dtype_and_values_without_mutation(self):
        original = trainable.NnuecTrainableV1.from_checkpoint(checkpoint_mapping(), device="cpu")
        metadata = original.metadata_state()
        pristine = {name: value.detach().clone() for name, value in original.state_dict().items()}
        for change, message in ((lambda state: state.pop("output_bias"), "namespace"),
                                (lambda state: state.__setitem__("output_bias", state["output_bias"].double()), "dtype"),
                                (lambda state: state["output_bias"].__setitem__(0, 1001.0), "excessive")):
            candidate = {name: value.clone() for name, value in pristine.items()}
            change(candidate)
            rng = torch.random.get_rng_state().clone()
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, message):
                trainable.NnuecTrainableV1.from_tensor_state(metadata, candidate)
            self.assertTrue(torch.equal(torch.random.get_rng_state(), rng))
            for name, value in original.state_dict().items():
                self.assertTrue(torch.equal(value, pristine[name]), name)
        wrong_schema = copy.deepcopy(metadata)
        wrong_schema["model_schema"] = relations.SCHEMA
        with self.assertRaisesRegex(ValueError, "metadata schema"):
            trainable.NnuecTrainableV1.from_tensor_state(wrong_schema, pristine)

    def test_tensor_state_source_aliases_do_not_create_shared_destination_parameters(self):
        original = trainable.NnuecTrainableV1.from_checkpoint(checkpoint_mapping(), device="cpu")
        state = {name: value.detach().clone() for name, value in original.state_dict().items()}
        state["local_bias"] = state["context_bias"]
        rebuilt = trainable.NnuecTrainableV1.from_tensor_state(original.metadata_state(), state)
        self.assertNotEqual(rebuilt.local_bias.untyped_storage().data_ptr(), rebuilt.context_bias.untyped_storage().data_ptr())
        with torch.no_grad():
            rebuilt.local_bias.add_(0.125)
        self.assertEqual(float(rebuilt.context_bias[0]), 0.0)
        self.assertEqual(float(state["context_bias"][0]), 0.0)


if __name__ == "__main__":
    unittest.main()
