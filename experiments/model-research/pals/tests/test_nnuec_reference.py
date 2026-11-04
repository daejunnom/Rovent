"""Independent bounded NNUEC numeric fixtures, not training or Rules proof.

The small explicit parameters below are synthetic mathematical test operands.
They are never exported as learned weights or installed into an engine. Root
owns execution of these CPU-only checks and the Rust/reference cross-check.
"""

import copy
import hashlib
import json
import struct
import unittest

from rz_pals_model import nnuec_reference as reference


def checkpoint_mapping(**config_changes):
    config = {"board_width": 1, "line_width": 1, "context_width": 1, "local_width": 1,
              "hidden_width": 1, "causal_window": 3, "max_lines": 4, "max_plies": 8}
    config.update(config_changes)
    dimensions = reference.NnuecConfig.from_mapping(config)
    parameters = {name: [0.0] * length for name, length in dimensions.parameter_lengths().items()}
    return {"schema": reference.SCHEMA, "semantics": reference.SEMANTICS, "config": config,
            "training": {"kind": "untrained"}, "parameters": parameters, "output_scale": 1.0}


def line(moves=(), *, perspective="white", anchor=()):
    return {"anchor_features": [*anchor, 768, 774, 783], "perspective": perspective,
            "moves": [{"from_square": origin, "to_square": target, "promotion": promotion}
                      for origin, target, promotion in moves]}


def input_mapping(*, context=(), local=None, boards=((), ()), side="white"):
    return {"schema": reference.INPUT_SCHEMA, "board_features": [
                [*row, 768 + (side != perspective), 774, 783]
                for row, perspective in zip(boards, ("white", "black"), strict=True)],
            "side_to_move": side, "context_lines": list(context),
            "local_line": line() if local is None else local}


def evaluate(checkpoint, inputs):
    model = reference.NnuecCheckpoint.from_mapping(checkpoint)
    return reference.evaluate(model, reference.ReferenceInput.from_mapping(inputs, model.config))


class NnuecReferenceTests(unittest.TestCase):
    def test_required_configuration_and_exact_schema_are_fail_closed(self):
        config = checkpoint_mapping()["config"]
        for name in config:
            missing = dict(config)
            del missing[name]
            with self.subTest(missing=name), self.assertRaisesRegex(ValueError, "exact fields"):
                reference.NnuecConfig.from_mapping(missing)
        for name, value in (("board_width", 0), ("line_width", 257), ("hidden_width", 65),
                            ("causal_window", 4), ("max_lines", 129), ("max_plies", 257),
                            ("local_width", True)):
            with self.subTest(name=name), self.assertRaises(ValueError):
                reference.NnuecConfig.from_mapping({**config, name: value})
        checkpoint = checkpoint_mapping()
        checkpoint["semantics"] = "board800-causal-line-bcl-fp32-v1"
        with self.assertRaisesRegex(ValueError, "unregistered"):
            reference.NnuecCheckpoint.from_mapping(checkpoint)
        checkpoint["semantics"] = reference.SEMANTICS
        checkpoint["training"] = {"kind": "bootstrap"}
        with self.assertRaisesRegex(ValueError, "bootstrap"):
            reference.NnuecCheckpoint.from_mapping(checkpoint)

    def test_board_exact_feature_sum_rounds_once_and_survives_cancellation(self):
        checkpoint = checkpoint_mapping()
        parameters = checkpoint["parameters"]
        parameters["board_bias"][0] = 1.0
        parameters["board_weights"][0] = 2.0 ** -24
        parameters["board_weights"][65] = 2.0 ** -24
        result = evaluate(checkpoint, input_mapping(boards=((0, 65), ())))
        self.assertEqual(result["board_accumulators"], [[1.0 + 2.0 ** -23], [1.0]])
        # Sequential FP32 feature addition would lose both half-ULP terms.
        self.assertNotEqual(result["board_accumulators"][0][0], 1.0)
        parameters["board_bias"][0] = 0.0
        for index, value in ((0, 1000.0), (65, 0.00001), (130, -1000.0), (129, -1000.0), (66, 0.00001)):
            parameters["board_weights"][index] = value
        left = evaluate(checkpoint, input_mapping(boards=((0, 65, 130), ())))
        right = evaluate(checkpoint, input_mapping(boards=((0, 129, 66), ())))
        self.assertEqual(left["board_accumulators"][0][0], reference.f32(0.00001))
        self.assertEqual(right["board_accumulators"][0], left["board_accumulators"][0])
        # Eight allowed +1000 terms produce an 8000 partial sum. Adding a
        # 2^-12 tie there, then eight -1000 terms, must retain the small term.
        parameters["board_weights"][:] = [0.0] * len(parameters["board_weights"])
        parameters["board_weights"][:17] = [1000.0] * 8 + [2.0 ** -12] + [-1000.0] * 8
        adversarial = evaluate(checkpoint, input_mapping(boards=(tuple(range(17)), ())))
        self.assertEqual(adversarial["board_accumulators"][0], [2.0 ** -12])

    def test_exact_sum_rounding_handles_subnormals_ties_sticky_and_capacity(self):
        smallest = struct.unpack("<f", struct.pack("<I", 1))[0]
        cases = (([1.0, 2.0 ** -24], 1.0),
                 ([1.0, 2.0 ** -24, smallest], 1.0 + 2.0 ** -23),
                 ([-1.0, -(2.0 ** -24), -smallest], -(1.0 + 2.0 ** -23)),
                 ([smallest, smallest], 2.0 * smallest),
                 ([1.0, -1.0, -0.0], 0.0))
        for operands, expected in cases:
            with self.subTest(operands=operands):
                actual = reference._exact_sum_fp32(operands, 192)
                self.assertEqual(struct.pack("<f", actual), struct.pack("<f", expected))
        maximum = struct.unpack("<f", struct.pack("<I", 0x7f7fffff))[0]
        with self.assertRaisesRegex(ValueError, "signed192 accumulation overflow"):
            reference._exact_sum_fp32([maximum], 192)

    def test_context_exact_sum_preserves_bias_after_large_contribution_cancellation(self):
        checkpoint = checkpoint_mapping(line_width=2)
        parameters = checkpoint["parameters"]
        parameters["from_weights"][:4] = [1000.0, 0.0, 0.0, 1000.0]
        parameters["causal_weights"][:2] = [1000.0, 1000.0]
        parameters["context_projection"][:] = [1000.0, -1000.0]
        parameters["context_bias"][0] = 1.0
        positive = line(((0, 8, 0),))
        negative = line(((1, 9, 0),))
        result = evaluate(checkpoint, input_mapping(context=(positive, negative)))
        self.assertEqual(result["context_contributions"], [[1e9], [-1e9]])
        self.assertEqual(result["context_accumulator"], [1.0])
        swapped = evaluate(checkpoint, input_mapping(context=(negative, positive)))
        self.assertEqual(swapped["context_accumulator"], result["context_accumulator"])
        self.assertEqual(evaluate(checkpoint, input_mapping())["context_accumulator"], [1.0])

    def test_causal_sequence_order_and_context_local_separation(self):
        checkpoint = checkpoint_mapping()
        parameters = checkpoint["parameters"]
        parameters["from_weights"][0] = 1.0
        parameters["from_weights"][1] = 3.0
        parameters["causal_weights"][:] = [1.0, 2.0, 3.0]
        parameters["context_projection"][0] = 2.0
        parameters["local_projection"][0] = 3.0
        parameters["context_bias"][0] = 0.25
        parameters["local_bias"][0] = 0.5
        ordered = line(((0, 8, 0), (1, 9, 0)))
        reversed_line = line(((1, 9, 0), (0, 8, 0)))
        inputs = input_mapping(context=(ordered,), local=ordered)
        before = copy.deepcopy(inputs)
        result = evaluate(checkpoint, inputs)
        # Tokens [1,3] -> causal [1,3+2*1] -> representation 6.
        self.assertEqual(result["local_representation"], [6.0])
        self.assertEqual(result["context_contributions"], [[12.0]])
        self.assertEqual(result["context_accumulator"], [12.25])
        self.assertEqual(result["local_accumulator"], [18.5])
        reversed_result = evaluate(checkpoint, input_mapping(context=(ordered,), local=reversed_line))
        self.assertEqual(reversed_result["local_representation"], [10.0])
        self.assertEqual(reversed_result["context_accumulator"], result["context_accumulator"])
        self.assertEqual(inputs, before)

    def test_window_three_and_five_read_all_plies_without_padding_tokens(self):
        sequence = line(tuple((index, index + 8, 0) for index in range(5)))
        results = []
        for window in (3, 5):
            checkpoint = checkpoint_mapping(causal_window=window)
            checkpoint["parameters"]["from_weights"][:5] = [1.0, 2.0, 3.0, 4.0, 5.0]
            checkpoint["parameters"]["causal_weights"][:] = [1.0] * window
            results.append(evaluate(checkpoint, input_mapping(local=sequence))["local_representation"])
        self.assertEqual(results, [[31.0], [35.0]])

    def test_local_bias_first_and_context_zero_start_dot_have_declared_rounding(self):
        checkpoint = checkpoint_mapping(line_width=2)
        parameters = checkpoint["parameters"]
        parameters["from_weights"][:2] = [1.0, 1.0]
        parameters["causal_weights"][:2] = [1.0, 1.0]
        parameters["local_projection"][:] = [2.0 ** -24, 2.0 ** -24]
        parameters["context_projection"][:] = [2.0 ** -24, 2.0 ** -24]
        parameters["local_bias"][0] = 1.0
        parameters["context_bias"][0] = 1.0
        item = line(((0, 8, 0),))
        result = evaluate(checkpoint, input_mapping(context=(item,), local=item))
        self.assertEqual(result["local_accumulator"], [1.0])
        self.assertEqual(result["context_accumulator"], [1.0 + 2.0 ** -23])

    def test_black_orientation_four_promotions_and_empty_line(self):
        checkpoint = checkpoint_mapping()
        parameters = checkpoint["parameters"]
        parameters["from_weights"][48] = 0.25
        parameters["to_weights"][56] = 0.5
        parameters["promotion_weights"][:] = [0.0, 1.0, 2.0, 3.0, 4.0]
        parameters["causal_weights"][0] = 1.0
        for promotion in range(1, 5):
            white = line(((48, 56, promotion),))
            black = line(((48 ^ 56, 56 ^ 56, promotion),), perspective="black")
            left = evaluate(checkpoint, input_mapping(local=white))
            right = evaluate(checkpoint, input_mapping(local=black))
            self.assertEqual(left["local_representation"], [0.75 + promotion])
            self.assertEqual(left["local_representation"], right["local_representation"])
        parameters["causal_bias"][0] = 5.0
        self.assertEqual(evaluate(checkpoint, input_mapping())["local_representation"], [0.0])

    def test_bias_first_row_major_heads_clip_bcl_and_wdl_perspective(self):
        checkpoint = checkpoint_mapping(board_width=2, context_width=2, hidden_width=2)
        parameters = checkpoint["parameters"]
        parameters["board_weights"][:4] = [2.0, -1.0, 0.25, 0.5]
        parameters["context_bias"][:] = [0.125, 0.25]
        parameters["local_bias"][0] = 0.375
        # Input white: [1,0, .25,.5, .125,.25, .375].
        parameters["hidden_weights"][:7] = [1.0] * 7
        parameters["hidden_weights"][7:] = [-1.0] * 7
        parameters["hidden_bias"][:] = [0.5, 5.0]
        parameters["output_weights"][:2] = [2.0, -1.0]
        parameters["output_bias"][0] = 0.25
        checkpoint["output_scale"] = 2.0
        result = evaluate(checkpoint, input_mapping(boards=((0,), (1,))))
        self.assertEqual(result["hidden"], [3.0, 2.5])
        self.assertEqual(result["raw_outputs"][0], 3.75)
        self.assertEqual(result["value"], 7.5)
        self.assertEqual(result["wdl"], [reference.f32(1.0 / 3.0)] * 3)
        black = evaluate(checkpoint, input_mapping(boards=((0,), (1,)), side="black"))
        self.assertEqual(black["board_accumulators"], result["board_accumulators"])
        self.assertIn("not_rules", result["reference_scope"])
        # A non-symmetric head directly observes B_stm before B_opponent.
        parameters["hidden_weights"][:7] = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        parameters["hidden_bias"][0] = 0.0
        white = evaluate(checkpoint, input_mapping(boards=((0,), (1,))))
        black = evaluate(checkpoint, input_mapping(boards=((0,), (1,)), side="black"))
        self.assertEqual(white["hidden"][0], 1.0)
        self.assertEqual(black["hidden"][0], 0.25)

    def test_weights_digest_is_binary32_ordered_and_excludes_training_declaration(self):
        checkpoint = checkpoint_mapping()
        model = reference.NnuecCheckpoint.from_mapping(checkpoint)
        # Independent protocol construction, intentionally not a JSON hash.
        wire = bytearray(b"rz-nnuec-fp32-v1\0board800-exactb192-causal-line-exactc256-fp32-v1\0")
        for dimension in (1, 1, 1, 1, 1, 3, 4, 8):
            wire.extend(dimension.to_bytes(4, "little"))
        for name in ("board_weights", "board_bias", "anchor_weights", "anchor_bias", "from_weights",
                     "to_weights", "promotion_weights", "ply_weights", "token_bias", "causal_weights",
                     "causal_bias", "context_projection", "context_bias", "local_projection", "local_bias",
                     "hidden_weights", "hidden_bias", "output_weights", "output_bias"):
            for value in checkpoint["parameters"][name]:
                wire.extend(struct.pack("<f", value))
        wire.extend(struct.pack("<f", 1.0))
        self.assertEqual(model.weights_sha256, hashlib.sha256(wire).hexdigest())
        checkpoint["training"] = {"kind": "learned", "run_id": "synthetic-declaration-only", "steps": 1,
                                  "dataset_sha256": "a" * 64}
        self.assertEqual(reference.NnuecCheckpoint.from_mapping(checkpoint).weights_sha256, model.weights_sha256)
        checkpoint["parameters"]["output_bias"][0] = -0.0
        self.assertNotEqual(reference.NnuecCheckpoint.from_mapping(checkpoint).weights_sha256, model.weights_sha256)

    def test_bad_weights_shapes_nonfinite_and_json_duplicates_are_rejected(self):
        for change in (lambda value: value["parameters"]["board_bias"].append(0.0),
                       lambda value: value["parameters"]["board_bias"].__setitem__(0, float("nan")),
                       lambda value: value["parameters"]["board_bias"].__setitem__(0, 1001.0),
                       lambda value: value.__setitem__("output_scale", 0.0),
                       lambda value: value.__setitem__("output_scale", True)):
            checkpoint = checkpoint_mapping()
            change(checkpoint)
            with self.subTest(change=change), self.assertRaises(ValueError):
                reference.NnuecCheckpoint.from_mapping(checkpoint)
        for raw, message in ((b'{"x":1,"x":2}', "duplicate"), (b'{"x":NaN}', "nonfinite"),
                             (b'{"x":Infinity}', "nonfinite")):
            with self.subTest(raw=raw), self.assertRaisesRegex(ValueError, message):
                reference.decode_json(raw)

    def test_input_bounds_reject_before_evaluation_without_mutating_operands(self):
        model = reference.NnuecCheckpoint.from_mapping(checkpoint_mapping(max_lines=1, max_plies=1))
        for change in (lambda value: value["board_features"].__setitem__(0, [1, 1]),
                       lambda value: value["board_features"].__setitem__(0, [800]),
                       lambda value: value["board_features"].__setitem__(0, [True]),
                       lambda value: value["board_features"].__setitem__(0, [65, 0, 768, 774, 783]),
                       lambda value: value["board_features"].__setitem__(0, [769, 774, 783]),
                       lambda value: value.__setitem__("context_lines", [line(), line()]),
                       lambda value: value.__setitem__("local_line", line(((0, 1, 0), (1, 2, 0)))),
                       lambda value: value.__setitem__("local_line", line(((0, 1, 5),)))):
            inputs = input_mapping()
            change(inputs)
            before = copy.deepcopy(inputs)
            with self.subTest(change=change), self.assertRaises(ValueError):
                reference.ReferenceInput.from_mapping(inputs, model.config)
            self.assertEqual(inputs, before)

    def test_bounded_json_entrypoint_preserves_supplied_inputs_and_identifiers(self):
        checkpoint = checkpoint_mapping()
        inputs = input_mapping(local=line(((0, 8, 0),)))
        raw_checkpoint = json.dumps(checkpoint).encode("utf-8")
        raw_input = json.dumps(inputs).encode("utf-8")
        result = json.loads(reference.reference_json(raw_checkpoint, raw_input))
        self.assertEqual(result["schema"], "rz-nnuec-fp32-output/1")
        self.assertEqual(result["weights_sha256"], reference.NnuecCheckpoint.from_mapping(checkpoint).weights_sha256)
        self.assertEqual(json.loads(raw_checkpoint), checkpoint)
        self.assertEqual(json.loads(raw_input), inputs)
        self.assertEqual(len(result["raw_outputs"]), 11)
        with self.assertRaisesRegex(ValueError, "bounded immutable"):
            reference.decode_json(bytearray(raw_input))


if __name__ == "__main__":
    unittest.main()
