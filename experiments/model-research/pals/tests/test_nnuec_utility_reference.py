"""Pure shadow numeric utility fixtures; no Rules bundle or lease proof."""

import copy
import hashlib
import json
import struct
import unittest

from rz_pals_model import nnuec_v2_reference as v2
from rz_pals_model import nnuec_utility_reference as reference
from test_nnuec_v2_reference import checkpoint_mapping as base_checkpoint, input_mapping as base_input, record, edge


def checkpoint_mapping(original, *, hidden=1):
    admitted = v2.NnuecV2Checkpoint.from_mapping(original)
    config = admitted.base.config
    width = 2 * config.board_width + 2 * config.context_width + config.local_width
    return {"schema": reference.SCHEMA, "semantics": reference.SEMANTICS,
            "base_weights_sha256": admitted.weights_sha256, "hidden_width": hidden, "training": {"kind": "untrained"},
            "parameters": {"hidden_weights": [0.0] * (hidden * width), "hidden_bias": [0.0] * hidden,
                           "output_weights": [0.0] * (7 * hidden), "output_bias": [0.0] * 7}}


def input_mapping(*, records=(), edges=(), target=(1,)):
    return {"schema": reference.INPUT_SCHEMA, "base_input": base_input(records=records, edges=edges), "target_ids": list(target)}


def evaluate(original, checkpoint, inputs):
    frozen = v2.NnuecV2Checkpoint.from_mapping(original)
    model = reference.UtilityCheckpoint.from_mapping(checkpoint, frozen)
    checked = reference.ReferenceInput.from_mapping(inputs, frozen.base.config)
    return reference.evaluate(frozen, model, checked)


class NnuecUtilityReferenceTests(unittest.TestCase):
    def test_selected_target_excludes_global_bias_and_reaches_all_seven_heads(self):
        original = base_checkpoint()
        original["parameters"]["kind_embeddings"][:] = [0.125, 0.25, 0.0, 0.0]
        original["base"]["parameters"]["context_projection"][0] = 1.0
        original["base"]["parameters"]["context_bias"][0] = 0.5
        checkpoint = checkpoint_mapping(original)
        checkpoint["parameters"]["hidden_weights"][:] = [0.0, 0.0, 1.0, 0.0, 2.0]
        checkpoint["parameters"]["output_weights"][:] = [float(value) for value in range(1, 8)]
        inputs = input_mapping(records=(record(1, 0), record(2, 1)), edges=(edge(1, 2),), target=(1, 2))
        result = evaluate(original, checkpoint, inputs)
        self.assertEqual(result["target_contribution"], [0.375])
        self.assertEqual(result["global_context"], [0.875])
        self.assertEqual(result["clipped_input"], [0.0, 0.0, 0.875, 0.0, 0.375])
        self.assertEqual(result["hidden"], [1.625])
        self.assertEqual(result["raw_outputs"], [1.625 * value for value in range(1, 8)])
        self.assertFalse(result["policy_actions_authorized"])
        self.assertEqual(result["target_known_mask"], [False] * 7)
        self.assertFalse(result["runtime_lease_authority"])

    def test_input_clipping_is_zero_to_one_and_never_zero_to_255(self):
        original = base_checkpoint()
        original["parameters"]["kind_embeddings"][:] = [1.0] * 4
        original["base"]["parameters"]["context_projection"][0] = 1.0
        original["base"]["parameters"]["context_bias"][0] = 2.0
        checkpoint = checkpoint_mapping(original)
        checkpoint["parameters"]["hidden_weights"][:] = [0.0, 0.0, 1.0, 0.0, 2.0]
        result = evaluate(original, checkpoint, input_mapping(records=(record(1), record(2)), target=(1, 2)))
        self.assertEqual(result["target_contribution"], [2.0])
        self.assertEqual(result["global_context"], [4.0])
        self.assertEqual(result["clipped_input"], [0.0, 0.0, 1.0, 0.0, 1.0])
        self.assertEqual(result["hidden"], [3.0])

    def test_internal_relation_contribution_is_included_in_target_exact_sum(self):
        original = base_checkpoint()
        original["parameters"]["kind_embeddings"][:] = [0.1, 0.2, 0.0, 0.0]
        original["parameters"]["pair_parent"][0] = 0.2
        original["parameters"]["pair_child"][0] = 0.3
        original["parameters"]["pair_bias"][0] = 0.05
        original["parameters"]["pair_projection"][0] = 1.0
        checkpoint = checkpoint_mapping(original)
        connected = input_mapping(records=(record(1), record(2, 1)), edges=(edge(1, 2),), target=(1, 2))
        disconnected = copy.deepcopy(connected)
        disconnected["base_input"]["context_edges"] = []
        self.assertAlmostEqual(evaluate(original, checkpoint, connected)["target_contribution"][0], 0.13, delta=1e-7)
        self.assertEqual(evaluate(original, checkpoint, disconnected)["target_contribution"], [0.0])

    def test_ids_are_lookup_only_while_the_target_and_context_provenance_are_sealed(self):
        original = base_checkpoint()
        original["parameters"]["kind_embeddings"][:] = [0.125] * 4
        original["base"]["parameters"]["context_projection"][0] = 1.0
        checkpoint = checkpoint_mapping(original)
        checkpoint["parameters"]["hidden_weights"][4] = 1.0
        checkpoint["parameters"]["output_weights"][0] = 1.0
        left = evaluate(original, checkpoint, input_mapping(records=(record(1),)))
        right = evaluate(original, checkpoint, input_mapping(records=(record(900),), target=(900,)))
        self.assertEqual(left["target_contribution"], right["target_contribution"])
        self.assertEqual(left["raw_outputs"], right["raw_outputs"])
        self.assertNotEqual(left["target_sha256"], right["target_sha256"])
        self.assertNotEqual(left["context_sha256"], right["context_sha256"])

    def test_empty_duplicate_dangling_and_crossing_targets_are_rejected(self):
        frozen = v2.NnuecV2Checkpoint.from_mapping(base_checkpoint())
        records = (record(1), record(2))
        cases = (((), (), "nonempty"), ((1, 1), (), "duplicate"), ((3,), (), "unknown"),
                 ((1,), (edge(1, 2),), "crossing"))
        for target, edges, message in cases:
            inputs = input_mapping(records=records, edges=edges, target=target)
            before = copy.deepcopy(inputs)
            with self.subTest(target=target), self.assertRaisesRegex(ValueError, message):
                reference.ReferenceInput.from_mapping(inputs, frozen.base.config)
            self.assertEqual(inputs, before)

    def test_numeric_anchor_aliases_do_not_grant_exact_bundle_completeness(self):
        original = base_checkpoint()
        checkpoint = checkpoint_mapping(original)
        # No wire field establishes that two equal numerical anchors share
        # Rules history. An unselected isolated alias is not auto-merged.
        inputs = input_mapping(records=(record(1), record(2)), target=(1,))
        result = evaluate(original, checkpoint, inputs)
        self.assertEqual(result["exact_rules_bundle_completeness"], "requires_actual_Rust_BoundFactory_evidence")
        self.assertIn("no_exact_rules_bundle", result["reference_scope"])
        self.assertFalse(result["runtime_lease_authority"])

    def test_incompatible_selected_target_anchor_or_original_first_is_rejected(self):
        frozen = v2.NnuecV2Checkpoint.from_mapping(base_checkpoint())
        for second in (record(2, anchor=(0,)), record(2, moves=((1, 9, 0),))):
            with self.subTest(second=second), self.assertRaisesRegex(ValueError, "anchor/original"):
                reference.ReferenceInput.from_mapping(input_mapping(records=(record(1), second), target=(1, 2)), frozen.base.config)

    def test_base_weight_binding_shapes_and_training_metadata_are_checked(self):
        original = base_checkpoint()
        frozen = v2.NnuecV2Checkpoint.from_mapping(original)
        checkpoint = checkpoint_mapping(original)
        for change, message in ((lambda value: value.__setitem__("base_weights_sha256", "a" * 64), "do not belong"),
                                (lambda value: value["parameters"]["output_bias"].append(0.0), "expected"),
                                (lambda value: value.__setitem__("hidden_width", 0), "hidden_width"),
                                (lambda value: value.__setitem__("training", {"kind": "bootstrap"}), "bootstrap")):
            candidate = copy.deepcopy(checkpoint)
            change(candidate)
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, message):
                reference.UtilityCheckpoint.from_mapping(candidate, frozen)
        changed_base = copy.deepcopy(original)
        changed_base["base"]["parameters"]["output_bias"][0] = 0.125
        with self.assertRaisesRegex(ValueError, "do not belong"):
            reference.UtilityCheckpoint.from_mapping(checkpoint, v2.NnuecV2Checkpoint.from_mapping(changed_base))

    def test_binary_weights_and_selected_target_seals_follow_their_own_domains(self):
        original = base_checkpoint()
        frozen = v2.NnuecV2Checkpoint.from_mapping(original)
        checkpoint = checkpoint_mapping(original)
        admitted = reference.UtilityCheckpoint.from_mapping(checkpoint, frozen)
        wire = bytearray((reference.SCHEMA + "\0" + reference.SEMANTICS + "\0" + frozen.weights_sha256 + "\0").encode("ascii"))
        wire.extend((1).to_bytes(4, "little"))
        for name in ("hidden_weights", "hidden_bias", "output_weights", "output_bias"):
            for value in checkpoint["parameters"][name]:
                wire.extend(struct.pack("<f", value))
        self.assertEqual(admitted.weights_sha256, hashlib.sha256(wire).hexdigest())
        checked = reference.ReferenceInput.from_mapping(input_mapping(records=(record(7),), target=(7,)), frozen.base.config)
        target_wire = bytearray((reference.INPUT_SCHEMA + "\0target\0" + checked.base_input.context_sha256 + "\0").encode("ascii"))
        target_wire.extend((1).to_bytes(4, "little"))
        target_wire.extend((7).to_bytes(8, "little"))
        self.assertEqual(checked.target_sha256, hashlib.sha256(target_wire).hexdigest())

    def test_json_result_remains_shadow_only_even_with_declared_learned_metadata(self):
        original = base_checkpoint()
        checkpoint = checkpoint_mapping(original)
        checkpoint["training"] = {"kind": "learned", "run_id": "synthetic-declaration-only", "steps": 1,
                                  "dataset_sha256": "a" * 64}
        inputs = input_mapping(records=(record(1),))
        raw_input = json.dumps(inputs).encode("utf-8")
        result = json.loads(reference.reference_json(json.dumps(original).encode(), json.dumps(checkpoint).encode(), raw_input))
        self.assertEqual(result["schema"], reference.OUTPUT_SCHEMA)
        self.assertFalse(result["policy_actions_authorized"])
        self.assertEqual(result["target_known_mask"], [False] * 7)
        self.assertEqual(result["input_bytes_sha256"], hashlib.sha256(raw_input).hexdigest())

    def test_numeric_input_binary_seal_includes_both_boards_local_suffix_and_weights(self):
        original = base_checkpoint()
        frozen = v2.NnuecV2Checkpoint.from_mapping(original)
        checkpoint = reference.UtilityCheckpoint.from_mapping(checkpoint_mapping(original), frozen)
        inputs = reference.ReferenceInput.from_mapping(input_mapping(records=(record(7),), target=(7,)), frozen.base.config)
        wire = bytearray((reference.INPUT_SCHEMA + "\0numeric\0" + frozen.weights_sha256 + "\0"
                          + checkpoint.weights_sha256 + "\0" + inputs.base_input.context_sha256 + "\0").encode("ascii"))
        for features in ((768, 774, 783), (769, 774, 783)):
            wire.extend((3).to_bytes(4, "little"))
            for feature in features:
                wire.extend(feature.to_bytes(2, "little"))
        wire.extend(bytes((0,)))
        wire.extend((3).to_bytes(4, "little"))
        for feature in (768, 774, 783):
            wire.extend(feature.to_bytes(2, "little"))
        wire.extend(bytes((0,)))
        wire.extend((0).to_bytes(4, "little"))
        wire.extend((1).to_bytes(4, "little"))
        wire.extend((7).to_bytes(8, "little"))
        self.assertEqual(reference.numeric_input_sha256(inputs, frozen.weights_sha256, checkpoint.weights_sha256),
                         hashlib.sha256(wire).hexdigest())
        before = reference.evaluate(frozen, checkpoint, inputs)
        changed = inputs.to_mapping()
        changed["base_input"]["local_line"]["moves"] = [{"from_square": 0, "to_square": 8, "promotion": 0}]
        later = reference.evaluate(frozen, checkpoint, reference.ReferenceInput.from_mapping(changed, frozen.base.config))
        self.assertEqual(before["target_sha256"], later["target_sha256"])
        self.assertNotEqual(before["input_sha256"], later["input_sha256"])


if __name__ == "__main__":
    unittest.main()
