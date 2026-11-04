"""Synthetic V2 kind/directed-relation numeric cases, never learned assets."""

import copy
import hashlib
import json
import struct
import unittest

from rz_pals_model import nnuec_reference as v1
from rz_pals_model import nnuec_v2_reference as reference
from test_nnuec_reference import checkpoint_mapping as base_checkpoint, input_mapping as base_input, line


def checkpoint_mapping(**changes):
    original = base_checkpoint(**changes)
    e, c = original["config"]["line_width"], original["config"]["context_width"]
    parameters = {name: [0.0] * length for name, length in zip(
        reference.PARAMETER_FIELDS, (4 * e, 2 * e, 2 * e, 2 * e, c * e), strict=True)}
    return {"schema": reference.SCHEMA, "semantics": reference.SEMANTICS,
            "base": original, "parameters": parameters}


def record(identifier, kind=0, *, moves=((0, 8, 0),), perspective="white", anchor=()):
    declared = line(moves, perspective=perspective, anchor=anchor)
    return {"id": identifier, "kind": kind,
            "original_first_move": declared["moves"][0] if declared["moves"] else None, **declared}


def edge(source, target, kind=0):
    return {"source": source, "target": target, "kind": kind}


def input_mapping(*, records=(), edges=(), local=None):
    result = base_input(local=local)
    result.update(schema=reference.INPUT_SCHEMA, context_lines=list(records), context_edges=list(edges))
    return result


def evaluate(checkpoint, inputs):
    model = reference.NnuecV2Checkpoint.from_mapping(checkpoint)
    checked = reference.ReferenceInput.from_mapping(inputs, model.base.config)
    return reference.evaluate(model, checked)


class NnuecV2ReferenceTests(unittest.TestCase):
    def test_four_record_kinds_are_consumed_as_typed_line_embeddings(self):
        checkpoint = checkpoint_mapping()
        checkpoint["parameters"]["kind_embeddings"][:] = [0.125, 0.25, 0.5, 0.75]
        checkpoint["base"]["parameters"]["context_projection"][0] = 1.0
        for kind, expected in enumerate((0.125, 0.25, 0.5, 0.75)):
            with self.subTest(kind=kind):
                result = evaluate(checkpoint, input_mapping(records=(record(0, kind),)))
                self.assertEqual(result["typed_line_representations"], [[expected]])
                self.assertEqual(result["context_accumulator"], [expected])

    def test_direction_and_both_relation_kinds_change_the_consumed_context(self):
        checkpoint = checkpoint_mapping()
        parameters = checkpoint["parameters"]
        parameters["kind_embeddings"][:] = [1.0, 2.0, 0.0, 0.0]
        parameters["pair_parent"][:] = [2.0, 5.0]
        parameters["pair_child"][:] = [3.0, 7.0]
        parameters["pair_bias"][:] = [0.5, 1.0]
        parameters["pair_projection"][0] = 1.0
        records = (record(10, 0), record(20, 1))
        expectations = ((edge(10, 20, 0), 8.5), (edge(20, 10, 0), 7.5), (edge(10, 20, 1), 20.0))
        for relation, expected in expectations:
            with self.subTest(relation=relation):
                result = evaluate(checkpoint, input_mapping(records=records, edges=(relation,)))
                self.assertEqual(result["pair_embeddings"], [[expected]])
                self.assertEqual(result["pair_contributions"], [[expected]])
                self.assertEqual(result["context_accumulator"], [expected])
        both = evaluate(checkpoint, input_mapping(records=records, edges=(edge(10, 20), edge(10, 20, 1))))
        self.assertEqual(both["context_accumulator"], [28.5])

    def test_relation_context_reaches_the_actual_forward_head(self):
        checkpoint = checkpoint_mapping()
        checkpoint["parameters"]["kind_embeddings"][:] = [0.1, 0.2, 0.0, 0.0]
        checkpoint["parameters"]["pair_parent"][0] = 0.2
        checkpoint["parameters"]["pair_child"][0] = 0.3
        checkpoint["parameters"]["pair_bias"][0] = 0.05
        checkpoint["parameters"]["pair_projection"][0] = 1.0
        checkpoint["base"]["parameters"]["hidden_weights"][2] = 1.0
        checkpoint["base"]["parameters"]["output_weights"][0] = 1.0
        records = (record(1, 0), record(2, 1))
        connected = evaluate(checkpoint, input_mapping(records=records, edges=(edge(1, 2),)))
        disconnected = evaluate(checkpoint, input_mapping(records=records))
        self.assertAlmostEqual(connected["value"], 0.13, delta=1e-7)
        self.assertEqual(disconnected["value"], 0.0)
        self.assertGreater(connected["hidden"][0], disconnected["hidden"][0])

    def test_lookup_ids_do_not_change_neural_numbers_but_do_change_wire_seal(self):
        checkpoint = checkpoint_mapping()
        checkpoint["parameters"]["kind_embeddings"][:] = [0.125, 0.25, 0.5, 0.75]
        checkpoint["parameters"]["pair_parent"][:] = [0.5, 0.75]
        checkpoint["parameters"]["pair_child"][:] = [0.25, 0.125]
        checkpoint["parameters"]["pair_projection"][0] = 1.0
        original = input_mapping(records=(record(1, 0), record(2, 1)), edges=(edge(1, 2),))
        renamed = input_mapping(records=(record(900, 0), record(0, 1)), edges=(edge(900, 0),))
        left, right = evaluate(checkpoint, original), evaluate(checkpoint, renamed)
        for name in ("weights_sha256", "context_accumulator", "typed_line_representations", "pair_embeddings",
                     "pair_contributions", "hidden", "raw_outputs", "value", "wdl"):
            with self.subTest(name=name):
                self.assertEqual(left[name], right[name])
        self.assertNotEqual(left["context_sha256"], right["context_sha256"])

    def test_selected_record_and_edge_order_are_sealed_separately_from_exact_sum(self):
        checkpoint = checkpoint_mapping()
        checkpoint["parameters"]["kind_embeddings"][:] = [0.125, 0.25, 0.5, 0.75]
        checkpoint["base"]["parameters"]["context_projection"][0] = 1.0
        records = (record(1, 0), record(2, 1), record(3, 2))
        edges = (edge(1, 2, 0), edge(2, 3, 1))
        original = evaluate(checkpoint, input_mapping(records=records, edges=edges))
        reordered = evaluate(checkpoint, input_mapping(records=records[::-1], edges=edges[::-1]))
        self.assertEqual(original["context_accumulator"], reordered["context_accumulator"])
        self.assertEqual(original["raw_outputs"], reordered["raw_outputs"])
        self.assertNotEqual(original["context_sha256"], reordered["context_sha256"])

    def test_dangling_duplicate_cycle_and_multiple_parent_graphs_are_rejected(self):
        model = reference.NnuecV2Checkpoint.from_mapping(checkpoint_mapping())
        records = (record(1), record(2), record(3))
        cases = (((edge(1, 4),), "dangling"),
                 ((edge(1, 2), edge(1, 2)), "duplicate directed"),
                 ((edge(1, 2, 0), edge(2, 1, 1)), "cycle"),
                 ((edge(1, 3, 0), edge(2, 3, 0)), "multiple same-kind"),
                 ((edge(1, 1),), "cycle"))
        for edges, message in cases:
            inputs = input_mapping(records=records, edges=edges)
            before = copy.deepcopy(inputs)
            with self.subTest(edges=edges), self.assertRaisesRegex(ValueError, message):
                reference.ReferenceInput.from_mapping(inputs, model.base.config)
            self.assertEqual(inputs, before)

    def test_relation_endpoints_must_keep_numeric_anchor_and_first_move(self):
        model = reference.NnuecV2Checkpoint.from_mapping(checkpoint_mapping())
        alternatives = (record(2, anchor=(0,)), record(2, perspective="black"),
                        record(2, moves=((1, 9, 0),)), record(2, moves=()))
        for second in alternatives:
            with self.subTest(second=second), self.assertRaisesRegex(ValueError, "common numeric anchor and original first move"):
                reference.ReferenceInput.from_mapping(input_mapping(records=(record(1), second), edges=(edge(1, 2),)), model.base.config)
        with self.assertRaisesRegex(ValueError, "duplicate record"):
            reference.ReferenceInput.from_mapping(input_mapping(records=(record(1), record(1))), model.base.config)

    def test_rebased_suffix_may_be_empty_or_have_another_actual_first_move(self):
        checkpoint = checkpoint_mapping()
        original = {"from_square": 0, "to_square": 8, "promotion": 0}
        left = record(1, moves=())
        right = record(2, moves=((1, 9, 0),))
        left["original_first_move"] = right["original_first_move"] = original
        result = evaluate(checkpoint, input_mapping(records=(left, right), edges=(edge(1, 2),)))
        self.assertEqual(len(result["pair_embeddings"]), 1)
        self.assertIn("not_exact_rules_anchor", result["reference_scope"])
        altered = copy.deepcopy(left)
        altered["original_first_move"]["promotion"] = 5
        model = reference.NnuecV2Checkpoint.from_mapping(checkpoint)
        with self.assertRaisesRegex(ValueError, "promotion"):
            reference.ReferenceInput.from_mapping(input_mapping(records=(altered,)), model.base.config)

    def test_v1_and_v2_assets_do_not_automatically_convert(self):
        v1_checkpoint = base_checkpoint()
        v1_input = base_input()
        with self.assertRaisesRegex(ValueError, "exact fields"):
            reference.NnuecV2Checkpoint.from_mapping(v1_checkpoint)
        with self.assertRaisesRegex(ValueError, "exact fields"):
            reference.ReferenceInput.from_mapping(v1_input, v1.NnuecConfig.from_mapping(v1_checkpoint["config"]))
        v2_checkpoint = checkpoint_mapping()
        with self.assertRaisesRegex(ValueError, "exact fields"):
            v1.NnuecCheckpoint.from_mapping(v2_checkpoint)
        altered = copy.deepcopy(v2_checkpoint)
        altered["parameters"]["pair_bias"].append(0.0)
        with self.assertRaisesRegex(ValueError, "expected"):
            reference.NnuecV2Checkpoint.from_mapping(altered)
        altered = copy.deepcopy(v2_checkpoint)
        altered["parameters"]["pair_parent"][0] = 1001.0
        with self.assertRaisesRegex(ValueError, "bounded"):
            reference.NnuecV2Checkpoint.from_mapping(altered)

    def test_v2_digest_binds_base_and_all_five_arrays_in_declared_binary_order(self):
        checkpoint = checkpoint_mapping()
        model = reference.NnuecV2Checkpoint.from_mapping(checkpoint)
        wire = bytearray(("rz-nnuec-fp32-v2\0board800-exactb192-causal-line-kind4-directedpair-originfirst-exactc256-fp32-v2\0"
                          + model.base.weights_sha256 + "\0").encode("ascii"))
        for name in ("kind_embeddings", "pair_parent", "pair_child", "pair_bias", "pair_projection"):
            for value in checkpoint["parameters"][name]:
                wire.extend(struct.pack("<f", value))
        self.assertEqual(model.weights_sha256, hashlib.sha256(wire).hexdigest())
        changed = copy.deepcopy(checkpoint)
        changed["parameters"]["kind_embeddings"][0] = 0.125
        self.assertNotEqual(reference.NnuecV2Checkpoint.from_mapping(changed).weights_sha256, model.weights_sha256)
        changed = copy.deepcopy(checkpoint)
        changed["base"]["parameters"]["output_bias"][0] = 0.125
        self.assertNotEqual(reference.NnuecV2Checkpoint.from_mapping(changed).weights_sha256, model.weights_sha256)

    def test_context_binary_seal_binds_graph_order_without_ids_as_features(self):
        checkpoint = reference.NnuecV2Checkpoint.from_mapping(checkpoint_mapping())
        inputs = input_mapping(records=(record(9, 2), record(7, 0)), edges=(edge(9, 7, 1),))
        admitted = reference.ReferenceInput.from_mapping(inputs, checkpoint.base.config)
        wire = bytearray(b"rz-nnuec-fp32-input/2\0context\0")
        wire.extend((2).to_bytes(4, "little"))
        for identifier, kind in ((9, 2), (7, 0)):
            wire.extend(identifier.to_bytes(8, "little"))
            wire.extend(bytes((kind, 0)))
            wire.extend(bytes((1, 0, 8, 0)))
            wire.extend((3).to_bytes(4, "little"))
            for feature in (768, 774, 783):
                wire.extend(feature.to_bytes(2, "little"))
            wire.extend((1).to_bytes(4, "little"))
            wire.extend(bytes((0, 8, 0)))
        wire.extend((1).to_bytes(4, "little"))
        wire.extend((9).to_bytes(8, "little"))
        wire.extend((7).to_bytes(8, "little"))
        wire.extend(bytes((1,)))
        self.assertEqual(admitted.context_sha256, hashlib.sha256(wire).hexdigest())

    def test_v2_exact_context_retains_small_bias_with_cancelling_lines(self):
        checkpoint = checkpoint_mapping(line_width=2)
        parameters = checkpoint["base"]["parameters"]
        parameters["from_weights"][:4] = [1000.0, 0.0, 0.0, 1000.0]
        parameters["causal_weights"][:2] = [1000.0, 1000.0]
        parameters["context_projection"][:] = [1000.0, -1000.0]
        parameters["context_bias"][0] = 1.0
        result = evaluate(checkpoint, input_mapping(records=(record(1), record(2, moves=((1, 9, 0),)))))
        self.assertEqual(result["context_contributions"], [[1e9], [-1e9]])
        self.assertEqual(result["context_accumulator"], [1.0])

    def test_json_entrypoint_preserves_original_byte_identity_and_new_output_domain(self):
        checkpoint = checkpoint_mapping()
        inputs = input_mapping(records=(record(1),))
        raw_checkpoint = json.dumps(checkpoint).encode("utf-8")
        raw_input = json.dumps(inputs).encode("utf-8")
        result = json.loads(reference.reference_json(raw_checkpoint, raw_input))
        self.assertEqual(result["schema"], "rz-nnuec-fp32-output/2")
        self.assertEqual(result["input_bytes_sha256"], hashlib.sha256(raw_input).hexdigest())
        self.assertEqual(result["weights_sha256"], reference.NnuecV2Checkpoint.from_mapping(checkpoint).weights_sha256)
        self.assertIn("not_exact_rules_anchor", result["reference_scope"])


if __name__ == "__main__":
    unittest.main()
