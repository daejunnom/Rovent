"""Synthetic submitted-input seals only; no utility model or training run."""
import copy
from dataclasses import FrozenInstanceError
import hashlib
import importlib.abc
import json
import sys
import unittest
from unittest import mock


class NeuralImportsForbidden(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"torch", "numpy", "onnx", "onnxruntime"}:
            raise AssertionError("submitted utility parser cannot import a neural framework")


_guard = NeuralImportsForbidden()
sys.meta_path.insert(0, _guard)
try:
    from rz_pals_model import rovent_utility_input as reader
    from rz_pals_model import nnuec_reference as numeric
    from rz_pals_model import nnuec_utility_reference as shadow
    from rz_pals_model import nnuec_v2_reference as v2
finally:
    sys.meta_path.remove(_guard)


def config():
    return numeric.NnuecConfig(board_width=1, line_width=1, context_width=1,
        local_width=1, hidden_width=1, causal_window=3, max_lines=4, max_plies=8)


def line(moves=()):
    return {"anchor_features": [768, 774, 783], "perspective": "white",
        "moves": [{"from_square": origin, "to_square": target, "promotion": promotion}
            for origin, target, promotion in moves]}


def record(identifier, kind=0):
    move = {"from_square": 0, "to_square": 8, "promotion": 0}
    return {"id": identifier, "kind": kind, "original_first_move": move,
        **line(((0, 8, 0),))}


def input_mapping():
    return {"schema": "rz-nnuec-utility-input/1", "base_input": {
        "schema": "rz-nnuec-fp32-input/2",
        "board_features": [[768, 774, 783], [769, 774, 783]],
        "side_to_move": "white", "context_lines": [record(0), record(1, 1)],
        "context_edges": [{"source": 0, "target": 1, "kind": 0}],
        "local_line": line()}, "target_ids": [0, 1]}


def _context_sha(value):
    """Independent numeric context wire from the documented V2 field order."""
    base = value["base_input"]
    wire = bytearray(b"rz-nnuec-fp32-input/2\0context\0")
    wire.extend(len(base["context_lines"]).to_bytes(4, "little"))
    for record_value in base["context_lines"]:
        wire.extend(record_value["id"].to_bytes(8, "little"))
        wire.extend(bytes((record_value["kind"], record_value["perspective"] == "black")))
        first = record_value["original_first_move"]
        wire.extend(bytes((first is not None,)))
        if first is not None:
            wire.extend(bytes(first[key] for key in ("from_square", "to_square", "promotion")))
        wire.extend(len(record_value["anchor_features"]).to_bytes(4, "little"))
        for feature in record_value["anchor_features"]:
            wire.extend(feature.to_bytes(2, "little"))
        wire.extend(len(record_value["moves"]).to_bytes(4, "little"))
        for move in record_value["moves"]:
            wire.extend(bytes(move[key] for key in ("from_square", "to_square", "promotion")))
    wire.extend(len(base["context_edges"]).to_bytes(4, "little"))
    for relation in base["context_edges"]:
        wire.extend(relation["source"].to_bytes(8, "little"))
        wire.extend(relation["target"].to_bytes(8, "little"))
        wire.append(relation["kind"])
    return hashlib.sha256(wire).hexdigest()


def fixture(*, value=None, text=None, targets=None):
    value = input_mapping() if value is None else value
    text = json.dumps(value, separators=(",", ":")) if text is None else text
    target_ids = list(value["target_ids"] if targets is None else targets)
    context = _context_sha(value)
    raw, weights, view = text.encode("utf-8"), "a" * 64, "1f" * 32
    input_wire = (b"rz-nnuec-submitted-utility-input/1\0" + weights.encode("ascii")
        + b"\0" + len(raw).to_bytes(8, "little") + raw)
    target_wire = (b"rz-nnuec-submitted-utility-target/1\0" + context.encode("ascii")
        + b"\0" + len(target_ids).to_bytes(4, "little")
        + b"".join(identifier.to_bytes(8, "little") for identifier in target_ids))
    target_sha = hashlib.sha256(target_wire).hexdigest()
    submitted_wire = (b"rz-nnuec-submitted-utility-submission-target/1\0"
        + bytes.fromhex(view) + target_sha.encode("ascii"))
    return {"input_json": text, "base_weights_sha256": weights,
        "context_sha256": context, "target_ids": target_ids, "view_seal_sha256": view,
        "input_sha256": hashlib.sha256(input_wire).hexdigest(), "target_sha256": target_sha,
        "submission_target_sha256": hashlib.sha256(submitted_wire).hexdigest(), "config": config()}


class SubmittedUtilityInputTests(unittest.TestCase):
    def setUp(self):
        sys.meta_path.insert(0, _guard)

    def tearDown(self):
        sys.meta_path.remove(_guard)

    def test_three_original_binary_frames_and_no_authority_are_consumed(self):
        supplied = fixture()
        result = reader.validate_submitted_utility_input(**supplied)
        self.assertEqual(result.input_bytes, supplied["input_json"].encode("utf-8"))
        self.assertEqual(result.reference_input.base_input.context_sha256, supplied["context_sha256"])
        self.assertEqual(result.target_ids, (0, 1))
        for name in ("input_sha256", "target_sha256", "submission_target_sha256"):
            self.assertEqual(getattr(result, name), supplied[name])
        observation = result.observation()
        self.assertFalse(observation["execution_authorized"])
        self.assertFalse(observation["training_target_authorized"])
        self.assertFalse(observation["original_preparation_authority_reissued"])
        self.assertFalse(observation["runtime_lease_authority"])
        self.assertEqual(observation["exact_rules_bundle_completeness"],
            "requires_original_Rust_BoundFactory_evidence")

    def test_raw_whitespace_escape_and_key_order_are_not_reserialized(self):
        value = input_mapping()
        compact = fixture(value=value)
        pretty_text = "\n" + json.dumps(value, indent=1).replace('"schema"', '"s\\u0063hema"') + "\n"
        pretty = fixture(value=value, text=pretty_text)
        left = reader.validate_submitted_utility_input(**compact)
        right = reader.validate_submitted_utility_input(**pretty)
        self.assertEqual(left.reference_input, right.reference_input)
        self.assertEqual(left.target_sha256, right.target_sha256)
        self.assertEqual(left.submission_target_sha256, right.submission_target_sha256)
        self.assertNotEqual(left.input_sha256, right.input_sha256)
        self.assertEqual(right.input_json, pretty_text)
        changed = {**pretty, "input_sha256": compact["input_sha256"]}
        with self.assertRaisesRegex(ValueError, "UTF8 input capture seal"):
            reader.validate_submitted_utility_input(**changed)

    def test_shadow_numeric_and_target_seals_are_not_capture_seals(self):
        supplied = fixture()
        parsed = shadow.ReferenceInput.from_mapping(input_mapping(), supplied["config"])
        self.assertNotEqual(parsed.target_sha256, supplied["target_sha256"])
        with self.assertRaisesRegex(ValueError, "ordered target capture seal"):
            reader.validate_submitted_utility_input(**{**supplied, "target_sha256": parsed.target_sha256})
        numeric_input = shadow.numeric_input_sha256(parsed, supplied["base_weights_sha256"], "b" * 64)
        self.assertNotEqual(numeric_input, supplied["input_sha256"])
        with self.assertRaisesRegex(ValueError, "UTF8 input capture seal"):
            reader.validate_submitted_utility_input(**{**supplied, "input_sha256": numeric_input})

    def test_every_pin_is_strict_and_each_original_binding_is_checked(self):
        supplied = fixture()
        pins = ("base_weights_sha256", "context_sha256", "view_seal_sha256",
            "input_sha256", "target_sha256", "submission_target_sha256")
        for name in pins:
            for wrong in (True, "A" * 64, "a" * 63):
                with self.subTest(name=name, wrong=wrong), self.assertRaisesRegex(ValueError, "SHA-256"):
                    reader.validate_submitted_utility_input(**{**supplied, name: wrong})
        cases = (("base_weights_sha256", "UTF8 input capture seal"),
            ("context_sha256", "numeric context seal"),
            ("view_seal_sha256", "view-bound submission target seal"),
            ("input_sha256", "UTF8 input capture seal"),
            ("target_sha256", "ordered target capture seal"),
            ("submission_target_sha256", "view-bound submission target seal"))
        for name, error in cases:
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, error):
                reader.validate_submitted_utility_input(**{**supplied, name: "c" * 64})

    def test_submission_frame_uses_raw_view_bytes_and_no_final_separator(self):
        supplied = fixture()
        wrong_frames = (
            b"rz-nnuec-submitted-utility-submission-target/1\0" + supplied["view_seal_sha256"].encode()
                + supplied["target_sha256"].encode(),
            b"rz-nnuec-submitted-utility-submission-target/1\0" + bytes.fromhex(supplied["view_seal_sha256"])
                + b"\0" + supplied["target_sha256"].encode(),
        )
        for raw in wrong_frames:
            with self.subTest(raw=raw), self.assertRaisesRegex(ValueError, "view-bound submission target seal"):
                reader.validate_submitted_utility_input(**{**supplied,
                    "submission_target_sha256": hashlib.sha256(raw).hexdigest()})

    def test_original_target_order_is_not_sorted_or_deduplicated(self):
        supplied = fixture()
        for targets in ([], [1, 0], [0, 0], [False], [-1], [64], [2**64], [0.0], "0"):
            with self.subTest(targets=targets), self.assertRaisesRegex(ValueError, "target|unsigned"):
                reader.validate_submitted_utility_input(**{**supplied, "target_ids": targets})
        value = input_mapping()
        value["target_ids"] = [1, 0]
        with self.assertRaisesRegex(ValueError, "JSON target order differs"):
            reader.validate_submitted_utility_input(**fixture(value=value, targets=(0, 1)))

    def test_unknown_target_and_crossing_edge_are_not_numeric_eligibility(self):
        value = input_mapping()
        value["target_ids"] = [0, 3]
        with self.assertRaisesRegex(ValueError, "unknown record"):
            reader.validate_submitted_utility_input(**fixture(value=value))
        value = input_mapping()
        value["target_ids"] = [0]
        with self.assertRaisesRegex(ValueError, "crossing relation"):
            reader.validate_submitted_utility_input(**fixture(value=value))

    def test_numeric_equal_anchors_do_not_claim_exact_bundle_completeness(self):
        value = input_mapping()
        value["base_input"]["context_edges"] = []
        value["target_ids"] = [0]
        observed = reader.validate_submitted_utility_input(**fixture(value=value))
        self.assertEqual(observed.target_ids, (0,))
        self.assertEqual(observed.observation()["exact_rules_bundle_completeness"],
            "requires_original_Rust_BoundFactory_evidence")
        self.assertFalse(observed.observation()["training_target_authorized"])

    def test_actual_selected_ids_and_empty_root_suffix_are_required(self):
        value = input_mapping()
        value["base_input"]["context_lines"] = [record(1), record(2, 1)]
        value["base_input"]["context_edges"] = [{"source": 1, "target": 2, "kind": 0}]
        value["target_ids"] = [1, 2]
        with self.assertRaisesRegex(ValueError, "record IDs differ"):
            reader.validate_submitted_utility_input(**fixture(value=value))
        value = input_mapping()
        value["base_input"]["local_line"] = line(((0, 8, 0),))
        with self.assertRaisesRegex(ValueError, "root local suffix must be empty"):
            reader.validate_submitted_utility_input(**fixture(value=value))

    def test_duplicate_keys_are_rejected_at_root_and_nested_levels(self):
        supplied = fixture()
        for text in (
                supplied["input_json"].replace('"target_ids":[0,1]', '"target_ids":[0,1],"target_ids":[0,1]'),
                supplied["input_json"].replace('"id":0', '"id":0,"id":0'),
                supplied["input_json"].replace('"schema":', '"s\\u0063hema":', 1).replace(
                    '"s\\u0063hema":', '"schema":"unused","s\\u0063hema":', 1)):
            with self.subTest(text=text), self.assertRaisesRegex(ValueError, "duplicate JSON keys"):
                reader.validate_submitted_utility_input(**{**supplied, "input_json": text})

    def test_nonfinite_invalid_utf8_and_malformed_json_are_rejected(self):
        supplied = fixture()
        for token in ("NaN", "Infinity", "-Infinity"):
            text = supplied["input_json"].replace('"id":0', '"id":' + token, 1)
            with self.subTest(token=token), self.assertRaisesRegex(ValueError, "nonfinite JSON"):
                reader.validate_submitted_utility_input(**{**supplied, "input_json": text})
        with self.assertRaisesRegex(ValueError, "not UTF8"):
            reader.validate_submitted_utility_input(**{**supplied, "input_json": "\ud800"})
        for text in ("{", "[]", '{"schema":1,}'):
            with self.subTest(text=text), self.assertRaisesRegex(ValueError, "bounded JSON|input object"):
                reader.validate_submitted_utility_input(**{**supplied, "input_json": text})

    def test_input_bytes_are_bounded_and_text_is_not_a_mapping_or_decoded_bytes(self):
        supplied = fixture()
        for value in (b"{}", {}, None, "", " " * (8 * 1024 * 1024 + 1)):
            with self.subTest(kind=type(value).__name__), self.assertRaisesRegex(ValueError, "JSON text|byte bound"):
                reader.validate_submitted_utility_input(**{**supplied, "input_json": value})
        raw_text = " " * (8 * 1024 * 1024 - len(supplied["input_json"])) + supplied["input_json"]
        admitted = reader.validate_submitted_utility_input(**fixture(text=raw_text))
        self.assertEqual(len(admitted.input_bytes), 8 * 1024 * 1024)

    def test_integer_features_do_not_accept_fp32_or_boolean_substitutes(self):
        supplied = fixture()
        for name, old, new in (("feature", '"anchor_features":[768', '"anchor_features":[768.0'),
                ("id", '"id":0', '"id":0.0'), ("kind", '"kind":0', '"kind":false'),
                ("target", '"target_ids":[0,1]', '"target_ids":[0.0,1]'),
                ("overflow", '"id":0', '"id":18446744073709551616')):
            text = supplied["input_json"].replace(old, new, 1)
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "integer outside"):
                reader.validate_submitted_utility_input(**{**supplied, "input_json": text})

    def test_unknown_input_fields_and_legacy_base_schema_are_not_converted(self):
        for change, error in ((lambda value: value.__setitem__("extra", True), "exact fields"),
                (lambda value: value["base_input"].__setitem__("schema", "rz-nnuec-fp32-input/1"), "unregistered V2")):
            value = input_mapping()
            change(value)
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, error):
                reader.validate_submitted_utility_input(**fixture(value=value))

    def test_config_is_explicit_and_observation_cannot_mutate_caller_state(self):
        supplied = fixture()
        before = copy.deepcopy(supplied)
        for bad in (None, supplied["config"].to_mapping(), True):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "explicit source NnuecConfig"):
                reader.validate_submitted_utility_input(**{**supplied, "config": bad})
        result = reader.validate_submitted_utility_input(**supplied)
        self.assertEqual(supplied, before)
        supplied["target_ids"][0] = 9
        observation = result.observation()
        observation["target_ids"][0] = 8
        self.assertEqual(result.target_ids, (0, 1))
        with self.assertRaises(FrozenInstanceError):
            result.input_sha256 = "c" * 64

    def test_numeric_parser_has_no_model_initialization_forward_or_file_io(self):
        forbidden = AssertionError("submitted utility input parsing cannot execute a model or open files")
        with mock.patch.object(shadow.UtilityCheckpoint, "from_mapping", side_effect=forbidden), \
                mock.patch.object(v2.NnuecV2Checkpoint, "from_mapping", side_effect=forbidden), \
                mock.patch.object(shadow, "evaluate", side_effect=forbidden), \
                mock.patch.object(v2, "evaluate", side_effect=forbidden), \
                mock.patch("builtins.open", side_effect=forbidden):
            result = reader.validate_submitted_utility_input(**fixture())
        self.assertFalse(result.observation()["execution_authorized"])


if __name__ == "__main__":
    unittest.main()
