"""R framing contract only; no model, Rules substitute or trainer execution."""
import copy
import hashlib
import importlib.abc
import json
import struct
import sys
import unittest


class NeuralImportsForbidden(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"torch", "numpy", "onnx", "onnxruntime"}:
            raise AssertionError("R metadata framing cannot import a neural framework")


_guard = NeuralImportsForbidden()
sys.meta_path.insert(0, _guard)
try:
    from rz_pals_model import rovent_r_readout as reader
finally:
    sys.meta_path.remove(_guard)


def framed(raw):
    return hashlib.sha256(reader.FLOAT_BITS_DOMAIN.encode() + b"\0" + raw).hexdigest()


def input_fixture(text='{"schema":"fixture-numeric-only","x":1e-8,"z":1.0}'):
    bound = {"model_semantics": "fixture-numeric-only", "parameter_sha256": "a" * 64,
        "collection_model_epoch": 3, "collection_context_epoch": 5}
    raw = text.encode()
    meaning = {**bound, "reference_input": json.loads(text)}
    return {"reference_input_json": text, "reference_input_bytes": len(raw),
        "reference_input_raw_sha256": hashlib.sha256(raw).hexdigest(),
        "native_input_sha256": hashlib.sha256(reader.CAPTURE_INPUT_DOMAIN.encode() + b"\0" + raw).hexdigest(),
        "semantic_input_sha256": reader.r_readout_f32_digest(reader.INPUT_DOMAIN, meaning)}, bound


class RReadoutFramingTests(unittest.TestCase):
    def setUp(self):
        sys.meta_path.insert(0, _guard)

    def tearDown(self):
        sys.meta_path.remove(_guard)

    def test_independent_binary_vectors_preserve_type_signed_zero_and_utf8_order(self):
        cases = [(None, b"\0"), (False, b"\1"), (True, b"\2"),
            (1, b"\3" + (1).to_bytes(8, "little")),
            (-1, b"\4" + (1).to_bytes(8, "little")),
            (1.0, b"\5" + struct.pack("<f", 1.0)),
            (-0.0, b"\5" + bytes.fromhex("00000080")),
            ("한", b"\6" + (3).to_bytes(8, "little") + "한".encode()),
            ([None, True], b"\7" + (2).to_bytes(8, "little") + b"\0\2")]
        for value, raw in cases:
            with self.subTest(value=value):
                self.assertEqual(reader.r_readout_f32_digest(reader.FLOAT_BITS_DOMAIN, value), framed(raw))
        raw = b"\10" + (2).to_bytes(8, "little")
        raw += (1).to_bytes(8, "little") + b"a\3" + (1).to_bytes(8, "little")
        raw += (2).to_bytes(8, "little") + "é".encode() + b"\2"
        self.assertEqual(reader.r_readout_f32_digest(reader.FLOAT_BITS_DOMAIN, {"é": True, "a": 1}), framed(raw))
        self.assertNotEqual(reader.r_readout_f32_digest(reader.FLOAT_BITS_DOMAIN, 1),
            reader.r_readout_f32_digest(reader.FLOAT_BITS_DOMAIN, 1.0))

    def test_native_exact_spelling_and_semantic_f32_meaning_are_separate(self):
        first, bound = input_fixture()
        second, _ = input_fixture('{"z":1.0,"x":0.00000001,"schema":"fixture-numeric-only"}')
        self.assertEqual(first["semantic_input_sha256"], second["semantic_input_sha256"])
        self.assertNotEqual(first["native_input_sha256"], second["native_input_sha256"])
        before = copy.deepcopy(first)
        parsed = reader.validate_input_evidence(first, bound)
        parsed["x"] = 9
        self.assertEqual(first, before)
        altered = {**first, "reference_input_json": second["reference_input_json"]}
        with self.assertRaisesRegex(ValueError, "raw/native prefix"):
            reader.validate_input_evidence(altered, bound)

    def test_model_parameter_identity_and_epoch_are_part_of_semantic_input(self):
        evidence, bound = input_fixture()
        for name, replacement in (("parameter_sha256", "b" * 64), ("collection_model_epoch", 4), ("collection_context_epoch", 6)):
            with self.subTest(field=name), self.assertRaisesRegex(ValueError, "separate semantic seal"):
                reader.validate_input_evidence(evidence, {**bound, name: replacement})

    def test_nonfinite_wide_integer_duplicate_key_and_deep_input_are_typed_rejections(self):
        for value in (float("nan"), float("inf"), 1e100, 1 << 64, -(1 << 63) - 1, {1: None}, (1,)):
            with self.subTest(kind=type(value).__name__), self.assertRaises(ValueError):
                reader.r_readout_f32_digest(reader.FLOAT_BITS_DOMAIN, value)
        for text in ('{"x":1,"x":2}', '{"x":NaN}', '[]'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                reader._original_json(text)
        value = None
        for _ in range(reader.MAX_DEPTH + 2):
            value = [value]
        with self.assertRaisesRegex(ValueError, "recursion bound"):
            reader.r_readout_f32_digest(reader.FLOAT_BITS_DOMAIN, value)

    def test_r_record_digest_never_relabels_pc_or_edits_original_fields(self):
        record = {"schema": reader.RECORD_DOMAIN, "record_sha256": "a" * 64,
            "numeric": {"raw_outputs_bits": [0] * 11}, "opaque_observation": "fixture-only"}
        before = copy.deepcopy(record)
        expected = reader.record_digest(record)
        self.assertEqual(expected, reader.record_digest({**record, "record_sha256": "b" * 64}))
        self.assertEqual(record, before)
        with self.assertRaisesRegex(ValueError, "separate R readout"):
            reader.record_digest({"schema": "rz-rovent-training-record/1"})


if __name__ == "__main__":
    unittest.main()
