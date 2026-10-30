"""Synthetic wire-family fixtures only; no valid input or native authority."""
import copy
from dataclasses import FrozenInstanceError
import importlib.abc
import sys
import unittest


class NeuralImportsForbidden(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"torch", "numpy", "onnx", "onnxruntime"}:
            raise AssertionError("neural imports forbidden in corpus wire-family tests")


_guard = NeuralImportsForbidden()
sys.meta_path.insert(0, _guard)
try:
    from rz_pals_model import rovent_corpus_codec as codec
finally:
    sys.meta_path.remove(_guard)


def fixture(family, native_revision=None):
    """Intentionally incomplete payload; only its wire family is meaningful."""
    raw = family.native_domains[0] if native_revision is None else native_revision
    snapshot = {} if family is codec.LEGACY_CODEC else {"schema": family.input_domain}
    return ({"schema": family.manifest_domain}, {"schema": family.index_domain},
            {"schema": family.record_domain, "input": snapshot,
             "native": {name: {"schema": raw} for name in
                        ("raw_submission", "raw_completion", "raw_retirement")}})


class CorpusCodecTests(unittest.TestCase):
    def setUp(self):
        self.guard = NeuralImportsForbidden()
        sys.meta_path.insert(0, self.guard)

    def tearDown(self):
        sys.meta_path.remove(self.guard)

    def test_exact_legacy_domain_registry_preserves_schema_less_generation_v3(self):
        value = codec.codec_from_manifest({"schema": "rz-rovent-corpus-manifest/1"})
        self.assertEqual((value.manifest_domain, value.index_domain, value.record_domain,
            value.publication_domain, value.input_domain, value.native_domains),
            ("rz-rovent-corpus-manifest/1", "rz-rovent-corpus-index/1", "rz-rovent-training-record/1",
             "rz-rovent-corpus-publication/1", "rz-pals-generation-input/3",
             ("rz-rovent-native-owner-event/1", "rz-rovent-native-owner-event/2")))
        self.assertEqual(value.source_domain, "rz-rovent-training-record/1")
        for revision in value.native_domains:
            manifest, index, row = fixture(value, revision)
            self.assertIs(codec.validate_container_pair(manifest, index), value)
            self.assertIs(codec.validate_record_codec(manifest, row), value)
            self.assertNotIn("schema", row["input"])

    def test_exact_submitted_domain_registry_keeps_new_source_domain(self):
        value = codec.codec_from_publication({"schema": "rz-rovent-corpus-publication/2"})
        self.assertEqual((value.manifest_domain, value.index_domain, value.record_domain,
            value.publication_domain, value.input_domain, value.native_domains),
            ("rz-rovent-corpus-manifest/2", "rz-rovent-corpus-index/2", "rz-rovent-training-record/2",
             "rz-rovent-corpus-publication/2", "rz-rovent-submitted-input/1",
             ("rz-rovent-raw-record/3",)))
        self.assertEqual(value.source_domain, "rz-rovent-training-record/2")
        manifest, index, row = fixture(value)
        self.assertIs(codec.validate_container_pair(manifest, index), value)
        self.assertIs(codec.validate_record_codec(manifest, row), value)

    def test_descriptors_are_immutable_and_contain_no_authority_claim(self):
        value = codec.codec_from_manifest({"schema": "rz-rovent-corpus-manifest/2"})
        with self.assertRaises(FrozenInstanceError):
            value.record_domain = "rz-rovent-training-record/1"
        self.assertIs(type(value.native_domains), tuple)
        self.assertFalse(hasattr(value, "execution_authorized"))
        self.assertFalse(hasattr(value, "actual_training_authority_verified"))

    def test_manifest_and_publication_reject_unknown_or_missing_schema(self):
        for select in (codec.codec_from_manifest, codec.codec_from_publication):
            for value in (None, [], "schema", {}, {"schema": None}, {"schema": 2},
                          {"schema": "rz-rovent-corpus-manifest/3"},
                          {"schema": "rz-pals-generation-input/3"}):
                with self.subTest(select=select.__name__, value=value):
                    with self.assertRaises(ValueError):
                        select(value)

    def test_manifest_and_publication_do_not_accept_each_others_domain(self):
        for family in (codec.LEGACY_CODEC, codec.SUBMITTED_CODEC):
            with self.assertRaisesRegex(ValueError, "unsupported"):
                codec.codec_from_manifest({"schema": family.publication_domain})
            with self.assertRaisesRegex(ValueError, "unsupported"):
                codec.codec_from_publication({"schema": family.manifest_domain})

    def test_manifest_and_index_family_mismatch_is_explicit(self):
        for first, second in ((codec.LEGACY_CODEC, codec.SUBMITTED_CODEC),
                              (codec.SUBMITTED_CODEC, codec.LEGACY_CODEC)):
            with self.assertRaisesRegex(ValueError, "manifest/index wire family mismatch"):
                codec.validate_container_pair({"schema": first.manifest_domain}, {"schema": second.index_domain})

    def test_record_family_cannot_be_relabelled_by_manifest(self):
        for first, second in ((codec.LEGACY_CODEC, codec.SUBMITTED_CODEC),
                              (codec.SUBMITTED_CODEC, codec.LEGACY_CODEC)):
            manifest, _index, _row = fixture(first)
            _other_manifest, _other_index, row = fixture(second)
            with self.assertRaisesRegex(ValueError, "manifest/record wire family mismatch"):
                codec.validate_record_codec(manifest, row)

    def test_submitted_input_schema_is_required_and_legacy_schema_is_forbidden(self):
        manifest, _index, row = fixture(codec.SUBMITTED_CODEC)
        for value in ({}, {"schema": None}, {"schema": "rz-pals-generation-input/3"}):
            row["input"] = value
            with self.assertRaises(ValueError):
                codec.validate_record_codec(manifest, row)
        manifest, _index, row = fixture(codec.LEGACY_CODEC)
        for value in ({"schema": "rz-rovent-submitted-input/1"}, {"schema": None},
                      {"schema": "rz-pals-generation-input/3"}):
            row["input"] = value
            with self.assertRaisesRegex(ValueError, "legacy Generation V3 input must remain schema-less"):
                codec.validate_record_codec(manifest, row)

    def test_original_native_revisions_cannot_cross_wire_families(self):
        for family, rejected in ((codec.LEGACY_CODEC, "rz-rovent-raw-record/3"),
                                 (codec.SUBMITTED_CODEC, "rz-rovent-native-owner-event/1"),
                                 (codec.SUBMITTED_CODEC, "rz-rovent-native-owner-event/2")):
            manifest, _index, row = fixture(family, rejected)
            with self.assertRaisesRegex(ValueError, "record/native wire family mismatch"):
                codec.validate_record_codec(manifest, row)

    def test_native_three_bodies_cannot_mix_legacy_revisions(self):
        manifest, _index, row = fixture(codec.LEGACY_CODEC)
        row["native"]["raw_completion"]["schema"] = "rz-rovent-native-owner-event/2"
        with self.assertRaisesRegex(ValueError, "one schema revision"):
            codec.validate_record_codec(manifest, row)

    def test_missing_native_or_body_schema_is_not_inferred(self):
        for family in (codec.LEGACY_CODEC, codec.SUBMITTED_CODEC):
            for missing in ("native", "raw_submission", "schema"):
                manifest, _index, row = fixture(family)
                if missing == "native":
                    row.pop("native")
                elif missing == "raw_submission":
                    row["native"].pop("raw_submission")
                else:
                    row["native"]["raw_submission"].pop("schema")
                with self.subTest(family=family.record_domain, missing=missing):
                    with self.assertRaises(ValueError):
                        codec.validate_record_codec(manifest, row)

    def test_family_validation_neither_mutates_nor_validates_unrelated_payload(self):
        for family in (codec.LEGACY_CODEC, codec.SUBMITTED_CODEC):
            manifest, index, row = fixture(family)
            manifest["not_validated_here"] = {"provenance": ["not a source pin"]}
            index["not_validated_here"] = ["not an original row index"]
            row["input"]["attack_ply"] = "not a valid model input"
            row["native"]["raw_retirement"]["accepted_retired"] = "not authority"
            before = copy.deepcopy((manifest, index, row))
            self.assertIs(codec.validate_container_pair(manifest, index), family)
            self.assertIs(codec.validate_record_codec(manifest, row), family)
            self.assertEqual((manifest, index, row), before)


if __name__ == "__main__":
    unittest.main()
