"""Original-byte submitted corpus contracts, without native or learning runs."""
import copy
import hashlib
import importlib.abc
import json
from pathlib import Path
import sys
import tempfile
import unittest


class NeuralImportsForbidden(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"torch", "numpy", "onnx", "onnxruntime"}:
            raise AssertionError("submitted corpus contracts cannot dispatch neural work")


_guard = NeuralImportsForbidden()
sys.meta_path.insert(0, _guard)
try:
    from rz_pals_model import generation_data as data
    from rz_pals_model import rovent_corpus as corpus
    from rz_pals_model.rovent_corpus_codec import LEGACY_CODEC, SUBMITTED_CODEC
    from rz_pals_model.rovent_generation import index_original_input
    from rz_pals_model import rovent_submitted_input as inputs
    from test_rovent_native_v3 import native_fixture
    from test_rovent_historical_corpus import synthetic_v2
    from test_rovent_corpus import original_fixture, write_corpus
finally:
    sys.meta_path.remove(_guard)


def raw(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n").encode()


def seal(manifest, record):
    record["input_sha256"] = inputs.seal_input(record["input"])
    record["origin_sha256"] = data.digest(corpus.ORIGIN_DOMAIN, corpus._origin(manifest, record))
    record["record_sha256"] = corpus._record_digest(record)


def fixture():
    recipe = synthetic_v2()[0]
    manifest, record = native_fixture()
    seal(manifest, record)
    return recipe, manifest, record


def indexed_origin(record):
    value = {name: record[name] for name in ("record_sha256", "input_sha256", "origin_sha256", "split")}
    value.update({name: record["input"][name] for name in
        ("game_id", "opening_id", "line_genealogy_id", "transposition_sha256")})
    return {**value, **record["lineage"]}


def write_submitted(root, manifest, records, *, ancestors=()):
    root.mkdir(parents=True, exist_ok=True)
    manifest = copy.deepcopy(manifest)
    manifest["ancestor_index_sha256"] = sorted(owner.index_semantic_sha256 for owner in ancestors)
    chunk = b"".join(raw(record) for record in records)
    index = {"schema": SUBMITTED_CODEC.index_domain,
        "manifest_sha256": data.digest(SUBMITTED_CODEC.manifest_domain, manifest), "record_count": len(records),
        "chunks": [{"path": "rows.jsonl", "bytes": len(chunk), "sha256": hashlib.sha256(chunk).hexdigest(), "records": len(records)}],
        "origins": [indexed_origin(record) for record in records]}
    manifest_raw, index_raw = raw(manifest), raw(index)
    for name, payload in (("manifest.json", manifest_raw), ("index.json", index_raw), ("rows.jsonl", chunk)):
        (root / name).write_bytes(payload)
    return {"manifest_path": root / "manifest.json", "manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "index_path": root / "index.json", "index_sha256": hashlib.sha256(index_raw).hexdigest(), "ancestors": ancestors}


class SubmittedCorpusTests(unittest.TestCase):
    def setUp(self):
        sys.meta_path.insert(0, _guard)

    def tearDown(self):
        sys.meta_path.remove(_guard)

    def test_actual_new_reader_keeps_new_domain_and_exact_submitted_input_seal(self):
        recipe, manifest, record = fixture()
        with tempfile.TemporaryDirectory() as directory:
            owner = corpus.RoventTrainingCorpus.from_submitted_files(**write_submitted(Path(directory), manifest, [record]))
            self.assertIs(owner._codec, SUBMITTED_CODEC)
            self.assertEqual(owner.manifest_semantic_sha256, data.digest(SUBMITTED_CODEC.manifest_domain, manifest))
            admission = owner.admit_stage(recipe, "common", 1, 1, [], [])
            sample = owner.training_sample(owner.rows("proposer")[0])
            self.assertEqual(sample["input"], record["input"])
            self.assertEqual(sample["input_sha256"], inputs.seal_input(record["input"]))
            self.assertFalse(admission["execution_authorized"])
            self.assertFalse(admission["actual_training_executed"])

    def test_old_and_submitted_constructors_do_not_convert_each_others_files(self):
        _recipe, manifest, record = fixture()
        with tempfile.TemporaryDirectory() as directory:
            paths = write_submitted(Path(directory), manifest, [record])
            with self.assertRaises(ValueError):
                corpus.RoventTrainingCorpus.from_files(**paths)
        _recipe, old_manifest, old_record = original_fixture()
        with tempfile.TemporaryDirectory() as directory:
            owner = write_corpus(Path(directory), old_manifest, [old_record])
            self.assertIs(getattr(owner, "_codec", LEGACY_CODEC), LEGACY_CODEC)
            with self.assertRaises(ValueError):
                corpus.RoventTrainingCorpus.from_submitted_files(Path(directory) / "manifest.json",
                    manifest_sha256=owner.corpus_sha256, index_path=Path(directory) / "index.json", index_sha256=owner.index_sha256)

    def test_new_index_and_record_hashes_are_not_old_domain_aliases(self):
        _recipe, manifest, record = fixture()
        projection = {**record, "record_sha256": "", "input": {"sealed_input_sha256": record["input_sha256"]}}
        self.assertNotEqual(record["record_sha256"], data.digest(corpus.RECORD_DOMAIN, projection))
        self.assertEqual(record["record_sha256"], data.digest(SUBMITTED_CODEC.record_domain, projection))
        with tempfile.TemporaryDirectory() as directory:
            owner = corpus.RoventTrainingCorpus.from_submitted_files(**write_submitted(Path(directory), manifest, [record]))
            self.assertNotEqual(owner.index_semantic_sha256, data.digest(corpus.INDEX_DOMAIN, owner._index))

    def test_original_index_projection_accepts_new_input_without_discarding_its_seal(self):
        recipe, manifest, record = fixture()
        before = copy.deepcopy(record["input"])
        projection = index_original_input(recipe, input_snapshot=record["input"], input_sha256=record["input_sha256"],
            record_sha256=record["record_sha256"], producer_sha256=data.digest(SUBMITTED_CODEC.manifest_domain, manifest),
            stage_id="common", round_id=manifest["round_id"], teacher_actor_id=None)
        self.assertEqual(projection["input_sha256"], inputs.seal_input(before))
        self.assertEqual(record["input"], before)
        with self.assertRaises(ValueError):
            data.validate_input(record["input"])

    def test_new_pc_record_cannot_acquire_legacy_v_or_nnuec_heads(self):
        _recipe, manifest, record = fixture()
        for mutate in (lambda row: row.update(nnuec_input={}),
                       lambda row: row["targets"].update(task={}),
                       lambda row: row["head_sources"].update(nnuec_wdl={})):
            bad = copy.deepcopy(record)
            mutate(bad)
            with self.subTest(mutate=mutate), self.assertRaisesRegex(ValueError, "cannot manufacture"):
                corpus._validate_row(manifest, bad, indexed_origin(bad), corpus.CorpusLimits())

    def test_owner_cannot_replace_its_explicit_wire_family_after_reading(self):
        _recipe, manifest, record = fixture()
        with tempfile.TemporaryDirectory() as directory:
            owner = corpus.RoventTrainingCorpus.from_submitted_files(**write_submitted(Path(directory), manifest, [record]))
            owner._codec = LEGACY_CODEC
            with self.assertRaisesRegex(ValueError, "wire family"):
                owner._check_owner(full=True)


if __name__ == "__main__":
    unittest.main()
