"""Current P/C↔R sampling-leak fixtures, with no optimizer/native authority."""
import copy
import importlib.abc
import sys
from pathlib import Path
import tempfile
import unittest


class NeuralImportsForbidden(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"torch", "numpy", "onnx", "onnxruntime", "tensorflow", "jax"}:
            raise AssertionError("neural imports forbidden in joint corpus fixtures")


_guard = NeuralImportsForbidden()
sys.meta_path.insert(0, _guard)
try:
    from rz_pals_model import rovent_joint_corpus as joint
    from rz_pals_model import rovent_corpus as pc
    from rz_pals_model import rovent_r_corpus as r
    from rz_pals_model import rovent_r_readout as wire
    from rz_pals_model import generation_data as data
    from test_rovent_r_corpus import fixture, config, seal, write, h, submitted_pc_owner
    from test_rovent_corpus import original_fixture, seal_record, write_corpus
finally:
    sys.meta_path.remove(_guard)


def reseal_pc(manifest, row):
    row["input_sha256"] = data.seal_input(row["input"])
    native = row["native"]
    native["source_sha256"] = manifest["source_sha256"]
    for key in ("raw_submission", "raw_completion", "raw_retirement"):
        native[key].update(source_sha256=manifest["source_sha256"], input_sha256=row["input_sha256"])
    native["raw_submission_sha256"] = data.digest(pc.NATIVE_RAW_DOMAIN, native["raw_submission"])
    native["raw_completion"]["submitted_sha256"] = native["raw_submission_sha256"]
    native["raw_completion_sha256"] = data.digest(pc.NATIVE_RAW_DOMAIN, native["raw_completion"])
    native["raw_retirement"].update(submitted_sha256=native["raw_submission_sha256"], completed_sha256=native["raw_completion_sha256"])
    native["raw_retirement_sha256"] = data.digest(pc.NATIVE_RAW_DOMAIN, native["raw_retirement"])
    seal_record(manifest, row)


def paired_owners(root, *, pc_split="train", r_split="train", leak=False, pc_already_in_r=False, pc_model_epoch=None,
                  submitted_pc=False):
    recipe, manifest, r_row, lock = fixture()
    splits = recipe.to_dict()["opening_splits"]
    pc_opening = next(key for key, split in splits.items() if split == pc_split)
    r_opening = next(key for key, split in splits.items() if split == r_split)
    if submitted_pc:
        pc_owner, pc_row = submitted_pc_owner(root / "pc", recipe, manifest["collection"], lock,
            opening=pc_opening, model_epoch=pc_model_epoch)
    else:
        _original_recipe, pc_manifest, pc_row = original_fixture(opening=pc_opening)
        for key in ("source_commit", "source_sha256", "runtime_sha256", "model_semantics_sha256", "recipe_sha256", "model_epoch", "context_epoch"):
            pc_manifest[key] = manifest["collection"][key]
        if pc_model_epoch is not None:
            pc_manifest["model_epoch"] = pc_model_epoch
        pc_row["input"].update(model_epoch=pc_manifest["model_epoch"], context_epoch=pc_manifest["context_epoch"])
        pc_row["split"] = pc_split
        reseal_pc(pc_manifest, pc_row)
        pc_root = root / "pc"
        pc_root.mkdir(parents=True)
        pc_owner = write_corpus(pc_root, pc_manifest, [pc_row])
        pc_owner.admit_stage(recipe, pc_manifest["stage_id"], pc_manifest["model_epoch"], pc_manifest["context_epoch"], [], [])
    r_row["lineage"]["opening_id"] = r_opening
    r_row["split"] = r_split
    if leak:
        r_row["lineage"]["transposition_sha256"] = pc_row["input"]["transposition_sha256"]
    if pc_already_in_r:
        manifest["pc_ancestor_index_sha256"] = [pc_owner.index_semantic_sha256]
        r_row["manifest_sha256"] = data.digest(wire.MANIFEST_DOMAIN, manifest)
    seal(r_row)
    r_root = root / "r"
    files = write(r_root, manifest, [r_row], lock)
    r_owner = r.RoventRCorpus.from_files(files, recipe=recipe, nnuec_config=config(),
        expected_nnuec_parameter_sha256=r_row["bound"]["parameter_sha256"], pc_corpora=(pc_owner,) if pc_already_in_r else ())
    return pc_owner, r_owner, pc_row, r_row


def build(pc_owner, r_owner, **kwargs):
    return joint.build_pc_r_companion(pc_owner, r_owner, training_input_sha256=kwargs.get("pc_inputs", []),
        training_origin_sha256=kwargs.get("pc_origins", []), retained_r_input_sha256=kwargs.get("r_inputs", []),
        retained_r_origin_sha256=kwargs.get("r_origins", []))


class JointCorpusTests(unittest.TestCase):
    def setUp(self):
        self.guard = NeuralImportsForbidden()
        sys.meta_path.insert(0, self.guard)

    def tearDown(self):
        sys.meta_path.remove(self.guard)

    def test_original_tokens_and_datasets_survive_without_new_authority(self):
        with tempfile.TemporaryDirectory() as directory:
            pc_owner, r_owner, pc_row, r_row = paired_owners(Path(directory))
            pc_pin, r_pin = pc_owner.index_semantic_sha256, r_owner.index_semantic_sha256
            view = build(pc_owner, r_owner)
            self.assertIs(type(view.pc_rows("proposer")[0]), pc.CorpusRow)
            self.assertIs(type(view.r_rows()[0]), r.RCorpusRow)
            self.assertEqual(view.r_training_sample(view.r_rows()[0])["record_sha256"], r_row["record_sha256"])
            self.assertEqual(pc_owner.training_sample(view.pc_rows("proposer")[0])["record_sha256"], pc_row["record_sha256"])
            self.assertEqual((pc_owner.index_semantic_sha256, r_owner.index_semantic_sha256), (pc_pin, r_pin))
            self.assertIs(view.observation()["execution_authorized"], False)
            self.assertIs(view.observation()["original_checkpoint_authority_verified"], False)

    def test_current_pc_outside_r_declared_ancestors_still_excludes_both_leaking_streams(self):
        with tempfile.TemporaryDirectory() as directory:
            pc_owner, r_owner, _pc_row, _r_row = paired_owners(Path(directory), r_split="holdout", leak=True)
            self.assertEqual(r_owner._manifest["pc_ancestor_index_sha256"], [])
            self.assertEqual(len(pc_owner.rows("proposer")), 1)
            self.assertEqual(len(r_owner.rows("holdout", optimization=False)), 1)
            original_r_pool = r_owner.pool_metadata("holdout", optimization=False)["pool_sha256"]
            view = build(pc_owner, r_owner)
            self.assertEqual(len(view.pc_rows("proposer")), 0)
            self.assertEqual(len(view.r_rows("holdout", optimization=False)), 0)
            self.assertNotEqual(view.pool_metadata("holdout", optimization=False)["pool_sha256"], original_r_pool)
            self.assertEqual(len(r_owner.all_occurrence_metadata()["occurrences"]), 1)
            self.assertEqual(view.observation()["lineage"]["excluded_occurrence_count"], 2)

    def test_previously_consumed_pc_training_origin_cannot_be_dropped_to_hide_new_r_holdout(self):
        with tempfile.TemporaryDirectory() as directory:
            pc_owner, r_owner, pc_row, _r_row = paired_owners(Path(directory), r_split="holdout", leak=True)
            with self.assertRaisesRegex(ValueError, "global holdout/cross-split contamination"):
                build(pc_owner, r_owner, pc_inputs=[pc_row["input_sha256"]], pc_origins=[pc_row["origin_sha256"]])

    def test_previously_consumed_r_training_origin_cannot_be_dropped_to_hide_current_pc_holdout(self):
        with tempfile.TemporaryDirectory() as directory:
            pc_owner, r_owner, _pc_row, r_row = paired_owners(Path(directory), pc_split="holdout", leak=True)
            with self.assertRaisesRegex(ValueError, "global holdout/cross-split contamination"):
                build(pc_owner, r_owner, r_inputs=[r_row["input"]["semantic_input_sha256"]], r_origins=[r_row["origin_sha256"]])

    def test_reused_same_source_occurrence_is_deduplicated_across_two_original_closures(self):
        with tempfile.TemporaryDirectory() as directory:
            pc_owner, r_owner, _pc_row, _r_row = paired_owners(Path(directory), pc_already_in_r=True)
            self.assertEqual(r_owner._union.observation().occurrence_count, 2)
            view = build(pc_owner, r_owner)
            self.assertEqual(view.observation()["lineage"]["occurrence_count"], 2)
            self.assertEqual(len(view.pc_rows("proposer")), 1)
            self.assertEqual(len(view.r_rows()), 1)

    def test_retained_input_and_origin_domains_must_match_exact_original_projection(self):
        with tempfile.TemporaryDirectory() as directory:
            pc_owner, r_owner, pc_row, r_row = paired_owners(Path(directory))
            for kwargs in ({"pc_inputs": [h("wrong input")], "pc_origins": [pc_row["origin_sha256"]]},
                           {"pc_inputs": [r_row["input"]["semantic_input_sha256"]], "pc_origins": [r_row["origin_sha256"]]}):
                with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, "origin/input projection differs|absent from its original source domain"):
                    build(pc_owner, r_owner, **kwargs)

    def test_joint_r_pool_uses_only_joint_eligible_order_and_new_resume_seal(self):
        with tempfile.TemporaryDirectory() as directory:
            pc_owner, r_owner, _pc_row, _r_row = paired_owners(Path(directory), pc_split="holdout", leak=True)
            old_pool = r_owner.pool_metadata()
            self.assertEqual(len(old_pool["occurrences"]), 1)
            view = build(pc_owner, r_owner)
            pool = view.pool_metadata()
            self.assertEqual(pool["occurrences"], [])
            self.assertNotEqual(pool["pool_sha256"], old_pool["pool_sha256"])
            self.assertEqual(pool["pool_sha256"], data.digest(r.POOL_DOMAIN,
                {"index_sha256": r_owner.index_semantic_sha256, "occurrences": []}))

    def test_original_r_tokens_from_excluded_rows_cannot_bypass_joint_dispatcher(self):
        with tempfile.TemporaryDirectory() as directory:
            pc_owner, r_owner, _pc_row, r_row = paired_owners(Path(directory), pc_split="holdout", leak=True)
            original_token = r_owner.rows()[0]
            view = build(pc_owner, r_owner)
            with self.assertRaisesRegex(ValueError, "excluded by the joint"):
                view.r_training_sample(original_token)
            with self.assertRaisesRegex(ValueError, "not joint training eligible"):
                view.original_auxiliary_evidence(r_row["record_sha256"])

    def test_joint_view_never_impersonates_a_corpus_and_requires_pc_stage_intake(self):
        with self.assertRaisesRegex(ValueError, "exact original P/C and R corpus owner types"):
            build(None, None)
        with tempfile.TemporaryDirectory() as directory:
            pc_owner, r_owner, _pc_row, _r_row = paired_owners(Path(directory))
            pc_owner._admission = None
            with self.assertRaisesRegex(ValueError, "stage admission must precede"):
                build(pc_owner, r_owner)

    def test_immutable_joint_eligibility_metadata_is_checked_without_reissuing_authority(self):
        with tempfile.TemporaryDirectory() as directory:
            pc_owner, r_owner, _pc_row, _r_row = paired_owners(Path(directory))
            view = build(pc_owner, r_owner)
            view._report["execution_authorized"] = True
            with self.assertRaisesRegex(ValueError, "immutable sampling projection changed"):
                view.observation()

    def test_frozen_pc_epoch_change_keeps_original_r_namespace_and_leakage_union(self):
        with tempfile.TemporaryDirectory() as directory:
            pc_owner, r_owner, _pc_row, _r_row = paired_owners(Path(directory), pc_model_epoch=2)
            self.assertNotEqual(pc_owner._manifest["model_epoch"], r_owner._manifest["collection"]["model_epoch"])
            old_pool = r_owner.pool_metadata()["pool_sha256"]
            view = build(pc_owner, r_owner)
            self.assertEqual(view.pool_metadata()["pool_sha256"], old_pool)
            self.assertEqual(r_owner._manifest["collection"]["model_epoch"], 1)
            self.assertFalse(view.observation()["execution_authorized"])

    def test_submitted_pc_opaque_tokens_keep_new_domain_in_joint_current_stream(self):
        with tempfile.TemporaryDirectory() as directory:
            pc_owner, r_owner, pc_row, _r_row = paired_owners(Path(directory), submitted_pc=True)
            view = build(pc_owner, r_owner, pc_inputs=[pc_row["input_sha256"]], pc_origins=[pc_row["origin_sha256"]])
            token = view.pc_rows("proposer")[0]
            self.assertIs(type(token), pc.CorpusRow)
            self.assertIs(token._owner, pc_owner)
            self.assertEqual(pc_owner.training_sample(token)["input"]["schema"], "rz-rovent-submitted-input/1")
            sources = {row.source.source_domain for row in view._union.all_occurrences()}
            self.assertEqual(sources, {"rz-rovent-training-record/2", "rz-rovent-r-readout-record/1"})
            self.assertIn(pc_row["record_sha256"], view.observation()["allowed_pc_record_sha256"])
            self.assertFalse(view.observation()["execution_authorized"])

    def test_submitted_pc_and_r_holdout_are_excluded_without_erasing_raw_occurrences(self):
        with tempfile.TemporaryDirectory() as directory:
            pc_owner, r_owner, pc_row, _r_row = paired_owners(Path(directory), submitted_pc=True, r_split="holdout", leak=True)
            view = build(pc_owner, r_owner)
            self.assertEqual(len(view.pc_rows("proposer")), 0)
            self.assertEqual(len(view.r_rows("holdout", optimization=False)), 0)
            self.assertEqual(view.observation()["lineage"]["occurrence_count"], 2)
            self.assertEqual(len(pc_owner.rows("proposer")), 1)
            with self.assertRaisesRegex(ValueError, "global holdout/cross-split contamination"):
                build(pc_owner, r_owner, pc_inputs=[pc_row["input_sha256"]], pc_origins=[pc_row["origin_sha256"]])

    def test_submitted_pc_declared_by_r_is_deduplicated_without_becoming_legacy_pc(self):
        with tempfile.TemporaryDirectory() as directory:
            pc_owner, r_owner, _pc_row, _r_row = paired_owners(Path(directory), submitted_pc=True, pc_already_in_r=True)
            view = build(pc_owner, r_owner)
            self.assertEqual(view.observation()["lineage"]["occurrence_count"], 2)
            self.assertEqual(len(view.pc_rows("proposer")), 1)
            self.assertEqual(len(view.r_rows()), 1)
            self.assertFalse(any(row.source.source_domain == "rz-rovent-training-record/1"
                                 for row in view._union.all_occurrences()))


if __name__ == "__main__":
    unittest.main()
