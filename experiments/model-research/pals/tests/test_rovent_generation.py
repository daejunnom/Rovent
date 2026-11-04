"""Preparation contracts only: no model, optimizer, forward, or training run."""
import copy
import hashlib
import importlib.abc
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest import mock


class NeuralImportsForbidden(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"torch", "numpy", "onnx", "onnxruntime"}:
            raise AssertionError("neural execution forbidden in Rovent series contract tests")


_guard = NeuralImportsForbidden()
sys.meta_path.insert(0, _guard)
try:
    from rz_pals_model import rovent_generation as series
    from rz_pals_model import rovent_generation_cli as cli
finally:
    sys.meta_path.remove(_guard)


def seal(value):
    return hashlib.sha256(value.encode()).hexdigest()


def stage(stage_id, kind, parent, teachers):
    return {"stage_id": stage_id, "kind": kind, "parent_stage_id": parent,
            "teacher_actor_ids": teachers, "round_id": stage_id + "-round",
            "role_updates": {"proposer": 2, "critic": 2, "validator": 0}}


def recipe_value():
    actors = [{"actor_id": family, "family": family, "source_revision": ("1" if family == "stockfish" else "2") * 40,
               **{part + "_sha256": seal(family + part) for part in ("source", "binary", "profile", "options", "assets", "rights")}}
              for family in ("stockfish", "reckless")]
    return {"schema": series.RECIPE_DOMAIN, "series_id": "rovent-first", "student_id": "common-student",
            "learner_id": "sole-learner", "source_sha256": seal("source"), "model_semantics_sha256": seal("semantics"),
            "actors": actors, "parameter_ownership": {"pc_shared": "pc", "p_private": "pc", "c_private": "pc",
                "v_private": "v", "nnuec": "nnuec", "membership_sha256": seal("membership")},
            "stages": [stage("common", "common_student", None, ["stockfish", "reckless"]),
                       stage("sf-ft", "teacher_finetune", "common", ["stockfish"]),
                       stage("reckless-ft", "teacher_finetune", "common", ["reckless"]),
                       stage("mixed", "mixed", "sf-ft", ["stockfish", "reckless"]),
                       stage("self", "self", "mixed", [])],
            "opening_splits": {"train-a": "train", "train-b": "train", "held": "holdout", "valid": "validation"},
            "global_holdout_openings": ["held"]}


def origin(name, *, stage_id="common", teacher="stockfish", opening="train-a"):
    return {"input_sha256": seal(name + "input"), "record_sha256": seal(name + "record"),
            "producer_sha256": seal(name + "producer"), "stage_id": stage_id, "round_id": stage_id + "-round",
            "teacher_actor_id": teacher, "opening_id": opening, "game_id": name + "-game",
            "line_genealogy_id": name + "-line", "rules_state_sha256": seal(name + "state"),
            "transposition_sha256": seal(name + "transposition")}


def checkpoint(recipe, records=(), *, stage_id="common", parent=None, complete=True):
    selected = recipe.stage(stage_id)
    return {"schema": series.CHECKPOINT_DOMAIN, "recipe_sha256": recipe.sha256, "stage_id": stage_id,
            "checkpoint_sha256": seal(stage_id + "checkpoint"), "model_sha256": seal(stage_id + "model"),
            "parent_checkpoint_sha256": parent, "parameter_membership_sha256": seal("membership"),
            "split_sha256": recipe.split_sha256, "state_sha256": {part: seal(part) for part in series.STATE_PARTS},
            "completed_updates": dict(zip(series.ROLES, selected.role_updates)) if complete else dict.fromkeys(series.ROLES, 0),
            "pending_microbatches": 0, "lineage_teacher_actor_ids": list(recipe.inherited_teachers(stage_id)),
            "training_input_sha256": sorted({record["input_sha256"] for record in records}),
            "training_origin_sha256": sorted({series.record_origin_digest(recipe, record) for record in records})}


def legacy_bundle():
    """Small schema-only source evidence; never real collection/training proof."""
    from test_generation_contract import row, policy, add_teacher_policy_evidence
    from test_generation_chunks import add_native_chain
    from rz_pals_model import generation_data as data
    record, receipts = row(), {}
    record["input"].update(opening_id="train-a", game_id="game-a", line_genealogy_id="line-a")
    record["input_sha256"] = data.seal_input(record["input"])
    target = policy(record)
    receipt = {"schema": "rz-pals-generation-head-evidence/1", "input_sha256": record["input_sha256"],
               "head": "policy", "target": {key: value for key, value in target.items() if key != "raw_evidence_sha256"}}
    target["raw_evidence_sha256"] = data.digest(receipt["schema"], receipt)
    record["future_targets"]["policy"] = target
    receipts[target["raw_evidence_sha256"]] = receipt
    add_teacher_policy_evidence(record, receipts, target)
    add_native_chain(record, receipts)
    splits = {"train-a": "train"}
    producer = {"schema": "rz-pals-actual-collection-authority/1", "source_kind": data.ACTUAL_SOURCE,
                "source_sha256": "1" * 64, "model_sha256": "4" * 64, "runtime_sha256": "5" * 64,
                "teacher_binary_sha256": "6" * 64, "teacher_profile_sha256": "7" * 64,
                "opening_split_sha256": data.digest("rz-pals-opening-split/1", splits), "registered_before_outcomes": True,
                "input_seals": [record["input_sha256"]], "actual_game_ids": ["game-a"]}
    return {"producer": producer, "trajectories": {}, "opening_splits": splits,
            "allowed_evidence": receipts, "records": [record]}


def legacy_recipe():
    value = recipe_value()
    value["actors"][0].update(binary_sha256="6" * 64, profile_sha256="7" * 64)
    return series.SeriesRecipe(value)


def json_pin(root, name, value):
    path = root / name
    raw = (json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
    path.write_bytes(raw)
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


def request_value(root, recipe, *, action="validate_recipe"):
    return {"schema": cli.REQUEST_DOMAIN, "action": action, "recipe": json_pin(root, "recipe.json", recipe.to_dict()),
            "checkpoint": None, "ancestors": [], "origins": None, "legacy_intake": None, "stage_id": None,
            "output_root": str(root), "output_name": "preparation-result.json", "max_records": 64,
            "max_output_bytes": 1024 * 1024, "max_wall_ns": 10**9}


class RoventSeriesTests(unittest.TestCase):
    def setUp(self):
        self.guard = NeuralImportsForbidden()
        sys.meta_path.insert(0, self.guard)

    def tearDown(self):
        sys.meta_path.remove(self.guard)

    def test_common_student_and_later_teacher_ft_keep_both_teacher_exposures(self):
        recipe = series.SeriesRecipe(recipe_value())
        self.assertEqual(recipe.inherited_teachers("sf-ft"), ("reckless", "stockfish"))
        self.assertEqual([s.stage_id for s in recipe.lineage("self")], ["common", "sf-ft", "mixed", "self"])
        self.assertEqual(recipe.stage("self").teacher_actor_ids, ())
        value = recipe.to_dict()
        value["student_id"] = "changed-copy"
        self.assertEqual(recipe.to_dict()["student_id"], "common-student")

    def test_frozen_actor_common_stage_and_optimizer_ownership_reject_aliases(self):
        mutations = [lambda value: value["actors"][1].update(family="stockfish"),
                     lambda value: value["actors"][1].update(source_revision="main"),
                     lambda value: value["parameter_ownership"].update(p_private="second-p-moments"),
                     lambda value: value["parameter_ownership"].update(nnuec="v"),
                     lambda value: value["stages"][0].update(teacher_actor_ids=["stockfish"]),
                     lambda value: value["stages"][1].update(parent_stage_id="reckless-ft"),
                     lambda value: value["stages"][4].update(teacher_actor_ids=["stockfish"]),
                     lambda value: value["stages"][0]["role_updates"].update(proposer=True),
                     lambda value: value.update(schema="rz-pals-train-generation/1"),
                     lambda value: value.update(teacher_learners=["one", "two"])]
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                value = recipe_value()
                mutate(value)
                with self.assertRaises(ValueError):
                    series.SeriesRecipe(value)

    def test_shared_common_fork_copies_weights_and_resets_state_while_resume_keeps_state(self):
        recipe = series.SeriesRecipe(recipe_value())
        original = checkpoint(recipe)
        before = copy.deepcopy(original)
        plans = series.common_teacher_forks(recipe, original)
        self.assertEqual({p["parent_checkpoint_sha256"] for p in plans["forks"]}, {original["checkpoint_sha256"]})
        self.assertEqual({p["model_sha256"] for p in plans["forks"]}, {original["model_sha256"]})
        for plan in plans["forks"]:
            self.assertEqual(plan["reset_state"], list(series.STATE_PARTS))
            self.assertEqual(plan["completed_updates"], dict.fromkeys(series.ROLES, 0))
            self.assertIs(plan["execution_authorized"], False)
        resumed = series.resume_plan(recipe, original, stage_id="common")
        self.assertEqual(resumed["checkpoint"], before)
        self.assertEqual(resumed["reset_state"], [])
        resumed["checkpoint"]["state_sha256"]["rng"] = seal("changed")
        self.assertEqual(original, before)
        with self.assertRaisesRegex(ValueError, "resume cannot change stage"):
            series.resume_plan(recipe, original, stage_id="sf-ft")
        incomplete = checkpoint(recipe, complete=False)
        with self.assertRaisesRegex(ValueError, "completed common/parent checkpoint"):
            series.common_teacher_forks(recipe, incomplete)

    def test_resume_rejects_partial_diagnostic_missing_rng_and_erased_teacher_history(self):
        recipe = series.SeriesRecipe(recipe_value())
        for mutation in (lambda value: value.update(pending_microbatches=1),
                         lambda value: value.update(schema="rz-pals-diagnostic-checkpoint/1"),
                         lambda value: value["state_sha256"].pop("rng"),
                         lambda value: value.update(lineage_teacher_actor_ids=["stockfish"])):
            value = checkpoint(recipe)
            mutation(value)
            with self.assertRaises(ValueError):
                series.resume_plan(recipe, value, stage_id="common")

    def test_original_selected_input_binding_cannot_rewrite_legacy_or_future_targets(self):
        from test_generation_contract import row
        record = row()
        record["input"]["opening_id"] = "train-a"
        record["input"]["game_id"] = "game-a"
        record["input"]["line_genealogy_id"] = "line-a"
        record["input_sha256"] = series.seal_input(record["input"])
        before = copy.deepcopy(record)
        recipe = series.SeriesRecipe(recipe_value())
        value = series.index_original_input(recipe, input_snapshot=record["input"], input_sha256=record["input_sha256"],
            record_sha256=seal("record"), producer_sha256=seal("original-producer"), stage_id="common",
            round_id="common-round", teacher_actor_id="reckless")
        self.assertEqual(record, before)
        self.assertEqual(value["input_sha256"], before["input_sha256"])
        self.assertEqual(value["teacher_actor_id"], "reckless")
        bad = copy.deepcopy(record["input"])
        bad["capture_sequence"] += 1
        with self.assertRaisesRegex(ValueError, "input seal mismatch"):
            series.index_original_input(recipe, input_snapshot=bad, input_sha256=record["input_sha256"],
                record_sha256=seal("record"), producer_sha256=seal("producer"), stage_id="common",
                round_id="common-round", teacher_actor_id="reckless")

    def test_transitive_global_leakage_excludes_ancestors_and_both_teacher_branches(self):
        recipe = series.SeriesRecipe(recipe_value())
        a = origin("a")
        b = origin("b", stage_id="sf-ft")
        c = origin("c", stage_id="reckless-ft", teacher="reckless", opening="held")
        b["line_genealogy_id"] = a["line_genealogy_id"]
        c["rules_state_sha256"] = b["rules_state_sha256"]
        index = series.GlobalLineageIndex(recipe, [a, b, c])
        self.assertEqual(index.rows("train", optimization=True), [])
        self.assertEqual(index.rows("holdout"), [])
        with self.assertRaisesRegex(ValueError, "global holdout/cross-split contamination"):
            index.admit_checkpoint_lineage(recipe, checkpoint(recipe, [a]), ancestors=[])
        with self.assertRaisesRegex(ValueError, "cannot enter optimization"):
            index.rows("holdout", optimization=True)

    def test_distinct_teacher_origins_survive_identical_input_seals_and_global_audit(self):
        recipe = series.SeriesRecipe(recipe_value())
        sf, reckless = origin("sf"), origin("reckless", teacher="reckless")
        reckless["input_sha256"] = sf["input_sha256"]
        rows = [sf, reckless]
        index = series.GlobalLineageIndex(recipe, rows)
        self.assertEqual(len(index.rows("train", optimization=True)), 2)
        self.assertNotEqual(series.record_origin_digest(recipe, sf), series.record_origin_digest(recipe, reckless))
        result = index.admit_checkpoint_lineage(recipe, checkpoint(recipe, rows), ancestors=[])
        self.assertEqual((result["training_origins"], result["training_inputs"]), (2, 1))
        self.assertIs(result["native_intake_verified"], False)
        self.assertIs(result["execution_authorized"], False)

    def test_own_origin_keeps_null_teacher_without_creating_teacher_policy_authority(self):
        recipe = series.SeriesRecipe(recipe_value())
        own = origin("own-rules", teacher=None)
        self.assertIsNone(series.validate_record_origin(recipe, own)["teacher_actor_id"])
        index = series.GlobalLineageIndex(recipe, [own])
        self.assertIsNone(index.rows("train", optimization=True)[0]["teacher_actor_id"])
        # An origin index is not teacher-policy admission. The legacy intake
        # still requires an actual frozen actor even for an otherwise valid row.
        with self.assertRaisesRegex(ValueError, "actor is not assigned"):
            series.audit_legacy_bundle(recipe, None, stage_id="common", teacher_actor_id=None,
                expected_model_sha256="4" * 64, expected_runtime_sha256="5" * 64)
        foreign = origin("foreign", stage_id="sf-ft", teacher="reckless")
        with self.assertRaisesRegex(ValueError, "frozen round/teacher origin"):
            series.validate_record_origin(recipe, foreign)

    def test_checkpoint_chain_cannot_erase_ancestor_or_consume_sibling_teacher_rows(self):
        recipe = series.SeriesRecipe(recipe_value())
        common, sf, reckless = origin("common"), origin("sf", stage_id="sf-ft"), origin("reckless", stage_id="reckless-ft", teacher="reckless")
        index = series.GlobalLineageIndex(recipe, [common, sf, reckless])
        parent = checkpoint(recipe, [common])
        child = checkpoint(recipe, [common, sf], stage_id="sf-ft", parent=parent["checkpoint_sha256"])
        result = index.admit_checkpoint_lineage(recipe, child, ancestors=[parent])
        self.assertEqual(result["training_origins"], 2)
        with self.assertRaisesRegex(ValueError, "all common/branch ancestor"):
            index.admit_checkpoint_lineage(recipe, child, ancestors=[])
        erased = checkpoint(recipe, [sf], stage_id="sf-ft", parent=parent["checkpoint_sha256"])
        with self.assertRaisesRegex(ValueError, "erased ancestor"):
            index.admit_checkpoint_lineage(recipe, erased, ancestors=[parent])
        sibling = checkpoint(recipe, [common, reckless], stage_id="sf-ft", parent=parent["checkpoint_sha256"])
        with self.assertRaisesRegex(ValueError, "future or sibling branch"):
            index.admit_checkpoint_lineage(recipe, sibling, ancestors=[parent])

    def test_legacy_intake_reuses_head_evidence_and_keeps_input_masks_without_training(self):
        recipe, bundle = legacy_recipe(), legacy_bundle()
        before = copy.deepcopy(bundle)
        result = series.audit_legacy_bundle(recipe, bundle, stage_id="common", teacher_actor_id="stockfish",
            expected_model_sha256="4" * 64, expected_runtime_sha256="5" * 64)
        self.assertEqual(result["status"], series.LegacyIntakeStatus.AUDITED.value)
        self.assertEqual(result["role_records"]["proposer"], 1)
        row_result = result["head_observations"][0]
        self.assertEqual(row_result["input_sha256"], before["records"][0]["input_sha256"])
        self.assertEqual(row_result["action_support_mask"], before["records"][0]["input"]["action_support_mask"])
        self.assertEqual(row_result["heads"]["policy"]["source"], "stockfish19_completed_multipv")
        self.assertIs(row_result["heads"]["policy"]["masked"], False)
        self.assertIs(row_result["heads"]["wdl"]["masked"], True)
        self.assertIsNone(row_result["task_target_known_mask"])
        self.assertIn("legacy_bundle_lacks_native_physical_receipt_chain", result["limitations"])
        self.assertIs(result["actual_training_admissible"], False)
        self.assertEqual(result["optimizer_updates"], 0)
        self.assertEqual(bundle, before)

    def test_legacy_reckless_is_unsupported_and_tampered_provenance_is_rejected(self):
        recipe = legacy_recipe()
        options = dict(stage_id="common", expected_model_sha256="4" * 64, expected_runtime_sha256="5" * 64)
        unsupported = series.audit_legacy_bundle(recipe, None, teacher_actor_id="reckless", **options)
        self.assertEqual(unsupported["status"], series.LegacyIntakeStatus.UNSUPPORTED_TEACHER.value)
        for mutate in (lambda bundle: bundle["records"][0]["input"].update(capture_sequence=11),
                       lambda bundle: bundle["records"][0]["future_targets"]["policy"].update(source="reckless_completed_multipv"),
                       lambda bundle: bundle["allowed_evidence"].clear(),
                       lambda bundle: bundle["producer"].update(teacher_binary_sha256="f" * 64)):
            bundle = legacy_bundle()
            mutate(bundle)
            result = series.audit_legacy_bundle(recipe, bundle, teacher_actor_id="stockfish", **options)
            self.assertEqual(result["status"], series.LegacyIntakeStatus.REJECTED.value)
            self.assertEqual(result["code"], "LegacyProvenanceRejected")
            self.assertIs(result["execution_authorized"], False)

    def test_original_chunk_reader_and_transient_index_cleanup_remain_in_use(self):
        from test_generation_chunks import write_corpus
        from rz_pals_model import generation_data as data
        recipe, bundle = legacy_recipe(), legacy_bundle()
        with tempfile.TemporaryDirectory(prefix="rz-rovent-intake-") as temporary:
            root = Path(temporary)
            corpus, index = root / "corpus", root / "transient-index"
            corpus.mkdir()
            index.mkdir()
            manifest = write_corpus(corpus, bundle["records"], splits=bundle["opening_splits"], evidence=bundle["allowed_evidence"])
            result = series.audit_legacy_chunks(recipe, manifest, stage_id="common", teacher_actor_id="stockfish",
                expected_model_sha256="4" * 64, expected_runtime_sha256="5" * 64,
                expected_manifest_sha256=data.digest(manifest["schema"], manifest), corpus_root=corpus, index_root=index)
            self.assertEqual(result["status"], series.LegacyIntakeStatus.AUDITED.value)
            self.assertEqual(len(result["origins"]), 1)
            self.assertEqual(list(index.iterdir()), [])
            self.assertIs(result["execution_authorized"], False)

    def test_cli_pins_files_and_atomically_preserves_existing_output(self):
        with tempfile.TemporaryDirectory(prefix="rz-rovent-cli-") as temporary:
            root = Path(temporary)
            request = request_value(root, legacy_recipe())
            pin = json_pin(root, "request.json", request)
            args = ["--request", pin["path"], "--request-sha256", pin["sha256"]]
            out, err = io.StringIO(), io.StringIO()
            with redirect_stdout(out), redirect_stderr(err):
                self.assertEqual(cli.main(args), 0)
            printed = json.loads(out.getvalue())
            self.assertEqual(printed["status"], "preparation_saved")
            saved = root / request["output_name"]
            original = saved.read_bytes()
            result = json.loads(original)
            self.assertEqual(result["recipe_sha256"], legacy_recipe().sha256)
            self.assertIs(result["execution_authorized"], False)
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self.assertEqual(cli.main(args), 2)
            self.assertEqual(saved.read_bytes(), original)
            self.assertEqual(list(root.glob("*.partial")), [])

    def test_cli_legacy_files_and_exact_resume_metadata_have_real_consumers(self):
        from rz_pals_model.generation_data import digest
        with tempfile.TemporaryDirectory(prefix="rz-rovent-cli-intake-") as temporary:
            root = Path(temporary)
            recipe, bundle = legacy_recipe(), legacy_bundle()
            request = request_value(root, recipe, action="audit_legacy_intake")
            request["legacy_intake"] = {"format": "legacy_whole_json_bundle", "artifact": json_pin(root, "bundle.json", bundle),
                "semantic_sha256": digest("rz-rovent-original-legacy-bundle/1", bundle), "stage_id": "common",
                "teacher_actor_id": "stockfish", "expected_model_sha256": "4" * 64, "expected_runtime_sha256": "5" * 64,
                "corpus_root": None, "index_root": None}
            result = cli.prepare_request(request, budget=cli.PreparationBudget(max_wall_ns=10**9))
            self.assertEqual(result["result"]["status"], series.LegacyIntakeStatus.AUDITED.value)
            request.update(action="prepare_resume", legacy_intake=None, stage_id="common",
                checkpoint=json_pin(root, "checkpoint-metadata.json", checkpoint(recipe)))
            result = cli.prepare_request(request, budget=cli.PreparationBudget(max_wall_ns=10**9))
            self.assertEqual(result["result"]["mode"], "resume_exact_state")
            self.assertEqual(result["result"]["reset_state"], [])
            self.assertIs(result["execution_authorized"], False)

    def test_cli_rejects_wrong_hash_secret_path_capacity_and_expired_budget_before_loading(self):
        with tempfile.TemporaryDirectory(prefix="rz-rovent-cli-reject-") as temporary:
            root = Path(temporary)
            pin = json_pin(root, "metadata.json", {"answer": 1})
            bad_pin = dict(pin, sha256="f" * 64)
            with self.assertRaisesRegex(ValueError, "hash/identity mismatch"):
                cli.read_pinned_json(bad_pin, cli.PreparationBudget(max_wall_ns=10**9))
            with mock.patch.object(cli.os, "open", side_effect=AssertionError("forbidden path was opened")):
                with self.assertRaisesRegex(ValueError, "secret filename forbidden"):
                    cli.read_pinned_json(dict(pin, path=str(root / ".env.json")), cli.PreparationBudget(max_wall_ns=10**9))
                with self.assertRaisesRegex(ValueError, "bounded unsigned"):
                    cli.read_pinned_json(dict(pin, bytes=cli.MAX_JSON_BYTES + 1), cli.PreparationBudget(max_wall_ns=10**9))
            now = [0]
            budget = cli.PreparationBudget(max_wall_ns=1, clock=lambda: now[0])
            now[0] = 2
            with self.assertRaisesRegex(ValueError, "PreparationDeadlineExceeded"):
                cli.read_pinned_json(pin, budget)
            with self.assertRaisesRegex(ValueError, "output exceeds"):
                cli.write_result_noreplace(str(root), "too-large.json", {"result": "bounded"}, maximum=1)
            with self.assertRaisesRegex(ValueError, "must use a basename"):
                cli.write_result_noreplace(str(root), str(root.parent / "outside.json"), {}, maximum=100)
            self.assertFalse((root / "too-large.json").exists())
            self.assertEqual(list(root.glob("*.partial")), [])

    def test_explicit_managed_checkpoint_json_metadata_is_read_without_discovery_or_binary_load(self):
        with tempfile.TemporaryDirectory(prefix="rz-rovent-managed-metadata-") as temporary:
            root = Path(temporary)
            # These are newly created, tiny contract fixtures. No user model,
            # checkpoint or excluded generated directory is inspected.
            metadata_root = root / "checkpoints" / "managed-owner"
            metadata_root.mkdir(parents=True)
            recipe = legacy_recipe()
            metadata = checkpoint(recipe)
            metadata_pin = json_pin(metadata_root, "checkpoint-metadata.json", metadata)
            request = request_value(root, recipe, action="prepare_resume")
            request.update(checkpoint=metadata_pin, stage_id="common")
            result = cli.prepare_request(request, budget=cli.PreparationBudget(max_wall_ns=10**9))
            self.assertEqual(result["result"]["checkpoint"], metadata)
            self.assertEqual(result["input_pins"]["checkpoint"], metadata_pin)
            self.assertIs(result["execution_authorized"], False)
            self.assertEqual(list(metadata_root.iterdir()), [Path(metadata_pin["path"])])


if __name__ == "__main__":
    unittest.main()
