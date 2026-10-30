"""Synthetic R semantic metadata joins; no original capture or target rights."""
import copy
import importlib.abc
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


class NeuralImportsForbidden(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"torch", "numpy", "onnx", "onnxruntime", "tensorflow", "jax"}:
            raise AssertionError("R semantic metadata must not import neural frameworks")


_guard = NeuralImportsForbidden()
sys.meta_path.insert(0, _guard)
try:
    from rz_pals_model import rovent_r_semantic as semantic
    from rz_pals_model import rovent_historical_r_corpus as historical
    from rz_pals_model import rovent_historical_corpus as common
    from rz_pals_model import rovent_r_corpus as corpus
    from rz_pals_model.generation_data import digest
    from test_rovent_historical_r_corpus import write_publication, reseal_publication, file_pin
    from test_rovent_r_corpus import h, raw
finally:
    sys.meta_path.remove(_guard)


def reseal(value):
    value["semantic_sha256"] = ""
    value["semantic_sha256"] = digest("rz-rovent-r-semantic-observations/1", value)
    return value


def source_fixture(root, *, empty=False, not_started=False):
    recipe, request, publication, rows = write_publication(root, empty=empty)
    if not_started:
        if not empty:
            raise ValueError("NotStarted fixture has no readouts")
        entry = publication["retirements"][0]
        body = json.loads((root / entry["file"]["path"]).read_bytes())
        old_bytes = entry["file"]["bytes"]
        body["retirement"]["ending"] = {"kind": "not_started"}
        identity = digest(corpus.RETIREMENT_DOMAIN, body["retirement"])
        body["retirement_sha256"] = identity
        payload = raw(body)
        path = "r-retirements/" + identity + ".json"
        (root / path).write_bytes(payload)
        publication["retirements"] = [{"retirement_sha256": identity, "file": file_pin(path, payload),
            "readout_record_sha256": []}]
        publication["total_payload_bytes"] += len(payload) - old_bytes
        reseal_publication(root, request, publication)
    budget = common.HistoricalReadBudget(64 << 20)
    source = historical.HistoricalRoventRCorpus.from_registration(request, recipe=recipe, read_budget=budget)
    return source, budget, publication, rows


def supplement(source, *, stages=("raw_published", "search_feedback_consumed", "selected_pc_view_consumed")):
    original = source.original_publication_metadata()
    publication = original["publication"]
    declaration = publication["retirements"][0]
    identity = declaration["retirement_sha256"]
    body = source.original_retirement_evidence(identity)["observation"]
    retired = body["retirement"]
    authority, job = retired["job"]["authority"], retired["job"]
    model = publication["admissions"][0]["nnuec_registration"]["model_identity"]
    task = {"state": 91, "line": None, "question": "AnalyzePosition", "root_moves": [], "model": 0, "epoch": 0,
        "value_identity": copy.deepcopy(model), "checker_identity": None, "cpu_condition": "synthetic-original-nnuec-condition",
        "profile": 7, "condition": 9, "input_revision": 0, "requested_depth": 1,
        "node_budget": job["total_native_node_credit"]}
    record_ids = declaration["readout_record_sha256"]
    bound = source.corpus.original_semantic_evidence(record_ids[0])["bound"] if record_ids else None
    base = {"retirement_sha256": identity, "native_execution_epoch": authority["process_epoch"], "native_execution_sequence": 5,
        "store_execution": 0, "observation": 0, "task_key": task,
        "task_key_sha256": digest("rz-rovent-r-semantic-task/1", task), "job_sequence": job["job_sequence"],
        "worker_owner": retired["worker_owner"], "frame_sequence": retired["frame_sequence"], "capture_sequence": retired["capture_sequence"],
        "question_sha256": job["question_sha256"], "binding_sha256": job["evaluation_namespace_sha256"],
        "submitted_view_sha256": h("synthetic-empty-capture-view") if bound is None else bound["view_seal_sha256"],
        "submitted_full_view_sha256": h("synthetic-empty-capture-full-view") if bound is None else bound["full_view_sha256"],
        "native_slice_work": {"nodes": 1, "quiescence_nodes": 0, "tt_hits": 0}, "selected_pc": None}
    facts = []
    for stage in stages:
        observation = copy.deepcopy(base)
        observation["stage"] = stage
        if stage == "selected_pc_view_consumed":
            observation["selected_pc"] = {"request_epoch": authority["process_epoch"], "request_sequence": 17,
                "question_sha256": h("synthetic-selected-question"), "view_sha256": h("synthetic-selected-view"),
                "full_view_sha256": h("synthetic-selected-full-view"), "output_sha256": h("synthetic-selected-output"),
                "record_revision": 0, "record_origin_state": 0}
        facts.append(observation)
    return reseal({"schema": semantic.DOMAIN, "r_publication_file_sha256": original["publication_file"]["sha256"],
        "native_admission_sha256": body["native_admission_sha256"], "authority": copy.deepcopy(authority),
        "retirements": [{"retirement_sha256": identity, "retirement": copy.deepcopy(retired),
            "readout_record_sha256": list(record_ids)}], "observations": facts, "unrecorded_witness_attempts": 0,
        "manager_failures": [], "per_readout_consumption": False, "whole_action_cost": None,
        "wdl_target": None, "utility_target": None, "semantic_sha256": ""})


class RSemanticObservationTests(unittest.TestCase):
    def setUp(self):
        sys.meta_path.insert(0, _guard)

    def tearDown(self):
        sys.meta_path.remove(_guard)

    def test_original_publication_rows_and_three_stages_remain_metadata_only(self):
        with tempfile.TemporaryDirectory() as directory:
            source, budget, publication, rows = source_fixture(Path(directory))
            before = copy.deepcopy(rows)
            sample_before = source.corpus.training_sample(source.corpus.rows()[0])
            pool_before = source.corpus.pool_metadata()
            reads_before = budget.observation()
            value = supplement(source)
            checked = semantic.validate_r_semantic_observations(value, source=source)
            report = checked.observation()
            self.assertEqual(checked.to_mapping(), value)
            self.assertEqual(report["distinct_attested_execution_count"], 1)
            self.assertEqual(report["retained_readout_reference_count"], 1)
            self.assertEqual(report["stages"], {"raw_published": 1, "search_feedback_consumed": 1, "selected_pc_view_consumed": 1})
            for name in ("execution_authorized", "training_target_authorized", "original_native_authority_reissued",
                    "past_physical_completion_reissued", "actual_rules_replay_verified", "per_readout_consumption"):
                self.assertIs(report[name], False)
            for name in ("whole_action_cost", "wdl_target", "utility_target"):
                self.assertIsNone(report[name])
            self.assertEqual(source.corpus.training_sample(source.corpus.rows()[0]), sample_before)
            self.assertEqual(source.corpus.pool_metadata(), pool_before)
            self.assertEqual(rows, before)
            self.assertEqual(budget.observation(), reads_before)
            self.assertEqual(source.original_publication_metadata()["publication"], publication)

    def test_empty_failed_capture_is_preserved_without_fabricated_input_row(self):
        with tempfile.TemporaryDirectory() as directory:
            source, _budget, _publication, _rows = source_fixture(Path(directory), empty=True)
            value = supplement(source, stages=())
            checked = semantic.validate_r_semantic_observations(value, source=source)
            self.assertEqual(checked.observation()["empty_retirement_count"], 1)
            self.assertEqual(checked.observation()["retained_readout_reference_count"], 0)
            self.assertEqual(checked.observation()["distinct_attested_execution_count"], 0)
            self.assertEqual(len(source.corpus.rows()), 0)
            terminal = source.original_retirement_evidence(value["retirements"][0]["retirement_sha256"])
            self.assertEqual(terminal["observation"]["retirement"]["ending"], {"kind": "failed"})

    def test_not_started_terminal_cannot_acquire_a_semantic_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            source, _budget, _publication, _rows = source_fixture(Path(directory), empty=True, not_started=True)
            semantic.validate_r_semantic_observations(supplement(source, stages=()), source=source)
            with self.assertRaisesRegex(ValueError, "Job/frame/capture/physical execution/work"):
                semantic.validate_r_semantic_observations(supplement(source, stages=("raw_published",)), source=source)

    def test_same_execution_work_is_not_duplicated_and_selected_tickets_are_distinct(self):
        with tempfile.TemporaryDirectory() as directory:
            source, *_ = source_fixture(Path(directory))
            value = supplement(source)
            second = copy.deepcopy(value["observations"][-1])
            second["selected_pc"]["request_sequence"] += 1
            value["observations"].append(second)
            checked = semantic.validate_r_semantic_observations(reseal(value), source=source)
            self.assertEqual(checked.observation()["distinct_attested_execution_count"], 1)
            self.assertEqual(checked.observation()["stages"]["selected_pc_view_consumed"], 2)
            value["observations"][-1]["native_slice_work"]["nodes"] += 1
            with self.assertRaisesRegex(ValueError, "same native execution evidence or slice work"):
                semantic.validate_r_semantic_observations(reseal(value), source=source)

    def test_stage_order_and_duplicate_occurrences_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            source, *_ = source_fixture(Path(directory))
            for stages, error in ((("search_feedback_consumed",), "feedback requires"),
                    (("raw_published", "selected_pc_view_consumed"), "requires original feedback"),
                    (("raw_published", "raw_published"), "RawPublished must be first"),
                    (("raw_published", "search_feedback_consumed", "selected_pc_view_consumed", "selected_pc_view_consumed"), "duplicate R semantic")):
                with self.subTest(stages=stages), self.assertRaisesRegex(ValueError, error):
                    semantic.validate_r_semantic_observations(supplement(source, stages=stages), source=source)

    def test_full_retirement_and_readout_coverage_cannot_be_selected_partially(self):
        with tempfile.TemporaryDirectory() as directory:
            source, *_ = source_fixture(Path(directory))
            original = supplement(source)
            for change, error in ((lambda value: value.__setitem__("retirements", []), "complete publication retirement coverage"),
                    (lambda value: value["retirements"][0].__setitem__("readout_record_sha256", []), "whole retirement count/seal/scope"),
                    (lambda value: value["retirements"][0].__setitem__("retirement_sha256", h("unknown-retirement")), "retirement is absent"),
                    (lambda value: value["retirements"][0]["readout_record_sha256"].append(h("foreign-readout")), "sorted unique|count/seal/scope|whole retirement/readout")):
                value = copy.deepcopy(original)
                change(value)
                with self.subTest(error=error), self.assertRaisesRegex(ValueError, error):
                    semantic.validate_r_semantic_observations(reseal(value), source=source)

    def test_publication_admission_authority_and_original_retirement_pins_are_joined(self):
        with tempfile.TemporaryDirectory() as directory:
            source, *_ = source_fixture(Path(directory))
            for key, error in (("r_publication_file_sha256", "raw publication file pin"),
                    ("native_admission_sha256", "admission is absent")):
                value = supplement(source)
                value[key] = h("other-" + key)
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, error):
                    semantic.validate_r_semantic_observations(reseal(value), source=source)
            value = supplement(source)
            value["authority"]["root_generation"] += 1
            with self.assertRaisesRegex(ValueError, "whole retirement/readout/original admission"):
                semantic.validate_r_semantic_observations(reseal(value), source=source)
            value = supplement(source)
            value["retirements"][0]["retirement"]["job"]["hard_deadline_ns"] += 1
            with self.assertRaisesRegex(ValueError, "whole retirement count/seal/scope"):
                semantic.validate_r_semantic_observations(reseal(value), source=source)

    def test_each_job_capture_binding_and_bound_view_is_joined(self):
        with tempfile.TemporaryDirectory() as directory:
            source, *_ = source_fixture(Path(directory))
            numeric = ("native_execution_epoch", "job_sequence", "worker_owner", "frame_sequence", "capture_sequence")
            for name in (*numeric, "question_sha256", "binding_sha256", "submitted_view_sha256", "submitted_full_view_sha256"):
                value = supplement(source)
                item = value["observations"][0]
                item[name] = item[name] + 1 if name in numeric else h("changed-" + name)
                with self.subTest(name=name), self.assertRaisesRegex(ValueError, "Job/frame/capture/physical execution/work|original readout Bound"):
                    semantic.validate_r_semantic_observations(reseal(value), source=source)

    def test_store_observation_and_task_identity_are_invariant_across_stages(self):
        with tempfile.TemporaryDirectory() as directory:
            source, *_ = source_fixture(Path(directory))
            for name in ("store_execution", "observation"):
                value = supplement(source)
                value["observations"][1][name] += 1
                with self.subTest(name=name), self.assertRaisesRegex(ValueError, "same native execution evidence"):
                    semantic.validate_r_semantic_observations(reseal(value), source=source)
            value = supplement(source)
            task = value["observations"][1]["task_key"]
            task["state"] += 1
            value["observations"][1]["task_key_sha256"] = digest(semantic.TASK_DOMAIN, task)
            with self.assertRaisesRegex(ValueError, "same native execution evidence"):
                semantic.validate_r_semantic_observations(reseal(value), source=source)

    def test_task_body_unknown_fields_enums_and_native_model_credit_are_strict(self):
        with tempfile.TemporaryDirectory() as directory:
            source, *_ = source_fixture(Path(directory))
            cases = ((lambda task: task.__setitem__("unknown", 1), "TaskKey fields"),
                (lambda task: task.__setitem__("question", "FutureQuestion"), "unknown original TaskQuestion"),
                (lambda task: task.__setitem__("question", {"FindAlternative": {"divergence_ply": 1, "extra": 0}}), "alternative task question fields"),
                (lambda task: task.__setitem__("line", 0), "TaskKey/model/Job credit"),
                (lambda task: task.__setitem__("model", 1), "TaskKey/model/Job credit"),
                (lambda task: task.__setitem__("checker_identity", {"kind": "owned", "identity": task["value_identity"]}), "TaskKey/model/Job credit"),
                (lambda task: task.__setitem__("node_budget", task["node_budget"] + 1), "TaskKey/model/Job credit"),
                (lambda task: task["value_identity"].__setitem__("weights_sha256", h("other-model")), "TaskKey/model/Job credit"))
            for change, error in cases:
                value = supplement(source)
                item = value["observations"][0]
                change(item["task_key"])
                item["task_key_sha256"] = digest(semantic.TASK_DOMAIN, item["task_key"])
                with self.subTest(error=error), self.assertRaisesRegex(ValueError, error):
                    semantic.validate_r_semantic_observations(reseal(value), source=source)

    def test_task_and_whole_document_byte_caps_are_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            source, *_ = source_fixture(Path(directory))
            value = supplement(source)
            value["observations"][0]["task_key"]["cpu_condition"] = "x" * (64 << 10)
            value["observations"][0]["task_key_sha256"] = digest(semantic.TASK_DOMAIN, value["observations"][0]["task_key"])
            with self.assertRaisesRegex(ValueError, "TaskKey byte cap"):
                semantic.validate_r_semantic_observations(reseal(value), source=source)
            value = supplement(source)
            value["manager_failures"] = [{"stage": "Settled", "code": "fixture", "detail": "x" * (4 << 20)}]
            with self.assertRaisesRegex(ValueError, "supplement byte cap"):
                semantic.validate_r_semantic_observations(reseal(value), source=source)

    def test_boolean_float_and_overflow_cannot_replace_native_integer_units(self):
        with tempfile.TemporaryDirectory() as directory:
            source, *_ = source_fixture(Path(directory))
            for target, key in (("observation", "worker_owner"), ("observation", "store_execution"),
                    ("task_key", "requested_depth"), ("native_slice_work", "nodes")):
                for wrong in (True, 1.0, -1, 2**64):
                    value = supplement(source)
                    item = value["observations"][0]
                    owner = item if target == "observation" else item[target]
                    owner[key] = wrong
                    if target == "task_key":
                        item["task_key_sha256"] = digest(semantic.TASK_DOMAIN, item["task_key"])
                    with self.subTest(target=target, key=key, wrong=wrong), self.assertRaisesRegex(ValueError, "unsigned integer"):
                        semantic.validate_r_semantic_observations(reseal(value), source=source)

    def test_target_claims_false_readout_consumption_and_manager_failures_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            source, *_ = source_fixture(Path(directory))
            for key in ("whole_action_cost", "wdl_target", "utility_target", "per_readout_consumption"):
                value = supplement(source)
                value[key] = True
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, "unknown target and per-readout scope"):
                    semantic.validate_r_semantic_observations(reseal(value), source=source)
            value = supplement(source)
            value["unrecorded_witness_attempts"] = 3
            value["manager_failures"] = [{"stage": "Settled", "code": "synthetic-failure", "detail": "original\nunknown tail"}]
            checked = semantic.validate_r_semantic_observations(reseal(value), source=source)
            self.assertEqual(checked.to_mapping()["manager_failures"], value["manager_failures"])
            self.assertEqual(checked.observation()["unrecorded_witness_attempts"], 3)
            self.assertIsNone(checked.observation()["whole_action_cost"])

    def test_json_duplicates_nonfinite_and_outer_digest_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            source, *_ = source_fixture(Path(directory))
            value = supplement(source)
            raw_text = json.dumps(value, separators=(",", ":"))
            self.assertEqual(semantic.parse_r_semantic_observations(raw_text, source=source).to_mapping(), value)
            for text in (raw_text.replace('"store_execution":0', '"store_execution":0,"store_execution":0', 1),
                    raw_text.replace('"unrecorded_witness_attempts":0', '"unrecorded_witness_attempts":NaN', 1)):
                with self.subTest(text=text), self.assertRaisesRegex(ValueError, "duplicate|nonfinite"):
                    semantic.parse_r_semantic_observations(text, source=source)
            value["semantic_sha256"] = h("other-supplement")
            with self.assertRaisesRegex(ValueError, "supplement digest mismatch"):
                semantic.validate_r_semantic_observations(value, source=source)

    def test_detached_getters_and_result_do_not_issue_or_mutate_original_authority(self):
        with tempfile.TemporaryDirectory() as directory:
            source, budget, _publication, rows = source_fixture(Path(directory))
            reads = budget.observation()
            record_sha = rows[0]["record_sha256"]
            original_rpc = source.corpus.original_auxiliary_evidence(record_sha)
            detached = source.corpus.original_semantic_evidence(record_sha)
            detached["question"]["job"]["job_sequence"] += 99
            self.assertEqual(source.corpus.original_auxiliary_evidence(record_sha), original_rpc)
            self.assertFalse(detached["execution_authorized"])
            value = supplement(source)
            checked = semantic.validate_r_semantic_observations(value, source=source)
            value["observations"].clear()
            changed = checked.to_mapping()
            changed["observations"].clear()
            self.assertEqual(len(checked.to_mapping()["observations"]), 3)
            self.assertEqual(budget.observation(), reads)
            with self.assertRaisesRegex(ValueError, "typed historical R corpus"):
                semantic.validate_r_semantic_observations(checked.to_mapping(), source=source.observation())

    def test_retirement_lookup_detects_mutation_and_metadata_join_performs_no_new_io(self):
        with tempfile.TemporaryDirectory() as directory:
            source, *_ = source_fixture(Path(directory))
            value = supplement(source)
            with mock.patch("builtins.open", side_effect=AssertionError("no unpinned source reopens")):
                semantic.validate_r_semantic_observations(value, source=source)
            identity = value["retirements"][0]["retirement_sha256"]
            source._retirements[identity]["retirement"]["attempted"] += 1
            with self.assertRaisesRegex(ValueError, "retirement evidence changed"):
                source.original_retirement_evidence(identity)


if __name__ == "__main__":
    unittest.main()
