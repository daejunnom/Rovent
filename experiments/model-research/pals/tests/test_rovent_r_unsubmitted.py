"""Synthetic absence/terminal wire joins; never an original capture authority."""
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
            raise AssertionError("R absence metadata must not import neural frameworks")


_guard = NeuralImportsForbidden()
sys.meta_path.insert(0, _guard)
try:
    from rz_pals_model import rovent_r_unsubmitted as absence
    from rz_pals_model import rovent_r_corpus as corpus
    from rz_pals_model.generation_data import digest
    from test_rovent_r_corpus import fixture, h
    from test_rovent_r_semantic import source_fixture
finally:
    sys.meta_path.remove(_guard)


def reseal(value):
    value["observations_sha256"] = ""
    value["observations_sha256"] = digest(absence.DOMAIN, value)
    return value


def make_observation(root, *, kind="unsubmitted", terminal=None):
    job = None if terminal is None else terminal["job"]
    return {"kind": kind, "frame_sequence": 1 if terminal is None else terminal["frame_sequence"],
        "question_sha256": h("unsubmitted original question") if job is None else job["question_sha256"],
        "full_view_sha256": h("unsubmitted full view"), "view_seal_sha256": h("unsubmitted view seal"),
        "binding_sha256": h("unsubmitted binding") if job is None else job["evaluation_namespace_sha256"],
        "capture_sequence": 1 if terminal is None else terminal["capture_sequence"],
        "original_deadline_clock_epoch": root["deadline_clock"], "original_deadline_ns": root["deadline_tick"],
        "job": copy.deepcopy(job), "worker_owner": None if terminal is None else terminal["worker_owner"],
        "retirement_sha256": None if terminal is None else digest(corpus.RETIREMENT_DOMAIN, terminal),
        "retirement": copy.deepcopy(terminal), "activation_observed": False}


def make_value(root, lock, parameter, *, publication_sha=None, observation=None):
    return reseal({"schema": absence.DOMAIN, "native_admission": copy.deepcopy(root),
        "native_admission_sha256": digest(corpus.ROOT_DOMAIN, root), "input_lock_sha256": lock["canonical_sha256"],
        "nnuec_weights_raw_sha256": lock["manifest"]["engine"]["nnuec"]["weights"]["sha256"],
        "nnuec_parameter_sha256": parameter,
        "authority": {key: root[key] for key in ("process_epoch", "game_generation", "root_generation", "implementation_sha256")},
        "r_publication_file_sha256": publication_sha,
        "observations": [make_observation(root) if observation is None else observation],
        "per_readout_consumption": False, "native_node_usage": None, "whole_action_cost": None,
        "wdl_target": None, "utility_target": None, "observations_sha256": ""})


def unsubmitted_fixture():
    _recipe, _manifest, row, lock = fixture()
    return make_value(row["native_admission"], lock, row["bound"]["parameter_sha256"]), lock


def published_fixture(root, *, not_started=True):
    source, budget, publication, _rows = source_fixture(root, empty=True, not_started=not_started)
    entry = publication["admissions"][0]
    lock = json.loads((root / entry["input_lock"]["path"]).read_bytes())
    original = source.original_publication_metadata()
    declaration = publication["retirements"][0]
    body = source.original_retirement_evidence(declaration["retirement_sha256"])["observation"]
    root_body = body["native_admission"]
    observation = make_observation(root_body, kind="not_started", terminal=body["retirement"])
    value = make_value(root_body, lock, entry["nnuec_parameter_sha256"],
        publication_sha=original["publication_file"]["sha256"], observation=observation)
    return value, lock, source, budget


class RUnsubmittedObservationTests(unittest.TestCase):
    def setUp(self):
        sys.meta_path.insert(0, _guard)

    def tearDown(self):
        sys.meta_path.remove(_guard)

    def test_unsubmitted_without_publication_preserves_unknown_assets_and_no_job(self):
        value, lock = unsubmitted_fixture()
        checked = absence.validate_r_unsubmitted_observations(value, input_lock=lock)
        self.assertEqual(checked.to_mapping(), value)
        report = checked.observation()
        self.assertEqual(report["unsubmitted_count"], 1)
        self.assertEqual(report["not_started_count"], 0)
        self.assertEqual(report["nnuec_asset_bytes_scope"], "unknown_without_publication")
        self.assertIsNone(checked.to_mapping()["observations"][0]["job"])
        for name in ("nnuec_parameter_identity_recomputed", "original_manager_custody_verified", "original_native_authority_reissued",
                "past_physical_completion_reissued", "actual_rules_replay_verified", "execution_authorized", "training_source_authorized",
                "training_target_authorized", "input_rows_issued", "per_readout_consumption"):
            self.assertIs(report[name], False)
        for name in ("native_node_usage", "whole_action_cost", "wdl_target", "utility_target"):
            self.assertIsNone(report[name])

    def test_not_started_joins_original_empty_terminal_body_and_preserves_sampler_and_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            value, lock, source, budget = published_fixture(Path(directory))
            reads = budget.observation()
            pool = source.corpus.pool_metadata()
            before = copy.deepcopy(value)
            checked = absence.validate_r_unsubmitted_observations(value, input_lock=lock, source=source)
            self.assertEqual(checked.to_mapping(), before)
            self.assertEqual(checked.observation()["not_started_count"], 1)
            self.assertEqual(checked.observation()["nnuec_asset_bytes_scope"], "verified_historical_source_bytes")
            self.assertEqual(len(source.corpus.rows()), 0)
            self.assertEqual(source.corpus.pool_metadata(), pool)
            self.assertEqual(budget.observation(), reads)

    def test_unsubmitted_can_share_original_publication_without_inventing_terminal(self):
        with tempfile.TemporaryDirectory() as directory:
            value, lock, source, _budget = published_fixture(Path(directory))
            value["observations"] = [make_observation(value["native_admission"])]
            checked = absence.validate_r_unsubmitted_observations(reseal(value), input_lock=lock, source=source)
            self.assertIsNone(checked.to_mapping()["observations"][0]["retirement"])

    def test_unsubmitted_rejects_invented_job_worker_or_terminal(self):
        for name, bad in (("job", {}), ("worker_owner", 0), ("retirement_sha256", h("invented terminal")), ("retirement", {})):
            with self.subTest(name=name):
                value, lock = unsubmitted_fixture()
                value["observations"][0][name] = bad
                with self.assertRaises(ValueError):
                    absence.validate_r_unsubmitted_observations(reseal(value), input_lock=lock)

    def test_not_started_requires_typed_owner_and_original_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            value, lock, source, _budget = published_fixture(Path(directory))
            for supplied in (None, source.original_publication_metadata(), object()):
                with self.subTest(supplied=type(supplied).__name__), self.assertRaises(ValueError):
                    absence.validate_r_unsubmitted_observations(value, input_lock=lock, source=supplied)
            value["r_publication_file_sha256"] = None
            with self.assertRaises(ValueError):
                absence.validate_r_unsubmitted_observations(reseal(value), input_lock=lock, source=source)

    def test_original_raw_weights_parameter_and_publication_sha_are_distinct_pins(self):
        with tempfile.TemporaryDirectory() as directory:
            value, lock, source, _budget = published_fixture(Path(directory))
            self.assertNotEqual(value["nnuec_weights_raw_sha256"], value["nnuec_parameter_sha256"])
            for name in ("native_admission_sha256", "input_lock_sha256", "nnuec_weights_raw_sha256", "nnuec_parameter_sha256",
                    "r_publication_file_sha256"):
                with self.subTest(name=name):
                    bad = copy.deepcopy(value)
                    bad[name] = h("wrong " + name)
                    with self.assertRaises(ValueError):
                        absence.validate_r_unsubmitted_observations(reseal(bad), input_lock=lock, source=source)

    def test_not_started_original_terminal_kind_body_digest_and_empty_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            value, lock, source, _budget = published_fixture(Path(directory), not_started=False)
            with self.assertRaises(ValueError):
                absence.validate_r_unsubmitted_observations(value, input_lock=lock, source=source)
        with tempfile.TemporaryDirectory() as directory:
            value, lock, source, _budget = published_fixture(Path(directory))
            for name, bad_value in (("retirement", None), ("retirement_sha256", h("wrong terminal")), ("worker_owner", 1)):
                with self.subTest(name=name):
                    bad = copy.deepcopy(value)
                    bad["observations"][0][name] = bad_value
                    with self.assertRaises(ValueError):
                        absence.validate_r_unsubmitted_observations(reseal(bad), input_lock=lock, source=source)

    def test_same_capture_sequence_is_rejected_but_distinct_occurrences_are_preserved(self):
        value, lock = unsubmitted_fixture()
        value["observations"].append(copy.deepcopy(value["observations"][0]))
        with self.assertRaises(ValueError):
            absence.validate_r_unsubmitted_observations(reseal(value), input_lock=lock)
        value["observations"][1]["capture_sequence"] = 2
        self.assertEqual(absence.validate_r_unsubmitted_observations(reseal(value), input_lock=lock).observation()["observation_count"], 2)

    def test_frame_capture_clock_deadline_and_activation_are_strict(self):
        for name, bad in (("frame_sequence", 0), ("capture_sequence", 0), ("original_deadline_clock_epoch", 2),
                ("original_deadline_ns", 1001), ("activation_observed", True), ("activation_observed", 0)):
            with self.subTest(name=name, bad=bad):
                value, lock = unsubmitted_fixture()
                value["observations"][0][name] = bad
                with self.assertRaises(ValueError):
                    absence.validate_r_unsubmitted_observations(reseal(value), input_lock=lock)

    def test_plain_uint_rejects_bool_negative_float_and_overflow_in_nested_body(self):
        for name in ("frame_sequence", "capture_sequence", "original_deadline_clock_epoch", "original_deadline_ns"):
            for bad in (True, -1, 1.0, 2**64):
                with self.subTest(name=name, bad=bad):
                    value, lock = unsubmitted_fixture()
                    value["observations"][0][name] = bad
                    with self.assertRaises(ValueError):
                        absence.validate_r_unsubmitted_observations(reseal(value), input_lock=lock)
        with tempfile.TemporaryDirectory() as directory:
            value, lock, source, _budget = published_fixture(Path(directory))
            for name, bad in (("worker_owner", True), ("worker_owner", 65536), ("retained_readouts", True), ("attempted", 2**64)):
                candidate = copy.deepcopy(value)
                candidate["observations"][0]["retirement"][name] = bad
                with self.subTest(name=name), self.assertRaises(ValueError):
                    absence.validate_r_unsubmitted_observations(reseal(candidate), input_lock=lock, source=source)

    def test_unknown_fields_domains_and_targets_cannot_issue_authority_or_zero_usage(self):
        for scope in ("top", "observation", "root", "origin", "lock"):
            value, lock = unsubmitted_fixture()
            target = {"top": value, "observation": value["observations"][0], "root": value["native_admission"],
                "origin": value["native_admission"]["native_origin"], "lock": lock}[scope]
            target["execution_authorized"] = True
            with self.subTest(scope=scope), self.assertRaises(ValueError):
                absence.validate_r_unsubmitted_observations(reseal(value), input_lock=lock)
        for name, bad in (("native_node_usage", 0), ("whole_action_cost", {}), ("wdl_target", [0, 1, 0]),
                ("utility_target", {}), ("per_readout_consumption", True), ("per_readout_consumption", 0), ("schema", corpus.ROOT_DOMAIN)):
            value, lock = unsubmitted_fixture()
            value[name] = bad
            with self.subTest(name=name), self.assertRaises(ValueError):
                absence.validate_r_unsubmitted_observations(reseal(value), input_lock=lock)

    def test_original_lock_root_source_and_descriptor_are_not_relabelled(self):
        value, lock = unsubmitted_fixture()
        bad_lock = copy.deepcopy(lock)
        bad_lock["manifest"]["coordinator"]["cpu_credit_limit"] += 1
        with self.assertRaises(ValueError):
            absence.validate_r_unsubmitted_observations(value, input_lock=bad_lock)
        for name in ("root_budget_sha256", "root_descriptor_sha256", "native_origin_sha256"):
            bad = copy.deepcopy(value)
            bad["native_admission"][name] = h(name)
            bad["native_admission_sha256"] = digest(corpus.ROOT_DOMAIN, bad["native_admission"])
            with self.subTest(name=name), self.assertRaises(ValueError):
                absence.validate_r_unsubmitted_observations(reseal(bad), input_lock=lock)
        value["native_admission"]["schema"] = "rz-rovent-collection-root-admission/2"
        value["native_admission_sha256"] = digest(value["native_admission"]["schema"], value["native_admission"])
        with self.assertRaises(ValueError):
            absence.validate_r_unsubmitted_observations(reseal(value), input_lock=lock)

    def test_strict_json_rejects_duplicate_nested_keys_nonfinite_and_invalid_utf8(self):
        value, lock = unsubmitted_fixture()
        text = json.dumps(value)
        self.assertEqual(absence.parse_r_unsubmitted_observations(text, input_lock=lock).to_mapping(), value)
        for text_bad in (text.replace('"activation_observed": false', '"activation_observed": false, "activation_observed": false'),
                text.replace('"native_node_usage": null', '"native_node_usage": NaN'),
                text.replace('"native_node_usage": null', '"native_node_usage": Infinity'),
                text.replace('"native_node_usage": null', '"native_node_usage": -Infinity'), "\ud800"):
            with self.subTest(prefix=text_bad[:50]), self.assertRaises(ValueError):
                absence.parse_r_unsubmitted_observations(text_bad, input_lock=lock)

    def test_bounded_count_original_utf8_bytes_and_digest(self):
        value, lock = unsubmitted_fixture()
        for values in ([], [copy.deepcopy(value["observations"][0])] * 65):
            bad = copy.deepcopy(value)
            bad["observations"] = values
            with self.assertRaises(ValueError):
                absence.validate_r_unsubmitted_observations(reseal(bad), input_lock=lock)
        value["observations_sha256"] = h("wrong digest")
        with self.assertRaises(ValueError):
            absence.validate_r_unsubmitted_observations(value, input_lock=lock)
        with self.assertRaises(ValueError):
            absence.parse_r_unsubmitted_observations(" " * (absence.MAX_BYTES + 1), input_lock=lock)

    def test_result_is_detached_and_constructor_does_not_accept_json_authority(self):
        value, lock = unsubmitted_fixture()
        checked = absence.validate_r_unsubmitted_observations(value, input_lock=lock)
        original = checked.to_mapping()
        value["observations"][0]["question_sha256"] = h("later mutation")
        lock["manifest"]["engine"]["mode"] = "cpu_pc"
        detached = checked.to_mapping()
        detached["observations"].clear()
        self.assertEqual(checked.to_mapping(), original)
        with self.assertRaises(TypeError):
            absence.RoventRUnsubmittedObservation(original)

    def test_validation_and_result_readback_do_not_open_paths_or_issue_terminal_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            value, lock, source, budget = published_fixture(Path(directory))
            before = budget.observation()
            with mock.patch("builtins.open", side_effect=AssertionError("metadata validation must not open files")):
                checked = absence.validate_r_unsubmitted_observations(value, input_lock=lock, source=source)
                checked.to_mapping()
                checked.observation()
            self.assertEqual(before, budget.observation())
            self.assertEqual(len(source.corpus.rows()), 0)


if __name__ == "__main__":
    unittest.main()
