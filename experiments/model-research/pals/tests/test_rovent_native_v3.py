"""Synthetic submitted-review wire only; no Rules/native/neural authority."""
import copy
import importlib.abc
import sys
import unittest


class NeuralImportsForbidden(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"torch", "numpy", "onnx", "onnxruntime"}:
            raise AssertionError("raw /3 contracts cannot import a neural runtime")


_guard = NeuralImportsForbidden()
sys.meta_path.insert(0, _guard)
try:
    from rz_pals_model import generation_data as data
    from rz_pals_model import rovent_native_v2 as legacy
    from rz_pals_model import rovent_native_v3 as submitted
    from rz_pals_model import rovent_submitted_input as inputs
    from test_rovent_historical_corpus import synthetic_v2, h
    from test_rovent_submitted_input import fixture as input_fixture
finally:
    sys.meta_path.remove(_guard)


def metadata_fixture():
    return {"local_index": 0, "record_revision": 23, "parent_revision": None,
        "supersedes_revision": None, "origin_state": 17, "source_state": 18,
        "source_revision": 23, "source_line_offset": 0, "original_first_move": 192,
        "value": -17, "completed_depth": 8, "score_scope": "CompletedIteration", "critical": True}


def native_fixture():
    """Construct a new synthetic wire; never convert a real legacy corpus."""
    _recipe, manifest, record, _lock, _binding = synthetic_v2()
    manifest["schema"] = "rz-rovent-corpus-manifest/2"
    record["schema"] = "rz-rovent-training-record/2"
    snapshot = record["input"]
    snapshot.pop("divergence_record_ids")
    snapshot.update(schema=inputs.INPUT_DOMAIN, attack_mapping=[], action_support_mask=[False] * 7)
    record["input_sha256"] = inputs.seal_input(snapshot)
    native = record["native"]
    for name in ("raw_submission", "raw_completion", "raw_retirement"):
        native[name].update(schema=submitted.NATIVE_DOMAIN, input_sha256=record["input_sha256"])
    ticket = native["raw_submission"]["ticket"]
    native["raw_submission"]["submitted_review"] = {"schema": submitted.REVIEW_DOMAIN,
        "sequence": ticket["view_sequence"], "generation": native["root_generation"],
        "root_state": native["raw_retirement"]["root_admission"]["store_root_state"],
        "policy": {"mode": "anchored_compact", "token_budget": snapshot["selected_view"]["token_budget"],
            "max_records": 64, "max_anchors": 4, "required_bundle": None},
        "input_sha256": h("ReviewView semantic input distinct from numeric native input"),
        "view_seal_sha256": ticket["view_digest"], "record_scope_sha256": h("record scope"),
        "anchor_states": [native["raw_retirement"]["root_admission"]["store_root_state"]],
        "source_anchors": [], "record_metadata": []}
    for name in ("raw_submission", "raw_completion", "raw_retirement"):
        if name != "raw_submission":
            native[name]["submitted_sha256"] = native["raw_submission_sha256"]
        if name == "raw_retirement":
            native[name]["completed_sha256"] = native["raw_completion_sha256"]
        native[name + "_sha256"] = data.digest(submitted.NATIVE_DOMAIN, native[name])
    return manifest, record


def anchored_review_fixture():
    """Explicit historical StateIds; identical FENs cannot reconstruct them."""
    _manifest, record = native_fixture()
    snapshot = input_fixture(selected_counter=True)
    view, payload = snapshot["selected_view"], snapshot["encoded_model_input"]
    view["anchors"].extend(copy.deepcopy(view["anchors"][0]) for _ in range(2))
    payload["anchors"].extend(copy.deepcopy(payload["anchors"][0]) for _ in range(2))
    view["records"][0]["anchor"] = payload["records"][0]["anchor"] = 1
    second = copy.deepcopy(view["records"][0])
    second.update(record_id="second-explicit-old-idea", kind="repair", anchor=2, moves=[320])
    view["records"].append(second)
    encoded_second = copy.deepcopy(payload["records"][0])
    encoded_second.update(kind="repair", anchor=2, line=[[0, 5, 0]])
    payload["records"].append(encoded_second)
    snapshot["view_sha256"] = data.view_digest(view)
    record["input"] = snapshot
    review = record["native"]["raw_submission"]["submitted_review"]
    original_replay = record["native"]["raw_submission"]["rules_replay"]["actual_root"]
    review.update(anchor_states=[review["root_state"], 17, 19],
        source_anchors=[copy.deepcopy(original_replay) for _ in view["records"]],
        record_metadata=[metadata_fixture(), {**metadata_fixture(), "local_index": 1,
            "record_revision": 29, "source_revision": 29, "origin_state": 19, "original_first_move": 320}])
    return record, review


class SubmittedNativeTests(unittest.TestCase):
    def setUp(self):
        sys.meta_path.insert(0, _guard)

    def tearDown(self):
        sys.meta_path.remove(_guard)

    def test_new_raw_wire_keeps_numeric_and_semantic_view_identities_separate(self):
        manifest, record = native_fixture()
        before = copy.deepcopy(record)
        self.assertIs(submitted.validate_native_v3(manifest, record), record["native"])
        self.assertEqual(record, before)
        review = record["native"]["raw_submission"]["submitted_review"]
        self.assertNotEqual(review["input_sha256"], record["native"]["input_digest"])

    def test_new_and_legacy_raw_parser_reject_each_others_domains(self):
        manifest, record = native_fixture()
        with self.assertRaises(ValueError):
            legacy.validate_native_v2(manifest, record)
        _recipe, old_manifest, old_record, _lock, _binding = synthetic_v2()
        before = copy.deepcopy(old_record)
        legacy.validate_native_v2(old_manifest, old_record)
        with self.assertRaises(ValueError):
            submitted.validate_native_v3(old_manifest, old_record)
        self.assertEqual(old_record, before)

    def test_mixed_original_raw_versions_are_not_retagged(self):
        manifest, record = native_fixture()
        for name in ("raw_submission", "raw_completion", "raw_retirement"):
            bad = copy.deepcopy(record)
            bad["native"][name]["schema"] = legacy.NATIVE_DOMAIN
            with self.subTest(name=name), self.assertRaises(ValueError):
                submitted.validate_native_v3(manifest, bad)

    def test_submitted_review_sequence_and_view_seal_join_the_original_ticket(self):
        _manifest, record = native_fixture()
        value = record["native"]["raw_submission"]["submitted_review"]
        for change in ({"sequence": value["sequence"] + 1}, {"view_seal_sha256": h("other view")}):
            with self.subTest(change=tuple(change)), self.assertRaisesRegex(ValueError, "frozen ticket"):
                submitted.validate_submitted_review(record, {**value, **change})

    def test_submitted_review_shape_policy_and_source_coverage_are_closed(self):
        _manifest, record = native_fixture()
        value = record["native"]["raw_submission"]["submitted_review"]
        for mutate in (lambda item: item.update(extra=True),
                       lambda item: item["policy"].update(max_records=0),
                       lambda item: item["policy"].update(mode="current_root_only", max_anchors=4),
                       lambda item: item.update(source_anchors=[{}])):
            bad = copy.deepcopy(value)
            mutate(bad)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                submitted.validate_submitted_review(record, bad)

    def test_selected_counter_revision_uses_original_metadata_not_record_id_text(self):
        snapshot = input_fixture(selected_counter=True)
        metadata = [metadata_fixture()]
        before = copy.deepcopy(snapshot)
        submitted.validate_submitted_record_metadata(snapshot, metadata)
        self.assertEqual(snapshot, before)
        snapshot["selected_view"]["records"][0]["record_id"] = "arbitrary-record-id-without-a-revision"
        snapshot["view_sha256"] = data.view_digest(snapshot["selected_view"])
        submitted.validate_submitted_record_metadata(snapshot, metadata)
        with self.assertRaisesRegex(ValueError, "submitted source"):
            submitted.validate_submitted_record_metadata(snapshot, [{**metadata[0], "source_revision": 24}])

    def test_selected_metadata_order_shape_and_original_suffix_are_bounded(self):
        snapshot = input_fixture(selected_counter=True)
        for change in ({"local_index": 1}, {"source_line_offset": 256},
                       {"record_revision": True}, {"original_first_move": 0}, {"future_counter": True}):
            with self.subTest(change=tuple(change)), self.assertRaises(ValueError):
                submitted.validate_submitted_record_metadata(snapshot, [{**metadata_fixture(), **change}])

    def test_future_unselected_counter_is_not_inserted_into_original_mapping(self):
        snapshot = input_fixture()
        before = copy.deepcopy(snapshot)
        submitted.validate_submitted_record_metadata(snapshot, [])
        self.assertEqual(snapshot, before)
        self.assertIsNone(snapshot["attack_mapping"][0]["selected_counter_local_index"])
        with self.assertRaises(ValueError):
            submitted.validate_submitted_record_metadata(snapshot, [metadata_fixture()])

    def test_submitted_store_root_is_distinct_from_contract_state_owner(self):
        _manifest, record = native_fixture()
        review = record["native"]["raw_submission"]["submitted_review"]
        for change in ({"generation": review["generation"] + 1}, {"root_state": review["root_state"] + 1}):
            with self.subTest(change=tuple(change)), self.assertRaisesRegex(ValueError, "Store generation/root"):
                submitted.validate_submitted_review(record, {**review, **change})

    def test_anchor_states_are_required_without_fen_or_record_id_inference(self):
        record, review = anchored_review_fixture()
        before = copy.deepcopy((record, review))
        self.assertIs(submitted.validate_submitted_review(record, review), review)
        self.assertEqual((record, review), before)
        missing = copy.deepcopy(review)
        missing.pop("anchor_states")
        with self.assertRaisesRegex(ValueError, "ReviewView evidence fields mismatch"):
            submitted.validate_submitted_review(record, missing)
        # Same anchor boards and arbitrary record IDs do not supply StateIds.
        changed = copy.deepcopy(review)
        changed["anchor_states"] = [review["root_state"], 991, 997]
        with self.assertRaisesRegex(ValueError, "record origin StateId"):
            submitted.validate_submitted_review(record, changed)
        changed["record_metadata"][0]["origin_state"] = 991
        changed["record_metadata"][1]["origin_state"] = 997
        self.assertIs(submitted.validate_submitted_review(record, changed), changed)

    def test_anchor_state_list_has_exact_selected_anchor_count_and_plain_container(self):
        record, review = anchored_review_fixture()
        submitted.validate_submitted_review(record, review)
        for states in (None, tuple(review["anchor_states"]), [], review["anchor_states"][:-1],
                       [*review["anchor_states"], 23]):
            with self.subTest(states=states), self.assertRaisesRegex(ValueError, "anchor StateId allocation"):
                submitted.validate_submitted_review(record, {**review, "anchor_states": states})

    def test_anchor_state_ids_reject_bool_negative_overflow_and_noninteger(self):
        record, review = anchored_review_fixture()
        submitted.validate_submitted_review(record, review)
        for state in (True, False, -1, 2**64, 17.0, "17", None):
            bad = copy.deepcopy(review)
            bad["anchor_states"][1] = state
            with self.subTest(state=state), self.assertRaisesRegex(ValueError, "invalid bounded unsigned integer"):
                submitted.validate_submitted_review(record, bad)
        maximum = copy.deepcopy(review)
        maximum["anchor_states"][1] = maximum["record_metadata"][0]["origin_state"] = 2**64 - 1
        self.assertIs(submitted.validate_submitted_review(record, maximum), maximum)

    def test_anchor_state_zero_joins_store_root_and_nonroot_order_joins_record_origin(self):
        record, review = anchored_review_fixture()
        submitted.validate_submitted_review(record, review)
        wrong_root = copy.deepcopy(review)
        wrong_root["anchor_states"][0] += 1
        with self.assertRaisesRegex(ValueError, "anchor zero StateId"):
            submitted.validate_submitted_review(record, wrong_root)
        swapped = copy.deepcopy(review)
        swapped["anchor_states"][1:] = reversed(swapped["anchor_states"][1:])
        with self.assertRaisesRegex(ValueError, "record origin StateId"):
            submitted.validate_submitted_review(record, swapped)
        wrong_origin = copy.deepcopy(review)
        wrong_origin["record_metadata"][0]["origin_state"] += 1
        with self.assertRaisesRegex(ValueError, "record origin StateId"):
            submitted.validate_submitted_review(record, wrong_origin)

    def test_record_origin_state_ids_keep_strict_u64_type_before_anchor_lookup(self):
        record, review = anchored_review_fixture()
        submitted.validate_submitted_review(record, review)
        for state in (True, -1, 2**64):
            bad = copy.deepcopy(review)
            bad["record_metadata"][0]["origin_state"] = state
            with self.subTest(state=state), self.assertRaisesRegex(ValueError, "invalid bounded unsigned integer"):
                submitted.validate_submitted_review(record, bad)

    def test_original_selected_metadata_requires_all_thirteen_exact_fields(self):
        snapshot = input_fixture(selected_counter=True)
        original = metadata_fixture()
        submitted.validate_submitted_record_metadata(snapshot, [original])
        for field in ("value", "completed_depth", "score_scope", "critical"):
            bad = copy.deepcopy(original)
            bad.pop(field)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "selected-record metadata fields mismatch"):
                submitted.validate_submitted_record_metadata(snapshot, [bad])
        with self.assertRaisesRegex(ValueError, "selected-record metadata fields mismatch"):
            submitted.validate_submitted_record_metadata(snapshot, [{**original, "derived_wdl": [0.0, 1.0, 0.0]}])

    def test_raw_value_depth_scope_and_critical_keep_exact_original_types(self):
        snapshot = input_fixture(selected_counter=True)
        original = metadata_fixture()
        cases = (("value", True, "nullable i32"), ("value", -(2**31) - 1, "nullable i32"),
                 ("value", 2**31, "nullable i32"), ("value", 1.0, "nullable i32"),
                 ("completed_depth", True, "bounded unsigned integer"),
                 ("completed_depth", -1, "bounded unsigned integer"),
                 ("completed_depth", 65536, "bounded unsigned integer"),
                 ("score_scope", "completed_iteration", "exact supported variant"),
                 ("score_scope", True, "exact supported variant"),
                 ("score_scope", {}, "exact supported variant"),
                 ("critical", 0, "must be a bool"), ("critical", None, "must be a bool"))
        for field, value, message in cases:
            with self.subTest(field=field, value=value), self.assertRaisesRegex(ValueError, message):
                submitted.validate_submitted_record_metadata(snapshot, [{**original, field: value}])
        for value in (None, -(2**31), 2**31 - 1):
            valid = {**original, "value": value, "completed_depth": 65535, "critical": False}
            self.assertEqual(submitted.validate_submitted_record_metadata(snapshot, [valid]), [valid])

    def test_completed_iteration_is_the_original_scope_flag_without_changing_numeric_input(self):
        snapshot = input_fixture(selected_counter=True)
        original = metadata_fixture()
        for scope in (None, "FrontierOnly", "RulesTerminal"):
            changed = {**original, "score_scope": scope}
            with self.subTest(scope=scope), self.assertRaisesRegex(ValueError, "completed-iteration flag"):
                submitted.validate_submitted_record_metadata(snapshot, [changed])
            value = copy.deepcopy(snapshot)
            value["selected_view"]["records"][0]["completed_iteration"] = False
            value["encoded_model_input"]["records"][0]["completed_iteration"] = False
            value["view_sha256"] = data.view_digest(value["selected_view"])
            before = copy.deepcopy(value)
            self.assertEqual(submitted.validate_submitted_record_metadata(value, [changed]), [changed])
            self.assertEqual(value, before)

    def test_rules_terminal_is_independent_of_raw_cpu_score_scope_and_value(self):
        for scope, rules_terminal in ((None, True), ("RulesTerminal", False), ("FrontierOnly", True)):
            snapshot = input_fixture(selected_counter=True)
            for record in (snapshot["selected_view"]["records"][0], snapshot["encoded_model_input"]["records"][0]):
                record.update(completed_iteration=False, rules_terminal=rules_terminal)
            snapshot["view_sha256"] = data.view_digest(snapshot["selected_view"])
            original = {**metadata_fixture(), "score_scope": scope, "value": None, "completed_depth": 0}
            before = copy.deepcopy(snapshot)
            with self.subTest(scope=scope, rules_terminal=rules_terminal):
                self.assertEqual(submitted.validate_submitted_record_metadata(snapshot, [original]), [original])
                self.assertEqual(snapshot, before)


if __name__ == "__main__":
    unittest.main()
