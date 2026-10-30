"""Synthetic root-local wires, never actual native/physical/utility evidence.

Expected hash frames below are independently copied from utility_owner.rs,
utility_route.rs and native_acceptance.rs. The descriptor and component pins
are synthetic metadata; no source artifact, Rules engine, model or checkpoint
is loaded. Root-local /1 observations must remain unusable as known pairs.
"""
import copy
import hashlib
import importlib.abc
import json
import sys
import unittest
from unittest import mock


class NeuralImportsForbidden(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"torch", "numpy", "onnx", "onnxruntime"}:
            raise AssertionError("root-local wire reader cannot import a neural runtime")


_guard = NeuralImportsForbidden()
sys.meta_path.insert(0, _guard)
try:
    from rz_pals_model import rovent_utility_root_local as reader
    from rz_pals_model import nnuec_utility_reference as utility_model
    from rz_pals_model import nnuec_v2_reference as base_model
    from test_rovent_collection_teacher import submitted_fixture
    from test_rovent_external_root import fixture as external_root_fixture
    from test_rovent_utility_input import config, input_mapping, fixture as input_fixture
finally:
    sys.meta_path.remove(_guard)


ACTION = "rz-rovent-utility-root-local-action/1"
PAIR = "rz-rovent-utility-root-local-pair/1"
ACTUAL_ACTION = "rz-rovent-utility-actual-route-action/1"
ACTUAL_PAIR = "rz-rovent-utility-actual-route-pair/1"
COST_SCOPE = "exclusive_submitted_global_target_route_and_descendants_with_required_drain"
OBJECTIVE = "exclusive_wall_nanoseconds"
SEMANTICS = "board800-exactb192-causal-line-kind4-directedpair-originfirst-exactc256-fp32-v2"
CONTEXT_POLICY = "exact-original-review-view:anchor-rules-history:kind4-parent-supersedes-origin-first:immutable-model:cancel-deadline-epoch:typed-owner-v2"
LOCAL_POLICY = "exact-begin-rules-history:ordered-checked-move16:append-undo:fixed-context-authority:same-owner-resume-v2"
RECIPIENTS = {"route_p": 4, "route_c": 5, "route_r": 6}
UNKNOWN_CODES = {"different_frozen_group": 0, "duplicate_actual_action": 1, "same_recipient": 2,
    "no_new_valid_witness": 3, "equal_observation": 4, "pareto_incomparable": 5}
COUNTERS = ("native_preparations", "submitted_role_tickets", "physical_role_completions", "completed_nn_inputs",
    "nn_input_count_unknown", "consumed_role_outputs", "accepted_nn_inputs", "accepted_nn_input_count_unknown",
    "consumed_nn_inputs", "consumed_nn_input_count_unknown", "zero_nn_outputs_consumed", "unused_role_results",
    "canceled_preparations", "cpu_slice_receipts", "cpu_progress_publications", "cpu_yields", "cpu_yield_requests",
    "cpu_yield_acknowledgements", "cpu_natural_completions", "validator_ticket_admissions", "foreign_reports",
    "foreign_reported_nodes", "foreign_node_count_unknown", "foreign_yield_requests", "foreign_yield_acknowledgements",
    "rejected_events", "notification_waits")
FALSE_AUTHORITY = ("past_native_authority_reissued", "past_physical_completion_reissued", "execution_authorized",
    "training_target_authorized", "known_pair_eligible", "actual_training_executed", "rules_history_replayed",
    "source_artifact_bytes_verified", "physical_completion_verified")


def h(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def u64(value):
    return value.to_bytes(8, "little")


def digest_bytes(value):
    return bytes.fromhex(value)


def json_bytes(value, *, pretty=False):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
        indent=2 if pretty else None, separators=None if pretty else (",", ":")).encode("utf-8")


def json_domain(domain, value):
    return hashlib.sha256(json_bytes([domain, value])).hexdigest()


def outer_sha(domain, raw):
    return hashlib.sha256(domain.encode("utf-8") + b"\0" + u64(len(raw)) + raw).hexdigest()


def artifact(name):
    return {"path": name, "sha256": h("synthetic artifact " + name), "bytes": 1,
        "source": "synthetic metadata only; no file is read", "license": "fixture-only"}


def policy_sha(domain, text):
    return hashlib.sha256(domain.encode("utf-8") + b"\0" + json_bytes(text)).hexdigest()


def source_aggregate(commit, components):
    body = b"rz-rovent-source-aggregate/1\0" + commit.encode("ascii") + len(components).to_bytes(4, "little")
    for name, value in sorted(components.items()):
        raw = name.encode("utf-8")
        body += len(raw).to_bytes(4, "little") + raw + value.encode("ascii")
    return hashlib.sha256(body).hexdigest()


def origin_seal(origin):
    ticket, rep = origin["ticket"], origin["ticket"]["representation"]
    body = u64(origin["frame_sequence"]) + u64(ticket["request_sequence"])
    body += b"".join(u64(ticket[key]) for key in ("process_epoch", "game_generation", "root_generation"))
    body += digest_bytes(ticket["implementation_sha256"]) + digest_bytes(ticket["question_digest"])
    body += digest_bytes(ticket["view_digest"]) + u64(ticket["view_sequence"]) + digest_bytes(origin["output_sha256"])
    body += b"".join(digest_bytes(rep[key]) for key in ("input", "history", "candidates", "mask_positions", "model", "encoding"))
    body += u64(rep["frozen_epoch"]) + u64(rep["record_revision"])
    body += bytes(({"proposer": 0, "critic": 1}[rep["role"]], 0)) + u64(ticket["deadline_tick"]) + b"\0"
    return hashlib.sha256(b"rz-rovent-accepted-local-origin-seal/1\0" + body).hexdigest()


def view_wire(ticket):
    authority = {key: ticket[key] for key in ("process_epoch", "game_generation", "root_generation", "implementation_sha256")}
    return {"authority": authority, "sequence": ticket["view_sequence"], "context_policy_sha256": ticket["context_policy"],
        "view_digest": ticket["view_digest"], "root_state": {"owner": ticket["root_state"]["owner"],
            "revision": ticket["root_state"]["revision"], "semantic_sha256": ticket["root_state"]["semantic"]},
        **{key: ticket[key] for key in ("actual_tokens", "selected_records", "anchors", "length_bucket")}}


def reseal_action(value):
    action, cost, semantic, inp = (value[key] for key in ("action", "cost", "semantic", "input"))
    group = (b"rz-rovent-utility-route-comparison-group/1\0" + digest_bytes(action["frozen_context_sha256"])
        + digest_bytes(cost["resource_profile_sha256"]) + digest_bytes(cost["cost_epoch_sha256"])
        + cost["scope"].encode("utf-8") + b"\0" + cost["objective"].encode("utf-8"))
    action["comparison_group_sha256"] = hashlib.sha256(group).hexdigest()
    authority = action["authority"]
    body = b"".join(u64(authority[key]) for key in ("process_epoch", "game_generation", "root_generation"))
    body += digest_bytes(authority["implementation_sha256"]) + u64(action["frame_sequence"]) + digest_bytes(action["question_sha256"])
    body += bytes((action["output_index"],)) + u64(action["manager_sequence"]) + digest_bytes(action["root_budget_sha256"])
    body += digest_bytes(action["comparison_group_sha256"]) + inp["submission_target_sha256"].encode("ascii")
    # std::time::Duration::as_nanos() is u128 even though the publication's
    # checked whole_wall_ns is a u64. Encoding only eight bytes is incorrect.
    body += cost["whole_wall_ns"].to_bytes(16, "little") + u64(len(semantic["new_witnesses"]))
    for witness in semantic["new_witnesses"]:
        body += digest_bytes(witness["identity_sha256"]) + digest_bytes(witness["raw_evidence_sha256"])
    body += u64(len(semantic["pc_consumption"]))
    for source in semantic["pc_consumption"]:
        body += digest_bytes(source["origin"]["seal_sha256"])
    body += u64(len(semantic["r_selected_pc_consumption"]))
    for source in semantic["r_selected_pc_consumption"]:
        body += u64(source["native_execution"]["epoch"]) + u64(source["native_execution"]["sequence"])
        if source["selected_pc_ticket"] is not None:
            body += u64(source["selected_pc_ticket"]["request_sequence"])
    action["original_action_sha256"] = hashlib.sha256(ACTUAL_ACTION.encode() + b"\0" + body).hexdigest()


def action_fixture(recipient="route_p", *, sequence=1, pretty_reference=False, external_root=False):
    """P/C consumption is synthetically linked, not accepted by a live owner."""
    role = "critic" if recipient == "route_c" else "proposer"
    if external_root:
        if role != "proposer":
            raise ValueError("the existing external-root fixture only supplies an original P ticket")
        _manifest, record, _lock, _registration, _expected = external_root_fixture()
    else:
        _manifest, record = submitted_fixture(role=role)
    root = copy.deepcopy(record["native"]["raw_retirement"]["root_admission"])
    origin = root["native_origin"]
    origin["source_sha256"] = source_aggregate(origin["source_commit"], origin["source_components"])
    root["native_origin_sha256"] = json_domain("rz-rovent-collection-native-origin/1", origin)
    root_sha = json_domain(root["schema"], root)
    model_identity = {"semantics": SEMANTICS, "weights_sha256": "a" * 64, "training": {"kind": "untrained"}}
    nnuec_registration = {"domain": "rz-nnuec-v2-registration/1", "schema": "rz-nnuec-fp32-v2",
        "semantics": SEMANTICS, "encoding": "rz-nnuec-fp32-input/2", "implementation_sha256": h("synthetic compiled V2 source"),
        "configuration_sha256": h("synthetic compiled V2 dimensions"), "context_namespace_policy": CONTEXT_POLICY,
        "local_namespace_policy": LOCAL_POLICY,
        "context_namespace_policy_sha256": policy_sha("rz-rovent-nnuec-context-policy/1", CONTEXT_POLICY),
        "local_namespace_policy_sha256": policy_sha("rz-rovent-nnuec-local-policy/1", LOCAL_POLICY),
        "model_identity": copy.deepcopy(model_identity)}
    utility_input = input_mapping()
    for line in utility_input["base_input"]["context_lines"]:
        line["original_first_move"]["to_square"] = 1
        line["moves"][0]["to_square"] = 1
    text = json.dumps(utility_input, ensure_ascii=False, indent=2) if pretty_reference else json.dumps(utility_input, separators=(",", ":"))
    supplied = input_fixture(value=utility_input, text=text)
    ticket = copy.deepcopy(record["native"]["raw_submission"]["ticket"])
    accepted = {"frame_sequence": 10, "ticket": ticket, "output_sha256": record["native"]["output_digest"],
        "kind": "local_accepted", "peer": None, "seal_sha256": ""}
    accepted["seal_sha256"] = origin_seal(accepted)
    authority = {key: root[key] for key in ("process_epoch", "game_generation", "root_generation", "implementation_sha256")}
    root_rules = {"rules_version": "rz-position/0.1.0", "variant": "standard_chess",
        "semantic_sha256": root["root_state"]["semantic"], "fen": record["input"]["rules_anchor"]["board_fen"],
        "history_origin": "Fen", "history_completeness": "UnknownPrefix",
        "known_history": [record["input"]["rules_anchor"]["board_fen"]]}
    before = dict.fromkeys(COUNTERS, 0)
    after = dict(before)
    after.update(native_preparations=1, submitted_role_tickets=1, physical_role_completions=1,
        completed_nn_inputs=1, consumed_role_outputs=1, accepted_nn_inputs=1, consumed_nn_inputs=1)
    action = {"schema": ACTION, "collection_sequence": sequence,
        "source": {"root_admission_sha256": root_sha, "root_admission": root,
            "native_origin_sha256": root["native_origin_sha256"], "input_lock_sha256": origin["input_lock_sha256"],
            "nnuec_registration": nnuec_registration, "weights_artifact": artifact("synthetic-weights.json"),
            "provenance_artifact": artifact("synthetic-provenance.json")},
        "input": {"reference_json": text, "bytes": len(text.encode("utf-8")),
            **{key: supplied[key] for key in ("input_sha256", "target_sha256", "submission_target_sha256", "context_sha256")},
            "target_record": 0, "target_record_ids": [0, 1], "base_identity": model_identity, "root_rules": root_rules,
            "records_metadata": [{"anchor": 0, "parent": None if i == 0 else 0, "supersedes": None,
                "source_state": 1, "source_revision": i + 1, "source_line_offset": 0, "original_first_move": 64} for i in range(2)],
            "source_rules": [copy.deepcopy(root_rules), copy.deepcopy(root_rules)],
            "provenance": {"root_state": root["store_root_state"], "generation": root["root_generation"], "sequence": 1,
                "full_view_sha256": h("synthetic utility full view"), "view_seal_sha256": supplied["view_seal_sha256"],
                "record_scope_sha256": h("synthetic record scope"), "policy_sha256": h("synthetic selected policy"),
                "model": copy.deepcopy(model_identity)}},
        "action": {"authority": authority, "frame_sequence": 10, "question_sha256": ticket["question_digest"],
            "recipient": recipient, "output_index": RECIPIENTS[recipient], "manager_sequence": sequence,
            "original_first_move": 64, "root_budget_sha256": root["root_budget_sha256"],
            "original_deadline": {"clock": root["deadline_clock"], "tick": root["deadline_tick"]}, "mode": origin["mode"],
            "budget": {"role_submissions_remaining": 4, "original_cpu_credit": 4096, "original_store_byte_limit": 1048576},
            "frozen_context_sha256": h("synthetic opaque frozen search scope"), "comparison_group_sha256": "", "original_action_sha256": ""},
        "semantic": {"baseline_witnesses": [], "new_witnesses": [], "pc_consumption": [{"origin": accepted,
            "question_sha256": ticket["question_digest"], "view": view_wire(ticket), "full_view_sha256": h("synthetic consumed full view")}],
            "r_selected_pc_consumption": [], "dependencies": [], "native_executions": []},
        "cost": {"scope": COST_SCOPE, "objective": OBJECTIVE, "resource_profile_sha256": h("synthetic resource profile"),
            "cost_epoch_sha256": h("synthetic resource cost epoch"), "whole_wall_ns": 123456,
            "baseline_counters": before, "final_counters": after, "native_credit_baseline": {"spent": 0, "reserved": 0},
            "native_credit_final": {"spent": 0, "reserved": 0}, "cpu_thread_time_ns": None, "energy": None, "retained_host_bytes": None},
        "support_mask": [i == RECIPIENTS[recipient] for i in range(7)], "target_known_mask": [False] * 7}
    reseal_action(action)
    return action


def r_action_fixture():
    """Synthetic exact Job/TaskKey and selected-PC metadata, not a CPU cap.

    TaskKey follows store.rs's fourteen serde fields. The Job is the closed
    twelve-field utility_owner.rs projection, not an arbitrary CPU JSON body.
    Capture and ReviewView counters deliberately name different sequences.
    """
    value = action_fixture("route_r")
    source, inp, action, semantic = (value[key] for key in ("source", "input", "action", "semantic"))
    root = source["root_admission"]
    root["native_origin"]["mode"] = action["mode"] = "HybridPcr"
    root["native_origin_sha256"] = json_domain("rz-rovent-collection-native-origin/1", root["native_origin"])
    source["native_origin_sha256"] = root["native_origin_sha256"]
    source["root_admission_sha256"] = json_domain(root["schema"], root)
    consumed = semantic["pc_consumption"][0]
    ticket = consumed["origin"]["ticket"]
    native = {"epoch": action["authority"]["process_epoch"], "sequence": 31}
    binding = h("synthetic exact R evaluation namespace")
    question = action["question_sha256"]
    job = {"authority": copy.deepcopy(action["authority"]), "job_sequence": 21,
        "state_owner": root["root_state"]["owner"], "state_revision": root["root_state"]["revision"],
        "state_semantic_sha256": root["root_state"]["semantic"], "question_sha256": question,
        "root_restriction_sha256": h("synthetic ordered native root restriction"), "evaluation_namespace_sha256": binding,
        "search_conditions_sha256": h("synthetic native CPU search conditions"), "total_native_node_credit": 64,
        "hard_deadline_clock_epoch": root["deadline_clock"], "hard_deadline_ns": root["deadline_tick"]}
    task = {"state": root["store_root_state"], "line": None, "question": "AnalyzeRootMoves", "root_moves": [64, 128],
        "model": 1, "epoch": 1, "value_identity": copy.deepcopy(inp["base_identity"]), "checker_identity": None,
        "cpu_condition": "synthetic condition", "profile": 1, "condition": 1, "input_revision": 1,
        "requested_depth": 1, "node_budget": 64}
    semantic["native_executions"] = [native]
    semantic["r_selected_pc_consumption"] = [{"stage": "SelectedPcViewConsumed",
        "authority": copy.deepcopy(action["authority"]), "frame_sequence": action["frame_sequence"],
        "question_sha256": question, "submitted_view": copy.deepcopy(consumed["view"]),
        "submitted_full_view_sha256": inp["provenance"]["full_view_sha256"], "job": job, "owner": 0,
        "task_key": task, "store_execution": 41, "native_execution": copy.deepcopy(native), "observation": 51,
        "capture_sequence": inp["provenance"]["sequence"] + 6, "binding_sha256": binding,
        "capture_full_view_sha256": inp["provenance"]["full_view_sha256"],
        "capture_view_seal_sha256": inp["provenance"]["view_seal_sha256"],
        "attested_slice_work": {"nodes": 64, "quiescence_nodes": 7, "tt_hits": 3},
        "selected_pc_ticket": copy.deepcopy(ticket), "selected_output_sha256": consumed["origin"]["output_sha256"],
        "selected_record": {"revision": 1, "origin_state": root["store_root_state"]},
        "selected_full_view_sha256": consumed["full_view_sha256"]}]
    value["cost"]["native_credit_final"]["spent"] = 64
    value["cost"]["final_counters"].update(cpu_slice_receipts=1, cpu_natural_completions=1)
    reseal_action(value)
    return value


def actual_comparison(left, right):
    la, ra = left["action"], right["action"]
    lw = {item["identity_sha256"] for item in left["semantic"]["new_witnesses"]}
    rw = {item["identity_sha256"] for item in right["semantic"]["new_witnesses"]}
    preferred, reason = None, None
    if la["original_action_sha256"] == ra["original_action_sha256"]:
        reason = "duplicate_actual_action"
    elif la["comparison_group_sha256"] != ra["comparison_group_sha256"]:
        reason = "different_frozen_group"
    elif la["recipient"] == ra["recipient"]:
        reason = "same_recipient"
    elif not lw and not rw:
        reason = "no_new_valid_witness"
    elif lw == rw and left["cost"]["whole_wall_ns"] == right["cost"]["whole_wall_ns"]:
        reason = "equal_observation"
    elif rw <= lw and left["cost"]["whole_wall_ns"] <= right["cost"]["whole_wall_ns"]:
        preferred = la["output_index"]
    elif lw <= rw and right["cost"]["whole_wall_ns"] <= left["cost"]["whole_wall_ns"]:
        preferred = ra["output_index"]
    else:
        reason = "pareto_incomparable"
    body = digest_bytes(la["original_action_sha256"]) + digest_bytes(ra["original_action_sha256"])
    body += bytes((255 if preferred is None else preferred, 255 if reason is None else UNKNOWN_CODES[reason]))
    return hashlib.sha256(ACTUAL_PAIR.encode() + b"\0" + body).hexdigest(), reason or "missing_comparison_cohort"


def pair_fixture(left, right):
    left_raw, right_raw = json_bytes(left), json_bytes(right)
    native, reason = actual_comparison(left, right)
    return {"schema": PAIR, "collection_sequence": 1, "left_action_sha256": outer_sha(ACTION, left_raw),
        "right_action_sha256": outer_sha(ACTION, right_raw), "original_comparison_sha256": native,
        "support_mask": [a or b for a, b in zip(left["support_mask"], right["support_mask"], strict=True)],
        "target_known_mask": [False] * 7, "preferred": None, "unknown_reason": reason}


class UtilityRootLocalReaderTests(unittest.TestCase):
    def setUp(self):
        sys.meta_path.insert(0, _guard)

    def tearDown(self):
        sys.meta_path.remove(_guard)

    def action(self, value=None, *, pretty=False):
        raw = json_bytes(action_fixture() if value is None else value, pretty=pretty)
        return reader.read_root_local_action(raw, outer_sha(ACTION, raw), config=config())

    def pair(self, left, right, value=None):
        value = pair_fixture(left, right) if value is None else value
        raw = json_bytes(value)
        return reader.read_root_local_pair(raw, outer_sha(PAIR, raw), left=self.action(left), right=self.action(right))

    def assert_no_authority(self, observation):
        for key in FALSE_AUTHORITY:
            self.assertIs(observation[key], False, key)

    def test_original_bytes_outer_domain_and_actual_recipient_are_preserved(self):
        for recipient, index in (("route_p", 4), ("route_c", 5)):
            value = action_fixture(recipient)
            raw = json_bytes(value, pretty=True)
            observed = reader.read_root_local_action(raw, outer_sha(ACTION, raw), config=config())
            self.assertEqual(observed.raw_bytes, raw)
            self.assertEqual(observed.sha256, outer_sha(ACTION, raw))
            self.assertEqual(observed.wire, value)
            report = observed.observation()
            self.assertEqual(report["recipient"], recipient)
            self.assertEqual(report["support_mask"], [i == index for i in range(7)])
            self.assertEqual(report["target_known_mask"], [False] * 7)
            self.assert_no_authority(report)

    def test_descriptor_bearing_external_root_two_preserves_its_distinct_source_shape(self):
        value = action_fixture(external_root=True)
        result = self.action(value)
        source = result.wire["source"]
        self.assertEqual(source["root_admission"]["schema"], "rz-rovent-collection-root-admission/2")
        self.assertIn("external_source", source["root_admission"])
        self.assertEqual(source["root_admission_sha256"], json_domain("rz-rovent-collection-root-admission/2", source["root_admission"]))
        self.assert_no_authority(result.observation())

    def test_reference_json_spelling_is_retained_instead_of_reserialized(self):
        value = action_fixture(pretty_reference=True)
        result = self.action(value)
        self.assertEqual(result.wire["input"]["reference_json"], value["input"]["reference_json"])
        compact = copy.deepcopy(value)
        compact["input"]["reference_json"] = json.dumps(json.loads(value["input"]["reference_json"]), separators=(",", ":"))
        raw = json_bytes(compact)
        with self.assertRaises(ValueError):
            reader.read_root_local_action(raw, outer_sha(ACTION, raw), config=config())

    def test_outer_hash_binds_exact_bytes_domain_and_length(self):
        raw = json_bytes(action_fixture())
        wrong = [hashlib.sha256(raw).hexdigest(), outer_sha(PAIR, raw),
            hashlib.sha256(ACTION.encode() + b"\0" + len(raw).to_bytes(4, "little") + raw).hexdigest()]
        for expected in wrong:
            with self.subTest(expected=expected), self.assertRaises(ValueError):
                reader.read_root_local_action(raw, expected, config=config())
        with self.assertRaises(ValueError):
            reader.read_root_local_action(raw + b"\n", outer_sha(ACTION, raw), config=config())

    def test_outer_parser_rejects_nonobjects_nonfinite_and_invalid_utf8_without_codec_conversion(self):
        for raw in (b"\xff", b"[]", b"{}", b'{"schema":NaN}', b'{"schema":"x",}'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                reader.read_root_local_action(raw, outer_sha(ACTION, raw), config=config())
        value = action_fixture()
        value["schema"] = "rz-rovent-utility-comparison-cohort-action/1"
        with self.assertRaises(ValueError):
            self.action(value)
        value = action_fixture()
        value["input"]["bytes"] = True
        with self.assertRaises(ValueError):
            self.action(value)

    def test_group_action_and_accepted_origin_seals_are_checked_beneath_a_valid_outer_hash(self):
        for path in (("action", "comparison_group_sha256"), ("action", "original_action_sha256"),
                ("source", "root_admission_sha256"), ("source", "native_origin_sha256"),
                ("input", "submission_target_sha256")):
            value = action_fixture()
            value[path[0]][path[1]] = "f" * 64
            with self.subTest(path=path), self.assertRaises(ValueError):
                self.action(value)
        value = action_fixture()
        value["semantic"]["pc_consumption"][0]["origin"]["seal_sha256"] = "f" * 64
        reseal_action(value)
        with self.assertRaises(ValueError):
            self.action(value)

    def test_support_and_target_masks_cannot_claim_unexecuted_or_known_axes(self):
        for name, index, replacement in (("support_mask", 0, True), ("support_mask", 5, True),
                ("support_mask", 4, False), ("support_mask", 4, 1), ("target_known_mask", 4, True)):
            value = action_fixture()
            value[name][index] = replacement
            with self.subTest(name=name, index=index, replacement=replacement), self.assertRaises(ValueError):
                self.action(value)

    def test_route_r_cannot_reuse_a_pc_route_without_selected_pc_cpu_consumption(self):
        value = r_action_fixture()
        value["semantic"]["r_selected_pc_consumption"] = []
        reseal_action(value)
        with self.assertRaises(ValueError):
            self.action(value)

    def test_route_r_preserves_separate_capture_clock_and_exact_selected_pc_job_task_join(self):
        value = r_action_fixture()
        source = value["semantic"]["r_selected_pc_consumption"][0]
        self.assertNotEqual(source["capture_sequence"], value["input"]["provenance"]["sequence"])
        self.assertEqual(len(source["job"]), 12)
        self.assertEqual(len(source["task_key"]), 14)
        observed = self.action(value)
        self.assertEqual(observed.wire, value)
        self.assertEqual(observed.observation()["support_mask"], [False] * 6 + [True])
        self.assert_no_authority(observed.observation())
        for field in ("binding_sha256", "capture_full_view_sha256", "capture_view_seal_sha256", "selected_output_sha256",
                "selected_full_view_sha256"):
            changed = copy.deepcopy(value)
            changed["semantic"]["r_selected_pc_consumption"][0][field] = "f" * 64
            reseal_action(changed)
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.action(changed)
        changed = copy.deepcopy(value)
        changed["semantic"]["native_executions"] = []
        with self.assertRaises(ValueError):
            self.action(changed)

    def test_cost_scope_monotonic_counters_and_observed_unknowns_are_preserved(self):
        for change in (lambda value: value["cost"].__setitem__("scope", "ordinary_root_wall"),
                lambda value: value["cost"]["baseline_counters"].__setitem__("completed_nn_inputs", 2),
                lambda value: value["cost"]["final_counters"].__setitem__("nn_input_count_unknown", 1),
                lambda value: value["cost"]["native_credit_final"].__setitem__("reserved", 1),
                lambda value: value["cost"].__setitem__("cpu_thread_time_ns", 123)):
            value = action_fixture()
            change(value)
            reseal_action(value)
            with self.subTest(value=value["cost"]), self.assertRaises(ValueError):
                self.action(value)

    def test_root_input_model_and_history_joins_are_not_replaced_by_matching_outer_hashes(self):
        for change in (lambda value: value["input"]["base_identity"].__setitem__("weights_sha256", "f" * 64),
                lambda value: value["source"]["nnuec_registration"].__setitem__("extra", True),
                lambda value: value["input"]["provenance"].__setitem__("generation", 99),
                lambda value: value["input"]["root_rules"]["known_history"].__setitem__(0, "8/8/8/8/8/8/4K3/7k b - - 0 1"),
                lambda value: value["input"]["root_rules"].__setitem__("history_origin", "StartPosition")):
            value = action_fixture()
            change(value)
            with self.subTest(value=value["input"]), self.assertRaises(ValueError):
                self.action(value)

    def test_pair_uses_actual_action_hashes_and_union_support_but_no_known_target(self):
        left, right = action_fixture(), action_fixture("route_c", sequence=2)
        value = pair_fixture(left, right)
        result = self.pair(left, right, value)
        self.assertEqual(result.wire, value)
        report = result.observation()
        self.assertEqual(report["unknown_reason"], "no_new_valid_witness")
        self.assertEqual(report["support_mask"], [False, False, False, False, True, True, False])
        self.assertEqual(report["target_known_mask"], [False] * 7)
        self.assertIsNone(report["preferred"])
        self.assert_no_authority(report)
        detached = result.wire
        detached["support_mask"][0] = True
        report["target_known_mask"][4] = True
        self.assertEqual(result.wire, value)
        self.assertEqual(result.observation()["target_known_mask"], [False] * 7)

    def test_pair_records_duplicate_same_recipient_and_different_group_with_original_priority(self):
        left = action_fixture()
        duplicate = copy.deepcopy(left)
        duplicate["collection_sequence"] = 8
        same = action_fixture(sequence=2)
        other = action_fixture("route_c", sequence=2)
        other["cost"]["cost_epoch_sha256"] = h("different original cost scope")
        reseal_action(other)
        for right, reason in ((duplicate, "duplicate_actual_action"), (same, "same_recipient"), (other, "different_frozen_group")):
            with self.subTest(reason=reason):
                self.assertEqual(self.pair(left, right).observation()["unknown_reason"], reason)

    def test_pair_never_promotes_actual_pareto_preference_without_comparison_cohort(self):
        left, right = action_fixture(), action_fixture("route_c", sequence=2)
        witness = {"kind": "counter", "observation": 11, "identity_sha256": h("synthetic new conditional witness"),
            "raw_evidence_sha256": h("synthetic raw witness"), "line": [64]}
        left["semantic"]["new_witnesses"] = [witness]
        right["cost"]["whole_wall_ns"] += 1
        reseal_action(left)
        reseal_action(right)
        result = self.pair(left, right)
        self.assertEqual(result.wire["unknown_reason"], "missing_comparison_cohort")
        self.assertIsNone(result.wire["preferred"])
        self.assertEqual(result.wire["target_known_mask"], [False] * 7)
        self.assert_no_authority(result.observation())

    def test_pair_cannot_relabel_endpoints_comparison_digest_or_knownness(self):
        left, right = action_fixture(), action_fixture("route_c", sequence=2)
        for name, replacement in (("left_action_sha256", "f" * 64), ("original_comparison_sha256", "f" * 64),
                ("unknown_reason", "missing_comparison_cohort"), ("preferred", "route_p"),
                ("support_mask", [False] * 7), ("target_known_mask", [False] * 4 + [True, True, False])):
            value = pair_fixture(left, right)
            value[name] = replacement
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.pair(left, right, value)

    def test_wire_and_report_are_detached_and_unknown_fields_or_duplicate_json_are_rejected(self):
        observed = self.action()
        wire, report = observed.wire, observed.observation()
        wire["support_mask"][0] = True
        report["target_known_mask"][4] = True
        self.assertFalse(observed.wire["support_mask"][0])
        self.assertEqual(observed.observation()["target_known_mask"], [False] * 7)
        value = action_fixture()
        value["unexpected"] = True
        with self.assertRaises(ValueError):
            self.action(value)
        raw = json_bytes(action_fixture()).replace(b'"collection_sequence":1', b'"collection_sequence":1,"collection_sequence":1', 1)
        with self.assertRaises(ValueError):
            reader.read_root_local_action(raw, outer_sha(ACTION, raw), config=config())

    def test_readers_do_not_open_files_initialize_models_or_run_forward(self):
        forbidden = AssertionError("historical root-local parsing cannot open sources or execute a model")
        value = action_fixture()
        with mock.patch("builtins.open", side_effect=forbidden), \
                mock.patch.object(utility_model.UtilityCheckpoint, "from_mapping", side_effect=forbidden), \
                mock.patch.object(base_model.NnuecV2Checkpoint, "from_mapping", side_effect=forbidden), \
                mock.patch.object(utility_model, "evaluate", side_effect=forbidden), \
                mock.patch.object(base_model, "evaluate", side_effect=forbidden):
            result = self.action(value)
        self.assert_no_authority(result.observation())


if __name__ == "__main__":
    unittest.main()
