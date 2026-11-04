"""Synthetic CPU_T wire fixtures; no actual teacher, Rules or training proof."""
import copy
import hashlib
import importlib.abc
import json
import math
import sys
import unittest


class NeuralImportsForbidden(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"torch", "numpy", "onnx", "onnxruntime"}:
            raise AssertionError("neural imports forbidden in Collection teacher wire tests")


_guard = NeuralImportsForbidden()
sys.meta_path.insert(0, _guard)
try:
    from rz_pals_model import rovent_collection_teacher as teacher
    from rz_pals_model import rovent_corpus as corpus
    from rz_pals_model import generation_data as data
    from rz_pals_model import rovent_submitted_input as submitted_inputs
    from rz_pals_model import rovent_native_v3 as submitted_native
    from rz_pals_model.virtual_review import input_digest
    from test_rovent_corpus import original_fixture, add_teacher_policy, seal_record, reseal_head, h
    from test_rovent_submitted_input import fixture as submitted_input_fixture
finally:
    sys.meta_path.remove(_guard)


def domain_nul(domain, value):
    """Independent Rust descriptor framing, not generation_digest or its constants."""
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(domain.encode("utf-8") + b"\0" + raw).hexdigest()


def seal_descriptor(descriptor):
    inp = descriptor["input"]
    budget = domain_nul("rz-rovent-root-budget/1", {key: value for key, value in inp.items() if key != "clock"})
    descriptor["root_budget_sha256"] = budget
    descriptor["deadline_seal_sha256"] = domain_nul("rz-rovent-root-deadline/1", {
        "domain": "rz-rovent-root-deadline/1", "startup_lock_sha256": inp["startup_lock_sha256"],
        "policy": inp["policy"], "authority": inp["authority"], "root_budget_sha256": budget,
        "clock": inp["clock"], "cleanup_max_ms": inp["declared_ceilings"]["cleanup_max_ms"]})
    return domain_nul("rz-rovent-root-admission/1", descriptor)


def seal_native(record, domain="rz-rovent-native-owner-event/2"):
    native = record["native"]
    native["raw_submission_sha256"] = data.digest(domain, native["raw_submission"])
    native["raw_completion"]["submitted_sha256"] = native["raw_submission_sha256"]
    native["raw_completion_sha256"] = data.digest(domain, native["raw_completion"])
    native["raw_retirement"].update(submitted_sha256=native["raw_submission_sha256"],
        completed_sha256=native["raw_completion_sha256"])
    native["raw_retirement_sha256"] = data.digest(domain, native["raw_retirement"])


def seal_root(record):
    retired = record["native"]["raw_retirement"]
    root = retired["root_admission"]
    root["native_origin_sha256"] = data.digest("rz-rovent-collection-native-origin/1", root["native_origin"])
    retired["root_admission_sha256"] = data.digest("rz-rovent-collection-root-admission/1", root)
    seal_native(record)


def fixture(*, collection=True, role="proposer"):
    """Closed serde DTO shape with deliberately synthetic custody and Rules data."""
    _, manifest, record = original_fixture()
    if role == "critic":
        snapshot, native = record["input"], record["native"]
        record["role"] = snapshot["role"] = snapshot["encoded_model_input"]["role"] = "critic"
        snapshot["selected_view"]["query"]["purpose"] = snapshot["encoded_model_input"]["query"]["purpose"] = "reply_policy"
        snapshot["view_sha256"] = data.view_digest(snapshot["selected_view"])
        record["input_sha256"] = data.seal_input(snapshot)
        native["head"] = "critic"
        native["input_digest"] = input_digest(snapshot["encoded_model_input"])
        output = native["raw_completion"]["output"]
        output["payload"].update(kind="counterexample", divergence_logits_bits=[])
        framed = (b"rz-rovent-accepted-critic-head/1\0" + (2).to_bytes(8, "little") + (0).to_bytes(8, "little")
            + b"".join(bits.to_bytes(4, "little") for bits in output["payload"]["candidate_logits_bits"] + output["payload"]["wdl_bits"]))
        native["output_digest"] = hashlib.sha256(framed).hexdigest()
        for name in ("raw_submission", "raw_completion", "raw_retirement"):
            event = native[name]
            event.update(head="critic", input_sha256=record["input_sha256"])
            event["ticket"]["representation"].update(role="critic", input=native["input_digest"])
        native["raw_submission"]["input_digest"] = native["input_digest"]
        output["representation"].update(role="critic", input=native["input_digest"])
        native["raw_completion"]["output_digest"] = native["raw_retirement"]["output_digest"] = native["output_digest"]
    elif role != "proposer":
        raise ValueError("synthetic Collection teacher fixture is P/C only")
    add_teacher_policy(manifest, record)
    native, snapshot = record["native"], record["input"]
    raw = record["head_sources"]["policy"]["raw_receipt"]
    original = copy.deepcopy(raw["binding"])
    ticket = native["raw_submission"]["ticket"]
    for name in ("raw_submission", "raw_completion", "raw_retirement"):
        native[name]["schema"] = "rz-rovent-native-owner-event/2"
        native[name]["ticket"]["root_state"]["semantic"] = snapshot["rules_anchor"]["rules_state_sha256"]
    replay = {"schema": "rz-rovent-rules-position-replay/1", "origin": "fen", "completeness": "unknown_prefix",
        "start_fen": snapshot["rules_anchor"]["board_fen"], "moves": copy.deepcopy(snapshot["actual_history"]),
        "observer_revision": 1, "anchor": copy.deepcopy(snapshot["rules_anchor"])}
    native["raw_submission"]["rules_replay"] = {"schema": "rz-rovent-submitted-rules-replay/1",
        "actual_root": copy.deepcopy(replay), "selected_anchors": [copy.deepcopy(replay)]}
    native["raw_retirement"]["acceptance_kind"] = "accepted_role_step"
    resources = {"cpu_quota_millicpus": 2000, "pools": [{"actors": ["manager"], "max_threads": 1,
        "cpu_charge_millicpus": 1000}], "memory_high_bytes": 6 << 30, "memory_max_bytes": 12 << 30,
        "gpu_devices": 0, "physical_nn_concurrency": 1, "topology_policy": "SharedControl"}
    descriptor = {"domain": "rz-rovent-root-admission/1", "input": {"policy": "dynamic_go",
        "startup_lock_sha256": h("original loaded InputLock"), "state_and_history_sha256": ticket["root_state"]["semantic"],
        "rules_profile_sha256": h("Rules profile"), "authority": {"epoch": 1, "game": 1, "root": 1,
        "implementation_sha256": ticket["implementation_sha256"]}, "node_unit": "rovent-position-visits/1",
        "search_configuration_sha256": h("search configuration"), "coordinator_policy_sha256": h("coordinator"),
        "original_go": {"coverage": "full_original_go", "requested_nodes": 4096, "depth_plies": None,
            "mate_moves": None, "searchmoves": None, "transport_simulation_cap": 4096,
            "original_parsed_limits": {"searchmoves": None, "depth_plies": None, "mate_moves": None,
                "movetime_ms": 1000, "white_time_ms": None, "black_time_ms": None, "white_increment_ms": None,
                "black_increment_ms": None, "moves_to_go": None, "nodes": 4096, "infinite": False, "ponder": False}},
        "allowed_root_moves": ["a1b1", "a1c1"], "clock": {key: {"clock_epoch": 1, "at_ns": tick}
            for key, tick in (("hard", 1000), ("soft", 500), ("admission", 800), ("cleanup", 2000))},
        "declared_ceilings": {"resource_plan": resources, "registered_max_chess_plies": 256,
            "cpu_credit_limit": 4096, "questions_max": 64, "ancestor_depth_max": 8, "waiting_jobs_max": 64,
            "role_tickets_max": 16, "wall_time_max_ms": 1000, "cleanup_max_ms": 500, "output_bytes_max": 1048576}},
        "root_budget_sha256": "", "deadline_seal_sha256": ""}
    descriptor_sha = seal_descriptor(descriptor)
    profiles = [{"domain": "rz-rovent-external-engine-profile/1", "purpose": "teacher",
        "profile": {"path": actor + ".json", "sha256": pin["profile_sha256"], "bytes": 1,
            "source": "synthetic only", "license": "synthetic only"},
        "binary": {"path": actor, "sha256": pin["binary_sha256"], "bytes": pin["binary_bytes"],
            "source": "synthetic only", "license": "synthetic only"},
        "canonical_sha256": pin["canonical_sha256"], "capability_sha256": pin["capability_sha256"],
        "source_commit": pin["source_commit"], "requested_threads": 1, "requested_hash_mib": 256}
        for actor, pin in manifest["teachers"].items()]
    origin = {"schema": "rz-rovent-collection-native-origin/1", "purpose": "collection", "process_epoch": 1,
        "mode": "HybridPc", "input_lock_sha256": descriptor["input"]["startup_lock_sha256"],
        "source_commit": manifest["source_commit"], "source_components": {"synthetic": h("synthetic source")},
        "source_sha256": manifest["source_sha256"], "weights_sha256": native["model_sha256"],
        "export_manifest_sha256": h("export manifest"), "backend_sha256": h("backend"),
        "runtime_library_sha256": manifest["runtime_sha256"], "runtime_bundle_sha256": None,
        "native_configuration_sha256": h("native configuration"), "model_semantics": "synthetic PC semantics",
        "model_semantics_sha256": manifest["model_semantics_sha256"], "encoding": "synthetic PC encoding",
        "encoding_sha256": native["encoding_sha256"], "encoding_registry_sha256": h("encoding registry"),
        "model_implementation_sha256": h("PC implementation"), "frozen_epoch": 1,
        "representation_model_domain": "loaded_checkpoint_sha256", "runtime_sha256_domain": "loaded_runtime_library_file_sha256",
        "runtime_sha256": manifest["runtime_sha256"], "teacher_profiles": profiles}
    root = {"schema": "rz-rovent-collection-root-admission/1", "native_origin": origin, "native_origin_sha256": "",
        "process_epoch": 1, "game_generation": 1, "root_generation": 1,
        "implementation_sha256": ticket["implementation_sha256"], "deadline_clock": 1, "deadline_tick": 1000,
        "root_budget_sha256": descriptor["root_budget_sha256"], "selective_policy_sha256": h("selective policy"),
        "root_state": copy.deepcopy(ticket["root_state"]), "store_root_state": 1,
        "root_moves": ["a1b1", "a1c1"], "depth_plies": None, "mate_moves": None, "requested_nodes": 4096,
        "root_descriptor": copy.deepcopy(descriptor), "root_descriptor_sha256": descriptor_sha}
    native["raw_retirement"]["root_admission"] = root
    pin, actor = raw["teacher_pin"], raw["actor_id"]
    if collection:
        scope = {"authority": {key: ticket[key] for key in ("process_epoch", "game_generation", "root_generation", "implementation_sha256")},
            "input_sha256": record["input_sha256"], "view_sha256": snapshot["view_sha256"],
            "source_sha256": manifest["source_sha256"], "role": record["role"],
            "rules_semantic": snapshot["rules_anchor"]["rules_state_sha256"], "input_lock": origin["input_lock_sha256"],
            "root_admission": descriptor_sha, "root_budget": descriptor["root_budget_sha256"],
            "frozen_actor": data.digest("rz-rovent-frozen-teacher-actor/1", {"actor_id": actor, "teacher_pin": pin})}
        raw["binding"] = {"schema": "rz-rovent-collection-teacher-binding/1", "job_id": {"collection_owner": 9, "sequence": 71},
            "input_scope": scope, "requested": original["requested"], "process": original["process"],
            "root_restriction": original["root_restriction"], "position_anchor": original["position_anchor"],
            "root_moves": None, "soft_quantum_ns": 100000000,
            "original_deadline": {"clock_epoch": 1, "at_ns": 950},
            "root_descriptor": descriptor, "root_descriptor_sha256": descriptor_sha}
    else:
        raw["binding"].update(root_descriptor=descriptor, root_descriptor_sha256=descriptor_sha,
            root_budget=descriptor["root_budget_sha256"], mode=origin["mode"])
        raw["binding"]["question"]["root_budget"] = descriptor["root_budget_sha256"]
    seal_root(record)
    reseal_head(record, "policy")
    seal_record(manifest, record)
    return manifest, record


def submitted_fixture(*, role="proposer"):
    """Construct new synthetic /3 binding bytes, never convert original data."""
    manifest, record = fixture(role=role)
    manifest["schema"] = "rz-rovent-corpus-manifest/2"
    record["schema"] = "rz-rovent-training-record/2"
    snapshot = record["input"]
    prepared = submitted_input_fixture(role=role)
    del snapshot["divergence_record_ids"]
    snapshot.update({name: copy.deepcopy(prepared[name]) for name in
        ("schema", "attack_mapping", "action_support_mask", "selected_view", "view_sha256", "encoded_model_input")})
    record["input_sha256"] = submitted_inputs.seal_input(snapshot)
    native = record["native"]
    native["input_digest"] = input_digest(snapshot["encoded_model_input"])
    actual_tokens = sum(len(row["moves"]) for row in snapshot["selected_view"]["records"]) + sum(
        len(snapshot["selected_view"]["query"][key]) for key in ("prefix", "proposal", "counter"))
    for name in ("raw_submission", "raw_completion", "raw_retirement"):
        raw = native[name]
        raw.update(schema="rz-rovent-raw-record/3", input_sha256=record["input_sha256"])
        raw["ticket"]["representation"]["input"] = native["input_digest"]
        raw["ticket"]["actual_tokens"] = actual_tokens
    native["raw_submission"]["input_digest"] = native["input_digest"]
    ticket = native["raw_submission"]["ticket"]
    native["raw_submission"]["submitted_review"] = {"schema": "rz-rovent-submitted-review-evidence/1",
        "sequence": ticket["view_sequence"], "generation": native["root_generation"],
        "root_state": native["raw_retirement"]["root_admission"]["store_root_state"],
        "policy": {"mode": "anchored_compact", "token_budget": snapshot["selected_view"]["token_budget"],
            "max_records": 64, "max_anchors": 4, "required_bundle": None},
        "input_sha256": h("synthetic original ReviewView semantic input"), "view_seal_sha256": ticket["view_digest"],
        "record_scope_sha256": h("synthetic original ReviewView scope"),
        "anchor_states": [native["raw_retirement"]["root_admission"]["store_root_state"]],
        "source_anchors": [], "record_metadata": []}
    output = native["raw_completion"]["output"]
    output["representation"]["input"] = native["input_digest"]
    if role == "critic":
        count = len(snapshot["encoded_model_input"]["divergences"])
        output["payload"]["divergence_logits_bits"] = [0] * count
        numbers = output["payload"]["candidate_logits_bits"] + output["payload"]["divergence_logits_bits"] + output["payload"]["wdl_bits"]
        framed = (b"rz-rovent-accepted-critic-head/1\0" + len(snapshot["legal_moves"]).to_bytes(8, "little")
            + count.to_bytes(8, "little") + b"".join(bits.to_bytes(4, "little") for bits in numbers))
        native["output_digest"] = hashlib.sha256(framed).hexdigest()
        native["raw_completion"]["output_digest"] = native["raw_retirement"]["output_digest"] = native["output_digest"]
    source, label = record["head_sources"]["policy"], record["targets"]["policy"]
    source["raw_receipt"]["input_sha256"] = record["input_sha256"]
    source["raw_receipt"]["binding"]["input_scope"].update(input_sha256=record["input_sha256"], view_sha256=snapshot["view_sha256"])
    label["view_sha256"] = snapshot["view_sha256"]
    seal_native(record, "rz-rovent-raw-record/3")
    reseal_head(record, "policy")
    seal_record(manifest, record)
    projected = {**record, "record_sha256": "", "input": {"sealed_input_sha256": record["input_sha256"]}}
    record["record_sha256"] = data.digest("rz-rovent-training-record/2", projected)
    return manifest, record


def validate_binding(manifest, record, *, submitted=False):
    raw = record["head_sources"]["policy"]["raw_receipt"]
    function = teacher.validate_collection_teacher_binding_v3 if submitted else teacher.validate_collection_teacher_binding
    return function(manifest, record, raw["actor_id"], raw["teacher_pin"], raw["binding"])


def validate_head(manifest, record):
    label = record["targets"]["policy"]
    corpus._target_policy(record["input"], label)
    corpus._head_sources(manifest, record, {**record["targets"], "nnuec_wdl": None})


def validate_row(manifest, record):
    indexed = {key: record[key] for key in ("record_sha256", "input_sha256", "origin_sha256", "split")}
    indexed.update({key: record["input"][key] for key in ("game_id", "opening_id", "line_genealogy_id", "transposition_sha256")})
    indexed.update(copy.deepcopy(record["lineage"]))
    return corpus._validate_row(manifest, record, indexed, corpus.CorpusLimits())


class CollectionTeacherTests(unittest.TestCase):
    def setUp(self):
        self.guard = NeuralImportsForbidden()
        sys.meta_path.insert(0, self.guard)

    def tearDown(self):
        sys.meta_path.remove(self.guard)

    def test_collection_full_corpus_row_keeps_conditional_rank_policy_and_no_wdl(self):
        manifest, record = fixture()
        result = validate_row(manifest, record)
        denominator = 1 + math.exp(-1)
        self.assertEqual(result["targets"]["policy"], [math.exp(-1) / denominator, 1 / denominator])
        self.assertIsNone(result["targets"]["wdl"])
        self.assertEqual(record["head_sources"]["policy"]["raw_receipt"]["completion"]["raw"]["score"],
            {"kind": "mate_moves", "value": 3})

    def test_new_job_does_not_require_or_create_reciprocal_or_native_execution_ids(self):
        manifest, record = fixture()
        before = copy.deepcopy(record)
        validate_head(manifest, record)
        self.assertEqual(record, before)
        binding = record["head_sources"]["policy"]["raw_receipt"]["binding"]
        self.assertEqual(binding["job_id"], {"collection_owner": 9, "sequence": 71})
        self.assertFalse({"question", "question_digest", "question_sequence", "execution_epoch", "execution_sequence", "owner"} & set(binding))

    def test_collection_critic_row_uses_same_independent_teacher_without_r(self):
        manifest, record = fixture(role="critic")
        self.assertEqual(validate_row(manifest, record)["role"], "critic")
        binding = record["head_sources"]["policy"]["raw_receipt"]["binding"]
        self.assertEqual(binding["input_scope"]["role"], "critic")
        self.assertEqual(record["native"]["raw_retirement"]["root_admission"]["native_origin"]["mode"], "HybridPc")

    def test_generation_view_is_distinct_from_native_ticket_view(self):
        manifest, record = fixture()
        binding = record["head_sources"]["policy"]["raw_receipt"]["binding"]
        self.assertNotEqual(binding["input_scope"]["view_sha256"], record["native"]["view_digest"])
        validate_head(manifest, record)
        binding["input_scope"]["view_sha256"] = record["native"]["view_digest"]
        reseal_head(record, "policy")
        with self.assertRaisesRegex(ValueError, "submitted input/view/source/role/Rules/frozen actor"):
            validate_head(manifest, record)

    def test_job_identity_is_positive_closed_and_bounded(self):
        for key, value in (("collection_owner", 0), ("collection_owner", 2**32), ("sequence", 0), ("sequence", True)):
            with self.subTest(key=key, value=value):
                manifest, record = fixture()
                record["head_sources"]["policy"]["raw_receipt"]["binding"]["job_id"][key] = value
                reseal_head(record, "policy")
                with self.assertRaisesRegex(ValueError, "positive bounded|bounded unsigned"):
                    validate_head(manifest, record)

    def test_scope_rejects_authority_and_implementation_crossing(self):
        for key, value in (("process_epoch", 2), ("game_generation", 2), ("root_generation", 2), ("implementation_sha256", h("other implementation"))):
            with self.subTest(key=key):
                manifest, record = fixture()
                record["head_sources"]["policy"]["raw_receipt"]["binding"]["input_scope"]["authority"][key] = value
                reseal_head(record, "policy")
                with self.assertRaisesRegex(ValueError, "original source authority|original implementation"):
                    validate_head(manifest, record)

    def test_input_source_role_rules_and_frozen_actor_are_exact(self):
        for key, value in (("input_sha256", h("other input")), ("source_sha256", h("other source")),
            ("role", "validator"), ("rules_semantic", h("other Rules endpoint")), ("frozen_actor", h("other teacher actor"))):
            with self.subTest(key=key):
                manifest, record = fixture()
                record["head_sources"]["policy"]["raw_receipt"]["binding"]["input_scope"][key] = value
                reseal_head(record, "policy")
                with self.assertRaisesRegex(ValueError, "submitted input/view/source/role/Rules/frozen actor"):
                    validate_head(manifest, record)

    def test_metadata_cannot_add_synthetic_question_or_execution_fields(self):
        for key in ("question", "execution_sequence", "unrecognized"):
            with self.subTest(key=key):
                manifest, record = fixture()
                record["head_sources"]["policy"]["raw_receipt"]["binding"][key] = 1
                reseal_head(record, "policy")
                with self.assertRaisesRegex(ValueError, "Collection teacher dispatch binding fields mismatch"):
                    validate_head(manifest, record)

    def test_process_and_go_constraint_keep_existing_physical_contract(self):
        manifest, record = fixture()
        raw = record["head_sources"]["policy"]["raw_receipt"]
        raw["binding"]["process"]["pid"] = 0
        reseal_head(record, "policy")
        with self.assertRaisesRegex(ValueError, "known original teacher physical process"):
            validate_head(manifest, record)
        manifest, record = fixture()
        record["head_sources"]["policy"]["raw_receipt"]["binding"]["requested"]["reported_nodes"] = 0
        reseal_head(record, "policy")
        with self.assertRaisesRegex(ValueError, "positive original teacher go limit"):
            validate_head(manifest, record)

    def test_job_deadline_stays_within_original_root_domain_and_hard_limit(self):
        for key, value in (("clock_epoch", 2), ("at_ns", 1001)):
            with self.subTest(key=key):
                manifest, record = fixture()
                record["head_sources"]["policy"]["raw_receipt"]["binding"]["original_deadline"][key] = value
                reseal_head(record, "policy")
                with self.assertRaisesRegex(ValueError, "Job deadline crossed original root deadline"):
                    validate_head(manifest, record)

    def test_scope_seals_do_not_substitute_wrapper_digest_for_descriptor_digest(self):
        manifest, record = fixture()
        binding = record["head_sources"]["policy"]["raw_receipt"]["binding"]
        wrapper_sha = record["native"]["raw_retirement"]["root_admission_sha256"]
        self.assertNotEqual(wrapper_sha, binding["root_descriptor_sha256"])
        binding["input_scope"]["root_admission"] = wrapper_sha
        reseal_head(record, "policy")
        with self.assertRaisesRegex(ValueError, "scope changed original InputLock/root descriptor/budget"):
            validate_head(manifest, record)

    def test_descriptor_has_independent_domain_nul_framing(self):
        manifest, record = fixture()
        binding = record["head_sources"]["policy"]["raw_receipt"]["binding"]
        descriptor = binding["root_descriptor"]
        self.assertEqual(binding["root_descriptor_sha256"], domain_nul("rz-rovent-root-admission/1", descriptor))
        teacher.validate_root_descriptor(descriptor, binding["root_descriptor_sha256"])
        with self.assertRaisesRegex(ValueError, "domain-NUL seals differ"):
            teacher.validate_root_descriptor(descriptor, data.digest("rz-rovent-root-admission/1", descriptor))

    def test_descriptor_closed_fields_types_and_clock_order_are_checked_before_seals(self):
        cases = (("extra", "root descriptor input fields mismatch"), ("bool", "bounded unsigned"),
                 ("clock", "deadline order"), ("go", "parsed go differs"))
        for mutation, error in cases:
            with self.subTest(mutation=mutation):
                _, record = fixture()
                binding = record["head_sources"]["policy"]["raw_receipt"]["binding"]
                inp = binding["root_descriptor"]["input"]
                if mutation == "extra":
                    inp["unknown"] = 1
                elif mutation == "bool":
                    inp["authority"]["epoch"] = True
                elif mutation == "clock":
                    inp["clock"]["soft"]["at_ns"] = 1001
                else:
                    inp["original_go"]["original_parsed_limits"]["nodes"] = 4097
                with self.assertRaisesRegex(ValueError, error):
                    teacher.validate_root_descriptor(binding["root_descriptor"], binding["root_descriptor_sha256"])

    def test_native_root_source_must_still_match_original_collection_manifest(self):
        manifest, record = fixture()
        record["native"]["raw_retirement"]["root_admission"]["native_origin"]["source_sha256"] = h("other collection source")
        seal_root(record)
        with self.assertRaisesRegex(ValueError, "collection source/runtime/model domain mismatch"):
            validate_head(manifest, record)

    def test_native_root_budget_and_descriptor_must_be_the_same_original(self):
        manifest, record = fixture()
        record["native"]["raw_retirement"]["root_admission"]["root_budget_sha256"] = h("different root budget")
        seal_root(record)
        with self.assertRaisesRegex(ValueError, "descriptor.*root|root.*descriptor"):
            validate_head(manifest, record)

    def test_reserialized_valid_descriptor_cannot_change_original_input_lock(self):
        manifest, record = fixture()
        binding = record["head_sources"]["policy"]["raw_receipt"]["binding"]
        binding["root_descriptor"]["input"]["startup_lock_sha256"] = h("another InputLock")
        binding["root_descriptor_sha256"] = seal_descriptor(binding["root_descriptor"])
        reseal_head(record, "policy")
        with self.assertRaisesRegex(ValueError, "original lock/root/budget/go/deadline|descriptor"):
            validate_head(manifest, record)

    def test_same_board_with_other_exact_history_is_not_the_submitted_endpoint(self):
        manifest, record = fixture()
        binding = record["head_sources"]["policy"]["raw_receipt"]["binding"]
        binding["position_anchor"]["rules_history_sha256"] = h("other exact history")
        reseal_head(record, "policy")
        with self.assertRaisesRegex(ValueError, "exact Rules endpoint"):
            validate_head(manifest, record)

    def test_root_restriction_checks_original_move_order(self):
        manifest, record = fixture()
        binding = record["head_sources"]["policy"]["raw_receipt"]["binding"]
        binding["root_moves"] = [128, 64]
        # a1c1 then a1b1, packed in this exact submitted order.
        encoded = b"rz-external-root-restriction/1\0\1" + (2).to_bytes(8, "little")
        for move in (b"a1c1", b"a1b1"):
            encoded += len(move).to_bytes(8, "little") + move
        binding["root_restriction"] = hashlib.sha256(encoded).hexdigest()
        reseal_head(record, "policy")
        validate_head(manifest, record)
        binding["root_moves"].reverse()
        reseal_head(record, "policy")
        with self.assertRaisesRegex(ValueError, "root restriction seal changed"):
            validate_head(manifest, record)

    def test_new_collection_requires_native_v2_original_root_data(self):
        manifest, record = fixture()
        record["native"]["raw_submission"]["schema"] = "rz-rovent-native-owner-event/1"
        with self.assertRaisesRegex(ValueError, "original native /2 root data"):
            validate_head(manifest, record)

    def test_descriptor_accepts_original_rust_thread_and_topology_ranges(self):
        _, record = fixture()
        descriptor = record["head_sources"]["policy"]["raw_receipt"]["binding"]["root_descriptor"]
        resources = descriptor["input"]["declared_ceilings"]["resource_plan"]
        resources["pools"][0]["max_threads"] = 1024
        resources["topology_policy"] = "x" * 4096
        self.assertIs(teacher.validate_root_descriptor(descriptor, seal_descriptor(descriptor)), descriptor["input"])

    def test_descriptor_actor_enum_and_canonical_set_order_match_rust(self):
        for actors, error in ((["not_a_rovent_actor"], "actor enum"),
            (["teacher", "manager"], "canonical set order"), (["manager", "manager"], "canonical set order")):
            with self.subTest(actors=actors):
                _, record = fixture()
                descriptor = record["head_sources"]["policy"]["raw_receipt"]["binding"]["root_descriptor"]
                descriptor["input"]["declared_ceilings"]["resource_plan"]["pools"][0]["actors"] = actors
                with self.assertRaisesRegex(ValueError, error):
                    teacher.validate_root_descriptor(descriptor, seal_descriptor(descriptor))

    def test_teacher_readiness_and_private_physical_summary_are_still_required(self):
        for mutation, error in (("readiness", "request/go/readiness/profile"), ("quarantined", "unknown/lost/quarantined")):
            with self.subTest(mutation=mutation):
                manifest, record = fixture()
                raw = record["head_sources"]["policy"]["raw_receipt"]
                if mutation == "readiness":
                    raw["completion"]["readiness_barrier_completed"] = False
                else:
                    raw["completion"]["attempt"]["process"]["quarantined"] = True
                reseal_head(record, "policy")
                with self.assertRaisesRegex(ValueError, error):
                    validate_head(manifest, record)

    def test_old_teacher_body_and_digest_are_preserved_exactly(self):
        _, manifest, record = original_fixture()
        add_teacher_policy(manifest, record)
        before = copy.deepcopy(record)
        validate_head(manifest, record)
        self.assertEqual(record, before)
        self.assertNotIn("root_descriptor", record["head_sources"]["policy"]["raw_receipt"]["binding"])

    def test_legacy_descriptor_variant_joins_original_root_without_changing_old_schema(self):
        manifest, record = fixture(collection=False)
        before = copy.deepcopy(record)
        validate_head(manifest, record)
        self.assertEqual(record, before)
        self.assertNotIn("schema", record["head_sources"]["policy"]["raw_receipt"]["binding"])

    def test_legacy_descriptor_pair_is_closed_not_arbitrary_optional_extras(self):
        for mutation in ("root_descriptor", "root_descriptor_sha256", "unknown"):
            with self.subTest(mutation=mutation):
                manifest, record = fixture(collection=False)
                binding = record["head_sources"]["policy"]["raw_receipt"]["binding"]
                if mutation == "unknown":
                    binding["unknown"] = 1
                else:
                    del binding[mutation]
                reseal_head(record, "policy")
                with self.assertRaisesRegex(ValueError, "original teacher dispatch binding fields mismatch"):
                    validate_head(manifest, record)

    def test_native_root_without_descriptor_preserves_historical_exact_variant(self):
        manifest, record = fixture()
        root = record["native"]["raw_retirement"]["root_admission"]
        del root["root_descriptor"]
        del root["root_descriptor_sha256"]
        seal_root(record)
        before = copy.deepcopy(record)
        corpus._native(manifest, record)
        self.assertEqual(record, before)

    def test_native_root_descriptor_fields_are_an_exact_pair(self):
        for missing in ("root_descriptor", "root_descriptor_sha256"):
            with self.subTest(missing=missing):
                manifest, record = fixture()
                del record["native"]["raw_retirement"]["root_admission"][missing]
                seal_root(record)
                with self.assertRaisesRegex(ValueError, "native /2 original root admission fields mismatch"):
                    corpus._native(manifest, record)

    def test_current_root_optional_zero_nodes_matches_rust_without_rewriting_old_variant(self):
        manifest, record = fixture()
        binding = record["head_sources"]["policy"]["raw_receipt"]["binding"]
        root = record["native"]["raw_retirement"]["root_admission"]
        descriptor = binding["root_descriptor"]
        descriptor["input"]["original_go"]["requested_nodes"] = 0
        descriptor["input"]["original_go"]["original_parsed_limits"]["nodes"] = 0
        descriptor_sha = seal_descriptor(descriptor)
        binding["root_descriptor_sha256"] = binding["input_scope"]["root_admission"] = descriptor_sha
        binding["input_scope"]["root_budget"] = descriptor["root_budget_sha256"]
        root.update(requested_nodes=0, root_descriptor=copy.deepcopy(descriptor), root_descriptor_sha256=descriptor_sha,
            root_budget_sha256=descriptor["root_budget_sha256"])
        seal_root(record)
        reseal_head(record, "policy")
        seal_record(manifest, record)
        self.assertIsNotNone(validate_row(manifest, record)["targets"]["policy"])
        del root["root_descriptor"]
        del root["root_descriptor_sha256"]
        seal_root(record)
        with self.assertRaisesRegex(ValueError, "positive root search limit"):
            corpus._native(manifest, record)

    def test_explicit_raw3_entry_preserves_new_submitted_input_and_generation_view_seal(self):
        for role in ("proposer", "critic"):
            with self.subTest(role=role):
                manifest, record = submitted_fixture(role=role)
                before = copy.deepcopy(record)
                submitted_native.validate_native_v3(manifest, record)
                validate_binding(manifest, record, submitted=True)
                self.assertEqual(record, before)
                scope = record["head_sources"]["policy"]["raw_receipt"]["binding"]["input_scope"]
                self.assertEqual(scope["input_sha256"], submitted_inputs.seal_input(record["input"]))
                self.assertEqual(scope["view_sha256"], record["input"]["view_sha256"])
                self.assertNotEqual(scope["view_sha256"], record["native"]["view_digest"])

    def test_legacy_entry_rejects_raw3_without_retagging_or_auto_dispatch(self):
        manifest, record = submitted_fixture()
        before = copy.deepcopy(record)
        with self.assertRaisesRegex(ValueError, "original native /2 root data"):
            validate_binding(manifest, record)
        self.assertEqual(record, before)

    def test_new_entry_rejects_legacy_family_without_retagging_or_auto_conversion(self):
        manifest, record = fixture()
        before = copy.deepcopy(record)
        with self.assertRaisesRegex(ValueError, "raw /3 entry requires the submitted corpus family"):
            validate_binding(manifest, record, submitted=True)
        self.assertEqual(record, before)

    def test_raw3_entry_does_not_accept_a_mixed_completion_family(self):
        manifest, record = submitted_fixture()
        record["native"]["raw_completion"]["schema"] = "rz-rovent-native-owner-event/2"
        with self.assertRaisesRegex(ValueError, "record/native wire family mismatch|one schema revision"):
            validate_binding(manifest, record, submitted=True)

    def test_raw3_entry_recomputes_new_seal_before_accepting_bound_scope(self):
        manifest, record = submitted_fixture()
        record["input"]["game_id"] = "another-game-with-unreissued-input-seal"
        with self.assertRaisesRegex(ValueError, "raw /3 original submitted input seal changed"):
            validate_binding(manifest, record, submitted=True)

    def test_raw3_entry_keeps_original_scope_and_loaded_root_checks(self):
        manifest, record = submitted_fixture(role="critic")
        binding = record["head_sources"]["policy"]["raw_receipt"]["binding"]
        binding["input_scope"]["view_sha256"] = record["native"]["view_digest"]
        with self.assertRaisesRegex(ValueError, "submitted input/view/source/role/Rules/frozen actor"):
            validate_binding(manifest, record, submitted=True)
        manifest, record = submitted_fixture()
        root = record["native"]["raw_retirement"]["root_admission"]
        root["native_origin"]["source_sha256"] = h("other native source")
        root["native_origin_sha256"] = data.digest("rz-rovent-collection-native-origin/1", root["native_origin"])
        record["native"]["raw_retirement"]["root_admission_sha256"] = data.digest("rz-rovent-collection-root-admission/1", root)
        with self.assertRaisesRegex(ValueError, "collection source/runtime/model domain mismatch"):
            validate_binding(manifest, record, submitted=True)

    def test_legacy_descriptor_path_stays_raw2_only(self):
        manifest, record = fixture(collection=False)
        raw = record["head_sources"]["policy"]["raw_receipt"]
        for name in ("raw_submission", "raw_completion", "raw_retirement"):
            record["native"][name]["schema"] = "rz-rovent-raw-record/3"
        before = copy.deepcopy(record)
        with self.assertRaisesRegex(ValueError, "original native /2 root data"):
            teacher.validate_legacy_teacher_root(manifest, record, raw["binding"])
        self.assertEqual(record, before)


if __name__ == "__main__":
    unittest.main()
