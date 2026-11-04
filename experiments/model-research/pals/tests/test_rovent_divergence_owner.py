"""Synthetic historical witness bytes only; no Rules/native/training authority.

Binary expectations below follow repair_witness.rs and native_acceptance.rs,
independently of the Python validator. None of these fixtures came from an
actual accepted owner, and their JSON seals cannot create one.
"""
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
            raise AssertionError("historical witness tests cannot import neural runtimes")


_guard = NeuralImportsForbidden()
sys.meta_path.insert(0, _guard)
try:
    from rz_pals_model import generation_data as data
    from rz_pals_model import rovent_divergence_owner as owner
    from rz_pals_model import rovent_divergence as targets
    from rz_pals_model import rovent_submitted_input as submitted
    from rz_pals_model import rovent_corpus as corpus
    from rz_pals_model.virtual_review import input_digest
    from test_rovent_historical_corpus import synthetic_submitted, reseal_submitted
    from test_rovent_submitted_input import fixture as input_fixture, components
finally:
    sys.meta_path.remove(_guard)


def h(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def binary_seal(domain, payload):
    return hashlib.sha256(domain.encode("ascii") + b"\0" + payload).hexdigest()


def json_seal(domain, value):
    # Independent canonical expected bytes; do not call a validator seal helper.
    payload = json.dumps([domain, value], sort_keys=True, ensure_ascii=False,
        allow_nan=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def u64(value):
    return value.to_bytes(8, "little")


def sha_bytes(value):
    return bytes.fromhex(value)


def line_bytes(moves):
    return u64(len(moves)) + b"".join(move.to_bytes(2, "little") for move in moves)


def f32_bits(values):
    return [struct.unpack("<I", struct.pack("<f", value))[0] for value in values]


def root_expectation(bits):
    values = [struct.unpack("<f", value.to_bytes(4, "little"))[0] for value in bits]
    return struct.unpack("<f", struct.pack("<f", values[0] - values[2]))[0]


def seal_origin(origin):
    ticket, rep = origin["ticket"], origin["ticket"]["representation"]
    payload = (u64(origin["frame"]["sequence"]) + u64(ticket["request_sequence"])
        + u64(ticket["process_epoch"]) + u64(ticket["game_generation"]) + u64(ticket["root_generation"])
        + sha_bytes(ticket["implementation_sha256"]) + sha_bytes(ticket["question_digest"])
        + sha_bytes(ticket["view_digest"]) + u64(ticket["view_sequence"]) + sha_bytes(origin["output_digest"]))
    payload += b"".join(sha_bytes(rep[key]) for key in ("input", "history", "candidates", "mask_positions", "model", "encoding"))
    payload += (u64(rep["frozen_epoch"]) + u64(rep["record_revision"])
        + bytes([{"proposer": 0, "critic": 1}[rep["role"]], 0]) + u64(ticket["deadline_tick"]))
    if origin["kind"] == "local_accepted":
        payload += b"\0"
    elif origin["kind"] == "peer_question" and origin["peer"] is not None:
        peer = origin["peer"]
        payload += b"\1" + u64(peer["id"]["sequence"]) + sha_bytes(peer["question_digest"])
    else:
        raise ValueError("the independent fixture needs an explicit local or complete peer origin")
    origin["seal_sha256"] = binary_seal("rz-rovent-accepted-local-origin-seal/1", payload)


def origin_for(ticket, *, frame, output):
    value = {"frame": {key: ticket[key] for key in
        ("process_epoch", "game_generation", "root_generation", "implementation_sha256")},
        "ticket": copy.deepcopy(ticket), "output_digest": output, "kind": "local_accepted", "peer": None,
        "seal_sha256": ""}
    value["frame"]["sequence"] = frame
    seal_origin(value)
    return value


def peer_for(origin, root_budget):
    """Exact Rust peer_json roster; synthetic metadata, never a live envelope."""
    ticket = origin["ticket"]
    authority = {key: ticket[key] for key in
        ("process_epoch", "game_generation", "root_generation", "implementation_sha256")}
    view = {"authority": copy.deepcopy(authority), **{key: copy.deepcopy(ticket[key]) for key in
        ("view_sequence", "context_policy", "view_digest", "root_state", "actual_tokens",
         "selected_records", "anchors", "length_bucket")}}
    return {"id": {"authority": copy.deepcopy(authority), "sequence": 1001},
        "parent": {"authority": copy.deepcopy(authority), "sequence": 1000},
        "sender": "Proposer", "recipient": "Critic", "purpose": "Counter",
        "anchor": copy.deepcopy(ticket["root_state"]), "anchor_root_ply": 0,
        "original_first_move": 64, "prefix_digest": h("original reciprocal prefix"),
        "line_digest": h("original reciprocal line"), "line_offset": 0,
        "question_digest": ticket["question_digest"], "view": view,
        "evidence_revision": ticket["representation"]["record_revision"], "root_budget": root_budget,
        "original_deadline": {"clock": ticket["deadline_clock"], "tick": ticket["deadline_tick"]}}


def reseal_witness(witness):
    source = witness["source"]
    seal_origin(source["accepted"])
    source["native_source_sha256"] = binary_seal("rz-rovent-pc-divergence-source/1",
        sha_bytes(source["accepted"]["seal_sha256"]) + sha_bytes(source["root_budget_sha256"])
        + source["counter_local_index"].to_bytes(2, "little") + u64(source["counter_revision"])
        + source["divergence_index"].to_bytes(2, "little") + u64(source["divergence_ply"])
        + line_bytes(source["proposal"]) + line_bytes(source["counter"]))
    for name in ("repair", "recheck"):
        value = witness[name]
        if value is not None:
            seal_origin(value["accepted"])
            value["native_line_sha256"] = binary_seal("rz-rovent-pc-accepted-repair-line/1",
                sha_bytes(value["accepted"]["seal_sha256"]) + u64(value["line_id"]) + u64(value["observation_id"])
                + u64(value["revision"]) + bytes([{"repair": 2, "counterexample": 1}[value["kind"]]])
                + line_bytes(value["moves"]))
    for name in ("repaired_endpoint", "counter_endpoint"):
        value = witness[name]
        if value is not None:
            seal_origin(value["accepted"])
            physical = value["physical"]
            value["native_endpoint_sha256"] = binary_seal("rz-rovent-pc-accepted-fresh-endpoint/1",
                sha_bytes(value["accepted"]["seal_sha256"]) + sha_bytes(value["pair"]["public_context"])
                + u64(physical["execution_epoch"]) + u64(physical["execution_sequence"])
                + u64(physical["completed_nn_inputs"]) + bytes([{"white": 0, "black": 1}[value["perspective"]]])
                + b"".join(bits.to_bytes(4, "little") for bits in value["raw_wdl_bits"])
                + line_bytes(value["line"]))
    if witness["native_hash_kind"] == "frozen_pair":
        framed = sha_bytes(source["native_source_sha256"])
        framed += b"".join(sha_bytes(witness[name][key]) for name, key in
            (("repair", "native_line_sha256"), ("recheck", "native_line_sha256"),
             ("repaired_endpoint", "native_endpoint_sha256"), ("counter_endpoint", "native_endpoint_sha256")))
        framed += u64(witness["event_sequence"]) + bytes([
            {"supported": 0, "refuted": 1, "unknown": 2, "disputed": 3}[witness["resolution"]]])
        domain = "rz-rovent-pc-divergence-witness/1"
    else:
        framed = sha_bytes(source["native_source_sha256"])
        for name, key in (("repair", "native_line_sha256"), ("recheck", "native_line_sha256"),
                          ("repaired_endpoint", "native_endpoint_sha256"), ("counter_endpoint", "native_endpoint_sha256")):
            value = witness[name]
            framed += b"\0" if value is None else b"\1" + sha_bytes(value[key])
        reason = witness["reason"].encode("utf-8")
        framed += u64(witness["event_sequence"]) + u64(len(reason)) + reason
        domain = "rz-rovent-pc-divergence-unresolved/1"
    witness["native_witness_sha256"] = binary_seal(domain, framed)


def reseal_head(source, label):
    source["raw_receipt_sha256"] = json_seal("rz-rovent-head-source/1", source["raw_receipt"])
    if label is not None:
        label["raw_evidence_sha256"] = source["raw_receipt_sha256"]


def reseal_original_row(manifest, record, head, label):
    """Reseal a changed synthetic submitted cell and its original C output."""
    snapshot, native = record["input"], record["native"]
    record["input_sha256"] = submitted.seal_input(snapshot)
    native["input_digest"] = input_digest(snapshot["encoded_model_input"])
    ticket = native["raw_submission"]["ticket"]
    ticket["representation"]["input"] = native["input_digest"]
    output = native["raw_completion"]["output"]
    output["representation"] = copy.deepcopy(ticket["representation"])
    bits = output["payload"]
    count = len(bits["divergence_logits_bits"])
    framed = (b"rz-rovent-accepted-critic-head/1\0" + u64(len(snapshot["legal_moves"])) + u64(count)
        + b"".join(value.to_bytes(4, "little") for value in
            (bits["candidate_logits_bits"] + bits["divergence_logits_bits"] + bits["wdl_bits"])))
    native["output_digest"] = hashlib.sha256(framed).hexdigest()
    for name in ("raw_submission", "raw_completion", "raw_retirement"):
        native[name].update(input_sha256=record["input_sha256"], ticket=copy.deepcopy(ticket))
        if name != "raw_submission":
            native[name]["output_digest"] = native["output_digest"]
    native["raw_submission"]["input_digest"] = native["input_digest"]
    source = head["raw_receipt"]["witness"]["source"]
    source["accepted"]["ticket"] = copy.deepcopy(ticket)
    source["accepted"]["output_digest"] = native["output_digest"]
    head["raw_receipt"]["input_sha256"] = record["input_sha256"]
    reseal_witness(head["raw_receipt"]["witness"])
    reseal_head(head, label)
    reseal_submitted(manifest, record)


def fixture(*, resolution="supported", cycle_sequence=1, collector_sequence=110):
    """A synthetic visible C counter and four later accepted metadata leaves."""
    _recipe, manifest, record, _lock, _binding = synthetic_submitted()
    snapshot = input_fixture(selected_counter=True, proposal_length=5)
    snapshot.update(opening_id=record["input"]["opening_id"], model_epoch=1, context_epoch=1,
        capture_sequence=100, question_id=h("source critic question"))
    proposal, counter = [64, 128, 192, 256, 320], [64, 128, 384, 448, 512]
    repair, recheck = [64, 128, 576, 448, 512], [64, 128, 576, 640, 512]
    view, encoded = snapshot["selected_view"], snapshot["encoded_model_input"]
    view["records"][0]["moves"] = counter
    view["query"].update(proposal=proposal, counter=counter)
    encoded["records"][0]["line"] = [components(move) for move in counter]
    encoded["query"].update(proposal=[components(move) for move in proposal], counter=[components(move) for move in counter])
    snapshot["view_sha256"] = data.view_digest(view)
    record.update(role="critic", input=snapshot, input_sha256=submitted.seal_input(snapshot))
    native = record["native"]
    native.update(head="critic", input_digest=input_digest(encoded), submitted_sequence=100, completed_sequence=101, retired_sequence=102)
    ticket = copy.deepcopy(native["raw_submission"]["ticket"])
    ticket.update(question_digest=snapshot["question_id"], actual_tokens=15, selected_records=1)
    ticket["representation"].update(input=native["input_digest"], role="critic", record_revision=23)
    output = native["raw_completion"]["output"]
    output["representation"] = copy.deepcopy(ticket["representation"])
    output["payload"].update(kind="counterexample", divergence_logits_bits=[0])
    bits = output["payload"]
    framed = (b"rz-rovent-accepted-critic-head/1\0" + u64(len(snapshot["legal_moves"])) + u64(1)
        + b"".join(value.to_bytes(4, "little") for value in
            (bits["candidate_logits_bits"] + bits["divergence_logits_bits"] + bits["wdl_bits"])))
    native["output_digest"] = hashlib.sha256(framed).hexdigest()
    for name, sequence in (("raw_submission", 100), ("raw_completion", 101), ("raw_retirement", 102)):
        raw = native[name]
        raw.update(head="critic", input_sha256=record["input_sha256"], ticket=copy.deepcopy(ticket), sequence=sequence)
        if name != "raw_submission":
            raw["output_digest"] = native["output_digest"]
    native["raw_submission"]["input_digest"] = native["input_digest"]
    review = native["raw_submission"]["submitted_review"]
    replay = native["raw_submission"]["rules_replay"]["actual_root"]
    review.update(source_anchors=[copy.deepcopy(replay)], record_metadata=[{"local_index": 0,
        "record_revision": 23, "parent_revision": None, "supersedes_revision": None,
        "origin_state": 1, "source_state": 1, "source_revision": 23, "source_line_offset": 0,
        "original_first_move": 64, "value": -17, "completed_depth": 8,
        "score_scope": "CompletedIteration", "critical": True}])
    reseal_submitted(manifest, record)
    original = origin_for(ticket, frame=11, output=native["output_digest"])
    def later(sequence, role, revision):
        result = copy.deepcopy(ticket)
        result.update(request_sequence=sequence, question_digest=h("question " + str(sequence)),
            view_sequence=sequence, view_digest=h("native view " + str(sequence)))
        result["representation"].update(input=h("input " + str(sequence)), role=role, record_revision=revision)
        return origin_for(result, frame=11 + sequence, output=h("accepted output " + str(sequence)))
    repair_origin, recheck_origin = later(2, "proposer", 23), later(3, "critic", 24)
    endpoint_origins = (later(4, "proposer", 25), later(5, "proposer", 25))
    root = native["raw_retirement"]["root_admission"]
    source = {"native_source_sha256": "", "accepted": original, "root_budget_sha256": root["root_budget_sha256"],
        "counter_local_index": 0, "counter_revision": 23, "divergence_index": 0, "divergence_ply": 1,
        "first_move": 64, "proposal": proposal, "counter": counter, "perspective": "white"}
    pair_ticket = endpoint_origins[0]["ticket"]
    pair = {"public_context": h("frozen public context"),
        "authority": {key: ticket[key] for key in ("process_epoch", "game_generation", "root_generation", "implementation_sha256")},
        "root_state": copy.deepcopy(ticket["root_state"]), "context_policy": ticket["context_policy"],
        "selected_records": 1, "anchors": 1, **{key: pair_ticket["representation"][key] for key in
            ("model", "encoding", "precision", "frozen_epoch", "record_revision")}, "role": "proposer", "perspective": "white"}
    loaded = root["native_origin"]
    model = {"semantics": "rz-pals-context-wdl/1;side-to-move;restricted-model-estimate;actual-prepared-input;no-cp-calibration;no-rules-proof",
        "model": "pals-native-virtual-v3:" + ":".join(loaded[key]
            for key in ("export_manifest_sha256", "weights_sha256", "backend_sha256")),
        "encoding": loaded["encoding"] + ":" + loaded["encoding_registry_sha256"],
        "precision": "fp32_tf32_off", "model_epoch_sha256": loaded["weights_sha256"]}
    probabilities = ([0.25, 0.5, 0.25], [0.5, 0.25, 0.25])
    if resolution == "refuted":
        probabilities = probabilities[::-1]
    elif resolution == "unknown":
        probabilities = (probabilities[0], probabilities[0])
    endpoints = []
    for accepted, moves, values in zip(endpoint_origins, (repair, recheck), probabilities):
        endpoints.append({"native_endpoint_sha256": "", "accepted": accepted, "line": moves, "pair": copy.deepcopy(pair),
            "model": copy.deepcopy(model), "perspective": "white", "raw_wdl_bits": f32_bits(values[::-1]), "root_wdl_bits": f32_bits(values),
            "physical": {"execution_epoch": 1, "execution_sequence": accepted["ticket"]["request_sequence"],
                "completed_nn_inputs": 1, "measurement_complete": True, "physical_fresh": True}})
    witness = {"schema": "rz-rovent-pc-divergence-witness/1", "native_witness_sha256": "",
        "native_hash_kind": "frozen_pair", "event_sequence": cycle_sequence,
        "reason": "equal frozen conditional expectations" if resolution == "unknown" else
            "actual Repair and C continuation; frozen model conditional comparison",
        "resolution": resolution, "target_known": resolution != "unknown", "source": source,
        "repair": {"native_line_sha256": "", "accepted": repair_origin, "line_id": 2, "observation_id": 3,
            "revision": 24, "kind": "repair", "moves": repair},
        "recheck": {"native_line_sha256": "", "accepted": recheck_origin, "line_id": 4, "observation_id": 5,
            "revision": 25, "kind": "counterexample", "moves": recheck},
        "repaired_endpoint": endpoints[0], "counter_endpoint": endpoints[1]}
    reseal_witness(witness)
    raw = {"schema": "rz-rovent-head-source/1", "kind": "pc_divergence", "input_sha256": record["input_sha256"],
        "anchor": copy.deepcopy(snapshot["rules_anchor"]), "observed_sequence": collector_sequence, "witness": witness}
    head = {"actor_id": record["lineage"]["actor_id"], "source": "rovent_actual_pc_divergence",
        "binary_sha256": None, "profile_sha256": None, "raw_receipt": raw, "raw_receipt_sha256": "",
        "observed_sequence": collector_sequence, "anchor": copy.deepcopy(snapshot["rules_anchor"])}
    label = None if resolution == "unknown" else {"schema": "rz-rovent-divergence-target/1",
        "observed_sequence": collector_sequence, "question_id": snapshot["question_id"], "view_sha256": snapshot["view_sha256"],
        "raw_evidence_sha256": "", "source": "rovent_resolved_pc_divergence", "slot": 0, "attack_ply": 1,
        "counter_question_id": ticket["question_digest"], "counter_native_view_sha256": ticket["view_digest"],
        "counter_local_index": 0, "counter_revision": 23, "repair_question_id": repair_origin["ticket"]["question_digest"],
        "status": "supported_after_repair" if resolution == "supported" else "refuted_by_repair"}
    reseal_head(head, label)
    record["head_sources"]["divergence"] = head
    record["targets"]["divergence"], record["divergence_index"] = label, None if label is None else 0
    reseal_submitted(manifest, record)
    return manifest, record, head, label


def peer_fixture():
    manifest, record, head, label = fixture()
    source = head["raw_receipt"]["witness"]["source"]
    origin = source["accepted"]
    origin.update(kind="peer_question", peer=peer_for(origin, source["root_budget_sha256"]))
    reseal_witness(head["raw_receipt"]["witness"])
    reseal_head(head, label)
    reseal_submitted(manifest, record)
    return manifest, record, head, label


class PcDivergenceOwnerTests(unittest.TestCase):
    def setUp(self):
        sys.meta_path.insert(0, _guard)

    def tearDown(self):
        sys.meta_path.remove(_guard)

    def test_known_conditional_outcomes_keep_input_and_native_seals_unchanged(self):
        for resolution, expected in (("supported", 1.0), ("refuted", 0.0)):
            manifest, record, head, label = fixture(resolution=resolution)
            before = copy.deepcopy((record, head, label))
            seal = submitted.seal_input(record["input"])
            # This fixture must independently meet the full native /3 contract
            # before the narrower owner-head checks can count as positive.
            corpus._native(manifest, record)
            owner.validate_pc_divergence_head(record, head, label)
            self.assertEqual(targets.divergence_target(record["input"], label), expected)
            self.assertEqual((record, head, label), before)
            self.assertEqual(submitted.seal_input(record["input"]), seal)
            corpus._head_sources(manifest, record, {**record["targets"], "nnuec_wdl": None})

    def test_complete_original_peer_metadata_joins_the_native_ticket_without_changing_input(self):
        manifest, record, head, label = peer_fixture()
        before, input_seal = copy.deepcopy(record), submitted.seal_input(record["input"])
        origin = head["raw_receipt"]["witness"]["source"]["accepted"]
        self.assertEqual(set(origin["peer"]), {"id", "parent", "sender", "recipient", "purpose", "anchor",
            "anchor_root_ply", "original_first_move", "prefix_digest", "line_digest", "line_offset",
            "question_digest", "view", "evidence_revision", "root_budget", "original_deadline"})
        corpus._native(manifest, record)
        observed = owner.validate_pc_divergence_head(record, head, label)
        self.assertIs(observed, head["raw_receipt"]["witness"])
        corpus._head_sources(manifest, record, {**record["targets"], "nnuec_wdl": None})
        self.assertEqual(record, before)
        self.assertEqual(submitted.seal_input(record["input"]), input_seal)
        local = copy.deepcopy(origin)
        local.update(kind="local_accepted", peer=None)
        seal_origin(local)
        self.assertNotEqual(origin["seal_sha256"], local["seal_sha256"])

    def test_rehashed_peer_view_deadline_parent_budget_and_recipient_must_join_original_scope(self):
        for change in ("allocation", "deadline_clock", "parent_sequence", "root_budget", "recipient"):
            manifest, record, head, label = peer_fixture()
            witness = head["raw_receipt"]["witness"]
            peer = witness["source"]["accepted"]["peer"]
            corpus._native(manifest, record)
            if change == "allocation":
                peer["view"]["actual_tokens"] += 1
            elif change == "deadline_clock":
                peer["original_deadline"]["clock"] += 1
            elif change == "parent_sequence":
                peer["parent"]["sequence"] = peer["id"]["sequence"]
            elif change == "root_budget":
                peer["root_budget"] = h("unrelated root ledger")
            else:
                peer["recipient"] = "Proposer"
            reseal_witness(witness)
            reseal_head(head, label)
            boundary = {"allocation": "peer view differs from its accepted ticket",
                "deadline_clock": "peer authority/question/deadline differs",
                "parent_sequence": "peer parent chronology/scope differs",
                "root_budget": "peer changed the original root budget",
                "recipient": "peer role/purpose differs"}[change]
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, boundary):
                owner.validate_pc_divergence_head(record, head, label)

    def test_local_origin_cannot_be_relabelled_as_peer_by_changing_only_public_json(self):
        for change in ("kind_only", "peer_only", "complete_peer_with_original_local_seal"):
            manifest, record, head, label = fixture()
            witness = head["raw_receipt"]["witness"]
            source, origin = witness["source"], witness["source"]["accepted"]
            corpus._native(manifest, record)
            original_local_seal = origin["seal_sha256"]
            if change != "kind_only":
                origin["peer"] = peer_for(origin, source["root_budget_sha256"])
            if change != "peer_only":
                origin["kind"] = "peer_question"
            # Only the outer JSON is rehashed. An inserted reciprocal envelope
            # must not retain an original local accepted invocation seal.
            reseal_head(head, label)
            self.assertEqual(origin["seal_sha256"], original_local_seal)
            boundary = ("native binary invocation seal differs" if change == "complete_peer_with_original_local_seal"
                else "local source cannot be relabelled as an actual peer question")
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, boundary):
                owner.validate_pc_divergence_head(record, head, label)

    def test_cycle_sequence_is_not_the_collector_or_input_sequence(self):
        _, record, head, label = fixture(cycle_sequence=1, collector_sequence=110)
        self.assertLess(head["raw_receipt"]["witness"]["event_sequence"], record["input"]["capture_sequence"])
        owner.validate_pc_divergence_head(record, head, label)
        bad = copy.deepcopy(head)
        bad["raw_receipt"]["observed_sequence"] = 99
        reseal_head(bad, label)
        with self.assertRaisesRegex(ValueError, "collector chronology or raw seal"):
            owner.validate_pc_divergence_head(record, bad, label)

    def test_unresolved_cycle_zero_is_preserved_without_a_known_label(self):
        manifest, record, head, _ = fixture(resolution="unknown")
        witness = head["raw_receipt"]["witness"]
        witness.update(native_hash_kind="unresolved", event_sequence=0, reason="before first completed cycle",
            repair=None, recheck=None, repaired_endpoint=None, counter_endpoint=None)
        reseal_witness(witness)
        reseal_head(head, None)
        reseal_submitted(manifest, record)
        before = copy.deepcopy(record)
        corpus._native(manifest, record)
        observed = owner.validate_pc_divergence_head(record, head, None)
        self.assertIs(observed, witness)
        self.assertEqual(observed["event_sequence"], 0)
        self.assertIs(record["targets"]["divergence"], None)
        self.assertEqual(record, before)

    def test_unmapped_visible_known_raw_is_preserved_but_never_gets_a_known_target(self):
        manifest, record, head, label = fixture()
        snapshot = record["input"]
        snapshot["attack_mapping"][0].update(selected_counter_local_index=None, selected_counter_revision=None)
        record["targets"]["divergence"], record["divergence_index"] = None, None
        reseal_original_row(manifest, record, head, label)
        before, seal = copy.deepcopy(record), submitted.seal_input(snapshot)
        corpus._native(manifest, record)
        observed = owner.validate_pc_divergence_head(record, head, None)
        self.assertIs(observed["target_known"], True)
        self.assertIs(record["targets"]["divergence"], None)
        corpus._head_sources(manifest, record, {**record["targets"], "nnuec_wdl": None})
        self.assertEqual(record, before)
        self.assertEqual(submitted.seal_input(snapshot), seal)
        with self.assertRaisesRegex(ValueError, "target differs from the actual owner witness"):
            owner.validate_pc_divergence_head(record, head, label)

    def test_finite_total_cmp_selects_positive_zero_over_negative_zero_attack_slot(self):
        manifest, record, head, _ = fixture(resolution="unknown")
        snapshot, witness = record["input"], head["raw_receipt"]["witness"]
        snapshot["attack_mapping"].append({"slot": 1, "attack_ply": 3,
            "selected_counter_local_index": 0, "selected_counter_revision": 23})
        snapshot["encoded_model_input"]["divergences"].append([3 / 256.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        logits = [0x80000000, 0x00000000]
        record["native"]["raw_completion"]["output"]["payload"]["divergence_logits_bits"] = logits
        witness.update(native_hash_kind="unresolved", event_sequence=0, reason="selected finite total_cmp attack",
            repair=None, recheck=None, repaired_endpoint=None, counter_endpoint=None)
        witness["source"].update(divergence_index=1, divergence_ply=3)
        reseal_original_row(manifest, record, head, None)
        self.assertEqual(struct.unpack("<f", logits[0].to_bytes(4, "little"))[0],
            struct.unpack("<f", logits[1].to_bytes(4, "little"))[0])
        self.assertNotEqual(logits[0], logits[1])
        corpus._native(manifest, record)
        owner.validate_pc_divergence_head(record, head, None)
        witness["source"].update(divergence_index=0, divergence_ply=1)
        reseal_witness(witness)
        reseal_head(head, None)
        with self.assertRaisesRegex(ValueError, "not the original finite FP32 selected head"):
            owner.validate_pc_divergence_head(record, head, None)

    def test_rehashed_wrong_original_slot_ply_revision_or_counter_namespace_is_rejected(self):
        for field, value in (("divergence_index", 1), ("divergence_ply", 2), ("counter_revision", 24),
                             ("counter_local_index", 1), ("first_move", 128)):
            _, record, head, label = fixture()
            head["raw_receipt"]["witness"]["source"][field] = value
            reseal_witness(head["raw_receipt"]["witness"])
            reseal_head(head, label)
            boundary = "submitted attack mapping" if field in ("divergence_ply", "counter_revision") else "submitted question/attack/anchor"
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, boundary):
                owner.validate_pc_divergence_head(record, head, label)

    def test_rehashed_original_selected_metadata_cannot_erase_visible_counter_identity(self):
        for field, value in (("source_revision", 24), ("record_revision", 24), ("source_line_offset", 1),
                             ("original_first_move", 128)):
            manifest, record, head, label = fixture()
            record["native"]["raw_submission"]["submitted_review"]["record_metadata"][0][field] = value
            reseal_submitted(manifest, record)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "visible revision/Rules anchor"):
                owner.validate_pc_divergence_head(record, head, label)

    def test_rehashed_frozen_pair_model_precision_and_authority_mismatch_is_rejected(self):
        for field, value in (("public_context", h("different context")), ("model", h("different model")),
                             ("precision", "Fp16"), ("frozen_epoch", 2), ("record_revision", 26),
                             ("perspective", "black")):
            _, record, head, label = fixture()
            endpoint = head["raw_receipt"]["witness"]["counter_endpoint"]
            endpoint["pair"][field] = value
            reseal_witness(head["raw_receipt"]["witness"])
            reseal_head(head, label)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "frozen Fresh pair differs|chronology or frozen pair"):
                owner.validate_pc_divergence_head(record, head, label)

    def test_nested_local_accepted_ticket_requires_an_integer_length_bucket(self):
        for name in ("repair", "recheck", "repaired_endpoint", "counter_endpoint"):
            _, record, head, label = fixture()
            ticket = head["raw_receipt"]["witness"][name]["accepted"]["ticket"]
            ticket["length_bucket"] = float(ticket["length_bucket"])
            # The native origin seal intentionally does not contain this view
            # allocation field, so the exact integer guard must reject it.
            reseal_head(head, label)
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "bounded unsigned integer"):
                owner.validate_pc_divergence_head(record, head, label)

    def test_rehashed_missing_physical_fresh_completion_never_becomes_known(self):
        for endpoint in ("repaired_endpoint", "counter_endpoint"):
            for field, value in (("measurement_complete", False), ("physical_fresh", False),
                                 ("completed_nn_inputs", 0), ("execution_epoch", 2)):
                _, record, head, label = fixture()
                head["raw_receipt"]["witness"][endpoint]["physical"][field] = value
                reseal_witness(head["raw_receipt"]["witness"])
                reseal_head(head, label)
                boundary = "positive original P/C witness identity" if field == "completed_nn_inputs" else "original confirmed physical observation"
                with self.subTest(endpoint=endpoint, field=field), self.assertRaisesRegex(ValueError, boundary):
                    owner.validate_pc_divergence_head(record, head, label)

    def test_rehashed_repair_c_reply_or_request_order_cannot_invent_the_original_cycle(self):
        for change in ("repair_prefix", "repair_parity", "c_reply_parity", "fresh_before_c", "duplicate_fresh", "foreign_owner"):
            _, record, head, label = fixture()
            witness = head["raw_receipt"]["witness"]
            if change == "repair_prefix":
                witness["repair"]["moves"][1] = 768
                witness["repaired_endpoint"]["line"] = copy.deepcopy(witness["repair"]["moves"])
            elif change == "repair_parity":
                witness["repair"]["moves"] = [64, 128, 384, 768, 512]
                witness["repaired_endpoint"]["line"] = copy.deepcopy(witness["repair"]["moves"])
            elif change == "c_reply_parity":
                witness["recheck"]["moves"] = [64, 128, 768, 448, 512]
                witness["counter_endpoint"]["line"] = copy.deepcopy(witness["recheck"]["moves"])
            elif change == "fresh_before_c":
                witness["repaired_endpoint"]["accepted"]["ticket"]["request_sequence"] = 2
            elif change == "duplicate_fresh":
                witness["counter_endpoint"]["accepted"]["ticket"]["request_sequence"] = 4
            else:
                witness["repair"]["accepted"]["ticket"]["root_generation"] = 2
                witness["repair"]["accepted"]["frame"]["root_generation"] = 2
            reseal_witness(witness)
            reseal_head(head, label)
            boundary = {"repair_prefix": "fixed original first move/attack prefix",
                "repair_parity": "actual P repair followed by an independent C reply",
                "c_reply_parity": "actual P repair followed by an independent C reply",
                "foreign_owner": "changed original owner scope/model/private role"}.get(change,
                    "Repair/C/Fresh chronology or frozen pair")
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, boundary):
                owner.validate_pc_divergence_head(record, head, label)

    def test_root_wdl_is_the_separate_perspective_flip_of_the_original_raw_endpoint(self):
        _, record, head, label = fixture()
        endpoint = head["raw_receipt"]["witness"]["counter_endpoint"]
        self.assertEqual(endpoint["root_wdl_bits"], endpoint["raw_wdl_bits"][::-1])
        owner.validate_pc_divergence_head(record, head, label)
        endpoint["root_wdl_bits"] = copy.deepcopy(endpoint["raw_wdl_bits"])
        reseal_witness(head["raw_receipt"]["witness"])
        reseal_head(head, label)
        with self.assertRaisesRegex(ValueError, "exact root perspective"):
            owner.validate_pc_divergence_head(record, head, label)

    def test_float32_subtraction_tie_is_unknown_even_when_float64_difference_is_not_equal(self):
        _, record, head, _ = fixture(resolution="unknown")
        witness = head["raw_receipt"]["witness"]
        for endpoint, probabilities in zip((witness["repaired_endpoint"], witness["counter_endpoint"]),
                ([0.75, 0.25, 2**-30], [0.75, 0.25, 0.0])):
            endpoint["root_wdl_bits"] = f32_bits(probabilities)
            endpoint["raw_wdl_bits"] = f32_bits(probabilities[::-1])
        left, right = witness["repaired_endpoint"]["root_wdl_bits"], witness["counter_endpoint"]["root_wdl_bits"]
        self.assertEqual(root_expectation(left), root_expectation(right))
        self.assertNotEqual(0.75 - 2**-30, 0.75)
        reseal_witness(witness)
        reseal_head(head, None)
        owner.validate_pc_divergence_head(record, head, None)

    def test_equal_fp32_expectations_and_partial_original_witness_remain_unlabelled(self):
        manifest, record, head, label = fixture(resolution="unknown")
        self.assertIsNone(label)
        witness = head["raw_receipt"]["witness"]
        self.assertEqual(root_expectation(witness["repaired_endpoint"]["root_wdl_bits"]),
            root_expectation(witness["counter_endpoint"]["root_wdl_bits"]))
        owner.validate_pc_divergence_head(record, head, None)
        witness.update(native_hash_kind="unresolved", reason="actual partial repair", recheck=None,
            repaired_endpoint=None, counter_endpoint=None)
        reseal_witness(witness)
        reseal_head(head, None)
        before = copy.deepcopy(record)
        owner.validate_pc_divergence_head(record, head, None)
        corpus._head_sources(manifest, record, {**record["targets"], "nnuec_wdl": None})
        self.assertEqual(record, before)

    def test_unknown_or_missing_endpoint_cannot_gain_a_known_label_by_rehashing_json(self):
        _, _known_record, _known_head, forged_label = fixture()
        for partial in (False, True):
            _, record, head, _ = fixture(resolution="unknown")
            witness = head["raw_receipt"]["witness"]
            if partial:
                witness.update(native_hash_kind="unresolved", reason="unfinished original endpoint",
                    counter_endpoint=None)
                reseal_witness(witness)
            label = copy.deepcopy(forged_label)
            reseal_head(head, label)
            with self.subTest(partial=partial), self.assertRaisesRegex(ValueError, "target differs from the actual owner witness"):
                owner.validate_pc_divergence_head(record, head, label)

    def test_json_cannot_add_current_authority_or_replace_conditional_wdl_by_cp(self):
        for target in ("raw", "witness", "endpoint"):
            _, record, head, label = fixture()
            raw = head["raw_receipt"]
            value = raw if target == "raw" else raw["witness"] if target == "witness" else raw["witness"]["repaired_endpoint"]
            value["execution_authorized"] = True
            reseal_head(head, label)
            with self.subTest(target=target), self.assertRaisesRegex(ValueError, "fields mismatch"):
                owner.validate_pc_divergence_head(record, head, label)
        _, record, head, label = fixture()
        head["raw_receipt"]["witness"]["repaired_endpoint"]["raw_cp"] = 500
        reseal_head(head, label)
        with self.assertRaisesRegex(ValueError, "fields mismatch"):
            owner.validate_pc_divergence_head(record, head, label)


if __name__ == "__main__":
    unittest.main()
