"""Synthetic initial-attack/future-counter joins, never an accepted owner.

Both small native cells and the binary source seals are explicit fixture bytes.
No fixture is a real collection, physical run, Rules proof or training target
quota. In particular a future counter never becomes part of the initial input.
"""
import copy
import importlib.abc
import json
import sys
import unittest


class NeuralImportsForbidden(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"torch", "numpy", "onnx", "onnxruntime"}:
            raise AssertionError("attack-source joins cannot import neural runtimes")


_guard = NeuralImportsForbidden()
sys.meta_path.insert(0, _guard)
try:
    from rz_pals_model import generation_data as data
    from rz_pals_model import rovent_corpus as corpus
    from rz_pals_model import rovent_divergence_owner as owner
    from rz_pals_model import rovent_submitted_input as submitted
    from rz_pals_model.virtual_review import input_digest
    from test_rovent_divergence_owner import fixture as visible_fixture
    from test_rovent_divergence_owner import (h, binary_seal, u64, sha_bytes, line_bytes,
        origin_for, seal_origin, reseal_witness, reseal_head)
    from test_rovent_historical_corpus import reseal_submitted
    from test_rovent_submitted_input import components
finally:
    sys.meta_path.remove(_guard)


ATTACK_DOMAIN = "rz-rovent-pc-attack-ranked-source/1"
WRAPPER_DOMAIN = "rz-rovent-pc-divergence-source-with-attack/1"


def seal_cell(manifest, record):
    """Reseal the original synthetic critic cell without a witness projection."""
    snapshot, native = record["input"], record["native"]
    snapshot["view_sha256"] = data.view_digest(snapshot["selected_view"])
    record["input_sha256"] = submitted.seal_input(snapshot)
    native["input_digest"] = input_digest(snapshot["encoded_model_input"])
    ticket = native["raw_submission"]["ticket"]
    ticket["representation"]["input"] = native["input_digest"]
    output = native["raw_completion"]["output"]
    output["representation"] = copy.deepcopy(ticket["representation"])
    bits = output["payload"]
    framed = (b"rz-rovent-accepted-critic-head/1\0" + u64(len(snapshot["legal_moves"]))
        + u64(len(bits["divergence_logits_bits"])) + b"".join(value.to_bytes(4, "little") for value in
            (bits["candidate_logits_bits"] + bits["divergence_logits_bits"] + bits["wdl_bits"])))
    import hashlib
    native["output_digest"] = hashlib.sha256(framed).hexdigest()
    for name in ("raw_submission", "raw_completion", "raw_retirement"):
        native[name].update(input_sha256=record["input_sha256"], ticket=copy.deepcopy(ticket))
        if name != "raw_submission":
            native[name]["output_digest"] = native["output_digest"]
    native["raw_submission"]["input_digest"] = native["input_digest"]
    reseal_submitted(manifest, record)


def reseal_attack_source(witness):
    """Independent Rust binary framing; no validator private helper is used."""
    wrapper = witness["source"]
    # Existing fixture code seals only the old visible-source leaves. Its
    # temporary witness digest is discarded, never retagged as the new seal.
    visible_only = {**witness, "source": wrapper["visible_source"]}
    reseal_witness(visible_only)
    attack = wrapper["attack_ranked_source"]
    seal_origin(attack["accepted"])
    attack["native_attack_sha256"] = binary_seal(ATTACK_DOMAIN,
        sha_bytes(attack["accepted"]["seal_sha256"]) + sha_bytes(attack["root_budget_sha256"])
        + attack["divergence_index"].to_bytes(2, "little") + u64(attack["divergence_ply"])
        + attack["first_move"].to_bytes(2, "little") + line_bytes(attack["proposal"]))
    wrapper["native_source_sha256"] = binary_seal(WRAPPER_DOMAIN,
        sha_bytes(wrapper["visible_source"]["native_source_sha256"]) + sha_bytes(attack["native_attack_sha256"])
        + u64(wrapper["visible_source_sequence"]) + u64(attack["source_sequence"]))
    framed = sha_bytes(wrapper["native_source_sha256"])
    facts = (("repair", "native_line_sha256"), ("recheck", "native_line_sha256"),
        ("repaired_endpoint", "native_endpoint_sha256"), ("counter_endpoint", "native_endpoint_sha256"))
    if witness["native_hash_kind"] == "frozen_pair":
        framed += b"".join(sha_bytes(witness[name][key]) for name, key in facts)
        framed += u64(witness["event_sequence"]) + bytes([
            {"supported": 0, "refuted": 1, "unknown": 2, "disputed": 3}[witness["resolution"]]])
        domain = "rz-rovent-pc-divergence-witness/1"
    else:
        for name, key in facts:
            value = witness[name]
            framed += b"\0" if value is None else b"\1" + sha_bytes(value[key])
        reason = witness["reason"].encode("utf-8")
        framed += u64(witness["event_sequence"]) + u64(len(reason)) + reason
        domain = "rz-rovent-pc-divergence-unresolved/1"
    witness["native_witness_sha256"] = binary_seal(domain, framed)


def future_evidence(record):
    return {"input_json": json.dumps(record["input"], sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False), "input_sha256": record["input_sha256"],
        "native": copy.deepcopy(record["native"])}


def sync_attack(manifest, record, head, label):
    seal_cell(manifest, record)
    attack = head["raw_receipt"]["witness"]["source"]["attack_ranked_source"]
    attack["accepted"]["ticket"] = copy.deepcopy(record["native"]["raw_submission"]["ticket"])
    attack["accepted"]["output_digest"] = record["native"]["output_digest"]
    head["raw_receipt"]["input_sha256"] = record["input_sha256"]
    reseal_attack_source(head["raw_receipt"]["witness"])
    reseal_head(head, label)
    reseal_submitted(manifest, record)


def fixture():
    manifest, future, future_head, future_label = visible_fixture()
    witness = future_head["raw_receipt"]["witness"]
    source = witness["source"]
    proposal, counter = [64, 128, 192, 256, 320, 384], [64, 128, 192, 448, 512, 576]
    repair, recheck = [64, 128, 192, 448, 640, 576], [64, 128, 192, 448, 640, 704]
    view, encoded = future["input"]["selected_view"], future["input"]["encoded_model_input"]
    view["query"].update(proposal=proposal, counter=counter)
    view["records"][0]["moves"] = counter
    encoded["query"].update(proposal=[components(m) for m in proposal], counter=[components(m) for m in counter])
    encoded["records"][0]["line"] = [components(m) for m in counter]
    future["input"]["attack_mapping"][0]["attack_ply"] = 3
    encoded["divergences"][0][0] = 3 / 256.0
    native = future["native"]
    native.update(request_sequence=11, execution_sequence=11)
    native["raw_submission"]["ticket"].update(request_sequence=11, actual_tokens=18)
    for name in ("raw_submission", "raw_completion", "raw_retirement"):
        native[name]["request_sequence"] = 11
    native["raw_completion"]["execution_sequence"] = 11
    native["raw_completion"]["output"]["request_sequence"] = 11
    source.update(divergence_ply=3, proposal=proposal, counter=counter)
    for name in ("repair", "recheck", "repaired_endpoint", "counter_endpoint"):
        witness[name]["accepted"]["ticket"]["request_sequence"] += 10
    witness["repair"]["moves"], witness["recheck"]["moves"] = repair, recheck
    for name, moves in (("repaired_endpoint", repair), ("counter_endpoint", recheck)):
        endpoint = witness[name]
        endpoint["line"] = moves
        endpoint["physical"]["execution_sequence"] += 10
        endpoint["raw_wdl_bits"] = copy.deepcopy(endpoint["root_wdl_bits"])
    seal_cell(manifest, future)
    source["accepted"]["ticket"] = copy.deepcopy(native["raw_submission"]["ticket"])
    source["accepted"]["output_digest"] = native["output_digest"]
    reseal_witness(witness)

    record, head, label = copy.deepcopy(future), copy.deepcopy(future_head), copy.deepcopy(future_label)
    snapshot, native = record["input"], record["native"]
    snapshot.update(capture_sequence=10, question_id=h("initial pre-counter attack question"))
    view, encoded = snapshot["selected_view"], snapshot["encoded_model_input"]
    view["records"], encoded["records"] = [], []
    view["query"].update(counter_present=False, counter=[])
    encoded["query"].update(counter_present=False, counter=[])
    snapshot["attack_mapping"] = [{"slot": slot, "attack_ply": ply,
        "selected_counter_local_index": None, "selected_counter_revision": None} for slot, ply in enumerate((1, 3))]
    encoded["divergences"] = [[ply / 256.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] for ply in (1, 3)]
    native.update(request_sequence=1, execution_sequence=1, submitted_sequence=10, completed_sequence=11, retired_sequence=12,
        view_digest=h("initial native view without future counter"))
    native["raw_submission"]["ticket"].update(request_sequence=1, question_digest=snapshot["question_id"],
        actual_tokens=6, selected_records=0, view_digest=native["view_digest"])
    native["raw_submission"]["ticket"]["representation"]["record_revision"] = 0
    for name, sequence in (("raw_submission", 10), ("raw_completion", 11), ("raw_retirement", 12)):
        native[name].update(request_sequence=1, sequence=sequence)
    native["raw_submission"]["view_digest"] = native["view_digest"]
    native["raw_submission"]["submitted_review"].update(source_anchors=[], record_metadata=[], view_seal_sha256=native["view_digest"])
    native["raw_completion"]["execution_sequence"] = 1
    native["raw_completion"]["output"]["request_sequence"] = 1
    native["raw_completion"]["output"]["payload"]["divergence_logits_bits"] = [0, 0x3f800000]
    seal_cell(manifest, record)
    attack = {"schema": ATTACK_DOMAIN, "native_attack_sha256": "",
        "accepted": origin_for(native["raw_submission"]["ticket"], frame=11, output=native["output_digest"]),
        "source_sequence": 5,
        "root_budget_sha256": source["root_budget_sha256"], "divergence_index": 1, "divergence_ply": 3,
        "first_move": 64, "proposal": copy.deepcopy(proposal), "perspective": "white"}
    head["raw_receipt"]["witness"]["source"] = {"schema": WRAPPER_DOMAIN, "native_source_sha256": "",
        "visible_source": copy.deepcopy(source), "visible_source_sequence": 9,
        "attack_ranked_source": attack, "visible_record_evidence": future_evidence(future)}
    head["raw_receipt"]["input_sha256"] = record["input_sha256"]
    label.update(question_id=snapshot["question_id"], view_sha256=snapshot["view_sha256"], slot=1, attack_ply=3)
    reseal_attack_source(head["raw_receipt"]["witness"])
    reseal_head(head, label)
    record["head_sources"]["divergence"] = head
    record["targets"]["divergence"], record["divergence_index"] = label, 1
    reseal_submitted(manifest, record)
    return manifest, record, head, label, future


class InitialAttackSourceTests(unittest.TestCase):
    def setUp(self):
        sys.meta_path.insert(0, _guard)

    def tearDown(self):
        sys.meta_path.remove(_guard)

    def test_original_attack_slot_and_future_visible_slot_are_independent_for_same_ply(self):
        manifest, record, head, label, future = fixture()
        before = copy.deepcopy(record)
        wrapper = head["raw_receipt"]["witness"]["source"]
        corpus._native(manifest, record)
        corpus._native(manifest, future)
        self.assertEqual(wrapper["attack_ranked_source"]["divergence_index"], 1)
        self.assertEqual(wrapper["visible_source"]["divergence_index"], 0)
        self.assertEqual(label["slot"], 1)
        visible = wrapper["visible_source"]
        # Actual CounterStart preserves proposal[..attack] and excludes its
        # move at attack. The synthetic chain must follow that same prefix.
        self.assertEqual(visible["proposal"][:3], visible["counter"][:3])
        self.assertNotEqual(visible["proposal"][3], visible["counter"][3])
        witness = head["raw_receipt"]["witness"]
        self.assertEqual(witness["repair"]["moves"][:4], visible["counter"][:4])
        self.assertNotEqual(witness["repair"]["moves"][4], visible["counter"][4])
        self.assertEqual(witness["recheck"]["moves"][:5], witness["repair"]["moves"][:5])
        self.assertNotEqual(witness["recheck"]["moves"][5], witness["repair"]["moves"][5])
        self.assertEqual(wrapper["attack_ranked_source"]["accepted"]["frame"]["sequence"],
            wrapper["visible_source"]["accepted"]["frame"]["sequence"])
        self.assertEqual(wrapper["attack_ranked_source"]["source_sequence"], 5)
        self.assertEqual(wrapper["visible_source_sequence"], 9)
        self.assertEqual(wrapper["attack_ranked_source"]["accepted"]["ticket"]["request_sequence"], 1)
        self.assertEqual(wrapper["visible_source"]["accepted"]["ticket"]["request_sequence"], 11)
        self.assertEqual(record["input"]["selected_view"]["records"], [])
        self.assertIs(record["input"]["selected_view"]["query"]["counter_present"], False)
        self.assertEqual(record["input"]["selected_view"]["query"]["counter"], [])
        owner.validate_pc_divergence_head(record, head, label, manifest=manifest)
        corpus._head_sources(manifest, record, {**record["targets"], "nnuec_wdl": None})
        self.assertEqual(record, before)
        evidence = wrapper["visible_record_evidence"]
        self.assertEqual(set(evidence), {"input_json", "input_sha256", "native"})
        original_text = evidence["input_json"]
        self.assertIs(type(original_text), str)
        self.assertEqual(submitted.seal_input(json.loads(original_text)), evidence["input_sha256"])
        self.assertEqual(evidence["input_json"], before["head_sources"]["divergence"]["raw_receipt"]
            ["witness"]["source"]["visible_record_evidence"]["input_json"])
        alternate_text = original_text.replace("0.01171875", "1.171875e-2", 1)
        self.assertNotEqual(alternate_text, original_text)
        self.assertEqual(submitted.seal_input(json.loads(alternate_text)), evidence["input_sha256"])
        alternate = copy.deepcopy(record)
        alternate_head, alternate_label = alternate["head_sources"]["divergence"], alternate["targets"]["divergence"]
        alternate_wrapper = alternate_head["raw_receipt"]["witness"]["source"]
        alternate_wrapper["visible_record_evidence"]["input_json"] = alternate_text
        # The exact UTF-8 JSON spelling belongs to the outer HEAD seal. Its
        # parsed FP32 input seal and original native binary source seal do not
        # change, and the reader must not normalize the stored source string.
        reseal_head(alternate_head, alternate_label)
        reseal_submitted(manifest, alternate)
        self.assertNotEqual(alternate_head["raw_receipt_sha256"], head["raw_receipt_sha256"])
        self.assertEqual(alternate_wrapper["native_source_sha256"], wrapper["native_source_sha256"])
        owner.validate_pc_divergence_head(alternate, alternate_head, alternate_label, manifest=manifest)
        self.assertEqual(alternate_wrapper["visible_record_evidence"]["input_json"], alternate_text)

    def test_cpu_source_order_is_separate_from_equal_frame_and_distinct_request_sequences(self):
        for change in ("initial_zero", "visible_zero", "equal_source", "reverse_source"):
            manifest, record, head, label, _future = fixture()
            witness = head["raw_receipt"]["witness"]
            wrapper, attack = witness["source"], witness["source"]["attack_ranked_source"]
            native_source_pin = wrapper["native_source_sha256"]
            native_attack_pin = attack["native_attack_sha256"]
            if change == "initial_zero":
                attack["source_sequence"] = 0
            elif change == "visible_zero":
                wrapper["visible_source_sequence"] = 0
            elif change == "equal_source":
                attack["source_sequence"] = wrapper["visible_source_sequence"]
            else:
                attack["source_sequence"] = wrapper["visible_source_sequence"] + 1
            # The composite source seals visible then initial CPU source
            # sequences in Rust's original little-endian order. The attack
            # leaf seal itself excludes them. Neither sequence is another ID.
            reseal_attack_source(witness)
            self.assertNotEqual(wrapper["native_source_sha256"], native_source_pin)
            self.assertEqual(attack["native_attack_sha256"], native_attack_pin)
            reseal_head(head, label)
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, "positive|source.*sequence|chronology|scope"):
                owner.validate_pc_divergence_head(record, head, label, manifest=manifest)

    def test_manifest_and_future_native_evidence_remain_required_even_without_label(self):
        manifest, record, head, _label, _future = fixture()
        with self.assertRaisesRegex(ValueError, "explicit corpus and future invocation evidence"):
            owner.validate_pc_divergence_head(record, head, None)
        for change in ("missing", "input_object", "native_none", "missing_raw", "mixed_raw", "input_seal"):
            bad = copy.deepcopy(head)
            wrapper = bad["raw_receipt"]["witness"]["source"]
            if change == "missing":
                wrapper["visible_record_evidence"] = None
            elif change == "input_object":
                evidence = wrapper["visible_record_evidence"]
                evidence["input"] = json.loads(evidence.pop("input_json"))
            elif change == "native_none":
                wrapper["visible_record_evidence"]["native"] = None
            elif change == "missing_raw":
                del wrapper["visible_record_evidence"]["native"]["raw_completion"]
            elif change == "mixed_raw":
                wrapper["visible_record_evidence"]["native"]["raw_completion"]["schema"] = "rz-rovent-native-owner-event/2"
            else:
                wrapper["visible_record_evidence"]["input_sha256"] = h("different future cell seal")
            reseal_attack_source(bad["raw_receipt"]["witness"])
            reseal_head(bad, None)
            boundary = {"missing": "future invocation evidence fields mismatch", "input_object": "future invocation evidence fields mismatch",
                "native_none": "future invocation requires its original native lifecycle object",
                "missing_raw": "future invocation requires its original native lifecycle object",
                "mixed_raw": "mixed raw versions rejected",
                "input_seal": "future evidence changed its original sealed input"}[change]
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, boundary):
                owner.validate_pc_divergence_head(record, bad, None, manifest=manifest)

    def test_known_some_raw_without_label_is_audited_and_does_not_create_a_target(self):
        manifest, record, head, _label, future = fixture()
        record["targets"]["divergence"], record["divergence_index"] = None, None
        reseal_submitted(manifest, record)
        before = copy.deepcopy(record)
        corpus._native(manifest, record)
        corpus._native(manifest, future)
        observed = owner.validate_pc_divergence_head(record, head, None, manifest=manifest)
        self.assertIs(observed["target_known"], True)
        self.assertIs(record["targets"]["divergence"], None)
        corpus._head_sources(manifest, record, {**record["targets"], "nnuec_wdl": None})
        self.assertEqual(record, before)
        attack = head["raw_receipt"]["witness"]["source"]["attack_ranked_source"]
        attack["accepted"]["output_digest"] = h("another original output")
        reseal_attack_source(head["raw_receipt"]["witness"])
        reseal_head(head, None)
        with self.assertRaisesRegex(ValueError, "initial attack changed its original input"):
            owner.validate_pc_divergence_head(record, head, None, manifest=manifest)

    def test_collector_observation_cannot_precede_the_future_native_retirement(self):
        manifest, record, head, _label, future = fixture()
        self.assertEqual(future["native"]["retired_sequence"], 102)
        head["observed_sequence"] = head["raw_receipt"]["observed_sequence"] = 101
        reseal_head(head, None)
        with self.assertRaisesRegex(ValueError, "collector receipt precedes its future visible retirement"):
            owner.validate_pc_divergence_head(record, head, None, manifest=manifest)

    def test_initial_future_mapping_insertion_and_some_to_legacy_retag_are_rejected(self):
        manifest, record, head, label, _future = fixture()
        record["input"]["attack_mapping"][1].update(selected_counter_local_index=0, selected_counter_revision=23)
        # Index 0 is not present in the original empty selected view. Do not
        # manufacture a valid input seal or append the future counter to it.
        with self.assertRaisesRegex(ValueError, "counter must already exist"):
            owner.validate_pc_divergence_head(record, head, label, manifest=manifest)
        manifest, record, head, label, _future = fixture()
        head["raw_receipt"]["witness"]["source"] = copy.deepcopy(head["raw_receipt"]["witness"]["source"]["visible_source"])
        reseal_witness(head["raw_receipt"]["witness"])
        reseal_head(head, label)
        with self.assertRaisesRegex(ValueError, "original retired critic|original owner|source"):
            owner.validate_pc_divergence_head(record, head, label, manifest=manifest)

    def test_rehashed_chronology_argmax_proposal_and_scope_mismatch_are_not_targets(self):
        for change in ("chronology", "capture_chronology", "argmax", "proposal", "counter_prefix", "root_budget", "perspective", "model", "deadline"):
            manifest, record, head, label, future = fixture()
            witness = head["raw_receipt"]["witness"]
            wrapper, attack = witness["source"], witness["source"]["attack_ranked_source"]
            if change == "chronology":
                native = record["native"]
                native.update(request_sequence=11, execution_sequence=11)
                native["raw_submission"]["ticket"]["request_sequence"] = 11
                for name in ("raw_submission", "raw_completion", "raw_retirement"):
                    native[name]["request_sequence"] = 11
                native["raw_completion"]["execution_sequence"] = 11
                native["raw_completion"]["output"]["request_sequence"] = 11
                sync_attack(manifest, record, head, label)
                corpus._native(manifest, record)
            elif change == "capture_chronology":
                record["input"]["capture_sequence"] = 100
                native = record["native"]
                native.update(submitted_sequence=100, completed_sequence=101, retired_sequence=102)
                for name, sequence in (("raw_submission", 100), ("raw_completion", 101), ("raw_retirement", 102)):
                    native[name]["sequence"] = sequence
                sync_attack(manifest, record, head, label)
                corpus._native(manifest, record)
            elif change == "argmax":
                record["native"]["raw_completion"]["output"]["payload"]["divergence_logits_bits"] = [0x40000000, 0]
                sync_attack(manifest, record, head, label)
            elif change == "proposal":
                attack["proposal"][2] = 768
            elif change == "counter_prefix":
                snapshot = future["input"]
                view, encoded = snapshot["selected_view"], snapshot["encoded_model_input"]
                view["query"]["counter"][2] = view["records"][0]["moves"][2] = 384
                encoded["query"]["counter"][2] = components(384)
                encoded["records"][0]["line"][2] = components(384)
                seal_cell(manifest, future)
                visible = wrapper["visible_source"]
                visible["counter"][2] = 384
                visible["accepted"]["ticket"] = copy.deepcopy(future["native"]["raw_submission"]["ticket"])
                visible["accepted"]["output_digest"] = future["native"]["output_digest"]
                wrapper["visible_record_evidence"] = future_evidence(future)
                # Matching query, selected record and numeric payload still
                # pass the independent full future native boundary. Only the
                # Some attack's preserved proposal prefix is contradicted.
                corpus._native(manifest, future)
            elif change == "root_budget":
                attack["root_budget_sha256"] = h("different root budget")
            elif change == "perspective":
                attack["perspective"] = "black"
            elif change == "model":
                attack["accepted"]["ticket"]["representation"]["model"] = h("different frozen model")
            else:
                attack["accepted"]["ticket"]["deadline_tick"] += 1
            reseal_attack_source(witness)
            if change == "counter_prefix":
                self.assertEqual(owner._source(future, wrapper["visible_source"]),
                    future["native"]["raw_submission"]["ticket"])
            reseal_head(head, label)
            boundary = ("future visible evidence crossed its original root/source/input chronology" if change == "capture_chronology"
                else "not the original finite FP32 selected head" if change == "argmax"
                else "changed original owner scope/model/private role" if change in ("model", "deadline")
                else "initial attack changed its original input or future visible chronology/scope")
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, boundary):
                owner.validate_pc_divergence_head(record, head, label, manifest=manifest)

    def test_native_source_seal_is_not_repaired_by_rehashing_only_the_outer_receipt(self):
        manifest, record, head, label, _future = fixture()
        head["raw_receipt"]["witness"]["source"]["attack_ranked_source"]["native_attack_sha256"] = h("different native attack")
        reseal_head(head, label)
        with self.assertRaisesRegex(ValueError, "binary invocation seal|attack.*seal"):
            owner.validate_pc_divergence_head(record, head, label, manifest=manifest)


if __name__ == "__main__":
    unittest.main()
