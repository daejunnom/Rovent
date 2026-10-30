"""Synthetic /2 wire operands only; never a live trial or learner proof."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from rz_pals_model import rovent_utility_intervention as metadata
from rz_pals_model import rovent_utility_intervention_files as files_module
from rz_pals_model import rovent_utility_intervention_payload as reader
from rz_pals_model import rovent_utility_intervention_preparation as preparation
from rz_pals_model import nnuec_reference as numeric
from rz_pals_model import rovent_utility_cohort_payload as shared
from test_rovent_utility_cohort_payload import (full_cohort_case, h, native_bytes_sha,
    framed_value, raw, seal_fixture_submitted_view)
from test_rovent_utility_intervention_files import pin, budget


def neutral_from_input(inp):
    """New synthetic /2 record using shared unchanged input/Rules operands."""
    return {"domain": metadata.TARGET_DOMAIN, "input_sha256": inp["input_sha256"],
        "target_sha256": inp["target_sha256"], "context_sha256": inp["context_sha256"],
        "target_record": inp["target_record"], "target_record_ids": copy.deepcopy(inp["target_record_ids"]),
        "input_bytes": inp["bytes"], "reference_json": inp["reference_json"],
        "root_snapshot_sha256": independent_snapshot(inp["root_rules"]),
        "source_snapshot_sha256": [independent_snapshot(rules) for rules in inp["source_rules"]],
        "record_scope_sha256": inp["provenance"]["record_scope_sha256"], "model_identity": copy.deepcopy(inp["base_identity"]),
        "original_first_move": inp["records_metadata"][inp["target_record"]]["original_first_move"],
        "root_rules": copy.deepcopy(inp["root_rules"]), "records_metadata": copy.deepcopy(inp["records_metadata"]),
        "source_rules": copy.deepcopy(inp["source_rules"]), "original_view_sha256": inp["provenance"]["view_seal_sha256"],
        "original_full_view_sha256": inp["provenance"]["full_view_sha256"]}


def independent_snapshot(rules):
    replay = rules["uci_replay"]
    start, moves = replay["start_fen"].encode(), replay["moves"]
    return hashlib.sha256(b"rz-pals-exact-review-anchor/1\0" + bytes((
        {"start_position": 0, "fen": 1}[replay["origin"]], {"complete": 0, "unknown_prefix": 1}[replay["completeness"]]))
        + len(start).to_bytes(8, "little") + start + len(moves).to_bytes(2, "little")
        + b"".join(item.to_bytes(2, "little") for item in moves) + rules["fen"].encode()).hexdigest()


def independent_question(value):
    data = (b"rz-pals-adaptive-checked-question/1\0" + value["sequence"].to_bytes(8, "little")
        + bytes([value["stage_code"]]) + b"".join(bytes.fromhex(value[name]) for name in (
            "view_digest", "view_seal_digest", "focus_prefix_sha256")))
    if value["goal_local_sha256"] is not None:
        data += b"goal-local-rules-origin\0" + bytes.fromhex(value["goal_local_sha256"])
    if value["legacy_full_records_sha256"] is not None:
        data += b"legacy-full-raw-snapshot\0" + bytes.fromhex(value["legacy_full_records_sha256"])
    return hashlib.sha256(data).hexdigest()


def intervention_case(*, change_arm=None):
    old_prepared, _, old_blobs, _, _, old = full_cohort_case()
    profile = {"domain": metadata.PROFILE_DOMAIN, "input": copy.deepcopy(old["profile"]["input"]),
        "target_record": 0, "policy": metadata.POLICY}
    profile_sha = metadata.intervention_profile_sha256(profile)
    source = {**copy.deepcopy(old["source"]), "schema": reader.SOURCE_DOMAIN}
    cohort = framed_value("rz-rovent-utility-intervention-owner/2", {"profile": profile_sha,
        "original_root": source["registration_root"]["root_descriptor_sha256"], "source": source["source_pins"]})
    blobs = {name: old_blobs[name] for name in ("input-lock.json", "nnuec-weights.json", "nnuec-provenance.json")}
    blobs.update({"cohort-profile.json": raw(profile), "source.json": raw(source)})
    trials = []
    for index, original in enumerate(old["trials"]):
        original_body = json.loads(original["materialized_json"])
        neutral = neutral_from_input(original_body["input"])
        neutral_sha = reader.neutral_target_sha256(neutral)
        if index == 0:
            blobs["neutral-target.json"] = raw(neutral)
        prepared = copy.deepcopy(original_body["input"])
        prepared["provenance"]["sequence"] += 100
        prepared["provenance"]["full_view_sha256"] = h("new synthetic actual arm " + str(index))
        seal_fixture_submitted_view(prepared)
        prepared["submission_target_sha256"] = hashlib.sha256(b"rz-nnuec-submitted-utility-submission-target/1\0"
            + bytes.fromhex(prepared["provenance"]["view_seal_sha256"]) + prepared["target_sha256"].encode()).hexdigest()
        original_offer = original_body["prelude"]["offers"][0]
        view = copy.deepcopy(original_offer["view"])
        view.update(sequence=prepared["provenance"]["sequence"], view_digest=prepared["provenance"]["view_seal_sha256"])
        preimage = {"sequence": 101 + index, "stage_code": (2, 5)[index],
            "view_digest": prepared["provenance"]["full_view_sha256"], "view_seal_digest": view["view_digest"],
            "focus_prefix_sha256": h("synthetic private focus"), "goal_local_sha256": None, "legacy_full_records_sha256": None}
        query = {"endpoint_rules": copy.deepcopy(neutral["root_rules"]), "prefix": [] if index else [64], "proposal": [64],
            "counterexample": None, "legal": [] if index else [64], "divergences": [0] if index else [],
            "question_preimage": preimage}
        arm = {"neutral_target_sha256": neutral_sha, "arm_query_sha256": independent_question(preimage),
            "recipient": ("route_p", "route_c")[index], "frame_sequence": 71 + index, "source_frame_sequence": 10,
            "stage": ("ProposalContinuation", "CriticDivergence")[index], "purpose": ("ProposePolicy", "DivergencePolicy")[index],
            "original_first_move": 64, "view": view, "full_view_sha256": prepared["provenance"]["full_view_sha256"],
            "public_context_sha256": neutral["record_scope_sha256"], "budget": copy.deepcopy(original_offer["budget"]),
            "deadline_clock": original_offer["deadline_clock"], "deadline_tick": original_offer["deadline_tick"],
            "query": query, "prepared_input": prepared}
        cost = copy.deepcopy(original["cost"])
        cost["cost_epoch_sha256"] = h("actual independent trial cost epoch " + str(index))
        action = {**copy.deepcopy(original_body["action"]), "frame_sequence": arm["frame_sequence"],
            "question_sha256": arm["arm_query_sha256"], "frozen_context_sha256": neutral_sha,
            "cost_epoch_sha256": cost["cost_epoch_sha256"]}
        semantic = copy.deepcopy(original_body["semantic"])
        consumed = semantic["pc_consumption"][0]
        consumed.update(question_sha256=arm["arm_query_sha256"], view=copy.deepcopy(view), full_view_sha256=arm["full_view_sha256"])
        consumed["origin"].update(frame_sequence=arm["frame_sequence"])
        consumed["origin"]["ticket"].update(question_sha256=arm["arm_query_sha256"], view=copy.deepcopy(view))
        # Re-seal only the synthetic original accepted-origin projection; the
        # parser still cannot issue this as a live accepted role result.
        projected = shared._ticket_projection(consumed["origin"]["ticket"], action["authority"])
        from test_rovent_utility_root_local import origin_seal
        consumed["origin"]["seal_sha256"] = origin_seal({**consumed["origin"], "ticket": projected})
        previous = None if index == 0 else {"trial_sequence": 1,
            "trial_sha256": native_bytes_sha(metadata.TRIAL_DOMAIN, blobs["trials/000000.json"]),
            "file_sha256": hashlib.sha256(blobs["trials/000000.json"]).hexdigest()}
        body = {"schema": metadata.MATERIALIZATION_DOMAIN, "cohort_id": cohort, "trial_sequence": index + 1,
            "recipient": arm["recipient"], "profile_sha256": profile_sha, "source": copy.deepcopy(source["source_pins"]),
            "original_root": copy.deepcopy(original["original_root"]), "preparation": copy.deepcopy(original["ready_prelude"]),
            "neutral_target": neutral, "arm_query": arm, "action": action, "semantic": semantic}
        if change_arm is not None:
            change_arm(body, index)
        body_raw = raw(body)
        trial = {"schema": metadata.TRIAL_DOMAIN, "cohort_id": cohort, "trial_sequence": index + 1,
            "recipient": arm["recipient"], "profile": profile, "profile_sha256": profile_sha,
            "source": copy.deepcopy(source["source_pins"]), "neutral_target": neutral, "neutral_target_sha256": neutral_sha,
            "arm_query": arm, "materialized_sha256": native_bytes_sha(metadata.MATERIALIZATION_DOMAIN, body_raw),
            "materialized_json": body_raw.decode(), "historical_unbound_input": None, "unresolved_source": None,
            "cost": cost, "original_root": body["original_root"], "preparation": body["preparation"],
            "physical_closure": copy.deepcopy(original["physical_closure"]), "previous_durable_ack": previous,
            "resolved": True, "comparable": True, "unknown_reason": None,
            "support_mask": [False] * 4 + [True, True, False], "target_known_mask": [False] * 7}
        blobs[f"materializations/{index:06d}.json"] = body_raw
        blobs[f"trials/{index:06d}.json"] = raw(trial)
        trials.append(trial)
    pair = {**copy.deepcopy(old["pair"]), "schema": metadata.PAIR_DOMAIN, "cohort_id": cohort,
        "neutral_target_sha256": trials[0]["neutral_target_sha256"],
        "left_trial_sha256": native_bytes_sha(metadata.TRIAL_DOMAIN, blobs["trials/000000.json"]),
        "right_trial_sha256": native_bytes_sha(metadata.TRIAL_DOMAIN, blobs["trials/000001.json"])}
    blobs["pairs/000000.json"] = raw(pair)
    publication = {"schema": files_module.PUBLICATION_DOMAIN, "cohort_id": cohort, "cost_scope": metadata.COST_SCOPE,
        **{name: pin(filename, blobs[filename]) for name, filename, _ in files_module.FIXED_FILES},
        "materializations": [pin(f"materializations/{i:06d}.json", blobs[f"materializations/{i:06d}.json"]) for i in range(2)],
        "trials": [pin(f"trials/{i:06d}.json", blobs[f"trials/{i:06d}.json"]) for i in range(2)],
        "pairs": [pin("pairs/000000.json", blobs["pairs/000000.json"])], "trial_count": 2, "pair_count": 1,
        "total_payload_bytes": sum(map(len, blobs.values()))}
    pub_raw = raw(publication)
    pub_pin = pin(files_module.PUBLICATION_FILE, pub_raw)
    entries = []
    for index, checkpoint in enumerate(publication["trials"]):
        size = publication["materializations"][index]["bytes"]
        entries.append({"trial_sequence": index + 1,
            "native_trial_sha256": native_bytes_sha(metadata.TRIAL_DOMAIN, blobs[checkpoint["path"]]),
            "checkpoint": copy.deepcopy(checkpoint), "collection_io": {"clock_epoch": index + 1,
                "started_tick_ns": 230, "finished_tick_ns": 232, "elapsed_ns": 2,
                "written_bytes": size + checkpoint["bytes"], "readback_bytes": size + 2 * checkpoint["bytes"],
                "synced_files": 3, "synced_directories": 6}})
    ack = {"schema": files_module.ACK_DOMAIN, "cohort_id": cohort, "cost_scope": metadata.COST_SCOPE,
        "publication": pub_pin, "trial_acknowledgements": entries, "collection_io": {
            "clock_epoch": 2, "started_tick_ns": 233, "finished_tick_ns": 236, "elapsed_ns": 3,
            "written_bytes": sum(publication[name]["bytes"] for name, _, _ in files_module.FIXED_FILES)
                + publication["pairs"][0]["bytes"] + len(pub_raw),
            "readback_bytes": publication["total_payload_bytes"] + len(pub_raw),
            "synced_files": len(blobs) + 1, "synced_directories": 4}}
    blobs[files_module.PUBLICATION_FILE], blobs[files_module.ACK_FILE] = pub_raw, raw(ack)
    registration = copy.deepcopy(old_prepared["registration"])
    registration["cost_epoch_sha256"] = trials[0]["cost"]["cost_epoch_sha256"]
    return old_prepared["recipe"], registration, old_prepared["parameter_spec"], blobs


def write_case(root, blobs):
    for name, body in blobs.items():
        path = root.joinpath(*name.split("/"))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
    return {"root": str(root), "publication": pin(str(root / files_module.PUBLICATION_FILE), blobs[files_module.PUBLICATION_FILE]),
        "durable_ack": pin(str(root / files_module.ACK_FILE), blobs[files_module.ACK_FILE])}


def reseal_files(blobs):
    """Rebuild synthetic FD pins after a nested semantic mutation."""
    publication = json.loads(blobs[files_module.PUBLICATION_FILE])
    for index in range(1, len(publication["trials"])):
        previous = blobs[f"trials/{index - 1:06d}.json"]
        body = json.loads(blobs[f"trials/{index:06d}.json"])
        body["previous_durable_ack"] = {"trial_sequence": index,
            "trial_sha256": native_bytes_sha(metadata.TRIAL_DOMAIN, previous),
            "file_sha256": hashlib.sha256(previous).hexdigest()}
        blobs[f"trials/{index:06d}.json"] = raw(body)
    for entry in publication["pairs"]:
        pair = json.loads(blobs[entry["path"]])
        for side in ("left", "right"):
            number = pair[side + "_trial_sequence"]
            pair[side + "_trial_sha256"] = native_bytes_sha(metadata.TRIAL_DOMAIN, blobs[f"trials/{number - 1:06d}.json"])
        blobs[entry["path"]] = raw(pair)
    publication["materializations"] = [entry for entry in publication["materializations"] if entry["path"] in blobs]
    for entry in files_module.payload_pins(publication):
        entry.update(pin(entry["path"], blobs[entry["path"]]))
    publication["total_payload_bytes"] = sum(entry["bytes"] for entry in files_module.payload_pins(publication))
    pub_raw = raw(publication)
    ack = json.loads(blobs[files_module.ACK_FILE])
    ack["publication"] = pin(files_module.PUBLICATION_FILE, pub_raw)
    for index, entry in enumerate(ack["trial_acknowledgements"]):
        checkpoint = publication["trials"][index]
        size = next((p["bytes"] for p in publication["materializations"] if p["path"] == f"materializations/{index:06d}.json"), 0)
        entry.update(checkpoint=copy.deepcopy(checkpoint), native_trial_sha256=native_bytes_sha(metadata.TRIAL_DOMAIN, blobs[checkpoint["path"]]))
        entry["collection_io"].update(written_bytes=size + checkpoint["bytes"], readback_bytes=size + 2 * checkpoint["bytes"],
            synced_files=2 + int(size > 0), synced_directories=3 * (1 + int(size > 0)))
    ack["collection_io"].update(written_bytes=sum(publication[name]["bytes"] for name, _, _ in files_module.FIXED_FILES)
        + sum(entry["bytes"] for entry in publication["pairs"]) + len(pub_raw),
        readback_bytes=publication["total_payload_bytes"] + len(pub_raw),
        synced_files=len(files_module.payload_pins(publication)) + 1)
    blobs[files_module.PUBLICATION_FILE], blobs[files_module.ACK_FILE] = pub_raw, raw(ack)


class InterventionPayloadTests(unittest.TestCase):
    def test_p_reproposal_keeps_full_target_and_uses_only_original_first_prefix(self):
        full = [64, 3896, 72]
        for recipient, prefix in (("route_p", [64]), ("route_c", []), ("route_r", full)):
            reader._query_prefix({"proposal": full, "prefix": prefix}, recipient, 64, full)
        with self.assertRaises(ValueError):
            reader._query_prefix({"proposal": full, "prefix": full}, "route_p", 64, full)
        with self.assertRaises(ValueError):
            reader._query_prefix({"proposal": [64], "prefix": [64]}, "route_p", 64, full)

    def test_frozen_proposal_rejects_counterexample_even_with_matching_file_pins(self):
        # Mutate the original nested query before the fixture seals its trial,
        # materialization, publication and ACK. This is a policy rejection,
        # not rejection of stale file hashes or a copied live authority.
        for recipient_index in (0, 1):
            for counterexample in ([], [64]):
                with self.subTest(recipient=recipient_index, counterexample=counterexample):
                    def change_arm(body, index):
                        if index == recipient_index:
                            body["arm_query"]["query"]["counterexample"] = copy.deepcopy(counterexample)
                    _, registration, _, blobs = intervention_case(change_arm=change_arm)
                    with tempfile.TemporaryDirectory() as directory:
                        reference = write_case(Path(directory).resolve(), blobs)
                        owner = files_module.read_intervention_files(reference, read_accounting=budget(4 << 20))
                        with self.assertRaisesRegex(ValueError, "frozen-proposal arm cannot contain a counterexample"):
                            reader.validate_intervention_payloads(owner, registration=registration)

    def test_neutral_hash_keeps_reference_bytes_and_original_serde_identity_order(self):
        _, _, blobs, _, _, case = full_cohort_case()
        neutral = neutral_from_input(case["input"])
        config = numeric.NnuecConfig.from_mapping(json.loads(blobs["nnuec-weights.json"])["base"]["config"])
        captured, sealed = reader._neutral(neutral, config=config, source=case["source"])
        self.assertEqual(captured.input_json, neutral["reference_json"])
        self.assertEqual(reader.snapshot_sha256(neutral["root_rules"]), independent_snapshot(neutral["root_rules"]))
        self.assertNotEqual(sealed, hashlib.sha256(metadata.canonical_bytes(neutral)).hexdigest())
        for name in ("original_view_sha256", "original_full_view_sha256"):
            changed = copy.deepcopy(neutral); changed[name] = "ab" * 32
            self.assertEqual(reader.neutral_target_sha256(changed), sealed)
        changed = copy.deepcopy(neutral)
        changed["reference_json"] += " "
        changed["input_bytes"] += 1
        self.assertNotEqual(reader.neutral_target_sha256(changed), sealed)
        with self.assertRaises(ValueError):
            reader._neutral(changed, config=config, source=case["source"])

    def test_actual_question_frame_uses_private_sequence_and_optional_original_tags(self):
        value = {"sequence": 123, "stage_code": 2, "view_digest": "10" * 32, "view_seal_digest": "11" * 32,
            "focus_prefix_sha256": "12" * 32, "goal_local_sha256": None, "legacy_full_records_sha256": None}
        for goal, legacy in ((None, None), ("13" * 32, None), (None, "14" * 32), ("13" * 32, "14" * 32)):
            value.update(goal_local_sha256=goal, legacy_full_records_sha256=legacy)
            self.assertEqual(reader.question_sha256(value), independent_question(value))
        changed = copy.deepcopy(value); changed["sequence"] = True
        with self.assertRaises(ValueError): reader.question_sha256(changed)

    def test_two_distinct_actual_arm_views_and_own_epochs_prepare_without_retagging(self):
        recipe, registration, spec, blobs = intervention_case()
        with tempfile.TemporaryDirectory() as directory:
            reference = write_case(Path(directory).resolve(), blobs)
            owner = files_module.read_intervention_files(reference, read_accounting=budget(4 << 20))
            original = reader.validate_intervention_payloads(owner, registration=registration)
            queries = original["report"]["role_queries"]
            self.assertNotEqual(queries[0]["view_sha256"], queries[1]["view_sha256"])
            self.assertNotEqual(original["trials"][0]["cost"]["cost_epoch_sha256"],
                original["trials"][1]["cost"]["cost_epoch_sha256"])
            self.assertEqual(original["report"]["known_pairs"], [])
            self.assertFalse(original["report"]["training_target_authorized"])
            self.assertFalse(original["report"]["conditional_comparison_verified"])
            result = preparation.prepare_intervention_from_files(reference, recipe, registration, spec,
                read_budget=budget(4 << 20))
            self.assertEqual(result["intervention_source"]["role_queries"], queries)
            self.assertEqual(result["registration"]["cost_epoch_sha256"], registration["cost_epoch_sha256"])
            self.assertFalse(result["execution_authorized"])

    def test_changed_actual_arm_preimage_or_preparation_fails_with_all_outer_pins_resealed(self):
        changes = (lambda arm: arm["query"]["question_preimage"].update(sequence=999),
            lambda arm: arm["prepared_input"]["provenance"].update(full_view_sha256="a0" * 32),
            lambda arm: arm["budget"].update(original_store_byte_limit=1),
            lambda arm: arm.update(source_frame_sequence=999),
            lambda arm: arm["view"].update(context_policy_sha256="a1" * 32))
        for change in changes:
            def mutate(body, index):
                if index == 1: change(body["arm_query"])
            _, registration, _, blobs = intervention_case(change_arm=mutate)
            with tempfile.TemporaryDirectory() as directory:
                reference = write_case(Path(directory).resolve(), blobs)
                owner = files_module.read_intervention_files(reference, read_accounting=budget(4 << 20))
                with self.assertRaises(ValueError): reader.validate_intervention_payloads(owner, registration=registration)

    def test_original_common_witness_budget_default_and_explicit_arm_budget_are_distinct(self):
        witness = {"observation": 7, "kind": "Counter", "identity_sha256": h("synthetic witness"),
            "source_question_sha256": h("synthetic question"), "source_frame_sequence": 2,
            "raw_evidence_sha256": h("synthetic raw evidence"), "line": [64]}
        row = {name: copy.deepcopy(value) for name, value in witness.items() if name != "line"}
        row.update(basis=None, unavailable_reason="missing_original_comparison")
        scope = {"trial": {"recipient": "route_c"}, "materialization": {"semantic": {
            "new_witnesses": [witness], "witness_preimages": [{"preimage_bytes": 2}], "witness_basis": [row]},
            "prelude": {"offers": [{"recipient": "route_c", "budget": {"original_store_byte_limit": 2}}]}}}
        self.assertFalse(shared._witness_basis(scope))
        del scope["materialization"]["prelude"]
        self.assertFalse(shared._witness_basis(scope, original_store_byte_limit=2))
        with self.assertRaises(ValueError): shared._witness_basis(scope, original_store_byte_limit=1)

    def test_pair_nineteen_field_codec_rejects_changed_neutral_even_after_outer_resealing(self):
        _, registration, _, blobs = intervention_case()
        pair = json.loads(blobs["pairs/000000.json"])
        self.assertEqual(len(pair), 19)
        pair["neutral_target_sha256"] = "ee" * 32
        blobs["pairs/000000.json"] = raw(pair)
        reseal_files(blobs)
        with tempfile.TemporaryDirectory() as directory:
            reference = write_case(Path(directory).resolve(), blobs)
            owner = files_module.read_intervention_files(reference, read_accounting=budget(4 << 20))
            with self.assertRaisesRegex(ValueError, "neutral content seal"):
                reader.validate_intervention_payloads(owner, registration=registration)

    def test_insufficient_actual_actions_stays_unknown_without_synthetic_role_queries(self):
        _, registration, _, blobs = intervention_case()
        for index in range(2):
            key = f"trials/{index:06d}.json"
            trial = json.loads(blobs[key])
            trial.update(arm_query=None, materialized_sha256=None, materialized_json=None,
                resolved=False, comparable=False, unknown_reason="InsufficientEligibleActions",
                support_mask=[False] * 4 + [True, False, False])
            trial["cost"]["complete_scope"] = False
            blobs[key] = raw(trial)
            del blobs[f"materializations/{index:06d}.json"]
        pair = json.loads(blobs["pairs/000000.json"])
        pair.update(unknown_reason="unresolved_or_masked_trial", support_mask=[False] * 4 + [True, False, False])
        blobs["pairs/000000.json"] = raw(pair)
        reseal_files(blobs)
        with tempfile.TemporaryDirectory() as directory:
            reference = write_case(Path(directory).resolve(), blobs)
            owner = files_module.read_intervention_files(reference, read_accounting=budget(4 << 20))
            result = reader.validate_intervention_payloads(owner, registration=registration)
            self.assertEqual(result["report"]["known_pairs"], [])
            self.assertEqual(result["report"]["role_queries"], [])
            self.assertTrue(all(t["unknown_reason"] == "InsufficientEligibleActions" for t in result["trials"]))
            self.assertFalse(result["report"]["training_target_authorized"])

    def test_resolved_arm_cannot_set_an_undeclared_or_ineligible_support_bit(self):
        for mutation in (lambda t: t["support_mask"].__setitem__(6, True),
                lambda t: t["support_mask"].__setitem__(5, False)):
            _, registration, _, blobs = intervention_case()
            trial = json.loads(blobs["trials/000001.json"])
            mutation(trial)
            blobs["trials/000001.json"] = raw(trial)
            reseal_files(blobs)
            with tempfile.TemporaryDirectory() as directory:
                reference = write_case(Path(directory).resolve(), blobs)
                owner = files_module.read_intervention_files(reference, read_accounting=budget(4 << 20))
                with self.assertRaises(ValueError):
                    reader.validate_intervention_payloads(owner, registration=registration)


if __name__ == "__main__":
    unittest.main()
