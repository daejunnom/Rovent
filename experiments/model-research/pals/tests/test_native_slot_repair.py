"""Synthetic conditional lineage checks, never native execution certification.

The real strict parent/D/Repair/semantic factories are used. Build, launch,
physical heads and Rules descriptors are explicitly synthetic caller facts.
Static twenty-path byte-custody inputs exercise a scoped synthetic profile;
the live checkout is not read. No compiler, process, Rules replay, model or
training runs. Historical literal review banks and allow-lists remain unchanged.
Known is conditional fixture coverage only. The usual engine path observes
publication at the next D, which stays not_examined. These bytes do not prove
execution of the narrow ordinary branch/role-limit path that can be known.
New closure-consumer cases use only static fixture text and a scoped synthetic
review registry; no production source pin or enrollment is changed.
"""
import copy
import hashlib
import json
import tempfile
import unittest
from unittest.mock import patch

from rz_pals_model import frozen_producer as frozen
from rz_pals_model import native_divergence as native
from rz_pals_model import native_slot_repair as slot_witness
from rz_pals_model import repair_context as repair
from rz_pals_model import semantic_verifier as semantic
from rz_pals_model import training
from test_native_divergence import NativeDivergenceFixture
from test_repair_context import RepairFixture
from test_semantic_verifier import SemanticFixture, tokens
import test_training as fixtures
from native_source_review_fixture import (
    SYNTHETIC_CUSTODY_SOURCE_ID as FIXTURE_SOURCE_ID,
    SYNTHETIC_CUSTODY_TRANSITION_PINS as FIXTURE_TRANSITION_PINS,
    SYNTHETIC_CUSTODY_COMPILED_FACT as FIXTURE_COMPILED_FACT,
    synthetic_custody_fixture_manifest as fixture_manifest,
    synthetic_custody_fixture_review_scope as fixture_review_scope,
    synthetic_custody_fixture_sources,
)


# Historical literal banks exercise table preservation and unknown-successor
# rejection. Ordinary custody fixtures below do not match a checkout to them.
REVIEWED_PRE_LINT_SOURCE_PINS = {
    "crates/rz-search/src/pals/engine.rs": {
        "bytes": 374235, "sha256": "b86af1f0e55f55add0d4cca277772f8d1c8edf3b3d93d147629c4ab6f881103a"},
    "crates/rz-arena/src/pals_collect/native.rs": {
        "bytes": 233158, "sha256": "1ba6d1e5c7c6928dc9962cc51fd28dfe9a01fff91076c37da646390ae6973aa9"},
}

REVIEWED_LINT_SOURCE_PINS = {
    "crates/rz-search/src/pals/engine.rs": {
        "bytes": 374216, "sha256": "4729c3595731d82e7252dd4aec316bf051092449d95f4fe8a4a83017038ee018"},
    "crates/rz-arena/src/pals_collect/native.rs": {
        "bytes": 233158, "sha256": "1ba6d1e5c7c6928dc9962cc51fd28dfe9a01fff91076c37da646390ae6973aa9"},
}

REVIEWED_CONTROLLED_SOURCE_PINS = {
    "crates/rz-search/src/pals/engine.rs": {
        "bytes": 382810, "sha256": "752d328ae53dc1b4c06a1f3b7cbbd311834dbb59961e598c4752d8fcb9222cf4"},
    "crates/rz-arena/src/pals_collect/native.rs": {
        "bytes": 233158, "sha256": "1ba6d1e5c7c6928dc9962cc51fd28dfe9a01fff91076c37da646390ae6973aa9"},
}

REVIEWED_CONTROLLED_FALLBACK_SOURCE_PINS = {
    "crates/rz-search/src/pals/engine.rs": {
        "bytes": 391894, "sha256": "4770a55361a994210a2238240ed4d2dea1a24d7a8b85524cc7c2db31d19f5e19"},
    "crates/rz-arena/src/pals_collect/native.rs": {
        "bytes": 233158, "sha256": "1ba6d1e5c7c6928dc9962cc51fd28dfe9a01fff91076c37da646390ae6973aa9"},
}

# Independently pinned historical live-binding pair; not a current-run pin.
REVIEWED_LIVE_BINDING_SOURCE_PINS = {
    "crates/rz-search/src/pals/engine.rs": {
        "bytes": 396086, "sha256": "004db31f9be5133bd1925b373e6df60e97da3d50979a7eb5a031b1dad4ca397b"},
    "crates/rz-arena/src/pals_collect/native.rs": {
        "bytes": 233158, "sha256": "1ba6d1e5c7c6928dc9962cc51fd28dfe9a01fff91076c37da646390ae6973aa9"},
}


REVIEWED_OBSERVED_PROGRESS_SOURCE_PINS = {
    "crates/rz-search/src/pals/engine.rs": {
        "bytes": 402637, "sha256": "4b272ed825d98f51ace408b2798093e88487f6b29f94e38b9d2a753f8fc263d1"},
    "crates/rz-arena/src/pals_collect/native.rs": {
        "bytes": 233158, "sha256": "1ba6d1e5c7c6928dc9962cc51fd28dfe9a01fff91076c37da646390ae6973aa9"},
}


REVIEWED_ORDERED_SCOPE_SOURCE_PINS = {
    "crates/rz-search/src/pals/engine.rs": {
        "bytes": 402843, "sha256": "70d4bae5ad4e19e86137e47cc66922182c5232cbe35b9b30d42bf43ec6850171"},
    "crates/rz-arena/src/pals_collect/native.rs": {
        "bytes": 233158, "sha256": "1ba6d1e5c7c6928dc9962cc51fd28dfe9a01fff91076c37da646390ae6973aa9"},
}


# Historical optional-review pair, independent of the production closed table.
# This preserves enrollment only, never a historical build or execution claim.
REVIEWED_OPTIONAL_REVIEW_SOURCE_PINS = {
    "crates/rz-search/src/pals/engine.rs": {
        "bytes": 414372, "sha256": "ce40b6f9ff8d13a41da8c9482b48528c1437896e9571c9630790c4b11b262726"},
    "crates/rz-arena/src/pals_collect/native.rs": {
        "bytes": 233208, "sha256": "02af59b9edbc350e6e1a204eb6543033bb4a859729eb2ed79c8b470cc6823a01"},
}


# Independent live-fixture pins after the compiled-source inventory-only review.
# Only the engine digest's source tuple changes; the collector body is unchanged.
REVIEWED_CURRENT_SOURCE_PINS = {
    "crates/rz-search/src/pals/engine.rs": {
        "bytes": 414554, "sha256": "0ba7c9e41b106f35516b8b234872a839f5b2f4227153ed79aa7ff1c503d12837"},
    "crates/rz-arena/src/pals_collect/native.rs": {
        "bytes": 233208, "sha256": "02af59b9edbc350e6e1a204eb6543033bb4a859729eb2ed79c8b470cc6823a01"},
}


def raw(value):
    return fixtures.frozen_fixture_bytes(value)


def lines(values):
    return b"".join(raw(value) + b"\n" for value in values)


def source_review_raws(fixture, **replacements):
    """Synthetic caller build assets, not evidence of a compiler/process run."""
    values = dict(zip(slot_witness._RAW_NAMES, (fixture.assets["native-work-summary.jsonl"], fixture.build_raw,
        fixture.manifest_raw, fixture.engine_raw, fixture.native_raw)))
    values.update(replacements)
    return values


class SlotFixture(RepairFixture):
    """One root, one D, one selected Reply slot, one Repair, later observation.

    No claim that these synthetic Rules facts describe a real played game.
    Complete/known history is declared only by the synthetic registered Rules
    receipt; this test never performs or duplicates a chess legality oracle.
    """
    def __init__(self, directory):
        self.ready = False
        original_bytes = fixtures.frozen_fixture_bytes

        def with_synthetic_compiled_fact(value):
            # Bind this declaration before the seed source's FIRST raw bytes,
            # producer registration and receipt are created. Mutating only
            # self.source later would leave the retained checked-source wire
            # without the fact used by the real divergence/slot factories.
            if (type(value) is list and len(value) == 2
                    and value[0] == training.CHECKED_SOURCE_DOMAIN
                    and type(value[1]) is dict and type(value[1].get("native")) is dict):
                value[1]["native"]["search_implementation_sha256"] = FIXTURE_COMPILED_FACT
            return original_bytes(value)

        with patch.object(fixtures, "frozen_fixture_bytes", side_effect=with_synthetic_compiled_fact):
            super().__init__(directory)
        seed_dir = self.root / "synthetic-slot-auxiliary"
        seed_dir.mkdir()
        seed = NativeDivergenceFixture(seed_dir)
        self.d_input, self.d_sidecar, self.d_tensor = copy.deepcopy(seed.input), copy.deepcopy(seed.sidecar), copy.deepcopy(seed.tensor)
        self.d_lineage, self.d_context = copy.deepcopy(seed.lineage), copy.deepcopy(seed.context)
        self.unexamined_site = copy.deepcopy(seed.context["divergence_sites"][0])
        root_snapshot = self.rows[0]["input"]["snapshot"]
        root_snapshot["position_command"] = "position startpos"
        proposal = self.lineages[1]["proposal"][:3]
        self.lineages[1]["proposal"] = proposal
        self.rows[1]["input"]["snapshot"].update(capture_sequence=4, input_revision=4,
            position_command="position startpos moves e2e4 e8d7")
        self.lineages[1]["request_sequence"] = 4
        self.d_input["snapshot"] = copy.deepcopy(root_snapshot)
        self.d_input["snapshot"].update(role="critic", capture_sequence=2, input_revision=2)
        self.d_tensor.update(records=[], required_critical_records=[])
        self.d_sidecar["record_sources"] = []
        self.d_lineage.update(request_sequence=2, proposal=proposal, divergence_plies=[1])
        self.d_tensor["divergence_features"] = [self.d_tensor["divergence_features"][1]]
        self.d_context["native_request"] = [1, 2]
        self.d_context["divergence_sites"] = [copy.deepcopy(self.d_context["divergence_sites"][1])]
        self.d_context["divergence_sites"][0]["slot"] = 0
        reply_row = copy.deepcopy(self.rows[0])
        reply_row["input"]["snapshot"].update(role="critic", capture_sequence=3, input_revision=3,
            position_command="position startpos moves e2e4", board_fen="4k3/8/8/8/4P3/8/8/4K3 b - - 0 1",
            rules_state_sha256=self.d_context["divergence_sites"][0]["prefix_rules_state_sha256"],
            rules_history_sha256=self.d_context["divergence_sites"][0]["prefix_rules_history_sha256"],
            transposition_sha256=fixtures.sha("synthetic Reply prefix transposition"), white_to_move=False,
            legal_moves=[fixtures.move(60, 51), fixtures.move(60, 52), fixtures.move(60, 53)])
        publication_row = copy.deepcopy(self.rows[1])
        publication_row["input"]["snapshot"].update(role="critic", capture_sequence=5, input_revision=5,
            position_command="position startpos moves e2e4 e8f7", board_fen="8/5k2/8/8/4P3/8/8/4K3 w - - 1 2",
            rules_state_sha256=fixtures.sha("synthetic later Reply state"), rules_history_sha256=fixtures.sha("synthetic later Reply history"),
            transposition_sha256=fixtures.sha("synthetic later Reply transposition"))
        self.rows.extend([reply_row, publication_row])
        self.lineages.extend([
            {**copy.deepcopy(self.lineages[1]), "request_sequence": 3, "native_query_kind": "Reply",
             "virtual_prefix": proposal[:1], "counterexample": None},
            {**copy.deepcopy(self.lineages[1]), "request_sequence": 5, "native_query_kind": "Reply",
             "virtual_prefix": [proposal[0], fixtures.move(60, 53)], "counterexample": None},
        ])
        self.sidecars.extend([fixtures.native_sidecar(reply_row), fixtures.native_sidecar(publication_row)])
        self.tensors.extend([json.loads(value["tensor_json"]) for value in self.sidecars[2:]])
        self.public = {"domain": "rz-pals-native-public-source/1", "game_id": root_snapshot["game_id"], "record_index": 1,
            "revision": 5, "origin_state_id": 999, "origin_state_id_is_advisory": True,
            "origin_rules_state_sha256": None, "origin_rules_identity_observation": "unknown", "kind": "Repair",
            "line": self.lineages[1]["counterexample"], "value": None, "completed_depth": 0, "scope": "Unknown",
            "white_score_perspective": True, "critical": False, "source_cpu_profile_sha256": self.source["cpu_profile_sha256"]}
        source_raws = synthetic_custody_fixture_sources()
        self.engine_raw, self.native_raw = (source_raws[path] for path in slot_witness._SOURCE_PATHS)
        self.manifest = fixture_manifest()
        self.build = {"schema": slot_witness.BUILD_SCHEMA, "source_commit": FIXTURE_SOURCE_ID,
            "actual_build_exit_code": 0, "source_verified_before_and_after_build": True,
            "binary": {"bytes": 32, "sha256": self.source["implementation_sha256"]},
            "compiler_artifact": {"reason": "compiler-artifact", "target": {"name": "pals_collect"},
                "features": ["pals-collection-onnx"], "fresh": False},
            "assurance": "synthetic caller build observation, no compiler or native process executed"}
        self.publish = True
        self.public_extra = None
        self.omit_output = None
        self.duplicate_public = False
        self.consume_reply = True
        self.bad_reply_history = False
        self.ready = True
        self.rebuild()

    def rebuild(self):
        if not self.ready:
            return super().rebuild()
        ordinary = []
        publics = ([self.public, *([self.public_extra] if self.public_extra is not None else [])] if self.publish else [])
        for row, sidecar, tensor, lineage in zip(self.rows, self.sidecars, self.tensors, self.lineages):
            snapshot = row["input"]["snapshot"]
            snapshot["public_records"] = ([{"observation_sha256": semantic.byte_pin(raw(public))["sha256"],
                                            "situation_revision": public["revision"]} for public in publics]
                                         if self.publish and lineage["request_sequence"] == 5 else [])
            if lineage["request_sequence"] == 3 and self.bad_reply_history:
                snapshot["rules_history_sha256"] = fixtures.sha("wrong actual Reply history")
            row["input"]["sha256"] = training.seal_snapshot(snapshot)
            identity = row["input"]["sha256"]
            lineage["input_sha256"] = identity
            query = [0.] * 16
            prefix, proposal, counter = lineage["virtual_prefix"], lineage["proposal"], lineage["counterexample"]
            query[:9] = [dict(Propose=0, Reply=1, Repair=2)[lineage["native_query_kind"]], len(prefix) / 256, len(proposal) / 256,
                         int(counter is not None), 0 if counter is None else len(counter) / 256,
                         len(snapshot["legal_moves"]) / 256, snapshot["input_revision"], 9.25, int(snapshot["white_to_move"])]
            for offset, movement in ((9, prefix[-1] if prefix else None), (12, proposal[0] if proposal else None)):
                if movement is not None:
                    source, target, promotion = training.move_components(movement)
                    query[offset:offset + 3] = [source / 63, target / 63, promotion / 4]
            query[15] = 1
            records = ([{"record_id": public["record_index"], "revision": public["revision"], "critical": False, "features": [0.] * 16}
                        for public in publics]
                       if snapshot["public_records"] else [])
            tensor.update(board=list(training._fen_board(snapshot["board_fen"])), records=records, required_critical_records=[],
                candidates=[dict(zip(("from", "to", "promotion"), training.move_components(move))) for move in snapshot["legal_moves"]],
                query=query, divergence_features=[], model_epoch=list(bytes.fromhex(snapshot["source"]["model_weights_sha256"])),
                situation_revision=snapshot["input_revision"], history_digest=list(bytes.fromhex(snapshot["rules_history_sha256"])))
            sidecar.update(input_sha256=identity, record_sources=copy.deepcopy(snapshot["public_records"]), model_epoch_kind="frozen_model_epoch",
                           tensor_json=json.dumps(tensor, separators=(",", ":"), allow_nan=False))
            fixtures.reseal_sidecar(sidecar)
            ordinary.append((row["input"], sidecar, lineage, tensor))
        d_snapshot = self.d_input["snapshot"]
        self.d_input["sha256"] = training.seal_snapshot(d_snapshot)
        identity = self.d_input["sha256"]
        self.d_tensor.update(board=list(training._fen_board(d_snapshot["board_fen"])), situation_revision=d_snapshot["input_revision"],
                            history_digest=list(bytes.fromhex(d_snapshot["rules_history_sha256"])))
        self.d_sidecar.update(input_sha256=identity, record_sources=[], model_epoch_kind="frozen_model_epoch",
                             tensor_json=json.dumps(self.d_tensor, separators=(",", ":"), allow_nan=False))
        fixtures.reseal_sidecar(self.d_sidecar)
        self.d_lineage["input_sha256"] = identity
        self.d_context.update(input_sha256=identity, native_request=[self.d_lineage["process_epoch"], self.d_lineage["request_sequence"]],
            tensor_sidecar_sha256=self.d_sidecar["sha256"], captured_input_revision=d_snapshot["input_revision"],
            proposal_move16=self.d_lineage["proposal"])
        self.d_context["challenged_line_sha256"] = hashlib.sha256(raw([
            native.CHALLENGED_DOMAIN, d_snapshot["rules_state_sha256"], self.d_lineage["proposal"]])).hexdigest()
        self.d_context["sha256"] = semantic.digest(native.CONTEXT_VERSION, {k: v for k, v in self.d_context.items() if k != "sha256"})
        self.d_lineage["native_divergence_context_sha256"] = self.d_context["sha256"]
        calls = [ordinary[0], (self.d_input, self.d_sidecar, self.d_lineage, self.d_tensor), ordinary[2], ordinary[1], ordinary[3]]
        journals, events, outputs = [], [], []
        template = json.loads(self.base["producer-prepared.jsonl"])
        for number, (frozen_input, sidecar, lineage, tensor) in enumerate(calls):
            snapshot, kind = frozen_input["snapshot"], lineage["native_query_kind"]
            journal = copy.deepcopy(template)
            journal["prepared"].update(input_sha256=frozen_input["sha256"], capture_sequence=snapshot["capture_sequence"],
                native_request=[lineage["process_epoch"], lineage["request_sequence"]], input_json=semantic.byte_pin(raw(frozen_input)),
                tensor_sidecar_json=semantic.byte_pin(raw(sidecar)), lineage_json=semantic.byte_pin(raw(lineage)), learning_input=kind != "Divergence")
            journal["sha256"] = training._sorted_canonical(training.PREPARED_PRODUCER_DOMAIN, journal["prepared"])
            journals.append(journal)
            consumed = self.consume_reply or lineage["request_sequence"] != 3
            stages = ("prepared", "physically_completed", "delivered", "search_consumed" if consumed else "logically_rejected")
            details = ({"prepared_before_submit": True, "native_query_kind": kind, "producer_metadata_admitted": True},
                       {"success": True, "logical_acceptance_inferred": False}, {"search_consumed": False},
                       {"search_consumed": True} if consumed else {"search_consumed": False, "reason": "synthetic canceled"})
            events.extend({"domain": "rz-pals-native-call-event/1", "game_id": snapshot["game_id"],
                "process_epoch": lineage["process_epoch"], "request_sequence": lineage["request_sequence"], "input_sha256": frozen_input["sha256"],
                "stage": stage, "observer_elapsed_us": number * 10 + stage_index + 1, "detail": details[stage_index]}
                for stage_index, stage in enumerate(stages))
            outputs.append({"domain": "rz-pals-native-physical-raw/1", "process_epoch": lineage["process_epoch"],
                "request_sequence": lineage["request_sequence"], "input_sha256": frozen_input["sha256"], "physical_completion_confirmed": True,
                "success": True, "raw": {"representation": "f32_ieee754_bits", "candidate_logits_bits": [0] * len(tensor["candidates"]),
                "wdl_logits_bits": [0] * 3, "divergence_logits_bits": [0] * len(lineage["divergence_plies"]) if kind == "Divergence" else None,
                "task_logits_bits": None, "private_latent_bits": [0] * 6144, "prediction_is_future_label": False}})
        bindings = [{"input_sha256": journal["prepared"]["input_sha256"], "game_id": journal["prepared"]["game_id"],
                     "producer_id": journal["prepared"]["producer_id"], "capture_sequence": journal["prepared"]["capture_sequence"],
                     "capture_evidence_sha256": journal["sha256"]} for journal in journals if journal["prepared"]["learning_input"]]
        capture = frozen.seal_capture({"version": frozen.CAPTURE_DOMAIN, "bindings": bindings})
        registry, split = json.loads(self.base["source-registry.jsonl"]), json.loads(self.base["split.jsonl"])
        view = training.ValidatedDataset(self.rows, split, registry, {}).current_view
        envelope = json.loads(self.base["producer-envelope.json"])["envelope"]
        envelope.update(raw_records=len(self.rows), unique_inputs=len(self.rows), current_view_sha256=view.sha256,
                        capture_sha256=capture["sha256"], capture_artifact=semantic.byte_pin(raw(capture)))
        envelope = frozen.seal_envelope(envelope)
        audit = json.loads(self.base["producer-audit.json"])
        audit.update(raw_records=len(self.rows), unique_inputs=len(self.rows), native_exact_metadata_inputs=len(self.rows),
                     envelope_sha256=envelope["sha256"], capture_sha256=capture["sha256"])
        self.work = {"domain": "rz-pals-native-search-work/1", "cpu_nodes": 1, "cpu_tasks_requested": 1, "cpu_reports_returned": 1,
            "cpu_work_observation_incomplete": False, "cpu_task_configuration_sha256": self.source["cpu_profile_sha256"],
            "search_result": None, "role_calls": 5, "search_consumed_role_outputs": 5 if self.consume_reply else 4}
        assets = dict(self.base)
        assets.update({"records.jsonl": lines(self.rows), "inputs.jsonl": lines([item[0] for item in ordinary]),
            "native-inputs.jsonl": lines([item[1] for item in ordinary]), "input-lineage.jsonl": lines([item[2] for item in calls]),
            "producer-prepared.jsonl": lines(journals), "producer-captures.json": raw(capture), "producer-envelope.json": raw(envelope),
            "producer-audit.json": raw(audit), "native-divergence-inputs.jsonl": lines([self.d_input]),
            "native-divergence-sidecars.jsonl": lines([self.d_sidecar]), "native-divergence-contexts.jsonl": lines([self.d_context]),
            "public-record-sources.jsonl": lines([self.public, copy.deepcopy(self.public)] if self.duplicate_public else publics),
            "native-events.jsonl": lines(events), "native-raw-outputs.jsonl": lines([value for value in outputs if value["request_sequence"] != self.omit_output]),
            "native-work-summary.jsonl": lines([self.work])})
        receipt = json.loads(self.base["receipt.json"])
        receipt["audit"]["records"] = len(self.rows)
        receipt["native_finish"] = {"receipt": {"physical_shutdown_confirmed": True, "native_buffers_released": True,
            "quarantined": False, "physical_runs_in_flight": 0, "observer_failures": 0,
            "physically_completed_role_calls": 5, "completed_role_inputs": 5, "delivered_role_inputs": 5,
            "search_consumed_role_inputs": self.work["search_consumed_role_outputs"], "failed_physical_role_calls": 0,
            "invalid_role_outputs": 0, "process_epoch": 1, "request_high_water": 5},
            "finish_error": None, "_collection_failure": None, "collection_accepted": True}
        for name, value in assets.items():
            if name != "receipt.json":
                receipt["artifacts"][name] = semantic.byte_pin(value)
        assets["receipt.json"] = raw(receipt)
        for name, value in assets.items():
            (self.collection / name).write_bytes(value)
        self.assets = assets
        # The inherited Repair-only fixture indexes root/Repair journals by
        # ordinary-row order, while the retained raw trace has native order.
        by_identity = {value["prepared"]["input_sha256"]: value for value in journals}
        self.journals = [by_identity[row["input"]["sha256"]] for row in self.rows]
        self.options["expected_receipt_sha256"] = semantic.byte_pin(assets["receipt.json"])["sha256"]
        self.parents = training.load_frozen_collected_dataset(self.collection, **self.options)
        self.root_index = next(i for i in self.parents.current_view.current_indices if self.parents.records[i]["input"]["sha256"] == self.rows[0]["input"]["sha256"])
        self.repair_index = next(i for i in self.parents.current_view.current_indices if self.parents.records[i]["input"]["sha256"] == self.rows[1]["input"]["sha256"])
        self.artifacts = {name: assets[name] for name in repair._NAMES}
        self.launch_raw = raw({"schema": native.LAUNCH_SCHEMA, "receipt_artifact": semantic.byte_pin(assets["receipt.json"]),
            "checked_source_artifact": semantic.byte_pin(self.source_raw), "collector_binary_sha256": self.source["implementation_sha256"],
            "assurance_scope": "independently_pinned_caller_collection_observation", "spawned": True, "reaped": True, "exit_code": 0, "timed_out": False})
        self._rules_checks()
        # Upgrade only the synthetic registered descriptors, not FEN parsing or
        # parent metadata, to known complete history in this conditional fixture.
        for fixture in self.semantic_fixtures:
            for descriptor in (fixture.receipt["root"], fixture.receipt["target"], fixture.receipt["claimed_line"]["final_state"]):
                descriptor.update(history_completeness="complete", history_origin="start_position", repetition_history_complete=True)
            if fixture.common["prefix"] == []:
                fixture.receipt["claimed_line"]["final_state"]["known_history_positions"] = len(self.lineages[1]["proposal"]) + 1
            fixture.reseal()
        self.prefix_check, self.proposal_check = [fixture.admit() for fixture in self.semantic_fixtures]
        self.context_and_pins()
        self.repair_cap = repair.admit_repair_context(**super().arguments())
        self.reply_fixture = self._semantic("reply", self.lineages[2]["virtual_prefix"], [])
        self.line_fixture = self._semantic("whole-line", [], self.public["line"])
        self.reply_cap, self.line_cap = self.reply_fixture.admit(), self.line_fixture.admit()
        d_artifacts = {name: assets[name] for name in (*native._BASE_NAMES, "native-divergence-contexts.jsonl")}
        d_pins = {key: self.pins[key] for key in native._COMMON_PINS}
        d_pins.update(artifacts={name: semantic.byte_pin(value) for name, value in d_artifacts.items()}, context=semantic.byte_pin(raw(self.d_context)),
                      rules_checks=[], auxiliary_input_sha256=self.d_input["sha256"])
        self.d_cap = native.admit_native_divergence(parents=self.parents, parent_index=self.root_index, artifacts=d_artifacts,
            registration_bytes=self.registration_raw, checked_source_bytes=self.source_raw, export_manifest_bytes=self.export_raw,
            launch_bytes=self.launch_raw, context_bytes=raw(self.d_context), expected_pins=d_pins)
        self.manifest_raw = raw(self.manifest)
        self.build["source_manifest"] = semantic.byte_pin(self.manifest_raw)
        self.build_raw = raw(self.build)
        self.refresh_pins()

    def _semantic(self, name, prefix, line):
        directory = self.root / ("synthetic-slot-rules-" + name)
        directory.mkdir(exist_ok=True)
        fixture = SemanticFixture(directory)
        fixture.parents, fixture.index = self.parents, self.root_index
        fixture.snapshot = copy.deepcopy(self.rows[0]["input"]["snapshot"])
        fixture.common.update(parent_input_sha256=self.rows[0]["input"]["sha256"], current_view_sha256=self.parents.current_view.sha256,
            question="unrestricted_recheck" if name == "reply" else "continuation_challenge", prefix=prefix,
            # Eligibility is determined by prefix/root restrictions, not by a
            # claimed line: an unrestricted whole-line question uses resume.
            claimed_line=line, allowed_tasks=["attack_repair"] if prefix else ["resume_task"])
        root = fixture.receipt["root"]
        self._descriptor(root, self.rows[0]["input"]["snapshot"], "root_legal", 1)
        root.update(history_completeness="complete", history_origin="start_position", repetition_history_complete=True)
        target = copy.deepcopy(root)
        if name == "reply":
            # Use the descriptor site pins, not a possibly corrupted actual
            # Reply snapshot, so the negative history fixture reaches the gate.
            expected = copy.deepcopy(self.rows[2]["input"]["snapshot"])
            active_site = next(site for site in self.d_context["divergence_sites"] if site["divergence_ply"] == len(prefix))
            expected["rules_history_sha256"] = active_site["prefix_rules_history_sha256"]
            self._descriptor(target, expected, "target_legal", 2)
        else:
            target["legal_tokens"] = tokens(target["legal_moves"], "target_legal", target["legal_moves"])
        fixture.receipt["target"] = target
        if line:
            final = copy.deepcopy(self.semantic_fixtures[0].receipt["claimed_line"]["final_state"])
            fixture.receipt["claimed_line"] = {"status": "legal_continuation", "legality_verified": True, "claim_truth": "unknown",
                "movements": tokens(line, "claimed_continuation"), "restriction_checked": False, "final_state": final}
        fixture.reseal()
        return fixture

    def refresh_pins(self):
        self.slot_pins = {name: semantic.byte_pin(value) for name, value in zip(slot_witness._RAW_NAMES,
            (self.assets["native-work-summary.jsonl"], self.build_raw, self.manifest_raw, self.engine_raw, self.native_raw))}
        self.slot_pins.update(divergence_sha256=self.d_cap.sha256, slot=0, initial_repair_sha256=self.repair_cap.sha256,
                             reply_rules_sha256=self.reply_cap.sha256, line_rules_sha256=self.line_cap.sha256)

    def slot_arguments(self):
        return dict(divergence=self.d_cap, slot=0, initial_repair=self.repair_cap, reply_prefix_rules=self.reply_cap,
            repaired_line_rules=self.line_cap, work_summary_bytes=self.assets["native-work-summary.jsonl"],
            build_registration_bytes=self.build_raw, source_manifest_bytes=self.manifest_raw,
            engine_source_bytes=self.engine_raw, native_source_bytes=self.native_raw, expected_pins=self.slot_pins)

    def admit_slot(self):
        return slot_witness.admit_native_slot_repair(**self.slot_arguments())


class NativeSlotClosureConsumerTests(unittest.TestCase):
    def fixture(self, lane="default_disabled"):
        from test_native_recheck_witness import SyntheticClosureBuild
        return SyntheticClosureBuild(lane)

    def review(self, fixture):
        values = {"work_summary": b"", **fixture.raws}
        return slot_witness._source_review(values, fixture.source)

    def test_new_disabled_pair_requires_explicit_full_closure_and_keeps_metadata_detached(self):
        from rz_pals_model import native_source_review as reviewed
        fixture = self.fixture()
        old_profiles = copy.deepcopy(slot_witness._SOURCE_REVIEW_PROFILES)
        with patch.object(reviewed, "REVIEWED_CLOSURE_PROFILES", ()):
            with self.assertRaisesRegex(slot_witness._Unsupported, "unreviewed_collector_transition_source_pair"):
                self.review(fixture)
        registered = copy.deepcopy(fixture.profile)
        with patch.object(reviewed, "REVIEWED_CLOSURE_PROFILES", (registered,)):
            result = self.review(fixture)
            proof = result["reviewed_transition_closure"]
            self.assertEqual(set(proof), {"schema", "review_id", "lane", "transition_sources", "component_pins",
                "compiled_search_implementation_sha256", "closure_sha256", "unreviewed_manifest_files",
                "historical_metadata_only", "execution_authorized", "original_native_authority_reissued",
                "rules_proof_authorized", "training_target_authorized"})
            self.assertEqual(proof["lane"], "default_disabled")
            self.assertEqual(proof["component_pins"], fixture.profile["component_pins"])
            self.assertEqual(result["reviewed_transition_sources"], fixture.profile["transition_sources"])
            self.assertIs(proof["historical_metadata_only"], True)
            for key in ("execution_authorized", "original_native_authority_reissued", "rules_proof_authorized", "training_target_authorized"):
                self.assertIs(proof[key], False)
            original_bytes = semantic.canonical(result)
            proof["component_pins"][slot_witness._SOURCE_PATHS[0]]["sha256"] = "0" * 64
            self.assertEqual(semantic.canonical(self.review(fixture)), original_bytes)
            self.assertEqual(registered, fixture.profile)
        self.assertEqual(slot_witness._SOURCE_REVIEW_PROFILES, old_profiles)

    def test_required_helpers_compiled_fact_and_lane_cannot_be_replaced_or_omitted(self):
        from rz_pals_model import native_source_review as reviewed
        helper = "crates/rz-search/src/pals/engine/semantic_state.rs"
        for change in ("replacement", "missing_helper", "profile_omission", "compiled_mismatch", "compiled_missing", "wrong_lane"):
            fixture = self.fixture()
            registered = copy.deepcopy(fixture.profile)
            if change == "replacement":
                next(row for row in fixture.manifest["files"] if row["path"] == helper).update(semantic.byte_pin(b"changed helper fixture\n"))
            elif change == "missing_helper":
                fixture.manifest["files"] = [row for row in fixture.manifest["files"] if row["path"] != helper]
            elif change == "profile_omission":
                del registered["component_pins"][helper]
                registered["closure_sha256"] = semantic.digest("rz-pals-native-reviewed-transition-closure/1", registered["component_pins"])
            elif change == "compiled_mismatch":
                fixture.source["native"]["search_implementation_sha256"] = "e" * 64
            elif change == "compiled_missing":
                del fixture.source["native"]["search_implementation_sha256"]
            else:
                registered["lane"] = "legacy_single_reply"
            fixture.reseal()
            boundary = {"replacement": "helper closure differs", "missing_helper": "helper closure differs",
                "profile_omission": "omits a required source helper", "compiled_mismatch": "compiled search closure differs",
                "compiled_missing": "original_compiled_search_closure_fact_missing", "wrong_lane": "unreviewed_collector_transition_source_pair"}[change]
            with self.subTest(change=change), patch.object(reviewed, "REVIEWED_CLOSURE_PROFILES", (registered,)):
                error_type = slot_witness._Unsupported if change in ("compiled_missing", "wrong_lane") else ValueError
                with self.assertRaisesRegex(error_type, boundary):
                    self.review(fixture)

    def test_actual_build_binary_and_source_binding_fail_before_closure_lookup(self):
        from rz_pals_model import native_source_review as reviewed
        for change in ("exit", "verified", "binary", "unbound_component"):
            fixture = self.fixture()
            if change == "exit":
                fixture.build["actual_build_exit_code"] = False
            elif change == "verified":
                fixture.build["source_verified_before_and_after_build"] = False
            elif change == "binary":
                fixture.build["binary"]["sha256"] = "e" * 64
            else:
                fixture.raws["engine_source"] += b"// absent from original manifest\n"
            fixture.reseal()
            boundary = "actual transition source bytes absent" if change == "unbound_component" else (
                "build binary differs" if change == "binary" else "successful pre/post collector build observation")
            with self.subTest(change=change), patch.object(reviewed, "review_transition_sources",
                    side_effect=AssertionError("closure lookup cannot mask original build failure")) as lookup:
                with self.assertRaisesRegex(ValueError, boundary):
                    self.review(fixture)
                lookup.assert_not_called()

    def test_new_closure_does_not_allow_active_policy_or_a_one_component_hybrid(self):
        from rz_pals_model import native_source_review as reviewed
        for name, path in zip(("engine_source", "native_source"), slot_witness._SOURCE_PATHS):
            fixture = self.fixture()
            fixture.raws[name] += b"// separately bound successor fixture\n"
            next(row for row in fixture.manifest["files"] if row["path"] == path).update(semantic.byte_pin(fixture.raws[name]))
            fixture.reseal()
            with self.subTest(component=name), patch.object(reviewed, "REVIEWED_CLOSURE_PROFILES", (copy.deepcopy(fixture.profile),)):
                with self.assertRaisesRegex(slot_witness._Unsupported, "unreviewed_collector_transition_source_pair"):
                    self.review(fixture)
        for owner in ("source", "native", "search_configuration", "independent_registry"):
            fixture = self.fixture()
            target = fixture.source if owner == "source" else fixture.source["native"] if owner == "native" else fixture.source["native"].setdefault(owner, {})
            target["pals_search_policy"] = "same_repaired_line_once_v1"
            with self.subTest(owner=owner), patch.object(reviewed, "REVIEWED_CLOSURE_PROFILES", (copy.deepcopy(fixture.profile),)):
                with self.assertRaisesRegex(slot_witness._Unsupported, "unreviewed_collector_refinement_policy"):
                    self.review(fixture)

    def test_historical_literal_lookup_keeps_old_bytes_without_closure_enrollment(self):
        from rz_pals_model import native_source_review as reviewed
        expected = dict(zip(slot_witness._SOURCE_PATHS, (
            {"bytes": 260542, "sha256": "4469f9d8721264c2053380fd8900ef00b23602d42220adb99b8199cef8b62afb"},
            {"bytes": 73365, "sha256": "564d2b29d873df1c7b0424bfcee0b52a8f314cbfd88fa68ef35f4fda99a600d5"})))
        before = semantic.canonical(expected)
        with patch.object(reviewed, "review_transition_sources", side_effect=AssertionError("historical lookup stays independent")) as lookup:
            retained = slot_witness._reviewed_source_pair(expected)
            retained[slot_witness._SOURCE_PATHS[0]]["sha256"] = "0" * 64
            self.assertEqual(semantic.canonical(slot_witness._reviewed_source_pair(expected)), before)
            lookup.assert_not_called()


class ReviewedFixtureSuccessorTests(unittest.TestCase):
    def test_bundle_capacity_bank_preserves_three_prior_banks_and_has_one_reviewed_change(self):
        import native_source_review_fixture as banks
        from rz_pals_model import native_source_review as reviewed
        old_factories = (banks.fixture_profile, banks.successor_fixture_profile, banks.pc_shared_fixture_profile)
        prior = [factory("default_disabled") for factory in old_factories]
        current = banks.bundle_capacity_fixture_profile("default_disabled")
        engine = slot_witness._SOURCE_PATHS[0]
        self.assertEqual(banks.bundle_capacity_fixture_manifest()["source_commit"],
                         "f1b82eae0f08241b53e885ed2e6e6d573221d54a")
        self.assertEqual(current["component_pins"][engine], {"bytes": 423275,
            "sha256": "7f1322ba591aef3c5f1f566c30f4f8184b9a736c935136fa4fed2c96273635e4"})
        self.assertEqual(len(current["component_pins"]), 14)
        self.assertEqual([path for path in prior[-1]["component_pins"]
            if prior[-1]["component_pins"][path] != current["component_pins"][path]], [engine])
        self.assertEqual(len({profile["review_id"] for profile in [*prior, current]}), 4)
        before = copy.deepcopy(reviewed.REVIEWED_CLOSURE_PROFILES)
        for lane in ("default_disabled", "legacy_single_reply"):
            with banks.bundle_capacity_fixture_review_scope(lane) as profile:
                manifest = banks.bundle_capacity_fixture_manifest()
                proof = reviewed.review_transition_sources(pair=profile["transition_sources"], manifest=manifest,
                    source={"native": {"search_implementation_sha256": banks.BUNDLE_CAPACITY_SYNTHETIC_COMPILED_FACT}}, lane=lane)
                self.assertEqual(proof["closure_sha256"], banks.BUNDLE_CAPACITY_REVIEWED_CLOSURE_SHA256)
                self.assertIs(proof["historical_metadata_only"], True)
                for name in ("execution_authorized", "original_native_authority_reissued",
                             "rules_proof_authorized", "training_target_authorized"):
                    self.assertIs(proof[name], False)
                for old in prior:
                    with self.assertRaisesRegex(ValueError, "compiled search closure differs"):
                        reviewed.review_transition_sources(pair=profile["transition_sources"], manifest=manifest,
                            source={"native": {"search_implementation_sha256": old["compiled_search_implementation_sha256"]}}, lane=lane)
            self.assertEqual(reviewed.REVIEWED_CLOSURE_PROFILES, before)
        self.assertEqual([factory("default_disabled") for factory in old_factories], prior)

    def test_pc_shared_bank_preserves_both_earlier_banks_and_requires_its_own_fact(self):
        import native_source_review_fixture as banks
        from rz_pals_model import native_source_review as reviewed
        prior = banks.fixture_profile("default_disabled")
        successor = banks.successor_fixture_profile("default_disabled")
        pc_shared = banks.pc_shared_fixture_profile("default_disabled")
        engine = slot_witness._SOURCE_PATHS[0]
        self.assertEqual(banks.pc_shared_fixture_manifest()["source_commit"],
                         "47b602087de810dbe77c1489a7459202de3dfc9f")
        self.assertEqual(pc_shared["component_pins"][engine], {"bytes": 423219,
            "sha256": "6f89838c7e95ff067bac86432c36ee221cf29acc334fe16d0f4f2b96c2d825d7"})
        self.assertEqual(len(pc_shared["component_pins"]), 14)
        self.assertEqual([path for path in successor["component_pins"]
            if successor["component_pins"][path] != pc_shared["component_pins"][path]], [engine])
        self.assertEqual(pc_shared["component_pins"][engine]["bytes"]
                         - successor["component_pins"][engine]["bytes"], 306)
        self.assertEqual(len({prior["review_id"], successor["review_id"], pc_shared["review_id"]}), 3)
        before = copy.deepcopy(reviewed.REVIEWED_CLOSURE_PROFILES)
        for lane in ("default_disabled", "legacy_single_reply"):
            with banks.pc_shared_fixture_review_scope(lane) as profile:
                source = {"native": {"search_implementation_sha256": banks.PC_SHARED_SYNTHETIC_COMPILED_FACT}}
                proof = reviewed.review_transition_sources(pair=profile["transition_sources"],
                    manifest=banks.pc_shared_fixture_manifest(), source=source, lane=lane)
                self.assertEqual(proof["closure_sha256"], banks.PC_SHARED_REVIEWED_CLOSURE_SHA256)
                self.assertIs(proof["historical_metadata_only"], True)
                for name in ("execution_authorized", "original_native_authority_reissued",
                             "rules_proof_authorized", "training_target_authorized"):
                    self.assertIs(proof[name], False)
                for old_fact in (banks.SYNTHETIC_COMPILED_FACT, banks.SUCCESSOR_SYNTHETIC_COMPILED_FACT):
                    with self.assertRaisesRegex(ValueError, "compiled search closure differs"):
                        reviewed.review_transition_sources(pair=profile["transition_sources"],
                            manifest=banks.pc_shared_fixture_manifest(),
                            source={"native": {"search_implementation_sha256": old_fact}}, lane=lane)
            self.assertEqual(reviewed.REVIEWED_CLOSURE_PROFILES, before)
        self.assertEqual(banks.fixture_profile("default_disabled"), prior)
        self.assertEqual(banks.successor_fixture_profile("default_disabled"), successor)

    def test_original_6ca_review_and_api_remain_distinct_from_explicit_successor(self):
        import native_source_review_fixture as banks
        old = banks.fixture_profile("default_disabled")
        current = banks.successor_fixture_profile("default_disabled")
        engine = slot_witness._SOURCE_PATHS[0]
        self.assertEqual(banks.REVIEWED_SOURCE_COMMIT, "6ca5946bceb9394d27575b8c3dc7df2ed9a48cb2")
        self.assertEqual(banks.fixture_manifest()["source_commit"], banks.REVIEWED_SOURCE_COMMIT)
        self.assertEqual(banks.successor_fixture_manifest()["source_commit"],
                         "932d7ae7006803dbb655249f7fa10e13483bdc12")
        self.assertEqual(old["component_pins"][engine], {"bytes": 422725,
            "sha256": "a86c61bc22dea02d505bc6526d482acc7dc197433b46229d6a3deed5d605e5ed"})
        self.assertEqual(current["component_pins"][engine], {"bytes": 422913,
            "sha256": "2729ef10909bdf21182d26b2edc4555f5cad7c17f55ac322111e7d867d8cbf53"})
        self.assertEqual(len(old["component_pins"]), 14)
        self.assertEqual(len(current["component_pins"]), 14)
        self.assertEqual([path for path in old["component_pins"]
            if old["component_pins"][path] != current["component_pins"][path]], [engine])
        self.assertNotEqual(old["review_id"], current["review_id"])
        self.assertNotEqual(old["closure_sha256"], current["closure_sha256"])
        current["component_pins"][engine]["bytes"] = 1
        self.assertEqual(banks.fixture_profile("default_disabled"), old)
        self.assertEqual(banks.successor_fixture_profile("default_disabled")["component_pins"][engine]["bytes"], 422913)

    def test_successor_scope_is_temporary_metadata_only_and_does_not_retag_prior_source(self):
        import native_source_review_fixture as banks
        from rz_pals_model import native_source_review as reviewed
        before = copy.deepcopy(reviewed.REVIEWED_CLOSURE_PROFILES)
        for lane in ("default_disabled", "legacy_single_reply"):
            with banks.successor_fixture_review_scope(lane) as profile:
                proof = reviewed.review_transition_sources(pair=profile["transition_sources"],
                    manifest=banks.successor_fixture_manifest(),
                    source={"native": {"search_implementation_sha256": banks.SUCCESSOR_SYNTHETIC_COMPILED_FACT}}, lane=lane)
                self.assertEqual(proof["review_id"], profile["review_id"])
                self.assertIs(proof["historical_metadata_only"], True)
                for name in ("execution_authorized", "original_native_authority_reissued",
                             "rules_proof_authorized", "training_target_authorized"):
                    self.assertIs(proof[name], False)
            self.assertEqual(reviewed.REVIEWED_CLOSURE_PROFILES, before)
        with banks.fixture_review_scope("default_disabled") as prior:
            with self.assertRaisesRegex(ValueError, "compiled search closure differs"):
                reviewed.review_transition_sources(pair=prior["transition_sources"],
                    manifest=banks.fixture_manifest(),
                    source={"native": {"search_implementation_sha256": banks.SUCCESSOR_SYNTHETIC_COMPILED_FACT}},
                    lane="default_disabled")
        self.assertEqual(reviewed.REVIEWED_CLOSURE_PROFILES, before)


class NativeSlotRepairTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(fixture_review_scope("default_disabled"))
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.fixture = SlotFixture(self.temp.name)

    def test_known_unique_chain_is_factual_conditional_and_all_targets_false(self):
        original_source = json.loads(self.fixture.source_raw)[1]
        self.assertEqual(original_source["native"]["search_implementation_sha256"], FIXTURE_COMPILED_FACT)
        self.assertEqual(original_source, self.fixture.source)
        checked = self.fixture.admit_slot()
        self.assertEqual(checked.status, "known")
        self.assertIs(checked.require_witness(), checked)
        body, admission = checked.context(), checked.admission()
        self.assertEqual(body["repaired_line"], self.fixture.public["line"])
        self.assertEqual(body["native_request"], [1, 2])
        self.assertEqual(body["reply_native_request"], [1, 3])
        self.assertEqual(body["initial_repair_native_request"], [1, 4])
        self.assertEqual(body["publication_native_request"], [1, 5])
        self.assertEqual(body["captured_input_revision"], 2)
        self.assertEqual(body["transition_profile"]["reviewed_transition_sources"], FIXTURE_TRANSITION_PINS)
        self.assertEqual(admission["scope"], "conditional_unique_prepared_lineage")
        self.assertFalse(body["direct_causal_ids_present"])
        self.assertFalse(body["before_dispatch_witness_claimed"])
        self.assertTrue(admission["all_target_masks_false"])
        for key in ("direct_causal_identity_admitted", "reply_policy_selected_response_admitted", "divergence_ranking_admitted",
                    "whole_line_ordinal_admitted", "strategic_repair_validity_admitted", "training_target_created", "loss_executed"):
            self.assertFalse(admission[key])

    def test_missing_publication_is_not_examined_not_a_synthetic_known_line(self):
        self.fixture.publish = False
        self.fixture.rebuild()
        checked = self.fixture.admit_slot()
        self.assertEqual(checked.status, "not_examined")
        self.assertEqual(checked.context()["reason"], "full_repaired_line_publication_not_observed")
        with self.assertRaises(ValueError):
            checked.require_witness()

    def test_missing_complete_raw_output_remains_unsupported(self):
        self.fixture.omit_output = 5
        self.fixture.rebuild()
        checked = self.fixture.admit_slot()
        self.assertEqual(checked.status, "unsupported")
        self.assertEqual(checked.context()["reason"], "missing_complete_physical_event_output_trace")

    def test_repair_after_unconsumed_reply_is_a_contradiction_not_absence(self):
        self.fixture.consume_reply = False
        self.fixture.rebuild()
        with self.assertRaisesRegex(ValueError, "unconsumed Reply"):
            self.fixture.admit_slot()

    def test_slot_or_independent_capability_pin_cannot_be_caller_rematched(self):
        args = self.fixture.slot_arguments()
        args["slot"] = 1
        with self.assertRaises(ValueError):
            slot_witness.admit_native_slot_repair(**args)
        args = self.fixture.slot_arguments()
        args["expected_pins"] = copy.deepcopy(args["expected_pins"])
        args["expected_pins"]["initial_repair_sha256"] = fixtures.sha("different Repair context")
        with self.assertRaises(ValueError):
            slot_witness.admit_native_slot_repair(**args)

    def test_actual_reply_history_must_match_same_D_site_registered_rules(self):
        self.fixture.bad_reply_history = True
        self.fixture.rebuild()
        with self.assertRaisesRegex(ValueError, "exact Reply/D slot"):
            self.fixture.admit_slot()

    def test_unknown_history_is_unsupported_without_python_history_completion(self):
        fixture = self.fixture.reply_fixture
        fixture.receipt["target"].update(history_completeness="unknown_prefix", history_origin="fen", repetition_history_complete=False)
        fixture.reseal()
        self.fixture.reply_cap = fixture.admit()
        self.fixture.refresh_pins()
        checked = self.fixture.admit_slot()
        self.assertEqual(checked.status, "unsupported")
        self.assertEqual(checked.context()["reason"], "unknown_full_rules_history")

    def test_source_pair_must_be_reviewed_and_bound_to_original_binary_build(self):
        self.fixture.build["binary"]["sha256"] = fixtures.sha("different collector binary")
        self.fixture.build_raw = raw(self.fixture.build)
        self.fixture.refresh_pins()
        with self.assertRaisesRegex(ValueError, "build binary"):
            self.fixture.admit_slot()
        self.fixture.build["binary"]["sha256"] = self.fixture.source["implementation_sha256"]
        self.fixture.engine_raw += b"\n"
        self.fixture.manifest["files"][0].update(semantic.byte_pin(self.fixture.engine_raw))
        self.fixture.manifest_raw = raw(self.fixture.manifest)
        self.fixture.build["source_manifest"] = semantic.byte_pin(self.fixture.manifest_raw)
        self.fixture.build_raw = raw(self.fixture.build)
        self.fixture.refresh_pins()
        checked = self.fixture.admit_slot()
        self.assertEqual(checked.status, "unsupported")
        self.assertEqual(checked.context()["reason"], "unreviewed_collector_transition_source_pair")

    def test_historical_pair_remains_a_separate_closed_profile(self):
        # The historical source bytes are not fabricated or loaded from Git.
        # This is the closed-table preservation seam, not a historical build
        # observation. Actual byte/binary binding is tested separately below.
        old = dict(zip(slot_witness._SOURCE_PATHS, (
            {"bytes": 260542, "sha256": "4469f9d8721264c2053380fd8900ef00b23602d42220adb99b8199cef8b62afb"},
            {"bytes": 73365, "sha256": "564d2b29d873df1c7b0424bfcee0b52a8f314cbfd88fa68ef35f4fda99a600d5"})))
        self.assertEqual(slot_witness._reviewed_source_pair(old), old)
        copied = slot_witness._reviewed_source_pair(old)
        copied[slot_witness._SOURCE_PATHS[0]]["sha256"] = "0" * 64
        self.assertEqual(slot_witness._reviewed_source_pair(old), old)
        earlier_disabled = dict(zip(slot_witness._SOURCE_PATHS, (
            {"bytes": 297307, "sha256": "9c0de95911cf54365abec3818f089c20287d80160ea05eee8acbd326f7b49d2c"},
            {"bytes": 73365, "sha256": "564d2b29d873df1c7b0424bfcee0b52a8f314cbfd88fa68ef35f4fda99a600d5"})))
        self.assertEqual(slot_witness._reviewed_source_pair(earlier_disabled), earlier_disabled)
        earlier_observer = dict(zip(slot_witness._SOURCE_PATHS, (
            {"bytes": 343988, "sha256": "0302e49b6a9784641ca90aba17490d404b793098f38836b7b3cf42a12ee131d1"},
            {"bytes": 153861, "sha256": "098005ae44fe5c5a9b0a2181d8cde32a40c3eeaaf0e733e2d92ec6859fd424e5"})))
        self.assertEqual(slot_witness._reviewed_source_pair(earlier_observer), earlier_observer)
        earlier_replay = dict(zip(slot_witness._SOURCE_PATHS, (
            {"bytes": 344097, "sha256": "0a0a4800e940700721ec856ab86c8d8dbb581f6291fd78976ce1f918ea5a40c3"},
            {"bytes": 193840, "sha256": "2c2bfb1c8717a68429d315fa576268ba95b9b6106caed57400b885f656d04ae7"})))
        self.assertEqual(slot_witness._reviewed_source_pair(earlier_replay), earlier_replay)
        earlier_selection = dict(zip(slot_witness._SOURCE_PATHS, (
            {"bytes": 355881, "sha256": "32e393b4ad83bfd33a1b2205cd110cdc0cf007909e7e9c7f617819ccf5c4e93c"},
            {"bytes": 220199, "sha256": "eea0654a5c11d6e0d69b5c5df71363a8e2d8f4fb7ad02cf5d766fb4b06ec456f"})))
        self.assertEqual(slot_witness._reviewed_source_pair(earlier_selection), earlier_selection)
        earlier_revision_fixture = dict(zip(slot_witness._SOURCE_PATHS, (
            {"bytes": 355881, "sha256": "32e393b4ad83bfd33a1b2205cd110cdc0cf007909e7e9c7f617819ccf5c4e93c"},
            {"bytes": 224380, "sha256": "a286a2403317be7f5f058b65488e7c9722d2b7f4b9bf683ea95e5c9f51a7aaf8"})))
        self.assertEqual(slot_witness._reviewed_source_pair(earlier_revision_fixture), earlier_revision_fixture)
        self.assertNotEqual(earlier_revision_fixture, old)

    def test_optional_review_pair_remains_a_separate_historical_profile(self):
        reviewed = slot_witness._reviewed_source_pair(REVIEWED_OPTIONAL_REVIEW_SOURCE_PINS)
        self.assertEqual(reviewed, REVIEWED_OPTIONAL_REVIEW_SOURCE_PINS)
        self.assertNotEqual(reviewed, REVIEWED_CURRENT_SOURCE_PINS)
        reviewed[slot_witness._SOURCE_PATHS[0]]["sha256"] = "0" * 64
        self.assertEqual(slot_witness._reviewed_source_pair(REVIEWED_OPTIONAL_REVIEW_SOURCE_PINS),
                         REVIEWED_OPTIONAL_REVIEW_SOURCE_PINS)

    def test_unknown_engine_or_native_source_pair_remains_unsupported(self):
        for name, index in (("engine_source", 0), ("native_source", 1)):
            with self.subTest(component=name):
                values = source_review_raws(self.fixture)
                values[name] += b"\n"
                manifest = copy.deepcopy(self.fixture.manifest)
                manifest["files"][index].update(semantic.byte_pin(values[name]))
                values["source_manifest"] = raw(manifest)
                build = copy.deepcopy(self.fixture.build)
                build["source_manifest"] = semantic.byte_pin(values["source_manifest"])
                values["build_registration"] = raw(build)
                with self.assertRaisesRegex(slot_witness._Unsupported, "unreviewed_collector_transition_source_pair"):
                    slot_witness._source_review(values, self.fixture.source)

    def test_synthetic_custody_profile_is_scoped_and_rejects_historical_hybrids(self):
        from native_source_review_fixture import IDLE_ARCHIVE_REVIEWED_SOURCE_COMMIT
        self.assertNotEqual(FIXTURE_SOURCE_ID, IDLE_ARCHIVE_REVIEWED_SOURCE_COMMIT)
        observed = {path: semantic.byte_pin(value) for path, value in zip(slot_witness._SOURCE_PATHS,
            (self.fixture.engine_raw, self.fixture.native_raw))}
        self.assertEqual(observed, FIXTURE_TRANSITION_PINS)
        # This new synthetic bank does not extend the product's historical
        # two-file allow-list. Its complete helper profile is scoped to this
        # test and remains necessary on every verify() call.
        with self.assertRaises(slot_witness._Unsupported):
            slot_witness._reviewed_source_pair(observed)
        from rz_pals_model import native_source_review as reviewed
        with patch.object(reviewed, "REVIEWED_CLOSURE_PROFILES", ()):
            with self.assertRaisesRegex(slot_witness._Unsupported, "unreviewed_collector_transition_source_pair"):
                slot_witness._source_review(source_review_raws(self.fixture), self.fixture.source)
        proof = slot_witness._source_review(source_review_raws(self.fixture), self.fixture.source)["reviewed_transition_closure"]
        expected_components = {
            "crates/rz-search/src/pals/engine.rs",
            "crates/rz-arena/src/pals_collect/native.rs",
            "crates/rz-search/src/pals/engine/semantic_state.rs",
            "crates/rz-search/src/pals/engine/researcher_admission.rs",
            "crates/rz-search/src/pals/engine/researcher_profile.rs",
            "crates/rz-search/src/pals/engine/review_lifecycle.rs",
            "crates/rz-search/src/pals/engine/cpu_lifecycle.rs",
            "crates/rz-search/src/pals/engine/followup_execution.rs",
            "crates/rz-search/src/pals/engine/archive_lifecycle.rs",
            "crates/rz-arena/src/pals_collect.rs",
            "crates/rz-arena/src/pals_collect/full_line.rs",
            "crates/rz-arena/src/pals_collect/divergence.rs",
            "crates/rz-arena/src/pals_collect/continuation.rs",
            "crates/rz-arena/src/pals_collect/producer.rs",
            "crates/rz-search/src/pals/engine/idle_archive.rs",
            "crates/rz-search/src/pals/engine/adaptive_port/archive_worker.rs",
            "crates/rz-search/src/pals/engine/preparing_root.rs",
            "crates/rz-search/src/pals/review_context.rs",
            "crates/rz-search/src/pals/engine/adaptive_root.rs",
            "crates/rz-search/src/pals/store/archive.rs",
        }
        self.assertEqual(set(proof["component_pins"]), expected_components)
        self.assertEqual(proof["component_pins"], {
            row["path"]: {name: row[name] for name in ("bytes", "sha256")}
            for row in self.fixture.manifest["files"]})
        self.assertEqual(proof["compiled_search_implementation_sha256"], FIXTURE_COMPILED_FACT)
        self.assertIs(proof["execution_authorized"], False)
        self.assertEqual(slot_witness._reviewed_source_pair(REVIEWED_CURRENT_SOURCE_PINS), REVIEWED_CURRENT_SOURCE_PINS)
        self.assertEqual(slot_witness._reviewed_source_pair(REVIEWED_PRE_LINT_SOURCE_PINS), REVIEWED_PRE_LINT_SOURCE_PINS)
        self.assertEqual(slot_witness._reviewed_source_pair(REVIEWED_LINT_SOURCE_PINS), REVIEWED_LINT_SOURCE_PINS)
        self.assertEqual(slot_witness._reviewed_source_pair(REVIEWED_CONTROLLED_SOURCE_PINS), REVIEWED_CONTROLLED_SOURCE_PINS)
        self.assertEqual(slot_witness._reviewed_source_pair(REVIEWED_CONTROLLED_FALLBACK_SOURCE_PINS), REVIEWED_CONTROLLED_FALLBACK_SOURCE_PINS)
        self.assertEqual(slot_witness._reviewed_source_pair(REVIEWED_LIVE_BINDING_SOURCE_PINS), REVIEWED_LIVE_BINDING_SOURCE_PINS)
        self.assertEqual(slot_witness._reviewed_source_pair(REVIEWED_OBSERVED_PROGRESS_SOURCE_PINS), REVIEWED_OBSERVED_PROGRESS_SOURCE_PINS)
        self.assertEqual(slot_witness._reviewed_source_pair(REVIEWED_ORDERED_SCOPE_SOURCE_PINS), REVIEWED_ORDERED_SCOPE_SOURCE_PINS)
        historical = dict(zip(slot_witness._SOURCE_PATHS, (
            {"bytes": 355881, "sha256": "32e393b4ad83bfd33a1b2205cd110cdc0cf007909e7e9c7f617819ccf5c4e93c"},
            {"bytes": 224380, "sha256": "a286a2403317be7f5f058b65488e7c9722d2b7f4b9bf683ea95e5c9f51a7aaf8"})))
        for path in slot_witness._SOURCE_PATHS:
            with self.subTest(historical_component=path):
                hybrid = copy.deepcopy(observed)
                hybrid[path] = copy.deepcopy(historical[path])
                with self.assertRaisesRegex(slot_witness._Unsupported, "unreviewed_collector_transition_source_pair"):
                    slot_witness._reviewed_source_pair(hybrid)
            with self.subTest(ordered_scope_component=path):
                hybrid = copy.deepcopy(observed)
                hybrid[path] = copy.deepcopy(REVIEWED_ORDERED_SCOPE_SOURCE_PINS[path])
                with self.assertRaisesRegex(slot_witness._Unsupported, "unreviewed_collector_transition_source_pair"):
                    slot_witness._reviewed_source_pair(hybrid)

    def test_actual_manifest_and_binary_bindings_precede_unknown_pair_mask(self):
        values = source_review_raws(self.fixture)
        values["engine_source"] += b"\n"  # Unknown but correctly manifest-bound engine.
        values["native_source"] += b"\n"  # Deliberately absent from the original manifest.
        manifest = copy.deepcopy(self.fixture.manifest)
        manifest["files"][0].update(semantic.byte_pin(values["engine_source"]))
        values["source_manifest"] = raw(manifest)
        build = copy.deepcopy(self.fixture.build)
        build["source_manifest"] = semantic.byte_pin(values["source_manifest"])
        values["build_registration"] = raw(build)
        with self.assertRaisesRegex(ValueError, "actual transition source bytes absent"):
            slot_witness._source_review(values, self.fixture.source)
        build["binary"]["sha256"] = fixtures.sha("unregistered binary despite unknown source")
        values["build_registration"] = raw(build)
        with self.assertRaisesRegex(ValueError, "build binary"):
            slot_witness._source_review(values, self.fixture.source)

    def test_active_refinement_declaration_is_not_a_disabled_collector_profile(self):
        values = source_review_raws(self.fixture)
        declared = copy.deepcopy(self.fixture.source)
        declared["native"]["search_version"] = "pals-restricted-refinement/0.1"
        slot_witness._source_review(values, declared)
        declared["native"]["search_version"] = "pals-restricted-refinement-post-repair-recheck/1"
        with self.assertRaisesRegex(slot_witness._Unsupported, "unreviewed_collector_refinement_policy"):
            slot_witness._source_review(values, declared)
        for owner in ("source", "native", "search_configuration", "independent_registry"):
            for marker in ("post_repair_recheck_policy", "pals_search_policy"):
                for policy in ("SameRepairedLineOnceV1", "Disabled", "FrozenModelWdlV2", "IterativeFrozenModelWdlV2"):
                    with self.subTest(owner=owner, marker=marker, policy=policy):
                        source = copy.deepcopy(self.fixture.source)
                        target = source if owner == "source" else source["native"] if owner == "native" else source["native"].setdefault(owner, {})
                        target[marker] = policy
                        with self.assertRaisesRegex(slot_witness._Unsupported, "unreviewed_collector_refinement_policy"):
                            slot_witness._source_review(values, source)

    def test_policy_registration_without_marker_is_still_not_a_disabled_profile(self):
        values = source_review_raws(self.fixture)
        registration = {
            "version": "rz-pals-native-refinement-registration/1",
            "base_registry_canonical_sha256": fixtures.sha("registered base"),
            "collector_binary_sha256": self.fixture.source["implementation_sha256"],
            "search_policy": {
                "version": "pals-post-repair-recheck/1",
                "policy": "same_repaired_line_once_v1",
                "search_identity": "pals-restricted-refinement-post-repair-recheck/1",
                "conditions_sha256": list(bytes.fromhex(
                    "bea44b7e9ab59f32dcffb1b4c597fd36a4b75037803aeeacd6c18785ce838166")),
            },
        }
        for owner in ("source", "native", "search_configuration", "independent_registry"):
            for marker, declaration in (("refinement_registration", registration),
                                        ("refinement_registration_sha256", semantic.byte_pin(raw(registration))["sha256"])):
                with self.subTest(owner=owner, marker=marker):
                    source = copy.deepcopy(self.fixture.source)
                    target = source if owner == "source" else source["native"] if owner == "native" else source["native"].setdefault(owner, {})
                    target[marker] = copy.deepcopy(declaration)
                    with self.assertRaisesRegex(slot_witness._Unsupported, "unreviewed_collector_refinement_policy"):
                        slot_witness._source_review(values, source)

    def test_all_policy_owners_preserve_omission_and_legacy_but_refuse_other_search_versions(self):
        values = source_review_raws(self.fixture)
        for owner in ("source", "native", "search_configuration", "independent_registry"):
            for version in (None, "pals-restricted-refinement-post-repair-recheck/1",
                            "pals-restricted-refinement-post-repair-continuation/1",
                            "rovezero.pals-full-line-v2/0.2", "post-repair-frozen-wdl-v2",
                            "post-repair-frozen-wdl-queue-v2", "unknown-search-version/1"):
                with self.subTest(owner=owner, version=version):
                    source = copy.deepcopy(self.fixture.source)
                    target = source if owner == "source" else source["native"] if owner == "native" else source["native"].setdefault(owner, {})
                    target["search_version"] = version
                    with self.assertRaisesRegex(slot_witness._Unsupported, "unreviewed_collector_refinement_policy"):
                        slot_witness._source_review(values, source)
                    target["search_version"] = "pals-restricted-refinement/0.1"
                    slot_witness._source_review(values, source)
                    del target["search_version"]
                    slot_witness._source_review(values, source)

    def test_returned_source_review_pins_cannot_mutate_closed_allowlist(self):
        values = source_review_raws(self.fixture)
        original = slot_witness._source_review(values, self.fixture.source)
        changed = slot_witness._source_review(values, self.fixture.source)
        changed["reviewed_transition_sources"][slot_witness._SOURCE_PATHS[0]]["bytes"] = 1
        changed["reviewed_transition_sources"][slot_witness._SOURCE_PATHS[1]]["sha256"] = "0" * 64
        changed["source_manifest"]["sha256"] = "1" * 64
        self.assertEqual(slot_witness._source_review(values, self.fixture.source), original)
        self.assertEqual(slot_witness._source_profile(values, self.fixture.source), original)

    def test_retain_original_work_summary_pin_no_partial_subtrace_can_be_added(self):
        args = self.fixture.slot_arguments()
        args["work_summary_bytes"] += b" "
        args["expected_pins"] = copy.deepcopy(args["expected_pins"])
        args["expected_pins"]["work_summary"] = semantic.byte_pin(args["work_summary_bytes"])
        with self.assertRaisesRegex(ValueError, "immutable byte pin"):
            slot_witness.admit_native_slot_repair(**args)

    def test_wrong_whole_repaired_claim_is_not_line_matching_authority(self):
        fixture = self.fixture.line_fixture
        fixture.common["claimed_line"] = self.fixture.lineages[1]["proposal"]
        fixture.receipt["claimed_line"]["movements"] = tokens(fixture.common["claimed_line"], "claimed_continuation")
        fixture.reseal()
        self.fixture.line_cap = fixture.admit()
        self.fixture.refresh_pins()
        with self.assertRaisesRegex(ValueError, "full root-anchored"):
            self.fixture.admit_slot()

    def test_unknown_constructor_and_callback_or_descriptor_substitution_are_refused(self):
        with self.assertRaises(ValueError):
            slot_witness.CheckedNativeSlotRepair()
        args = self.fixture.slot_arguments()
        args["divergence"] = self.fixture.d_cap.context()
        with self.assertRaisesRegex(ValueError, "descriptor-only"):
            slot_witness.admit_native_slot_repair(**args)
        args = self.fixture.slot_arguments()
        args["reply_prefix_rules"] = lambda: self.fixture.reply_cap
        with self.assertRaises(ValueError):
            slot_witness.admit_native_slot_repair(**args)

    def test_capability_copies_are_detached_and_persisted_seal_is_not_reload_authority(self):
        checked = self.fixture.admit_slot()
        before = checked.sha256
        copied = checked.context()
        copied["repaired_line"][0] = fixtures.move(12, 20)
        self.assertEqual(checked.context()["repaired_line"], self.fixture.public["line"])
        with self.assertRaises(AttributeError):
            checked._identity = "0" * 64
        persisted = raw({"context": checked.context(), "admission": checked.admission()})
        self.assertEqual(json.loads(persisted)["admission"]["scope"], slot_witness.SCOPE)
        # Re-admit only from the same real raw-backed factories/independent
        # pins; the persisted derived descriptor has no admission constructor.
        self.assertEqual(self.fixture.admit_slot().sha256, before)
        with self.assertRaises(ValueError):
            slot_witness.CheckedNativeSlotRepair(json.loads(persisted))

    def test_raw_parent_mutation_is_not_repaired_by_witness_verification(self):
        checked = self.fixture.admit_slot()
        self.fixture.parents.records[self.fixture.root_index]["input"]["snapshot"]["input_revision"] += 1
        with self.assertRaises(ValueError):
            checked.verify()

    def test_exact_repeated_selected_public_raw_bytes_collapse(self):
        self.fixture.duplicate_public = True
        self.fixture.rebuild()
        checked = self.fixture.admit_slot()
        self.assertEqual(checked.status, "known")
        self.assertEqual(checked.context()["public_source"], semantic.byte_pin(raw(self.fixture.public)))

    def test_different_epoch_in_owned_reply_is_not_one_chronological_window(self):
        self.fixture.lineages[2]["process_epoch"] = 2
        self.fixture.rebuild()
        with self.assertRaisesRegex(ValueError, "epoch/request/capture"):
            self.fixture.admit_slot()

    def test_missing_rules_capability_is_unsupported_without_callback_fallback(self):
        args = self.fixture.slot_arguments()
        args["reply_prefix_rules"] = None
        args["expected_pins"] = copy.deepcopy(args["expected_pins"])
        args["expected_pins"]["reply_rules_sha256"] = None
        checked = slot_witness.admit_native_slot_repair(**args)
        self.assertEqual(checked.status, "unsupported")
        self.assertEqual(checked.context()["reason"], "missing_registered_rules_capability")

    def _trace(self):
        return slot_witness._trace(self.fixture.assets, self.fixture.assets["native-work-summary.jsonl"],
                                   self.fixture.parents, self.fixture.source)

    def _chain(self, trace, groups):
        # These source-only negative tests deliberately corrupt a derived
        # internal view; they never use it to construct an admitted capability.
        return slot_witness._chain(self.fixture.d_cap, 0, self.fixture.repair_cap, self.fixture.reply_cap,
            self.fixture.line_cap, trace, groups, self.fixture.assets, self.fixture.source, self.fixture.slot_pins)

    def test_duplicate_reply_window_cannot_be_resolved_by_caller_selection(self):
        trace, groups = self._trace()
        duplicate = copy.deepcopy(trace[2])
        duplicate["index"] = trace[2]["index"]
        groups[0].insert(3, duplicate)
        with self.assertRaisesRegex(ValueError, "duplicate Reply"):
            self._chain(trace, groups)

    def test_multiple_initial_repair_branches_cannot_be_arbitrarily_chosen(self):
        trace, groups = self._trace()
        duplicate = copy.deepcopy(trace[3])
        duplicate["lineage"]["virtual_prefix"][-1] = fixtures.move(60, 53)
        groups[0].insert(4, duplicate)
        with self.assertRaisesRegex(ValueError, "multiple initial Repair"):
            self._chain(trace, groups)

    def test_next_D_window_never_fills_missing_publication_in_first_window(self):
        trace, groups = self._trace()
        boundary = copy.deepcopy(trace[1])
        boundary["index"] = trace[4]["index"]
        # The original selected public observation lies at the exclusive
        # boundary and cannot prove publication in this D window.
        groups[0].insert(4, boundary)
        value = self._chain(trace, groups)
        self.assertEqual(value["status"], "not_examined")
        self.assertEqual(value["reason"], "full_repaired_line_publication_not_observed")

    def test_distinct_selected_repaired_lines_are_ambiguous_not_caller_choice(self):
        self.fixture.public_extra = copy.deepcopy(self.fixture.public)
        self.fixture.public_extra.update(record_index=2, revision=6,
            line=self.fixture.public["line"][:-1] + [fixtures.move(4, 5)])
        self.fixture.rows[3]["input"]["snapshot"]["input_revision"] = 6
        self.fixture.rebuild()
        with self.assertRaisesRegex(ValueError, "multiple public repaired lines"):
            self.fixture.admit_slot()

    def test_second_unobserved_ordered_slot_stays_not_examined(self):
        proposal = self.fixture.lineages[1]["proposal"] + [fixtures.move(52, 44)]
        for lineage in self.fixture.lineages[1:]:
            lineage["proposal"] = copy.deepcopy(proposal)
        self.fixture.d_lineage.update(proposal=proposal, divergence_plies=[3, 1])
        observed = copy.deepcopy(self.fixture.d_context["divergence_sites"][0])
        observed["slot"] = 1
        self.fixture.d_context["divergence_sites"] = [self.fixture.unexamined_site, observed]
        self.fixture.d_tensor["divergence_features"] = [[3.25, 0., 0., 0., 0., 0., 0., 0.],
                                                       [1.25, 0., 0., 0., 0., 0., 0., 0.]]
        self.fixture.rebuild()
        args = self.fixture.slot_arguments()
        for name, pin in (("initial_repair", "initial_repair_sha256"), ("reply_prefix_rules", "reply_rules_sha256"),
                          ("repaired_line_rules", "line_rules_sha256")):
            args[name] = None
            args["expected_pins"][pin] = None
        checked = slot_witness.admit_native_slot_repair(**args)
        self.assertEqual(checked.status, "not_examined")
        self.assertEqual(checked.context()["site"]["divergence_ply"], 3)
        self.assertEqual(checked.context()["reason"], "slot_reply_not_observed_in_exact_D_window")

    def test_late_duplicate_consumption_is_a_contradiction_not_an_absence_mask(self):
        assets = dict(self.fixture.assets)
        events = [value for _, value in native._rows(assets["native-events.jsonl"])]
        events.insert(12, copy.deepcopy(events[11]))
        assets["native-events.jsonl"] = lines(events)
        with self.assertRaisesRegex(ValueError, "late/double/misordered"):
            slot_witness._trace(assets, assets["native-work-summary.jsonl"], self.fixture.parents, self.fixture.source)

    def test_resealed_reply_revision_regression_is_rejected_in_same_root_group(self):
        # Raw snapshot, purpose/query tensor, prepared journal, envelope,
        # receipt and independent pins are all refreshed by the real fixture
        # factories. This is not a stale-seal rejection.
        self.fixture.rows[2]["input"]["snapshot"]["input_revision"] = 1
        self.fixture.rebuild()
        with self.assertRaisesRegex(ValueError, "same-root input revision regressed"):
            self.fixture.admit_slot()

    def test_all_physical_output_ids_reject_bool_epoch_and_sequence(self):
        # Focused full-trace boundary: True compares equal to request 1, so
        # the old untyped value comparison would admit both corruptions.
        # No capability is constructed from these deliberately altered views.
        for field in ("process_epoch", "request_sequence"):
            with self.subTest(field=field):
                assets = dict(self.fixture.assets)
                outputs = [value for _, value in native._rows(assets["native-raw-outputs.jsonl"])]
                outputs[0][field] = True
                assets["native-raw-outputs.jsonl"] = lines(outputs)
                with self.assertRaises(ValueError):
                    slot_witness._trace(assets, assets["native-work-summary.jsonl"], self.fixture.parents, self.fixture.source)

    def test_public_projection_index_change_is_explicitly_unsupported(self):
        self.fixture.public_extra = copy.deepcopy(self.fixture.public)
        self.fixture.public_extra["record_index"] = 2
        self.fixture.rebuild()
        checked = self.fixture.admit_slot()
        self.assertEqual(checked.status, "unsupported")
        self.assertEqual(checked.context()["reason"], "unsupported_public_projection_change")
        self.assertTrue(checked.admission()["all_target_masks_false"])

    def test_public_projection_critical_change_is_explicitly_unsupported(self):
        trace, _ = self._trace()
        assets = dict(self.fixture.assets)
        promoted = copy.deepcopy(self.fixture.public)
        promoted["critical"] = True
        assets["public-record-sources.jsonl"] = lines([self.fixture.public, promoted])
        # Two successive projection views are not two identical record IDs
        # selected by the same input. Test their raw identity classification
        # directly; the full factory catches this exact unsupported reason.
        with self.assertRaisesRegex(slot_witness._Unsupported, "unsupported_public_projection_change"):
            slot_witness._publications(trace, assets, self.fixture.source)

    def test_same_revision_immutable_line_change_remains_contradictory(self):
        self.fixture.public_extra = copy.deepcopy(self.fixture.public)
        self.fixture.public_extra["record_index"] = 2
        self.fixture.public_extra["line"][-1] = fixtures.move(4, 5)
        self.fixture.rebuild()
        with self.assertRaisesRegex(ValueError, "contradictory native public source"):
            self.fixture.admit_slot()

    def test_same_semantics_different_lexical_bytes_are_not_collapsed(self):
        trace, _ = self._trace()
        assets = dict(self.fixture.assets)
        original = raw(self.fixture.public)
        assets["public-record-sources.jsonl"] = original + b"\n" + b" " + original + b"\n"
        with self.assertRaisesRegex(ValueError, "distinct lexical bytes"):
            slot_witness._publications(trace, assets, self.fixture.source)

    def test_same_hash_different_raw_bytes_cannot_be_collapsed(self):
        trace, _ = self._trace()
        assets = dict(self.fixture.assets)
        original = raw(self.fixture.public)
        other = b" " + original
        assets["public-record-sources.jsonl"] = original + b"\n" + other + b"\n"
        real_pin = slot_witness.byte_pin
        expected = real_pin(original)

        def synthetic_collision(value):
            if value == other:
                return {"bytes": len(other), "sha256": expected["sha256"]}
            return real_pin(value)

        # A deliberately injected hash collision tests the lexical comparison;
        # this patch cannot enroll any capability or caller authority.
        with patch.object(slot_witness, "byte_pin", side_effect=synthetic_collision):
            with self.assertRaisesRegex(ValueError, "different lexical bytes"):
                slot_witness._publications(trace, assets, self.fixture.source)

    def test_unsupported_projection_cannot_mask_a_later_immutable_contradiction(self):
        trace, _ = self._trace()
        assets = dict(self.fixture.assets)
        reindexed = copy.deepcopy(self.fixture.public)
        reindexed["record_index"] = 2
        contradicted = copy.deepcopy(self.fixture.public)
        contradicted["line"][-1] = fixtures.move(4, 5)
        assets["public-record-sources.jsonl"] = lines([self.fixture.public, reindexed, contradicted])
        with self.assertRaisesRegex(ValueError, "contradictory native public source"):
            slot_witness._publications(trace, assets, self.fixture.source)
