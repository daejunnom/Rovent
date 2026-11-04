"""Synthetic R wire/file fixtures, not real Rules/native/training evidence."""
import copy
import hashlib
import importlib.abc
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest


class NeuralImportsForbidden(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"torch", "numpy", "onnx", "onnxruntime"}:
            raise AssertionError("neural imports forbidden in R corpus intake fixtures")


_guard = NeuralImportsForbidden()
sys.meta_path.insert(0, _guard)
try:
    from rz_pals_model import rovent_r_corpus as corpus
    from rz_pals_model import rovent_r_readout as wire
    from rz_pals_model import generation_data as data
    from rz_pals_model.rovent_generation import SeriesRecipe
    from rz_pals_model.nnuec_reference import NnuecConfig
    from rz_pals_model.nnuec_v2_reference import ReferenceInput, SEMANTICS
    from test_rovent_generation import recipe_value
finally:
    sys.meta_path.remove(_guard)


def h(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def raw(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n").encode("utf-8")


def artifact(name, sha256):
    return {"path": name, "sha256": sha256, "bytes": 1, "source": "fixture-only", "license": "fixture-only"}


def config():
    return NnuecConfig(1, 1, 1, 1, 1, 3, 4, 8)


def domain_nul(domain, value):
    # Independent descriptor expected framing; generation_digest is different.
    return hashlib.sha256(domain.encode() + b"\0" + raw(value).rstrip(b"\n")).hexdigest()


def fixture():
    """Small wire joins only. No model or Rules implementation is called."""
    components = {"synthetic-source": h("component")}
    source_sha = corpus._source_aggregate("1" * 40, components)
    declared = recipe_value()
    declared["source_sha256"] = source_sha
    recipe = SeriesRecipe(declared)
    stage = recipe.stage("common")
    teachers = {actor.actor_id: {"engine": actor.family, "source_commit": actor.source_revision,
        "binary_sha256": actor.binary_sha256, "binary_bytes": 1, "profile_sha256": actor.profile_sha256,
        "canonical_sha256": h(actor.actor_id + "canonical"), "capability_sha256": h(actor.actor_id + "capability"),
        "options": {}, "assets": {}} for actor in recipe.actors}
    profiles = [{"domain": "rz-rovent-external-engine-profile/1", "purpose": "teacher",
        "profile": artifact(actor + ".json", pin["profile_sha256"]), "binary": artifact(actor, pin["binary_sha256"]),
        "canonical_sha256": pin["canonical_sha256"], "capability_sha256": pin["capability_sha256"],
        "source_commit": pin["source_commit"], "requested_threads": 1, "requested_hash_mib": 256} for actor, pin in teachers.items()]
    collection = {"schema": "rz-rovent-corpus-manifest/1", "corpus_id": "synthetic-r-only", "source_commit": "1" * 40,
        "source_sha256": source_sha, "runtime_sha256": h("loaded ORT file, not endpoint or Python"),
        "model_semantics_sha256": declared["model_semantics_sha256"], "model_epoch": 1, "context_epoch": 2,
        "recipe_sha256": recipe.sha256, "stage_id": "common", "round": 0, "round_id": stage.round_id,
        "student": declared["student_id"], "learner": declared["learner_id"], "one_learner": True,
        "inherited_input_sha256": [], "inherited_origin_sha256": [], "teachers": teachers,
        "opening_splits": declared["opening_splits"], "ancestor_index_sha256": []}
    manifest = {"schema": wire.MANIFEST_DOMAIN, "collection": collection,
        "pc_ancestor_index_sha256": [], "r_ancestor_index_sha256": []}
    anchor = {"rules_state_sha256": h("exact current Rules state"), "rules_history_sha256": h("exact history"),
        "board_fen": "8/8/8/8/8/8/4K3/7k w - - 0 1", "white_to_move": True}
    replay = {"schema": "rz-rovent-rules-position-replay/1", "origin": "fen", "completeness": "unknown_prefix",
        "start_fen": anchor["board_fen"], "moves": [], "observer_revision": 1, "anchor": anchor}
    evidence = {"anchor": anchor, "replay": replay, "replay_sha256": data.digest(replay["schema"], replay)}
    nnuec = {"schema": "rz-nnuec-fp32-v2", "semantics": SEMANTICS, "encoding": "fixture-origin-first",
        "implementation_sha256": h("NNUEC implementation"), "configuration_sha256": h("registered NNUEC configuration"),
        "weights": artifact("nnuec.json", h("raw NNUEC checkpoint file")), "provenance": artifact("nnuec-provenance.json", h("NNUEC provenance")),
        "diagnostic": False, "context_namespace_policy_sha256": h("C namespace"), "local_namespace_policy_sha256": h("L namespace"),
        "search_conditions_sha256": h("search conditions")}
    search = {"node_unit": "rovent-position-visits/1", "requested_nodes": 4096, "depth_plies": None, "mate_moves": None,
        "searchmoves": None, "registered_max_chess_plies": 256, "root_state_and_history_sha256": anchor["rules_state_sha256"],
        "rules_profile_sha256": h("Rules profile"), "search_configuration_sha256": h("search configuration")}
    resources = {"cpu_quota_millicpus": 2000, "pools": [{"actors": ["manager"], "max_threads": 1, "cpu_charge_millicpus": 1000}],
        "memory_high_bytes": 6 << 30, "memory_max_bytes": 12 << 30, "gpu_devices": 0, "physical_nn_concurrency": 1,
        "topology_policy": "SharedControl"}
    coordinator = {"policy_sha256": h("coordinator"), "cpu_credit_limit": 4096, "questions_max": 64, "ancestor_depth_max": 8,
        "waiting_jobs_max": 64, "role_tickets_max": 16}
    execution = {"domain": "rz-rovent-execution/1", "contract_revision": "0.1", "run_id": "synthetic-not-executable",
        "purpose": "collection", "engine": {"source_commit": collection["source_commit"], "source_components": components,
            "mode": "hybrid_pcr", "search": search, "teachers": profiles, "nnuec": nnuec},
        "resources": resources, "coordinator": coordinator, "wall_time_max_ms": 1000, "cleanup_max_ms": 500,
        "output_bytes_max": 1048576, "agent_renewable": True, "provenance": []}
    lock = {"domain": "rz-rovent-input-lock/1", "manifest": execution,
        "canonical_sha256": data.digest("rz-rovent-execution-canonical/1", execution)}
    authority = {"epoch": 1, "game": 1, "root": 1, "implementation_sha256": h("UCI original implementation")}
    ceilings = {"resource_plan": resources, "registered_max_chess_plies": 256,
        **{key: coordinator[key] for key in ("cpu_credit_limit", "questions_max", "ancestor_depth_max", "waiting_jobs_max", "role_tickets_max")},
        **{key: execution[key] for key in ("wall_time_max_ms", "cleanup_max_ms", "output_bytes_max")}}
    descriptor_input = {"policy": "exact_registered", "startup_lock_sha256": lock["canonical_sha256"],
        "state_and_history_sha256": anchor["rules_state_sha256"], "rules_profile_sha256": search["rules_profile_sha256"],
        "authority": authority, "node_unit": search["node_unit"], "search_configuration_sha256": search["search_configuration_sha256"],
        "coordinator_policy_sha256": coordinator["policy_sha256"], "original_go": {"coverage": "semantic_context_only",
            "requested_nodes": 4096, "depth_plies": None, "mate_moves": None, "searchmoves": None,
            "transport_simulation_cap": 4096, "original_parsed_limits": None}, "allowed_root_moves": ["e2e3"],
        "clock": {key: {"clock_epoch": 1, "at_ns": tick} for key, tick in (("hard", 1000), ("soft", 500), ("admission", 800), ("cleanup", 2000))},
        "declared_ceilings": ceilings}
    budget = domain_nul(corpus.BUDGET_DOMAIN, {key: value for key, value in descriptor_input.items() if key != "clock"})
    deadline = domain_nul(corpus.DEADLINE_DOMAIN, {"domain": corpus.DEADLINE_DOMAIN, "startup_lock_sha256": lock["canonical_sha256"],
        "policy": descriptor_input["policy"], "authority": authority, "root_budget_sha256": budget,
        "clock": descriptor_input["clock"], "cleanup_max_ms": 500})
    descriptor = {"domain": corpus.DESCRIPTOR_DOMAIN, "input": descriptor_input, "root_budget_sha256": budget, "deadline_seal_sha256": deadline}
    descriptor_sha = domain_nul(corpus.DESCRIPTOR_DOMAIN, descriptor)
    origin = {"schema": corpus.ORIGIN_DOMAIN, "purpose": "collection", "process_epoch": 1, "mode": "HybridPcr",
        "input_lock_sha256": lock["canonical_sha256"], "source_commit": collection["source_commit"], "source_components": components,
        "source_sha256": source_sha, "weights_sha256": h("P/C raw checkpoint, not NNUEC"), "export_manifest_sha256": h("P/C export"),
        "backend_sha256": h("backend"), "runtime_library_sha256": collection["runtime_sha256"], "runtime_bundle_sha256": None,
        "native_configuration_sha256": h("native configuration"), "model_semantics": "synthetic-PC-semantics",
        "model_semantics_sha256": collection["model_semantics_sha256"], "encoding": "synthetic-PC-encoding",
        "encoding_sha256": h("PC encoding"), "encoding_registry_sha256": h("encoding registry"), "model_implementation_sha256": h("PC implementation"),
        "frozen_epoch": 1, "representation_model_domain": "loaded_checkpoint_sha256",
        "runtime_sha256_domain": "loaded_runtime_library_file_sha256", "runtime_sha256": collection["runtime_sha256"], "teacher_profiles": profiles}
    admission = {"schema": corpus.ROOT_DOMAIN, "native_origin": origin, "native_origin_sha256": data.digest(corpus.ORIGIN_DOMAIN, origin),
        "process_epoch": 1, "game_generation": 1, "root_generation": 1, "implementation_sha256": authority["implementation_sha256"],
        "deadline_clock": 1, "deadline_tick": 1000, "root_budget_sha256": budget, "selective_policy_sha256": h("selective policy"),
        "root_state": {"owner": 1, "revision": 1, "semantic": anchor["rules_state_sha256"]}, "store_root_state": 1,
        "root_moves": ["e2e3"], "depth_plies": None, "mate_moves": None, "requested_nodes": 4096,
        "root_descriptor": descriptor, "root_descriptor_sha256": descriptor_sha}
    job = {"authority": {"process_epoch": 1, "game_generation": 1, "root_generation": 1, "implementation_sha256": authority["implementation_sha256"]},
        "job_sequence": 1, "state_owner": 1, "state_revision": 1, "state_semantic_sha256": anchor["rules_state_sha256"],
        "question_sha256": h("original question"), "root_restriction_sha256": h("root restriction"), "evaluation_namespace_sha256": h("evaluation namespace"),
        "search_conditions_sha256": nnuec["search_conditions_sha256"], "total_native_node_credit": 4096, "hard_deadline_clock_epoch": 1, "hard_deadline_ns": 1000}
    reference_input = {"schema": "rz-nnuec-fp32-input/2", "board_features": [[768, 774, 783], [769, 774, 783]],
        "side_to_move": "white", "context_lines": [], "context_edges": [],
        "local_line": {"anchor_features": [768, 774, 783], "perspective": "white", "moves": []}}
    bound = {"model_semantics": SEMANTICS, "parameter_sha256": h("parsed NNUEC parameters"), "training_state": {"kind": "untrained"}, "weights_file_sha256": nnuec["weights"]["sha256"],
        "collection_model_epoch": 1, "collection_context_epoch": 2, "root_state": 1, "generation": 1, "sequence": 1,
        "full_view_sha256": h("full view"), "view_seal_sha256": h("view seal"), "record_scope_sha256": h("record scope"),
        "policy_sha256": h("review policy"), "context_sha256": ReferenceInput.from_mapping(reference_input, config()).context_sha256,
        "context_query": {"purpose": "ProposePolicy", "prefix": [], "proposal": [], "counterexample": None, "legal": [], "divergences": []},
        "selection_policy": {"mode": "current_root_only", "token_budget": 512, "max_records": 64, "max_anchors": 1, "required_bundle": None},
        "actual_tokens": 0, "token_budget": 512, "length_bucket": 8,
        "context_records": [], "context_anchors": [evidence], "context_source_anchors": [], "context_anchor_states": [1],
        "local_anchor": evidence, "local_encoded_perspective": "white", "current_rules": evidence}
    bound["view_seal_sha256"] = hashlib.sha256(b"rz-pals-review-view-seal/1\0" + bytes.fromhex(bound["full_view_sha256"])
        + (1).to_bytes(8, "little") * 3).hexdigest()
    original_json = raw(reference_input).decode().rstrip("\n")
    input_evidence = {"reference_input_json": original_json, "reference_input_bytes": len(original_json.encode()),
        "reference_input_raw_sha256": h(original_json), "native_input_sha256": hashlib.sha256(wire.CAPTURE_INPUT_DOMAIN.encode() + b"\0" + original_json.encode()).hexdigest(),
        "semantic_input_sha256": wire.r_readout_f32_digest(wire.INPUT_DOMAIN,
            {key: bound[key] for key in ("model_semantics", "parameter_sha256", "collection_model_epoch", "collection_context_epoch")} | {"reference_input": reference_input})}
    question = {"job": job, "worker_owner": 0, "frame_sequence": 1, "capture_sequence": 1, "readout_sequence": 0, "observed_tick_ns": 1100}
    retirement = {"job": job, "worker_owner": 0, "frame_sequence": 1, "capture_sequence": 1,
        "ending": {"kind": "cpu_terminal", "completion": "completed"}, "limits": {"maximum_readouts": 64, "every_nth_score": 2, "maximum_input_bytes": 1048576},
        "attempted": 3, "selected": 1, "retained_readouts": 1, "overflow": 0, "retained_input_bytes": len(original_json.encode()),
        "selected_input_preparation_ns": 12, "error": None}
    numeric = {"schema": "rz-nnuec-fp32-output/2", "parameter_sha256": bound["parameter_sha256"], "board_accumulators_bits": [[0], [0]],
        "context_accumulator_bits": [0], "local_accumulator_bits": [0], "hidden_bits": [0], "raw_outputs_bits": [0] * 11,
        "value_bits": 0, "wdl_logits_bits": [0] * 3, "wdl_bits": [0x3e800000, 0x3f000000, 0x3e800000]}
    row = {"schema": wire.RECORD_DOMAIN, "manifest_sha256": data.digest(wire.MANIFEST_DOMAIN, manifest), "split": "train",
        "lineage": {"game_id": "r-game", "opening_id": "train-a", "line_genealogy_id": "r-line", "transposition_sha256": h("state-transposition"),
            "actor_id": "student-white", "teacher_actor_id": None, "ancestor_origin_sha256": [], "pc_genealogy": None},
        "native_admission": admission, "native_admission_sha256": data.digest(corpus.ROOT_DOMAIN, admission),
        "root_descriptor": descriptor, "root_descriptor_sha256": descriptor_sha, "startup_input_lock_sha256": lock["canonical_sha256"],
        "question": question, "bound": bound, "input": input_evidence, "numeric": numeric, "retirement": retirement,
        "retirement_sha256": data.digest(corpus.RETIREMENT_DOMAIN, retirement), "semantic_consumption": None, "whole_action_cost": None,
        "wdl": None, "utility": {"support_mask": [False] * 7, "target_known_mask": [False] * 7, "unknown_reason": "no_actual_action_closure"},
        "origin_sha256": "", "record_sha256": ""}
    seal(row)
    return recipe, manifest, row, lock


def seal(row):
    row["retirement_sha256"] = data.digest(corpus.RETIREMENT_DOMAIN, row["retirement"])
    origin = {key: row[key] for key in ("manifest_sha256", "lineage", "question", "native_admission_sha256", "root_descriptor_sha256")}
    origin.update({key: row["input"][key] for key in ("native_input_sha256", "semantic_input_sha256")})
    row["origin_sha256"] = data.digest(wire.ORIGIN_DOMAIN, origin)
    row["record_sha256"] = wire.record_digest(row)


def pin(name, payload):
    return {"path": name, "sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}


def write(root, manifest, rows, lock):
    root.mkdir(parents=True, exist_ok=True)
    payload = b"".join(raw(row) for row in rows)
    chunk = {**pin("records.jsonl", payload), "records": len(rows)}
    origins = [{"source_domain": wire.RECORD_DOMAIN, "record_sha256": row["record_sha256"],
        "input_sha256": row["input"]["semantic_input_sha256"], "origin_sha256": row["origin_sha256"],
        "split": row["split"], "lineage": row["lineage"]} for row in rows]
    index = {"schema": wire.INDEX_DOMAIN, "manifest_sha256": data.digest(wire.MANIFEST_DOMAIN, manifest),
        "record_count": len(rows), "chunks": [chunk], "origins": origins}
    blobs = {"manifest.json": raw(manifest), "index.json": raw(index), "records.jsonl": payload, "lock.json": raw(lock)}
    for name, body in blobs.items():
        (root / name).write_bytes(body)
    return corpus.RCorpusFiles.from_mapping({"root": str(root), "manifest": pin("manifest.json", blobs["manifest.json"]),
        "index": pin("index.json", blobs["index.json"]), "input_locks": {lock["canonical_sha256"]: pin("lock.json", blobs["lock.json"])}})


def open_owner(root, *, source=None, rows=None, **options):
    recipe, manifest, first, lock = fixture() if source is None else source
    files = write(root, manifest, [first] if rows is None else rows, lock)
    return corpus.RoventRCorpus.from_files(files, recipe=recipe, nnuec_config=config(),
        expected_nnuec_parameter_sha256=first["bound"]["parameter_sha256"], **options)


def submitted_pc_owner(root, recipe, collection, lock, *, opening="train-a", model_epoch=None):
    """New-format synthetic wire via the explicit loader, never a live source.

    Reuse only the schema fixture's original raw bodies. No real corpus is
    converted, and no parent Rules replay or historical native proof is claimed.
    """
    from rz_pals_model import rovent_corpus as pc
    from rz_pals_model import rovent_corpus_codec as codecs
    from rz_pals_model import rovent_submitted_input as submitted
    from rz_pals_model import rovent_native_v3 as native_v3
    from test_rovent_native_v3 import native_fixture
    manifest, row = native_fixture()
    family = codecs.SUBMITTED_CODEC
    for key in ("source_commit", "source_sha256", "runtime_sha256", "model_semantics_sha256", "recipe_sha256",
                "model_epoch", "context_epoch", "teachers", "opening_splits"):
        manifest[key] = copy.deepcopy(collection[key])
    if model_epoch is not None:
        manifest["model_epoch"] = model_epoch
    row["input"].update(opening_id=opening, model_epoch=manifest["model_epoch"], context_epoch=manifest["context_epoch"])
    row["split"] = manifest["opening_splits"][opening]
    row["input_sha256"] = submitted.seal_input(row["input"])
    native = row["native"]
    native["source_sha256"] = manifest["source_sha256"]
    root_admission = native["raw_retirement"]["root_admission"]
    origin = root_admission["native_origin"]
    origin.update(source_commit=manifest["source_commit"], source_sha256=manifest["source_sha256"],
        source_components=copy.deepcopy(lock["manifest"]["engine"]["source_components"]),
        runtime_sha256=manifest["runtime_sha256"], runtime_library_sha256=manifest["runtime_sha256"],
        model_semantics_sha256=manifest["model_semantics_sha256"])
    root_admission["native_origin_sha256"] = data.digest(corpus.ORIGIN_DOMAIN, origin)
    native["raw_retirement"]["root_admission_sha256"] = data.digest(corpus.ROOT_DOMAIN, root_admission)
    for name in ("raw_submission", "raw_completion", "raw_retirement"):
        native[name].update(source_sha256=manifest["source_sha256"], input_sha256=row["input_sha256"])
        if name != "raw_submission":
            native[name]["submitted_sha256"] = native["raw_submission_sha256"]
        if name == "raw_retirement":
            native[name]["completed_sha256"] = native["raw_completion_sha256"]
        native[name + "_sha256"] = data.digest(native_v3.NATIVE_DOMAIN, native[name])
    row["origin_sha256"] = data.digest(pc.ORIGIN_DOMAIN, pc._origin(manifest, row))
    row["record_sha256"] = pc._record_digest(row)
    indexed = {key: row[key] for key in ("record_sha256", "input_sha256", "origin_sha256", "split")}
    indexed.update({key: row["input"][key] for key in ("game_id", "opening_id", "line_genealogy_id", "transposition_sha256")})
    indexed.update(copy.deepcopy(row["lineage"]))
    root.mkdir(parents=True, exist_ok=True)
    chunk = raw(row)
    (root / "chunks").mkdir()
    index = {"schema": family.index_domain, "manifest_sha256": data.digest(family.manifest_domain, manifest),
        "record_count": 1, "chunks": [{"path": "chunks/000000.jsonl", "bytes": len(chunk),
            "sha256": hashlib.sha256(chunk).hexdigest(), "records": 1}], "origins": [indexed]}
    manifest_bytes, index_bytes = raw(manifest), raw(index)
    (root / "manifest.json").write_bytes(manifest_bytes)
    (root / "index.json").write_bytes(index_bytes)
    (root / "chunks" / "000000.jsonl").write_bytes(chunk)
    owner = pc.RoventTrainingCorpus.from_submitted_files(root / "manifest.json",
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(), index_path=root / "index.json",
        index_sha256=hashlib.sha256(index_bytes).hexdigest())
    owner.admit_stage(recipe, manifest["stage_id"], manifest["model_epoch"], manifest["context_epoch"], [], [])
    return owner, row


def add_candidate_wdl(row):
    """Synthetic terminal metadata only; current R producer emits no WDL yet."""
    anchor = copy.deepcopy(row["bound"]["current_rules"]["anchor"])
    result = {"schema": "rz-pals-rules-outcome-evidence/1", "game_id": row["lineage"]["game_id"],
        "ending": "insufficient_material", "outcome": "draw", "rules_terminal": True}
    trajectory = {"game_id": result["game_id"], "ending": result["ending"], "outcome": result["outcome"],
        "anchors": [anchor], "result_evidence_sha256": data.digest(result["schema"], result)}
    replay = copy.deepcopy(row["bound"]["current_rules"]["replay"])
    raw_receipt = {"schema": corpus.WDL_DOMAIN, "kind": "rules_wdl",
        "native_input_sha256": row["input"]["native_input_sha256"], "semantic_input_sha256": row["input"]["semantic_input_sha256"],
        "anchor": anchor, "trajectory": trajectory, "rules_receipt": result,
        "rules_replay": {"schema": "rz-rovent-actual-rules-trajectory-replay/1", "initial": replay, "actual_selected_moves": [],
            "final_position": copy.deepcopy(replay), "completion": "automatic", "claim": None}}
    raw_sha = data.digest(corpus.WDL_DOMAIN, raw_receipt)
    row["wdl"] = {"raw_receipt": raw_receipt, "raw_receipt_sha256": raw_sha, "target": {"observed_sequence": 100,
        "question_id": "synthetic-timeline-not-original-authority", "view_sha256": row["bound"]["full_view_sha256"],
        "raw_evidence_sha256": raw_sha, "source": "rules_selfplay_trajectory", "game_id": result["game_id"], "anchor": anchor,
        "trajectory_membership_sha256": data.trajectory_digest(trajectory), "rules_result_sha256": trajectory["result_evidence_sha256"],
        "ending": result["ending"], "outcome": result["outcome"]}}
    seal(row)


def replace_input(row, value):
    original_json = raw(value).decode().rstrip("\n")
    bound = row["bound"]
    bound["context_sha256"] = ReferenceInput.from_mapping(value, config()).context_sha256
    row["input"] = {"reference_input_json": original_json, "reference_input_bytes": len(original_json.encode()),
        "reference_input_raw_sha256": h(original_json), "native_input_sha256": hashlib.sha256(wire.CAPTURE_INPUT_DOMAIN.encode() + b"\0" + original_json.encode()).hexdigest(),
        "semantic_input_sha256": wire.r_readout_f32_digest(wire.INPUT_DOMAIN,
            {key: bound[key] for key in ("model_semantics", "parameter_sha256", "collection_model_epoch", "collection_context_epoch")} | {"reference_input": value})}
    row["retirement"]["retained_input_bytes"] = len(original_json.encode())
    seal(row)


def add_context(row):
    move = 12 | (20 << 6)
    bound = row["bound"]
    bound["context_records"] = [{"local_id": index, "record_id": "record-" + str(index), "record_revision": 10 + index,
        "parent_revision": None if index == 0 else 10, "supersedes_revision": None, "origin_state": 1,
        "kind": index, "anchor": 0, "moves": [move], "original_first_move": move, "parent": None if index == 0 else 0,
        "supersedes": None, "source_state": 1, "source_revision": 1, "source_line_offset": 0,
        "record_perspective": "black", "encoded_perspective": "white", "completed_depth": 2,
        "score_scope": "CompletedIteration", "value": 7, "cpu_observation": None, "critical": True} for index in range(2)]
    bound["actual_tokens"] = 2
    bound["context_source_anchors"] = [copy.deepcopy(bound["context_anchors"][0]) for _ in range(2)]
    value = json.loads(row["input"]["reference_input_json"])
    value["context_lines"] = [{"id": record["local_id"], "kind": record["kind"], "original_first_move": corpus._move(move),
        "anchor_features": [768, 774, 783], "perspective": "white", "moves": [corpus._move(move)]} for record in bound["context_records"]]
    value["context_edges"] = [{"source": 0, "target": 1, "kind": 0}]
    replace_input(row, value)


class RCorpusTests(unittest.TestCase):
    def setUp(self):
        self.guard = NeuralImportsForbidden()
        sys.meta_path.insert(0, self.guard)

    def tearDown(self):
        sys.meta_path.remove(self.guard)

    def test_pinned_r_source_preserves_zero_attempt_and_separate_parameter_file_domains(self):
        with tempfile.TemporaryDirectory() as directory:
            owner = open_owner(Path(directory))
            sample = owner.training_sample(owner.rows()[0])
            self.assertNotEqual(sample["nnuec_model_sha256"], sample["nnuec_weights_file_sha256"])
            self.assertIsNone(sample["nnuec_wdl"])
            self.assertNotIn("role", sample)
            self.assertNotIn("targets", sample)
            self.assertEqual(owner._records[0]["question"]["readout_sequence"], 0)
            self.assertGreater(owner._records[0]["question"]["observed_tick_ns"], owner._records[0]["question"]["job"]["hard_deadline_ns"])
            report = owner.observation()
            self.assertFalse(report["execution_authorized"])
            self.assertFalse(report["actual_rules_replay_verified"])
            self.assertFalse(report["old_run_closure_proven"])
            self.assertEqual(report["occurrence_count"], 1)

    def test_repeated_semantic_inputs_keep_distinct_score_occurrences_and_one_unique_input(self):
        recipe, manifest, first, lock = fixture()
        second = copy.deepcopy(first)
        second["question"]["readout_sequence"] = 2
        for row in (first, second):
            row["retirement"].update(selected=2, retained_readouts=2,
                retained_input_bytes=2 * row["input"]["reference_input_bytes"])
            seal(row)
        with tempfile.TemporaryDirectory() as directory:
            owner = open_owner(Path(directory), source=(recipe, manifest, first, lock), rows=[first, second])
            self.assertEqual(len(owner.rows()), 2)
            self.assertEqual(owner.observation()["unique_input_count"], 1)
            self.assertEqual(len(owner.pool_metadata()["occurrences"]), 2)

    def test_same_original_capture_attempt_cannot_be_counted_twice(self):
        recipe, manifest, first, lock = fixture()
        second = copy.deepcopy(first)
        second["lineage"]["game_id"] = "another-raw-label"
        seal(second)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "duplicate original R capture/score attempt"):
                open_owner(Path(directory), source=(recipe, manifest, first, lock), rows=[first, second])

    def test_input_lock_raw_file_pin_and_canonical_digest_are_independent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recipe, manifest, row, lock = fixture()
            files = write(root, manifest, [row], lock)
            self.assertNotEqual(files.input_locks[0][0], files.input_locks[0][1].sha256)
            (root / "lock.json").write_bytes(raw(lock) + b" ")
            with self.assertRaisesRegex(ValueError, "bounded single-link regular file"):
                corpus.RoventRCorpus.from_files(files, recipe=recipe, nnuec_config=config(),
                    expected_nnuec_parameter_sha256=row["bound"]["parameter_sha256"])

    def test_descriptor_uses_domain_nul_canonical_and_rejects_generation_digest(self):
        source = fixture()
        row = source[2]
        row["root_descriptor_sha256"] = data.digest(corpus.DESCRIPTOR_DOMAIN, row["root_descriptor"])
        row["native_admission"]["root_descriptor_sha256"] = row["root_descriptor_sha256"]
        row["native_admission_sha256"] = data.digest(corpus.ROOT_DOMAIN, row["native_admission"])
        seal(row)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "domain-NUL"):
                open_owner(Path(directory), source=source)

    def test_runtime_endpoint_or_python_pin_cannot_replace_original_library_domain(self):
        for replacement in (h("runtime endpoint"), h("Python interpreter"), h("source aggregate")):
            with self.subTest(replacement=replacement):
                source = fixture()
                row = source[2]
                row["native_admission"]["native_origin"].update(runtime_sha256=replacement, runtime_library_sha256=replacement)
                row["native_admission"]["native_origin_sha256"] = data.digest(corpus.ORIGIN_DOMAIN, row["native_admission"]["native_origin"])
                row["native_admission_sha256"] = data.digest(corpus.ROOT_DOMAIN, row["native_admission"])
                seal(row)
                with tempfile.TemporaryDirectory() as directory:
                    with self.assertRaisesRegex(ValueError, "source/runtime/model domains"):
                        open_owner(Path(directory), source=source)

    def test_r_retirement_preserves_sampling_gaps_and_refuses_ordinal_relabel(self):
        for bad in (1, 3, True):
            source = fixture()
            source[2]["question"]["readout_sequence"] = bad
            seal(source[2])
            with tempfile.TemporaryDirectory() as directory:
                with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "sampling|unsigned"):
                    open_owner(Path(directory), source=source)

    def test_numeric_nonfinite_or_boolean_bits_are_rejected_before_sampling(self):
        for bits in (True, 0x7f800000, 0xff800000, 0x7fc00000, 2**32):
            source = fixture()
            source[2]["numeric"]["hidden_bits"] = [bits]
            seal(source[2])
            with tempfile.TemporaryDirectory() as directory:
                with self.subTest(bits=bits), self.assertRaisesRegex(ValueError, "finite|unsigned"):
                    open_owner(Path(directory), source=source)

    def test_v_free_unknown_cost_and_consumption_cannot_be_replaced_by_metadata_claims(self):
        for key, value in (("semantic_consumption", {"accepted": True}), ("whole_action_cost", {"wall_time_ns": 12})):
            source = fixture()
            source[2][key] = value
            seal(source[2])
            with tempfile.TemporaryDirectory() as directory:
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, "must remain unknown"):
                    open_owner(Path(directory), source=source)
        source = fixture()
        source[2]["utility"]["target_known_mask"][0] = True
        seal(source[2])
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "V-free R utility"):
                open_owner(Path(directory), source=source)

    def test_wrong_parameter_identity_cannot_be_substituted_with_raw_file_sha(self):
        recipe, manifest, row, lock = fixture()
        with tempfile.TemporaryDirectory() as directory:
            files = write(Path(directory), manifest, [row], lock)
            with self.assertRaisesRegex(ValueError, "parsed parameter identity/raw NNUEC"):
                corpus.RoventRCorpus.from_files(files, recipe=recipe, nnuec_config=config(),
                    expected_nnuec_parameter_sha256=row["bound"]["weights_file_sha256"])

    def test_strict_relative_paths_and_explicit_file_registration(self):
        with self.assertRaisesRegex(ValueError, "normalized relative"):
            corpus.RFilePin.from_mapping({"path": "../outside.json", "sha256": h("file"), "bytes": 1})
        with self.assertRaisesRegex(ValueError, "fields mismatch"):
            corpus.RCorpusFiles.from_mapping({"root": "/unused", "manifest": {}, "index": {}, "input_locks": {}, "authorized": True})

    def test_shared_read_failure_keeps_actual_charge_and_unknown_reservation(self):
        source = fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files = write(root, source[1], [source[2]], source[3])
            (root / "manifest.json").write_bytes(b"x" * files.manifest.bytes)
            budget = corpus.HistoricalReadBudget(64 * 1024 * 1024)
            with self.assertRaisesRegex(ValueError, "SHA|hash|digest|raw byte pin"):
                corpus.RoventRCorpus.from_files(files, recipe=source[0], nnuec_config=config(),
                    expected_nnuec_parameter_sha256=source[2]["bound"]["parameter_sha256"], read_accounting=budget)
            report = budget.observation()
            self.assertEqual(report["charged_bytes"], files.manifest.bytes)
            self.assertTrue(report["unknown"])

    def test_corrupted_or_foreign_owner_token_cannot_select_a_sample(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, second = open_owner(root / "one"), open_owner(root / "two")
            with self.assertRaisesRegex(ValueError, "same recovered R corpus token owner"):
                first.training_sample(second.rows()[0])
            first._records[0]["retirement"]["selected_input_preparation_ns"] += 1
            with self.assertRaisesRegex(ValueError, "changed or globally excluded"):
                first.training_sample(first.rows()[0])

    def test_validation_holdout_never_enters_optimizer(self):
        with tempfile.TemporaryDirectory() as directory:
            owner = open_owner(Path(directory))
            for split in ("validation", "holdout"):
                with self.subTest(split=split), self.assertRaisesRegex(ValueError, "cannot enter R optimization"):
                    owner.rows(split)
                self.assertEqual(len(owner.rows(split, optimization=False)), 0)

    def test_atomic_capture_publication_cannot_omit_retained_rows(self):
        source = fixture()
        source[2]["retirement"].update(selected=2, retained_readouts=2)
        seal(source[2])
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "atomic capture publication omits"):
                open_owner(Path(directory), source=source)

    def test_cross_split_repeated_input_excludes_both_without_erasing_occurrences(self):
        recipe, manifest, first, lock = fixture()
        second = copy.deepcopy(first)
        second["split"] = "holdout"
        second["lineage"].update(opening_id=next(key for key, split in recipe.to_dict()["opening_splits"].items() if split == "holdout"),
            game_id="held-game", line_genealogy_id="held-line")
        second["question"]["capture_sequence"] = 2
        second["retirement"]["capture_sequence"] = 2
        seal(second)
        with tempfile.TemporaryDirectory() as directory:
            owner = open_owner(Path(directory), source=(recipe, manifest, first, lock), rows=[first, second])
            self.assertEqual(len(owner.rows()), 0)
            self.assertEqual(len(owner.rows("holdout", optimization=False)), 0)
            self.assertEqual(owner.observation()["occurrence_count"], 2)
            self.assertEqual(owner.observation()["lineage"]["excluded_occurrence_count"], 2)
            self.assertEqual(owner.pool_metadata()["occurrences"], [])
            self.assertEqual(len(owner.all_occurrence_metadata()["occurrences"]), 2)
            self.assertNotEqual(owner.pool_sha256, owner.all_occurrences_sha256)

    def test_candidate_rules_wdl_is_not_reported_as_actual_timeline_authority(self):
        source = fixture()
        add_candidate_wdl(source[2])
        with tempfile.TemporaryDirectory() as directory:
            owner = open_owner(Path(directory), source=source)
            self.assertEqual(owner.training_sample(owner.rows()[0])["nnuec_wdl"], [0.0, 1.0, 0.0])
            report = owner.observation()
            self.assertEqual(report["candidate_wdl_train_unique_input_count"], 1)
            self.assertFalse(report["wdl_original_timeline_authority_verified"])
            self.assertFalse(report["actual_rules_replay_verified"])
            self.assertFalse(report["execution_authorized"])

    def test_wdl_cannot_broadcast_real_game_result_to_a_different_virtual_endpoint(self):
        source = fixture()
        add_candidate_wdl(source[2])
        evidence = source[2]["wdl"]
        foreign = copy.deepcopy(evidence["target"]["anchor"])
        foreign["rules_state_sha256"] = h("another virtual endpoint")
        evidence["target"]["anchor"] = foreign
        evidence["raw_receipt"]["anchor"] = foreign
        evidence["raw_receipt_sha256"] = data.digest(corpus.WDL_DOMAIN, evidence["raw_receipt"])
        evidence["target"]["raw_evidence_sha256"] = evidence["raw_receipt_sha256"]
        seal(source[2])
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "current-anchor Rules evidence"):
                open_owner(Path(directory), source=source)

    def test_metadata_duplicate_and_nonfinite_json_are_rejected_with_exact_raw_pins(self):
        for payload in (b'{"schema":"x","schema":"y"}', b'{"schema":NaN}', b'{"schema":Infinity}'):
            source = fixture()
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                files = write(root, source[1], [source[2]], source[3])
                (root / "manifest.json").write_bytes(payload)
                mapping = files.to_mapping()
                mapping["manifest"] = pin("manifest.json", payload)
                with self.subTest(payload=payload), self.assertRaisesRegex(ValueError, "duplicate JSON field|nonfinite corpus JSON"):
                    corpus.RoventRCorpus.from_files(corpus.RCorpusFiles.from_mapping(mapping), recipe=source[0], nnuec_config=config(),
                        expected_nnuec_parameter_sha256=source[2]["bound"]["parameter_sha256"])

    def test_explicit_aggregate_limit_is_not_silently_truncated(self):
        source = fixture()
        with tempfile.TemporaryDirectory() as directory:
            files = write(Path(directory), source[1], [source[2]], source[3])
            limit = files.manifest.bytes - 1
            limits = corpus.pc.CorpusLimits(max_file_bytes=limit, max_total_bytes=limit)
            with self.assertRaisesRegex(ValueError, "aggregate pinned byte limit"):
                corpus.RoventRCorpus.from_files(files, recipe=source[0], nnuec_config=config(), limits=limits,
                    expected_nnuec_parameter_sha256=source[2]["bound"]["parameter_sha256"])

    def test_raw_record_perspective_is_distinct_from_actual_anchor_encoding(self):
        source = fixture()
        add_context(source[2])
        with tempfile.TemporaryDirectory() as directory:
            owner = open_owner(Path(directory), source=source)
            sample = owner.training_sample(owner.rows()[0])
            self.assertEqual(sample["nnuec_input"]["context_lines"][0]["perspective"], "white")
            self.assertEqual(owner._records[0]["bound"]["context_records"][0]["record_perspective"], "black")
            self.assertFalse(owner.observation()["full_review_view_reconstructed"])

    def test_context_parent_revision_or_original_anchor_state_mismatch_is_rejected(self):
        for field, value in (("parent_revision", 9), ("origin_state", 2)):
            source = fixture()
            add_context(source[2])
            source[2]["bound"]["context_records"][1][field] = value
            seal(source[2])
            with tempfile.TemporaryDirectory() as directory:
                with self.subTest(field=field), self.assertRaisesRegex(ValueError, "original revision/order|original selected anchor state"):
                    open_owner(Path(directory), source=source)

    def test_original_auxiliary_evidence_preserves_native_semantic_and_zero_attempt_pins(self):
        with tempfile.TemporaryDirectory() as directory:
            owner = open_owner(Path(directory))
            sample = owner.training_sample(owner.rows()[0])
            reads_before = copy.deepcopy(owner.observation()["reads"])
            evidence = owner.original_auxiliary_evidence(sample["record_sha256"])
            self.assertNotEqual(evidence["native_input_sha256"], evidence["semantic_input_sha256"])
            self.assertEqual(evidence["semantic_input_sha256"], sample["input_sha256"])
            self.assertEqual(evidence["occurrence"]["readout_sequence"], 0)
            self.assertEqual(evidence["retirement_sha256"], sample["retirement_sha256"])
            self.assertIsNone(evidence["wdl_raw_sha256"])
            self.assertIs(evidence["execution_authorized"], False)
            self.assertNotIn("target_known", evidence)
            self.assertEqual(owner.observation()["reads"], reads_before)
            with self.assertRaisesRegex(ValueError, "absent from this owner"):
                owner.original_auxiliary_evidence(h("foreign record"))

    def test_empty_r_collection_is_zero_occurrences_without_fabricated_input_rows(self):
        recipe, manifest, row, _lock = fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_bytes = raw(manifest)
            index_bytes = raw({"schema": wire.INDEX_DOMAIN, "manifest_sha256": data.digest(wire.MANIFEST_DOMAIN, manifest),
                "record_count": 0, "chunks": [], "origins": []})
            (root / "manifest.json").write_bytes(manifest_bytes)
            (root / "index.json").write_bytes(index_bytes)
            files = corpus.RCorpusFiles.from_mapping({"root": str(root), "manifest": pin("manifest.json", manifest_bytes),
                "index": pin("index.json", index_bytes), "input_locks": {}})
            owner = corpus.RoventRCorpus.from_files(files, recipe=recipe, nnuec_config=config(),
                expected_nnuec_parameter_sha256=row["bound"]["parameter_sha256"])
            self.assertEqual(len(owner.rows()), 0)
            self.assertEqual(owner.observation()["occurrence_count"], 0)
            self.assertEqual(owner.observation()["candidate_wdl_train_unique_input_count"], 0)

    def test_sampling_pool_seals_actual_eligible_row_order_separately_from_raw_index_order(self):
        recipe, manifest, first, lock = fixture()
        second = copy.deepcopy(first)
        second["question"]["readout_sequence"] = 2
        for row in (first, second):
            row["retirement"].update(selected=2, retained_readouts=2, retained_input_bytes=2 * row["input"]["reference_input_bytes"])
            seal(row)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files = write(root, manifest, [first, second], lock)
            index = json.loads((root / "index.json").read_text(encoding="utf-8"))
            index["origins"].reverse()
            body = raw(index)
            (root / "index.json").write_bytes(body)
            mapping = files.to_mapping()
            mapping["index"] = pin("index.json", body)
            owner = corpus.RoventRCorpus.from_files(corpus.RCorpusFiles.from_mapping(mapping), recipe=recipe,
                nnuec_config=config(), expected_nnuec_parameter_sha256=first["bound"]["parameter_sha256"])
            pool, all_rows = owner.pool_metadata(), owner.all_occurrence_metadata()
            self.assertEqual([item["record_sha256"] for item in pool["occurrences"]], [first["record_sha256"], second["record_sha256"]])
            self.assertEqual([item["record_sha256"] for item in all_rows["occurrences"]], [second["record_sha256"], first["record_sha256"]])
            self.assertNotEqual(pool["pool_sha256"], all_rows["pool_sha256"])
            self.assertEqual(pool["pool_sha256"], data.digest(corpus.POOL_DOMAIN,
                {"index_sha256": owner.index_semantic_sha256, "occurrences": pool["occurrences"]}))

    def test_source_anchors_cannot_be_omitted_or_replaced_by_selected_anchor_count(self):
        source = fixture()
        add_context(source[2])
        source[2]["bound"]["context_source_anchors"].pop()
        seal(source[2])
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "source anchors must match selected record order/count"):
                open_owner(Path(directory), source=source)

    def test_submitted_pc_ancestor_uses_explicit_new_source_domain(self):
        recipe, manifest, row, lock = fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pc_owner, pc_row = submitted_pc_owner(root / "pc", recipe, manifest["collection"], lock)
            manifest["pc_ancestor_index_sha256"] = [pc_owner.index_semantic_sha256]
            row["manifest_sha256"] = data.digest(wire.MANIFEST_DOMAIN, manifest)
            seal(row)
            owner = open_owner(root / "r", source=(recipe, manifest, row, lock), pc_corpora=(pc_owner,))
            occurrences = owner._union.all_occurrences()
            actual = next(value for value in occurrences if value.record_sha256 == pc_row["record_sha256"])
            self.assertEqual(actual.source.source_domain, "rz-rovent-training-record/2")
            self.assertFalse(any(value.source.source_domain == "rz-rovent-training-record/1" for value in occurrences))
            self.assertEqual(len(owner.rows()), 1)
            self.assertFalse(owner.observation()["execution_authorized"])

    def test_r_causal_genealogy_resolves_exact_submitted_pc_domain(self):
        recipe, manifest, row, lock = fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pc_owner, pc_row = submitted_pc_owner(root / "pc", recipe, manifest["collection"], lock)
            manifest["pc_ancestor_index_sha256"] = [pc_owner.index_semantic_sha256]
            row["lineage"]["pc_genealogy"] = {key: pc_row[key] for key in ("record_sha256", "input_sha256", "origin_sha256")}
            row["manifest_sha256"] = data.digest(wire.MANIFEST_DOMAIN, manifest)
            seal(row)
            owner = open_owner(root / "r", source=(recipe, manifest, row, lock), pc_corpora=(pc_owner,))
            occurrence = next(value for value in owner._union.all_occurrences() if value.source.source_domain == wire.RECORD_DOMAIN)
            self.assertEqual([(source.source_domain, source.origin_sha256) for source in occurrence.ancestor_origins],
                             [("rz-rovent-training-record/2", pc_row["origin_sha256"])])

    def test_original_r_manifest_pc_index_set_accepts_both_pc_wire_families(self):
        from rz_pals_model import rovent_corpus as pc
        from test_rovent_corpus import original_fixture, seal_record, reseal_native, write_corpus
        recipe, manifest, row, lock = fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            new_owner, new_row = submitted_pc_owner(root / "new", recipe, manifest["collection"], lock)
            _old_recipe, old_manifest, old_row = original_fixture()
            for key in ("source_commit", "source_sha256", "runtime_sha256", "model_semantics_sha256", "recipe_sha256",
                        "model_epoch", "context_epoch"):
                old_manifest[key] = manifest["collection"][key]
            old_row["input"].update(model_epoch=old_manifest["model_epoch"], context_epoch=old_manifest["context_epoch"])
            old_row["input_sha256"] = data.seal_input(old_row["input"])
            old_row["native"]["source_sha256"] = old_manifest["source_sha256"]
            for name in ("raw_submission", "raw_completion", "raw_retirement"):
                old_row["native"][name].update(source_sha256=old_manifest["source_sha256"], input_sha256=old_row["input_sha256"])
            reseal_native(old_row)
            seal_record(old_manifest, old_row)
            (root / "old").mkdir()
            old_owner = write_corpus(root / "old", old_manifest, [old_row])
            manifest["pc_ancestor_index_sha256"] = sorted([old_owner.index_semantic_sha256, new_owner.index_semantic_sha256])
            row["manifest_sha256"] = data.digest(wire.MANIFEST_DOMAIN, manifest)
            seal(row)
            owner = open_owner(root / "r", source=(recipe, manifest, row, lock), pc_corpora=(new_owner, old_owner))
            sources = {value.record_sha256: value.source.source_domain for value in owner._union.all_occurrences()}
            self.assertEqual(sources[old_row["record_sha256"]], "rz-rovent-training-record/1")
            self.assertEqual(sources[new_row["record_sha256"]], "rz-rovent-training-record/2")
            self.assertEqual(len(owner._union.all_occurrences()), 3)
            self.assertEqual(len(owner.rows()), 1)


if __name__ == "__main__":
    unittest.main()
