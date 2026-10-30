"""Original corpus intake contracts; no model import, update or inference."""
import copy
import hashlib
import importlib.abc
import json
from pathlib import Path
import sys
import tempfile
import unittest


class NeuralImportsForbidden(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"torch", "numpy", "onnx", "onnxruntime"}:
            raise AssertionError("neural execution forbidden in original corpus contract tests")


_guard = NeuralImportsForbidden()
sys.meta_path.insert(0, _guard)
try:
    from rz_pals_model import rovent_corpus as corpus
    from rz_pals_model import generation_data as data
    from rz_pals_model.virtual_review import input_digest
    from test_generation_contract import row as legacy_shape_fixture
    from test_rovent_generation import recipe_value
finally:
    sys.meta_path.remove(_guard)


def raw_pin(root, raw, name="source.json"):
    path = root / name
    path.write_bytes(raw)
    return path, hashlib.sha256(raw).hexdigest(), len(raw)


def h(text):
    return hashlib.sha256(text.encode()).hexdigest()


def original_fixture(*, opening="train-a", request_sequence=1):
    """Synthetic wire fixture; never original producer custody or Rules proof."""
    recipe = corpus.SeriesRecipe(recipe_value())
    declarations = recipe.to_dict()
    teachers = {actor.actor_id: {"engine": actor.family, "source_commit": actor.source_revision,
        "binary_sha256": actor.binary_sha256, "binary_bytes": 1, "profile_sha256": actor.profile_sha256,
        "canonical_sha256": h(actor.actor_id + "canonical"), "capability_sha256": h(actor.actor_id + "capability"),
        "options": {}, "assets": {}} for actor in recipe.actors}
    manifest = {"schema": corpus.MANIFEST_DOMAIN, "corpus_id": "synthetic-contract-only", "source_commit": "1" * 40,
        "source_sha256": declarations["source_sha256"], "runtime_sha256": h("runtime"),
        "model_semantics_sha256": declarations["model_semantics_sha256"], "model_epoch": 1, "context_epoch": 1,
        "recipe_sha256": recipe.sha256, "stage_id": "common", "round": 0, "round_id": "common-round",
        "student": declarations["student_id"], "learner": declarations["learner_id"], "one_learner": True,
        "inherited_input_sha256": [], "inherited_origin_sha256": [], "teachers": teachers,
        "opening_splits": declarations["opening_splits"], "ancestor_index_sha256": []}
    snapshot = legacy_shape_fixture()["input"]
    snapshot.update(opening_id=opening, model_epoch=1, context_epoch=1, capture_sequence=10 * request_sequence,
                    worker_id="native-owner-1", job_id="job-" + str(request_sequence), slice_id="slice-" + str(request_sequence))
    input_seal = data.seal_input(snapshot)
    representation = {"input": input_digest(snapshot["encoded_model_input"]), "history": h("endpoint-history"),
        "candidates": h("candidates"), "mask_positions": h("masks"), "model": h("model"), "encoding": h("encoding"),
        "precision": "Fp32", "frozen_epoch": 1, "record_revision": 0, "role": "proposer"}
    ticket = {"process_epoch": 1, "game_generation": 1, "root_generation": 1, "request_sequence": request_sequence,
        "implementation_sha256": h("implementation"), "question_digest": snapshot["question_id"], "view_sequence": 1,
        "view_digest": h("native-source-view-distinct-from-generation-view"),
        "context_policy": snapshot["selected_view"]["context_policy_sha256"], "actual_tokens": 0,
        "selected_records": 0, "anchors": 1, "length_bucket": 8,
        "root_state": {"owner": 1, "revision": 1, "semantic": h("root-state")}, "representation": representation,
        "deadline_clock": 1, "deadline_tick": 1000}
    native = {"head": "proposer", "source_sha256": manifest["source_sha256"], "model_sha256": representation["model"],
        "encoding_sha256": representation["encoding"], "input_digest": representation["input"], "view_digest": ticket["view_digest"],
        "output_digest": "", "process_epoch": 1, "game_generation": 1, "root_generation": 1,
        "request_sequence": request_sequence, "execution_epoch": 1, "execution_sequence": request_sequence,
        "completed_nn_inputs": 1, "submitted_sequence": snapshot["capture_sequence"],
        "completed_sequence": snapshot["capture_sequence"] + 1, "retired_sequence": snapshot["capture_sequence"] + 2,
        "owner_run_invoked": True, "physical_confirmed": True, "accepted_retired": True}
    logits = [0, 0]
    wdl_bits = [0x3E800000, 0x3F000000, 0x3E800000]
    framed = (b"rz-pals-accepted-proposer-head/1\0" + (2).to_bytes(8, "little")
              + b"".join(value.to_bytes(4, "little") for value in logits + wdl_bits))
    native["output_digest"] = hashlib.sha256(framed).hexdigest()
    common = {key: native[key] for key in ("process_epoch", "game_generation", "root_generation", "request_sequence", "head", "source_sha256")}
    common.update(schema=corpus.NATIVE_RAW_DOMAIN, input_sha256=input_seal, ticket=copy.deepcopy(ticket))
    native["raw_submission"] = {**copy.deepcopy(common), "kind": "submitted", "sequence": native["submitted_sequence"],
        **{key: native[key] for key in ("model_sha256", "encoding_sha256", "input_digest", "view_digest")}}
    native["raw_submission_sha256"] = data.digest(corpus.NATIVE_RAW_DOMAIN, native["raw_submission"])
    output = {"request_epoch": 1, "request_sequence": request_sequence, "game_generation": 1, "root_generation": 1,
        "implementation_sha256": ticket["implementation_sha256"], "representation": copy.deepcopy(representation),
        "payload": {"kind": "proposal", "candidate_logits_bits": logits, "wdl_bits": wdl_bits}, "private_latent_bits": [0] * 6144}
    native["raw_completion"] = {**copy.deepcopy(common), "kind": "completed", "sequence": native["completed_sequence"],
        "submitted_sha256": native["raw_submission_sha256"], "execution_epoch": 1, "execution_sequence": request_sequence,
        "completed_nn_inputs": 1, "physical_confirmed": True, "owner_run_invoked": True,
        "measurement_complete": True, "physical_fresh": True, "output_digest": native["output_digest"],
        "owner_service_elapsed_ns": 100, "output": output}
    native["raw_completion_sha256"] = data.digest(corpus.NATIVE_RAW_DOMAIN, native["raw_completion"])
    native["raw_retirement"] = {**copy.deepcopy(common), "kind": "retired", "sequence": native["retired_sequence"],
        "submitted_sha256": native["raw_submission_sha256"], "completed_sha256": native["raw_completion_sha256"],
        "output_digest": native["output_digest"], "accepted_retired": True}
    native["raw_retirement_sha256"] = data.digest(corpus.NATIVE_RAW_DOMAIN, native["raw_retirement"])
    record = {"schema": corpus.RECORD_DOMAIN, "role": "proposer", "split": declarations["opening_splits"][opening],
        "input": snapshot, "input_sha256": input_seal, "origin_sha256": "", "record_sha256": "",
        "lineage": {"actor_id": "student-white", "teacher_actor_id": None, "ancestor_origin_sha256": []}, "native": native,
        "head_sources": {}, "targets": dict.fromkeys(("policy", "wdl", "divergence", "task"), None),
        "divergence_index": None, "nnuec_input": None, "nnuec_input_sha256": None, "nnuec_model_sha256": None,
        "nnuec_semantics": None, "nnuec_wdl": None}
    seal_record(manifest, record)
    return recipe, manifest, record


def seal_record(manifest, record):
    snapshot, lineage = record["input"], record["lineage"]
    origin = {key: manifest[key] for key in ("source_sha256", "recipe_sha256", "stage_id", "round", "round_id", "student", "learner")}
    origin.update(lineage)
    origin.update(input_sha256=record["input_sha256"], **{key: snapshot[key] for key in
        ("game_id", "opening_id", "line_genealogy_id", "transposition_sha256")})
    record["origin_sha256"] = data.digest(corpus.ORIGIN_DOMAIN, origin)
    projected = {**record, "record_sha256": "", "input": {"sealed_input_sha256": record["input_sha256"]}}
    record["record_sha256"] = data.digest(corpus.RECORD_DOMAIN, projected)


def add_rules_wdl(record):
    snapshot = record["input"]
    outcome = {"schema": "rz-pals-rules-outcome-evidence/1", "game_id": snapshot["game_id"], "ending": "checkmate",
               "outcome": "white_win", "rules_terminal": True}
    outcome_sha = data.digest(outcome["schema"], outcome)
    trajectory = {"game_id": snapshot["game_id"], "ending": "checkmate", "outcome": "white_win",
                  "anchors": [copy.deepcopy(snapshot["rules_anchor"])], "result_evidence_sha256": outcome_sha}
    raw = {"schema": corpus.HEAD_DOMAIN, "kind": "rules_wdl", "input_sha256": record["input_sha256"],
           "anchor": copy.deepcopy(snapshot["rules_anchor"]), "trajectory": trajectory, "rules_receipt": outcome}
    raw_sha = data.digest(corpus.HEAD_DOMAIN, raw)
    label = {"observed_sequence": snapshot["capture_sequence"] + 3, "question_id": snapshot["question_id"],
        "view_sha256": snapshot["view_sha256"], "raw_evidence_sha256": raw_sha, "source": "rules_selfplay_trajectory",
        "game_id": snapshot["game_id"], "anchor": copy.deepcopy(snapshot["rules_anchor"]),
        "trajectory_membership_sha256": data.trajectory_digest(trajectory), "rules_result_sha256": outcome_sha,
        "ending": "checkmate", "outcome": "white_win"}
    record["targets"]["wdl"] = label
    record["head_sources"]["wdl"] = {"actor_id": record["lineage"]["actor_id"], "source": label["source"],
        "binary_sha256": None, "profile_sha256": None, "raw_receipt": raw, "raw_receipt_sha256": raw_sha,
        "observed_sequence": label["observed_sequence"], "anchor": copy.deepcopy(snapshot["rules_anchor"])}


def add_teacher_policy(manifest, record):
    snapshot, actor = record["input"], "stockfish"
    pin = manifest["teachers"][actor]
    record["lineage"]["teacher_actor_id"] = actor
    identity = {"adapter_semantics": "external-uci-checker-v1", "binary_sha256": pin["binary_sha256"],
        "launch_arguments_sha256": h("launch"), "declared_name": "Stockfish 19 schema fixture", "declared_version": "19",
        "declared_source": "https://github.com/official-stockfish/Stockfish/tree/sf_19", "declared_license": "GPL-3.0",
        "options": {}, "assets": [], "model_metadata": {"weights_sha256": None, "training": "unknown",
        "declared_rights": None, "precision": None}}
    constraint = {"kind": "combined", "depth": 8, "reported_nodes": 65536}
    ranked = [128, 64]
    raw_report = {"identity": identity, "observed_uci": {"name": identity["declared_name"], "author": "schema fixture"},
        "request_id": 1, "best_move": 128, "pv": [128], "score": {"kind": "mate_moves", "value": 3},
        "bound": "ExactReported", "wdl_per_mille": None, "perspective": "SideToMove", "requested_depth": 8,
        "reported_depth": 8, "seldepth": 8, "root_restricted": False, "completion": "BestMove",
        "work": {"nodes": 42, "qnodes": None, "tt_hits": None}, "elapsed_ns": 1000}
    ticket = record["native"]["raw_submission"]["ticket"]
    question = {"process_epoch": 1, "game_generation": 1, "root_generation": 1,
        "implementation_sha256": ticket["implementation_sha256"], "sequence": 1, "parent": None,
        "sender": "Proposer", "recipient": "Researcher", "purpose": "TacticalReview",
        "anchor": {"owner": 1, "revision": 1, "semantic": snapshot["rules_anchor"]["rules_state_sha256"]},
        "anchor_root_ply": 0, "original_first_move": None, "prefix_digest": h("empty-prefix"), "line_digest": None,
        "line_offset": 0, "question_digest": h("external-question"), "evidence_revision": 0, "root_budget": h("root-budget"),
        "deadline_clock": 1, "deadline_tick": 1000, "view": {key: ticket[key] for key in
            ("view_digest", "view_sequence", "context_policy", "actual_tokens", "selected_records", "anchors", "length_bucket")}}
    body = {"schema": corpus.HEAD_DOMAIN, "kind": "teacher_policy", "actor_id": actor,
        "input_sha256": record["input_sha256"], "anchor": copy.deepcopy(snapshot["rules_anchor"]), "teacher_pin": copy.deepcopy(pin),
        "binding": {"question_sequence": 1, "question_digest": h("external-question"), "root_budget": h("root-budget"),
            "root_restriction": hashlib.sha256(b"rz-external-root-restriction/1\0\0").hexdigest(),
            "execution_epoch": 1, "execution_sequence": 1, "owner": 0,
            "process": {"pid": 123, "process_group": 123, "proc_start_ticks": 100}, "requested": copy.deepcopy(constraint),
            "question": question, "position_anchor": copy.deepcopy(snapshot["rules_anchor"]), "root_moves": None,
            "mode": "HybridPcr", "soft_quantum_ns": 8000000, "context_token_budget": 512},
        "completion": {"profile": "stockfish19_embedded_nnue", "purpose": "teacher", "issued_go": "go depth 8 nodes 65536",
            "requested": copy.deepcopy(constraint), "issued": copy.deepcopy(constraint), "readiness_barrier_completed": True,
            "go_issued": True, "raw": raw_report,
            "opinions": [{"rank": rank, "depth": 8, "seldepth": 8, "pv": [move],
                "score": {"kind": "mate_moves", "value": 3}, "bound": "ExactReported", "reported_wdl": None}
                for rank, move in enumerate(ranked, 1)], "requested_multipv": 2, "conditional_policy_target_known": True,
            "option_application_observed": None, "model_loading_observed": None, "stop_control": None,
            "attempt": {"elapsed_ns": 1100, "work": copy.deepcopy(raw_report["work"]), "request_id": 1,
                "partial_report_matches": True, "process": {"identity": {"pid": 123, "process_group": 123, "proc_start_ticks": 100},
                    "stop_sent": False, "quit_sent": False, "exit_observed": False, "stdout_drained": False,
                    "stderr_drained": False, "exit_code": None, "exit_signal": None, "cleanup_complete": False,
                    "quarantined": False, "ownership_lost": False, "stdout_bytes": 200, "stderr_bytes": 0}},
            "timing": {"go_issued": True, "stop_sent": False, "bestmove_drained": True, "ready_completed": True}}}
    raw_sha = data.digest(corpus.HEAD_DOMAIN, body)
    label = {"observed_sequence": snapshot["capture_sequence"] + 3, "question_id": snapshot["question_id"],
        "view_sha256": snapshot["view_sha256"], "raw_evidence_sha256": raw_sha, "source": "rovent_completed_teacher_multipv",
        "completed": True, "stale": False, "anchor": copy.deepcopy(snapshot["rules_anchor"]), "ranked_moves": ranked,
        "teacher_binary_sha256": pin["binary_sha256"], "teacher_profile_sha256": pin["profile_sha256"],
        "go_sequence": 1, "completed_depth": 8}
    record["targets"]["policy"] = label
    record["head_sources"]["policy"] = {"actor_id": actor, "source": label["source"], "binary_sha256": pin["binary_sha256"],
        "profile_sha256": pin["profile_sha256"], "raw_receipt": body, "raw_receipt_sha256": raw_sha,
        "observed_sequence": label["observed_sequence"], "anchor": copy.deepcopy(snapshot["rules_anchor"])}


def reseal_head(record, head):
    source = record["head_sources"][head]
    identity = data.digest(corpus.HEAD_DOMAIN, source["raw_receipt"])
    source["raw_receipt_sha256"] = identity
    record["targets"][head]["raw_evidence_sha256"] = identity


def reseal_native(record):
    native = record["native"]
    native["raw_submission_sha256"] = data.digest(corpus.NATIVE_RAW_DOMAIN, native["raw_submission"])
    native["raw_completion"]["submitted_sha256"] = native["raw_submission_sha256"]
    native["raw_completion_sha256"] = data.digest(corpus.NATIVE_RAW_DOMAIN, native["raw_completion"])
    native["raw_retirement"].update(submitted_sha256=native["raw_submission_sha256"], completed_sha256=native["raw_completion_sha256"])
    native["raw_retirement_sha256"] = data.digest(corpus.NATIVE_RAW_DOMAIN, native["raw_retirement"])


def write_corpus(root, manifest, records, *, ancestors=()):
    origins = []
    for record in records:
        origin = {key: record[key] for key in ("record_sha256", "input_sha256", "origin_sha256", "split")}
        origin.update({key: record["input"][key] for key in ("game_id", "opening_id", "line_genealogy_id", "transposition_sha256")})
        origin.update(copy.deepcopy(record["lineage"]))
        origins.append(origin)
    raw = lambda value: json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode() + b"\n"
    manifest_path, manifest_sha, _ = raw_pin(root, raw(manifest), "manifest.json")
    (root / "chunks").mkdir()
    chunk = b"".join(raw(record) for record in records)
    raw_pin(root, chunk, "chunks/000000.jsonl")
    index = {"schema": corpus.INDEX_DOMAIN, "manifest_sha256": data.digest(corpus.MANIFEST_DOMAIN, manifest),
        "record_count": len(records), "chunks": [{"path": "chunks/000000.jsonl", "bytes": len(chunk),
        "sha256": hashlib.sha256(chunk).hexdigest(), "records": len(records)}], "origins": origins}
    index_path, index_sha, _ = raw_pin(root, raw(index), "index.json")
    return corpus.RoventTrainingCorpus.from_files(manifest_path, manifest_sha256=manifest_sha,
        index_path=index_path, index_sha256=index_sha, ancestors=ancestors)


def parent_and_current(root):
    """Distinct stage origins intentionally sharing one exact input seal."""
    recipe, parent_manifest, parent_record = original_fixture()
    parent_root, current_root = root / "parent", root / "current"
    parent_root.mkdir()
    current_root.mkdir()
    parent = write_corpus(parent_root, parent_manifest, [parent_record])
    manifest, record = copy.deepcopy(parent_manifest), copy.deepcopy(parent_record)
    manifest.update(corpus_id="same-stage-resume-contract-only", stage_id="sf-ft", round=1, round_id="sf-ft-round",
        teachers={"stockfish": manifest["teachers"]["stockfish"]},
        inherited_input_sha256=[parent_record["input_sha256"]], inherited_origin_sha256=[parent_record["origin_sha256"]],
        ancestor_index_sha256=[parent.index_semantic_sha256])
    record["lineage"]["ancestor_origin_sha256"] = [parent_record["origin_sha256"]]
    seal_record(manifest, record)
    current = write_corpus(current_root, manifest, [record], ancestors=(parent,))
    return recipe, manifest, parent_record, record, current


class OriginalCorpusFileTests(unittest.TestCase):
    def setUp(self):
        self.guard = NeuralImportsForbidden()
        sys.meta_path.insert(0, self.guard)

    def tearDown(self):
        sys.meta_path.remove(self.guard)

    def test_original_bytes_pin_precedes_strict_decode_and_never_imports_neural_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, identity, size = raw_pin(root, b'{"record":1}\n')
            self.assertEqual(corpus._read_pinned_json(path, identity, size, size), {"record": 1})
            with self.assertRaisesRegex(ValueError, "byte pin/identity mismatch"):
                corpus._read_pinned_json(path, "0" * 64, size, size)

    def test_duplicate_and_nonfinite_fields_reject_the_actual_decode_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for raw, message in ((b'{"head":1,"head":2}', "duplicate JSON field"),
                                 (b'{"head":NaN}', "nonfinite corpus JSON")):
                path, identity, size = raw_pin(root, raw)
                with self.assertRaisesRegex(ValueError, message):
                    corpus._read_pinned_json(path, identity, size, size)

    def test_bounded_file_and_declared_length_reject_before_a_large_input_is_materialized(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, identity, size = raw_pin(root, b'{"record":1}')
            with self.assertRaisesRegex(ValueError, "bounded unsigned integer"):
                corpus._read_pinned_json(path, identity, size, size - 1)
            with self.assertRaisesRegex(ValueError, "single-link regular file"):
                corpus._read_pinned_json(path, identity, size - 1, size)

    def test_explicit_json_in_a_managed_checkpoint_directory_is_allowed_without_folder_scanning(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "checkpoints"
            root.mkdir()
            path, identity, size = raw_pin(root, b'{"schema":"source-metadata"}')
            self.assertEqual(corpus._read_pinned_json(path, identity, size, size)["schema"], "source-metadata")

    def test_secret_source_and_duplicate_hardlink_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, identity, size = raw_pin(root, b'{}', "credentials.json")
            with self.assertRaisesRegex(ValueError, "secret path"):
                corpus._read_pinned_json(path, identity, size, size)
            path, identity, size = raw_pin(root, b'{}', "source.json")
            second = root / "alias.json"
            try:
                second.hardlink_to(path)
            except (OSError, NotImplementedError):
                self.skipTest("host does not support hardlinks")
            with self.assertRaisesRegex(ValueError, "single-link regular file"):
                corpus._read_pinned_json(path, identity, size, size)

    def test_limits_require_positive_bounded_distinct_source_counts(self):
        self.assertEqual(corpus.CorpusLimits(max_file_bytes=128, max_total_bytes=256).max_file_bytes, 128)
        with self.assertRaisesRegex(ValueError, "positive bounded corpus limits"):
            corpus.CorpusLimits(max_records=0)
        with self.assertRaisesRegex(ValueError, "per-file limit"):
            corpus.CorpusLimits(max_file_bytes=256, max_total_bytes=128)


class OriginalCorpusOwnerTests(unittest.TestCase):
    def setUp(self):
        self.guard = NeuralImportsForbidden()
        sys.meta_path.insert(0, self.guard)

    def tearDown(self):
        sys.meta_path.remove(self.guard)

    def test_accepted_wire_fixture_yields_a_row_without_live_authority_or_actual_training_claims(self):
        recipe, manifest, record = original_fixture()
        add_rules_wdl(record)
        seal_record(manifest, record)
        with tempfile.TemporaryDirectory() as directory:
            owner = write_corpus(Path(directory), manifest, [record])
            report = owner.admit_stage(recipe, "common", 1, 1, [], [])
            self.assertFalse(report["execution_authorized"])
            self.assertFalse(report["actual_training_admissible"])
            self.assertFalse(report["original_producer_authority_present"])
            self.assertEqual(report["optimizer_updates"], 0)
            rows = owner.rows("proposer")
            self.assertEqual(len(rows), 1)
            sample = owner.training_sample(rows[0])
            self.assertEqual(sample["targets"]["wdl"], [1.0, 0.0, 0.0])
            self.assertEqual(sample["native_retirement_sha256"], record["native"]["raw_retirement_sha256"])
            self.assertEqual(sample["head_raw_sha256"], {"wdl": record["head_sources"]["wdl"]["raw_receipt_sha256"]})
            sample["input"]["game_id"] = "caller-edited-copy"
            self.assertEqual(owner.training_sample(rows[:1][0])["input"]["game_id"], record["input"]["game_id"])

    def test_teacher_mate_report_only_supervises_rank_policy_and_keeps_wdl_unknown(self):
        recipe, manifest, record = original_fixture()
        add_teacher_policy(manifest, record)
        seal_record(manifest, record)
        with tempfile.TemporaryDirectory() as directory:
            owner = write_corpus(Path(directory), manifest, [record])
            owner.admit_stage(recipe, "common", 1, 1, [], [])
            sample = owner.training_sample(owner.rows("proposer")[0])
            self.assertGreater(sample["targets"]["policy"][1], sample["targets"]["policy"][0])
            self.assertIsNone(sample["targets"]["wdl"])
            self.assertIsNone(sample["nnuec_wdl"])
            self.assertEqual(sample["head_sources"]["policy"]["raw_receipt"]["completion"]["raw"]["score"],
                             {"kind": "mate_moves", "value": 3})

    def test_teacher_relative_timing_retains_strict_original_observations(self):
        recipe, manifest, record = original_fixture()
        add_teacher_policy(manifest, record)
        timing = record["head_sources"]["policy"]["raw_receipt"]["completion"]["timing"]
        timing["durations_ns"] = {name: None for name in (
            "admitted_to_run_started", "run_started_to_go_issued", "go_issued_to_stop_sent",
            "go_issued_to_bestmove_drained", "stop_sent_to_bestmove_drained",
            "bestmove_drained_to_ready_completed", "ready_completed_to_returned", "admitted_to_returned")}
        timing["durations_ns"]["go_issued_to_bestmove_drained"] = 10
        reseal_head(record, "policy")
        seal_record(manifest, record)
        with tempfile.TemporaryDirectory() as directory:
            owner = write_corpus(Path(directory), manifest, [record])
            owner.admit_stage(recipe, "common", 1, 1, [], [])
            self.assertIsNotNone(owner.training_sample(owner.rows("proposer")[0])["targets"]["policy"])
        for name, value, error in (("go_issued_to_stop_sent", 1, "duration without observed stop"),
                                   ("admitted_to_returned", -1, "invalid bounded unsigned integer")):
            changed = copy.deepcopy(record)
            changed["head_sources"]["policy"]["raw_receipt"]["completion"]["timing"]["durations_ns"][name] = value
            reseal_head(changed, "policy")
            seal_record(manifest, changed)
            with tempfile.TemporaryDirectory() as directory, self.assertRaisesRegex(ValueError, error):
                write_corpus(Path(directory), manifest, [changed])

    def test_native_bits_must_match_the_original_accepted_head_even_after_file_and_record_resealing(self):
        _, manifest, record = original_fixture()
        record["native"]["raw_completion"]["output"]["payload"]["candidate_logits_bits"][0] = 0x3F800000
        reseal_native(record)
        seal_record(manifest, record)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "raw output differs from accepted head digest"):
                write_corpus(Path(directory), manifest, [record])

    def test_readiness_is_checked_from_raw_completion_after_all_envelope_pins_are_resealed(self):
        _, manifest, record = original_fixture()
        add_teacher_policy(manifest, record)
        record["head_sources"]["policy"]["raw_receipt"]["completion"]["readiness_barrier_completed"] = False
        reseal_head(record, "policy")
        seal_record(manifest, record)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "request/go/readiness/profile completion was not joined"):
                write_corpus(Path(directory), manifest, [record])

    def test_actual_rules_anchor_is_required_in_the_recovered_trajectory(self):
        _, manifest, record = original_fixture()
        add_rules_wdl(record)
        raw = record["head_sources"]["wdl"]["raw_receipt"]
        raw["trajectory"]["anchors"][0]["rules_state_sha256"] = h("other-state")
        record["targets"]["wdl"]["trajectory_membership_sha256"] = data.trajectory_digest(raw["trajectory"])
        reseal_head(record, "wdl")
        seal_record(manifest, record)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "exact actual Rules trajectory anchor/result"):
                write_corpus(Path(directory), manifest, [record])

    def test_cross_split_game_and_transposition_are_excluded_symmetrically(self):
        recipe, manifest, first = original_fixture()
        _, _, held = original_fixture(opening="held", request_sequence=2)
        with tempfile.TemporaryDirectory() as directory:
            owner = write_corpus(Path(directory), manifest, [first, held])
            report = owner.admit_stage(recipe, "common", 1, 1, [], [])
            self.assertGreater(report["cross_split_identities"], 0)
            self.assertEqual(len(owner.rows("proposer")), 0)
            self.assertEqual(len(owner.rows("proposer", "holdout", optimization=False)), 0)

    def test_holdout_rows_cannot_be_optimization_rows(self):
        recipe, manifest, record = original_fixture(opening="held")
        with tempfile.TemporaryDirectory() as directory:
            owner = write_corpus(Path(directory), manifest, [record])
            owner.admit_stage(recipe, "common", 1, 1, [], [])
            with self.assertRaisesRegex(ValueError, "holdout cannot enter optimization"):
                owner.rows("proposer", "holdout")
            held = owner.rows("proposer", "holdout", optimization=False)
            self.assertEqual(owner.training_sample(held[0])["split"], "holdout")

    def test_rows_are_bound_to_their_original_admitted_owner(self):
        recipe, manifest, record = original_fixture()
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            first = write_corpus(Path(a), manifest, [record])
            second = write_corpus(Path(b), manifest, [record])
            first.admit_stage(recipe, "common", 1, 1, [], [])
            second.admit_stage(recipe, "common", 1, 1, [], [])
            with self.assertRaisesRegex(ValueError, "same admitted original corpus row owner"):
                second.training_sample(first.rows("proposer")[0])

    def test_same_stage_retained_rows_require_checkpoint_authority_without_changing_frozen_base(self):
        recipe, manifest, record = original_fixture()
        before = copy.deepcopy(manifest)
        with tempfile.TemporaryDirectory() as directory:
            owner = write_corpus(Path(directory), manifest, [record])
            report = owner.admit_stage(recipe, "common", 1, 1, [record["input_sha256"]], [record["origin_sha256"]])
            self.assertEqual(report["base_inherited_input_sha256"], [])
            self.assertEqual(report["base_inherited_origin_sha256"], [])
            self.assertEqual(report["same_stage_retained_input_sha256"], [record["input_sha256"]])
            self.assertEqual(report["same_stage_retained_origin_sha256"], [record["origin_sha256"]])
            requirement = report["same_stage_checkpoint_authority"]
            self.assertEqual(requirement["schema"], "rz-rovent-same-stage-checkpoint-authority-requirement/1")
            self.assertEqual(requirement["kind"], "original_checkpoint_authority_required")
            self.assertTrue(requirement["required"])
            self.assertFalse(requirement["verified"])
            self.assertFalse(requirement["execution_authorized"])
            self.assertFalse(report["execution_authorized"])
            self.assertFalse(report["actual_training_admissible"])
            self.assertEqual(manifest, before)
            with self.assertRaisesRegex(ValueError, "retained original origin/input seal projection mismatch"):
                owner.admit_stage(recipe, "common", 1, 1, [h("unrelated-input")], [record["origin_sha256"]])
            self.assertEqual(len(owner.rows("proposer")), 1)

    def test_base_and_same_stage_input_projections_may_overlap_without_erasing_either_origin(self):
        with tempfile.TemporaryDirectory() as directory:
            recipe, manifest, parent_record, record, owner = parent_and_current(Path(directory))
            self.assertEqual(parent_record["input_sha256"], record["input_sha256"])
            origins = sorted([parent_record["origin_sha256"], record["origin_sha256"]])
            report = owner.admit_stage(recipe, "sf-ft", 1, 1, [record["input_sha256"]], origins)
            self.assertEqual(report["base_inherited_input_sha256"], [record["input_sha256"]])
            self.assertEqual(report["same_stage_retained_input_sha256"], [record["input_sha256"]])
            self.assertEqual(report["base_inherited_origin_sha256"], [parent_record["origin_sha256"]])
            self.assertEqual(report["same_stage_retained_origin_sha256"], [record["origin_sha256"]])
            self.assertEqual(report["base_inherited_origin_count"], 1)
            self.assertEqual(report["same_stage_retained_origin_count"], 1)
            expected = data.digest("rz-rovent-corpus-same-stage-retained-seals/1",
                {"input_sha256": [record["input_sha256"]], "origin_sha256": [record["origin_sha256"]]})
            self.assertEqual(report["same_stage_retained_seals_sha256"], expected)
            self.assertEqual(manifest["inherited_origin_sha256"], [parent_record["origin_sha256"]])

    def test_same_stage_resume_cannot_erase_the_manifest_ancestor_base(self):
        with tempfile.TemporaryDirectory() as directory:
            recipe, _, _, record, owner = parent_and_current(Path(directory))
            with self.assertRaisesRegex(ValueError, "erased/changed ancestor-base training seals"):
                owner.admit_stage(recipe, "sf-ft", 1, 1, [record["input_sha256"]], [record["origin_sha256"]])

    def test_same_stage_retention_rejects_validation_and_holdout_rows(self):
        for opening in ("valid", "held"):
            with self.subTest(opening=opening), tempfile.TemporaryDirectory() as directory:
                recipe, manifest, record = original_fixture(opening=opening)
                owner = write_corpus(Path(directory), manifest, [record])
                with self.assertRaisesRegex(ValueError, "same-stage retained origin is not an original current-stage self corpus train row"):
                    owner.admit_stage(recipe, "common", 1, 1, [record["input_sha256"]], [record["origin_sha256"]])

    def test_same_stage_retention_rejects_foreign_or_known_nonself_stage_origins(self):
        for other_stage, actor in (("common", None), ("sf-ft", "stockfish"), ("reckless-ft", "reckless")):
            with self.subTest(other_stage=other_stage), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                recipe, manifest, record = original_fixture()
                _, other_manifest, other_record = original_fixture(request_sequence=2)
                other_manifest.update(corpus_id="foreign-" + other_stage, stage_id=other_stage,
                    round_id=other_stage + "-round", round=1)
                if actor is not None:
                    other_manifest["teachers"] = {actor: other_manifest["teachers"][actor]}
                seal_record(other_manifest, other_record)
                other_root, current_root = root / "other", root / "current"
                other_root.mkdir()
                current_root.mkdir()
                other = write_corpus(other_root, other_manifest, [other_record])
                manifest["ancestor_index_sha256"] = [other.index_semantic_sha256]
                current = write_corpus(current_root, manifest, [record], ancestors=(other,))
                with self.assertRaisesRegex(ValueError, "same-stage retained origin is not an original current-stage self corpus train row"):
                    current.admit_stage(recipe, "common", 1, 1,
                        [other_record["input_sha256"]], [other_record["origin_sha256"]])
                with self.assertRaisesRegex(ValueError, "same-stage retained origin is not an original current-stage self corpus train row"):
                    current.admit_stage(recipe, "common", 1, 1, [record["input_sha256"]], [h("unknown-foreign-origin")])


if __name__ == "__main__":
    unittest.main()
