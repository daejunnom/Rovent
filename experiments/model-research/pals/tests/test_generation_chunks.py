"""Pure JSONL/index checks; no model, optimizer or native NN execution."""
import copy
import hashlib
import json
import os
from pathlib import Path
import struct
import tempfile
from unittest import mock

from test_generation_contract import SourceOnlyTests, row, policy, wdl, add_teacher_policy_evidence
from rz_pals_model import generation_data as data
from rz_pals_model import generation_training as train
from rz_pals_model import generation_chunks as chunks


def record(game="g1", opening="o1", *, distinct_state=False):
    result = row()
    snapshot = result["input"]
    snapshot.update(game_id=game, opening_id=opening, line_genealogy_id="line-" + game,
                    transposition_sha256=hashlib.sha256(game.encode()).hexdigest(), question_id=hashlib.sha256(("question-" + game).encode()).hexdigest())
    if distinct_state:
        snapshot["rules_anchor"]["rules_state_sha256"] = hashlib.sha256((game + "-state").encode()).hexdigest()
        snapshot["view_sha256"] = data.view_digest(snapshot["selected_view"])
    result["input_sha256"] = data.seal_input(snapshot)
    return result


def write_corpus(root, records, *, splits=None, evidence=None, trajectories=None, record_groups=None, native=True):
    """Small contract fixtures only: never manufacture admission sample quotas."""
    splits = splits or {r["input"]["opening_id"]: "train" for r in records}
    evidence, trajectories = copy.deepcopy(evidence or {}), trajectories or {}
    if native:
        for unique in {r["input_sha256"]: r for r in records}.values():
            add_native_chain(unique, evidence)
    producer = {"schema": "rz-pals-actual-collection-authority/2", "source_kind": data.ACTUAL_SOURCE,
                "source_sha256": "1" * 64, "model_sha256": "4" * 64, "runtime_sha256": "5" * 64,
                "teacher_binary_sha256": "6" * 64, "teacher_profile_sha256": "7" * 64,
                "opening_split_sha256": data.digest("rz-pals-opening-split/1", splits), "registered_before_outcomes": True,
                "actual_game_ids": sorted({r["input"]["game_id"] for r in records})}
    manifest = {"schema": chunks.MANIFEST_DOMAIN, "source_kind": data.ACTUAL_SOURCE, "producer": producer,
                "opening_splits": splits, "chunks": [], "total_bytes": 0, "total_records": len(records)}
    groups = [("inputs", [{"input_sha256": seal} for seal in sorted({r["input_sha256"] for r in records})])]
    groups += [("evidence", [{"sha256": key, "receipt": value} for key, value in evidence.items()])]
    groups += [("trajectories", [{"sha256": key, "trajectory": value} for key, value in trajectories.items()])]
    groups += [("records", values) for values in (record_groups or [records])]
    for kind, values in groups:
        if not values:
            continue
        index = len(manifest["chunks"])
        relative = f"{kind}-{index:06d}.jsonl"
        raw = b"".join(json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode() + b"\n" for value in values)
        (root / relative).write_bytes(raw)
        manifest["chunks"].append({"index": index, "kind": kind, "path": relative,
                                    "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw), "rows": len(values)})
        manifest["total_bytes"] += len(raw)
    return manifest


def append_chunk(root, manifest, kind, values):
    """Append pinned raw fixture locations without deduplicating their rows."""
    index = len(manifest["chunks"])
    relative = f"{kind}-{index:06d}.jsonl"
    raw = b"".join(json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode() + b"\n"
                   for value in values)
    (root / relative).write_bytes(raw)
    entry = {"index": index, "kind": kind, "path": relative, "sha256": hashlib.sha256(raw).hexdigest(),
             "bytes": len(raw), "rows": len(values)}
    manifest["chunks"].append(entry)
    manifest["total_bytes"] += len(raw)
    if kind == "records":
        manifest["total_records"] += len(values)
    return entry


def add_native_chain(record, receipts):
    """Schema fixture only; never executed and never fills real admission quota."""
    from rz_pals_model.virtual_review import input_digest
    ticket = {"epoch": 0, "request_sequence": 1, "game": 1, "root": 1, "question_sha256": record["input"]["question_id"],
              "source_view_sha256": "c" * 64, "view_sequence": 1, "context_policy_sha256": "c" * 64,
              "representation_input_sha256": "8" * 64, "representation_history_sha256": "b" * 64,
              "representation_model_sha256": "4" * 64, "representation_encoding_sha256": "7" * 64,
              "frozen_epoch": 0, "record_revision": 10, "deadline_tick": 1000}
    submission = {"schema": chunks.NATIVE_DOMAINS[0], "input_sha256": record["input_sha256"], "ticket": ticket,
                  "source_view_seal_sha256": ticket["source_view_sha256"], "physical_nn_admitted": False,
                  "native_input_sha256": input_digest(record["input"]["encoded_model_input"]), "native_model_epoch_sha256": "4" * 64,
                  "native_backend_sha256": "5" * 64, "native_anchor_history_sha256": ["9" * 64] * len(record["input"]["selected_view"]["anchors"]),
                  "generation_view_sha256": record["input"]["view_sha256"],
                  **{name: record["input"][name] for name in ("game_id", "opening_id", "role", "model_epoch", "context_epoch", "capture_sequence", "worker_id", "job_id", "slice_id")},
                  "actual_root_anchor": record["input"]["selected_view"]["anchors"][0], "query_anchor": record["input"]["rules_anchor"],
                  "source_view_content_sha256": "a" * 64, "native_endpoint_history_sha256": ticket["representation_history_sha256"]}
    submission_sha = data.digest(submission["schema"], submission)
    completion = {"schema": chunks.NATIVE_DOMAINS[1], "input_sha256": record["input_sha256"], "submission_sha256": submission_sha,
                  "ticket": ticket, "physical_completion": "confirmed", "execution": {"epoch": 0, "sequence": 1},
                  "completed_nn_inputs": 2, "measurement_complete": True, "physical_fresh": True,
                  "invocation_key_sha256": "d" * 64, "accepted_seed_sha256": None,
                  "output": {"kind": "proposal", "candidate_logits_f32": ["00000000"] * len(record["input"]["legal_moves"]),
                             "wdl_f32": ["3e4ccccd", "3e99999a", "3f000000"]}, "admissible_actual_nn": True}
    completion_sha = data.digest(completion["schema"], completion)
    retirement = {"schema": chunks.NATIVE_DOMAINS[2], "input_sha256": record["input_sha256"], "submission_sha256": submission_sha,
                  "completion_sha256": completion_sha, "ticket": ticket, "accepted": True}
    for value in (submission, completion, retirement):
        receipts[data.digest(value["schema"], value)] = value
    return submission, completion, retirement


def add_policy(record, receipts):
    target = policy(record)
    receipt = {"schema": "rz-pals-generation-head-evidence/1", "input_sha256": record["input_sha256"],
               "head": "policy", "target": {k: v for k, v in target.items() if k != "raw_evidence_sha256"}}
    target["raw_evidence_sha256"] = data.digest(receipt["schema"], receipt)
    receipts[target["raw_evidence_sha256"]] = receipt
    record["future_targets"]["policy"] = target
    add_teacher_policy_evidence(record, receipts, target)


class ChunkTests(SourceOnlyTests):
    def setUp(self):
        super().setUp()
        self.temporary = tempfile.TemporaryDirectory(prefix="rz-generation-chunk-tests-")
        self.root = Path(self.temporary.name)
        self.corpus = self.root / "corpus"
        self.index_root = self.root / "index"
        self.corpus.mkdir()
        self.index_root.mkdir()

    def tearDown(self):
        self.temporary.cleanup()
        super().tearDown()

    def opened(self, manifest, **kwargs):
        return chunks.ChunkGenerationDataset(self.corpus, manifest, expected_manifest_sha256=chunks.validate_manifest(manifest),
                                              index_root=self.index_root, **kwargs)

    def test_v_cli_requires_external_manifest_pin_and_keeps_admission_gate(self):
        # A tiny schema fixture is deliberately insufficient for V admission.
        # It exercises real chunk hashing and joins; it is not training data.
        splits = {"o1": "train"}
        splits.update({f"train-{i}": "train" for i in range(47)})
        splits.update({f"validation-{i}": "validation" for i in range(8)})
        splits.update({f"holdout-{i}": "holdout" for i in range(8)})
        manifest = write_corpus(self.corpus, [record()], splits=splits)
        registered = chunks.validate_manifest(manifest)
        spec = train.generation_spec(generation_id="g0", source_sha256="1" * 64,
                                     dataset_sha256="2" * 64,
                                     split_sha256=data.digest("rz-pals-opening-split/1", splits),
                                     model_sha256="4" * 64, runtime_sha256="5" * 64)
        spec_path, manifest_path = self.root / "spec.json", self.corpus / "manifest.json"
        spec_path.write_text(json.dumps(spec), encoding="utf-8")
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        args = ["--spec", str(spec_path), "--dataset", str(manifest_path), "--phase", "v",
                "--frozen-pc-sha256", "4" * 64, "--model-epoch", "1"]
        with mock.patch.dict(os.environ, {"APPDATA": str(self.root), "XDG_CACHE_HOME": str(self.root)}):
            with self.assertRaisesRegex(ValueError, "externally registered"):
                train.main(args)
            with self.assertRaises(ValueError):
                train.main(args + ["--dataset-manifest-sha256", "not-a-digest"])
            with self.assertRaisesRegex(ValueError, "externally"):
                train.main(args + ["--dataset-manifest-sha256", "f" * 64])
            # The correct external seal reaches the real quota gate. A fixture
            # with no valid V targets must never be accepted as a real dataset.
            with self.assertRaisesRegex(ValueError, "InsufficientValidTargets"):
                train.main(args + ["--dataset-manifest-sha256", registered])

            changed = copy.deepcopy(manifest)
            changed["producer"]["source_sha256"] = "9" * 64
            manifest_path.write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "externally"):
                train.main(args + ["--dataset-manifest-sha256", registered])

            with self.assertRaisesRegex(ValueError, "P/C chunk validation uses"):
                train.main(["--spec", str(spec_path), "--dataset", str(manifest_path), "--phase", "pc",
                            "--dataset-manifest-sha256", registered])

    def test_lazy_rows_and_role_sampler_preserve_checkpoint_without_nn(self):
        records = [record("g1", distinct_state=True), record("g2", distinct_state=True)]
        records[1]["input"]["encoded_model_input"]["anchors"][0]["metadata"][0] = 0.25
        records[1]["input_sha256"] = data.seal_input(records[1]["input"])
        manifest = write_corpus(self.corpus, records)
        with self.opened(manifest) as dataset:
            rows = dataset.rows("train", optimization=True)
            self.assertIsInstance(rows, chunks.ChunkRows)
            sampler = train.GenerationSampler(rows, "proposer")
            self.assertIs(sampler.records, rows)
            self.assertEqual(sampler.algorithm, "python-mt19937-generation-chunk-role-v1")
            indices, pool = rows.role_pool("proposer")
            self.assertEqual(indices, [0, 1])
            self.assertEqual(pool, data.digest("rz-pals-generation-chunk-sampler-pool/1", [dataset.sha256, [r["input_sha256"] for r in records]]))
            sampler.next_accumulation()
            resumed = train.GenerationSampler(rows, "proposer", state=sampler.state())
            self.assertEqual(sampler.next_accumulation(), resumed.next_accumulation())
            with self.assertRaises(TypeError):
                rows[:]
            with self.assertRaisesRegex(ValueError, "optimization"):
                dataset.rows("holdout", optimization=True)
            self.assertLessEqual(dataset._cache_bytes, chunks.MAX_CACHE_BYTES)
            self.assertFalse(hasattr(dataset, "_records"))
            with self.assertRaisesRegex(ValueError, "InsufficientValidTargets"):
                dataset.admit("pc")
        self.assertEqual(list(self.index_root.iterdir()), [])
        with self.assertRaisesRegex(ValueError, "closed"):
            rows[0]

    def test_chunk_bytes_corruption_and_replacement_fail_immutable_read(self):
        manifest = write_corpus(self.corpus, [record()])
        with self.opened(manifest) as dataset:
            rows = dataset.rows("train")
            path = self.corpus / manifest["chunks"][-1]["path"]
            raw = path.read_bytes()
            path.write_bytes(raw.replace(b"g1", b"g9", 1))
            with self.assertRaisesRegex(ValueError, "hash"):
                rows[0]
        path.write_bytes(raw)
        with self.opened(manifest) as dataset:
            replacement = self.corpus / "replacement.jsonl"
            replacement.write_bytes(raw)
            replacement.replace(path)
            with self.assertRaisesRegex(ValueError, "replaced"):
                dataset.sha256

    def test_reordering_changed_pins_missing_or_duplicate_chunks_refused(self):
        manifest = write_corpus(self.corpus, [record()])
        registered = chunks.validate_manifest(manifest)
        modified = copy.deepcopy(manifest)
        modified["chunks"].reverse()
        with self.assertRaisesRegex(ValueError, "reordered"):
            chunks.validate_manifest(modified)
        for index, entry in enumerate(modified["chunks"]):
            entry["index"] = index
        with self.assertRaisesRegex(ValueError, "externally"):
            chunks.ChunkGenerationDataset(self.corpus, modified, expected_manifest_sha256=registered, index_root=self.index_root)
        modified = copy.deepcopy(manifest)
        modified["producer"]["source_sha256"] = "9" * 64
        with self.assertRaisesRegex(ValueError, "externally"):
            chunks.ChunkGenerationDataset(self.corpus, modified, expected_manifest_sha256=registered, index_root=self.index_root)
        modified = copy.deepcopy(manifest)
        modified["chunks"][1]["path"] = modified["chunks"][0]["path"]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            chunks.validate_manifest(modified)
        (self.corpus / manifest["chunks"][0]["path"]).unlink()
        with self.assertRaises(FileNotFoundError):
            self.opened(manifest)

    def test_duplicate_rows_count_once_and_conflicting_targets_are_rejected(self):
        original = record()
        manifest = write_corpus(self.corpus, [original, copy.deepcopy(original)])
        with self.opened(manifest) as dataset:
            self.assertEqual(len(dataset.rows("train")), 1)
            self.assertEqual(dataset.exclusions["duplicate_inputs_excluded"], 1)
        receipts = {}
        conflict = copy.deepcopy(original)
        add_policy(conflict, receipts)
        manifest = write_corpus(self.corpus, [original, conflict], evidence=receipts)
        with self.assertRaisesRegex(ValueError, "conflicting"):
            self.opened(manifest)

    def test_global_crosssplit_leak_found_in_final_chunk_excludes_both(self):
        first, last = record("g1", "o1"), record("g2", "o2")
        manifest = write_corpus(self.corpus, [first, last], splits={"o1": "train", "o2": "holdout"}, record_groups=[[first], [last]])
        with self.opened(manifest) as dataset:
            self.assertEqual(len(dataset.rows("train")), 0)
            self.assertEqual(len(dataset.rows("holdout")), 0)
            self.assertEqual(dataset.exclusions["cross_split_records_excluded"], 2)
            self.assertEqual(dataset.exclusions["cross_split_identities"], 2)

    def test_unknown_origin_and_unpublished_input_are_rejected(self):
        manifest = write_corpus(self.corpus, [record()])
        manifest["source_kind"] = "diagnostic"
        with self.assertRaisesRegex(ValueError, "origin"):
            chunks.validate_manifest(manifest)
        manifest = write_corpus(self.corpus, [record()])
        input_chunk = manifest["chunks"][0]
        raw = json.dumps({"input_sha256": "f" * 64}).encode() + b"\n"
        (self.corpus / input_chunk["path"]).write_bytes(raw)
        manifest["total_bytes"] += len(raw) - input_chunk["bytes"]
        input_chunk.update(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
        with self.assertRaisesRegex(ValueError, "publication"):
            self.opened(manifest)

    def test_full_receipt_join_is_preserved_without_all_evidence_dictionary(self):
        r, receipts = record(), {}
        add_policy(r, receipts)
        manifest = write_corpus(self.corpus, [r], evidence=receipts)
        with self.opened(manifest) as dataset:
            self.assertEqual(data.coverage(dataset.rows("train"))["proposer_policy_valid"], 1)
            self.assertFalse(hasattr(dataset, "_evidence"))
        receipts[next(iter(receipts))]["target"]["completed_depth"] = 99
        manifest = write_corpus(self.corpus, [r], evidence=receipts)
        with self.assertRaisesRegex(ValueError, "receipt seal"):
            self.opened(manifest)

    def test_identical_evidence_definitions_keep_each_original_chunk_occurrence(self):
        published, receipts = record(), {}
        add_policy(published, receipts)
        manifest = write_corpus(self.corpus, [published], evidence=receipts)
        first = next(entry for entry in manifest["chunks"] if entry["kind"] == "evidence")
        original_raw = (self.corpus / first["path"]).read_bytes()
        original_rows = [data.strict_json(line) for line in original_raw.decode().splitlines()]
        # Key order/raw bytes may differ; only canonical receipt content may
        # reuse one dictionary definition. All original bytes stay pinned.
        repeated = [{"sha256": row["sha256"], "receipt": dict(reversed(list(row["receipt"].items())))}
                    for row in original_rows]
        last = append_chunk(self.corpus, manifest, "evidence", repeated)
        with self.opened(manifest) as dataset:
            self.assertEqual(dataset._db.execute("SELECT COUNT(*) FROM evidence").fetchone()[0], len(original_rows))
            self.assertEqual(dataset._db.execute("SELECT COUNT(*) FROM evidence_occurrences").fetchone()[0], 2 * len(original_rows))
            self.assertEqual(dataset._db.execute("SELECT COUNT(*) FROM heads").fetchone()[0], 1)
            self.assertEqual(dataset._db.execute("SELECT COUNT(*) FROM teacher_inputs").fetchone()[0], 1)
            self.assertEqual(dataset._db.execute("SELECT COUNT(*) FROM native").fetchone()[0], 3)
            self.assertEqual(data.coverage(dataset.rows("train"))["proposer_policy_valid"], 1)
            for row in original_rows:
                occurrences = list(dataset.evidence_occurrences(row["sha256"]))
                self.assertEqual([item["chunk"] for item in occurrences], [first["index"], last["index"]])
                for item in occurrences:
                    self.assertEqual(item["chunk_sha256"], manifest["chunks"][item["chunk"]]["sha256"])
                    raw = (self.corpus / item["path"]).read_bytes()[item["offset"]:item["offset"] + item["length"]]
                    self.assertEqual(data.strict_json(raw.decode())["sha256"], row["sha256"])
            self.assertEqual(list(dataset.evidence_occurrences("f" * 64)), [])
            locations = dataset.evidence_occurrences(original_rows[0]["sha256"])
        with self.assertRaisesRegex(ValueError, "closed"):
            next(locations)
        self.assertEqual((self.corpus / first["path"]).read_bytes(), original_raw)

    def test_repeated_unsubmitted_retirement_retains_raw_causality_and_capacity(self):
        published, receipts = record(), {}
        _, _, retired = add_native_chain(published, receipts)
        cancelled = copy.deepcopy(retired)
        cancelled.update(input_sha256=None, submission_sha256=None, completion_sha256=None, accepted=False)
        seal = data.digest(cancelled["schema"], cancelled)
        receipts[seal] = cancelled
        manifest = write_corpus(self.corpus, [published], evidence=receipts, native=False)
        append_chunk(self.corpus, manifest, "evidence", [{"sha256": seal, "receipt": cancelled}])
        with self.opened(manifest) as dataset:
            self.assertEqual(len(list(dataset.evidence_occurrences(seal))), 2)
            self.assertEqual(dataset._db.execute("SELECT COUNT(*) FROM native").fetchone()[0], 3)
            self.assertEqual(len(dataset.rows("train")), 1)
        # Unique definitions fit, but every raw occurrence consumes the row
        # and byte budgets; dictionary reuse cannot hide its storage cost.
        with mock.patch.object(chunks, "MAX_EVIDENCE", len(receipts)):
            with self.assertRaisesRegex(ValueError, "entry capacity"):
                self.opened(manifest)
        with mock.patch.object(chunks, "MAX_CORPUS_BYTES", manifest["total_bytes"] - 1):
            with self.assertRaises(ValueError):
                self.opened(manifest)
        malformed = copy.deepcopy(cancelled)
        malformed["ticket"]["request_sequence"] = True
        changed = copy.deepcopy(manifest)
        append_chunk(self.corpus, changed, "evidence", [{"sha256": seal, "receipt": malformed}])
        # Even a hypothetical digest collision cannot bypass the repeated
        # receipt's domain-specific causal identity validation.
        with mock.patch.object(chunks, "validate_evidence_receipt", side_effect=lambda _, value: value):
            with self.assertRaises(ValueError):
                self.opened(changed)

    def test_identical_evidence_seal_with_different_canonical_content_is_rejected(self):
        manifest = write_corpus(self.corpus, [record()])
        first = next(entry for entry in manifest["chunks"] if entry["kind"] == "evidence")
        row = data.strict_json((self.corpus / first["path"]).read_bytes().splitlines()[0].decode())
        collision = copy.deepcopy(row)
        collision["receipt"]["ticket"]["deadline_tick"] += 1
        append_chunk(self.corpus, manifest, "evidence", [collision])
        with self.assertRaisesRegex(ValueError, "receipt seal"):
            self.opened(manifest)
        # The full canonical comparison is an independent check, even if
        # two distinct valid-shaped receipts had the same cryptographic seal.
        with mock.patch.object(chunks, "validate_evidence_receipt", side_effect=lambda _, value: value):
            with self.assertRaisesRegex(ValueError, "conflicting canonical receipt"):
                self.opened(manifest)

    def test_evidence_dictionary_reuse_does_not_relax_input_or_trajectory_uniqueness(self):
        published = record()
        manifest = write_corpus(self.corpus, [published])
        append_chunk(self.corpus, manifest, "inputs", [{"input_sha256": published["input_sha256"]}])
        with self.assertRaisesRegex(ValueError, "Integrity"):
            self.opened(manifest)
        trajectory = {"game_id": "g1", "ending": "checkmate", "outcome": "white_win",
                      "anchors": [published["input"]["rules_anchor"]], "result_evidence_sha256": "d" * 64}
        seal = data.trajectory_digest(trajectory)
        manifest = write_corpus(self.corpus, [published], trajectories={seal: trajectory})
        append_chunk(self.corpus, manifest, "trajectories", [{"sha256": seal, "trajectory": trajectory}])
        with self.assertRaisesRegex(ValueError, "Integrity"):
            self.opened(manifest)

    def test_paths_row_count_storage_and_index_caps_do_not_fallback(self):
        manifest = write_corpus(self.corpus, [record()])
        for invalid in ("../escape.jsonl", "/escape.jsonl", "C:/escape.jsonl", "a\\escape.jsonl"):
            modified = copy.deepcopy(manifest)
            modified["chunks"][0]["path"] = invalid
            with self.assertRaises(ValueError):
                self.opened(modified)
        modified = copy.deepcopy(manifest)
        modified["chunks"][0]["rows"] = 2
        with self.assertRaisesRegex(ValueError, "row count"):
            self.opened(modified)
        modified = copy.deepcopy(manifest)
        modified["chunks"][0]["bytes"] = chunks.MAX_CHUNK_BYTES + 1
        with self.assertRaises(ValueError):
            chunks.validate_manifest(modified)
        with self.assertRaisesRegex(ValueError, "capacity|Capacity"):
            self.opened(manifest, max_index_bytes=32768)
        self.assertEqual(list(self.index_root.iterdir()), [])

    def test_input_seal_normalizes_f32_bits_without_mutating_payload(self):
        r = record()
        snapshot = r["input"]
        value = 1.0 / 16384
        snapshot["encoded_model_input"]["anchors"][0]["metadata"][0] = value
        expected = copy.deepcopy(snapshot)
        for anchor in expected["encoded_model_input"]["anchors"]:
            anchor["metadata"] = [struct.pack(">f", float(v)).hex() for v in anchor["metadata"]]
        self.assertEqual(data.seal_input(snapshot), data.digest(data.INPUT_DOMAIN, expected))
        self.assertEqual(snapshot["encoded_model_input"]["anchors"][0]["metadata"][0], value)

    def test_native_completed_and_accepted_chain_is_required(self):
        r = record()
        manifest = write_corpus(self.corpus, [r], native=False)
        with self.assertRaisesRegex(ValueError, "retirement"):
            self.opened(manifest)
        for field, bad in (("physical_completion", "unknown"), ("completed_nn_inputs", 0),
                           ("measurement_complete", False), ("execution", None), ("invocation_key_sha256", None)):
            receipts = {}
            submission, completion, retirement = add_native_chain(r, receipts)
            completion[field] = bad
            retirement["completion_sha256"] = data.digest(completion["schema"], completion)
            receipts = {data.digest(v["schema"], v): v for v in (submission, completion, retirement)}
            manifest = write_corpus(self.corpus, [r], evidence=receipts, native=False)
            with self.assertRaises(ValueError):
                self.opened(manifest)

    def test_unsubmitted_retirement_keeps_only_complete_null_tuple_as_raw_evidence(self):
        r, receipts = record(), {}
        _, _, retired = add_native_chain(r, receipts)
        cancelled = copy.deepcopy(retired)
        cancelled.update(input_sha256=None, submission_sha256=None, completion_sha256=None, accepted=False)
        self.assertFalse(chunks._native_retirement(cancelled))
        receipts[data.digest(cancelled["schema"], cancelled)] = cancelled
        manifest = write_corpus(self.corpus, [r], evidence=receipts, native=False)
        with self.opened(manifest) as corpus:
            self.assertEqual(len(corpus.rows("train", optimization=True)), 1)
        mutations = (
            lambda x: x.update(submission_sha256="a" * 64),
            lambda x: x.update(completion_sha256="a" * 64),
            lambda x: x.update(accepted=True),
            lambda x: x.update(accepted=0),
            lambda x: x.update(extra="unknown"),
            lambda x: x["ticket"].update(request_sequence=True),
            lambda x: x["ticket"].update(epoch=-1),
            lambda x: x["ticket"].update(deadline_tick=2**64),
            lambda x: x["ticket"].update(question_sha256="unknown"),
        )
        for mutation in mutations:
            malformed = copy.deepcopy(cancelled)
            mutation(malformed)
            evidence = copy.deepcopy(receipts)
            evidence[data.digest(malformed["schema"], malformed)] = malformed
            manifest = write_corpus(self.corpus, [r], evidence=evidence, native=False)
            with self.subTest(retirement=malformed):
                with self.assertRaises(ValueError):
                    self.opened(manifest)

    def test_request_ids_do_not_manufacture_unique_semantic_training_samples(self):
        masked, first, later = record("g1"), record("g2"), record("g3")
        receipts = {}
        add_policy(first, receipts)
        add_policy(later, receipts)
        # All three raw input seals differ; actual NN input/history/model/role
        # are identical. Keep the earliest genuinely supervised publication.
        manifest = write_corpus(self.corpus, [masked, first, later], evidence=receipts)
        with self.opened(manifest) as corpus:
            rows = corpus.rows("train", optimization=True)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["input_sha256"], first["input_sha256"])
            self.assertEqual(corpus.exclusions["duplicate_semantic_inputs_excluded"], 2)
            self.assertEqual(corpus._db.execute("SELECT COUNT(*) FROM records").fetchone()[0], 3)
            self.assertEqual(data.coverage(rows)["proposer_valid_samples"], 1)
            with self.assertRaisesRegex(ValueError, "InsufficientValidTargets"):
                corpus.admit("pc")
        # A meaningful tensor change has its own identity; a request ID does
        # not. The source input seals and raw chunks are never rewritten.
        a, b, c = (add_native_chain(value, {})[0] for value in (masked, first, later))
        self.assertEqual(chunks.semantic_sample_seal(masked, a), chunks.semantic_sample_seal(first, b))
        b["ticket"]["representation_history_sha256"] = "d" * 64
        self.assertNotEqual(chunks.semantic_sample_seal(masked, a), chunks.semantic_sample_seal(first, b))

    def test_streamed_late_policy_hydrates_one_row_without_rewriting_input_or_publication(self):
        published = record()
        labelled, receipts = copy.deepcopy(published), {}
        add_policy(labelled, receipts)
        manifest = write_corpus(self.corpus, [published], evidence=receipts)
        original = (self.corpus / manifest["chunks"][-1]["path"]).read_bytes()
        with self.opened(manifest) as corpus:
            rows = corpus.rows("train", optimization=True)
            self.assertEqual(rows[0]["future_targets"], labelled["future_targets"])
            self.assertEqual(list(rows)[0]["future_targets"], labelled["future_targets"])
            self.assertEqual(rows[0]["input"], published["input"])
            self.assertEqual(rows[0]["input_sha256"], published["input_sha256"])
            self.assertEqual(data.coverage(rows)["proposer_valid_samples"], 1)
            self.assertIsNone(published["future_targets"]["policy"])
            with self.assertRaisesRegex(ValueError, "InsufficientValidTargets"):
                corpus.admit("pc")
        self.assertEqual((self.corpus / manifest["chunks"][-1]["path"]).read_bytes(), original)
        self.assertIsNone(data.strict_json(original.decode())["future_targets"]["policy"])

    def test_streamed_final_wdl_requires_exact_rules_trajectory_and_future_event(self):
        published, receipts = record(), {}
        result = {"schema": "rz-pals-rules-outcome-evidence/1", "game_id": "g1", "ending": "checkmate",
                  "outcome": "white_win", "rules_terminal": True}
        result_seal = data.digest(result["schema"], result)
        receipts[result_seal] = result
        trajectory = {"game_id": "g1", "ending": "checkmate", "outcome": "white_win",
                      "anchors": [published["input"]["rules_anchor"]], "result_evidence_sha256": result_seal}
        trajectory_seal = data.trajectory_digest(trajectory)
        target = wdl(published)
        target.update(trajectory_membership_sha256=trajectory_seal, rules_result_sha256=result_seal)
        receipt = {"schema": chunks.HEAD_EVIDENCE_DOMAIN, "input_sha256": published["input_sha256"], "head": "wdl",
                   "target": {k: v for k, v in target.items() if k != "raw_evidence_sha256"}}
        receipt_seal = data.digest(receipt["schema"], receipt)
        receipts[receipt_seal] = receipt
        manifest = write_corpus(self.corpus, [published], evidence=receipts, trajectories={trajectory_seal: trajectory})
        with self.opened(manifest) as corpus:
            actual = corpus.rows("train")[0]
            self.assertEqual(actual["future_targets"]["wdl"]["raw_evidence_sha256"], receipt_seal)
            self.assertEqual(data.validate_record(actual)["wdl"], [1.0, 0.0, 0.0])
            self.assertIsNone(published["future_targets"]["wdl"])
        for mutation in (lambda v: v.update(observed_sequence=10),
                         lambda v: v["anchor"].update(rules_state_sha256="3" * 64),
                         lambda v: v.update(view_sha256="2" * 64)):
            invalid_receipt = copy.deepcopy(receipt)
            mutation(invalid_receipt["target"])
            evidence = {result_seal: result, data.digest(invalid_receipt["schema"], invalid_receipt): invalid_receipt}
            manifest = write_corpus(self.corpus, [published], evidence=evidence, trajectories={trajectory_seal: trajectory})
            with self.subTest(target=invalid_receipt["target"]):
                with self.assertRaises(ValueError):
                    self.opened(manifest)

    def test_streamed_head_conflict_duplicate_and_wrong_private_role_are_rejected(self):
        published, receipts = record(), {}
        add_policy(published, receipts)
        original = copy.deepcopy(published)
        conflicting = copy.deepcopy(next(iter(receipts.values())))
        conflicting["target"]["ranked_moves"].reverse()
        conflicting_seal = data.digest(conflicting["schema"], conflicting)
        evidence = {conflicting_seal: conflicting}
        manifest = write_corpus(self.corpus, [published], evidence=evidence)
        with self.assertRaisesRegex(ValueError, "conflicts"):
            self.opened(manifest)
        self.assertEqual(published, original)
        evidence.update(receipts)
        manifest = write_corpus(self.corpus, [published], evidence=evidence)
        with self.assertRaisesRegex(ValueError, "Integrity"):
            self.opened(manifest)
        validator = record()
        validator["input"]["role"] = "validator"
        validator["input"]["encoded_model_input"]["role"] = "validator"
        validator["input_sha256"] = data.seal_input(validator["input"])
        validator["future_targets"]["policy"] = None
        invalid = copy.deepcopy(conflicting)
        invalid["input_sha256"] = validator["input_sha256"]
        invalid["target"]["question_id"] = validator["input"]["question_id"]
        manifest = write_corpus(self.corpus, [validator], evidence={data.digest(invalid["schema"], invalid): invalid})
        with self.assertRaisesRegex(ValueError, "private role"):
            self.opened(manifest)

    def test_raw_runtime_audit_is_bounded_strict_metadata_without_label_authority(self):
        receipt = {"schema": "rz-pals-generation-runtime-audit-evidence/1", "witness_json": '{"status":"not_executed"}'}
        self.assertEqual(data.validate_evidence_receipt(data.digest(receipt["schema"], receipt), receipt), receipt)
        for witness in ('{"key":1,"key":2}', '{"x":NaN}', '{"nested":{"x":1e9999}}',
                        '{"x":[-1e9999]}', '[]', '"text"', '{', " " * (1024 * 1024 + 1)):
            invalid = dict(receipt, witness_json=witness)
            with self.subTest(witness=witness[:40]):
                with self.assertRaises(ValueError):
                    data.validate_evidence_receipt(data.digest(invalid["schema"], invalid), invalid)

    def test_raw_game_pgn_failure_and_clock_metadata_cannot_mint_head_targets(self):
        audit = {"schema": "rz-pals-collection-clock/3", "base_ms": 120000, "increment_ms": 1000,
                 "opening_plies": 0, "accepted_plies": 0, "white_remaining_ns": 120000000000,
                 "black_remaining_ns": 120000000000, "white_wall_charged_ns": 0, "black_wall_charged_ns": 0,
                 "white_increments": 0, "black_increments": 0}
        pgn = {"schema": "rz-pals-generation-game-pgn-evidence/1", "game_id": "g1", "opening_id": "o1",
               "engine_a_white": True, "pgn": '[Event "schema-only failed game"]\n\n*', "clock_audit": audit}
        failure = {"schema": "rz-pals-generation-game-failure-evidence/1", "game_id": "g1", "opening_id": "o1",
                   "stage": "actual_decision", "error": "schema-only Infrastructure"}
        receipts = {data.digest(value["schema"], value): value for value in (pgn, failure)}
        published = record()
        manifest = write_corpus(self.corpus, [published], evidence=receipts)
        with self.opened(manifest) as corpus:
            hydrated = corpus.rows("train")[0]
            self.assertEqual(hydrated["future_targets"], dict.fromkeys(chunks.HEADS))
            self.assertEqual(data.coverage([hydrated]).get("proposer_valid_samples", 0), 0)
        for mutation in (lambda v: v.update(engine_a_white=1), lambda v: v.update(extra="unknown"),
                         lambda v: v["clock_audit"].update(white_increments=1),
                         lambda v: v["clock_audit"].update(base_ms=True),
                         lambda v: v["clock_audit"].update(white_wall_charged_ns=-1),
                         lambda v: v["clock_audit"].update(schema="unknown")):
            invalid = copy.deepcopy(pgn)
            mutation(invalid)
            with self.subTest(raw=invalid):
                with self.assertRaises(ValueError):
                    data.validate_evidence_receipt(data.digest(invalid["schema"], invalid), invalid)
        for mutation in (lambda v: v.update(error=""), lambda v: v.update(error=False),
                         lambda v: v.update(stage="control\ncharacter"), lambda v: v.update(extra="unknown")):
            invalid = copy.deepcopy(failure)
            mutation(invalid)
            with self.subTest(raw=invalid):
                with self.assertRaises(ValueError):
                    data.validate_evidence_receipt(data.digest(invalid["schema"], invalid), invalid)

    def test_warm_seed_records_stay_raw_and_cannot_enter_fresh_training_pool(self):
        warm = record("g1")
        receipts = {}
        add_policy(warm, receipts)
        submission, completion, retirement = add_native_chain(warm, {})
        completion.update(physical_fresh=False, accepted_seed_sha256="d" * 64)
        retirement["completion_sha256"] = data.digest(completion["schema"], completion)
        receipts.update({data.digest(v["schema"], v): v for v in (submission, completion, retirement)})
        manifest = write_corpus(self.corpus, [warm], evidence=receipts, native=False)
        with self.opened(manifest) as corpus:
            self.assertEqual(len(corpus.rows("train", optimization=True)), 0)
            self.assertEqual(corpus.exclusions["warm_seed_inputs_excluded"], 1)
            self.assertEqual(corpus.exclusion_reasons["warm_seed_inputs_excluded"], "UnsupportedWarmTrainingNamespace")
            self.assertEqual(corpus._db.execute("SELECT COUNT(*) FROM records WHERE physical_fresh=0").fetchone()[0], 1)
            with self.assertRaisesRegex(ValueError, "UnsupportedWarmTrainingNamespace"):
                corpus.admit("pc")
        # A later real Fresh execution of the same logical context is usable;
        # the earlier Warm publication is not transformed or substituted.
        fresh = record("g2")
        add_policy(fresh, receipts)
        add_native_chain(fresh, receipts)
        manifest = write_corpus(self.corpus, [warm, fresh], evidence=receipts, native=False)
        with self.opened(manifest) as corpus:
            rows = corpus.rows("train", optimization=True)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["input_sha256"], fresh["input_sha256"])
            self.assertEqual(corpus.exclusions["warm_seed_inputs_excluded"], 1)
            self.assertEqual(corpus.exclusions["duplicate_semantic_inputs_excluded"], 0)
            self.assertEqual(corpus._db.execute("SELECT COUNT(*) FROM records").fetchone()[0], 2)

    def test_legacy_whole_bundle_cannot_be_actual_training_admission(self):
        # Preserve old parsing, but native three-stage authority is required
        # for every actual optimizer/export entry, regardless of claimed counts.
        with self.assertRaisesRegex(ValueError, "native-chain"):
            train.generation_admission(train.generation_spec(generation_id="g1", source_sha256="1" * 64,
                dataset_sha256="2" * 64, split_sha256="3" * 64, model_sha256="4" * 64, runtime_sha256="5" * 64),
                object.__new__(data.GenerationDataset), "pc")

    def test_final_link_rejected_on_windows_without_relying_on_o_nofollow(self):
        manifest = write_corpus(self.corpus, [record()])
        final = self.corpus / manifest["chunks"][-1]["path"]
        original = Path.is_symlink
        with mock.patch.object(Path, "is_symlink", lambda path: path == final or original(path)):
            with self.assertRaisesRegex(ValueError, "linked"):
                self.opened(manifest)

    def test_invalid_manifest_is_not_copied_before_capacity_validation(self):
        manifest = write_corpus(self.corpus, [record()])
        manifest["unexpected"] = ["large untrusted field"]
        with mock.patch.object(chunks.copy, "deepcopy", side_effect=AssertionError("copy before strict validation")):
            with self.assertRaisesRegex(ValueError, "fields"):
                self.opened(manifest)

    def test_cleanup_failure_clears_cache_and_preserves_owned_retry(self):
        manifest = write_corpus(self.corpus, [record()])
        dataset = self.opened(manifest)
        dataset.rows("train")[0]
        temporary = dataset._temporary
        with mock.patch.object(temporary, "cleanup", side_effect=PermissionError("test owned index lock")):
            with self.assertRaisesRegex(ValueError, "CleanupFailed"):
                dataset.close()
        self.assertEqual(dataset._cache_bytes, 0)
        self.assertEqual(dataset.cleanup_failure["error_type"], "PermissionError")
        self.assertTrue(Path(dataset.cleanup_failure["owned_index_root"]).is_relative_to(self.index_root))
        dataset.close()
        self.assertEqual(list(self.index_root.iterdir()), [])

    def test_changed_manifest_cannot_resume_old_role_pool(self):
        records = [record()]
        manifest = write_corpus(self.corpus, records)
        with self.opened(manifest) as dataset:
            old = train.GenerationSampler(dataset.rows("train"), "proposer").state()
        modified = copy.deepcopy(manifest)
        modified["producer"]["source_sha256"] = "9" * 64
        with self.opened(modified) as dataset:
            with self.assertRaisesRegex(ValueError, "pool"):
                train.GenerationSampler(dataset.rows("train"), "proposer", state=old)

    def test_head_rules_endpoint_is_separate_from_public_root_anchor(self):
        r = record()
        snapshot, view = r["input"], r["input"]["selected_view"]
        endpoint = copy.deepcopy(snapshot["rules_anchor"])
        endpoint.update(rules_state_sha256="1" * 64, rules_history_sha256="2" * 64,
                        board_fen="8/8/8/8/8/8/8/K6k b - - 1 1", white_to_move=False)
        snapshot["rules_anchor"] = endpoint
        view["query"].update(prefix=[64], white_to_move=False)
        snapshot["encoded_model_input"]["query"].update(prefix=[[0, 1, 0]], perspective=-1)
        snapshot["view_sha256"] = data.view_digest(view)
        r["input_sha256"] = data.seal_input(snapshot)
        data.validate_record(r)
        self.assertNotEqual(snapshot["rules_anchor"], view["anchors"][0])
        view["query"]["white_to_move"] = True
        with self.assertRaisesRegex(ValueError, "viewpoint"):
            data.view_digest(view)
