"""Tiny synthetic pinned-file contracts; no native/Rules/training authority."""
import copy
import hashlib
import importlib.abc
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


class NeuralImportsForbidden(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"torch", "numpy", "onnx", "onnxruntime"}:
            raise AssertionError("neural imports forbidden in historical corpus contract tests")


_guard = NeuralImportsForbidden()
sys.meta_path.insert(0, _guard)
try:
    from rz_pals_model import generation_data as data
    from rz_pals_model import rovent_corpus as original
    from rz_pals_model import rovent_historical_corpus as historical
    from rz_pals_model import rovent_native_v2 as native_v2
    from rz_pals_model import rovent_corpus_codec as codecs
    from test_rovent_corpus import original_fixture, seal_record, add_rules_wdl, write_corpus
finally:
    sys.meta_path.remove(_guard)


def h(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def raw(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                       allow_nan=False) + "\n").encode("utf-8")


def file_pin(path, payload):
    return {"path": path, "sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}


def artifact(name, identity, size=1):
    return {"path": name, "sha256": identity, "bytes": size,
            "source": "https://example.invalid/contract-fixture", "license": "fixture-only"}


def source_aggregate(commit, components):
    framed = b"rz-rovent-source-aggregate/1\0" + commit.encode() + len(components).to_bytes(4, "little")
    for name, identity in sorted(components.items()):
        framed += len(name.encode()).to_bytes(4, "little") + name.encode() + identity.encode()
    return hashlib.sha256(framed).hexdigest()


def synthetic_v2():
    """Valid data-wire joins only; Rust must independently replay real Rules."""
    recipe, manifest, record = original_fixture()
    components = {"fixture-collection-source": h("compiled collection component")}
    source_sha = source_aggregate(manifest["source_commit"], components)
    recipe_declaration = recipe.to_dict()
    recipe_declaration["source_sha256"] = source_sha
    recipe = original.SeriesRecipe(recipe_declaration)
    manifest.update(source_sha256=source_sha, runtime_sha256=h("actual loaded ORT library"), recipe_sha256=recipe.sha256)
    native = record["native"]
    native["source_sha256"] = source_sha
    snapshot = record["input"]
    replay = {"schema": native_v2.POSITION_DOMAIN, "origin": "fen", "completeness": "unknown_prefix",
              "start_fen": snapshot["selected_view"]["anchors"][0]["board_fen"],
              "moves": copy.deepcopy(snapshot["actual_history"]), "observer_revision": 1,
              "anchor": copy.deepcopy(snapshot["selected_view"]["anchors"][0])}
    profiles = []
    for actor, pin in manifest["teachers"].items():
        profiles.append({"domain": "rz-rovent-external-engine-profile/1", "purpose": "teacher",
            "profile": artifact("profiles/" + actor + ".json", pin["profile_sha256"]),
            "binary": artifact("bin/" + actor, pin["binary_sha256"], pin["binary_bytes"]),
            "canonical_sha256": pin["canonical_sha256"], "capability_sha256": pin["capability_sha256"],
            "source_commit": pin["source_commit"], "requested_threads": 1, "requested_hash_mib": 256})
    # A complete outer lock is supplied as canonical metadata. Rust owns its
    # full typed manifest/resource/model admission; these empty declarations
    # intentionally do not claim to pass that separate parent validation.
    execution = {"domain": "rz-rovent-execution/1", "contract_revision": "0.1", "run_id": "synthetic-source-only",
        "purpose": "collection", "engine": {"source_commit": manifest["source_commit"], "source_components": components},
        "resources": {}, "coordinator": {}, "wall_time_max_ms": 1000, "cleanup_max_ms": 500,
        "output_bytes_max": 1024**2, "agent_renewable": True, "provenance": []}
    lock = {"domain": "rz-rovent-input-lock/1", "manifest": execution,
            "canonical_sha256": data.digest("rz-rovent-execution-canonical/1", execution)}
    origin = {"schema": native_v2.ORIGIN_DOMAIN, "purpose": "collection", "process_epoch": native["process_epoch"],
        "mode": "HybridPcr", "input_lock_sha256": lock["canonical_sha256"],
        "source_commit": manifest["source_commit"], "source_components": components, "source_sha256": source_sha,
        "weights_sha256": native["model_sha256"], "export_manifest_sha256": h("collection export manifest"),
        "backend_sha256": h("collection backend"), "runtime_library_sha256": manifest["runtime_sha256"],
        "runtime_bundle_sha256": None,
        "native_configuration_sha256": h("collection native configuration"),
        "model_semantics": "synthetic-semantics", "model_semantics_sha256": manifest["model_semantics_sha256"],
        "encoding": "synthetic-encoding", "encoding_sha256": native["encoding_sha256"],
        "encoding_registry_sha256": h("registered encoding differs from runtime"),
        "model_implementation_sha256": h("model implementation"), "frozen_epoch": 1,
        "representation_model_domain": "loaded_checkpoint_sha256",
        "runtime_sha256_domain": "loaded_runtime_library_file_sha256", "runtime_sha256": manifest["runtime_sha256"],
        "teacher_profiles": profiles}
    ticket = native["raw_submission"]["ticket"]
    root = {"schema": native_v2.ROOT_DOMAIN, "native_origin": origin,
        "native_origin_sha256": data.digest(native_v2.ORIGIN_DOMAIN, origin),
        **{key: ticket[key] for key in ("process_epoch", "game_generation", "root_generation",
                                      "implementation_sha256", "deadline_clock", "deadline_tick")},
        "root_budget_sha256": h("root budget"), "selective_policy_sha256": h("selective policy"),
        "root_state": copy.deepcopy(ticket["root_state"]), "store_root_state": 1,
        "root_moves": ["a2a3"], "depth_plies": None, "mate_moves": None, "requested_nodes": 4096}
    for name in ("raw_submission", "raw_completion", "raw_retirement"):
        native[name].update(schema=native_v2.NATIVE_DOMAIN, source_sha256=source_sha)
    native["raw_submission"]["rules_replay"] = {"schema": native_v2.SUBMISSION_DOMAIN,
        "actual_root": replay, "selected_anchors": [copy.deepcopy(replay)]}
    native["raw_retirement"].update(acceptance_kind="accepted_role_step", root_admission=root,
        root_admission_sha256=data.digest(native_v2.ROOT_DOMAIN, root))
    reseal_native(manifest, record)
    binding = {"model_semantics_sha256": manifest["model_semantics_sha256"],
        "model_identity_sha256": native["model_sha256"], "weights_sha256": origin["weights_sha256"],
        "encoding_sha256": native["encoding_sha256"]}
    return recipe, manifest, record, lock, binding


def reseal_native(manifest, record):
    native = record["native"]
    native["raw_submission_sha256"] = data.digest(native_v2.NATIVE_DOMAIN, native["raw_submission"])
    native["raw_completion"]["submitted_sha256"] = native["raw_submission_sha256"]
    native["raw_completion_sha256"] = data.digest(native_v2.NATIVE_DOMAIN, native["raw_completion"])
    native["raw_retirement"].update(submitted_sha256=native["raw_submission_sha256"],
                                     completed_sha256=native["raw_completion_sha256"])
    native["raw_retirement_sha256"] = data.digest(native_v2.NATIVE_DOMAIN, native["raw_retirement"])
    seal_record(manifest, record)


def write_publication(root, *, source=None, runtime_bundle=None):
    recipe, manifest, record, lock, binding = synthetic_v2() if source is None else source
    root.mkdir(parents=True, exist_ok=True)
    origin = {key: record[key] for key in ("record_sha256", "input_sha256", "origin_sha256", "split")}
    origin.update({key: record["input"][key] for key in
                   ("game_id", "opening_id", "line_genealogy_id", "transposition_sha256")})
    origin.update(copy.deepcopy(record["lineage"]))
    payloads = {"corpus-manifest.json": raw(manifest), "chunks/000000.jsonl": raw(record),
                "input-locks/original.json": raw(lock)}
    bundle_pin = None
    if runtime_bundle is not None:
        payloads["runtime-bundles/original.json"] = raw(runtime_bundle)
        bundle_pin = file_pin("runtime-bundles/original.json", payloads["runtime-bundles/original.json"])
    chunk = file_pin("chunks/000000.jsonl", payloads["chunks/000000.jsonl"])
    index = {"schema": original.INDEX_DOMAIN, "manifest_sha256": data.digest(original.MANIFEST_DOMAIN, manifest),
             "record_count": 1, "chunks": [{**chunk, "records": 1}], "origins": [origin]}
    payloads["corpus-index.json"] = raw(index)
    admission_sha = record["native"]["raw_retirement"]["root_admission_sha256"]
    publication = {"schema": historical.PUBLICATION_DOMAIN,
        "manifest": file_pin("corpus-manifest.json", payloads["corpus-manifest.json"]),
        "index": file_pin("corpus-index.json", payloads["corpus-index.json"]), "chunks": [chunk],
        "canonical_manifest_sha256": data.digest(original.MANIFEST_DOMAIN, manifest),
        "canonical_index_sha256": data.digest(original.INDEX_DOMAIN, index),
        "source_commit": manifest["source_commit"], "source_sha256": manifest["source_sha256"],
        "runtime_sha256": manifest["runtime_sha256"], "record_count": 1,
        "total_payload_bytes": sum(len(payload) for payload in payloads.values()),
        "native_admission_sha256": [admission_sha],
        "admissions": [{"native_admission_sha256": admission_sha,
                        "input_lock": file_pin("input-locks/original.json", payloads["input-locks/original.json"]),
                        "canonical_input_lock_sha256": lock["canonical_sha256"], "runtime_bundle_manifest": bundle_pin}]}
    payloads["publication.json"] = raw(publication)
    for name, payload in payloads.items():
        path = root.joinpath(*name.split("/"))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    registration = {"root": str(root), "publication": file_pin(str(root / "publication.json"), payloads["publication.json"]),
        "collection_source_commit": manifest["source_commit"], "collection_source_sha256": manifest["source_sha256"],
        "collection_runtime_library_sha256": manifest["runtime_sha256"],
        "model_semantics_sha256": manifest["model_semantics_sha256"], "model_epoch": 1, "context_epoch": 1,
        "recipe_sha256": recipe.sha256, "stage_id": manifest["stage_id"], "role_bindings": {"proposer": binding}}
    return registration, publication, manifest, index, record, payloads


def synthetic_submitted():
    """New synthetic /2 wire, not conversion of recovered original data."""
    from test_rovent_native_v3 import native_fixture
    recipe, _manifest, _record, lock, binding = synthetic_v2()
    manifest, record = native_fixture()
    # The raw /3 fixture intentionally checks only its native wire. Seal the
    # distinct training origin/record here for the complete historical reader.
    reseal_submitted(manifest, record)
    return recipe, manifest, record, lock, binding


def reseal_submitted(manifest, record):
    native = record["native"]
    domain = "rz-rovent-raw-record/3"
    native["raw_submission_sha256"] = data.digest(domain, native["raw_submission"])
    native["raw_completion"]["submitted_sha256"] = native["raw_submission_sha256"]
    native["raw_completion_sha256"] = data.digest(domain, native["raw_completion"])
    native["raw_retirement"].update(submitted_sha256=native["raw_submission_sha256"],
        completed_sha256=native["raw_completion_sha256"])
    native["raw_retirement_sha256"] = data.digest(domain, native["raw_retirement"])
    seal_record(manifest, record)
    projected = {**record, "record_sha256": "", "input": {"sealed_input_sha256": record["input_sha256"]}}
    record["record_sha256"] = data.digest("rz-rovent-training-record/2", projected)


def write_submitted_publication(root, *, source=None):
    """Small independent publication /2 body using the approved exact fields."""
    recipe, manifest, record, lock, binding = synthetic_submitted() if source is None else source
    root.mkdir(parents=True, exist_ok=True)
    origin = {key: record[key] for key in ("record_sha256", "input_sha256", "origin_sha256", "split")}
    origin.update({key: record["input"][key] for key in
        ("game_id", "opening_id", "line_genealogy_id", "transposition_sha256")})
    origin.update(copy.deepcopy(record["lineage"]))
    payloads = {"corpus-manifest.json": raw(manifest), "chunks/000000.jsonl": raw(record),
        "input-locks/original.json": raw(lock)}
    chunk = file_pin("chunks/000000.jsonl", payloads["chunks/000000.jsonl"])
    index = {"schema": "rz-rovent-corpus-index/2",
        "manifest_sha256": data.digest("rz-rovent-corpus-manifest/2", manifest),
        "record_count": 1, "chunks": [{**chunk, "records": 1}], "origins": [origin]}
    payloads["corpus-index.json"] = raw(index)
    admission_sha = record["native"]["raw_retirement"]["root_admission_sha256"]
    publication = {"schema": "rz-rovent-corpus-publication/2",
        "manifest": file_pin("corpus-manifest.json", payloads["corpus-manifest.json"]),
        "index": file_pin("corpus-index.json", payloads["corpus-index.json"]), "chunks": [chunk],
        "canonical_manifest_sha256": data.digest("rz-rovent-corpus-manifest/2", manifest),
        "canonical_index_sha256": data.digest("rz-rovent-corpus-index/2", index),
        "source_commit": manifest["source_commit"], "source_sha256": manifest["source_sha256"],
        "runtime_sha256": manifest["runtime_sha256"], "record_count": 1,
        "total_payload_bytes": sum(len(payload) for payload in payloads.values()),
        "native_admission_sha256": [admission_sha],
        "admissions": [{"native_admission_sha256": admission_sha,
            "input_lock": file_pin("input-locks/original.json", payloads["input-locks/original.json"]),
            "canonical_input_lock_sha256": lock["canonical_sha256"], "runtime_bundle_manifest": None}]}
    payloads["publication.json"] = raw(publication)
    for name, payload in payloads.items():
        path = root.joinpath(*name.split("/"))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    registration = {"root": str(root), "publication": file_pin(str(root / "publication.json"), payloads["publication.json"]),
        "collection_source_commit": manifest["source_commit"], "collection_source_sha256": manifest["source_sha256"],
        "collection_runtime_library_sha256": manifest["runtime_sha256"],
        "model_semantics_sha256": manifest["model_semantics_sha256"], "model_epoch": 1, "context_epoch": 1,
        "recipe_sha256": recipe.sha256, "stage_id": manifest["stage_id"], "role_bindings": {"proposer": binding}}
    return registration, publication, manifest, index, record, payloads


class HistoricalCorpusTests(unittest.TestCase):
    def test_cpu_publication_requires_explicit_null_bundle_without_inventing_cuda(self):
        with tempfile.TemporaryDirectory() as directory:
            registration, publication, _, _, _, _ = write_publication(Path(directory))
            self.assertIsNone(publication["admissions"][0]["runtime_bundle_manifest"])
            before = copy.deepcopy(publication)
            self.assertIs(historical._publication(publication, original.CorpusLimits()), publication)
            self.assertEqual(publication, before)
            missing = copy.deepcopy(publication)
            del missing["admissions"][0]["runtime_bundle_manifest"]
            with self.assertRaisesRegex(ValueError, "fields mismatch"):
                historical._publication(missing, original.CorpusLimits())

    def test_cuda_raw_metadata_file_and_binary_semantic_digest_are_separate_and_budgeted(self):
        filenames = ("libcudart.so.12", "libcublas.so.12", "libcublasLt.so.12", "libcufft.so.11", "libcurand.so.10",
            "libnvrtc.so.12", "libnvrtc-builtins.so.12.8", "libnvJitLink.so.12", "libcudnn.so.9", "libcudnn_adv.so.9",
            "libcudnn_cnn.so.9", "libcudnn_ops.so.9", "libcudnn_engines_precompiled.so.9",
            "libcudnn_engines_runtime_compiled.so.9", "libcudnn_graph.so.9", "libcudnn_heuristic.so.9")
        files = [{"role": role, "filename": name, "bytes": 4, "sha256": h(name)} for role, name in (
            ("core", "libonnxruntime.so.1.22.0"), ("providers_shared", "libonnxruntime_providers_shared.so"),
            ("providers_cuda", "libonnxruntime_providers_cuda.so"))]
        files.extend({"role": "nvidia_dependency", "filename": name, "bytes": 4, "sha256": h(name)} for name in filenames)
        bundle = {"schema_version": 2, "files": files,
            "loader_aliases": [{"filename": "libcudnn.so.9.8.0", "target": "libcudnn.so.9"}]}
        source = synthetic_v2()
        recipe, manifest, record, lock, _binding = source
        files[0]["sha256"] = manifest["runtime_sha256"]
        bundle_bytes = raw(bundle)
        raw_sha = hashlib.sha256(bundle_bytes).hexdigest()
        semantic_sha = historical._runtime_bundle_digest(bundle)
        self.assertNotEqual(raw_sha, semantic_sha)
        lock["manifest"]["provenance"] = [artifact("runtime.json", raw_sha, len(bundle_bytes))]
        lock["canonical_sha256"] = data.digest("rz-rovent-execution-canonical/1", lock["manifest"])
        admission = record["native"]["raw_retirement"]["root_admission"]
        admission["native_origin"].update(input_lock_sha256=lock["canonical_sha256"], runtime_bundle_sha256=semantic_sha)
        admission["native_origin_sha256"] = data.digest(native_v2.ORIGIN_DOMAIN, admission["native_origin"])
        record["native"]["raw_retirement"]["root_admission_sha256"] = data.digest(native_v2.ROOT_DOMAIN, admission)
        reseal_native(manifest, record)
        with tempfile.TemporaryDirectory() as directory:
            registration, publication, _, _, _, _ = write_publication(Path(directory), source=source, runtime_bundle=bundle)
            budget = historical.HistoricalReadBudget(8 * 1024**2)
            owner = historical.HistoricalRoventCorpus.from_registration(registration, read_budget=budget)
            self.assertFalse(owner.observation()["execution_authorized"])
            self.assertGreaterEqual(budget.observation()["charged_bytes"], len(bundle_bytes))
            self.assertFalse(budget.observation()["unknown"])
            duplicate = copy.deepcopy(publication["admissions"][0])
            duplicate["native_admission_sha256"] = h("second root sharing one bundle")
            publication["admissions"].append(duplicate)
            publication["admissions"].sort(key=lambda item: item["native_admission_sha256"])
            publication["native_admission_sha256"] = [item["native_admission_sha256"] for item in publication["admissions"]]
            historical._publication(publication, original.CorpusLimits())
            duplicate["runtime_bundle_manifest"]["bytes"] += 1
            with self.assertRaisesRegex(ValueError, "conflicting file pins"):
                historical._publication(publication, original.CorpusLimits())

    def test_cuda_binary_codec_rejects_alias_escape_nonprofile_and_file_mismatch(self):
        for value in ({"schema_version": 1, "files": []}, {"schema_version": 2, "files": [], "loader_aliases": []},
                      {"schema_version": 3, "files": []}):
            with self.subTest(schema=value["schema_version"]), self.assertRaises(ValueError):
                historical._runtime_bundle_digest(value)

    def test_shared_read_budget_reuses_the_original_bootstrap_ceiling(self):
        from rz_pals_model.rovent_training_bootstrap import MAX_READ_BYTES
        self.assertEqual(historical._MAX_SHARED_READ_BYTES, MAX_READ_BYTES)
        budget = historical.HistoricalReadBudget(MAX_READ_BYTES)
        self.assertEqual(budget.observation()["maximum_bytes"], MAX_READ_BYTES)
        with self.assertRaises(ValueError):
            historical.HistoricalReadBudget(MAX_READ_BYTES + 1)

    def setUp(self):
        self.guard = NeuralImportsForbidden()
        sys.meta_path.insert(0, self.guard)

    def tearDown(self):
        sys.meta_path.remove(self.guard)

    def test_pinned_publication_reuses_original_reader_with_exact_shared_read_charges(self):
        with tempfile.TemporaryDirectory() as directory:
            registration, publication, _, _, record, payloads = write_publication(Path(directory))
            budget = historical.HistoricalReadBudget(8 * 1024**2)
            owner = historical.HistoricalRoventCorpus.from_registration(registration, read_budget=budget)
            self.assertIs(type(owner.corpus), original.RoventTrainingCorpus)
            report = owner.observation()
            self.assertEqual(report["record_count"], 1)
            self.assertEqual(report["native_admission_sha256"], publication["native_admission_sha256"])
            self.assertEqual(owner.corpus._records[0]["input_sha256"], record["input_sha256"])
            self.assertEqual(report["collection_runtime_library_sha256"], h("actual loaded ORT library"))
            self.assertNotEqual(report["collection_runtime_library_sha256"], h("Python trainer runtime"))
            self.assertNotEqual(report["collection_runtime_library_sha256"], h("runtime endpoint implementation"))
            self.assertNotEqual(report["collection_source_sha256"], report["collection_runtime_library_sha256"])
            self.assertEqual(report["rules_replay_scope"], "rust_parent_validation_required")
            self.assertIs(report["execution_authorized"], False)
            self.assertIs(report["past_native_authority_reissued"], False)
            self.assertIs(report["past_physical_completion_reissued"], False)
            original_bytes = sum(len(payloads[name]) for name in
                                 ("corpus-manifest.json", "corpus-index.json", "chunks/000000.jsonl"))
            expected_read = len(payloads["publication.json"]) + publication["total_payload_bytes"] + original_bytes
            self.assertEqual(budget.observation(), {"charged_bytes": expected_read, "unknown": False,
                                                    "maximum_bytes": 8 * 1024**2})
            report["role_bindings"]["proposer"]["weights_sha256"] = h("caller mutation")
            self.assertNotEqual(owner.observation()["role_bindings"], report["role_bindings"])

    def test_old_native_domain_is_preserved_and_not_converted_for_historical_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            _, manifest, row = original_fixture()
            owner = write_corpus(Path(directory), manifest, [row])
            self.assertEqual(owner._records[0]["native"]["raw_submission"]["schema"], original.NATIVE_RAW_DOMAIN)
            with self.assertRaisesRegex(ValueError, "requires explicit native raw /2; no conversion"):
                historical._join_record({"role_bindings": {}}, row, {})

    def test_all_published_chunks_are_hashed_before_original_owner_is_created(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registration, _, _, _, _, payloads = write_publication(root)
            chunk = root / "chunks" / "000000.jsonl"
            payload = payloads["chunks/000000.jsonl"]
            chunk.write_bytes(payload.replace(b'"accepted_retired":true', b'"accepted_retired":null', 1))
            budget = historical.HistoricalReadBudget(8 * 1024**2)
            with self.assertRaisesRegex(ValueError, "raw byte pin/identity mismatch"):
                historical.HistoricalRoventCorpus.from_registration(registration, read_budget=budget)
            self.assertTrue(budget.observation()["unknown"])
            self.assertGreaterEqual(budget.observation()["charged_bytes"], len(payload))

    def test_same_byte_count_bad_publication_hash_records_real_read_and_unknown(self):
        with tempfile.TemporaryDirectory() as directory:
            registration, _, _, _, _, payloads = write_publication(Path(directory))
            registration["publication"]["sha256"] = h("different publication")
            budget = historical.HistoricalReadBudget(8 * 1024**2)
            with self.assertRaisesRegex(ValueError, "raw byte pin/identity mismatch"):
                historical.HistoricalRoventCorpus.from_registration(registration, read_budget=budget)
            self.assertEqual(budget.observation()["charged_bytes"], len(payloads["publication.json"]))
            self.assertTrue(budget.observation()["unknown"])

    def test_read_credit_is_reserved_before_a_file_descriptor_is_opened(self):
        with tempfile.TemporaryDirectory() as directory:
            registration, _, _, _, _, _ = write_publication(Path(directory))
            budget = historical.HistoricalReadBudget(registration["publication"]["bytes"])
            with mock.patch.object(original.os, "open", side_effect=AssertionError("must reserve before open")) as opened:
                with self.assertRaisesRegex(ValueError, "read reservation exhausted"):
                    historical.HistoricalRoventCorpus.from_registration(registration, read_budget=budget)
            opened.assert_not_called()
            self.assertEqual(budget.observation()["charged_bytes"], 0)

    def test_publication_relative_paths_reject_traversal_absolute_and_platform_escape(self):
        for path in ("../chunks/x.jsonl", "/chunks/x.jsonl", "C:/chunks/x.jsonl", "chunks\\x.jsonl", "chunks//x.jsonl"):
            pin = {"path": path, "sha256": h("tiny chunk"), "bytes": 1}
            with self.subTest(path=path):
                with self.assertRaisesRegex(ValueError, "normalized relative historical publication file required"):
                    historical._relative_pin(pin, maximum=1024, description="test pin")

    def test_collection_library_runtime_cannot_be_replaced_by_python_endpoint_or_source(self):
        for impostor in (h("Python trainer runtime"), h("runtime endpoint implementation"), h("compiled source aggregate")):
            with tempfile.TemporaryDirectory() as directory:
                registration, publication, manifest, index, _, _ = write_publication(Path(directory))
                registration["collection_runtime_library_sha256"] = impostor
                with self.subTest(impostor=impostor):
                    with self.assertRaisesRegex(ValueError, "collection source/runtime file domain mismatch"):
                        historical._join_metadata(registration, publication, manifest, index)

    def test_role_binding_names_keep_model_weights_encoding_and_semantics_independent(self):
        for field in ("model_semantics_sha256", "model_identity_sha256", "weights_sha256", "encoding_sha256"):
            with tempfile.TemporaryDirectory() as directory:
                registration, _, _, _, _, _ = write_publication(Path(directory))
                registration["role_bindings"]["proposer"][field] = h("different-" + field)
                with self.subTest(field=field):
                    with self.assertRaisesRegex(ValueError, "registered collection role model/weights/encoding"):
                        historical.HistoricalRoventCorpus.from_registration(registration,
                            read_budget=historical.HistoricalReadBudget(8 * 1024**2))

    def test_publication_canonical_index_epoch_and_recipe_joins_are_exact(self):
        with tempfile.TemporaryDirectory() as directory:
            registration, publication, manifest, index, _, _ = write_publication(Path(directory))
            broken = copy.deepcopy(publication)
            broken["canonical_index_sha256"] = h("different canonical index")
            with self.assertRaisesRegex(ValueError, "canonical manifest/index mismatch"):
                historical._join_metadata(registration, broken, manifest, index)
            for field, value in (("model_epoch", 2), ("context_epoch", 2), ("recipe_sha256", h("different recipe")),
                                 ("stage_id", "different-stage")):
                broken = {**registration, field: value}
                with self.subTest(field=field):
                    with self.assertRaisesRegex(ValueError, "manifest model/epoch/recipe/stage mismatch"):
                        historical._join_metadata(broken, publication, manifest, index)

    def test_shared_input_lock_is_counted_once_and_conflicting_artifacts_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            _, publication, _, _, _, _ = write_publication(Path(directory))
            second = copy.deepcopy(publication["admissions"][0])
            second["native_admission_sha256"] = h("another root admission")
            publication["admissions"].append(second)
            publication["admissions"].sort(key=lambda entry: entry["native_admission_sha256"])
            publication["native_admission_sha256"] = [entry["native_admission_sha256"] for entry in publication["admissions"]]
            self.assertIs(historical._publication(publication, original.CorpusLimits()), publication)
            second["input_lock"]["path"] = "input-locks/another.json"
            with self.assertRaisesRegex(ValueError, "one canonical input lock must use one artifact path"):
                historical._publication(publication, original.CorpusLimits())

    def test_explicit_ancestor_set_cannot_be_silently_omitted(self):
        with tempfile.TemporaryDirectory() as directory:
            source = synthetic_v2()
            source[1]["ancestor_index_sha256"] = [h("missing ancestor index")]
            registration, _, _, _, _, _ = write_publication(Path(directory), source=source)
            with self.assertRaisesRegex(ValueError, "original ancestor publication set mismatch"):
                historical.HistoricalRoventCorpus.from_registration(registration,
                    read_budget=historical.HistoricalReadBudget(8 * 1024**2))

    def test_native_v2_rejects_missing_replay_and_unknown_original_root_fields(self):
        _, manifest, record, _, _ = synthetic_v2()
        del record["native"]["raw_submission"]["rules_replay"]
        with self.assertRaisesRegex(ValueError, "native /2 original raw_submission fields mismatch"):
            original._native(manifest, record)
        _, manifest, record, _, _ = synthetic_v2()
        record["native"]["raw_retirement"]["root_admission"]["execution_authorized"] = True
        # Shape validation must reject this unknown authority field before a
        # matching old digest could be mistaken for a current capability.
        with self.assertRaisesRegex(ValueError, "native /2 original root admission fields mismatch"):
            native_v2._root_admission(manifest, record, record["native"]["raw_retirement"])

    def test_native_v2_mixed_domains_and_history_are_not_silently_rewritten(self):
        _, manifest, record, _, _ = synthetic_v2()
        record["native"]["raw_completion"]["schema"] = original.NATIVE_RAW_DOMAIN
        with self.assertRaisesRegex(ValueError, "mixed native raw versions rejected"):
            original._native(manifest, record)
        _, manifest, record, _, _ = synthetic_v2()
        replay = record["native"]["raw_submission"]["rules_replay"]
        replay["actual_root"]["moves"] = [128]
        reseal_native(manifest, record)
        with self.assertRaisesRegex(ValueError, "original actual history differs from anchor zero"):
            original._native(manifest, record)

    def test_native_v2_wdl_trace_is_only_a_metadata_join_and_requires_exact_trajectory(self):
        _, _, record, _, _ = synthetic_v2()
        add_rules_wdl(record)
        source = record["head_sources"]["wdl"]
        label = record["targets"]["wdl"]
        replay = record["native"]["raw_submission"]["rules_replay"]["actual_root"]
        source["raw_receipt"]["rules_replay"] = {"schema": native_v2.TRAJECTORY_DOMAIN,
            "initial": copy.deepcopy(replay), "actual_selected_moves": [], "final_position": copy.deepcopy(replay),
            "completion": "automatic", "claim": None}
        # This checks shape/seal-independent joins, not legal chess or a real
        # checkmate. The historical owner still requires Rust parent replay.
        self.assertIsNone(native_v2.validate_wdl_replay_v2(record, source["raw_receipt"], label))
        source["raw_receipt"]["rules_replay"]["actual_selected_moves"] = [128]
        with self.assertRaisesRegex(ValueError, "does not join original trajectory anchors/history"):
            native_v2.validate_wdl_replay_v2(record, source["raw_receipt"], label)

    def test_duplicate_publication_json_and_bool_byte_pin_are_rejected_at_their_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registration, _, _, _, _, payloads = write_publication(root)
            changed = payloads["publication.json"].replace(b'"record_count":1', b'"record_count":1,"record_count":2')
            (root / "publication.json").write_bytes(changed)
            registration["publication"] = file_pin(str(root / "publication.json"), changed)
            with self.assertRaisesRegex(ValueError, "duplicate JSON field in original corpus"):
                historical.HistoricalRoventCorpus.from_registration(registration,
                    read_budget=historical.HistoricalReadBudget(8 * 1024**2))
            registration["publication"]["bytes"] = True
            with self.assertRaisesRegex(ValueError, "invalid bounded unsigned integer"):
                historical.HistoricalRoventCorpus.from_registration(registration,
                    read_budget=historical.HistoricalReadBudget(8 * 1024**2))


class SubmittedHistoricalCorpusTests(unittest.TestCase):
    def setUp(self):
        self.guard = NeuralImportsForbidden()
        sys.meta_path.insert(0, self.guard)

    def tearDown(self):
        sys.meta_path.remove(self.guard)

    def test_explicit_submitted_publication_uses_new_constructor_and_preserves_original_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registration, publication, _, _, record, payloads = write_submitted_publication(root)
            budget = historical.HistoricalReadBudget(8 * 1024**2)
            with mock.patch.object(original.RoventTrainingCorpus, "from_files",
                    side_effect=AssertionError("submitted bytes must not enter the legacy constructor")):
                owner = historical.HistoricalRoventCorpus.from_registration(registration, read_budget=budget)
            self.assertIs(type(owner.corpus), original.RoventTrainingCorpus)
            self.assertIs(owner.corpus._codec, codecs.SUBMITTED_CODEC)
            self.assertEqual(owner.corpus._records[0], record)
            self.assertEqual(owner.corpus.manifest_semantic_sha256, publication["canonical_manifest_sha256"])
            report = owner.observation()
            for key in ("execution_authorized", "past_native_authority_reissued", "past_physical_completion_reissued"):
                self.assertIs(report[key], False)
            review = record["native"]["raw_submission"]["submitted_review"]
            self.assertNotEqual(review["input_sha256"], record["native"]["input_digest"])
            self.assertNotEqual(review["input_sha256"], record["input_sha256"])
            original_bytes = sum(len(payloads[name]) for name in
                ("corpus-manifest.json", "corpus-index.json", "chunks/000000.jsonl"))
            self.assertEqual(budget.observation(), {"charged_bytes": len(payloads["publication.json"])
                + publication["total_payload_bytes"] + original_bytes, "unknown": False, "maximum_bytes": 8 * 1024**2})
            for name, payload in payloads.items():
                self.assertEqual(root.joinpath(*name.split("/")).read_bytes(), payload)

    def test_original_and_submitted_publication_parsers_reject_each_other_without_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, old, _, _, _, _ = write_publication(root / "old")
            _, new, _, _, _, _ = write_submitted_publication(root / "new")
            before = copy.deepcopy((old, new))
            with self.assertRaises(ValueError):
                historical._publication(new, original.CorpusLimits())
            with self.assertRaises(ValueError):
                historical._submitted_publication(old, original.CorpusLimits())
            self.assertEqual((old, new), before)
            self.assertIs(historical._publication(old, original.CorpusLimits()), old)
            self.assertIs(historical._submitted_publication(new, original.CorpusLimits()), new)

    def test_submitted_publication_rejects_mixed_containers_and_legacy_canonical_seals(self):
        with tempfile.TemporaryDirectory() as directory:
            registration, publication, manifest, index, _, _ = write_submitted_publication(Path(directory))
            for change in ("manifest", "index", "publication", "canonical_manifest", "canonical_index"):
                reg, pub, body, rows = copy.deepcopy((registration, publication, manifest, index))
                if change == "manifest":
                    body["schema"] = original.MANIFEST_DOMAIN
                elif change == "index":
                    rows["schema"] = original.INDEX_DOMAIN
                elif change == "publication":
                    pub["schema"] = historical.PUBLICATION_DOMAIN
                elif change == "canonical_manifest":
                    pub["canonical_manifest_sha256"] = data.digest(original.MANIFEST_DOMAIN, body)
                    rows["manifest_sha256"] = pub["canonical_manifest_sha256"]
                else:
                    pub["canonical_index_sha256"] = data.digest(original.INDEX_DOMAIN, rows)
                with self.subTest(change=change), self.assertRaises(ValueError):
                    historical._join_submitted_metadata(reg, pub, body, rows)

    def test_submitted_record_requires_every_original_raw3_body_and_selected_review(self):
        _, manifest, record, _, _ = synthetic_submitted()
        for change in ("raw_submission", "raw_completion", "raw_retirement", "missing_review", "view_sequence",
                       "root_state", "source_anchors"):
            bad = copy.deepcopy(record)
            review = bad["native"]["raw_submission"]["submitted_review"]
            if change.startswith("raw_"):
                bad["native"][change]["schema"] = native_v2.NATIVE_DOMAIN
            elif change == "missing_review":
                bad["native"]["raw_submission"].pop("submitted_review")
            elif change == "view_sequence":
                review["sequence"] += 1
            elif change == "root_state":
                review["root_state"] += 1
            else:
                review["source_anchors"] = [bad["native"]["raw_submission"]["rules_replay"]["actual_root"]]
            reseal_submitted(manifest, bad)
            with self.subTest(change=change), self.assertRaises(ValueError):
                historical._join_submitted_record({}, manifest, bad, {})

    def test_new_historical_intake_retains_exact_old_ancestor_index_domain(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registration, _, _, _, _, _ = write_publication(root / "old")
            ancestor = historical.HistoricalRoventCorpus.from_registration(registration,
                read_budget=historical.HistoricalReadBudget(8 * 1024**2))
            source = synthetic_submitted()
            source[1]["ancestor_index_sha256"] = [ancestor.corpus.index_semantic_sha256]
            registration, _, _, _, _, _ = write_submitted_publication(root / "new", source=source)
            owner = historical.HistoricalRoventCorpus.from_registration(registration, ancestors=(ancestor,),
                read_budget=historical.HistoricalReadBudget(8 * 1024**2))
            self.assertIs(owner.corpus._codec, codecs.SUBMITTED_CODEC)
            self.assertIs(owner.corpus._ancestors[0]._codec, codecs.LEGACY_CODEC)
            self.assertEqual(owner.observation()["ancestor_index_sha256"], [ancestor.corpus.index_semantic_sha256])
            self.assertEqual(ancestor.corpus.index_semantic_sha256,
                data.digest(original.INDEX_DOMAIN, ancestor.corpus._index))
            self.assertNotEqual(ancestor.corpus.index_semantic_sha256,
                data.digest("rz-rovent-corpus-index/2", ancestor.corpus._index))

    def test_submitted_historical_bytes_do_not_bypass_hash_or_shared_credit_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registration, _, _, _, _, payloads = write_submitted_publication(root)
            budget = historical.HistoricalReadBudget(registration["publication"]["bytes"])
            with mock.patch.object(original.os, "open", side_effect=AssertionError("reserve before open")) as opened:
                with self.assertRaisesRegex(ValueError, "read reservation exhausted"):
                    historical.HistoricalRoventCorpus.from_registration(registration, read_budget=budget)
            opened.assert_not_called()
            chunk = root / "chunks" / "000000.jsonl"
            changed = payloads["chunks/000000.jsonl"].replace(b'"accepted_retired":true', b'"accepted_retired":null', 1)
            self.assertNotEqual(changed, payloads["chunks/000000.jsonl"])
            chunk.write_bytes(changed)
            budget = historical.HistoricalReadBudget(8 * 1024**2)
            with self.assertRaisesRegex(ValueError, "raw byte pin/identity mismatch"):
                historical.HistoricalRoventCorpus.from_registration(registration, read_budget=budget)
            self.assertIs(budget.observation()["unknown"], True)


if __name__ == "__main__":
    unittest.main()
