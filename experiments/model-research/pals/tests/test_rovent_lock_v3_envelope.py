import copy
import hashlib
import unittest

from rz_pals_model.rovent_lock_v3_envelope import (
    CPU_PUBLICATION, CUDA_PUBLICATION, validate_publication_envelope,
)


def _hash(data):
    return hashlib.sha256(data.encode()).hexdigest()


def _pin(path, data):
    return {"path": path, "sha256": _hash(data), "bytes": len(data.encode())}


def _publication(cpu):
    manifest = _pin("corpus-manifest.json", "manifest")
    index = _pin("corpus-index.json", "index")
    chunk = _pin("chunks/one.jsonl", "row")
    lock = _pin("input-locks/one.json", "lock")
    runtime = _hash("cpu runtime")
    admission = {
        "native_admission_sha256": _hash("native admission"),
        "input_lock": lock,
        "input_lock_bytes_origin": "validated_semantic_snapshot",
        "canonical_input_lock_sha256": _hash("canonical lock"),
        ("cpu_runtime_library_sha256" if cpu else "system_runtime_identity_sha256"):
            (runtime if cpu else _hash("cuda runtime")),
    }
    return {"schema": CPU_PUBLICATION if cpu else CUDA_PUBLICATION,
            "manifest": manifest, "index": index, "chunks": [chunk],
            "canonical_manifest_sha256": _hash("canonical manifest"),
            "canonical_index_sha256": _hash("canonical index"),
            "source_commit": hashlib.sha1(b"test source").hexdigest(),
            "source_sha256": _hash("source"), "runtime_sha256": runtime,
            "record_count": 1,
            "total_payload_bytes": sum(x["bytes"] for x in (manifest, index, chunk, lock)),
            "native_admission_sha256": [admission["native_admission_sha256"]],
            "admissions": [admission]}


def _check(value):
    return validate_publication_envelope(value, max_file_bytes=1024,
                                         max_total_bytes=4096, max_records=10)


def _wire_row(cpu):
    from rz_pals_model.generation_data import digest
    family = "cpu-" if cpu else ""
    record_domain = f"rz-rovent-training-record-{family}lock-v3/1"
    raw_domain = f"rz-rovent-native-{family}lock-v3-raw/1"
    root_domain = f"rz-rovent-collection-root-admission-{family}lock-v3/1"
    origin_domain = f"rz-rovent-collection-native-origin-{family}lock-v3/1"
    origin = {"schema": origin_domain}
    root = {"schema": root_domain, "native_origin": origin,
            "native_origin_sha256": digest(origin_domain, origin)}
    events = {name: {"schema": raw_domain, "kind": kind} for name, kind in
              (("raw_submission", "submitted"), ("raw_completion", "completed"), ("raw_retirement", "retired"))}
    events["raw_retirement"].update(root_admission=root, root_admission_sha256=digest(root_domain, root))
    native = dict(events)
    for key, event in events.items():
        native[key + "_sha256"] = digest(raw_domain, event)
    native.update(process_epoch=1, game_generation=1, root_generation=1, request_sequence=1)
    row = {"schema": record_domain, "role": "proposer", "split": "train",
           "input": {"game_id": "g", "opening_id": "o", "line_genealogy_id": "l",
                     "transposition_sha256": _hash("state")},
           "input_sha256": _hash("input"), "origin_sha256": _hash("origin"), "record_sha256": "",
           "lineage": {"actor_id": "a", "teacher_actor_id": None, "ancestor_origin_sha256": []},
           "native": native, "head_sources": {}, "targets": {}, "divergence_index": None,
           "nnuec_input": None, "nnuec_input_sha256": None, "nnuec_model_sha256": None,
           "nnuec_semantics": None, "nnuec_wdl": None}
    projected = dict(row, input={"sealed_input_sha256": row["input_sha256"]})
    row["record_sha256"] = digest(record_domain, projected)
    return row


class LockV3RecordSealTests(unittest.TestCase):
    def test_family_and_seals_are_checked_without_training_admission(self):
        from rz_pals_model.rovent_lock_v3_rows import inspect_record_seals
        for cpu in (True, False):
            row = _wire_row(cpu)
            self.assertIsNone(inspect_record_seals(row, "cpu" if cpu else "cuda"))
            with self.assertRaises(ValueError):
                inspect_record_seals(row, "cuda" if cpu else "cpu")
            row["native"]["raw_submission"]["kind"] = "tampered"
            with self.assertRaises(ValueError):
                inspect_record_seals(row, "cpu" if cpu else "cuda")

    def test_unknown_record_field_rejected(self):
        from rz_pals_model.rovent_lock_v3_rows import inspect_record_seals
        row = _wire_row(True)
        row["unexpected"] = True
        with self.assertRaises(ValueError):
            inspect_record_seals(row, "cpu")


class LockV3EnvelopeTests(unittest.TestCase):
    def test_cpu_and_cuda_stay_distinct_and_are_never_training_admitted(self):
        for cpu in (True, False):
            value = _publication(cpu)
            original = copy.deepcopy(value)
            result = _check(value)
            self.assertEqual(result.family, "cpu" if cpu else "cuda")
            self.assertFalse(result.training_eligible)
            self.assertEqual(value, original)

    def test_cross_runtime_spoof_and_unknown_fields_rejected(self):
        for cpu in (True, False):
            value = _publication(cpu)
            value["admissions"][0]["system_runtime_identity_sha256" if cpu
                                     else "cpu_runtime_library_sha256"] = _hash("wrong family")
            with self.assertRaises(ValueError):
                _check(value)
            value = _publication(cpu)
            value["extra"] = 1
            with self.assertRaises(ValueError):
                _check(value)
        value = _publication(True)
        value["admissions"][0]["cpu_runtime_library_sha256"] = _hash("wrong runtime")
        with self.assertRaises(ValueError):
            _check(value)

    def test_reject_bad_origin_overlap_and_payload_count(self):
        for case in ("origin", "overlap", "bytes"):
            value = _publication(True)
            if case == "origin":
                value["admissions"][0]["input_lock_bytes_origin"] = "assumed_original"
            elif case == "overlap":
                value["chunks"].append(copy.deepcopy(value["chunks"][0]))
            else:
                value["total_payload_bytes"] += 1
            with self.subTest(case=case), self.assertRaises(ValueError):
                _check(value)

    def test_legacy_publication_is_not_silently_converted(self):
        value = _publication(True)
        value["schema"] = "rz-rovent-corpus-publication/2"
        with self.assertRaises(ValueError):
            _check(value)


class LockV3PinnedMetadataTests(unittest.TestCase):
    def test_cpu_metadata_uses_pinned_bytes_and_remains_ineligible(self):
        import json
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace
        from rz_pals_model.generation_data import digest
        from rz_pals_model.rovent_historical_corpus import HistoricalReadBudget, _source_aggregate
        from rz_pals_model.rovent_lock_v3_envelope import inspect_pinned_metadata, inspect_pinned_row_seals

        limits = SimpleNamespace(max_file_bytes=16384, max_total_bytes=65536, max_records=10)
        publication = _publication(True)
        row = _wire_row(True)
        component = {"engine": _hash("engine")}
        publication["source_sha256"] = _source_aggregate(publication["source_commit"], component)
        runtime = {"kind": "cpu"}
        runtime_lock = {"domain": "rz-system-runtime-lock-v6/1", "runtime": runtime,
                        "canonical_sha256": digest("rz-system-runtime-lock-v6/1", runtime)}
        body = {"domain": "rz-rovent-execution/3", "contract_revision": "0.1", "run_id": "one",
                "purpose": "collection", "engine": {"source_commit": publication["source_commit"],
                "source_components": component, "pc": {"backend": "ort_cpu", "device": {"device": "cpu"},
                "runtime_artifacts": [{"sha256": publication["runtime_sha256"]}]}},
                "resources": {}, "coordinator": {}, "wall_time_max_ms": 1, "cleanup_max_ms": 1,
                "output_bytes_max": 1, "agent_renewable": False, "provenance": [], "startup_document": {},
                "startup_configuration_sha256": _hash("startup"),
                "startup_search_configuration_sha256": _hash("search"), "own_r_root_tie": {},
                "system_runtime_lock": runtime_lock}
        lock_sha = digest("rz-rovent-execution-canonical/3", body)
        lock = {"domain": "rz-rovent-input-lock/3", "manifest": body, "canonical_sha256": lock_sha}
        retirement = row["native"]["raw_retirement"]
        row_root = retirement["root_admission"]
        row_origin = row_root["native_origin"]
        row_origin["input_lock_sha256"] = lock_sha
        row_root["native_origin_sha256"] = digest(row_origin["schema"], row_origin)
        retirement["root_admission_sha256"] = digest(row_root["schema"], row_root)
        row["native"]["raw_retirement_sha256"] = digest(retirement["schema"], retirement)
        projected = dict(row, input={"sealed_input_sha256": row["input_sha256"]}, record_sha256="")
        row["record_sha256"] = digest(row["schema"], projected)
        seal = retirement["root_admission_sha256"]
        publication["native_admission_sha256"] = [seal]
        publication["admissions"][0]["native_admission_sha256"] = seal
        row_bytes = (json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode()
        publication["chunks"][0] = {"path": "chunks/one.jsonl", "sha256": hashlib.sha256(row_bytes).hexdigest(),
                                    "bytes": len(row_bytes)}
        indexed_origin = {key: row[key] for key in ("record_sha256", "input_sha256", "origin_sha256", "split")}
        indexed_origin.update({key: row["input"][key] for key in
                               ("game_id", "opening_id", "line_genealogy_id", "transposition_sha256")})
        indexed_origin.update(row["lineage"])
        registration = {"schema": "rz-rovent-corpus-manifest/2", "corpus_id": "one", "source_commit": publication["source_commit"],
            "source_sha256": publication["source_sha256"], "runtime_sha256": publication["runtime_sha256"],
            "model_semantics_sha256": _hash("model"), "model_epoch": 1, "context_epoch": 1,
            "recipe_sha256": _hash("recipe"), "stage_id": "stage", "round": 0, "round_id": "round",
            "student": "student", "learner": "learner", "one_learner": True, "inherited_input_sha256": [],
            "inherited_origin_sha256": [], "teachers": {}, "opening_splits": {"start": "train"},
            "ancestor_index_sha256": []}
        manifest = {"schema": "rz-rovent-corpus-manifest-cpu-lock-v3/1", "registration": registration,
                    "startup_lock_sha256": lock_sha}
        publication["canonical_manifest_sha256"] = digest(manifest["schema"], manifest)
        chunk = publication["chunks"][0]
        index = {"schema": "rz-rovent-corpus-index-cpu-lock-v3/1",
                 "manifest_sha256": publication["canonical_manifest_sha256"], "record_count": 1,
                 "chunks": [{**chunk, "records": 1}], "origins": [indexed_origin]}
        publication["canonical_index_sha256"] = digest(index["schema"], index)
        publication["admissions"][0]["canonical_input_lock_sha256"] = lock_sha
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for key, obj in (("manifest", manifest), ("index", index), ("input_lock", lock)):
                ref = publication["admissions"][0][key] if key == "input_lock" else publication[key]
                raw = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
                target = root / ref["path"]
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(raw)
                ref["sha256"] = hashlib.sha256(raw).hexdigest()
                ref["bytes"] = len(raw)
            publication["total_payload_bytes"] = sum(p["bytes"] for p in
                (publication["manifest"], publication["index"], chunk, publication["admissions"][0]["input_lock"]))
            chunk_path = root / chunk["path"]
            chunk_path.parent.mkdir(parents=True, exist_ok=True)
            chunk_path.write_bytes(row_bytes)
            result = inspect_pinned_metadata(root, publication, read_budget=HistoricalReadBudget(65536), limits=limits)
            self.assertEqual(result.family, "cpu")
            self.assertEqual(result.locks_checked, 1)
            self.assertEqual(result.chunks_checked, 0)
            self.assertFalse(result.training_eligible)
            row_observation = inspect_pinned_row_seals(root, publication, read_budget=HistoricalReadBudget(65536), limits=limits)
            self.assertEqual(row_observation.chunks_checked, 1)
            self.assertEqual(row_observation.lifecycle_checked, 0)
            self.assertFalse(row_observation.training_eligible)
            # A synthetically sealed row still lacks actual typed submitted input.
            with self.assertRaises(ValueError):
                inspect_pinned_row_seals(root, publication, read_budget=HistoricalReadBudget(65536),
                                         limits=limits, inspect_lifecycle=True)
            chunk_path.write_bytes(b"tampered")
            with self.assertRaises(ValueError):
                inspect_pinned_row_seals(root, publication, read_budget=HistoricalReadBudget(65536), limits=limits)
            # Rebuild the same metadata as a distinct CUDA family. Its observed
            # system identity is not the runtime lock canonical digest.
            publication["schema"] = CUDA_PUBLICATION
            config = {"system_identity_sha256": _hash("cuda system identity"),
                      "libraries": [{"role": "ort-core", "sha256": publication["runtime_sha256"]}]}
            runtime = {"kind": "system_cuda", "configuration": config}
            runtime_lock["runtime"] = runtime
            runtime_lock["canonical_sha256"] = digest(runtime_lock["domain"], runtime)
            body["engine"]["pc"] = {"backend": "ort_cuda", "device": {"device": "cuda", "ordinal": 0}}
            lock_sha = digest("rz-rovent-execution-canonical/3", body)
            lock["canonical_sha256"] = lock_sha
            manifest["schema"] = "rz-rovent-corpus-manifest-lock-v3/1"
            manifest["startup_lock_sha256"] = lock_sha
            publication["canonical_manifest_sha256"] = digest(manifest["schema"], manifest)
            index["schema"] = "rz-rovent-corpus-index-lock-v3/1"
            index["manifest_sha256"] = publication["canonical_manifest_sha256"]
            publication["canonical_index_sha256"] = digest(index["schema"], index)
            admission = publication["admissions"][0]
            del admission["cpu_runtime_library_sha256"]
            admission["system_runtime_identity_sha256"] = config["system_identity_sha256"]
            admission["canonical_input_lock_sha256"] = lock_sha
            for key, obj in (("manifest", manifest), ("index", index), ("input_lock", lock)):
                ref = admission[key] if key == "input_lock" else publication[key]
                raw = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
                (root / ref["path"]).write_bytes(raw)
                ref["sha256"] = hashlib.sha256(raw).hexdigest()
                ref["bytes"] = len(raw)
            publication["total_payload_bytes"] = sum(p["bytes"] for p in
                (publication["manifest"], publication["index"], chunk, admission["input_lock"]))
            result = inspect_pinned_metadata(root, publication, read_budget=HistoricalReadBudget(65536), limits=limits)
            self.assertEqual(result.family, "cuda")
            self.assertFalse(result.training_eligible)
            (root / publication["manifest"]["path"]).write_bytes(b"tampered")
            with self.assertRaises(ValueError):
                inspect_pinned_metadata(root, publication, read_budget=HistoricalReadBudget(65536), limits=limits)


class LockV3RootIdentityTests(unittest.TestCase):
    def test_cpu_provider_sessions_and_lock_join(self):
        from rz_pals_model.rovent_lock_v3_rows import inspect_root_runtime_identity
        row = _wire_row(True)
        model_sha, encoding_sha, runtime_sha = _hash("weights"), _hash("encoding"), _hash("runtime")
        config_sha, lock_sha = _hash("backend configuration"), _hash("input lock")
        model = {"semantics": "model", "encoding": "encoding", "weights": {"sha256": model_sha}}
        pc = {"backend": "ort_cpu", "device": {"device": "cpu"}, "proposer": model,
              "validator": {"present": True}, "runtime_artifacts": [{"sha256": runtime_sha}],
              "backend_configuration_sha256": config_sha}
        engine = {"mode": "cpu_pc", "researcher": None, "source_commit": "a" * 40, "source_components": {"engine": _hash("engine")},
                  "pc": pc, "teachers": []}
        body = {"engine": engine, "startup_configuration_sha256": _hash("startup"),
                "startup_search_configuration_sha256": _hash("search"),
                "system_runtime_lock": {"runtime": {"kind": "cpu"}}}
        lock = {"manifest": body, "canonical_sha256": lock_sha}
        declaration = {"purpose": "collection", "mode": engine["mode"], "source_commit": engine["source_commit"],
                       "source_components": engine["source_components"], "model_semantics": model["semantics"],
                       "encoding": model["encoding"], "validator_model": pc["validator"], "teacher_profiles": []}
        startup = {"domain": "rz-rovent-native-startup/1", "process_epoch": 1, "mode": "cpu_pc", "topology": "roles_only",
                   "weights_sha256": model_sha, "export_manifest_sha256": _hash("export"), "backend_sha256": _hash("backend"),
                   "runtime_library_sha256": runtime_sha, "provider_ready": True, "provider": "cpu",
                   "cpu_session_roles": ["public", "shared_pc", "validator"], "native_configuration_sha256": config_sha,
                   "startup_configuration_sha256": body["startup_configuration_sha256"],
                   "search_configuration_sha256": body["startup_search_configuration_sha256"], "validator_loaded": True,
                   "physical_workers_created": 1, "neural_service_executed": False}
        namespace = {"value_semantics": "s", "value_model": "m", "value_encoding": "e", "value_precision": "fp32_tf32_off",
                     "representation_model_domain": "loaded_checkpoint_sha256", "representation_model_sha256": model_sha,
                     "representation_encoding_sha256": encoding_sha, "representation_precision": "fp32", "frozen_epoch": 1}
        origin = {"schema": "rz-rovent-collection-native-origin-cpu-lock-v3/1", "input_lock_domain": "rz-rovent-input-lock/3",
                  "input_lock_sha256": lock_sha, "declaration": declaration, "startup_observation": startup,
                  "registered_namespace": namespace, "process_boundary_observation": "unknown", "neural_completion": "unknown",
                  "physical_completion": "unknown"}
        state = {"owner": 1, "revision": 1, "semantic": _hash("state")}
        root = {"schema": "rz-rovent-collection-root-admission-cpu-lock-v3/1", "native_origin": origin,
                "native_origin_sha256": _hash("origin"), "process_epoch": 1, "game_generation": 1, "root_generation": 1,
                "implementation_sha256": _hash("implementation"), "deadline_clock": 1, "deadline_tick": 10,
                "root_budget_sha256": _hash("budget"), "selective_policy_sha256": _hash("policy"), "root_state": state,
                "store_root_state": 1, "root_moves": ["e2e4"], "depth_plies": None, "mate_moves": None,
                "requested_nodes": 10, "root_descriptor": {}, "root_descriptor_sha256": _hash("descriptor"),
                "allocation_clock": {}, "allocation_clock_sha256": _hash("clock")}
        native = row["native"]
        native.update(model_sha256=model_sha, encoding_sha256=encoding_sha)
        native["raw_retirement"]["root_admission"] = root
        native["raw_submission"]["ticket"] = {"implementation_sha256": root["implementation_sha256"],
            "deadline_clock": 1, "deadline_tick": 10, "root_state": state, "representation": {"frozen_epoch": 1}}
        self.assertIsNone(inspect_root_runtime_identity(row, "cpu", lock))
        for mutator in (lambda x: x.update(cpu_session_roles=["public", "shared_pc"]),
                        lambda x: x.update(validator_loaded=False),
                        lambda x: x.update(native_configuration_sha256=_hash("different")),
                        lambda x: x.update(provider="cuda"),
                        lambda x: x.update(system_libraries=[])):
            saved = copy.deepcopy(startup)
            mutator(startup)
            with self.assertRaises(ValueError):
                inspect_root_runtime_identity(row, "cpu", lock)
            startup.clear()
            startup.update(saved)
        # Re-seal each entire synthetic root and record, so rejected topology
        # mismatches cannot be explained by a stale outer digest.
        from rz_pals_model.generation_data import digest
        from rz_pals_model.rovent_lock_v3_rows import inspect_record_seals
        for mode, researcher, expected, forged in (
            ("cpu_pc", None, "roles_only", "own_researcher"),
            ("cpu_pcr", {"provider": "own"}, "own_researcher", "roles_only"),
            ("cpu_pcr", {"provider": "external", "profile": {"purpose": "research_checker"}},
             "external_researcher", "roles_only"),
        ):
            engine["mode"], engine["researcher"] = mode, researcher
            declaration["mode"] = startup["mode"] = mode
            for topology in (expected, forged):
                startup["topology"] = topology
                root["native_origin_sha256"] = digest(origin["schema"], origin)
                retirement = native["raw_retirement"]
                retirement["root_admission_sha256"] = digest(root["schema"], root)
                for event_name in ("raw_submission", "raw_completion", "raw_retirement"):
                    event = native[event_name]
                    native[event_name + "_sha256"] = digest(event["schema"], event)
                projected = dict(row, input={"sealed_input_sha256": row["input_sha256"]}, record_sha256="")
                row["record_sha256"] = digest(row["schema"], projected)
                inspect_record_seals(row, "cpu")
                if topology == expected:
                    self.assertIsNone(inspect_root_runtime_identity(row, "cpu", lock))
                else:
                    with self.assertRaises(ValueError):
                        inspect_root_runtime_identity(row, "cpu", lock)
        for mode, researcher, topology in (
            ("cpu_pc", {"provider": "own"}, "roles_only"),
            ("cpu_pcr", None, "roles_only"),
            ("cpu_pcr", {"provider": "external", "profile": {"purpose": "teacher"}}, "external_researcher"),
        ):
            engine["mode"], engine["researcher"] = mode, researcher
            declaration["mode"] = startup["mode"] = mode
            startup["topology"] = topology
            with self.assertRaises(ValueError):
                inspect_root_runtime_identity(row, "cpu", lock)
        engine["mode"], engine["researcher"] = "cpu_pc", None
        declaration["mode"] = startup["mode"] = "cpu_pc"
        startup["topology"] = "roles_only"
        # Separate CUDA observation: physical library mappings are never inferred
        # from the CPU provider or from the existence of a runtime lock.
        pc["backend"] = "ort_cuda"
        core = {"role": "ort-core", "canonical_path": "/system/ort.so", "sha256": runtime_sha,
                "bytes": 12, "device": 1, "inode": 2, "required_mapping": True}
        provider = {"role": "ort-provider-cuda", "canonical_path": "/system/provider.so", "sha256": _hash("provider"),
                    "bytes": 13, "device": 1, "inode": 3, "required_mapping": True}
        system = {"system_identity_sha256": _hash("system"), "libraries": [core, provider]}
        body["system_runtime_lock"]["runtime"] = {"kind": "system_cuda", "configuration": system}
        del startup["provider"]
        del startup["cpu_session_roles"]
        startup["domain"] = "rz-rovent-native-system-startup/1"
        startup["system_runtime_identity_sha256"] = system["system_identity_sha256"]
        startup["unregistered_native_mappings"] = []
        startup["system_libraries"] = [{key: item[key] for key in ("role", "canonical_path", "sha256", "bytes", "device", "inode")}
                                       | {"mapped": True} for item in (core, provider)]
        self.assertIsNone(inspect_root_runtime_identity(row, "cuda", lock))
        startup["system_libraries"][1]["mapped"] = None
        with self.assertRaises(ValueError):
            inspect_root_runtime_identity(row, "cuda", lock)
        startup["system_libraries"][1]["mapped"] = False
        with self.assertRaises(ValueError):
            inspect_root_runtime_identity(row, "cuda", lock)
        startup["system_libraries"][1]["mapped"] = True
        startup["unregistered_native_mappings"] = ["/unknown.so"]
        with self.assertRaises(ValueError):
            inspect_root_runtime_identity(row, "cuda", lock)


class PinnedStatCompatibilityTests(unittest.TestCase):
    def test_windows_path_and_fd_ctime_can_differ_but_identity_and_birthtime_cannot(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from rz_pals_model import rovent_corpus as corpus
        first = {"st_dev": 1, "st_ino": 2, "st_mode": 0o100600, "st_nlink": 1,
                 "st_size": 5, "st_mtime_ns": 20, "st_ctime_ns": 30, "st_birthtime_ns": 10}
        second = dict(first, st_ctime_ns=10)
        with patch.object(corpus.os, "name", "nt"):
            self.assertEqual(corpus._cross_source_identity(SimpleNamespace(**first)),
                             corpus._cross_source_identity(SimpleNamespace(**second)))
            for field in ("st_dev", "st_ino", "st_mode", "st_nlink", "st_size", "st_mtime_ns", "st_birthtime_ns"):
                changed = dict(second, **{field: second[field] + 1})
                self.assertNotEqual(corpus._cross_source_identity(SimpleNamespace(**first)),
                                    corpus._cross_source_identity(SimpleNamespace(**changed)))
        with patch.object(corpus.os, "name", "posix"):
            self.assertNotEqual(corpus._cross_source_identity(SimpleNamespace(**first)),
                                corpus._cross_source_identity(SimpleNamespace(**second)))

    def test_descriptor_ctime_change_and_inode_substitution_are_rejected(self):
        import os
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace
        from unittest.mock import patch
        from rz_pals_model import rovent_corpus as corpus
        original = os.fstat
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "input.json"
            raw = b'{"data":1}'
            path.write_bytes(raw)
            for field, changed_call, message in (("st_ino", 1, "changed before read"),
                                                ("st_ctime_ns", 2, "raw byte pin/identity mismatch")):
                calls = [0]
                def alter(fd):
                    observed = original(fd)
                    calls[0] += 1
                    if calls[0] != changed_call:
                        return observed
                    keys = ("st_dev", "st_ino", "st_mode", "st_nlink", "st_size", "st_mtime_ns", "st_ctime_ns")
                    values = {key: getattr(observed, key) for key in keys}
                    values["st_birthtime_ns"] = getattr(observed, "st_birthtime_ns", None)
                    values[field] += 1
                    return SimpleNamespace(**values)
                with patch.object(corpus.os, "fstat", alter), self.assertRaisesRegex(ValueError, message):
                    corpus._read_pinned_bytes(path, hashlib.sha256(raw).hexdigest(), len(raw), len(raw))


if __name__ == "__main__":
    unittest.main()
