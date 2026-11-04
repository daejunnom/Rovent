"""Isolated Rust-serde observation wire parity, without a historical source owner.

The snapshot adapter below exists only in this test entry. It neither inherits
HistoricalRoventRCorpus nor enters a full validator/trainer. Production pure
shape/seal/join validators consume original DTO operands; no source file, model,
Rules playback, native custody or physical completion is verified here.
"""
from __future__ import annotations

import copy
import importlib.abc
import json
from pathlib import Path
import sys


MAX_INPUT_BYTES = 16 << 20
FIXTURE_DOMAIN = "rz-rovent-r-observation-wire-fixture/1"
RESULT_DOMAIN = "rz-rovent-r-observation-wire-result/1"
CLASSIFICATION = "synthetic_wire_only_no_capture_authority"
NEURAL_ROOTS = frozenset(("torch", "numpy", "onnx", "onnxruntime", "tensorflow", "jax"))


class _NoNeuralImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] in NEURAL_ROOTS:
            raise ImportError("neural imports are forbidden in observation wire parity")
        return None


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate observation wire object field")
        result[key] = value
    return result


def _nonfinite(_value):
    raise ValueError("nonfinite observation wire JSON number")


def _base_fixture(base, pc, corpus, wire, fields, sha, input_lock, config_type):
    fields(base, ("schema", "classification", "manifest", "records", "origins", "input_locks",
        "nnuec_config", "edge_payload", "edge_sha256"), "original typed Rust R base fixture")
    if base["schema"] != "rz-rovent-r-wire-fixture/1" or base["classification"] != CLASSIFICATION:
        raise ValueError("explicit synthetic R base fixture required")
    manifest = fields(base["manifest"], ("schema", "collection", "pc_ancestor_index_sha256", "r_ancestor_index_sha256"),
        "original typed Rust R manifest")
    if manifest["schema"] != wire.MANIFEST_DOMAIN:
        raise ValueError("explicit R manifest domain required")
    pc._manifest(manifest["collection"], pc.CorpusLimits())
    for key in ("pc_ancestor_index_sha256", "r_ancestor_index_sha256"):
        pc._sha_list(manifest[key], 256, key)
    config = config_type.from_mapping(base["nnuec_config"])
    records, origins, declarations = base["records"], base["origins"], base["input_locks"]
    if (type(records) is not list or len(records) != 2 or type(origins) is not list or len(origins) != 2
            or type(declarations) is not dict or not 1 <= len(declarations) <= 2):
        raise ValueError("bounded original two-record R base fixture required")
    locks = {sha(key): input_lock(value, key) for key, value in declarations.items()}
    if type(records[0]) is not dict or type(records[0].get("bound")) is not dict:
        raise ValueError("original R Bound operand required")
    parameter = sha(records[0]["bound"].get("parameter_sha256"))
    samples = [corpus._validate_record(record, manifest, origin, locks, config, parameter)
        for record, origin in zip(records, origins, strict=True)]
    if ([record["question"]["readout_sequence"] for record in records] != [0, 2]
            or records[0]["retirement_sha256"] != records[1]["retirement_sha256"]
            or records[0]["retirement"]["retained_readouts"] != len(records)
            or samples[0]["input_sha256"] != samples[1]["input_sha256"]
            or samples[0]["record_sha256"] == samples[1]["record_sha256"]
            or any(record["retirement"]["ending"] != {"kind": "unknown"} for record in records)
            or any(record["wdl"] is not None or record["semantic_consumption"] is not None
                or record["whole_action_cost"] is not None for record in records)
            or wire.r_readout_f32_digest(wire.FLOAT_BITS_DOMAIN, base["edge_payload"]) != sha(base["edge_sha256"])):
        raise ValueError("original R base occurrence/unknown-scope/framing parity differs")
    return manifest, records, locks


class _SyntheticWireSnapshots:
    """Test-only detached DTO lookup; deliberately not a historical owner type."""
    def __init__(self, *, publication, publication_sha, root, admission_sha, retirements, records):
        self._publication = publication
        self._publication_sha = publication_sha
        self._root, self._admission_sha = root, admission_sha
        self._retirements = {item["retirement_sha256"]: item for item in retirements}
        self._records = {item["record_sha256"]: item for item in records}
        if len(self._retirements) != len(retirements) or len(self._records) != len(records):
            raise ValueError("synthetic observation lookup has duplicate original identity")
        self.corpus = self

    def original_retirement_evidence(self, identity):
        part = self._retirements[identity]
        return copy.deepcopy({"publication_file_sha256": self._publication_sha,
            "observation": {"native_admission": self._root, "native_admission_sha256": self._admission_sha,
                "retirement": part["retirement"], "retirement_sha256": identity,
                "readout_record_sha256": part["readout_record_sha256"]},
            "execution_authorized": False})

    def original_semantic_evidence(self, identity):
        row = self._records[identity]
        return copy.deepcopy({name: row[name] for name in ("native_admission_sha256", "retirement_sha256",
            "retirement", "bound", "semantic_consumption", "whole_action_cost")})


def validate_fixture(fixture):
    """Return wire parity only; never invoke a full typed-source validator."""
    from rz_pals_model.generation_data import digest, fields, sha, uint
    from rz_pals_model import rovent_corpus as pc
    from rz_pals_model import rovent_r_corpus as corpus
    from rz_pals_model import rovent_r_readout as wire
    from rz_pals_model import rovent_r_semantic as semantic
    from rz_pals_model import rovent_r_unsubmitted as unsubmitted
    from rz_pals_model import rovent_historical_r_corpus as historical
    from rz_pals_model.nnuec_reference import NnuecConfig
    from rz_pals_model.rovent_historical_corpus import _input_lock

    fields(fixture, ("schema", "classification", "base", "publication", "publication_file_sha256", "semantic", "unsubmitted"),
        "typed Rust R observation fixture")
    if fixture["schema"] != FIXTURE_DOMAIN or fixture["classification"] != CLASSIFICATION:
        raise ValueError("explicit observation wire-only classification required")
    manifest, records, locks = _base_fixture(fixture["base"], pc, corpus, wire, fields, sha, _input_lock, NnuecConfig)
    publication = historical._publication(fixture["publication"], pc.CorpusLimits())
    if (publication["canonical_manifest_sha256"] != digest(wire.MANIFEST_DOMAIN, manifest)
            or publication["record_count"] != len(records)
            or any(publication[key] != manifest["collection"][key] for key in ("source_commit", "source_sha256", "runtime_sha256"))):
        raise ValueError("observation publication/base metadata join differs")
    publication_sha = sha(fixture["publication_file_sha256"])
    sidecar = fields(fixture["unsubmitted"], unsubmitted._TOP, "typed Rust unsubmitted fixture")
    semantic_value = fields(fixture["semantic"], semantic._TOP, "typed Rust semantic fixture")
    semantic._encoded(semantic_value, semantic.MAX_BYTES, "observation semantic fixture")
    unsubmitted._encoded(sidecar, unsubmitted.MAX_BYTES, "observation unsubmitted fixture")
    if (semantic_value["schema"] != semantic.DOMAIN or sidecar["schema"] != unsubmitted.DOMAIN
            or type(semantic_value["per_readout_consumption"]) is not bool or semantic_value["per_readout_consumption"]
            or type(sidecar["per_readout_consumption"]) is not bool or sidecar["per_readout_consumption"]
            or any(semantic_value[key] is not None for key in ("whole_action_cost", "wdl_target", "utility_target"))
            or any(sidecar[key] is not None for key in unsubmitted._NO_TARGET)
            or semantic_value["r_publication_file_sha256"] != publication_sha
            or sidecar["r_publication_file_sha256"] != publication_sha
            or semantic_value["native_admission_sha256"] != sidecar["native_admission_sha256"]
            or semantic_value["authority"] != sidecar["authority"]):
        raise ValueError("observation wire domain/unknown target/original source scope differs")
    semantic._authority(semantic_value["authority"])
    unsubmitted._authority(sidecar["authority"])
    lock = locks.get(sha(sidecar["input_lock_sha256"]))
    if lock is None:
        raise ValueError("observation fixture lacks original InputLock operand")
    nnuec = unsubmitted._admission(sidecar["native_admission"], lock, sha(sidecar["native_admission_sha256"]),
        sidecar["authority"], sha(sidecar["nnuec_weights_raw_sha256"]))
    entry = next((part for part in publication["admissions"] if part["native_admission_sha256"] == sidecar["native_admission_sha256"]), None)
    if (entry is None or entry["canonical_input_lock_sha256"] != lock["canonical_sha256"]
            or entry["nnuec_parameter_sha256"] != sha(sidecar["nnuec_parameter_sha256"])
            or any(entry[name][key] != nnuec[declaration][key]
                for name, declaration in (("nnuec_weights", "weights"), ("nnuec_provenance", "provenance")) for key in ("sha256", "bytes"))):
        raise ValueError("observation wire publication/NNUEC declarations differ")
    # These are original serde snapshots only, not a constructed historical owner.
    adapter = _SyntheticWireSnapshots(publication=publication, publication_sha=publication_sha,
        root=sidecar["native_admission"], admission_sha=sidecar["native_admission_sha256"],
        retirements=semantic_value["retirements"], records=records)
    terminals, readouts, _empty = semantic._retirements(semantic_value["retirements"], adapter, publication,
        sidecar["native_admission_sha256"], sidecar["authority"])
    executions, stages = semantic._observations(semantic_value["observations"], terminals, readouts,
        sidecar["authority"], entry["nnuec_registration"]["model_identity"])
    uint(semantic_value["unrecorded_witness_attempts"])
    for failure in semantic._list(semantic_value["manager_failures"], 64, "wire semantic manager failure"):
        fields(failure, ("stage", "code", "detail"), "wire semantic manager failure")
        if failure["stage"] not in ("Dispatch", "NativeEvent", "Accepted", "Abandoned", "Settled"):
            raise ValueError("wire semantic manager failure stage differs")
        for name, maximum, empty in (("code", 128, False), ("detail", 2048, True)):
            text = failure[name]
            if type(text) is not str or not empty and not text or len(text.encode("utf-8")) > maximum:
                raise ValueError("wire semantic bounded failure text required")
    counts = unsubmitted._items(sidecar["observations"], sidecar, adapter, publication)
    if (digest(semantic.DOMAIN, {**semantic_value, "semantic_sha256": ""}) != sha(semantic_value["semantic_sha256"])
            or digest(unsubmitted.DOMAIN, {**sidecar, "observations_sha256": ""}) != sha(sidecar["observations_sha256"])):
        raise ValueError("original observation wire outer digest differs")
    if (len(terminals) != 2 or executions != 1
            or stages != {"raw_published": 1, "search_feedback_consumed": 1, "selected_pc_view_consumed": 1}
            or counts != {"unsubmitted": 1, "not_started": 1}):
        raise ValueError("explicit synthetic observation fixture occurrences differ")
    return {"schema": RESULT_DOMAIN, "status": "accepted_wire_only", "records": len(records), "retirements": len(terminals),
        "semantic_execution_count": executions, "semantic_stages": stages, "unsubmitted_count": counts["unsubmitted"],
        "not_started_count": counts["not_started"], "semantic_sha256": semantic_value["semantic_sha256"],
        "observations_sha256": sidecar["observations_sha256"], "full_typed_source_validation": False,
        "source_files_verified": False, "rules_replayed": False, "original_native_authority": False,
        "physical_completion_verified": False, "training_authorized": False, "execution_authorized": False, "neural_imports": False}


def main():
    if not (sys.flags.isolated and sys.flags.no_site and sys.flags.dont_write_bytecode):
        raise RuntimeError("observation wire decoder requires -I -S -B isolation")
    if any(name.split(".", 1)[0] in NEURAL_ROOTS for name in sys.modules):
        raise RuntimeError("observation decoder began with an imported neural runtime")
    sys.meta_path.insert(0, _NoNeuralImports())
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    raw = sys.stdin.buffer.read(MAX_INPUT_BYTES + 1)
    if not raw or len(raw) > MAX_INPUT_BYTES:
        raise ValueError("finite nonempty observation wire fixture input required")
    fixture = json.loads(raw.decode("utf-8", "strict"), object_pairs_hook=_unique_object, parse_constant=_nonfinite)
    result = validate_fixture(fixture)
    if any(name.split(".", 1)[0] in NEURAL_ROOTS for name in sys.modules):
        raise RuntimeError("observation decoder imported a neural runtime")
    sys.stdout.write(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        sys.stderr.write((type(error).__name__ + ": " + str(error))[:4096] + "\n")
        raise SystemExit(1)
