"""Separate lock-V3 publication envelope checks, not a corpus reader.

This does not validate the files, Rules replay, native lifecycle, source
provenance or training eligibility. It never projects V3 into older families.
Callers must not treat a successful metadata check as permission to train.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from .generation_data import fields, sha, uint
from .rovent_historical_corpus import _relative_pin

CUDA_PUBLICATION = "rz-rovent-corpus-publication-lock-v3/1"
CPU_PUBLICATION = "rz-rovent-corpus-publication-cpu-lock-v3/1"


@dataclass(frozen=True)
class PublicationEnvelopeObservation:
    """Structural observation only: no native authority or corpus admission."""
    family: str
    payload_bytes: int
    record_count: int
    admission_count: int
    training_eligible: bool = False


def validate_publication_envelope(value, *, max_file_bytes, max_total_bytes, max_records):
    """Validate explicit V3 wire family and bounded declared file identities.

    Caller-provided bounds supplement the producer's limits. No file is opened.
    The two runtime identity domains are intentionally mutually exclusive.
    """
    for limit in (max_file_bytes, max_total_bytes, max_records):
        if uint(limit) == 0:
            raise ValueError("positive read limits required")
    fields(value, ("schema", "manifest", "index", "chunks", "canonical_manifest_sha256",
                   "canonical_index_sha256", "source_commit", "source_sha256", "runtime_sha256",
                   "record_count", "total_payload_bytes", "native_admission_sha256", "admissions"),
           "V3 publication")
    schema = value["schema"]
    if schema not in (CPU_PUBLICATION, CUDA_PUBLICATION):
        raise ValueError("explicit CPU or CUDA lock-V3 publication required")
    family = "cpu" if schema == CPU_PUBLICATION else "cuda"
    for name in ("canonical_manifest_sha256", "canonical_index_sha256", "source_sha256", "runtime_sha256"):
        sha(value[name])
    commit = value["source_commit"]
    if type(commit) is not str or len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
        raise ValueError("full lowercase source commit required")
    if uint(value["record_count"], max_records) == 0:
        raise ValueError("nonempty corpus required")
    metadata_cap = min(max_file_bytes, 16 * 1024 * 1024)
    chunk_cap = min(max_file_bytes, 32 * 1024 * 1024)
    def pin(item, cap, description):
        return _relative_pin(item, maximum=cap, description=description)
    manifest = pin(value["manifest"], metadata_cap, "V3 manifest pin")
    index = pin(value["index"], metadata_cap, "V3 index pin")
    if manifest["path"] != "corpus-manifest.json" or index["path"] != "corpus-index.json":
        raise ValueError("canonical corpus metadata paths required")
    def identity(path):
        return path.casefold() if os.name == "nt" else path
    paths = {identity(manifest["path"]), identity(index["path"])}
    total = manifest["bytes"] + index["bytes"]
    chunks = value["chunks"]
    if type(chunks) is not list or not 1 <= len(chunks) <= min(8192, max_records):
        raise ValueError("bounded nonempty chunks required")
    for chunk in chunks:
        pin(chunk, chunk_cap, "V3 chunk pin")
        path = chunk["path"]
        key = identity(path)
        if not path.startswith("chunks/") or not path.endswith(".jsonl") or key in paths:
            raise ValueError("invalid or duplicate chunk path")
        paths.add(key)
        total += chunk["bytes"]
    seals = value["native_admission_sha256"]
    if (type(seals) is not list or not 1 <= len(seals) <= max_records
            or seals != sorted(set(sha(item) for item in seals))):
        raise ValueError("sorted unique native admission seals required")
    entries = value["admissions"]
    if type(entries) is not list or len(entries) != len(seals):
        raise ValueError("exact admission coverage required")
    seen = []
    locks = {}
    raw_canonicals = {}
    for entry in entries:
        runtime_field = "cpu_runtime_library_sha256" if family == "cpu" else "system_runtime_identity_sha256"
        fields(entry, ("native_admission_sha256", "input_lock", "input_lock_bytes_origin",
                       "canonical_input_lock_sha256", runtime_field), "V3 admission")
        seen.append(sha(entry["native_admission_sha256"]))
        canonical = sha(entry["canonical_input_lock_sha256"])
        observed = sha(entry[runtime_field])
        if family == "cpu" and observed != value["runtime_sha256"]:
            raise ValueError("CPU runtime library SHA must match publication")
        if entry["input_lock_bytes_origin"] not in ("original_file", "validated_semantic_snapshot"):
            raise ValueError("invalid input lock bytes origin")
        source = pin(entry["input_lock"], metadata_cap, "V3 input lock pin")
        path = source["path"]
        key = identity(path)
        if not path.startswith("input-locks/") or not path.endswith(".json") or key in paths:
            raise ValueError("invalid or conflicting input lock path")
        raw_key = (source["sha256"], source["bytes"])
        if raw_key in raw_canonicals and raw_canonicals[raw_key] != canonical:
            raise ValueError("identical lock bytes have conflicting canonical identities")
        raw_canonicals[raw_key] = canonical
        if key in locks:
            if locks[key] != (source, canonical):
                raise ValueError("conflicting input lock pin")
        else:
            locks[key] = (source, canonical)
            total += source["bytes"]
    if seen != seals:
        raise ValueError("admission entries must match the sorted seal set")
    if uint(value["total_payload_bytes"], max_total_bytes) != total:
        raise ValueError("unique pinned payload byte count mismatch")
    return PublicationEnvelopeObservation(family, total, value["record_count"], len(seen))

@dataclass(frozen=True)
class PinnedMetadataObservation:
    """Pinned metadata joined only; chunk and native row verification remain open."""
    family: str
    locks_checked: int
    chunks_checked: int = 0
    training_eligible: bool = False
    lifecycle_checked: int = 0


def _inspect_pinned_metadata(root, publication, *, read_budget, limits):
    """Read explicitly named metadata and perform a limited V3 identity join.

    This validates raw metadata pins and some canonical identities, not the
    full typed lock, native owner/root, rows, source proof or Rules replay.
    In particular this function never opens chunk bodies. It returns no data
    usable by the historical training corpus API.
    """
    from . import rovent_corpus as corpus
    from .generation_data import digest
    from .rovent_corpus_codec import SUBMITTED_CODEC
    from .rovent_historical_corpus import _absolute, _read_file, _source_aggregate

    envelope = validate_publication_envelope(
        publication, max_file_bytes=limits.max_file_bytes,
        max_total_bytes=limits.max_total_bytes, max_records=limits.max_records)
    root = _absolute(str(root), "V3 corpus root")
    manifest = _read_file(root, publication["manifest"], read_budget, json_metadata=True)
    index = _read_file(root, publication["index"], read_budget, json_metadata=True)
    manifest_domain = ("rz-rovent-corpus-manifest-cpu-lock-v3/1" if envelope.family == "cpu"
                       else "rz-rovent-corpus-manifest-lock-v3/1")
    index_domain = ("rz-rovent-corpus-index-cpu-lock-v3/1" if envelope.family == "cpu"
                    else "rz-rovent-corpus-index-lock-v3/1")
    fields(manifest, ("schema", "registration", "startup_lock_sha256"), "V3 corpus manifest")
    fields(index, ("schema", "manifest_sha256", "record_count", "chunks", "origins"), "V3 corpus index")
    registration = corpus._manifest(manifest["registration"], limits, codec=SUBMITTED_CODEC)
    if (manifest["schema"] != manifest_domain or index["schema"] != index_domain
            or digest(manifest_domain, manifest) != publication["canonical_manifest_sha256"]
            or digest(index_domain, index) != publication["canonical_index_sha256"]
            or index["manifest_sha256"] != publication["canonical_manifest_sha256"]):
        raise ValueError("V3 manifest/index family or canonical digest mismatch")
    for key in ("source_commit", "source_sha256", "runtime_sha256"):
        if registration[key] != publication[key]:
            raise ValueError("V3 corpus registration source/runtime mismatch")
    if (uint(index["record_count"], limits.max_records) != publication["record_count"]
            or type(index["chunks"]) is not list
            or len(index["chunks"]) != len(publication["chunks"])
            or type(index["origins"]) is not list
            or len(index["origins"]) != publication["record_count"]):
        raise ValueError("V3 index count mismatch")
    record_count = 0
    for item, pinned in zip(index["chunks"], publication["chunks"]):
        fields(item, ("path", "sha256", "bytes", "records"), "V3 index chunk")
        if any(item[name] != pinned[name] for name in ("path", "sha256", "bytes")):
            raise ValueError("V3 index and publication chunk pins differ")
        record_count += uint(item["records"], limits.max_records)
    if record_count != publication["record_count"]:
        raise ValueError("V3 indexed chunk record total mismatch")

    lock_cache = {}
    for entry in publication["admissions"]:
        pin = entry["input_lock"]
        key = pin["path"]
        if key not in lock_cache:
            lock_cache[key] = _read_file(root, pin, read_budget, json_metadata=True)
        lock = fields(lock_cache[key], ("domain", "manifest", "canonical_sha256"), "V3 input lock")
        body = fields(lock["manifest"], ("domain", "contract_revision", "run_id", "purpose", "engine",
            "resources", "coordinator", "wall_time_max_ms", "cleanup_max_ms", "output_bytes_max",
            "agent_renewable", "provenance", "startup_document", "startup_configuration_sha256",
            "startup_search_configuration_sha256", "own_r_root_tie", "system_runtime_lock"), "V3 execution manifest")
        if (lock["domain"] != "rz-rovent-input-lock/3" or body["domain"] != "rz-rovent-execution/3"
                or body["purpose"] != "collection"
                or digest("rz-rovent-execution-canonical/3", body) != sha(lock["canonical_sha256"])
                or lock["canonical_sha256"] != entry["canonical_input_lock_sha256"]
                or lock["canonical_sha256"] != manifest["startup_lock_sha256"]):
            raise ValueError("V3 original input lock identity mismatch")
        engine = body["engine"]
        if type(engine) is not dict or engine.get("source_commit") != publication["source_commit"]:
            raise ValueError("V3 input lock source commit mismatch")
        if _source_aggregate(engine["source_commit"], engine.get("source_components")) != publication["source_sha256"]:
            raise ValueError("V3 input lock source component identity mismatch")
        runtime_lock = fields(body["system_runtime_lock"], ("domain", "runtime", "canonical_sha256"),
                              "V3 runtime lock")
        runtime = runtime_lock["runtime"]
        if (runtime_lock["domain"] != "rz-system-runtime-lock-v6/1" or type(runtime) is not dict
                or digest(runtime_lock["domain"], runtime) != sha(runtime_lock["canonical_sha256"])):
            raise ValueError("V3 runtime lock canonical identity mismatch")
        pc = engine.get("pc")
        if type(pc) is not dict:
            raise ValueError("V3 PC runtime declaration required")
        if envelope.family == "cpu":
            fields(runtime, ("kind",), "V3 CPU runtime source")
            artifacts = pc.get("runtime_artifacts")
            if (runtime["kind"] != "cpu" or pc.get("backend") != "ort_cpu"
                    or pc.get("device") != {"device": "cpu"}
                    or type(artifacts) is not list or len(artifacts) != 1
                    or type(artifacts[0]) is not dict or artifacts[0].get("sha256") != publication["runtime_sha256"]):
                raise ValueError("V3 CPU runtime artifact mismatch")
        else:
            fields(runtime, ("kind", "configuration"), "V3 CUDA runtime source")
            config = runtime["configuration"]
            libraries = config.get("libraries") if type(config) is dict else None
            core = [lib for lib in libraries if type(lib) is dict and lib.get("role") == "ort-core"] if type(libraries) is list else []
            if (runtime["kind"] != "system_cuda" or pc.get("backend") != "ort_cuda"
                    or type(config) is not dict or config.get("system_identity_sha256") != entry["system_runtime_identity_sha256"]
                    or len(core) != 1 or core[0].get("sha256") != publication["runtime_sha256"]):
                raise ValueError("V3 CUDA runtime identity mismatch")
    return PinnedMetadataObservation(envelope.family, len(lock_cache)), manifest, index, lock_cache




def inspect_pinned_metadata(root, publication, *, read_budget, limits):
    """Inspect metadata pins only; no chunk, Rules, or training acceptance."""
    observation, _, _, _ = _inspect_pinned_metadata(root, publication, read_budget=read_budget, limits=limits)
    return observation


def inspect_pinned_row_seals(root, publication, *, read_budget, limits, inspect_lifecycle=False):
    """Read bounded chunk bytes once and inspect row and owner-event seals.

    A successful result is not corpus admission: root descriptors, actual Rules
    replay, full target semantics, and provenance remain independently required.
    """
    from . import rovent_corpus as corpus
    from .rovent_historical_corpus import _absolute
    from .rovent_lock_v3_rows import inspect_record_seals, inspect_native_lifecycle, inspect_root_runtime_identity
    if type(inspect_lifecycle) is not bool:
        raise ValueError("explicit boolean lifecycle selection required")

    observation, manifest, index, lock_cache = _inspect_pinned_metadata(root, publication, read_budget=read_budget, limits=limits)
    root = _absolute(str(root), "V3 corpus root")
    origin_map = {}
    for origin in index["origins"]:
        if type(origin) is not dict:
            raise ValueError("typed V3 origin index required")
        identity = sha(origin.get("record_sha256"))
        if identity in origin_map:
            raise ValueError("duplicate V3 indexed record identity")
        origin_map[identity] = origin
    admissions = {item["native_admission_sha256"]: item for item in publication["admissions"]}
    seen, seen_origins, seen_tickets = set(), set(), set()
    chunks_checked = 0
    for chunk in index["chunks"]:
        path = corpus._relative_chunk(root, chunk["path"])
        count = uint(chunk["records"], limits.max_records)
        if count == 0:
            raise ValueError("nonempty V3 chunk required")
        records = corpus._read_pinned_records(path, chunk["sha256"], chunk["bytes"],
                    min(limits.max_file_bytes, 32 * 1024 * 1024), count, read_accounting=read_budget)
        for record in records:
            if inspect_lifecycle:
                inspect_native_lifecycle(manifest["registration"], record, observation.family)
            else:
                inspect_record_seals(record, observation.family)
            root_sha = record["native"]["raw_retirement"]["root_admission_sha256"]
            admission = admissions.get(root_sha)
            if admission is None:
                raise ValueError("V3 row root is absent from publication admissions")
            row_root = record["native"]["raw_retirement"]["root_admission"]
            lock = lock_cache[admission["input_lock"]["path"]]
            if row_root["native_origin"].get("input_lock_sha256") != lock["canonical_sha256"]:
                raise ValueError("V3 row root differs from its original pinned lock")
            if inspect_lifecycle:
                inspect_root_runtime_identity(record, observation.family, lock)
            identity = record["record_sha256"]
            if identity not in origin_map or identity in seen:
                raise ValueError("V3 chunk has duplicate or unindexed record")
            seen.add(identity)
            origin = origin_map[identity]
            snapshot, lineage = record["input"], record["lineage"]
            if type(snapshot) is not dict or type(lineage) is not dict:
                raise ValueError("V3 indexed input and lineage objects required")
            expected = {key: record[key] for key in ("record_sha256", "input_sha256", "origin_sha256", "split")}
            for key in ("game_id", "opening_id", "line_genealogy_id", "transposition_sha256"):
                if key not in snapshot:
                    raise ValueError("V3 indexed input is missing an origin field")
                expected[key] = snapshot[key]
            fields(lineage, ("actor_id", "teacher_actor_id", "ancestor_origin_sha256"), "V3 row lineage")
            expected.update(lineage)
            if origin != expected:
                raise ValueError("V3 indexed origin does not match original record")
            if record["origin_sha256"] in seen_origins:
                raise ValueError("duplicate V3 record origin identity")
            seen_origins.add(record["origin_sha256"])
            native = record["native"]
            try:
                ticket = tuple(uint(native[key]) for key in
                    ("process_epoch", "game_generation", "root_generation", "request_sequence"))
            except KeyError as error:
                raise ValueError("V3 native ticket identity missing") from error
            if 0 in ticket or ticket in seen_tickets:
                raise ValueError("duplicate or empty V3 native ticket identity")
            seen_tickets.add(ticket)
        chunks_checked += 1
    if len(seen) != index["record_count"]:
        raise ValueError("V3 chunk and origin coverage differ")
    return PinnedMetadataObservation(observation.family, observation.locks_checked, chunks_checked,
                                     lifecycle_checked=len(seen) if inspect_lifecycle else 0)
