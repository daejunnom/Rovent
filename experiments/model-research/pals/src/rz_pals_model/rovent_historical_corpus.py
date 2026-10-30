"""Pinned historical corpus files, distinct from current native authority.

This owner verifies durable file bytes and joins collection identities. It
neither executes Rules nor reconstructs a producer, native ticket, physical
completion, original trainer lease, or permission to execute a new window.
Rust's parent independently validates original Rules replay and loaded locks.
Only explicitly registered publication files are opened; there is no discovery
or legacy format conversion and no neural framework import.
"""
from __future__ import annotations

import copy
import hashlib
import os
from pathlib import Path

from .generation_data import digest, fields, identifier, sha, uint
from . import rovent_corpus as original
from .rovent_corpus_codec import (LEGACY_CODEC, SUBMITTED_CODEC, codec_from_publication,
    validate_container_pair, validate_record_codec)
from .rovent_training_bootstrap import MAX_READ_BYTES as _MAX_SHARED_READ_BYTES

PUBLICATION_DOMAIN = "rz-rovent-corpus-publication/1"
SUBMITTED_PUBLICATION_DOMAIN = "rz-rovent-corpus-publication/2"
OBSERVATION_DOMAIN = "rz-rovent-historical-corpus-observation/1"
NATIVE_DOMAIN = "rz-rovent-native-owner-event/2"
ROOT_DOMAIN = "rz-rovent-collection-root-admission/1"
ORIGIN_DOMAIN = "rz-rovent-collection-native-origin/1"
_ROLES = ("proposer", "critic", "validator")
_MAX_METADATA_BYTES = 16 * 1024 * 1024
_MAX_CHUNK_BYTES = 32 * 1024 * 1024


class HistoricalReadBudget:
    """Local shared read accounting using the child PinnedReadBudget protocol.

    This stdlib implementation also supports standalone historical validation.
    Each lease reserves the expected body and one growth probe; actual returned
    blocks move reservation to charged bytes. Failure retains unused credit and
    sets unknown. Metadata validation does not grant execution authority.
    """
    def __init__(self, maximum_bytes):
        if uint(maximum_bytes, _MAX_SHARED_READ_BYTES) == 0:
            raise ValueError("positive bounded historical read budget required")
        self._maximum, self._reserved, self._charged, self._unknown = maximum_bytes, 0, 0, False
        self._leases = {}

    def reserve_external(self, upper_bytes):
        upper_bytes = uint(upper_bytes, _MAX_SHARED_READ_BYTES)
        if (self._unknown or upper_bytes == 0
                or self._charged + self._reserved + upper_bytes > self._maximum):
            raise ValueError("historical shared read reservation exhausted")
        self._reserved += upper_bytes
        lease = object()
        self._leases[lease] = {"charged": 0, "remaining": upper_bytes, "failed": False}
        return lease

    def _lease(self, lease):
        if lease not in self._leases or self._leases[lease]["failed"]:
            raise ValueError("original active historical read lease required")
        return self._leases[lease]

    def charge_external(self, lease, count):
        state = self._lease(lease)
        count = uint(count, _MAX_SHARED_READ_BYTES)
        if count > state["remaining"]:
            raise ValueError("historical actual read exceeds reserved upper bound")
        state["charged"] += count
        state["remaining"] -= count
        self._charged += count
        self._reserved -= count

    def finish_external(self, lease, *, expected_bytes):
        state = self._lease(lease)
        if state["charged"] != uint(expected_bytes) or state["remaining"] != 1:
            raise ValueError("historical read lease did not close exact bytes plus EOF probe")
        self._reserved -= state["remaining"]
        del self._leases[lease]

    def fail_external(self, lease):
        state = self._lease(lease)
        state["failed"], self._unknown = True, True

    def observation(self):
        return {"maximum_bytes": self._maximum, "charged_bytes": self._charged,
                "unknown": self._unknown or self._reserved != 0}


def _absolute(value, description):
    if type(value) is not str or not value or len(value.encode("utf-8")) > 4096:
        raise ValueError(f"absolute bounded historical {description} required")
    path = Path(value)
    if (not path.is_absolute() or any(part == ".." for part in path.parts)
            or any(ord(char) < 32 or 127 <= ord(char) <= 159 for char in value)):
        raise ValueError(f"absolute bounded historical {description} required")
    return path


def _registration(value):
    fields(value, ("root", "publication", "collection_source_commit", "collection_source_sha256",
                   "collection_runtime_library_sha256", "model_semantics_sha256", "model_epoch",
                   "context_epoch", "recipe_sha256", "stage_id", "role_bindings"),
           "historical corpus registration")
    root = _absolute(value["root"], "publication root")
    pin = fields(value["publication"], ("path", "sha256", "bytes"), "historical publication file pin")
    if _absolute(pin["path"], "publication file") != root / "publication.json":
        raise ValueError("historical publication must be the registered root publication.json")
    sha(pin["sha256"])
    if uint(pin["bytes"], _MAX_METADATA_BYTES) == 0:
        raise ValueError("nonempty bounded historical publication required")
    revision = value["collection_source_commit"]
    if (type(revision) is not str or len(revision) != 40
            or any(char not in "0123456789abcdef" for char in revision)):
        raise ValueError("historical collection full source commit required")
    for name in ("collection_source_sha256", "collection_runtime_library_sha256",
                 "model_semantics_sha256", "recipe_sha256"):
        sha(value[name])
    for name in ("model_epoch", "context_epoch"):
        if uint(value[name]) == 0:
            raise ValueError("positive historical model/context epoch required")
    identifier(value["stage_id"])
    bindings = value["role_bindings"]
    if type(bindings) is not dict or not 1 <= len(bindings) <= len(_ROLES):
        raise ValueError("bounded registered historical role bindings required")
    for role, binding in bindings.items():
        if role not in _ROLES:
            raise ValueError("unknown historical role binding")
        fields(binding, ("model_semantics_sha256", "model_identity_sha256", "weights_sha256", "encoding_sha256"),
               "historical role model binding")
        for value_sha in binding.values():
            sha(value_sha)
    return root


def _relative_pin(value, *, maximum, description):
    fields(value, ("path", "sha256", "bytes"), description)
    path = value["path"]
    if (type(path) is not str or not path or len(path.encode("utf-8")) > 4096
            or "\\" in path or ":" in path or path.startswith("/")
            or any(part in ("", ".", "..") for part in path.split("/"))
            or any(ord(char) < 32 or 127 <= ord(char) <= 159 for char in path)):
        raise ValueError("normalized relative historical publication file required")
    sha(value["sha256"])
    if uint(value["bytes"], maximum) == 0:
        raise ValueError("nonempty bounded historical payload file required")
    return value


def _publication(value, limits):
    """The original /1 parser does not admit or convert submitted /2 bytes."""
    return _publication_for_codec(value, limits, LEGACY_CODEC)


def _submitted_publication(value, limits):
    """Select the explicit submitted family without changing its body."""
    return _publication_for_codec(value, limits, SUBMITTED_CODEC)


def _publication_for_codec(value, limits, codec):
    fields(value, ("schema", "manifest", "index", "chunks", "canonical_manifest_sha256",
                   "canonical_index_sha256", "source_commit", "source_sha256", "runtime_sha256",
                   "record_count", "total_payload_bytes", "native_admission_sha256", "admissions"),
           "historical corpus publication")
    if codec not in (LEGACY_CODEC, SUBMITTED_CODEC) or value["schema"] != codec.publication_domain:
        raise ValueError("explicit historical Rovent publication domain required")
    for name in ("manifest", "index"):
        _relative_pin(value[name], maximum=min(_MAX_METADATA_BYTES, limits.max_file_bytes),
                      description="historical publication metadata pin")
    if (value["manifest"]["path"] != "corpus-manifest.json"
            or value["index"]["path"] != "corpus-index.json"):
        raise ValueError("historical publication canonical metadata names required")
    if type(value["chunks"]) is not list or not 1 <= len(value["chunks"]) <= min(8192, limits.max_records):
        raise ValueError("bounded nonempty historical publication chunks required")
    paths = {value["manifest"]["path"], value["index"]["path"]}
    total = value["manifest"]["bytes"] + value["index"]["bytes"]
    for chunk in value["chunks"]:
        _relative_pin(chunk, maximum=min(_MAX_CHUNK_BYTES, limits.max_file_bytes),
                      description="historical publication chunk pin")
        path = chunk["path"]
        identity = path.casefold() if os.name == "nt" else path
        if not path.startswith("chunks/") or not path.endswith(".jsonl") or identity in paths:
            raise ValueError("historical publication duplicate/overlapping chunk path")
        paths.add(identity)
        total += chunk["bytes"]
    if uint(value["record_count"], limits.max_records) == 0:
        raise ValueError("nonempty historical record count required")
    for name in ("canonical_manifest_sha256", "canonical_index_sha256", "source_sha256", "runtime_sha256"):
        sha(value[name])
    revision = value["source_commit"]
    if (type(revision) is not str or len(revision) != 40
            or any(char not in "0123456789abcdef" for char in revision)):
        raise ValueError("historical publication full source commit required")
    admissions = value["native_admission_sha256"]
    if (type(admissions) is not list or not 1 <= len(admissions) <= limits.max_records
            or admissions != sorted(set(sha(item) for item in admissions))):
        raise ValueError("sorted unique bounded historical admission seals required")
    entries = value["admissions"]
    if type(entries) is not list or len(entries) != len(admissions):
        raise ValueError("historical publication requires every original input lock pin")
    locks, bundles, canonical_paths, path_canonicals, seen_admissions = {}, {}, {}, {}, []
    for entry in entries:
        fields(entry, ("native_admission_sha256", "input_lock", "canonical_input_lock_sha256", "runtime_bundle_manifest"),
               "historical publication admission lock pin")
        seen_admissions.append(sha(entry["native_admission_sha256"]))
        canonical = sha(entry["canonical_input_lock_sha256"])
        pin = _relative_pin(entry["input_lock"], maximum=min(_MAX_METADATA_BYTES, limits.max_file_bytes),
                            description="historical original input lock pin")
        path = pin["path"]
        identity = path.casefold() if os.name == "nt" else path
        if not path.startswith("input-locks/") or not path.endswith(".json") or identity in paths:
            raise ValueError("historical lock artifact overlaps corpus payload")
        if identity in locks and locks[identity] != pin:
            raise ValueError("historical shared lock path changed raw file pin")
        if identity in path_canonicals and path_canonicals[identity] != canonical:
            raise ValueError("historical one lock artifact cannot name multiple canonical locks")
        if canonical in canonical_paths and canonical_paths[canonical] != identity:
            raise ValueError("historical one canonical input lock must use one artifact path")
        if identity not in locks:
            locks[identity] = pin
            total += pin["bytes"]
        canonical_paths[canonical] = identity
        path_canonicals[identity] = canonical
        bundle = entry["runtime_bundle_manifest"]
        if bundle is not None:
            _relative_pin(bundle, maximum=min(64 * 1024, limits.max_file_bytes),
                description="historical original CUDA bundle metadata pin")
            bundle_path = bundle["path"].casefold() if os.name == "nt" else bundle["path"]
            if (not bundle["path"].startswith("runtime-bundles/") or not bundle["path"].endswith(".json")
                    or bundle_path in paths or bundle_path in locks):
                raise ValueError("historical runtime bundle overlaps corpus/lock payload")
            identity = bundle["sha256"]
            if identity in bundles:
                if bundles[identity] != bundle:
                    raise ValueError("historical runtime bundle raw SHA names conflicting file pins")
            else:
                if bundle_path in {item["path"].casefold() if os.name == "nt" else item["path"] for item in bundles.values()}:
                    raise ValueError("historical runtime bundle path has conflicting raw SHA")
                bundles[identity] = bundle
                total += bundle["bytes"]
    if seen_admissions != admissions:
        raise ValueError("historical publication admission entries differ from exact sorted set")
    if uint(value["total_payload_bytes"], limits.max_total_bytes) != total:
        raise ValueError("historical publication exact unique payload byte mismatch")
    return value


def _read_file(root, pin, budget, *, relative=True, json_metadata=False):
    path = root.joinpath(*pin["path"].split("/")) if relative else Path(pin["path"])
    raw = original._read_pinned_bytes(path, pin["sha256"], pin["bytes"], pin["bytes"],
        suffixes=(".json",) if json_metadata else (".jsonl",), read_accounting=budget)
    return original._decode_json(raw) if json_metadata else None


def _join_metadata(registration, publication, manifest, index):
    return _join_metadata_for_codec(registration, publication, manifest, index, LEGACY_CODEC)


def _join_submitted_metadata(registration, publication, manifest, index):
    return _join_metadata_for_codec(registration, publication, manifest, index, SUBMITTED_CODEC)


def _join_metadata_for_codec(registration, publication, manifest, index, codec):
    # The caller already bounded metadata files and allocation. The same
    # original schema checker is used here and by the final full reader.
    if codec_from_publication(publication) is not codec:
        raise ValueError("historical publication/manifest wire family mismatch; no conversion")
    original._manifest(manifest, original.CorpusLimits(), codec=codec)
    fields(index, ("schema", "manifest_sha256", "record_count", "chunks", "origins"),
           "historical original corpus index")
    if validate_container_pair(manifest, index) is not codec:
        raise ValueError("historical original corpus index domain required")
    for key, registration_key in (("source_commit", "collection_source_commit"),
                                  ("source_sha256", "collection_source_sha256"),
                                  ("runtime_sha256", "collection_runtime_library_sha256")):
        if publication[key] != registration[registration_key] or manifest[key] != registration[registration_key]:
            raise ValueError("historical collection source/runtime file domain mismatch")
    for key in ("model_semantics_sha256", "model_epoch", "context_epoch", "recipe_sha256", "stage_id"):
        if manifest[key] != registration[key]:
            raise ValueError("historical manifest model/epoch/recipe/stage mismatch")
    if (digest(codec.manifest_domain, manifest) != publication["canonical_manifest_sha256"]
            or digest(codec.index_domain, index) != publication["canonical_index_sha256"]
            or index.get("manifest_sha256") != publication["canonical_manifest_sha256"]):
        raise ValueError("historical publication canonical manifest/index mismatch")
    if (index.get("record_count") != publication["record_count"] or type(index.get("chunks")) is not list
            or len(index["chunks"]) != len(publication["chunks"])):
        raise ValueError("historical publication index count/chunk mismatch")
    for chunk, pin in zip(index["chunks"], publication["chunks"]):
        fields(chunk, ("path", "bytes", "sha256", "records"), "historical original index chunk")
        if any(chunk[key] != pin[key] for key in ("path", "bytes", "sha256")):
            raise ValueError("historical publication/index exact chunk pin mismatch")


def _source_aggregate(commit, components):
    if type(components) is not dict or not 1 <= len(components) <= 64:
        raise ValueError("bounded original collection source components required")
    framed = bytearray(b"rz-rovent-source-aggregate/1\0" + commit.encode("ascii"))
    framed.extend(len(components).to_bytes(4, "little"))
    for name in sorted(components):
        identifier(name)
        raw = name.encode("utf-8")
        framed.extend(len(raw).to_bytes(4, "little"))
        framed.extend(raw)
        framed.extend(sha(components[name]).encode("ascii"))
    from hashlib import sha256
    return sha256(framed).hexdigest()


def _input_lock(value, expected_canonical):
    fields(value, ("domain", "manifest", "canonical_sha256"), "historical original input lock")
    manifest = value["manifest"]
    fields(manifest, ("domain", "contract_revision", "run_id", "purpose", "engine", "resources",
                      "coordinator", "wall_time_max_ms", "cleanup_max_ms", "output_bytes_max",
                      "agent_renewable", "provenance"), "historical original execution manifest")
    if (value["domain"] != "rz-rovent-input-lock/1" or manifest["domain"] != "rz-rovent-execution/1"
            or value["canonical_sha256"] != expected_canonical
            or digest("rz-rovent-execution-canonical/1", manifest) != sha(expected_canonical)):
        raise ValueError("historical input lock canonical metadata seal mismatch")
    return value


def _runtime_bundle_digest(value):
    """Independent metadata codec; no library is opened or dlopened here."""
    if type(value) is not dict:
        raise ValueError("historical original CUDA bundle object required")
    fields(value, ("schema_version", "files") + (("loader_aliases",) if "loader_aliases" in value else ()),
        "historical original CUDA bundle")
    schema = value["schema_version"]
    if type(schema) is not int or schema not in (1, 2):
        raise ValueError("historical explicit CUDA bundle schema 1/2 required")
    nvidia = ("libcudart.so.12", "libcublas.so.12", "libcublasLt.so.12", "libcufft.so.11", "libcurand.so.10",
        "libnvrtc.so.12", "libnvrtc-builtins.so.12.8", "libnvJitLink.so.12", "libcudnn.so.9", "libcudnn_adv.so.9",
        "libcudnn_cnn.so.9", "libcudnn_ops.so.9", "libcudnn_engines_precompiled.so.9",
        "libcudnn_engines_runtime_compiled.so.9", "libcudnn_graph.so.9", "libcudnn_heuristic.so.9")
    names = {"libonnxruntime.so.1.22.0": ("core", 1), "libonnxruntime_providers_shared.so": ("providers_shared", 2),
        "libonnxruntime_providers_cuda.so": ("providers_cuda", 3), **{name: ("nvidia_dependency", 4) for name in nvidia}}
    files = value["files"]
    if type(files) is not list or len(files) != len(names):
        raise ValueError("historical CUDA bundle requires the closed 19-file profile")
    seen, total = set(), 0
    framed = bytearray(b"RoveZero/native-cuda-bundle\0" +
        (b"binary-codec-v1\0" if schema == 1 else b"binary-codec-v2-aliases\0"))
    framed.extend(schema.to_bytes(4, "big") + len(files).to_bytes(4, "big"))
    for item in files:
        fields(item, ("role", "filename", "bytes", "sha256"), "historical CUDA library metadata")
        name = item["filename"]
        if type(name) is not str or name not in names or name in seen or item["role"] != names[name][0]:
            raise ValueError("historical CUDA library role/name/duplicate mismatch")
        seen.add(name)
        total += uint(item["bytes"], 1024**3)
        if not item["bytes"] or total > 4 * 1024**3:
            raise ValueError("historical CUDA library declared byte capacity exceeded")
        sha(item["sha256"])
    for item in sorted(files, key=lambda row: row["filename"]):
        name = item["filename"].encode("ascii")
        framed.extend(len(name).to_bytes(4, "big") + name + bytes([names[item["filename"]][1]]))
        framed.extend(item["bytes"].to_bytes(8, "big") + bytes.fromhex(item["sha256"]))
    aliases = value.get("loader_aliases")
    if schema == 1:
        if aliases is not None:
            raise ValueError("historical CUDA V1 cannot declare loader aliases")
    else:
        allowed = {name + ".8.0": name for name in nvidia if name.startswith("libcudnn")}
        if type(aliases) is not list or not 1 <= len(aliases) <= len(allowed):
            raise ValueError("historical CUDA V2 requires explicit closed loader aliases")
        seen = set()
        for item in aliases:
            fields(item, ("filename", "target"), "historical CUDA loader alias")
            name = item["filename"]
            if type(name) is not str or name in seen or allowed.get(name) != item["target"]:
                raise ValueError("historical CUDA loader alias changed closed profile")
            seen.add(name)
        framed.extend(len(aliases).to_bytes(4, "big"))
        for item in sorted(aliases, key=lambda row: row["filename"]):
            for name in (item["filename"], item["target"]):
                raw = name.encode("ascii")
                framed.extend(len(raw).to_bytes(4, "big") + raw)
    return hashlib.sha256(framed).hexdigest()


def _join_record(registration, record, admissions, runtime_bundles=None):
    native = record["native"]
    for name in ("raw_submission", "raw_completion", "raw_retirement"):
        if native[name].get("schema") != NATIVE_DOMAIN:
            raise ValueError("historical publication requires explicit native raw /2; no conversion")
    return _join_record_assets(registration, record, admissions, runtime_bundles)


def _join_submitted_record(registration, manifest, record, admissions, runtime_bundles=None):
    """Require submitted raw /3 and its original selected metadata, never /2."""
    if validate_record_codec(manifest, record) is not SUBMITTED_CODEC:
        raise ValueError("historical submitted publication requires its explicit record family")
    from .rovent_native_v3 import validate_native_v3
    validate_native_v3(manifest, record)
    return _join_record_assets(registration, record, admissions, runtime_bundles)


def _join_record_assets(registration, record, admissions, runtime_bundles):
    """Original lock, loaded assets and root seals shared by the two codecs."""
    native = record["native"]
    retired = native["raw_retirement"]
    admission = retired.get("root_admission")
    from .rovent_external_root import ROOT_DOMAIN as EXTERNAL_ROOT_DOMAIN, ROOT_FIELDS, join_locked_external_source
    root_domain = admission.get("schema") if type(admission) is dict else None
    external = (record.get("schema") == SUBMITTED_CODEC.record_domain
        and all(native[name].get("schema") == SUBMITTED_CODEC.native_domains[0]
            for name in ("raw_submission", "raw_completion", "raw_retirement"))
        and root_domain == EXTERNAL_ROOT_DOMAIN)
    if root_domain != ROOT_DOMAIN and not external:
        raise ValueError("historical publication requires original root admission body")
    if external:
        fields(admission, ROOT_FIELDS, "historical external Collection root /2")
    admission_sha = sha(retired.get("root_admission_sha256"))
    origin = admission.get("native_origin")
    if (digest(root_domain, admission) != admission_sha or type(origin) is not dict
            or origin.get("schema") != ORIGIN_DOMAIN
            or digest(ORIGIN_DOMAIN, origin) != sha(admission.get("native_origin_sha256"))):
        raise ValueError("historical original root/native origin seal mismatch")
    lock = admissions.get(admission_sha)
    if lock is None or origin.get("input_lock_sha256") != lock["canonical_sha256"]:
        raise ValueError("historical root admission lacks its published original input lock")
    if external:
        join_locked_external_source(admission, lock)
    engine = lock["manifest"]["engine"]
    if type(engine) is not dict:
        raise ValueError("historical original lock engine object required")
    bundle = None if runtime_bundles is None else runtime_bundles.get(admission_sha)
    if origin.get("runtime_bundle_sha256") is None:
        if bundle is not None:
            raise ValueError("historical CPU origin cannot acquire a CUDA bundle")
    else:
        if bundle is None or _runtime_bundle_digest(bundle["body"]) != origin["runtime_bundle_sha256"]:
            raise ValueError("historical runtime bundle semantic digest differs from its raw metadata file")
        artifacts = lock["manifest"].get("provenance")
        if (type(artifacts) is not list or not any(type(item) is dict and item.get("sha256") == bundle["pin"]["sha256"]
                and item.get("bytes") == bundle["pin"]["bytes"] for item in artifacts)):
            raise ValueError("historical runtime bundle raw file pin absent from original lock provenance")
        core = next(row for row in bundle["body"]["files"] if row["role"] == "core")
        if core["sha256"] != origin.get("runtime_library_sha256"):
            raise ValueError("historical CUDA bundle core is not the registered collection runtime library")
    if (type(engine) is not dict or engine.get("source_commit") != registration["collection_source_commit"]
            or engine.get("source_components") != origin.get("source_components")
            or _source_aggregate(registration["collection_source_commit"], engine["source_components"])
                != registration["collection_source_sha256"]):
        raise ValueError("historical original lock/source component aggregate mismatch")
    binding = registration["role_bindings"].get(record["role"])
    if binding is None:
        raise ValueError("historical row role was not registered")
    if (origin.get("source_commit") != registration["collection_source_commit"]
            or origin.get("source_sha256") != registration["collection_source_sha256"]
            or native["source_sha256"] != registration["collection_source_sha256"]
            or origin.get("runtime_sha256_domain") != "loaded_runtime_library_file_sha256"
            or origin.get("runtime_library_sha256") != registration["collection_runtime_library_sha256"]
            or origin.get("runtime_sha256") != origin["runtime_library_sha256"]):
        raise ValueError("historical row collection source/runtime library domain mismatch")
    for key, observed in (("model_semantics_sha256", origin.get("model_semantics_sha256")),
                          ("model_identity_sha256", native["model_sha256"]),
                          ("weights_sha256", origin.get("weights_sha256")),
                          ("encoding_sha256", native["encoding_sha256"])):
        if observed != binding[key]:
            raise ValueError("historical row differs from registered collection role model/weights/encoding")
    return admission_sha


def _read_observation(budget):
    observed = budget.observation()
    fields(observed, ("charged_bytes", "unknown", "maximum_bytes"), "historical shared read observation")
    if (type(observed["unknown"]) is not bool or uint(observed["maximum_bytes"], _MAX_SHARED_READ_BYTES) == 0
            or uint(observed["charged_bytes"], _MAX_SHARED_READ_BYTES) > observed["maximum_bytes"]):
        raise ValueError("bounded exact historical shared read observation required")
    return copy.deepcopy(observed)


class HistoricalRoventCorpus:
    """Typed historical data owner; original authority stays in the Rust parent."""
    def __init__(self):
        raise TypeError("use pinned historical corpus registration")

    @classmethod
    def from_registration(cls, registration, *, read_budget, ancestors=(), limits=original.CorpusLimits()):
        if (type(limits) is not original.CorpusLimits or not all(callable(getattr(read_budget, name, None))
                for name in ("reserve_external", "charge_external", "finish_external", "fail_external", "observation"))):
            raise ValueError("typed bounded historical read budget and corpus limits required")
        if (not isinstance(ancestors, (tuple, list)) or len(ancestors) > 256
                or any(type(owner) is not cls for owner in ancestors)):
            raise ValueError("bounded typed historical ancestor owners required")
        root = _registration(registration)
        registration = copy.deepcopy(registration)
        publication_body = _read_file(root, registration["publication"], read_budget,
                                      relative=False, json_metadata=True)
        codec = codec_from_publication(publication_body)
        if codec is LEGACY_CODEC:
            publication = _publication(publication_body, limits)
            join_metadata = _join_metadata
            read_corpus = original.RoventTrainingCorpus.from_files
        else:
            publication = _submitted_publication(publication_body, limits)
            join_metadata = _join_submitted_metadata
            read_corpus = original.RoventTrainingCorpus.from_submitted_files
        manifest = _read_file(root, publication["manifest"], read_budget, json_metadata=True)
        index = _read_file(root, publication["index"], read_budget, json_metadata=True)
        join_metadata(registration, publication, manifest, index)
        for pin in publication["chunks"]:
            _read_file(root, pin, read_budget)
        locks_by_path, admissions, bundles_by_sha, runtime_bundles = {}, {}, {}, {}
        for entry in publication["admissions"]:
            pin = entry["input_lock"]
            path = pin["path"]
            if path not in locks_by_path:
                locks_by_path[path] = _input_lock(
                    _read_file(root, pin, read_budget, json_metadata=True), entry["canonical_input_lock_sha256"])
            lock = locks_by_path[path]
            if lock["canonical_sha256"] != entry["canonical_input_lock_sha256"]:
                raise ValueError("historical shared original input lock canonical seal changed")
            admissions[entry["native_admission_sha256"]] = lock
            pin = entry["runtime_bundle_manifest"]
            if pin is not None:
                if pin["sha256"] not in bundles_by_sha:
                    body = _read_file(root, pin, read_budget, json_metadata=True)
                    _runtime_bundle_digest(body)
                    bundles_by_sha[pin["sha256"]] = {"body": body, "pin": pin}
                runtime_bundles[entry["native_admission_sha256"]] = bundles_by_sha[pin["sha256"]]
        declared_ancestors = tuple(manifest["ancestor_index_sha256"])
        # The final reader owns the decoded manifest/index. Drop the first-pass
        # copies before that reader allocates its bounded row/index storage.
        del manifest, index
        # Reuse the full original decoder and bounded owner rows. Every read in
        # this second pass uses the same per-call budget as the publication,
        # first pass and original lock artifacts; there is no global patch.
        corpus = read_corpus(root / publication["manifest"]["path"],
            manifest_sha256=publication["manifest"]["sha256"],
            index_path=root / publication["index"]["path"], index_sha256=publication["index"]["sha256"],
            ancestors=tuple(owner.corpus for owner in ancestors), limits=limits, read_accounting=read_budget)
        if (corpus._codec is not codec
                or corpus.manifest_semantic_sha256 != publication["canonical_manifest_sha256"]
                or corpus.index_semantic_sha256 != publication["canonical_index_sha256"]):
            raise ValueError("historical original reader changed publication canonical seals")
        if codec is LEGACY_CODEC:
            actual_admissions = sorted({_join_record(registration, record, admissions, runtime_bundles)
                for record in corpus._records})
        else:
            actual_admissions = sorted({_join_submitted_record(registration, corpus._manifest, record,
                admissions, runtime_bundles) for record in corpus._records})
        if actual_admissions != publication["native_admission_sha256"]:
            raise ValueError("historical publication full native admission set mismatch")
        expected_ancestors = sorted(owner.corpus.index_semantic_sha256 for owner in ancestors)
        if expected_ancestors != list(declared_ancestors) or len(set(expected_ancestors)) != len(expected_ancestors):
            raise ValueError("historical original ancestor publication set mismatch")
        owner = cls.__new__(cls)
        owner._corpus, owner._registration, owner._publication, owner._codec = corpus, registration, publication, codec
        owner._observation = {"schema": OBSERVATION_DOMAIN, "historical_evidence_only": True,
            "publication_sha256": registration["publication"]["sha256"],
            "manifest_sha256": corpus.corpus_sha256, "index_sha256": corpus.index_sha256,
            "canonical_manifest_sha256": corpus.manifest_semantic_sha256,
            "canonical_index_sha256": corpus.index_semantic_sha256,
            "record_count": len(corpus._records), "collection_source_sha256": registration["collection_source_sha256"],
            "collection_runtime_library_sha256": registration["collection_runtime_library_sha256"],
            "model_semantics_sha256": registration["model_semantics_sha256"],
            "model_epoch": registration["model_epoch"], "context_epoch": registration["context_epoch"],
            "recipe_sha256": registration["recipe_sha256"], "stage_id": registration["stage_id"],
            "role_bindings": copy.deepcopy(registration["role_bindings"]),
            "ancestor_index_sha256": expected_ancestors, "native_admission_sha256": actual_admissions,
            "past_native_authority_reissued": False, "past_physical_completion_reissued": False,
            "execution_authorized": False, "rules_replay_scope": "rust_parent_validation_required",
            "full_input_lock_scope": "canonical_metadata_join_only_rust_parent_validation_required",
            "lineage_scope": "original_stage_admission_and_rust_parent_validation_required",
            "read_accounting": _read_observation(read_budget)}
        owner._owner_sha256 = digest(OBSERVATION_DOMAIN, [registration, publication, owner._observation])
        return owner

    def _check(self):
        if (self._codec is not codec_from_publication(self._publication)
                or self._corpus._codec is not self._codec):
            raise ValueError("historical corpus owner changed its original wire family")
        if digest(OBSERVATION_DOMAIN, [self._registration, self._publication, self._observation]) != self._owner_sha256:
            raise ValueError("historical corpus observation owner metadata changed")
        self._corpus._check_owner(full=True)

    @property
    def corpus(self):
        self._check()
        return self._corpus

    def observation(self):
        self._check()
        return copy.deepcopy(self._observation)
