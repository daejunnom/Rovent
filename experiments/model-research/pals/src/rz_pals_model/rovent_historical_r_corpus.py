"""Explicit historical R publications, without restoring original authority.

All file bytes, NNUEC assets, lock snapshots and complete retirement coverage
are checked in the shared bounded reader. Original source paths are preserved
as declarations and are never opened. Rust must separately load the retained
NNUEC, replay Rules, rebuild ReviewView seals and validate current parent RPC
custody. These Python observations cannot issue those capabilities.
"""
from __future__ import annotations

import copy
import math
import os
from pathlib import Path

from .generation_data import digest, fields, identifier, sha, uint
from . import rovent_corpus as pc
from . import rovent_r_corpus as r
from . import rovent_r_readout as wire
from .rovent_historical_corpus import (_absolute, _relative_pin, _input_lock,
    _read_observation, _runtime_bundle_digest, _source_aggregate)
from .rovent_native_v2 import _artifact, _commit

PUBLICATION_DOMAIN = "rz-rovent-r-readout-publication/1"
RETIREMENT_OBSERVATION_DOMAIN = "rz-rovent-r-retirement-observation/1"
NNUEC_REGISTRATION_DOMAIN = "rz-rovent-r-nnuec-registration/1"
OBSERVATION_DOMAIN = "rz-rovent-historical-r-corpus-observation/1"
DEPENDENCIES_DOMAIN = "rz-rovent-historical-r-dependencies/1"
_MAX_METADATA = 16 << 20
_MAX_WEIGHTS = 32 << 20
_MAX_PROVENANCE = 256 << 10
_MAX_BUNDLE = 64 << 10
_NNUEC_FIELDS = ("schema", "semantics", "encoding", "implementation_sha256", "configuration_sha256", "weights", "provenance",
    "diagnostic", "context_namespace_policy_sha256", "local_namespace_policy_sha256", "search_conditions_sha256")


def _nnuec_pin(value):
    fields(value, _NNUEC_FIELDS, "historical R declared NNUEC")
    if value["schema"] != "rz-nnuec-fp32-v2":
        raise ValueError("historical actual R requires explicit NNUEC V2")
    for key in ("semantics", "encoding"):
        identifier(value[key])
    for key in ("implementation_sha256", "configuration_sha256", "context_namespace_policy_sha256", "local_namespace_policy_sha256", "search_conditions_sha256"):
        sha(value[key])
    _artifact(value["weights"])
    _artifact(value["provenance"])
    if (type(value["diagnostic"]) is not bool or value["weights"]["bytes"] > _MAX_WEIGHTS
            or value["provenance"]["bytes"] > _MAX_PROVENANCE):
        raise ValueError("historical R NNUEC source declarations exceed bounds")
    return value


def _registration(value):
    fields(value, ("root", "publication", "collection_source_commit", "collection_source_sha256",
        "collection_runtime_library_sha256", "collection_model_semantics_sha256", "collection_model_epoch", "collection_context_epoch",
        "collection_recipe_sha256", "collection_stage_id", "nnuec", "nnuec_parameter_sha256"), "historical R dataset registration")
    root = _absolute(value["root"], "R publication root")
    pin = fields(value["publication"], ("path", "sha256", "bytes"), "historical R publication file pin")
    path = _absolute(pin["path"], "R publication file")
    if path.parent != root or path.name != "r-publication.json":
        raise ValueError("historical R publication must be root/r-publication.json")
    sha(pin["sha256"])
    r._positive(pin["bytes"], _MAX_METADATA)
    _commit(value["collection_source_commit"])
    for key in ("collection_source_sha256", "collection_runtime_library_sha256", "collection_model_semantics_sha256",
                "collection_recipe_sha256", "nnuec_parameter_sha256"):
        sha(value[key])
    for key in ("collection_model_epoch", "collection_context_epoch"):
        r._positive(value[key])
    identifier(value["collection_stage_id"])
    _nnuec_pin(value["nnuec"])
    return root


def _pin(value, maximum, description):
    return _relative_pin(value, maximum=maximum, description=description)


def _publication(value, limits):
    fields(value, ("schema", "manifest", "index", "chunks", "retirements", "canonical_manifest_sha256", "canonical_index_sha256",
        "source_commit", "source_sha256", "runtime_sha256", "record_count", "total_payload_bytes", "admissions"), "historical R publication")
    if value["schema"] != PUBLICATION_DOMAIN:
        raise ValueError("explicit historical R publication domain required")
    for key in ("canonical_manifest_sha256", "canonical_index_sha256", "source_sha256", "runtime_sha256"):
        sha(value[key])
    _commit(value["source_commit"])
    uint(value["record_count"], limits.max_records)
    for key, expected in (("manifest", "r-corpus-manifest.json"), ("index", "r-corpus-index.json")):
        _pin(value[key], min(_MAX_METADATA, limits.max_file_bytes), "historical R metadata file")
        if value[key]["path"] != expected:
            raise ValueError("historical R canonical metadata names required")
    if (type(value["chunks"]) is not list or len(value["chunks"]) > min(8192, limits.max_records)
            or type(value["retirements"]) is not list or not 1 <= len(value["retirements"]) <= limits.max_records
            or type(value["admissions"]) is not list or not 1 <= len(value["admissions"]) <= limits.max_records):
        raise ValueError("bounded historical R chunks/retirements/admissions required")
    files, chunk_paths, retired, rows, source_files = {}, set(), set(), set(), {}

    def add(pin):
        path = pin["path"].casefold() if os.name == "nt" else pin["path"]
        if path in files and files[path] != pin:
            raise ValueError("historical R payload path has conflicting pins")
        files[path] = pin

    add(value["manifest"])
    add(value["index"])
    for chunk in value["chunks"]:
        _pin(chunk, min(32 << 20, limits.max_file_bytes), "historical R chunk file")
        path = chunk["path"].casefold() if os.name == "nt" else chunk["path"]
        if not chunk["path"].startswith("r-chunks/") or not chunk["path"].endswith(".jsonl") or path in chunk_paths:
            raise ValueError("historical R chunk path/duplicate differs")
        chunk_paths.add(path)
        add(chunk)
    for terminal in value["retirements"]:
        fields(terminal, ("retirement_sha256", "file", "readout_record_sha256"), "historical R complete retirement")
        identity = sha(terminal["retirement_sha256"])
        _pin(terminal["file"], min(_MAX_METADATA, limits.max_file_bytes), "historical R retirement file")
        if terminal["file"]["path"] != "r-retirements/" + identity + ".json" or identity in retired:
            raise ValueError("historical R retirement path/duplicate differs")
        pc._sha_list(terminal["readout_record_sha256"], 64, "complete R retirement rows")
        for record in terminal["readout_record_sha256"]:
            if record in rows:
                raise ValueError("historical R row belongs to more than one retirement")
            rows.add(record)
        retired.add(identity)
        add(terminal["file"])
    if len(rows) != value["record_count"]:
        raise ValueError("historical R complete publication row coverage differs")
    last, canonical_locks = None, {}
    for admission in value["admissions"]:
        fields(admission, ("native_admission_sha256", "input_lock", "canonical_input_lock_sha256", "runtime_bundle_manifest",
            "nnuec_weights", "nnuec_provenance", "nnuec_parameter_sha256", "nnuec_registration", "nnuec_registration_sha256"), "historical R admission")
        identity = sha(admission["native_admission_sha256"])
        if last is not None and last >= identity:
            raise ValueError("sorted unique historical R admissions required")
        last = identity
        canonical = sha(admission["canonical_input_lock_sha256"])
        sha(admission["nnuec_parameter_sha256"])
        sha(admission["nnuec_registration_sha256"])
        for name, maximum, directory, filename in (
            ("input_lock", _MAX_METADATA, "input-locks/", canonical),
            ("nnuec_weights", _MAX_WEIGHTS, "nnuec-weights/", admission["nnuec_weights"].get("sha256") if type(admission["nnuec_weights"]) is dict else ""),
            ("nnuec_provenance", _MAX_PROVENANCE, "nnuec-provenance/", admission["nnuec_provenance"].get("sha256") if type(admission["nnuec_provenance"]) is dict else "")):
            pin = _pin(admission[name], min(maximum, limits.max_file_bytes), "historical R original source file")
            if pin["path"] != directory + filename + ".json":
                raise ValueError("historical R original source path differs from content pin")
            key = (name, pin["sha256"])
            if key in source_files and source_files[key] != pin:
                raise ValueError("one historical R raw source SHA has conflicting pins")
            source_files[key] = pin
            add(pin)
        if canonical in canonical_locks and canonical_locks[canonical] != admission["input_lock"]:
            raise ValueError("one historical R canonical lock has multiple artifacts")
        canonical_locks[canonical] = admission["input_lock"]
        if digest(NNUEC_REGISTRATION_DOMAIN, admission["nnuec_registration"]) != admission["nnuec_registration_sha256"]:
            raise ValueError("historical R original NNUEC registration projection seal differs")
        bundle = admission["runtime_bundle_manifest"]
        if bundle is not None:
            _pin(bundle, min(_MAX_BUNDLE, limits.max_file_bytes), "historical R CUDA bundle file")
            if bundle["path"] != "runtime-bundles/" + bundle["sha256"] + ".json":
                raise ValueError("historical R original bundle raw-file path differs")
            add(bundle)
    total = sum(pin["bytes"] for pin in files.values())
    if uint(value["total_payload_bytes"], limits.max_total_bytes) != total:
        raise ValueError("historical R exact unique payload byte total differs")
    return value


class _Reads:
    def __init__(self, root, budget, limits, publication_pin):
        if not all(callable(getattr(budget, method, None)) for method in
                   ("reserve_external", "charge_external", "finish_external", "fail_external", "observation")):
            raise ValueError("historical R shared bounded read accounting required")
        _read_observation(budget)
        self.root, self.budget, self.limits = root, budget, limits
        self._publication_path = publication_pin["path"]
        self._pins, self._used = {}, 0

    def read(self, pin, *, maximum, relative=True, json_body=True):
        path = self.root.joinpath(*pin["path"].split("/")) if relative else _absolute(pin["path"], "R source file")
        if relative and path == Path(self._publication_path):
            raise ValueError("historical R payload overlaps its publication")
        identity = str(path).casefold() if os.name == "nt" else str(path)
        if identity in self._pins and self._pins[identity] != pin:
            raise ValueError("historical R same file has conflicting raw pin")
        if identity not in self._pins:
            self._used += uint(pin["bytes"], maximum)
            if self._used > self.limits.max_total_bytes:
                raise ValueError("historical R aggregate registered source bytes exceeded")
            self._pins[identity] = copy.deepcopy(pin)
        raw = pc._read_pinned_bytes(path, pin["sha256"], pin["bytes"], maximum,
            suffixes=(".json",) if json_body else (".jsonl",), read_accounting=self.budget)
        return pc._decode_json(raw) if json_body else raw


def _join_metadata(registration, publication, manifest, index=None):
    fields(manifest, ("schema", "collection", "pc_ancestor_index_sha256", "r_ancestor_index_sha256"), "historical R manifest")
    if manifest["schema"] != wire.MANIFEST_DOMAIN:
        raise ValueError("historical R manifest domain differs")
    collection = pc._manifest(manifest["collection"], pc.CorpusLimits())
    for key in ("pc_ancestor_index_sha256", "r_ancestor_index_sha256"):
        pc._sha_list(manifest[key], 256, key)
    for key, registered in (("source_commit", "collection_source_commit"), ("source_sha256", "collection_source_sha256"),
        ("runtime_sha256", "collection_runtime_library_sha256"), ("model_semantics_sha256", "collection_model_semantics_sha256"),
        ("model_epoch", "collection_model_epoch"), ("context_epoch", "collection_context_epoch"),
        ("recipe_sha256", "collection_recipe_sha256"), ("stage_id", "collection_stage_id")):
        if type(collection[key]) is not type(registration[registered]) or collection[key] != registration[registered]:
            raise ValueError("historical R manifest differs from registered collection identity")
    if (publication["source_commit"] != collection["source_commit"] or publication["source_sha256"] != collection["source_sha256"]
            or publication["runtime_sha256"] != collection["runtime_sha256"]
            or digest(wire.MANIFEST_DOMAIN, manifest) != publication["canonical_manifest_sha256"]):
        raise ValueError("historical R publication source/runtime/canonical manifest differs")
    if index is not None:
        fields(index, ("schema", "manifest_sha256", "record_count", "chunks", "origins"), "historical R index")
        if (index["schema"] != wire.INDEX_DOMAIN or index["manifest_sha256"] != publication["canonical_manifest_sha256"]
                or digest(wire.INDEX_DOMAIN, index) != publication["canonical_index_sha256"]
                or type(index["record_count"]) is not int or index["record_count"] != publication["record_count"]
                or type(index["chunks"]) is not list or len(index["chunks"]) != len(publication["chunks"])):
            raise ValueError("historical R original index/coverage/canonical seal differs")
        for chunk, pin in zip(index["chunks"], publication["chunks"], strict=True):
            fields(chunk, ("path", "sha256", "bytes", "records"), "historical R indexed chunk")
            if any(chunk[key] != pin[key] for key in ("path", "sha256", "bytes")):
                raise ValueError("historical R publication/index exact chunk pin differs")


def read_historical_r_dependencies(registration, *, read_budget, limits=pc.CorpusLimits()):
    """Only pinned graph metadata; final intake reads and verifies all files."""
    if type(limits) is not pc.CorpusLimits:
        raise ValueError("typed finite historical R limits required")
    root = _registration(registration)
    reader = _Reads(root, read_budget, limits, registration["publication"])
    publication = _publication(reader.read(registration["publication"], maximum=min(_MAX_METADATA, limits.max_file_bytes), relative=False), limits)
    manifest = reader.read(publication["manifest"], maximum=min(_MAX_METADATA, limits.max_file_bytes))
    _join_metadata(registration, publication, manifest)
    return {"schema": DEPENDENCIES_DOMAIN, "publication_sha256": registration["publication"]["sha256"],
        "canonical_manifest_sha256": publication["canonical_manifest_sha256"], "canonical_index_sha256": publication["canonical_index_sha256"],
        "pc_ancestor_index_sha256": copy.deepcopy(manifest["pc_ancestor_index_sha256"]),
        "r_ancestor_index_sha256": copy.deepcopy(manifest["r_ancestor_index_sha256"]), "execution_authorized": False}


def _weights_metadata(value):
    """Validate plain source operands without creating a model/checkpoint."""
    from . import nnuec_reference as base
    from . import nnuec_v2_reference as v2
    fields(value, ("schema", "semantics", "base", "parameters"), "historical R V2 weights source")
    if value["schema"] != v2.SCHEMA or value["semantics"] != v2.SEMANTICS:
        raise ValueError("historical R weights require exact origin-first V2 semantics")
    original = fields(value["base"], ("schema", "semantics", "config", "training", "parameters", "output_scale"), "historical R V2 base operands")
    if original["schema"] != base.SCHEMA or original["semantics"] != base.SEMANTICS:
        raise ValueError("historical R explicit V1 base declaration differs")
    config = base.NnuecConfig.from_mapping(original["config"])
    base._training(original["training"])
    scale = original["output_scale"]
    if not _finite_source_number(scale) or not 0 < scale <= 10000:
        raise ValueError("historical R finite positive original output scale required")
    arrays = fields(original["parameters"], base.PARAMETER_FIELDS, "historical R base parameter operands")
    extra = fields(value["parameters"], v2.PARAMETER_FIELDS, "historical R V2 relation operands")
    e, c = config.line_width, config.context_width
    lengths = [*config.parameter_lengths().items(), *zip(v2.PARAMETER_FIELDS, (4*e, 2*e, 2*e, 2*e, c*e), strict=True)]
    for name, length in lengths:
        values = arrays[name] if name in arrays else extra[name]
        if type(values) is not list or len(values) != length:
            raise ValueError("historical R source parameter operand shape differs")
        for number in values:
            if not _finite_source_number(number) or abs(base.f32(number)) > base.MAX_PARAMETER_MAGNITUDE:
                raise ValueError("historical R finite bounded FP32 source operands required")
    return config, copy.deepcopy(original["training"])


def _finite_source_number(value):
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _nnuec_registration(value, expected, parameter, training):
    fields(value, ("domain", "schema", "semantics", "encoding", "implementation_sha256", "configuration_sha256",
        "context_namespace_policy", "local_namespace_policy", "context_namespace_policy_sha256", "local_namespace_policy_sha256",
        "model_identity"), "historical R original NNUEC registration")
    if value["domain"] != "rz-nnuec-v2-registration/1":
        raise ValueError("historical R NNUEC registration domain differs")
    for key in ("schema", "semantics", "encoding", "implementation_sha256", "configuration_sha256",
                "context_namespace_policy_sha256", "local_namespace_policy_sha256"):
        if value[key] != expected[key]:
            raise ValueError("historical R original NNUEC registration differs from declared source")
    for key in ("context_namespace_policy", "local_namespace_policy"):
        text = value[key]
        if type(text) is not str or not text or len(text.encode()) > 4096 or any(ord(char) < 32 for char in text):
            raise ValueError("historical R bounded original namespace policy required")
    identity = fields(value["model_identity"], ("semantics", "weights_sha256", "training"), "historical R semantic parameter identity")
    if identity["semantics"] != expected["semantics"] or identity["weights_sha256"] != parameter or identity["training"] != training:
        raise ValueError("historical R semantic parameter identity/training declaration differs")


def _provenance(value, expected, parameter, training, lock):
    fields(value, ("domain", "weights_raw_sha256", "semantic_weights_sha256", "schema", "semantics", "encoding", "implementation_sha256",
        "configuration_sha256", "context_namespace_policy_sha256", "local_namespace_policy_sha256", "training", "diagnostic", "provenance"), "historical R NNUEC provenance")
    if (value["domain"] != "rz-rovent-nnuec-provenance/1" or value["weights_raw_sha256"] != expected["weights"]["sha256"]
            or value["semantic_weights_sha256"] != parameter or value["training"] != training
            or type(value["diagnostic"]) is not bool or value["diagnostic"] != expected["diagnostic"]):
        raise ValueError("historical R original NNUEC provenance/raw/semantic identities differ")
    for key in ("schema", "semantics", "encoding", "implementation_sha256", "configuration_sha256", "context_namespace_policy_sha256", "local_namespace_policy_sha256"):
        if value[key] != expected[key]:
            raise ValueError("historical R NNUEC provenance differs from lock registration")
    if type(value["provenance"]) is not list or len(value["provenance"]) > 32:
        raise ValueError("bounded historical R NNUEC provenance artifacts required")
    paths, contents = set(), set()
    for artifact in value["provenance"]:
        _artifact(artifact)
        content = (artifact["sha256"], artifact["bytes"])
        if artifact["path"] in paths or content in contents or artifact not in lock["manifest"]["provenance"]:
            raise ValueError("historical R NNUEC provenance is duplicated or absent from original lock")
        paths.add(artifact["path"])
        contents.add(content)


def _admission_body(value, identity, lock, registration, bundle):
    """Also check empty retirements; never invent a zero-input R row."""
    fields(value, ("schema", "native_origin", "native_origin_sha256", "process_epoch", "game_generation", "root_generation",
        "implementation_sha256", "deadline_clock", "deadline_tick", "root_budget_sha256", "selective_policy_sha256", "root_state",
        "store_root_state", "root_moves", "depth_plies", "mate_moves", "requested_nodes", "root_descriptor", "root_descriptor_sha256"), "historical R original admission body")
    if value["schema"] != r.ROOT_DOMAIN or digest(r.ROOT_DOMAIN, value) != identity:
        raise ValueError("historical R original admission body/seal differs")
    inp = r._root_descriptor(value["root_descriptor"], value["root_descriptor_sha256"], lock)
    for raw, authority in (("process_epoch", "epoch"), ("game_generation", "game"), ("root_generation", "root")):
        if uint(value[raw]) != inp["authority"][authority]:
            raise ValueError("historical R original root authority differs")
    state = fields(value["root_state"], ("owner", "revision", "semantic"), "historical R root state")
    uint(state["owner"])
    uint(state["revision"])
    sha(state["semantic"])
    uint(value["store_root_state"])
    sha(value["selective_policy_sha256"])
    if (value["implementation_sha256"] != inp["authority"]["implementation_sha256"]
            or value["root_budget_sha256"] != value["root_descriptor"]["root_budget_sha256"]
            or uint(value["deadline_clock"]) != inp["clock"]["hard"]["clock_epoch"]
            or uint(value["deadline_tick"]) != inp["clock"]["hard"]["at_ns"]
            or state["semantic"] != inp["state_and_history_sha256"] or value["root_moves"] != inp["allowed_root_moves"]
            or any(value[key] != inp["original_go"][key] for key in ("depth_plies", "mate_moves", "requested_nodes"))):
        raise ValueError("historical R original admission go/root/deadline differs")
    origin = fields(value["native_origin"], ("schema", "purpose", "process_epoch", "mode", "input_lock_sha256", "source_commit", "source_components",
        "source_sha256", "weights_sha256", "export_manifest_sha256", "backend_sha256", "runtime_library_sha256", "runtime_bundle_sha256",
        "native_configuration_sha256", "model_semantics", "model_semantics_sha256", "encoding", "encoding_sha256", "encoding_registry_sha256",
        "model_implementation_sha256", "frozen_epoch", "representation_model_domain", "runtime_sha256_domain", "runtime_sha256", "teacher_profiles"), "historical R original native origin")
    if (origin["schema"] != r.ORIGIN_DOMAIN or origin["purpose"] != "collection"
            or digest(r.ORIGIN_DOMAIN, origin) != sha(value["native_origin_sha256"])
            or uint(origin["process_epoch"]) != value["process_epoch"]):
        raise ValueError("historical R original native origin seal/scope differs")
    for key in ("weights_sha256", "export_manifest_sha256", "backend_sha256", "native_configuration_sha256", "model_implementation_sha256",
                "encoding_sha256", "encoding_registry_sha256"):
        sha(origin[key])
    for key in ("model_semantics", "encoding"):
        identifier(origin[key])
    r._positive(origin["frozen_epoch"])
    engine = lock["manifest"]["engine"]
    if (origin["input_lock_sha256"] != lock["canonical_sha256"] or inp["startup_lock_sha256"] != lock["canonical_sha256"]
            or origin["source_commit"] != registration["collection_source_commit"]
            or engine.get("source_commit") != origin["source_commit"] or engine.get("source_components") != origin["source_components"]
            or _source_aggregate(origin["source_commit"], origin["source_components"]) != registration["collection_source_sha256"]
            or origin["source_sha256"] != registration["collection_source_sha256"]
            or origin["runtime_sha256_domain"] != "loaded_runtime_library_file_sha256"
            or origin["runtime_library_sha256"] != registration["collection_runtime_library_sha256"]
            or origin["runtime_sha256"] != origin["runtime_library_sha256"]
            or origin["model_semantics_sha256"] != registration["collection_model_semantics_sha256"]
            or origin["representation_model_domain"] != "loaded_checkpoint_sha256"
            or origin["teacher_profiles"] != engine.get("teachers")
            or engine.get("nnuec") != registration["nnuec"]
            or origin["mode"] not in ("HybridPcr", "HybridPc", "CpuPcr", "CpuPc")
            or engine.get("mode") != {"HybridPcr": "hybrid_pcr", "HybridPc": "hybrid_pc", "CpuPcr": "cpu_pcr", "CpuPc": "cpu_pc"}[origin["mode"]]):
        raise ValueError("historical R original source/runtime/NNUEC lock domains differ")
    if origin["runtime_bundle_sha256"] is None:
        if bundle is not None:
            raise ValueError("historical CPU R origin cannot acquire a CUDA bundle")
    elif bundle is None or _runtime_bundle_digest(bundle) != sha(origin["runtime_bundle_sha256"]):
        raise ValueError("historical R CUDA spec digest differs from original bundle body")


def _retirement_body(value, expected_sha, record_shas, admission):
    fields(value, ("job", "worker_owner", "frame_sequence", "capture_sequence", "ending", "limits", "attempted", "selected",
        "retained_readouts", "overflow", "retained_input_bytes", "selected_input_preparation_ns", "error"), "historical R retirement body")
    r._job(value["job"])
    uint(value["worker_owner"], 65535)
    r._positive(value["frame_sequence"])
    r._positive(value["capture_sequence"])
    limits = fields(value["limits"], ("maximum_readouts", "every_nth_score", "maximum_input_bytes"), "historical R capture limits")
    r._positive(limits["maximum_readouts"], 64)
    r._positive(limits["every_nth_score"], 2**32-1)
    r._positive(limits["maximum_input_bytes"], 16 << 20)
    for key in ("attempted", "selected", "retained_readouts", "overflow", "retained_input_bytes", "selected_input_preparation_ns"):
        uint(value[key], 64 if key == "retained_readouts" else 2**64-1)
    ending = value["ending"]
    if type(ending) is not dict or ending.get("kind") not in ("cpu_terminal", "not_started", "failed", "unknown"):
        raise ValueError("explicit historical R capture ending required")
    fields(ending, ("kind", "completion") if ending["kind"] == "cpu_terminal" else ("kind",), "historical R capture ending")
    if ending["kind"] == "cpu_terminal":
        identifier(ending["completion"])
    if value["error"] is not None:
        identifier(value["error"])
    authority = value["job"]["authority"]
    if (digest(r.RETIREMENT_DOMAIN, value) != expected_sha or value["retained_readouts"] != len(record_shas)
            or not value["retained_readouts"] <= value["selected"] <= min(value["attempted"], limits["maximum_readouts"])
            or value["retained_input_bytes"] > limits["maximum_input_bytes"]
            or value["retained_readouts"] == 0 and value["retained_input_bytes"] != 0
            or any(authority[key] != admission[key] for key in ("process_epoch", "game_generation", "root_generation", "implementation_sha256"))
            or value["job"]["hard_deadline_clock_epoch"] != admission["deadline_clock"]
            or value["job"]["hard_deadline_ns"] > admission["deadline_tick"]):
        raise ValueError("historical R whole retirement count/seal/scope differs")


def _select_owners(expected, owners, kind):
    if not isinstance(owners, (tuple, list)) or len(owners) > (257 if kind == "pc" else 256):
        raise ValueError("bounded available historical R ancestor owners required")
    indexed = {}
    for owner in owners:
        if kind == "pc":
            if type(owner) is not pc.RoventTrainingCorpus:
                raise ValueError("typed original PC ancestor owner required")
            owner._check_owner(full=True)
            corpus = owner
        else:
            if type(owner) is not HistoricalRoventRCorpus:
                raise ValueError("typed original historical R ancestor owner required")
            corpus = owner.corpus
            corpus._check_owner(full=True)
        identity = corpus.index_semantic_sha256
        if identity in indexed:
            raise ValueError("duplicate available historical source canonical index")
        indexed[identity] = corpus
    if any(identity not in indexed for identity in expected):
        raise ValueError("missing explicitly registered original PC/R ancestor index")
    return tuple(indexed[identity] for identity in expected)


class HistoricalRoventRCorpus:
    """Historical R data owner; actual parent source authority remains opaque."""
    def __init__(self):
        raise TypeError("use explicit pinned historical R registration")

    @classmethod
    def from_registration(cls, registration, *, recipe, read_budget, pc_corpora=(), r_ancestors=(),
                          nnuec_config=None, limits=pc.CorpusLimits()):
        if type(limits) is not pc.CorpusLimits:
            raise ValueError("typed finite historical R limits required")
        root = _registration(registration)
        registration = copy.deepcopy(registration)
        reads = _Reads(root, read_budget, limits, registration["publication"])
        publication = _publication(reads.read(registration["publication"], maximum=min(_MAX_METADATA, limits.max_file_bytes), relative=False), limits)
        manifest = reads.read(publication["manifest"], maximum=min(_MAX_METADATA, limits.max_file_bytes))
        index = reads.read(publication["index"], maximum=min(_MAX_METADATA, limits.max_file_bytes))
        _join_metadata(registration, publication, manifest, index)
        pc_selected = _select_owners(manifest["pc_ancestor_index_sha256"], pc_corpora, "pc")
        r_selected = _select_owners(manifest["r_ancestor_index_sha256"], r_ancestors, "r")
        locks, bundles, source_metadata, admission_map = {}, {}, {}, {}
        expected_config = nnuec_config
        for entry in publication["admissions"]:
            canonical = entry["canonical_input_lock_sha256"]
            if canonical not in locks:
                locks[canonical] = _input_lock(reads.read(entry["input_lock"], maximum=min(_MAX_METADATA, limits.max_file_bytes)), canonical)
            lock = locks[canonical]
            if lock["manifest"]["engine"].get("nnuec") != registration["nnuec"]:
                raise ValueError("historical R loaded lock changes registered NNUEC source")
            parameter = entry["nnuec_parameter_sha256"]
            if parameter != registration["nnuec_parameter_sha256"]:
                raise ValueError("historical R semantic parameter identity differs from registered original source")
            for name in ("weights", "provenance"):
                pin = entry["nnuec_" + name]
                declared = registration["nnuec"][name]
                if any(pin[key] != declared[key] for key in ("sha256", "bytes")):
                    raise ValueError("historical R archived NNUEC file differs from original raw artifact pin")
            weights_pin, provenance_pin = entry["nnuec_weights"], entry["nnuec_provenance"]
            source_key = (weights_pin["sha256"], provenance_pin["sha256"])
            if source_key not in source_metadata:
                weights = reads.read(weights_pin, maximum=min(_MAX_WEIGHTS, limits.max_file_bytes))
                config, training = _weights_metadata(weights)
                del weights
                provenance = reads.read(provenance_pin, maximum=min(_MAX_PROVENANCE, limits.max_file_bytes))
                source_metadata[source_key] = (config, training, provenance)
            config, training, provenance = source_metadata[source_key]
            _provenance(provenance, registration["nnuec"], parameter, training, lock)
            _nnuec_registration(entry["nnuec_registration"], registration["nnuec"], parameter, training)
            if expected_config is None:
                expected_config = config
            elif type(expected_config) is not type(config) or expected_config != config:
                raise ValueError("historical R source dimensions differ from expected registered training configuration")
            bundle_pin = entry["runtime_bundle_manifest"]
            bundle = None
            if bundle_pin is not None:
                if type(lock["manifest"]["provenance"]) is not list or not any(
                    type(artifact) is dict and artifact.get("sha256") == bundle_pin["sha256"] and artifact.get("bytes") == bundle_pin["bytes"]
                    for artifact in lock["manifest"]["provenance"]):
                    raise ValueError("historical R bundle raw file pin absent from original lock provenance")
                if bundle_pin["sha256"] not in bundles:
                    bundles[bundle_pin["sha256"]] = reads.read(bundle_pin, maximum=min(_MAX_BUNDLE, limits.max_file_bytes))
                bundle = bundles[bundle_pin["sha256"]]
                _runtime_bundle_digest(bundle)
            admission_map[entry["native_admission_sha256"]] = (entry, lock, bundle, training)
        if expected_config is None:
            raise ValueError("historical R publication lacks original NNUEC source dimensions")
        terminals, by_record, native_bodies, row_lock_shas = {}, {}, {}, set()
        for entry in publication["retirements"]:
            body = reads.read(entry["file"], maximum=min(_MAX_METADATA, limits.max_file_bytes))
            fields(body, ("schema", "native_admission", "native_admission_sha256", "retirement", "retirement_sha256", "readout_record_sha256"),
                "historical R retirement observation")
            if (body["schema"] != RETIREMENT_OBSERVATION_DOMAIN or body["retirement_sha256"] != entry["retirement_sha256"]
                    or body["readout_record_sha256"] != entry["readout_record_sha256"]):
                raise ValueError("historical R raw retirement observation differs from publication")
            admission_sha = sha(body["native_admission_sha256"])
            if admission_sha not in admission_map:
                raise ValueError("historical R retirement lacks original published admission")
            admission, lock, bundle, _training = admission_map[admission_sha]
            _admission_body(body["native_admission"], admission_sha, lock, registration, bundle)
            if admission_sha in native_bodies and native_bodies[admission_sha] != body["native_admission"]:
                raise ValueError("historical R one admission has conflicting original bodies")
            native_bodies[admission_sha] = body["native_admission"]
            _retirement_body(body["retirement"], body["retirement_sha256"], body["readout_record_sha256"], body["native_admission"])
            terminals[body["retirement_sha256"]] = body
            for identity in body["readout_record_sha256"]:
                if identity in by_record:
                    raise ValueError("historical R row appears in multiple whole retirements")
                by_record[identity] = body["retirement_sha256"]
                row_lock_shas.add(admission["canonical_input_lock_sha256"])
        if set(native_bodies) != set(admission_map):
            raise ValueError("historical R full original admission/retirement set differs")
        if type(index["origins"]) is not list or {item.get("record_sha256") for item in index["origins"] if type(item) is dict} != set(by_record):
            raise ValueError("historical R publication retirement/index occurrence coverage differs")
        input_pins = {}
        for entry in publication["admissions"]:
            canonical = entry["canonical_input_lock_sha256"]
            if canonical in row_lock_shas:
                input_pins[canonical] = entry["input_lock"]
        files = r.RCorpusFiles.from_mapping({"root": str(root), "manifest": publication["manifest"], "index": publication["index"], "input_locks": input_pins})
        # The final R owner reads the same immutable sources again under this
        # same budget. No global patch, uncharged read or second Rules owner.
        del manifest, index
        corpus = r.RoventRCorpus.from_files(files, recipe=recipe, nnuec_config=expected_config,
            expected_nnuec_parameter_sha256=registration["nnuec_parameter_sha256"], pc_corpora=pc_selected,
            r_ancestors=r_selected, limits=limits, read_accounting=read_budget)
        if (corpus.manifest_semantic_sha256 != publication["canonical_manifest_sha256"]
                or corpus.index_semantic_sha256 != publication["canonical_index_sha256"]):
            raise ValueError("historical R final reader changed original publication canonical seals")
        for row in corpus._records:
            terminal = terminals[by_record[row["record_sha256"]]]
            if (row["retirement"] != terminal["retirement"] or row["retirement_sha256"] != terminal["retirement_sha256"]
                    or row["native_admission"] != terminal["native_admission"]
                    or row["native_admission_sha256"] != terminal["native_admission_sha256"]
                    or row["bound"]["training_state"] != admission_map[row["native_admission_sha256"]][3]):
                raise ValueError("historical R row differs from original whole retirement/source declaration")
        owner = cls.__new__(cls)
        owner._corpus, owner._registration, owner._publication = corpus, registration, publication
        # Preserve the original empty/error terminals as well as terminals
        # with readouts. A later metadata supplement must not invent a row to
        # recover them or reopen an unpinned historical path.
        owner._retirements = terminals
        owner._retirement_observation_pins = {identity: digest("rz-rovent-historical-r-retirement-owner/1", body)
            for identity, body in terminals.items()}
        owner._retirement_lookup_sha256 = digest("rz-rovent-historical-r-retirement-lookup/1",
            owner._retirement_observation_pins)
        owner._observation = {"schema": OBSERVATION_DOMAIN, "publication_sha256": registration["publication"]["sha256"],
            "manifest_file_sha256": corpus.corpus_sha256, "index_file_sha256": corpus.index_sha256,
            "canonical_manifest_sha256": corpus.manifest_semantic_sha256, "canonical_index_sha256": corpus.index_semantic_sha256,
            "nnuec_parameter_sha256": registration["nnuec_parameter_sha256"], "nnuec_weights_file_sha256": registration["nnuec"]["weights"]["sha256"],
            "nnuec_provenance_file_sha256": registration["nnuec"]["provenance"]["sha256"], "retirement_count": len(terminals),
            "ancestor_publication_sha256": sorted(ancestor._registration["publication"]["sha256"] for ancestor in r_ancestors
                if ancestor.corpus.index_semantic_sha256 in {source.index_semantic_sha256 for source in r_selected}),
            "empty_retirement_count": sum(not item["readout_record_sha256"] for item in terminals.values()),
            "record_count": len(corpus._records), "reads": _read_observation(read_budget),
            "historical_data_integrity_only": True, "nnuec_parameter_identity_recomputed": False,
            "original_nnuec_loaded_authority_present": False, "actual_rules_replay_verified": False,
            "full_review_view_reconstructed": False, "original_capture_authority_reissued": False,
            "past_native_authority_reissued": False, "past_physical_completion_reissued": False,
            "wdl_original_timeline_authority_verified": False, "old_run_closure_proven": False, "execution_authorized": False}
        owner._metadata_sha256 = digest("rz-rovent-historical-r-owner-state/1", [registration, publication, owner._observation])
        return owner

    def _check_owner(self):
        self._corpus._check_owner(full=True)
        if digest("rz-rovent-historical-r-owner-state/1", [self._registration, self._publication, self._observation]) != self._metadata_sha256:
            raise ValueError("historical R immutable publication/registration observation changed")
        if (set(self._retirements) != set(self._retirement_observation_pins)
                or digest("rz-rovent-historical-r-retirement-lookup/1", self._retirement_observation_pins)
                    != self._retirement_lookup_sha256):
            raise ValueError("historical R original retirement lookup coverage changed")

    @property
    def corpus(self):
        self._check_owner()
        return self._corpus

    def observation(self):
        self._check_owner()
        return copy.deepcopy(self._observation)

    def original_publication_metadata(self):
        """Original parsed publication and its separately verified raw file pin."""
        self._check_owner()
        return copy.deepcopy({"publication_file": self._registration["publication"],
            "publication": self._publication, "execution_authorized": False})

    def original_retirement_evidence(self, retirement_sha256):
        """Return one original terminal without fabricating an empty input row."""
        self._check_owner()
        identity = sha(retirement_sha256)
        body = self._retirements.get(identity)
        if body is None:
            raise ValueError("original historical R retirement is absent from this owner")
        if digest("rz-rovent-historical-r-retirement-owner/1", body) != self._retirement_observation_pins[identity]:
            raise ValueError("original historical R retirement evidence changed")
        entry = next(item for item in self._publication["retirements"] if item["retirement_sha256"] == identity)
        _retirement_body(body["retirement"], identity, entry["readout_record_sha256"], body["native_admission"])
        return copy.deepcopy({"publication_file_sha256": self._registration["publication"]["sha256"],
            "file": entry["file"], "observation": body, "execution_authorized": False})
