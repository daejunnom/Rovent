"""Explicit raw /3 external Collection root evidence, never a live owner.

The original Rust worker joins its process, profile, clock and RootAdmission
Arc before publishing this /2 root. These pure checks preserve those observed
bytes and the caller-pinned lock; they do not recreate the worker, prove option
application, turn external CP into WDL, or authorize another external query.
"""
from __future__ import annotations

import hashlib
import json
import unicodedata

from .generation_data import digest, fields, sha, uint, _teacher_identity
from .rovent_native_v2 import _artifact, _commit, _root_admission, _root_admission_common

ROOT_DOMAIN = "rz-rovent-collection-root-admission/2"
SOURCE_DOMAIN = "rz-rovent-collection-profiled-external-source/1"
PROFILE_DOMAIN = "rz-rovent-external-engine-profile/1"
PROFILE_CANONICAL_DOMAIN = "rz-rovent-external-engine-profile-canonical/1"
CAPABILITY_DOMAIN = "rz-rovent-external-capabilities/1"
EXTERNAL_SEMANTICS = ("uci-native-cp-or-signed-mate-moves;root-side-to-move;reported-bounds;"
    "selectivity-unknown;no-cp-wdl-calibration;no-rules-proof")
ROOT_FIELDS = ("schema", "native_origin", "native_origin_sha256", "process_epoch", "game_generation", "root_generation",
    "implementation_sha256", "deadline_clock", "deadline_tick", "root_budget_sha256", "selective_policy_sha256",
    "root_state", "store_root_state", "root_moves", "depth_plies", "mate_moves", "requested_nodes",
    "root_descriptor", "root_descriptor_sha256", "external_source", "external_source_sha256")
SOURCE_FIELDS = ("schema", "root_source_sequence", "worker_owner", "clock_epoch", "authority", "engine_profile",
    "purpose", "process", "profile", "profile_file_sha256", "profile_file_bytes", "profile_canonical_sha256",
    "profile_capability_sha256", "binary", "startup_input_lock_sha256", "observed_uci",
    "options_application_observed", "model_loading_observed")
_ENGINES = {
    "stockfish19_embedded_nnue": ("Stockfish 19", "19", "https://github.com/official-stockfish/Stockfish",
        "edb0d9db6731067ec50ce619ff372b463bc4dd5d", "GPL-3.0-or-later", [True, True, True, True, False, False]),
    "reckless090_embedded_nnue": ("Reckless 0.9.0", "0.9.0", "https://github.com/codedeliveryservice/Reckless",
        "0e92358f5acd66e5ac77b1bf558202e47c515435", "AGPL-3.0-only", [False, False, False, True, False, False]),
}


def _text(value, maximum, *, empty=False, tab=False):
    if (type(value) is not str or not empty and not value.strip() or len(value.encode("utf-8")) > maximum
            or any(unicodedata.category(char) == "Cc" and not (tab and char == "\t") for char in value)):
        raise ValueError("bounded external root metadata text required")
    return value


def _linux_path(value, *, root=False):
    _text(value, 4096)
    if value == "/" and root:
        return
    if (not value.startswith("/") or "\\" in value
            or any(part in ("", ".", "..") for part in value.split("/")[1:])):
        raise ValueError("external profile requires a normalized absolute Linux path")


def _positive(value, maximum=2**64 - 1):
    if not uint(value, maximum):
        raise ValueError("positive external root observation required")
    return value


def _option(value, maximum, *, minimum=0):
    # The Rust option builder emits decimal String values, not JSON numbers.
    if type(value) is not str or not value.isascii() or not value.isdigit():
        raise ValueError("external resource option requires its exact decimal string")
    parsed = int(value)
    if not minimum <= parsed <= maximum or str(parsed) != value:
        raise ValueError("external resource option exceeds its registered bounds")
    return parsed


def validate_external_profile(value):
    """Mirror the closed source profile meaning; perform no file/process I/O."""
    fields(value, ("schema", "engine", "purpose", "program", "working_directory", "arguments", "identity",
        "expected_uci_name", "max_depth", "max_prefix_plies", "handshake_timeout_ms", "max_task_wall_time_ms",
        "stop_grace_ms", "shutdown_grace_ms", "max_output_bytes", "max_line_bytes"), "external root profile")
    engine = value["engine"]
    if (value["schema"] != PROFILE_DOMAIN or type(engine) is not str or engine not in _ENGINES
            or value["purpose"] != "research_checker" or value["arguments"] != []
            or type(value["arguments"]) is not list):
        raise ValueError("explicit embedded research-checker profile required")
    _linux_path(value["program"])
    _linux_path(value["working_directory"], root=True)
    _positive(value["max_depth"], 64)
    uint(value["max_prefix_plies"], 4096)
    for key in ("handshake_timeout_ms", "max_task_wall_time_ms", "stop_grace_ms", "shutdown_grace_ms"):
        _positive(value[key], 180000)
    if value["max_task_wall_time_ms"] <= value["stop_grace_ms"]:
        raise ValueError("external task window must retain its original stop reserve")
    _positive(value["max_output_bytes"], 16 * 1024 * 1024)
    _positive(value["max_line_bytes"], 65536)
    if value["max_line_bytes"] > value["max_output_bytes"]:
        raise ValueError("external line cap exceeds original output cap")
    identity = _teacher_identity(value["identity"])
    name, version, repository, commit, license_name, capabilities = _ENGINES[engine]
    empty_args = hashlib.sha256(b"rz-external-uci-arguments/1\0" + bytes(8)).hexdigest()
    if (identity["adapter_semantics"] != EXTERNAL_SEMANTICS or identity["launch_arguments_sha256"] != empty_args
            or identity["declared_name"] != name or value["expected_uci_name"] != name
            or identity["declared_version"] != version or identity["declared_license"] != license_name
            or identity["declared_source"] not in (repository + "/tree/" + commit, repository + "/commit/" + commit)
            or identity["assets"] != [] or identity["model_metadata"] != {
                "weights_sha256": None, "training": "unknown", "declared_rights": None, "precision": None}):
        raise ValueError("external embedded profile source/name/license/assets/unknown model declaration mismatch")
    options = identity["options"]
    overhead = "Move Overhead" if engine == "stockfish19_embedded_nnue" else "MoveOverhead"
    threads = _option(options.get("Threads"), 512, minimum=1)
    hash_mib = _option(options.get("Hash"), 262144, minimum=1)
    _option(options.get("MultiPV"), 4, minimum=1)
    _option(options.get(overhead), 2000)
    expected = {key: options[key] for key in ("Threads", "Hash", "MultiPV", overhead)}
    expected["UCI_Chess960"] = "false"
    if engine == "stockfish19_embedded_nnue":
        expected.update({"Skill Level": "20", "UCI_LimitStrength": "false", "Ponder": "false", "NumaPolicy": "none",
            "SyzygyPath": "", "SyzygyProbeLimit": "0", "nodestime": "0"})
    else:
        expected["Minimal"] = "false"
    if options != expected:
        raise ValueError("external engine-specific required option profile mismatch")
    capability_bytes = json.dumps([CAPABILITY_DOMAIN, engine, value["purpose"], commit, name, EXTERNAL_SEMANTICS,
        capabilities], ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
    return {"source_commit": commit, "requested_threads": threads, "requested_hash_mib": hash_mib,
        "canonical_sha256": digest(PROFILE_CANONICAL_DOMAIN, value),
        "capability_sha256": hashlib.sha256(capability_bytes).hexdigest()}


def validate_external_source(root):
    source = fields(root["external_source"], SOURCE_FIELDS, "external Collection root source")
    if (source["schema"] != SOURCE_DOMAIN or source["purpose"] != "research_checker"
            or digest(SOURCE_DOMAIN, source) != sha(root["external_source_sha256"])):
        raise ValueError("external Collection source domain or original source seal mismatch")
    # Initial worker roots use sequence zero. This is not the P/C question or
    # collector sequence, and a worker owner may also legitimately be zero.
    uint(source["root_source_sequence"])
    uint(source["worker_owner"], 65535)
    if uint(source["clock_epoch"]) != root["deadline_clock"]:
        raise ValueError("external Collection worker clock differs from original root clock")
    authority = fields(source["authority"], ("process_epoch", "game_generation", "root_generation", "implementation_sha256"),
                       "external Collection original authority")
    for key in authority:
        if type(authority[key]) is not type(root[key]) or authority[key] != root[key]:
            raise ValueError("external Collection source crossed original root authority")
    process = fields(source["process"], ("pid", "process_group", "proc_start_ticks"), "external Collection process")
    for key in process:
        _positive(process[key], 2**64 - 1 if key == "proc_start_ticks" else 2**32 - 1)
    registered = validate_external_profile(source["profile"])
    if (source["engine_profile"] != source["profile"]["engine"]
            or sha(source["profile_canonical_sha256"]) != registered["canonical_sha256"]
            or sha(source["profile_capability_sha256"]) != registered["capability_sha256"]
            or sha(source["startup_input_lock_sha256"]) != root["native_origin"]["input_lock_sha256"]):
        raise ValueError("external Collection original profile/capability/startup lock mismatch")
    sha(source["profile_file_sha256"])
    _positive(source["profile_file_bytes"], 65536)
    _artifact(source["binary"])
    if (source["binary"]["sha256"] != source["profile"]["identity"]["binary_sha256"]
            or source["binary"]["license"] != source["profile"]["identity"]["declared_license"]
            or source["binary"]["bytes"] > 256 * 1024 * 1024):
        raise ValueError("external Collection original binary pin differs from declared profile")
    observed = fields(source["observed_uci"], ("name", "author"), "external Collection observed UCI")
    if observed["name"] != source["profile"]["expected_uci_name"]:
        raise ValueError("external Collection actual UCI name differs from original profile")
    if observed["author"] is not None:
        _text(observed["author"], 256, empty=True, tab=True)
    if source["options_application_observed"] is not None or source["model_loading_observed"] is not None:
        raise ValueError("external Collection option/model observations must remain unknown")
    return registered


def validate_external_root(manifest, record, retired):
    from .rovent_corpus_codec import SUBMITTED_CODEC, codec_from_manifest
    from .rovent_submitted_input import INPUT_DOMAIN
    native = record["native"]
    if (codec_from_manifest(manifest) is not SUBMITTED_CODEC or record["input"].get("schema") != INPUT_DOMAIN
            or any(native[key].get("schema") != "rz-rovent-raw-record/3"
                for key in ("raw_submission", "raw_completion", "raw_retirement"))):
        raise ValueError("external Collection root /2 requires explicit submitted input and native raw /3")
    root = fields(retired["root_admission"], ROOT_FIELDS, "external Collection root /2")
    if root["schema"] != ROOT_DOMAIN or digest(ROOT_DOMAIN, root) != sha(retired["root_admission_sha256"]):
        raise ValueError("external Collection root /2 original seal mismatch")
    _root_admission_common(manifest, record, root, descriptor=True)
    if root["native_origin"]["mode"] not in ("HybridPcr", "CpuPcr"):
        raise ValueError("external Collection root /2 requires an original P/C/R mode")
    validate_external_source(root)
    return root


def validate_submitted_root(manifest, record, retired):
    """Select a new explicit raw /3 root without changing either original body."""
    root = retired.get("root_admission")
    if type(root) is dict and root.get("schema") == ROOT_DOMAIN:
        return validate_external_root(manifest, record, retired)
    _root_admission(manifest, record, retired)
    return root


def join_locked_external_source(root, lock):
    """Compare the historical source with the exact original lock declaration."""
    registered = validate_external_source(root)
    researcher = fields(lock["manifest"]["engine"].get("researcher"), ("provider", "profile"), "original external researcher")
    if researcher["provider"] != "external" or lock["canonical_sha256"] != root["native_origin"]["input_lock_sha256"]:
        raise ValueError("external Collection requires its original external researcher lock")
    pin = fields(researcher["profile"], ("domain", "profile", "canonical_sha256", "capability_sha256", "purpose", "binary",
        "source_commit", "requested_threads", "requested_hash_mib"), "original external researcher profile pin")
    _artifact(pin["profile"])
    _artifact(pin["binary"])
    _commit(pin["source_commit"])
    _positive(pin["requested_threads"], 256)
    _positive(pin["requested_hash_mib"], 1048576)
    source = root["external_source"]
    if (pin["domain"] != PROFILE_DOMAIN or pin["purpose"] != "research_checker"
            or pin["profile"]["sha256"] != source["profile_file_sha256"] or pin["profile"]["bytes"] != source["profile_file_bytes"]
            or pin["binary"] != source["binary"] or any(pin[key] != registered[key] for key in
                ("source_commit", "requested_threads", "requested_hash_mib", "canonical_sha256", "capability_sha256"))):
        raise ValueError("external Collection source differs from original lock profile/binary/source/resource pins")
    return source
