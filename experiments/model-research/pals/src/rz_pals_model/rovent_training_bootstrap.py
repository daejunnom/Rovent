"""Standard-library bootstrap for one original Rovent learner child.

This module neither imports a neural library nor issues training authority.
The original Rust parent retains its registration descriptors and corpus
custody.  Here the child checks its exact startup bytes, private source/runtime
pins and its own resource membership before a caller may connect that parent's
live authority socket.  A successful readback is a bounded observation, not a
physical-completion fence or permission reconstructed from JSON.

The functions have no import-time filesystem or process side effects.  In
particular, cgroup migration and ``sys.path`` changes occur only in the explicit
``bootstrap_original_child`` call.  No site hook or ``.pth`` file is executed.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import stat
import sys
import time

STARTUP_DOMAIN = "rz-rovent-original-trainer-startup/1"
REGISTRATION_DOMAIN = "rz-rovent-original-training-registration/1"
RESOURCE_DOMAIN = "rz-rovent-training-resource-startup/1"
SOURCE_DOMAIN = "rz-rovent-trainer-source-bundle/1"
RECOVERED_STARTUP_DOMAIN = "rz-rovent-recovered-training-startup/1"
RECOVERED_REGISTRATION_DOMAIN = "rz-rovent-recovered-training-registration/1"
RECOVERED_SOURCE_DOMAIN = "rz-rovent-recovered-trainer-source-bundle/1"
RECOVERED_WINDOW_DOMAIN = "rz-rovent-train-window/2"
DEPENDENCY_DOMAIN = "rz-rovent-trainer-dependency-set/1"
DEPENDENCY_SCOPE = "declared_pinned_endpoints_only"
MAX_JSON_BYTES = 16 << 20
MAX_SOURCE_BYTES = 4 << 20
MAX_SOURCE_TOTAL_BYTES = 32 << 20
MAX_EXECUTABLE_BYTES = 256 << 20
MAX_RUNTIME_MODULE_BYTES = 2 << 30
MAX_READ_BYTES = (18 << 30) * 260
MAX_WALL_NS = 24 * 60 * 60 * 10**9
MAX_CLEANUP_NS = 30 * 10**9
PACKAGE_RELATIVE = "experiments/model-research/pals/src"
MODULE_RELATIVE = PACKAGE_RELATIVE + "/rz_pals_model/"
_RESOURCE_FIELDS = ("cpu_quota_us", "cpu_period_us", "memory_high_bytes",
                    "memory_max_bytes", "memory_swap_max_bytes", "pids_max")
_POLICY_FIELDS = ("cpu_quota_millicpus", "memory_high_bytes", "memory_max_bytes",
                  "pids_max", "affinity", "device")
_ROLES = ("proposer", "critic", "validator")
_RUNTIME_ENVIRONMENT = ("TMPDIR", "TMP", "TEMP", "HOME", "XDG_CACHE_HOME",
                        "CUDA_CACHE_PATH", "TORCH_HOME", "TORCH_EXTENSIONS_DIR")


class OriginalBootstrapUnavailable(ValueError):
    """The required isolated Linux bootstrap cannot be established."""


def _fields(value, expected, description):
    if type(value) is not dict or set(value) != set(expected):
        raise ValueError(description + " fields mismatch")
    return value


def _uint(value, maximum=2**64 - 1, *, positive=False):
    if type(value) is not int or not (1 if positive else 0) <= value <= maximum:
        raise ValueError("bounded exact unsigned integer required")
    return value


def _sha(value):
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("exact lowercase SHA-256 required")
    return value


def _identifier(value):
    if type(value) is not str or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", value) is None:
        raise ValueError("bounded plain original execution identifier required")
    return value


def _absolute(value):
    if type(value) is not str or not 1 <= len(value.encode("utf-8")) <= 4096 or any(ord(c) < 32 for c in value):
        raise ValueError("bounded nonsecret absolute bootstrap path required")
    # Pure validation also works in source-only Windows fixture tests.  Actual
    # nofollow access below is deliberately a Linux capability, with no weaker
    # platform fallback.
    path = PureWindowsPath(value) if re.match(r"^[A-Za-z]:", value) else PurePosixPath(value)
    if not path.is_absolute() or "\\" in value and not isinstance(path, PureWindowsPath):
        raise ValueError("absolute canonical bootstrap path required")
    if str(path) != value:
        raise ValueError("bootstrap path must use its canonical separator/component spelling")
    raw_parts = re.split(r"[/\\]", value)
    if any(part in (".", "..") for part in raw_parts):
        raise ValueError("bootstrap path traversal forbidden")
    for part in path.parts:
        lower = part.lower()
        if (lower.startswith(".env") or lower.endswith((".key", ".pem"))
                or any(word in lower for word in ("credential", "service-account", "service_account", "serviceaccount",
                    "ssh-key", "sshkey", "api-key", "api_key", "apikey", "private_key", "client_secret", "access_token", "refresh_token"))):
            raise ValueError("secret bootstrap path forbidden")
    return path


def _child_path(root, child, description, *, equal=False):
    root, child = _absolute(root), _absolute(child)
    if type(root) is not type(child) or not child.is_relative_to(root) or not equal and child == root:
        raise ValueError(description + " escapes its explicit owner")
    return child


def _pin(value, maximum=MAX_JSON_BYTES):
    _fields(value, ("path", "sha256", "bytes"), "original file pin")
    _absolute(value["path"])
    _sha(value["sha256"])
    _uint(value["bytes"], maximum, positive=True)
    return value


def _strict_json(raw, *, maximum=MAX_JSON_BYTES):
    if type(raw) is not bytes or not 0 < len(raw) <= maximum:
        raise ValueError("bounded exact original JSON bytes required")
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate original JSON key")
            result[key] = value
        return result
    def bad_constant(_value):
        raise ValueError("nonfinite original JSON number")
    def finite_float(value):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("nonfinite original JSON number")
        return number
    try:
        return json.loads(raw.decode("utf-8", "strict"), object_pairs_hook=pairs,
                          parse_constant=bad_constant, parse_float=finite_float)
    except (UnicodeError, RecursionError) as error:
        raise ValueError("invalid bounded original JSON") from error


def _digest(domain, value):
    raw = json.dumps([domain, value], sort_keys=True, ensure_ascii=False,
                     separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _policy(value):
    _fields(value, _POLICY_FIELDS, "original trainer resource policy")
    _uint(value["cpu_quota_millicpus"], 256000, positive=True)
    _uint(value["memory_high_bytes"], 1 << 50, positive=True)
    _uint(value["memory_max_bytes"], 1 << 50, positive=True)
    _uint(value["pids_max"], 4096, positive=True)
    if value["memory_high_bytes"] > value["memory_max_bytes"]:
        raise ValueError("resource memory.high exceeds memory.max")
    cpus = value["affinity"]
    if type(cpus) is not list or not 1 <= len(cpus) <= 1024:
        raise ValueError("finite explicit CPU affinity required")
    for cpu in cpus:
        _uint(cpu, 1023)
    if len(set(cpus)) != len(cpus):
        raise ValueError("duplicate CPU affinity")
    if type(value["device"]) is not str or not (value["device"] == "cpu" or re.fullmatch(r"cuda:(0|[1-9][0-9]?)", value["device"]) and int(value["device"].split(":")[1]) < 64):
        raise ValueError("explicit cpu or canonical cuda:N device required")
    return value


def _resource_readback(value, policy):
    _fields(value, _RESOURCE_FIELDS, "configured original resource readback")
    for name in _RESOURCE_FIELDS:
        _uint(value[name])
    expected = {"cpu_quota_us": policy["cpu_quota_millicpus"] * 1000, "cpu_period_us": 1000000,
        "memory_high_bytes": policy["memory_high_bytes"], "memory_max_bytes": policy["memory_max_bytes"],
        "memory_swap_max_bytes": 0, "pids_max": policy["pids_max"]}
    if value != expected:
        raise ValueError("configured resources differ from the declared policy")
    return value


def _resources(value, policy):
    _fields(value, ("domain", "migration", "path", "device", "inode", "membership", "policy",
        "prepared_readback", "readback", "available_controllers_snapshot", "subtree_controllers_snapshot", "scope"), "original resource startup")
    if (value["domain"] != RESOURCE_DOMAIN or value["migration"] != "child_self_only_before_torch"
            or value["scope"] != "configured_leaf_readback_not_child_membership_or_lifetime_proof"
            or value["policy"] != policy):
        raise ValueError("original resource domain, migration or policy differs")
    _absolute(value["path"])
    membership = _absolute(value["membership"])
    if not isinstance(membership, PurePosixPath):
        raise ValueError("Linux unified cgroup membership required")
    _uint(value["device"])
    _uint(value["inode"], positive=True)
    for name in ("prepared_readback", "readback"):
        _resource_readback(value[name], policy)
    for name in ("available_controllers_snapshot", "subtree_controllers_snapshot"):
        values = value[name]
        if type(values) is not list or not 1 <= len(values) <= 64:
            raise ValueError("bounded unique resource controller snapshot required")
        if any(type(item) is not str or re.fullmatch(r"[a-z][a-z0-9_]{0,31}", item) is None for item in values):
            raise ValueError("unregistered resource controller name")
        if len(set(values)) != len(values):
            raise ValueError("duplicate resource controller snapshot")
        if not {"cpu", "memory", "pids"} <= set(values):
            raise ValueError("required resource controller is unavailable")
    return value


def _registration(value):
    _fields(value, ("schema", "execution_id", "action", "export_request", "recipe", "membership", "window_spec", "trainer_source_sha256",
        "python_runtime_sha256", "model_semantics_sha256", "current_corpus", "ancestor_corpora", "checkpoint",
        "pc_initial", "nnuec_initial", "rules_profile", "checkpoint_mode", "expected_initial_completed",
        "expected_ancestor_completed", "resources_policy", "require_total_quota", "output_root", "control_root", "checkpoint_directory",
        "receipt_path", "max_rpc_messages", "max_rpc_bytes", "max_read_bytes", "max_output_bytes"), "original Rovent registration projection")
    if value["schema"] != REGISTRATION_DOMAIN:
        raise ValueError("new original Rovent registration domain required")
    _identifier(value["execution_id"])
    for name in ("recipe", "membership", "window_spec", "rules_profile"):
        _pin(value[name])
    for name in ("trainer_source_sha256", "python_runtime_sha256", "model_semantics_sha256"):
        _sha(value[name])
    ancestors = value["ancestor_corpora"]
    if type(ancestors) is not list or len(ancestors) > 256:
        raise ValueError("bounded original corpus ancestry required")
    manifests = set()
    for corpus in (value["current_corpus"], *ancestors):
        _fields(corpus, ("manifest", "index", "root"), "registered original corpus")
        _pin(corpus["manifest"])
        _pin(corpus["index"])
        _absolute(corpus["root"])
        if corpus["manifest"]["path"] == corpus["index"]["path"] or corpus["manifest"]["sha256"] in manifests:
            raise ValueError("aliased/duplicate original corpus pin")
        manifests.add(corpus["manifest"]["sha256"])
    mode = value["checkpoint_mode"]
    if mode == "initialize":
        if value["checkpoint"] is not None or value["pc_initial"] is None or value["nnuec_initial"] is None:
            raise ValueError("initialize requires separate original P/C and NNUEC assets")
    elif mode not in ("resume", "fork") or value["checkpoint"] is None or value["pc_initial"] is not None or value["nnuec_initial"] is not None:
        raise ValueError("resume/fork requires exactly one original tensor checkpoint")
    if type(value["action"]) is not str or value["action"] not in ("train", "export"):
        raise ValueError("explicit original train or export action required")
    if value["action"] == "train":
        if value["export_request"] is not None:
            raise ValueError("train action cannot carry an export request")
    else:
        if mode != "resume" or value["export_request"] is None:
            raise ValueError("export action requires a pinned export request and resume checkpoint")
        _pin(value["export_request"])
        # Pin shape only.  The consumer reads the exact export request, and the
        # original parent retains completed-window/five-state authority.  This
        # validator cannot authorize an export by parsing this projection.
    for name in ("checkpoint", "pc_initial", "nnuec_initial"):
        if value[name] is not None:
            _pin(value[name], 2 << 30)
    for name in ("expected_initial_completed", "expected_ancestor_completed"):
        _fields(value[name], _ROLES, "registered role update counts")
        for number in value[name].values():
            _uint(number)
    _policy(value["resources_policy"])
    if type(value["require_total_quota"]) is not bool:
        raise ValueError("original total-quota requirement must be an exact boolean")
    # The original parent separately observes/adopts any enclosing total quota.
    # This projection does not mint that authority from the requested flag, and
    # the child leaf readback must never be relabelled as the enclosing quota.
    output, control = _absolute(value["output_root"]), _absolute(value["control_root"])
    if type(output) is not type(control) or output.is_relative_to(control) or control.is_relative_to(output):
        raise ValueError("output and private control roots must have separate owners")
    _child_path(value["output_root"], value["checkpoint_directory"], "checkpoint directory")
    receipt = _child_path(value["output_root"], value["receipt_path"], "terminal receipt")
    if receipt.parent != output or receipt.suffix != ".json":
        raise ValueError("terminal receipt must be a direct JSON child of the output root")
    _uint(value["max_rpc_messages"], 1000000, positive=True)
    _uint(value["max_rpc_bytes"], 1 << 30, positive=True)
    _uint(value["max_read_bytes"], MAX_READ_BYTES, positive=True)
    _uint(value["max_output_bytes"], 18 << 30, positive=True)
    if value["max_output_bytes"] <= MAX_JSON_BYTES:
        raise ValueError("separate terminal/output byte credit required")
    return value


def validate_startup(raw, *, expected_sha256, expected_bytes, execution_until_ns, whole_until_ns, now_ns):
    """Validate an exact startup projection without touching files or resources.

    ``registration_sha256`` identifies the original parent's raw registration
    descriptor.  Those bytes are not repeated here; this function intentionally
    does not hash a reserialized projection and call that a raw-byte check.
    """
    _sha(expected_sha256)
    _uint(expected_bytes, MAX_JSON_BYTES, positive=True)
    if type(raw) is not bytes or len(raw) != expected_bytes or hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("original startup exact bytes or SHA-256 differ")
    value = _strict_json(raw)
    _fields(value, ("schema", "execution_id", "window_id", "registration_sha256", "registration", "control",
        "original_started_ns", "execution_until_ns", "whole_until_ns", "source_root", "package_root", "source_files",
        "dependency_directories", "runtime_module_pins", "python_runtime", "resources"), "original Rovent startup")
    if value["schema"] != STARTUP_DOMAIN:
        raise ValueError("new original Rovent startup domain required")
    _identifier(value["execution_id"])
    _identifier(value["window_id"])
    _sha(value["registration_sha256"])
    registration = _registration(value["registration"])
    if value["execution_id"] != registration["execution_id"]:
        raise ValueError("startup execution differs from original registration")
    for number in (value["original_started_ns"], value["execution_until_ns"], value["whole_until_ns"], execution_until_ns, whole_until_ns, now_ns):
        _uint(number)
    if (value["execution_until_ns"] != execution_until_ns or value["whole_until_ns"] != whole_until_ns
            or not value["original_started_ns"] <= now_ns < execution_until_ns < whole_until_ns
            or execution_until_ns - value["original_started_ns"] > MAX_WALL_NS
            or whole_until_ns - execution_until_ns > MAX_CLEANUP_NS):
        raise ValueError("original startup deadline mismatch, renewal or expiration")
    control = _fields(value["control"], ("socket_path", "parent_pid", "parent_start_ticks", "control_device", "control_inode"), "original control startup")
    endpoint = _child_path(registration["control_root"], control["socket_path"], "control socket")
    if endpoint.parent != _absolute(registration["control_root"]) or len(control["socket_path"].encode()) >= 104:
        raise ValueError("bounded direct original control socket required")
    _uint(control["parent_pid"], 2**31 - 1, positive=True)
    _uint(control["parent_start_ticks"], positive=True)
    _uint(control["control_device"])
    _uint(control["control_inode"], positive=True)
    source_root = _child_path(registration["output_root"], value["source_root"], "private source root")
    if source_root != _absolute(registration["output_root"]) / "source" or _absolute(value["package_root"]) != source_root / PACKAGE_RELATIVE:
        raise ValueError("private source/package layout differs from the original installer")
    sources = value["source_files"]
    if type(sources) is not list or not 1 <= len(sources) <= 128:
        raise ValueError("bounded nonempty original source inventory required")
    inventory, source_bytes = {}, 0
    for item in sources:
        _fields(item, ("relative_path", "sha256", "bytes"), "original source inventory entry")
        relative = item["relative_path"]
        if (type(relative) is not str or not relative.startswith(MODULE_RELATIVE) or not relative.endswith(".py")
                or "\\" in relative or any(part in ("", ".", "..") for part in relative.split("/"))
                or relative in inventory):
            raise ValueError("source entry escaped its plain private Python package")
        _absolute(str(source_root / relative))
        _sha(item["sha256"])
        source_bytes += _uint(item["bytes"], MAX_SOURCE_BYTES, positive=True)
        inventory[relative] = {"bytes": item["bytes"], "sha256": item["sha256"]}
    if source_bytes > MAX_SOURCE_TOTAL_BYTES or _digest(SOURCE_DOMAIN, inventory) != registration["trainer_source_sha256"]:
        raise ValueError("compiled original source inventory digest or total differs")
    required = {MODULE_RELATIVE + name for name in ("__init__.py", "rovent_training_child.py", "rovent_training_bootstrap.py")}
    if not required <= inventory.keys():
        raise ValueError("original entry/bootstrap/package pins are absent")
    dependencies = value["dependency_directories"]
    if type(dependencies) is not list or not 1 <= len(dependencies) <= 8:
        raise ValueError("finite registered dependency directories required")
    paths, identities = set(), set()
    for dependency in dependencies:
        _fields(dependency, ("path", "device", "inode"), "original dependency directory")
        path = _absolute(dependency["path"])
        identity = (_uint(dependency["device"]), _uint(dependency["inode"], positive=True))
        if str(path) in paths or identity in identities or path.is_relative_to(source_root) or source_root.is_relative_to(path):
            raise ValueError("duplicate/aliased source or dependency directory")
        paths.add(str(path))
        identities.add(identity)
    runtime = _fields(value["python_runtime"], ("version", "executable"), "original Python runtime")
    if type(runtime["version"]) is not str or re.fullmatch(r"3\.[0-9]{1,2}\.[0-9]{1,3}", runtime["version"]) is None:
        raise ValueError("explicit original Python release required")
    _pin(runtime["executable"], MAX_EXECUTABLE_BYTES)
    modules = value["runtime_module_pins"]
    if type(modules) is not list or not 1 <= len(modules) <= 64:
        raise ValueError("finite original runtime module pins required")
    module_paths = set()
    for module in modules:
        _pin(module, MAX_RUNTIME_MODULE_BYTES)
        if module["path"] in module_paths:
            raise ValueError("duplicate original runtime module pin")
        module_paths.add(module["path"])
    required_read = source_bytes + runtime["executable"]["bytes"] + sum(module["bytes"] for module in modules)
    if required_read > registration["max_read_bytes"]:
        raise ValueError("bootstrap pins exceed original aggregate read credit")
    _resources(value["resources"], registration["resources_policy"])
    return copy.deepcopy(value)


def _historical_dataset_request(value):
    """Strict caller request, never a reconstructed native collection owner."""
    _fields(value, ("root", "publication", "collection_source_commit", "collection_source_sha256",
        "collection_runtime_library_sha256", "model_semantics_sha256", "model_epoch", "context_epoch",
        "recipe_sha256", "stage_id", "role_bindings"), "historical dataset registration")
    root = _absolute(value["root"])
    _pin(value["publication"])
    if _absolute(value["publication"]["path"]) != root / "publication.json":
        raise ValueError("historical dataset must pin the registered publication.json")
    if type(value["collection_source_commit"]) is not str or re.fullmatch(r"[0-9a-f]{40}", value["collection_source_commit"]) is None:
        raise ValueError("full historical collection commit required")
    for name in ("collection_source_sha256", "collection_runtime_library_sha256", "model_semantics_sha256", "recipe_sha256"):
        _sha(value[name])
    _identifier(value["stage_id"])
    _uint(value["model_epoch"])
    _uint(value["context_epoch"])
    roles = value["role_bindings"]
    if type(roles) is not dict or not 1 <= len(roles) <= 3 or not set(roles) <= set(_ROLES):
        raise ValueError("explicit bounded historical role bindings required")
    for binding in roles.values():
        _fields(binding, ("model_semantics_sha256", "model_identity_sha256", "weights_sha256", "encoding_sha256"), "historical role binding")
        for identity in binding.values():
            _sha(identity)
    return value


def _historical_checkpoint_request(value):
    _fields(value, ("root", "publication", "reclamation"), "historical checkpoint registration")
    _absolute(value["root"])
    _pin(value["publication"])
    _pin(value["reclamation"])
    _child_path(value["root"], value["publication"]["path"], "historical checkpoint publication")
    # The original closure file is an input for the new OS owner.  Neither its
    # pin nor this projection grants numeric PID/signal/reap/GPU authority.
    return value


def recovered_dependency_set_digest(endpoints):
    """Location-independent sorted endpoint digest, independent of Python/ORT."""
    if type(endpoints) is not list or not 2 <= len(endpoints) <= 64:
        raise ValueError("finite explicit recovered runtime endpoint inventory required")
    names, inventory = set(), []
    for endpoint in endpoints:
        _fields(endpoint, ("name", "artifact"), "recovered trainer endpoint")
        name = endpoint["name"]
        if type(name) is not str or not 1 <= len(name.encode()) <= 256 or any(ord(char) < 32 for char in name) or name in names:
            raise ValueError("bounded unique logical recovered endpoint name required")
        names.add(name)
        pin = _pin(endpoint["artifact"], MAX_RUNTIME_MODULE_BYTES)
        inventory.append({"name": name, "sha256": pin["sha256"], "bytes": pin["bytes"]})
    if not {"torch.__file__", "torch._C.__file__"} <= names:
        raise ValueError("explicit Torch Python and native endpoint pins required")
    return _digest(DEPENDENCY_DOMAIN, sorted(inventory, key=lambda item: item["name"]))


def _r_original_artifact(value):
    """Original declared NNUEC artifact metadata, never an archive lookup path."""
    _fields(value, ("path", "sha256", "bytes", "source", "license"), "original R NNUEC artifact reference")
    for name in ("path", "source", "license"):
        text = value[name]
        if type(text) is not str or not 1 <= len(text.encode()) <= 4096 or any(ord(char) < 32 or 127 <= ord(char) <= 159 for char in text):
            raise ValueError("bounded original R artifact metadata required")
    _sha(value["sha256"])
    _uint(value["bytes"], positive=True)
    # This raw path is preserved as the original declaration only.  Its bytes
    # are opened through separately pinned publication-relative archive files,
    # never by joining or trusting this field as a filesystem authority.


def _r_auxiliary_dataset_request(value):
    """Strict optional R source request, kept separate from every P/C row."""
    _fields(value, ("root", "publication", "collection_source_commit", "collection_source_sha256",
        "collection_runtime_library_sha256", "collection_model_semantics_sha256", "collection_model_epoch",
        "collection_context_epoch", "collection_recipe_sha256", "collection_stage_id", "nnuec",
        "nnuec_parameter_sha256"), "separate recovered R auxiliary dataset")
    root = _absolute(value["root"])
    _pin(value["publication"])
    if _child_path(str(root), value["publication"]["path"], "R auxiliary publication") != root / "r-publication.json":
        raise ValueError("R auxiliary publication must pin its explicit root/r-publication.json")
    if type(value["collection_source_commit"]) is not str or re.fullmatch(r"[0-9a-f]{40}", value["collection_source_commit"]) is None:
        raise ValueError("full original R collection commit required")
    for name in ("collection_source_sha256", "collection_runtime_library_sha256", "collection_model_semantics_sha256",
            "collection_recipe_sha256", "nnuec_parameter_sha256"):
        _sha(value[name])
    for name in ("collection_model_epoch", "collection_context_epoch"):
        _uint(value[name], positive=True)
    _identifier(value["collection_stage_id"])
    nnuec = _fields(value["nnuec"], ("schema", "semantics", "encoding", "implementation_sha256", "configuration_sha256",
        "weights", "provenance", "diagnostic", "context_namespace_policy_sha256", "local_namespace_policy_sha256",
        "search_conditions_sha256"), "original R typed NNUEC pin")
    for name in ("schema", "semantics", "encoding"):
        text = nnuec[name]
        if type(text) is not str or not 1 <= len(text.encode()) <= 256 or any(ord(char) < 32 for char in text):
            raise ValueError("bounded original R NNUEC schema/semantics/encoding required")
    if (nnuec["schema"] != "rz-nnuec-fp32-v2"
            or nnuec["semantics"] != "board800-exactb192-causal-line-kind4-directedpair-originfirst-exactc256-fp32-v2"
            or nnuec["encoding"] != "rz-nnuec-fp32-input/2"):
        raise ValueError("historical R auxiliary source requires its explicit registered NNUEC V2 codec")
    for name in ("implementation_sha256", "configuration_sha256", "context_namespace_policy_sha256",
            "local_namespace_policy_sha256", "search_conditions_sha256"):
        _sha(nnuec[name])
    if type(nnuec["diagnostic"]) is not bool:
        raise ValueError("exact original R NNUEC diagnostic flag required")
    for name in ("weights", "provenance"):
        _r_original_artifact(nnuec[name])
    if nnuec["weights"]["bytes"] > 32 << 20 or nnuec["provenance"]["bytes"] > 256 << 10:
        raise ValueError("historical R original weights/provenance exceed declared source bounds")
    return value


def r_auxiliary_dataset_presence(registration):
    """Preserve absent/null/enabled wire state without introducing defaults."""
    if type(registration) is not dict:
        raise ValueError("R auxiliary presence requires its registration object")
    if "r_auxiliary_dataset" not in registration:
        return "absent"
    return "null" if registration["r_auxiliary_dataset"] is None else "enabled"


def _r_auxiliary_ancestor_requests(registration):
    if "ancestor_r_auxiliary_datasets" not in registration or registration["ancestor_r_auxiliary_datasets"] is None:
        return ()
    ancestors = registration["ancestor_r_auxiliary_datasets"]
    if type(ancestors) is not list or len(ancestors) > 256:
        raise ValueError("bounded explicitly present R auxiliary ancestors required")
    if r_auxiliary_dataset_presence(registration) != "enabled":
        raise ValueError("R auxiliary ancestor list requires an enabled current R dataset")
    for source in ancestors:
        _r_auxiliary_dataset_request(source)
    identities = [source["publication"]["sha256"] for source in ancestors]
    if identities != sorted(set(identities)) or registration["r_auxiliary_dataset"]["publication"]["sha256"] in identities:
        raise ValueError("R auxiliary ancestors must be sorted unique publications distinct from current")
    return ancestors


def validate_recovered_registration(value):
    """Validate the distinct recovered DTO without converting an original /1."""
    _fields(value, ("schema", "execution_id", "action", "export_request", "recipe", "membership", "window_spec",
        "trainer_source_sha256", "trainer_python_version", "trainer_python_executable", "dependency_directories", "trainer_dependency_set_sha256",
        "runtime_endpoints", "current_dataset", "ancestor_datasets", "initialization_source", "rules_profile",
        "expected_initial_completed", "expected_ancestor_completed", "resources_policy", "require_total_quota",
        "output_root", "control_root", "checkpoint_directory", "receipt_path", "max_rpc_messages", "max_rpc_bytes",
        "max_read_bytes", "max_child_read_bytes", "max_output_bytes", "max_artifact_files")
        + (("r_auxiliary_dataset",) if type(value) is dict and "r_auxiliary_dataset" in value else ())
        + (("ancestor_r_auxiliary_datasets",) if type(value) is dict and "ancestor_r_auxiliary_datasets" in value else ()),
        # Both optional fields retain their original absent/null/list/object
        # presence; no disabled run receives a re-signed default value.
        "recovered Rovent training registration")
    if value["schema"] != RECOVERED_REGISTRATION_DOMAIN:
        raise ValueError("separate recovered registration domain required; original conversion forbidden")
    _identifier(value["execution_id"])
    for name in ("recipe", "membership", "window_spec", "rules_profile"):
        _pin(value[name])
    _sha(value["trainer_source_sha256"])
    _pin(value["trainer_python_executable"], MAX_RUNTIME_MODULE_BYTES)
    if (type(value["trainer_python_version"]) is not str
            or re.fullmatch(r"3\.[0-9]{1,2}\.[0-9]{1,3}", value["trainer_python_version"]) is None
            or any(str(int(part)) != part for part in value["trainer_python_version"].split("."))):
        raise ValueError("explicit recovered Python version required")
    directories = value["dependency_directories"]
    if type(directories) is not list or not 1 <= len(directories) <= 8:
        raise ValueError("one to eight explicitly registered recovered dependency directories required")
    registered_paths = [str(_absolute(path)) for path in directories]
    if len(set(registered_paths)) != len(registered_paths):
        raise ValueError("duplicate recovered registered dependency directory")
    if _sha(value["trainer_dependency_set_sha256"]) != recovered_dependency_set_digest(value["runtime_endpoints"]):
        raise ValueError("recovered declared dependency endpoint digest differs")
    if r_auxiliary_dataset_presence(value) == "enabled":
        _r_auxiliary_dataset_request(value["r_auxiliary_dataset"])
    _r_auxiliary_ancestor_requests(value)
    _historical_dataset_request(value["current_dataset"])
    ancestors = value["ancestor_datasets"]
    if type(ancestors) is not list or len(ancestors) > 256:
        raise ValueError("bounded recovered historical dataset ancestry required")
    seen = {value["current_dataset"]["publication"]["sha256"]}
    for ancestor in ancestors:
        _historical_dataset_request(ancestor)
        identity = ancestor["publication"]["sha256"]
        if identity in seen:
            raise ValueError("duplicate/current historical dataset publication in ancestors")
        seen.add(identity)
    for name in ("expected_initial_completed", "expected_ancestor_completed"):
        _fields(value[name], _ROLES, "recovered exact role counts")
        for count in value[name].values():
            _uint(count, 65536)
    action = value["action"]
    if type(action) is not str or action not in ("train", "export", "ipc_check"):
        raise ValueError("explicit recovered train, export or ipc_check action required")
    source = value["initialization_source"]
    if type(source) is not dict:
        raise ValueError("tagged recovered initialization_source required")
    kind = source.get("kind")
    if kind == "initialize":
        _fields(source, ("kind", "pc_checkpoint", "nnuec_checkpoint"), "recovered initialize source")
        _pin(source["pc_checkpoint"], MAX_RUNTIME_MODULE_BYTES)
        _pin(source["nnuec_checkpoint"], MAX_RUNTIME_MODULE_BYTES)
        if action not in ("train", "ipc_check") or any(value["expected_initial_completed"].values()):
            raise ValueError("recovered initialize cannot export or claim existing stage updates")
    elif kind == "resume":
        _fields(source, ("kind", "checkpoint"), "recovered resume source")
        _historical_checkpoint_request(source["checkpoint"])
    elif kind == "fork":
        _fields(source, ("kind", "checkpoint", "parent_stage_id"), "recovered fork source")
        _historical_checkpoint_request(source["checkpoint"])
        _identifier(source["parent_stage_id"])
        if action not in ("train", "ipc_check") or any(value["expected_initial_completed"].values()):
            raise ValueError("recovered fork requires fresh current-stage counts and train action")
    else:
        raise ValueError("unregistered recovered initialization source")
    if action in ("train", "ipc_check"):
        if value["export_request"] is not None:
            raise ValueError("recovered train or ipc_check cannot carry an export request")
    else:
        _pin(value["export_request"])
        if kind != "resume":
            raise ValueError("recovered export requires an explicitly resumed checkpoint")
    _policy(value["resources_policy"])
    if type(value["require_total_quota"]) is not bool:
        raise ValueError("recovered total-quota requirement must be an exact boolean")
    output, control = _absolute(value["output_root"]), _absolute(value["control_root"])
    if type(output) is not type(control) or output.is_relative_to(control) or control.is_relative_to(output):
        raise ValueError("recovered output/control owners overlap")
    _child_path(value["output_root"], value["checkpoint_directory"], "recovered checkpoint directory")
    _child_path(value["output_root"], value["receipt_path"], "recovered terminal receipt")
    _uint(value["max_rpc_messages"], 2000000, positive=True)
    _uint(value["max_rpc_bytes"], 18 << 30, positive=True)
    _uint(value["max_read_bytes"], MAX_READ_BYTES, positive=True)
    _uint(value["max_child_read_bytes"], value["max_read_bytes"], positive=True)
    _uint(value["max_output_bytes"], 18 << 30, positive=True)
    _uint(value["max_artifact_files"], 32768, positive=True)
    return copy.deepcopy(value)


def validate_recovered_window_binding(startup, window):
    """Join a separately validated /2 window; old field aliases are rejected.

    Full recipe/control validation belongs to ``rovent_training_window_v2``.
    This stdlib-only projection compares collection identities to historical
    data and trainer identities to this new process's declared files.
    """
    registration = validate_recovered_registration(startup["registration"])
    if type(window) is not dict or window.get("schema") != RECOVERED_WINDOW_DOMAIN:
        raise ValueError("new recovered child admits only explicit train-window /2")
    dataset = registration["current_dataset"]
    expected = {"window_id": startup["window_id"], "trainer_source_sha256": registration["trainer_source_sha256"],
        "trainer_python_executable_sha256": registration["trainer_python_executable"]["sha256"],
        "trainer_dependency_set_sha256": registration["trainer_dependency_set_sha256"], "trainer_dependency_scope": DEPENDENCY_SCOPE,
        "collection_source_sha256": dataset["collection_source_sha256"],
        "collection_runtime_library_sha256": dataset["collection_runtime_library_sha256"],
        "collection_model_semantics_sha256": dataset["model_semantics_sha256"],
        "collection_model_epoch": dataset["model_epoch"], "collection_context_epoch": dataset["context_epoch"],
        "recipe_sha256": dataset["recipe_sha256"], "stage_id": dataset["stage_id"],
        "device": registration["resources_policy"]["device"]}
    if any(name not in window or type(window[name]) is not type(expected_value) or window[name] != expected_value
            for name, expected_value in expected.items()):
        raise ValueError("recovered window changed collection/trainer/resource identity namespace")
    if (type(window.get("max_output_bytes")) is not int or not 0 < window["max_output_bytes"] <= registration["max_output_bytes"]):
        raise ValueError("recovered window exceeds its original output credit")
    if window.get("training_model_semantics_sha256") != dataset["model_semantics_sha256"]:
        raise ValueError("recovered architecture compatibility needs a separately registered policy")
    current_presence = r_auxiliary_dataset_presence(registration)
    window_presence = "absent" if "nnuec_r_auxiliary" not in window else "null" if window["nnuec_r_auxiliary"] is None else "enabled"
    if current_presence != window_presence:
        raise ValueError("registered R auxiliary dataset and /2 recipe must preserve exact presence")
    if current_presence == "enabled":
        auxiliary = window["nnuec_r_auxiliary"]
        _fields(auxiliary, ("schema", "manifest_file_sha256", "index_file_sha256", "sampling_seed", "rows_per_pc_update",
            "weight", "normalization", "optimizer_owner", "supervision_source"), "separate R auxiliary /2 recipe")
        if auxiliary["schema"] != "rz-rovent-r-auxiliary-recipe/1":
            raise ValueError("R auxiliary recipe requires its own explicit source domain")
        for name in ("manifest_file_sha256", "index_file_sha256"):
            _sha(auxiliary[name])
        admitted = startup.get("r_auxiliary_admission")
        if (type(admitted) is not dict or auxiliary["manifest_file_sha256"] != admitted.get("manifest_file_sha256")
                or auxiliary["index_file_sha256"] != admitted.get("index_file_sha256")):
            raise ValueError("R auxiliary recipe raw file pins differ from the separately admitted source")
        if registration["r_auxiliary_dataset"]["nnuec"]["semantics"] != window.get("nnuec_semantics"):
            raise ValueError("R auxiliary NNUEC semantics differ from the current learner declaration")
    return {"schema": "rz-rovent-recovered-window-binding/1", "window_schema": RECOVERED_WINDOW_DOMAIN,
        "trainer_dependency_scope": DEPENDENCY_SCOPE, "transitive_loaded_native_mappings": "unknown",
        "past_native_authority_reissued": False, "past_physical_completion_reissued": False, "execution_authorized": False}


def _publication_metadata_file(value):
    _fields(value, ("path", "sha256", "bytes"), "historical relative publication file")
    path = value["path"]
    if (type(path) is not str or not 1 <= len(path.encode()) <= 4096 or "\\" in path
            or PurePosixPath(path).is_absolute() or any(part in ("", ".", "..") for part in path.split("/"))):
        raise ValueError("plain bounded historical publication-relative path required")
    _absolute("/" + path)
    _sha(value["sha256"])
    _uint(value["bytes"], MAX_JSON_BYTES, positive=True)


def _checkpoint_metadata_admission(value, initialization):
    if initialization["kind"] == "initialize":
        if value is not None:
            raise ValueError("new initialization cannot borrow a historical checkpoint admission")
        return
    _fields(value, ("schema", "publication_sha256", "tensor_sha256", "update_chain_sha256", "snapshot_completed_updates",
        "post_snapshot_optimizer_updates", "recorded_tail_cost", "recorded_closure_classification",
        "historical_window_schema", "historical_window_identity_scope", "historical_metadata_integrity_only", "tensor_state_validated",
        "past_native_authority_reissued", "past_physical_completion_reissued", "execution_authorized"), "historical checkpoint metadata admission")
    if (value["schema"] != "rz-rovent-historical-checkpoint-metadata-admission/1"
            or value["historical_metadata_integrity_only"] is not True or value["tensor_state_validated"] is not False
            or any(value[name] is not False for name in ("past_native_authority_reissued", "past_physical_completion_reissued", "execution_authorized"))):
        raise ValueError("historical checkpoint metadata cannot reissue tensor/native/physical/execution authority")
    for name in ("publication_sha256", "tensor_sha256", "update_chain_sha256"):
        _sha(value[name])
    if value["publication_sha256"] != initialization["checkpoint"]["publication"]["sha256"]:
        raise ValueError("historical checkpoint admission changed its registered publication")
    for name in ("snapshot_completed_updates", "post_snapshot_optimizer_updates"):
        _fields(value[name], _ROLES, "historical checkpoint role counts")
        for count in value[name].values():
            _uint(count, 65536)
    schema_scopes = {"rz-rovent-train-window/1": "original_v1_unseparated",
        "rz-rovent-train-window/2": "explicit_v2_recorded_pins"}
    if (value["recorded_closure_classification"] not in ("recorded_closed", "unresolved")
            or type(value["historical_window_schema"]) is not str or value["historical_window_schema"] not in schema_scopes
            or value["historical_window_identity_scope"] != schema_scopes[value["historical_window_schema"]]):
        raise ValueError("explicit historical closure classification and identity scope required")
    tail = value["recorded_tail_cost"]
    if type(tail) is not dict:
        raise ValueError("explicit historical known/unknown tail-cost record required")
    if tail.get("status") == "known":
        _fields(tail, ("status", "delta", "evidence"), "known historical tail cost")
        if type(tail["delta"]) is not dict or not 1 <= len(tail["delta"]) <= 64:
            raise ValueError("finite historical tail-cost counters required")
        for name, amount in tail["delta"].items():
            _identifier(name)
            _uint(amount)
        _publication_metadata_file(tail["evidence"])
    elif tail.get("status") == "unknown":
        _fields(tail, ("status", "reason", "evidence"), "unknown historical tail cost")
        if type(tail["reason"]) is not str or not 1 <= len(tail["reason"].encode()) <= 4096 or any(ord(char) < 32 for char in tail["reason"]):
            raise ValueError("bounded explicit historical unknown-cost reason required")
        if tail["evidence"] is not None:
            _publication_metadata_file(tail["evidence"])
    else:
        raise ValueError("unregistered historical tail-cost status")


def validate_recovered_startup(raw, *, expected_sha256, expected_bytes, execution_until_ns, whole_until_ns, now_ns):
    """Strict recovered startup; it never converts an original /1 envelope."""
    _sha(expected_sha256)
    _uint(expected_bytes, MAX_JSON_BYTES, positive=True)
    if type(raw) is not bytes or len(raw) != expected_bytes or hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("recovered startup exact bytes or SHA-256 differ")
    value = _strict_json(raw)
    _fields(value, ("schema", "execution_id", "window_id", "registration_sha256", "registration", "control",
        "original_started_ns", "execution_until_ns", "whole_until_ns", "source_root", "package_root", "source_files",
        "dependency_directories", "runtime_module_pins", "python_runtime", "resources", "collection_admission",
        "checkpoint_admission", "child_read_credit", "artifact_watch_profile")
        + (("r_auxiliary_admission",) if type(value) is dict and "r_auxiliary_admission" in value else ()),
        "recovered Rovent startup")
    if value["schema"] != RECOVERED_STARTUP_DOMAIN:
        raise ValueError("separate recovered startup domain required; original conversion forbidden")
    _identifier(value["execution_id"])
    _identifier(value["window_id"])
    _sha(value["registration_sha256"])
    registration = validate_recovered_registration(value["registration"])
    if value["execution_id"] != registration["execution_id"]:
        raise ValueError("recovered execution differs from original registered request")
    for number in (value["original_started_ns"], value["execution_until_ns"], value["whole_until_ns"], execution_until_ns, whole_until_ns, now_ns):
        _uint(number)
    if (value["execution_until_ns"] != execution_until_ns or value["whole_until_ns"] != whole_until_ns
            or not value["original_started_ns"] <= now_ns < execution_until_ns < whole_until_ns
            or execution_until_ns - value["original_started_ns"] > MAX_WALL_NS
            or whole_until_ns - execution_until_ns > MAX_CLEANUP_NS):
        raise ValueError("recovered startup deadline mismatch, renewal or expiration")
    control = _fields(value["control"], ("socket_path", "parent_pid", "parent_start_ticks", "control_device", "control_inode"), "recovered current control startup")
    endpoint = _child_path(registration["control_root"], control["socket_path"], "recovered current control socket")
    if endpoint.parent != _absolute(registration["control_root"]) or len(control["socket_path"].encode()) >= 104:
        raise ValueError("bounded direct recovered control socket required")
    _uint(control["parent_pid"], 2**31 - 1, positive=True)
    _uint(control["parent_start_ticks"], positive=True)
    _uint(control["control_device"])
    _uint(control["control_inode"], positive=True)
    source_root = _child_path(registration["output_root"], value["source_root"], "recovered private source root")
    if source_root != _absolute(registration["output_root"]) / "source" or _absolute(value["package_root"]) != source_root / PACKAGE_RELATIVE:
        raise ValueError("recovered source/package layout differs from its original installer")
    sources = value["source_files"]
    if type(sources) is not list or not 1 <= len(sources) <= 128:
        raise ValueError("bounded recovered source inventory required")
    inventory, source_bytes = {}, 0
    for item in sources:
        _fields(item, ("relative_path", "sha256", "bytes"), "recovered source inventory entry")
        relative = item["relative_path"]
        if (type(relative) is not str or not relative.startswith(MODULE_RELATIVE) or not relative.endswith(".py")
                or "\\" in relative or any(part in ("", ".", "..") for part in relative.split("/")) or relative in inventory):
            raise ValueError("recovered source escaped its private Python package")
        _absolute(str(source_root / relative))
        _sha(item["sha256"])
        source_bytes += _uint(item["bytes"], MAX_SOURCE_BYTES, positive=True)
        inventory[relative] = {"bytes": item["bytes"], "sha256": item["sha256"]}
    if source_bytes > MAX_SOURCE_TOTAL_BYTES or _digest(RECOVERED_SOURCE_DOMAIN, inventory) != registration["trainer_source_sha256"]:
        raise ValueError("recovered source inventory must use its separately registered digest domain")
    required = {MODULE_RELATIVE + name for name in ("__init__.py", "rovent_recovered_training_child.py", "rovent_training_bootstrap.py")}
    if not required <= inventory.keys():
        raise ValueError("recovered entry/bootstrap/package pins are absent")
    dependencies = value["dependency_directories"]
    if type(dependencies) is not list or not 1 <= len(dependencies) <= 8:
        raise ValueError("finite registered recovered dependency directories required")
    paths, identities = set(), set()
    for dependency in dependencies:
        _fields(dependency, ("path", "device", "inode"), "recovered dependency directory")
        path = _absolute(dependency["path"])
        identity = (_uint(dependency["device"]), _uint(dependency["inode"], positive=True))
        if str(path) in paths or identity in identities or path.is_relative_to(source_root) or source_root.is_relative_to(path):
            raise ValueError("duplicate/aliased recovered dependency directory")
        paths.add(str(path))
        identities.add(identity)
    if [dependency["path"] for dependency in dependencies] != registration["dependency_directories"]:
        raise ValueError("recovered startup dependency identity roster differs from the registered paths or order")
    runtime = _fields(value["python_runtime"], ("version", "executable"), "recovered actual Python runtime")
    if runtime["version"] != registration["trainer_python_version"] or runtime["executable"] != registration["trainer_python_executable"]:
        raise ValueError("recovered runtime must use the registered trainer Python image, not collection ORT")
    _pin(runtime["executable"], MAX_RUNTIME_MODULE_BYTES)
    modules = value["runtime_module_pins"]
    expected_modules = [endpoint["artifact"] for endpoint in registration["runtime_endpoints"]]
    if modules != expected_modules:
        raise ValueError("recovered runtime module pins changed the registered endpoint order or bytes")
    for module in modules:
        _pin(module, MAX_RUNTIME_MODULE_BYTES)
    required_read = source_bytes + runtime["executable"]["bytes"] + sum(module["bytes"] for module in modules)
    credit = _fields(value["child_read_credit"], ("reserved_bytes", "scope"), "current child read reservation")
    if (type(credit["reserved_bytes"]) is not int or credit["reserved_bytes"] != registration["max_child_read_bytes"]
            or credit["scope"] != "charged_in_parent_shared_budget_no_refund"):
        raise ValueError("child read credit differs from the original parent's whole reservation")
    if required_read >= credit["reserved_bytes"]:
        raise ValueError("recovered bootstrap pins exceed original aggregate read credit")
    profile = _fields(value["artifact_watch_profile"], ("schema", "maximum_total_bytes", "maximum_file_bytes",
        "maximum_files", "scope"), "new trainer owned-output artifact watch")
    for name in ("maximum_total_bytes", "maximum_file_bytes", "maximum_files"):
        _uint(profile[name], positive=True)
    control_reserve = min(registration["max_output_bytes"] // 8, 64 << 20)
    if profile != {"schema": "rz-rovent-training-artifact-watch/1",
            "maximum_total_bytes": registration["max_output_bytes"] - control_reserve,
            "maximum_file_bytes": 2 << 30, "maximum_files": registration["max_artifact_files"],
            "scope": "new_trainer_owned_output_tree_only"}:
        raise ValueError("registered new trainer artifact-watch profile differs")
    _policy(value["resources"]["policy"])
    _resources(value["resources"], registration["resources_policy"])
    admission = _fields(value["collection_admission"], ("schema", "current_publication_sha256", "ancestor_publication_sha256",
        "past_native_authority_reissued", "past_physical_completion_reissued", "historical_evidence_only", "execution_authorized"), "historical collection-set admission")
    expected_ancestors = sorted(dataset["publication"]["sha256"] for dataset in registration["ancestor_datasets"])
    if (admission["schema"] != "rz-rovent-recovered-dataset-set/1"
            or admission["current_publication_sha256"] != registration["current_dataset"]["publication"]["sha256"]
            or admission["ancestor_publication_sha256"] != expected_ancestors or admission["historical_evidence_only"] is not True
            or any(admission[name] is not False for name in ("past_native_authority_reissued", "past_physical_completion_reissued", "execution_authorized"))):
        raise ValueError("recovered collection data cannot reissue historical native/physical/execution authority")
    _checkpoint_metadata_admission(value["checkpoint_admission"], registration["initialization_source"])
    auxiliary_presence = r_auxiliary_dataset_presence(registration)
    if auxiliary_presence != "enabled":
        if "r_auxiliary_admission" in value:
            raise ValueError("disabled R auxiliary source cannot add a startup admission or default")
    else:
        source = registration["r_auxiliary_dataset"]
        report = _fields(value.get("r_auxiliary_admission"), ("schema", "current_publication_sha256",
            "ancestor_publication_sha256", "manifest_file_sha256", "index_file_sha256", "nnuec_parameter_sha256",
            "record_count", "past_native_authority_reissued", "past_physical_completion_reissued",
            "historical_evidence_only", "execution_authorized"), "separate historical R dataset admission")
        for name in ("current_publication_sha256", "manifest_file_sha256", "index_file_sha256", "nnuec_parameter_sha256"):
            _sha(report[name])
        _uint(report["record_count"], 262144)
        ancestors = _r_auxiliary_ancestor_requests(registration)
        expected = sorted(item["publication"]["sha256"] for item in ancestors)
        if (report["schema"] != "rz-rovent-recovered-r-dataset-set/1"
                or report["current_publication_sha256"] != source["publication"]["sha256"]
                or report["ancestor_publication_sha256"] != expected or report["nnuec_parameter_sha256"] != source["nnuec_parameter_sha256"]
                or report["historical_evidence_only"] is not True
                or any(report[name] is not False for name in ("past_native_authority_reissued", "past_physical_completion_reissued", "execution_authorized"))):
            raise ValueError("R auxiliary startup source cannot alias P/C data or reissue historical authority")
    return copy.deepcopy(value)


def _require_linux_nofollow():
    if (sys.platform != "linux" or not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY")
            or os.open not in os.supports_dir_fd or not hasattr(os, "sched_setaffinity")
            or not hasattr(os, "sched_getaffinity")):
        raise OriginalBootstrapUnavailable("isolated Linux nofollow/cgroup/affinity bootstrap required; no weaker fallback")


def _identity(info):
    return (info.st_dev, info.st_ino, info.st_mode, info.st_nlink, info.st_size,
            info.st_mtime_ns, info.st_ctime_ns)


def _open_directory(path):
    """Walk every absolute component through nofollow directory descriptors."""
    _require_linux_nofollow()
    pure = _absolute(str(path))
    if not isinstance(pure, PurePosixPath):
        raise OriginalBootstrapUnavailable("Linux absolute directory required")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    fd = os.open("/", flags)
    try:
        for part in pure.parts[1:]:
            child = os.open(part, flags, dir_fd=fd)
            os.close(fd)
            fd = child
            info = os.fstat(fd)
            if not stat.S_ISDIR(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
                raise ValueError("original path contains a link/reparse/non-directory ancestor")
        result, fd = fd, None
        return result
    finally:
        if fd is not None:
            os.close(fd)


def _open_file(path):
    pure = _absolute(str(path))
    directory = _open_directory(pure.parent)
    try:
        fd = os.open(pure.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | getattr(os, "O_CLOEXEC", 0), dir_fd=directory)
        return fd
    finally:
        os.close(directory)


class _ExternalReadLease:
    __slots__ = ("owner", "maximum", "charged", "remaining", "closed")

    def __init__(self, owner, maximum):
        self.owner, self.maximum = owner, maximum
        self.charged, self.remaining, self.closed = 0, maximum, False


class PinnedReadBudget:
    """One finite read account shared by bootstrap and subsequent metadata IO.

    Failed reads remain charged.  In the recovered codec this is the child's
    source/runtime/dataset/tensor file-read account.  Startup stdin, stdout and
    RPC have separate original-parent bounds and are not added to this account.
    This account is local child evidence; it does
    not replace the independent parent descriptor/read account or its authority.
    """
    def __init__(self, maximum, *, until_ns, clock=time.monotonic_ns, strict_growth_credit=False):
        self.maximum = _uint(maximum, MAX_READ_BYTES, positive=True)
        self.until_ns = _uint(until_ns, positive=True)
        if not callable(clock):
            raise ValueError("monotonic read clock required")
        self.clock, self.spent, self.failed = clock, 0, False
        self._external_reserved = 0
        if type(strict_growth_credit) is not bool:
            raise ValueError("exact strict growth-probe credit flag required")
        self.strict_growth_credit = strict_growth_credit

    def _check(self):
        if self.failed:
            raise ValueError("failed original read account is quarantined")
        if self.clock() >= self.until_ns:
            raise TimeoutError("original read deadline expired")

    def account_external(self, count):
        """Charge another exact loader's bounded read before it starts.

        This method performs no content validation and no refund.  The caller
        must retain the external loader's own regular-file/hash checks.  Every
        repeated checkpoint/corpus pass must be charged again.
        """
        self._check()
        _uint(count, MAX_READ_BYTES)
        if count > self.maximum - self.spent - self._external_reserved:
            raise ValueError("external loader exceeds original aggregate read credit")
        self.spent += count
        return self.spent

    def reserve_external(self, count):
        """Reserve one exact loader ceiling; a lease is never execution authority."""
        self._check()
        _uint(count, MAX_READ_BYTES, positive=True)
        if count > self.maximum - self.spent - self._external_reserved:
            raise ValueError("external loader reservation exceeds original aggregate read credit")
        self._external_reserved += count
        return _ExternalReadLease(self, count)

    def _external_lease(self, lease, *, closed=False):
        if type(lease) is not _ExternalReadLease or lease.owner is not self or lease.closed and not closed:
            raise ValueError("original read reservation owner or lifetime differs")
        return lease

    def charge_external(self, lease, count):
        """Convert only actually returned block bytes from reservation to charge."""
        lease = self._external_lease(lease)
        _uint(count, MAX_READ_BYTES)
        if count > lease.remaining:
            self.failed = True
            raise ValueError("external read exceeded its reserved byte ceiling")
        lease.remaining -= count
        lease.charged += count
        self._external_reserved -= count
        self.spent += count
        if self.spent + self._external_reserved > self.maximum:
            self.failed = True
            raise ValueError("external read exceeded original aggregate credit")
        if self.clock() >= self.until_ns:
            self.failed = True
            raise TimeoutError("original read deadline expired after the charged block")
        if self.failed:
            raise ValueError("failed original read account is quarantined after charging the returned block")
        return lease.charged

    def finish_external(self, lease, *, expected_bytes=None):
        """Close a validated EOF/hash boundary and release only known unused IO.

        A caller may reserve ``pin.bytes + 1`` for a growth probe.  After exact
        identity/size/hash and EOF checks, ``expected_bytes=pin.bytes`` closes the
        unused probe capacity; no actually read byte is refunded.
        """
        self._check()
        lease = self._external_lease(lease)
        expected = lease.maximum if expected_bytes is None else _uint(expected_bytes, lease.maximum)
        if lease.charged != expected:
            self.failed = True
            raise ValueError("external exact read completion differs from the pinned byte count")
        unused = lease.remaining
        self._external_reserved -= unused
        lease.remaining, lease.closed = 0, True
        return {"charged_bytes": lease.charged, "known_unused_reserved_bytes": unused}

    def fail_external(self, lease):
        """Keep unread reservation on uncertain failure; never retry/refund it."""
        self._external_lease(lease, closed=True)
        self.failed = True

    def observation(self):
        """Current child accounting only; the parent keeps its whole reservation."""
        return {"charged_bytes": self.spent, "unknown": self.failed or self._external_reserved != 0,
            "maximum_bytes": self.maximum}

    def _read(self, pin, *, maximum, retain):
        self._check()
        _pin(pin, maximum)
        if pin["bytes"] + int(self.strict_growth_credit) > self.maximum - self.spent - self._external_reserved:
            raise ValueError("original aggregate read credit exhausted")
        fd = None
        try:
            fd = _open_file(pin["path"])
            before = os.fstat(fd)
            if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_size != pin["bytes"]:
                raise ValueError("original regular/single-link file identity or size differs")
            total, hasher, chunks = 0, hashlib.sha256(), []
            while total < pin["bytes"]:
                self._check()
                block = os.read(fd, min(65536, pin["bytes"] - total))
                self.spent += len(block)
                total += len(block)
                if not block:
                    break
                hasher.update(block)
                if retain:
                    chunks.append(block)
            self._check()
            extra = os.read(fd, 1)
            self.spent += len(extra)
            after = os.fstat(fd)
            # Reopening through a second nofollow walk detects path replacement
            # without using a symlink-following path.stat after the FD read.
            current_fd = _open_file(pin["path"])
            try:
                current = os.fstat(current_fd)
            finally:
                os.close(current_fd)
            if (total != pin["bytes"] or extra or self.spent > self.maximum
                    or _identity(before) != _identity(after) or _identity(after) != _identity(current)
                    or hasher.hexdigest() != pin["sha256"]):
                raise ValueError("original pinned file changed or differs during readback")
            identity = {"path": pin["path"], "sha256": hasher.hexdigest(), "bytes": total,
                "device": after.st_dev, "inode": after.st_ino}
            return (b"".join(chunks) if retain else None), identity
        except BaseException:
            self.failed = True
            raise
        finally:
            if fd is not None:
                os.close(fd)

    def read(self, pin, *, maximum=MAX_JSON_BYTES):
        """Return bounded original bytes and actual file identity."""
        return self._read(pin, maximum=maximum, retain=True)

    def verify(self, pin, *, maximum=MAX_JSON_BYTES):
        """Stream a pin without retaining executable/library-sized bytes."""
        return self._read(pin, maximum=maximum, retain=False)[1]

    def json(self, pin, *, maximum=MAX_JSON_BYTES):
        raw, _identity_readback = self.read(pin, maximum=min(maximum, MAX_JSON_BYTES))
        return _strict_json(raw, maximum=min(maximum, MAX_JSON_BYTES))


def read_original_pin(pin, *, maximum=MAX_JSON_BYTES, until_ns, clock=time.monotonic_ns):
    """Standalone bounded metadata read; aggregate callers use one shared budget."""
    _pin(pin, maximum)
    return PinnedReadBudget(pin["bytes"], until_ns=until_ns, clock=clock).read(pin, maximum=maximum)


def _kernel_text(directory, name, maximum, *, until_ns, clock):
    if type(name) is not str or "/" in name or name in ("", ".", ".."):
        raise ValueError("plain kernel snapshot filename required")
    if clock() >= until_ns:
        raise TimeoutError("original resource readback deadline expired")
    fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | getattr(os, "O_CLOEXEC", 0), dir_fd=directory)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("resource snapshot is not a kernel regular file")
        chunks, total = [], 0
        while True:
            if clock() >= until_ns:
                raise TimeoutError("original resource snapshot expired")
            block = os.read(fd, min(4096, maximum + 1 - total))
            total += len(block)
            if total > maximum:
                raise ValueError("kernel snapshot exceeds explicit byte cap")
            if not block:
                break
            chunks.append(block)
        after = os.fstat(fd)
        if (before.st_dev, before.st_ino, before.st_mode) != (after.st_dev, after.st_ino, after.st_mode):
            raise ValueError("kernel snapshot descriptor changed")
        # Kernel mount/membership paths may contain UTF-8 user names.  Numeric
        # controls are still checked separately as canonical ASCII digits.
        return b"".join(chunks).decode("utf-8", "strict")
    finally:
        os.close(fd)


def _canonical_number(text):
    if re.fullmatch(r"0|[1-9][0-9]{0,19}", text) is None:
        raise ValueError("finite canonical kernel integer required")
    return _uint(int(text))


def _resource_snapshot(directory, *, until_ns, clock):
    cpu = _kernel_text(directory, "cpu.max", 128, until_ns=until_ns, clock=clock).split()
    if len(cpu) != 2:
        raise ValueError("kernel cpu.max field count differs")
    result = {"cpu_quota_us": _canonical_number(cpu[0]), "cpu_period_us": _canonical_number(cpu[1])}
    for field, filename in (("memory_high_bytes", "memory.high"), ("memory_max_bytes", "memory.max"),
            ("memory_swap_max_bytes", "memory.swap.max"), ("pids_max", "pids.max")):
        result[field] = _canonical_number(_kernel_text(directory, filename, 128, until_ns=until_ns, clock=clock).strip())
    return result


def _membership(text):
    entries = [line[3:] for line in text.splitlines() if line.startswith("0::")]
    if len(entries) != 1:
        raise ValueError("one unified cgroup membership required")
    path = _absolute(entries[0])
    if not isinstance(path, PurePosixPath):
        raise ValueError("Linux unified membership required")
    return str(path)


def _mount_unescape(value):
    def replace(match):
        return chr(int(match.group(1), 8))
    return re.sub(r"\\([0-7]{3})", replace, value)


def _verify_cgroup_mount(text, resources, directory_info):
    """Join the original path to a real cgroup2 mount and namespace membership."""
    candidates = []
    path = PurePosixPath(resources["path"])
    for line in text.splitlines():
        left, separator, right = line.partition(" - ")
        if not separator:
            raise ValueError("malformed Linux mountinfo snapshot")
        before, after = left.split(), right.split()
        if len(before) < 6 or len(after) < 3:
            raise ValueError("incomplete Linux mountinfo snapshot")
        if after[0] != "cgroup2":
            continue
        root = PurePosixPath(_mount_unescape(before[3]))
        mount = PurePosixPath(_mount_unescape(before[4]))
        if not path.is_relative_to(mount):
            continue
        if before[2] != str(os.major(directory_info.st_dev)) + ":" + str(os.minor(directory_info.st_dev)):
            continue
        candidates.append((len(mount.parts), root / path.relative_to(mount)))
    if not candidates or str(max(candidates, key=lambda item: item[0])[1]) != resources["membership"]:
        raise ValueError("resource directory is not the declared cgroup2 namespace member")


def _join_resources(startup, *, until_ns, clock):
    """Move only this still-single-threaded child; never change cgroup limits."""
    resources, policy = startup["resources"], startup["resources"]["policy"]
    pid = os.getpid()
    proc, leaf = None, None
    try:
        proc = _open_directory("/proc/" + str(pid))
        leaf = _open_directory(resources["path"])
        leaf_info = os.fstat(leaf)
        if ((leaf_info.st_dev, leaf_info.st_ino) != (resources["device"], resources["inode"])
                or leaf_info.st_uid != os.geteuid() or stat.S_IMODE(leaf_info.st_mode) != 0o700):
            raise ValueError("original cgroup leaf descriptor identity differs")
        task = os.open("task", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=proc)
        try:
            if os.listdir(task) != [str(pid)]:
                raise OriginalBootstrapUnavailable("resource admission requires the single-threaded pre-neural child")
        finally:
            os.close(task)
        mountinfo = _kernel_text(proc, "mountinfo", 256 << 10, until_ns=until_ns, clock=clock)
        _verify_cgroup_mount(mountinfo, resources, leaf_info)
        _resource_readback(_resource_snapshot(leaf, until_ns=until_ns, clock=clock), policy)
        before_procs = _kernel_text(leaf, "cgroup.procs", 64 << 10, until_ns=until_ns, clock=clock).split()
        if before_procs and before_procs != [str(pid)]:
            raise ValueError("original cgroup leaf already contains another process")
        before_affinity = set(os.sched_getaffinity(0))
        expected_affinity = set(policy["affinity"])
        if not expected_affinity <= before_affinity:
            raise ValueError("registered affinity exceeds the inherited allowed CPU set")
        if clock() >= until_ns:
            raise TimeoutError("original resource admission expired before migration")
        procs = os.open("cgroup.procs", os.O_WRONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=leaf)
        try:
            if not stat.S_ISREG(os.fstat(procs).st_mode):
                raise ValueError("cgroup migration target is not a kernel regular file")
            raw = (str(pid) + "\n").encode("ascii")
            if os.write(procs, raw) != len(raw):
                raise ValueError("partial self-only cgroup migration; original parent must preserve failure")
        finally:
            os.close(procs)
        os.sched_setaffinity(0, expected_affinity)
        membership = _membership(_kernel_text(proc, "cgroup", 16 << 10, until_ns=until_ns, clock=clock))
        current_procs = _kernel_text(leaf, "cgroup.procs", 64 << 10, until_ns=until_ns, clock=clock).split()
        affinity = set(os.sched_getaffinity(0))
        observed = _resource_snapshot(leaf, until_ns=until_ns, clock=clock)
        _resource_readback(observed, policy)
        check_leaf = _open_directory(resources["path"])
        try:
            current_leaf = os.fstat(check_leaf)
        finally:
            os.close(check_leaf)
        if (membership != resources["membership"] or current_procs != [str(pid)] or affinity != expected_affinity
                or (current_leaf.st_dev, current_leaf.st_ino) != (leaf_info.st_dev, leaf_info.st_ino)
                or clock() >= until_ns):
            raise ValueError("actual child membership/affinity/resource identity differs")
        return {"pid": pid, "membership": membership, "affinity": sorted(affinity), "readback": observed,
            "scope": "self_only_pre_neural_child_snapshot_not_process_lifetime_or_physical_completion_proof",
            "automatic_cleanup": False}
    finally:
        if leaf is not None:
            os.close(leaf)
        if proc is not None:
            os.close(proc)


def _directory_readback(expected):
    fd = _open_directory(expected["path"])
    try:
        observed = os.fstat(fd)
        if (observed.st_dev, observed.st_ino) != (expected["device"], expected["inode"]):
            raise ValueError("original dependency directory identity differs")
        return dict(expected)
    finally:
        os.close(fd)


def _owned_empty_child(parent, name, *, until_ns, clock):
    """Create or open one empty 0700 child without following another owner."""
    if re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", name) is None:
        raise ValueError("plain owned runtime directory basename required")
    if clock() >= until_ns:
        raise TimeoutError("original runtime directory preparation expired")
    created = False
    try:
        os.mkdir(name, 0o700, dir_fd=parent)
        created = True
    except FileExistsError:
        pass
    fd = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0), dir_fd=parent)
    try:
        info = os.fstat(fd)
        if (not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid()
                or stat.S_IMODE(info.st_mode) != 0o700):
            raise ValueError("runtime directory must have the current owner and exact 0700 mode")
        with os.scandir(fd) as entries:
            if next(entries, None) is not None:
                raise ValueError("original runtime directory must be new or empty; existing files are preserved")
        if clock() >= until_ns:
            raise TimeoutError("original runtime directory preparation expired")
        result, fd = fd, None
        return result, {"device": info.st_dev, "inode": info.st_ino, "created": created, "mode": "0700"}
    finally:
        if fd is not None:
            os.close(fd)


def _prepare_owned_runtime_environment(startup, *, until_ns, clock):
    """Keep lazy library caches and temporary files under the observed output.

    Only these eight public path variables are written.  No inherited secret
    value is copied, no global cache is inspected, and no file is removed on
    failure.  The original parent's output-tree owner still accounts every
    subsequently created file; this helper issues no output or GPU authority.
    """
    _require_linux_nofollow()
    if clock() >= until_ns:
        raise TimeoutError("original runtime environment preparation expired")
    temporary_module = sys.modules.get("tempfile")
    if temporary_module is not None and getattr(temporary_module, "tempdir", None) is not None:
        raise OriginalBootstrapUnavailable("temporary directory was cached before the owned runtime environment")
    root_path = _absolute(startup["registration"]["output_root"])
    root = _open_directory(root_path)
    owned, observations = {}, []
    try:
        initial = os.fstat(root)
        if initial.st_uid != os.geteuid() or stat.S_IMODE(initial.st_mode) != 0o700:
            raise ValueError("original output root must retain its current owner and 0700 mode")
        for name in ("tmp", "home", "runtime-cache"):
            child, observation = _owned_empty_child(root, name, until_ns=until_ns, clock=clock)
            owned[name] = child
            observations.append({"path": str(root_path / name), **observation})
        for name in ("xdg", "cuda", "torch", "torch-extensions"):
            child, observation = _owned_empty_child(owned["runtime-cache"], name, until_ns=until_ns, clock=clock)
            owned["runtime-cache/" + name] = child
            observations.append({"path": str(root_path / "runtime-cache" / name), **observation})
        environment = {"TMPDIR": str(root_path / "tmp"), "TMP": str(root_path / "tmp"),
            "TEMP": str(root_path / "tmp"), "HOME": str(root_path / "home"),
            "XDG_CACHE_HOME": str(root_path / "runtime-cache" / "xdg"),
            "CUDA_CACHE_PATH": str(root_path / "runtime-cache" / "cuda"),
            "TORCH_HOME": str(root_path / "runtime-cache" / "torch"),
            "TORCH_EXTENSIONS_DIR": str(root_path / "runtime-cache" / "torch-extensions")}
        # Reopen all named children immediately before publication.  An owner
        # with the same UID can still change a directory later; the parent
        # retains its independent tree watch and process-lifetime boundary.
        for name, descriptor in owned.items():
            check = _open_directory(root_path / name)
            try:
                current, retained = os.fstat(check), os.fstat(descriptor)
                if ((current.st_dev, current.st_ino) != (retained.st_dev, retained.st_ino)
                        or current.st_uid != os.geteuid() or stat.S_IMODE(current.st_mode) != 0o700):
                    raise ValueError("owned runtime directory identity changed before environment publication")
            finally:
                os.close(check)
        check_root = _open_directory(root_path)
        try:
            final = os.fstat(check_root)
            if ((initial.st_dev, initial.st_ino) != (final.st_dev, final.st_ino)
                    or final.st_uid != os.geteuid() or stat.S_IMODE(final.st_mode) != 0o700
                    or clock() >= until_ns):
                raise ValueError("original output root identity/deadline changed during environment preparation")
        finally:
            os.close(check_root)
        for name in _RUNTIME_ENVIRONMENT:
            os.environ[name] = environment[name]
        readback = {name: os.environ.get(name) for name in _RUNTIME_ENVIRONMENT}
        if readback != environment or clock() >= until_ns:
            raise ValueError("owned public runtime environment readback differs")
        return {"schema": "rz-rovent-owned-runtime-environment/1", "variables": readback,
            "directories": observations, "global_cache_inspected": False, "automatic_cleanup": False,
            "scope": "public_paths_inside_original_output_tree_not_output_credit_or_process_lifetime_authority"}
    finally:
        for descriptor in owned.values():
            os.close(descriptor)
        os.close(root)


@dataclass(frozen=True)
class BootstrapReadback:
    report: dict
    budget: PinnedReadBudget


def _validate_inherited_search_path(package, dependencies):
    """Keep the isolated interpreter's stdlib paths and one private package."""
    version = "python" + str(sys.version_info.major) + "." + str(sys.version_info.minor)
    archive = "python" + str(sys.version_info.major) + str(sys.version_info.minor) + ".zip"
    allowed = {package}
    for prefix in {sys.base_prefix, sys.base_exec_prefix}:
        for libdir in {"lib", getattr(sys, "platlibdir", "lib")}:
            base = str(Path(prefix) / libdir / version)
            allowed.update((base, str(Path(base) / "lib-dynload"), str(Path(prefix) / libdir / archive)))
    # The mapped interpreter may use an explicitly configured stdlib directory
    # outside its install prefix.  Its already imported stdlib module locations
    # are observations, not arbitrary caller paths or site-package locations.
    for module in (os, stat):
        location = getattr(module, "__file__", None)
        if type(location) is str and Path(location).name in ("os.py", "stat.py"):
            base = str(Path(location).parent)
            allowed.update((base, str(Path(base) / "lib-dynload")))
    if (any(type(path) is not str or path not in allowed for path in sys.path)
            or len(sys.path) != len(set(sys.path)) or sys.path.count(package) > 1
            or any(item["path"] in sys.path for item in dependencies)):
        raise ValueError("unregistered, duplicate or inherited dependency search path")


def bootstrap_original_child(startup, *, until_ns, clock=time.monotonic_ns):
    """Explicitly verify the isolated child, then self-join the original leaf.

    The caller must first use ``validate_startup`` with the original argv/raw
    pins.  This routine repeats structural validation without inventing new
    deadlines.  It returns a live shared read budget for subsequent metadata IO.
    A failed migration is never rolled back or hidden by a resource fallback.
    """
    return _bootstrap_child_material(startup, until_ns=until_ns, clock=clock,
        validator=validate_startup, entry_basename="rovent_training_child.py", executable_maximum=MAX_EXECUTABLE_BYTES,
        report_domain="rz-rovent-original-trainer-bootstrap-readback/1", recovered=False)


def bootstrap_recovered_child(startup, *, until_ns, clock=time.monotonic_ns):
    """Bootstrap only the new recovered codec before any neural import.

    This function does not load a tensor, observe an old process, or signal a
    historical PID.  The current parent owns reclamation, the new execution
    peer and each resulting process/physical-completion lifetime.
    """
    return _bootstrap_child_material(startup, until_ns=until_ns, clock=clock,
        validator=validate_recovered_startup, entry_basename="rovent_recovered_training_child.py", executable_maximum=MAX_RUNTIME_MODULE_BYTES,
        report_domain="rz-rovent-recovered-trainer-bootstrap-readback/1", recovered=True)


def _bootstrap_child_material(startup, *, until_ns, clock, validator, entry_basename, executable_maximum, report_domain, recovered):
    _require_linux_nofollow()
    if not (sys.flags.isolated and sys.flags.no_site and sys.flags.dont_write_bytecode):
        raise OriginalBootstrapUnavailable("original learner requires explicit -I -S -B")
    if any(name.split(".", 1)[0] in {"torch", "numpy", "onnx", "onnxruntime"} for name in sys.modules):
        raise OriginalBootstrapUnavailable("resource bootstrap must precede every neural/numeric runtime import")
    raw_projection = json.dumps(startup, sort_keys=True, ensure_ascii=False,
        allow_nan=False, separators=(",", ":")).encode("utf-8")
    # This is structural revalidation only.  The externally pinned original raw
    # SHA was already checked by the caller and must remain its socket binding.
    validator(raw_projection, expected_sha256=hashlib.sha256(raw_projection).hexdigest(),
        expected_bytes=len(raw_projection), execution_until_ns=until_ns,
        whole_until_ns=startup["whole_until_ns"], now_ns=clock())
    maximum = startup["child_read_credit"]["reserved_bytes"] if recovered else startup["registration"]["max_read_bytes"]
    budget = PinnedReadBudget(maximum, until_ns=until_ns, clock=clock, strict_growth_credit=recovered)
    source_root = Path(startup["source_root"])
    source_readbacks = []
    inventory_paths = set()
    for source in startup["source_files"]:
        path = source_root / source["relative_path"]
        pin = {"path": str(path), "bytes": source["bytes"], "sha256": source["sha256"]}
        source_readbacks.append(budget.verify(pin, maximum=MAX_SOURCE_BYTES))
        inventory_paths.add(str(path))
    main = sys.modules.get("__main__")
    entry = getattr(main, "__file__", None)
    if (type(entry) is not str or str(Path(entry).absolute()) not in inventory_paths
            or Path(entry).name != entry_basename
            or str(Path(__file__).absolute()) not in inventory_paths):
        raise ValueError("executing original entry/bootstrap files are absent from the pinned source inventory")
    executable = budget.verify(startup["python_runtime"]["executable"], maximum=executable_maximum)
    mapped = os.stat("/proc/" + str(os.getpid()) + "/exe")
    actual_version = ".".join(str(part) for part in sys.version_info[:3]) if recovered else sys.version.split()[0]
    if (actual_version != startup["python_runtime"]["version"]
            or (mapped.st_dev, mapped.st_ino) != (executable["device"], executable["inode"])):
        raise ValueError("actual mapped Python interpreter/version differs from original runtime pin")
    runtime_modules = [budget.verify(pin, maximum=MAX_RUNTIME_MODULE_BYTES) for pin in startup["runtime_module_pins"]]
    dependencies = [_directory_readback(item) for item in startup["dependency_directories"]]
    package = str(Path(startup["package_root"]))
    # The original runpy bootstrap may already have inserted this one source
    # package root.  No dependency path may be inherited before registration.
    _validate_inherited_search_path(package, dependencies)
    resources = _join_resources(startup, until_ns=until_ns, clock=clock)
    environment = _prepare_owned_runtime_environment(startup, until_ns=until_ns, clock=clock)
    if package not in sys.path:
        sys.path.insert(0, package)
    for dependency in dependencies:
        sys.path.append(dependency["path"])
    report = {"schema": report_domain,
        "source_files": source_readbacks, "runtime_module_pins": runtime_modules,
        "python_runtime": {"version": sys.version, "executable": executable,
            "mapped_executable_identity": {"device": mapped.st_dev, "inode": mapped.st_ino}},
        "dependency_directories": dependencies, "dependency_pth_executed": False,
        "isolated": True, "no_site": True, "bytecode_disabled": True, "resources": resources,
        "runtime_environment": environment,
        "read_bytes": budget.spent, "original_read_limit_bytes": budget.maximum,
        "registration_projection_scope": "original_parent_raw_registration_pin_not_child_byte_reconstruction",
        "execution_authorized": False, "physical_completion": "not_started"}
    if recovered:
        # Startup carries the declared parent watch.  Its concrete ownership,
        # admission and final observation remain with the current Rust parent.
        report["artifact_watch_profile"] = copy.deepcopy(startup["artifact_watch_profile"])
    return BootstrapReadback(report, budget)
