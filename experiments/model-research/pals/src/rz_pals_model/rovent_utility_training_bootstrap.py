"""Exact utility startup codec over explicitly shared OS/file primitives.

The common material helper's ``recovered=True`` selects strict child read
credit, exact Python version and artifact-watch readback only. It does not
select a utility authority, reinterpret a P/C/V codec, or revive an old lease.
This module supplies its own registration/startup validator, source domain,
entry file and report domain. The original stdin byte SHA remains separate.
"""
from __future__ import annotations

import copy
import hashlib
import re
import time

from .rovent_training_bootstrap import (MAX_JSON_BYTES, MAX_READ_BYTES, MAX_SOURCE_BYTES,
    MAX_SOURCE_TOTAL_BYTES, MAX_RUNTIME_MODULE_BYTES, MAX_WALL_NS, MAX_CLEANUP_NS,
    PACKAGE_RELATIVE, MODULE_RELATIVE, _absolute, _child_path, _digest, _fields, _identifier,
    _pin, _policy, _resources, _sha, _strict_json, _uint, _bootstrap_child_material,
    recovered_dependency_set_digest)


STARTUP_DOMAIN = "rz-rovent-utility-trainer-startup/1"
REGISTRATION_DOMAIN = "rz-rovent-utility-training-registration/1"
REGISTRATION_V2_DOMAIN = "rz-rovent-utility-training-registration/2"
SOURCE_DOMAIN = "rz-rovent-utility-trainer-source-bundle/1"
RESULT_DOMAIN = "rz-rovent-utility-trainer-result/1"
REGISTRATION_FIELDS = ("schema", "execution_id", "action", "export_request", "preparation", "window_spec",
    "trainer_source_sha256", "trainer_python_executable", "trainer_python_version", "dependency_directories",
    "trainer_dependency_set_sha256", "runtime_endpoints", "current_dataset", "ancestor_datasets",
    "initialization_source", "frozen_nnuec_base", "frozen_pc_sources", "resources_policy", "require_total_quota",
    "output_root", "control_root", "checkpoint_directory", "receipt_path", "max_rpc_messages", "max_rpc_bytes",
    "max_read_bytes", "max_child_read_bytes", "max_output_bytes", "max_artifact_files")
STARTUP_FIELDS = ("schema", "execution_id", "window_id", "registration_sha256", "registration", "control",
    "original_started_ns", "execution_until_ns", "whole_until_ns", "source_root", "package_root", "source_files",
    "dependency_directories", "runtime_module_pins", "python_runtime", "resources", "artifact_watch_profile", "child_read_credit")


def _checkpoint_reference(value):
    _fields(value, ("root", "publication"), "utility original publication source")
    _absolute(value["root"])
    _pin(value["publication"])
    _child_path(value["root"], value["publication"]["path"], "utility publication")
    return value


def _dataset_reference(value):
    _fields(value, ("root", "publication", "durable_ack"), "utility original cohort dataset source")
    _absolute(value["root"])
    for name in ("publication", "durable_ack"):
        _pin(value[name])
        _child_path(value["root"], value[name]["path"], "utility cohort " + name)
    return value


def validate_registration(value):
    """Validate declaration bytes only, never original dataset/lease authority."""
    return _validate_registration(value, expected_domain=REGISTRATION_DOMAIN)


def _ancestor_v2(value):
    if type(value) is not dict:
        raise ValueError("explicit original mixed ancestor object required")
    kind = value.get("kind")
    if kind in ("pc_legacy", "pc_submitted_input_v1"):
        _fields(value, ("kind", "registration"), "original PC ancestor")
        from .rovent_historical_corpus import _registration as validate_pc_registration
        validate_pc_registration(value["registration"])
    elif kind == "r_auxiliary":
        _fields(value, ("kind", "registration", "recipe"), "original R ancestor")
        from .rovent_historical_r_corpus import _registration as validate_r_registration
        validate_r_registration(value["registration"])
        _pin(value["recipe"], MAX_JSON_BYTES)
    elif kind == "utility_cohort":
        _fields(value, ("kind", "registration", "preparation"), "original utility ancestor")
        _dataset_reference(value["registration"])
        _pin(value["preparation"], 65536)
    else:
        raise ValueError("explicit original PC/R/utility ancestor kind required")
    return value["registration"]["publication"]["sha256"]


def validate_utility_registration_v2(value):
    """Separate mixed-ancestry codec; never re-tag an original /1 object."""
    return _validate_registration(value, expected_domain=REGISTRATION_V2_DOMAIN)


def _validate_registration(value, *, expected_domain):
    return _validate_registration_core(value, expected_domain=expected_domain, intervention=False)


def _validate_registration_core(value, *, expected_domain, intervention):
    if (type(intervention) is not bool
            or intervention and expected_domain != "rz-rovent-utility-training-registration/3"
            or not intervention and expected_domain not in (REGISTRATION_DOMAIN, REGISTRATION_V2_DOMAIN)):
        raise ValueError("explicit closed utility registration source family required")
    _fields(value, (*REGISTRATION_FIELDS, "data_family") if intervention else REGISTRATION_FIELDS, "utility execution registration")
    if value["schema"] != expected_domain:
        raise ValueError("separate utility execution registration required; no codec conversion")
    if intervention and value["data_family"] != "intervention_v2":
        raise ValueError("explicit intervention_v2 execution source family required")
    _identifier(value["execution_id"])
    _pin(value["preparation"], 65536); _pin(value["window_spec"], 1024**2)
    _sha(value["trainer_source_sha256"])
    _pin(value["trainer_python_executable"], MAX_RUNTIME_MODULE_BYTES)
    version = value["trainer_python_version"]
    if (type(version) is not str or re.fullmatch(r"3\.[0-9]{1,2}\.[0-9]{1,3}", version) is None
            or any(str(int(part)) != part for part in version.split("."))):
        raise ValueError("exact utility trainer Python release required")
    directories = value["dependency_directories"]
    if type(directories) is not list or not 1 <= len(directories) <= 8:
        raise ValueError("bounded original utility dependency directories required")
    paths = [str(_absolute(path)) for path in directories]
    if len(set(paths)) != len(paths):
        raise ValueError("duplicate utility dependency directory")
    if _sha(value["trainer_dependency_set_sha256"]) != recovered_dependency_set_digest(value["runtime_endpoints"]):
        raise ValueError("utility runtime endpoint declaration differs from its pure dependency seal")
    _dataset_reference(value["current_dataset"])
    if type(value["ancestor_datasets"]) is not list or len(value["ancestor_datasets"]) > 256:
        raise ValueError("bounded original utility dataset ancestry required")
    seen = {value["current_dataset"]["publication"]["sha256"]}
    for ancestor in value["ancestor_datasets"]:
        if intervention:
            from .rovent_utility_intervention_execution import validate_intervention_ancestor_registration
            identity = validate_intervention_ancestor_registration(ancestor)
        elif expected_domain == REGISTRATION_V2_DOMAIN:
            identity = _ancestor_v2(ancestor)
        else:
            _dataset_reference(ancestor)
            identity = ancestor["publication"]["sha256"]
        if identity in seen:
            raise ValueError("duplicate or current utility ancestor publication")
        seen.add(identity)
    _pin(value["frozen_nnuec_base"], MAX_RUNTIME_MODULE_BYTES)
    frozen = value["frozen_pc_sources"]
    if type(frozen) is not list or len(frozen) != 2:
        raise ValueError("both original frozen P/C source pins are required")
    roles = set()
    for source in frozen:
        _fields(source, ("role", "artifact", "model_semantics_sha256", "model_identity_sha256", "encoding_sha256"), "frozen P/C source")
        if type(source["role"]) is not str or source["role"] not in ("proposer", "critic") or source["role"] in roles:
            raise ValueError("distinct proposer/critic original source pins required")
        roles.add(source["role"])
        _pin(source["artifact"], MAX_RUNTIME_MODULE_BYTES)
        for name in ("model_semantics_sha256", "model_identity_sha256", "encoding_sha256"):
            _sha(source[name])
    source = value["initialization_source"]
    if type(source) is not dict:
        raise ValueError("explicit tagged utility initialization source required")
    mode = source.get("mode")
    if mode == "initialize":
        _fields(source, ("mode", "utility"), "initial utility source")
        _pin(source["utility"], MAX_RUNTIME_MODULE_BYTES)
    elif mode in ("resume", "fork"):
        _fields(source, ("mode", "checkpoint"), "historical utility checkpoint source")
        _checkpoint_reference(source["checkpoint"])
    else:
        raise ValueError("registered initialize/resume/fork utility source required")
    if value["action"] == "train":
        if value["export_request"] is not None:
            raise ValueError("utility train cannot carry an export request")
    elif value["action"] == "export":
        _pin(value["export_request"], 65536)
        if mode != "resume":
            raise ValueError("learned utility export requires a separately verified resumed source")
    else:
        raise ValueError("explicit utility train or export action required")
    _policy(value["resources_policy"])
    if type(value["require_total_quota"]) is not bool:
        raise ValueError("exact utility total-quota requirement required")
    output, control = _absolute(value["output_root"]), _absolute(value["control_root"])
    if type(output) is not type(control) or output.is_relative_to(control) or control.is_relative_to(output):
        raise ValueError("utility output and control owners overlap")
    _child_path(value["output_root"], value["checkpoint_directory"], "utility checkpoints")
    receipt = _child_path(value["output_root"], value["receipt_path"], "utility terminal receipt")
    if (expected_domain == REGISTRATION_V2_DOMAIN or intervention) and (receipt.parent != output or receipt.suffix != ".json"
            or receipt == _absolute(value["checkpoint_directory"])):
        raise ValueError("utility /2 terminal must be a distinct direct output-root JSON file")
    _uint(value["max_rpc_messages"], 2000000, positive=True)
    _uint(value["max_rpc_bytes"], 18 << 30, positive=True)
    _uint(value["max_read_bytes"], MAX_READ_BYTES, positive=True)
    _uint(value["max_child_read_bytes"], value["max_read_bytes"], positive=True)
    _uint(value["max_output_bytes"], 18 << 30, positive=True)
    _uint(value["max_artifact_files"], 32768, positive=True)
    return copy.deepcopy(value)


def validate_startup(raw, *, expected_sha256, expected_bytes, execution_until_ns, whole_until_ns, now_ns):
    from .rovent_utility_training_family import LEGACY
    return _validate_startup_core(raw, expected_sha256=expected_sha256, expected_bytes=expected_bytes,
        execution_until_ns=execution_until_ns, whole_until_ns=whole_until_ns, now_ns=now_ns, family=LEGACY)


def _validate_startup_core(raw, *, expected_sha256, expected_bytes, execution_until_ns, whole_until_ns, now_ns, family):
    from .rovent_utility_training_family import require_family
    family = require_family(family)
    _sha(expected_sha256); _uint(expected_bytes, MAX_JSON_BYTES, positive=True)
    if type(raw) is not bytes or len(raw) != expected_bytes or hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("original utility startup bytes/length/SHA differ")
    value = _strict_json(raw)
    _fields(value, (*STARTUP_FIELDS, "data_family") if family.intervention else STARTUP_FIELDS, "utility startup")
    if value["schema"] != family.domain(STARTUP_DOMAIN):
        raise ValueError("separate utility startup domain required")
    if family.intervention and value["data_family"] != "intervention_v2":
        raise ValueError("explicit intervention startup data family required")
    _identifier(value["execution_id"]); _identifier(value["window_id"]); _sha(value["registration_sha256"])
    original_registration = value["registration"]
    if family.intervention:
        from .rovent_utility_intervention_execution import validate_intervention_registration
        registration = validate_intervention_registration(original_registration)
    elif type(original_registration) is dict and original_registration.get("schema") == REGISTRATION_V2_DOMAIN:
        registration = validate_utility_registration_v2(original_registration)
    else:
        registration = validate_registration(original_registration)
    if value["execution_id"] != registration["execution_id"]:
        raise ValueError("utility startup execution differs from original registration")
    for number in (value["original_started_ns"], value["execution_until_ns"], value["whole_until_ns"], execution_until_ns, whole_until_ns, now_ns):
        _uint(number)
    if (value["execution_until_ns"] != execution_until_ns or value["whole_until_ns"] != whole_until_ns
            or not value["original_started_ns"] <= now_ns < execution_until_ns < whole_until_ns
            or execution_until_ns - value["original_started_ns"] > MAX_WALL_NS
            or whole_until_ns - execution_until_ns > MAX_CLEANUP_NS):
        raise ValueError("utility startup deadline changed, expired or renewed")
    control = _fields(value["control"], ("socket_path", "parent_pid", "parent_start_ticks", "control_device", "control_inode"), "utility control")
    endpoint = _child_path(registration["control_root"], control["socket_path"], "utility control socket")
    if endpoint.parent != _absolute(registration["control_root"]) or len(control["socket_path"].encode()) >= 104:
        raise ValueError("bounded direct utility control socket required")
    _uint(control["parent_pid"], 2**31 - 1, positive=True)
    _uint(control["parent_start_ticks"], positive=True); _uint(control["control_device"]); _uint(control["control_inode"], positive=True)
    source_root = _child_path(registration["output_root"], value["source_root"], "utility private source root")
    if source_root != _absolute(registration["output_root"]) / "source" or _absolute(value["package_root"]) != source_root / PACKAGE_RELATIVE:
        raise ValueError("utility source/package layout differs from original installer")
    sources = value["source_files"]
    if type(sources) is not list or not 1 <= len(sources) <= 128:
        raise ValueError("bounded utility source inventory required")
    inventory, total = {}, 0
    for source in sources:
        _fields(source, ("relative_path", "sha256", "bytes"), "utility source entry")
        relative = source["relative_path"]
        if (type(relative) is not str or not relative.startswith(MODULE_RELATIVE) or not relative.endswith(".py")
                or "\\" in relative or any(part in ("", ".", "..") for part in relative.split("/")) or relative in inventory):
            raise ValueError("utility source entry escaped its original package")
        _absolute(str(source_root / relative)); _sha(source["sha256"])
        total += _uint(source["bytes"], MAX_SOURCE_BYTES, positive=True)
        inventory[relative] = {"bytes": source["bytes"], "sha256": source["sha256"]}
    if total > MAX_SOURCE_TOTAL_BYTES or _digest(SOURCE_DOMAIN, inventory) != registration["trainer_source_sha256"]:
        raise ValueError("utility original source bundle digest/total differs")
    required = {MODULE_RELATIVE + name for name in ("__init__.py", "rovent_utility_training_child.py",
        "rovent_utility_training_bootstrap.py", "rovent_training_bootstrap.py", "rovent_utility_training_authority.py",
        "rovent_utility_training_window.py", "rovent_utility_training.py", "rovent_utility_training_run.py",
        "rovent_utility_tensor_training.py", "nnuec_utility_trainable.py")}
    if family.intervention:
        required.update(MODULE_RELATIVE + name for name in ("rovent_utility_intervention_child.py",
            "rovent_utility_intervention_execution.py", "rovent_utility_intervention_authority.py",
            "rovent_utility_intervention_dataset.py", "rovent_utility_intervention_numerical.py",
            "rovent_utility_intervention_training.py", "rovent_utility_intervention_checkpoint.py",
            "rovent_utility_intervention_preparation.py", "rovent_utility_intervention_payload.py",
            "rovent_utility_intervention_files.py", "rovent_utility_intervention.py", "rovent_utility_training_family.py"))
    if not required <= inventory.keys():
        raise ValueError("actual utility entry/bootstrap/learner source pins are absent")
    dependencies = value["dependency_directories"]
    if type(dependencies) is not list or not 1 <= len(dependencies) <= 8:
        raise ValueError("bounded utility dependency readback required")
    paths, identities = set(), set()
    for dependency in dependencies:
        _fields(dependency, ("path", "device", "inode"), "utility dependency directory")
        path = _absolute(dependency["path"])
        identity = (_uint(dependency["device"]), _uint(dependency["inode"], positive=True))
        if str(path) in paths or identity in identities or path.is_relative_to(source_root) or source_root.is_relative_to(path):
            raise ValueError("aliased utility source/dependency directory")
        paths.add(str(path)); identities.add(identity)
    if [item["path"] for item in dependencies] != registration["dependency_directories"]:
        raise ValueError("utility dependency readback changed original registered order")
    runtime = _fields(value["python_runtime"], ("version", "executable"), "utility Python runtime")
    if runtime["version"] != registration["trainer_python_version"] or runtime["executable"] != registration["trainer_python_executable"]:
        raise ValueError("utility runtime differs from registered trainer Python image")
    expected_modules = [endpoint["artifact"] for endpoint in registration["runtime_endpoints"]]
    if value["runtime_module_pins"] != expected_modules:
        raise ValueError("utility runtime endpoints changed their original pin order")
    credit = _fields(value["child_read_credit"], ("reserved_bytes", "scope"), "utility child read credit")
    if (type(credit["reserved_bytes"]) is not int or credit["reserved_bytes"] != registration["max_child_read_bytes"]
            or credit["scope"] != "charged_in_parent_shared_budget_no_refund"):
        raise ValueError("utility child read credit differs from original parent reservation")
    if total + runtime["executable"]["bytes"] + sum(pin["bytes"] for pin in expected_modules) >= credit["reserved_bytes"]:
        raise ValueError("utility bootstrap pins exhaust original child read credit")
    profile = _fields(value["artifact_watch_profile"], ("schema", "maximum_total_bytes", "maximum_file_bytes", "maximum_files", "scope"), "utility artifact watch")
    for name in ("maximum_total_bytes", "maximum_file_bytes", "maximum_files"):
        _uint(profile[name], positive=True)
    reserve = min(registration["max_output_bytes"] // 8, 64 << 20)
    if profile != {"schema": "rz-rovent-training-artifact-watch/1", "maximum_total_bytes": registration["max_output_bytes"] - reserve,
            "maximum_file_bytes": 2 << 30, "maximum_files": registration["max_artifact_files"], "scope": "new_trainer_owned_output_tree_only"}:
        raise ValueError("utility original artifact watch profile differs")
    _resources(value["resources"], registration["resources_policy"])
    return copy.deepcopy(value)


def bootstrap_utility_child(startup, *, until_ns, clock=time.monotonic_ns):
    return _bootstrap_child_material(startup, until_ns=until_ns, clock=clock,
        validator=validate_startup, entry_basename="rovent_utility_training_child.py",
        executable_maximum=MAX_RUNTIME_MODULE_BYTES,
        report_domain="rz-rovent-utility-trainer-bootstrap-readback/1", recovered=True)
