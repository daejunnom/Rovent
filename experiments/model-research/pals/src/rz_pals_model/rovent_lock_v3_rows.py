"""Partial lock-V3 row and lifecycle checks with no training admission.

The Rules replay, typed input lock, descriptor and allocation-clock checks
remain the Rust owner's responsibility. These checks never issue a corpus or
claim that a producer or device is currently live.
"""
from __future__ import annotations

from .generation_data import digest, fields, sha

_FAMILIES = {
    "cpu": ("rz-rovent-training-record-cpu-lock-v3/1", "rz-rovent-native-cpu-lock-v3-raw/1",
            "rz-rovent-collection-root-admission-cpu-lock-v3/1",
            "rz-rovent-collection-native-origin-cpu-lock-v3/1"),
    "cuda": ("rz-rovent-training-record-lock-v3/1", "rz-rovent-native-lock-v3-raw/1",
             "rz-rovent-collection-root-admission-lock-v3/1",
             "rz-rovent-collection-native-origin-lock-v3/1"),
}
_RECORD_FIELDS = ("schema", "role", "split", "input", "input_sha256", "origin_sha256", "record_sha256", "lineage",
                  "native", "head_sources", "targets", "divergence_index", "nnuec_input", "nnuec_input_sha256",
                  "nnuec_model_sha256", "nnuec_semantics", "nnuec_wdl")


def inspect_record_seals(record, family):
    """Check the original record and root domains/seals, not full row validity."""
    if family not in _FAMILIES:
        raise ValueError("explicit V3 CPU or CUDA family required")
    record_domain, raw_domain, root_domain, origin_domain = _FAMILIES[family]
    fields(record, _RECORD_FIELDS, "V3 P/C record")
    if record["schema"] != record_domain or record["role"] not in ("proposer", "critic"):
        raise ValueError("V3 record runtime family or P/C role mismatch")
    sha(record["input_sha256"])
    sha(record["origin_sha256"])
    raw = dict(record)
    raw["input"] = {"sealed_input_sha256": record["input_sha256"]}
    raw["record_sha256"] = ""
    if digest(record_domain, raw) != sha(record["record_sha256"]):
        raise ValueError("V3 original record semantic seal mismatch")
    native = record["native"]
    if type(native) is not dict:
        raise ValueError("V3 original native lifecycle required")
    for key in ("raw_submission", "raw_completion", "raw_retirement"):
        event = native.get(key)
        if type(event) is not dict or event.get("schema") != raw_domain:
            raise ValueError("V3 native event family mismatch")
        if digest(raw_domain, event) != sha(native.get(key + "_sha256")):
            raise ValueError("V3 original native event seal mismatch")
    retirement = native["raw_retirement"]
    root = retirement.get("root_admission")
    if type(root) is not dict or root.get("schema") != root_domain:
        raise ValueError("V3 root family mismatch")
    if digest(root_domain, root) != sha(retirement.get("root_admission_sha256")):
        raise ValueError("V3 original root seal mismatch")
    origin = root.get("native_origin")
    if type(origin) is not dict or origin.get("schema") != origin_domain:
        raise ValueError("V3 original native origin family mismatch")
    if digest(origin_domain, origin) != sha(root.get("native_origin_sha256")):
        raise ValueError("V3 original native origin seal mismatch")
    return None


def inspect_native_lifecycle(manifest_registration, record, family):
    """Check submitted-input/lifecycle receipts; Rules and root remain open."""
    from .rovent_corpus import _native_common
    from .rovent_native_v2 import _COMMON, _submission
    from .rovent_native_v3 import validate_submitted_review
    from .rovent_submitted_input import validate_input, seal_input
    from .generation_data import uint

    inspect_record_seals(record, family)
    snapshot = validate_input(record["input"])
    if snapshot["role"] != record["role"] or seal_input(snapshot) != record["input_sha256"]:
        raise ValueError("V3 submitted input role or seal mismatch")
    native = record["native"]
    extras = {
        "raw_submission": ("model_sha256", "encoding_sha256", "input_digest", "view_digest", "rules_replay", "submitted_review"),
        "raw_completion": ("submitted_sha256", "execution_epoch", "execution_sequence", "completed_nn_inputs", "physical_confirmed",
                           "owner_run_invoked", "measurement_complete", "physical_fresh", "output_digest", "owner_service_elapsed_ns", "output"),
        "raw_retirement": ("submitted_sha256", "completed_sha256", "output_digest", "accepted_retired", "acceptance_kind",
                           "root_admission", "root_admission_sha256"),
    }
    for name, suffix in extras.items():
        event = fields(native[name], (*_COMMON, *suffix), "V3 raw lifecycle event")
        for key in ("sequence", "process_epoch", "game_generation", "root_generation", "request_sequence"):
            uint(event[key])
    completed = native["raw_completion"]
    for key in ("execution_epoch", "execution_sequence", "completed_nn_inputs"):
        uint(completed[key])
    if completed["owner_service_elapsed_ns"] is not None:
        uint(completed["owner_service_elapsed_ns"])
    kind = native["raw_retirement"]["acceptance_kind"]
    if kind not in ("accepted_role_step", "root_ranking_published") or (kind == "root_ranking_published" and record["role"] != "proposer"):
        raise ValueError("V3 native acceptance kind/role mismatch")
    _native_common(manifest_registration, record, _FAMILIES[family][1])
    _submission(record, native["raw_submission"]["rules_replay"])
    validate_submitted_review(record, native["raw_submission"]["submitted_review"])
    return None


def inspect_root_runtime_identity(record, family, lock):
    """Check original root identity and provider observations, not full root validity."""
    from .generation_data import uint

    if family not in _FAMILIES:
        raise ValueError("explicit V3 runtime family required")
    root = fields(record["native"]["raw_retirement"]["root_admission"],
        ("schema", "native_origin", "native_origin_sha256", "process_epoch", "game_generation", "root_generation",
         "implementation_sha256", "deadline_clock", "deadline_tick", "root_budget_sha256", "selective_policy_sha256",
         "root_state", "store_root_state", "root_moves", "depth_plies", "mate_moves", "requested_nodes",
         "root_descriptor", "root_descriptor_sha256", "allocation_clock", "allocation_clock_sha256"), "V3 root")
    origin = fields(root["native_origin"], ("schema", "input_lock_domain", "input_lock_sha256", "declaration",
        "startup_observation", "registered_namespace", "process_boundary_observation", "neural_completion", "physical_completion"),
        "V3 native origin")
    declaration = fields(origin["declaration"], ("purpose", "mode", "source_commit", "source_components", "model_semantics",
        "encoding", "validator_model", "teacher_profiles"), "V3 native declaration")
    common = ("domain", "process_epoch", "mode", "topology", "weights_sha256", "export_manifest_sha256", "backend_sha256",
        "runtime_library_sha256", "provider_ready", "native_configuration_sha256", "startup_configuration_sha256",
        "search_configuration_sha256", "validator_loaded", "physical_workers_created", "neural_service_executed")
    extra = ("provider", "cpu_session_roles") if family == "cpu" else (
        "system_runtime_identity_sha256", "system_libraries", "unregistered_native_mappings")
    observed = fields(origin["startup_observation"], (*common, *extra), "V3 startup observation")
    namespace = fields(origin["registered_namespace"], ("value_semantics", "value_model", "value_encoding", "value_precision",
        "representation_model_domain", "representation_model_sha256", "representation_encoding_sha256", "representation_precision",
        "frozen_epoch"), "V3 representation namespace")
    body, native = lock["manifest"], record["native"]
    engine = body["engine"]
    model = engine["pc"]["proposer"]
    ticket = native["raw_submission"]["ticket"]
    if (origin["input_lock_domain"] != "rz-rovent-input-lock/3" or origin["input_lock_sha256"] != lock["canonical_sha256"]
            or declaration["purpose"] != "collection" or declaration["mode"] != engine["mode"]
            or declaration["source_commit"] != engine["source_commit"]
            or declaration["source_components"] != engine["source_components"]
            or declaration["model_semantics"] != model["semantics"] or declaration["encoding"] != model["encoding"]
            or declaration["validator_model"] != engine["pc"]["validator"] or declaration["teacher_profiles"] != engine["teachers"]
            or observed["mode"] != declaration["mode"] or observed["process_epoch"] != root["process_epoch"]
            or observed["weights_sha256"] != model["weights"]["sha256"]
            or observed["weights_sha256"] != native["model_sha256"]
            or observed["startup_configuration_sha256"] != body["startup_configuration_sha256"]
            or observed["search_configuration_sha256"] != body["startup_search_configuration_sha256"]
            or observed["validator_loaded"] is not (engine["pc"]["validator"] is not None)
            or observed["provider_ready"] is not True or observed["physical_workers_created"] != 1
            or type(observed["physical_workers_created"]) is not int or observed["neural_service_executed"] is not False
            or observed["topology"] not in ("roles_only", "own_researcher", "external_researcher")
            or any(origin[key] != "unknown" for key in ("process_boundary_observation", "neural_completion", "physical_completion"))):
        raise ValueError("V3 original root/runtime declaration join mismatch")
    for key in ("export_manifest_sha256", "backend_sha256", "runtime_library_sha256", "native_configuration_sha256"):
        sha(observed[key])
    if (namespace["representation_model_domain"] != "loaded_checkpoint_sha256"
            or namespace["representation_model_sha256"] != observed["weights_sha256"]
            or namespace["representation_encoding_sha256"] != native["encoding_sha256"]
            or namespace["representation_precision"] != "fp32" or namespace["value_precision"] != "fp32_tf32_off"
            or not all(type(namespace[key]) is str and namespace[key] for key in ("value_semantics", "value_model", "value_encoding"))
            or uint(namespace["frozen_epoch"]) == 0):
        raise ValueError("V3 original representation namespace mismatch")
    for key in ("process_epoch", "game_generation", "root_generation"):
        if uint(root[key]) == 0 or root[key] != native[key]:
            raise ValueError("V3 original root scope mismatch")
    if (ticket["implementation_sha256"] != root["implementation_sha256"]
            or ticket["deadline_clock"] != root["deadline_clock"] or ticket["deadline_tick"] != root["deadline_tick"]
            or ticket["root_state"] != root["root_state"]
            or ticket["representation"]["frozen_epoch"] != namespace["frozen_epoch"]):
        raise ValueError("V3 original ticket/root scope mismatch")
    if family == "cpu":
        runtime = body["system_runtime_lock"]["runtime"]
        artifacts = engine["pc"]["runtime_artifacts"]
        # Match the actual bootstrap assembly: a P/C-only mode has no
        # researcher, while P/C/R requires the declared own or external R.
        # This is checked only in the new CPU family; historical CUDA
        # observations retain their existing compatibility behavior.
        researcher = engine.get("researcher", ...)
        if engine["mode"] == "cpu_pc" and researcher is None:
            expected_topology = "roles_only"
        elif engine["mode"] == "cpu_pcr" and type(researcher) is dict and researcher.get("provider") == "own":
            expected_topology = "own_researcher"
        elif (engine["mode"] == "cpu_pcr" and type(researcher) is dict
              and researcher.get("provider") == "external"
              and type(researcher.get("profile")) is dict
              and researcher["profile"].get("purpose") == "research_checker"):
            expected_topology = "external_researcher"
        else:
            raise ValueError("V3 CPU mode/researcher assembly mismatch")
        if (runtime != {"kind": "cpu"} or engine["pc"]["backend"] != "ort_cpu"
                or engine["pc"]["device"] != {"device": "cpu"}
                or observed["topology"] != expected_topology
                or type(artifacts) is not list or len(artifacts) != 1
                or observed["runtime_library_sha256"] != artifacts[0]["sha256"]
                or observed["native_configuration_sha256"] != engine["pc"]["backend_configuration_sha256"]
                or observed["domain"] != "rz-rovent-native-startup/1" or observed["provider"] != "cpu"
                or observed["cpu_session_roles"] != ["public", "shared_pc", "validator"]
                or engine["pc"]["validator"] is None or observed["validator_loaded"] is not True):
            raise ValueError("V3 actual CPU provider/session observation mismatch")
    else:
        runtime = body["system_runtime_lock"]["runtime"]
        if type(runtime) is not dict or runtime.get("kind") != "system_cuda" or type(runtime.get("configuration")) is not dict:
            raise ValueError("V3 actual system CUDA source required")
        system = runtime["configuration"]
        registered, captured = system.get("libraries"), observed["system_libraries"]
        if (engine["pc"]["backend"] != "ort_cuda" or observed["domain"] not in
                ("rz-rovent-native-startup/1", "rz-rovent-native-system-startup/1")
                or observed["system_runtime_identity_sha256"] != system.get("system_identity_sha256")
                or observed["unregistered_native_mappings"] != []
                or type(registered) is not list or type(captured) is not list or len(registered) != len(captured)):
            raise ValueError("V3 actual CUDA provider observation mismatch")
        captured_by_role = {}
        for item in captured:
            fields(item, ("role", "canonical_path", "sha256", "bytes", "device", "inode", "mapped"), "V3 observed system library")
            if type(item["role"]) is not str or item["role"] in captured_by_role or type(item["mapped"]) is not bool:
                raise ValueError("V3 CUDA mapping must be known and unique")
            captured_by_role[item["role"]] = item
        roles = set()
        for expected in registered:
            if type(expected) is not dict or type(expected.get("role")) is not str or expected["role"] in roles:
                raise ValueError("V3 CUDA registered library roles must be unique")
            roles.add(expected["role"])
            actual = captured_by_role.get(expected["role"])
            if (actual is None or any(actual[key] != expected.get(key) for key in ("canonical_path", "sha256", "bytes", "device", "inode"))
                    or type(expected.get("required_mapping")) is not bool
                    or (expected["required_mapping"] and actual["mapped"] is not True)):
                raise ValueError("V3 observed CUDA library identity or mapping mismatch")
            if expected["role"] == "ort-core" and actual["sha256"] != observed["runtime_library_sha256"]:
                raise ValueError("V3 CUDA ORT core identity mismatch")
        if not {"ort-core", "ort-provider-cuda"} <= roles:
            raise ValueError("V3 CUDA core and provider roles are required")
    return None
