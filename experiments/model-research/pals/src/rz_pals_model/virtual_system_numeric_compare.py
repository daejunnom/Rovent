"""Compare independently captured V3 system-CUDA and CPU Torch evidence.

This reader performs no model, runtime or GPU execution.  It does not turn a
numerical comparison into kernel-placement, process-exit, game, or training
evidence.  V5 bundle-based readers and their wire format are unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat

from . import numeric_capture as nc

RAW_DOMAIN = "rz-pals-numerical-raw-run-v6/1"
REPORT_DOMAIN = "rz-pals-system-numerical-comparison-v6/1"
RAW_ATOL = 1e-4
RAW_RTOL = 1e-3
PROBABILITY_ATOL = 1e-4
RUN_FIELDS = {"schema_version", "domain", "run_id", "producer", "producer_source_component",
              "bindings", "invocation", "batch", "observed_provider", "completed_nn_inputs",
              "public_cache_hits", "physical_completion_known", "cases"}
CASE_FIELDS = {"id", "input", "input_tensors", "input_tensor_sha256", "candidate_order",
               "raw_output", "physical_completion_id"}
OUTPUT_FIELDS = {"candidate_logits", "wdl_logits", "private_latent", "divergence_logits", "task_logits"}


def _read(path, limit, seen):
    path = Path(path)
    if not path.is_absolute():
        raise ValueError("absolute numerical artifact path required")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    original = path.lstat()
    if not stat.S_ISREG(original.st_mode):
        raise ValueError("numerical artifact must be a regular non-symlink file")
    fd = os.open(path, flags)
    try:
        before = os.fstat(fd)
        identity = (before.st_dev, before.st_ino)
        if (not stat.S_ISREG(before.st_mode)
                or (before.st_dev, before.st_ino) != (original.st_dev, original.st_ino)
                or not 0 < before.st_size <= limit
                or identity in seen):
            raise ValueError("bounded distinct regular numerical artifact required")
        seen.add(identity)
        with os.fdopen(fd, "rb", closefd=False) as stream:
            raw = stream.read(limit + 1)
        after = os.fstat(fd)
        if (len(raw) != before.st_size or len(raw) > limit
                or (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
                != (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)):
            raise ValueError("numerical artifact changed during read")
    finally:
        os.close(fd)
    return raw


def _load(path, seen, total):
    raw = _read(path, nc.MAX_FILE_BYTES, seen)
    total[0] += len(raw)
    if total[0] > nc.MAX_TOTAL_BYTES:
        raise ValueError("numerical evidence exceeds total byte bound")
    return nc._strict_json(raw), hashlib.sha256(raw).hexdigest()


def _vector(value, length, name):
    if type(value) is not list or len(value) != length:
        raise ValueError("numerical output shape differs: " + name)
    if any(type(item) not in (int, float) or abs(item) > 3.4028234663852886e38
           or not math.isfinite(item) for item in value):
        raise ValueError("numerical output contains nonfinite/non-numeric value: " + name)
    return value


def _outputs(raw, role, candidates, divergences):
    if type(raw) is not dict or set(raw) != OUTPUT_FIELDS:
        raise ValueError("exact numerical output fields required")
    _vector(raw["candidate_logits"], candidates, "candidate_logits")
    _vector(raw["wdl_logits"], 3, "wdl_logits")
    _vector(raw["private_latent"], 16 * 384, "private_latent")
    if role == "critic":
        _vector(raw["divergence_logits"], divergences, "divergence_logits")
    elif raw["divergence_logits"] is not None:
        raise ValueError("divergence head belongs only to critic")
    if role == "validator":
        _vector(raw["task_logits"], 7, "task_logits")
    elif raw["task_logits"] is not None:
        raise ValueError("task head belongs only to validator")


def _softmax(values):
    if not values:
        raise ValueError("empty probability head")
    maximum = max(values)
    exps = [math.exp(value - maximum) for value in values]
    total = sum(exps)
    if not math.isfinite(total) or total <= 0:
        raise ValueError("probability normalization failed")
    return [value / total for value in exps]


def _compare_raw(left, right):
    maximum = 0.0
    for reference, native in zip(left, right):
        delta = abs(native - reference)
        maximum = max(maximum, delta)
        if delta > RAW_ATOL + RAW_RTOL * abs(reference):
            raise ValueError("numerical raw tolerance exceeded")
    return maximum


def _compare_probability(left, right):
    maximum = max(abs(a - b) for a, b in zip(_softmax(left), _softmax(right)))
    if maximum > PROBABILITY_ATOL:
        raise ValueError("numerical policy/WDL tolerance exceeded")
    return maximum


def _validate_run(run, group, bindings, producer):
    if (type(run) is not dict or set(run) != RUN_FIELDS or run["schema_version"] != 6
            or type(run["schema_version"]) is not int or run["domain"] != RAW_DOMAIN
            or run["producer"] != producer or run["bindings"] != bindings
            or run["invocation"] != "fresh" or type(run["batch"]) is not int or run["batch"] != 1
            or run["public_cache_hits"] != 0 or type(run["public_cache_hits"]) is not int
            or type(run["cases"]) is not list or len(run["cases"]) != len(group["cases"])):
        raise ValueError("numerical run identity, binding or case count differs")
    reference = producer == "independent_pytorch_fp32"
    expected_source = bindings["source_components"]["model_reference" if reference else "model_native"]
    expected_id = ("reference." if reference else "native-") + group["id"]
    if (run["run_id"] != expected_id or run["producer_source_component"] != expected_source
            or run["observed_provider"] != (None if reference else "CUDAExecutionProvider")
            or type(run["physical_completion_known"]) is not bool
            or run["physical_completion_known"] != (not reference)
            or type(run["completed_nn_inputs"]) is not int
            or run["completed_nn_inputs"] != (0 if reference else 2 * len(group["cases"]))):
        raise ValueError("numerical producer/provider/physical completion differs")


def _digest_array(value, name):
    """Decode Rust's [u8; 32] JSON without accepting bools as bytes."""
    if (type(value) is not list or len(value) != 32
            or any(type(item) is not int or not 0 <= item <= 255 for item in value)):
        raise ValueError("invalid SHA-256 bytes: " + name)
    return bytes(value).hex()


def _control_inventory(inventory, bindings):
    """Compare static inventory declarations with the selected model plan."""
    if (type(inventory) is not dict
            or inventory.get("schema") != "rovezero.pals-virtual-static-control-inventory.v1"
            or inventory.get("actual_provider_placement") != "not_observed"
            or inventory.get("cpu_allowlist") != "not_created"
            or inventory.get("artifact_kind") != "fresh"
            or inventory.get("manifest_sha256") != bindings.get("export_manifest_sha256")):
        raise ValueError("control inventory does not describe the plan's model")
    expected_graphs = bindings.get("graphs")
    graph_entries = inventory.get("graphs")
    if type(expected_graphs) is not dict or type(graph_entries) is not list:
        raise ValueError("control graph roster missing")
    inventory_graphs = {}
    for graph in graph_entries:
        if type(graph) is not dict or type(graph.get("role")) is not str or graph["role"] in inventory_graphs:
            raise ValueError("invalid or duplicate control inventory graph")
        inventory_graphs[graph["role"]] = graph.get("sha256")
    if inventory_graphs != expected_graphs:
        raise ValueError("control inventory graph identities differ from the plan")
    return expected_graphs


def _control_witness(witness, inventory, bindings, inventory_sha256):
    """Cross-check a producer's witness; do not manufacture fresh GPU evidence.

    The Rust producer is responsible for validating actual named placements and
    profile events. This reader checks the separately retained declarations,
    identities and coherent counts; it does not replay ORT's placement gate.
    """
    if (type(witness) is not dict or type(inventory) is not dict
            or witness.get("schema") != "rovezero.pals-virtual-system-cuda-placement.v1"
            or witness.get("layout") != inventory.get("layout")
            or witness.get("artifact_kind") != "fresh"
            or inventory.get("artifact_kind") != "fresh"
            or witness.get("optimization") != "disable"
            or witness.get("physical_completion") != "not_inferred"
            or type(witness.get("declared_device_id")) is not int
            or witness["declared_device_id"] < 0):
        raise ValueError("invalid system CUDA placement witness scope")
    if (_digest_array(witness.get("inventory_sha256"), "inventory") != inventory_sha256
            or _digest_array(witness.get("manifest_sha256"), "manifest") != bindings.get("export_manifest_sha256")
            or _digest_array(witness.get("ort_sha256"), "ORT") != bindings.get("runtime_library_sha256")):
        raise ValueError("control witness inventory, model or ORT identity differs")
    runtime_identity = _digest_array(witness.get("system_runtime_identity_sha256"), "system runtime")
    expected_graphs = _control_inventory(inventory, bindings)
    placements = witness.get("initialization")
    kernels = witness.get("kernels")
    if type(placements) is not list or type(kernels) is not dict or set(kernels) != set(expected_graphs):
        raise ValueError("control witness graph coverage differs")
    seen = set()
    for placement in placements:
        if type(placement) is not dict or type(placement.get("role")) is not str:
            raise ValueError("invalid graph placement")
        role = placement["role"]
        counts = [placement.get(key) for key in ("assigned_nodes", "cuda_nodes", "approved_cpu_control_nodes")]
        if (role in seen or role not in expected_graphs
                or _digest_array(placement.get("graph_sha256"), "graph") != expected_graphs[role]
                or placement.get("optimization") != "disable"
                or any(type(value) is not int or value < 0 for value in counts)
                or counts[0] == 0 or counts[1] == 0 or counts[0] != counts[1] + counts[2]
                or type(placement.get("approved_transfers")) is not list):
            raise ValueError("control witness placement identity or counts differ")
        _digest_array(placement.get("log_sha256"), "placement log")
        seen.add(role)
    if seen != set(expected_graphs):
        raise ValueError("control witness does not cover every graph")
    for role, value in kernels.items():
        if (type(value) is not dict or type(value.get("cuda_kernels")) is not int
                or value["cuda_kernels"] <= 0 or type(value.get("neural_kernels")) is not int
                or value["neural_kernels"] <= 0):
            raise ValueError("control witness lacks CUDA neural kernel observations")
        _digest_array(value.get("profile_sha256"), "kernel profile")
    if "shared_pc" in kernels:
        pc = kernels["shared_pc"]
        if any(type(pc.get(key)) is not int or pc[key] <= 0
               for key in ("proposer_private_kernels", "critic_private_kernels")):
            raise ValueError("control witness lacks one of the P/C private branches")
    libraries = bindings.get("system_cuda_libraries")
    mapped, missing = witness.get("mapped_roles"), witness.get("missing_optional_roles")
    if (type(libraries) is not list or type(mapped) is not list or type(missing) is not list
            or any(type(role) is not str for role in mapped + missing)):
        raise ValueError("system library observation roster missing")
    registered = {item["role"]: item["required_mapping"] for item in libraries}
    if (len(mapped) != len(set(mapped)) or len(missing) != len(set(missing))
            or set(mapped) & set(missing) or set(mapped) | set(missing) != set(registered)
            or any(required and role not in mapped for role, required in registered.items())
            or any(registered[role] for role in missing)):
        raise ValueError("system library observation does not match the registered roster")
    return {"inventory_sha256": inventory_sha256,
            "manifest_sha256": bindings["export_manifest_sha256"],
            "ort_sha256": bindings["runtime_library_sha256"],
            "system_runtime_identity_sha256": runtime_identity,
            "system_runtime_identity_scope": "producer_reported_not_recomputed",
            "declared_device_id": witness["declared_device_id"],
            "graph_roles": sorted(seen), "mapped_roles": mapped,
            "missing_optional_roles": missing,
            "verification_scope": "producer_witness_fields_and_cross_artifact_identity_only",
            "profiles_replayed": False, "current_mapping": "not_attested",
            "physical_completion": "not_attested", "process_exit": "not_attested"}


def _control_evidence(plan_sha256, bindings, inventory_path, inventory_sha256,
                      observation_path, observation_sha256, witness_path):
    """Read optional capture evidence without making it a numerical prerequisite."""
    for value in (inventory_sha256, observation_sha256):
        if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise ValueError("registered control evidence SHA-256 required")
    seen = set()
    inventory, actual_inventory = _load(inventory_path, seen, [0])
    observation, actual_observation = _load(observation_path, seen, [0])
    if actual_inventory != inventory_sha256 or actual_observation != observation_sha256:
        raise ValueError("control evidence bytes differ from registered SHA-256")
    _control_inventory(inventory, bindings)
    if (type(observation) is not dict
            or observation.get("schema") != "rovezero.virtual-system-cuda-placement-observation.v1"
            or observation.get("plan_sha256") != plan_sha256
            or observation.get("inventory_sha256") != inventory_sha256
            or observation.get("manifest_sha256") != bindings.get("export_manifest_sha256")):
        raise ValueError("control observation does not describe this plan and inventory")
    status = observation.get("final_status")
    if status not in ("verified", "mapping_unavailable", "unregistered_or_preexisting_mappings"):
        raise ValueError("unrecognized system placement observation status")
    result = {"observation_sha256": actual_observation, "inventory_sha256": actual_inventory,
              "producer_status": status,
              "reader_verification": "cross_artifact_fields_only" if status == "verified" else "not_available",
              "profile_scope": observation.get("profile_scope"),
              "first_registered_role_coverage_status": observation.get("first_registered_role_coverage_status"),
              "initial_partial_observation": observation.get("initial_partial_observation"),
              "initial_partial_error": observation.get("initial_partial_error"),
              "inventory_producer_source_sha256": inventory.get("producer_source_sha256"),
              "inventory_producer_provenance": "not_independently_attested",
              "same_numeric_run": "not_attested",
              "physical_completion": "not_attested", "process_exit": "not_attested",
              "current_mapping": "not_attested"}
    if status == "verified":
        if witness_path is None:
            raise ValueError("verified control observation requires its original witness")
        witness, actual_witness = _load(witness_path, seen, [0])
        if observation.get("witness_sha256") != actual_witness:
            raise ValueError("actual placement witness differs from capture observation")
        result.update(_control_witness(witness, inventory, bindings, actual_inventory))
        result["witness_sha256"] = actual_witness
    elif witness_path is not None or observation.get("witness_sha256") is not None:
        raise ValueError("unavailable mapping cannot issue a verified placement witness")
    return result


def compare(plan_path, plan_sha256, reference_dir, native_dir, *,
            control_inventory=None, control_inventory_sha256=None,
            placement_observation=None, placement_observation_sha256=None,
            placement_witness=None):
    if type(plan_sha256) is not str or re.fullmatch(r"[0-9a-f]{64}", plan_sha256) is None:
        raise ValueError("registered plan SHA-256 required")
    seen, plan_total = set(), [0]
    plan, actual_plan_sha = _load(plan_path, seen, plan_total)
    if actual_plan_sha != plan_sha256:
        raise ValueError("actual plan SHA-256 differs")
    expected_cases = nc.validate_system_case_plan(plan)
    bindings = plan["bindings"]
    reference_dir, native_dir = Path(reference_dir), Path(native_dir)
    if not reference_dir.is_absolute() or not native_dir.is_absolute():
        raise ValueError("absolute reference and native directories required")
    maxima = {"raw": 0.0, "policy": 0.0, "wdl": 0.0}
    evidence, case_count, completion_ids = [], 0, set()
    reference_total, native_total = [0], [0]
    for group in plan["groups"]:
        reference, ref_sha = _load(reference_dir / ("reference." + group["id"] + ".json"), seen, reference_total)
        native, native_sha = _load(native_dir / ("native-" + group["id"] + ".json"), seen, native_total)
        _validate_run(reference, group, bindings, "independent_pytorch_fp32")
        _validate_run(native, group, bindings, "rust_ort_native")
        evidence.append({"group": group["id"], "reference_sha256": ref_sha, "native_sha256": native_sha})
        for expected, left, right in zip(group["cases"], reference["cases"], native["cases"]):
            if (type(left) is not dict or set(left) != CASE_FIELDS or type(right) is not dict
                    or set(right) != CASE_FIELDS or left["id"] != expected["id"]
                    or right["id"] != expected["id"] or left["input"] != expected["input"]
                    or right["input"] != expected["input"]):
                raise ValueError("numerical case or exact typed input differs")
            payload = nc.python_input(expected["input"]["input"])
            tensors = nc.captured_tensors(payload)
            digest = nc.tensor_digest(tensors, payload["role"])
            # Python value equality alone treats 0.0 == -0.0 and 1 == 1.0.
            # Re-encode each producer's own typed input, then independently
            # validate/framing-hash the bytes each producer actually reported.
            left_payload = nc.python_input(left["input"]["input"])
            right_payload = nc.python_input(right["input"]["input"])
            left_prepared = nc.captured_tensors(left_payload)
            right_prepared = nc.captured_tensors(right_payload)
            if (nc.tensor_digest(left_prepared, payload["role"]) != digest
                    or nc.tensor_digest(right_prepared, payload["role"]) != digest
                    or nc.tensor_digest(left["input_tensors"], payload["role"]) != digest
                    or nc.tensor_digest(right["input_tensors"], payload["role"]) != digest
                    or left["input_tensors"] != tensors or right["input_tensors"] != tensors
                    or left["input_tensor_sha256"] != digest or right["input_tensor_sha256"] != digest
                    or left["candidate_order"] != expected["input"]["input"]["candidates"]
                    or right["candidate_order"] != left["candidate_order"]
                    or any(type(candidate) is not dict
                           or set(candidate) != {"from", "to", "promotion"}
                           or any(type(candidate[key]) is not int for key in candidate)
                           for candidate in left["candidate_order"] + right["candidate_order"])):
                raise ValueError("actual tensor bytes, digest or candidate ordering differ")
            completion = right["physical_completion_id"]
            if (left["physical_completion_id"] is not None or type(completion) is not str
                    or re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", completion) is None
                    or (native["run_id"], completion) in completion_ids):
                raise ValueError("native case completion identity absent or duplicated")
            completion_ids.add((native["run_id"], completion))
            a, b = left["raw_output"], right["raw_output"]
            _outputs(a, payload["role"], len(payload["candidates"]), len(payload["divergences"]))
            _outputs(b, payload["role"], len(payload["candidates"]), len(payload["divergences"]))
            for name in OUTPUT_FIELDS:
                if a[name] is not None:
                    maxima["raw"] = max(maxima["raw"], _compare_raw(a[name], b[name]))
            maxima["policy"] = max(maxima["policy"], _compare_probability(a["candidate_logits"], b["candidate_logits"]))
            maxima["wdl"] = max(maxima["wdl"], _compare_probability(a["wdl_logits"], b["wdl_logits"]))
            case_count += 1
    if case_count != expected_cases:
        raise ValueError("numerical plan was not fully compared")
    result = {"domain": REPORT_DOMAIN, "status": "passed", "plan_sha256": actual_plan_sha,
            "comparison": "fixed_input_fresh_b1_cpu_torch_vs_system_cuda_raw",
            "coverage_scope": "declared_plan_only",
            "observed_role_buckets": sorted({(group["role"], group["line_bucket"]) for group in plan["groups"]}),
            "cases": case_count, "groups": len(plan["groups"]), "raw_max_abs": maxima["raw"],
            "reference_bytes_read": reference_total[0], "native_bytes_read": native_total[0],
            "policy_max_abs": maxima["policy"], "wdl_max_abs": maxima["wdl"],
            "tolerances": {"raw_atol": RAW_ATOL, "raw_rtol": RAW_RTOL, "policy_wdl_atol": PROBABILITY_ATOL},
            "artifact_hashes": evidence, "kernel_placement": "unknown", "process_exit": "not_attested",
            "system_library_mapping": "unknown", "training_eligible": False,
            "v5_numerical_acceptance": False,
            "native_completion_scope": "producer_reported_per_request_not_independent_process_or_device_fence",
            "rules_legality": "not_attested", "performance": "not_measured", "strength": "not_measured"}
    control_args = (control_inventory, control_inventory_sha256,
                    placement_observation, placement_observation_sha256)
    if any(value is not None for value in control_args) or placement_witness is not None:
        try:
            if any(value is None for value in control_args):
                raise ValueError("explicit control inventory and placement observation pins required")
            result["control_evidence"] = _control_evidence(
                actual_plan_sha, bindings, control_inventory, control_inventory_sha256,
                placement_observation, placement_observation_sha256, placement_witness)
            result["control_evidence_status"] = "observed"
        except Exception as error:
            # A separate malformed or unavailable placement artifact must not
            # erase the independently completed numerical comparison. The CLI
            # reports this partial outcome with its own nonzero exit code.
            result["control_evidence_status"] = "failed"
            result["control_evidence"] = {"error_type": type(error).__name__,
                                          "error": str(error)[:2048],
                                          "current_mapping": "not_attested",
                                          "physical_completion": "not_attested",
                                          "process_exit": "not_attested"}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--reference-dir", required=True, type=Path)
    parser.add_argument("--native-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--control-inventory", type=Path)
    parser.add_argument("--control-inventory-sha256")
    parser.add_argument("--placement-observation", type=Path)
    parser.add_argument("--placement-observation-sha256")
    parser.add_argument("--placement-witness", type=Path)
    args = parser.parse_args()
    if not args.output.is_absolute():
        raise ValueError("absolute comparison output path required")
    result = compare(args.plan, args.plan_sha256, args.reference_dir, args.native_dir,
                     control_inventory=args.control_inventory,
                     control_inventory_sha256=args.control_inventory_sha256,
                     placement_observation=args.placement_observation,
                     placement_observation_sha256=args.placement_observation_sha256,
                     placement_witness=args.placement_witness)
    item = nc._publish(result, args.output, 0)
    print(json.dumps({"domain": REPORT_DOMAIN, "status": "written", "sha256": item["sha256"],
                      "bytes": item["bytes"], "numerical_status": result["status"],
                      "control_evidence_status": result.get("control_evidence_status", "not_requested")}))
    return 2 if result.get("control_evidence_status") == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
