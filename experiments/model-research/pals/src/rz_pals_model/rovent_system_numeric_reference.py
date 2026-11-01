"""Compare original Rovent logical inputs and raw outputs with CPU Torch.

This is a numerical observation, not a Rules, tensor-byte, provider-placement,
training, strength or performance admission.  The producer's V3 input-lock
canonical identity and the input-lock file's raw digest are different fields.
The old V5 numerical formats and their CUDA-bundle semantics are unchanged.
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
import sys
import time

from .numeric_capture import _verify_model_checkpoint, python_input

DOMAIN = "rz-rovent-system-numeric-observation/1"
RESULT_DOMAIN = "rz-rovent-system-numeric-reference/1"
RAW_ATOL = 1e-4
RAW_RTOL = 1e-3
PROBABILITY_ATOL = 1e-4
MAX_OBSERVATION_BYTES = 64 * 1024 * 1024
MAX_JSON_BYTES = 4 * 1024 * 1024
MAX_CASES = 256


class UnsupportedReferenceConfiguration(ValueError):
    """This comparator supports a shared P/C checkpoint only."""


def _sha(value):
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _read_json(path, maximum):
    path = Path(path)
    if not path.is_absolute():
        raise ValueError("absolute input path required")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    fd = os.open(path, flags)
    try:
        with os.fdopen(fd, "rb", closefd=False) as stream:
            before = os.fstat(fd)
            if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= maximum:
                raise ValueError("bounded regular JSON input required")
            data = stream.read(maximum + 1)
            after = os.fstat(fd)
        if len(data) != before.st_size or (before.st_dev, before.st_ino, before.st_size,
                before.st_mtime_ns) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
            raise ValueError("JSON input changed while reading")
        def unique(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("duplicate JSON object key")
                result[key] = value
            return result
        value = json.loads(data, object_pairs_hook=unique,
                           parse_constant=lambda value: (_ for _ in ()).throw(ValueError("nonfinite JSON")))
        return value, hashlib.sha256(data).hexdigest()
    finally:
        os.close(fd)


def _asset_path(root, artifact):
    if type(artifact) is not dict or not _sha(artifact.get("sha256")):
        raise ValueError("input lock artifact identity required")
    relative = Path(artifact.get("path", ""))
    if relative.is_absolute() or not relative.parts or any(p in ("", ".", "..") for p in relative.parts):
        raise ValueError("snapshot-relative artifact path required")
    target = (root / relative).resolve(strict=True)
    if not target.is_relative_to(root):
        raise ValueError("artifact escapes snapshot root")
    return target


def _digest(path, maximum, deadline):
    digest = hashlib.sha256()
    size = 0
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    with os.fdopen(os.open(path, flags), "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= maximum:
            raise ValueError("bounded regular artifact required")
        while True:
            if time.monotonic() >= deadline:
                raise TimeoutError("reference verification window elapsed")
            block = stream.read(1024 * 1024)
            if not block:
                break
            size += len(block)
            if size > maximum:
                raise ValueError("artifact exceeds size bound")
            digest.update(block)
        after = os.fstat(stream.fileno())
    if size != before.st_size or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError("artifact changed while hashing")
    return digest.hexdigest(), size


def _values(value, label):
    if type(value) is not list:
        raise ValueError(label + " must be a list")
    result = []
    for number in value:
        if type(number) not in (int, float) or not math.isfinite(number):
            raise ValueError(label + " contains nonfinite or nonnumeric output")
        result.append(float(number))
    return result


def _raw_comparison(actual, reference, label):
    actual = _values(actual, label)
    reference = _values(reference, label + " reference")
    if len(actual) != len(reference):
        raise ValueError(label + " output shape differs")
    differences = [abs(a - b) for a, b in zip(actual, reference)]
    failures = sum(diff > RAW_ATOL + RAW_RTOL * abs(b)
                   for diff, b in zip(differences, reference))
    return {"count": len(actual), "max_absolute": max(differences, default=0.0),
            "outside_tolerance": failures, "passed": failures == 0}


def _softmax(values):
    if not values:
        return []
    pivot = max(values)
    exps = [math.exp(x - pivot) for x in values]
    total = sum(exps)
    return [x / total for x in exps]


def _probability_comparison(actual, reference, label):
    actual = _softmax(_values(actual, label))
    reference = _softmax(_values(reference, label + " reference"))
    if len(actual) != len(reference):
        raise ValueError(label + " probability shape differs")
    maximum = max((abs(a - b) for a, b in zip(actual, reference)), default=0.0)
    return {"count": len(actual), "max_absolute": maximum, "passed": maximum <= PROBABILITY_ATOL}


def _critic_divergence_coverage(captures, results):
    """Summarize only comparisons with an actual nonempty critic input head."""
    indices = [index for index, case in enumerate(captures)
               if isinstance(case, dict) and case.get("role") == "critic"
               and isinstance(case.get("input"), dict)
               and isinstance(case["input"].get("divergences"), list)
               and len(case["input"]["divergences"]) > 0]
    if not indices:
        return {"status": "not_observed", "input_count": 0, "compared_count": 0}
    by_index = {row.get("index"): row for row in results if isinstance(row, dict)}
    comparisons = []
    for index in indices:
        row = by_index.get(index, {})
        field = row.get("fields", {}).get("divergence_logits")
        expected_count = len(captures[index]["input"]["divergences"])
        if (not isinstance(field, dict) or field.get("count") != expected_count
                or type(field.get("passed")) is not bool):
            continue
        comparisons.append(field["passed"])
    if any(value is False for value in comparisons):
        status = "mismatch"
    elif len(comparisons) != len(indices):
        status = "incomplete"
    else:
        status = "match"
    return {"status": status, "input_count": len(indices), "compared_count": len(comparisons)}


def _write_new(path, value):
    path = Path(path)
    if not path.is_absolute():
        raise ValueError("absolute output path required")
    raw = (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def compare(observation_path, lock_path, canonical_manifest_path, snapshot_root, export_relative, max_seconds):
    if type(max_seconds) is not int or not 1 <= max_seconds <= 3600:
        raise ValueError("bounded positive reference window required")
    deadline = time.monotonic() + max_seconds
    root = Path(snapshot_root)
    if not root.is_absolute():
        raise ValueError("absolute snapshot root required")
    root = root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("snapshot root must be a directory")
    observation, observation_sha = _read_json(observation_path, MAX_OBSERVATION_BYTES)
    lock, lock_raw_sha = _read_json(lock_path, MAX_JSON_BYTES)
    if (type(observation) is not dict or observation.get("domain") != DOMAIN
            or observation.get("status") not in ("complete", "partial", "unavailable", "unknown")
            or type(observation.get("unavailable")) is not bool
            or type(observation.get("session_physical_confirmed")) is not bool
            or type(observation.get("process_epoch")) is not int or observation["process_epoch"] <= 0
            or type(observation.get("captures")) is not list
            or len(observation["captures"]) > MAX_CASES):
        raise ValueError("invalid bounded original observation")
    if not Path(lock_path).resolve(strict=True).is_relative_to(root):
        raise ValueError("original input lock must belong to the snapshot")
    if not observation["captures"]:
        return {"domain": RESULT_DOMAIN, "status": "partial", "reason": "no_completed_capture",
                "comparison_status": "incomplete", "coverage_status": "partial",
                "observation_raw_sha256": observation_sha, "input_lock_raw_sha256": lock_raw_sha,
                "input_lock_canonical_sha256_declared": observation.get("input_lock_sha256"),
                "canonical_digest_recomputed": False, "source_observation_status": observation["status"],
                "observation_claims_session_physical_confirmed": observation["session_physical_confirmed"],
                "physical_completion": "not_independently_verified", "capture_origin": "not_independently_verified",
                "model_reference_executed": False, "tensor_bytes_equivalence": "not_observed",
                "critic_nonempty_divergence": _critic_divergence_coverage([], []),
                "gpu_placement": "not_inferred", "performance_claim": False, "cases": []}
    if (type(lock) is not dict or not _sha(observation.get("input_lock_sha256"))
            or lock.get("canonical_sha256") != observation["input_lock_sha256"]):
        raise ValueError("original input-lock canonical identity differs")
    if not Path(canonical_manifest_path).resolve(strict=True).is_relative_to(root):
        raise ValueError("canonical manifest must belong to the snapshot")
    canonical, canonical_sha = _read_json(canonical_manifest_path, MAX_JSON_BYTES)
    if (canonical_sha != lock["canonical_sha256"] or type(canonical) is not list or len(canonical) != 2
            or canonical[0] != "rz-rovent-execution-canonical/3"
            or json.dumps(canonical[1], sort_keys=True, allow_nan=False)
                != json.dumps(lock["manifest"], sort_keys=True, allow_nan=False)):
        raise ValueError("canonical bytes or manifest differ from input lock")
    engine = lock["manifest"]["engine"]
    pc = engine["pc"]
    roles = ("proposer", "critic")
    weights = [pc[role]["weights"] for role in roles]
    if any(item["sha256"] != weights[0]["sha256"] or item["bytes"] != weights[0]["bytes"] for item in weights):
        raise UnsupportedReferenceConfiguration("this reference supports one shared P/C checkpoint identity")
    checkpoint = _asset_path(root, weights[0])
    checkpoint_sha, checkpoint_bytes = _digest(checkpoint, 1024 * 1024 * 1024, deadline)
    if checkpoint_sha != weights[0]["sha256"] or checkpoint_bytes != weights[0]["bytes"]:
        raise ValueError("actual checkpoint differs from input lock")
    for item in weights[1:]:
        other_path = _asset_path(root, item)
        if other_path != checkpoint:
            other_sha, other_bytes = _digest(other_path, 1024 * 1024 * 1024, deadline)
            if other_sha != checkpoint_sha or other_bytes != checkpoint_bytes:
                raise ValueError("second role checkpoint differs from input lock")
    metadata_path = None
    expected_metadata = None
    for role in roles:
        provenance = pc[role]["provenance"]
        meta_path = _asset_path(root, provenance)
        meta_sha, meta_bytes = _digest(meta_path, MAX_JSON_BYTES, deadline)
        if meta_sha != provenance["sha256"] or meta_bytes != provenance["bytes"]:
            raise ValueError("checkpoint metadata differs from input lock")
        if expected_metadata is not None and expected_metadata != (meta_sha, meta_bytes):
            raise UnsupportedReferenceConfiguration("this reference supports one shared P/C metadata identity")
        if metadata_path is None:
            metadata_path = meta_path
            expected_metadata = (meta_sha, meta_bytes)
    entries = [x for x in lock["manifest"]["provenance"] if x.get("path") == export_relative]
    if len(entries) != 1:
        raise ValueError("one original export manifest required")
    export_path = _asset_path(root, entries[0])
    manifest, manifest_sha = _read_json(export_path, MAX_JSON_BYTES)
    if manifest_sha != entries[0]["sha256"] or export_path.stat().st_size != entries[0]["bytes"]:
        raise ValueError("actual export manifest differs from input lock")
    graphs = {}
    for graph in manifest["graphs"]:
        filename = graph["file"]
        if (type(filename) is not str or Path(filename).name != filename
                or graph["role"] in graphs):
            raise ValueError("unique graph role and basename required")
        graph_path = (Path(entries[0]["path"]).parent / filename).as_posix()
        matches = [x for x in lock["manifest"]["provenance"] if x.get("path") == graph_path]
        if len(matches) != 1 or matches[0]["sha256"] != graph["sha256"]:
            raise ValueError("graph provenance differs from export manifest")
        actual_sha, actual_bytes = _digest(_asset_path(root, matches[0]), 1024 * 1024 * 1024, deadline)
        if actual_sha != graph["sha256"] or actual_bytes != matches[0]["bytes"]:
            raise ValueError("actual graph differs from original provenance")
        graphs[graph["role"]] = actual_sha

    # Imports and real model execution occur only after CLI --execute and all
    # bounded declaration/file checks above.  CPU threads are explicitly set.
    import torch
    from .artifacts import load_checkpoint, validate_virtual_export_manifest
    from .model import VirtualReviewTensorInput
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, metadata = load_checkpoint(checkpoint, metadata_path=metadata_path)
    if metadata.get("checkpoint_sha256") != checkpoint_sha:
        raise ValueError("loaded checkpoint identity differs")
    metadata_after_sha, metadata_after_bytes = _digest(metadata_path, MAX_JSON_BYTES, deadline)
    if (metadata_after_sha, metadata_after_bytes) != expected_metadata:
        raise ValueError("checkpoint metadata changed while loading")
    _verify_model_checkpoint(model, checkpoint, checkpoint_sha, deadline, torch, require_pt_suffix=False)
    validate_virtual_export_manifest(manifest, metadata)
    if not model.config.virtual_review or any(p.device.type != "cpu" or p.dtype != torch.float32
                                              for p in model.parameters()):
        raise ValueError("CPU FP32 virtual-review reference required")
    results = []
    def compare_case(case):
            if (type(case) is not dict or case.get("role") not in roles
                    or case.get("model_epoch_sha256") != checkpoint_sha
                    or case.get("physical_fresh") is not True
                    or type(case.get("completed_nn_inputs")) is not int
                    or case["completed_nn_inputs"] <= 0
                    or type(case.get("ticket")) is not dict or type(case.get("authority")) is not dict
                    or case["authority"].get("epoch") != observation["process_epoch"]):
                raise ValueError("original physical Fresh case identity differs")
            payload = python_input(case["input"])
            if payload["role"] != case["role"]:
                raise ValueError("case role and input role differ")
            data = VirtualReviewTensorInput.from_payload(payload)
            data.validate(model.config)
            memory = model.public_encoder(*data.public_args(model.config))
            raw = model.role_graph(payload["role"])(*data.role_args(memory, model.config))
            expected = {"candidate_logits": raw[0][0, :len(payload["candidates"])].tolist(),
                        "wdl_logits": raw[1][0].tolist(),
                        "private_latent": raw[2][0].flatten().tolist(),
                        "divergence_logits": raw[3][0, :len(payload["divergences"])].tolist()
                            if payload["role"] == "critic" else None,
                        "task_logits": None}
            actual = case["raw_output"]
            if type(actual) is not dict or set(actual) != set(expected):
                raise ValueError("raw output fields differ")
            fields = {}
            for name, reference in expected.items():
                if reference is None:
                    if actual[name] is not None:
                        raise ValueError("unexpected " + name)
                else:
                    fields[name] = _raw_comparison(actual[name], reference, name)
            fields["candidate_policy"] = _probability_comparison(actual["candidate_logits"],
                                                                   expected["candidate_logits"], "candidate")
            fields["wdl_probability"] = _probability_comparison(actual["wdl_logits"], expected["wdl_logits"], "WDL")
            passed = all(field["passed"] for field in fields.values())
            return {"role": case["role"], "ticket": case["ticket"], "authority": case["authority"],
                    "capture_kind": case.get("capture_kind"),
                    "logical_input_sha256": hashlib.sha256(json.dumps(case["input"], sort_keys=True,
                        separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest(),
                    "fields": fields, "status": "match" if passed else "mismatch", "passed": passed}
    with torch.no_grad():
        for index, case in enumerate(observation["captures"]):
            if time.monotonic() >= deadline:
                results.extend({"index": remaining, "status": "not_run_deadline", "passed": False}
                               for remaining in range(index, len(observation["captures"])))
                break
            try:
                results.append({"index": index, **compare_case(case)})
            except Exception as error:
                results.append({"index": index, "status": "reference_case_error", "passed": False,
                                "role": case.get("role") if isinstance(case, dict) else None,
                                "ticket": case.get("ticket") if isinstance(case, dict) else None,
                                "error_type": type(error).__name__, "error": str(error)[:2048]})
                if isinstance(error, (MemoryError, OSError, RuntimeError)):
                    results.extend({"index": remaining, "status": "not_run_shared_failure", "passed": False}
                                   for remaining in range(index + 1, len(observation["captures"])))
                    break
    all_passed = bool(results) and all(case["passed"] for case in results)
    any_mismatch = any(case["status"] == "mismatch" for case in results)
    role_coverage = {case.get("role") for case in results if case.get("status") in ("match", "mismatch")}
    comparison_status = ("mismatch" if any_mismatch else "match" if all_passed else "incomplete")
    coverage_status = "complete" if role_coverage == set(roles) else "partial"
    full = (observation["status"] == "complete" and not observation["unavailable"]
            and observation["session_physical_confirmed"] and coverage_status == "complete")
    return {"domain": RESULT_DOMAIN, "status": ("logical_raw_match" if all_passed and full else
            "logical_raw_mismatch" if any_mismatch else "partial"),
            "comparison_status": comparison_status, "coverage_status": coverage_status,
            "observation_raw_sha256": observation_sha, "input_lock_raw_sha256": lock_raw_sha,
            "input_lock_canonical_sha256_declared": observation["input_lock_sha256"],
            "canonical_digest_recomputed": True, "typed_manifest_validation_reperformed": False,
            "artifact_path_observation": "resolved_beneath_snapshot_intermediate_symlinks_not_rejected",
            "checkpoint_sha256": checkpoint_sha, "model_checkpoint_bits_verified": True,
            "export_manifest_sha256": manifest_sha, "export_artifact": export_relative,
            "export_selection": "caller_selected_lock_provenance_not_independently_bound_to_live_native_arguments",
            "graph_sha256_by_role": graphs,
            "model_reference": "cpu_torch_fp32",
            "tf32": False, "intra_threads": torch.get_num_threads(), "inter_threads": torch.get_num_interop_threads(),
            "raw_atol": RAW_ATOL, "raw_rtol": RAW_RTOL, "policy_wdl_atol": PROBABILITY_ATOL,
            "source_observation_status": observation["status"],
            "observation_claims_session_physical_confirmed": observation["session_physical_confirmed"],
            "physical_completion": "not_independently_verified",
            "capture_origin": "not_independently_verified",
            "role_coverage": sorted(role_coverage),
            "critic_nonempty_divergence": _critic_divergence_coverage(observation["captures"], results),
            "observation_claims_critic_nonempty_divergence": observation.get("critic_nonempty_divergence_observed"),
            "logical_input_comparison": True, "tensor_bytes_equivalence": "not_observed",
            "gpu_placement": "not_inferred", "rules_legality": "not_inferred", "numerical_acceptance": False,
            "performance_claim": False,
            "cases": results}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--observation", required=True)
    parser.add_argument("--input-lock", required=True)
    parser.add_argument("--canonical-manifest", required=True)
    parser.add_argument("--snapshot-root", required=True)
    parser.add_argument("--export-artifact", required=True,
                        help="original snapshot-relative export manifest path")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-seconds", type=int, default=600)
    args = parser.parse_args(argv)
    if not args.execute:
        parser.error("explicit --execute is required to load or run the CPU reference")
    output = Path(args.output)
    if not output.is_absolute() or output.exists():
        parser.error("an unused absolute output path is required")
    parent = output.parent.resolve(strict=True)
    snapshot = Path(args.snapshot_root).resolve(strict=True)
    if parent == snapshot or parent.is_relative_to(snapshot):
        parser.error("the reference receipt must be outside the original snapshot")
    try:
        result = compare(args.observation, args.input_lock, args.canonical_manifest,
                         args.snapshot_root, args.export_artifact, args.max_seconds)
    except Exception as error:
        result = {"domain": RESULT_DOMAIN,
                  "status": ("unsupported_reference_configuration" if isinstance(error, UnsupportedReferenceConfiguration)
                             else "reference_error"), "error_type": type(error).__name__,
                  "error": str(error)[:2048], "tensor_bytes_equivalence": "not_observed",
                  "gpu_placement": "not_inferred", "performance_claim": False,
                  "inputs": {"observation": args.observation, "input_lock": args.input_lock,
                             "canonical_manifest": args.canonical_manifest, "snapshot_root": args.snapshot_root,
                             "export_artifact": args.export_artifact},
                  "input_digest_on_error": "not_observed"}
        _write_new(args.output, result)
        print(json.dumps({"domain": RESULT_DOMAIN, "status": result["status"]}))
        return 1
    _write_new(args.output, result)
    print(json.dumps({"domain": RESULT_DOMAIN, "status": result["status"],
                      "comparison_status": result["comparison_status"],
                      "coverage_status": result["coverage_status"], "cases": len(result["cases"])}))
    return 0 if result["comparison_status"] == "match" else 1


if __name__ == "__main__":
    sys.exit(main())
