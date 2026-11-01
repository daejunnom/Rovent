"""Opt-in actual Torch raw capture for the independent V5 numerical reader.

Importing this module never imports Torch, NumPy, ONNX or ORT. The pure case
plan certifies tensor math only; its synthetic boards/moves are not Rules,
training targets, games, strength or performance evidence. Execution must be
explicit through ``capture_reference`` with an already verified frozen model.
The Rust native producer runs independently from the same immutable plan.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import struct
import time

from .virtual_review import BUCKETS, INPUT_SCHEMA, prepare_payload, registered_encoding_sha256, validate_payload

PLAN_DOMAIN = "rz-pals-numerical-case-plan-v5/1"
G0_BOOTSTRAP_CASE_SPEC_DOMAIN = "rz-pals-g0-numerical-cases/2"
RAW_DOMAIN = "rz-pals-numerical-raw-run-v5/1"
TENSOR_DOMAIN = "rz-pals-captured-tensor-bytes-v5/1"
MAX_FILE_BYTES = 4 * 1024 * 1024
MAX_MANIFEST_BYTES = 256 * 1024
MAX_TOTAL_BYTES = 64 * 1024 * 1024
MAX_CASES = 256
_ROLES = ("proposer", "critic", "validator")
_BINDINGS = {"model_profile", "model_schema", "model_semantics", "encoding_schema",
             "model_semantic_sha256", "encoding_sha256", "model_epoch", "weights_sha256",
             "export_manifest_sha256", "graphs", "runtime_library_sha256", "cuda_libraries",
             "cuda_manifest_sha256", "backend", "precision", "tf32", "source_components"}
_G0_IMMUTABLE_BINDINGS = _BINDINGS - {"model_epoch", "weights_sha256", "export_manifest_sha256", "graphs"}
_SOURCE = {"model_native", "model_encoding", "nn_runtime", "model_reference", "model_export"}
_INTEGER = {"anchor_board", "record_anchor", "record_line_tokens", "candidates", "query_anchor",
            "query_line_tokens", "record_relations"}
_BOOL = {"anchor_mask", "record_line_mask", "record_mask", "candidate_mask", "divergence_mask", "query_line_mask"}
def source_pins():
    """Read the same five bounded actual components the native binary embeds.

    The V5 manifest, native producer and reference all use the single registered
    logical-path/size/raw-SHA inventory and its common component framing. This
    registration never imports a model or certifies that the component ran.
    """
    from .v5_source_registry import source_pins as registered_source_pins
    result = registered_source_pins(_SOURCE)
    if set(result) != _SOURCE or not all(_sha(value) for value in result.values()):
        raise ValueError("complete registered numerical source roster required")
    return result


def _sha(value):
    return type(value) is str and re.fullmatch("[0-9a-f]{64}", value) is not None


def _id(value):
    return type(value) is str and re.fullmatch("[A-Za-z0-9_.-]{1,128}", value) is not None


def validate_bindings(value):
    if type(value) is not dict or set(value) != _BINDINGS:
        raise ValueError("exact V5 numerical binding fields required")
    if (value["model_profile"] != "virtual_review_v1" or value["model_schema"] != "rovezero.pals-model.v3"
            or value["model_semantics"] != "rovezero.pals-model-semantics.v3"
            or value["encoding_schema"] != "rovezero.pals-virtual-review.v1"
            or value["precision"] != "fp32" or value["tf32"] is not False
            or value["backend"] not in ("ort_cpu", "ort_cuda")
            or type(value["model_epoch"]) is not int or not 0 <= value["model_epoch"] < 2 ** 64):
        raise ValueError("registered V3 Fresh FP32 TF32-off bindings required")
    for name in ("model_semantic_sha256", "encoding_sha256", "weights_sha256", "export_manifest_sha256", "runtime_library_sha256"):
        if not _sha(value[name]):
            raise ValueError("numeric binding digest: " + name)
    if value["encoding_sha256"] != registered_encoding_sha256():
        raise ValueError("actual registered V3 tensor encoding digest differs")
    if type(value["graphs"]) is not dict or set(value["graphs"]) not in ({"public", "shared_pc"}, {"public", "shared_pc", "validator"}) or not all(_sha(x) for x in value["graphs"].values()):
        raise ValueError("exact Fresh graph byte pins required")
    if type(value["source_components"]) is not dict or set(value["source_components"]) != _SOURCE or not all(_id(k) and _sha(v) for k, v in value["source_components"].items()):
        raise ValueError("exact registered source component pins required")
    cuda = value["cuda_libraries"]
    if type(cuda) is not list or not all(_sha(x) for x in cuda) or len(set(cuda)) != len(cuda):
        raise ValueError("bounded unique CUDA library identity roster required")
    if value["backend"] == "ort_cpu":
        if cuda or value["cuda_manifest_sha256"] is not None:
            raise ValueError("CPU binding cannot claim CUDA bundle")
    elif not cuda or not _sha(value["cuda_manifest_sha256"]):
        raise ValueError("CUDA binding requires registered native bundle")
    return copy.deepcopy(value)


def _move_dict(token):
    return dict(zip(("from", "to", "promotion"), token))


def rust_input(payload):
    validate_payload(payload)
    result = copy.deepcopy(payload)
    result["candidates"] = [_move_dict(x) for x in payload["candidates"]]
    for record in result["records"]:
        record["line"] = [_move_dict(x) for x in record["line"]]
    for name in ("prefix", "proposal", "counter"):
        result["query"][name] = [_move_dict(x) for x in result["query"][name]]
    return result


def python_input(typed):
    if type(typed) is not dict:
        raise ValueError("typed Rust input object required")
    result = copy.deepcopy(typed)
    def tokens(values):
        if type(values) is not list:
            raise ValueError("bounded move sequence required")
        out = []
        for value in values:
            if type(value) is not dict or set(value) != {"from", "to", "promotion"}:
                raise ValueError("exact MoveToken object required")
            out.append([value["from"], value["to"], value["promotion"]])
        return out
    result["candidates"] = tokens(result["candidates"])
    for record in result["records"]:
        record["line"] = tokens(record["line"])
    for name in ("prefix", "proposal", "counter"):
        result["query"][name] = tokens(result["query"][name])
    validate_payload(result)
    return result


def _shape_flat(data):
    if type(data) is not list or not data:
        return [], [data]
    shapes, children = zip(*(_shape_flat(x) for x in data))
    if any(x != shapes[0] for x in shapes):
        raise ValueError("ragged tensor capture refused")
    return [len(data), *shapes[0]], [x for child in children for x in child]


def captured_tensors(payload):
    """Capture exact prepared source tensor bytes, not a semantic cache key."""
    result = []
    for name, data in sorted(prepare_payload(payload)["feed"].items()):
        shape, values = _shape_flat(data)
        dtype = "i64_le" if name in _INTEGER else "bool_u8" if name in _BOOL else "f32_le"
        raw = b"".join(struct.pack("<q" if dtype == "i64_le" else "<?" if dtype == "bool_u8" else "<f", v) for v in values)
        result.append({"name": name, "dtype": dtype, "shape": shape, "bytes": list(raw)})
    tensor_digest(result, payload["role"])
    return result


def tensor_digest(tensors, role):
    """Byte-for-byte framing of pals_numerical_tensor_digest_v5 in Rust."""
    if role not in _ROLES or type(tensors) is not list or not 1 <= len(tensors) <= 32:
        raise ValueError("bounded tensor roster required")
    digest = hashlib.sha256()
    def frame(raw):
        digest.update(struct.pack("<Q", len(raw))); digest.update(raw)
    frame(TENSOR_DOMAIN.encode())
    frame(role.encode())
    previous = None
    for tensor in tensors:
        if type(tensor) is not dict or set(tensor) != {"name", "dtype", "shape", "bytes"}:
            raise ValueError("exact tensor capture fields required")
        name, dtype, shape, values = (tensor[k] for k in ("name", "dtype", "shape", "bytes"))
        if not _id(name) or previous is not None and previous >= name:
            raise ValueError("unique lexically sorted tensor names required")
        previous = name
        if type(shape) is not list or not 1 <= len(shape) <= 4 or any(type(x) is not int or not 0 < x < 2 ** 64 for x in shape):
            raise ValueError("bounded positive tensor shape required")
        if dtype not in ("f32_le", "i64_le", "bool_u8") or type(values) is not list or len(values) > MAX_FILE_BYTES or any(type(x) is not int or not 0 <= x <= 255 for x in values):
            raise ValueError("bounded raw dtype bytes required")
        width = {"f32_le": 4, "i64_le": 8, "bool_u8": 1}[dtype]
        if math.prod(shape) * width != len(values):
            raise ValueError("tensor dtype/shape/bytes disagree")
        raw = bytes(values)
        if dtype == "bool_u8" and any(x > 1 for x in values) or dtype == "f32_le" and any(not math.isfinite(v[0]) for v in struct.iter_unpack("<f", raw)):
            raise ValueError("nonfinite tensor or invalid BOOL mask")
        frame(name.encode()); frame({"f32_le": "f32", "i64_le": "i64", "bool_u8": "bool"}[dtype].encode())
        digest.update(struct.pack("<Q", len(shape)))
        for dim in shape:
            digest.update(struct.pack("<Q", dim))
        frame(raw)
    return digest.hexdigest()


def make_case_plan(bindings):
    """Register deterministic Fresh B1 math coverage, never model execution."""
    bindings = validate_bindings(bindings)
    roles = _ROLES if "validator" in bindings["graphs"] else _ROLES[:2]
    groups = _make_case_groups(roles)
    result = {"schema_version": 5, "domain": PLAN_DOMAIN, "bindings": bindings, "invocation": "fresh", "batch": 1, "groups": groups}
    validate_case_plan(result)
    return result


def make_g0_bootstrap_case_spec(immutable_bindings):
    """Predeclare G0 math cases without identifiers for future generated assets.

    The first original initializer has not produced weights or graphs yet. This
    independent /2 spec contains only already registered model/runtime/source
    semantics and deterministic Collection cases. The parent binds actual epoch
    1 and generated file identities after its original preparation child ends;
    this factory never manufactures those identities or execution authority.
    """
    value = immutable_bindings
    if type(value) is not dict or set(value) != _G0_IMMUTABLE_BINDINGS:
        raise ValueError("exact immutable G0 bootstrap bindings without generated assets required")
    if (value["model_profile"] != "virtual_review_v1" or value["model_schema"] != "rovezero.pals-model.v3"
            or value["model_semantics"] != "rovezero.pals-model-semantics.v3"
            or value["encoding_schema"] != "rovezero.pals-virtual-review.v1"
            or value["precision"] != "fp32" or value["tf32"] is not False
            or value["backend"] not in ("ort_cpu", "ort_cuda")):
        raise ValueError("registered G0 V3 Fresh FP32 TF32-off immutable semantics required")
    for name in ("model_semantic_sha256", "encoding_sha256", "runtime_library_sha256"):
        if not _sha(value[name]):
            raise ValueError("G0 immutable binding digest: " + name)
    if value["encoding_sha256"] != registered_encoding_sha256():
        raise ValueError("actual registered G0 V3 tensor encoding digest differs")
    sources = value["source_components"]
    if (type(sources) is not dict or set(sources) != _SOURCE
            or not all(_id(name) and _sha(pin) for name, pin in sources.items())):
        raise ValueError("exact registered G0 numerical source components required")
    cuda = value["cuda_libraries"]
    if type(cuda) is not list or not all(_sha(pin) for pin in cuda) or len(set(cuda)) != len(cuda):
        raise ValueError("unique CUDA library identities required for G0 immutable binding")
    if value["backend"] == "ort_cpu":
        if cuda or value["cuda_manifest_sha256"] is not None:
            raise ValueError("CPU G0 bootstrap cannot claim a CUDA bundle")
    elif not cuda or not _sha(value["cuda_manifest_sha256"]):
        raise ValueError("CUDA G0 bootstrap requires its already registered native bundle")
    return {"schema": G0_BOOTSTRAP_CASE_SPEC_DOMAIN, "immutable_bindings": copy.deepcopy(value),
            "groups": _make_case_groups(_ROLES)}


def _make_case_groups(roles):
    """Math cases with native plain-name IDs; retain dotted historical parsing."""
    groups = []
    for role in roles:
        for bucket in BUCKETS:
            cases = []
            for anchors in range(1, 5):
                for white in (0, 1):
                    line = [[(ply * 7) % 64, (ply * 7 + 8) % 64, ply % 5] for ply in range(bucket)]
                    metadata = [0.] * 16; metadata[0] = float(white)
                    payload = {"schema": INPUT_SCHEMA, "role": role,
                        "anchors": [{"board": [(square + a) % 13 for square in range(64)], "metadata": list(metadata)} for a in range(anchors)],
                        "records": [{"kind": "counterexample", "perspective": 1 if white else -1, "completed_iteration": True, "rules_terminal": False,
                                     "line": line[:-1], "anchor": anchors - 1, "parent": None, "supersedes": None}],
                        "query": {"purpose": "task_policy" if role == "validator" else "reply_policy", "perspective": 1 if white else -1,
                                  "prefix": [], "proposal": line, "counter": [], "counter_present": False, "anchors": [0, anchors - 1, 0]},
                        "candidates": [[48, 56, p] for p in (1, 2, 3, 4)], "divergences": [[0.] * 8] if role == "critic" else [], "token_budget": 512}
                    cases.append({"id": f"{role}_L{bucket}_A{anchors}_W{white}", "input": {"model_input_kind": "virtual_review", "input": rust_input(payload)}})
            if bucket == 8:
                for name in ("empty", "order_reversed"):
                    extra = copy.deepcopy(cases[0]); extra["id"] = role + "_" + name
                    if name == "empty":
                        extra["input"]["input"]["records"] = []
                    else:
                        extra["input"]["input"]["candidates"].reverse()
                    cases.append(extra)
            groups.append({"id": f"{role}_L{bucket}", "role": role, "line_bucket": bucket, "cases": cases})
    return groups


def validate_case_plan(plan):
    if type(plan) is not dict or set(plan) != {"schema_version", "domain", "bindings", "invocation", "batch", "groups"} or type(plan["schema_version"]) is not int or plan["schema_version"] != 5 or plan["domain"] != PLAN_DOMAIN or plan["invocation"] != "fresh" or type(plan["batch"]) is not int or plan["batch"] != 1:
        raise ValueError("registered Fresh B1 case plan required")
    validate_bindings(plan["bindings"])
    if type(plan["groups"]) is not list or not 1 <= len(plan["groups"]) <= 32:
        raise ValueError("bounded case groups required")
    seen, group_ids, total = set(), set(), 0
    for group in plan["groups"]:
        if type(group) is not dict or set(group) != {"id", "role", "line_bucket", "cases"} or not _id(group["id"]) or not _id("reference." + group["id"]) or group["role"] not in _ROLES or type(group["line_bucket"]) is not int or group["line_bucket"] not in BUCKETS or type(group["cases"]) is not list or not 1 <= len(group["cases"]) <= 10:
            raise ValueError("bounded role/bucket group required")
        if group["id"] in group_ids:
            raise ValueError("duplicate numeric group/output identity")
        group_ids.add(group["id"])
        if group["role"] == "validator" and "validator" not in plan["bindings"]["graphs"]:
            raise ValueError("V-free assets cannot include a V case")
        for case in group["cases"]:
            if type(case) is not dict or set(case) != {"id", "input"} or not _id(case["id"]) or case["id"] in seen:
                raise ValueError("unique actual numeric case identity required")
            seen.add(case["id"]); total += 1
            value = case["input"]
            if type(value) is not dict or set(value) != {"model_input_kind", "input"} or value["model_input_kind"] != "virtual_review":
                raise ValueError("new V3 input codec required")
            payload = python_input(value["input"])
            if not payload["candidates"] or payload["anchors"][0]["metadata"][0] not in (0., 1.):
                raise ValueError("numeric case requires candidates and exact root side-to-move")
            admission = prepare_payload(payload)["admission"]
            if payload["role"] != group["role"] or admission["line_bucket"] != group["line_bucket"]:
                raise ValueError("typed case role/physical bucket differs")
    if total > MAX_CASES:
        raise ValueError("numeric case bound exceeded")
    return total


def _publish(value, destination, total):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    if not raw or len(raw) > MAX_FILE_BYTES or total + len(raw) > MAX_TOTAL_BYTES:
        raise ValueError("raw numerical evidence output bound exceeded")
    path = Path(destination).expanduser().resolve()
    from .artifacts import output_directory, publish_noreplace
    output_directory(path.parent)
    partial = path.with_suffix(path.suffix + ".partial")
    if path.exists():
        raise FileExistsError("numerical evidence already registered")
    with partial.open("xb") as stream:
        if stream.write(raw) != len(raw):
            raise OSError("incomplete numerical evidence write")
        stream.flush(); os.fsync(stream.fileno())
    publish_noreplace(partial, path)
    return {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def write_case_plan(bindings, destination):
    """Pure exclusive case-plan publication, no inference/library loading."""
    return _publish(make_case_plan(bindings), destination, 0)


def _regular_read(path, limit):
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= limit:
        raise ValueError("bounded regular artifact required")
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    if len(raw) != info.st_size or len(raw) > limit:
        raise ValueError("artifact size changed during read")
    return raw


def _strict_json(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate numerical manifest key")
            result[key] = value
        return result
    def constant(_):
        raise ValueError("nonfinite numerical manifest constant")
    return json.loads(raw, object_pairs_hook=unique, parse_constant=constant)


def _digest_asset(path, deadline):
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= 2 * 1024 ** 3:
        raise ValueError("bounded regular graph/checkpoint required")
    digest = hashlib.sha256(); size = 0
    with path.open("rb") as stream:
        while True:
            if time.monotonic() >= deadline:
                raise TimeoutError("numeric preflight window exhausted")
            block = stream.read(1024 * 1024)
            if not block:
                break
            size += len(block)
            if size > info.st_size:
                raise ValueError("artifact expanded during numeric preflight")
            digest.update(block)
    if size != info.st_size:
        raise ValueError("artifact size changed during numeric preflight")
    return digest.hexdigest()


def _verify_model_checkpoint(model, checkpoint_path, expected_sha256, deadline, torch, *, require_pt_suffix=True):
    """Same private file descriptor for pin/decode/pin; never mutate caller.

    A full generation checkpoint can include V and optimizer state. Such state
    is not resumed here. Deployment compares exactly the P/C view, permits only
    the explicitly excluded V-private keys and drops decoded state immediately.
    This verifies mathematical weight identity, not training eligibility.
    """
    path = Path(checkpoint_path).expanduser()
    if (require_pt_suffix and path.suffix != ".pt") or path.is_symlink():
        raise ValueError("owned regular .pt checkpoint required")
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= 2 * 1024 ** 3:
        raise ValueError("checkpoint admission byte bound")
    with path.open("rb") as stream:
        opened = os.fstat(stream.fileno())
        if (opened.st_dev, opened.st_ino, opened.st_size) != (info.st_dev, info.st_ino, info.st_size):
            raise ValueError("checkpoint descriptor identity changed")
        def pinned_hash():
            stream.seek(0); digest = hashlib.sha256(); size = 0
            while True:
                if time.monotonic() >= deadline:
                    raise TimeoutError("checkpoint preflight window exhausted")
                block = stream.read(1024 * 1024)
                if not block:
                    break
                size += len(block)
                if size > info.st_size:
                    raise ValueError("checkpoint expanded during capture")
                digest.update(block)
            if size != info.st_size or digest.hexdigest() != expected_sha256:
                raise ValueError("actual checkpoint bytes differ")
        pinned_hash(); stream.seek(0)
        checkpoint = torch.load(stream, map_location="cpu", weights_only=True)
        pinned_hash()
    if type(checkpoint) is not dict or checkpoint.get("config") != model.config.to_dict():
        raise ValueError("actual checkpoint config differs from reference owner")
    if checkpoint.get("schema") == "rovezero.pals-model-semantics.v3":
        incoming = checkpoint.get("state_dict")
    elif checkpoint.get("schema") == "rz-pals-actual-generation-checkpoint/1" and checkpoint.get("eligibility") == "actual_training_pending_acceptance":
        incoming = checkpoint.get("model")
    else:
        raise ValueError("unregistered mathematical checkpoint codec")
    if not isinstance(incoming, dict):
        raise ValueError("checkpoint has no actual model state")
    current = model.state_dict()
    retained = {name for name in incoming if not name.startswith("experts.validator.")} if "validator" not in model.experts else set(incoming)
    if set(current) != retained:
        raise ValueError("model membership differs from actual checkpoint")
    for name, observed in current.items():
        if time.monotonic() >= deadline:
            raise TimeoutError("checkpoint model proof window exhausted")
        expected = incoming[name]
        if (not torch.is_tensor(expected) or expected.device.type != "cpu" or expected.dtype != observed.dtype
                or expected.shape != observed.shape or expected.dtype != torch.float32
                or not torch.isfinite(expected).all().item()
                or not torch.equal(observed.detach().contiguous().reshape(-1).view(torch.uint8), expected.contiguous().reshape(-1).view(torch.uint8))):
            raise ValueError("actual checkpoint model bits differ: " + name)
    del current, incoming, checkpoint


def capture_reference(model, metadata, export_directory, plan, output_root, *, checkpoint_path, expected_reference_source_sha256, max_seconds=600, execute=False):
    """Explicit CPU Torch inference, with bounded independent raw receipts.

    The caller must supply the registered current reference source pin. This
    captures actual model math only. It never starts ORT, runs an optimizer,
    consumes native outputs, writes a ``passed`` receipt or claims CUDA fences.
    """
    if execute is not True:
        raise ValueError("actual reference capture requires explicit execute=True")
    started = time.monotonic()
    validate_case_plan(plan)
    bindings = plan["bindings"]
    actual_sources = source_pins()
    if (not _sha(expected_reference_source_sha256) or bindings["source_components"]["model_reference"] != expected_reference_source_sha256
            or expected_reference_source_sha256 != actual_sources["model_reference"]
            or bindings["source_components"] != actual_sources):
        raise ValueError("current registered reference source differs")
    if type(max_seconds) is not int or not 1 <= max_seconds <= 3600:
        raise ValueError("finite reference capture window required")
    deadline = started + max_seconds
    expected_roles = set(_ROLES) if "validator" in bindings["graphs"] else set(_ROLES[:2])
    if (model.training or not model.config.virtual_review or model.config.profile != bindings["model_profile"]
            or metadata.get("checkpoint_sha256") != bindings["weights_sha256"]
            or metadata.get("config") != model.config.to_dict() or set(model.experts) != expected_roles):
        raise ValueError("frozen verified V3 model/checkpoint binding required")
    from .artifacts import validate_virtual_export_manifest
    directory = Path(export_directory).expanduser().resolve()
    manifest_path = directory / "export.json"
    manifest_raw = _regular_read(manifest_path, MAX_MANIFEST_BYTES)
    if hashlib.sha256(manifest_raw).hexdigest() != bindings["export_manifest_sha256"]:
        raise ValueError("actual export manifest changed")
    manifest = _strict_json(manifest_raw)
    validate_virtual_export_manifest(manifest, metadata)
    if {x["role"]: x["sha256"] for x in manifest["graphs"]} != bindings["graphs"]:
        raise ValueError("exact Fresh graph roster differs")
    for graph in manifest["graphs"]:
        if _digest_asset(directory / graph["file"], deadline) != graph["sha256"]:
            raise ValueError("actual graph bytes changed")
    import torch
    from .model import PalsModel, VirtualReviewTensorInput
    if type(model) is not PalsModel:
        raise ValueError("actual registered PalsModel reference owner required")
    if torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32:
        raise ValueError("TF32 must be explicitly off before capture")
    if not 1 <= torch.get_num_threads() <= 2 or not 1 <= torch.get_num_interop_threads() <= 2:
        raise ValueError("reference caller must configure the registered CPU-two thread limit")
    if any(p.device.type != "cpu" or p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError("independent CPU FP32 reference required")
    _verify_model_checkpoint(model, checkpoint_path, bindings["weights_sha256"], deadline, torch)
    artifacts = []; total = 0
    with torch.no_grad():
        for group in plan["groups"]:
            cases = []
            for case in group["cases"]:
                if time.monotonic() - started >= max_seconds:
                    raise TimeoutError("reference capture window exhausted; preserve prior files")
                payload = python_input(case["input"]["input"])
                data = VirtualReviewTensorInput.from_payload(payload); data.validate(model.config)
                memory = model.public_encoder(*data.public_args(model.config))
                raw = model.role_graph(payload["role"])(*data.role_args(memory, model.config))
                output = {"candidate_logits": raw[0][0, :len(payload["candidates"])].tolist(), "wdl_logits": raw[1][0].tolist(),
                          "private_latent": raw[2][0].flatten().tolist(), "divergence_logits": raw[3][0, :len(payload["divergences"])].tolist() if payload["role"] == "critic" else None,
                          "task_logits": raw[3][0].tolist() if payload["role"] == "validator" else None}
                tensors = captured_tensors(payload)
                cases.append({"id": case["id"], "input": copy.deepcopy(case["input"]), "input_tensors": tensors,
                              "input_tensor_sha256": tensor_digest(tensors, payload["role"]), "candidate_order": copy.deepcopy(case["input"]["input"]["candidates"]),
                              "raw_output": output, "physical_completion_id": None})
                # Public tensors and graph-local latent are owned only by this
                # call, rather than retained for the full case roster.
                del raw, memory, data
            receipt = {"schema_version": 5, "domain": RAW_DOMAIN, "run_id": "reference." + group["id"],
                       "producer": "independent_pytorch_fp32", "producer_source_component": bindings["source_components"]["model_reference"], "bindings": bindings,
                       "invocation": "fresh", "batch": 1, "observed_provider": None, "completed_nn_inputs": 0, "public_cache_hits": 0,
                       "physical_completion_known": False, "cases": cases}
            item = _publish(receipt, Path(output_root) / ("reference." + group["id"] + ".json"), total)
            total += item["bytes"]; artifacts.append(item)
            del cases, receipt
    return {"domain": RAW_DOMAIN, "producer": "independent_pytorch_fp32", "artifacts": artifacts, "bytes": total,
            "cases": validate_case_plan(plan), "native_status": "not_run", "rules_certified": False,
            "reference_software": {"torch_version": str(torch.__version__), "device": "cpu", "dtype": "fp32",
                                   "intra_threads": torch.get_num_threads(), "inter_threads": torch.get_num_interop_threads(),
                                   "tf32": False}}
