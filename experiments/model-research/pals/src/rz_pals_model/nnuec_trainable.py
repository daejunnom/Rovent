"""Trainable Torch reference for the explicit Rust NNUEC FP32 V1/V2 wires.

This is a differentiable correctness reference, not an optimized inference
backend. B/C exact accumulation intentionally synchronizes through CPU bigint
arithmetic, including for CUDA tensors. Its backward is the mathematical sum
gradient with FP32 rounding treated as a straight-through operation; it is not
the almost-everywhere-zero derivative of a quantizer. Other encoder/projection
and head operations retain the scalar FP32 order of the independent reference.

No optimizer, random initialization, collection or training loop is provided.
Constructors require an explicitly validated checkpoint. Caller-owned training
must serialize mutations, forward and export at an optimizer-update boundary.
V1 and V2 have separate constructors/writers and never convert automatically.
"""

from dataclasses import dataclass
from collections import OrderedDict
import hashlib
import json
import math
import os
from pathlib import Path
import stat

import torch
from torch import nn

from . import nnuec_reference as reference
from . import nnuec_v2_reference as relation_reference
from .artifacts import publish_noreplace


def _v1_shapes(config):
    b, e, c, local, h = (config.board_width, config.line_width, config.context_width,
                        config.local_width, config.hidden_width)
    return dict(zip(reference.PARAMETER_FIELDS, (
        (800, b), (b,), (800, e), (e,), (64, e), (64, e), (5, e),
        (config.max_plies, e), (e,), (config.causal_window, e), (e,),
        (c, e), (c,), (local, e), (local,), (h, 2 * b + c + local),
        (h,), (11, h), (11,),
    ), strict=True))


def _v2_shapes(config):
    e, c = config.line_width, config.context_width
    return dict(zip(relation_reference.PARAMETER_FIELDS, ((4, e), (2, e), (2, e), (2, e), (c, e)), strict=True))


def _decode_metadata(state, model_schema, model_semantics):
    reference._fields(state, ("schema", "model_schema", "model_semantics", "config", "output_scale", "training"),
                      "NNUEC Torch metadata")
    config = reference.NnuecConfig.from_mapping(state["config"])
    scale = state["output_scale"]
    if (state["schema"] != "rz-nnuec-torch-metadata/1" or state["model_schema"] != model_schema
            or state["model_semantics"] != model_semantics or type(scale) is not float
            or not 0.0 < scale <= 10000.0 or reference.f32(scale) != scale):
        raise ValueError("NNUEC tensor metadata schema/semantics/output_scale mismatch")
    return config, scale, reference._training(state["training"])


def _copy_tensor_state(state, shapes, device):
    """Preflight every source tensor before allocating any new owner tensor."""
    if type(state) not in (dict, OrderedDict) or set(state) != set(shapes):
        raise ValueError("NNUEC tensor-state namespace/membership mismatch")
    destination = torch.device(device)
    if destination.type not in ("cpu", "cuda"):
        raise ValueError("NNUEC tensor-state owner device must be CPU or CUDA")
    for name, shape in shapes.items():
        value = state[name]
        if (not isinstance(value, torch.Tensor) or tuple(value.shape) != shape or value.dtype != torch.float32
                or value.layout != torch.strided or not value.is_contiguous() or value.device.type not in ("cpu", "cuda")):
            raise ValueError(f"NNUEC tensor-state {name}: shape/dtype/layout/device mismatch")
        if not bool(torch.isfinite(value).all().item()) or bool((torch.abs(value) > 1000.0).any().item()):
            raise ValueError(f"NNUEC tensor-state {name}: nonfinite/excessive parameter")
    try:
        versions = tuple((name, id(state[name]), state[name]._version) for name in shapes)
    except RuntimeError as error:
        raise ValueError("NNUEC tensor-state requires version-tracked source tensors") from error
    # No JSON/list intermediate, random initializer or source-storage reuse.
    owned = {name: state[name].detach().to(device=destination, dtype=torch.float32,
                                          copy=True, memory_format=torch.contiguous_format) for name in shapes}
    if set(state) != set(shapes) or tuple((name, id(state[name]), state[name]._version) for name in shapes) != versions:
        raise ValueError("NNUEC tensor-state changed while copying; quiescent source owner required")
    return owned


def _owned_v1(cls, config, scale, training, tensors):
    result = cls.__new__(cls)
    nn.Module.__init__(result)
    result.config, result.output_scale, result._training_declaration = config, scale, training
    for name in reference.PARAMETER_FIELDS:
        result.register_parameter(name, nn.Parameter(tensors[name]))
    result._validate_parameters()
    return result


class _ExactSum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, operands, bits, max_operand, max_result, strict_result):
        if (operands.dtype != torch.float32 or operands.layout != torch.strided
                or operands.ndim != 2 or operands.shape[0] < 1 or operands.shape[1] < 1
                or operands.device.type not in ("cpu", "cuda")):
            raise ValueError("exact FP32 sum requires a nonempty CPU/CUDA [terms,width] tensor")
        # This explicit host fallback is part of the reference capability.
        rows = operands.detach().to(device="cpu", dtype=torch.float32).contiguous().tolist()
        if any(not math.isfinite(value) or abs(value) > max_operand for row in rows for value in row):
            raise ValueError("exact FP32 sum operand outside admitted magnitude")
        values = [reference._exact_sum_fp32((row[column] for row in rows), bits)
                  for column in range(operands.shape[1])]
        if any(abs(value) >= max_result if strict_result else abs(value) > max_result for value in values):
            raise ValueError("exact FP32 sum result outside admitted magnitude")
        ctx.terms = operands.shape[0]
        return operands.new_tensor(values, dtype=torch.float32)

    @staticmethod
    def backward(ctx, gradient):
        # A declared rounding-through surrogate, not a numerical derivative of
        # discrete rounding. Each input contribution owns one identical term.
        return gradient.unsqueeze(0).expand(ctx.terms, -1), None, None, None, None


def exact_sum_fp32(operands, *, bits, max_operand, max_result, strict_result=False):
    """Exact forward/rounding-through sum backward; host synchronization required."""
    if (not isinstance(operands, torch.Tensor) or type(bits) is not int or bits not in (192, 256)
            or type(strict_result) is not bool or type(max_operand) not in (int, float)
            or type(max_result) not in (int, float) or not 0.0 < max_operand < float("inf")
            or not 0.0 < max_result < float("inf")):
        raise ValueError("explicit signed capacity and finite positive bounds required")
    return _ExactSum.apply(operands, bits, max_operand, max_result, strict_result)


def _relu(values):
    # Rust's declared max(x, +0) also canonicalizes an exact negative zero.
    return torch.where(values > 0.0, values, torch.zeros_like(values))


def _dense(weights, values, bias=None):
    if weights.ndim != 2 or values.ndim != 1 or weights.shape[1] != values.shape[0]:
        raise ValueError("row-major NNUEC dense shape mismatch")
    output = weights.new_zeros((weights.shape[0],)) if bias is None else bias
    # Eager multiply and add are distinct operations. torch.compile/autocast
    # are outside this reference's admitted callable capability.
    for column in range(values.shape[0]):
        product = torch.mul(weights[:, column], values[column])
        output = torch.add(output, product)
    return output


def _sparse(weights, bias, features, *, exact=False):
    if exact:
        index = torch.tensor(features, dtype=torch.int64, device=weights.device)
        selected = torch.index_select(weights, 0, index)
        terms = torch.cat((bias.unsqueeze(0), selected), dim=0)
        return exact_sum_fp32(terms, bits=192, max_operand=1000.0, max_result=float(1 << 20))
    result = bias
    for feature in features:
        result = torch.add(result, weights[feature])
    return result


def _line(model, line):
    width = model.config.line_width
    if not line.moves:
        return model.token_bias.new_zeros((width,))
    anchor = _sparse(model.anchor_weights, model.anchor_bias, line.anchor_features)
    tokens = []
    encoded = model.token_bias.new_zeros((width,))
    for ply, move in enumerate(line.moves):
        orientation = 56 if line.perspective == "black" else 0
        token = torch.add(model.token_bias, anchor)
        token = torch.add(token, model.from_weights[move.from_square ^ orientation])
        token = torch.add(token, model.to_weights[move.to_square ^ orientation])
        token = torch.add(token, model.promotion_weights[move.promotion])
        token = torch.add(token, model.ply_weights[ply])
        tokens.append(token)
        value = model.causal_bias
        for lag in range(min(model.config.causal_window, ply + 1)):
            product = torch.mul(tokens[ply - lag], model.causal_weights[lag])
            value = torch.add(value, product)
        encoded = torch.add(encoded, _relu(value))
    return encoded


def _finish(model, board, context, local, side, *, schema, semantics, context_sha256=None):
    order = 0 if side == "white" else 1
    values = torch.cat((board[order], board[1 - order], context, local))
    values = torch.clamp(values, min=0.0, max=1.0)
    hidden = _relu(_dense(model.hidden_weights, values, model.hidden_bias))
    raw = _dense(model.output_weights, hidden, model.output_bias)
    logits = raw[1:4]
    maximum = torch.maximum(torch.maximum(logits[0], logits[1]), logits[2])
    exponentials = torch.exp(torch.sub(logits, maximum))
    denominator = torch.add(torch.add(exponentials[0], exponentials[1]), exponentials[2])
    wdl = torch.div(exponentials, denominator)
    value = torch.mul(raw[0], model.output_scale)
    if any(not bool(torch.isfinite(tensor).all().item()) for tensor in (hidden, raw, value, wdl)):
        raise ValueError("NNUEC trainable reference produced nonfinite output")
    return TrainableOutput(schema, semantics, board, context, local, hidden, raw, value, logits, wdl, context_sha256)


@dataclass(frozen=True)
class TrainableOutput:
    schema: str
    semantics: str
    board_accumulators: torch.Tensor
    context_accumulator: torch.Tensor
    local_accumulator: torch.Tensor
    hidden: torch.Tensor
    raw_outputs: torch.Tensor
    value: torch.Tensor
    wdl_logits: torch.Tensor
    wdl: torch.Tensor
    context_sha256: str | None = None

    def numeric_mapping(self):
        """Detach a diagnostic result; this is not an exported frozen model."""
        result = {"schema": self.schema, "semantics": self.semantics}
        for name in ("board_accumulators", "context_accumulator", "local_accumulator", "hidden",
                     "raw_outputs", "value", "wdl_logits", "wdl"):
            result[name] = getattr(self, name).detach().to(device="cpu").tolist()
        if self.context_sha256 is not None:
            result["context_sha256"] = self.context_sha256
        return result


class _TrainableContract(nn.Module):
    @staticmethod
    def capabilities():
        return {"precision": "float32", "devices": ["cpu", "cuda"], "torch_compile": False,
                "autocast": False, "exact_sum_forward": "cpu_bigint_checked_single_rounding",
                "exact_sum_backward": "rounding_through_mathematical_sum_surrogate",
                "host_synchronization": True, "optimized_backend": False,
                "cuda_numeric_equivalence": "requires_independent_target_device_check",
                "training_loop_present": False, "optimizer_present": False}

    def parameter_spec(self):
        return {name: {"shape": tuple(shape), "dtype": "float32"} for name, shape in self._expected_shapes().items()}

    def metadata_state(self):
        """Required companion to state_dict; scalars are not optimizer groups."""
        owner = self._metadata_owner()
        return {"schema": "rz-nnuec-torch-metadata/1", "model_schema": self.model_schema,
                "model_semantics": self.model_semantics, "config": owner.config.to_mapping(),
                "output_scale": owner.output_scale, "training": dict(owner._training_declaration)}

    def validate_metadata_state(self, state):
        owner = self._metadata_owner()
        candidate_config, scale, training = _decode_metadata(state, self.model_schema, self.model_semantics)
        if candidate_config != owner.config or scale != owner.output_scale:
            raise ValueError("NNUEC resume metadata cannot change model/schema/config/output_scale")
        return {**state, "config": candidate_config.to_mapping(), "training": dict(training)}

    def restore_metadata_state(self, state):
        """Preflight first; caller couples this with tensor/RNG atomic apply."""
        candidate = self.validate_metadata_state(state)
        self._metadata_owner()._training_declaration = reference._training(candidate["training"])

    def _validate_parameters(self):
        self.validate_metadata_state(self.metadata_state())
        parameters = dict(self.named_parameters())
        if set(parameters) != set(self._expected_shapes()):
            raise ValueError("NNUEC named parameter membership changed")
        device = None
        for name, shape in self._expected_shapes().items():
            value = parameters[name]
            if (tuple(value.shape) != shape or value.dtype != torch.float32 or value.layout != torch.strided
                    or not value.is_contiguous() or value.device.type not in ("cpu", "cuda")):
                raise ValueError(f"NNUEC {name}: shape/dtype/layout/device mismatch")
            if device is None:
                device = value.device
            if value.device != device:
                raise ValueError("NNUEC parameters must share one device")
            if not bool(torch.isfinite(value).all().item()) or bool((torch.abs(value) > 1000.0).any().item()):
                raise ValueError(f"NNUEC {name}: nonfinite or excessive parameter")
        if torch.is_autocast_enabled("cpu") or torch.is_autocast_enabled("cuda"):
            raise ValueError("NNUEC trainable reference does not support autocast")

    def _flat_parameters(self):
        self._validate_parameters()
        parameters = tuple(self.named_parameters())
        versions = tuple((id(value), value._version) for _, value in parameters)
        flat = {name: value.detach().to(device="cpu").contiguous().reshape(-1).tolist() for name, value in parameters}
        if tuple((id(value), value._version) for _, value in self.named_parameters()) != versions:
            raise ValueError("NNUEC parameters changed during export; quiescent owner required")
        return flat

    def write_checkpoint(self, path, *, training=None):
        mapping = self.export_checkpoint(training=training)
        checked = (reference.NnuecCheckpoint.from_mapping(mapping) if mapping["schema"] == reference.SCHEMA
                   else relation_reference.NnuecV2Checkpoint.from_mapping(mapping))
        raw = (json.dumps(mapping, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
        if len(raw) > reference.MAX_JSON_BYTES:
            raise ValueError("NNUEC exported checkpoint exceeds reader byte bound")
        destination = Path(path).expanduser().absolute()
        if destination.suffix != ".json" or destination.parent.resolve(strict=True) != destination.parent:
            raise ValueError("NNUEC export requires a plain existing absolute JSON parent")
        repository = next((parent for parent in Path(__file__).resolve().parents if (parent / ".git").exists()), None)
        if repository is not None and (destination == repository or repository in destination.parents):
            raise ValueError("NNUEC generated checkpoints must remain outside source")
        partial = destination.with_name(destination.name + ".partial")
        if destination.exists() or partial.exists():
            raise FileExistsError("NNUEC checkpoint or preserved partial already exists")
        # Exclusive same-directory publication reuses the established no-replace
        # primitive. Failure preserves the owned partial or published evidence.
        with partial.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        publish_noreplace(partial, destination)
        with destination.open("rb") as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
                raise ValueError("NNUEC published checkpoint must be a single-link regular file")
            observed = stream.read(reference.MAX_JSON_BYTES + 1)
            after = os.fstat(stream.fileno())
        identity = lambda details: (details.st_dev, details.st_ino, details.st_size, details.st_mtime_ns)
        if identity(before) != identity(after) or identity(after) != identity(destination.lstat()) or observed != raw:
            raise ValueError("NNUEC published checkpoint identity changed")
        return {"schema": mapping["schema"], "semantics": mapping["semantics"],
                "sha256": hashlib.sha256(observed).hexdigest(), "bytes": len(observed),
                "weights_sha256": checked.weights_sha256}


class NnuecTrainableV1(_TrainableContract):
    model_schema = reference.SCHEMA
    model_semantics = reference.SEMANTICS

    def __init__(self, checkpoint, *, device="cpu"):
        super().__init__()
        if type(checkpoint) is not reference.NnuecCheckpoint:
            raise ValueError("explicit validated V1 checkpoint required")
        self.config = checkpoint.config
        self.output_scale = checkpoint.output_scale
        self._training_declaration = checkpoint.training
        for (name, shape), values in zip(_v1_shapes(self.config).items(), checkpoint.parameters, strict=True):
            tensor = torch.tensor(values, dtype=torch.float32, device=device).reshape(shape)
            self.register_parameter(name, nn.Parameter(tensor))
        self._validate_parameters()

    @classmethod
    def from_checkpoint(cls, checkpoint, *, device="cpu"):
        return cls(reference.NnuecCheckpoint.from_mapping(checkpoint), device=device)

    @classmethod
    def from_tensor_state(cls, metadata, state, *, device="cpu"):
        """Own exact validated Tensor copies; caller restores freeze flags separately."""
        config, scale, training = _decode_metadata(metadata, cls.model_schema, cls.model_semantics)
        owned = _copy_tensor_state(state, _v1_shapes(config), device)
        if _decode_metadata(metadata, cls.model_schema, cls.model_semantics) != (config, scale, training):
            raise ValueError("NNUEC tensor metadata changed while copying")
        return _owned_v1(cls, config, scale, training, owned)

    def _expected_shapes(self):
        return _v1_shapes(self.config)

    def _metadata_owner(self):
        return self

    def forward(self, inputs):
        self._validate_parameters()
        checked = (reference.ReferenceInput.from_mapping(inputs, self.config) if type(inputs) is dict
                   else reference.ReferenceInput.from_mapping(inputs.to_mapping(), self.config))
        board = torch.stack(tuple(_sparse(self.board_weights, self.board_bias, features, exact=True)
                                  for features in checked.board_features))
        contributions = [_dense(self.context_projection, _line(self, line)) for line in checked.context_lines]
        terms = torch.stack((self.context_bias, *contributions))
        context = exact_sum_fp32(terms, bits=256, max_operand=reference.MAX_CONTEXT_CONTRIBUTION,
                                 max_result=reference.MAX_CONTEXT_MAGNITUDE)
        local = _dense(self.local_projection, _line(self, checked.local_line), self.local_bias)
        return _finish(self, board, context, local, checked.side_to_move,
                       schema=reference.OUTPUT_SCHEMA, semantics=reference.SEMANTICS)

    def export_checkpoint(self, *, training=None):
        metadata = self.metadata_state()
        flat = self._flat_parameters()
        if self.metadata_state() != metadata:
            raise ValueError("NNUEC metadata changed during export")
        mapping = {"schema": reference.SCHEMA, "semantics": reference.SEMANTICS,
                   "config": self.config.to_mapping(), "training": dict(self._training_declaration) if training is None else training,
                   "parameters": flat, "output_scale": self.output_scale}
        return reference.NnuecCheckpoint.from_mapping(mapping).to_mapping()


class NnuecTrainableV2(_TrainableContract):
    model_schema = relation_reference.SCHEMA
    model_semantics = relation_reference.SEMANTICS

    def __init__(self, checkpoint, *, device="cpu"):
        super().__init__()
        if type(checkpoint) is not relation_reference.NnuecV2Checkpoint:
            raise ValueError("explicit validated V2 checkpoint required; no V1 conversion")
        self.base = NnuecTrainableV1(checkpoint.base, device=device)
        for (name, shape), values in zip(_v2_shapes(self.base.config).items(), checkpoint.parameters, strict=True):
            self.register_parameter(name, nn.Parameter(torch.tensor(values, dtype=torch.float32, device=device).reshape(shape)))
        self._validate_parameters()

    @classmethod
    def from_checkpoint(cls, checkpoint, *, device="cpu"):
        return cls(relation_reference.NnuecV2Checkpoint.from_mapping(checkpoint), device=device)

    @classmethod
    def from_tensor_state(cls, metadata, state, *, device="cpu"):
        config, scale, training = _decode_metadata(metadata, cls.model_schema, cls.model_semantics)
        shapes = {**{"base." + name: shape for name, shape in _v1_shapes(config).items()}, **_v2_shapes(config)}
        owned = _copy_tensor_state(state, shapes, device)
        if _decode_metadata(metadata, cls.model_schema, cls.model_semantics) != (config, scale, training):
            raise ValueError("NNUEC V2 tensor metadata changed while copying")
        result = cls.__new__(cls)
        nn.Module.__init__(result)
        result.base = _owned_v1(NnuecTrainableV1, config, scale, training,
                                {name: owned["base." + name] for name in reference.PARAMETER_FIELDS})
        for name in relation_reference.PARAMETER_FIELDS:
            result.register_parameter(name, nn.Parameter(owned[name]))
        result._validate_parameters()
        return result

    def _expected_shapes(self):
        return {**{"base." + name: shape for name, shape in _v1_shapes(self.base.config).items()},
                **_v2_shapes(self.base.config)}

    def _metadata_owner(self):
        return self.base

    def forward(self, inputs):
        self._validate_parameters()
        checked = (relation_reference.ReferenceInput.from_mapping(inputs, self.base.config) if type(inputs) is dict
                   else relation_reference.ReferenceInput.from_mapping(inputs.to_mapping(), self.base.config))
        board = torch.stack(tuple(_sparse(self.base.board_weights, self.base.board_bias, features, exact=True)
                                  for features in checked.board_features))
        representations, contributions = {}, []
        for record in checked.context_lines:
            typed = torch.add(_line(self.base, record.line), self.kind_embeddings[record.kind])
            representations[record.id] = typed
            contributions.append(_dense(self.base.context_projection, typed))
        for edge in checked.context_edges:
            parent = torch.mul(representations[edge.source], self.pair_parent[edge.kind])
            child = torch.mul(representations[edge.target], self.pair_child[edge.kind])
            pair = _relu(torch.add(torch.add(self.pair_bias[edge.kind], parent), child))
            contributions.append(_dense(self.pair_projection, pair))
        context = exact_sum_fp32(torch.stack((self.base.context_bias, *contributions)), bits=256,
                                 max_operand=relation_reference.MAX_CONTEXT_OPERAND,
                                 max_result=relation_reference.MAX_CONTEXT_MAGNITUDE, strict_result=True)
        local = _dense(self.base.local_projection, _line(self.base, checked.local_line), self.base.local_bias)
        return _finish(self.base, board, context, local, checked.side_to_move,
                       schema=relation_reference.OUTPUT_SCHEMA, semantics=relation_reference.SEMANTICS,
                       context_sha256=checked.context_sha256)

    def export_checkpoint(self, *, training=None):
        metadata = self.metadata_state()
        flat = self._flat_parameters()
        if self.metadata_state() != metadata:
            raise ValueError("NNUEC metadata changed during export")
        original = {"schema": reference.SCHEMA, "semantics": reference.SEMANTICS,
                    "config": self.base.config.to_mapping(),
                    "training": dict(self.base._training_declaration) if training is None else training,
                    "parameters": {name: flat["base." + name] for name in reference.PARAMETER_FIELDS},
                    "output_scale": self.base.output_scale}
        mapping = {"schema": relation_reference.SCHEMA, "semantics": relation_reference.SEMANTICS, "base": original,
                   "parameters": {name: flat[name] for name in relation_reference.PARAMETER_FIELDS}}
        return relation_reference.NnuecV2Checkpoint.from_mapping(mapping).to_mapping()
