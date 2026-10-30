"""Independent shadow-only NNUEC target-bundle/risk/receiver head reference.

The supplied V2 model computes its real line and internal-pair contributions.
This head reads B_stm, B_other, global C, local L and the selected target's C
contribution without the global bias. Lookup IDs and provenance seals are not
numeric inputs. Inputs are clipped to 0..1 in the declared order.

This pure numeric boundary cannot establish exact Rules/history anchors or a
live runtime lease. It checks selected numeric inputs and edge closure only.
Exact bundle completeness remains with the actual Rust BoundFactory: equal
board-800 features must never be used to merge different historical anchors.
Seven raw shadow hints grant neither selection authority nor training labels.
"""

from dataclasses import dataclass
import argparse
import hashlib
import json
import math
import struct
import sys

from . import nnuec_reference as numeric
from . import nnuec_v2_reference as model_reference


SCHEMA = "rz-nnuec-utility-fp32-v1"
SEMANTICS = "nnuec-v2-targetbundle-risk-route-shadow-fp32-v1"
INPUT_SCHEMA = "rz-nnuec-utility-input/1"
OUTPUT_SCHEMA = "rz-nnuec-utility-output/1"
MAX_CHECKPOINT_BYTES = 8 << 20
PARAMETER_FIELDS = ("hidden_weights", "hidden_bias", "output_weights", "output_bias")
OUTPUT_NAMES = ("retain", "drop", "revisit", "missing_risk", "route_p", "route_c", "route_r")
REFERENCE_SCOPE = "numeric_shadow_only_no_exact_rules_bundle_or_runtime_lease_authority"


@dataclass(frozen=True)
class UtilityCheckpoint:
    base_weights_sha256: str
    hidden_width: int
    training: tuple
    parameters: tuple
    input_width: int

    def __post_init__(self):
        digest = self.base_weights_sha256
        if type(digest) is not str or len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError("utility requires a lowercase V2 base weights SHA-256")
        numeric._integer(self.hidden_width, 1, 64, "utility hidden_width")
        numeric._integer(self.input_width, 5, 1280, "utility input_width")
        if tuple(numeric._training(dict(self.training))) != self.training:
            raise ValueError("utility training declaration must be immutable and canonical")
        if type(self.parameters) is not tuple or len(self.parameters) != 4:
            raise ValueError("utility requires exactly four immutable parameter arrays")
        lengths = (self.hidden_width * self.input_width, self.hidden_width, 7 * self.hidden_width, 7)
        for name, values, length in zip(PARAMETER_FIELDS, self.parameters, lengths, strict=True):
            if type(values) is not tuple or len(values) != length:
                raise ValueError(f"utility {name}: expected {length} FP32 parameters")
            if any(type(value) is not float or not math.isfinite(value) or abs(value) > 1000.0
                   or numeric.f32(value) != value for value in values):
                raise ValueError(f"utility {name}: bounded already-rounded FP32 parameters required")

    @classmethod
    def from_mapping(cls, value, base_checkpoint):
        numeric._fields(value, ("schema", "semantics", "base_weights_sha256", "hidden_width", "training", "parameters"), "utility checkpoint")
        if value["schema"] != SCHEMA or value["semantics"] != SEMANTICS:
            raise ValueError("unregistered utility head schema or semantics")
        if type(base_checkpoint) is not model_reference.NnuecV2Checkpoint:
            raise ValueError("utility requires its explicit validated V2 base")
        if value["base_weights_sha256"] != base_checkpoint.weights_sha256:
            raise ValueError("utility weights do not belong to these frozen V2 base weights")
        hidden = numeric._integer(value["hidden_width"], 1, 64, "utility hidden_width")
        config = base_checkpoint.base.config
        width = 2 * config.board_width + 2 * config.context_width + config.local_width
        supplied = numeric._fields(value["parameters"], PARAMETER_FIELDS, "utility parameters")
        lengths = (hidden * width, hidden, 7 * hidden, 7)
        arrays = []
        for name, length in zip(PARAMETER_FIELDS, lengths, strict=True):
            values = supplied[name]
            if type(values) is not list or len(values) != length:
                raise ValueError(f"utility {name}: expected {length} FP32 parameters")
            arrays.append(tuple(numeric.f32(item) for item in values))
        return cls(value["base_weights_sha256"], hidden, numeric._training(value["training"]), tuple(arrays), width)

    def to_mapping(self):
        return {"schema": SCHEMA, "semantics": SEMANTICS, "base_weights_sha256": self.base_weights_sha256,
                "hidden_width": self.hidden_width, "training": dict(self.training),
                "parameters": {name: list(values) for name, values in zip(PARAMETER_FIELDS, self.parameters, strict=True)}}

    def validate_base(self, checkpoint):
        if type(checkpoint) is not model_reference.NnuecV2Checkpoint or checkpoint.weights_sha256 != self.base_weights_sha256:
            raise ValueError("utility frozen V2 base identity mismatch")
        config = checkpoint.base.config
        if self.input_width != 2 * config.board_width + 2 * config.context_width + config.local_width:
            raise ValueError("utility explicit input dimensions mismatch")

    @property
    def weights_sha256(self):
        digest = hashlib.sha256()
        digest.update((SCHEMA + "\0" + SEMANTICS + "\0" + self.base_weights_sha256 + "\0").encode("ascii"))
        digest.update(struct.pack("<I", self.hidden_width))
        for values in self.parameters:
            for value in values:
                digest.update(struct.pack("<f", value))
        return digest.hexdigest()


@dataclass(frozen=True)
class ReferenceInput:
    base_input: model_reference.ReferenceInput
    target_ids: tuple

    @classmethod
    def from_mapping(cls, value, config):
        numeric._fields(value, ("schema", "base_input", "target_ids"), "utility input")
        if value["schema"] != INPUT_SCHEMA:
            raise ValueError("unregistered utility input schema")
        checked = model_reference.ReferenceInput.from_mapping(value["base_input"], config)
        target = value["target_ids"]
        if type(target) is not list or not 1 <= len(target) <= config.max_lines:
            raise ValueError("utility requires a nonempty bounded target ID list")
        identifiers = tuple(numeric._integer(item, 0, (1 << 64) - 1, "utility target ID") for item in target)
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("duplicate utility target ID")
        records = {record.id: record for record in checked.context_lines}
        if any(identifier not in records for identifier in identifiers):
            raise ValueError("utility target references an unknown record")
        selected = set(identifiers)
        if any((edge.source in selected) != (edge.target in selected) for edge in checked.context_edges):
            raise ValueError("utility target leaves a crossing relation edge")
        first = records[identifiers[0]]
        if first.original_first_move is None:
            raise ValueError("utility target requires declared original first move")
        for identifier in identifiers[1:]:
            record = records[identifier]
            if (record.line.anchor_features != first.line.anchor_features
                    or record.line.perspective != first.line.perspective
                    or record.original_first_move != first.original_first_move):
                raise ValueError("utility target numeric anchor/original first move mismatch")
        # Do not infer that unselected equal numeric anchors have equal Rules
        # history, nor claim whole exact-anchor closure from this feature wire.
        return cls(checked, identifiers)

    def to_mapping(self):
        return {"schema": INPUT_SCHEMA, "base_input": self.base_input.to_mapping(), "target_ids": list(self.target_ids)}

    @property
    def target_sha256(self):
        """Pure numeric provenance; not the runtime's submission/lease seal."""
        digest = hashlib.sha256()
        digest.update((INPUT_SCHEMA + "\0target\0" + self.base_input.context_sha256 + "\0").encode("ascii"))
        digest.update(struct.pack("<I", len(self.target_ids)))
        for identifier in self.target_ids:
            digest.update(struct.pack("<Q", identifier))
        return digest.hexdigest()


def numeric_input_sha256(inputs, base_weights_sha256, weights_sha256):
    """Seal actual numerical B/L/target inputs, separately from live view authority."""
    digest = hashlib.sha256()
    current = inputs.base_input
    digest.update((INPUT_SCHEMA + "\0numeric\0" + base_weights_sha256 + "\0"
                   + weights_sha256 + "\0" + current.context_sha256 + "\0").encode("ascii"))
    for features in current.board_features:
        digest.update(struct.pack("<I", len(features)))
        for feature in features:
            digest.update(struct.pack("<H", feature))
    digest.update(bytes((current.side_to_move == "black",)))
    digest.update(struct.pack("<I", len(current.local_line.anchor_features)))
    for feature in current.local_line.anchor_features:
        digest.update(struct.pack("<H", feature))
    digest.update(bytes((current.local_line.perspective == "black",)))
    digest.update(struct.pack("<I", len(current.local_line.moves)))
    for move in current.local_line.moves:
        digest.update(struct.pack("<BBB", move.from_square, move.to_square, move.promotion))
    digest.update(struct.pack("<I", len(inputs.target_ids)))
    for identifier in inputs.target_ids:
        digest.update(struct.pack("<Q", identifier))
    return digest.hexdigest()


def evaluate(base_checkpoint, checkpoint, inputs):
    if type(checkpoint) is not UtilityCheckpoint or type(inputs) is not ReferenceInput:
        raise ValueError("validated utility checkpoint and selected input required")
    checkpoint.validate_base(base_checkpoint)
    inputs = ReferenceInput.from_mapping(inputs.to_mapping(), base_checkpoint.base.config)
    selected = set(inputs.target_ids)
    current = inputs.base_input
    context, _, line_contributions, _, pair_contributions = model_reference._context(base_checkpoint, current)
    target_terms = [contribution for record, contribution in zip(current.context_lines, line_contributions, strict=True)
                    if record.id in selected]
    target_terms.extend(contribution for edge, contribution in zip(current.context_edges, pair_contributions, strict=True)
                        if edge.source in selected and edge.target in selected)
    config = base_checkpoint.base.config
    target = tuple(numeric._exact_sum_fp32((0.0, *(term[column] for term in target_terms)), 256)
                   for column in range(config.context_width))
    if any(abs(value) >= model_reference.MAX_CONTEXT_MAGNITUDE for value in target):
        raise ValueError("utility target contribution exceeds V2 exact sum bound")
    original = dict(zip(numeric.PARAMETER_FIELDS, base_checkpoint.base.parameters, strict=True))
    board = tuple(numeric._sparse(original, features, config.board_width, "board_weights", "board_bias", exact_board=True)
                  for features in current.board_features)
    local = numeric._dense(original["local_projection"], numeric._encode_line(base_checkpoint.base, current.local_line),
                           config.local_width, original["local_bias"])
    side = 0 if current.side_to_move == "white" else 1
    values = tuple(numeric._clip(value) for group in (board[side], board[1 - side], context, local, target) for value in group)
    parameters = dict(zip(PARAMETER_FIELDS, checkpoint.parameters, strict=True))
    hidden = tuple(numeric._relu(value) for value in numeric._dense(
        parameters["hidden_weights"], values, checkpoint.hidden_width, parameters["hidden_bias"],
    ))
    raw = numeric._dense(parameters["output_weights"], hidden, 7, parameters["output_bias"])
    return {"schema": OUTPUT_SCHEMA, "semantics": SEMANTICS, "base_weights_sha256": base_checkpoint.weights_sha256,
            "weights_sha256": checkpoint.weights_sha256, "context_sha256": current.context_sha256,
            "target_sha256": inputs.target_sha256,
            "input_sha256": numeric_input_sha256(inputs, base_checkpoint.weights_sha256, checkpoint.weights_sha256),
            "hidden": list(hidden), "raw_outputs": list(raw), "output_names": list(OUTPUT_NAMES),
            "target_contribution": list(target), "global_context": list(context), "clipped_input": list(values),
            "reference_scope": REFERENCE_SCOPE, "policy_actions_authorized": False, "target_known_mask": [False] * 7,
            "exact_rules_bundle_completeness": "requires_actual_Rust_BoundFactory_evidence",
            "runtime_lease_authority": False}


def reference_json(base_bytes, checkpoint_bytes, input_bytes):
    if type(checkpoint_bytes) is not bytes or len(checkpoint_bytes) > MAX_CHECKPOINT_BYTES:
        raise ValueError("bounded immutable utility checkpoint bytes <= 8MiB required")
    frozen = model_reference.NnuecV2Checkpoint.from_mapping(numeric.decode_json(base_bytes))
    checkpoint = UtilityCheckpoint.from_mapping(numeric.decode_json(checkpoint_bytes), frozen)
    inputs = ReferenceInput.from_mapping(numeric.decode_json(input_bytes), frozen.base.config)
    result = evaluate(frozen, checkpoint, inputs)
    result["input_bytes_sha256"] = hashlib.sha256(input_bytes).hexdigest()
    return json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8") + b"\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", help="explicit frozen V2 checkpoint")
    parser.add_argument("checkpoint", help="independently declared shadow utility checkpoint")
    parser.add_argument("input", help="V2 numeric input and selected target IDs")
    arguments = parser.parse_args(argv)
    try:
        result = reference_json(numeric._read_bounded(arguments.base), numeric._read_bounded(arguments.checkpoint),
                                numeric._read_bounded(arguments.input))
    except (OSError, ValueError) as error:
        parser.exit(2, f"NNUEC utility reference rejected: {error}\n")
    sys.stdout.buffer.write(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
