"""Independent scalar FP32 reference for the explicit NNUEC B/C/L profile.

This module imports only the Python standard library. It neither parses chess
positions nor decides legality: the input is the ordered board-800 features and
checked move coordinates supplied by the Rust Rules consumer. Context lines C
and the local continuation L are distinct inputs. Nothing is trained, exported
to ONNX, registered as an arena asset or evaluated merely by importing it.

Encoder, projection and head multiply/add operations round separately to IEEE
binary32, without FMA. B sums admitted FP32 feature terms exactly in signed192;
C sums already-rounded line contributions exactly in signed256. Each exact sum
rounds once to FP32. This is accumulation of FP32 operands, not quantized weights.
Weights are output-row-major. Widths are required bounded configuration values,
not a newly selected default architecture. The byte digest below is shared with
the Rust reader; JSON formatting and declared training metadata are not weights.
"""

from dataclasses import dataclass
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import sys


SCHEMA = "rz-nnuec-fp32-v1"
SEMANTICS = "board800-exactb192-causal-line-exactc256-fp32-v1"
INPUT_SCHEMA = "rz-nnuec-fp32-input/1"
OUTPUT_SCHEMA = "rz-nnuec-fp32-output/1"
REFERENCE_SCOPE = "independent_scalar_numeric_only_not_rules_or_training_proof"
BOARD_FEATURES = 800
OUTPUT_ROWS = 11
MAX_JSON_BYTES = 32 << 20
MAX_PARAMETER_MAGNITUDE = 1000.0
MAX_CONTEXT_CONTRIBUTION = float(1 << 60)
MAX_CONTEXT_MAGNITUDE = float(1 << 68)
CONFIG_FIELDS = (
    "board_width", "line_width", "context_width", "local_width", "hidden_width",
    "causal_window", "max_lines", "max_plies",
)
PARAMETER_FIELDS = (
    "board_weights", "board_bias", "anchor_weights", "anchor_bias",
    "from_weights", "to_weights", "promotion_weights", "ply_weights",
    "token_bias", "causal_weights", "causal_bias", "context_projection",
    "context_bias", "local_projection", "local_bias", "hidden_weights",
    "hidden_bias", "output_weights", "output_bias",
)


def _fields(value, names, label):
    if type(value) is not dict or set(value) != set(names):
        raise ValueError(f"{label}: exact fields required")
    return value


def _integer(value, low, high, label):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{label}: integer outside {low}..{high}")
    return value


def f32(value):
    """Round a finite scalar to binary32, rejecting overflow and booleans."""
    if type(value) not in (int, float):
        raise ValueError("FP32 scalar must be a number")
    try:
        result = struct.unpack("<f", struct.pack("<f", value))[0]
    except (OverflowError, struct.error) as error:
        raise ValueError("FP32 overflow") from error
    if not math.isfinite(result):
        raise ValueError("finite FP32 scalar required")
    return result


def _add(left, right):
    return f32(left + right)


def _mul(left, right):
    return f32(left * right)


def _exact_sum_fp32(values, bits):
    """Independent bigint sum in units 2^-149, with checked signed capacity.

    Only B's bounded parameters and C's bounded projected contributions use
    this path. In particular, signed256 is not a claim that an arbitrary sum
    of all finite binary32 operands is supported.
    """
    total = 0
    limit = 1 << (bits - 1)
    for value in values:
        raw = struct.unpack("<I", struct.pack("<f", f32(value)))[0]
        exponent = (raw >> 23) & 255
        fraction = raw & 0x7fffff
        magnitude = fraction if exponent == 0 else (fraction | 0x800000) << (exponent - 1)
        total += -magnitude if raw >> 31 else magnitude
        if not -limit <= total < limit:
            raise ValueError(f"exact signed{bits} accumulation overflow")
    if total == 0:
        return 0.0
    sign = 0x80000000 if total < 0 else 0
    magnitude = abs(total)
    highest = magnitude.bit_length() - 1
    if highest <= 22:
        return struct.unpack("<f", struct.pack("<I", sign | magnitude))[0]
    shift = highest - 23
    mantissa = magnitude >> shift
    if shift:
        remainder = magnitude - (mantissa << shift)
        half = 1 << (shift - 1)
        if remainder > half or (remainder == half and mantissa & 1):
            mantissa += 1
            if mantissa == 0x1000000:
                mantissa >>= 1
                highest += 1
    exponent = highest - 22
    if exponent >= 255:
        raise ValueError("exact accumulation cannot round to finite FP32")
    raw = sign | (exponent << 23) | (mantissa & 0x7fffff)
    return struct.unpack("<f", struct.pack("<I", raw))[0]


def _relu(value):
    return value if value > 0.0 else 0.0


def _clip(value):
    # Preserve the sign of an in-range zero, as Rust f32::clamp does.
    return 0.0 if value < 0.0 else 1.0 if value > 1.0 else value


def _dense(weights, values, width, bias=None):
    stride = len(values)
    output = []
    for index in range(width):
        # Dense heads and L add each product directly to the bias. C uses a
        # zero-start FP32 dot, then exact summation of immutable contributions.
        total = 0.0 if bias is None else bias[index]
        start = index * stride
        for column, value in enumerate(values):
            total = _add(total, _mul(weights[start + column], value))
        output.append(total)
    return tuple(output)


@dataclass(frozen=True)
class NnuecConfig:
    board_width: int
    line_width: int
    context_width: int
    local_width: int
    hidden_width: int
    causal_window: int
    max_lines: int
    max_plies: int

    def __post_init__(self):
        for name in CONFIG_FIELDS[:4]:
            _integer(getattr(self, name), 1, 256, name)
        _integer(self.hidden_width, 1, 64, "hidden_width")
        if type(self.causal_window) is not int or self.causal_window not in (3, 5):
            raise ValueError("causal_window must be 3 or 5")
        _integer(self.max_lines, 1, 128, "max_lines")
        _integer(self.max_plies, 1, 256, "max_plies")

    @classmethod
    def from_mapping(cls, value):
        return cls(**_fields(value, CONFIG_FIELDS, "config"))

    def to_mapping(self):
        return {name: getattr(self, name) for name in CONFIG_FIELDS}

    def parameter_lengths(self):
        b, e, c, local, hidden = (
            self.board_width, self.line_width, self.context_width,
            self.local_width, self.hidden_width,
        )
        return dict(zip(PARAMETER_FIELDS, (
            BOARD_FEATURES * b, b, BOARD_FEATURES * e, e,
            64 * e, 64 * e, 5 * e, self.max_plies * e, e,
            self.causal_window * e, e, c * e, c, local * e, local,
            hidden * (2 * b + c + local), hidden,
            OUTPUT_ROWS * hidden, OUTPUT_ROWS,
        ), strict=True))


def _training(value):
    if type(value) is not dict:
        raise ValueError("training declaration required")
    kind = value.get("kind")
    if kind == "untrained":
        _fields(value, ("kind",), "untrained declaration")
        return (("kind", "untrained"),)
    if kind != "learned":
        raise ValueError("NNUEC requires untrained or learned; bootstrap is separate")
    _fields(value, ("kind", "run_id", "steps", "dataset_sha256"), "learned declaration")
    run = value["run_id"]
    if (type(run) is not str or not run.strip() or len(run.encode("utf-8")) > 128
            or any(ord(character) < 32 or 127 <= ord(character) <= 159 for character in run)):
        raise ValueError("bounded learned run_id required")
    _integer(value["steps"], 1, (1 << 64) - 1, "learned steps")
    digest = value["dataset_sha256"]
    if (type(digest) is not str or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)):
        raise ValueError("lowercase dataset SHA-256 required")
    # A checkpoint declaration is deliberately not proof of a training run.
    return tuple((name, value[name]) for name in ("kind", "run_id", "steps", "dataset_sha256"))


@dataclass(frozen=True)
class NnuecCheckpoint:
    config: NnuecConfig
    training: tuple
    parameters: tuple
    output_scale: float

    def __post_init__(self):
        if type(self.config) is not NnuecConfig:
            raise ValueError("validated NNUEC configuration required")
        normalized_training = _training(dict(self.training))
        if self.training != normalized_training:
            raise ValueError("canonical immutable training declaration required")
        if type(self.parameters) is not tuple or len(self.parameters) != len(PARAMETER_FIELDS):
            raise ValueError("all immutable NNUEC parameter arrays required")
        for (name, length), values in zip(self.config.parameter_lengths().items(), self.parameters, strict=True):
            if type(values) is not tuple or len(values) != length:
                raise ValueError(f"{name}: expected {length} scalar values")
            for value in values:
                if (type(value) is not float or not math.isfinite(value)
                        or abs(value) > MAX_PARAMETER_MAGNITUDE or f32(value) != value):
                    raise ValueError(f"{name}: bounded already-rounded FP32 values required")
        if (type(self.output_scale) is not float or not 0.0 < self.output_scale <= 10000.0
                or f32(self.output_scale) != self.output_scale):
            raise ValueError("positive FP32 output_scale <= 10000 required")

    @classmethod
    def from_mapping(cls, value):
        _fields(value, ("schema", "semantics", "config", "training", "parameters", "output_scale"), "checkpoint")
        if value["schema"] != SCHEMA or value["semantics"] != SEMANTICS:
            raise ValueError("unregistered NNUEC schema or semantics")
        config = NnuecConfig.from_mapping(value["config"])
        parameters = _fields(value["parameters"], PARAMETER_FIELDS, "parameters")
        converted = []
        for name, length in config.parameter_lengths().items():
            values = parameters[name]
            if type(values) is not list or len(values) != length:
                raise ValueError(f"{name}: expected {length} scalar values")
            converted.append(tuple(f32(item) for item in values))
        return cls(config, _training(value["training"]), tuple(converted), f32(value["output_scale"]))

    def to_mapping(self):
        return {"schema": SCHEMA, "semantics": SEMANTICS, "config": self.config.to_mapping(),
                "training": dict(self.training), "parameters": {
                    name: list(values) for name, values in zip(PARAMETER_FIELDS, self.parameters, strict=True)},
                "output_scale": self.output_scale}

    def parameter(self, name):
        return self.parameters[PARAMETER_FIELDS.index(name)]

    @property
    def weights_sha256(self):
        digest = hashlib.sha256()
        digest.update((SCHEMA + "\0" + SEMANTICS + "\0").encode("utf-8"))
        for name in CONFIG_FIELDS:
            digest.update(struct.pack("<I", getattr(self.config, name)))
        for values in self.parameters:
            for value in values:
                digest.update(struct.pack("<f", value))
        digest.update(struct.pack("<f", self.output_scale))
        return digest.hexdigest()


def _features(value, label, perspective, expected_stm_feature):
    # At most 64 occupied squares plus seven metadata terms. This does not
    # impose an unverified 32-piece legality restriction on accepted FENs.
    if type(value) is not list or len(value) > 71:
        raise ValueError(f"{label}: bounded ordered sparse feature list required")
    result = tuple(_integer(item, 0, BOARD_FEATURES - 1, label) for item in value)
    if len(set(result)) != len(result):
        raise ValueError(f"{label}: duplicate sparse feature")
    split = next((index for index, feature in enumerate(result) if feature >= 768), len(result))
    pieces, metadata = result[:split], result[split:]
    squares = tuple((feature % 64) ^ (56 if perspective == "black" else 0) for feature in pieces)
    if any(left >= right for left, right in zip(squares, squares[1:])):
        raise ValueError(f"{label}: piece features must follow original board-square order")
    if (not 3 <= len(metadata) <= 7 or metadata[0] != expected_stm_feature
            or not 774 <= metadata[-2] <= 782 or not 783 <= metadata[-1] <= 799
            or any(not 770 <= feature <= 773 for feature in metadata[1:-2])
            or tuple(sorted(metadata[1:-2])) != metadata[1:-2]):
        raise ValueError(f"{label}: ordered board-800 metadata or perspective mismatch")
    return result


@dataclass(frozen=True)
class Move:
    from_square: int
    to_square: int
    promotion: int

    def __post_init__(self):
        _integer(self.from_square, 0, 63, "from_square")
        _integer(self.to_square, 0, 63, "to_square")
        _integer(self.promotion, 0, 4, "promotion")
        if self.from_square == self.to_square:
            raise ValueError("a move must change square")

    @classmethod
    def from_mapping(cls, value):
        return cls(**_fields(value, ("from_square", "to_square", "promotion"), "move"))

    def to_mapping(self):
        return {"from_square": self.from_square, "to_square": self.to_square, "promotion": self.promotion}


@dataclass(frozen=True)
class Line:
    anchor_features: tuple
    perspective: str
    moves: tuple

    @classmethod
    def from_mapping(cls, value, config):
        _fields(value, ("anchor_features", "perspective", "moves"), "line")
        if value["perspective"] not in ("white", "black"):
            raise ValueError("line perspective must be white or black")
        if type(value["moves"]) is not list or len(value["moves"]) > config.max_plies:
            raise ValueError("line exceeds max_plies")
        return cls(_features(value["anchor_features"], "anchor_features", value["perspective"], 768), value["perspective"],
                   tuple(Move.from_mapping(item) for item in value["moves"]))

    def to_mapping(self):
        return {"anchor_features": list(self.anchor_features), "perspective": self.perspective,
                "moves": [move.to_mapping() for move in self.moves]}


@dataclass(frozen=True)
class ReferenceInput:
    board_features: tuple
    side_to_move: str
    context_lines: tuple
    local_line: Line

    @classmethod
    def from_mapping(cls, value, config):
        _fields(value, ("schema", "board_features", "side_to_move", "context_lines", "local_line"), "input")
        if value["schema"] != INPUT_SCHEMA or value["side_to_move"] not in ("white", "black"):
            raise ValueError("NNUEC input schema or side_to_move mismatch")
        boards = value["board_features"]
        if type(boards) is not list or len(boards) != 2:
            raise ValueError("white and black ordered board features required")
        lines = value["context_lines"]
        if type(lines) is not list or len(lines) > config.max_lines:
            raise ValueError("context_lines exceeds max_lines")
        return cls(tuple(_features(item, "board_features", perspective,
                                   768 + (value["side_to_move"] != perspective))
                         for item, perspective in zip(boards, ("white", "black"), strict=True)), value["side_to_move"],
                   tuple(Line.from_mapping(item, config) for item in lines),
                   Line.from_mapping(value["local_line"], config))

    def to_mapping(self):
        return {"schema": INPUT_SCHEMA, "board_features": [list(row) for row in self.board_features],
                "side_to_move": self.side_to_move, "context_lines": [line.to_mapping() for line in self.context_lines],
                "local_line": self.local_line.to_mapping()}


def _sparse(parameters, features, width, weights, bias, *, exact_board=False):
    if exact_board:
        return tuple(_exact_sum_fp32(
            (parameters[bias][column], *(parameters[weights][feature * width + column] for feature in features)),
            192,
        ) for column in range(width))
    output = list(parameters[bias])
    for feature in features:
        for column in range(width):
            output[column] = _add(output[column], parameters[weights][feature * width + column])
    return tuple(output)


def _encode_line(checkpoint, line):
    config = checkpoint.config
    parameters = dict(zip(PARAMETER_FIELDS, checkpoint.parameters, strict=True))
    width = config.line_width
    if not line.moves:
        return (0.0,) * width
    anchor = _sparse(parameters, line.anchor_features, width, "anchor_weights", "anchor_bias")
    tokens = []
    encoded = [0.0] * width
    for ply, move in enumerate(line.moves):
        origin = move.from_square ^ (56 if line.perspective == "black" else 0)
        target = move.to_square ^ (56 if line.perspective == "black" else 0)
        token = []
        for column in range(width):
            value = _add(parameters["token_bias"][column], anchor[column])
            for name, row in (("from_weights", origin), ("to_weights", target),
                              ("promotion_weights", move.promotion), ("ply_weights", ply)):
                value = _add(value, parameters[name][row * width + column])
            token.append(value)
        tokens.append(token)
        for column in range(width):
            value = parameters["causal_bias"][column]
            for lag in range(min(config.causal_window, ply + 1)):
                value = _add(value, _mul(tokens[ply - lag][column],
                                        parameters["causal_weights"][lag * width + column]))
            encoded[column] = _add(encoded[column], _relu(value))
    return tuple(encoded)


def evaluate(checkpoint, inputs):
    """Full independent recomputation; this reference does not emulate a TT.

    The conditionally interpreted row 4 is a state hint, not a candidate policy
    distribution or a universal mate/value proof. Rows 5..10 remain raw hints.
    WDL softmax is auxiliary; raw logits are the primary scalar cross-check.
    """
    if type(checkpoint) is not NnuecCheckpoint or type(inputs) is not ReferenceInput:
        raise ValueError("validated checkpoint and reference input required")
    # Validate publicly constructible frozen input objects at the boundary too.
    inputs = ReferenceInput.from_mapping(inputs.to_mapping(), checkpoint.config)
    config = checkpoint.config
    parameters = dict(zip(PARAMETER_FIELDS, checkpoint.parameters, strict=True))
    board = tuple(_sparse(parameters, features, config.board_width, "board_weights", "board_bias", exact_board=True)
                  for features in inputs.board_features)
    context_contributions = []
    for line in inputs.context_lines:
        representation = _encode_line(checkpoint, line)
        contribution = _dense(parameters["context_projection"], representation, config.context_width)
        if any(abs(value) > MAX_CONTEXT_CONTRIBUTION for value in contribution):
            raise ValueError("context contribution exceeds admitted FP32 operand bound")
        context_contributions.append(contribution)
    context = [_exact_sum_fp32(
        (parameters["context_bias"][column], *(row[column] for row in context_contributions)), 256,
    ) for column in range(config.context_width)]
    if any(abs(value) > MAX_CONTEXT_MAGNITUDE for value in context):
        raise ValueError("context accumulator exceeds admitted FP32 result bound")
    local_representation = _encode_line(checkpoint, inputs.local_line)
    local = _dense(parameters["local_projection"], local_representation, config.local_width,
                   parameters["local_bias"])
    side = 0 if inputs.side_to_move == "white" else 1
    dense_input = tuple(_clip(value) for group in (board[side], board[1 - side], context, local) for value in group)
    hidden = tuple(_relu(value) for value in _dense(parameters["hidden_weights"], dense_input,
                                                   config.hidden_width, parameters["hidden_bias"]))
    raw = _dense(parameters["output_weights"], hidden, OUTPUT_ROWS, parameters["output_bias"])
    logits = raw[1:4]
    maximum = max(logits)
    exponentials = tuple(f32(math.exp(_add(value, -maximum))) for value in logits)
    denominator = 0.0
    for value in exponentials:
        denominator = _add(denominator, value)
    wdl = tuple(f32(value / denominator) for value in exponentials)
    return {"schema": OUTPUT_SCHEMA, "semantics": SEMANTICS, "weights_sha256": checkpoint.weights_sha256,
            "reference_scope": REFERENCE_SCOPE, "board_accumulators": [list(row) for row in board],
            "context_contributions": [list(row) for row in context_contributions],
            "context_accumulator": context, "local_representation": list(local_representation),
            "local_accumulator": list(local), "hidden": list(hidden), "raw_outputs": list(raw),
            "value": _mul(raw[0], checkpoint.output_scale), "wdl_logits": list(logits), "wdl": list(wdl)}


def _reject_constant(value):
    raise ValueError(f"JSON nonfinite constant forbidden: {value}")


def _unique_object(pairs):
    result = {}
    for name, value in pairs:
        if name in result:
            raise ValueError(f"duplicate JSON field: {name}")
        result[name] = value
    return result


def decode_json(raw):
    if type(raw) is not bytes or len(raw) > MAX_JSON_BYTES:
        raise ValueError("bounded immutable JSON bytes required")
    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object,
                          parse_constant=_reject_constant)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as error:
        raise ValueError("invalid bounded NNUEC JSON") from error


def reference_json(checkpoint_bytes, input_bytes):
    checkpoint = NnuecCheckpoint.from_mapping(decode_json(checkpoint_bytes))
    inputs = ReferenceInput.from_mapping(decode_json(input_bytes), checkpoint.config)
    return json.dumps(evaluate(checkpoint, inputs), sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8") + b"\n"


def _read_bounded(path):
    with Path(path).open("rb") as stream:
        raw = stream.read(MAX_JSON_BYTES + 1)
    if len(raw) > MAX_JSON_BYTES:
        raise ValueError("NNUEC JSON file exceeds byte limit")
    return raw


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", help="explicit bounded NNUEC FP32 checkpoint JSON")
    parser.add_argument("input", help="Rust Rules-checked ordered-feature input JSON")
    arguments = parser.parse_args(argv)
    try:
        result = reference_json(_read_bounded(arguments.checkpoint), _read_bounded(arguments.input))
    except (OSError, ValueError) as error:
        parser.exit(2, f"NNUEC reference rejected: {error}\n")
    sys.stdout.buffer.write(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
