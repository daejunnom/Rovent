"""Independent NNUEC V2 kind and directed-relation FP32 reference.

V1 assets and semantics remain separate. A V2 checkpoint explicitly contains a
complete V1 base plus five new parameter arrays; accepting it never upgrades or
rewrites a V1 asset. Scalar line encoding and the B/L/head kernels reuse the
already independent standard-library reference. New typed-line and relation
contributions are evaluated here and summed with checked exact256 arithmetic.

Numeric feature anchors are not full Rules/history identities. Their equality
and a common original-first-move declaration are necessary reference-wire
checks; the Rust owner must additionally check exact Rules anchors and rebase
provenance. An actual rebased suffix may be empty or start with another move.
Record IDs are lookup keys only. They do not enter an embedding or a head.
"""

from dataclasses import dataclass
import argparse
import hashlib
import json
import math
import struct
import sys

from . import nnuec_reference as base


SCHEMA = "rz-nnuec-fp32-v2"
SEMANTICS = "board800-exactb192-causal-line-kind4-directedpair-originfirst-exactc256-fp32-v2"
INPUT_SCHEMA = "rz-nnuec-fp32-input/2"
OUTPUT_SCHEMA = "rz-nnuec-fp32-output/2"
REFERENCE_SCOPE = "independent_scalar_kind_relation_numeric_not_exact_rules_anchor_proof"
PARAMETER_FIELDS = ("kind_embeddings", "pair_parent", "pair_child", "pair_bias", "pair_projection")
MAX_CONTEXT_OPERAND = float(1 << 70)
MAX_CONTEXT_MAGNITUDE = float(1 << 79)


@dataclass(frozen=True)
class NnuecV2Checkpoint:
    base: base.NnuecCheckpoint
    parameters: tuple

    def __post_init__(self):
        if type(self.base) is not base.NnuecCheckpoint:
            raise ValueError("V2 requires an explicit validated V1 base checkpoint")
        if type(self.parameters) is not tuple or len(self.parameters) != len(PARAMETER_FIELDS):
            raise ValueError("V2 requires five immutable kind/relation parameter arrays")
        for (name, length), values in zip(self.parameter_lengths().items(), self.parameters, strict=True):
            if type(values) is not tuple or len(values) != length:
                raise ValueError(f"{name}: expected {length} FP32 parameters")
            if any(type(value) is not float or not math.isfinite(value)
                   or abs(value) > base.MAX_PARAMETER_MAGNITUDE or base.f32(value) != value for value in values):
                raise ValueError(f"{name}: bounded already-rounded FP32 parameters required")

    def parameter_lengths(self):
        e, c = self.base.config.line_width, self.base.config.context_width
        return dict(zip(PARAMETER_FIELDS, (4 * e, 2 * e, 2 * e, 2 * e, c * e), strict=True))

    @classmethod
    def from_mapping(cls, value):
        base._fields(value, ("schema", "semantics", "base", "parameters"), "V2 checkpoint")
        if value["schema"] != SCHEMA or value["semantics"] != SEMANTICS:
            raise ValueError("unregistered NNUEC V2 schema or semantics; no V1 conversion")
        admitted_base = base.NnuecCheckpoint.from_mapping(value["base"])
        parameters = base._fields(value["parameters"], PARAMETER_FIELDS, "V2 parameters")
        e, c = admitted_base.config.line_width, admitted_base.config.context_width
        arrays = []
        for name, length in zip(PARAMETER_FIELDS, (4 * e, 2 * e, 2 * e, 2 * e, c * e), strict=True):
            values = parameters[name]
            if type(values) is not list or len(values) != length:
                raise ValueError(f"{name}: expected {length} FP32 parameters")
            arrays.append(tuple(base.f32(value) for value in values))
        return cls(admitted_base, tuple(arrays))

    def to_mapping(self):
        return {"schema": SCHEMA, "semantics": SEMANTICS, "base": self.base.to_mapping(),
                "parameters": {name: list(values) for name, values in zip(PARAMETER_FIELDS, self.parameters, strict=True)}}

    @property
    def weights_sha256(self):
        digest = hashlib.sha256()
        digest.update((SCHEMA + "\0" + SEMANTICS + "\0" + self.base.weights_sha256 + "\0").encode("ascii"))
        for values in self.parameters:
            for value in values:
                digest.update(struct.pack("<f", value))
        return digest.hexdigest()


@dataclass(frozen=True)
class Record:
    id: int
    kind: int
    original_first_move: base.Move | None
    line: base.Line

    @classmethod
    def from_mapping(cls, value, config):
        base._fields(value, ("id", "kind", "original_first_move", "anchor_features", "perspective", "moves"), "V2 record")
        identifier = base._integer(value["id"], 0, (1 << 64) - 1, "record id")
        kind = base._integer(value["kind"], 0, 3, "record kind")
        original = None if value["original_first_move"] is None else base.Move.from_mapping(value["original_first_move"])
        line = base.Line.from_mapping({name: value[name] for name in ("anchor_features", "perspective", "moves")}, config)
        return cls(identifier, kind, original, line)

    def to_mapping(self):
        return {"id": self.id, "kind": self.kind,
                "original_first_move": None if self.original_first_move is None else self.original_first_move.to_mapping(),
                **self.line.to_mapping()}


@dataclass(frozen=True)
class Edge:
    source: int
    target: int
    kind: int

    @classmethod
    def from_mapping(cls, value):
        base._fields(value, ("source", "target", "kind"), "V2 edge")
        return cls(base._integer(value["source"], 0, (1 << 64) - 1, "edge source"),
                   base._integer(value["target"], 0, (1 << 64) - 1, "edge target"),
                   base._integer(value["kind"], 0, 1, "edge kind"))

    def to_mapping(self):
        return {"source": self.source, "target": self.target, "kind": self.kind}


def _check_graph(records, edges):
    identifiers = {record.id: record for record in records}
    if len(identifiers) != len(records):
        raise ValueError("duplicate record lookup id")
    seen, incoming = set(), set()
    adjacency = {identifier: [] for identifier in identifiers}
    indegree = dict.fromkeys(identifiers, 0)
    for edge in edges:
        if edge.source not in identifiers or edge.target not in identifiers:
            raise ValueError("dangling relation endpoint")
        key = (edge.source, edge.target, edge.kind)
        if key in seen:
            raise ValueError("duplicate directed relation")
        seen.add(key)
        source, target = identifiers[edge.source], identifiers[edge.target]
        if (source.line.anchor_features != target.line.anchor_features
                or source.line.perspective != target.line.perspective
                or source.original_first_move is None or source.original_first_move != target.original_first_move):
            raise ValueError("relation requires common numeric anchor and original first move")
        incoming_key = (edge.target, edge.kind)
        if incoming_key in incoming:
            raise ValueError("multiple same-kind relation parents for one record")
        incoming.add(incoming_key)
        adjacency[edge.source].append(edge.target)
        indegree[edge.target] += 1
    # Kahn's algorithm checks the union of both directed edge kinds, without
    # recursive depth depending on untrusted relation order.
    queue = [identifier for identifier, degree in indegree.items() if degree == 0]
    visited = 0
    while queue:
        identifier = queue.pop()
        visited += 1
        for target in adjacency[identifier]:
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)
    if visited != len(records):
        raise ValueError("directed relation cycle")


@dataclass(frozen=True)
class ReferenceInput:
    board_features: tuple
    side_to_move: str
    context_lines: tuple
    context_edges: tuple
    local_line: base.Line

    @classmethod
    def from_mapping(cls, value, config):
        base._fields(value, ("schema", "board_features", "side_to_move", "context_lines", "context_edges", "local_line"), "V2 input")
        if value["schema"] != INPUT_SCHEMA:
            raise ValueError("unregistered V2 input schema; no V1 conversion")
        lines, edges = value["context_lines"], value["context_edges"]
        if type(lines) is not list or len(lines) > config.max_lines:
            raise ValueError("V2 record count exceeds max_lines")
        if type(edges) is not list or len(edges) > 2 * config.max_lines:
            raise ValueError("V2 edge count exceeds two relations per record")
        records = tuple(Record.from_mapping(line, config) for line in lines)
        admitted_edges = tuple(Edge.from_mapping(edge) for edge in edges)
        _check_graph(records, admitted_edges)
        # Reuse scalar B/local validators without constructing a converted V1
        # input asset or discarding the V2 graph at an acceptance boundary.
        side, boards = value["side_to_move"], value["board_features"]
        if side not in ("white", "black") or type(boards) is not list or len(boards) != 2:
            raise ValueError("V2 requires side_to_move and both ordered board feature lists")
        board_features = tuple(base._features(features, "board_features", perspective,
                                              768 + (side != perspective))
                               for features, perspective in zip(boards, ("white", "black"), strict=True))
        local = base.Line.from_mapping(value["local_line"], config)
        return cls(board_features, side, records, admitted_edges, local)

    def to_mapping(self):
        return {"schema": INPUT_SCHEMA, "board_features": [list(row) for row in self.board_features],
                "side_to_move": self.side_to_move, "context_lines": [record.to_mapping() for record in self.context_lines],
                "context_edges": [edge.to_mapping() for edge in self.context_edges], "local_line": self.local_line.to_mapping()}

    @property
    def context_sha256(self):
        """Seal actual lookup IDs and selected order, never neural features."""
        digest = hashlib.sha256()
        digest.update((INPUT_SCHEMA + "\0context\0").encode("ascii"))
        digest.update(struct.pack("<I", len(self.context_lines)))
        for record in self.context_lines:
            digest.update(struct.pack("<QBB", record.id, record.kind, record.line.perspective == "black"))
            original = record.original_first_move
            digest.update(bytes((original is not None,)))
            if original is not None:
                digest.update(struct.pack("<BBB", original.from_square, original.to_square, original.promotion))
            digest.update(struct.pack("<I", len(record.line.anchor_features)))
            for feature in record.line.anchor_features:
                digest.update(struct.pack("<H", feature))
            digest.update(struct.pack("<I", len(record.line.moves)))
            for move in record.line.moves:
                digest.update(struct.pack("<BBB", move.from_square, move.to_square, move.promotion))
        digest.update(struct.pack("<I", len(self.context_edges)))
        for edge in self.context_edges:
            digest.update(struct.pack("<QQB", edge.source, edge.target, edge.kind))
        return digest.hexdigest()


def _context(checkpoint, inputs):
    parameters = dict(zip(PARAMETER_FIELDS, checkpoint.parameters, strict=True))
    original = dict(zip(base.PARAMETER_FIELDS, checkpoint.base.parameters, strict=True))
    e, c = checkpoint.base.config.line_width, checkpoint.base.config.context_width
    representations, line_contributions = {}, []
    for record in inputs.context_lines:
        raw = base._encode_line(checkpoint.base, record.line)
        typed = tuple(base._add(value, parameters["kind_embeddings"][record.kind * e + column])
                      for column, value in enumerate(raw))
        representations[record.id] = typed
        line_contributions.append(base._dense(original["context_projection"], typed, c))
    pair_contributions, pair_embeddings = [], []
    for edge in inputs.context_edges:
        parent, child = representations[edge.source], representations[edge.target]
        pair = []
        for column in range(e):
            row = edge.kind * e + column
            value = base._add(parameters["pair_bias"][row], base._mul(parent[column], parameters["pair_parent"][row]))
            value = base._add(value, base._mul(child[column], parameters["pair_child"][row]))
            pair.append(base._relu(value))
        pair_embeddings.append(tuple(pair))
        pair_contributions.append(base._dense(parameters["pair_projection"], pair, c))
    contributions = (*line_contributions, *pair_contributions)
    if any(abs(value) > MAX_CONTEXT_OPERAND for contribution in contributions for value in contribution):
        raise ValueError("V2 context contribution exceeds 2^70 operand bound")
    context = tuple(base._exact_sum_fp32(
        (original["context_bias"][column], *(contribution[column] for contribution in contributions)), 256,
    ) for column in range(c))
    if any(abs(value) >= MAX_CONTEXT_MAGNITUDE for value in context):
        raise ValueError("V2 context sum exceeds strict 2^79 bound")
    return context, tuple(representations[record.id] for record in inputs.context_lines), line_contributions, pair_embeddings, pair_contributions


def evaluate(checkpoint, inputs):
    if type(checkpoint) is not NnuecV2Checkpoint or type(inputs) is not ReferenceInput:
        raise ValueError("validated V2 checkpoint and V2 input required")
    inputs = ReferenceInput.from_mapping(inputs.to_mapping(), checkpoint.base.config)
    context, typed, line_contributions, pair_embeddings, pair_contributions = _context(checkpoint, inputs)
    original = dict(zip(base.PARAMETER_FIELDS, checkpoint.base.parameters, strict=True))
    board = tuple(base._sparse(original, features, checkpoint.base.config.board_width,
                               "board_weights", "board_bias", exact_board=True)
                  for features in inputs.board_features)
    local_representation = base._encode_line(checkpoint.base, inputs.local_line)
    local = base._dense(original["local_projection"], local_representation,
                        checkpoint.base.config.local_width, original["local_bias"])
    side = 0 if inputs.side_to_move == "white" else 1
    dense_input = tuple(base._clip(value) for group in (
        board[side], board[1 - side], context, local,
    ) for value in group)
    hidden = tuple(base._relu(value) for value in base._dense(
        original["hidden_weights"], dense_input, checkpoint.base.config.hidden_width, original["hidden_bias"],
    ))
    raw = base._dense(original["output_weights"], hidden, base.OUTPUT_ROWS, original["output_bias"])
    maximum = max(raw[1:4])
    exponentials = tuple(base.f32(math.exp(base._add(value, -maximum))) for value in raw[1:4])
    denominator = base._add(base._add(exponentials[0], exponentials[1]), exponentials[2])
    wdl = tuple(base.f32(value / denominator) for value in exponentials)
    return {"schema": OUTPUT_SCHEMA, "semantics": SEMANTICS, "weights_sha256": checkpoint.weights_sha256,
            "context_sha256": inputs.context_sha256, "reference_scope": REFERENCE_SCOPE,
            "board_accumulators": [list(row) for row in board],
            "context_accumulator": list(context), "context_contributions": [list(row) for row in line_contributions],
            "typed_line_representations": [list(row) for row in typed],
            "pair_embeddings": [list(row) for row in pair_embeddings],
            "pair_contributions": [list(row) for row in pair_contributions],
            "local_representation": list(local_representation), "local_accumulator": list(local),
            "hidden": list(hidden), "raw_outputs": list(raw), "value": base._mul(raw[0], checkpoint.base.output_scale),
            "wdl_logits": list(raw[1:4]), "wdl": list(wdl)}


def reference_json(checkpoint_bytes, input_bytes):
    checkpoint = NnuecV2Checkpoint.from_mapping(base.decode_json(checkpoint_bytes))
    inputs = ReferenceInput.from_mapping(base.decode_json(input_bytes), checkpoint.base.config)
    output = evaluate(checkpoint, inputs)
    output["input_bytes_sha256"] = hashlib.sha256(input_bytes).hexdigest()
    return json.dumps(output, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8") + b"\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", help="explicit V2 checkpoint with its declared V1 base")
    parser.add_argument("input", help="Rules-checked V2 record/relation input JSON")
    arguments = parser.parse_args(argv)
    try:
        result = reference_json(base._read_bounded(arguments.checkpoint), base._read_bounded(arguments.input))
    except (OSError, ValueError) as error:
        parser.exit(2, f"NNUEC V2 reference rejected: {error}\n")
    sys.stdout.buffer.write(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
