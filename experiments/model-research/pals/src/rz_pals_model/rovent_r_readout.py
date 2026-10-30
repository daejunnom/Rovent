"""Separate actual-R observation seals; no P/C/native authority conversion.

The binary framing is an explicit new ABI. Original Rust JSON bytes, parsed
FP32 input meaning, and raw checkpoint-file identity are different pins. This
module never runs a model, reconstructs Rules, or grants a training lease.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import struct

from .generation_data import fields, identifier, sha, uint

RECORD_DOMAIN = "rz-rovent-r-readout-record/1"
MANIFEST_DOMAIN = "rz-rovent-r-readout-corpus-manifest/1"
INDEX_DOMAIN = "rz-rovent-r-readout-corpus-index/1"
ORIGIN_DOMAIN = "rz-rovent-r-readout-origin/1"
INPUT_DOMAIN = "rz-rovent-r-readout-semantic-input/1"
FLOAT_BITS_DOMAIN = "rz-rovent-r-readout-f32-framing/1"
CAPTURE_INPUT_DOMAIN = "rz-nnuec-actual-r-readout-input/1"
MAX_READOUT_BYTES = 16 * 1024**2
MAX_DEPTH = 64


def r_readout_f32_digest(domain, value):
    """Hash the Rust R ABI's tags, little-endian lengths and FP32 bit words.

    ``generation_data.digest`` is deliberately not changed. Its existing JSON
    domain remains material to old records. A JSON floating 1.0 is distinct
    from integer 1 here; object keys use UTF8 byte order, not locale order.
    """
    identifier(domain)
    state = hashlib.sha256(domain.encode("utf-8") + b"\0")
    # A compact JSON integer can expand to a tag plus eight binary bytes.
    # This work bound deliberately exceeds every admitted 16MiB R record's
    # framing; it is not a second, narrower input ABI limit.
    remaining = [MAX_READOUT_BYTES * 16]

    def update(raw):
        remaining[0] -= len(raw)
        if remaining[0] < 0:
            raise ValueError("R semantic framing byte capacity exceeded")
        state.update(raw)

    def length(count):
        update(uint(count).to_bytes(8, "little"))

    def walk(item, depth):
        if depth > MAX_DEPTH:
            raise ValueError("R semantic input recursion bound")
        kind = type(item)
        if item is None:
            update(b"\x00")
        elif kind is bool:
            update(b"\x02" if item else b"\x01")
        elif kind is int:
            if item >= 0:
                update(b"\x03" + uint(item).to_bytes(8, "little"))
            else:
                if item < -(1 << 63):
                    raise ValueError("R semantic signed integer outside i64")
                update(b"\x04" + (-item).to_bytes(8, "little"))
        elif kind is float:
            if not math.isfinite(item):
                raise ValueError("R semantic input must be finite FP32")
            try:
                raw = struct.pack("<f", item)
            except (OverflowError, struct.error) as error:
                raise ValueError("R semantic input outside finite FP32") from error
            if not math.isfinite(struct.unpack("<f", raw)[0]):
                raise ValueError("R semantic input must be finite FP32")
            update(b"\x05" + raw)
        elif kind is str:
            try:
                raw = item.encode("utf-8")
            except UnicodeEncodeError as error:
                raise ValueError("R semantic UTF8 string required") from error
            update(b"\x06")
            length(len(raw))
            update(raw)
        elif kind is list:
            update(b"\x07")
            length(len(item))
            for child in item:
                walk(child, depth + 1)
        elif kind is dict:
            if any(type(key) is not str for key in item):
                raise ValueError("R semantic object keys must be UTF8 strings")
            try:
                ordered = sorted((key.encode("utf-8"), child) for key, child in item.items())
            except UnicodeEncodeError as error:
                raise ValueError("R semantic object key must be UTF8") from error
            update(b"\x08")
            length(len(ordered))
            for raw_key, child in ordered:
                length(len(raw_key))
                update(raw_key)
                walk(child, depth + 1)
        else:
            raise ValueError("R semantic framing requires plain JSON values")

    walk(value, 0)
    return state.hexdigest()


def _object(pairs):
    result = {}
    for name, value in pairs:
        if name in result:
            raise ValueError("R original JSON contains duplicate object keys")
        result[name] = value
    return result


def _constant(_text):
    raise ValueError("R original JSON contains a nonfinite constant")


def _original_json(text):
    if type(text) is not str:
        raise ValueError("R original Rust UTF8 serialization required")
    try:
        raw = text.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ValueError("R original serialization is not UTF8") from error
    if not 0 < len(raw) <= MAX_READOUT_BYTES:
        raise ValueError("R original serialized input byte cap")
    try:
        parsed = json.loads(text, object_pairs_hook=_object, parse_constant=_constant)
    except (json.JSONDecodeError, RecursionError) as error:
        raise ValueError("R original serialization is not bounded JSON") from error
    if type(parsed) is not dict:
        raise ValueError("R original reference input must be an object")
    return raw, parsed


def validate_input_evidence(value, bound):
    """Verify three independent seals without reserializing the native input.

    This is byte/ABI validation. Exact Rules replay and original worker custody
    remain separate requirements owned by the Rust dataset/process consumers.
    """
    fields(value, ("reference_input_json", "reference_input_bytes", "reference_input_raw_sha256",
        "native_input_sha256", "semantic_input_sha256"), "actual R original input evidence")
    for name in ("reference_input_raw_sha256", "native_input_sha256", "semantic_input_sha256"):
        sha(value[name])
    raw, parsed = _original_json(value["reference_input_json"])
    if (uint(value["reference_input_bytes"], MAX_READOUT_BYTES) != len(raw)
            or hashlib.sha256(raw).hexdigest() != value["reference_input_raw_sha256"]
            or hashlib.sha256(CAPTURE_INPUT_DOMAIN.encode("ascii") + b"\0" + raw).hexdigest()
                != value["native_input_sha256"]):
        raise ValueError("R original UTF8/raw/native prefix seal mismatch")
    identifier(bound["model_semantics"])
    sha(bound["parameter_sha256"])
    for key in ("collection_model_epoch", "collection_context_epoch"):
        if not uint(bound[key]):
            raise ValueError("R original positive model/context epoch required")
    meaning = {"model_semantics": bound["model_semantics"], "parameter_sha256": bound["parameter_sha256"],
        "collection_model_epoch": bound["collection_model_epoch"],
        "collection_context_epoch": bound["collection_context_epoch"], "reference_input": parsed}
    if r_readout_f32_digest(INPUT_DOMAIN, meaning) != value["semantic_input_sha256"]:
        raise ValueError("R parsed FP32 input meaning differs from its separate semantic seal")
    return copy.deepcopy(parsed)


def record_digest(record):
    """New R record framing; does not accept a P/C training record as R data."""
    if type(record) is not dict or record.get("schema") != RECORD_DOMAIN:
        raise ValueError("explicit separate R readout record domain required")
    return r_readout_f32_digest(RECORD_DOMAIN,
        {name: value for name, value in record.items() if name != "record_sha256"})
