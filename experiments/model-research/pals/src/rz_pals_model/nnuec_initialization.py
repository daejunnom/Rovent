"""Explicit fresh NNUEC V2 initialization, without Torch or a forward call.

Configuration, seed, scale, original source commit and finite output/time bounds
are required. The writer creates a new external directory and never reads an
old checkpoint or changes diagnostic/learned eligibility. The existing V1/V2
checkpoint validators supply the parameter contract and semantic weights hash.
The receipt records initialization only; actual Rust loading/registration,
numeric acceptance, Collection origin and training remain separate operations.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import stat
import struct
import time

from . import nnuec_reference as base
from . import nnuec_v2_reference as relation
from .artifacts import publish_noreplace


INITIALIZATION_DOMAIN = "rz-nnuec-v2-initialization/1"
ALGORITHM = "splitmix64-signed24-dyadic-fp32/1"
RECEIPT_FILENAME = "nnuec-initialization.json"
WEIGHTS_FILENAME = "nnuec.json"
CONFIG_FILENAME = "nnuec-initialization-config.json"
MAX_CONFIG_BYTES = 64 << 10
MAX_RECEIPT_BYTES = 1 << 20
MAX_OUTPUT_BYTES = base.MAX_JSON_BYTES + MAX_CONFIG_BYTES + MAX_RECEIPT_BYTES
_MASK64 = (1 << 64) - 1
SOURCE_FILES = ("__init__.py", "nnuec_initialization.py", "nnuec_reference.py", "nnuec_v2_reference.py", "artifacts.py", "config.py")


def _sha(value, label):
    if type(value) is not str or len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
        raise ValueError(f"{label}: lowercase SHA-256 required")
    return value


def _source_commit(value):
    if type(value) is not str or len(value) != 40 or any(ch not in "0123456789abcdef" for ch in value):
        raise ValueError("explicit source commit required")
    return value


def _strict_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate initialization JSON field")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError("nonfinite initialization JSON scalar")


def _json(raw):
    return json.loads(raw.decode("utf-8", "strict"), object_pairs_hook=_strict_pairs, parse_constant=_reject_constant)


def _plain_chain(path, *, directory=False):
    path = Path(path)
    if not path.is_absolute():
        raise ValueError("absolute original initialization path required")
    for item in (*reversed(path.parents), path):
        details = item.lstat()
        if stat.S_ISLNK(details.st_mode) or getattr(details, "st_file_attributes", 0) & 0x400:
            raise ValueError("initialization paths cannot traverse links or reparse points")
        if item != path and not stat.S_ISDIR(details.st_mode):
            raise ValueError("initialization parent is not a directory")
    details = path.lstat()
    if directory and not stat.S_ISDIR(details.st_mode):
        raise ValueError("initialization directory required")
    if not directory and (not stat.S_ISREG(details.st_mode) or details.st_nlink != 1):
        raise ValueError("single-link regular initialization input required")
    return path, details


def _read_original(path, maximum):
    path, before = _plain_chain(path)
    if not 0 < before.st_size <= maximum:
        raise ValueError("original initialization input byte cap")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    with os.fdopen(os.open(path, flags), "rb") as source:
        opened = os.fstat(source.fileno())
        identity = lambda details: (details.st_dev, details.st_ino, details.st_size, details.st_mtime_ns, details.st_nlink)
        if identity(opened) != identity(before):
            raise ValueError("initialization input changed before reading")
        raw = source.read(maximum + 1)
        if len(raw) != opened.st_size or len(raw) > maximum or identity(os.fstat(source.fileno())) != identity(opened):
            raise ValueError("initialization input changed while reading")
    return raw


class _Budget:
    def __init__(self, seconds, maximum, clock):
        base._integer(seconds, 1, 3600, "initialization wall seconds")
        base._integer(maximum, 1, MAX_OUTPUT_BYTES, "initialization output bytes")
        self.clock = clock
        self.until = clock() + seconds
        self.maximum = maximum
        self.written = 0

    def check(self):
        if self.clock() >= self.until:
            raise TimeoutError("NNUEC initialization window expired; preserve partial artifacts")

    def charge(self, count):
        self.check()
        if count > self.maximum - self.written:
            raise ValueError("NNUEC initialization output byte credit exhausted; preserve partial artifacts")
        self.written += count


class _SplitMix64:
    """One explicit integer recurrence; no ambient RNG or versioned random API."""
    def __init__(self, seed):
        self.state = base._integer(seed, 0, _MASK64, "initialization seed")

    def next(self):
        self.state = (self.state + 0x9E3779B97F4A7C15) & _MASK64
        value = self.state
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _MASK64
        return value ^ (value >> 31)

    def parameter(self, scale):
        # Signed24 * binary32 scale has at most 48 significant bits: binary64
        # computes the dyadic product/division exactly, then f32 rounds once.
        mantissa = (self.next() >> 40) - (1 << 23)
        return base.f32((mantissa * scale) / (1 << 23))


def _v2_lengths(config):
    e, c = config.line_width, config.context_width
    return dict(zip(relation.PARAMETER_FIELDS, (4 * e, 2 * e, 2 * e, 2 * e, c * e), strict=True))


def _generate(config, seed, scale, output_scale, budget):
    rng = _SplitMix64(seed)
    inventory = []

    def arrays(lengths, prefix):
        result = []
        for name, length in lengths.items():
            values, digest = [], hashlib.sha256()
            for index in range(length):
                if index % 4096 == 0:
                    budget.check()
                value = rng.parameter(scale)
                digest.update(struct.pack("<f", value))
                values.append(value)
            result.append(tuple(values))
            inventory.append({"name": prefix + name, "elements": length, "dtype": "float32",
                              "bytes": 4 * length, "raw_f32le_sha256": digest.hexdigest()})
        return tuple(result)

    original = base.NnuecCheckpoint(config, (("kind", "untrained"),), arrays(config.parameter_lengths(), "base."), output_scale)
    checkpoint = relation.NnuecV2Checkpoint(original, arrays(_v2_lengths(config), "relation."))
    budget.check()
    return checkpoint, inventory


def _write_raw(directory, name, raw, maximum, budget):
    if len(raw) > maximum:
        raise ValueError("initialization publication exceeds its artifact byte cap")
    partial, destination = directory / (name + ".partial"), directory / name
    digest = hashlib.sha256()
    with partial.open("xb") as output:
        for offset in range(0, len(raw), 64 << 10):
            chunk = raw[offset:offset + (64 << 10)]
            budget.charge(len(chunk))
            output.write(chunk)
            digest.update(chunk)
        output.flush()
        os.fsync(output.fileno())
    budget.check()
    publish_noreplace(partial, destination)
    readback = _read_original(destination, maximum)
    if hashlib.sha256(readback).digest() != digest.digest() or readback != raw:
        raise ValueError("initialization publication changed after original write")
    budget.check()
    return {"file": name, "bytes": len(raw), "sha256": digest.hexdigest()}


def _canonical(value):
    return (json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("ascii")


def _source_inventory(budget):
    source_root = Path(__file__).absolute().parent
    result = []
    for name in SOURCE_FILES:
        budget.check()
        raw = _read_original(source_root / name, 4 << 20)
        result.append({"file": name, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()})
    return result


def initialize_nnuec_v2(config_path, *, config_sha256, seed, parameter_scale, output_scale,
                        source_commit, output_directory, maximum_wall_seconds, maximum_output_bytes,
                        execute=False, clock=time.monotonic):
    """Create only a new untrained asset; never accepts an old model as input.

    The caller must additionally supervise the process using its finite window.
    Checks bracket bounded Python work; they do not claim hard preemption of a
    JSON allocation, source read, fsync, or filesystem operation.
    """
    if execute is not True:
        raise ValueError("explicit NNUEC initialization execution required")
    budget = _Budget(maximum_wall_seconds, maximum_output_bytes, clock)
    _source_commit(source_commit)
    base._integer(seed, 0, _MASK64, "initialization seed")
    scale, output_scale = base.f32(parameter_scale), base.f32(output_scale)
    if not 0.0 < scale <= base.MAX_PARAMETER_MAGNITUDE or not 0.0 < output_scale <= 10000.0:
        raise ValueError("positive bounded FP32 parameter/output scales required")
    raw_config = _read_original(config_path, MAX_CONFIG_BYTES)
    if hashlib.sha256(raw_config).hexdigest() != _sha(config_sha256, "original config"):
        raise ValueError("original initialization config SHA-256 mismatch")
    config = base.NnuecConfig.from_mapping(_json(raw_config))
    parameter_count = sum(config.parameter_lengths().values()) + sum(_v2_lengths(config).values())
    reserve = parameter_count * 32 + 131072
    if reserve > base.MAX_JSON_BYTES or reserve + len(raw_config) + MAX_RECEIPT_BYTES > budget.maximum:
        raise ValueError("insufficient conservative weights/config/receipt publication credit")
    source_pins = _source_inventory(budget)
    directory = Path(output_directory)
    if not directory.is_absolute() or directory.name in ("", ".", ".."):
        raise ValueError("new absolute external initialization directory required")
    parent, _ = _plain_chain(directory.parent, directory=True)
    if any((ancestor / ".git").exists() for ancestor in (parent, *parent.parents)):
        raise ValueError("NNUEC initialization outputs must remain outside Git checkouts")
    # exist_ok=False preserves any previous complete or partial invocation.
    budget.check()
    directory.mkdir(mode=0o700)
    budget.check()
    checkpoint, inventory = _generate(config, seed, scale, output_scale, budget)
    config_pin = _write_raw(directory, CONFIG_FILENAME, raw_config, MAX_CONFIG_BYTES, budget)
    mapping = checkpoint.to_mapping()
    # The unchanged independent parser checks every emitted shape/training
    # field and provides a second digest readback; it runs no neural kernel.
    readback = relation.NnuecV2Checkpoint.from_mapping(mapping)
    if readback.weights_sha256 != checkpoint.weights_sha256:
        raise ValueError("NNUEC initializer checkpoint roundtrip changed semantic weights")
    weights = _write_raw(directory, WEIGHTS_FILENAME, _canonical(mapping), base.MAX_JSON_BYTES, budget)
    if _source_inventory(budget) != source_pins:
        raise ValueError("NNUEC initializer source changed during execution; preserve artifacts")
    receipt = {"domain": INITIALIZATION_DOMAIN, "algorithm": ALGORITHM,
        "seed": seed, "parameter_scale_fp32_bits": struct.unpack("<I", struct.pack("<f", scale))[0],
        "output_scale_fp32_bits": struct.unpack("<I", struct.pack("<f", output_scale))[0],
        "configuration": config.to_mapping(), "original_config": config_pin,
        "schema": relation.SCHEMA, "semantics": relation.SEMANTICS,
        "training": {"kind": "untrained"}, "weights": weights,
        "semantic_weights_sha256": checkpoint.weights_sha256,
        "parameter_count": parameter_count, "parameter_inventory": inventory,
        "parameter_order": "base.PARAMETER_FIELDS then relation.PARAMETER_FIELDS; flat row order unchanged",
        "source_commit": source_commit, "source_files": source_pins,
        "software": {"python_version": platform.python_version(), "python_implementation": platform.python_implementation(),
            "fp32_rounding": "IEEE-binary32-struct-pack-round-nearest-even", "ambient_rng_used": False},
        "resource_declaration": {"maximum_wall_seconds": maximum_wall_seconds,
            "maximum_output_bytes": maximum_output_bytes, "weights_byte_cap": base.MAX_JSON_BYTES},
        "claims": {"initialization_executed": True, "neural_forwards": 0, "optimizer_updates": 0,
            "trained": False, "numeric_status": "not_run", "quality_status": "not_run",
            "rust_registration_status": "not_run", "arena_eligible": False,
            "collection_origin_granted": False, "diagnostic_asset_relabelled": False}}
    receipt_pin = _write_raw(directory, RECEIPT_FILENAME, _canonical(receipt), MAX_RECEIPT_BYTES, budget)
    budget.check()
    return {"domain": INITIALIZATION_DOMAIN, "receipt": receipt_pin, "weights": weights,
            "semantic_weights_sha256": checkpoint.weights_sha256, "output_bytes": budget.written,
            "neural_forwards": 0, "optimizer_updates": 0, "collection_origin_granted": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--config-json", required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--parameter-scale", type=float, required=True)
    parser.add_argument("--output-scale", type=float, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--maximum-wall-seconds", type=int, required=True)
    parser.add_argument("--maximum-output-bytes", type=int, required=True)
    args = parser.parse_args(argv)
    result = initialize_nnuec_v2(args.config_json, config_sha256=args.config_sha256,
        seed=args.seed, parameter_scale=args.parameter_scale, output_scale=args.output_scale,
        source_commit=args.source_commit, output_directory=args.output_directory,
        maximum_wall_seconds=args.maximum_wall_seconds, maximum_output_bytes=args.maximum_output_bytes,
        execute=args.execute)
    print(json.dumps(result, ensure_ascii=True, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
