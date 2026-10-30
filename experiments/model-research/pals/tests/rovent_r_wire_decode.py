"""Isolated strict consumer of Rust-serde synthetic actual-R wire operands.

The Rust test supplies real typed DTO serialization. This finite entry invokes
the production data validators, but does not open a corpus/asset, replay Rules,
load weights, create a model, or issue original capture/trainer authority.
"""
from __future__ import annotations

import importlib.abc
import json
from pathlib import Path
import sys

MAX_INPUT_BYTES = 16 << 20
FIXTURE_DOMAIN = "rz-rovent-r-wire-fixture/1"
RESULT_DOMAIN = "rz-rovent-r-wire-result/1"
CLASSIFICATION = "synthetic_wire_only_no_capture_authority"
NEURAL_ROOTS = frozenset(("torch", "numpy", "onnx", "onnxruntime", "tensorflow", "jax"))


class _NoNeuralImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] in NEURAL_ROOTS:
            raise ImportError("neural imports are forbidden in the R wire decoder")
        return None


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate R wire object field")
        result[key] = value
    return result


def _nonfinite(_value):
    raise ValueError("nonfinite R wire JSON number")


def main():
    if not (sys.flags.isolated and sys.flags.no_site and sys.flags.dont_write_bytecode):
        raise RuntimeError("R wire decoder requires -I -S -B isolation")
    if any(name.split(".", 1)[0] in NEURAL_ROOTS for name in sys.modules):
        raise RuntimeError("R wire decoder began with an imported neural runtime")
    sys.meta_path.insert(0, _NoNeuralImports())
    source = Path(__file__).resolve().parent.parent / "src"
    sys.path.insert(0, str(source))
    raw = sys.stdin.buffer.read(MAX_INPUT_BYTES + 1)
    if not raw or len(raw) > MAX_INPUT_BYTES:
        raise ValueError("finite nonempty R wire fixture input required")
    fixture = json.loads(raw.decode("utf-8", "strict"), object_pairs_hook=_unique_object,
                         parse_constant=_nonfinite)
    from rz_pals_model.generation_data import fields, sha
    from rz_pals_model import rovent_corpus as pc
    from rz_pals_model import rovent_r_corpus as corpus
    from rz_pals_model import rovent_r_readout as wire
    from rz_pals_model.nnuec_reference import NnuecConfig
    from rz_pals_model.rovent_historical_corpus import _input_lock

    fields(fixture, ("schema", "classification", "manifest", "records", "origins", "input_locks",
                     "nnuec_config", "edge_payload", "edge_sha256"), "typed Rust R wire fixture")
    if fixture["schema"] != FIXTURE_DOMAIN or fixture["classification"] != CLASSIFICATION:
        raise ValueError("explicit synthetic R wire-only classification required")
    manifest = fields(fixture["manifest"], ("schema", "collection", "pc_ancestor_index_sha256", "r_ancestor_index_sha256"),
                      "typed Rust R manifest")
    if manifest["schema"] != wire.MANIFEST_DOMAIN:
        raise ValueError("explicit Rust R manifest domain required")
    pc._manifest(manifest["collection"], pc.CorpusLimits())
    for key in ("pc_ancestor_index_sha256", "r_ancestor_index_sha256"):
        pc._sha_list(manifest[key], 256, key)
    config = NnuecConfig.from_mapping(fixture["nnuec_config"])
    records, origins, declarations = fixture["records"], fixture["origins"], fixture["input_locks"]
    if (type(records) is not list or len(records) != 2 or type(origins) is not list or len(origins) != 2
            or type(declarations) is not dict or not 1 <= len(declarations) <= 2):
        raise ValueError("R wire fixture needs two typed records/origins and bounded lock declarations")
    locks = {sha(key): _input_lock(value, key) for key, value in declarations.items()}
    if type(records[0]) is not dict or type(records[0].get("bound")) is not dict:
        raise ValueError("original typed R Bound required")
    # This expected metadata is a wire fixture operand, never proof that the
    # corresponding weights were loaded or that a native owner executed.
    expected_parameter = sha(records[0]["bound"].get("parameter_sha256"))
    samples = [corpus._validate_record(record, manifest, origin, locks, config, expected_parameter)
               for record, origin in zip(records, origins, strict=True)]
    sequences = [record["question"]["readout_sequence"] for record in records]
    if sequences != [0, 2] or records[0]["retirement_sha256"] != records[1]["retirement_sha256"]:
        raise ValueError("Rust R fixture changed original sampled attempts or retirement projection")
    if (records[0]["retirement"]["retained_readouts"] != len(records)
            or samples[0]["input_sha256"] != samples[1]["input_sha256"]
            or samples[0]["record_sha256"] == samples[1]["record_sha256"]
            or any(record["retirement"]["ending"] != {"kind": "unknown"} for record in records)
            or any(record["wdl"] is not None or record["semantic_consumption"] is not None
                   or record["whole_action_cost"] is not None for record in records)):
        raise ValueError("Rust R fixture must preserve repeated observations and unknown authority/targets")
    edge = wire.r_readout_f32_digest(wire.FLOAT_BITS_DOMAIN, fixture["edge_payload"])
    if edge != sha(fixture["edge_sha256"]):
        raise ValueError("Rust/Python R independent FP32 framing differs")
    if any(name.split(".", 1)[0] in NEURAL_ROOTS for name in sys.modules):
        raise RuntimeError("R wire decoder imported a neural runtime")
    result = {"schema": RESULT_DOMAIN, "status": "accepted_wire_only", "records": len(records),
        "readout_sequences": sequences, "edge_sha256": edge, "original_capture_authority": False,
        "rules_replayed": False, "execution_authorized": False, "neural_imports": False}
    sys.stdout.write(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        sys.stderr.write((type(error).__name__ + ": " + str(error))[:4096] + "\n")
        raise SystemExit(1)
