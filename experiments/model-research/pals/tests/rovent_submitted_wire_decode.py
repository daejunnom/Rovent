"""Isolated strict consumer of actual Rust-serde submitted-family operands.

Only input/container shape and original family hashes are checked. There is no
corpus/asset file read, native witness, Rules replay, neural runtime or trainer.
The empty typed index is deliberately not an actual published corpus.
"""
from __future__ import annotations

import importlib.abc
import json
from pathlib import Path
import sys

MAX_INPUT_BYTES = 64 << 10
FIXTURE_DOMAIN = "rz-rovent-submitted-wire-fixture/1"
RESULT_DOMAIN = "rz-rovent-submitted-wire-result/1"
NEURAL_ROOTS = frozenset(("torch", "numpy", "onnx", "onnxruntime", "tensorflow", "jax"))


class _NoNeuralImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] in NEURAL_ROOTS:
            raise ImportError("neural imports are forbidden in the submitted wire decoder")
        return None


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate submitted wire object field")
        result[key] = value
    return result


def _nonfinite(_value):
    raise ValueError("nonfinite submitted wire JSON number")


def main():
    if not (sys.flags.isolated and sys.flags.no_site and sys.flags.dont_write_bytecode):
        raise RuntimeError("submitted wire decoder requires -I -S -B isolation")
    if any(name.split(".", 1)[0] in NEURAL_ROOTS for name in sys.modules):
        raise RuntimeError("submitted wire decoder began with an imported neural runtime")
    sys.meta_path.insert(0, _NoNeuralImports())
    source = Path(__file__).resolve().parent.parent / "src"
    sys.path.insert(0, str(source))
    raw = sys.stdin.buffer.read(MAX_INPUT_BYTES + 1)
    if not raw or len(raw) > MAX_INPUT_BYTES:
        raise ValueError("finite nonempty submitted wire fixture required")
    fixture = json.loads(raw.decode("utf-8", "strict"), object_pairs_hook=_unique_object,
                         parse_constant=_nonfinite)
    from rz_pals_model.generation_data import digest, fields, sha, uint
    from rz_pals_model.rovent_submitted_input import validate_input, seal_input
    from rz_pals_model.rovent_corpus import CorpusLimits, _manifest
    from rz_pals_model.rovent_corpus_codec import (SUBMITTED_CODEC, validate_container_pair,
                                                   validate_record_codec)

    fields(fixture, ("schema", "classification", "execution_authorized", "input", "manifest", "index",
                     "expected_input_sha256", "expected_manifest_sha256", "expected_index_sha256", "source_domains"),
           "actual typed Rust submitted fixture")
    if (fixture["schema"] != FIXTURE_DOMAIN
            or fixture["classification"] != "synthetic_wire_only_no_capture_authority"
            or fixture["execution_authorized"] is not False):
        raise ValueError("explicit synthetic wire-only classification required")
    snapshot = validate_input(fixture["input"])
    input_sha = seal_input(snapshot)
    if input_sha != sha(fixture["expected_input_sha256"]):
        raise ValueError("Rust/Python original submitted FP32 input seals differ")
    manifest, index = fixture["manifest"], fixture["index"]
    if validate_container_pair(manifest, index) is not SUBMITTED_CODEC:
        raise ValueError("submitted manifest2/index2 require their original domains")
    _manifest(manifest, CorpusLimits(), codec=SUBMITTED_CODEC)
    if (manifest["model_epoch"] != snapshot["model_epoch"]
            or manifest["context_epoch"] != snapshot["context_epoch"]
            or manifest["opening_splits"].get(snapshot["opening_id"]) != "train"):
        raise ValueError("typed input/container namespace differs")
    manifest_sha = digest(SUBMITTED_CODEC.manifest_domain, manifest)
    fields(index, ("schema", "manifest_sha256", "record_count", "chunks", "origins"), "typed empty index2")
    if (sha(index["manifest_sha256"]) != manifest_sha or uint(index["record_count"]) != 0
            or type(index["chunks"]) is not list or index["chunks"]
            or type(index["origins"]) is not list or index["origins"]):
        raise ValueError("wire index must remain an empty synthetic metadata operand")
    index_sha = digest(SUBMITTED_CODEC.index_domain, index)
    if (manifest_sha != sha(fixture["expected_manifest_sha256"])
            or index_sha != sha(fixture["expected_index_sha256"])):
        raise ValueError("Rust/Python original manifest2/index2 seals differ")
    domains = fields(fixture["source_domains"], ("record_domain", "native_raw_domain"), "source family metadata")
    # This deliberately incomplete local object checks family routing only.
    # It is never passed to a row/native/content validator or published.
    probe = {"schema": domains["record_domain"], "input": snapshot,
        "native": {name: {"schema": domains["native_raw_domain"]}
                   for name in ("raw_submission", "raw_completion", "raw_retirement")}}
    if validate_record_codec(manifest, probe) is not SUBMITTED_CODEC:
        raise ValueError("source record2/raw3 family changed")
    mapping = snapshot["attack_mapping"]
    if len(mapping) != 128 or [entry["attack_ply"] for entry in mapping] != list(range(1, 129)):
        raise ValueError("wire fixture must preserve all 128 proposal-relative attack slots")
    if any(name.split(".", 1)[0] in NEURAL_ROOTS for name in sys.modules):
        raise RuntimeError("submitted wire decoder imported a neural runtime")
    result = {"schema": RESULT_DOMAIN, "status": "accepted_wire_only", "input_sha256": input_sha,
        "manifest_sha256": manifest_sha, "index_sha256": index_sha, "attack_slots": len(mapping),
        "visible_counter": mapping[0]["selected_counter_local_index"] is not None,
        "execution_authorized": False, "original_capture_authority": False, "rules_replayed": False,
        "native_witness_validated": False, "neural_imports": False}
    sys.stdout.write(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        sys.stderr.write((type(error).__name__ + ": " + str(error))[:4096] + "\n")
        raise SystemExit(1)
