"""Decode actual Rust utility metadata through F's strict production validators.

The caller owns the existing bounded metadata runner. This stdlib-only entry
does not open advertised artifacts, instantiate a cohort/model/optimizer, run
training, issue an authority or claim an old physical completion.
"""
from __future__ import annotations

import importlib.abc
import json
from pathlib import Path
import sys

MAX_INPUT_BYTES = 65536
FIXTURE_DOMAIN = "rz-rovent-utility-metadata-cross-language-fixture/1"
RESULT_DOMAIN = "rz-rovent-utility-metadata-cross-language-result/1"
NEURAL_ROOTS = frozenset(("torch", "numpy", "onnx", "onnxruntime", "ort", "tensorflow", "jax", "cupy"))


class _NoNeuralImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] in NEURAL_ROOTS:
            raise ImportError("neural imports are forbidden in the utility metadata wire decoder")
        return None


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate utility metadata object field")
        result[key] = value
    return result


def _nonfinite(_value):
    raise ValueError("nonfinite utility metadata number")


def main():
    if not (sys.flags.isolated and sys.flags.no_site and sys.flags.dont_write_bytecode):
        raise RuntimeError("utility metadata decoder requires -I -S -B isolation")
    if any(name.split(".", 1)[0] in NEURAL_ROOTS for name in sys.modules):
        raise RuntimeError("utility metadata decoder began with an imported neural runtime")
    sys.meta_path.insert(0, _NoNeuralImports())
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    raw = sys.stdin.buffer.read(MAX_INPUT_BYTES + 1)
    if not raw or len(raw) > MAX_INPUT_BYTES:
        raise ValueError("finite nonempty utility metadata fixture required")
    fixture = json.loads(raw.decode("utf-8", "strict"), object_pairs_hook=_unique_object, parse_constant=_nonfinite)
    from rz_pals_model.rovent_training_bootstrap import _digest
    from rz_pals_model.rovent_utility_training import (_fields, _sha, COHORT_PREPARATION_DOMAIN,
        validate_training_preparation, recipe_sha256, parameter_spec_sha256)
    from rz_pals_model.rovent_utility_training_bootstrap import REGISTRATION_DOMAIN, validate_registration
    from rz_pals_model.rovent_utility_training_window import validate_window_spec, utility_window_digest
    from rz_pals_model.rovent_utility_training_authority import COST_DOMAIN, validate_utility_checkpoint_cost

    digest_names = ("recipe", "source_registration", "parameter_spec", "preparation", "window",
                    "execution_registration", "checkpoint_cost")
    _fields(fixture, ("schema", "registration", "preparation", "window", "checkpoint_cost")
        + tuple("expected_" + name + "_sha256" for name in digest_names), "actual Rust utility metadata fixture")
    if fixture["schema"] != FIXTURE_DOMAIN:
        raise ValueError("explicit Rust utility metadata fixture domain required")
    registration = validate_registration(fixture["registration"])
    preparation = validate_training_preparation(fixture["preparation"])
    if preparation["schema"] != COHORT_PREPARATION_DOMAIN:
        raise ValueError("this utility admission lane requires the original cohort V2 preparation")
    window = validate_window_spec(preparation, fixture["window"])
    expense = validate_utility_checkpoint_cost(fixture["checkpoint_cost"])
    source = preparation["registration"]
    if (registration["trainer_source_sha256"] != source["trainer_source_sha256"]
            or window["device"] != registration["resources_policy"]["device"]
            or window["max_read_bytes"] > registration["max_child_read_bytes"]
            or window["max_output_bytes"] > registration["max_output_bytes"]):
        raise ValueError("utility source/window declarations changed original trainer/resource/read/output pins")
    actual = {
        "recipe": recipe_sha256(preparation["recipe"]),
        "source_registration": preparation["registration_sha256"],
        "parameter_spec": parameter_spec_sha256(preparation["parameter_spec"], source["frozen_base"]["config"],
                                               preparation["recipe"]["optimizer_owner"]),
        "preparation": preparation["preparation_sha256"],
        "window": utility_window_digest(window),
        "execution_registration": _digest(REGISTRATION_DOMAIN, registration),
        "checkpoint_cost": _digest(COST_DOMAIN, expense),
    }
    for name in digest_names:
        if actual[name] != _sha(fixture["expected_" + name + "_sha256"]):
            raise ValueError("Rust and independent Python utility canonical digest differ: " + name)
    if any(name.split(".", 1)[0] in NEURAL_ROOTS for name in sys.modules):
        raise RuntimeError("utility metadata decoder imported a neural runtime")
    result = {"schema": RESULT_DOMAIN, "status": "accepted_metadata_only",
        **{name + "_sha256": actual[name] for name in digest_names}, "performed_optimizer_updates": 0,
        "past_native_authority_reissued": False, "execution_authorized": False, "neural_imports": False}
    sys.stdout.write(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        sys.stderr.write((type(error).__name__ + ": " + str(error))[:4096] + "\n")
        raise SystemExit(1)
