"""Explicit collection/trainer identities for a new Rovent window codec.

This module validates metadata only. It neither converts a /1 window nor
authorizes a process, materializes tensors or proves a loaded runtime. The
original /1 parser and canonical bytes remain separate. Live consumers must
join collection identities to the retained corpus and trainer identities to
the new process's source, executable and declared dependency endpoint pins.
"""
from __future__ import annotations

import math
import struct

from .generation_data import digest, fields, identifier, sha, uint
from .generation_training import OPTIMIZER, _typed_equal
from .rovent_generation import SeriesRecipe

WINDOW_DOMAIN = "rz-rovent-train-window/2"
PROFILE_DOMAIN = "rz-rovent-fixed-training-profile/2"
DEPENDENCY_DOMAIN = "rz-rovent-trainer-dependency-set/1"
DEPENDENCY_SCOPE = "declared_pinned_endpoints_only"
MAX_OUTPUT_BYTES = 2 * 1024**3
HEADS = ("policy", "wdl", "divergence", "task")
COLLECTION_FIELDS = ("collection_source_sha256", "collection_runtime_library_sha256",
    "collection_model_semantics_sha256", "collection_model_epoch", "collection_context_epoch")
TRAINER_FIELDS = ("trainer_source_sha256", "trainer_python_executable_sha256", "trainer_dependency_set_sha256",
    "trainer_dependency_scope", "training_model_semantics_sha256", "training_model_epoch")
CONTROL_FIELDS = ("phase", "frozen_pc_sha256", "device", "precision", "tf32", "deterministic",
    "cpu_threads", "interop_threads", "microbatch", "accumulation", "optimizer", "sampling_seed", "training_seed",
    "loss_weights", "nnuec_loss", "max_updates", "max_groups", "max_samples", "max_wall_ns", "max_output_bytes",
    "checkpoint_interval")
WINDOW_FIELDS = (("schema", "window_id", "recipe_sha256", "stage_id") + COLLECTION_FIELDS + TRAINER_FIELDS
    + ("nnuec_collection_model_sha256", "nnuec_semantics") + CONTROL_FIELDS)


def _finite_weight(value, *, positive=False):
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value) and (value > 0 if positive else value >= 0)
    except OverflowError:
        return False


def validate_learning_controls(value):
    """Pure controls shared by explicit codecs, without re-tagging identities."""
    if value["phase"] == "pc":
        if value["frozen_pc_sha256"] is not None:
            raise ValueError("P/C learning cannot borrow a frozen V collection identity")
    elif value["phase"] == "v":
        sha(value["frozen_pc_sha256"])
    else:
        raise ValueError("explicit P/C or newly frozen V collection phase required")
    device = value["device"]
    if type(device) is not str or not (device == "cpu" or device.startswith("cuda:") and device[5:].isascii()
            and device[5:].isdigit() and str(int(device[5:])) == device[5:] and int(device[5:]) < 16):
        raise ValueError("explicit canonical CPU/CUDA device required; no silent fallback")
    if (value["precision"] != "fp32" or value["tf32"] is not False or value["deterministic"] is not True
            or type(value["microbatch"]) is not int or value["microbatch"] != 1
            or type(value["accumulation"]) is not int or value["accumulation"] != 8
            or not _typed_equal(value["optimizer"], OPTIMIZER)):
        raise ValueError("registered FP32/deterministic microbatch1 accumulation8 AdamW recipe required")
    for name in ("cpu_threads", "interop_threads"):
        if not 1 <= uint(value[name], 64):
            raise ValueError("explicit bounded host thread counts required")
    uint(value["sampling_seed"], 2**32 - 1)
    uint(value["training_seed"], 2**32 - 1)
    fields(value["loss_weights"], HEADS, "explicit P/C/V loss weights")
    for weight in value["loss_weights"].values():
        if not _finite_weight(weight):
            raise ValueError("finite nonnegative explicitly registered loss weights required")
    nnuec = fields(value["nnuec_loss"], ("loss", "target", "normalization", "weight"), "explicit NNUEC learning recipe")
    if (nnuec["loss"] != "wdl_cross_entropy" or nnuec["target"] != "same_exact_rules_anchor"
            or nnuec["normalization"] != "valid_rows" or not _finite_weight(nnuec["weight"], positive=True)):
        raise ValueError("NNUEC loss/Rules target/normalization/positive weight must be explicitly resolved")
    for name, maximum in (("max_updates", 65536), ("max_groups", 65536), ("max_samples", 524288),
                          ("max_wall_ns", 24 * 3600 * 10**9), ("max_output_bytes", MAX_OUTPUT_BYTES),
                          ("checkpoint_interval", 250)):
        if not 1 <= uint(value[name], maximum):
            raise ValueError("finite positive window work/time/output/checkpoint bound required")
    if value["max_samples"] < 8 or value["max_groups"] < value["max_updates"]:
        raise ValueError("window cannot fit its explicit group/sample budget")


def validate_window_spec(recipe, value):
    """Validate only /2. A JSON success is never an original window authority."""
    fields(value, WINDOW_FIELDS + (("nnuec_r_auxiliary",) if type(value) is dict and "nnuec_r_auxiliary" in value else ()), "Rovent /2 training window")
    if type(recipe) is not SeriesRecipe or value["schema"] != WINDOW_DOMAIN:
        raise ValueError("explicit Rovent /2 window required; legacy conversion is forbidden")
    identifier(value["window_id"])
    recipe.stage(value["stage_id"])
    registered = recipe.to_dict()
    for name in ("recipe_sha256", "collection_source_sha256", "collection_runtime_library_sha256",
                 "collection_model_semantics_sha256", "trainer_source_sha256", "trainer_python_executable_sha256",
                 "trainer_dependency_set_sha256", "training_model_semantics_sha256", "nnuec_collection_model_sha256"):
        sha(value[name])
    if (value["recipe_sha256"] != recipe.sha256
            or value["collection_source_sha256"] != registered["source_sha256"]
            or value["training_model_semantics_sha256"] != registered["model_semantics_sha256"]):
        raise ValueError("window changed registered recipe, collection source or training model semantics")
    if value["collection_model_semantics_sha256"] != value["training_model_semantics_sha256"]:
        raise ValueError("this codec requires explicitly equal collection/training model semantics")
    if value["trainer_dependency_scope"] != DEPENDENCY_SCOPE:
        raise ValueError("declared dependency endpoints cannot claim transitive loaded runtime coverage")
    for name in ("collection_model_epoch", "collection_context_epoch", "training_model_epoch"):
        uint(value[name])
    identifier(value["nnuec_semantics"])
    validate_learning_controls(value)
    if "nnuec_r_auxiliary" in value:
        from .rovent_r_auxiliary import validate_auxiliary_recipe
        validate_auxiliary_recipe(value["nnuec_r_auxiliary"], recipe=recipe)
    return value


def validate_collection_binding(recipe, spec, manifest):
    """Join a separately validated corpus to collection fields, never Python.

    This is a projection comparison, not a corpus parser or historical native
    capability. Its caller must have validated the exact manifest and rows.
    """
    validate_window_spec(recipe, spec)
    if type(manifest) is not dict:
        raise ValueError("validated collection manifest object required")
    expected = {"recipe_sha256": spec["recipe_sha256"], "stage_id": spec["stage_id"],
        "source_sha256": spec["collection_source_sha256"],
        "runtime_sha256": spec["collection_runtime_library_sha256"],
        "model_semantics_sha256": spec["collection_model_semantics_sha256"],
        "model_epoch": spec["collection_model_epoch"], "context_epoch": spec["collection_context_epoch"]}
    if any(name not in manifest or not _typed_equal(manifest[name], item) for name, item in expected.items()):
        raise ValueError("collection manifest differs from its collection namespace")
    return {"schema": "rz-rovent-window-collection-binding/2", "collection": expected,
        "past_native_authority_reissued": False, "execution_authorized": False}


def trainer_dependency_set_digest(value):
    """Canonical declared endpoints only; physical paths are joined elsewhere."""
    if type(value) is not list or not 2 <= len(value) <= 64:
        raise ValueError("bounded declared Torch endpoint inventory required")
    names = []
    for item in value:
        fields(item, ("name", "sha256", "bytes"), "declared trainer dependency endpoint")
        names.append(identifier(item["name"]))
        sha(item["sha256"])
        if not uint(item["bytes"]):
            raise ValueError("positive original dependency file byte pin required")
    if (names != sorted(names) or len(set(names)) != len(names)
            or not {"torch.__file__", "torch._C.__file__"} <= set(names)):
        raise ValueError("sorted unique logical endpoints must include both actual Torch module files")
    return digest(DEPENDENCY_DOMAIN, value)


def validate_trainer_binding(recipe, spec, *, source_sha256, python_executable_sha256, dependency_endpoints):
    """Compare declared pins after the live caller independently observes files.

    No executable or dependency file is opened here. The source inventory,
    actual loaded Python image, module endpoints and FD identities must be
    owned and checked by the new process; metadata agreement is insufficient.
    """
    validate_window_spec(recipe, spec)
    sha(source_sha256)
    sha(python_executable_sha256)
    dependencies = trainer_dependency_set_digest(dependency_endpoints)
    if (spec["trainer_source_sha256"] != source_sha256
            or spec["trainer_python_executable_sha256"] != python_executable_sha256
            or spec["trainer_dependency_set_sha256"] != dependencies):
        raise ValueError("trainer source, Python executable or dependency endpoints differ from the window")
    return {"schema": "rz-rovent-window-trainer-binding/2", "trainer_source_sha256": source_sha256,
        "trainer_python_executable_sha256": python_executable_sha256, "trainer_dependency_set_sha256": dependencies,
        "dependency_scope": DEPENDENCY_SCOPE, "transitive_loaded_native_mappings": "unknown",
        "past_native_authority_reissued": False, "execution_authorized": False}


def training_profile_digest(spec):
    """A new profile domain; do not pass /2 fields to the /1 canonicalizer."""
    fields(spec, WINDOW_FIELDS + (("nnuec_r_auxiliary",) if type(spec) is dict and "nnuec_r_auxiliary" in spec else ()), "Rovent /2 training window profile")
    if spec["schema"] != WINDOW_DOMAIN:
        raise ValueError("Rovent /2 fixed training profile requires a /2 window")
    validate_learning_controls(spec)
    renewal = {"window_id", "max_updates", "max_groups", "max_samples", "max_wall_ns", "max_output_bytes",
        "checkpoint_interval", "phase", "collection_model_epoch", "training_model_epoch", "frozen_pc_sha256",
        "nnuec_collection_model_sha256"}
    normalized = {name: value for name, value in spec.items() if name not in renewal}
    def coefficient(value):
        if type(value) in (int, float):
            try:
                numeric = float(value)
                if not math.isfinite(numeric):
                    raise ValueError("finite f64 profile coefficient required")
                return struct.pack(">d", numeric).hex()
            except (OverflowError, struct.error) as error:
                raise ValueError("finite f64 profile coefficient required") from error
        if type(value) is list:
            return [coefficient(item) for item in value]
        if type(value) is dict:
            return {name: coefficient(item) for name, item in value.items()}
        return value
    normalized["optimizer"] = coefficient(spec["optimizer"])
    normalized["loss_weights"] = coefficient(spec["loss_weights"])
    normalized["nnuec_loss"] = {**spec["nnuec_loss"], "weight": coefficient(spec["nnuec_loss"]["weight"])}
    if spec.get("nnuec_r_auxiliary") is not None:
        from .rovent_r_auxiliary import validate_auxiliary_recipe
        auxiliary = validate_auxiliary_recipe(spec["nnuec_r_auxiliary"])
        normalized["nnuec_r_auxiliary"] = {**auxiliary, "weight": coefficient(auxiliary["weight"])}
    return digest(PROFILE_DOMAIN, {"algorithm": "coefficient-f64-bits-v1", "window": normalized})
