"""Owned external artifacts, initialization provenance and independent export."""
import errno
import hashlib
import json
import os
from pathlib import Path
import stat
import sys

from .config import (DEFAULT_INIT_PROFILE, MAX_LINE_PLIES, MODEL_SEMANTICS_V2, MODEL_SEMANTICS_V3, MODEL_SCHEMA_V3, ModelConfig,
                     SCHEMA, TASKS, public_tensor_names, role_tensor_names)

FULL_LINE_RULES_FEATURES = ["full_line.record_tokens", "full_line.record_lengths", "full_line.record_mask",
                            "full_line.query_tokens", "full_line.query_lengths", "full_line.query_mask",
                            "full_line.record_kind", "full_line.parent_local_index", "full_line.supersedes_local_index"]
FULL_LINE_RULES_BOUNDS = {"max_records": 128, "max_line_plies": 256, "max_candidates": 256}
VIRTUAL_RULES_FEATURES = ["review.anchor_board", "review.anchor_rules_metadata", "review.record_anchor_local",
    "review.record_moves", "review.record_lengths", "review.record_masks", "review.record_kind",
    "review.parent_local", "review.supersedes_local", "review.query_moves", "review.query_lengths",
    "review.query_masks", "review.query_anchor_local", "review.physical_line_bucket"]
VIRTUAL_RULES_BOUNDS = {"max_records": 64, "max_anchors": 4, "max_line_plies": 256,
                      "line_buckets": [8,16,32,64,128,256], "max_candidates": 256,
                      "token_budgets": [128,256,512], "root_anchor": 0}


class ExportWriteBudget:
    """Monotone serialized-write credit for one sequential export invocation.

    Reserve before opening a partial file. Failed writes retain their charge;
    this counter neither refunds partials nor grants a storage lease or quota.
    The caller accounts separately for journals, logs and other metadata.
    """
    __slots__ = ("_maximum_bytes", "_reserved_bytes")

    def __init__(self, maximum_bytes):
        if type(maximum_bytes) is not int or maximum_bytes <= 0:
            raise ValueError("export write budget must be a positive integer")
        self._maximum_bytes, self._reserved_bytes = maximum_bytes, 0

    @property
    def maximum_bytes(self):
        return self._maximum_bytes

    @property
    def reserved_bytes(self):
        return self._reserved_bytes

    def __call__(self, next_bytes):
        if type(next_bytes) is not int or next_bytes < 0:
            raise ValueError("serialized export byte count must be a nonnegative integer")
        if next_bytes > self._maximum_bytes - self._reserved_bytes:
            raise ValueError("export write budget exceeded before file creation")
        self._reserved_bytes += next_bytes


class AtomicPublicationError(OSError):
    """A publication failed validation; already published bytes are retained."""
    def __init__(self, code, message, *, publication_state="not_published"):
        super().__init__(errno.EIO, message)
        self.code = code
        self.publication_state = publication_state


class AtomicPublicationUnsupported(AtomicPublicationError):
    """No known link-free, atomic no-replace primitive is available."""
    def __init__(self, native_errno=None):
        super().__init__("artifact.publication.unsupported",
                         "link-free atomic no-replace publication is unsupported")
        self.native_errno = native_errno


def _publication_identity(details, *, directory=False):
    expected = stat.S_ISDIR if directory else stat.S_ISREG
    if (not expected(details.st_mode) or stat.S_ISLNK(details.st_mode)
            or getattr(details, "st_file_attributes", 0) & 0x400
            or not directory and details.st_nlink != 1):
        raise AtomicPublicationError("artifact.publication.identity", "publication requires a plain owned directory and a single-link regular partial")
    return ((details.st_dev, details.st_ino) if directory else
            (details.st_dev, details.st_ino, details.st_size, details.st_mtime_ns))


def _linux_rename_noreplace(directory_fd, source_name, target_name):
    """The only Linux FFI: libc's typed renameat2, without a syscall fallback."""
    import ctypes
    try:
        rename = ctypes.CDLL(None, use_errno=True).renameat2
    except (AttributeError, OSError) as error:
        raise AtomicPublicationUnsupported(errno.ENOSYS) from error
    rename.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint)
    rename.restype = ctypes.c_int
    ctypes.set_errno(0)
    result = rename(directory_fd, os.fsencode(source_name), directory_fd, os.fsencode(target_name), 1)  # RENAME_NOREPLACE
    if result != 0:
        native_errno = ctypes.get_errno() or errno.EIO
        if native_errno in {errno.ENOSYS, errno.EINVAL, errno.ENOTSUP, errno.EOPNOTSUPP}:
            raise AtomicPublicationUnsupported(native_errno)
        raise OSError(native_errno, os.strerror(native_errno), target_name)


def publish_noreplace(partial, destination):
    """Move a complete, caller-fsynced partial without hard links or overwrite.

    Linux uses one retained parent FD and renameat2(RENAME_NOREPLACE); Windows
    uses os.rename, whose native operation refuses an existing destination.
    Unknown OS/kernel/filesystem support fails closed, with no replace/link
    fallback. The caller retains its partial cleanup policy on failure. A
    post-rename identity/fsync failure retains the final and remains an error.
    This does not grant ownership of ancestors or of concurrently mutated data.
    """
    partial, destination = Path(partial).absolute(), Path(destination).absolute()
    if (partial.parent != destination.parent or partial == destination
            or any(name in ("", ".", "..") or "\x00" in name for name in (partial.name, destination.name))):
        raise ValueError("publication requires distinct names in the same directory")
    directory = partial.parent
    directory_fd = None
    primary_error = None
    publication_state = "not_published"
    try:
        parent_identity = _publication_identity(directory.lstat(), directory=True)
        if sys.platform == "linux":
            directory_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
            if _publication_identity(os.fstat(directory_fd), directory=True) != parent_identity:
                raise AtomicPublicationError("artifact.publication.parent_changed", "publication parent changed before opening")
        elif os.name != "nt":
            raise AtomicPublicationUnsupported()
        original = _publication_identity(partial.lstat())
        if directory_fd is not None:
            if _publication_identity(os.stat(partial.name, dir_fd=directory_fd, follow_symlinks=False)) != original:
                raise AtomicPublicationError("artifact.publication.partial_changed", "publication partial changed before rename")
        if _publication_identity(directory.lstat(), directory=True) != parent_identity:
            raise AtomicPublicationError("artifact.publication.parent_changed", "publication parent changed before rename")
        publication_state = "unknown"
        if directory_fd is not None:
            _linux_rename_noreplace(directory_fd, partial.name, destination.name)
        else:
            os.rename(partial, destination)
        publication_state = "published"
        if directory_fd is not None:
            published = os.stat(destination.name, dir_fd=directory_fd, follow_symlinks=False)
        else:
            published = destination.lstat()
        if _publication_identity(published) != original:
            raise AtomicPublicationError("artifact.publication.partial_changed", "published file identity differs from the completed partial", publication_state="published")
        if _publication_identity(directory.lstat(), directory=True) != parent_identity:
            raise AtomicPublicationError("artifact.publication.parent_changed", "publication parent changed after rename", publication_state="published")
        if directory_fd is not None:
            os.fsync(directory_fd)
    except BaseException as error:
        primary_error = error
        if isinstance(error, (FileExistsError, AtomicPublicationUnsupported)):
            publication_state = "not_published"
        error.publication_state = publication_state
        raise
    finally:
        if directory_fd is not None:
            try:
                os.close(directory_fd)
            except OSError as cleanup_error:
                if primary_error is not None:
                    primary_error.publication_cleanup_error = cleanup_error
                else:
                    error = AtomicPublicationError("artifact.publication.close", "could not close retained publication parent", publication_state=publication_state)
                    raise error from cleanup_error


def validate_virtual_rules_declaration(declaration, config):
    from .virtual_review import RECORD_FEATURES, QUERY_FEATURES
    config.validate()
    if not config.virtual_review or declaration.get("schema") != "rovezero.pals-rules-descriptor.v3" or declaration.get("rules_input_profile") != "rz-pals-rules-virtual-review-v1" or declaration.get("config") != config.to_dict():
        raise ValueError("V3 Rules profile schema/config mismatch")
    if declaration.get("record_features") != list(RECORD_FEATURES) or declaration.get("query_features") != list(QUERY_FEATURES) or declaration.get("virtual_review_features") != VIRTUAL_RULES_FEATURES or declaration.get("bounds") != VIRTUAL_RULES_BOUNDS:
        raise ValueError("V3 Rules selected-review vocabulary/bounds mismatch")
    for key, count in (("metadata_features",16),("divergence_features",8)):
        row = declaration.get(key)
        if type(row) is not list or len(row) != count or any(type(x) is not str or not x for x in row):
            raise ValueError("V3 Rules inherited feature vocabulary")
    if any(not _sha256_text(declaration.get(key)) for key in ("rules_input_semantic_sha256", "rules_encoder_source_sha256")) or declaration.get("learned_input_compatibility") != "unverified_declaration_only":
        raise ValueError("V3 Rules declaration identity/learned claim")
    base = declaration.get("rules_base_semantics")
    expected = ["rz-pals-rules-virtual-review-v1",base] + declaration["metadata_features"] + list(RECORD_FEATURES) + list(QUERY_FEATURES) + declaration["divergence_features"] + VIRTUAL_RULES_FEATURES
    if type(base) is not str or not base or declaration.get("semantic_fields") != expected or declaration.get("semantic_digest_algorithm") != "sha256_u64le_length_prefixed_utf8_fields":
        raise ValueError("V3 Rules semantic field framing")
    hasher = hashlib.sha256()
    for field in expected:
        raw = field.encode("utf-8"); hasher.update(len(raw).to_bytes(8,"little")); hasher.update(raw)
    if hasher.hexdigest() != declaration["rules_input_semantic_sha256"]:
        raise ValueError("V3 Rules semantic digest mismatch")
    return hasher.hexdigest()


def validate_rules_declaration(declaration, config):
    """V2 has a separate semantic extension; the V1 58-field digest stays fixed."""
    config.validate()
    if config.virtual_review:
        return validate_virtual_rules_declaration(declaration, config)
    schema = "rovezero.pals-rules-descriptor.v2" if config.full_line else "rovezero.pals-rules-descriptor.v1"
    profile = "rz-pals-rules-full-line-v2" if config.full_line else "rz-pals-rules-fields-v1"
    if declaration.get("schema") != schema or declaration.get("rules_input_profile") != profile or declaration.get("config") != config.to_dict():
        raise ValueError("Rules profile schema/config mismatch")
    for key in ("rules_input_semantic_sha256", "rules_encoder_source_sha256"):
        if not _sha256_text(declaration.get(key)): raise ValueError("Rules declaration digest missing")
    if declaration.get("learned_input_compatibility") != "unverified_declaration_only":
        raise ValueError("source declaration cannot assert learned compatibility")
    features = []
    for key, count in (("metadata_features", 16), ("record_features", 16), ("query_features", 16), ("divergence_features", 8)):
        values = declaration.get(key)
        if not isinstance(values, list) or len(values) != count or any(not isinstance(value, str) or not value for value in values):
            raise ValueError("Rules feature vocabulary mismatch")
        features += values
    if config.full_line:
        if declaration.get("full_line_features") != FULL_LINE_RULES_FEATURES or declaration.get("bounds") != FULL_LINE_RULES_BOUNDS:
            raise ValueError("Rules full-line vocabulary/bounds mismatch")
        features += FULL_LINE_RULES_FEATURES
    elif "full_line_features" in declaration or "bounds" in declaration:
        raise ValueError("legacy Rules descriptor rejects full-line extension")
    fields = declaration.get("semantic_fields")
    if declaration.get("semantic_digest_algorithm") != "sha256_u64le_length_prefixed_utf8_fields" or not isinstance(fields, list) or len(fields) != (67 if config.full_line else 58) or fields[0] != profile or not isinstance(fields[1], str) or not fields[1] or fields[1] != declaration.get("rules_base_semantics") or fields[2:] != features:
        raise ValueError("Rules semantic fields/canonicalization mismatch")
    semantic = hashlib.sha256()
    for field in fields:
        value = field.encode("utf-8")
        semantic.update(len(value).to_bytes(8, "little"))
        semantic.update(value)
    if semantic.hexdigest() != declaration["rules_input_semantic_sha256"]:
        raise ValueError("Rules semantic digest disagrees with declared fields")
    return semantic.hexdigest()


def digest_file(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _sha256_text(value):
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def read_rules_profile(path, config):
    """Verify the source declaration's semantic fields, not learned suitability."""
    encoded = Path(path).read_bytes()
    if len(encoded) > 65536:
        raise ValueError("Rules profile exceeds bounded declaration size")
    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result: raise ValueError("duplicate Rules profile key")
            result[key] = value
        return result
    declaration = json.loads(encoded, object_pairs_hook=unique_pairs)
    if config.profile != "legacy_summary_v1":
        semantic_digest = validate_rules_declaration(declaration, config)
        canonical = json.dumps(declaration, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        return {"rules_input_profile": declaration["rules_input_profile"], "rules_input_semantic_sha256": semantic_digest,
                "rules_encoder_source_sha256": declaration["rules_encoder_source_sha256"],
                "rules_profile_descriptor_sha256": hashlib.sha256(encoded).hexdigest(),
                "rules_profile_canonical_sha256": hashlib.sha256(canonical).hexdigest(),
                "rules_input_declaration": declaration, "learned_input_compatibility": "unverified_declaration_only"}
    if declaration.get("schema") != "rovezero.pals-rules-descriptor.v1" or declaration.get("rules_input_profile") != "rz-pals-rules-fields-v1" or declaration.get("config") != config.to_dict():
        raise ValueError("Rules profile schema/config mismatch")
    for key in ("rules_input_semantic_sha256", "rules_encoder_source_sha256"):
        if not _sha256_text(declaration.get(key)): raise ValueError("Rules declaration digest missing")
    if declaration.get("learned_input_compatibility") != "unverified_declaration_only":
        raise ValueError("source declaration cannot assert learned compatibility")
    features = []
    for key, count in (("metadata_features", 16), ("record_features", 16), ("query_features", 16), ("divergence_features", 8)):
        values = declaration.get(key)
        if not isinstance(values, list) or len(values) != count or any(not isinstance(value, str) or not value for value in values):
            raise ValueError("Rules feature vocabulary mismatch")
        features += values
    fields = declaration.get("semantic_fields")
    if declaration.get("semantic_digest_algorithm") != "sha256_u64le_length_prefixed_utf8_fields" or not isinstance(fields, list) or len(fields) != 58 or fields[0] != declaration["rules_input_profile"] or not isinstance(fields[1], str) or not fields[1] or fields[1] != declaration.get("rules_base_semantics") or fields[2:] != features:
        raise ValueError("Rules semantic fields/canonicalization mismatch")
    semantic = hashlib.sha256()
    for field in fields:
        value = field.encode("utf-8")
        semantic.update(len(value).to_bytes(8, "little"))
        semantic.update(value)
    if semantic.hexdigest() != declaration["rules_input_semantic_sha256"]:
        raise ValueError("Rules semantic digest disagrees with declared fields")
    canonical = json.dumps(declaration, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return {"rules_input_profile": declaration["rules_input_profile"],
            "rules_input_semantic_sha256": semantic.hexdigest(),
            "rules_encoder_source_sha256": declaration["rules_encoder_source_sha256"],
            "rules_profile_descriptor_sha256": hashlib.sha256(encoded).hexdigest(),
            "rules_profile_canonical_sha256": hashlib.sha256(canonical).hexdigest(),
            "rules_input_declaration": declaration, "learned_input_compatibility": "unverified_declaration_only"}


def output_directory(path, *, output_guard=None):
    """Refuse source-tree weights/reports and resolve symlinks before creation."""
    result = Path(path).expanduser().resolve()
    source = Path(__file__).resolve()
    repository = next((p for p in source.parents if (p / ".git").exists()), None)
    if repository is not None and (result == repository or repository in result.parents):
        raise ValueError("generated model artifacts must be outside repository")
    result.mkdir(parents=True, exist_ok=True)
    marker = result / "pals-artifact-owner.json"
    if marker.exists():
        owner = json.loads(marker.read_text(encoding="utf-8"))
        if owner.get("schema") != SCHEMA or owner.get("owner") != "rz-pals-model":
            raise ValueError("output directory belongs to another workload")
    else:
        atomic_json(marker, {"schema": SCHEMA, "owner": "rz-pals-model"}, output_guard=output_guard)
    return result


def atomic_json(path, value, *, output_guard=None):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite registered artifact: {path.name}")
    temporary = path.with_name(path.name + ".partial")
    if temporary.exists():
        raise FileExistsError("incomplete artifact already exists; inspect before retry")
    if output_guard is not None:
        raw = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
        output_guard(len(raw))
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def initialize_checkpoint(directory, seed, profile=DEFAULT_INIT_PROFILE):
    import torch
    from .model import initialize
    output = output_directory(directory)
    checkpoint = output / "untrained.pt"
    if checkpoint.exists() or checkpoint.with_suffix(".pt.partial").exists():
        raise FileExistsError("checkpoint already exists")
    model = initialize(seed, ModelConfig.for_profile(profile))
    temporary = checkpoint.with_suffix(".pt.partial")
    # Only tensors and JSON-like primitives; loading uses weights_only=True.
    torch.save({"state_dict": model.state_dict(), "config": model.config.to_dict(), "seed": seed,
                "schema": model.config.model_semantics, "training_steps": 0, "trained": False}, temporary)
    temporary.replace(checkpoint)
    metadata = {"schema": model.config.model_semantics, "checkpoint": checkpoint.name, "checkpoint_sha256": digest_file(checkpoint),
                "config": model.config.to_dict(), "seed": seed, "trained": False, "training_steps": 0,
                "external_weights": False, "external_teacher": False, "precision": "fp32", "tf32": False,
                "torch_version": torch.__version__, "parameter_count": sum(p.numel() for p in model.parameters()),
                "roles": list(model.experts), "status": "random_initialization_only"}
    atomic_json(output / "checkpoint.json", metadata)
    return metadata


def load_checkpoint(path):
    import torch
    from .model import PalsModel
    path = Path(path).resolve()
    metadata_path = path.with_name("checkpoint.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema") not in (SCHEMA, MODEL_SEMANTICS_V2, MODEL_SEMANTICS_V3) or metadata.get("checkpoint_sha256") != digest_file(path):
        raise ValueError("checkpoint identity mismatch")
    virtual = metadata.get("schema") == MODEL_SEMANTICS_V3
    teacher_valid = type(metadata.get("external_teacher")) is bool if virtual else metadata.get("external_teacher") is False
    if metadata.get("external_weights") is not False or not teacher_valid:
        raise ValueError("first PALS profile requires explicit own-weight/no-external-teacher provenance")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    steps = checkpoint.get("training_steps")
    if checkpoint.get("schema") != metadata["schema"] or not isinstance(steps, int) or isinstance(steps, bool) or steps < 0 or checkpoint.get("trained") is not (steps > 0):
        raise ValueError("checkpoint must declare consistent training provenance")
    if metadata.get("training_steps") != steps or metadata.get("trained") is not checkpoint["trained"]:
        raise ValueError("checkpoint/receipt training provenance mismatch")
    if checkpoint.get("config") != metadata.get("config"):
        raise ValueError("checkpoint configuration mismatch")
    config = ModelConfig(**checkpoint["config"])
    config.validate()
    if metadata["schema"] != config.model_semantics:
        raise ValueError("checkpoint schema/profile semantics mismatch")
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(checkpoint["seed"])
        model = PalsModel(config)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.eval(), metadata


def validate_export_manifest(manifest, metadata):
    """Never accept public-only execution as P/C neural acceptance."""
    from .onnx_shared import ARTIFACT_SCHEMA, SHARED_LAYOUT, SHARED_LAYOUT_REVISION
    if manifest.get("schema") == MODEL_SCHEMA_V3:
        return validate_virtual_export_manifest(manifest, metadata)
    shared = manifest.get("schema") == ARTIFACT_SCHEMA
    if manifest.get("schema") not in (SCHEMA, ARTIFACT_SCHEMA) or manifest.get("config") != metadata.get("config") or manifest.get("checkpoint_sha256") != metadata.get("checkpoint_sha256"):
        raise ValueError("export schema/config/checkpoint mismatch")
    config = ModelConfig(**manifest["config"])
    config.validate()
    if config.profile != "legacy_summary_v1" and (manifest.get("model_semantics") != config.model_semantics or manifest.get("encoding_schema") != config.encoding or manifest.get("model_profile") != config.profile):
        raise ValueError("export model profile/semantic/encoding mismatch")
    if shared and (manifest.get("model_semantics") != config.model_semantics or manifest.get("layout") != SHARED_LAYOUT or manifest.get("layout_revision") != SHARED_LAYOUT_REVISION):
        raise ValueError("export shared layout/semantic revision mismatch")
    if shared and (manifest.get("rules_input_profile") != ("rz-pals-rules-full-line-v2" if config.full_line else "rz-pals-rules-fields-v1") or not _sha256_text(manifest.get("rules_input_semantic_sha256")) or not _sha256_text(manifest.get("rules_encoder_source_sha256"))):
        raise ValueError("shared production layout requires Rules semantic profile")
    if manifest.get("trained") is not metadata.get("trained") or manifest.get("training_steps") != metadata.get("training_steps"):
        raise ValueError("export training provenance mismatch")
    if manifest.get("validator_present") is False:
        roles = ["proposer", "critic"]
    elif manifest.get("validator_present") is True:
        roles = ["proposer", "critic", "validator"]
    else:
        raise ValueError("export must declare validator presence")
    if manifest.get("roles") != roles or manifest.get("task_names") != list(TASKS):
        raise ValueError("export role/task vocabulary mismatch")
    if shared and roles != ["proposer", "critic"]:
        raise ValueError("shared_pc_if requires V-free P/C roles")
    graphs = manifest.get("graphs")
    if not isinstance(graphs, list) or len(graphs) != (2 if shared else len(roles) + 1):
        raise ValueError("export required graph count mismatch")
    graph_roles = [graph.get("role") for graph in graphs]
    if graph_roles != (["public", "shared_pc"] if shared else ["public", *roles]):
        raise ValueError("export requires one public graph and exactly the declared private roles")
    for graph in graphs:
        role = graph["role"]
        expected_file = "public_memory.onnx" if role == "public" else ("shared_pc_if.onnx" if role == "shared_pc" else f"role_{role}.onnx")
        digest = graph.get("sha256")
        if graph.get("file") != expected_file or not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest) or graph.get("opset") != 17:
            raise ValueError("export graph file/digest/opset mismatch")
        if role == "public":
            expected_inputs = public_tensor_names(config)
            expected_outputs = ["memory_key", "memory_value", "memory_mask"]
        else:
            expected_inputs = (["role_is_critic"] if role == "shared_pc" else []) + ["memory_key", "memory_value", "memory_mask", "candidates", "candidate_mask"]
            if role in ("critic", "shared_pc"): expected_inputs += ["divergence_features", "divergence_mask"]
            expected_inputs += ["query"]
            if config.full_line:
                expected_inputs += ["query_line_tokens", "query_line_mask", "record_relations"]
            expected_outputs = ["candidate_logits", "wdl_logits", "private_latent"]
            if role in ("critic", "shared_pc"): expected_outputs += ["divergence_logits"]
            if role == "shared_pc": expected_outputs += ["is_critic"]
            if role == "validator": expected_outputs += ["task_logits"]
        for key, expected in (("inputs", expected_inputs), ("outputs", expected_outputs)):
            descriptors = graph.get(key)
            if not isinstance(descriptors, list) or [value.get("name") for value in descriptors] != expected:
                raise ValueError("export graph required tensor names mismatch")
        line_shapes = {"record_line_tokens": ("INT64", ["batch", "records", MAX_LINE_PLIES, 3]),
                       "record_line_mask": ("BOOL", ["batch", "records", MAX_LINE_PLIES]),
                       "query_line_tokens": ("INT64", ["batch", 3, MAX_LINE_PLIES, 3]),
                       "query_line_mask": ("BOOL", ["batch", 3, MAX_LINE_PLIES]),
                       "record_relations": ("INT64", ["batch", "records", 2])}
        for descriptor in graph["inputs"]:
            if descriptor["name"] in line_shapes:
                dtype, shape = line_shapes[descriptor["name"]]
                if descriptor.get("dtype") != dtype or descriptor.get("shape") != shape:
                    raise ValueError("export full-line tensor shape/dtype mismatch")
    return roles


def annotate_public_memory_shapes(document, config):
    """Restore the architecture's literal KV axes lost by legacy transpose export.

    Batch and token axes stay symbolic. An already contradictory literal shape
    is rejected, never rewritten into an apparently supported descriptor.
    """
    config.validate()
    outputs = {value.name: value for value in document.graph.output}
    for name in ("memory_key", "memory_value"):
        if name not in outputs:
            raise ValueError("public export is missing required memory output")
        dimensions = outputs[name].type.tensor_type.shape.dim
        if len(dimensions) != 4:
            raise ValueError("public KV output must have rank four")
        for axis, expected in ((1, config.kv_heads), (3, config.head_dimension)):
            dimension = dimensions[axis]
            if dimension.dim_value and dimension.dim_value != expected:
                raise ValueError("public KV output contradicts fixed model architecture")
            dimension.ClearField("dim_param")
            dimension.dim_value = expected


def export_checkpoint(checkpoint, directory, include_validator=False, layout="separate_pc", rules_profile_json=None, *, output_guard=None):
    if output_guard is not None:
        if not callable(output_guard):
            raise TypeError("export output guard must be callable")
        if layout != "virtual_review_shared_pc_if_v1":
            raise ValueError("bounded export requires the virtual review layout")
    import onnx
    import torch
    from .model import fixture_input
    from .onnx_shared import ARTIFACT_SCHEMA, SHARED_LAYOUT, SHARED_LAYOUT_REVISION, build_shared_pc_graph
    from .onnx_virtual_review import SHARED_LAYOUT as VIRTUAL_LAYOUT
    if layout not in ("separate_pc", SHARED_LAYOUT, VIRTUAL_LAYOUT) or (layout == SHARED_LAYOUT and include_validator):
        raise ValueError("unsupported layout or validator in V-free shared_pc_if")
    model, metadata = load_checkpoint(checkpoint)
    if model.config.virtual_review:
        if layout != VIRTUAL_LAYOUT:
            raise ValueError("V3 profile requires explicit virtual_review_shared_pc_if_v1 layout")
        return export_virtual_checkpoint(model, metadata, directory, include_validator=include_validator,
                                         rules_profile_json=rules_profile_json, output_guard=output_guard)
    if layout == VIRTUAL_LAYOUT:
        raise ValueError("legacy/V2 model cannot use V3 virtual review layout")
    if layout == SHARED_LAYOUT and rules_profile_json is None:
        raise ValueError("shared_pc_if requires --rules-profile-json from the native Rules descriptor")
    rules_profile = read_rules_profile(rules_profile_json, model.config) if rules_profile_json is not None else None
    output = output_directory(directory)
    if not include_validator:
        model = model.without_validator()
    data = fixture_input(profile=model.config.profile)
    data.validate(model.config)
    graphs = []
    public_args = data.public_args(model.config)
    # CPU export does not execute TF32. Never change caller autograd/backend
    # globals; a future preparation consumer may export then build its losses.
    with torch.no_grad():
        memory = model.public_encoder(*public_args)
    for value in memory[:2]:
        if value.ndim != 4 or value.shape[1] != model.config.kv_heads or value.shape[3] != model.config.head_dimension:
            raise ValueError("actual public KV shape contradicts fixed model architecture")

    artifact_schema = ARTIFACT_SCHEMA if layout == SHARED_LAYOUT else SCHEMA
    def save_document(document, temporary, destination, filename, role):
        onnx.checker.check_model(document, full_check=True)
        properties = {"schema": artifact_schema, "model_semantics": model.config.model_semantics,
            "layout": layout, "layout_revision": str(SHARED_LAYOUT_REVISION if layout == SHARED_LAYOUT else 1), "role": role,
            "checkpoint_sha256": metadata["checkpoint_sha256"], "training_steps": str(metadata["training_steps"]), "trained": str(metadata["trained"]).lower(),
            "precision": "fp32", "expected_ort": "1.22.0", "public_kv": "shared_role_neutral"}
        if model.config.profile != "legacy_summary_v1":
            properties.update({"encoding_schema": model.config.encoding, "model_profile": model.config.profile})
        if rules_profile is not None:
            properties.update({key: rules_profile[key] for key in ("rules_input_profile", "rules_input_semantic_sha256", "rules_encoder_source_sha256")})
        onnx.helper.set_model_props(document, properties)
        onnx.save(document, str(temporary))
        temporary.replace(destination)
        inputs = [{"name": v.name, "dtype": onnx.TensorProto.DataType.Name(v.type.tensor_type.elem_type),
                   "shape": [d.dim_param or d.dim_value for d in v.type.tensor_type.shape.dim]} for v in document.graph.input]
        outputs = [{"name": v.name, "dtype": onnx.TensorProto.DataType.Name(v.type.tensor_type.elem_type),
                    "shape": [d.dim_param or d.dim_value for d in v.type.tensor_type.shape.dim]} for v in document.graph.output]
        graphs.append({"file": filename, "sha256": digest_file(destination), "role": role,
                       "inputs": inputs, "outputs": outputs, "opset": 17})

    def export(graph, args, filename, names, output_names, dynamic_axes, role):
        destination = output / filename
        temporary = destination.with_suffix(".onnx.partial")
        if destination.exists() or temporary.exists():
            raise FileExistsError(f"graph already exists: {filename}")
        with torch.no_grad():
            torch.onnx.export(graph, args, str(temporary), input_names=names, output_names=output_names,
                              dynamic_axes=dynamic_axes, opset_version=17, dynamo=False,
                              export_params=True, do_constant_folding=True)
        document = onnx.load(str(temporary), load_external_data=False)
        if role == "public":
            annotate_public_memory_shapes(document, model.config)
        save_document(document, temporary, destination, filename, role)

    public_names = public_tensor_names(model.config)
    axes = {name: {0: "batch"} for name in public_names}
    axes["records"][1] = axes["record_mask"][1] = "records"
    if model.config.full_line:
        axes["record_line_tokens"][1] = axes["record_line_mask"][1] = "records"
    axes.update({"memory_key": {0: "batch", 2: "memory_tokens"}, "memory_value": {0: "batch", 2: "memory_tokens"}, "memory_mask": {0: "batch", 1: "memory_tokens"}})
    export(model.public_encoder, public_args, "public_memory.onnx", public_names,
           ["memory_key", "memory_value", "memory_mask"], axes, "public")
    names = role_tensor_names(model.config)
    for role in (model.experts if layout == "separate_pc" else ()):
        outputs = ["candidate_logits", "wdl_logits", "private_latent"]
        if role == "critic": outputs.append("divergence_logits")
        if role == "validator": outputs.append("task_logits")
        axes = {name: {0: "batch"} for name in names + outputs}
        axes["memory_key"][2] = axes["memory_value"][2] = "memory_tokens"
        axes["memory_mask"][1] = "memory_tokens"
        axes["candidates"][1] = axes["candidate_mask"][1] = axes["candidate_logits"][1] = "candidates"
        axes["divergence_features"][1] = axes["divergence_mask"][1] = "divergences"
        if role == "critic": axes["divergence_logits"][1] = "divergences"
        if model.config.full_line:
            axes["record_relations"][1] = "records"
        export(model.role_graph(role), data.role_args(memory, model.config), f"role_{role}.onnx", names, outputs, axes, role)
    reader_bank = None
    if layout == SHARED_LAYOUT:
        filename = "shared_pc_if.onnx"
        destination = output / filename
        temporary = destination.with_suffix(".onnx.partial")
        if destination.exists() or temporary.exists():
            raise FileExistsError("shared P/C graph already exists")
        document, reader_bank = build_shared_pc_graph(model)
        save_document(document, temporary, destination, filename, "shared_pc")
    manifest = {"schema": artifact_schema, "config": model.config.to_dict(), "checkpoint_sha256": metadata["checkpoint_sha256"],
                "trained": metadata["trained"], "training_steps": metadata["training_steps"], "roles": list(model.experts), "validator_present": include_validator,
                "graphs": graphs, "runtime": {"onnxruntime": "1.22.0", "rust_ort": "2.0.0-rc.10", "compatibility": "requires_actual_numeric_check"},
                "candidate_promotion": {"0": "none", "1": "queen", "2": "rook", "3": "bishop", "4": "knight"},
                "wdl_perspective": "input_side_to_move", "task_names": list(TASKS), "numeric_status": "not_run", "cuda_status": "not_run"}
    if layout == SHARED_LAYOUT:
        manifest.update({"model_semantics": model.config.model_semantics, "layout": layout, "layout_revision": SHARED_LAYOUT_REVISION,
                         "reader_initializer_bank": reader_bank, "role_batching": "one_scalar_role_per_physical_batch"})
    if model.config.profile != "legacy_summary_v1":
        manifest.update({"model_semantics": model.config.model_semantics, "encoding_schema": model.config.encoding,
                         "model_profile": model.config.profile})
    if rules_profile is not None:
        manifest.update(rules_profile)
    atomic_json(output / "export.json", manifest)
    return manifest


def validate_virtual_export_manifest(manifest, metadata):
    """V3 never passes through V1/V2 graph-schema or fixed-L validation."""
    from .onnx_virtual_review import SHARED_LAYOUT, LAYOUT_REVISION, virtual_io_spec
    config = ModelConfig(**manifest.get("config", {})); config.validate()
    if not config.virtual_review or manifest.get("schema") != MODEL_SCHEMA_V3 or manifest.get("config") != metadata.get("config") or manifest.get("checkpoint_sha256") != metadata.get("checkpoint_sha256"):
        raise ValueError("V3 export schema/config/checkpoint mismatch")
    if manifest.get("precision") != "fp32" or manifest.get("tf32") is not False:
        raise ValueError("V3 registered FP32/TF32-off execution declaration")
    if manifest.get("model_semantics") != config.model_semantics or manifest.get("encoding_schema") != config.encoding or manifest.get("model_profile") != config.profile or manifest.get("layout") != SHARED_LAYOUT or manifest.get("layout_revision") != LAYOUT_REVISION:
        raise ValueError("V3 export semantic/layout identity")
    roles = ["proposer","critic","validator"] if manifest.get("validator_present") is True else ["proposer","critic"] if manifest.get("validator_present") is False else None
    if roles is None or manifest.get("roles") != roles or manifest.get("task_names") != list(TASKS) or manifest.get("trained") is not metadata.get("trained") or manifest.get("training_steps") != metadata.get("training_steps"):
        raise ValueError("V3 export role/training provenance")
    declaration = manifest.get("rules_input_declaration")
    digest = validate_virtual_rules_declaration(declaration if type(declaration) is dict else {}, config)
    if manifest.get("rules_input_semantic_sha256") != digest or manifest.get("rules_encoder_source_sha256") != declaration["rules_encoder_source_sha256"]:
        raise ValueError("V3 export Rules provenance differs")
    spec = virtual_io_spec(config)
    graph_roles = ["public","shared_pc"] + (["validator"] if "validator" in roles else [])
    graphs = manifest.get("graphs")
    if type(graphs) is not list or [g.get("role") for g in graphs] != graph_roles:
        raise ValueError("V3 export public/P-C/V graph ownership")
    for graph in graphs:
        role = graph["role"]
        expected_file = {"public":"public_memory.onnx","shared_pc":"shared_pc_if.onnx","validator":"role_validator.onnx"}[role]
        if graph.get("file") != expected_file or not _sha256_text(graph.get("sha256")) or graph.get("opset") != 17:
            raise ValueError("V3 export graph filename/digest/opset")
        expected = spec[role]
        for key in ("inputs","outputs"):
            descriptors = [{"name":name,"dtype":dtype,"shape":list(shape)} for name,dtype,shape in expected[key]]
            if graph.get(key) != descriptors:
                raise ValueError("V3 export graph exact dynamic I/O descriptors")
    if manifest.get("physical_line_buckets") != [8,16,32,64,128,256] or manifest.get("session_policy") != "one_dynamic_L_graph_no_bucket_weight_copies":
        raise ValueError("V3 export length/session policy")
    return roles


def export_virtual_checkpoint(model, metadata, directory, *, include_validator=False, rules_profile_json=None, output_guard=None):
    """Build explicit V3 graphs without running a neural forward during export."""
    import onnx
    from .onnx_virtual_review import (SHARED_LAYOUT, LAYOUT_REVISION, virtual_io_spec,
        build_virtual_public_graph, build_virtual_shared_pc_graph, build_virtual_validator_graph, _save)
    if not model.config.virtual_review or rules_profile_json is None:
        raise ValueError("V3 export requires explicit source-pinned virtual Rules descriptor")
    rules_profile = read_rules_profile(rules_profile_json,model.config)
    output = output_directory(directory, output_guard=output_guard)
    spec, graphs, banks = virtual_io_spec(model.config), [], {}
    # P/C graph has no validator private initializer or role path. The optional
    # independent V graph is a collection artifact, omitted from Deployment.
    pc = model.virtual_review_pc_export_view()
    items = [("public","public_memory.onnx",build_virtual_public_graph(pc)),
             ("shared_pc","shared_pc_if.onnx",build_virtual_shared_pc_graph(pc))]
    if include_validator:
        if "validator" not in model.experts: raise ValueError("collection export requires private V bank")
        items.append(("validator","role_validator.onnx",build_virtual_validator_graph(model)))
    for role, filename, (document, evidence) in items:
        onnx.helper.set_model_props(document,{"schema":MODEL_SCHEMA_V3,"model_semantics":model.config.model_semantics,
            "model_profile":model.config.profile,"encoding_schema":model.config.encoding,"layout":SHARED_LAYOUT,
            "layout_revision":str(LAYOUT_REVISION),"role":role,"checkpoint_sha256":metadata["checkpoint_sha256"],
            "trained":str(metadata["trained"]).lower(),"training_steps":str(metadata["training_steps"]),
            "precision":"fp32","expected_ort":"1.22.0", **{key:rules_profile[key] for key in ("rules_input_profile","rules_input_semantic_sha256","rules_encoder_source_sha256")}})
        onnx.checker.check_model(document,full_check=True)
        saved = _save(document,evidence,output/filename,output_guard=output_guard)
        declaration = spec[role]
        graphs.append({"role":role,"file":filename,"sha256":saved["sha256"],"opset":17,
            **{key:[{"name":name,"dtype":dtype,"shape":list(shape)} for name,dtype,shape in declaration[key]] for key in ("inputs","outputs")}})
        banks[role] = evidence
    roles = ["proposer","critic"] + (["validator"] if include_validator else [])
    manifest = {"schema":MODEL_SCHEMA_V3,"model_semantics":model.config.model_semantics,"encoding_schema":model.config.encoding,
        "model_profile":model.config.profile,"config":model.config.to_dict(),"checkpoint_sha256":metadata["checkpoint_sha256"],
        "trained":metadata["trained"],"training_steps":metadata["training_steps"],"roles":roles,"validator_present":include_validator,
        "layout":SHARED_LAYOUT,"layout_revision":LAYOUT_REVISION,"graphs":graphs,"task_names":list(TASKS),
        "physical_line_buckets":[8,16,32,64,128,256],"session_policy":"one_dynamic_L_graph_no_bucket_weight_copies",
        "initializer_banks":banks,"precision":"fp32","tf32":False,"runtime":{"onnxruntime":"1.22.0","rust_ort":"2.0.0-rc.10","compatibility":"requires_actual_numeric_check"},
        "candidate_promotion":{"0":"none","1":"queen","2":"rook","3":"bishop","4":"knight"},
        "wdl_perspective":"input_side_to_move","numeric_status":"not_run","cuda_status":"not_run", **rules_profile}
    validate_virtual_export_manifest(manifest,metadata)
    atomic_json(output/"export.json",manifest,output_guard=output_guard)
    return manifest


def validate_virtual_warm_export_manifest(manifest, metadata):
    from .onnx_virtual_review import WARM_LAYOUT,LAYOUT_REVISION,virtual_io_spec
    config=ModelConfig(**manifest.get("config",{}));config.validate()
    if not config.virtual_review or manifest.get("schema")!=MODEL_SCHEMA_V3 or manifest.get("config")!=metadata.get("config") or manifest.get("checkpoint_sha256")!=metadata.get("checkpoint_sha256"):
        raise ValueError("V3 Warm asset identity")
    required={"layout":WARM_LAYOUT,"layout_revision":LAYOUT_REVISION,"model_semantics":config.model_semantics,
        "model_profile":config.profile,"encoding_schema":config.encoding,"roles":["proposer","critic"],"validator_present":False,
        "approximate":True,"warm_seed_policy":"accepted_final_latent_same_model_role_rules_query_epoch_physical_fence",
        "precision":"fp32","tf32":False,"physical_line_buckets":[8,16,32,64,128,256],
        "session_policy":"one_dynamic_L_graph_no_bucket_weight_copies","task_names":list(TASKS)}
    if any(manifest.get(key)!=value or type(manifest.get(key)) is not type(value) for key,value in required.items()):
        raise ValueError("V3 Warm semantic/capability declaration")
    if manifest.get("trained") is not metadata.get("trained") or manifest.get("training_steps")!=metadata.get("training_steps"):
        raise ValueError("V3 Warm training provenance")
    declaration=manifest.get("rules_input_declaration")
    digest=validate_virtual_rules_declaration(declaration if type(declaration)is dict else {},config)
    if manifest.get("rules_input_semantic_sha256")!=digest or manifest.get("rules_encoder_source_sha256")!=declaration["rules_encoder_source_sha256"]:
        raise ValueError("V3 Warm Rules provenance")
    graphs=manifest.get("graphs");spec=virtual_io_spec(config)
    if type(graphs)is not list or [g.get("role") for g in graphs]!=["public","shared_warm"]:
        raise ValueError("V3 Warm requires exactly V-free public/private graphs")
    for graph in graphs:
        role=graph["role"];filename="public_memory.onnx" if role=="public" else "shared_pc_if_approx_warm.onnx"
        if graph.get("file")!=filename or not _sha256_text(graph.get("sha256")) or graph.get("opset")!=17:
            raise ValueError("V3 Warm graph pin")
        if any(graph.get(key)!=[{"name":name,"dtype":dtype,"shape":list(shape)} for name,dtype,shape in spec[role][key]] for key in ("inputs","outputs")):
            raise ValueError("V3 Warm exact dynamic inputs/output declaration")
    return ["proposer","critic"]


def export_virtual_warm_checkpoint(checkpoint, directory, *, rules_profile_json):
    """Explicit separate approximate artifact; never relabel a Fresh graph."""
    import onnx
    from .onnx_virtual_review import (WARM_LAYOUT,LAYOUT_REVISION,virtual_io_spec,
        build_virtual_public_graph,build_virtual_warm_shared_pc_graph,_save)
    model,metadata=load_checkpoint(checkpoint)
    if not model.config.virtual_review:
        raise ValueError("virtual Warm exporter refuses legacy/V2 input assets")
    model=model.virtual_review_pc_export_view();rules_profile=read_rules_profile(rules_profile_json,model.config)
    output=output_directory(directory);spec=virtual_io_spec(model.config);graphs=[];banks={}
    for role,filename,(document,evidence) in (("public","public_memory.onnx",build_virtual_public_graph(model)),
        ("shared_warm","shared_pc_if_approx_warm.onnx",build_virtual_warm_shared_pc_graph(model))):
        onnx.helper.set_model_props(document,{"schema":MODEL_SCHEMA_V3,"layout":WARM_LAYOUT,"layout_revision":str(LAYOUT_REVISION),
            "model_semantics":model.config.model_semantics,"model_profile":model.config.profile,"encoding_schema":model.config.encoding,
            "role":role,"checkpoint_sha256":metadata["checkpoint_sha256"],"trained":str(metadata["trained"]).lower(),
            "training_steps":str(metadata["training_steps"]),"precision":"fp32","expected_ort":"1.22.0", "approximate":"true",
            **{key:rules_profile[key] for key in ("rules_input_profile","rules_input_semantic_sha256","rules_encoder_source_sha256")}})
        onnx.checker.check_model(document,full_check=True);saved=_save(document,evidence,output/filename)
        graphs.append({"role":role,"file":filename,"sha256":saved["sha256"],"opset":17,
            **{key:[{"name":name,"dtype":dtype,"shape":list(shape)} for name,dtype,shape in spec[role][key]] for key in ("inputs","outputs")}})
        banks[role]=evidence
    manifest={"schema":MODEL_SCHEMA_V3,"model_semantics":model.config.model_semantics,"model_profile":model.config.profile,
        "encoding_schema":model.config.encoding,"config":model.config.to_dict(),"checkpoint_sha256":metadata["checkpoint_sha256"],
        "trained":metadata["trained"],"training_steps":metadata["training_steps"],"layout":WARM_LAYOUT,"layout_revision":LAYOUT_REVISION,
        "roles":["proposer","critic"],"validator_present":False,"task_names":list(TASKS),"graphs":graphs,"initializer_banks":banks,
        "physical_line_buckets":[8,16,32,64,128,256],"session_policy":"one_dynamic_L_graph_no_bucket_weight_copies",
        "approximate":True,"warm_seed_policy":"accepted_final_latent_same_model_role_rules_query_epoch_physical_fence",
        "precision":"fp32","tf32":False,"runtime":{"onnxruntime":"1.22.0","rust_ort":"2.0.0-rc.10","compatibility":"requires_actual_seeded_numeric_check"},
        "candidate_promotion":{"0":"none","1":"queen","2":"rook","3":"bishop","4":"knight"},"wdl_perspective":"input_side_to_move",
        "numeric_status":"not_run","cuda_status":"not_run",**rules_profile}
    validate_virtual_warm_export_manifest(manifest,metadata);atomic_json(output/"export.json",manifest)
    return manifest


def virtual_warm_numeric_check(checkpoint, directory):
    """Fresh and nonzero seeded Torch/ORT comparison for the separate graph.

    The numerical seed comes from a reference Fresh output. This certifies math,
    not a product accepted ticket, Rules query, native fence or GPU lifetime.
    """
    import numpy as np
    import onnxruntime as ort
    import torch
    from .model import virtual_review_fixture_input
    from .virtual_review import BUCKETS
    model,metadata=load_checkpoint(checkpoint);model=model.virtual_review_pc_export_view()
    directory=Path(directory).resolve();manifest=json.loads((directory/"export.json").read_text(encoding="utf-8"))
    validate_virtual_warm_export_manifest(manifest,metadata)
    if ort.__version__!="1.22.0":raise ValueError("V3 seeded reference requires ORT1.22.0")
    options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1
    options.add_session_config_entry("session.intra_op.allow_spinning","0");options.add_session_config_entry("session.inter_op.allow_spinning","0")
    sessions={}
    for graph in manifest["graphs"]:
        path=directory/graph["file"]
        if digest_file(path)!=graph["sha256"]:raise ValueError("V3 Warm graph hash changed")
        sessions[graph["role"]]=ort.InferenceSession(str(path),sess_options=options,providers=["CPUExecutionProvider"])
    cases=[]
    with torch.no_grad():
        for bucket in BUCKETS:
            data=virtual_review_fixture_input(3,4,2,1,bucket=bucket,anchors=4);data.validate(model.config)
            public_feed={name:value.numpy() for name,value in zip(public_tensor_names(model.config),data.public_args(model.config))}
            public=sessions["public"].run(None,public_feed);reference=model.public_encoder(*data.public_args(model.config))
            for expected,actual in zip(reference,public):
                expected=expected.numpy()
                if expected.shape!=actual.shape or (not np.array_equal(expected,actual) if expected.dtype==np.bool_ else not np.allclose(expected,actual,atol=1e-4,rtol=1e-3)):
                    raise ValueError("V3 seeded public memory mismatch")
            for role in ("proposer","critic"):
                fresh=model.role_graph(role)(*data.role_args(reference,model.config))
                seed=fresh[2].detach().clone()
                if not torch.isfinite(seed).all() or not torch.any(seed!=0):raise ValueError("V3 seeded reference requires finite nonzero full latent")
                for warm in (False,True):
                    reference_args=(*data.role_args(reference,model.config),seed,torch.tensor(warm,dtype=torch.bool))
                    expected=model.warm_role_graph(role)(*reference_args)
                    actual_args=data.role_args(tuple(torch.from_numpy(x) for x in public),model.config)
                    feed={name:x.numpy() for name,x in zip(role_tensor_names(model.config),actual_args)}
                    feed.update({"role_is_critic":np.asarray(role=="critic",dtype=np.bool_),"initial_latent":seed.numpy(),"warm_start":np.asarray(warm,dtype=np.bool_)})
                    actual=sessions["shared_warm"].run(None,feed)
                    if actual[4].shape!=() or bool(actual[4])!=(role=="critic"):raise ValueError("V3 Warm role tag")
                    if role=="proposer" and not np.array_equal(actual[3],np.zeros_like(actual[3])):raise ValueError("V3 Warm P emitted critic output")
                    actual=actual[:4] if role=="critic" else actual[:3]
                    maxima=[]
                    for expected_output,actual_output in zip(expected,actual):
                        expected_output=expected_output.numpy()
                        if expected_output.shape!=actual_output.shape or not np.isfinite(actual_output).all() or not np.allclose(expected_output,actual_output,atol=1e-4,rtol=1e-3):
                            raise ValueError("V3 seeded output tolerance")
                        maxima.append(float(np.max(np.abs(expected_output-actual_output))))
                    for index in (0,1):
                        if np.max(np.abs(torch.softmax(expected[index],-1).numpy()-torch.softmax(torch.from_numpy(actual[index]),-1).numpy()))>1e-4:raise ValueError("V3 seeded policy/WDL tolerance")
                    cases.append({"role":role,"line_bucket":bucket,"mode":"Warm" if warm else "Fresh","max_absolute_differences":maxima,
                        "seed":"complete_nonzero_reference_fresh_final_latent","product_accepted_seed":False})
    if len(cases)!=24:raise ValueError("V3 seeded case coverage incomplete")
    return {"schema":"rovezero.pals-virtual-review-seeded-numeric.v1","status":"passed","cases":cases,
        "checkpoint_sha256":metadata["checkpoint_sha256"],"sessions_created":len(sessions),"provider":"CPUExecutionProvider",
        "rust_ort":"not_run","cuda":"not_run","physical_lifetime":"not_observed","performance_evidence_eligible":False}


def numeric_check(checkpoint, directory):
    """Independent Python ORT1.22 reference, not Rust/CUDA acceptance."""
    import numpy as np
    import onnxruntime as ort
    import torch
    from .model import fixture_input
    model, metadata = load_checkpoint(checkpoint)
    directory = Path(directory).resolve()
    manifest = json.loads((directory / "export.json").read_text(encoding="utf-8"))
    validate_export_manifest(manifest, metadata)
    if model.config.virtual_review:
        return virtual_numeric_check(model, metadata, directory, manifest)
    if ort.__version__ != "1.22.0" or manifest["checkpoint_sha256"] != metadata["checkpoint_sha256"]:
        raise ValueError("reference runtime/checkpoint identity mismatch")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    sessions = {}
    for graph in manifest["graphs"]:
        path = directory / graph["file"]
        if digest_file(path) != graph["sha256"]:
            raise ValueError("export graph hash mismatch")
        sessions[graph["role"]] = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    reports = []
    with torch.no_grad():
        for records, candidates, divergences, batch in ((1, 1, 1, 1), (3, 7, 4, 1), (5, 9, 3, 2)):
            data = fixture_input(records, candidates, divergences, batch, profile=model.config.profile)
            data.validate(model.config)
            public_feed = dict(zip(public_tensor_names(model.config), [x.numpy() for x in data.public_args(model.config)]))
            public = sessions["public"].run(None, public_feed)
            reference_memory = model.public_encoder(*data.public_args(model.config))
            for name, expected, actual in zip(("memory_key", "memory_value", "memory_mask"), reference_memory, public):
                expected = expected.numpy()
                if expected.shape != actual.shape:
                    raise ValueError(f"public shape mismatch: {name}")
                if expected.dtype == np.bool_:
                    if not np.array_equal(expected, actual): raise ValueError("public mask mismatch")
                elif not np.allclose(expected, actual, atol=1e-4, rtol=1e-3):
                    raise ValueError(f"public numeric mismatch: {name}")
            for role in manifest["roles"]:
                expected = model.role_graph(role)(*data.role_args(reference_memory, model.config))
                tensors = data.role_args(tuple(torch.from_numpy(v) for v in public), model.config)
                all_feed = dict(zip(role_tensor_names(model.config), [x.numpy() for x in tensors]))
                session = sessions.get(role)
                if manifest.get("layout") == "shared_pc_if":
                    session = sessions["shared_pc"]
                    all_feed["role_is_critic"] = np.asarray(role == "critic", dtype=np.bool_)
                actual = session.run(None, {value.name: all_feed[value.name] for value in session.get_inputs()})
                if manifest.get("layout") == "shared_pc_if":
                    if actual[4].shape != () or actual[4].dtype != np.bool_ or bool(actual[4]) != (role == "critic"):
                        raise ValueError("shared role tag mismatch")
                    if role == "proposer":
                        if actual[3].shape != (batch, divergences) or not np.array_equal(actual[3], np.zeros_like(actual[3])):
                            raise ValueError("P must emit empty-semantic zero divergence tensor")
                        actual = actual[:3]
                    else:
                        actual = actual[:4]
                if len(expected) != len(actual):
                    raise ValueError("role output count mismatch")
                maxima = []
                for index, (e, a) in enumerate(zip(expected, actual)):
                    e = e.numpy()
                    if e.shape != a.shape or not np.isfinite(a).all() or not np.allclose(e, a, atol=1e-4, rtol=1e-3):
                        raise ValueError(f"role numeric mismatch: {role} output {index}")
                    maxima.append(float(np.max(np.abs(e-a))))
                # The probabilities are independently normalized, never compared
                # by summing logits or treating masked padding as a legal move.
                for index in (0, 1):
                    ep = torch.softmax(expected[index], -1).numpy()
                    ap = torch.softmax(torch.from_numpy(actual[index]), -1).numpy()
                    if np.max(np.abs(ep-ap)) > 1e-4:
                        raise ValueError(f"policy/WDL numeric mismatch: {role}")
                reports.append({"role": role, "records": records, "candidates": candidates, "divergences": divergences,
                                "batch": batch, "max_absolute_differences": maxima})
    if len(reports) != 3 * len(manifest["roles"]):
        raise ValueError("numeric role acceptance case count mismatch")
    result = {"schema": model.config.model_semantics, "artifact_schema": manifest["schema"], "layout": manifest.get("layout", "separate_pc"),
              "checkpoint_sha256": metadata["checkpoint_sha256"], "trained": metadata["trained"],
              "runtime": ort.__version__, "provider": "CPUExecutionProvider", "rust_ort": "not_run", "cuda": "not_run", "cases": reports, "status": "passed"}
    if manifest.get("layout") == "shared_pc_if":
        from .numeric import verify_cpu_hard_routes
        data = fixture_input(3, 7, 4, 1, profile=model.config.profile)
        with torch.no_grad(): memory = model.public_encoder(*data.public_args(model.config))
        feed = dict(zip(role_tensor_names(model.config), [value.numpy() for value in data.role_args(memory, model.config)]))
        result["hard_route_cpu_profile"] = verify_cpu_hard_routes(directory / "shared_pc_if.onnx", feed)
    return result


def virtual_numeric_check(model, metadata, directory, manifest):
    """Independent reference command, covering every L bucket and anchor mask.

    This function is deliberately opt-in through numeric_check. It is not called
    by schema tests, model initialization, export or a benchmark. One session per
    graph is reused for all buckets; no length-specific bank/session is created.
    """
    import numpy as np
    import onnxruntime as ort
    import torch
    from .model import VirtualReviewTensorInput
    from .virtual_review import INPUT_SCHEMA, BUCKETS
    if ort.__version__ != "1.22.0": raise ValueError("V3 reference requires ORT1.22.0")
    options = ort.SessionOptions(); options.intra_op_num_threads = 1; options.inter_op_num_threads = 1
    options.add_session_config_entry("session.intra_op.allow_spinning","0")
    options.add_session_config_entry("session.inter_op.allow_spinning","0")
    sessions = {}
    for graph in manifest["graphs"]:
        path = Path(directory)/graph["file"]
        if digest_file(path) != graph["sha256"]: raise ValueError("V3 graph hash changed")
        sessions[graph["role"]] = ort.InferenceSession(str(path),sess_options=options,providers=["CPUExecutionProvider"])
    reports = []
    with torch.no_grad():
        for bucket in BUCKETS:
            for active_anchors in (1,2,3,4):
                line = [[(ply*7)%64,(ply*7+8)%64,ply%5] for ply in range(bucket)]
                value = {"schema":INPUT_SCHEMA,"role":"critic",
                    "anchors":[{"board":[(square+anchor)%13 for square in range(64)],"metadata":[field/16 for field in range(16)]} for anchor in range(active_anchors)],
                    "records":[{"kind":"counterexample","perspective":-1,"completed_iteration":True,"rules_terminal":False,
                        "line":line,"anchor":active_anchors-1,"parent":None,"supersedes":None}],
                    "query":{"purpose":"reply_policy","perspective":1,"prefix":[],"proposal":line,"counter":[],"counter_present":False,"anchors":[0,active_anchors-1,0]},
                    "candidates":[[48,56,promotion] for promotion in (1,2,3,4)],"divergences":[[0.]*8],"token_budget":512}
                data = VirtualReviewTensorInput.from_payload(value)
                # A=4 physical shape with 1..4 eligible anchors checks masked
                # board slots without leaking padded piece/metadata features.
                missing = 4-active_anchors
                if missing:
                    values = {**data.__dict__,
                        "anchor_board":torch.nn.functional.pad(data.anchor_board,(0,0,0,missing)),
                        "anchor_metadata":torch.nn.functional.pad(data.anchor_metadata,(0,0,0,missing)),
                        "anchor_mask":torch.nn.functional.pad(data.anchor_mask,(0,missing))}
                    data = VirtualReviewTensorInput(**values)
                data.validate(model.config)
                public_feed = {name:value.numpy() for name,value in zip(public_tensor_names(model.config),data.public_args(model.config))}
                actual_memory = sessions["public"].run(None,public_feed)
                expected_memory = model.public_encoder(*data.public_args(model.config))
                public_maxima=[]
                for name,expected,actual in zip(("memory_key","memory_value","memory_mask","anchor_context"),expected_memory,actual_memory):
                    expected=expected.numpy()
                    if expected.shape != actual.shape or (not np.array_equal(expected,actual) if expected.dtype==np.bool_ else not np.isfinite(actual).all() or not np.allclose(expected,actual,atol=1e-4,rtol=1e-3)):
                        raise ValueError("V3 public reference mismatch: "+name)
                    public_maxima.append(0. if expected.dtype==np.bool_ else float(np.max(np.abs(expected-actual))))
                for role in manifest["roles"]:
                    expected=model.role_graph(role)(*data.role_args(expected_memory,model.config))
                    args=data.role_args(tuple(torch.from_numpy(x) for x in actual_memory),model.config)
                    feed={name:x.numpy() for name,x in zip(role_tensor_names(model.config),args)}
                    if role=="validator": session=sessions["validator"]
                    else:
                        session=sessions["shared_pc"];feed["role_is_critic"]=np.asarray(role=="critic",dtype=np.bool_)
                    actual=session.run(None,{x.name:feed[x.name] for x in session.get_inputs()})
                    if role!="validator":
                        if actual[4].shape!=() or bool(actual[4])!=(role=="critic"): raise ValueError("V3 actual role tag")
                        if role=="proposer" and not np.array_equal(actual[3],np.zeros_like(actual[3])): raise ValueError("V3 P emitted critic private output")
                        actual=actual[:4] if role=="critic" else actual[:3]
                    if len(actual)!=len(expected): raise ValueError("V3 private output count")
                    maxima=[]
                    for index,(left,right) in enumerate(zip(expected,actual)):
                        left=left.numpy()
                        if left.shape!=right.shape or not np.isfinite(right).all() or not np.allclose(left,right,atol=1e-4,rtol=1e-3):
                            raise ValueError(f"V3 private reference mismatch: {role}/{index}")
                        maxima.append(float(np.max(np.abs(left-right))))
                    for index in (0,1):
                        if np.max(np.abs(torch.softmax(expected[index],-1).numpy()-torch.softmax(torch.from_numpy(actual[index]),-1).numpy()))>1e-4:
                            raise ValueError("V3 legal-policy/WDL normalization tolerance")
                    reports.append({"role":role,"line_bucket":bucket,"active_anchors":active_anchors,"physical_anchors":4,
                        "public_max_absolute_difference":public_maxima,"private_max_absolute_difference":maxima})
    expected_count=len(BUCKETS)*4*len(manifest["roles"])
    if len(reports)!=expected_count: raise ValueError("V3 numeric case coverage incomplete")
    return {"schema":MODEL_SCHEMA_V3,"model_semantics":model.config.model_semantics,"checkpoint_sha256":metadata["checkpoint_sha256"],
        "cases":reports,"status":"passed","provider":"CPUExecutionProvider","rust_ort":"not_run","cuda":"not_run",
        "sessions_created":len(sessions),"session_per_bucket":False,"rules_certified_training_data":False}


def rust_fixtures(checkpoint, export_directory, destination):
    """Raw independent Torch outputs for the Rust ORT acceptance command.

    These are bounded tensor fixtures and carry no claim of legal game states.
    The Rust Rules tests certify legal candidate construction independently.
    """
    import dataclasses
    import torch
    from .model import fixture_input
    model, metadata = load_checkpoint(checkpoint)
    if model.config.virtual_review:
        return virtual_rust_fixtures(model, metadata, export_directory, destination)
    export_directory = Path(export_directory).resolve()
    manifest = json.loads((export_directory / "export.json").read_text(encoding="utf-8"))
    validate_export_manifest(manifest, metadata)
    if manifest["checkpoint_sha256"] != metadata["checkpoint_sha256"]:
        raise ValueError("fixture/export checkpoint mismatch")
    output = Path(destination).expanduser().resolve()
    output_directory(output.parent)
    if output.exists(): raise FileExistsError("fixture already registered")
    cases = []
    shapes = (("empty_context", "proposer", 0, 0, 0, False, 1),
              ("critic_divergence", "critic", 3, 7, 4, False, 1),
              ("proposer_order", "proposer", 3, 7, 0, False, 1),
              ("proposer_order_reversed", "proposer", 3, 7, 0, True, 1),
              ("same_board_other_history", "proposer", 3, 7, 0, False, 2),
              ("all_promotions", "critic", 1, 4, 1, False, 3))
    if model.config.full_line:
        shapes += (("full_middle_record_changed", "critic", 3, 7, 4, False, 1),
                   ("full_middle_query_changed", "proposer", 3, 7, 0, False, 1),
                   ("record_relations_changed", "critic", 3, 7, 4, False, 1),
                   ("empty_line_padding", "proposer", 1, 1, 0, False, 1))
    with torch.no_grad():
        for name, role, records, candidates, divergences, reverse, history in shapes:
            if role not in manifest["roles"]:
                raise ValueError("fixture role absent from exported model")
            data = fixture_input(max(records, 1), max(candidates, 1), max(divergences, 1), profile=model.config.profile)
            data = dataclasses.replace(data, record_mask=torch.ones((1, max(records, 1)), dtype=torch.bool) if records else torch.zeros((1, 1), dtype=torch.bool),
                                       candidate_mask=torch.ones((1, max(candidates, 1)), dtype=torch.bool) if candidates else torch.zeros((1, 1), dtype=torch.bool),
                                       divergence_mask=torch.ones((1, max(divergences, 1)), dtype=torch.bool) if divergences else torch.zeros((1, 1), dtype=torch.bool))
            if not records: data = dataclasses.replace(data, records=torch.zeros_like(data.records))
            if not candidates: data = dataclasses.replace(data, candidates=torch.zeros_like(data.candidates))
            if not divergences: data = dataclasses.replace(data, divergence_features=torch.zeros_like(data.divergence_features))
            if reverse:
                data = dataclasses.replace(data, candidates=torch.flip(data.candidates, (1,)))
            if name == "all_promotions":
                data = dataclasses.replace(data, candidates=torch.tensor([[[48, 56, promotion] for promotion in (1, 2, 3, 4)]], dtype=torch.int64))
            if model.config.full_line:
                if not records or name == "empty_line_padding":
                    data.record_line_tokens.zero_(); data.record_line_mask.zero_(); data.record_relations.fill_(-1)
                if name == "empty_line_padding":
                    data.query_line_tokens.zero_(); data.query_line_mask.zero_()
                if name == "full_middle_record_changed":
                    data.record_line_tokens[0, 0, 1, 0] = 31
                if name == "full_middle_query_changed":
                    data.query_line_tokens[0, 1, 1, 0] = 31
                if name == "record_relations_changed":
                    data.record_relations[0, 2] = torch.tensor([-1, 0])
            data.validate(model.config)
            memory = model.public_encoder(*data.public_args(model.config))
            raw = model.role_graph(role)(*data.role_args(memory, model.config))
            input_value = {"role": role, "board": data.board[0].tolist(), "metadata": data.metadata[0].tolist(),
                "records": [{"record_id": index+1, "revision": 1, "critical": index == 0,
                             "features": data.records[0,index].tolist()} for index in range(records)],
                "required_critical_records": [1] if records else [],
                "candidates": [{"from": values[0], "to": values[1], "promotion": values[2]} for values in data.candidates[0,:candidates].tolist()],
                "divergence_features": data.divergence_features[0,:divergences].tolist(), "query": data.query[0].tolist(),
                "situation_revision": 1, "history_digest": [history]*32,
                "model_epoch": list(bytes.fromhex(metadata["checkpoint_sha256"]))}
            if model.config.full_line:
                def line(values, active):
                    return [{"from": token[0], "to": token[1], "promotion": token[2]} for token in values[active].tolist()]
                input_value["full_line"] = {"records": [
                    {"moves": line(data.record_line_tokens[0, index], data.record_line_mask[0, index]),
                     "parent": None if data.record_relations[0, index, 0] < 0 else int(data.record_relations[0, index, 0]),
                     "supersedes": None if data.record_relations[0, index, 1] < 0 else int(data.record_relations[0, index, 1])}
                    for index in range(records)],
                    **{name: line(data.query_line_tokens[0, index], data.query_line_mask[0, index])
                       for index, name in enumerate(("query_prefix", "query_proposal", "query_counter"))}}
            expected = {"candidate_logits": raw[0][0,:candidates].tolist(), "wdl_logits": raw[1][0].tolist(),
                        "private_latent": raw[2][0].flatten().tolist(),
                        "divergence_logits": raw[3][0,:divergences].tolist() if role == "critic" else None,
                        "task_logits": raw[3][0].tolist() if role == "validator" else None}
            public_memory = {"tokens": int(memory[0].shape[2]), "memory_key": memory[0].flatten().tolist(),
                             "memory_value": memory[1].flatten().tolist(), "memory_mask": memory[2].flatten().tolist()}
            cases.append({"name": name, "input": input_value, "expected": expected, "public_memory": public_memory})
    result = {"schema": model.config.model_semantics, "checkpoint_sha256": metadata["checkpoint_sha256"], "trained": metadata["trained"],
              "rules_certified": False, "reference": "pytorch_fp32_tf32_off", "cases": cases}
    if model.config.profile != "legacy_summary_v1":
        result.update({"config": model.config.to_dict(), "model_profile": model.config.profile,
                       "encoding_schema": model.config.encoding})
    atomic_json(output, result)
    return {"schema": model.config.model_semantics, "file": output.name, "sha256": digest_file(output), "cases": len(cases), "trained": metadata["trained"]}


def virtual_rust_fixtures(model, metadata, export_directory, destination):
    """Independent V3 reference with a distinct codec, never a V2 conversion."""
    import copy
    import torch
    from .model import VirtualReviewTensorInput
    from .virtual_review import INPUT_SCHEMA, BUCKETS, input_digest
    manifest=json.loads((Path(export_directory)/"export.json").read_text(encoding="utf-8"))
    validate_virtual_export_manifest(manifest,metadata)
    output=Path(destination).expanduser().resolve();output_directory(output.parent)
    if output.exists(): raise FileExistsError("V3 reference fixture already registered")
    cases=[]
    with torch.no_grad():
        for bucket in BUCKETS:
            for role in manifest["roles"]:
                line=[[(ply*7)%64,(ply*7+8)%64,ply%5] for ply in range(bucket)]
                payload={"schema":INPUT_SCHEMA,"role":role,
                    "anchors":[{"board":[(square+anchor)%13 for square in range(64)],"metadata":[field/16 for field in range(16)]} for anchor in range(4)],
                    "records":[{"kind":"counterexample","perspective":-1,"completed_iteration":True,"rules_terminal":False,
                        "line":line,"anchor":3,"parent":None,"supersedes":None}],
                    "query":{"purpose":"task_policy" if role=="validator" else "reply_policy","perspective":1,
                        "prefix":[],"proposal":line,"counter":[],"counter_present":False,"anchors":[0,3,0]},
                    "candidates":[[48,56,promotion] for promotion in (1,2,3,4)],"divergences":[[0.]*8] if role=="critic" else [],"token_budget":512}
                data=VirtualReviewTensorInput.from_payload(payload);data.validate(model.config)
                memory=model.public_encoder(*data.public_args(model.config));raw=model.role_graph(role)(*data.role_args(memory,model.config))
                rust=copy.deepcopy(payload)
                def moves(line):return [{"from":token[0],"to":token[1],"promotion":token[2]} for token in line]
                rust["candidates"]=moves(rust["candidates"])
                for record in rust["records"]:record["line"]=moves(record["line"])
                for name in ("prefix","proposal","counter"):rust["query"][name]=moves(rust["query"][name])
                expected={"candidate_logits":raw[0][0].tolist(),"wdl_logits":raw[1][0].tolist(),"private_latent":raw[2][0].flatten().tolist(),
                    "divergence_logits":raw[3][0].tolist() if role=="critic" else None,"task_logits":raw[3][0].tolist() if role=="validator" else None}
                public={"tokens":int(memory[0].shape[2]),"memory_key":memory[0].flatten().tolist(),"memory_value":memory[1].flatten().tolist(),
                    "memory_mask":memory[2].flatten().tolist(),"anchors":4,"anchor_context":memory[3].flatten().tolist()}
                cases.append({"name":f"virtual_L{bucket}_{role}","input":rust,"input_sha256":input_digest(payload),"expected":expected,"public_memory":public})
    result={"schema":"rovezero.pals-virtual-review-fixtures.v1","model_semantics":model.config.model_semantics,"model_profile":model.config.profile,
        "config":model.config.to_dict(),"encoding_schema":model.config.encoding,"checkpoint_sha256":metadata["checkpoint_sha256"],"trained":metadata["trained"],
        "reference":"pytorch_fp32_tf32_off","rules_certified":False,"cases":cases,"line_buckets":list(BUCKETS)}
    atomic_json(output,result)
    return {"schema":result["schema"],"file":output.name,"sha256":digest_file(output),"cases":len(cases),"trained":metadata["trained"]}
