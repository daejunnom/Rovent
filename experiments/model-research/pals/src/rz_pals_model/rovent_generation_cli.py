"""Pinned, bounded Rovent preparation commands; no neural execution option.

Run as ``python -m rz_pals_model.rovent_generation_cli --request /.../request.json
--request-sha256 <raw-file-pin>``. The existing product CLI, legacy data domains
and actual trainer/checkpoint loaders are intentionally unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
import time

from .artifacts import publish_noreplace
from .generation_data import fields, sha, strict_json, uint
from .rovent_generation import (GlobalLineageIndex, LegacyIntakeStatus, SeriesRecipe, audit_legacy_bundle,
    audit_legacy_chunks, common_teacher_forks, fork_plan, resume_plan, validate_checkpoint_metadata)

REQUEST_DOMAIN = "rz-rovent-generation-preparation-request/1"
TRAINING_REQUEST_DOMAIN = "rz-rovent-generation-preparation-request/2"
RESULT_DOMAIN = "rz-rovent-generation-preparation-result/1"
MAX_JSON_BYTES = 16 * 1024**2
MAX_INPUT_BYTES = 64 * 1024**2
MAX_WALL_NS = 120 * 10**9
ACTIONS = ("validate_recipe", "validate_checkpoint_metadata", "prepare_resume", "prepare_fork",
           "prepare_common_forks", "audit_legacy_intake", "audit_global_lineage")
TRAINING_ACTIONS = ("prepare_training_stage", "validate_training_inventory", "prepare_model_export")


def _path(value, *, directory=False):
    if type(value) is not str or not value or len(value) > 4096 or any(ord(c) < 32 for c in value):
        raise ValueError("explicit bounded preparation path required")
    path = Path(value)
    if not path.is_absolute() or any(part in (".", "..") for part in path.parts):
        raise ValueError("canonical absolute preparation path required")
    # The caller selects one pinned JSON file, never a directory scan. Managed
    # checkpoint/model/cache directories may contain legitimate metadata. Block
    # conventional credential basenames to avoid accidental credential intake;
    # this is not a claim that filename checks identify every possible secret.
    lower = path.name.lower()
    if not directory and (lower.startswith((".env", "client_secret")) or lower in (
            "credentials.json", "credential.json", "application_default_credentials.json", "service-account.json",
            "service_account.json", "secrets.json", "api-key.json", "api_key.json", "ssh-key.json", "ssh_key.json")):
        raise ValueError("conventional secret filename forbidden for preparation")
    if not directory and path.suffix != ".json":
        raise ValueError("preparation reads JSON metadata only, never checkpoint/model binaries")
    return path


def _plain_chain(path, *, directory=False):
    for parent in reversed(path.parents):
        detail = parent.lstat()
        if not stat.S_ISDIR(detail.st_mode) or stat.S_ISLNK(detail.st_mode) or getattr(detail, "st_file_attributes", 0) & 0x400:
            raise ValueError("linked/reparse preparation ancestor forbidden")
    detail = path.lstat()
    predicate = stat.S_ISDIR if directory else stat.S_ISREG
    if (not predicate(detail.st_mode) or stat.S_ISLNK(detail.st_mode)
            or getattr(detail, "st_file_attributes", 0) & 0x400 or not directory and detail.st_nlink != 1):
        raise ValueError("plain preparation directory or single-link metadata file required")
    return detail


def _stamp(detail, *, named_comparison=False):
    # Windows path stat reports creation time as ctime, while fstat reports
    # ChangeTime. Compare their common birthtime across FD/name observations;
    # retain ChangeTime for comparisons of the held FD before/after reading.
    timestamp = (getattr(detail, "st_birthtime_ns", detail.st_ctime_ns)
                 if named_comparison and os.name == "nt" else detail.st_ctime_ns)
    return (detail.st_dev, detail.st_ino, detail.st_mode, detail.st_nlink,
            detail.st_size, detail.st_mtime_ns, timestamp)


def _pin(value):
    fields(value, ("path", "sha256", "bytes"), "preparation JSON pin")
    _path(value["path"])
    sha(value["sha256"])
    if not 1 <= uint(value["bytes"], MAX_JSON_BYTES):
        raise ValueError("nonempty bounded metadata file required")
    return value


class PreparationBudget:
    def __init__(self, *, max_wall_ns, clock=time.monotonic_ns):
        self.clock, self.until_ns = clock, clock() + uint(max_wall_ns, MAX_WALL_NS)
        self.input_bytes = 0

    def check(self):
        if self.clock() >= self.until_ns:
            raise ValueError("PreparationDeadlineExceeded")

    def reserve_input(self, count):
        self.check()
        if self.input_bytes + count > MAX_INPUT_BYTES:
            raise ValueError("PreparationInputByteCapacityExceeded")
        self.input_bytes += count


def read_pinned_json(pin, budget):
    """Reject size/pin/path mismatch before parsing; hold the original file FD."""
    pin = _pin(pin)
    path = _path(pin["path"])
    before = _plain_chain(path)
    if before.st_size != pin["bytes"]:
        raise ValueError("preparation metadata byte pin mismatch")
    budget.reserve_input(before.st_size)
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(descriptor, "rb") as stream:
        held = os.fstat(stream.fileno())
        if _stamp(held, named_comparison=True) != _stamp(before, named_comparison=True):
            raise ValueError("preparation metadata identity changed before read")
        blocks, total, hasher = [], 0, hashlib.sha256()
        while True:
            budget.check()
            block = stream.read(min(65536, pin["bytes"] + 1 - total))
            if not block:
                break
            total += len(block)
            if total > pin["bytes"]:
                raise ValueError("preparation metadata grew during read")
            blocks.append(block)
            hasher.update(block)
        if (total != pin["bytes"] or hasher.hexdigest() != pin["sha256"]
                or _stamp(os.fstat(stream.fileno())) != _stamp(held)
                or _stamp(_plain_chain(path)) != _stamp(before)):
            raise ValueError("preparation metadata hash/identity mismatch")
    return strict_json(b"".join(blocks).decode("utf-8", "strict"))


def validate_request(value):
    names = ("schema", "action", "recipe", "checkpoint", "ancestors", "origins", "legacy_intake", "stage_id",
             "output_root", "output_name", "max_records", "max_output_bytes", "max_wall_ns")
    if type(value) is not dict:
        raise ValueError("Rovent preparation request object required")
    training_version = value.get("schema") == TRAINING_REQUEST_DOMAIN
    fields(value, names + (("training_preparation",) if training_version else ()), "Rovent preparation request")
    if value["schema"] not in (REQUEST_DOMAIN, TRAINING_REQUEST_DOMAIN) or value["action"] not in ACTIONS + TRAINING_ACTIONS:
        raise ValueError("new preparation domain and supported action required")
    training = None
    if value["action"] in TRAINING_ACTIONS:
        if not training_version:
            raise ValueError("new training preparation actions require request V2")
        training = fields(value["training_preparation"], ("membership", "inventory", "mode", "export_kind"), "trainer/checkpoint/export preparation")
        _pin(training["membership"])
        if value["action"] == "prepare_training_stage":
            if training["mode"] not in ("initialize", "resume", "fork") or training["export_kind"] is not None:
                raise ValueError("explicit stage operation with no foreign export override required")
            if training["mode"] == "initialize":
                if training["inventory"] is not None:
                    raise ValueError("initialization cannot load a foreign state inventory")
            else:
                _pin(training["inventory"])
        else:
            _pin(training["inventory"])
            if training["mode"] is not None or (value["action"] == "validate_training_inventory" and training["export_kind"] is not None):
                raise ValueError("unexpected checkpoint inventory mode/export override")
            if value["action"] == "prepare_model_export" and training["export_kind"] not in ("collection", "deployment"):
                raise ValueError("explicit model export kind required")
    elif training_version and value["training_preparation"] is not None:
        raise ValueError("unexpected training preparation inputs")
    _pin(value["recipe"])
    checkpoint_needed = value["action"] in ("validate_checkpoint_metadata", "prepare_resume", "prepare_fork",
                                            "prepare_common_forks", "audit_global_lineage") or (
        value["action"] == "prepare_training_stage" and training["mode"] != "initialize")
    if checkpoint_needed:
        _pin(value["checkpoint"])
    elif value["checkpoint"] is not None:
        raise ValueError("unexpected checkpoint metadata input")
    if type(value["ancestors"]) is not list or len(value["ancestors"]) > 256:
        raise ValueError("bounded ancestor metadata pins required")
    for pin in value["ancestors"]:
        _pin(pin)
    if value["action"] == "audit_global_lineage":
        _pin(value["origins"])
    elif value["origins"] is not None or value["ancestors"]:
        raise ValueError("unexpected lineage input")
    if value["action"] in ("prepare_resume", "prepare_fork", "prepare_training_stage"):
        from .rovent_generation import _plain_id
        _plain_id(value["stage_id"])
    elif value["stage_id"] is not None:
        raise ValueError("unexpected stage override")
    if value["action"] == "audit_legacy_intake":
        intake = fields(value["legacy_intake"], ("format", "artifact", "semantic_sha256", "stage_id", "teacher_actor_id",
            "expected_model_sha256", "expected_runtime_sha256", "corpus_root", "index_root"), "original legacy intake")
        _pin(intake["artifact"])
        sha(intake["semantic_sha256"])
        sha(intake["expected_model_sha256"])
        sha(intake["expected_runtime_sha256"])
        if type(intake["format"]) is not str:
            raise ValueError("typed original format required")
        if intake["format"] == "rz-pals-generation-chunk-manifest/1":
            _path(intake["corpus_root"], directory=True)
            _path(intake["index_root"], directory=True)
        elif intake["corpus_root"] is not None or intake["index_root"] is not None:
            raise ValueError("unexpected corpus/index path for whole JSON or unsupported format")
    elif value["legacy_intake"] is not None:
        raise ValueError("unexpected legacy intake input")
    _path(value["output_root"], directory=True)
    name = value["output_name"]
    if (type(name) is not str or not 1 <= len(name) <= 128 or not name.endswith(".json")
            or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-." for c in name)
            or name in (".", "..") or name.startswith(".")):
        raise ValueError("bounded plain output JSON basename required")
    _path(str(Path(value["output_root"]) / name))
    for key, maximum in (("max_records", 4096), ("max_output_bytes", MAX_JSON_BYTES), ("max_wall_ns", MAX_WALL_NS)):
        if not uint(value[key], maximum):
            raise ValueError("nonzero finite preparation resource limits required")
    return value


def prepare_request(value, *, budget):
    """Reusable file consumer. It does not publish until all checks finish."""
    value = validate_request(value)
    recipe = SeriesRecipe(read_pinned_json(value["recipe"], budget))
    checkpoint = None if value["checkpoint"] is None else read_pinned_json(value["checkpoint"], budget)
    action = value["action"]
    if action == "validate_recipe":
        result = {"recipe_sha256": recipe.sha256, "split_sha256": recipe.split_sha256,
                  "student_id": recipe.to_dict()["student_id"], "learner_id": recipe.to_dict()["learner_id"],
                  "stages": [stage.to_dict() for stage in recipe.stages]}
    elif action == "validate_checkpoint_metadata":
        result = {"checkpoint": validate_checkpoint_metadata(recipe, checkpoint), "checkpoint_bytes_loaded": False,
                  "actual_optimizer_state_verified": False}
    elif action == "prepare_resume":
        result = resume_plan(recipe, checkpoint, stage_id=value["stage_id"])
    elif action == "prepare_fork":
        result = fork_plan(recipe, checkpoint, stage_id=value["stage_id"])
    elif action == "prepare_common_forks":
        result = common_teacher_forks(recipe, checkpoint)
    elif action == "audit_global_lineage":
        records = read_pinned_json(value["origins"], budget)
        if type(records) is not list or len(records) > value["max_records"]:
            raise ValueError("global lineage metadata record capacity exceeded")
        index = GlobalLineageIndex(recipe, records)
        result = index.admit_checkpoint_lineage(recipe, checkpoint,
            ancestors=[read_pinned_json(pin, budget) for pin in value["ancestors"]])
    elif action in TRAINING_ACTIONS:
        from .rovent_training import prepare_export, prepare_training_stage, validate_checkpoint_inventory
        training = value["training_preparation"]
        membership = read_pinned_json(training["membership"], budget)
        if action == "prepare_training_stage":
            inventory = None if training["inventory"] is None else read_pinned_json(training["inventory"], budget)
            result = prepare_training_stage(recipe, membership, stage_id=value["stage_id"], mode=training["mode"],
                checkpoint=checkpoint, inventory=inventory)
        else:
            inventory = read_pinned_json(training["inventory"], budget)
            result = (validate_checkpoint_inventory(recipe, membership, inventory) if action == "validate_training_inventory" else
                      prepare_export(recipe, membership, inventory, kind=training["export_kind"]))
    else:
        from .generation_data import digest
        intake = value["legacy_intake"]
        original = read_pinned_json(intake["artifact"], budget)
        options = {key: intake[key] for key in ("stage_id", "teacher_actor_id", "expected_model_sha256", "expected_runtime_sha256")}
        options.update(max_records=value["max_records"], check_budget=budget.check)
        if intake["format"] == "legacy_whole_json_bundle":
            if digest("rz-rovent-original-legacy-bundle/1", original) != intake["semantic_sha256"]:
                raise ValueError("original whole-bundle semantic pin mismatch")
            result = audit_legacy_bundle(recipe, original, **options)
        elif intake["format"] == "rz-pals-generation-chunk-manifest/1":
            corpus = _path(intake["corpus_root"], directory=True)
            index_root = _path(intake["index_root"], directory=True)
            _plain_chain(corpus, directory=True)
            _plain_chain(index_root, directory=True)
            # Check every declared chunk path before the original reader opens
            # any corpus file. No discovery scan or heavyweight fallback.
            from .generation_chunks import validate_manifest
            validate_manifest(original)
            for chunk in original["chunks"]:
                _path(str(corpus / chunk["path"]), directory=True)
            result = audit_legacy_chunks(recipe, original, expected_manifest_sha256=intake["semantic_sha256"],
                corpus_root=corpus, index_root=index_root, **options)
        else:
            result = {"status": LegacyIntakeStatus.UNSUPPORTED_FORMAT.value,
                      "code": "OriginalFormatNotSupported", "original_format": intake["format"]}
    budget.check()
    result = {"schema": RESULT_DOMAIN, "action": action, "recipe_sha256": recipe.sha256,
            "input_pins": {key: value[key] for key in ("recipe", "checkpoint", "ancestors", "origins", "legacy_intake")},
            "result": result, "actual_training_executed": False, "optimizer_updates": 0,
            "execution_authorized": False, "arena_eligible": False}
    if value["schema"] == TRAINING_REQUEST_DOMAIN:
        result["input_pins"]["training_preparation"] = value["training_preparation"]
    return result


def write_result_noreplace(root, name, result, *, maximum):
    root = _path(root, directory=True)
    _plain_chain(root, directory=True)
    target = _path(str(root / name))
    if type(name) is not str or Path(name).name != name or target.parent != root:
        raise ValueError("preparation output must use a basename in its declared root")
    if target.exists() or target.is_symlink():
        raise FileExistsError("preparation result already exists")
    content = (json.dumps(result, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n").encode()
    if len(content) > uint(maximum, MAX_JSON_BYTES):
        raise ValueError("preparation output exceeds declared byte credit")
    descriptor, temporary = tempfile.mkstemp(prefix="rz-rovent-preparation-", suffix=".partial", dir=root)
    partial = Path(temporary)
    original = _stamp(os.fstat(descriptor))[:2]
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        if partial.exists() and _stamp(partial.lstat())[:2] == original:
            partial.unlink()
        raise
    # On uncertain publication preserve the completed partial/final for the
    # caller. Never delete another owner's recreated path or replace evidence.
    try:
        _plain_chain(root, directory=True)
        publish_noreplace(temporary, target)
    except OSError as error:
        if getattr(error, "publication_state", None) == "not_published" and partial.exists() and _stamp(partial.lstat())[:2] == original:
            partial.unlink()
        raise
    return {"path": str(target), "sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True)
    parser.add_argument("--request-sha256", required=True)
    args = parser.parse_args(argv)
    try:
        request_path = _path(args.request)
        detail = _plain_chain(request_path)
        budget = PreparationBudget(max_wall_ns=MAX_WALL_NS)
        value = read_pinned_json({"path": str(request_path), "sha256": args.request_sha256, "bytes": detail.st_size}, budget)
        value = validate_request(value)
        budget.until_ns = min(budget.until_ns, budget.clock() + value["max_wall_ns"])
        result = prepare_request(value, budget=budget)
        receipt = write_result_noreplace(value["output_root"], value["output_name"], result, maximum=value["max_output_bytes"])
        print(json.dumps({"status": "preparation_saved", "action": value["action"], "receipt": receipt,
                          "execution_authorized": False}, ensure_ascii=False, separators=(",", ":")))
        return 0
    except (ValueError, OSError, UnicodeError) as error:
        print(json.dumps({"status": "preparation_rejected", "error_type": type(error).__name__,
                          "code": "PreparationRejected", "execution_authorized": False}, separators=(",", ":")), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
