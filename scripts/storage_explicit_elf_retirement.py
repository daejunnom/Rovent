"""Explicit, journaled retirement of old CPU-build ELF leaves; never automatic.

The caller must pin an exact plan, the original retained output record, build
receipt and every selected file. This path does not assert reproducibility,
physical GPU completion, reclaimed physical disk, or permission to recover a
lease. Unrelated evidence and the original catalog row remain in place.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys

import storage_artifact_review as review
from managed_storage import StorageError, read_record_snapshot
from managed_outputs import _catalog_rows, _refuse_custody_registry
from storage_consumer_registry import require_catalog_binding
from storage_guard import CatalogGuard, lifetime, owner_proven_dead
import storage_code_retirement as base

SCHEMA = "rovezero.explicit-retained-cpu-elf-retirement.v1"
AUTHORIZATION = "explicit-user-approved-retained-cpu-elf"
NAMES = frozenset({"rz-experiments", "nnuec_numeric_probe", "rz-rovent-native",
                   "rovent_startup_metadata"})
# A process census covers open/mapped files, not another catalog's registered
# future inputs. Until a locked dependency-closure check is connected, this
# entry point must fail before opening or modifying any path.
CROSS_CATALOG_DEPENDENCY_GATE_CONNECTED = False


def _plan(value):
    review.fields(value, ("schema", "authorization", "catalog", "key", "record_sha256",
                          "receipt", "source_commit", "files"))
    base._need(value["schema"] == SCHEMA and value["authorization"] == AUTHORIZATION,
               "explicit CPU ELF retirement authorization required")
    base._path(value["catalog"])
    base._need(isinstance(value["key"], str) and review.HEX.fullmatch(value["key"]) is not None
               and isinstance(value["record_sha256"], str)
               and review.HEX.fullmatch(value["record_sha256"]) is not None, "exact catalog pins required")
    review.raw_pin(value["receipt"], 1 << 20)
    base._need(isinstance(value["source_commit"], str)
               and len(value["source_commit"]) == 40
               and all(c in "0123456789abcdef" for c in value["source_commit"]),
               "exact recorded source commit required")
    files = value["files"]
    base._need(type(files) is list and 1 <= len(files) <= len(NAMES),
               "bounded explicit ELF leaves required")
    seen = set()
    for item in files:
        review.fields(item, ("relative", "bytes", "sha256"))
        base._need(item["relative"] in NAMES and item["relative"] not in seen,
                   "unique allowlisted ELF leaf required")
        review.raw_pin({"path": "/" + item["relative"], "bytes": item["bytes"],
                        "sha256": item["sha256"]}, review.ELF_CAP)
        seen.add(item["relative"])
    return value


def _receipt(value, plan):
    base._need(type(value) is dict
               and value.get("schema") == "rovent.managed-cpu-four-target-build.v1"
               and value.get("gpu_executed") is False
               and value.get("source_commit_requested_by_windows_git") == plan["source_commit"]
               and type(value.get("builds")) is dict, "matching CPU-only build receipt required")
    builds = value["builds"]
    for item in plan["files"]:
        matching = [row for row in builds.values() if type(row) is dict
                    and row.get("artifact") == item["relative"]]
        base._need(len(matching) == 1 and matching[0].get("returncode") == 0
                   and matching[0].get("retained") == {"bytes": item["bytes"],
                                                          "sha256": item["sha256"]},
                   "each candidate must match its original successful build receipt")


def observe_explicit_cpu_elf_dependencies(plan_pin, *, output_catalogs=(), build_catalogs=(), seconds=300):
    """Observe named catalogs for plan leaves, without granting unlink authority.

    This checks the plan pin, but is not a retirement preflight: no owner,
    candidate, receipt, process, GPU completion, or complete catalog closure
    is certified. The retirement gate remains independent and closed.
    """
    if sys.platform != "linux":
        raise base.CodeCacheRetirementUnsupported("explicit CPU ELF observation requires Linux")
    from storage_catalog_dependencies import observe

    inputs = base._Inputs(seconds)
    first = result = None
    try:
        plan = _plan(review.decode(inputs.open(plan_pin, 1 << 20).read((1 << 20) + 1)))
        receipt = base._path(plan["receipt"]["path"])
        base._need(receipt.name == "build-receipt.json", "original receipt filename required")
        catalogs = [plan["catalog"]]
        catalogs.extend(path for path in output_catalogs if str(path) != plan["catalog"])
        observations = []
        for item in plan["files"]:
            inputs.tick()
            result_for_leaf = observe(receipt.parent / item["relative"],
                                      output_catalogs=catalogs, build_catalogs=build_catalogs)
            observations.append(result_for_leaf)
        inputs.check()
        result = {"schema": "rovezero.explicit-cpu-elf-dependency-observation.v1",
                  "observations": observations, "candidate_locations_inferred_from_receipt": True,
                  "retirement_preflight": False, "deletion_authorized": False}
    except BaseException as error:
        first = error
    finally:
        first = inputs.close(first)
    if first is not None:
        raise first
    return result


def retire_explicit_cpu_elf(plan_pin, *, seconds=300, dependency_registry=None):
    """Retire only pinned leaves of one completed original CPU output."""
    if sys.platform != "linux":
        raise base.CodeCacheRetirementUnsupported("explicit CPU ELF retirement requires Linux")
    if not CROSS_CATALOG_DEPENDENCY_GATE_CONNECTED:
        raise StorageError("cross-catalog active/unknown dependency protection is not connected")
    if dependency_registry is None:
        raise StorageError("a shared consumer dependency registry is required")
    inputs = base._Inputs(seconds)
    guard = journal = dependency_guard = dependency_manager = None
    dependency_guard_entered = False
    first = result = None
    retired, removed, attempted = [], 0, None
    evidence = {"schema": "rovezero.explicit-retained-cpu-elf-failure.v1",
                "status": "FAILED_OR_PARTIAL_PRESERVED", "plan": plan_pin,
                "retired_files": retired, "logical_removed_bytes": 0, "journal": None,
                "journal_path": None, "partial": False,
                "physical_completion_attested": False, "physical_space_reclaimed": "unknown",
                "vhd_return": "unknown", "observation_limits": list(base._LIMITS)}
    try:
        plan = _plan(review.decode(inputs.open(plan_pin, 1 << 20).read((1 << 20) + 1)))
        catalog = base._path(plan["catalog"])
        inputs.directory(catalog, controlled=True)
        guard = CatalogGuard(catalog / "catalog.lock")
        guard.acquire()
        # A registry selected by the caller has no cleanup authority unless
        # this exact catalog is bound to it. In particular an unconfigured
        # marker or a different registry must preserve the retirement target.
        require_catalog_binding(catalog, dependency_registry, guard)
        # Catalog -> registry is the common lock order.  Keep the registry
        # lock until after the last unlink: a consumer must not register an
        # input in the gap between this check and that unlink.
        dependency_manager = dependency_registry.guard()
        dependency_guard = dependency_manager.__enter__()
        dependency_guard_entered = True
        _refuse_custody_registry(catalog)
        rows = _catalog_rows(catalog)
        base._need(not any(row["state"] == "active" for _, row, _ in rows),
                   "active catalog row prevents retirement")
        matches = [entry for entry in rows if entry[1]["key"] == plan["key"]]
        base._need(len(matches) == 1, "exact original output row required")
        record_path, row, output = matches[0]
        output = base._path(output)
        inputs.directory(output, controlled=True)
        original = read_record_snapshot(record_path, schema="rovezero.managed-output.v1")
        base._need(original[0] == row and original[1] == plan["record_sha256"],
                   "original output record pin differs")
        base._need(row["state"] == "retained" and row.get("tree_gone") is True
                   and row.get("budget_error") is None and row.get("deferred") is False,
                   "completed, nondeferred budget-clean output required")
        observer = lifetime()
        outer = row.get("outer_supervisor", {}).get("owner", {})

        def owners_dead():
            return (owner_proven_dead(row.get("pid"), row.get("lifetime"), observer)
                    and owner_proven_dead(outer.get("pid"), outer.get("lifetime"), observer))

        base._need(owners_dead(), "original output and outer owners must both be proven dead")
        snapshots = {str(path): read_record_snapshot(path, schema="rovezero.managed-output.v1")
                     for path, _, _ in rows}
        for path in (record_path, output / ".rovezero-output.json"):
            inputs.open({"path": str(path), "bytes": path.lstat().st_size,
                         "sha256": plan["record_sha256"]}, 8192)
        base._need(base._path(plan["receipt"]["path"]) == output / "build-receipt.json",
                   "receipt must be original output build-receipt.json")
        receipt = review.decode(inputs.open(plan["receipt"], 1 << 20).read((1 << 20) + 1))
        _receipt(receipt, plan)
        candidates = []
        for item in plan["files"]:
            path = base._path(output / item["relative"])
            stream = inputs.open({"path": str(path), "bytes": item["bytes"],
                                  "sha256": item["sha256"]}, review.ELF_CAP)
            base._need(stream.read(4) == b"\x7fELF", "actual ELF bytes required")
            inputs.candidates.add(path)
            candidates.append(path)
        target_inodes = {base._inode(inputs.files[path]["before"]) for path in candidates}
        base._need(len(target_inodes) == len(candidates)
                   and all(base._inode(item["before"]) not in target_inodes
                           for path, item in inputs.files.items() if path not in inputs.candidates),
                   "candidate inode alias is forbidden")
        targets = target_inodes | {base._inode(inputs.directories[output][1])}

        def dependencies_clear():
            observation = dependency_guard.conflicts(candidates)
            base._need(isinstance(observation, dict)
                       and isinstance(observation.get("conflicts"), list)
                       and isinstance(observation.get("unknown"), list)
                       and observation.get("coverage") == "registered_only",
                       "consumer dependency observation malformed")
            base._need(not observation["conflicts"] and not observation["unknown"],
                       "active or unknown consumer dependency prevents retirement")

        def recheck():
            inputs.check()
            guard.assert_held()
            require_catalog_binding(catalog, dependency_registry, guard)
            dependencies_clear()
            _refuse_custody_registry(catalog)
            current = _catalog_rows(catalog)
            base._need(current == rows and all(read_record_snapshot(path, schema="rovezero.managed-output.v1")
                       == snapshots[str(path)] for path, _, _ in current), "catalog or marker changed")
            base._need(lifetime() == observer and owners_dead(), "original owner proof changed")
            if journal is not None:
                journal.check()

        recheck()
        census = base._process_census(targets, inputs.fds() | {guard.stream.fileno()}, inputs.tick)
        recheck()
        header = {"schema": "rovezero.explicit-retained-cpu-elf-journal.v1", "event": "preflight_complete",
                  "plan": plan_pin, "record_sha256": plan["record_sha256"], "key": plan["key"],
                  "receipt": plan["receipt"], "source_commit_recorded": plan["source_commit"],
                  "source_reproducibility": "not_asserted", "code": plan["files"], "census": census,
                  "consumer_dependency_coverage": "registered_only",
                  "legacy_consumer_coverage": "unknown",
                  "physical_completion_attested": False, "observation_limits": list(base._LIMITS)}
        reserved = len(base._encode(header)) + len(candidates) * 1024 + 16384
        base._need(reserved <= base._JOURNAL_CAP, "bounded retirement journal required")
        base._budget(rows, plan["key"], reserved, 0, inputs)
        name = ".rovezero-explicit-cpu-elf-retirement-" + plan_pin["sha256"] + ".jsonl"
        evidence["journal_path"] = str(output / name)
        journal = base._Journal(inputs, output, name, reserved)
        journal.append(header)
        os.fsync(inputs.directory(output))
        for index, path in enumerate(candidates):
            recheck()
            base._budget(rows, plan["key"], reserved, journal.written, inputs)
            attempted = {"index": index, "unlink_attempted": False, "unlink_returned": False}
            journal.append({"event": "unlink_intent", "index": index})
            recheck()
            inputs.hash_file(path)
            base._process_census(targets, inputs.fds() | {guard.stream.fileno(), journal.fd}, inputs.tick)
            recheck()
            attempted["unlink_attempted"] = True
            os.unlink(path.name, dir_fd=inputs.directories[path.parent][0])
            attempted["unlink_returned"] = True
            inputs.deleted.add(path)
            retired.append(plan["files"][index]["relative"])
            removed += plan["files"][index]["bytes"]
            os.fsync(inputs.directories[path.parent][0])
            inputs.check_file(path)
            journal.append({"event": "unlinked_directory_fsynced", "index": index,
                            "logical_removed_bytes": plan["files"][index]["bytes"]})
        recheck()
        journal.append({"event": "complete", "retired_files": len(retired), "logical_removed_bytes": removed})
        result = {"schema": "rovezero.explicit-retained-cpu-elf-result.v1", "status": "RETIRED",
                  "retired_files": list(retired), "logical_removed_bytes": removed, "journal": journal.pin(),
                  "original_row_unchanged": True, "source_reproducibility": "not_asserted",
                  "consumer_dependency_coverage": "registered_only", "legacy_consumer_coverage": "unknown",
                  "physical_completion_attested": False, "physical_space_reclaimed": "unknown",
                  "vhd_return": "unknown", "census": census, "observation_limits": list(base._LIMITS)}
    except BaseException as error:
        first = error
        if journal is not None:
            try:
                journal.append({"event": "failed_or_partial_preserved", "type": type(error).__name__[:128],
                                "detail": base._describe(error), "retired_files": len(retired),
                                "logical_removed_bytes": removed, "last_attempt": attempted})
            except BaseException as secondary:
                base._note(first, "partial journal append/fsync failed: " + base._describe(secondary))
    finally:
        evidence.update(logical_removed_bytes=removed,
                        partial=bool(retired) or bool(attempted and attempted["unlink_attempted"]))
        if journal is not None:
            try:
                evidence["journal"] = journal.pin()
            except BaseException as error:
                if first is None:
                    first = error
                else:
                    base._note(first, "journal pin failed: " + base._describe(error))
            try:
                journal.close()
            except BaseException as error:
                if first is None:
                    first = error
                else:
                    base._note(first, "journal close failed: " + base._describe(error))
        first = inputs.close(first)
        if dependency_guard_entered:
            try:
                dependency_manager.__exit__(None, None, None)
            except BaseException as error:
                if first is None:
                    first = error
                else:
                    base._note(first, "consumer dependency guard release failed: " + base._describe(error))
        if guard is not None:
            try:
                guard.release()
            except BaseException as error:
                if first is None:
                    first = error
                else:
                    base._note(first, "catalog guard release failed: " + base._describe(error))
    if first is not None:
        try:
            first.code_cache_retirement = evidence
        except BaseException:
            pass
        raise first
    return result
