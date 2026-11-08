#!/usr/bin/env python3
"""Explicit bounded inventories and exact-owner recovery; never age-only delete."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tarfile
import zipfile
import zlib

sys.dont_write_bytecode = True
# Permit an isolated -I -S invocation without caller PYTHONPATH or user packages.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from managed_storage import StorageError, checked, recover_build, tree_size
from managed_outputs import recover_output


def inventory(roots):
    selected = [checked(path) for path in roots]
    if len(selected) > 16 or any(a == b or a.is_relative_to(b) or b.is_relative_to(a)
                               for i, a in enumerate(selected) for b in selected[i + 1:]):
        raise StorageError("at most 16 disjoint explicit roots required")
    rows = []
    for root in selected:
        if not root.is_dir():
            raise StorageError("inventory requires an existing explicit root")
        children = list(root.iterdir())
        if len(children) > 1024:
            raise StorageError("inventory child bound exceeded")
        rows.append({"root": str(root), "length_bytes": tree_size(root),
            "children": [{"name": path.name, "length_bytes": tree_size(path),
                          "classification": "unclassified_preserved"} for path in sorted(children)]})
    return {"schema": "rovezero.storage-inventory.v1", "roots": rows,
            "sum_root_length_bytes": sum(row["length_bytes"] for row in rows),
            "cross_root_hardlink_deduplication": "not_performed",
            "whole_C_drive_or_VHD_physical_usage": "unknown", "deletion_performed": False}


def report_failure(error):
    # The durable journal is authoritative for the observed deletion prefix;
    # stdout remains empty on failure and no diagnostic can replace it.
    try:
        partial = getattr(error, "code_cache_retirement", None)
        if type(partial) is dict:
            raw = json.dumps(partial, ensure_ascii=True, sort_keys=True, allow_nan=False)
            if len(raw.encode("utf-8")) <= 32 << 10:
                print("code_cache_retirement_partial=" + raw, file=sys.stderr)
    except BaseException:
        pass
    try:
        detail = str(error)[:4096]
    except BaseException:
        detail = type(error).__name__
    try:
        print(f"storage_maintenance_error={detail}; unknown artifacts and leases preserved", file=sys.stderr)
    except BaseException:
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="operation", required=True)
    scan = commands.add_parser("inventory")
    scan.add_argument("--root", type=Path, action="append", required=True)
    build = commands.add_parser("recover-build")
    build.add_argument("--root", type=Path, required=True)
    build.add_argument("--owner-sha256", required=True)
    build.add_argument("--consumer-registry-root", type=Path,
                       help="Existing shared input-consumer registry; use with its pinned ID")
    build.add_argument("--consumer-registry-id", help="Pinned consumer registry identity")
    build.add_argument("--consumer-registry-required", action="store_true",
                       help="Preserve recovery inputs if the registry is unavailable")
    inspect = commands.add_parser("inspect-build-recovery", help="read-only exact-owner recovery gate diagnosis; no recovery or new lease")
    inspect.add_argument("--root", type=Path, required=True)
    inspect.add_argument("--owner-sha256")
    init_registry = commands.add_parser("init-consumer-registry", help="initialize a scoped registry or read an existing identity; never overwrite")
    init_registry.add_argument("--root", type=Path, required=True)
    init_registry.add_argument("--scope", required=True)
    init_registry.add_argument("--group-id", type=int)
    recover_registry = commands.add_parser("recover-empty-consumer-catalog", help="bind an empty required catalog; no asset deletion")
    recover_registry.add_argument("--catalog", type=Path, required=True)
    recover_registry.add_argument("--consumer-registry-root", type=Path, required=True)
    recover_registry.add_argument("--consumer-registry-id", required=True)
    deps = commands.add_parser("observe-catalog-dependencies", help="observe explicitly named catalogs; never authorizes deletion")
    deps.add_argument("--candidate", type=Path, required=True)
    deps.add_argument("--output-catalog", type=Path, action="append", default=[])
    deps.add_argument("--build-catalog", type=Path, action="append", default=[])
    output = commands.add_parser("recover-output")
    output.add_argument("--catalog", type=Path, required=True)
    output.add_argument("--key", required=True)
    output.add_argument("--record-sha256", required=True)
    research = commands.add_parser("verify-research", help="verify a pinned research ZIP; never delete")
    research.add_argument("--archive", type=Path, required=True)
    research.add_argument("--archive-bytes", type=int, required=True)
    research.add_argument("--archive-sha256", required=True)
    research.add_argument("--seconds", type=int, default=180)
    code = commands.add_parser("audit-code-cache", help="audit registered ELF duplicates; no unlink authority")
    code.add_argument("--plan", type=Path, required=True)
    code.add_argument("--plan-bytes", type=int, required=True)
    code.add_argument("--plan-sha256", required=True)
    code.add_argument("--seconds", type=int, default=300)
    retire = commands.add_parser("retire-code-cache", help="explicit Linux-only retirement of verified duplicate ELF leaves")
    retire.add_argument("--plan", type=Path, required=True)
    retire.add_argument("--plan-bytes", type=int, required=True)
    retire.add_argument("--plan-sha256", required=True)
    retire.add_argument("--seconds", type=int, default=300)
    explicit_elf = commands.add_parser("retire-explicit-cpu-elf", help="explicitly retire pinned old CPU ELF leaves; no automatic cleanup")
    explicit_elf.add_argument("--plan", type=Path, required=True)
    explicit_elf.add_argument("--plan-bytes", type=int, required=True)
    explicit_elf.add_argument("--plan-sha256", required=True)
    explicit_elf.add_argument("--seconds", type=int, default=300)
    explicit_observe = commands.add_parser("observe-explicit-cpu-elf-dependencies", help="observe plan leaves in named catalogs; never authorizes deletion")
    explicit_observe.add_argument("--plan", type=Path, required=True)
    explicit_observe.add_argument("--plan-bytes", type=int, required=True)
    explicit_observe.add_argument("--plan-sha256", required=True)
    explicit_observe.add_argument("--output-catalog", type=Path, action="append", default=[])
    explicit_observe.add_argument("--build-catalog", type=Path, action="append", default=[])
    explicit_observe.add_argument("--seconds", type=int, default=300)
    legacy = commands.add_parser("register-legacy-observation", help="record a bounded legacy build observation; no deletion authority")
    legacy.add_argument("--inputs-root", type=Path, required=True)
    legacy.add_argument("--directory", type=Path, required=True)
    legacy.add_argument("--output", type=Path, required=True)
    legacy.add_argument("--max-bytes", type=int, required=True)
    legacy.add_argument("--max-entries", type=int, default=4096)
    legacy.add_argument("--seconds", type=int, default=60)
    legacy_inspect = commands.add_parser("inspect-legacy-observation", help="read a registered legacy observation; no deletion authority")
    legacy_inspect.add_argument("--observation", type=Path, required=True)
    args = parser.parse_args()
    if args.operation == "inventory":
        result = inventory(args.root)
    elif args.operation == "recover-build":
        # Use the same explicitly pinned scope and unavailable behavior as the
        # build runner. Required-but-unavailable prevents the destructive part
        # of recovery without granting authority from an empty registry.
        from run_managed import _consumer_registry_options
        dependency_options = _consumer_registry_options(args.consumer_registry_root,
                                                        args.consumer_registry_id,
                                                        args.consumer_registry_required)
        result = recover_build(args.root, args.owner_sha256, **dependency_options)
    elif args.operation == "inspect-build-recovery":
        from storage_recovery_status import inspect_build_recovery
        result = inspect_build_recovery(args.root, args.owner_sha256)
    elif args.operation == "init-consumer-registry":
        from managed_storage import read_record_snapshot
        from storage_consumer_registry import ConsumerRegistry, REGISTRY_SCHEMA
        registry_root = checked(args.root)
        if os.path.lexists(registry_root):
            record, _, _ = read_record_snapshot(registry_root / "registry.json", schema=REGISTRY_SCHEMA)
            registry = ConsumerRegistry(registry_root, registry_id=record.get("registry_id"))
            if record.get("scope") != args.scope:
                raise StorageError("existing consumer registry scope differs")
            if args.group_id is not None and (record.get("sharing") != "group"
                                               or registry_root.stat().st_gid != args.group_id):
                raise StorageError("existing consumer registry group differs")
        else:
            registry = ConsumerRegistry.initialize(registry_root, scope=args.scope, group_id=args.group_id)
        result = {"registry_id": registry.registry_id}
    elif args.operation == "recover-empty-consumer-catalog":
        from storage_consumer_registry import ConsumerRegistry, recover_empty_required_catalog
        from storage_guard import CatalogGuard
        registry = ConsumerRegistry(args.consumer_registry_root, registry_id=args.consumer_registry_id)
        with CatalogGuard(checked(args.catalog) / "catalog.lock") as guard:
            bound = recover_empty_required_catalog(args.catalog, registry, guard)
        result = {"schema": "rovezero.empty-consumer-catalog-recovery.v1",
                  "catalog": str(checked(args.catalog)), "registry_id": bound.registry_id,
                  "nonempty_catalog_recovered": False, "asset_deletion_performed": False}
    elif args.operation == "observe-catalog-dependencies":
        from storage_catalog_dependencies import observe
        result = observe(args.candidate, output_catalogs=args.output_catalog, build_catalogs=args.build_catalog)
    elif args.operation == "recover-output":
        result = recover_output(args.catalog, args.key, args.record_sha256)
    elif args.operation == "verify-research":
        from storage_artifact_review import verify_research
        result = verify_research({"path": str(args.archive), "bytes": args.archive_bytes,
                                  "sha256": args.archive_sha256}, seconds=args.seconds)
    elif args.operation == "audit-code-cache":
        from storage_artifact_review import audit_code_cache
        result = audit_code_cache({"path": str(args.plan), "bytes": args.plan_bytes,
                                   "sha256": args.plan_sha256}, seconds=args.seconds)
    elif args.operation == "register-legacy-observation":
        from storage_legacy_inventory import observe
        observed = observe(args.inputs_root, args.directory, args.output, max_bytes=args.max_bytes,
                           max_entries=args.max_entries, seconds=args.seconds)
        result = {"schema": observed["schema"], "source": observed["source"],
                  "entry_count": observed["entry_count"], "total_file_bytes": observed["total_file_bytes"],
                  "registration_only": True, "deletion_authorized": False}
    elif args.operation == "inspect-legacy-observation":
        from storage_legacy_inventory import inspect_observation
        result = inspect_observation(args.observation)
    elif args.operation == "retire-explicit-cpu-elf":
        from storage_explicit_elf_retirement import retire_explicit_cpu_elf
        result = retire_explicit_cpu_elf({"path": str(args.plan), "bytes": args.plan_bytes,
                                          "sha256": args.plan_sha256}, seconds=args.seconds)
    elif args.operation == "observe-explicit-cpu-elf-dependencies":
        from storage_explicit_elf_retirement import observe_explicit_cpu_elf_dependencies
        result = observe_explicit_cpu_elf_dependencies(
            {"path": str(args.plan), "bytes": args.plan_bytes, "sha256": args.plan_sha256},
            output_catalogs=args.output_catalog, build_catalogs=args.build_catalog, seconds=args.seconds)
    else:
        from storage_code_retirement import retire_code_cache
        result = retire_code_cache({"path": str(args.plan), "bytes": args.plan_bytes,
                                    "sha256": args.plan_sha256}, seconds=args.seconds)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, StorageError, zipfile.BadZipFile, tarfile.TarError, EOFError, zlib.error) as error:
        report_failure(error)
        raise SystemExit(125)
