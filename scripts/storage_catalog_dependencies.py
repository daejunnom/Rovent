"""Read-only, scoped dependency observations; never a deletion capability.

Existing ownership records do not seal their consumers' complete input sets.
Consequently neither a lack of matches nor a completed observation proves that
an artifact is unused, and callers must not use this report as an unlink gate.
"""
from __future__ import annotations

from pathlib import Path

from managed_storage import StorageError, checked, read_record_snapshot

_LIMIT = 2048


def _overlap(left, right):
    return left == right or left.is_relative_to(right) or right.is_relative_to(left)


def observe(candidate, *, output_catalogs=(), build_catalogs=()):
    """Compare explicitly supplied catalogs; report partial data as unknown.

    This snapshot is neither locked across catalogs nor held across deletion.
    Files other than the small named ownership records are never opened.
    """
    target = checked(Path(candidate))
    catalogs = [("output", Path(p)) for p in output_catalogs]
    catalogs += [("build", Path(p)) for p in build_catalogs]
    if not catalogs or len(catalogs) > 32:
        raise StorageError("one to 32 explicit catalogs required")
    matches, unknown, observed = [], [], 0
    for kind, raw in catalogs:
        try:
            root = checked(raw)
            if not root.is_dir():
                raise StorageError("catalog directory unavailable")
            directory = root if kind == "output" else checked(root / "slots")
            entries = list(directory.iterdir())
            if len(entries) > _LIMIT:
                raise StorageError("catalog entry bound exceeded")
        except (OSError, StorageError):
            unknown.append({"catalog": str(raw), "reason": "catalog_unavailable_or_invalid"})
            continue
        for entry in entries:
            if kind == "output" and entry.name == "catalog.lock":
                continue
            if kind == "output" and entry.suffix != ".json":
                unknown.append({"catalog": str(root), "reason": "unexpected_catalog_entry"})
                continue
            record_path = entry if kind == "output" else entry / "record.json"
            try:
                record, _, _ = read_record_snapshot(
                    record_path, schema="rovezero.managed-output.v1" if kind == "output"
                    else "rovezero.managed-build.v1")
                observed += 1
                references = []
                for field in ("source", "output"):
                    value = record.get(field)
                    if value is not None:
                        references.append((field, value))
                # A declared list is evidence of references, not proof of a
                # complete input declaration for the existing legacy schema.
                declared = record.get("inputs")
                if type(declared) is list and len(declared) <= _LIMIT:
                    for item in declared:
                        references.append(("inputs", item if isinstance(item, str)
                                           else item.get("path") if type(item) is dict else None))
                else:
                    unknown.append({"record": str(record_path), "reason": "inputs_not_recorded_or_invalid"})
                for field, value in references:
                    if not isinstance(value, str) or not Path(value).is_absolute():
                        unknown.append({"record": str(record_path), "reason": "reference_invalid"})
                        continue
                    try:
                        reference = checked(Path(value))
                    except (OSError, StorageError):
                        unknown.append({"record": str(record_path), "reason": "reference_unavailable"})
                        continue
                    if _overlap(target, reference):
                        matches.append({"record": str(record_path), "state": record.get("state", "unknown"),
                                        "field": field, "reference": str(reference)})
            except (OSError, StorageError):
                unknown.append({"record": str(record_path), "reason": "record_unavailable_or_invalid"})
    return {"schema": "rovezero.catalog-dependency-observation.v1", "candidate": str(target),
            "records_observed": observed, "matches": matches, "unknown": unknown,
            "catalogs_locked_together": False, "input_completeness": "unknown",
            "deletion_authorized": False}
