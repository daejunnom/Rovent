"""Read a pinned *partial* generation archive without admitting it for training.

The temporary header projection is private to this function, is never written,
and grants no execution or training authority. Original chunk bytes and every
native/action evidence join are checked by the existing strict chunk reader.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import stat

from .generation_chunks import ChunkGenerationDataset, MANIFEST_DOMAIN, SPLITS, validate_manifest
from .generation_data import ACTUAL_SOURCE, _count_target_coverage, fields, sha, strict_json, validate_record

PARTIAL_DOMAIN = "rz-pals-generation-partial-chunk-manifest/1"
PARTIAL_SOURCE = "actual_selfplay_partial"
MAX_MANIFEST_BYTES = 16 * 1024 * 1024


def validate_partial_diagnostic(manifest_path, *, expected_raw_sha256, index_root):
    """Return only a diagnostic report; never return a reusable dataset owner."""
    expected = sha(expected_raw_sha256)
    path = Path(manifest_path)
    if not path.is_absolute() or path.name != "manifest.partial.json" or path.is_symlink():
        raise ValueError("an absolute unlinked original partial manifest is required")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    with os.fdopen(os.open(path, flags), "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or not 0 < before.st_size <= MAX_MANIFEST_BYTES:
            raise ValueError("partial manifest original type, link count or size is invalid")
        raw = stream.read(MAX_MANIFEST_BYTES + 1)
        after = os.fstat(stream.fileno())
        named = path.stat(follow_symlinks=False)
    identity = lambda item: (item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns, item.st_ctime_ns)
    if (len(raw) != before.st_size or identity(before) != identity(after) or identity(before) != identity(named)
            or hashlib.sha256(raw).hexdigest() != expected):
        raise ValueError("partial manifest changed or differs from the external raw SHA-256")
    original = strict_json(raw.decode("utf-8", "strict"))
    fields(original, ("schema", "source_kind", "producer", "opening_splits", "chunks", "total_bytes",
                      "total_records", "incomplete_reason"), "partial generation manifest")
    reason = original["incomplete_reason"]
    if (original["schema"] != PARTIAL_DOMAIN or original["source_kind"] != PARTIAL_SOURCE
            or type(reason) is not str or not reason or len(reason) > 4096):
        raise ValueError("the original archive is not a bounded partial generation")

    # Reuse the complete reader's exact record/evidence/native/action checks.
    # Neither this private projection nor its digest escapes this function.
    projection = {key: value for key, value in original.items() if key != "incomplete_reason"}
    projection["schema"], projection["source_kind"] = MANIFEST_DOMAIN, ACTUAL_SOURCE
    semantic_pin = validate_manifest(projection)
    counts = Counter()
    split_rows = {}
    with ChunkGenerationDataset(path.parent, projection, expected_manifest_sha256=semantic_pin,
                                index_root=index_root) as dataset:
        for split in SPLITS:
            rows = dataset.rows(split)
            split_rows[split] = len(rows)
            for row in rows:
                _count_target_coverage(counts, row, validate_record(row))
        exclusions = dataset.exclusions
        # Rehash all held chunks and immutable index before discarding the owner.
        dataset.sha256
    return {"schema": "rz-pals-partial-context-diagnostic/1", "status": "strict_content_validated",
            "partial_manifest_raw_sha256": expected, "incomplete_reason": reason,
            "raw_record_count": original["total_records"], "selected_rows_by_split": split_rows,
            "coverage": dict(counts), "exclusions": exclusions,
            "actual_training_admissible": False, "optimizer_updates_executed": 0,
            "scope": "partial archive content and original evidence joins only; no training or physical-execution authority"}


def main(argv=None):
    parser = argparse.ArgumentParser(description="strict, non-training audit of an original partial PALS archive")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--manifest-raw-sha256", required=True)
    parser.add_argument("--index-root", required=True)
    args = parser.parse_args(argv)
    report = validate_partial_diagnostic(args.manifest, expected_raw_sha256=args.manifest_raw_sha256,
                                         index_root=args.index_root)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
