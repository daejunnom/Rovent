"""Actual bounded source-byte pins shared with the compiled V5 inventory.

This is source registration, not a claim that every embedded component ran.
No model, neural library, optimizer, credential or generated tree is opened.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
import struct

DOMAIN = "rz-pals-source-component-v5/1"
CATALOG_DOMAIN = "rz-pals-source-inventory-v5/1"
ALGORITHM = "sha256-domain-u64le;sorted-path-u64le-utf8;size-u64le;sha256-raw32"
CATALOG = "crates/rz-experiments/src/pals_source_inventory_v5.json"
MAX_FILE = 4 * 1024 * 1024
MAX_TOTAL = 32 * 1024 * 1024


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate source inventory key")
        result[key] = value
    return result


def _path(path):
    if (type(path) is not str or not 1 <= len(path) <= 512
            or not path.startswith(("crates/", "scripts/", "experiments/model-research/pals/src/", ".github/workflows/"))
            or "\\" in path or ":" in path or any(ord(c) < 32 for c in path)
            or any(x in path.lower() for x in ("credential", "service-account", "service_account", "api-key", "api_key", "ssh-key"))):
        raise ValueError("source path is outside the registered boundary")
    for part in path.split("/"):
        if part.lower() in ("", ".", "..", "node_modules", "dist", "dist-server", "build", "coverage", "models", "checkpoints", ".cache") or part.lower().startswith(".env"):
            raise ValueError("source inventory cannot authorize secret or generated paths")
    if path != CATALOG and path.rsplit(".", 1)[-1] not in ("rs", "py", "toml", "yml", "yaml"):
        raise ValueError("unregistered source suffix")
    return path


def _read_pin(root, logical, maximum):
    logical = _path(logical)
    path = root
    for part in logical.split("/"):
        path = path / part
        if stat.S_ISLNK(path.lstat().st_mode):
            raise ValueError("source inventory path contains a symlink")
    flags = (os.O_RDONLY | getattr(os, "O_BINARY", 0)
             | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or not 1 <= before.st_size <= maximum:
            raise ValueError("source file type or size exceeds its bound")
        hasher = hashlib.sha256()
        raw_catalog = bytearray() if logical == CATALOG else None
        consumed = 0
        while consumed < before.st_size:
            block = os.read(descriptor, min(65536, before.st_size - consumed))
            if not block:
                break
            consumed += len(block)
            hasher.update(block)
            if raw_catalog is not None:
                raw_catalog.extend(block)
        extra = os.read(descriptor, 1)
        after = os.fstat(descriptor)
        if (consumed != before.st_size or extra or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
                != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)):
            raise ValueError("source file changed during registration")
        return (logical, consumed, hasher.digest()), bytes(raw_catalog) if raw_catalog is not None else None
    finally:
        os.close(descriptor)


def component_digest(files):
    if not 1 <= len(files) <= 256 or len({row[0] for row in files}) != len(files):
        raise ValueError("bounded unique source inventory required")
    domain = DOMAIN.encode("utf-8")
    hasher = hashlib.sha256(struct.pack("<Q", len(domain)) + domain)
    for path, size, digest in sorted(files):
        path = _path(path).encode("utf-8")
        if type(size) is not int or not 1 <= size <= MAX_FILE or type(digest) is not bytes or len(digest) != 32:
            raise ValueError("invalid actual source pin")
        hasher.update(struct.pack("<Q", len(path)) + path + struct.pack("<Q", size) + digest)
    return hasher.hexdigest()


def source_pins(names=None):
    root = Path(__file__).absolute().parents[5]
    catalog_pin, raw = _read_pin(root, CATALOG, 128 * 1024)
    catalog = json.loads(raw, object_pairs_hook=_object)
    if (set(catalog) != {"domain", "algorithm", "components"} or catalog["domain"] != CATALOG_DOMAIN
            or catalog["algorithm"] != ALGORITHM or type(catalog["components"]) is not dict
            or not 1 <= len(catalog["components"]) <= 64):
        raise ValueError("source inventory domain or algorithm differs")
    components = catalog["components"]
    selected = set(components) if names is None else set(names)
    if not selected or not selected <= set(components):
        raise ValueError("unregistered source component")
    actual = {CATALOG: catalog_pin}
    total = catalog_pin[1]
    result = {}
    for name in sorted(selected):
        paths = components[name]
        if (type(name) is not str or type(paths) is not list or not 1 <= len(paths) <= 256
                or any(type(p) is not str for p in paths) or len(set(paths)) != len(paths)):
            raise ValueError("invalid component source inventory")
        files = []
        for path in paths:
            _path(path)
            if path not in actual:
                pin, _ = _read_pin(root, path, MAX_FILE)
                total += pin[1]
                if total > MAX_TOTAL:
                    raise ValueError("source inventory exceeds aggregate budget")
                actual[path] = pin
            files.append(actual[path])
        result[name] = component_digest(files)
    # The same catalog defines every lookup; a changed path list cannot be
    # accepted after its rows were hashed. Caller also pins before/after runs.
    if _read_pin(root, CATALOG, 128 * 1024)[0] != catalog_pin:
        raise ValueError("source inventory changed during registration")
    return result
