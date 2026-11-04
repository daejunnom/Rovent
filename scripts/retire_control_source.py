#!/usr/bin/env python3
"""Preserve and retire one interrupted, regenerable control source image.

The caller must select exact evidence and explicitly attest that no privileged
writer outside the observable namespace is using the leaf. This tool does not
recover an output lease or infer physical GPU completion. Default is dry-run.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import time
import zipfile

SOURCE_NAMES = frozenset({
    "storage_guard.py", "managed_storage.py", "managed_outputs.py",
    "managed_output_quarantine.py", "managed_output_quarantine_transition.py",
    "managed_output_preservation_session.py", "managed_output_unattributed_preservation.py",
    "managed_process.py", "run_managed.py",
})
ROSTER = SOURCE_NAMES | {"preparation-start.json", "failure.json"}
MAX_FILE = 1 << 20
MAX_RECORDS = 4096
CANONICAL_PARENT = Path('/root/.cache/rovezero/control-source-inputs-v2')
CANONICAL_CATALOG = Path('/root/.cache/rovezero/storage/outputs-v1')
CANONICAL_REGISTRY = Path('/root/.cache/rovezero/storage/outputs-v1.interrupted-unbound-custody-v1')


class Refusal(RuntimeError):
    pass


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise Refusal(reason)


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def stamp(st: os.stat_result) -> tuple[int, ...]:
    return (st.st_dev, st.st_ino, st.st_mode, st.st_uid, st.st_gid,
            st.st_nlink, st.st_size, st.st_mtime_ns, st.st_ctime_ns)


def read_regular(parent_fd: int, name: str) -> tuple[int, tuple[int, ...], bytes]:
    require('/' not in name and name not in ('', '.', '..'), 'invalid direct file name')
    fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=parent_fd)
    try:
        before = stamp(os.fstat(fd))
        require(stat.S_ISREG(before[2]) and before[3] == before[4] == 0
                and before[5] == 1 and 0 < before[6] <= MAX_FILE,
                'expected bounded root-owned single-link file')
        data = bytearray()
        while len(data) < before[6]:
            block = os.read(fd, min(65536, before[6] - len(data)))
            require(bool(block), 'unexpected EOF')
            data.extend(block)
        require(not os.read(fd, 1), 'file grew while read')
        require(stamp(os.fstat(fd)) == before == stamp(os.stat(name, dir_fd=parent_fd,
                                                               follow_symlinks=False)),
                'file identity changed')
        return fd, before, bytes(data)
    except BaseException:
        os.close(fd)
        raise


def direct_dir(path: Path) -> int:
    require(path.is_absolute() and '..' not in path.parts, 'absolute normalized directory required')
    fd = os.open('/', os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        for name in path.parts[1:]:
            child = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                            dir_fd=fd)
            os.close(fd)
            fd = child
        return fd
    except BaseException:
        os.close(fd)
        raise


def private_dir(fd: int) -> tuple[int, ...]:
    s = stamp(os.fstat(fd))
    require(stat.S_ISDIR(s[2]) and s[3] == s[4] == 0
            and stat.S_IMODE(s[2]) == 0o700, 'root-owned private directory required')
    return s


def secure_artifact_parent(path: Path) -> int:
    """Hold a private root directory with no writable untrusted ancestors."""
    for ancestor in (path, *path.parents):
        fd = direct_dir(ancestor)
        try:
            info = os.fstat(fd)
            require(info.st_uid == 0 and not (info.st_mode & (stat.S_IWGRP | stat.S_IWOTH)),
                    'artifact path has an untrusted writable ancestor')
        finally:
            os.close(fd)
    fd = direct_dir(path)
    try:
        private_dir(fd)
        return fd
    except BaseException:
        os.close(fd)
        raise


def scan_references(directory: Path, needle: bytes) -> tuple[tuple[str, str], ...]:
    """Only the named root-owned catalog/registry metadata; not payload data."""
    directory_fd = direct_dir(directory)
    rows: list[tuple[str, str]] = []
    try:
        names = sorted(name for name in os.listdir(directory_fd) if name.endswith('.json'))
        require(len(names) <= MAX_RECORDS, 'metadata reference bound exceeded')
        for name in names:
            fd, _, raw = read_regular(directory_fd, name)
            try:
                require(needle not in raw, 'protected metadata directly references selected leaf')
                # Decode JSON as well: escaping may hide a direct path in the
                # wire representation. Refuse malformed or unbounded trees.
                values = [json.loads(raw)]
                visits = 0
                while values:
                    value = values.pop()
                    visits += 1
                    require(visits <= 50000, 'metadata reference structure bound exceeded')
                    if isinstance(value, str):
                        require(needle.decode() not in value, 'protected metadata references selected leaf')
                    elif isinstance(value, dict):
                        values.extend(value.keys())
                        values.extend(value.values())
                    elif isinstance(value, list):
                        values.extend(value)
                rows.append((name, digest(raw)))
            finally:
                os.close(fd)
        return tuple(rows)
    finally:
        os.close(directory_fd)


def check_open_fds(leaf: Path) -> None:
    """An observable FD check, not proof about other namespaces or writers."""
    result = subprocess.run(['lsof', '-t', '+D', str(leaf)], capture_output=True,
                            check=False, timeout=20)
    require(result.returncode in (0, 1) and not result.stderr,
            'open-file observation unavailable')
    pids = {int(value) for value in result.stdout.splitlines()}
    require(not (pids - {os.getpid()}), 'selected leaf has another observed open process')


def open_catalog_lock(catalog: Path) -> int:
    catalog_fd = direct_dir(catalog)
    try:
        fd = os.open('catalog.lock', os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=catalog_fd)
        info = os.fstat(fd)
        require(stat.S_ISREG(info.st_mode) and info.st_uid == 0 and info.st_nlink == 1,
                'catalog guard identity invalid')
        require(stamp(info) == stamp(os.stat('catalog.lock', dir_fd=catalog_fd,
                                             follow_symlinks=False)), 'catalog guard replaced')
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return fd
    finally:
        os.close(catalog_fd)


def write_exclusive(parent_fd: int, name: str, raw: bytes) -> None:
    fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                 0o600, dir_fd=parent_fd)
    try:
        with os.fdopen(fd, 'wb', closefd=False) as stream:
            require(stream.write(raw) == len(raw), 'short artifact write')
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(fd)
    os.fsync(parent_fd)


def append_journal(fd: int, record: dict) -> None:
    raw = (json.dumps(record, sort_keys=True, separators=(',', ':')) + '\n').encode()
    offset = 0
    while offset < len(raw):
        written = os.write(fd, raw[offset:])
        require(written > 0, 'short journal write')
        offset += written
    os.fsync(fd)


def retire(args: argparse.Namespace) -> dict:
    require(os.name == 'posix' and os.geteuid() == 0, 'Linux root required')
    require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,127}', args.leaf) is not None
            and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,127}', args.sibling) is not None
            and args.leaf != args.sibling, 'two distinct direct image names required')
    parent = Path(args.parent)
    catalog = Path(args.catalog)
    registry = Path(args.registry)
    require(parent == CANONICAL_PARENT and catalog == CANONICAL_CATALOG
            and registry == CANONICAL_REGISTRY, 'canonical source and metadata roots required')
    require(type(args.timeout) is int and 1 <= args.timeout <= 600, 'finite maintenance timeout required')
    until = time.monotonic() + args.timeout
    def tick() -> None:
        require(time.monotonic() < until, 'maintenance deadline exceeded; preserve archive and journal')
    leaf = parent / args.leaf
    require(not (Path(args.artifacts) == parent or Path(args.artifacts).is_relative_to(parent)),
            'artifact output must be outside source parent')
    parent_fd = direct_dir(parent)
    held: list[int] = []
    lock_fd = None
    try:
        parent_stamp = private_dir(parent_fd)
        fcntl.flock(parent_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        lock_fd = open_catalog_lock(catalog)
        catalog_lock_stamp = stamp(os.fstat(lock_fd))
        def recheck_catalog_lock() -> None:
            fd = direct_dir(catalog)
            try:
                require(stamp(os.stat('catalog.lock', dir_fd=fd, follow_symlinks=False))[:5]
                        == stamp(os.fstat(lock_fd))[:5] == catalog_lock_stamp[:5],
                        'catalog guard name or identity changed')
            finally:
                os.close(fd)
        leaf_fd = os.open(args.leaf, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                          dir_fd=parent_fd)
        held.append(leaf_fd)
        leaf_stamp = private_dir(leaf_fd)
        require(leaf_stamp[:2] == (args.expected_device, args.expected_inode)
                and stamp(os.stat(args.leaf, dir_fd=parent_fd, follow_symlinks=False)) == leaf_stamp,
                'original leaf identity differs')
        require(set(os.listdir(leaf_fd)) == ROSTER, 'unexpected or missing original entries')
        sibling_fd = os.open(args.sibling, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                             dir_fd=parent_fd)
        held.append(sibling_fd)
        sibling_stamp = private_dir(sibling_fd)
        require({'manifest.json', 'preparation-start.json'} | SOURCE_NAMES == set(os.listdir(sibling_fd)),
                'sibling image is not a complete source image')
        files: dict[str, tuple[int, tuple[int, ...], bytes]] = {}
        for name in sorted(ROSTER):
            tick()
            item = read_regular(leaf_fd, name)
            held.append(item[0])
            files[name] = item
        require(digest(files['preparation-start.json'][2]) == args.expected_start_sha256
                and digest(files['failure.json'][2]) == args.expected_failure_sha256,
                'unique failure evidence differs from selected pins')
        start, failure = (json.loads(files[name][2]) for name in ('preparation-start.json', 'failure.json'))
        require(start.get('status') == 'preparing_source_input_cache'
                and start.get('cache_leaf') == str(leaf)
                and start.get('cache_leaf_identity') == list(leaf_stamp[:2])
                and failure.get('cache_leaf') == str(leaf)
                and failure.get('status') == 'partial_cache_retained'
                and failure.get('terminal_sealed') is False,
                'leaf is not the selected interrupted source image')
        actual_boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        old_boot = start.get('copy_process_observation', {}).get('boot')
        require(isinstance(old_boot, str) and re.fullmatch(r'[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}', old_boot)
                and old_boot != actual_boot,
                'original producer boot has not changed')
        sibling_manifest_fd, _, sibling_raw = read_regular(sibling_fd, 'manifest.json')
        held.append(sibling_manifest_fd)
        sibling = json.loads(sibling_raw)
        require(sibling.get('status') == 'prepared_source_input_cache_only'
                and sibling.get('cache_preparation_complete') is True
                and sibling.get('cache_leaf') == str(parent / args.sibling)
                and sibling.get('cache_leaf_identity') == list(sibling_stamp[:2]),
                'sibling is not a completed source image')
        sibling_start_fd, _, sibling_start_raw = read_regular(sibling_fd, 'preparation-start.json')
        held.append(sibling_start_fd)
        sibling_start_pin = sibling.get('start_metadata', {}).get('raw_pin', {})
        require(sibling_start_pin.get('path') == str(parent / args.sibling / 'preparation-start.json')
                and sibling_start_pin.get('bytes') == len(sibling_start_raw)
                and sibling_start_pin.get('sha256') == digest(sibling_start_raw),
                'completed sibling start evidence differs')
        sibling_start = json.loads(sibling_start_raw)
        require(sibling_start.get('cache_leaf') == str(parent / args.sibling)
                and sibling_start.get('cache_leaf_identity') == list(sibling_stamp[:2]),
                'completed sibling original start identity differs')
        sibling_pins = list(sibling.get('policy_sources7', {}).values()) + list(
            sibling.get('supervisor_sources2', {}).values())
        require(len(sibling_pins) == len(SOURCE_NAMES), 'completed sibling role roster differs')
        pins_by_name = {}
        for value in sibling_pins:
            pin = value.get('root_owned_copy', {}).get('raw_pin', {})
            path = Path(pin.get('path', ''))
            require(path.parent == parent / args.sibling and path.name in SOURCE_NAMES
                    and path.name not in pins_by_name, 'completed sibling source pin differs')
            pins_by_name[path.name] = pin
        for name in sorted(SOURCE_NAMES):
            tick()
            fd, _, raw = read_regular(sibling_fd, name)
            held.append(fd)
            pin = pins_by_name[name]
            require(raw == files[name][2] and pin.get('bytes') == len(raw)
                    and pin.get('sha256') == digest(raw), 'source differs from completed sibling')
        needle = str(leaf).encode()
        references = (scan_references(catalog, needle), scan_references(registry, needle))
        check_open_fds(leaf)
        require(len(os.listdir(parent_fd)) <= 8, 'source image parent leaf bound exceeded')
        before_leaves = {name: stamp(os.stat(name, dir_fd=parent_fd, follow_symlinks=False))[:5]
                         for name in os.listdir(parent_fd) if name != args.leaf}
        inventory = {name: {'bytes': len(value[2]), 'sha256': digest(value[2])}
                     for name, value in files.items()}
        result = {'schema': 'rovezero.partial-control-image-retirement/1', 'leaf': str(leaf),
                  'identity': list(leaf_stamp[:2]), 'sibling': str(parent / args.sibling),
                  'original_boot_changed': True, 'files': inventory,
                  'protected_reference_counts': [len(r) for r in references],
                  'physical_completion': 'unknown_unchanged', 'retired': False,
                  'observation_limit': 'FD scan is current namespace only; privileged uncooperative writers are not excluded'}
        if not args.apply:
            return result
        require(args.assume_no_external_writers, 'explicit external-writer assumption required')
        artifact_fd = secure_artifact_parent(Path(args.artifacts))
        held.append(artifact_fd)
        artifact_stamp = stamp(os.fstat(artifact_fd))
        require(not Path(args.artifacts).is_relative_to(catalog)
                and not Path(args.artifacts).is_relative_to(registry), 'artifact path overlaps protected metadata')
        prefix = args.artifact_prefix
        require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,127}', prefix) is not None,
                'invalid fresh artifact prefix')
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, 'w', compression=zipfile.ZIP_STORED) as archive:
            for name in sorted(files):
                archive.writestr(name, files[name][2])
            archive.writestr('inventory.json', json.dumps(result, sort_keys=True).encode())
        raw_zip = buffer.getvalue()
        require(len(raw_zip) <= 2 * MAX_FILE, 'archive bound exceeded')
        zip_name, summary_name, journal_name = prefix + '.zip', prefix + '.json', prefix + '.journal.jsonl'
        write_exclusive(artifact_fd, zip_name, raw_zip)
        zfd = os.open(zip_name, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=artifact_fd)
        held.append(zfd)
        zip_stamp = stamp(os.fstat(zfd))
        require(stat.S_ISREG(zip_stamp[2]) and zip_stamp[5] == 1
                and zip_stamp[6] == len(raw_zip) and os.pread(zfd, len(raw_zip) + 1, 0) == raw_zip
                and zip_stamp == stamp(os.stat(zip_name, dir_fd=artifact_fd, follow_symlinks=False)),
                'published archive raw bytes or identity differs')
        with os.fdopen(os.dup(zfd), 'rb') as stream, zipfile.ZipFile(stream) as archive:
            require(archive.testzip() is None and len(archive.infolist()) == len(ROSTER) + 1
                    and set(archive.namelist()) == set(ROSTER) | {'inventory.json'},
                    'archive integrity or roster mismatch')
            for name in ROSTER:
                require(archive.read(name) == files[name][2], 'archive bytes differ from original')
        result['archive_sha256'] = digest(raw_zip)
        summary_raw = (json.dumps(result, sort_keys=True) + '\n').encode()
        write_exclusive(artifact_fd, summary_name, summary_raw)
        summary_fd, summary_stamp, summary_readback = read_regular(artifact_fd, summary_name)
        held.append(summary_fd)
        require(summary_readback == summary_raw, 'summary readback differs')
        journal_fd = os.open(journal_name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                             0o600, dir_fd=artifact_fd)
        held.append(journal_fd)
        os.fsync(artifact_fd)
        journal_identity = stamp(os.fstat(journal_fd))[:6]
        def check_evidence() -> None:
            named_artifact_fd = secure_artifact_parent(Path(args.artifacts))
            try:
                require(stamp(os.fstat(named_artifact_fd))[:5] == artifact_stamp[:5]
                        == stamp(os.fstat(artifact_fd))[:5], 'artifact parent identity changed')
            finally:
                os.close(named_artifact_fd)
            require(zip_stamp == stamp(os.fstat(zfd)) == stamp(os.stat(
                zip_name, dir_fd=artifact_fd, follow_symlinks=False))
                and os.pread(zfd, len(raw_zip) + 1, 0) == raw_zip,
                'saved archive changed before unlink')
            require(summary_stamp == stamp(os.fstat(summary_fd)) == stamp(os.stat(
                summary_name, dir_fd=artifact_fd, follow_symlinks=False))
                and os.pread(summary_fd, len(summary_raw) + 1, 0) == summary_raw,
                'saved summary changed before unlink')
            require(stamp(os.fstat(journal_fd))[:6] == journal_identity
                    == stamp(os.stat(journal_name, dir_fd=artifact_fd, follow_symlinks=False))[:6],
                    'saved journal identity changed')
        require(references == (scan_references(catalog, needle), scan_references(registry, needle)),
                'protected references changed before retirement')
        check_open_fds(leaf)
        remaining = set(ROSTER)
        for name in sorted(ROSTER):
            tick()
            recheck_catalog_lock()
            named_parent_fd = direct_dir(parent)
            try:
                require(stamp(os.fstat(named_parent_fd))[:5] == parent_stamp[:5]
                        == stamp(os.fstat(parent_fd))[:5], 'source parent identity changed')
            finally:
                os.close(named_parent_fd)
            check_evidence()
            require(set(os.listdir(leaf_fd)) == remaining, 'leaf roster changed before unlink')
            fd, original, raw = files[name]
            require(stamp(os.fstat(fd)) == original == stamp(os.stat(name, dir_fd=leaf_fd,
                                                                      follow_symlinks=False)),
                    'original source identity changed before unlink')
            require(os.pread(fd, len(raw) + 1, 0) == raw, 'original source bytes changed before unlink')
            append_journal(journal_fd, {'event': 'unlink_intent', 'name': name, 'sha256': digest(raw)})
            os.unlink(name, dir_fd=leaf_fd)
            os.fsync(leaf_fd)
            remaining.remove(name)
            append_journal(journal_fd, {'event': 'unlinked', 'name': name})
        require(not os.listdir(leaf_fd), 'leaf not empty after retirement')
        tick()
        recheck_catalog_lock()
        check_evidence()
        named_parent_fd = direct_dir(parent)
        try:
            require(stamp(os.fstat(named_parent_fd))[:5] == parent_stamp[:5], 'source parent identity changed')
        finally:
            os.close(named_parent_fd)
        require(stamp(os.stat(args.leaf, dir_fd=parent_fd, follow_symlinks=False))[:5] == leaf_stamp[:5],
                'leaf directory replaced')
        os.rmdir(args.leaf, dir_fd=parent_fd)
        os.fsync(parent_fd)
        require(before_leaves == {name: stamp(os.stat(name, dir_fd=parent_fd, follow_symlinks=False))[:5]
                                  for name in os.listdir(parent_fd)}, 'another leaf changed')
        require(references == (scan_references(catalog, needle), scan_references(registry, needle)),
                'protected references changed during retirement')
        check_evidence()
        append_journal(journal_fd, {'event': 'retired', 'physical_completion': 'unknown_unchanged'})
        result['retired'] = True
        return result
    finally:
        # Preserve the primary operation failure. Closing a descriptor cannot
        # turn a partially journaled removal into a different success result.
        import sys
        primary = sys.exc_info()[1]
        close_error = None
        for fd in (*reversed(held), *(() if lock_fd is None else (lock_fd,)), parent_fd):
            try:
                os.close(fd)
            except OSError as error:
                close_error = close_error or error
        if close_error is not None:
            if primary is not None and hasattr(primary, 'add_note'):
                primary.add_note('descriptor close failed after original operation: ' + str(close_error))
            elif primary is None:
                raise close_error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ('parent', 'leaf', 'sibling', 'catalog', 'registry', 'artifacts',
                   'artifact-prefix', 'expected-start-sha256', 'expected-failure-sha256'):
        parser.add_argument('--' + option, required=True)
    parser.add_argument('--expected-device', required=True, type=int)
    parser.add_argument('--expected-inode', required=True, type=int)
    parser.add_argument('--timeout', type=int, default=180)
    parser.add_argument('--assume-no-external-writers', action='store_true')
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    print(json.dumps(retire(args), sort_keys=True))


if __name__ == '__main__':
    main()
