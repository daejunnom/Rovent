#!/usr/bin/env python3
"""Windows-side, bounded RoveZero usage snapshot and cross-OS advisory.

The scheduled observer scans the default RoveZero roaming root and discovered
Codex package-local RoveZero roots, not arbitrary disks. The Linux public total
is read separately. These totals do not constitute host-wide allocation.
"""
from __future__ import annotations

from datetime import datetime, timezone
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import time
import uuid

GIB = 1024 ** 3
THRESHOLD = 18 * GIB
MAX_BYTES = 8192
MAX_AGE = 3600
WINDOWS_SCHEMA = 'rovezero.total-usage-windows-snapshot.v1'
LINUX_SCHEMA = 'rovezero.total-usage-public-snapshot.v1'
COMBINED_SCHEMA = 'rovezero.total-usage-combined-snapshot.v1'
ALLOCATED_MEASURE = 'sum_of_observed_unique_file_reported_allocations'
PAIRING_SCHEMA = 'rovezero.usage-pairing.v1'
MONITOR_CONFIG_SCHEMA = 'rovezero.storage-monitor-configuration.v1'
MONITOR_REGISTRY_KEY = r'Software\RoveZero\StorageMonitor'


def windows_monitor_installed():
    """Read the per-user installation marker; no monitor files are opened."""
    if os.name != 'nt':
        return False
    import winreg
    try:
        key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, MONITOR_REGISTRY_KEY)
    except FileNotFoundError:
        return False
    with key:
        try:
            value, kind = winreg.QueryValueEx(key, 'Enabled')
        except FileNotFoundError as exc:
            # An existing setup key with a missing marker is a damaged or
            # incomplete installation, not proof the monitor was never used.
            raise ValueError('monitor_installation_marker_missing') from exc
    if kind != winreg.REG_DWORD or value not in (0, 1):
        raise ValueError('invalid_monitor_installation_marker')
    return bool(value)


def _scoped_monitor_dir(value):
    """Validate a proposed installation/override against discovered roots."""
    path = Path(value)
    if not path.is_absolute() or '..' in path.parts:
        raise ValueError('monitor_override_not_absolute_normalized')
    target = _normalize_windows_path(path)
    # Any discovered spelling is a valid installation/override scope even
    # when several spellings resolve to the same directory identity.
    for root in _windows_root_candidates():
        allowed = _normalize_windows_path(root / 'storage' / 'total-usage')
        try:
            if os.path.commonpath([target, allowed]) == allowed:
                return path
        except ValueError:
            continue  # Different volumes are outside the approved root.
    raise ValueError('monitor_override_outside_rovezero')


def _installed_monitor_dir():
    """Resolve the installation-pinned directory independently of APPDATA."""
    import winreg
    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, MONITOR_REGISTRY_KEY) as key:
        raw, kind = winreg.QueryValueEx(key, 'Configuration')
    if kind != winreg.REG_SZ or not isinstance(raw, str) or len(raw) > 8192:
        raise ValueError('invalid_monitor_configuration')
    config = json.loads(raw)
    if (not isinstance(config, dict) or config.get('schema') != MONITOR_CONFIG_SCHEMA
            or not isinstance(config.get('directory'), str)
            or type(config.get('volume_serial')) is not int
            or type(config.get('file_index')) is not int):
        raise ValueError('invalid_monitor_configuration')
    path = Path(config['directory'])
    if not path.is_absolute() or '..' in path.parts:
        raise ValueError('invalid_monitor_configuration')
    identity = _identity_nofollow(path, directory=True)
    if (identity['volume_serial'], identity['file_index']) != (config['volume_serial'], config['file_index']):
        raise ValueError('installed_monitor_directory_changed')
    return path


def register_monitor_dir(value):
    """Explicit installation: validate a root and atomically publish its config."""
    if os.name != 'nt':
        raise OSError('windows_required')
    import winreg
    path = _scoped_monitor_dir(value)
    identity = _identity_nofollow(path, directory=True)
    config = {'schema': MONITOR_CONFIG_SCHEMA, 'directory': str(path),
              'volume_serial': identity['volume_serial'], 'file_index': identity['file_index']}
    with winreg.CreateKeyEx(winreg.HKEY_CURRENT_USER, MONITOR_REGISTRY_KEY, 0, winreg.KEY_SET_VALUE) as key:
        # One REG_SZ value holds the complete identity; Enabled is written last.
        winreg.SetValueEx(key, 'Configuration', 0, winreg.REG_SZ, json.dumps(config, sort_keys=True))
        winreg.SetValueEx(key, 'Enabled', 0, winreg.REG_DWORD, 1)
    return {'status': 'monitor_registered', 'schema': MONITOR_CONFIG_SCHEMA}


def monitor_dir():
    # Once installed, every process must use the same pinned directory even if
    # APPDATA is redirected or an unrelated environment override is present.
    # Missing/corrupt settings are errors, not permission to pick another root.
    if windows_monitor_installed():
        return _installed_monitor_dir()
    override = os.environ.get('RZ_USAGE_MONITOR_DIR')
    if override:
        return _scoped_monitor_dir(override)
    appdata = os.environ.get('APPDATA')
    if not appdata or not os.path.isabs(appdata):
        raise ValueError('appdata_unavailable')
    return Path(appdata) / 'RoveZero' / 'storage' / 'total-usage'


def _windows_root_candidates():
    """Find standard roots without exposing user or package names in output.

    Both paths are scanned together, allowing the scanner to deduplicate
    overlapping roots/hardlinks. Unknown installation or custom roots always
    keep the public coverage partial.
    """
    appdata = os.environ.get('APPDATA')
    if not appdata or not os.path.isabs(appdata):
        raise ValueError('appdata_unavailable')
    roots = []
    seen = set()

    def append_once(path):
        # Normalize lexically only. The scanner and pairing verifier perform
        # their own no-follow/handle checks before trusting a real directory.
        identity = _normalize_windows_path(path)
        if identity not in seen:
            seen.add(identity)
            roots.append(path)

    append_once(Path(appdata) / 'RoveZero')
    local = os.environ.get('LOCALAPPDATA')
    if local and os.path.isabs(local):
        packages = Path(local) / 'Packages'
        try:
            with os.scandir(packages) as children:
                for child in children:
                    if (child.name.casefold().startswith('openai.codex_')
                            and child.is_dir(follow_symlinks=False)):
                        candidate = Path(child.path) / 'LocalCache' / 'Roaming' / 'RoveZero'
                        if candidate.is_dir() and not candidate.is_symlink():
                            append_once(candidate)
        except OSError:
            pass  # public coverage remains partial
    return roots


def discover_windows_roots():
    return _distinct_windows_roots(_windows_root_candidates())


def _distinct_windows_roots(roots):
    """Remove proven aliases and covered same-volume descendants only.

    Canonical handle identity, not path spelling, establishes an alias.  A
    missing or uninspectable root is retained so the scanner can report it as
    partial.  A descendant on another volume remains a separate observation.
    """
    identities = {}
    unknown = []
    for path in roots:
        try:
            identity = _identity_nofollow(path, directory=True)
            key = (identity['volume_serial'], identity['file_index'])
            previous = identities.get(key)
            if previous is None or _normalize_windows_path(path) < _normalize_windows_path(previous[0]):
                identities[key] = (path, identity)
        except (OSError, ValueError, TypeError, KeyError):
            unknown.append((path, None))
    ordered = sorted([*identities.values(), *unknown],
                     key=lambda item: (len(Path(item[0]).parts), _normalize_windows_path(item[0])))
    selected = []
    for path, identity in ordered:
        covered = False
        if identity is not None:
            for parent, parent_identity in selected:
                if parent_identity is None or parent_identity['volume_serial'] != identity['volume_serial']:
                    continue
                parent_name = _normalize_windows_path(parent)
                child_name = _normalize_windows_path(path)
                try:
                    if child_name != parent_name and os.path.commonpath([parent_name, child_name]) == parent_name:
                        relative = Path(path).relative_to(parent)
                        current = Path(parent)
                        for part in relative.parts:
                            current /= part
                            step = _identity_nofollow(current, directory=True)
                            if step['volume_serial'] != parent_identity['volume_serial']:
                                break
                        else:
                            covered = True
                            break
                except (OSError, ValueError, TypeError, KeyError):
                    pass
        if not covered:
            selected.append((path, identity))
    return [path for path, _ in selected]


def _winapi():
    import ctypes
    from ctypes import wintypes

    class FileInfo(ctypes.Structure):
        _fields_ = [('attributes', wintypes.DWORD),
                    ('creation', wintypes.FILETIME), ('access', wintypes.FILETIME),
                    ('write', wintypes.FILETIME), ('volume', wintypes.DWORD),
                    ('size_high', wintypes.DWORD), ('size_low', wintypes.DWORD),
                    ('links', wintypes.DWORD), ('index_high', wintypes.DWORD),
                    ('index_low', wintypes.DWORD)]

    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                                   ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD,
                                   wintypes.HANDLE]
    kernel.CreateFileW.restype = wintypes.HANDLE
    kernel.GetFileType.argtypes = [wintypes.HANDLE]
    kernel.GetFileType.restype = wintypes.DWORD
    kernel.GetFileInformationByHandle.argtypes = [wintypes.HANDLE, ctypes.POINTER(FileInfo)]
    kernel.GetFileInformationByHandle.restype = wintypes.BOOL
    kernel.GetFinalPathNameByHandleW.argtypes = [wintypes.HANDLE, wintypes.LPWSTR,
                                                wintypes.DWORD, wintypes.DWORD]
    kernel.GetFinalPathNameByHandleW.restype = wintypes.DWORD
    kernel.ReadFile.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD,
                                ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p]
    kernel.ReadFile.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    return ctypes, wintypes, kernel, FileInfo


def _normalize_windows_path(value):
    value = str(value)
    if value.startswith('\\\\?\\UNC\\'):
        value = '\\\\' + value[8:]
    elif value.startswith('\\\\?\\'):
        value = value[4:]
    return os.path.normcase(os.path.abspath(value))


def _read_nofollow(path):
    """Read only the exact regular, single-link snapshot; no reparse target."""
    if os.name != 'nt':
        raise OSError('windows_handle_required')
    ctypes, wintypes, kernel, file_info = _winapi()
    # OPEN_REPARSE_POINT + GENERIC_READ, share read/write/delete; no file is
    # read until type, reparse status, hardlinks and actual final path match.
    handle = kernel.CreateFileW(str(path), 0x80000000, 7, None, 3, 0x00200000, None)
    if handle == ctypes.c_void_p(-1).value:
        raise OSError('snapshot_open_failed')
    try:
        info = file_info()
        if not kernel.GetFileInformationByHandle(handle, ctypes.byref(info)):
            raise OSError('snapshot_info_failed')
        size = (info.size_high << 32) | info.size_low
        # FILE_TYPE_DISK, regular non-reparse non-directory, no hardlink alias.
        if (kernel.GetFileType(handle) != 1 or info.attributes & (0x400 | 0x10)
                or info.links != 1 or size > MAX_BYTES):
            raise ValueError('snapshot_type_or_size_invalid')
        final = ctypes.create_unicode_buffer(32768)
        count = kernel.GetFinalPathNameByHandleW(handle, final, len(final), 0)
        if not count or count >= len(final):
            raise OSError('snapshot_final_path_unknown')
        if _normalize_windows_path(final.value) != _normalize_windows_path(path):
            raise ValueError('snapshot_parent_redirected')
        data = ctypes.create_string_buffer(MAX_BYTES + 1)
        got = wintypes.DWORD()
        if not kernel.ReadFile(handle, data, MAX_BYTES + 1, ctypes.byref(got), None):
            raise OSError('snapshot_read_failed')
        if got.value > MAX_BYTES or got.value != size:
            raise ValueError('snapshot_changed_or_oversized')
        return data.raw[:got.value]
    finally:
        kernel.CloseHandle(handle)


def _identity_nofollow(path, *, directory):
    """Metadata-only handle identity; reject reparse points and redirects."""
    if os.name != 'nt':
        raise OSError('windows_handle_required')
    ctypes, _, kernel, file_info = _winapi()
    # FILE_READ_ATTRIBUTES; BACKUP_SEMANTICS permits opening directories.
    flags = 0x00200000 | (0x02000000 if directory else 0)
    handle = kernel.CreateFileW(str(path), 0x80, 7, None, 3, flags, None)
    if handle == ctypes.c_void_p(-1).value:
        raise OSError('identity_open_failed')
    try:
        info = file_info()
        if not kernel.GetFileInformationByHandle(handle, ctypes.byref(info)):
            raise OSError('identity_query_failed')
        if (kernel.GetFileType(handle) != 1 or info.attributes & 0x400
                or bool(info.attributes & 0x10) != directory
                or (not directory and info.links != 1)):
            raise ValueError('identity_type_or_links_invalid')
        final = ctypes.create_unicode_buffer(32768)
        count = kernel.GetFinalPathNameByHandleW(handle, final, len(final), 0)
        if not count or count >= len(final):
            raise OSError('identity_final_path_unknown')
        actual = _normalize_windows_path(final.value)
        if actual != _normalize_windows_path(path):
            raise ValueError('identity_parent_redirected')
        drive, _ = os.path.splitdrive(actual)
        if len(drive) != 2 or drive[1] != ':':
            raise ValueError('identity_not_local_drive')
        return {'canonical_path': actual, 'volume_serial': int(info.volume),
                'file_index': (int(info.index_high) << 32) | int(info.index_low)}
    finally:
        kernel.CloseHandle(handle)


def _hex_tag(value):
    return (isinstance(value, str) and len(value) == 64
            and all(c in '0123456789abcdef' for c in value))


def _local_path_tag(pair_tag, path):
    return hashlib.sha256(b'RZ-WINROOT-v1\0' + pair_tag.encode('ascii') + b'\0'
                          + _normalize_windows_path(path).encode('utf-8')).hexdigest()


def _registered_distro(distro_id):
    """Resolve only a registered distro selected by exact registry GUID."""
    if os.name != 'nt':
        raise OSError('windows_required')
    import winreg
    distro_id = '{' + str(uuid.UUID(distro_id)).lower() + '}'
    with winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                        r'Software\Microsoft\Windows\CurrentVersion\Lxss' + '\\' + distro_id) as key:
        base, _ = winreg.QueryValueEx(key, 'BasePath')
        name, _ = winreg.QueryValueEx(key, 'DistributionName')
    if (not isinstance(base, str) or not os.path.isabs(base)
            or not isinstance(name, str) or not name or len(name) > 256
            or any(ord(c) < 32 for c in name)):
        raise ValueError('distro_base_unknown')
    return name, Path(base) / 'ext4.vhdx'


def _registered_vhd(distro_id):
    _, path = _registered_distro(distro_id)
    return _identity_nofollow(path, directory=False)


_VERIFY_ORIGIN_SCRIPT = r'''
import hashlib, json, pathlib, sys
pair = sys.argv[1].encode('ascii')
machine = pathlib.Path('/etc/machine-id').read_bytes().strip()
boot = pathlib.Path('/proc/sys/kernel/random/boot_id').read_bytes().strip()
if not machine or not boot:
    raise ValueError('identity_unavailable')
digest = lambda data: hashlib.sha256(data).hexdigest()
print(json.dumps({'pair_tag': digest(b'RZ-PAIR-v1\0' + pair),
 'machine_tag': digest(b'RZ-MACHINE-v1\0' + pair + b'\0' + machine),
 'boot_tag': digest(b'RZ-BOOT-v1\0' + pair + b'\0' + boot)}))
'''


def _verify_distro_origin(distro_id, pair_id, linux):
    """Explicit registration only: compare tags from the selected live WSL.

    This can start the chosen distro. The caller must invoke it only during
    authorized setup, never from periodic observation or admission.
    """
    pair_id = str(uuid.UUID(pair_id)).lower()
    proof = linux.get('origin_proof') if isinstance(linux, dict) else None
    expected_pair = hashlib.sha256(b'RZ-PAIR-v1\0' + pair_id.encode('ascii')).hexdigest()
    if not isinstance(proof, dict) or proof.get('pair_tag') != expected_pair:
        raise ValueError('pair_id_mismatch')
    name, vhd_path = _registered_distro(distro_id)
    vhd_before = _identity_nofollow(vhd_path, directory=False)
    system_root = os.environ.get('SystemRoot')
    if not system_root or not os.path.isabs(system_root):
        raise OSError('system_root_unknown')
    executable = Path(system_root) / 'System32' / 'wsl.exe'
    result = subprocess.run([str(executable), '-d', name, '-u', 'root', '--exec',
                             'python3', '-I', '-S', '-c', _VERIFY_ORIGIN_SCRIPT, pair_id],
                            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, timeout=15, check=False,
                            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    if result.returncode != 0 or len(result.stdout) > 4096:
        raise OSError('selected_distro_identity_unavailable')
    actual = json.loads(result.stdout)
    if not isinstance(actual, dict) or any(actual.get(key) != proof.get(key) for key in
                                           ('pair_tag', 'machine_tag', 'boot_tag')):
        raise ValueError('selected_distro_identity_mismatch')
    if _registered_distro(distro_id) != (name, vhd_path):
        raise ValueError('selected_distro_registration_changed')
    if _identity_nofollow(vhd_path, directory=False) != vhd_before:
        raise ValueError('selected_distro_vhd_changed')
    return vhd_before


def _sealed_identity(pair_tag, identity):
    return {'path_tag': _local_path_tag(pair_tag, identity['canonical_path']),
            'volume_serial': identity['volume_serial'], 'file_index': identity['file_index']}


def prepare_pair_manifest(linux, distro_id, roots, *, now=None):
    """Prepare local pairing evidence; never starts WSL or reads a VHD body.

    The user/admin installation is responsible for ensuring the root-owned
    Linux pairing configuration and this Windows pairing selection correspond.
    The identifier is not a secret or an authentication mechanism.
    """
    _validated(linux, LINUX_SCHEMA, now=now)
    proof = linux.get('origin_proof')
    if (not isinstance(proof, dict)
            or any(not _hex_tag(proof.get(k)) for k in ('pair_tag', 'machine_tag', 'boot_tag'))
            or proof.get('single_linux_device') is not True):
        raise ValueError('linux_pair_proof_incomplete')
    distro_id = str(uuid.UUID(distro_id)).lower()
    tag = proof['pair_tag']
    vhd = _registered_vhd(distro_id)
    windows_roots = [_identity_nofollow(Path(path), directory=True) for path in roots]
    if not windows_roots:
        raise ValueError('windows_roots_empty')
    unique = {(item['volume_serial'], item['file_index']) for item in windows_roots}
    if len(unique) != len(windows_roots):
        raise ValueError('windows_roots_alias')
    for item in windows_roots:
        try:
            common = os.path.commonpath([vhd['canonical_path'], item['canonical_path']])
        except ValueError:
            # Windows commonpath raises for different drives, which are disjoint.
            continue
        if common == item['canonical_path']:
            raise ValueError('vhd_inside_windows_root')
    return {'schema': PAIRING_SCHEMA, 'pair_tag': tag, 'machine_tag': proof['machine_tag'],
            'distro_id': distro_id, 'vhd': _sealed_identity(tag, vhd),
            'windows_roots': sorted((_sealed_identity(tag, row) for row in windows_roots),
                                    key=lambda row: row['path_tag'])}


def pairing_matches(manifest, linux, roots, *, now=None):
    """Recheck configured pair and live handle identities; fail to unknown."""
    try:
        if not isinstance(manifest, dict) or manifest.get('schema') != PAIRING_SCHEMA:
            return False
        current = prepare_pair_manifest(linux, manifest['distro_id'], roots, now=now)
        return manifest == current
    except (KeyError, OSError, ValueError, TypeError, OverflowError, ImportError):
        return False


def _scan_roots_match_handles(result, roots):
    """Bind each scanned root's observed identity to its current Win32 handle."""
    try:
        if not isinstance(result, dict):
            return False
        rows = result.get('roots')
        if not isinstance(rows, list) or len(rows) != len(roots) or not roots:
            return False
        expected = {_normalize_windows_path(root): root for root in roots}
        if len(expected) != len(roots):
            return False
        observed = set()
        for row in rows:
            if (not isinstance(row, dict) or not isinstance(row.get('root'), str)
                    or row.get('identity_unchanged') is not True
                    or type(row.get('scan_device')) is not int
                    or type(row.get('scan_inode')) is not int):
                return False
            key = _normalize_windows_path(row['root'])
            if key not in expected or key in observed:
                return False
            identity = _identity_nofollow(expected[key], directory=True)
            if (identity['volume_serial'], identity['file_index']) != (row['scan_device'], row['scan_inode']):
                return False
            observed.add(key)
        return observed == set(expected)
    except (OSError, ValueError, KeyError, TypeError):
        return False


def _validated(value, schema, *, now=None, max_age=MAX_AGE):
    if (not isinstance(value, dict) or value.get('schema') != schema
            or value.get('decision') not in ('exceeded', 'unknown', 'within')
            or value.get('coverage') not in ('partial', 'complete_for_scoped_regular_files')
            or value.get('allocated_measure') != ALLOCATED_MEASURE
            or any(type(value.get(k)) is not int or value[k] < 0 for k in
                   ('allocated_bytes_lower_bound', 'logical_bytes_lower_bound'))
            or type(value.get('threshold_bytes')) is not int or value['threshold_bytes'] <= 0):
        raise ValueError('snapshot_fields_invalid')
    observed = datetime.fromisoformat(value['observed_at_utc'])
    if observed.tzinfo is None:
        raise ValueError('snapshot_time_without_timezone')
    age = (time.time() if now is None else now) - observed.timestamp()
    if age < -60 or age > max_age:
        raise ValueError('snapshot_stale_or_future')
    return value


def _read_snapshot(path, schema, *, now=None, max_age=MAX_AGE):
    return _validated(json.loads(_read_nofollow(path)), schema, now=now, max_age=max_age)


def read_windows_combined(path=None, *, now=None, max_age_seconds=MAX_AGE):
    """Startup-only bounded read; unavailable/stale must be treated as unknown."""
    if path is None:
        path = monitor_dir() / 'combined.json'
    value = _read_snapshot(path, COMBINED_SCHEMA, now=now, max_age=max_age_seconds)
    components = value.get('components')
    if not isinstance(components, dict):
        raise ValueError('snapshot_components_invalid')
    windows = dict(components.get('windows', {}), schema=WINDOWS_SCHEMA)
    linux = dict(components.get('linux', {}), schema=LINUX_SCHEMA)
    # Revalidate each component against the current time. A combined file can
    # be fresh while one of the samples it contains has since gone stale.
    # A paired assertion is valid only for its bounded observation time; the
    # scheduled scan checks live handles before writing it. A local public
    # snapshot is advisory, not an authorization or cryptographic attestation.
    disjoint = (value.get('aggregation') == 'disjoint_sum'
                and value.get('pairing_status') == 'verified_at_scan')
    result = combine(windows, linux, proven_disjoint=disjoint, now=now,
                     threshold=value['threshold_bytes'], max_age=max_age_seconds)
    result['observed_at_utc'] = value['observed_at_utc']
    result['pairing_status'] = ('verified_at_scan' if result['aggregation'] == 'disjoint_sum'
                                else 'unverified')
    return result


def combine(windows, linux, *, proven_disjoint=False, now=None, threshold=THRESHOLD,
            max_age=MAX_AGE):
    """Combine available observations without claiming uncertain overlap away."""
    if type(threshold) is not int or threshold <= 0:
        raise ValueError('threshold_invalid')
    now = time.time() if now is None else now
    parts = {}
    for label, value, schema in (('windows', windows, WINDOWS_SCHEMA),
                                 ('linux', linux, LINUX_SCHEMA)):
        try:
            _validated(value, schema, now=now, max_age=max_age)
            parts[label] = {key: value[key] for key in
                            ('observed_at_utc', 'coverage', 'decision', 'threshold_bytes',
                             'allocated_bytes_lower_bound', 'logical_bytes_lower_bound', 'allocated_measure')}
            proof = value.get('origin_proof') if label == 'linux' else None
            if (isinstance(proof, dict)
                    and all(_hex_tag(proof.get(k)) for k in ('pair_tag', 'machine_tag', 'boot_tag'))
                    and type(proof.get('single_linux_device')) is bool):
                parts[label]['origin_proof'] = {k: proof[k] for k in
                                                ('pair_tag', 'machine_tag', 'boot_tag', 'single_linux_device')}
        except (ValueError, KeyError, TypeError, OverflowError):
            parts[label] = {'status': 'unknown'}
    valid = [row for row in parts.values() if row.get('status') != 'unknown']
    allocated = [row['allocated_bytes_lower_bound'] for row in valid]
    logical = [row['logical_bytes_lower_bound'] for row in valid]
    # If cross-OS overlap cannot be excluded, the maximum is the only safe
    # aggregate lower bound; never count a WSL VHDX plus its interior blindly.
    use_sum = proven_disjoint and len(valid) == 2
    total_allocated = sum(allocated) if use_sum else max(allocated, default=0)
    total_logical = sum(logical) if use_sum else max(logical, default=0)
    partial = (len(valid) != 2 or not use_sum
               or any(row['coverage'] != 'complete_for_scoped_regular_files' for row in valid))
    decision = 'exceeded' if total_allocated > threshold else 'unknown' if partial else 'within'
    return {'schema': COMBINED_SCHEMA, 'observed_at_utc': datetime.fromtimestamp(now, timezone.utc).isoformat(),
            'decision': decision, 'coverage': 'partial' if partial else 'complete_for_scoped_regular_files',
            'threshold_bytes': threshold, 'allocated_bytes_lower_bound': total_allocated,
            'logical_bytes_lower_bound': total_logical, 'allocated_measure': ALLOCATED_MEASURE,
            'host_physical_bytes': 'unknown',
            'aggregation': 'disjoint_sum' if use_sum else 'overlap_unknown_max', 'components': parts}


def _atomic_write(path, value):
    path = Path(path)
    # Check each existing ancestor before creating its child: mkdir(parents)
    # first could traverse a reparse point and write outside the approved tree.
    current = Path(path.anchor)
    for part in path.parent.parts[1:]:
        current /= part
        try:
            current.mkdir(exist_ok=True)
        except FileExistsError:
            pass
        info = current.lstat()
        if (not stat.S_ISDIR(info.st_mode) or os.path.islink(current)
                or getattr(info, 'st_file_attributes', 0) & 0x400):
            raise ValueError('snapshot_parent_reparse')
    payload = (json.dumps(value, sort_keys=True) + '\n').encode('utf-8')
    if len(payload) > MAX_BYTES:
        raise ValueError('snapshot_oversized')
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.usage-', delete=False) as f:
        temp = Path(f.name)
        try:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        except BaseException:
            temp.unlink(missing_ok=True)
            raise
    try:
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


class _WindowsMutex:
    """Single observer across Windows sessions for the current APPDATA root."""
    def __init__(self, root, *, _fixture_namespace=None):
        import ctypes
        from ctypes import wintypes
        self.ctypes = ctypes
        self.kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        self.kernel.CreateMutexW.argtypes = [ctypes.c_void_p, wintypes.BOOL, wintypes.LPCWSTR]
        self.kernel.CreateMutexW.restype = wintypes.HANDLE
        self.kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        self.kernel.WaitForSingleObject.restype = wintypes.DWORD
        self.kernel.ReleaseMutex.argtypes = [wintypes.HANDLE]
        self.kernel.ReleaseMutex.restype = wintypes.BOOL
        self.kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        self.kernel.CloseHandle.restype = wintypes.BOOL
        user = os.environ.get('USERNAME')
        user_key = (os.environ.get('USERDOMAIN', '') + '\\' + user).casefold() if user else _normalize_windows_path(root)
        # A private fixture-only injection keeps tests from taking a live
        # user's real observer mutex. Production callers never set this value.
        if _fixture_namespace is not None:
            user_key = 'fixture:' + str(_fixture_namespace)
        token = hashlib.sha256(user_key.encode('utf-8')).hexdigest()[:32]
        self.handle = self.kernel.CreateMutexW(None, False, 'Global\\RoveZeroTotalUsage-' + token)
        if not self.handle:
            raise OSError('usage_mutex_unavailable')
        self.acquired = False

    def __enter__(self):
        wait = self.kernel.WaitForSingleObject(self.handle, 0)
        if wait in (0, 0x80):  # acquired or abandoned: observation is read-only
            self.acquired = True
        elif wait != 258:  # WAIT_TIMEOUT means another observer is running
            self.kernel.CloseHandle(self.handle)
            self.handle = None
            raise OSError('usage_mutex_wait_failed')
        return self.acquired

    def __exit__(self, *_):
        if self.acquired:
            self.kernel.ReleaseMutex(self.handle)
        if self.handle:
            self.kernel.CloseHandle(self.handle)


def register_pairing(distro_id, pair_id):
    """Explicit setup only; the installer verifies the selected Linux config.

    Pair tags are identifiers, not credentials or a security attestation.
    Failure leaves an existing pairing file unchanged.
    """
    if os.name != 'nt':
        raise OSError('windows_required')
    base = monitor_dir()
    root = Path(os.environ['APPDATA']) / 'RoveZero'
    with _WindowsMutex(root) as acquired:
        if not acquired:
            raise OSError('usage_observer_already_running')
        linux = _read_snapshot(base / 'linux.json', LINUX_SCHEMA)
        observed_vhd = _verify_distro_origin(distro_id, pair_id, linux)
        roots = discover_windows_roots()
        manifest = prepare_pair_manifest(linux, distro_id, roots)
        if manifest['vhd'] != _sealed_identity(linux['origin_proof']['pair_tag'], observed_vhd):
            raise ValueError('selected_distro_vhd_changed_before_registration')
        _atomic_write(base / 'pairing.json', manifest)
        return {'status': 'pairing_registered', 'schema': PAIRING_SCHEMA}


def scan_once(*, max_entries=2_000_000, max_seconds=120.0, threshold_bytes=THRESHOLD):
    """Run only from a scheduled task; publish per-OS and combined totals."""
    if os.name != 'nt':
        raise OSError('windows_required')
    base = monitor_dir()
    root = Path(os.environ['APPDATA']) / 'RoveZero'
    with _WindowsMutex(root) as acquired:
        if not acquired:
            return {'status': 'already_running'}
        spec = importlib.util.spec_from_file_location(
            'rovezero_total_usage', Path(__file__).with_name('rovezero_total_usage.py'))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        roots = discover_windows_roots()
        result = module.scan_total_usage(roots, threshold_bytes=threshold_bytes,
                                         max_entries=max_entries, max_seconds=max_seconds)
        windows = {key: result[key] for key in ('observed_at_utc', 'decision', 'coverage',
                   'threshold_bytes', 'allocated_bytes_lower_bound', 'logical_bytes_lower_bound')}
        windows.update(schema=WINDOWS_SCHEMA, scope='windows_standard_and_codex_roots_custom_unknown',
                       host_physical_bytes='unknown', consistency='best_effort_non_atomic',
                       allocated_measure=ALLOCATED_MEASURE)
        # Other package identities and user-selected locations may exist even
        # if both discovered roots were scanned without an error.
        windows['coverage'] = 'partial'
        if windows['decision'] == 'within':
            windows['decision'] = 'unknown'
        _atomic_write(base / 'windows.json', windows)
        try:
            linux = _read_snapshot(base / 'linux.json', LINUX_SCHEMA)
        except (ValueError, OSError, KeyError, TypeError, OverflowError):
            linux = None
        try:
            manifest = json.loads(_read_nofollow(base / 'pairing.json'))
            # Root discovery must still match after the scan; identity handles
            # are then rechecked by pairing_matches before allowing a sum.
            unchanged_roots = sorted(map(_normalize_windows_path, roots)) == sorted(
                map(_normalize_windows_path, discover_windows_roots()))
            paired = (unchanged_roots and _scan_roots_match_handles(result, roots)
                      and pairing_matches(manifest, linux, roots))
        except (OSError, ValueError, KeyError, TypeError, OverflowError):
            paired = False
        combined = combine(windows, linux, proven_disjoint=paired, threshold=threshold_bytes)
        combined['pairing_status'] = 'verified_at_scan' if paired else 'unverified'
        _atomic_write(base / 'combined.json', combined)
        return combined


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-seconds', type=float, default=120.0)
    parser.add_argument('--max-entries', type=int, default=2_000_000)
    parser.add_argument('--threshold-bytes', type=int, default=THRESHOLD)
    parser.add_argument('--register-pairing', metavar='DISTRO_GUID',
                        help='explicit setup: pair the existing Linux mirror with one registered WSL distro')
    parser.add_argument('--pair-id', metavar='UUID',
                        help='non-secret UUID from the matching root-owned Linux configuration')
    parser.add_argument('--register-monitor-dir', metavar='PATH',
                        help='explicit setup: pin an existing validated Windows monitor directory')
    args = parser.parse_args()
    if bool(args.register_pairing) != bool(args.pair_id):
        parser.error('--register-pairing and --pair-id must be provided together')
    if args.register_monitor_dir and args.register_pairing:
        parser.error('register the monitor directory and pairing in separate steps')
    try:
        if args.register_monitor_dir:
            result = register_monitor_dir(args.register_monitor_dir)
        elif args.register_pairing:
            result = register_pairing(args.register_pairing, args.pair_id)
        else:
            result = scan_once(max_entries=args.max_entries, max_seconds=args.max_seconds,
                               threshold_bytes=args.threshold_bytes)
        print(json.dumps(result, sort_keys=True))
        return 0
    except Exception as exc:
        # Leave the previous snapshot intact; readers must label it stale.
        print('rovezero_windows_usage_scan_error=' + type(exc).__name__, file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
