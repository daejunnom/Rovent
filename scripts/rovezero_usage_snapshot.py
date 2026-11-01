#!/usr/bin/env python3
"""Low-frequency total-storage snapshot and non-blocking startup warning.

The snapshot contains totals only; it grants no authority over storage owners.
Use a separately supervised privileged timer for scanning, never an admission
path or a catalog lock. A missed or partial observation remains unknown.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import time
import uuid

SNAPSHOT = Path('/var/lib/rovezero/total-usage/snapshot.json')
SCHEMA = 'rovezero.total-usage-public-snapshot.v1'
MAX_BYTES = 8192
MAX_AGE_SECONDS = 3600
DEFAULT_THRESHOLD_BYTES = 18 * 1024 ** 3
_last_check = float('-inf')
_last_warning = None
CHECK_INTERVAL_SECONDS = 60


def read_snapshot(path=SNAPSHOT, *, now=None, max_age_seconds=MAX_AGE_SECONDS,
                  require_root_owner=True):
    """Read at most one small public snapshot; never follow a final symlink."""
    if not hasattr(os, 'O_NOFOLLOW'):
        raise OSError('snapshot_nofollow_unavailable')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK)
    try:
        info = os.fstat(fd)
        if (not stat.S_ISREG(info.st_mode) or info.st_size > MAX_BYTES
                or info.st_mode & 0o022
                or (require_root_owner and info.st_uid != 0)):
            raise ValueError('untrusted_snapshot')
        with os.fdopen(fd, 'rb') as stream:
            fd = -1
            raw = stream.read(MAX_BYTES + 1)
        if len(raw) > MAX_BYTES:
            raise ValueError('oversized_snapshot')
        value = json.loads(raw)
        if (not isinstance(value, dict) or value.get('schema') != SCHEMA
                or value.get('decision') not in ('exceeded', 'unknown', 'within')
                or value.get('coverage') not in ('partial', 'complete_for_scoped_regular_files')
                or value.get('allocated_measure') != 'sum_of_observed_unique_file_reported_allocations'
                or type(value.get('allocated_bytes_lower_bound')) is not int
                or value['allocated_bytes_lower_bound'] < 0
                or type(value.get('logical_bytes_lower_bound')) is not int
                or value['logical_bytes_lower_bound'] < 0
                or type(value.get('threshold_bytes')) is not int
                or value['threshold_bytes'] <= 0):
            raise ValueError('invalid_snapshot')
        if (value['decision'] == 'exceeded'
                and value['allocated_bytes_lower_bound'] <= value['threshold_bytes']):
            raise ValueError('inconsistent_snapshot_decision')
        observed = datetime.fromisoformat(value['observed_at_utc'])
        if observed.tzinfo is None:
            raise ValueError('snapshot_time_without_timezone')
        age = (time.time() if now is None else now) - observed.timestamp()
        if age < -60 or age > max_age_seconds:
            return {'decision': 'unknown', 'reason': 'stale_or_future', 'age_seconds': int(age)}
        return value
    finally:
        if fd >= 0:
            os.close(fd)


def confirmed_total_usage_exceeded():
    """Check one fresh advisory snapshot without taking an ownership gate.

    A partial observation can prove an exceedance when its allocated lower
    bound alone crosses the threshold.  Missing, stale, or inconsistent data
    are not evidence that maintenance is due.
    """
    try:
        if sys.platform == 'win32':
            from rovezero_windows_usage_snapshot import read_windows_combined
            value = read_windows_combined()
        elif sys.platform.startswith('linux'):
            value = read_snapshot()
        else:
            return False
        allocated = value.get('allocated_bytes_lower_bound')
        threshold = value.get('threshold_bytes')
        return (value.get('decision') == 'exceeded'
                and type(allocated) is int and type(threshold) is int
                and threshold > 0 and allocated > max(threshold, DEFAULT_THRESHOLD_BYTES))
    except Exception:
        return False


def warn_cached_total_usage():
    """Throttled startup hook; bounded cached reads, failure is advisory."""
    global _last_check, _last_warning
    if not (sys.platform.startswith('linux') or sys.platform == 'win32'):
        return
    now = time.monotonic()
    if now - _last_check < CHECK_INTERVAL_SECONDS:
        return
    _last_check = now
    try:
        if sys.platform == 'win32':
            try:
                from rovezero_windows_usage_snapshot import (
                    monitor_dir, read_windows_combined, windows_monitor_installed)
                # A monitor is opt-in at installation. Avoid unrelated warnings
                # on deployments that have never had a configured monitor.
                # The separate registry marker survives loss of snapshot files.
                if not windows_monitor_installed() and not monitor_dir().is_dir():
                    return
                value = read_windows_combined()
            except Exception:
                value = {'decision': 'unknown', 'reason': 'snapshot_unavailable'}
        else:
            try:
                value = read_snapshot()
            except Exception:
                value = {'decision': 'unknown', 'reason': 'snapshot_unavailable'}
        if value.get('decision') == 'within' and value.get('coverage') != 'partial':
            _last_warning = None
            return
        # The payload contains no path, filename, identity or file contents.
        public = {key: value[key] for key in ('decision', 'reason', 'coverage',
                  'allocated_bytes_lower_bound', 'logical_bytes_lower_bound',
                  'threshold_bytes', 'observed_at_utc', 'source_observed_at_utc',
                  'scope', 'windows_source_scope', 'age_seconds') if key in value}
        if isinstance(value.get('components'), dict):
            public['source_observed_at_utc'] = {
                name: row.get('observed_at_utc', 'unknown')
                for name, row in value['components'].items()
                if name in ('linux', 'windows') and isinstance(row, dict)
            }
            public['aggregation'] = value.get('aggregation', 'unknown')
        signature = (public.get('decision'), public.get('reason'), public.get('coverage'),
                     public.get('observed_at_utc'))
        if signature == _last_warning:
            return
        sys.stderr.write('rovezero_storage_total_warning: '
                         + json.dumps(public, sort_keys=True) + '\n')
        _last_warning = signature
    except Exception:
        # Closed stderr or any diagnostic error cannot change an operation.
        pass


def default_roots_and_discovery(home_parent=Path('/home'), root_cache_parent=Path('/root/.cache'),
                                var_tmp_parent=Path('/var/tmp')):
    """Discover default cache roots and their direct legacy sibling parents."""
    roots = []
    parents = [var_tmp_parent]
    root_cache = root_cache_parent / 'rovezero'
    if root_cache_parent.is_dir() and not root_cache_parent.is_symlink():
        parents.append(root_cache_parent)
    if root_cache.is_dir() and not root_cache.is_symlink():
        roots.append(root_cache)
    try:
        with os.scandir(home_parent) as users:
            for entry in users:
                if entry.is_dir(follow_symlinks=False):
                    cache_parent = Path(entry.path) / '.cache'
                    if cache_parent.is_dir() and not cache_parent.is_symlink():
                        parents.append(cache_parent)
                    candidate = cache_parent / 'rovezero'
                    if candidate.is_dir() and not candidate.is_symlink():
                        roots.append(candidate)
    except OSError:
        # Caller always marks coverage partial because undiscovered users and
        # nondefault XDG cache roots cannot be excluded by this observation.
        pass
    return roots, parents


def _publish(path, value):
    path = Path(path)
    payload = (json.dumps(value, sort_keys=True) + '\n').encode('utf-8')
    parent = path.parent
    if not parent.is_absolute() or parent.resolve(strict=True) != parent:
        raise ValueError('snapshot_parent_redirected')
    expected = parent.stat(follow_symlinks=False)
    directory = os.open(parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    temp_name = '.snapshot-' + uuid.uuid4().hex
    try:
        actual = os.fstat(directory)
        if (not stat.S_ISDIR(actual.st_mode)
                or (actual.st_dev, actual.st_ino) != (expected.st_dev, expected.st_ino)):
            raise ValueError('snapshot_parent_changed')
        fd = os.open(temp_name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                     0o644, dir_fd=directory)
        try:
            with os.fdopen(fd, 'wb') as f:
                os.fchmod(f.fileno(), 0o644)
                f.write(payload)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temp_name, path.name, src_dir_fd=directory, dst_dir_fd=directory)
        finally:
            try:
                os.unlink(temp_name, dir_fd=directory)
            except FileNotFoundError:
                pass
    finally:
        os.close(directory)


def origin_proof(result, pair_id):
    """Seal this WSL observation to an installation pair without exposing IDs.

    The pair is an administrative scope association, not an authentication
    secret. Fail closed when a scanned root is no longer on the root device or
    the operating-system identifiers are unavailable.
    """
    try:
        pair = str(uuid.UUID(pair_id)).encode('ascii')
        machine = Path('/etc/machine-id').read_text(encoding='ascii').strip().lower()
        boot = str(uuid.UUID(Path('/proc/sys/kernel/random/boot_id').read_text(encoding='ascii').strip()))
        if len(machine) != 32 or any(ch not in '0123456789abcdef' for ch in machine):
            return None
        roots = result['roots']
        if not roots:
            return None
        device = os.stat('/').st_dev
        for item in roots:
            path = Path(item['root'])
            info = path.lstat()
            if (not path.is_absolute() or not stat.S_ISDIR(info.st_mode)
                    or stat.S_ISLNK(info.st_mode) or info.st_dev != device
                    or item.get('identity_unchanged') is not True
                    or item.get('scan_device') != info.st_dev
                    or item.get('scan_inode') != info.st_ino):
                return None
        def seal(domain, value):
            return hashlib.sha256(domain + pair + b'\0' + value.encode('ascii')).hexdigest()
        return {
            'pair_tag': hashlib.sha256(b'RZ-PAIR-v1\0' + pair).hexdigest(),
            'machine_tag': seal(b'RZ-MACHINE-v1\0', machine),
            'boot_tag': seal(b'RZ-BOOT-v1\0', boot),
            'single_linux_device': True,
        }
    except (KeyError, TypeError, ValueError, OSError, UnicodeError):
        return None


def paired_observation(mirror_dir, linux, *, now=None):
    """Read a paired Windows snapshot only as a bounded advisory observation.

    The Windows observer has already checked the configured VHD and root
    identities. Its two source timestamps are rechecked here; the result is not
    a simultaneous host-physical measurement or an ownership authorization.
    """
    try:
        directory = os.open(mirror_dir, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
        try:
            fd = os.open('combined.json', os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK,
                         dir_fd=directory)
            try:
                info = os.fstat(fd)
                if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size > MAX_BYTES:
                    return None
                with os.fdopen(fd, 'rb') as stream:
                    fd = -1
                    raw = stream.read(MAX_BYTES + 1)
            finally:
                if fd >= 0:
                    os.close(fd)
        finally:
            os.close(directory)
        if len(raw) > MAX_BYTES:
            return None
        data = json.loads(raw)
        if (data.get('schema') != 'rovezero.total-usage-combined-snapshot.v1'
                or data.get('aggregation') != 'disjoint_sum'
                or data.get('pairing_status') != 'verified_at_scan'
                or data.get('allocated_measure') != 'sum_of_observed_unique_file_reported_allocations'):
            return None
        components = data['components']
        left, right = components['linux'], components['windows']
        proof, own = left['origin_proof'], linux['origin_proof']
        if (not isinstance(own, dict) or not isinstance(proof, dict)
                or not own.get('single_linux_device') or not proof.get('single_linux_device')
                or any(proof.get(k) != own.get(k) for k in ('pair_tag', 'machine_tag', 'boot_tag'))):
            return None
        current = time.time() if now is None else now
        for component in (left, right):
            observed = datetime.fromisoformat(component['observed_at_utc'])
            if observed.tzinfo is None or not -60 <= current - observed.timestamp() <= MAX_AGE_SECONDS:
                return None
            for key in ('allocated_bytes_lower_bound', 'logical_bytes_lower_bound'):
                if type(component[key]) is not int or component[key] < 0:
                    return None
            if component.get('allocated_measure') != 'sum_of_observed_unique_file_reported_allocations':
                return None
        # Use the current Linux scan, not the older Linux copy that Windows
        # consumed when it generated combined.json. The Windows component is
        # accepted only within its own explicit age bound.
        allocated = linux['allocated_bytes_lower_bound'] + right['allocated_bytes_lower_bound']
        logical = linux['logical_bytes_lower_bound'] + right['logical_bytes_lower_bound']
        result = dict(linux, allocated_bytes_lower_bound=allocated,
                      logical_bytes_lower_bound=logical, coverage='partial',
                      decision='exceeded' if allocated > linux['threshold_bytes'] else 'unknown',
                      scope='paired_observations_not_host_physical_total',
                      source_observed_at_utc={'linux': linux['observed_at_utc'],
                                              'windows': right['observed_at_utc']})
        # The older source is the age bound for this paired observation.
        result['observed_at_utc'] = min(
            (linux['observed_at_utc'], right['observed_at_utc']),
            key=lambda value: datetime.fromisoformat(value).timestamp())
        return result
    except (OSError, ValueError, TypeError, KeyError, OverflowError):
        return None


def create_snapshot(path=SNAPSHOT, *, extra_roots=(), max_entries=2_000_000,
                    max_seconds=120.0, mirror_dir=None, pair_id=None,
                    threshold_bytes=DEFAULT_THRESHOLD_BYTES):
    """Run only from the timer; publish total-only JSON by atomic replacement."""
    # Isolated Python (-I) omits the script directory from sys.path. Load the
    # adjacent installed module explicitly, without importing ambient modules.
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'rovezero_total_usage', Path(__file__).with_name('rovezero_total_usage.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    scan_total_usage = module.scan_total_usage
    roots, parents = default_roots_and_discovery()
    result = scan_total_usage(roots + list(extra_roots),
                              discover_parents=parents,
                              threshold_bytes=threshold_bytes,
                              max_entries=max_entries, max_seconds=max_seconds,
                              stop_when_exceeded=True)
    # Unknown XDG cache roots and nonstandard user-home locations prevent an
    # exhaustive claim. A proven threshold exceedance remains meaningful.
    value = {key: result[key] for key in ('observed_at_utc', 'decision',
             'threshold_bytes', 'allocated_bytes_lower_bound', 'logical_bytes_lower_bound',
             'allocated_measure')}
    value.update(schema=SCHEMA, coverage='partial',
                 scope='default_rovezero_caches_siblings_and_var_tmp_xdg_exceptions_unknown',
                 consistency='best_effort_non_atomic', host_physical_bytes='unknown')
    value['origin_proof'] = origin_proof(result, pair_id) if pair_id else None
    if value['decision'] == 'within':
        value['decision'] = 'unknown'
    path = Path(path)
    path.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
    local_value = value
    if mirror_dir is not None:
        # Configured only by the root-owned systemd environment. Publish totals
        # only, and never follow a redirected mirror directory. An unavailable
        # mirror does not invalidate the local observation.
        try:
            mirror = Path(mirror_dir)
            if not mirror.is_absolute() or mirror.resolve(strict=True) != mirror or not mirror.is_dir():
                raise ValueError('invalid_mirror_directory')
            _publish(mirror / 'linux.json', value)
            local_value = paired_observation(mirror, value) or value
        except Exception:
            print('rovezero_total_usage_mirror=unavailable', file=sys.stderr)
    _publish(path, local_value)
    return local_value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--extra-root', action='append', default=[])
    parser.add_argument('--max-seconds', type=float, default=120.0)
    parser.add_argument('--max-entries', type=int, default=2_000_000)
    parser.add_argument('--threshold-bytes', type=int, default=DEFAULT_THRESHOLD_BYTES)
    args = parser.parse_args()
    if sys.platform != 'linux' or os.geteuid() != 0:
        parser.error('system timer scan requires Linux and root')
    try:
        value = create_snapshot(extra_roots=args.extra_root, max_entries=args.max_entries,
                                max_seconds=args.max_seconds,
                                mirror_dir=os.environ.get('RZ_USAGE_MIRROR_DIR') or None,
                                pair_id=os.environ.get('RZ_USAGE_PAIR_ID') or None,
                                threshold_bytes=args.threshold_bytes)
        print(json.dumps(value, sort_keys=True))
        return 0
    except Exception as exc:
        # Leave prior snapshot untouched; readers flag it stale or unavailable.
        print('rovezero_total_usage_scan_error=' + type(exc).__name__, file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
