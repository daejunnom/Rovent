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
import subprocess
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
WINDOWS_REFRESH_TASK = 'RoveZero Total Storage Warning'
WINDOWS_TASK_RUNNER = Path('/mnt/c/Windows/System32/schtasks.exe')
WINDOWS_REFRESH_STATE = 'windows-refresh.state'
MIRROR_CONFIG = 'paired-mirror.json'
PUBLIC_SCAN_CAUSES = frozenset({'threshold_short_circuit', 'deadline', 'entry_limit',
    'privacy_excluded', 'other_mount_excluded', 'other_observation_limit',
    'unvisited_roots', 'custom_scope_unknown'})


def public_scan_summary(value):
    """Filter externally supplied summaries so warnings never expose paths."""
    if not isinstance(value, dict):
        return None
    keys = ('visited_roots', 'unvisited_roots', 'entries', 'unique_regular_files')
    if any(type(value.get(key)) is not int or value[key] < 0 for key in keys):
        return None
    causes = value.get('partial_causes')
    if not isinstance(causes, list):
        return None
    return dict({key: value[key] for key in keys}, partial_causes=sorted({
        cause for cause in causes if isinstance(cause, str) and cause in PUBLIC_SCAN_CAUSES}))


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
            # Retain when and where the old sample was taken, without
            # presenting its allocated bytes as a current measurement.
            return {'decision': 'unknown', 'reason': 'stale_or_future',
                    'age_seconds': int(age), 'observed_at_utc': value['observed_at_utc'],
                    'scope': value.get('scope', 'unknown'),
                    'coverage': value.get('coverage', 'unknown')}
        # A Windows scan can finish after the Linux timer has published its
        # snapshot. Re-read only the tiny, previously configured mirror here;
        # never traverse the storage trees on an engine startup path.
        config = _read_mirror_config(Path(path).parent, require_root_owner=require_root_owner)
        mirror = (config['directory'] if config is not None
                  and value.get('_mirror_revision') == config['revision'] else None)
        source = value
        if value.get('scope') == 'paired_observations_not_host_physical_total':
            source = value.get('_linux_observation')
            if not _valid_linux_observation(source, value, now=now, max_age_seconds=max_age_seconds):
                source = None
        if source is not None and mirror is not None:
            paired = paired_observation(mirror, source, now=now, max_age_seconds=max_age_seconds)
            if paired is not None:
                paired['windows_refresh'] = 'paired'
                return paired
        return value
    finally:
        if fd >= 0:
            os.close(fd)


def _valid_linux_observation(source, paired, *, now, max_age_seconds):
    """Recombine only a fresh raw Linux sample embedded in the paired total."""
    try:
        if (not isinstance(source, dict) or source.get('schema') != SCHEMA
                or source.get('scope') == 'paired_observations_not_host_physical_total'
                or source.get('threshold_bytes') != paired['threshold_bytes']
                or source.get('origin_proof') != paired.get('origin_proof')
                or source.get('allocated_measure') != paired['allocated_measure']
                or any(type(source.get(k)) is not int or source[k] < 0
                       for k in ('allocated_bytes_lower_bound', 'logical_bytes_lower_bound'))):
            return False
        observed = datetime.fromisoformat(source['observed_at_utc'])
        age = (time.time() if now is None else now) - observed.timestamp()
        return observed.tzinfo is not None and -60 <= age <= max_age_seconds
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def _current_total_usage():
    if sys.platform == 'win32':
        from rovezero_windows_usage_snapshot import read_windows_combined
        return read_windows_combined()
    if sys.platform.startswith('linux'):
        return read_snapshot()
    raise OSError('unsupported_platform')


def _confirmed_exceeded(value):
    allocated = value.get('allocated_bytes_lower_bound')
    threshold = value.get('threshold_bytes')
    return (value.get('decision') == 'exceeded'
            and type(allocated) is int and type(threshold) is int
            and threshold > 0 and allocated > max(threshold, DEFAULT_THRESHOLD_BYTES))


def confirmed_total_usage_exceeded():
    """Check one fresh advisory snapshot without taking an ownership gate.

    A partial observation can prove an exceedance when its allocated lower
    bound alone crosses the threshold.  Missing, stale, or inconsistent data
    are not evidence that maintenance is due.
    """
    try:
        return _confirmed_exceeded(_current_total_usage())
    except Exception:
        return False


def claim_total_usage_maintenance(*, state_path=None):
    """Claim at most one attempt per fresh exceeded observation and euid.

    The state file is only an advisory throttle, never cleanup authority. An
    unknown observation, busy lock, or unsafe/inaccessible state skips the
    attempt without affecting the completed run. Newly generated artifacts
    can remain until the next observer sample (normally 30 minutes).
    """
    fd = directory = None
    try:
        value = _current_total_usage()
        if value.get('decision') == 'unknown':
            raise ValueError('observation_unknown')
        if not _confirmed_exceeded(value):
            return False
        if not sys.platform.startswith('linux'):
            raise OSError('maintenance_platform_unavailable')
        import fcntl
        observed = value['observed_at_utc']
        source = value.get('source_observed_at_utc')
        identity = json.dumps({'observed_at_utc': observed, 'sources': source},
                              sort_keys=True, separators=(',', ':')).encode('utf-8')
        digest = hashlib.sha256(identity).hexdigest().encode('ascii')
        path = Path(state_path) if state_path is not None else Path('/tmp') / (
            'rovezero-total-maintenance-' + str(os.geteuid()) + '.state')
        if not path.is_absolute() or path.name in ('', '.', '..'):
            raise ValueError('invalid_state_path')
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
        fd = os.open(path.name, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK,
                     0o600, dir_fd=directory)
        info = os.fstat(fd)
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid()
                or info.st_nlink != 1 or info.st_mode & 0o077 or info.st_size > 64):
            raise ValueError('untrusted_state')
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        info = os.fstat(fd)
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid()
                or info.st_nlink != 1 or info.st_mode & 0o077 or info.st_size > 64):
            raise ValueError('untrusted_state')
        named = os.stat(path.name, dir_fd=directory, follow_symlinks=False)
        if (named.st_dev, named.st_ino) != (info.st_dev, info.st_ino):
            raise ValueError('state_identity_changed')
        previous = os.read(fd, 65)
        if previous == digest:
            return False
        if previous and (len(previous) != 64 or any(c not in b'0123456789abcdef' for c in previous)):
            # A partial write is uncertain. Only a demonstrably newer
            # observation may replace it; the previous attempt is never
            # retried against the same snapshot.
            stamps = [observed] + (list(source.values()) if isinstance(source, dict) else [])
            latest = max(datetime.fromisoformat(s).timestamp() for s in stamps)
            if latest <= info.st_mtime + 1:
                raise ValueError('uncertain_state')
        os.lseek(fd, 0, os.SEEK_SET)
        if os.write(fd, digest) != len(digest):
            raise OSError('short_state_write')
        os.ftruncate(fd, len(digest))
        return True
    except Exception as error:
        reason = ('lock_busy' if isinstance(error, BlockingIOError) else
                  str(error) if isinstance(error, ValueError) and str(error) in {
                      'observation_unknown', 'invalid_state_path', 'untrusted_state',
                      'state_identity_changed', 'uncertain_state'} else
                  'state_or_observation_unavailable')
        try:
            sys.stderr.write('rovezero_storage_maintenance_deferred=' + reason + '\n')
        except BaseException:
            pass
        return False
    finally:
        for handle in (fd, directory):
            if handle is not None:
                try:
                    os.close(handle)
                except OSError:
                    pass


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
                  'scope', 'windows_source_scope', 'windows_refresh', 'age_seconds') if key in value}
        summary = public_scan_summary(value.get('scan_summary'))
        if summary is not None:
            public['scan_summary'] = summary
        summaries = value.get('source_scan_summary')
        if isinstance(summaries, dict):
            public['source_scan_summary'] = {name: row for name in ('linux', 'windows')
                if (row := public_scan_summary(summaries.get(name))) is not None}
        public.setdefault('observed_at_utc', 'unknown')
        public.setdefault('scope', 'unknown')
        public.setdefault('coverage', 'unknown')
        if 'age_seconds' not in public:
            try:
                observed = datetime.fromisoformat(public['observed_at_utc'])
                if observed.tzinfo is None:
                    raise ValueError('snapshot_time_without_timezone')
                public['age_seconds'] = int(time.time() - observed.timestamp())
            except (TypeError, ValueError, OverflowError, OSError):
                public['age_seconds'] = 'unknown'
        if isinstance(value.get('components'), dict):
            public['source_observed_at_utc'] = {
                name: row.get('observed_at_utc', 'unknown')
                for name, row in value['components'].items()
                if name in ('linux', 'windows') and isinstance(row, dict)
            }
            public['aggregation'] = value.get('aggregation', 'unknown')
            public['source_scan_summary'] = {name: row for name, component in value['components'].items()
                if name in ('linux', 'windows') and isinstance(component, dict)
                if (row := public_scan_summary(component.get('scan_summary'))) is not None}
        signature = (public.get('decision'), public.get('reason'), public.get('coverage'),
                     public.get('observed_at_utc'), public.get('scope'),
                     public.get('windows_refresh'), public.get('allocated_bytes_lower_bound'),
                     public.get('logical_bytes_lower_bound'),
                     json.dumps(public.get('source_observed_at_utc'), sort_keys=True))
        if signature == _last_warning:
            return
        sys.stderr.write('rovezero_storage_total_warning: '
                         + json.dumps(public, sort_keys=True) + '\n')
        _last_warning = signature
    except Exception:
        # Closed stderr or any diagnostic error cannot change an operation.
        pass


def default_roots_and_discovery(home_parent=Path('/home'), root_cache_parent=Path('/root/.cache'),
                                var_tmp_parent=Path('/var/tmp'), tmp_parent=Path('/tmp')):
    """Discover default cache roots and their direct legacy sibling parents."""
    roots = []
    parents = [var_tmp_parent, tmp_parent]
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


def _publish(path, value, *, mode=0o644):
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
                     mode, dir_fd=directory)
        try:
            with os.fdopen(fd, 'wb') as f:
                os.fchmod(f.fileno(), mode)
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


def _read_mirror_config(parent, *, require_root_owner=True):
    """Read root-private local configuration without exposing paths in totals."""
    fd = None
    try:
        fd = os.open(Path(parent) / MIRROR_CONFIG, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK)
        info = os.fstat(fd)
        if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_mode & 0o077
                or info.st_size > MAX_BYTES or (require_root_owner and info.st_uid != 0)):
            return None
        with os.fdopen(fd, 'rb') as stream:
            fd = None
            raw = stream.read(MAX_BYTES + 1)
        if len(raw) > MAX_BYTES:
            return None
        value = json.loads(raw)
        if (not isinstance(value, dict) or set(value) != {'schema', 'directory', 'revision'}
                or value['schema'] != 'rovezero.paired-mirror-config.v1'):
            return None
        directory = value['directory']
        revision = value['revision']
        if (not isinstance(directory, str) or len(directory) > 4096 or not Path(directory).is_absolute()
                or not isinstance(revision, str) or len(revision) != 32
                or any(ch not in '0123456789abcdef' for ch in revision)):
            return None
        return value
    except (OSError, ValueError, TypeError):
        return None
    finally:
        if fd is not None:
            os.close(fd)


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
            if (not path.is_absolute() or not (stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode))
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


def paired_observation(mirror_dir, linux, *, now=None, max_age_seconds=MAX_AGE_SECONDS):
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
            if observed.tzinfo is None or not -60 <= current - observed.timestamp() <= max_age_seconds:
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
        result['source_scan_summary'] = {name: summary for name, component in
            (('linux', linux), ('windows', right))
            if (summary := public_scan_summary(component.get('scan_summary'))) is not None}
        # The older source is the age bound for this paired observation.
        result['observed_at_utc'] = min(
            (linux['observed_at_utc'], right['observed_at_utc']),
            key=lambda value: datetime.fromisoformat(value).timestamp())
        return result
    except (OSError, ValueError, TypeError, KeyError, OverflowError):
        return None


def request_windows_refresh_once(state_parent, linux, *, runner=WINDOWS_TASK_RUNNER):
    """Request the existing Windows observer once per paired Linux boot.

    This is advisory and runs in the low-priority timer, never in a product
    admission path. Persist the attempt before starting the short-lived task
    request so interruption cannot create an unbounded stream of processes.
    """
    import fcntl
    proof = linux.get('origin_proof')
    if not isinstance(proof, dict) or not proof.get('single_linux_device'):
        return 'pair_identity_unknown'
    tags = [proof.get('pair_tag'), proof.get('boot_tag')]
    if any(not isinstance(tag, str) or len(tag) != 64
           or any(ch not in '0123456789abcdef' for ch in tag) for tag in tags):
        return 'pair_identity_unknown'
    directory = fd = None
    try:
        directory = os.open(state_parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
        fd = os.open(WINDOWS_REFRESH_STATE, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC,
                     0o600, dir_fd=directory)
        info = os.fstat(fd)
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid()
                or info.st_nlink != 1 or info.st_mode & 0o077 or info.st_size > 256):
            return 'state_unavailable'
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return 'request_in_progress'
        named = os.stat(WINDOWS_REFRESH_STATE, dir_fd=directory, follow_symlinks=False)
        if (named.st_dev, named.st_ino) != (info.st_dev, info.st_ino):
            return 'state_unavailable'
        identity = (tags[0] + ':' + tags[1]).encode('ascii')
        previous = os.read(fd, 257)
        if previous == identity:
            # The process might have stopped after recording the attempt but
            # before Windows received it. Do not claim delivery occurred.
            return 'attempt_already_recorded'
        if previous and (len(previous) != 129 or previous[64:65] != b':'
                         or any(ch not in b'0123456789abcdef' for ch in previous[:64] + previous[65:])):
            # A partial state write is uncertain. Never start a second task
            # from an unreadable previous attempt.
            return 'state_unavailable'
        os.lseek(fd, 0, os.SEEK_SET)
        if os.write(fd, identity) != len(identity):
            return 'state_unavailable'
        os.ftruncate(fd, len(identity))
        os.fsync(fd)
        try:
            result = subprocess.run([str(runner), '/Run', '/TN', WINDOWS_REFRESH_TASK],
                                    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL, timeout=5, check=False)
            return 'requested' if result.returncode == 0 else 'request_failed'
        except (OSError, subprocess.SubprocessError):
            return 'request_failed'
    except (OSError, ValueError):
        return 'state_unavailable'
    finally:
        for handle in (fd, directory):
            if handle is not None:
                os.close(handle)


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
                              stop_when_exceeded=False)
    # Unknown XDG cache roots and nonstandard user-home locations prevent an
    # exhaustive claim. A proven threshold exceedance remains meaningful.
    value = {key: result[key] for key in ('observed_at_utc', 'decision',
             'threshold_bytes', 'allocated_bytes_lower_bound', 'logical_bytes_lower_bound',
             'allocated_measure')}
    value.update(schema=SCHEMA, coverage='partial',
                 scope='default_rovezero_caches_and_rovezero_rovent_tmp_siblings_xdg_exceptions_unknown',
                 consistency='best_effort_non_atomic', host_physical_bytes='unknown',
                 scan_summary=module.public_scan_summary(result))
    value['scan_summary']['partial_causes'].append('custom_scope_unknown')
    value['origin_proof'] = origin_proof(result, pair_id) if pair_id else None
    if value['decision'] == 'within':
        value['decision'] = 'unknown'
    path = Path(path)
    path.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
    local_value = value
    paired_value = None
    if mirror_dir is not None:
        # Configured only by the root-owned systemd environment. Publish totals
        # only, and never follow a redirected mirror directory. An unavailable
        # mirror does not invalidate the local observation.
        try:
            mirror = Path(mirror_dir)
            if not mirror.is_absolute() or mirror.resolve(strict=True) != mirror or not mirror.is_dir():
                raise ValueError('invalid_mirror_directory')
            _publish(mirror / 'linux.json', value)
            revision = uuid.uuid4().hex
            _publish(path.parent / MIRROR_CONFIG,
                     {'schema': 'rovezero.paired-mirror-config.v1', 'directory': str(mirror),
                      'revision': revision},
                     mode=0o600)
            paired = paired_observation(mirror, value)
            # Preserve the raw Linux sample separately, so an updated Windows
            # sample can be combined without counting Windows twice. Users
            # unable to read the root-private mirror configuration can still
            # consume the already-paired public total.
            local_value = dict(paired or value)
            local_value['_mirror_revision'] = revision
            if paired:
                local_value['_linux_observation'] = value
            local_value['windows_refresh'] = ('not_needed' if paired else
                request_windows_refresh_once(path.parent, value))
            if paired:
                paired_value = dict(paired, windows_refresh='paired')
        except Exception:
            print('rovezero_total_usage_mirror=unavailable', file=sys.stderr)
    _publish(path, local_value)
    return paired_value or local_value


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
