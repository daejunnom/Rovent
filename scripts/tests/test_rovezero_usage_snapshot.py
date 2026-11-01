"""Focused checks for bounded advisory snapshot consumption."""
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
import types

import rovezero_usage_snapshot as usage


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'snapshot.json'
        self.value = {
            'schema': usage.SCHEMA,
            'observed_at_utc': datetime.fromtimestamp(1000, timezone.utc).isoformat(),
            'decision': 'exceeded', 'coverage': 'partial',
            'allocated_measure': 'sum_of_observed_unique_file_reported_allocations',
            'threshold_bytes': 18, 'allocated_bytes_lower_bound': 19,
            'logical_bytes_lower_bound': 20,
        }

    def write(self, value):
        self.path.write_text(json.dumps(value), encoding='utf-8')
        self.path.chmod(0o644)

    def test_only_fresh_proven_allocated_exceedance_enables_maintenance(self):
        with mock.patch.object(usage.sys, 'platform', 'linux'), \
             mock.patch.object(usage, 'DEFAULT_THRESHOLD_BYTES', 18):
            for value, expected in (
                (dict(self.value), True),
                (dict(self.value, decision='unknown'), False),
                (dict(self.value, allocated_bytes_lower_bound=18), False),
                ({'decision': 'unknown', 'reason': 'stale_or_future'}, False),
            ):
                with self.subTest(value=value), mock.patch.object(usage, 'read_snapshot', return_value=value):
                    self.assertEqual(usage.confirmed_total_usage_exceeded(), expected)
            with mock.patch.object(usage, 'read_snapshot', side_effect=OSError):
                self.assertFalse(usage.confirmed_total_usage_exceeded())

    def test_windows_confirmed_exceedance_uses_combined_observation(self):
        module = types.SimpleNamespace(read_windows_combined=mock.Mock(return_value=dict(self.value)))
        with mock.patch.dict(usage.sys.modules, {'rovezero_windows_usage_snapshot': module}), \
             mock.patch.object(usage.sys, 'platform', 'win32'), \
             mock.patch.object(usage, 'DEFAULT_THRESHOLD_BYTES', 18):
            self.assertTrue(usage.confirmed_total_usage_exceeded())
        module.read_windows_combined.assert_called_once_with()

    @unittest.skipUnless(sys.platform.startswith('linux'), 'Linux advisory flock')
    def test_claim_once_per_snapshot_across_concurrent_callers(self):
        state = self.path.with_name('maintenance.state')
        with mock.patch.object(usage, '_current_total_usage', return_value=dict(self.value)), \
             mock.patch.object(usage, 'DEFAULT_THRESHOLD_BYTES', 18):
            with ThreadPoolExecutor(max_workers=8) as pool:
                results = list(pool.map(lambda _: usage.claim_total_usage_maintenance(state_path=state), range(8)))
            self.assertEqual(results.count(True), 1)
            self.assertFalse(usage.claim_total_usage_maintenance(state_path=state))
        next_value = dict(self.value, observed_at_utc='1970-01-01T00:17:00+00:00')
        with mock.patch.object(usage, '_current_total_usage', return_value=next_value), \
             mock.patch.object(usage, 'DEFAULT_THRESHOLD_BYTES', 18):
            self.assertTrue(usage.claim_total_usage_maintenance(state_path=state))

    @unittest.skipUnless(sys.platform.startswith('linux'), 'Linux advisory flock')
    def test_claim_rejects_symlink_and_unknown_observation(self):
        victim = self.path.with_name('victim')
        victim.write_text('unchanged', encoding='utf-8')
        state = self.path.with_name('maintenance.state')
        state.symlink_to(victim)
        sink = io.StringIO()
        with mock.patch.object(usage, '_current_total_usage', return_value=dict(self.value)), \
             mock.patch.object(usage, 'DEFAULT_THRESHOLD_BYTES', 18), \
             mock.patch.object(usage.sys, 'stderr', sink):
            self.assertFalse(usage.claim_total_usage_maintenance(state_path=state))
        self.assertEqual(victim.read_text(encoding='utf-8'), 'unchanged')
        self.assertIn('rovezero_storage_maintenance_deferred=', sink.getvalue())
        with mock.patch.object(usage, '_current_total_usage', return_value={'decision': 'unknown'}), \
             mock.patch.object(usage.sys, 'stderr', sink):
            self.assertFalse(usage.claim_total_usage_maintenance(state_path=state))
        self.assertIn('observation_unknown', sink.getvalue())

    @unittest.skipUnless(hasattr(__import__('os'), 'O_NOFOLLOW'), 'Linux nofollow')
    def test_complete_snapshot_and_stale_snapshot_are_distinct(self):
        self.write(self.value)
        self.assertEqual(usage.read_snapshot(self.path, now=1001,
                                             require_root_owner=False)['decision'], 'exceeded')
        stale = usage.read_snapshot(self.path, now=5000, require_root_owner=False)
        self.assertEqual(stale['decision'], 'unknown')
        self.assertEqual(stale['reason'], 'stale_or_future')
        self.assertEqual(stale['observed_at_utc'], self.value['observed_at_utc'])
        self.assertEqual(stale['coverage'], self.value['coverage'])
        self.assertNotIn('allocated_bytes_lower_bound', stale)

    @unittest.skipUnless(hasattr(__import__('os'), 'O_NOFOLLOW'), 'Linux nofollow')
    def test_refuses_symlink_and_untrusted_or_invalid_snapshot(self):
        self.write(self.value)
        link = self.path.with_name('link')
        link.symlink_to(self.path)
        with self.assertRaises(OSError):
            usage.read_snapshot(link, now=1001, require_root_owner=False)
        self.path.chmod(0o666)
        with self.assertRaises(ValueError):
            usage.read_snapshot(self.path, now=1001, require_root_owner=False)
        self.write(dict(self.value, threshold_bytes=0))
        with self.assertRaises(ValueError):
            usage.read_snapshot(self.path, now=1001, require_root_owner=False)

    def test_warning_is_nonblocking_and_suppressed_off_linux(self):
        usage._last_check = float('-inf')
        usage._last_warning = None
        disabled_monitor = types.SimpleNamespace(windows_monitor_installed=lambda: False,
                                                 monitor_dir=lambda: self.path,
                                                 read_windows_combined=mock.Mock())
        with mock.patch.dict(usage.sys.modules, {'rovezero_windows_usage_snapshot': disabled_monitor}), \
             mock.patch.object(usage.sys, 'platform', 'win32'), \
             mock.patch.object(usage, 'read_snapshot') as reader:
            usage.warn_cached_total_usage()
            reader.assert_not_called()
            disabled_monitor.read_windows_combined.assert_not_called()
        usage._last_check = float('-inf')
        usage._last_warning = None
        sink = io.StringIO()
        with mock.patch.object(usage.sys, 'platform', 'linux'), \
             mock.patch.object(usage, 'read_snapshot', side_effect=OSError), \
             mock.patch.object(usage.sys, 'stderr', sink):
            usage.warn_cached_total_usage()
            usage.warn_cached_total_usage()
        self.assertEqual(sink.getvalue().count('rovezero_storage_total_warning:'), 1)
        self.assertIn('snapshot_unavailable', sink.getvalue())
        self.assertIn('"observed_at_utc": "unknown"', sink.getvalue())
        self.assertIn('"age_seconds": "unknown"', sink.getvalue())

    def test_warning_includes_sample_time_scope_and_age(self):
        usage._last_check = float('-inf')
        usage._last_warning = None
        sink = io.StringIO()
        sample = dict(self.value, scope='linux_only')
        with mock.patch.object(usage.sys, 'platform', 'linux'), \
             mock.patch.object(usage, 'read_snapshot', return_value=sample), \
             mock.patch.object(usage.time, 'time', return_value=1030), \
             mock.patch.object(usage.sys, 'stderr', sink):
            usage.warn_cached_total_usage()
        public = json.loads(sink.getvalue().split(': ', 1)[1])
        self.assertEqual(public['observed_at_utc'], sample['observed_at_utc'])
        self.assertEqual(public['scope'], 'linux_only')
        self.assertEqual(public['age_seconds'], 30)

    def test_windows_installed_monitor_reports_cached_combined_total(self):
        usage._last_check = float('-inf')
        usage._last_warning = None
        installed = types.SimpleNamespace(windows_monitor_installed=lambda: True,
            monitor_dir=lambda: Path(self.tmp.name),
            read_windows_combined=mock.Mock(return_value=dict(self.value)))
        sink = io.StringIO()
        with mock.patch.dict(usage.sys.modules, {'rovezero_windows_usage_snapshot': installed}), \
             mock.patch.object(usage.sys, 'platform', 'win32'), \
             mock.patch.object(usage.sys, 'stderr', sink):
            usage.warn_cached_total_usage()
        installed.read_windows_combined.assert_called_once_with()
        self.assertIn('"decision": "exceeded"', sink.getvalue())

    def test_windows_installed_marker_warns_when_snapshot_directory_is_lost(self):
        usage._last_check = float('-inf')
        usage._last_warning = None
        installed = types.SimpleNamespace(windows_monitor_installed=lambda: True,
            monitor_dir=lambda: self.path, read_windows_combined=mock.Mock(side_effect=FileNotFoundError))
        sink = io.StringIO()
        with mock.patch.dict(usage.sys.modules, {'rovezero_windows_usage_snapshot': installed}), \
             mock.patch.object(usage.sys, 'platform', 'win32'), \
             mock.patch.object(usage.sys, 'stderr', sink):
            usage.warn_cached_total_usage()
        self.assertIn('"decision": "unknown"', sink.getvalue())
        self.assertIn('snapshot_unavailable', sink.getvalue())

    def test_default_discovery_includes_rovezero_cache_sibling_parents(self):
        root = Path(self.tmp.name)
        root_cache = root / 'root-cache'
        home = root / 'home'
        var_tmp = root / 'var-tmp'
        for parent in (root_cache, home / 'user' / '.cache', var_tmp):
            parent.mkdir(parents=True)
        (root_cache / 'rovezero').mkdir()
        (home / 'user' / '.cache' / 'rovezero').mkdir()
        roots, parents = usage.default_roots_and_discovery(home, root_cache, var_tmp)
        self.assertEqual(set(roots), {root_cache / 'rovezero', home / 'user' / '.cache' / 'rovezero'})
        self.assertEqual(set(parents), {root_cache, home / 'user' / '.cache', var_tmp})

    def test_origin_proof_is_bound_to_pair_and_does_not_expose_host_ids(self):
        machine = '0123456789abcdef0123456789abcdef'
        boot = '6ba7b810-9dad-11d1-80b4-00c04fd430c8'
        pair = '3c1f640d-f646-4f6f-b7f7-233d076d7f21'
        roots = {'roots': [{'root': self.tmp.name, 'scan_device': 42,
                            'scan_inode': 7, 'identity_unchanged': True}]}
        fake_dir = types.SimpleNamespace(st_dev=42, st_ino=7, st_mode=0o040755)
        with mock.patch.object(usage.os, 'stat', return_value=fake_dir), \
             mock.patch.object(Path, 'lstat', return_value=fake_dir), \
             mock.patch.object(Path, 'read_text', side_effect=[machine, boot]):
            proof = usage.origin_proof(roots, pair)
        self.assertTrue(proof['single_linux_device'])
        self.assertNotIn(machine, json.dumps(proof))
        self.assertNotIn(boot, json.dumps(proof))
        self.assertNotIn(pair, json.dumps(proof))
        self.assertIsNone(usage.origin_proof(roots, 'not-a-uuid'))
        other_device = types.SimpleNamespace(st_dev=99, st_ino=7, st_mode=0o040755)
        with mock.patch.object(usage.os, 'stat', return_value=fake_dir), \
             mock.patch.object(Path, 'lstat', return_value=other_device), \
             mock.patch.object(Path, 'read_text', side_effect=[machine, boot]):
            self.assertIsNone(usage.origin_proof(roots, pair))

    @unittest.skipUnless(hasattr(__import__('os'), 'O_NOFOLLOW'), 'Linux nofollow')
    def test_paired_observations_cross_threshold_and_reject_other_boot(self):
        proof = {'pair_tag': '1' * 64, 'machine_tag': '2' * 64,
                 'boot_tag': '3' * 64, 'single_linux_device': True}
        row = {'observed_at_utc': datetime.fromtimestamp(1000, timezone.utc).isoformat(),
               'allocated_bytes_lower_bound': 10, 'logical_bytes_lower_bound': 10,
               'allocated_measure': 'sum_of_observed_unique_file_reported_allocations'}
        data = {'schema': 'rovezero.total-usage-combined-snapshot.v1',
                'aggregation': 'disjoint_sum', 'pairing_status': 'verified_at_scan',
                'allocated_measure': row['allocated_measure'],
                'components': {'linux': dict(row, origin_proof=proof), 'windows': dict(row)}}
        (Path(self.tmp.name) / 'combined.json').write_text(json.dumps(data), encoding='utf-8')
        linux = dict(self.value, threshold_bytes=18, origin_proof=proof,
                     allocated_bytes_lower_bound=10, logical_bytes_lower_bound=10)
        paired = usage.paired_observation(self.tmp.name, linux, now=1001)
        self.assertEqual(paired['allocated_bytes_lower_bound'], 20)
        self.assertEqual(paired['decision'], 'exceeded')
        self.assertEqual(paired['coverage'], 'partial')
        self.assertIsNone(usage.paired_observation(
            self.tmp.name, dict(linux, origin_proof=dict(proof, boot_tag='4' * 64)), now=1001))
        self.assertIsNone(usage.paired_observation(self.tmp.name, linux, now=5000))

    @unittest.skipUnless(sys.platform.startswith('linux'), 'Linux advisory snapshot')
    def test_cached_linux_snapshot_rechecks_later_windows_pair_without_rescan(self):
        proof = {'pair_tag': '1' * 64, 'machine_tag': '2' * 64,
                 'boot_tag': '3' * 64, 'single_linux_device': True}
        row = {'observed_at_utc': datetime.fromtimestamp(1000, timezone.utc).isoformat(),
               'allocated_bytes_lower_bound': 10, 'logical_bytes_lower_bound': 10,
               'allocated_measure': 'sum_of_observed_unique_file_reported_allocations'}
        combined = {'schema': 'rovezero.total-usage-combined-snapshot.v1',
                    'aggregation': 'disjoint_sum', 'pairing_status': 'verified_at_scan',
                    'allocated_measure': row['allocated_measure'],
                    'components': {'linux': dict(row, origin_proof=proof), 'windows': dict(row)}}
        (Path(self.tmp.name) / 'combined.json').write_text(json.dumps(combined), encoding='utf-8')
        revision = 'e' * 32
        usage._publish(Path(self.tmp.name) / usage.MIRROR_CONFIG,
                       {'schema': 'rovezero.paired-mirror-config.v1', 'directory': self.tmp.name,
                        'revision': revision}, mode=0o600)
        self.write(dict(self.value, decision='unknown', threshold_bytes=18,
                        allocated_bytes_lower_bound=10, logical_bytes_lower_bound=10,
                        origin_proof=proof, _mirror_revision=revision))
        paired = usage.read_snapshot(self.path, now=1001, require_root_owner=False)
        self.assertEqual((paired['allocated_bytes_lower_bound'], paired['decision']), (20, 'exceeded'))
        self.assertEqual(paired['windows_refresh'], 'paired')
        raw_linux = dict(self.value, decision='unknown', threshold_bytes=18,
                         allocated_bytes_lower_bound=10, logical_bytes_lower_bound=10,
                         origin_proof=proof, scope='linux_only', _mirror_revision=revision)
        self.write(dict(paired, _linux_observation=raw_linux))
        twice = usage.read_snapshot(self.path, now=1001, require_root_owner=False)
        self.assertEqual(twice['allocated_bytes_lower_bound'], 20)
        with mock.patch.object(usage, '_read_mirror_config', return_value=None):
            unprivileged = usage.read_snapshot(self.path, now=1001, require_root_owner=False)
        self.assertEqual(unprivileged['allocated_bytes_lower_bound'], 20)
        self.write(dict(raw_linux, _mirror_revision='f' * 32))
        stale_config = usage.read_snapshot(self.path, now=1001, require_root_owner=False)
        self.assertEqual((stale_config['allocated_bytes_lower_bound'], stale_config['decision']), (10, 'unknown'))
        without_config = dict(raw_linux)
        without_config.pop('_mirror_revision')
        self.write(without_config)
        removed_config = usage.read_snapshot(self.path, now=1001, require_root_owner=False)
        self.assertEqual((removed_config['allocated_bytes_lower_bound'], removed_config['decision']), (10, 'unknown'))
        self.write(raw_linux)
        combined['components']['linux']['origin_proof']['boot_tag'] = '4' * 64
        (Path(self.tmp.name) / 'combined.json').write_text(json.dumps(combined), encoding='utf-8')
        local = usage.read_snapshot(self.path, now=1001, require_root_owner=False)
        self.assertEqual((local['allocated_bytes_lower_bound'], local['decision']), (10, 'unknown'))
        # Older or externally supplied already-paired snapshots must not add
        # the same Windows bytes a second time.
        self.write(dict(self.value, decision='unknown', threshold_bytes=25,
                        allocated_bytes_lower_bound=20, logical_bytes_lower_bound=20,
                        scope='paired_observations_not_host_physical_total',
                        origin_proof=proof, _mirror_revision=revision))
        already_paired = usage.read_snapshot(self.path, now=1001, require_root_owner=False)
        self.assertEqual((already_paired['allocated_bytes_lower_bound'], already_paired['decision']),
                         (20, 'unknown'))

    @unittest.skipUnless(sys.platform.startswith('linux'), 'Linux advisory snapshot')
    def test_scan_preserves_raw_linux_alongside_existing_windows_pair(self):
        root = Path(self.tmp.name) / 'source'
        mirror = Path(self.tmp.name) / 'mirror'
        root.mkdir()
        mirror.mkdir()
        (root / 'file').write_bytes(b'content')
        proof = {'pair_tag': '1' * 64, 'machine_tag': '2' * 64,
                 'boot_tag': '3' * 64, 'single_linux_device': True}

        def paired(_mirror, linux):
            return dict(linux, allocated_bytes_lower_bound=linux['allocated_bytes_lower_bound'] + 10,
                        scope='paired_observations_not_host_physical_total')

        with mock.patch.object(usage, 'default_roots_and_discovery', return_value=([root], [])), \
             mock.patch.object(usage, 'origin_proof', return_value=proof), \
             mock.patch.object(usage, 'paired_observation', side_effect=paired), \
             mock.patch.object(usage, 'request_windows_refresh_once') as request:
            published = usage.create_snapshot(self.path, mirror_dir=mirror, pair_id='configured')
        stored = json.loads(self.path.read_text(encoding='utf-8'))
        self.assertEqual(published['allocated_bytes_lower_bound'], stored['allocated_bytes_lower_bound'])
        self.assertEqual(stored['allocated_bytes_lower_bound'],
                         stored['_linux_observation']['allocated_bytes_lower_bound'] + 10)
        self.assertEqual(stored['scope'], 'paired_observations_not_host_physical_total')
        self.assertNotIn(str(mirror), json.dumps(stored))
        self.assertEqual((Path(self.tmp.name) / usage.MIRROR_CONFIG).stat().st_mode & 0o777, 0o600)
        request.assert_not_called()

    @unittest.skipUnless(sys.platform.startswith('linux'), 'Linux advisory snapshot')
    def test_scan_disabling_mirror_never_reuses_previous_configuration(self):
        root = Path(self.tmp.name) / 'source'
        mirror = Path(self.tmp.name) / 'mirror'
        root.mkdir()
        mirror.mkdir()
        (root / 'file').write_bytes(b'content')
        proof = {'pair_tag': '1' * 64, 'machine_tag': '2' * 64,
                 'boot_tag': '3' * 64, 'single_linux_device': True}
        row = {'observed_at_utc': datetime.now(timezone.utc).isoformat(),
               'allocated_bytes_lower_bound': 1000, 'logical_bytes_lower_bound': 1000,
               'allocated_measure': 'sum_of_observed_unique_file_reported_allocations'}
        combined = {'schema': 'rovezero.total-usage-combined-snapshot.v1',
                    'aggregation': 'disjoint_sum', 'pairing_status': 'verified_at_scan',
                    'allocated_measure': row['allocated_measure'],
                    'components': {'linux': dict(row, origin_proof=proof), 'windows': dict(row)}}
        (mirror / 'combined.json').write_text(json.dumps(combined), encoding='utf-8')
        with mock.patch.object(usage, 'default_roots_and_discovery', return_value=([root], [])), \
             mock.patch.object(usage, 'origin_proof', return_value=proof):
            usage.create_snapshot(self.path, mirror_dir=mirror, pair_id='configured')
            first = json.loads(self.path.read_text(encoding='utf-8'))
            private = json.loads((Path(self.tmp.name) / usage.MIRROR_CONFIG).read_text(encoding='utf-8'))
            self.assertEqual(first['_mirror_revision'], private['revision'])
            usage.create_snapshot(self.path, pair_id='configured')
        second = usage.read_snapshot(self.path, require_root_owner=False)
        self.assertNotIn('_mirror_revision', second)
        self.assertEqual(second['scope'], 'default_rovezero_caches_siblings_and_var_tmp_xdg_exceptions_unknown')
        self.assertTrue((Path(self.tmp.name) / usage.MIRROR_CONFIG).exists())

    @unittest.skipUnless(sys.platform.startswith('linux'), 'Linux advisory flock')
    def test_windows_refresh_requests_once_per_pair_boot_and_failure_is_advisory(self):
        proof = {'pair_tag': 'a' * 64, 'boot_tag': 'b' * 64, 'single_linux_device': True}
        linux = {'origin_proof': proof}
        with mock.patch.object(usage.subprocess, 'run', return_value=types.SimpleNamespace(returncode=0)) as run:
            self.assertEqual(usage.request_windows_refresh_once(self.tmp.name, linux), 'requested')
            self.assertEqual(usage.request_windows_refresh_once(self.tmp.name, linux), 'attempt_already_recorded')
            self.assertEqual(run.call_count, 1)
            self.assertEqual(run.call_args.args[0][1:], ['/Run', '/TN', usage.WINDOWS_REFRESH_TASK])
        other = {'origin_proof': dict(proof, boot_tag='c' * 64)}
        with mock.patch.object(usage.subprocess, 'run', side_effect=OSError):
            self.assertEqual(usage.request_windows_refresh_once(self.tmp.name, other), 'request_failed')
            self.assertEqual(usage.request_windows_refresh_once(self.tmp.name, other), 'attempt_already_recorded')

    @unittest.skipUnless(sys.platform.startswith('linux'), 'Linux advisory flock')
    def test_windows_refresh_rejects_symlink_or_damaged_state_without_running_task(self):
        target = self.path.with_name('victim')
        target.write_text('unchanged', encoding='utf-8')
        state = Path(self.tmp.name) / usage.WINDOWS_REFRESH_STATE
        state.symlink_to(target)
        linux = {'origin_proof': {'pair_tag': 'a' * 64, 'boot_tag': 'b' * 64,
                                  'single_linux_device': True}}
        with mock.patch.object(usage.subprocess, 'run') as run:
            self.assertEqual(usage.request_windows_refresh_once(self.tmp.name, linux), 'state_unavailable')
            run.assert_not_called()
        self.assertEqual(target.read_text(encoding='utf-8'), 'unchanged')
        state.unlink()
        state.write_bytes(b'partial')
        state.chmod(0o600)
        with mock.patch.object(usage.subprocess, 'run') as run:
            self.assertEqual(usage.request_windows_refresh_once(self.tmp.name, linux), 'state_unavailable')
            run.assert_not_called()


if __name__ == '__main__':
    unittest.main()
