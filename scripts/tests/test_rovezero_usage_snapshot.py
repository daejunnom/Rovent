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


if __name__ == '__main__':
    unittest.main()
