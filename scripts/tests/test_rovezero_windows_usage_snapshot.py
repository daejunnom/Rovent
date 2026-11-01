"""Focused checks for Windows snapshot lifetime, privacy, and aggregation."""
from datetime import datetime, timezone
import json
import hashlib
import io
import os
from pathlib import Path
import tempfile
import threading
import unittest
import uuid
from unittest import mock

import rovezero_windows_usage_snapshot as usage


def sample(schema, allocated, observed=1000, coverage='complete_for_scoped_regular_files'):
    return {'schema': schema, 'observed_at_utc': datetime.fromtimestamp(observed, timezone.utc).isoformat(),
            'decision': 'within', 'coverage': coverage, 'threshold_bytes': 18,
            'allocated_bytes_lower_bound': allocated, 'logical_bytes_lower_bound': allocated + 1,
            'allocated_measure': usage.ALLOCATED_MEASURE}


class CombinedTests(unittest.TestCase):
    def setUp(self):
        # Never consult the user's real registry configuration in fixtures.
        patcher = mock.patch.object(usage, '_read_source_roots_config', return_value=None)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_changed_source_configuration_never_reuses_prior_paired_sum(self):
        old = '3c1f640d-f646-4f6f-b7f7-233d076d7f21'
        new = '1996a1d1-1526-48ff-ac81-6c9230cd09bb'
        windows = sample(usage.WINDOWS_SCHEMA, 10 * usage.GIB)
        windows['source_roots_revision'] = old
        linux = sample(usage.LINUX_SCHEMA, 10 * usage.GIB)
        combined = usage.combine(windows, linux, proven_disjoint=True, now=1001)
        combined['pairing_status'] = 'verified_at_scan'
        with mock.patch.object(usage, '_read_snapshot', return_value=combined), \
             mock.patch.object(usage, '_read_source_roots_config', return_value={'revision': new}):
            result = usage.read_windows_combined(Path('ignored'), now=1001)
        self.assertEqual(result['aggregation'], 'overlap_unknown_max')
        self.assertEqual(result['windows_source_scope'], 'changed_or_unavailable')
        self.assertEqual(result['decision'], 'unknown')

    def test_disjoint_sum_warns_even_when_neither_side_individually_exceeds(self):
        result = usage.combine(sample(usage.WINDOWS_SCHEMA, 10 * usage.GIB),
                               sample(usage.LINUX_SCHEMA, 10 * usage.GIB),
                               proven_disjoint=True, now=1001, threshold=18 * usage.GIB)
        self.assertEqual((result['allocated_bytes_lower_bound'], result['decision']),
                         (20 * usage.GIB, 'exceeded'))
        self.assertEqual(result['aggregation'], 'disjoint_sum')

    def test_overlap_or_stale_cannot_create_a_false_total(self):
        windows = sample(usage.WINDOWS_SCHEMA, 10)
        linux = sample(usage.LINUX_SCHEMA, 9)
        result = usage.combine(windows, linux, proven_disjoint=False, now=1001, threshold=18)
        self.assertEqual((result['allocated_bytes_lower_bound'], result['decision']), (10, 'unknown'))
        self.assertEqual(result['aggregation'], 'overlap_unknown_max')
        linux['observed_at_utc'] = datetime.fromtimestamp(5000, timezone.utc).isoformat()
        result = usage.combine(windows, linux, proven_disjoint=True, now=5001, threshold=18)
        self.assertEqual((result['allocated_bytes_lower_bound'], result['coverage']), (9, 'partial'))
        self.assertEqual(result['components']['windows'], {'status': 'unknown'})

    def test_proven_exceeded_survives_partial_coverage(self):
        result = usage.combine(sample(usage.WINDOWS_SCHEMA, 19, coverage='partial'), None,
                               proven_disjoint=False, now=1001, threshold=18)
        self.assertEqual(result['decision'], 'exceeded')
        self.assertEqual(result['coverage'], 'partial')

    def test_paired_partial_observations_can_still_exceed(self):
        windows = sample(usage.WINDOWS_SCHEMA, 10 * usage.GIB, coverage='partial')
        linux = sample(usage.LINUX_SCHEMA, 10 * usage.GIB, coverage='partial')
        result = usage.combine(windows, linux, proven_disjoint=True,
                               now=1001, threshold=18 * usage.GIB)
        self.assertEqual((result['allocated_bytes_lower_bound'], result['decision'], result['coverage']),
                         (20 * usage.GIB, 'exceeded', 'partial'))

    def test_fresh_combined_file_rechecks_component_ages_at_consumption(self):
        original = usage.combine(sample(usage.WINDOWS_SCHEMA, 10, observed=1400),
                                 sample(usage.LINUX_SCHEMA, 9, observed=1000),
                                 proven_disjoint=True, now=1401, threshold=18)
        with mock.patch.object(usage, '_read_nofollow', return_value=json.dumps(original).encode()), \
                mock.patch.dict(os.environ, {'APPDATA': str(Path.home())}):
            result = usage.read_windows_combined('ignored', now=4700)
        self.assertEqual((result['allocated_bytes_lower_bound'], result['decision']), (10, 'unknown'))
        self.assertEqual(result['components']['linux'], {'status': 'unknown'})

    def test_component_staleness_respects_stricter_caller_max_age(self):
        original = usage.combine(sample(usage.WINDOWS_SCHEMA, 10, observed=995),
                                 sample(usage.LINUX_SCHEMA, 9, observed=950),
                                 proven_disjoint=True, now=1000, threshold=18)
        with mock.patch.object(usage, '_read_nofollow', return_value=json.dumps(original).encode()):
            result = usage.read_windows_combined('ignored', now=1001, max_age_seconds=10)
        self.assertEqual(result['components']['linux'], {'status': 'unknown'})
        self.assertEqual(result['decision'], 'unknown')

    def test_fresh_pairing_retains_separate_origin_proof(self):
        windows = sample(usage.WINDOWS_SCHEMA, 10)
        linux = sample(usage.LINUX_SCHEMA, 9)
        proof = {'pair_tag': 'a' * 64, 'machine_tag': 'b' * 64,
                 'boot_tag': 'c' * 64, 'single_linux_device': True}
        linux['origin_proof'] = dict(proof, ignored_private_value='not_public')
        original = usage.combine(windows, linux, proven_disjoint=True, now=1001, threshold=18)
        original['pairing_status'] = 'verified_at_scan'
        with mock.patch.object(usage, '_read_nofollow', return_value=json.dumps(original).encode()):
            result = usage.read_windows_combined('ignored', now=1002)
        self.assertEqual((result['decision'], result['allocated_bytes_lower_bound']), ('exceeded', 19))
        self.assertEqual(result['components']['linux']['origin_proof'], proof)

    def test_pair_manifest_revalidates_device_and_root_identities(self):
        linux = sample(usage.LINUX_SCHEMA, 9)
        linux['origin_proof'] = {'pair_tag': 'a' * 64, 'machine_tag': 'b' * 64,
                                 'boot_tag': 'c' * 64, 'single_linux_device': True}
        distro = '12345678-1234-4234-8234-123456789abc'
        roots = [Path('C:/fixture/RoveZero')]
        vhd = {'canonical_path': 'd:\\wsl\\ext4.vhdx', 'volume_serial': 10, 'file_index': 100}
        root = {'canonical_path': 'c:\\fixture\\rovezero', 'volume_serial': 20, 'file_index': 200}
        with mock.patch.object(usage, '_registered_vhd', return_value=vhd), \
                mock.patch.object(usage, '_identity_nofollow', return_value=root):
            manifest = usage.prepare_pair_manifest(linux, distro, roots, now=1001)
            self.assertTrue(usage.pairing_matches(manifest, linux, roots, now=1001))
            self.assertNotIn('fixture', json.dumps(manifest))
            self.assertFalse(usage.pairing_matches(manifest, linux, roots, now=5001))
            changed = dict(linux, origin_proof=dict(linux['origin_proof'], single_linux_device=False))
            self.assertFalse(usage.pairing_matches(manifest, changed, roots, now=1001))
        with mock.patch.object(usage, '_registered_vhd', return_value=dict(vhd, file_index=101)), \
                mock.patch.object(usage, '_identity_nofollow', return_value=root):
            self.assertFalse(usage.pairing_matches(manifest, linux, roots, now=1001))


@unittest.skipUnless(os.name == 'nt', 'Windows handle semantics')
class WindowsHandleTests(unittest.TestCase):
    def setUp(self):
        # Fixture tests must not inherit a real user's installed monitor.
        self.shared_patch = mock.patch.object(usage, '_shared_monitor_dir', return_value=None)
        self.shared_patch.start()
        self.addCleanup(self.shared_patch.stop)
        self.installed_patch = mock.patch.object(usage, 'windows_monitor_installed', return_value=False)
        self.installed = self.installed_patch.start()
        self.addCleanup(self.installed_patch.stop)
        # Keep fixture observers isolated from an actual user-level timer while
        # still testing mutual exclusion among fixture threads/processes.
        real_mutex = usage._WindowsMutex
        fixture_namespace = uuid.uuid4().hex
        class FixtureMutex(real_mutex):
            def __init__(self, root):
                super().__init__(root, _fixture_namespace=fixture_namespace)
        self.mutex_patch = mock.patch.object(usage, '_WindowsMutex', FixtureMutex)
        self.mutex_patch.start()
        self.addCleanup(self.mutex_patch.stop)

    def test_package_shared_setup_ignores_distinct_registry_and_appdata(self):
        self.shared_patch.stop()
        self.installed_patch.stop()
        with tempfile.TemporaryDirectory() as directory:
            local = Path(directory) / 'local'
            package = local / 'Packages' / 'OpenAI.Codex_fixture' / 'LocalCache' / 'Roaming'
            target = package / 'RoveZero' / 'storage' / 'total-usage'
            target.mkdir(parents=True)
            normal = Path(directory) / 'normal'
            (normal / 'RoveZero').mkdir(parents=True)
            source = Path(directory) / 'checkout' / 'RoveZero'
            source.mkdir(parents=True)
            (source / '.git').touch()
            with mock.patch.dict(os.environ, {'LOCALAPPDATA': str(local), 'APPDATA': str(normal)}, clear=False):
                usage.register_monitor_dir(target)
                usage.register_source_roots([source])
                config = usage._read_source_roots_config()
                self.assertEqual(usage._checked_source_roots(config), [source])
                with mock.patch.dict(os.environ, {'APPDATA': str(package),
                        'RZ_USAGE_MONITOR_DIR': str(normal / 'other')}):
                    self.assertTrue(usage.windows_monitor_installed())
                    self.assertEqual(usage.monitor_dir(), target)
                    self.assertEqual(usage._read_source_roots_config()['revision'], config['revision'])
                config_file = target / usage.SHARED_MONITOR_FILE
                shared_config = json.loads(usage._read_nofollow(config_file))
                del shared_config['sources']
                usage._atomic_write(config_file, shared_config)
                with self.assertRaises(ValueError):
                    usage._read_source_roots_config()
                config_file.unlink()
                with self.assertRaises(ValueError):
                    usage.monitor_dir()

    def test_windows_monitor_installation_marker_is_independent_of_snapshot_files(self):
        import winreg
        self.installed_patch.stop()
        with mock.patch.object(winreg, 'OpenKey', side_effect=FileNotFoundError):
            self.assertFalse(usage.windows_monitor_installed())
        with mock.patch.object(winreg, 'OpenKey', return_value=mock.MagicMock()), \
             mock.patch.object(winreg, 'QueryValueEx', side_effect=FileNotFoundError):
            with self.assertRaises(ValueError):
                usage.windows_monitor_installed()
        with mock.patch.object(winreg, 'OpenKey', return_value=mock.MagicMock()), \
             mock.patch.object(winreg, 'QueryValueEx', return_value=(1, winreg.REG_DWORD)):
            self.assertTrue(usage.windows_monitor_installed())
        with mock.patch.object(winreg, 'OpenKey', return_value=mock.MagicMock()), \
             mock.patch.object(winreg, 'QueryValueEx', return_value=('bad', winreg.REG_SZ)):
            with self.assertRaises(ValueError):
                usage.windows_monitor_installed()

    def test_installed_directory_is_pinned_across_appdata_redirection(self):
        import winreg
        self.installed_patch.stop()
        with tempfile.TemporaryDirectory() as directory:
            normal = Path(directory) / 'normal-roaming'
            redirected = (Path(directory) / 'Packages' / 'OpenAI.Codex_fixture' /
                          'LocalCache' / 'Roaming')
            target = normal / 'RoveZero' / 'storage' / 'total-usage'
            target.mkdir(parents=True)
            (redirected / 'RoveZero').mkdir(parents=True)
            values = {}
            def query(_key, name):
                if name not in values:
                    raise FileNotFoundError(name)
                return values[name]
            def set_value(_key, name, _reserved, kind, value):
                values[name] = (value, kind)
            with mock.patch.object(winreg, 'OpenKey', return_value=mock.MagicMock()), \
                 mock.patch.object(winreg, 'CreateKeyEx', return_value=mock.MagicMock()), \
                 mock.patch.object(winreg, 'QueryValueEx', side_effect=query), \
                 mock.patch.object(winreg, 'SetValueEx', side_effect=set_value), \
                 mock.patch.dict(os.environ, {'APPDATA': str(normal), 'LOCALAPPDATA': directory}, clear=False):
                self.assertEqual(usage.register_monitor_dir(target)['status'], 'monitor_registered')
                self.assertEqual(usage.monitor_dir(), target)
                with mock.patch.dict(os.environ, {'APPDATA': str(redirected),
                                                  'RZ_USAGE_MONITOR_DIR': str(redirected / 'other')}):
                    self.assertEqual(usage.monitor_dir(), target)
                target.rmdir()
                with self.assertRaises(OSError):
                    usage.monitor_dir()
                # Even a supplied alternate override cannot hide lost/corrupt setup.
                with mock.patch.dict(os.environ, {'RZ_USAGE_MONITOR_DIR': str(redirected / 'other')}):
                    with self.assertRaises(OSError):
                        usage.monitor_dir()
                values.pop('Configuration')
                with self.assertRaises(FileNotFoundError):
                    usage.monitor_dir()
                values['Configuration'] = ('not-json', winreg.REG_SZ)
                with self.assertRaises(ValueError):
                    usage.monitor_dir()
                values['Configuration'] = ('{}', winreg.REG_DWORD)
                with self.assertRaises(ValueError):
                    usage.monitor_dir()

    def test_corrupt_installed_configuration_emits_unknown_without_fallback(self):
        import winreg
        import rovezero_usage_snapshot as warning
        self.installed_patch.stop()
        def query(_key, name):
            return (1, winreg.REG_DWORD) if name == 'Enabled' else ('broken', winreg.REG_SZ)
        warning._last_check = float('-inf')
        warning._last_warning = None
        sink = io.StringIO()
        with mock.patch.object(winreg, 'OpenKey', return_value=mock.MagicMock()), \
             mock.patch.object(winreg, 'QueryValueEx', side_effect=query), \
             mock.patch.object(warning.sys, 'stderr', sink):
            warning.warn_cached_total_usage()
        self.assertIn('"decision": "unknown"', sink.getvalue())
        self.assertIn('snapshot_unavailable', sink.getvalue())

    def test_discovery_deduplicates_redirected_appdata_root(self):
        with tempfile.TemporaryDirectory() as directory:
            base = (Path(directory) / 'Packages' / 'OpenAI.Codex_fixture' /
                    'LocalCache' / 'Roaming')
            (base / 'RoveZero').mkdir(parents=True)
            with mock.patch.dict(os.environ, {'APPDATA': str(base), 'LOCALAPPDATA': directory}, clear=False):
                roots = usage.discover_windows_roots()
            self.assertEqual(len(roots), 1)
            self.assertEqual(usage._normalize_windows_path(roots[0]),
                             usage._normalize_windows_path(base / 'RoveZero'))

    def test_configured_source_roots_are_explicit_checked_and_deduplicated(self):
        import winreg
        with tempfile.TemporaryDirectory() as directory:
            appdata = Path(directory) / 'roaming'
            (appdata / 'RoveZero').mkdir(parents=True)
            source = Path(directory) / 'Desktop' / 'RoveZero'
            source.mkdir(parents=True)
            (source / '.git').touch()
            worktree = Path(directory) / 'worktrees' / 'slot' / 'RoveZero'
            worktree.mkdir(parents=True)
            (worktree / '.git').mkdir()
            values = {}
            def query(_key, name):
                if name not in values:
                    raise FileNotFoundError(name)
                return values[name]
            def set_value(_key, name, _reserved, kind, value):
                values[name] = (value, kind)
            self.installed.return_value = True
            with mock.patch.dict(os.environ, {'APPDATA': str(appdata), 'LOCALAPPDATA': directory}, clear=False), \
                 mock.patch.object(winreg, 'OpenKey', return_value=mock.MagicMock()), \
                 mock.patch.object(winreg, 'QueryValueEx', side_effect=query), \
                 mock.patch.object(winreg, 'SetValueEx', side_effect=set_value):
                result = usage.register_source_roots([source, source, worktree])
                self.assertEqual(result, {'status': 'source_roots_registered', 'count': 2})
                config = usage._read_source_roots_config()
                self.assertEqual(set(usage._checked_source_roots(config)), {source, worktree})
                self.assertEqual(set(usage.discover_windows_roots()), {appdata / 'RoveZero', source, worktree})
                (source / '.git').unlink()
                with self.assertRaises(OSError):
                    usage.discover_windows_roots()
                values.pop('SourceRootsEnabled')
                with self.assertRaises(ValueError):
                    usage._read_source_roots_config()

    def test_discovery_deduplicates_distinct_paths_with_same_handle_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            normal = Path(directory) / 'Roaming'
            redirected = (Path(directory) / 'Packages' / 'OpenAI.Codex_fixture' /
                          'LocalCache' / 'Roaming')
            (normal / 'RoveZero').mkdir(parents=True)
            (redirected / 'RoveZero').mkdir(parents=True)
            def same_identity(path, *, directory):
                return {'canonical_path': usage._normalize_windows_path(path),
                        'volume_serial': 8, 'file_index': 13}
            with mock.patch.object(usage, '_identity_nofollow', side_effect=same_identity), \
                 mock.patch.dict(os.environ, {'APPDATA': str(normal), 'LOCALAPPDATA': directory}, clear=False):
                normal_roots = usage.discover_windows_roots()
                # Both valid spellings remain usable for a monitor override,
                # even though only one is selected for allocation accounting.
                for base in (normal, redirected):
                    path = base / 'RoveZero' / 'storage' / 'total-usage'
                    self.assertEqual(usage._scoped_monitor_dir(path), path)
                with mock.patch.dict(os.environ, {'APPDATA': str(redirected)}, clear=False):
                    redirected_roots = usage.discover_windows_roots()
            self.assertEqual(len(normal_roots), 1)
            self.assertEqual(len(redirected_roots), 1)
            self.assertEqual(usage._normalize_windows_path(normal_roots[0]),
                             usage._normalize_windows_path(redirected_roots[0]))

    def test_root_overlap_is_distinct_from_alias_and_preserves_other_volumes(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory) / 'parent'
            child = parent / 'child'
            unknown = Path(directory) / 'unknown'
            def identities(path, *, directory):
                if path == unknown:
                    raise OSError('identity unavailable')
                return {'canonical_path': usage._normalize_windows_path(path),
                        'volume_serial': 1, 'file_index': 1 if path == parent else 2}
            with mock.patch.object(usage, '_identity_nofollow', side_effect=identities):
                self.assertEqual(usage._distinct_windows_roots([child, parent, unknown]), [parent, unknown])
            def different_volume(path, *, directory):
                return {'canonical_path': usage._normalize_windows_path(path),
                        'volume_serial': 1 if path == parent else 2, 'file_index': 1}
            with mock.patch.object(usage, '_identity_nofollow', side_effect=different_volume):
                self.assertEqual(usage._distinct_windows_roots([child, parent]), [parent, child])
            middle = parent / 'mounted'
            nested = middle / 'nested'
            def intermediate_mount(path, *, directory):
                return {'canonical_path': usage._normalize_windows_path(path),
                        'volume_serial': 2 if path == middle else 1,
                        'file_index': 1 if path == parent else 3}
            with mock.patch.object(usage, '_identity_nofollow', side_effect=intermediate_mount):
                self.assertEqual(usage._distinct_windows_roots([nested, parent]), [parent, nested])

    def test_monitor_override_requires_rovezero_storage_scope(self):
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.dict(os.environ, {'APPDATA': directory, 'LOCALAPPDATA': directory}, clear=False):
            allowed = Path(directory) / 'RoveZero' / 'storage' / 'total-usage' / 'paired'
            with mock.patch.dict(os.environ, {'RZ_USAGE_MONITOR_DIR': str(allowed)}):
                self.assertEqual(usage.monitor_dir(), allowed)
            outside = Path(directory) / 'other' / 'storage' / 'total-usage'
            with mock.patch.dict(os.environ, {'RZ_USAGE_MONITOR_DIR': str(outside)}):
                with self.assertRaises(ValueError):
                    usage.monitor_dir()
            with mock.patch.dict(os.environ, {'RZ_USAGE_MONITOR_DIR': 'relative/total-usage'}):
                with self.assertRaises(ValueError):
                    usage.monitor_dir()

    def test_nofollow_snapshot_refuses_hardlinks_and_bounds_reads(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'snapshot.json'
            path.write_bytes(b'{}')
            self.assertEqual(usage._read_nofollow(path), b'{}')
            alias = Path(directory) / 'alias'
            os.link(path, alias)
            with self.assertRaises(ValueError):
                usage._read_nofollow(path)
            alias.unlink()
            path.write_bytes(b'x' * (usage.MAX_BYTES + 1))
            with self.assertRaises(ValueError):
                usage._read_nofollow(path)

    def test_mutex_rejects_concurrent_observer(self):
        with tempfile.TemporaryDirectory() as directory:
            results = []
            with usage._WindowsMutex(Path(directory)) as first:
                self.assertTrue(first)
                def contender():
                    with usage._WindowsMutex(Path(directory)) as second:
                        results.append(second)
                thread = threading.Thread(target=contender)
                thread.start()
                thread.join(timeout=5)
                self.assertFalse(thread.is_alive())
            self.assertEqual(results, [False])

    def test_fixture_scan_publishes_totals_without_paths(self):
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.dict(os.environ, {'APPDATA': directory, 'LOCALAPPDATA': directory}):
            root = Path(directory) / 'RoveZero'
            root.mkdir()
            (root / 'fixture.bin').write_bytes(b'x' * 128)
            package_root = (Path(directory) / 'Packages' / 'OpenAI.Codex_fixture' /
                            'LocalCache' / 'Roaming' / 'RoveZero')
            package_root.mkdir(parents=True)
            (package_root / 'fixture2.bin').write_bytes(b'y' * 256)
            now = datetime.now(timezone.utc).timestamp()
            linux = sample(usage.LINUX_SCHEMA, 4096, observed=now, coverage='partial')
            usage._atomic_write(usage.monitor_dir() / 'linux.json', linux)
            result = usage.scan_once(max_entries=1000, max_seconds=5)
            self.assertEqual(result['aggregation'], 'overlap_unknown_max')
            self.assertEqual(result['coverage'], 'partial')
            self.assertEqual(result['components']['linux']['allocated_bytes_lower_bound'], 4096)
            self.assertNotIn(directory.lower(), json.dumps(result).lower())
            windows = usage._read_snapshot(usage.monitor_dir() / 'windows.json', usage.WINDOWS_SCHEMA)
            self.assertEqual(windows['scope'], 'windows_standard_codex_and_configured_sources_custom_unknown')
            self.assertGreaterEqual(windows['logical_bytes_lower_bound'], 384)

    def test_observed_root_identity_must_match_the_same_current_handle(self):
        import rovezero_total_usage as scanner
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'RoveZero'
            root.mkdir()
            (root / 'fixture.bin').write_bytes(b'x')
            result = scanner.scan_total_usage([root], max_entries=100, max_seconds=5)
            self.assertTrue(usage._scan_roots_match_handles(result, [root]))
            changed = dict(result, roots=[dict(result['roots'][0], scan_inode=0)])
            self.assertFalse(usage._scan_roots_match_handles(changed, [root]))
            skipped = dict(result, roots=[])
            self.assertFalse(usage._scan_roots_match_handles(skipped, [root]))
            incomplete = dict(result, coverage='partial',
                              roots=[dict(result['roots'][0], partial=True)])
            self.assertTrue(usage._scan_roots_match_handles(incomplete, [root]))
            changed_root = dict(result, roots=[dict(result['roots'][0], identity_unchanged=False)])
            self.assertFalse(usage._scan_roots_match_handles(changed_root, [root]))

    def test_explicit_pairing_registration_writes_only_sealed_identities(self):
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.dict(os.environ, {'APPDATA': directory, 'LOCALAPPDATA': directory}):
            root = Path(directory) / 'RoveZero'
            root.mkdir()
            now = datetime.now(timezone.utc).timestamp()
            linux = sample(usage.LINUX_SCHEMA, 4096, observed=now, coverage='partial')
            linux['origin_proof'] = {'pair_tag': 'a' * 64, 'machine_tag': 'b' * 64,
                                     'boot_tag': 'c' * 64, 'single_linux_device': True}
            usage._atomic_write(usage.monitor_dir() / 'linux.json', linux)
            vhd = {'canonical_path': 'd:\\wsl\\ext4.vhdx', 'volume_serial': 10, 'file_index': 100}
            root_identity = {'canonical_path': 'c:\\fixture\\rovezero', 'volume_serial': 20, 'file_index': 200}
            with mock.patch.object(usage, '_registered_vhd', return_value=vhd), \
                    mock.patch.object(usage, '_identity_nofollow', return_value=root_identity), \
                    mock.patch.object(usage, '_verify_distro_origin', return_value=vhd):
                result = usage.register_pairing('12345678-1234-4234-8234-123456789abc',
                                                'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa')
            self.assertEqual(result['status'], 'pairing_registered')
            raw = usage._read_nofollow(usage.monitor_dir() / 'pairing.json')
            self.assertNotIn(directory.lower().encode(), raw.lower())
            self.assertNotIn(b'fixture', raw.lower())
            self.assertEqual(json.loads(raw)['schema'], usage.PAIRING_SCHEMA)

    def test_selected_distro_mismatch_and_timeout_preserve_old_pairing(self):
        pair_id = 'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa'
        digest = hashlib.sha256(b'RZ-PAIR-v1\0' + pair_id.encode('ascii')).hexdigest()
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.dict(os.environ, {'APPDATA': directory, 'LOCALAPPDATA': directory,
                                             'SystemRoot': 'C:\\Windows'}):
            (Path(directory) / 'RoveZero').mkdir()
            now = datetime.now(timezone.utc).timestamp()
            linux = sample(usage.LINUX_SCHEMA, 4096, observed=now, coverage='partial')
            linux['origin_proof'] = {'pair_tag': digest, 'machine_tag': 'b' * 64,
                                     'boot_tag': 'c' * 64, 'single_linux_device': True}
            usage._atomic_write(usage.monitor_dir() / 'linux.json', linux)
            old = {'schema': usage.PAIRING_SCHEMA, 'existing': True}
            usage._atomic_write(usage.monitor_dir() / 'pairing.json', old)
            self.assertTrue(usage.monitor_dir().is_dir())
            other = {'pair_tag': digest, 'machine_tag': 'd' * 64, 'boot_tag': 'c' * 64}
            result = mock.Mock(returncode=0, stdout=json.dumps(other).encode())
            identity = {'canonical_path': 'd:\\fixture', 'volume_serial': 10, 'file_index': 100}
            with mock.patch.object(usage, '_registered_distro', return_value=('fixture', Path('D:/x'))), \
                    mock.patch.object(usage, '_identity_nofollow', return_value=identity), \
                    mock.patch.object(usage.subprocess, 'run', return_value=result):
                with self.assertRaisesRegex(ValueError, 'identity_mismatch'):
                    usage.register_pairing('12345678-1234-4234-8234-123456789abc', pair_id)
            self.assertEqual(json.loads(usage._read_nofollow(usage.monitor_dir() / 'pairing.json')), old)
            with mock.patch.object(usage, '_registered_distro', return_value=('fixture', Path('D:/x'))), \
                    mock.patch.object(usage, '_identity_nofollow', return_value=identity), \
                    mock.patch.object(usage.subprocess, 'run',
                                      side_effect=usage.subprocess.TimeoutExpired('wsl', 15)):
                with self.assertRaises(usage.subprocess.TimeoutExpired):
                    usage.register_pairing('12345678-1234-4234-8234-123456789abc', pair_id)
            self.assertEqual(json.loads(usage._read_nofollow(usage.monitor_dir() / 'pairing.json')), old)


if __name__ == '__main__':
    unittest.main()
