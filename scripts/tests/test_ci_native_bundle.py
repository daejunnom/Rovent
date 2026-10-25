"""Pure metadata/FD custody cases; no Cargo, target ELF or neural execution."""
from __future__ import annotations

import copy
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
import tarfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ci_native_bundle as bundle


class NativeBundleTests(unittest.TestCase):
    def test_compressed_artifact_and_uncompressed_payload_have_separate_bounds(self):
        rows = {str(index): {'pin': {'bytes': size << 20}}
                for index, size in enumerate((200, 200, 150))}
        self.assertGreater(bundle.selected_elf_payload(rows), bundle.ARTIFACT_CAP)
        self.assertLessEqual(bundle.selected_elf_payload(rows), bundle.ELF_PAYLOAD_CAP)
        for sizes in ((200 << 20,) * 4, (bundle.ELF_CAP + 1,), (True,), (0,)):
            with self.subTest(sizes=sizes), self.assertRaisesRegex(RuntimeError, 'SelectedELF'):
                bundle.selected_elf_payload({str(i): {'pin': {'bytes': size}}
                                             for i, size in enumerate(sizes)})

    def test_compressed_writer_roundtrip_and_exact_byte_accounting(self):
        payload = b'\0' * (2 << 20)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'synthetic.tar.gz'
            with path.open('xb') as stream:
                writer = bundle.BoundedArchiveWriter(stream, 16 << 10)
                with tarfile.open(fileobj=writer, mode='w:gz', format=tarfile.USTAR_FORMAT) as archive:
                    member = tarfile.TarInfo('bin/synthetic')
                    member.size = len(payload)
                    archive.addfile(member, io.BytesIO(payload))
                self.assertEqual(writer.written, stream.tell())
                self.assertLessEqual(writer.written, writer.maximum)
            with tarfile.open(path, 'r:gz') as archive:
                self.assertEqual(archive.getnames(), ['bin/synthetic'])
                self.assertEqual(archive.extractfile('bin/synthetic').read(), payload)

    def test_compressed_writer_refuses_before_crossing_disk_bound(self):
        payload = b''.join(hashlib.sha256(str(i).encode()).digest() for i in range(2048))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'synthetic-overflow.tar.gz'
            with path.open('xb') as stream:
                writer = bundle.BoundedArchiveWriter(stream, 1024)
                with self.assertRaisesRegex(RuntimeError, 'CompressedArchiveBudget'):
                    with tarfile.open(fileobj=writer, mode='w:gz', format=tarfile.USTAR_FORMAT) as archive:
                        member = tarfile.TarInfo('bin/synthetic')
                        member.size = len(payload)
                        archive.addfile(member, io.BytesIO(payload))
                self.assertLessEqual(writer.written, 1024)
                self.assertEqual(writer.written, stream.tell())

    def test_command_progress_preserves_original_failure_and_tree_closure(self):
        # No subprocess/ELF runs: exercise the actual log/receipt boundary with
        # supervisor outcomes, including exit 0 with an unclosed process tree.
        for code, gone, failure in ((0, True, None),
                                    (7, True, None),
                                    (124, True, None),
                                    (130, True, None),
                                    (-15, True, None),
                                    (125, True, 'OriginalWholeBuildDeadline'),
                                    (0, False, None)):
            with self.subTest(code=code, gone=gone), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                build = bundle.Build(root, root, 60)
                env = {key: 'synthetic' for key in ('CARGO_BUILD_JOBS', 'CARGO_INCREMENTAL',
                       'RUSTUP_TOOLCHAIN', 'RZ_PALS_SOURCE_COMMIT', 'CARGO_TARGET_DIR')}
                observed = io.StringIO()

                def supervise(*_args):
                    os.write(1, b'synthetic command output\n')
                    return code, gone, failure

                with patch.object(bundle, 'run_group', supervise), patch.object(bundle.sys, 'stderr', observed):
                    if code == 0 and gone is True and failure is None:
                        self.assertEqual(build.command('synthetic', ['no-executable'], env),
                                         b'synthetic command output\n')
                    elif code == 7:
                        with self.assertRaises(bundle.CommandExitFailure) as caught:
                            build.command('synthetic', ['no-executable'], env)
                        self.assertIs(caught.exception.row, build.commands[0])
                        self.assertEqual(caught.exception.row['exit_code'], 7)
                        self.assertEqual(caught.exception.row['stdout_pin']['bytes'],
                                         len(b'synthetic command output\n'))
                        self.assertEqual(caught.exception.row['stderr_pin'],
                                         {'bytes': 0, 'sha256': hashlib.sha256(b'').hexdigest()})
                    else:
                        with self.assertRaisesRegex(RuntimeError, 'CommandOrOriginalTreeClosureFailed:synthetic'):
                            build.command('synthetic', ['no-executable'], env)
                events = [json.loads(line) for line in observed.getvalue().splitlines()]
                self.assertEqual([row['event'] for row in events],
                                 ['native_bundle_command_start', 'native_bundle_command_terminal'])
                terminal = events[-1]
                self.assertEqual(terminal['status'], 'process_and_tree_closed'
                                 if code == 0 and gone is True and failure is None
                                 else 'command_or_original_tree_failed')
                self.assertEqual(terminal['exit_code'], code)
                self.assertEqual(terminal['original_tree_cleanup_verified'], gone)
                self.assertEqual(terminal['supervisor_error'], failure)
                self.assertEqual(build.commands[0]['exit_code'], code)
                self.assertEqual(build.commands[0]['original_tree_cleanup_verified'], gone)

    def test_closed_target_exit_preserves_empty_stdout_and_builds_next_target(self):
        specs = (('synthetic', 'bin', 'failed', 'feature'),
                 ('synthetic', 'bin', 'continued', 'feature'))
        metadata = {'packages': [{'name': 'synthetic', 'targets': [
            {'name': name, 'kind': ['bin'], 'required-features': ['feature']} for _, _, name, _ in specs]}]}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            build = bundle.Build(root, root, 60)
            env = {key: 'synthetic' for key in ('CARGO_BUILD_JOBS', 'CARGO_INCREMENTAL',
                   'RUSTUP_TOOLCHAIN', 'RZ_PALS_SOURCE_COMMIT')}
            env['CARGO_TARGET_DIR'] = str(root / 'target')
            calls, events, failures, binaries = [], [], [], {}

            def supervise(wrapper, *_args):
                name = wrapper[wrapper.index('--bin') + 1]
                calls.append(name)
                events.append(('dispatch', name))
                if name == 'failed':
                    os.write(2, b'original compiler diagnostic\n')
                    return 7, True, None
                os.write(1, b'synthetic compiler report\n')
                return 0, True, None

            copied = {'private_copy': {'bytes': 13, 'sha256': 'a' * 64}}
            with patch.object(bundle, 'run_group', supervise), \
                    patch.object(bundle, 'cargo_artifact', return_value=(root / 'continued', {'target': 'continued'})) as cargo, \
                    patch.object(bundle, 'copy_cargo_elf', return_value=copied) as copy_elf, \
                    patch.object(bundle.sys, 'stderr', io.StringIO()):
                bundle.build_selected_targets(build, cargo='synthetic-cargo', env=env, metadata=metadata,
                    source=root, private=root / 'private', specs=specs, registered_paths=set(),
                    binaries=binaries, failures=failures,
                    failure_check=lambda: events.append(('inputs_checked',)), warm=None)
            self.assertEqual(calls, ['failed', 'continued'])
            failed_dispatch = events.index(('dispatch', 'failed'))
            continued_dispatch = events.index(('dispatch', 'continued'))
            self.assertIn(('inputs_checked',), events[failed_dispatch + 1:continued_dispatch])
            self.assertEqual(set(binaries), {'continued'})
            self.assertTrue(cargo.called)
            self.assertTrue(all(call.args[4] == specs[1] for call in cargo.call_args_list))
            self.assertTrue(copy_elf.called)
            self.assertTrue(all(call.args[:2] == (root / 'continued', root / 'private' / 'continued')
                                for call in copy_elf.call_args_list))
            self.assertEqual(len(failures), 1)
            self.assertEqual(failures[0]['target'], list(specs[0]))
            self.assertEqual(failures[0]['exit_code'], 7)
            self.assertIs(failures[0]['original_tree_cleanup_verified'], True)
            self.assertIsNone(failures[0]['supervisor_error'])
            self.assertEqual(failures[0]['stdout_pin'], {'bytes': 0, 'sha256': hashlib.sha256(b'').hexdigest()})
            self.assertEqual(failures[0]['stderr_pin'],
                             {'bytes': len(b'original compiler diagnostic\n'),
                              'sha256': hashlib.sha256(b'original compiler diagnostic\n').hexdigest()})
            self.assertEqual((root / 'logs' / 'build-failed.stderr').read_bytes(), b'original compiler diagnostic\n')

    def test_shared_command_failures_do_not_dispatch_later_target(self):
        specs = (('synthetic', 'bin', 'first', 'feature'), ('synthetic', 'bin', 'later', 'feature'))
        for result in ((124, True, None), (130, True, None), (-15, True, None),
                       (0, False, None), (1, True, 'ActualFreeSpaceReserve')):
            with self.subTest(result=result), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                build = bundle.Build(root, root, 60)
                env = {key: 'synthetic' for key in ('CARGO_BUILD_JOBS', 'CARGO_INCREMENTAL',
                       'RUSTUP_TOOLCHAIN', 'RZ_PALS_SOURCE_COMMIT', 'CARGO_TARGET_DIR')}
                failures, binaries, calls = [], {}, []

                def supervise(wrapper, *_args):
                    calls.append(wrapper[wrapper.index('--bin') + 1])
                    os.write(2, b'preserved shared failure\n')
                    return result

                with patch.object(bundle, 'run_group', supervise), \
                        patch.object(bundle, 'cargo_artifact') as cargo, \
                        patch.object(bundle.sys, 'stderr', io.StringIO()):
                    with self.assertRaises(RuntimeError) as caught:
                        bundle.build_selected_targets(build, cargo='synthetic-cargo', env=env, metadata={},
                            source=root, private=root / 'private', specs=specs, registered_paths=set(),
                            binaries=binaries, failures=failures, failure_check=lambda: None, warm=None)
                self.assertNotIsInstance(caught.exception, bundle.CommandExitFailure)
                self.assertEqual(calls, ['first'])
                self.assertEqual((build.commands[0]['exit_code'], build.commands[0]['original_tree_cleanup_verified'],
                                  build.commands[0]['supervisor_error']), result)
                self.assertEqual(failures, [])
                self.assertEqual(binaries, {})
                cargo.assert_not_called()
                self.assertEqual((root / 'logs' / 'build-first.stderr').read_bytes(), b'preserved shared failure\n')

    def test_closed_target_failure_does_not_hide_changed_shared_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            build = bundle.Build(root, root, 60)
            env = {key: 'synthetic' for key in ('CARGO_BUILD_JOBS', 'CARGO_INCREMENTAL',
                   'RUSTUP_TOOLCHAIN', 'RZ_PALS_SOURCE_COMMIT', 'CARGO_TARGET_DIR')}
            failures, binaries = [], {}
            mutation = RuntimeError('synthetic original source/lock mutation')

            def supervise(*_args):
                os.write(2, b'original failed compiler\n')
                return 7, True, None

            def fail_inputs():
                raise mutation

            with patch.object(bundle, 'run_group', supervise), \
                    patch.object(bundle, 'cargo_artifact') as cargo, \
                    patch.object(bundle.sys, 'stderr', io.StringIO()):
                with self.assertRaises(RuntimeError) as caught:
                    bundle.build_selected_targets(build, cargo='synthetic-cargo', env=env, metadata={},
                        source=root, private=root / 'private',
                        specs=(('synthetic', 'bin', 'first', 'feature'), ('synthetic', 'bin', 'later', 'feature')),
                        registered_paths=set(), binaries=binaries, failures=failures,
                        failure_check=fail_inputs, warm=None)
            self.assertIs(caught.exception, mutation)
            self.assertEqual(len(build.commands), 1)
            self.assertEqual(failures[0]['exit_code'], 7)
            self.assertEqual(binaries, {})
            cargo.assert_not_called()

    def test_partial_archive_contains_only_verified_successes_under_separate_name(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact, private = root / 'artifact', root / 'private'
            artifact.mkdir()
            private.mkdir()
            raw = b'synthetic retained bytes, never executed'
            (private / 'successful').write_bytes(raw)
            binaries = {'successful': {'pin': bundle.file_pin(private / 'successful')}}
            filename = 'partial-native-validation-binaries.tar.gz'
            budgets = []
            def observe_budget(amount):
                budgets.append((amount, (artifact / filename).exists()))
            build = SimpleNamespace(budget=observe_budget)
            path, pin = bundle.archive_selected_binaries(build, artifact, private, binaries,
                                                       filename)
            self.assertTrue(any(amount >= len(raw) and not already_created
                                for amount, already_created in budgets))
            self.assertEqual(pin, bundle.file_pin(path))
            self.assertFalse((artifact / 'native-validation-binaries.tar.gz').exists())
            with tarfile.open(path, 'r:gz') as archive:
                self.assertEqual(archive.getnames(), ['bin/successful'])
                self.assertEqual(archive.extractfile('bin/successful').read(), raw)
            denied = artifact / 'denied-native-validation-binaries.tar.gz'
            budget_error = RuntimeError('synthetic archive budget rejected')
            def reject_budget(_amount):
                raise budget_error
            with self.assertRaises(RuntimeError) as caught:
                bundle.archive_selected_binaries(SimpleNamespace(budget=reject_budget),
                    artifact, private, binaries, denied.name)
            self.assertIs(caught.exception, budget_error)
            self.assertFalse(denied.exists())
            self.assertEqual(bundle.file_pin(path), pin)

    def test_cross_source_creation_time_preserves_all_named_file_identity_fields(self):
        fields = dict(st_dev=7, st_ino=(1 << 100) + 11, st_mode=stat.S_IFREG | 0o666,
                      st_nlink=1, st_size=40, st_mtime_ns=1000,
                      st_ctime_ns=2000, st_birthtime_ns=900)
        opened = SimpleNamespace(**fields)
        named = SimpleNamespace(**{**fields, 'st_ctime_ns': fields['st_birthtime_ns']})
        self.assertNotEqual(bundle.identity(opened), bundle.identity(named))
        self.assertEqual(bundle.identity(opened, creation_time=True),
                         bundle.identity(named, creation_time=True))
        with patch.object(bundle.os, 'name', 'nt'):
            self.assertTrue(bundle.same_file_identity(opened, opened, named))
        with patch.object(bundle.os, 'name', 'posix'):
            self.assertFalse(bundle.same_file_identity(opened, opened, named))
        for field in ('st_dev', 'st_ino', 'st_mode', 'st_nlink', 'st_size',
                      'st_mtime_ns', 'st_birthtime_ns'):
            with self.subTest(field=field):
                changed = SimpleNamespace(**{**vars(named), field: getattr(named, field) + 1})
                self.assertNotEqual(bundle.identity(opened, creation_time=True),
                                    bundle.identity(changed, creation_time=True))

    def test_same_file_identity_keeps_original_fd_change_time_check(self):
        fields = dict(st_dev=7, st_ino=11, st_mode=stat.S_IFREG | 0o600,
                      st_nlink=1, st_size=40, st_mtime_ns=1000,
                      st_ctime_ns=2000, st_birthtime_ns=900)
        opened = SimpleNamespace(**fields)
        changed = SimpleNamespace(**{**fields, 'st_ctime_ns': 2001})
        self.assertTrue(bundle.same_file_identity(opened, opened, opened))
        self.assertFalse(bundle.same_file_identity(opened, changed, opened))

    def test_compile_only_cleanup_preserves_published_bundle_and_unique_results(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact = root / 'artifact'
            artifact.mkdir()
            archive = artifact / 'native-validation-binaries.tar.gz'
            archive.write_bytes(b'synthetic already-verified archive')
            (artifact / 'only-game.pgn').write_bytes(b'unique-result')
            (root / 'unclassified-other-data').mkdir()
            for name in ('target-native', 'private-elf', 'cargo-home', 'logs', 'tmp'):
                (root / name).mkdir()
                (root / name / 'owned-copy').write_bytes(b'regenerable')
            receipt = {'status': 'SAME_HEAD_RELEASE_ELFS_AND_STATIC_SOURCE_METADATA_VERIFIED',
                       'commands': [{'original_tree_cleanup_verified': True}],
                       'binary_bundle': bundle.file_pin(archive)}
            journal = bundle.cleanup_compiler_outputs(root, artifact, receipt)
            self.assertTrue(journal['complete'])
            self.assertTrue((root / 'logs' / 'owned-copy').exists())
            self.assertFalse((root / 'private-elf').exists())
            self.assertFalse((root / 'target-native').exists())
            self.assertEqual(archive.read_bytes(), b'synthetic already-verified archive')
            self.assertEqual((artifact / 'only-game.pgn').read_bytes(), b'unique-result')
            self.assertTrue((root / 'unclassified-other-data').exists())

    def test_uncertain_tree_or_changed_bundle_prevents_all_compiler_cleanup(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact = root / 'artifact'
            artifact.mkdir()
            archive = artifact / 'native-validation-binaries.tar.gz'
            archive.write_bytes(b'retained')
            (root / 'target-native').mkdir()
            receipt = {'status': 'SAME_HEAD_RELEASE_ELFS_AND_STATIC_SOURCE_METADATA_VERIFIED',
                       'commands': [{'original_tree_cleanup_verified': False}],
                       'binary_bundle': bundle.file_pin(archive)}
            with self.assertRaisesRegex(RuntimeError, 'VerifiedBundleAndTreeClosure'):
                bundle.cleanup_compiler_outputs(root, artifact, receipt)
            receipt['commands'][0]['original_tree_cleanup_verified'] = True
            archive.write_bytes(b'changed')
            with self.assertRaisesRegex(RuntimeError, 'VerifiedBundleAndTreeClosure'):
                bundle.cleanup_compiler_outputs(root, artifact, receipt)
            self.assertTrue((root / 'target-native').exists())
            self.assertNotIn('regenerable_cleanup', receipt)

    @unittest.skipUnless(sys.platform == 'linux', 'Linux-only bundle FD/ctime custody; no Windows emulation')
    def test_cargo_hardlink_is_copied_into_single_link_custody(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, alias, private = root / 'cargo', root / 'deps-alias', root / 'private'
            source.write_bytes(b'\x7fELF_HEADER_ONLY_SYNTHETIC_NOT_EXECUTABLE')
            os.link(source, alias)
            with self.assertRaisesRegex(RuntimeError, 'ArtifactTypeOrSize'):
                bundle.file_pin(source)
            budgets = []
            copied = bundle.copy_cargo_elf(source, private, budgets.append)
            self.assertEqual(copied['original_cargo']['nlink'], 2)
            self.assertEqual(private.stat().st_nlink, 1)
            self.assertEqual(bundle.file_pin(private), {k: copied['original_cargo'][k] for k in ('bytes', 'sha256')})
            self.assertEqual(budgets[0], source.stat().st_size)
            with self.assertRaisesRegex(RuntimeError, 'CargoElfCopyPlacement'):
                bundle.copy_cargo_elf(source, private, budgets.append)

    @unittest.skipUnless(sys.platform == 'linux', 'Linux-only bundle FD/ctime custody; no Windows emulation')
    def test_non_elf_and_copy_admission_failure_are_not_published(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, private = root / 'source', root / 'private'
            source.write_bytes(b'NOT_ELF')
            with self.assertRaisesRegex(RuntimeError, 'CargoOutputNotElf'):
                bundle.copy_cargo_elf(source, private, lambda _: None)
            self.assertFalse(private.exists())
            source.write_bytes(b'\x7fELFfixture')
            def reject(_):
                raise RuntimeError('BudgetDenied')
            with self.assertRaisesRegex(RuntimeError, 'BudgetDenied'):
                bundle.copy_cargo_elf(source, private, reject)
            self.assertFalse(private.exists())

    def test_explicit_source_paths_reject_secrets_heavy_and_traversal(self):
        for path in ('crates/x/.env', 'scripts/credentials.json', 'crates/x/models/w.bin',
                     'scripts/../x.py', '/scripts/x.py', 'scripts\\x.py'):
            with self.subTest(path=path), self.assertRaisesRegex(RuntimeError, 'ForbiddenSourcePath'):
                bundle.allowed_source(path)
        self.assertFalse(bundle.allowed_source('docs/research/PALS-IMPLEMENTATION.md'))
        self.assertTrue(bundle.allowed_source('scripts/ci_native_bundle.py'))

    @unittest.skipUnless(sys.platform == 'linux', 'Linux-only bundle FD/ctime custody; no Windows emulation')
    def test_source_bytes_must_match_exact_tracked_git_blob(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary)
            tree = []
            for logical in sorted(bundle.SINGLETONS | {bundle.CONTROL}):
                path = source / logical
                path.parent.mkdir(parents=True, exist_ok=True)
                raw = b'' if logical == 'LICENSE' else b'SYNTHETIC_PUBLIC_SOURCE\n'
                path.write_bytes(raw)
                oid = hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()
                tree.append(b'100644 blob ' + oid.encode() + b'\t' + logical.encode() + b'\0')
            roster = b''.join(tree)
            rows, controls = bundle.source_inventory(source, roster)
            self.assertEqual(len(rows), len(bundle.SINGLETONS))
            self.assertEqual(set(controls), {bundle.CONTROL})
            (source / 'Cargo.lock').write_bytes(b'CHANGED')
            with self.assertRaisesRegex(RuntimeError, 'CheckoutBytesDifferFromGitBlob'):
                bundle.source_inventory(source, roster)

    def test_component_framing_is_order_independent_and_content_sensitive(self):
        names = list(bundle.CORE5) + ['component' + str(n) for n in range(20)]
        rows = [{'path': 'crates/b.rs', 'bytes': 2, 'sha256': 'ab' * 32},
                {'path': 'crates/a.rs', 'bytes': 1, 'sha256': 'cd' * 32}]
        catalog = {'domain': 'rz-pals-source-inventory-v5/1',
                   'algorithm': 'sha256-domain-u64le;sorted-path-u64le-utf8;size-u64le;sha256-raw32',
                   'components': {name: [row['path'] for row in rows] for name in names}}
        initial = bundle.component_inventory(catalog, rows)
        reversed_catalog = copy.deepcopy(catalog)
        for paths in reversed_catalog['components'].values():
            paths.reverse()
        self.assertEqual(initial, bundle.component_inventory(reversed_catalog, list(reversed(rows))))
        changed = copy.deepcopy(rows)
        changed[0]['bytes'] += 1
        self.assertNotEqual(initial, bundle.component_inventory(catalog, changed))
        catalog['components'][names[0]].append(rows[0]['path'])
        with self.assertRaisesRegex(RuntimeError, 'CatalogSourceMissing'):
            bundle.component_inventory(catalog, rows)

    def test_warm_requires_both_registered_source_and_explicit_exact_feature_target(self):
        name = bundle.WARM[2]
        features = bundle.WARM[3].split(',')
        logical = 'crates/rz-uci/examples/' + name + '.rs'
        manifest = {'example': [{'name': name, 'required-features': features}],
                    'features': {feature: [] for feature in features}}
        spec, status = bundle.warm_spec(manifest, {logical})
        self.assertEqual(spec, bundle.WARM)
        self.assertEqual(status, 'SOURCE_PRESENT_REQUIRES_ACTUAL_BUILD')
        self.assertIsNone(bundle.warm_spec(manifest, set())[0])
        self.assertIsNone(bundle.warm_spec({'features': manifest['features']}, {logical})[0])
        manifest['example'][0]['required-features'] = ['onnx-cpu']
        self.assertEqual(bundle.warm_spec(manifest, {logical})[1], 'NOT_READY_REQUIRED_FEATURE_OR_SOURCE_CONTRACT')

    def test_actual_cargo_artifact_must_join_source_kind_features_and_shared_target_custody(self):
        source = Path('/synthetic-source')
        target_root = Path('/synthetic-target-native')
        spec = next(row for row in bundle.SPECS if row[2] == 'pals_continuation_diagnostic')
        logical = 'crates/rz-uci/examples/pals_continuation_diagnostic.rs'
        target = {'name': spec[2], 'kind': ['example'], 'crate_types': ['bin'],
                  'src_path': str(source / logical), 'edition': '2024'}
        metadata = {'packages': [{'name': 'rz-uci', 'id': 'SYNTHETIC_PACKAGE',
                                 'manifest_path': str(source / 'crates/rz-uci/Cargo.toml'), 'targets': [target]}]}
        artifact = {'reason': 'compiler-artifact', 'package_id': 'SYNTHETIC_PACKAGE',
                    'target': copy.deepcopy(target), 'executable': str(target_root / 'release/examples' / spec[2]),
                    'profile': {'test': False},
                    'features': ['onnx-cpu', 'pals-adaptive', 'pals-adaptive-diagnostics', 'search-work-receipts']}
        messages = [artifact, {'reason': 'build-finished', 'success': True}]
        def check(value, paths=None):
            raw = b'\n'.join(json.dumps(row).encode() for row in value)
            return bundle.cargo_artifact(raw, metadata, source, target_root, spec,
                                         {logical} if paths is None else paths)
        self.assertEqual(check(messages)[0], Path(artifact['executable']))
        for label, mutate in (
                ('duplicate', lambda v: v.insert(0, copy.deepcopy(v[0]))),
                ('failed-finish', lambda v: v[1].update(success=False)),
                ('wrong-slot', lambda v: v[0].update(executable='/wrong-slot/diagnostic')),
                ('wrong-package', lambda v: v[0].update(package_id='OTHER')),
                ('test-profile', lambda v: v[0]['profile'].update(test=True)),
                ('default', lambda v: v[0]['features'].append('default')),
                ('warm-mixed', lambda v: v[0]['features'].append('experimental-io-binding'))):
            changed = copy.deepcopy(messages)
            mutate(changed)
            with self.subTest(label=label), self.assertRaises(RuntimeError):
                check(changed)
        with self.assertRaisesRegex(RuntimeError, 'ExactRegisteredCargoTarget'):
            check(messages, set())

    def test_functional_bundle_keeps_legacy_targets_and_requires_exact_six_tools(self):
        legacy = (
            ('rz-arena', 'bin', 'rz-pals-g0-bootstrap-numerical', 'pals-collection-onnx,native-cuda'),
            ('rz-arena', 'bin', 'rz-pals-g0-collection-deployment-numerical', 'pals-collection-onnx,native-cuda'),
            ('rz-eval', 'example', 'virtual_review_numeric_capture', 'onnx,contracts'),
            ('rz-uci', 'bin', 'rz-pals-adaptive', 'pals-adaptive,onnx-cuda'),
            ('rz-uci', 'example', 'pals_rules_descriptor', 'pals-adaptive,onnx-cuda'),
            ('rz-uci', 'bin', 'rz-rovent', 'pals-adaptive,onnx-cuda'),
            ('rz-arena', 'bin', 'rz-rovent-trainer', 'pals-collection-onnx,native-cuda'),
            ('rz-uci', 'example', 'pals_continuation_diagnostic', 'pals-adaptive-diagnostics'),
        )
        required = {'rz-rovent-native', 'rz-rovent-collect', 'rz-rovent-utility-trainer',
                    'rovent_startup_metadata', 'virtual_review_numeric_capture', 'rz-experiments'}
        legacy_names = {row[2] for row in legacy}
        self.assertEqual(tuple(row for row in bundle.SPECS if row[2] in legacy_names), legacy)
        self.assertEqual(len(bundle.SPECS), 13)
        native = ('rz-arena', 'bin', 'rz-rovent-native',
                  'pals-collection-onnx,native-cuda,native-validation-diagnostics')
        self.assertIn(native, bundle.SPECS)
        self.assertIn(native, bundle.FUNCTIONAL_SPECS)
        arena = bundle.tomllib.loads((Path(__file__).resolve().parents[2]
            / 'crates/rz-arena/Cargo.toml').read_text(encoding='utf-8'))
        enabled, pending = set(), ['native-validation-diagnostics']
        while pending:
            feature = pending.pop()
            if feature not in enabled:
                enabled.add(feature)
                pending.extend(arena['features'].get(feature, []))
        self.assertTrue({'rz-uci/pals-adaptive-diagnostics',
                         'rz-uci/native-validation-diagnostics'} <= enabled)
        self.assertEqual(arena['features'].get('default', []), [])
        self.assertEqual(next(t for t in arena['bin'] if t['name'] == 'rz-rovent-native')
                         ['required-features'], ['pals-collection-onnx'])
        regular = tuple((*row[:3], 'pals-collection-onnx,native-cuda')
                        if row[2] == 'rz-rovent-native' else row for row in bundle.SPECS)
        with self.assertRaises(RuntimeError):
            bundle.validate_functional_specs(regular)
        self.assertEqual({row[2] for row in bundle.FUNCTIONAL_SPECS}, required)
        self.assertEqual(len(bundle.FUNCTIONAL_SPECS), 6)
        self.assertEqual(sum(row[2] == 'virtual_review_numeric_capture' for row in bundle.SPECS), 1)
        self.assertEqual(bundle.WARM, ('rz-uci', 'example', 'virtual_review_warm_numeric_capture',
                                      'pals-adaptive,onnx-cpu,experimental-io-binding'))
        bundle.validate_functional_specs(bundle.SPECS)
        bundle.validate_functional_specs(bundle.SPECS + (bundle.WARM,))
        for original, replacement in (('rz-rovent-native', 'rz-rovent'),
                                      ('rz-rovent-utility-trainer', 'rz-rovent-trainer')):
            changed = tuple((row[0], row[1], replacement, row[3]) if row[2] == original else row
                            for row in bundle.SPECS)
            with self.subTest(alias=original), self.assertRaises(RuntimeError):
                bundle.validate_functional_specs(changed)
        for name in required:
            for index, replacement in ((1, 'example' if name.startswith('rz-') else 'bin'),
                                       (3, 'unregistered-features')):
                changed = tuple(tuple(replacement if cursor == index else part
                                      for cursor, part in enumerate(row)) if row[2] == name else row
                                for row in bundle.SPECS)
                with self.subTest(name=name, field=index), self.assertRaises(RuntimeError):
                    bundle.validate_functional_specs(changed)
        with self.assertRaises(RuntimeError):
            bundle.validate_functional_specs(bundle.SPECS[:-1])

    def test_each_functional_tool_uses_exact_cargo_source_kind_and_feature_admission(self):
        source = Path('/synthetic-source')
        target_root = Path('/synthetic-target-functional')
        paths = {
            'rz-rovent-native': 'crates/rz-arena/src/bin/rovent-native.rs',
            'rz-rovent-collect': 'crates/rz-arena/src/bin/rovent-collect.rs',
            'rz-rovent-utility-trainer': 'crates/rz-arena/src/bin/rovent-utility-trainer.rs',
            'rovent_startup_metadata': 'crates/rz-uci/examples/rovent_startup_metadata.rs',
            'virtual_review_numeric_capture': 'crates/rz-eval/examples/virtual_review_numeric_capture.rs',
            'rz-experiments': 'crates/rz-experiments/src/main.rs',
        }
        for spec in bundle.FUNCTIONAL_SPECS:
            package, kind, name, features = spec
            logical = paths[name]
            target = {'name': name, 'kind': [kind], 'crate_types': ['bin'],
                      'src_path': str(source / logical), 'edition': '2024'}
            metadata = {'packages': [{'name': package, 'id': 'SYNTHETIC_PACKAGE',
                                     'manifest_path': str(source / 'crates' / package / 'Cargo.toml'),
                                     'targets': [target]}]}
            output = target_root / 'release' / ('examples' if kind == 'example' else '') / name
            artifact = {'reason': 'compiler-artifact', 'package_id': 'SYNTHETIC_PACKAGE',
                        'target': copy.deepcopy(target), 'executable': str(output),
                        'profile': {'test': False}, 'features': features.split(',')}
            messages = [artifact, {'reason': 'build-finished', 'success': True}]
            def check(value, actual_metadata=None, registered=None):
                raw = b'\n'.join(json.dumps(row).encode() for row in value)
                return bundle.cargo_artifact(raw, metadata if actual_metadata is None else actual_metadata,
                    source, target_root, spec, {logical} if registered is None else registered)
            with self.subTest(name=name, case='original'):
                self.assertEqual(check(messages)[0], output)
            for label, mutate in (
                    ('wrong-kind', lambda rows: rows[0]['target'].update(kind=['lib'])),
                    ('missing-feature', lambda rows: rows[0]['features'].pop()),
                    ('wrong-package', lambda rows: rows[0].update(package_id='OTHER')),
                    ('wrong-target-name', lambda rows: rows[0]['target'].update(name='different-tool')),
                    ('wrong-slot', lambda rows: rows[0].update(executable='/other-target/tool'))):
                changed = copy.deepcopy(messages)
                mutate(changed)
                with self.subTest(name=name, case=label), self.assertRaises(RuntimeError):
                    check(changed)
            with self.subTest(name=name, case='unregistered-source'), self.assertRaises(RuntimeError):
                check(messages, registered=set())
            foreign_metadata, foreign_messages = copy.deepcopy(metadata), copy.deepcopy(messages)
            foreign = str(Path('/outside-source/unregistered.rs'))
            foreign_metadata['packages'][0]['targets'][0]['src_path'] = foreign
            foreign_messages[0]['target']['src_path'] = foreign
            with self.subTest(name=name, case='outside-source'), self.assertRaises(RuntimeError):
                check(foreign_messages, actual_metadata=foreign_metadata)

    def test_strict_metadata_rejects_duplicate_keys_and_nonfinite_values(self):
        for raw in (b'{"x":1,"x":2}', b'{"x":NaN}'):
            with self.assertRaises(RuntimeError):
                bundle.strict_json(raw)

    def test_rovent_compile_metadata_cannot_claim_native_execution_or_replace_registered_sources(self):
        components = {'uci': 'ab' * 32, 'rules': 'cd' * 32}
        head = '19' * 20
        value = {'domain': 'rz-rovent-build-capabilities/1', 'source_commit': head,
                 'source_components': dict(components, rovent_bootstrap='ef' * 32,
                                           rovent_process_boundary='42' * 32),
                 'target_os': 'linux', 'target_arch': 'x86_64',
                 'onnx_cpu_compiled': True, 'onnx_cuda_compiled': True,
                 'native_runtime_loaded': False, 'native_model_loaded': False,
                 'neural_execution_observed': False}
        bundle.validate_rovent_build_metadata(value, head, components)
        for field in ('native_runtime_loaded', 'native_model_loaded', 'neural_execution_observed'):
            wrong = copy.deepcopy(value)
            wrong[field] = True
            with self.subTest(field=field), self.assertRaisesRegex(RuntimeError, 'StaticRoventBuildMetadata'):
                bundle.validate_rovent_build_metadata(wrong, head, components)
        for name in ('uci', 'rules', 'rovent_bootstrap', 'rovent_process_boundary'):
            wrong = copy.deepcopy(value)
            wrong['source_components'][name] = 'NOT_A_REGISTERED_DIGEST'
            with self.subTest(component=name), self.assertRaisesRegex(RuntimeError, 'StaticRoventRegisteredSourceMismatch'):
                bundle.validate_rovent_build_metadata(wrong, head, components)

    def test_registry_public_methods_remain_static_and_source_bound_elsewhere(self):
        registry = {'schema': 'rz-pals-report-model-registry-query/1', 'model_profile': 'virtual_review_v1',
                    'registry_methods': ['PalsModelProfile::VirtualReviewV1.registered_semantics_sha256',
                                         'pals_model::virtual_review::registered_encoding_sha256'],
                    'model_semantic_sha256': 'ab' * 32, 'encoding_sha256': 'cd' * 32,
                    'numerical_authority_issued': False, 'model_initialized': False,
                    'ort_loaded': False, 'neural_executed': False, 'training_executed': False}
        bundle.validate_registry(registry)
        for key in ('numerical_authority_issued', 'model_initialized', 'ort_loaded', 'neural_executed', 'training_executed'):
            changed = copy.deepcopy(registry)
            changed[key] = True
            with self.subTest(key=key), self.assertRaisesRegex(RuntimeError, 'StaticRegistryGetterOutput'):
                bundle.validate_registry(changed)
        registry['model_semantic_sha256'] = 'not-an-actual-registry-digest'
        with self.assertRaisesRegex(RuntimeError, 'StaticRegistryGetterOutput'):
            bundle.validate_registry(registry)


class RuntimeLoaderBundleTests(unittest.TestCase):
    """Single-target CI production path with a fake supervisor; no Cargo runs."""

    logical = 'crates/rz-eval/examples/runtime_loader_check.rs'
    spec = ('rz-eval', 'example', 'runtime_loader_check', 'onnx')

    def metadata(self, source, target_root, required=('onnx',)):
        target = {'name': self.spec[2], 'kind': ['example'], 'crate_types': ['bin'],
                  'src_path': str(source / self.logical), 'edition': '2024',
                  'required-features': list(required)}
        metadata = {'workspace_root': str(source), 'packages': [{
            'name': 'rz-eval', 'id': 'SYNTHETIC_LOADER_PACKAGE',
            'manifest_path': str(source / 'crates/rz-eval/Cargo.toml'), 'targets': [target]}]}
        output = target_root / 'release/examples/runtime_loader_check'
        messages = [{'reason': 'compiler-artifact', 'package_id': 'SYNTHETIC_LOADER_PACKAGE',
                     'target': copy.deepcopy(target), 'executable': str(output),
                     'profile': {'test': False}, 'features': ['onnx']},
                    {'reason': 'build-finished', 'success': True}]
        return metadata, messages, output

    def test_loader_cargo_admission_uses_only_its_source_kind_and_features(self):
        self.assertEqual(bundle.RUNTIME_LOADER_SPEC, self.spec)
        source, target_root = Path('/synthetic-loader-source'), Path('/synthetic-loader-target')
        metadata, messages, output = self.metadata(source, target_root)

        def admit(value, registered=None):
            raw = b'\n'.join(json.dumps(row).encode() for row in value)
            return bundle.cargo_artifact(raw, metadata, source, target_root, self.spec,
                                        {self.logical} if registered is None else registered)

        self.assertEqual(admit(messages)[0], output)
        for label, change in (
                ('kind', lambda rows: rows[0]['target'].update(kind=['bin'])),
                ('source', lambda rows: rows[0]['target'].update(src_path=str(source / 'other.rs'))),
                ('feature', lambda rows: rows[0].update(features=[])),
                ('default', lambda rows: rows[0]['features'].append('default')),
                ('release-path', lambda rows: rows[0].update(executable=str(target_root / 'other')))):
            altered = copy.deepcopy(messages)
            change(altered)
            with self.subTest(change=label), self.assertRaises(RuntimeError):
                admit(altered)
        with self.assertRaises(RuntimeError):
            admit(messages, registered=set())

    def run_fixture(self, *, result=(0, True, None), required=('onnx',), profile='runtime-loader-only'):
        temporary = tempfile.TemporaryDirectory(prefix='ci-runtime-loader-fixture-')
        self.addCleanup(temporary.cleanup)
        base = Path(temporary.name).resolve()
        source, root = base / 'source', base / 'RoveZero' / 'owned'
        source.mkdir()
        source_file = source / self.logical
        source_file.parent.mkdir(parents=True)
        source_file.write_bytes(b'// synthetic source; not compiled\n')
        (source / 'Cargo.lock').write_bytes(b'synthetic lock\n')
        (source / 'rust-toolchain.toml').write_text('[toolchain]\nchannel="1.96.0"\n', encoding='utf-8')
        rows = [{'path': self.logical, **bundle.file_pin(source_file)}]
        controls = {'synthetic_control': {'bytes': 1, 'sha256': 'a' * 64}}
        metadata, messages, output = self.metadata(source, root / 'target-native', required)
        head, dispatched = '19' * 20, []

        def supervise(wrapper, cwd, env, seconds, check):
            argv = wrapper[wrapper.index('--internal-command') + 1:]
            dispatched.append(list(argv))
            if argv[1:3] == ['rev-parse', 'HEAD']:
                raw = (head + '\n').encode()
            elif argv[1:2] == ['ls-tree']:
                raw = b'synthetic roster\n'
            elif argv[1:2] == ['--version']:
                raw = (Path(argv[0]).name + ' 1.96.0 synthetic\n').encode()
            elif argv[1:2] == ['metadata']:
                raw = json.dumps(metadata).encode()
            elif argv[1:2] == ['build']:
                self.assertEqual(argv[argv.index('--example') + 1], self.spec[2])
                self.assertEqual(argv[argv.index('--features') + 1], 'onnx')
                if result != (0, True, None):
                    os.write(2, b'synthetic original loader compiler failure\n')
                    return result
                output.parent.mkdir(parents=True)
                output.write_bytes(b'\x7fELF-synthetic-fixture-never-executed')
                raw = b'\n'.join(json.dumps(row).encode() for row in messages)
            else:
                raise AssertionError('unrelated target/getter dispatched')
            os.write(1, raw)
            return 0, True, None

        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.dict(os.environ, {'RUNNER_TEMP': str(base), 'GITHUB_OUTPUT': ''}))
            stack.enter_context(patch.object(bundle.sys, 'platform', 'linux'))
            stack.enter_context(patch.object(bundle.platform, 'machine', return_value='x86_64'))
            stack.enter_context(patch.object(bundle.platform, 'libc_ver', return_value=('synthetic', '0')))
            stack.enter_context(patch.object(bundle.os, 'sched_getaffinity', return_value={0, 1}, create=True))
            stack.enter_context(patch.object(bundle.os, 'sched_setaffinity', create=True))
            stack.enter_context(patch.object(bundle.signal, 'signal'))
            stack.enter_context(patch.object(bundle.shutil, 'which', side_effect=lambda name: '/synthetic/' + name))
            stack.enter_context(patch.object(bundle, 'source_inventory', return_value=(rows, controls)))
            stack.enter_context(patch.object(bundle, 'run_group', side_effect=supervise))
            for name in ('component_inventory', 'warm_spec', 'validate_functional_specs',
                         'validate_registry', 'validate_rovent_build_metadata'):
                stack.enter_context(patch.object(bundle, name,
                    side_effect=AssertionError('unrelated full-bundle admission requested')))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            stack.enter_context(patch.object(bundle.sys, 'stderr', io.StringIO()))
            code = bundle.run(source, root, head, 60, **({} if profile is None else {'profile': profile}))
        receipt = json.loads((root / 'artifact/build-receipt.json').read_bytes())
        return code, root, receipt, dispatched

    def test_loader_profile_builds_one_target_without_other_binaries_or_getters(self):
        code, root, receipt, dispatched = self.run_fixture()
        self.assertEqual(code, 0)
        self.assertEqual(receipt['schema'], 'rovezero.ci-runtime-loader-diagnostic/1')
        self.assertEqual(receipt['status'], bundle.RUNTIME_LOADER_SUCCESS)
        self.assertEqual(receipt['requested_targets'], [list(self.spec)])
        self.assertEqual(set(receipt['binaries']), {self.spec[2]})
        self.assertEqual(receipt['binaries'][self.spec[2]]['cargo_metadata_required_features'], ['onnx'])
        self.assertEqual(len([row for row in dispatched if row[1:2] == ['build']]), 1)
        for flag in ('diagnostic_binary_executed', 'runtime_loaded', 'model_loaded',
                     'session_created', 'full_native_bundle_admitted', 'neural_execution',
                     'gpu_execution', 'training_execution'):
            self.assertIs(receipt[flag], False)
        artifact = root / 'artifact'
        archive = artifact / 'runtime-loader-diagnostic-binaries.tar.gz'
        self.assertEqual(receipt['binary_bundle'], bundle.file_pin(archive))
        self.assertFalse((artifact / 'native-validation-binaries.tar.gz').exists())
        with tarfile.open(archive, 'r:gz') as packed:
            self.assertEqual(packed.getnames(), ['bin/runtime_loader_check'])
        for name in ('compiled-source-inventory.json', 'native-source-pins.json',
                     'static-model-registry.json', 'rules-descriptor.json'):
            self.assertFalse((artifact / name).exists())
        self.assertEqual(json.loads((artifact / 'source-file-pins.json').read_bytes())['schema'],
                         'rovezero.ci-runtime-loader-source-pins/1')
        self.assertEqual(json.loads((artifact / 'git-source-provenance.json').read_bytes())['schema'],
                         'rovezero.ci-runtime-loader-source-provenance/1')

    def test_loader_compiler_failure_preserves_logs_without_admitting_an_archive(self):
        code, root, receipt, _ = self.run_fixture(result=(7, True, None))
        self.assertEqual(code, 1)
        self.assertEqual(receipt['status'], 'FAILED_NOT_ADMITTED')
        self.assertEqual(receipt['binaries'], {})
        self.assertEqual(len(receipt['target_failures']), 1)
        self.assertEqual(receipt['target_failures'][0]['target'], list(self.spec))
        self.assertEqual(receipt['target_failures'][0]['exit_code'], 7)
        self.assertEqual(receipt['target_failures'][0]['stdout_pin'],
                         {'bytes': 0, 'sha256': hashlib.sha256(b'').hexdigest()})
        artifact = root / 'artifact'
        self.assertEqual((artifact / 'command-logs/build-runtime_loader_check.stderr').read_bytes(),
                         b'synthetic original loader compiler failure\n')
        self.assertNotIn('binary_bundle', receipt)
        self.assertNotIn('partial_binary_bundle', receipt)
        self.assertEqual(list(artifact.glob('*binaries.tar.gz')), [])

    def test_loader_required_feature_mismatch_preserves_failure_not_partial_admission(self):
        code, root, receipt, _ = self.run_fixture(required=())
        self.assertEqual(code, 1)
        self.assertEqual(receipt['status'], 'FAILED_NOT_ADMITTED')
        self.assertEqual(receipt['binaries'][self.spec[2]]['cargo_metadata_required_features'], [])
        self.assertNotIn('binary_bundle', receipt)
        self.assertNotIn('partial_binary_bundle', receipt)
        self.assertEqual(list((root / 'artifact').glob('*binaries.tar.gz')), [])
        self.assertIs(receipt['full_native_bundle_admitted'], False)

    def test_omitted_profile_remains_the_existing_native_bundle_path(self):
        code, _root, receipt, dispatched = self.run_fixture(profile=None)
        self.assertEqual(code, 1)  # The synthetic source deliberately has no full catalog.
        self.assertEqual(receipt['schema'], 'rovezero.ci-cpu-validation-bundle/1')
        self.assertNotIn('diagnostic_profile', receipt)
        self.assertFalse(any(row[1:2] == ['build'] for row in dispatched))
        self.assertNotIn(self.spec, bundle.SPECS)
        bundle.validate_functional_specs(bundle.SPECS)


if __name__ == '__main__':
    unittest.main()
