"""Pure metadata/FD custody cases; no Cargo, target ELF or neural execution."""
from __future__ import annotations

import copy
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
        self.assertEqual(set(arena['features']['native-validation-diagnostics']),
                         {'rz-uci/pals-adaptive-diagnostics',
                          'rz-uci/native-validation-diagnostics'})
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


if __name__ == '__main__':
    unittest.main()
