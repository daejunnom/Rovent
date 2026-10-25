#!/usr/bin/env python3
"""Build a bounded same-HEAD validation bundle; never run a neural workload.

CPU compilation and explicit static metadata getters only. Cargo ELF
FD/copy custody and component framing follow the reviewed report payload;
process-tree custody reuses managed_process.run_group. No Cloud entry, model,
runtime DSO, weight, credential discovery, cache adoption, or retry is present.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import selectors
import shutil
import signal
import stat
import subprocess
import sys
import tarfile
import time
import tomllib

# Also works for the isolated -I -S child wrapper: only these tracked sibling
# supervisors are added, never a caller PYTHONPATH or user package directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from managed_process import run_group
from managed_storage import checked, tree_size

ROOT_CAP = 12 << 30
ARTIFACT_CAP = 512 << 20
ELF_CAP = 256 << 20
ELF_PAYLOAD_CAP = 768 << 20
LOG_CAP = 8 << 20
SOURCE_CAP = 128 << 20
# Release compilation can exhaust the old 45-minute window after several
# feature profiles have already completed. Keep an explicit finite build-only
# ceiling; the requested window is recorded in the original resource receipt.
MAX_BUILD_SECONDS = 10800
SINGLETONS = {'Cargo.toml', 'Cargo.lock', 'LICENSE', 'rust-toolchain.toml',
              'experiments/model-research/pals/pyproject.toml',
              'experiments/baselines/fastchess-clock-ponder-v2.patch'}
PREFIXES = ('crates/', 'scripts/', 'experiments/model-research/pals/src/')
CONTROL = '.github/workflows/contracts.yml'
CORE5 = ('model_native', 'model_encoding', 'nn_runtime', 'model_reference', 'model_export')
# Explicit compile capabilities preserve the original native report profile;
# compiling CUDA support does not load a provider or execute GPU/NN work.
SPECS = (
    ('rz-arena', 'bin', 'rz-pals-g0-bootstrap-numerical', 'pals-collection-onnx,native-cuda'),
    ('rz-arena', 'bin', 'rz-pals-g0-collection-deployment-numerical', 'pals-collection-onnx,native-cuda'),
    ('rz-eval', 'example', 'virtual_review_numeric_capture', 'onnx,contracts'),
    ('rz-uci', 'bin', 'rz-pals-adaptive', 'pals-adaptive,onnx-cuda'),
    ('rz-uci', 'example', 'pals_rules_descriptor', 'pals-adaptive,onnx-cuda'),
    ('rz-uci', 'bin', 'rz-rovent', 'pals-adaptive,onnx-cuda'),
    ('rz-arena', 'bin', 'rz-rovent-trainer', 'pals-collection-onnx,native-cuda'),
    ('rz-arena', 'bin', 'rz-rovent-native', 'pals-collection-onnx,native-cuda,native-validation-diagnostics'),
    ('rz-arena', 'bin', 'rz-rovent-collect', 'pals-collection-onnx,native-cuda'),
    ('rz-arena', 'bin', 'rz-rovent-utility-trainer', 'pals-collection-onnx,native-cuda'),
    ('rz-uci', 'example', 'rovent_startup_metadata', 'pals-adaptive,onnx-cuda'),
    ('rz-experiments', 'bin', 'rz-experiments', 'pals-content-admission'),
    ('rz-uci', 'example', 'pals_continuation_diagnostic', 'pals-adaptive-diagnostics'),
)
WARM = ('rz-uci', 'example', 'virtual_review_warm_numeric_capture',
        'pals-adaptive,onnx-cpu,experimental-io-binding')
RUNTIME_LOADER_SPEC = ('rz-eval', 'example', 'runtime_loader_check', 'onnx')
RUNTIME_LOADER_SUCCESS = 'SAME_HEAD_RUNTIME_LOADER_ELF_AND_SOURCE_PROVENANCE_VERIFIED'
# This standalone example calls the CUDA-enabled eval backend directly. Its
# required features do not certify the separate engine/native_bootstrap path.
FUNCTIONAL_SPECS = (
    ('rz-arena', 'bin', 'rz-rovent-native', 'pals-collection-onnx,native-cuda,native-validation-diagnostics'),
    ('rz-arena', 'bin', 'rz-rovent-collect', 'pals-collection-onnx,native-cuda'),
    ('rz-arena', 'bin', 'rz-rovent-utility-trainer', 'pals-collection-onnx,native-cuda'),
    ('rz-uci', 'example', 'rovent_startup_metadata', 'pals-adaptive,onnx-cuda'),
    ('rz-eval', 'example', 'virtual_review_numeric_capture', 'onnx,contracts'),
    ('rz-experiments', 'bin', 'rz-experiments', 'pals-content-admission'),
)


def require(value, code):
    if not value:
        raise RuntimeError(code)


def validate_functional_specs(specs):
    # This only closes the existing producer's declared 13/optional-Warm 14
    # targets. It grants no artifact, local ABI, provider or runtime admission.
    require(type(specs) is tuple and len(specs) in (13, 14)
            and all(type(row) is tuple and len(row) == 4
                    and all(type(part) is str for part in row) for row in specs),
            'ClosedFunctionalBundleSpecShape')
    by_name = {row[2]: row for row in specs}
    require(len(by_name) == len(specs), 'DuplicateBundleTarget')
    require(all(by_name.get(row[2]) == row for row in FUNCTIONAL_SPECS),
            'RequiredSixFunctionalTargetsKindsAndFeatures')
    require(specs == SPECS or specs == SPECS + (WARM,), 'ClosedFunctionalBundleSpecRoster')


def strict_json(raw):
    def unique(pairs):
        value = {}
        for key, item in pairs:
            require(key not in value, 'DuplicateJsonKey')
            value[key] = item
        return value
    return json.loads(raw, object_pairs_hook=unique,
                      parse_constant=lambda _: (_ for _ in ()).throw(RuntimeError('NonfiniteJson')))


def identity(meta, *, creation_time=False):
    # CPython 3.12+ Windows path stat reports creation time in st_ctime,
    # while fstat reports ChangeTime. Only FD-versus-path comparisons use
    # their common birthtime; FD-versus-FD checks retain ChangeTime.
    timestamp = getattr(meta, 'st_birthtime_ns', meta.st_ctime_ns) if creation_time else meta.st_ctime_ns
    return (meta.st_dev, meta.st_ino, meta.st_mode, meta.st_nlink,
            meta.st_size, meta.st_mtime_ns, timestamp)


def same_file_identity(before, after, named):
    return (identity(before) == identity(after)
            and identity(before, creation_time=os.name == 'nt')
            == identity(named, creation_time=os.name == 'nt'))


def pinned_file(path, maximum=SOURCE_CAP, cargo=False, allow_empty=False):
    """Hash a retained O_NOFOLLOW FD; Cargo alone may own up to 32 hardlinks."""
    checked(path)
    fd = os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0))
    stream = os.fdopen(fd, 'rb')
    try:
        before = os.fstat(fd)
        require(stat.S_ISREG(before.st_mode) and 1 <= before.st_nlink <= (32 if cargo else 1)
                and (0 if allow_empty else 1) <= before.st_size <= maximum, 'ArtifactTypeOrSize')
        digest = hashlib.sha256()
        blob = hashlib.sha1(b'blob ' + str(before.st_size).encode() + b'\0')
        while raw := stream.read(65536):
            digest.update(raw)
            blob.update(raw)
        require(same_file_identity(before, os.fstat(fd), path.lstat()), 'ArtifactIdentityChanged')
        stream.seek(0)
        return stream, {'bytes': before.st_size, 'sha256': digest.hexdigest()}, before, blob.hexdigest()
    except BaseException:
        stream.close()
        raise


def file_pin(path, maximum=SOURCE_CAP):
    stream, pin, _, _ = pinned_file(path, maximum)
    stream.close()
    return pin


def copy_cargo_elf(source, destination, check_budget):
    """Original report's Cargo FD -> private nlink=1 custody, no source unlink."""
    require(not destination.exists(), 'CargoElfCopyPlacement')
    stream, expected, before, _ = pinned_file(source, ELF_CAP, cargo=True)
    with stream:
        check_budget(expected['bytes'])
        require(stream.read(4) == b'\x7fELF', 'CargoOutputNotElf')
        stream.seek(0)
        digest, count = hashlib.sha256(), 0
        with destination.open('xb') as target:
            while raw := stream.read(65536):
                check_budget(0)
                target.write(raw)
                digest.update(raw)
                count += len(raw)
            target.flush()
            os.fsync(target.fileno())
        require(same_file_identity(before, os.fstat(stream.fileno()), source.lstat()), 'CargoOriginalElfChanged')
    destination.chmod(0o700)
    actual = file_pin(destination, ELF_CAP)
    require(actual == expected == {'bytes': count, 'sha256': digest.hexdigest()}
            and destination.lstat().st_nlink == 1, 'PrivateElfCopyReadback')
    return {'original_cargo': {**expected, 'device': before.st_dev, 'inode': before.st_ino,
                               'nlink': before.st_nlink, 'mtime_ns': before.st_mtime_ns},
            'private_copy': {**actual, 'nlink': 1}, 'original_fd_and_named_identity_stable': True}


def allowed_source(path):
    parts = path.split('/')
    forbidden = {'.git', 'node_modules', 'dist', 'dist-server', 'build', 'coverage',
                 'models', 'checkpoints', '.cache', '__pycache__'}
    require(0 < len(path) <= 512 and '\\' not in path and ':' not in path
            and all(p not in ('', '.', '..') and p.lower() not in forbidden
                    and not p.lower().startswith('.env') for p in parts)
            and not any(s in path.lower() for s in ('credential', 'service_account', 'service-account',
                                                     'ssh-key', 'api-key', 'api_key')),
            'ForbiddenSourcePath')
    return path == CONTROL or path in SINGLETONS or path.startswith(PREFIXES)


def source_inventory(source, tree_raw):
    rows, controls = [], {}
    for entry in tree_raw.split(b'\0'):
        if not entry:
            continue
        header, name = entry.split(b'\t', 1)
        name = name.decode('utf-8')
        require(allowed_source(name), 'SourceOutsideExplicitRoster')
        mode, kind, oid = header.decode('ascii').split()
        require(mode in ('100644', '100755') and kind == 'blob' and re.fullmatch('[a-f0-9]{40}', oid), 'GitTrackedRegularSource')
        stream, pin, _, blob = pinned_file(source / name, 16 << 20, allow_empty=True)
        stream.close()
        require(blob == oid, 'CheckoutBytesDifferFromGitBlob:' + name)
        row = {'path': name, **pin, 'git_blob': oid, 'git_mode': mode}
        if name == CONTROL:
            controls[name] = row
        else:
            rows.append(row)
    require(1 <= len(rows) <= 10000 and len({r['path'] for r in rows}) == len(rows)
            and sum(r['bytes'] for r in rows) <= SOURCE_CAP
            and SINGLETONS <= {r['path'] for r in rows} and set(controls) == {CONTROL}, 'ExactTrackedSourceRoster')
    return sorted(rows, key=lambda row: row['path']), controls


def component_inventory(catalog, rows):
    require(set(catalog) == {'domain', 'algorithm', 'components'}
            and catalog['domain'] == 'rz-pals-source-inventory-v5/1'
            and catalog['algorithm'] == 'sha256-domain-u64le;sorted-path-u64le-utf8;size-u64le;sha256-raw32'
            and len(catalog['components']) == 25, 'CatalogFraming')
    by_path = {row['path']: row for row in rows}
    result = {}
    for name, paths in catalog['components'].items():
        require(1 <= len(paths) <= 256 and len(set(paths)) == len(paths)
                and all(p in by_path for p in paths), 'CatalogSourceMissing')
        domain = b'rz-pals-source-component-v5/1'
        digest = hashlib.sha256(len(domain).to_bytes(8, 'little') + domain)
        for path in sorted(paths):
            encoded, row = path.encode(), by_path[path]
            digest.update(len(encoded).to_bytes(8, 'little') + encoded)
            digest.update(row['bytes'].to_bytes(8, 'little') + bytes.fromhex(row['sha256']))
        result[name] = digest.hexdigest()
    require(set(CORE5) <= set(result), 'Core5Missing')
    return result


def warm_spec(manifest, paths):
    target = next((r for r in manifest.get('example', []) if r.get('name') == WARM[2]), None)
    source_path = 'crates/rz-uci/examples/' + WARM[2] + '.rs'
    if target is None or source_path not in paths:
        return None, 'NOT_PRESENT_IN_REGISTERED_SOURCE_AND_EXPLICIT_TARGET'
    if (set(target.get('required-features', [])) != set(WARM[3].split(','))
            or not set(WARM[3].split(',')) <= set(manifest['features'])
            or target.get('path', 'examples/' + WARM[2] + '.rs') != 'examples/' + WARM[2] + '.rs'):
        return None, 'NOT_READY_REQUIRED_FEATURE_OR_SOURCE_CONTRACT'
    return WARM, 'SOURCE_PRESENT_REQUIRES_ACTUAL_BUILD'


def cargo_artifact(raw, metadata, source, target_root, spec, registered_paths):
    package_name, kind, name, features = spec
    packages = [p for p in metadata['packages'] if p['name'] == package_name
                and p['manifest_path'] == str(source / 'crates' / package_name / 'Cargo.toml')]
    require(len(packages) == 1, 'ExactCargoPackage')
    targets = [t for t in packages[0]['targets'] if t['name'] == name
               and t['kind'] == [kind] and t['crate_types'] == ['bin']]
    require(len(targets) == 1 and Path(targets[0]['src_path']).is_relative_to(source)
            and Path(targets[0]['src_path']).relative_to(source).as_posix() in registered_paths, 'ExactRegisteredCargoTarget')
    messages = [strict_json(line) for line in raw.splitlines() if line.strip()]
    artifacts = [m for m in messages if m.get('reason') == 'compiler-artifact'
                 and m.get('package_id') == packages[0]['id'] and m.get('target', {}).get('name') == name]
    finishes = [m for m in messages if m.get('reason') == 'build-finished']
    require(len(artifacts) == len(finishes) == 1 and finishes[0].get('success') is True, 'ExactlyOneSuccessfulCargoArtifact')
    artifact = artifacts[0]
    output = target_root / 'release' / ('examples' if kind == 'example' else '') / name
    require(all(artifact['target'][k] == targets[0][k] for k in ('name', 'kind', 'crate_types', 'src_path', 'edition'))
            and artifact['executable'] == str(output) and artifact['profile']['test'] is False
            and len(artifact['features']) == len(set(artifact['features']))
            and set(features.split(',')) <= set(artifact['features'])
            and 'default' not in artifact['features'], 'ActualReleaseTargetPathFeatures')
    if name == 'pals_continuation_diagnostic':
        require(sorted(artifact['features']) == ['onnx-cpu', 'pals-adaptive', 'pals-adaptive-diagnostics', 'search-work-receipts'], 'FreshDiagnosticCannotAcquireWarmFeatures')
    return output, artifact


def validate_registry(value):
    require(value.get('schema') == 'rz-pals-report-model-registry-query/1'
            and value.get('model_profile') == 'virtual_review_v1'
            and value.get('registry_methods') == [
                'PalsModelProfile::VirtualReviewV1.registered_semantics_sha256',
                'pals_model::virtual_review::registered_encoding_sha256']
            and all(value.get(name) is False for name in (
                'numerical_authority_issued', 'model_initialized', 'ort_loaded', 'neural_executed', 'training_executed'))
            and all(type(value.get(name)) is str and re.fullmatch('[a-f0-9]{64}', value[name])
                    for name in ('model_semantic_sha256', 'encoding_sha256')),
            'StaticRegistryGetterOutput')


def validate_rovent_build_metadata(value, expected_head, components):
    """Observe compile metadata only; neither load nor admit a native runtime."""
    fields = {'domain', 'source_commit', 'source_components', 'target_os', 'target_arch',
              'onnx_cpu_compiled', 'onnx_cuda_compiled', 'native_runtime_loaded',
              'native_model_loaded', 'neural_execution_observed'}
    require(type(value) is dict and set(value) == fields
            and value['domain'] == 'rz-rovent-build-capabilities/1'
            and value['source_commit'] == expected_head
            and value['target_os'] == 'linux' and value['target_arch'] == 'x86_64'
            and value['onnx_cpu_compiled'] is True and value['onnx_cuda_compiled'] is True
            and all(value[name] is False for name in (
                'native_runtime_loaded', 'native_model_loaded', 'neural_execution_observed')),
            'StaticRoventBuildMetadata')
    observed = value['source_components']
    private = {'rovent_bootstrap', 'rovent_process_boundary'}
    require(type(observed) is dict and set(observed) == set(components) | private
            and all(observed[name] == digest for name, digest in components.items())
            and all(type(observed[name]) is str and re.fullmatch('[a-f0-9]{64}', observed[name])
                    for name in private), 'StaticRoventRegisteredSourceMismatch')


def stream_child(argv):
    """Drain both pipes but save at most 8MiB each; remain in the owned group.

    The shared non-reaping supervisor owns this wrapper's original session and
    drains/kills the entire group even when the command exits or leaves pipes.
    No new session, detached process, shell, fallback or retry is created here.
    """
    affinity = [int(value) for value in os.environ['RZ_CI_CPU_AFFINITY'].split(',')]
    require(len(affinity) == 2 and sorted(os.sched_getaffinity(0)) == affinity, 'ChildCpu2AffinityReadback')
    child = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, close_fds=True)
    counts, kept = [0, 0], [0, 0]
    selector = selectors.DefaultSelector()
    try:
        for index, pipe in enumerate((child.stdout, child.stderr)):
            os.set_blocking(pipe.fileno(), False)
            selector.register(pipe, selectors.EVENT_READ, index)
        while selector.get_map():
            for key, _ in selector.select(0.2):
                data = os.read(key.fileobj.fileno(), 65536)
                if not data:
                    selector.unregister(key.fileobj)
                    key.fileobj.close()
                    continue
                index = key.data
                counts[index] += len(data)
                save = memoryview(data[:max(0, LOG_CAP - kept[index])])
                while save:
                    written = os.write(index + 1, save)
                    require(written > 0, 'CommandLogShortWrite')
                    save = save[written:]
                    kept[index] += written
        code = child.wait()
        return code if counts == kept else 125
    finally:
        selector.close()


def cleanup_compiler_outputs(root, artifact, receipt, *,
                             success_status='SAME_HEAD_RELEASE_ELFS_AND_STATIC_SOURCE_METADATA_VERIFIED',
                             bundle_filename='native-validation-binaries.tar.gz'):
    """Retire only this fresh build's regenerable folders after custody closes."""
    from managed_storage import remove_tree, write_record
    require((success_status, bundle_filename) in (
        ('SAME_HEAD_RELEASE_ELFS_AND_STATIC_SOURCE_METADATA_VERIFIED', 'native-validation-binaries.tar.gz'),
        (RUNTIME_LOADER_SUCCESS, 'runtime-loader-diagnostic-binaries.tar.gz')),
        'ClosedCompileCleanupProfile')
    require(receipt['status'] == success_status
            and all(row.get('original_tree_cleanup_verified') is True for row in receipt['commands'])
            and file_pin(artifact / bundle_filename, ARTIFACT_CAP) == receipt['binary_bundle'],
            'VerifiedBundleAndTreeClosureBeforeCleanup')
    names = ('target-native', 'target-warm', 'target-diagnostic', 'cargo-home', 'tmp', 'private-elf')
    # No model, source, original external asset or failure evidence is listed.
    for name in names:
        tree_size(root / name)
    journal = {'schema': 'rovezero.ci-compile-cleanup.v1',
               'scope': 'fresh_ci_compile_only', 'bundle_pin': receipt['binary_bundle'],
               'planned_direct_children': list(names), 'removed': [], 'complete': False}
    write_record(artifact / 'compile-cleanup.json', journal)
    for name in names:
        if (root / name).exists():
            remove_tree(root / name, root)
        journal['removed'].append(name)
        write_record(artifact / 'compile-cleanup.json', journal)
    journal['complete'] = True
    write_record(artifact / 'compile-cleanup.json', journal)
    return journal


class CommandExitFailure(RuntimeError):
    """A nonzero command with bounded logs and a verified closed original tree."""
    def __init__(self, row):
        super().__init__('ClosedCommandExitFailed:' + row['label'])
        self.row = row


class Build:
    def __init__(self, root, source, seconds):
        self.root, self.source = root, source
        self.deadline = time.monotonic() + seconds
        self.commands = []
        self.next_scan = 0.0
        (root / 'logs').mkdir(mode=0o700)

    def budget(self, extra=0):
        now = time.monotonic()
        require(now < self.deadline, 'OriginalWholeBuildDeadline')
        if extra or now >= self.next_scan:
            require(tree_size(self.root, allow_disappearing=True) + extra + (16 << 20) <= ROOT_CAP, 'OwnedRoot12GiBCap')
            require(shutil.disk_usage(self.root).free >= extra + (16 << 20), 'ActualFreeSpaceReserve')
            self.next_scan = now + 5

    def command(self, label, argv, env):
        self.budget()
        out, err = self.root / 'logs' / (label + '.stdout'), self.root / 'logs' / (label + '.stderr')
        row = {'label': label, 'argv': argv, 'cwd': str(self.source), 'stdout': out.name, 'stderr': err.name}
        row['build_environment'] = {key: env[key] for key in (
            'CARGO_BUILD_JOBS', 'CARGO_INCREMENTAL', 'RUSTUP_TOOLCHAIN', 'RZ_PALS_SOURCE_COMMIT')}
        row['cargo_target_dir'] = env.get('CARGO_TARGET_DIR')
        self.commands.append(row)
        started = time.monotonic()

        def observe(event, **values):
            # Progress is best effort; command and process-tree receipts decide admission.
            try:
                print(json.dumps({
                    'event': event, 'label': label,
                    'elapsed_seconds': round(time.monotonic() - started, 3),
                    'stdout_log': str(out), 'stderr_log': str(err), **values,
                }, sort_keys=True, allow_nan=False), file=sys.stderr, flush=True)
            except (OSError, ValueError):
                pass

        observe('native_bundle_command_start', status='starting', argv=argv,
                cargo_target_dir=row['cargo_target_dir'])

        def check():
            self.budget()
            require(all(p.stat().st_size <= LOG_CAP for p in (out, err)), 'CommandLog8MiBCap')

        # run_group inherits actual OS stdout/stderr descriptors. There are no
        # shell wrappers and no changes to the shared supervisor/default jobs.
        with out.open('xb') as stdout, err.open('xb') as stderr:
            saved = [os.dup(1), os.dup(2)]
            try:
                sys.stdout.flush()
                sys.stderr.flush()
                os.dup2(stdout.fileno(), 1)
                os.dup2(stderr.fileno(), 2)
                wrapper = [sys.executable, '-I', '-S', '-B', str(Path(__file__).resolve()), '--internal-command', *argv]
                row['supervised_wrapper_argv'] = wrapper
                code, gone, failure = run_group(wrapper, self.source, env,
                                                max(1, int(self.deadline - time.monotonic())), check)
            finally:
                os.dup2(saved[0], 1)
                os.dup2(saved[1], 2)
                for fd in saved:
                    os.close(fd)
                stdout.flush()
                stderr.flush()
                os.fsync(stdout.fileno())
                os.fsync(stderr.fileno())
        row.update(exit_code=code, original_tree_cleanup_verified=gone, supervisor_error=failure)
        observe('native_bundle_command_terminal',
                status='process_and_tree_closed' if code == 0 and gone is True and failure is None
                else 'command_or_original_tree_failed',
                exit_code=code, original_tree_cleanup_verified=gone,
                supervisor_error=None if failure is None else str(failure)[:2048])
        # The shared supervisor also uses 124/130 for timeout/cancellation even
        # when its original tree was drained. Those are never target-local.
        require(type(code) is int and code >= 0 and code not in (124, 130)
                and gone is True and failure is None,
                'CommandOrOriginalTreeClosureFailed:' + label)
        check()
        stream, row['stdout_pin'], before, _ = pinned_file(out, LOG_CAP, allow_empty=code != 0)
        with stream:
            raw = stream.read()
            require(same_file_identity(before, os.fstat(stream.fileno()), out.lstat()), 'CommandStdoutFDChanged')
        # Empty stderr is ordinary; stdout-consuming commands are nonempty.
        row['stderr_bytes'] = err.stat().st_size
        stream, row['stderr_pin'], _, _ = pinned_file(err, LOG_CAP, allow_empty=True)
        stream.close()
        if code != 0:
            raise CommandExitFailure(row)
        return raw


def build_selected_targets(build, *, cargo, env, metadata, source, private, specs,
                           registered_paths, binaries, failures, failure_check, warm):
    """Continue only independent Cargo exits; custody/input failures still abort."""
    for spec in specs:
        package, kind, name, features = spec
        argv = [cargo, 'build', '--locked', '--release', '--no-default-features',
                '-p', package, '--features', features, '--' + kind, name, '--message-format=json-render-diagnostics']
        try:
            raw = build.command('build-' + name, argv, env)
        except CommandExitFailure as error:
            row = error.row
            failures.append({'target': list(spec), 'command_label': row['label'],
                             'exit_code': row['exit_code'],
                             'original_tree_cleanup_verified': row['original_tree_cleanup_verified'],
                             'supervisor_error': row['supervisor_error'],
                             'stdout_pin': dict(row['stdout_pin']), 'stderr_pin': dict(row['stderr_pin'])})
            # A closed failing compiler does not excuse changed shared inputs.
            failure_check()
            continue
        output, compiled = cargo_artifact(raw, metadata, source, Path(env['CARGO_TARGET_DIR']),
                                          spec, registered_paths)
        copy_result = copy_cargo_elf(output, private / name, build.budget)
        required = next(t.get('required-features', []) for p in metadata['packages'] if p['name'] == package
                        for t in p['targets'] if t['name'] == name and t['kind'] == [kind])
        binaries[name] = {'pin': copy_result['private_copy'], 'cargo_copy': copy_result,
                          'compiler_artifact': compiled, 'requested_features': features.split(','),
                          'cargo_metadata_required_features': required,
                          'compiled_capability_scope': 'standalone_eval_warm_capture_only_not_engine_native_bootstrap_acceptance' if spec == warm else 'explicit_cargo_target_only_runtime_not_admitted',
                          'role': 'optional_warm_validation' if spec == warm else ('diagnostic_auxiliary' if name == 'pals_continuation_diagnostic' else 'native_validation')}


def build_runtime_loader_target(build, *, cargo, env, metadata, source, private,
                                registered_paths, binaries, failures, failure_check):
    """One model-free diagnostic ELF; the full13 and all runtime work are separate."""
    require(not binaries and not failures, 'FreshRuntimeLoaderTargetResult')
    build_selected_targets(build, cargo=cargo, env=env, metadata=metadata, source=source,
                           private=private, specs=(RUNTIME_LOADER_SPEC,),
                           registered_paths=registered_paths, binaries=binaries, failures=failures,
                           failure_check=failure_check, warm=None)
    if failures:
        require(not binaries and len(failures) == 1, 'SingleRuntimeLoaderTargetFailure')
        return
    require(set(binaries) == {RUNTIME_LOADER_SPEC[2]}, 'SingleRuntimeLoaderTargetRoster')
    row = binaries[RUNTIME_LOADER_SPEC[2]]
    require(row['cargo_metadata_required_features'] == ['onnx'], 'RuntimeLoaderRequiredOnnxFeature')
    row.update(role='runtime_loader_diagnostic',
               compiled_capability_scope='model_free_runtime_loader_only_no_session_model_NN_or_runtime_acceptance')


def preserve_runtime_loader_sources(artifact, source, expected_head, rows, controls,
                                    lock, lock_pin, lock_meta):
    """Save actual source/lock provenance without querying another target or loading ORT."""
    pins = [{key: row[key] for key in ('path', 'bytes', 'sha256')} for row in rows]
    write_json(artifact / 'source-file-pins.json', {
        'schema': 'rovezero.ci-runtime-loader-source-pins/1',
        'source_commit': expected_head, 'files': pins})
    write_json(artifact / 'git-source-provenance.json', {
        'schema': 'rovezero.ci-runtime-loader-source-provenance/1',
        'source_commit': expected_head, 'tracked_files': rows, 'build_controls': controls})
    lock.seek(0)
    with (artifact / 'Cargo.lock').open('xb') as target:
        shutil.copyfileobj(lock, target, 65536)
        target.flush()
        os.fsync(target.fileno())
    require(same_file_identity(lock_meta, os.fstat(lock.fileno()), (source / 'Cargo.lock').lstat()),
            'BundledLockOriginalFDChanged')
    require(file_pin(artifact / 'Cargo.lock') == lock_pin, 'BundledLockPin')


def selected_elf_payload(binaries):
    sizes = [row['pin']['bytes'] for row in binaries.values()]
    require(sizes and all(type(size) is int and 0 < size <= ELF_CAP for size in sizes)
            and sum(sizes) <= ELF_PAYLOAD_CAP, 'SelectedELF768MiBPayloadBudget')
    return sum(sizes)


class BoundedArchiveWriter:
    """Bound compressed bytes separately from the verified ELF payload."""
    def __init__(self, stream, maximum):
        require(type(maximum) is int and maximum > 0, 'PositiveCompressedArchiveBudget')
        self.stream, self.maximum, self.written = stream, maximum, 0
        self.name = stream.name

    def write(self, data):
        require(self.written + len(data) <= self.maximum, 'CompressedArchiveBudget')
        written = self.stream.write(data)
        require(written == len(data), 'CompressedArchiveShortWrite')
        self.written += written
        return written

    def flush(self):
        self.stream.flush()

    def tell(self):
        return self.written


def archive_selected_binaries(build, artifact, private, binaries, filename):
    """Preserve only verified private copies, without granting bundle admission."""
    total = selected_elf_payload(binaries)
    build.budget(total)
    bundle = artifact / filename
    compressed_cap = ARTIFACT_CAP - tree_size(artifact) - (16 << 20)
    require(compressed_cap > 0, 'CompressedArchiveMetadataReserve')
    with bundle.open('xb') as raw, tarfile.open(fileobj=BoundedArchiveWriter(raw, compressed_cap),
                                              mode='w:gz', format=tarfile.USTAR_FORMAT) as archive:
        for name in sorted(binaries):
            stream, actual, before, _ = pinned_file(private / name, ELF_CAP)
            with stream:
                require(actual == {k: binaries[name]['pin'][k] for k in ('bytes', 'sha256')}, 'PreBundlePrivatePin')
                info = tarfile.TarInfo('bin/' + name)
                info.size, info.mode = actual['bytes'], 0o700
                archive.addfile(info, stream)
                require(same_file_identity(before, os.fstat(stream.fileno()), (private / name).lstat()), 'BundleMemberFDChanged')
    return bundle, file_pin(bundle, ARTIFACT_CAP)


def write_json(path, value):
    raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    require(len(raw) <= 4 << 20, 'SmallMetadata4MiBCap')
    with path.open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def run(source, root, expected_head, seconds, *, profile='native-validation'):
    require(profile in ('native-validation', 'runtime-loader-only'), 'ExplicitBuildProfile')
    loader_only = profile == 'runtime-loader-only'
    success_status = RUNTIME_LOADER_SUCCESS if loader_only else 'SAME_HEAD_RELEASE_ELFS_AND_STATIC_SOURCE_METADATA_VERIFIED'
    bundle_filename = 'runtime-loader-diagnostic-binaries.tar.gz' if loader_only else 'native-validation-binaries.tar.gz'
    require(sys.platform == 'linux' and platform.machine() in ('x86_64', 'AMD64')
            and re.fullmatch('[a-f0-9]{40}', expected_head)
            and 1 <= seconds <= MAX_BUILD_SECONDS, 'LinuxFiniteBuildProfile')
    source, root = checked(source), checked(root)
    runner_temp = checked(Path(os.environ['RUNNER_TEMP']))
    require(root.is_relative_to(runner_temp / 'RoveZero') and not root.is_relative_to(source)
            and not source.is_relative_to(root) and not root.exists(), 'FreshRunnerTempOwnedRoot')
    root.parent.mkdir(parents=True, exist_ok=True)
    root.mkdir(mode=0o700)
    artifact = root / 'artifact'
    artifact.mkdir(mode=0o700)
    private = root / 'private-elf'
    private.mkdir(mode=0o700)
    available = sorted(os.sched_getaffinity(0))
    require(len(available) >= 2, 'TwoAvailableCpusRequired')
    affinity = available[:2]
    os.sched_setaffinity(0, affinity)
    require(sorted(os.sched_getaffinity(0)) == affinity, 'Cpu2AffinityReadback')
    build = Build(root, source, seconds)
    def cancelled(_signum, _frame):
        # The shared supervisor handles KeyboardInterrupt by draining the
        # reserved original process group before returning an unsuccessful row.
        raise KeyboardInterrupt('OwnedBuildCancelled')
    signal.signal(signal.SIGTERM, cancelled)
    receipt = {'schema': 'rovezero.ci-runtime-loader-diagnostic/1' if loader_only else 'rovezero.ci-cpu-validation-bundle/1',
               'status': 'FAILED_NOT_ADMITTED',
               'source_commit': expected_head, 'commands': build.commands,
               'resource': {'cpu_affinity_actual': affinity, 'build_jobs': 2, 'seconds': seconds,
                            'child_affinity_guard': 'isolated_wrapper_exact_readback_before_exec',
                            'managed_bytes_cap': ROOT_CAP, 'artifact_bytes_cap': ARTIFACT_CAP,
                            'uncompressed_elf_payload_bytes_cap': ELF_PAYLOAD_CAP},
               'neural_execution': False, 'gpu_execution': False, 'training_execution': False,
               'optimizer_updates': 0, 'mapped_image': None, 'numerical_or_runtime_acceptance': None,
               'host': {'machine': platform.machine(), 'libc_observed': platform.libc_ver(),
                        'glibc_compatibility_claim': None},
               'metadata_getter_scope': 'source_and_selected_Cargo_provenance_only_no_ELF_execution' if loader_only else 'static_source_inventory_core5_registry_rules_only',
               'binaries': {}, 'target_failures': []}
    if loader_only:
        receipt.update(diagnostic_profile=profile, diagnostic_binary_executed=False,
                       runtime_loaded=False, model_loaded=False, session_created=False,
                       full_native_bundle_admitted=False)
    receipt['uploaded_command_log_directory'] = 'command-logs'
    env = {key: os.environ[key] for key in ('PATH', 'HOME', 'RUSTUP_HOME') if key in os.environ}
    for folder in ('cargo-home', 'tmp'):
        (root / folder).mkdir(mode=0o700)
    env.update(LANG='C.UTF-8', LC_ALL='C.UTF-8', CARGO_HOME=str(root / 'cargo-home'),
               TMPDIR=str(root / 'tmp'), CARGO_BUILD_JOBS='2', CARGO_INCREMENTAL='0',
               RUSTUP_TOOLCHAIN='1.96.0', RZ_PALS_SOURCE_COMMIT=expected_head,
               CARGO_TERM_COLOR='never', CARGO_NET_GIT_FETCH_WITH_CLI='false',
               PYTHONDONTWRITEBYTECODE='1',
               RZ_CI_CPU_AFFINITY=','.join(str(cpu) for cpu in affinity),
               GIT_CONFIG_NOSYSTEM='1', GIT_CONFIG_GLOBAL='/dev/null')
    lock = None
    try:
        tools = {name: shutil.which(name) for name in ('git', 'cargo', 'rustc')}
        require(all(tools.values()), 'RequiredBuildTools')
        actual_head = build.command('git-head', [tools['git'], 'rev-parse', 'HEAD'], env).decode().strip()
        require(actual_head == expected_head, 'SameCheckoutHead')
        tree = build.command('git-source-roster', [tools['git'], 'ls-tree', '-r', '-z', '--full-tree', 'HEAD', '--',
                             *sorted(SINGLETONS), *PREFIXES, CONTROL], env)
        rows, controls = source_inventory(source, tree)
        if not loader_only:
            components = component_inventory(strict_json((source / 'crates/rz-experiments/src/pals_source_inventory_v5.json').read_bytes()), rows)
        lock, lock_pin, lock_meta, _ = pinned_file(source / 'Cargo.lock')
        toolchain = tomllib.loads((source / 'rust-toolchain.toml').read_text())
        require(toolchain['toolchain']['channel'] == '1.96.0', 'RegisteredRustToolchain')
        for name in ('rustc', 'cargo'):
            raw = build.command(name + '-version', [tools[name], '--version', '--verbose'], env)
            require((name + ' 1.96.0 ').encode() in raw, 'ActualToolchainVersion')
        env['CARGO_TARGET_DIR'] = str(root / 'target-native')
        metadata = strict_json(build.command('cargo-metadata', [tools['cargo'], 'metadata', '--locked', '--no-deps',
                                                             '--no-default-features', '--format-version', '1'], env))
        require(metadata['workspace_root'] == str(source), 'ActualCargoWorkspace')
        if loader_only:
            warm = None
            selected_specs = (RUNTIME_LOADER_SPEC,)
            preserve_runtime_loader_sources(artifact, source, expected_head, rows, controls,
                                            lock, lock_pin, lock_meta)
        else:
            manifest = tomllib.loads((source / 'crates/rz-uci/Cargo.toml').read_text())
            warm, receipt['warm_status'] = warm_spec(manifest, {r['path'] for r in rows})
            selected_specs = SPECS + ((warm,) if warm else ())
            validate_functional_specs(selected_specs)

        def recheck_registered_inputs():
            require(source_inventory(source, tree) == (rows, controls)
                    and same_file_identity(lock_meta, os.fstat(lock.fileno()), (source / 'Cargo.lock').lstat())
                    and file_pin(source / 'Cargo.lock') == lock_pin, 'SourceOrLockChangedDuringBuild')

        receipt['requested_targets'] = [list(spec) for spec in selected_specs]
        # Reuse Cargo fingerprints, but catch only a verified closed ordinary
        # compiler exit. Validation, storage, timeout and shared-input errors abort.
        target_arguments = dict(cargo=tools['cargo'], env=env, metadata=metadata,
                                source=source, private=private, registered_paths={r['path'] for r in rows},
                                binaries=receipt['binaries'], failures=receipt['target_failures'],
                                failure_check=recheck_registered_inputs)
        if loader_only:
            build_runtime_loader_target(build, **target_arguments)
        else:
            build_selected_targets(build, **target_arguments, specs=selected_specs, warm=warm)
        if receipt['target_failures']:
            recheck_registered_inputs()
            require(build.command('git-head-final', [tools['git'], 'rev-parse', 'HEAD'], env).decode().strip() == expected_head,
                    'HeadChangedDuringBuild')
            if receipt['binaries']:
                partial_filename = 'partial-runtime-loader-diagnostic-binaries.tar.gz' if loader_only else 'partial-native-validation-binaries.tar.gz'
                _, partial_pin = archive_selected_binaries(build, artifact, private, receipt['binaries'], partial_filename)
                receipt['partial_binary_bundle'] = partial_pin
                receipt['partial_binary_bundle_file'] = partial_filename
                receipt['partial_bundle_scope'] = 'successful_targets_only_failed_full_bundle_not_admitted'
            raise RuntimeError('RequiredRuntimeLoaderTargetFailed' if loader_only else 'RequiredCargoTargetsFailed')
        require(set(receipt['binaries']) == {spec[2] for spec in selected_specs}, 'CompleteSelectedTargetRoster')
        if warm:
            receipt['warm_status'] = 'BUILT_SAME_HEAD_NOT_EXECUTED_OR_NUMERICALLY_ADMITTED'
        if loader_only:
            recheck_registered_inputs()
            require(build.command('git-head-final', [tools['git'], 'rev-parse', 'HEAD'], env).decode().strip() == expected_head,
                    'HeadChangedDuringBuild')
            write_json(artifact / 'cargo-metadata.json', metadata)
            bundle, bundle_pin = archive_selected_binaries(build, artifact, private, receipt['binaries'], bundle_filename)
            receipt.update(status=success_status, source_files=len(rows),
                           Cargo_lock_before_after=lock_pin, Cargo_lock_same_fd_stable=True,
                           binary_bundle=bundle_pin, binary_bundle_file=bundle_filename,
                           metadata={p.name: file_pin(p) for p in artifact.iterdir() if p.name != bundle.name})
        else:
            compiled = strict_json(build.command('compiled-source-inventory', [str(private / 'rz-pals-adaptive'), '--source-inventory'], env))
            require(compiled == {'domain': 'rz-pals-compiled-source-registration-v5/1', 'source_commit': expected_head,
                                 'components': components, 'neural_execution': False, 'training_execution': False}, 'CompiledInventorySourceMismatch')
            rovent = strict_json(build.command('rovent-build-metadata', [str(private / 'rz-rovent'), '--build-capabilities-json'], env))
            validate_rovent_build_metadata(rovent, expected_head, components)
            receipt['rovent_build_binding'] = {
                'ELF': receipt['binaries']['rz-rovent']['pin'],
                'metadata': rovent,
                'registered_components_independently_matched': True,
                'private_component_digests': 'observed_only_not_independently_recomputed',
                'scope': 'same_actual_ELF_compile_metadata_no_runtime_model_or_execution_authority'}
            native = strict_json(build.command('native-source-pins', [str(private / 'virtual_review_numeric_capture'), '--source-pins'], env))
            require(native == {name: components[name] for name in CORE5}, 'NativeCore5SourceMismatch')
            registry = strict_json(build.command('static-model-registry', [str(private / 'virtual_review_numeric_capture'), '--model-registry'], env))
            validate_registry(registry)
            receipt['static_registry_binding'] = {
                'capture_ELF': receipt['binaries']['virtual_review_numeric_capture']['pin'],
                'registered_source_core5': native,
                'registered_semantic_sha256': registry['model_semantic_sha256'],
                'registered_encoding_sha256': registry['encoding_sha256'],
                'scope': 'same_actual_ELF_public_registry_methods_and_compiled_source_pins_no_numerical_authority'}
            rules = strict_json(build.command('rules-descriptor', [str(private / 'pals_rules_descriptor'), '--model-profile=virtual_review_v1'], env))
            require(rules.get('schema') == 'rovezero.pals-rules-descriptor.v3'
                    and rules.get('rules_input_profile') == 'rz-pals-rules-virtual-review-v1'
                    and rules.get('config', {}).get('profile') == 'virtual_review_v1'
                    and rules.get('learned_input_compatibility') == 'unverified_declaration_only'
                    and rules.get('bounds') == {'max_records': 64, 'max_anchors': 4, 'max_line_plies': 256,
                                                'line_buckets': [8, 16, 32, 64, 128, 256], 'max_candidates': 256,
                                                'token_budgets': [128, 256, 512], 'root_anchor': 0}, 'StaticRulesDescriptorProfile')
            semantic = hashlib.sha256()
            for field in rules['semantic_fields']:
                encoded = field.encode('utf-8')
                semantic.update(len(encoded).to_bytes(8, 'little') + encoded)
            encoder = hashlib.sha256(b'rz-pals-virtual-rules-encoder-source/1\0')
            for name, logical in (
                    ('pals_native.rs', 'crates/rz-uci/src/pals_native.rs'),
                    ('pals_native/virtual_encoding.rs', 'crates/rz-uci/src/pals_native/virtual_encoding.rs'),
                    ('pals_native/virtual_selected_input.rs', 'crates/rz-uci/src/pals_native/virtual_selected_input.rs'),
                    ('pals_native/ticketed/virtual_bridge.rs', 'crates/rz-uci/src/pals_native/ticketed/virtual_bridge.rs'),
                    ('rz-eval/pals_model/virtual_review.rs', 'crates/rz-eval/src/pals_model/virtual_review.rs')):
                stream, _, _, _ = pinned_file(source / logical, 16 << 20)
                with stream:
                    raw = stream.read()
                encoded = name.encode()
                encoder.update(len(encoded).to_bytes(8, 'little') + encoded)
                encoder.update(len(raw).to_bytes(8, 'little') + raw)
            require(rules.get('rules_input_semantic_sha256') == semantic.hexdigest()
                    and rules.get('rules_encoder_source_sha256') == encoder.hexdigest(), 'StaticRulesDescriptorSourceFraming')
            recheck_registered_inputs()
            require(build.command('git-head-final', [tools['git'], 'rev-parse', 'HEAD'], env).decode().strip() == expected_head,
                    'HeadChangedDuringBuild')
            pins = [{key: row[key] for key in ('path', 'bytes', 'sha256')} for row in rows]
            for name, value in (('source-file-pins.json', {'schema': 'rovezero.ci-native-source-pins/1',
                                                         'source_commit': expected_head, 'files': pins, 'source_components': components}),
                                ('git-source-provenance.json', {'schema': 'rovezero.ci-git-source-provenance/1',
                                                                'source_commit': expected_head, 'tracked_files': rows, 'build_controls': controls}),
                                ('compiled-source-inventory.json', compiled), ('native-source-pins.json', native),
                                ('static-model-registry.json', registry), ('rules-descriptor.json', rules)):
                write_json(artifact / name, value)
            lock.seek(0)
            with (artifact / 'Cargo.lock').open('xb') as target:
                shutil.copyfileobj(lock, target, 65536)
                target.flush()
                os.fsync(target.fileno())
            require(same_file_identity(lock_meta, os.fstat(lock.fileno()), (source / 'Cargo.lock').lstat()),
                    'BundledLockOriginalFDChanged')
            require(file_pin(artifact / 'Cargo.lock') == lock_pin, 'BundledLockPin')
            bundle, bundle_pin = archive_selected_binaries(build, artifact, private, receipt['binaries'],
                                                           'native-validation-binaries.tar.gz')
            receipt.update(status='SAME_HEAD_RELEASE_ELFS_AND_STATIC_SOURCE_METADATA_VERIFIED',
                           source_files=len(rows), component_count=len(components), source_components=components,
                           Cargo_lock_before_after=lock_pin, Cargo_lock_same_fd_stable=True,
                            binary_bundle=bundle_pin, metadata={p.name: file_pin(p) for p in artifact.iterdir() if p.name != bundle.name})
        build.budget()
    except BaseException as error:
        receipt['status'] = 'FAILED_NOT_ADMITTED'
        receipt['error'] = {'type': type(error).__name__, 'message': str(error)[:2048]}
    finally:
        if lock is not None:
            lock.close()
        # Preserve only bounded closed logs. An unresolved tree never produces
        # a successful bundle/closure claim; the job's managed root is retained.
        if all(row.get('original_tree_cleanup_verified') is True for row in build.commands):
            logs = artifact / 'command-logs'
            logs.mkdir(mode=0o700)
            for path in sorted((root / 'logs').iterdir()):
                try:
                    stream, log_pin, before, _ = pinned_file(path, LOG_CAP, allow_empty=True)
                    with stream, (logs / path.name).open('xb') as target:
                        shutil.copyfileobj(stream, target, 65536)
                        target.flush()
                        os.fsync(target.fileno())
                        require(same_file_identity(before, os.fstat(stream.fileno()), path.lstat()), 'ClosedCommandLogFDChanged')
                    # This existing closed-log preservation also covers timeout
                    # or cancellation rows that stopped before command pinning.
                    for row in build.commands:
                        for channel in ('stdout', 'stderr'):
                            if row[channel] == path.name:
                                require(channel + '_pin' not in row or row[channel + '_pin'] == log_pin,
                                        'ClosedCommandLogPinChanged')
                                row[channel + '_pin'] = log_pin
                                if channel == 'stderr':
                                    row['stderr_bytes'] = log_pin['bytes']
                except (OSError, RuntimeError) as error:
                    receipt['status'] = 'FAILED_NOT_ADMITTED'
                    receipt['log_preservation_error'] = {'type': type(error).__name__, 'message': str(error)[:512]}
                    break
        if receipt['status'] == success_status:
            receipt['regenerable_cleanup'] = {'journal': 'compile-cleanup.json',
                'acceptance': 'terminal journal is required separately from build success'}
        write_json(artifact / 'build-receipt.json', receipt)
        require(tree_size(artifact) <= ARTIFACT_CAP, 'Artifact512MiBCap')
        if receipt['status'] == success_status:
            try:
                if loader_only:
                    cleanup_compiler_outputs(root, artifact, receipt, success_status=success_status,
                                             bundle_filename=bundle_filename)
                else:
                    cleanup_compiler_outputs(root, artifact, receipt)
            except (OSError, RuntimeError) as error:
                receipt['status'] = 'FAILED_NOT_ADMITTED'
                receipt['cleanup_error'] = {'type': type(error).__name__, 'message': str(error)[:512]}
                write_json(artifact / 'cleanup-failure.json', {
                    'status': receipt['status'], 'cleanup_error': receipt['cleanup_error']})
        if os.environ.get('GITHUB_OUTPUT'):
            with open(os.environ['GITHUB_OUTPUT'], 'a', encoding='utf-8') as output:
                output.write('artifact_ready=true\n')
    print(json.dumps({'status': receipt['status'], 'artifact': str(artifact), 'source_commit': expected_head}))
    return 0 if receipt['status'] == success_status else 1


if __name__ == '__main__':
    if sys.argv[1:2] == ['--internal-command']:
        require(sys.platform == 'linux' and bool(sys.argv[2:]), 'LinuxOwnedInternalCommand')
        sys.exit(stream_child(sys.argv[2:]))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--expected-head', required=True)
    parser.add_argument('--timeout', type=int, default=2700)
    parser.add_argument('--profile', choices=('native-validation', 'runtime-loader-only'), default='native-validation')
    args = parser.parse_args()
    sys.exit(run(args.source_root, args.output_root, args.expected_head, args.timeout, profile=args.profile))
