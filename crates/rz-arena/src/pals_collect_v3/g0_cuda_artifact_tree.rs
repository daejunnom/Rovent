//! G0의 검증된 CUDA cache 소유권을 process와 staging 감시에 연결한다.
//! 일반 감시 정책이나 직렬화된 입력에서 이 capability를 발급하지 않는다.

use super::{numerical_process as numerical, training_process as common};
use crate::{
    ArenaError,
    process::{DelegatedArtifactTreeWatch, DelegatedTreeSnapshot},
};
use common::{LinuxHeld, ReadCredit};
use rz_eval::{
    error::{BackendError, FailureKind, FailureStage},
    runtime_pin::{
        CudaRuntimeBundleSpec, PreparedCudaArtifactTree, RuntimeBundleFileRole, RuntimeCache,
    },
};
use serde_json::Value;
use std::{fs::File, path::Path, sync::atomic::AtomicBool, time::Instant};

fn invalid(message: &str) -> ArenaError {
    ArenaError::Invalid(message.into())
}

/// V5 CUDA captures identify a private bundle and require copying it.  A
/// System CUDA execution needs a separate lock and child receipt; it must not
/// reinterpret an existing V5 capture or silently choose another backend.
pub(super) fn legacy_cuda_execution_disabled() -> ArenaError {
    invalid(
        "G0 legacy CUDA bundle execution is disabled; system CUDA execution is not yet connected",
    )
}

fn backend_error(error: BackendError) -> ArenaError {
    // Keep the typed I/O/resource distinction and the bounded static stage,
    // without exposing native diagnostics or a private path.
    let detail = format!("G0 CUDA cache: {}", error.detail);
    match error.kind {
        FailureKind::Io => ArenaError::Io(detail),
        FailureKind::ResourceExhausted => ArenaError::Budget(detail),
        _ => ArenaError::Invalid(detail),
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn supervise_verified(
    program: &File,
    args: &[std::ffi::OsString],
    directory: &File,
    limits: crate::process::ProcessLimits,
    cancel: Option<&AtomicBool>,
    watch: &crate::process::OwnedArtifactTreeWatch,
    input: crate::process::OriginalProcessInput<'_>,
    verify: &mut dyn FnMut(&File) -> Result<(), ()>,
    tree: Option<&PreparedCudaArtifactTree>,
) -> Result<crate::process::OriginalProcessOutput, ArenaError> {
    if let Some(tree) = tree {
        crate::process::supervise_verified_input_with_delegated_tree_in_original_window(
            program, args, directory, limits, cancel, watch, input, verify, tree,
        )
    } else {
        crate::process::supervise_verified_input_in_original_window(
            program, args, directory, limits, cancel, watch, input, verify,
        )
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn supervise_numerical(
    program: &File,
    args: &[std::ffi::OsString],
    directory: &File,
    limits: crate::process::ProcessLimits,
    cancel: Option<&AtomicBool>,
    watch: &crate::process::OwnedArtifactTreeWatch,
    input: crate::process::OriginalProcessInput<'_>,
    verify: &mut dyn FnMut(&File) -> Result<(), ()>,
    environment: crate::process::OriginalNumericalEnvironment,
    tree: Option<&PreparedCudaArtifactTree>,
) -> Result<crate::process::OriginalProcessOutput, ArenaError> {
    if let Some(tree) = tree {
        crate::process::supervise_numerical_input_with_delegated_tree_in_original_window(
            program,
            args,
            directory,
            limits,
            cancel,
            watch,
            input,
            verify,
            environment,
            tree,
        )
    } else {
        crate::process::supervise_numerical_input_in_original_window(
            program,
            args,
            directory,
            limits,
            cancel,
            watch,
            input,
            verify,
            environment,
        )
    }
}

impl DelegatedArtifactTreeWatch for PreparedCudaArtifactTree {
    fn matches_directory(&self, directory: &File) -> Result<bool, &'static str> {
        PreparedCudaArtifactTree::matches_directory(self, directory)
            .map_err(|_| "g0.cuda_cache.original_directory_changed")
    }

    fn observe_directory(&self, directory: &File) -> Result<DelegatedTreeSnapshot, &'static str> {
        if !PreparedCudaArtifactTree::matches_directory(self, directory)
            .map_err(|_| "g0.cuda_cache.original_directory_changed")?
        {
            return Err("g0.cuda_cache.wrong_directory");
        }
        let observed = self
            .observe()
            .map_err(|_| "g0.cuda_cache.artifacts_changed")?;
        Ok(DelegatedTreeSnapshot {
            bytes: observed.bytes(),
            files: observed.files(),
            directories: observed.directories(),
            entries: observed.entries(),
            max_file_bytes: observed.max_file_bytes(),
            depth: observed.depth(),
        })
    }
}

/// Only already-held immutable input pins may prepare a cache. The original
/// credit/deadline is charged before every bounded read, including EOF probes.
/// This performs no DSO loading, provider setup or neural inference.
pub(super) fn prepare(
    held: &mut LinuxHeld,
    runtime_cache: &Path,
    runtime_library: &Path,
    cuda_manifest: Option<&Path>,
    output_cap: u64,
    credit: &mut ReadCredit<'_>,
) -> Result<(), ArenaError> {
    let Some(manifest_path) = cuda_manifest else {
        if held.cuda_artifact_tree.is_some() {
            return Err(invalid("G0 CPU cannot inherit a CUDA artifact capability"));
        }
        return credit.take(0);
    };
    if held.cuda_artifact_tree.is_some() {
        return Err(invalid(
            "G0 CUDA cache must be prepared once by its original owner",
        ));
    }
    credit.take(0)?;
    let mut manifests = held
        .inputs
        .iter()
        .filter(|input| input.pin.path == manifest_path);
    let manifest = manifests
        .next()
        .ok_or_else(|| invalid("G0 retained CUDA manifest absent"))?;
    if manifests.next().is_some()
        || manifest.pin.bytes > 64 << 10
        || common::same_named(&manifest.pin, &manifest.file)? != manifest.identity
    {
        return Err(invalid("G0 original CUDA manifest owner differs"));
    }
    let raw = common::verify_bytes(&manifest.file, &manifest.pin, true, credit)?;
    let spec = CudaRuntimeBundleSpec::from_json(
        std::str::from_utf8(&raw).map_err(|_| invalid("G0 CUDA manifest UTF-8"))?,
    )
    .map_err(|_| invalid("G0 CUDA manifest validation"))?;
    // The manifest is an independent held input and may be beside libraries/.
    // Match the native consumer: DSO basenames belong to the core's parent.
    let source_root = runtime_library
        .parent()
        .ok_or_else(|| invalid("G0 CUDA runtime library source parent"))?;
    let mut original_files = Vec::with_capacity(spec.files.len());
    let mut incoming = 0u64;
    let mut core = false;
    for file in &spec.files {
        let path = source_root.join(&file.filename);
        let mut inputs = held.inputs.iter().filter(|input| input.pin.path == path);
        let input = inputs
            .next()
            .ok_or_else(|| invalid("G0 declared CUDA file is not retained"))?;
        if inputs.next().is_some()
            || input.pin.bytes != file.bytes
            || input.pin.sha256 != file.sha256
            || common::same_named(&input.pin, &input.file)? != input.identity
        {
            return Err(invalid("G0 CUDA source pin or original inode differs"));
        }
        if file.role == RuntimeBundleFileRole::Core {
            core = path == runtime_library;
        }
        incoming = incoming
            .checked_add(file.bytes)
            .ok_or_else(|| invalid("G0 CUDA size overflow"))?;
        original_files.push(input);
    }
    if !core {
        return Err(invalid(
            "G0 CUDA core differs from the original runtime library",
        ));
    }
    for alias in spec.loader_aliases.iter().flatten() {
        incoming = incoming
            .checked_add(alias.target.len() as u64)
            .ok_or_else(|| invalid("G0 CUDA alias size overflow"))?;
    }
    // No child is running yet. Even the temporary copy must fit the same
    // registered staging ceiling; no skipped prefix or uncharged second root.
    numerical::ensure_output_credit(
        &held.staging,
        output_cap,
        incoming,
        credit.until,
        credit.cancel,
    )?;
    let cache = RuntimeCache::open(runtime_cache).map_err(backend_error)?;
    let mut original_error = None;
    let prepared = cache.prepare_cuda_artifact_tree(source_root, &spec, &mut |bytes| {
        credit.take(bytes).map_err(|error| {
            original_error = Some(error);
            BackendError::new(
                FailureKind::ResourceExhausted,
                FailureStage::Admission,
                "G0 original read/deadline/cancel credit refused",
            )
        })
    });
    if let Some(error) = original_error {
        return Err(error);
    }
    let prepared = prepared.map_err(backend_error)?;
    // Retain the new owner before post-preparation checks can fail. Failure
    // custody must keep these file/alias owners alive as well.
    held.cuda_artifact_tree = Some(prepared);
    for input in original_files {
        if common::same_named(&input.pin, &input.file)? != input.identity {
            return Err(invalid("G0 CUDA input changed during cache preparation"));
        }
    }
    credit.take(0)?;
    held.cuda_artifact_tree_observation = Some(child_observation(
        runtime_cache,
        &spec,
        held.cuda_artifact_tree
            .as_ref()
            .expect("retained CUDA owner"),
    )?);
    credit.take(0)?;
    ensure_output_credit(held, output_cap, 0, credit.until, credit.cancel)
}

/// Bounded child-side accounting metadata, sealed by the existing original
/// startup SHA. Only `tree` authorizes selection; this value is never decoded
/// into a Rust capability. The child's separate quota checker validates it.
fn child_observation(
    runtime_cache: &Path,
    spec: &CudaRuntimeBundleSpec,
    tree: &PreparedCudaArtifactTree,
) -> Result<Value, ArenaError> {
    use std::os::unix::{fs::MetadataExt, io::AsRawFd};
    tree.observe()
        .map_err(|_| invalid("G0 prepared CUDA tree changed before child observation"))?;
    let parent = common::open_directory(runtime_cache)?;
    let parent_file = common::directory_file(&parent)?;
    let mut matched = None;
    let mut entries = 0usize;
    for entry in parent
        .entries()
        .map_err(|_| invalid("G0 CUDA cache parent entries"))?
    {
        entries += 1;
        if entries > 64 {
            return Err(invalid("G0 CUDA cache parent entry cap"));
        }
        let entry = entry.map_err(|_| invalid("G0 CUDA cache parent entry"))?;
        let path = runtime_cache.join(entry.file_name());
        if !std::fs::symlink_metadata(&path)
            .map_err(|_| invalid("G0 CUDA entry metadata"))?
            .is_dir()
        {
            continue;
        }
        let directory = common::open_directory(&path)?;
        let file = common::directory_file(&directory)?;
        if tree
            .matches_directory(&file)
            .map_err(|_| invalid("G0 CUDA entry owner changed"))?
            && matched.replace((path, file)).is_some()
        {
            return Err(invalid("G0 CUDA duplicate cache directory"));
        }
    }
    let (path, file) = matched.ok_or_else(|| invalid("G0 CUDA exact prepared entry absent"))?;
    let directory_value = |path: &Path, file: &File| -> Result<Value, ArenaError> {
        let m = file
            .metadata()
            .map_err(|_| invalid("G0 CUDA directory observation"))?;
        Ok(
            serde_json::json!({"path":path,"device":m.dev(),"inode":m.ino(),"mode":m.mode() & 0o7777}),
        )
    };
    let entry_fd_path = std::path::PathBuf::from(format!("/proc/self/fd/{}", file.as_raw_fd()));
    let metadata_value = |filename: &str| -> Result<Value, ArenaError> {
        let m = std::fs::symlink_metadata(entry_fd_path.join(filename))
            .map_err(|_| invalid("G0 CUDA descendant observation"))?;
        let nanoseconds = |s: i64, ns: i64| {
            s.checked_mul(1_000_000_000)
                .and_then(|s| s.checked_add(ns))
                .ok_or_else(|| invalid("G0 CUDA metadata time overflow"))
        };
        Ok(
            serde_json::json!({"filename":filename,"device":m.dev(),"inode":m.ino(),"bytes":m.len(),
            "mode":m.mode() & 0o7777,"nlink":m.nlink(),
            "mtime_ns":nanoseconds(m.mtime(),m.mtime_nsec())?,"ctime_ns":nanoseconds(m.ctime(),m.ctime_nsec())?}),
        )
    };
    let files = spec
        .files
        .iter()
        .map(|f| metadata_value(&f.filename))
        .collect::<Result<Vec<_>, _>>()?;
    let aliases = spec
        .loader_aliases
        .iter()
        .flatten()
        .map(|a| {
            let mut value = metadata_value(&a.filename)?;
            value["target"] = Value::String(a.target.clone());
            Ok(value)
        })
        .collect::<Result<Vec<_>, ArenaError>>()?;
    let result = serde_json::json!({"schema":"rz-pals-prepared-cuda-artifact-tree-observation/1",
        "parent":directory_value(runtime_cache,&parent_file)?,"directory":directory_value(&path,&file)?,
        "files":files,"aliases":aliases});
    tree.observe()
        .map_err(|_| invalid("G0 prepared CUDA tree changed during child observation"))?;
    Ok(result)
}

pub(super) fn child_observation_for_dispatch(
    held: &LinuxHeld,
) -> Result<Option<Value>, ArenaError> {
    match (
        &held.cuda_artifact_tree,
        &held.cuda_artifact_tree_observation,
    ) {
        (None, None) => Ok(None),
        (Some(tree), Some(observation)) => {
            tree.observe()
                .map_err(|_| invalid("G0 prepared CUDA tree changed before child dispatch"))?;
            Ok(Some(observation.clone()))
        }
        _ => Err(invalid(
            "G0 CUDA owner and child observation custody differ",
        )),
    }
}

pub(super) fn ensure_output_credit(
    held: &LinuxHeld,
    cap: u64,
    additional: u64,
    until: Instant,
    cancel: &AtomicBool,
) -> Result<(), ArenaError> {
    super::g0_numerical_process::linux::ensure_output_credit_delegated(
        &held.staging,
        cap,
        additional,
        held.cuda_artifact_tree
            .as_ref()
            .map(|tree| tree as &dyn DelegatedArtifactTreeWatch),
        until,
        cancel,
    )
}

pub(super) fn scan_staging(
    held: &LinuxHeld,
    cap: u64,
    until: Instant,
    cancel: &AtomicBool,
) -> Result<Value, ArenaError> {
    let Some(tree) = &held.cuda_artifact_tree else {
        return serde_json::to_value(super::selfplay_cli::scan_staging(
            &held.staging,
            cap,
            until,
            cancel,
        )?)
        .map_err(|_| invalid("G0 staging observation serialization"));
    };
    let observation =
        super::selfplay_cli::scan_staging_delegated(&held.staging, cap, tree, until, cancel)?;
    let accounted = observation.accounted_bytes()?;
    let delegated = observation.delegated;
    Ok(serde_json::json!({
        "schema": "rz-pals-g0-cuda-staging-observation/1",
        "ordinary": observation.ordinary,
        "delegated": {"bytes": delegated.bytes, "nondirectory_entries": delegated.files,
            "directories": delegated.directories, "entries": delegated.entries,
            "max_file_bytes": delegated.max_file_bytes, "depth": delegated.depth},
        "accounted_bytes": accounted,
        "admission": "retained_original_cuda_cache_fds",
        "serialized_observation_is_not_a_capability": true,
    }))
}
