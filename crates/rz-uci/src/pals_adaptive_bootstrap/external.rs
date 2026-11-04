//! V5 external CPU admission from the parent's already pinned snapshots.
//! Loading never starts a process or rewrites the profile's raw/canonical
//! identity. Only effective executable/cwd paths relocate into this attempt.

#[cfg(target_os = "linux")]
#[allow(dead_code)]
pub(crate) mod system_v6;

use crate::pals_adaptive_startup::PinnedPalsStartupV5;
use crate::pals_checker_profile::LoadedPalsAdaptiveCheckerProfileV5;
#[cfg(target_os = "linux")]
use crate::pals_checker_profile::PalsCheckerProfile;
#[cfg(target_os = "linux")]
use rz_experiments::PalsStartupAssetPurposeV5;
use rz_experiments::{PalsCpuSelectionV5, PalsEngineModeV5};
use rz_search::cpu_checker::CheckerError;
use rz_search::external_cpu::{
    CpuEndpoint, CpuEndpointSelection, CpuExecutionUse, ExternalCpuConfig, ExternalLaunchPin,
    ExternalSlicePolicy,
};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExternalSnapshotRelocationV5 {
    pub declared_program: PathBuf,
    pub declared_working_directory: PathBuf,
    pub effective_program: PathBuf,
    pub effective_working_directory: PathBuf,
    pub profile_file_sha256: String,
    pub profile_canonical_sha256: String,
    pub binary_sha256: String,
    /// This bridge admitted declarations/bytes only. Worker startup separately
    /// proves executable pinning, UCI identity, options and readiness.
    pub process_start_performed: bool,
}

pub struct ExternalBootstrapAdmissionV5 {
    profile: LoadedPalsAdaptiveCheckerProfileV5,
    config: ExternalCpuConfig,
    selection: CpuEndpointSelection,
    slice: ExternalSlicePolicy,
    relocation: ExternalSnapshotRelocationV5,
    launch_pin: ExternalLaunchPin,
    #[cfg(target_os = "linux")]
    _private_root: cap_std::fs::Dir,
    #[cfg(target_os = "linux")]
    _runtime_directory: cap_std::fs::Dir,
}
impl ExternalBootstrapAdmissionV5 {
    pub fn profile(&self) -> &LoadedPalsAdaptiveCheckerProfileV5 {
        &self.profile
    }
    pub fn config(&self) -> &ExternalCpuConfig {
        &self.config
    }
    pub fn selection(&self) -> CpuEndpointSelection {
        self.selection
    }
    pub fn slice_policy(&self) -> ExternalSlicePolicy {
        self.slice
    }
    pub fn relocation(&self) -> &ExternalSnapshotRelocationV5 {
        &self.relocation
    }
    pub fn launch_pin(&self) -> ExternalLaunchPin {
        self.launch_pin.clone()
    }
}

/// Keep this admission alongside the startup owner for the selected process
/// lifetime. A absent external endpoint admits no external files or fallback.
pub fn load_external_cpu_v5(
    startup: &PinnedPalsStartupV5,
    until: Instant,
    cancel: &AtomicBool,
) -> Result<Option<ExternalBootstrapAdmissionV5>, CheckerError> {
    require_admission_window(until, cancel)?;
    startup
        .config
        .validate_against(&startup.lock)
        .map_err(|_| fail("startup_registration_mismatch"))?;
    let selection = endpoint_selection(startup)?;
    let Some(external) = startup.config.endpoint.cpu.external.as_ref() else {
        return Ok(None);
    };
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (external, selection);
        Err(CheckerError::Unsupported(
            "adaptive external CPU snapshot execution requires Linux",
        ))
    }
    #[cfg(target_os = "linux")]
    {
        use crate::pals_checker_profile::MAX_CHECKER_PROFILE_BYTES;
        use std::time::Duration;
        let profile_asset = startup
            .asset(PalsStartupAssetPurposeV5::ExternalProfile)
            .map_err(|_| fail("profile_snapshot_missing"))?;
        let binary_asset = startup
            .asset(PalsStartupAssetPurposeV5::ExternalBinary)
            .map_err(|_| fail("binary_snapshot_missing"))?;
        if profile_asset.declaration.artifact != external.profile
            || binary_asset.declaration.artifact != external.binary
        {
            return Err(fail("external_asset_registration_mismatch"));
        }
        let private_root = startup
            .private_root_capability()
            .map_err(|_| fail("private_root_capability_unavailable"))?;
        let raw = verify_held_snapshot(
            &private_root,
            profile_asset,
            MAX_CHECKER_PROFILE_BYTES as u64,
            true,
            until,
            cancel,
        )?;
        let profile =
            PalsCheckerProfile::from_verified_adaptive_bytes(&raw, &external.profile.sha256)?;
        let declared = profile.profile();
        if declared.identity.binary_sha256 != external.binary.sha256
            || declared.identity.declared_license != external.license
            || declared.identity.options != external.options
            || external.version != "Stockfish 19"
            || declared.identity.declared_version != "19"
            || !declared
                .identity
                .declared_source
                .ends_with(&external.source_commit)
            || !declared.arguments.is_empty()
            || !declared.identity.assets.is_empty()
        {
            return Err(fail("profile_and_endpoint_meaning_differ"));
        }
        verify_held_snapshot(
            &private_root,
            binary_asset,
            256 * 1024 * 1024,
            false,
            until,
            cancel,
        )?;
        let binary = binary_asset
            .file
            .try_clone()
            .map_err(|_| fail("binary_descriptor_retention_failed"))?;
        verify_native_snapshot_header(&binary)?;
        let runtime_directory = startup
            .runtime_directory_capability()
            .map_err(|_| fail("runtime_directory_capability_unavailable"))?;
        let launch_pin = ExternalLaunchPin::from_owned_descriptors(
            binary,
            runtime_directory
                .try_clone()
                .map_err(|_| fail("runtime_directory_pin_retention_failed"))?
                .into_std_file(),
        )?;
        require_admission_window(until, cancel)?;
        let mut config = profile.adaptive_config()?;
        let declared_program = config.program.clone();
        let declared_working_directory = config.working_directory.clone();
        config.program = startup
            .private_cwd()
            .join(&binary_asset.declaration.snapshot_path);
        config.working_directory = startup
            .private_cwd()
            .join(&startup.config.runtime_directory);
        config.validate()?;
        let slice = ExternalSlicePolicy {
            quantum: Duration::from_millis(external.quantum_min_ms),
            stop_drain_reserve: Duration::from_millis(external.stop_drain_reserve_ms),
            requested_reported_nodes: external.requested_nodes_max,
            multipv: 1,
        };
        slice.validate(1)?;
        if external.conditional_multipv_max != 4
            || config.stop_grace != slice.stop_drain_reserve
            || config.max_task_wall_time < slice.quantum + slice.stop_drain_reserve
            || external.reported_nodes_are_native_credit
            || !external.stop_bestmove_readyok_before_next_go
            || external.exact_stack_resume
            || external.sigstop
        {
            return Err(fail("slice_or_foreign_credit_semantics_differ"));
        }
        let relocation = ExternalSnapshotRelocationV5 {
            declared_program,
            declared_working_directory,
            effective_program: config.program.clone(),
            effective_working_directory: config.working_directory.clone(),
            profile_file_sha256: profile.file_sha256().to_owned(),
            profile_canonical_sha256: profile.canonical_sha256().to_owned(),
            binary_sha256: config.identity.binary_sha256.clone(),
            process_start_performed: false,
        };
        Ok(Some(ExternalBootstrapAdmissionV5 {
            profile,
            config,
            selection,
            slice,
            relocation,
            launch_pin,
            _private_root: private_root,
            _runtime_directory: runtime_directory,
        }))
    }
}

fn require_admission_window(until: Instant, cancel: &AtomicBool) -> Result<(), CheckerError> {
    if cancel.load(Ordering::Acquire) {
        Err(fail("original_startup_canceled"))
    } else if Instant::now() >= until {
        Err(fail("original_startup_expired"))
    } else {
        Ok(())
    }
}

fn endpoint_selection(startup: &PinnedPalsStartupV5) -> Result<CpuEndpointSelection, CheckerError> {
    let endpoint = &startup.config.endpoint;
    let select = |kind| match kind {
        PalsCpuSelectionV5::Disabled => CpuEndpoint::Disabled,
        PalsCpuSelectionV5::Own => CpuEndpoint::Own,
        PalsCpuSelectionV5::Stockfish19Full => CpuEndpoint::ExternalUci,
    };
    let selection = CpuEndpointSelection {
        execution_use: match endpoint.mode {
            PalsEngineModeV5::Collection => CpuExecutionUse::Collection,
            PalsEngineModeV5::Deployment => CpuExecutionUse::Deployment,
        },
        cpu_t: select(endpoint.cpu.cpu_t),
        cpu_r: select(endpoint.cpu.cpu_r),
        load_v: endpoint.v_load,
    };
    selection.validate()?;
    match endpoint.mode {
        PalsEngineModeV5::Collection => selection.validate_generation_collection()?,
        PalsEngineModeV5::Deployment
            if endpoint.v_generate || endpoint.v_load || endpoint.v_call =>
        {
            return Err(fail("deployment_cannot_generate_load_or_call_v"));
        }
        PalsEngineModeV5::Deployment => {}
    }
    Ok(selection)
}

fn fail(code: &'static str) -> CheckerError {
    CheckerError::External {
        stage: "adaptive_external_bootstrap",
        code,
    }
}

#[cfg(target_os = "linux")]
fn same_metadata(a: &std::fs::Metadata, b: &std::fs::Metadata) -> bool {
    use std::os::unix::fs::MetadataExt;
    a.is_file()
        && b.is_file()
        && a.dev() == b.dev()
        && a.ino() == b.ino()
        && a.len() == b.len()
        && a.mode() == b.mode()
        && a.mtime() == b.mtime()
        && a.mtime_nsec() == b.mtime_nsec()
        && a.ctime() == b.ctime()
        && a.ctime_nsec() == b.ctime_nsec()
}

#[cfg(target_os = "linux")]
fn verify_held_snapshot(
    root: &cap_std::fs::Dir,
    asset: &crate::pals_adaptive_startup::PinnedStartupAssetV5,
    maximum: u64,
    retain_bytes: bool,
    until: Instant,
    cancel: &AtomicBool,
) -> Result<Vec<u8>, CheckerError> {
    use cap_fs_ext::{DirExt, FollowSymlinks, OpenOptionsFollowExt};
    use cap_std::fs::{OpenOptions, OpenOptionsExt};
    use sha2::{Digest as _, Sha256};
    use std::os::unix::fs::FileExt;
    require_admission_window(until, cancel)?;
    let before = asset
        .file
        .metadata()
        .map_err(|_| fail("held_snapshot_metadata_unavailable"))?;
    if !before.is_file()
        || before.len() == 0
        || before.len() > maximum
        || before.len() != asset.declaration.artifact.bytes
    {
        return Err(fail("held_snapshot_size_or_kind_mismatch"));
    }
    let (parent, name) = asset
        .declaration
        .snapshot_path
        .rsplit_once('/')
        .ok_or_else(|| fail("snapshot_path_parent_missing"))?;
    let mut directory = root
        .try_clone()
        .map_err(|_| fail("snapshot_root_retention_failed"))?;
    for part in parent.split('/') {
        directory = directory
            .open_dir_nofollow(part)
            .map_err(|_| fail("snapshot_parent_changed_or_linked"))?;
    }
    let mut options = OpenOptions::new();
    options
        .read(true)
        .follow(FollowSymlinks::No)
        .custom_flags(nix::libc::O_NONBLOCK | nix::libc::O_NOFOLLOW);
    let current = directory
        .open_with(name, &options)
        .map_err(|_| fail("snapshot_path_changed_or_linked"))?
        .into_std();
    if !same_metadata(
        &before,
        &current
            .metadata()
            .map_err(|_| fail("snapshot_path_metadata_unavailable"))?,
    ) {
        return Err(fail("snapshot_path_no_longer_names_held_descriptor"));
    }
    let mut hash = Sha256::new();
    let mut offset = 0;
    let mut buffer = [0_u8; 65_536];
    let mut retained = if retain_bytes {
        Vec::with_capacity(before.len() as usize)
    } else {
        Vec::new()
    };
    while offset < before.len() {
        require_admission_window(until, cancel)?;
        let limit = usize::try_from((before.len() - offset).min(buffer.len() as u64))
            .map_err(|_| fail("snapshot_read_size_overflow"))?;
        let count = asset
            .file
            .read_at(&mut buffer[..limit], offset)
            .map_err(|_| fail("held_snapshot_pread_failed"))?;
        if count == 0 {
            return Err(fail("held_snapshot_changed_during_read"));
        }
        hash.update(&buffer[..count]);
        if retain_bytes {
            retained.extend_from_slice(&buffer[..count]);
        }
        offset += count as u64;
    }
    let after = asset
        .file
        .metadata()
        .map_err(|_| fail("held_snapshot_recheck_failed"))?;
    require_admission_window(until, cancel)?;
    let path_after = directory
        .open_with(name, &options)
        .map_err(|_| fail("snapshot_path_changed_after_read"))?
        .into_std()
        .metadata()
        .map_err(|_| fail("snapshot_path_recheck_failed"))?;
    if !same_metadata(&before, &after)
        || !same_metadata(&after, &path_after)
        || format!("{:x}", hash.finalize()) != asset.declaration.artifact.sha256
    {
        return Err(fail("held_snapshot_changed_or_hash_mismatch"));
    }
    Ok(retained)
}

#[cfg(target_os = "linux")]
fn verify_native_snapshot_header(file: &std::fs::File) -> Result<(), CheckerError> {
    use std::os::unix::fs::{FileExt, MetadataExt};
    let mut header = [0; 4];
    file.read_exact_at(&mut header, 0)
        .map_err(|_| fail("external_snapshot_header_unreadable"))?;
    if header != *b"\x7fELF"
        || file
            .metadata()
            .map_err(|_| fail("external_snapshot_mode_unavailable"))?
            .mode()
            & 0o111
            == 0
    {
        return Err(fail("external_snapshot_requires_executable_native_elf"));
    }
    // The native owner repeats byte/ELF pinning at its explicit start boundary.
    Ok(())
}

#[cfg(test)]
mod tests;
