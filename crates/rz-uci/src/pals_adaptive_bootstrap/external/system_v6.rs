//! V6 external CPU admission using the original pinned startup owner.
//!
//! This layer reads no model or runtime library, launches no process, and
//! establishes no GPU or physical-completion evidence. The external worker
//! separately verifies its executable pin, UCI identity, options and drain.

use super::{fail, require_admission_window, verify_native_snapshot_header};
use crate::pals_adaptive_startup::{PalsSystemSourceReadFailureV6, PinnedPalsSystemStartupV6};
use crate::pals_checker_profile::{
    LoadedPalsAdaptiveCheckerProfileV5, MAX_CHECKER_PROFILE_BYTES, PalsCheckerProfile,
};
use rz_experiments::{
    PalsCpuSelectionV5, PalsEngineModeV5, PalsStartupAssetPurposeV5, PalsSystemEndpointV6,
};
use rz_search::cpu_checker::CheckerError;
use rz_search::external_cpu::{
    CpuEndpoint, CpuEndpointSelection, CpuExecutionUse, ExternalCpuConfig, ExternalLaunchPin,
    ExternalSlicePolicy,
};
use std::{
    path::PathBuf,
    sync::{Arc, atomic::AtomicBool},
    time::{Duration, Instant},
};

/// Preserve original source-read and external-profile failures separately.
#[derive(Debug)]
pub(crate) enum ExternalSystemAdmissionErrorV6 {
    OriginalSource(PalsSystemSourceReadFailureV6),
    External(CheckerError),
}
impl std::fmt::Display for ExternalSystemAdmissionErrorV6 {
    fn fmt(&self, out: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::OriginalSource(error) => write!(out, "original V6 source: {error}"),
            Self::External(error) => write!(out, "external CPU admission: {error}"),
        }
    }
}
impl From<CheckerError> for ExternalSystemAdmissionErrorV6 {
    fn from(value: CheckerError) -> Self {
        Self::External(value)
    }
}
impl From<PalsSystemSourceReadFailureV6> for ExternalSystemAdmissionErrorV6 {
    fn from(value: PalsSystemSourceReadFailureV6) -> Self {
        Self::OriginalSource(value)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ExternalSnapshotRelocationV6 {
    pub declared_program: PathBuf,
    pub declared_working_directory: PathBuf,
    pub effective_program: PathBuf,
    pub effective_working_directory: PathBuf,
    pub profile_file_sha256: String,
    pub profile_canonical_sha256: String,
    pub binary_sha256: String,
    /// This is input admission only, not a process-start or UCI observation.
    pub process_start_performed: bool,
}

/// Retains the original source owner and the exact runtime directory beside
/// the binary FD pin, including across future worker startup and shutdown.
pub(crate) struct ExternalBootstrapAdmissionV6 {
    profile: LoadedPalsAdaptiveCheckerProfileV5,
    config: ExternalCpuConfig,
    selection: CpuEndpointSelection,
    slice: ExternalSlicePolicy,
    relocation: ExternalSnapshotRelocationV6,
    launch_pin: ExternalLaunchPin,
    _original_startup: Arc<PinnedPalsSystemStartupV6>,
    _runtime_directory: cap_std::fs::Dir,
}
impl ExternalBootstrapAdmissionV6 {
    pub(crate) fn profile(&self) -> &LoadedPalsAdaptiveCheckerProfileV5 {
        &self.profile
    }
    pub(crate) fn config(&self) -> &ExternalCpuConfig {
        &self.config
    }
    pub(crate) fn selection(&self) -> CpuEndpointSelection {
        self.selection
    }
    pub(crate) fn slice_policy(&self) -> ExternalSlicePolicy {
        self.slice
    }
    pub(crate) fn relocation(&self) -> &ExternalSnapshotRelocationV6 {
        &self.relocation
    }
    pub(crate) fn launch_pin(&self) -> ExternalLaunchPin {
        self.launch_pin.clone()
    }
}

/// Use the exact external profile and executable previously opened and
/// verified by this V6 owner. No runtime selection or GPU asset is changed.
/// The caller's window is checked before and after each bounded operation;
/// source reads also check the original owner window between chunks. A single
/// blocked kernel read cannot be preempted or claimed to end at either bound.
pub(crate) fn load_external_cpu_v6(
    startup: Arc<PinnedPalsSystemStartupV6>,
    until: Instant,
    cancel: &AtomicBool,
) -> Result<Option<ExternalBootstrapAdmissionV6>, ExternalSystemAdmissionErrorV6> {
    require_admission_window(until, cancel)?;
    startup.verify_source_unchanged()?;
    let endpoint = &startup.lock().config.endpoint;
    let selection = endpoint_selection(endpoint)?;
    let Some(external) = endpoint.cpu.external.as_ref() else {
        return Ok(None);
    };

    let profile_asset = startup
        .asset(PalsStartupAssetPurposeV5::ExternalProfile)
        .map_err(|_| fail("profile_snapshot_missing"))?;
    let binary_asset = startup
        .asset(PalsStartupAssetPurposeV5::ExternalBinary)
        .map_err(|_| fail("binary_snapshot_missing"))?;
    if profile_asset.declaration().artifact != external.profile
        || binary_asset.declaration().artifact != external.binary
    {
        return Err(fail("external_asset_registration_mismatch").into());
    }
    require_admission_window(until, cancel)?;
    let raw = startup.read_asset_exact(
        PalsStartupAssetPurposeV5::ExternalProfile,
        MAX_CHECKER_PROFILE_BYTES,
    )?;
    require_admission_window(until, cancel)?;
    let profile = PalsCheckerProfile::from_verified_adaptive_bytes(&raw, &external.profile.sha256)?;
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
        return Err(fail("profile_and_endpoint_meaning_differ").into());
    }

    // The V6 owner already read and hashed this exact held FD at admission.
    // Clone that descriptor instead of reopening its pathname or rehashing a
    // large binary; the final owner check rechecks its metadata before return.
    require_admission_window(until, cancel)?;
    let binary = binary_asset
        .file()
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
        .join(&binary_asset.declaration().snapshot_path);
    config.working_directory = startup
        .private_cwd()
        .join(&startup.lock().config.runtime_directory);
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
        return Err(fail("slice_or_foreign_credit_semantics_differ").into());
    }
    let relocation = ExternalSnapshotRelocationV6 {
        declared_program,
        declared_working_directory,
        effective_program: config.program.clone(),
        effective_working_directory: config.working_directory.clone(),
        profile_file_sha256: profile.file_sha256().to_owned(),
        profile_canonical_sha256: profile.canonical_sha256().to_owned(),
        binary_sha256: config.identity.binary_sha256.clone(),
        process_start_performed: false,
    };
    startup.verify_source_unchanged()?;
    require_admission_window(until, cancel)?;
    Ok(Some(ExternalBootstrapAdmissionV6 {
        profile,
        config,
        selection,
        slice,
        relocation,
        launch_pin,
        _original_startup: startup,
        _runtime_directory: runtime_directory,
    }))
}

fn endpoint_selection(
    endpoint: &PalsSystemEndpointV6,
) -> Result<CpuEndpointSelection, CheckerError> {
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
