//! Separate Rovent UCI entry. No legacy V5 DTO, command-line attestation or
//! fabricated weights path participates in this source-bound native assembly.

use super::configuration::{
    ExecutionConfiguration, NativeNnuecSelection, ResearchValueConfiguration,
};
use super::external_session::RoventExternalProductStartupOwners;
use super::nnuec_source::LoadedRoventNnuecSource;
use super::process_boundary::{
    RoventProcessBoundaryInputs, RoventProcessBoundaryInputsV2, VerifiedOwnedRoventProcessBoundary,
    VerifiedRoventProcessBoundary, verify_actual_owned_process_boundary,
    verify_actual_owned_process_boundary_v2, verify_actual_process_boundary,
};
use super::product::{
    RoventNativeProductDriver, RoventProductStartupOwners, RoventProductStartupRejection,
    RoventProductStorageStartupRejection,
};
use super::{
    NativeStartupDiagnostic, NativeStartupDiagnosticPhase, NativeStartupDiagnosticStage,
    PreparedRoventNative, RoventCollectionPreloadSources, RoventExternalRAssetSelection,
    RoventExternalRSourceSelection, RoventNativeAssetSelection, RoventNativeGraphAsset,
    RoventNativeResources, RoventNativeService, RoventNativeStartupRejection,
    RoventPinnedExternalRSource, RoventPinnedNativeSources, RoventStartupControl,
};
use crate::pals_adaptive::NativeQuestionProjection;
use crate::pals_adaptive_bootstrap::protocol::serve_driver;
use crate::pals_native::ticketed::{TicketNativeEvent, VirtualReviewCollectionObserver};
use crate::rovent_root_admission::RoventRootAdmissionPolicy;
use crate::search_driver::SearchSessionDriver;
use crate::{
    EngineIdentity,
    engine::{EngineSettings, OwnerRegistry, ProcessClock},
};
use cap_fs_ext::DirExt;
use cap_std::fs::Dir;
use rz_contracts::ProcessEpoch;
use rz_eval::pals_onnx::virtual_review::VirtualReviewProfileDirectory;
use rz_experiments::{ArtifactRef, RoventDevice, RoventInputLock, RoventInputLockV2};
use rz_search::pals::engine::{adaptive_port::AdaptivePortQuestion, pc_port::PcPortQuestion};
use serde::Deserialize;
use std::{
    cell::Cell,
    error::Error,
    fmt,
    fs::File,
    io::Read,
    path::{Component, PathBuf},
    sync::{Arc, atomic::AtomicBool},
    time::{Duration, Instant},
};

const SELECTION_DOMAIN: &str = "rz-rovent-native-assets/1";
const EXTERNAL_SELECTION_DOMAIN: &str = "rz-rovent-external-r-sources/1";
const MAX_LOCK_BYTES: u64 = 1 << 20;
const MAX_SELECTION_BYTES: u64 = 64 << 10;

mod cohort_prepare;
mod collection;
mod evidence;
pub mod storage;
mod v2_external;
mod v2_rejections;
pub use cohort_prepare::{
    EXPECTED_UTILITY_COHORT_PREPARATION, ExpectedUtilityCohortPreparation,
    prepare_expected_utility_cohort_source_pins,
};
use collection::CollectionLaunch;
pub use collection::{
    RoventCollectionPreparedLaunch, RoventCollectionPreparedNative, run_collection,
    run_collection_with_driver, run_collection_with_resource_admission,
};
pub use evidence::{LaunchRetirementEvidence, LaunchRetirementRegistrationCustody};
use v2_external::{ExternalLaunchV2, open_external_launch_v2};
use v2_rejections::{reject_external_pcr_product_v2, reject_own_nnuec_product_v2};

#[derive(Debug)]
pub struct RoventLaunchError {
    pub stage: &'static str,
    pub detail: String,
}
impl fmt::Display for RoventLaunchError {
    fn fmt(&self, output: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(output, "Rovent {}: {}", self.stage, self.detail)
    }
}
impl Error for RoventLaunchError {}
fn error(stage: &'static str, detail: impl fmt::Display) -> RoventLaunchError {
    let mut detail = detail.to_string();
    if detail.len() > 512 {
        let mut end = 512;
        while !detail.is_char_boundary(end) {
            end -= 1;
        }
        detail.truncate(end);
    }
    RoventLaunchError { stage, detail }
}

#[derive(Debug, Eq, PartialEq)]
struct Arguments {
    input_lock: String,
    native_assets: String,
    external_r_sources: Option<String>,
    retirement_sink: Option<String>,
    evidence_root: Option<PathBuf>,
    process_epoch: u64,
    startup_ms: u64,
    root_policy: RoventRootAdmissionPolicy,
    startup_diagnostics: bool,
    lifecycle_diagnostics: bool,
    lifecycle_diagnostics_v2: bool,
    initial_stop_source_diagnostics: bool,
    storage: Option<StorageArguments>,
}

/// A complete explicit selection, not an archive-on default. The private
/// snapshot root is deliberately absent: only the actual process cwd supplies it.
#[derive(Debug, Eq, PartialEq)]
struct StorageArguments {
    profile: String,
    profile_sha256: String,
    profile_bytes: u64,
    archive_root: PathBuf,
    checkout_root: PathBuf,
    protected_root: PathBuf,
}

impl Arguments {
    fn parse(arguments: &[String]) -> Result<Self, RoventLaunchError> {
        let mut input_lock = None;
        let mut native_assets = None;
        let mut external_r_sources = None;
        let mut retirement_sink = None;
        let mut evidence_root = None;
        let mut process_epoch = None;
        let mut startup_ms = None;
        let mut root_policy = None;
        let mut startup_diagnostics = None;
        let mut lifecycle_diagnostics = None;
        let mut lifecycle_diagnostics_v2 = false;
        let mut initial_stop_source_diagnostics = None;
        let mut storage_profile = None;
        let mut storage_sha256 = None;
        let mut storage_bytes = None;
        let mut archive_root = None;
        let mut storage_checkout_root = None;
        let mut storage_protected_root = None;
        for argument in arguments {
            let (name, value) = argument
                .split_once('=')
                .ok_or_else(|| error("Arguments", "every option requires its explicit =value"))?;
            match name {
                "--input-lock" if input_lock.is_none() => input_lock = Some(private_path(value)?),
                "--native-assets" if native_assets.is_none() => {
                    native_assets = Some(private_path(value)?)
                }
                "--external-r-sources" if external_r_sources.is_none() => {
                    external_r_sources = Some(private_path(value)?)
                }
                "--retirement-sink" if retirement_sink.is_none() => {
                    retirement_sink = Some(private_path(value)?)
                }
                "--evidence-root" if evidence_root.is_none() => {
                    evidence_root = Some(managed_path(value)?)
                }
                "--storage-profile" if storage_profile.is_none() => {
                    storage_profile = Some(private_path(value)?)
                }
                "--storage-profile-sha256" if storage_sha256.is_none() => {
                    if value.len() != 64
                        || !value
                            .bytes()
                            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
                    {
                        return Err(error(
                            "Arguments",
                            "explicit lowercase storage SHA256 required",
                        ));
                    }
                    storage_sha256 = Some(value.to_owned());
                }
                "--storage-profile-bytes" if storage_bytes.is_none() => {
                    storage_bytes = Some(number(value, rz_experiments::MAX_MANIFEST_BYTES as u64)?)
                }
                "--archive-root" if archive_root.is_none() => {
                    archive_root = Some(managed_path(value)?)
                }
                "--storage-checkout-root" if storage_checkout_root.is_none() => {
                    storage_checkout_root = Some(managed_path(value)?)
                }
                "--storage-protected-root" if storage_protected_root.is_none() => {
                    storage_protected_root = Some(managed_path(value)?)
                }
                "--process-epoch" if process_epoch.is_none() => {
                    process_epoch = Some(number(value, u64::MAX)?)
                }
                "--startup-ms" if startup_ms.is_none() => {
                    startup_ms = Some(number(value, 3_600_000)?)
                }
                "--startup-diagnostics" if startup_diagnostics.is_none() => {
                    startup_diagnostics = Some(match value {
                        "stderr-v1" => true,
                        _ => {
                            return Err(error(
                                "Arguments",
                                "unknown explicit startup diagnostic selection",
                            ));
                        }
                    });
                }
                "--lifecycle-diagnostics" if lifecycle_diagnostics.is_none() => {
                    lifecycle_diagnostics = Some(match value {
                        "stderr-v1" => true,
                        "stderr-v2" => {
                            lifecycle_diagnostics_v2 = true;
                            false
                        }
                        _ => {
                            return Err(error(
                                "Arguments",
                                "unknown explicit lifecycle diagnostic selection",
                            ));
                        }
                    });
                }
                "--initial-stop-source-diagnostics"
                    if initial_stop_source_diagnostics.is_none() =>
                {
                    initial_stop_source_diagnostics = Some(match value {
                        "stderr-v1" => true,
                        _ => {
                            return Err(error(
                                "Arguments",
                                "unknown explicit initial-stop source diagnostic selection",
                            ));
                        }
                    });
                }
                "--root-policy" if root_policy.is_none() => {
                    root_policy = Some(match value {
                        "exact-registered" => RoventRootAdmissionPolicy::ExactRegistered,
                        "dynamic-go" => RoventRootAdmissionPolicy::DynamicGo,
                        _ => {
                            return Err(error(
                                "Arguments",
                                "unknown explicit root admission policy",
                            ));
                        }
                    })
                }
                _ => return Err(error("Arguments", "unknown or repeated option")),
            }
        }
        let storage = match (
            storage_profile,
            storage_sha256,
            storage_bytes,
            archive_root,
            storage_checkout_root,
            storage_protected_root,
        ) {
            (None, None, None, None, None, None) => None,
            (
                Some(profile),
                Some(profile_sha256),
                Some(profile_bytes),
                Some(archive_root),
                Some(checkout_root),
                Some(protected_root),
            ) => Some(StorageArguments {
                profile,
                profile_sha256,
                profile_bytes,
                archive_root,
                checkout_root,
                protected_root,
            }),
            _ => {
                return Err(error(
                    "Arguments",
                    "storage requires all six explicit options",
                ));
            }
        };
        if initial_stop_source_diagnostics == Some(true) && lifecycle_diagnostics != Some(true) {
            return Err(error(
                "Arguments",
                "initial-stop source diagnostics require --lifecycle-diagnostics=stderr-v1",
            ));
        }
        Ok(Self {
            input_lock: input_lock.ok_or_else(|| error("Arguments", "--input-lock is required"))?,
            native_assets: native_assets
                .ok_or_else(|| error("Arguments", "--native-assets is required"))?,
            external_r_sources,
            retirement_sink,
            evidence_root,
            process_epoch: process_epoch
                .ok_or_else(|| error("Arguments", "--process-epoch is required"))?,
            startup_ms: startup_ms.ok_or_else(|| error("Arguments", "--startup-ms is required"))?,
            root_policy: root_policy
                .ok_or_else(|| error("Arguments", "--root-policy is required"))?,
            startup_diagnostics: startup_diagnostics.unwrap_or(false),
            lifecycle_diagnostics: lifecycle_diagnostics.unwrap_or(false),
            lifecycle_diagnostics_v2,
            initial_stop_source_diagnostics: initial_stop_source_diagnostics.unwrap_or(false),
            storage,
        })
    }
}
fn private_path(value: &str) -> Result<String, RoventLaunchError> {
    if value.len() > 4096 || !rz_experiments::pals_private_snapshot_path_v5(value) {
        return Err(error(
            "Arguments",
            "input must use the closed inputs snapshot namespace",
        ));
    }
    Ok(value.into())
}
fn managed_path(value: &str) -> Result<PathBuf, RoventLaunchError> {
    let path = PathBuf::from(value);
    if value.is_empty()
        || value.len() > 4096
        || value.contains('\0')
        || !path.is_absolute()
        || path.components().any(|part| part == Component::ParentDir)
    {
        return Err(error(
            "Arguments",
            "explicit absolute managed directory required",
        ));
    }
    Ok(path)
}
fn number(value: &str, maximum: u64) -> Result<u64, RoventLaunchError> {
    if value.is_empty() || !value.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(error("Arguments", "integer syntax"));
    }
    let number: u64 = value
        .parse()
        .map_err(|_| error("Arguments", "integer overflow"))?;
    if number == 0 || number > maximum {
        return Err(error("Arguments", "integer bound"));
    }
    Ok(number)
}

/// The off path still uses the public borrowed verifier. Only explicit storage
/// selection moves the same original objects into its actual owned sibling.
enum LaunchProcessOwners {
    Off(RoventProcessBoundaryInputs),
    Storage {
        boundary: Arc<VerifiedOwnedRoventProcessBoundary>,
        transferred: Cell<bool>,
    },
}
impl LaunchProcessOwners {
    fn verify(
        inputs: RoventProcessBoundaryInputs,
        storage: bool,
    ) -> Result<Self, RoventLaunchError> {
        if storage {
            match verify_actual_owned_process_boundary(inputs) {
                Ok(boundary) => Ok(Self::Storage {
                    boundary,
                    transferred: Cell::new(false),
                }),
                Err(rejected) => {
                    let cause = error("ProcessBoundary", &rejected.error);
                    // No replacement FD/proof is created. Preserve even partial
                    // original custody on this selected-path rejection.
                    let _quarantine = Box::leak(Box::new(rejected));
                    Err(cause)
                }
            }
        } else {
            Ok(Self::Off(inputs))
        }
    }
    fn original_root(&self) -> &Dir {
        match self {
            Self::Off(inputs) => &inputs.original_root,
            Self::Storage { boundary, .. } => boundary.original_root(),
        }
    }
    fn startup_view(&self) -> Result<VerifiedRoventProcessBoundary<'_>, RoventLaunchError> {
        match self {
            Self::Off(inputs) => verify_actual_process_boundary(
                &inputs.lock,
                &inputs.lock.manifest.engine.binary,
                &inputs.original_binary,
                &inputs.private_cwd,
                &inputs.original_root,
                &inputs.control,
            ),
            Self::Storage { boundary, .. } => boundary.startup_view(),
        }
        .map_err(|failure| error("ProcessBoundary", failure))
    }

    /// This only records an actual ownership handoff to the product or its
    /// returned rejection. It never records physical completion or readiness.
    fn transferred_to_product(&self) {
        if let Self::Storage { transferred, .. } = self {
            transferred.set(true);
        }
    }
}
impl Drop for LaunchProcessOwners {
    fn drop(&mut self) {
        if let Self::Storage {
            boundary,
            transferred,
        } = self
        {
            if !transferred.get() {
                // Factory rejection may retain model sources but not this
                // original File/Dir. Keep both until an actual owner adopted
                // them; early errors cannot stand in for a physical fence.
                std::mem::forget(Arc::clone(boundary));
            }
        }
    }
}

/// Startup errors cannot silently release selected storage while a partially
/// prepared product may still own physical work. A successful product attach
/// removes the custody here and adopts the same original owners instead.
struct PendingLaunchStorage<'boundary, 'process> {
    custody: Option<storage::RoventStorageCompiledCustody<'boundary, 'process>>,
    boundary: Arc<VerifiedOwnedRoventProcessBoundary>,
}
impl Drop for PendingLaunchStorage<'_, '_> {
    fn drop(&mut self) {
        if let Some(custody) = self.custody.take() {
            std::mem::forget((
                custody.into_byte_custody().into_owners(),
                Arc::clone(&self.boundary),
            ));
        }
    }
}

fn prepare_launch_storage<'boundary, 'process>(
    selection: Option<&StorageArguments>,
    owners: &LaunchProcessOwners,
    view: &'boundary VerifiedRoventProcessBoundary<'process>,
    lock: &Arc<RoventInputLock>,
    control: &RoventStartupControl,
) -> Result<Option<PendingLaunchStorage<'boundary, 'process>>, RoventLaunchError> {
    let Some(selection) = selection else {
        return Ok(None);
    };
    let LaunchProcessOwners::Storage { boundary, .. } = owners else {
        return Err(error(
            "StorageCustody",
            "original owned process boundary is required",
        ));
    };
    let custody = match storage::open_rovent_storage_byte_custody(
        Arc::clone(lock),
        boundary.original_root(),
        storage::RoventStorageAdmissionRequest {
            profile_path: &selection.profile,
            profile_sha256: &selection.profile_sha256,
            profile_bytes: selection.profile_bytes,
            archive_root: &selection.archive_root,
            checkout_root: &selection.checkout_root,
            private_snapshot_root: boundary.private_cwd(),
            protected_execution_root: &selection.protected_root,
        },
        control.clone(),
    ) {
        Ok(custody) => custody,
        Err(rejected) => {
            let cause = error("StorageCustody", &rejected.error);
            std::mem::forget((rejected.owners, Arc::clone(boundary)));
            return Err(cause);
        }
    };
    match storage::check_rovent_storage_compiled_custody(custody, view) {
        Ok(custody) => Ok(Some(PendingLaunchStorage {
            custody: Some(custody),
            boundary: Arc::clone(boundary),
        })),
        Err(rejected) => {
            let cause = error("StorageCompiledCustody", &rejected.error);
            std::mem::forget((rejected.custody.into_owners(), Arc::clone(boundary)));
            Err(cause)
        }
    }
}

/// A rejected attachment retains its original custody until the real product
/// shutdown. Successful attachment proceeds to the normal UCI owner; setup is
/// still deferred until the first admitted go, before root registration.
fn reject_storage_runtime_registration(
    driver: RoventNativeProductDriver,
    rejected: RoventProductStorageStartupRejection<'_, '_>,
    cleanup_until: Instant,
) -> Result<(), RoventLaunchError> {
    let primary = &rejected.failure;
    let shutdown = driver.finish_session(cleanup_until);
    let result = match shutdown {
        Ok(Some(crate::search_driver::DriverPhysicalCompletion::Confirmed)) => {
            Err(error("StorageRuntimeRegistration", &primary.detail))
        }
        Ok(_) => Err(error(
            "PhysicalCompletionUnknown",
            format!(
                "storage registration failed: {}; original product shutdown is unknown",
                primary.detail
            ),
        )),
        Err(failure) => Err(error(
            "PhysicalCompletionUnknown",
            format!(
                "storage registration failed: {}; original shutdown failed: {}: {}",
                primary.detail, failure.code, failure.detail
            ),
        )),
    };
    // Both retain the same physical-close flag. Unknown Drop quarantines the
    // rejected registration as well as the real product inner/source owners.
    drop(rejected);
    drop(driver);
    result
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SelectionWire {
    domain: String,
    startup_configuration: ArtifactRef,
    export_manifest: ArtifactRef,
    rules_descriptor: ArtifactRef,
    graphs: Vec<GraphWire>,
    cuda_control_inventory: Option<ArtifactRef>,
    runtime_bundle_manifest: Option<ArtifactRef>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct GraphWire {
    role: String,
    registered_file: String,
    artifact: ArtifactRef,
}
impl SelectionWire {
    fn parse(bytes: &[u8]) -> Result<RoventNativeAssetSelection, RoventLaunchError> {
        let wire: Self = serde_json::from_slice(bytes)
            .map_err(|_| error("Assets", "strict native asset selection JSON"))?;
        if wire.domain != SELECTION_DOMAIN || wire.graphs.len() != 2 {
            return Err(error(
                "Assets",
                "native selection domain or graph cardinality",
            ));
        }
        Ok(RoventNativeAssetSelection {
            startup_configuration: wire.startup_configuration,
            export_manifest: wire.export_manifest,
            rules_descriptor: wire.rules_descriptor,
            graphs: wire
                .graphs
                .into_iter()
                .map(|graph| RoventNativeGraphAsset {
                    role: graph.role,
                    registered_file: graph.registered_file,
                    artifact: graph.artifact,
                })
                .collect(),
            cuda_control_inventory: wire.cuda_control_inventory,
            runtime_bundle_manifest: wire.runtime_bundle_manifest,
        })
    }
}

/// A separate source mapping leaves the original native-assets /1 codec
/// unchanged. The mapping itself must be an original pinned provenance input.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ExternalSelectionWire {
    domain: String,
    working_directory_snapshot_path: String,
    assets: Vec<RoventExternalRAssetSelection>,
}
impl ExternalSelectionWire {
    fn parse(bytes: &[u8]) -> Result<RoventExternalRSourceSelection, RoventLaunchError> {
        let wire: Self = serde_json::from_slice(bytes)
            .map_err(|_| error("ExternalSources", "strict external source selection JSON"))?;
        if wire.domain != EXTERNAL_SELECTION_DOMAIN || wire.assets.len() > 32 {
            return Err(error("ExternalSources", "source domain or asset count"));
        }
        Ok(RoventExternalRSourceSelection {
            working_directory_snapshot_path: private_path(&wire.working_directory_snapshot_path)?,
            assets: wire.assets,
        })
    }
}

/// The caller passes only process arguments. No account, token or external
/// credential file is inspected by this native UCI entry.
pub fn run(arguments: Vec<String>) -> Result<(), RoventLaunchError> {
    run_inner(arguments, None, None, None)
}

/// 같은 원 lock으로 자원 admission을 한 번 수행한 뒤 native launch를 잇는다.
/// 실제 OS scope와 실패 시 FD custody는 상위 caller가 전체 실행 동안 보유한다.
/// build-capabilities 조회는 실행하지 않으므로 이 callback도 호출하지 않는다.
pub fn run_with_resource_admission<'a>(
    arguments: Vec<String>,
    admit_resource: impl FnOnce(&Arc<RoventInputLock>) -> Result<(), RoventLaunchError> + 'a,
) -> Result<(), RoventLaunchError> {
    run_inner(arguments, None, Some(Box::new(admit_resource)), None)
}

type ResourceAdmission<'a> =
    Box<dyn FnOnce(&Arc<RoventInputLock>) -> Result<(), RoventLaunchError> + 'a>;

/// The parser retains the actual version and never projects a /2 lock to /1.
#[derive(Debug)]
enum LaunchInputLock {
    V1(Arc<RoventInputLock>),
    V2(Arc<RoventInputLockV2>),
}
fn parse_launch_lock(bytes: &[u8]) -> Result<LaunchInputLock, RoventLaunchError> {
    let input =
        std::str::from_utf8(bytes).map_err(|_| error("InputLock", "UTF-8 input required"))?;
    match RoventInputLock::from_json(input) {
        Ok(lock) => Ok(LaunchInputLock::V1(Arc::new(lock))),
        Err(original_failure) => match RoventInputLockV2::from_json(input) {
            Ok(lock) => Ok(LaunchInputLock::V2(Arc::new(lock))),
            Err(_) => Err(error("InputLock", original_failure)),
        },
    }
}

type ProductRunner<'a> = Box<
    dyn FnOnce(
            RoventNativeProductDriver,
            ProcessClock,
            Arc<OwnerRegistry>,
            EngineSettings,
        ) -> Result<(), RoventLaunchError>
        + 'a,
>;

fn run_inner<'a>(
    arguments: Vec<String>,
    collection: Option<CollectionLaunch<'a>>,
    resource_admission: Option<ResourceAdmission<'a>>,
    product_runner: Option<ProductRunner<'a>>,
) -> Result<(), RoventLaunchError> {
    if arguments == ["--build-capabilities-json"] {
        let sources = super::process_boundary::compiled_process_source_components()
            .map_err(|failure| error("CompiledSources", failure))?;
        println!(
            "{}",
            serde_json::json!({
                "domain":"rz-rovent-build-capabilities/1",
                "source_commit":env!("RZ_PALS_COMPILED_SOURCE_COMMIT"),
                "source_components":sources,
                "target_os":std::env::consts::OS,
                "target_arch":std::env::consts::ARCH,
                "onnx_cpu_compiled":cfg!(feature="onnx-cpu"),
                "onnx_cuda_compiled":cfg!(feature="onnx-cuda"),
                "native_runtime_loaded":false,
                "native_model_loaded":false,
                "neural_execution_observed":false,
            })
        );
        return Ok(());
    }
    let (teacher_selections, collection_observer, collection_assemble) = match collection {
        Some(request) => (
            Some(request.teacher_selections),
            Some(request.observer),
            Some(request.assemble),
        ),
        None => (None, None, None),
    };
    let arguments = Arguments::parse(&arguments)?;
    let cwd = std::env::current_dir().map_err(|failure| error("Cwd", failure))?;
    let root = Dir::open_ambient_dir(&cwd, cap_std::ambient_authority())
        .map_err(|failure| error("Cwd", failure))?;
    let lock_file = root
        .open_with(
            &arguments.input_lock,
            cap_std::fs::OpenOptions::new().read(true),
        )
        .map_err(|failure| error("InputLock", failure))?
        .into_std();
    let lock_bytes = read_bounded(&lock_file, MAX_LOCK_BYTES)?;
    let lock = match parse_launch_lock(&lock_bytes)? {
        LaunchInputLock::V1(lock) => lock,
        LaunchInputLock::V2(lock) => {
            if teacher_selections.is_some()
                || collection_observer.is_some()
                || collection_assemble.is_some()
                || resource_admission.is_some()
                || product_runner.is_some()
            {
                return Err(error(
                    "InputLockUnsupportedV2",
                    "V1-only launch callbacks cannot consume /2",
                ));
            }
            return run_v2_native(arguments, lock, root, cwd, lock_file);
        }
    };
    if arguments.lifecycle_diagnostics_v2 {
        return Err(error(
            "InputLockUnsupportedV1",
            "V2 lifecycle observation requires an original /2 lock",
        ));
    }
    if arguments.startup_ms > lock.manifest.wall_time_max_ms {
        return Err(error(
            "StartupWindow",
            "startup exceeds the original declared window",
        ));
    }
    let begin = Instant::now();
    let diagnostic = NativeStartupDiagnostic::new(arguments.startup_diagnostics, begin);
    let clock = ProcessClock::new(ProcessEpoch(arguments.process_epoch));
    let until = begin
        .checked_add(Duration::from_millis(arguments.startup_ms))
        .ok_or_else(|| error("StartupWindow", "startup deadline overflow"))?;
    let cleanup_until = until
        .checked_add(Duration::from_millis(lock.manifest.cleanup_max_ms))
        .ok_or_else(|| error("StartupWindow", "cleanup deadline overflow"))?;
    let control = RoventStartupControl {
        until,
        cleanup_until,
        cancellation: Arc::new(AtomicBool::new(false)),
    };
    let binary = lock
        .manifest
        .engine
        .binary
        .open_verified_at(&root, 128 << 20)
        .map_err(|failure| error("Binary", failure))?;
    let process_owners = LaunchProcessOwners::verify(
        RoventProcessBoundaryInputs {
            lock: Arc::clone(&lock),
            original_binary: binary,
            original_root: root,
            private_cwd: cwd.clone(),
            control: control.clone(),
        },
        arguments.storage.is_some(),
    )?;
    let root = process_owners.original_root();
    let boundary = process_owners.startup_view()?;
    let mut pending_storage = prepare_launch_storage(
        arguments.storage.as_ref(),
        &process_owners,
        &boundary,
        &lock,
        &control,
    )?;
    if let Some(admit_resource) = resource_admission {
        // This is the same parsed and process-verified source owner. No model,
        // ORT provider, teacher preload, external process or worker exists yet.
        control
            .check()
            .map_err(|failure| error("ResourceAdmission", failure))?;
        diagnostic.trace(NativeStartupDiagnosticStage::ResourceAdmission, || {
            admit_resource(&lock)
        })?;
        control
            .check()
            .map_err(|failure| error("ResourceAdmission", failure))?;
    }
    let selected: Vec<_> = lock
        .manifest
        .provenance
        .iter()
        .filter(|artifact| artifact.path == arguments.native_assets)
        .collect();
    if selected.len() != 1 {
        return Err(error(
            "Assets",
            "selection is not a unique original provenance artifact",
        ));
    }
    let selection_file = selected[0]
        .open_verified_at(root, MAX_SELECTION_BYTES)
        .map_err(|failure| error("Assets", failure))?;
    let selection = SelectionWire::parse(&read_bounded(&selection_file, MAX_SELECTION_BYTES)?)?;
    let collection_preload = teacher_selections
        .map(|selections| {
            RoventCollectionPreloadSources::open(
                Arc::clone(&lock),
                root.try_clone().map_err(|failure| error("Cwd", failure))?,
                &cwd,
                selections,
                &control,
            )
            .map(Arc::new)
            .map_err(|failure| error("CollectionSources", failure))
        })
        .transpose()?;
    let source_root = root.try_clone().map_err(|failure| error("Cwd", failure))?;
    let sources = Arc::new(
        diagnostic
            .trace(
                NativeStartupDiagnosticStage::SourcePins,
                || match &collection_preload {
                    Some(preload) => RoventPinnedNativeSources::open_collection(
                        Arc::clone(&lock),
                        source_root,
                        cwd.clone(),
                        selection,
                        &control,
                        Arc::clone(preload),
                    ),
                    None => RoventPinnedNativeSources::open(
                        Arc::clone(&lock),
                        source_root,
                        cwd.clone(),
                        selection,
                        &control,
                    ),
                },
            )
            .map_err(|failure| error("Assets", failure))?,
    );
    sources
        .configuration()
        .require_supported_native_capabilities()
        .map_err(|failure| error("Capability", failure))?;
    let (external_source, external_selection_file) = match (
        sources.configuration().fields().execution,
        arguments.external_r_sources.as_deref(),
    ) {
        (ExecutionConfiguration::ExternalResearcher { .. }, Some(path)) => {
            // /1 and /2 have no explicit query selection and remain unsupported.
            sources
                .configuration()
                .external_question_limit()
                .map_err(|failure| error("ExternalQuestionPolicy", failure))?
                .ok_or_else(|| {
                    error("ExternalQuestionPolicy", "explicit external query required")
                })?;
            let matching: Vec<_> = lock
                .manifest
                .provenance
                .iter()
                .filter(|artifact| artifact.path == path)
                .collect();
            if matching.len() != 1 {
                return Err(error(
                    "ExternalSources",
                    "selection is not a unique original provenance artifact",
                ));
            }
            let mapping = matching[0]
                .open_verified_at(root, MAX_SELECTION_BYTES)
                .map_err(|failure| error("ExternalSources", failure))?;
            let selected =
                ExternalSelectionWire::parse(&read_bounded(&mapping, MAX_SELECTION_BYTES)?)?;
            let source = RoventPinnedExternalRSource::open(&sources, &selected, &control)
                .map_err(|failure| error("ExternalSources", failure))?;
            (Some(source), Some(mapping))
        }
        (ExecutionConfiguration::ExternalResearcher { .. }, None) => {
            return Err(error(
                "ExternalSources",
                "--external-r-sources is required for the external researcher",
            ));
        }
        (_, Some(_)) => {
            return Err(error(
                "ExternalSources",
                "external source selection requires the external researcher",
            ));
        }
        (_, None) => (None, None),
    };
    let retirement_evidence = match (
        sources.configuration().fields().execution,
        arguments.retirement_sink.as_deref(),
        arguments.evidence_root.as_deref(),
    ) {
        (
            ExecutionConfiguration::ExternalResearcher { .. },
            Some(registration),
            Some(managed_root),
        ) => {
            Some(
                LaunchRetirementEvidence::open(
                    &lock,
                    root,
                    registration,
                    managed_root,
                    &cwd,
                    // This bounds only source admission/FS scanning. Each actual
                    // root supplies its own original finite persistence window.
                    control.until,
                    control.cancellation.as_ref(),
                )
                .map_err(|failure| error("RetirementEvidence", failure))?,
            )
        }
        (ExecutionConfiguration::ExternalResearcher { .. }, _, _) => {
            return Err(error(
                "RetirementEvidence",
                "external researcher requires --retirement-sink and --evidence-root",
            ));
        }
        (_, None, None) => None,
        _ => {
            return Err(error(
                "RetirementEvidence",
                "retirement sink arguments require an external researcher",
            ));
        }
    };
    let cuda_profile = if matches!(lock.manifest.engine.pc.device, RoventDevice::Cuda { .. }) {
        Some(diagnostic.trace(
            NativeStartupDiagnosticStage::CudaProfileDirectory,
            || -> Result<VirtualReviewProfileDirectory, RoventLaunchError> {
                let name = format!(
                    "rovent-cuda-profile-{}-{}",
                    std::process::id(),
                    arguments.process_epoch
                );
                root.create_dir(&name)
                    .map_err(|failure| error("ProfileDirectory", failure))?;
                let directory = root
                    .open_dir_nofollow(&name)
                    .map_err(|failure| error("ProfileDirectory", failure))?;
                VirtualReviewProfileDirectory::from_owned_directory(directory.into_std_file())
                    .map_err(|failure| error("ProfileDirectory", failure))
            },
        )?)
    } else {
        None
    };
    let configuration = sources.configuration();
    let resources = RoventNativeResources {
        runtime: diagnostic
            .trace(NativeStartupDiagnosticStage::RuntimePin, || {
                sources.pin_runtime()
            })
            .map_err(|failure| error("Runtime", failure))?,
        configuration: configuration
            .fields()
            .native
            .onnx(&lock.manifest.engine.pc.device, cuda_profile.is_some())
            .map_err(|failure| error("Configuration", failure))?,
        loading_profile: configuration
            .fields()
            .native
            .loading_profile
            .map(|profile| profile.actual()),
        cuda_profile,
        drain_limit: Duration::from_millis(configuration.fields().native.drain_ms),
        tickets: configuration
            .fields()
            .native
            .tickets(configuration.fields().review.token_budget)
            .map_err(|failure| error("Configuration", failure))?,
        command_capacity: configuration.fields().native.command_capacity,
    };
    control
        .check()
        .map_err(|failure| error("StartupWindow", failure))?;
    let loaded_nnuec =
        if sources.configuration().fields().nnuec_selection() != NativeNnuecSelection::Absent {
            Some(
                diagnostic
                    .trace(NativeStartupDiagnosticStage::NnuecSource, || {
                        LoadedRoventNnuecSource::load(Arc::clone(&sources), &control)
                    })
                    .map_err(|failure| error("NnuecSource", failure))?,
            )
        } else {
            None
        };
    let owners = Arc::new(OwnerRegistry::default());
    let identity = EngineIdentity {
        name: format!("Rovent {:?}", lock.manifest.engine.mode),
        author: "Rovent contributors".into(),
    };
    let prepared_product = if lock.manifest.engine.mode.has_native_r() {
        let prepared = prepare_launch_native::<AdaptivePortQuestion>(
            Arc::clone(&sources),
            resources,
            &clock,
            control.clone(),
            collection_observer,
            &diagnostic,
        )
        .map_err(|failure| reject_factory(failure, cleanup_until, &diagnostic))?;
        match (external_source, loaded_nnuec, retirement_evidence) {
            (Some(external), None, Some(retirement_evidence)) => {
                RoventCollectionPreparedNative::ExternalPcr {
                    prepared: Box::new(prepared),
                    external: Box::new(external),
                    retirement_evidence: Box::new(retirement_evidence),
                }
            }
            (None, Some(loaded), None) => RoventCollectionPreparedNative::OwnNnuecPcr {
                prepared: Box::new(prepared),
                loaded: Box::new(loaded),
            },
            (None, None, None) => RoventCollectionPreparedNative::OwnBoardPcr(Box::new(prepared)),
            _ => unreachable!("exclusive own/external source and evidence configuration"),
        }
    } else {
        let prepared = prepare_launch_native::<PcPortQuestion>(
            Arc::clone(&sources),
            resources,
            &clock,
            control.clone(),
            collection_observer,
            &diagnostic,
        )
        .map_err(|failure| reject_factory(failure, cleanup_until, &diagnostic))?;
        match loaded_nnuec {
            Some(loaded) => RoventCollectionPreparedNative::PcNnuec {
                prepared: Box::new(prepared),
                loaded: Box::new(loaded),
            },
            None => RoventCollectionPreparedNative::Pc(Box::new(prepared)),
        }
    };
    diagnostic.emit(
        NativeStartupDiagnosticStage::ProductAssembly,
        NativeStartupDiagnosticPhase::Entry,
    );
    let mut driver = match (collection_preload, collection_assemble) {
        (Some(preload), Some(assemble)) => assemble(RoventCollectionPreparedLaunch {
            native: prepared_product,
            preload,
            control: control.clone(),
            clock: clock.clone(),
            owners: Arc::clone(&owners),
            identity,
            root_policy: arguments.root_policy,
        }),
        (None, None) => assemble_native_product(
            prepared_product,
            clock.clone(),
            Arc::clone(&owners),
            identity,
            arguments.root_policy,
        ),
        _ => unreachable!("paired private Collection source and assembly request"),
    }
    .map_err(|failure| {
        diagnostic.emit(
            NativeStartupDiagnosticStage::ProductAssembly,
            NativeStartupDiagnosticPhase::Error,
        );
        reject_product(failure, cleanup_until, &diagnostic)
    })?;
    diagnostic.emit(
        NativeStartupDiagnosticStage::ProductAssembly,
        NativeStartupDiagnosticPhase::Exit,
    );
    if let Some(mut pending) = pending_storage.take() {
        let custody = pending
            .custody
            .take()
            .expect("selected original storage custody");
        let attached = driver.attach_storage_runtime_registration(
            custody,
            Arc::clone(&pending.boundary),
            &clock,
        );
        process_owners.transferred_to_product();
        if let Err(rejected) = attached {
            return reject_storage_runtime_registration(driver, rejected, cleanup_until);
        }
    }
    let lifecycle_diagnostics = arguments
        .lifecycle_diagnostics
        .then(|| driver.enable_lifecycle_stderr_diagnostics());
    if arguments.initial_stop_source_diagnostics {
        driver
            .enable_initial_stop_source_stderr_diagnostics()
            .map_err(|cause| error("Arguments", cause))?;
    }
    let mut settings = EngineSettings {
        shutdown_limit: Duration::from_millis(lock.manifest.cleanup_max_ms),
        ..EngineSettings::default()
    };
    settings.search.untimed_limit = Duration::from_millis(lock.manifest.wall_time_max_ms);
    // Keep original startup lock/selection/binary/cwd descriptors alive through
    // the same UCI whole-session physical finish, rather than a detached SSH job.
    let result = match product_runner {
        Some(run) => {
            // The callback gets the real owner, not a forged report or root.
            // Returning Ok without its actual physical finish is not success.
            let completion = driver.physical_close_observation();
            let result = run(driver, clock, owners, settings);
            if completion.is_confirmed() {
                result
            } else {
                Err(error(
                    "PhysicalCompletionUnknown",
                    match result {
                        Ok(()) => {
                            "finite product runner returned before actual physical finish".into()
                        }
                        Err(failure) => format!(
                            "finite product runner failed before actual physical finish: {failure}"
                        ),
                    },
                ))
            }
        }
        None => {
            let driver: Arc<dyn SearchSessionDriver> = Arc::new(driver);
            diagnostic.trace(NativeStartupDiagnosticStage::UciServe, || {
                serve_driver(driver, clock, owners, settings)
                    .map_err(|failure| error("UciSession", failure))
            })
        }
    };
    // The ordinary protocol/product runner has already attempted its original
    // whole-session cleanup. Diagnostic write failure is handled only now, and
    // cannot skip cleanup or reclassify an unknown fence as Confirmed.
    let result = match lifecycle_diagnostics
        .as_ref()
        .map(|diagnostics| diagnostics.complete_session(result.is_ok()))
        .transpose()
    {
        Ok(_) => result,
        Err(cause) => match result {
            Ok(()) => Err(error("LifecycleDiagnostics", cause)),
            Err(original) => {
                // Keep the original primary stage/cause. This bounded secondary
                // notice is best-effort when the diagnostic FD itself failed.
                use std::io::Write;
                let _ = writeln!(
                    std::io::stderr().lock(),
                    "rovent-lifecycle-diagnostic-failure/1 cause={cause}"
                );
                Err(original)
            }
        },
    };
    drop(pending_storage);
    drop(boundary);
    drop((external_selection_file, selection_file, lock_file));
    drop(process_owners);
    result
}

fn prepare_launch_native<Q: NativeQuestionProjection>(
    sources: Arc<RoventPinnedNativeSources>,
    resources: RoventNativeResources,
    clock: &ProcessClock,
    control: RoventStartupControl,
    observer: Option<Box<dyn VirtualReviewCollectionObserver>>,
    diagnostic: &NativeStartupDiagnostic,
) -> Result<PreparedRoventNative<Q>, RoventNativeStartupRejection<Q>> {
    diagnostic.trace(NativeStartupDiagnosticStage::NativePreparation, || {
        super::prepare_native_with_optional_observer(
            sources,
            resources,
            clock.runtime_clock(),
            control,
            observer,
            diagnostic.clone(),
        )
    })
}

/// /2 native products retain their original process, selected files and lock
/// through actual UCI drain. Collection, storage and V1 diagnostics remain
/// unsupported here; the separate V2 lifecycle observer is opt-in.
fn run_v2_native(
    arguments: Arguments,
    lock: Arc<RoventInputLockV2>,
    root: Dir,
    cwd: PathBuf,
    lock_file: File,
) -> Result<(), RoventLaunchError> {
    if arguments.storage.is_some()
        || arguments.startup_diagnostics
        || arguments.lifecycle_diagnostics
        || arguments.initial_stop_source_diagnostics
        || !matches!(
            lock.manifest.purpose,
            rz_experiments::RoventPurpose::Validation | rz_experiments::RoventPurpose::Deployment
        )
    {
        return Err(error(
            "InputLockUnsupportedV2",
            "the /2 native route does not support this asset, storage, or diagnostic",
        ));
    }
    let topology = super::admitted_topology_v2(&lock)
        .map_err(|failure| error("InputLockUnsupportedV2", failure))?;
    if arguments.startup_ms > lock.manifest.wall_time_max_ms {
        return Err(error(
            "StartupWindow",
            "startup exceeds the original declared window",
        ));
    }
    let begin = Instant::now();
    let until = begin
        .checked_add(Duration::from_millis(arguments.startup_ms))
        .ok_or_else(|| error("StartupWindow", "startup deadline overflow"))?;
    let cleanup_until = until
        .checked_add(Duration::from_millis(lock.manifest.cleanup_max_ms))
        .ok_or_else(|| error("StartupWindow", "cleanup deadline overflow"))?;
    let control = RoventStartupControl {
        until,
        cleanup_until,
        cancellation: Arc::new(AtomicBool::new(false)),
    };
    let clock = ProcessClock::new(ProcessEpoch(arguments.process_epoch));
    let binary = lock
        .manifest
        .engine
        .binary
        .open_verified_at(&root, 128 << 20)
        .map_err(|failure| error("Binary", failure))?;
    let boundary = verify_actual_owned_process_boundary_v2(RoventProcessBoundaryInputsV2 {
        lock: Arc::clone(&lock),
        original_binary: binary,
        original_root: root,
        private_cwd: cwd.clone(),
        control: control.clone(),
    })
    .map_err(|rejected| error("ProcessBoundary", rejected.error))?;
    let root = boundary.original_root();
    let view = boundary
        .startup_view()
        .map_err(|failure| error("ProcessBoundary", failure))?;
    let selected: Vec<_> = lock
        .manifest
        .provenance
        .iter()
        .filter(|artifact| artifact.path == arguments.native_assets)
        .collect();
    if selected.len() != 1 {
        return Err(error(
            "Assets",
            "selection is not a unique original provenance artifact",
        ));
    }
    let selection_file = selected[0]
        .open_verified_at(root, MAX_SELECTION_BYTES)
        .map_err(|failure| error("Assets", failure))?;
    let selection = SelectionWire::parse(&read_bounded(&selection_file, MAX_SELECTION_BYTES)?)?;
    let source_root = root.try_clone().map_err(|failure| error("Cwd", failure))?;
    let sources = Arc::new(
        RoventPinnedNativeSources::<RoventInputLockV2>::open_v2(
            Arc::clone(&lock),
            source_root,
            cwd,
            selection,
            &control,
        )
        .map_err(|failure| error("Assets", failure))?,
    );
    sources
        .configuration()
        .require_supported_native_capabilities()
        .map_err(|failure| error("Capability", failure))?;
    let nnuec_selection = sources.configuration().fields().nnuec_selection();
    match (
        topology,
        sources.configuration().fields().execution,
        nnuec_selection,
    ) {
        (
            super::RoventStartupTopology::RolesOnly,
            ExecutionConfiguration::RolesOnly { .. },
            NativeNnuecSelection::Absent | NativeNnuecSelection::SharedPc,
        ) => {}
        (
            super::RoventStartupTopology::OwnResearcher,
            ExecutionConfiguration::OwnResearcher {
                value: ResearchValueConfiguration::BaselineBoard,
                ..
            },
            NativeNnuecSelection::Absent,
        ) => {}
        (
            super::RoventStartupTopology::OwnResearcher,
            ExecutionConfiguration::OwnResearcher {
                value: ResearchValueConfiguration::NnuecV2 { .. },
                ..
            },
            NativeNnuecSelection::OwnResearcher,
        ) => {}
        (
            super::RoventStartupTopology::ExternalResearcher,
            ExecutionConfiguration::ExternalResearcher { .. },
            NativeNnuecSelection::Absent,
        ) => {}
        _ => {
            return Err(error(
                "InputLockUnsupportedV2",
                "V2 execution configuration or NNUEC topology is unsupported",
            ));
        }
    }
    // External source admission happens before a native worker exists. Keep
    // the original selection descriptor separately until the actual UCI
    // session has physically finished, including the unknown-completion path.
    let (external, external_selection_file) =
        match open_external_launch_v2(&arguments, &sources, &boundary, &control)? {
            Some(ExternalLaunchV2 {
                source,
                retirement,
                selection_file,
            }) => (Some((source, retirement)), Some(selection_file)),
            None => (None, None),
        };
    let cuda_profile = if matches!(lock.manifest.engine.pc.device, RoventDevice::Cuda { .. }) {
        let name = format!(
            "rovent-cuda-profile-{}-{}",
            std::process::id(),
            arguments.process_epoch
        );
        root.create_dir(&name)
            .map_err(|failure| error("ProfileDirectory", failure))?;
        let directory = root
            .open_dir_nofollow(&name)
            .map_err(|failure| error("ProfileDirectory", failure))?;
        Some(
            VirtualReviewProfileDirectory::from_owned_directory(directory.into_std_file())
                .map_err(|failure| error("ProfileDirectory", failure))?,
        )
    } else {
        None
    };
    let configuration = sources.configuration();
    let resources = RoventNativeResources {
        runtime: sources
            .pin_runtime()
            .map_err(|failure| error("Runtime", failure))?,
        configuration: configuration
            .fields()
            .native
            .onnx(&lock.manifest.engine.pc.device, cuda_profile.is_some())
            .map_err(|failure| error("Configuration", failure))?,
        loading_profile: configuration
            .fields()
            .native
            .loading_profile
            .map(|profile| profile.actual()),
        cuda_profile,
        drain_limit: Duration::from_millis(configuration.fields().native.drain_ms),
        tickets: configuration
            .fields()
            .native
            .tickets(configuration.fields().review.token_budget)
            .map_err(|failure| error("Configuration", failure))?,
        command_capacity: configuration.fields().native.command_capacity,
    };
    let diagnostic = NativeStartupDiagnostic::default();
    let loaded_nnuec = match nnuec_selection {
        NativeNnuecSelection::Absent => None,
        NativeNnuecSelection::SharedPc | NativeNnuecSelection::OwnResearcher => Some(
            LoadedRoventNnuecSource::load(Arc::clone(&sources), &control)
                .map_err(|failure| error("NnuecSource", failure))?,
        ),
    };
    let owners = Arc::new(OwnerRegistry::default());
    let identity = EngineIdentity {
        name: format!("Rovent {:?}", lock.manifest.engine.mode),
        author: "Rovent contributors".into(),
    };
    // Each arm transfers the one original prepared worker into its matching
    // product. The common error path retains the same boundary and FDs.
    macro_rules! launch_v2_product {
        ($question:ty, $constructor:path, $reject:path $(, $extra:expr)*) => {{
            let prepared = super::prepare_native_v2::<$question>(
                Arc::clone(&sources),
                resources,
                clock.runtime_clock(),
                control.clone(),
            )
            .map_err(|rejection| reject_factory(rejection, cleanup_until, &diagnostic))?;
            let driver = $constructor(
                prepared,
                $($extra,)*
                Arc::clone(&boundary),
                clock.clone(),
                Arc::clone(&owners),
                identity,
                arguments.root_policy,
            )
            .map_err(|rejection| $reject(rejection, cleanup_until, &diagnostic))?;
            let completion = driver.physical_close_observation();
            let driver: Arc<dyn SearchSessionDriver> = if arguments.lifecycle_diagnostics_v2 {
                Arc::new(super::product::diagnostics_v2::ObservedV2::new(driver, lock.canonical_sha256.clone()))
            } else {
                Arc::new(driver)
            };
            Ok((driver, completion))
        }};
    }
    let assembled = (|| -> Result<_, RoventLaunchError> {
        match (topology, loaded_nnuec, external) {
            (super::RoventStartupTopology::RolesOnly, None, None) => launch_v2_product!(
                PcPortQuestion,
                super::product::RoventNativeProductDriverV2::pc,
                reject_product_v2
            ),
            (super::RoventStartupTopology::RolesOnly, Some(loaded), None) => launch_v2_product!(
                PcPortQuestion,
                super::product::RoventNativeProductDriverV2::pc_with_shared_nnuec,
                reject_product_v2,
                loaded
            ),
            (super::RoventStartupTopology::OwnResearcher, None, None) => launch_v2_product!(
                AdaptivePortQuestion,
                super::product::RoventOwnPcrProductDriverV2::own_board_pcr,
                reject_own_pcr_product_v2
            ),
            (super::RoventStartupTopology::OwnResearcher, Some(loaded), None) => {
                launch_v2_product!(
                    AdaptivePortQuestion,
                    super::product::RoventOwnNnuecProductDriverV2::own_nnuec_pcr,
                    reject_own_nnuec_product_v2,
                    loaded
                )
            }
            (
                super::RoventStartupTopology::ExternalResearcher,
                None,
                Some((external, retirement)),
            ) => launch_v2_product!(
                AdaptivePortQuestion,
                super::product::RoventExternalPcrProductDriverV2::external_pcr,
                reject_external_pcr_product_v2,
                external,
                retirement
            ),
            _ => unreachable!("unsupported topology or NNUEC selection rejected before startup"),
        }
    })();
    let (driver, completion) = match assembled {
        Ok(assembled) => assembled,
        Err(failure) => {
            drop(view);
            if failure.stage == "PhysicalCompletionUnknown" {
                std::mem::forget((
                    boundary,
                    sources,
                    selection_file,
                    external_selection_file,
                    lock_file,
                ));
            }
            return Err(failure);
        }
    };
    let mut settings = EngineSettings {
        shutdown_limit: Duration::from_millis(lock.manifest.cleanup_max_ms),
        ..EngineSettings::default()
    };
    settings.search.untimed_limit = Duration::from_millis(lock.manifest.wall_time_max_ms);
    let result = serve_driver(driver, clock, owners, settings)
        .map_err(|failure| error("UciSession", failure));
    drop(view);
    if !completion.is_confirmed() {
        std::mem::forget((
            boundary,
            sources,
            selection_file,
            external_selection_file,
            lock_file,
        ));
        return Err(error(
            "PhysicalCompletionUnknown",
            match result {
                Ok(()) => "UCI returned before actual physical finish".into(),
                Err(failure) => format!("UCI failed before actual physical finish: {failure}"),
            },
        ));
    }
    result
}

// Assembly rejection returns the original prepared/startup owners and pins;
// retaining them must not require a fresh allocation on the failure path.
#[allow(clippy::result_large_err)]
fn assemble_native_product(
    native: RoventCollectionPreparedNative,
    clock: ProcessClock,
    owners: Arc<OwnerRegistry>,
    identity: EngineIdentity,
    root_policy: RoventRootAdmissionPolicy,
) -> Result<RoventNativeProductDriver, RoventProductStartupRejection> {
    match native {
        RoventCollectionPreparedNative::Pc(prepared) => {
            RoventNativeProductDriver::pc(*prepared, clock, owners, identity, root_policy)
        }
        RoventCollectionPreparedNative::PcNnuec { prepared, loaded } => {
            RoventNativeProductDriver::pc_with_shared_nnuec(
                *prepared,
                *loaded,
                clock,
                owners,
                identity,
                root_policy,
            )
        }
        RoventCollectionPreparedNative::OwnBoardPcr(prepared) => {
            RoventNativeProductDriver::own_board_pcr(
                *prepared,
                clock,
                owners,
                identity,
                root_policy,
            )
        }
        RoventCollectionPreparedNative::OwnNnuecPcr { prepared, loaded } => {
            RoventNativeProductDriver::own_nnuec_pcr(
                *prepared,
                *loaded,
                clock,
                owners,
                identity,
                root_policy,
            )
        }
        RoventCollectionPreparedNative::ExternalPcr {
            prepared,
            external,
            retirement_evidence,
        } => RoventNativeProductDriver::external_pcr(
            *prepared,
            *external,
            *retirement_evidence,
            clock,
            owners,
            identity,
            root_policy,
            None,
        ),
    }
}

fn read_bounded(file: &File, maximum: u64) -> Result<Vec<u8>, RoventLaunchError> {
    let metadata = file
        .metadata()
        .map_err(|failure| error("BoundedInput", failure))?;
    if !metadata.is_file() || metadata.len() == 0 || metadata.len() > maximum {
        return Err(error("BoundedInput", "not a bounded regular input"));
    }
    let mut bytes = Vec::new();
    file.take(maximum + 1)
        .read_to_end(&mut bytes)
        .map_err(|failure| error("BoundedInput", failure))?;
    if bytes.len() as u64 != metadata.len() {
        return Err(error("BoundedInput", "input changed while reading"));
    }
    Ok(bytes)
}
fn reject_factory<Q: NativeQuestionProjection>(
    mut failure: RoventNativeStartupRejection<Q>,
    until: Instant,
    diagnostic: &NativeStartupDiagnostic,
) -> RoventLaunchError {
    let detail = error("NativeStartup", &failure.error);
    if failure.native.as_mut().is_some_and(|native| {
        diagnostic.emit(
            NativeStartupDiagnosticStage::FactoryDrain,
            NativeStartupDiagnosticPhase::Entry,
        );
        let confirmed = drain_native(native, until);
        diagnostic.emit(
            NativeStartupDiagnosticStage::FactoryDrain,
            if confirmed {
                NativeStartupDiagnosticPhase::Exit
            } else {
                NativeStartupDiagnosticPhase::Error
            },
        );
        !confirmed
    }) {
        let _quarantine = Box::leak(Box::new(failure));
        return error("PhysicalCompletionUnknown", detail);
    }
    detail
}
fn reject_product(
    mut failure: RoventProductStartupRejection,
    until: Instant,
    diagnostic: &NativeStartupDiagnostic,
) -> RoventLaunchError {
    let detail = error("ProductAssembly", &failure.failure);
    diagnostic.emit(
        NativeStartupDiagnosticStage::ProductDrain,
        NativeStartupDiagnosticPhase::Entry,
    );
    let confirmed = match &mut failure.owners {
        RoventProductStartupOwners::Pc(prepared) => drain_native(&mut prepared.service, until),
        RoventProductStartupOwners::PcNnuec { prepared, .. } => {
            // The loaded source remains in this same rejection while native
            // cleanup runs. No actor exists before the shared root is admitted.
            drain_native(&mut prepared.service, until)
        }
        RoventProductStartupOwners::Pcr(prepared) => drain_native(&mut prepared.service, until),
        RoventProductStartupOwners::NnuecRejected(rejected) => {
            drain_native(&mut rejected.prepared.service, until)
        }
        RoventProductStartupOwners::PcrRejected(rejected) => {
            drain_native(&mut rejected.sources.native, until)
        }
        RoventProductStartupOwners::PcrDriver(driver) => {
            driver.finish_session(until).is_ok_and(|completion| {
                completion == Some(crate::search_driver::DriverPhysicalCompletion::Confirmed)
            })
        }
        RoventProductStartupOwners::ExternalRejected { rejected, .. } => {
            match &mut rejected.owners {
                // These stages own only original dormant descriptors. Keep
                // the whole rejection alive until native physical cleanup.
                RoventExternalProductStartupOwners::Prepared { prepared, .. }
                | RoventExternalProductStartupOwners::SourcePreparation { prepared, .. }
                | RoventExternalProductStartupOwners::PreparedSource { prepared, .. } => {
                    drain_native(&mut prepared.service, until)
                }
                RoventExternalProductStartupOwners::Session(rejection) => {
                    drain_native(&mut rejection.sources.native, until)
                }
                RoventExternalProductStartupOwners::Driver(driver) => {
                    driver.finish_session(until).is_ok_and(|completion| {
                        completion
                            == Some(crate::search_driver::DriverPhysicalCompletion::Confirmed)
                    })
                }
            }
        }
        RoventProductStartupOwners::ExternalEvidenceMismatch { prepared, .. } => {
            drain_native(&mut prepared.service, until)
        }
    };
    diagnostic.emit(
        NativeStartupDiagnosticStage::ProductDrain,
        if confirmed {
            NativeStartupDiagnosticPhase::Exit
        } else {
            NativeStartupDiagnosticPhase::Error
        },
    );
    if !confirmed {
        let _quarantine = Box::leak(Box::new(failure));
        return error("PhysicalCompletionUnknown", detail);
    }
    detail
}

fn reject_product_v2(
    mut failure: super::product::RoventProductV2StartupRejection,
    until: Instant,
    diagnostic: &NativeStartupDiagnostic,
) -> RoventLaunchError {
    let detail = error("ProductAssembly", &failure.failure);
    diagnostic.emit(
        NativeStartupDiagnosticStage::ProductDrain,
        NativeStartupDiagnosticPhase::Entry,
    );
    let confirmed = drain_native(&mut failure.prepared.service, until);
    diagnostic.emit(
        NativeStartupDiagnosticStage::ProductDrain,
        if confirmed {
            NativeStartupDiagnosticPhase::Exit
        } else {
            NativeStartupDiagnosticPhase::Error
        },
    );
    if !confirmed {
        let _quarantine = Box::leak(Box::new(failure));
        return error("PhysicalCompletionUnknown", detail);
    }
    detail
}
fn reject_own_pcr_product_v2(
    mut failure: super::product::RoventOwnPcrV2StartupRejection,
    until: Instant,
    diagnostic: &NativeStartupDiagnostic,
) -> RoventLaunchError {
    use super::product::RoventOwnPcrV2RejectedOwners;
    let detail = error("ProductAssembly", &failure.failure);
    diagnostic.emit(
        NativeStartupDiagnosticStage::ProductDrain,
        NativeStartupDiagnosticPhase::Entry,
    );
    let confirmed = match &mut failure.owners {
        RoventOwnPcrV2RejectedOwners::Prepared(prepared) => {
            drain_native(&mut prepared.service, until)
        }
        RoventOwnPcrV2RejectedOwners::Session(rejected) => {
            drain_native(&mut rejected.sources.native, until)
        }
        RoventOwnPcrV2RejectedOwners::Driver(driver) => {
            driver.finish_session(until).is_ok_and(|completion| {
                completion == Some(crate::search_driver::DriverPhysicalCompletion::Confirmed)
            })
        }
    };
    diagnostic.emit(
        NativeStartupDiagnosticStage::ProductDrain,
        if confirmed {
            NativeStartupDiagnosticPhase::Exit
        } else {
            NativeStartupDiagnosticPhase::Error
        },
    );
    if !confirmed {
        let _quarantine = Box::leak(Box::new(failure));
        return error("PhysicalCompletionUnknown", detail);
    }
    detail
}
fn drain_native<Q: NativeQuestionProjection>(
    native: &mut RoventNativeService<Q>,
    until: Instant,
) -> bool {
    native.begin_shutdown(until);
    let mut finished = None;
    let mut retained = Vec::new();
    loop {
        let signal = native.signal();
        let observed = signal.version();
        while let Some(event) = native.poll() {
            match event {
                TicketNativeEvent::Finished {
                    physical_shutdown_confirmed,
                } => finished = Some(finished.unwrap_or(true) && physical_shutdown_confirmed),
                TicketNativeEvent::Ready { .. } | TicketNativeEvent::RootQuiesced(_) => {}
                other => retained.push(other),
            }
        }
        let joined = native.try_join();
        if matches!(joined, Ok(true)) && finished == Some(true) {
            return true;
        }
        if joined.is_err() || finished == Some(false) || Instant::now() >= until {
            // Unexpected payloads retain their original leases on unknown
            // completion; no ordinary startup error frees them speculatively.
            std::mem::forget(retained);
            return false;
        }
        signal.wait_changed(
            observed,
            until
                .saturating_duration_since(Instant::now())
                .min(Duration::from_millis(10)),
        );
    }
}

#[cfg(test)]
#[path = "launch/tests.rs"]
mod tests;
