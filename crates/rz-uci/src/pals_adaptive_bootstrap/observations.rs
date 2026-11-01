//! Actual-process V5 evidence join. Declaration, native load, root fence,
//! process finish and independently registered numerical evidence retain
//! separate scopes. No model, worker, optimizer or performance run starts here.
use super::resource_readback::{
    ActualResourceObservationsV5, NativeResourceOwnerV5, require_original_startup_active,
};
use super::startup_snapshot::ActualStartupSnapshotV5;
use crate::pals_adaptive::{
    AdaptiveCloseMode, AdaptiveCompletionDiagnostic, AdaptiveManagerCounters,
    AdaptiveNativeAccountingSnapshot, AdaptiveRootObservationSnapshot,
    AdaptiveSessionProcessReceipt, AdaptiveSessionRetention,
};
use crate::pals_adaptive_startup::PinnedPalsStartupV5;
use crate::pals_native::startup_legacy::NativeLegacySelectedStartupSnapshotV5;
use crate::pals_native::startup_virtual::NativeVirtualReviewStartupSnapshotV5;
use crate::search_driver::{DriverPhysicalCompletion, SearchSessionFailure};
use cap_fs_ext::{DirExt, FollowSymlinks, OpenOptionsFollowExt};
use cap_std::fs::{Dir, OpenOptions};
use rz_contracts::pals::SearchAuthority;
use rz_experiments::{
    ArtifactRef, ManifestError, PALS_NATIVE_STARTUP_V5_DOMAIN, PALS_NATIVE_TERMINATION_V5_DOMAIN,
    PalsAdaptiveStartupRecordV5, PalsAdaptiveTerminationRecordV5, PalsArtifactDomainV5,
    PalsCoordinatorObservationV5, PalsEndpointReceiptV5, PalsModelBackendV3,
    PalsModelVerificationV5, PalsNativeNumericalCheckV5, PalsObservedV3, PalsPhysicalStateV3,
    PalsRoleV3, PalsStartupAssetPurposeV5, PalsStartupContentAdmissionV5,
    pals_coordinator_evidence_bytes_v5, pals_model_verification_evidence_bytes_v5,
    pals_resource_readback_evidence_bytes_v5,
};
use rz_search::pals::coordinator::{
    CoordinatorHistoryUsage, RoleTicketCumulativeObservation, RoleTicketPhaseOccupancy,
    ValidatorActionHistoryUsage,
};
use serde::Serialize;
use sha2::{Digest as _, Sha256};
use std::{
    io::Write,
    path::Path,
    sync::{Arc, Mutex},
    time::Instant,
};

const METADATA_BYTES_MAX: usize = 128 << 10;
const OWNED_ROOT_BYTES_MAX: u64 = 18 << 30;
const ROOT_OBSERVATIONS_MAX: u64 = 65_536;
// Two additional bounded envelopes preserve startup and last-root resource
// scope separately. They are not rewritten as a single lifetime observation.
const OUTPUT_FILES_MAX: usize = 10;

fn fail(message: &str) -> ManifestError {
    ManifestError::Integrity(format!("V5 actual observation: {message}"))
}
fn need(ok: bool, message: &str) -> Result<(), ManifestError> {
    if ok { Ok(()) } else { Err(fail(message)) }
}
fn hex(bytes: [u8; 32]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}
fn observed<T>(value: T, method: &str) -> PalsObservedV3<T> {
    PalsObservedV3::Observed {
        value,
        method: method.into(),
    }
}

/// This helper is called only after the opaque export/content owner and this
/// process's actual factory snapshot have joined. Generated means membership
/// in that verified export, not creation of a model or optimizer in this
/// process. A P/C-only factory establishes both absence facts; neither fact is
/// a substitute for a physical per-role call observation.
fn observe_verified_factory_roles(
    endpoint: &mut PalsEndpointReceiptV5,
    exported_roles: &std::collections::BTreeSet<PalsRoleV3>,
    validator_loaded: bool,
) -> Result<(), ManifestError> {
    let validator_exported = exported_roles.contains(&PalsRoleV3::Validator);
    let expected_roles = if validator_exported {
        std::collections::BTreeSet::from([
            PalsRoleV3::Proposer,
            PalsRoleV3::Critic,
            PalsRoleV3::Validator,
        ])
    } else {
        std::collections::BTreeSet::from([PalsRoleV3::Proposer, PalsRoleV3::Critic])
    };
    need(
        *exported_roles == expected_roles && validator_loaded == validator_exported,
        "exact actual factory roles differ from the verified export",
    )?;
    endpoint.v_loaded = observed(
        validator_loaded,
        "actual native factory validator presence/absence after exact verified export-role join",
    );
    endpoint.v_generated = observed(
        validator_exported,
        "validator membership/absence in this verified export; no optimizer/export execution in this process is asserted",
    );
    Ok(())
}
fn require_owned_write_budget(current: u64, added: usize) -> Result<(), ManifestError> {
    need(
        current
            .checked_add(added as u64)
            .is_some_and(|n| n <= OWNED_ROOT_BYTES_MAX),
        "owned private root would exceed 18GiB",
    )
}

/// The Root's already verified executable/CWD values are checked against the
/// retained transport again, without rehashing/loading any asset here. Private
/// fields prevent an external producer from replacing this join with a flag.
pub(crate) struct ActualProcessIdentityV5 {
    process_id: u32,
    process_run_id: String,
}
impl ActualProcessIdentityV5 {
    pub(crate) fn join(
        pinned: &PinnedPalsStartupV5,
        actual_binary_sha256: [u8; 32],
        actual_cwd: &Path,
    ) -> Result<Self, ManifestError> {
        let pid = std::process::id();
        need(
            pid > 0 && pid <= i32::MAX as u32,
            "invalid actual process identity",
        )?;
        need(
            actual_cwd.is_absolute()
                && actual_cwd == pinned.private_cwd()
                && std::env::current_dir().map_err(|_| fail("actual CWD unavailable"))?
                    == actual_cwd
                && hex(actual_binary_sha256) == pinned.config.endpoint.binary.sha256,
            "Root verified executable/CWD differs from the private startup owner",
        )?;
        Ok(Self {
            process_id: pid,
            process_run_id: format!("native-process-{pid}"),
        })
    }
}

#[derive(Clone, Copy)]
enum OutputKind {
    Model,
    FactorySnapshot,
    Startup,
    RawTotals,
    Coordinator,
    Termination,
    FailedTermination,
    SessionReleaseObservation,
    StartupResources,
    FinalResources,
}
impl OutputKind {
    fn name(self) -> &'static str {
        match self {
            Self::Model => "pals-model-verification.v5.json",
            Self::FactorySnapshot => "pals-native-factory-snapshot.v5.json",
            Self::Startup => "pals-native-startup.v5.json",
            Self::RawTotals => "pals-root-observation-totals.v5.json",
            Self::Coordinator => "pals-coordinator-observation.v5.json",
            Self::Termination => "pals-native-termination.v5.json",
            Self::FailedTermination => "pals-native-failed-termination-observation.v5.json",
            Self::SessionReleaseObservation => "pals-session-release-observation.v1.json",
            Self::StartupResources => "pals-startup-resource-readback.v5.json",
            Self::FinalResources => "pals-final-resource-readback.v5.json",
        }
    }
}

/// All writes are main-owner operations before admission or after final drain.
/// A root callback only adds fixed-size counters; it performs no filesystem I/O.
struct BoundedObservationWriterV5 {
    root: Dir,
    owner: Dir,
    logical_prefix: String,
    written_files: usize,
}
impl BoundedObservationWriterV5 {
    fn new(
        pinned: &PinnedPalsStartupV5,
        id: &ActualProcessIdentityV5,
    ) -> Result<Self, ManifestError> {
        let runtime = pinned.runtime_directory_capability()?;
        runtime
            .create_dir(&id.process_run_id)
            .map_err(|_| fail("actual process output owner already exists or cannot be created"))?;
        let owner = runtime
            .open_dir_nofollow(&id.process_run_id)
            .map_err(|_| fail("actual process output directory changed"))?;
        Ok(Self {
            root: pinned.private_root_capability()?,
            owner,
            logical_prefix: format!("{}/{}", pinned.config.runtime_directory, id.process_run_id),
            written_files: 0,
        })
    }
    fn artifact_name(&self, kind: OutputKind) -> ArtifactRef {
        // Only the excluded self-reference uses this temporary identity.
        ArtifactRef {
            path: format!("{}/{}", self.logical_prefix, kind.name()),
            sha256: "0".repeat(64),
            bytes: 1,
            source: "https://github.com/daejunnom/RoveZero".into(),
            license: "MIT".into(),
        }
    }
    fn write_json<T: Serialize>(
        &mut self,
        kind: OutputKind,
        value: &T,
    ) -> Result<ArtifactRef, ManifestError> {
        let bytes =
            serde_json::to_vec(value).map_err(|_| fail("bounded metadata serialization failed"))?;
        self.write_bytes(kind, &bytes)
    }
    fn write_bytes(
        &mut self,
        kind: OutputKind,
        bytes: &[u8],
    ) -> Result<ArtifactRef, ManifestError> {
        need(
            !bytes.is_empty() && bytes.len() <= METADATA_BYTES_MAX,
            "metadata exceeds its 128KiB bound",
        )?;
        need(
            self.written_files < OUTPUT_FILES_MAX,
            "actual process evidence file bound exceeded",
        )?;
        let current = owned_root_bytes(&self.root)?;
        require_owned_write_budget(current, bytes.len())?;
        let mut options = OpenOptions::new();
        options
            .write(true)
            .create_new(true)
            .follow(FollowSymlinks::No);
        let mut file = self
            .owner
            .open_with(kind.name(), &options)
            .map_err(|_| fail("evidence output already exists or exclusive creation failed"))?;
        file.write_all(bytes)
            .and_then(|_| file.sync_all())
            .map_err(|_| fail("actual process evidence persistence failed"))?;
        self.written_files += 1;
        let artifact = ArtifactRef {
            path: format!("{}/{}", self.logical_prefix, kind.name()),
            sha256: format!("{:x}", Sha256::digest(bytes)),
            bytes: bytes.len() as u64,
            source: "https://github.com/daejunnom/RoveZero".into(),
            license: "MIT".into(),
        };
        artifact.validate()?;
        Ok(artifact)
    }
}

/// Count lengths only in the explicitly granted, already owned attempt.
/// No file contents are read and links/credentials/unsupported entries fail
/// closed. This writer's serialized scan/write is not a filesystem quota or
/// authority for another owner to grow output beyond its own registered bound.
fn owned_root_bytes(root: &Dir) -> Result<u64, ManifestError> {
    fn walk(dir: &Dir, depth: usize, entries: &mut usize) -> Result<u64, ManifestError> {
        need(depth <= 32, "owned root directory depth exceeds bound")?;
        let mut bytes = 0u64;
        for entry in dir
            .entries()
            .map_err(|_| fail("owned root length scan failed"))?
        {
            *entries = entries
                .checked_add(1)
                .ok_or_else(|| fail("owned root entry overflow"))?;
            need(*entries <= 65_536, "owned root entry count exceeds bound")?;
            let entry = entry.map_err(|_| fail("owned root entry unavailable"))?;
            let name = entry.file_name();
            let safe = name
                .to_str()
                .ok_or_else(|| fail("owned output name is not UTF8"))?;
            let lower = safe.to_ascii_lowercase();
            need(
                !lower.starts_with(".env")
                    && !lower.contains("credential")
                    && !lower.contains("service_account")
                    && !lower.contains("service-account")
                    && !lower.contains("api_key")
                    && !lower.contains("api-key")
                    && !lower.starts_with("id_rsa")
                    && !lower.starts_with("id_ed25519"),
                "secret-like entry is outside evidence writer authority",
            )?;
            let metadata = dir
                .symlink_metadata(&name)
                .map_err(|_| fail("owned root entry changed"))?;
            let amount = if metadata.is_dir() {
                let child = dir
                    .open_dir_nofollow(&name)
                    .map_err(|_| fail("owned directory changed or linked"))?;
                walk(&child, depth + 1, entries)?
            } else {
                need(
                    metadata.is_file(),
                    "owned root contains a link or unsupported entry",
                )?;
                let mut options = OpenOptions::new();
                options.read(true).follow(FollowSymlinks::No);
                #[cfg(target_os = "linux")]
                {
                    use cap_std::fs::OpenOptionsExt;
                    options.custom_flags(nix::libc::O_NOFOLLOW | nix::libc::O_NONBLOCK);
                }
                let file = dir
                    .open_with(&name, &options)
                    .map_err(|_| fail("owned regular file changed"))?;
                let pinned = file
                    .into_std()
                    .metadata()
                    .map_err(|_| fail("owned pinned file length unavailable"))?;
                need(pinned.is_file(), "owned pinned entry is not a regular file")?;
                #[cfg(target_os = "linux")]
                {
                    use std::os::unix::fs::MetadataExt;
                    need(
                        pinned.nlink() == 1,
                        "owned evidence root contains a hard-link alias",
                    )?;
                }
                pinned.len()
            };
            bytes = bytes
                .checked_add(amount)
                .ok_or_else(|| fail("owned root byte overflow"))?;
            need(
                bytes <= OWNED_ROOT_BYTES_MAX,
                "owned private root exceeds 18GiB",
            )?;
        }
        Ok(bytes)
    }
    walk(root, 0, &mut 0)
}

fn join_virtual_factory_snapshot(
    pinned: &PinnedPalsStartupV5,
    admission: &PalsStartupContentAdmissionV5,
    snapshot: &NativeVirtualReviewStartupSnapshotV5,
) -> Result<(), ManifestError> {
    let assets: Vec<_> = pinned
        .assets()
        .iter()
        .map(|asset| (&asset.declaration, &asset.file))
        .collect();
    admission.verify_binding(&pinned.config, &pinned.lock, &assets)?;
    let model = &pinned.config.endpoint.model;
    let domain = admission.artifact_domain();
    need(
        !pinned.metadata_preflight
            && !pinned.config.training_executed
            && snapshot.schema == "rz-pals-native-virtual-startup-snapshot-v5/1"
            && snapshot.model_profile == model.profile
            && hex(snapshot.content_admission_config_sha256) == admission.config_sha256()
            && hex(snapshot.content_admission_semantic_lock_sha256)
                == admission.semantic_lock_sha256()
            && hex(snapshot.numerical_acceptance_sha256) == admission.numerical().acceptance.sha256
            && hex(snapshot.loaded_weights_sha256) == model.weights.sha256
            && hex(snapshot.model_semantic_sha256) == model.semantic_sha256
            && hex(snapshot.encoding_sha256) == model.encoding_sha256
            && hex(snapshot.model_implementation_sha256) == model.implementation_sha256
            && snapshot.registered_model_epoch == model.epoch
            && snapshot.validator_loaded == model.exported_roles.contains(&PalsRoleV3::Validator)
            && snapshot.physical_workers_created == 1
            && !snapshot.neural_service_executed
            && !snapshot.tensor_optimizer_content_verified
            && !admission.tensor_optimizer_content_verified()
            && domain.artifact_domain == model.artifact_domain
            && domain.checkpoint_sha256 == model.weights.sha256
            && domain.producer_metadata_verified
            && domain.registered_artifact_bytes_verified_by_launch_owner
            && domain.numerical_acceptance_verified,
        "actual native load snapshot does not join its opaque content owner",
    )?;
    let provider = match model.backend {
        PalsModelBackendV3::OrtCpu => "CPUExecutionProvider",
        PalsModelBackendV3::OrtCuda => "CUDAExecutionProvider",
        PalsModelBackendV3::RustCpu | PalsModelBackendV3::DeterministicMock => {
            return Err(fail("unsupported actual native ORT observation backend"));
        }
    };
    need(
        snapshot.configured_provider == provider,
        "actual selected provider differs from registered backend",
    )?;
    need(
        hex(snapshot.export_manifest_sha256)
            == pinned
                .asset(PalsStartupAssetPurposeV5::ExportManifest)?
                .declaration
                .artifact
                .sha256
            && hex(snapshot.runtime_library_sha256)
                == pinned
                    .asset(PalsStartupAssetPurposeV5::RuntimeLibrary)?
                    .declaration
                    .artifact
                    .sha256,
        "actual loaded export/runtime differs from retained asset descriptors",
    )
}

fn join_legacy_factory_snapshot(
    pinned: &PinnedPalsStartupV5,
    admission: &PalsStartupContentAdmissionV5,
    snapshot: &NativeLegacySelectedStartupSnapshotV5,
) -> Result<(), ManifestError> {
    use rz_experiments::{PalsEngineModeV5, PalsViewSelectionV5};
    let assets: Vec<_> = pinned
        .assets()
        .iter()
        .map(|asset| (&asset.declaration, &asset.file))
        .collect();
    admission.verify_binding(&pinned.config, &pinned.lock, &assets)?;
    let endpoint = &pinned.config.endpoint;
    let model = &endpoint.model;
    let domain = admission.artifact_domain();
    let source = &snapshot.source;
    let receipt = &snapshot.startup_owner_receipt;
    let selected = match endpoint.context.selection {
        PalsViewSelectionV5::V2Selected => true,
        PalsViewSelectionV5::V2Full => false,
        PalsViewSelectionV5::VirtualReview => {
            return Err(fail("legacy factory cannot admit compact context"));
        }
    };
    let schema = if selected {
        "rz-pals-native-legacy-selected-startup-snapshot-v5/1"
    } else {
        "rz-pals-native-legacy-full-startup-snapshot-v5/1"
    };
    need(
        !pinned.metadata_preflight
            && !pinned.config.training_executed
            && endpoint.mode == PalsEngineModeV5::Deployment
            && snapshot.schema == schema
            && snapshot.selected_view == selected
            && snapshot.model_profile == "full_line_interaction_v2"
            && snapshot.model_profile == model.profile
            && source.model_configuration.profile.as_str() == model.profile
            && !source.trained
            && !receipt.trained
            && hex(snapshot.content_admission_config_sha256) == admission.config_sha256()
            && hex(snapshot.content_admission_semantic_lock_sha256)
                == admission.semantic_lock_sha256()
            && hex(snapshot.numerical_acceptance_sha256) == admission.numerical().acceptance.sha256
            && hex(source.checkpoint_sha256) == model.weights.sha256
            && hex(snapshot.model_semantic_sha256) == model.semantic_sha256
            && hex(snapshot.encoding_sha256) == model.encoding_sha256
            && snapshot.encoding_sha256 == source.encoding_semantic_sha256
            && hex(snapshot.model_implementation_sha256) == model.implementation_sha256
            && snapshot.registered_model_epoch == model.epoch
            && !snapshot.validator_loaded
            && !model.exported_roles.contains(&PalsRoleV3::Validator)
            && snapshot.physical_workers_created == 1
            && snapshot.configured_intra_threads == endpoint.resources.ort_intra_threads as usize
            && snapshot.configured_inter_threads == endpoint.resources.ort_inter_threads as usize
            && !snapshot.configured_spinning
            && !snapshot.configured_tf32
            && !snapshot.configured_approximate_warm
            && !snapshot.tensor_optimizer_content_verified
            && !admission.tensor_optimizer_content_verified()
            && domain.artifact_domain == model.artifact_domain
            && domain.checkpoint_sha256 == model.weights.sha256
            && domain.producer_metadata_verified
            && domain.registered_artifact_bytes_verified_by_launch_owner
            && domain.numerical_acceptance_verified
            && receipt.process_epoch == snapshot.process_epoch
            && receipt.model_epoch == source.checkpoint_sha256
            && receipt.export_manifest_sha256 == source.export_manifest_sha256
            && receipt.encoding_semantic_sha256 == source.encoding_semantic_sha256
            && receipt.physical_runs_in_flight == 0
            && !receipt.quarantined
            && !receipt.physical_shutdown_confirmed
            && !receipt.native_buffers_released,
        "actual legacy load/probe snapshot differs from its original content owner or V2 context",
    )?;
    let provider = match model.backend {
        PalsModelBackendV3::OrtCpu => "CPUExecutionProvider",
        PalsModelBackendV3::OrtCuda => "CUDAExecutionProvider",
        _ => {
            return Err(fail(
                "legacy actual snapshot requires the declared ORT backend",
            ));
        }
    };
    need(
        snapshot.configured_provider == provider,
        "legacy loaded provider differs from its declared backend",
    )?;
    need(
        hex(source.export_manifest_sha256)
            == pinned
                .asset(PalsStartupAssetPurposeV5::ExportManifest)?
                .declaration
                .artifact
                .sha256
            && hex(source.execution.runtime_sha256)
                == pinned
                    .asset(PalsStartupAssetPurposeV5::RuntimeLibrary)?
                    .declaration
                    .artifact
                    .sha256
            && receipt.execution.runtime_sha256 == source.execution.runtime_sha256
            && receipt.execution.runtime_bundle_sha256 == source.execution.runtime_bundle_sha256
            && source.execution.provider
                == if model.backend == PalsModelBackendV3::OrtCpu {
                    "cpu"
                } else {
                    "cuda"
                },
        "actual legacy loaded export/runtime differs from the retained asset descriptors",
    )
}
fn join_factory_snapshot(
    pinned: &PinnedPalsStartupV5,
    admission: &PalsStartupContentAdmissionV5,
    snapshot: &ActualStartupSnapshotV5,
) -> Result<(), ManifestError> {
    match snapshot {
        ActualStartupSnapshotV5::Virtual(snapshot) => {
            join_virtual_factory_snapshot(pinned, admission, snapshot)
        }
        ActualStartupSnapshotV5::Legacy(snapshot) => {
            join_legacy_factory_snapshot(pinned, admission, snapshot)
        }
    }
}

pub(crate) struct PalsAdaptiveObservationBridgeV5 {
    pinned: Arc<PinnedPalsStartupV5>,
    identity: ActualProcessIdentityV5,
    snapshot: ActualStartupSnapshotV5,
    writer: Mutex<BoundedObservationWriterV5>,
    aggregate: Mutex<RootAggregateV5>,
    startup: PalsEndpointReceiptV5,
    resources: Mutex<ActualResourceObservationsV5>,
}
impl PalsAdaptiveObservationBridgeV5 {
    pub(crate) fn new(
        pinned: Arc<PinnedPalsStartupV5>,
        admission: &PalsStartupContentAdmissionV5,
        snapshot: ActualStartupSnapshotV5,
        actual_binary_sha256: [u8; 32],
        actual_cwd: &Path,
        resource_owner: NativeResourceOwnerV5,
    ) -> Result<Self, ManifestError> {
        join_factory_snapshot(&pinned, admission, &snapshot)?;
        let identity = ActualProcessIdentityV5::join(&pinned, actual_binary_sha256, actual_cwd)?;
        let mut writer = BoundedObservationWriterV5::new(&pinned, &identity)?;
        let model = &pinned.config.endpoint.model;
        let mut verification = PalsModelVerificationV5 {
            weights: model.weights.clone(),
            provenance: model.provenance.clone(),
            artifact_domain: admission.artifact_domain().artifact_domain,
            checkpoint_domain: model.checkpoint_domain.clone(),
            semantic_sha256: model.semantic_sha256.clone(),
            encoding_sha256: model.encoding_sha256.clone(),
            profile: model.profile.clone(),
            exported_roles: model.exported_roles.clone(),
            backend: model.backend,
            precision: model.precision,
            content_domain_verified: true,
            diagnostic_asset: model.artifact_domain == PalsArtifactDomainV5::Diagnostic,
            evidence: writer.artifact_name(OutputKind::Model),
        };
        let bytes = pals_model_verification_evidence_bytes_v5(
            &model_endpoint(&pinned),
            identity.process_id,
            &identity.process_run_id,
            &verification,
        )?;
        verification.evidence = writer.write_bytes(OutputKind::Model, &bytes)?;
        writer.write_json(OutputKind::FactorySnapshot, &snapshot)?;
        let mut startup = PalsEndpointReceiptV5::with_unknown_checks(
            model_endpoint(&pinned),
            identity.process_run_id.clone(),
        );
        startup.model_verification = observed(
            verification,
            "opaque actual retained content admission joined to this process native factory snapshot; tensor/optimizer decode is not asserted",
        );
        startup.native_numerical_check = observed(
            PalsNativeNumericalCheckV5 {
                weights_sha256: model.weights.sha256.clone(),
                semantic_sha256: model.semantic_sha256.clone(),
                encoding_sha256: model.encoding_sha256.clone(),
                runtime_implementation_sha256: pinned
                    .config
                    .endpoint
                    .runtime
                    .implementation_sha256
                    .clone(),
                source_commit: pinned.config.endpoint.source_commit.clone(),
                passed: true,
                logits_absolute_tolerance: "1e-4".into(),
                logits_relative_tolerance: "1e-3".into(),
                policy_wdl_absolute_tolerance: "1e-4".into(),
                evidence: admission.numerical().acceptance.clone(),
            },
            "reused independently registered raw tensor/reference/native numerical admission; not a new numerical run or NN count in this process",
        );
        observe_verified_factory_roles(
            &mut startup,
            &model.exported_roles,
            snapshot.validator_loaded(),
        )?;
        // Configured threads/TF32 are requests. They never fill applied resource
        // readback, actual TF32, RSS/VRAM peaks or process-tree exit observations.
        Ok(Self {
            pinned,
            identity,
            snapshot,
            writer: Mutex::new(writer),
            aggregate: Mutex::new(RootAggregateV5::default()),
            startup,
            resources: Mutex::new(ActualResourceObservationsV5::new(resource_owner)),
        })
    }
    pub(crate) fn publish_startup(
        &self,
        native_thread: Option<rz_runtime::adaptive::CpuThreadReadback>,
        external: Option<rz_search::cpu_checker::ExternalProcessIdentity>,
        until: Instant,
        cancel: &std::sync::atomic::AtomicBool,
    ) -> Result<ArtifactRef, ManifestError> {
        require_original_startup_active(cancel)?;
        let mut resources = self
            .resources
            .lock()
            .map_err(|_| fail("resource observation owner poisoned"))?;
        let mut writer = self
            .writer
            .lock()
            .map_err(|_| fail("evidence writer poisoned"))?;
        let mut value = resources.observe_startup(
            self.pinned.config.endpoint.resources.evidence_profile,
            native_thread,
            external,
            until,
            cancel,
            writer.artifact_name(OutputKind::StartupResources),
        )?;
        let bytes = pals_resource_readback_evidence_bytes_v5(
            &model_endpoint(&self.pinned),
            self.identity.process_id,
            &self.identity.process_run_id,
            &value,
        )?;
        require_original_startup_active(cancel)?;
        value.evidence = writer.write_bytes(OutputKind::StartupResources, &bytes)?;
        let mut observation = self.startup.clone();
        observation.resource_readback = observed(
            value,
            "actual native loader settings joined to read-only OS observations at this process startup; unknown fields and direct/visible quota scope retained",
        );
        let record = PalsAdaptiveStartupRecordV5 {
            schema_version: 5,
            domain: PALS_NATIVE_STARTUP_V5_DOMAIN.into(),
            lock_sha256: self.pinned.lock.canonical_sha256.clone(),
            startup_config_sha256: self.pinned.config_sha256.clone(),
            endpoint_id: model_endpoint(&self.pinned),
            process_id: self.identity.process_id,
            process_run_id: self.identity.process_run_id.clone(),
            source_commit: self.pinned.config.endpoint.source_commit.clone(),
            component_source_pins: self.pinned.config.source_components.clone(),
            observation,
        };
        require_original_startup_active(cancel)?;
        writer.write_json(OutputKind::Startup, &record)
    }
    /// Call once from the actual root-fence observer before manager dismantle.
    /// This bounded merge does not write files or call a worker/model. Only
    /// the current actual manager thread's fixed OS metadata is read here.
    pub(crate) fn observe_closed_root(
        &self,
        actual: &AdaptiveRootObservationSnapshot,
    ) -> Result<(), ManifestError> {
        let mut aggregate = self
            .aggregate
            .lock()
            .map_err(|_| fail("root observation aggregate poisoned"))?;
        // Validate all root-counter joins/overflow checks before the bounded
        // resource merge; either merge failure preserves the previous state.
        let mut next = *aggregate;
        next.add(actual, self.snapshot.process_epoch())?;
        self.resources
            .lock()
            .map_err(|_| fail("resource observation owner poisoned"))?
            .observe_closed_root(
                actual,
                rz_runtime::adaptive::observe_current_cpu_thread(
                    rz_runtime::adaptive::CpuActor::Manager,
                )
                .ok(),
            )?;
        *aggregate = next;
        Ok(())
    }
    pub(crate) fn publish_termination(
        &self,
        finish: &AdaptiveSessionProcessReceipt,
        completion: Option<&AdaptiveCompletionDiagnostic>,
    ) -> Result<ArtifactRef, ManifestError> {
        need(
            finish.epoch().0 == self.snapshot.process_epoch()
                && finish.permits_process_close()
                && finish.native_owner_joined()
                && finish.native_physical_shutdown_confirmed()
                && finish.store_archive_closed(),
            "physical/process/archive completion is unknown; normal termination refused",
        )?;
        let aggregate = self
            .aggregate
            .lock()
            .map_err(|_| fail("root observation aggregate poisoned"))?;
        aggregate.require_process_finish_join(finish.last_authority())?;
        let resources = self
            .resources
            .lock()
            .map_err(|_| fail("resource observation owner poisoned"))?;
        let mut writer = self
            .writer
            .lock()
            .map_err(|_| fail("evidence writer poisoned"))?;
        let mut raw_metadata = aggregate.metadata();
        raw_metadata["pals_completion_diagnostic"] =
            completion_diagnostic_metadata(completion, self.snapshot.process_epoch());
        raw_metadata["applied_resource_readback"] = serde_json::json!({
            "producer":"actual native loader, original ready thread owners, bounded read-only OS probe",
            "diagnostic":resources.diagnostic(),
            "host_cpu_probe_diagnostic":resources.host_diagnostic(),
            "scope":"startup and latest settled-root snapshots; unknown application, hidden ancestors, peak and external option values are not synthesized"
        });
        writer.write_json(OutputKind::RawTotals, &raw_metadata)?;
        let mut endpoint = self.startup.clone();
        if let Some(mut value) =
            resources.final_readback(writer.artifact_name(OutputKind::FinalResources))
        {
            let bytes = pals_resource_readback_evidence_bytes_v5(
                &endpoint.endpoint_id,
                self.identity.process_id,
                &self.identity.process_run_id,
                &value,
            )?;
            value.evidence = writer.write_bytes(OutputKind::FinalResources, &bytes)?;
            endpoint.resource_readback = observed(
                value,
                "startup OS snapshot and latest settled root's original ready owner readings; no lifetime/peak or declared-to-applied conversion",
            );
        }
        if let Some(mut coordinator) =
            aggregate.complete_coordinator(writer.artifact_name(OutputKind::Coordinator))
        {
            coordinator.owner_joined = finish.native_owner_joined();
            coordinator.archive_closed = finish.store_archive_closed();
            // The actual factory owns a physical worker/session even for a
            // zero-NN root. Its confirmed final drain is Completed; zero input
            // work does not make the owner's physical lifetime NotRequired.
            coordinator.physical_state = PalsPhysicalStateV3::Completed;
            let bytes = pals_coordinator_evidence_bytes_v5(
                &endpoint.endpoint_id,
                self.identity.process_id,
                &self.identity.process_run_id,
                &coordinator,
            )?;
            coordinator.evidence = writer.write_bytes(OutputKind::Coordinator, &bytes)?;
            endpoint.coordinator = observed(
                coordinator,
                "actual manager root snapshots aggregated once; root fence and final process owner join are separate",
            );
        }
        if let Some(called) = aggregate.validator_physically_called() {
            endpoint.v_called = observed(
                called,
                "every original actual native command's role delta or owner-issued pre-run skip joined to settled root fences and this final process owner join; public graph and V role Run stay separate",
            );
        }
        let record = PalsAdaptiveTerminationRecordV5 {
            schema_version: 5,
            domain: PALS_NATIVE_TERMINATION_V5_DOMAIN.into(),
            lock_sha256: self.pinned.lock.canonical_sha256.clone(),
            startup_config_sha256: self.pinned.config_sha256.clone(),
            endpoint_id: endpoint.endpoint_id.clone(),
            process_id: self.identity.process_id,
            process_run_id: self.identity.process_run_id.clone(),
            observation: endpoint,
            admission_closed: true,
            question_authority_revoked: true,
            cpu_workers_joined: finish.latest_root().is_none_or(|r| {
                r.own_cpu_joined == r.expected_own_cpu_owners && r.own_cpu_panicked == 0
            }),
            gpu_physical_drain_known: finish.native_physical_shutdown_confirmed(),
            archive_closed: finish.store_archive_closed(),
            training_executed: false,
        };
        writer.write_json(OutputKind::Termination, &record)
    }

    /// A separate, non-acceptance observation written after required receipts.
    /// No value here proves placement, GPU completion, input count or strength.
    pub(crate) fn publish_session_release_observation(
        &self,
        finish: Option<&AdaptiveSessionProcessReceipt>,
    ) -> Result<ArtifactRef, ManifestError> {
        if let Some(receipt) = finish {
            need(receipt.epoch().0 == self.snapshot.process_epoch(),
                "session release observation belongs to another process epoch")?;
        }
        let provider = match self.pinned.config.endpoint.model.backend {
            PalsModelBackendV3::OrtCpu => "CPUExecutionProvider",
            PalsModelBackendV3::OrtCuda => "CUDAExecutionProvider",
            PalsModelBackendV3::RustCpu | PalsModelBackendV3::DeterministicMock => "unknown",
        };
        let release = finish.and_then(|receipt| receipt.native_session_release());
        let session = match release {
            Some(observation) => serde_json::json!({
                "status":"released", "completed_synchronous_runs":observation.release().completed_synchronous_runs,
                "zero_run":observation.release().completed_synchronous_runs == 0,
                "observation_source":format!("{:?}", observation.source()),
            }),
            None => serde_json::json!({"status":"unknown", "completed_synchronous_runs":null, "zero_run":null, "observation_source":null}),
        };
        let record = serde_json::json!({
            "domain":"rz-pals-session-release-observation/1",
            "process_id":self.identity.process_id,
            "process_epoch":self.snapshot.process_epoch(),
            "process_run_id":self.identity.process_run_id,
            "lock_sha256":self.pinned.lock.canonical_sha256,
            "startup_config_sha256":self.pinned.config_sha256,
            "configured_provider":provider,
            "actual_execution_provider":{"status":"unknown"},
            "actual_device_placement":{"status":"unknown"},
            "native_owner_joined":finish.map(|receipt| receipt.native_owner_joined()),
            "session_release":session,
            "gpu_fence_or_nn_acceptance":false,
            "completed_nn_inputs":"unknown",
            "scope":"original process-local owner observation; not external device identity or performance evidence"
        });
        self.writer.lock().map_err(|_| fail("evidence writer poisoned"))?
            .write_json(OutputKind::SessionReleaseObservation, &record)
    }

    /// Preserve already observed root totals and the actual retained-owner
    /// readback after Root's finite finish attempt fails. This is a raw failure
    /// artifact with its own domain, never a normal termination receipt or an
    /// acceptance/performance claim. Missing receipt/retention remain Unknown;
    /// the Search failure's completion classification is not a GPU fence.
    pub(crate) fn publish_failed_termination(
        &self,
        finish: Option<&AdaptiveSessionProcessReceipt>,
        failure: &SearchSessionFailure,
        retained: Option<&AdaptiveSessionRetention>,
        completion: Option<&AdaptiveCompletionDiagnostic>,
    ) -> Result<ArtifactRef, ManifestError> {
        if let Some(receipt) = finish {
            need(
                receipt.epoch().0 == self.snapshot.process_epoch(),
                "failed termination receipt belongs to a different process epoch",
            )?;
        }
        let aggregate = self
            .aggregate
            .lock()
            .map_err(|_| fail("root observation aggregate poisoned"))?;
        let facts = failure_observation_facts(&aggregate, finish, failure, retained);
        let record = serde_json::json!({
            "schema_version":5,
            "domain":"rz-pals-failed-termination-observation-v5/1",
            "endpoint_id":model_endpoint(&self.pinned),
            "process_id":self.identity.process_id,
            "process_run_id":self.identity.process_run_id,
            "process_epoch":self.snapshot.process_epoch(),
            "source_commit":self.pinned.config.endpoint.source_commit,
            "lock_sha256":self.pinned.lock.canonical_sha256,
            "startup_config_sha256":self.pinned.config_sha256,
            "accepted":false,
            "performance_eligible":false,
            "strength_eligible":false,
            "normal_termination_receipt":false,
            "training_executed":false,
            "observations":facts,
            "pals_completion_diagnostic":completion_diagnostic_metadata(completion, self.snapshot.process_epoch()),
        });
        self.writer
            .lock()
            .map_err(|_| fail("evidence writer poisoned"))?
            .write_json(OutputKind::FailedTermination, &record)
    }
}

fn completion_diagnostic_metadata(
    actual: Option<&AdaptiveCompletionDiagnostic>,
    process_epoch: u64,
) -> serde_json::Value {
    if let Some(actual) = actual.filter(|actual| actual.process_epoch.0 == process_epoch) {
        return actual.as_json();
    }
    serde_json::json!({
        "schema":"rz-pals-semantic-completion-diagnostic/1",
        "scope":"latest returned adaptive run's final semantic result and driver return; not an active-root observation",
        "status":"unavailable",
        "last":null,
        "observed_driver_returns":null,
        "successful_driver_return_reason_counts":null,
        "semantic_observed_but_driver_failed":null,
        "unobserved_completion_returns":null,
        "unavailable_reason":"driver_diagnostic_unavailable_or_epoch_mismatch",
        "grants_output_or_acceptance_authority":false
    })
}

#[cfg(test)]
#[test]
fn missing_or_foreign_completion_is_unavailable_without_default_reason_or_authority() {
    let missing = completion_diagnostic_metadata(None, 31);
    assert_eq!(missing["status"], "unavailable");
    assert!(missing["last"].is_null());
    assert!(missing["successful_driver_return_reason_counts"].is_null());
    assert_eq!(missing["grants_output_or_acceptance_authority"], false);
    let foreign = AdaptiveCompletionDiagnostic {
        process_epoch: rz_contracts::ProcessEpoch(32),
        implementation: rz_contracts::Digest([4; 32]),
        last: None,
        observed_driver_returns: Some(0),
        successful_driver_return_reason_counts: Some(Default::default()),
        semantic_observed_but_driver_failed: Some(0),
        unobserved_completion_returns: Some(0),
        unavailable_reason: None,
    };
    assert_eq!(completion_diagnostic_metadata(Some(&foreign), 31), missing);
}

/// Pure serialization of caller-owned actual failure readbacks. Authored
/// fixtures may exercise this formatting but cannot mint process acceptance.
fn failure_observation_facts(
    aggregate: &RootAggregateV5,
    finish: Option<&AdaptiveSessionProcessReceipt>,
    failure: &SearchSessionFailure,
    retained: Option<&AdaptiveSessionRetention>,
) -> serde_json::Value {
    fn bounded(text: &str, max: usize) -> &str {
        if text.len() <= max {
            return text;
        }
        let mut end = max;
        while !text.is_char_boundary(end) {
            end -= 1;
        }
        &text[..end]
    }
    let finish_facts = finish.map_or_else(
        || serde_json::json!({"status":"unknown","scope":"no process finish receipt"}),
        |receipt| {
            serde_json::json!({
                "status":"observed",
                "scope":"actual session process finish producer, not Search error classification",
                "process_epoch":receipt.epoch().0,
                "native_owner_joined":receipt.native_owner_joined(),
                "native_physical_shutdown_confirmed":receipt.native_physical_shutdown_confirmed(),
                "store_archive_closed":receipt.store_archive_closed(),
                "permits_process_close":receipt.permits_process_close(),
                "root_authority_join":aggregate.require_process_finish_join(receipt.last_authority()).is_ok(),
            })
        },
    );
    let retention_facts = retained.map_or_else(
        || serde_json::json!({"status":"unknown","scope":"owner retention readback unavailable"}),
        |retained| {
            serde_json::json!({
                "status":"observed",
                "scope":"actual driver retained_owner_status after finite finish attempt; counts are not physical completion proof",
                "stage":bounded(retained.stage,128),
                "retained_cpu_engines":retained.retained_cpu_engines,
                "retained_cpu_workers":retained.retained_cpu_workers,
                "native_pending":retained.native_pending,
                "archive_owner_pending":retained.archive_owner_pending,
            })
        },
    );
    serde_json::json!({
        "search_failure":{
            "code":bounded(failure.code,128),
            "detail":bounded(&failure.detail,1024),
            "detail_truncated":failure.detail.len()>1024,
            "reported_search_physical_completion":match failure.physical_completion {
                DriverPhysicalCompletion::Confirmed=>"confirmed",
                DriverPhysicalCompletion::Unknown=>"unknown",
            },
            "scope":"Search failure classification only; never a native GPU/session shutdown fence",
        },
        "process_finish":finish_facts,
        "retained_owners":retention_facts,
        "root_counter_aggregate":aggregate.metadata(),
        "root_counter_scope":"only actual settled root snapshots previously accepted by this bridge; failed/live root work is not guessed or set to zero",
    })
}
fn model_endpoint(pinned: &PinnedPalsStartupV5) -> String {
    pinned.config.endpoint.id.clone()
}

// A finite number of fixed-width counters replaces a root-sized history. These
// names retain the manager's original scopes; no declaration fills an unknown.
macro_rules! manager_totals {
    ($($field:ident),+ $(,)?) => {
        #[derive(Clone, Copy, Debug, Default, Serialize)]
        struct ManagerTotals { $( $field: u64, )+ }
        impl ManagerTotals {
            fn add(&mut self, actual: AdaptiveManagerCounters) -> Result<(), ManifestError> {
                $( self.$field = self.$field.checked_add(actual.$field)
                    .ok_or_else(|| fail(concat!("manager counter overflow: ", stringify!($field))))?; )+
                Ok(())
            }
        }
    };
}
manager_totals!(
    native_preparations,
    submitted_role_tickets,
    physical_role_completions,
    completed_nn_inputs,
    nn_input_count_unknown,
    consumed_role_outputs,
    accepted_nn_inputs,
    accepted_nn_input_count_unknown,
    consumed_nn_inputs,
    consumed_nn_input_count_unknown,
    zero_nn_outputs_consumed,
    unused_role_results,
    canceled_preparations,
    cpu_slice_receipts,
    cpu_progress_publications,
    cpu_yields,
    cpu_yield_requests,
    cpu_yield_acknowledgements,
    cpu_natural_completions,
    validator_ticket_admissions,
    foreign_reports,
    foreign_reported_nodes,
    foreign_node_count_unknown,
    rejected_events,
    notification_waits,
);

macro_rules! ticket_totals {
    ($($field:ident),+ $(,)?) => {
        #[derive(Clone, Copy, Debug, Default, Serialize)]
        struct TicketTotals { $( $field: u64, )+ }
        impl TicketTotals {
            fn add(&mut self, actual: RoleTicketCumulativeObservation) -> Result<(), ManifestError> {
                $( self.$field = self.$field.checked_add(actual.$field)
                    .ok_or_else(|| fail(concat!("ticket counter overflow: ", stringify!($field))))?; )+
                Ok(())
            }
        }
    };
}
ticket_totals!(
    admitted,
    service_dispatched,
    physical_start_notifications,
    physically_completed,
    delivered,
    logically_accepted,
    terminal_before_service_dispatch,
    released,
    revoked,
    physical_unknown_events,
);

#[derive(Clone, Copy, Debug, Default, Serialize)]
struct PhasePeaks {
    prepared: u64,
    submitted: u64,
    physically_running: u64,
    completed: u64,
    delivered: u64,
    quarantined: u64,
}
impl PhasePeaks {
    fn merge(&mut self, peaks: RoleTicketPhaseOccupancy) {
        self.prepared = self.prepared.max(peaks.prepared);
        self.submitted = self.submitted.max(peaks.submitted);
        self.physically_running = self.physically_running.max(peaks.physically_running);
        self.completed = self.completed.max(peaks.completed);
        self.delivered = self.delivered.max(peaks.delivered);
        self.quarantined = self.quarantined.max(peaks.quarantined);
    }
}

#[derive(Clone, Copy, Debug, Default, Serialize)]
struct HistoryPeaks {
    role_requests: u32,
    cpu_executions: u32,
    replay_bindings: u32,
    frame_genealogy: u32,
    ranking_fanouts: u32,
    repair_questions: u32,
    validator_source_contexts: u32,
    validator_action_keys: u32,
}
impl HistoryPeaks {
    fn merge(
        &mut self,
        history: CoordinatorHistoryUsage,
        validator: ValidatorActionHistoryUsage,
    ) -> Result<(), ManifestError> {
        fn peak(old: &mut u32, actual: usize) -> Result<(), ManifestError> {
            *old = (*old)
                .max(u32::try_from(actual).map_err(|_| fail("history peak exceeds V5 width"))?);
            Ok(())
        }
        peak(&mut self.role_requests, history.role_requests)?;
        peak(&mut self.cpu_executions, history.cpu_executions)?;
        peak(&mut self.replay_bindings, history.replay_bindings)?;
        peak(&mut self.frame_genealogy, history.frame_genealogy)?;
        peak(&mut self.ranking_fanouts, history.ranking_fanouts)?;
        peak(&mut self.repair_questions, history.repair_questions)?;
        peak(
            &mut self.validator_source_contexts,
            validator.source_contexts,
        )?;
        peak(&mut self.validator_action_keys, validator.action_keys)?;
        Ok(())
    }
}

/// Only `from_actual` imports an opaque manager-produced closure. The smaller
/// private value also lets source tests exercise arithmetic without pretending
/// that authored fixture counters are an actual execution or a closure proof.
struct ClosedRootFacts {
    authority: SearchAuthority,
    close_mode: AdaptiveCloseMode,
    manager: AdaptiveManagerCounters,
    native_accounting: AdaptiveNativeAccountingSnapshot,
    tickets: RoleTicketCumulativeObservation,
    history: CoordinatorHistoryUsage,
    validator_history: ValidatorActionHistoryUsage,
    cpu_limit: u64,
    cpu_spent: u64,
    cpu_reserved: u64,
    registered_validator_profile: bool,
    completed_nn_inputs: Option<u64>,
    accepted_nn_inputs: Option<u64>,
    cache_evaluations_accepted: Option<u64>,
}
impl ClosedRootFacts {
    fn from_actual(actual: &AdaptiveRootObservationSnapshot) -> Result<Self, ManifestError> {
        let receipt = actual.shutdown_receipt();
        need(
            receipt.authority == actual.authority()
                && receipt.permits_semantic_closure()
                && receipt.pending_tickets == 0
                && receipt.native_cpu_reserved == 0
                && receipt.own_cpu_panicked == 0
                && !receipt.quarantined,
            "root snapshot is not its settled manager closure",
        )?;
        Ok(Self {
            authority: actual.authority(),
            close_mode: actual.close_mode(),
            manager: actual.manager_counters(),
            native_accounting: actual.native_accounting(),
            tickets: actual.role_ticket_accounting(),
            history: actual.history_peak(),
            validator_history: actual.validator_history_peak(),
            cpu_limit: actual.cpu_credit_limit(),
            cpu_spent: actual.cpu_spent(),
            cpu_reserved: actual.cpu_outstanding_reserved(),
            registered_validator_profile: actual.validator_loaded(),
            completed_nn_inputs: actual.completed_nn_inputs(),
            accepted_nn_inputs: actual.accepted_nn_inputs(),
            cache_evaluations_accepted: actual.cache_evaluations_accepted(),
        })
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct RootAggregateV5 {
    roots: u64,
    root_quiesce_closures: u64,
    process_shutdown_root_closures: u64,
    last_authority: Option<SearchAuthority>,
    manager: ManagerTotals,
    native_accounting: AdaptiveNativeAccountingSnapshot,
    tickets: TicketTotals,
    phase_peaks: PhasePeaks,
    held_peak: u64,
    history: HistoryPeaks,
    cpu_limit: u64,
    cpu_spent: u64,
    cpu_reserved: u64,
    cache_evaluations_accepted: u64,
    cache_accepted_unknown_roots: u64,
    physical_unknown_observed: bool,
    registered_validator_profile_seen: bool,
}
impl RootAggregateV5 {
    fn add(
        &mut self,
        actual: &AdaptiveRootObservationSnapshot,
        epoch: u64,
    ) -> Result<(), ManifestError> {
        self.add_facts(ClosedRootFacts::from_actual(actual)?, epoch)
    }
    fn add_facts(&mut self, actual: ClosedRootFacts, epoch: u64) -> Result<(), ManifestError> {
        // No partial mutation on overflow or an invalid/duplicate closure.
        let mut next = *self;
        need(
            next.roots < ROOT_OBSERVATIONS_MAX && actual.authority.epoch.0 == epoch,
            "root observation bound or process epoch differs",
        )?;
        if let Some(previous) = next.last_authority {
            need(
                actual.authority.implementation == previous.implementation
                    && actual.authority.game.0 >= previous.game.0
                    && (actual.authority.game.0 > previous.game.0
                        || actual.authority.root.0 > previous.root.0),
                "root closure duplicate, out of order or foreign source",
            )?;
        }
        need(
            actual.tickets.current == RoleTicketPhaseOccupancy::default()
                && actual.cpu_reserved == 0
                && actual.cpu_spent <= actual.cpu_limit,
            "closed root still owns work or exceeds native credit",
        )?;
        need(
            actual.completed_nn_inputs
                == (actual.manager.nn_input_count_unknown == 0)
                    .then_some(actual.manager.completed_nn_inputs)
                && actual.accepted_nn_inputs
                    == (actual.manager.accepted_nn_input_count_unknown == 0)
                        .then_some(actual.manager.accepted_nn_inputs),
            "NN getters do not join their original manager counters",
        )?;
        next.manager.add(actual.manager)?;
        next.native_accounting
            .checked_merge(actual.native_accounting)
            .map_err(|_| fail("native accounting aggregate overflow"))?;
        next.tickets.add(actual.tickets)?;
        next.phase_peaks.merge(actual.tickets.phase_peaks);
        next.held_peak = next.held_peak.max(actual.tickets.held_peak);
        next.history
            .merge(actual.history, actual.validator_history)?;
        next.cpu_limit = next
            .cpu_limit
            .checked_add(actual.cpu_limit)
            .ok_or_else(|| fail("CPU credit limit overflow"))?;
        next.cpu_spent = next
            .cpu_spent
            .checked_add(actual.cpu_spent)
            .ok_or_else(|| fail("CPU spent overflow"))?;
        next.cpu_reserved = next
            .cpu_reserved
            .checked_add(actual.cpu_reserved)
            .ok_or_else(|| fail("CPU reservation overflow"))?;
        match actual.cache_evaluations_accepted {
            Some(count) => {
                next.cache_evaluations_accepted = next
                    .cache_evaluations_accepted
                    .checked_add(count)
                    .ok_or_else(|| fail("cache acceptance overflow"))?
            }
            None => next.cache_accepted_unknown_roots += 1,
        }
        next.physical_unknown_observed |= actual.tickets.physical_unknown_events > 0;
        next.registered_validator_profile_seen |= actual.registered_validator_profile;
        match actual.close_mode {
            AdaptiveCloseMode::RootQuiesce => next.root_quiesce_closures += 1,
            AdaptiveCloseMode::ProcessShutdown => next.process_shutdown_root_closures += 1,
        }
        next.roots += 1;
        next.last_authority = Some(actual.authority);
        *self = next;
        Ok(())
    }
    fn metadata(&self) -> serde_json::Value {
        serde_json::json!({
            "schema":"rz-pals-actual-root-counter-aggregate-v5/1",
            "scope":"actual roots of one native process, each authority once; no historical numerical-run count",
            "roots":self.roots,
            "root_quiesce_closures":self.root_quiesce_closures,
            "process_shutdown_root_closures":self.process_shutdown_root_closures,
            "last_authority":self.last_authority.map(|a| serde_json::json!({
                "process_epoch":a.epoch.0,"game_generation":a.game.0,
                "root_generation":a.root.0,"implementation_sha256":hex(a.implementation.0)
            })),
            "manager_original_counters":self.manager,
            "native_owner_command_accounting":self.native_accounting,
            "ticket_original_lifecycle_totals":self.tickets,
            "independent_ticket_phase_peaks":self.phase_peaks,
            "simultaneous_ticket_held_peak":self.held_peak,
            "original_history_peaks":self.history,
            "summed_root_cpu_credit_limits":self.cpu_limit,
            "native_cpu_spent":self.cpu_spent,
            "native_cpu_outstanding_reserved":self.cpu_reserved,
            "physical_unknown_observed":self.physical_unknown_observed,
            "registered_validator_profile_seen":self.registered_validator_profile_seen,
            "cache_evaluations_accepted_known_subtotal":self.cache_evaluations_accepted,
            "cache_evaluations_accepted_unknown_roots":self.cache_accepted_unknown_roots,
            "dispatch_scope":"service_dispatched is original mailbox dispatch; not physical NN or cache dispatch",
            "v_physical_calls":self.validator_physically_called().map_or_else(
                ||serde_json::json!({"status":"unknown","scope":"incomplete actual original-command accounting; profile or NN count is not V-call evidence"}),
                |called|serde_json::json!({"status":"observed","value":called,"scope":"settled root command accounting only; final process owner join is still required"})
            ),
            "applied_resource_readback":"separate actual-owner resource envelopes; root work counters alone grant no resource application evidence"
        })
    }
    fn require_process_finish_join(
        &self,
        last: Option<SearchAuthority>,
    ) -> Result<(), ManifestError> {
        need(
            self.last_authority == last && !self.physical_unknown_observed,
            "process finish does not join every observed root or preserves unknown physical work",
        )
    }
    fn validator_physically_called(&self) -> Option<bool> {
        if self.manager.nn_input_count_unknown != 0
            || self.physical_unknown_observed
            || self
                .native_accounting
                .public_nn_runs_completed()
                .checked_add(self.native_accounting.role_nn_runs_completed())?
                != self.manager.completed_nn_inputs
        {
            return None;
        }
        self.native_accounting
            .validator_called_for_dispatches(self.tickets.service_dispatched)
    }
    fn complete_coordinator(&self, evidence: ArtifactRef) -> Option<PalsCoordinatorObservationV5> {
        // Mailbox dispatch is not physical NN dispatch. A public K/V hit still
        // executes the private role graph and never becomes a full-result cache
        // completion. Every original command must have an owner delta or an
        // explicit pre-run skip; legacy/missing accounting remains Unknown.
        if !self
            .native_accounting
            .physical_partition_complete_for_dispatches(self.tickets.service_dispatched)
            || self.cache_accepted_unknown_roots != 0
            || self.manager.nn_input_count_unknown != 0
            || self.manager.accepted_nn_input_count_unknown != 0
            || self.manager.consumed_nn_input_count_unknown != 0
            || !self.native_accounting.search_consumption_origin_known()
            || self.native_accounting.search_consumed_outputs()
                != self.manager.consumed_role_outputs
            || self.native_accounting.cache_evaluations_accepted()
                != Some(self.cache_evaluations_accepted)
            || self
                .native_accounting
                .public_nn_runs_completed()
                .checked_add(self.native_accounting.role_nn_runs_completed())?
                != self.manager.completed_nn_inputs
            || self.physical_unknown_observed
        {
            return None;
        }
        let terminal = self
            .tickets
            .terminal_before_service_dispatch
            .checked_add(self.native_accounting.terminal_without_nn_dispatch())?;
        let physical_tickets = self.native_accounting.physical_execution_tickets();
        if terminal.checked_add(physical_tickets)? != self.tickets.admitted
            || self.tickets.released != self.tickets.admitted
        {
            return None;
        }
        Some(PalsCoordinatorObservationV5 {
            role_request_history_peak: self.history.role_requests,
            cpu_execution_history_peak: self.history.cpu_executions,
            frame_genealogy_peak: self.history.frame_genealogy,
            ranking_fanout_history_peak: self.history.ranking_fanouts,
            validator_source_contexts_peak: observed(
                self.history.validator_source_contexts,
                "actual retained coordinator V source-context peak",
            ),
            validator_action_keys_peak: observed(
                self.history.validator_action_keys,
                "actual retained coordinator V action-key peak",
            ),
            view_submitted: self.tickets.admitted,
            eligible_view_consumed: self.manager.consumed_role_outputs,
            tickets_submitted: self.tickets.admitted,
            tickets_terminal_before_dispatch: terminal,
            tickets_cache_completed: 0,
            tickets_dispatched: physical_tickets,
            tickets_queued: 0,
            physical_in_flight: 0,
            physical_completion_unknown: 0,
            tickets_physically_completed: physical_tickets,
            tickets_accepted: self.native_accounting.owner_accepted_outputs(),
            tickets_released: self.tickets.released,
            tickets_quarantined: 0,
            nn_inputs_completed: self.manager.completed_nn_inputs,
            nn_inputs_accepted: self.manager.accepted_nn_inputs,
            cache_evaluations_accepted: self.cache_evaluations_accepted,
            search_evaluations_consumed: self.manager.consumed_role_outputs,
            unused_results: self.manager.unused_role_results,
            cpu_credit_limit: self.cpu_limit,
            cpu_spent: self.cpu_spent,
            cpu_outstanding_reserved: self.cpu_reserved,
            yield_requests: self.manager.cpu_yield_requests,
            yield_acks: self.manager.cpu_yield_acknowledgements,
            natural_completions: self.manager.cpu_natural_completions,
            // Root fences prove the above settled partition. Only the caller's
            // independently confirmed final process receipt fills these flags.
            physical_state: PalsPhysicalStateV3::Completed,
            owner_joined: false,
            archive_closed: false,
            evidence,
        })
    }
}

#[cfg(test)]
#[path = "observations/tests.rs"]
mod tests;
