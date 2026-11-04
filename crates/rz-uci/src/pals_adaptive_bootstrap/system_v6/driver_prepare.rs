//! Prepare the V6 search-side declaration using the original startup owner.
//!
//! This creates no worker, archive, driver or native session.  It borrows the
//! existing custody so that an error cannot discard a live GPU owner.  Numeric
//! validation, actual resource readback, physical completion and provenance
//! observations are intentionally not turned into successful execution.

use super::super::*;
use super::{PalsSystemReceiptPublicationV6, PalsSystemStartupCustodyV6};
use crate::pals_adaptive_bootstrap::external::system_v6::{
    ExternalBootstrapAdmissionV6, load_external_cpu_v6,
};
use crate::pals_native::startup_system_v6::PalsSystemNativeSnapshotV6;

/// Pure search-side preparation; the original custody and all workers remain
/// in the caller.  The admitted external FD, if present, is retained here and
/// is not itself evidence of process execution or checker readiness.
#[must_use]
pub(super) struct PreparedV6SearchDeclaration {
    pub(super) original_owner: Arc<()>,
    pub(super) engine: PalsEngine<AdaptivePortModel>,
    pub(super) metadata: crate::pals_adaptive::AdaptiveCheckerMetadata,
    pub(super) registration: PalsCheckerRegistration,
    pub(super) options: AdaptiveSessionOptions,
    pub(super) external: Option<ExternalBootstrapAdmissionV6>,
}

/// The original factory's one snapshot is consumed once and kept beside the
/// same native service.  A missing diagnostic receipt is tracked separately
/// and never becomes a functional startup failure or physical completion.
pub(super) enum OriginalSnapshotState {
    Pending,
    Ready(PalsSystemNativeSnapshotV6),
    Failed(rz_eval::error::BackendError),
    Disconnected,
}

pub(super) enum OriginalReceiptState {
    Pending,
    Observed(PalsSystemReceiptPublicationV6),
    Disconnected,
}

#[must_use]
pub(super) struct V6SearchPreparationOwner {
    pub(super) custody: PalsSystemStartupCustodyV6,
    pub(super) snapshot: OriginalSnapshotState,
    pub(super) receipt: OriginalReceiptState,
    pub(super) owner_marker: Arc<()>,
}

impl V6SearchPreparationOwner {
    pub(super) fn new(custody: PalsSystemStartupCustodyV6) -> Self {
        Self {
            custody,
            snapshot: OriginalSnapshotState::Pending,
            receipt: OriginalReceiptState::Pending,
            owner_marker: Arc::new(()),
        }
    }

    /// Read each original receiver without blocking.  Store every received
    /// object exactly once, including receipt-owned capabilities and errors.
    pub(super) fn poll_original_startup(&mut self) {
        if matches!(self.snapshot, OriginalSnapshotState::Pending) {
            match self.custody.native.startup_snapshot.try_recv() {
                Ok(Ok(snapshot)) => self.snapshot = OriginalSnapshotState::Ready(snapshot),
                Ok(Err(error)) => self.snapshot = OriginalSnapshotState::Failed(error),
                Err(mpsc::TryRecvError::Empty) => {}
                Err(mpsc::TryRecvError::Disconnected) => {
                    self.snapshot = OriginalSnapshotState::Disconnected
                }
            }
        }
        if matches!(self.receipt, OriginalReceiptState::Pending) {
            match self.custody.native.receipt_publication.try_recv() {
                Ok(receipt) => self.receipt = OriginalReceiptState::Observed(receipt),
                Err(mpsc::TryRecvError::Empty) => {}
                Err(mpsc::TryRecvError::Disconnected) => {
                    self.receipt = OriginalReceiptState::Disconnected
                }
            }
        }
    }

    /// The caller retains `self` on all outcomes, including snapshot failure,
    /// disconnect and declaration failure.  Polling the receipt is optional
    /// diagnostics; its absence does not prevent this functional preparation.
    pub(super) fn prepare_search_declaration(
        &self,
        identity: EngineIdentity,
    ) -> Result<PreparedV6SearchDeclaration, AdaptiveBootstrapError> {
        use crate::pals_native::ticketed::NativeFactoryStartupStatus;

        // The factory can publish its snapshot before the ticket backend is
        // returned. Keep both observations: a snapshot alone is not Ready.
        match self.custody.native.service.startup_status() {
            NativeFactoryStartupStatus::Ready => {}
            NativeFactoryStartupStatus::Pending => {
                return Err(boundary("SystemV6FactoryPending"));
            }
            NativeFactoryStartupStatus::BeforeNative => {
                return Err(boundary("SystemV6FactoryBeforeNative"));
            }
            NativeFactoryStartupStatus::PhysicalUnknown => {
                return Err(boundary("SystemV6FactoryPhysicalUnknown"));
            }
            NativeFactoryStartupStatus::Unobserved => {
                return Err(boundary("SystemV6FactoryUnobserved"));
            }
        }
        let snapshot = match &self.snapshot {
            OriginalSnapshotState::Pending => return Err(boundary("SystemV6SnapshotPending")),
            OriginalSnapshotState::Ready(snapshot) => snapshot,
            OriginalSnapshotState::Failed(error) => {
                return Err(failure("SystemV6SnapshotFailed", error));
            }
            OriginalSnapshotState::Disconnected => {
                return Err(boundary("SystemV6SnapshotDisconnected"));
            }
        };
        prepare_search_declaration(
            &self.custody,
            snapshot,
            identity,
            Arc::clone(&self.owner_marker),
        )
    }
}

/// This private helper only receives the snapshot obtained from the exact
/// original factory receiver above.  No V5 numerical receipt is forged.
fn prepare_search_declaration(
    custody: &PalsSystemStartupCustodyV6,
    snapshot: &PalsSystemNativeSnapshotV6,
    identity: EngineIdentity,
    original_owner: Arc<()>,
) -> Result<PreparedV6SearchDeclaration, AdaptiveBootstrapError> {
    let cpu = &custody.cpu;
    let source = &cpu.original_source;
    require_original_startup_window(cpu.original_until, &cpu.original_cancelled)?;
    source
        .check_native_admission(&custody.parent, &custody.admission)
        .map_err(|error| failure("SystemV6SearchAdmission", &error))?;
    if snapshot.process_generation != cpu.clock.epoch().0 {
        return Err(boundary("SystemV6SnapshotProcessEpoch"));
    }
    let declaration = &source.lock().config.endpoint;
    if declaration.context.selection != PalsViewSelectionV5::VirtualReview
        || declaration.model.profile != "virtual_review_v1"
        || declaration.resources.topology != PalsTopologyPolicyV5::SharedControl
    {
        return Err(boundary("UnsupportedSystemV6SearchProfile"));
    }
    let external = load_external_cpu_v6(
        Arc::clone(source),
        cpu.original_until,
        &cpu.original_cancelled,
    )
    .map_err(|error| failure("SystemV6ExternalAdmission", &error))?;
    cpu.intent
        .check_external_loaded(external.is_some(), cpu.clock.epoch(), Instant::now())
        .map_err(|error| failure("SystemV6ExternalIntent", &error))?;
    let metadata = if let Some(external) = &external {
        let checker = ExternalUciCpuChecker::create(external.config().clone())
            .map_err(|error| failure("SystemV6ExternalMetadata", &error))?;
        crate::pals_adaptive::AdaptiveCheckerMetadata::capture(&checker)
            .map_err(|error| failure("SystemV6ExternalMetadata", &error))?
    } else {
        let own = cpu
            .own_workers
            .first()
            .ok_or_else(|| boundary("SystemV6OwnCpuMissing"))?;
        crate::pals_adaptive::AdaptiveCheckerMetadata::capture_owned(&own.engine)
            .map_err(|error| failure("SystemV6OwnCpuMetadata", &error))?
    };
    let model = AdaptivePortModel::new_with_namespace(
        snapshot.role_model_identity(),
        Some(
            snapshot
                .model_value_identity()
                .map_err(|error| failure("SystemV6ModelValue", &error))?,
        ),
        snapshot
            .adaptive_model_namespace()
            .map_err(|error| failure("SystemV6ModelNamespace", &error))?,
    )
    .map_err(|error| failure("SystemV6ModelMirror", &error))?;
    let settings = &cpu.search.settings;
    let mirror = metadata.clone().into_mirror();
    let registration =
        PalsCheckerRegistration::capture_with_resolver(&mirror, &model, settings.resolver)
            .map_err(|error| failure("SystemV6CheckerRegistration", &error))?;
    let engine = PalsEngine::new_with_boxed_checker_and_policies(
        settings.config.clone(),
        model,
        Box::new(mirror),
        settings.resolver,
        settings.post_repair_recheck,
    )
    .map_err(|error| failure("SystemV6SearchStore", &error))?
    .with_selected_review_context(
        ReviewSelectionPolicy {
            mode: ReviewAnchorMode::AnchoredCompact,
            token_budget: declaration.context.token_budget as u16,
            max_records: declaration.context.selected_records_max.min(64) as u16,
            max_anchors: declaration.context.anchors_max as u8,
            required_bundle: None,
        },
        ReviewIndexLimits {
            active_bundles: declaration.context.active_bundles_max as usize,
            records_per_bundle: declaration.context.selected_records_max.min(64) as usize,
        },
    )
    .map_err(|error| failure("SystemV6ReviewIndex", &error))?;
    let endpoints = external.as_ref().map_or_else(
        || match declaration.mode {
            PalsEngineModeV5::Collection => CpuEndpointSelection::collection(),
            PalsEngineModeV5::Deployment => CpuEndpointSelection::deployment(),
        },
        |external| external.selection(),
    );
    let policy = match declaration.coordinator.policy {
        PalsCoordinatorPolicyV5::Synchronous => CoordinatorPolicy::Synchronous,
        PalsCoordinatorPolicyV5::CoordinatorFixed => CoordinatorPolicy::Fixed,
        PalsCoordinatorPolicyV5::CoordinatorAdaptive => CoordinatorPolicy::Adaptive,
    };
    let controller = ControllerConfig {
        policy,
        fixed_node_credit: declaration.coordinator.slices.slice_native_nodes_max,
        fixed_soft_lease: Duration::from_micros(declaration.coordinator.slices.soft_lease_micros),
        initial_workers: declaration.coordinator.slices.workers as u8,
        max_workers: declaration.coordinator.slices.workers as u8,
        ..ControllerConfig::default()
    };
    // Source keys come from the running binary, never from the lock's claims.
    let compiled = compiled_source_components_v5()
        .map_err(|error| failure("SystemV6SourceInventory", &error))?;
    let digest = |name| -> Result<Digest, AdaptiveBootstrapError> {
        parse_sha256(
            compiled
                .get(name)
                .ok_or_else(|| boundary("SystemV6SourceComponentMissing"))?,
        )
        .map(Digest)
        .map_err(|error| failure("SystemV6SourceComponent", &error))
    };
    let options = AdaptiveSessionOptions {
        implementation: digest("coordinator")?,
        identity,
        max_rounds: settings.limits.max_rounds,
        max_cpu_nodes: settings.limits.max_cpu_nodes,
        cpu_depth: settings.limits.cpu_depth,
        token_budget: declaration.context.token_budget as u16,
        policy,
        endpoints,
        controller,
        device_digest: snapshot.backend,
        cpu_device_digest: digest("resource_boundary")?,
        external_policy: external
            .as_ref()
            .map_or_else(Default::default, |item| item.slice_policy()),
    };
    require_original_startup_window(cpu.original_until, &cpu.original_cancelled)?;
    source
        .check_native_admission(&custody.parent, &custody.admission)
        .map_err(|error| failure("SystemV6SearchAdmission", &error))?;
    Ok(PreparedV6SearchDeclaration {
        original_owner,
        engine,
        metadata,
        registration,
        options,
        external,
    })
}
