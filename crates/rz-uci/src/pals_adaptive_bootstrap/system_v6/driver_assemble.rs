//! Staged pre-admission of the original V6 owners.
//!
//! This module is not a CLI entry point. Its optional admission remains in
//! the staging object owned by the original live supervisor; dropping that
//! object while physical completion is unknown is not a safe close.

use super::super::*;
use super::cpu_prepare::PalsSystemCpuPreparationV6;
use super::driver_prepare::{
    OriginalReceiptState, OriginalSnapshotState, PreparedV6SearchDeclaration,
    V6SearchPreparationOwner,
};
use super::{PalsSystemNativeStartV6, PalsSystemReceiptPublicationV6, PalsSystemStartupCustodyV6};
use crate::pals_adaptive::PalsV6Admission;
use crate::pals_adaptive_bootstrap::external::system_v6::ExternalBootstrapAdmissionV6;
use std::panic::{AssertUnwindSafe, catch_unwind};

/// Capabilities retained by a future completed driver transfer. A value of
/// this type alone is not proof of process or physical GPU completion.
#[must_use]
pub(super) struct V6TransferredOwners {
    pub(super) cpu: PalsSystemCpuPreparationV6,
    pub(super) parent: Arc<crate::rovent_bootstrap::launch::ConnectedParent>,
    pub(super) admission: Arc<crate::rovent_bootstrap::launch::PalsPostImageAdmissionV6>,
    pub(super) original_snapshot: OriginalSnapshotState,
    pub(super) original_receipt: OriginalReceiptState,
    pub(super) snapshot_receiver: mpsc::Receiver<
        Result<
            crate::pals_native::startup_system_v6::PalsSystemNativeSnapshotV6,
            rz_eval::error::BackendError,
        >,
    >,
    pub(super) receipt_receiver: mpsc::Receiver<PalsSystemReceiptPublicationV6>,
    pub(super) external_admission: Option<ExternalBootstrapAdmissionV6>,
    pub(super) rules_owners: Arc<OwnerRegistry>,
    pub(super) owner_marker: Arc<()>,
    pub(super) declaration_marker: Arc<()>,
    pub(super) archive_path: std::path::PathBuf,
    pub(super) declared_options: AdaptiveSessionOptions,
    pub(super) declared_registration: PalsCheckerRegistration,
}

#[must_use]
pub(super) struct V6AssembledDriver {
    pub(super) driver: ActualDriver,
    pub(super) original: V6TransferredOwners,
}

/// The last attempted stage, not a substitute for the original capability.
/// A panic in Creating/Opening/Constructing/Installing leaves the outcome of
/// that operation unknown, even when an earlier stage completed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum V6ArchiveStage {
    Unplanned,
    Planned,
    Creating,
    Created,
    Opening,
    Opened,
    ConstructingCapability,
    CapabilityHeld,
    Installing,
    Installed,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum V6ExternalStage {
    Unplanned,
    NotRequested,
    Starting,
    Started,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum V6PreparationObservation {
    Prepared,
    Failed,
    Panicked,
    AlreadyAttempted,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum V6AdmissionObservation {
    Admitted,
    Rejected,
    PanickedSourcesRetained,
    PanickedOriginalStageRetained,
    PanickedCustodyUnknown,
    CustodyUnknown,
    NotPrepared,
    AlreadyAttempted,
}

/// Owns the actual source, declaration, started worker and held archive root.
/// A fatal panic is caught only within `prepare_once`; outer code must still
/// retain this object and the process. Abort/OOM, OS termination and unwinding
/// inside a worker's own start routine are not attested or repaired here.
#[must_use = "retain the original supervisor and this owner while physical completion is unknown"]
pub(super) struct V6PreAdmissionOwner {
    owner: Option<V6SearchPreparationOwner>,
    declaration: Option<PreparedV6SearchDeclaration>,
    pub(super) rules_owners: Arc<OwnerRegistry>,
    pub(super) archive_stage: V6ArchiveStage,
    pub(super) archive_name: Option<String>,
    pub(super) archive_path: Option<std::path::PathBuf>,
    pub(super) archive_directory: Option<cap_std::fs::Dir>,
    pub(super) archive_capability: Option<ArchiveRootCapability>,
    pub(super) external_stage: V6ExternalStage,
    pub(super) external_worker: Option<ExternalManagerWorker>,
    pub(super) first_error: Option<AdaptiveBootstrapError>,
    pub(super) panic_payload: Option<Box<dyn std::any::Any + Send>>,
    pub(super) admission_sources: Option<AdaptiveSessionSources<ActualBackend>>,
    pub(super) transferred: Option<V6TransferredOwners>,
    pub(super) admitted_driver: Option<ActualDriver>,
    pub(super) admission_failure: Option<SearchSessionFailure>,
    pub(super) admission_panic: Option<Box<dyn std::any::Any + Send>>,
    admission_attempted: bool,
    attempted: bool,
    prepared: bool,
}

impl V6PreAdmissionOwner {
    /// The exact service remains in one of these original slots on rejected
    /// admission. An admitted driver owns its own shutdown path instead.
    pub(super) fn failed_native_service(&mut self) -> Option<&mut ActualService> {
        if let Some(owner) = self.owner.as_mut() {
            return Some(&mut owner.custody.native.service);
        }
        self.admission_sources
            .as_mut()
            .map(|source| &mut source.native)
    }

    /// No process, archive or driver is created by construction itself.
    pub(super) fn new(
        owner: V6SearchPreparationOwner,
        declaration: PreparedV6SearchDeclaration,
        rules_owners: Arc<OwnerRegistry>,
    ) -> Self {
        Self {
            owner: Some(owner),
            declaration: Some(declaration),
            rules_owners,
            archive_stage: V6ArchiveStage::Unplanned,
            archive_name: None,
            archive_path: None,
            archive_directory: None,
            archive_capability: None,
            external_stage: V6ExternalStage::Unplanned,
            external_worker: None,
            first_error: None,
            panic_payload: None,
            admission_sources: None,
            transferred: None,
            admitted_driver: None,
            admission_failure: None,
            admission_panic: None,
            admission_attempted: false,
            attempted: false,
            prepared: false,
        }
    }

    /// One bounded attempt; no automatic retry after an error or panic.
    /// The original owners remain in `self` on each caught outcome. This
    /// does not cover a panic within an external worker after thread creation.
    pub(super) fn prepare_once(&mut self) -> V6PreparationObservation {
        if self.attempted {
            return V6PreparationObservation::AlreadyAttempted;
        }
        self.attempted = true;
        match catch_unwind(AssertUnwindSafe(|| self.prepare_inner())) {
            Ok(Ok(())) => {
                self.prepared = true;
                V6PreparationObservation::Prepared
            }
            Ok(Err(error)) => {
                self.first_error = Some(error);
                V6PreparationObservation::Failed
            }
            Err(payload) => {
                self.panic_payload = Some(payload);
                V6PreparationObservation::Panicked
            }
        }
    }

    /// Transfer only to caller-owned slots and invoke the guarded admission
    /// once. On any outcome the supervisor retains this entire object.
    /// Admission success is neither UCI execution nor physical completion.
    pub(super) fn admit_once(&mut self) -> V6AdmissionObservation {
        if self.admission_attempted {
            return V6AdmissionObservation::AlreadyAttempted;
        }
        if !self.prepared {
            return V6AdmissionObservation::NotPrepared;
        }
        self.admission_attempted = true;
        match catch_unwind(AssertUnwindSafe(|| self.admit_inner())) {
            Ok(Ok(PalsV6Admission::Admitted(driver))) => {
                self.admitted_driver = Some(driver);
                if self.admission_sources.is_none() && self.transferred.is_some() {
                    V6AdmissionObservation::Admitted
                } else {
                    V6AdmissionObservation::CustodyUnknown
                }
            }
            Ok(Ok(PalsV6Admission::Rejected(error))) => {
                self.admission_failure = Some(error);
                if self.admission_sources.is_some() && self.transferred.is_some() {
                    V6AdmissionObservation::Rejected
                } else {
                    V6AdmissionObservation::CustodyUnknown
                }
            }
            Ok(Ok(PalsV6Admission::Panicked {
                payload,
                source_retained,
            })) => {
                self.admission_panic = Some(payload);
                if source_retained && self.admission_sources.is_some() {
                    V6AdmissionObservation::PanickedSourcesRetained
                } else {
                    V6AdmissionObservation::PanickedCustodyUnknown
                }
            }
            Ok(Err(error)) => {
                self.first_error = Some(error);
                V6AdmissionObservation::Rejected
            }
            Err(payload) => {
                self.admission_panic = Some(payload);
                if self.admission_sources.is_some() {
                    V6AdmissionObservation::PanickedSourcesRetained
                } else if self.owner.is_some() && self.declaration.is_some() {
                    V6AdmissionObservation::PanickedOriginalStageRetained
                } else {
                    V6AdmissionObservation::PanickedCustodyUnknown
                }
            }
        }
    }

    fn admit_inner(&mut self) -> Result<PalsV6Admission<ActualBackend>, AdaptiveBootstrapError> {
        self.check_original_admission()?;
        if self.archive_stage != V6ArchiveStage::Installed
            || !matches!(
                self.external_stage,
                V6ExternalStage::NotRequested | V6ExternalStage::Started
            )
            || self.owner.is_none()
            || self.declaration.is_none()
            || self.archive_path.is_none()
            || self.transferred.is_some()
            || self.admission_sources.is_some()
            || self.admitted_driver.is_some()
        {
            return Err(boundary("SystemV6TransferState"));
        }
        let declaration = self.declaration.as_ref().expect("checked declaration");
        if declaration.external.is_some() != self.external_worker.is_some() {
            return Err(boundary("SystemV6ExternalOwnerMissing"));
        }
        // These clones and the Arc allocation can panic: perform them before
        // removing any original owner from this caller-owned staging object.
        let options = declaration.options.clone();
        let registration = declaration.registration.clone();
        let owner = self.owner.as_ref().expect("checked owner");
        let limits = owner.custody.cpu.limits;
        let clock = owner.custody.cpu.clock.clone();
        let archive_path = self.archive_path.as_ref().expect("checked path").clone();
        let original_rules = Arc::clone(&self.rules_owners);
        let issuer_rules = Arc::clone(&self.rules_owners);
        let issuer: AdaptiveRulesOwnerIssuer = Arc::new(move || issuer_rules.allocate());
        let hooks = AdaptiveSessionCollectionHooks::default();

        // No allocation, callback, validation or fallible operation between
        // these checked moves and storing both original custody slots.
        let V6SearchPreparationOwner {
            custody,
            snapshot,
            receipt,
            owner_marker,
        } = self.owner.take().expect("checked owner");
        let PalsSystemStartupCustodyV6 {
            mut cpu,
            parent,
            admission,
            native,
        } = custody;
        let PalsSystemNativeStartV6 {
            service,
            receipt_publication,
            startup_snapshot,
        } = native;
        let PreparedV6SearchDeclaration {
            original_owner,
            engine,
            metadata,
            registration: declared_registration,
            options: declared_options,
            external,
        } = self.declaration.take().expect("checked declaration");
        let own_workers = std::mem::take(&mut cpu.own_workers);
        let external_worker = self.external_worker.take();
        self.transferred = Some(V6TransferredOwners {
            cpu,
            parent,
            admission,
            original_snapshot: snapshot,
            original_receipt: receipt,
            snapshot_receiver: startup_snapshot,
            receipt_receiver: receipt_publication,
            external_admission: external,
            rules_owners: original_rules,
            owner_marker,
            declaration_marker: original_owner,
            archive_path,
            declared_options,
            declared_registration,
        });
        self.admission_sources = Some(AdaptiveSessionSources {
            native: service,
            engine,
            cpu: own_workers,
            external: external_worker,
            checker_metadata: metadata,
        });

        Ok(AdaptiveSessionDriver::admit_pals_system_v6_in_slot(
            options,
            limits,
            &mut self.admission_sources,
            clock,
            issuer,
            registration,
            None,
            None,
            hooks,
        ))
    }

    fn check_original_admission(&self) -> Result<(), AdaptiveBootstrapError> {
        let owner = self
            .owner
            .as_ref()
            .ok_or_else(|| boundary("SystemV6OwnerTransferred"))?;
        let cpu = &owner.custody.cpu;
        require_original_startup_window(cpu.original_until, &cpu.original_cancelled)?;
        cpu.original_source
            .check_native_admission(&owner.custody.parent, &owner.custody.admission)
            .map_err(|error| failure("SystemV6AssemblyAdmission", &error))?;
        if owner.custody.native.service.startup_status()
            != crate::pals_native::ticketed::NativeFactoryStartupStatus::Ready
        {
            return Err(boundary("SystemV6NativeFactoryNotReady"));
        }
        Ok(())
    }

    fn prepare_inner(&mut self) -> Result<(), AdaptiveBootstrapError> {
        let owner = self
            .owner
            .as_ref()
            .ok_or_else(|| boundary("SystemV6OwnerTransferred"))?;
        let declaration = self
            .declaration
            .as_ref()
            .ok_or_else(|| boundary("SystemV6DeclarationTransferred"))?;
        if !Arc::ptr_eq(&owner.owner_marker, &declaration.original_owner) {
            return Err(boundary("SystemV6DeclarationOwnerMismatch"));
        }
        let cpu = &owner.custody.cpu;
        if cpu.original_source.lock().config.endpoint.mode != PalsEngineModeV5::Deployment {
            return Err(boundary("SystemV6CollectionHooksUnsupported"));
        }
        self.check_original_admission()?;

        let cpu = &self.owner.as_ref().expect("owner checked").custody.cpu;
        let name = format!(
            "adaptive-archive-{}-{}",
            std::process::id(),
            cpu.clock.epoch().0
        );
        let path = cpu
            .original_source
            .private_cwd()
            .join(&cpu.original_source.lock().config.runtime_directory)
            .join(&name);
        self.archive_name = Some(name);
        self.archive_path = Some(path);
        self.archive_stage = V6ArchiveStage::Planned;
        let parent = cpu
            .original_source
            .runtime_directory_capability()
            .map_err(|error| failure("SystemV6ArchiveRoot", &error))?;
        let name = self.archive_name.as_ref().expect("archive planned");
        self.archive_stage = V6ArchiveStage::Creating;
        parent
            .create_dir(name)
            .map_err(|error| failure("SystemV6ArchiveCreate", &error))?;
        self.archive_stage = V6ArchiveStage::Created;
        self.archive_stage = V6ArchiveStage::Opening;
        self.archive_directory = Some(
            parent
                .open_dir_nofollow(name)
                .map_err(|error| failure("SystemV6ArchiveOpen", &error))?,
        );
        self.archive_stage = V6ArchiveStage::Opened;
        self.archive_stage = V6ArchiveStage::ConstructingCapability;
        // Keep the first held directory in the stage even if validation of
        // its cloned handle fails or unwinds.
        let directory = self
            .archive_directory
            .as_ref()
            .expect("archive opened")
            .try_clone()
            .map_err(|error| failure("SystemV6ArchiveClone", &error))?;
        let path = self.archive_path.as_ref().expect("archive planned").clone();
        self.archive_capability = Some(
            ArchiveRootCapability::from_directory(path, directory)
                .map_err(|error| failure("SystemV6ArchiveCapability", &error))?,
        );
        self.archive_stage = V6ArchiveStage::CapabilityHeld;
        let path = self.archive_path.as_ref().expect("archive planned").clone();
        let capability = self
            .archive_capability
            .as_ref()
            .expect("archive capability")
            .clone();
        let config = ArchiveConfig::new(path, cpu.original_source.private_cwd().join("inputs"));
        self.archive_stage = V6ArchiveStage::Installing;
        self.declaration
            .as_mut()
            .expect("declaration checked")
            .engine
            .enable_archive_with_capability(config, capability)
            .map_err(|error| failure("SystemV6ArchiveEnable", &error))?;
        self.archive_stage = V6ArchiveStage::Installed;

        if let Some(external) = &self
            .declaration
            .as_ref()
            .expect("declaration checked")
            .external
        {
            self.check_original_admission()?;
            let cpu = &self.owner.as_ref().expect("owner checked").custody.cpu;
            let worker_owner = cpu
                .intent
                .external_worker_owner(cpu.clock.epoch(), Instant::now())
                .map_err(|error| failure("SystemV6ExternalOwnerIntent", &error))?;
            let runtime_clock = cpu.clock.runtime_clock();
            let clock = ExternalWorkerClock {
                domain: cpu.clock.domain(),
                origin: runtime_clock.origin(),
            };
            // Complete every clone before a worker can be spawned. After the
            // call returns, only infallible moves precede storage of its handle.
            let config = external.config().clone();
            let spawn_config = config.clone();
            let launch_pin = external.launch_pin();
            self.external_stage = V6ExternalStage::Starting;
            let worker = ExternalCpuWorker::spawn_pinned(
                spawn_config,
                worker_owner,
                clock,
                cpu.original_until,
                launch_pin,
            )
            .map_err(|error| failure("SystemV6ExternalOwner", &error))?;
            self.external_worker = Some(ExternalManagerWorker {
                worker,
                config,
                owner: worker_owner,
            });
            self.external_stage = V6ExternalStage::Started;
        } else {
            self.external_stage = V6ExternalStage::NotRequested;
        }
        self.check_original_admission()?;
        Ok(())
    }
}
