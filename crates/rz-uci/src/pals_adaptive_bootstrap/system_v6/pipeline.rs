//! Unconnected V6 orchestration using the original supervisor and factory.
//!
//! The caller must retain this object and the process whenever physical
//! completion is unknown. It is not a replacement for the outer supervisor,
//! an executable entry point, an OS resource observation, or a GPU fence.
use super::super::*;
use super::before_native_recovery::{BeforeNativeRecoveryObservation, BeforeNativeRecoveryV6};
use super::cpu_prepare::{PalsSystemCpuPreparationV6, prepare_cpu_v6};
use super::driver_assemble::{
    V6AdmissionObservation, V6PreAdmissionOwner, V6PreparationObservation,
};
use super::driver_lifecycle::{FinishObservationV6, PalsSystemDriverLifecycleV6};
use super::driver_prepare::{OriginalSnapshotState, V6SearchPreparationOwner};
use super::failed_startup_drain::{FailedNativeDrainObservation, V6FailedStartupDrain};
use super::{PalsSystemStartupRejectionV6, start_owned_system_native_v6};
use crate::pals_adaptive_startup::PinnedPalsSystemStartupV6;
use crate::pals_native::ticketed::NativeFactoryStartupStatus;
use crate::rovent_bootstrap::launch::{ConnectedParent, PalsPostImageAdmissionV6};
use std::any::Any;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::PathBuf;

/// These are orchestration observations, never claims of GPU or game success.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum V6PipelineObservation {
    Initial,
    NativeStartInProgress,
    WaitingNative,
    ReadyForService,
    ServiceInProgress,
    ServiceReturned,
    Failed,
    AlreadyAttempted,
}

/// The original error stays attached to the stage that encountered it.
pub(super) enum V6PipelineFailure {
    Parent(std::io::Error),
    ParentGrant(std::io::Error),
    Bootstrap(AdaptiveBootstrapError),
    FactoryRejected,
    FactoryNotReady(NativeFactoryStartupStatus),
    SnapshotDisconnected,
    Preparation(V6PreparationObservation),
    Admission(V6AdmissionObservation),
    LifecycleAdmission,
    ServicePanic,
}

/// The captured parent PID must be obtained before publishing a parent
/// endpoint; the source is the original opened and pinned startup owner.
#[must_use = "retain this pipeline and its original process until ownership is resolved"]
pub(super) struct V6Pipeline {
    source: Arc<PinnedPalsSystemStartupV6>,
    clock: ProcessClock,
    parent_socket: PathBuf,
    parent_pid: i32,
    identity: EngineIdentity,
    rules_owners: Arc<OwnerRegistry>,
    settings: EngineSettings,
    original_cleanup_until: Instant,
    parent: Option<Arc<ConnectedParent>>,
    admission: Option<Arc<PalsPostImageAdmissionV6>>,
    cpu: Option<PalsSystemCpuPreparationV6>,
    startup_rejection: Option<PalsSystemStartupRejectionV6>,
    search_owner: Option<V6SearchPreparationOwner>,
    pre_admission: Option<V6PreAdmissionOwner>,
    lifecycle: Option<PalsSystemDriverLifecycleV6>,
    service_panic: Option<Box<dyn Any + Send>>,
    failure: Option<V6PipelineFailure>,
    failed_native_drain: Option<V6FailedStartupDrain>,
    first_failed_native_drain: Option<FailedNativeDrainObservation>,
    before_native_recovery: Option<BeforeNativeRecoveryV6>,
    state: V6PipelineObservation,
}

impl V6Pipeline {
    pub(super) fn new(
        source: Arc<PinnedPalsSystemStartupV6>,
        clock: ProcessClock,
        parent_socket: PathBuf,
        parent_pid: i32,
        identity: EngineIdentity,
        rules_owners: Arc<OwnerRegistry>,
        settings: EngineSettings,
        original_cleanup_until: Instant,
    ) -> Self {
        let window = &source.lock().config.window;
        let registered_cleanup_until = source
            .original_started()
            .checked_add(Duration::from_millis(window.wall_time_max_ms))
            .and_then(|end| end.checked_add(Duration::from_millis(window.cleanup_max_ms)))
            // An unrepresentable bound cannot authorize a later deadline.
            .unwrap_or_else(|| source.original_started());
        Self {
            source,
            clock,
            parent_socket,
            parent_pid,
            identity,
            rules_owners,
            settings,
            original_cleanup_until: original_cleanup_until.min(registered_cleanup_until),
            parent: None,
            admission: None,
            cpu: None,
            startup_rejection: None,
            search_owner: None,
            pre_admission: None,
            lifecycle: None,
            service_panic: None,
            failure: None,
            failed_native_drain: None,
            first_failed_native_drain: None,
            before_native_recovery: None,
            state: V6PipelineObservation::Initial,
        }
    }

    pub(super) fn observation(&self) -> V6PipelineObservation {
        self.state
    }
    pub(super) fn failure(&self) -> Option<&V6PipelineFailure> {
        self.failure.as_ref()
    }
    pub(super) fn startup_rejection(&self) -> Option<&PalsSystemStartupRejectionV6> {
        self.startup_rejection.as_ref()
    }
    pub(super) fn search_owner(&self) -> Option<&V6SearchPreparationOwner> {
        self.search_owner.as_ref()
    }
    pub(super) fn pre_admission(&self) -> Option<&V6PreAdmissionOwner> {
        self.pre_admission.as_ref()
    }
    pub(super) fn lifecycle(&self) -> Option<&PalsSystemDriverLifecycleV6> {
        self.lifecycle.as_ref()
    }
    pub(super) fn lifecycle_mut(&mut self) -> Option<&mut PalsSystemDriverLifecycleV6> {
        self.lifecycle.as_mut()
    }
    pub(super) fn service_panic(&self) -> Option<&(dyn Any + Send)> {
        self.service_panic.as_deref()
    }
    pub(super) fn first_failed_native_drain(&self) -> Option<FailedNativeDrainObservation> {
        self.first_failed_native_drain
    }
    pub(super) fn failed_native_drain(&self) -> Option<&V6FailedStartupDrain> {
        self.failed_native_drain.as_ref()
    }
    pub(super) fn before_native_recovery(&self) -> Option<&BeforeNativeRecoveryV6> {
        self.before_native_recovery.as_ref()
    }
    /// Reuse the exact service and recovery key after an actual BeforeNative
    /// startup outcome; this never asserts any GPU-wide fence.
    pub(super) fn poll_before_native_recovery(
        &mut self,
    ) -> Option<BeforeNativeRecoveryObservation> {
        self.before_native_recovery
            .as_mut()
            .map(BeforeNativeRecoveryV6::poll_once)
    }
    /// Request shutdown of the same retained native worker after a failed
    /// startup. The caller supplies the original registered cleanup deadline.
    /// This neither clears the failure nor closes external/archive owners.
    pub(super) fn begin_failed_native_drain(
        &mut self,
        until: Instant,
    ) -> FailedNativeDrainObservation {
        if self.state != V6PipelineObservation::Failed {
            return FailedNativeDrainObservation::OwnerUnavailable;
        }
        let service = if let Some(owner) = self.search_owner.as_mut() {
            owner.custody.native.service.as_mut()
        } else {
            self.pre_admission
                .as_mut()
                .and_then(V6PreAdmissionOwner::failed_native_service)
        };
        let Some(service) = service else {
            return FailedNativeDrainObservation::OwnerUnavailable;
        };
        if self.failed_native_drain.is_none() {
            match V6FailedStartupDrain::new(service, until.min(self.original_cleanup_until)) {
                Ok(drain) => self.failed_native_drain = Some(drain),
                Err(observation) => return observation,
            }
        }
        self.failed_native_drain
            .as_mut()
            .expect("drain inserted")
            .begin(service)
    }

    /// One nonblocking event/join observation on the same retained service.
    pub(super) fn poll_failed_native_drain(&mut self) -> FailedNativeDrainObservation {
        if self.begin_before_native_recovery_if_ready() {
            return FailedNativeDrainObservation::BeforeNativeRecoveryStarted;
        }
        let service = if let Some(owner) = self.search_owner.as_mut() {
            owner.custody.native.service.as_mut()
        } else {
            self.pre_admission
                .as_mut()
                .and_then(V6PreAdmissionOwner::failed_native_service)
        };
        match (self.failed_native_drain.as_mut(), service) {
            (Some(drain), Some(service)) => drain.poll_once(service),
            _ => FailedNativeDrainObservation::OwnerUnavailable,
        }
    }
    fn begin_before_native_recovery_if_ready(&mut self) -> bool {
        if self.before_native_recovery.is_some() {
            return true;
        }
        let before_native = self
            .search_owner
            .as_ref()
            .and_then(|owner| owner.custody.native.service.as_ref())
            .is_some_and(|service| {
                service.startup_status() == NativeFactoryStartupStatus::BeforeNative
            });
        if before_native {
            if let Some(service) = self
                .search_owner
                .as_mut()
                .and_then(|owner| owner.custody.native.service.take())
            {
                self.before_native_recovery = Some(BeforeNativeRecoveryV6::new(
                    service,
                    self.original_cleanup_until,
                ));
                return true;
            }
        }
        false
    }

    fn fail(&mut self, failure: V6PipelineFailure) -> V6PipelineObservation {
        self.failure = Some(failure);
        self.state = V6PipelineObservation::Failed;
        // This only starts the same native worker's bounded drain when one
        // was actually retained. A caller must continue polling; no failure
        // or missing physical completion is converted to success here.
        if !self.begin_before_native_recovery_if_ready() {
            self.first_failed_native_drain =
                Some(self.begin_failed_native_drain(self.original_cleanup_until));
        }
        self.state
    }

    /// One initial attempt. No native execution occurs before the original
    /// parent authenticates and acknowledges the same post-image grant.
    pub(super) fn connect_and_start(&mut self) -> V6PipelineObservation {
        if self.state != V6PipelineObservation::Initial {
            return V6PipelineObservation::AlreadyAttempted;
        }
        let parent = match self
            .source
            .authenticate_direct_parent(&self.parent_socket, self.parent_pid)
        {
            Ok(parent) => parent,
            Err(error) => return self.fail(V6PipelineFailure::Parent(error)),
        };
        self.parent = Some(parent);
        let admission = match self.source.await_parent_image_grant(
            &self.parent_socket,
            self.parent_pid,
            self.parent.as_ref().expect("original parent"),
        ) {
            Ok(admission) => admission,
            Err(error) => return self.fail(V6PipelineFailure::ParentGrant(error)),
        };
        self.admission = Some(admission);
        let cpu = match prepare_cpu_v6(Arc::clone(&self.source), &self.clock) {
            Ok(cpu) => cpu,
            Err(error) => return self.fail(V6PipelineFailure::Bootstrap(error)),
        };
        self.cpu = Some(cpu);
        // Existing factory startup takes the CPU owner by value. If it
        // unwinds inside that function this stage remains explicitly unknown;
        // external supervision must not treat the process as safely closed.
        self.state = V6PipelineObservation::NativeStartInProgress;
        let result = start_owned_system_native_v6(
            self.cpu.take().expect("original CPU owner"),
            Arc::clone(self.parent.as_ref().expect("original parent")),
            Arc::clone(self.admission.as_ref().expect("original admission")),
            None,
        );
        match result {
            Ok(custody) => {
                self.search_owner = Some(V6SearchPreparationOwner::new(custody));
                self.state = V6PipelineObservation::WaitingNative;
                self.state
            }
            Err(rejection) => {
                self.startup_rejection = Some(rejection);
                self.fail(V6PipelineFailure::FactoryRejected)
            }
        }
    }

    /// Observe once without blocking. Pending can be polled by the existing
    /// supervisor within the same original startup deadline and cancellation.
    pub(super) fn poll_and_admit(&mut self) -> V6PipelineObservation {
        if self.state != V6PipelineObservation::WaitingNative {
            return V6PipelineObservation::AlreadyAttempted;
        }
        let (until, cancelled) = self.source.original_startup_control();
        if let Err(error) = require_original_startup_window(until, &cancelled) {
            return self.fail(V6PipelineFailure::Bootstrap(error));
        }
        let owner = self.search_owner.as_mut().expect("original native owner");
        owner.poll_original_startup();
        let Some(service) = owner.custody.native.service.as_ref() else {
            return self.fail(V6PipelineFailure::FactoryNotReady(
                NativeFactoryStartupStatus::Unobserved,
            ));
        };
        let status = service.startup_status();
        match status {
            NativeFactoryStartupStatus::Pending => return self.state,
            NativeFactoryStartupStatus::Ready => {}
            other => return self.fail(V6PipelineFailure::FactoryNotReady(other)),
        }
        match &owner.snapshot {
            OriginalSnapshotState::Pending => return self.state,
            OriginalSnapshotState::Disconnected => {
                return self.fail(V6PipelineFailure::SnapshotDisconnected);
            }
            OriginalSnapshotState::Ready(_) | OriginalSnapshotState::Failed(_) => {}
        }
        let declaration = match owner.prepare_search_declaration(self.identity.clone()) {
            Ok(value) => value,
            Err(error) => return self.fail(V6PipelineFailure::Bootstrap(error)),
        };
        let owner = self.search_owner.take().expect("original native owner");
        self.pre_admission = Some(V6PreAdmissionOwner::new(
            owner,
            declaration,
            Arc::clone(&self.rules_owners),
        ));
        let stage = self
            .pre_admission
            .as_mut()
            .expect("original pre-admission stage");
        let prepared = stage.prepare_once();
        if prepared != V6PreparationObservation::Prepared {
            return self.fail(V6PipelineFailure::Preparation(prepared));
        }
        let admitted = stage.admit_once();
        if admitted != V6AdmissionObservation::Admitted {
            return self.fail(V6PipelineFailure::Admission(admitted));
        }
        let stage = self
            .pre_admission
            .take()
            .expect("original pre-admission stage");
        match PalsSystemDriverLifecycleV6::new(stage) {
            Ok(lifecycle) => {
                self.lifecycle = Some(lifecycle);
                self.state = V6PipelineObservation::ReadyForService;
                self.state
            }
            Err(stage) => {
                self.pre_admission = Some(stage);
                self.fail(V6PipelineFailure::LifecycleAdmission)
            }
        }
    }

    /// Serve exactly once using the existing UCI lifecycle. The facade is not
    /// exposed elsewhere; only protocol/engine can invoke finish_session.
    pub(super) fn serve(&mut self) -> V6PipelineObservation {
        if self.state != V6PipelineObservation::ReadyForService {
            return V6PipelineObservation::AlreadyAttempted;
        }
        self.state = V6PipelineObservation::ServiceInProgress;
        let lifecycle = self.lifecycle.as_mut().expect("original lifecycle");
        let driver = lifecycle.driver_for_service();
        let result = catch_unwind(AssertUnwindSafe(|| {
            protocol::serve_driver(
                driver,
                self.clock.clone(),
                Arc::clone(&self.rules_owners),
                self.settings,
            )
        }));
        match result {
            Ok(result) => {
                // Called once while retaining the lifecycle, including errors.
                if lifecycle.record_service_outcome(result).is_err() {
                    return self.fail(V6PipelineFailure::LifecycleAdmission);
                }
                self.state = V6PipelineObservation::ServiceReturned;
                self.state
            }
            Err(payload) => {
                self.service_panic = Some(payload);
                self.fail(V6PipelineFailure::ServicePanic)
            }
        }
    }

    /// Only reports the original lifecycle observation; it does not attempt
    /// a second shutdown or treat a service return as a GPU completion fence.
    pub(super) fn finish_observation(&mut self, until: Instant) -> FinishObservationV6 {
        match self.lifecycle.as_mut() {
            Some(lifecycle) => lifecycle.finish_until(until),
            None => FinishObservationV6::ServiceNotReturned,
        }
    }
}
