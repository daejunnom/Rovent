//! Staged child entry using the original V6 source and pipeline.
//!
//! This is not the CLI entry yet. The outer supervisor must supply its own
//! original cleanup deadline and retain this object and process on unknown
//! completion. A successful release ACK is not a device-wide completion fence.
use super::super::*;
use super::before_native_recovery::BeforeNativeRecoveryObservation;
use super::driver_lifecycle::FinishObservationV6;
use super::failed_startup_drain::FailedNativeDrainObservation;
use super::pipeline::{V6Pipeline, V6PipelineObservation};
use crate::pals_adaptive_startup::{PalsSystemStartupOptionsV6, PinnedPalsSystemStartupV6};
use crate::rovent_bootstrap::launch::ConnectedParent;
use std::any::Any;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::Path;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum ChildStepV6 {
    Pipeline(V6PipelineObservation),
    /// The pipeline and panic payload remain, but an in-flight ownership
    /// transfer may have unwound and dropped its local CPU owner. Native
    /// worker ownership and physical completion are therefore unknown.
    TransitionPanickedCustodyUnknown,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum ChildCleanupV6 {
    Lifecycle(FinishObservationV6),
    FailedNative(FailedNativeDrainObservation),
    BeforeNative(BeforeNativeRecoveryObservation),
    /// No original cleanup authority is available, or a transition panicked.
    OwnerUnresolved,
}

/// Never discard this owner or terminate its process merely because a UCI
/// request returned, a cleanup deadline expired, or the parent disconnected.
#[must_use = "retain the original child owner and process on unknown completion"]
pub(super) struct ChildEntryV6 {
    pipeline: V6Pipeline,
    original_cleanup_until: Instant,
    cleanup_cancelled: AtomicBool,
    panic: Option<Box<dyn Any + Send>>,
}

impl ChildEntryV6 {
    /// Capture the direct parent before opening/publishing its endpoint. The
    /// deadline must be the one received from the original outer supervisor;
    /// this constructor does not invent a new execution or cleanup window.
    pub(super) fn pin(
        options: &PalsSystemStartupOptionsV6,
        cwd: &Path,
        started: Instant,
        original_cleanup_until: Instant,
        startup_cancelled: Arc<AtomicBool>,
    ) -> Result<Self, AdaptiveBootstrapError> {
        if options.preflight {
            return Err(boundary("SystemV6ChildRequiresNative"));
        }
        let parent_pid = ConnectedParent::capture_original_parent()
            .map_err(|error| failure("SystemV6OriginalParent", &error))?;
        let parent_socket = options
            .parent_socket
            .clone()
            .ok_or_else(|| boundary("SystemV6ParentSocketRequired"))?;
        let startup_until = started
            .checked_add(Duration::from_millis(options.startup_ms))
            .ok_or_else(|| boundary("SystemV6StartupDeadline"))?;
        let source = Arc::new(
            PinnedPalsSystemStartupV6::open(
                &options.lock_path,
                &options.lock_sha256,
                options.startup_ms,
                cwd,
                started,
                startup_until,
                startup_cancelled,
            )
            .map_err(|error| failure("SystemV6PinnedStartup", &error))?,
        );
        let clock = ProcessClock::new(ProcessEpoch(source.lock().config.process_generation));
        let identity = EngineIdentity {
            name: format!(
                "RoveZero PALS adaptive {}",
                source.lock().config.endpoint.id
            ),
            author: "RoveZero contributors".into(),
        };
        let mut settings = EngineSettings {
            max_workers: 1,
            shutdown_limit: Duration::from_millis(source.lock().config.native_options.drain_ms),
            ..EngineSettings::default()
        };
        settings.search.untimed_limit =
            Duration::from_millis(source.lock().config.window.wall_time_max_ms);
        let pipeline = V6Pipeline::new(
            source,
            clock,
            parent_socket,
            parent_pid,
            identity,
            Arc::new(OwnerRegistry::default()),
            settings,
            original_cleanup_until,
        );
        let original_cleanup_until = pipeline.original_cleanup_until();
        Ok(Self {
            pipeline,
            original_cleanup_until,
            cleanup_cancelled: AtomicBool::new(false),
            panic: None,
        })
    }

    /// Advance one real stage. Waiting for native readiness is a separate
    /// observation, and UCI service runs once on the same admitted driver.
    pub(super) fn step(&mut self) -> ChildStepV6 {
        if self.panic.is_some() {
            return ChildStepV6::TransitionPanickedCustodyUnknown;
        }
        let result = catch_unwind(AssertUnwindSafe(|| match self.pipeline.observation() {
            V6PipelineObservation::Initial => self.pipeline.connect_and_start(),
            V6PipelineObservation::WaitingNative => self.pipeline.poll_and_admit(),
            V6PipelineObservation::ReadyForService => self.pipeline.serve(),
            other => other,
        }));
        match result {
            Ok(observation) => ChildStepV6::Pipeline(observation),
            Err(payload) => {
                self.panic = Some(payload);
                ChildStepV6::TransitionPanickedCustodyUnknown
            }
        }
    }

    /// One bounded call against the retained pipeline and original deadline.
    /// The caller schedules subsequent observations; this does not spin.
    pub(super) fn cleanup_once(&mut self) -> ChildCleanupV6 {
        if self.panic.is_some() {
            return ChildCleanupV6::OwnerUnresolved;
        }
        let result = catch_unwind(AssertUnwindSafe(|| self.observe_cleanup_once()));
        match result {
            Ok(observation) => observation,
            Err(payload) => {
                self.panic = Some(payload);
                ChildCleanupV6::OwnerUnresolved
            }
        }
    }

    fn observe_cleanup_once(&mut self) -> ChildCleanupV6 {
        if self.pipeline.lifecycle().is_some() {
            return ChildCleanupV6::Lifecycle(
                self.pipeline
                    .finish_observation(self.original_cleanup_until, &self.cleanup_cancelled),
            );
        }
        if self.pipeline.observation() != V6PipelineObservation::Failed {
            return ChildCleanupV6::OwnerUnresolved;
        }
        if let Some(observation) = self.pipeline.poll_before_native_recovery() {
            return ChildCleanupV6::BeforeNative(observation);
        }
        ChildCleanupV6::FailedNative(self.pipeline.poll_failed_native_drain())
    }

    /// The caller must inspect the original failure, service result, release
    /// acknowledgement and physical scope independently before any exit.
    pub(super) fn pipeline(&self) -> &V6Pipeline {
        &self.pipeline
    }
}
