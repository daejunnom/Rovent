//! Original V6 driver custody after successful session assembly.
//!
//! This type does not spawn, serve, signal or terminate a process. Its caller
//! is responsible for keeping this value and the original process alive on an
//! unknown outcome. In particular, `#[must_use]` cannot enforce that duty.

use super::driver_assemble::V6PreAdmissionOwner;
use super::driver_facade::{V6DriverFacade, V6FacadeFinishObservation};
use super::driver_prepare::{OriginalReceiptState, OriginalSnapshotState};
use super::resource_observation::StartupResourceObservationV6;
use crate::{
    engine::EngineError,
    pals_adaptive::AdaptiveSessionProcessReceipt,
    search_driver::{SearchSessionDriver, SearchSessionFailure},
};
use std::{
    sync::{Arc, mpsc},
    time::Instant,
};

/// A result of a single finish request, not permission to exit the process.
/// An error remains attached to the original owner for later examination.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum FinishObservationV6 {
    ServiceNotReturned,
    DeadlineExpired,
    /// Driver-owned process scopes closed; UCI service and external process
    /// outcomes remain separate and the GPU device as a whole is not fenced.
    OwnerScopesClosed,
    Incomplete,
    /// The payload and original driver remain retained. Completion is unknown.
    PanicRetained,
}

/// The same admitted driver and startup authorities remain together through
/// normal service and physical shutdown. This scope cannot validate the
/// parent/child, cgroup, device-wide state or output lease by itself.
#[must_use = "retain the original V6 process and custody on unknown completion"]
pub(super) struct PalsSystemDriverLifecycleV6 {
    driver: Arc<V6DriverFacade>,
    original: V6PreAdmissionOwner,
    served: Option<Result<(), EngineError>>,
    first_finish_failure: Option<SearchSessionFailure>,
    incomplete_receipt: Option<AdaptiveSessionProcessReceipt>,
    closed_receipt: Option<AdaptiveSessionProcessReceipt>,
    startup_resources: Option<StartupResourceObservationV6>,
    resource_observation_panic: Option<Box<dyn std::any::Any + Send>>,
    resource_publication: Option<Result<(), String>>,
    resource_publication_panic: Option<Box<dyn std::any::Any + Send>>,
}

impl PalsSystemDriverLifecycleV6 {
    /// Consume the assembly result as one unit. This preserves its original
    /// source/CPU intent, parent, admission, external capability, archive and
    /// both observation channels; no caller can recombine unrelated owners.
    pub(super) fn new(mut original: V6PreAdmissionOwner) -> Result<Self, V6PreAdmissionOwner> {
        if original.transferred.is_none() || original.admitted_driver.is_none() {
            return Err(original);
        }
        let driver = original
            .admitted_driver
            .take()
            .expect("checked admitted driver");
        Ok(Self {
            driver: Arc::new(V6DriverFacade::new(driver)),
            original,
            served: None,
            first_finish_failure: None,
            incomplete_receipt: None,
            closed_receipt: None,
            startup_resources: None,
            resource_observation_panic: None,
            resource_publication: None,
            resource_publication_panic: None,
        })
    }

    /// Preserve the first actual UCI outcome, including its original error.
    /// On a duplicate call return the new outcome to its caller untouched.
    pub(super) fn record_service_outcome(
        &mut self,
        result: Result<(), EngineError>,
    ) -> Result<(), Result<(), EngineError>> {
        if self.served.is_some() {
            return Err(result);
        }
        self.served = Some(result);
        Ok(())
    }

    pub(super) fn service_outcome(&self) -> Option<&Result<(), EngineError>> {
        self.served.as_ref()
    }

    pub(super) fn service_succeeded(&self) -> Option<bool> {
        self.served.as_ref().map(Result::is_ok)
    }

    pub(super) fn start_checker(
        &self,
        until: Instant,
        cancelled: Arc<std::sync::atomic::AtomicBool>,
    ) -> Result<Option<bool>, SearchSessionFailure> {
        self.driver.start_checker(until, cancelled)
    }

    /// Diagnostic readback is independent of checker and NN success. The
    /// original driver and source remain held even if the observer panics.
    pub(super) fn observe_startup_resources(&mut self, until: Instant) {
        if self.startup_resources.is_some() || self.resource_observation_panic.is_some() {
            return;
        }
        let Some(original) = self.original.transferred.as_ref() else {
            return;
        };
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.driver.observe_startup_resources(original, until)
        })) {
            Ok(observation) => self.startup_resources = Some(observation),
            Err(payload) => self.resource_observation_panic = Some(payload),
        }
    }

    /// Publish only through the exact factory's original receipt owner. A
    /// pending/missing writer or failed write is a separate diagnostic and
    /// never changes the result of model execution.
    pub(super) fn publish_startup_resources(&mut self) {
        if self.resource_publication.is_some()
            || self.resource_publication_panic.is_some()
            || (self.startup_resources.is_none() && self.resource_observation_panic.is_none())
        {
            return;
        }
        self.poll_original_startup_receipt();
        let Some(original) = self.original.transferred.as_ref() else {
            return;
        };
        let publication = match &original.original_receipt {
            OriginalReceiptState::Pending => return,
            OriginalReceiptState::Disconnected => {
                self.resource_publication =
                    Some(Err("original receipt publisher disconnected".into()));
                return;
            }
            OriginalReceiptState::Observed(publication) => publication,
        };
        let result =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<(), String> {
                let owner = publication
                    .owner
                    .as_ref()
                    .ok_or_else(|| "original receipt owner unavailable".to_owned())?;
                if !Arc::ptr_eq(&owner.owner, &original.cpu.original_source)
                    || !Arc::ptr_eq(&owner.parent, &original.parent)
                {
                    return Err("resource receipt owner differs from original lifecycle".into());
                }
                owner
                    .parent
                    .ensure_live()
                    .map_err(|error| error.to_string())?;
                let observation = match &self.startup_resources {
                    Some(value) => value.to_receipt_json(),
                    None => serde_json::json!({
                        "domain": "rz-pals-system-v6-startup-resource-observation/1",
                        "observation": "observer_panicked",
                        "gpu_physical_completion": "unknown",
                        "numeric_acceptance": false,
                        "performance_acceptance": false,
                    }),
                };
                let bytes = serde_json::to_vec(&observation).map_err(|error| error.to_string())?;
                owner
                    .writer
                    .write_bytes("pals-startup-resources.v6.json", &bytes)
            }));
        match result {
            Ok(value) => self.resource_publication = Some(value),
            Err(payload) => self.resource_publication_panic = Some(payload),
        }
    }

    pub(super) fn startup_resources(&self) -> Option<&StartupResourceObservationV6> {
        self.startup_resources.as_ref()
    }

    pub(super) fn resource_publication(&self) -> Option<&Result<(), String>> {
        self.resource_publication.as_ref()
    }

    /// Only the pipeline's pre-service state may call this. An arbitrary
    /// service error is not proof that no UCI worker was created.
    pub(super) fn finish_before_service(&mut self, until: Instant) -> FinishObservationV6 {
        if self.served.is_some() {
            return FinishObservationV6::Incomplete;
        }
        match self.driver.finish_before_service(until) {
            Ok(receipt) if receipt.permits_process_close() => {
                self.closed_receipt = Some(receipt);
                FinishObservationV6::OwnerScopesClosed
            }
            Ok(receipt) => {
                self.incomplete_receipt = Some(receipt);
                FinishObservationV6::Incomplete
            }
            Err(error) => {
                self.first_finish_failure = Some(error);
                if matches!(
                    self.driver.finish_observation(),
                    V6FacadeFinishObservation::Panicked
                ) {
                    FinishObservationV6::PanicRetained
                } else {
                    FinishObservationV6::Incomplete
                }
            }
        }
    }

    /// Give the existing UCI protocol the exact assembled driver. The caller
    /// remains responsible for ensuring that no clone independently initiates
    /// process finish while this facade is doing so.
    pub(super) fn driver_for_service(&self) -> Arc<dyn SearchSessionDriver> {
        self.driver.clone()
    }

    pub(super) fn original_snapshot(&self) -> &OriginalSnapshotState {
        &self
            .original
            .transferred
            .as_ref()
            .expect("retained transferred owner")
            .original_snapshot
    }

    /// Diagnostic receipt absence or disconnection is independent of model
    /// execution and physical completion. Preserve an observed receipt in
    /// this same owner and never make a missing receipt a functional failure.
    pub(super) fn poll_original_startup_receipt(&mut self) -> &OriginalReceiptState {
        let original = self
            .original
            .transferred
            .as_mut()
            .expect("retained transferred owner");
        if matches!(original.original_receipt, OriginalReceiptState::Pending) {
            original.original_receipt = match original.receipt_receiver.try_recv() {
                Ok(receipt) => OriginalReceiptState::Observed(receipt),
                Err(mpsc::TryRecvError::Empty) => OriginalReceiptState::Pending,
                Err(mpsc::TryRecvError::Disconnected) => OriginalReceiptState::Disconnected,
            };
        }
        &original.original_receipt
    }

    pub(super) fn first_finish_failure(&self) -> Option<&SearchSessionFailure> {
        self.first_finish_failure.as_ref()
    }

    pub(super) fn closed_receipt(&self) -> Option<&AdaptiveSessionProcessReceipt> {
        self.closed_receipt.as_ref()
    }

    pub(super) fn incomplete_receipt(&self) -> Option<&AdaptiveSessionProcessReceipt> {
        self.incomplete_receipt.as_ref()
    }

    /// Request bounded finish from the same driver. `&mut self` serializes
    /// calls through this facade; callers must also avoid a separate direct
    /// `finish_process` call through another clone of the driver. The driver
    /// itself retains original state if its internal lock is busy or finish is
    /// incomplete. No retry changes the original search/request deadline.
    pub(super) fn finish_until(&mut self, until: Instant) -> FinishObservationV6 {
        if self.closed_receipt.is_some() {
            return FinishObservationV6::OwnerScopesClosed;
        }
        if matches!(
            self.driver.finish_observation(),
            V6FacadeFinishObservation::Panicked
        ) {
            return FinishObservationV6::PanicRetained;
        }
        // A partial failure may have consumed a worker join handle. Its
        // absence on a later call cannot prove that the original join worked.
        if self.first_finish_failure.is_some() || self.incomplete_receipt.is_some() {
            return FinishObservationV6::Incomplete;
        }
        if self.served.is_none() {
            return FinishObservationV6::ServiceNotReturned;
        }
        // The protocol owns the only service handle; its finish_session call
        // is issued after the original UCI worker drain. A returned service
        // error alone is not evidence that this happened.
        match self.driver.finish_observation() {
            V6FacadeFinishObservation::Closed(receipt) if receipt.permits_process_close() => {
                self.closed_receipt = Some(receipt);
                FinishObservationV6::OwnerScopesClosed
            }
            // Although the current driver rejects this itself, never convert
            // a future incomplete receipt into a physical close here.
            V6FacadeFinishObservation::Closed(receipt)
            | V6FacadeFinishObservation::Incomplete(receipt) => {
                if self.incomplete_receipt.is_none() {
                    self.incomplete_receipt = Some(receipt);
                }
                FinishObservationV6::Incomplete
            }
            V6FacadeFinishObservation::Failed(error) => {
                if self.first_finish_failure.is_none() {
                    self.first_finish_failure = Some(error);
                }
                FinishObservationV6::Incomplete
            }
            V6FacadeFinishObservation::Panicked => FinishObservationV6::PanicRetained,
            V6FacadeFinishObservation::Pending if Instant::now() >= until => {
                FinishObservationV6::DeadlineExpired
            }
            V6FacadeFinishObservation::Pending | V6FacadeFinishObservation::BusyOrPoisoned => {
                FinishObservationV6::Incomplete
            }
        }
    }
}
