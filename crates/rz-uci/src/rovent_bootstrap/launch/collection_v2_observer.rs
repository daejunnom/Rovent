//! V2 native Collection observer custody across startup and physical shutdown.
//!
//! The proxy executes callbacks on the original native owner. The separate
//! custody keeps the *same* observer alive when a factory closure is dropped,
//! panics, or physical completion is unknown. No root/manager lock may be held
//! while a callback or reclamation acquires this mutex.

use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicBool, Ordering};

use crate::pals_native::ticketed::{
    NativeTicketQuestion, TicketExecutionEvidence, TicketOwnerDispatchContext,
    TicketPreparation, VirtualReviewCaptureFailure, VirtualReviewCaptureFailureStage,
    VirtualReviewCollectionObserver, VirtualReviewPhysicalInvocationEvidence,
    VirtualReviewPreparedTicket,
};
use rz_contracts::pals::RoleOutput;
use rz_contracts::pals_adaptive::RoleTicketEnvelope;
use rz_eval::pals_onnx::VirtualReviewCudaPlacementWitness;
use rz_search::pals::coordinator::{PhysicalRoleCompletion, RoleFailure};
use rz_search::pals::review_context::ReviewView;

struct SharedObserver {
    observer: Mutex<Option<Box<dyn VirtualReviewCollectionObserver>>>,
    poison_observed: AtomicBool,
}

/// Original custody. Dropping an unrecovered owner deliberately preserves it:
/// dropping a proxy or a failed startup must not free a potentially live owner.
/// The launch owner must retain this value for normal recovery; the Drop
/// quarantine is a last-resort lifetime guard, not completion evidence.
pub(super) struct V2ObserverCustody {
    shared: Option<Arc<SharedObserver>>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum V2ObserverReclaimError {
    /// A proxy still exists. Join/retire the physical owner first.
    ProxyStillOwned,
    /// An observer callback panicked while holding the original mutex.
    Poisoned,
    NotPoisoned,
    AlreadyReclaimed,
}

impl V2ObserverCustody {
    pub(super) fn new(observer: Box<dyn VirtualReviewCollectionObserver>) -> Self {
        Self { shared: Some(Arc::new(SharedObserver {
            observer: Mutex::new(Some(observer)),
            poison_observed: AtomicBool::new(false),
        })) }
    }

    /// Create a proxy before moving it to the native startup factory. The
    /// custody must outlive the proxy and all physical work it might observe.
    pub(super) fn proxy(&self) -> Option<Box<dyn VirtualReviewCollectionObserver>> {
        self.shared.as_ref().map(|shared| Box::new(V2ObserverProxy {
            shared: Arc::clone(shared),
        }) as Box<dyn VirtualReviewCollectionObserver>)
    }

    pub(super) fn poison_observed(&self) -> bool {
        self.shared.as_ref().is_some_and(|shared| {
            shared.poison_observed.load(Ordering::Acquire) || shared.observer.is_poisoned()
        })
    }

    /// The caller must first establish either (1) that no worker was spawned
    /// or (2) actual physical completion, native join and any separate
    /// teacher/observer settlement required by the enclosing owner. Absence
    /// of proxies alone is deliberately not evidence of those conditions.
    /// Errors leave the original Box in custody, including after poisoning.
    pub(super) fn try_reclaim_after_verified_boundary(
        &mut self,
    ) -> Result<Box<dyn VirtualReviewCollectionObserver>, V2ObserverReclaimError> {
        let shared = self.shared.as_ref().ok_or(V2ObserverReclaimError::AlreadyReclaimed)?;
        if Arc::strong_count(shared) != 1 {
            return Err(V2ObserverReclaimError::ProxyStillOwned);
        }
        let mut guard = shared.observer.lock().map_err(|_| {
            shared.poison_observed.store(true, Ordering::Release);
            V2ObserverReclaimError::Poisoned
        })?;
        let observer = guard.take().ok_or(V2ObserverReclaimError::AlreadyReclaimed)?;
        drop(guard);
        self.shared = None;
        Ok(observer)
    }

    /// Explicit recovery from an observer callback panic. This never clears
    /// or suppresses the poison: returning `Poisoned` alongside the original
    /// owner makes the failure part of the caller's recovery record. The same
    /// verified physical boundary required by the ordinary method applies.
    pub(super) fn reclaim_poisoned_after_verified_boundary(
        &mut self,
    ) -> Result<(Box<dyn VirtualReviewCollectionObserver>, V2ObserverReclaimError), V2ObserverReclaimError> {
        let shared = self.shared.as_ref().ok_or(V2ObserverReclaimError::AlreadyReclaimed)?;
        if Arc::strong_count(shared) != 1 {
            return Err(V2ObserverReclaimError::ProxyStillOwned);
        }
        let mut guard = match shared.observer.lock() {
            Err(poisoned) => poisoned.into_inner(),
            Ok(_) => return Err(V2ObserverReclaimError::NotPoisoned),
        };
        shared.poison_observed.store(true, Ordering::Release);
        let observer = guard.take().ok_or(V2ObserverReclaimError::AlreadyReclaimed)?;
        drop(guard);
        self.shared = None;
        Ok((observer, V2ObserverReclaimError::Poisoned))
    }
}

impl Drop for V2ObserverCustody {
    fn drop(&mut self) {
        if let Some(shared) = self.shared.take() {
            std::mem::forget(shared);
        }
    }
}

struct V2ObserverProxy {
    shared: Arc<SharedObserver>,
}

impl V2ObserverProxy {
    fn with_observer<T>(
        &self,
        f: impl FnOnce(&mut dyn VirtualReviewCollectionObserver) -> T,
    ) -> Result<T, RoleFailure> {
        let mut guard = self.shared.observer.lock().map_err(|_| {
            self.shared.poison_observed.store(true, Ordering::Release);
            RoleFailure::Backend("V2 collection observer custody poisoned".into())
        })?;
        let observer = guard.as_deref_mut().ok_or_else(|| {
            RoleFailure::Backend("V2 collection observer already reclaimed".into())
        })?;
        Ok(f(observer))
    }
}

impl VirtualReviewCollectionObserver for V2ObserverProxy {
    fn prepared(&mut self, question: &TicketPreparation<ReviewView, NativeTicketQuestion>, prepared: &VirtualReviewPreparedTicket) -> Result<(), RoleFailure> {
        self.with_observer(|observer| observer.prepared(question, prepared))?
    }
    fn submitted(&mut self, ticket: &RoleTicketEnvelope, question: &TicketPreparation<ReviewView, NativeTicketQuestion>, prepared: &VirtualReviewPreparedTicket) -> Result<(), RoleFailure> {
        self.with_observer(|observer| observer.submitted(ticket, question, prepared))?
    }
    fn submitted_classified(&mut self, context: &TicketOwnerDispatchContext<'_, ReviewView, NativeTicketQuestion, VirtualReviewPreparedTicket>) -> Result<(), VirtualReviewCaptureFailure> {
        self.with_observer(|observer| observer.submitted_classified(context))
            .map_err(VirtualReviewCaptureFailure::Required)?
    }
    fn preserve_observation_failure(&mut self, stage: VirtualReviewCaptureFailureStage, ticket: &RoleTicketEnvelope, failure: &RoleFailure) -> bool {
        // This trait method cannot return an error. Refuse isolation on poison;
        // the flag exposes the failure and the original observer stays held.
        self.with_observer(|observer| observer.preserve_observation_failure(stage, ticket, failure)).unwrap_or(false)
    }
    fn completed(&mut self, ticket: &RoleTicketEnvelope, physical: PhysicalRoleCompletion, output: Option<&RoleOutput>, evidence: Option<TicketExecutionEvidence>, invocation: Option<VirtualReviewPhysicalInvocationEvidence>) -> Result<(), RoleFailure> {
        self.with_observer(|observer| observer.completed(ticket, physical, output, evidence, invocation))?
    }
    fn completed_classified(&mut self, ticket: &RoleTicketEnvelope, physical: PhysicalRoleCompletion, output: Option<&RoleOutput>, evidence: Option<TicketExecutionEvidence>, invocation: Option<VirtualReviewPhysicalInvocationEvidence>) -> Result<(), VirtualReviewCaptureFailure> {
        self.with_observer(|observer| observer.completed_classified(ticket, physical, output, evidence, invocation))
            .map_err(VirtualReviewCaptureFailure::Required)?
    }
    fn retired(&mut self, ticket: &RoleTicketEnvelope, accepted: bool) -> Result<(), RoleFailure> {
        self.with_observer(|observer| observer.retired(ticket, accepted))?
    }
    fn retired_classified(&mut self, ticket: &RoleTicketEnvelope, accepted: bool) -> Result<(), VirtualReviewCaptureFailure> {
        self.with_observer(|observer| observer.retired_classified(ticket, accepted))
            .map_err(VirtualReviewCaptureFailure::Required)?
    }
    fn runtime_audited(&mut self, witness: &VirtualReviewCudaPlacementWitness) -> Result<(), RoleFailure> {
        self.with_observer(|observer| observer.runtime_audited(witness))?
    }
}
