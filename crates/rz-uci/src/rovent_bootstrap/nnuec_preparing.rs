//! One original initial NNUEC preparation, using the session's same actor.
//!
//! The parent session completes its actual root Rules scan, source control and
//! live admission gate before moving the real root and actor here. Work roots
//! zero/one and initially expired/canceled roots never enter this helper, and
//! their parent does not spawn a preparation actor for this path. This helper
//! never creates an actor, substitute engine, researcher, clock or deadline.
//!
//! One `step` crosses one bounded source/actor/worker boundary. There is no
//! waiting loop, sleep, blocking shutdown or replacement of an original owner.
//! Busy/None returns the exact state. Errors return every retained owner for
//! the parent's cleanup under its original cleanup deadline. Neither a missing
//! Option slot nor returning this transport proves physical completion.

use crate::pals_adaptive::AdaptiveCheckerMetadata;
use rz_contracts::pals_adaptive::CpuWorkerId;
use rz_runtime::CompletionSignal;
use rz_search::cpu_checker::{CheckerError, CpuChecker};
use rz_search::cpu_value::nnuec::v2::review::{
    BoundNnuecReviewContext, Completion, NnuecPreparationTerminalAck, PreparationActor,
    PreparationError, Provenance, Request,
};
use rz_search::cpu_worker::{
    CpuWorker, CpuWorkerError, MaterializationCompletion, MaterializationHandoff,
    MaterializationRequest, MaterializedNnuecEngine, NnuecReadyRegistration, StartingNnuecWorker,
};
use rz_search::pals::engine::PalsError;
use rz_search::pals::engine::adaptive_port::{
    AdaptiveEnginePort, AdaptiveNnuecPreparationError, AdaptivePortModel,
    AdaptivePreparingRootCustody, PreparingRootViewSeal,
};
use rz_search::pals::engine::preparing_root::{
    PreparingRootFinishRejection, PreparingRootHandoff, ResearcherPreparingRoot,
};
use std::sync::Arc;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum InitialNnuecPhase {
    PrepareReview,
    SubmitReview,
    AwaitReview,
    PrepareMaterialization,
    SubmitMaterialization,
    AwaitMaterialization,
    ClaimMaterialization,
    CaptureMetadata,
    SpawnWorker,
    AwaitReady,
    AcceptReady,
    FinishRoot,
    AdoptPort,
}

#[derive(Debug)]
pub enum InitialNnuecFailure {
    Root(AdaptiveNnuecPreparationError),
    Preparation(PreparationError),
    Worker(CpuWorkerError),
    Metadata(CheckerError),
    Finish(PalsError),
    Invalid(&'static str),
}

/// Actual owner transport for the parent's bounded drain/retirement path.
/// Pending actor work can own the request/engine while its local slot is None.
/// Keep the same actor until unclaimed materialization and cache custody settle.
/// An accepted worker's joined engine must follow its same-actor retirement
/// path before closing that actor. This transport does not perform that work.
#[must_use = "actual owner slots and outstanding work require explicit cleanup"]
pub struct InitialNnuecCleanupOwners {
    pub root: Option<ResearcherPreparingRoot<AdaptivePortModel>>,
    pub actor: PreparationActor,
    pub original_request: Option<Request>,
    pub original_seal: Option<PreparingRootViewSeal>,
    pub expected: Option<Provenance>,
    pub bound: Option<Arc<BoundNnuecReviewContext>>,
    pub materialization_request: Option<MaterializationRequest>,
    pub materialization_handoff: Option<MaterializationHandoff>,
    pub materialized_engine: Option<MaterializedNnuecEngine>,
    pub starting_worker: Option<StartingNnuecWorker>,
    pub worker: Option<CpuWorker>,
    pub ready: Option<NnuecReadyRegistration>,
    pub metadata: Option<AdaptiveCheckerMetadata>,
    pub checker_mirror: Option<Box<dyn CpuChecker>>,
    pub root_handoff: Option<PreparingRootHandoff<AdaptivePortModel>>,
    pub unmatched_review: Option<Completion>,
    /// Actual same-actor C terminal; keep its one-use retirement capability
    /// even when the original result is a cancellation or preparation error.
    pub preparation_terminal_ack: Option<NnuecPreparationTerminalAck>,
    pub unmatched_materialization: Option<MaterializationCompletion>,
    /// A missing local Request is not a completion observation.
    pub preparation_outstanding: bool,
    /// Success terminal keeps the actor seat until handoff.claim succeeds.
    /// Claim failure leaves the unique engine in this same actor's custody.
    pub materialization_outstanding: bool,
}

/// Opaque cleanup custody, not an assertion of a clean root or joined worker.
#[must_use = "retain original actor/root/worker custody through actual cleanup"]
pub struct InitialNnuecCleanup {
    phase: InitialNnuecPhase,
    failure: Option<InitialNnuecFailure>,
    owners: Box<InitialNnuecCleanupOwners>,
}
impl InitialNnuecCleanup {
    pub fn phase(&self) -> InitialNnuecPhase {
        self.phase
    }
    pub fn failure(&self) -> Option<&InitialNnuecFailure> {
        self.failure.as_ref()
    }
    /// Transfer real owner objects to the parent's existing cancel/drain/join
    /// and actor-side retirement implementation. No destructor runs them here.
    pub fn into_parts(
        self,
    ) -> (
        InitialNnuecPhase,
        Option<InitialNnuecFailure>,
        InitialNnuecCleanupOwners,
    ) {
        (self.phase, self.failure, *self.owners)
    }
}

#[must_use = "an initial failure still owns actual cleanup obligations"]
pub struct InitialNnuecRetainedFailure {
    phase: InitialNnuecPhase,
    error: InitialNnuecFailure,
    owners: Box<InitialNnuecCleanupOwners>,
}
impl InitialNnuecRetainedFailure {
    pub fn phase(&self) -> InitialNnuecPhase {
        self.phase
    }
    pub fn error(&self) -> &InitialNnuecFailure {
        &self.error
    }
    pub fn into_cleanup(self) -> InitialNnuecCleanup {
        InitialNnuecCleanup {
            phase: self.phase,
            failure: Some(self.error),
            owners: self.owners,
        }
    }
}

/// Actual Ready + same-Store port. No native job or R analysis ran here.
#[must_use = "transfer the actual ready worker and same actor to session custody"]
pub struct InitialNnuecReady {
    pub port: AdaptiveEnginePort<AdaptivePortModel>,
    pub custody: AdaptivePreparingRootCustody,
    pub worker: CpuWorker,
    pub actor: PreparationActor,
    pub metadata: AdaptiveCheckerMetadata,
}

#[must_use = "pending/failure variants retain real owners and cannot be discarded"]
pub enum InitialNnuecStep {
    Pending(InitialNnuecPreparing),
    Ready(Box<InitialNnuecReady>),
    RetainedFailure(InitialNnuecRetainedFailure),
}

#[must_use = "retain this move-only preparation or explicitly transfer cleanup custody"]
pub struct InitialNnuecPreparing {
    phase: InitialNnuecPhase,
    // One physical custody allocation survives every Pending/failure/cleanup
    // transport. Empty Option slots still have the largest owner's inline
    // layout, so moving the whole bundle at every step is not stack-bounded.
    owners: Box<InitialNnuecCleanupOwners>,
    worker_owner: CpuWorkerId,
    signal: CompletionSignal,
}
impl InitialNnuecPreparing {
    /// Moves only the actual root and session's same preparation actor. The
    /// parent already owns the original control/cleanup deadline and live gate.
    pub fn new(
        root: ResearcherPreparingRoot<AdaptivePortModel>,
        actor: PreparationActor,
        owner: CpuWorkerId,
        signal: CompletionSignal,
    ) -> Self {
        Self {
            phase: InitialNnuecPhase::PrepareReview,
            owners: Box::new(InitialNnuecCleanupOwners {
                root: Some(root),
                actor,
                original_request: None,
                original_seal: None,
                expected: None,
                bound: None,
                materialization_request: None,
                materialization_handoff: None,
                materialized_engine: None,
                starting_worker: None,
                worker: None,
                ready: None,
                metadata: None,
                checker_mirror: None,
                root_handoff: None,
                unmatched_review: None,
                preparation_terminal_ack: None,
                unmatched_materialization: None,
                preparation_outstanding: false,
                materialization_outstanding: false,
            }),
            worker_owner: owner,
            signal,
        }
    }
    pub fn phase(&self) -> InitialNnuecPhase {
        self.phase
    }
    pub fn into_cleanup(self) -> InitialNnuecCleanup {
        InitialNnuecCleanup {
            phase: self.phase,
            failure: None,
            owners: self.owners,
        }
    }

    fn fail(self, error: InitialNnuecFailure) -> InitialNnuecStep {
        InitialNnuecStep::RetainedFailure(InitialNnuecRetainedFailure {
            phase: self.phase,
            error,
            owners: self.owners,
        })
    }
    fn invalid(self, detail: &'static str) -> InitialNnuecStep {
        self.fail(InitialNnuecFailure::Invalid(detail))
    }

    /// One non-waiting boundary per call; failures retain unique owners rather
    /// than using `?` on a consuming operation or replacing a failed owner.
    pub fn step(mut self) -> InitialNnuecStep {
        match self.phase {
            InitialNnuecPhase::PrepareReview => {
                let Some(root) = self.owners.root.as_mut() else {
                    return self.invalid("original preparing root absent");
                };
                match root.prepare_initial_nnuec_review() {
                    Ok(original) => {
                        let (request, seal) = original.into_parts();
                        self.owners.expected = Some(request.provenance().clone());
                        self.owners.original_request = Some(request);
                        self.owners.original_seal = Some(seal);
                        self.phase = InitialNnuecPhase::SubmitReview;
                    }
                    Err(error) => return self.fail(InitialNnuecFailure::Root(error)),
                }
            }
            InitialNnuecPhase::SubmitReview => {
                let Some(request) = self.owners.original_request.take() else {
                    return self.invalid("original initial C request absent");
                };
                match self.owners.actor.try_submit(request) {
                    Ok(()) => {
                        self.owners.preparation_outstanding = true;
                        self.phase = InitialNnuecPhase::AwaitReview;
                    }
                    Err((PreparationError::Busy, request)) => {
                        self.owners.original_request = Some(request);
                    }
                    Err((error, request)) => {
                        self.owners.original_request = Some(request);
                        return self.fail(InitialNnuecFailure::Preparation(error));
                    }
                }
            }
            InitialNnuecPhase::AwaitReview => match self.owners.actor.try_complete() {
                Ok(None) => {}
                Err(error) => return self.fail(InitialNnuecFailure::Preparation(error)),
                Ok(Some(completion)) => {
                    if self.owners.expected.as_ref() != Some(&completion.provenance) {
                        self.owners.unmatched_review = Some(completion);
                        return self.invalid("initial C completion differs from original request");
                    }
                    self.owners.preparation_outstanding = false;
                    self.owners.preparation_terminal_ack = Some(completion.terminal_ack);
                    match completion.result {
                        Ok(bound) => {
                            let matches = self.owners.expected.as_ref() == Some(bound.provenance());
                            self.owners.bound = Some(Arc::new(bound));
                            if !matches {
                                return self
                                    .invalid("initial C bound differs from original request");
                            }
                            self.phase = InitialNnuecPhase::PrepareMaterialization;
                        }
                        Err(error) => return self.fail(InitialNnuecFailure::Preparation(error)),
                    }
                }
            },
            InitialNnuecPhase::PrepareMaterialization => {
                let (Some(root), Some(seal), Some(bound)) = (
                    self.owners.root.as_ref(),
                    self.owners.original_seal.as_ref(),
                    self.owners.bound.as_ref(),
                ) else {
                    return self.invalid("original root/seal/bound materialization custody absent");
                };
                // The original Arc stays in this owner even when validation
                // rejects the clone supplied to the root's actual API.
                match root.materialize_initial_nnuec_review(seal, Arc::clone(bound)) {
                    Ok(request) => {
                        self.owners.materialization_request = Some(request);
                        self.phase = InitialNnuecPhase::SubmitMaterialization;
                    }
                    Err(error) => return self.fail(InitialNnuecFailure::Root(error)),
                }
            }
            InitialNnuecPhase::SubmitMaterialization => {
                let Some(request) = self.owners.materialization_request.take() else {
                    return self.invalid("original materialization request absent");
                };
                match self.owners.actor.try_materialize(request) {
                    Ok(()) => {
                        self.owners.materialization_outstanding = true;
                        self.phase = InitialNnuecPhase::AwaitMaterialization;
                    }
                    Err((PreparationError::Busy, request)) => {
                        self.owners.materialization_request = Some(request);
                    }
                    Err((error, request)) => {
                        self.owners.materialization_request = Some(request);
                        return self.fail(InitialNnuecFailure::Preparation(error));
                    }
                }
            }
            InitialNnuecPhase::AwaitMaterialization => match self.owners.actor.try_materialized() {
                Ok(None) => {}
                Err(error) => return self.fail(InitialNnuecFailure::Preparation(error)),
                Ok(Some(completion)) => {
                    if self.owners.expected.as_ref() != Some(&completion.provenance) {
                        self.owners.unmatched_materialization = Some(completion);
                        return self.invalid("materialization differs from original initial C");
                    }
                    match completion.result {
                        Ok(handoff) => {
                            self.owners.materialization_handoff = Some(handoff);
                            self.phase = InitialNnuecPhase::ClaimMaterialization;
                        }
                        Err((request, error)) => {
                            self.owners.materialization_outstanding = false;
                            self.owners.materialization_request = Some(request);
                            return self.fail(InitialNnuecFailure::Worker(error));
                        }
                    }
                }
            },
            InitialNnuecPhase::ClaimMaterialization => {
                let Some(handoff) = self.owners.materialization_handoff.take() else {
                    return self.invalid("original materialization handoff absent");
                };
                match handoff.claim() {
                    Ok(engine) => {
                        self.owners.materialization_outstanding = false;
                        self.owners.materialized_engine = Some(engine);
                        self.phase = InitialNnuecPhase::CaptureMetadata;
                    }
                    // The actual API keeps a failed claim's unique engine on
                    // this same actor. Retain actor and outstanding obligation.
                    Err(error) => return self.fail(InitialNnuecFailure::Worker(error)),
                }
            }
            InitialNnuecPhase::CaptureMetadata => {
                let Some(engine) = self.owners.materialized_engine.as_ref() else {
                    return self.invalid("actual materialized engine absent before metadata");
                };
                match AdaptiveCheckerMetadata::capture_owned(engine.engine_for_registration()) {
                    Ok(metadata) => {
                        self.owners.checker_mirror = Some(Box::new(metadata.clone().into_mirror()));
                        self.owners.metadata = Some(metadata);
                        self.phase = InitialNnuecPhase::SpawnWorker;
                    }
                    Err(error) => return self.fail(InitialNnuecFailure::Metadata(error)),
                }
            }
            InitialNnuecPhase::SpawnWorker => {
                if self.owners.root.is_none() {
                    return self.invalid("original root absent before actual worker spawn");
                }
                let Some(engine) = self.owners.materialized_engine.take() else {
                    return self.invalid("actual materialized engine absent before worker spawn");
                };
                let result = match self.owners.root.as_ref() {
                    Some(root) => root.source().try_spawn_worker(
                        engine,
                        self.worker_owner,
                        self.signal.clone(),
                    ),
                    None => {
                        self.owners.materialized_engine = Some(engine);
                        return self.invalid("original worker source absent");
                    }
                };
                match result {
                    Ok(starting) => {
                        self.owners.starting_worker = Some(starting);
                        self.phase = InitialNnuecPhase::AwaitReady;
                    }
                    Err((engine, error)) => {
                        self.owners.materialized_engine = Some(engine);
                        return self.fail(InitialNnuecFailure::Worker(error));
                    }
                }
            }
            InitialNnuecPhase::AwaitReady => {
                let Some(starting) = self.owners.starting_worker.as_ref() else {
                    return self.invalid("actual Starting worker absent");
                };
                match starting.try_ready() {
                    Ok(None) => {}
                    Ok(Some(_)) => self.phase = InitialNnuecPhase::AcceptReady,
                    Err(error) => return self.fail(InitialNnuecFailure::Worker(error)),
                }
            }
            InitialNnuecPhase::AcceptReady => {
                let Some(starting) = self.owners.starting_worker.take() else {
                    return self.invalid("actual Starting worker absent before Ready acceptance");
                };
                match starting.accept_ready() {
                    Ok((worker, ready)) => {
                        self.owners.worker = Some(worker);
                        self.owners.ready = Some(ready);
                        self.phase = InitialNnuecPhase::FinishRoot;
                    }
                    Err((starting, CpuWorkerError::AlreadyReserved)) => {
                        self.owners.starting_worker = Some(starting);
                        self.phase = InitialNnuecPhase::AwaitReady;
                    }
                    Err((starting, error)) => {
                        self.owners.starting_worker = Some(starting);
                        return self.fail(InitialNnuecFailure::Worker(error));
                    }
                }
            }
            InitialNnuecPhase::FinishRoot => {
                match (
                    self.owners.root.take(),
                    self.owners.original_seal.take(),
                    self.owners.ready.take(),
                    self.owners.checker_mirror.take(),
                ) {
                    (Some(root), Some(seal), Some(ready), Some(checker)) => {
                        match root.finish_with_ready(seal, ready, checker) {
                            Ok(handoff) => {
                                self.owners.root_handoff = Some(handoff);
                                self.phase = InitialNnuecPhase::AdoptPort;
                            }
                            Err(PreparingRootFinishRejection {
                                root,
                                original_review_seal,
                                ready,
                                checker,
                                error,
                            }) => {
                                self.owners.root = Some(root);
                                self.owners.original_seal = Some(original_review_seal);
                                self.owners.ready = Some(ready);
                                self.owners.checker_mirror = Some(checker);
                                // The actual worker and captured metadata
                                // have never left their original owner slots.
                                return self.fail(InitialNnuecFailure::Finish(error));
                            }
                        }
                    }
                    (root, seal, ready, checker) => {
                        self.owners.root = root;
                        self.owners.original_seal = seal;
                        self.owners.ready = ready;
                        self.owners.checker_mirror = checker;
                        return self
                            .invalid("original root/seal/Ready/mirror finish custody absent");
                    }
                }
            }
            InitialNnuecPhase::AdoptPort => {
                match (
                    self.owners.root_handoff.take(),
                    self.owners.worker.take(),
                    self.owners.metadata.take(),
                ) {
                    (Some(handoff), Some(worker), Some(metadata)) => {
                        let (port, custody) =
                            AdaptiveEnginePort::from_preparing_handoff(handoff).into_parts();
                        return InitialNnuecStep::Ready(Box::new(InitialNnuecReady {
                            port,
                            custody,
                            worker,
                            actor: self.owners.actor,
                            metadata,
                        }));
                    }
                    (handoff, worker, metadata) => {
                        self.owners.root_handoff = handoff;
                        self.owners.worker = worker;
                        self.owners.metadata = metadata;
                        return self.invalid("actual port handoff/worker/metadata custody absent");
                    }
                }
            }
        }
        InitialNnuecStep::Pending(self)
    }
}
