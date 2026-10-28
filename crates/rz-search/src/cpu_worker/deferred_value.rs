//! Deferred native-R construction from an actual accepted NNUEC ReviewView.
//!
//! Startup retains only the immutable model/configuration. It neither creates
//! a bootstrap evaluator nor claims an R owner exists before a real Bound C.
//! The preparation actor performs the first TT allocation; the manager only
//! accepts its original-source result and the new worker's actual readback.

use super::{
    CPU_WORKER_COMMAND_CAPACITY, CpuWorker, CpuWorkerError, CpuWorkerProfile, CpuWorkerShutdown,
};
use crate::cpu::{
    CPU_PAUSED_STACK_MAX_BYTES, CpuConfig, CpuEngine, CpuError, CpuOrderingPolicy, CpuResumePolicy,
    CpuRootTiePolicy,
};
use crate::cpu_value::nnuec::NnuecError;
use crate::cpu_value::nnuec::v2::Model;
use crate::cpu_value::nnuec::v2::cpu::V2CpuValue;
use crate::cpu_value::nnuec::v2::review::{BoundNnuecReviewContext, Provenance};
use crate::cpu_value::{CpuValueError, CpuValueEvaluator};
use crate::pals::store::{PalsStores, StateId};
use rz_contracts::Digest;
use rz_contracts::pals_adaptive::CpuWorkerId;
use rz_runtime::CompletionSignal;
use rz_runtime::adaptive::LogicalCpu;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

/// An explicit declaration, not an observed placement or an allocated TT.
/// There is no default NNUEC profile or precision conversion here.
#[derive(Clone, Debug)]
pub struct DeferredNnuecCpuPlan {
    config: CpuConfig,
    ordering: CpuOrderingPolicy,
    root_tie_policy: CpuRootTiePolicy,
    paused_bytes_limit: usize,
    side_scope: bool,
    command_capacity: usize,
    affinity: Option<Vec<LogicalCpu>>,
}
impl DeferredNnuecCpuPlan {
    pub fn new(
        config: CpuConfig,
        ordering: CpuOrderingPolicy,
        paused_bytes_limit: usize,
        side_scope: bool,
        command_capacity: usize,
        affinity: Option<Vec<LogicalCpu>>,
    ) -> Result<Self, CpuWorkerError> {
        Self::new_with_root_tie_policy(
            config,
            ordering,
            paused_bytes_limit,
            side_scope,
            command_capacity,
            affinity,
            CpuRootTiePolicy::LegacyFirstSearched,
        )
    }

    /// Select the immutable root semantics before any native owner exists.
    pub fn new_with_root_tie_policy(
        config: CpuConfig,
        ordering: CpuOrderingPolicy,
        paused_bytes_limit: usize,
        side_scope: bool,
        command_capacity: usize,
        affinity: Option<Vec<LogicalCpu>>,
        root_tie_policy: CpuRootTiePolicy,
    ) -> Result<Self, CpuWorkerError> {
        // These are the actual native constructor's supported bounds. The
        // allocation remains deferred until the actor has an accepted Bound.
        config.tt_allocation_bytes()?;
        if !(1..=64).contains(&config.max_depth)
            || config.quiescence_ply > 32
            || !(1..=CPU_PAUSED_STACK_MAX_BYTES).contains(&paused_bytes_limit)
            || !(1..=CPU_WORKER_COMMAND_CAPACITY).contains(&command_capacity)
            || affinity.as_ref().is_some_and(|cpus| {
                cpus.is_empty()
                    || cpus.len() > 1024
                    || cpus
                        .iter()
                        .enumerate()
                        .any(|(i, cpu)| cpus[..i].contains(cpu))
            })
        {
            return Err(CpuWorkerError::Invalid("deferred NNUEC CPU declaration"));
        }
        Ok(Self {
            config,
            ordering,
            root_tie_policy,
            paused_bytes_limit,
            side_scope,
            command_capacity,
            affinity,
        })
    }
    pub fn config(&self) -> &CpuConfig {
        &self.config
    }
    pub fn ordering_policy(&self) -> CpuOrderingPolicy {
        self.ordering
    }
    pub fn root_tie_policy(&self) -> CpuRootTiePolicy {
        self.root_tie_policy
    }
    pub fn paused_bytes_limit(&self) -> usize {
        self.paused_bytes_limit
    }
    pub fn preserves_paused_side_scope(&self) -> bool {
        self.side_scope
    }
    pub fn command_capacity(&self) -> usize {
        self.command_capacity
    }
    pub fn requested_affinity(&self) -> Option<&[LogicalCpu]> {
        self.affinity.as_deref()
    }
}

struct Declaration {
    model: Arc<Model>,
    plan: DeferredNnuecCpuPlan,
}

#[path = "unstarted_value.rs"]
mod unstarted;
pub use unstarted::{
    NnuecMaterializationReleaseAck, NnuecMaterializationReleaseCompletion,
    NnuecMaterializationReleaseId,
};

/// A typed future native owner. This value contains no CpuEngine, checker,
/// worker, TT, paused token or substitute evaluator.
#[derive(Clone)]
pub struct DeferredNnuecRSource {
    declaration: Arc<Declaration>,
}
impl DeferredNnuecRSource {
    pub fn new(model: Arc<Model>, plan: DeferredNnuecCpuPlan) -> Self {
        Self {
            declaration: Arc::new(Declaration { model, plan }),
        }
    }
    pub fn model(&self) -> &Arc<Model> {
        &self.declaration.model
    }
    pub fn same_authority(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.declaration, &other.declaration)
    }
    pub fn cpu_plan(&self) -> &DeferredNnuecCpuPlan {
        &self.declaration.plan
    }
    /// Actual semantic-owner validation precedes this call. The additional
    /// native constructor checks the same Store/pin/root/generation and full
    /// original view seal; no view consumption or native node is charged.
    pub fn prepare_materialization(
        &self,
        bound: Arc<BoundNnuecReviewContext>,
        stores: &PalsStores,
        original_view_seal: [u8; 32],
    ) -> Result<MaterializationRequest, NnuecError> {
        let value = Arc::new(V2CpuValue::from_bound(
            Arc::clone(self.model()),
            Arc::clone(&bound),
            stores,
            original_view_seal,
        )?);
        let binding = value
            .job_binding_digest()
            .map_err(|_| NnuecError::StateMismatch)?
            .ok_or(NnuecError::StateMismatch)?;
        Ok(MaterializationRequest {
            declaration: Arc::clone(&self.declaration),
            value,
            bound,
            binding,
        })
    }
    /// Only a completion from this exact immutable declaration is accepted.
    /// The engine has never searched; thread start creates its first R owner.
    /// A definite pre-thread failure returns the same unique engine/Bound.
    #[allow(clippy::result_large_err)]
    pub fn try_spawn_worker(
        &self,
        materialized: MaterializedNnuecEngine,
        owner: CpuWorkerId,
        signal: CompletionSignal,
    ) -> Result<StartingNnuecWorker, (MaterializedNnuecEngine, CpuWorkerError)> {
        if !Arc::ptr_eq(&self.declaration, &materialized.declaration) {
            return Err((
                materialized,
                CpuWorkerError::Invalid("deferred NNUEC source/binding revoked"),
            ));
        }
        if let Err(reason) = validate_materialized_readback(
            &materialized.engine,
            &self.declaration.model,
            &materialized.bound,
            materialized.binding,
            self.declaration.plan.root_tie_policy,
            "deferred NNUEC source/binding revoked",
        ) {
            return Err((materialized, reason));
        }
        let Some(stack_owner_id) = materialized
            .engine
            .paused_stack_snapshot()
            .filter(|snapshot| snapshot.complete && snapshot.owner_id != 0)
            .map(|snapshot| snapshot.owner_id)
        else {
            return Err((materialized, CpuWorkerError::Quarantined));
        };
        let MaterializedNnuecEngine {
            declaration,
            engine,
            bound,
            binding,
        } = materialized;
        let plan = &declaration.plan;
        let worker = match &plan.affinity {
            Some(cpus) => CpuWorker::spawn_notified_retaining_with_thread_affinity(
                owner,
                engine,
                signal,
                plan.command_capacity,
                cpus.clone(),
            ),
            None => CpuWorker::spawn_notified_retaining_with_capacity(
                owner,
                engine,
                signal,
                plan.command_capacity,
            ),
        };
        match worker {
            Ok(mut worker) => {
                worker.deferred_nnuec_source = Some(self.clone());
                Ok(StartingNnuecWorker {
                    worker,
                    declaration,
                    bound,
                    binding,
                    stack_owner_id,
                })
            }
            Err((engine, error)) => Err((
                MaterializedNnuecEngine {
                    declaration,
                    engine,
                    bound,
                    binding,
                },
                error,
            )),
        }
    }
}

/// May be queued only on the one preparation actor. Private fields prevent a
/// numeric model declaration, fabricated profile or latest-view substitution.
pub struct MaterializationRequest {
    declaration: Arc<Declaration>,
    value: Arc<V2CpuValue>,
    bound: Arc<BoundNnuecReviewContext>,
    binding: Digest,
}
impl std::fmt::Debug for MaterializationRequest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MaterializationRequest")
            .field("provenance", self.bound.provenance())
            .field("binding", &self.binding)
            .finish_non_exhaustive()
    }
}
impl MaterializationRequest {
    pub fn provenance(&self) -> &Provenance {
        self.bound.provenance()
    }
    pub fn binding_digest(&self) -> Digest {
        self.binding
    }
    pub(crate) fn model_matches(&self, model: &Arc<Model>) -> bool {
        Arc::ptr_eq(&self.declaration.model, model)
    }
    pub(crate) fn bound(&self) -> &BoundNnuecReviewContext {
        &self.bound
    }
    // This is deliberately crate-private: the actor owns expensive initial
    // TT/history allocation. A manager facade cannot run this constructor.
    #[allow(clippy::result_large_err)]
    pub(crate) fn materialize(self) -> Result<MaterializedNnuecEngine, (Self, CpuWorkerError)> {
        if let Some(reason) = self.bound.authority_revocation() {
            return Err((self, CpuWorkerError::NnuecAuthorityRevoked(reason)));
        }
        let plan = &self.declaration.plan;
        let value: Arc<dyn CpuValueEvaluator> = self.value.clone();
        let result = if plan.side_scope {
            CpuEngine::with_paused_side_scope_and_tie_policy(
                plan.config.clone(),
                value,
                plan.ordering,
                plan.root_tie_policy,
            )
        } else {
            CpuEngine::with_evaluator_ordering_resume_and_tie_policy(
                plan.config.clone(),
                value,
                plan.ordering,
                CpuResumePolicy::PausedStack,
                plan.root_tie_policy,
            )
        };
        let mut engine = match result {
            Ok(engine) => engine,
            Err(error) => return Err((self, error.into())),
        };
        if let Err(error) = engine.set_paused_stack_max_bytes(plan.paused_bytes_limit) {
            return Err((self, error.into()));
        }
        if let Err(reason) = validate_materialized_readback(
            &engine,
            &self.declaration.model,
            &self.bound,
            self.binding,
            plan.root_tie_policy,
            "deferred NNUEC materialization readback",
        ) {
            return Err((self, reason));
        }
        self.bound.record_materialized();
        Ok(MaterializedNnuecEngine {
            declaration: self.declaration,
            engine,
            bound: self.bound,
            binding: self.binding,
        })
    }
}

/// A live binding readback remains mandatory. An explicit original authority
/// revocation requires the same typed reason and actual immutable V2 owner.
/// A StateMismatch or a foreign/malformed owner is never reclassified from
/// the current cancel flag or clock into a control observation.
fn validate_materialized_readback(
    engine: &CpuEngine,
    model: &Arc<Model>,
    bound: &Arc<BoundNnuecReviewContext>,
    binding: Digest,
    root_tie_policy: CpuRootTiePolicy,
    invalid: &'static str,
) -> Result<(), CpuWorkerError> {
    let observed = engine.job_binding_digest();
    if engine.value_identity() != model.identity() || engine.root_tie_policy() != root_tie_policy {
        return Err(CpuWorkerError::Invalid(invalid));
    }
    match observed {
        Ok(Some(actual)) if actual == binding => {}
        Err(CpuError::Value(CpuValueError::AuthorityRevoked(typed))) => {
            if let Some(reason) = bound.authority_revocation() {
                let actual = engine.nnuec_retirement_owner().ok().flatten();
                if crate::cpu_value::CpuValueAuthorityRevocation::from(reason) == typed
                    && actual.as_ref().is_some_and(|owner| {
                        owner.matches_model(model)
                            && owner.matches_bound(bound)
                            && owner.binding_digest() == binding
                    })
                {
                    return Err(CpuWorkerError::NnuecAuthorityRevoked(reason));
                }
            }
            return Err(CpuWorkerError::Invalid(invalid));
        }
        _ => return Err(CpuWorkerError::Invalid(invalid)),
    }
    if let Some(reason) = bound.authority_revocation() {
        return Err(CpuWorkerError::NnuecAuthorityRevoked(reason));
    }
    Ok(())
}

/// Actor-produced, unique, unsearched engine. The declaration authority is
/// private and cannot be replaced by an equal public profile/hash.
pub struct MaterializedNnuecEngine {
    declaration: Arc<Declaration>,
    engine: CpuEngine,
    bound: Arc<BoundNnuecReviewContext>,
    binding: Digest,
}
impl std::fmt::Debug for MaterializedNnuecEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MaterializedNnuecEngine")
            .field("provenance", self.bound.provenance())
            .field("binding", &self.binding)
            .finish_non_exhaustive()
    }
}
impl MaterializedNnuecEngine {
    pub fn provenance(&self) -> &Provenance {
        self.bound.provenance()
    }
    pub fn binding_digest(&self) -> Digest {
        self.binding
    }
    /// Capture the declaration mirror from this actual unique engine before
    /// its ownership transfer. This borrow cannot run mutable CPU analysis.
    pub fn engine_for_registration(&self) -> &CpuEngine {
        &self.engine
    }
}

/// A single-use, O(1) transfer from the preparation owner. An unclaimed TT is
/// still owned by that actor and is reclaimed there during its final join.
pub struct MaterializationHandoff {
    engine: Arc<Mutex<Option<MaterializedNnuecEngine>>>,
    outstanding: Arc<AtomicUsize>,
}
impl MaterializationHandoff {
    pub fn claim(self) -> Result<MaterializedNnuecEngine, CpuWorkerError> {
        let mut slot = self
            .engine
            .lock()
            .map_err(|_| CpuWorkerError::Quarantined)?;
        let engine = slot.take().ok_or(CpuWorkerError::UnknownJob)?;
        if self
            .outstanding
            .compare_exchange(2, 0, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            // Preserve the unique engine in actor custody if the control
            // sequence is inconsistent; no replacement work is admitted.
            *slot = Some(engine);
            return Err(CpuWorkerError::Quarantined);
        }
        Ok(engine)
    }
}

/// Retained exclusively on the existing preparation thread, not in a manager
/// completion queue. It never owns a running CPU worker or a paused token.
pub(crate) struct MaterializationOwner {
    engine: Arc<Mutex<Option<MaterializedNnuecEngine>>>,
    outstanding: Arc<AtomicUsize>,
}
impl MaterializationOwner {
    pub(crate) fn handoff(
        engine: MaterializedNnuecEngine,
        outstanding: Arc<AtomicUsize>,
    ) -> (Self, MaterializationHandoff) {
        let engine = Arc::new(Mutex::new(Some(engine)));
        (
            Self {
                engine: Arc::clone(&engine),
                outstanding: Arc::clone(&outstanding),
            },
            MaterializationHandoff {
                engine,
                outstanding,
            },
        )
    }
}
impl Drop for MaterializationOwner {
    fn drop(&mut self) {
        // This destructor runs on the preparation owner. A never-claimed
        // engine has performed no search and can be released there safely.
        let mut slot = match self.engine.lock() {
            Ok(slot) => slot,
            Err(poisoned) => poisoned.into_inner(),
        };
        if slot.take().is_some() {
            let _ = self
                .outstanding
                .compare_exchange(2, 0, Ordering::AcqRel, Ordering::Acquire);
        }
    }
}

/// Materialization has its own terminal channel; it cannot be mistaken for a
/// neural preparation completion or a replacement-value installation receipt.
pub struct MaterializationCompletion {
    pub provenance: Provenance,
    pub result: Result<MaterializationHandoff, (MaterializationRequest, CpuWorkerError)>,
}

/// Owns the actual thread while its Ready/affinity observations are pending.
/// Shutdown transfers remain possible even when acceptance is revoked.
pub struct StartingNnuecWorker {
    worker: CpuWorker,
    declaration: Arc<Declaration>,
    bound: Arc<BoundNnuecReviewContext>,
    binding: Digest,
    stack_owner_id: u64,
}
impl std::fmt::Debug for StartingNnuecWorker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StartingNnuecWorker")
            .field("provenance", self.bound.provenance())
            .field("binding", &self.binding)
            .field("stack_owner_id", &self.stack_owner_id)
            .finish_non_exhaustive()
    }
}
impl StartingNnuecWorker {
    pub fn provenance(&self) -> &Provenance {
        self.bound.provenance()
    }
    pub fn try_ready(&self) -> Result<Option<CpuWorkerProfile>, CpuWorkerError> {
        if let Some(reason) = self.bound.authority_revocation() {
            return Err(CpuWorkerError::NnuecAuthorityRevoked(reason));
        }
        match self.worker.ready_readback() {
            None => return Ok(None),
            Some(Err(_)) => return Err(CpuWorkerError::Quarantined),
            Some(Ok(_)) => {}
        }
        if self.worker.thread_affinity_requested() {
            match self.worker.thread_affinity_ready() {
                None => return Ok(None),
                Some(Err(_)) => return Err(CpuWorkerError::Quarantined),
                Some(Ok(_)) => {}
            }
        }
        let profile = self.worker.acknowledged_profile()?;
        if profile.evaluation_namespace != self.binding
            || profile.max_depth != self.declaration.plan.config.max_depth
            || self.worker.command_capacity() != self.declaration.plan.command_capacity
            || self.worker.paused_bytes_limit() != self.declaration.plan.paused_bytes_limit
            || self.worker.thread_affinity_requested() != self.declaration.plan.affinity.is_some()
        {
            return Err(CpuWorkerError::Quarantined);
        }
        Ok(Some(profile))
    }
    /// The caller retains this actual owner on failure and must use the normal
    /// cancel/drain/join path; no failed thread is replaced by a mock owner.
    #[allow(clippy::result_large_err)]
    pub fn accept_ready(
        self,
    ) -> Result<(CpuWorker, NnuecReadyRegistration), (Self, CpuWorkerError)> {
        match self.try_ready() {
            Ok(Some(profile)) => Ok((
                self.worker,
                NnuecReadyRegistration {
                    declaration: self.declaration,
                    bound: self.bound,
                    profile,
                    binding: self.binding,
                    stack_owner_id: self.stack_owner_id,
                },
            )),
            Ok(None) => Err((self, CpuWorkerError::AlreadyReserved)),
            Err(error) => Err((self, error)),
        }
    }
    /// Shutdown-only facade: unlike exposing CpuWorker, this cannot prepare a
    /// Job or admit native work before the initial Ready gate succeeds.
    pub fn request_shutdown(&self) {
        self.worker.request_shutdown();
    }
    pub fn try_join(&mut self) -> Result<Option<CpuWorkerShutdown>, CpuWorkerError> {
        self.worker.try_join()
    }
    pub fn take_joined_engine(&mut self) -> Result<CpuEngine, CpuWorkerError> {
        self.worker.take_engine()
    }
    /// Cleanup only, including a revoked/expired Ready gate. Actual known
    /// worker join is still required by the same private worker boundary.
    pub fn take_joined_nnuec_retirement(
        &mut self,
    ) -> Result<super::RecoveredNnuecEngine, CpuWorkerError> {
        self.worker.take_nnuec_retirement()
    }
    /// Blocking recovery belongs outside the manager event loop.
    pub fn shutdown(self) -> Result<CpuWorkerShutdown, CpuWorkerError> {
        self.worker.shutdown()
    }
}

/// Only the successful initial actual Ready gate can mint this registration.
/// It is separate from a replacement receipt and has no invented old profile.
/// The semantic owner must additionally validate its own original Q/view seal.
pub struct NnuecReadyRegistration {
    declaration: Arc<Declaration>,
    bound: Arc<BoundNnuecReviewContext>,
    profile: CpuWorkerProfile,
    binding: Digest,
    stack_owner_id: u64,
}
impl NnuecReadyRegistration {
    /// Equal weights/configuration cannot substitute another declared source.
    pub fn matches_source(&self, source: &DeferredNnuecRSource) -> bool {
        Arc::ptr_eq(&self.declaration, &source.declaration)
    }
    pub fn profile(&self) -> CpuWorkerProfile {
        self.profile
    }
    pub fn actual_stack_owner_id(&self) -> u64 {
        self.stack_owner_id
    }
    pub fn binding_digest(&self) -> Digest {
        self.binding
    }
    pub fn value_identity(&self) -> &crate::cpu_value::CpuValueIdentity {
        self.declaration.model.identity()
    }
    pub fn cpu_plan(&self) -> &DeferredNnuecCpuPlan {
        &self.declaration.plan
    }
    pub fn provenance(&self) -> &Provenance {
        self.bound.provenance()
    }
    pub fn validates(
        &self,
        stores: &PalsStores,
        root: StateId,
        generation: u64,
        original_view_seal: [u8; 32],
    ) -> bool {
        self.stack_owner_id != 0
            && self.profile.evaluation_namespace == self.binding
            && self.profile.max_depth == self.declaration.plan.config.max_depth
            && self.bound.provenance().model == *self.declaration.model.identity()
            && self
                .bound
                .validates(stores, root, generation, original_view_seal)
    }
}

#[cfg(test)]
#[path = "deferred_value_tests.rs"]
mod tests;
