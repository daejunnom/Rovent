//! One-owner, one-in-flight prepared value installation. This is separate
//! from CPU Job/Slice admission and never transfers or disposes a paused Job.

use super::{
    CpuEngine, CpuWorkerError, CpuWorkerProfile, cpu_evaluation_namespace_digest,
    cpu_search_conditions_digest,
};
use crate::cpu_value::nnuec::NnuecError;
use crate::cpu_value::nnuec::v2::Model;
use crate::cpu_value::nnuec::v2::cpu::V2CpuValue;
use crate::cpu_value::nnuec::v2::review::{
    BoundInstanceAuthority, BoundNnuecReviewContext, Provenance,
};
use crate::cpu_value::{CpuValueEvaluator, CpuValueIdentity};
use crate::pals::store::PalsStores;
use rz_contracts::Digest;
use rz_runtime::CompletionSignal;
use rz_runtime::adaptive::MailboxOccupancy;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;
use std::sync::mpsc::{self, Receiver, SyncSender, TryRecvError, TrySendError};

/// Only actual accepted ReviewView custody can construct an installation.
/// No weights/model are cloned; the immutable evaluator and view are pinned.
pub struct PreparedValueInstallation {
    evaluator: Arc<dyn CpuValueEvaluator>,
    bound: Arc<BoundNnuecReviewContext>,
    binding: Digest,
}
impl std::fmt::Debug for PreparedValueInstallation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PreparedValueInstallation")
            .field("value", self.evaluator.identity())
            .field("provenance", self.bound.provenance())
            .field("binding", &self.binding)
            .finish_non_exhaustive()
    }
}
impl PreparedValueInstallation {
    pub fn from_bound(
        model: Arc<Model>,
        bound: Arc<BoundNnuecReviewContext>,
        stores: &PalsStores,
        original_view_seal: [u8; 32],
    ) -> Result<Self, NnuecError> {
        let evaluator =
            V2CpuValue::from_bound(model, Arc::clone(&bound), stores, original_view_seal)?;
        let binding = evaluator
            .job_binding_digest()
            .map_err(|_| NnuecError::StateMismatch)?
            .ok_or(NnuecError::StateMismatch)?;
        if !bound.authority_is_live() {
            return Err(NnuecError::StateMismatch);
        }
        Ok(Self {
            evaluator: Arc::new(evaluator),
            bound,
            binding,
        })
    }
    pub fn binding_digest(&self) -> Digest {
        self.binding
    }
    pub fn provenance(&self) -> &Provenance {
        self.bound.provenance()
    }
    pub fn value_identity(&self) -> &CpuValueIdentity {
        self.evaluator.identity()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct InstallId(u64);
impl InstallId {
    pub fn sequence(self) -> u64 {
        self.0
    }
}

pub struct InstallRejected {
    payload: PreparedValueInstallation,
    reason: CpuWorkerError,
}
impl std::fmt::Debug for InstallRejected {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InstallRejected")
            .field("reason", &self.reason)
            .field("payload", &self.payload)
            .finish()
    }
}
impl InstallRejected {
    pub(super) fn new(payload: PreparedValueInstallation, reason: CpuWorkerError) -> Self {
        Self { payload, reason }
    }
    pub fn reason(&self) -> &CpuWorkerError {
        &self.reason
    }
    pub fn into_parts(self) -> (CpuWorkerError, PreparedValueInstallation) {
        (self.reason, self.payload)
    }
}

/// Readback originates from the real owner after replacing its idle engine.
/// The new constructor owns a fresh TT/history; no old evidence is relabelled.
#[derive(Clone, Debug)]
pub struct ValueInstallReadback {
    profile: CpuWorkerProfile,
    paused_bytes_limit: usize,
    side_scope_preserved: bool,
    stack_owner_id: u64,
}
impl ValueInstallReadback {
    pub fn actual_stack_owner_id(&self) -> u64 {
        self.stack_owner_id
    }
    pub fn profile(&self) -> CpuWorkerProfile {
        self.profile
    }
    pub fn tt_reset(&self) -> bool {
        true
    }
    pub fn ordering_history_reset(&self) -> bool {
        true
    }
    pub fn paused_bytes_limit(&self) -> usize {
        self.paused_bytes_limit
    }
    pub fn side_scope_preserved(&self) -> bool {
        self.side_scope_preserved
    }
}

/// The channel authority and request matching fields are private. Manager
/// polling verifies this receipt before it can expose a new worker profile.
#[derive(Clone, Debug)]
pub struct ValueInstallReceipt {
    authority: Arc<()>,
    id: InstallId,
    previous_profile: CpuWorkerProfile,
    binding: Digest,
    bound_authority: BoundInstanceAuthority,
    result: Result<ValueInstallReadback, CpuWorkerError>,
}
impl ValueInstallReceipt {
    /// This identifies the exact installed Bound instance without retaining
    /// its ReviewView/Store pin. Equal semantic binding bytes are insufficient.
    pub fn matches_bound(&self, bound: &Arc<BoundNnuecReviewContext>) -> bool {
        self.result.is_ok() && self.bound_authority.matches(bound.instance_authority())
    }
    pub fn installation_id(&self) -> InstallId {
        self.id
    }
    pub fn binding_digest(&self) -> Digest {
        self.binding
    }
    pub fn is_installed(&self) -> bool {
        self.result.is_ok()
    }
    pub fn installed_profile(&self) -> Option<CpuWorkerProfile> {
        self.result.as_ref().ok().map(ValueInstallReadback::profile)
    }
    pub fn result(&self) -> Result<&ValueInstallReadback, &CpuWorkerError> {
        self.result.as_ref()
    }
    pub fn previous_profile(&self) -> CpuWorkerProfile {
        self.previous_profile
    }
}

struct Request {
    authority: Arc<()>,
    id: InstallId,
    previous_profile: CpuWorkerProfile,
    payload: PreparedValueInstallation,
}
struct Pending {
    id: InstallId,
    previous_profile: CpuWorkerProfile,
    binding: Digest,
    bound_authority: BoundInstanceAuthority,
}
struct Acknowledged {
    id: InstallId,
    binding: Digest,
    profile: CpuWorkerProfile,
    bound_authority: BoundInstanceAuthority,
}

pub(super) struct Endpoint {
    authority: Arc<()>,
    sender: SyncSender<Request>,
    receiver: Receiver<ValueInstallReceipt>,
    sequence: u64,
    pending: Option<Pending>,
    acknowledged: Option<Acknowledged>,
    quarantined: bool,
}
pub(super) struct Owner {
    authority: Arc<()>,
    receiver: Receiver<Request>,
    sender: SyncSender<ValueInstallReceipt>,
    /// Normal one-in-flight admission reserves the single terminal slot.
    /// Preserve a protocol-failure terminal instead of overwriting a receipt.
    retained_terminal: Option<ValueInstallReceipt>,
    quarantined: bool,
}
pub(super) fn channel() -> (Endpoint, Owner) {
    let (sender, requests) = mpsc::sync_channel(1);
    let (receipts, receiver) = mpsc::sync_channel(1);
    let authority = Arc::new(());
    (
        Endpoint {
            authority: Arc::clone(&authority),
            sender,
            receiver,
            sequence: 0,
            pending: None,
            acknowledged: None,
            quarantined: false,
        },
        Owner {
            authority,
            receiver: requests,
            sender: receipts,
            retained_terminal: None,
            quarantined: false,
        },
    )
}

impl Endpoint {
    pub(super) fn pending(&self) -> bool {
        self.pending.is_some() || self.quarantined
    }
    pub(super) fn try_submit(
        &mut self,
        payload: PreparedValueInstallation,
        previous_profile: CpuWorkerProfile,
    ) -> Result<InstallId, InstallRejected> {
        let reject = if self.quarantined {
            Some(CpuWorkerError::Quarantined)
        } else if self.pending.is_some() {
            Some(CpuWorkerError::Capacity)
        } else if !payload.bound.authority_is_live() {
            Some(CpuWorkerError::Invalid("prepared value authority revoked"))
        } else {
            None
        };
        if let Some(reason) = reject {
            return Err(InstallRejected::new(payload, reason));
        }
        let Some(sequence) = self.sequence.checked_add(1) else {
            self.quarantined = true;
            return Err(InstallRejected::new(payload, CpuWorkerError::Quarantined));
        };
        let id = InstallId(sequence);
        let binding = payload.binding;
        let bound_authority = payload.bound.instance_authority().clone();
        match self.sender.try_send(Request {
            authority: Arc::clone(&self.authority),
            id,
            previous_profile,
            payload,
        }) {
            Ok(()) => {
                self.sequence = sequence;
                self.pending = Some(Pending {
                    id,
                    previous_profile,
                    binding,
                    bound_authority,
                });
                Ok(id)
            }
            Err(TrySendError::Full(request)) => Err(InstallRejected::new(
                request.payload,
                CpuWorkerError::Capacity,
            )),
            Err(TrySendError::Disconnected(request)) => Err(InstallRejected::new(
                request.payload,
                CpuWorkerError::Disconnected,
            )),
        }
    }
    pub(super) fn try_ack(&mut self) -> Result<Option<ValueInstallReceipt>, CpuWorkerError> {
        if self.quarantined {
            return Err(CpuWorkerError::Quarantined);
        }
        let receipt = match self.receiver.try_recv() {
            Ok(receipt) => receipt,
            Err(TryRecvError::Empty) => return Ok(None),
            Err(TryRecvError::Disconnected) => {
                if self.pending.is_some() {
                    self.quarantined = true;
                    return Err(CpuWorkerError::Quarantined);
                }
                return Ok(None);
            }
        };
        let valid = self.pending.as_ref().is_some_and(|pending| {
            Arc::ptr_eq(&receipt.authority, &self.authority)
                && receipt.id == pending.id
                && receipt.previous_profile == pending.previous_profile
                && receipt.binding == pending.binding
                && receipt.bound_authority.matches(&pending.bound_authority)
                && match &receipt.result {
                    Ok(readback) => {
                        readback.profile.owner == pending.previous_profile.owner
                            && readback.profile.max_depth == pending.previous_profile.max_depth
                            && readback.profile.search_conditions
                                == pending.previous_profile.search_conditions
                            && readback.profile.evaluation_namespace == pending.binding
                    }
                    Err(_) => true,
                }
        });
        if !valid {
            self.quarantined = true;
            return Err(CpuWorkerError::Quarantined);
        }
        self.pending = None;
        if let Some(profile) = receipt.installed_profile() {
            self.acknowledged = Some(Acknowledged {
                id: receipt.id,
                binding: receipt.binding,
                profile,
                bound_authority: receipt.bound_authority.clone(),
            });
        }
        if matches!(&receipt.result, Err(CpuWorkerError::Quarantined)) {
            self.quarantined = true;
        }
        Ok(Some(receipt))
    }
    pub(super) fn validates_receipt(&self, receipt: &ValueInstallReceipt) -> bool {
        !self.pending()
            && Arc::ptr_eq(&receipt.authority, &self.authority)
            && self.acknowledged.as_ref().is_some_and(|acknowledged| {
                receipt.id == acknowledged.id
                    && receipt.binding == acknowledged.binding
                    && receipt
                        .bound_authority
                        .matches(&acknowledged.bound_authority)
                    && receipt.installed_profile() == Some(acknowledged.profile)
            })
    }
}

pub(super) fn registered_namespace(
    engine: &CpuEngine,
    binding: Option<Digest>,
) -> Result<Digest, CpuWorkerError> {
    match (engine.job_binding_digest()?, binding) {
        (Some(actual), Some(registered)) if actual != registered => {
            Err(CpuWorkerError::Quarantined)
        }
        (Some(actual), _) => Ok(actual),
        (None, Some(_)) => Err(CpuWorkerError::Quarantined),
        (None, None) => cpu_evaluation_namespace_digest(engine),
    }
}

impl Owner {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn service(
        &mut self,
        engine: &mut CpuEngine,
        current_binding: &mut Option<Digest>,
        actual_idle: bool,
        engine_reusable: bool,
        occupancy: MailboxOccupancy,
        shutdown: bool,
        signal: &CompletionSignal,
    ) -> Result<(), CpuWorkerError> {
        if self.quarantined {
            return Err(CpuWorkerError::Quarantined);
        }
        let request = match self.receiver.try_recv() {
            Ok(request) => request,
            Err(TryRecvError::Empty | TryRecvError::Disconnected) => return Ok(()),
        };
        let identity = Arc::clone(&request.authority);
        let id = request.id;
        let previous_profile = request.previous_profile;
        let binding = request.payload.binding;
        let bound_authority = request.payload.bound.instance_authority().clone();
        let result = catch_unwind(AssertUnwindSafe(|| {
            self.install(
                request,
                engine,
                current_binding,
                actual_idle,
                engine_reusable,
                occupancy,
                shutdown,
            )
        }));
        let (result, fatal) = match result {
            Ok(result) => {
                let fatal = matches!(&result, Err(CpuWorkerError::Quarantined));
                (result, fatal)
            }
            Err(_) => (Err(CpuWorkerError::Quarantined), true),
        };
        let receipt = ValueInstallReceipt {
            authority: identity,
            id,
            previous_profile,
            binding,
            bound_authority,
            result,
        };
        let published = match self.sender.try_send(receipt) {
            Ok(()) => true,
            Err(TrySendError::Full(receipt) | TrySendError::Disconnected(receipt)) => {
                self.retained_terminal = Some(receipt);
                false
            }
        };
        signal.notify();
        if fatal || !published {
            self.quarantined = true;
            return Err(CpuWorkerError::Quarantined);
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn install(
        &self,
        request: Request,
        engine: &mut CpuEngine,
        current_binding: &mut Option<Digest>,
        actual_idle: bool,
        engine_reusable: bool,
        occupancy: MailboxOccupancy,
        shutdown: bool,
    ) -> Result<ValueInstallReadback, CpuWorkerError> {
        if !Arc::ptr_eq(&request.authority, &self.authority) || !engine_reusable {
            return Err(CpuWorkerError::Quarantined);
        }
        if shutdown || occupancy.closed {
            return Err(CpuWorkerError::Disconnected);
        }
        // Inherent paused_stack_bytes()==None means no retained task, not an
        // observed byte count. Require the actual supported complete snapshot
        // as the separate evidence that this PausedStack owner is known idle.
        let paused = engine
            .paused_stack_snapshot()
            .filter(|snapshot| snapshot.complete)
            .ok_or(CpuWorkerError::Quarantined)?;
        if !actual_idle
            || paused.tokens_retained != 0
            || paused.bytes_current != 0
            || engine.paused_stack_bytes().is_some()
            || engine.pending_reciprocal_request().is_some()
            || occupancy.commands != 0
            || occupancy.reliable_queued != 0
            || occupancy.reliable_reserved != 0
            || occupancy.progress_present
        {
            return Err(CpuWorkerError::Capacity);
        }
        let previous = request.previous_profile;
        if previous.max_depth != engine.config().max_depth
            || previous.search_conditions != cpu_search_conditions_digest(engine)
            // Old bound authority may have expired while this actual owner is
            // known idle. Compare its already registered immutable namespace;
            // only the new payload needs live authority for replacement.
            || previous.evaluation_namespace
                != current_binding.map_or_else(|| cpu_evaluation_namespace_digest(engine), Ok)?
        {
            return Err(CpuWorkerError::Invalid(
                "prepared value previous profile mismatch",
            ));
        }
        if !request.payload.bound.authority_is_live() {
            return Err(CpuWorkerError::Invalid("prepared value authority revoked"));
        }
        let side_scope = engine.preserves_paused_on_fresh_tasks();
        let root_tie_policy = engine.root_tie_policy();
        let evaluator = Arc::clone(&request.payload.evaluator);
        let mut replacement = if side_scope {
            CpuEngine::with_paused_side_scope_and_tie_policy(
                engine.config().clone(),
                evaluator,
                engine.ordering_policy(),
                root_tie_policy,
            )?
        } else {
            CpuEngine::with_evaluator_ordering_resume_and_tie_policy(
                engine.config().clone(),
                evaluator,
                engine.ordering_policy(),
                engine.resume_policy(),
                root_tie_policy,
            )?
        };
        replacement.set_paused_stack_max_bytes(engine.paused_stack_max_bytes())?;
        if replacement.value_identity() != request.payload.evaluator.identity()
            || replacement.job_binding_digest()? != Some(request.payload.binding)
            || replacement.config().max_depth != previous.max_depth
            || cpu_search_conditions_digest(&replacement) != previous.search_conditions
            || replacement.ordering_policy() != engine.ordering_policy()
            || replacement.root_tie_policy() != root_tie_policy
            || replacement.search_identity() != engine.search_identity()
            || replacement.resume_policy() != engine.resume_policy()
            || replacement.preserves_paused_on_fresh_tasks() != side_scope
            || replacement.paused_stack_max_bytes() != engine.paused_stack_max_bytes()
            || !request.payload.bound.authority_is_live()
        {
            return Err(CpuWorkerError::Invalid(
                "prepared value replacement readback mismatch",
            ));
        }
        let readback = ValueInstallReadback {
            profile: CpuWorkerProfile {
                evaluation_namespace: request.payload.binding,
                ..previous
            },
            paused_bytes_limit: replacement.paused_stack_max_bytes(),
            side_scope_preserved: side_scope,
            stack_owner_id: replacement
                .paused_stack_snapshot()
                .filter(|snapshot| snapshot.complete && snapshot.owner_id != 0)
                .ok_or(CpuWorkerError::Quarantined)?
                .owner_id,
        };
        request.payload.bound.record_materialized();
        let old = std::mem::replace(engine, replacement);
        *current_binding = Some(request.payload.binding);
        // An unexpected destructor unwind is quarantined by service; it is
        // never reported as a verified unchanged engine or a successful ack.
        drop(old);
        Ok(readback)
    }
}

#[cfg(test)]
#[path = "prepared_value_tests.rs"]
mod tests;
