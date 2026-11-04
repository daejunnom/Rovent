//! Original V3 teacher query dispatch custody. No target or publication is issued here.
use super::super::producer_v3::SubmittedTeacherQueryV3;
use super::super::{ArenaError, invalid};
use super::v3_session::RoventCollectionTeacherSessionV3;
use rz_contracts::pals_adaptive::RoleTicketEnvelope;
use rz_contracts::{Digest, pals::Role};
use rz_experiments::pals_data_v3::GenerationRoleV3;
use rz_search::external_cpu::ExternalGoConstraint;
use rz_search::external_cpu::collection_teacher::{
    TeacherCompletion, TeacherDispatch, TeacherInputScope, TeacherJob, TeacherJobId,
};
use rz_uci::rovent_external_launch::VerifiedRoventExternalStartup;
use std::any::Any;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// This object and the original session must be retained together by the root
/// worker until any admitted job has been drained and its owner joined.
pub(crate) struct TeacherDispatchOwnerV3 {
    session: Arc<Mutex<RoventCollectionTeacherSessionV3>>,
    actor_id: String,
    next_sequence: u64,
    pending: Option<PendingV3>,
}

struct PendingV3 {
    capsule: Arc<SubmittedTeacherQueryV3>,
    id: TeacherJobId,
    soft_quantum: Duration,
    dispatch: Option<TeacherDispatch>,
    startup: VerifiedRoventExternalStartup,
    cancel: Box<dyn Fn() -> Result<(), String> + Send>,
    cancel_requested: bool,
    cancel_failure: Option<String>,
    /// Unwind around submission may have occurred after mailbox admission.
    submission_panic: Option<Box<dyn Any + Send>>,
    completion: Option<TeacherCompletion>,
    completion_panic: Option<Box<dyn Any + Send>>,
    dispatch_mismatch: bool,
}

pub(crate) enum TeacherSubmitV3 {
    Submitted(TeacherJobId),
    /// No physical job was issued; retain and retry the same capsule later.
    StartupPending(Arc<SubmittedTeacherQueryV3>),
    /// Original capsule and possibly admitted physical owner remain retained.
    SubmissionUnknown(TeacherJobId),
}

pub(crate) struct TeacherRejectedV3 {
    pub error: ArenaError,
    pub capsule: Arc<SubmittedTeacherQueryV3>,
}

/// Completion is raw evidence only. Validation failure and cancellation are
/// retained alongside the exact original capsule, dispatch and terminal.
pub(crate) struct TeacherCompletedV3 {
    pub capsule: Arc<SubmittedTeacherQueryV3>,
    pub soft_quantum: Duration,
    pub actor_id: String,
    pub dispatch: TeacherDispatch,
    pub completion: TeacherCompletion,
    pub startup: VerifiedRoventExternalStartup,
    pub cancel_requested: bool,
    pub cancel_failure: Option<String>,
    pub validation_error: Option<String>,
}

impl TeacherDispatchOwnerV3 {
    pub(crate) fn new(
        session: Arc<Mutex<RoventCollectionTeacherSessionV3>>,
        actor_id: String,
    ) -> Result<Self, String> {
        {
            let mut owner = session
                .lock()
                .map_err(|_| "V3 teacher session lock poisoned".to_string())?;
            if !owner.claim_dispatch_owner(&actor_id) {
                return Err("V3 teacher actor dispatcher unavailable or already claimed".into());
            }
        }
        Ok(Self {
            session,
            actor_id,
            next_sequence: 1,
            pending: None,
        })
    }

    pub(crate) fn has_pending(&self) -> bool {
        self.pending.is_some()
    }

    pub(crate) fn pending_ticket(&self) -> Option<RoleTicketEnvelope> {
        self.pending
            .as_ref()
            .map(|pending| pending.capsule.ticket())
    }

    pub(crate) fn pending_id(&self) -> Option<TeacherJobId> {
        self.pending.as_ref().map(|pending| pending.id)
    }

    pub(crate) fn submission_unknown(&self) -> bool {
        self.pending
            .as_ref()
            .is_some_and(|pending| pending.submission_panic.is_some())
    }

    /// Transfer this exact owner, including any unknown terminal and original
    /// session, to the root's recovery custodian without manufacturing a fence.
    pub(crate) fn into_custody(self) -> Box<dyn Any + Send> {
        Box::new(self)
    }

    /// Called only by the original root's single collection worker. On any
    /// pre-admission rejection the exact capsule is returned to that worker.
    pub(crate) fn submit(
        &mut self,
        capsule: Arc<SubmittedTeacherQueryV3>,
        constraint: ExternalGoConstraint,
        soft_quantum: Duration,
    ) -> Result<TeacherSubmitV3, Box<TeacherRejectedV3>> {
        macro_rules! reject {
            ($error:expr) => {
                return Err(Box::new(TeacherRejectedV3 {
                    error: $error,
                    capsule,
                }))
            };
        }
        if self.pending.is_some() {
            reject!(invalid("V3 teacher actor has pending work"));
        }
        let mut session = match self.session.lock() {
            Ok(session) => session,
            Err(_) => reject!(invalid("V3 teacher session lock poisoned")),
        };
        let input = capsule.input_seal();
        let admission = session.original_admission();
        if session.is_closing()
            || capsule.is_canceled()
            || !capsule.has_policy_candidates()
            || !capsule.admission().same_original_admission(admission)
            || input.teacher_actor_id.as_deref() != Some(self.actor_id.as_str())
        {
            reject!(invalid("V3 teacher query root, actor or admission differs"));
        }
        if let Err(error) = admission.validate_ticket(&input.ticket) {
            reject!(invalid(error.to_string()));
        }
        let manifest_sha = match session
            .manifest_lock()
            .sha256(admission.native_origin().original_input_lock())
        {
            Ok(value) => value,
            Err(error) => reject!(invalid(error.to_string())),
        };
        if input.manifest_sha256 != manifest_sha {
            reject!(invalid("V3 teacher query manifest differs"));
        }
        let position = match capsule.original_teacher_query_position() {
            Ok(position) => position,
            Err(error) => reject!(error),
        };
        let role = match input.role {
            GenerationRoleV3::Proposer => Role::Proposer,
            GenerationRoleV3::Critic => Role::Critic,
            GenerationRoleV3::Validator => reject!(invalid("V3 teacher policy is P/C only")),
        };
        if input.ticket.role != role {
            reject!(invalid("V3 teacher ticket role differs"));
        }
        let parse = |value: &str| {
            rz_eval::asset::parse_sha256(value)
                .map(Digest)
                .map_err(|error| invalid(error.to_string()))
        };
        let input_digest = match parse(&input.input_sha256) {
            Ok(v) => v,
            Err(e) => reject!(e),
        };
        let view_digest = match parse(&input.selected_view_sha256) {
            Ok(v) => v,
            Err(e) => reject!(e),
        };
        let rules_digest = match parse(&input.anchor.rules_state_sha256) {
            Ok(v) => v,
            Err(e) => reject!(e),
        };
        let source_digest = match parse(&session.registration().source_sha256) {
            Ok(v) => v,
            Err(e) => reject!(e),
        };
        let registration = match session.actor_registration(&self.actor_id) {
            Some(value) => value,
            None => reject!(invalid("V3 teacher actor absent")),
        };
        let pin = match session.registration().teachers.get(&self.actor_id) {
            Some(value) => value.clone(),
            None => reject!(invalid("V3 teacher actor pin absent")),
        };
        let root = admission.root_admission();
        let root_digest = match root.descriptor().digest() {
            Ok(value) => value,
            Err(error) => reject!(invalid(error.to_string())),
        };
        let scope = TeacherInputScope {
            authority: input.ticket.authority,
            input_sha256: input_digest,
            view_sha256: view_digest,
            source_sha256: source_digest,
            role,
            rules_semantic: rules_digest,
            input_lock: root.startup_lock_digest(),
            root_admission: root_digest,
            root_budget: admission.root_budget(),
            frozen_actor: registration.frozen_actor,
        };
        let id = TeacherJobId {
            collection_owner: registration.collection_owner,
            sequence: self.next_sequence,
        };
        let next = match self.next_sequence.checked_add(1) {
            Some(value) => value,
            None => reject!(invalid("V3 teacher sequence exhausted")),
        };
        let Some(lifecycle) = session.lifecycle_mut(&self.actor_id) else {
            reject!(invalid("V3 teacher actor has not started"));
        };
        if lifecycle.owner().worker().registration() != registration
            || !lifecycle
                .owner()
                .admission()
                .same_original_admission(capsule.admission())
        {
            reject!(invalid("V3 teacher worker registration differs"));
        }
        let owner = lifecycle.owner();
        let startup = match owner.verified_startup() {
            Ok(Some(startup)) => startup,
            Ok(None) => return Ok(TeacherSubmitV3::StartupPending(capsule)),
            Err(error) => reject!(invalid(error.to_string())),
        };
        if startup.canonical_sha256() != pin.canonical_sha256
            || startup.capability_sha256() != pin.capability_sha256
            || startup.input_lock_sha256() != super::super::hex(registration.input_lock)
            || owner.launch().binary().sha256 != pin.binary_sha256
            || owner.launch().binary().bytes != pin.binary_bytes
        {
            reject!(invalid(
                "V3 teacher startup or binary differs from registered actor"
            ));
        }
        let control = lifecycle.owner().worker().control_mailbox(id);
        let cancel_control = control.clone();
        let cancel: Box<dyn Fn() -> Result<(), String> + Send> = Box::new(move || {
            cancel_control
                .request_cancel(1)
                .map_err(|error| format!("{error:?}"))
        });
        let job = TeacherJob {
            id,
            input_scope: scope,
            position,
            root_moves: None,
            constraint,
            original_deadline: input.ticket.original_deadline,
            soft_quantum,
            control,
        };
        let claim = match capsule.try_claim_teacher_issuance() {
            Ok(claim) => claim,
            Err(error) => reject!(error),
        };
        if capsule.is_canceled() {
            claim.rejected();
            reject!(invalid("V3 teacher query canceled before submission"));
        }
        // Establish custody before the physical mailbox is touched.
        self.pending = Some(PendingV3 {
            capsule,
            id,
            soft_quantum,
            dispatch: None,
            startup,
            cancel,
            cancel_requested: false,
            cancel_failure: None,
            submission_panic: None,
            completion: None,
            completion_panic: None,
            dispatch_mismatch: false,
        });
        self.next_sequence = next;
        if self
            .pending
            .as_ref()
            .expect("original query retained")
            .capsule
            .is_canceled()
        {
            claim.rejected();
            let original = self.pending.take().expect("original query retained");
            return Err(Box::new(TeacherRejectedV3 {
                error: invalid("V3 teacher query canceled before mailbox submission"),
                capsule: original.capsule,
            }));
        }
        let outcome = catch_unwind(AssertUnwindSafe(|| lifecycle.submit(job)));
        match outcome {
            Ok(Ok(dispatch)) => {
                claim.admitted();
                let pending = self.pending.as_mut().expect("original query retained");
                pending.dispatch_mismatch = dispatch.id() != id
                    || dispatch.process_identity() != pending.startup.process_identity();
                pending.dispatch = Some(dispatch);
                if pending.capsule.is_canceled() || pending.dispatch_mismatch {
                    pending.cancel_requested = true;
                    pending.cancel_failure = (pending.cancel)().err();
                }
                Ok(TeacherSubmitV3::Submitted(id))
            }
            Ok(Err(error)) => {
                // This return from the original worker means bounded mailbox
                // admission failed; an unwind is handled separately below.
                claim.rejected();
                let original = self.pending.take().expect("original query retained");
                Err(Box::new(TeacherRejectedV3 {
                    error: invalid(error.to_string()),
                    capsule: original.capsule,
                }))
            }
            Err(payload) => {
                // Dropping the unresolved claim records unknown, never retry.
                drop(claim);
                let pending = self.pending.as_mut().expect("original query retained");
                pending.submission_panic = Some(payload);
                pending.cancel_requested = true;
                pending.cancel_failure = (pending.cancel)().err();
                Ok(TeacherSubmitV3::SubmissionUnknown(id))
            }
        }
    }

    /// A cancellation signal is not completion and never releases the capsule.
    pub(crate) fn cancel_pending(&mut self, ticket: &RoleTicketEnvelope) -> Result<bool, String> {
        let Some(pending) = self.pending.as_mut() else {
            return Ok(false);
        };
        if pending.capsule.ticket() != *ticket {
            return Ok(false);
        }
        if !pending.cancel_requested {
            pending.cancel_requested = true;
            pending.cancel_failure = (pending.cancel)().err();
        }
        if let Some(error) = &pending.cancel_failure {
            return Err(error.clone());
        }
        Ok(true)
    }

    /// An unknown submission cannot be converted into a completion by
    /// guessing a dispatch; retain it for the owning worker's recovery path.
    pub(crate) fn take_completed(&mut self) -> Result<Option<TeacherCompletedV3>, String> {
        let Some(pending) = self.pending.as_mut() else {
            return Ok(None);
        };
        if pending.capsule.is_canceled() && !pending.cancel_requested {
            pending.cancel_requested = true;
            pending.cancel_failure = (pending.cancel)().err();
        }
        if pending.dispatch.is_none() {
            return Ok(None);
        }
        if pending.completion_panic.is_some() {
            return Err("V3 teacher completion observation is uncertain".into());
        }
        let mut session = self
            .session
            .lock()
            .map_err(|_| "V3 teacher session lock poisoned".to_string())?;
        let lifecycle = session
            .lifecycle_mut(&self.actor_id)
            .ok_or("V3 teacher lifecycle missing")?;
        if pending.completion.is_none() {
            match catch_unwind(AssertUnwindSafe(|| lifecycle.take_completion())) {
                Ok(Some(completion)) => pending.completion = Some(completion),
                Ok(None) => return Ok(None),
                Err(payload) => {
                    pending.completion_panic = Some(payload);
                    return Err("V3 teacher completion observation panicked".into());
                }
            }
        }
        let validation = catch_unwind(AssertUnwindSafe(|| {
            lifecycle.validate_completion(
                pending.completion.as_ref().expect("completion retained"),
                pending
                    .dispatch
                    .as_ref()
                    .expect("original dispatch retained"),
            )
        }));
        let validation_error = match validation {
            Ok(result) if pending.dispatch_mismatch => Some(format!(
                "V3 teacher dispatch differs from startup process or job: {:?}",
                result.err()
            )),
            Ok(result) => result.err().map(|e| e.to_string()),
            Err(payload) => {
                pending.completion_panic = Some(payload);
                return Err("V3 teacher completion validation panicked".into());
            }
        };
        // Cancellation may race completion validation. Keep the original
        // capsule too: callers must recheck it again at any later target join.
        if pending.capsule.is_canceled() && !pending.cancel_requested {
            pending.cancel_requested = true;
            pending.cancel_failure = (pending.cancel)().err();
        }
        let pending = self.pending.take().expect("original query retained");
        Ok(Some(TeacherCompletedV3 {
            capsule: pending.capsule,
            soft_quantum: pending.soft_quantum,
            actor_id: self.actor_id.clone(),
            dispatch: pending.dispatch.expect("original dispatch retained"),
            completion: pending.completion.expect("original terminal retained"),
            startup: pending.startup,
            cancel_requested: pending.cancel_requested,
            cancel_failure: pending.cancel_failure,
            validation_error,
        }))
    }
}
