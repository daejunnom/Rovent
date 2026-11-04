//! One-root bounded teacher pump. It retains raw evidence, not corpus targets.
//!
//! The root custodian must retain the same `Arc<Mutex<TeacherPumpV3>>` outside
//! the worker and preserve it on panic, poisoning, or an unknown shutdown.
//! Sender closure is distinct from route revocation and callback fencing.
use super::super::producer_v3::SubmittedTeacherQueryV3;
use super::super::invalid;
use super::v3_dispatch::{
    TeacherCompletedV3, TeacherDispatchOwnerV3, TeacherRejectedV3, TeacherSubmitV3,
};
use super::v3_session::RoventCollectionTeacherSessionV3;
use rz_contracts::pals_adaptive::RoleTicketEnvelope;
use rz_search::external_cpu::ExternalGoConstraint;
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, mpsc};
use std::time::Duration;

/// Successful queue admission does not mean a teacher job was dispatched.
pub(crate) enum TeacherPumpStepV3 {
    Idle,
    InputHeld,
    StartupPending,
    Submitted,
    SubmissionUnknown,
    ObservationUnknown,
    Completed,
    Rejected,
    OutputBackpressure,
    InputDisconnected,
    Closed,
}

pub(crate) enum TeacherPumpCancelV3 {
    Held,
    Signaled,
    /// The route must still revoke the original ticket; it may be in flight
    /// or in the sender's queue, which is not inspectable by the receiver.
    NotObserved,
}

pub(crate) struct TeacherPumpSetupErrorV3 {
    pub error: String,
    pub receiver: mpsc::Receiver<Arc<SubmittedTeacherQueryV3>>,
    pub session: Arc<Mutex<RoventCollectionTeacherSessionV3>>,
    pub dispatchers: BTreeMap<String, TeacherDispatchOwnerV3>,
}

pub(crate) struct TeacherPumpV3 {
    // This is the same owner used by the dispatcher and retained by the root
    // custodian; it also retains scope and worker recovery keys on panic.
    session: Arc<Mutex<RoventCollectionTeacherSessionV3>>,
    dispatchers: BTreeMap<String, TeacherDispatchOwnerV3>,
    observation_errors: BTreeMap<String, String>,
    receiver: mpsc::Receiver<Arc<SubmittedTeacherQueryV3>>,
    held: Option<Arc<SubmittedTeacherQueryV3>>,
    completed: Option<TeacherCompletedV3>,
    rejected: Option<Box<TeacherRejectedV3>>,
    input_disconnected: bool,
    closed: bool,
    constraint: ExternalGoConstraint,
    soft_quantum: Duration,
}

impl TeacherPumpV3 {
    /// Consume the original root route's bounded receiver. The route owner
    /// retains its sender and its separate callback-fence responsibility.
    /// The original session's registered actors each own one dispatcher.
    pub(crate) fn new(
        session: Arc<Mutex<RoventCollectionTeacherSessionV3>>,
        receiver: mpsc::Receiver<Arc<SubmittedTeacherQueryV3>>,
        constraint: ExternalGoConstraint,
        soft_quantum: Duration,
    ) -> Result<Arc<Mutex<Self>>, TeacherPumpSetupErrorV3> {
        let actors: Vec<String> = match session.lock() {
            Ok(owner) => owner.registration().teachers.keys().cloned().collect(),
            Err(_) => return Err(TeacherPumpSetupErrorV3 {
                error: "V3 teacher session lock poisoned".into(), receiver,
                session: Arc::clone(&session), dispatchers: BTreeMap::new(),
            }),
        };
        let mut dispatchers = BTreeMap::new();
        for actor_id in actors {
            match TeacherDispatchOwnerV3::new(Arc::clone(&session), actor_id.clone()) {
                Ok(dispatcher) => { dispatchers.insert(actor_id, dispatcher); }
                Err(error) => return Err(TeacherPumpSetupErrorV3 {
                    error, receiver, session, dispatchers,
                }),
            }
        }
        Ok(Arc::new(Mutex::new(Self {
            session, dispatchers, observation_errors: BTreeMap::new(),
            receiver, held: None, completed: None, rejected: None,
            input_disconnected: false, closed: false, constraint, soft_quantum,
        })))
    }

    pub(crate) fn session(&self) -> &Arc<Mutex<RoventCollectionTeacherSessionV3>> {
        &self.session
    }

    /// Closing admission does not assert that callbacks or senders stopped.
    /// Continue draining the existing dispatcher and separately fence the
    /// native route before deciding how to retire queued original capsules.
    pub(crate) fn close_admission(&mut self) { self.closed = true; }

    pub(crate) fn input_disconnected(&self) -> bool { self.input_disconnected }
    pub(crate) fn has_pending(&self) -> bool {
        self.dispatchers.values().any(TeacherDispatchOwnerV3::has_pending)
    }

    pub(crate) fn observation_errors(&self) -> &BTreeMap<String, String> {
        &self.observation_errors
    }

    /// The native observer must separately revoke this exact ticket. A match
    /// in the held slot has not been issued; retain it as a rejected input.
    pub(crate) fn cancel_ticket(
        &mut self,
        ticket: &RoleTicketEnvelope,
    ) -> Result<TeacherPumpCancelV3, String> {
        let mut signaled = false;
        let mut first_error = None;
        for dispatcher in self.dispatchers.values_mut() {
            if dispatcher.pending_ticket().as_ref() == Some(ticket) {
                signaled = true;
                if let Err(error) = dispatcher.cancel_pending(ticket) {
                    if first_error.is_none() { first_error = Some(error); }
                }
            }
        }
        if let Some(error) = first_error { return Err(error); }
        if signaled { return Ok(TeacherPumpCancelV3::Signaled); }
        if self.held.as_ref().is_some_and(|query| query.ticket() == *ticket) {
            if self.rejected.is_some() {
                return Err("V3 teacher rejection slot full; held query retained".into());
            }
            let capsule = self.held.take().expect("held query observed");
            self.rejected = Some(Box::new(TeacherRejectedV3 {
                error: invalid("V3 teacher query canceled before dispatch"), capsule,
            }));
            return Ok(TeacherPumpCancelV3::Held);
        }
        Ok(TeacherPumpCancelV3::NotObserved)
    }

    /// At most one receiver item or one physical completion per call. The
    /// manager itself never calls this; its root worker owns the pump.
    pub(crate) fn try_pump(&mut self) -> Result<TeacherPumpStepV3, String> {
        if self.completed.is_some() || self.rejected.is_some() {
            return Ok(TeacherPumpStepV3::OutputBackpressure);
        }
        let mut unknown = false;
        let mut submission_unknown = false;
        for (actor, dispatcher) in &mut self.dispatchers {
            if !dispatcher.has_pending() { continue; }
            submission_unknown |= dispatcher.submission_unknown();
            match dispatcher.take_completed() {
                Ok(Some(completed)) => {
                    self.observation_errors.remove(actor);
                    self.completed = Some(completed);
                    return Ok(TeacherPumpStepV3::Completed);
                }
                Ok(None) => {}
                Err(error) => {
                    self.observation_errors.insert(actor.clone(), error);
                    unknown = true;
                }
            }
        }
        if self.closed { return Ok(if unknown { TeacherPumpStepV3::ObservationUnknown }
            else if submission_unknown { TeacherPumpStepV3::SubmissionUnknown }
            else { TeacherPumpStepV3::Closed }); }
        if self.held.is_none() {
            match self.receiver.try_recv() {
                Ok(capsule) => {
                    self.held = Some(capsule);
                    return Ok(TeacherPumpStepV3::InputHeld);
                }
                Err(mpsc::TryRecvError::Empty) => return Ok(if unknown {
                    TeacherPumpStepV3::ObservationUnknown
                } else if submission_unknown { TeacherPumpStepV3::SubmissionUnknown }
                else { TeacherPumpStepV3::Idle }),
                Err(mpsc::TryRecvError::Disconnected) => {
                    self.input_disconnected = true;
                    return Ok(TeacherPumpStepV3::InputDisconnected);
                }
            }
        }
        // Keep the same original immutable owner in this pump across submit,
        // including an unwind before dispatcher custody is fully installed.
        let capsule = Arc::clone(self.held.as_ref().expect("held query observed"));
        let actor_id = capsule.input_seal().teacher_actor_id;
        let Some(dispatcher) = actor_id.as_deref().and_then(|actor| self.dispatchers.get_mut(actor)) else {
            let capsule = self.held.take().expect("held query retained");
            self.rejected = Some(Box::new(TeacherRejectedV3 {
                error: invalid("V3 teacher query actor is absent from root roster"), capsule,
            }));
            return Ok(TeacherPumpStepV3::Rejected);
        };
        if dispatcher.has_pending() { return Ok(TeacherPumpStepV3::Idle); }
        match dispatcher.submit(capsule, self.constraint, self.soft_quantum) {
            Ok(TeacherSubmitV3::Submitted(_)) => {
                self.held.take();
                Ok(TeacherPumpStepV3::Submitted)
            }
            Ok(TeacherSubmitV3::StartupPending(capsule)) => {
                self.held = Some(capsule);
                Ok(TeacherPumpStepV3::StartupPending)
            }
            Ok(TeacherSubmitV3::SubmissionUnknown(_)) => {
                self.held.take();
                Ok(TeacherPumpStepV3::SubmissionUnknown)
            }
            Err(rejected) => {
                self.held.take();
                self.rejected = Some(rejected);
                Ok(TeacherPumpStepV3::Rejected)
            }
        }
    }

    /// Raw terminal plus original capsule; target construction is separate.
    pub(crate) fn take_completed(&mut self) -> Option<TeacherCompletedV3> {
        self.completed.take()
    }
    pub(crate) fn take_rejected(&mut self) -> Option<Box<TeacherRejectedV3>> {
        self.rejected.take()
    }
    /// Transfer one held or queued capsule. This is custody only: it does not
    /// attest that senders, native callbacks, or a raced dispatch have stopped.
    pub(crate) fn take_held_or_queued(&mut self) -> Option<Arc<SubmittedTeacherQueryV3>> {
        if !self.closed { return None; }
        if let Some(held) = &self.held {
            if self.dispatchers.values().any(|dispatcher|
                dispatcher.pending_ticket() == Some(held.ticket())) { return None; }
        }
        if self.held.is_some() { return self.held.take(); }
        match self.receiver.try_recv() {
            Ok(capsule) => Some(capsule),
            Err(mpsc::TryRecvError::Disconnected) => { self.input_disconnected = true; None }
            Err(mpsc::TryRecvError::Empty) => None,
        }
    }
}
