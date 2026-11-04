//! Candidate original game owner for V3 Collection.
//! This module is not registered or connected to a product bridge. Its
//! close proof is intentionally unconstructible until an audited bridge
//! proves the original native fence and all root/producer custody.
use super::launch::PreparedPcCollectionInputsV3;
use super::policy_row_pipeline_v3::PolicyRowPipelineOwnerV3;
use rz_uci::pals_adaptive::RoventCollectionRootAdmissionV3;
use std::convert::Infallible;
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::sync::{Arc, Mutex, TryLockError, Weak};

const ABORT_REQUESTED: u8 = 1;
const PUBLICATION_GRANTED: u8 = 2;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PhaseV3 {
    Open,
    Closed,
    Preparing,
    Granted,
    Unknown,
}

struct StateV3<S> {
    phase: PhaseV3,
    supervisor: Option<Weak<Mutex<S>>>,
    pipeline: Option<PolicyRowPipelineOwnerV3>,
    first: Option<RoventCollectionRootAdmissionV3>,
    pending: Vec<RoventCollectionRootAdmissionV3>,
    admitted: Vec<RoventCollectionRootAdmissionV3>,
}

pub(super) struct GameAuthorityV3<S> {
    identity: Arc<()>,
    inputs: Arc<PreparedPcCollectionInputsV3>,
    control: AtomicU8,
    unknown: AtomicBool,
    state: Mutex<StateV3<S>>,
}

/// No constructor exists here; matching identity is necessary but is not
/// sufficient to prove a real bridge closed all admission and producer work.
pub(super) struct GameClosureProofV3 {
    owner: Arc<()>,
    _requires_audited_bridge: Infallible,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum GameStepV3 {
    Ready,
    Existing,
    Busy,
    Closed,
    Aborted,
    Unknown,
    Capacity,
    Foreign,
}

pub(super) enum RootReservationV3<S> {
    Reserved(GameRootReservationV3<S>),
    Existing,
    Rejected(GameStepV3),
}

impl<S> GameAuthorityV3<S> {
    /// Created before the first root. This retains the original inputs, but
    /// alone does not establish a live product/launch or publication proof.
    pub(super) fn new_candidate(inputs: Arc<PreparedPcCollectionInputsV3>) -> Arc<Self> {
        Arc::new(Self {
            identity: Arc::new(()),
            inputs,
            control: AtomicU8::new(0),
            unknown: AtomicBool::new(false),
            state: Mutex::new(StateV3 {
                phase: PhaseV3::Open,
                supervisor: None,
                pipeline: None,
                first: None,
                pending: Vec::new(),
                admitted: Vec::new(),
            }),
        })
    }

    /// Weak identity prevents a supervisor -> bridge -> game -> supervisor
    /// ownership cycle. The launch retains the actual strong supervisor.
    pub(super) fn try_bind_supervisor(&self, supervisor: &Arc<Mutex<S>>) -> GameStepV3 {
        let mut state = match self.lock() {
            Ok(s) => s,
            Err(e) => return e,
        };
        if state.phase != PhaseV3::Open {
            return GameStepV3::Closed;
        }
        let candidate = Arc::downgrade(supervisor);
        match &state.supervisor {
            Some(original) if Weak::ptr_eq(original, &candidate) => GameStepV3::Existing,
            Some(_) => GameStepV3::Foreign,
            None => {
                state.supervisor = Some(candidate);
                GameStepV3::Ready
            }
        }
    }

    /// Called only after the first root's original game pipeline is created.
    pub(super) fn try_bind_pipeline(&self, pipeline: &PolicyRowPipelineOwnerV3) -> GameStepV3 {
        let mut state = match self.lock() {
            Ok(s) => s,
            Err(e) => return e,
        };
        if state.phase != PhaseV3::Open {
            return GameStepV3::Closed;
        }
        match &state.pipeline {
            Some(original) if original.same_owner(pipeline) => GameStepV3::Existing,
            Some(_) => GameStepV3::Foreign,
            None => {
                state.pipeline = Some(pipeline.clone());
                GameStepV3::Ready
            }
        }
    }

    pub(super) fn matches_inputs(&self, inputs: &Arc<PreparedPcCollectionInputsV3>) -> bool {
        Arc::ptr_eq(&self.inputs, inputs)
    }

    fn lock(&self) -> Result<std::sync::MutexGuard<'_, StateV3<S>>, GameStepV3> {
        if self.unknown.load(Ordering::SeqCst) {
            return Err(GameStepV3::Unknown);
        }
        match self.state.try_lock() {
            Ok(state) => Ok(state),
            Err(TryLockError::WouldBlock) => Err(GameStepV3::Busy),
            Err(TryLockError::Poisoned(_)) => {
                self.unknown.store(true, Ordering::SeqCst);
                Err(GameStepV3::Unknown)
            }
        }
    }

    /// Reservation is retained independently of the state lock. The bridge
    /// keeps this guard while registering the exact root slot and commits it
    /// once that slot exists; queue Full may then retry that same slot.
    pub(super) fn try_reserve_root(
        self: &Arc<Self>,
        root: &RoventCollectionRootAdmissionV3,
    ) -> RootReservationV3<S> {
        if (self.control.load(Ordering::SeqCst) & ABORT_REQUESTED != 0) {
            return RootReservationV3::Rejected(GameStepV3::Aborted);
        }
        let mut state = match self.lock() {
            Ok(s) => s,
            Err(e) => return RootReservationV3::Rejected(e),
        };
        if (self.control.load(Ordering::SeqCst) & ABORT_REQUESTED != 0) {
            return RootReservationV3::Rejected(GameStepV3::Aborted);
        }
        if state.phase != PhaseV3::Open {
            return RootReservationV3::Rejected(GameStepV3::Closed);
        }
        if let Some(first) = &state.first {
            let old = first.root_admission().authority();
            let new = root.root_admission().authority();
            if !first
                .native_origin()
                .same_original_origin(root.native_origin())
                || old.epoch != new.epoch
                || old.implementation != new.implementation
                || old.game != new.game
            {
                return RootReservationV3::Rejected(GameStepV3::Foreign);
            }
        }
        if state
            .admitted
            .iter()
            .any(|a| a.same_original_admission(root))
        {
            return RootReservationV3::Existing;
        }
        if state
            .pending
            .iter()
            .any(|a| a.same_original_admission(root))
        {
            return RootReservationV3::Rejected(GameStepV3::Busy);
        }
        if state
            .admitted
            .iter()
            .chain(&state.pending)
            .any(|a| a.root_admission().authority() == root.root_admission().authority())
        {
            return RootReservationV3::Rejected(GameStepV3::Foreign);
        }
        let reserved_roots = state.pending.len().saturating_add(1);
        if state.admitted.len().saturating_add(state.pending.len())
            >= usize::from(self.inputs.plan().maximum_roots)
            || state.pending.try_reserve(1).is_err()
            || state.admitted.try_reserve(reserved_roots).is_err()
        {
            return RootReservationV3::Rejected(GameStepV3::Capacity);
        }
        if state.first.is_none() {
            state.first = Some(root.clone());
        }
        state.pending.push(root.clone());
        drop(state);
        RootReservationV3::Reserved(GameRootReservationV3 {
            owner: Arc::clone(self),
            root: root.clone(),
            resolved: false,
        })
    }

    /// Sticky request only, not physical termination or cleanup evidence.
    pub(super) fn request_abort(&self) {
        self.control.fetch_or(ABORT_REQUESTED, Ordering::SeqCst);
    }

    pub(super) fn try_close(&self, proof: &GameClosureProofV3) -> GameStepV3 {
        if !Arc::ptr_eq(&self.identity, &proof.owner) {
            return GameStepV3::Foreign;
        }
        if (self.control.load(Ordering::SeqCst) & ABORT_REQUESTED != 0) {
            return GameStepV3::Aborted;
        }
        let mut state = match self.lock() {
            Ok(s) => s,
            Err(e) => return e,
        };
        if (self.control.load(Ordering::SeqCst) & ABORT_REQUESTED != 0) {
            return GameStepV3::Aborted;
        }
        if !state.pending.is_empty() {
            return GameStepV3::Busy;
        }
        if state.supervisor.as_ref().and_then(Weak::upgrade).is_none() || state.pipeline.is_none() {
            return GameStepV3::Foreign;
        }
        match state.phase {
            PhaseV3::Open => {
                state.phase = PhaseV3::Closed;
                GameStepV3::Ready
            }
            PhaseV3::Closed => GameStepV3::Existing,
            PhaseV3::Unknown => GameStepV3::Unknown,
            _ => GameStepV3::Closed,
        }
    }

    /// This guard cannot be created without a bridge-issued close proof. The
    /// future final atomic publication hook must use this exact game owner.
    pub(super) fn try_begin_publication(
        self: &Arc<Self>,
        proof: &GameClosureProofV3,
    ) -> Result<GamePublishGuardV3<S>, GameStepV3> {
        if !Arc::ptr_eq(&self.identity, &proof.owner) {
            return Err(GameStepV3::Foreign);
        }
        if (self.control.load(Ordering::SeqCst) & ABORT_REQUESTED != 0) {
            return Err(GameStepV3::Aborted);
        }
        let mut state = self.lock()?;
        if (self.control.load(Ordering::SeqCst) & ABORT_REQUESTED != 0) {
            return Err(GameStepV3::Aborted);
        }
        if state.phase != PhaseV3::Closed || !state.pending.is_empty() {
            return Err(GameStepV3::Closed);
        }
        state.phase = PhaseV3::Preparing;
        drop(state);
        Ok(GamePublishGuardV3 {
            owner: Arc::clone(self),
            resolved: false,
        })
    }
}

pub(super) struct GameRootReservationV3<S> {
    owner: Arc<GameAuthorityV3<S>>,
    root: RoventCollectionRootAdmissionV3,
    resolved: bool,
}
impl<S> GameRootReservationV3<S> {
    /// Call only after the exact root slot exists. If abort raced, the caller
    /// must retain the slot and propagate cancellation; it is never erased.
    pub(super) fn commit_registered(&mut self) -> GameStepV3 {
        let mut state = match self.owner.lock() {
            Ok(s) => s,
            Err(e) => return e,
        };
        let Some(index) = state
            .pending
            .iter()
            .position(|a| a.same_original_admission(&self.root))
        else {
            return GameStepV3::Unknown;
        };
        let root = state.pending.remove(index);
        state.admitted.push(root);
        self.resolved = true;
        if (self.owner.control.load(Ordering::SeqCst) & ABORT_REQUESTED != 0) {
            GameStepV3::Aborted
        } else {
            GameStepV3::Ready
        }
    }
    /// Only for a proven failure before any bridge slot/queue mutation.
    pub(super) fn cancel_unstarted(&mut self) -> GameStepV3 {
        let mut state = match self.owner.lock() {
            Ok(s) => s,
            Err(e) => return e,
        };
        let Some(index) = state
            .pending
            .iter()
            .position(|a| a.same_original_admission(&self.root))
        else {
            return GameStepV3::Unknown;
        };
        state.pending.remove(index);
        self.resolved = true;
        GameStepV3::Ready
    }
}
impl<S> Drop for GameRootReservationV3<S> {
    fn drop(&mut self) {
        if !self.resolved {
            self.owner.unknown.store(true, Ordering::SeqCst);
        }
    }
}

pub(super) struct GamePublishGuardV3<S> {
    owner: Arc<GameAuthorityV3<S>>,
    resolved: bool,
}
impl<S> GamePublishGuardV3<S> {
    /// The future writer must call this as its authorization immediately
    /// before its real publication commit; it is not durable-write evidence.
    pub(super) fn try_grant(&mut self) -> GameStepV3 {
        let mut state = match self.owner.lock() {
            Ok(s) => s,
            Err(e) => return e,
        };
        if state.phase != PhaseV3::Preparing {
            return GameStepV3::Unknown;
        }
        match self.owner.control.compare_exchange(
            0,
            PUBLICATION_GRANTED,
            Ordering::SeqCst,
            Ordering::SeqCst,
        ) {
            Ok(_) => {}
            Err(actual) if actual & ABORT_REQUESTED != 0 => return GameStepV3::Aborted,
            Err(_) => return GameStepV3::Unknown,
        }
        state.phase = PhaseV3::Granted;
        self.resolved = true;
        GameStepV3::Ready
    }
}
impl<S> Drop for GamePublishGuardV3<S> {
    fn drop(&mut self) {
        if !self.resolved {
            self.owner.unknown.store(true, Ordering::SeqCst);
            if let Ok(mut state) = self.owner.state.try_lock() {
                state.phase = PhaseV3::Unknown;
            }
        }
    }
}
