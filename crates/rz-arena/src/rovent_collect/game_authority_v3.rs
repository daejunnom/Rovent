//! Candidate original game owner for V3 Collection.
//! This candidate has partial bridge bindings, not complete product closure.
//! Its close proof is intentionally unconstructible until an audited bridge
//! proves the original native fence and all root/producer custody.
use super::launch::{ManagedOutputOwnerV3, PreparedPcCollectionInputsV3};
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
    AdmissionSealed,
    Closed,
    Preparing,
    Granted,
    Unknown,
}

struct StateV3<S> {
    phase: PhaseV3,
    supervisor: Option<Weak<Mutex<S>>>,
    pipeline: Option<PolicyRowPipelineOwnerV3>,
    output: Option<Weak<ManagedOutputOwnerV3>>,
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

/// A distinct, earlier proof: only the original native launch/bridge may
/// attest no further root admission. It does not attest root drains, teacher
/// joins, output completion, or any physical device release. A real bridge
/// constructor is intentionally not implemented in this isolated candidate.
pub(super) struct GameAdmissionSealProofV3 {
    owner: Arc<()>,
    _requires_original_native_fence: Infallible,
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
                output: None,
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
        if !matches!(state.phase, PhaseV3::Open | PhaseV3::AdmissionSealed) {
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
        if !matches!(state.phase, PhaseV3::Open | PhaseV3::AdmissionSealed) {
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

    /// Pin the exact output owner separately from inputs: two output owners
    /// may legitimately reference the same immutable input plan.
    pub(super) fn try_bind_output(&self, output: &Arc<ManagedOutputOwnerV3>) -> GameStepV3 {
        if !output.matches_inputs(&self.inputs) {
            return GameStepV3::Foreign;
        }
        let mut state = match self.lock() {
            Ok(s) => s,
            Err(e) => return e,
        };
        if !matches!(state.phase, PhaseV3::Open | PhaseV3::AdmissionSealed) {
            return GameStepV3::Closed;
        }
        let candidate = Arc::downgrade(output);
        match &state.output {
            Some(original) if Weak::ptr_eq(original, &candidate) => GameStepV3::Existing,
            Some(_) => GameStepV3::Foreign,
            None => {
                state.output = Some(candidate);
                GameStepV3::Ready
            }
        }
    }

    pub(super) fn matches_inputs(&self, inputs: &Arc<PreparedPcCollectionInputsV3>) -> bool {
        Arc::ptr_eq(&self.inputs, inputs)
    }

    pub(super) fn same_owner(self: &Arc<Self>, other: &Arc<Self>) -> bool {
        Arc::ptr_eq(self, other)
    }

    /// A request observation only; it does not prove that any root has
    /// observed cancellation or that a worker has physically stopped.
    pub(super) fn abort_requested(&self) -> bool {
        self.control.load(Ordering::SeqCst) & ABORT_REQUESTED != 0
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
            abort_receipt: None,
        })
    }

    /// Sticky request only, not physical termination or cleanup evidence.
    pub(super) fn request_abort(&self) {
        self.control.fetch_or(ABORT_REQUESTED, Ordering::SeqCst);
    }

    /// Only admission is sealed here. Existing reservations and original
    /// root work remain held and must be separately settled and observed.
    pub(super) fn try_seal_admission(&self, proof: &GameAdmissionSealProofV3) -> GameStepV3 {
        if !Arc::ptr_eq(&self.identity, &proof.owner) {
            return GameStepV3::Foreign;
        }
        self.try_seal_validated_admission()
    }

    fn try_seal_validated_admission(&self) -> GameStepV3 {
        let mut state = match self.lock() {
            Ok(s) => s,
            Err(e) => return e,
        };
        match state.phase {
            PhaseV3::Open => {
                state.phase = PhaseV3::AdmissionSealed;
                GameStepV3::Ready
            }
            PhaseV3::AdmissionSealed => GameStepV3::Existing,
            PhaseV3::Unknown => GameStepV3::Unknown,
            _ => GameStepV3::Closed,
        }
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
        if state.supervisor.as_ref().and_then(Weak::upgrade).is_none()
            || state.pipeline.is_none()
            || state.output.as_ref().and_then(Weak::upgrade).is_none()
        {
            return GameStepV3::Foreign;
        }
        match state.phase {
            PhaseV3::AdmissionSealed => {
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

impl GameAuthorityV3<super::pc_worker_v3::PcWorkerSupervisorV3> {
    /// Only the original P-C bridge can create this nonclone native fence
    /// witness. It is admission evidence only, not worker or output proof.
    pub(super) fn try_seal_pc(
        self: &Arc<Self>,
        proof: &super::pc_hooks_lock_v3::PcGameAdmissionSealV3,
    ) -> GameStepV3 {
        if !proof.matches_game(self) {
            return GameStepV3::Foreign;
        }
        self.try_seal_validated_admission()
    }
}

pub(super) struct GameRootReservationV3<S> {
    owner: Arc<GameAuthorityV3<S>>,
    root: RoventCollectionRootAdmissionV3,
    resolved: bool,
    abort_receipt: Option<GameRootAbortReceiptV3<S>>,
}

/// The original gate observed abort at this exact registration commit.
/// This does not prove that worker, teacher, or queue work is absent.
pub(super) struct GameRootAbortReceiptV3<S> {
    owner: Arc<GameAuthorityV3<S>>,
    root: RoventCollectionRootAdmissionV3,
}
impl<S> GameRootAbortReceiptV3<S> {
    pub(super) fn matches_game(&self, game: &Arc<GameAuthorityV3<S>>) -> bool {
        Arc::ptr_eq(&self.owner, game)
    }
    pub(super) fn matches_original(&self, root: &RoventCollectionRootAdmissionV3) -> bool {
        self.root.same_original_admission(root)
    }
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
            self.abort_receipt = Some(GameRootAbortReceiptV3 {
                owner: Arc::clone(&self.owner),
                root: self.root.clone(),
            });
            GameStepV3::Aborted
        } else {
            GameStepV3::Ready
        }
    }

    /// Transfer into a pre-owned empty slot without exposing an intermediate
    /// receipt whose owner could be lost during a later fallible operation.
    pub(super) fn transfer_abort_receipt(
        &mut self,
        destination: &mut Option<GameRootAbortReceiptV3<S>>,
    ) -> bool {
        if destination.is_some() || self.abort_receipt.is_none() {
            return false;
        }
        *destination = self.abort_receipt.take();
        true
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
        if !self.resolved || self.abort_receipt.is_some() {
            self.owner.unknown.store(true, Ordering::SeqCst);
        }
    }
}

pub(crate) struct GamePublishGuardV3<S> {
    owner: Arc<GameAuthorityV3<S>>,
    resolved: bool,
}
impl<S> GamePublishGuardV3<S> {
    /// Match the bound owners without taking the pipeline mutex: callers may
    /// already hold that mutex while passing its original finished writer.
    /// The pipeline must itself verify its retained output and writer slots.
    pub(super) fn try_matches_output_pipeline(
        &self,
        output: &Arc<ManagedOutputOwnerV3>,
        pipeline: &PolicyRowPipelineOwnerV3,
    ) -> Result<Option<bool>, &'static str> {
        let state = match self.owner.lock() {
            Ok(state) => state,
            Err(GameStepV3::Busy) => return Ok(None),
            Err(_) => return Err("V3 game authority state unknown"),
        };
        Ok(Some(
            state.phase == PhaseV3::Preparing
                && output.matches_inputs(&self.owner.inputs)
                && state
                    .output
                    .as_ref()
                    .is_some_and(|original| Weak::ptr_eq(original, &Arc::downgrade(output)))
                && state
                    .pipeline
                    .as_ref()
                    .is_some_and(|original| original.same_owner(pipeline)),
        ))
    }

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
            Err(actual) if actual & ABORT_REQUESTED != 0 => {
                // Authorization is known not to have been granted. Any
                // earlier output mutation remains the output owner's own
                // separate custody/uncertainty, not a game-gate unknown.
                state.phase = PhaseV3::Closed;
                self.resolved = true;
                return GameStepV3::Aborted;
            }
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
