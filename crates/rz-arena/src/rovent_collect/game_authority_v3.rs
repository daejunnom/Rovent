//! Candidate original game owner for V3 Collection.
//! This candidate has partial bridge bindings, not complete product closure.
//! Its close proof is intentionally unconstructible until an audited bridge
//! proves the original native fence and all root/producer custody.
use super::launch::{
    ManagedOutputOwnerV3, ManagedPublicationUnstartedEvidenceV3, PreparedPcCollectionInputsV3,
};
use super::policy_row_pipeline_v3::PolicyRowPipelineOwnerV3;
use rz_uci::pals_adaptive::RoventCollectionRootAdmissionV3;
use std::convert::Infallible;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{SyncSender, TrySendError};
use std::sync::{Arc, Mutex, TryLockError, Weak};

const ABORT_REQUESTED: u64 = 1;
const PUBLICATION_GRANTED: u64 = 2;
const NO_DATA_FINISHED: u64 = 4;
const ADMISSION_SEALED: u64 = 8;
const PUBLICATION_NOT_STARTED: u64 = 16;
const DISPATCH_CLAIM: u64 = 32;
const CONTROL_FLAGS: u64 = DISPATCH_CLAIM - 1;
const EFFECT_CLAIM: u64 = 1u64 << 32;
const DISPATCH_MASK: u64 = (EFFECT_CLAIM - 1) & !CONTROL_FLAGS;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PhaseV3 {
    Open,
    AdmissionSealed,
    Closed,
    Preparing,
    Granted,
    NoData,
    KnownNotWritten,
    Unknown,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum GameEffectKindV3 {
    TeacherScope,
    TeacherActor,
    NativeWorker,
    RouteInstall,
}

struct GameEffectRecordV3 {
    root: RoventCollectionRootAdmissionV3,
    kind: GameEffectKindV3,
    stage: Arc<()>,
    actor: Option<Arc<()>>,
    active: bool,
}

/// Placeholder for a future original stage/session witness. This has no
/// constructor until each real acquisition owner supplies its typed proof.
pub(super) struct GameEffectSourceV3<S> {
    owner: Arc<GameAuthorityV3<S>>,
    root: RoventCollectionRootAdmissionV3,
    kind: GameEffectKindV3,
    stage: Arc<()>,
    actor: Option<Arc<()>>,
    _requires_original_stage: Infallible,
}

/// Not constructible until the actual start primitive preserves its original
/// result/owner. It never represents worker join or physical completion.
pub(super) struct GameEffectOutcomeV3<S> {
    owner: Arc<GameAuthorityV3<S>>,
    root: RoventCollectionRootAdmissionV3,
    kind: GameEffectKindV3,
    stage: Arc<()>,
    actor: Option<Arc<()>>,
    _requires_original_result: Infallible,
}

fn same_effect_actor(a: &Option<Arc<()>>, b: &Option<Arc<()>>) -> bool {
    match (a, b) {
        (Some(a), Some(b)) => Arc::ptr_eq(a, b),
        (None, None) => true,
        _ => false,
    }
}

struct StateV3<S> {
    phase: PhaseV3,
    supervisor: Option<Weak<Mutex<S>>>,
    pipeline: Option<PolicyRowPipelineOwnerV3>,
    output: Option<Weak<ManagedOutputOwnerV3>>,
    dispatch_channel: Option<Arc<()>>,
    first: Option<RoventCollectionRootAdmissionV3>,
    pending: Vec<RoventCollectionRootAdmissionV3>,
    admitted: Vec<RoventCollectionRootAdmissionV3>,
    dispatching: Vec<RoventCollectionRootAdmissionV3>,
    dispatched: Vec<RoventCollectionRootAdmissionV3>,
    effects: Vec<GameEffectRecordV3>,
}

pub(super) struct GameAuthorityV3<S> {
    identity: Arc<()>,
    inputs: Arc<PreparedPcCollectionInputsV3>,
    control: AtomicU64,
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
            control: AtomicU64::new(0),
            unknown: AtomicBool::new(false),
            state: Mutex::new(StateV3 {
                phase: PhaseV3::Open,
                supervisor: None,
                pipeline: None,
                output: None,
                dispatch_channel: None,
                first: None,
                pending: Vec::new(),
                admitted: Vec::new(),
                dispatching: Vec::new(),
                dispatched: Vec::new(),
                effects: Vec::new(),
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

    /// The bridge moves its original sender into this channel wrapper at
    /// creation and binds its identity once; a different sender cannot use
    /// an existing game's dispatch claim.
    pub(super) fn try_bind_dispatch_channel<T>(
        &self,
        channel: &GameDispatchChannelV3<T>,
    ) -> GameStepV3 {
        let mut state = match self.lock() {
            Ok(s) => s,
            Err(e) => return e,
        };
        if state.phase != PhaseV3::Open {
            return GameStepV3::Closed;
        }
        match &state.dispatch_channel {
            Some(original) if Arc::ptr_eq(original, &channel.identity) => GameStepV3::Existing,
            Some(_) => GameStepV3::Foreign,
            None => {
                state.dispatch_channel = Some(Arc::clone(&channel.identity));
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
        if self.control.load(Ordering::SeqCst) & ABORT_REQUESTED != 0 {
            return RootReservationV3::Rejected(GameStepV3::Aborted);
        }
        let mut state = match self.lock() {
            Ok(s) => s,
            Err(e) => return RootReservationV3::Rejected(e),
        };
        if self.control.load(Ordering::SeqCst) & ABORT_REQUESTED != 0 {
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
            committed_aborted: None,
            abort_receipt: None,
        })
    }

    /// Sticky request only, not physical termination or cleanup evidence.
    pub(super) fn request_abort(&self) {
        self.control.fetch_or(ABORT_REQUESTED, Ordering::SeqCst);
    }

    /// Only observes that no root queue send claim remains after abort.
    /// Already-sent roots, actor starts, workers and device work still need
    /// their own cancellation, drain and physical completion evidence.
    pub(super) fn try_observe_abort_dispatch_quiescent(&self) -> GameStepV3 {
        if !self.abort_requested() {
            return GameStepV3::Busy;
        }
        let state = match self.lock() {
            Ok(state) => state,
            Err(error) => return error,
        };
        let control = self.control.load(Ordering::SeqCst);
        if control & ABORT_REQUESTED == 0 {
            return GameStepV3::Unknown;
        }
        if !state.dispatching.is_empty() || control & DISPATCH_MASK != 0 {
            return GameStepV3::Busy;
        }
        GameStepV3::Ready
    }

    /// This observes only in-flight acquisition claims. Started actors and
    /// workers still need independent cancellation, joins and physical proof.
    pub(super) fn try_observe_abort_effects_quiescent(&self) -> GameStepV3 {
        if !self.abort_requested() {
            return GameStepV3::Busy;
        }
        let state = match self.lock() {
            Ok(s) => s,
            Err(e) => return e,
        };
        if state.effects.iter().any(|effect| effect.active)
            || self.control.load(Ordering::SeqCst) & !(EFFECT_CLAIM - 1) != 0
        {
            return GameStepV3::Busy;
        }
        GameStepV3::Ready
    }

    /// Candidate-only until each stage owner can issue an original source and
    /// actual outcome witness. No generic bool can resolve this claim.
    pub(super) fn try_claim_effect(
        self: &Arc<Self>,
        source: &GameEffectSourceV3<S>,
        destination: &mut Option<GameEffectClaimV3<S>>,
    ) -> GameStepV3 {
        if destination.is_some() {
            return GameStepV3::Busy;
        }
        if !Arc::ptr_eq(self, &source.owner) {
            return GameStepV3::Foreign;
        }
        if (source.kind == GameEffectKindV3::TeacherActor) != source.actor.is_some() {
            return GameStepV3::Foreign;
        }
        let mut state = match self.lock() {
            Ok(s) => s,
            Err(e) => return e,
        };
        if state.phase != PhaseV3::Open {
            return GameStepV3::Closed;
        }
        if !state
            .admitted
            .iter()
            .any(|root| root.same_original_admission(&source.root))
        {
            return GameStepV3::Foreign;
        }
        if state.effects.iter().any(|effect| {
            effect.root.same_original_admission(&source.root)
                && effect.kind == source.kind
                && Arc::ptr_eq(&effect.stage, &source.stage)
                && same_effect_actor(&effect.actor, &source.actor)
        }) {
            return GameStepV3::Existing;
        }
        let limit = usize::from(self.inputs.plan().maximum_roots)
            .checked_mul(self.inputs.plan().teacher_slots.len().saturating_add(3));
        if limit.is_none_or(|limit| state.effects.len() >= limit)
            || state.effects.try_reserve(1).is_err()
        {
            return GameStepV3::Capacity;
        }
        let retained_root = source.root.clone();
        let state_root = source.root.clone();
        let stage = Arc::clone(&source.stage);
        let retained_stage = Arc::clone(&source.stage);
        let actor = source.actor.clone();
        let retained_actor = source.actor.clone();
        let owner = Arc::clone(self);
        let mut current = self.control.load(Ordering::SeqCst);
        loop {
            if current & ABORT_REQUESTED != 0 {
                return GameStepV3::Aborted;
            }
            if current
                & (ADMISSION_SEALED
                    | PUBLICATION_GRANTED
                    | NO_DATA_FINISHED
                    | PUBLICATION_NOT_STARTED)
                != 0
            {
                return GameStepV3::Closed;
            }
            let Some(next) = current.checked_add(EFFECT_CLAIM) else {
                return GameStepV3::Capacity;
            };
            match self
                .control
                .compare_exchange(current, next, Ordering::SeqCst, Ordering::SeqCst)
            {
                Ok(_) => break,
                Err(observed) => current = observed,
            }
        }
        state.effects.push(GameEffectRecordV3 {
            root: state_root,
            kind: source.kind,
            stage,
            actor,
            active: true,
        });
        *destination = Some(GameEffectClaimV3 {
            owner,
            root: retained_root,
            kind: source.kind,
            stage: retained_stage,
            actor: retained_actor,
            resolved: false,
        });
        GameStepV3::Ready
    }

    /// Claims queue dispatch against the same atomic abort word. The caller
    /// pre-retains `destination` in the root slot; no mutex stays held while
    /// the actual bounded channel send executes.
    pub(super) fn try_claim_dispatch<T: GameDispatchEnvelopeV3>(
        self: &Arc<Self>,
        root: &RoventCollectionRootAdmissionV3,
        envelope: &T,
        destination: &mut Option<GameDispatchClaimV3<S>>,
    ) -> GameStepV3 {
        if destination.is_some() {
            return GameStepV3::Busy;
        }
        let mut state = match self.lock() {
            Ok(s) => s,
            Err(e) => return e,
        };
        if state.phase != PhaseV3::Open {
            return GameStepV3::Closed;
        }
        if state.dispatch_channel.is_none() {
            return GameStepV3::Foreign;
        }
        let identity = envelope.original_identity();
        if !identity.matches_root(root) {
            return GameStepV3::Foreign;
        }
        if !state
            .admitted
            .iter()
            .any(|a| a.same_original_admission(root))
        {
            return GameStepV3::Foreign;
        }
        if state
            .dispatched
            .iter()
            .any(|a| a.same_original_admission(root))
        {
            return GameStepV3::Existing;
        }
        if state
            .dispatching
            .iter()
            .any(|a| a.same_original_admission(root))
        {
            return GameStepV3::Busy;
        }
        let reserve = state.dispatching.len().saturating_add(1);
        if state.dispatching.try_reserve(1).is_err()
            || state.dispatched.try_reserve(reserve).is_err()
        {
            return GameStepV3::Capacity;
        }
        // Clones and allocation precede the successful claim CAS.
        let retained_root = root.clone();
        let state_root = root.clone();
        let owner = Arc::clone(self);
        let mut current = self.control.load(Ordering::SeqCst);
        loop {
            if current & ABORT_REQUESTED != 0 {
                return GameStepV3::Aborted;
            }
            if current
                & (ADMISSION_SEALED
                    | PUBLICATION_GRANTED
                    | NO_DATA_FINISHED
                    | PUBLICATION_NOT_STARTED)
                != 0
            {
                return GameStepV3::Closed;
            }
            if current & DISPATCH_MASK == DISPATCH_MASK {
                return GameStepV3::Capacity;
            }
            let Some(next) = current.checked_add(DISPATCH_CLAIM) else {
                return GameStepV3::Capacity;
            };
            match self
                .control
                .compare_exchange(current, next, Ordering::SeqCst, Ordering::SeqCst)
            {
                Ok(_) => break,
                Err(observed) => current = observed,
            }
        }
        // Both vectors have reserved capacity and the caller's slot was
        // checked empty. No callback or fallible operation occurs here.
        state.dispatching.push(state_root);
        *destination = Some(GameDispatchClaimV3 {
            owner,
            root: retained_root,
            envelope: identity,
            send_result: None,
            attempted: false,
            settled: false,
            receipt: None,
        });
        GameStepV3::Ready
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
                self.control.fetch_or(ADMISSION_SEALED, Ordering::SeqCst);
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
        if self.control.load(Ordering::SeqCst) & ABORT_REQUESTED != 0 {
            return GameStepV3::Aborted;
        }
        let mut state = match self.lock() {
            Ok(s) => s,
            Err(e) => return e,
        };
        if self.control.load(Ordering::SeqCst) & ABORT_REQUESTED != 0 {
            return GameStepV3::Aborted;
        }
        if !state.pending.is_empty()
            || !state.dispatching.is_empty()
            || self.control.load(Ordering::SeqCst) & !CONTROL_FLAGS != 0
        {
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
        if self.control.load(Ordering::SeqCst) & ABORT_REQUESTED != 0 {
            return Err(GameStepV3::Aborted);
        }
        let mut state = self.lock()?;
        if self.control.load(Ordering::SeqCst) & ABORT_REQUESTED != 0 {
            return Err(GameStepV3::Aborted);
        }
        if state.phase != PhaseV3::Closed
            || !state.pending.is_empty()
            || !state.dispatching.is_empty()
            || self.control.load(Ordering::SeqCst) & !CONTROL_FLAGS != 0
        {
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
    committed_aborted: Option<bool>,
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
        if let Some(aborted) = self.committed_aborted {
            return if aborted {
                GameStepV3::Aborted
            } else {
                GameStepV3::Ready
            };
        }
        if self.resolved {
            return GameStepV3::Unknown;
        }
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
        if self.owner.control.load(Ordering::SeqCst) & ABORT_REQUESTED != 0 {
            self.abort_receipt = Some(GameRootAbortReceiptV3 {
                owner: Arc::clone(&self.owner),
                root: self.root.clone(),
            });
            self.committed_aborted = Some(true);
            self.resolved = true;
            GameStepV3::Aborted
        } else {
            self.committed_aborted = Some(false);
            self.resolved = true;
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

/// Only the actual P-C and Own root envelopes can use this dispatch path.
pub(super) trait GameDispatchEnvelopeV3: Clone {
    fn original_identity(&self) -> GameDispatchEnvelopeIdentityV3;
}

#[derive(Clone)]
pub(super) enum GameDispatchEnvelopeIdentityV3 {
    Pc(RoventCollectionRootAdmissionV3),
    Own(RoventCollectionRootAdmissionV3, Arc<AtomicBool>),
}
impl GameDispatchEnvelopeIdentityV3 {
    fn matches_root(&self, root: &RoventCollectionRootAdmissionV3) -> bool {
        match self {
            Self::Pc(a) | Self::Own(a, _) => a.same_original_admission(root),
        }
    }
    fn same_original(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Pc(a), Self::Pc(b)) => a.same_original_admission(b),
            (Self::Own(a, ac), Self::Own(b, bc)) => {
                a.same_original_admission(b) && Arc::ptr_eq(ac, bc)
            }
            _ => false,
        }
    }
}

pub(super) struct GameDispatchChannelV3<T> {
    identity: Arc<()>,
    sender: SyncSender<T>,
}
impl<T> GameDispatchChannelV3<T> {
    pub(super) fn new(sender: SyncSender<T>) -> Self {
        Self {
            identity: Arc::new(()),
            sender,
        }
    }
}
impl GameDispatchEnvelopeV3 for RoventCollectionRootAdmissionV3 {
    fn original_identity(&self) -> GameDispatchEnvelopeIdentityV3 {
        GameDispatchEnvelopeIdentityV3::Pc(self.clone())
    }
}
impl GameDispatchEnvelopeV3 for super::own_hooks_lock_v3::OwnRootRequestV3 {
    fn original_identity(&self) -> GameDispatchEnvelopeIdentityV3 {
        GameDispatchEnvelopeIdentityV3::Own(self.admission.clone(), self.revocation_flag())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum GameDispatchOutcomeV3 {
    Sent,
    Full,
    Disconnected,
    Aborted,
    AdmissionSealed,
}

/// Original completed send observation. Sent remains separate from the
/// same root's eventual cancellation, worker join and physical completion.
pub(super) struct GameDispatchReceiptV3<S> {
    owner: Arc<GameAuthorityV3<S>>,
    root: RoventCollectionRootAdmissionV3,
    envelope: GameDispatchEnvelopeIdentityV3,
    outcome: GameDispatchOutcomeV3,
    abort_observed: bool,
}
impl<S> GameDispatchReceiptV3<S> {
    pub(super) fn matches_game(&self, game: &Arc<GameAuthorityV3<S>>) -> bool {
        Arc::ptr_eq(&self.owner, game)
    }
    pub(super) fn matches_root(&self, root: &RoventCollectionRootAdmissionV3) -> bool {
        self.root.same_original_admission(root)
    }
    pub(super) fn matches_envelope<T: GameDispatchEnvelopeV3>(&self, envelope: &T) -> bool {
        self.envelope.same_original(&envelope.original_identity())
    }
    pub(super) fn outcome(&self) -> GameDispatchOutcomeV3 {
        self.outcome
    }
    pub(super) fn abort_observed(&self) -> bool {
        self.abort_observed
    }
}

/// Bounded aggregation of equivalent known-Full observations only. The
/// original game owner permanently binds one dispatch channel; abort-observed,
/// sent, disconnected and uncertain work never enter this summary.
pub(super) struct GameDispatchFullSummaryV3<S> {
    count: u64,
    last: GameDispatchReceiptV3<S>,
}
impl<S> GameDispatchFullSummaryV3<S> {
    pub(super) fn count(&self) -> u64 {
        self.count
    }
    pub(super) fn last(&self) -> &GameDispatchReceiptV3<S> {
        &self.last
    }

    /// Both slots belong to the caller. On mismatch, abort or overflow no
    /// value moves and the original receipt is retained for separate handling.
    pub(super) fn try_merge_in_slots(
        summary: &mut Option<Self>,
        receipt: &mut Option<GameDispatchReceiptV3<S>>,
    ) -> GameStepV3 {
        let Some(next) = receipt.as_ref() else {
            return GameStepV3::Busy;
        };
        if next.outcome != GameDispatchOutcomeV3::Full || next.abort_observed {
            return GameStepV3::Foreign;
        }
        if let Some(original) = summary.as_ref() {
            if !Arc::ptr_eq(&original.last.owner, &next.owner)
                || !original.last.root.same_original_admission(&next.root)
                || !original.last.envelope.same_original(&next.envelope)
                || original.last.outcome != GameDispatchOutcomeV3::Full
                || original.last.abort_observed
            {
                return GameStepV3::Foreign;
            }
            let Some(count) = original.count.checked_add(1) else {
                return GameStepV3::Unknown;
            };
            let next = receipt.take().expect("checked original Full receipt");
            let original = summary.as_mut().expect("checked original Full summary");
            original.last = next;
            original.count = count;
        } else {
            let next = receipt.take().expect("checked original Full receipt");
            *summary = Some(Self {
                count: 1,
                last: next,
            });
        }
        GameStepV3::Ready
    }
}

pub(super) struct GameDispatchClaimV3<S> {
    owner: Arc<GameAuthorityV3<S>>,
    root: RoventCollectionRootAdmissionV3,
    envelope: GameDispatchEnvelopeIdentityV3,
    send_result: Option<GameDispatchOutcomeV3>,
    attempted: bool,
    settled: bool,
    receipt: Option<GameDispatchReceiptV3<S>>,
}

impl<S> GameDispatchClaimV3<S> {
    /// The original envelope remains in the caller's slot on every result.
    /// This sends one clone sharing its original admission/cancel owners;
    /// a second invocation never sends again. A panic leaves claim Unknown.
    pub(super) fn try_send<T: GameDispatchEnvelopeV3>(
        &mut self,
        channel: &GameDispatchChannelV3<T>,
        envelope: &T,
    ) -> Result<GameDispatchOutcomeV3, GameStepV3> {
        if !self.envelope.same_original(&envelope.original_identity()) {
            return Err(GameStepV3::Foreign);
        }
        {
            let state = self.owner.lock()?;
            if !state
                .dispatch_channel
                .as_ref()
                .is_some_and(|original| Arc::ptr_eq(original, &channel.identity))
            {
                return Err(GameStepV3::Foreign);
            }
        }
        if let Some(result) = self.send_result {
            return Ok(result);
        }
        if self.settled {
            return Err(GameStepV3::Unknown);
        }
        if self.attempted {
            self.owner.unknown.store(true, Ordering::SeqCst);
            return Err(GameStepV3::Unknown);
        }
        self.attempted = true;
        let copy = envelope.clone();
        let control = self.owner.control.load(Ordering::SeqCst);
        let result = if control & ABORT_REQUESTED != 0 {
            GameDispatchOutcomeV3::Aborted
        } else if control & ADMISSION_SEALED != 0 {
            GameDispatchOutcomeV3::AdmissionSealed
        } else {
            match channel.sender.try_send(copy) {
                Ok(()) => GameDispatchOutcomeV3::Sent,
                Err(TrySendError::Full(_)) => GameDispatchOutcomeV3::Full,
                Err(TrySendError::Disconnected(_)) => GameDispatchOutcomeV3::Disconnected,
            }
        };
        self.send_result = Some(result);
        Ok(result)
    }

    /// Records the actual prior send result in this original claim. Busy
    /// preserves the claim and never repeats an already attempted send.
    pub(super) fn try_settle(&mut self) -> GameStepV3 {
        if self.settled {
            return GameStepV3::Existing;
        }
        let Some(outcome) = self.send_result else {
            return GameStepV3::Busy;
        };
        let mut state = match self.owner.lock() {
            Ok(s) => s,
            Err(e) => return e,
        };
        let Some(index) = state
            .dispatching
            .iter()
            .position(|a| a.same_original_admission(&self.root))
        else {
            return GameStepV3::Unknown;
        };
        let prior =
            match self
                .owner
                .control
                .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |word| {
                    if (word & DISPATCH_MASK) < DISPATCH_CLAIM {
                        None
                    } else {
                        Some(word - DISPATCH_CLAIM)
                    }
                }) {
                Ok(value) => value,
                Err(_) => return GameStepV3::Unknown,
            };
        let root = state.dispatching.remove(index);
        if outcome == GameDispatchOutcomeV3::Sent {
            state.dispatched.push(root);
        }
        self.receipt = Some(GameDispatchReceiptV3 {
            owner: Arc::clone(&self.owner),
            root: self.root.clone(),
            envelope: self.envelope.clone(),
            outcome,
            abort_observed: prior & ABORT_REQUESTED != 0 || self.owner.abort_requested(),
        });
        self.settled = true;
        GameStepV3::Ready
    }

    pub(super) fn transfer_receipt(
        &mut self,
        destination: &mut Option<GameDispatchReceiptV3<S>>,
    ) -> bool {
        if destination.is_some() || self.receipt.is_none() {
            return false;
        }
        *destination = self.receipt.take();
        true
    }
}
impl<S> Drop for GameDispatchClaimV3<S> {
    fn drop(&mut self) {
        if !self.settled || self.receipt.is_some() {
            self.owner.unknown.store(true, Ordering::SeqCst);
        }
    }
}

/// The caller must install this original guard in the recovery slot before
/// entering the external start primitive. Dropping it unresolved keeps the
/// game Unknown and does not refund the outstanding acquisition.
pub(super) struct GameEffectClaimV3<S> {
    owner: Arc<GameAuthorityV3<S>>,
    root: RoventCollectionRootAdmissionV3,
    kind: GameEffectKindV3,
    stage: Arc<()>,
    actor: Option<Arc<()>>,
    resolved: bool,
}
impl<S> GameEffectClaimV3<S> {
    /// Consumes no result: the real stage/result witness remains with the
    /// caller. It is only an acquisition result, never a physical join.
    pub(super) fn try_complete(&mut self, result: &GameEffectOutcomeV3<S>) -> GameStepV3 {
        if !Arc::ptr_eq(&self.owner, &result.owner)
            || !self.root.same_original_admission(&result.root)
            || self.kind != result.kind
            || !Arc::ptr_eq(&self.stage, &result.stage)
            || !same_effect_actor(&self.actor, &result.actor)
        {
            return GameStepV3::Foreign;
        }
        if self.resolved {
            return GameStepV3::Existing;
        }
        let mut state = match self.owner.lock() {
            Ok(s) => s,
            Err(e) => return e,
        };
        let Some(index) = state.effects.iter().position(|effect| {
            effect.active
                && effect.root.same_original_admission(&self.root)
                && effect.kind == self.kind
                && Arc::ptr_eq(&effect.stage, &self.stage)
                && same_effect_actor(&effect.actor, &self.actor)
        }) else {
            return GameStepV3::Unknown;
        };
        if self
            .owner
            .control
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |word| {
                if word & !(EFFECT_CLAIM - 1) == 0 {
                    None
                } else {
                    Some(word - EFFECT_CLAIM)
                }
            })
            .is_err()
        {
            return GameStepV3::Unknown;
        }
        state.effects[index].active = false;
        self.resolved = true;
        GameStepV3::Ready
    }
}
impl<S> Drop for GameEffectClaimV3<S> {
    fn drop(&mut self) {
        if !self.resolved {
            self.owner.unknown.store(true, Ordering::SeqCst);
        }
    }
}

pub(crate) struct GamePublishGuardV3<S> {
    owner: Arc<GameAuthorityV3<S>>,
    resolved: bool,
}

/// Same in-memory game authority identity only. This is not a publication,
/// output, worker, or physical completion receipt.
pub(super) struct GamePublicationOriginV3 {
    identity: Arc<()>,
}

impl<S> GamePublishGuardV3<S> {
    pub(super) fn original_game_identity(&self) -> GamePublicationOriginV3 {
        GamePublicationOriginV3 {
            identity: Arc::clone(&self.owner.identity),
        }
    }

    pub(super) fn matches_original_game(&self, origin: &GamePublicationOriginV3) -> bool {
        Arc::ptr_eq(&self.owner.identity, &origin.identity)
    }
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
            matches!(
                state.phase,
                PhaseV3::Preparing | PhaseV3::Granted | PhaseV3::NoData | PhaseV3::KnownNotWritten
            ) && output.matches_inputs(&self.owner.inputs)
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

    /// Settle a known pre-mutation failure only using the exact output owner's
    /// nonclone witness for this game. The witness stays in the caller slot
    /// on Busy, Unknown and success; prior chunk writes remain separate.
    pub(super) fn try_finish_publication_unstarted(
        &mut self,
        output: &Arc<ManagedOutputOwnerV3>,
        evidence: &ManagedPublicationUnstartedEvidenceV3,
    ) -> GameStepV3 {
        if !evidence.matches_original(output, self) {
            return GameStepV3::Foreign;
        }
        let mut state = match self.owner.lock() {
            Ok(state) => state,
            Err(error) => return error,
        };
        if state.phase != PhaseV3::Preparing {
            return GameStepV3::Unknown;
        }
        match self.owner.control.compare_exchange(
            ADMISSION_SEALED,
            ADMISSION_SEALED | PUBLICATION_NOT_STARTED,
            Ordering::SeqCst,
            Ordering::SeqCst,
        ) {
            Ok(_) => {
                state.phase = PhaseV3::KnownNotWritten;
                self.resolved = true;
                GameStepV3::Ready
            }
            Err(actual) if actual & ABORT_REQUESTED != 0 => {
                state.phase = PhaseV3::Closed;
                self.resolved = true;
                GameStepV3::Aborted
            }
            Err(_) => GameStepV3::Unknown,
        }
    }

    /// The caller must hold the original pipeline lock and have observed its
    /// original writer pristine with no pending, output or uncertain work.
    /// NoData is terminal for this game; it is not a publication receipt.
    pub(super) fn try_finish_no_data(&mut self) -> GameStepV3 {
        let mut state = match self.owner.lock() {
            Ok(state) => state,
            Err(error) => return error,
        };
        if state.phase != PhaseV3::Preparing {
            return GameStepV3::Unknown;
        }
        match self.owner.control.compare_exchange(
            ADMISSION_SEALED,
            ADMISSION_SEALED | NO_DATA_FINISHED,
            Ordering::SeqCst,
            Ordering::SeqCst,
        ) {
            Ok(_) => {
                state.phase = PhaseV3::NoData;
                self.resolved = true;
                GameStepV3::Ready
            }
            Err(actual) if actual & ABORT_REQUESTED != 0 => {
                state.phase = PhaseV3::Closed;
                self.resolved = true;
                GameStepV3::Aborted
            }
            Err(_) => GameStepV3::Unknown,
        }
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
            ADMISSION_SEALED,
            ADMISSION_SEALED | PUBLICATION_GRANTED,
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
