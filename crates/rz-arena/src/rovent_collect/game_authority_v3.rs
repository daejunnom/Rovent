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
const GAME_CLOSED: u64 = 32;
const DISPATCH_CLAIM: u64 = 64;
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

/// The independently observable operations belonging to one original P-C
/// recovery. An acquisition claim is not worker or device completion.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum PcOperationKindV3 {
    RouteInstall,
    NativeWorker,
    AttachRoute,
    PrepareAdmission,
    BindPolicyConsumer,
    PublishPrepared,
}

/// Produced by the original recovery's operation slot. `NoEffect` may only
/// represent a producer result that proves no effect occurred.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum PcOperationObservationV3 {
    Pending,
    NoEffect,
    FailedNoEffect,
    Committed,
    Unknown,
}

struct PcOperationRecordV3 {
    root: RoventCollectionRootAdmissionV3,
    kind: PcOperationKindV3,
    slot: Arc<()>,
    generation: u64,
    active: bool,
    retryable: bool,
    failed_no_effect: bool,
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

/// The original typed stage/receive owners move together into this source.
/// A stage source alone is never promoted to a game effect capability.
pub(super) struct GameEffectSourceV3<S> {
    owner: Arc<GameAuthorityV3<S>>,
    root: RoventCollectionRootAdmissionV3,
    kind: GameEffectKindV3,
    stage: Arc<()>,
    actor: Option<Arc<()>>,
    original: GameEffectOriginalV3,
}

enum GameEffectOriginalV3 {
    PcScope {
        proof: super::pc_worker_v3::PcReceivedStageProofV3,
        source: super::teacher::v3_session::TeacherScopeEffectSourceV3,
    },
    PcActor {
        proof: super::pc_worker_v3::PcReceivedActorProofV3,
        source: super::teacher::v3_session::TeacherActorEffectSourceV3,
    },
    OwnScope {
        proof: super::own_worker_v3::OwnReceivedStageProofV3,
        source: super::teacher::v3_session::TeacherScopeEffectSourceV3,
    },
    OwnActor {
        proof: super::own_worker_v3::OwnReceivedActorProofV3,
        source: super::teacher::v3_session::TeacherActorEffectSourceV3,
    },
}

impl<S> GameEffectSourceV3<S> {
    pub(super) fn pc_scope_source(
        &self,
    ) -> Option<&super::teacher::v3_session::TeacherScopeEffectSourceV3> {
        match &self.original {
            GameEffectOriginalV3::PcScope { source, .. } => Some(source),
            _ => None,
        }
    }

    pub(super) fn pc_actor_source(
        &self,
    ) -> Option<&super::teacher::v3_session::TeacherActorEffectSourceV3> {
        match &self.original {
            GameEffectOriginalV3::PcActor { source, .. } => Some(source),
            _ => None,
        }
    }

    pub(super) fn own_scope_source(
        &self,
    ) -> Option<&super::teacher::v3_session::TeacherScopeEffectSourceV3> {
        match &self.original {
            GameEffectOriginalV3::OwnScope { source, .. } => Some(source),
            _ => None,
        }
    }

    pub(super) fn own_actor_source(
        &self,
    ) -> Option<&super::teacher::v3_session::TeacherActorEffectSourceV3> {
        match &self.original {
            GameEffectOriginalV3::OwnActor { source, .. } => Some(source),
            _ => None,
        }
    }
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
    // P-C roots own independent Start cancellation gates. Own roots keep their
    // existing execution path until their original Start is wired separately.
    pc_root_cancel: Vec<(
        RoventCollectionRootAdmissionV3,
        Arc<rz_uci::search_driver::RoventRootCancelGateV3>,
    )>,
    dispatching: Vec<RoventCollectionRootAdmissionV3>,
    dispatched: Vec<RoventCollectionRootAdmissionV3>,
    effects: Vec<GameEffectRecordV3>,
    pc_operations: Vec<PcOperationRecordV3>,
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
    /// This original Start/root is canceled; the game may admit a later root.
    RootCanceled,
    Failed,
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
                pc_root_cancel: Vec::new(),
                dispatching: Vec::new(),
                dispatched: Vec::new(),
                effects: Vec::new(),
                pc_operations: Vec::new(),
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

    /// Compare the exact admitted set only after native admission was sealed.
    /// This is not an observer drain, supervisor join or publication proof.
    /// None preserves Busy or unfinished original reservation/claim custody.
    pub(super) fn try_matches_sealed_roots(
        &self,
        roots: &[RoventCollectionRootAdmissionV3],
    ) -> Result<Option<bool>, &'static str> {
        let state = match self.lock() {
            Ok(state) => state,
            Err(GameStepV3::Busy) => return Ok(None),
            Err(_) => return Err("V3 game authority state unknown"),
        };
        if state.phase != PhaseV3::AdmissionSealed {
            return Ok(None);
        }
        if !state.pending.is_empty()
            || !state.dispatching.is_empty()
            || state.effects.iter().any(|effect| effect.active)
            || state.pc_operations.iter().any(|operation| operation.active)
            || self.control.load(Ordering::SeqCst) & !CONTROL_FLAGS != 0
        {
            return Ok(None);
        }
        if roots.len() != state.admitted.len() {
            return Ok(Some(false));
        }
        for (index, root) in roots.iter().enumerate() {
            if roots[..index]
                .iter()
                .any(|prior| prior.same_original_admission(root))
                || !state
                    .admitted
                    .iter()
                    .any(|original| original.same_original_admission(root))
            {
                return Ok(Some(false));
            }
        }
        Ok(Some(true))
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

    /// Bind only the root gate already installed on this exact UCI Start.
    /// A canceled root remains bound so its original no-send/worker custody
    /// can be settled; this never asserts a native or device completion.
    pub(super) fn try_bind_pc_root_cancel(
        &self,
        root: &RoventCollectionRootAdmissionV3,
        gate: &Arc<rz_uci::search_driver::RoventRootCancelGateV3>,
    ) -> GameStepV3 {
        use rz_uci::search_driver::RoventRootCancelBindingV3 as Binding;
        match root
            .root_admission()
            .original_context()
            .observe_original_collection_root_cancel_v3(gate)
        {
            Binding::Bound | Binding::Canceled => {}
            Binding::Pending => return GameStepV3::Busy,
            Binding::Unknown => return GameStepV3::Unknown,
            Binding::Foreign => return GameStepV3::Foreign,
        }
        let mut state = match self.lock() {
            Ok(state) => state,
            Err(error) => return error,
        };
        if !matches!(state.phase, PhaseV3::Open | PhaseV3::AdmissionSealed) {
            return GameStepV3::Closed;
        }
        if !state
            .admitted
            .iter()
            .any(|original| original.same_original_admission(root))
        {
            return GameStepV3::Foreign;
        }
        if let Some((_, existing)) = state
            .pc_root_cancel
            .iter()
            .find(|(original, _)| original.same_original_admission(root))
        {
            return if Arc::ptr_eq(existing, gate) {
                GameStepV3::Existing
            } else {
                GameStepV3::Foreign
            };
        }
        if state.pc_root_cancel.len() >= usize::from(self.inputs.plan().maximum_roots)
            || state.pc_root_cancel.try_reserve(1).is_err()
        {
            return GameStepV3::Capacity;
        }
        state.pc_root_cancel.push((root.clone(), Arc::clone(gate)));
        GameStepV3::Ready
    }

    /// Observe the gate installed by the original Start for this exact root.
    /// This reports a request only; it is not a no-send or completion proof.
    pub(super) fn try_observe_pc_root_cancel(
        &self,
        root: &RoventCollectionRootAdmissionV3,
    ) -> GameStepV3 {
        use rz_uci::search_driver::RoventRootCancelBindingV3 as Binding;
        let state = match self.lock() {
            Ok(state) => state,
            Err(error) => return error,
        };
        if !state
            .admitted
            .iter()
            .any(|original| original.same_original_admission(root))
        {
            return GameStepV3::Foreign;
        }
        let Some((_, gate)) = state
            .pc_root_cancel
            .iter()
            .find(|(original, _)| original.same_original_admission(root))
        else {
            return GameStepV3::Busy;
        };
        match root
            .root_admission()
            .original_context()
            .observe_original_collection_root_cancel_v3(gate)
        {
            Binding::Bound => GameStepV3::Ready,
            Binding::Canceled => GameStepV3::RootCanceled,
            Binding::Pending => GameStepV3::Busy,
            Binding::Unknown => GameStepV3::Unknown,
            Binding::Foreign => GameStepV3::Foreign,
        }
    }

    /// Cancel and all root-local producer claims settled on the same Start.
    /// This does not prove worker, target, native, or device completion.
    pub(super) fn try_observe_pc_root_cancel_quiescent(
        &self,
        root: &RoventCollectionRootAdmissionV3,
    ) -> GameStepV3 {
        let state = match self.lock() {
            Ok(state) => state,
            Err(error) => return error,
        };
        self.observe_pc_root_cancel_quiescent_locked(&state, root)
    }

    fn observe_pc_root_cancel_quiescent_locked(
        &self,
        state: &StateV3<S>,
        root: &RoventCollectionRootAdmissionV3,
    ) -> GameStepV3 {
        use rz_uci::search_driver::{
            RoventRootCancelBindingV3 as Binding, RoventRootClaimObservationV3 as Observation,
        };
        if !state
            .admitted
            .iter()
            .any(|original| original.same_original_admission(root))
        {
            return GameStepV3::Foreign;
        }
        let Some((_, gate)) = state
            .pc_root_cancel
            .iter()
            .find(|(original, _)| original.same_original_admission(root))
        else {
            return GameStepV3::Busy;
        };
        match root
            .root_admission()
            .original_context()
            .observe_original_collection_root_cancel_v3(gate)
        {
            Binding::Canceled => {}
            Binding::Bound | Binding::Pending => return GameStepV3::Busy,
            Binding::Unknown => return GameStepV3::Unknown,
            Binding::Foreign => return GameStepV3::Foreign,
        }
        match gate.observe_canceled_claims() {
            Observation::Quiescent => GameStepV3::Ready,
            Observation::NotBound | Observation::NotCanceled | Observation::Active => {
                GameStepV3::Busy
            }
            Observation::Unknown => GameStepV3::Unknown,
        }
    }

    /// A P-C effect must claim the same root-local atomic word that the
    /// original Start cancels. Looking at a cancellation bool and later
    /// claiming the separate game word would leave a check/act race.
    fn try_claim_pc_root_cancel(
        &self,
        state: &StateV3<S>,
        root: &RoventCollectionRootAdmissionV3,
    ) -> Result<rz_uci::search_driver::RoventRootCancelClaimV3, GameStepV3> {
        use rz_uci::search_driver::RoventRootCancelClaimErrorV3 as Error;
        let Some((_, gate)) = state
            .pc_root_cancel
            .iter()
            .find(|(original, _)| original.same_original_admission(root))
        else {
            return Err(GameStepV3::Busy);
        };
        gate.try_claim().map_err(|error| match error {
            Error::NotBound => GameStepV3::Busy,
            Error::Canceled => GameStepV3::RootCanceled,
            Error::Unknown => GameStepV3::Unknown,
            Error::Capacity => GameStepV3::Capacity,
        })
    }

    /// If the independent game-wide CAS rejects an attempt, the original
    /// root claim has made no producer call and may be explicitly returned.
    fn return_unstarted_pc_root_claim(
        &self,
        claim: Option<rz_uci::search_driver::RoventRootCancelClaimV3>,
        result: GameStepV3,
    ) -> GameStepV3 {
        if claim.is_some_and(|claim| claim.resolve().is_err()) {
            self.unknown.store(true, Ordering::SeqCst);
            GameStepV3::Unknown
        } else {
            result
        }
    }

    /// Resolve only after the original producer outcome has been retained by
    /// the matching game claim. This is not a worker join or physical fence.
    fn resolve_pc_root_claim(
        &self,
        claim: &mut Option<rz_uci::search_driver::RoventRootCancelClaimV3>,
    ) -> bool {
        if claim.take().is_some_and(|claim| claim.resolve().is_err()) {
            self.unknown.store(true, Ordering::SeqCst);
            false
        } else {
            true
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
            || state.pc_operations.iter().any(|operation| operation.active)
            || self.control.load(Ordering::SeqCst) & !(EFFECT_CLAIM - 1) != 0
        {
            return GameStepV3::Busy;
        }
        GameStepV3::Ready
    }

    /// Candidate-only until each stage owner can issue an original source and
    /// actual outcome witness. No generic bool can resolve this claim.
    fn try_claim_effect(
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
        let mut root_claim = if matches!(
            &source.original,
            GameEffectOriginalV3::PcScope { .. } | GameEffectOriginalV3::PcActor { .. }
        ) {
            match self.try_claim_pc_root_cancel(&state, &source.root) {
                Ok(claim) => Some(claim),
                Err(error) => return error,
            }
        } else {
            None
        };
        let mut current = self.control.load(Ordering::SeqCst);
        loop {
            if current & ABORT_REQUESTED != 0 {
                return self.return_unstarted_pc_root_claim(root_claim.take(), GameStepV3::Aborted);
            }
            if current
                & (ADMISSION_SEALED
                    | PUBLICATION_GRANTED
                    | NO_DATA_FINISHED
                    | PUBLICATION_NOT_STARTED)
                != 0
            {
                return self.return_unstarted_pc_root_claim(root_claim.take(), GameStepV3::Closed);
            }
            let Some(next) = current.checked_add(EFFECT_CLAIM) else {
                return self
                    .return_unstarted_pc_root_claim(root_claim.take(), GameStepV3::Capacity);
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
            root_cancel_claim: root_claim,
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
        // Only P-C needs a distinct original receipt witness. Allocate it
        // before the claim CAS so Own dispatch keeps its existing behavior.
        let receipt_identity = match &identity {
            GameDispatchEnvelopeIdentityV3::Pc(_) => Some(Arc::new(())),
            GameDispatchEnvelopeIdentityV3::Own(_, _) => None,
        };
        let mut root_claim = if matches!(&identity, GameDispatchEnvelopeIdentityV3::Pc(_)) {
            match self.try_claim_pc_root_cancel(&state, root) {
                Ok(claim) => Some(claim),
                Err(error) => return error,
            }
        } else {
            None
        };
        let mut current = self.control.load(Ordering::SeqCst);
        loop {
            if current & ABORT_REQUESTED != 0 {
                return self.return_unstarted_pc_root_claim(root_claim.take(), GameStepV3::Aborted);
            }
            if current
                & (ADMISSION_SEALED
                    | PUBLICATION_GRANTED
                    | NO_DATA_FINISHED
                    | PUBLICATION_NOT_STARTED)
                != 0
            {
                return self.return_unstarted_pc_root_claim(root_claim.take(), GameStepV3::Closed);
            }
            if current & DISPATCH_MASK == DISPATCH_MASK {
                return self
                    .return_unstarted_pc_root_claim(root_claim.take(), GameStepV3::Capacity);
            }
            let Some(next) = current.checked_add(DISPATCH_CLAIM) else {
                return self
                    .return_unstarted_pc_root_claim(root_claim.take(), GameStepV3::Capacity);
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
            receipt_identity,
            root_cancel_claim: root_claim,
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
                match self.control.compare_exchange(
                    ADMISSION_SEALED,
                    ADMISSION_SEALED | GAME_CLOSED,
                    Ordering::SeqCst,
                    Ordering::SeqCst,
                ) {
                    Ok(_) => {}
                    Err(actual) if actual & ABORT_REQUESTED != 0 => return GameStepV3::Aborted,
                    Err(_) => return GameStepV3::Unknown,
                }
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
    /// Observe one original P-C receipt only after the game is permanently
    /// closed to further dispatch by sticky abort or admission seal. Normal
    /// Full and Disconnected remain retryable before either condition.
    fn try_check_pc_not_sent(
        self: &Arc<Self>,
        root: &RoventCollectionRootAdmissionV3,
        envelope: &RoventCollectionRootAdmissionV3,
        receipt: &GameDispatchReceiptV3<super::pc_worker_v3::PcWorkerSupervisorV3>,
    ) -> GameStepV3 {
        let state = match self.lock() {
            Ok(state) => state,
            Err(error) => return error,
        };
        if self.unknown.load(Ordering::SeqCst) {
            return GameStepV3::Unknown;
        }
        if !receipt.matches_game(self)
            || !receipt.matches_root(root)
            || !receipt.matches_envelope(envelope)
            || !root.same_original_admission(envelope)
            || receipt.receipt_identity.is_none()
            || !matches!(
                receipt.outcome,
                GameDispatchOutcomeV3::Full
                    | GameDispatchOutcomeV3::Disconnected
                    | GameDispatchOutcomeV3::Aborted
                    | GameDispatchOutcomeV3::AdmissionSealed
            )
        {
            return GameStepV3::Foreign;
        }
        if !state
            .admitted
            .iter()
            .any(|original| original.same_original_admission(root))
        {
            return GameStepV3::Foreign;
        }
        if self.control.load(Ordering::SeqCst) & (ABORT_REQUESTED | ADMISSION_SEALED) == 0 {
            return GameStepV3::Busy;
        }
        if state
            .dispatching
            .iter()
            .any(|original| original.same_original_admission(root))
        {
            return GameStepV3::Busy;
        }
        if state
            .dispatched
            .iter()
            .any(|original| original.same_original_admission(root))
        {
            return GameStepV3::Foreign;
        }
        GameStepV3::Ready
    }

    /// Retain an exact-receipt witness in the original caller's destination.
    /// This proves only no P-C dispatch, not worker or physical completion.
    pub(super) fn try_observe_pc_not_sent(
        self: &Arc<Self>,
        root: &RoventCollectionRootAdmissionV3,
        envelope: &RoventCollectionRootAdmissionV3,
        receipt: &GameDispatchReceiptV3<super::pc_worker_v3::PcWorkerSupervisorV3>,
        destination: &mut Option<PcNotSentDispatchWitnessV3>,
    ) -> GameStepV3 {
        if destination.is_some() {
            return GameStepV3::Busy;
        }
        match self.try_check_pc_not_sent(root, envelope, receipt) {
            GameStepV3::Ready => {}
            other => return other,
        }
        let Some(identity) = receipt.receipt_identity.as_ref() else {
            return GameStepV3::Unknown;
        };
        *destination = Some(PcNotSentDispatchWitnessV3 {
            owner: Arc::clone(self),
            root: root.clone(),
            receipt_identity: Arc::clone(identity),
        });
        GameStepV3::Ready
    }

    /// After sticky game abort or quiescent original root cancel, reobserve
    /// one pre-retained original route slot.
    /// A missing operation record means no acquisition occurred; an inactive
    /// retryable record means the original producer's no-effect result was
    /// already validated. Neither case proves a teacher or native fence.
    pub(super) fn try_observe_pc_route_never_started(
        self: &Arc<Self>,
        slot_owner: &Arc<Mutex<super::pc_worker_v3::PcOperationSlotV3>>,
        slot: &super::pc_worker_v3::PcOperationSlotV3,
        recovery: &super::pc_worker_v3::PcRootRecoveryV3,
        registration: &super::native_capture::RoventNativeCaptureRegistration,
        fence: &super::native_capture::RoventPreRouteFenceWitnessV3,
    ) -> GameStepV3 {
        // Actual original registration must first confirm its sticky fence.
        // Busy and unknown are never converted to a no-route observation.
        match fence.try_matches_original(registration, slot.admission()) {
            Ok(Some(true)) => {}
            Ok(Some(false)) => return GameStepV3::Foreign,
            Ok(None) => return GameStepV3::Busy,
            Err(_) => return GameStepV3::Unknown,
        }
        if slot.kind() != PcOperationKindV3::RouteInstall {
            return GameStepV3::Foreign;
        }
        match slot.try_matches_abort_source(slot_owner, recovery, registration, 0) {
            Some(true) => {}
            Some(false) => return GameStepV3::Foreign,
            None => return GameStepV3::Unknown,
        }
        let Some(proof) = slot.abort_source_proof() else {
            return GameStepV3::Unknown;
        };
        match proof.matches_game(self) {
            Some(true) => {}
            Some(false) => return GameStepV3::Foreign,
            None => return GameStepV3::Unknown,
        }
        let state = match self.lock() {
            Ok(state) => state,
            Err(error) => return error,
        };
        if !self.abort_requested() {
            match self.observe_pc_root_cancel_quiescent_locked(&state, slot.admission()) {
                GameStepV3::Ready => {}
                other => return other,
            }
        }
        if !state
            .admitted
            .iter()
            .any(|root| root.same_original_admission(slot.admission()))
        {
            return GameStepV3::Foreign;
        }
        let mut records = state.pc_operations.iter().filter(|operation| {
            operation.root.same_original_admission(slot.admission())
                && operation.kind == PcOperationKindV3::RouteInstall
        });
        let original = records.next();
        if records.next().is_some() {
            return GameStepV3::Unknown;
        }
        match original {
            None if slot.has_no_attempt_data() => GameStepV3::Ready,
            None => GameStepV3::Busy,
            Some(record) if !Arc::ptr_eq(&record.slot, slot.identity()) => GameStepV3::Foreign,
            Some(record) if record.generation == 0 => GameStepV3::Unknown,
            Some(record) if record.active => GameStepV3::Busy,
            Some(record) if record.retryable && record.failed_no_effect => GameStepV3::Unknown,
            Some(record) if record.retryable && slot.has_no_attempt_data() => GameStepV3::Ready,
            // The producer's error proves this attempt did not install a
            // route, but can also report an existing route or ticket. Do not
            // release this root until original registration has a durable
            // no-route/no-late-ticket fence.
            Some(record) if record.failed_no_effect => GameStepV3::Busy,
            Some(_) => GameStepV3::Busy,
        }
    }

    /// After sticky game abort or quiescent original root cancel, prove that
    /// this original recovery's later slot has
    /// never acquired an operation claim. A retained record, including an
    /// already settled NoEffect, stays unresolved for this narrower proof.
    /// This observes only acquisition, not worker or physical completion.
    pub(super) fn try_observe_pc_operation_never_claimed(
        self: &Arc<Self>,
        slot_owner: &Arc<Mutex<super::pc_worker_v3::PcOperationSlotV3>>,
        slot: &super::pc_worker_v3::PcOperationSlotV3,
        recovery: &super::pc_worker_v3::PcRootRecoveryV3,
        registration: &super::native_capture::RoventNativeCaptureRegistration,
        index: usize,
    ) -> GameStepV3 {
        if index == 0 {
            return GameStepV3::Foreign;
        }
        match slot.try_matches_abort_source(slot_owner, recovery, registration, index) {
            Some(true) => {}
            Some(false) => return GameStepV3::Foreign,
            None => return GameStepV3::Unknown,
        }
        if !slot.has_no_attempt_data() {
            return GameStepV3::Busy;
        }
        let Some(proof) = slot.abort_source_proof() else {
            return GameStepV3::Unknown;
        };
        match proof.matches_game(self) {
            Some(true) => {}
            Some(false) => return GameStepV3::Foreign,
            None => return GameStepV3::Unknown,
        }
        let state = match self.lock() {
            Ok(state) => state,
            Err(error) => return error,
        };
        if !self.abort_requested() {
            match self.observe_pc_root_cancel_quiescent_locked(&state, slot.admission()) {
                GameStepV3::Ready => {}
                other => return other,
            }
        }
        if !state
            .admitted
            .iter()
            .any(|root| root.same_original_admission(slot.admission()))
        {
            return GameStepV3::Foreign;
        }
        if state.pc_operations.iter().any(|operation| {
            operation.root.same_original_admission(slot.admission())
                && operation.kind == slot.kind()
        }) {
            // Do not infer no-effect from an existing record. This includes
            // old successful and settled no-effect attempts alike.
            return GameStepV3::Busy;
        }
        GameStepV3::Ready
    }

    /// Claim one operation against the same atomic abort word as teacher
    /// effects. The original recovery owns both `slot` and `destination`.
    /// A claim is authorization to attempt an operation, not its outcome.
    pub(super) fn try_claim_pc_operation(
        self: &Arc<Self>,
        proof: &super::pc_worker_v3::PcReceivedStageProofV3,
        slot: &super::pc_worker_v3::PcOperationSlotV3,
        supervisor: &super::pc_worker_v3::PcWorkerSupervisorV3,
        recovery: &super::pc_worker_v3::PcRootRecoveryV3,
        destination: &mut Option<PcOperationClaimV3>,
    ) -> GameStepV3 {
        if destination.is_some() {
            return GameStepV3::Busy;
        }
        match proof.matches_game(self) {
            Some(true) => {}
            Some(false) => return GameStepV3::Foreign,
            None => return GameStepV3::Unknown,
        }
        if !proof.matches_admission(slot.admission()) {
            return GameStepV3::Foreign;
        }
        match slot.try_revalidate_with_guards(proof, supervisor, recovery) {
            Ok(Some(true)) => {}
            Ok(Some(false)) => return GameStepV3::Foreign,
            Ok(None) => return GameStepV3::Busy,
            Err(_) => return GameStepV3::Unknown,
        }
        let mut state = match self.lock() {
            Ok(state) => state,
            Err(error) => return error,
        };
        if state.phase != PhaseV3::Open {
            return GameStepV3::Closed;
        }
        if !state
            .admitted
            .iter()
            .any(|root| root.same_original_admission(slot.admission()))
        {
            return GameStepV3::Foreign;
        }
        let index = state.pc_operations.iter().position(|operation| {
            operation.root.same_original_admission(slot.admission())
                && operation.kind == slot.kind()
        });
        let generation = if let Some(index) = index {
            let operation = &state.pc_operations[index];
            if !Arc::ptr_eq(&operation.slot, slot.identity()) {
                return GameStepV3::Foreign;
            }
            if operation.active {
                return GameStepV3::Busy;
            }
            if !operation.retryable {
                return GameStepV3::Existing;
            }
            let Some(next) = operation.generation.checked_add(1) else {
                return GameStepV3::Capacity;
            };
            next
        } else {
            let limit = usize::from(self.inputs.plan().maximum_roots).checked_mul(6);
            if limit.is_none_or(|limit| state.pc_operations.len() >= limit)
                || state.pc_operations.try_reserve(1).is_err()
            {
                return GameStepV3::Capacity;
            }
            1
        };
        // All allocations and identity clones precede the successful CAS.
        let root = slot.admission().clone();
        let record_root = root.clone();
        let identity = Arc::clone(slot.identity());
        let record_identity = Arc::clone(slot.identity());
        let owner = Arc::clone(self);
        let kind = slot.kind();
        let mut root_claim = match self.try_claim_pc_root_cancel(&state, &root) {
            Ok(claim) => Some(claim),
            Err(error) => return error,
        };
        let mut current = self.control.load(Ordering::SeqCst);
        loop {
            if current & ABORT_REQUESTED != 0 {
                return self.return_unstarted_pc_root_claim(root_claim.take(), GameStepV3::Aborted);
            }
            if current
                & (ADMISSION_SEALED
                    | PUBLICATION_GRANTED
                    | NO_DATA_FINISHED
                    | PUBLICATION_NOT_STARTED
                    | GAME_CLOSED)
                != 0
            {
                return self.return_unstarted_pc_root_claim(root_claim.take(), GameStepV3::Closed);
            }
            let Some(next) = current.checked_add(EFFECT_CLAIM) else {
                return self
                    .return_unstarted_pc_root_claim(root_claim.take(), GameStepV3::Capacity);
            };
            match self
                .control
                .compare_exchange(current, next, Ordering::SeqCst, Ordering::SeqCst)
            {
                Ok(_) => break,
                Err(observed) => current = observed,
            }
        }
        if let Some(index) = index {
            let operation = &mut state.pc_operations[index];
            operation.generation = generation;
            operation.active = true;
            operation.retryable = false;
            operation.failed_no_effect = false;
        } else {
            state.pc_operations.push(PcOperationRecordV3 {
                root: record_root,
                kind,
                slot: record_identity,
                generation,
                active: true,
                retryable: false,
                failed_no_effect: false,
            });
        }
        *destination = Some(PcOperationClaimV3 {
            owner,
            root,
            kind,
            slot: identity,
            generation,
            root_cancel_claim: root_claim,
            resolved: false,
        });
        GameStepV3::Ready
    }

    /// Move the original received-stage proof and nonclone scope source into
    /// an already retained destination only after same-owner revalidation.
    /// Neither Busy nor a rejection consumes either original input slot.
    pub(super) fn try_issue_pc_scope_source_in_slot(
        self: &Arc<Self>,
        proof_slot: &mut Option<super::pc_worker_v3::PcReceivedStageProofV3>,
        source_slot: &mut Option<super::teacher::v3_session::TeacherScopeEffectSourceV3>,
        supervisor: &super::pc_worker_v3::PcWorkerSupervisorV3,
        recovery: &super::pc_worker_v3::PcRootRecoveryV3,
        root: &RoventCollectionRootAdmissionV3,
        destination: &mut Option<GameEffectSourceV3<super::pc_worker_v3::PcWorkerSupervisorV3>>,
        error_destination: &mut Option<&'static str>,
    ) -> GameStepV3 {
        if error_destination.is_some() {
            return GameStepV3::Failed;
        }
        if destination.is_some() {
            return GameStepV3::Busy;
        }
        let (Some(proof), Some(source)) = (proof_slot.as_ref(), source_slot.as_ref()) else {
            return GameStepV3::Busy;
        };
        match proof.matches_game(self) {
            Some(true) => {}
            Some(false) => return GameStepV3::Foreign,
            None => return GameStepV3::Unknown,
        }
        if !proof.matches_admission(root) || !proof.matches_scope_source(source) {
            return GameStepV3::Foreign;
        }
        match proof.try_revalidate_with_guards(supervisor, recovery) {
            Ok(Some(true)) => {}
            Ok(None) => return GameStepV3::Busy,
            Ok(Some(false)) => return GameStepV3::Foreign,
            Err(error) => {
                *error_destination = Some(error);
                return GameStepV3::Failed;
            }
        }
        let owner = Arc::clone(self);
        let root = root.clone();
        let stage = Arc::clone(source.identity());
        // Both source slots were checked above. No fallible operation or
        // external callback occurs between their take and destination write.
        let proof = proof_slot
            .take()
            .expect("validated original received proof");
        let source = source_slot.take().expect("validated original scope source");
        *destination = Some(GameEffectSourceV3 {
            owner,
            root,
            kind: GameEffectKindV3::TeacherScope,
            stage,
            actor: None,
            original: GameEffectOriginalV3::PcScope { proof, source },
        });
        GameStepV3::Ready
    }

    /// A source is not a dispatch/start permit. Revalidate its original
    /// received stage under the caller's existing guards before game claim.
    pub(super) fn try_claim_pc_scope(
        self: &Arc<Self>,
        source: &GameEffectSourceV3<super::pc_worker_v3::PcWorkerSupervisorV3>,
        supervisor: &super::pc_worker_v3::PcWorkerSupervisorV3,
        recovery: &super::pc_worker_v3::PcRootRecoveryV3,
        destination: &mut Option<GameEffectClaimV3<super::pc_worker_v3::PcWorkerSupervisorV3>>,
        error_destination: &mut Option<&'static str>,
    ) -> GameStepV3 {
        if error_destination.is_some() {
            return GameStepV3::Failed;
        }
        let GameEffectOriginalV3::PcScope {
            proof,
            source: original,
        } = &source.original
        else {
            return GameStepV3::Foreign;
        };
        if !Arc::ptr_eq(self, &source.owner) {
            return GameStepV3::Foreign;
        }
        match proof.matches_game(self) {
            Some(true) => {}
            Some(false) => return GameStepV3::Foreign,
            None => return GameStepV3::Unknown,
        }
        if !proof.matches_admission(&source.root)
            || !proof.matches_scope_source(original)
            || !original.matches_stage_identity(&source.stage)
        {
            return GameStepV3::Foreign;
        }
        match proof.try_revalidate_with_guards(supervisor, recovery) {
            Ok(Some(true)) => {}
            Ok(None) => return GameStepV3::Busy,
            Ok(Some(false)) => return GameStepV3::Foreign,
            Err(error) => {
                *error_destination = Some(error);
                return GameStepV3::Failed;
            }
        }
        self.try_claim_effect(source, destination)
    }

    /// Bind one actual received root and its original Ready actor into the
    /// caller-owned slot. Identity observation is not a start or join.
    pub(super) fn try_issue_pc_actor_source_in_slot(
        self: &Arc<Self>,
        proof_slot: &mut Option<super::pc_worker_v3::PcReceivedActorProofV3>,
        source_slot: &mut Option<super::teacher::v3_session::TeacherActorEffectSourceV3>,
        supervisor: &super::pc_worker_v3::PcWorkerSupervisorV3,
        recovery: &super::pc_worker_v3::PcRootRecoveryV3,
        stage: &super::teacher::v3_session::TeacherSessionStagingV3,
        session: &super::teacher::v3_session::RoventCollectionTeacherSessionV3,
        root: &RoventCollectionRootAdmissionV3,
        destination: &mut Option<GameEffectSourceV3<super::pc_worker_v3::PcWorkerSupervisorV3>>,
        error_destination: &mut Option<&'static str>,
    ) -> GameStepV3 {
        if error_destination.is_some() {
            return GameStepV3::Failed;
        }
        if destination.is_some() {
            return GameStepV3::Busy;
        }
        let (Some(proof), Some(source)) = (proof_slot.as_ref(), source_slot.as_ref()) else {
            return GameStepV3::Busy;
        };
        match proof.matches_game(self) {
            Some(true) => {}
            Some(false) => return GameStepV3::Foreign,
            None => return GameStepV3::Unknown,
        }
        if !proof.matches_admission(root) || !proof.matches_actor_source(source) {
            return GameStepV3::Foreign;
        }
        match proof.try_revalidate_with_guards(supervisor, recovery, stage, session, source) {
            Ok(Some(true)) => {}
            Ok(None) => return GameStepV3::Busy,
            Ok(Some(false)) => return GameStepV3::Foreign,
            Err(error) => {
                *error_destination = Some(error);
                return GameStepV3::Failed;
            }
        }
        let owner = Arc::clone(self);
        let root = root.clone();
        let stage = Arc::clone(source.session_identity());
        let actor = Some(Arc::clone(source.actor_identity()));
        let proof = proof_slot
            .take()
            .expect("validated original received actor proof");
        let source = source_slot.take().expect("validated original actor source");
        *destination = Some(GameEffectSourceV3 {
            owner,
            root,
            kind: GameEffectKindV3::TeacherActor,
            stage,
            actor,
            original: GameEffectOriginalV3::PcActor { proof, source },
        });
        GameStepV3::Ready
    }

    pub(super) fn try_claim_pc_actor(
        self: &Arc<Self>,
        source: &GameEffectSourceV3<super::pc_worker_v3::PcWorkerSupervisorV3>,
        supervisor: &super::pc_worker_v3::PcWorkerSupervisorV3,
        recovery: &super::pc_worker_v3::PcRootRecoveryV3,
        stage: &super::teacher::v3_session::TeacherSessionStagingV3,
        session: &super::teacher::v3_session::RoventCollectionTeacherSessionV3,
        destination: &mut Option<GameEffectClaimV3<super::pc_worker_v3::PcWorkerSupervisorV3>>,
        error_destination: &mut Option<&'static str>,
    ) -> GameStepV3 {
        if error_destination.is_some() {
            return GameStepV3::Failed;
        }
        let GameEffectOriginalV3::PcActor {
            proof,
            source: original,
        } = &source.original
        else {
            return GameStepV3::Foreign;
        };
        if !Arc::ptr_eq(self, &source.owner) {
            return GameStepV3::Foreign;
        }
        match proof.matches_game(self) {
            Some(true) => {}
            Some(false) => return GameStepV3::Foreign,
            None => return GameStepV3::Unknown,
        }
        if !proof.matches_admission(&source.root)
            || !proof.matches_actor_source(original)
            || !original.matches_session_identity(&source.stage)
            || !source
                .actor
                .as_ref()
                .is_some_and(|actor| original.matches_actor_identity(actor))
        {
            return GameStepV3::Foreign;
        }
        match proof.try_revalidate_with_guards(supervisor, recovery, stage, session, original) {
            Ok(Some(true)) => {}
            Ok(None) => return GameStepV3::Busy,
            Ok(Some(false)) => return GameStepV3::Foreign,
            Err(error) => {
                *error_destination = Some(error);
                return GameStepV3::Failed;
            }
        }
        self.try_claim_effect(source, destination)
    }

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

    /// Close only the same P-C game's admission and semantic root set. The
    /// original proof revalidates its fence, joined/retired root owners and
    /// observer drains without blocking while this game lock is held. This
    /// is not durable output, GPU completion or capture-sender execution
    /// evidence; those remain independently held by their original owners.
    pub(super) fn try_close_pc(
        self: &Arc<Self>,
        proof: &super::pc_hooks_lock_v3::PcGameClosureProofV3,
        error_destination: &mut Option<super::ArenaError>,
    ) -> GameStepV3 {
        if error_destination.is_some() {
            return GameStepV3::Failed;
        }
        if !proof.matches_game(self) {
            return GameStepV3::Foreign;
        }
        if self.abort_requested() {
            return GameStepV3::Aborted;
        }
        let mut state = match self.lock() {
            Ok(state) => state,
            Err(error) => return error,
        };
        if self.abort_requested() {
            return GameStepV3::Aborted;
        }
        if state.phase != PhaseV3::AdmissionSealed && state.phase != PhaseV3::Closed {
            return GameStepV3::Closed;
        }
        if !state.pending.is_empty()
            || !state.dispatching.is_empty()
            || state.effects.iter().any(|effect| effect.active)
            || state.pc_operations.iter().any(|operation| operation.active)
            || self.control.load(Ordering::SeqCst) & !CONTROL_FLAGS != 0
        {
            return GameStepV3::Busy;
        }
        let Some(supervisor) = state.supervisor.as_ref().and_then(Weak::upgrade) else {
            return GameStepV3::Foreign;
        };
        let Some(pipeline) = state.pipeline.as_ref() else {
            return GameStepV3::Foreign;
        };
        if !proof.matches_supervisor(&supervisor)
            || !proof.matches_pipeline(pipeline)
            || state.output.as_ref().and_then(Weak::upgrade).is_none()
            || !proof.matches_roots(&state.admitted)
        {
            return GameStepV3::Foreign;
        }
        match proof.try_revalidate_observers() {
            Ok(Some(true)) => {}
            Ok(None) => return GameStepV3::Busy,
            Ok(Some(false)) => return GameStepV3::Foreign,
            Err(error) => {
                *error_destination = Some(error);
                return GameStepV3::Failed;
            }
        }
        if state.phase == PhaseV3::Closed {
            return GameStepV3::Existing;
        }
        match self.control.compare_exchange(
            ADMISSION_SEALED,
            ADMISSION_SEALED | GAME_CLOSED,
            Ordering::SeqCst,
            Ordering::SeqCst,
        ) {
            Ok(_) => {}
            Err(actual) if actual & ABORT_REQUESTED != 0 => return GameStepV3::Aborted,
            Err(_) => return GameStepV3::Unknown,
        }
        state.phase = PhaseV3::Closed;
        GameStepV3::Ready
    }

    /// Install one guard directly into caller-retained custody. The same P-C
    /// proof is rechecked; this does not create a generic closure proof or
    /// attest output durability, physical device release or capture drain.
    pub(super) fn try_begin_publication_pc(
        self: &Arc<Self>,
        proof: &super::pc_hooks_lock_v3::PcGameClosureProofV3,
        destination: &mut Option<GamePublishGuardV3<super::pc_worker_v3::PcWorkerSupervisorV3>>,
        error_destination: &mut Option<super::ArenaError>,
    ) -> GameStepV3 {
        if error_destination.is_some() {
            return GameStepV3::Failed;
        }
        if destination.is_some() {
            return GameStepV3::Busy;
        }
        if !proof.matches_game(self) {
            return GameStepV3::Foreign;
        }
        if self.abort_requested() {
            return GameStepV3::Aborted;
        }
        let mut state = match self.lock() {
            Ok(state) => state,
            Err(error) => return error,
        };
        if state.phase != PhaseV3::Closed {
            return GameStepV3::Closed;
        }
        if !state.pending.is_empty()
            || !state.dispatching.is_empty()
            || state.effects.iter().any(|effect| effect.active)
            || state.pc_operations.iter().any(|operation| operation.active)
            || self.control.load(Ordering::SeqCst) & !CONTROL_FLAGS != 0
        {
            return GameStepV3::Busy;
        }
        let Some(supervisor) = state.supervisor.as_ref().and_then(Weak::upgrade) else {
            return GameStepV3::Foreign;
        };
        let Some(pipeline) = state.pipeline.as_ref() else {
            return GameStepV3::Foreign;
        };
        if !proof.matches_supervisor(&supervisor)
            || !proof.matches_pipeline(pipeline)
            || state.output.as_ref().and_then(Weak::upgrade).is_none()
            || !proof.matches_roots(&state.admitted)
        {
            return GameStepV3::Foreign;
        }
        match proof.try_revalidate_observers() {
            Ok(Some(true)) => {}
            Ok(None) => return GameStepV3::Busy,
            Ok(Some(false)) => return GameStepV3::Foreign,
            Err(error) => {
                *error_destination = Some(error);
                return GameStepV3::Failed;
            }
        }
        if self.abort_requested() {
            return GameStepV3::Aborted;
        }
        let owner = Arc::clone(self);
        state.phase = PhaseV3::Preparing;
        *destination = Some(GamePublishGuardV3 {
            owner,
            resolved: false,
        });
        GameStepV3::Ready
    }
}

impl GameAuthorityV3<super::own_worker_v3::OwnWorkerSupervisorV3> {
    /// Move the original received-stage proof and nonclone scope source into
    /// an already retained destination only after same-owner revalidation.
    /// Neither Busy nor a rejection consumes either original input slot.
    pub(super) fn try_issue_own_scope_source_in_slot(
        self: &Arc<Self>,
        proof_slot: &mut Option<super::own_worker_v3::OwnReceivedStageProofV3>,
        source_slot: &mut Option<super::teacher::v3_session::TeacherScopeEffectSourceV3>,
        supervisor: &super::own_worker_v3::OwnWorkerSupervisorV3,
        recovery: &super::own_worker_v3::OwnRootRecoveryV3,
        root: &RoventCollectionRootAdmissionV3,
        destination: &mut Option<GameEffectSourceV3<super::own_worker_v3::OwnWorkerSupervisorV3>>,
        error_destination: &mut Option<&'static str>,
    ) -> GameStepV3 {
        if error_destination.is_some() {
            return GameStepV3::Failed;
        }
        if destination.is_some() {
            return GameStepV3::Busy;
        }
        let (Some(proof), Some(source)) = (proof_slot.as_ref(), source_slot.as_ref()) else {
            return GameStepV3::Busy;
        };
        match proof.matches_game(self) {
            Some(true) => {}
            Some(false) => return GameStepV3::Foreign,
            None => return GameStepV3::Unknown,
        }
        if !proof.matches_admission(root) || !proof.matches_scope_source(source) {
            return GameStepV3::Foreign;
        }
        match proof.try_revalidate_with_guards(supervisor, recovery) {
            Ok(Some(true)) => {}
            Ok(None) => return GameStepV3::Busy,
            Ok(Some(false)) => return GameStepV3::Foreign,
            Err(error) => {
                *error_destination = Some(error);
                return GameStepV3::Failed;
            }
        }
        let owner = Arc::clone(self);
        let root = root.clone();
        let stage = Arc::clone(source.identity());
        // Both source slots were checked above. No fallible operation or
        // external callback occurs between their take and destination write.
        let proof = proof_slot
            .take()
            .expect("validated original received proof");
        let source = source_slot.take().expect("validated original scope source");
        *destination = Some(GameEffectSourceV3 {
            owner,
            root,
            kind: GameEffectKindV3::TeacherScope,
            stage,
            actor: None,
            original: GameEffectOriginalV3::OwnScope { proof, source },
        });
        GameStepV3::Ready
    }

    /// A source is not a dispatch/start permit. Revalidate its original
    /// received stage under the caller's existing guards before game claim.
    pub(super) fn try_claim_own_scope(
        self: &Arc<Self>,
        source: &GameEffectSourceV3<super::own_worker_v3::OwnWorkerSupervisorV3>,
        supervisor: &super::own_worker_v3::OwnWorkerSupervisorV3,
        recovery: &super::own_worker_v3::OwnRootRecoveryV3,
        destination: &mut Option<GameEffectClaimV3<super::own_worker_v3::OwnWorkerSupervisorV3>>,
        error_destination: &mut Option<&'static str>,
    ) -> GameStepV3 {
        if error_destination.is_some() {
            return GameStepV3::Failed;
        }
        let GameEffectOriginalV3::OwnScope {
            proof,
            source: original,
        } = &source.original
        else {
            return GameStepV3::Foreign;
        };
        if !Arc::ptr_eq(self, &source.owner) {
            return GameStepV3::Foreign;
        }
        match proof.matches_game(self) {
            Some(true) => {}
            Some(false) => return GameStepV3::Foreign,
            None => return GameStepV3::Unknown,
        }
        if !proof.matches_admission(&source.root)
            || !proof.matches_scope_source(original)
            || !original.matches_stage_identity(&source.stage)
        {
            return GameStepV3::Foreign;
        }
        match proof.try_revalidate_with_guards(supervisor, recovery) {
            Ok(Some(true)) => {}
            Ok(None) => return GameStepV3::Busy,
            Ok(Some(false)) => return GameStepV3::Foreign,
            Err(error) => {
                *error_destination = Some(error);
                return GameStepV3::Failed;
            }
        }
        self.try_claim_effect(source, destination)
    }

    /// Bind one actual received root and its original Ready actor into the
    /// caller-owned slot. Identity observation is not a start or join.
    pub(super) fn try_issue_own_actor_source_in_slot(
        self: &Arc<Self>,
        proof_slot: &mut Option<super::own_worker_v3::OwnReceivedActorProofV3>,
        source_slot: &mut Option<super::teacher::v3_session::TeacherActorEffectSourceV3>,
        supervisor: &super::own_worker_v3::OwnWorkerSupervisorV3,
        recovery: &super::own_worker_v3::OwnRootRecoveryV3,
        stage: &super::teacher::v3_session::TeacherSessionStagingV3,
        session: &super::teacher::v3_session::RoventCollectionTeacherSessionV3,
        root: &RoventCollectionRootAdmissionV3,
        destination: &mut Option<GameEffectSourceV3<super::own_worker_v3::OwnWorkerSupervisorV3>>,
        error_destination: &mut Option<&'static str>,
    ) -> GameStepV3 {
        if error_destination.is_some() {
            return GameStepV3::Failed;
        }
        if destination.is_some() {
            return GameStepV3::Busy;
        }
        let (Some(proof), Some(source)) = (proof_slot.as_ref(), source_slot.as_ref()) else {
            return GameStepV3::Busy;
        };
        match proof.matches_game(self) {
            Some(true) => {}
            Some(false) => return GameStepV3::Foreign,
            None => return GameStepV3::Unknown,
        }
        if !proof.matches_admission(root) || !proof.matches_actor_source(source) {
            return GameStepV3::Foreign;
        }
        match proof.try_revalidate_with_guards(supervisor, recovery, stage, session, source) {
            Ok(Some(true)) => {}
            Ok(None) => return GameStepV3::Busy,
            Ok(Some(false)) => return GameStepV3::Foreign,
            Err(error) => {
                *error_destination = Some(error);
                return GameStepV3::Failed;
            }
        }
        let owner = Arc::clone(self);
        let root = root.clone();
        let stage = Arc::clone(source.session_identity());
        let actor = Some(Arc::clone(source.actor_identity()));
        let proof = proof_slot
            .take()
            .expect("validated original received actor proof");
        let source = source_slot.take().expect("validated original actor source");
        *destination = Some(GameEffectSourceV3 {
            owner,
            root,
            kind: GameEffectKindV3::TeacherActor,
            stage,
            actor,
            original: GameEffectOriginalV3::OwnActor { proof, source },
        });
        GameStepV3::Ready
    }

    pub(super) fn try_claim_own_actor(
        self: &Arc<Self>,
        source: &GameEffectSourceV3<super::own_worker_v3::OwnWorkerSupervisorV3>,
        supervisor: &super::own_worker_v3::OwnWorkerSupervisorV3,
        recovery: &super::own_worker_v3::OwnRootRecoveryV3,
        stage: &super::teacher::v3_session::TeacherSessionStagingV3,
        session: &super::teacher::v3_session::RoventCollectionTeacherSessionV3,
        destination: &mut Option<GameEffectClaimV3<super::own_worker_v3::OwnWorkerSupervisorV3>>,
        error_destination: &mut Option<&'static str>,
    ) -> GameStepV3 {
        if error_destination.is_some() {
            return GameStepV3::Failed;
        }
        let GameEffectOriginalV3::OwnActor {
            proof,
            source: original,
        } = &source.original
        else {
            return GameStepV3::Foreign;
        };
        if !Arc::ptr_eq(self, &source.owner) {
            return GameStepV3::Foreign;
        }
        match proof.matches_game(self) {
            Some(true) => {}
            Some(false) => return GameStepV3::Foreign,
            None => return GameStepV3::Unknown,
        }
        if !proof.matches_admission(&source.root)
            || !proof.matches_actor_source(original)
            || !original.matches_session_identity(&source.stage)
            || !source
                .actor
                .as_ref()
                .is_some_and(|actor| original.matches_actor_identity(actor))
        {
            return GameStepV3::Foreign;
        }
        match proof.try_revalidate_with_guards(supervisor, recovery, stage, session, original) {
            Ok(Some(true)) => {}
            Ok(None) => return GameStepV3::Busy,
            Ok(Some(false)) => return GameStepV3::Foreign,
            Err(error) => {
                *error_destination = Some(error);
                return GameStepV3::Failed;
            }
        }
        self.try_claim_effect(source, destination)
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
    receipt_identity: Option<Arc<()>>,
    outcome: GameDispatchOutcomeV3,
    abort_observed: bool,
    seal_observed: bool,
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
    pub(super) fn seal_observed(&self) -> bool {
        self.seal_observed
    }
}

/// Binds one exact P-C non-Sent receipt and its original game/root. The game
/// state is reobserved on use; this is not a worker or physical fence.
pub(super) struct PcNotSentDispatchWitnessV3 {
    owner: Arc<GameAuthorityV3<super::pc_worker_v3::PcWorkerSupervisorV3>>,
    root: RoventCollectionRootAdmissionV3,
    receipt_identity: Arc<()>,
}

impl PcNotSentDispatchWitnessV3 {
    pub(super) fn try_matches_original(
        &self,
        game: &Arc<GameAuthorityV3<super::pc_worker_v3::PcWorkerSupervisorV3>>,
        root: &RoventCollectionRootAdmissionV3,
        envelope: &RoventCollectionRootAdmissionV3,
        receipt: &GameDispatchReceiptV3<super::pc_worker_v3::PcWorkerSupervisorV3>,
    ) -> GameStepV3 {
        if !Arc::ptr_eq(&self.owner, game)
            || !self.root.same_original_admission(root)
            || !receipt
                .receipt_identity
                .as_ref()
                .is_some_and(|identity| Arc::ptr_eq(identity, &self.receipt_identity))
        {
            return GameStepV3::Foreign;
        }
        game.try_check_pc_not_sent(root, envelope, receipt)
    }
}

/// Bounded aggregation of equivalent known-Full observations only. The
/// original game owner permanently binds one dispatch channel; abort/seal
/// observed, sent, disconnected and uncertain work never enter this summary.
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
        if next.outcome != GameDispatchOutcomeV3::Full || next.abort_observed || next.seal_observed
        {
            return GameStepV3::Foreign;
        }
        if let Some(original) = summary.as_ref() {
            if !Arc::ptr_eq(&original.last.owner, &next.owner)
                || !original.last.root.same_original_admission(&next.root)
                || !original.last.envelope.same_original(&next.envelope)
                || original.last.outcome != GameDispatchOutcomeV3::Full
                || original.last.abort_observed
                || original.last.seal_observed
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
    receipt_identity: Option<Arc<()>>,
    root_cancel_claim: Option<rz_uci::search_driver::RoventRootCancelClaimV3>,
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
            receipt_identity: self.receipt_identity.as_ref().map(Arc::clone),
            outcome,
            abort_observed: prior & ABORT_REQUESTED != 0 || self.owner.abort_requested(),
            seal_observed: prior & ADMISSION_SEALED != 0,
        });
        if !self
            .owner
            .resolve_pc_root_claim(&mut self.root_cancel_claim)
        {
            return GameStepV3::Unknown;
        }
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
    root_cancel_claim: Option<rz_uci::search_driver::RoventRootCancelClaimV3>,
    resolved: bool,
}
impl<S> GameEffectClaimV3<S> {
    /// Validates only the same pre-registered scope start authority. An
    /// acquired claim does not attest dispatch receipt or physical completion.
    pub(super) fn authorizes_teacher_scope(
        &self,
        source: &super::teacher::v3_session::TeacherScopeEffectSourceV3,
    ) -> bool {
        !self.resolved
            && !self.owner.unknown.load(Ordering::SeqCst)
            && self.kind == GameEffectKindV3::TeacherScope
            && self.actor.is_none()
            && source.matches_admission(&self.root)
            && source.matches_stage_identity(&self.stage)
    }

    /// Matches one original actor in one original teacher session; it does
    /// not attest queue receipt, actor readiness or physical completion.
    pub(super) fn authorizes_teacher_actor(
        &self,
        source: &super::teacher::v3_session::TeacherActorEffectSourceV3,
    ) -> bool {
        !self.resolved
            && !self.owner.unknown.load(Ordering::SeqCst)
            && self.kind == GameEffectKindV3::TeacherActor
            && source.matches_admission(&self.root)
            && source.matches_session_identity(&self.stage)
            && self
                .actor
                .as_ref()
                .is_some_and(|actor| source.matches_actor_identity(actor))
    }

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
        self.try_complete_verified()
    }

    fn try_complete_verified(&mut self) -> GameStepV3 {
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
        if !self
            .owner
            .resolve_pc_root_claim(&mut self.root_cancel_claim)
        {
            return GameStepV3::Unknown;
        }
        self.resolved = true;
        GameStepV3::Ready
    }
}

impl GameEffectClaimV3<super::pc_worker_v3::PcWorkerSupervisorV3> {
    /// Settle only the original scope's actual returned result. The outcome
    /// remains in caller custody; this is not a physical retirement proof.
    pub(super) fn try_complete_pc_scope(
        &mut self,
        source: &GameEffectSourceV3<super::pc_worker_v3::PcWorkerSupervisorV3>,
        outcome: &super::teacher::v3_session::TeacherScopeEffectOutcomeV3,
        supervisor: &super::pc_worker_v3::PcWorkerSupervisorV3,
        recovery: &super::pc_worker_v3::PcRootRecoveryV3,
        error_destination: &mut Option<&'static str>,
    ) -> GameStepV3 {
        if error_destination.is_some() {
            return GameStepV3::Failed;
        }
        if !Arc::ptr_eq(&self.owner, &source.owner)
            || !self.root.same_original_admission(&source.root)
            || self.kind != GameEffectKindV3::TeacherScope
            || source.kind != GameEffectKindV3::TeacherScope
            || !Arc::ptr_eq(&self.stage, &source.stage)
            || self.actor.is_some()
            || source.actor.is_some()
        {
            return GameStepV3::Foreign;
        }
        let GameEffectOriginalV3::PcScope {
            proof,
            source: original,
        } = &source.original
        else {
            return GameStepV3::Foreign;
        };
        match proof.matches_game(&self.owner) {
            Some(true) => {}
            Some(false) => return GameStepV3::Foreign,
            None => return GameStepV3::Unknown,
        }
        if !proof.matches_admission(&self.root)
            || !proof.matches_scope_source(original)
            || !original.matches_outcome(outcome)
        {
            return GameStepV3::Foreign;
        }
        match proof.try_revalidate_with_guards(supervisor, recovery) {
            Ok(Some(true)) => {}
            Ok(None) => return GameStepV3::Busy,
            Ok(Some(false)) => return GameStepV3::Foreign,
            Err(error) => {
                *error_destination = Some(error);
                return GameStepV3::Failed;
            }
        }
        self.try_complete_verified()
    }
    /// An actor result resolves only this attempted effect. It does not prove
    /// the spawned worker has retired or its physical resources are free.
    pub(super) fn try_complete_pc_actor(
        &mut self,
        source: &GameEffectSourceV3<super::pc_worker_v3::PcWorkerSupervisorV3>,
        outcome: &super::teacher::v3_session::TeacherActorEffectOutcomeV3,
        supervisor: &super::pc_worker_v3::PcWorkerSupervisorV3,
        recovery: &super::pc_worker_v3::PcRootRecoveryV3,
        stage: &super::teacher::v3_session::TeacherSessionStagingV3,
        session: &super::teacher::v3_session::RoventCollectionTeacherSessionV3,
        error_destination: &mut Option<&'static str>,
    ) -> GameStepV3 {
        if error_destination.is_some() {
            return GameStepV3::Failed;
        }
        if !Arc::ptr_eq(&self.owner, &source.owner)
            || !self.root.same_original_admission(&source.root)
            || self.kind != GameEffectKindV3::TeacherActor
            || source.kind != GameEffectKindV3::TeacherActor
            || !Arc::ptr_eq(&self.stage, &source.stage)
            || !same_effect_actor(&self.actor, &source.actor)
            || self.actor.is_none()
        {
            return GameStepV3::Foreign;
        }
        let GameEffectOriginalV3::PcActor {
            proof,
            source: original,
        } = &source.original
        else {
            return GameStepV3::Foreign;
        };
        match proof.matches_game(&self.owner) {
            Some(true) => {}
            Some(false) => return GameStepV3::Foreign,
            None => return GameStepV3::Unknown,
        }
        if !proof.matches_admission(&self.root)
            || !proof.matches_actor_source(original)
            || !original.matches_session_identity(&source.stage)
            || !source
                .actor
                .as_ref()
                .is_some_and(|actor| original.matches_actor_identity(actor))
            || !original.matches_outcome(outcome)
        {
            return GameStepV3::Foreign;
        }
        match proof.try_revalidate_with_guards(supervisor, recovery, stage, session, original) {
            Ok(Some(true)) => {}
            Ok(None) => return GameStepV3::Busy,
            Ok(Some(false)) => return GameStepV3::Foreign,
            Err(error) => {
                *error_destination = Some(error);
                return GameStepV3::Failed;
            }
        }
        self.try_complete_verified()
    }
}
impl GameEffectClaimV3<super::own_worker_v3::OwnWorkerSupervisorV3> {
    /// Settle only the original scope's actual returned result. The outcome
    /// remains in caller custody; this is not a physical retirement proof.
    pub(super) fn try_complete_own_scope(
        &mut self,
        source: &GameEffectSourceV3<super::own_worker_v3::OwnWorkerSupervisorV3>,
        outcome: &super::teacher::v3_session::TeacherScopeEffectOutcomeV3,
        supervisor: &super::own_worker_v3::OwnWorkerSupervisorV3,
        recovery: &super::own_worker_v3::OwnRootRecoveryV3,
        error_destination: &mut Option<&'static str>,
    ) -> GameStepV3 {
        if error_destination.is_some() {
            return GameStepV3::Failed;
        }
        if !Arc::ptr_eq(&self.owner, &source.owner)
            || !self.root.same_original_admission(&source.root)
            || self.kind != GameEffectKindV3::TeacherScope
            || source.kind != GameEffectKindV3::TeacherScope
            || !Arc::ptr_eq(&self.stage, &source.stage)
            || self.actor.is_some()
            || source.actor.is_some()
        {
            return GameStepV3::Foreign;
        }
        let GameEffectOriginalV3::OwnScope {
            proof,
            source: original,
        } = &source.original
        else {
            return GameStepV3::Foreign;
        };
        match proof.matches_game(&self.owner) {
            Some(true) => {}
            Some(false) => return GameStepV3::Foreign,
            None => return GameStepV3::Unknown,
        }
        if !proof.matches_admission(&self.root)
            || !proof.matches_scope_source(original)
            || !original.matches_outcome(outcome)
        {
            return GameStepV3::Foreign;
        }
        match proof.try_revalidate_with_guards(supervisor, recovery) {
            Ok(Some(true)) => {}
            Ok(None) => return GameStepV3::Busy,
            Ok(Some(false)) => return GameStepV3::Foreign,
            Err(error) => {
                *error_destination = Some(error);
                return GameStepV3::Failed;
            }
        }
        self.try_complete_verified()
    }
    /// An actor result resolves only this attempted effect. It does not prove
    /// the spawned worker has retired or its physical resources are free.
    pub(super) fn try_complete_own_actor(
        &mut self,
        source: &GameEffectSourceV3<super::own_worker_v3::OwnWorkerSupervisorV3>,
        outcome: &super::teacher::v3_session::TeacherActorEffectOutcomeV3,
        supervisor: &super::own_worker_v3::OwnWorkerSupervisorV3,
        recovery: &super::own_worker_v3::OwnRootRecoveryV3,
        stage: &super::teacher::v3_session::TeacherSessionStagingV3,
        session: &super::teacher::v3_session::RoventCollectionTeacherSessionV3,
        error_destination: &mut Option<&'static str>,
    ) -> GameStepV3 {
        if error_destination.is_some() {
            return GameStepV3::Failed;
        }
        if !Arc::ptr_eq(&self.owner, &source.owner)
            || !self.root.same_original_admission(&source.root)
            || self.kind != GameEffectKindV3::TeacherActor
            || source.kind != GameEffectKindV3::TeacherActor
            || !Arc::ptr_eq(&self.stage, &source.stage)
            || !same_effect_actor(&self.actor, &source.actor)
            || self.actor.is_none()
        {
            return GameStepV3::Foreign;
        }
        let GameEffectOriginalV3::OwnActor {
            proof,
            source: original,
        } = &source.original
        else {
            return GameStepV3::Foreign;
        };
        match proof.matches_game(&self.owner) {
            Some(true) => {}
            Some(false) => return GameStepV3::Foreign,
            None => return GameStepV3::Unknown,
        }
        if !proof.matches_admission(&self.root)
            || !proof.matches_actor_source(original)
            || !original.matches_session_identity(&source.stage)
            || !source
                .actor
                .as_ref()
                .is_some_and(|actor| original.matches_actor_identity(actor))
            || !original.matches_outcome(outcome)
        {
            return GameStepV3::Foreign;
        }
        match proof.try_revalidate_with_guards(supervisor, recovery, stage, session, original) {
            Ok(Some(true)) => {}
            Ok(None) => return GameStepV3::Busy,
            Ok(Some(false)) => return GameStepV3::Foreign,
            Err(error) => {
                *error_destination = Some(error);
                return GameStepV3::Failed;
            }
        }
        self.try_complete_verified()
    }
}
/// A single attempt on one pre-retained P-C operation slot. Unresolved
/// claims are never silently refunded on drop or treated as device fences.
pub(super) struct PcOperationClaimV3 {
    owner: Arc<GameAuthorityV3<super::pc_worker_v3::PcWorkerSupervisorV3>>,
    root: RoventCollectionRootAdmissionV3,
    kind: PcOperationKindV3,
    slot: Arc<()>,
    generation: u64,
    root_cancel_claim: Option<rz_uci::search_driver::RoventRootCancelClaimV3>,
    resolved: bool,
}

impl PcOperationClaimV3 {
    pub(super) fn generation(&self) -> u64 {
        self.generation
    }

    /// Reobserve the original producer result retained in this recovery. A
    /// no-effect result permits a new claim only after abort is checked again.
    pub(super) fn try_complete_pc_operation(
        &mut self,
        proof: &super::pc_worker_v3::PcReceivedStageProofV3,
        slot: &super::pc_worker_v3::PcOperationSlotV3,
        supervisor: &super::pc_worker_v3::PcWorkerSupervisorV3,
        recovery: &super::pc_worker_v3::PcRootRecoveryV3,
    ) -> GameStepV3 {
        if self.resolved {
            return GameStepV3::Existing;
        }
        if !self.root.same_original_admission(slot.admission())
            || self.kind != slot.kind()
            || !Arc::ptr_eq(&self.slot, slot.identity())
            || !proof.matches_admission(&self.root)
        {
            return GameStepV3::Foreign;
        }
        match proof.matches_game(&self.owner) {
            Some(true) => {}
            Some(false) => return GameStepV3::Foreign,
            None => return GameStepV3::Unknown,
        }
        match slot.try_revalidate_with_guards(proof, supervisor, recovery) {
            Ok(Some(true)) => {}
            Ok(Some(false)) => return GameStepV3::Foreign,
            Ok(None) => return GameStepV3::Busy,
            Err(_) => return GameStepV3::Unknown,
        }
        let (retryable, failed_no_effect) = match slot.observation(self.generation, recovery) {
            PcOperationObservationV3::Pending => return GameStepV3::Busy,
            PcOperationObservationV3::NoEffect => (true, false),
            PcOperationObservationV3::FailedNoEffect => {
                (false, self.kind == PcOperationKindV3::RouteInstall)
            }
            PcOperationObservationV3::Committed => (false, false),
            PcOperationObservationV3::Unknown => return GameStepV3::Unknown,
        };
        let mut state = match self.owner.lock() {
            Ok(state) => state,
            Err(error) => return error,
        };
        let Some(index) = state.pc_operations.iter().position(|operation| {
            operation.active
                && operation.root.same_original_admission(&self.root)
                && operation.kind == self.kind
                && Arc::ptr_eq(&operation.slot, &self.slot)
                && operation.generation == self.generation
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
        state.pc_operations[index].active = false;
        state.pc_operations[index].retryable = retryable;
        state.pc_operations[index].failed_no_effect = failed_no_effect;
        if !self
            .owner
            .resolve_pc_root_claim(&mut self.root_cancel_claim)
        {
            return GameStepV3::Unknown;
        }
        self.resolved = true;
        GameStepV3::Ready
    }
}

impl Drop for PcOperationClaimV3 {
    fn drop(&mut self) {
        if !self.resolved {
            self.owner.unknown.store(true, Ordering::SeqCst);
        }
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
            ADMISSION_SEALED | GAME_CLOSED,
            ADMISSION_SEALED | GAME_CLOSED | PUBLICATION_NOT_STARTED,
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
            ADMISSION_SEALED | GAME_CLOSED,
            ADMISSION_SEALED | GAME_CLOSED | NO_DATA_FINISHED,
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
            ADMISSION_SEALED | GAME_CLOSED,
            ADMISSION_SEALED | GAME_CLOSED | PUBLICATION_GRANTED,
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
