//! Exclusive native-session archive lifecycle, off unless an original owned
//! registration was installed. The original semantic owner moves through the
//! existing one-slot worker; no worker result is a native physical fence.

use super::launch::storage::RoventStorageSetupRejection;
use super::product::SharedRoventStorageRegistration;
use crate::rovent_root_admission::RoventRootAdmission;
use crate::search_driver::{DriverPhysicalCompletion, SearchSessionFailure};
use rz_contracts::RootGeneration;
use rz_contracts::pals::SearchAuthority;
use rz_position::Position;
use rz_runtime::CompletionSignal;
use rz_search::pals::engine::adaptive_port::AdaptivePortModel;
use rz_search::pals::engine::adaptive_port::archive_worker::{
    ArchiveOperation, ArchiveOperationKind, ArchiveOwner, ArchiveQuarantine,
    ArchiveShutdownReceipt, ArchiveTransfer, ArchiveWork, ArchiveWorkReceipt, ArchiveWorker,
    ArchiveWorkerError, UnusedIdleArchiveSetupCompletion, UnusedIdleArchiveSetupReceipt,
};
use rz_search::pals::engine::idle_archive::{
    IdleArchiveOwnerKind, IdleSemanticArchiveOwner, UnusedIdleArchiveSetup,
    UnusedIdleArchiveSetupIdentity,
};
use rz_search::pals::engine::preparing_root::ResearcherPreparingOwner;
use rz_search::pals::engine::{PalsEngine, PcSemanticOwner, RoleModel};
use std::any::Any;
use std::collections::VecDeque;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::time::{Duration, Instant};

const MAINTENANCE_IO_BYTES: u64 = 16 * 1024 * 1024;
const RECEIPTS_MAX: usize = 64;

mod observation;
pub(crate) use observation::StorageLifecycleObservation;

pub(crate) enum StorageSemanticOwner<M: RoleModel + 'static = AdaptivePortModel> {
    Pc(Box<PcSemanticOwner<M>>),
    Pcr(Box<ResearcherPreparingOwner<M>>),
    EnginePcr(Box<PalsEngine<M>>),
}

pub(crate) struct StorageLifecycleFailure<M: RoleModel + 'static = AdaptivePortModel> {
    /// Present only when submission never took the incoming original owner.
    /// The caller must put it back in its same-kind retained session slot.
    pub(crate) owner: Option<StorageSemanticOwner<M>>,
    pub(crate) failure: SearchSessionFailure,
}
impl<M: RoleModel + 'static> From<SearchSessionFailure> for StorageLifecycleFailure<M> {
    fn from(failure: SearchSessionFailure) -> Self {
        Self {
            owner: None,
            failure,
        }
    }
}
impl<M: RoleModel + 'static> StorageSemanticOwner<M> {
    fn kind(&self) -> IdleArchiveOwnerKind {
        match self {
            Self::Pc(_) => IdleArchiveOwnerKind::Pc,
            Self::Pcr(_) => IdleArchiveOwnerKind::Researcher,
            Self::EnginePcr(_) => IdleArchiveOwnerKind::EnginePcr,
        }
    }
    fn into_idle(
        self,
    ) -> Result<IdleSemanticArchiveOwner<M>, (Self, rz_search::pals::engine::PalsError)> {
        match self {
            Self::Pc(owner) => owner
                .into_idle_archive()
                .map_err(|(owner, error)| (Self::Pc(owner), error)),
            Self::Pcr(owner) => owner
                .into_idle_archive()
                .map_err(|(owner, error)| (Self::Pcr(owner), error)),
            Self::EnginePcr(owner) => owner
                .into_idle_archive()
                .map_err(|(owner, error)| (Self::EnginePcr(owner), error)),
        }
    }
}

#[derive(Clone, Copy)]
enum EmptyGamePhaseKind {
    UnusedSetup,
    NewGame,
}

/// An observed fresh game Store has no cold segment in its own unique archive
/// directory. This is not a fabricated RestoreRootIfArchived(None) receipt.
struct EmptyGamePhase {
    kind: EmptyGamePhaseKind,
    requested_scope: SearchAuthority,
    owner: rz_search::pals::store::ArchiveOwnerSnapshot,
}

enum Pending {
    Setup {
        transfer: ArchiveTransfer,
        scope: SearchAuthority,
        identity: UnusedIdleArchiveSetupIdentity,
        cleanup_until: Instant,
    },
    Work {
        transfer: ArchiveTransfer,
        authority: SearchAuthority,
        kind: ArchiveOperationKind,
        cleanup_until: Instant,
    },
}
impl Pending {
    fn transfer(&self) -> &ArchiveTransfer {
        match self {
            Self::Setup { transfer, .. } | Self::Work { transfer, .. } => transfer,
        }
    }
    fn cleanup_until(&self) -> Instant {
        match self {
            Self::Setup { cleanup_until, .. } | Self::Work { cleanup_until, .. } => *cleanup_until,
        }
    }
}

/// Retained on every failed caller path. A pending owner lives in the original
/// worker/orphan slot, never in a replacement Store or a newly opened archive.
struct WorkerCustody<M: RoleModel + 'static = AdaptivePortModel> {
    worker: Option<ArchiveWorker<M>>,
    original: Option<StorageSemanticOwner<M>>,
    idle: Option<IdleSemanticArchiveOwner<M>>,
    unexpected: Option<Box<ArchiveOwner<M>>>,
    setup: Option<UnusedIdleArchiveSetup>,
    setup_admission_rejection: Option<RoventStorageSetupRejection>,
    setup_rejected: Option<Box<UnusedIdleArchiveSetupCompletion<M>>>,
    quarantine: Option<Box<ArchiveQuarantine<M>>>,
    rejected_work: Option<Box<ArchiveWork>>,
    pending: Option<Pending>,
    setup_receipt: Option<UnusedIdleArchiveSetupReceipt>,
    receipts: VecDeque<ArchiveWorkReceipt>,
    receipt_sequence: u64,
    phase_receipt_start: u64,
    observation_sequence_exhausted: bool,
    shutdown: Option<ArchiveShutdownReceipt>,
    shutdown_observed: Option<ArchiveShutdownReceipt>,
    phase_scope: Option<SearchAuthority>,
    phase_joins: VecDeque<(Option<SearchAuthority>, ArchiveShutdownReceipt)>,
    join_error: Option<ArchiveWorkerError>,
    poll_error: Option<ArchiveWorkerError>,
    operation_error: Option<rz_search::pals::engine::PalsError>,
    port_error: Option<crate::pals_adaptive::AdaptiveManagerError>,
    caller_panic: Option<Box<dyn Any + Send>>,
    first_failure: Option<SearchSessionFailure>,
    last_scope: Option<SearchAuthority>,
    last_registered_scope: Option<SearchAuthority>,
    empty_game_phases: VecDeque<EmptyGamePhase>,
    original_kind: Option<IdleArchiveOwnerKind>,
    finish_completed: bool,
}
impl<M: RoleModel + 'static> WorkerCustody<M> {
    fn new() -> Self {
        Self {
            worker: None,
            original: None,
            idle: None,
            unexpected: None,
            setup: None,
            setup_admission_rejection: None,
            setup_rejected: None,
            quarantine: None,
            rejected_work: None,
            pending: None,
            setup_receipt: None,
            receipts: VecDeque::new(),
            receipt_sequence: 0,
            phase_receipt_start: 0,
            observation_sequence_exhausted: false,
            shutdown: None,
            shutdown_observed: None,
            phase_scope: None,
            phase_joins: VecDeque::new(),
            join_error: None,
            poll_error: None,
            operation_error: None,
            port_error: None,
            caller_panic: None,
            first_failure: None,
            last_scope: None,
            last_registered_scope: None,
            empty_game_phases: VecDeque::new(),
            original_kind: None,
            finish_completed: false,
        }
    }
    fn fail(&mut self, code: &'static str, detail: impl std::fmt::Debug) -> SearchSessionFailure {
        let failure = SearchSessionFailure {
            physical_completion: DriverPhysicalCompletion::Unknown,
            code,
            detail: format!("{detail:?}"),
        };
        self.first_failure.get_or_insert(failure).clone()
    }
    fn operation_failure(
        &mut self,
        code: &'static str,
        error: rz_search::pals::engine::PalsError,
    ) -> SearchSessionFailure {
        let failure = self.fail(code, &error);
        self.operation_error.get_or_insert(error);
        failure
    }
    fn ensure_worker(&mut self, signal: &CompletionSignal) -> Result<(), SearchSessionFailure> {
        if self.worker.is_none() {
            let signal = signal.clone();
            match ArchiveWorker::start_with_notifier(move || signal.notify()) {
                Ok(worker) => {
                    self.worker = Some(worker);
                    self.shutdown = None;
                    self.phase_scope = None;
                    self.phase_receipt_start = self.receipt_sequence;
                }
                Err(error) => return Err(self.fail("RoventStorageWorker", error)),
            }
        }
        Ok(())
    }
    fn adopt(
        &mut self,
        original: StorageSemanticOwner<M>,
    ) -> Result<(), (StorageSemanticOwner<M>, SearchSessionFailure)> {
        // The caller moves exactly one owner only after the prior root fence.
        // Even a violated invariant retains the incoming owner without a drop.
        if self.original.is_some()
            || self.idle.is_some()
            || self.pending.is_some()
            || self.setup_rejected.is_some()
            || self.quarantine.is_some()
            || self.unexpected.is_some()
        {
            return Err((
                original,
                self.fail(
                    "RoventStorageRetained",
                    "previous archive owner has not returned",
                ),
            ));
        }
        let kind = original.kind();
        self.original = Some(original);
        if self.original_kind.is_some_and(|expected| expected != kind) {
            let original = self.original.take().expect("same rejected semantic owner");
            return Err((
                original,
                self.fail("RoventStorageOwnerKind", "semantic owner kind changed"),
            ));
        }
        self.original_kind = Some(kind);
        match self
            .original
            .take()
            .expect("original semantic owner")
            .into_idle()
        {
            Ok(idle) => {
                self.last_registered_scope = idle.last_registered_scope();
                self.idle = Some(idle);
            }
            Err((original, error)) => {
                return Err((original, self.operation_failure("RoventStorageIdle", error)));
            }
        }
        Ok(())
    }
    fn restore(&mut self) -> Result<StorageSemanticOwner<M>, SearchSessionFailure> {
        if let Some(failure) = &self.first_failure {
            return Err(failure.clone());
        }
        let Some(idle) = self.idle.take() else {
            return Err(self.fail("RoventStorageReturn", "original Idle owner absent"));
        };
        let returned = match self.original_kind {
            Some(IdleArchiveOwnerKind::Pc) => idle.into_pc().map(StorageSemanticOwner::Pc),
            Some(IdleArchiveOwnerKind::Researcher) => {
                idle.into_researcher().map(StorageSemanticOwner::Pcr)
            }
            Some(IdleArchiveOwnerKind::EnginePcr) => {
                idle.into_engine().map(StorageSemanticOwner::EnginePcr)
            }
            None => Err(idle),
        };
        match returned {
            Ok(owner) => Ok(owner),
            Err(idle) => {
                self.idle = Some(idle);
                Err(self.fail("RoventStorageReturn", "original kind cannot be restored"))
            }
        }
    }
    fn submit_setup(
        &mut self,
        setup: UnusedIdleArchiveSetup,
        scope: SearchAuthority,
        cleanup_until: Instant,
    ) -> Result<(), SearchSessionFailure> {
        self.setup = Some(setup);
        let owner = self.idle.take().expect("retained original Idle");
        let setup = self.setup.take().expect("retained original setup");
        let identity = setup.identity();
        self.phase_scope = Some(scope);
        match self
            .worker
            .as_mut()
            .expect("original worker")
            .submit_unused_idle_setup(owner, scope, setup)
        {
            Ok(transfer) => {
                self.pending = Some(Pending::Setup {
                    transfer,
                    scope,
                    identity,
                    cleanup_until,
                })
            }
            Err(rejected) => {
                self.idle = Some(rejected.owner);
                self.setup = Some(rejected.setup);
                return Err(self.operation_failure("RoventStorageSetupSubmission", rejected.error));
            }
        }
        Ok(())
    }
    fn submit_work(
        &mut self,
        next: SearchAuthority,
        operation: ArchiveOperation,
        kind: ArchiveOperationKind,
        until: Instant,
        cleanup_until: Instant,
    ) -> Result<(), SearchSessionFailure> {
        let authority = if kind == ArchiveOperationKind::FinishIdleGame {
            // Finish has no later root and must not invent a root/nonce merely
            // to pass the ordinary next-root namespace validator. The existing
            // worker seal checks this actual last registered game and current
            // Store generation, including any attested root retirement.
            let Some(last) = self.last_registered_scope else {
                return Err(self.fail("RoventStorageAuthority", "no registered game to finish"));
            };
            if next != last {
                return Err(self.fail(
                    "RoventStorageAuthority",
                    "finish scope is not original last",
                ));
            }
            SearchAuthority {
                root: RootGeneration(
                    self.idle
                        .as_ref()
                        .expect("original Idle")
                        .stores()
                        .generation(),
                ),
                ..last
            }
        } else {
            match self
                .idle
                .as_ref()
                .expect("original Idle")
                .adaptive_archive_authority(next)
            {
                Ok(authority) => authority,
                Err(error) => return Err(self.operation_failure("RoventStorageAuthority", error)),
            }
        };
        let owner = self.idle.take().expect("original Idle");
        self.phase_scope.get_or_insert(next);
        match self.worker.as_mut().expect("original worker").submit(
            ArchiveOwner::Idle(owner),
            authority,
            ArchiveWork {
                operation,
                deadline: until,
                max_io_bytes: MAINTENANCE_IO_BYTES,
            },
        ) {
            Ok(transfer) => {
                self.pending = Some(Pending::Work {
                    transfer,
                    authority,
                    kind,
                    cleanup_until,
                })
            }
            Err(rejected) => {
                self.unexpected = Some(Box::new(rejected.owner));
                self.rejected_work = Some(Box::new(rejected.work));
                return Err(self.operation_failure("RoventStorageSubmission", rejected.error));
            }
        }
        Ok(())
    }
    fn push_receipt(&mut self, receipt: ArchiveWorkReceipt) {
        match self.receipt_sequence.checked_add(1) {
            Some(next) => self.receipt_sequence = next,
            None => self.observation_sequence_exhausted = true,
        }
        if self.receipts.len() == RECEIPTS_MAX {
            self.receipts.pop_front();
        }
        self.receipts.push_back(receipt);
    }
    fn pump(&mut self) {
        match self.pending.as_ref() {
            Some(Pending::Setup {
                transfer,
                scope,
                identity,
                ..
            }) => {
                let scope = *scope;
                let identity = *identity;
                let nonce = transfer.nonce();
                match self
                    .worker
                    .as_mut()
                    .expect("original worker")
                    .try_receive_unused_idle_setup(transfer)
                {
                    Ok(Some(completed)) => {
                        self.pending.take();
                        match completed.install(scope, nonce, identity) {
                            Ok((owner, receipt)) => {
                                self.idle = Some(owner);
                                self.setup_receipt = Some(receipt);
                            }
                            Err(completed) => {
                                let error = format!("{:?}", completed.failure());
                                self.setup_rejected = Some(Box::new(completed));
                                self.fail("RoventStorageSetupReturn", error);
                            }
                        }
                    }
                    Ok(None) => {}
                    Err(error) => {
                        self.poll_error.get_or_insert(error);
                        self.fail("RoventStorageSetupPoll", error);
                    }
                }
            }
            Some(Pending::Work {
                transfer,
                authority,
                kind,
                ..
            }) => {
                let authority = *authority;
                let kind = *kind;
                let nonce = transfer.nonce();
                match self
                    .worker
                    .as_mut()
                    .expect("original worker")
                    .try_receive(transfer)
                {
                    Ok(Some(completed)) => {
                        self.pending.take();
                        match completed.install(authority, nonce) {
                            Ok((ArchiveOwner::Idle(owner), receipt)) => {
                                self.idle = Some(owner);
                                if receipt.operation != kind {
                                    self.push_receipt(receipt);
                                    self.fail(
                                        "RoventStorageOperation",
                                        "returned operation differs",
                                    );
                                } else {
                                    self.finish_completed |=
                                        kind == ArchiveOperationKind::FinishIdleGame;
                                    self.push_receipt(receipt);
                                }
                            }
                            Ok((owner, receipt)) => {
                                self.unexpected = Some(Box::new(owner));
                                self.push_receipt(receipt);
                                self.fail(
                                    "RoventStorageOwnerKind",
                                    "worker returned a non-Idle owner",
                                );
                            }
                            Err(quarantine) => {
                                let error = format!("{:?}", quarantine.reason);
                                self.quarantine = Some(Box::new(quarantine));
                                self.fail("RoventStorageWorkReturn", error);
                            }
                        }
                    }
                    Ok(None) => {}
                    Err(error) => {
                        self.poll_error.get_or_insert(error);
                        self.fail("RoventStorageWorkPoll", error);
                    }
                }
            }
            None => {}
        }
    }
    fn cancel_pending(&mut self) {
        if let Some(pending) = &self.pending {
            pending.transfer().cancel();
        }
        if let Some(worker) = &mut self.worker {
            worker.request_shutdown();
        }
    }
    fn wait(
        &mut self,
        until: Instant,
        cancel: &AtomicBool,
        signal: &CompletionSignal,
    ) -> Result<(), SearchSessionFailure> {
        while self.pending.is_some() {
            let observed = signal.version();
            self.pump();
            if self.pending.is_none() {
                break;
            }
            if self.first_failure.is_some()
                || cancel.load(Ordering::Acquire)
                || Instant::now() >= until
            {
                self.cancel_pending();
                return Err(self.fail(
                    "RoventStorageOriginalControl",
                    "original go canceled/expired or completion unknown",
                ));
            }
            signal.wait_changed(
                observed,
                until
                    .saturating_duration_since(Instant::now())
                    .min(Duration::from_millis(1)),
            );
        }
        self.first_failure
            .as_ref()
            .map_or(Ok(()), |error| Err(error.clone()))
    }
    // Preserve the original work/cleanup clocks and cancellation as separate
    // arguments rather than constructing another control or implicit budget.
    #[allow(clippy::too_many_arguments)]
    fn work(
        &mut self,
        scope: SearchAuthority,
        operation: ArchiveOperation,
        kind: ArchiveOperationKind,
        until: Instant,
        cleanup_until: Instant,
        cancel: &AtomicBool,
        signal: &CompletionSignal,
    ) -> Result<(), SearchSessionFailure> {
        self.submit_work(scope, operation, kind, until, cleanup_until)?;
        self.wait(until, cancel, signal)
    }
    fn observe_empty_game(
        &mut self,
        kind: EmptyGamePhaseKind,
        scope: SearchAuthority,
    ) -> Result<(), SearchSessionFailure> {
        let owner = self.idle.as_ref().expect("original Idle");
        let snapshot = match owner.archive_owner_snapshot() {
            Ok(Some(snapshot)) => snapshot,
            Ok(None) => return Err(self.fail("RoventStorageEmptyGame", "archive owner absent")),
            Err(error) => return Err(self.operation_failure("RoventStorageEmptyGame", error)),
        };
        if owner.stores().generation() != 0
            || owner.stores().root().is_some()
            || owner.stores().hot_stats() != Default::default()
            || !snapshot.complete
            || !snapshot.root_boundary_checked
            || snapshot.admission_closed
            || snapshot.lifecycle != rz_search::pals::store::ArchiveOwnerLifecycle::Open
            || snapshot.generation != 0
            || snapshot.reserved_generations != 0
            || snapshot.committed_chunks != 0
        {
            return Err(self.fail("RoventStorageEmptyGame", snapshot));
        }
        if self.empty_game_phases.len() == RECEIPTS_MAX {
            self.empty_game_phases.pop_front();
        }
        self.empty_game_phases.push_back(EmptyGamePhase {
            kind,
            requested_scope: scope,
            owner: snapshot,
        });
        Ok(())
    }
    fn verify_empty_game_handoff(
        &mut self,
        scope: SearchAuthority,
    ) -> Result<(), SearchSessionFailure> {
        let Some(observed) = self.empty_game_phases.back() else {
            return Err(self.fail(
                "RoventStorageEmptyGame",
                "original empty-game observation absent",
            ));
        };
        let owner_id = observed.owner.owner_id;
        if observed.requested_scope != scope
            || !matches!(
                observed.kind,
                EmptyGamePhaseKind::UnusedSetup | EmptyGamePhaseKind::NewGame
            )
        {
            return Err(self.fail("RoventStorageEmptyGame", "empty-game scope differs"));
        }
        let Some(idle) = self.idle.as_ref() else {
            return Err(self.fail("RoventStorageEmptyGame", "joined original owner absent"));
        };
        match idle.archive_owner_snapshot() {
            Ok(Some(current))
                if current.owner_id == owner_id
                    && current.complete
                    && !current.admission_closed
                    && idle.stores().generation() == 0
                    && idle.stores().root().is_none()
                    && idle.stores().hot_stats() == Default::default() =>
            {
                Ok(())
            }
            Ok(current) => Err(self.fail("RoventStorageEmptyGame", current)),
            Err(error) => Err(self.operation_failure("RoventStorageEmptyGame", error)),
        }
    }
    fn join(
        &mut self,
        until: Instant,
        signal: &CompletionSignal,
    ) -> Result<(), SearchSessionFailure> {
        if self.shutdown.is_some() {
            return Ok(());
        }
        if self.worker.is_none() {
            return Ok(());
        }
        let until = self
            .pending
            .as_ref()
            .map_or(until, |pending| until.min(pending.cleanup_until()));
        self.cancel_pending();
        loop {
            let observed = signal.version();
            self.pump();
            if self.pending.is_none() {
                match self.worker.as_mut().expect("original worker").try_join() {
                    Ok(Some(receipt))
                        if receipt.worker_joined
                            && receipt.owners_returned
                            && receipt.admission_closed =>
                    {
                        self.shutdown = Some(receipt);
                        self.shutdown_observed = Some(receipt);
                        if self.phase_joins.len() == RECEIPTS_MAX {
                            self.phase_joins.pop_front();
                        }
                        self.phase_joins.push_back((self.phase_scope, receipt));
                        return Ok(());
                    }
                    Ok(Some(receipt)) => {
                        self.shutdown_observed = Some(receipt);
                        return Err(self.fail("RoventStorageJoin", receipt));
                    }
                    Err(error) => {
                        self.join_error.get_or_insert(error);
                        return Err(self.fail("RoventStorageJoin", error));
                    }
                    Ok(None) => {}
                }
            }
            if Instant::now() >= until {
                return Err(self.fail("RoventStorageJoin", "original cleanup deadline exhausted"));
            }
            signal.wait_changed(
                observed,
                until
                    .saturating_duration_since(Instant::now())
                    .min(Duration::from_millis(1)),
            );
        }
    }
}

/// One optional archive helper per actual native session. The shared original
/// registration remains held by this value and the product until full closure.
pub(crate) struct SessionStorageLifecycle<M: RoleModel + 'static = AdaptivePortModel> {
    registration: Arc<SharedRoventStorageRegistration>,
    core: WorkerCustody<M>,
    active_admission: Option<Arc<RoventRootAdmission>>,
}
impl<M: RoleModel + 'static> SessionStorageLifecycle<M> {
    /// Read-only bounded facts from the original installed receipts. No owner,
    /// archive I/O allowance or completion authority is transferred.
    pub(crate) fn observation(&self) -> StorageLifecycleObservation {
        observation::project(&self.core)
    }
    pub(crate) fn new(registration: Arc<SharedRoventStorageRegistration>) -> Box<Self> {
        Box::new(Self {
            registration,
            core: WorkerCustody::new(),
            active_admission: None,
        })
    }
    pub(crate) fn configured(&self) -> bool {
        self.core.setup_receipt.is_some()
    }
    fn retain_panic(&mut self, payload: Box<dyn Any + Send>) -> StorageLifecycleFailure<M> {
        self.core.caller_panic.get_or_insert(payload);
        self.core
            .fail(
                "RoventStorageCallerPanic",
                "original panic and worker/semantic custody retained",
            )
            .into()
    }
    pub(crate) fn abort_and_join(
        &mut self,
        original: SearchSessionFailure,
        until: Instant,
        signal: &CompletionSignal,
    ) -> Result<(), SearchSessionFailure> {
        self.core.first_failure.get_or_insert(original);
        let _ = self.core.join(until, signal);
        Err(self
            .core
            .first_failure
            .as_ref()
            .expect("original archive failure")
            .clone())
    }
    pub(crate) fn retain_wrong_return(&mut self, owner: StorageSemanticOwner<M>) {
        // Only a returned same-call value enters this slot; restore removed
        // the prior value, and the session closes before any subsequent go.
        debug_assert!(self.core.original.is_none());
        self.core.original = Some(owner);
        self.core.fail(
            "RoventStorageOwnerKind",
            "returned semantic kind differs from session",
        );
    }
    pub(crate) fn retain_port_error(
        &mut self,
        error: crate::pals_adaptive::AdaptiveManagerError,
    ) -> SearchSessionFailure {
        let failure = self.core.fail("RoventStoragePortFence", &error);
        self.core.port_error.get_or_insert(error);
        failure
    }
    pub(crate) fn prepare(
        &mut self,
        owner: StorageSemanticOwner<M>,
        admission: Arc<RoventRootAdmission>,
        reset: bool,
        position: &Position,
        signal: &CompletionSignal,
    ) -> Result<StorageSemanticOwner<M>, StorageLifecycleFailure<M>> {
        match catch_unwind(AssertUnwindSafe(|| {
            self.prepare_inner(owner, admission, reset, position, signal)
        })) {
            Ok(result) => result,
            Err(payload) => Err(self.retain_panic(payload)),
        }
    }
    fn prepare_inner(
        &mut self,
        owner: StorageSemanticOwner<M>,
        admission: Arc<RoventRootAdmission>,
        reset: bool,
        position: &Position,
        signal: &CompletionSignal,
    ) -> Result<StorageSemanticOwner<M>, StorageLifecycleFailure<M>> {
        self.active_admission = Some(Arc::clone(&admission));
        self.core
            .adopt(owner)
            .map_err(|(owner, failure)| StorageLifecycleFailure {
                owner: Some(owner),
                failure,
            })?;
        if let Some(failure) = &self.core.first_failure {
            return Err(failure.clone().into());
        }
        let context = admission.original_context();
        let scope = admission.authority();
        if let Err(error) = self
            .registration
            .profile()
            .validate_for_manifest(&self.registration.input_lock_owner().manifest)
        {
            return Err(self.core.fail("RoventStorageResources", error).into());
        }
        self.core.ensure_worker(signal)?;
        let fresh_game = if !self.configured() {
            let setup = match self.registration.take_unused_setup(Arc::clone(&admission)) {
                Ok(setup) => setup,
                Err(rejected) => {
                    let detail = rejected.error.to_string();
                    self.core.setup_admission_rejection = Some(rejected);
                    return Err(self.core.fail("RoventStorageSetupAdmission", detail).into());
                }
            };
            if setup.identity() != self.registration.identity() {
                self.core.setup = Some(setup);
                return Err(self
                    .core
                    .fail(
                        "RoventStorageSetupIdentity",
                        "original registration attribution differs",
                    )
                    .into());
            }
            self.core
                .submit_setup(setup, scope, context.shutdown_deadline)?;
            self.core.wait(
                context.control.deadline,
                &context.control.cancellation,
                signal,
            )?;
            Some(EmptyGamePhaseKind::UnusedSetup)
        } else if reset {
            self.core.work(
                scope,
                ArchiveOperation::NewGame {
                    next_authority: scope,
                },
                ArchiveOperationKind::NewGame,
                context.control.deadline,
                context.shutdown_deadline,
                &context.control.cancellation,
                signal,
            )?;
            Some(EmptyGamePhaseKind::NewGame)
        } else {
            None
        };
        if let Some(kind) = fresh_game {
            // Setup and a completed NewGame retain new_game_ready and a unique,
            // empty game archive. No registered last authority exists for this
            // game yet. Observe that state, then hand off to the real original
            // root registration; do not forge a maintenance/read result.
            self.core.observe_empty_game(kind, scope)?;
        } else {
            self.core.work(
                scope,
                ArchiveOperation::PrepareRoot {
                    position: position.clone(),
                },
                ArchiveOperationKind::PrepareRoot,
                context.control.deadline,
                context.shutdown_deadline,
                &context.control.cancellation,
                signal,
            )?;
            self.core.work(
                scope,
                ArchiveOperation::RestoreRootIfArchived {
                    position: position.clone(),
                },
                ArchiveOperationKind::RestoreRootIfArchived,
                context.control.deadline,
                context.shutdown_deadline,
                &context.control.cancellation,
                signal,
            )?;
        }
        self.core.last_scope = Some(scope);
        // The existing NNUEC root port owns its own single archive worker.
        // Return the semantic owner only after this idle phase actually joined;
        // the two physical helpers must never overlap under archive=1.
        self.core.join(context.shutdown_deadline, signal)?;
        self.core.worker.take();
        if fresh_game.is_some() {
            self.core.verify_empty_game_handoff(scope)?;
        }
        if context.control.cancellation.load(Ordering::Acquire)
            || Instant::now() >= context.control.deadline
        {
            return Err(self
                .core
                .fail(
                    "RoventStorageOriginalControl",
                    "original go stopped before archive handoff",
                )
                .into());
        }
        self.core.restore().map_err(Into::into)
    }
    /// Caller proves its original actor/native root fence before lending the
    /// owner. A final closed Idle remains retained here and is never restored
    /// as a searchable PC/PCR owner. Join is an independent actual observation.
    pub(crate) fn finish(
        &mut self,
        owner: Option<StorageSemanticOwner<M>>,
        until: Instant,
        signal: &CompletionSignal,
    ) -> Result<(), StorageLifecycleFailure<M>> {
        match catch_unwind(AssertUnwindSafe(|| self.finish_inner(owner, until, signal))) {
            Ok(result) => result,
            Err(payload) => Err(self.retain_panic(payload)),
        }
    }
    fn finish_inner(
        &mut self,
        owner: Option<StorageSemanticOwner<M>>,
        until: Instant,
        signal: &CompletionSignal,
    ) -> Result<(), StorageLifecycleFailure<M>> {
        if self.core.finish_completed
            && self.core.shutdown.is_some()
            && self.core.first_failure.is_none()
        {
            if let Some(owner) = owner {
                return Err(StorageLifecycleFailure {
                    owner: Some(owner),
                    failure: self.core.fail(
                        "RoventStorageClosed",
                        "a semantic owner was supplied after archive join",
                    ),
                });
            }
            return Ok(());
        }
        if let Some(owner) = owner {
            if self.core.first_failure.is_none() && !self.configured() {
                self.core.original = Some(owner); // no admitted go, no archive worker/IO
                return Ok(());
            }
            self.core
                .adopt(owner)
                .map_err(|(owner, failure)| StorageLifecycleFailure {
                    owner: Some(owner),
                    failure,
                })?;
        }
        if self.core.first_failure.is_none() && self.configured() && !self.core.finish_completed {
            self.core.ensure_worker(signal)?;
            let Some(scope) = self.core.last_scope else {
                return Err(self
                    .core
                    .fail("RoventStorageFinish", "original scope absent")
                    .into());
            };
            let _ = self.core.work(
                scope,
                ArchiveOperation::FinishIdleGame,
                ArchiveOperationKind::FinishIdleGame,
                until,
                until,
                &AtomicBool::new(false),
                signal,
            );
        }
        let joined = self.core.join(until, signal);
        if let Some(failure) = &self.core.first_failure {
            return Err(failure.clone().into());
        }
        joined.map_err(Into::into)
    }
}

#[cfg(test)]
mod tests {
    //! These exercise the existing Search worker with a real fixture-owned
    //! directory capability. They do not manufacture an opaque process/root
    //! admission, start a product, request NN inference or attest GPU readiness.
    use super::*;
    use rz_contracts::{Digest, GameGeneration, PrecisionProfile, ProcessEpoch, RootGeneration};
    use rz_search::pals::engine::adaptive_port::AdaptiveModelNamespace;
    use rz_search::pals::engine::idle_archive::UnusedIdleArchiveSetupControl;
    use rz_search::pals::engine::{
        LegalOrderRoleMock, PalsConfig, PalsSemanticConfig, PostRepairRecheckPolicy,
        ResearcherSemanticProfile, ResolverPolicy, RoleModel,
    };
    use rz_search::pals::review_context::{ReviewIndexLimits, ReviewSelectionPolicy};
    use rz_search::pals::store::{ArchiveConfig, ArchiveRootCapability, ArchiveRuntimeLimits};
    use std::path::PathBuf;
    use std::sync::atomic::AtomicU64;

    struct Scratch(PathBuf, bool);
    impl Scratch {
        fn new() -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let path = std::env::temp_dir().join(format!(
                "rz-native-storage-lifecycle-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            std::fs::create_dir(&path).unwrap();
            Self(path, false)
        }
    }
    impl Drop for Scratch {
        fn drop(&mut self) {
            if self.1
                && self.0.parent() == Some(std::env::temp_dir().as_path())
                && self.0.file_name().is_some_and(|name| {
                    name.to_string_lossy()
                        .starts_with("rz-native-storage-lifecycle-")
                })
                && std::fs::symlink_metadata(&self.0)
                    .is_ok_and(|metadata| metadata.is_dir() && !metadata.file_type().is_symlink())
            {
                let _ = std::fs::remove_dir_all(&self.0);
            }
        }
    }
    fn scope() -> SearchAuthority {
        SearchAuthority {
            epoch: ProcessEpoch(9301),
            game: GameGeneration(1),
            root: RootGeneration(1),
            implementation: Digest([71; 32]),
        }
    }
    fn model() -> AdaptivePortModel {
        let value = LegalOrderRoleMock.value_identity().unwrap().clone();
        let namespace = AdaptiveModelNamespace::new(
            value.clone(),
            Digest([72; 32]),
            Digest([73; 32]),
            PrecisionProfile::Fp32,
            1,
        )
        .unwrap();
        AdaptivePortModel::new_with_namespace(value.model.clone(), Some(value), namespace).unwrap()
    }
    fn pc(until: Instant) -> StorageSemanticOwner {
        let mut owner = Box::new(
            PcSemanticOwner::new_selected(
                PalsSemanticConfig {
                    beam_width: 1,
                    line_plies: 2,
                    max_nodes: 257,
                    max_records: 128,
                    max_role_calls: 128,
                },
                model(),
                ReviewSelectionPolicy::default(),
                ReviewIndexLimits::default(),
            )
            .unwrap(),
        );
        owner
            .try_adaptive_new_game(scope(), until, &AtomicBool::new(false))
            .unwrap();
        StorageSemanticOwner::Pc(owner)
    }
    fn limits() -> ArchiveRuntimeLimits {
        ArchiveRuntimeLimits {
            game_bytes_max: 1 << 20,
            global_bytes_max: 4 << 20,
            index_entries_max: 256,
            index_bytes_max: 512 << 10,
            load_bytes_max: 512 << 10,
            load_deadline_max_ms: 20_000,
            record_payload_bytes_max: 512 << 10,
            max_load_pins: 16,
        }
    }
    fn setup(
        scratch: &Scratch,
        limits: ArchiveRuntimeLimits,
        until: Instant,
        cancellation: Arc<AtomicBool>,
    ) -> UnusedIdleArchiveSetup {
        let held =
            cap_std::fs::Dir::open_ambient_dir(&scratch.0, cap_std::ambient_authority()).unwrap();
        UnusedIdleArchiveSetup::new(
            ArchiveConfig::new(scratch.0.clone(), std::env::current_dir().unwrap()),
            ArchiveRootCapability::from_directory(scratch.0.clone(), held).unwrap(),
            limits,
            UnusedIdleArchiveSetupIdentity {
                profile: Digest([74; 32]),
                input_lock: Digest([75; 32]),
                registration: Digest([76; 32]),
            },
            UnusedIdleArchiveSetupControl {
                deadline: until,
                cancellation,
                max_io_bytes: MAINTENANCE_IO_BYTES,
            },
        )
    }
    fn configure(
        core: &mut WorkerCustody,
        original: StorageSemanticOwner,
        payload: UnusedIdleArchiveSetup,
        until: Instant,
        signal: &CompletionSignal,
    ) {
        core.adopt(original)
            .unwrap_or_else(|(_, error)| panic!("actual original fixture owner: {error:?}"));
        core.ensure_worker(signal).unwrap();
        core.submit_setup(payload, scope(), until).unwrap();
        core.wait(until, &AtomicBool::new(false), signal).unwrap();
    }

    #[test]
    fn same_pc_owner_returns_only_after_idle_phase_join_and_final_checkpoint_has_its_own_join() {
        let mut scratch = Scratch::new();
        let until = Instant::now() + Duration::from_secs(20);
        let signal = CompletionSignal::default();
        let mut core = WorkerCustody::new();
        let original = pc(until);
        let StorageSemanticOwner::Pc(owner) = &original else {
            unreachable!()
        };
        let store = owner.stores().read_owner();
        let original_box = std::ptr::from_ref(owner.as_ref());
        configure(
            &mut core,
            original,
            setup(&scratch, limits(), until, Arc::new(AtomicBool::new(false))),
            until,
            &signal,
        );
        assert!(core.setup_receipt.as_ref().unwrap().configuration_complete);
        assert!(store.matches(core.idle.as_ref().unwrap().stores()));
        core.observe_empty_game(EmptyGamePhaseKind::UnusedSetup, scope())
            .unwrap();
        assert!(core.receipts.is_empty()); // no fabricated cold-lookup receipt
        assert!(core.shutdown.is_none());
        core.join(until, &signal).unwrap();
        core.verify_empty_game_handoff(scope()).unwrap();
        assert!(core.phase_joins.back().unwrap().1.worker_joined);
        let mut current_setup = observation::project(&core);
        current_setup.require_after_root(scope());
        assert!(current_setup.complete);
        // StoppedInitial returns before storage.prepare for the attempted next
        // go. Its current authority must not inherit this earlier setup phase.
        let mut stopped_initial = observation::project(&core);
        stopped_initial.require_after_root(SearchAuthority {
            root: RootGeneration(scope().root.0 + 1),
            ..scope()
        });
        assert!(!stopped_initial.complete);
        // Native finish may fail before storage.finish is entered. Reading
        // this joined setup is not coverage of a final checkpoint/close.
        let mut unentered_finish = observation::project(&core);
        unentered_finish.require_after_finish();
        assert!(!unentered_finish.complete);
        let setup_observation = serde_json::to_value(observation::project(&core)).unwrap();
        assert_eq!(
            setup_observation["facts"]["phase_receipts"],
            serde_json::json!([])
        );
        assert_eq!(setup_observation["facts"]["setup_receipt_observed"], true);
        assert!(setup_observation.get("after_root_scope").is_none());
        assert!(setup_observation.get("after_finish_complete").is_none());
        assert_eq!(
            setup_observation["facts"]["original_phase_join"]["worker_joined"],
            true
        );
        core.worker.take();
        assert!(!core.finish_completed); // a setup-phase join is not final close
        let StorageSemanticOwner::Pc(mut owner) = core.restore().unwrap() else {
            panic!("same PC kind")
        };
        assert_eq!(std::ptr::from_ref(owner.as_ref()), original_box);
        assert!(store.matches(owner.stores()));
        assert_eq!(owner.stores().generation(), 0);
        assert!(owner.stores().root().is_none());
        let registered = owner
            .register_adaptive_root(&Position::startpos(), scope())
            .unwrap();
        owner.abandon_unconsumed_adaptive_root(scope()).unwrap();
        drop(registered);
        core.adopt(StorageSemanticOwner::Pc(owner))
            .unwrap_or_else(|(_, error)| panic!("retired original: {error:?}"));
        core.ensure_worker(&signal).unwrap();
        assert!(core.shutdown.is_none()); // distinct actual worker phase
        let same_game = SearchAuthority {
            root: RootGeneration(2),
            ..scope()
        };
        core.work(
            same_game,
            ArchiveOperation::PrepareRoot {
                position: Position::startpos(),
            },
            ArchiveOperationKind::PrepareRoot,
            until,
            until,
            &AtomicBool::new(false),
            &signal,
        )
        .unwrap();
        core.work(
            same_game,
            ArchiveOperation::RestoreRootIfArchived {
                position: Position::startpos(),
            },
            ArchiveOperationKind::RestoreRootIfArchived,
            until,
            until,
            &AtomicBool::new(false),
            &signal,
        )
        .unwrap();
        assert!(core.receipts.back().unwrap().loaded_root.is_none());
        // A completed archive operation is still not a worker join.
        assert!(!observation::project(&core).complete);
        core.join(until, &signal).unwrap();
        let mut current_maintenance = observation::project(&core);
        current_maintenance.require_after_root(same_game);
        assert!(current_maintenance.complete);
        let mut next_without_phase = observation::project(&core);
        next_without_phase.require_after_root(SearchAuthority {
            root: RootGeneration(same_game.root.0 + 1),
            ..same_game
        });
        assert!(!next_without_phase.complete);
        let mut native_finish_failed = observation::project(&core);
        native_finish_failed.require_after_finish();
        assert!(!native_finish_failed.complete);
        let maintenance = serde_json::to_value(observation::project(&core)).unwrap();
        let actual_receipts = maintenance["facts"]["phase_receipts"].as_array().unwrap();
        assert_eq!(actual_receipts.len(), 2);
        assert_eq!(actual_receipts[0]["operation"], "prepare_root");
        assert_eq!(actual_receipts[1]["operation"], "restore_root_if_archived");
        assert!(actual_receipts[1]["loaded_root"].is_null());
        assert_eq!(
            actual_receipts[1]["nonce"],
            core.receipts.back().unwrap().nonce
        );
        assert_eq!(
            maintenance["facts"]["original_phase_join"]["worker_joined"],
            true
        );
        core.worker.take();
        let StorageSemanticOwner::Pc(mut owner) = core.restore().unwrap() else {
            panic!("same PC kind after ordinary maintenance")
        };
        let registered = owner
            .register_adaptive_root(&Position::startpos(), same_game)
            .unwrap();
        owner.abandon_unconsumed_adaptive_root(same_game).unwrap();
        drop(registered);
        core.adopt(StorageSemanticOwner::Pc(owner))
            .unwrap_or_else(|(_, error)| panic!("second original root: {error:?}"));
        core.ensure_worker(&signal).unwrap();
        let next = SearchAuthority {
            game: GameGeneration(2),
            root: RootGeneration(3),
            ..scope()
        };
        core.work(
            next,
            ArchiveOperation::NewGame {
                next_authority: next,
            },
            ArchiveOperationKind::NewGame,
            until,
            until,
            &AtomicBool::new(false),
            &signal,
        )
        .unwrap();
        let newgame = core.receipts.back().unwrap();
        assert_eq!(newgame.next_authority, Some(next));
        assert!(newgame.idle_checkpoint.as_ref().unwrap().committed_bytes > 0);
        assert!(!store.matches(core.idle.as_ref().unwrap().stores()));
        core.observe_empty_game(EmptyGamePhaseKind::NewGame, next)
            .unwrap();
        core.join(until, &signal).unwrap();
        core.verify_empty_game_handoff(next).unwrap();
        let mut current_newgame = observation::project(&core);
        current_newgame.require_after_root(next);
        assert!(current_newgame.complete);
        let mut newgame_is_not_finish = observation::project(&core);
        newgame_is_not_finish.require_after_finish();
        assert!(!newgame_is_not_finish.complete);
        core.worker.take();
        let StorageSemanticOwner::Pc(mut owner) = core.restore().unwrap() else {
            panic!("same PC kind after actual NewGame")
        };
        let registered = owner
            .register_adaptive_root(&Position::startpos(), next)
            .unwrap();
        owner.abandon_unconsumed_adaptive_root(next).unwrap();
        drop(registered);
        core.adopt(StorageSemanticOwner::Pc(owner))
            .unwrap_or_else(|(_, error)| panic!("new game original root: {error:?}"));
        core.ensure_worker(&signal).unwrap();
        core.work(
            next,
            ArchiveOperation::FinishIdleGame,
            ArchiveOperationKind::FinishIdleGame,
            until,
            until,
            &AtomicBool::new(false),
            &signal,
        )
        .unwrap();
        assert!(core.finish_completed);
        assert!(core.receipts.back().unwrap().idle_checkpoint.is_some());
        core.join(until, &signal).unwrap();
        assert_eq!(core.phase_joins.len(), 4);
        let mut final_coverage = observation::project(&core);
        final_coverage.require_after_finish();
        assert!(final_coverage.complete);
        let mut final_is_not_root = observation::project(&core);
        final_is_not_root.require_after_root(next);
        assert!(!final_is_not_root.complete);
        let original_checkpoint = core.receipts.back_mut().unwrap().idle_checkpoint.take();
        let mut checkpoint_missing = observation::project(&core);
        checkpoint_missing.require_after_finish();
        assert!(!checkpoint_missing.complete);
        core.receipts.back_mut().unwrap().idle_checkpoint = original_checkpoint;
        let finished = serde_json::to_value(observation::project(&core)).unwrap();
        let final_receipt = &finished["facts"]["phase_receipts"][0];
        assert_eq!(final_receipt["operation"], "finish_idle_game");
        assert_eq!(
            final_receipt["idle_checkpoint"]["committed_bytes"],
            core.receipts
                .back()
                .unwrap()
                .idle_checkpoint
                .as_ref()
                .unwrap()
                .committed_bytes
        );
        assert_eq!(finished["facts"]["finish_completed"], true);
        assert_eq!(finished["facts"]["physical_completion_attested"], false);
        let closed = core
            .idle
            .as_ref()
            .unwrap()
            .transition_snapshot()
            .unwrap()
            .unwrap();
        assert!(closed.completed && closed.original_closed.unwrap().admission_closed);
        scratch.1 = true;
    }

    #[test]
    fn storage_observation_lost_phase_never_uses_an_empty_success() {
        let mut core = WorkerCustody::<AdaptivePortModel>::new();
        core.phase_scope = Some(scope());
        core.shutdown = Some(ArchiveShutdownReceipt {
            worker_joined: true,
            owners_returned: true,
            admission_closed: true,
        });
        // Metadata-shaped join booleans alone cannot cover a missing actual
        // setup/empty phase, ordinary pair of work receipts or final checkpoint.
        let mut no_phase = observation::project(&core);
        no_phase.require_after_root(scope());
        assert!(!no_phase.complete);
        let mut no_finish = observation::project(&core);
        no_finish.require_after_finish();
        assert!(!no_finish.complete);
        core.receipt_sequence = 3;
        let lost = serde_json::to_value(observation::project(&core)).unwrap();
        assert_eq!(lost["complete"], false);
        assert_eq!(lost["facts"]["phase_receipts_known"], false);
        assert!(lost["facts"]["phase_receipts"].is_null());
        core.observation_sequence_exhausted = true;
        assert!(!observation::project(&core).complete);
    }

    #[test]
    fn pcr_setup_returns_same_deferred_owner_model_and_store_without_materializing_a_researcher() {
        let mut scratch = Scratch::new();
        let until = Instant::now() + Duration::from_secs(20);
        let signal = CompletionSignal::default();
        let (source, model_source) =
            super::super::nnuec_session::pre_materialization_tests::model_source();
        let mut owner = Box::new(
            ResearcherPreparingOwner::new_selected(
                ResearcherSemanticProfile {
                    config: PalsConfig {
                        beam_width: 1,
                        line_plies: 2,
                        max_nodes: 257,
                        max_records: 128,
                        max_role_calls: 128,
                        cpu_nodes_per_task: 256,
                    },
                    maximum_root_prefix_plies: 16,
                },
                model(),
                ResolverPolicy::ModelWdlRestricted,
                PostRepairRecheckPolicy::Disabled,
                ReviewSelectionPolicy::default(),
                ReviewIndexLimits::default(),
                source,
            )
            .unwrap(),
        );
        owner
            .try_adaptive_new_game(scope(), until, &AtomicBool::new(false))
            .unwrap();
        let store = owner.stores().read_owner();
        let original_box = std::ptr::from_ref(owner.as_ref());
        let mut core = WorkerCustody::new();
        configure(
            &mut core,
            StorageSemanticOwner::Pcr(owner),
            setup(&scratch, limits(), until, Arc::new(AtomicBool::new(false))),
            until,
            &signal,
        );
        assert_eq!(
            core.setup_receipt.as_ref().unwrap().owner_kind,
            IdleArchiveOwnerKind::Researcher
        );
        core.join(until, &signal).unwrap();
        core.worker.take();
        let StorageSemanticOwner::Pcr(owner) = core.restore().unwrap() else {
            panic!("same researcher kind")
        };
        assert_eq!(std::ptr::from_ref(owner.as_ref()), original_box);
        assert!(store.matches(owner.stores()));
        assert!(Arc::ptr_eq(owner.source().model(), &model_source));
        assert!(owner.last_authority().is_none());
        // There is deliberately no registered root or researcher/TT here.
        // Empty direct close is not a hot-game FinishIdleGame attestation.
        let mut idle = owner
            .into_idle_archive()
            .unwrap_or_else(|_| panic!("same original unused researcher owner"));
        assert_eq!(idle.stores().hot_stats(), Default::default());
        assert!(
            idle.close_archive_owner()
                .unwrap()
                .unwrap()
                .admission_closed
        );
        assert!(!core.finish_completed);
        scratch.1 = true;
    }

    #[test]
    fn canceled_submission_keeps_payload_owner_original_error_and_actual_join_separate() {
        let mut scratch = Scratch::new();
        let until = Instant::now() + Duration::from_secs(20);
        let signal = CompletionSignal::default();
        let mut core = WorkerCustody::new();
        core.adopt(pc(until))
            .unwrap_or_else(|(_, error)| panic!("owner: {error:?}"));
        let store = core.idle.as_ref().unwrap().stores().read_owner();
        core.ensure_worker(&signal).unwrap();
        let cancellation = Arc::new(AtomicBool::new(true));
        let original = setup(&scratch, limits(), until, cancellation);
        let identity = original.identity();
        let first = core.submit_setup(original, scope(), until).unwrap_err();
        assert!(core.operation_error.is_some());
        assert_eq!(core.setup.as_ref().unwrap().identity(), identity);
        assert!(store.matches(core.idle.as_ref().unwrap().stores()));
        assert!(core.setup_receipt.is_none() && !core.finish_completed);
        core.join(until, &signal).unwrap();
        assert_eq!(core.first_failure.as_ref().unwrap().code, first.code);
        assert_eq!(core.first_failure.as_ref().unwrap().detail, first.detail);
        assert!(core.shutdown.unwrap().owners_returned);
        scratch.1 = true;
    }

    #[test]
    fn wrong_install_preserves_actual_completion_and_does_not_become_final_close_after_join() {
        let mut scratch = Scratch::new();
        let until = Instant::now() + Duration::from_secs(20);
        let signal = CompletionSignal::default();
        let mut core = WorkerCustody::new();
        core.adopt(pc(until))
            .unwrap_or_else(|(_, error)| panic!("owner: {error:?}"));
        let store = core.idle.as_ref().unwrap().stores().read_owner();
        core.ensure_worker(&signal).unwrap();
        core.submit_setup(
            setup(&scratch, limits(), until, Arc::new(AtomicBool::new(false))),
            scope(),
            until,
        )
        .unwrap();
        let Some(Pending::Setup {
            scope: expected, ..
        }) = core.pending.as_mut()
        else {
            unreachable!()
        };
        expected.root = RootGeneration(99);
        assert!(core.wait(until, &AtomicBool::new(false), &signal).is_err());
        assert!(core.setup_rejected.as_ref().unwrap().failure().is_some());
        core.join(until, &signal).unwrap();
        assert!(core.setup_receipt.is_none() && !core.finish_completed);
        let rejected = *core.setup_rejected.take().unwrap();
        let parts = rejected.into_owner_and_setup_for_shutdown();
        assert!(parts.result.is_err());
        assert!(store.matches(parts.owner.stores()));
        assert_eq!(parts.receipt.scope, scope());
        // Preserve the original failed archive fixture, like Search's own
        // failed transition fixtures; do not delete a retained/partial owner.
        scratch.1 = false;
    }
}
