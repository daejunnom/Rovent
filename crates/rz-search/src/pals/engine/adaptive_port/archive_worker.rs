//! Exclusive archive maintenance. The manager lends the complete owner, never
//! a cloned Store, and cannot perform semantic commits until its checked return.
//! Commands and completion notifications each have one slot. Cancellation is
//! separate from that queue; a timed-out owner is quarantined, never recycled.

use super::profiled_retirement_archive::{
    MAX_PROFILED_RETIREMENT_IO_BYTES, ProfiledRetirementCommit, RegisteredProfiledRetirementSink,
};
use super::{AdaptiveEnginePort, PalsEngine, PalsError, RoleModel};
use super::{ProfiledExternalRetirementIdentity, ProfiledExternalRootRetirement};
use crate::pals::engine::adaptive_root::UnusedArchiveSetupSeal;
use crate::pals::engine::archive_lifecycle::IdleArchiveCheckpoint;
use crate::pals::engine::idle_archive::{
    IdleArchiveOwnerKind, IdleArchiveTransitionKind, IdleSemanticArchiveOwner,
    UnusedIdleArchiveSetup, UnusedIdleArchiveSetupIdentity,
};
use crate::pals::store::{
    ArchiveIoBudget, ArchiveOwnerSnapshot, ArchiveStats, ForeignAppendPermit, ForeignGoIssuedProof,
    Observation, ProfiledForeignAppendPermit, ProfiledForeignGoIssuedProof, SituationId,
    StoreError, StorePins, StoreReadOwner,
};
use rz_contracts::pals::SearchAuthority;
use rz_position::Position;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, SyncSender, TryRecvError, TrySendError};
use std::sync::{Arc, Mutex, Weak};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

const MAX_IO_BYTES: u64 = 16 * 1024 * 1024;

/// Before construction, root maintenance may compact the engine arena. An
/// active port retains pending node references and permits Store-only reclaim.
// This is the one exclusive engine/port transfer in a one-slot worker, not a
// collection of variant values. Preserve rejected/quarantined owners by value
// without adding a heap allocation at a pressure-recovery boundary.
#[allow(clippy::large_enum_variant)]
pub enum ArchiveOwner<M: RoleModel> {
    Engine(PalsEngine<M>),
    Port(AdaptiveEnginePort<M>),
    Idle(IdleSemanticArchiveOwner<M>),
}
impl<M: RoleModel> ArchiveOwner<M> {
    fn legacy_engine(&self) -> Result<&PalsEngine<M>, PalsError> {
        match self {
            Self::Engine(engine) => Ok(engine),
            Self::Port(port) => Ok(&port.engine),
            Self::Idle(_) => {
                Err(StoreError::InvalidConditions("idle owner is not an Engine").into())
            }
        }
    }
    fn legacy_engine_mut(&mut self) -> Result<&mut PalsEngine<M>, PalsError> {
        match self {
            Self::Engine(engine) => Ok(engine),
            Self::Port(port) => Ok(&mut port.engine),
            Self::Idle(_) => {
                Err(StoreError::InvalidConditions("idle owner is not an Engine").into())
            }
        }
    }
    #[cfg(test)]
    fn engine(&self) -> &PalsEngine<M> {
        self.legacy_engine().unwrap()
    }
    #[cfg(test)]
    fn engine_mut(&mut self) -> &mut PalsEngine<M> {
        self.legacy_engine_mut().unwrap()
    }
}

// One command slot owns the original observation and its one-use proof. Keep
// them together by value; no additional box/clone is needed per archive dispatch.
#[allow(clippy::large_enum_variant)]
pub enum ArchiveOperation {
    /// Existing root preparation, including checked topology persistence and
    /// compaction. Only an Engine owner may use this pre-construction barrier.
    PrepareRoot { position: Position },
    /// Existing cold load and metadata verification, without model/CPU work.
    LoadRoot { position: Position },
    /// Idle-only exact cold lookup. Only its normal missing state is None;
    /// failed reads, missing engine metadata and integrity errors stay errors.
    RestoreRootIfArchived { position: Position },
    /// Stable numeric engine indices, including two pending immutable tickets.
    ReclaimHot { incoming_pins: StorePins },
    /// Read-only cold uniqueness scan. The exact raw append is reserved by an
    /// opaque one-use permit and remains a manager hot commit after return.
    PrepareForeign {
        observation: Observation,
        issued: ForeignGoIssuedProof,
    },
    /// Separate profiled query namespace; no legacy job or completed-depth
    /// evidence is manufactured when the original go omitted a limit.
    PrepareProfiledForeign {
        observation: Observation,
        issued: ProfiledForeignGoIssuedProof,
    },
    /// Persist one original drained root without releasing its source/view
    /// owners. Only the dedicated install may return this move-only payload.
    PersistProfiledRetirement {
        payload: PersistProfiledRetirementWork,
    },
    /// Official lifecycle reset on the exclusive worker. This also supports an
    /// explicitly declared archive-off profile; it never changes archive mode.
    NewGame { next_authority: SearchAuthority },
    /// Idle-only final all-hot checkpoint and original archive close. The UCI
    /// owner has already proved its physical fence; this never proves one.
    FinishIdleGame,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ArchiveOperationKind {
    PrepareRoot,
    LoadRoot,
    RestoreRootIfArchived,
    ReclaimHot,
    PrepareForeign,
    NewGame,
    PrepareProfiledForeign,
    PersistProfiledRetirement,
    FinishIdleGame,
}
impl ArchiveOperation {
    fn kind(&self) -> ArchiveOperationKind {
        match self {
            Self::PrepareRoot { .. } => ArchiveOperationKind::PrepareRoot,
            Self::LoadRoot { .. } => ArchiveOperationKind::LoadRoot,
            Self::RestoreRootIfArchived { .. } => ArchiveOperationKind::RestoreRootIfArchived,
            Self::ReclaimHot { .. } => ArchiveOperationKind::ReclaimHot,
            Self::PrepareForeign { .. } => ArchiveOperationKind::PrepareForeign,
            Self::PrepareProfiledForeign { .. } => ArchiveOperationKind::PrepareProfiledForeign,
            Self::PersistProfiledRetirement { .. } => {
                ArchiveOperationKind::PersistProfiledRetirement
            }
            Self::NewGame { .. } => ArchiveOperationKind::NewGame,
            Self::FinishIdleGame => ArchiveOperationKind::FinishIdleGame,
        }
    }
}
pub struct ArchiveWork {
    pub operation: ArchiveOperation,
    pub deadline: Instant,
    pub max_io_bytes: u64,
}

/// The exact source recovery and sink stay together before execution, during
/// an unwind and after physical completion. A commit never replaces them.
pub struct PersistProfiledRetirementWork {
    retirement: ProfiledExternalRootRetirement,
    sink: RegisteredProfiledRetirementSink,
    commit: Option<ProfiledRetirementCommit>,
}
impl PersistProfiledRetirementWork {
    pub fn new(
        retirement: ProfiledExternalRootRetirement,
        sink: RegisteredProfiledRetirementSink,
    ) -> Self {
        Self {
            retirement,
            sink,
            commit: None,
        }
    }
    pub fn retirement(&self) -> &ProfiledExternalRootRetirement {
        &self.retirement
    }
    pub fn sink(&self) -> &RegisteredProfiledRetirementSink {
        &self.sink
    }
    pub fn commit(&self) -> Option<&ProfiledRetirementCommit> {
        self.commit.as_ref()
    }
    pub fn into_parts(
        self,
    ) -> (
        ProfiledExternalRootRetirement,
        RegisteredProfiledRetirementSink,
        Option<ProfiledRetirementCommit>,
    ) {
        (self.retirement, self.sink, self.commit)
    }
}

/// A small source-only change stamp for the manager's no-progress latch. It
/// includes counts and authority, never a cloned node/record/statistics graph.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ArchiveMaintenanceStamp {
    pub store_generation: u64,
    pub public_revision: u64,
    pub hot: ArchiveStats,
}
pub struct ArchiveMaintenanceProposal {
    pub incoming_pins: StorePins,
    pub max_io_bytes: u64,
    pub deadline: Instant,
    pub stamp: ArchiveMaintenanceStamp,
}
impl<M: RoleModel> AdaptiveEnginePort<M> {
    /// Actual remaining aggregate search allowance, not another 16MiB grant.
    pub fn archive_remaining_io(&self) -> u64 {
        self.engine.archive_remaining_io()
    }
    /// Call before a semantic commit, not on every manager statistics tick.
    /// The no-pressure path reads counters only. Active ports cannot compact
    /// their engine arena; the worker later builds the full resident pin union.
    pub fn archive_maintenance_proposal(
        &self,
    ) -> Result<Option<ArchiveMaintenanceProposal>, PalsError> {
        if !self.engine.archive_enabled() || !self.engine.semantic.stores.archive_pressure() {
            return Ok(None);
        }
        self.engine.ensure_archive_admission_open()?;
        if Instant::now() >= self.limits.deadline {
            return Err(StoreError::ArchiveDeadline.into());
        }
        let max_io_bytes = self.archive_remaining_io().min(MAX_IO_BYTES);
        if max_io_bytes == 0 {
            return Err(StoreError::ArchiveByteBudget.into());
        }
        let stores = &self.engine.semantic.stores;
        let root = self
            .engine
            .semantic
            .nodes
            .get(self.root)
            .ok_or(StoreError::InvalidHandle("adaptive archive root"))?;
        let situation = stores.situations.get(root.situation)?;
        if situation.state != root.state || stores.root() != Some(root.situation) {
            return Err(StoreError::StaleConsumer.into());
        }
        let mut pins = stores.review_pin_union()?;
        pins.states.insert(root.state);
        pins.situations.insert(root.situation);
        pins.lines.insert(situation.focus);
        merge_pins(&mut pins, &self.goal_local_protected_pins()?);
        merge_pins(&mut pins, &self.profiled_external_protected_pins()?);
        for admission in self.cpu_admissions.values() {
            if let Some(execution) = admission.execution {
                pins.executions.insert(execution);
                pins.states.insert(stores.tasks.get(execution)?.key.state);
            }
        }
        validate_pins(stores, &pins)?;
        Ok(Some(ArchiveMaintenanceProposal {
            incoming_pins: pins,
            max_io_bytes,
            deadline: self.limits.deadline,
            stamp: ArchiveMaintenanceStamp {
                store_generation: stores.generation(),
                public_revision: self.engine.semantic.revision,
                hot: stores.hot_stats(),
            },
        }))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ArchiveWorkerError {
    Busy,
    StaleTransfer,
    Closed,
    CompletionUnknown,
    SequenceExhausted,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ArchiveShutdownReceipt {
    pub admission_closed: bool,
    pub worker_joined: bool,
    /// Every accepted transfer's actual owner was received before joining.
    pub owners_returned: bool,
}
/// Rejection does not consume ownership; it returns the exact submitted value.
pub struct ArchiveRejected<M: RoleModel> {
    pub owner: ArchiveOwner<M>,
    pub work: ArchiveWork,
    pub error: PalsError,
}
/// A transfer is opaque and belongs to one worker. Cancel remains reachable
/// even while its bounded command queue is occupied.
pub struct ArchiveTransfer {
    nonce: u64,
    worker: Arc<()>,
    cancel: Arc<AtomicBool>,
}

#[derive(Clone, Debug)]
pub struct UnusedIdleArchiveSetupReceipt {
    pub nonce: u64,
    pub scope: SearchAuthority,
    pub identity: UnusedIdleArchiveSetupIdentity,
    pub owner_kind: IdleArchiveOwnerKind,
    pub archive_owner: Option<ArchiveOwnerSnapshot>,
    pub remaining_io_bytes: Option<u64>,
    pub configuration_complete: bool,
    pub elapsed: Duration,
}
pub struct UnusedIdleArchiveSetupRejected<M: RoleModel> {
    pub owner: IdleSemanticArchiveOwner<M>,
    pub setup: UnusedIdleArchiveSetup,
    pub error: PalsError,
}
/// A wrong/late install retains the complete original owner and held payload.
/// This is not a root-registration or native physical completion receipt.
pub struct UnusedIdleArchiveSetupCompletion<M: RoleModel> {
    owner: IdleSemanticArchiveOwner<M>,
    setup: UnusedIdleArchiveSetup,
    seal: UnusedArchiveSetupSeal,
    idle_identity: Weak<()>,
    receipt: UnusedIdleArchiveSetupReceipt,
    result: Result<(), ArchiveQuarantineReason>,
}
/// Shutdown extraction keeps the first operation/install result with the exact
/// owner, held setup payload and actual observation. It grants no reuse.
pub struct UnusedIdleArchiveSetupShutdownParts<M: RoleModel> {
    pub owner: IdleSemanticArchiveOwner<M>,
    pub setup: UnusedIdleArchiveSetup,
    pub receipt: UnusedIdleArchiveSetupReceipt,
    pub result: Result<(), ArchiveQuarantineReason>,
}
impl<M: RoleModel> UnusedIdleArchiveSetupCompletion<M> {
    pub fn receipt(&self) -> &UnusedIdleArchiveSetupReceipt {
        &self.receipt
    }
    pub fn failure(&self) -> Option<&ArchiveQuarantineReason> {
        self.result.as_ref().err()
    }
    #[allow(clippy::result_large_err)]
    pub fn install(
        mut self,
        expected_scope: SearchAuthority,
        expected_nonce: u64,
        expected_identity: UnusedIdleArchiveSetupIdentity,
    ) -> Result<(IdleSemanticArchiveOwner<M>, UnusedIdleArchiveSetupReceipt), Self> {
        if self.result.is_ok() {
            self.result = (|| {
                if expected_scope != self.receipt.scope
                    || expected_nonce != self.receipt.nonce
                    || expected_identity != self.receipt.identity
                    || !self.idle_identity.ptr_eq(&self.owner.identity())
                    || !self.receipt.configuration_complete
                    || self.receipt.owner_kind != self.owner.owner_kind()
                {
                    return Err(ArchiveQuarantineReason::InvalidReturn(
                        "unused setup identity/nonce",
                    ));
                }
                if self.setup.control.cancellation.load(Ordering::Acquire) {
                    return Err(ArchiveQuarantineReason::Canceled);
                }
                if Instant::now() >= self.setup.control.deadline {
                    return Err(ArchiveQuarantineReason::Deadline);
                }
                self.seal
                    .validate(self.owner.semantic())
                    .map_err(ArchiveQuarantineReason::Operation)?;
                let actual = self
                    .owner
                    .archive_owner_snapshot()
                    .map_err(ArchiveQuarantineReason::Operation)?;
                if actual != self.receipt.archive_owner
                    || actual.is_none_or(|value| {
                        !value.complete
                            || value.admission_closed
                            || value.runtime_limits != Some(self.setup.limits)
                    })
                    || self.owner.stores().hot_stats() != ArchiveStats::default()
                    || !self.owner.semantic().nodes.is_empty()
                    || !self.owner.semantic().records.is_empty()
                {
                    return Err(ArchiveQuarantineReason::InvalidReturn(
                        "unused setup actual owner",
                    ));
                }
                Ok(())
            })();
        }
        if self.result.is_err() {
            self.owner.retain_unused_setup_failure();
            return Err(self);
        }
        Ok((self.owner, self.receipt))
    }
    pub fn into_owner_and_setup_for_shutdown(self) -> UnusedIdleArchiveSetupShutdownParts<M> {
        UnusedIdleArchiveSetupShutdownParts {
            owner: self.owner,
            setup: self.setup,
            receipt: self.receipt,
            result: self.result,
        }
    }
}
impl ArchiveTransfer {
    pub fn nonce(&self) -> u64 {
        self.nonce
    }
    pub fn cancel(&self) {
        self.cancel.store(true, Ordering::Release);
    }
}

struct ArchiveSeal {
    authority: SearchAuthority,
    store_owner: StoreReadOwner,
    store_generation: u64,
    game_generation: Option<u64>,
    root: Option<SituationId>,
    revision: u64,
    node_handles: Vec<(crate::pals::store::StateId, SituationId)>,
    record_revisions: Vec<u64>,
    review_pins: StorePins,
    active_cpu_pins: StorePins,
    profiled_retirement_pins: StorePins,
    profiled_retirement_identity: Option<ProfiledExternalRetirementIdentity>,
    archive_owner: Option<ArchiveOwnerSnapshot>,
    registry_owner: Weak<()>,
    next_game: Option<SearchAuthority>,
    is_port: bool,
    idle_owner: Option<Weak<()>>,
}

#[derive(Clone, Debug)]
pub struct ArchiveWorkReceipt {
    pub nonce: u64,
    pub authority: SearchAuthority,
    pub operation: ArchiveOperationKind,
    pub generation_before: u64,
    pub generation_after: u64,
    pub hot_before: ArchiveStats,
    pub hot_after: ArchiveStats,
    pub io_bytes: Option<u64>,
    pub elapsed: Duration,
    pub loaded_root: Option<usize>,
    pub archive_owner: Option<ArchiveOwnerSnapshot>,
    pub next_authority: Option<SearchAuthority>,
    pub idle_checkpoint: Option<Arc<IdleArchiveCheckpoint>>,
}
#[derive(Clone, Debug)]
pub enum ArchiveQuarantineReason {
    Operation(PalsError),
    Panic,
    Canceled,
    Deadline,
    InvalidReturn(&'static str),
}
/// Failed maintenance still owns all sessions, Store facts, pins and tokens.
/// Only explicit shutdown/recovery may take it; it is not an accepted return.
pub struct ArchiveQuarantine<M: RoleModel> {
    pub reason: ArchiveQuarantineReason,
    pub receipt: ArchiveWorkReceipt,
    owner: ArchiveOwner<M>,
    foreign_permit: Option<ForeignAppendPermit>,
    profiled_foreign_permit: Option<ProfiledForeignAppendPermit>,
    profiled_retirement: Option<PersistProfiledRetirementWork>,
}
impl<M: RoleModel> ArchiveQuarantine<M> {
    pub fn retains_foreign_permit(&self) -> bool {
        self.foreign_permit.is_some()
    }
    pub fn retains_profiled_foreign_permit(&self) -> bool {
        self.profiled_foreign_permit.is_some()
    }
    pub fn profiled_retirement(&self) -> Option<&PersistProfiledRetirementWork> {
        self.profiled_retirement.as_ref()
    }
    pub fn retains_profiled_retirement(&self) -> bool {
        self.profiled_retirement.is_some()
    }
    /// A retirement has its own failed descriptors, pins and possible commit.
    /// Owner-only teardown cannot silently discard those original resources.
    #[allow(clippy::result_large_err)]
    pub fn into_owner_for_shutdown(self) -> Result<ArchiveOwner<M>, Self> {
        if self.profiled_retirement.is_some() {
            Err(self)
        } else {
            Ok(self.owner)
        }
    }
}

pub struct ArchiveCompletion<M: RoleModel> {
    owner: ArchiveOwner<M>,
    seal: ArchiveSeal,
    receipt: ArchiveWorkReceipt,
    result: Result<(), ArchiveQuarantineReason>,
    deadline: Instant,
    cancel: Arc<AtomicBool>,
    foreign_permit: Option<ForeignAppendPermit>,
    profiled_foreign_permit: Option<ProfiledForeignAppendPermit>,
    profiled_retirement: Option<PersistProfiledRetirementWork>,
}
impl<M: RoleModel> ArchiveCompletion<M> {
    pub fn receipt(&self) -> &ArchiveWorkReceipt {
        &self.receipt
    }
    /// No clock/root authority is replaced by the worker's submitted copy.
    /// A late result is useful evidence but cannot resume semantic admission.
    // Invalid returns retain every engine/session/pin in the original owner;
    // quarantine must not depend on allocating a new boxed error.
    #[allow(clippy::result_large_err)]
    pub fn install(
        mut self,
        expected_authority: SearchAuthority,
        expected_nonce: u64,
    ) -> Result<(ArchiveOwner<M>, ArchiveWorkReceipt), ArchiveQuarantine<M>> {
        if self.result.is_ok() && self.receipt.operation == ArchiveOperationKind::PrepareForeign {
            self.result = Err(ArchiveQuarantineReason::InvalidReturn(
                "foreign scan requires install_with_foreign",
            ));
        }
        if self.result.is_ok()
            && self.receipt.operation == ArchiveOperationKind::PrepareProfiledForeign
        {
            self.result = Err(ArchiveQuarantineReason::InvalidReturn(
                "profiled foreign scan requires install_with_profiled_foreign",
            ));
        }
        if self.result.is_ok()
            && (self.receipt.operation == ArchiveOperationKind::PersistProfiledRetirement
                || self.profiled_retirement.is_some())
        {
            self.result = Err(ArchiveQuarantineReason::InvalidReturn(
                "retirement persistence requires install_with_profiled_retirement",
            ));
        }
        self.install_checked(expected_authority, expected_nonce)
            .map(|(owner, receipt, _, _, _)| (owner, receipt))
    }
    // Same by-value quarantine as install, including the one-use foreign proof.
    #[allow(clippy::result_large_err)]
    pub fn install_with_foreign(
        mut self,
        expected_authority: SearchAuthority,
        expected_nonce: u64,
    ) -> Result<(ArchiveOwner<M>, ArchiveWorkReceipt, ForeignAppendPermit), ArchiveQuarantine<M>>
    {
        if self.result.is_ok()
            && (self.receipt.operation != ArchiveOperationKind::PrepareForeign
                || self.foreign_permit.is_none()
                || self.profiled_retirement.is_some())
        {
            self.result = Err(ArchiveQuarantineReason::InvalidReturn(
                "foreign permit missing",
            ));
        }
        self.install_checked(expected_authority, expected_nonce)
            .map(|(owner, receipt, permit, _, _)| {
                (owner, receipt, permit.expect("checked foreign permit"))
            })
    }

    /// Only this path returns the exact profiled permit. A wrong operation,
    /// late return or invalid owner keeps it in the original quarantine.
    #[allow(clippy::result_large_err)]
    pub fn install_with_profiled_foreign(
        mut self,
        expected_authority: SearchAuthority,
        expected_nonce: u64,
    ) -> Result<
        (
            ArchiveOwner<M>,
            ArchiveWorkReceipt,
            ProfiledForeignAppendPermit,
        ),
        ArchiveQuarantine<M>,
    > {
        if self.result.is_ok()
            && (self.receipt.operation != ArchiveOperationKind::PrepareProfiledForeign
                || self.profiled_foreign_permit.is_none()
                || self.profiled_retirement.is_some())
        {
            self.result = Err(ArchiveQuarantineReason::InvalidReturn(
                "profiled foreign permit missing",
            ));
        }
        self.install_checked(expected_authority, expected_nonce)
            .map(|(owner, receipt, _, permit, _)| {
                (
                    owner,
                    receipt,
                    permit.expect("checked profiled foreign permit"),
                )
            })
    }
    /// The original retirement and sink are returned only with their actual
    /// opaque commit. Wrong-kind, late, canceled and invalid-owner results
    /// retain the whole payload in quarantine, including any completed commit.
    #[allow(clippy::result_large_err)]
    pub fn install_with_profiled_retirement(
        mut self,
        expected_authority: SearchAuthority,
        expected_nonce: u64,
    ) -> Result<
        (
            ArchiveOwner<M>,
            ArchiveWorkReceipt,
            PersistProfiledRetirementWork,
        ),
        ArchiveQuarantine<M>,
    > {
        if self.result.is_ok()
            && (self.receipt.operation != ArchiveOperationKind::PersistProfiledRetirement
                || self
                    .profiled_retirement
                    .as_ref()
                    .is_none_or(|payload| payload.commit.is_none()))
        {
            self.result = Err(ArchiveQuarantineReason::InvalidReturn(
                "profiled retirement commit missing",
            ));
        }
        self.install_checked(expected_authority, expected_nonce)
            .map(|(owner, receipt, _, _, payload)| {
                (
                    owner,
                    receipt,
                    payload.expect("checked profiled retirement payload"),
                )
            })
    }
    // All install paths share this allocation-free owner rejection.
    #[allow(clippy::result_large_err)]
    // Return each original owner/permit without a box or unchecked wrapper.
    #[allow(clippy::type_complexity)]
    fn install_checked(
        self,
        expected_authority: SearchAuthority,
        expected_nonce: u64,
    ) -> Result<
        (
            ArchiveOwner<M>,
            ArchiveWorkReceipt,
            Option<ForeignAppendPermit>,
            Option<ProfiledForeignAppendPermit>,
            Option<PersistProfiledRetirementWork>,
        ),
        ArchiveQuarantine<M>,
    > {
        let checked = self.result.clone().and_then(|()| {
            if expected_nonce != self.receipt.nonce || expected_authority != self.seal.authority {
                return Err(ArchiveQuarantineReason::InvalidReturn("transfer authority"));
            }
            if self.cancel.load(Ordering::Acquire) {
                return Err(ArchiveQuarantineReason::Canceled);
            }
            if Instant::now() >= self.deadline {
                return Err(ArchiveQuarantineReason::Deadline);
            }
            self.validate_return()
        });
        match checked {
            Ok(()) => Ok((
                self.owner,
                self.receipt,
                self.foreign_permit,
                self.profiled_foreign_permit,
                self.profiled_retirement,
            )),
            Err(reason) => Err(ArchiveQuarantine {
                reason,
                receipt: self.receipt,
                owner: self.owner,
                foreign_permit: self.foreign_permit,
                profiled_foreign_permit: self.profiled_foreign_permit,
                profiled_retirement: self.profiled_retirement,
            }),
        }
    }
    fn validate_return(&self) -> Result<(), ArchiveQuarantineReason> {
        let invalid = ArchiveQuarantineReason::InvalidReturn;
        let permit_shape_matches = match self.receipt.operation {
            ArchiveOperationKind::PrepareForeign => {
                self.foreign_permit.is_some()
                    && self.profiled_foreign_permit.is_none()
                    && self.profiled_retirement.is_none()
            }
            ArchiveOperationKind::PrepareProfiledForeign => {
                self.profiled_foreign_permit.is_some()
                    && self.foreign_permit.is_none()
                    && self.profiled_retirement.is_none()
            }
            ArchiveOperationKind::PersistProfiledRetirement => {
                self.foreign_permit.is_none()
                    && self.profiled_foreign_permit.is_none()
                    && self
                        .profiled_retirement
                        .as_ref()
                        .is_some_and(|payload| payload.commit.is_some())
            }
            _ => {
                self.foreign_permit.is_none()
                    && self.profiled_foreign_permit.is_none()
                    && self.profiled_retirement.is_none()
            }
        };
        if !permit_shape_matches {
            return Err(invalid("archive typed permit differs from operation"));
        }
        if let ArchiveOwner::Idle(owner) = &self.owner {
            return self.validate_idle_return(owner);
        }
        let engine = self
            .owner
            .legacy_engine()
            .map_err(ArchiveQuarantineReason::Operation)?;
        if self.receipt.operation == ArchiveOperationKind::NewGame {
            let next = self
                .seal
                .next_game
                .ok_or_else(|| invalid("newgame target missing"))?;
            if self.seal.is_port
                || !matches!(self.owner, ArchiveOwner::Engine(_))
                || !self
                    .seal
                    .registry_owner
                    .ptr_eq(&engine.adaptive_registry_identity())
                || engine.adaptive_new_game_scope() != Some(next)
                || engine.semantic.game_generation != Some(next.game.0)
                || engine.semantic.stores.generation() != 0
                || engine.semantic.stores.root().is_some()
                || self.seal.store_owner.matches(&engine.semantic.stores)
                || engine
                    .semantic
                    .stores
                    .tasks
                    .active_states()
                    .next()
                    .is_some()
                || engine
                    .semantic
                    .stores
                    .review_pin_count()
                    .map_err(|error| ArchiveQuarantineReason::Operation(error.into()))?
                    != 0
                || !engine.semantic.nodes.is_empty()
                || !engine.semantic.records.is_empty()
            {
                return Err(invalid("official newgame owner transition"));
            }
            let actual = engine
                .archive_owner_snapshot()
                .map_err(ArchiveQuarantineReason::Operation)?;
            match (self.seal.archive_owner, actual) {
                (Some(old), Some(new)) => {
                    let previous = engine
                        .semantic
                        .archive_state
                        .previous_closed_owner
                        .ok_or_else(|| invalid("previous archive close missing"))?;
                    if previous.owner_id != old.owner_id
                        || previous.root_sha256 != old.root_sha256
                        || previous.repository_sha256 != old.repository_sha256
                        || !previous.admission_closed
                        || new.owner_id == old.owner_id
                        || new.admission_closed
                    {
                        return Err(invalid("newgame archive owner handoff"));
                    }
                }
                (None, None) => {}
                _ => return Err(invalid("newgame changed declared archive mode")),
            }
            return Ok(());
        }
        if !self.seal.store_owner.matches(&engine.semantic.stores)
            || engine.semantic.game_generation != self.seal.game_generation
            || engine.semantic.revision != self.seal.revision
            || matches!(self.owner, ArchiveOwner::Port(_)) != self.seal.is_port
        {
            return Err(invalid("store owner, game or semantic revision"));
        }
        let generation = engine.semantic.stores.generation();
        if generation != self.seal.store_generation {
            // Root preparation alone may retire the previous exact root after
            // its attested metadata commit; it does not admit the next root.
            let retired = self.receipt.operation == ArchiveOperationKind::PrepareRoot
                && self.seal.root.is_some()
                && engine.semantic.stores.root().is_none()
                && self.seal.store_generation.checked_add(1) == Some(generation)
                && engine.last_engine_archive().is_some_and(|receipt| {
                    receipt
                        .actual_root()
                        .is_some_and(|(root, _, old_generation)| {
                            Some(root) == self.seal.root
                                && old_generation == self.seal.store_generation
                        })
                });
            if !retired {
                return Err(invalid("store generation"));
            }
        } else if engine.semantic.stores.root() != self.seal.root {
            return Err(invalid("store root"));
        }
        let actual = engine
            .archive_owner_snapshot()
            .map_err(ArchiveQuarantineReason::Operation)?;
        let explicitly_archive_off = self.receipt.operation
            == ArchiveOperationKind::PersistProfiledRetirement
            && self.seal.archive_owner.is_none()
            && actual.is_none();
        if !explicitly_archive_off {
            let actual = actual.ok_or_else(|| invalid("archive owner missing"))?;
            let expected = self
                .seal
                .archive_owner
                .ok_or_else(|| invalid("submitted archive owner missing"))?;
            if actual.owner_id != expected.owner_id
                || actual.root_sha256 != expected.root_sha256
                || actual.repository_sha256 != expected.repository_sha256
                || actual.admission_closed
            {
                return Err(invalid("archive persistence owner"));
            }
        }
        validate_pins(&engine.semantic.stores, &self.seal.review_pins)
            .map_err(|_| invalid("submitted review pins"))?;
        validate_pins(&engine.semantic.stores, &self.seal.active_cpu_pins)
            .map_err(|_| invalid("worker CPU task pins"))?;
        validate_pins(&engine.semantic.stores, &self.seal.profiled_retirement_pins)
            .map_err(|_| invalid("profiled retirement pins"))?;
        if let Some(payload) = &self.profiled_retirement {
            let identity = self
                .seal
                .profiled_retirement_identity
                .as_ref()
                .ok_or_else(|| invalid("submitted profiled retirement identity missing"))?;
            if !payload.retirement.matches_identity(identity)
                || payload.retirement.authority() != self.seal.authority
                || !matches!(self.owner, ArchiveOwner::Engine(_))
            {
                return Err(invalid("profiled retirement original owner"));
            }
            payload
                .retirement
                .validate_owner(&engine.semantic.stores)
                .map_err(ArchiveQuarantineReason::Operation)?;
            let commit = payload
                .commit
                .as_ref()
                .ok_or_else(|| invalid("profiled retirement commit missing"))?;
            payload
                .sink
                .validate_commit(&engine.semantic.stores, &payload.retirement, commit)
                .map_err(ArchiveQuarantineReason::Operation)?;
        } else if self.seal.profiled_retirement_identity.is_some() {
            return Err(invalid("submitted profiled retirement payload missing"));
        }
        if matches!(
            self.receipt.operation,
            ArchiveOperationKind::ReclaimHot
                | ArchiveOperationKind::PrepareForeign
                | ArchiveOperationKind::PrepareProfiledForeign
                | ArchiveOperationKind::PersistProfiledRetirement
        ) && (engine.semantic.nodes.len() != self.seal.node_handles.len()
            || engine
                .semantic
                .nodes
                .iter()
                .map(|node| (node.state, node.situation))
                .ne(self.seal.node_handles.iter().copied())
            || engine
                .semantic
                .records
                .iter()
                .map(|record| record.revision)
                .ne(self.seal.record_revisions.iter().copied()))
        {
            return Err(invalid("active engine topology"));
        }
        if let ArchiveOwner::Port(port) = &self.owner {
            if port.authority != self.seal.authority {
                return Err(invalid("active port authority"));
            }
        }
        Ok(())
    }

    fn validate_idle_return(
        &self,
        owner: &IdleSemanticArchiveOwner<M>,
    ) -> Result<(), ArchiveQuarantineReason> {
        let invalid = ArchiveQuarantineReason::InvalidReturn;
        let semantic = owner.semantic();
        if self.seal.is_port
            || !self
                .seal
                .idle_owner
                .as_ref()
                .is_some_and(|id| id.ptr_eq(&owner.identity()))
            || !self
                .seal
                .registry_owner
                .ptr_eq(&semantic.adaptive_registry_identity())
            || !matches!(
                self.receipt.operation,
                ArchiveOperationKind::PrepareRoot
                    | ArchiveOperationKind::LoadRoot
                    | ArchiveOperationKind::RestoreRootIfArchived
                    | ArchiveOperationKind::NewGame
                    | ArchiveOperationKind::FinishIdleGame
            )
        {
            return Err(invalid("idle semantic owner, registry or operation"));
        }
        if self.receipt.operation == ArchiveOperationKind::NewGame {
            owner
                .validate_idle()
                .map_err(ArchiveQuarantineReason::Operation)?;
            let next = self
                .seal
                .next_game
                .ok_or_else(|| invalid("idle newgame target missing"))?;
            let previous = semantic
                .archive_state
                .previous_closed_owner
                .ok_or_else(|| invalid("idle original archive close missing"))?;
            let actual = owner
                .archive_owner_snapshot()
                .map_err(ArchiveQuarantineReason::Operation)?
                .ok_or_else(|| invalid("idle replacement archive owner missing"))?;
            self.validate_idle_checkpoint(owner)?;
            if semantic.adaptive_new_game_scope() != Some(next)
                || semantic.game_generation != Some(next.game.0)
                || semantic.stores.generation() != 0
                || semantic.stores.root().is_some()
                || self.seal.store_owner.matches(owner.stores())
                || !semantic.nodes.is_empty()
                || !semantic.records.is_empty()
                || semantic.revision != 0
                || Some(previous.owner_id) != self.seal.archive_owner.map(|old| old.owner_id)
                || !previous.admission_closed
                || !previous.complete
                || actual.owner_id == previous.owner_id
                || actual.admission_closed
                || !actual.complete
                || actual.root_sha256 != previous.root_sha256
                || actual.repository_sha256 != previous.repository_sha256
                || actual.runtime_limits != previous.runtime_limits
                || actual.session_write_ledger_id != previous.session_write_ledger_id
                || actual.session_write_bytes_max != previous.session_write_bytes_max
                || actual.session_write_bytes_consumed != previous.session_write_bytes_consumed
            {
                return Err(invalid("idle original newgame Store/ledger transition"));
            }
            return Ok(());
        }
        if !self.seal.store_owner.matches(owner.stores())
            || semantic.game_generation != self.seal.game_generation
            || semantic.revision != self.seal.revision
        {
            return Err(invalid("idle original Store/game/revision"));
        }
        if self.receipt.operation == ArchiveOperationKind::FinishIdleGame {
            semantic
                .ensure_adaptive_logical_quiescence()
                .map_err(ArchiveQuarantineReason::Operation)?;
            self.validate_idle_checkpoint(owner)?;
            let pending = owner
                .transition_snapshot()
                .map_err(ArchiveQuarantineReason::Operation)?
                .ok_or_else(|| invalid("idle final transition missing"))?;
            let closed = pending
                .original_closed
                .ok_or_else(|| invalid("idle final close missing"))?;
            if !pending.completed
                || pending.next_authority.is_some()
                || pending.replacement.is_some()
                || Some(closed.owner_id) != self.seal.archive_owner.map(|old| old.owner_id)
                || !closed.admission_closed
                || !closed.complete
                || owner.stores().generation() != self.seal.store_generation
                || owner.stores().root() != self.seal.root
            {
                return Err(invalid("idle original final checkpoint/close"));
            }
            validate_pins(owner.stores(), &self.seal.review_pins)
                .map_err(|_| invalid("idle final source pins"))?;
            return Ok(());
        }
        owner
            .validate_idle()
            .map_err(ArchiveQuarantineReason::Operation)?;
        let generation = owner.stores().generation();
        if generation != self.seal.store_generation {
            let retired = self.receipt.operation == ArchiveOperationKind::PrepareRoot
                && self.seal.root.is_some()
                && owner.stores().root().is_none()
                && self.seal.store_generation.checked_add(1) == Some(generation)
                && semantic.last_engine_archive().is_some_and(|receipt| {
                    receipt.actual_root().is_some_and(|(root, _, old)| {
                        Some(root) == self.seal.root && old == self.seal.store_generation
                    })
                });
            if !retired {
                return Err(invalid("idle Store generation"));
            }
        } else if owner.stores().root() != self.seal.root {
            return Err(invalid("idle Store root"));
        }
        let actual = owner
            .archive_owner_snapshot()
            .map_err(ArchiveQuarantineReason::Operation)?;
        match (self.seal.archive_owner, actual) {
            (Some(before), Some(after))
                if before.owner_id == after.owner_id
                    && before.root_sha256 == after.root_sha256
                    && before.repository_sha256 == after.repository_sha256
                    && !after.admission_closed => {}
            _ => return Err(invalid("idle original archive owner")),
        }
        validate_pins(owner.stores(), &self.seal.review_pins)
            .map_err(|_| invalid("idle original index/source pins"))?;
        Ok(())
    }

    fn validate_idle_checkpoint(
        &self,
        owner: &IdleSemanticArchiveOwner<M>,
    ) -> Result<(), ArchiveQuarantineReason> {
        let invalid = ArchiveQuarantineReason::InvalidReturn;
        let checkpoint = self
            .receipt
            .idle_checkpoint
            .as_ref()
            .ok_or_else(|| invalid("idle durable checkpoint missing"))?;
        if owner.last_checkpoint() != self.receipt.idle_checkpoint
            || checkpoint.hot != self.receipt.hot_before
            || checkpoint.records != self.seal.record_revisions.len()
            || checkpoint.nodes != self.seal.node_handles.len()
            || Some(checkpoint.owner.owner_id) != self.seal.archive_owner.map(|old| old.owner_id)
            || !checkpoint.owner.complete
            || checkpoint.committed_bytes == 0
        {
            return Err(invalid("idle checkpoint differs from original hot facts"));
        }
        Ok(())
    }
}

struct Command<M: RoleModel> {
    owner: ArchiveOwner<M>,
    seal: ArchiveSeal,
    nonce: u64,
    work: ArchiveWork,
    cancel: Arc<AtomicBool>,
}
struct SetupCommand<M: RoleModel> {
    owner: IdleSemanticArchiveOwner<M>,
    setup: UnusedIdleArchiveSetup,
    seal: UnusedArchiveSetupSeal,
    idle_identity: Weak<()>,
    scope: SearchAuthority,
    nonce: u64,
}
#[allow(clippy::large_enum_variant)]
enum WorkerCommand<M: RoleModel> {
    Maintenance(Command<M>),
    Setup(SetupCommand<M>),
}
#[allow(clippy::large_enum_variant)]
enum WorkerCompletion<M: RoleModel> {
    Maintenance(ArchiveCompletion<M>),
    Setup(UnusedIdleArchiveSetupCompletion<M>),
}
pub struct ArchiveWorker<M: RoleModel + 'static> {
    identity: Arc<()>,
    command: Option<SyncSender<WorkerCommand<M>>>,
    completion: Receiver<u64>,
    join: Option<JoinHandle<()>>,
    orphan: Arc<Mutex<Option<WorkerCompletion<M>>>>,
    next_nonce: u64,
    pending: Option<(u64, Arc<AtomicBool>)>,
    pending_setup: bool,
    shutdown_receipt: Option<ArchiveShutdownReceipt>,
    shutdown_unknown: bool,
}
impl<M: RoleModel + 'static> ArchiveWorker<M> {
    pub fn start() -> std::io::Result<Self> {
        Self::start_inner(None)
    }
    /// Connect the caller's existing completion signal without introducing a
    /// runtime dependency or dispatching any semantic work in this callback.
    pub fn start_with_notifier(
        notifier: impl Fn() + Send + Sync + 'static,
    ) -> std::io::Result<Self> {
        Self::start_inner(Some(Arc::new(notifier)))
    }
    fn start_inner(notifier: Option<Arc<dyn Fn() + Send + Sync>>) -> std::io::Result<Self> {
        let (command, receive_command) = mpsc::sync_channel::<WorkerCommand<M>>(1);
        let (send_completion, completion) = mpsc::sync_channel(1);
        let orphan: Arc<Mutex<Option<WorkerCompletion<M>>>> = Arc::new(Mutex::new(None));
        let thread_orphan = Arc::clone(&orphan);
        let join = thread::Builder::new()
            .name("rz-pals-archive-owner".into())
            .spawn(move || {
                while let Ok(command) = receive_command.recv() {
                    let (nonce, completed) = match command {
                        WorkerCommand::Maintenance(command) => {
                            let completed = execute(command);
                            (
                                completed.receipt.nonce,
                                WorkerCompletion::Maintenance(completed),
                            )
                        }
                        WorkerCommand::Setup(command) => {
                            let completed = execute_unused_setup(command);
                            (completed.receipt.nonce, WorkerCompletion::Setup(completed))
                        }
                    };
                    // Ownership always resides in the shared one-owner slot.
                    // The channel carries only notification, so dropping its
                    // receiver cannot destroy a queued session/Store owner.
                    *thread_orphan.lock().unwrap_or_else(|p| p.into_inner()) = Some(completed);
                    if send_completion.send(nonce).is_err() {
                        break;
                    }
                    if let Some(notifier) = &notifier {
                        notifier();
                    }
                }
                // Wake the existing manager signal on idle admission close as
                // well. The manager still checks is_finished before joining;
                // this notification itself is not a completion attestation.
                if let Some(notifier) = &notifier {
                    notifier();
                }
            })?;
        Ok(Self {
            identity: Arc::new(()),
            command: Some(command),
            completion,
            join: Some(join),
            orphan,
            next_nonce: 1,
            pending: None,
            pending_setup: false,
            shutdown_receipt: None,
            shutdown_unknown: false,
        })
    }
    pub fn is_busy(&self) -> bool {
        self.pending.is_some()
    }
    // Busy, closed and failed sends must return the exact original owner. A
    // second heap allocation on rejection would undermine bounded recovery.
    #[allow(clippy::result_large_err)]
    pub fn submit(
        &mut self,
        owner: ArchiveOwner<M>,
        authority: SearchAuthority,
        work: ArchiveWork,
    ) -> Result<ArchiveTransfer, ArchiveRejected<M>> {
        let reject = |owner, work, detail| ArchiveRejected {
            owner,
            work,
            error: PalsError::Store(StoreError::InvalidConditions(detail)),
        };
        if self.pending.is_some() {
            return Err(reject(owner, work, "archive worker busy"));
        }
        let Some(command) = &self.command else {
            return Err(reject(owner, work, "archive worker closed"));
        };
        let seal = match seal_owner(&owner, authority, &work) {
            Ok(seal) => seal,
            Err(error) => return Err(ArchiveRejected { owner, work, error }),
        };
        let nonce = self.next_nonce;
        let Some(next) = nonce.checked_add(1) else {
            return Err(reject(owner, work, "archive transfer sequence exhausted"));
        };
        let cancel = Arc::new(AtomicBool::new(false));
        let message = Command {
            owner,
            seal,
            nonce,
            work,
            cancel: Arc::clone(&cancel),
        };
        match command.try_send(WorkerCommand::Maintenance(message)) {
            Ok(()) => {
                self.next_nonce = next;
                self.pending = Some((nonce, Arc::clone(&cancel)));
                self.pending_setup = false;
                Ok(ArchiveTransfer {
                    nonce,
                    worker: Arc::clone(&self.identity),
                    cancel,
                })
            }
            Err(TrySendError::Full(WorkerCommand::Maintenance(message))) => {
                Err(reject(message.owner, message.work, "archive command full"))
            }
            Err(TrySendError::Disconnected(WorkerCommand::Maintenance(message))) => Err(reject(
                message.owner,
                message.work,
                "archive worker unavailable",
            )),
            Err(
                TrySendError::Full(WorkerCommand::Setup(_))
                | TrySendError::Disconnected(WorkerCommand::Setup(_)),
            ) => unreachable!("submitted maintenance variant"),
        }
    }
    #[allow(clippy::result_large_err)]
    pub fn submit_unused_idle_setup(
        &mut self,
        owner: IdleSemanticArchiveOwner<M>,
        scope: SearchAuthority,
        setup: UnusedIdleArchiveSetup,
    ) -> Result<ArchiveTransfer, UnusedIdleArchiveSetupRejected<M>> {
        let rejected = |owner, setup, error| UnusedIdleArchiveSetupRejected {
            owner,
            setup,
            error,
        };
        let invalid = |message| PalsError::Store(StoreError::InvalidConditions(message));
        if self.pending.is_some() {
            return Err(rejected(owner, setup, invalid("archive worker busy")));
        }
        let Some(command) = &self.command else {
            return Err(rejected(owner, setup, invalid("archive worker closed")));
        };
        if setup.control.max_io_bytes == 0 || setup.control.max_io_bytes > MAX_IO_BYTES {
            return Err(rejected(owner, setup, StoreError::ArchiveByteBudget.into()));
        }
        if let Err(error) = crate::pals::engine::archive_lifecycle::allocation_controls(Some((
            setup.control.deadline,
            &setup.control.cancellation,
        ))) {
            return Err(rejected(owner, setup, error.into()));
        }
        let seal = match owner.seal_unused_setup(scope) {
            Ok(seal) => seal,
            Err(error) => return Err(rejected(owner, setup, error)),
        };
        let nonce = self.next_nonce;
        let Some(next) = nonce.checked_add(1) else {
            return Err(rejected(
                owner,
                setup,
                invalid("archive transfer sequence exhausted"),
            ));
        };
        let cancel = Arc::clone(&setup.control.cancellation);
        let idle_identity = owner.identity();
        let message = WorkerCommand::Setup(SetupCommand {
            owner,
            setup,
            seal,
            idle_identity,
            scope,
            nonce,
        });
        match command.try_send(message) {
            Ok(()) => {
                self.next_nonce = next;
                self.pending = Some((nonce, Arc::clone(&cancel)));
                self.pending_setup = true;
                Ok(ArchiveTransfer {
                    nonce,
                    worker: Arc::clone(&self.identity),
                    cancel,
                })
            }
            Err(TrySendError::Full(WorkerCommand::Setup(message))) => Err(rejected(
                message.owner,
                message.setup,
                invalid("archive command full"),
            )),
            Err(TrySendError::Disconnected(WorkerCommand::Setup(message))) => Err(rejected(
                message.owner,
                message.setup,
                invalid("archive worker gone"),
            )),
            Err(
                TrySendError::Full(WorkerCommand::Maintenance(_))
                | TrySendError::Disconnected(WorkerCommand::Maintenance(_)),
            ) => unreachable!("submitted setup variant"),
        }
    }
    /// Nonblocking manager poll. mpsc supplies the existing bounded completion
    /// notification, with no detached ownership or manager tick sleep.
    pub fn try_receive(
        &mut self,
        transfer: &ArchiveTransfer,
    ) -> Result<Option<ArchiveCompletion<M>>, ArchiveWorkerError> {
        if self.pending_setup
            || !Arc::ptr_eq(&transfer.worker, &self.identity)
            || self.pending.as_ref().map(|p| p.0) != Some(transfer.nonce)
        {
            return Err(ArchiveWorkerError::StaleTransfer);
        }
        match self.completion.try_recv() {
            Ok(nonce) => {
                if nonce != transfer.nonce {
                    return Err(ArchiveWorkerError::CompletionUnknown);
                }
                let mut slot = self.orphan.lock().unwrap_or_else(|p| p.into_inner());
                if !matches!(slot.as_ref(), Some(WorkerCompletion::Maintenance(_))) {
                    return Err(ArchiveWorkerError::CompletionUnknown);
                }
                let Some(WorkerCompletion::Maintenance(completion)) = slot.take() else {
                    unreachable!("checked maintenance completion")
                };
                self.pending = None;
                Ok(Some(completion))
            }
            Err(TryRecvError::Empty) => Ok(None),
            Err(TryRecvError::Disconnected) => Err(ArchiveWorkerError::CompletionUnknown),
        }
    }
    pub fn try_receive_unused_idle_setup(
        &mut self,
        transfer: &ArchiveTransfer,
    ) -> Result<Option<UnusedIdleArchiveSetupCompletion<M>>, ArchiveWorkerError> {
        if !self.pending_setup
            || !Arc::ptr_eq(&transfer.worker, &self.identity)
            || self.pending.as_ref().map(|p| p.0) != Some(transfer.nonce)
        {
            return Err(ArchiveWorkerError::StaleTransfer);
        }
        match self.completion.try_recv() {
            Ok(nonce) => {
                if nonce != transfer.nonce {
                    return Err(ArchiveWorkerError::CompletionUnknown);
                }
                let mut slot = self.orphan.lock().unwrap_or_else(|p| p.into_inner());
                if !matches!(slot.as_ref(), Some(WorkerCompletion::Setup(_))) {
                    return Err(ArchiveWorkerError::CompletionUnknown);
                }
                let Some(WorkerCompletion::Setup(completion)) = slot.take() else {
                    unreachable!("checked setup completion")
                };
                self.pending = None;
                self.pending_setup = false;
                Ok(Some(completion))
            }
            Err(TryRecvError::Empty) => Ok(None),
            Err(TryRecvError::Disconnected) => Err(ArchiveWorkerError::CompletionUnknown),
        }
    }
    /// Admission close and cooperative cancel only. Neither dropping the last
    /// sender nor setting cancel proves that archive I/O physically completed.
    pub fn request_shutdown(&mut self) {
        self.command.take();
        if let Some((_, cancel)) = &self.pending {
            cancel.store(true, Ordering::Release);
        }
    }
    /// A manager-safe poll: unfinished worker or unreturned owner yields None.
    /// join is called only after JoinHandle reports the thread actually ended.
    pub fn try_join(&mut self) -> Result<Option<ArchiveShutdownReceipt>, ArchiveWorkerError> {
        if self.shutdown_unknown {
            return Err(ArchiveWorkerError::CompletionUnknown);
        }
        if let Some(receipt) = self.shutdown_receipt {
            return Ok(Some(receipt));
        }
        if self.pending.is_some() || self.command.is_some() {
            return Ok(None);
        }
        let Some(join) = &self.join else {
            return Err(ArchiveWorkerError::CompletionUnknown);
        };
        if !join.is_finished() {
            return Ok(None);
        }
        let joined = self
            .join
            .take()
            .expect("checked archive worker handle")
            .join();
        if joined.is_err() {
            self.shutdown_unknown = true;
            return Err(ArchiveWorkerError::CompletionUnknown);
        }
        let receipt = ArchiveShutdownReceipt {
            admission_closed: true,
            worker_joined: true,
            owners_returned: true,
        };
        self.shutdown_receipt = Some(receipt);
        Ok(Some(receipt))
    }
    /// Blocking helper for tests/off-manager callers after ownership return.
    /// The manager uses request_shutdown/try_join; an idle thread may still need
    /// scheduling before this join completes. A pending owner is never joined.
    pub fn finish(&mut self) -> Result<(), ArchiveWorkerError> {
        if self.pending.is_some() {
            return Err(ArchiveWorkerError::Busy);
        }
        self.request_shutdown();
        if self.shutdown_unknown {
            return Err(ArchiveWorkerError::CompletionUnknown);
        }
        if self.join.take().is_some_and(|join| join.join().is_err()) {
            self.shutdown_unknown = true;
            return Err(ArchiveWorkerError::CompletionUnknown);
        }
        self.shutdown_receipt = Some(ArchiveShutdownReceipt {
            admission_closed: true,
            worker_joined: true,
            owners_returned: true,
        });
        Ok(())
    }
}
impl<M: RoleModel + 'static> Drop for ArchiveWorker<M> {
    fn drop(&mut self) {
        if let Some((_, cancel)) = &self.pending {
            cancel.store(true, Ordering::Release);
            // Dropping an unknown owner must not drop physical resources. This
            // bounded one-owner quarantine is deliberate and process-scoped;
            // normal callers receive/drain then finish and never take this path.
            std::mem::forget(Arc::clone(&self.orphan));
        }
        self.command.take();
    }
}

fn execute_unused_setup<M: RoleModel>(
    mut command: SetupCommand<M>,
) -> UnusedIdleArchiveSetupCompletion<M> {
    let started = Instant::now();
    let mut result = catch_archive_work(|| {
        command
            .owner
            .execute_unused_setup(command.scope, &command.seal, &command.setup)
            .map_err(ArchiveQuarantineReason::Operation)
    });
    if matches!(result, Err(ArchiveQuarantineReason::Panic)) {
        command
            .owner
            .quarantine_unused_setup_panic(command.scope, &command.setup);
    }
    if result.is_ok() && command.setup.control.cancellation.load(Ordering::Acquire) {
        result = Err(ArchiveQuarantineReason::Canceled);
    }
    if result.is_ok() && Instant::now() >= command.setup.control.deadline {
        result = Err(ArchiveQuarantineReason::Deadline);
    }
    let actual = match command.owner.archive_owner_snapshot() {
        Ok(actual) => actual,
        Err(error) => {
            if result.is_ok() {
                result = Err(ArchiveQuarantineReason::Operation(error));
            }
            None
        }
    };
    if result.is_err() {
        command.owner.retain_unused_setup_failure();
    }
    let receipt = UnusedIdleArchiveSetupReceipt {
        nonce: command.nonce,
        scope: command.scope,
        identity: command.setup.identity,
        owner_kind: command.owner.owner_kind(),
        archive_owner: actual,
        remaining_io_bytes: command
            .owner
            .setup_snapshot()
            .map_or(Some(command.setup.control.max_io_bytes), |state| {
                state.remaining_io_bytes
            }),
        configuration_complete: result.is_ok(),
        elapsed: started.elapsed(),
    };
    UnusedIdleArchiveSetupCompletion {
        owner: command.owner,
        setup: command.setup,
        seal: command.seal,
        idle_identity: command.idle_identity,
        receipt,
        result,
    }
}

fn seal_owner<M: RoleModel>(
    owner: &ArchiveOwner<M>,
    authority: SearchAuthority,
    work: &ArchiveWork,
) -> Result<ArchiveSeal, PalsError> {
    if let ArchiveOwner::Idle(idle) = owner {
        return seal_idle_owner(idle, authority, work);
    }
    if matches!(
        work.operation,
        ArchiveOperation::RestoreRootIfArchived { .. } | ArchiveOperation::FinishIdleGame
    ) {
        return Err(StoreError::InvalidConditions(
            "operation requires original idle semantic owner",
        )
        .into());
    }
    let engine = owner.legacy_engine()?;
    let is_profiled_retirement = matches!(
        &work.operation,
        ArchiveOperation::PersistProfiledRetirement { .. }
    );
    if engine.semantic.game_generation != Some(authority.game.0)
        || engine.semantic.stores.generation() != authority.root.0
    {
        return Err(StoreError::StaleConsumer.into());
    }
    let allowance_ceiling = if is_profiled_retirement {
        MAX_PROFILED_RETIREMENT_IO_BYTES
    } else {
        MAX_IO_BYTES
    };
    if work.max_io_bytes == 0 || work.max_io_bytes > allowance_ceiling {
        return Err(StoreError::InvalidConditions("archive worker I/O allowance").into());
    }
    if Instant::now() >= work.deadline {
        return Err(StoreError::ArchiveDeadline.into());
    }
    let mut profiled_retirement_pins = StorePins::default();
    let mut profiled_retirement_identity = None;
    if let ArchiveOperation::PersistProfiledRetirement { payload } = &work.operation {
        if !matches!(owner, ArchiveOwner::Engine(_)) || payload.commit.is_some() {
            return Err(StoreError::InvalidConditions(
                "profiled retirement requires untouched payload and exclusive Engine owner",
            )
            .into());
        }
        payload.sink.validate_owner(
            &engine.semantic.stores,
            &payload.retirement,
            authority,
            work.deadline,
            work.max_io_bytes,
        )?;
        merge_pins(
            &mut profiled_retirement_pins,
            payload.retirement.protected_pins()?,
        );
        validate_pins(&engine.semantic.stores, &profiled_retirement_pins)?;
        profiled_retirement_identity = Some(payload.retirement.identity());
    }
    if let ArchiveOperation::PrepareForeign { issued, .. } = &work.operation {
        if work.deadline > issued.retain_until() {
            return Err(StoreError::ArchiveDeadline.into());
        }
    }
    if let ArchiveOperation::PrepareProfiledForeign { issued, .. } = &work.operation {
        if work.deadline > issued.retain_until() {
            return Err(StoreError::ArchiveDeadline.into());
        }
    }
    if let ArchiveOwner::Port(port) = owner {
        if port.authority != authority || work.deadline > port.limits.deadline {
            return Err(StoreError::StaleConsumer.into());
        }
        if !matches!(
            work.operation,
            ArchiveOperation::ReclaimHot { .. }
                | ArchiveOperation::PrepareForeign { .. }
                | ArchiveOperation::PrepareProfiledForeign { .. }
        ) {
            return Err(
                StoreError::InvalidConditions("active port cannot compact engine nodes").into(),
            );
        }
        if work.max_io_bytes > engine.archive_remaining_io() {
            return Err(StoreError::ArchiveByteBudget.into());
        }
    }
    if matches!(work.operation, ArchiveOperation::PrepareRoot { .. })
        && work.max_io_bytes != MAX_IO_BYTES
    {
        return Err(StoreError::InvalidConditions(
            "root preparation uses registered 16MiB allowance",
        )
        .into());
    }
    if matches!(
        work.operation,
        ArchiveOperation::PrepareRoot { .. }
            | ArchiveOperation::LoadRoot { .. }
            | ArchiveOperation::NewGame { .. }
    ) && engine
        .semantic
        .stores
        .tasks
        .active_states()
        .next()
        .is_some()
    {
        return Err(
            StoreError::InvalidConditions("root maintenance requires drained CPU tasks").into(),
        );
    }
    engine.ensure_archive_admission_open()?;
    let archive_owner = engine.archive_owner_snapshot()?;
    if !is_profiled_retirement
        && !matches!(work.operation, ArchiveOperation::NewGame { .. })
        && (archive_owner.is_none() || !engine.archive_enabled())
    {
        return Err(StoreError::ArchiveDisabled.into());
    }
    if archive_owner.is_some_and(|owner| owner.admission_closed) {
        return Err(StoreError::ArchiveDisabled.into());
    }
    let next_game = if let ArchiveOperation::NewGame { next_authority } = work.operation {
        if engine.adaptive_archive_authority(next_authority)? != authority {
            return Err(StoreError::StaleConsumer.into());
        }
        Some(next_authority)
    } else {
        None
    };
    if let ArchiveOperation::ReclaimHot { incoming_pins } = &work.operation {
        validate_pins(&engine.semantic.stores, incoming_pins)?;
    }
    let mut active_cpu_pins = StorePins::default();
    if let ArchiveOwner::Port(port) = owner {
        merge_pins(&mut active_cpu_pins, &port.goal_local_protected_pins()?);
        merge_pins(
            &mut active_cpu_pins,
            &port.profiled_external_protected_pins()?,
        );
        for admission in port.cpu_admissions.values() {
            active_cpu_pins.executions.extend(admission.execution);
            if let Some(execution) = admission.execution {
                active_cpu_pins
                    .states
                    .insert(engine.semantic.stores.tasks.get(execution)?.key.state);
            }
        }
        validate_pins(&engine.semantic.stores, &active_cpu_pins)?;
    }
    Ok(ArchiveSeal {
        authority,
        store_owner: engine.semantic.stores.read_owner(),
        store_generation: engine.semantic.stores.generation(),
        game_generation: engine.semantic.game_generation,
        root: engine.semantic.stores.root(),
        revision: engine.semantic.revision,
        node_handles: engine
            .semantic
            .nodes
            .iter()
            .map(|node| (node.state, node.situation))
            .collect(),
        record_revisions: engine
            .semantic
            .records
            .iter()
            .map(|record| record.revision)
            .collect(),
        review_pins: engine.semantic.stores.review_pin_union()?,
        active_cpu_pins,
        profiled_retirement_pins,
        profiled_retirement_identity,
        archive_owner,
        registry_owner: engine.adaptive_registry_identity(),
        next_game,
        is_port: matches!(owner, ArchiveOwner::Port(_)),
        idle_owner: None,
    })
}

fn seal_idle_owner<M: RoleModel>(
    owner: &IdleSemanticArchiveOwner<M>,
    authority: SearchAuthority,
    work: &ArchiveWork,
) -> Result<ArchiveSeal, PalsError> {
    let transition = match work.operation {
        ArchiveOperation::NewGame { next_authority } => {
            Some(IdleArchiveTransitionKind::NewGame(next_authority))
        }
        ArchiveOperation::FinishIdleGame => Some(IdleArchiveTransitionKind::Finish),
        _ => None,
    };
    if let Some(kind) = transition {
        owner.validate_transition_request(kind, work.deadline)?;
    } else {
        owner.validate_idle()?;
    }
    let semantic = owner.semantic();
    let last = semantic
        .last_adaptive_authority()
        .ok_or(StoreError::StaleConsumer)?;
    if last.epoch != authority.epoch
        || last.implementation != authority.implementation
        || last.game != authority.game
        || semantic.game_generation != Some(authority.game.0)
        || owner.stores().generation() != authority.root.0
    {
        return Err(StoreError::StaleConsumer.into());
    }
    if !matches!(
        work.operation,
        ArchiveOperation::PrepareRoot { .. }
            | ArchiveOperation::LoadRoot { .. }
            | ArchiveOperation::RestoreRootIfArchived { .. }
            | ArchiveOperation::NewGame { .. }
            | ArchiveOperation::FinishIdleGame
    ) {
        return Err(StoreError::InvalidConditions("unsupported idle archive operation").into());
    }
    if work.max_io_bytes == 0
        || work.max_io_bytes > MAX_IO_BYTES
        || ((matches!(work.operation, ArchiveOperation::PrepareRoot { .. })
            || transition.is_some())
            && work.max_io_bytes != MAX_IO_BYTES)
    {
        return Err(StoreError::ArchiveByteBudget.into());
    }
    if Instant::now() >= work.deadline {
        return Err(StoreError::ArchiveDeadline.into());
    }
    let archive_owner = owner.archive_owner_snapshot()?;
    let pending = owner.transition_snapshot()?;
    if archive_owner.is_none()
        || ((!owner.archive_enabled() || archive_owner.is_some_and(|value| value.admission_closed))
            && pending.is_none())
    {
        return Err(StoreError::ArchiveDisabled.into());
    }
    if let Some(IdleArchiveTransitionKind::NewGame(next)) = transition {
        if owner.adaptive_archive_authority(next)? != authority {
            return Err(StoreError::StaleConsumer.into());
        }
    }
    let mut review_pins = owner.projection_pins()?;
    merge_pins(&mut review_pins, &owner.stores().review_pin_union()?);
    validate_pins(owner.stores(), &review_pins)?;
    Ok(ArchiveSeal {
        authority,
        store_owner: owner.stores().read_owner(),
        store_generation: owner.stores().generation(),
        game_generation: semantic.game_generation,
        root: owner.stores().root(),
        revision: semantic.revision,
        node_handles: semantic
            .nodes
            .iter()
            .map(|node| (node.state, node.situation))
            .collect(),
        record_revisions: semantic
            .records
            .iter()
            .map(|record| record.revision)
            .collect(),
        review_pins,
        active_cpu_pins: StorePins::default(),
        profiled_retirement_pins: StorePins::default(),
        profiled_retirement_identity: None,
        archive_owner,
        registry_owner: semantic.adaptive_registry_identity(),
        next_game: match transition {
            Some(IdleArchiveTransitionKind::NewGame(next)) => Some(next),
            _ => None,
        },
        is_port: false,
        idle_owner: Some(owner.identity()),
    })
}

fn validate_pins(
    stores: &crate::pals::store::PalsStores,
    pins: &StorePins,
) -> Result<(), StoreError> {
    for &id in &pins.states {
        stores.states.get(id)?;
    }
    for &id in &pins.lines {
        stores.lines.get(id)?;
    }
    for &id in &pins.situations {
        stores.situations.get(id)?;
    }
    for &id in &pins.observations {
        stores.observations.get(id)?;
    }
    for &id in &pins.executions {
        stores.tasks.get(id)?;
    }
    Ok(())
}
fn merge_pins(target: &mut StorePins, additional: &StorePins) {
    target.states.extend(&additional.states);
    target.lines.extend(&additional.lines);
    target.situations.extend(&additional.situations);
    target.observations.extend(&additional.observations);
    target.executions.extend(&additional.executions);
}

fn catch_archive_work(
    operation: impl FnOnce() -> Result<(), ArchiveQuarantineReason>,
) -> Result<(), ArchiveQuarantineReason> {
    catch_unwind(AssertUnwindSafe(operation)).unwrap_or(Err(ArchiveQuarantineReason::Panic))
}

fn execute_idle<M: RoleModel>(mut command: Command<M>) -> ArchiveCompletion<M> {
    let began = Instant::now();
    let ArchiveOwner::Idle(owner) = &mut command.owner else {
        unreachable!("only the sealed idle branch calls execute_idle");
    };
    let before = owner.stores().hot_stats();
    let kind = command.work.operation.kind();
    let mut loaded_root = None;
    let transition = matches!(
        kind,
        ArchiveOperationKind::NewGame | ArchiveOperationKind::FinishIdleGame
    );
    let available = owner
        .pending_transition_io()
        .filter(|_| transition)
        .unwrap_or(command.work.max_io_bytes);
    let mut budget = ArchiveIoBudget {
        deadline: command.work.deadline,
        max_bytes: available,
    };
    let result = catch_archive_work(|| {
        if command.cancel.load(Ordering::Acquire) {
            return Err(ArchiveQuarantineReason::Canceled);
        }
        if Instant::now() >= command.work.deadline {
            return Err(ArchiveQuarantineReason::Deadline);
        }
        let operation = match command.work.operation {
            ArchiveOperation::PrepareRoot { position } => {
                let result = owner.prepare_root(&position, command.work.deadline, &command.cancel);
                budget.max_bytes = owner.remaining_io();
                result
            }
            ArchiveOperation::LoadRoot { position } => owner
                .load_root(&position, &mut budget)
                .map(|root| loaded_root = Some(root)),
            ArchiveOperation::RestoreRootIfArchived { position } => owner
                .restore_root_if_archived(&position, &mut budget)
                .map(|root| loaded_root = root),
            ArchiveOperation::NewGame { next_authority } => {
                let result = owner.advance_transition(
                    IdleArchiveTransitionKind::NewGame(next_authority),
                    command.work.deadline,
                    &command.cancel,
                );
                budget.max_bytes = owner.transition_remaining_io();
                result
            }
            ArchiveOperation::FinishIdleGame => {
                let result = owner.advance_transition(
                    IdleArchiveTransitionKind::Finish,
                    command.work.deadline,
                    &command.cancel,
                );
                budget.max_bytes = owner.transition_remaining_io();
                result
            }
            _ => Err(StoreError::InvalidConditions("unsealed idle archive operation").into()),
        };
        operation.map_err(ArchiveQuarantineReason::Operation)?;
        if command.cancel.load(Ordering::Acquire) {
            return Err(ArchiveQuarantineReason::Canceled);
        }
        if Instant::now() >= command.work.deadline {
            return Err(ArchiveQuarantineReason::Deadline);
        }
        Ok(())
    });
    if transition && matches!(result, Err(ArchiveQuarantineReason::Panic)) {
        owner.quarantine_transition_unknown();
    }
    let receipt = ArchiveWorkReceipt {
        nonce: command.nonce,
        authority: command.seal.authority,
        operation: kind,
        generation_before: command.seal.store_generation,
        generation_after: owner.stores().generation(),
        hot_before: before,
        hot_after: owner.stores().hot_stats(),
        io_bytes: if matches!(result, Err(ArchiveQuarantineReason::Panic))
            || (kind == ArchiveOperationKind::PrepareRoot && result.is_err())
        {
            None
        } else {
            available.checked_sub(budget.max_bytes)
        },
        elapsed: began.elapsed(),
        loaded_root,
        archive_owner: owner.archive_owner_snapshot().ok().flatten(),
        next_authority: command.seal.next_game,
        idle_checkpoint: if transition {
            owner.last_checkpoint()
        } else {
            None
        },
    };
    ArchiveCompletion {
        owner: command.owner,
        seal: command.seal,
        receipt,
        result,
        deadline: command.work.deadline,
        cancel: command.cancel,
        foreign_permit: None,
        profiled_foreign_permit: None,
        profiled_retirement: None,
    }
}

fn execute<M: RoleModel>(mut command: Command<M>) -> ArchiveCompletion<M> {
    if matches!(command.owner, ArchiveOwner::Idle(_)) {
        return execute_idle(command);
    }
    let began = Instant::now();
    let before = command
        .owner
        .legacy_engine()
        .expect("sealed Engine/Port owner")
        .stores()
        .hot_stats();
    let mut budget = ArchiveIoBudget {
        deadline: command.work.deadline,
        max_bytes: command.work.max_io_bytes,
    };
    let kind = command.work.operation.kind();
    // Keep the original source/sink/commit outside the unwind closure. Merely
    // moving this operation into match inside it would drop those owners on a
    // panic or an early pre-operation cancel/deadline return.
    let (operation, mut profiled_retirement) = match command.work.operation {
        ArchiveOperation::PersistProfiledRetirement { payload } => (None, Some(payload)),
        operation => (Some(operation), None),
    };
    let profiled_io_before = profiled_retirement
        .as_ref()
        .and_then(|payload| payload.sink.data_io_bytes());
    let mut loaded_root = None;
    let mut foreign_permit = None;
    let mut profiled_foreign_permit = None;
    let result = catch_archive_work(|| {
        if command.cancel.load(Ordering::Acquire) {
            return Err(ArchiveQuarantineReason::Canceled);
        }
        if Instant::now() >= command.work.deadline {
            return Err(ArchiveQuarantineReason::Deadline);
        }
        let engine = command
            .owner
            .legacy_engine_mut()
            .map_err(ArchiveQuarantineReason::Operation)?;
        let operation: Result<(), PalsError> = if let Some(payload) = profiled_retirement.as_mut() {
            // persist borrows the move-only owners and returns the actual ACK.
            // Save that ACK before any post-operation cancellation checks.
            payload
                .sink
                .persist(
                    &engine.semantic.stores,
                    &payload.retirement,
                    &mut budget,
                    &command.cancel,
                )
                .map(|commit| payload.commit = Some(commit))
        } else {
            (|| match operation {
                None
                | Some(ArchiveOperation::PersistProfiledRetirement { .. })
                | Some(ArchiveOperation::RestoreRootIfArchived { .. })
                | Some(ArchiveOperation::FinishIdleGame) => {
                    Err(StoreError::InvalidConditions("archive operation payload shape").into())
                }
                Some(ArchiveOperation::PrepareRoot { position }) => {
                    let result = engine.prepare_archive_root(
                        &position,
                        command.work.deadline,
                        &command.cancel,
                    );
                    budget.max_bytes = engine.archive_remaining_io();
                    result
                }
                Some(ArchiveOperation::LoadRoot { position }) => engine
                    .load_archived_root_with_budget(&position, &mut budget)
                    .map(|root| {
                        loaded_root = Some(root);
                    }),
                Some(ArchiveOperation::ReclaimHot { incoming_pins }) => {
                    let mut pins = engine.resident_archive_pins()?;
                    merge_pins(&mut pins, &incoming_pins);
                    // Independent review leases are already protected by Store;
                    // adding the submission union also covers release during I/O.
                    merge_pins(&mut pins, &command.seal.review_pins);
                    merge_pins(&mut pins, &command.seal.active_cpu_pins);
                    engine.semantic.stores.set_archive_pins(pins)?;
                    let initial = budget.max_bytes;
                    let result = engine
                        .semantic
                        .stores
                        .archive_inactive_with_budget(&mut budget);
                    if command.seal.is_port {
                        engine
                            .charge_engine_archive_io(initial.saturating_sub(budget.max_bytes))?;
                    }
                    result?;
                    Ok(())
                }
                Some(ArchiveOperation::PrepareForeign {
                    observation,
                    issued,
                }) => {
                    let initial = budget.max_bytes;
                    let prepared = engine.semantic.stores.prepare_foreign_append_permit(
                        observation,
                        issued,
                        &mut budget,
                    );
                    if command.seal.is_port {
                        engine
                            .charge_engine_archive_io(initial.saturating_sub(budget.max_bytes))?;
                    }
                    foreign_permit = Some(prepared?);
                    Ok(())
                }
                Some(ArchiveOperation::PrepareProfiledForeign {
                    observation,
                    issued,
                }) => {
                    let initial = budget.max_bytes;
                    let prepared = engine
                        .semantic
                        .stores
                        .prepare_profiled_foreign_append_permit(observation, issued, &mut budget)
                        .map(|permit| {
                            // Preserve the actual scan result even if subsequent I/O
                            // accounting fails and this owner must be quarantined.
                            profiled_foreign_permit = Some(permit);
                        });
                    if command.seal.is_port {
                        engine
                            .charge_engine_archive_io(initial.saturating_sub(budget.max_bytes))?;
                    }
                    prepared?;
                    Ok(())
                }
                Some(ArchiveOperation::NewGame { next_authority }) => engine.try_adaptive_new_game(
                    next_authority,
                    command.work.deadline,
                    &command.cancel,
                ),
            })()
        };
        operation.map_err(ArchiveQuarantineReason::Operation)?;
        if command.cancel.load(Ordering::Acquire) {
            return Err(ArchiveQuarantineReason::Canceled);
        }
        if Instant::now() >= command.work.deadline {
            return Err(ArchiveQuarantineReason::Deadline);
        }
        Ok(())
    });
    if matches!(result, Err(ArchiveQuarantineReason::Panic)) {
        if let Some(payload) = profiled_retirement.as_mut() {
            // A panic may interrupt file/hash/sync work. Preserve its sink and
            // descriptors without treating the remaining reservation as free.
            payload.sink.quarantine_completion_unknown();
            budget.max_bytes = 0;
        }
    }
    let engine = command
        .owner
        .legacy_engine()
        .expect("original Engine/Port owner");
    let receipt = ArchiveWorkReceipt {
        nonce: command.nonce,
        authority: command.seal.authority,
        operation: kind,
        generation_before: command.seal.store_generation,
        generation_after: engine.semantic.stores.generation(),
        hot_before: before,
        hot_after: engine.semantic.stores.hot_stats(),
        io_bytes: if matches!(result, Err(ArchiveQuarantineReason::Panic))
            || (kind == ArchiveOperationKind::PrepareRoot && result.is_err())
            || kind == ArchiveOperationKind::NewGame
        {
            None
        } else if let Some(payload) = &profiled_retirement {
            profiled_io_before.and_then(|before| payload.sink.data_io_bytes()?.checked_sub(before))
        } else {
            command.work.max_io_bytes.checked_sub(budget.max_bytes)
        },
        elapsed: began.elapsed(),
        loaded_root,
        archive_owner: engine.archive_owner_snapshot().ok().flatten(),
        next_authority: command.seal.next_game,
        idle_checkpoint: None,
    };
    ArchiveCompletion {
        owner: command.owner,
        seal: command.seal,
        receipt,
        result,
        deadline: command.work.deadline,
        cancel: command.cancel,
        foreign_permit,
        profiled_foreign_permit,
        profiled_retirement,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::CpuConfig;
    use crate::pals::engine::{LegalOrderRoleMock, PalsConfig};
    use crate::pals::store::{ArchiveConfig, PalsStores, StoreLimits};
    use rz_contracts::{Digest, GameGeneration, ProcessEpoch, RootGeneration};
    use std::path::PathBuf;
    use std::sync::atomic::AtomicU64;

    struct Scratch(PathBuf);
    impl Drop for Scratch {
        fn drop(&mut self) {
            // This exact unique test directory was created by this fixture.
            // No cache, source, foreign generation or report root is traversed.
            if self.0.file_name().is_some_and(|name| {
                name.to_string_lossy()
                    .starts_with("adaptive-archive-worker-")
            }) {
                let _ = std::fs::remove_dir_all(&self.0);
            }
        }
    }
    fn fixture(archive: bool) -> (PalsEngine<LegalOrderRoleMock>, Scratch) {
        static SERIAL: AtomicU64 = AtomicU64::new(0);
        let output = if cfg!(windows) {
            std::env::var_os("APPDATA").map(PathBuf::from)
        } else {
            std::env::var_os("RUNNER_TEMP").map(PathBuf::from)
        }
        .unwrap_or_else(std::env::temp_dir);
        let root = output.join("RoveZero").join("tmp").join(format!(
            "adaptive-archive-worker-{}-{}",
            std::process::id(),
            SERIAL.fetch_add(1, Ordering::Relaxed)
        ));
        let mut engine = PalsEngine::new(
            PalsConfig {
                beam_width: 1,
                line_plies: 2,
                max_nodes: 257,
                max_records: 32,
                max_role_calls: 64,
                cpu_nodes_per_task: 256,
            },
            LegalOrderRoleMock,
            crate::cpu::CpuEngine::new(CpuConfig {
                tt_entries: 64,
                max_depth: 4,
                quiescence_ply: 4,
                ..CpuConfig::default()
            })
            .unwrap(),
        )
        .unwrap();
        engine.semantic.stores = PalsStores::new(StoreLimits {
            states: 8,
            situations: 8,
            line_chunks: 16,
            observations: 8,
            ..StoreLimits::default()
        });
        if archive {
            let repository = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .unwrap()
                .parent()
                .unwrap()
                .to_owned();
            engine
                .enable_archive(ArchiveConfig::new(root.clone(), repository))
                .unwrap();
        }
        engine
            .semantic
            .stores
            .focus_actual_moves(Position::startpos().snapshot())
            .unwrap();
        (engine, Scratch(root))
    }
    fn authority(engine: &PalsEngine<LegalOrderRoleMock>) -> SearchAuthority {
        SearchAuthority {
            epoch: ProcessEpoch(1),
            game: GameGeneration(engine.semantic.game_generation.unwrap()),
            root: RootGeneration(engine.semantic.stores.generation()),
            implementation: Digest([7; 32]),
        }
    }
    fn populate(engine: &mut PalsEngine<LegalOrderRoleMock>) -> Vec<crate::pals::store::StateId> {
        (100..105)
            .map(|fullmove| {
                let position = Position::from_fen(&format!(
                    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 {fullmove}"
                ))
                .unwrap();
                let situation = engine
                    .semantic
                    .stores
                    .insert_situation(position.snapshot())
                    .unwrap();
                engine
                    .semantic
                    .stores
                    .situations
                    .get(situation)
                    .unwrap()
                    .state
            })
            .collect()
    }
    fn work(bytes: u64) -> ArchiveWork {
        ArchiveWork {
            operation: ArchiveOperation::ReclaimHot {
                incoming_pins: StorePins::default(),
            },
            deadline: Instant::now() + Duration::from_secs(10),
            max_io_bytes: bytes,
        }
    }
    fn engine_unused_setup_fixture() -> (
        Box<PalsEngine<LegalOrderRoleMock>>,
        Scratch,
        SearchAuthority,
        UnusedIdleArchiveSetup,
    ) {
        use crate::pals::engine::idle_archive::UnusedIdleArchiveSetupControl;
        use crate::pals::review_context::{ReviewIndexLimits, ReviewSelectionPolicy};
        use crate::pals::store::{ArchiveRootCapability, ArchiveRuntimeLimits};
        let (mut engine, scratch) = fixture(false);
        let scope = SearchAuthority {
            epoch: ProcessEpoch(1),
            game: GameGeneration(1),
            root: RootGeneration(1),
            implementation: Digest([7; 32]),
        };
        let until = Instant::now() + Duration::from_secs(20);
        engine
            .try_adaptive_new_game(scope, until, &AtomicBool::new(false))
            .unwrap();
        let engine = Box::new(
            engine
                .with_selected_review_context(
                    ReviewSelectionPolicy::default(),
                    ReviewIndexLimits::default(),
                )
                .unwrap(),
        );
        std::fs::create_dir_all(&scratch.0).unwrap();
        let directory =
            cap_std::fs::Dir::open_ambient_dir(&scratch.0, cap_std::ambient_authority()).unwrap();
        let setup = UnusedIdleArchiveSetup::new(
            ArchiveConfig::new(scratch.0.clone(), PathBuf::from(env!("CARGO_MANIFEST_DIR"))),
            ArchiveRootCapability::from_directory(scratch.0.clone(), directory).unwrap(),
            ArchiveRuntimeLimits {
                game_bytes_max: 1 << 20,
                global_bytes_max: 4 << 20,
                index_entries_max: 256,
                index_bytes_max: 512 << 10,
                load_bytes_max: 512 << 10,
                load_deadline_max_ms: 20_000,
                record_payload_bytes_max: 512 << 10,
                max_load_pins: 16,
            },
            UnusedIdleArchiveSetupIdentity {
                profile: Digest([82; 32]),
                input_lock: Digest([83; 32]),
                registration: Digest([84; 32]),
            },
            UnusedIdleArchiveSetupControl {
                deadline: until,
                cancellation: Arc::new(AtomicBool::new(false)),
                max_io_bytes: MAX_IO_BYTES,
            },
        );
        (engine, scratch, scope, setup)
    }
    fn receive_engine_unused_setup(
        worker: &mut ArchiveWorker<LegalOrderRoleMock>,
        transfer: &ArchiveTransfer,
    ) -> UnusedIdleArchiveSetupCompletion<LegalOrderRoleMock> {
        let until = Instant::now() + Duration::from_secs(20);
        loop {
            if let Some(completed) = worker.try_receive_unused_idle_setup(transfer).unwrap() {
                return completed;
            }
            assert!(Instant::now() < until, "bounded Engine setup completion");
            thread::sleep(Duration::from_millis(1));
        }
    }

    #[test]
    fn engine_idle_setup_uses_original_worker_nonce_and_returns_same_full_box() {
        let (engine, scratch, scope, setup) = engine_unused_setup_fixture();
        let original = std::ptr::from_ref(engine.as_ref());
        let store = engine.stores().read_owner();
        let checker = std::ptr::from_ref(engine.researcher.cpu.as_ref());
        let identity = setup.identity();
        let idle = engine
            .into_idle_archive()
            .unwrap_or_else(|_| panic!("original Engine idle"));
        let mut worker = ArchiveWorker::start().unwrap();
        let transfer = worker
            .submit_unused_idle_setup(idle, scope, setup)
            .unwrap_or_else(|_| panic!("original unused setup"));
        let completed = receive_engine_unused_setup(&mut worker, &transfer);
        let (mut idle, receipt) = completed
            .install(scope, transfer.nonce(), identity)
            .unwrap_or_else(|_| panic!("original Engine setup install"));
        assert_eq!(receipt.owner_kind, IdleArchiveOwnerKind::EnginePcr);
        assert_eq!(receipt.nonce, transfer.nonce());
        assert!(receipt.configuration_complete);
        assert!(store.matches(idle.stores()));
        assert!(idle.last_registered_scope().is_none());
        assert_eq!(idle.stores().generation(), 0);
        assert!(idle.stores().root().is_none());
        assert!(
            idle.close_archive_owner()
                .unwrap()
                .unwrap()
                .admission_closed
        );
        worker.finish().unwrap();
        let engine = idle.into_engine().unwrap_or_else(|_| panic!("same Engine"));
        assert_eq!(std::ptr::from_ref(engine.as_ref()), original);
        assert!(std::ptr::eq(
            std::ptr::from_ref(engine.researcher.cpu.as_ref()),
            checker
        ));
        assert!(store.matches(engine.stores()));
        drop(engine);
        drop(scratch);
    }

    #[test]
    fn engine_idle_wrong_setup_install_retains_original_engine_payload_and_first_cause() {
        let (engine, scratch, scope, setup) = engine_unused_setup_fixture();
        let store = engine.stores().read_owner();
        let identity = setup.identity();
        let idle = engine
            .into_idle_archive()
            .unwrap_or_else(|_| panic!("original Engine"));
        let mut worker = ArchiveWorker::start().unwrap();
        let transfer = worker
            .submit_unused_idle_setup(idle, scope, setup)
            .unwrap_or_else(|_| panic!("unused setup"));
        let completed = receive_engine_unused_setup(&mut worker, &transfer);
        let rejected = completed
            .install(
                SearchAuthority {
                    root: RootGeneration(99),
                    ..scope
                },
                transfer.nonce(),
                identity,
            )
            .err()
            .expect("wrong install rejected");
        assert!(matches!(
            rejected.failure(),
            Some(ArchiveQuarantineReason::InvalidReturn(
                "unused setup identity/nonce"
            ))
        ));
        let original = rejected.into_owner_and_setup_for_shutdown();
        assert!(store.matches(original.owner.stores()));
        assert_eq!(original.receipt.owner_kind, IdleArchiveOwnerKind::EnginePcr);
        assert_eq!(original.setup.identity(), identity);
        assert!(original.result.is_err());
        assert!(original.owner.into_engine().is_err());
        worker.finish().unwrap();
        std::mem::forget(scratch); // preserve the actual failed/partial archive fixture
    }

    #[test]
    fn engine_idle_setup_panic_keeps_partial_archive_original_store_and_payload() {
        let (engine, scratch, scope, setup) = engine_unused_setup_fixture();
        let store = engine.stores().read_owner();
        let identity = setup.identity();
        let mut idle = engine
            .into_idle_archive()
            .unwrap_or_else(|_| panic!("original Engine"));
        idle.panic_setup_after_enable_for_test();
        let mut worker = ArchiveWorker::start().unwrap();
        let transfer = worker
            .submit_unused_idle_setup(idle, scope, setup)
            .unwrap_or_else(|_| panic!("original setup"));
        let rejected = receive_engine_unused_setup(&mut worker, &transfer)
            .install(scope, transfer.nonce(), identity)
            .err()
            .expect("partial panic retained");
        assert!(matches!(
            rejected.failure(),
            Some(ArchiveQuarantineReason::Panic)
        ));
        let parts = rejected.into_owner_and_setup_for_shutdown();
        assert!(store.matches(parts.owner.stores()));
        assert_eq!(parts.receipt.owner_kind, IdleArchiveOwnerKind::EnginePcr);
        assert_eq!(parts.setup.identity(), identity);
        assert!(parts.owner.setup_snapshot().unwrap().completion_unknown);
        assert!(parts.owner.archive_owner_snapshot().unwrap().is_some());
        assert!(parts.owner.into_engine().is_err());
        worker.finish().unwrap();
        std::mem::forget(scratch);
    }

    #[test]
    fn legacy_engine_submit_still_rejects_idle_only_restore_and_finish() {
        for operation in [
            ArchiveOperation::RestoreRootIfArchived {
                position: Position::startpos(),
            },
            ArchiveOperation::FinishIdleGame,
        ] {
            let (engine, scratch, _, _) = engine_unused_setup_fixture();
            let authority = authority(&engine);
            let original = engine.stores().read_owner();
            let mut worker = ArchiveWorker::start().unwrap();
            let rejected = worker
                .submit(
                    ArchiveOwner::Engine(*engine),
                    authority,
                    ArchiveWork {
                        operation,
                        deadline: Instant::now() + Duration::from_secs(20),
                        max_io_bytes: MAX_IO_BYTES,
                    },
                )
                .err()
                .expect("ordinary Engine is not explicit Engine-idle");
            assert!(matches!(
                rejected.error,
                PalsError::Store(StoreError::InvalidConditions(
                    "operation requires original idle semantic owner"
                ))
            ));
            let ArchiveOwner::Engine(engine) = rejected.owner else {
                panic!("original Engine kind")
            };
            assert!(original.matches(engine.stores()));
            worker.finish().unwrap();
            drop(engine);
            drop(scratch);
        }
    }
    fn receive(
        worker: &mut ArchiveWorker<LegalOrderRoleMock>,
        transfer: &ArchiveTransfer,
    ) -> ArchiveCompletion<LegalOrderRoleMock> {
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            if let Some(completion) = worker.try_receive(transfer).unwrap() {
                return completion;
            }
            assert!(Instant::now() < deadline, "bounded archive test completion");
            thread::sleep(Duration::from_millis(1));
        }
    }

    #[test]
    fn actual_archive_reclaim_returns_same_owner_and_preserves_submitted_pins() {
        let (mut engine, scratch) = fixture(true);
        let inactive = populate(&mut engine);
        let original_owner = engine.semantic.stores.read_owner();
        let mut review = StorePins::default();
        review.states.insert(inactive[1]);
        let lease = engine.semantic.stores.pin_review(review).unwrap();
        let mut incoming = StorePins::default();
        incoming.states.insert(inactive[0]);
        let authority = authority(&engine);
        let mut worker = ArchiveWorker::start().unwrap();
        let transfer = worker
            .submit(
                ArchiveOwner::Engine(engine),
                authority,
                ArchiveWork {
                    operation: ArchiveOperation::ReclaimHot {
                        incoming_pins: incoming,
                    },
                    ..work(MAX_IO_BYTES)
                },
            )
            .unwrap_or_else(|_| panic!("fixture transfer rejected"));
        let completion = receive(&mut worker, &transfer);
        let (owner, receipt) = completion
            .install(authority, transfer.nonce())
            .unwrap_or_else(|_| panic!("actual verified archive return rejected"));
        let ArchiveOwner::Engine(engine) = owner else {
            panic!("owner kind changed")
        };
        assert!(original_owner.matches(&engine.semantic.stores));
        assert!(lease.validates(&engine.semantic.stores));
        assert_eq!(receipt.generation_before, receipt.generation_after);
        assert!(receipt.io_bytes.is_some_and(|bytes| bytes > 0));
        assert!(receipt.hot_after.states < receipt.hot_before.states);
        assert!(engine.semantic.stores.states.get(inactive[0]).is_ok());
        assert!(engine.semantic.stores.states.get(inactive[1]).is_ok());
        assert!(matches!(
            engine.semantic.stores.states.get(inactive[2]),
            Err(StoreError::ColdRecord(_))
        ));
        assert_eq!(engine.semantic.revision, 0);
        assert!(engine.semantic.nodes.is_empty() && engine.semantic.records.is_empty());
        drop(lease);
        worker.finish().unwrap();
        drop(engine);
        drop(scratch);
    }

    #[test]
    fn archive_disabled_rejection_returns_original_store_without_fallback() {
        let (engine, scratch) = fixture(false);
        let source = engine.semantic.stores.read_owner();
        let authority = authority(&engine);
        let mut worker = ArchiveWorker::start().unwrap();
        let rejected =
            match worker.submit(ArchiveOwner::Engine(engine), authority, work(MAX_IO_BYTES)) {
                Err(rejected) => rejected,
                Ok(_) => panic!("disabled archive accepted"),
            };
        assert!(matches!(
            rejected.error,
            PalsError::Store(StoreError::ArchiveDisabled)
        ));
        assert!(source.matches(rejected.owner.engine().stores()));
        assert!(!worker.is_busy());
        worker.finish().unwrap();
        drop(rejected);
        drop(scratch);
    }

    #[test]
    fn failed_io_allowance_preserves_hot_facts_in_quarantine() {
        let (mut engine, scratch) = fixture(true);
        populate(&mut engine);
        let authority = authority(&engine);
        let source = engine.semantic.stores.read_owner();
        let before = engine.semantic.stores.hot_stats();
        let mut worker = ArchiveWorker::start().unwrap();
        let transfer = worker
            .submit(ArchiveOwner::Engine(engine), authority, work(1))
            .unwrap_or_else(|_| panic!("registered small allowance rejected"));
        let quarantine = match receive(&mut worker, &transfer).install(authority, transfer.nonce())
        {
            Err(quarantine) => quarantine,
            Ok(_) => panic!("impossible allowance treated as success"),
        };
        assert!(matches!(
            quarantine.reason,
            ArchiveQuarantineReason::Operation(_)
        ));
        let owner = quarantine
            .into_owner_for_shutdown()
            .unwrap_or_else(|_| panic!("ordinary archive quarantine retained retirement payload"));
        assert!(source.matches(owner.engine().stores()));
        assert_eq!(owner.engine().stores().hot_stats(), before);
        worker.finish().unwrap();
        drop(owner);
        drop(scratch);
    }

    #[test]
    fn ordinary_archive_completion_cannot_issue_a_profiled_permit() {
        let (mut engine, scratch) = fixture(true);
        populate(&mut engine);
        let authority = authority(&engine);
        let source = engine.semantic.stores.read_owner();
        let mut worker = ArchiveWorker::start().unwrap();
        let transfer = worker
            .submit(ArchiveOwner::Engine(engine), authority, work(MAX_IO_BYTES))
            .unwrap_or_else(|_| panic!("fixture transfer rejected"));
        let quarantine = match receive(&mut worker, &transfer)
            .install_with_profiled_foreign(authority, transfer.nonce())
        {
            Err(quarantine) => quarantine,
            Ok(_) => panic!("ordinary archive work issued a profiled permit"),
        };
        assert!(matches!(
            quarantine.reason,
            ArchiveQuarantineReason::InvalidReturn("profiled foreign permit missing")
        ));
        assert!(!quarantine.retains_foreign_permit());
        assert!(!quarantine.retains_profiled_foreign_permit());
        assert!(source.matches(quarantine.owner.engine().stores()));
        assert_eq!(
            quarantine.receipt.operation,
            ArchiveOperationKind::ReclaimHot
        );
        worker.finish().unwrap();
        drop(quarantine);
        drop(scratch);
    }

    #[test]
    fn ordinary_archive_completion_cannot_issue_a_profiled_retirement_commit() {
        let (mut engine, scratch) = fixture(true);
        populate(&mut engine);
        let authority = authority(&engine);
        let source = engine.semantic.stores.read_owner();
        let mut worker = ArchiveWorker::start().unwrap();
        let transfer = worker
            .submit(ArchiveOwner::Engine(engine), authority, work(MAX_IO_BYTES))
            .unwrap_or_else(|_| panic!("fixture transfer rejected"));
        let quarantine = match receive(&mut worker, &transfer)
            .install_with_profiled_retirement(authority, transfer.nonce())
        {
            Err(quarantine) => quarantine,
            Ok(_) => panic!("ordinary archive work issued a profiled retirement commit"),
        };
        assert!(matches!(
            quarantine.reason,
            ArchiveQuarantineReason::InvalidReturn("profiled retirement commit missing")
        ));
        assert!(!quarantine.retains_profiled_retirement());
        assert!(quarantine.profiled_retirement().is_none());
        assert!(source.matches(quarantine.owner.engine().stores()));
        assert_eq!(
            quarantine.receipt.operation,
            ArchiveOperationKind::ReclaimHot
        );
        worker.finish().unwrap();
        drop(quarantine);
        drop(scratch);
    }

    #[test]
    fn retirement_install_preserves_the_original_archive_failure_metadata() {
        let (mut engine, scratch) = fixture(true);
        populate(&mut engine);
        let authority = authority(&engine);
        let source = engine.semantic.stores.read_owner();
        let before = engine.semantic.stores.hot_stats();
        let mut worker = ArchiveWorker::start().unwrap();
        let transfer = worker
            .submit(ArchiveOwner::Engine(engine), authority, work(1))
            .unwrap_or_else(|_| panic!("registered small allowance rejected"));
        let quarantine = match receive(&mut worker, &transfer)
            .install_with_profiled_retirement(authority, transfer.nonce())
        {
            Err(quarantine) => quarantine,
            Ok(_) => panic!("failed archive work issued a profiled retirement commit"),
        };
        assert!(matches!(
            quarantine.reason,
            ArchiveQuarantineReason::Operation(_)
        ));
        assert!(!quarantine.retains_profiled_retirement());
        assert!(source.matches(quarantine.owner.engine().stores()));
        assert_eq!(quarantine.owner.engine().stores().hot_stats(), before);
        worker.finish().unwrap();
        drop(quarantine);
        drop(scratch);
    }

    #[test]
    fn profiled_install_preserves_the_original_archive_failure_metadata() {
        let (mut engine, scratch) = fixture(true);
        populate(&mut engine);
        let authority = authority(&engine);
        let source = engine.semantic.stores.read_owner();
        let before = engine.semantic.stores.hot_stats();
        let mut worker = ArchiveWorker::start().unwrap();
        let transfer = worker
            .submit(ArchiveOwner::Engine(engine), authority, work(1))
            .unwrap_or_else(|_| panic!("registered small allowance rejected"));
        let quarantine = match receive(&mut worker, &transfer)
            .install_with_profiled_foreign(authority, transfer.nonce())
        {
            Err(quarantine) => quarantine,
            Ok(_) => panic!("failed archive work issued a profiled permit"),
        };
        assert!(matches!(
            quarantine.reason,
            ArchiveQuarantineReason::Operation(_)
        ));
        assert!(source.matches(quarantine.owner.engine().stores()));
        assert_eq!(quarantine.owner.engine().stores().hot_stats(), before);
        assert!(!quarantine.retains_profiled_foreign_permit());
        worker.finish().unwrap();
        drop(quarantine);
        drop(scratch);
    }

    #[test]
    fn wrong_nonce_is_quarantined_with_actual_owner() {
        let (mut engine, scratch) = fixture(true);
        populate(&mut engine);
        let authority = authority(&engine);
        let source = engine.semantic.stores.read_owner();
        let mut worker = ArchiveWorker::start().unwrap();
        let transfer = worker
            .submit(ArchiveOwner::Engine(engine), authority, work(MAX_IO_BYTES))
            .unwrap_or_else(|_| panic!("fixture transfer rejected"));
        let quarantine =
            match receive(&mut worker, &transfer).install(authority, transfer.nonce() + 1) {
                Err(value) => value,
                Ok(_) => panic!("foreign nonce accepted"),
            };
        assert!(matches!(
            quarantine.reason,
            ArchiveQuarantineReason::InvalidReturn("transfer authority")
        ));
        let owner = quarantine
            .into_owner_for_shutdown()
            .unwrap_or_else(|_| panic!("ordinary archive quarantine retained retirement payload"));
        assert!(source.matches(owner.engine().stores()));
        worker.finish().unwrap();
        drop(owner);
        drop(scratch);
    }

    #[test]
    fn canceled_after_completion_cannot_reopen_admission() {
        let (mut engine, scratch) = fixture(true);
        populate(&mut engine);
        let authority = authority(&engine);
        let mut worker = ArchiveWorker::start().unwrap();
        let transfer = worker
            .submit(ArchiveOwner::Engine(engine), authority, work(MAX_IO_BYTES))
            .unwrap_or_else(|_| panic!("fixture transfer rejected"));
        let completion = receive(&mut worker, &transfer);
        transfer.cancel();
        let quarantine = match completion.install(authority, transfer.nonce()) {
            Err(value) => value,
            Ok(_) => panic!("canceled ownership accepted"),
        };
        assert!(matches!(
            quarantine.reason,
            ArchiveQuarantineReason::Canceled
        ));
        assert!(quarantine.receipt.io_bytes.is_some_and(|bytes| bytes > 0));
        worker.finish().unwrap();
        drop(quarantine);
        drop(scratch);
    }

    #[test]
    fn expired_submission_does_not_consume_owner_or_start_work() {
        let (engine, scratch) = fixture(true);
        let authority = authority(&engine);
        let source = engine.semantic.stores.read_owner();
        let mut worker = ArchiveWorker::start().unwrap();
        let submitted_deadline = Instant::now();
        let rejected = match worker.submit(
            ArchiveOwner::Engine(engine),
            authority,
            ArchiveWork {
                deadline: submitted_deadline,
                ..work(MAX_IO_BYTES)
            },
        ) {
            Err(value) => value,
            Ok(_) => panic!("expired work submitted"),
        };
        assert!(matches!(
            rejected.error,
            PalsError::Store(StoreError::ArchiveDeadline)
        ));
        assert!(source.matches(rejected.owner.engine().stores()));
        assert_eq!(rejected.work.deadline, submitted_deadline);
        assert_eq!(rejected.work.max_io_bytes, MAX_IO_BYTES);
        assert!(matches!(
            rejected.work.operation,
            ArchiveOperation::ReclaimHot { .. }
        ));
        assert!(!worker.is_busy());
        worker.finish().unwrap();
        drop(rejected);
        drop(scratch);
    }

    #[test]
    fn unrelated_worker_cannot_receive_transfer_and_busy_finish_is_nonblocking() {
        let (mut engine, scratch) = fixture(true);
        populate(&mut engine);
        let authority = authority(&engine);
        let mut worker = ArchiveWorker::start().unwrap();
        let mut other = ArchiveWorker::<LegalOrderRoleMock>::start().unwrap();
        let transfer = worker
            .submit(ArchiveOwner::Engine(engine), authority, work(MAX_IO_BYTES))
            .unwrap_or_else(|_| panic!("fixture transfer rejected"));
        assert!(matches!(
            other.try_receive(&transfer),
            Err(ArchiveWorkerError::StaleTransfer)
        ));
        assert_eq!(worker.finish(), Err(ArchiveWorkerError::Busy));
        let completion = receive(&mut worker, &transfer);
        let (owner, _) = completion
            .install(authority, transfer.nonce())
            .unwrap_or_else(|_| panic!("fixture return rejected"));
        assert!(matches!(
            worker.try_receive(&transfer),
            Err(ArchiveWorkerError::StaleTransfer)
        ));
        worker.finish().unwrap();
        other.finish().unwrap();
        drop(owner);
        drop(scratch);
    }

    #[test]
    fn queued_notification_drop_preserves_actual_owner_in_one_owner_slot() {
        let (mut engine, scratch) = fixture(true);
        populate(&mut engine);
        let authority = authority(&engine);
        let source = engine.semantic.stores.read_owner();
        let mut worker = ArchiveWorker::start().unwrap();
        let retained = Arc::clone(&worker.orphan);
        let join = worker.join.take().unwrap();
        let transfer = worker
            .submit(ArchiveOwner::Engine(engine), authority, work(MAX_IO_BYTES))
            .unwrap_or_else(|_| panic!("fixture transfer rejected"));
        let deadline = Instant::now() + Duration::from_secs(5);
        while retained.lock().unwrap().is_none() {
            assert!(Instant::now() < deadline);
            thread::sleep(Duration::from_millis(1));
        }
        // The completion notification can already be queued; it owns no Store.
        drop(worker);
        join.join().unwrap();
        let completed = retained
            .lock()
            .unwrap()
            .take()
            .expect("actual owner retained");
        let WorkerCompletion::Maintenance(completed) = completed else {
            panic!("original maintenance completion kind");
        };
        assert!(source.matches(completed.owner.engine().stores()));
        let quarantine = match completed.install(authority, transfer.nonce()) {
            Err(value) => value,
            Ok(_) => panic!("abandoned transfer accepted"),
        };
        assert!(matches!(
            quarantine.reason,
            ArchiveQuarantineReason::Canceled
        ));
        drop(quarantine);
        drop(scratch);
    }

    #[test]
    fn panic_guard_keeps_borrowed_owner_and_hot_facts() {
        let (engine, scratch) = fixture(true);
        let mut owner = ArchiveOwner::Engine(engine);
        let source = owner.engine().stores().read_owner();
        let before = owner.engine().stores().hot_stats();
        let result = catch_archive_work(|| {
            let _exact_borrow = owner.engine_mut();
            panic!("bounded archive failure fixture");
        });
        assert!(matches!(result, Err(ArchiveQuarantineReason::Panic)));
        assert!(source.matches(owner.engine().stores()));
        assert_eq!(owner.engine().stores().hot_stats(), before);
        drop(owner);
        drop(scratch);
    }

    #[test]
    fn existing_notification_signal_is_called_after_completion_publication() {
        let (mut engine, scratch) = fixture(true);
        populate(&mut engine);
        let authority = authority(&engine);
        let notified = Arc::new(AtomicU64::new(0));
        let producer = Arc::clone(&notified);
        let mut worker = ArchiveWorker::start_with_notifier(move || {
            producer.fetch_add(1, Ordering::Release);
        })
        .unwrap();
        let transfer = worker
            .submit(ArchiveOwner::Engine(engine), authority, work(MAX_IO_BYTES))
            .unwrap_or_else(|_| panic!("fixture transfer rejected"));
        let (owner, _) = receive(&mut worker, &transfer)
            .install(authority, transfer.nonce())
            .unwrap_or_else(|_| panic!("fixture return rejected"));
        worker.finish().unwrap();
        assert_eq!(notified.load(Ordering::Acquire), 2);
        drop(owner);
        drop(scratch);
    }

    #[test]
    fn root_maintenance_does_not_retire_worker_owned_paused_task() {
        use crate::pals::store::{TaskAdmission, TaskConsumer, TaskKey, TaskQuestion, TaskStatus};
        let (mut engine, scratch) = fixture(true);
        let root = engine.semantic.stores.root().unwrap();
        let situation = engine.semantic.stores.situations.get(root).unwrap();
        let key = TaskKey {
            state: situation.state,
            line: None,
            question: TaskQuestion::AnalyzePosition,
            root_moves: Vec::new(),
            model: 1,
            epoch: 0,
            value_identity: engine.researcher.cpu_registered_value.clone(),
            checker_identity: Some(engine.researcher.checker_registered_identity.clone()),
            cpu_condition: Some(engine.researcher.cpu_registered_condition.clone()),
            profile: 1,
            condition: 1,
            input_revision: 0,
            requested_depth: 1,
            node_budget: 32,
        };
        let consumer = TaskConsumer {
            id: 1,
            situation: root,
            revision: situation.revision,
            generation: engine.semantic.stores.generation(),
            deadline_tick: 100_000,
        };
        let TaskAdmission::Start(execution) = engine
            .semantic
            .stores
            .request_task(key, consumer, 0)
            .unwrap()
        else {
            panic!("fixture admission")
        };
        engine
            .semantic
            .stores
            .pause_task(execution, 17, None)
            .unwrap();
        let authority = authority(&engine);
        let mut worker = ArchiveWorker::start().unwrap();
        let rejected = match worker.submit(
            ArchiveOwner::Engine(engine),
            authority,
            ArchiveWork {
                operation: ArchiveOperation::PrepareRoot {
                    position: Position::startpos(),
                },
                ..work(MAX_IO_BYTES)
            },
        ) {
            Err(value) => value,
            Ok(_) => panic!("live paused task crossed compaction boundary"),
        };
        assert!(matches!(
            rejected.error,
            PalsError::Store(StoreError::InvalidConditions(
                "root maintenance requires drained CPU tasks"
            ))
        ));
        assert!(matches!(
            rejected
                .owner
                .engine()
                .stores()
                .tasks
                .get(execution)
                .unwrap()
                .status,
            TaskStatus::Paused {
                checkpoint: 17,
                evidence: None
            }
        ));
        worker.finish().unwrap();
        drop(rejected);
        drop(scratch);
    }

    #[test]
    fn shutdown_close_and_join_poll_return_only_real_join_receipt() {
        let mut worker = ArchiveWorker::<LegalOrderRoleMock>::start().unwrap();
        assert_eq!(worker.try_join().unwrap(), None);
        worker.request_shutdown();
        let deadline = Instant::now() + Duration::from_secs(5);
        let receipt = loop {
            if let Some(receipt) = worker.try_join().unwrap() {
                break receipt;
            }
            assert!(Instant::now() < deadline);
            thread::sleep(Duration::from_millis(1));
        };
        assert!(receipt.admission_closed && receipt.worker_joined && receipt.owners_returned);
        assert_eq!(worker.try_join().unwrap(), Some(receipt));
        worker.request_shutdown();
        assert_eq!(worker.try_join().unwrap(), Some(receipt));
    }

    #[test]
    fn shutdown_does_not_join_or_discard_unreturned_owner() {
        let (mut engine, scratch) = fixture(true);
        populate(&mut engine);
        let authority = authority(&engine);
        let source = engine.semantic.stores.read_owner();
        let mut worker = ArchiveWorker::start().unwrap();
        let transfer = worker
            .submit(ArchiveOwner::Engine(engine), authority, work(MAX_IO_BYTES))
            .unwrap_or_else(|_| panic!("fixture transfer rejected"));
        worker.request_shutdown();
        assert_eq!(worker.try_join().unwrap(), None);
        let completion = receive(&mut worker, &transfer);
        let quarantine = match completion.install(authority, transfer.nonce()) {
            Err(value) => value,
            Ok(_) => panic!("shutdown canceled transfer admitted"),
        };
        let owner = quarantine
            .into_owner_for_shutdown()
            .unwrap_or_else(|_| panic!("ordinary archive quarantine retained retirement payload"));
        assert!(source.matches(owner.engine().stores()));
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            if worker.try_join().unwrap().is_some() {
                break;
            }
            assert!(Instant::now() < deadline);
            thread::sleep(Duration::from_millis(1));
        }
        drop(owner);
        drop(scratch);
    }
}
