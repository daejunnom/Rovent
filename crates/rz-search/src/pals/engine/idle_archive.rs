//! Exact archive maintenance while the original semantic owner has no root work.
//! This custody is not a physical fence. The concrete UCI owner must retain its
//! actual native/actor acknowledgements before lending either idle owner here.
use super::adaptive_root::{IdleArchiveGameSeal, UnusedArchiveSetupSeal};
use super::archive_lifecycle::{
    IdleArchiveCheckpoint, SEARCH_ARCHIVE_IO_BYTES, allocation_controls,
    observe_selected_archive_usage,
};
use super::preparing_root::ResearcherPreparingOwner;
use super::*;
use crate::pals::store::{
    ArchiveConfig, ArchiveIoBudget, ArchiveOwnerSnapshot, ArchiveRootCapability,
    ArchiveRuntimeLimits, ArchiveStats, StorePins,
};
use rz_contracts::pals::SearchAuthority;
use std::sync::Weak;

enum IdleOwner<M: RoleModel> {
    Researcher(Box<ResearcherPreparingOwner<M>>),
    Pc(Box<PcSemanticOwner<M>>),
    Engine(Box<PalsEngine<M>>),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum IdleArchiveTransitionKind {
    NewGame(SearchAuthority),
    Finish,
}
struct PendingIdleArchive {
    kind: IdleArchiveTransitionKind,
    seal: Option<IdleArchiveGameSeal>,
    store: crate::pals::store::StoreReadOwner,
    generation: u64,
    revision: u64,
    deadline: Instant,
    remaining_io: u64,
    preflight_complete: bool,
    engine_reset_attempted: bool,
    engine_reset_completed: bool,
    engine_reset_error: Option<PalsError>,
    checkpoint: Option<Arc<IdleArchiveCheckpoint>>,
    replacement: Option<Box<PalsStores>>,
    replacement_inherited: bool,
    replacement_observed: bool,
    completed: bool,
    completion_unknown: bool,
    #[cfg(test)]
    cancel_after_close: bool,
}

/// Read-only bounded transition facts. These grant no root, archive or physical
/// authority; a failed/closed transition remains inside its original wrapper.
#[derive(Clone, Debug)]
pub struct IdleArchiveTransitionSnapshot {
    pub next_authority: Option<SearchAuthority>,
    pub original_deadline: Instant,
    pub remaining_io_bytes: u64,
    pub checkpoint: Option<Arc<IdleArchiveCheckpoint>>,
    pub original_closed: Option<ArchiveOwnerSnapshot>,
    pub replacement: Option<ArchiveOwnerSnapshot>,
    pub completed: bool,
    pub completion_unknown: bool,
}

/// Attribution to the caller's checked registration, never root/file authority.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct UnusedIdleArchiveSetupIdentity {
    pub profile: rz_contracts::Digest,
    pub input_lock: rz_contracts::Digest,
    pub registration: rz_contracts::Digest,
}
pub struct UnusedIdleArchiveSetupControl {
    pub deadline: Instant,
    pub cancellation: Arc<AtomicBool>,
    pub max_io_bytes: u64,
}
/// Move-only ownership of the original held capability and finite first-go
/// control. Construction performs no archive I/O and issues no root authority.
pub struct UnusedIdleArchiveSetup {
    pub(super) config: ArchiveConfig,
    pub(super) capability: ArchiveRootCapability,
    pub(super) limits: ArchiveRuntimeLimits,
    pub(super) identity: UnusedIdleArchiveSetupIdentity,
    pub(super) control: UnusedIdleArchiveSetupControl,
}
impl UnusedIdleArchiveSetup {
    pub fn new(
        config: ArchiveConfig,
        capability: ArchiveRootCapability,
        limits: ArchiveRuntimeLimits,
        identity: UnusedIdleArchiveSetupIdentity,
        control: UnusedIdleArchiveSetupControl,
    ) -> Self {
        Self {
            config,
            capability,
            limits,
            identity,
            control,
        }
    }
    pub fn identity(&self) -> UnusedIdleArchiveSetupIdentity {
        self.identity
    }
    pub fn runtime_limits(&self) -> ArchiveRuntimeLimits {
        self.limits
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum IdleArchiveOwnerKind {
    Pc,
    Researcher,
    EnginePcr,
}
#[derive(Clone, Debug)]
pub struct UnusedIdleArchiveSetupSnapshot {
    pub scope: SearchAuthority,
    pub identity: UnusedIdleArchiveSetupIdentity,
    pub original_deadline: Instant,
    pub remaining_io_bytes: Option<u64>,
    pub configuration_complete: bool,
    pub completion_unknown: bool,
}

/// One original heap owner moves through the existing one-slot archive worker.
/// No checker, TT, model execution, root authority or physical receipt is made.
pub struct IdleSemanticArchiveOwner<M: RoleModel> {
    owner: IdleOwner<M>,
    identity: Arc<()>,
    pending: Option<Box<PendingIdleArchive>>,
    last_checkpoint: Option<Arc<IdleArchiveCheckpoint>>,
    last_transition_remaining: Option<u64>,
    setup: Option<UnusedIdleArchiveSetupSnapshot>,
    #[cfg(test)]
    panic_setup_after_enable: bool,
    #[cfg(test)]
    cancel_after_close: bool,
}

impl<M: RoleModel> ResearcherPreparingOwner<M> {
    #[allow(clippy::result_large_err)]
    pub fn into_idle_archive(
        self: Box<Self>,
    ) -> Result<IdleSemanticArchiveOwner<M>, (Box<Self>, PalsError)> {
        if let Err(error) = self.semantic.ensure_adaptive_logical_quiescence() {
            return Err((self, error));
        }
        Ok(IdleSemanticArchiveOwner {
            owner: IdleOwner::Researcher(self),
            identity: Arc::new(()),
            pending: None,
            last_checkpoint: None,
            last_transition_remaining: None,
            setup: None,
            #[cfg(test)]
            panic_setup_after_enable: false,
            #[cfg(test)]
            cancel_after_close: false,
        })
    }
}
impl<M: RoleModel> PcSemanticOwner<M> {
    #[allow(clippy::result_large_err)]
    pub fn into_idle_archive(
        self: Box<Self>,
    ) -> Result<IdleSemanticArchiveOwner<M>, (Box<Self>, PalsError)> {
        if let Err(error) = self.semantic.ensure_adaptive_logical_quiescence() {
            return Err((self, error));
        }
        Ok(IdleSemanticArchiveOwner {
            owner: IdleOwner::Pc(self),
            identity: Arc::new(()),
            pending: None,
            last_checkpoint: None,
            last_transition_remaining: None,
            setup: None,
            #[cfg(test)]
            panic_setup_after_enable: false,
            #[cfg(test)]
            cancel_after_close: false,
        })
    }
}

impl<M: RoleModel> PalsEngine<M> {
    /// Retain this complete own-R engine, including its original registered
    /// checker mirror. This does not turn it into a deferred or PC owner and
    /// grants no root, CPU reset, archive or physical-completion authority.
    #[allow(clippy::result_large_err)]
    pub fn into_idle_archive(
        self: Box<Self>,
    ) -> Result<IdleSemanticArchiveOwner<M>, (Box<Self>, PalsError)> {
        if let Err(error) = self.validate_engine_idle_archive_owner() {
            return Err((self, error));
        }
        Ok(IdleSemanticArchiveOwner {
            owner: IdleOwner::Engine(self),
            identity: Arc::new(()),
            pending: None,
            last_checkpoint: None,
            last_transition_remaining: None,
            setup: None,
            #[cfg(test)]
            panic_setup_after_enable: false,
            #[cfg(test)]
            cancel_after_close: false,
        })
    }
}

impl<M: RoleModel> IdleSemanticArchiveOwner<M> {
    // Rejected kind/setup transitions return the complete original linear
    // owner. Keep this error move-only without a new allocation or lost custody.
    #[allow(clippy::result_large_err)]
    pub fn into_researcher(self) -> Result<Box<ResearcherPreparingOwner<M>>, Self> {
        if self.pending.is_some() || self.setup_retained() {
            return Err(self);
        }
        match self.owner {
            IdleOwner::Researcher(owner) => Ok(owner),
            IdleOwner::Pc(_) | IdleOwner::Engine(_) => Err(self),
        }
    }
    #[allow(clippy::result_large_err)]
    pub fn into_pc(self) -> Result<Box<PcSemanticOwner<M>>, Self> {
        if self.pending.is_some() || self.setup_retained() {
            return Err(self);
        }
        match self.owner {
            IdleOwner::Pc(owner) => Ok(owner),
            IdleOwner::Researcher(_) | IdleOwner::Engine(_) => Err(self),
        }
    }
    #[allow(clippy::result_large_err)]
    pub fn into_engine(self) -> Result<Box<PalsEngine<M>>, Self> {
        if self.pending.is_some() || self.setup_retained() {
            return Err(self);
        }
        match self.owner {
            IdleOwner::Engine(owner) => Ok(owner),
            IdleOwner::Pc(_) | IdleOwner::Researcher(_) => Err(self),
        }
    }
    pub fn stores(&self) -> &PalsStores {
        &self.semantic().stores
    }
    /// Read-only original registration attribution, not a root or maintenance
    /// permit. The worker still seals this same owner and current generation.
    pub fn last_registered_scope(&self) -> Option<SearchAuthority> {
        self.semantic().last_adaptive_authority()
    }
    pub fn setup_snapshot(&self) -> Option<&UnusedIdleArchiveSetupSnapshot> {
        self.setup.as_ref()
    }
    pub fn owner_kind(&self) -> IdleArchiveOwnerKind {
        match &self.owner {
            IdleOwner::Pc(_) => IdleArchiveOwnerKind::Pc,
            IdleOwner::Researcher(_) => IdleArchiveOwnerKind::Researcher,
            IdleOwner::Engine(_) => IdleArchiveOwnerKind::EnginePcr,
        }
    }
    fn setup_retained(&self) -> bool {
        self.setup
            .as_ref()
            .is_some_and(|state| !state.configuration_complete || state.completion_unknown)
    }
    #[cfg(test)]
    pub(super) fn panic_setup_after_enable_for_test(&mut self) {
        self.panic_setup_after_enable = true;
    }
    pub(super) fn seal_unused_setup(
        &self,
        scope: SearchAuthority,
    ) -> Result<UnusedArchiveSetupSeal, PalsError> {
        self.validate_idle()?;
        let semantic = self.semantic();
        semantic.stores.validate_unused_archive_setup()?;
        if self.setup.is_some()
            || !semantic.nodes.is_empty()
            || !semantic.records.is_empty()
            || semantic.revision != 0
            || semantic.consumer_id != 0
            || !semantic.repair_queue.is_empty()
            || !semantic.repair_questions.is_empty()
            || !semantic.repair_counts.is_empty()
            || semantic.archive_state.config.is_some()
            || semantic.archive_state.root_capability.is_some()
            || semantic.archive_state.previous_closed_owner.is_some()
            || semantic.archive_state.closed_current_owner.is_some()
            || semantic.archive_state.deadline.is_some()
            || semantic.archive_state.remaining_io != 0
            || semantic.archive_state.active_search_pins != StorePins::default()
            || semantic.last_engine_archive().is_some()
            || self.last_checkpoint.is_some()
            || self.last_transition_remaining.is_some()
            || semantic.review_index.as_ref().is_none_or(|index| {
                index.active_bundle_count() != 0 || index.indexed_record_count() != 0
            })
        {
            return Err(
                StoreError::InvalidConditions("archive setup needs unused semantic owner").into(),
            );
        }
        semantic.seal_unused_archive_setup(scope)
    }
    pub(super) fn execute_unused_setup(
        &mut self,
        scope: SearchAuthority,
        seal: &UnusedArchiveSetupSeal,
        setup: &UnusedIdleArchiveSetup,
    ) -> Result<(), PalsError> {
        self.seal_unused_setup(scope)?;
        seal.validate(self.semantic())?;
        self.setup = Some(UnusedIdleArchiveSetupSnapshot {
            scope,
            identity: setup.identity,
            original_deadline: setup.control.deadline,
            remaining_io_bytes: Some(setup.control.max_io_bytes),
            configuration_complete: false,
            completion_unknown: false,
        });
        allocation_controls(Some((setup.control.deadline, &setup.control.cancellation)))?;
        let semantic = self.semantic_mut();
        semantic
            .stores
            .enable_archive_with_capability(setup.config.clone(), setup.capability.clone())?;
        semantic.archive_state.config = Some(setup.config.clone());
        semantic.archive_state.root_capability = Some(setup.capability.clone());
        #[cfg(test)]
        if std::mem::take(&mut self.panic_setup_after_enable) {
            panic!("authored interruption after original archive enable");
        }
        let semantic = self.semantic_mut();
        semantic.stores.set_archive_runtime_limits(setup.limits)?;
        let config = semantic
            .archive_state
            .config
            .as_mut()
            .expect("original setup config");
        config.game_bytes = setup.limits.game_bytes_max;
        config.global_bytes = setup.limits.global_bytes_max;
        config.record_bytes = setup.limits.record_payload_bytes_max;
        config.max_load_pins = setup.limits.max_load_pins as usize;
        allocation_controls(Some((setup.control.deadline, &setup.control.cancellation)))?;
        let maximum_deadline = Instant::now()
            .checked_add(Duration::from_millis(setup.limits.load_deadline_max_ms))
            .ok_or(StoreError::InvalidConditions("archive setup load duration"))?;
        let mut budget = ArchiveIoBudget {
            deadline: setup.control.deadline.min(maximum_deadline),
            max_bytes: setup.control.max_io_bytes.min(setup.limits.load_bytes_max),
        };
        let before = budget.max_bytes;
        let result = self
            .semantic_mut()
            .stores
            .measure_archive_usage_with_budget(&mut budget);
        let remaining = setup
            .control
            .max_io_bytes
            .checked_sub(before.saturating_sub(budget.max_bytes))
            .ok_or(StoreError::ArchiveByteBudget)?;
        self.setup
            .as_mut()
            .expect("original setup state")
            .remaining_io_bytes = Some(remaining);
        result?;
        allocation_controls(Some((setup.control.deadline, &setup.control.cancellation)))?;
        seal.validate(self.semantic())?;
        let actual = self
            .archive_owner_snapshot()?
            .ok_or(StoreError::ArchiveDisabled)?;
        if !actual.complete
            || actual.admission_closed
            || actual.runtime_limits != Some(setup.limits)
        {
            return Err(
                StoreError::InvalidConditions("archive setup observation incomplete").into(),
            );
        }
        self.setup
            .as_mut()
            .expect("original setup state")
            .configuration_complete = true;
        Ok(())
    }
    pub(super) fn retain_unused_setup_failure(&mut self) {
        if let Some(state) = &mut self.setup {
            state.configuration_complete = false;
        }
    }
    pub(super) fn quarantine_unused_setup_panic(
        &mut self,
        scope: SearchAuthority,
        setup: &UnusedIdleArchiveSetup,
    ) {
        let state = self.setup.get_or_insert(UnusedIdleArchiveSetupSnapshot {
            scope,
            identity: setup.identity,
            original_deadline: setup.control.deadline,
            remaining_io_bytes: None,
            configuration_complete: false,
            completion_unknown: true,
        });
        state.configuration_complete = false;
        state.completion_unknown = true;
        state.remaining_io_bytes = None;
    }
    pub fn archive_enabled(&self) -> bool {
        self.stores().archive_enabled()
    }
    pub fn archive_owner_snapshot(&self) -> Result<Option<ArchiveOwnerSnapshot>, PalsError> {
        Ok(self
            .stores()
            .archive_owner_snapshot()?
            .or(self.semantic().archive_state.closed_current_owner))
    }
    /// The concrete caller already owns its physical fence. This closes only
    /// the original Store archive and preserves its closed observation.
    pub fn close_archive_owner(&mut self) -> Result<Option<ArchiveOwnerSnapshot>, PalsError> {
        self.validate_idle()?;
        if self.stores().hot_stats() != ArchiveStats::default() {
            return Err(StoreError::InvalidConditions(
                "hot idle game requires its final checkpoint before close",
            )
            .into());
        }
        let semantic = self.semantic_mut();
        let closed = semantic.stores.close_archive_owner()?;
        semantic.archive_state.closed_current_owner = closed;
        Ok(closed)
    }
    /// Same read-only maintenance namespace as the existing engine adapter.
    /// The next root is still registered only by its original owner afterwards.
    pub fn adaptive_archive_authority(
        &self,
        next: SearchAuthority,
    ) -> Result<SearchAuthority, PalsError> {
        self.semantic().adaptive_archive_authority(next)
    }
    /// Explicit selected storage only. Call on a controlled startup worker;
    /// this may open the original held archive root and never enables itself.
    pub fn enable_archive_with_capability(
        &mut self,
        config: ArchiveConfig,
        capability: ArchiveRootCapability,
    ) -> Result<(), PalsError> {
        self.validate_idle()?;
        let semantic = self.semantic_mut();
        semantic
            .stores
            .enable_archive_with_capability(config.clone(), capability.clone())?;
        semantic.archive_state.config = Some(config);
        semantic.archive_state.root_capability = Some(capability);
        semantic.archive_state.reopen_error = None;
        semantic.archive_state.closed_current_owner = None;
        Ok(())
    }
    pub fn set_archive_runtime_limits(
        &mut self,
        limits: ArchiveRuntimeLimits,
    ) -> Result<(), PalsError> {
        self.validate_idle()?;
        let semantic = self.semantic_mut();
        semantic.stores.set_archive_runtime_limits(limits)?;
        let actual =
            *semantic
                .stores
                .archive_runtime_limits()
                .ok_or(StoreError::InvalidConditions(
                    "idle archive selector missing",
                ))?;
        if let Some(config) = &mut semantic.archive_state.config {
            config.game_bytes = actual.game_bytes_max;
            config.global_bytes = actual.global_bytes_max;
            config.record_bytes = actual.record_payload_bytes_max;
            config.max_load_pins = actual.max_load_pins as usize;
        }
        Ok(())
    }
    pub(super) fn semantic(&self) -> &PalsSemanticState<M> {
        match &self.owner {
            IdleOwner::Researcher(owner) => &owner.semantic,
            IdleOwner::Pc(owner) => &owner.semantic,
            IdleOwner::Engine(owner) => &owner.semantic,
        }
    }
    fn semantic_mut(&mut self) -> &mut PalsSemanticState<M> {
        match &mut self.owner {
            IdleOwner::Researcher(owner) => &mut owner.semantic,
            IdleOwner::Pc(owner) => &mut owner.semantic,
            IdleOwner::Engine(owner) => &mut owner.semantic,
        }
    }
    pub(super) fn identity(&self) -> Weak<()> {
        Arc::downgrade(&self.identity)
    }
    fn config(&self) -> PalsSemanticConfig {
        match &self.owner {
            IdleOwner::Researcher(owner) => owner.profile().sequence_config(),
            IdleOwner::Pc(owner) => owner.profile.sequence_config(),
            IdleOwner::Engine(owner) => owner.semantic_sequence_config(),
        }
    }
    fn cpu_value(&self) -> Option<&CpuValueIdentity> {
        match &self.owner {
            IdleOwner::Researcher(owner) => Some(owner.source().model().identity()),
            IdleOwner::Pc(_) => None,
            IdleOwner::Engine(owner) => owner.researcher.cpu_registered_value.as_ref(),
        }
    }
    fn legacy_model_wdl(&self) -> bool {
        match &self.owner {
            IdleOwner::Engine(owner) => {
                !owner.followup_lane && !owner.post_repair_recheck.uses_frozen_model_wdl()
            }
            IdleOwner::Researcher(_) | IdleOwner::Pc(_) => false,
        }
    }
    pub(super) fn validate_idle(&self) -> Result<(), PalsError> {
        if self.pending.is_some() || self.setup_retained() {
            return Err(
                StoreError::InvalidConditions("idle archive transition is retained").into(),
            );
        }
        if let Some(error) = &self.semantic().archive_state.reopen_error {
            return Err(PalsError::Store(error.clone()));
        }
        self.semantic().ensure_archive_admission_open()?;
        match &self.owner {
            IdleOwner::Engine(owner) => owner.validate_engine_idle_archive_owner(),
            IdleOwner::Researcher(_) | IdleOwner::Pc(_) => {
                self.semantic().ensure_adaptive_logical_quiescence()
            }
        }
    }
    pub(super) fn projection_pins(&self) -> Result<StorePins, PalsError> {
        match &self.semantic().review_index {
            Some(index) => Ok(index.archive_projection_pins(self.stores())?),
            None => Err(PalsError::InvalidConfig),
        }
    }
    pub(super) fn prepare_root(
        &mut self,
        position: &Position,
        deadline: Instant,
        cancel: &AtomicBool,
    ) -> Result<(), PalsError> {
        self.validate_idle()?;
        allocation_controls(Some((deadline, cancel)))?;
        if !self.archive_enabled() {
            return Err(StoreError::ArchiveDisabled.into());
        }
        let config = self.config();
        let pins = self.projection_pins()?;
        let semantic = self.semantic_mut();
        semantic.archive_state.deadline = Some(deadline);
        semantic.archive_state.remaining_io = SEARCH_ARCHIVE_IO_BYTES;
        semantic.archive_state.retried_allocation = false;
        semantic.archive_state.active_search_pins = StorePins::default();
        if semantic.stores.archive_runtime_limits().is_some() {
            let mut budget = semantic.engine_archive_budget()?;
            let before = budget.max_bytes;
            let result = semantic
                .stores
                .measure_archive_usage_with_budget(&mut budget);
            semantic.charge_engine_archive_io(before.saturating_sub(budget.max_bytes))?;
            result?;
        }
        semantic
            .stores
            .set_archive_pins(semantic.resident_archive_pins_with(&pins)?)?;
        if semantic.engine_archive_pressure(&config, position) {
            // Idle validation rejects every resume token, active task and view.
            semantic.reclaim_archive_for_root(
                &config,
                &pins,
                &|_| false,
                position,
                deadline,
                cancel,
            )?;
        }
        allocation_controls(Some((deadline, cancel)))?;
        Ok(())
    }
    pub(super) fn remaining_io(&self) -> u64 {
        self.semantic().archive_remaining_io()
    }

    pub fn last_checkpoint(&self) -> Option<Arc<IdleArchiveCheckpoint>> {
        self.pending
            .as_ref()
            .and_then(|pending| pending.checkpoint.clone())
            .or_else(|| self.last_checkpoint.clone())
    }
    pub fn transition_snapshot(&self) -> Result<Option<IdleArchiveTransitionSnapshot>, PalsError> {
        self.pending
            .as_ref()
            .map(|pending| {
                Ok(IdleArchiveTransitionSnapshot {
                    next_authority: match pending.kind {
                        IdleArchiveTransitionKind::NewGame(next) => Some(next),
                        IdleArchiveTransitionKind::Finish => None,
                    },
                    original_deadline: pending.deadline,
                    remaining_io_bytes: pending.remaining_io,
                    checkpoint: pending.checkpoint.clone(),
                    original_closed: self.semantic().archive_state.closed_current_owner,
                    replacement: pending
                        .replacement
                        .as_ref()
                        .map(|store| store.archive_owner_snapshot())
                        .transpose()?
                        .flatten(),
                    completed: pending.completed,
                    completion_unknown: pending.completion_unknown,
                })
            })
            .transpose()
    }
    pub(super) fn transition_remaining_io(&self) -> u64 {
        self.pending
            .as_ref()
            .map(|pending| pending.remaining_io)
            .or(self.last_transition_remaining)
            .unwrap_or(SEARCH_ARCHIVE_IO_BYTES)
    }
    pub(super) fn pending_transition_io(&self) -> Option<u64> {
        self.pending.as_ref().map(|pending| pending.remaining_io)
    }
    pub(super) fn quarantine_transition_unknown(&mut self) {
        if let Some(pending) = &mut self.pending {
            pending.completion_unknown = true;
            // A panic can interrupt a real commit before its debit returns.
            // Keep the original owners and reserve, never refund that allowance.
            pending.remaining_io = 0;
        }
    }
    #[cfg(test)]
    pub(super) fn cancel_after_close_for_test(&mut self) {
        self.cancel_after_close = true;
    }
    pub(super) fn validate_transition_request(
        &self,
        kind: IdleArchiveTransitionKind,
        deadline: Instant,
    ) -> Result<(), PalsError> {
        self.semantic().ensure_adaptive_logical_quiescence()?;
        if let IdleOwner::Engine(owner) = &self.owner {
            owner.validate_engine_idle_archive_owner()?;
        }
        if let Some(pending) = &self.pending {
            if pending.kind != kind
                || deadline > pending.deadline
                || pending.completed
                || pending.completion_unknown
                || !pending.store.matches(self.stores())
                || pending.generation != self.stores().generation()
                || pending.revision != self.semantic().revision
            {
                return Err(StoreError::StaleConsumer.into());
            }
            if let Some(seal) = &pending.seal {
                seal.validate(self.semantic())?;
            }
        } else {
            self.validate_idle()?;
            if let IdleArchiveTransitionKind::NewGame(next) = kind {
                self.semantic().validate_idle_new_game_target(next)?;
            }
        }
        Ok(())
    }
    pub(super) fn advance_transition(
        &mut self,
        kind: IdleArchiveTransitionKind,
        deadline: Instant,
        cancel: &AtomicBool,
    ) -> Result<(), PalsError> {
        self.validate_transition_request(kind, deadline)?;
        allocation_controls(Some((deadline, cancel)))?;
        if self.pending.is_none() {
            let seal = match kind {
                IdleArchiveTransitionKind::NewGame(next) => {
                    Some(self.semantic().validate_idle_new_game_target(next)?)
                }
                IdleArchiveTransitionKind::Finish => None,
            };
            self.pending = Some(Box::new(PendingIdleArchive {
                kind,
                seal,
                store: self.stores().read_owner(),
                generation: self.stores().generation(),
                revision: self.semantic().revision,
                deadline,
                remaining_io: SEARCH_ARCHIVE_IO_BYTES,
                preflight_complete: false,
                engine_reset_attempted: false,
                engine_reset_completed: false,
                engine_reset_error: None,
                checkpoint: None,
                replacement: None,
                replacement_inherited: false,
                replacement_observed: false,
                completed: false,
                completion_unknown: false,
                #[cfg(test)]
                cancel_after_close: std::mem::take(&mut self.cancel_after_close),
            }));
        }
        let config = self.config();
        // Borrow separate heap-owned slots; panic/error cannot drop a local
        // replacement or make the original transition lose its remaining bound.
        let pending = self.pending.as_mut().expect("original transition slot");
        match &mut self.owner {
            IdleOwner::Engine(owner) => {
                if matches!(kind, IdleArchiveTransitionKind::NewGame(_))
                    && !pending.preflight_complete
                {
                    if let Some(error) = &pending.engine_reset_error {
                        return Err(error.clone());
                    }
                    if pending.engine_reset_attempted && !pending.engine_reset_completed {
                        return Err(StoreError::InvalidConditions(
                            "original Engine reset completion unknown",
                        )
                        .into());
                    }
                    owner
                        .semantic
                        .model
                        .preflight_new_game(deadline.min(pending.deadline), cancel)?;
                    owner.validate_engine_idle_archive_owner()?;
                    // The original caller already reset and armed its actual
                    // CPU. Mark this before consuming that mirror arm, even on
                    // canceled delivery; a retry must not consume another arm.
                    pending.engine_reset_attempted = true;
                    match owner
                        .preflight_engine_idle_archive_reset(deadline.min(pending.deadline), cancel)
                    {
                        Ok(()) => pending.engine_reset_completed = true,
                        Err(error) => {
                            pending.engine_reset_error = Some(error.clone());
                            return Err(error);
                        }
                    }
                    pending.preflight_complete = true;
                }
                advance_pending(&mut owner.semantic, &config, pending, deadline, cancel)?;
                if matches!(kind, IdleArchiveTransitionKind::NewGame(_)) {
                    owner.complete_engine_idle_archive_new_game();
                }
            }
            IdleOwner::Researcher(owner) => {
                advance_pending(&mut owner.semantic, &config, pending, deadline, cancel)?;
            }
            IdleOwner::Pc(owner) => {
                advance_pending(&mut owner.semantic, &config, pending, deadline, cancel)?;
            }
        }
        self.last_checkpoint = pending.checkpoint.clone();
        self.last_transition_remaining = Some(pending.remaining_io);
        if matches!(kind, IdleArchiveTransitionKind::NewGame(_)) {
            self.pending = None;
        }
        Ok(())
    }

    pub(super) fn restore_root_if_archived(
        &mut self,
        position: &Position,
        budget: &mut ArchiveIoBudget,
    ) -> Result<Option<usize>, PalsError> {
        self.validate_idle()?;
        let config = self.config();
        let cpu = self.cpu_value().cloned();
        let legacy_model_wdl = self.legacy_model_wdl();
        let pins = self.projection_pins()?;
        let semantic = self.semantic_mut();
        semantic
            .stores
            .set_archive_pins(semantic.resident_archive_pins_with(&pins)?)?;
        let result = semantic.load_archived_root_if_present_with_budget(
            &config,
            cpu.as_ref(),
            legacy_model_wdl,
            position,
            budget,
        );
        let restore = semantic
            .resident_archive_pins_with(&pins)
            .and_then(|pins| semantic.stores.set_archive_pins(pins).map_err(Into::into));
        match result {
            Err(error) => Err(error),
            Ok(root) => {
                restore?;
                Ok(root)
            }
        }
    }
    pub(super) fn load_root(
        &mut self,
        position: &Position,
        budget: &mut ArchiveIoBudget,
    ) -> Result<usize, PalsError> {
        self.validate_idle()?;
        let config = self.config();
        let cpu = self.cpu_value().cloned();
        let legacy_model_wdl = self.legacy_model_wdl();
        let pins = self.projection_pins()?;
        let semantic = self.semantic_mut();
        semantic
            .stores
            .set_archive_pins(semantic.resident_archive_pins_with(&pins)?)?;
        // Rovent never accepts a legacy Wdl observation as ContextWdl.
        let result = semantic.load_archived_root_with_budget(
            &config,
            cpu.as_ref(),
            legacy_model_wdl,
            position,
            budget,
        );
        let restored_pins = semantic
            .resident_archive_pins_with(&pins)
            .and_then(|restored| {
                semantic
                    .stores
                    .set_archive_pins(restored)
                    .map_err(Into::into)
            });
        match result {
            Err(error) => Err(error),
            Ok(root) => {
                restored_pins?;
                Ok(root)
            }
        }
    }
}

fn advance_pending<M: RoleModel>(
    semantic: &mut PalsSemanticState<M>,
    config: &PalsSemanticConfig,
    pending: &mut PendingIdleArchive,
    deadline: Instant,
    cancel: &AtomicBool,
) -> Result<(), PalsError> {
    let deadline = deadline.min(pending.deadline);
    allocation_controls(Some((deadline, cancel)))?;
    if !pending.preflight_complete {
        if matches!(pending.kind, IdleArchiveTransitionKind::NewGame(_)) {
            semantic.model.preflight_new_game(deadline, cancel)?;
        }
        pending.preflight_complete = true;
    }
    if pending.checkpoint.is_none() {
        let mut budget = ArchiveIoBudget {
            deadline,
            max_bytes: pending.remaining_io,
        };
        let result = semantic.checkpoint_idle_archive_game(&mut budget);
        pending.remaining_io = budget.max_bytes;
        pending.checkpoint = Some(Arc::new(result?));
    }
    allocation_controls(Some((deadline, cancel)))?;
    if matches!(pending.kind, IdleArchiveTransitionKind::NewGame(_)) {
        if pending.replacement.is_none() {
            // Put the unused Store in its retained slot before any fallible I/O.
            pending.replacement = Some(Box::new(PalsStores::new(config.store_limits())));
        }
        let next = pending
            .replacement
            .as_mut()
            .expect("original unused replacement");
        if !next.archive_enabled() {
            let archive = semantic
                .archive_state
                .config
                .as_ref()
                .ok_or(StoreError::ArchiveDisabled)?
                .clone();
            let held = semantic.archive_state.root_capability.as_ref().ok_or(
                StoreError::InvalidConditions("idle newgame needs original held archive root"),
            )?;
            next.enable_archive_with_capability(archive, held.clone())?;
        }
        if next.archive_runtime_limits().is_none() {
            if let Some(limits) = semantic.stores.archive_runtime_limits().copied() {
                next.set_archive_runtime_limits(limits)?;
            }
        }
        if !pending.replacement_inherited {
            next.inherit_archive_write_budget_from(&semantic.stores)?;
            pending.replacement_inherited = true;
        }
        if !pending.replacement_observed {
            // This existing finite metadata scan has zero payload-byte debit.
            // It retains the original deadline and selected read limits.
            observe_selected_archive_usage(next, deadline, cancel)?;
            pending.replacement_observed = true;
        }
    }
    allocation_controls(Some((deadline, cancel)))?;
    if semantic.archive_state.closed_current_owner.is_none() {
        semantic.archive_state.closed_current_owner = semantic.stores.close_archive_owner()?;
    }
    #[cfg(test)]
    if std::mem::take(&mut pending.cancel_after_close) {
        // Schedule a control change at the real irreversible close, without a
        // fake close result, sleep, replacement receipt or public authority.
        cancel.store(true, Ordering::Release);
    }
    // Closing cannot be undone. A control change here keeps this same original
    // closed Store, checkpoint and replacement in pending, without a new budget.
    allocation_controls(Some((deadline, cancel)))?;
    if matches!(pending.kind, IdleArchiveTransitionKind::NewGame(_)) {
        semantic.commit_idle_new_game_scope(
            pending.seal.as_ref().expect("original newgame seal"),
            &mut pending.replacement,
        )?;
    }
    pending.completed = true;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::{CpuConfig, CpuEngine, CpuLimits};
    use crate::cpu_checker::{
        CheckerAttempt, CheckerCapabilities, CheckerError, CheckerReport, CpuChecker,
        OwnedCheckerDescriptor, OwnedCpuChecker,
    };
    use rz_contracts::{Digest, GameGeneration, ProcessEpoch, RootGeneration};
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, AtomicUsize};

    // Authored reset-delivery fixture around a real own checker. It observes
    // calls/arm consumption; it is not an actual native-worker reset proof.
    struct ResetProbe {
        checker: OwnedCpuChecker<CpuEngine>,
        calls: Arc<AtomicUsize>,
        armed: Arc<AtomicBool>,
        fail: Arc<AtomicBool>,
        panic_on_reset: Arc<AtomicBool>,
    }
    impl CpuChecker for ResetProbe {
        fn identity(&self) -> &CheckerIdentity {
            self.checker.identity()
        }
        fn conditions(&self) -> &str {
            self.checker.conditions()
        }
        fn capabilities(&self) -> CheckerCapabilities {
            self.checker.capabilities()
        }
        fn validate_namespace(&self) -> Result<(), CheckerError> {
            self.checker.validate_namespace()
        }
        fn owned_descriptor(&self) -> Option<OwnedCheckerDescriptor> {
            self.checker.owned_descriptor()
        }
        fn last_attempt(&self) -> Option<&CheckerAttempt> {
            self.checker.last_attempt()
        }
        fn reset_attempt(&mut self) {
            self.checker.reset_attempt();
        }
        fn analyze(
            &mut self,
            _: &Position,
            _: CpuLimits,
            _: &AtomicBool,
        ) -> Result<CheckerReport, CheckerError> {
            Err(CheckerError::Unsupported(
                "no search in Engine idle fixture",
            ))
        }
        fn new_game(&mut self, deadline: Instant, cancel: &AtomicBool) -> Result<(), CheckerError> {
            if !self.armed.swap(false, Ordering::AcqRel) {
                return Err(CheckerError::Invalid("original fixture reset arm absent"));
            }
            self.calls.fetch_add(1, Ordering::AcqRel);
            if self.panic_on_reset.swap(false, Ordering::AcqRel) {
                panic!("original fixture reset panic");
            }
            if self.fail.swap(false, Ordering::AcqRel) {
                return Err(CheckerError::Invalid(
                    "original fixture reset delivery failure",
                ));
            }
            self.checker.new_game(deadline, cancel)
        }
        fn shutdown(
            &mut self,
            deadline: Instant,
        ) -> Result<crate::cpu_checker::CheckerShutdown, CheckerError> {
            self.checker.shutdown(deadline)
        }
    }
    struct EngineFixture {
        owner: Box<PalsEngine<LegalOrderRoleMock>>,
        calls: Arc<AtomicUsize>,
        armed: Arc<AtomicBool>,
        fail: Arc<AtomicBool>,
        panic_on_reset: Arc<AtomicBool>,
    }
    fn engine_scope() -> SearchAuthority {
        SearchAuthority {
            epoch: ProcessEpoch(9401),
            game: GameGeneration(1),
            root: RootGeneration(1),
            implementation: Digest([81; 32]),
        }
    }
    fn unused_game_zero_fixture(_until: Instant) -> EngineFixture {
        let calls = Arc::new(AtomicUsize::new(0));
        let armed = Arc::new(AtomicBool::new(true));
        let fail = Arc::new(AtomicBool::new(false));
        let panic_on_reset = Arc::new(AtomicBool::new(false));
        let checker = ResetProbe {
            checker: OwnedCpuChecker::new(
                CpuEngine::new(CpuConfig {
                    tt_entries: 64,
                    max_depth: 4,
                    quiescence_ply: 4,
                    ..CpuConfig::default()
                })
                .unwrap(),
            )
            .unwrap(),
            calls: Arc::clone(&calls),
            armed: Arc::clone(&armed),
            fail: Arc::clone(&fail),
            panic_on_reset: Arc::clone(&panic_on_reset),
        };
        let owner = Box::new(
            PalsEngine::new_with_boxed_checker_and_rovent_profile(
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
                LegalOrderRoleMock,
                Box::new(checker),
                ResolverPolicy::ModelWdlRestricted,
                PostRepairRecheckPolicy::Disabled,
            )
            .unwrap()
            .with_selected_review_context(
                ReviewSelectionPolicy::default(),
                ReviewIndexLimits::default(),
            )
            .unwrap(),
        );
        EngineFixture {
            owner,
            calls,
            armed,
            fail,
            panic_on_reset,
        }
    }
    fn engine_fixture(until: Instant) -> EngineFixture {
        let mut fixture = unused_game_zero_fixture(until);
        fixture
            .owner
            .try_adaptive_new_game(engine_scope(), until, &AtomicBool::new(false))
            .unwrap();
        assert_eq!(fixture.calls.swap(0, Ordering::AcqRel), 1);
        fixture
    }

    #[test]
    fn engine_unused_same_game_zero_preparation_keeps_scope_and_consumes_reset_once() {
        let until = Instant::now() + Duration::from_secs(20);
        let mut fixture = unused_game_zero_fixture(until);
        let scope = SearchAuthority {
            game: GameGeneration(0),
            ..engine_scope()
        };
        let store = fixture.owner.stores().read_owner();
        fixture
            .owner
            .prepare_unused_archive_game(scope, until, &AtomicBool::new(false))
            .unwrap();
        assert_eq!(fixture.owner.semantic.game_generation, Some(0));
        assert_eq!(
            fixture.owner.semantic.adaptive_new_game_scope(),
            Some(scope)
        );
        assert_eq!(fixture.owner.last_adaptive_authority(), None);
        assert!(store.matches(fixture.owner.stores()));
        fixture
            .owner
            .prepare_unused_archive_game(scope, until, &AtomicBool::new(false))
            .unwrap();
        assert_eq!(fixture.calls.load(Ordering::Acquire), 1);
        assert!(!fixture.armed.load(Ordering::Acquire));
        let other = SearchAuthority {
            root: RootGeneration(2),
            ..scope
        };
        assert!(
            fixture
                .owner
                .prepare_unused_archive_game(other, until, &AtomicBool::new(false))
                .is_err()
        );
        assert!(
            fixture
                .owner
                .prepare_unused_archive_game(
                    scope,
                    until + Duration::from_secs(1),
                    &AtomicBool::new(false)
                )
                .is_err()
        );
        assert_eq!(fixture.calls.load(Ordering::Acquire), 1);
        let idle = fixture
            .owner
            .into_idle_archive()
            .unwrap_or_else(|_| panic!("prepared original Engine"));
        idle.seal_unused_setup(scope).unwrap();
    }

    #[test]
    fn engine_unused_game_reset_failure_preserves_first_error_and_blocks_setup_or_replay() {
        let until = Instant::now() + Duration::from_secs(20);
        let mut fixture = unused_game_zero_fixture(until);
        let scope = SearchAuthority {
            game: GameGeneration(0),
            ..engine_scope()
        };
        let store = fixture.owner.stores().read_owner();
        fixture.fail.store(true, Ordering::Release);
        let first = fixture
            .owner
            .prepare_unused_archive_game(scope, until, &AtomicBool::new(false))
            .unwrap_err();
        fixture.armed.store(true, Ordering::Release);
        let repeated = fixture
            .owner
            .prepare_unused_archive_game(scope, until, &AtomicBool::new(false))
            .unwrap_err();
        assert_eq!(format!("{first:?}"), format!("{repeated:?}"));
        assert_eq!(fixture.calls.load(Ordering::Acquire), 1);
        assert!(fixture.armed.load(Ordering::Acquire));
        assert_eq!(fixture.owner.semantic.adaptive_new_game_scope(), None);
        assert!(store.matches(fixture.owner.stores()));
        let (owner, _) = fixture
            .owner
            .into_idle_archive()
            .err()
            .expect("failed preparation retained");
        assert!(store.matches(owner.stores()));
        let pending = owner
            .semantic
            .archive_state
            .unused_engine_game
            .as_ref()
            .unwrap();
        assert!(pending.reset_attempted && !pending.reset_completed);
        assert!(pending.first_error.is_some() && !pending.completed);
        assert_eq!(pending.deadline, until);
    }

    #[test]
    fn engine_unused_game_panic_keeps_original_payload_and_unknown_without_another_reset() {
        let until = Instant::now() + Duration::from_secs(20);
        let mut fixture = unused_game_zero_fixture(until);
        let scope = SearchAuthority {
            game: GameGeneration(0),
            ..engine_scope()
        };
        let store = fixture.owner.stores().read_owner();
        fixture.panic_on_reset.store(true, Ordering::Release);
        let first = fixture
            .owner
            .prepare_unused_archive_game(scope, until, &AtomicBool::new(false))
            .unwrap_err();
        fixture.armed.store(true, Ordering::Release);
        let repeated = fixture
            .owner
            .prepare_unused_archive_game(scope, until, &AtomicBool::new(false))
            .unwrap_err();
        assert_eq!(format!("{first:?}"), format!("{repeated:?}"));
        assert_eq!(fixture.calls.load(Ordering::Acquire), 1);
        assert!(fixture.armed.load(Ordering::Acquire));
        assert!(store.matches(fixture.owner.stores()));
        let pending = fixture
            .owner
            .semantic
            .archive_state
            .unused_engine_game
            .as_ref()
            .unwrap();
        assert!(pending.panic.is_some() && pending.completion_unknown && !pending.completed);
        assert!(pending.reset_attempted && !pending.reset_completed);
        assert_eq!(fixture.owner.semantic.adaptive_new_game_scope(), None);
    }
    struct EngineScratch(PathBuf, bool);
    impl EngineScratch {
        fn new() -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let path = std::env::temp_dir().join(format!(
                "rz-engine-idle-archive-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            std::fs::create_dir(&path).unwrap();
            Self(path, false)
        }
    }
    impl Drop for EngineScratch {
        fn drop(&mut self) {
            if self.1
                && self.0.parent() == Some(std::env::temp_dir().as_path())
                && self.0.file_name().is_some_and(|name| {
                    name.to_string_lossy()
                        .starts_with("rz-engine-idle-archive-")
                })
                && std::fs::symlink_metadata(&self.0)
                    .is_ok_and(|metadata| metadata.is_dir() && !metadata.file_type().is_symlink())
            {
                let _ = std::fs::remove_dir_all(&self.0);
            }
        }
    }
    fn engine_setup(scratch: &EngineScratch, until: Instant) -> UnusedIdleArchiveSetup {
        let directory =
            cap_std::fs::Dir::open_ambient_dir(&scratch.0, cap_std::ambient_authority()).unwrap();
        UnusedIdleArchiveSetup::new(
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
                max_io_bytes: SEARCH_ARCHIVE_IO_BYTES,
            },
        )
    }
    fn configured_engine(
        fixture: EngineFixture,
        scratch: &EngineScratch,
        until: Instant,
    ) -> (
        IdleSemanticArchiveOwner<LegalOrderRoleMock>,
        Arc<AtomicUsize>,
        Arc<AtomicBool>,
        Arc<AtomicBool>,
    ) {
        let mut owner = fixture
            .owner
            .into_idle_archive()
            .unwrap_or_else(|_| panic!("original Engine idle"));
        let seal = owner.seal_unused_setup(engine_scope()).unwrap();
        owner
            .execute_unused_setup(engine_scope(), &seal, &engine_setup(scratch, until))
            .unwrap();
        (owner, fixture.calls, fixture.armed, fixture.fail)
    }
    fn register_engine_root(
        owner: &mut Box<PalsEngine<LegalOrderRoleMock>>,
        scope: SearchAuthority,
    ) {
        let position = Position::startpos();
        let registered = owner.register_adaptive_root(&position, scope).unwrap();
        owner.abandon_unconsumed_adaptive_root(scope).unwrap();
        drop(registered);
        owner.intern(position).unwrap();
    }

    #[test]
    fn engine_idle_preserves_whole_box_checker_store_and_rejects_other_kinds_and_live_pin() {
        let until = Instant::now() + Duration::from_secs(20);
        let EngineFixture { owner, calls, .. } = engine_fixture(until);
        let original = std::ptr::from_ref(owner.as_ref());
        let store = owner.stores().read_owner();
        let checker = std::ptr::from_ref(owner.researcher.cpu.as_ref());
        let pin = owner.stores().pin_review(StorePins::default()).unwrap();
        let (owner, _) = owner
            .into_idle_archive()
            .err()
            .expect("live original pin rejected");
        assert_eq!(std::ptr::from_ref(owner.as_ref()), original);
        assert!(store.matches(owner.stores()));
        drop(pin);
        let idle = owner
            .into_idle_archive()
            .unwrap_or_else(|_| panic!("original idle"));
        assert_eq!(idle.owner_kind(), IdleArchiveOwnerKind::EnginePcr);
        let idle = idle.into_pc().err().expect("Engine is not PC");
        let idle = idle
            .into_researcher()
            .err()
            .expect("Engine is not preparing");
        let owner = idle
            .into_engine()
            .unwrap_or_else(|_| panic!("same full Engine"));
        assert_eq!(std::ptr::from_ref(owner.as_ref()), original);
        assert!(std::ptr::eq(
            std::ptr::from_ref(owner.researcher.cpu.as_ref()),
            checker
        ));
        assert!(store.matches(owner.stores()));
        assert_eq!(calls.load(Ordering::Acquire), 0);
    }

    #[test]
    fn engine_idle_newgame_consumes_reset_once_checkpoints_hot_and_finish_is_separate() {
        let mut scratch = EngineScratch::new();
        let until = Instant::now() + Duration::from_secs(20);
        let (idle, calls, armed, _) = configured_engine(engine_fixture(until), &scratch, until);
        let mut owner = idle
            .into_engine()
            .unwrap_or_else(|_| panic!("configured Engine"));
        let original = std::ptr::from_ref(owner.as_ref());
        register_engine_root(&mut owner, engine_scope());
        let old_store = owner.stores().read_owner();
        let old_archive = owner.archive_owner_snapshot().unwrap().unwrap();
        let old_hot = owner.stores().hot_stats();
        let mut idle = owner
            .into_idle_archive()
            .unwrap_or_else(|_| panic!("retired Engine"));
        let next = SearchAuthority {
            game: GameGeneration(2),
            root: RootGeneration(2),
            ..engine_scope()
        };
        armed.store(true, Ordering::Release);
        idle.advance_transition(
            IdleArchiveTransitionKind::NewGame(next),
            until,
            &AtomicBool::new(false),
        )
        .unwrap();
        assert_eq!(calls.load(Ordering::Acquire), 1);
        let checkpoint = idle.last_checkpoint().unwrap();
        assert_eq!(checkpoint.hot, old_hot);
        assert!(checkpoint.committed_bytes > 0);
        let actual = idle.archive_owner_snapshot().unwrap().unwrap();
        assert_ne!(actual.owner_id, old_archive.owner_id);
        assert_eq!(
            actual.session_write_ledger_id,
            old_archive.session_write_ledger_id
        );
        assert_eq!(
            actual.session_write_bytes_consumed,
            checkpoint.owner.session_write_bytes_consumed
        );
        assert!(
            actual.session_write_bytes_consumed.unwrap()
                >= old_archive.session_write_bytes_consumed.unwrap()
        );
        assert!(!old_store.matches(idle.stores()));
        let mut owner = idle
            .into_engine()
            .unwrap_or_else(|_| panic!("same full Engine after NewGame"));
        assert_eq!(std::ptr::from_ref(owner.as_ref()), original);
        register_engine_root(&mut owner, next);
        let mut idle = owner
            .into_idle_archive()
            .unwrap_or_else(|_| panic!("new-game Engine"));
        idle.advance_transition(
            IdleArchiveTransitionKind::Finish,
            until,
            &AtomicBool::new(false),
        )
        .unwrap();
        assert!(idle.transition_snapshot().unwrap().unwrap().completed);
        assert!(idle.last_checkpoint().unwrap().owner.complete);
        assert!(
            idle.archive_owner_snapshot()
                .unwrap()
                .unwrap()
                .admission_closed
        );
        assert!(idle.into_engine().is_err());
        assert_eq!(calls.load(Ordering::Acquire), 1);
        scratch.1 = true;
    }

    #[test]
    fn engine_idle_failed_reset_retains_original_error_and_does_not_consume_another_arm() {
        let scratch = EngineScratch::new();
        let until = Instant::now() + Duration::from_secs(20);
        let (idle, calls, armed, fail) = configured_engine(engine_fixture(until), &scratch, until);
        let mut owner = idle
            .into_engine()
            .unwrap_or_else(|_| panic!("original Engine"));
        register_engine_root(&mut owner, engine_scope());
        let store = owner.stores().read_owner();
        let mut idle = owner
            .into_idle_archive()
            .unwrap_or_else(|_| panic!("original idle"));
        let kind = IdleArchiveTransitionKind::NewGame(SearchAuthority {
            game: GameGeneration(2),
            root: RootGeneration(2),
            ..engine_scope()
        });
        armed.store(true, Ordering::Release);
        fail.store(true, Ordering::Release);
        let first = idle
            .advance_transition(kind, until, &AtomicBool::new(false))
            .unwrap_err();
        assert!(matches!(
            first,
            PalsError::Checker(CheckerError::Invalid(
                "original fixture reset delivery failure"
            ))
        ));
        armed.store(true, Ordering::Release); // not permission for this original command
        let repeated = idle
            .advance_transition(kind, until, &AtomicBool::new(false))
            .unwrap_err();
        assert_eq!(format!("{first:?}"), format!("{repeated:?}"));
        assert_eq!(calls.load(Ordering::Acquire), 1);
        assert!(armed.load(Ordering::Acquire));
        assert!(store.matches(idle.stores()));
        let pending = idle.pending.as_ref().unwrap();
        assert!(pending.engine_reset_attempted && !pending.engine_reset_completed);
        assert!(pending.engine_reset_error.is_some());
        assert_eq!(pending.deadline, until);
        assert_eq!(pending.remaining_io, SEARCH_ARCHIVE_IO_BYTES);
        assert!(pending.checkpoint.is_none() && pending.replacement.is_none());
        assert!(idle.into_engine().is_err());
    }

    #[test]
    fn engine_idle_restore_policy_preserves_actual_registered_value_and_recheck_semantics() {
        let until = Instant::now() + Duration::from_secs(20);
        let mut fixture = engine_fixture(until);
        fixture.owner.followup_lane = false;
        let value = fixture.owner.researcher.cpu_registered_value.clone();
        let mut idle = fixture
            .owner
            .into_idle_archive()
            .unwrap_or_else(|_| panic!("Engine"));
        assert_eq!(idle.cpu_value(), value.as_ref());
        assert!(idle.legacy_model_wdl());
        let IdleOwner::Engine(owner) = &mut idle.owner else {
            unreachable!()
        };
        owner.followup_lane = true;
        assert!(!idle.legacy_model_wdl());
        let IdleOwner::Engine(owner) = &mut idle.owner else {
            unreachable!()
        };
        owner.followup_lane = false;
        owner.post_repair_recheck = PostRepairRecheckPolicy::FrozenModelWdlV2;
        assert!(!idle.legacy_model_wdl());
        assert_eq!(idle.cpu_value(), value.as_ref());
    }

    #[test]
    fn engine_idle_typed_restore_distinguishes_real_absence_from_present_engine_metadata() {
        let mut scratch = EngineScratch::new();
        let until = Instant::now() + Duration::from_secs(20);
        let (idle, _, _, _) = configured_engine(engine_fixture(until), &scratch, until);
        let mut owner = idle
            .into_engine()
            .unwrap_or_else(|_| panic!("original Engine"));
        register_engine_root(&mut owner, engine_scope());
        let store = owner.stores().read_owner();
        let value = owner.researcher.cpu_registered_value.clone();
        let mut idle = owner
            .into_idle_archive()
            .unwrap_or_else(|_| panic!("actual idle"));
        let position = Position::startpos();
        let mut budget = ArchiveIoBudget {
            deadline: until,
            max_bytes: SEARCH_ARCHIVE_IO_BYTES,
        };
        assert_eq!(
            idle.restore_root_if_archived(&position, &mut budget)
                .unwrap(),
            None
        );
        let committed = idle
            .semantic_mut()
            .checkpoint_idle_archive_game(&mut budget)
            .unwrap();
        assert!(committed.nodes > 0 && committed.committed_bytes > 0);
        let root = idle
            .restore_root_if_archived(&position, &mut budget)
            .unwrap()
            .expect("actual committed Engine metadata");
        assert!(
            idle.semantic().nodes[root]
                .position
                .snapshot()
                .same_state(&position.snapshot())
        );
        assert_eq!(idle.cpu_value(), value.as_ref());
        assert!(store.matches(idle.stores()));
        idle.advance_transition(
            IdleArchiveTransitionKind::Finish,
            until,
            &AtomicBool::new(false),
        )
        .unwrap();
        scratch.1 = true;
    }

    #[test]
    fn preparing_idle_transfer_preserves_actual_source_and_rejects_a_live_view() {
        let owner = Box::new(super::super::preparing_root::tests::owner());
        let store = owner.stores().read_owner();
        let model = Arc::clone(owner.source().model());
        let pin = owner.stores().pin_review(StorePins::default()).unwrap();
        let (owner, error) = owner
            .into_idle_archive()
            .err()
            .expect("live original view rejected");
        assert!(matches!(
            error,
            PalsError::Store(StoreError::InvalidConditions(_))
        ));
        assert!(store.matches(owner.stores()));
        assert!(Arc::ptr_eq(owner.source().model(), &model));
        assert_eq!(owner.stores().review_pin_count().unwrap(), 1);
        drop(pin);
        let idle = owner
            .into_idle_archive()
            .unwrap_or_else(|_| panic!("idle original owner"));
        let idle = idle
            .into_pc()
            .err()
            .expect("wrong typed owner returned intact");
        let owner = idle
            .into_researcher()
            .unwrap_or_else(|_| panic!("original researcher owner"));
        assert!(store.matches(owner.stores()));
        assert!(Arc::ptr_eq(owner.source().model(), &model));
        assert_eq!(owner.stores().review_pin_count().unwrap(), 0);
        assert!(!owner.stores().archive_enabled());
    }
}
