//! Opt-in scalar copies of original NNUEC receipts. No view, model, Bound,
//! continuation token or release capability is retained by this journal.
use super::*;
use crate::pals_adaptive::NnuecPreparationRecord;
use rz_search::cpu_value::nnuec::v2::review::{PreparationStatus, Provenance};
use rz_search::cpu_worker::{NnuecReadyRegistration, ValueInstallReceipt};
use serde::Serialize;
pub(crate) mod cpu_slices;

const DETAIL_LIMIT: usize = 8;

#[derive(Clone, Debug, Serialize)]
pub(crate) struct NnuecOwnerObservation {
    pub authority: Authority,
    pub root_admission_sha256: String,
    pub initial_ready: Option<ReadyObservation>,
    pub preparations: Vec<PreparationRecord>,
    pub preparation_records_available: Option<u64>,
    pub preparation_records_omitted_by_manager: Option<u64>,
    pub preparation_details_omitted: Option<u64>,
    pub submitted_records_observed: Option<u64>,
    pub actor_terminal_ack_records_observed: Option<u64>,
    pub installation_ack_records_observed: Option<u64>,
    pub cancellation_requested_records_observed: Option<u64>,
    pub preparation_scope_observed: bool,
    pub last_joined_installation: Option<InstallationObservation>,
    pub last_joined_engine_stack: Option<StackObservation>,
    pub retirement: Option<RetirementObservation>,
    pub root_failure_code: Option<&'static str>,
    // Those chains require independent original semantic/readout evidence.
    // An installed evaluator or a joined worker cannot manufacture them.
    pub accepted_critic_chain_observed: Option<bool>,
    pub conditional_scalar_value_observed: Option<bool>,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct Authority {
    pub epoch: u64,
    pub game: u64,
    pub root: u64,
    pub implementation_sha256: String,
}
impl Authority {
    fn from_original(value: SearchAuthority) -> Self {
        Self {
            epoch: value.epoch.0,
            game: value.game.0,
            root: value.root.0,
            implementation_sha256: hex(value.implementation),
        }
    }
    pub(crate) fn matches(&self, value: SearchAuthority) -> bool {
        self.epoch == value.epoch.0
            && self.game == value.game.0
            && self.root == value.root.0
            && self.implementation_sha256 == hex(value.implementation)
    }
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct ProvenanceObservation {
    root_state: usize,
    generation: u64,
    view_sequence: u64,
    full_view_sha256: String,
    view_seal_sha256: String,
    record_scope_sha256: String,
    policy_sha256: String,
}
impl From<&Provenance> for ProvenanceObservation {
    fn from(value: &Provenance) -> Self {
        Self {
            root_state: value.root.0,
            generation: value.generation,
            view_sequence: value.sequence,
            full_view_sha256: hex(Digest(value.full_view_digest)),
            view_seal_sha256: hex(Digest(value.view_seal)),
            record_scope_sha256: hex(Digest(value.record_scope_digest)),
            policy_sha256: hex(Digest(value.policy_digest)),
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct ReadyObservation {
    owner: u16,
    actual_stack_owner_id: u64,
    binding_sha256: String,
    provenance: ProvenanceObservation,
    replacement_installation_receipt: bool,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct PreparationRecord {
    ordinal: u64,
    frame_sequence: u64,
    question_sha256: String,
    view_sha256: String,
    view_sequence: u64,
    owner: u16,
    stage: String,
    submitted: bool,
    terminal_status: Option<&'static str>,
    actor_terminal_ack_observed: bool,
    driver_attested: bool,
    cancellation_requested: bool,
    manager_error_observed: bool,
    installation_requested_sequence: Option<u64>,
    installation_receipt_sequence: Option<u64>,
    installation_succeeded: Option<bool>,
    binding_sha256: Option<String>,
    installed_owner: Option<u16>,
    actual_stack_owner_id: Option<u64>,
}
impl PreparationRecord {
    fn from_original(ordinal: u64, value: &NnuecPreparationRecord) -> Self {
        let ack = &value.original_ack;
        Self {
            ordinal,
            frame_sequence: value.frame.sequence,
            question_sha256: hex(value.question),
            view_sha256: hex(value.view.view_digest),
            view_sequence: value.view.view_sequence,
            owner: value.owner.0,
            stage: ack.stage.clone(),
            submitted: value.submitted,
            terminal_status: value
                .actor_terminal
                .map(|terminal| match terminal.status() {
                    PreparationStatus::Completed => "completed",
                    PreparationStatus::Failed => "failed",
                    PreparationStatus::Cancelled => "cancelled",
                    PreparationStatus::Expired => "expired",
                }),
            actor_terminal_ack_observed: ack.actor_terminal_ack_observed,
            driver_attested: ack.driver_attested,
            cancellation_requested: value.cancellation_requested,
            manager_error_observed: value.manager_error.is_some(),
            installation_requested_sequence: ack.installation_requested_sequence,
            installation_receipt_sequence: ack.installation_receipt_sequence,
            installation_succeeded: ack.installation_succeeded,
            binding_sha256: ack.binding.map(hex),
            installed_owner: ack.installed_owner.map(|v| v.0),
            actual_stack_owner_id: ack.actual_stack_owner_id,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct InstallationObservation {
    installation_sequence: u64,
    binding_sha256: String,
    acknowledgement_received: bool,
    installed: bool,
    installed_owner: Option<u16>,
    actual_stack_owner_id: Option<u64>,
}
impl From<&ValueInstallReceipt> for InstallationObservation {
    fn from(value: &ValueInstallReceipt) -> Self {
        Self {
            installation_sequence: value.installation_id().sequence(),
            binding_sha256: hex(value.binding_digest()),
            acknowledgement_received: true,
            installed: value.is_installed(),
            installed_owner: value.installed_profile().map(|p| p.owner.0),
            actual_stack_owner_id: value.result().ok().map(|r| r.actual_stack_owner_id()),
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct StackObservation {
    scope: &'static str,
    owner_id: u64,
    event_sequence: u64,
    tokens_created: u64,
    tokens_resumed: u64,
    tokens_invalidated: u64,
    tokens_retained: u32,
    replayed_consumed_work: u64,
    admission_closed: bool,
    owner_released: bool,
    complete: bool,
}
impl From<rz_search::cpu::CpuPausedStackSnapshot> for StackObservation {
    fn from(value: rz_search::cpu::CpuPausedStackSnapshot) -> Self {
        Self {
            scope: "last_joined_engine_lifetime_not_root_aggregate",
            owner_id: value.owner_id,
            event_sequence: value.event_sequence,
            tokens_created: value.tokens_created,
            tokens_resumed: value.tokens_resumed,
            tokens_invalidated: value.tokens_invalidated,
            tokens_retained: value.tokens_retained,
            replayed_consumed_work: value.replayed_consumed_work,
            admission_closed: value.admission_closed,
            owner_released: value.owner_released,
            complete: value.complete,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct RetirementObservation {
    owner: u16,
    actual_stack_owner_id: u64,
    binding_sha256: String,
    preparation_epoch: u64,
    cache_release_epoch: u64,
    engine_release_ack_received: bool,
    cache_release_ack_received: bool,
    engine_and_cache_ack_match: bool,
    returned_preparing_owner_observed: bool,
    provenance: ProvenanceObservation,
}

#[derive(Default)]
pub(super) struct NnuecDiagnosticJournal {
    enabled: AtomicBool,
    unavailable: AtomicBool,
    latest: Mutex<Option<NnuecOwnerObservation>>,
    cpu_slices: cpu_slices::CpuSliceJournal,
}
impl NnuecDiagnosticJournal {
    pub(super) fn enable(&self) {
        self.enabled.store(true, Ordering::Release);
    }
    pub(super) fn begin(&self, authority: SearchAuthority, root_admission: Digest) {
        if !self.enabled.load(Ordering::Acquire) {
            return;
        }
        match self.latest.try_lock() {
            Ok(mut latest) => {
                *latest = Some(NnuecOwnerObservation {
                    authority: Authority::from_original(authority),
                    root_admission_sha256: hex(root_admission),
                    initial_ready: None,
                    preparations: Vec::new(),
                    preparation_records_available: None,
                    preparation_records_omitted_by_manager: None,
                    preparation_details_omitted: None,
                    submitted_records_observed: None,
                    actor_terminal_ack_records_observed: None,
                    installation_ack_records_observed: None,
                    cancellation_requested_records_observed: None,
                    preparation_scope_observed: false,
                    last_joined_installation: None,
                    last_joined_engine_stack: None,
                    retirement: None,
                    root_failure_code: None,
                    accepted_critic_chain_observed: None,
                    conditional_scalar_value_observed: None,
                })
            }
            Err(_) => self.unavailable.store(true, Ordering::Release),
        }
        self.cpu_slices.begin(authority, root_admission);
    }
    pub(super) fn install_cpu_slices(&self, manager: &Manager) {
        if self.enabled.load(Ordering::Acquire) {
            self.cpu_slices.install(manager);
        }
    }
    pub(super) fn cpu_slice_snapshot(
        &self,
    ) -> Result<Option<cpu_slices::CpuSliceOwnerObservation>, &'static str> {
        if !self.enabled.load(Ordering::Acquire) {
            return Ok(None);
        }
        self.cpu_slices.snapshot()
    }
    fn update(&self, apply: impl FnOnce(&mut NnuecOwnerObservation)) {
        if !self.enabled.load(Ordering::Acquire) {
            return;
        }
        match self.latest.try_lock() {
            Ok(mut latest) => {
                if let Some(latest) = latest.as_mut() {
                    apply(latest);
                }
            }
            Err(_) => self.unavailable.store(true, Ordering::Release),
        }
    }
    pub(super) fn initial_ready(&self, ready: &NnuecReadyRegistration) {
        self.update(|latest| {
            latest.initial_ready = Some(ReadyObservation {
                owner: ready.profile().owner.0,
                actual_stack_owner_id: ready.actual_stack_owner_id(),
                binding_sha256: hex(ready.binding_digest()),
                provenance: ready.provenance().into(),
                replacement_installation_receipt: false,
            })
        });
    }
    pub(super) fn preparations(
        &self,
        values: impl Iterator<Item = NnuecPreparationRecord>,
        omitted: u64,
    ) {
        self.update(|latest| {
            latest.preparations.clear();
            let (mut available, mut submitted, mut actor_ack, mut installation_ack, mut canceled) =
                (0, 0, 0, 0, 0);
            latest.preparation_records_omitted_by_manager = Some(omitted);
            for (ordinal, value) in values.enumerate() {
                available += 1;
                submitted += u64::from(value.submitted);
                actor_ack += u64::from(value.original_ack.actor_terminal_ack_observed);
                installation_ack +=
                    u64::from(value.original_ack.installation_receipt_sequence.is_some());
                canceled += u64::from(value.cancellation_requested);
                if latest.preparations.len() < DETAIL_LIMIT {
                    latest
                        .preparations
                        .push(PreparationRecord::from_original(ordinal as u64, &value));
                }
            }
            latest.preparation_records_available = Some(available);
            latest.preparation_details_omitted = Some(available - latest.preparations.len() as u64);
            latest.submitted_records_observed = Some(submitted);
            latest.actor_terminal_ack_records_observed = Some(actor_ack);
            latest.installation_ack_records_observed = Some(installation_ack);
            latest.cancellation_requested_records_observed = Some(canceled);
            latest.preparation_scope_observed = true;
        });
    }
    pub(super) fn joined(&self, owner: &AdaptiveNnuecRetiredOwner) {
        self.update(|latest| {
            latest.last_joined_installation = owner.last_installation.as_ref().map(Into::into);
            latest.last_joined_engine_stack = owner.engine.paused_stack_snapshot().map(Into::into);
        });
    }
    pub(super) fn retired(&self, retired: &retirement::NnuecRootRetired) {
        self.update(|latest| {
            latest.retirement = Some(RetirementObservation {
                owner: retired.engine_release.owner().0,
                actual_stack_owner_id: retired.engine_release.actual_stack_owner_id(),
                binding_sha256: hex(retired.engine_release.binding_digest()),
                preparation_epoch: retired.engine_release.preparation_epoch(),
                cache_release_epoch: retired.cache_release.epoch(),
                engine_release_ack_received: true,
                cache_release_ack_received: true,
                engine_and_cache_ack_match: retired
                    .engine_release
                    .matches_cache_release(&retired.cache_release),
                returned_preparing_owner_observed: true,
                provenance: retired.engine_release.provenance().into(),
            })
        });
    }
    pub(super) fn failure(&self, code: &'static str) {
        self.update(|latest| {
            latest.root_failure_code.get_or_insert(code);
        });
    }
    pub(crate) fn snapshot(&self) -> Result<Option<NnuecOwnerObservation>, &'static str> {
        if self.unavailable.load(Ordering::Acquire) {
            return Err("nnuec_original_diagnostic_unavailable");
        }
        self.latest
            .try_lock()
            .map(|v| v.clone())
            .map_err(|_| "nnuec_original_diagnostic_busy_or_poisoned")
    }
}

fn hex(value: Digest) -> String {
    value.0.iter().map(|byte| format!("{byte:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    fn authority() -> SearchAuthority {
        SearchAuthority {
            epoch: rz_contracts::ProcessEpoch(1),
            game: rz_contracts::GameGeneration(2),
            root: rz_contracts::RootGeneration(3),
            implementation: Digest([4; 32]),
        }
    }
    fn preparation(sequence: u64) -> NnuecPreparationRecord {
        let authority = authority();
        NnuecPreparationRecord {
            frame: rz_search::pals::coordinator::FrameId {
                authority,
                sequence,
            },
            question: Digest([5; 32]),
            view: rz_contracts::pals_adaptive::ReviewViewSeal {
                authority,
                view_sequence: sequence,
                context_policy: Digest([6; 32]),
                view_digest: Digest([7; 32]),
                root_state: rz_contracts::StateIdentity {
                    owner: rz_contracts::OwnerId(1),
                    revision: rz_contracts::StateRevision(1),
                    semantic: Digest([8; 32]),
                },
                actual_tokens: 8,
                selected_records: 1,
                anchors: 1,
                length_bucket: 8,
            },
            owner: CpuWorkerId(0),
            submitted: true,
            actor_terminal: None,
            cpu_thread_time: None,
            cancellation_requested: sequence % 2 == 0,
            manager_error: None,
            original_ack: crate::pals_adaptive::NnuecPreparationAckObservation {
                stage: "CpuVerify".into(),
                actor_terminal_ack_observed: false,
                driver_attested: false,
                installation_requested_sequence: None,
                installation_receipt_sequence: None,
                installation_succeeded: None,
                binding: None,
                installed_owner: None,
                actual_stack_owner_id: None,
            },
        }
    }
    #[test]
    fn disabled_journal_does_not_create_or_traverse_observations() {
        let journal = NnuecDiagnosticJournal::default();
        journal.preparations(
            std::iter::from_fn(|| -> Option<NnuecPreparationRecord> {
                panic!("disabled diagnostic must not advance an iterator")
            }),
            4,
        );
        assert!(journal.snapshot().unwrap().is_none());
    }
    #[test]
    fn unobserved_receipts_and_semantic_values_remain_null() {
        let journal = NnuecDiagnosticJournal::default();
        journal.enable();
        let authority = SearchAuthority {
            epoch: rz_contracts::ProcessEpoch(1),
            game: rz_contracts::GameGeneration(2),
            root: rz_contracts::RootGeneration(3),
            implementation: Digest([4; 32]),
        };
        journal.begin(authority, Digest([9; 32]));
        let snapshot = journal.snapshot().unwrap().unwrap();
        assert!(snapshot.authority.matches(authority));
        let raw = serde_json::to_value(snapshot).unwrap();
        for field in [
            "initial_ready",
            "last_joined_installation",
            "last_joined_engine_stack",
            "retirement",
            "accepted_critic_chain_observed",
            "conditional_scalar_value_observed",
            "preparation_records_available",
            "preparation_records_omitted_by_manager",
            "preparation_details_omitted",
            "submitted_records_observed",
            "actor_terminal_ack_records_observed",
            "installation_ack_records_observed",
            "cancellation_requested_records_observed",
        ] {
            assert!(raw[field].is_null(), "{field}");
        }
        assert_eq!(raw["preparation_scope_observed"], false);
    }
    #[test]
    fn observed_empty_preparation_scope_is_distinct_from_missing_scope() {
        let journal = NnuecDiagnosticJournal::default();
        journal.enable();
        journal.begin(
            SearchAuthority {
                epoch: rz_contracts::ProcessEpoch(1),
                game: rz_contracts::GameGeneration(2),
                root: rz_contracts::RootGeneration(3),
                implementation: Digest([4; 32]),
            },
            Digest([9; 32]),
        );
        journal.preparations(std::iter::empty(), 5);
        let original = journal.snapshot().unwrap().unwrap();
        assert!(original.preparation_scope_observed);
        assert_eq!(original.preparation_records_available, Some(0));
        assert_eq!(original.preparation_records_omitted_by_manager, Some(5));
        assert_eq!(original.submitted_records_observed, Some(0));
        assert!(original.conditional_scalar_value_observed.is_none());
    }
    #[test]
    fn ordered_detail_prefix_is_bounded_and_dropped_history_stays_explicit() {
        let journal = NnuecDiagnosticJournal::default();
        journal.enable();
        journal.begin(authority(), Digest([9; 32]));
        journal.preparations((11..21).map(preparation), 6);
        let original = journal.snapshot().unwrap().unwrap();
        assert_eq!(original.preparations.len(), DETAIL_LIMIT);
        assert_eq!(original.preparation_records_available, Some(10));
        assert_eq!(original.preparation_records_omitted_by_manager, Some(6));
        assert_eq!(original.preparation_details_omitted, Some(2));
        assert_eq!(original.submitted_records_observed, Some(10));
        assert_eq!(original.installation_ack_records_observed, Some(0));
        assert_eq!(original.cancellation_requested_records_observed, Some(5));
        assert_eq!(
            original
                .preparations
                .iter()
                .map(|p| p.frame_sequence)
                .collect::<Vec<_>>(),
            (11..19).collect::<Vec<_>>()
        );
        assert!(
            original
                .preparations
                .iter()
                .all(|p| p.terminal_status.is_none()
                    && p.installation_succeeded.is_none()
                    && p.actual_stack_owner_id.is_none())
        );
        assert!(serde_json::to_vec(&original).unwrap().len() < 8192);
        let mut next = authority();
        next.root = rz_contracts::RootGeneration(9);
        journal.begin(next, Digest([10; 32]));
        let current = journal.snapshot().unwrap().unwrap();
        assert!(current.authority.matches(next));
        assert_eq!(current.root_admission_sha256, hex(Digest([10; 32])));
        assert!(!current.authority.matches(authority()));
        assert!(current.preparations.is_empty());
        assert!(current.preparation_records_available.is_none());
        assert!(current.retirement.is_none());
    }
}
