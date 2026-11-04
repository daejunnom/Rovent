//! Bounded aggregate evidence: no per-go journal grows with game length.
//! None means an observation was unavailable; an observed no-work path is zero.
use super::{SearchKind, SearchSessionContext, SearchSessionFailure, SearchSessionReport};
use rz_contracts::{ContractError, ErrorCode, Stage, pals::SearchAuthority};
use rz_position::BoardMove;
use rz_search::{
    cpu::{CpuReport, CpuScoreScope, CpuWork},
    pals::engine::PalsCounters,
};
use sha2::{Digest as _, Sha256};
use std::sync::Mutex;

#[derive(Clone, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "search-work-receipts", derive(serde::Serialize))]
pub struct PalsResolverIdentity {
    pub version: String,
    pub semantics_sha256: [u8; 32],
}
impl PalsResolverIdentity {
    pub fn from_semantics(version: &'static str, semantics: &'static str) -> Self {
        Self {
            version: version.into(),
            semantics_sha256: Sha256::digest(semantics.as_bytes()).into(),
        }
    }
    fn registered() -> Self {
        Self::from_semantics(
            rz_search::pals::engine::PALS_VALUE_RESOLVER_VERSION,
            rz_search::pals::engine::PALS_VALUE_RESOLVER_SEMANTICS,
        )
    }
    fn registered_model_wdl() -> Self {
        Self::from_semantics(
            rz_search::pals::value::MODEL_WDL_RESOLVER_VERSION,
            rz_search::pals::value::MODEL_WDL_RESOLVER_SEMANTICS,
        )
    }
}

/// Immutable startup selection, not proof that a post-Repair Reply ran or won.
/// The value resolver, model identity and completion counters stay independent.
#[derive(Clone, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "search-work-receipts", derive(serde::Serialize))]
pub struct PalsSearchPolicyIdentity {
    pub version: String,
    pub policy: String,
    pub search_identity: String,
    pub conditions_sha256: [u8; 32],
}
impl PalsSearchPolicyIdentity {
    pub const VERSION: &'static str = "pals-post-repair-recheck/1";
    pub const POLICY: &'static str = "same_repaired_line_once_v1";

    pub fn validate(&self) -> Result<(), ContractError> {
        self.selected_policy().map(|_| ())
    }
    pub(crate) fn selected_policy(
        &self,
    ) -> Result<rz_search::pals::engine::PostRepairRecheckPolicy, ContractError> {
        use rz_search::pals::engine::PostRepairRecheckPolicy;
        for selected in [
            PostRepairRecheckPolicy::SameRepairedLineOnceV1,
            PostRepairRecheckPolicy::ActualOpponentContinuationV1,
            PostRepairRecheckPolicy::FrozenModelWdlV2,
            PostRepairRecheckPolicy::IterativeFrozenModelWdlV2,
        ] {
            let conditions: [u8; 32] =
                Sha256::digest(selected.conditions().unwrap().as_bytes()).into();
            if Some(self.version.as_str()) == selected.registration_version()
                && Some(self.policy.as_str()) == selected.registration_policy()
                && self.search_identity == selected.search_identity()
                && self.conditions_sha256 == conditions
            {
                return Ok(selected);
            }
        }
        Err(ContractError::new(
            ErrorCode::IdentityMismatch,
            Stage::Admission,
            "PALS selected search policy identity differs from its closed lane",
        ))
    }
}

#[cfg(feature = "search-work-receipts")]
impl<'de> serde::Deserialize<'de> for PalsSearchPolicyIdentity {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(serde::Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Wire {
            version: String,
            policy: String,
            search_identity: String,
            conditions_sha256: [u8; 32],
        }
        let wire = <Wire as serde::Deserialize<'de>>::deserialize(deserializer)?;
        let identity = Self {
            version: wire.version,
            policy: wire.policy,
            search_identity: wire.search_identity,
            conditions_sha256: wire.conditions_sha256,
        };
        identity.validate().map_err(serde::de::Error::custom)?;
        Ok(identity)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "search-work-receipts", derive(serde::Serialize))]
pub struct ProcessSearchWorkReceipt {
    pub schema_version: u32,
    pub search_kind: SearchKind,
    pub go_invocations: u64,
    pub successful_returns: u64,
    pub failed_returns: u64,
    pub canceled_returns: u64,
    pub deadline_returns: u64,
    pub physical_unknown_returns: u64,
    pub active_invocations: u64,
    pub unobserved_work_invocations: u64,
    pub cpu: Option<CpuWorkTotals>,
    pub pals: Option<PalsWorkTotals>,
    /// Startup-selected identity remains available even for a failed go. The
    /// driver rejects a returned result with a different resolver version.
    #[cfg_attr(
        feature = "search-work-receipts",
        serde(default, skip_serializing_if = "Option::is_none")
    )]
    pub pals_resolver: Option<PalsResolverIdentity>,
    /// Absent in every legacy/default receipt. A present marker identifies only
    /// the actual startup-selected immutable lane, including failed/early go.
    #[cfg_attr(
        feature = "search-work-receipts",
        serde(default, skip_serializing_if = "Option::is_none")
    )]
    pub pals_search_policy: Option<PalsSearchPolicyIdentity>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "search-work-receipts", derive(serde::Serialize))]
pub struct CpuWorkTotals {
    pub tasks_requested: Option<u64>,
    pub reports_returned: Option<u64>,
    pub requested_depth_completed: Option<u64>,
    pub shallower_completed_iterations: Option<u64>,
    pub frontier_only_reports: Option<u64>,
    pub rules_terminal_reports: Option<u64>,
    /// Exact natural result accepted by UCI's owner after its physical fence.
    /// Transport write/exit success remains a separate process observation.
    pub reports_accepted_for_uci_output: Option<u64>,
    pub completed_reports_accepted_for_uci_output: Option<u64>,
    pub nodes: Option<u64>,
    pub quiescence_nodes: Option<u64>,
    pub tt_hits: Option<u64>,
    pub max_completed_depth: Option<u16>,
}
impl CpuWorkTotals {
    fn zero() -> Self {
        Self {
            tasks_requested: Some(0),
            reports_returned: Some(0),
            requested_depth_completed: Some(0),
            shallower_completed_iterations: Some(0),
            frontier_only_reports: Some(0),
            rules_terminal_reports: Some(0),
            reports_accepted_for_uci_output: Some(0),
            completed_reports_accepted_for_uci_output: Some(0),
            nodes: Some(0),
            quiescence_nodes: Some(0),
            tt_hits: Some(0),
            max_completed_depth: Some(0),
        }
    }
    fn observe(&mut self, report: &CpuObservation) {
        add(&mut self.tasks_requested, 1);
        add(&mut self.reports_returned, 1);
        add(&mut self.nodes, report.nodes);
        add(&mut self.quiescence_nodes, report.quiescence_nodes);
        add(&mut self.tt_hits, report.tt_hits);
        match report.score_scope {
            CpuScoreScope::CompletedIteration
                if report.completed_depth >= report.requested_depth =>
            {
                add(&mut self.requested_depth_completed, 1)
            }
            CpuScoreScope::CompletedIteration => add(&mut self.shallower_completed_iterations, 1),
            CpuScoreScope::FrontierOnly => add(&mut self.frontier_only_reports, 1),
            CpuScoreScope::RulesTerminal => add(&mut self.rules_terminal_reports, 1),
        }
        self.max_completed_depth = self
            .max_completed_depth
            .map(|value| value.max(report.completed_depth));
    }
    fn failed_attempt(&mut self) {
        add(&mut self.tasks_requested, 1);
        // No actual work snapshot was observed; earlier successful totals cannot be
        // presented as the full sum of this process after the missing attempt.
        self.nodes = None;
        self.quiescence_nodes = None;
        self.tt_hits = None;
        self.max_completed_depth = None;
    }
}

macro_rules! pals_totals {
    ($($field:ident),+ $(,)?) => {
        #[derive(Clone, Debug, Eq, PartialEq)]
        #[cfg_attr(feature = "search-work-receipts", derive(serde::Serialize))]
        pub struct PalsWorkTotals {
            $(pub $field: Option<u64>,)+
            pub retained_situations_peak: Option<usize>,
            pub unknown_root_children: Option<u64>,
            /// Observed foreign nodes remain a partial observed sum if this is
            /// true. Neither zero nor a reserved budget replaces missing work.
            pub external_checker_work_incomplete: Option<bool>,
        }
        impl PalsWorkTotals {
            fn zero() -> Self { Self { $($field: Some(0),)+ retained_situations_peak: Some(0), unknown_root_children: Some(0), external_checker_work_incomplete: Some(false) } }
            fn observe(&mut self, counters: PalsCounters, root_coverage_observed: bool) {
                $(add(&mut self.$field, counters.$field);)+
                self.external_checker_work_incomplete = self.external_checker_work_incomplete.map(|was_incomplete| was_incomplete || counters.external_checker_work_incomplete);
                self.retained_situations_peak = self.retained_situations_peak.map(|value| value.max(counters.retained_situations));
                if root_coverage_observed { add(&mut self.unknown_root_children, counters.unknown_root_children as u64); }
                else { self.unknown_root_children = None; }
                if counters.cpu_work_observation_incomplete {
                    self.cpu_nodes = None; self.cpu_quiescence_nodes = None; self.cpu_tt_hits = None;
                }
            }
            fn unknown(&mut self) { $(self.$field = None;)+ self.retained_situations_peak = None; self.unknown_root_children = None; self.external_checker_work_incomplete = None; }
        }
    };
}
pals_totals!(
    rounds,
    proposals,
    refutations,
    repairs,
    supported_refutations,
    supported_repairs,
    role_calls,
    consumed_role_outputs,
    proposer_calls,
    critic_calls,
    repair_calls,
    completed_proposer_calls,
    completed_critic_calls,
    completed_repair_calls,
    accepted_proposer_outputs,
    accepted_critic_outputs,
    accepted_repair_outputs,
    value_calls,
    completed_value_calls,
    accepted_value_outputs,
    external_checker_tasks,
    external_checker_reports,
    external_checker_nodes_observed,
    external_checker_node_budget_reserved,
    consumed_external_checker_tasks,
    cpu_tasks_requested,
    cpu_tasks,
    cpu_nodes,
    cpu_quiescence_nodes,
    cpu_tt_hits,
    completed_cpu_tasks,
    partial_cpu_iterations,
    consumed_cpu_tasks,
    reused_completed_cpu_tasks_consumed,
    consumed_partial_cpu_values,
    consumed_frontier_cpu_values,
    consumed_cached_cpu_values,
    evidence_cache_hits,
    examined_edges
);

fn add(total: &mut Option<u64>, increment: u64) {
    *total = total.and_then(|value| value.checked_add(increment));
}
fn fault() -> ContractError {
    ContractError::new(
        ErrorCode::BackendFailure,
        Stage::Output,
        "search work receipt owner unavailable or counter exhausted",
    )
}
// One fixed-size counter snapshot lives on the stack per admitted go, never
// per node. Keeping it inline avoids a heap allocation on failure/cancel
// accounting paths; this bounded scalar snapshot has no owned buffers.
#[allow(clippy::large_enum_variant)]
pub(crate) enum AttemptObservation {
    NoWork,
    CpuStarted,
    CpuFailed(CpuWork),
    Cpu(CpuObservation),
    PalsStarted,
    Pals {
        counters: PalsCounters,
        root_coverage_observed: bool,
    },
}
pub(crate) struct CpuObservation {
    nodes: u64,
    quiescence_nodes: u64,
    tt_hits: u64,
    completed_depth: u16,
    score_scope: CpuScoreScope,
    requested_depth: u16,
    best_move: Option<BoardMove>,
}
impl AttemptObservation {
    pub fn cpu(report: &CpuReport, requested_depth: u16) -> Self {
        Self::Cpu(CpuObservation {
            nodes: report.nodes,
            quiescence_nodes: report.quiescence_nodes,
            tt_hits: report.tt_hits,
            completed_depth: report.completed_depth,
            score_scope: report.score_scope,
            requested_depth,
            best_move: report.best_move,
        })
    }
}
#[derive(Clone, Copy)]
struct PendingCpuOutput {
    authority: SearchAuthority,
    best_move: Option<BoardMove>,
    complete: bool,
}
pub(crate) struct ProcessWorkJournal {
    receipt: Mutex<ProcessSearchWorkReceipt>,
    // A bounded set of unacknowledged reports, never a growing go journal.
    pending_cpu: Mutex<Vec<PendingCpuOutput>>,
}
impl ProcessWorkJournal {
    pub fn new(kind: SearchKind) -> Self {
        Self::with_resolver(
            kind,
            (kind == SearchKind::Pals).then(PalsResolverIdentity::registered),
        )
    }
    pub fn new_pals(resolver: PalsResolverIdentity) -> Self {
        Self::with_resolver(SearchKind::Pals, Some(resolver))
    }
    pub fn new_pals_with_search_policy(
        resolver: PalsResolverIdentity,
        policy: PalsSearchPolicyIdentity,
    ) -> Result<Self, ContractError> {
        let selected = policy.selected_policy()?;
        let own = resolver == PalsResolverIdentity::registered();
        let model = resolver == PalsResolverIdentity::registered_model_wdl();
        if !(own || selected.uses_frozen_model_wdl() && model) {
            return Err(ContractError::new(
                ErrorCode::UnsupportedContract,
                Stage::Admission,
                "post-Repair recheck requires a registered resolver; legacy v1 permits only the own CPU value resolver",
            ));
        }
        Ok(Self::with_registration(
            SearchKind::Pals,
            Some(resolver),
            Some(policy),
        ))
    }
    fn with_resolver(kind: SearchKind, resolver: Option<PalsResolverIdentity>) -> Self {
        Self::with_registration(kind, resolver, None)
    }
    fn with_registration(
        kind: SearchKind,
        resolver: Option<PalsResolverIdentity>,
        policy: Option<PalsSearchPolicyIdentity>,
    ) -> Self {
        Self {
            receipt: Mutex::new(ProcessSearchWorkReceipt {
                schema_version: 1,
                search_kind: kind,
                go_invocations: 0,
                successful_returns: 0,
                failed_returns: 0,
                canceled_returns: 0,
                deadline_returns: 0,
                physical_unknown_returns: 0,
                active_invocations: 0,
                unobserved_work_invocations: 0,
                cpu: (kind == SearchKind::Cpu).then(CpuWorkTotals::zero),
                pals: (kind == SearchKind::Pals).then(PalsWorkTotals::zero),
                pals_resolver: resolver,
                pals_search_policy: policy,
            }),
            pending_cpu: Mutex::new(Vec::with_capacity(16)),
        }
    }
    pub fn begin(&self) -> Result<(), ContractError> {
        let mut receipt = self.receipt.lock().map_err(|_| fault())?;
        receipt.go_invocations = receipt.go_invocations.checked_add(1).ok_or_else(fault)?;
        receipt.active_invocations = receipt
            .active_invocations
            .checked_add(1)
            .ok_or_else(fault)?;
        Ok(())
    }
    pub fn finish(
        &self,
        context: &SearchSessionContext,
        result: &Result<SearchSessionReport, SearchSessionFailure>,
        observation: AttemptObservation,
    ) -> Result<(), ContractError> {
        let mut receipt = self.receipt.lock().map_err(|_| fault())?;
        receipt.active_invocations = receipt
            .active_invocations
            .checked_sub(1)
            .ok_or_else(fault)?;
        if result.is_ok() {
            receipt.successful_returns = receipt
                .successful_returns
                .checked_add(1)
                .ok_or_else(fault)?;
        } else {
            receipt.failed_returns = receipt.failed_returns.checked_add(1).ok_or_else(fault)?;
        }
        if context
            .control
            .cancellation
            .load(std::sync::atomic::Ordering::Acquire)
            || context.cancellation.is_canceled()
        {
            receipt.canceled_returns = receipt.canceled_returns.checked_add(1).ok_or_else(fault)?;
        } else if std::time::Instant::now() >= context.control.admission_deadline {
            receipt.deadline_returns = receipt.deadline_returns.checked_add(1).ok_or_else(fault)?;
        }
        if result.as_ref().err().is_some_and(|failure| {
            failure.physical_completion == super::DriverPhysicalCompletion::Unknown
        }) {
            receipt.physical_unknown_returns = receipt
                .physical_unknown_returns
                .checked_add(1)
                .ok_or_else(fault)?;
        }
        let mut unknown = false;
        match observation {
            AttemptObservation::NoWork => {}
            AttemptObservation::Cpu(report) => {
                receipt.cpu.as_mut().ok_or_else(fault)?.observe(&report);
                if result.is_ok() && context.accepts()? {
                    let mut pending = self.pending_cpu.lock().map_err(|_| fault())?;
                    // A single UCI owner cannot subsequently accept a report
                    // from a replaced generation. Prune only that known scope.
                    pending.retain(|entry| {
                        entry.authority.epoch != context.authority.epoch
                            || entry.authority.implementation != context.authority.implementation
                            || (entry.authority.game == context.authority.game
                                && entry.authority.root == context.authority.root)
                    });
                    if pending.len() < 16 {
                        pending.push(PendingCpuOutput {
                            authority: context.authority,
                            best_move: report.best_move,
                            complete: report.score_scope == CpuScoreScope::RulesTerminal
                                || (report.score_scope == CpuScoreScope::CompletedIteration
                                    && report.completed_depth >= report.requested_depth),
                        });
                    } else {
                        let totals = receipt.cpu.as_mut().ok_or_else(fault)?;
                        totals.reports_accepted_for_uci_output = None;
                        totals.completed_reports_accepted_for_uci_output = None;
                        unknown = true;
                    }
                }
            }
            AttemptObservation::CpuStarted => {
                receipt.cpu.as_mut().ok_or_else(fault)?.failed_attempt();
                unknown = true;
            }
            AttemptObservation::CpuFailed(work) => {
                let totals = receipt.cpu.as_mut().ok_or_else(fault)?;
                add(&mut totals.tasks_requested, 1);
                add(&mut totals.nodes, work.nodes);
                add(&mut totals.quiescence_nodes, work.quiescence_nodes);
                add(&mut totals.tt_hits, work.tt_hits);
                if work.nodes > 0 {
                    // The invocation observed actual work but supplied no
                    // report of its last completed iteration. Preserve that
                    // distinct missing gauge rather than inventing depth 0.
                    totals.max_completed_depth = None;
                    unknown = true;
                }
            }
            AttemptObservation::Pals {
                counters,
                root_coverage_observed,
            } => {
                unknown = counters.cpu_work_observation_incomplete;
                receipt
                    .pals
                    .as_mut()
                    .ok_or_else(fault)?
                    .observe(counters, root_coverage_observed);
            }
            AttemptObservation::PalsStarted => {
                receipt.pals.as_mut().ok_or_else(fault)?.unknown();
                unknown = true;
            }
        }
        if unknown {
            receipt.unobserved_work_invocations = receipt
                .unobserved_work_invocations
                .checked_add(1)
                .ok_or_else(fault)?;
        }
        Ok(())
    }
    pub fn snapshot(&self) -> Result<ProcessSearchWorkReceipt, ContractError> {
        self.receipt
            .lock()
            .map(|receipt| receipt.clone())
            .map_err(|_| fault())
    }
    pub fn accept_cpu_output(
        &self,
        authority: SearchAuthority,
        bestmove: &str,
        from_report: bool,
    ) -> Result<(), ContractError> {
        let best_move = if bestmove == "0000" {
            None
        } else {
            Some(BoardMove::from_uci(bestmove).map_err(|_| fault())?)
        };
        let accepted = {
            let mut pending = self.pending_cpu.lock().map_err(|_| fault())?;
            pending
                .iter()
                .position(|entry| entry.authority == authority && entry.best_move == best_move)
                .map(|index| pending.swap_remove(index))
        };
        if let Some(accepted) = accepted {
            let mut receipt = self.receipt.lock().map_err(|_| fault())?;
            let totals = receipt.cpu.as_mut().ok_or_else(fault)?;
            if from_report {
                add(&mut totals.reports_accepted_for_uci_output, 1);
                if accepted.complete {
                    add(&mut totals.completed_reports_accepted_for_uci_output, 1);
                }
            } else {
                // A frozen fallback can coincide with the final reported move.
                // Without the exact completion handoff, coincidence does not
                // prove that UCI consumed this task instead of earlier progress.
                totals.reports_accepted_for_uci_output = None;
                if accepted.complete {
                    totals.completed_reports_accepted_for_uci_output = None;
                }
                receipt.unobserved_work_invocations = receipt
                    .unobserved_work_invocations
                    .checked_add(1)
                    .ok_or_else(fault)?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        contracts::SessionScope,
        search_driver::{DriverWork, SearchKind},
    };
    use rz_contracts::{
        CancelToken, Digest, GameGeneration, ProcessEpoch, RootGeneration, pals::SearchAuthority,
    };
    use rz_search::driver::SearchControl;
    use std::{
        sync::Arc,
        time::{Duration, Instant},
    };
    fn context() -> SearchSessionContext {
        let authority = SearchAuthority {
            epoch: ProcessEpoch(1),
            game: GameGeneration(1),
            root: RootGeneration(1),
            implementation: Digest([1; 32]),
        };
        let until = Instant::now() + Duration::from_secs(5);
        SearchSessionContext {
            authority,
            root_moves: None,
            goal: rz_contracts::rovent::SearchGoal::default(),
            original_go: None,
            original_time_budget: None,
            original_start: None,
            effective_clock: None,
            requested_nodes: None,
            control: SearchControl::new(until, 10),
            cancellation: CancelToken::new(),
            shutdown_deadline: until,
            current: Arc::new(Mutex::new(SessionScope::Search(authority))),
        }
    }
    fn policy_fixture() -> PalsSearchPolicyIdentity {
        PalsSearchPolicyIdentity {
            version: PalsSearchPolicyIdentity::VERSION.into(),
            policy: PalsSearchPolicyIdentity::POLICY.into(),
            search_identity: rz_search::pals::engine::POST_REPAIR_RECHECK_SEARCH_VERSION.into(),
            conditions_sha256: Sha256::digest(
                rz_search::pals::engine::POST_REPAIR_RECHECK_CONDITIONS.as_bytes(),
            )
            .into(),
        }
    }

    #[test]
    fn selected_policy_survives_early_and_unknown_failure_without_invented_work() {
        let policy = policy_fixture();
        let journal = ProcessWorkJournal::new_pals_with_search_policy(
            PalsResolverIdentity::registered(),
            policy.clone(),
        )
        .unwrap();
        let before = journal.snapshot().unwrap();
        assert_eq!(before.pals_search_policy.as_ref(), Some(&policy));
        assert_eq!(before.go_invocations, 0);
        assert_eq!(before.pals.unwrap().role_calls, Some(0));
        let context = context();
        context.cancellation.cancel();
        journal.begin().unwrap();
        journal
            .finish(
                &context,
                &Ok(SearchSessionReport {
                    mate_strategy: None,
                    best_move: None,
                    work: DriverWork::Pals {
                        rounds: 0,
                        cpu_nodes: 0,
                        completed_tasks: 0,
                        consumed_role_outputs: 0,
                        retained_situations: 0,
                    },
                }),
                AttemptObservation::NoWork,
            )
            .unwrap();
        let early = journal.snapshot().unwrap();
        assert_eq!(early.pals_search_policy.as_ref(), Some(&policy));
        assert_eq!((early.successful_returns, early.canceled_returns), (1, 1));
        assert_eq!(early.pals.unwrap().completed_cpu_tasks, Some(0));
        journal.begin().unwrap();
        journal
            .finish(
                &context,
                &Err(SearchSessionFailure::physical_completion_unknown()),
                AttemptObservation::PalsStarted,
            )
            .unwrap();
        let failed = journal.snapshot().unwrap();
        assert_eq!(failed.pals_search_policy.as_ref(), Some(&policy));
        assert_eq!(
            (failed.failed_returns, failed.physical_unknown_returns),
            (1, 1)
        );
        assert_eq!(failed.unobserved_work_invocations, 1);
        let totals = failed.pals.unwrap();
        assert_eq!(totals.completed_cpu_tasks, None);
        assert_eq!(totals.role_calls, None);
        assert_eq!(totals.cpu_nodes, None);
    }

    #[test]
    fn wrong_policy_or_foreign_resolver_cannot_register_the_owned_lane() {
        for field in ["version", "policy", "search", "conditions"] {
            let mut wrong = policy_fixture();
            match field {
                "version" => wrong.version = "pals".into(),
                "policy" => wrong.policy = "disabled".into(),
                "search" => {
                    wrong.search_identity = rz_search::pals::engine::PALS_SEARCH_VERSION.into()
                }
                "conditions" => wrong.conditions_sha256[0] ^= 1,
                _ => unreachable!(),
            }
            assert!(
                ProcessWorkJournal::new_pals_with_search_policy(
                    PalsResolverIdentity::registered(),
                    wrong,
                )
                .is_err()
            );
        }
        let foreign = PalsResolverIdentity::from_semantics(
            rz_search::pals::value::MODEL_WDL_RESOLVER_VERSION,
            rz_search::pals::value::MODEL_WDL_RESOLVER_SEMANTICS,
        );
        assert!(
            ProcessWorkJournal::new_pals_with_search_policy(foreign, policy_fixture()).is_err()
        );
    }

    #[test]
    fn frozen_wdl_lanes_accept_both_registered_resolvers_and_reject_drift() {
        use rz_search::pals::engine::PostRepairRecheckPolicy;
        for selected in [
            PostRepairRecheckPolicy::FrozenModelWdlV2,
            PostRepairRecheckPolicy::IterativeFrozenModelWdlV2,
        ] {
            let policy = PalsSearchPolicyIdentity {
                version: selected.registration_version().unwrap().into(),
                policy: selected.registration_policy().unwrap().into(),
                search_identity: selected.search_identity().into(),
                conditions_sha256: Sha256::digest(selected.conditions().unwrap().as_bytes()).into(),
            };
            for resolver in [
                PalsResolverIdentity::registered(),
                PalsResolverIdentity::registered_model_wdl(),
            ] {
                let actual = ProcessWorkJournal::new_pals_with_search_policy(
                    resolver.clone(),
                    policy.clone(),
                )
                .unwrap()
                .snapshot()
                .unwrap();
                assert_eq!(actual.pals_resolver, Some(resolver.clone()));
                assert_eq!(actual.pals_search_policy, Some(policy.clone()));
                let mut drifted = resolver;
                drifted.semantics_sha256[0] ^= 1;
                assert!(
                    ProcessWorkJournal::new_pals_with_search_policy(drifted, policy.clone())
                        .is_err()
                );
            }
        }
    }

    #[test]
    #[cfg(feature = "search-work-receipts")]
    fn legacy_marker_is_omitted_and_selected_wire_is_closed_and_validated() {
        let legacy = ProcessWorkJournal::new(SearchKind::Pals)
            .snapshot()
            .unwrap();
        let explicit = ProcessWorkJournal::new_pals(PalsResolverIdentity::registered())
            .snapshot()
            .unwrap();
        assert_eq!(
            serde_json::to_vec(&legacy).unwrap(),
            serde_json::to_vec(&explicit).unwrap()
        );
        assert!(
            serde_json::to_value(&legacy)
                .unwrap()
                .get("pals_search_policy")
                .is_none()
        );
        let policy = policy_fixture();
        let encoded = serde_json::to_value(&policy).unwrap();
        assert_eq!(
            serde_json::from_value::<PalsSearchPolicyIdentity>(encoded.clone()).unwrap(),
            policy
        );
        for field in [
            "extra",
            "version",
            "policy",
            "search_identity",
            "conditions_sha256",
        ] {
            let mut wrong = encoded.clone();
            wrong[field] = serde_json::Value::Null;
            assert!(
                serde_json::from_value::<PalsSearchPolicyIdentity>(wrong).is_err(),
                "{field}"
            );
        }
        let mut wrong_lane = encoded.clone();
        wrong_lane["version"] = "pals".into();
        assert!(serde_json::from_value::<PalsSearchPolicyIdentity>(wrong_lane).is_err());
        let raw = serde_json::to_string(&policy).unwrap();
        let duplicate = format!("{{\"version\":\"{}\",{}", policy.version, &raw[1..]);
        assert!(serde_json::from_str::<PalsSearchPolicyIdentity>(&duplicate).is_err());
    }

    #[test]
    fn foreign_work_and_model_values_keep_the_selected_resolver_and_unknown_scope() {
        let resolver = PalsResolverIdentity::from_semantics(
            rz_search::pals::value::MODEL_WDL_RESOLVER_VERSION,
            rz_search::pals::value::MODEL_WDL_RESOLVER_SEMANTICS,
        );
        let journal = ProcessWorkJournal::new_pals(resolver.clone());
        let context = context();
        for counters in [
            PalsCounters {
                value_calls: 3,
                completed_value_calls: 2,
                accepted_value_outputs: 1,
                external_checker_tasks: 2,
                external_checker_reports: 1,
                external_checker_nodes_observed: 7,
                external_checker_node_budget_reserved: 40,
                consumed_external_checker_tasks: 1,
                external_checker_work_incomplete: true,
                ..PalsCounters::default()
            },
            PalsCounters {
                value_calls: 1,
                completed_value_calls: 1,
                external_checker_tasks: 1,
                external_checker_reports: 1,
                external_checker_nodes_observed: 3,
                external_checker_node_budget_reserved: 20,
                ..PalsCounters::default()
            },
        ] {
            journal.begin().unwrap();
            journal
                .finish(
                    &context,
                    &Err(SearchSessionFailure::debug("CheckerRejected", &"fixture")),
                    AttemptObservation::Pals {
                        counters,
                        root_coverage_observed: false,
                    },
                )
                .unwrap();
        }
        let receipt = journal.snapshot().unwrap();
        assert_eq!(receipt.pals_resolver, Some(resolver));
        let totals = receipt.pals.unwrap();
        assert_eq!(totals.value_calls, Some(4));
        assert_eq!(totals.completed_value_calls, Some(3));
        assert_eq!(totals.accepted_value_outputs, Some(1));
        assert_eq!(totals.external_checker_tasks, Some(3));
        assert_eq!(totals.external_checker_reports, Some(2));
        assert_eq!(totals.external_checker_nodes_observed, Some(10));
        assert_eq!(totals.external_checker_node_budget_reserved, Some(60));
        assert_eq!(totals.consumed_external_checker_tasks, Some(1));
        assert_eq!(totals.external_checker_work_incomplete, Some(true));
        assert_eq!(totals.cpu_nodes, Some(0));
        assert_eq!(totals.completed_cpu_tasks, Some(0));
        assert_eq!(totals.unknown_root_children, None);

        // Losing the entire attempt ledger also loses whether foreign work was
        // completely observed. Existing sums cannot describe an unknown attempt.
        journal.begin().unwrap();
        journal
            .finish(
                &context,
                &Err(SearchSessionFailure::physical_completion_unknown()),
                AttemptObservation::PalsStarted,
            )
            .unwrap();
        let totals = journal.snapshot().unwrap().pals.unwrap();
        assert_eq!(totals.external_checker_nodes_observed, None);
        assert_eq!(totals.external_checker_work_incomplete, None);
        assert_eq!(totals.value_calls, None);
    }
    #[test]
    fn failed_and_canceled_calls_keep_real_work_and_distinguish_root_unknown() {
        let journal = ProcessWorkJournal::new(SearchKind::Pals);
        let context = context();
        journal.begin().unwrap();
        let counters = PalsCounters {
            proposer_calls: 2,
            completed_proposer_calls: 1,
            accepted_proposer_outputs: 1,
            role_calls: 2,
            consumed_role_outputs: 1,
            cpu_tasks_requested: 1,
            cpu_tasks: 1,
            cpu_nodes: 4,
            partial_cpu_iterations: 1,
            consumed_partial_cpu_values: 1,
            retained_situations: 3,
            ..PalsCounters::default()
        };
        journal
            .finish(
                &context,
                &Err(SearchSessionFailure::physical_completion_unknown()),
                AttemptObservation::Pals {
                    counters,
                    root_coverage_observed: false,
                },
            )
            .unwrap();
        journal.begin().unwrap();
        context.control.cancel();
        journal
            .finish(
                &context,
                &Ok(SearchSessionReport {
                    mate_strategy: None,
                    best_move: None,
                    work: DriverWork::Pals {
                        rounds: 0,
                        cpu_nodes: 0,
                        completed_tasks: 0,
                        consumed_role_outputs: 0,
                        retained_situations: 0,
                    },
                }),
                AttemptObservation::NoWork,
            )
            .unwrap();
        let receipt = journal.snapshot().unwrap();
        assert_eq!(
            (
                receipt.go_invocations,
                receipt.successful_returns,
                receipt.failed_returns
            ),
            (2, 1, 1)
        );
        assert_eq!(
            (
                receipt.active_invocations,
                receipt.canceled_returns,
                receipt.physical_unknown_returns
            ),
            (0, 1, 1)
        );
        assert_eq!(receipt.unobserved_work_invocations, 0);
        assert_eq!(
            receipt.pals_resolver.as_ref(),
            Some(&PalsResolverIdentity::registered())
        );
        let totals = receipt.pals.unwrap();
        assert_eq!(totals.cpu_nodes, Some(4));
        assert_eq!(totals.partial_cpu_iterations, Some(1));
        assert_eq!(totals.completed_cpu_tasks, Some(0));
        assert_eq!(totals.proposer_calls, Some(2));
        assert_eq!(totals.accepted_proposer_outputs, Some(1));
        assert_eq!(totals.unknown_root_children, None);
        assert_eq!(totals.retained_situations_peak, Some(3));
    }
    #[test]
    fn missing_failed_cpu_observation_is_unknown_not_zero() {
        let journal = ProcessWorkJournal::new(SearchKind::Cpu);
        let context = context();
        journal.begin().unwrap();
        journal
            .finish(
                &context,
                &Err(SearchSessionFailure::physical_completion_unknown()),
                AttemptObservation::CpuStarted,
            )
            .unwrap();
        let receipt = journal.snapshot().unwrap();
        assert_eq!(receipt.unobserved_work_invocations, 1);
        assert!(receipt.pals_resolver.is_none());
        #[cfg(feature = "search-work-receipts")]
        assert!(
            serde_json::to_value(&receipt)
                .unwrap()
                .get("pals_resolver")
                .is_none()
        );
        let totals = receipt.cpu.unwrap();
        assert_eq!(totals.tasks_requested, Some(1));
        assert_eq!(totals.reports_returned, Some(0));
        assert_eq!(totals.nodes, None);
        assert_eq!(totals.quiescence_nodes, None);
    }
    #[test]
    fn failed_cpu_work_is_retained_without_inventing_a_completed_report() {
        for (work, unknown, depth) in [
            (
                CpuWork {
                    nodes: 3,
                    quiescence_nodes: 1,
                    tt_hits: 2,
                },
                1,
                None,
            ),
            (CpuWork::default(), 0, Some(0)),
        ] {
            let journal = ProcessWorkJournal::new(SearchKind::Cpu);
            let context = context();
            journal.begin().unwrap();
            journal
                .finish(
                    &context,
                    &Err(SearchSessionFailure::debug(
                        "CpuSearch",
                        &"observed failure",
                    )),
                    AttemptObservation::CpuFailed(work),
                )
                .unwrap();
            let receipt = journal.snapshot().unwrap();
            assert_eq!(receipt.unobserved_work_invocations, unknown);
            let totals = receipt.cpu.unwrap();
            assert_eq!(totals.tasks_requested, Some(1));
            assert_eq!(totals.reports_returned, Some(0));
            assert_eq!(totals.nodes, Some(work.nodes));
            assert_eq!(totals.quiescence_nodes, Some(work.quiescence_nodes));
            assert_eq!(totals.tt_hits, Some(work.tt_hits));
            assert_eq!(totals.max_completed_depth, depth);
            assert_eq!(totals.completed_reports_accepted_for_uci_output, Some(0));
        }
    }
    #[test]
    fn cpu_completion_is_consumed_only_at_the_exact_natural_output_handoff() {
        let journal = ProcessWorkJournal::new(SearchKind::Cpu);
        let context = context();
        let movement = BoardMove::from_uci("e2e4").unwrap();
        journal.begin().unwrap();
        let report = Ok(SearchSessionReport {
            mate_strategy: None,
            best_move: Some(movement),
            work: DriverWork::Cpu {
                nodes: 1,
                completed_depth: 2,
            },
        });
        journal
            .finish(
                &context,
                &report,
                AttemptObservation::Cpu(CpuObservation {
                    nodes: 1,
                    quiescence_nodes: 0,
                    tt_hits: 0,
                    completed_depth: 2,
                    requested_depth: 2,
                    score_scope: CpuScoreScope::CompletedIteration,
                    best_move: Some(movement),
                }),
            )
            .unwrap();
        let mut stale = context.authority;
        stale.root = RootGeneration(99);
        journal.accept_cpu_output(stale, "e2e4", true).unwrap();
        assert_eq!(
            journal
                .snapshot()
                .unwrap()
                .cpu
                .unwrap()
                .completed_reports_accepted_for_uci_output,
            Some(0)
        );
        journal
            .accept_cpu_output(context.authority, "e2e4", true)
            .unwrap();
        journal
            .accept_cpu_output(context.authority, "e2e4", true)
            .unwrap();
        let totals = journal.snapshot().unwrap().cpu.unwrap();
        assert_eq!(totals.requested_depth_completed, Some(1));
        assert_eq!(totals.completed_reports_accepted_for_uci_output, Some(1));
    }
}
