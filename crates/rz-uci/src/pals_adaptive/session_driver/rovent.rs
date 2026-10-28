//! Opt-in Rovent PCR root admission over the shared physical session owner.
//! Legacy V5 construction does not install this policy or emit these records.

use super::*;
use crate::pals_adaptive::AdaptiveSearchDriver as _;
use crate::pals_adaptive::{
    AdaptiveManagerCounters, AdaptiveNativeAccountingSnapshot, RoventPeerFollowupPolicy,
    RoventReciprocalPolicy,
};
use crate::search_driver::{DriverPhysicalCompletion, DriverWork};
use rz_contracts::rovent::ExecutionMode;
use rz_search::pals::engine::adaptive_port::{AdaptiveFrontierStartup, AdaptiveModelNamespace};
use rz_search::pals::rovent_engine::RootMoveDomain;
use rz_search::pals::rovent_frontier::{
    FrontierError, FrontierGoalStatus, FrontierLimits, FrontierRegistration, FrontierSnapshot,
    FrontierStop,
};
use std::num::NonZeroU16;

pub const ROVENT_NATIVE_ROOT_DOMAIN: &str = "rz-rovent-native-root/1";
#[cfg(feature = "pals-adaptive")]
pub const ROVENT_NATIVE_ROOT_V2_DOMAIN: &str = "rz-rovent-native-root/2";
#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
pub const ROVENT_PROFILED_EXTERNAL_ROOT_DOMAIN: &str = "rz-rovent-profiled-external-root/1";
#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
const ROVENT_PROFILED_EXTERNAL_ROOT_V2_DOMAIN: &str = "rz-rovent-profiled-external-root/2";

/// Admitted alongside the actual native/CPU sources, never inferred from a
/// SearchKind label. PC uses its distinct typed role-only owner instead.
#[derive(Clone)]
pub struct RoventPcrSessionProfile {
    pub mode: ExecutionMode,
    pub maximum_goal_plies: NonZeroU16,
    pub input_lock: Digest,
    pub root_budget: Digest,
    pub selective_policy: Digest,
    pub model_namespace: Arc<AdaptiveModelNamespace>,
    pub frontier_limits: FrontierLimits,
    pub reciprocal: RoventReciprocalPolicy,
    pub peers: RoventPeerFollowupPolicy,
}
impl RoventPcrSessionProfile {
    fn validate<B>(&self, sources: &AdaptiveSessionSources<B>) -> Result<(), SearchSessionFailure>
    where
        B: TicketNativeBackend<ReviewView, AdaptivePortQuestion>,
    {
        if !self.mode.has_native_r()
            || self.maximum_goal_plies != self.frontier_limits.maximum_goal_plies
            || self.reciprocal.mode != self.mode
            || self.reciprocal.root_budget != self.root_budget
            || self.input_lock == Digest([0; 32])
            || self.root_budget == Digest([0; 32])
            || self.selective_policy == Digest([0; 32])
            || sources.external.is_some()
            || !matches!(
                sources.engine.checker_identity(),
                rz_search::cpu_checker::CheckerIdentity::Owned(_)
            )
        {
            return Err(failure(
                "RoventPcrAssembly",
                "invalid explicit own-R PCR native root profile",
            ));
        }
        self.model_namespace
            .verify_identity(
                sources.engine.model_identity(),
                sources.engine.model_value_identity(),
            )
            .map_err(|error| SearchSessionFailure::debug("RoventModelNamespace", &error))
    }
    pub(super) fn attach<B>(
        &self,
        manager: &mut SessionManager<B>,
        context: &SearchSessionContext,
        clock: &ProcessClock,
        collection_origin: Option<&crate::pals_adaptive::RoventNativeCollectionOrigin>,
        admission: &RoventPcrRootAdmission,
    ) -> Result<(), AdaptiveManagerError>
    where
        B: TicketNativeBackend<ReviewView, AdaptivePortQuestion>,
        B::Output: ProposerRankingOutput,
    {
        context.goal.validate(self.maximum_goal_plies)?;
        if !context.accepts()? {
            return Err(AdaptiveManagerError::Invalid(
                "Rovent root revoked before frontier admission",
            ));
        }
        let ledger = manager.cpu_ledger_identity();
        let port = manager
            .driver()
            .port()
            .ok_or(AdaptiveManagerError::AuxiliaryPending)?;
        let root_rules = port.root_rules();
        let actual_namespace = port
            .model_namespace()
            .map_err(|error| AdaptiveManagerError::Semantic(error.to_string()))?;
        if actual_namespace != self.model_namespace.as_ref() {
            return Err(AdaptiveManagerError::Invalid(
                "Rovent startup/port native namespaces differ",
            ));
        }
        let root_moves = Arc::new(
            RootMoveDomain::new(&root_rules, context.root_moves.as_deref())
                .map_err(|error| AdaptiveManagerError::Semantic(error.to_string()))?,
        );
        let registration = FrontierRegistration {
            authority: context.authority,
            mode: self.mode,
            original_deadline: clock.deadline(context.control.deadline)?,
            root_budget: admission.budget,
            goal: context.goal,
            nodes: context.requested_nodes,
            selective_policy: self.selective_policy,
            root_rules,
            root_moves: Arc::clone(&root_moves),
            root_state: port.root_state_id(),
            model_namespace: Arc::clone(&self.model_namespace),
            native_ledger: Some(ledger.clone()),
        };
        manager.driver_mut().attach_rovent_frontier(
            registration,
            self.frontier_limits.clone(),
            AdaptiveFrontierStartup {
                mode: self.mode,
                root_budget: admission.budget,
                root_moves: &root_moves,
                native_ledger: Some(&ledger),
            },
            clock.domain(),
            clock.now()?,
        )?;
        manager.configure_rovent_reciprocal(RoventReciprocalPolicy {
            root_budget: admission.budget,
            ..self.reciprocal
        })?;
        manager.configure_rovent_peer_followups(self.peers)?;
        if let Some(origin) = collection_origin {
            #[cfg(not(all(feature = "pals-adaptive", feature = "onnx-cpu")))]
            {
                let _ = origin;
                return Err(AdaptiveManagerError::Invalid(
                    "actual native collection is unavailable in this build",
                ));
            }
            #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
            {
                let frontier = manager
                    .driver()
                    .port()
                    .and_then(|port| port.rovent_frontier())
                    .ok_or(AdaptiveManagerError::Invalid(
                        "actual Rovent collection frontier absent",
                    ))?;
                let admission =
                    crate::pals_adaptive::RoventCollectionRootAdmission::from_attached_frontier(
                        origin,
                        frontier,
                        self.input_lock,
                        admission.budget,
                        self.selective_policy,
                        Arc::clone(admission.original.as_ref().ok_or(
                            AdaptiveManagerError::Invalid(
                                "collection requires the original per-go root capability",
                            ),
                        )?),
                    )?;
                manager.admit_collection_root(admission)?;
            }
        }
        Ok(())
    }
}

/// Scope-specific copied observations from the real root. This is distinct
/// from legacy process-work attestation and does not certify unobserved heads.
#[derive(Clone, Debug)]
pub struct RoventNativeRootReceipt {
    pub domain: &'static str,
    pub authority: SearchAuthority,
    pub mode: ExecutionMode,
    pub input_lock: Digest,
    pub root_budget: Digest,
    #[cfg(feature = "pals-adaptive")]
    pub root_admission: Option<Arc<crate::rovent_root_admission::RoventRootAdmission>>,
    pub frontier: Option<FrontierSnapshot>,
    pub counters: AdaptiveManagerCounters,
    pub native_accounting: Option<AdaptiveNativeAccountingSnapshot>,
    pub native_reservations: crate::pals_adaptive::RoventNativeAccountingObservation,
    pub physical: Option<AdaptiveShutdownReceipt>,
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    pub profiled_physical: Option<crate::pals_adaptive::ProfiledExternalShutdownReceipt>,
}

/// Actual /2 capability and root observations. The physical observations are
/// sampled before the manager's owner return and are not the final root or
/// process fence; missing evidence never certifies process completion.
#[cfg(feature = "pals-adaptive")]
#[derive(Clone, Debug)]
pub struct RoventNativeRootReceiptV2 {
    pub domain: &'static str,
    pub authority: SearchAuthority,
    pub mode: ExecutionMode,
    pub input_lock: Digest,
    pub root_budget: Digest,
    pub root_admission: Arc<crate::rovent_root_admission::source_seal::RoventRootAdmissionV2>,
    pub frontier: Option<FrontierSnapshot>,
    pub counters: AdaptiveManagerCounters,
    pub native_accounting: Option<AdaptiveNativeAccountingSnapshot>,
    pub native_reservations: crate::pals_adaptive::RoventNativeAccountingObservation,
    pub physical: Option<AdaptiveShutdownReceipt>,
    #[cfg(feature = "onnx-cpu")]
    pub profiled_physical: Option<crate::pals_adaptive::ProfiledExternalShutdownReceipt>,
}

pub(super) enum RoventRootObservation {
    V1(Arc<RoventNativeRootReceipt>),
    #[cfg(feature = "pals-adaptive")]
    V2(Arc<RoventNativeRootReceiptV2>),
}
impl RoventRootObservation {
    pub(super) fn frontier(&self) -> Option<&FrontierSnapshot> {
        match self {
            Self::V1(receipt) => receipt.frontier.as_ref(),
            #[cfg(feature = "pals-adaptive")]
            Self::V2(receipt) => receipt.frontier.as_ref(),
        }
    }
    pub(super) fn work(&self) -> Result<DriverWork, SearchSessionFailure> {
        match self {
            Self::V1(receipt) => receipt.work(),
            #[cfg(feature = "pals-adaptive")]
            Self::V2(receipt) => receipt.work(),
        }
    }
}
impl RoventNativeRootReceipt {
    pub fn work(&self) -> Result<DriverWork, SearchSessionFailure> {
        root_work(self.frontier.as_ref(), &self.counters)
    }
}
#[cfg(feature = "pals-adaptive")]
impl RoventNativeRootReceiptV2 {
    pub fn work(&self) -> Result<DriverWork, SearchSessionFailure> {
        root_work(self.frontier.as_ref(), &self.counters)
    }
}

fn root_work(
    frontier: Option<&FrontierSnapshot>,
    counters: &AdaptiveManagerCounters,
) -> Result<DriverWork, SearchSessionFailure> {
    let frontier = frontier.ok_or_else(|| {
        failure(
            "RoventFrontierMissing",
            "actual root never admitted its goal controller",
        )
    })?;
    let visits = if frontier.work.native_usage_known {
        Some(frontier.work.total_node_visits().ok_or_else(|| {
            failure(
                "RoventWorkOverflow",
                "actual node observations overflow the registered domain",
            )
        })?)
    } else {
        None
    };
    Ok(DriverWork::RoventNative {
        root_entries: frontier.work.root_entries,
        total_node_visits: visits,
        checked_transitions: frontier.work.checked_transitions,
        native_nodes: frontier
            .work
            .native_usage_known
            .then_some(frontier.work.native_nodes),
        native_qsearch_nodes: frontier
            .work
            .native_usage_known
            .then_some(frontier.work.native_qsearch_nodes),
        submitted_role_tickets: counters.submitted_role_tickets,
        consumed_role_outputs: counters.consumed_role_outputs,
        completed_nn_inputs: (counters.nn_input_count_unknown == 0)
            .then_some(counters.completed_nn_inputs),
        consumed_nn_inputs: (counters.consumed_nn_input_count_unknown == 0)
            .then_some(counters.consumed_nn_inputs),
        completed_depth: frontier.horizon.completed_depth.map(|depth| depth.0),
        maximum_examined_ply: frontier.horizon.maximum_examined_ply.0,
    })
}

impl<B> AdaptiveSessionDriver<B>
where
    B: TicketNativeBackend<ReviewView, AdaptivePortQuestion>,
    B::Output: ProposerRankingOutput,
{
    pub(super) fn rovent_root_legal(
        &self,
        position: &Position,
        context: &SearchSessionContext,
    ) -> Result<Option<Vec<BoardMove>>, SearchSessionFailure> {
        let Some(profile) = &self.rovent else {
            return Ok(None);
        };
        context
            .goal
            .validate(profile.maximum_goal_plies)
            .map_err(|error| SearchSessionFailure::debug("RoventSearchGoal", &error))?;
        let legal = position.legal_moves();
        let Some(requested) = &context.root_moves else {
            return Ok(Some(legal));
        };
        if requested.is_empty() || requested.iter().any(|movement| !legal.contains(movement)) {
            return Err(failure(
                "RoventRootRestriction",
                "invalid original root move restriction",
            ));
        }
        Ok(Some(
            legal
                .into_iter()
                .filter(|movement| requested.contains(movement))
                .collect(),
        ))
    }
    /// Reuse actual CPU/TT, native and archive lifecycle ownership. Selecting
    /// this constructor additionally admits the new root policy, not a label.
    #[allow(clippy::too_many_arguments, clippy::result_large_err)]
    pub fn new_rovent_pcr_with_resource_limits(
        options: AdaptiveSessionOptions,
        limits: AdaptiveExecutionLimits,
        sources: AdaptiveSessionSources<B>,
        clock: ProcessClock,
        issuer: AdaptiveRulesOwnerIssuer,
        registration: PalsCheckerRegistration,
        profile: RoventPcrSessionProfile,
    ) -> Result<Self, AdaptiveSessionAdmissionRejection<B>> {
        Self::admit_rovent_pcr_with_collection(
            options,
            limits,
            sources,
            clock,
            issuer,
            registration,
            profile,
            None,
            AdaptiveSessionCollectionHooks::default(),
        )
    }
    /// 원 PreparedRoventNative의 loaded identity를 보존한 수집 caller만
    /// 기존 root admission/closure hooks를 설치한다. 가중치 사본은 만들지 않는다.
    #[allow(clippy::too_many_arguments, clippy::result_large_err)]
    pub fn new_rovent_pcr_with_resource_limits_and_collection_hooks(
        options: AdaptiveSessionOptions,
        limits: AdaptiveExecutionLimits,
        sources: AdaptiveSessionSources<B>,
        clock: ProcessClock,
        issuer: AdaptiveRulesOwnerIssuer,
        registration: PalsCheckerRegistration,
        profile: RoventPcrSessionProfile,
        origin: crate::pals_adaptive::RoventNativeCollectionOrigin,
        hooks: AdaptiveSessionCollectionHooks<B>,
    ) -> Result<Self, AdaptiveSessionAdmissionRejection<B>> {
        Self::admit_rovent_pcr_with_collection(
            options,
            limits,
            sources,
            clock,
            issuer,
            registration,
            profile,
            Some(origin),
            hooks,
        )
    }
    #[allow(clippy::too_many_arguments, clippy::result_large_err)]
    fn admit_rovent_pcr_with_collection(
        options: AdaptiveSessionOptions,
        limits: AdaptiveExecutionLimits,
        sources: AdaptiveSessionSources<B>,
        clock: ProcessClock,
        issuer: AdaptiveRulesOwnerIssuer,
        registration: PalsCheckerRegistration,
        profile: RoventPcrSessionProfile,
        origin: Option<crate::pals_adaptive::RoventNativeCollectionOrigin>,
        hooks: AdaptiveSessionCollectionHooks<B>,
    ) -> Result<Self, AdaptiveSessionAdmissionRejection<B>> {
        if let Err(failure) = profile.validate(&sources) {
            return Err(AdaptiveSessionAdmissionRejection { failure, sources });
        }
        if let Some(origin) = &origin {
            if let Err(error) =
                origin.verify_profile(profile.mode, profile.input_lock, &profile.model_namespace)
            {
                return Err(AdaptiveSessionAdmissionRejection {
                    failure: SearchSessionFailure::debug("RoventCollectionOrigin", &error),
                    sources,
                });
            }
        }
        let mut admitted = Self::admit_with_resource_limits(
            options,
            limits,
            sources,
            clock,
            issuer,
            registration,
            None,
            None,
            hooks,
            true,
        )?;
        admitted.rovent = Some(profile);
        admitted.rovent_collection_origin = origin;
        Ok(admitted)
    }
    pub fn latest_rovent_root(
        &self,
    ) -> Result<Option<Arc<RoventNativeRootReceipt>>, ContractError> {
        self.latest_rovent_root
            .lock()
            .map(|receipt| receipt.clone())
            .map_err(|_| poisoned("Rovent root observation mutex poisoned"))
    }
    #[cfg(feature = "pals-adaptive")]
    pub fn latest_rovent_root_v2(
        &self,
    ) -> Result<Option<Arc<RoventNativeRootReceiptV2>>, ContractError> {
        self.latest_rovent_root_v2
            .lock()
            .map(|receipt| receipt.clone())
            .map_err(|_| poisoned("Rovent V2 root observation mutex poisoned"))
    }
    pub(super) fn observe_rovent_root(
        &self,
        manager: &SessionManager<B>,
        context: &SearchSessionContext,
        admission: Option<&RoventPcrRootAdmission>,
    ) -> Result<Option<RoventRootObservation>, SearchSessionFailure> {
        let Some(profile) = &self.rovent else {
            return Ok(None);
        };
        let admission = admission
            .ok_or_else(|| failure("RoventRootObservation", "original root admission absent"))?;
        let actual = manager
            .driver()
            .rovent_frontier_receipt()
            .map_err(|error| SearchSessionFailure::debug("RoventFrontierObservation", &error))?;
        #[cfg(feature = "pals-adaptive")]
        if let Some(original) = &admission.original_v2 {
            if admission.original.is_some() {
                return Err(failure(
                    "RoventRootObservation",
                    "multiple original root capabilities",
                ));
            }
            let receipt = Arc::new(RoventNativeRootReceiptV2 {
                domain: match self.researcher_class {
                    SessionResearcherClass::Legacy => ROVENT_NATIVE_ROOT_V2_DOMAIN,
                    #[cfg(feature = "onnx-cpu")]
                    SessionResearcherClass::ProfiledExternalV2 => {
                        ROVENT_PROFILED_EXTERNAL_ROOT_V2_DOMAIN
                    }
                    #[cfg(feature = "onnx-cpu")]
                    SessionResearcherClass::ProfiledExternal => {
                        return Err(failure(
                            "RoventRootObservation",
                            "V1 external researcher cannot own a V2 root",
                        ));
                    }
                },
                authority: context.authority,
                mode: profile.mode,
                input_lock: profile.input_lock,
                root_budget: admission.budget,
                root_admission: Arc::clone(original),
                frontier: actual.map(|receipt| receipt.snapshot),
                counters: manager.counters(),
                native_accounting: manager
                    .root_observation_snapshot()
                    .ok()
                    .map(|observation| observation.native_accounting()),
                native_reservations: manager.rovent_native_accounting(),
                physical: manager.shutdown_receipt().ok(),
                #[cfg(feature = "onnx-cpu")]
                profiled_physical: manager.profiled_shutdown_receipt().ok(),
            });
            *self.latest_rovent_root_v2.lock().map_err(|_| {
                failure(
                    "RoventRootObservation",
                    "Rovent V2 observation mutex poisoned",
                )
            })? = Some(Arc::clone(&receipt));
            return Ok(Some(RoventRootObservation::V2(receipt)));
        }
        let receipt = Arc::new(RoventNativeRootReceipt {
            domain: match self.researcher_class {
                SessionResearcherClass::Legacy => ROVENT_NATIVE_ROOT_DOMAIN,
                #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
                SessionResearcherClass::ProfiledExternal => ROVENT_PROFILED_EXTERNAL_ROOT_DOMAIN,
                #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
                SessionResearcherClass::ProfiledExternalV2 => {
                    return Err(failure(
                        "RoventRootObservation",
                        "V2 external researcher cannot own a V1 root",
                    ));
                }
            },
            authority: context.authority,
            mode: profile.mode,
            input_lock: profile.input_lock,
            root_budget: admission.budget,
            #[cfg(feature = "pals-adaptive")]
            root_admission: admission.original.clone(),
            frontier: actual.map(|receipt| receipt.snapshot),
            counters: manager.counters(),
            native_accounting: manager
                .root_observation_snapshot()
                .ok()
                .map(|observation| observation.native_accounting()),
            native_reservations: manager.rovent_native_accounting(),
            physical: manager.shutdown_receipt().ok(),
            #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
            profiled_physical: manager.profiled_shutdown_receipt().ok(),
        });
        *self
            .latest_rovent_root
            .lock()
            .map_err(|_| failure("RoventRootObservation", "Rovent observation mutex poisoned"))? =
            Some(Arc::clone(&receipt));
        Ok(Some(RoventRootObservation::V1(receipt)))
    }
    /// A normal control boundary closes admission without claiming a completed
    /// horizon. Only the original typed public-record bound joins control
    /// stops; other storage/physical failures remain failures.
    pub(super) fn rovent_should_quiesce(
        &self,
        manager: &SessionManager<B>,
    ) -> Result<bool, AdaptiveManagerError> {
        if self.rovent.is_none() {
            return Ok(false);
        }
        let Some(receipt) = manager.driver().rovent_frontier_receipt()? else {
            return Err(AdaptiveManagerError::Invalid("Rovent frontier absent"));
        };
        match receipt.snapshot.goal {
            FrontierGoalStatus::Pending => Ok(false),
            FrontierGoalStatus::DepthSatisfied
            | FrontierGoalStatus::MateSatisfied
            | FrontierGoalStatus::SelectiveFrontier
            | FrontierGoalStatus::GoalUnresolved
            | FrontierGoalStatus::Stopped(
                FrontierStop::Canceled
                | FrontierStop::Deadline
                | FrontierStop::WorkBudget
                | FrontierStop::RoleBudget,
            ) => Ok(true),
            FrontierGoalStatus::Stopped(FrontierStop::NativeUnknown) => {
                Err(AdaptiveManagerError::PhysicalUnknown)
            }
            FrontierGoalStatus::Stopped(FrontierStop::Capacity) => {
                if manager.driver().rovent_publication_capacity_closed()? {
                    Ok(true)
                } else {
                    Err(AdaptiveManagerError::Frontier(FrontierError::Capacity(
                        "Rovent frontier stopped",
                    )))
                }
            }
            FrontierGoalStatus::Stopped(FrontierStop::EvidenceReclaimed) => Err(
                AdaptiveManagerError::Frontier(FrontierError::StoreUnavailable),
            ),
            FrontierGoalStatus::Stopped(FrontierStop::Failed) => Err(
                AdaptiveManagerError::Frontier(FrontierError::Invalid("Rovent frontier failed")),
            ),
        }
    }
    /// Only the matching original controller state can turn an initial Rules
    /// scan's typed control stop into normal quiescence. A similar-looking
    /// backend or ownership error does not establish this boundary.
    pub(super) fn rovent_controlled_scan_stop(
        &self,
        manager: &SessionManager<B>,
        error: &AdaptiveManagerError,
    ) -> Result<bool, AdaptiveManagerError> {
        if self.rovent.is_none() {
            return Ok(false);
        }
        let Some(receipt) = manager.driver().rovent_frontier_receipt()? else {
            return Ok(false);
        };
        Ok(matches!(
            (error, receipt.snapshot.goal),
            (
                AdaptiveManagerError::Frontier(FrontierError::NodeBudget),
                FrontierGoalStatus::Stopped(FrontierStop::WorkBudget)
            ) | (
                AdaptiveManagerError::Frontier(FrontierError::Deadline),
                FrontierGoalStatus::Stopped(FrontierStop::Deadline)
            ) | (
                AdaptiveManagerError::Frontier(FrontierError::Revoked),
                FrontierGoalStatus::Stopped(FrontierStop::Canceled)
            )
        ))
    }
    pub(super) fn finish_rovent_session(
        &self,
        until: Instant,
    ) -> Result<Option<DriverPhysicalCompletion>, SearchSessionFailure> {
        if self.rovent.is_none() {
            return Ok(None);
        }
        let actual = self.finish_process(until)?;
        if actual.permits_process_close() {
            Ok(Some(DriverPhysicalCompletion::Confirmed))
        } else {
            Err(SearchSessionFailure::physical_completion_unknown())
        }
    }
}
