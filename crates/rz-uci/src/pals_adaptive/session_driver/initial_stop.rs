//! Normal initial control stops retain the actual idle process owners.
//!
//! This opt-in helper is not a physical fence producer. Its private custody is
//! created by the actual startup/clean-transfer paths, and the same idle mutex
//! remains held while a stopped source is acknowledged. No root registration,
//! model reset, selected view, CPU thread or ledger is made here.

use super::*;
use crate::rovent_root_admission::source_seal::{StoppedInitial, StoppedInitialV2};
use rz_runtime::contracts::ContractClock as _;

pub const ROVENT_PCR_INITIAL_DOMAIN: &str = "rz-rovent-pcr-native-initial-stop/1";
pub const ROVENT_PCR_INITIAL_DOMAIN_V2: &str = "rz-rovent-pcr-native-initial-stop/2";

// One idle-custody slot per session retains its actual previous root receipt.
// A larger variant does not add a queue or grant a new physical fence.
#[allow(clippy::large_enum_variant)]
#[derive(Clone, Debug)]
pub(super) enum RoventPcrIdleCustody {
    StartupReady,
    RetiredRoot(AdaptiveShutdownReceipt),
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    RetiredProfiledRoot(super::super::ProfiledExternalShutdownReceipt),
}

/// Producer-only evidence for a Start that did not create live root work.
/// Actual prior native/CPU evidence is retained, rather than charged to this
/// Start or replaced with a fabricated zero-work shutdown receipt.
#[derive(Clone, Debug)]
pub struct RoventPcrInitialReceipt {
    authority: SearchAuthority,
    source: Arc<StoppedInitial>,
    idle_custody: RoventPcrIdleCustody,
}

/// Producer-only /2 initial stop. Its source owns the original V2 lock and
/// the idle custody retains the actual preceding physical receipt, if any.
#[derive(Clone, Debug)]
pub struct RoventPcrInitialReceiptV2 {
    authority: SearchAuthority,
    source: Arc<StoppedInitialV2>,
    idle_custody: RoventPcrIdleCustody,
}
impl RoventPcrInitialReceiptV2 {
    pub fn domain(&self) -> &'static str {
        ROVENT_PCR_INITIAL_DOMAIN_V2
    }
    pub fn authority(&self) -> SearchAuthority {
        self.authority
    }
    pub fn source(&self) -> &Arc<StoppedInitialV2> {
        &self.source
    }
    pub fn previous_clean_root(&self) -> Option<&AdaptiveShutdownReceipt> {
        match &self.idle_custody {
            RoventPcrIdleCustody::RetiredRoot(receipt) => Some(receipt),
            _ => None,
        }
    }
    /// Previous physical external-root evidence, not evidence for this stop.
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    pub fn previous_clean_profiled_root(
        &self,
    ) -> Option<&super::super::ProfiledExternalShutdownReceipt> {
        match &self.idle_custody {
            RoventPcrIdleCustody::RetiredProfiledRoot(receipt) => Some(receipt),
            _ => None,
        }
    }
    pub fn used_original_startup_ready(&self) -> bool {
        matches!(self.idle_custody, RoventPcrIdleCustody::StartupReady)
    }
}
impl RoventPcrInitialReceipt {
    pub fn domain(&self) -> &'static str {
        ROVENT_PCR_INITIAL_DOMAIN
    }
    pub fn authority(&self) -> SearchAuthority {
        self.authority
    }
    pub fn source(&self) -> &Arc<StoppedInitial> {
        &self.source
    }
    pub fn previous_clean_root(&self) -> Option<&AdaptiveShutdownReceipt> {
        match &self.idle_custody {
            RoventPcrIdleCustody::StartupReady => None,
            RoventPcrIdleCustody::RetiredRoot(receipt) => Some(receipt),
            #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
            RoventPcrIdleCustody::RetiredProfiledRoot(_) => None,
        }
    }
    pub fn used_original_startup_ready(&self) -> bool {
        matches!(self.idle_custody, RoventPcrIdleCustody::StartupReady)
    }
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    pub fn previous_clean_profiled_root(
        &self,
    ) -> Option<&super::super::ProfiledExternalShutdownReceipt> {
        match &self.idle_custody {
            RoventPcrIdleCustody::RetiredProfiledRoot(receipt) => Some(receipt),
            _ => None,
        }
    }
    fn report(&self) -> SearchSessionReport {
        stopped_report(self.source.fallback())
    }
}

fn stopped_report(fallback: Option<BoardMove>) -> SearchSessionReport {
    SearchSessionReport {
        best_move: fallback,
        work: DriverWork::RoventNative {
            root_entries: 0,
            total_node_visits: Some(0),
            checked_transitions: 0,
            native_nodes: Some(0),
            native_qsearch_nodes: Some(0),
            submitted_role_tickets: 0,
            consumed_role_outputs: 0,
            completed_nn_inputs: Some(0),
            consumed_nn_inputs: Some(0),
            completed_depth: None,
            maximum_examined_ply: 0,
        },
        mate_strategy: None,
    }
}

impl<B> AdaptiveSessionDriver<B>
where
    B: TicketNativeBackend<ReviewView, AdaptivePortQuestion>,
    B::Output: ProposerRankingOutput,
{
    pub fn latest_rovent_initial_v2(
        &self,
    ) -> Result<Option<Arc<RoventPcrInitialReceiptV2>>, ContractError> {
        self.latest_rovent_initial_v2
            .lock()
            .map(|receipt| receipt.clone())
            .map_err(|_| poisoned("Rovent V2 initial receipt owner poisoned"))
    }

    pub fn latest_rovent_initial(
        &self,
    ) -> Result<Option<Arc<RoventPcrInitialReceipt>>, ContractError> {
        self.latest_rovent_initial
            .lock()
            .map(|receipt| receipt.clone())
            .map_err(|_| poisoned("Rovent initial receipt owner poisoned"))
    }

    /// Caller must hold the original SessionState mutex and borrow its exact
    /// Idle owners. The source contains no live authority to start new work.
    pub(super) fn finish_rovent_initial(
        &self,
        owners: &IdleOwners<B>,
        context: &SearchSessionContext,
        stopped: StoppedInitial,
    ) -> Result<SearchSessionReport, SearchSessionFailure> {
        self.validate_initial_idle(owners, self.rovent_root_policy.is_some())?;
        // No reset-request sequence or game/root metadata is consumed here.
        // Private custody plus the actual matching Ready clock/idle owner are
        // required; occupancy zero by itself never certifies physical reuse.
        let receipt = Arc::new(RoventPcrInitialReceipt {
            authority: context.authority,
            source: Arc::new(stopped),
            idle_custody: owners.rovent_idle_custody.clone(),
        });
        let report = receipt.report();
        let legal = receipt.source.allowed_root_moves().to_vec();
        *self
            .pending_output
            .lock()
            .map_err(|_| SearchSessionFailure::physical_completion_unknown())? =
            Some(PendingOutput {
                authority: context.authority,
                report: report.best_move,
                legal,
            });
        *self
            .latest_rovent_initial
            .lock()
            .map_err(|_| SearchSessionFailure::physical_completion_unknown())? = Some(receipt);
        Ok(report)
    }

    /// /2 uses the exact original owner and the same idle physical custody;
    /// the stopped source must not be reconstructed from metadata or a /1 lock.
    pub(super) fn finish_rovent_initial_v2(
        &self,
        owners: &IdleOwners<B>,
        context: &SearchSessionContext,
        stopped: StoppedInitialV2,
    ) -> Result<SearchSessionReport, SearchSessionFailure> {
        let same_owner = self
            .rovent_root_policy_v2
            .as_ref()
            .is_some_and(|policy| policy.owner.same_original_owner(stopped.owner()));
        self.validate_initial_idle(owners, same_owner)?;
        #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
        if let Some(policy) = &self.rovent_root_policy_v2 {
            let lock = policy
                .owner
                .v2()
                .ok_or_else(SearchSessionFailure::physical_completion_unknown)?;
            if matches!(
                lock.manifest.engine.researcher.as_ref(),
                Some(rz_experiments::RoventResearcher::External { .. })
            ) {
                let SessionResearcherSource::ProfiledExternalV2(source) = &owners.researcher_source
                else {
                    return Err(SearchSessionFailure::physical_completion_unknown());
                };
                let same_idle_custody = match &owners.rovent_idle_custody {
                    RoventPcrIdleCustody::StartupReady => {
                        source.ready_for_root_policy_registration()
                    }
                    RoventPcrIdleCustody::RetiredProfiledRoot(receipt) => {
                        receipt.permits_root_reuse()
                            && source
                                .retained_root_admission_v2()
                                .is_some_and(|root| root.authority() == receipt.authority())
                            && source.retained_root_source().is_some_and(|worker_source| {
                                worker_source.same_source(receipt.source())
                            })
                    }
                    RoventPcrIdleCustody::RetiredRoot(_) => false,
                };
                if !source.same_original_lock(lock)
                    || !source.permits_initial_idle()
                    || !same_idle_custody
                {
                    return Err(SearchSessionFailure::physical_completion_unknown());
                }
            }
        }
        let receipt = Arc::new(RoventPcrInitialReceiptV2 {
            authority: context.authority,
            source: Arc::new(stopped),
            idle_custody: owners.rovent_idle_custody.clone(),
        });
        let report = stopped_report(receipt.source.fallback());
        let legal = receipt.source.allowed_root_moves().to_vec();
        *self
            .pending_output
            .lock()
            .map_err(|_| SearchSessionFailure::physical_completion_unknown())? =
            Some(PendingOutput {
                authority: context.authority,
                report: report.best_move,
                legal,
            });
        *self
            .latest_rovent_initial_v2
            .lock()
            .map_err(|_| SearchSessionFailure::physical_completion_unknown())? = Some(receipt);
        Ok(report)
    }

    /// Caller holds the actual idle mutex throughout both validation and receipt
    /// publication. `registered` is checked before any state is consumed.
    pub(super) fn validate_initial_idle(
        &self,
        owners: &IdleOwners<B>,
        registered: bool,
    ) -> Result<(), SearchSessionFailure> {
        let original = &self.clock.runtime_clock();
        let ready = owners
            .native
            .ready_clock()
            .ok_or_else(SearchSessionFailure::physical_completion_unknown)?;
        if self.closed.load(Ordering::Acquire)
            || !registered
            || owners.engine.is_none()
            || owners.native_finished.is_some()
            || owners.native_joined
            || owners.native.is_quarantined()
            || owners.native.occupancy() != 0
            || ready.domain() != original.domain()
            || ready.origin() != original.origin()
            || matches!(
                &owners.rovent_idle_custody,
                RoventPcrIdleCustody::RetiredRoot(receipt) if !receipt.permits_root_reuse()
            )
            || {
                #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
                {
                    matches!(&owners.rovent_idle_custody, RoventPcrIdleCustody::RetiredProfiledRoot(receipt) if !receipt.permits_root_reuse())
                }
                #[cfg(not(all(feature = "pals-adaptive", feature = "onnx-cpu")))]
                {
                    false
                }
            }
        {
            return Err(SearchSessionFailure::physical_completion_unknown());
        }
        Ok(())
    }
}
