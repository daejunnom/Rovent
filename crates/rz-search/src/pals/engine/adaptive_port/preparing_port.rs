//! Same-owner adoption after the real initial C preparation and R Ready.
//! The handoff, rather than caller metadata or relaxed quiescence, admits the
//! Store/Frontier/Rules root. Session cleanup custody remains explicit.
use super::*;
use crate::cpu_worker::{CpuCreditLedger, DeferredNnuecRSource, NnuecReadyRegistration};
use crate::pals::engine::preparing_root::PreparingRootHandoff;
use crate::pals::rovent_engine::RootMoveDomain;
use std::sync::atomic::AtomicBool;

pub struct AdaptivePreparingRootAdmission<M: RoleModel> {
    port: AdaptiveEnginePort<M>,
    custody: AdaptivePreparingRootCustody,
}
impl<M: RoleModel> AdaptivePreparingRootAdmission<M> {
    pub fn into_parts(self) -> (AdaptiveEnginePort<M>, AdaptivePreparingRootCustody) {
        (self.port, self.custody)
    }
}

/// Returned owners, not a declaration of physical drain or goal completion.
/// The session moves the SAME ledger into its manager and retains the actor,
/// cancellation and first-Bound pins through real replacement or shutdown.
/// External code cannot construct this transport from metadata.
// The private unit seal restricts issuance to this module; it is an ownership
// boundary, not an API promise about future public fields.
#[allow(clippy::manual_non_exhaustive)]
pub struct AdaptivePreparingRootCustody {
    pub ledger: CpuCreditLedger,
    pub root_moves: Arc<RootMoveDomain>,
    pub root_budget: ContractDigest,
    pub cleanup_deadline: Instant,
    pub original_view: Arc<ReviewView>,
    pub original_review_seal: PreparingRootViewSeal,
    pub ready: NnuecReadyRegistration,
    pub source: DeferredNnuecRSource,
    pub cancellation: Arc<AtomicBool>,
    _same_root: (),
}

impl<M: RoleModel> AdaptiveEnginePort<M> {
    /// Handoff already validated actual Ready/source/stack owner and the two
    /// exact original leases before consuming its private root nonce. No new
    /// Store, Rules owner, ledger, clock origin or Frontier is created here.
    pub fn from_preparing_handoff(
        handoff: PreparingRootHandoff<M>,
    ) -> AdaptivePreparingRootAdmission<M> {
        let parts = handoff.into_parts();
        let port = Self::from_admitted_root(
            parts.engine,
            parts.root,
            parts.rules,
            parts.authority,
            parts.deadline,
            parts.limits,
            parts.rules_owner_issuer,
            parts.started,
            parts.root_rules_scanned,
        );
        AdaptivePreparingRootAdmission {
            port,
            custody: AdaptivePreparingRootCustody {
                ledger: parts.ledger,
                root_moves: parts.root_moves,
                root_budget: parts.root_budget,
                cleanup_deadline: parts.cleanup_deadline,
                original_view: parts.original_view,
                original_review_seal: parts.original_review_seal,
                ready: parts.ready,
                source: parts.source,
                cancellation: parts.cancellation,
                _same_root: (),
            },
        }
    }
}
