//! Closure evidence for the actual profiled external source kind. There is no
//! native CPU ledger field: absence cannot be represented as a zero charge.
use super::*;
use rz_search::external_cpu::{
    ProfiledExternalCurrentRootFence, ProfiledExternalProcessCloseProof, ProfiledExternalRootSource,
};
use rz_search::pals::rovent_frontier::FrontierExternalWork;

#[derive(Clone, Debug)]
pub struct ProfiledExternalShutdownReceipt {
    pub(super) authority: SearchAuthority,
    pub(super) source: ProfiledExternalRootSource,
    pub(super) native_root_fence: Option<NativeRootFence>,
    pub(super) external_root_fence: Option<ProfiledExternalCurrentRootFence>,
    pub(super) process_close_proof: Option<ProfiledExternalProcessCloseProof>,
    pub(super) native_process_joined: bool,
    pub(super) native_physical_shutdown_confirmed: bool,
    pub(super) external_process_joined: bool,
    pub(super) auxiliary_shutdown_confirmed: bool,
    pub(super) pending_tickets: usize,
    pub(super) retained_outputs: usize,
    pub(super) quarantined: bool,
    pub(super) request_sequence_high_water: u64,
    pub(super) counters: AdaptiveManagerCounters,
    pub(super) reported_work: Option<FrontierExternalWork>,
    pub(super) retained_semantic_host_bytes: Option<u64>,
}

impl ProfiledExternalShutdownReceipt {
    pub fn authority(&self) -> SearchAuthority {
        self.authority
    }
    pub fn source_kind(&self) -> AdaptiveResearcherSourceKind {
        AdaptiveResearcherSourceKind::ProfiledExternal
    }
    pub fn source(&self) -> &ProfiledExternalRootSource {
        &self.source
    }
    pub fn native_root_fence(&self) -> Option<NativeRootFence> {
        self.native_root_fence
    }
    pub fn external_root_fence(&self) -> Option<&ProfiledExternalCurrentRootFence> {
        self.external_root_fence.as_ref()
    }
    pub fn process_close_proof(&self) -> Option<&ProfiledExternalProcessCloseProof> {
        self.process_close_proof.as_ref()
    }
    pub fn counters(&self) -> AdaptiveManagerCounters {
        self.counters
    }
    pub fn native_process_joined(&self) -> bool {
        self.native_process_joined
    }
    pub fn native_physical_shutdown_confirmed(&self) -> bool {
        self.native_physical_shutdown_confirmed
    }
    pub fn external_process_joined(&self) -> bool {
        self.external_process_joined
    }
    pub fn pending_tickets(&self) -> usize {
        self.pending_tickets
    }
    pub(crate) fn retained_outputs(&self) -> usize {
        self.retained_outputs
    }
    pub(crate) fn auxiliary_shutdown_confirmed(&self) -> bool {
        self.auxiliary_shutdown_confirmed
    }
    pub fn quarantined(&self) -> bool {
        self.quarantined
    }
    pub fn request_sequence_high_water(&self) -> u64 {
        self.request_sequence_high_water
    }
    pub fn reported_work(&self) -> Option<FrontierExternalWork> {
        self.reported_work
    }
    pub fn retained_semantic_host_bytes(&self) -> Option<u64> {
        self.retained_semantic_host_bytes
    }
    fn settled(&self) -> bool {
        !self.quarantined
            && self.pending_tickets == 0
            && self.retained_outputs == 0
            && self.auxiliary_shutdown_confirmed
            && self
                .reported_work
                .is_some_and(|work| work.outstanding_questions == 0)
    }
    pub fn permits_root_reuse(&self) -> bool {
        self.settled()
            && !self.native_process_joined
            && !self.external_process_joined
            && self.process_close_proof.is_none()
            && self
                .native_root_fence
                .is_some_and(|fence| fence.authority() == self.authority && fence.permits_reuse())
            && self.external_root_fence.as_ref().is_some_and(|fence| {
                fence.authority() == self.authority
                    && fence.source().same_source(&self.source)
                    && fence.permits_reuse()
            })
    }
    pub fn permits_archive_close(&self) -> bool {
        self.settled()
            && self.native_process_joined
            && self.native_physical_shutdown_confirmed
            && self.external_process_joined
            && self.process_close_proof.as_ref().is_some_and(|proof| {
                proof.source().same_source(&self.source)
                    && proof.authority() == self.authority
                    && proof.permits_semantic_closure()
            })
    }
    pub fn permits_semantic_closure(&self) -> bool {
        self.permits_root_reuse() || self.permits_archive_close()
    }
}

pub struct ProfiledExternalPersistentParts<Q, B, D, L = rz_experiments::RoventInputLock>
where
    B: TicketNativeBackend<ReviewView, Q>,
    L: profiled_external_source::ExternalManagerLock,
{
    pub(super) native: TicketedNativeService<ReviewView, Q, B>,
    pub(super) external: profiled_external_source::ProfiledExternalManagerSource<L>,
    pub(super) driver: D,
    pub(super) continuation: AdaptiveSequenceContinuation,
    pub(super) receipt: ProfiledExternalShutdownReceipt,
}

impl<Q, B, D, L> ProfiledExternalPersistentParts<Q, B, D, L>
where
    B: TicketNativeBackend<ReviewView, Q>,
    L: profiled_external_source::ExternalManagerLock,
{
    pub fn receipt(&self) -> &ProfiledExternalShutdownReceipt {
        &self.receipt
    }
    pub fn into_components(
        self,
    ) -> (
        TicketedNativeService<ReviewView, Q, B>,
        profiled_external_source::ProfiledExternalManagerSource<L>,
        D,
        AdaptiveSequenceContinuation,
        ProfiledExternalShutdownReceipt,
    ) {
        (
            self.native,
            self.external,
            self.driver,
            self.continuation,
            self.receipt,
        )
    }
}
