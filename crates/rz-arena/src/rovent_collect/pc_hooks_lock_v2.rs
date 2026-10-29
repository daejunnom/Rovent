//! Typed adapter around the shared bounded single-root Collection /2 worker.
//! This layer does not combine P-C and Own manager or handoff authority.
use super::collection_hooks_core_v2::{
    CollectionBridgeCoreV2, CollectionWorkerCoreV2, CoreAdmissionFailure,
};
use super::teacher::v2_session::{RoventTeacherRosterEntryV2, TeacherSessionRetirementSealV2};
use super::{ArenaError, RoventCollectionOwnerLockV2};
use rz_experiments::rovent_corpus_lock_v2::record::RoventCorpusManifestLockV2;
use rz_uci::pals_adaptive::{AdaptiveManagerError, RoventCollectionRootAdmissionV2};
use rz_uci::rovent_bootstrap::RoventStartupControl;
use rz_uci::rovent_bootstrap::pc_session::{
    RoventPcCollectionAdmissionFailureV2, RoventPcCollectionAdmissionPortV2,
    RoventPcCollectionHandoffV2, RoventPcCollectionHooksV2,
};
use std::sync::{Arc, Mutex, mpsc};

pub(crate) use super::collection_hooks_core_v2::CollectionWorkerExitV2 as RoventPcCollectionWorkerExitV2;
pub(crate) type RoventPcCollectionWorkerV2 = CollectionWorkerCoreV2<RoventPcCollectionHandoffV2>;

pub(crate) struct RoventPcCollectionBridgeV2 {
    core: Arc<CollectionBridgeCoreV2<RoventPcCollectionHandoffV2>>,
}

fn handoff_matches(
    handoff: &mut RoventPcCollectionHandoffV2,
    expected: &RoventCollectionRootAdmissionV2,
    seal: &TeacherSessionRetirementSealV2,
) -> bool {
    let (admission, _, settlement, fence) = handoff.parts_mut();
    admission.same_original_admission(expected)
        && seal.matches_admission(admission)
        && settlement.is_ok()
        && fence.permits_semantic_closure()
}

fn handoff_submit(
    handoff: &mut RoventPcCollectionHandoffV2,
    ack: rz_uci::rovent_external_launch::RoventTeacherScopeAggregateAckV2,
) -> Result<(), rz_uci::rovent_external_launch::RoventTeacherScopeAggregateAckV2> {
    handoff.submit_teacher_aggregate_ack(ack)
}

impl RoventPcCollectionBridgeV2 {
    pub(crate) fn new(
        owner: RoventCollectionOwnerLockV2,
        manifest: Arc<RoventCorpusManifestLockV2>,
        roster: Vec<RoventTeacherRosterEntryV2>,
    ) -> Result<(Arc<Self>, RoventPcCollectionWorkerV2), ArenaError> {
        let (core, worker) =
            CollectionBridgeCoreV2::new(owner, manifest, roster, handoff_matches, handoff_submit)?;
        Ok((Arc::new(Self { core }), worker))
    }
}

impl RoventPcCollectionHooksV2 for RoventPcCollectionBridgeV2 {
    fn register_startup_control(
        &self,
        control: &RoventStartupControl,
    ) -> Result<(), AdaptiveManagerError> {
        self.core.register_startup_control(control)
    }
    fn admit(
        &self,
        port: &mut RoventPcCollectionAdmissionPortV2<'_>,
        admission: RoventCollectionRootAdmissionV2,
    ) -> Result<(), RoventPcCollectionAdmissionFailureV2> {
        self.core
            .admit(admission, |admission, owner| {
                port.install(admission, owner.observer())
                    .map_err(|original| CoreAdmissionFailure {
                        error: original.error.clone(),
                        admission: original.admission.clone(),
                        owners: Box::new(original),
                    })
            })
            .map_err(|failure| RoventPcCollectionAdmissionFailureV2 {
                error: failure.error,
                admission: failure.admission,
                owners: failure.owners,
            })
    }
    fn pump(&self) -> Result<(), AdaptiveManagerError> {
        self.core.pump()
    }
    fn revoke(&self) {
        self.core.revoke();
    }
    fn handoff_sender(&self) -> mpsc::SyncSender<Arc<Mutex<RoventPcCollectionHandoffV2>>> {
        self.core.handoff_sender()
    }
}
