//! Typed adapter around the shared bounded single-root Collection /2 worker.
//! This layer does not combine P-C and Own manager or handoff authority.
use super::collection_hooks_core_v2::{
    CollectionBridgeCoreV2, CollectionWorkerCoreV2, CoreAdmissionFailure,
};
use super::teacher::v2_session::{RoventTeacherRosterEntryV2, TeacherSessionRetirementSealV2};
use super::{ArenaError, RoventCollectionOwnerLockV2};
use rz_experiments::rovent_corpus_lock_v2::record::RoventCorpusManifestLockV2;
use rz_uci::pals_adaptive::{AdaptiveCollectionBoundaryV2, AdaptiveCollectionError,
    AdaptiveCollectionStage, AdaptiveManagerError, AdaptiveShutdownReceipt,
    RoventCollectionRootAdmissionV2, RoventNoCaptureRootCapabilityV2};
use rz_uci::rovent_bootstrap::RoventStartupControl;
use rz_uci::rovent_bootstrap::nnuec_session::{
    RoventNnuecCollectionAdmissionFailureV2, RoventNnuecCollectionAdmissionPortV2,
    RoventNnuecCollectionAdmissionOutcomeV2, RoventNnuecCollectionHandoffV2,
    RoventNnuecCollectionHooksV2,
};
use std::sync::{Arc, Mutex, mpsc};

pub(crate) use super::collection_hooks_core_v2::CollectionWorkerExitV2 as RoventNnuecCollectionWorkerExitV2;
pub(crate) type RoventNnuecCollectionWorkerV2 =
    CollectionWorkerCoreV2<RoventNnuecCollectionHandoffV2>;

pub(crate) struct RoventNnuecCollectionBridgeV2 {
    core: Arc<CollectionBridgeCoreV2<RoventNnuecCollectionHandoffV2>>,
    deferred: Mutex<DeferredCollectionHistoryV2>,
}

struct DeferredCollectionHistoryV2 {
    retained: Vec<(RoventCollectionRootAdmissionV2, AdaptiveCollectionError)>,
    summarized: u64,
    first_summarized: Option<String>,
    summarized_reasons: Vec<(AdaptiveCollectionError, u64)>,
    other_summarized_reasons: u64,
}

fn handoff_matches(
    handoff: &mut RoventNnuecCollectionHandoffV2,
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
    handoff: &mut RoventNnuecCollectionHandoffV2,
    ack: rz_uci::rovent_external_launch::RoventTeacherScopeAggregateAckV2,
) -> Result<(), rz_uci::rovent_external_launch::RoventTeacherScopeAggregateAckV2> {
    handoff.submit_teacher_aggregate_ack(ack)
}

impl RoventNnuecCollectionBridgeV2 {
    pub(crate) fn new(
        owner: RoventCollectionOwnerLockV2,
        manifest: Arc<RoventCorpusManifestLockV2>,
        roster: Vec<RoventTeacherRosterEntryV2>,
    ) -> Result<(Arc<Self>, RoventNnuecCollectionWorkerV2), ArenaError> {
        let (core, worker) =
            CollectionBridgeCoreV2::new(owner, manifest, roster, handoff_matches, handoff_submit)?;
        Ok((Arc::new(Self { core, deferred: Mutex::new(DeferredCollectionHistoryV2 {
            retained: Vec::with_capacity(256), summarized: 0, first_summarized: None,
            summarized_reasons: Vec::with_capacity(16), other_summarized_reasons: 0,
        }) }), worker))
    }

    /// Observations retain the actual original root admission; they never
    /// confer collection or physical completion.
    pub(crate) fn try_deferred_observations(&self) -> Result<Option<(Vec<(String, AdaptiveCollectionError)>, u64, Option<String>, Vec<(AdaptiveCollectionError, u64)>, u64)>, ArenaError> {
        match self.deferred.try_lock() {
            Ok(roots) => Ok(Some((roots.retained.iter().map(|(admission, reason)| (admission.raw_sha256().to_owned(), reason.clone())).collect(), roots.summarized, roots.first_summarized.clone(), roots.summarized_reasons.clone(), roots.other_summarized_reasons))),
            Err(std::sync::TryLockError::WouldBlock) => Ok(None),
            Err(std::sync::TryLockError::Poisoned(_)) => Err(ArenaError::Invalid("deferred collection root custody poisoned".into())),
        }
    }

    pub(crate) fn try_has_deferred_observations(&self) -> Result<Option<bool>, ArenaError> {
        match self.deferred.try_lock() {
            Ok(roots) => Ok(Some(!roots.retained.is_empty() || roots.summarized != 0)),
            Err(std::sync::TryLockError::WouldBlock) => Ok(None),
            Err(std::sync::TryLockError::Poisoned(_)) => Err(ArenaError::Invalid("deferred collection root custody poisoned".into())),
        }
    }

    fn defer_before_start(
        &self,
        port: &mut RoventNnuecCollectionAdmissionPortV2<'_>,
        admission: RoventCollectionRootAdmissionV2,
        reason: AdaptiveCollectionError,
    ) -> Result<RoventNnuecCollectionAdmissionOutcomeV2, RoventNnuecCollectionAdmissionFailureV2> {
        let mut deferred = match self.deferred.try_lock() {
            Ok(roots) => roots,
            Err(std::sync::TryLockError::WouldBlock) => return Ok(RoventNnuecCollectionAdmissionOutcomeV2::Pending { admission }),
            Err(std::sync::TryLockError::Poisoned(_)) => return Err(RoventNnuecCollectionAdmissionFailureV2 {
                error: AdaptiveManagerError::Invalid("deferred collection root custody poisoned"), admission, owners: Box::new(()),
            }),
        };
        let capability = match port.no_capture_capability(&admission) {
            Ok(capability) => capability,
            Err(AdaptiveManagerError::AuxiliaryPending) => return Ok(RoventNnuecCollectionAdmissionOutcomeV2::Pending { admission }),
            Err(error) => return Err(RoventNnuecCollectionAdmissionFailureV2 { error, admission, owners: Box::new(()) }),
        };
        let evidence = if deferred.retained.len() < 256 {
            Some((admission.clone(), reason.clone()))
        } else { None };
        let first_summarized = if evidence.is_none() && deferred.first_summarized.is_none() {
            Some(admission.raw_sha256().to_owned())
        } else { None };
        let new_summary_reason = if evidence.is_none()
            && deferred.summarized_reasons.len() < 16
            && !deferred.summarized_reasons.iter().any(|(original, _)| *original == reason) {
            Some(reason.clone())
        } else { None };
        match self.core.try_install_no_capture(&capability) {
            Ok(true) => {
                if let Some(evidence) = evidence {
                    deferred.retained.push(evidence);
                } else {
                    deferred.summarized = deferred.summarized.saturating_add(1);
                    if let Some(first) = first_summarized { deferred.first_summarized = Some(first); }
                    if let Some((_, count)) = deferred.summarized_reasons.iter_mut().find(|(original, _)| *original == reason) {
                        *count = count.saturating_add(1);
                    } else if let Some(reason) = new_summary_reason {
                        deferred.summarized_reasons.push((reason, 1));
                    } else {
                        deferred.other_summarized_reasons = deferred.other_summarized_reasons.saturating_add(1);
                    }
                }
                Ok(RoventNnuecCollectionAdmissionOutcomeV2::DeferredNoCapture { capability, reason })
            }
            Ok(false) => Ok(RoventNnuecCollectionAdmissionOutcomeV2::Pending { admission }),
            Err(error) => Err(RoventNnuecCollectionAdmissionFailureV2 {
                error: AdaptiveManagerError::Collection(AdaptiveCollectionError::new(
                    AdaptiveCollectionStage::Dispatch, "RoventNoCaptureMarker", error.to_string())),
                admission, owners: Box::new(error),
            }),
        }
    }
}

impl RoventNnuecCollectionHooksV2 for RoventNnuecCollectionBridgeV2 {
    fn register_startup_control(
        &self,
        control: &RoventStartupControl,
    ) -> Result<(), AdaptiveManagerError> {
        self.core.register_startup_control(control)
    }
    fn admit(
        &self,
        port: &mut RoventNnuecCollectionAdmissionPortV2<'_>,
        admission: RoventCollectionRootAdmissionV2,
    ) -> Result<RoventNnuecCollectionAdmissionOutcomeV2, RoventNnuecCollectionAdmissionFailureV2> {
        // A previous single-root bridge can no longer start work for this
        // admission. Only this known pre-install state may be deferred; an
        // in-progress reservation is not evidence that no worker was started.
        if self.core.closed_before_next_admission(&admission) {
            let reason = AdaptiveCollectionError::new(AdaptiveCollectionStage::Dispatch,
                "RoventCollectionPreviousRootRetained", "prior single-root collector remains closed; this root has no collection");
            return self.defer_before_start(port, admission, reason);
        }
        self.core
            .admit(admission, |admission, owner| {
                port.install(admission, owner.observer())
                    .map_err(|original| CoreAdmissionFailure {
                        error: original.error.clone(),
                        admission: original.admission.clone(),
                        owners: Box::new(original),
                    })
            })
            .map(|()| RoventNnuecCollectionAdmissionOutcomeV2::Installed)
            .map_err(|failure| RoventNnuecCollectionAdmissionFailureV2 {
                error: failure.error,
                admission: failure.admission,
                owners: failure.owners,
            })
    }
    fn finish_deferred(
        &self,
        capability: &RoventNoCaptureRootCapabilityV2,
        fence: &AdaptiveShutdownReceipt,
    ) -> Result<AdaptiveCollectionBoundaryV2, AdaptiveManagerError> {
        self.core.try_finish_no_capture(capability, fence).map(|finished| {
            if finished { AdaptiveCollectionBoundaryV2::Ready } else { AdaptiveCollectionBoundaryV2::Busy }
        }).map_err(|error| AdaptiveManagerError::Collection(AdaptiveCollectionError::new(
            AdaptiveCollectionStage::Settled, "RoventNoCaptureFinish", error.to_string())))
    }
    fn defer_no_capture(
        &self,
        port: &mut RoventNnuecCollectionAdmissionPortV2<'_>,
        admission: RoventCollectionRootAdmissionV2,
        reason: AdaptiveCollectionError,
    ) -> Result<RoventNnuecCollectionAdmissionOutcomeV2, RoventNnuecCollectionAdmissionFailureV2> {
        // UCI invokes this only before starting any observer/teacher for the
        // same admission; the port independently verifies that precondition.
        self.defer_before_start(port, admission, reason)
    }
    fn pump(&self) -> Result<(), AdaptiveManagerError> {
        self.core.pump()
    }
    fn revoke(&self) {
        self.core.revoke();
    }
    fn handoff_sender(&self) -> mpsc::SyncSender<Arc<Mutex<RoventNnuecCollectionHandoffV2>>> {
        self.core.handoff_sender()
    }
}
