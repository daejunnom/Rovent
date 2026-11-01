//! Opaque in-process publication custody for an original /3 collection row.
//!
//! A decoded row or a digest cannot construct this owner. Only the actual
//! collection producer may create it after its ticket and target checks.
use rz_experiments::rovent_corpus_lock_v3::record::RoventTrainingRecordLockV3;
use rz_uci::pals_adaptive::RoventCollectionRootAdmissionV3;

pub struct PublishedRoventLockV3Record {
    pub(super) record: RoventTrainingRecordLockV3,
    pub(super) manifest_sha256: String,
    pub(super) admission: RoventCollectionRootAdmissionV3,
}

impl PublishedRoventLockV3Record {
    pub fn record(&self) -> &RoventTrainingRecordLockV3 {
        &self.record
    }

    pub fn manifest_sha256(&self) -> &str {
        &self.manifest_sha256
    }

    pub fn admission(&self) -> &RoventCollectionRootAdmissionV3 {
        &self.admission
    }
}
