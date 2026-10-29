//! Original live custody for a published input-lock /2 record.
//!
//! This type deliberately has no public constructor or deserializer. A future
//! producer must join an actually accepted and retired request with the same
//! live admission before constructing it; a wire record is not that authority.

use rz_experiments::rovent_corpus_lock_v2::record::RoventTrainingRecordLockV2;
use rz_uci::pals_adaptive::RoventCollectionRootAdmissionV2;
use rz_contracts::pals_adaptive::RoleTicketEnvelope;

/// Original accepted and search-consumed row, before target attachment and
/// publication. It is issued only by the owning producer, never by decoding.
pub struct RoventRetiredLockV2Record {
    pub(super) record: RoventTrainingRecordLockV2,
    pub(super) manifest_sha256: String,
    pub(super) admission: RoventCollectionRootAdmissionV2,
    pub(super) ticket: RoleTicketEnvelope,
}

impl RoventRetiredLockV2Record {
    pub fn record(&self) -> &RoventTrainingRecordLockV2 {
        &self.record
    }

    pub fn ticket(&self) -> &RoleTicketEnvelope {
        &self.ticket
    }
}

pub struct PublishedRoventLockV2Record {
    pub(super) record: RoventTrainingRecordLockV2,
    pub(super) manifest_sha256: String,
    pub(super) admission: RoventCollectionRootAdmissionV2,
}

impl PublishedRoventLockV2Record {
    pub fn record(&self) -> &RoventTrainingRecordLockV2 {
        &self.record
    }

    pub fn manifest_sha256(&self) -> &str {
        &self.manifest_sha256
    }

    pub fn admission(&self) -> &RoventCollectionRootAdmissionV2 {
        &self.admission
    }
}
