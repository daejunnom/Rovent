//! Original live custody for a published input-lock /2 record.
//!
//! This type deliberately has no public constructor or deserializer. A future
//! producer must join an actually accepted and retired request with the same
//! live admission before constructing it; a wire record is not that authority.

use rz_contracts::pals_adaptive::RoleTicketEnvelope;
use rz_experiments::rovent_corpus_lock_v2::record::RoventTrainingRecordLockV2;
use rz_uci::pals_adaptive::RoventCollectionRootAdmissionV2;

/// Original accepted and search-consumed row, before target attachment and
/// publication. It is issued only by the owning producer, never by decoding.
pub struct RoventRetiredLockV2Record {
    pub(super) record: RoventTrainingRecordLockV2,
    pub(super) manifest_sha256: String,
    pub(super) manifest_owner:
        std::sync::Arc<rz_experiments::rovent_corpus_lock_v2::record::RoventCorpusManifestLockV2>,
    // Private in-process identity of the exact producer that retired the row.
    // No serialized digest can reconstruct this ownership.
    pub(super) producer_owner: std::sync::Arc<()>,
    pub(super) admission: RoventCollectionRootAdmissionV2,
    pub(super) ticket: RoleTicketEnvelope,
}

impl RoventRetiredLockV2Record {
    pub(crate) fn matches_manifest_owner(
        &self,
        manifest: &std::sync::Arc<
            rz_experiments::rovent_corpus_lock_v2::record::RoventCorpusManifestLockV2,
        >,
    ) -> bool {
        std::sync::Arc::ptr_eq(&self.manifest_owner, manifest)
    }

    pub fn record(&self) -> &RoventTrainingRecordLockV2 {
        &self.record
    }

    pub fn ticket(&self) -> &RoleTicketEnvelope {
        &self.ticket
    }

    /// A view of the originally submitted input for matching the actual
    /// teacher job. It is not a teacher result or publication capability.
    pub fn submitted_input_seal(&self) -> super::RoventSubmittedInputSeal {
        super::input_seal(
            &self.manifest_sha256,
            self.ticket,
            &self.record.0.input.common_validation_projection(),
            &self.record.0.input_sha256,
            &self.record.0.lineage,
        )
    }

    /// Reconstruct the actual query position from this row's original Rules
    /// replay. This supplies no live root or teacher execution authority.
    pub fn original_teacher_query_position(
        &self,
    ) -> Result<rz_position::Position, super::ArenaError> {
        let input = &self.record.0.input;
        input.validate()?;
        let replay: super::rules_replay::RoventSubmittedRulesReplay =
            serde_json::from_value(self.record.0.native.raw_submission["rules_replay"].clone())
                .map_err(|error| super::invalid(error.to_string()))?;
        let position = replay.original_query_position(&input.common_validation_projection())?;
        let legal = super::packed(&position.legal_moves())?;
        if matches!(
            input.selected_view.query.purpose.as_str(),
            "divergence_policy" | "task_policy"
        ) {
            super::check(
                input.legal_moves.is_empty(),
                "scalar-only query has policy candidates",
            )?;
        } else {
            super::check(
                legal.len() == input.legal_moves.len()
                    && input
                        .legal_moves
                        .iter()
                        .all(|movement| legal.contains(movement)),
                "input candidates are not the complete query Rules legal set",
            )?;
        }
        Ok(position)
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
