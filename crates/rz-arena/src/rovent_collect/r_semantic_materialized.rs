//! Cohort-only bounded detachment before physical pin zero. The original
//! Closure is consumed only after its complete metadata has been materialized.
//! Durable bytes remain unavailable until the same actual final-retirement
//! capability is presented. Legacy RStream/raw/semantic formats are unchanged.

use super::{
    ArenaError, CapturedRoventRRetirement, ROVENT_R_SEMANTIC_MAX_BYTES,
    ROVENT_R_UNSUBMITTED_MAX_BYTES, RoventRCollectionClosure, RoventRCorpusPublication,
    RoventRSemanticObservations, RoventRUnsubmittedObservations,
    capture_rovent_r_semantic_observations, capture_rovent_r_unsubmitted_observations, check,
    invalid,
};
use rz_contracts::pals::SearchAuthority;
use rz_search::cpu_value::nnuec::v2::capture::{NnuecCaptureEnding, NnuecCaptureLineage};
use rz_uci::pals_adaptive::{
    UtilityInterventionTrialPhysicalRetirement, UtilityTrialPhysicalRetirement,
};
use rz_uci::rovent_bootstrap::nnuec_source::LoadedRoventNnuecSource;
use rz_uci::rovent_root_admission::RoventRootAdmission;
use serde::Serialize;
use sha2::{Digest as _, Sha256};
use std::io::{self, Write};
use std::sync::{Arc, Weak};
use std::time::Instant;

pub const ROVENT_R_SEMANTIC_MATERIALIZATION_MAX_BYTES: usize = 32 << 20;

/// No model/Bound/View/Reservation owner is retained here. These are immutable
/// historical bytes and small original-instance witnesses, never a live cap.
pub struct RoventRMaterializedSemantics {
    root: Weak<RoventRootAdmission>,
    source: Weak<LoadedRoventNnuecSource>,
    authority: SearchAuthority,
    admission_sha256: String,
    input_lock_sha256: String,
    parameter_sha256: String,
    lineages: Vec<NnuecCaptureLineage>,
    created_at: Instant,
    finished_at: Instant,
    staged: MaterializedMetadata,
}

/// On failure the original owners and any complete/partial metadata buffers
/// remain reachable. The caller may quarantine them; no release ACK is issued.
pub struct RoventRSemanticMaterializationRejected {
    pub error: ArenaError,
    pub closure: RoventRCollectionClosure,
    pub captured: Vec<CapturedRoventRRetirement>,
    staged: MaterializedMetadata,
}
impl RoventRSemanticMaterializationRejected {
    pub fn retained_serialized_bytes(&self) -> usize {
        self.staged.byte_len()
    }
}

#[derive(Default)]
struct MaterializedMetadata {
    publication: Option<RoventRCorpusPublication>,
    publication_sha256: Option<String>,
    publication_bytes: Vec<u8>,
    semantic: Option<RoventRSemanticObservations>,
    semantic_bytes: Vec<u8>,
    unsubmitted: Option<RoventRUnsubmittedObservations>,
    unsubmitted_bytes: Vec<u8>,
}
impl MaterializedMetadata {
    fn byte_len(&self) -> usize {
        // Each buffer is individually bounded and the aggregate is checked
        // before append; <=32 MiB cannot overflow the supported usize range.
        self.publication_bytes.len() + self.semantic_bytes.len() + self.unsubmitted_bytes.len()
    }
}

/// The explicit bound counts retained serialized metadata, not DTO/weak-cap
/// allocation overhead or a measured host-memory peak. Raw readout packets
/// have the separate E-owned packing budget.
pub fn materialize_rovent_r_semantics_retaining(
    closure: RoventRCollectionClosure,
    captured: Vec<CapturedRoventRRetirement>,
    planned_publication: Option<&RoventRCorpusPublication>,
    planned_publication_bytes: Option<&[u8]>,
    maximum_serialized_bytes: usize,
) -> Result<RoventRMaterializedSemantics, Box<RoventRSemanticMaterializationRejected>> {
    let created_at = Instant::now();
    let mut staged = MaterializedMetadata::default();
    let prepared = (|| {
        check(
            (1..=ROVENT_R_SEMANTIC_MATERIALIZATION_MAX_BYTES).contains(&maximum_serialized_bytes)
                && closure.captures().len() <= 64
                && captured.len() <= 64
                && closure.witnesses().len() <= 256
                && planned_publication.is_some() == planned_publication_bytes.is_some()
                && planned_publication.is_some() != captured.is_empty()
                && closure
                    .nnuec_source()
                    .semantic_weights_identity()
                    .weights_sha256
                    .is_some(),
            "cohort R materialization explicit bound/publication scope",
        )?;
        check(
            closure.captures().iter().all(|original| {
                original
                    .retirement()
                    .is_none_or(|terminal| terminal.ending() != NnuecCaptureEnding::Unknown)
            }),
            "cohort R physical completion unknown; original caps retained",
        )?;
        if let (Some(publication), Some(bytes)) = (planned_publication, planned_publication_bytes) {
            publication.validate()?;
            encode_bounded(
                publication,
                &mut staged.publication_bytes,
                maximum_serialized_bytes,
            )?;
            check(
                bytes == staged.publication_bytes,
                "cohort R planned publication differs from original serialized bytes",
            )?;
            let sha256 = format!("{:x}", Sha256::digest(&staged.publication_bytes));
            staged.publication = Some(publication.clone());
            staged.publication_sha256 = Some(sha256.clone());
            staged.semantic = Some(capture_rovent_r_semantic_observations(
                &closure,
                &captured,
                publication,
                &sha256,
            )?);
            let remaining = maximum_serialized_bytes - staged.byte_len();
            encode_bounded(
                staged
                    .semantic
                    .as_ref()
                    .expect("actual semantic projection"),
                &mut staged.semantic_bytes,
                remaining.min(ROVENT_R_SEMANTIC_MAX_BYTES),
            )?;
        } else {
            check(
                closure.witnesses().is_empty()
                    && closure
                        .captures()
                        .iter()
                        .all(|held| held.retirement().is_none()),
                "cohort R original terminals or semantic witnesses require a complete publication",
            )?;
        }
        staged.unsubmitted = capture_rovent_r_unsubmitted_observations(
            &closure,
            &captured,
            staged.publication_sha256.as_deref(),
        )?;
        if let Some(unsubmitted) = &staged.unsubmitted {
            let remaining = maximum_serialized_bytes - staged.byte_len();
            encode_bounded(
                unsubmitted,
                &mut staged.unsubmitted_bytes,
                remaining.min(ROVENT_R_UNSUBMITTED_MAX_BYTES),
            )?;
            unsubmitted.validate_historical(
                closure.admission().raw_sha256(),
                staged.publication.as_ref(),
                staged.publication_sha256.as_deref(),
            )?;
        }
        check(
            closure.original_retirement().failures().is_empty()
                && closure.original_retirement().unrecorded_witness_attempts() == 0,
            "cohort R delivery unresolved; original caps retained",
        )?;
        Ok(())
    })();
    if let Err(error) = prepared {
        return Err(Box::new(RoventRSemanticMaterializationRejected {
            error,
            closure,
            captured,
            staged,
        }));
    }
    let mut detached = RoventRMaterializedSemantics {
        root: Arc::downgrade(closure.admission().root_admission()),
        source: Arc::downgrade(closure.nnuec_source()),
        authority: closure.admission().authority(),
        admission_sha256: closure.admission().raw_sha256().into(),
        input_lock_sha256: closure
            .admission()
            .native_origin()
            .original_input_lock()
            .canonical_sha256
            .clone(),
        parameter_sha256: closure
            .nnuec_source()
            .semantic_weights_identity()
            .weights_sha256
            .clone()
            .expect("actual collector already checked parameter identity"),
        lineages: closure
            .captures()
            .iter()
            .map(|original| original.reservation().lineage())
            .collect(),
        created_at,
        finished_at: created_at,
        staged,
    };
    // Exactly these original strong owners release only after complete
    // bounded detachment. This is not the actor/engine/cache/PinZero ACK.
    drop(captured);
    drop(closure);
    detached.finished_at = Instant::now();
    Ok(detached)
}

/// Actual final retirement authorizes handing historical bytes to E's durable
/// writer. It does not claim files have been written, synced or read back.
pub struct RoventRPublicationReadySemantics {
    original: RoventRMaterializedSemantics,
}
pub struct RoventRSemanticAuthorizationRejected {
    pub error: ArenaError,
    pub materialized: RoventRMaterializedSemantics,
}
// Private variants can only borrow the actual domain-specific final caps.
// Shared checks retain the exact /1 source, interval and publication guards.
enum ActualRPublicationRetirement<'a> {
    Ready(&'a UtilityTrialPhysicalRetirement),
    Intervention(&'a UtilityInterventionTrialPhysicalRetirement),
}
impl ActualRPublicationRetirement<'_> {
    fn root_fence_tick(&self) -> u64 {
        match self {
            Self::Ready(value) => value.cost().root_fence_tick,
            Self::Intervention(value) => value.cost().root_fence_tick,
        }
    }
    fn physical_pin_zero_tick(&self) -> u64 {
        match self {
            Self::Ready(value) => value.cost().physical_pin_zero_tick,
            Self::Intervention(value) => value.cost().physical_pin_zero_tick,
        }
    }
    fn matches_root_admission(&self, root: &Arc<RoventRootAdmission>) -> bool {
        match self {
            Self::Ready(value) => value.matches_root_admission(root),
            Self::Intervention(value) => value.matches_root_admission(root),
        }
    }
    fn authority(&self) -> SearchAuthority {
        match self {
            Self::Ready(value) => value.authority(),
            Self::Intervention(value) => value.authority(),
        }
    }
    fn input_lock_sha256(&self) -> &str {
        match self {
            Self::Ready(value) => &value.source_pins().input_lock_sha256,
            Self::Intervention(value) => &value.source_pins().input_lock_sha256,
        }
    }
    fn parameter_sha256(&self) -> &str {
        match self {
            Self::Ready(value) => &value.source_pins().frozen_nnuec_model_sha256,
            Self::Intervention(value) => &value.source_pins().frozen_nnuec_model_sha256,
        }
    }
}
impl RoventRMaterializedSemantics {
    pub fn retained_serialized_bytes(&self) -> usize {
        self.staged.byte_len()
    }
    pub fn original_capture_count(&self) -> usize {
        self.lineages.len()
    }
    pub fn matches_source(&self, source: &Arc<LoadedRoventNnuecSource>) -> bool {
        self.source.ptr_eq(&Arc::downgrade(source))
    }
    pub fn authorize_after_retirement_retaining(
        self,
        retirement: &UtilityTrialPhysicalRetirement,
        original_source: &Arc<LoadedRoventNnuecSource>,
        publication: Option<&RoventRCorpusPublication>,
        publication_bytes: Option<&[u8]>,
    ) -> Result<RoventRPublicationReadySemantics, Box<RoventRSemanticAuthorizationRejected>> {
        self.authorize_actual_retirement_retaining(
            ActualRPublicationRetirement::Ready(retirement),
            original_source,
            publication,
            publication_bytes,
        )
    }
    /// /2 원 cap만 받는다. /1으로 재태깅하거나 역사 JSON에서 인가하지 않는다.
    pub fn authorize_after_intervention_retirement_retaining(
        self,
        retirement: &UtilityInterventionTrialPhysicalRetirement,
        original_source: &Arc<LoadedRoventNnuecSource>,
        publication: Option<&RoventRCorpusPublication>,
        publication_bytes: Option<&[u8]>,
    ) -> Result<RoventRPublicationReadySemantics, Box<RoventRSemanticAuthorizationRejected>> {
        self.authorize_actual_retirement_retaining(
            ActualRPublicationRetirement::Intervention(retirement),
            original_source,
            publication,
            publication_bytes,
        )
    }
    fn authorize_actual_retirement_retaining(
        self,
        retirement: ActualRPublicationRetirement<'_>,
        original_source: &Arc<LoadedRoventNnuecSource>,
        publication: Option<&RoventRCorpusPublication>,
        publication_bytes: Option<&[u8]>,
    ) -> Result<RoventRPublicationReadySemantics, Box<RoventRSemanticAuthorizationRejected>> {
        let checked = (|| {
            let root = self
                .root
                .upgrade()
                .ok_or_else(|| invalid("cohort R original root owner is unavailable"))?;
            let tick = |time: Instant| {
                time.checked_duration_since(root.runtime_clock().origin())
                    .and_then(|elapsed| u64::try_from(elapsed.as_nanos()).ok())
                    .ok_or_else(|| invalid("cohort R original materialization clock overflow"))
            };
            check(
                tick(self.created_at)? >= retirement.root_fence_tick()
                    && tick(self.finished_at)? <= retirement.physical_pin_zero_tick()
                    && self.finished_at >= self.created_at,
                "cohort R bounded materialization is outside the actual fence-to-PinZero interval",
            )?;
            check(
                retirement.matches_root_admission(&root)
                    && retirement.authority() == self.authority
                    && self.matches_source(original_source)
                    && retirement.input_lock_sha256() == self.input_lock_sha256
                    && retirement.parameter_sha256() == self.parameter_sha256,
                "cohort R final retirement differs from original root/source",
            )?;
            check(
                publication == self.staged.publication.as_ref()
                    && publication_bytes
                        == (!self.staged.publication_bytes.is_empty())
                            .then_some(self.staged.publication_bytes.as_slice()),
                "cohort R final publication differs from original materialized bytes",
            )?;
            if let Some(publication) = publication {
                check(
                    publication
                        .admissions
                        .iter()
                        .any(|origin| origin.native_admission_sha256 == self.admission_sha256),
                    "cohort R final publication original admission missing",
                )?;
            }
            Ok(())
        })();
        match checked {
            Ok(()) => Ok(RoventRPublicationReadySemantics { original: self }),
            Err(error) => Err(Box::new(RoventRSemanticAuthorizationRejected {
                error,
                materialized: self,
            })),
        }
    }
}
impl RoventRPublicationReadySemantics {
    pub fn semantic(&self) -> Option<&RoventRSemanticObservations> {
        self.original.staged.semantic.as_ref()
    }
    pub fn semantic_bytes(&self) -> Option<&[u8]> {
        self.semantic()
            .map(|_| self.original.staged.semantic_bytes.as_slice())
    }
    pub fn unsubmitted(&self) -> Option<&RoventRUnsubmittedObservations> {
        self.original.staged.unsubmitted.as_ref()
    }
    pub fn unsubmitted_bytes(&self) -> Option<&[u8]> {
        self.unsubmitted()
            .map(|_| self.original.staged.unsubmitted_bytes.as_slice())
    }
    pub fn publication_sha256(&self) -> Option<&str> {
        self.original.staged.publication_sha256.as_deref()
    }
}

fn encode_bounded<T: Serialize>(
    value: &T,
    bytes: &mut Vec<u8>,
    limit: usize,
) -> Result<(), ArenaError> {
    struct Bounded<'a> {
        bytes: &'a mut Vec<u8>,
        limit: usize,
    }
    impl Write for Bounded<'_> {
        fn write(&mut self, input: &[u8]) -> io::Result<usize> {
            if self
                .bytes
                .len()
                .checked_add(input.len())
                .is_none_or(|len| len > self.limit)
            {
                return Err(io::Error::other("cohort R serialized metadata capacity"));
            }
            self.bytes.extend_from_slice(input);
            Ok(input.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
    check(
        bytes.is_empty(),
        "cohort R metadata buffer must be original and empty",
    )?;
    serde_json::to_writer(Bounded { bytes, limit }, value)
        .map_err(|error| invalid(error.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn bounded_metadata_encoding_matches_existing_serde_bytes_without_new_wire_format() {
        let value = serde_json::json!({"cost":null,"utility":null,"utf8":"원 관측"});
        let expected = serde_json::to_vec(&value).unwrap();
        let mut bytes = Vec::new();
        encode_bounded(&value, &mut bytes, expected.len()).unwrap();
        assert_eq!(bytes, expected);
    }
    #[test]
    fn capacity_failure_preserves_bounded_partial_bytes_and_does_not_rewrite_existing_buffer() {
        let mut bytes = Vec::new();
        assert!(
            encode_bounded(&serde_json::json!({"original":"abcdefgh"}), &mut bytes, 8).is_err()
        );
        assert!(bytes.len() <= 8);
        let original = bytes.clone();
        assert!(encode_bounded(&0, &mut bytes, 8).is_err());
        assert_eq!(bytes, original);
    }
}
