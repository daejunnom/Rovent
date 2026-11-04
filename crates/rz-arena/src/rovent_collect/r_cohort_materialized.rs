//! Cohort-only memory sink for the existing R writer. Original raw/1 bytes
//! and source checks are reused; no durable dataset I/O occurs here. The
//! 32 MiB limit covers retained serialized backing, not writer/DTO overhead,
//! shared live inputs, allocator capacity or a measured process-memory peak.

use super::{
    ArenaError, CapturedRoventRRetirement, FinishedRoventRCorpus, RoventRCollectionClosure,
    RoventRCorpusPublication, RoventRCorpusWriter, RoventRMaterializedSemantics,
    RoventRPublicationPacket, RoventRPublicationReadySemantics,
    RoventRSemanticMaterializationRejected, check, invalid,
    materialize_rovent_r_semantics_retaining,
};
use rz_experiments::rovent_r_readout::{RoventRCaptureRetirement, RoventRReadoutManifest};
use rz_uci::pals_adaptive::{
    UtilityInterventionTrialPhysicalRetirement, UtilityTrialPhysicalRetirement,
};
use rz_uci::rovent_bootstrap::{RoventStartupControl, nnuec_source::LoadedRoventNnuecSource};
use serde::Serialize;
use serde_json::Value;
use sha2::{Digest as _, Sha256};
use std::{
    collections::BTreeSet,
    io::{self, Write},
    sync::Arc,
};

pub const ROVENT_R_COHORT_SERIALIZED_MAX_BYTES: usize = 32 << 20;
const MAX_PACKETS: usize = 256;

/// Success owns no Closure/Captured/Reservation/Bound/View/Model strong lease.
/// Neither packet nor Finished/metadata bytes are exposed before actual final
/// retirement. Source FD reading remains the existing writer's explicit work.
pub struct RoventRCohortMaterialization {
    packed: Packed,
    semantic: Option<RoventRMaterializedSemantics>,
}
pub struct RoventRCohortMaterializationRejected {
    pub error: ArenaError,
    pub closure: Option<RoventRCollectionClosure>,
    pub captured: Vec<CapturedRoventRRetirement>,
    pub writer: Option<RoventRCorpusWriter>,
    pub semantic_rejection: Option<Box<RoventRSemanticMaterializationRejected>>,
    packed: Packed,
}
impl RoventRCohortMaterializationRejected {
    /// An exceptional rejected incoming packet is preserved separately and
    /// may exceed the admission budget. It never forms a successful cap.
    pub fn retained_serialized_bytes(&self) -> Option<usize> {
        self.packed
            .backing_bytes()
            .checked_add(
                self.packed
                    .rejected_packet
                    .as_ref()
                    .map_or(0, |p| p.bytes.len()),
            )
            .and_then(|total| {
                total.checked_add(
                    self.semantic_rejection
                        .as_ref()
                        .map_or(0, |rejected| rejected.retained_serialized_bytes()),
                )
            })
    }
    pub fn packets(&self) -> &[RoventRPublicationPacket] {
        &self.packed.packets
    }
    pub fn rejected_packet(&self) -> Option<&RoventRPublicationPacket> {
        self.packed.rejected_packet.as_ref()
    }
    pub fn finished(&self) -> Option<&FinishedRoventRCorpus> {
        self.packed.finished.as_ref()
    }
}

#[derive(Default)]
struct Packed {
    packets: Vec<RoventRPublicationPacket>,
    packet_bytes: usize,
    rejected_packet: Option<RoventRPublicationPacket>,
    finished: Option<FinishedRoventRCorpus>,
    publication: Option<RoventRCorpusPublication>,
    publication_bytes: Vec<u8>,
    manifest_bytes: Vec<u8>,
    index_bytes: Vec<u8>,
    allocation_bytes: usize,
}
impl Packed {
    fn backing_bytes(&self) -> usize {
        // Individual admission is <=32 MiB and all additions are checked
        // before mutation. This sum cannot overflow a supported usize.
        self.packet_bytes
            + self.publication_bytes.len()
            + self.manifest_bytes.len()
            + self.index_bytes.len()
            + self.allocation_bytes
    }
}

/// Every error returns the same original capture owners, writer/Finished,
/// accepted packets, rejected packet and partial metadata. The caller must
/// retain/quarantine that state instead of inferring an empty successful root.
pub fn materialize_rovent_r_cohort_retaining(
    closure: RoventRCollectionClosure,
    captured: Vec<CapturedRoventRRetirement>,
    manifest: &RoventRReadoutManifest,
    control: &RoventStartupControl,
    maximum_serialized_bytes: usize,
) -> Result<RoventRCohortMaterialization, Box<RoventRCohortMaterializationRejected>> {
    let mut state = RoventRCohortMaterializationRejected {
        error: invalid("cohort R materialization not completed"),
        closure: Some(closure),
        captured,
        writer: None,
        semantic_rejection: None,
        packed: Packed::default(),
    };
    let prepared = (|| {
        control
            .check()
            .map_err(|error| invalid(error.to_string()))?;
        check(
            (1..=ROVENT_R_COHORT_SERIALIZED_MAX_BYTES).contains(&maximum_serialized_bytes)
                && state.captured.len() <= 64,
            "cohort R explicit serialized backing bound",
        )?;
        manifest.validate()?;
        validate_original_capture_scope(
            state.closure.as_ref().expect("original cohort closure"),
            &state.captured,
        )?;
        // Reject an oversized known input before the old writer creates its
        // source/row buffers. Counting serializes metadata without storing it,
        // replaying Rules, evaluating a model or reading source FD bytes.
        state.packed.allocation_bytes = allocation_backing_bytes(&state.captured)?;
        check(
            state.packed.allocation_bytes <= maximum_serialized_bytes,
            "cohort R allocation backing bound",
        )?;
        let raw_limit = maximum_serialized_bytes - state.packed.allocation_bytes;
        preflight_raw_bytes(&state.captured, raw_limit, control)?;
        if !state.captured.is_empty() {
            state.writer = Some(RoventRCorpusWriter::new(manifest)?);
            for capture in &state.captured {
                let packed = &mut state.packed;
                state
                    .writer
                    .as_mut()
                    .expect("original R writer")
                    .append_retirement(capture, control, |packet| {
                        retain_packet(packed, packet, raw_limit, control)
                    })?;
            }
            let writer = state.writer.take().expect("same original R writer");
            let packed = &mut state.packed;
            match writer
                .finish_retaining(|packet| retain_packet(packed, packet, raw_limit, control))
            {
                Ok(finished) => state.packed.finished = Some(finished),
                Err(rejected) => {
                    let rejected = *rejected;
                    state.writer = Some(rejected.writer);
                    return Err(rejected.error);
                }
            }
            let finished = state
                .packed
                .finished
                .as_ref()
                .expect("original finished R owner");
            let mut left = maximum_serialized_bytes - state.packed.backing_bytes();
            encode_bounded(finished.manifest(), &mut state.packed.manifest_bytes, left)?;
            left = maximum_serialized_bytes - state.packed.backing_bytes();
            encode_bounded(finished.index(), &mut state.packed.index_bytes, left)?;
            state.packed.publication = Some(finished.planned_publication()?);
            left = maximum_serialized_bytes - state.packed.backing_bytes();
            encode_bounded(
                state
                    .packed
                    .publication
                    .as_ref()
                    .expect("planned original publication"),
                &mut state.packed.publication_bytes,
                left,
            )?;
        }
        control
            .check()
            .map_err(|error| invalid(error.to_string()))?;
        Ok(())
    })();
    if let Err(error) = prepared {
        state.error = error;
        return Err(Box::new(state));
    }
    // The semantic owner accounts for its own duplicate publication buffer
    // and sidecar bytes inside this remaining aggregate allowance.
    let remaining = maximum_serialized_bytes - state.packed.backing_bytes();
    let closure = state
        .closure
        .take()
        .expect("original closure retained until detachment");
    let captured = std::mem::take(&mut state.captured);
    let publication_bytes = state
        .packed
        .publication
        .as_ref()
        .map(|_| state.packed.publication_bytes.as_slice());
    match materialize_rovent_r_semantics_retaining(
        closure,
        captured,
        state.packed.publication.as_ref(),
        publication_bytes,
        remaining,
    ) {
        Ok(semantic) => Ok(RoventRCohortMaterialization {
            packed: state.packed,
            semantic: Some(semantic),
        }),
        Err(rejected) => {
            state.error = invalid(rejected.error.to_string());
            state.semantic_rejection = Some(rejected);
            Err(Box::new(state))
        }
    }
}

pub struct RoventRCohortAuthorizationRejected {
    pub error: ArenaError,
    pub materialized: RoventRCohortMaterialization,
}
pub struct RoventRCohortPublicationReady {
    packed: Packed,
    semantic: RoventRPublicationReadySemantics,
}
impl RoventRCohortMaterialization {
    pub fn retained_serialized_bytes(&self) -> usize {
        self.packed.backing_bytes()
            + self
                .semantic
                .as_ref()
                .map_or(0, |semantic| semantic.retained_serialized_bytes())
    }
    pub fn authorize_after_retirement_retaining(
        self,
        retirement: &UtilityTrialPhysicalRetirement,
        original_source: &Arc<LoadedRoventNnuecSource>,
    ) -> Result<RoventRCohortPublicationReady, Box<RoventRCohortAuthorizationRejected>> {
        self.authorize_actual_retirement_retaining(|semantic, publication, bytes| {
            semantic.authorize_after_retirement_retaining(
                retirement,
                original_source,
                publication,
                bytes,
            )
        })
    }
    /// /2가 발급한 물리 퇴역 cap으로만 기존 R 원문을 게시한다.
    pub fn authorize_after_intervention_retirement_retaining(
        self,
        retirement: &UtilityInterventionTrialPhysicalRetirement,
        original_source: &Arc<LoadedRoventNnuecSource>,
    ) -> Result<RoventRCohortPublicationReady, Box<RoventRCohortAuthorizationRejected>> {
        self.authorize_actual_retirement_retaining(|semantic, publication, bytes| {
            semantic.authorize_after_intervention_retirement_retaining(
                retirement,
                original_source,
                publication,
                bytes,
            )
        })
    }
    fn authorize_actual_retirement_retaining(
        mut self,
        authorize: impl FnOnce(
            RoventRMaterializedSemantics,
            Option<&RoventRCorpusPublication>,
            Option<&[u8]>,
        ) -> Result<
            RoventRPublicationReadySemantics,
            Box<super::RoventRSemanticAuthorizationRejected>,
        >,
    ) -> Result<RoventRCohortPublicationReady, Box<RoventRCohortAuthorizationRejected>> {
        let semantic = self.semantic.take().expect("original detached semantics");
        let bytes = self
            .packed
            .publication
            .as_ref()
            .map(|_| self.packed.publication_bytes.as_slice());
        match authorize(semantic, self.packed.publication.as_ref(), bytes) {
            Ok(semantic) => Ok(RoventRCohortPublicationReady {
                packed: self.packed,
                semantic,
            }),
            Err(rejected) => {
                let rejected = *rejected;
                self.semantic = Some(rejected.materialized);
                Err(Box::new(RoventRCohortAuthorizationRejected {
                    error: rejected.error,
                    materialized: self,
                }))
            }
        }
    }
}
impl RoventRCohortPublicationReady {
    pub fn packets(&self) -> &[RoventRPublicationPacket] {
        &self.packed.packets
    }
    /// Original allocation metadata stays in this same Finished owner.
    pub fn finished(&self) -> Option<&FinishedRoventRCorpus> {
        self.packed.finished.as_ref()
    }
    /// Original planned declaration. File durability is still E's separate ACK.
    pub fn publication(&self) -> Option<&RoventRCorpusPublication> {
        self.packed.publication.as_ref()
    }
    pub fn publication_bytes(&self) -> Option<&[u8]> {
        self.packed
            .publication
            .as_ref()
            .map(|_| self.packed.publication_bytes.as_slice())
    }
    pub fn semantic(&self) -> &RoventRPublicationReadySemantics {
        &self.semantic
    }
    pub fn manifest_bytes(&self) -> Option<&[u8]> {
        self.finished()
            .map(|_| self.packed.manifest_bytes.as_slice())
    }
    pub fn index_bytes(&self) -> Option<&[u8]> {
        self.finished().map(|_| self.packed.index_bytes.as_slice())
    }
}

fn retain_packet(
    packed: &mut Packed,
    packet: RoventRPublicationPacket,
    limit: usize,
    control: &RoventStartupControl,
) -> Result<(), ArenaError> {
    let checked = (|| {
        control
            .check()
            .map_err(|error| invalid(error.to_string()))?;
        packet.declaration.validate()?;
        check(
            packed.rejected_packet.is_none()
                && packed.packets.len() < MAX_PACKETS
                && packet.declaration.bytes == packet.bytes.len() as u64
                && packet.declaration.sha256 == format!("{:x}", Sha256::digest(&packet.bytes))
                && !packed
                    .packets
                    .iter()
                    .any(|p| p.declaration.path == packet.declaration.path),
            "cohort R original packet pin/path/capacity mismatch",
        )?;
        check(
            packed
                .packet_bytes
                .checked_add(packet.bytes.len())
                .is_some_and(|total| total <= limit),
            "cohort R packet serialized backing capacity",
        )
    })();
    if let Err(error) = checked {
        packed.rejected_packet = Some(packet);
        return Err(error);
    }
    packed.packet_bytes += packet.bytes.len();
    packed.packets.push(packet);
    Ok(())
}

fn allocation_backing_bytes(captured: &[CapturedRoventRRetirement]) -> Result<usize, ArenaError> {
    let mut admissions = BTreeSet::new();
    let mut bytes = 0usize;
    for original in captured {
        if admissions.insert(original.admission().raw_sha256()) {
            let size = original
                .admission()
                .root_admission()
                .allocation_clock_canonical_bytes()
                .len();
            check(size <= 4096, "cohort R original allocation byte cap")?;
            bytes = bytes
                .checked_add(size)
                .ok_or_else(|| invalid("cohort R allocation byte overflow"))?;
        }
    }
    Ok(bytes)
}

/// This admission precedes source FD reading. A different actual capture cap
/// with equal public metadata cannot borrow this Closure's source authority.
fn validate_original_capture_scope(
    closure: &RoventRCollectionClosure,
    captured: &[CapturedRoventRRetirement],
) -> Result<(), ArenaError> {
    check(
        closure
            .captures()
            .iter()
            .filter(|held| held.retirement().is_some())
            .count()
            == captured.len(),
        "cohort R packing requires complete original terminal coverage",
    )?;
    let mut seen = BTreeSet::new();
    for original in captured {
        check(
            Arc::ptr_eq(original.nnuec_source(), closure.nnuec_source())
                && Arc::ptr_eq(
                    original.admission().root_admission(),
                    closure.admission().root_admission(),
                )
                && original.admission().raw_sha256() == closure.admission().raw_sha256()
                && closure.captures().iter().any(|held| {
                    held.retirement().is_some() && original.matches_reservation(held.reservation())
                })
                && seen.insert(original.retirement_sha256()),
            "cohort R packing requires the same original capture/source/root instances",
        )?;
    }
    Ok(())
}

fn preflight_raw_bytes(
    captured: &[CapturedRoventRRetirement],
    limit: usize,
    control: &RoventStartupControl,
) -> Result<(), ArenaError> {
    #[derive(Serialize)]
    struct RetirementBody<'a> {
        schema: &'static str,
        native_admission: &'a Value,
        native_admission_sha256: &'a str,
        retirement: &'a RoventRCaptureRetirement,
        retirement_sha256: &'a str,
        readout_record_sha256: Vec<&'a str>,
    }
    let mut sources = BTreeSet::new();
    let mut remaining = limit;
    let mut charge = |bytes: usize| -> Result<(), ArenaError> {
        remaining = remaining.checked_sub(bytes).ok_or_else(|| {
            invalid("cohort R known raw packets exceed serialized backing capacity")
        })?;
        Ok(())
    };
    for original in captured {
        control
            .check()
            .map_err(|error| invalid(error.to_string()))?;
        let source = original.nnuec_source();
        let key = (
            source.input_lock_digest(),
            source.weights_artifact().sha256.clone(),
            source.provenance_artifact().sha256.clone(),
        );
        if sources.insert(key) {
            for bytes in [
                source.weights_artifact().bytes,
                source.provenance_artifact().bytes,
            ] {
                charge(
                    usize::try_from(bytes).map_err(|_| invalid("cohort R source byte overflow"))?,
                )?;
            }
            charge(encoded_len(
                original.admission().native_origin().original_input_lock(),
                limit,
            )?)?;
            // The existing source helper bounds any runtime-bundle metadata
            // at 64 KiB. Reserve that conservative allowance without reading
            // its FD twice or inventing a bundle cache hit.
            charge(64 << 10)?;
        }
        for row in original.rows() {
            charge(
                encoded_len(row.record(), limit)?
                    .checked_add(1)
                    .ok_or_else(|| invalid("cohort R row byte overflow"))?,
            )?;
        }
        let body = RetirementBody {
            schema: super::ROVENT_R_RETIREMENT_OBSERVATION_DOMAIN,
            native_admission: original.admission().raw_receipt(),
            native_admission_sha256: original.admission().raw_sha256(),
            retirement: original.retirement(),
            retirement_sha256: original.retirement_sha256(),
            readout_record_sha256: original
                .rows()
                .iter()
                .map(|row| row.record().record_sha256.as_str())
                .collect(),
        };
        charge(encoded_len(&body, limit)?)?;
    }
    Ok(())
}

struct Counter {
    used: usize,
    limit: usize,
}
impl Write for Counter {
    fn write(&mut self, input: &[u8]) -> io::Result<usize> {
        self.used = self
            .used
            .checked_add(input.len())
            .filter(|total| *total <= self.limit)
            .ok_or_else(|| io::Error::other("cohort R serialized backing capacity"))?;
        Ok(input.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
fn encoded_len<T: Serialize>(value: &T, limit: usize) -> Result<usize, ArenaError> {
    let mut count = Counter { used: 0, limit };
    serde_json::to_writer(&mut count, value).map_err(|error| invalid(error.to_string()))?;
    Ok(count.used)
}
fn encode_bounded<T: Serialize>(
    value: &T,
    bytes: &mut Vec<u8>,
    limit: usize,
) -> Result<(), ArenaError> {
    let required = encoded_len(value, limit)?;
    check(
        bytes.is_empty(),
        "cohort R original metadata buffer must be empty",
    )?;
    // Count first, reserve exactly this additional backing, then serialize the
    // same immutable value. This does not promise allocator/host peak memory.
    bytes
        .try_reserve_exact(required)
        .map_err(|_| invalid("cohort R metadata allocation failed"))?;
    serde_json::to_writer(bytes, value).map_err(|error| invalid(error.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn metadata_preflight_and_encoding_keep_original_serde_bytes_and_capacity_rejection() {
        let value = serde_json::json!({"원문":"보존","cost":null,"target":null});
        let expected = serde_json::to_vec(&value).unwrap();
        assert_eq!(encoded_len(&value, expected.len()).unwrap(), expected.len());
        assert!(encoded_len(&value, expected.len() - 1).is_err());
        let mut bytes = Vec::new();
        encode_bounded(&value, &mut bytes, expected.len()).unwrap();
        assert_eq!(bytes, expected);
        let mut rejected = Vec::new();
        assert!(encode_bounded(&value, &mut rejected, expected.len() - 1).is_err());
        assert!(rejected.is_empty());
    }
}
