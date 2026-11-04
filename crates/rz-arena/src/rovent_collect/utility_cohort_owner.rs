//! Cohort payload pins and the separate durable collection-I/O receipt.
//!
//! Decoding these historical documents never creates an executable trial,
//! physical retirement or permission to release an original owner. The actual
//! publisher consumes the UCI-issued materialized/final-retirement capabilities;
//! historical consumers must additionally verify their original payloads.
use super::*;
use serde::{Deserialize, Serialize};
mod source;
pub use source::*;
pub(in crate::rovent_collect) mod historical;

/// 원 payload 바이트의 순수 역사 검증이다. 현재 root/실행/학습 권한을
/// 발급하지 않으며 FD와 원 file pin의 생존은 호출자가 별도로 보존한다.
pub fn verify_historical_utility_cohort(
    publication: &RoventUtilityCohortPublication,
    payloads: &BTreeMap<String, Vec<u8>>,
) -> Result<RoventHistoricalUtilityCohortEvidence, ArenaError> {
    historical::verify_historical_utility_cohort(publication, payloads)
}
#[cfg(target_os = "linux")]
mod trial_writer;
#[cfg(target_os = "linux")]
pub use trial_writer::RoventUtilityCohortTrialWriter;

pub const ROVENT_UTILITY_COHORT_PUBLICATION: &str = "rz-rovent-utility-cohort-publication/1";
pub const ROVENT_UTILITY_COHORT_DURABLE_ACK: &str = "rz-rovent-utility-cohort-durable-ack/1";
pub const ROVENT_UTILITY_COHORT_COST_SCOPE: &str = "cohort_route_with_bounded_materialization_and_physical_pin_zero_excluding_durable_dataset_io_v1";
pub const ROVENT_UTILITY_COHORT_PUBLICATION_FILE: &str = "utility-cohort-publication.json";
pub const ROVENT_UTILITY_COHORT_ACK_FILE: &str = "utility-cohort-durable-ack.json";
pub const ROVENT_UTILITY_COHORT_METADATA_MAX: usize = 64 << 10;
pub const ROVENT_UTILITY_COHORT_PAYLOAD_MAX: u64 = 256 << 20;
pub const ROVENT_UTILITY_COHORT_MATERIALIZATION_MAX: u64 = 32 << 20;

/// Historical evidence returned after the original raw payloads, Rules,
/// source namespace, independent trials and pair proofs all agree. This is
/// detached data; the parent separately owns original FDs and learner intake.
#[derive(Clone, Debug)]
pub struct RoventHistoricalUtilityCohortEvidence {
    pub cohort_id: String,
    pub profile_sha256: String,
    pub usage: String,
    pub collection_source_sha256: String,
    pub cost_epoch_sha256: String,
    pub resource_profile_sha256: String,
    pub frozen_base_weights_sha256: String,
    pub input_reference_json: String,
    pub input_sha256: String,
    pub target_sha256: String,
    pub submission_target_sha256: String,
    pub utility_context_sha256: String,
    pub known_pairs: Vec<[u8; 2]>,
    pub known_pair_sha256: Vec<String>,
    pub all_pair_sha256: Vec<String>,
    pub frozen_pc_sources: Vec<RoventHistoricalUtilityFrozenPcSource>,
    pub split: GenerationSplitV3,
    pub opening_id: String,
    pub game_id: String,
    pub line_genealogy_id: String,
    pub transposition_sha256: String,
    pub rules_state_sha256: String,
    pub ancestor_dataset_publication_sha256: Vec<String>,
    pub ancestor_datasets: Vec<RoventUtilityCohortAncestorSource>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoventUtilityCohortAncestorSource {
    pub kind: String,
    pub publication_sha256: String,
    pub metadata: Value,
}
impl RoventUtilityCohortAncestorSource {
    pub fn validate(&self) -> Result<(), ArenaError> {
        let keys: &[&str] = match self.kind.as_str() {
            "pc_legacy" | "pc_submitted_input_v1" | "r_auxiliary" => &[
                "manifest_file_sha256",
                "index_file_sha256",
                "canonical_manifest_sha256",
                "canonical_index_sha256",
            ],
            "utility_cohort" => &[
                "dataset_sha256",
                "pool_sha256",
                "preparation_sha256",
                "context_sha256",
            ],
            _ => return Err(invalid("utility cohort unknown ancestor family")),
        };
        check(
            generation_sha(&self.publication_sha256)
                && self.metadata.as_object().is_some_and(|metadata| {
                    metadata.len() == keys.len()
                        && keys.iter().all(|key| {
                            metadata
                                .get(*key)
                                .and_then(Value::as_str)
                                .is_some_and(generation_sha)
                        })
                }),
            "utility cohort exact original ancestor descriptor",
        )
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoventHistoricalUtilityFrozenPcSource {
    pub role: String,
    pub artifact: rz_experiments::ArtifactRef,
    pub model_semantics_sha256: String,
    pub model_identity_sha256: String,
    pub encoding_sha256: String,
}

/// One original cohort and one frozen decision context. Every file SHA is the
/// raw file-byte SHA, distinct from native/domain seals inside its body.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoventUtilityCohortPublication {
    pub schema: String,
    pub cohort_id: String,
    pub cost_scope: String,
    pub cohort_profile: RoventPublicationFile,
    pub frozen_context: RoventPublicationFile,
    pub source: RoventPublicationFile,
    pub input_lock: RoventPublicationFile,
    pub nnuec_weights: RoventPublicationFile,
    pub nnuec_provenance: RoventPublicationFile,
    pub materializations: Vec<RoventPublicationFile>,
    pub trials: Vec<RoventPublicationFile>,
    pub pairs: Vec<RoventPublicationFile>,
    pub trial_count: u64,
    pub pair_count: u64,
    /// All unique payload bytes; excludes publication and its later ACK.
    pub total_payload_bytes: u64,
}
impl RoventUtilityCohortPublication {
    pub fn from_json_historical(raw: &str) -> Result<Self, ArenaError> {
        check(
            raw.len() <= ROVENT_UTILITY_COHORT_METADATA_MAX,
            "utility cohort publication exceeds 64 KiB",
        )?;
        let value: Value = rz_experiments::decode_json(raw)?;
        check(
            value.is_object(),
            "utility cohort publication object required",
        )?;
        for field in [
            "cohort_profile",
            "frozen_context",
            "source",
            "input_lock",
            "nnuec_weights",
            "nnuec_provenance",
        ] {
            check(
                value[field].is_object(),
                "utility cohort file pin object required",
            )?;
        }
        for field in ["materializations", "trials", "pairs"] {
            check(
                value[field]
                    .as_array()
                    .is_some_and(|pins| pins.iter().all(Value::is_object)),
                "utility cohort file pin object list required",
            )?;
        }
        let value: Self =
            serde_json::from_value(value).map_err(|error| invalid(error.to_string()))?;
        value.validate()?;
        Ok(value)
    }

    /// Closed shape/coverage only. This does not assert Rules, native
    /// acceptance, physical pin-zero or availability of a known pair.
    pub fn validate(&self) -> Result<(), ArenaError> {
        check(
            self.schema == ROVENT_UTILITY_COHORT_PUBLICATION
                && generation_sha(&self.cohort_id)
                && self.cost_scope == ROVENT_UTILITY_COHORT_COST_SCOPE
                && (1..=3).contains(&self.trials.len())
                && self.materializations.len() <= self.trials.len()
                && self.pairs.len() <= 3
                && self.pairs.len() <= self.trials.len() * (self.trials.len() - 1) / 2
                && self.trial_count == self.trials.len() as u64
                && self.pair_count == self.pairs.len() as u64,
            "utility cohort closed domain/count/cost scope",
        )?;
        let fixed = [
            (&self.cohort_profile, "cohort-profile.json", 64 << 10),
            (&self.frozen_context, "frozen-context.json", 32 << 20),
            (&self.source, "source.json", 1 << 20),
            (&self.input_lock, "input-lock.json", 4 << 20),
            (&self.nnuec_weights, "nnuec-weights.json", 32 << 20),
            (&self.nnuec_provenance, "nnuec-provenance.json", 1 << 20),
        ];
        let mut paths = BTreeSet::new();
        let mut bytes = 0u64;
        let mut add = |pin: &RoventPublicationFile, max: u64| -> Result<(), ArenaError> {
            pin.validate()?;
            check(
                pin.bytes <= max && paths.insert(pin.path.clone()),
                "utility cohort duplicate path/payload cap",
            )?;
            bytes = bytes
                .checked_add(pin.bytes)
                .ok_or_else(|| invalid("utility cohort payload byte overflow"))?;
            Ok(())
        };
        for (pin, name, maximum) in fixed {
            check(pin.path == name, "utility cohort fixed original file path")?;
            add(pin, maximum)?;
        }
        // Materializations can be absent for a genuinely unresolved trial.
        // Their original ordinal is retained in the filename and trial body;
        // an absent materialization is never synthesized as empty input.
        let mut previous = None;
        for pin in &self.materializations {
            let ordinal = payload_ordinal(&pin.path, "materializations")?;
            check(
                ordinal < self.trials.len() && previous.is_none_or(|last| last < ordinal),
                "utility cohort materialization ordinal/order",
            )?;
            previous = Some(ordinal);
            add(pin, ROVENT_UTILITY_COHORT_MATERIALIZATION_MAX)?;
        }
        for (ordinal, pin) in self.trials.iter().enumerate() {
            check(
                payload_ordinal(&pin.path, "trials")? == ordinal,
                "utility cohort original trial order",
            )?;
            // The original UCI checkpoint embeds the materialized JSON string
            // so one original checkpoint FD can acknowledge its durability.
            add(pin, ROVENT_UTILITY_COHORT_MATERIALIZATION_MAX)?;
        }
        for (index, pin) in self.pairs.iter().enumerate() {
            check(
                payload_ordinal(&pin.path, "pairs")? == index,
                "utility cohort original pair order",
            )?;
            add(pin, 64 << 10)?;
        }
        check(
            bytes == self.total_payload_bytes && bytes <= ROVENT_UTILITY_COHORT_PAYLOAD_MAX,
            "utility cohort exact unique payload bytes",
        )
    }

    pub fn payloads(&self) -> impl Iterator<Item = &RoventPublicationFile> {
        [
            &self.cohort_profile,
            &self.frozen_context,
            &self.source,
            &self.input_lock,
            &self.nnuec_weights,
            &self.nnuec_provenance,
        ]
        .into_iter()
        .chain(self.materializations.iter())
        .chain(self.trials.iter())
        .chain(self.pairs.iter())
    }

    pub fn to_json_bytes(&self) -> Result<Vec<u8>, ArenaError> {
        self.validate()?;
        let bytes = serde_json::to_vec(self).map_err(|error| invalid(error.to_string()))?;
        check(
            bytes.len() <= ROVENT_UTILITY_COHORT_METADATA_MAX,
            "utility cohort publication serialization cap",
        )?;
        Ok(bytes)
    }
}

/// The final publication pass only, after the last engine pin-zero boundary.
/// Inter-trial checkpoint intervals belong to their original trial ACKs.
/// Writes exclude already durable trial/materialization files; readback and
/// sync cover every payload and publication again. This ACK's own I/O is not
/// included in its self-referential counters.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoventUtilityCohortCollectionIo {
    pub clock_epoch: u64,
    pub started_tick_ns: u64,
    pub finished_tick_ns: u64,
    pub elapsed_ns: u64,
    pub written_bytes: u64,
    pub readback_bytes: u64,
    pub synced_files: u64,
    pub synced_directories: u64,
}

/// Projection of the original FD-bearing UCI ACK and its caller-observed
/// complete checkpoint I/O interval. The native trial seal and raw file pin
/// keep their separate meanings. This object is never accepted as a live ACK.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoventUtilityCohortTrialAcknowledgement {
    pub trial_sequence: u64,
    pub native_trial_sha256: String,
    pub checkpoint: RoventPublicationFile,
    pub collection_io: RoventUtilityCohortCollectionIo,
}

/// Historical file receipt, not the private move-only ACK accepted by UCI.
/// Keeping it in a separate externally pinned file avoids a publication
/// self-hash cycle and never subtracts I/O duration from an engine cost.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoventUtilityCohortDurableAck {
    pub schema: String,
    pub cohort_id: String,
    pub cost_scope: String,
    pub publication: RoventPublicationFile,
    pub collection_io: RoventUtilityCohortCollectionIo,
    pub trial_acknowledgements: Vec<RoventUtilityCohortTrialAcknowledgement>,
}
impl RoventUtilityCohortDurableAck {
    pub fn from_json_historical(
        raw: &str,
        publication: &RoventUtilityCohortPublication,
        publication_file: &RoventPublicationFile,
    ) -> Result<Self, ArenaError> {
        check(
            raw.len() <= ROVENT_UTILITY_COHORT_METADATA_MAX,
            "utility cohort durable ACK exceeds 64 KiB",
        )?;
        let value: Value = rz_experiments::decode_json(raw)?;
        check(
            value.is_object()
                && value["publication"].is_object()
                && value["collection_io"].is_object()
                && value["trial_acknowledgements"]
                    .as_array()
                    .is_some_and(|acks| {
                        acks.iter().all(|ack| {
                            ack.is_object()
                                && ack["checkpoint"].is_object()
                                && ack["collection_io"].is_object()
                        })
                    }),
            "utility cohort ACK/publication/collection I/O objects required",
        )?;
        let value: Self =
            serde_json::from_value(value).map_err(|error| invalid(error.to_string()))?;
        value.validate_historical(publication, publication_file)?;
        Ok(value)
    }

    pub fn validate_historical(
        &self,
        publication: &RoventUtilityCohortPublication,
        publication_file: &RoventPublicationFile,
    ) -> Result<(), ArenaError> {
        publication.validate()?;
        publication_file.validate()?;
        let expected_readback_bytes = publication
            .total_payload_bytes
            .checked_add(publication_file.bytes)
            .ok_or_else(|| invalid("utility cohort collection byte overflow"))?;
        let checkpoint_bytes = publication
            .materializations
            .iter()
            .chain(publication.trials.iter())
            .try_fold(0u64, |sum, pin| sum.checked_add(pin.bytes))
            .ok_or_else(|| invalid("utility cohort checkpoint byte overflow"))?;
        let expected_written_bytes = expected_readback_bytes
            .checked_sub(checkpoint_bytes)
            .ok_or_else(|| invalid("utility cohort final publication byte underflow"))?;
        let io = &self.collection_io;
        check(
            self.trial_acknowledgements.len() == publication.trials.len(),
            "utility cohort complete original inter-trial ACK coverage",
        )?;
        for (ordinal, ack) in self.trial_acknowledgements.iter().enumerate() {
            let materialized_bytes = publication
                .materializations
                .iter()
                .find(|pin| payload_ordinal(&pin.path, "materializations").ok() == Some(ordinal))
                .map_or(0, |pin| pin.bytes);
            let interval = &ack.collection_io;
            check(
                ack.trial_sequence == ordinal as u64 + 1
                    && generation_sha(&ack.native_trial_sha256)
                    && ack.checkpoint == publication.trials[ordinal]
                    && interval.clock_epoch > 0
                    && interval
                        .finished_tick_ns
                        .checked_sub(interval.started_tick_ns)
                        == Some(interval.elapsed_ns)
                    && interval.written_bytes == materialized_bytes + ack.checkpoint.bytes
                    && interval.readback_bytes == materialized_bytes + 2 * ack.checkpoint.bytes
                    && interval.synced_files == 2 + u64::from(materialized_bytes > 0)
                    && interval.synced_directories == 3 * (1 + u64::from(materialized_bytes > 0)),
                "utility cohort original checkpoint/ACK/I/O accounting",
            )?;
        }
        check(
            self.schema == ROVENT_UTILITY_COHORT_DURABLE_ACK
                && self.cohort_id == publication.cohort_id
                && self.cost_scope == publication.cost_scope
                && self.publication == *publication_file
                && publication_file.path == ROVENT_UTILITY_COHORT_PUBLICATION_FILE
                && publication_file.bytes <= ROVENT_UTILITY_COHORT_METADATA_MAX as u64
                && io.clock_epoch > 0
                && io.finished_tick_ns.checked_sub(io.started_tick_ns) == Some(io.elapsed_ns)
                && io.written_bytes == expected_written_bytes
                && io.readback_bytes == expected_readback_bytes
                && io.synced_files == publication.payloads().count() as u64 + 1
                && io.synced_directories > 0,
            "utility cohort original publication/collection I/O receipt",
        )
    }

    /// 각 원 trial의 시계 안에서만 게시 I/O가 실제 pin0 뒤인지 조인한다.
    /// 원 FD/pin과 전체 trial proof는 호출자의 별도 역사 검증을 유지한다.
    /// 서로 다른 Product의 tick 숫자는 비교하지 않는다.
    pub fn validate_original_trial_io(
        &self,
        publication: &RoventUtilityCohortPublication,
        payloads: &BTreeMap<String, Vec<u8>>,
    ) -> Result<(), ArenaError> {
        self.validate_historical(publication, &self.publication)?;
        for (ordinal, (pin, ack)) in publication
            .trials
            .iter()
            .zip(&self.trial_acknowledgements)
            .enumerate()
        {
            let raw = payloads
                .get(&pin.path)
                .ok_or_else(|| invalid("utility cohort original trial bytes missing"))?;
            check(
                raw.len() as u64 == pin.bytes
                    && hex(Digest(Sha256::digest(raw).into())) == pin.sha256,
                "utility cohort I/O original trial file pin differs",
            )?;
            let text = std::str::from_utf8(raw)
                .map_err(|_| invalid("utility cohort I/O original trial UTF8"))?;
            let trial: Value = rz_experiments::decode_json(text)?;
            let domain = "rz-rovent-utility-cohort-trial/1";
            let mut native = Sha256::new();
            native.update(domain.as_bytes());
            native.update([0]);
            native.update((raw.len() as u64).to_le_bytes());
            native.update(raw);
            check(
                trial.is_object()
                    && trial["schema"] == domain
                    && trial["cohort_id"] == self.cohort_id
                    && trial["trial_sequence"].as_u64() == Some(ack.trial_sequence)
                    && hex(Digest(native.finalize().into())) == ack.native_trial_sha256,
                "utility cohort I/O original checkpoint identity differs",
            )?;
            let clock = trial["cost"]["clock_epoch"]
                .as_u64()
                .ok_or_else(|| invalid("utility cohort original trial cost clock unavailable"))?;
            let pin_zero = trial["cost"]["physical_pin_zero_tick"]
                .as_u64()
                .ok_or_else(|| invalid("utility cohort original trial pin0 unavailable"))?;
            check(
                ack.collection_io.clock_epoch == clock
                    && ack.collection_io.started_tick_ns >= pin_zero,
                "utility cohort checkpoint I/O precedes original pin0 or changes clock",
            )?;
            if ordinal + 1 == publication.trials.len() {
                check(
                    self.collection_io.clock_epoch == clock
                        && self.collection_io.started_tick_ns >= ack.collection_io.finished_tick_ns,
                    "utility cohort final I/O precedes last original ACK or changes clock",
                )?;
            }
        }
        Ok(())
    }
}

fn payload_ordinal(path: &str, directory: &str) -> Result<usize, ArenaError> {
    let number = path
        .strip_prefix(directory)
        .and_then(|tail| tail.strip_prefix('/'))
        .and_then(|tail| tail.strip_suffix(".json"))
        .filter(|number| number.len() == 6 && number.bytes().all(|b| b.is_ascii_digit()))
        .ok_or_else(|| invalid("utility cohort normalized ordinal path"))?;
    number
        .parse()
        .map_err(|_| invalid("utility cohort ordinal overflow"))
}

#[cfg(test)]
mod tests {
    use super::*;
    fn synthetic_io_fixture() -> (
        RoventUtilityCohortPublication,
        RoventPublicationFile,
        RoventUtilityCohortDurableAck,
    ) {
        // 명시 합성 file pins: 현재 source/ACK/실행 권한을 만들지 않는다.
        let pin = |path: &str, bytes| RoventPublicationFile {
            path: path.into(),
            sha256: "a".repeat(64),
            bytes,
        };
        let publication = RoventUtilityCohortPublication {
            schema: ROVENT_UTILITY_COHORT_PUBLICATION.into(),
            cohort_id: "b".repeat(64),
            cost_scope: ROVENT_UTILITY_COHORT_COST_SCOPE.into(),
            cohort_profile: pin("cohort-profile.json", 1),
            frozen_context: pin("frozen-context.json", 1),
            source: pin("source.json", 1),
            input_lock: pin("input-lock.json", 1),
            nnuec_weights: pin("nnuec-weights.json", 1),
            nnuec_provenance: pin("nnuec-provenance.json", 1),
            materializations: vec![],
            trials: vec![pin("trials/000000.json", 2), pin("trials/000001.json", 2)],
            pairs: vec![],
            trial_count: 2,
            pair_count: 0,
            total_payload_bytes: 10,
        };
        let publication_file = pin(ROVENT_UTILITY_COHORT_PUBLICATION_FILE, 3);
        let ack = RoventUtilityCohortDurableAck {
            schema: ROVENT_UTILITY_COHORT_DURABLE_ACK.into(),
            cohort_id: publication.cohort_id.clone(),
            cost_scope: publication.cost_scope.clone(),
            publication: publication_file.clone(),
            collection_io: RoventUtilityCohortCollectionIo {
                clock_epoch: 702,
                started_tick_ns: 1,
                finished_tick_ns: 2,
                elapsed_ns: 1,
                written_bytes: 9,
                readback_bytes: 13,
                synced_files: 9,
                synced_directories: 1,
            },
            trial_acknowledgements: (0..2)
                .map(|ordinal| RoventUtilityCohortTrialAcknowledgement {
                    trial_sequence: ordinal + 1,
                    native_trial_sha256: "c".repeat(64),
                    checkpoint: publication.trials[ordinal as usize].clone(),
                    collection_io: RoventUtilityCohortCollectionIo {
                        clock_epoch: 700 + ordinal,
                        started_tick_ns: 1000 - ordinal * 900,
                        finished_tick_ns: 1100 - ordinal * 900,
                        elapsed_ns: 100,
                        written_bytes: 2,
                        readback_bytes: 4,
                        synced_files: 2,
                        synced_directories: 3,
                    },
                })
                .collect(),
        };
        (publication, publication_file, ack)
    }

    #[test]
    fn historical_io_shape_preserves_independent_trial_clock_domains() {
        let (publication, publication_file, ack) = synthetic_io_fixture();
        ack.validate_historical(&publication, &publication_file)
            .unwrap();
        let mut reversed = ack;
        reversed.trial_acknowledgements[0]
            .collection_io
            .finished_tick_ns = 0;
        assert!(
            reversed
                .validate_historical(&publication, &publication_file)
                .is_err()
        );
    }

    #[test]
    fn historical_io_joins_each_original_clock_and_pin_zero() {
        let (mut publication, _, mut ack) = synthetic_io_fixture();
        let mut payloads = BTreeMap::new();
        for ordinal in 0..publication.trials.len() {
            // 원 실행/Rules proof가 없는 명시 합성 IO projection이다.
            // 전체 historical helper나 live trial 권한을 대체하지 않는다.
            let raw = serde_json::to_vec(&serde_json::json!({
                "schema":"rz-rovent-utility-cohort-trial/1",
                "cohort_id":publication.cohort_id,
                "trial_sequence":ordinal as u64+1,
                "cost":{"clock_epoch":700+ordinal as u64,
                    "physical_pin_zero_tick":999-ordinal as u64*900}
            }))
            .unwrap();
            let pin = &mut publication.trials[ordinal];
            pin.bytes = raw.len() as u64;
            pin.sha256 = hex(Digest(Sha256::digest(&raw).into()));
            let entry = &mut ack.trial_acknowledgements[ordinal];
            entry.checkpoint = pin.clone();
            entry.collection_io.written_bytes = pin.bytes;
            entry.collection_io.readback_bytes = 2 * pin.bytes;
            let mut native = Sha256::new();
            native.update(b"rz-rovent-utility-cohort-trial/1\0");
            native.update(pin.bytes.to_le_bytes());
            native.update(&raw);
            entry.native_trial_sha256 = hex(Digest(native.finalize().into()));
            payloads.insert(pin.path.clone(), raw);
        }
        publication.total_payload_bytes =
            6 + publication.trials.iter().map(|p| p.bytes).sum::<u64>();
        ack.collection_io.readback_bytes = publication.total_payload_bytes + ack.publication.bytes;
        ack.collection_io.clock_epoch = 701;
        ack.collection_io.started_tick_ns = 201;
        ack.collection_io.finished_tick_ns = 202;
        ack.validate_original_trial_io(&publication, &payloads)
            .unwrap();

        let mut wrong_clock = ack.clone();
        wrong_clock.trial_acknowledgements[0]
            .collection_io
            .clock_epoch = 701;
        assert!(
            wrong_clock
                .validate_original_trial_io(&publication, &payloads)
                .is_err()
        );
        let mut before_pin_zero = ack.clone();
        before_pin_zero.trial_acknowledgements[0]
            .collection_io
            .started_tick_ns = 998;
        before_pin_zero.trial_acknowledgements[0]
            .collection_io
            .finished_tick_ns = 1098;
        assert!(
            before_pin_zero
                .validate_original_trial_io(&publication, &payloads)
                .is_err()
        );
        let mut before_last_ack = ack;
        before_last_ack.collection_io.started_tick_ns = 199;
        before_last_ack.collection_io.finished_tick_ns = 200;
        assert!(
            before_last_ack
                .validate_original_trial_io(&publication, &payloads)
                .is_err()
        );
    }
}
