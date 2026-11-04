//! 원 post-fence 미제출 또는 실제 NotStarted cap의 별도 역사 관측이다.
//! terminal 부재/activation 부재만으로 미시작을 추정하지 않으며, Job·row·
//! 사용량 0·전체 비용·WDL·utility target을 합성하지 않는다.

use super::*;
use rz_experiments::rovent_r_readout::{
    R_READOUT_RETIREMENT_DOMAIN, RoventRAuthority, RoventRCaptureEnding, RoventRCaptureRetirement,
    RoventRJob,
};
use rz_search::cpu_value::nnuec::v2::capture::{NnuecCaptureEnding, NnuecCaptureReservation};
use rz_uci::pals_adaptive::ROVENT_COLLECTION_ROOT_ADMISSION;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

pub const ROVENT_R_UNSUBMITTED_OBSERVATIONS: &str = "rz-rovent-r-unsubmitted-observations/1";
pub const ROVENT_R_UNSUBMITTED_MAX_BYTES: usize = 1 << 20;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RoventRUnsubmittedKind {
    /// Actual manager post-fence custody, with no prepared/enqueued Job.
    Unsubmitted,
    /// Actual original worker's verified-unqueued terminal, not absence.
    NotStarted,
}
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoventRUnsubmittedObservation {
    pub kind: RoventRUnsubmittedKind,
    pub frame_sequence: u64,
    pub question_sha256: String,
    pub full_view_sha256: String,
    pub view_seal_sha256: String,
    pub binding_sha256: String,
    pub capture_sequence: u64,
    pub original_deadline_clock_epoch: u64,
    pub original_deadline_ns: u64,
    pub job: Option<RoventRJob>,
    pub worker_owner: Option<u16>,
    pub retirement_sha256: Option<String>,
    pub retirement: Option<RoventRCaptureRetirement>,
    pub activation_observed: bool,
}
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoventRUnsubmittedObservations {
    pub schema: String,
    pub native_admission: Value,
    pub native_admission_sha256: String,
    pub input_lock_sha256: String,
    pub nnuec_weights_raw_sha256: String,
    pub nnuec_parameter_sha256: String,
    pub authority: RoventRAuthority,
    pub r_publication_file_sha256: Option<String>,
    pub observations: Vec<RoventRUnsubmittedObservation>,
    pub per_readout_consumption: bool,
    pub native_node_usage: Option<Value>,
    pub whole_action_cost: Option<Value>,
    pub wdl_target: Option<Value>,
    pub utility_target: Option<Value>,
    pub observations_sha256: String,
}

fn observation(
    original: &NnuecCaptureReservation,
    kind: RoventRUnsubmittedKind,
    job: Option<RoventRJob>,
    worker_owner: Option<u16>,
    retirement_sha256: Option<String>,
    retirement: Option<RoventRCaptureRetirement>,
) -> RoventRUnsubmittedObservation {
    let lineage = original.lineage();
    RoventRUnsubmittedObservation {
        kind,
        frame_sequence: lineage.frame().sequence,
        question_sha256: hex(lineage.question_digest()),
        full_view_sha256: hex(lineage.full_view_digest()),
        view_seal_sha256: hex(lineage.view_seal_digest()),
        binding_sha256: hex(lineage.binding_digest()),
        capture_sequence: lineage.capture_sequence(),
        original_deadline_clock_epoch: lineage.original_deadline().clock.0.0,
        original_deadline_ns: lineage.original_deadline().at.0,
        job,
        worker_owner,
        retirement_sha256,
        retirement,
        activation_observed: lineage.activation_observed(),
    }
}

/// Only original root-closure capabilities are projected. The unsubmitted
/// branch additionally requires the actual manager's retained move-only
/// custody; a public delivery enum or a missing terminal alone is insufficient.
pub fn capture_rovent_r_unsubmitted_observations(
    closure: &RoventRCollectionClosure,
    captured: &[CapturedRoventRRetirement],
    r_publication_file_sha256: Option<&str>,
) -> Result<Option<RoventRUnsubmittedObservations>, ArenaError> {
    let admission = closure.admission();
    let mut observations = Vec::new();
    let mut witnessed_unsubmitted = 0;
    for held in closure.captures() {
        let original = held.reservation();
        let lineage = original.lineage();
        check(
            original.matches_model(closure.nnuec_source().model())
                && lineage.frame().authority == admission.authority(),
            "R absence original loaded model mismatch",
        )?;
        if let Some(terminal) = original.retirement() {
            if terminal.ending() != NnuecCaptureEnding::NotStarted {
                continue;
            }
            let source = captured
                .iter()
                .find(|source| source.matches_reservation(original))
                .ok_or_else(|| {
                    invalid("NotStarted observation lacks original captured terminal")
                })?;
            check(
                original.matches_retirement(terminal)
                    && !lineage.activation_observed()
                    && Arc::ptr_eq(source.nnuec_source(), closure.nnuec_source())
                    && Arc::ptr_eq(
                        source.admission().root_admission(),
                        admission.root_admission(),
                    )
                    && matches!(source.retirement().ending, RoventRCaptureEnding::NotStarted)
                    && source.retirement().job.job_sequence == terminal.job().id.sequence
                    && source.retirement().frame_sequence == lineage.frame().sequence
                    && source.retirement().capture_sequence == lineage.capture_sequence()
                    && source.retirement().worker_owner == terminal.owner().0,
                "NotStarted observation differs from original verified-unqueued terminal",
            )?;
            check(
                r_publication_file_sha256.is_some(),
                "NotStarted requires its separate original R terminal publication",
            )?;
            observations.push(observation(
                original,
                RoventRUnsubmittedKind::NotStarted,
                Some(source.retirement().job.clone()),
                Some(terminal.owner().0),
                Some(source.retirement_sha256().into()),
                Some(source.retirement().clone()),
            ));
            continue;
        }
        let custody = closure
            .original_retirement()
            .unsubmitted_captures()
            .iter()
            .find(|custody| custody.reservation().matches_lineage(&lineage))
            .ok_or_else(|| invalid("missing terminal is not actual manager unsubmitted custody"))?;
        check(
            held.delivery() == RoventRCaptureDelivery::UnsubmittedObserved
                && !custody.was_enqueued()
                && custody.job().is_none()
                && custody.reservation().retirement().is_none()
                && !lineage.activation_observed()
                && custody.frame() == lineage.frame()
                && custody.question() == lineage.question_digest(),
            "R unsubmitted observation lacks original not-enqueued/activation-free custody",
        )?;
        witnessed_unsubmitted += 1;
        observations.push(observation(
            original,
            RoventRUnsubmittedKind::Unsubmitted,
            None,
            None,
            None,
            None,
        ));
    }
    check(
        witnessed_unsubmitted == closure.original_retirement().unsubmitted_captures().len(),
        "R absence projection omits original unsubmitted custody",
    )?;
    if observations.is_empty() {
        return Ok(None);
    }
    let authority = admission.authority();
    let mut result = RoventRUnsubmittedObservations {
        schema: ROVENT_R_UNSUBMITTED_OBSERVATIONS.into(),
        native_admission: admission.raw_receipt().clone(),
        native_admission_sha256: admission.raw_sha256().into(),
        input_lock_sha256: hex(closure.nnuec_source().input_lock_digest()),
        nnuec_weights_raw_sha256: closure.nnuec_source().weights_artifact().sha256.clone(),
        nnuec_parameter_sha256: closure
            .nnuec_source()
            .semantic_weights_identity()
            .weights_sha256
            .clone()
            .ok_or_else(|| invalid("R absence semantic weights identity missing"))?,
        authority: RoventRAuthority {
            process_epoch: authority.epoch.0,
            game_generation: authority.game.0,
            root_generation: authority.root.0,
            implementation_sha256: hex(authority.implementation),
        },
        r_publication_file_sha256: r_publication_file_sha256.map(str::to_owned),
        observations,
        per_readout_consumption: false,
        native_node_usage: None,
        whole_action_cost: None,
        wdl_target: None,
        utility_target: None,
        observations_sha256: String::new(),
    };
    result.observations_sha256 = result.digest()?;
    result.validate_common(admission.raw_sha256())?;
    Ok(Some(result))
}

impl RoventRUnsubmittedObservations {
    fn digest(&self) -> Result<String, ArenaError> {
        let mut value = self.clone();
        value.observations_sha256.clear();
        Ok(generation_digest(
            ROVENT_R_UNSUBMITTED_OBSERVATIONS,
            &value,
        )?)
    }
    fn validate_common(&self, expected_admission_sha256: &str) -> Result<(), ArenaError> {
        check(
            self.schema == ROVENT_R_UNSUBMITTED_OBSERVATIONS
                && self.authority.process_epoch != 0
                && generation_sha(&self.authority.implementation_sha256)
                && generation_sha(expected_admission_sha256)
                && self.native_admission_sha256 == expected_admission_sha256
                && self.native_admission["schema"] == ROVENT_COLLECTION_ROOT_ADMISSION
                && generation_digest(ROVENT_COLLECTION_ROOT_ADMISSION, &self.native_admission)?
                    == self.native_admission_sha256
                && self.native_admission["process_epoch"].as_u64()
                    == Some(self.authority.process_epoch)
                && self.native_admission["game_generation"].as_u64()
                    == Some(self.authority.game_generation)
                && self.native_admission["root_generation"].as_u64()
                    == Some(self.authority.root_generation)
                && self.native_admission["implementation_sha256"]
                    == self.authority.implementation_sha256
                && self.native_admission["native_origin"]["input_lock_sha256"]
                    == self.input_lock_sha256
                && [
                    &self.input_lock_sha256,
                    &self.nnuec_weights_raw_sha256,
                    &self.nnuec_parameter_sha256,
                ]
                .iter()
                .all(|value| generation_sha(value))
                && (1..=64).contains(&self.observations.len())
                && self
                    .r_publication_file_sha256
                    .as_ref()
                    .is_none_or(|value| generation_sha(value))
                && !self.per_readout_consumption
                && self.native_node_usage.is_none()
                && self.whole_action_cost.is_none()
                && self.wdl_target.is_none()
                && self.utility_target.is_none(),
            "R unsubmitted strict domain/source/root/no-target scope",
        )?;
        let mut sequences = BTreeSet::new();
        for value in &self.observations {
            check(
                !value.activation_observed
                    && value.frame_sequence > 0
                    && value.capture_sequence > 0
                    && sequences.insert(value.capture_sequence)
                    && [
                        &value.question_sha256,
                        &value.full_view_sha256,
                        &value.view_seal_sha256,
                        &value.binding_sha256,
                    ]
                    .iter()
                    .all(|value| generation_sha(value))
                    && value.original_deadline_clock_epoch == self.authority.process_epoch
                    && self.native_admission["deadline_clock"].as_u64()
                        == Some(value.original_deadline_clock_epoch)
                    && self.native_admission["deadline_tick"]
                        .as_u64()
                        .is_some_and(|deadline| value.original_deadline_ns <= deadline),
                "R unsubmitted original frame/question/input/deadline/activation mismatch",
            )?;
            match value.kind {
                RoventRUnsubmittedKind::Unsubmitted => check(
                    value.job.is_none()
                        && value.worker_owner.is_none()
                        && value.retirement_sha256.is_none()
                        && value.retirement.is_none(),
                    "unsubmitted absence cannot invent a Job/owner/terminal",
                )?,
                RoventRUnsubmittedKind::NotStarted => {
                    let job = value
                        .job
                        .as_ref()
                        .ok_or_else(|| invalid("NotStarted original Job missing"))?;
                    let terminal = value
                        .retirement
                        .as_ref()
                        .ok_or_else(|| invalid("NotStarted original retirement body missing"))?;
                    check(
                        value.worker_owner.is_some()
                            && value
                                .retirement_sha256
                                .as_ref()
                                .is_some_and(|sha| generation_sha(sha))
                            && self.r_publication_file_sha256.is_some()
                            && job.authority == self.authority
                            && job.question_sha256 == value.question_sha256
                            && job.evaluation_namespace_sha256 == value.binding_sha256
                            && job.hard_deadline_clock_epoch == value.original_deadline_clock_epoch
                            && job.hard_deadline_ns <= value.original_deadline_ns,
                        "NotStarted requires exact original Job/owner/terminal metadata",
                    )?;
                    check(
                        matches!(terminal.ending, RoventRCaptureEnding::NotStarted)
                            && terminal.job == *job
                            && Some(terminal.worker_owner) == value.worker_owner
                            && terminal.frame_sequence == value.frame_sequence
                            && terminal.capture_sequence == value.capture_sequence
                            && terminal.retained_readouts == 0
                            && Some(generation_digest(R_READOUT_RETIREMENT_DOMAIN, terminal)?)
                                == value.retirement_sha256,
                        "NotStarted original terminal body/digest mismatch",
                    )?;
                }
            }
        }
        check(
            self.observations_sha256 == self.digest()?,
            "R unsubmitted observations digest mismatch",
        )?;
        check(
            serde_json::to_vec(self)
                .map_err(|error| invalid(error.to_string()))?
                .len()
                <= ROVENT_R_UNSUBMITTED_MAX_BYTES,
            "R unsubmitted observations byte cap",
        )
    }
    /// Historical metadata only. Original root/source/physical authority is
    /// never restored from this JSON or from its absence observations.
    pub fn validate_historical(
        &self,
        expected_admission_sha256: &str,
        publication: Option<&RoventRCorpusPublication>,
        expected_publication_file_sha256: Option<&str>,
    ) -> Result<(), ArenaError> {
        self.validate_common(expected_admission_sha256)?;
        check(
            self.r_publication_file_sha256.as_deref() == expected_publication_file_sha256
                && publication.is_some() == expected_publication_file_sha256.is_some(),
            "R unsubmitted original optional publication mismatch",
        )?;
        if let Some(publication) = publication {
            publication.validate()?;
            check(
                publication.admissions.iter().any(|admission| {
                    admission.native_admission_sha256 == self.native_admission_sha256
                }),
                "R unsubmitted publication original admission mismatch",
            )?;
            for observation in &self.observations {
                if let Some(sha) = &observation.retirement_sha256 {
                    let retired = publication
                        .retirements
                        .iter()
                        .find(|retired| &retired.retirement_sha256 == sha)
                        .ok_or_else(|| {
                            invalid("NotStarted terminal absent from original R publication")
                        })?;
                    check(
                        retired.readout_record_sha256.is_empty(),
                        "NotStarted cannot contain neural readout rows",
                    )?;
                }
            }
        }
        Ok(())
    }
    pub fn from_json_historical(
        input: &str,
        expected_admission_sha256: &str,
        publication: Option<&RoventRCorpusPublication>,
        expected_publication_file_sha256: Option<&str>,
    ) -> Result<Self, ArenaError> {
        check(
            input.len() <= ROVENT_R_UNSUBMITTED_MAX_BYTES,
            "R unsubmitted JSON byte cap",
        )?;
        let result: Self = rz_experiments::decode_json(input)?;
        result.validate_historical(
            expected_admission_sha256,
            publication,
            expected_publication_file_sha256,
        )?;
        Ok(result)
    }
}
