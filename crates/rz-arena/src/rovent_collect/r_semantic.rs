//! 원 R publication에 결합하는 Job/slice/선택 record 범위의 소비 관측이다.
//! 공개 DTO의 역사 검증은 live cap·개별 readout 소비·전체 비용·학습 target을
//! 발급하지 않는다. actual producer는 원 Closure와 원 capture owner를 빌린다.

use super::*;
use rz_experiments::rovent_r_readout::{
    R_READOUT_RETIREMENT_DOMAIN, RoventRAuthority, RoventRCaptureEnding, RoventRCaptureRetirement,
};
use rz_search::pals::engine::adaptive_port::{CpuSemanticStage, CpuSemanticWitness};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

pub const ROVENT_R_SEMANTIC_OBSERVATIONS: &str = "rz-rovent-r-semantic-observations/1";
pub const ROVENT_R_SEMANTIC_TASK: &str = "rz-rovent-r-semantic-task/1";
pub const ROVENT_R_SEMANTIC_MAX_BYTES: usize = 4 << 20;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RoventRSemanticStage {
    RawPublished,
    SearchFeedbackConsumed,
    SelectedPcViewConsumed,
}
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoventRSemanticWork {
    pub nodes: u64,
    pub quiescence_nodes: u64,
    pub tt_hits: u64,
}
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoventRSemanticSelectedPc {
    pub request_epoch: u64,
    pub request_sequence: u64,
    pub question_sha256: String,
    pub view_sha256: String,
    pub full_view_sha256: String,
    pub output_sha256: String,
    pub record_revision: u64,
    pub record_origin_state: u64,
}
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoventRSemanticObservation {
    pub retirement_sha256: String,
    pub stage: RoventRSemanticStage,
    pub native_execution_epoch: u64,
    pub native_execution_sequence: u64,
    pub store_execution: u64,
    pub observation: u64,
    /// Original exact serde TaskKey body, never reconstructed from CpuJob.
    pub task_key: Value,
    pub task_key_sha256: String,
    pub job_sequence: u64,
    pub worker_owner: u16,
    pub frame_sequence: u64,
    pub capture_sequence: u64,
    pub question_sha256: String,
    pub binding_sha256: String,
    pub submitted_view_sha256: String,
    pub submitted_full_view_sha256: String,
    /// Exactly one attested native slice, never summed over semantic stages.
    pub native_slice_work: RoventRSemanticWork,
    pub selected_pc: Option<RoventRSemanticSelectedPc>,
}
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoventRSemanticRetirement {
    pub retirement_sha256: String,
    pub retirement: RoventRCaptureRetirement,
    pub readout_record_sha256: Vec<String>,
}
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoventRSemanticFailure {
    pub stage: String,
    pub code: String,
    pub detail: String,
}
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoventRSemanticObservations {
    pub schema: String,
    pub r_publication_file_sha256: String,
    pub native_admission_sha256: String,
    pub authority: RoventRAuthority,
    pub retirements: Vec<RoventRSemanticRetirement>,
    /// Original observation delivery order; selected P/C facts follow their
    /// actual RawPublished and SearchFeedbackConsumed predecessors.
    pub observations: Vec<RoventRSemanticObservation>,
    pub unrecorded_witness_attempts: u64,
    pub manager_failures: Vec<RoventRSemanticFailure>,
    pub per_readout_consumption: bool,
    pub whole_action_cost: Option<Value>,
    pub wdl_target: Option<Value>,
    pub utility_target: Option<Value>,
    pub semantic_sha256: String,
}

fn authority_wire(authority: rz_contracts::pals::SearchAuthority) -> RoventRAuthority {
    RoventRAuthority {
        process_epoch: authority.epoch.0,
        game_generation: authority.game.0,
        root_generation: authority.root.0,
        implementation_sha256: hex(authority.implementation),
    }
}

/// Publication's byte SHA is a caller-owned durable-file pin. Its filesystem
/// sync/readback ACK remains the upper publisher's separate responsibility.
/// No DTO can substitute the actual closure/capture capabilities here.
pub fn capture_rovent_r_semantic_observations(
    closure: &RoventRCollectionClosure,
    captures: &[CapturedRoventRRetirement],
    publication: &RoventRCorpusPublication,
    publication_file_sha256: &str,
) -> Result<RoventRSemanticObservations, ArenaError> {
    check(
        generation_sha(publication_file_sha256)
            && captures.len() <= 64
            && publication.retirements.len() == captures.len(),
        "R semantic publication/capture scope bound",
    )?;
    publication.validate()?;
    let admission = closure.admission();
    check(
        publication
            .admissions
            .iter()
            .any(|origin| origin.native_admission_sha256 == admission.raw_sha256()),
        "R semantic publication lacks the original admission",
    )?;
    let mut retired = Vec::with_capacity(captures.len());
    let mut originals = Vec::with_capacity(captures.len());
    for captured in captures {
        let held = closure
            .captures()
            .iter()
            .find(|held| captured.matches_reservation(held.reservation()))
            .ok_or_else(|| invalid("R publication capture has no same original reservation"))?;
        let original = held
            .retirement()
            .ok_or_else(|| invalid("R semantic capture has no actual original terminal"))?;
        check(
            held.reservation().matches_retirement(original)
                && original.matches_model(closure.nnuec_source().model())
                && Arc::ptr_eq(captured.nnuec_source(), closure.nnuec_source())
                && Arc::ptr_eq(
                    captured.admission().root_admission(),
                    admission.root_admission(),
                )
                && captured.admission().raw_sha256() == admission.raw_sha256()
                && captured.retirement().job.job_sequence == original.job().id.sequence
                && captured.retirement().capture_sequence == original.capture_sequence()
                && captured.retirement().frame_sequence == original.frame().sequence
                && captured.retirement().worker_owner == original.owner().0,
            "R semantic original source/admission/Job/frame/capture mismatch",
        )?;
        check(
            !originals.iter().any(
                |(previous, _): &(
                    _,
                    &rz_search::cpu_value::nnuec::v2::capture::NnuecCaptureRetirement,
                )| { *previous == captured.retirement_sha256() },
            ),
            "R semantic repeated original retirement",
        )?;
        let mut readouts: Vec<_> = captured
            .rows()
            .iter()
            .map(|row| row.record().record_sha256.clone())
            .collect();
        readouts.sort();
        let declared = publication
            .retirements
            .iter()
            .find(|retirement| retirement.retirement_sha256 == captured.retirement_sha256())
            .ok_or_else(|| invalid("R semantic retirement absent from publication"))?;
        check(
            declared.readout_record_sha256 == readouts,
            "R semantic complete original readout coverage differs from publication",
        )?;
        retired.push(RoventRSemanticRetirement {
            retirement_sha256: captured.retirement_sha256().into(),
            retirement: captured.retirement().clone(),
            readout_record_sha256: readouts,
        });
        originals.push((captured.retirement_sha256(), original));
    }
    // Every original terminal is included, including empty/failed/NotStarted.
    // No row is fabricated for a reservation with no physical terminal.
    check(
        closure
            .captures()
            .iter()
            .filter(|held| held.retirement().is_some())
            .count()
            == originals.len(),
        "R semantic publication omits an original terminal",
    )?;
    let mut observations = Vec::with_capacity(closure.witnesses().len());
    for witness in closure.witnesses() {
        let (sha, original) = originals
            .iter()
            .find(|(_, original)| original.matches_lineage(witness.capture_lineage()))
            .ok_or_else(|| {
                invalid("R semantic witness has no same activated original retirement")
            })?;
        check(
            witness.authority() == admission.authority()
                && witness.job() == original.job()
                && witness.owner() == original.owner()
                && witness.frame() == original.frame()
                && witness.question() == original.question_digest()
                && witness.job().hard_deadline.clock
                    == admission.root_admission().hard_deadline().clock
                && witness.job().hard_deadline.at <= admission.root_admission().hard_deadline().at
                && witness.capture_lineage().view_seal_digest()
                    == witness.submitted_view().view_digest
                && witness.capture_lineage().full_view_digest()
                    == witness.submitted_full_view_digest(),
            "R semantic witness differs from original Job/frame/Q/view/clock",
        )?;
        observations.push(observation_wire(sha, witness)?);
    }
    let failures = closure
        .original_retirement()
        .failures()
        .iter()
        .map(|failure| RoventRSemanticFailure {
            stage: format!("{:?}", failure.stage),
            code: failure.code.into(),
            detail: failure.detail.clone(),
        })
        .collect();
    let mut result = RoventRSemanticObservations {
        schema: ROVENT_R_SEMANTIC_OBSERVATIONS.into(),
        r_publication_file_sha256: publication_file_sha256.into(),
        native_admission_sha256: admission.raw_sha256().into(),
        authority: authority_wire(admission.authority()),
        retirements: retired,
        observations,
        unrecorded_witness_attempts: closure.original_retirement().unrecorded_witness_attempts(),
        manager_failures: failures,
        per_readout_consumption: false,
        whole_action_cost: None,
        wdl_target: None,
        utility_target: None,
        semantic_sha256: String::new(),
    };
    result.semantic_sha256 = result.digest()?;
    result.validate_historical(publication, publication_file_sha256)?;
    Ok(result)
}

fn observation_wire(
    sha: &str,
    witness: &CpuSemanticWitness,
) -> Result<RoventRSemanticObservation, ArenaError> {
    let selected_pc = match witness.stage() {
        CpuSemanticStage::SelectedPcViewConsumed => {
            let ticket = witness
                .selected_pc_ticket()
                .ok_or_else(|| invalid("selected P/C semantic ticket missing"))?;
            let (record_revision, record_origin_state) = witness
                .selected_record()
                .ok_or_else(|| invalid("selected semantic record missing"))?;
            check(
                ticket.authority == witness.authority(),
                "selected P/C semantic authority mismatch",
            )?;
            Some(RoventRSemanticSelectedPc {
                request_epoch: ticket.id.epoch.0,
                request_sequence: ticket.id.sequence,
                question_sha256: hex(ticket.question),
                view_sha256: hex(ticket.view.view_digest),
                full_view_sha256: hex(witness
                    .selected_full_view_digest()
                    .ok_or_else(|| invalid("selected full view missing"))?),
                output_sha256: hex(witness
                    .selected_pc_output_digest()
                    .ok_or_else(|| invalid("selected output missing"))?),
                record_revision,
                record_origin_state: record_origin_state.0 as u64,
            })
        }
        _ => None,
    };
    let work = witness.attested_slice_work();
    Ok(RoventRSemanticObservation {
        retirement_sha256: sha.into(),
        stage: match witness.stage() {
            CpuSemanticStage::RawPublished => RoventRSemanticStage::RawPublished,
            CpuSemanticStage::SearchFeedbackConsumed => {
                RoventRSemanticStage::SearchFeedbackConsumed
            }
            CpuSemanticStage::SelectedPcViewConsumed => {
                RoventRSemanticStage::SelectedPcViewConsumed
            }
        },
        native_execution_epoch: witness.native_execution().epoch.0,
        native_execution_sequence: witness.native_execution().sequence,
        store_execution: witness.store_execution().0 as u64,
        observation: witness.observation().0 as u64,
        task_key: serde_json::to_value(witness.task_key())
            .map_err(|error| invalid(error.to_string()))?,
        task_key_sha256: generation_digest(ROVENT_R_SEMANTIC_TASK, witness.task_key())?,
        job_sequence: witness.job().id.sequence,
        worker_owner: witness.owner().0,
        frame_sequence: witness.frame().sequence,
        capture_sequence: witness.capture_lineage().capture_sequence(),
        question_sha256: hex(witness.question()),
        binding_sha256: hex(witness.job().evaluation_namespace),
        submitted_view_sha256: hex(witness.submitted_view().view_digest),
        submitted_full_view_sha256: hex(witness.submitted_full_view_digest()),
        native_slice_work: RoventRSemanticWork {
            nodes: work.nodes,
            quiescence_nodes: work.quiescence_nodes,
            tt_hits: work.tt_hits,
        },
        selected_pc,
    })
}

impl RoventRSemanticObservations {
    fn digest(&self) -> Result<String, ArenaError> {
        let mut original = self.clone();
        original.semantic_sha256.clear();
        Ok(generation_digest(
            ROVENT_R_SEMANTIC_OBSERVATIONS,
            &original,
        )?)
    }
    /// Metadata-only verification. The caller separately verifies original
    /// publication file bytes and complete R Rules/history/source evidence.
    pub fn validate_historical(
        &self,
        publication: &RoventRCorpusPublication,
        publication_file_sha256: &str,
    ) -> Result<(), ArenaError> {
        check(
            self.schema == ROVENT_R_SEMANTIC_OBSERVATIONS
                && generation_sha(publication_file_sha256)
                && self.r_publication_file_sha256 == publication_file_sha256
                && generation_sha(&self.native_admission_sha256)
                && self.authority.process_epoch != 0
                && generation_sha(&self.authority.implementation_sha256)
                && self.retirements.len() <= 64
                && self.retirements.len() == publication.retirements.len()
                && self.observations.len() <= 256
                && self.manager_failures.len() <= 64
                && !self.per_readout_consumption
                && self.whole_action_cost.is_none()
                && self.wdl_target.is_none()
                && self.utility_target.is_none(),
            "R semantic strict scope/size/authority or unsupported target claim",
        )?;
        publication.validate()?;
        let admission = publication
            .admissions
            .iter()
            .find(|origin| origin.native_admission_sha256 == self.native_admission_sha256)
            .ok_or_else(|| invalid("R semantic admission absent from original publication"))?;
        let model_identity: rz_search::cpu_value::CpuValueIdentity =
            serde_json::from_value(admission.nnuec_registration["model_identity"].clone())
                .map_err(|error| invalid(error.to_string()))?;
        let mut retired = BTreeMap::new();
        for terminal in &self.retirements {
            let declared = publication
                .retirements
                .iter()
                .find(|original| original.retirement_sha256 == terminal.retirement_sha256)
                .ok_or_else(|| invalid("R semantic original retirement absent"))?;
            check(
                terminal.retirement.job.authority == self.authority
                    && terminal.retirement.job.hard_deadline_clock_epoch
                        == self.authority.process_epoch
                    && generation_digest(R_READOUT_RETIREMENT_DOMAIN, &terminal.retirement)?
                        == terminal.retirement_sha256
                    && terminal.readout_record_sha256 == declared.readout_record_sha256
                    && terminal.readout_record_sha256.len()
                        == usize::from(terminal.retirement.retained_readouts)
                    && retired
                        .insert(terminal.retirement_sha256.as_str(), terminal)
                        .is_none(),
                "R semantic complete retirement/readout identity mismatch",
            )?;
        }
        let mut executions: BTreeMap<(u64, u64), (&RoventRSemanticObservation, u8)> =
            BTreeMap::new();
        let mut seen = BTreeSet::new();
        for observation in &self.observations {
            let terminal = retired
                .get(observation.retirement_sha256.as_str())
                .ok_or_else(|| invalid("R semantic observation has no original retirement"))?;
            let job = &terminal.retirement.job;
            let task_bytes = serde_json::to_vec(&observation.task_key)
                .map_err(|error| invalid(error.to_string()))?;
            check(
                task_bytes.len() <= 64 << 10,
                "R semantic original TaskKey byte cap",
            )?;
            let task: rz_search::pals::store::TaskKey =
                serde_json::from_slice(&task_bytes).map_err(|error| invalid(error.to_string()))?;
            check(
                serde_json::to_value(&task).map_err(|error| invalid(error.to_string()))?
                    == observation.task_key
                    && generation_digest(ROVENT_R_SEMANTIC_TASK, &observation.task_key)?
                        == observation.task_key_sha256
                    && task.node_budget == job.total_native_node_credit
                    && task.value_identity.as_ref() == Some(&model_identity)
                    && task.checker_identity.is_none()
                    && task.line.is_none()
                    && task.model == 0
                    && task.epoch == 0,
                "R semantic exact original TaskKey body/digest",
            )?;
            check(
                !matches!(terminal.retirement.ending, RoventRCaptureEnding::NotStarted)
                    && observation.native_execution_epoch == self.authority.process_epoch
                    && observation.native_execution_sequence > 0
                    && observation.job_sequence == job.job_sequence
                    && observation.worker_owner == terminal.retirement.worker_owner
                    && observation.frame_sequence == terminal.retirement.frame_sequence
                    && observation.capture_sequence == terminal.retirement.capture_sequence
                    && observation.question_sha256 == job.question_sha256
                    && observation.binding_sha256 == job.evaluation_namespace_sha256
                    && observation.native_slice_work.quiescence_nodes
                        <= observation.native_slice_work.nodes
                    && observation.native_slice_work.nodes <= job.total_native_node_credit
                    && [
                        &observation.task_key_sha256,
                        &observation.submitted_view_sha256,
                        &observation.submitted_full_view_sha256,
                    ]
                    .iter()
                    .all(|sha| generation_sha(sha)),
                "R semantic Job/frame/capture/clock/work mismatch",
            )?;
            let key = (
                observation.native_execution_epoch,
                observation.native_execution_sequence,
            );
            let (first, stages) = executions.entry(key).or_insert((observation, 0));
            check(
                first.retirement_sha256 == observation.retirement_sha256
                    && first.store_execution == observation.store_execution
                    && first.observation == observation.observation
                    && first.task_key_sha256 == observation.task_key_sha256
                    && first.submitted_view_sha256 == observation.submitted_view_sha256
                    && first.submitted_full_view_sha256 == observation.submitted_full_view_sha256
                    && first.native_slice_work == observation.native_slice_work,
                "R semantic stage repeats native execution with different original evidence/work",
            )?;
            let stage = match observation.stage {
                RoventRSemanticStage::RawPublished => {
                    check(
                        *stages == 0 && observation.selected_pc.is_none(),
                        "R raw semantic ordering",
                    )?;
                    1
                }
                RoventRSemanticStage::SearchFeedbackConsumed => {
                    check(
                        *stages == 1 && observation.selected_pc.is_none(),
                        "R feedback requires earlier raw fact",
                    )?;
                    2
                }
                RoventRSemanticStage::SelectedPcViewConsumed => {
                    check(
                        *stages & 2 != 0,
                        "R selected P/C requires original feedback consumption",
                    )?;
                    let selected = observation
                        .selected_pc
                        .as_ref()
                        .ok_or_else(|| invalid("R selected P/C evidence missing"))?;
                    check(
                        selected.request_epoch == self.authority.process_epoch
                            && selected.request_sequence > 0
                            && [
                                &selected.question_sha256,
                                &selected.view_sha256,
                                &selected.full_view_sha256,
                                &selected.output_sha256,
                            ]
                            .iter()
                            .all(|sha| generation_sha(sha)),
                        "R selected P/C identity",
                    )?;
                    4
                }
            };
            let selected = observation
                .selected_pc
                .as_ref()
                .map(|value| (value.request_epoch, value.request_sequence));
            check(
                seen.insert((key, stage, selected)),
                "R duplicate semantic occurrence",
            )?;
            *stages |= stage;
        }
        for failure in &self.manager_failures {
            check(
                matches!(
                    failure.stage.as_str(),
                    "Dispatch" | "NativeEvent" | "Accepted" | "Abandoned" | "Settled"
                ) && !failure.code.is_empty()
                    && failure.code.len() <= 128
                    && failure.detail.len() <= 2048,
                "R semantic bounded original failure",
            )?;
        }
        check(
            self.semantic_sha256 == self.digest()?,
            "R semantic observations digest mismatch",
        )?;
        let size = serde_json::to_vec(self)
            .map_err(|error| invalid(error.to_string()))?
            .len();
        check(
            size <= ROVENT_R_SEMANTIC_MAX_BYTES,
            "R semantic observations byte cap",
        )
    }
    pub fn from_json_historical(
        input: &str,
        publication: &RoventRCorpusPublication,
        publication_file_sha256: &str,
    ) -> Result<Self, ArenaError> {
        check(
            input.len() <= ROVENT_R_SEMANTIC_MAX_BYTES,
            "R semantic input byte cap",
        )?;
        let result: Self = rz_experiments::decode_json(input)?;
        result.validate_historical(publication, publication_file_sha256)?;
        Ok(result)
    }
}
