//! Actual R observation stages. This module is registered separately from the
//! preceding execution checkpoint. None of these capabilities supplies a game
//! result, a complete action cost, or per-readout P/C consumption.
use super::*;
use crate::cpu::CpuWork;
use crate::cpu_value::nnuec::v2::capture::NnuecCaptureLineage;
use crate::cpu_worker::{CpuSliceReport, SettledCpuSlice};
use crate::pals::coordinator::AcceptedRoleStep;
use crate::pals::store::StoreReadOwner;
use rz_contracts::pals_adaptive::{CpuJobEnvelope, CpuWorkerId};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CpuSemanticStage {
    RawPublished,
    SearchFeedbackConsumed,
    SelectedPcViewConsumed,
}

struct CpuSemanticOrigin {
    store: StoreReadOwner,
    generation: u64,
    root: StateId,
    authority: SearchAuthority,
    frame: FrameId,
    question: ContractDigest,
    view: ReviewViewSeal,
    full_view: ContractDigest,
    job: CpuJobEnvelope,
    owner: CpuWorkerId,
    task: TaskKey,
    execution: ExecutionId,
    observation: ObservationId,
    settlement: SettledCpuSlice,
    capture: NnuecCaptureLineage,
}

#[derive(Clone)]
struct SelectedPcConsumption {
    ticket: RoleTicketEnvelope,
    output: ContractDigest,
    record_revision: u64,
    record_origin: StateId,
    full_view: ContractDigest,
}

/// The original semantic owner is the sole constructor. Copies share original
/// immutable evidence; they neither repeat a worker execution nor add work.
/// The origin contains no ReviewView, Bound, model, or source-input Arc.
#[derive(Clone)]
pub struct CpuSemanticWitness {
    origin: Arc<CpuSemanticOrigin>,
    stage: CpuSemanticStage,
    selected: Option<SelectedPcConsumption>,
}
impl CpuSemanticWitness {
    pub fn stage(&self) -> CpuSemanticStage {
        self.stage
    }
    pub fn authority(&self) -> SearchAuthority {
        self.origin.authority
    }
    pub fn frame(&self) -> FrameId {
        self.origin.frame
    }
    pub fn question(&self) -> ContractDigest {
        self.origin.question
    }
    pub fn submitted_view(&self) -> ReviewViewSeal {
        self.origin.view
    }
    pub fn submitted_full_view_digest(&self) -> ContractDigest {
        self.origin.full_view
    }
    pub fn job(&self) -> CpuJobEnvelope {
        self.origin.job
    }
    pub fn owner(&self) -> CpuWorkerId {
        self.origin.owner
    }
    pub fn task_key(&self) -> &TaskKey {
        &self.origin.task
    }
    pub fn store_execution(&self) -> ExecutionId {
        self.origin.execution
    }
    /// Physical grant identity and Store task identity are different domains.
    pub fn native_execution(&self) -> rz_contracts::ExecutionId {
        self.origin.settlement.grant().execution
    }
    pub fn observation(&self) -> ObservationId {
        self.origin.observation
    }
    pub fn capture_lineage(&self) -> &NnuecCaptureLineage {
        &self.origin.capture
    }
    /// One attested native slice, never preparation/queue/drain or total cost.
    pub fn attested_slice_work(&self) -> CpuWork {
        self.origin.settlement.actual_work()
    }
    pub fn selected_pc_ticket(&self) -> Option<&RoleTicketEnvelope> {
        self.selected.as_ref().map(|selected| &selected.ticket)
    }
    pub fn selected_pc_output_digest(&self) -> Option<ContractDigest> {
        self.selected.as_ref().map(|selected| selected.output)
    }
    pub fn selected_record(&self) -> Option<(u64, StateId)> {
        self.selected
            .as_ref()
            .map(|selected| (selected.record_revision, selected.record_origin))
    }
    pub fn selected_full_view_digest(&self) -> Option<ContractDigest> {
        self.selected.as_ref().map(|selected| selected.full_view)
    }
}

impl<M: RoleModel> AdaptiveEnginePort<M> {
    /// Readback only: a later projection/continuation error must not erase an
    /// already published observation. This issues no semantic witness itself.
    pub fn cpu_semantic_publication_recorded(
        &self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        report: &CpuSliceReport,
    ) -> bool {
        self.cpu_admissions
            .get(&work.input.question_digest)
            .is_some_and(|admission| {
                admission.frame == work.frame
                    && admission.stage == work.input.stage
                    && admission.view_seal == work.input.view_seal
                    && admission
                        .committed_execution
                        .is_some_and(|(physical, execution, _)| {
                            physical == report.grant.execution
                                && admission.settled_receipt
                                    == Some((
                                        physical,
                                        execution,
                                        cpu_commit::cpu_receipt_digest(report),
                                    ))
                        })
            })
    }

    /// Invoke after the actual CPU Store commit while its original admission
    /// is retained. A settled report alone cannot enter this factory. The
    /// returned bounded pair contains no WDL, utility, or whole-cost target.
    pub fn cpu_semantic_witnesses(
        &self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        report: &CpuSliceReport,
        settled: &SettledCpuSlice,
        capture: &NnuecCaptureLineage,
    ) -> Result<Vec<CpuSemanticWitness>, PalsError> {
        AdaptivePortQuestion::validate_frontier_source(work, &self.engine.semantic.stores)?;
        let admission = self.cpu_admissions.get(&work.input.question_digest).ok_or(
            StoreError::InvalidEvidence("R semantic witness has no original admission"),
        )?;
        let (physical, execution, observation) =
            admission
                .committed_execution
                .ok_or(StoreError::InvalidEvidence(
                    "R semantic witness has no published observation",
                ))?;
        let job = admission.job.ok_or(StoreError::InvalidEvidence(
            "R semantic witness has no dispatched Job",
        ))?;
        let owner = admission.owner.ok_or(StoreError::InvalidEvidence(
            "R semantic witness has no original owner",
        ))?;
        let frontier = self
            .rovent_frontier()
            .ok_or(crate::pals::rovent_frontier::FrontierError::ForeignRoot)?;
        let task = self.engine.semantic.stores.tasks.get(execution)?;
        let fact = self.engine.semantic.stores.observations.get(observation)?;
        if frontier.registration().native_ledger.as_ref() != Some(settled.ledger_identity())
            || settled.grant() != report.grant
            || report.actual_work != Some(settled.actual_work())
            || report.admission_rejected != settled.admission_rejected()
            || report.terminal != settled.terminal()
            || settled.admission_rejected()
            || report.grant.execution != physical
            || report.grant.job != job.id
            || report.grant.owner != owner
            || job.question != work.input.question_digest
            || job.hard_deadline != work.original_deadline
            || work.frame != admission.frame
            || work.original_first_move != admission.first_move
            || work.input.stage != admission.stage
            || work.input.question.context != admission.context
            || work.input.view_seal != admission.view_seal
            || admission.execution != Some(execution)
            || admission.settled_receipt
                != Some((physical, execution, cpu_commit::cpu_receipt_digest(report)))
            || admission.task_key != task.key
            || job.evaluation_namespace != admission.evaluation_namespace
            || job.search_conditions != admission.search_conditions
            || fact.execution != Some(execution)
            || fact.value_identity.as_ref() != Some(&admission.value_identity)
            || fact.cpu_condition != task.key.cpu_condition
            || capture.frame() != work.frame
            || capture.question_digest() != work.input.question_digest
            || capture.original_deadline() != work.original_deadline
            || capture.view_seal_digest() != ContractDigest(work.input.view.seal_digest())
            || capture.full_view_digest() != ContractDigest(work.input.view.digest())
            || capture.binding_digest() != job.evaluation_namespace
            || !capture.matches_activated_job(job, owner)
        {
            return Err(StoreError::InvalidEvidence(
                "R semantic witness differs from original settled Store evidence",
            )
            .into());
        }
        let origin = Arc::new(CpuSemanticOrigin {
            store: self.engine.semantic.stores.read_owner(),
            generation: self.engine.semantic.stores.generation(),
            root: self.root_state_id(),
            authority: self.authority,
            frame: work.frame,
            question: work.input.question_digest,
            view: work.input.view_seal,
            full_view: ContractDigest(work.input.view.digest()),
            job,
            owner,
            task: task.key.clone(),
            execution,
            observation,
            settlement: settled.clone(),
            capture: capture.clone(),
        });
        let mut witnesses = Vec::with_capacity(2);
        witnesses.push(CpuSemanticWitness {
            origin: Arc::clone(&origin),
            stage: CpuSemanticStage::RawPublished,
            selected: None,
        });
        if admission.consumed_observation == Some(observation) {
            witnesses.push(CpuSemanticWitness {
                origin,
                stage: CpuSemanticStage::SearchFeedbackConsumed,
                selected: None,
            });
        }
        Ok(witnesses)
    }

    /// Promote a real consumed R observation only when that exact observation
    /// was in this immutable native P/C input and its accepted head was used.
    /// No current-slot view, public DTO, or raw retirement can substitute.
    pub fn selected_cpu_pc_consumption(
        &self,
        source: &CpuSemanticWitness,
        input: &FrameInput<ReviewView, AdaptivePortQuestion>,
        accepted: &AcceptedRoleStep,
        output: &RoleOutput,
    ) -> Result<Option<CpuSemanticWitness>, PalsError> {
        let origin = &source.origin;
        if source.stage != CpuSemanticStage::SearchFeedbackConsumed
            || !origin.store.matches(&self.engine.semantic.stores)
            || origin.generation != self.engine.semantic.stores.generation()
            || origin.authority != self.authority
            || origin.root != self.root_state_id()
            || self.engine.semantic.stores.tasks.get(origin.execution)?.key != origin.task
            || self
                .engine
                .semantic
                .stores
                .observations
                .get(origin.observation)?
                .execution
                != Some(origin.execution)
        {
            return Err(StoreError::InvalidEvidence(
                "R P/C consumption lost original semantic source",
            )
            .into());
        }
        AdaptivePortQuestion::validate_frontier_input(input, &self.engine.semantic.stores)?;
        let ticket = accepted.ticket();
        self.verify_registered_model_namespace(ticket, output)?;
        let head_used = if input.question.goal_local.is_some() {
            self.rovent_goal_consumption_recorded(ticket, &input.view)
        } else {
            self.role_consumption_recorded(ticket, &input.view)?
        };
        if !matches!(ticket.role, Role::Proposer | Role::Critic)
            || !accepted.validates_output(output)
            || ticket.authority != self.authority
            || ticket.question != input.question_digest
            || ticket.view != input.view_seal
            || ticket.original_deadline != self.deadline
            || output.id != ticket.id
            || output.authority != ticket.authority
            || !head_used
        {
            return Err(RoleError::InvalidOutput.into());
        }
        validate_payload(input.question.context.purpose, &input.view, output)?;
        let Some(record) = input
            .view
            .records()
            .iter()
            .find(|record| record.cpu_observation == Some(origin.observation))
        else {
            return Ok(None);
        };
        Ok(Some(CpuSemanticWitness {
            origin: Arc::clone(origin),
            stage: CpuSemanticStage::SelectedPcViewConsumed,
            selected: Some(SelectedPcConsumption {
                ticket: *ticket,
                output: accepted.output_digest(),
                record_revision: record.revision,
                record_origin: record.origin_state,
                full_view: ContractDigest(input.view.digest()),
            }),
        }))
    }
}
