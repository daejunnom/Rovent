//! Actual profiled R dispatch/terminal custody. Foreign reported work never
//! enters CpuCreditLedger and no ExternalJob/Combined adapter is involved.
use super::*;
use rz_search::external_cpu::{
    ProfiledExternalCompletion, ProfiledExternalDispatch, ProfiledExternalJob,
    ProfiledExternalRootSource,
};
use rz_search::pals::engine::adaptive_port::AdaptiveProfiledExternalPreparation;
use rz_search::pals::rovent_frontier::FrontierExternalReservation;

pub(super) struct ProfiledExternalBinding<Q> {
    pub work: CoordinatorWork<ReviewView, Q>,
    pub source: ProfiledExternalRootSource,
    pub job: ProfiledExternalJob,
    pub dispatch: ProfiledExternalDispatch,
    pub reservation: Option<FrontierExternalReservation>,
}

impl<Q, B, D> AdaptiveExecutionManager<Q, B, D>
where
    Q: Clone + Send + Sync + 'static,
    B: TicketNativeBackend<ReviewView, Q>,
    B::Output: ProposerRankingOutput,
    D: AdaptiveSearchDriver<Q>,
{
    pub(super) fn begin_profiled_cpu(
        &mut self,
        work: CoordinatorWork<ReviewView, Q>,
    ) -> Result<bool, AdaptiveManagerError> {
        if !self.can_admit_work() {
            return Ok(false);
        }
        if self.profiled_binding.is_some() || self.profiled_orphan.is_some() {
            return Ok(false);
        }
        let external = self.researcher.profiled()?;
        let source = external.current_source().clone();
        let config = external.configuration().clone();
        let constraint = external.question_constraint();
        let quantum = external.soft_quantum();
        let signal = external.signal();
        let id = self.issue_reciprocal_question()?;
        let execution = ExecutionId {
            epoch: self.registration.authority.epoch,
            sequence: self.issue_execution()?,
        };
        let now = self.check_running()?;
        let job = match self.driver.prepare_profiled_external(
            &work,
            &config,
            &source,
            id,
            execution,
            constraint,
            quantum,
            signal,
            self.clock.domain(),
            now,
        )? {
            AdaptiveProfiledExternalPreparation::Ready(job) => job,
            AdaptiveProfiledExternalPreparation::Unsupported(reason) => {
                return Err(AdaptiveManagerError::ProfiledQueryUnsupported(reason));
            }
        };
        // A hit may close future admission during bounded semantic prep. No
        // foreign request has been enqueued; retain this original source work
        // through its explicit rejection rather than submitting after the gate.
        if !self.can_admit_work() {
            self.driver.reject_profiled_external(&work)?;
            return Ok(false);
        }
        let dispatch = match self.researcher.profiled_mut()?.submit(job.clone()) {
            Ok(dispatch) => dispatch,
            Err(error) => {
                self.driver.reject_profiled_external(&work)?;
                return Err(AdaptiveManagerError::ProfiledExternal(error));
            }
        };
        self.profiled_binding = Some(ProfiledExternalBinding {
            work: work.clone(),
            source: source.clone(),
            job,
            dispatch: dispatch.clone(),
            reservation: None,
        });
        // Physical submission precedes semantic binding. Any later rejection
        // keeps the actual job/dispatch until cancel/drain; it is never free.
        // Journal an already submitted dispatch at its actual clock tick even
        // when the logical deadline/revocation raced with physical enqueue.
        let journal_now = self
            .clock
            .try_now()
            .map_err(|_| AdaptiveManagerError::Deadline)?;
        let admission = self.driver.profiled_external_admitted(
            &work,
            &source,
            &dispatch,
            self.clock.domain(),
            journal_now,
        )?;
        self.profiled_binding
            .as_mut()
            .expect("same submitted external owner")
            .reservation = Some(admission.reservation);
        self.machine.bind_profiled_external_cpu(
            work.frame,
            &admission.binding,
            self.clock.domain(),
            self.check_running()?,
        )?;
        Ok(true)
    }

    pub(super) fn poll_profiled_events(&mut self) -> Result<bool, AdaptiveManagerError> {
        if self.researcher_source_kind() != AdaptiveResearcherSourceKind::ProfiledExternal {
            return Ok(false);
        }
        if self.profiled_orphan.is_some() || self.deferred.len() >= 128 {
            return Ok(false);
        }
        let Some(returned) = self.researcher.profiled_mut()?.take_completion() else {
            return Ok(false);
        };
        if returned.validation().is_err()
            || !self.profiled_binding.as_ref().is_some_and(|binding| {
                returned.dispatch().is_some_and(|dispatch| {
                    dispatch.scope() == binding.dispatch.scope()
                        && dispatch.question() == binding.dispatch.question()
                })
            })
        {
            self.profiled_orphan = Some(returned);
            self.phase = ManagerPhase::Quarantined;
            return Err(AdaptiveManagerError::PhysicalUnknown);
        }
        if self
            .profiled_binding
            .as_ref()
            .is_some_and(|binding| binding.reservation.is_none())
        {
            // The worker really returned this matching terminal, but a
            // post-submit semantic projection failed before a frontier row
            // could be issued. Preserve both the original binding and the
            // owner-validated terminal; neither is a Rules/teacher result.
            // Keep quarantine sticky while the physical owner drains, rather
            // than endlessly retrying a nonexistent reservation or claiming
            // that the already observed terminal itself is unknown.
            self.profiled_orphan = Some(returned);
            self.phase = ManagerPhase::Quarantined;
            return Err(AdaptiveManagerError::Invalid(
                "external terminal observed without frontier reservation",
            ));
        }
        let binding = self.profiled_binding.take().expect("same actual dispatch");
        let (_, packet, validation) = returned.into_parts();
        if let Err(error) = validation {
            return Err(AdaptiveManagerError::ProfiledExternal(error));
        }
        let packet = Arc::new(packet);
        let nodes = packet
            .result()
            .as_ref()
            .ok()
            .and_then(|outcome| outcome.report.raw.work.nodes);
        // Arc allocation happens once. All archive/semantic retries keep this
        // exact owner-issued packet allocation and original Work/View.
        self.defer(DeferredCommit::ProfiledExternal { binding, packet })?;
        self.counters.foreign_reports = self
            .counters
            .foreign_reports
            .checked_add(1)
            .ok_or(AdaptiveManagerError::Capacity("profiled report count"))?;
        if let Some(nodes) = nodes {
            self.counters.foreign_reported_nodes = self
                .counters
                .foreign_reported_nodes
                .checked_add(nodes)
                .ok_or(AdaptiveManagerError::Capacity("profiled reported-node sum"))?;
        } else {
            self.counters.foreign_node_count_unknown = self
                .counters
                .foreign_node_count_unknown
                .checked_add(1)
                .ok_or(AdaptiveManagerError::Capacity(
                    "profiled unknown work count",
                ))?;
        }
        self.flush_deferred()?;
        Ok(true)
    }

    pub(super) fn commit_profiled_packet(
        &mut self,
        binding: &ProfiledExternalBinding<Q>,
        packet: &Arc<ProfiledExternalCompletion>,
    ) -> Result<(), AdaptiveManagerError> {
        let reservation = binding
            .reservation
            .ok_or(AdaptiveManagerError::PhysicalUnknown)?;
        let execution = StepExecution::ProfiledExternal {
            question: binding.job.question.id,
            execution: binding.job.execution,
        };
        let current = self.phase == ManagerPhase::Running
            && self.check_running().is_ok()
            && !self.effective_hard_expired()
            && self
                .machine
                .frame(binding.work.frame)
                .is_some_and(|frame| frame.phase == FramePhase::Waiting(execution));
        if !current {
            return self.driver.observe_profiled_external(
                &binding.work,
                &binding.source,
                reservation,
                packet,
            );
        }
        if self
            .accepted_roles
            .iter()
            .any(|held| held.child == Some(binding.work.frame))
        {
            let response = self.driver.consume_profiled_reciprocal_cpu(
                &binding.work,
                &binding.source,
                reservation,
                packet,
            )?;
            self.machine.resolve_reciprocal_profiled_external_response(
                StepEvidence {
                    frame: binding.work.frame,
                    stage: binding.work.input.stage,
                    execution,
                    submitted_question: binding.work.input.question_digest,
                    submitted_view: binding.work.input.view_seal,
                    completed_evidence: response.evidence(),
                },
                &response,
                self.clock.domain(),
                self.check_running()?,
            )?;
            self.release_accepted_role_waiters(binding.work.frame)?;
            return Ok(());
        }
        let checked = self.driver.commit_profiled_external(
            &binding.work,
            &binding.source,
            reservation,
            packet,
        )?;
        self.resolve(binding.work.clone(), execution, checked)
    }
}
