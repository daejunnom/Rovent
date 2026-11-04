//! Bounded dependencies over the existing manager-owned refinement frames.
//! No worker, stage machine, budget ledger or physical owner is duplicated.

use super::*;

#[path = "reciprocal/utility_dependency.rs"]
mod utility_dependency;
pub use utility_dependency::ReciprocalDependencyProof;

fn recipient(stage: RefinementStage) -> Option<SearchRecipient> {
    match stage.role() {
        Some(Role::Proposer) => Some(SearchRecipient::Proposer),
        Some(Role::Critic) => Some(SearchRecipient::Critic),
        // Legacy V selects collection tasks. It is never the R researcher.
        Some(Role::Validator) => None,
        None => Some(SearchRecipient::Researcher),
    }
}

fn same_work(left: &QuestionEnvelope, right: &QuestionEnvelope) -> bool {
    left.recipient == right.recipient
        && left.purpose == right.purpose
        && left.anchor == right.anchor
        && left.anchor_root_ply == right.anchor_root_ply
        && left.original_first_move == right.original_first_move
        && left.prefix_digest == right.prefix_digest
        && left.line_digest == right.line_digest
        && left.line_offset == right.line_offset
        && left.question_digest == right.question_digest
        && left.view == right.view
        && left.evidence_revision == right.evidence_revision
        && left.root_budget == right.root_budget
        && left.original_deadline == right.original_deadline
}

impl<V, Q> RefinementMachine<V, Q> {
    /// Register the real pre-existing candidate frame before its first native R
    /// dispatch. This descriptor grants no work or new role allowance.
    pub fn register_reciprocal_parent(
        &mut self,
        frame: FrameId,
        question: QuestionEnvelope,
        clock: ClockDomain,
        now: MonotonicTick,
    ) -> Result<(), CoordinatorError> {
        self.check_control(clock, now)?;
        self.check_not_waiting(frame)?;
        let root = self
            .reciprocal
            .as_ref()
            .ok_or(CoordinatorError::MissingEvidence)?;
        question.validate(root.mode, clock, now, self.token_budget)?;
        let parent = self.frame(frame).ok_or(CoordinatorError::UnknownRequest)?;
        if self.reciprocal_question(frame).is_some()
            || question.parent.is_some()
            || question.id.authority != self.authority
            || question.root_budget != root.root_budget
            || question.original_deadline != parent.original_deadline
            || question.original_first_move != Some(parent.original_first_move)
            || question.question_digest != parent.input.question_digest
            || question.view != parent.input.view_seal
            || recipient(parent.input.stage) != Some(question.recipient)
            || parent.phase != FramePhase::Ready
        {
            return Err(CoordinatorError::IdentityMismatch(
                "original reciprocal parent/frame/view",
            ));
        }
        if root.questions.len() >= MAX_RECIPROCAL_QUESTION_HISTORY {
            return Err(CoordinatorError::Capacity(
                "root reciprocal question history",
            ));
        }
        if root
            .questions
            .last()
            .is_some_and(|(last, _)| last.id.sequence >= question.id.sequence)
        {
            return Err(CoordinatorError::Duplicate);
        }
        self.reciprocal
            .as_mut()
            .expect("checked root")
            .questions
            .push((question, frame));
        Ok(())
    }

    /// Distinct from occupancy Yield: only a settled owner report containing
    /// the actual opaque native-node question may park this CPU frame.
    pub fn suspend_cpu_for_reciprocal(
        &mut self,
        id: FrameId,
        report: &crate::cpu_worker::CpuSliceReport,
        clock: ClockDomain,
        now: MonotonicTick,
    ) -> Result<(), CoordinatorError> {
        self.check_control(clock, now)?;
        self.check_not_waiting(id)?;
        let frame = self.frame(id).ok_or(CoordinatorError::UnknownRequest)?;
        let execution = StepExecution::Cpu {
            job: report.grant.job,
            execution: report.grant.execution,
        };
        let request = report
            .reciprocal_request
            .as_ref()
            .ok_or(CoordinatorError::MissingEvidence)?;
        let valid = report.result.as_ref().is_ok_and(|actual| {
            actual.report.completion == crate::cpu::CpuCompletion::AwaitReciprocal
                && actual.report.resume.is_none()
                && report.actual_work.is_some_and(|work| {
                    work.nodes == actual.report.nodes
                        && work.quiescence_nodes == actual.report.quiescence_nodes
                        && work.tt_hits == actual.report.tt_hits
                        && work.nodes > 0
                        && work.nodes <= report.grant.node_credit
                })
        });
        if frame.phase != FramePhase::Waiting(execution)
            || frame.cpu_owner != Some((report.grant.job, report.grant.owner))
            || report.terminal
            || report.admission_rejected
            || !report.continuation_retained
            || !valid
            || request.scope().job.id != report.grant.job
            || request.scope().owner != report.grant.owner
            || self.reciprocal_question(id) != Some(request.scope().parent)
        {
            return Err(CoordinatorError::IdentityMismatch(
                "actual reciprocal paused CPU owner/question",
            ));
        }
        let frame = self.match_frame_mut(id)?;
        frame.phase = FramePhase::Suspended;
        frame.zero_work_yields = 0;
        Ok(())
    }
    /// Opt in once, before admitting any root work. The budget identity names
    /// the manager's existing ledger; it issues neither nodes nor storage.
    pub fn enable_reciprocal_questions(
        &mut self,
        mode: ExecutionMode,
        root_budget: Digest,
    ) -> Result<(), CoordinatorError> {
        if !self.admission_open || !self.frame_genealogy.is_empty() || self.reciprocal.is_some() {
            return Err(CoordinatorError::InvalidTransition(
                "reciprocal configuration must precede root admission",
            ));
        }
        self.reciprocal = Some(ReciprocalRoot {
            mode,
            root_budget,
            questions: Vec::with_capacity(MAX_RECIPROCAL_QUESTION_HISTORY),
            accepted_origins: HashMap::new(),
        });
        Ok(())
    }

    pub fn reciprocal_question(&self, frame: FrameId) -> Option<QuestionEnvelope> {
        let current = self.frame(frame)?;
        self.reciprocal
            .as_ref()?
            .questions
            .iter()
            .find_map(|(question, held)| {
                (*held == frame
                    && question.question_digest == current.input.question_digest
                    && question.view == current.input.view_seal)
                    .then_some(*question)
            })
    }

    /// A joined ID points to the original frame and original sealed response;
    /// no result is relabelled as an evaluation of the joining caller's view.
    pub fn reciprocal_frame(&self, question: ReciprocalQuestionId) -> Option<FrameId> {
        self.reciprocal
            .as_ref()?
            .questions
            .iter()
            .find_map(|(held, frame)| (held.id == question).then_some(*frame))
    }

    pub fn waiting_on(&self, frame: FrameId) -> Option<FrameId> {
        self.dependencies
            .iter()
            .find_map(|edge| (edge.waiter == frame).then_some(edge.dependency))
    }

    pub fn waiting_origin_seal(&self, frame: FrameId) -> Option<Digest> {
        self.dependencies
            .iter()
            .find(|edge| edge.waiter == frame)
            .and_then(|edge| edge.origin_seal)
    }

    /// Cancellation of one parent does not discard a joined child's physical
    /// owner. The manager can cancel these bounded orphans through the normal
    /// ticket/CPU control paths, then retain them until actual drain completes.
    pub fn orphaned_reciprocal_frames(&self) -> impl Iterator<Item = FrameId> + '_ {
        self.dependent_credit_owners
            .iter()
            .filter_map(|(child, parent)| {
                (self.frame(*parent).is_none()
                    && !self
                        .dependencies
                        .iter()
                        .any(|edge| edge.dependency == *child))
                .then_some(*child)
            })
    }

    pub(super) fn check_not_waiting(&self, frame: FrameId) -> Result<(), CoordinatorError> {
        if self.waiting_on(frame).is_some() {
            return Err(CoordinatorError::InvalidTransition(
                "frame awaits an independent reciprocal response",
            ));
        }
        Ok(())
    }

    /// The engine port must first check Rules/Store pins and seal `input`.
    /// All six P/C/R directions reuse existing role and CPU bindings. An R
    /// started parent must reach a real stable checkpoint before asking: its
    /// private token remains on that frame/owner while the child is serviced
    /// elsewhere. The CPU ledger independently verifies settled reservations.
    ///
    /// Role allowance is transferred, never cloned. CPU and Store declarations
    /// remain the same original root limits; their owners still reserve actual
    /// work. A joined request consumes no second submission or reservation.
    pub fn admit_reciprocal_question(
        &mut self,
        parent: FrameId,
        question: QuestionEnvelope,
        input: FrameInput<V, Q>,
        role_credit: u64,
        clock: ClockDomain,
        now: MonotonicTick,
    ) -> Result<DependentFrameAdmission, CoordinatorError> {
        self.admit_reciprocal_inner(parent, question, input, role_credit, None, clock, now)
    }

    /// The parent keeps its original Waiting(Role) execution. The accepted
    /// result can finish once its child returns, without a second P/C forward.
    #[allow(clippy::too_many_arguments)]
    pub fn admit_after_accepted_role(
        &mut self,
        evidence: StepEvidence,
        accepted: &super::super::AcceptedRoleStep,
        question: QuestionEnvelope,
        input: FrameInput<V, Q>,
        role_credit: u64,
        clock: ClockDomain,
        now: MonotonicTick,
    ) -> Result<DependentFrameAdmission, CoordinatorError> {
        let ticket = accepted.ticket();
        if evidence.execution != StepExecution::Role(ticket.id)
            || evidence.submitted_question != ticket.question
            || evidence.submitted_view != ticket.view
            || evidence.completed_evidence != accepted.output_digest()
            || evidence.stage.role() != Some(ticket.role)
        {
            return Err(CoordinatorError::IdentityMismatch(
                "accepted original P/C checkpoint",
            ));
        }
        let peer = self.reciprocal_question(evidence.frame);
        let origin = accepted.origin(evidence.frame, peer)?;
        self.register_accepted_role_origin(evidence.frame, origin, accepted, clock, now)?;
        self.admit_reciprocal_inner(
            evidence.frame,
            question,
            input,
            role_credit,
            Some(evidence),
            clock,
            now,
        )
    }

    pub fn register_accepted_role_origin(
        &mut self,
        frame: FrameId,
        origin: super::super::AcceptedRoleOrigin,
        accepted: &super::super::AcceptedRoleStep,
        clock: ClockDomain,
        now: MonotonicTick,
    ) -> Result<(), CoordinatorError> {
        use super::super::AcceptedRoleOriginKind;
        self.check_control(clock, now)?;
        let held = self.frame(frame).ok_or(CoordinatorError::UnknownRequest)?;
        let ticket = origin.ticket();
        let peer = match origin.kind() {
            AcceptedRoleOriginKind::LocalAccepted => None,
            AcceptedRoleOriginKind::PeerQuestion(peer) => Some(peer),
        };
        if origin.frame() != frame
            || !origin.matches(accepted)
            || held.phase != FramePhase::Waiting(StepExecution::Role(ticket.id))
            || held.input.question_digest != ticket.question
            || held.input.view_seal != ticket.view
            || held.input.stage.role() != Some(ticket.role)
            || held.original_deadline != ticket.original_deadline
            || peer != self.reciprocal_question(frame)
        {
            return Err(CoordinatorError::IdentityMismatch(
                "accepted local/peer origin and original source",
            ));
        }
        let root = self
            .reciprocal
            .as_mut()
            .ok_or(CoordinatorError::MissingEvidence)?;
        let key = (frame, ticket.question);
        if let Some(previous) = root.accepted_origins.get(&key) {
            return if *previous == origin {
                Ok(())
            } else {
                Err(CoordinatorError::IdentityMismatch(
                    "accepted source origin changed",
                ))
            };
        }
        if root.accepted_origins.len() >= MAX_RECIPROCAL_QUESTION_HISTORY {
            return Err(CoordinatorError::Capacity("accepted source origins"));
        }
        root.accepted_origins.insert(key, origin);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn admit_reciprocal_inner(
        &mut self,
        parent: FrameId,
        question: QuestionEnvelope,
        input: FrameInput<V, Q>,
        role_credit: u64,
        accepted: Option<StepEvidence>,
        clock: ClockDomain,
        now: MonotonicTick,
    ) -> Result<DependentFrameAdmission, CoordinatorError> {
        self.check_control(clock, now)?;
        self.check_not_waiting(parent)?;
        let root = self
            .reciprocal
            .as_ref()
            .ok_or(CoordinatorError::InvalidTransition(
                "reciprocal questions were not configured",
            ))?;
        question.validate(root.mode, clock, now, self.token_budget)?;
        let parent_frame = self.frame(parent).ok_or(CoordinatorError::UnknownRequest)?;
        let accepted_checkpoint = accepted.is_some_and(|evidence| {
            parent_frame.phase == FramePhase::Waiting(evidence.execution)
                && evidence.stage == parent_frame.input.stage
                && evidence.submitted_question == parent_frame.input.question_digest
                && evidence.submitted_view == parent_frame.input.view_seal
        });
        if !accepted_checkpoint
            && !matches!(
                parent_frame.phase,
                FramePhase::Ready | FramePhase::Suspended
            )
        {
            return Err(CoordinatorError::InvalidTransition(
                "dependent admission requires a stable parent checkpoint",
            ));
        }
        if question.id.authority != self.authority
            || question.original_deadline != self.hard_deadline
            || question.root_budget != root.root_budget
            || question.original_first_move != Some(parent_frame.original_first_move)
            || question.question_digest != input.question_digest
            || question.view != input.view_seal
            || input.view_seal.root_state != parent_frame.input.view_seal.root_state
            || recipient(parent_frame.input.stage) != Some(question.sender)
            || recipient(input.stage) != Some(question.recipient)
            || question.parent != self.reciprocal_question(parent).map(|held| held.id)
        {
            return Err(CoordinatorError::IdentityMismatch(
                "reciprocal parent/question/view/recipient/original root budget",
            ));
        }
        if root
            .questions
            .last()
            .is_some_and(|(previous, _)| question.id.sequence <= previous.id.sequence)
        {
            return Err(CoordinatorError::Duplicate);
        }
        // An owner seal/issuance sequence can change without adding any model
        // input or evidence. Retired identical semantics are not fresh work.
        if root.questions.iter().any(|(held, frame)| {
            self.frame(*frame).is_none()
                && held.recipient == question.recipient
                && held.purpose == question.purpose
                && held.anchor.semantic == question.anchor.semantic
                && held.anchor_root_ply == question.anchor_root_ply
                && held.original_first_move == question.original_first_move
                && held.prefix_digest == question.prefix_digest
                && held.question_digest == question.question_digest
                && held.evidence_revision == question.evidence_revision
                && held.root_budget == question.root_budget
                && held.original_deadline == question.original_deadline
                && held.view.context_policy == question.view.context_policy
                && held.view.root_state == question.view.root_state
        }) {
            return Err(CoordinatorError::ReciprocalNoProgress);
        }
        if root.questions.len() >= MAX_RECIPROCAL_QUESTION_HISTORY {
            return Err(CoordinatorError::Capacity(
                "root reciprocal question history",
            ));
        }
        if self.dependencies.len() >= MAX_COORDINATOR_FRAMES {
            return Err(CoordinatorError::Capacity("reciprocal waiting frames"));
        }
        // Ancestors include the legacy origin, which has no reciprocal ID.
        // Identical work may not be requested back from its own active caller.
        let mut ancestor = Some(parent);
        for depth in 0..=MAX_RECIPROCAL_QUESTION_DEPTH {
            let Some(id) = ancestor else { break };
            if depth == MAX_RECIPROCAL_QUESTION_DEPTH {
                return Err(CoordinatorError::Capacity("reciprocal ancestor depth"));
            }
            let Some(frame) = self.frame(id) else { break };
            if recipient(frame.input.stage) == Some(question.recipient)
                && frame.input.question_digest == input.question_digest
                && frame.input.view_seal == input.view_seal
            {
                return Err(CoordinatorError::InvalidTransition(
                    "reciprocal question would wait on its own ancestor",
                ));
            }
            ancestor = frame.parent;
        }
        let existing = root
            .questions
            .iter()
            .find_map(|(held, frame)| same_work(held, &question).then_some(*frame));
        if let Some(existing) = existing {
            let Some(existing_frame) = self.frame(existing) else {
                return Err(CoordinatorError::ReciprocalNoProgress);
            };
            if existing_frame.input.stage != input.stage
                || existing_frame.input.public_context != input.public_context
                || existing_frame.budget.original_cpu_credit
                    != parent_frame.budget.original_cpu_credit
                || existing_frame.budget.original_store_byte_limit
                    != parent_frame.budget.original_store_byte_limit
            {
                return Err(CoordinatorError::IdentityMismatch(
                    "joined reciprocal input or original resource scope differs",
                ));
            }
            self.check_dependency_cycle(parent, existing)?;
            self.dependencies.push(FrameDependency {
                waiter: parent,
                dependency: existing,
                question: question.id,
                origin_seal: accepted.map(|evidence| {
                    root.accepted_origins[&(parent, evidence.submitted_question)].seal_digest()
                }),
            });
            self.reciprocal
                .as_mut()
                .expect("checked root")
                .questions
                .push((question, existing));
            return Ok(DependentFrameAdmission::Joined(existing));
        }
        if role_credit > parent_frame.budget.role_submissions_remaining
            || (question.recipient != SearchRecipient::Researcher && role_credit == 0)
        {
            return Err(CoordinatorError::Capacity(
                "transferred reciprocal role credit",
            ));
        }
        let first_move = parent_frame.original_first_move;
        let budget = FrameBudget {
            role_submissions_remaining: role_credit,
            original_cpu_credit: parent_frame.budget.original_cpu_credit,
            original_store_byte_limit: parent_frame.budget.original_store_byte_limit,
        };
        let child = self.admit_internal(
            first_move,
            input,
            budget,
            FrameLineage {
                parent: Some(parent),
                supersedes: None,
            },
            clock,
            now,
        )?;
        self.match_frame_mut(parent)
            .expect("checked live parent")
            .budget
            .role_submissions_remaining -= role_credit;
        self.dependencies.push(FrameDependency {
            waiter: parent,
            dependency: child,
            question: question.id,
            origin_seal: accepted.map(|evidence| {
                self.reciprocal
                    .as_ref()
                    .expect("registered root")
                    .accepted_origins[&(parent, evidence.submitted_question)]
                    .seal_digest()
            }),
        });
        self.dependent_credit_owners.push((child, parent));
        self.reciprocal
            .as_mut()
            .expect("checked root")
            .questions
            .push((question, child));
        Ok(DependentFrameAdmission::Admitted(child))
    }

    fn check_dependency_cycle(
        &self,
        waiter: FrameId,
        dependency: FrameId,
    ) -> Result<(), CoordinatorError> {
        let mut next = Some(dependency);
        for _ in 0..=MAX_COORDINATOR_FRAMES {
            let Some(frame) = next else { return Ok(()) };
            if frame == waiter {
                return Err(CoordinatorError::InvalidTransition(
                    "reciprocal dependency cycle",
                ));
            }
            next = self.waiting_on(frame);
        }
        Err(CoordinatorError::Capacity("reciprocal dependency depth"))
    }

    pub(super) fn check_dependent_retirement(
        &self,
        frame: FrameId,
    ) -> Result<(), CoordinatorError> {
        if self.admission_open {
            self.check_not_waiting(frame)?;
        }
        self.check_dependent_credit_return(frame)
    }

    pub(super) fn check_dependent_credit_return(
        &self,
        frame: FrameId,
    ) -> Result<(), CoordinatorError> {
        if let Some((_, parent)) = self
            .dependent_credit_owners
            .iter()
            .find(|(child, _)| *child == frame)
            && let (Some(child), Some(parent)) = (self.frame(frame), self.frame(*parent))
            && parent
                .budget
                .role_submissions_remaining
                .checked_add(child.budget.role_submissions_remaining)
                .is_none()
        {
            return Err(CoordinatorError::Capacity(
                "returned reciprocal role credit overflow",
            ));
        }
        Ok(())
    }

    pub(super) fn retire_dependencies(&mut self, frame: FrameId, remaining_roles: u64) {
        if let Some(index) = self
            .dependent_credit_owners
            .iter()
            .position(|(child, _)| *child == frame)
        {
            let (_, parent) = self.dependent_credit_owners.swap_remove(index);
            if let Some(parent) = self.frames.iter_mut().find(|held| held.id == parent) {
                parent.budget.role_submissions_remaining = parent
                    .budget
                    .role_submissions_remaining
                    .checked_add(remaining_roles)
                    .expect("validated before dependent retirement");
            }
        }
        self.dependencies
            .retain(|edge| edge.waiter != frame && edge.dependency != frame);
    }

    /// Consume the exact attested worker result once. The manager is still the
    /// sole evidence writer. This does not claim mate, Fresh-pair completion or
    /// successful application of the reply to an R private stack.
    pub fn resolve_reciprocal_response(
        &mut self,
        evidence: StepEvidence,
        clock: ClockDomain,
        now: MonotonicTick,
    ) -> Result<DependentFrameResponse, CoordinatorError> {
        if !matches!(evidence.execution, StepExecution::Role(_)) {
            return Err(CoordinatorError::InvalidTransition(
                "CPU dependent response requires its actual terminal receipt",
            ));
        }
        self.resolve_reciprocal_response_checked(evidence, clock, now)
    }

    /// A yielded or unknown CPU report cannot drop its owner/token by being
    /// relabelled as a complete dependent answer. Exact-credit settlement is
    /// still performed once by the existing CpuCreditLedger before this call.
    pub fn resolve_reciprocal_cpu_response(
        &mut self,
        evidence: StepEvidence,
        report: &crate::cpu_worker::CpuSliceReport,
        clock: ClockDomain,
        now: MonotonicTick,
    ) -> Result<DependentFrameResponse, CoordinatorError> {
        self.check_reciprocal_cpu_terminal(evidence, report)?;
        if !report.result.as_ref().is_ok_and(|value| {
            !matches!(
                value.report.completion,
                crate::cpu::CpuCompletion::Yielded
                    | crate::cpu::CpuCompletion::AwaitReciprocal
                    | crate::cpu::CpuCompletion::Canceled
                    | crate::cpu::CpuCompletion::Deadline
            )
        }) {
            return Err(CoordinatorError::IdentityMismatch(
                "dependent CPU terminal owner receipt",
            ));
        }
        self.resolve_reciprocal_response_checked(evidence, clock, now)
    }

    fn check_reciprocal_cpu_terminal(
        &self,
        evidence: StepEvidence,
        report: &crate::cpu_worker::CpuSliceReport,
    ) -> Result<(), CoordinatorError> {
        let frame = self
            .frame(evidence.frame)
            .ok_or(CoordinatorError::UnknownRequest)?;
        if evidence.execution
            != (StepExecution::Cpu {
                job: report.grant.job,
                execution: report.grant.execution,
            })
            || frame.cpu_owner != Some((report.grant.job, report.grant.owner))
            || !report.terminal
            || report.continuation_retained
            || report.admission_rejected
            || report.actual_work.is_none()
            || report.result.as_ref().is_ok_and(|value| {
                matches!(
                    value.report.completion,
                    crate::cpu::CpuCompletion::Yielded | crate::cpu::CpuCompletion::AwaitReciprocal
                ) || value.report.resume.is_some()
            })
        {
            return Err(CoordinatorError::IdentityMismatch(
                "dependent CPU terminal owner receipt",
            ));
        }
        Ok(())
    }

    /// Known terminal failure may clean up after deadline/revoke. An unknown
    /// work count or a retained token must remain tracked for quarantine/drain.
    /// This returns Unknown evidence, never a successful reciprocal answer.
    pub fn retire_reciprocal_cpu_failure(
        &mut self,
        evidence: StepEvidence,
        report: &crate::cpu_worker::CpuSliceReport,
        reason: &'static str,
    ) -> Result<ClosedFrame, CoordinatorError> {
        self.check_reciprocal_cpu_terminal(evidence, report)?;
        self.check_dependent_credit_return(evidence.frame)?;
        if !self
            .dependent_credit_owners
            .iter()
            .any(|(child, _)| *child == evidence.frame)
            || report.result.as_ref().is_ok_and(|value| {
                !matches!(
                    value.report.completion,
                    crate::cpu::CpuCompletion::Canceled | crate::cpu::CpuCompletion::Deadline
                )
            })
        {
            return Err(CoordinatorError::InvalidTransition(
                "dependent CPU failure requires a terminal failure receipt",
            ));
        }
        let frame = self
            .frame(evidence.frame)
            .ok_or(CoordinatorError::UnknownRequest)?;
        if frame.phase != FramePhase::Waiting(evidence.execution)
            || frame.input.stage != evidence.stage
            || frame.input.question_digest != evidence.submitted_question
            || frame.input.view_seal != evidence.submitted_view
        {
            return Err(CoordinatorError::IdentityMismatch(
                "dependent CPU failure original execution/question/view",
            ));
        }
        self.retire_cpu_failure_frame(evidence, reason)
    }

    /// Cleanup of the original known no-value Slice. This is neither an
    /// unsubmitted Ready decline nor a successful reciprocal/goal response.
    /// The caller must first settle the native ledger and obtain the original
    /// receipt seal from the Search no-value observer; no synthetic terminal
    /// packet or score is accepted here.
    pub fn retire_cpu_no_value(
        &mut self,
        original: &CoordinatorWork<V, Q>,
        report: &crate::cpu_worker::CpuSliceReport,
        receipt_seal: Digest,
    ) -> Result<ClosedFrame, CoordinatorError> {
        let evidence = StepEvidence {
            frame: original.frame,
            stage: original.input.stage,
            execution: StepExecution::Cpu {
                job: report.grant.job,
                execution: report.grant.execution,
            },
            submitted_question: original.input.question_digest,
            submitted_view: original.input.view_seal,
            completed_evidence: receipt_seal,
        };
        self.check_reciprocal_cpu_terminal(evidence, report)?;
        let stop = report
            .result
            .as_ref()
            .err()
            .and_then(crate::cpu::CpuError::value_authority_stop);
        let frame = self
            .frame(original.frame)
            .ok_or(CoordinatorError::UnknownRequest)?;
        if stop.is_none()
            || report.reciprocal_request.is_some()
            || report.controls.is_none_or(|controls| {
                !controls.terminal
                    || controls.cancel_ack > controls.cancel_sequence
                    || controls.yield_ack > controls.yield_sequence
                    || controls.publish_ack > controls.publish_sequence
            })
            || frame.phase != FramePhase::Waiting(evidence.execution)
            || frame.last_execution != Some(evidence.execution)
            || frame.id.authority != report.grant.job.authority
            || report.grant.execution.epoch != frame.id.authority.epoch
            || frame.original_first_move != original.original_first_move
            || frame.original_deadline != original.original_deadline
            || frame.input.stage != original.input.stage
            || frame.input.question_digest != original.input.question_digest
            || frame.input.view_seal != original.input.view_seal
            || frame.input.public_context != original.input.public_context
            || !Arc::ptr_eq(&frame.input.question, &original.input.question)
            || !Arc::ptr_eq(&frame.input.view, &original.input.view)
            || frame.budget.original_cpu_credit != original.budget.original_cpu_credit
            || frame.budget.original_store_byte_limit != original.budget.original_store_byte_limit
        {
            return Err(CoordinatorError::IdentityMismatch(
                "no-value CPU terminal original execution/question/view/limits",
            ));
        }
        // Cleanup is legal after revoke/expiry. Original controls and immutable
        // binding stay unchanged; there is no renewed check_control window.
        self.check_dependent_credit_return(evidence.frame)?;
        self.retire_cpu_failure_frame(evidence, "CPU value authority ended without a value")
    }

    fn retire_cpu_failure_frame(
        &mut self,
        evidence: StepEvidence,
        reason: &'static str,
    ) -> Result<ClosedFrame, CoordinatorError> {
        let frame = self
            .frame(evidence.frame)
            .ok_or(CoordinatorError::UnknownRequest)?;
        let remaining_roles = frame.budget.role_submissions_remaining;
        let closed = ClosedFrame {
            id: frame.id,
            original_first_move: frame.original_first_move,
            evidence: vec![evidence.completed_evidence],
            unknown: Some(reason),
            completion_scope: None,
        };
        self.frames.retain(|held| held.id != evidence.frame);
        self.retire_dependencies(evidence.frame, remaining_roles);
        Ok(closed)
    }

    fn resolve_reciprocal_response_checked(
        &mut self,
        evidence: StepEvidence,
        clock: ClockDomain,
        now: MonotonicTick,
    ) -> Result<DependentFrameResponse, CoordinatorError> {
        self.check_control(clock, now)?;
        self.check_dependent_retirement(evidence.frame)?;
        if !self
            .dependent_credit_owners
            .iter()
            .any(|(child, _)| *child == evidence.frame)
        {
            return Err(CoordinatorError::InvalidTransition(
                "response requires an admitted dependent frame",
            ));
        }
        let frame = self
            .frame(evidence.frame)
            .ok_or(CoordinatorError::UnknownRequest)?;
        if frame.phase != FramePhase::Waiting(evidence.execution)
            || frame.input.stage != evidence.stage
            || frame.input.question_digest != evidence.submitted_question
            || frame.input.view_seal != evidence.submitted_view
        {
            return Err(CoordinatorError::IdentityMismatch(
                "dependent response original execution/question/view",
            ));
        }
        let remaining = frame.budget.role_submissions_remaining;
        let response = DependentFrameResponse {
            frame: evidence.frame,
            question: evidence.submitted_question,
            view: evidence.submitted_view,
            evidence: evidence.completed_evidence,
            execution: evidence.execution,
        };
        self.frames.retain(|held| held.id != evidence.frame);
        self.retire_dependencies(evidence.frame, remaining);
        Ok(response)
    }
}

impl<Q> RefinementMachine<crate::pals::review_context::ReviewView, Q> {
    /// A foreign peer reply requires the opaque actual Store consumer proof;
    /// a raw UCI packet or public completed-evidence digest is insufficient.
    pub fn resolve_reciprocal_profiled_external_response(
        &mut self,
        evidence: StepEvidence,
        proof: &crate::pals::engine::adaptive_port::CheckedProfiledExternalResponse,
        clock: ClockDomain,
        now: MonotonicTick,
    ) -> Result<DependentFrameResponse, CoordinatorError> {
        let frame = self
            .frame(evidence.frame)
            .ok_or(CoordinatorError::UnknownRequest)?;
        if !proof.validates(
            &evidence,
            frame.id,
            frame.original_first_move,
            frame.original_deadline,
            &frame.input,
        ) {
            return Err(CoordinatorError::IdentityMismatch(
                "profiled peer actual Store consumption",
            ));
        }
        self.resolve_reciprocal_response_checked(evidence, clock, now)
    }
}

#[cfg(test)]
#[path = "reciprocal_tests.rs"]
mod tests;
