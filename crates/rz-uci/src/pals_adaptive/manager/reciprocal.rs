//! Actual R -> P/C -> R custody over the existing native service and CPU owner.
//! A node question is distinct from occupancy Yield and legacy partial feedback.

use super::*;
use rz_search::pals::coordinator::{AcceptedRoleStep, DependentFrameAdmission};

/// Explicit opt-in; legacy V5 assembly never creates this policy or a scope.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RoventReciprocalPolicy {
    pub mode: ExecutionMode,
    pub root_budget: Digest,
    pub recipient: SearchRecipient,
    pub max_questions_per_job: u16,
    pub node_interval: u64,
    pub first_question_sequence: u64,
}

/// One ordinary accepted source may ask its P/C peer, then native R. Reciprocal
/// leaves return directly, so this finite registered policy is not recursion.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RoventPeerFollowupPolicy {
    pub pc_peer: bool,
    pub researcher: bool,
    pub max_sources_per_root: u16,
}
#[derive(Clone, Copy)]
pub(super) struct RoventQuestionRouting {
    mode: ExecutionMode,
    root_budget: Digest,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PeerFollowupDisposition {
    Admitted,
    Joined,
    NativeOwnerOccupied,
    NativeCreditExhausted,
    RoleCreditExhausted,
    ModeUnsupported,
    RootSourceCap,
    NoEligibleQuestion,
    NoProgress,
    ViewLeaseCapacity,
    AdmissionClosed,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PeerFollowupObservation {
    pub source: RequestId,
    pub recipient: SearchRecipient,
    pub disposition: PeerFollowupDisposition,
}

pub(super) struct AcceptedRoleCustody<Q, O> {
    pub(super) binding: RoleBinding<Q, O>,
    pub(super) child: Option<FrameId>,
    pub(super) alias: Option<QuestionEnvelope>,
    pub(super) peer_done: bool,
    pub(super) researcher_done: bool,
}

pub(super) struct OwnReciprocalCustody {
    // The ledger has settled this exact receipt. Retention never charges it twice.
    pub(super) report: CpuSliceReport,
    pub(super) child: Option<FrameId>,
    pub(super) question: Option<QuestionEnvelope>,
    pub(super) reply: Option<CpuReciprocalReply>,
    pub(super) observed: bool,
}

impl<Q, B, D> AdaptiveExecutionManager<Q, B, D>
where
    Q: Clone + Send + Sync + 'static,
    B: TicketNativeBackend<ReviewView, Q>,
    B::Output: ProposerRankingOutput,
    D: AdaptiveSearchDriver<Q>,
{
    pub fn configure_rovent_peer_followups(
        &mut self,
        policy: RoventPeerFollowupPolicy,
    ) -> Result<(), AdaptiveManagerError> {
        self.check_running()?;
        if self.root_started
            || self.peer_policy.is_some()
            || self.reciprocal_routing.is_none()
            || !(policy.pc_peer || policy.researcher)
            || !(1..=64).contains(&policy.max_sources_per_root)
        {
            return Err(AdaptiveManagerError::Invalid(
                "Rovent peer policy idle registration",
            ));
        }
        self.peer_policy = Some(policy);
        Ok(())
    }

    pub fn peer_followup_observations(&self) -> impl Iterator<Item = &PeerFollowupObservation> {
        self.peer_observations.iter()
    }

    pub(super) fn retain_accepted_role_for_peer(
        &mut self,
        index: usize,
    ) -> Result<(), AdaptiveManagerError> {
        if self.accepted_roles.len() >= MAX_ROLE_PENDING {
            return Err(AdaptiveManagerError::Capacity(
                "accepted peer parent custody",
            ));
        }
        let binding = &self.roles[index];
        if binding.reciprocal_acceptance.is_none()
            || binding.work.is_none()
            || binding.delivered.is_none()
        {
            return Err(AdaptiveManagerError::Invalid(
                "accepted peer source proof/view/output custody",
            ));
        }
        let policy = self.peer_policy.ok_or(AdaptiveManagerError::Invalid(
            "peer source has no registered policy",
        ))?;
        let capped = self.peer_sources_admitted >= policy.max_sources_per_root;
        if !capped {
            self.peer_sources_admitted += 1;
        }
        let binding = self.roles.remove(index);
        self.accepted_roles.push(AcceptedRoleCustody {
            binding,
            child: None,
            alias: None,
            peer_done: capped || !policy.pc_peer,
            researcher_done: capped || !policy.researcher,
        });
        Ok(())
    }

    pub(super) fn drive_accepted_role_parents(&mut self) -> Result<bool, AdaptiveManagerError> {
        if self.phase != ManagerPhase::Running || !self.driver.semantic_owner_available() {
            return Ok(false);
        }
        let mut changed = false;
        let mut index = 0;
        while index < self.accepted_roles.len() {
            if self.accepted_roles[index].binding.consumption_unknown {
                return Err(AdaptiveManagerError::Invalid(
                    "accepted role consumption remains unknown",
                ));
            }
            if self.accepted_roles[index].child.is_some() {
                index += 1;
                continue;
            }
            let mut held = self.accepted_roles.remove(index);
            let consumed_before = self.counters.consumed_role_outputs;
            let result = self.advance_accepted_role_parent(&mut held);
            if result.is_err() {
                if let (Some(ticket), Some(proof), Some(output)) = (
                    held.binding.ticket,
                    held.binding.reciprocal_acceptance.as_ref(),
                    held.binding.delivered.as_ref(),
                ) {
                    match self
                        .driver
                        .role_consumption_recorded(&ticket, &held.binding.input.view)
                    {
                        Ok(true) => {
                            if !held.binding.consumption_recorded {
                                if let Err(error) =
                                    self.record_role_consumption(ticket, held.binding.physical)
                                {
                                    self.record_failure(error);
                                }
                                held.binding.consumption_recorded = true;
                                #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
                                if let Err(error) = collection::notify_adaptive_consumed_v2(
                                    &mut self.collection_observer_v2,
                                    self.collection_admission_v2.as_ref(),
                                    self.native.native_acceptance_verifier().as_ref(),
                                    proof,
                                    output.proposer_ranking_output(),
                                    true,
                                ) {
                                    self.record_failure(error.into());
                                }
                            }
                            self.observe_cpu_pc_head(
                                &held.binding.input,
                                proof,
                                output.proposer_ranking_output(),
                            );
                            if let Some(work) = &held.binding.work {
                                if let Err(error) = self.utility_role_committed(
                                    work,
                                    proof,
                                    output.proposer_ranking_output(),
                                ) {
                                    self.record_failure(error);
                                }
                            }
                        }
                        Ok(false) => {}
                        Err(error) => {
                            // The source may have been consumed before the
                            // original transition failed. Keep the owner and
                            // do not later classify this result as unused.
                            if !held.binding.consumption_recorded {
                                held.binding.consumption_unknown = true;
                            }
                            self.record_failure(error);
                        }
                    }
                }
            }
            match result {
                Ok(true) => {
                    changed = true;
                }
                Ok(false) => {
                    self.accepted_roles.insert(index, held);
                    index += 1;
                    changed = true;
                }
                Err(error) => {
                    self.accepted_roles.insert(index, held);
                    if self.semantic_replay_can_wait(&error)? {
                        return Ok(
                            changed || self.counters.consumed_role_outputs != consumed_before
                        );
                    }
                    return Err(error);
                }
            }
            if self.has_collection_failure() {
                break;
            }
        }
        Ok(changed)
    }

    fn advance_accepted_role_parent(
        &mut self,
        held: &mut AcceptedRoleCustody<Q, B::Output>,
    ) -> Result<bool, AdaptiveManagerError> {
        let config = self
            .reciprocal_routing
            .ok_or(AdaptiveManagerError::Invalid(
                "accepted peer lost root registration",
            ))?;
        let work = held
            .binding
            .work
            .as_ref()
            .ok_or(AdaptiveManagerError::Invalid(
                "accepted peer lost original frame",
            ))?
            .clone();
        let source = held.binding.ticket.ok_or(AdaptiveManagerError::Invalid(
            "accepted peer lost original ticket",
        ))?;
        self.utility_begin_commit(&work, StepExecution::Role(source.id))?;
        let recipient = if !held.peer_done {
            match source.role {
                Role::Proposer => Some(SearchRecipient::Critic),
                Role::Critic => Some(SearchRecipient::Proposer),
                Role::Validator => {
                    return Err(AdaptiveManagerError::Invalid(
                        "V is not a reciprocal source",
                    ));
                }
            }
        } else if !held.researcher_done {
            Some(SearchRecipient::Researcher)
        } else {
            None
        };
        if let Some(recipient) = recipient {
            let budget = self
                .machine
                .frame(work.frame)
                .ok_or(AdaptiveManagerError::Invalid(
                    "accepted peer parent disappeared",
                ))?
                .budget;
            let unsupported = if !self.can_admit_work() {
                Some(PeerFollowupDisposition::AdmissionClosed)
            } else if !self.role_call_admission_available() {
                Some(PeerFollowupDisposition::RoleCreditExhausted)
            } else if recipient == SearchRecipient::Researcher {
                self.native_peer_unavailable(config.mode)?
            } else if budget.role_submissions_remaining == 0
                || !self.role_call_admission_available()
            {
                Some(PeerFollowupDisposition::RoleCreditExhausted)
            } else {
                None
            };
            if let Some(disposition) = unsupported {
                self.observe_peer(source.id, recipient, disposition)?;
                if recipient == SearchRecipient::Researcher {
                    held.researcher_done = true;
                } else {
                    held.peer_done = true;
                }
                return Ok(false);
            }
            let id = self.issue_reciprocal_question()?;
            let accepted = held
                .binding
                .reciprocal_acceptance
                .as_ref()
                .expect("retained accepted role proof");
            let output = held
                .binding
                .delivered
                .as_ref()
                .expect("retained accepted role output")
                .proposer_ranking_output();
            if !held.binding.consumption_recorded {
                self.driver
                    .accept_accepted_role_source(&work, accepted, output)?;
                self.driver.role_consumed(&source, &work.input.view)?;
                held.binding.consumption_recorded = true;
                self.record_role_consumption(source, held.binding.physical)?;
                #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
                if let Err(error) = collection::notify_adaptive_consumed_v2(
                    &mut self.collection_observer_v2,
                    self.collection_admission_v2.as_ref(),
                    self.native.native_acceptance_verifier().as_ref(),
                    accepted,
                    output,
                    false,
                ) {
                    self.record_failure(error.into());
                    return Ok(false);
                }
                self.observe_cpu_pc_head(&held.binding.input, accepted, output);
            }
            self.utility_role_committed(&work, accepted, output)?;
            let preparation = self.driver.prepare_accepted_role_reciprocal(
                &work,
                accepted,
                output,
                id,
                config.mode,
                config.root_budget,
                recipient,
            );
            let preparation = match preparation {
                Ok(prepared) => prepared,
                Err(AdaptiveManagerError::ReviewLeaseBackpressure {
                    current,
                    capacity,
                    required_new,
                    replacement_reserve,
                }) => {
                    self.review_leases.record_blocked(
                        current,
                        capacity,
                        required_new,
                        replacement_reserve,
                    );
                    self.observe_peer(
                        source.id,
                        recipient,
                        PeerFollowupDisposition::ViewLeaseCapacity,
                    )?;
                    if recipient == SearchRecipient::Researcher {
                        held.researcher_done = true;
                    } else {
                        held.peer_done = true;
                    }
                    // This optional dispatch is exhausted, not the source
                    // acceptance. Its later original commit must still pass.
                    return Ok(false);
                }
                Err(error) => return Err(error),
            };
            let Some(prepared) = preparation else {
                self.observe_peer(
                    source.id,
                    recipient,
                    PeerFollowupDisposition::NoEligibleQuestion,
                )?;
                if recipient == SearchRecipient::Researcher {
                    held.researcher_done = true;
                } else {
                    held.peer_done = true;
                }
                return Ok(false);
            };
            let now = self.check_running()?;
            self.machine.register_accepted_role_origin(
                work.frame,
                prepared.origin,
                accepted,
                self.clock.domain(),
                now,
            )?;
            let evidence = StepEvidence {
                frame: work.frame,
                stage: work.input.stage,
                execution: StepExecution::Role(source.id),
                submitted_question: work.input.question_digest,
                submitted_view: work.input.view_seal,
                completed_evidence: accepted.output_digest(),
            };
            let role_credit = u64::from(recipient != SearchRecipient::Researcher);
            let admitted = match self.machine.admit_after_accepted_role(
                evidence,
                accepted,
                prepared.question,
                prepared.input,
                role_credit,
                self.clock.domain(),
                now,
            ) {
                Ok(admitted) => admitted,
                Err(CoordinatorError::ReciprocalNoProgress) => {
                    self.driver
                        .discard_accepted_role_reciprocal(prepared.question.id)?;
                    self.observe_peer(source.id, recipient, PeerFollowupDisposition::NoProgress)?;
                    if recipient == SearchRecipient::Researcher {
                        held.researcher_done = true;
                    } else {
                        held.peer_done = true;
                    }
                    return Ok(false);
                }
                Err(error) => return Err(error.into()),
            };
            let (child, disposition) = match admitted {
                DependentFrameAdmission::Admitted(id) => (id, PeerFollowupDisposition::Admitted),
                DependentFrameAdmission::Joined(id) => (id, PeerFollowupDisposition::Joined),
            };
            #[cfg(feature = "pals-adaptive-diagnostics")]
            if source.role == Role::Critic && recipient == SearchRecipient::Researcher {
                if let Some(diagnostic) = self.cpu_slice_diagnostic.get() {
                    let revision = self.driver.diagnostic_accepted_source_revision(accepted);
                    // Joined children retain their ORIGINAL view. Inspect that
                    // actual frame, never a relabelled joining caller's input.
                    let actual = self.machine.frame(child);
                    let selected = revision.and_then(|revision| {
                        actual?
                            .input
                            .view
                            .records_metadata()
                            .iter()
                            .position(|record| record.source_revision == revision)
                            .and_then(|index| u16::try_from(index).ok())
                    });
                    diagnostic.record_nnuec_chain(
                        super::super::cpu_slice_observation::NnuecChainEvent::AcceptedCriticChild(
                            super::super::cpu_slice_observation::AcceptedCriticChildObservation {
                                source_frame: work.frame,
                                ticket: *accepted.ticket(),
                                output: accepted.output_digest(),
                                origin_seal: prepared.origin.seal_digest(),
                                question: prepared.question,
                                child,
                                joined: matches!(disposition, PeerFollowupDisposition::Joined),
                                actual_child_view: actual.map(|frame| frame.input.view_seal),
                                actual_child_public_context: actual
                                    .map(|frame| frame.input.public_context),
                                published_record_revision: revision,
                                selected_record_index: selected,
                                selected_record_count: actual
                                    .map(|frame| frame.input.view.records_metadata().len()),
                            },
                        ),
                    );
                }
            }
            held.child = Some(child);
            held.alias = Some(prepared.question);
            self.utility_register_peer(work.frame, child)?;
            if recipient == SearchRecipient::Researcher {
                held.researcher_done = true;
            } else {
                held.peer_done = true;
            }
            self.observe_peer(source.id, recipient, disposition)?;
            return Ok(false);
        }
        let output = held
            .binding
            .delivered
            .as_ref()
            .expect("retained accepted role output")
            .proposer_ranking_output();
        // The submitted snapshot remains the original input/output lineage.
        // Actual dependent children may have spent transferred role credit
        // since submission, so only the machine can supply the current budget.
        let current = self
            .machine
            .bound_work(work.frame, StepExecution::Role(source.id))?;
        if source.authority != work.frame.authority
            || source.question != work.input.question_digest
            || source.view != work.input.view_seal
            || source.original_deadline != work.original_deadline
            || current.frame != work.frame
            || current.original_first_move != work.original_first_move
            || current.original_deadline != work.original_deadline
            || current.input.stage != work.input.stage
            || current.input.stage.role() != Some(source.role)
            || current.input.question_digest != work.input.question_digest
            || current.input.view_seal != work.input.view_seal
            || current.input.public_context != work.input.public_context
            || !Arc::ptr_eq(&current.input.question, &work.input.question)
            || !Arc::ptr_eq(&current.input.view, &work.input.view)
            || current.budget.original_cpu_credit != work.budget.original_cpu_credit
            || current.budget.original_store_byte_limit != work.budget.original_store_byte_limit
            || current.budget.role_submissions_remaining > work.budget.role_submissions_remaining
        {
            return Err(AdaptiveManagerError::Invalid(
                "accepted peer parent readback differs from original bound source",
            ));
        }
        let checked = self
            .driver
            .commit_role(&current, &source, output, held.binding.physical)?;
        self.driver.role_consumed(&source, &work.input.view)?;
        if !held.binding.consumption_recorded {
            held.binding.consumption_recorded = true;
            self.record_role_consumption(source, held.binding.physical)?;
            if let Some(accepted) = &held.binding.reciprocal_acceptance {
                // A parent may reach its final commit without admitting a
                // peer (for example when the follow-up budget is exhausted).
                // Record that actual consumption once, using the original
                // accepted proof and retained output just as on the peer path.
                #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
                if let Err(error) = collection::notify_adaptive_consumed_v2(
                    &mut self.collection_observer_v2,
                    self.collection_admission_v2.as_ref(),
                    self.native.native_acceptance_verifier().as_ref(),
                    accepted,
                    output,
                    false,
                ) {
                    self.record_failure(error.into());
                }
                self.observe_cpu_pc_head(&held.binding.input, accepted, output);
            }
        }
        self.utility_role_committed(
            &work,
            held.binding
                .reciprocal_acceptance
                .as_ref()
                .expect("actual accepted source"),
            output,
        )?;
        self.resolve(current, StepExecution::Role(source.id), checked)?;
        Ok(true)
    }

    fn observe_peer(
        &mut self,
        source: RequestId,
        recipient: SearchRecipient,
        disposition: PeerFollowupDisposition,
    ) -> Result<(), AdaptiveManagerError> {
        if self.peer_observations.len() >= 128 {
            return Err(AdaptiveManagerError::Capacity("root peer observations"));
        }
        self.peer_observations.push(PeerFollowupObservation {
            source,
            recipient,
            disposition,
        });
        Ok(())
    }

    pub(super) fn native_peer_unavailable(
        &self,
        mode: ExecutionMode,
    ) -> Result<Option<PeerFollowupDisposition>, AdaptiveManagerError> {
        #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
        if let ResearcherResources::ProfiledExternal(source) = &self.researcher {
            if !mode.has_native_r() {
                return Ok(Some(PeerFollowupDisposition::ModeUnsupported));
            }
            source
                .current_source()
                .validate_current()
                .map_err(AdaptiveManagerError::ProfiledExternal)?;
            // A busy external owner cannot block its ancestor CPU stack: this
            // source has no paused native stack. Admit the sealed dependency;
            // the normal ready queue waits for that same owner's actual seat.
            return Ok(None);
        }
        if self.researcher_source_kind() != AdaptiveResearcherSourceKind::Native {
            return Ok(Some(PeerFollowupDisposition::ModeUnsupported));
        }
        let (spent, reserved) = self.native_ledger()?.global_usage()?;
        Ok(if !mode.has_native_r() {
            Some(PeerFollowupDisposition::ModeUnsupported)
        } else if !self
            .own
            .iter()
            .any(|slot| slot.binding.is_none() && !slot.joined)
        {
            Some(PeerFollowupDisposition::NativeOwnerOccupied)
        } else if spent.saturating_add(reserved) >= self.registration.native_cpu_node_limit {
            Some(PeerFollowupDisposition::NativeCreditExhausted)
        } else {
            None
        })
    }

    pub(super) fn release_accepted_role_waiters(
        &mut self,
        frame: FrameId,
    ) -> Result<(), AdaptiveManagerError> {
        let mut aliases = Vec::new();
        for held in &mut self.accepted_roles {
            if held.child == Some(frame) {
                if let Some(alias) = held.alias.take() {
                    aliases.push(alias.id);
                }
                held.child = None;
            }
        }
        for question in aliases {
            self.defer(DeferredCommit::PeerDiscard { question })?;
        }
        Ok(())
    }
    pub fn configure_rovent_reciprocal(
        &mut self,
        policy: RoventReciprocalPolicy,
    ) -> Result<(), AdaptiveManagerError> {
        self.check_running()?;
        if self.root_started
            || self.researcher_source_kind() != AdaptiveResearcherSourceKind::Native
            || self.reciprocal_policy.is_some()
            || self.reciprocal_routing.is_some()
            || !self.roles.is_empty()
            || self.own.iter().any(|slot| slot.binding.is_some())
            || self.foreign.is_some()
            || !policy.mode.has_native_r()
            || !matches!(
                policy.recipient,
                SearchRecipient::Proposer | SearchRecipient::Critic
            )
            || !(1..=64).contains(&policy.max_questions_per_job)
            || policy.node_interval == 0
            || policy.first_question_sequence == 0
        {
            return Err(AdaptiveManagerError::Invalid(
                "Rovent reciprocal idle registration",
            ));
        }
        self.machine
            .enable_reciprocal_questions(policy.mode, policy.root_budget)?;
        self.next_reciprocal_question = policy.first_question_sequence;
        self.reciprocal_routing = Some(RoventQuestionRouting {
            mode: policy.mode,
            root_budget: policy.root_budget,
        });
        self.reciprocal_policy = Some(policy);
        Ok(())
    }
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    pub fn configure_rovent_external_routing(
        &mut self,
        mode: ExecutionMode,
        root_budget: Digest,
        first_question_sequence: u64,
    ) -> Result<(), AdaptiveManagerError> {
        self.check_running()?;
        let source = self.researcher.profiled()?;
        if self.root_started
            || !self.roles.is_empty()
            || self.reciprocal_routing.is_some()
            || !mode.has_native_r()
            || first_question_sequence == 0
            || source
                .root_budget_digest()
                .map_err(AdaptiveManagerError::ProfiledExternal)?
                != root_budget
            || self.driver.rovent_root_budget()? != Some(root_budget)
        {
            return Err(AdaptiveManagerError::Invalid(
                "profiled external original routing registration",
            ));
        }
        if let Some(root) = source.root_admission_v1() {
            self.driver
                .validate_profiled_external_registration(source.current_source(), root)?;
        } else if let Some(root) = source.root_admission_v2() {
            self.driver
                .validate_profiled_external_registration_v2(source.current_source(), root)?;
        } else {
            return Err(AdaptiveManagerError::Invalid(
                "profiled external original root absent",
            ));
        }
        self.machine
            .enable_reciprocal_questions(mode, root_budget)?;
        self.next_reciprocal_question = first_question_sequence;
        self.reciprocal_routing = Some(RoventQuestionRouting { mode, root_budget });
        // There is no native callback policy, paused token or node interval.
        Ok(())
    }

    pub(super) fn issue_reciprocal_question(
        &mut self,
    ) -> Result<ReciprocalQuestionId, AdaptiveManagerError> {
        let sequence = self.next_reciprocal_question;
        self.next_reciprocal_question =
            sequence
                .checked_add(1)
                .ok_or(AdaptiveManagerError::Capacity(
                    "reciprocal question sequence",
                ))?;
        Ok(ReciprocalQuestionId {
            authority: self.registration.authority,
            sequence,
        })
    }

    /// Called only after the original CpuCreditLedger settlement and grant echo.
    /// Keep the actual checkpoint receipt; no generic Yield transition is issued.
    pub(super) fn retain_reciprocal_wait(
        &mut self,
        slot: usize,
        mut binding: OwnBinding<Q>,
        report: CpuSliceReport,
    ) -> Result<(), AdaptiveManagerError> {
        let outcome = (|| {
            let request =
                report
                    .reciprocal_request
                    .as_ref()
                    .ok_or(AdaptiveManagerError::Invalid(
                        "AwaitReciprocal has no owner-sealed request",
                    ))?;
            if self.reciprocal_policy.is_none()
                || binding.feedback_forwarded
                || binding.reciprocal.is_some()
                || report.terminal
                || !report.continuation_retained
                || report.actual_work.is_none()
                || request.scope().job != binding.envelope
                || request.scope().owner != binding.grant.owner
            {
                return Err(AdaptiveManagerError::Invalid(
                    "reciprocal checkpoint custody",
                ));
            }
            let now = self.check_running()?;
            self.machine.suspend_cpu_for_reciprocal(
                binding.work.frame,
                &report,
                self.clock.domain(),
                now,
            )?;
            Ok(())
        })();
        binding.resume_pending = false;
        binding.progress_since_slice = false;
        binding.progress_published_at = None;
        binding.requested_yield = None;
        // Even failed semantic admission must retain the same opaque token and
        // receipt until Cancel/JobTerminated and the existing physical fence.
        binding.reciprocal = Some(OwnReciprocalCustody {
            report,
            child: None,
            question: None,
            reply: None,
            observed: false,
        });
        self.own[slot].binding = Some(binding);
        outcome
    }

    /// Bounded hot preparation. Encoding, inference, queue dispatch and actual
    /// acceptance still belong to the existing native physical owner.
    pub(super) fn admit_waiting_reciprocal_questions(
        &mut self,
    ) -> Result<bool, AdaptiveManagerError> {
        let Some(policy) = self.reciprocal_policy else {
            return Ok(false);
        };
        if self.phase != ManagerPhase::Running || !self.driver.semantic_owner_available() {
            return Ok(false);
        }
        let mut changed = false;
        for index in 0..self.own.len() {
            let Some(binding) = self.own[index].binding.as_ref() else {
                continue;
            };
            let Some(custody) = binding
                .reciprocal
                .as_ref()
                .filter(|held| held.child.is_none())
            else {
                continue;
            };
            if self.roles.len() >= self.role_pending_limit()
                || !self.role_call_admission_available()
            {
                break;
            }
            let parent = binding.work.clone();
            let observed = custody.observed;
            let report = custody.report.clone();
            let utility_settled = binding.semantic_settlement.clone();
            let request = custody
                .report
                .reciprocal_request
                .as_ref()
                .ok_or(AdaptiveManagerError::Invalid(
                    "retained reciprocal checkpoint lost request",
                ))?
                .clone();
            if !observed {
                self.driver.cpu_reciprocal_wait(&parent, &report)?;
                self.utility_await_committed(&parent, &report, utility_settled.as_ref())?;
                self.own[index]
                    .binding
                    .as_mut()
                    .expect("retained CPU owner")
                    .reciprocal
                    .as_mut()
                    .expect("retained checkpoint")
                    .observed = true;
            }
            let id = self.issue_reciprocal_question()?;
            let prepared = self.driver.prepare_cpu_reciprocal(
                &parent,
                &request,
                id,
                policy.mode,
                policy.root_budget,
            )?;
            let now = self.check_running()?;
            let admitted = self.machine.admit_reciprocal_question(
                parent.frame,
                prepared.question,
                prepared.input,
                1,
                self.clock.domain(),
                now,
            )?;
            let child = match admitted {
                DependentFrameAdmission::Admitted(child)
                | DependentFrameAdmission::Joined(child) => child,
            };
            let custody = self.own[index]
                .binding
                .as_mut()
                .expect("retained CPU owner")
                .reciprocal
                .as_mut()
                .expect("retained reciprocal checkpoint");
            custody.child = Some(child);
            custody.question = Some(prepared.question);
            self.utility_register_peer(parent.frame, child)?;
            changed = true;
        }
        Ok(changed)
    }

    pub(super) fn is_reciprocal_child(&self, frame: FrameId) -> bool {
        self.accepted_roles
            .iter()
            .any(|held| held.child == Some(frame))
            || self.own.iter().any(|slot| {
                slot.binding
                    .as_ref()
                    .and_then(|binding| binding.reciprocal.as_ref())
                    .is_some_and(|held| held.child == Some(frame))
            })
    }

    pub(super) fn consume_reciprocal_role(
        &mut self,
        index: usize,
        work: &CoordinatorWork<ReviewView, Q>,
        ticket: &RoleTicketEnvelope,
        output: &RoleOutput,
        physical: Option<TicketExecutionEvidence>,
    ) -> Result<(), AdaptiveManagerError> {
        let accepted: &AcceptedRoleStep = self.roles[index].reciprocal_acceptance.as_ref().ok_or(
            AdaptiveManagerError::Invalid("reciprocal output lacks actual retired owner proof"),
        )?;
        let completed_evidence = accepted.output_digest();
        let requests: Vec<_> = self
            .own
            .iter()
            .filter_map(|slot| {
                let custody = slot.binding.as_ref()?.reciprocal.as_ref()?;
                (custody.child == Some(work.frame)).then(|| {
                    Some((
                        custody.report.reciprocal_request.as_ref()?.clone(),
                        custody.question?.id,
                    ))
                })?
            })
            .collect();
        let has_accepted_parent = self
            .accepted_roles
            .iter()
            .any(|held| held.child == Some(work.frame));
        if (requests.is_empty() && !has_accepted_parent) || requests.len() > self.own.len() {
            return Err(AdaptiveManagerError::Invalid(
                "reciprocal child lost original CPU parents",
            ));
        }
        if has_accepted_parent {
            self.driver
                .consume_accepted_role_reciprocal(work, accepted, output, physical)?;
        }
        let replies = if requests.is_empty() {
            Vec::new()
        } else {
            self.driver
                .consume_cpu_reciprocal(work, accepted, output, physical, &requests)?
        };
        if replies.len() != requests.len()
            || replies
                .iter()
                .zip(&requests)
                .any(|(reply, (_, id))| reply.question().id != *id)
        {
            return Err(AdaptiveManagerError::Invalid(
                "reciprocal reply custody changed",
            ));
        }
        // Semantic consumption is marked only after physical acceptance. This
        // is one canonical output even when several sealed parents joined it.
        self.driver.role_consumed(ticket, &work.input.view)?;
        self.mark_role_consumed(index, false)?;
        if self.has_collection_failure() {
            return Ok(());
        }
        self.utility_role_at(index, work, output)?;
        let now = self.check_running()?;
        self.machine.resolve_reciprocal_response(
            StepEvidence {
                frame: work.frame,
                stage: work.input.stage,
                execution: StepExecution::Role(ticket.id),
                submitted_question: work.input.question_digest,
                submitted_view: work.input.view_seal,
                completed_evidence,
            },
            self.clock.domain(),
            now,
        )?;
        self.release_accepted_role_waiters(work.frame)?;
        for (reply, (_, id)) in replies.into_iter().zip(requests) {
            let binding = self
                .own
                .iter_mut()
                .filter_map(|slot| slot.binding.as_mut())
                .find(|binding| {
                    binding
                        .reciprocal
                        .as_ref()
                        .and_then(|held| held.question)
                        .is_some_and(|held| held.id == id)
                })
                .ok_or(AdaptiveManagerError::Invalid(
                    "accepted reciprocal parent disappeared",
                ))?;
            let custody = binding
                .reciprocal
                .as_mut()
                .expect("matched original parent");
            custody.reply = Some(reply);
            binding.resume_pending = true;
        }
        Ok(())
    }
}
