//! Authored neural service with real CPU owner, ledger, ticket retirement and resume.
use super::*;
use rz_contracts::rovent::{
    ChessPly, ExecutionMode, QuestionEnvelope, QuestionPurpose, ReciprocalQuestionId,
    SearchRecipient,
};
use rz_search::cpu::{CpuReciprocalReply, CpuReciprocalRequest};
use rz_search::pals::coordinator::AcceptedRoleStep;

impl DriverFixture {
    // Exercise the independent original work, accepted head/output, question
    // identity, mode, budget and recipient checks used by the actual boundary.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn fixture_prepare_peer(
        &mut self,
        work: &CoordinatorWork<ReviewView, u64>,
        accepted: &AcceptedRoleStep,
        output: &RoleOutput,
        id: ReciprocalQuestionId,
        _: ExecutionMode,
        budget: Digest,
        recipient: SearchRecipient,
    ) -> Result<AdaptivePeerQuestion<u64>, AdaptiveManagerError> {
        assert!(accepted.validates_output(output));
        self.validate_role(work, accepted.ticket(), output)?;
        let previous = self
            .reciprocal_questions
            .values()
            .find(|question| {
                question.question_digest == work.input.question_digest
                    && question.view == work.input.view_seal
            })
            .copied();
        let origin = accepted.origin(work.frame, previous).unwrap();
        let prefix = if work.input.stage == RefinementStage::CriticDivergence {
            let RolePayload::Counterexample {
                divergence_logits, ..
            } = &output.payload
            else {
                unreachable!()
            };
            assert_eq!(work.input.view.query().legal.len(), 0);
            assert_eq!(
                divergence_logits.len(),
                work.input.view.query().divergences.len()
            );
            let index = divergence_logits
                .iter()
                .enumerate()
                .max_by(|left, right| left.1.total_cmp(right.1))
                .unwrap()
                .0;
            let ply = work.input.view.query().divergences[index];
            let mut prefix = work.input.view.query().prefix.clone();
            prefix.extend_from_slice(&work.input.view.query().proposal[..ply]);
            prefix
        } else {
            work.input.view.query().prefix.clone()
        };
        let mut endpoint = Position::from_snapshot(
            self.rules.rules().snapshot(),
            rz_position::PositionLimits::default(),
        )
        .unwrap();
        for movement in &prefix {
            endpoint.make_move(*movement).unwrap();
        }
        let legal = endpoint.legal_moves();
        let purpose = if recipient == SearchRecipient::Critic {
            RoleQueryPurpose::ReplyPolicy
        } else {
            RoleQueryPurpose::ProposePolicy
        };
        let view = Arc::new(
            self.index
                .select(
                    &self.stores,
                    self.state,
                    ReviewQuerySequences {
                        purpose,
                        prefix: &prefix,
                        proposal: &[],
                        counterexample: None,
                        legal: &legal,
                        divergences: &[],
                    },
                    ReviewSelectionPolicy::default(),
                )
                .unwrap(),
        );
        let seal = ReviewViewSeal {
            authority: self.authority,
            view_sequence: view.sequence(),
            context_policy: Digest(view.policy_digest()),
            view_digest: Digest(view.seal_digest()),
            root_state: self.rules.snapshot().identity(),
            actual_tokens: view.token_count(),
            selected_records: view.records().len() as u16,
            anchors: view.anchors().len() as u8,
            length_bucket: view.length_bucket(),
        };
        let state = ContractPosition::new(OwnerId(71), endpoint)
            .export()
            .unwrap();
        let question = QuestionEnvelope {
            id,
            parent: previous.map(|question| question.id),
            sender: if accepted.ticket().role == Role::Proposer {
                SearchRecipient::Proposer
            } else {
                SearchRecipient::Critic
            },
            recipient,
            purpose: if recipient == SearchRecipient::Researcher {
                QuestionPurpose::TacticalReview
            } else if recipient == SearchRecipient::Critic {
                QuestionPurpose::Counter
            } else {
                QuestionPurpose::Propose
            },
            anchor: state.snapshot().identity(),
            anchor_root_ply: ChessPly(prefix.len() as u16),
            original_first_move: Some(work.original_first_move),
            prefix_digest: Digest(view.digest()),
            line_digest: None,
            line_offset: ChessPly(0),
            question_digest: Digest(view.digest()),
            view: seal,
            evidence_revision: 0,
            root_budget: budget,
            original_deadline: work.original_deadline,
        };
        self.peer_sources.push((accepted.ticket().role, recipient));
        self.reciprocal_questions.insert(id.sequence, question);
        Ok(AdaptivePeerQuestion {
            origin,
            question,
            input: FrameInput {
                stage: match recipient {
                    SearchRecipient::Proposer => RefinementStage::ProposalContinuation,
                    SearchRecipient::Critic => RefinementStage::CriticReply,
                    SearchRecipient::Researcher => RefinementStage::CpuValidation,
                },
                question: Arc::new(id.sequence),
                question_digest: question.question_digest,
                public_context: Digest(view.record_scope_digest()),
                view_seal: seal,
                view,
            },
        })
    }
    pub(super) fn fixture_reciprocal_parent(
        &mut self,
        work: &CoordinatorWork<ReviewView, u64>,
        job: &rz_contracts::pals_adaptive::CpuJobEnvelope,
        id: ReciprocalQuestionId,
        mode: ExecutionMode,
        root_budget: Digest,
    ) -> Result<QuestionEnvelope, AdaptiveManagerError> {
        assert!(mode.has_native_r());
        Ok(QuestionEnvelope {
            id,
            parent: None,
            sender: SearchRecipient::Proposer,
            recipient: SearchRecipient::Researcher,
            purpose: QuestionPurpose::TacticalReview,
            anchor: job.state_identity,
            anchor_root_ply: ChessPly(work.input.view.query().prefix.len() as u16),
            original_first_move: Some(work.original_first_move),
            prefix_digest: work.input.question_digest,
            line_digest: None,
            line_offset: ChessPly(0),
            question_digest: work.input.question_digest,
            view: work.input.view_seal,
            evidence_revision: 0,
            root_budget,
            original_deadline: job.hard_deadline,
        })
    }

    pub(super) fn fixture_prepare_reciprocal(
        &mut self,
        work: &CoordinatorWork<ReviewView, u64>,
        request: &CpuReciprocalRequest,
        id: ReciprocalQuestionId,
        _: ExecutionMode,
        root_budget: Digest,
    ) -> Result<AdaptiveReciprocalQuestion<u64>, AdaptiveManagerError> {
        let mut prefix = work.input.view.query().prefix.clone();
        prefix.extend_from_slice(request.relative_prefix());
        let mut endpoint = Position::from_snapshot(
            self.rules.rules().snapshot(),
            rz_position::PositionLimits::default(),
        )
        .unwrap();
        for movement in &prefix {
            endpoint.make_move(*movement).unwrap();
        }
        assert!(
            endpoint
                .snapshot()
                .same_state(&request.position().snapshot())
        );
        let legal = endpoint.legal_moves();
        let view = Arc::new(
            self.index
                .select(
                    &self.stores,
                    self.state,
                    ReviewQuerySequences {
                        purpose: if request.scope().recipient == SearchRecipient::Critic {
                            RoleQueryPurpose::ReplyPolicy
                        } else {
                            RoleQueryPurpose::ProposePolicy
                        },
                        prefix: &prefix,
                        proposal: &[],
                        counterexample: None,
                        legal: &legal,
                        divergences: &[],
                    },
                    ReviewSelectionPolicy::default(),
                )
                .unwrap(),
        );
        let view_seal = ReviewViewSeal {
            authority: self.authority,
            view_sequence: view.sequence(),
            context_policy: Digest(view.policy_digest()),
            view_digest: Digest(view.seal_digest()),
            root_state: self.rules.snapshot().identity(),
            actual_tokens: view.token_count(),
            selected_records: view.records().len() as u16,
            anchors: view.anchors().len() as u8,
            length_bucket: view.length_bucket(),
        };
        let anchor = ContractPosition::new(OwnerId(71), endpoint)
            .export()
            .unwrap();
        let question = QuestionEnvelope {
            id,
            parent: Some(request.scope().parent.id),
            sender: SearchRecipient::Researcher,
            recipient: request.scope().recipient,
            purpose: if request.scope().recipient == SearchRecipient::Critic {
                QuestionPurpose::Counter
            } else {
                QuestionPurpose::Propose
            },
            anchor: anchor.snapshot().identity(),
            anchor_root_ply: ChessPly(request.scope().parent.anchor_root_ply.0 + request.ply()),
            original_first_move: Some(work.original_first_move),
            prefix_digest: Digest(view.digest()),
            line_digest: None,
            line_offset: ChessPly(0),
            question_digest: Digest(view.digest()),
            view: view_seal,
            evidence_revision: 0,
            root_budget,
            original_deadline: work.original_deadline,
        };
        self.reciprocal_questions.insert(id.sequence, question);
        Ok(AdaptiveReciprocalQuestion {
            question,
            input: FrameInput {
                stage: if request.scope().recipient == SearchRecipient::Critic {
                    RefinementStage::CriticReply
                } else {
                    RefinementStage::ProposalContinuation
                },
                question: Arc::new(id.sequence),
                question_digest: question.question_digest,
                public_context: Digest(view.record_scope_digest()),
                view_seal,
                view,
            },
        })
    }

    pub(super) fn fixture_consume_reciprocal(
        &mut self,
        work: &CoordinatorWork<ReviewView, u64>,
        accepted: &AcceptedRoleStep,
        output: &RoleOutput,
        requests: &[(CpuReciprocalRequest, ReciprocalQuestionId)],
    ) -> Result<Vec<CpuReciprocalReply>, AdaptiveManagerError> {
        assert!(accepted.validates_output(output));
        self.validate_role(work, accepted.ticket(), output)?;
        let candidate_logits = match &output.payload {
            RolePayload::Proposal {
                candidate_logits, ..
            }
            | RolePayload::Counterexample {
                candidate_logits, ..
            } => candidate_logits,
            _ => unreachable!(),
        };
        let mut order: Vec<_> = work
            .input
            .view
            .query()
            .legal
            .iter()
            .copied()
            .zip(candidate_logits.iter().copied())
            .collect();
        order.sort_by(|left, right| right.1.total_cmp(&left.1));
        let mut replies = Vec::new();
        for (request, id) in requests {
            let question = self.reciprocal_questions[&id.sequence];
            let anchor = ContractPosition::new(OwnerId(71), request.position().clone())
                .export()
                .unwrap();
            let preferred = order
                .iter()
                .map(|(movement, _)| *movement)
                .filter(|movement| request.candidates().contains(movement))
                .collect();
            replies.push(CpuReciprocalReply::new(request, question, &anchor, preferred).unwrap());
        }
        self.reciprocal_replies += replies.len();
        Ok(replies)
    }
}

#[test]
fn actual_reciprocal_native_acceptance_resumes_the_same_cpu_owner_and_settles_once() {
    let (mut manager, facts, root, until) = fixture(FixtureScope::OwnCpuReciprocal, true, 4096);
    facts.child_gate.store(true, Ordering::Release);
    manager
        .configure_rovent_reciprocal(RoventReciprocalPolicy {
            mode: ExecutionMode::CpuPcr,
            root_budget: Digest([92; 32]),
            recipient: SearchRecipient::Proposer,
            max_questions_per_job: 1,
            node_interval: 1,
            first_question_sequence: 1,
        })
        .unwrap();
    manager.prepare_root_ranking(root).unwrap();
    pump(&mut manager, until, |manager| {
        manager.driver().actual_cpu.len() == 1 && !manager.has_pending_work()
    });
    assert!(manager.failures().next().is_none());
    assert_eq!(manager.driver().reciprocal_waits.len(), 1);
    assert_eq!(manager.driver().reciprocal_replies, 1);
    assert_eq!(manager.driver().consumed.len(), 3);
    let actual = manager.driver().actual_cpu[0].cumulative_nodes;
    assert_eq!(
        manager.native_ledger().unwrap().global_usage().unwrap(),
        (actual, 0)
    );
    assert_eq!(manager.counters().cpu_yields, 0);
    assert_eq!(facts.accepted.lock().unwrap().len(), 3);
    assert!(shutdown(&mut manager, until).permits_archive_close());
}

#[test]
fn reciprocal_mode_registration_rejects_r_free_and_late_opt_in_without_new_owners() {
    let (mut manager, _, root, until) = fixture(FixtureScope::TwoIndependentRoles, true, 4096);
    let policy = RoventReciprocalPolicy {
        mode: ExecutionMode::CpuPc,
        root_budget: Digest([92; 32]),
        recipient: SearchRecipient::Proposer,
        max_questions_per_job: 1,
        node_interval: 1,
        first_question_sequence: 1,
    };
    assert!(manager.configure_rovent_reciprocal(policy).is_err());
    assert!(manager.reciprocal_policy.is_none());
    manager.prepare_root_ranking(root).unwrap();
    assert!(
        manager
            .configure_rovent_reciprocal(RoventReciprocalPolicy {
                mode: ExecutionMode::CpuPcr,
                ..policy
            })
            .is_err()
    );
    assert!(shutdown(&mut manager, until).permits_archive_close());
}

#[test]
fn actual_r_to_critic_reply_preserves_the_original_owner_and_root_budget() {
    let (mut manager, facts, root, until) = fixture(FixtureScope::OwnCpuReciprocal, true, 4096);
    facts.child_gate.store(true, Ordering::Release);
    manager
        .configure_rovent_reciprocal(RoventReciprocalPolicy {
            mode: ExecutionMode::CpuPcr,
            root_budget: Digest([92; 32]),
            recipient: SearchRecipient::Critic,
            max_questions_per_job: 1,
            node_interval: 1,
            first_question_sequence: 1,
        })
        .unwrap();
    manager.prepare_root_ranking(root).unwrap();
    pump(&mut manager, until, |manager| {
        manager.driver().actual_cpu.len() == 1 && !manager.has_pending_work()
    });
    assert!(manager.failures().next().is_none());
    assert_eq!(manager.driver().reciprocal_replies, 1);
    assert_eq!(
        facts
            .accepted
            .lock()
            .unwrap()
            .iter()
            .filter(|ticket| ticket.role == Role::Critic)
            .count(),
        1
    );
    assert_eq!(
        manager.native_ledger().unwrap().global_usage().unwrap(),
        (manager.driver().actual_cpu[0].cumulative_nodes, 0)
    );
    assert!(shutdown(&mut manager, until).permits_archive_close());
}

#[test]
fn root_revoke_retains_waiting_cpu_token_until_actual_discard_and_never_consumes_reply() {
    let (mut manager, facts, root, until) = fixture(FixtureScope::OwnCpuReciprocal, true, 4096);
    facts.child_gate.store(true, Ordering::Release);
    manager
        .configure_rovent_reciprocal(RoventReciprocalPolicy {
            mode: ExecutionMode::CpuPcr,
            root_budget: Digest([92; 32]),
            recipient: SearchRecipient::Proposer,
            max_questions_per_job: 1,
            node_interval: 1,
            first_question_sequence: 1,
        })
        .unwrap();
    manager.prepare_root_ranking(root).unwrap();
    pump(&mut manager, until, |manager| {
        manager.own.iter().any(|slot| {
            slot.binding
                .as_ref()
                .is_some_and(|binding| binding.reciprocal.is_some())
        })
    });
    let pending_job = manager.own[0].binding.as_ref().unwrap().envelope.id;
    assert_eq!(
        manager
            .native_ledger()
            .unwrap()
            .usage(pending_job)
            .unwrap()
            .1,
        0
    );
    assert_eq!(
        manager
            .native_peer_unavailable(ExecutionMode::CpuPcr)
            .unwrap(),
        Some(PeerFollowupDisposition::NativeOwnerOccupied)
    );
    assert_eq!(
        manager.own[0].binding.as_ref().unwrap().envelope.id,
        pending_job
    );
    assert!(manager.driver().actual_cpu.is_empty());
    let receipt = shutdown(&mut manager, until);
    assert!(receipt.permits_archive_close());
    assert_eq!(manager.driver().reciprocal_replies, 0);
    assert!(manager.driver().actual_cpu.is_empty());
    assert_eq!(
        manager.native_ledger().unwrap().global_usage().unwrap().1,
        0
    );
    assert!(manager.own[0].binding.is_none());
}

#[test]
fn accepted_policy_sources_dispatch_both_peers_and_native_r_without_reexecuting_the_parent() {
    let (mut manager, facts, root, until) = fixture(FixtureScope::PeerAllDirections, true, 4096);
    facts.child_gate.store(true, Ordering::Release);
    manager
        .configure_rovent_reciprocal(RoventReciprocalPolicy {
            mode: ExecutionMode::CpuPcr,
            root_budget: Digest([92; 32]),
            recipient: SearchRecipient::Proposer,
            max_questions_per_job: 1,
            node_interval: 1,
            first_question_sequence: 1,
        })
        .unwrap();
    manager
        .configure_rovent_peer_followups(RoventPeerFollowupPolicy {
            pc_peer: true,
            researcher: true,
            max_sources_per_root: 2,
        })
        .unwrap();
    manager.prepare_root_ranking(root).unwrap();
    pump(&mut manager, until, |manager| {
        manager.driver().actual_cpu.len() == 1 && !manager.has_pending_work()
    });
    assert!(
        manager.failures().next().is_none(),
        "failures={:?}; peer_observations={:?}; peer_sources={:?}; opinions={}; R_answers={}; R_replies={}; CPU_receipts={}; phase={:?}",
        manager.failures().collect::<Vec<_>>(),
        manager.peer_followup_observations().collect::<Vec<_>>(),
        manager.driver().peer_sources,
        manager.driver().peer_opinions,
        manager.driver().peer_cpu_answers,
        manager.driver().reciprocal_replies,
        manager.driver().actual_cpu.len(),
        manager.phase
    );
    let driver = manager.driver();
    assert_eq!(
        driver.peer_sources,
        vec![
            (Role::Proposer, SearchRecipient::Critic),
            (Role::Proposer, SearchRecipient::Researcher),
            (Role::Critic, SearchRecipient::Proposer),
            (Role::Critic, SearchRecipient::Researcher),
        ]
    );
    assert_eq!(driver.peer_opinions, 2);
    assert_eq!(driver.peer_cpu_answers, 2);
    assert_eq!(driver.reciprocal_replies, 1);
    assert_eq!(facts.accepted.lock().unwrap().len(), 6);
    assert_eq!(driver.consumed.len(), 6);
    assert_eq!(
        manager
            .peer_followup_observations()
            .filter(|observation| observation.disposition == PeerFollowupDisposition::Admitted)
            .count(),
        4
    );
    assert!(shutdown(&mut manager, until).permits_archive_close());
}

#[test]
fn repeated_optional_peer_questions_close_as_typed_no_progress_and_commit_the_source() {
    let (mut manager, facts, root, until) = fixture(FixtureScope::PeerAllDirections, true, 4096);
    manager.driver_mut().peer_repeat_question = true;
    facts.child_gate.store(true, Ordering::Release);
    manager
        .configure_rovent_reciprocal(RoventReciprocalPolicy {
            mode: ExecutionMode::CpuPcr,
            root_budget: Digest([92; 32]),
            recipient: SearchRecipient::Proposer,
            max_questions_per_job: 1,
            node_interval: 1,
            first_question_sequence: 1,
        })
        .unwrap();
    manager
        .configure_rovent_peer_followups(RoventPeerFollowupPolicy {
            pc_peer: true,
            researcher: true,
            max_sources_per_root: 2,
        })
        .unwrap();
    manager.prepare_root_ranking(root).unwrap();
    pump(&mut manager, until, |manager| {
        manager.driver().actual_cpu.len() == 1 && !manager.has_pending_work()
    });
    assert!(
        manager.failures().next().is_none(),
        "failures={:?}; peers={:?}",
        manager.failures().collect::<Vec<_>>(),
        manager.peer_followup_observations().collect::<Vec<_>>()
    );
    assert!(
        manager
            .peer_followup_observations()
            .any(|observation| observation.disposition == PeerFollowupDisposition::NoProgress)
    );
    assert_eq!(manager.driver().peer_opinions, 1);
    assert_eq!(manager.driver().peer_cpu_answers, 1);
    assert_eq!(manager.driver().closed.len(), 1);
    assert_eq!(manager.driver().consumed.len(), 5);
    assert_eq!(facts.accepted.lock().unwrap().len(), 5);
    assert!(manager.accepted_roles.is_empty());
    assert_eq!(
        manager.native_ledger().unwrap().global_usage().unwrap().1,
        0
    );
    assert!(shutdown(&mut manager, until).permits_archive_close());
}
