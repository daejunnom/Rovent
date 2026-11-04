//! Original consumed contextual Task provenance. It owns no ReviewView pin,
//! cannot submit/resume work, and survives the bounded admission's retirement
//! only beside its actual retained Node evidence.
use super::*;
use crate::cpu_worker::CpuWorkerProfile;
use crate::pals::store::StoreReadOwner;
use rz_contracts::pals_adaptive::{CpuJobEnvelope, CpuWorkerId};

#[derive(Clone, Debug)]
pub(in crate::pals::engine) struct CpuEvidenceContextSeal {
    store: StoreReadOwner,
    generation: u64,
    root: StateId,
    authority: SearchAuthority,
    frame: FrameId,
    question: ContractDigest,
    task: TaskKey,
    execution: ExecutionId,
    observation: ObservationId,
    job: CpuJobEnvelope,
    owner: CpuWorkerId,
    profile: CpuWorkerProfile,
    original_consumer: TaskConsumer,
}

impl CpuEvidenceContextSeal {
    /// Fixed provenance (including its original TaskConsumer) and the cloned
    /// TaskKey's owned namespace payload. Arc headers, Store/map internals and
    /// shared allocations remain outside this known subset.
    pub(super) fn known_owned_backing_bytes(&self) -> Option<u64> {
        u64::try_from(std::mem::size_of::<Self>())
            .ok()?
            .checked_add(self.task.known_owned_payload_bytes()?)
    }
}

/// Live installation admission is checked before the Store consumes anything.
/// Finalizing that already consumed fact never reopens a canceled authority.
pub(super) struct PreparedCpuEvidenceContext {
    profile: CpuWorkerProfile,
}

impl<M: RoleModel> AdaptiveEnginePort<M> {
    pub(super) fn prepare_cpu_context_seal(
        &self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        admission: &CpuPortAdmission,
    ) -> Result<Option<PreparedCpuEvidenceContext>, PalsError> {
        if self.engine.owned_identity()?.semantics != crate::cpu_value::nnuec::v2::SEMANTICS {
            return Ok(None);
        }
        let profile = self
            .installed_cpu_binding(work)?
            .ok_or(StoreError::InvalidEvidence(
                "consumed CPU context has no original installation",
            ))?;
        if admission.owner != Some(profile.owner)
            || admission.profiles.as_slice() != [profile]
            || profile.evaluation_namespace != admission.evaluation_namespace
            || profile.search_conditions != admission.search_conditions
        {
            return Err(StoreError::InvalidEvidence(
                "CPU context differs from original installed profile",
            )
            .into());
        }
        self.admit_utility_conditional_cpu_context(&admission.task_key)?;
        Ok(Some(PreparedCpuEvidenceContext { profile }))
    }

    pub(super) fn consumed_cpu_context_seal(
        &self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        admission: &CpuPortAdmission,
        prepared: Option<PreparedCpuEvidenceContext>,
        execution: ExecutionId,
        observation: ObservationId,
    ) -> Result<Option<Arc<CpuEvidenceContextSeal>>, PalsError> {
        let Some(PreparedCpuEvidenceContext { profile }) = prepared else {
            return Ok(None);
        };
        let stored = self.cpu_admissions.get(&work.input.question_digest).ok_or(
            StoreError::InvalidEvidence("consumed CPU context lost original admission"),
        )?;
        let job = admission.job.ok_or(StoreError::InvalidEvidence(
            "consumed CPU context has no Job",
        ))?;
        let owner = admission.owner.ok_or(StoreError::InvalidEvidence(
            "consumed CPU context has no owner",
        ))?;
        let original_consumer = admission
            .original_consumer
            .ok_or(StoreError::InvalidEvidence(
                "consumed CPU context has no original consumer",
            ))?;
        if stored.consumed_observation != Some(observation)
            || stored
                .committed_execution
                .is_none_or(|(_, original, observed)| {
                    original != execution || observed != observation
                })
            || admission.execution != Some(execution)
            || stored.original_consumer != Some(original_consumer)
            || profile.owner != owner
            || admission.profiles.as_slice() != [profile]
            || profile.evaluation_namespace != admission.evaluation_namespace
            || profile.search_conditions != admission.search_conditions
            || job.question != work.input.question_digest
            || job.id.authority != self.authority
            || job.evaluation_namespace != admission.evaluation_namespace
            || job.search_conditions != admission.search_conditions
            || self.engine.semantic.stores.tasks.get(execution)?.key != admission.task_key
            || self
                .engine
                .semantic
                .stores
                .observations
                .get(observation)?
                .execution
                != Some(execution)
        {
            return Err(StoreError::InvalidEvidence(
                "CPU context differs from original consumed task",
            )
            .into());
        }
        // Consumption may have grown the original Store consumer vector.
        // Recheck its actual retained capacity before adding another seal.
        self.admit_utility_conditional_cpu_context(&admission.task_key)?;
        Ok(Some(Arc::new(CpuEvidenceContextSeal {
            store: self.engine.semantic.stores.read_owner(),
            generation: self.engine.semantic.stores.generation(),
            root: self.root_state_id(),
            authority: self.authority,
            frame: work.frame,
            question: work.input.question_digest,
            task: admission.task_key.clone(),
            execution,
            observation,
            job,
            owner,
            profile,
            original_consumer,
        })))
    }

    fn admit_utility_conditional_cpu_context(&self, task: &TaskKey) -> Result<(), PalsError> {
        if !self.utility_continuations.enabled() {
            return Ok(());
        }
        let additional = u64::try_from(std::mem::size_of::<CpuEvidenceContextSeal>())
            .ok()
            .and_then(|fixed| fixed.checked_add(task.known_owned_payload_bytes()?))
            .ok_or(PalsError::Capacity)?;
        self.check_utility_conditional_known_backing(additional)
    }

    /// Project only the exact source already consumed by this Node. It adds
    /// no authority to a raw Task row and stays usable for history after a
    /// deadline or cancellation. Missing old provenance remains unknown.
    pub(super) fn utility_conditional_cpu_metadata(
        &self,
        node: usize,
        observation: ObservationId,
        execution: ExecutionId,
    ) -> Result<Option<super::utility_conditional_basis::UtilityCpuAcceptedMetadata>, PalsError>
    {
        use super::utility_conditional_basis::{
            UtilityCpuAcceptedMetadata, UtilityCpuContextMetadata,
        };
        let endpoint = self
            .engine
            .semantic
            .nodes
            .get(node)
            .ok_or(RoleError::InvalidOutput)?;
        let Some(evidence) = &endpoint.evidence else {
            return Ok(None);
        };
        let task = self.engine.semantic.stores.tasks.get(execution)?;
        if evidence.provenance != Some((observation, execution))
            || !matches!(task.status, TaskStatus::Completed(id) if id == observation)
            || !self.conditional_cpu_task_condition_matches(
                evidence,
                &task.key,
                execution,
                observation,
            )
        {
            return Ok(None);
        }
        if let Some(context) = &evidence.context {
            if !self
                .engine
                .semantic
                .stores
                .tasks
                .historical_consumer_was_consumed(execution, context.original_consumer)?
            {
                return Ok(None);
            }
            return Ok(Some(UtilityCpuAcceptedMetadata {
                original_consumer: context.original_consumer,
                contextual: Some(UtilityCpuContextMetadata {
                    generation: context.generation,
                    root: context.root,
                    authority: context.authority,
                    frame: context.frame,
                    question: context.question,
                    task: context.task.clone(),
                    execution: context.execution,
                    observation: context.observation,
                    job: context.job,
                    owner: context.owner,
                    profile: context.profile,
                    original_consumer: context.original_consumer,
                }),
                legacy_condition: None,
            }));
        }
        let mut originals = self.cpu_admissions.values().filter(|admission| {
            admission.node == node
                && admission.execution == Some(execution)
                && admission.consumed_observation == Some(observation)
                && admission
                    .committed_execution
                    .is_some_and(|(_, actual, observed)| {
                        actual == execution && observed == observation
                    })
                && admission.generation == self.engine.semantic.stores.generation()
                && admission.task_key == task.key
        });
        let Some(admission) = originals.next() else {
            return Ok(None);
        };
        if originals.next().is_some() {
            return Err(StoreError::InvalidEvidence("ambiguous original CPU consumer").into());
        }
        let Some(original_consumer) = admission.original_consumer else {
            return Ok(None);
        };
        if !self
            .engine
            .semantic
            .stores
            .tasks
            .historical_consumer_was_consumed(execution, original_consumer)?
        {
            return Ok(None);
        }
        Ok(Some(UtilityCpuAcceptedMetadata {
            original_consumer,
            contextual: None,
            legacy_condition: Some(self.engine.cpu_condition()),
        }))
    }

    pub(super) fn conditional_cpu_task_condition_matches(
        &self,
        accepted: &CpuEvidence,
        task: &TaskKey,
        execution: ExecutionId,
        observation: ObservationId,
    ) -> bool {
        let Some(context) = &accepted.context else {
            return task.cpu_condition.as_ref() == Some(&self.engine.cpu_condition());
        };
        context.store.matches(&self.engine.semantic.stores)
            && context.generation == self.engine.semantic.stores.generation()
            && context.authority == self.authority
            && context.frame.authority == self.authority
            && context.root == self.root_state_id()
            && context.task == *task
            && context.execution == execution
            && context.observation == observation
            && context.job.id.authority == self.authority
            && context.job.question == context.question
            && context.job.evaluation_namespace == context.profile.evaluation_namespace
            && context.job.search_conditions == context.profile.search_conditions
            && context.profile.owner == context.owner
            && accepted.provenance == Some((observation, execution))
            && task.value_identity.as_ref() == Some(&accepted.value_identity)
    }
}

#[cfg(test)]
mod tests {
    use super::super::conditional_evidence::tests::{accepted_raw_source, port as contract_port};
    use super::*;
    use rz_contracts::pals_adaptive::CpuJobId;

    // Reuse the Store semantic fixture, which makes no physical search claim.
    // Actual Bound/worker minting is exercised by the UCI owner integration.
    fn consumed_context(
        port: &AdaptiveEnginePort<LegalOrderRoleMock>,
        node: usize,
        original_consumer: TaskConsumer,
    ) -> CpuEvidenceContextSeal {
        let evidence = port.engine.semantic.nodes[node].evidence.as_ref().unwrap();
        let (observation, execution) = evidence.provenance.unwrap();
        let task = port.engine.semantic.stores.tasks.get(execution).unwrap();
        assert!(matches!(task.status, TaskStatus::Completed(id) if id == observation));
        let profile = CpuWorkerProfile {
            owner: CpuWorkerId(1),
            max_depth: 8,
            evaluation_namespace: ContractDigest([31; 32]),
            search_conditions: ContractDigest([32; 32]),
        };
        let question = ContractDigest([33; 32]);
        CpuEvidenceContextSeal {
            store: port.engine.semantic.stores.read_owner(),
            generation: port.engine.semantic.stores.generation(),
            root: port.root_state_id(),
            authority: port.authority,
            frame: FrameId {
                authority: port.authority,
                sequence: 1,
            },
            question,
            task: task.key.clone(),
            execution,
            observation,
            job: CpuJobEnvelope {
                id: CpuJobId {
                    authority: port.authority,
                    sequence: 1,
                },
                state_identity: port
                    .issue_rules_position(port.engine.semantic.nodes[node].position.clone())
                    .unwrap()
                    .export()
                    .unwrap()
                    .snapshot()
                    .identity(),
                question,
                root_restriction: ContractDigest([0; 32]),
                evaluation_namespace: profile.evaluation_namespace,
                search_conditions: profile.search_conditions,
                total_native_node_credit: task.key.node_budget,
                hard_deadline: port.deadline,
            },
            owner: profile.owner,
            profile,
            original_consumer,
        }
    }

    #[test]
    fn retained_consumed_context_does_not_depend_on_active_admission() {
        let mut port = contract_port(Position::startpos());
        let movement = BoardMove::from_uci("e2e4").unwrap();
        let (_, consumer) = accepted_raw_source(&mut port, &[movement], 14, 2);
        let node = port.replay_node(&[movement]).unwrap();
        let seal = consumed_context(&port, node, consumer);
        let task = seal.task.clone();
        let execution = seal.execution;
        let observation = seal.observation;
        port.engine.semantic.nodes[node]
            .evidence
            .as_mut()
            .unwrap()
            .context = Some(Arc::new(seal));
        port.cpu_admissions.clear();
        let evidence = port.engine.semantic.nodes[node].evidence.as_ref().unwrap();
        assert!(port.conditional_cpu_task_condition_matches(
            evidence,
            &task,
            execution,
            observation
        ));
        let metadata = port
            .utility_conditional_cpu_metadata(node, observation, execution)
            .unwrap()
            .unwrap();
        assert_eq!(metadata.original_consumer, consumer);
        assert!(metadata.contextual.is_some());
        let foreign = contract_port(Position::startpos());
        assert!(!foreign.conditional_cpu_task_condition_matches(
            evidence,
            &task,
            execution,
            observation
        ));
        assert!(!port.conditional_cpu_task_condition_matches(
            evidence,
            &task,
            execution,
            ObservationId(observation.0 + 1)
        ));
        let mut other_task = task;
        other_task
            .cpu_condition
            .as_mut()
            .unwrap()
            .push_str(";nnuec-context-binding=another-view");
        assert!(!port.conditional_cpu_task_condition_matches(
            evidence,
            &other_task,
            execution,
            observation
        ));
    }
}
