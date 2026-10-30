//! Search-owned selected review publication and exact endpoint binding.
//!
//! An immutable view is a submitted input, not a mutable pointer to the newest
//! public memory. Frozen Fresh pairs retain one record/anchor scope while each
//! endpoint keeps its own query, actual input seal, and store read lease.
use std::sync::Arc;

use super::*;
use crate::pals::review_context::{
    ReviewContextIndex, ReviewError, ReviewQuerySequences, ReviewView,
};

pub(super) type FrozenReviewViews = (Arc<ReviewView>, Arc<ReviewView>);

/// Event-only lookup of the bounded raw projection. This is never called by a
/// native query. Parents and supersedes must already exist as raw facts; their
/// numeric revisions do not authorize invented relationships or new witnesses.
fn missing_review_closure<'a>(
    index: &ReviewContextIndex,
    records: &'a [RoleRecord],
    roots: impl IntoIterator<Item = u64>,
    limit: usize,
) -> Result<Vec<&'a RoleRecord>, ReviewError> {
    if records.len() > limit {
        return Err(ReviewError::IndexCapacity);
    }
    let mut raw = std::collections::BTreeMap::new();
    for record in records {
        if raw.insert(record.revision, record).is_some() {
            return Err(ReviewError::DuplicateRevision);
        }
    }
    let mut pending: Vec<_> = roots.into_iter().collect();
    let mut required = std::collections::BTreeSet::new();
    while let Some(revision) = pending.pop() {
        if index.contains_revision(revision) || !required.insert(revision) {
            continue;
        }
        let record = raw
            .get(&revision)
            .ok_or(ReviewError::MissingRelation(revision))?;
        for parent in [record.parent_revision, record.supersedes_revision]
            .into_iter()
            .flatten()
        {
            if parent >= revision {
                return Err(ReviewError::RelationMismatch);
            }
            pending.push(parent);
        }
    }
    Ok(required
        .into_iter()
        .map(|revision| raw[&revision])
        .collect())
}

impl<M: RoleModel> PalsSemanticState<M> {
    /// New-game store owners may restart numeric handles. Recreate the selected
    /// index rather than rebinding old IDs. Pair-owned views release by RAII.
    pub(super) fn reset_review_context(&mut self) {
        self.last_review_issue = None;
        self.review_index = self.review_index.as_ref().map(|index| {
            // Limits originate from the fallible validated constructor and are
            // immutable. Recreating its empty maps cannot invalidate them.
            ReviewContextIndex::new(index.limits()).expect("immutable validated review limits")
        });
    }

    /// Raw records remain owned by their original engine/archive owners. This
    /// only publishes a checked projection into the bounded active index.
    pub(super) fn publish_review_record(
        &mut self,
        config: &PalsSemanticConfig,
        record: &RoleRecord,
        evidence: Option<ObservationId>,
    ) -> Result<(), PalsError> {
        match (&mut self.review_index, self.review_policy) {
            (Some(index), Some(_)) => {
                let ancestors = missing_review_closure(
                    index,
                    &self.records,
                    [record.parent_revision, record.supersedes_revision]
                        .into_iter()
                        .flatten(),
                    config.max_records,
                )?;
                for ancestor in ancestors {
                    // Archive/index reappearance is not a new accepted witness.
                    index.restore_record(&self.stores, ancestor, None)?;
                }
                index.publish(&self.stores, record, evidence)?;
                Ok(())
            }
            (None, None) => Ok(()),
            _ => Err(PalsError::InvalidConfig),
        }
    }

    /// Raw engine topology has already been committed successfully by its
    /// exact-history archive loader. A selected-index issue cannot retroactively
    /// turn that successful raw recovery into a failed or incomplete load.
    pub(super) fn restore_review_context_after_load(&mut self, config: &PalsSemanticConfig) {
        let result = match (&mut self.review_index, self.review_policy) {
            (Some(index), Some(_)) => (|| -> Result<(), ReviewError> {
                let roots = self.records.iter().map(|record| record.revision);
                let records =
                    missing_review_closure(index, &self.records, roots, config.max_records)?;
                for record in records {
                    index.restore_record(&self.stores, record, None)?;
                }
                Ok(())
            })(),
            (None, None) => Ok(()),
            _ => Err(ReviewError::InvalidPolicy),
        };
        if let Err(issue) = result {
            self.last_review_issue = Some(issue);
        }
    }

    /// Bind the question's exact virtual prefix to the concrete engine node.
    /// Same-FEN and repetition keys never authorize this endpoint. Legal move
    /// order remains caller-owned, but every move must belong to this endpoint.
    fn review_root_for_question(
        &self,
        config: &PalsSemanticConfig,
        node: usize,
        question: RoleQuestion<'_>,
        legal: &[BoardMove],
    ) -> Result<StateId, PalsError> {
        let actual = self
            .stores
            .root()
            .ok_or(StoreError::InvalidHandle("review actual root"))?;
        let state = self.stores.situations.get(actual)?.state;
        let target = self
            .nodes
            .get(node)
            .ok_or(StoreError::InvalidHandle("review query node"))?;
        if question.prefix.len() > config.line_plies
            || question.proposal.len() > config.line_plies
            || question
                .refutation
                .is_some_and(|line| line.len() > config.line_plies)
            || legal.len() > 256
        {
            return Err(ReviewError::ViewCapacity.into());
        }
        let mut replay = Position::from_snapshot(
            self.stores.states.get(state)?,
            rz_position::PositionLimits::default(),
        )?;
        for movement in question.prefix {
            replay.make_move(*movement)?;
        }
        if !replay.snapshot().same_state(&target.position.snapshot())
            || !self
                .stores
                .states
                .get(target.state)?
                .same_state(&target.position.snapshot())
            || self.stores.situations.get(target.situation)?.state != target.state
        {
            return Err(ReviewError::AnchorMismatch.into());
        }
        let actual_legal = replay.legal_moves();
        let mut unique = std::collections::BTreeSet::new();
        for movement in legal {
            if !actual_legal.contains(movement) || !unique.insert(Move16::pack(*movement)?.bits()) {
                return Err(ReviewError::InvalidLine.into());
            }
        }
        // Divergence examines an ordered set of proposal indices rather than
        // the endpoint's move head. Other current heads receive all legal moves.
        if !matches!(
            question.purpose,
            RoleQueryPurpose::DivergencePolicy | RoleQueryPurpose::TaskPolicy
        ) && legal.len() != actual_legal.len()
        {
            return Err(ReviewError::InvalidLine.into());
        }
        Ok(state)
    }

    /// Dependent questions may not silently omit the very proposal/attack they
    /// are reviewing. The index lookup uses exact actual-root ownership and the
    /// original first move; an absent bundle is explicit context capacity loss,
    /// never a fabricated record revision or a guessed relation handle.
    fn review_policy_for_question(
        &self,
        root: StateId,
        question: RoleQuestion<'_>,
    ) -> Result<ReviewSelectionPolicy, PalsError> {
        let mut policy = self.review_policy.ok_or(PalsError::InvalidConfig)?;
        if policy.required_bundle.is_some()
            || matches!(
                question.purpose,
                RoleQueryPurpose::ProposePolicy | RoleQueryPurpose::ValueFresh
            )
        {
            return Ok(policy);
        }
        let proposal_first = question.proposal.first().copied();
        let refutation_first = question.refutation.and_then(|line| line.first().copied());
        if proposal_first
            .zip(refutation_first)
            .is_some_and(|(proposal, refutation)| proposal != refutation)
        {
            return Err(ReviewError::AnchorMismatch.into());
        }
        let Some(first) = proposal_first.or(refutation_first) else {
            // There is no dependent line in this question. Do not invent one.
            return Ok(policy);
        };
        let index = self.review_index.as_ref().ok_or(PalsError::InvalidConfig)?;
        policy.required_bundle = Some(
            index
                .bundle_for_first_move(root, Some(first))?
                .ok_or(ReviewError::ViewCapacity)?,
        );
        Ok(policy)
    }

    pub(super) fn prepare_review_view(
        &mut self,
        config: &PalsSemanticConfig,
        node: usize,
        question: RoleQuestion<'_>,
        legal: &[BoardMove],
    ) -> Result<Option<Arc<ReviewView>>, PalsError> {
        match (&self.review_index, self.review_policy) {
            (None, None) => return Ok(None),
            (Some(_), Some(_)) => {}
            _ => return Err(PalsError::InvalidConfig),
        }
        let root = self.review_root_for_question(config, node, question, legal)?;
        let query = ReviewQuerySequences {
            purpose: question.purpose,
            prefix: question.prefix,
            proposal: question.proposal,
            counterexample: question.refutation,
            legal,
            divergences: question.divergences,
        };
        let policy = self.review_policy_for_question(root, question)?;
        let index = self.review_index.as_mut().ok_or(PalsError::InvalidConfig)?;
        Ok(Some(Arc::new(index.select(
            &self.stores,
            root,
            query,
            policy,
        )?)))
    }

    /// Prepare both endpoint envelopes before either can publish an evaluation.
    /// The selector admits the larger query first and forks its immutable record
    /// scope, never selecting the second endpoint against newer public memory.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn prepare_frozen_review_views(
        &mut self,
        config: &PalsSemanticConfig,
        first_node: usize,
        first_question: RoleQuestion<'_>,
        first_legal: &[BoardMove],
        second_node: usize,
        second_question: RoleQuestion<'_>,
        second_legal: &[BoardMove],
    ) -> Result<Option<FrozenReviewViews>, PalsError> {
        match (&self.review_index, self.review_policy) {
            (None, None) => return Ok(None),
            (Some(_), Some(_)) => {}
            _ => return Err(PalsError::InvalidConfig),
        }
        let root =
            self.review_root_for_question(config, first_node, first_question, first_legal)?;
        if self.review_root_for_question(config, second_node, second_question, second_legal)?
            != root
        {
            return Err(ReviewError::AnchorMismatch.into());
        }
        let first = ReviewQuerySequences {
            purpose: first_question.purpose,
            prefix: first_question.prefix,
            proposal: first_question.proposal,
            counterexample: first_question.refutation,
            legal: first_legal,
            divergences: first_question.divergences,
        };
        let second = ReviewQuerySequences {
            purpose: second_question.purpose,
            prefix: second_question.prefix,
            proposal: second_question.proposal,
            counterexample: second_question.refutation,
            legal: second_legal,
            divergences: second_question.divergences,
        };
        let policy = self.review_policy_for_question(root, first_question)?;
        let second_policy = self.review_policy_for_question(root, second_question)?;
        if policy != second_policy {
            return Err(ReviewError::RelationMismatch.into());
        }
        let index = self.review_index.as_mut().ok_or(PalsError::InvalidConfig)?;
        let (first, second) =
            index.select_frozen_pair(&self.stores, root, first, second, policy)?;
        if first.record_scope_digest() != second.record_scope_digest()
            || first.root_state() != second.root_state()
            || first.generation() != second.generation()
            || !first.validates(&self.stores)
            || !second.validates(&self.stores)
        {
            return Err(ReviewError::RelationMismatch.into());
        }
        Ok(Some((first, second)))
    }
}

impl<M: RoleModel> PalsEngine<M> {
    pub(super) fn reset_review_context(&mut self) {
        self.semantic.reset_review_context();
    }
    #[cfg(test)]
    pub(super) fn publish_review_record(
        &mut self,
        record: &RoleRecord,
        evidence: Option<ObservationId>,
    ) -> Result<(), PalsError> {
        let config = self.semantic_sequence_config();
        self.semantic
            .publish_review_record(&config, record, evidence)
    }
    #[cfg(test)]
    pub(super) fn restore_review_context_after_load(&mut self) {
        let config = self.semantic_sequence_config();
        self.semantic.restore_review_context_after_load(&config);
    }
    #[cfg(test)]
    fn review_policy_for_question(
        &self,
        root: StateId,
        question: RoleQuestion<'_>,
    ) -> Result<ReviewSelectionPolicy, PalsError> {
        self.semantic.review_policy_for_question(root, question)
    }
    pub(super) fn prepare_review_view(
        &mut self,
        node: usize,
        question: RoleQuestion<'_>,
        legal: &[BoardMove],
    ) -> Result<Option<Arc<ReviewView>>, PalsError> {
        let config = self.semantic_sequence_config();
        self.semantic
            .prepare_review_view(&config, node, question, legal)
    }
    #[allow(clippy::too_many_arguments)]
    pub(super) fn prepare_frozen_review_views(
        &mut self,
        first_node: usize,
        first_question: RoleQuestion<'_>,
        first_legal: &[BoardMove],
        second_node: usize,
        second_question: RoleQuestion<'_>,
        second_legal: &[BoardMove],
    ) -> Result<Option<FrozenReviewViews>, PalsError> {
        let config = self.semantic_sequence_config();
        self.semantic.prepare_frozen_review_views(
            &config,
            first_node,
            first_question,
            first_legal,
            second_node,
            second_question,
            second_legal,
        )
    }

    /// Both Fresh WDL calls retain their own physical input lifetime. This
    /// helper freezes selected records only; it never reuses raw NN outputs or
    /// claims that different endpoint queries are the same model input.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn evaluate_model_value_pair(
        &mut self,
        first_node: usize,
        first_line: &[BoardMove],
        second_node: usize,
        second_line: &[BoardMove],
        limits: PalsLimits,
        cancel: &AtomicBool,
        counters: &mut PalsCounters,
    ) -> Result<(), PalsError> {
        // Preserve the old terminal/mixed/disabled path without adding view
        // allocation, model capability requirements, or new acceptance gates.
        if self.semantic.review_policy.is_none()
            || self.semantic.nodes[first_node].terminal.is_some()
            || self.semantic.nodes[second_node].terminal.is_some()
            || self.stopped(limits, cancel).is_some()
        {
            self.evaluate_model_value(first_node, first_line, limits, cancel, counters)?;
            return self.evaluate_model_value(second_node, second_line, limits, cancel, counters);
        }
        let revision = self.semantic.revision;
        let generation = self.semantic.stores.generation();
        let actual_root = self.semantic.stores.root();
        let identity = self
            .semantic
            .model_registered_value
            .as_ref()
            .ok_or(RoleError::Unavailable)?
            .clone();
        if self.semantic.model.value_identity() != Some(&identity) {
            return Err(RoleError::InvalidOutput.into());
        }
        let first_legal = self.semantic.nodes[first_node].position.legal_moves();
        let second_legal = self.semantic.nodes[second_node].position.legal_moves();
        let first_question = RoleQuestion {
            purpose: RoleQueryPurpose::ValueFresh,
            prefix: first_line,
            proposal: first_line,
            refutation: None,
            divergences: &[],
        };
        let second_question = RoleQuestion {
            purpose: RoleQueryPurpose::ValueFresh,
            prefix: second_line,
            proposal: second_line,
            refutation: None,
            divergences: &[],
        };
        let Some((first_view, second_view)) = self.prepare_frozen_review_views(
            first_node,
            first_question,
            &first_legal,
            second_node,
            second_question,
            &second_legal,
        )?
        else {
            return Err(PalsError::InvalidConfig);
        };
        self.evaluate_model_value_with_view(
            first_node,
            first_line,
            limits,
            cancel,
            counters,
            Some(first_view),
        )?;
        if self.semantic.revision != revision
            || self.semantic.stores.generation() != generation
            || self.semantic.stores.root() != actual_root
            || self.semantic.model.value_identity() != Some(&identity)
        {
            return Err(RoleError::InvalidOutput.into());
        }
        self.evaluate_model_value_with_view(
            second_node,
            second_line,
            limits,
            cancel,
            counters,
            Some(second_view),
        )?;
        if self.semantic.revision != revision
            || self.semantic.stores.generation() != generation
            || self.semantic.stores.root() != actual_root
            || self.semantic.model.value_identity() != Some(&identity)
        {
            return Err(RoleError::InvalidOutput.into());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pals::review_context::{ReviewIndexLimits, ReviewSelectionPolicy};

    fn selected_engine() -> PalsEngine<LegalOrderRoleMock> {
        PalsEngine::new(
            PalsConfig::default(),
            LegalOrderRoleMock,
            CpuEngine::new(crate::cpu::CpuConfig::default()).unwrap(),
        )
        .unwrap()
        .with_selected_review_context(
            ReviewSelectionPolicy::default(),
            ReviewIndexLimits::default(),
        )
        .unwrap()
    }

    fn setup_root(engine: &mut PalsEngine<LegalOrderRoleMock>) -> (Position, StateId) {
        let position = Position::startpos();
        let situation = engine
            .semantic
            .stores
            .focus_actual_moves(position.snapshot())
            .unwrap();
        let state = engine
            .semantic
            .stores
            .situations
            .get(situation)
            .unwrap()
            .state;
        engine.intern(position.clone()).unwrap();
        (position, state)
    }

    fn moves(text: &str) -> Vec<BoardMove> {
        text.split_whitespace()
            .map(|movement| BoardMove::from_uci(movement).unwrap())
            .collect()
    }

    fn value_question(line: &[BoardMove]) -> RoleQuestion<'_> {
        RoleQuestion {
            purpose: RoleQueryPurpose::ValueFresh,
            prefix: line,
            proposal: line,
            refutation: None,
            divergences: &[],
        }
    }

    fn raw_record(
        state: StateId,
        revision: u64,
        kind: RecordKind,
        line: &str,
        parent: Option<u64>,
    ) -> RoleRecord {
        RoleRecord {
            revision,
            parent_revision: parent,
            supersedes_revision: None,
            origin_state: state,
            kind,
            line: moves(line),
            value: None,
            completed_depth: 0,
            score_scope: None,
            cpu_observation: None,
            perspective: Color::White,
            critical: true,
        }
    }

    fn engine_with_retired_bundle() -> (PalsEngine<LegalOrderRoleMock>, StateId) {
        let mut engine = selected_engine();
        engine.semantic.review_index = Some(
            ReviewContextIndex::new(ReviewIndexLimits {
                active_bundles: 4,
                records_per_bundle: 64,
            })
            .unwrap(),
        );
        let (position, state) = setup_root(&mut engine);
        for record in [
            raw_record(state, 1, RecordKind::Proposal, "e2e4", None),
            raw_record(state, 2, RecordKind::Counterexample, "e2e4 e7e5", Some(1)),
        ] {
            engine.publish_review_record(&record, None).unwrap();
            engine.semantic.records.push(record);
        }
        // Two distinct eligible input consumptions, not two sequences carrying
        // one identical digest, exhaust the witness's observation protection.
        let witness_bundle = engine
            .semantic
            .review_index
            .as_ref()
            .unwrap()
            .bundle_for_revision(2)
            .unwrap();
        let mut previous_digest = None;
        for (ordinal, line) in ["", "e2e4"].into_iter().enumerate() {
            let prefix = moves(line);
            let mut endpoint = position.clone();
            for movement in &prefix {
                endpoint.make_move(*movement).unwrap();
            }
            let node = engine.intern(endpoint.clone()).unwrap();
            let view = engine
                .prepare_review_view(node, value_question(&prefix), &endpoint.legal_moves())
                .unwrap()
                .unwrap();
            assert_eq!(view.records().len(), 2);
            if let Some(previous) = previous_digest {
                assert_ne!(
                    previous,
                    view.digest(),
                    "independent eligible queries must differ"
                );
            }
            previous_digest = Some(view.digest());
            engine
                .semantic
                .review_index
                .as_mut()
                .unwrap()
                .consume(&view, &engine.semantic.stores)
                .unwrap();
            assert_eq!(
                engine
                    .semantic
                    .review_index
                    .as_ref()
                    .unwrap()
                    .bundle(witness_bundle)
                    .unwrap()
                    .protected_consumptions(),
                1 - ordinal as u8
            );
        }
        for record in [
            raw_record(state, 3, RecordKind::Proposal, "d2d4", None),
            raw_record(state, 4, RecordKind::Proposal, "c2c4", None),
            raw_record(state, 5, RecordKind::Proposal, "b2b4", None),
        ] {
            engine.publish_review_record(&record, None).unwrap();
            engine.semantic.records.push(record);
        }
        assert_eq!(
            engine
                .semantic
                .review_index
                .as_ref()
                .unwrap()
                .active_bundle_count(),
            4
        );
        assert_eq!(engine.semantic.review_index.as_mut().unwrap().compact(), 2);
        assert!(
            !engine
                .semantic
                .review_index
                .as_ref()
                .unwrap()
                .contains_revision(1)
        );
        assert!(
            !engine
                .semantic
                .review_index
                .as_ref()
                .unwrap()
                .contains_revision(2)
        );
        (engine, state)
    }

    #[test]
    fn endpoint_binding_requires_exact_history_not_equal_fen() {
        let mut engine = selected_engine();
        let (position, _) = setup_root(&mut engine);
        let imported = Position::from_fen(&position.to_fen()).unwrap();
        assert_eq!(imported.to_fen(), position.to_fen());
        let node = engine.intern(imported).unwrap();
        assert!(matches!(
            engine.prepare_review_view(node, value_question(&[]), &position.legal_moves()),
            Err(PalsError::Review(ReviewError::AnchorMismatch))
        ));
        assert_eq!(engine.semantic.stores.review_pin_count().unwrap(), 0);
    }

    #[test]
    fn endpoint_binding_rejects_prefix_target_and_legal_list_mismatch() {
        let mut engine = selected_engine();
        let (mut position, _) = setup_root(&mut engine);
        position.make_uci("e2e4").unwrap();
        let node = engine.intern(position.clone()).unwrap();
        let prefix = moves("d2d4");
        assert!(matches!(
            engine.prepare_review_view(node, value_question(&prefix), &position.legal_moves()),
            Err(PalsError::Review(ReviewError::AnchorMismatch))
        ));
        let prefix = moves("e2e4");
        let mut legal = position.legal_moves();
        legal.pop();
        assert!(matches!(
            engine.prepare_review_view(node, value_question(&prefix), &legal),
            Err(PalsError::Review(ReviewError::InvalidLine))
        ));
        assert_eq!(engine.semantic.stores.review_pin_count().unwrap(), 0);
    }

    #[test]
    fn frozen_pair_retains_one_record_scope_and_independent_query_leases() {
        let mut engine = selected_engine();
        let (position, state) = setup_root(&mut engine);
        let record = RoleRecord {
            revision: 1,
            parent_revision: None,
            supersedes_revision: None,
            origin_state: state,
            kind: RecordKind::Proposal,
            line: moves("g1f3"),
            value: None,
            completed_depth: 0,
            score_scope: None,
            cpu_observation: None,
            perspective: Color::White,
            critical: true,
        };
        engine.publish_review_record(&record, None).unwrap();
        let mut first_position = position.clone();
        first_position.make_uci("e2e4").unwrap();
        let mut second_position = position;
        second_position.make_uci("d2d4").unwrap();
        let first_node = engine.intern(first_position.clone()).unwrap();
        let second_node = engine.intern(second_position.clone()).unwrap();
        let first_line = moves("e2e4");
        let second_line = moves("d2d4");
        let (first, second) = engine
            .prepare_frozen_review_views(
                first_node,
                value_question(&first_line),
                &first_position.legal_moves(),
                second_node,
                value_question(&second_line),
                &second_position.legal_moves(),
            )
            .unwrap()
            .unwrap();
        assert_eq!(first.records().len(), 1);
        assert_eq!(first.record_scope_digest(), second.record_scope_digest());
        assert_ne!(first.digest(), second.digest());
        assert_ne!(first.seal_digest(), second.seal_digest());
        assert_eq!(engine.semantic.stores.review_pin_count().unwrap(), 2);
        drop(first);
        assert_eq!(engine.semantic.stores.review_pin_count().unwrap(), 1);
        assert!(second.validates(&engine.semantic.stores));
        drop(second);
        assert_eq!(engine.semantic.stores.review_pin_count().unwrap(), 0);
    }

    #[test]
    fn index_reset_keeps_registered_limits_and_revokes_previous_index_consumption() {
        let mut engine = selected_engine();
        let (position, _) = setup_root(&mut engine);
        let root_node = engine.intern(position.clone()).unwrap();
        let view = engine
            .prepare_review_view(root_node, value_question(&[]), &position.legal_moves())
            .unwrap()
            .unwrap();
        let limits = engine.semantic.review_index.as_ref().unwrap().limits();
        engine.semantic.last_review_issue = Some(ReviewError::ViewCapacity);
        engine.reset_review_context();
        assert_eq!(
            engine.semantic.review_index.as_ref().unwrap().limits(),
            limits
        );
        assert!(engine.semantic.last_review_issue.is_none());
        assert!(matches!(
            engine
                .semantic
                .review_index
                .as_mut()
                .unwrap()
                .consume(&view, &engine.semantic.stores),
            Err(ReviewError::OwnerMismatch)
        ));
        assert_eq!(engine.semantic.stores.review_pin_count().unwrap(), 1);
        drop(view);
        assert_eq!(engine.semantic.stores.review_pin_count().unwrap(), 0);
    }

    #[test]
    fn cold_index_restore_preserves_raw_facts_without_reviving_witness_protection() {
        let (mut engine, _) = engine_with_retired_bundle();
        let before: Vec<_> = engine
            .semantic
            .records
            .iter()
            .map(|record| (record.revision, record.line.clone(), record.parent_revision))
            .collect();
        engine.restore_review_context_after_load();
        assert!(engine.semantic.last_review_issue.is_none());
        let index = engine.semantic.review_index.as_ref().unwrap();
        let bundle = index.bundle_for_revision(2).unwrap();
        assert_eq!(index.bundle(bundle).unwrap().protected_consumptions(), 0);
        assert_eq!(index.bundle(bundle).unwrap().revisions(), &[1, 2]);
        assert_eq!(
            before,
            engine
                .semantic
                .records
                .iter()
                .map(|record| (record.revision, record.line.clone(), record.parent_revision))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn new_repair_restores_old_relation_closure_and_protects_only_the_new_witness() {
        let (mut engine, state) = engine_with_retired_bundle();
        let mut repair = raw_record(state, 6, RecordKind::Repair, "e2e4 e7e5 g1f3", Some(2));
        repair.supersedes_revision = Some(1);
        engine.publish_review_record(&repair, None).unwrap();
        let index = engine.semantic.review_index.as_ref().unwrap();
        let bundle = index.bundle_for_revision(6).unwrap();
        assert_eq!(index.bundle(bundle).unwrap().revisions(), &[1, 2, 6]);
        assert_eq!(index.bundle(bundle).unwrap().protected_consumptions(), 2);
        // This helper only publishes the index. Its caller owns raw commit.
        assert_eq!(engine.semantic.records.len(), 5);
    }

    #[test]
    fn successful_raw_load_keeps_typed_missing_relation_as_separate_review_issue() {
        let mut engine = selected_engine();
        let (_, state) = setup_root(&mut engine);
        engine.semantic.records.push(raw_record(
            state,
            10,
            RecordKind::Repair,
            "e2e4 e7e5",
            Some(9),
        ));
        let nodes_before = engine.semantic.nodes.len();
        engine.restore_review_context_after_load();
        assert_eq!(
            engine.semantic.last_review_issue,
            Some(ReviewError::MissingRelation(9))
        );
        assert_eq!(engine.semantic.records.len(), 1);
        assert_eq!(engine.semantic.records[0].revision, 10);
        assert_eq!(engine.semantic.nodes.len(), nodes_before);
        assert!(
            !engine
                .semantic
                .review_index
                .as_ref()
                .unwrap()
                .contains_revision(10)
        );
    }

    #[test]
    fn dependent_question_requires_its_actual_coherent_bundle() {
        let mut engine = selected_engine();
        let (position, state) = setup_root(&mut engine);
        let node = engine.intern(position.clone()).unwrap();
        let proposal = moves("e2e4");
        let counter = moves("e2e4 e7e5");
        let question = RoleQuestion {
            purpose: RoleQueryPurpose::RepairPolicy,
            prefix: &[],
            proposal: &proposal,
            refutation: Some(&counter),
            divergences: &[],
        };
        assert!(matches!(
            engine.prepare_review_view(node, question, &position.legal_moves()),
            Err(PalsError::Review(ReviewError::ViewCapacity))
        ));
        for record in [
            raw_record(state, 1, RecordKind::Proposal, "e2e4", None),
            raw_record(state, 2, RecordKind::Counterexample, "e2e4 e7e5", Some(1)),
            raw_record(state, 3, RecordKind::Repair, "e2e4 e7e5 g1f3", Some(2)),
        ] {
            engine.publish_review_record(&record, None).unwrap();
            engine.semantic.records.push(record);
        }
        engine.semantic.review_policy.as_mut().unwrap().max_records = 2;
        // A three-record dependency cannot be trimmed to fit two records.
        assert!(matches!(
            engine.prepare_review_view(node, question, &position.legal_moves()),
            Err(PalsError::Review(ReviewError::ViewCapacity))
        ));
        assert_eq!(engine.semantic.stores.review_pin_count().unwrap(), 0);
    }

    #[test]
    fn dependent_policy_rejects_first_move_mismatch_and_keeps_explicit_authority() {
        let mut engine = selected_engine();
        let (_, state) = setup_root(&mut engine);
        let proposal = moves("e2e4");
        let counter = moves("d2d4 d7d5");
        let question = RoleQuestion {
            purpose: RoleQueryPurpose::ReplyPolicy,
            prefix: &[],
            proposal: &proposal,
            refutation: Some(&counter),
            divergences: &[],
        };
        assert!(matches!(
            engine.review_policy_for_question(state, question),
            Err(PalsError::Review(ReviewError::AnchorMismatch))
        ));
        let record = raw_record(state, 1, RecordKind::Proposal, "d2d4", None);
        engine.publish_review_record(&record, None).unwrap();
        let explicit = engine
            .semantic
            .review_index
            .as_ref()
            .unwrap()
            .bundle_for_revision(1)
            .unwrap();
        engine
            .semantic
            .review_policy
            .as_mut()
            .unwrap()
            .required_bundle = Some(explicit);
        assert_eq!(
            engine
                .review_policy_for_question(state, question)
                .unwrap()
                .required_bundle,
            Some(explicit)
        );
        assert_eq!(engine.semantic.stores.review_pin_count().unwrap(), 0);
    }

    #[test]
    fn selected_search_runs_actual_mock_refinement_and_matches_legacy_work() {
        let position = Position::startpos();
        let cancel = AtomicBool::new(false);
        let limits = || PalsLimits {
            max_rounds: 1,
            max_cpu_nodes: 2_000,
            cpu_depth: 1,
            deadline: Instant::now() + Duration::from_secs(20),
        };
        let mut legacy = PalsEngine::new(
            PalsConfig::default(),
            LegalOrderRoleMock,
            CpuEngine::new(crate::cpu::CpuConfig::default()).unwrap(),
        )
        .unwrap();
        let expected = legacy.search(&position, limits(), &cancel).unwrap();
        let mut selected = selected_engine();
        let actual = selected.search(&position, limits(), &cancel).unwrap();

        // Exercise real ranking, divergence, publication, and CPU refinement;
        // successful standalone view preparation is not this integration proof.
        assert_eq!(actual.counters.rounds, 1);
        assert!(actual.counters.accepted_proposer_outputs > 0);
        assert!(actual.counters.accepted_critic_outputs > 0);
        assert!(actual.counters.completed_cpu_tasks > 0);
        assert!(!selected.semantic.records.is_empty());
        assert!(position.legal_moves().contains(&actual.best_move.unwrap()));
        assert_eq!(actual.best_move, expected.best_move);
        assert_eq!(actual.score, expected.score);
        assert_eq!(actual.value_scope, expected.value_scope);
        assert_eq!(actual.completion, expected.completion);
        assert_eq!(actual.counters, expected.counters);
        assert!(selected.semantic.last_review_issue.is_none());
        assert_eq!(selected.semantic.stores.review_pin_count().unwrap(), 0);

        // Inspect a selected view after actual publication, then release its
        // lease. This is a correctness bound, not a time or memory acceptance.
        let root = selected.intern(position.clone()).unwrap();
        let view = selected
            .prepare_review_view(root, value_question(&[]), &position.legal_moves())
            .unwrap()
            .unwrap();
        assert!(!view.records().is_empty());
        assert!(view.records().len() <= 64);
        drop(view);
        assert_eq!(selected.semantic.stores.review_pin_count().unwrap(), 0);
        assert!(selected.semantic.last_review_issue.is_none());
    }
}
