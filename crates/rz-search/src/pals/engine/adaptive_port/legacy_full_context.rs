//! Submission-time ownership for the original V2 full-context baseline.
//!
//! This is not a ReviewView and never raises its selected token/row limits.
//! Search owns the only mutable public-memory writer. It freezes the complete
//! bounded raw projection here, then workers run the existing full encoder on
//! that immutable projection. Original encoder filtering/order/overflow rules
//! remain at the native boundary rather than being repeated as a new selector.

use super::*;
use crate::pals::store::StoreReadPin;

/// The existing legacy encoder rejects more than 16,384 scanned raw rows.
/// The engine's independently validated max_records (currently at most 128)
/// remains the tighter source-projection bound; neither limit is truncated.
const LEGACY_ENCODER_RAW_RECORD_SCAN_LIMIT: usize = 16_384;
const LEGACY_ENCODER_LINE_LIMIT: usize = 256;

/// Search-issued immutable raw input and its original Store dependency lease.
/// A native caller cannot create this from imported rows, JSON or a selected
/// ReviewView, and never needs access to mutable Store memory during encoding.
#[derive(Debug)]
pub struct LegacyFullRoleRecords {
    records: Vec<RoleRecord>,
    authority: SearchAuthority,
    revision: u64,
    generation: u64,
    root_state: [u8; 32],
    digest: [u8; 32],
    pin: StoreReadPin,
}

impl LegacyFullRoleRecords {
    pub fn authority(&self) -> SearchAuthority {
        self.authority
    }
    pub fn records(&self) -> &[RoleRecord] {
        &self.records
    }
    pub fn revision(&self) -> u64 {
        self.revision
    }
    pub fn generation(&self) -> u64 {
        self.generation
    }
    /// Exact Rules/history semantic digest, not a numeric StateId or FEN key.
    pub fn root_state(&self) -> [u8; 32] {
        self.root_state
    }
    /// Authority digest of every captured raw row, including rows that the
    /// original encoder may subsequently omit under its registered width.
    pub fn digest(&self) -> [u8; 32] {
        self.digest
    }
    pub fn validates(&self, stores: &PalsStores) -> bool {
        self.pin.validates(stores)
    }
    pub(crate) fn retirement_protected_pins(
        &self,
        stores: &PalsStores,
    ) -> Result<crate::pals::store::StorePins, PalsError> {
        Ok(self.pin.retirement_seed(stores)?)
    }
    /// Known owned snapshot backing, not Store resident bytes or an RSS peak.
    /// Shared copies are conservatively charged per submitted question; no
    /// physical sharing deduction is inferred from the Arc identity.
    pub fn owned_host_bytes(&self) -> Option<u64> {
        let mut bytes = std::mem::size_of::<Self>().checked_add(
            self.records
                .capacity()
                .checked_mul(std::mem::size_of::<RoleRecord>())?,
        )?;
        for record in &self.records {
            bytes = bytes.checked_add(
                record
                    .line
                    .capacity()
                    .checked_mul(std::mem::size_of::<BoardMove>())?,
            )?;
        }
        u64::try_from(bytes).ok()
    }

    fn capture<M: RoleModel>(
        engine: &PalsEngine<M>,
        authority: SearchAuthority,
    ) -> Result<Self, PalsError> {
        if engine.semantic.records.len() > engine.config.max_records
            || engine.semantic.records.len() > LEGACY_ENCODER_RAW_RECORD_SCAN_LIMIT
        {
            return Err(PalsError::Capacity);
        }
        if engine.semantic.game_generation != Some(authority.game.0)
            || engine.semantic.stores.generation() != authority.root.0
        {
            return Err(StoreError::StaleConsumer.into());
        }
        let root = engine
            .semantic
            .stores
            .root()
            .ok_or(StoreError::InvalidHandle("legacy full context actual root"))?;
        let root_state = engine.semantic.stores.situations.get(root)?.state;
        let root_digest = rz_position::contracts::rules_snapshot_semantic_digest(
            engine.semantic.stores.states.get(root_state)?,
        )
        .0;
        let mut hash = Sha256::new();
        hash.update(b"rz-pals-legacy-full-raw-snapshot/1\0");
        hash.update(authority.epoch.0.to_le_bytes());
        hash.update(authority.game.0.to_le_bytes());
        hash.update(authority.root.0.to_le_bytes());
        hash.update(authority.implementation.0);
        hash.update(engine.semantic.revision.to_le_bytes());
        hash.update(engine.semantic.stores.generation().to_le_bytes());
        hash.update(root_digest);
        hash.update(
            u64::try_from(engine.semantic.records.len())
                .map_err(|_| PalsError::Capacity)?
                .to_le_bytes(),
        );
        let mut records = Vec::new();
        records
            .try_reserve_exact(engine.semantic.records.len())
            .map_err(|_| PalsError::Capacity)?;
        for source in &engine.semantic.records {
            if source.revision > engine.semantic.revision
                || source.line.len() > LEGACY_ENCODER_LINE_LIMIT
            {
                return Err(RoleError::InvalidOutput.into());
            }
            // Keep the raw row exactly: no rebase, suffix selection, relation
            // rewrite, changed critical status, or injected actual-game line.
            let mut line = Vec::new();
            line.try_reserve_exact(source.line.len())
                .map_err(|_| PalsError::Capacity)?;
            hash.update(source.revision.to_le_bytes());
            for revision in [source.parent_revision, source.supersedes_revision] {
                hash.update([u8::from(revision.is_some())]);
                if let Some(revision) = revision {
                    hash.update(revision.to_le_bytes());
                }
            }
            hash.update(
                rz_position::contracts::rules_snapshot_semantic_digest(
                    engine.semantic.stores.states.get(source.origin_state)?,
                )
                .0,
            );
            hash.update([match source.kind {
                RecordKind::Proposal => 0,
                RecordKind::Counterexample => 1,
                RecordKind::Repair => 2,
                RecordKind::CpuVerification => 3,
            }]);
            hash.update(
                u64::try_from(source.line.len())
                    .map_err(|_| PalsError::Capacity)?
                    .to_le_bytes(),
            );
            for &movement in &source.line {
                hash.update(Move16::pack(movement)?.bits().to_le_bytes());
                line.push(movement);
            }
            hash.update([u8::from(source.value.is_some())]);
            if let Some(value) = source.value {
                hash.update(value.to_le_bytes());
            }
            hash.update(source.completed_depth.to_le_bytes());
            hash.update([match source.score_scope {
                None => 0,
                Some(CpuScoreScope::FrontierOnly) => 1,
                Some(CpuScoreScope::CompletedIteration) => 2,
                Some(CpuScoreScope::RulesTerminal) => 3,
            }]);
            hash.update([u8::from(source.cpu_observation.is_some())]);
            if let Some(observation) = source.cpu_observation {
                engine.semantic.stores.observations.get(observation)?;
                hash.update(
                    u64::try_from(observation.0)
                        .map_err(|_| PalsError::Capacity)?
                        .to_le_bytes(),
                );
            }
            hash.update([
                u8::from(source.perspective == Color::Black),
                u8::from(source.critical),
            ]);
            records.push(RoleRecord {
                revision: source.revision,
                parent_revision: source.parent_revision,
                supersedes_revision: source.supersedes_revision,
                origin_state: source.origin_state,
                kind: source.kind,
                line,
                value: source.value,
                completed_depth: source.completed_depth,
                score_scope: source.score_scope,
                cpu_observation: source.cpu_observation,
                perspective: source.perspective,
                critical: source.critical,
            });
        }
        // This baseline-only walk reuses the engine's exact resident closure,
        // including pending/public model observations, CPU provenance, and
        // Store-expanded dependencies. Selected/compact never performs it.
        let mut pins = engine.resident_archive_pins()?;
        pins.states.insert(root_state);
        pins.situations.insert(root);
        let pin = engine.semantic.stores.pin_review(pins)?;
        let snapshot = Self {
            records,
            authority,
            revision: engine.semantic.revision,
            generation: engine.semantic.stores.generation(),
            root_state: root_digest,
            digest: hash.finalize().into(),
            pin,
        };
        snapshot.owned_host_bytes().ok_or(PalsError::Capacity)?;
        Ok(snapshot)
    }
}

impl<M: RoleModel> AdaptiveEnginePort<M> {
    pub(super) fn prepare_legacy_full_records(
        &mut self,
        context: &RoleLogicalContext,
        view: &ReviewView,
    ) -> Result<Option<Arc<LegacyFullRoleRecords>>, PalsError> {
        if !self.engine.semantic.legacy_full_native_context {
            return Ok(None);
        }
        let root_digest =
            rz_position::contracts::rules_snapshot_semantic_digest(&view.anchors()[0].snapshot).0;
        if context.public_revision != self.engine.semantic.revision
            || context.search_generation != self.engine.semantic.stores.generation()
            || view.generation() != context.search_generation
            || view.anchor_mode() != crate::pals::review_context::ReviewAnchorMode::CurrentRootOnly
            || !view.validates(&self.engine.semantic.stores)
        {
            return Err(RoleError::InvalidOutput.into());
        }
        if let Some(snapshot) = self
            .last_legacy_full_snapshot
            .as_ref()
            .and_then(std::sync::Weak::upgrade)
            .filter(|snapshot| {
                snapshot.authority() == self.authority
                    && snapshot.revision() == context.public_revision
                    && snapshot.generation() == context.search_generation
                    && snapshot.root_state() == root_digest
                    && snapshot.validates(&self.engine.semantic.stores)
            })
        {
            return Ok(Some(snapshot));
        }
        let snapshot = Arc::new(LegacyFullRoleRecords::capture(
            &self.engine,
            self.authority,
        )?);
        if snapshot.root_state() != root_digest {
            return Err(ReviewError::AnchorMismatch.into());
        }
        self.last_legacy_full_snapshot = Some(Arc::downgrade(&snapshot));
        Ok(Some(snapshot))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selected_default_never_captures_or_leases_legacy_full_rows() {
        let mut port = super::super::port_tests::setup(Position::startpos(), 2);
        let work = super::super::port_tests::work(
            &mut port,
            "e2e4",
            RefinementStage::ProposalContinuation,
            1,
        );
        assert!(work.input.question.legacy_full_records.is_none());
        assert!(port.last_legacy_full_snapshot.is_none());
        assert_eq!(port.engine.semantic.stores.review_pin_count().unwrap(), 1);
    }

    #[test]
    fn full_snapshot_keeps_more_than_selected_limit_and_freezes_original_rows() {
        let mut port = super::super::port_tests::setup(Position::startpos(), 2);
        port.engine.semantic.legacy_full_native_context = true;
        let state = port.engine.semantic.nodes[port.root].state;
        for revision in 1..=97 {
            port.engine.semantic.records.push(RoleRecord {
                revision,
                parent_revision: (revision > 1).then_some(revision - 1),
                supersedes_revision: None,
                origin_state: state,
                kind: RecordKind::Proposal,
                line: vec![BoardMove::from_uci("e2e4").unwrap()],
                value: Some(revision as i32),
                completed_depth: 1,
                score_scope: Some(CpuScoreScope::CompletedIteration),
                cpu_observation: None,
                perspective: Color::White,
                critical: true,
            });
        }
        port.engine.semantic.revision = 97;
        let work = super::super::port_tests::work(
            &mut port,
            "e2e4",
            RefinementStage::ProposalContinuation,
            1,
        );
        let captured = work.input.question.legacy_full_records.as_ref().unwrap();
        assert_eq!(captured.records().len(), 97);
        assert_eq!(captured.revision(), 97);
        assert!(work.input.view.records().len() <= 64);
        assert_eq!(captured.records()[65].value, Some(66));
        let digest = captured.digest();
        port.engine.semantic.records[65].value = Some(1234);
        port.engine.semantic.records[65].line.clear();
        assert_eq!(captured.records()[65].value, Some(66));
        assert_eq!(captured.records()[65].line.len(), 1);
        assert_eq!(captured.digest(), digest);
        assert!(captured.validates(&port.engine.semantic.stores));
    }

    #[test]
    fn unchanged_revision_shares_original_full_scope_and_last_drop_releases_lease() {
        let mut port = super::super::port_tests::setup(Position::startpos(), 2);
        port.engine.semantic.legacy_full_native_context = true;
        let first = super::super::port_tests::work(
            &mut port,
            "e2e4",
            RefinementStage::ProposalContinuation,
            1,
        );
        let second = super::super::port_tests::work(
            &mut port,
            "d2d4",
            RefinementStage::ProposalContinuation,
            2,
        );
        assert!(Arc::ptr_eq(
            first.input.question.legacy_full_records.as_ref().unwrap(),
            second.input.question.legacy_full_records.as_ref().unwrap(),
        ));
        assert_eq!(port.engine.semantic.stores.review_pin_count().unwrap(), 3);
        drop(first);
        assert_eq!(port.engine.semantic.stores.review_pin_count().unwrap(), 2);
        drop(second);
        assert_eq!(port.engine.semantic.stores.review_pin_count().unwrap(), 0);
        assert!(
            port.last_legacy_full_snapshot
                .as_ref()
                .unwrap()
                .upgrade()
                .is_none()
        );
    }

    #[test]
    fn raw_projection_overflow_and_new_root_generation_are_not_truncated_or_rebound() {
        let mut port = super::super::port_tests::setup(Position::startpos(), 2);
        let snapshot = LegacyFullRoleRecords::capture(&port.engine, port.authority).unwrap();
        let mut changed = Position::startpos();
        changed.make_uci("e2e4").unwrap();
        port.engine
            .semantic
            .stores
            .focus_actual_moves(changed.snapshot())
            .unwrap();
        assert!(!snapshot.validates(&port.engine.semantic.stores));
        let state = port.engine.semantic.nodes[port.root].state;
        port.engine.semantic.records = (0..=port.engine.config.max_records)
            .map(|index| RoleRecord {
                revision: index as u64,
                parent_revision: None,
                supersedes_revision: None,
                origin_state: state,
                kind: RecordKind::Proposal,
                line: Vec::new(),
                value: None,
                completed_depth: 0,
                score_scope: None,
                cpu_observation: None,
                perspective: Color::White,
                critical: false,
            })
            .collect();
        port.engine.semantic.revision = port.engine.config.max_records as u64;
        assert!(matches!(
            LegacyFullRoleRecords::capture(&port.engine, port.authority),
            Err(PalsError::Capacity)
        ));
    }
}
