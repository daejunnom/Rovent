//! Search-owned immutable, bounded review inputs. Raw RoleRecords and cold
//! archives remain evidence; selecting a view never deletes or rewrites them.
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Weak};

use rz_position::{
    BoardMove, Color, MAX_UCI_REPLAY_PLIES, Position, PositionLimits, PositionSnapshot,
};
use sha2::{Digest, Sha256};

use super::engine::{RecordKind, RoleQueryPurpose, RoleRecord};
use super::store::{
    EvidenceScope, Move16, ObservationId, ObservationKind, PalsStores, StateId, StoreError,
    StorePins, StoreReadOwner, StoreReadPin,
};

pub mod historical;

pub const REVIEW_CONTEXT_VERSION: &str = "rz-pals-selected-review/1";
const MAX_LINE: usize = 256;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReviewIndexLimits {
    pub active_bundles: usize,
    pub records_per_bundle: usize,
}
impl Default for ReviewIndexLimits {
    fn default() -> Self {
        Self {
            active_bundles: 256,
            records_per_bundle: 64,
        }
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReviewAnchorMode {
    CurrentRootOnly,
    AnchoredCompact,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReviewSelectionPolicy {
    pub mode: ReviewAnchorMode,
    pub token_budget: u16,
    pub max_records: u16,
    pub max_anchors: u8,
    /// A dependent Repair/review must retain this entire coherent bundle.
    pub required_bundle: Option<u64>,
}
impl Default for ReviewSelectionPolicy {
    fn default() -> Self {
        Self {
            mode: ReviewAnchorMode::CurrentRootOnly,
            token_budget: 512,
            max_records: 64,
            max_anchors: 1,
            required_bundle: None,
        }
    }
}
impl ReviewSelectionPolicy {
    pub fn validate(self) -> Result<(), ReviewError> {
        if !matches!(self.token_budget, 128 | 256 | 512)
            || self.max_records == 0
            || self.max_records > 64
            || self.max_anchors == 0
            || self.max_anchors > 4
            || (self.mode == ReviewAnchorMode::CurrentRootOnly && self.max_anchors != 1)
        {
            return Err(ReviewError::InvalidPolicy);
        }
        Ok(())
    }
    pub fn digest(self) -> [u8; 32] {
        let mut hash = Sha256::new();
        hash.update(b"rz-pals-context-policy/1\0");
        hash.update([self.mode as u8, self.max_anchors]);
        hash.update(self.token_budget.to_le_bytes());
        hash.update(self.max_records.to_le_bytes());
        // Required bundle is submission authority, not a model input feature.
        hash.finalize().into()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ReviewError {
    InvalidPolicy,
    OwnerMismatch,
    StaleGeneration,
    DuplicateRevision,
    MissingRelation(u64),
    RelationMismatch,
    InvalidLine,
    AnchorMismatch,
    ViewCapacity,
    /// This original (anchor, first move) bundle cannot retain another record.
    /// Active-bundle admission and sequence exhaustion remain IndexCapacity.
    BundleRecordCapacity,
    IndexCapacity,
    AlreadyConsumed,
    Store(StoreError),
}
impl From<StoreError> for ReviewError {
    fn from(value: StoreError) -> Self {
        Self::Store(value)
    }
}
impl std::fmt::Display for ReviewError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "PALS review: {self:?}")
    }
}
impl std::error::Error for ReviewError {}

#[derive(Clone, Copy)]
pub struct ReviewQuerySequences<'a> {
    pub purpose: RoleQueryPurpose,
    pub prefix: &'a [BoardMove],
    pub proposal: &'a [BoardMove],
    pub counterexample: Option<&'a [BoardMove]>,
    pub legal: &'a [BoardMove],
    pub divergences: &'a [usize],
}
#[derive(Clone, Debug)]
pub struct ReviewQueryInput {
    pub purpose: RoleQueryPurpose,
    pub prefix: Vec<BoardMove>,
    pub proposal: Vec<BoardMove>,
    pub counterexample: Option<Vec<BoardMove>>,
    pub legal: Vec<BoardMove>,
    pub divergences: Vec<usize>,
}
#[derive(Clone, Debug)]
pub struct ReviewAnchor {
    pub state: StateId,
    pub snapshot: PositionSnapshot,
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReviewRecordMetadata {
    pub anchor: u8,
    pub parent: Option<u16>,
    pub supersedes: Option<u16>,
    pub source_state: StateId,
    pub source_revision: u64,
    pub source_line_offset: u16,
    pub original_first_move: Option<BoardMove>,
}

/// The pin, source seal and atomic consumption authority are owned by the
/// immutable view. None of its selected input can be edited after submission.
#[derive(Debug)]
pub struct ReviewView {
    index_owner: Weak<()>,
    sequence: u64,
    root_state: StateId,
    generation: u64,
    policy: ReviewSelectionPolicy,
    records: Vec<RoleRecord>,
    metadata: Vec<ReviewRecordMetadata>,
    anchors: Vec<ReviewAnchor>,
    // 원 selected-record 순서의 source 이력이다. 모델 입력·seal의 추가 feature가
    // 아니며 native 수집과 NNUEC 준비가 같은 immutable backing을 공유한다.
    source_snapshots: Arc<[PositionSnapshot]>,
    query: ReviewQueryInput,
    bundles: Vec<u64>,
    tokens: u16,
    bucket: u16,
    digest: [u8; 32],
    seal: [u8; 32],
    pin: StoreReadPin,
    consumed: AtomicBool,
}
impl ReviewView {
    pub fn sequence(&self) -> u64 {
        self.sequence
    }
    pub fn root_state(&self) -> StateId {
        self.root_state
    }
    pub fn generation(&self) -> u64 {
        self.generation
    }
    pub fn records(&self) -> &[RoleRecord] {
        &self.records
    }
    pub fn records_metadata(&self) -> &[ReviewRecordMetadata] {
        &self.metadata
    }
    pub fn anchors(&self) -> &[ReviewAnchor] {
        &self.anchors
    }
    /// 원 metadata와 같은 순서·개수의 source snapshot이다. 각 snapshot의 깊은
    /// history는 Rules의 immutable Arc를 공유하며 Store 조회나 FEN 복원을 하지 않는다.
    /// 반환값은 역사 입력 자료로서 현재 source·pin·실행·소비 권한을 발급하지 않는다.
    pub fn source_snapshots(&self) -> &Arc<[PositionSnapshot]> {
        &self.source_snapshots
    }
    pub fn query(&self) -> &ReviewQueryInput {
        &self.query
    }
    /// Historical selected-input metadata only. This cannot select a bundle,
    /// reconstruct a view pin or authorize another head consumption.
    pub(crate) fn selected_bundle_ids(&self) -> &[u64] {
        &self.bundles
    }
    /// Historical observation only; it does not replace the actual index's
    /// original view/Store checks or authorize another consumption.
    pub(crate) fn consumption_observed(&self) -> bool {
        self.consumed.load(Ordering::Acquire)
    }
    pub(crate) fn retirement_protected_pins(
        &self,
        stores: &PalsStores,
    ) -> Result<StorePins, ReviewError> {
        Ok(self.pin.retirement_seed(stores)?)
    }
    pub fn token_count(&self) -> u16 {
        self.tokens
    }
    pub fn token_budget(&self) -> u16 {
        self.policy.token_budget
    }
    pub fn anchor_mode(&self) -> ReviewAnchorMode {
        self.policy.mode
    }
    /// 원 선택 정책의 읽기 전용 메타데이터다. 이 값은 source·Store 소유권이나
    /// required bundle의 현재 제출 권한을 발급하지 않는다.
    pub fn selection_policy(&self) -> ReviewSelectionPolicy {
        self.policy
    }
    pub fn length_bucket(&self) -> u16 {
        self.bucket
    }
    pub fn digest(&self) -> [u8; 32] {
        self.digest
    }
    pub fn seal_digest(&self) -> [u8; 32] {
        self.seal
    }
    pub fn policy_digest(&self) -> [u8; 32] {
        self.policy.digest()
    }
    pub fn validates(&self, stores: &PalsStores) -> bool {
        self.pin.validates(stores)
    }
    /// Known owned host backing, not a measured peak. Shared Rules history
    /// chains are conservatively charged once per retained anchor/source;
    /// physical sharing is deliberately not guessed from Arc identities.
    pub fn owned_host_bytes(&self) -> Option<u64> {
        let mut bytes = std::mem::size_of::<Self>();
        for backing in [
            self.records
                .capacity()
                .checked_mul(std::mem::size_of::<RoleRecord>())?,
            self.metadata
                .capacity()
                .checked_mul(std::mem::size_of::<ReviewRecordMetadata>())?,
            self.anchors
                .capacity()
                .checked_mul(std::mem::size_of::<ReviewAnchor>())?,
            self.source_snapshots
                .len()
                .checked_mul(std::mem::size_of::<PositionSnapshot>())?,
            self.bundles
                .capacity()
                .checked_mul(std::mem::size_of::<u64>())?,
            self.query
                .prefix
                .capacity()
                .checked_mul(std::mem::size_of::<BoardMove>())?,
            self.query
                .proposal
                .capacity()
                .checked_mul(std::mem::size_of::<BoardMove>())?,
            self.query
                .legal
                .capacity()
                .checked_mul(std::mem::size_of::<BoardMove>())?,
            self.query
                .divergences
                .capacity()
                .checked_mul(std::mem::size_of::<usize>())?,
        ] {
            bytes = bytes.checked_add(backing)?;
        }
        if let Some(counter) = &self.query.counterexample {
            bytes = bytes.checked_add(
                counter
                    .capacity()
                    .checked_mul(std::mem::size_of::<BoardMove>())?,
            )?;
        }
        for record in &self.records {
            bytes = bytes.checked_add(
                record
                    .line
                    .capacity()
                    .checked_mul(std::mem::size_of::<BoardMove>())?,
            )?;
        }
        for anchor in &self.anchors {
            bytes = bytes.checked_add(anchor.snapshot.retained_history_bytes()?)?;
        }
        for source in self.source_snapshots.iter() {
            bytes = bytes.checked_add(source.retained_history_bytes()?)?;
        }
        u64::try_from(bytes).ok()
    }
    /// Fresh endpoint queries differ; their immutable public record/anchor
    /// scope must still be identical. No private endpoint/query is erased from
    /// the actual full input digest exposed by digest().
    pub fn record_scope_digest(&self) -> [u8; 32] {
        record_scope_digest(&self.anchors, &self.records, &self.metadata, self.policy)
            .expect("sealed validated immutable record scope")
    }
}

#[derive(Clone, Debug)]
pub struct ReviewBundle {
    pub id: u64,
    pub anchor: StateId,
    pub original_first_move: Option<BoardMove>,
    revisions: Vec<u64>,
    protected_consumptions: u8,
    consumed_inputs: Vec<[u8; 32]>,
}
impl ReviewBundle {
    pub fn revisions(&self) -> &[u64] {
        &self.revisions
    }
    pub fn protected_consumptions(&self) -> u8 {
        self.protected_consumptions
    }
}
#[derive(Clone, Debug)]
struct IndexedRecord {
    record: RoleRecord,
    evidence: Option<ObservationId>,
    bundle: u64,
}

/// The single Search writer updates only the changed record/bundle. Selection
/// traverses at most the registered active bundle index, never the raw archive.
#[derive(Debug)]
pub struct ReviewContextIndex {
    identity: Arc<()>,
    limits: ReviewIndexLimits,
    owner: Option<StoreReadOwner>,
    next_bundle: u64,
    next_view: u64,
    bundles: BTreeMap<u64, ReviewBundle>,
    records: BTreeMap<u64, IndexedRecord>,
    groups: BTreeMap<(StateId, Option<u16>), u64>,
}
impl Default for ReviewContextIndex {
    fn default() -> Self {
        Self::new(ReviewIndexLimits::default()).expect("registered review limits")
    }
}
impl ReviewContextIndex {
    pub fn new(limits: ReviewIndexLimits) -> Result<Self, ReviewError> {
        if limits.active_bundles == 0
            || limits.active_bundles > 256
            || limits.records_per_bundle == 0
            || limits.records_per_bundle > 64
        {
            return Err(ReviewError::InvalidPolicy);
        }
        Ok(Self {
            identity: Arc::new(()),
            limits,
            owner: None,
            next_bundle: 0,
            // Zero is reserved by ReviewViewSeal for an unissued view.
            next_view: 1,
            bundles: BTreeMap::new(),
            records: BTreeMap::new(),
            groups: BTreeMap::new(),
        })
    }
    pub fn active_bundle_count(&self) -> usize {
        self.bundles.len()
    }
    pub fn limits(&self) -> ReviewIndexLimits {
        self.limits
    }
    pub fn indexed_record_count(&self) -> usize {
        self.records.len()
    }
    pub fn contains_revision(&self, revision: u64) -> bool {
        self.records.contains_key(&revision)
    }
    pub fn bundle_for_revision(&self, revision: u64) -> Option<u64> {
        self.records.get(&revision).map(|entry| entry.bundle)
    }
    pub fn bundle_for_first_move(
        &self,
        anchor: StateId,
        movement: Option<BoardMove>,
    ) -> Result<Option<u64>, ReviewError> {
        let first = movement
            .map(Move16::pack)
            .transpose()?
            .map(|movement| movement.bits());
        Ok(self.groups.get(&(anchor, first)).copied())
    }
    pub fn bundle(&self, id: u64) -> Option<&ReviewBundle> {
        self.bundles.get(&id)
    }
    /// One root transition updates the bounded index's legacy critical flags.
    /// It does not rescan the raw RoleRecord store for each native question.
    pub fn refresh_root(&mut self, root: StateId) {
        let mut newest = std::collections::HashSet::new();
        for entry in self.records.values_mut().rev() {
            entry.record.critical = entry.record.origin_state == root
                && newest.insert((entry.record.kind, entry.record.line.first().copied()));
        }
    }
    fn check_owner(&self, stores: &PalsStores) -> Result<(), ReviewError> {
        if self
            .owner
            .as_ref()
            .is_some_and(|owner| !owner.matches(stores))
        {
            return Err(ReviewError::OwnerMismatch);
        }
        Ok(())
    }
    /// Preserve the existing bounded projection during idle Store maintenance.
    /// This is not a submitted view, another lease, or consumption authority.
    /// Keeping the source facts hot prevents optional cold-anchor skips from
    /// changing an otherwise identical question after metadata reclamation.
    pub(super) fn archive_projection_pins(
        &self,
        stores: &PalsStores,
    ) -> Result<StorePins, ReviewError> {
        self.check_owner(stores)?;
        if self.bundles.len() > self.limits.active_bundles
            || self.records.len() > self.limits.active_bundles * self.limits.records_per_bundle
        {
            return Err(ReviewError::IndexCapacity);
        }
        let mut pins = StorePins::default();
        for (revision, entry) in &self.records {
            let bundle = self
                .bundles
                .get(&entry.bundle)
                .ok_or(ReviewError::RelationMismatch)?;
            if entry.record.revision != *revision
                || bundle.anchor != entry.record.origin_state
                || bundle.revisions.len() > self.limits.records_per_bundle
                || !bundle.revisions.contains(revision)
            {
                return Err(ReviewError::RelationMismatch);
            }
            for relation in [
                entry.record.parent_revision,
                entry.record.supersedes_revision,
            ]
            .into_iter()
            .flatten()
            {
                if self
                    .records
                    .get(&relation)
                    .is_none_or(|parent| parent.bundle != entry.bundle)
                    || relation >= *revision
                {
                    return Err(ReviewError::MissingRelation(relation));
                }
            }
            stores.states.get(entry.record.origin_state)?;
            pins.states.insert(entry.record.origin_state);
            for observation in [entry.evidence, entry.record.cpu_observation]
                .into_iter()
                .flatten()
            {
                stores.observations.get(observation)?;
                pins.observations.insert(observation);
            }
        }
        Ok(pins)
    }
    /// Publication requires actual store ownership and a complete Rules line.
    /// Parent/supersedes records must have been published earlier in the same
    /// coherent original-first-move bundle; no guessed relationship is issued.
    pub fn publish(
        &mut self,
        stores: &PalsStores,
        record: &RoleRecord,
        evidence: Option<ObservationId>,
    ) -> Result<u64, ReviewError> {
        self.publish_impl(stores, record, evidence, true)
    }
    /// Restoring an already observed archive record is not a new witness or an
    /// eligible consumption opportunity. Preserve existing protection while
    /// rebuilding its immutable index projection and relationship closure.
    pub fn restore_record(
        &mut self,
        stores: &PalsStores,
        record: &RoleRecord,
        evidence: Option<ObservationId>,
    ) -> Result<u64, ReviewError> {
        self.check_owner(stores)?;
        if let Some(existing) = self.records.get(&record.revision) {
            if existing.record.origin_state != record.origin_state
                || existing.record.kind != record.kind
                || existing.record.line != record.line
                || existing.record.parent_revision != record.parent_revision
                || existing.record.supersedes_revision != record.supersedes_revision
                || existing.record.perspective != record.perspective
                || existing.record.value != record.value
                || existing.record.completed_depth != record.completed_depth
                || existing.record.score_scope != record.score_scope
                || existing.record.cpu_observation != record.cpu_observation
            {
                return Err(ReviewError::RelationMismatch);
            }
            return Ok(existing.bundle);
        }
        self.publish_impl(stores, record, evidence, false)
    }
    fn publish_impl(
        &mut self,
        stores: &PalsStores,
        record: &RoleRecord,
        evidence: Option<ObservationId>,
        newly_observed: bool,
    ) -> Result<u64, ReviewError> {
        self.check_owner(stores)?;
        if self.records.contains_key(&record.revision) {
            return Err(ReviewError::DuplicateRevision);
        }
        if record.line.len() > MAX_LINE {
            return Err(ReviewError::InvalidLine);
        }
        let anchor = stores.states.get(record.origin_state)?;
        checked_replay(anchor, &record.line)?;
        if let Some(id) = evidence {
            let observation = stores.observations.get(id)?;
            let kind = match record.kind {
                RecordKind::Proposal => ObservationKind::Proposal,
                RecordKind::Counterexample => ObservationKind::Refutation,
                RecordKind::Repair => ObservationKind::Repair,
                RecordKind::CpuVerification => return Err(ReviewError::RelationMismatch),
            };
            if observation.state != record.origin_state
                || observation.kind != kind
                || !matches!(observation.scope, EvidenceScope::Model { .. })
                || observation
                    .line
                    .is_none_or(|line| match stores.lines.moves(line) {
                        Ok(moves) => moves != record.line,
                        Err(_) => true,
                    })
            {
                return Err(ReviewError::RelationMismatch);
            }
        }
        if let Some(id) = record.cpu_observation {
            let observation = stores.observations.get(id)?;
            if !matches!(
                observation.scope,
                EvidenceScope::DepthLimited { .. }
                    | EvidenceScope::ProfiledExternalUci { .. }
                    | EvidenceScope::ExternalUci { .. }
                    | EvidenceScope::RulesTerminal
            ) || !matches!(
                observation.kind,
                ObservationKind::CpuAnalysis
                    | ObservationKind::ExternalCpuAnalysis
                    | ObservationKind::ExactTerminal
            ) {
                return Err(ReviewError::RelationMismatch);
            }
            let observed_anchor = stores.states.get(observation.state)?;
            let mut position = Position::from_snapshot(anchor, PositionLimits::default())
                .map_err(|_| ReviewError::InvalidLine)?;
            let mut connected = position.snapshot().same_state(observed_anchor);
            for movement in &record.line {
                position
                    .make_move(*movement)
                    .map_err(|_| ReviewError::InvalidLine)?;
                connected |= position.snapshot().same_state(observed_anchor);
            }
            if !connected {
                return Err(ReviewError::AnchorMismatch);
            }
        }
        let first = record.line.first().copied();
        let key = (
            record.origin_state,
            first
                .map(Move16::pack)
                .transpose()?
                .map(|movement| movement.bits()),
        );
        let existing = self.groups.get(&key).copied();
        for relation in [record.parent_revision, record.supersedes_revision]
            .into_iter()
            .flatten()
        {
            let parent = self
                .records
                .get(&relation)
                .ok_or(ReviewError::MissingRelation(relation))?;
            if Some(parent.bundle) != existing || parent.record.revision >= record.revision {
                return Err(ReviewError::RelationMismatch);
            }
        }
        if let Some(id) = existing {
            if self.bundles[&id].revisions.len() >= self.limits.records_per_bundle {
                return Err(ReviewError::BundleRecordCapacity);
            }
        } else {
            self.compact_internal(None);
            if self.bundles.len() >= self.limits.active_bundles {
                return Err(ReviewError::IndexCapacity);
            }
        }
        let id = match existing {
            Some(id) => id,
            None => {
                let id = self.next_bundle;
                self.next_bundle = id.checked_add(1).ok_or(ReviewError::IndexCapacity)?;
                self.groups.insert(key, id);
                self.bundles.insert(
                    id,
                    ReviewBundle {
                        id,
                        anchor: record.origin_state,
                        original_first_move: first,
                        revisions: Vec::new(),
                        protected_consumptions: 0,
                        consumed_inputs: Vec::new(),
                    },
                );
                id
            }
        };
        for revision in &self.bundles[&id].revisions {
            let previous = self
                .records
                .get_mut(revision)
                .expect("indexed bundle relation");
            if previous.record.kind == record.kind {
                previous.record.critical = false;
            }
        }
        let bundle = self.bundles.get_mut(&id).expect("published bundle");
        bundle.revisions.push(record.revision);
        if newly_observed && matches!(record.kind, RecordKind::Counterexample | RecordKind::Repair)
        {
            bundle.protected_consumptions = 2;
            bundle.consumed_inputs.clear();
        }
        self.records.insert(
            record.revision,
            IndexedRecord {
                record: record.clone(),
                evidence,
                bundle: id,
            },
        );
        if self.owner.is_none() {
            self.owner = Some(stores.read_owner());
        }
        Ok(id)
    }
    /// High water 80%, target 55%, remove whole unprotected bundles. This only
    /// discards index projections: original RoleRecord/archive owners persist.
    pub fn compact(&mut self) -> usize {
        self.compact_internal(None)
    }
    fn compact_internal(&mut self, exclude: Option<u64>) -> usize {
        let high = (self.limits.active_bundles * 80).div_ceil(100);
        if self.bundles.len() < high.max(1) {
            return 0;
        }
        let target = self.limits.active_bundles * 55 / 100;
        let eligible: Vec<_> = self
            .bundles
            .iter()
            .filter_map(|(id, bundle)| {
                (Some(*id) != exclude && bundle.protected_consumptions == 0).then_some(*id)
            })
            .collect();
        let mut removed = 0;
        for id in eligible {
            if self.bundles.len() <= target {
                break;
            }
            let bundle = self.bundles.remove(&id).expect("selected live bundle");
            for revision in bundle.revisions {
                self.records.remove(&revision);
            }
            self.groups.remove(&(
                bundle.anchor,
                bundle
                    .original_first_move
                    .map(|mv| Move16::pack(mv).expect("checked move").bits()),
            ));
            removed += 1;
        }
        removed
    }

    pub fn select(
        &mut self,
        stores: &PalsStores,
        root: StateId,
        query: ReviewQuerySequences<'_>,
        policy: ReviewSelectionPolicy,
    ) -> Result<ReviewView, ReviewError> {
        self.check_owner(stores)?;
        policy.validate()?;
        if self.owner.is_none() {
            self.owner = Some(stores.read_owner());
        }
        let root_snapshot = stores.states.get(root)?;
        let query_lines = [
            query.prefix,
            query.proposal,
            query.counterexample.unwrap_or(&[]),
        ];
        let query_tokens: usize = query_lines.iter().map(|line| line.len()).sum();
        if query_lines.iter().any(|line| line.len() > MAX_LINE)
            || query_tokens > usize::from(policy.token_budget)
        {
            return Err(ReviewError::ViewCapacity);
        }
        for line in query_lines {
            checked_replay(root_snapshot, line)?;
        }
        if query.legal.len() > 256 || query.divergences.len() > 256 {
            return Err(ReviewError::ViewCapacity);
        }
        let query_position = checked_replay(root_snapshot, query.prefix)?;
        let mut unique = std::collections::BTreeSet::new();
        let legal = Position::from_snapshot(&query_position, PositionLimits::default())
            .map_err(|_| ReviewError::InvalidLine)?
            .legal_moves();
        for movement in query.legal {
            if !legal.contains(movement) || !unique.insert(Move16::pack(*movement)?.bits()) {
                return Err(ReviewError::InvalidLine);
            }
        }
        let mut divergence_unique = std::collections::BTreeSet::new();
        if query
            .divergences
            .iter()
            .any(|ply| *ply >= query.proposal.len() || !divergence_unique.insert(*ply))
        {
            return Err(ReviewError::InvalidLine);
        }
        if policy
            .required_bundle
            .is_some_and(|id| !self.bundles.contains_key(&id))
        {
            return Err(ReviewError::MissingRelation(
                policy.required_bundle.expect("checked"),
            ));
        }
        // Required first, then protected newest, then other newest. The largest
        // ordering scratch is bounded by 256 and is not a manager-tick snapshot.
        let mut ordered = Vec::with_capacity(self.bundles.len());
        ordered.extend(policy.required_bundle);
        for protected in [true, false] {
            ordered.extend(self.bundles.iter().rev().filter_map(|(id, bundle)| {
                (Some(*id) != policy.required_bundle
                    && (bundle.protected_consumptions != 0) == protected)
                    .then_some(*id)
            }));
        }
        let mut records = Vec::new();
        let mut metadata = Vec::new();
        let mut source_snapshots = Vec::new();
        let mut selected_bundles = Vec::new();
        let mut anchors = vec![ReviewAnchor {
            state: root,
            snapshot: root_snapshot.clone(),
        }];
        let mut pins = StorePins::default();
        pins.states.insert(root);
        let mut tokens = query_tokens;
        for id in ordered {
            let bundle = &self.bundles[&id];
            let required = Some(id) == policy.required_bundle;
            let anchor = match stores.states.get(bundle.anchor) {
                Ok(anchor) => anchor,
                Err(error) => {
                    if required {
                        return Err(error.into());
                    }
                    continue;
                }
            };
            let mut offsets = Vec::with_capacity(bundle.revisions.len());
            let mut all_rebased = true;
            for revision in &bundle.revisions {
                let line = &self.records[revision].record.line;
                match checked_rebase_offset(anchor, root_snapshot, line)? {
                    Some(offset) => offsets.push(offset),
                    None => {
                        all_rebased = false;
                        offsets.push(0);
                    }
                }
            }
            if !all_rebased && policy.mode == ReviewAnchorMode::CurrentRootOnly {
                if required {
                    return Err(ReviewError::AnchorMismatch);
                }
                continue;
            }
            let anchor_index = if all_rebased {
                0
            } else {
                match anchors
                    .iter()
                    .position(|entry| entry.snapshot.same_state(anchor))
                {
                    Some(index) => index,
                    None if anchors.len() < usize::from(policy.max_anchors) => anchors.len(),
                    None => {
                        if required {
                            return Err(ReviewError::ViewCapacity);
                        }
                        continue;
                    }
                }
            };
            if !all_rebased {
                offsets.fill(0);
            }
            let additional: usize = bundle
                .revisions
                .iter()
                .zip(&offsets)
                .map(|(rev, offset)| self.records[rev].record.line.len() - offset)
                .sum();
            if records.len() + bundle.revisions.len() > usize::from(policy.max_records)
                || tokens + additional > usize::from(policy.token_budget)
            {
                if required {
                    return Err(ReviewError::ViewCapacity);
                }
                continue;
            }
            if anchor_index == anchors.len() {
                anchors.push(ReviewAnchor {
                    state: bundle.anchor,
                    snapshot: anchor.clone(),
                });
            }
            let local: BTreeMap<_, _> = bundle
                .revisions
                .iter()
                .enumerate()
                .map(|(index, revision)| (*revision, (records.len() + index) as u16))
                .collect();
            for (revision, offset) in bundle.revisions.iter().zip(offsets) {
                let entry = &self.records[revision];
                let mut selected = entry.record.clone();
                selected.line = selected.line[offset..].to_vec();
                selected.origin_state = anchors[anchor_index].state;
                metadata.push(ReviewRecordMetadata {
                    anchor: anchor_index as u8,
                    parent: selected.parent_revision.map(|rev| local[&rev]),
                    supersedes: selected.supersedes_revision.map(|rev| local[&rev]),
                    source_state: entry.record.origin_state,
                    source_revision: *revision,
                    source_line_offset: offset as u16,
                    original_first_move: bundle.original_first_move,
                });
                source_snapshots.push(anchor.clone());
                pins.states.insert(entry.record.origin_state);
                pins.observations.extend(entry.evidence);
                pins.observations.extend(entry.record.cpu_observation);
                records.push(selected);
            }
            selected_bundles.push(id);
            tokens += additional;
        }
        let query = ReviewQueryInput {
            purpose: query.purpose,
            prefix: query.prefix.to_vec(),
            proposal: query.proposal.to_vec(),
            counterexample: query.counterexample.map(<[BoardMove]>::to_vec),
            legal: query.legal.to_vec(),
            divergences: query.divergences.to_vec(),
        };
        let longest = records
            .iter()
            .map(|record| record.line.len())
            .chain([
                query.prefix.len(),
                query.proposal.len(),
                query.counterexample.as_ref().map_or(0, Vec::len),
            ])
            .max()
            .unwrap_or(0);
        let bucket = [8, 16, 32, 64, 128, 256]
            .into_iter()
            .find(|bucket| longest <= *bucket)
            .ok_or(ReviewError::ViewCapacity)? as u16;
        let sequence = self.next_view;
        self.next_view = sequence.checked_add(1).ok_or(ReviewError::IndexCapacity)?;
        let digest = input_digest(
            &anchors,
            &records,
            &metadata,
            &query,
            policy,
            tokens as u16,
            bucket,
        )?;
        let seal = view_seal_digest(digest, sequence, stores.generation(), root, &metadata);
        let pin = stores.pin_review(pins)?;
        Ok(ReviewView {
            index_owner: Arc::downgrade(&self.identity),
            sequence,
            root_state: root,
            generation: stores.generation(),
            policy,
            records,
            metadata,
            anchors,
            source_snapshots: source_snapshots.into(),
            query,
            bundles: selected_bundles,
            tokens: tokens as u16,
            bucket,
            digest,
            seal,
            pin,
            consumed: AtomicBool::new(false),
        })
    }

    /// Call only after actual eligible search acceptance. Encoding, submission,
    /// completion and rejected/stale replies do not consume protected chances.
    pub fn consume(&mut self, view: &ReviewView, stores: &PalsStores) -> Result<(), ReviewError> {
        self.check_owner(stores)?;
        if !view.index_owner.ptr_eq(&Arc::downgrade(&self.identity)) {
            return Err(ReviewError::OwnerMismatch);
        }
        if !view.validates(stores) {
            return Err(ReviewError::StaleGeneration);
        }
        if view.consumed.swap(true, Ordering::AcqRel) {
            return Err(ReviewError::AlreadyConsumed);
        }
        for id in &view.bundles {
            if let Some(bundle) = self.bundles.get_mut(id) {
                // New counter/repair published after sealing retains two new
                // chances; an older accepted view cannot consume those chances.
                if bundle.protected_consumptions != 0
                    && !bundle.consumed_inputs.contains(&view.digest)
                    && bundle.revisions.iter().all(|revision| {
                        view.metadata
                            .iter()
                            .any(|record| record.source_revision == *revision)
                    })
                {
                    // Two protected chances imply a maximum of two retained
                    // digests. A new ticket for the same input is a replay,
                    // not a second distinct observation opportunity.
                    bundle.consumed_inputs.push(view.digest);
                    bundle.protected_consumptions = bundle.protected_consumptions.saturating_sub(1);
                }
            }
        }
        Ok(())
    }

    /// Select once for the larger actually-read query token cost, then seal
    /// both endpoints over exactly those immutable records and anchors.
    pub fn select_frozen_pair(
        &mut self,
        stores: &PalsStores,
        root: StateId,
        first: ReviewQuerySequences<'_>,
        second: ReviewQuerySequences<'_>,
        policy: ReviewSelectionPolicy,
    ) -> Result<(Arc<ReviewView>, Arc<ReviewView>), ReviewError> {
        let cost = |query: ReviewQuerySequences<'_>| {
            query
                .prefix
                .len()
                .saturating_add(query.proposal.len())
                .saturating_add(query.counterexample.unwrap_or(&[]).len())
        };
        if cost(first) >= cost(second) {
            let first = Arc::new(self.select(stores, root, first, policy)?);
            let second = Arc::new(self.fork_query(stores, &first, second)?);
            Ok((first, second))
        } else {
            let second = Arc::new(self.select(stores, root, second, policy)?);
            let first = Arc::new(self.fork_query(stores, &second, first)?);
            Ok((first, second))
        }
    }
    pub(in crate::pals) fn fork_query(
        &mut self,
        stores: &PalsStores,
        source: &ReviewView,
        query: ReviewQuerySequences<'_>,
    ) -> Result<ReviewView, ReviewError> {
        self.check_owner(stores)?;
        if !source.index_owner.ptr_eq(&Arc::downgrade(&self.identity)) {
            return Err(ReviewError::OwnerMismatch);
        }
        if !source.validates(stores) {
            return Err(ReviewError::StaleGeneration);
        }
        let root = &source.anchors[0].snapshot;
        let lines = [
            query.prefix,
            query.proposal,
            query.counterexample.unwrap_or(&[]),
        ];
        if lines.iter().any(|line| line.len() > MAX_LINE)
            || query.legal.len() > 256
            || query.divergences.len() > 256
        {
            return Err(ReviewError::ViewCapacity);
        }
        for line in lines {
            checked_replay(root, line)?;
        }
        let query_position = checked_replay(root, query.prefix)?;
        let legal = Position::from_snapshot(&query_position, PositionLimits::default())
            .map_err(|_| ReviewError::InvalidLine)?
            .legal_moves();
        let mut seen = std::collections::BTreeSet::new();
        for movement in query.legal {
            if !legal.contains(movement) || !seen.insert(Move16::pack(*movement)?.bits()) {
                return Err(ReviewError::InvalidLine);
            }
        }
        let mut seen = std::collections::BTreeSet::new();
        if query
            .divergences
            .iter()
            .any(|ply| *ply >= query.proposal.len() || !seen.insert(*ply))
        {
            return Err(ReviewError::InvalidLine);
        }
        let tokens = source
            .records
            .iter()
            .map(|record| record.line.len())
            .sum::<usize>()
            + lines.iter().map(|line| line.len()).sum::<usize>();
        if tokens > usize::from(source.policy.token_budget) {
            return Err(ReviewError::ViewCapacity);
        }
        let query = ReviewQueryInput {
            purpose: query.purpose,
            prefix: query.prefix.to_vec(),
            proposal: query.proposal.to_vec(),
            counterexample: query.counterexample.map(<[BoardMove]>::to_vec),
            legal: query.legal.to_vec(),
            divergences: query.divergences.to_vec(),
        };
        let longest = source
            .records
            .iter()
            .map(|record| record.line.len())
            .chain([
                query.prefix.len(),
                query.proposal.len(),
                query.counterexample.as_ref().map_or(0, Vec::len),
            ])
            .max()
            .unwrap_or(0);
        let bucket = [8, 16, 32, 64, 128, 256]
            .into_iter()
            .find(|bucket| longest <= *bucket)
            .ok_or(ReviewError::ViewCapacity)? as u16;
        let sequence = self.next_view;
        self.next_view = sequence.checked_add(1).ok_or(ReviewError::IndexCapacity)?;
        let digest = input_digest(
            &source.anchors,
            &source.records,
            &source.metadata,
            &query,
            source.policy,
            tokens as u16,
            bucket,
        )?;
        let seal = view_seal_digest(
            digest,
            sequence,
            stores.generation(),
            source.root_state,
            &source.metadata,
        );
        Ok(ReviewView {
            index_owner: Arc::downgrade(&self.identity),
            sequence,
            root_state: source.root_state,
            generation: source.generation,
            policy: source.policy,
            records: source.records.clone(),
            metadata: source.metadata.clone(),
            anchors: source.anchors.clone(),
            source_snapshots: Arc::clone(&source.source_snapshots),
            query,
            bundles: source.bundles.clone(),
            tokens: tokens as u16,
            bucket,
            digest,
            seal,
            pin: source.pin.reborrow(stores)?,
            consumed: AtomicBool::new(false),
        })
    }
}

fn checked_replay(
    anchor: &PositionSnapshot,
    line: &[BoardMove],
) -> Result<PositionSnapshot, ReviewError> {
    let mut position = Position::from_snapshot(anchor, PositionLimits::default())
        .map_err(|_| ReviewError::InvalidLine)?;
    for movement in line {
        position
            .make_move(*movement)
            .map_err(|_| ReviewError::InvalidLine)?;
    }
    Ok(position.snapshot())
}
/// Only exact actual history matching can strip an already-played virtual
/// prefix. It is never copied into review tokens or used as a guessed anchor.
pub fn checked_rebase_offset(
    anchor: &PositionSnapshot,
    root: &PositionSnapshot,
    line: &[BoardMove],
) -> Result<Option<usize>, ReviewError> {
    if line.len() > MAX_LINE {
        return Err(ReviewError::InvalidLine);
    }
    let original_endpoint = checked_replay(anchor, line)?;
    if anchor.same_state(root) {
        return Ok(Some(0));
    }
    let origin = anchor
        .uci_replay(MAX_UCI_REPLAY_PLIES)
        .map_err(|_| ReviewError::AnchorMismatch)?;
    let actual = root
        .uci_replay(MAX_UCI_REPLAY_PLIES)
        .map_err(|_| ReviewError::AnchorMismatch)?;
    if origin.origin != actual.origin
        || origin.completeness != actual.completeness
        || origin.start_fen != actual.start_fen
        || !actual.moves.starts_with(&origin.moves)
    {
        return Ok(None);
    }
    let actual_suffix = &actual.moves[origin.moves.len()..];
    if !line.starts_with(actual_suffix) {
        return Ok(None);
    }
    let offset = actual_suffix.len();
    let replayed_root = checked_replay(anchor, actual_suffix)?;
    if !replayed_root.same_state(root) {
        return Err(ReviewError::AnchorMismatch);
    }
    if !checked_replay(root, &line[offset..])?.same_state(&original_endpoint) {
        return Err(ReviewError::AnchorMismatch);
    }
    Ok(Some(offset))
}
fn write_moves(hash: &mut Sha256, moves: &[BoardMove]) -> Result<(), ReviewError> {
    hash.update((moves.len() as u16).to_le_bytes());
    for movement in moves {
        hash.update(Move16::pack(*movement)?.bits().to_le_bytes());
    }
    Ok(())
}
pub(in crate::pals) fn snapshot_digest(
    snapshot: &PositionSnapshot,
) -> Result<[u8; 32], ReviewError> {
    let trace = snapshot
        .uci_replay(MAX_UCI_REPLAY_PLIES)
        .map_err(|_| ReviewError::AnchorMismatch)?;
    let mut hash = Sha256::new();
    hash.update(b"rz-pals-exact-review-anchor/1\0");
    hash.update([trace.origin as u8, trace.completeness as u8]);
    hash.update((trace.start_fen.len() as u64).to_le_bytes());
    hash.update(trace.start_fen.as_bytes());
    write_moves(&mut hash, &trace.moves)?;
    hash.update(snapshot.to_fen().as_bytes());
    Ok(hash.finalize().into())
}

/// 같은 입력의 별도 제출을 구별하는 기존 seal framing이다. 현재 owner나 pin은
/// 발급하지 않으며 live view와 historical metadata가 같은 계산을 사용한다.
fn view_seal_digest(
    input: [u8; 32],
    sequence: u64,
    generation: u64,
    root: StateId,
    metadata: &[ReviewRecordMetadata],
) -> [u8; 32] {
    let mut seal = Sha256::new();
    seal.update(b"rz-pals-review-view-seal/1\0");
    seal.update(input);
    seal.update(sequence.to_le_bytes());
    seal.update(generation.to_le_bytes());
    seal.update((root.0 as u64).to_le_bytes());
    for record in metadata {
        seal.update(record.source_revision.to_le_bytes());
        seal.update((record.source_state.0 as u64).to_le_bytes());
        seal.update(record.source_line_offset.to_le_bytes());
    }
    seal.finalize().into()
}

fn record_scope_digest(
    anchors: &[ReviewAnchor],
    records: &[RoleRecord],
    metadata: &[ReviewRecordMetadata],
    policy: ReviewSelectionPolicy,
) -> Result<[u8; 32], ReviewError> {
    let query = ReviewQueryInput {
        purpose: RoleQueryPurpose::ProposePolicy,
        prefix: Vec::new(),
        proposal: Vec::new(),
        counterexample: None,
        legal: Vec::new(),
        divergences: Vec::new(),
    };
    let input = input_digest(
        anchors,
        records,
        metadata,
        &query,
        policy,
        records.iter().map(|record| record.line.len() as u16).sum(),
        256,
    )?;
    let mut hash = Sha256::new();
    hash.update(b"rz-pals-frozen-public-review-scope/1\0");
    hash.update(input);
    Ok(hash.finalize().into())
}

fn input_digest(
    anchors: &[ReviewAnchor],
    records: &[RoleRecord],
    metadata: &[ReviewRecordMetadata],
    query: &ReviewQueryInput,
    policy: ReviewSelectionPolicy,
    tokens: u16,
    bucket: u16,
) -> Result<[u8; 32], ReviewError> {
    let mut hash = Sha256::new();
    hash.update(REVIEW_CONTEXT_VERSION.as_bytes());
    hash.update([0]);
    hash.update(policy.digest());
    hash.update(tokens.to_le_bytes());
    hash.update(bucket.to_le_bytes());
    hash.update([anchors.len() as u8]);
    for anchor in anchors {
        hash.update(snapshot_digest(&anchor.snapshot)?);
    }
    hash.update((records.len() as u16).to_le_bytes());
    for (record, metadata) in records.iter().zip(metadata) {
        hash.update([
            record.kind as u8,
            u8::from(record.perspective == Color::Black),
            metadata.anchor,
            u8::from(matches!(
                record.score_scope,
                Some(crate::cpu::CpuScoreScope::CompletedIteration)
            )),
            u8::from(matches!(
                record.score_scope,
                Some(crate::cpu::CpuScoreScope::RulesTerminal)
            )),
            u8::from(!record.line.is_empty()),
        ]);
        hash.update(metadata.parent.unwrap_or(u16::MAX).to_le_bytes());
        hash.update(metadata.supersedes.unwrap_or(u16::MAX).to_le_bytes());
        write_moves(&mut hash, &record.line)?;
        if policy.mode == ReviewAnchorMode::CurrentRootOnly {
            // The initial selected-view path still uses V2's old 16 features
            // and canonical record metadata. Compact V3 excludes CP/depth.
            hash.update(record.revision.to_le_bytes());
            hash.update((record.origin_state.0 as u64).to_le_bytes());
            hash.update([u8::from(record.critical), u8::from(record.value.is_some())]);
            hash.update(record.value.unwrap_or(0).to_le_bytes());
            hash.update(record.completed_depth.to_le_bytes());
            hash.update([match record.score_scope {
                None => 0,
                Some(crate::cpu::CpuScoreScope::FrontierOnly) => 1,
                Some(crate::cpu::CpuScoreScope::CompletedIteration) => 2,
                Some(crate::cpu::CpuScoreScope::RulesTerminal) => 3,
            }]);
        }
    }
    hash.update([
        query.purpose as u8,
        u8::from(query.counterexample.is_some()),
    ]);
    write_moves(&mut hash, &query.prefix)?;
    write_moves(&mut hash, &query.proposal)?;
    write_moves(&mut hash, query.counterexample.as_deref().unwrap_or(&[]))?;
    write_moves(&mut hash, &query.legal)?;
    hash.update((query.divergences.len() as u16).to_le_bytes());
    for ply in &query.divergences {
        hash.update((*ply as u16).to_le_bytes());
    }
    Ok(hash.finalize().into())
}

#[cfg(test)]
mod tests {
    use super::super::store::StoreLimits;
    use super::*;

    fn line(text: &str) -> Vec<BoardMove> {
        text.split_whitespace()
            .map(|item| BoardMove::from_uci(item).unwrap())
            .collect()
    }
    fn stores(position: &Position) -> (PalsStores, StateId) {
        let mut stores = PalsStores::new(StoreLimits::default());
        let situation = stores.focus_actual_moves(position.snapshot()).unwrap();
        let state = stores.situations.get(situation).unwrap().state;
        (stores, state)
    }
    fn record(
        state: StateId,
        revision: u64,
        kind: RecordKind,
        text: &str,
        parent: Option<u64>,
    ) -> RoleRecord {
        RoleRecord {
            revision,
            parent_revision: parent,
            supersedes_revision: None,
            origin_state: state,
            kind,
            line: line(text),
            value: None,
            completed_depth: 0,
            score_scope: None,
            cpu_observation: None,
            perspective: Color::White,
            critical: true,
        }
    }
    fn empty_query() -> ReviewQuerySequences<'static> {
        ReviewQuerySequences {
            purpose: RoleQueryPurpose::ProposePolicy,
            prefix: &[],
            proposal: &[],
            counterexample: None,
            legal: &[],
            divergences: &[],
        }
    }

    #[test]
    fn selected_bundle_keeps_parent_supersedes_and_query_duplicates_are_charged() {
        let (stores, state) = stores(&Position::startpos());
        let mut index = ReviewContextIndex::default();
        let p = record(state, 1, RecordKind::Proposal, "e2e4 e7e5 g1f3", None);
        let c = record(
            state,
            2,
            RecordKind::Counterexample,
            "e2e4 e7e5 g1f3",
            Some(1),
        );
        let mut r = record(state, 3, RecordKind::Repair, "e2e4 e7e5 b1c3", Some(2));
        r.supersedes_revision = Some(1);
        let bundle = index.publish(&stores, &p, None).unwrap();
        index.publish(&stores, &c, None).unwrap();
        index.publish(&stores, &r, None).unwrap();
        let same = line("e2e4 e7e5");
        let view = index
            .select(
                &stores,
                state,
                ReviewQuerySequences {
                    purpose: RoleQueryPurpose::RepairPolicy,
                    prefix: &same,
                    proposal: &same,
                    counterexample: Some(&same),
                    legal: &[],
                    divergences: &[],
                },
                ReviewSelectionPolicy {
                    required_bundle: Some(bundle),
                    ..ReviewSelectionPolicy::default()
                },
            )
            .unwrap();
        assert_eq!(view.records().len(), 3);
        assert_eq!(view.token_count(), 15);
        assert_eq!(view.records_metadata()[2].parent, Some(1));
        assert_eq!(view.records_metadata()[2].supersedes, Some(0));
        assert_eq!(view.length_bucket(), 8);
        assert_eq!(stores.review_pin_count().unwrap(), 1);
    }

    #[test]
    fn actual_history_is_rebase_proof_not_review_tokens() {
        let start = Position::startpos();
        let (mut stores, origin) = stores(&start);
        let mut index = ReviewContextIndex::default();
        index
            .publish(
                &stores,
                &record(origin, 1, RecordKind::Proposal, "e2e4 e7e5 g1f3", None),
                None,
            )
            .unwrap();
        let mut actual = start;
        actual.make_uci("e2e4").unwrap();
        actual.make_uci("e7e5").unwrap();
        let situation = stores.focus_actual_moves(actual.snapshot()).unwrap();
        let current = stores.situations.get(situation).unwrap().state;
        let view = index
            .select(
                &stores,
                current,
                empty_query(),
                ReviewSelectionPolicy::default(),
            )
            .unwrap();
        assert_eq!(view.records()[0].line, line("g1f3"));
        assert_eq!(view.token_count(), 1);
        assert_eq!(view.records_metadata()[0].source_line_offset, 2);
        assert_eq!(view.source_snapshots().len(), view.records_metadata().len());
        assert!(
            stores
                .states
                .get(origin)
                .unwrap()
                .weak_position_identity()
                .matches(&view.source_snapshots()[0])
        );
        assert!(!view.source_snapshots()[0].same_state(&actual.snapshot()));
        assert_eq!(
            view.records_metadata()[0].original_first_move,
            Some(BoardMove::from_uci("e2e4").unwrap())
        );
        assert_eq!(view.anchors().len(), 1);
        assert!(view.anchors()[0].snapshot.same_state(&actual.snapshot()));
    }

    #[test]
    fn fen_equality_different_history_cannot_rebase_but_compact_retains_exact_anchor() {
        let mut actual = Position::startpos();
        for movement in ["g1f3", "g8f6", "f3g1", "f6g8"] {
            actual.make_uci(movement).unwrap();
        }
        let imported = Position::from_fen(&actual.to_fen()).unwrap();
        assert_eq!(actual.to_fen(), imported.to_fen());
        assert!(!actual.snapshot().same_state(&imported.snapshot()));
        let (mut stores, origin) = stores(&actual);
        let current = stores.states.insert(imported.snapshot()).unwrap();
        let mut index = ReviewContextIndex::default();
        let bundle = index
            .publish(
                &stores,
                &record(origin, 1, RecordKind::Proposal, "e2e4", None),
                None,
            )
            .unwrap();
        assert!(matches!(
            index.select(
                &stores,
                current,
                empty_query(),
                ReviewSelectionPolicy {
                    required_bundle: Some(bundle),
                    ..ReviewSelectionPolicy::default()
                }
            ),
            Err(ReviewError::AnchorMismatch)
        ));
        let view = index
            .select(
                &stores,
                current,
                empty_query(),
                ReviewSelectionPolicy {
                    mode: ReviewAnchorMode::AnchoredCompact,
                    max_anchors: 4,
                    required_bundle: Some(bundle),
                    ..ReviewSelectionPolicy::default()
                },
            )
            .unwrap();
        assert_eq!(view.anchors().len(), 2);
        assert_eq!(view.records_metadata()[0].anchor, 1);
        assert!(view.anchors()[1].snapshot.same_state(&actual.snapshot()));
        assert_eq!(view.source_snapshots().len(), 1);
        assert!(
            actual
                .snapshot()
                .weak_position_identity()
                .matches(&view.source_snapshots()[0])
        );
    }

    #[test]
    fn independent_exact_history_owner_can_rebase_and_illegal_suffix_is_rejected() {
        let anchor = Position::startpos().snapshot();
        let mut independent = Position::startpos();
        independent.make_uci("e2e4").unwrap();
        assert_eq!(
            checked_rebase_offset(&anchor, &independent.snapshot(), &line("e2e4 e7e5")).unwrap(),
            Some(1)
        );
        assert_eq!(
            checked_rebase_offset(&anchor, &independent.snapshot(), &line("e2e4 e2e3")),
            Err(ReviewError::InvalidLine)
        );
        let mut other = Position::startpos();
        other.make_uci("d2d4").unwrap();
        assert_eq!(
            checked_rebase_offset(&anchor, &other.snapshot(), &line("e2e4 e7e5")).unwrap(),
            None
        );
    }

    #[test]
    fn frozen_queries_share_original_source_array_without_extra_source_pin_or_digest_features() {
        let (stores, state) = stores(&Position::startpos());
        let mut index = ReviewContextIndex::default();
        index
            .publish(
                &stores,
                &record(state, 1, RecordKind::Proposal, "e2e4 e7e5", None),
                None,
            )
            .unwrap();
        let first = index
            .select(
                &stores,
                state,
                empty_query(),
                ReviewSelectionPolicy::default(),
            )
            .unwrap();
        let second = index.fork_query(&stores, &first, empty_query()).unwrap();
        assert!(Arc::ptr_eq(
            first.source_snapshots(),
            second.source_snapshots()
        ));
        assert_eq!(first.digest(), second.digest());
        assert_eq!(first.record_scope_digest(), second.record_scope_digest());
        assert_ne!(first.seal_digest(), second.seal_digest());
        assert_eq!(stores.review_pin_count().unwrap(), 2);
        let retained = Arc::clone(first.source_snapshots());
        let history = retained[0].weak_position_identity();
        drop(first);
        drop(second);
        assert_eq!(stores.review_pin_count().unwrap(), 0);
        assert!(history.matches(&retained[0]));
        assert!(retained[0].same_state(stores.states.get(state).unwrap()));
    }

    #[test]
    fn empty_selected_view_retains_no_source_rows() {
        let (stores, state) = stores(&Position::startpos());
        let mut index = ReviewContextIndex::default();
        let view = index
            .select(
                &stores,
                state,
                empty_query(),
                ReviewSelectionPolicy::default(),
            )
            .unwrap();
        assert!(view.records_metadata().is_empty());
        assert!(view.source_snapshots().is_empty());
        assert_eq!(stores.review_pin_count().unwrap(), 1);
    }

    #[test]
    fn two_eligible_consumptions_protect_new_witness_and_duplicate_cannot_charge() {
        let (stores, state) = stores(&Position::startpos());
        let mut index = ReviewContextIndex::default();
        index
            .publish(
                &stores,
                &record(state, 1, RecordKind::Proposal, "e2e4", None),
                None,
            )
            .unwrap();
        let bundle = index
            .publish(
                &stores,
                &record(state, 2, RecordKind::Counterexample, "e2e4 e7e5", Some(1)),
                None,
            )
            .unwrap();
        let first = index
            .select(
                &stores,
                state,
                empty_query(),
                ReviewSelectionPolicy::default(),
            )
            .unwrap();
        assert_eq!(index.bundle(bundle).unwrap().protected_consumptions(), 2);
        index.consume(&first, &stores).unwrap();
        assert_eq!(index.bundle(bundle).unwrap().protected_consumptions(), 1);
        assert_eq!(
            index.consume(&first, &stores),
            Err(ReviewError::AlreadyConsumed)
        );
        let second = index
            .select(
                &stores,
                state,
                empty_query(),
                ReviewSelectionPolicy::default(),
            )
            .unwrap();
        index.consume(&second, &stores).unwrap();
        assert_eq!(index.bundle(bundle).unwrap().protected_consumptions(), 1);
        assert_eq!(first.digest(), second.digest());
        assert_ne!(first.seal_digest(), second.seal_digest());
        let distinct = index
            .select(
                &stores,
                state,
                ReviewQuerySequences {
                    purpose: RoleQueryPurpose::ReplyPolicy,
                    ..empty_query()
                },
                ReviewSelectionPolicy::default(),
            )
            .unwrap();
        index.consume(&distinct, &stores).unwrap();
        assert_eq!(index.bundle(bundle).unwrap().protected_consumptions(), 0);
    }

    #[test]
    fn late_old_view_cannot_consume_new_repairs_protection() {
        let (stores, state) = stores(&Position::startpos());
        let mut index = ReviewContextIndex::default();
        let bundle = index
            .publish(
                &stores,
                &record(state, 1, RecordKind::Proposal, "e2e4", None),
                None,
            )
            .unwrap();
        let old = index
            .select(
                &stores,
                state,
                empty_query(),
                ReviewSelectionPolicy::default(),
            )
            .unwrap();
        index
            .publish(
                &stores,
                &record(state, 2, RecordKind::Counterexample, "e2e4 e7e5", Some(1)),
                None,
            )
            .unwrap();
        index.consume(&old, &stores).unwrap();
        assert_eq!(index.bundle(bundle).unwrap().protected_consumptions(), 2);
    }

    #[test]
    fn lease_owners_generations_and_releases_are_independent() {
        let (mut stores, state) = stores(&Position::startpos());
        let mut index = ReviewContextIndex::default();
        let first = index
            .select(
                &stores,
                state,
                empty_query(),
                ReviewSelectionPolicy::default(),
            )
            .unwrap();
        let second = index
            .select(
                &stores,
                state,
                empty_query(),
                ReviewSelectionPolicy::default(),
            )
            .unwrap();
        assert_eq!(stores.review_pin_count().unwrap(), 2);
        drop(first);
        assert_eq!(stores.review_pin_count().unwrap(), 1);
        let (foreign, collision) = self::stores(&Position::startpos());
        assert_eq!(state, collision);
        assert!(!second.validates(&foreign));
        assert!(matches!(
            index.select(
                &foreign,
                collision,
                empty_query(),
                ReviewSelectionPolicy::default()
            ),
            Err(ReviewError::OwnerMismatch)
        ));
        let mut child = Position::startpos();
        child.make_uci("e2e4").unwrap();
        stores.focus_actual_moves(child.snapshot()).unwrap();
        assert!(!second.validates(&stores));
        assert_eq!(
            index.consume(&second, &stores),
            Err(ReviewError::StaleGeneration)
        );
        // Stale logical generation does not release the still-owned physical input.
        assert_eq!(stores.review_pin_count().unwrap(), 1);
        drop(second);
        assert_eq!(stores.review_pin_count().unwrap(), 0);
    }

    #[test]
    fn required_bundle_cannot_be_partially_cut_to_meet_budget() {
        let (stores, state) = stores(&Position::startpos());
        let mut index = ReviewContextIndex::default();
        let long = ["g1f3", "g8f6", "f3g1", "f6g8"]
            .into_iter()
            .cycle()
            .take(64)
            .collect::<Vec<_>>()
            .join(" ");
        index
            .publish(
                &stores,
                &record(state, 1, RecordKind::Proposal, &long, None),
                None,
            )
            .unwrap();
        index
            .publish(
                &stores,
                &record(state, 2, RecordKind::Counterexample, &long, Some(1)),
                None,
            )
            .unwrap();
        let bundle = index
            .publish(
                &stores,
                &record(state, 3, RecordKind::Repair, &long, Some(2)),
                None,
            )
            .unwrap();
        assert!(matches!(
            index.select(
                &stores,
                state,
                empty_query(),
                ReviewSelectionPolicy {
                    token_budget: 128,
                    required_bundle: Some(bundle),
                    ..ReviewSelectionPolicy::default()
                }
            ),
            Err(ReviewError::ViewCapacity)
        ));
        assert_eq!(stores.review_pin_count().unwrap(), 0);
        assert_eq!(index.indexed_record_count(), 3);
    }

    #[test]
    fn physical_line_bucket_tracks_selected_input_and_padding_is_not_charged() {
        let (stores, state) = stores(&Position::startpos());
        let mut index = ReviewContextIndex::default();
        let text = ["g1f3", "g8f6", "f3g1", "f6g8"]
            .into_iter()
            .cycle()
            .take(9)
            .collect::<Vec<_>>()
            .join(" ");
        index
            .publish(
                &stores,
                &record(state, 1, RecordKind::Proposal, &text, None),
                None,
            )
            .unwrap();
        let view = index
            .select(
                &stores,
                state,
                empty_query(),
                ReviewSelectionPolicy::default(),
            )
            .unwrap();
        assert_eq!(view.length_bucket(), 16);
        assert_eq!(view.token_count(), 9);
    }

    #[test]
    fn full_bundle_keeps_original_records_and_live_view_without_widening_index_errors() {
        let (stores, state) = stores(&Position::startpos());
        let mut index = ReviewContextIndex::new(ReviewIndexLimits {
            active_bundles: 2,
            records_per_bundle: 4,
        })
        .unwrap();
        let bundle = index
            .publish(
                &stores,
                &record(state, 1, RecordKind::Proposal, "e2e4", None),
                None,
            )
            .unwrap();
        for revision in 2..=4 {
            index
                .publish(
                    &stores,
                    &record(
                        state,
                        revision,
                        RecordKind::Counterexample,
                        "e2e4 e7e5",
                        Some(revision - 1),
                    ),
                    None,
                )
                .unwrap();
        }
        let view = index
            .select(
                &stores,
                state,
                empty_query(),
                ReviewSelectionPolicy::default(),
            )
            .unwrap();
        assert_eq!(view.records().len(), 4);
        assert_eq!(stores.review_pin_count().unwrap(), 1);
        assert_eq!(
            index.publish(
                &stores,
                &record(state, 5, RecordKind::Repair, "e2e4 e7e5 g1f3", Some(4)),
                None
            ),
            Err(ReviewError::BundleRecordCapacity),
        );
        assert_eq!(index.bundle(bundle).unwrap().revisions(), &[1, 2, 3, 4]);
        assert_eq!(index.indexed_record_count(), 4);
        assert!(!index.contains_revision(5));
        assert!(view.validates(&stores));
        let pins = index.archive_projection_pins(&stores).unwrap();
        assert!(pins.states.contains(&state));
        assert_eq!(index.bundle(bundle).unwrap().protected_consumptions(), 2);
        let (foreign, _) = self::stores(&Position::startpos());
        assert_eq!(
            index.archive_projection_pins(&foreign),
            Err(ReviewError::OwnerMismatch)
        );
        assert_eq!(stores.review_pin_count().unwrap(), 1);
        // A protected second bundle makes the separate active-bundle cap
        // unreclaimable. It must retain the original general error class.
        index
            .publish(
                &stores,
                &record(state, 6, RecordKind::Counterexample, "d2d4 d7d5", None),
                None,
            )
            .unwrap();
        assert_eq!(
            index.publish(
                &stores,
                &record(state, 7, RecordKind::Proposal, "c2c4", None),
                None
            ),
            Err(ReviewError::IndexCapacity),
        );
        assert_eq!(
            index.publish(
                &stores,
                &record(state, 8, RecordKind::Repair, "e2e4 e7e5 g1f3", Some(99)),
                None
            ),
            Err(ReviewError::MissingRelation(99)),
        );
        assert_eq!(index.bundle(bundle).unwrap().revisions(), &[1, 2, 3, 4]);
        assert_eq!(index.indexed_record_count(), 5);
        index.consume(&view, &stores).unwrap();
        assert_eq!(index.bundle(bundle).unwrap().protected_consumptions(), 1);
        drop(view);
        assert_eq!(stores.review_pin_count().unwrap(), 0);
    }

    #[test]
    fn pressure_removes_entire_unprotected_bundle_not_raw_evidence() {
        let (stores, state) = stores(&Position::startpos());
        let mut index = ReviewContextIndex::new(ReviewIndexLimits {
            active_bundles: 4,
            records_per_bundle: 64,
        })
        .unwrap();
        for (revision, movement) in [(1, "e2e4"), (2, "d2d4"), (3, "c2c4")] {
            index
                .publish(
                    &stores,
                    &record(state, revision, RecordKind::Proposal, movement, None),
                    None,
                )
                .unwrap();
        }
        let protected = index
            .publish(
                &stores,
                &record(state, 4, RecordKind::Counterexample, "e2e4 e7e5", Some(1)),
                None,
            )
            .unwrap();
        assert_eq!(index.active_bundle_count(), 3);
        assert_eq!(index.compact(), 0);
        index
            .publish(
                &stores,
                &record(state, 5, RecordKind::Proposal, "b2b4", None),
                None,
            )
            .unwrap();
        assert_eq!(index.compact(), 2);
        assert_eq!(index.active_bundle_count(), 2);
        assert_eq!(index.bundle(protected).unwrap().revisions(), &[1, 4]);
        assert!(index.bundle_for_revision(2).is_none());
        assert!(stores.states.get(state).is_ok());
        // A valid one-slot profile must be able to replace an inactive bundle;
        // rounding the 55% target upward to one would permanently saturate it.
        let mut tiny = ReviewContextIndex::new(ReviewIndexLimits {
            active_bundles: 1,
            records_per_bundle: 64,
        })
        .unwrap();
        tiny.publish(
            &stores,
            &record(state, 1, RecordKind::Proposal, "e2e4", None),
            None,
        )
        .unwrap();
        tiny.publish(
            &stores,
            &record(state, 2, RecordKind::Proposal, "d2d4", None),
            None,
        )
        .unwrap();
        assert!(!tiny.contains_revision(1));
        assert!(tiny.contains_revision(2));
        assert_eq!(tiny.active_bundle_count(), 1);
    }

    #[test]
    fn missing_relation_and_sixteen_lease_capacity_are_explicit() {
        let (stores, state) = stores(&Position::startpos());
        let mut index = ReviewContextIndex::default();
        assert_eq!(
            index.publish(
                &stores,
                &record(state, 1, RecordKind::Repair, "e2e4", Some(99)),
                None
            ),
            Err(ReviewError::MissingRelation(99))
        );
        let leases: Vec<_> = (0..16)
            .map(|_| {
                index
                    .select(
                        &stores,
                        state,
                        empty_query(),
                        ReviewSelectionPolicy::default(),
                    )
                    .unwrap()
            })
            .collect();
        assert!(matches!(
            index.select(
                &stores,
                state,
                empty_query(),
                ReviewSelectionPolicy::default()
            ),
            Err(ReviewError::Store(StoreError::PinSaturated(
                "review view leases"
            )))
        ));
        drop(leases);
        assert_eq!(stores.review_pin_count().unwrap(), 0);
    }

    #[test]
    fn same_store_numeric_bundle_collision_does_not_transfer_index_authority() {
        let (stores, state) = stores(&Position::startpos());
        let mut a = ReviewContextIndex::default();
        let mut b = ReviewContextIndex::default();
        let p = record(state, 1, RecordKind::Proposal, "e2e4", None);
        let c = record(state, 2, RecordKind::Counterexample, "e2e4 e7e5", Some(1));
        for index in [&mut a, &mut b] {
            index.publish(&stores, &p, None).unwrap();
            index.publish(&stores, &c, None).unwrap();
        }
        let view = a
            .select(
                &stores,
                state,
                empty_query(),
                ReviewSelectionPolicy::default(),
            )
            .unwrap();
        assert_eq!(b.consume(&view, &stores), Err(ReviewError::OwnerMismatch));
        assert_eq!(b.bundle(0).unwrap().protected_consumptions(), 2);
        a.consume(&view, &stores).unwrap();
        assert_eq!(a.bundle(0).unwrap().protected_consumptions(), 1);
    }

    #[test]
    fn v2_raw_features_and_ordered_candidates_are_included_in_input_identity() {
        let (stores, state) = stores(&Position::startpos());
        let mut a = ReviewContextIndex::default();
        let mut b = ReviewContextIndex::default();
        let p = record(state, 1, RecordKind::Proposal, "e2e4", None);
        let mut changed = p.clone();
        changed.value = Some(10);
        changed.completed_depth = 3;
        a.publish(&stores, &p, None).unwrap();
        b.publish(&stores, &changed, None).unwrap();
        let original = a
            .select(
                &stores,
                state,
                empty_query(),
                ReviewSelectionPolicy::default(),
            )
            .unwrap();
        let altered = b
            .select(
                &stores,
                state,
                empty_query(),
                ReviewSelectionPolicy::default(),
            )
            .unwrap();
        assert_ne!(original.digest(), altered.digest());
        let compact = ReviewSelectionPolicy {
            mode: ReviewAnchorMode::AnchoredCompact,
            max_anchors: 4,
            ..ReviewSelectionPolicy::default()
        };
        assert_eq!(
            a.select(&stores, state, empty_query(), compact)
                .unwrap()
                .digest(),
            b.select(&stores, state, empty_query(), compact)
                .unwrap()
                .digest()
        );
        let forward = line("e2e4 d2d4");
        let backward = line("d2d4 e2e4");
        let q1 = ReviewQuerySequences {
            legal: &forward,
            ..empty_query()
        };
        let q2 = ReviewQuerySequences {
            legal: &backward,
            ..empty_query()
        };
        assert_ne!(
            a.select(&stores, state, q1, compact).unwrap().digest(),
            a.select(&stores, state, q2, compact).unwrap().digest()
        );
    }

    #[test]
    fn review_leases_protect_actual_archive_reclamation_and_close_until_last_drop() {
        use super::super::store::{ArchiveConfig, ArchiveIoBudget};
        use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
        struct Temporary(std::path::PathBuf);
        impl Drop for Temporary {
            fn drop(&mut self) {
                if self.0.parent() == Some(std::env::temp_dir().as_path())
                    && self.0.file_name().is_some_and(|name| {
                        name.to_string_lossy().starts_with("rz-review-pin-test-")
                    })
                    && std::fs::symlink_metadata(&self.0)
                        .is_ok_and(|meta| meta.is_dir() && !meta.file_type().is_symlink())
                {
                    let _ = std::fs::remove_dir_all(&self.0);
                }
            }
        }
        let root = Temporary(std::env::temp_dir().join(format!(
                "rz-review-pin-test-{}-{}",
                std::process::id(),
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            )));
        std::fs::create_dir(&root.0).unwrap();
        let (mut stores, old) = stores(&Position::startpos());
        stores
            .enable_archive(ArchiveConfig::new(
                root.0.join("archive"),
                std::env::current_dir().unwrap(),
            ))
            .unwrap();
        let mut index = ReviewContextIndex::default();
        index
            .publish(
                &stores,
                &record(old, 1, RecordKind::Proposal, "e2e4", None),
                None,
            )
            .unwrap();
        let first = index
            .select(
                &stores,
                old,
                empty_query(),
                ReviewSelectionPolicy::default(),
            )
            .unwrap();
        let second = index
            .select(
                &stores,
                old,
                empty_query(),
                ReviewSelectionPolicy::default(),
            )
            .unwrap();
        let mut current = Position::startpos();
        current.make_uci("d2d4").unwrap();
        stores.focus_actual_moves(current.snapshot()).unwrap();
        let mut inactive = Position::startpos();
        inactive.make_uci("b2b4").unwrap();
        let unused = stores.states.insert(inactive.snapshot()).unwrap();
        let budget = || ArchiveIoBudget {
            deadline: Instant::now() + Duration::from_secs(10),
            max_bytes: 8 * 1024 * 1024,
        };
        assert!(stores.archive_inactive(budget()).unwrap().is_some());
        assert!(stores.states.get(old).is_ok());
        assert_eq!(
            stores.states.get(unused).unwrap_err(),
            StoreError::ColdRecord("state")
        );
        drop(first);
        assert_eq!(stores.review_pin_count().unwrap(), 1);
        assert!(matches!(
            stores.close_archive_owner(),
            Err(StoreError::InvalidConditions(
                "archive close retains pending ownership"
            ))
        ));
        drop(second);
        assert!(stores.archive_inactive(budget()).unwrap().is_some());
        assert_eq!(
            stores.states.get(old).unwrap_err(),
            StoreError::ColdRecord("state")
        );
        assert!(stores.close_archive_owner().unwrap().is_some());
    }

    #[test]
    fn impossible_query_and_duplicate_legal_mask_are_rejected_before_pin() {
        let (stores, state) = stores(&Position::startpos());
        let mut index = ReviewContextIndex::default();
        let illegal = line("e2e4 e2e3");
        assert!(matches!(
            index.select(
                &stores,
                state,
                ReviewQuerySequences {
                    proposal: &illegal,
                    ..empty_query()
                },
                ReviewSelectionPolicy::default()
            ),
            Err(ReviewError::InvalidLine)
        ));
        let duplicate = line("e2e4 e2e4");
        assert!(matches!(
            index.select(
                &stores,
                state,
                ReviewQuerySequences {
                    legal: &duplicate,
                    ..empty_query()
                },
                ReviewSelectionPolicy::default()
            ),
            Err(ReviewError::InvalidLine)
        ));
        assert_eq!(stores.review_pin_count().unwrap(), 0);
    }

    #[test]
    fn task_policy_is_append_only_and_sealed_with_its_registered_budget() {
        for (expected, purpose) in [
            RoleQueryPurpose::ProposePolicy,
            RoleQueryPurpose::ReplyPolicy,
            RoleQueryPurpose::RepairPolicy,
            RoleQueryPurpose::DivergencePolicy,
            RoleQueryPurpose::ValueFresh,
        ]
        .into_iter()
        .enumerate()
        {
            assert_eq!(purpose as u8, expected as u8);
        }
        assert_eq!(RoleQueryPurpose::TaskPolicy as u8, 5);
        let (stores, state) = stores(&Position::startpos());
        let mut index = ReviewContextIndex::default();
        for token_budget in [128, 256, 512] {
            let policy = ReviewSelectionPolicy {
                mode: ReviewAnchorMode::AnchoredCompact,
                max_anchors: 4,
                token_budget,
                ..ReviewSelectionPolicy::default()
            };
            let task = index
                .select(
                    &stores,
                    state,
                    ReviewQuerySequences {
                        purpose: RoleQueryPurpose::TaskPolicy,
                        ..empty_query()
                    },
                    policy,
                )
                .unwrap();
            let proposer = index.select(&stores, state, empty_query(), policy).unwrap();
            assert_eq!(task.query().purpose, RoleQueryPurpose::TaskPolicy);
            assert_eq!(task.token_budget(), token_budget);
            assert_ne!(task.digest(), proposer.digest());
            assert_ne!(task.seal_digest(), proposer.seal_digest());
            assert_eq!(task.record_scope_digest(), proposer.record_scope_digest());
        }
        assert_eq!(stores.review_pin_count().unwrap(), 0);
    }

    #[test]
    fn checked_rebase_keeps_castling_en_passant_and_four_promotions_exact() {
        for (fen, text) in [
            ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1g1 e8c8"),
            ("k7/8/8/3pP3/8/8/8/7K w - d6 0 2", "e5d6 a8a7"),
        ] {
            let start = Position::from_fen(fen).unwrap();
            let moves = line(text);
            let anchor = start.snapshot();
            let mut current = start;
            current.make_move(moves[0]).unwrap();
            assert_eq!(
                checked_rebase_offset(&anchor, &current.snapshot(), &moves).unwrap(),
                Some(1)
            );
        }
        for suffix in ['q', 'r', 'b', 'n'] {
            let start = Position::from_fen("7k/P7/8/8/8/8/8/K7 w - - 0 1").unwrap();
            let movement = BoardMove::from_uci(&format!("a7a8{suffix}")).unwrap();
            let anchor = start.snapshot();
            let mut current = start;
            current.make_move(movement).unwrap();
            assert_eq!(
                checked_rebase_offset(&anchor, &current.snapshot(), &[movement]).unwrap(),
                Some(1)
            );
        }
    }
}
