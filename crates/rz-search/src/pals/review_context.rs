//! Search-owned immutable, bounded review inputs. Raw RoleRecords and cold
//! archives remain evidence; selecting a view never deletes or rewrites them.
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Weak};

use rz_position::{
    BoardMove, Color, MAX_UCI_REPLAY_PLIES, Position, PositionLimits, PositionSnapshot,
};
use sha2::{Digest, Sha256};

#[cfg(feature = "archive-origin-diagnostics")]
pub use super::engine::archive_lifecycle::ArchiveRecordOrigin;
use super::engine::{RecordKind, RoleQueryPurpose, RoleRecord};
use super::store::{
    EvidenceScope, Move16, ObservationId, ObservationKind, PalsStores, StateId, StoreError,
    StorePins, StoreReadOwner, StoreReadPin,
};

pub mod cpu7_capture;
pub mod historical;

pub const REVIEW_CONTEXT_VERSION: &str = "rz-pals-selected-review/1";
pub(in crate::pals) const MAX_LINE: usize = 256;

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

/// Original, bounded index contents observed at CPU7 selection. A compacted
/// index is not the complete game journal. This is detached evidence only:
/// it cannot create a view, trial, or worker authority.
#[derive(Debug)]
pub struct CohortCpu7OriginalRecord {
    record: RoleRecord,
    evidence: Option<ObservationId>,
}
impl CohortCpu7OriginalRecord {
    /// Detached history only; this does not issue a Store pin or a live view.
    pub(in crate::pals) fn historical(record: RoleRecord, evidence: Option<ObservationId>) -> Self {
        Self { record, evidence }
    }
    pub fn record(&self) -> &RoleRecord {
        &self.record
    }
    pub fn evidence(&self) -> Option<ObservationId> {
        self.evidence
    }
}

#[derive(Debug)]
pub struct CohortCpu7OriginalBundle {
    bundle_id: u64,
    generation: u64,
    root: StateId,
    root_snapshot: PositionSnapshot,
    anchor: PositionSnapshot,
    records: Vec<CohortCpu7OriginalRecord>,
    selected_len: usize,
    digest: [u8; 32],
}
impl CohortCpu7OriginalBundle {
    pub fn bundle_id(&self) -> u64 {
        self.bundle_id
    }
    pub fn generation(&self) -> u64 {
        self.generation
    }
    pub fn root(&self) -> StateId {
        self.root
    }
    pub fn root_snapshot(&self) -> &PositionSnapshot {
        &self.root_snapshot
    }
    pub fn anchor(&self) -> &PositionSnapshot {
        &self.anchor
    }
    pub fn records(&self) -> &[CohortCpu7OriginalRecord] {
        &self.records
    }
    pub fn selected_len(&self) -> usize {
        self.selected_len
    }
    pub fn digest(&self) -> [u8; 32] {
        self.digest
    }
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
    cohort_cpu7: Option<Arc<CohortCpu7OriginalBundle>>,
    cohort_cpu7_binding: Option<[u8; 32]>,
    // Selection-local diagnostic only. Never carried to another query/view.
    cpu7_raw_capture:
        Option<Result<cpu7_capture::SelectedCpuRawCapture, cpu7_capture::CaptureUnavailable>>,
    records: Vec<RoleRecord>,
    metadata: Vec<ReviewRecordMetadata>,
    // Optional diagnostic evidence, aligned with selected records if present.
    // It never authorizes a load, selection, submission, or consumption.
    #[cfg(feature = "archive-origin-diagnostics")]
    archive_origins: Vec<Option<ArchiveRecordOrigin>>,
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
    /// True only for an original view selected by the explicit CPU7 cohort
    /// policy; it is not an acceptance or physical-completion receipt.
    pub fn is_cohort_cpu7(&self) -> bool {
        self.cohort_cpu7.is_some()
    }
    pub fn cohort_cpu7_original_bundle(&self) -> Option<&CohortCpu7OriginalBundle> {
        self.cohort_cpu7.as_deref()
    }
    pub fn cohort_cpu7_binding(&self) -> Option<[u8; 32]> {
        self.cohort_cpu7_binding
    }
    pub fn cpu7_raw_capture(
        &self,
    ) -> Option<&Result<cpu7_capture::SelectedCpuRawCapture, cpu7_capture::CaptureUnavailable>>
    {
        self.cpu7_raw_capture.as_ref()
    }
    pub(in crate::pals) fn capture_cpu7_on_original_view(
        &mut self,
        stores: &PalsStores,
        control: Result<cpu7_capture::CapturePermit<'_>, cpu7_capture::CaptureUnavailable>,
    ) {
        self.cpu7_raw_capture = Some(control.and_then(|control| {
            cpu7_capture::capture_selected_cpu_raw_permitted(self, stores, control)
        }));
    }
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
    /// Bounded historical provenance for an actually selected record. Store
    /// and archive owners matched when this view was formed, not necessarily
    /// when this getter is called. `None` is unknown, not evidence of a hot
    /// origin. This is not submission or physical-completion evidence and it
    /// does not issue a Store capability.
    #[cfg(feature = "archive-origin-diagnostics")]
    pub fn selected_archive_origin(&self, index: usize) -> Option<&ArchiveRecordOrigin> {
        self.archive_origins.get(index).and_then(Option::as_ref)
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
    /// Project the original CPU7 view's pinned Store dependency closure.
    /// The result is historical data only: it grants no execution authority,
    /// physical completion, or proof that any observation was actually run.
    /// `check` must enforce the caller's original finite deadline and cancel
    /// state without re-entering the Store or blocking. Output is discarded on
    /// any error, including an unavailable hot
    /// record, owner change, exhausted record/byte limit, deadline or cancel.
    pub fn project_cohort_cpu7_store(
        &self,
        stores: &PalsStores,
        max_records: usize,
        max_bytes: usize,
        check: &mut dyn FnMut() -> Result<(), StoreError>,
    ) -> Result<Vec<u8>, StoreError> {
        if self.cohort_cpu7.is_none() {
            return Err(StoreError::InvalidConditions("CPU7 view required"));
        }
        stores.project_review_store_closure(&self.pin, max_records, max_bytes, check)
    }
    /// Capture this exact live CPU7 selection and its Store projection under
    /// caller-owned deadline/cancel control. This detached evidence is not yet
    /// a published trial sidecar or a physical-completion proof.
    pub fn capture_cohort_cpu7_evidence(
        &self,
        stores: &PalsStores,
        max_records: usize,
        max_bytes: usize,
        check: &mut dyn FnMut() -> Result<(), StoreError>,
    ) -> Result<Vec<u8>, StoreError> {
        stores.capture_cohort_cpu7_view(self, &self.pin, max_records, max_bytes, check)
    }
    /// Known owned host backing, not a measured peak. Shared Rules history
    /// chains are conservatively charged once per retained anchor/source;
    /// physical sharing is deliberately not guessed from Arc identities.
    pub fn owned_host_bytes(&self) -> Option<u64> {
        let mut bytes = std::mem::size_of::<Self>();
        #[cfg(feature = "archive-origin-diagnostics")]
        {
            bytes = bytes.checked_add(
                self.archive_origins
                    .capacity()
                    .checked_mul(std::mem::size_of::<Option<ArchiveRecordOrigin>>())?,
            )?;
        }
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
        if let Some(original) = &self.cohort_cpu7 {
            bytes = bytes.checked_add(std::mem::size_of::<CohortCpu7OriginalBundle>())?;
            bytes = bytes.checked_add(
                original
                    .records
                    .capacity()
                    .checked_mul(std::mem::size_of::<CohortCpu7OriginalRecord>())?,
            )?;
            for record in &original.records {
                bytes = bytes.checked_add(
                    record
                        .record
                        .line
                        .capacity()
                        .checked_mul(std::mem::size_of::<BoardMove>())?,
                )?;
            }
            bytes = bytes.checked_add(original.anchor.retained_history_bytes()?)?;
            bytes = bytes.checked_add(original.root_snapshot.retained_history_bytes()?)?;
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
    #[cfg(feature = "archive-origin-diagnostics")]
    archive_origin: Option<ArchiveRecordOrigin>,
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
    /// Called only for a newly installed source record after the load owner has
    /// completed installation, release and re-pinning. Existing entries are
    /// never relabelled unless the caller has established that the entry was
    /// absent before that same load. Failure is diagnostic-only.
    #[cfg(feature = "archive-origin-diagnostics")]
    pub(crate) fn attach_archive_origin(
        &mut self,
        stores: &PalsStores,
        original: &RoleRecord,
        origin: ArchiveRecordOrigin,
    ) {
        if self.check_owner(stores).is_err() || !origin.matches_original(stores, original) {
            return;
        }
        let Some(entry) = self.records.get_mut(&original.revision) else {
            return;
        };
        let current = &entry.record;
        // `critical` is deliberately mutable index state and can differ after
        // normal root refresh or another publication of the same kind.
        if entry.archive_origin.is_some()
            || current.revision != original.revision
            || current.parent_revision != original.parent_revision
            || current.supersedes_revision != original.supersedes_revision
            || current.origin_state != original.origin_state
            || current.kind != original.kind
            || current.line != original.line
            || current.value != original.value
            || current.completed_depth != original.completed_depth
            || current.score_scope != original.score_scope
            || current.cpu_observation != original.cpu_observation
            || current.perspective != original.perspective
        {
            return;
        }
        entry.archive_origin = Some(origin);
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
                #[cfg(feature = "archive-origin-diagnostics")]
                archive_origin: None,
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
        self.select_inner(stores, root, query, policy, None)
    }

    /// Experimental cohort-only prefix. The original index and observations
    /// are untouched; this does not manufacture completion or trial authority.
    /// The required bundle must contain seven actual CPU verification records.
    pub fn select_cohort_cpu7(
        &mut self,
        stores: &PalsStores,
        root: StateId,
        query: ReviewQuerySequences<'_>,
        policy: ReviewSelectionPolicy,
    ) -> Result<ReviewView, ReviewError> {
        if policy.required_bundle.is_none() || policy.mode != ReviewAnchorMode::AnchoredCompact {
            return Err(ReviewError::InvalidPolicy);
        }
        self.select_inner(stores, root, query, policy, Some(7))
    }

    /// Optional diagnostic capture at the same single-writer selection call.
    /// Capture failure is returned separately and never changes the selected
    /// view or makes an otherwise valid neural request fail.
    pub fn select_cohort_cpu7_with_raw_capture(
        &mut self,
        stores: &PalsStores,
        root: StateId,
        query: ReviewQuerySequences<'_>,
        policy: ReviewSelectionPolicy,
        control: Option<&cpu7_capture::CaptureControl<'_>>,
    ) -> Result<
        (
            ReviewView,
            Result<cpu7_capture::SelectedCpuRawCapture, cpu7_capture::CaptureUnavailable>,
        ),
        ReviewError,
    > {
        let view = self.select_cohort_cpu7(stores, root, query, policy)?;
        let capture = cpu7_capture::capture_selected_cpu_raw(&view, stores, control);
        Ok((view, capture))
    }

    /// A readiness hint only: selection still validates owner, Rules,
    /// relations, budget and all original records before issuing any view.
    pub fn cohort_cpu7_ready(&self, stores: &PalsStores, policy: ReviewSelectionPolicy) -> bool {
        if policy.mode != ReviewAnchorMode::AnchoredCompact {
            return false;
        }
        let Some(bundle) = policy.required_bundle.and_then(|id| self.bundles.get(&id)) else {
            return false;
        };
        let (mut cpus, mut total) = (0, 0);
        let mut observations = std::collections::BTreeSet::new();
        for revision in &bundle.revisions {
            let Some(entry) = self.records.get(revision) else {
                return false;
            };
            total += 1;
            if total == 1 && entry.record.kind != RecordKind::Proposal {
                return false;
            }
            if entry.record.kind == RecordKind::CpuVerification {
                let Some(observation) = entry.record.cpu_observation else {
                    return false;
                };
                if !observations.insert(observation) {
                    return false;
                }
                let Ok(actual) = stores.observations.get(observation) else {
                    return false;
                };
                if !matches!(
                    (actual.kind, actual.scope),
                    (
                        ObservationKind::CpuAnalysis,
                        EvidenceScope::DepthLimited { .. }
                    ) | (
                        ObservationKind::ExternalCpuAnalysis,
                        EvidenceScope::ProfiledExternalUci { .. }
                            | EvidenceScope::ExternalUci { .. }
                    )
                ) {
                    return false;
                }
                cpus += 1;
            }
            if cpus == 7 {
                return total <= usize::from(policy.max_records);
            }
        }
        false
    }

    fn select_inner(
        &mut self,
        stores: &PalsStores,
        root: StateId,
        query: ReviewQuerySequences<'_>,
        mut policy: ReviewSelectionPolicy,
        cohort_cpu_prefix: Option<usize>,
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
        #[cfg(feature = "archive-origin-diagnostics")]
        let mut archive_origins = Vec::new();
        #[cfg(feature = "archive-origin-diagnostics")]
        let mut archive_origins_unavailable = false;
        let mut source_snapshots = Vec::new();
        let mut selected_bundles = Vec::new();
        let mut cpu7_original_bundle = None;
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
            if cohort_cpu_prefix.is_some() && !required {
                continue;
            }
            let revisions = if required {
                if let Some(limit) = cohort_cpu_prefix {
                    let mut observed = 0usize;
                    let end = bundle.revisions.iter().position(|revision| {
                        if self.records[revision].record.kind == RecordKind::CpuVerification {
                            observed += 1;
                        }
                        observed == limit
                    });
                    let end = end.ok_or(ReviewError::ViewCapacity)?;
                    let selected = &bundle.revisions[..=end];
                    if selected.len() > usize::from(policy.max_records)
                        || self.records[&selected[0]].record.kind != RecordKind::Proposal
                    {
                        return Err(ReviewError::ViewCapacity);
                    }
                    let mut observations = std::collections::BTreeSet::new();
                    for revision in selected {
                        let record = &self.records[revision].record;
                        if record.kind == RecordKind::CpuVerification {
                            let observation =
                                record.cpu_observation.ok_or(ReviewError::ViewCapacity)?;
                            if !observations.insert(observation) {
                                return Err(ReviewError::RelationMismatch);
                            }
                            let actual = stores.observations.get(observation)?;
                            if !matches!(
                                (actual.kind, actual.scope),
                                (
                                    ObservationKind::CpuAnalysis,
                                    EvidenceScope::DepthLimited { .. }
                                ) | (
                                    ObservationKind::ExternalCpuAnalysis,
                                    EvidenceScope::ProfiledExternalUci { .. }
                                        | EvidenceScope::ExternalUci { .. }
                                )
                            ) {
                                return Err(ReviewError::ViewCapacity);
                            }
                        }
                    }
                    // Bind the actual prefix length into the existing context
                    // policy digest, without exceeding the registered cap.
                    policy.max_records = selected.len() as u16;
                    selected
                } else {
                    bundle.revisions.as_slice()
                }
            } else {
                bundle.revisions.as_slice()
            };
            // A prefix is valid only when every included parent/supersedes
            // really belongs to the same retained prefix, never a missing tail.
            if required && cohort_cpu_prefix.is_some() {
                for revision in revisions {
                    let record = &self.records[revision].record;
                    for parent in [record.parent_revision, record.supersedes_revision]
                        .into_iter()
                        .flatten()
                    {
                        if !revisions.contains(&parent) {
                            return Err(ReviewError::MissingRelation(parent));
                        }
                    }
                }
            }
            let anchor = match stores.states.get(bundle.anchor) {
                Ok(anchor) => anchor,
                Err(error) => {
                    if required {
                        return Err(error.into());
                    }
                    continue;
                }
            };
            if required && cohort_cpu_prefix.is_some() {
                let mut original = Vec::with_capacity(bundle.revisions.len());
                for revision in &bundle.revisions {
                    let entry = self
                        .records
                        .get(revision)
                        .ok_or(ReviewError::MissingRelation(*revision))?;
                    if entry.bundle != id || entry.record.origin_state != bundle.anchor {
                        return Err(ReviewError::RelationMismatch);
                    }
                    pins.states.insert(entry.record.origin_state);
                    pins.observations.extend(entry.evidence);
                    pins.observations.extend(entry.record.cpu_observation);
                    original.push(CohortCpu7OriginalRecord {
                        record: entry.record.clone(),
                        evidence: entry.evidence,
                    });
                }
                let digest = cohort_cpu7_original_digest(
                    id,
                    stores.generation(),
                    root,
                    root_snapshot,
                    anchor,
                    &original,
                    revisions.len(),
                )?;
                cpu7_original_bundle = Some(Arc::new(CohortCpu7OriginalBundle {
                    bundle_id: id,
                    generation: stores.generation(),
                    root,
                    root_snapshot: root_snapshot.clone(),
                    anchor: anchor.clone(),
                    records: original,
                    selected_len: revisions.len(),
                    digest,
                }));
            }
            let mut offsets = Vec::with_capacity(revisions.len());
            let mut all_rebased = true;
            for revision in revisions {
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
            let additional: usize = revisions
                .iter()
                .zip(&offsets)
                .map(|(rev, offset)| self.records[rev].record.line.len() - offset)
                .sum();
            if records.len() + revisions.len() > usize::from(policy.max_records)
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
            let local: BTreeMap<_, _> = revisions
                .iter()
                .enumerate()
                .map(|(index, revision)| (*revision, (records.len() + index) as u16))
                .collect();
            for (revision, offset) in revisions.iter().zip(offsets) {
                let entry = &self.records[revision];
                #[cfg(feature = "archive-origin-diagnostics")]
                if !archive_origins_unavailable {
                    let origin = entry
                        .archive_origin
                        .as_ref()
                        .filter(|origin| origin.matches_owner(stores));
                    if archive_origins.is_empty() && origin.is_some() {
                        if archive_origins
                            .try_reserve_exact(usize::from(policy.max_records))
                            .is_err()
                        {
                            archive_origins_unavailable = true;
                        } else {
                            archive_origins.resize_with(records.len(), || None);
                        }
                    }
                    if !archive_origins_unavailable
                        && (origin.is_some() || !archive_origins.is_empty())
                    {
                        archive_origins.push(origin.cloned());
                    }
                }
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
        let cohort_cpu7_binding = cpu7_original_bundle
            .as_ref()
            .map(|source| cohort_cpu7_view_binding(source.digest(), policy.digest(), digest, seal));
        let pin = stores.pin_review(pins)?;
        Ok(ReviewView {
            index_owner: Arc::downgrade(&self.identity),
            sequence,
            root_state: root,
            generation: stores.generation(),
            policy,
            cohort_cpu7: cpu7_original_bundle,
            cohort_cpu7_binding,
            cpu7_raw_capture: None,
            records,
            metadata,
            #[cfg(feature = "archive-origin-diagnostics")]
            archive_origins,
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
        let cohort_cpu7_binding = source.cohort_cpu7.as_ref().map(|original| {
            cohort_cpu7_view_binding(original.digest(), source.policy.digest(), digest, seal)
        });
        #[cfg(feature = "archive-origin-diagnostics")]
        let archive_origins = {
            let mut origins = Vec::new();
            if origins
                .try_reserve_exact(source.archive_origins.len())
                .is_ok()
            {
                origins.extend(source.archive_origins.iter().map(|origin| {
                    origin
                        .as_ref()
                        .filter(|origin| origin.matches_owner(stores))
                        .cloned()
                }));
            }
            origins
        };
        Ok(ReviewView {
            index_owner: Arc::downgrade(&self.identity),
            sequence,
            root_state: source.root_state,
            generation: source.generation,
            policy: source.policy,
            cohort_cpu7: source.cohort_cpu7.clone(),
            cohort_cpu7_binding,
            cpu7_raw_capture: None,
            records: source.records.clone(),
            metadata: source.metadata.clone(),
            #[cfg(feature = "archive-origin-diagnostics")]
            archive_origins,
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
/// Link an original index snapshot to this exact selected query view. This
/// remains per-root metadata, not a substitute for trial or model authority.
pub(in crate::pals) fn cohort_cpu7_view_binding(
    original: [u8; 32],
    policy: [u8; 32],
    input: [u8; 32],
    view_seal: [u8; 32],
) -> [u8; 32] {
    let mut hash = Sha256::new();
    hash.update(b"rz-pals-cohort-cpu7-view-binding/1\0");
    hash.update(original);
    hash.update(policy);
    hash.update(input);
    hash.update(view_seal);
    hash.finalize().into()
}

/// Hash original index records and observation handles independently from the
/// selected model input. Observation/line payloads are not hashed here. Each
/// root has its own evidence digest and identity; this is not historical proof.
pub(in crate::pals) fn cohort_cpu7_original_digest(
    bundle_id: u64,
    generation: u64,
    root: StateId,
    root_snapshot: &PositionSnapshot,
    anchor: &PositionSnapshot,
    records: &[CohortCpu7OriginalRecord],
    selected_len: usize,
) -> Result<[u8; 32], ReviewError> {
    fn optional(hash: &mut Sha256, value: Option<u64>) {
        hash.update([u8::from(value.is_some())]);
        hash.update(value.unwrap_or_default().to_le_bytes());
    }
    if selected_len == 0 || selected_len > records.len() || records.len() > 64 {
        return Err(ReviewError::ViewCapacity);
    }
    let mut hash = Sha256::new();
    hash.update(b"rz-pals-cohort-cpu7-original-bundle/1\0");
    hash.update(bundle_id.to_le_bytes());
    hash.update(generation.to_le_bytes());
    hash.update((root.0 as u64).to_le_bytes());
    hash.update(snapshot_digest(root_snapshot)?);
    hash.update(snapshot_digest(anchor)?);
    hash.update((selected_len as u16).to_le_bytes());
    hash.update((records.len() as u16).to_le_bytes());
    for item in records {
        let record = &item.record;
        hash.update(record.revision.to_le_bytes());
        optional(&mut hash, record.parent_revision);
        optional(&mut hash, record.supersedes_revision);
        hash.update((record.origin_state.0 as u64).to_le_bytes());
        hash.update([
            record.kind as u8,
            u8::from(record.perspective == Color::Black),
            u8::from(record.critical),
        ]);
        write_moves(&mut hash, &record.line)?;
        hash.update([u8::from(record.value.is_some())]);
        hash.update(record.value.unwrap_or_default().to_le_bytes());
        hash.update(record.completed_depth.to_le_bytes());
        hash.update([match record.score_scope {
            None => 0,
            Some(crate::cpu::CpuScoreScope::FrontierOnly) => 1,
            Some(crate::cpu::CpuScoreScope::CompletedIteration) => 2,
            Some(crate::cpu::CpuScoreScope::RulesTerminal) => 3,
        }]);
        optional(&mut hash, record.cpu_observation.map(|id| id.0 as u64));
        optional(&mut hash, item.evidence.map(|id| id.0 as u64));
    }
    Ok(hash.finalize().into())
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

    // Typed fixture evidence, not a claim that a physical CPU worker ran.
    fn cpu_fixture_observation(
        stores: &mut PalsStores,
        state: StateId,
        sequence: u64,
    ) -> ObservationId {
        use super::super::store::{BoundKind, Observation, RawScore};
        stores
            .observations
            .append(Observation {
                state,
                line: None,
                source: sequence,
                epoch: 1,
                value_identity: Some(crate::cpu_value::CpuValueIdentity {
                    semantics: crate::cpu::BOOTSTRAP_SCORE_VERSION.into(),
                    weights_sha256: None,
                    training: crate::cpu_value::CpuTrainingState::Bootstrap,
                }),
                checker_identity: None,
                checker_work: None,
                external_report: None,
                model_value_identity: None,
                model_value_input: None,
                cpu_condition: Some("cpu7-fixture-conditions".into()),
                cpu_pv: None,
                scope: EvidenceScope::DepthLimited {
                    depth: 1,
                    profile: 1,
                    condition: 1,
                },
                score: RawScore::Cpu {
                    value: 0,
                    perspective: Color::White,
                    bound: BoundKind::ExactWithinSearch,
                },
                budget: 1,
                kind: ObservationKind::CpuAnalysis,
                supersedes: None,
                execution: None,
            })
            .unwrap()
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
    fn cohort_cpu_prefix_requires_seven_and_preserves_later_observations() {
        let (mut stores, state) = stores(&Position::startpos());
        let mut index = ReviewContextIndex::default();
        let p = record(state, 1, RecordKind::Proposal, "e2e4 e7e5", None);
        let c = record(
            state,
            2,
            RecordKind::Counterexample,
            "e2e4 e7e5 g1f3",
            Some(1),
        );
        let bundle = index.publish(&stores, &p, None).unwrap();
        index.publish(&stores, &c, None).unwrap();
        for revision in 3..=8 {
            let mut cpu = record(
                state,
                revision,
                RecordKind::CpuVerification,
                "e2e4 e7e5",
                Some(if revision == 3 { 1 } else { revision - 1 }),
            );
            cpu.supersedes_revision = (revision > 3).then_some(revision - 1);
            cpu.cpu_observation = Some(cpu_fixture_observation(&mut stores, state, revision));
            index.publish(&stores, &cpu, None).unwrap();
        }
        let policy = ReviewSelectionPolicy {
            required_bundle: Some(bundle),
            mode: ReviewAnchorMode::AnchoredCompact,
            ..ReviewSelectionPolicy::default()
        };
        assert!(matches!(
            index.select_cohort_cpu7(&stores, state, empty_query(), policy),
            Err(ReviewError::ViewCapacity)
        ));
        let mut seventh = record(state, 9, RecordKind::CpuVerification, "e2e4 e7e5", Some(8));
        seventh.supersedes_revision = Some(8);
        seventh.cpu_observation = Some(cpu_fixture_observation(&mut stores, state, 9));
        index.publish(&stores, &seventh, None).unwrap();
        let limited = ReviewSelectionPolicy {
            max_records: 8,
            ..policy
        };
        assert!(!index.cohort_cpu7_ready(&stores, limited));
        assert!(matches!(
            index.select_cohort_cpu7(&stores, state, empty_query(), limited),
            Err(ReviewError::ViewCapacity)
        ));
        let legacy = ReviewSelectionPolicy {
            mode: ReviewAnchorMode::CurrentRootOnly,
            ..policy
        };
        assert!(!index.cohort_cpu7_ready(&stores, legacy));
        assert!(matches!(
            index.select_cohort_cpu7(&stores, state, empty_query(), legacy),
            Err(ReviewError::InvalidPolicy)
        ));
        let (_, missing_control) = index
            .select_cohort_cpu7_with_raw_capture(&stores, state, empty_query(), policy, None)
            .unwrap();
        assert!(matches!(
            missing_control,
            Err(cpu7_capture::CaptureUnavailable::MissingOriginalControl)
        ));
        let canceled = AtomicBool::new(false);
        let mut control = cpu7_capture::CaptureControl {
            deadline: std::time::Instant::now() + std::time::Duration::from_secs(10),
            canceled: &canceled,
            limits: cpu7_capture::CaptureLimits {
                max_observations: 7,
                max_line_chunks: 0,
                max_states: 1,
                max_owned_bytes: 1 << 16,
                max_retained_history_bytes: 1 << 20,
            },
        };
        let (captured_view, raw) = index
            .select_cohort_cpu7_with_raw_capture(
                &stores,
                state,
                empty_query(),
                policy,
                Some(&control),
            )
            .unwrap();
        let raw = raw.unwrap();
        assert_eq!(
            raw.view_binding(),
            captured_view.cohort_cpu7_binding().unwrap()
        );
        assert_eq!(raw.original_bundle().selected_len(), 9);
        assert_eq!(raw.states().len(), 1);
        assert!(raw.lines().is_empty());
        assert_eq!(
            serde_json::from_slice::<serde_json::Value>(raw.observations_json())
                .unwrap()
                .as_array()
                .unwrap()
                .len(),
            7
        );
        // A test scope exercises the reservation mechanics; it is not an
        // admission or a claim about the UCI session owner.
        struct DiagnosticTestScope;
        impl cpu7_capture::CaptureScope for DiagnosticTestScope {
            fn check_scope(&self) -> Result<(), cpu7_capture::CaptureUnavailable> {
                Ok(())
            }
        }
        let cancel = Arc::new(AtomicBool::new(false));
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        let grant = cpu7_capture::RootCaptureGrant::unverified_diagnostic(
            Arc::clone(&cancel),
            deadline,
            deadline,
            Arc::new(DiagnosticTestScope),
        )
        .with_diagnostic_limits(control.limits, std::time::Duration::from_secs(10));
        let permit = grant.control(Some(&cancel), deadline).unwrap();
        let guarded =
            cpu7_capture::capture_selected_cpu_raw_permitted(&captured_view, &stores, permit)
                .unwrap();
        assert_eq!(
            guarded.view_binding(),
            captured_view.cohort_cpu7_binding().unwrap()
        );
        assert!(guarded.original_bundle_projection().is_none());
        assert!(matches!(
            grant.control(Some(&cancel), deadline),
            Err(cpu7_capture::CaptureUnavailable::CaptureBusy)
        ));
        drop(guarded);
        drop(grant.control(Some(&cancel), deadline).unwrap());

        let mut projected_limits = control.limits;
        projected_limits.max_observations = 64;
        projected_limits.max_line_chunks = 64;
        projected_limits.max_states = 16;
        projected_limits.max_owned_bytes = 1 << 20;
        let projection_grant = cpu7_capture::RootCaptureGrant::unverified_diagnostic(
            Arc::clone(&cancel),
            deadline,
            deadline,
            Arc::new(DiagnosticTestScope),
        )
        .with_diagnostic_limits(projected_limits, std::time::Duration::from_secs(10))
        .with_original_bundle_projection(cpu7_capture::BundleProjectionLimits {
            max_records: 128,
            max_serialized_bytes: 1 << 20,
        });
        let projected = cpu7_capture::capture_selected_cpu_raw_permitted(
            &captured_view,
            &stores,
            projection_grant.control(Some(&cancel), deadline).unwrap(),
        )
        .unwrap();
        let projection = projected.original_bundle_projection().unwrap().unwrap();
        let value: serde_json::Value = serde_json::from_slice(projection.bytes()).unwrap();
        assert_eq!(value["schema"], "rz-pals-cpu7-view-evidence/1");
        assert_eq!(value["selected_len"], 9);
        drop(projected);
        assert!(matches!(
            projection_grant.control(Some(&cancel), deadline),
            Err(cpu7_capture::CaptureUnavailable::CaptureBusy)
        ));
        drop(projection);
        drop(projection_grant.control(Some(&cancel), deadline).unwrap());

        // A separate one-byte projection budget cannot hold the JSON object;
        // the already-complete selected-observation capture remains valid.
        let limited_projection_grant = cpu7_capture::RootCaptureGrant::unverified_diagnostic(
            Arc::clone(&cancel),
            deadline,
            deadline,
            Arc::new(DiagnosticTestScope),
        )
        .with_diagnostic_limits(projected_limits, std::time::Duration::from_secs(10))
        .with_original_bundle_projection(cpu7_capture::BundleProjectionLimits {
            max_records: 128,
            max_serialized_bytes: 1,
        });
        let raw_with_failed_projection = cpu7_capture::capture_selected_cpu_raw_permitted(
            &captured_view,
            &stores,
            limited_projection_grant
                .control(Some(&cancel), deadline)
                .unwrap(),
        )
        .unwrap();
        assert!(matches!(
            raw_with_failed_projection.original_bundle_projection(),
            Some(Err(_))
        ));
        assert_eq!(
            serde_json::from_slice::<serde_json::Value>(
                raw_with_failed_projection.observations_json()
            )
            .unwrap()
            .as_array()
            .unwrap()
            .len(),
            7
        );
        drop(raw_with_failed_projection);
        drop(
            limited_projection_grant
                .control(Some(&cancel), deadline)
                .unwrap(),
        );

        // Failure must release the same root reservation, not poison it.
        let mut tiny = control.limits;
        tiny.max_owned_bytes = 1;
        let small_grant = cpu7_capture::RootCaptureGrant::unverified_diagnostic(
            Arc::clone(&cancel),
            deadline,
            deadline,
            Arc::new(DiagnosticTestScope),
        )
        .with_diagnostic_limits(tiny, std::time::Duration::from_secs(10));
        let permit = small_grant.control(Some(&cancel), deadline).unwrap();
        assert!(matches!(
            cpu7_capture::capture_selected_cpu_raw_permitted(&captured_view, &stores, permit),
            Err(cpu7_capture::CaptureUnavailable::Capacity)
        ));
        drop(small_grant.control(Some(&cancel), deadline).unwrap());
        control.limits.max_owned_bytes = 1;
        let (still_valid, exhausted) = index
            .select_cohort_cpu7_with_raw_capture(
                &stores,
                state,
                empty_query(),
                policy,
                Some(&control),
            )
            .unwrap();
        assert!(still_valid.is_cohort_cpu7());
        assert!(matches!(
            exhausted,
            Err(cpu7_capture::CaptureUnavailable::Capacity)
        ));
        canceled.store(true, Ordering::Release);
        let (still_valid, interrupted) = index
            .select_cohort_cpu7_with_raw_capture(
                &stores,
                state,
                empty_query(),
                policy,
                Some(&control),
            )
            .unwrap();
        assert!(still_valid.is_cohort_cpu7());
        assert!(matches!(
            interrupted,
            Err(cpu7_capture::CaptureUnavailable::Canceled)
        ));
        canceled.store(false, Ordering::Release);
        control.limits.max_owned_bytes = 1 << 16;
        control.limits.max_retained_history_bytes = 1;
        let (still_valid, history_limit) = index
            .select_cohort_cpu7_with_raw_capture(
                &stores,
                state,
                empty_query(),
                policy,
                Some(&control),
            )
            .unwrap();
        assert!(still_valid.is_cohort_cpu7());
        assert!(matches!(
            history_limit,
            Err(cpu7_capture::CaptureUnavailable::Capacity)
        ));
        control.deadline = std::time::Instant::now();
        let (still_valid, expired) = index
            .select_cohort_cpu7_with_raw_capture(
                &stores,
                state,
                empty_query(),
                policy,
                Some(&control),
            )
            .unwrap();
        assert!(still_valid.is_cohort_cpu7());
        assert!(matches!(
            expired,
            Err(cpu7_capture::CaptureUnavailable::Deadline)
        ));
        let first = index
            .select_cohort_cpu7(&stores, state, empty_query(), policy)
            .unwrap();
        assert_eq!(first.records().len(), 9);
        assert!(first.is_cohort_cpu7());
        let first_evidence = first
            .capture_cohort_cpu7_evidence(&stores, 128, 1 << 20, &mut || Ok(()))
            .unwrap();
        let captured: serde_json::Value = serde_json::from_slice(&first_evidence).unwrap();
        assert_eq!(captured["schema"], "rz-pals-cpu7-view-evidence/1");
        assert_eq!(captured["records"].as_array().unwrap().len(), 9);
        assert_eq!(captured["selected_len"], 9);
        assert_eq!(
            captured["store"]["observations"].as_array().unwrap().len(),
            7
        );
        let standalone = first
            .project_cohort_cpu7_store(&stores, 128, 1 << 20, &mut || Ok(()))
            .unwrap();
        assert_eq!(
            captured["store"],
            serde_json::from_slice::<serde_json::Value>(&standalone).unwrap()
        );
        assert_eq!(
            first.capture_cohort_cpu7_evidence(&stores, 128, 1, &mut || Ok(())),
            Err(StoreError::ArchiveByteBudget)
        );
        assert_eq!(
            first.capture_cohort_cpu7_evidence(&stores, 128, 1 << 20, &mut || Err(
                StoreError::CancelledConsumer
            )),
            Err(StoreError::CancelledConsumer)
        );
        assert_eq!(
            first.cohort_cpu7_original_bundle().unwrap().records().len(),
            9
        );
        assert_ne!(first.selection_policy().digest(), policy.digest());
        // Keep the actual pinned source while later observations arrive.
        let first_scope = first.record_scope_digest();
        for extra in 10..=12 {
            let mut later = record(
                state,
                extra,
                RecordKind::CpuVerification,
                "e2e4 e7e5",
                Some(extra - 1),
            );
            later.supersedes_revision = Some(extra - 1);
            later.cpu_observation = Some(cpu_fixture_observation(&mut stores, state, extra));
            index.publish(&stores, &later, None).unwrap();
        }
        let again = index
            .select_cohort_cpu7(&stores, state, empty_query(), policy)
            .unwrap();
        assert_eq!(again.record_scope_digest(), first_scope);
        assert_eq!(again.records_metadata()[8].supersedes, Some(7));
        let original = again.cohort_cpu7_original_bundle().unwrap();
        assert_eq!(original.records().len(), 12);
        assert_eq!(original.selected_len(), 9);
        let evidence = again
            .capture_cohort_cpu7_evidence(&stores, 128, 1 << 20, &mut || Ok(()))
            .unwrap();
        let captured: serde_json::Value = serde_json::from_slice(&evidence).unwrap();
        assert_eq!(captured["records"].as_array().unwrap().len(), 12);
        assert_eq!(
            captured["store"]["observations"].as_array().unwrap().len(),
            10
        );
        assert_eq!(
            original.records()[8].record().revision,
            again.records_metadata()[8].source_revision
        );
        assert_eq!(
            first.cohort_cpu7_original_bundle().unwrap().records().len(),
            9
        );
        assert_ne!(
            original.digest(),
            first.cohort_cpu7_original_bundle().unwrap().digest()
        );
        assert_ne!(again.cohort_cpu7_binding(), first.cohort_cpu7_binding());
        let ordinary = index.select(&stores, state, empty_query(), policy).unwrap();
        assert_eq!(ordinary.records().len(), 12);
        assert!(!ordinary.is_cohort_cpu7());
        assert!(matches!(
            ordinary.capture_cohort_cpu7_evidence(&stores, 128, 1 << 20, &mut || Ok(())),
            Err(StoreError::InvalidConditions(_))
        ));
    }

    #[test]
    fn cpu7_raw_capture_preserves_marker_model_dependencies_and_non_cpu_record_cpu() {
        use super::super::store::{Observation, ObservationKind, RawScore};
        let (mut stores, state) = stores(&Position::startpos());
        let mut index = ReviewContextIndex::default();
        let root_line = stores.lines.root(state).unwrap();
        let p_line = stores.lines.append(root_line, &line("e2e4 e7e5")).unwrap();
        let c_line = stores
            .lines
            .append(root_line, &line("e2e4 e7e5 g1f3"))
            .unwrap();
        let r_line = stores
            .lines
            .append(root_line, &line("e2e4 e7e5 b1c3"))
            .unwrap();
        let extra_cpu = cpu_fixture_observation(&mut stores, state, 40);
        let marker = Observation {
            state,
            line: Some(p_line),
            source: 2,
            epoch: 0,
            value_identity: None,
            checker_identity: None,
            checker_work: None,
            external_report: None,
            model_value_identity: None,
            model_value_input: None,
            cpu_condition: None,
            cpu_pv: None,
            scope: EvidenceScope::Model {
                model: 2,
                encoding: 3,
                input: 4,
            },
            score: RawScore::Unknown,
            budget: 0,
            kind: ObservationKind::Proposal,
            supersedes: None,
            execution: None,
        };
        let marker_id = stores.observations.append(marker.clone()).unwrap();
        let identity = crate::pals::value::ModelValueIdentity {
            semantics: crate::pals::value::MODEL_WDL_VALUE_SEMANTICS.into(),
            model: "fixture-model".into(),
            encoding: "fixture-encoding".into(),
            precision: "fp32".into(),
            model_epoch: [7; 32],
        };
        let mut position = Position::startpos();
        position
            .make_move(BoardMove::from_uci("e2e4").unwrap())
            .unwrap();
        let repaired_state = stores.states.insert(position.snapshot()).unwrap();
        position
            .make_move(BoardMove::from_uci("e7e5").unwrap())
            .unwrap();
        let counter_state = stores.states.insert(position.snapshot()).unwrap();
        let mut repaired = marker.clone();
        repaired.state = repaired_state;
        repaired.line = None;
        repaired.model_value_identity = Some(identity.clone());
        repaired.model_value_input = Some([8; 32]);
        repaired.score = RawScore::ContextWdl {
            win: 0.1,
            draw: 0.2,
            loss: 0.7,
            perspective: Color::Black,
            context_revision: 43,
        };
        let repaired_id = stores.observations.append(repaired).unwrap();
        let mut counter = marker.clone();
        counter.state = counter_state;
        counter.line = None;
        counter.model_value_identity = Some(identity);
        counter.model_value_input = Some([9; 32]);
        counter.score = RawScore::ContextWdl {
            win: 0.2,
            draw: 0.3,
            loss: 0.5,
            perspective: Color::White,
            context_revision: 43,
        };
        let counter_id = stores.observations.append(counter).unwrap();
        let mut conditional = marker.clone();
        conditional.kind = ObservationKind::Refutation;
        conditional.line = Some(c_line);
        conditional.supersedes = Some(marker_id);
        conditional.score = RawScore::ConditionalWdl {
            expectation: 0.2 - 0.5,
            perspective: Color::White,
            repaired: repaired_id,
            counter: counter_id,
            context_revision: 43,
        };
        let conditional_id = stores.observations.append(conditional).unwrap();
        let mut repair = marker.clone();
        repair.kind = ObservationKind::Repair;
        repair.line = Some(r_line);
        repair.supersedes = Some(conditional_id);
        repair.score = RawScore::ConditionalRepairWdl {
            expectation: 0.7 - 0.1,
            perspective: Color::White,
            before: counter_id,
            after: repaired_id,
            context_revision: 43,
        };
        let repair_id = stores.observations.append(repair).unwrap();
        let mut p = record(state, 1, RecordKind::Proposal, "e2e4 e7e5", None);
        p.cpu_observation = Some(extra_cpu);
        let bundle = index.publish(&stores, &p, Some(marker_id)).unwrap();
        let c = record(
            state,
            2,
            RecordKind::Counterexample,
            "e2e4 e7e5 g1f3",
            Some(1),
        );
        index.publish(&stores, &c, Some(conditional_id)).unwrap();
        let mut r = record(state, 3, RecordKind::Repair, "e2e4 e7e5 b1c3", Some(2));
        r.supersedes_revision = Some(2);
        index.publish(&stores, &r, Some(repair_id)).unwrap();
        let mut expected = vec![
            extra_cpu,
            marker_id,
            conditional_id,
            repair_id,
            repaired_id,
            counter_id,
        ];
        for revision in 4..=10 {
            let mut cpu = record(
                state,
                revision,
                RecordKind::CpuVerification,
                "e2e4 e7e5",
                Some(if revision == 4 { 1 } else { revision - 1 }),
            );
            cpu.supersedes_revision = (revision > 4).then_some(revision - 1);
            let id = cpu_fixture_observation(&mut stores, state, revision);
            cpu.cpu_observation = Some(id);
            expected.push(id);
            index.publish(&stores, &cpu, None).unwrap();
        }
        let policy = ReviewSelectionPolicy {
            required_bundle: Some(bundle),
            mode: ReviewAnchorMode::AnchoredCompact,
            ..ReviewSelectionPolicy::default()
        };
        let canceled = AtomicBool::new(false);
        let mut control = cpu7_capture::CaptureControl {
            deadline: std::time::Instant::now() + std::time::Duration::from_secs(10),
            canceled: &canceled,
            limits: cpu7_capture::CaptureLimits {
                max_observations: 32,
                max_line_chunks: 16,
                max_states: 8,
                max_owned_bytes: 1 << 17,
                max_retained_history_bytes: 1 << 20,
            },
        };
        let (_, result) = index
            .select_cohort_cpu7_with_raw_capture(
                &stores,
                state,
                empty_query(),
                policy,
                Some(&control),
            )
            .unwrap();
        let capture = result.unwrap();
        let raw: serde_json::Value = serde_json::from_slice(capture.observations_json()).unwrap();
        let rows = raw.as_array().unwrap();
        let actual: std::collections::BTreeSet<_> = rows
            .iter()
            .map(|row| row["id"].as_u64().unwrap() as usize)
            .collect();
        assert_eq!(actual, expected.iter().map(|id| id.0).collect());
        let marker_wire = rows.iter().find(|row| row["id"] == marker_id.0).unwrap();
        assert_eq!(marker_wire["score"]["kind"], "unknown");
        assert!(marker_wire["model_value_identity"].is_null());
        assert!(marker_wire["model_value_input"].is_null());
        let conditional_wire = rows
            .iter()
            .find(|row| row["id"] == conditional_id.0)
            .unwrap();
        assert_eq!(conditional_wire["score"]["kind"], "conditional_wdl");
        assert_eq!(conditional_wire["score"]["repaired"], repaired_id.0);
        assert_eq!(conditional_wire["score"]["counter"], counter_id.0);
        assert_eq!(conditional_wire["supersedes"], marker_id.0);
        let repair_wire = rows.iter().find(|row| row["id"] == repair_id.0).unwrap();
        assert_eq!(repair_wire["score"]["kind"], "conditional_repair_wdl");
        assert_eq!(repair_wire["score"]["before"], counter_id.0);
        assert_eq!(repair_wire["score"]["after"], repaired_id.0);
        assert_eq!(repair_wire["supersedes"], conditional_id.0);
        assert!(
            rows.iter().find(|row| row["id"] == repaired_id.0).unwrap()["model_value_identity"]
                .is_object()
        );
        assert_eq!(capture.states().len(), 3);
        assert_eq!(capture.lines().len(), 4);
        control.limits.max_observations = 7;
        let (view, limited) = index
            .select_cohort_cpu7_with_raw_capture(
                &stores,
                state,
                empty_query(),
                policy,
                Some(&control),
            )
            .unwrap();
        assert!(view.is_cohort_cpu7());
        assert!(matches!(
            limited,
            Err(cpu7_capture::CaptureUnavailable::Capacity)
        ));
    }

    #[test]
    fn cohort_cpu_prefix_keeps_required_repair_and_excludes_other_bundle() {
        let (mut stores, state) = stores(&Position::startpos());
        let mut index = ReviewContextIndex::default();
        let proposal = record(state, 1, RecordKind::Proposal, "e2e4 e7e5", None);
        let bundle = index.publish(&stores, &proposal, None).unwrap();
        index
            .publish(
                &stores,
                &record(
                    state,
                    2,
                    RecordKind::Counterexample,
                    "e2e4 e7e5 g1f3",
                    Some(1),
                ),
                None,
            )
            .unwrap();
        index
            .publish(
                &stores,
                &record(state, 3, RecordKind::Repair, "e2e4 e7e5 b1c3", Some(2)),
                None,
            )
            .unwrap();
        for revision in 4..=10 {
            let mut cpu = record(
                state,
                revision,
                RecordKind::CpuVerification,
                "e2e4 e7e5",
                Some(revision - 1),
            );
            if revision > 4 {
                cpu.supersedes_revision = Some(revision - 1);
            }
            cpu.cpu_observation = Some(cpu_fixture_observation(&mut stores, state, revision));
            index.publish(&stores, &cpu, None).unwrap();
        }
        index
            .publish(
                &stores,
                &record(state, 11, RecordKind::Proposal, "d2d4", None),
                None,
            )
            .unwrap();
        let policy = ReviewSelectionPolicy {
            required_bundle: Some(bundle),
            mode: ReviewAnchorMode::AnchoredCompact,
            ..ReviewSelectionPolicy::default()
        };
        let view = index
            .select_cohort_cpu7(&stores, state, empty_query(), policy)
            .unwrap();
        assert_eq!(view.records().len(), 10);
        assert_eq!(view.selection_policy().max_records, 10);
        assert_eq!(view.records()[2].kind, RecordKind::Repair);
        assert_eq!(view.records_metadata()[2].parent, Some(1));
        assert!(
            view.records()
                .iter()
                .all(|record| record.line.first() == Some(&BoardMove::from_uci("e2e4").unwrap()))
        );
    }

    #[test]
    fn cohort_cpu_prefix_rejects_kind_without_original_observation() {
        let (stores, state) = stores(&Position::startpos());
        let mut index = ReviewContextIndex::default();
        let proposal = record(state, 1, RecordKind::Proposal, "e2e4", None);
        let bundle = index.publish(&stores, &proposal, None).unwrap();
        for revision in 2..=8 {
            let cpu = record(
                state,
                revision,
                RecordKind::CpuVerification,
                "e2e4",
                Some(revision - 1),
            );
            index.publish(&stores, &cpu, None).unwrap();
        }
        let policy = ReviewSelectionPolicy {
            required_bundle: Some(bundle),
            mode: ReviewAnchorMode::AnchoredCompact,
            ..ReviewSelectionPolicy::default()
        };
        assert!(!index.cohort_cpu7_ready(&stores, policy));
        assert!(matches!(
            index.select_cohort_cpu7(&stores, state, empty_query(), policy),
            Err(ReviewError::ViewCapacity)
        ));
    }

    #[test]
    fn cohort_cpu_prefix_does_not_count_rules_terminal_as_cpu_search() {
        let root = Position::from_fen("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1").unwrap();
        let (mut stores, state) = stores(&root);
        let mut terminal = root;
        terminal.make_uci("f7g7").unwrap();
        let exact = stores.append_rules_terminal(&terminal, 1, 1).unwrap();
        let mut index = ReviewContextIndex::default();
        let bundle = index
            .publish(
                &stores,
                &record(state, 1, RecordKind::Proposal, "f7g7", None),
                None,
            )
            .unwrap();
        let mut first = record(state, 2, RecordKind::CpuVerification, "f7g7", Some(1));
        first.cpu_observation = Some(exact);
        index.publish(&stores, &first, None).unwrap();
        for revision in 3..=8 {
            let mut cpu = record(
                state,
                revision,
                RecordKind::CpuVerification,
                "f7g7",
                Some(revision - 1),
            );
            cpu.cpu_observation = Some(cpu_fixture_observation(&mut stores, state, revision));
            index.publish(&stores, &cpu, None).unwrap();
        }
        let policy = ReviewSelectionPolicy {
            required_bundle: Some(bundle),
            mode: ReviewAnchorMode::AnchoredCompact,
            ..ReviewSelectionPolicy::default()
        };
        assert!(!index.cohort_cpu7_ready(&stores, policy));
        assert!(matches!(
            index.select_cohort_cpu7(&stores, state, empty_query(), policy),
            Err(ReviewError::ViewCapacity)
        ));
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
