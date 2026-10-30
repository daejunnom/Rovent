//! Submitted utility-training input, captured by the original C-preparation
//! actor. This is neither a utility forward nor an R-node readout, action
//! allocation, policy decision, semantic witness or known training target.

use super::retirement::NnuecCacheReleaseAck;
use super::review::{
    BoundInstanceAuthority, BoundNnuecReviewContext, Provenance, RequestIdentity,
    UtilityInputPreparationSource,
};
use super::utility::Reference;
use super::{Model, ReferenceEdge, ReferenceInput};
use crate::cpu_value::CpuValueIdentity;
use crate::cpu_value::nnuec::NnuecError;
use crate::pals::review_context::{ReviewRecordMetadata, ReviewView};
use crate::pals::store::{PalsStores, StateId, StoreReadOwner};
use rz_position::{BoardMove, Position, PositionLimits, PositionSnapshot};
use sha2::{Digest as _, Sha256};
use std::sync::{
    Arc, Weak,
    atomic::{AtomicBool, AtomicU64, Ordering},
};
use std::time::Instant;

pub const INPUT_CAPTURE_DOMAIN: &str = "rz-nnuec-submitted-utility-input/1";
pub const TARGET_CAPTURE_DOMAIN: &str = "rz-nnuec-submitted-utility-target/1";
pub const SUBMISSION_TARGET_CAPTURE_DOMAIN: &str = "rz-nnuec-submitted-utility-submission-target/1";
pub const MAX_INPUT_BYTES: usize = 8 * 1024 * 1024;

/// The original move-only ACK travelled through this Request before its first
/// same-actor C preparation. This historical evidence retains no model/View,
/// has no public constructor/Clone, and says nothing about TT/GPU/P-C caches.
#[derive(Debug)]
pub struct NnuecUtilityCachePrelude {
    release: NnuecCacheReleaseAck,
    request: RequestIdentity,
    instance: BoundInstanceAuthority,
}
impl NnuecUtilityCachePrelude {
    pub fn epoch(&self) -> u64 {
        self.release.epoch()
    }
    /// Compare the original Request and Bound instance, not public hashes.
    /// Historical matching never restores live preparation/action authority.
    pub fn matches_input(&self, input: &NnuecUtilityInputCapture) -> bool {
        self.request.same_request(&input.request)
            && self.instance.matches(&input.instance)
            && self.release.prepares(&input.invalidations, input.epoch)
    }
}

/// Immutable input from exactly one selected Request. The weak original
/// model/View and small request/Bound authorities cannot keep a Store pin or
/// recreate a live preparation. Rules snapshots share their original history.
pub struct NnuecUtilityInputCapture {
    request: RequestIdentity,
    instance: BoundInstanceAuthority,
    store: StoreReadOwner,
    model: Weak<Model>,
    view: Weak<ReviewView>,
    identity: CpuValueIdentity,
    provenance: Provenance,
    target_record: u16,
    original_first_move: BoardMove,
    root: PositionSnapshot,
    metadata: Vec<ReviewRecordMetadata>,
    sources: Arc<[PositionSnapshot]>,
    reference: Reference,
    bytes: Vec<u8>,
    input_sha256: String,
    target_sha256: String,
    submission_target_sha256: String,
    context_sha256: String,
    deadline: Instant,
    cancel: Arc<AtomicBool>,
    stopping: Arc<AtomicBool>,
    invalidations: Arc<AtomicU64>,
    epoch: u64,
    cache_prelude: Option<NnuecUtilityCachePrelude>,
}

pub(crate) fn target_record_ids(
    view: &ReviewView,
    target_record: u16,
) -> Result<Vec<u64>, NnuecError> {
    let metadata = view.records_metadata();
    if metadata.len() > 64 || metadata.len() != view.records().len() {
        return Err(super::invalid("submitted utility selected-record capacity"));
    }
    let target = metadata
        .get(usize::from(target_record))
        .ok_or(super::invalid("submitted utility target record"))?;
    if target.original_first_move.is_none() {
        return Err(super::invalid(
            "submitted utility target has no original first move",
        ));
    }
    let ids: Vec<_> = metadata
        .iter()
        .enumerate()
        .filter_map(|(index, record)| {
            (record.source_state == target.source_state
                && record.original_first_move == target.original_first_move)
                .then_some(index as u64)
        })
        .collect();
    if ids.is_empty() {
        return Err(super::invalid("submitted utility empty target bundle"));
    }
    // A target is the whole exact Rules bundle, not an arbitrary numeric
    // anchor group. An incident relation may not point outside that bundle.
    for (index, record) in metadata.iter().enumerate() {
        for parent in [record.parent, record.supersedes].into_iter().flatten() {
            if usize::from(parent) >= metadata.len()
                || ids.contains(&(index as u64)) != ids.contains(&u64::from(parent))
            {
                return Err(super::invalid("submitted utility target crosses relation"));
            }
        }
    }
    Ok(ids)
}

impl NnuecUtilityInputCapture {
    /// Called only after the same actor has prepared the exact original Bound.
    /// No model board/local/forward method or utility head runs in this path.
    pub(super) fn from_prepared(
        source: UtilityInputPreparationSource<'_>,
    ) -> Result<Self, NnuecError> {
        let (model, bound, store, target_record, cache_prelude) = source.into_parts();
        if !bound.authority_is_live() || bound.provenance().model != *model.identity() {
            return Err(NnuecError::StateMismatch);
        }
        if cache_prelude.as_ref().is_some_and(|release| {
            !release.prepares(bound.actor_invalidations(), bound.preparation_epoch())
                || bound.reused_record_scope()
        }) {
            return Err(NnuecError::StateMismatch);
        }
        model.check_context(bound.context())?;
        let view = bound.view();
        let ids = target_record_ids(view, target_record)?;
        let metadata = view.records_metadata();
        let original_first_move = metadata[usize::from(target_record)]
            .original_first_move
            .ok_or(super::invalid("submitted utility target first move"))?;
        let root = view
            .anchors()
            .first()
            .ok_or(super::invalid("submitted utility root anchor"))?
            .snapshot
            .clone();
        let position = Position::from_snapshot(
            &root,
            PositionLimits {
                max_history_positions: root.known_history_len().max(1),
                ..PositionLimits::default()
            },
        )
        .map_err(|_| NnuecError::StateMismatch)?;
        let reference = Reference {
            schema: super::utility::INPUT_SCHEMA.into(),
            base_input: ReferenceInput {
                schema: super::INPUT_SCHEMA.into(),
                board_features: super::super::rules::reference_board_features(&position),
                side_to_move: super::color_name(position.side_to_move()).into(),
                context_lines: bound.context().reference_lines(),
                context_edges: bound
                    .context()
                    .edge_order
                    .iter()
                    .map(|edge| ReferenceEdge {
                        source: edge.source,
                        target: edge.target,
                        kind: edge.kind as u8,
                    })
                    .collect(),
                local_line: super::super::rules::reference_line_from_rules(&position, &[])?,
            },
            target_ids: ids,
        };
        let bytes =
            serde_json::to_vec(&reference).map_err(|error| NnuecError::Json(error.to_string()))?;
        if bytes.is_empty() || bytes.len() > MAX_INPUT_BYTES {
            return Err(NnuecError::Capacity("submitted utility input bytes"));
        }
        let context_sha256 = bound.numeric_context_sha256().to_owned();
        let mut input = Sha256::new();
        input.update(INPUT_CAPTURE_DOMAIN.as_bytes());
        input.update([0]);
        input.update(
            model
                .identity()
                .weights_sha256
                .as_ref()
                .ok_or(NnuecError::StateMismatch)?
                .as_bytes(),
        );
        input.update([0]);
        input.update((bytes.len() as u64).to_le_bytes());
        input.update(&bytes);
        let input_sha256 = format!("{:x}", input.finalize());
        let mut target = Sha256::new();
        target.update(TARGET_CAPTURE_DOMAIN.as_bytes());
        target.update([0]);
        target.update(context_sha256.as_bytes());
        target.update([0]);
        target.update((reference.target_ids.len() as u32).to_le_bytes());
        for id in &reference.target_ids {
            target.update(id.to_le_bytes());
        }
        let target_sha256 = format!("{:x}", target.finalize());
        let mut submitted = Sha256::new();
        submitted.update(SUBMISSION_TARGET_CAPTURE_DOMAIN.as_bytes());
        submitted.update([0]);
        submitted.update(bound.provenance().view_seal);
        submitted.update(target_sha256.as_bytes());
        let submission_target_sha256 = format!("{:x}", submitted.finalize());
        if !bound.authority_is_live() {
            return Err(NnuecError::StateMismatch);
        }
        let cache_prelude = cache_prelude.map(|release| NnuecUtilityCachePrelude {
            release,
            request: bound.request_identity(),
            instance: bound.instance_authority().clone(),
        });
        Ok(Self {
            request: bound.request_identity(),
            instance: bound.instance_authority().clone(),
            store,
            model: Arc::downgrade(model),
            view: Arc::downgrade(view),
            identity: model.identity().clone(),
            provenance: bound.provenance().clone(),
            target_record,
            original_first_move,
            root,
            metadata: metadata.to_vec(),
            sources: Arc::clone(bound.source_snapshots_arc()),
            reference,
            bytes,
            input_sha256,
            target_sha256,
            submission_target_sha256,
            context_sha256,
            deadline: bound.original_hard_deadline(),
            cancel: Arc::clone(bound.original_cancellation()),
            stopping: Arc::clone(bound.actor_stopping()),
            invalidations: Arc::clone(bound.actor_invalidations()),
            epoch: bound.preparation_epoch(),
            cache_prelude,
        })
    }
    pub fn provenance(&self) -> &Provenance {
        &self.provenance
    }
    pub fn model_identity(&self) -> &CpuValueIdentity {
        &self.identity
    }
    pub fn reference(&self) -> &Reference {
        &self.reference
    }
    pub fn input_bytes(&self) -> &[u8] {
        &self.bytes
    }
    pub fn input_sha256(&self) -> &str {
        &self.input_sha256
    }
    pub fn target_sha256(&self) -> &str {
        &self.target_sha256
    }
    pub fn submission_target_sha256(&self) -> &str {
        &self.submission_target_sha256
    }
    pub fn context_sha256(&self) -> &str {
        &self.context_sha256
    }
    pub fn target_record(&self) -> u16 {
        self.target_record
    }
    pub fn target_record_ids(&self) -> &[u64] {
        &self.reference.target_ids
    }
    pub fn original_first_move(&self) -> BoardMove {
        self.original_first_move
    }
    pub fn root_snapshot(&self) -> &PositionSnapshot {
        &self.root
    }
    pub fn records_metadata(&self) -> &[ReviewRecordMetadata] {
        &self.metadata
    }
    pub fn source_snapshots(&self) -> &[PositionSnapshot] {
        &self.sources
    }
    /// Same original weak Store owner and generation only. No time check,
    /// view reselection, pin creation, or retired/live authority conversion.
    pub fn matches_original_store(&self, stores: &PalsStores) -> bool {
        self.store.matches(stores) && stores.generation() == self.provenance.generation
    }
    pub fn cache_prelude(&self) -> Option<&NnuecUtilityCachePrelude> {
        self.cache_prelude.as_ref()
    }
    pub fn matches_original_model(&self, model: &Arc<Model>) -> bool {
        self.model.ptr_eq(&Arc::downgrade(model)) && self.identity == *model.identity()
    }
    pub(crate) fn matches_request(&self, request: &RequestIdentity) -> bool {
        self.request.same_request(request)
    }
    pub fn matches_bound(&self, bound: &Arc<BoundNnuecReviewContext>) -> bool {
        self.instance.matches(bound.instance_authority())
            && bound.matches_request(&self.request)
            && self.view.ptr_eq(&Arc::downgrade(bound.view()))
    }
    pub fn matches_capture_lineage(&self, lineage: &super::capture::NnuecCaptureLineage) -> bool {
        lineage.matches_original_bound(&self.instance, &self.request)
    }
    /// Physical cache release by this same actor after the original input's
    /// preparation epoch. This does not prove an engine/worker was released;
    /// the action owner must join its actual final installed-owner ACK too.
    pub fn matches_context_cache_release(
        &self,
        release: &super::retirement::NnuecCacheReleaseAck,
    ) -> bool {
        release.clears(&self.invalidations, self.epoch)
    }
    /// Exact original C terminal, not another Request with equal view hashes.
    /// C-only closure must additionally consume allows_c_only_cleanup() and
    /// the same actor's cache release; this matcher alone grants no closure.
    pub fn matches_preparation_terminal(
        &self,
        terminal: &super::review::NnuecPreparationTerminalAck,
    ) -> bool {
        terminal.matches_request(&self.request)
            && terminal.model_identity() == &self.identity
            && terminal.provenance() == &self.provenance
            && terminal.preparation_epoch() == self.epoch
    }
    /// Live source validation only. Historical getters deliberately remain
    /// readable after retirement; neither path grants action/worker rights.
    pub fn validates(
        &self,
        stores: &PalsStores,
        root: StateId,
        generation: u64,
        view: &Arc<ReviewView>,
    ) -> bool {
        self.store.matches(stores)
            && self.model.upgrade().is_some()
            && self.view.ptr_eq(&Arc::downgrade(view))
            && view.validates(stores)
            && self.provenance.root == root
            && self.provenance.generation == generation
            && stores.generation() == generation
            && view.root_state() == root
            && view.generation() == generation
            && view.digest() == self.provenance.full_view_digest
            && view.seal_digest() == self.provenance.view_seal
            && !self.cancel.load(Ordering::Acquire)
            && !self.stopping.load(Ordering::Acquire)
            && Instant::now() < self.deadline
            && self.invalidations.load(Ordering::Acquire) == self.epoch
    }
}
