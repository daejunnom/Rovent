//! Actual, pinned ReviewView -> relational C preparation. The manager admits
//! immutable custody only; Rules replay and NNUEC preparation run on the
//! explicitly created actor. This does not consume a view or select a policy.

use super::{Context, Edge, EdgeKind, LineKind, Model};
use crate::cpu_value::{CpuValueIdentity, nnuec::NnuecError};
use crate::cpu_worker::deferred_value::MaterializationOwner;
use crate::cpu_worker::{
    CpuWorkerError, MaterializationCompletion, MaterializationRequest, MaterializedNnuecEngine,
    NnuecAuthorityRevocation, NnuecEngineRetirementCompletion, NnuecEngineRetirementId,
    NnuecMaterializationReleaseCompletion, NnuecMaterializationReleaseId, RecoveredNnuecEngine,
};
use crate::pals::engine::RecordKind;
use crate::pals::review_context::{ReviewView, checked_rebase_offset};
use crate::pals::store::{PalsStores, StateId, StoreReadOwner};
use rz_position::{MAX_UCI_REPLAY_PLIES, Position, PositionIdentity, PositionSnapshot};
use rz_runtime::CompletionSignal;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::{self, Receiver, SyncSender, TryRecvError, TrySendError};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

#[path = "nnuec_v2_review_retirement.rs"]
mod retirement;
#[path = "nnuec_v2_preparation_terminal.rs"]
mod terminal;
pub use terminal::NnuecPreparationTerminalAck;
#[path = "nnuec_v2_shared_readout.rs"]
mod shared_readout;
pub use shared_readout::SharedNnuecReadout;
use shared_readout::SharedReadoutTarget;

#[cfg(test)]
pub(crate) fn shared_readout_fixture_model() -> Arc<Model> {
    Arc::new(Model::new(super::tests::checkpoint()).unwrap())
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PreparationError {
    StaleView,
    InvalidView(&'static str),
    Cancelled,
    Expired,
    Busy,
    WorkerStopped,
    WorkerPanicked,
    Neural(NnuecError),
}
impl From<NnuecError> for PreparationError {
    fn from(value: NnuecError) -> Self {
        Self::Neural(value)
    }
}
impl std::fmt::Display for PreparationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "NNUEC review preparation: {self:?}")
    }
}
impl std::error::Error for PreparationError {}

/// Identifies the original submission, including its query. The public C
/// record scope is additional evidence, never a replacement for this seal.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Provenance {
    pub root: StateId,
    pub generation: u64,
    pub sequence: u64,
    pub full_view_digest: [u8; 32],
    pub view_seal: [u8; 32],
    pub record_scope_digest: [u8; 32],
    pub policy_digest: [u8; 32],
    pub model: CpuValueIdentity,
}

/// Fields cannot be forged by callers. The view's actual hot pin remains
/// alive while queued, preparing, cached, or held by an accepted consumer.
#[derive(Clone, Debug)]
pub(crate) struct RequestIdentity(Arc<()>);
impl RequestIdentity {
    pub(crate) fn same_request(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

pub struct Request {
    identity: RequestIdentity,
    store_owner: StoreReadOwner,
    utility_target: Option<u16>,
    utility_cache_prelude: Option<super::retirement::NnuecCacheReleaseAck>,
    shared_readout: Option<SharedReadoutTarget>,
    view: Arc<ReviewView>,
    sources: Arc<[PositionSnapshot]>,
    provenance: Provenance,
    deadline: Instant,
    cancel: Arc<AtomicBool>,
}
impl Request {
    /// Bounded custody capture, with no encoder/forward/Rules replay. Source
    /// snapshots share immutable history rather than copying an actual game.
    pub fn admit(
        model: &Model,
        view: Arc<ReviewView>,
        stores: &PalsStores,
        expected_root: StateId,
        expected_generation: u64,
        deadline: Instant,
        cancel: Arc<AtomicBool>,
    ) -> Result<Self, PreparationError> {
        check_control(deadline, &cancel)?;
        if !view.validates(stores)
            || view.root_state() != expected_root
            || view.generation() != expected_generation
            || stores.generation() != expected_generation
        {
            return Err(PreparationError::StaleView);
        }
        if view.records().len() > 64
            || view.records().len() > model.base_checkpoint().config.max_lines
            || view.records_metadata().len() != view.records().len()
            || view.source_snapshots().len() != view.records().len()
            || view.anchors().is_empty()
            || view.anchors().len() > 4
        {
            return Err(PreparationError::InvalidView(
                "selected view shape/capacity",
            ));
        }
        let root = stores
            .states
            .get(expected_root)
            .map_err(|_| PreparationError::StaleView)?;
        if view.anchors()[0].state != expected_root || !root.same_state(&view.anchors()[0].snapshot)
        {
            return Err(PreparationError::StaleView);
        }
        // The selected view already retained these actual bundle sources in
        // row order. Share that exact immutable array, rather than scanning
        // the Store again or making another collection of history owners.
        let sources = Arc::clone(view.source_snapshots());
        let provenance = Provenance {
            root: expected_root,
            generation: expected_generation,
            sequence: view.sequence(),
            full_view_digest: view.digest(),
            view_seal: view.seal_digest(),
            record_scope_digest: view.record_scope_digest(),
            policy_digest: view.policy_digest(),
            model: model.identity().clone(),
        };
        Ok(Self {
            identity: RequestIdentity(Arc::new(())),
            store_owner: stores.read_owner(),
            utility_target: None,
            utility_cache_prelude: None,
            shared_readout: None,
            view,
            sources,
            provenance,
            deadline,
            cancel,
        })
    }
    pub fn provenance(&self) -> &Provenance {
        &self.provenance
    }
    pub(crate) fn identity(&self) -> RequestIdentity {
        self.identity.clone()
    }
    pub(crate) fn matches_view(&self, view: &Arc<ReviewView>) -> bool {
        Arc::ptr_eq(&self.view, view)
    }
    /// Select one original whole bundle before this unique Request reaches
    /// the actor. Failure returns that same Request with its old choice intact.
    #[allow(clippy::result_large_err)]
    pub fn with_utility_input_capture(
        mut self,
        target_record: u16,
    ) -> Result<Self, (PreparationError, Self)> {
        let checked = (|| {
            if self.utility_target.is_some() || self.shared_readout.is_some() {
                return Err(PreparationError::InvalidView(
                    "utility input capture already selected",
                ));
            }
            check_control(self.deadline, &self.cancel)?;
            super::utility_capture::target_record_ids(&self.view, target_record)?;
            Ok(())
        })();
        if let Err(error) = checked {
            return Err((error, self));
        }
        self.utility_target = Some(target_record);
        Ok(self)
    }
    /// Move the actual actor ACK into this exact selected Request before
    /// submission. This alone grants no cache classification: only the actor
    /// can verify its authority, epoch and absence of an intervening prepare.
    /// Failure preserves the same Request and the unused, move-only ACK.
    #[allow(clippy::result_large_err)]
    pub fn with_utility_cache_prelude(
        mut self,
        release: super::retirement::NnuecCacheReleaseAck,
    ) -> Result<
        Self,
        (
            PreparationError,
            Self,
            super::retirement::NnuecCacheReleaseAck,
        ),
    > {
        let checked = (|| {
            check_control(self.deadline, &self.cancel)?;
            if self.utility_target.is_none() || self.utility_cache_prelude.is_some() {
                return Err(PreparationError::InvalidView(
                    "utility cache prelude requires one selected original Request",
                ));
            }
            Ok(())
        })();
        if let Err(error) = checked {
            return Err((error, self, release));
        }
        self.utility_cache_prelude = Some(release);
        Ok(self)
    }
}

#[derive(Clone, Debug)]
pub(crate) struct BoundInstanceAuthority(Arc<BoundInstanceState>);
#[derive(Debug, Default)]
struct BoundInstanceState {
    materialized: AtomicBool,
}
impl BoundInstanceAuthority {
    pub(crate) fn matches(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
    pub(super) fn was_materialized(&self) -> bool {
        self.0.materialized.load(Ordering::Acquire)
    }
}

/// Only this actor factory can issue the transport, after consuming an
/// explicit selection on the original Request. Metadata alone cannot mint it.
pub(super) struct UtilityInputPreparationSource<'a> {
    model: &'a Arc<Model>,
    bound: &'a BoundNnuecReviewContext,
    store: StoreReadOwner,
    target_record: u16,
    cache_prelude: Option<super::retirement::NnuecCacheReleaseAck>,
}
impl<'a> UtilityInputPreparationSource<'a> {
    pub(super) fn into_parts(
        self,
    ) -> (
        &'a Arc<Model>,
        &'a BoundNnuecReviewContext,
        StoreReadOwner,
        u16,
        Option<super::retirement::NnuecCacheReleaseAck>,
    ) {
        (
            self.model,
            self.bound,
            self.store,
            self.target_record,
            self.cache_prelude,
        )
    }
}

pub struct BoundNnuecReviewContext {
    instance: BoundInstanceAuthority,
    request: RequestIdentity,
    utility_input: Option<Arc<super::utility_capture::NnuecUtilityInputCapture>>,
    shared_readout: Option<SharedNnuecReadout>,
    context: Arc<Context>,
    numeric_context_sha256: String,
    view: Arc<ReviewView>,
    source_snapshots: Arc<[PositionSnapshot]>,
    root: PositionIdentity,
    provenance: Provenance,
    reused_record_scope: bool,
    deadline: Instant,
    cancel: Arc<AtomicBool>,
    invalidations: Arc<AtomicU64>,
    preparation_epoch: u64,
    stopping: Arc<AtomicBool>,
}
impl BoundNnuecReviewContext {
    pub fn utility_input_capture(
        &self,
    ) -> Option<&Arc<super::utility_capture::NnuecUtilityInputCapture>> {
        self.utility_input.as_ref()
    }
    pub(super) fn source_snapshots_arc(&self) -> &Arc<[PositionSnapshot]> {
        &self.source_snapshots
    }
    pub(super) fn original_hard_deadline(&self) -> Instant {
        self.deadline
    }
    pub(super) fn original_cancellation(&self) -> &Arc<AtomicBool> {
        &self.cancel
    }
    pub(super) fn actor_stopping(&self) -> &Arc<AtomicBool> {
        &self.stopping
    }
    pub(super) fn actor_invalidations(&self) -> &Arc<AtomicU64> {
        &self.invalidations
    }
    pub(crate) fn matches_request(&self, request: &RequestIdentity) -> bool {
        self.request.same_request(request)
    }
    pub(crate) fn request_identity(&self) -> RequestIdentity {
        self.request.clone()
    }
    pub(crate) fn instance_authority(&self) -> &BoundInstanceAuthority {
        &self.instance
    }
    /// Record an actual typed TT/evaluator construction. This is monotonic
    /// historical evidence, not permission to construct or resume anything.
    pub(crate) fn record_materialized(&self) {
        self.instance.0.materialized.store(true, Ordering::Release);
    }
    /// Original source snapshots in selected-record/metadata order. History
    /// is shared from admission, never inferred from a rebased numeric anchor.
    pub fn context_source_snapshots(&self) -> &[PositionSnapshot] {
        &self.source_snapshots
    }
    pub(crate) fn preparation_epoch(&self) -> u64 {
        self.preparation_epoch
    }
    pub(crate) fn matches_preparation_actor(
        &self,
        stopping: &Arc<AtomicBool>,
        invalidations: &Arc<AtomicU64>,
    ) -> bool {
        Arc::ptr_eq(&self.stopping, stopping) && Arc::ptr_eq(&self.invalidations, invalidations)
    }
    pub fn context(&self) -> &Arc<Context> {
        &self.context
    }
    /// Computed by the preparation actor, never by the manager's dispatch
    /// path. This numerical seal is separate from full Rules/view provenance.
    pub fn numeric_context_sha256(&self) -> &str {
        &self.numeric_context_sha256
    }
    pub fn view(&self) -> &Arc<ReviewView> {
        &self.view
    }
    pub fn provenance(&self) -> &Provenance {
        &self.provenance
    }
    pub fn reused_record_scope(&self) -> bool {
        self.reused_record_scope
    }
    /// Worker-side authority check without borrowing the single-writer store.
    /// Store/root/original-ticket admission remains required in validates().
    pub fn authority_is_live(&self) -> bool {
        self.authority_revocation().is_none()
    }
    /// Original control only; never infer a stop from a readback, foreign
    /// actor, allocation failure or caller-provided error string.
    pub(crate) fn authority_revocation(&self) -> Option<NnuecAuthorityRevocation> {
        // Observe the original soft control, then give any observed hard
        // revocation priority. Cancellation/expiry cannot hide a stopped
        // actor or an invalidated immutable context. These are observations,
        // not a new deadline, Bound, actor or permission to resume.
        let cancelled = self.cancel.load(Ordering::Acquire);
        let expired = Instant::now() >= self.deadline;
        if self.stopping.load(Ordering::Acquire) {
            Some(NnuecAuthorityRevocation::ActorStopped)
        } else if self.invalidations.load(Ordering::Acquire) != self.preparation_epoch {
            Some(NnuecAuthorityRevocation::Invalidated)
        } else if cancelled {
            Some(NnuecAuthorityRevocation::Cancelled)
        } else if expired {
            Some(NnuecAuthorityRevocation::Expired)
        } else {
            None
        }
    }
    /// Acceptance must use the original ticket seal, never the latest view.
    /// Preparation does not call ReviewContextIndex::consume.
    pub fn validates(
        &self,
        stores: &PalsStores,
        expected_root: StateId,
        expected_generation: u64,
        original_view_seal: [u8; 32],
    ) -> bool {
        self.view.validates(stores)
            && self.authority_is_live()
            && self.provenance.root == expected_root
            && self.provenance.generation == expected_generation
            && self.provenance.view_seal == original_view_seal
            && self.view.seal_digest() == original_view_seal
            && stores
                .states
                .get(expected_root)
                .is_ok_and(|root| root.position_identity() == self.root)
    }
}

struct CachedScope {
    view: Arc<ReviewView>,
    digest: [u8; 32],
    context: Arc<Context>,
    numeric_context_sha256: String,
}
/// Owned exclusively by the actor below, with one bounded cached scope.
struct Factory {
    model: Arc<Model>,
    cached: Option<CachedScope>,
    stopping: Arc<AtomicBool>,
    invalidations: Arc<AtomicU64>,
    invalidation_ack: Arc<AtomicU64>,
    active_epoch: u64,
}
impl Factory {
    fn sync_cache_epoch(&mut self) {
        let current = self.invalidations.load(Ordering::Acquire);
        if self.active_epoch != current {
            self.cached = None;
            self.active_epoch = current;
        }
        self.invalidation_ack.store(current, Ordering::Release);
    }
    fn control(&self, request: &Request) -> Result<(), PreparationError> {
        if self.stopping.load(Ordering::Acquire)
            || self.active_epoch != self.invalidations.load(Ordering::Acquire)
        {
            return Err(PreparationError::Cancelled);
        }
        check_control(request.deadline, &request.cancel)
    }
    #[cfg(test)]
    fn prepare(&mut self, request: Request) -> Result<BoundNnuecReviewContext, PreparationError> {
        let epoch = self.invalidations.load(Ordering::Acquire);
        self.prepare_at(request, epoch)
    }
    fn prepare_at(
        &mut self,
        request: Request,
        epoch: u64,
    ) -> Result<BoundNnuecReviewContext, PreparationError> {
        self.sync_cache_epoch();
        if self.active_epoch != epoch {
            return Err(PreparationError::Cancelled);
        }
        self.control(&request)?;
        if &request.provenance.model != self.model.identity() {
            return Err(PreparationError::StaleView);
        }
        if let Some(release) = &request.utility_cache_prelude {
            if request.utility_target.is_none()
                || !release.prepares(&self.invalidations, epoch)
                || self.cached.is_some()
            {
                return Err(PreparationError::InvalidView(
                    "utility cache prelude is not the first same-actor epoch preparation",
                ));
            }
        }
        let view = &request.view;
        let cached = self.cached.as_ref().filter(|cached| {
            cached.digest == request.provenance.record_scope_digest
                && same_numeric_scope(&cached.view, view)
        });
        let reused_record_scope = cached.is_some();
        let context = if let Some(cached) = cached {
            Arc::clone(&cached.context)
        } else {
            let mut context = self.model.empty_context();
            for (index, ((record, metadata), source)) in view
                .records()
                .iter()
                .zip(view.records_metadata())
                .zip(request.sources.iter())
                .enumerate()
            {
                self.control(&request)?;
                let anchor = view
                    .anchors()
                    .get(usize::from(metadata.anchor))
                    .ok_or(PreparationError::InvalidView("record anchor"))?;
                if record.origin_state != anchor.state
                    || record.revision != metadata.source_revision
                {
                    return Err(PreparationError::InvalidView("record source provenance"));
                }
                let offset = usize::from(metadata.source_line_offset);
                if offset
                    .checked_add(record.line.len())
                    .is_none_or(|length| length > 256)
                {
                    return Err(PreparationError::InvalidView("original line capacity"));
                }
                let mut original = Vec::with_capacity(offset + record.line.len());
                if offset != 0 {
                    let actual = anchor
                        .snapshot
                        .uci_replay(MAX_UCI_REPLAY_PLIES)
                        .map_err(|_| PreparationError::InvalidView("rebase actual history"))?;
                    let start = actual
                        .moves
                        .len()
                        .checked_sub(offset)
                        .ok_or(PreparationError::InvalidView("rebase history offset"))?;
                    original.extend_from_slice(&actual.moves[start..]);
                }
                original.extend_from_slice(&record.line);
                if original.first().copied() != metadata.original_first_move
                    || checked_rebase_offset(source, &anchor.snapshot, &original)
                        .map_err(|_| PreparationError::InvalidView("Rules rebase"))?
                        != Some(offset)
                {
                    return Err(PreparationError::InvalidView(
                        "original bundle/rebase mismatch",
                    ));
                }
                let anchor_position = Position::from_snapshot(&anchor.snapshot, Default::default())
                    .map_err(|e| PreparationError::Neural(NnuecError::Rules(e.to_string())))?;
                let kind = match record.kind {
                    RecordKind::Proposal => LineKind::Proposal,
                    RecordKind::Counterexample => LineKind::Counterexample,
                    RecordKind::Repair => LineKind::Repair,
                    RecordKind::CpuVerification => LineKind::CpuVerification,
                };
                let line = self.model.encode_review_line(
                    &anchor_position,
                    &record.line,
                    kind,
                    metadata.original_first_move,
                )?;
                context = self.model.with_line(&context, index as u64, line)?;
            }
            for (index, metadata) in view.records_metadata().iter().enumerate() {
                self.control(&request)?;
                let record = &view.records()[index];
                for (source, revision, kind) in [
                    (metadata.parent, record.parent_revision, EdgeKind::Parent),
                    (
                        metadata.supersedes,
                        record.supersedes_revision,
                        EdgeKind::Supersedes,
                    ),
                ] {
                    match (source, revision) {
                        (None, None) => {}
                        (Some(source), Some(revision))
                            if usize::from(source) < index
                                && view.records()[usize::from(source)].revision == revision =>
                        {
                            context = self.model.with_edge(
                                &context,
                                Edge {
                                    source: u64::from(source),
                                    target: index as u64,
                                    kind,
                                },
                            )?;
                        }
                        _ => {
                            return Err(PreparationError::InvalidView("local relation provenance"));
                        }
                    }
                }
            }
            context
        };
        let numeric_context_sha256 = if let Some(cached) = cached {
            cached.numeric_context_sha256.clone()
        } else {
            context.sha256()?
        };
        self.control(&request)?;
        self.cached = Some(CachedScope {
            view: Arc::clone(view),
            digest: request.provenance.record_scope_digest,
            context: Arc::clone(&context),
            numeric_context_sha256: numeric_context_sha256.clone(),
        });
        let shared_readout = request.shared_readout;
        let utility_target = request.utility_target;
        let cache_prelude = request.utility_cache_prelude;
        let store_owner = request.store_owner;
        let mut bound = BoundNnuecReviewContext {
            instance: BoundInstanceAuthority(Arc::new(BoundInstanceState::default())),
            request: request.identity,
            utility_input: None,
            shared_readout: None,
            root: view.anchors()[0].snapshot.position_identity(),
            context,
            numeric_context_sha256,
            view: request.view,
            source_snapshots: request.sources,
            provenance: request.provenance,
            reused_record_scope,
            deadline: request.deadline,
            cancel: request.cancel,
            invalidations: Arc::clone(&self.invalidations),
            preparation_epoch: epoch,
            stopping: Arc::clone(&self.stopping),
        };
        if let Some(target) = shared_readout {
            bound.shared_readout = Some(self.shared_readout(&bound, target)?);
        }
        if let Some(target_record) = utility_target {
            let captured = super::utility_capture::NnuecUtilityInputCapture::from_prepared(
                UtilityInputPreparationSource {
                    model: &self.model,
                    bound: &bound,
                    store: store_owner,
                    target_record,
                    cache_prelude,
                },
            )
            .map_err(|error| match bound.authority_revocation() {
                Some(NnuecAuthorityRevocation::Cancelled) => PreparationError::Cancelled,
                Some(NnuecAuthorityRevocation::Expired) => PreparationError::Expired,
                Some(NnuecAuthorityRevocation::ActorStopped) => PreparationError::WorkerStopped,
                Some(NnuecAuthorityRevocation::Invalidated) => PreparationError::StaleView,
                None => PreparationError::Neural(error),
            })?;
            bound.utility_input = Some(Arc::new(captured));
        }
        Ok(bound)
    }
}
fn same_numeric_scope(first: &ReviewView, second: &ReviewView) -> bool {
    first.root_state() == second.root_state()
        && first.generation() == second.generation()
        && first.records_metadata() == second.records_metadata()
        && first.anchors().len() == second.anchors().len()
        && first
            .anchors()
            .iter()
            .zip(second.anchors())
            .all(|(a, b)| a.state == b.state && a.snapshot.same_state(&b.snapshot))
        && first.records().len() == second.records().len()
        && first.records().iter().zip(second.records()).all(|(a, b)| {
            a.kind == b.kind
                && a.line == b.line
                && a.revision == b.revision
                && a.parent_revision == b.parent_revision
                && a.supersedes_revision == b.supersedes_revision
                && a.origin_state == b.origin_state
        })
}
fn check_control(deadline: Instant, cancel: &AtomicBool) -> Result<(), PreparationError> {
    if cancel.load(Ordering::Acquire) {
        return Err(PreparationError::Cancelled);
    }
    if Instant::now() >= deadline {
        return Err(PreparationError::Expired);
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PreparationStatus {
    Completed,
    Failed,
    Cancelled,
    Expired,
}

/// Actual owner durations for the same Completion provenance. This is not a
/// role ticket, neural-input count, node charge, cache authority, or acceptance
/// witness. No terminal after an admitted request remains an unknown outcome;
/// a missing observation must never be converted into zero service/success.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PreparationObservation {
    queue_wait: Duration,
    actor_dispatch: Duration,
    service: Duration,
    status: PreparationStatus,
    reused_record_scope: Option<bool>,
}
impl PreparationObservation {
    pub fn queue_wait(&self) -> Duration {
        self.queue_wait
    }
    /// Actor work between dequeue and this request's service start, including
    /// observing an independently requested cache epoch. It is not queue wait.
    pub fn actor_dispatch(&self) -> Duration {
        self.actor_dispatch
    }
    pub fn service(&self) -> Duration {
        self.service
    }
    pub fn status(&self) -> PreparationStatus {
        self.status
    }
    /// Only a successfully prepared result can report record-scope reuse.
    /// This never means a complete query/input cache hit or zero work.
    pub fn reused_record_scope(&self) -> Option<bool> {
        self.reused_record_scope
    }
}

pub struct Completion {
    pub provenance: Provenance,
    pub result: Result<BoundNnuecReviewContext, PreparationError>,
    pub observation: PreparationObservation,
    pub terminal_ack: NnuecPreparationTerminalAck,
}
enum Command {
    Prepare(Request, u64, Instant),
    Materialize(MaterializationRequest),
    RetireEngine(NnuecEngineRetirementId, RecoveredNnuecEngine),
    ReleaseMaterialized(NnuecMaterializationReleaseId, MaterializedNnuecEngine),
    Wake,
}
/// One explicitly accounted preparation worker; no default background thread.
/// At most one queued/running/uncollected completion exists. The result channel
/// is logically bounded by that gate, and terminal results are never coalesced.
pub struct PreparationActor {
    original_model: std::sync::Weak<Model>,
    sender: Option<SyncSender<Command>>,
    receiver: Receiver<Completion>,
    materialized: Receiver<MaterializationCompletion>,
    retired: Receiver<NnuecEngineRetirementCompletion>,
    released_materialized: Receiver<NnuecMaterializationReleaseCompletion>,
    retirement_authority: Arc<()>,
    retirement_sequence: AtomicU64,
    outstanding: Arc<AtomicUsize>,
    stopping: Arc<AtomicBool>,
    invalidations: Arc<AtomicU64>,
    invalidation_ack: Arc<AtomicU64>,
    worker: Option<JoinHandle<()>>,
}
impl PreparationActor {
    pub fn spawn(model: Arc<Model>) -> Result<Self, PreparationError> {
        Self::spawn_notified(model, CompletionSignal::default())
    }
    /// Same explicitly counted actor for C preparation and initial TT
    /// materialization. No relay thread or second preparation owner is added.
    pub fn spawn_notified(
        model: Arc<Model>,
        signal: CompletionSignal,
    ) -> Result<Self, PreparationError> {
        let original_model = Arc::downgrade(&model);
        let (sender, commands) = mpsc::sync_channel::<Command>(1);
        let (results, receiver) = mpsc::channel();
        let (materializations, materialized) = mpsc::channel();
        let (retirements, retired) = mpsc::channel();
        let (materialization_releases, released_materialized) = mpsc::channel();
        let outstanding = Arc::new(AtomicUsize::new(0));
        let worker_outstanding = Arc::clone(&outstanding);
        let stopping = Arc::new(AtomicBool::new(false));
        let stop = Arc::clone(&stopping);
        let invalidations = Arc::new(AtomicU64::new(0));
        let invalidation_ack = Arc::new(AtomicU64::new(0));
        let worker_invalidations = Arc::clone(&invalidations);
        let worker_ack = Arc::clone(&invalidation_ack);
        let worker = thread::Builder::new()
            .name("nnuec-review-prepare".into())
            .spawn(move || {
                let mut factory = Factory {
                    model,
                    cached: None,
                    stopping: Arc::clone(&stop),
                    invalidations: worker_invalidations,
                    invalidation_ack: worker_ack,
                    active_epoch: 0,
                };
                // Holds at most one unclaimed engine. Destruction/TT release
                // stays on this actor even when a manager drops its terminal.
                let mut materialization_owner = None;
                while let Ok(command) = commands.recv() {
                    let dequeued_at = Instant::now();
                    factory.sync_cache_epoch();
                    match command {
                        Command::Wake => {}
                        Command::Prepare(request, epoch, enqueued_at) => {
                            let started_at = Instant::now();
                            let original_view = Arc::downgrade(&request.view);
                            let original_request = request.identity();
                            let provenance = request.provenance.clone();
                            let result = if stop.load(Ordering::Acquire) {
                                Err(PreparationError::Cancelled)
                            } else {
                                factory.prepare_at(request, epoch)
                            };
                            factory.sync_cache_epoch();
                            let finished_at = Instant::now();
                            let status = match &result {
                                Ok(_) => PreparationStatus::Completed,
                                Err(PreparationError::Cancelled) => PreparationStatus::Cancelled,
                                Err(PreparationError::Expired) => PreparationStatus::Expired,
                                Err(_) => PreparationStatus::Failed,
                            };
                            let observation = PreparationObservation {
                                queue_wait: dequeued_at.duration_since(enqueued_at),
                                actor_dispatch: started_at.duration_since(dequeued_at),
                                service: finished_at.duration_since(started_at),
                                status,
                                reused_record_scope: result
                                    .as_ref()
                                    .ok()
                                    .map(|bound| bound.reused_record_scope()),
                            };
                            let terminal_ack = NnuecPreparationTerminalAck::new(
                                provenance.clone(),
                                original_request,
                                factory.model.identity().clone(),
                                Arc::downgrade(&factory.model),
                                original_view,
                                status,
                                result
                                    .as_ref()
                                    .ok()
                                    .map(|bound| bound.instance_authority().clone()),
                                Arc::clone(&factory.invalidations),
                                epoch,
                            );
                            if results
                                .send(Completion {
                                    provenance,
                                    result,
                                    observation,
                                    terminal_ack,
                                })
                                .is_err()
                            {
                                break;
                            }
                        }
                        Command::Materialize(request) => {
                            let provenance = request.provenance().clone();
                            // Equal model bytes alone do not grant another
                            // actor's cancellation/epoch authority. Preserve
                            // the original Bound, never the latest cache view.
                            let same_actor = request.model_matches(&factory.model)
                                && Arc::ptr_eq(&request.bound().stopping, &factory.stopping)
                                && Arc::ptr_eq(
                                    &request.bound().invalidations,
                                    &factory.invalidations,
                                );
                            let result = if !same_actor {
                                Err((
                                    request,
                                    CpuWorkerError::Invalid(
                                        "foreign or revoked NNUEC preparation actor",
                                    ),
                                ))
                            } else if let Some(reason) = request.bound().authority_revocation() {
                                Err((request, CpuWorkerError::NnuecAuthorityRevoked(reason)))
                            } else {
                                request.materialize()
                            };
                            let result = result.map(|engine| {
                                let (owner, handoff) = MaterializationOwner::handoff(
                                    engine,
                                    Arc::clone(&worker_outstanding),
                                );
                                materialization_owner = Some(owner);
                                handoff
                            });
                            factory.sync_cache_epoch();
                            if materializations
                                .send(MaterializationCompletion { provenance, result })
                                .is_err()
                            {
                                break;
                            }
                        }
                        Command::RetireEngine(id, owner) => {
                            // Cancellation/expiry revoke search authority but
                            // do not prevent same-actor physical cleanup.
                            let result = owner.release_on_actor(
                                &factory.model,
                                &factory.stopping,
                                &factory.invalidations,
                            );
                            factory.sync_cache_epoch();
                            if retirements
                                .send(NnuecEngineRetirementCompletion::new(id, result))
                                .is_err()
                            {
                                break;
                            }
                        }
                        Command::ReleaseMaterialized(id, engine) => {
                            let result = engine.release_on_actor(
                                &factory.model,
                                &factory.stopping,
                                &factory.invalidations,
                            );
                            factory.sync_cache_epoch();
                            if materialization_releases
                                .send(NnuecMaterializationReleaseCompletion::new(id, result))
                                .is_err()
                            {
                                break;
                            }
                        }
                    }
                    signal.notify();
                }
                // Explicit owner drop precedes thread-exit/join receipt.
                drop(materialization_owner);
            })
            .map_err(|_| PreparationError::WorkerStopped)?;
        Ok(Self {
            original_model,
            sender: Some(sender),
            receiver,
            materialized,
            retired,
            released_materialized,
            retirement_authority: Arc::new(()),
            retirement_sequence: AtomicU64::new(0),
            outstanding,
            stopping,
            invalidations,
            invalidation_ack,
            worker: Some(worker),
        })
    }
    /// Failure returns ownership of the request and its pin to the caller.
    // Full/closed queues return that original owner without a new allocation.
    #[allow(clippy::result_large_err)]
    pub fn try_submit(&self, request: Request) -> Result<(), (PreparationError, Request)> {
        if self.stopping.load(Ordering::Acquire) {
            return Err((PreparationError::WorkerStopped, request));
        }
        if self
            .outstanding
            .compare_exchange(0, 1, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return Err((PreparationError::Busy, request));
        }
        match self
            .sender
            .as_ref()
            .expect("live actor sender")
            .try_send(Command::Prepare(
                request,
                self.invalidations.load(Ordering::Acquire),
                Instant::now(),
            )) {
            Ok(()) => Ok(()),
            Err(error) => {
                self.outstanding.store(0, Ordering::Release);
                match error {
                    TrySendError::Full(Command::Prepare(request, _, _)) => {
                        Err((PreparationError::Busy, request))
                    }
                    TrySendError::Disconnected(Command::Prepare(request, _, _)) => {
                        Err((PreparationError::WorkerStopped, request))
                    }
                    _ => unreachable!("only a prepare command was submitted"),
                }
            }
        }
    }
    pub fn try_complete(&self) -> Result<Option<Completion>, PreparationError> {
        match self.receiver.try_recv() {
            Ok(completion) => {
                self.outstanding.store(0, Ordering::Release);
                Ok(Some(completion))
            }
            Err(TryRecvError::Empty) => Ok(None),
            Err(TryRecvError::Disconnected) => Err(PreparationError::WorkerStopped),
        }
    }
    /// Shares the single outstanding seat with neural preparation. Failure
    /// retains the exact source/Bound request. A success terminal retains its
    /// seat until the unique engine handoff is claimed, not merely polled.
    #[allow(clippy::result_large_err)]
    pub fn try_materialize(
        &self,
        request: MaterializationRequest,
    ) -> Result<(), (PreparationError, MaterializationRequest)> {
        if self.stopping.load(Ordering::Acquire) {
            return Err((PreparationError::WorkerStopped, request));
        }
        if self
            .outstanding
            .compare_exchange(0, 2, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return Err((PreparationError::Busy, request));
        }
        match self
            .sender
            .as_ref()
            .expect("live actor sender")
            .try_send(Command::Materialize(request))
        {
            Ok(()) => Ok(()),
            Err(error) => {
                self.outstanding.store(0, Ordering::Release);
                match error {
                    TrySendError::Full(Command::Materialize(request)) => {
                        Err((PreparationError::Busy, request))
                    }
                    TrySendError::Disconnected(Command::Materialize(request)) => {
                        Err((PreparationError::WorkerStopped, request))
                    }
                    _ => unreachable!("only a materialize command was submitted"),
                }
            }
        }
    }
    pub fn try_materialized(&self) -> Result<Option<MaterializationCompletion>, PreparationError> {
        match self.materialized.try_recv() {
            Ok(completion) => {
                if completion.result.is_err() {
                    self.outstanding.store(0, Ordering::Release);
                }
                Ok(Some(completion))
            }
            Err(TryRecvError::Empty) => Ok(None),
            Err(TryRecvError::Disconnected) => Err(PreparationError::WorkerStopped),
        }
    }
    pub fn shutdown(mut self) -> Result<(), PreparationError> {
        self.stop_and_join()
    }
    /// Nonblocking closure for the manager. Existing Bound authority is
    /// revoked immediately; the actor still owns in-progress preparation and
    /// any unclaimed engine until its actual thread exit is joined.
    pub fn request_shutdown(&mut self) {
        self.stopping.store(true, Ordering::Release);
        self.sender.take();
    }
    /// Join only an already finished preparation owner. This is separate from
    /// cache-release acknowledgement and from every installed CPU owner join.
    pub fn try_join(&mut self) -> Result<Option<()>, PreparationError> {
        let worker = self
            .worker
            .as_ref()
            .ok_or(PreparationError::WorkerStopped)?;
        if !worker.is_finished() {
            return Ok(None);
        }
        let worker = self
            .worker
            .take()
            .expect("finished preparation owner retained");
        worker
            .join()
            .map_err(|_| PreparationError::WorkerPanicked)?;
        Ok(Some(()))
    }
    /// Independent atomic control cannot be blocked by the ordinary request
    /// slot. It revokes old Bound acceptance immediately; the worker publishes
    /// a cache-release acknowledgement after observing it. Pending terminals
    /// remain available for collection and retain their original provenance.
    pub fn invalidate_cache(&self) -> Result<u64, PreparationError> {
        if self.stopping.load(Ordering::Acquire) || self.sender.is_none() {
            return Err(PreparationError::WorkerStopped);
        }
        let previous = self
            .invalidations
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |epoch| {
                epoch.checked_add(1)
            })
            .map_err(|_| PreparationError::InvalidView("preparation epoch exhausted"))?;
        match self
            .sender
            .as_ref()
            .expect("live actor sender")
            .try_send(Command::Wake)
        {
            Ok(()) | Err(TrySendError::Full(_)) => Ok(previous + 1),
            Err(TrySendError::Disconnected(_)) => Err(PreparationError::WorkerStopped),
        }
    }
    pub fn cache_invalidated_through(&self) -> u64 {
        self.invalidation_ack.load(Ordering::Acquire)
    }
    fn stop_and_join(&mut self) -> Result<(), PreparationError> {
        self.request_shutdown();
        if self
            .worker
            .take()
            .is_some_and(|worker| worker.join().is_err())
        {
            return Err(PreparationError::WorkerPanicked);
        }
        Ok(())
    }
}
impl Drop for PreparationActor {
    fn drop(&mut self) {
        let _ = self.stop_and_join();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pals::engine::{RoleQueryPurpose, RoleRecord};
    use crate::pals::review_context::{
        ReviewContextIndex, ReviewQuerySequences, ReviewSelectionPolicy,
    };
    use crate::pals::store::StoreLimits;
    use rz_position::{BoardMove, Color};
    use std::time::Duration;

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
            line: text
                .split_whitespace()
                .map(|s| s.parse().unwrap())
                .collect(),
            value: None,
            completed_depth: 0,
            score_scope: None,
            cpu_observation: None,
            perspective: Color::White,
            critical: true,
        }
    }
    fn query(purpose: RoleQueryPurpose) -> ReviewQuerySequences<'static> {
        ReviewQuerySequences {
            purpose,
            prefix: &[],
            proposal: &[],
            counterexample: None,
            legal: &[],
            divergences: &[],
        }
    }
    fn setup() -> (PalsStores, ReviewContextIndex, StateId, u64) {
        let mut stores = PalsStores::new(StoreLimits::default());
        let root = stores
            .focus_actual_moves(Position::startpos().snapshot())
            .unwrap();
        let state = stores.situations.get(root).unwrap().state;
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
        (stores, index, state, bundle)
    }
    fn factory() -> Factory {
        Factory {
            model: Arc::new(Model::new(super::super::tests::checkpoint()).unwrap()),
            cached: None,
            stopping: Arc::new(AtomicBool::new(false)),
            invalidations: Arc::new(AtomicU64::new(0)),
            invalidation_ack: Arc::new(AtomicU64::new(0)),
            active_epoch: 0,
        }
    }
    fn request(
        factory: &Factory,
        view: Arc<ReviewView>,
        stores: &PalsStores,
        root: StateId,
    ) -> Request {
        Request::admit(
            &factory.model,
            view,
            stores,
            root,
            stores.generation(),
            Instant::now() + Duration::from_secs(10),
            Arc::new(AtomicBool::new(false)),
        )
        .unwrap()
    }

    #[test]
    fn actual_frozen_views_reuse_c_but_keep_query_seals_and_consumption_authority() {
        let (stores, mut index, state, bundle) = setup();
        let mut factory = factory();
        let (a, b) = index
            .select_frozen_pair(
                &stores,
                state,
                query(RoleQueryPurpose::ProposePolicy),
                query(RoleQueryPurpose::ValueFresh),
                ReviewSelectionPolicy::default(),
            )
            .unwrap();
        assert_ne!(a.digest(), b.digest());
        let protected = index.bundle(bundle).unwrap().protected_consumptions();
        let original_a = request(&factory, Arc::clone(&a), &stores, state);
        assert!(Arc::ptr_eq(&original_a.sources, a.source_snapshots()));
        let prepared_a = factory.prepare(original_a).unwrap();
        assert!(Arc::ptr_eq(
            &prepared_a.source_snapshots,
            a.source_snapshots()
        ));
        let prepared_b = factory
            .prepare(request(&factory, Arc::clone(&b), &stores, state))
            .unwrap();
        assert!(!prepared_a.reused_record_scope());
        assert!(prepared_b.reused_record_scope());
        assert!(Arc::ptr_eq(
            &prepared_b.source_snapshots,
            b.source_snapshots()
        ));
        assert!(Arc::ptr_eq(prepared_a.context(), prepared_b.context()));
        assert!(
            !prepared_a
                .instance_authority()
                .matches(prepared_b.instance_authority())
        );
        for prepared in [&prepared_a, &prepared_b] {
            assert_eq!(
                prepared.context_source_snapshots().len(),
                prepared.view().records().len()
            );
            for (source, metadata) in prepared
                .context_source_snapshots()
                .iter()
                .zip(prepared.view().records_metadata())
            {
                assert!(source.same_state(stores.states.get(metadata.source_state).unwrap()));
            }
        }
        assert_eq!(
            prepared_a.numeric_context_sha256(),
            prepared_b.numeric_context_sha256()
        );
        assert_eq!(
            prepared_a.numeric_context_sha256(),
            prepared_a.context().sha256().unwrap()
        );
        assert_ne!(
            prepared_a.provenance().full_view_digest,
            prepared_b.provenance().full_view_digest
        );
        assert!(prepared_a.validates(&stores, state, stores.generation(), a.seal_digest()));
        assert!(!prepared_a.validates(&stores, state, stores.generation(), b.seal_digest()));
        assert_eq!(
            index.bundle(bundle).unwrap().protected_consumptions(),
            protected
        );
        index.consume(&a, &stores).unwrap();
        assert_eq!(
            index.bundle(bundle).unwrap().protected_consumptions(),
            protected - 1
        );
    }

    #[test]
    fn actual_rules_rebase_preserves_original_bundle_move_and_empty_suffix_relation() {
        let (mut stores, mut index, _state, bundle) = setup();
        let mut actual = Position::startpos();
        actual
            .make_move("e2e4".parse::<BoardMove>().unwrap())
            .unwrap();
        let situation = stores.focus_actual_moves(actual.snapshot()).unwrap();
        let current = stores.situations.get(situation).unwrap().state;
        let view = Arc::new(
            index
                .select(
                    &stores,
                    current,
                    query(RoleQueryPurpose::ReplyPolicy),
                    ReviewSelectionPolicy {
                        required_bundle: Some(bundle),
                        ..ReviewSelectionPolicy::default()
                    },
                )
                .unwrap(),
        );
        assert!(view.records()[0].line.is_empty());
        assert_eq!(view.records_metadata()[0].source_line_offset, 1);
        let mut factory = factory();
        let bound = factory
            .prepare(request(&factory, Arc::clone(&view), &stores, current))
            .unwrap();
        assert_eq!(bound.context().line_count(), 2);
        assert_eq!(bound.context().edge_count(), 1);
        assert_eq!(bound.context_source_snapshots().len(), 2);
        assert!(
            bound
                .context_source_snapshots()
                .iter()
                .all(|source| source.same_state(&Position::startpos().snapshot()))
        );
        assert!(!bound.context_source_snapshots()[0].same_state(&view.anchors()[0].snapshot));
        let reference = bound.context().reference_lines();
        assert_eq!(reference[0].moves.len(), 0);
        assert!(reference[0].original_first_move.is_some());
        assert_eq!(reference[1].moves.len(), 1);
        assert_eq!(
            reference[0]
                .original_first_move
                .as_ref()
                .unwrap()
                .from_square,
            12
        );
    }

    #[test]
    fn foreign_store_stale_root_cancel_and_expiry_reject_without_consumption() {
        let (mut stores, mut index, state, bundle) = setup();
        let factory = factory();
        let view = Arc::new(
            index
                .select(
                    &stores,
                    state,
                    query(RoleQueryPurpose::ProposePolicy),
                    ReviewSelectionPolicy::default(),
                )
                .unwrap(),
        );
        let (foreign, _, _, _) = setup();
        assert!(
            Request::admit(
                &factory.model,
                Arc::clone(&view),
                &foreign,
                state,
                stores.generation(),
                Instant::now() + Duration::from_secs(1),
                Arc::new(AtomicBool::new(false))
            )
            .is_err()
        );
        let cancelled = request(&factory, Arc::clone(&view), &stores, state);
        cancelled.cancel.store(true, Ordering::Release);
        let mut preparing = factory;
        assert!(matches!(
            preparing.prepare(cancelled),
            Err(PreparationError::Cancelled)
        ));
        let mut expired = request(&preparing, Arc::clone(&view), &stores, state);
        expired.deadline = Instant::now() - Duration::from_secs(1);
        assert!(matches!(
            preparing.prepare(expired),
            Err(PreparationError::Expired)
        ));
        assert_eq!(index.bundle(bundle).unwrap().protected_consumptions(), 2);
        let next = Position::from_fen("8/8/8/8/8/8/4K3/7k w - - 0 1").unwrap();
        stores.focus_actual_moves(next.snapshot()).unwrap();
        assert!(
            Request::admit(
                &preparing.model,
                view,
                &stores,
                state,
                stores.generation(),
                Instant::now() + Duration::from_secs(1),
                Arc::new(AtomicBool::new(false))
            )
            .is_err()
        );
    }

    #[test]
    fn actor_retains_terminal_slot_until_collected_and_joins_on_shutdown() {
        let (stores, mut index, state, _) = setup();
        let model = Arc::new(Model::new(super::super::tests::checkpoint()).unwrap());
        let view = Arc::new(
            index
                .select(
                    &stores,
                    state,
                    query(RoleQueryPurpose::ProposePolicy),
                    ReviewSelectionPolicy::default(),
                )
                .unwrap(),
        );
        let make_request = || {
            Request::admit(
                &model,
                Arc::clone(&view),
                &stores,
                state,
                stores.generation(),
                Instant::now() + Duration::from_secs(10),
                Arc::new(AtomicBool::new(false)),
            )
            .unwrap()
        };
        let actor = PreparationActor::spawn(Arc::clone(&model)).unwrap();
        let observed_from = Instant::now();
        let first_request = make_request();
        let original_request = first_request.identity();
        actor
            .try_submit(first_request)
            .unwrap_or_else(|_| panic!("first admission"));
        assert!(matches!(
            actor.try_submit(make_request()),
            Err((PreparationError::Busy, _))
        ));
        let until = Instant::now() + Duration::from_secs(10);
        let completion = loop {
            if let Some(completion) = actor.try_complete().unwrap() {
                break completion;
            }
            assert!(
                Instant::now() < until,
                "actor did not publish terminal result"
            );
            thread::yield_now();
        };
        assert_eq!(
            completion.observation.status(),
            PreparationStatus::Completed
        );
        assert_eq!(completion.observation.reused_record_scope(), Some(false));
        assert_eq!(
            completion.terminal_ack.status(),
            PreparationStatus::Completed
        );
        assert!(completion.terminal_ack.matches_original(&model, &view));
        assert!(completion.terminal_ack.allows_c_only_cleanup());
        assert!(completion.terminal_ack.matches_request(&original_request));
        let same_view_unsubmitted = make_request();
        assert_eq!(same_view_unsubmitted.provenance(), &completion.provenance);
        assert!(
            !completion
                .terminal_ack
                .matches_request(&same_view_unsubmitted.identity())
        );
        assert_eq!(completion.provenance.view_seal, view.seal_digest());
        assert!(
            completion.observation.queue_wait()
                + completion.observation.actor_dispatch()
                + completion.observation.service()
                <= observed_from.elapsed()
        );
        let bound = completion.result.unwrap();
        assert!(bound.matches_request(&original_request));
        assert!(!bound.matches_request(&same_view_unsubmitted.identity()));
        assert!(bound.validates(&stores, state, stores.generation(), view.seal_digest()));
        let second_request = make_request();
        let second_identity = second_request.identity();
        actor
            .try_submit(second_request)
            .unwrap_or_else(|_| panic!("second admission"));
        let reused = loop {
            if let Some(completion) = actor.try_complete().unwrap() {
                break completion;
            }
            assert!(Instant::now() < until);
            thread::yield_now();
        };
        assert_eq!(reused.observation.status(), PreparationStatus::Completed);
        assert_eq!(reused.observation.reused_record_scope(), Some(true));
        assert_eq!(reused.provenance.view_seal, view.seal_digest());
        let rebound = reused.result.unwrap();
        assert!(Arc::ptr_eq(bound.context(), rebound.context()));
        assert!(rebound.matches_request(&second_identity));
        assert!(!rebound.matches_request(&original_request));
        assert!(!bound.matches_request(&second_identity));
        actor.shutdown().unwrap();
    }

    #[test]
    fn actor_observations_preserve_actual_cancel_expiry_and_failure_outcomes() {
        let (stores, mut index, state, _) = setup();
        let model = Arc::new(Model::new(super::super::tests::checkpoint()).unwrap());
        let view = Arc::new(
            index
                .select(
                    &stores,
                    state,
                    query(RoleQueryPurpose::ProposePolicy),
                    ReviewSelectionPolicy::default(),
                )
                .unwrap(),
        );
        let actor = PreparationActor::spawn(Arc::clone(&model)).unwrap();
        for (status, expected) in [
            (PreparationStatus::Cancelled, PreparationError::Cancelled),
            (PreparationStatus::Expired, PreparationError::Expired),
            (PreparationStatus::Failed, PreparationError::StaleView),
        ] {
            let cancel = Arc::new(AtomicBool::new(false));
            let mut request = Request::admit(
                &model,
                Arc::clone(&view),
                &stores,
                state,
                stores.generation(),
                Instant::now() + Duration::from_secs(10),
                Arc::clone(&cancel),
            )
            .unwrap();
            match status {
                PreparationStatus::Cancelled => cancel.store(true, Ordering::Release),
                PreparationStatus::Expired => {
                    request.deadline = Instant::now() - Duration::from_secs(1)
                }
                PreparationStatus::Failed => {
                    request.provenance.model.semantics.push_str("-foreign")
                }
                PreparationStatus::Completed => unreachable!(),
            }
            let original = request.provenance().clone();
            actor
                .try_submit(request)
                .unwrap_or_else(|_| panic!("actual observed admission"));
            let until = Instant::now() + Duration::from_secs(10);
            let completion = loop {
                if let Some(completion) = actor.try_complete().unwrap() {
                    break completion;
                }
                assert!(Instant::now() < until);
                thread::yield_now();
            };
            assert_eq!(completion.provenance, original);
            assert_eq!(completion.observation.status(), status);
            assert_eq!(completion.observation.reused_record_scope(), None);
            assert_eq!(completion.terminal_ack.provenance(), &original);
            assert_eq!(completion.terminal_ack.status(), status);
            assert_eq!(completion.terminal_ack.model_identity(), model.identity());
            assert!(completion.terminal_ack.matches_original(&model, &view));
            assert!(completion.terminal_ack.allows_c_only_cleanup());
            assert_eq!(completion.result.err(), Some(expected));
        }
        actor.shutdown().unwrap();
    }

    #[test]
    fn late_cancel_expiry_and_cache_epoch_revoke_prepared_acceptance() {
        let (stores, mut index, state, _) = setup();
        let view = Arc::new(
            index
                .select(
                    &stores,
                    state,
                    query(RoleQueryPurpose::ProposePolicy),
                    ReviewSelectionPolicy::default(),
                )
                .unwrap(),
        );
        let mut factory = factory();
        let request = request(&factory, Arc::clone(&view), &stores, state);
        let cancel = Arc::clone(&request.cancel);
        let mut bound = factory.prepare(request).unwrap();
        assert!(bound.validates(&stores, state, stores.generation(), view.seal_digest()));
        assert_eq!(bound.authority_revocation(), None);
        cancel.store(true, Ordering::Release);
        assert_eq!(
            bound.authority_revocation(),
            Some(NnuecAuthorityRevocation::Cancelled)
        );
        assert!(!bound.validates(&stores, state, stores.generation(), view.seal_digest()));
        cancel.store(false, Ordering::Release);
        bound.deadline = Instant::now() - Duration::from_secs(1);
        assert_eq!(
            bound.authority_revocation(),
            Some(NnuecAuthorityRevocation::Expired)
        );
        assert!(!bound.validates(&stores, state, stores.generation(), view.seal_digest()));
        bound.deadline = Instant::now() + Duration::from_secs(1);
        factory.invalidations.fetch_add(1, Ordering::AcqRel);
        cancel.store(true, Ordering::Release);
        bound.deadline = Instant::now() - Duration::from_secs(1);
        assert_eq!(
            bound.authority_revocation(),
            Some(NnuecAuthorityRevocation::Invalidated)
        );
        assert!(!bound.validates(&stores, state, stores.generation(), view.seal_digest()));
        factory.sync_cache_epoch();
        assert!(factory.cached.is_none());
        assert_eq!(factory.invalidation_ack.load(Ordering::Acquire), 1);
        factory.stopping.store(true, Ordering::Release);
        assert_eq!(
            bound.authority_revocation(),
            Some(NnuecAuthorityRevocation::ActorStopped)
        );
    }

    #[test]
    fn utility_capture_selection_is_explicit_bounded_and_failure_atomic() {
        let (stores, mut index, state, _) = setup();
        let view = Arc::new(
            index
                .select(
                    &stores,
                    state,
                    query(RoleQueryPurpose::ProposePolicy),
                    ReviewSelectionPolicy::default(),
                )
                .unwrap(),
        );
        let mut factory = factory();
        let plain = factory
            .prepare(request(&factory, Arc::clone(&view), &stores, state))
            .unwrap();
        assert!(plain.utility_input_capture().is_none());
        let original = request(&factory, Arc::clone(&view), &stores, state);
        let identity = original.identity();
        let (error, original) = original
            .with_utility_input_capture(64)
            .err()
            .expect("out-of-range selection");
        assert!(matches!(error, PreparationError::Neural(_)));
        assert!(original.identity().same_request(&identity));
        assert!(original.utility_target.is_none());
        let original = original
            .with_utility_input_capture(0)
            .unwrap_or_else(|_| panic!("actual target selection"));
        let (_, original) = original
            .with_utility_input_capture(1)
            .err()
            .expect("second selection cannot replace original");
        assert_eq!(original.utility_target, Some(0));
        let bound = Arc::new(factory.prepare(original).unwrap());
        let captured = bound.utility_input_capture().unwrap();
        assert!(captured.matches_request(&identity));
        assert!(captured.matches_bound(&bound));
        assert_eq!(captured.target_record_ids(), &[0, 1]);
        assert_eq!(captured.target_record(), 0);
        assert_eq!(
            captured.original_first_move(),
            "e2e4".parse::<BoardMove>().unwrap()
        );
        assert!(captured.validates(&stores, state, stores.generation(), &view));
        assert!(!captured.validates(
            &PalsStores::new(StoreLimits::default()),
            state,
            stores.generation(),
            &view
        ));
        assert_eq!(
            serde_json::to_vec(captured.reference()).unwrap(),
            captured.input_bytes()
        );
    }

    #[test]
    fn same_actor_captures_original_utility_input_without_future_records_or_view_pins() {
        let (stores, mut index, state, _) = setup();
        let model = Arc::clone(&factory().model);
        let view = Arc::new(
            index
                .select(
                    &stores,
                    state,
                    query(RoleQueryPurpose::ProposePolicy),
                    ReviewSelectionPolicy::default(),
                )
                .unwrap(),
        );
        let weak_view = Arc::downgrade(&view);
        let actor = PreparationActor::spawn(Arc::clone(&model)).unwrap();
        let mut bounds = Vec::new();
        for _ in 0..2 {
            let request = Request::admit(
                &model,
                Arc::clone(&view),
                &stores,
                state,
                stores.generation(),
                Instant::now() + Duration::from_secs(10),
                Arc::new(AtomicBool::new(false)),
            )
            .unwrap()
            .with_utility_input_capture(0)
            .unwrap_or_else(|_| panic!("utility capture selection"));
            actor
                .try_submit(request)
                .unwrap_or_else(|_| panic!("actual actor admission"));
            let until = Instant::now() + Duration::from_secs(10);
            loop {
                if let Some(completion) = actor.try_complete().unwrap() {
                    bounds.push(Arc::new(completion.result.unwrap()));
                    break;
                }
                assert!(Instant::now() < until);
                thread::yield_now();
            }
        }
        let captured = Arc::clone(bounds[0].utility_input_capture().unwrap());
        let second = Arc::clone(bounds[1].utility_input_capture().unwrap());
        assert!(Arc::ptr_eq(bounds[0].context(), bounds[1].context()));
        assert!(captured.matches_bound(&bounds[0]));
        assert!(!captured.matches_bound(&bounds[1]));
        assert_eq!(captured.input_sha256(), second.input_sha256());
        assert_eq!(captured.input_bytes(), second.input_bytes());
        assert_eq!(captured.reference().base_input.context_lines.len(), 2);
        assert!(captured.reference().base_input.local_line.moves.is_empty());
        assert!(
            captured
                .root_snapshot()
                .same_state(&Position::startpos().snapshot())
        );
        assert!(captured.matches_original_model(&model));
        let foreign_model = Arc::new(Model::new(super::super::tests::checkpoint()).unwrap());
        assert_eq!(foreign_model.identity(), model.identity());
        assert!(!captured.matches_original_model(&foreign_model));
        let before = captured.input_bytes().to_vec();
        index
            .publish(
                &stores,
                &record(state, 3, RecordKind::Repair, "e2e4 c7c5", Some(2)),
                None,
            )
            .unwrap();
        assert_eq!(captured.reference().base_input.context_lines.len(), 2);
        assert_eq!(captured.input_bytes(), before);
        let weak_bound = Arc::downgrade(&bounds[0]);
        actor.shutdown().unwrap();
        assert!(!captured.validates(&stores, state, stores.generation(), &view));
        drop(bounds);
        drop(view);
        assert!(weak_bound.upgrade().is_none());
        assert!(weak_view.upgrade().is_none());
        assert_eq!(stores.review_pin_count().unwrap(), 0);
        assert_eq!(captured.records_metadata().len(), 2);
        assert_eq!(captured.source_snapshots().len(), 2);
        assert_eq!(captured.input_bytes(), before);
    }

    #[test]
    fn utility_capture_rebased_bundle_keeps_original_move_and_current_global_empty_local_input() {
        let (mut stores, mut index, _, bundle) = setup();
        let mut actual = Position::startpos();
        actual
            .make_move("e2e4".parse::<BoardMove>().unwrap())
            .unwrap();
        let situation = stores.focus_actual_moves(actual.snapshot()).unwrap();
        let state = stores.situations.get(situation).unwrap().state;
        let view = Arc::new(
            index
                .select(
                    &stores,
                    state,
                    query(RoleQueryPurpose::ReplyPolicy),
                    ReviewSelectionPolicy {
                        required_bundle: Some(bundle),
                        ..ReviewSelectionPolicy::default()
                    },
                )
                .unwrap(),
        );
        let mut factory = factory();
        let original = request(&factory, Arc::clone(&view), &stores, state)
            .with_utility_input_capture(0)
            .unwrap_or_else(|_| panic!("rebased target selection"));
        let bound = factory.prepare(original).unwrap();
        let captured = bound.utility_input_capture().unwrap();
        assert!(captured.root_snapshot().same_state(&actual.snapshot()));
        assert_eq!(
            captured.reference().base_input.board_features,
            super::super::super::rules::reference_board_features(&actual)
        );
        assert!(captured.reference().base_input.local_line.moves.is_empty());
        assert!(
            captured.reference().base_input.context_lines[0]
                .moves
                .is_empty()
        );
        assert_eq!(
            captured.original_first_move(),
            "e2e4".parse::<BoardMove>().unwrap()
        );
        assert_eq!(captured.target_record_ids(), &[0, 1]);
        assert!(
            captured
                .source_snapshots()
                .iter()
                .all(|source| source.same_state(&Position::startpos().snapshot()))
        );
    }

    #[test]
    fn independent_cache_invalidation_wakes_idle_actor_and_preserves_terminal_slot() {
        let (stores, mut index, state, _) = setup();
        let model = Arc::new(Model::new(super::super::tests::checkpoint()).unwrap());
        let view = Arc::new(
            index
                .select(
                    &stores,
                    state,
                    query(RoleQueryPurpose::ProposePolicy),
                    ReviewSelectionPolicy::default(),
                )
                .unwrap(),
        );
        let actor = PreparationActor::spawn(Arc::clone(&model)).unwrap();
        let request = Request::admit(
            &model,
            Arc::clone(&view),
            &stores,
            state,
            stores.generation(),
            Instant::now() + Duration::from_secs(10),
            Arc::new(AtomicBool::new(false)),
        )
        .unwrap();
        actor
            .try_submit(request)
            .unwrap_or_else(|_| panic!("first admission"));
        let until = Instant::now() + Duration::from_secs(10);
        let bound = loop {
            if let Some(completion) = actor.try_complete().unwrap() {
                break completion.result.unwrap();
            }
            assert!(Instant::now() < until);
            thread::yield_now();
        };
        let epoch = actor.invalidate_cache().unwrap();
        assert!(!bound.validates(&stores, state, stores.generation(), view.seal_digest()));
        while actor.cache_invalidated_through() < epoch {
            assert!(Instant::now() < until);
            thread::yield_now();
        }
        assert!(actor.try_complete().unwrap().is_none());
        actor.shutdown().unwrap();
    }

    fn actual_cache_prelude_ack(
        actor: &PreparationActor,
    ) -> super::super::retirement::NnuecCacheReleaseAck {
        let ticket = actor.invalidate_context_cache().unwrap();
        let until = Instant::now() + Duration::from_secs(10);
        loop {
            if let Some(release) = actor.try_context_cache_release(&ticket).unwrap() {
                return release;
            }
            assert!(Instant::now() < until, "actual prelude cache ACK timed out");
            thread::yield_now();
        }
    }

    #[test]
    fn optional_shared_readout_keeps_plain_preparation_and_runs_exact_query_on_actor() {
        let (stores, mut index, state, _) = setup();
        let model = shared_readout_fixture_model();
        let prefix: Vec<BoardMove> = "e2e4 e7e5"
            .split_whitespace()
            .map(|text| text.parse().unwrap())
            .collect();
        let mut endpoint = Position::startpos();
        for movement in &prefix {
            endpoint.make_move(*movement).unwrap();
        }
        let view = Arc::new(
            index
                .select(
                    &stores,
                    state,
                    ReviewQuerySequences {
                        prefix: &prefix,
                        ..query(RoleQueryPurpose::ReplyPolicy)
                    },
                    ReviewSelectionPolicy::default(),
                )
                .unwrap(),
        );
        let actor = PreparationActor::spawn(Arc::clone(&model)).unwrap();
        assert!(actor.worker_thread_id().is_some());
        let admit = || {
            Request::admit(
                &model,
                Arc::clone(&view),
                &stores,
                state,
                stores.generation(),
                Instant::now() + Duration::from_secs(10),
                Arc::new(AtomicBool::new(false)),
            )
            .unwrap()
        };
        let plain = admit();
        assert!(!plain.has_shared_readout());
        let plain_bound = actual_actor_prepare(&actor, plain).unwrap();
        assert!(plain_bound.shared_readout().is_none());
        assert!(plain_bound.utility_input_capture().is_none());
        drop(plain_bound);
        let request = admit()
            .with_shared_readout(&endpoint.snapshot())
            .unwrap_or_else(|_| panic!("original explicit shared target"));
        let identity = request.identity();
        let (error, request) = request.with_utility_input_capture(0).err().unwrap();
        assert!(matches!(error, PreparationError::InvalidView(_)));
        assert!(request.identity().same_request(&identity));
        let bound = actual_actor_prepare(&actor, request).unwrap();
        assert!(bound.utility_input_capture().is_none());
        let readout = bound.shared_readout().unwrap();
        assert!(readout.matches_request(&identity));
        assert!(readout.endpoint().same_state(&endpoint.snapshot()));
        let mut local = model.local(&Position::startpos()).unwrap();
        for movement in prefix {
            model.append_local(&mut local, movement).unwrap();
        }
        let board = model.board(&endpoint).unwrap();
        let reference = model
            .forward(&board, bound.context(), &local, &endpoint)
            .unwrap();
        assert_eq!(
            serde_json::to_vec(readout.output()).unwrap(),
            serde_json::to_vec(&reference).unwrap()
        );
        assert_eq!(readout.input_sha256().len(), 64);
        drop(bound);
        let release = actual_cache_prelude_ack(&actor);
        assert!(actor.verifies_shared_cache_release(&model, &release));
        actor.shutdown().unwrap();
    }

    #[test]
    fn shared_actor_errors_keep_original_terminal_and_exact_control_class() {
        let (stores, mut index, state, _) = setup();
        let model = shared_readout_fixture_model();
        let view = Arc::new(
            index
                .select(
                    &stores,
                    state,
                    query(RoleQueryPurpose::ProposePolicy),
                    ReviewSelectionPolicy::default(),
                )
                .unwrap(),
        );
        let actor = PreparationActor::spawn(Arc::clone(&model)).unwrap();
        for expected in [
            PreparationError::Cancelled,
            PreparationError::Expired,
            PreparationError::InvalidView("shared readout endpoint Rules mismatch"),
        ] {
            let cancel = Arc::new(AtomicBool::new(false));
            let mut endpoint = Position::startpos();
            if matches!(expected, PreparationError::InvalidView(_)) {
                endpoint.make_move("e2e4".parse().unwrap()).unwrap();
            }
            let mut request = Request::admit(
                &model,
                Arc::clone(&view),
                &stores,
                state,
                stores.generation(),
                Instant::now() + Duration::from_secs(10),
                Arc::clone(&cancel),
            )
            .unwrap()
            .with_shared_readout(&endpoint.snapshot())
            .unwrap_or_else(|_| panic!("original shared Request"));
            let identity = request.identity();
            match expected {
                PreparationError::Cancelled => cancel.store(true, Ordering::Release),
                PreparationError::Expired => request.deadline = Instant::now(),
                _ => {}
            }
            actor
                .try_submit(request)
                .unwrap_or_else(|_| panic!("original actor submission"));
            let until = Instant::now() + Duration::from_secs(10);
            let completion = loop {
                if let Some(completion) = actor.try_complete().unwrap() {
                    break completion;
                }
                assert!(Instant::now() < until);
                thread::yield_now();
            };
            assert_eq!(completion.result.err(), Some(expected));
            assert!(completion.terminal_ack.matches_request(&identity));
            assert!(completion.terminal_ack.matches_original(&model, &view));
            assert!(completion.terminal_ack.allows_c_only_cleanup());
            assert_ne!(
                completion.terminal_ack.status(),
                PreparationStatus::Completed
            );
        }
        actor.shutdown().unwrap();
    }
    fn actual_utility_request(
        actor_model: &Model,
        view: &Arc<ReviewView>,
        stores: &PalsStores,
        state: StateId,
    ) -> Request {
        Request::admit(
            actor_model,
            Arc::clone(view),
            stores,
            state,
            stores.generation(),
            Instant::now() + Duration::from_secs(10),
            Arc::new(AtomicBool::new(false)),
        )
        .unwrap()
        .with_utility_input_capture(0)
        .unwrap_or_else(|_| panic!("explicit target selection"))
    }
    fn actual_actor_prepare(
        actor: &PreparationActor,
        request: Request,
    ) -> Result<BoundNnuecReviewContext, PreparationError> {
        actor
            .try_submit(request)
            .unwrap_or_else(|_| panic!("actual prelude Request admission"));
        let until = Instant::now() + Duration::from_secs(10);
        loop {
            if let Some(completion) = actor.try_complete().unwrap() {
                return completion.result;
            }
            assert!(Instant::now() < until, "actual preparation timed out");
            thread::yield_now();
        }
    }

    #[test]
    fn utility_cache_prelude_uses_original_ack_request_and_bound_without_retaining_view() {
        let (mut stores, mut index, state, _) = setup();
        let model = Arc::clone(&factory().model);
        let view = Arc::new(
            index
                .select(
                    &stores,
                    state,
                    query(RoleQueryPurpose::ProposePolicy),
                    ReviewSelectionPolicy::default(),
                )
                .unwrap(),
        );
        let weak_view = Arc::downgrade(&view);
        let actor = PreparationActor::spawn(Arc::clone(&model)).unwrap();
        let ack = actual_cache_prelude_ack(&actor);
        let epoch = ack.epoch();
        let request = actual_utility_request(&model, &view, &stores, state)
            .with_utility_cache_prelude(ack)
            .unwrap_or_else(|_| panic!("move actual ACK before submission"));
        let identity = request.identity();
        let bound = Arc::new(actual_actor_prepare(&actor, request).unwrap());
        assert!(!bound.reused_record_scope());
        let captured = Arc::clone(bound.utility_input_capture().unwrap());
        let prelude = captured.cache_prelude().expect("actual first C prelude");
        assert_eq!(prelude.epoch(), epoch);
        assert!(prelude.matches_input(&captured));
        assert!(captured.matches_request(&identity));
        assert!(captured.matches_original_store(&stores));
        assert!(!captured.matches_original_store(&PalsStores::new(StoreLimits::default())));

        // Same public input and same C cache are not the first Request again.
        let second = Arc::new(
            actual_actor_prepare(
                &actor,
                actual_utility_request(&model, &view, &stores, state),
            )
            .unwrap(),
        );
        let other = second.utility_input_capture().unwrap();
        assert!(second.reused_record_scope());
        assert_eq!(captured.input_bytes(), other.input_bytes());
        assert!(other.cache_prelude().is_none());
        assert!(!prelude.matches_input(other));
        actor.shutdown().unwrap();
        drop(second);
        drop(bound);
        drop(view);
        assert!(weak_view.upgrade().is_none());
        assert_eq!(stores.review_pin_count().unwrap(), 0);
        assert!(
            prelude.matches_input(&captured),
            "past evidence is still readable"
        );
        stores
            .focus_actual_moves(Position::startpos().snapshot())
            .unwrap();
        assert!(
            !captured.matches_original_store(&stores),
            "new generation is not the original Store scope"
        );
    }

    #[test]
    fn utility_cache_prelude_rejects_foreign_actor_and_intervening_preparation() {
        let (stores, mut index, state, _) = setup();
        let model = Arc::clone(&factory().model);
        let view = Arc::new(
            index
                .select(
                    &stores,
                    state,
                    query(RoleQueryPurpose::ProposePolicy),
                    ReviewSelectionPolicy::default(),
                )
                .unwrap(),
        );
        let actor = PreparationActor::spawn(Arc::clone(&model)).unwrap();
        let foreign = PreparationActor::spawn(Arc::clone(&model)).unwrap();
        // Equal epoch numbers and the same actual model cannot replace the actor.
        let foreign_ack = actual_cache_prelude_ack(&foreign);
        let own_ack = actual_cache_prelude_ack(&actor);
        assert_eq!(foreign_ack.epoch(), own_ack.epoch());
        let request = actual_utility_request(&model, &view, &stores, state)
            .with_utility_cache_prelude(foreign_ack)
            .unwrap_or_else(|_| panic!("Request custody only"));
        assert!(matches!(
            actual_actor_prepare(&actor, request),
            Err(PreparationError::InvalidView(
                "utility cache prelude is not the first same-actor epoch preparation"
            ))
        ));
        let intervening = actual_actor_prepare(
            &actor,
            actual_utility_request(&model, &view, &stores, state),
        )
        .unwrap();
        assert!(
            intervening
                .utility_input_capture()
                .unwrap()
                .cache_prelude()
                .is_none()
        );
        let request = actual_utility_request(&model, &view, &stores, state)
            .with_utility_cache_prelude(own_ack)
            .unwrap_or_else(|_| panic!("original ACK custody"));
        assert!(matches!(
            actual_actor_prepare(&actor, request),
            Err(PreparationError::InvalidView(
                "utility cache prelude is not the first same-actor epoch preparation"
            ))
        ));
        actor.shutdown().unwrap();
        foreign.shutdown().unwrap();
    }

    #[test]
    fn utility_cache_prelude_rejects_superseded_epoch_and_returns_unused_ack_on_selection_error() {
        let (stores, mut index, state, _) = setup();
        let model = Arc::clone(&factory().model);
        let view = Arc::new(
            index
                .select(
                    &stores,
                    state,
                    query(RoleQueryPurpose::ProposePolicy),
                    ReviewSelectionPolicy::default(),
                )
                .unwrap(),
        );
        let actor = PreparationActor::spawn(Arc::clone(&model)).unwrap();
        let ack = actual_cache_prelude_ack(&actor);
        let plain = Request::admit(
            &model,
            Arc::clone(&view),
            &stores,
            state,
            stores.generation(),
            Instant::now() + Duration::from_secs(10),
            Arc::new(AtomicBool::new(false)),
        )
        .unwrap();
        let identity = plain.identity();
        let (error, plain, ack) = match plain.with_utility_cache_prelude(ack) {
            Err(originals) => originals,
            Ok(_) => panic!("prelude without a selected target was accepted"),
        };
        assert_eq!(
            error,
            PreparationError::InvalidView(
                "utility cache prelude requires one selected original Request"
            )
        );
        assert!(plain.identity().same_request(&identity));
        let old_epoch = ack.epoch();
        let current = actual_cache_prelude_ack(&actor);
        assert!(current.epoch() > old_epoch);
        let request = plain
            .with_utility_input_capture(0)
            .unwrap_or_else(|_| panic!("same Request target selection"))
            .with_utility_cache_prelude(ack)
            .unwrap_or_else(|_| panic!("old ACK moved, not reissued"));
        assert!(matches!(
            actual_actor_prepare(&actor, request),
            Err(PreparationError::InvalidView(
                "utility cache prelude is not the first same-actor epoch preparation"
            ))
        ));
        // The failed old epoch did not consume the current ACK or fill C cache.
        let request = actual_utility_request(&model, &view, &stores, state)
            .with_utility_cache_prelude(current)
            .unwrap_or_else(|_| panic!("actual current ACK"));
        let bound = actual_actor_prepare(&actor, request).unwrap();
        assert!(
            bound
                .utility_input_capture()
                .unwrap()
                .cache_prelude()
                .is_some()
        );
        actor.shutdown().unwrap();
    }
}
