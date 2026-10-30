//! 자체 CPU_T/CPU_R 코어: 유한 iterative deepening, PVS, aspiration, quiescence.
//!
//! 규칙과 종료 판정은 `rz-position`이 소유한다. 이 모듈의 material/PST 점수는
//! `bootstrap-material-pst-v1`의 차례 관점 raw 단위이며 학습 값이나 보정된 CP가 아니다.
//! TT의 짧은 hash는 slot 선택만 담당한다. 재사용은 전체 알려진 Rules 이력의
//! `PositionIdentity`, profile, 남은 깊이와 bound를 함께 검사한다.
//! 기본 재귀 정책은 중단된 노드를 저장하지 않는다. opt-in PausedStack은 실제
//! 미완료 프레임과 Rules/accumulator 소유자를 보존한다. 체크에서 quiescence 한도에 도달하면 static
//! stand-pat을 반환하는 대신 현재 iteration을 명시적으로 중단한다.
//! LegalSeeV1은 명시적으로 선택하는 ordering 변경이다. 제한된 동일 칸의
//! material 교환 minimax만 계산하며 게임 값·전술 증명·pruning bound가 아니다.

#[cfg(test)]
mod cooperative_tests;
mod see;
mod stack;

use crate::cpu_value::{
    BootstrapCpuValue, CpuAccumulator, CpuValueCacheContext, CpuValueError, CpuValueEvaluator,
    CpuValueIdentity,
};
use rz_position::{
    BoardMove, Color, PieceKind, PlayStatus, Position, PositionError, PositionIdentity, Square,
    TerminalReason,
};
use sha2::{Digest, Sha256};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

pub const CPU_SEARCH_VERSION: &str = "rz-cpu-pvs/0.1";
pub const CPU_SEE_SEARCH_VERSION: &str = "rz-cpu-pvs-legal-see/0.1";
pub const CPU_PAUSED_STACK_SEARCH_VERSION: &str = "rz-cpu-pvs-paused-stack/0.1";
pub const CPU_PAUSED_STACK_SEE_SEARCH_VERSION: &str = "rz-cpu-pvs-legal-see-paused-stack/0.1";
pub const CPU_PAUSED_SIDE_SCOPE_SEARCH_VERSION: &str = "rz-cpu-pvs-paused-side-scope/1";
pub const CPU_PAUSED_SIDE_SCOPE_SEE_SEARCH_VERSION: &str =
    "rz-cpu-pvs-legal-see-paused-side-scope/1";
pub const CPU_ROOT_TIE_SEARCH_VERSION: &str = "rz-cpu-pvs-root-tie-confirmed/1";
pub const CPU_ROOT_TIE_SEE_SEARCH_VERSION: &str = "rz-cpu-pvs-legal-see-root-tie-confirmed/1";
pub const CPU_ROOT_TIE_PAUSED_SEARCH_VERSION: &str = "rz-cpu-pvs-paused-root-tie-confirmed/1";
pub const CPU_ROOT_TIE_PAUSED_SEE_SEARCH_VERSION: &str =
    "rz-cpu-pvs-legal-see-paused-root-tie-confirmed/1";
pub const CPU_ROOT_TIE_SIDE_SEARCH_VERSION: &str = "rz-cpu-pvs-side-root-tie-confirmed/1";
pub const CPU_ROOT_TIE_SIDE_SEE_SEARCH_VERSION: &str =
    "rz-cpu-pvs-legal-see-side-root-tie-confirmed/1";
pub const CPU_PAUSED_STACK_MAX_BYTES: usize = 8 * 1024 * 1024;
pub const CPU_PAUSED_STACK_OWNER_ACCOUNTING_DOMAIN: &str =
    "rz-pals-paused-stack-owner-accounting-v4/1";
pub const CPU_PAUSED_STACK_OWNER_ACCOUNTING_METHOD: &str = "cpu_actual_paused_frame_owner_v4";
pub const CPU_SEARCH_CONDITIONS: &str = "iterative-deepening:1..requested;root-window:full-first,aspiration40-following,full-when-mate-or-fail-inclusive;pvs:first-full,following-zero-window,strict-interior-research;qsearch:tactical-capture-ep-promotion,all-check-evasions,no-check-standpat;q-limit:checked-abort;tt:direct-mapped,full-history-value-profile,equal-remaining-depth,completed-nodes-only;selectivity:no-reductions-no-nullmove;ties:Rules-order;score:side-to-move-raw";
pub const BOOTSTRAP_SCORE_VERSION: &str = "bootstrap-material-pst-v1";
pub const CPU_MATE_SCORE: i32 = 30_000;
pub const CPU_MATE_THRESHOLD: i32 = 29_000;
pub const CPU_FRONTIER_SCORE_LIMIT: i32 = 20_000;
const INFINITY: i32 = 32_000;
const MATE_THRESHOLD: i32 = CPU_MATE_THRESHOLD;
const MAX_DEPTH: u16 = 64;
const MAX_QUIESCENCE_PLY: u16 = 32;
const MAX_TT_ENTRIES: usize = 1_048_576;
const MAX_ROOT_MOVES: usize = 256;
const MAX_KNOWN_HISTORY: usize = 4096;
const ASPIRATION_WINDOW: i32 = 40;
const MAX_SEE_POSITIONS_PER_ORDER: u32 = 4096;
static NEXT_PAUSED_STACK_OWNER_ID: AtomicU64 = AtomicU64::new(1);

/// Compiled owner provenance, independent of token metadata and observations.
/// Domain followed by each logical path and raw source in this fixed order;
/// every item is prefixed by its byte length as a big-endian u64. Changes to
/// these included sources change the pin; it is not a binary/feature identity.
pub fn compiled_paused_stack_owner_sha256() -> [u8; 32] {
    let mut hash = Sha256::new();
    for bytes in [
        CPU_PAUSED_STACK_OWNER_ACCOUNTING_DOMAIN.as_bytes(),
        b"cpu.rs".as_slice(),
        include_bytes!("cpu.rs").as_slice(),
        b"cpu/stack.rs".as_slice(),
        include_bytes!("cpu/stack.rs").as_slice(),
        b"cpu_checker.rs".as_slice(),
        include_bytes!("cpu_checker.rs").as_slice(),
    ] {
        hash.update((bytes.len() as u64).to_be_bytes());
        hash.update(bytes);
    }
    hash.finalize().into()
}

/// Ordering is immutable for the lifetime of one engine and its private TT.
/// Existing constructors retain the historical MVV/LVA policy and namespace.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum CpuOrderingPolicy {
    LegacyMvvLvaV1,
    LegalSeeV1,
}

impl CpuOrderingPolicy {
    pub fn identity(self) -> &'static str {
        match self {
            Self::LegacyMvvLvaV1 => "cpu-ordering-mvv-lva-v1",
            Self::LegalSeeV1 => "cpu-ordering-rules-legal-target-exchange-v1",
        }
    }
}

/// Explicit startup choice. Historical constructors keep recursive search and
/// completed-iteration resume. PausedStack retains one actual owned traversal.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub enum CpuResumePolicy {
    #[default]
    CompletedIteration,
    PausedStack,
}

/// An immutable search-semantic choice, separate from move ordering. Legacy
/// constructors preserve their original first-searched root tie behavior.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub enum CpuRootTiePolicy {
    #[default]
    LegacyFirstSearched,
    RulesOrderConfirmed,
}

impl CpuRootTiePolicy {
    pub fn identity(self) -> &'static str {
        match self {
            Self::LegacyFirstSearched => "cpu-root-tie-first-searched-v1",
            Self::RulesOrderConfirmed => "cpu-root-tie-rules-confirmed-v1",
        }
    }
}

impl CpuResumePolicy {
    pub fn identity(self) -> &'static str {
        match self {
            Self::CompletedIteration => "cpu-resume-completed-iteration-v1",
            Self::PausedStack => "cpu-resume-paused-stack-v1",
        }
    }
}

/// First implementation has no selective reductions in any profile. Distinct
/// identities prevent independent verification work from sharing TT evidence.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum CpuProfile {
    PlanAssisted,
    Independent,
    RelaxedSelectivity,
}

impl CpuProfile {
    pub fn identity(self) -> &'static str {
        match self {
            Self::PlanAssisted => "cpu-plan-assisted-conservative-v1",
            Self::Independent => "cpu-independent-conservative-v1",
            Self::RelaxedSelectivity => "cpu-relaxed-no-reductions-v1",
        }
    }
}

#[derive(Clone, Debug)]
pub struct CpuConfig {
    pub profile: CpuProfile,
    /// Bounded direct-mapped TT slots, never an unbounded map.
    pub tt_entries: usize,
    pub max_depth: u16,
    pub quiescence_ply: u16,
}

impl Default for CpuConfig {
    fn default() -> Self {
        Self {
            profile: CpuProfile::PlanAssisted,
            tt_entries: 8192,
            max_depth: 16,
            quiescence_ply: 16,
        }
    }
}

impl CpuConfig {
    /// Requested contiguous TT slot layout in bytes on the current build target.
    /// Zero slots disables the TT and returns zero. This excludes allocations
    /// retained by entry identities, allocator rounding/bookkeeping, the Vec
    /// header and search history; it is neither total TT usage nor memory peak.
    /// `try_reserve_exact` may still receive a larger capacity from the allocator.
    pub fn tt_allocation_bytes(&self) -> Result<u64, CpuError> {
        if self.tt_entries > MAX_TT_ENTRIES {
            return Err(CpuError::InvalidConfig("TT exceeds 1,048,576 slots"));
        }
        checked_inline_slot_bytes(self.tt_entries, std::mem::size_of::<Option<TtEntry>>())
    }
}

fn checked_inline_slot_bytes(entries: usize, slot_bytes: usize) -> Result<u64, CpuError> {
    let bytes = entries
        .checked_mul(slot_bytes)
        .ok_or(CpuError::InvalidConfig(
            "TT slot bytes overflow address space",
        ))?;
    u64::try_from(bytes)
        .map_err(|_| CpuError::InvalidConfig("TT slot bytes cannot be represented as u64"))
}

#[derive(Clone, Copy, Debug)]
pub struct CpuLimits {
    pub max_depth: u16,
    /// Counts full-search and quiescence positions, including TT probes.
    /// LegalSeeV1 additionally charges all entered exchange positions to this
    /// same budget; exchange work never receives a fresh deadline or budget.
    pub max_nodes: u64,
    pub deadline: Option<Instant>,
}

impl Default for CpuLimits {
    fn default() -> Self {
        Self {
            max_depth: 8,
            max_nodes: 100_000,
            deadline: None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CpuCompletion {
    DepthLimit,
    NodeLimit,
    Deadline,
    Canceled,
    /// Cooperative worker release. This is neither cancellation nor expiry;
    /// an owned PausedStack continuation remains live for the same job.
    Yielded,
    /// An entered node owns a sealed P/C question. This is not a voluntary
    /// Yield and plain resume may not bypass the missing response.
    AwaitReciprocal,
    /// A checked quiescence frontier cannot be statically evaluated safely.
    QuiescenceLimit,
    Terminal(TerminalReason),
}

/// The runtime owns atomic queue-independent control. Search supplies immutable
/// Job/worker scope; a successor job always receives a different mailbox.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CpuMailboxScope {
    pub job: rz_contracts::pals_adaptive::CpuJobId,
    pub owner: rz_contracts::pals_adaptive::CpuWorkerId,
}
pub type CpuControlMailbox = rz_runtime::adaptive::ControlMailbox<CpuMailboxScope>;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CpuYieldCause {
    Requested,
    NodeCredit,
    SoftLease,
}

/// One worker occupancy grant. The hard job deadline stays in CpuLimits; the
/// soft lease is checked separately and never pretends to expire the job.
pub struct CpuSliceControl {
    pub node_credit: u64,
    pub soft_lease: Option<Instant>,
    pub mailbox: Arc<CpuControlMailbox>,
    observed_yield_sequence: AtomicU64,
    observed_yield_cause: AtomicU64,
    nnuec_capture: Option<crate::cpu_value::nnuec::v2::capture::NnuecActiveCapture>,
}

impl CpuSliceControl {
    pub fn new(
        node_credit: u64,
        soft_lease: Option<Instant>,
        mailbox: Arc<CpuControlMailbox>,
    ) -> Result<Self, CpuError> {
        if node_credit == 0 {
            return Err(CpuError::InvalidLimits("slice node credit is zero"));
        }
        Ok(Self {
            node_credit,
            soft_lease,
            mailbox,
            observed_yield_sequence: AtomicU64::new(0),
            observed_yield_cause: AtomicU64::new(0),
            nnuec_capture: None,
        })
    }
    pub(crate) fn with_nnuec_capture(
        mut self,
        capture: crate::cpu_value::nnuec::v2::capture::NnuecActiveCapture,
    ) -> Result<Self, CpuError> {
        let scope = self.mailbox.scope();
        if !capture.matches_slice(scope.job, scope.owner) {
            return Err(CpuError::InvalidLimits(
                "NNUEC capture belongs to another Job/owner",
            ));
        }
        self.nnuec_capture = Some(capture);
        Ok(self)
    }
    fn observe_yield(&self, cause: CpuYieldCause, sequence: u64) {
        self.observed_yield_sequence
            .store(sequence, Ordering::Release);
        self.observed_yield_cause.store(
            match cause {
                CpuYieldCause::Requested => 1,
                CpuYieldCause::NodeCredit => 2,
                CpuYieldCause::SoftLease => 3,
            },
            Ordering::Release,
        );
    }
}

#[derive(Clone, Debug)]
pub struct CpuControlledReport {
    pub report: CpuReport,
    pub yield_cause: Option<CpuYieldCause>,
    pub yield_ack_sequence: Option<u64>,
    /// This sequence only acknowledges a completed iteration publication.
    pub publish_ack_sequence: u64,
}

/// Immutable scheduling context for actual R -> P/C -> R node feedback.
/// The manager still owns question IDs, Store pins and the original ledger.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CpuReciprocalScope {
    pub job: rz_contracts::pals_adaptive::CpuJobEnvelope,
    pub owner: rz_contracts::pals_adaptive::CpuWorkerId,
    pub parent: rz_contracts::rovent::QuestionEnvelope,
    pub recipient: rz_contracts::rovent::SearchRecipient,
    pub max_questions: u16,
    pub node_interval: u64,
}

impl CpuReciprocalScope {
    fn validate(&self) -> Result<(), CpuError> {
        use rz_contracts::rovent::SearchRecipient;
        if self.job.id.authority != self.parent.id.authority
            || self.job.question != self.parent.question_digest
            || self.job.state_identity != self.parent.anchor
            || self.job.hard_deadline != self.parent.original_deadline
            || self.job.hard_deadline.clock.0 != self.job.id.authority.epoch
            || self.parent.recipient != SearchRecipient::Researcher
            || !matches!(
                self.recipient,
                SearchRecipient::Proposer | SearchRecipient::Critic
            )
            || self.job.id.sequence == 0
            || self.parent.id.sequence == 0
            || !(1..=64).contains(&self.max_questions)
            || self.node_interval == 0
            || self.job.total_native_node_credit == 0
        {
            return Err(CpuError::InvalidLimits("invalid reciprocal CPU job scope"));
        }
        self.parent
            .view
            .validate(self.job.id.authority, 512)
            .map_err(|_| CpuError::InvalidLimits("invalid reciprocal frozen view"))?;
        Ok(())
    }
}

/// Issued only by an entered native node. Private pointer authority makes a
/// copied description insufficient to resume another task or another node.
#[derive(Clone, Debug)]
pub struct CpuReciprocalRequest {
    seal: Arc<()>,
    scope: CpuReciprocalScope,
    position: Position,
    /// Actual entered moves from this R job's anchor; never a guessed PV or
    /// an accumulated played-game trace. Issued with the same opaque seal.
    relative_prefix: Vec<BoardMove>,
    ply: u16,
    remaining_depth: u16,
    ordinal: u16,
    candidates: Vec<BoardMove>,
}

impl CpuReciprocalRequest {
    pub fn scope(&self) -> &CpuReciprocalScope {
        &self.scope
    }
    pub fn position(&self) -> &Position {
        &self.position
    }
    pub fn relative_prefix(&self) -> &[BoardMove] {
        &self.relative_prefix
    }
    pub fn ply(&self) -> u16 {
        self.ply
    }
    pub fn remaining_depth(&self) -> u16 {
        self.remaining_depth
    }
    /// Local observation only; the manager separately issues the global question ID.
    pub fn ordinal(&self) -> u16 {
        self.ordinal
    }
    pub fn candidates(&self) -> &[BoardMove] {
        &self.candidates
    }
    /// Exact issued-request equality, including its private owner authority.
    /// Matching public descriptions alone cannot authorize another R parent.
    pub fn same_request(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.seal, &other.seal)
            && self.scope == other.scope
            && self.position.position_identity() == other.position.position_identity()
            && self.relative_prefix == other.relative_prefix
            && self.ply == other.ply
            && self.remaining_depth == other.remaining_depth
            && self.ordinal == other.ordinal
            && self.candidates == other.candidates
    }
}

/// Accepted neural feedback changes move ordering only. It is neither a
/// minimax bound nor a Rules proof, and cannot fabricate a completed node.
#[derive(Clone, Debug)]
pub struct CpuReciprocalReply {
    request: CpuReciprocalRequest,
    question: rz_contracts::rovent::QuestionEnvelope,
    preferred: Vec<BoardMove>,
}

impl CpuReciprocalReply {
    pub fn new(
        request: &CpuReciprocalRequest,
        question: rz_contracts::rovent::QuestionEnvelope,
        anchor: &rz_position::contracts::ContractState,
        preferred: Vec<BoardMove>,
    ) -> Result<Self, CpuError> {
        use rz_contracts::rovent::SearchRecipient;
        let parent = request.scope.parent;
        if question.parent != Some(parent.id)
            || question.id.authority != parent.id.authority
            || question.id.sequence <= parent.id.sequence
            || question.sender != SearchRecipient::Researcher
            || question.recipient != request.scope.recipient
            || question.view.authority != parent.view.authority
            || question.view.context_policy != parent.view.context_policy
            || question.view.root_state != parent.view.root_state
            || question.root_budget != parent.root_budget
            || question.original_first_move != parent.original_first_move
            || question.original_deadline != parent.original_deadline
            || (question.line_digest.is_none() && question.line_offset.0 != 0)
            || question.anchor != anchor.snapshot().identity()
            || !anchor
                .rules()
                .snapshot()
                .same_state(&request.position.snapshot())
            || question.anchor_root_ply.0
                != parent
                    .anchor_root_ply
                    .0
                    .checked_add(request.ply)
                    .ok_or(CpuError::InvalidLimits("reciprocal anchor ply overflow"))?
            || preferred.len() > request.candidates.len()
            || preferred.iter().enumerate().any(|(index, movement)| {
                !request.candidates.contains(movement) || preferred[..index].contains(movement)
            })
        {
            return Err(CpuError::ResumeMismatch(
                "reciprocal response seal, anchor or candidates mismatch",
            ));
        }
        question
            .view
            .validate(parent.id.authority, 512)
            .map_err(|_| CpuError::ResumeMismatch("reciprocal child view is invalid"))?;
        Ok(Self {
            request: request.clone(),
            question,
            preferred,
        })
    }

    pub fn question(&self) -> rz_contracts::rovent::QuestionEnvelope {
        self.question
    }
    pub fn preferred(&self) -> &[BoardMove] {
        &self.preferred
    }
}

#[derive(Clone, Debug)]
pub struct CpuReciprocalControlledReport {
    pub controlled: CpuControlledReport,
    pub request: Option<CpuReciprocalRequest>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CpuScoreScope {
    /// No complete search iteration; provisional frontier value from the
    /// selected evaluator. Training/weights provenance is recorded separately.
    FrontierOnly,
    /// Completed minimax estimate for the declared depth and qsearch profile;
    /// finite lookahead is not an exact whole-game outcome.
    CompletedIteration,
    /// Rules certified the current state, independent of search depth.
    RulesTerminal,
}

#[derive(Clone, Debug)]
pub struct CpuReport {
    pub best_move: Option<BoardMove>,
    /// Verified legal prefix. TT reuse can shorten this independently of the
    /// completed lookahead depth; length never proves full line coverage.
    pub pv: Vec<BoardMove>,
    /// Root side-to-move value. Never interpreted as calibrated centipawns.
    pub score: i32,
    pub completed_depth: u16,
    pub nodes: u64,
    pub quiescence_nodes: u64,
    pub tt_hits: u64,
    pub completion: CpuCompletion,
    pub score_scope: CpuScoreScope,
    pub profile: CpuProfile,
    pub score_provenance: &'static str,
    pub value_identity: CpuValueIdentity,
    pub search_version: &'static str,
    pub root_restricted: bool,
    pub elapsed: Duration,
    pub reused_completed_depth: u16,
    /// Historical policy resumes after a completed iteration. The explicit
    /// PausedStack policy may instead return a one-use engine-owned stack token.
    pub resume: Option<CpuResumeToken>,
}

#[derive(Clone, Copy, Debug)]
pub struct CpuIterationProgress {
    pub best_move: Option<BoardMove>,
    pub score: i32,
    pub completed_depth: u16,
    pub nodes: u64,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CpuWork {
    /// Actual positions visited by this invocation only, including qsearch.
    /// Explicit LegalSeeV1 also includes entered material-exchange positions.
    /// Reused completed depth does not re-charge the previous invocation.
    pub nodes: u64,
    pub quiescence_nodes: u64,
    pub tt_hits: u64,
}

/// Actual PausedStack owner facts, accumulated from construction until owner
/// destruction. Consumers copy the latest snapshot rather than summing it.
/// New-game resets do not reset the owner ID, event sequence or totals. A bare
/// CpuEngine cannot prove terminal admission closure or final owner release.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CpuPausedStackSnapshot {
    pub owner_id: u64,
    pub event_sequence: u64,
    pub tokens_created: u64,
    pub tokens_resumed: u64,
    pub tokens_invalidated: u64,
    pub tokens_retained: u32,
    pub tokens_peak: u32,
    pub bytes_current: u64,
    pub bytes_peak: u64,
    /// Current immutable owner ceiling, enforced by the retained traversal.
    /// This is a configured limit, not an observed RSS or allocation peak.
    pub bytes_limit_current: u64,
    pub stale_context_attempts: u64,
    pub stale_context_rejections: u64,
    /// Previously consumed nodes present in a resumed invocation's initial
    /// actual work counters. These counters start empty in the owned machine;
    /// retained cumulative diagnostics are never installed into Control.
    pub replayed_consumed_work: u64,
    pub admission_closed: bool,
    pub owner_released: bool,
    /// False after any owner-ID/counter overflow or failed ownership invariant.
    /// Partial facts remain available, but cannot authorize known release.
    pub complete: bool,
}

impl CpuPausedStackSnapshot {
    fn new() -> Self {
        // Exhaustion stays unknown; never wrap or reuse an issued owner ID.
        let owner_id = NEXT_PAUSED_STACK_OWNER_ID
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |next| {
                next.checked_add(1)
            })
            .unwrap_or(0);
        Self {
            owner_id,
            event_sequence: 0,
            tokens_created: 0,
            tokens_resumed: 0,
            tokens_invalidated: 0,
            tokens_retained: 0,
            tokens_peak: 0,
            bytes_current: 0,
            bytes_peak: 0,
            bytes_limit_current: CPU_PAUSED_STACK_MAX_BYTES as u64,
            stale_context_attempts: 0,
            stale_context_rejections: 0,
            replayed_consumed_work: 0,
            admission_closed: false,
            owner_released: false,
            complete: owner_id != 0,
        }
    }

    fn add(counter: &mut u64, amount: u64, complete: &mut bool) {
        if let Some(next) = counter.checked_add(amount) {
            *counter = next;
        } else {
            *complete = false;
        }
    }

    fn event(&mut self) {
        Self::add(&mut self.event_sequence, 1, &mut self.complete);
    }

    fn created(&mut self, bytes: usize) {
        self.event();
        Self::add(&mut self.tokens_created, 1, &mut self.complete);
        self.tokens_peak = self.tokens_peak.max(1);
        match u64::try_from(bytes) {
            Ok(bytes) => self.bytes_peak = self.bytes_peak.max(bytes),
            Err(_) => self.complete = false,
        }
    }

    fn resumed(&mut self) {
        self.event();
        Self::add(&mut self.tokens_resumed, 1, &mut self.complete);
    }

    fn invalidated(&mut self) {
        self.event();
        Self::add(&mut self.tokens_invalidated, 1, &mut self.complete);
    }

    fn rejected_context(&mut self) {
        self.event();
        Self::add(&mut self.stale_context_attempts, 1, &mut self.complete);
        Self::add(&mut self.stale_context_rejections, 1, &mut self.complete);
    }

    fn resumed_initial_work(&mut self, work: CpuWork) {
        Self::add(
            &mut self.replayed_consumed_work,
            work.nodes,
            &mut self.complete,
        );
    }
}

#[derive(Clone, Debug)]
pub struct CpuResumeToken {
    root: PositionIdentity,
    root_moves: Option<Vec<BoardMove>>,
    profile: CpuProfile,
    quiescence_ply: u16,
    value_identity: CpuValueIdentity,
    search_conditions: String,
    completed_depth: u16,
    score: i32,
    pv: Vec<BoardMove>,
    paused_stack: Option<PausedStackToken>,
}

#[derive(Clone, Debug)]
struct PausedStackToken {
    owner: Arc<()>,
    serial: u64,
    cumulative: CpuWork,
    retained_bytes: usize,
}

impl CpuResumeToken {
    pub fn is_paused_stack(&self) -> bool {
        self.paused_stack.is_some()
    }

    /// Previously consumed work is diagnostic only; report counters always
    /// cover newly consumed work from the current invocation.
    pub fn cumulative_work(&self) -> Option<CpuWork> {
        self.paused_stack.as_ref().map(|token| token.cumulative)
    }

    pub fn paused_stack_bytes(&self) -> Option<usize> {
        self.paused_stack.as_ref().map(|token| token.retained_bytes)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CpuError {
    InvalidConfig(&'static str),
    InvalidLimits(&'static str),
    InvalidRootMoves(&'static str),
    ResumeMismatch(&'static str),
    PausedStackMemoryLimit {
        retained_bytes: usize,
        maximum_bytes: usize,
    },
    Unsupported(&'static str),
    Allocation,
    Rules(PositionError),
    Value(CpuValueError),
}

impl std::fmt::Display for CpuError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidConfig(message) => write!(f, "CPU configuration: {message}"),
            Self::InvalidLimits(message) => write!(f, "CPU limits: {message}"),
            Self::InvalidRootMoves(message) => write!(f, "CPU root moves: {message}"),
            Self::ResumeMismatch(message) => write!(f, "CPU resume: {message}"),
            Self::PausedStackMemoryLimit {
                retained_bytes,
                maximum_bytes,
            } => write!(
                f,
                "CPU paused stack retains {retained_bytes} bytes, above {maximum_bytes} bytes"
            ),
            Self::Unsupported(message) => write!(f, "CPU capability unavailable: {message}"),
            Self::Allocation => f.write_str("CPU TT allocation refused"),
            Self::Rules(error) => write!(f, "CPU rules failure: {error}"),
            Self::Value(error) => write!(f, "CPU value failure: {error}"),
        }
    }
}

impl std::error::Error for CpuError {}

impl CpuError {
    /// No value is implied. Only an exact evaluator control error can name
    /// this no-score terminal; other CPU/Rules/numerical failures stay errors.
    pub fn value_authority_stop(&self) -> Option<CpuCompletion> {
        use crate::cpu_value::CpuValueAuthorityRevocation;
        match self {
            Self::Value(CpuValueError::AuthorityRevoked(
                CpuValueAuthorityRevocation::Cancelled,
            )) => Some(CpuCompletion::Canceled),
            Self::Value(CpuValueError::AuthorityRevoked(CpuValueAuthorityRevocation::Expired)) => {
                Some(CpuCompletion::Deadline)
            }
            _ => None,
        }
    }
}

impl From<PositionError> for CpuError {
    fn from(value: PositionError) -> Self {
        Self::Rules(value)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Bound {
    Exact,
    Lower,
    Upper,
}

#[derive(Clone, Debug)]
struct TtEntry {
    hash: u64,
    identity: PositionIdentity,
    profile: CpuProfile,
    value_identity: Arc<CpuValueIdentity>,
    value_context: CpuValueCacheContext,
    depth: u16,
    /// Mate distance normalized to this node, not the root that stored it.
    score: i32,
    bound: Bound,
    best_move: Option<BoardMove>,
}

pub struct CpuEngine {
    config: CpuConfig,
    ordering: CpuOrderingPolicy,
    root_tie_policy: CpuRootTiePolicy,
    tt: Vec<Option<TtEntry>>,
    /// Saturating side/from/to quiet history; fixed 32 KiB.
    history: Box<[[[i32; 64]; 64]; 2]>,
    evaluator: Arc<dyn CpuValueEvaluator>,
    value_identity: Arc<CpuValueIdentity>,
    last_attempt_work: Option<CpuWork>,
    resume_policy: CpuResumePolicy,
    paused_stack_max_bytes: usize,
    /// Owner configuration remains sealed across new-game cleanup. Token
    /// serials and last-attempt observations deliberately reset at that boundary.
    paused_stack_limit_sealed: bool,
    paused: Option<stack::PausedTask>,
    stack_owner: Arc<()>,
    stack_serial: u64,
    paused_stack_telemetry: Option<CpuPausedStackSnapshot>,
    /// Opt-in one-shot work owns another bounded TT/history, never the paused
    /// traversal's owner. The side engine cannot retain an interrupted stack.
    paused_side_scope: Option<Box<CpuEngine>>,
    side_scope_observation: Option<CpuSideScopeSnapshot>,
}

/// Actual one-shot attempts and fixed owner storage. Fixed bytes are a
/// reservation declaration, not peak RSS or the 8MiB paused-frame charge.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CpuSideScopeSnapshot {
    pub attempts: u64,
    pub work: CpuWork,
    pub fixed_reserved_bytes: usize,
    pub complete: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CpuCapabilities {
    pub max_depth: u16,
    pub max_prefix_plies: usize,
    pub max_root_moves: usize,
    pub root_moves: bool,
    pub divergence: bool,
    pub completed_iteration_resume: bool,
    pub selective_reductions: bool,
}

/// Startup-selected own CPU checker. Value identity and search implementation
/// identity are independent immutable namespaces. Consumers validate returned
/// report identities and may not transfer TT or opaque resume tokens between
/// implementations. Unsupported capabilities fail explicitly; no fallback.
pub trait CpuSearcher: Send {
    fn config(&self) -> &CpuConfig;
    fn value_identity(&self) -> &CpuValueIdentity;
    fn search_identity(&self) -> &'static str;
    /// Exact immutable implementation/configuration declaration. Custom
    /// checkers must declare their own conditions rather than inheriting PVS.
    fn search_conditions(&self) -> String;
    fn capabilities(&self) -> CpuCapabilities;
    fn resume_policy(&self) -> CpuResumePolicy {
        CpuResumePolicy::CompletedIteration
    }
    /// Fresh side tasks can run without replacing the single paused owner.
    /// This is an explicit scheduling capability, not ordinary PausedStack.
    fn preserves_paused_on_fresh_tasks(&self) -> bool {
        false
    }
    fn paused_side_scope_snapshot(&self) -> Option<CpuSideScopeSnapshot> {
        None
    }
    /// Retained traversal bytes, excluding the already configured TT/history
    /// owner. None means there is no live paused traversal.
    fn paused_stack_bytes(&self) -> Option<usize> {
        None
    }
    /// None means this implementation does not observe an actual PausedStack
    /// owner. Queries neither consume a token nor create an accounting event.
    fn paused_stack_snapshot(&self) -> Option<CpuPausedStackSnapshot> {
        None
    }
    /// Read-only ownership query, without executing or charging work. For a
    /// PausedStack token this must prove that its actual frame owner is live.
    /// Unknown custom ownership is not reported as current.
    fn token_is_current(&self, _token: &CpuResumeToken) -> bool {
        false
    }
    /// Explicit support for disposing only this owner's paused traversal.
    /// Custom PausedStack owners must opt in; default false is not proof that
    /// they have no retained frames.
    fn supports_paused_stack_discard(&self) -> bool {
        false
    }
    /// Invalidate the live paused traversal, preserving TT, ordering history,
    /// and the last observed work. Returns true only when frames were dropped.
    /// This is not a new search attempt or a new-game reset.
    fn discard_paused_stack(&mut self) -> bool {
        false
    }
    /// None means work was not observed. Never substitute invented zero work.
    /// The default preserves unknown accounting for custom implementations.
    fn last_attempt_work(&self) -> Option<CpuWork> {
        None
    }
    fn clear(&mut self);
    fn analyze(
        &mut self,
        position: &Position,
        limits: CpuLimits,
        cancel: &AtomicBool,
    ) -> Result<CpuReport, CpuError>;
    fn analyze_root_moves(
        &mut self,
        _position: &Position,
        _moves: &[BoardMove],
        _limits: CpuLimits,
        _cancel: &AtomicBool,
    ) -> Result<CpuReport, CpuError> {
        Err(CpuError::Unsupported("root moves"))
    }
    fn analyze_divergence(
        &mut self,
        _position: &Position,
        _prefix: &[BoardMove],
        _limits: CpuLimits,
        _cancel: &AtomicBool,
    ) -> Result<CpuReport, CpuError> {
        Err(CpuError::Unsupported("divergence"))
    }
    fn resume(
        &mut self,
        _position: &Position,
        _token: &CpuResumeToken,
        _limits: CpuLimits,
        _cancel: &AtomicBool,
    ) -> Result<CpuReport, CpuError> {
        Err(CpuError::Unsupported("completed iteration resume"))
    }
}

impl<C: CpuSearcher + ?Sized> CpuSearcher for Box<C> {
    fn config(&self) -> &CpuConfig {
        (**self).config()
    }
    fn value_identity(&self) -> &CpuValueIdentity {
        (**self).value_identity()
    }
    fn search_identity(&self) -> &'static str {
        (**self).search_identity()
    }
    fn search_conditions(&self) -> String {
        (**self).search_conditions()
    }
    fn capabilities(&self) -> CpuCapabilities {
        (**self).capabilities()
    }
    fn resume_policy(&self) -> CpuResumePolicy {
        (**self).resume_policy()
    }
    fn preserves_paused_on_fresh_tasks(&self) -> bool {
        (**self).preserves_paused_on_fresh_tasks()
    }
    fn paused_side_scope_snapshot(&self) -> Option<CpuSideScopeSnapshot> {
        (**self).paused_side_scope_snapshot()
    }
    fn paused_stack_bytes(&self) -> Option<usize> {
        (**self).paused_stack_bytes()
    }
    fn paused_stack_snapshot(&self) -> Option<CpuPausedStackSnapshot> {
        (**self).paused_stack_snapshot()
    }
    fn token_is_current(&self, token: &CpuResumeToken) -> bool {
        (**self).token_is_current(token)
    }
    fn supports_paused_stack_discard(&self) -> bool {
        (**self).supports_paused_stack_discard()
    }
    fn discard_paused_stack(&mut self) -> bool {
        (**self).discard_paused_stack()
    }
    fn last_attempt_work(&self) -> Option<CpuWork> {
        (**self).last_attempt_work()
    }
    fn clear(&mut self) {
        (**self).clear();
    }
    fn analyze(
        &mut self,
        p: &Position,
        l: CpuLimits,
        c: &AtomicBool,
    ) -> Result<CpuReport, CpuError> {
        (**self).analyze(p, l, c)
    }
    fn analyze_root_moves(
        &mut self,
        p: &Position,
        m: &[BoardMove],
        l: CpuLimits,
        c: &AtomicBool,
    ) -> Result<CpuReport, CpuError> {
        (**self).analyze_root_moves(p, m, l, c)
    }
    fn analyze_divergence(
        &mut self,
        p: &Position,
        m: &[BoardMove],
        l: CpuLimits,
        c: &AtomicBool,
    ) -> Result<CpuReport, CpuError> {
        (**self).analyze_divergence(p, m, l, c)
    }
    fn resume(
        &mut self,
        p: &Position,
        t: &CpuResumeToken,
        l: CpuLimits,
        c: &AtomicBool,
    ) -> Result<CpuReport, CpuError> {
        (**self).resume(p, t, l, c)
    }
}

impl CpuSearcher for CpuEngine {
    fn config(&self) -> &CpuConfig {
        CpuEngine::config(self)
    }
    fn value_identity(&self) -> &CpuValueIdentity {
        CpuEngine::value_identity(self)
    }
    fn search_identity(&self) -> &'static str {
        CpuEngine::search_identity(self)
    }
    fn search_conditions(&self) -> String {
        CpuEngine::search_conditions(self)
    }
    fn capabilities(&self) -> CpuCapabilities {
        CpuCapabilities {
            max_depth: self.config.max_depth,
            max_prefix_plies: 64,
            max_root_moves: MAX_ROOT_MOVES,
            root_moves: true,
            divergence: true,
            completed_iteration_resume: true,
            selective_reductions: false,
        }
    }
    fn resume_policy(&self) -> CpuResumePolicy {
        CpuEngine::resume_policy(self)
    }
    fn preserves_paused_on_fresh_tasks(&self) -> bool {
        CpuEngine::preserves_paused_on_fresh_tasks(self)
    }
    fn paused_side_scope_snapshot(&self) -> Option<CpuSideScopeSnapshot> {
        CpuEngine::paused_side_scope_snapshot(self)
    }
    fn paused_stack_bytes(&self) -> Option<usize> {
        CpuEngine::paused_stack_bytes(self)
    }
    fn paused_stack_snapshot(&self) -> Option<CpuPausedStackSnapshot> {
        CpuEngine::paused_stack_snapshot(self)
    }
    fn token_is_current(&self, token: &CpuResumeToken) -> bool {
        CpuEngine::token_is_current(self, token)
    }
    fn supports_paused_stack_discard(&self) -> bool {
        CpuEngine::supports_paused_stack_discard(self)
    }
    fn discard_paused_stack(&mut self) -> bool {
        CpuEngine::discard_paused_stack(self)
    }
    fn last_attempt_work(&self) -> Option<CpuWork> {
        CpuEngine::last_attempt_work(self)
    }
    fn clear(&mut self) {
        CpuEngine::clear(self);
    }
    fn analyze(
        &mut self,
        p: &Position,
        l: CpuLimits,
        c: &AtomicBool,
    ) -> Result<CpuReport, CpuError> {
        CpuEngine::analyze(self, p, l, c)
    }
    fn analyze_root_moves(
        &mut self,
        p: &Position,
        m: &[BoardMove],
        l: CpuLimits,
        c: &AtomicBool,
    ) -> Result<CpuReport, CpuError> {
        CpuEngine::analyze_root_moves(self, p, m, l, c)
    }
    fn analyze_divergence(
        &mut self,
        p: &Position,
        m: &[BoardMove],
        l: CpuLimits,
        c: &AtomicBool,
    ) -> Result<CpuReport, CpuError> {
        CpuEngine::analyze_divergence(self, p, m, l, c)
    }
    fn resume(
        &mut self,
        p: &Position,
        t: &CpuResumeToken,
        l: CpuLimits,
        c: &AtomicBool,
    ) -> Result<CpuReport, CpuError> {
        CpuEngine::resume(self, p, t, l, c)
    }
}

struct Control<'a> {
    limits: CpuLimits,
    cancellation: &'a AtomicBool,
    cooperative: Option<&'a CpuSliceControl>,
    nodes: u64,
    quiescence_nodes: u64,
    tt_hits: u64,
    value: CpuAccumulator,
}

#[derive(Debug)]
enum Abort {
    Stop(CpuCompletion),
    /// A value callback may revoke after the safe control boundary. Its
    /// unfinished frame is disposed, never offered as a resumable checkpoint.
    ValueAuthorityStop(CpuCompletion),
    Error(CpuError),
}

impl From<PositionError> for Abort {
    fn from(error: PositionError) -> Self {
        Self::Error(CpuError::Rules(error))
    }
}

impl From<CpuValueError> for Abort {
    fn from(error: CpuValueError) -> Self {
        let error = CpuError::Value(error);
        match error.value_authority_stop() {
            Some(completion) => Self::ValueAuthorityStop(completion),
            None => Self::Error(error),
        }
    }
}

struct NodeValue {
    score: i32,
    pv: Vec<BoardMove>,
}

impl CpuEngine {
    pub fn new(config: CpuConfig) -> Result<Self, CpuError> {
        Self::with_evaluator(config, Arc::new(BootstrapCpuValue::default()))
    }

    /// Explicit ordering-policy opt-in; existing `new` remains legacy.
    pub fn with_ordering(config: CpuConfig, ordering: CpuOrderingPolicy) -> Result<Self, CpuError> {
        Self::with_evaluator_and_ordering(config, Arc::new(BootstrapCpuValue::default()), ordering)
    }

    pub fn with_evaluator(
        config: CpuConfig,
        evaluator: Arc<dyn CpuValueEvaluator>,
    ) -> Result<Self, CpuError> {
        Self::with_evaluator_and_ordering(config, evaluator, CpuOrderingPolicy::LegacyMvvLvaV1)
    }

    pub fn with_evaluator_and_ordering(
        config: CpuConfig,
        evaluator: Arc<dyn CpuValueEvaluator>,
        ordering: CpuOrderingPolicy,
    ) -> Result<Self, CpuError> {
        Self::with_evaluator_ordering_and_resume_policy(
            config,
            evaluator,
            ordering,
            CpuResumePolicy::CompletedIteration,
        )
    }

    pub fn with_resume_policy(
        config: CpuConfig,
        resume_policy: CpuResumePolicy,
    ) -> Result<Self, CpuError> {
        Self::with_evaluator_ordering_and_resume_policy(
            config,
            Arc::new(BootstrapCpuValue::default()),
            CpuOrderingPolicy::LegacyMvvLvaV1,
            resume_policy,
        )
    }

    pub fn with_evaluator_ordering_and_resume_policy(
        config: CpuConfig,
        evaluator: Arc<dyn CpuValueEvaluator>,
        ordering: CpuOrderingPolicy,
        resume_policy: CpuResumePolicy,
    ) -> Result<Self, CpuError> {
        Self::with_evaluator_ordering_resume_and_tie_policy(
            config,
            evaluator,
            ordering,
            resume_policy,
            CpuRootTiePolicy::LegacyFirstSearched,
        )
    }

    /// Select the root tie semantics once, before any TT, task or token exists.
    pub fn with_evaluator_ordering_resume_and_tie_policy(
        config: CpuConfig,
        evaluator: Arc<dyn CpuValueEvaluator>,
        ordering: CpuOrderingPolicy,
        resume_policy: CpuResumePolicy,
        root_tie_policy: CpuRootTiePolicy,
    ) -> Result<Self, CpuError> {
        config.tt_allocation_bytes()?;
        evaluator.identity().validate().map_err(CpuError::Value)?;
        if config.max_depth == 0 || config.max_depth > MAX_DEPTH {
            return Err(CpuError::InvalidConfig("depth must be in 1..=64"));
        }
        if config.quiescence_ply > MAX_QUIESCENCE_PLY {
            return Err(CpuError::InvalidConfig("quiescence ply must be in 0..=32"));
        }
        let mut tt = Vec::new();
        tt.try_reserve_exact(config.tt_entries)
            .map_err(|_| CpuError::Allocation)?;
        tt.resize_with(config.tt_entries, || None);
        Ok(Self {
            config,
            ordering,
            root_tie_policy,
            tt,
            history: Box::new([[[0; 64]; 64]; 2]),
            value_identity: Arc::new(evaluator.identity().clone()),
            evaluator,
            last_attempt_work: None,
            resume_policy,
            paused_stack_max_bytes: CPU_PAUSED_STACK_MAX_BYTES,
            paused_stack_limit_sealed: false,
            paused: None,
            stack_owner: Arc::new(()),
            stack_serial: 0,
            paused_stack_telemetry: (resume_policy == CpuResumePolicy::PausedStack)
                .then(CpuPausedStackSnapshot::new),
            paused_side_scope: None,
            side_scope_observation: None,
        })
    }

    /// Separate S profile: hold one primary PausedStack while other fresh
    /// questions execute with a distinct CompletedIteration TT/history owner.
    /// Frozen evaluator weights are shared; mutable TT/ordering are never
    /// copied, transferred or shared across scopes. Existing constructors keep
    /// the fresh-analysis replacement contract.
    pub fn with_paused_side_scope(
        config: CpuConfig,
        evaluator: Arc<dyn CpuValueEvaluator>,
        ordering: CpuOrderingPolicy,
    ) -> Result<Self, CpuError> {
        Self::with_paused_side_scope_and_tie_policy(
            config,
            evaluator,
            ordering,
            CpuRootTiePolicy::LegacyFirstSearched,
        )
    }

    /// The isolated side owner and retained primary share the same immutable
    /// root tie semantics, while their TT/history owners remain separate.
    pub fn with_paused_side_scope_and_tie_policy(
        config: CpuConfig,
        evaluator: Arc<dyn CpuValueEvaluator>,
        ordering: CpuOrderingPolicy,
        root_tie_policy: CpuRootTiePolicy,
    ) -> Result<Self, CpuError> {
        let fixed_reserved_bytes = Self::side_scope_allocation_bytes(&config)?;
        let side = Self::with_evaluator_ordering_resume_and_tie_policy(
            config.clone(),
            evaluator.clone(),
            ordering,
            CpuResumePolicy::CompletedIteration,
            root_tie_policy,
        )?;
        let mut primary = Self::with_evaluator_ordering_resume_and_tie_policy(
            config,
            evaluator,
            ordering,
            CpuResumePolicy::PausedStack,
            root_tie_policy,
        )?;
        primary.paused_side_scope = Some(Box::new(side));
        primary.side_scope_observation = Some(CpuSideScopeSnapshot {
            attempts: 0,
            work: CpuWork::default(),
            fixed_reserved_bytes,
            complete: true,
        });
        Ok(primary)
    }

    fn side_scope_allocation_bytes(config: &CpuConfig) -> Result<usize, CpuError> {
        usize::try_from(config.tt_allocation_bytes()?)
            .map_err(|_| CpuError::Allocation)?
            .checked_add(std::mem::size_of::<[[[i32; 64]; 64]; 2]>())
            .and_then(|bytes| bytes.checked_add(std::mem::size_of::<Self>()))
            .ok_or(CpuError::Allocation)
    }

    /// Reserved fixed side-scope storage, not a measured peak. Active recursive
    /// frames, allocator overhead and the primary 8MiB pause charge are separate.
    pub fn paused_side_scope_reserved_bytes(&self) -> Option<usize> {
        self.paused_side_scope
            .as_ref()
            .and_then(|side| Self::side_scope_allocation_bytes(side.config()).ok())
    }

    pub fn preserves_paused_on_fresh_tasks(&self) -> bool {
        self.paused_side_scope.is_some()
    }

    pub fn paused_side_scope_snapshot(&self) -> Option<CpuSideScopeSnapshot> {
        self.side_scope_observation
    }

    pub fn config(&self) -> &CpuConfig {
        &self.config
    }

    pub fn ordering_policy(&self) -> CpuOrderingPolicy {
        self.ordering
    }

    pub fn root_tie_policy(&self) -> CpuRootTiePolicy {
        self.root_tie_policy
    }

    pub fn search_identity(&self) -> &'static str {
        if self.root_tie_policy == CpuRootTiePolicy::RulesOrderConfirmed {
            if self.paused_side_scope.is_some() {
                return match self.ordering {
                    CpuOrderingPolicy::LegacyMvvLvaV1 => CPU_ROOT_TIE_SIDE_SEARCH_VERSION,
                    CpuOrderingPolicy::LegalSeeV1 => CPU_ROOT_TIE_SIDE_SEE_SEARCH_VERSION,
                };
            }
            return match (self.ordering, self.resume_policy) {
                (CpuOrderingPolicy::LegacyMvvLvaV1, CpuResumePolicy::CompletedIteration) => {
                    CPU_ROOT_TIE_SEARCH_VERSION
                }
                (CpuOrderingPolicy::LegalSeeV1, CpuResumePolicy::CompletedIteration) => {
                    CPU_ROOT_TIE_SEE_SEARCH_VERSION
                }
                (CpuOrderingPolicy::LegacyMvvLvaV1, CpuResumePolicy::PausedStack) => {
                    CPU_ROOT_TIE_PAUSED_SEARCH_VERSION
                }
                (CpuOrderingPolicy::LegalSeeV1, CpuResumePolicy::PausedStack) => {
                    CPU_ROOT_TIE_PAUSED_SEE_SEARCH_VERSION
                }
            };
        }
        if self.paused_side_scope.is_some() {
            return match self.ordering {
                CpuOrderingPolicy::LegacyMvvLvaV1 => CPU_PAUSED_SIDE_SCOPE_SEARCH_VERSION,
                CpuOrderingPolicy::LegalSeeV1 => CPU_PAUSED_SIDE_SCOPE_SEE_SEARCH_VERSION,
            };
        }
        match (self.ordering, self.resume_policy) {
            (CpuOrderingPolicy::LegacyMvvLvaV1, CpuResumePolicy::CompletedIteration) => {
                CPU_SEARCH_VERSION
            }
            (CpuOrderingPolicy::LegalSeeV1, CpuResumePolicy::CompletedIteration) => {
                CPU_SEE_SEARCH_VERSION
            }
            (CpuOrderingPolicy::LegacyMvvLvaV1, CpuResumePolicy::PausedStack) => {
                CPU_PAUSED_STACK_SEARCH_VERSION
            }
            (CpuOrderingPolicy::LegalSeeV1, CpuResumePolicy::PausedStack) => {
                CPU_PAUSED_STACK_SEE_SEARCH_VERSION
            }
        }
    }

    pub fn resume_policy(&self) -> CpuResumePolicy {
        self.resume_policy
    }

    /// Configure the actual retained-frame ceiling before the first search.
    /// The legacy constructors keep the 8MiB ceiling. This never transfers or
    /// recreates the TT, ordering state, value namespace or token owner.
    /// A held side scope receives the same ceiling before either owner starts.
    pub fn set_paused_stack_max_bytes(&mut self, maximum_bytes: usize) -> Result<(), CpuError> {
        if !(1..=CPU_PAUSED_STACK_MAX_BYTES).contains(&maximum_bytes) {
            return Err(CpuError::InvalidConfig(
                "paused stack limit must be in 1..=8388608 bytes",
            ));
        }
        if maximum_bytes == self.paused_stack_max_bytes {
            return Ok(());
        }
        if self.paused_stack_limit_sealed
            || self.paused.is_some()
            || self.last_attempt_work.is_some()
            || self.stack_serial != 0
            || self.paused_side_scope.as_ref().is_some_and(|side| {
                side.paused_stack_limit_sealed
                    || side.paused.is_some()
                    || side.last_attempt_work.is_some()
                    || side.stack_serial != 0
            })
        {
            return Err(CpuError::InvalidConfig(
                "paused stack limit is immutable after either CPU scope starts",
            ));
        }
        if let Some(side) = self.paused_side_scope.as_mut() {
            side.paused_stack_max_bytes = maximum_bytes;
            if let Some(telemetry) = side.paused_stack_telemetry.as_mut() {
                telemetry.bytes_limit_current = maximum_bytes as u64;
            }
        }
        self.paused_stack_max_bytes = maximum_bytes;
        if let Some(telemetry) = self.paused_stack_telemetry.as_mut() {
            telemetry.bytes_limit_current = maximum_bytes as u64;
        }
        Ok(())
    }

    pub fn paused_stack_max_bytes(&self) -> usize {
        self.paused_stack_max_bytes
    }

    pub fn paused_stack_bytes(&self) -> Option<usize> {
        self.paused.as_ref().map(|task| task.retained_bytes())
    }

    /// Read actual frame retention and lifetime totals without changing them.
    /// Historical token bytes and completed iteration handles are not owners.
    pub fn paused_stack_snapshot(&self) -> Option<CpuPausedStackSnapshot> {
        let mut snapshot = self.paused_stack_telemetry?;
        snapshot.tokens_retained = u32::from(self.paused.is_some());
        snapshot.bytes_current = match self.paused_stack_bytes() {
            Some(bytes) => match u64::try_from(bytes) {
                Ok(bytes) => bytes,
                Err(_) => {
                    snapshot.complete = false;
                    0
                }
            },
            None => 0,
        };
        snapshot.complete &= snapshot.owner_id != 0
            && snapshot.bytes_limit_current == self.paused_stack_max_bytes as u64
            && snapshot.bytes_current <= snapshot.bytes_limit_current
            && snapshot
                .tokens_resumed
                .checked_add(snapshot.tokens_invalidated)
                .and_then(|consumed| consumed.checked_add(u64::from(snapshot.tokens_retained)))
                == Some(snapshot.tokens_created);
        // Terminal closure belongs to the owning checker bridge, not an idle
        // synchronous engine or a Node/Task lifecycle projection.
        snapshot.admission_closed = false;
        snapshot.owner_released = false;
        Some(snapshot)
    }

    /// Paused tokens require the one live engine-owned traversal. Historical
    /// completed tokens have no frame owner and report namespace compatibility.
    pub fn token_is_current(&self, token: &CpuResumeToken) -> bool {
        if token.is_paused_stack() {
            return stack::token_is_current(self, token);
        }
        token.completed_depth > 0
            && token.completed_depth <= self.config.max_depth
            && token.profile == self.config.profile
            && token.quiescence_ply == self.config.quiescence_ply
            && token.value_identity == *self.value_identity
            && self.evaluator.identity() == self.value_identity.as_ref()
            && token.search_conditions == self.search_conditions()
    }

    pub fn supports_paused_stack_discard(&self) -> bool {
        true
    }

    /// Drop only the actual paused traversal and its one-use token owner. TT,
    /// ordering history, and the measured last attempt remain available.
    pub fn discard_paused_stack(&mut self) -> bool {
        let retained = self.paused.is_some();
        self.invalidate_paused_stack();
        retained
    }

    /// Full immutable value namespace used by this engine's TT and resume.
    pub fn value_identity(&self) -> &CpuValueIdentity {
        &self.value_identity
    }
    /// Preserve an actual evaluator's immutable contextual registration across
    /// known owner recovery. This does not recreate a revoked view authority.
    pub fn job_binding_digest(&self) -> Result<Option<rz_contracts::Digest>, CpuError> {
        self.validate_frozen_value_namespace()?;
        self.evaluator.job_binding_digest().map_err(CpuError::Value)
    }
    /// Known joined-owner cleanup keeps original immutable value identity but
    /// intentionally does not require a live cancellation/deadline authority.
    pub(crate) fn nnuec_retirement_owner(
        &self,
    ) -> Result<Option<crate::cpu_value::nnuec::v2::retirement::NnuecValueRetirementOwner>, CpuError>
    {
        self.validate_frozen_value_namespace()?;
        let owner = self.evaluator.nnuec_retirement_owner();
        if owner
            .as_ref()
            .is_some_and(|owner| owner.model_identity() != self.value_identity.as_ref())
        {
            return Err(CpuError::Value(
                crate::cpu_value::CpuValueError::StateMismatch,
            ));
        }
        Ok(owner)
    }

    pub fn search_conditions(&self) -> String {
        let base = match self.root_tie_policy {
            CpuRootTiePolicy::LegacyFirstSearched => CPU_SEARCH_CONDITIONS.to_owned(),
            CpuRootTiePolicy::RulesOrderConfirmed => CPU_SEARCH_CONDITIONS.replace(
                "ties:Rules-order",
                "root-ties:Rules-order-confirmed;internal-pv-ties:search-order;root-tie-confirmation:scout-bound-research",
            ),
        };
        let legacy = format!(
            "{base};profile={};max_depth={};q_plies={};tt_entries={}",
            self.config.profile.identity(),
            self.config.max_depth,
            self.config.quiescence_ply,
            self.config.tt_entries
        );
        let conditions = match self.ordering {
            CpuOrderingPolicy::LegacyMvvLvaV1 => legacy,
            CpuOrderingPolicy::LegalSeeV1 => format!(
                "{legacy};ordering={};exchange:all-Rules-legal-recaptures-on-target,optional-stop-zero,material-only,P100-N320-B330-R500-Q900-K0,no-pruning,max-plies32,max-positions-per-order{MAX_SEE_POSITIONS_PER_ORDER},typed-fail-on-exhaustion;ordering-priority:TT,nonnegative-exchange,quiet-history,negative-exchange;node-work:search+qsearch+exchange;exchange-cancel-deadline:original",
                self.ordering.identity()
            ),
        };
        let conditions = match self.resume_policy {
            CpuResumePolicy::CompletedIteration => conditions,
            CpuResumePolicy::PausedStack => format!(
                "{conditions};resume={};paused-owner:single-use-root-full-history-restriction-profile-value-conditions-target-depth;paused-memory:max-1-task-{}-bytes;consumed-work:no-replay-no-recharge;{}",
                self.resume_policy.identity(),
                self.paused_stack_max_bytes,
                if self.paused_side_scope.is_some() {
                    "cancel-newgame:invalidate;fresh-analysis:held-parent-side-scope"
                } else {
                    "cancel-newgame-fresh-analysis:invalidate"
                }
            ),
        };
        if let Some(bytes) = self.paused_side_scope_reserved_bytes() {
            format!(
                "{conditions};fresh-task-scope:held-parent-one-shot-v1;side-resume:none;side-TT-ordering:separate-owner;side-fixed-reserved-bytes={bytes};cancel-newgame:invalidate-all;side-and-primary-work:shared-caller-node-deadline-budget"
            )
        } else {
            conditions
        }
    }

    /// Actual counters for the last normally returned attempt, including
    /// errors. Before the first attempt or after clear there is no observation.
    /// An admission error has observed zero searched positions, not a result.
    pub fn last_attempt_work(&self) -> Option<CpuWork> {
        self.last_attempt_work
    }

    /// New game ownership boundary; no TT/history evidence crosses this call.
    pub fn clear(&mut self) {
        self.invalidate_paused_stack();
        self.tt.iter_mut().for_each(|entry| *entry = None);
        *self.history = [[[0; 64]; 64]; 2];
        self.last_attempt_work = None;
        if let Some(side) = self.paused_side_scope.as_mut() {
            side.clear();
        }
    }

    fn invalidate_paused_stack(&mut self) {
        if self.paused.take().is_some()
            && let Some(telemetry) = self.paused_stack_telemetry.as_mut()
        {
            telemetry.invalidated();
        }
        // Tokens retain their old Arc; a fresh allocation cannot share its
        // identity while an old token remains alive.
        self.stack_owner = Arc::new(());
        self.stack_serial = 0;
    }

    fn observe_paused_stack_created(&mut self) {
        if let Some(task) = self.paused.as_ref()
            && let Some(telemetry) = self.paused_stack_telemetry.as_mut()
        {
            telemetry.created(task.retained_bytes());
        }
    }

    fn observe_paused_stack_resumed(&mut self) {
        if let Some(telemetry) = self.paused_stack_telemetry.as_mut() {
            telemetry.resumed();
        }
    }

    fn observe_paused_stack_rejection(&mut self, resume: Option<&CpuResumeToken>) {
        if resume.is_some_and(CpuResumeToken::is_paused_stack)
            && let Some(telemetry) = self.paused_stack_telemetry.as_mut()
        {
            telemetry.rejected_context();
        }
    }

    fn observe_paused_stack_initial_work(&mut self, work: CpuWork) {
        if let Some(telemetry) = self.paused_stack_telemetry.as_mut() {
            telemetry.resumed_initial_work(work);
        }
    }

    pub fn analyze(
        &mut self,
        position: &Position,
        limits: CpuLimits,
        cancellation: &AtomicBool,
    ) -> Result<CpuReport, CpuError> {
        self.run(position, None, limits, cancellation, None, &mut |_| {})
    }

    pub fn analyze_with_progress(
        &mut self,
        position: &Position,
        limits: CpuLimits,
        cancellation: &AtomicBool,
        mut observer: impl FnMut(CpuIterationProgress),
    ) -> Result<CpuReport, CpuError> {
        self.run(position, None, limits, cancellation, None, &mut observer)
    }

    /// Execute one real slice of an immutable job. Only PausedStack owns the
    /// unfinished root/PVS/qsearch/SEE frames required for exact continuation.
    #[allow(clippy::too_many_arguments)]
    pub fn analyze_controlled(
        &mut self,
        position: &Position,
        root_moves: Option<&[BoardMove]>,
        limits: CpuLimits,
        cancellation: &AtomicBool,
        control: &CpuSliceControl,
        resume: Option<&CpuResumeToken>,
        mut observer: impl FnMut(CpuIterationProgress),
    ) -> Result<CpuControlledReport, CpuError> {
        self.paused_stack_limit_sealed = true;
        self.last_attempt_work = Some(CpuWork::default());
        if self.resume_policy != CpuResumePolicy::PausedStack {
            return Err(CpuError::Unsupported(
                "cooperative control requires PausedStack",
            ));
        }
        if control.node_credit == 0 || control.mailbox.snapshot().terminal {
            return Err(CpuError::InvalidLimits(
                "slice credit is zero or its job is closed",
            ));
        }
        if resume.is_some_and(|token| token.root_moves.as_deref() != root_moves) {
            return Err(CpuError::ResumeMismatch("slice root restriction changed"));
        }
        let report = stack::run_controlled(
            self,
            position,
            root_moves,
            limits,
            cancellation,
            resume,
            &mut observer,
            control,
        )?;
        self.finish_controlled_report(report, control)
    }

    fn finish_controlled_report(
        &mut self,
        report: CpuReport,
        control: &CpuSliceControl,
    ) -> Result<CpuControlledReport, CpuError> {
        let yield_cause = if report.completion == CpuCompletion::Yielded {
            match control.observed_yield_cause.load(Ordering::Acquire) {
                1 => Some(CpuYieldCause::Requested),
                2 => Some(CpuYieldCause::NodeCredit),
                3 => Some(CpuYieldCause::SoftLease),
                _ => None,
            }
        } else {
            None
        };
        let yield_ack_sequence = if yield_cause == Some(CpuYieldCause::Requested) {
            let sequence = control.observed_yield_sequence.load(Ordering::Acquire);
            if control.mailbox.acknowledge_yield(sequence).is_err() {
                self.discard_paused_stack();
                return Err(CpuError::InvalidLimits("CPU Yield acknowledgement failed"));
            }
            Some(sequence)
        } else {
            None
        };
        Ok(CpuControlledReport {
            report,
            yield_cause,
            yield_ack_sequence,
            publish_ack_sequence: control.mailbox.snapshot().publish_ack,
        })
    }

    /// Read-only admission, used by the private worker before consuming its
    /// terminal seat or continuation. Invalid feedback leaves both untouched.
    pub fn validate_reciprocal_resume(
        &self,
        token: Option<&CpuResumeToken>,
        scope: &CpuReciprocalScope,
        reply: Option<&CpuReciprocalReply>,
    ) -> Result<(), CpuError> {
        scope.validate()?;
        stack::validate_reciprocal_resume(self, token, scope, reply)
    }

    pub fn pending_reciprocal_request(&self) -> Option<CpuReciprocalRequest> {
        stack::pending_reciprocal_request(self)
    }

    /// The same owned native traversal consumes a P/C ordering response at its
    /// saved node stage. It never restarts the node or issues a fresh budget.
    #[allow(clippy::too_many_arguments)]
    pub fn analyze_controlled_reciprocal(
        &mut self,
        position: &Position,
        root_moves: Option<&[BoardMove]>,
        limits: CpuLimits,
        cancellation: &AtomicBool,
        control: &CpuSliceControl,
        resume: Option<&CpuResumeToken>,
        scope: &CpuReciprocalScope,
        reply: Option<&CpuReciprocalReply>,
        mut observer: impl FnMut(CpuIterationProgress),
    ) -> Result<CpuReciprocalControlledReport, CpuError> {
        self.last_attempt_work = Some(CpuWork::default());
        self.validate_reciprocal_resume(resume, scope, reply)?;
        if resume.is_none() {
            let anchor = rz_position::contracts::ContractPosition::new(
                scope.job.state_identity.owner,
                position.clone(),
            )
            .export()
            .map_err(|_| CpuError::ResumeMismatch("reciprocal Rules export failed"))?;
            if anchor.snapshot().identity() != scope.job.state_identity {
                return Err(CpuError::ResumeMismatch(
                    "reciprocal job names another Rules state or history",
                ));
            }
        }
        if self.resume_policy != CpuResumePolicy::PausedStack {
            return Err(CpuError::Unsupported(
                "reciprocal feedback requires PausedStack",
            ));
        }
        if control.node_credit == 0
            || control.mailbox.snapshot().terminal
            || control.mailbox.scope().job != scope.job.id
            || control.mailbox.scope().owner != scope.owner
            || resume.is_some_and(|token| token.root_moves.as_deref() != root_moves)
        {
            return Err(CpuError::ResumeMismatch(
                "reciprocal control or root restriction changed",
            ));
        }
        let report = stack::run_reciprocal(
            self,
            position,
            root_moves,
            limits,
            cancellation,
            resume,
            &mut observer,
            control,
            scope,
            reply,
        )?;
        let request = self.pending_reciprocal_request();
        Ok(CpuReciprocalControlledReport {
            controlled: self.finish_controlled_report(report, control)?,
            request,
        })
    }

    /// Restrict only root choices. Every response below the root remains legal
    /// Rules search; restricted root values never enter the unrestricted TT.
    pub fn analyze_root_moves(
        &mut self,
        position: &Position,
        root_moves: &[BoardMove],
        limits: CpuLimits,
        cancellation: &AtomicBool,
    ) -> Result<CpuReport, CpuError> {
        self.run(
            position,
            Some(root_moves),
            limits,
            cancellation,
            None,
            &mut |_| {},
        )
    }

    /// Apply a checked prefix before analyzing the actual divergence state.
    /// The returned value/PV belong to that state's side to move.
    pub fn analyze_divergence(
        &mut self,
        position: &Position,
        prefix: &[BoardMove],
        limits: CpuLimits,
        cancellation: &AtomicBool,
    ) -> Result<CpuReport, CpuError> {
        self.paused_stack_limit_sealed = true;
        self.last_attempt_work = Some(CpuWork::default());
        if self.paused_side_scope.is_none() || cancellation.load(Ordering::Acquire) {
            self.invalidate_paused_stack();
        }
        if prefix.len() > MAX_DEPTH as usize {
            return Err(CpuError::InvalidLimits("divergence prefix exceeds 64 ply"));
        }
        let mut divergence = position.clone();
        for &mv in prefix {
            divergence.make_move(mv)?;
        }
        self.analyze(&divergence, limits, cancellation)
    }

    pub fn resume(
        &mut self,
        position: &Position,
        token: &CpuResumeToken,
        limits: CpuLimits,
        cancellation: &AtomicBool,
    ) -> Result<CpuReport, CpuError> {
        self.paused_stack_limit_sealed = true;
        self.last_attempt_work = Some(CpuWork::default());
        if self.resume_policy == CpuResumePolicy::PausedStack {
            return stack::run(
                self,
                position,
                token.root_moves.as_deref(),
                limits,
                cancellation,
                Some(token),
                &mut |_| {},
            );
        }
        if token.paused_stack.is_some() {
            return Err(CpuError::ResumeMismatch(
                "paused stack belongs to another policy",
            ));
        }
        if token.root != position.position_identity() {
            return Err(CpuError::ResumeMismatch(
                "Rules state or full known history changed",
            ));
        }
        if token.profile != self.config.profile
            || token.quiescence_ply != self.config.quiescence_ply
            || token.value_identity != *self.value_identity
            || token.search_conditions != self.search_conditions()
        {
            return Err(CpuError::ResumeMismatch(
                "CPU search conditions or value namespace changed",
            ));
        }
        if limits.max_depth < token.completed_depth {
            return Err(CpuError::ResumeMismatch(
                "new depth is below completed depth",
            ));
        }
        self.run(
            position,
            token.root_moves.as_deref(),
            limits,
            cancellation,
            Some(token),
            &mut |_| {},
        )
    }

    fn run_paused_side_scope(
        &mut self,
        position: &Position,
        root_moves: Option<&[BoardMove]>,
        limits: CpuLimits,
        cancellation: &AtomicBool,
        observer: &mut dyn FnMut(CpuIterationProgress),
    ) -> Result<CpuReport, CpuError> {
        let version = self.search_identity();
        let (result, work) = {
            let side = self
                .paused_side_scope
                .as_mut()
                .ok_or(CpuError::Unsupported("held-parent side scope"))?;
            let result = side.run(position, root_moves, limits, cancellation, None, observer);
            (result, side.last_attempt_work())
        };
        // The last attempt and shared caller budget cover this side execution
        // only, not the parent's cumulative pause diagnostics.
        self.last_attempt_work = work;
        if let Some(observation) = self.side_scope_observation.as_mut() {
            if let Some(attempts) = observation.attempts.checked_add(1) {
                observation.attempts = attempts;
            } else {
                observation.complete = false;
            }
            if let Some(work) = work {
                for (total, added) in [
                    (&mut observation.work.nodes, work.nodes),
                    (
                        &mut observation.work.quiescence_nodes,
                        work.quiescence_nodes,
                    ),
                    (&mut observation.work.tt_hits, work.tt_hits),
                ] {
                    if let Some(sum) = total.checked_add(added) {
                        *total = sum;
                    } else {
                        observation.complete = false;
                    }
                }
            } else {
                observation.complete = false;
            }
        }
        if cancellation.load(Ordering::Acquire) {
            self.invalidate_paused_stack();
        }
        result.map(|mut report| {
            // This report belongs to the declared composite implementation.
            // One-shot side work has no continuation token or resumed depth.
            report.search_version = version;
            report.resume = None;
            report.reused_completed_depth = 0;
            report
        })
    }

    fn run(
        &mut self,
        position: &Position,
        root_moves: Option<&[BoardMove]>,
        limits: CpuLimits,
        cancellation: &AtomicBool,
        resume: Option<&CpuResumeToken>,
        observer: &mut dyn FnMut(CpuIterationProgress),
    ) -> Result<CpuReport, CpuError> {
        self.paused_stack_limit_sealed = true;
        if resume.is_none() && self.paused.is_some() && self.paused_side_scope.is_some() {
            return self.run_paused_side_scope(
                position,
                root_moves,
                limits,
                cancellation,
                observer,
            );
        }
        if self.resume_policy == CpuResumePolicy::PausedStack {
            return stack::run(
                self,
                position,
                root_moves,
                limits,
                cancellation,
                resume,
                observer,
            );
        }
        self.last_attempt_work = Some(CpuWork::default());
        self.validate_limits(position, limits)?;
        let started = Instant::now();
        let root_view = position.ordered_legal_moves();
        let status = position.play_status_from_view(&root_view)?;
        let mut legal = root_view.moves().to_vec();
        if let Some(requested) = root_moves {
            if requested.len() > MAX_ROOT_MOVES {
                return Err(CpuError::InvalidRootMoves("restriction exceeds 256 moves"));
            }
            if requested.is_empty() {
                return Err(CpuError::InvalidRootMoves("empty restriction"));
            }
            for (index, mv) in requested.iter().enumerate() {
                if !legal.contains(mv) {
                    return Err(CpuError::InvalidRootMoves("move is not legal at root"));
                }
                if requested[..index].contains(mv) {
                    return Err(CpuError::InvalidRootMoves("duplicate move"));
                }
            }
            // Rules' deterministic order remains the tie breaker.
            legal.retain(|mv| requested.contains(mv));
        }
        if let PlayStatus::Terminal { reason, winner } = status {
            return Ok(CpuReport {
                best_move: None,
                pv: Vec::new(),
                score: terminal_score(position, winner, 0),
                completed_depth: 0,
                nodes: 0,
                quiescence_nodes: 0,
                tt_hits: 0,
                completion: CpuCompletion::Terminal(reason),
                score_scope: CpuScoreScope::RulesTerminal,
                profile: self.config.profile,
                score_provenance: "rz-position/rules-terminal",
                value_identity: (*self.value_identity).clone(),
                search_version: self.search_identity(),
                root_restricted: root_moves.is_some(),
                elapsed: started.elapsed(),
                reused_completed_depth: 0,
                resume: None,
            });
        }
        let mut control = Control {
            limits,
            cancellation,
            cooperative: None,
            nodes: 0,
            quiescence_nodes: 0,
            tt_hits: 0,
            value: self
                .evaluator
                .initialize(position)
                .map_err(CpuError::Value)?,
        };
        let mut work = position.clone();
        let mut best = resume
            .map(|token| token.pv.clone())
            .unwrap_or_else(|| legal.first().copied().into_iter().collect());
        let mut score = if let Some(token) = resume {
            token.score
        } else {
            self.frontier_score(&control, position)
                .map_err(CpuError::Value)?
        };
        let mut completed_depth = resume.map_or(0, |token| token.completed_depth);
        let mut completion = CpuCompletion::DepthLimit;
        let mut value_authority_stopped = false;
        observer(CpuIterationProgress {
            best_move: best.first().copied(),
            score,
            completed_depth,
            nodes: 0,
        });
        let admission = control.check_stop();
        if let Err(Abort::Stop(reason)) = &admission {
            completion = *reason;
        }

        for depth in
            (completed_depth.saturating_add(1)..=limits.max_depth).filter(|_| admission.is_ok())
        {
            let window = if completed_depth == 0 || score.abs() >= MATE_THRESHOLD {
                (-INFINITY, INFINITY)
            } else {
                (score - ASPIRATION_WINDOW, score + ASPIRATION_WINDOW)
            };
            let searched = self.root_search(
                &mut work,
                &legal,
                depth,
                window.0,
                window.1,
                best.first().copied(),
                &mut control,
            );
            let searched = match searched {
                Ok(value) if value.score <= window.0 || value.score >= window.1 => self
                    .root_search(
                        &mut work,
                        &legal,
                        depth,
                        -INFINITY,
                        INFINITY,
                        best.first().copied(),
                        &mut control,
                    ),
                result => result,
            };
            match searched {
                Ok(value) => {
                    score = value.score;
                    best = value.pv;
                    completed_depth = depth;
                    observer(CpuIterationProgress {
                        best_move: best.first().copied(),
                        score,
                        completed_depth,
                        nodes: control.nodes,
                    });
                }
                Err(Abort::Stop(reason)) => {
                    completion = reason;
                    break;
                }
                Err(Abort::ValueAuthorityStop(reason)) => {
                    completion = reason;
                    value_authority_stopped = true;
                    break;
                }
                Err(Abort::Error(error)) => {
                    self.last_attempt_work = Some(control.actual_work());
                    return Err(error);
                }
            }
        }
        // Observers are outside node search and may cancel or consume time.
        // Preserve the completed result while recording the final stop reason.
        if !value_authority_stopped && let Err(Abort::Stop(reason)) = control.check_stop() {
            completion = reason;
        }
        self.last_attempt_work = Some(control.actual_work());
        let token = (completed_depth > 0 && !value_authority_stopped).then(|| CpuResumeToken {
            root: position.position_identity(),
            root_moves: root_moves.map(<[BoardMove]>::to_vec),
            profile: self.config.profile,
            quiescence_ply: self.config.quiescence_ply,
            value_identity: (*self.value_identity).clone(),
            search_conditions: self.search_conditions(),
            completed_depth,
            score,
            pv: best.clone(),
            paused_stack: None,
        });
        Ok(CpuReport {
            best_move: best.first().copied(),
            pv: best,
            score,
            completed_depth,
            nodes: control.nodes,
            quiescence_nodes: control.quiescence_nodes,
            tt_hits: control.tt_hits,
            completion,
            score_scope: if completed_depth == 0 {
                CpuScoreScope::FrontierOnly
            } else {
                CpuScoreScope::CompletedIteration
            },
            profile: self.config.profile,
            score_provenance: self.evaluator.provenance(),
            value_identity: (*self.value_identity).clone(),
            search_version: self.search_identity(),
            root_restricted: root_moves.is_some(),
            elapsed: started.elapsed(),
            reused_completed_depth: resume.map_or(0, |token| token.completed_depth),
            resume: token,
        })
    }

    pub(crate) fn validate_frozen_value_namespace(&self) -> Result<(), CpuError> {
        if self.evaluator.identity() != self.value_identity.as_ref() {
            return Err(CpuError::InvalidConfig(
                "evaluator semantics or frozen weights changed",
            ));
        }
        Ok(())
    }

    fn validate_limits(&self, position: &Position, limits: CpuLimits) -> Result<(), CpuError> {
        self.validate_frozen_value_namespace()?;
        if limits.max_depth == 0 || limits.max_depth > self.config.max_depth {
            return Err(CpuError::InvalidLimits("depth outside configured bound"));
        }
        if limits.max_nodes == 0 {
            return Err(CpuError::InvalidLimits("node budget must be positive"));
        }
        if position.known_history_len() > MAX_KNOWN_HISTORY {
            return Err(CpuError::InvalidLimits("known history exceeds 4096 states"));
        }
        Ok(())
    }

    fn frontier_score(
        &self,
        control: &Control<'_>,
        position: &Position,
    ) -> Result<i32, CpuValueError> {
        let score = match control
            .cooperative
            .and_then(|slice| slice.nnuec_capture.as_ref())
        {
            Some(capture) => {
                self.evaluator
                    .score_with_capture(&control.value, position, capture)?
            }
            None => self.evaluator.score(&control.value, position)?,
        };
        if !(-CPU_FRONTIER_SCORE_LIMIT..=CPU_FRONTIER_SCORE_LIMIT).contains(&score) {
            return Err(CpuValueError::ScoreOutsideFrontierNamespace);
        }
        Ok(score)
    }

    #[allow(clippy::too_many_arguments)]
    fn root_search(
        &mut self,
        position: &mut Position,
        legal: &[BoardMove],
        depth: u16,
        mut alpha: i32,
        beta: i32,
        previous_best: Option<BoardMove>,
        control: &mut Control<'_>,
    ) -> Result<NodeValue, Abort> {
        control.visit(false)?;
        let mut moves = legal.to_vec();
        self.order_moves(position, &mut moves, previous_best, control)?;
        let mut best = NodeValue {
            score: -INFINITY,
            pv: Vec::new(),
        };
        for (index, mv) in moves.into_iter().enumerate() {
            let undo = position.make_move(mv)?;
            let value_undo = match self.evaluator.apply_delta(&mut control.value, undo.delta()) {
                Ok(undo) => undo,
                Err(error) => {
                    position.unmake(undo)?;
                    return Err(error.into());
                }
            };
            let child = self
                .pvs_child(position, depth - 1, 1, alpha, beta, index == 0, control)
                .and_then(|child| {
                    // A zero-window fail-low equal to alpha is only an upper
                    // bound. Prove an earlier Rules move actually ties before
                    // replacing the current root choice.
                    if self.root_tie_policy == CpuRootTiePolicy::RulesOrderConfirmed
                        && index != 0
                        && -child.score == best.score
                        && alpha == best.score
                        && alpha > -INFINITY
                        && root_move_precedes(legal, mv, best.pv.first().copied())
                    {
                        self.negamax(position, depth - 1, 1, -beta, 1 - alpha, control)
                    } else {
                        Ok(child)
                    }
                });
            position.unmake(undo)?;
            self.evaluator
                .restore(&mut control.value, value_undo, position)?;
            let child = child?;
            let score = -child.score;
            if score > best.score
                || (self.root_tie_policy == CpuRootTiePolicy::RulesOrderConfirmed
                    && score == best.score
                    && alpha == best.score
                    && root_move_precedes(legal, mv, best.pv.first().copied()))
            {
                best.score = score;
                best.pv.clear();
                best.pv.push(mv);
                best.pv.extend(child.pv);
            }
            alpha = alpha.max(score);
            if alpha >= beta {
                break;
            }
        }
        control.check_stop()?;
        Ok(best)
    }

    #[allow(clippy::too_many_arguments)]
    fn pvs_child(
        &mut self,
        position: &mut Position,
        depth: u16,
        ply: u16,
        alpha: i32,
        beta: i32,
        first: bool,
        control: &mut Control<'_>,
    ) -> Result<NodeValue, Abort> {
        if first {
            return self.negamax(position, depth, ply, -beta, -alpha, control);
        }
        let mut value = self.negamax(position, depth, ply, -alpha - 1, -alpha, control)?;
        if -value.score > alpha && -value.score < beta {
            value = self.negamax(position, depth, ply, -beta, -alpha, control)?;
        }
        Ok(value)
    }

    fn negamax(
        &mut self,
        position: &mut Position,
        depth: u16,
        ply: u16,
        mut alpha: i32,
        beta: i32,
        control: &mut Control<'_>,
    ) -> Result<NodeValue, Abort> {
        if depth == 0 {
            return self.quiescence(
                position,
                ply,
                self.config.quiescence_ply,
                alpha,
                beta,
                control,
            );
        }
        control.visit(false)?;
        let view = position.ordered_legal_moves();
        if let PlayStatus::Terminal { winner, .. } = position.play_status_from_view(&view)? {
            return Ok(NodeValue {
                score: terminal_score(position, winner, ply),
                pv: Vec::new(),
            });
        }
        let alpha_original = alpha;
        let hash = tt_hash(position);
        let identity = position.position_identity();
        let value_context = self.evaluator.cache_context(&control.value, position)?;
        let tt = self.probe(hash, &identity, &value_context).cloned();
        if let Some(entry) = &tt {
            control.tt_hits += 1;
            // A depth-8 bound cannot masquerade as a depth-10 result.
            // A deeper static frontier is not mathematically a bound on a
            // shallower frontier. Require the same remaining depth so replay
            // preserves this profile's fixed-depth estimate exactly.
            if entry.depth == depth {
                let score = from_tt_score(entry.score, ply);
                if entry.bound == Bound::Exact
                    || (entry.bound == Bound::Lower && score >= beta)
                    || (entry.bound == Bound::Upper && score <= alpha)
                {
                    return Ok(NodeValue {
                        score,
                        pv: entry.best_move.into_iter().collect(),
                    });
                }
            }
        }
        let mut moves = view.moves().to_vec();
        self.order_moves(
            position,
            &mut moves,
            tt.as_ref().and_then(|entry| entry.best_move),
            control,
        )?;
        let mut best = NodeValue {
            score: -INFINITY,
            pv: Vec::new(),
        };
        for (index, mv) in moves.into_iter().enumerate() {
            let quiet = !is_tactical(position, mv);
            let side = position.side_to_move() as usize;
            let undo = position.make_move(mv)?;
            let value_undo = match self.evaluator.apply_delta(&mut control.value, undo.delta()) {
                Ok(undo) => undo,
                Err(error) => {
                    position.unmake(undo)?;
                    return Err(error.into());
                }
            };
            let child = self.pvs_child(
                position,
                depth - 1,
                ply + 1,
                alpha,
                beta,
                index == 0,
                control,
            );
            position.unmake(undo)?;
            self.evaluator
                .restore(&mut control.value, value_undo, position)?;
            let child = child?;
            let score = -child.score;
            if score > best.score {
                best.score = score;
                best.pv.clear();
                best.pv.push(mv);
                best.pv.extend(child.pv);
            }
            alpha = alpha.max(score);
            if alpha >= beta {
                if quiet {
                    let history =
                        &mut self.history[side][mv.from.index() as usize][mv.to.index() as usize];
                    *history = history.saturating_add(i32::from(depth).pow(2)).min(20_000);
                }
                break;
            }
        }
        // A cancellation/deadline after the last child still forbids this
        // parent from publishing a completed TT bound.
        control.check_stop()?;
        let bound = if best.score <= alpha_original {
            Bound::Upper
        } else if best.score >= beta {
            Bound::Lower
        } else {
            Bound::Exact
        };
        self.store(TtEntry {
            hash,
            identity,
            profile: self.config.profile,
            value_identity: Arc::clone(&self.value_identity),
            value_context,
            depth,
            score: to_tt_score(best.score, ply),
            bound,
            best_move: best.pv.first().copied(),
        });
        Ok(best)
    }

    fn quiescence(
        &mut self,
        position: &mut Position,
        ply: u16,
        remaining: u16,
        mut alpha: i32,
        beta: i32,
        control: &mut Control<'_>,
    ) -> Result<NodeValue, Abort> {
        control.visit(true)?;
        let view = position.ordered_legal_moves();
        if let PlayStatus::Terminal { winner, .. } = position.play_status_from_view(&view)? {
            return Ok(NodeValue {
                score: terminal_score(position, winner, ply),
                pv: Vec::new(),
            });
        }
        let in_check = position.in_check();
        if in_check && remaining == 0 {
            return Err(Abort::Stop(CpuCompletion::QuiescenceLimit));
        }
        let mut best = NodeValue {
            // Stand-pat is forbidden in check, including at the depth cap.
            score: if in_check {
                -INFINITY
            } else {
                self.frontier_score(control, position)?
            },
            pv: Vec::new(),
        };
        if !in_check {
            if best.score >= beta || remaining == 0 {
                return Ok(best);
            }
            alpha = alpha.max(best.score);
        }
        let mut moves = view.moves().to_vec();
        if !in_check {
            moves.retain(|mv| is_tactical(position, *mv));
        }
        self.order_moves(position, &mut moves, None, control)?;
        for mv in moves {
            let undo = position.make_move(mv)?;
            let value_undo = match self.evaluator.apply_delta(&mut control.value, undo.delta()) {
                Ok(undo) => undo,
                Err(error) => {
                    position.unmake(undo)?;
                    return Err(error.into());
                }
            };
            let child = self.quiescence(position, ply + 1, remaining - 1, -beta, -alpha, control);
            position.unmake(undo)?;
            self.evaluator
                .restore(&mut control.value, value_undo, position)?;
            let child = child?;
            let score = -child.score;
            if score > best.score {
                best.score = score;
                best.pv.clear();
                best.pv.push(mv);
                best.pv.extend(child.pv);
            }
            alpha = alpha.max(score);
            if alpha >= beta {
                break;
            }
        }
        control.check_stop()?;
        Ok(best)
    }

    fn order_moves(
        &self,
        position: &Position,
        moves: &mut [BoardMove],
        best: Option<BoardMove>,
        control: &mut Control<'_>,
    ) -> Result<(), Abort> {
        self.order_moves_with_see_budget(
            position,
            moves,
            best,
            control,
            MAX_SEE_POSITIONS_PER_ORDER,
        )
    }

    fn order_moves_with_see_budget(
        &self,
        position: &Position,
        moves: &mut [BoardMove],
        best: Option<BoardMove>,
        control: &mut Control<'_>,
        maximum_exchange_positions: u32,
    ) -> Result<(), Abort> {
        if self.ordering == CpuOrderingPolicy::LegacyMvvLvaV1 {
            // Preserve historical stable sorting and MVV/LVA/history keys.
            moves.sort_by_key(|mv| std::cmp::Reverse(self.order_score(position, *mv, best)));
            return Ok(());
        }
        let mut scored = Vec::new();
        scored
            .try_reserve_exact(moves.len())
            .map_err(|_| Abort::Error(CpuError::Allocation))?;
        let mut exchange_positions = 0;
        let mut visit_exchange = || {
            control.check_stop()?;
            if exchange_positions >= maximum_exchange_positions {
                return Err(Abort::Error(CpuError::Unsupported(
                    "legal SEE ordering exchange position budget exhausted",
                )));
            }
            control.visit(false)?;
            exchange_positions += 1;
            Ok(())
        };
        for &mv in moves.iter() {
            let key = if best == Some(mv) {
                (3u8, 0)
            } else if is_tactical(position, mv) {
                let exchange = see::score(position, mv, &mut visit_exchange)?;
                (if exchange >= 0 { 2 } else { 0 }, exchange)
            } else {
                (1, self.order_score(position, mv, None))
            };
            scored.push((mv, key));
        }
        // Evaluate once per move; equal keys retain the original Rules order.
        // An interrupted computation leaves caller order intact and never
        // replaces an unobserved exchange value with zero or MVV/LVA.
        scored.sort_by_key(|(_, key)| std::cmp::Reverse(*key));
        control.check_stop()?;
        for (target, (mv, _)) in moves.iter_mut().zip(scored) {
            *target = mv;
        }
        Ok(())
    }

    fn order_score(&self, position: &Position, mv: BoardMove, best: Option<BoardMove>) -> i32 {
        if best == Some(mv) {
            return 1_000_000;
        }
        let attacker = position
            .piece_at(mv.from)
            .map_or(0, |piece| material(piece.kind));
        let victim = position.piece_at(mv.to).map_or_else(
            || {
                if is_en_passant(position, mv) {
                    material(PieceKind::Pawn)
                } else {
                    0
                }
            },
            |piece| material(piece.kind),
        );
        if victim > 0 || mv.promotion.is_some() {
            // MVV/LVA is an ordering hint, not an exact SEE or pruning bound.
            return 100_000 + victim * 16 - attacker + mv.promotion.map_or(0, material);
        }
        self.history[position.side_to_move() as usize][mv.from.index() as usize]
            [mv.to.index() as usize]
    }

    fn probe(
        &self,
        hash: u64,
        identity: &PositionIdentity,
        value_context: &CpuValueCacheContext,
    ) -> Option<&TtEntry> {
        if self.tt.is_empty() {
            return None;
        }
        self.tt[hash as usize % self.tt.len()]
            .as_ref()
            .filter(|entry| {
                entry.hash == hash
                    && entry.profile == self.config.profile
                    && entry.value_identity == self.value_identity
                    && entry.value_context == *value_context
                    && entry.identity == *identity
            })
    }

    fn store(&mut self, entry: TtEntry) {
        if self.tt.is_empty() {
            return;
        }
        let index = entry.hash as usize % self.tt.len();
        if self.tt[index].as_ref().is_some_and(|old| {
            old.hash == entry.hash
                && old.identity == entry.identity
                && old.value_identity == entry.value_identity
                && old.value_context == entry.value_context
                && old.depth > entry.depth
        }) {
            return;
        }
        self.tt[index] = Some(entry);
    }
}

// Both moves were admitted from this Rules-ordered root domain. A missing
// incumbent is not a tie; an unknown candidate must never gain priority.
fn root_move_precedes(
    legal: &[BoardMove],
    candidate: BoardMove,
    current: Option<BoardMove>,
) -> bool {
    match (
        legal.iter().position(|movement| *movement == candidate),
        current.and_then(|best| legal.iter().position(|movement| *movement == best)),
    ) {
        (Some(candidate), Some(current)) => candidate < current,
        _ => false,
    }
}

impl Control<'_> {
    fn actual_work(&self) -> CpuWork {
        CpuWork {
            nodes: self.nodes,
            quiescence_nodes: self.quiescence_nodes,
            tt_hits: self.tt_hits,
        }
    }

    fn check_stop(&self) -> Result<(), Abort> {
        if self.cancellation.load(Ordering::Acquire)
            || self.cooperative.is_some_and(|slice| {
                let snapshot = slice.mailbox.snapshot();
                snapshot.is_cancelled() || snapshot.terminal
            })
        {
            Err(Abort::Stop(CpuCompletion::Canceled))
        } else if self
            .limits
            .deadline
            .is_some_and(|deadline| Instant::now() >= deadline)
        {
            Err(Abort::Stop(CpuCompletion::Deadline))
        } else if let Some(slice) = self.cooperative {
            // The hard remaining credit wins over a soft slice boundary.
            if self.nodes >= self.limits.max_nodes {
                return Err(Abort::Stop(CpuCompletion::NodeLimit));
            }
            if let Some(sequence) = slice.mailbox.snapshot().pending_yield() {
                slice.observe_yield(CpuYieldCause::Requested, sequence);
                Err(Abort::Stop(CpuCompletion::Yielded))
            } else if self.nodes >= slice.node_credit {
                slice.observe_yield(CpuYieldCause::NodeCredit, 0);
                Err(Abort::Stop(CpuCompletion::Yielded))
            } else if slice
                .soft_lease
                .is_some_and(|lease| Instant::now() >= lease)
            {
                slice.observe_yield(CpuYieldCause::SoftLease, 0);
                Err(Abort::Stop(CpuCompletion::Yielded))
            } else {
                Ok(())
            }
        } else {
            Ok(())
        }
    }

    fn visit(&mut self, quiescence: bool) -> Result<(), Abort> {
        self.check_stop()?;
        if self.nodes >= self.limits.max_nodes {
            return Err(Abort::Stop(CpuCompletion::NodeLimit));
        }
        self.nodes += 1;
        if quiescence {
            self.quiescence_nodes += 1;
        }
        Ok(())
    }
}

fn tt_hash(position: &Position) -> u64 {
    let mut hasher = DefaultHasher::new();
    position.repetition_identity().hash(&mut hasher);
    position.halfmove_clock().hash(&mut hasher);
    position.fullmove_number().hash(&mut hasher);
    position.known_history_len().hash(&mut hasher);
    hasher.finish()
}

fn terminal_score(position: &Position, winner: Option<Color>, ply: u16) -> i32 {
    match winner {
        Some(winner) if winner == position.side_to_move() => CPU_MATE_SCORE - i32::from(ply),
        Some(_) => -CPU_MATE_SCORE + i32::from(ply),
        None => 0,
    }
}

fn to_tt_score(score: i32, ply: u16) -> i32 {
    if score >= MATE_THRESHOLD {
        score + i32::from(ply)
    } else if score <= -MATE_THRESHOLD {
        score - i32::from(ply)
    } else {
        score
    }
}

fn from_tt_score(score: i32, ply: u16) -> i32 {
    if score >= MATE_THRESHOLD {
        score - i32::from(ply)
    } else if score <= -MATE_THRESHOLD {
        score + i32::from(ply)
    } else {
        score
    }
}

fn material(kind: PieceKind) -> i32 {
    match kind {
        PieceKind::Pawn => 100,
        PieceKind::Knight => 320,
        PieceKind::Bishop => 330,
        PieceKind::Rook => 500,
        PieceKind::Queen => 900,
        PieceKind::King => 0,
    }
}

fn is_en_passant(position: &Position, mv: BoardMove) -> bool {
    position.en_passant_target() == Some(mv.to)
        && position.piece_at(mv.to).is_none()
        && position
            .piece_at(mv.from)
            .is_some_and(|piece| piece.kind == PieceKind::Pawn)
        && mv.from.file() != mv.to.file()
}

fn is_tactical(position: &Position, mv: BoardMove) -> bool {
    mv.promotion.is_some() || position.piece_at(mv.to).is_some() || is_en_passant(position, mv)
}

/// Deterministic, untrained side-to-move raw score, with bounded magnitude far
/// below the exact mate namespace. No win probability or calibrated CP claim.
pub fn evaluate_bootstrap(position: &Position) -> i32 {
    let mut white = 0i32;
    for index in 0..64 {
        let square = Square::new(index).expect("bounded board index");
        let Some(piece) = position.piece_at(square) else {
            continue;
        };
        let rank = if piece.color == Color::White {
            square.rank()
        } else {
            7 - square.rank()
        };
        let center = 6 - (2 * i32::from(square.file()) - 7).abs() - (2 * i32::from(rank) - 7).abs();
        let pst = match piece.kind {
            PieceKind::Pawn => i32::from(rank) * 8,
            PieceKind::Knight => center * 4,
            PieceKind::Bishop => center * 2,
            PieceKind::Rook => i32::from(rank) * 2,
            PieceKind::Queen => center,
            PieceKind::King => -center * 2,
        };
        let value = material(piece.kind) + pst;
        white += if piece.color == Color::White {
            value
        } else {
            -value
        };
    }
    let value = if position.side_to_move() == Color::White {
        white
    } else {
        -white
    };
    value.clamp(-20_000, 20_000)
}

#[cfg(test)]
mod tests {
    use super::*;

    // The second Rules move wins at depth one, then both tie (or the first
    // is strictly worse) at depth two. It makes the real previous-best PVS
    // path exercise an earlier move whose scout score can be only a bound.
    struct RootTieValue {
        base: BootstrapCpuValue,
        earlier_is_worse: bool,
        identity: CpuValueIdentity,
        fail_score: Arc<AtomicBool>,
    }
    impl RootTieValue {
        fn new(earlier_is_worse: bool) -> Self {
            Self {
                base: BootstrapCpuValue::default(),
                earlier_is_worse,
                identity: CpuValueIdentity {
                    semantics: if earlier_is_worse {
                        "test-root-tie-bound-v1"
                    } else {
                        "test-root-tie-equal-v1"
                    }
                    .to_owned(),
                    weights_sha256: None,
                    training: crate::cpu_value::CpuTrainingState::Bootstrap,
                },
                fail_score: Arc::new(AtomicBool::new(false)),
            }
        }
    }
    impl CpuValueEvaluator for RootTieValue {
        fn identity(&self) -> &CpuValueIdentity {
            &self.identity
        }
        fn provenance(&self) -> &'static str {
            "root-tie-test"
        }
        fn initialize(&self, p: &Position) -> Result<CpuAccumulator, CpuValueError> {
            self.base.initialize(p)
        }
        fn cache_context(
            &self,
            a: &CpuAccumulator,
            p: &Position,
        ) -> Result<CpuValueCacheContext, CpuValueError> {
            self.base.cache_context(a, p)
        }
        fn score(&self, _: &CpuAccumulator, p: &Position) -> Result<i32, CpuValueError> {
            if self.fail_score.load(Ordering::Acquire) {
                return Err(CpuValueError::NonFiniteForward);
            }
            let f3 = p.piece_at("f3".parse().unwrap());
            let c3 = p.piece_at("c3".parse().unwrap());
            let white_knight = |piece: Option<rz_position::Piece>| {
                piece.is_some_and(|p| p.color == Color::White && p.kind == PieceKind::Knight)
            };
            if p.side_to_move() == Color::Black && white_knight(f3) {
                Ok(-100)
            } else if self.earlier_is_worse
                && p.side_to_move() == Color::White
                && white_knight(c3)
                && p.piece_at("b5".parse().unwrap()).is_some_and(|piece| {
                    piece.color == Color::Black && piece.kind == PieceKind::Pawn
                })
            {
                Ok(-10)
            } else {
                Ok(0)
            }
        }
        fn apply_delta(
            &self,
            a: &mut CpuAccumulator,
            d: &rz_position::RuleMoveDelta,
        ) -> Result<crate::cpu_value::CpuAccumulatorUndo, CpuValueError> {
            self.base.apply_delta(a, d)
        }
        fn restore(
            &self,
            a: &mut CpuAccumulator,
            u: crate::cpu_value::CpuAccumulatorUndo,
            p: &Position,
        ) -> Result<(), CpuValueError> {
            self.base.restore(a, u, p)
        }
    }

    pub(super) fn root_tie_engine(
        policy: CpuResumePolicy,
        earlier_is_worse: bool,
        tt_entries: usize,
    ) -> CpuEngine {
        root_tie_engine_with_ordering(
            policy,
            earlier_is_worse,
            tt_entries,
            CpuOrderingPolicy::LegacyMvvLvaV1,
        )
    }

    pub(super) fn root_tie_engine_with_failure() -> (CpuEngine, Arc<AtomicBool>) {
        let value = RootTieValue::new(false);
        let fail_score = Arc::clone(&value.fail_score);
        let cpu = CpuEngine::with_evaluator_ordering_resume_and_tie_policy(
            CpuConfig {
                tt_entries: 0,
                max_depth: 2,
                quiescence_ply: 0,
                ..CpuConfig::default()
            },
            Arc::new(value),
            CpuOrderingPolicy::LegacyMvvLvaV1,
            CpuResumePolicy::PausedStack,
            CpuRootTiePolicy::RulesOrderConfirmed,
        )
        .unwrap();
        (cpu, fail_score)
    }

    fn root_tie_engine_with_ordering(
        policy: CpuResumePolicy,
        earlier_is_worse: bool,
        tt_entries: usize,
        ordering: CpuOrderingPolicy,
    ) -> CpuEngine {
        CpuEngine::with_evaluator_ordering_resume_and_tie_policy(
            CpuConfig {
                tt_entries,
                max_depth: 2,
                quiescence_ply: 0,
                ..CpuConfig::default()
            },
            Arc::new(RootTieValue::new(earlier_is_worse)),
            ordering,
            policy,
            CpuRootTiePolicy::RulesOrderConfirmed,
        )
        .unwrap()
    }

    #[test]
    fn root_rules_tie_is_proved_before_overriding_previous_best() {
        let root = Position::startpos();
        let original = root.snapshot();
        let moves = [
            BoardMove::from_uci("b1c3").unwrap(),
            BoardMove::from_uci("g1f3").unwrap(),
        ];
        let rules = root.ordered_legal_moves();
        assert!(
            rules.moves().iter().position(|m| *m == moves[0])
                < rules.moves().iter().position(|m| *m == moves[1])
        );
        for (policy, ordering, tt) in [
            (
                CpuResumePolicy::CompletedIteration,
                CpuOrderingPolicy::LegacyMvvLvaV1,
                0,
            ),
            (
                CpuResumePolicy::PausedStack,
                CpuOrderingPolicy::LegalSeeV1,
                64,
            ),
        ] {
            for earlier_is_worse in [false, true] {
                let cancel = AtomicBool::new(false);
                let mut first =
                    root_tie_engine_with_ordering(policy, earlier_is_worse, tt, ordering);
                assert_eq!(
                    first
                        .analyze_root_moves(&root, &moves, limits(1), &cancel)
                        .unwrap()
                        .best_move,
                    Some(moves[1])
                );
                let mut whole =
                    root_tie_engine_with_ordering(policy, earlier_is_worse, tt, ordering);
                let report = whole
                    .analyze_root_moves(&root, &moves, limits(2), &cancel)
                    .unwrap();
                assert_eq!((report.completed_depth, report.score), (2, 0));
                assert_eq!(
                    report.best_move,
                    Some(if earlier_is_worse { moves[1] } else { moves[0] })
                );
                assert!(root.matches_snapshot(&original));
            }
        }
    }

    #[test]
    fn root_tie_scout_bound_is_not_an_exact_equal_score() {
        let cancel = AtomicBool::new(false);
        let mut p = Position::startpos();
        p.make_move(BoardMove::from_uci("b1c3").unwrap()).unwrap();
        let original = p.snapshot();
        let mut cpu = root_tie_engine(CpuResumePolicy::CompletedIteration, true, 0);
        let mut scout_control = ordering_control(&cpu, &p, &cancel, limits(2));
        let scout = cpu
            .negamax(&mut p, 1, 1, -1, 0, &mut scout_control)
            .unwrap();
        assert_eq!(-scout.score, 0, "scout can return the false tie bound");
        assert!(original.same_state(&p.snapshot()));
        let mut exact_control = ordering_control(&cpu, &p, &cancel, limits(2));
        let exact = cpu
            .negamax(&mut p, 1, 1, -INFINITY, INFINITY, &mut exact_control)
            .unwrap();
        assert_eq!(-exact.score, -10, "full search finds the later b7b5 reply");
        assert!(original.same_state(&p.snapshot()));
    }

    #[test]
    fn explicit_root_tie_policy_preserves_legacy_and_separates_tokens() {
        let position = Position::startpos();
        let moves = [
            BoardMove::from_uci("b1c3").unwrap(),
            BoardMove::from_uci("g1f3").unwrap(),
        ];
        let cancel = AtomicBool::new(false);
        let config = CpuConfig {
            tt_entries: 0,
            max_depth: 2,
            quiescence_ply: 0,
            ..CpuConfig::default()
        };
        let mut legacy_owner = CpuEngine::with_evaluator_ordering_and_resume_policy(
            config.clone(),
            Arc::new(RootTieValue::new(false)),
            CpuOrderingPolicy::LegacyMvvLvaV1,
            CpuResumePolicy::PausedStack,
        )
        .unwrap();
        let first = legacy_owner
            .analyze_root_moves(
                &position,
                &moves,
                CpuLimits {
                    max_nodes: 1,
                    ..limits(2)
                },
                &cancel,
            )
            .unwrap();
        let live_token = first.resume.unwrap();
        assert!(live_token.is_paused_stack() && legacy_owner.token_is_current(&live_token));
        let mut other_policy = root_tie_engine(CpuResumePolicy::PausedStack, false, 0);
        assert_eq!(
            legacy_owner.root_tie_policy(),
            CpuRootTiePolicy::LegacyFirstSearched
        );
        assert_ne!(
            legacy_owner.search_identity(),
            other_policy.search_identity()
        );
        assert_ne!(
            legacy_owner.search_conditions(),
            other_policy.search_conditions()
        );
        assert!(matches!(
            other_policy.resume(&position, &live_token, limits(2), &cancel),
            Err(CpuError::ResumeMismatch(_))
        ));
        assert!(legacy_owner.token_is_current(&live_token));
        let legacy_done = legacy_owner
            .resume(&position, &live_token, limits(2), &cancel)
            .unwrap();
        assert_eq!(
            (legacy_done.completed_depth, legacy_done.best_move),
            (2, Some(moves[1]))
        );

        let mut side = CpuEngine::with_paused_side_scope_and_tie_policy(
            config,
            Arc::new(RootTieValue::new(false)),
            CpuOrderingPolicy::LegacyMvvLvaV1,
            CpuRootTiePolicy::RulesOrderConfirmed,
        )
        .unwrap();
        assert_eq!(side.search_identity(), CPU_ROOT_TIE_SIDE_SEARCH_VERSION);
        assert_eq!(
            side.paused_side_scope.as_ref().unwrap().root_tie_policy(),
            CpuRootTiePolicy::RulesOrderConfirmed
        );
        assert_eq!(
            side.paused_side_scope.as_ref().unwrap().search_identity(),
            CPU_ROOT_TIE_SEARCH_VERSION
        );
        let first = side
            .analyze_root_moves(
                &position,
                &moves,
                CpuLimits {
                    max_nodes: 1,
                    ..limits(2)
                },
                &cancel,
            )
            .unwrap();
        let retained = first.resume.unwrap();
        assert!(retained.is_paused_stack() && side.token_is_current(&retained));
        let side_report = side
            .analyze_root_moves(&position, &moves, limits(2), &cancel)
            .unwrap();
        assert_eq!(
            (side_report.search_version, side_report.best_move),
            (CPU_ROOT_TIE_SIDE_SEARCH_VERSION, Some(moves[0]))
        );
        assert!(side_report.resume.is_none() && side.token_is_current(&retained));
        let primary = side
            .resume(&position, &retained, limits(2), &cancel)
            .unwrap();
        assert_eq!(
            (primary.completed_depth, primary.best_move),
            (2, Some(moves[0]))
        );
    }
    #[test]
    fn only_original_soft_value_revocation_can_be_a_controlled_stop() {
        use crate::cpu_value::CpuValueAuthorityRevocation;
        for (reason, completion) in [
            (
                CpuValueAuthorityRevocation::Cancelled,
                CpuCompletion::Canceled,
            ),
            (
                CpuValueAuthorityRevocation::Expired,
                CpuCompletion::Deadline,
            ),
        ] {
            assert!(matches!(
                Abort::from(CpuValueError::AuthorityRevoked(reason)),
                Abort::ValueAuthorityStop(actual) if actual == completion,
            ));
        }
        for error in [
            CpuValueError::AuthorityRevoked(CpuValueAuthorityRevocation::ActorStopped),
            CpuValueError::AuthorityRevoked(CpuValueAuthorityRevocation::Invalidated),
            CpuValueError::StateMismatch,
            CpuValueError::NonFiniteForward,
            CpuValueError::ScoreOutsideFrontierNamespace,
            CpuValueError::AccumulatorOverflow,
        ] {
            assert!(matches!(
                Abort::from(error.clone()),
                Abort::Error(CpuError::Value(actual)) if actual == error,
            ));
        }
    }

    fn held_cpu() -> CpuEngine {
        CpuEngine::with_paused_side_scope(
            CpuConfig {
                tt_entries: 64,
                max_depth: 2,
                ..CpuConfig::default()
            },
            Arc::new(BootstrapCpuValue::default()),
            CpuOrderingPolicy::LegacyMvvLvaV1,
        )
        .unwrap()
    }

    #[test]
    fn paused_side_scope_keeps_parent_tt_history_and_resumes_without_recharge() {
        let mut cpu = held_cpu();
        let position = Position::startpos();
        let cancel = AtomicBool::new(false);
        let partial = CpuLimits {
            max_depth: 1,
            max_nodes: 8,
            deadline: None,
        };
        let first = cpu.analyze(&position, partial, &cancel).unwrap();
        let mut token = first.resume.clone().unwrap();
        assert!(token.is_paused_stack() && cpu.token_is_current(&token));
        let parent_tt = format!("{:?}", cpu.tt);
        let parent_history = cpu.history.clone();
        let owner = cpu.paused_stack_snapshot().unwrap();
        let side = cpu
            .analyze_divergence(
                &position,
                &[BoardMove::from_uci("e2e4").unwrap()],
                partial,
                &cancel,
            )
            .unwrap();
        assert!(side.nodes > 0 && side.nodes <= partial.max_nodes);
        assert!(side.resume.is_none());
        assert_eq!(side.search_version, CPU_PAUSED_SIDE_SCOPE_SEARCH_VERSION);
        assert_eq!(format!("{:?}", cpu.tt), parent_tt);
        assert_eq!(cpu.history, parent_history);
        assert_eq!(cpu.paused_stack_snapshot().unwrap(), owner);
        assert!(cpu.token_is_current(&token));
        let side_observed = cpu.paused_side_scope_snapshot().unwrap();
        assert_eq!(side_observed.attempts, 1);
        assert_eq!(side_observed.work.nodes, side.nodes);
        assert!(side_observed.complete && side_observed.fixed_reserved_bytes > 0);
        let mut charged = first.nodes;
        let mut final_report = None;
        for _ in 0..256 {
            let next = cpu.resume(&position, &token, partial, &cancel).unwrap();
            assert!(!cpu.token_is_current(&token));
            charged += next.nodes;
            if next.completed_depth == 1 {
                final_report = Some(next);
                break;
            }
            token = next.resume.clone().unwrap();
            assert_eq!(token.cumulative_work().unwrap().nodes, charged);
        }
        let finished = final_report.expect("finite depth-one resume must finish");
        let mut baseline = held_cpu();
        let uninterrupted = baseline
            .analyze(
                &position,
                CpuLimits {
                    max_depth: 1,
                    max_nodes: 2048,
                    deadline: None,
                },
                &cancel,
            )
            .unwrap();
        assert_eq!(
            (finished.completed_depth, finished.score, finished.pv),
            (
                uninterrupted.completed_depth,
                uninterrupted.score,
                uninterrupted.pv
            )
        );
        assert_eq!(charged, uninterrupted.nodes);
        let closed = cpu.paused_stack_snapshot().unwrap();
        assert_eq!(closed.tokens_peak, 1);
        assert!(closed.bytes_peak <= CPU_PAUSED_STACK_MAX_BYTES as u64);
        assert_eq!(closed.replayed_consumed_work, 0);
        assert_eq!(cpu.paused_side_scope_snapshot().unwrap(), side_observed);
    }

    #[test]
    fn paused_side_scope_cancel_and_clear_invalidate_only_the_real_owner() {
        let mut cpu = held_cpu();
        let position = Position::startpos();
        let cancel = AtomicBool::new(false);
        let limited = CpuLimits {
            max_depth: 2,
            max_nodes: 1,
            deadline: None,
        };
        let token = cpu
            .analyze(&position, limited, &cancel)
            .unwrap()
            .resume
            .unwrap();
        let wrong_prefix = [BoardMove::from_uci("e2e5").unwrap()];
        assert!(
            cpu.analyze_divergence(&position, &wrong_prefix, limited, &cancel)
                .is_err()
        );
        assert!(cpu.token_is_current(&token));
        cancel.store(true, Ordering::Release);
        let side = cpu.analyze(&position, limited, &cancel).unwrap();
        assert_eq!(side.completion, CpuCompletion::Canceled);
        assert_eq!(side.nodes, 0);
        assert!(!cpu.token_is_current(&token));
        let actual = cpu.paused_stack_snapshot().unwrap();
        assert!(actual.complete && actual.tokens_retained == 0 && actual.bytes_current == 0);
        cancel.store(false, Ordering::Release);
        let fresh = cpu
            .analyze(&position, limited, &cancel)
            .unwrap()
            .resume
            .unwrap();
        assert!(cpu.token_is_current(&fresh));
        cpu.clear();
        assert!(!cpu.token_is_current(&fresh));
        assert!(
            cpu.paused_side_scope
                .as_ref()
                .unwrap()
                .tt
                .iter()
                .all(Option::is_none)
        );
        assert_eq!(cpu.paused_stack_snapshot().unwrap().tokens_peak, 1);
        assert_eq!(cpu.last_attempt_work(), None);
    }

    #[test]
    fn completed_iteration_has_no_paused_owner_snapshot_through_boxed_boundary() {
        let mut engine: Box<dyn CpuSearcher> = Box::new(
            CpuEngine::new(CpuConfig {
                tt_entries: 0,
                ..CpuConfig::default()
            })
            .unwrap(),
        );
        assert_eq!(engine.resume_policy(), CpuResumePolicy::CompletedIteration);
        assert_eq!(engine.paused_stack_snapshot(), None);
        engine
            .analyze(
                &Position::startpos(),
                CpuLimits {
                    max_depth: 1,
                    max_nodes: 1,
                    deadline: None,
                },
                &AtomicBool::new(false),
            )
            .unwrap();
        engine.clear();
        assert_eq!(engine.paused_stack_snapshot(), None);
    }

    fn engine() -> CpuEngine {
        CpuEngine::new(CpuConfig {
            tt_entries: 1024,
            max_depth: 10,
            ..CpuConfig::default()
        })
        .unwrap()
    }

    fn see_engine() -> CpuEngine {
        CpuEngine::with_ordering(
            CpuConfig {
                tt_entries: 16,
                max_depth: 4,
                quiescence_ply: 4,
                ..CpuConfig::default()
            },
            CpuOrderingPolicy::LegalSeeV1,
        )
        .unwrap()
    }

    fn ordering_control<'a>(
        cpu: &CpuEngine,
        position: &Position,
        cancel: &'a AtomicBool,
        limits: CpuLimits,
    ) -> Control<'a> {
        Control {
            limits,
            cancellation: cancel,
            cooperative: None,
            nodes: 0,
            quiescence_nodes: 0,
            tt_hits: 0,
            value: cpu.evaluator.initialize(position).unwrap(),
        }
    }

    #[test]
    fn legal_see_is_explicit_and_preserves_legacy_identity_and_resume_boundary() {
        let config = CpuConfig::default();
        let mut legacy = CpuEngine::new(config.clone()).unwrap();
        let mut see =
            CpuEngine::with_ordering(config.clone(), CpuOrderingPolicy::LegalSeeV1).unwrap();
        let expected = format!(
            "{CPU_SEARCH_CONDITIONS};profile={};max_depth={};q_plies={};tt_entries={}",
            config.profile.identity(),
            config.max_depth,
            config.quiescence_ply,
            config.tt_entries
        );
        assert_eq!(legacy.ordering_policy(), CpuOrderingPolicy::LegacyMvvLvaV1);
        assert_eq!(legacy.search_identity(), CPU_SEARCH_VERSION);
        assert_eq!(legacy.search_conditions(), expected);
        assert_eq!(CpuSearcher::search_identity(&see), CPU_SEE_SEARCH_VERSION);
        assert!(
            see.search_conditions()
                .contains(CpuOrderingPolicy::LegalSeeV1.identity())
        );
        assert!(
            see.search_conditions()
                .contains("node-work:search+qsearch+exchange")
        );
        assert_eq!(legacy.value_identity(), see.value_identity());
        assert_eq!(
            legacy.config().tt_allocation_bytes().unwrap(),
            see.config().tt_allocation_bytes().unwrap()
        );

        let position = Position::startpos();
        let cancel = AtomicBool::new(false);
        let report = legacy.analyze(&position, limits(1), &cancel).unwrap();
        assert_eq!(report.search_version, CPU_SEARCH_VERSION);
        let token = report.resume.unwrap();
        assert!(matches!(
            see.resume(&position, &token, limits(2), &cancel),
            Err(CpuError::ResumeMismatch(_))
        ));
        let report = see.analyze(&position, limits(1), &cancel).unwrap();
        assert_eq!(report.search_version, CPU_SEE_SEARCH_VERSION);
        assert_eq!(report.profile, config.profile);
        assert!(matches!(
            legacy.resume(&position, &report.resume.unwrap(), limits(2), &cancel),
            Err(CpuError::ResumeMismatch(_))
        ));
    }

    #[test]
    fn legal_see_places_losing_capture_after_quiet_but_keeps_tt_priority() {
        let position = Position::from_fen("7k/8/2p5/3p4/8/8/8/K2Q4 w - - 0 1").unwrap();
        let losing = "d1d5".parse().unwrap();
        let quiet = "a1b1".parse().unwrap();
        let cancel = AtomicBool::new(false);
        let legacy = engine();
        let see = see_engine();
        let mut old_moves = [quiet, losing];
        let mut control = ordering_control(&legacy, &position, &cancel, limits(1));
        legacy
            .order_moves(&position, &mut old_moves, None, &mut control)
            .unwrap();
        assert_eq!(old_moves, [losing, quiet]);
        assert_eq!(control.nodes, 0);
        let mut moves = old_moves;
        let mut control = ordering_control(&see, &position, &cancel, limits(1));
        see.order_moves(&position, &mut moves, None, &mut control)
            .unwrap();
        assert_eq!(moves, [quiet, losing]);
        assert!(control.nodes > 0);
        assert_eq!(control.quiescence_nodes, 0);
        let mut control = ordering_control(&see, &position, &cancel, limits(1));
        see.order_moves(&position, &mut moves, Some(losing), &mut control)
            .unwrap();
        assert_eq!(moves, [losing, quiet]);
        assert_eq!(control.nodes, 0);
    }

    #[test]
    fn legal_see_equal_keys_keep_rules_order_without_creating_tt_work() {
        let position = Position::startpos();
        let mut moves = position.ordered_legal_moves().moves().to_vec();
        let original = moves.clone();
        let cpu = see_engine();
        let cancel = AtomicBool::new(false);
        let mut control = ordering_control(&cpu, &position, &cancel, limits(1));
        cpu.order_moves(&position, &mut moves, None, &mut control)
            .unwrap();
        assert_eq!(moves, original);
        assert_eq!(control.nodes, 0);
        assert_eq!(control.tt_hits, 0);
    }

    #[test]
    fn legal_see_work_limit_is_typed_and_leaves_move_order_and_input_unchanged() {
        let position = Position::from_fen("7k/8/2p5/3p4/8/8/8/K2Q4 w - - 0 1").unwrap();
        let before = position.snapshot();
        let original = ["d1d5".parse().unwrap(), "a1b1".parse().unwrap()];
        let cpu = see_engine();
        let cancel = AtomicBool::new(false);
        let mut moves = original;
        let mut control = ordering_control(&cpu, &position, &cancel, limits(1));
        assert!(matches!(
            cpu.order_moves_with_see_budget(&position, &mut moves, None, &mut control, 1),
            Err(Abort::Error(CpuError::Unsupported(
                "legal SEE ordering exchange position budget exhausted"
            )))
        ));
        assert_eq!(control.nodes, 1);
        assert_eq!(moves, original);
        assert!(position.matches_snapshot(&before));
        let mut control = ordering_control(
            &cpu,
            &position,
            &cancel,
            CpuLimits {
                max_nodes: 1,
                ..limits(1)
            },
        );
        assert!(matches!(
            cpu.order_moves(&position, &mut moves, None, &mut control),
            Err(Abort::Stop(CpuCompletion::NodeLimit))
        ));
        assert_eq!(control.nodes, 1);
        assert_eq!(moves, original);
        assert!(position.matches_snapshot(&before));
    }

    #[test]
    fn legal_see_cancel_and_original_deadline_never_become_an_exchange_score() {
        let position = Position::from_fen("7k/8/2p5/3p4/8/8/8/K2Q4 w - - 0 1").unwrap();
        let original = ["d1d5".parse().unwrap(), "a1b1".parse().unwrap()];
        let cpu = see_engine();
        let canceled = AtomicBool::new(true);
        let mut moves = original;
        let mut control = ordering_control(&cpu, &position, &canceled, limits(1));
        assert!(matches!(
            cpu.order_moves(&position, &mut moves, None, &mut control),
            Err(Abort::Stop(CpuCompletion::Canceled))
        ));
        assert_eq!(control.nodes, 0);
        let cancel = AtomicBool::new(false);
        let mut control = ordering_control(
            &cpu,
            &position,
            &cancel,
            CpuLimits {
                deadline: Some(Instant::now()),
                ..limits(1)
            },
        );
        assert!(matches!(
            cpu.order_moves(&position, &mut moves, None, &mut control),
            Err(Abort::Stop(CpuCompletion::Deadline))
        ));
        assert_eq!(control.nodes, 0);
        assert_eq!(moves, original);
    }

    #[test]
    fn legal_see_entered_positions_are_in_actual_partial_search_work() {
        let position = Position::from_fen("7k/8/2p5/3p4/4P3/8/4Q3/K7 w - - 0 1").unwrap();
        let before = position.snapshot();
        let mut cpu = see_engine();
        let report = cpu
            .analyze(
                &position,
                CpuLimits {
                    max_nodes: 2,
                    ..limits(1)
                },
                &AtomicBool::new(false),
            )
            .unwrap();
        assert_eq!(report.search_version, CPU_SEE_SEARCH_VERSION);
        assert_eq!(report.completion, CpuCompletion::NodeLimit);
        assert_eq!(report.completed_depth, 0);
        assert_eq!(report.score_scope, CpuScoreScope::FrontierOnly);
        assert_eq!(report.nodes, 2);
        assert_eq!(cpu.last_attempt_work().unwrap().nodes, 2);
        assert!(position.legal_moves().contains(&report.best_move.unwrap()));
        assert!(position.matches_snapshot(&before));
    }

    fn limits(depth: u16) -> CpuLimits {
        CpuLimits {
            max_depth: depth,
            max_nodes: 30_000,
            deadline: None,
        }
    }

    fn run(fen: &str, depth: u16) -> (Position, CpuReport) {
        let position = Position::from_fen(fen).unwrap();
        let report = engine()
            .analyze(&position, limits(depth), &AtomicBool::new(false))
            .unwrap();
        (position, report)
    }

    #[test]
    fn mate_in_one_both_colors_is_certified_by_rules() {
        for fen in [
            "7k/5Q2/6K1/8/8/8/8/8 w - - 0 1",
            "8/8/8/8/8/6k1/5q2/7K b - - 0 1",
        ] {
            let (position, report) = run(fen, 1);
            let selected = report.best_move.unwrap();
            let child = position.preview_move(selected).unwrap();
            assert_eq!(
                child.classify_position().unwrap().play_status,
                PlayStatus::Terminal {
                    reason: TerminalReason::Checkmate,
                    winner: Some(position.side_to_move()),
                }
            );
            assert_eq!(report.score, CPU_MATE_SCORE - 1);
            assert_eq!(report.completed_depth, 1);
        }
    }

    #[test]
    fn terminal_and_stalemate_do_not_emit_a_fake_move() {
        let (_, mate) = run("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1", 2);
        assert_eq!(mate.score, -CPU_MATE_SCORE);
        assert_eq!(
            mate.completion,
            CpuCompletion::Terminal(TerminalReason::Checkmate)
        );
        assert_eq!(mate.best_move, None);
        let (_, stale) = run("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1", 2);
        assert_eq!(stale.score, 0);
        assert_eq!(
            stale.completion,
            CpuCompletion::Terminal(TerminalReason::Stalemate)
        );
        assert!(stale.pv.is_empty());
    }

    #[test]
    fn avoids_a_verified_opponent_mate_in_one_when_an_escape_exists() {
        let (position, report) = run("6k1/8/5K2/8/8/8/8/6Q1 b - - 0 1", 2);
        let permits_mate = |mv| {
            let child = position.preview_move(mv).unwrap();
            child.legal_moves().into_iter().any(|reply| {
                matches!(
                    child
                        .preview_move(reply)
                        .unwrap()
                        .classify_position()
                        .unwrap()
                        .play_status,
                    PlayStatus::Terminal {
                        reason: TerminalReason::Checkmate,
                        winner: Some(Color::White)
                    }
                )
            })
        };
        let legal = position.legal_moves();
        assert!(legal.iter().copied().any(permits_mate));
        assert!(legal.iter().copied().any(|mv| !permits_mate(mv)));
        assert_eq!(report.completed_depth, 2);
        assert!(!permits_mate(report.best_move.unwrap()));
        assert!(report.score > -MATE_THRESHOLD);
    }

    #[test]
    fn cancellation_deadline_and_node_limit_preserve_a_legal_fallback() {
        let position = Position::startpos();
        let mut engine = engine();
        let canceled = engine
            .analyze(&position, limits(2), &AtomicBool::new(true))
            .unwrap();
        assert_eq!(canceled.completion, CpuCompletion::Canceled);
        assert_eq!(canceled.completed_depth, 0);
        assert!(
            position
                .legal_moves()
                .contains(&canceled.best_move.unwrap())
        );
        assert_eq!(canceled.score_scope, CpuScoreScope::FrontierOnly);
        let expired = engine
            .analyze(
                &position,
                CpuLimits {
                    deadline: Some(Instant::now()),
                    ..limits(2)
                },
                &AtomicBool::new(false),
            )
            .unwrap();
        assert_eq!(expired.completion, CpuCompletion::Deadline);
        let bounded = engine
            .analyze(
                &position,
                CpuLimits {
                    max_nodes: 2,
                    ..limits(2)
                },
                &AtomicBool::new(false),
            )
            .unwrap();
        assert_eq!(bounded.completion, CpuCompletion::NodeLimit);
        assert_eq!(bounded.nodes, 2);
        assert_eq!(bounded.completed_depth, 0);
        assert!(engine.tt.iter().all(Option::is_none));
    }

    #[test]
    fn all_check_evasions_including_quiet_moves_are_searched() {
        // Black is checked by Ra1 and the only evasion is quiet Kb8.
        let position = Position::from_fen("k7/8/1K6/8/8/8/8/R7 b - - 0 1").unwrap();
        assert!(position.in_check());
        let legal = position.legal_moves();
        assert!(!legal.is_empty());
        assert!(legal.iter().all(|mv| !is_tactical(&position, *mv)));
        let mut position = position;
        let original = position.snapshot();
        let mut cpu = engine();
        let cancel = AtomicBool::new(false);
        let mut control = Control {
            limits: limits(1),
            cancellation: &cancel,
            cooperative: None,
            nodes: 0,
            quiescence_nodes: 0,
            tt_hits: 0,
            value: cpu.evaluator.initialize(&position).unwrap(),
        };
        let value = cpu
            .quiescence(&mut position, 0, 4, -INFINITY, INFINITY, &mut control)
            .unwrap();
        assert!(!value.pv.is_empty());
        assert!(legal.contains(&value.pv[0]));
        assert!(original.same_state(&position.snapshot()));
        let stopped = cpu.quiescence(&mut position, 0, 0, -INFINITY, INFINITY, &mut control);
        assert!(matches!(
            stopped,
            Err(Abort::Stop(CpuCompletion::QuiescenceLimit))
        ));
    }

    #[test]
    fn root_restrictions_cover_castling_en_passant_and_four_promotions() {
        let cases = [
            ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", vec!["e1g1", "e1c1"]),
            ("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 2", vec!["e5d6"]),
            (
                "4k3/P7/8/8/8/8/8/4K3 w - - 0 1",
                vec!["a7a8q", "a7a8r", "a7a8b", "a7a8n"],
            ),
        ];
        let mut cpu = engine();
        for (fen, moves) in cases {
            let position = Position::from_fen(fen).unwrap();
            let before = position.snapshot();
            for text in moves {
                let mv = BoardMove::from_uci(text).unwrap();
                let report = cpu
                    .analyze_root_moves(&position, &[mv], limits(1), &AtomicBool::new(false))
                    .unwrap();
                assert_eq!(report.best_move, Some(mv));
                assert_eq!(report.completed_depth, 1);
                assert!(report.root_restricted);
                assert!(before.same_state(&position.snapshot()));
            }
        }
    }

    #[test]
    fn exact_history_and_remaining_depth_are_required_for_tt_reuse() {
        let mut played = Position::startpos();
        played
            .apply_uci_moves(&["g1f3", "g8f6", "f3g1", "f6g8"])
            .unwrap();
        let imported = Position::from_fen(&played.to_fen()).unwrap();
        assert_ne!(played.position_identity(), imported.position_identity());
        let mut cpu = engine();
        let hash = tt_hash(&played);
        cpu.store(TtEntry {
            hash,
            identity: played.position_identity(),
            profile: CpuProfile::PlanAssisted,
            value_identity: Arc::clone(&cpu.value_identity),
            value_context: CpuValueCacheContext::default(),
            depth: 8,
            score: 1234,
            bound: Bound::Exact,
            best_move: played.legal_moves().first().copied(),
        });
        assert!(
            cpu.probe(
                hash,
                &played.position_identity(),
                &CpuValueCacheContext::default()
            )
            .is_some()
        );
        assert!(
            cpu.probe(
                hash,
                &imported.position_identity(),
                &CpuValueCacheContext::default()
            )
            .is_none()
        );
        let mut deeper = played.clone();
        let cancel = AtomicBool::new(false);
        let mut control = Control {
            limits: CpuLimits {
                max_nodes: 1,
                ..limits(10)
            },
            cancellation: &cancel,
            cooperative: None,
            nodes: 0,
            quiescence_nodes: 0,
            tt_hits: 0,
            value: cpu.evaluator.initialize(&deeper).unwrap(),
        };
        let result = cpu.negamax(&mut deeper, 10, 0, -INFINITY, INFINITY, &mut control);
        assert!(matches!(result, Err(Abort::Stop(CpuCompletion::NodeLimit))));
        assert_eq!(
            cpu.probe(
                hash,
                &played.position_identity(),
                &CpuValueCacheContext::default()
            )
            .unwrap()
            .depth,
            8
        );
    }

    #[test]
    fn resume_retains_completed_depth_and_rejects_changed_history_or_profile() {
        let mut cpu = engine();
        let position = Position::from_fen("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1").unwrap();
        let first = cpu
            .analyze(&position, limits(1), &AtomicBool::new(false))
            .unwrap();
        assert_eq!(first.completed_depth, 1);
        let token = first.resume.unwrap();
        let resumed = cpu
            .resume(&position, &token, limits(2), &AtomicBool::new(false))
            .unwrap();
        assert_eq!(resumed.completed_depth, 2);
        let changed = position.preview_move(position.legal_moves()[0]).unwrap();
        assert!(matches!(
            cpu.resume(&changed, &token, limits(2), &AtomicBool::new(false)),
            Err(CpuError::ResumeMismatch(_))
        ));
        let mut independent = CpuEngine::new(CpuConfig {
            profile: CpuProfile::Independent,
            ..CpuConfig::default()
        })
        .unwrap();
        assert!(matches!(
            independent.resume(&position, &token, limits(2), &AtomicBool::new(false)),
            Err(CpuError::ResumeMismatch(_))
        ));
        for config in [
            CpuConfig {
                tt_entries: cpu.config.tt_entries + 1,
                ..cpu.config.clone()
            },
            CpuConfig {
                max_depth: cpu.config.max_depth - 1,
                ..cpu.config.clone()
            },
        ] {
            let mut changed_conditions = CpuEngine::new(config).unwrap();
            assert!(matches!(
                changed_conditions.resume(&position, &token, limits(2), &AtomicBool::new(false)),
                Err(CpuError::ResumeMismatch(_))
            ));
        }
        let canceled = cpu
            .resume(&position, &token, limits(1), &AtomicBool::new(true))
            .unwrap();
        assert_eq!(canceled.completion, CpuCompletion::Canceled);
        assert_eq!(canceled.completed_depth, 1);
        assert_eq!(canceled.reused_completed_depth, 1);
        assert_eq!(canceled.nodes, 0);
        let expired = cpu
            .resume(
                &position,
                &token,
                CpuLimits {
                    deadline: Some(Instant::now()),
                    ..limits(1)
                },
                &AtomicBool::new(false),
            )
            .unwrap();
        assert_eq!(expired.completion, CpuCompletion::Deadline);
    }

    #[test]
    fn progress_observer_reports_only_fallback_and_completed_iterations() {
        let position = Position::from_fen("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1").unwrap();
        let mut progress = Vec::new();
        let report = engine()
            .analyze_with_progress(&position, limits(1), &AtomicBool::new(false), |value| {
                progress.push(value)
            })
            .unwrap();
        assert_eq!(progress.len(), 2);
        assert_eq!(progress[0].completed_depth, 0);
        assert_eq!(progress[1].completed_depth, 1);
        assert_eq!(progress[1].best_move, report.best_move);
        assert_eq!(progress[1].score, report.score);
        let mut stopped = Vec::new();
        let report = engine()
            .analyze_with_progress(
                &position,
                CpuLimits {
                    max_nodes: 1,
                    ..limits(2)
                },
                &AtomicBool::new(false),
                |value| stopped.push(value),
            )
            .unwrap();
        assert_eq!(report.completion, CpuCompletion::NodeLimit);
        assert_eq!(stopped.len(), 1);
        assert_eq!(stopped[0].completed_depth, 0);
        let canceled = AtomicBool::new(false);
        let report = engine()
            .analyze_with_progress(&position, limits(1), &canceled, |value| {
                if value.completed_depth == 1 {
                    canceled.store(true, Ordering::Release);
                }
            })
            .unwrap();
        assert_eq!(report.completion, CpuCompletion::Canceled);
        assert_eq!(report.completed_depth, 1);
        assert_eq!(report.score, CPU_MATE_SCORE - 1);
    }

    #[test]
    fn exact_float_delta_and_fresh_state_reuse_the_same_tt_value() {
        use crate::cpu_value::{CpuFloatCheckpoint, CpuFloatValue};
        let mut checkpoint = CpuFloatCheckpoint::zeros_untrained();
        for (square, kind, weight) in [
            (0, PieceKind::Rook, 1_000_000.0),
            (1, PieceKind::Knight, 0.03),
            (6, PieceKind::Knight, -1_000_000.0),
            (18, PieceKind::Knight, 0.04),
        ] {
            checkpoint.feature_weights[(kind as usize * 64 + square) * 256] = weight;
        }
        checkpoint.hidden_weights[0] = 1.0;
        checkpoint.hidden_weights[256] = 1.0;
        checkpoint.output_weights[0] = 1.0;
        let evaluator = Arc::new(CpuFloatValue::new(checkpoint).unwrap());
        let mut position = Position::startpos();
        let mut incremental = evaluator.initialize(&position).unwrap();
        let rules = position.make_uci("b1c3").unwrap();
        evaluator
            .apply_delta(&mut incremental, rules.delta())
            .unwrap();
        let mut cpu = CpuEngine::with_evaluator(
            CpuConfig {
                tt_entries: 128,
                ..CpuConfig::default()
            },
            evaluator.clone(),
        )
        .unwrap();
        let cancel = AtomicBool::new(false);
        let mut control = Control {
            limits: limits(1),
            cancellation: &cancel,
            cooperative: None,
            nodes: 0,
            quiescence_nodes: 0,
            tt_hits: 0,
            value: incremental,
        };
        let first = cpu
            .negamax(&mut position, 1, 0, -INFINITY, INFINITY, &mut control)
            .unwrap();
        let mut fresh = Control {
            limits: limits(1),
            cancellation: &cancel,
            cooperative: None,
            nodes: 0,
            quiescence_nodes: 0,
            tt_hits: 0,
            value: evaluator.initialize(&position).unwrap(),
        };
        assert_eq!(control.value.values(), fresh.value.values());
        let cached = cpu
            .negamax(&mut position, 1, 0, -INFINITY, INFINITY, &mut fresh)
            .unwrap();
        assert_eq!(cached.score, first.score);
        assert_eq!(fresh.tt_hits, 1);
        assert_eq!(fresh.nodes, 1);
        let mut uncached = Control {
            limits: limits(1),
            cancellation: &cancel,
            cooperative: None,
            nodes: 0,
            quiescence_nodes: 0,
            tt_hits: 0,
            value: evaluator.initialize(&position).unwrap(),
        };
        let actual = CpuEngine::with_evaluator(
            CpuConfig {
                tt_entries: 0,
                ..CpuConfig::default()
            },
            evaluator,
        )
        .unwrap()
        .negamax(&mut position, 1, 0, -INFINITY, INFINITY, &mut uncached)
        .unwrap();
        assert_eq!(actual.score, cached.score);
    }

    #[test]
    fn tt_does_not_change_full_search_score_and_pv_stays_legal() {
        let position = Position::from_fen("4k3/8/8/8/8/3p4/4P3/4K3 w - - 0 1").unwrap();
        let before = position.snapshot();
        let with = engine()
            .analyze(&position, limits(3), &AtomicBool::new(false))
            .unwrap();
        let without = CpuEngine::new(CpuConfig {
            tt_entries: 0,
            ..CpuConfig::default()
        })
        .unwrap()
        .analyze(&position, limits(3), &AtomicBool::new(false))
        .unwrap();
        assert_eq!(with.completed_depth, 3);
        assert_eq!(with.score, without.score);
        assert_eq!(with.best_move, without.best_move);
        let mut replay = position.clone();
        for mv in with.pv {
            replay.make_move(mv).unwrap();
        }
        assert!(before.same_state(&position.snapshot()));
    }

    #[test]
    fn mate_distance_round_trip_is_root_independent() {
        for score in [CPU_MATE_SCORE - 5, -CPU_MATE_SCORE + 9, 100, -300] {
            assert_eq!(from_tt_score(to_tt_score(score, 3), 3), score);
        }
        assert_eq!(
            from_tt_score(to_tt_score(CPU_MATE_SCORE - 5, 3), 1),
            CPU_MATE_SCORE - 3
        );
    }

    #[test]
    fn failed_search_retains_actual_work_and_clear_removes_observation() {
        use crate::cpu_value::{CpuAccumulatorUndo, CpuValueEvaluator};
        use rz_position::RuleMoveDelta;
        use std::sync::atomic::AtomicUsize;
        struct InjectFailure {
            base: BootstrapCpuValue,
            scores: AtomicUsize,
        }
        impl CpuValueEvaluator for InjectFailure {
            fn identity(&self) -> &CpuValueIdentity {
                self.base.identity()
            }
            fn provenance(&self) -> &'static str {
                self.base.provenance()
            }
            fn initialize(&self, p: &Position) -> Result<CpuAccumulator, CpuValueError> {
                self.base.initialize(p)
            }
            fn score(&self, a: &CpuAccumulator, p: &Position) -> Result<i32, CpuValueError> {
                if self.scores.fetch_add(1, Ordering::Relaxed) > 0 {
                    return Err(CpuValueError::NonFiniteForward);
                }
                self.base.score(a, p)
            }
            fn apply_delta(
                &self,
                a: &mut CpuAccumulator,
                d: &RuleMoveDelta,
            ) -> Result<CpuAccumulatorUndo, CpuValueError> {
                self.base.apply_delta(a, d)
            }
            fn restore(
                &self,
                a: &mut CpuAccumulator,
                u: CpuAccumulatorUndo,
                p: &Position,
            ) -> Result<(), CpuValueError> {
                self.base.restore(a, u, p)
            }
        }
        let position = Position::startpos();
        let before = position.snapshot();
        let mut cpu = CpuEngine::with_evaluator(
            CpuConfig::default(),
            Arc::new(InjectFailure {
                base: BootstrapCpuValue::default(),
                scores: AtomicUsize::new(0),
            }),
        )
        .unwrap();
        assert_eq!(cpu.last_attempt_work(), None);
        assert!(matches!(
            cpu.analyze(&position, limits(1), &AtomicBool::new(false)),
            Err(CpuError::Value(CpuValueError::NonFiniteForward))
        ));
        let actual = cpu.last_attempt_work().unwrap();
        assert!(actual.nodes > 0 && actual.nodes <= limits(1).max_nodes);
        assert!(actual.quiescence_nodes > 0 && actual.quiescence_nodes <= actual.nodes);
        assert!(before.same_state(&position.snapshot()));
        assert!(
            cpu.analyze(
                &position,
                CpuLimits {
                    max_nodes: 0,
                    ..limits(1)
                },
                &AtomicBool::new(false)
            )
            .is_err()
        );
        assert_eq!(cpu.last_attempt_work(), Some(CpuWork::default()));
        cpu.clear();
        assert_eq!(cpu.last_attempt_work(), None);
    }

    #[test]
    fn cpu_checker_is_replaceable_and_unsupported_resume_is_explicit() {
        struct NoResumeMock {
            base: CpuEngine,
        }
        impl CpuSearcher for NoResumeMock {
            fn config(&self) -> &CpuConfig {
                self.base.config()
            }
            fn value_identity(&self) -> &CpuValueIdentity {
                self.base.value_identity()
            }
            fn search_identity(&self) -> &'static str {
                "explicit-cpu-boundary-mock-v1"
            }
            fn search_conditions(&self) -> String {
                format!(
                    "explicit-no-resume-test-wrapper;{}",
                    self.base.search_conditions()
                )
            }
            fn capabilities(&self) -> CpuCapabilities {
                CpuCapabilities {
                    root_moves: false,
                    divergence: false,
                    completed_iteration_resume: false,
                    ..self.base.capabilities()
                }
            }
            fn clear(&mut self) {
                self.base.clear();
            }
            fn analyze(
                &mut self,
                p: &Position,
                l: CpuLimits,
                c: &AtomicBool,
            ) -> Result<CpuReport, CpuError> {
                let mut r = self.base.analyze(p, l, c)?;
                r.search_version = self.search_identity();
                r.resume = None;
                Ok(r)
            }
        }
        let position = Position::from_fen("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1").unwrap();
        let mut checker: Box<dyn CpuSearcher> = Box::new(engine());
        let first = checker
            .analyze(&position, limits(1), &AtomicBool::new(false))
            .unwrap();
        let token = first.resume.unwrap();
        checker = Box::new(NoResumeMock { base: engine() });
        assert!(!checker.capabilities().completed_iteration_resume);
        assert!(matches!(
            checker.resume(&position, &token, limits(2), &AtomicBool::new(false)),
            Err(CpuError::Unsupported(_))
        ));
        let second = checker
            .analyze(&position, limits(1), &AtomicBool::new(false))
            .unwrap();
        assert_eq!(second.search_version, "explicit-cpu-boundary-mock-v1");
        assert_eq!(second.value_identity, first.value_identity);
        assert_eq!(checker.last_attempt_work(), None);
    }

    #[test]
    fn tt_slot_allocation_bytes_are_bounded_layout_not_peak() {
        let slot_bytes = std::mem::size_of::<Option<TtEntry>>();
        for entries in [0, 1, 8192, MAX_TT_ENTRIES] {
            let config = CpuConfig {
                tt_entries: entries,
                ..CpuConfig::default()
            };
            assert_eq!(
                config.tt_allocation_bytes().unwrap(),
                (entries * slot_bytes) as u64
            );
        }
        for entries in [MAX_TT_ENTRIES + 1, usize::MAX] {
            assert!(matches!(
                CpuConfig {
                    tt_entries: entries,
                    ..CpuConfig::default()
                }
                .tt_allocation_bytes(),
                Err(CpuError::InvalidConfig(_))
            ));
        }
        assert_eq!(checked_inline_slot_bytes(0, usize::MAX).unwrap(), 0);
        assert!(matches!(
            checked_inline_slot_bytes(usize::MAX, 2),
            Err(CpuError::InvalidConfig(
                "TT slot bytes overflow address space"
            ))
        ));
    }

    #[test]
    fn invalid_budgets_and_root_moves_are_not_silent_fallbacks() {
        assert!(
            CpuEngine::new(CpuConfig {
                tt_entries: MAX_TT_ENTRIES + 1,
                ..CpuConfig::default()
            })
            .is_err()
        );
        let position = Position::startpos();
        let mut cpu = engine();
        assert!(
            cpu.analyze(
                &position,
                CpuLimits {
                    max_nodes: 0,
                    ..limits(1)
                },
                &AtomicBool::new(false)
            )
            .is_err()
        );
        assert!(
            cpu.analyze_root_moves(&position, &[], limits(1), &AtomicBool::new(false))
                .is_err()
        );
        let mv = BoardMove::from_uci("e2e4").unwrap();
        assert!(
            cpu.analyze_root_moves(&position, &[mv, mv], limits(1), &AtomicBool::new(false))
                .is_err()
        );
        let illegal = BoardMove::from_uci("e2e5").unwrap();
        assert!(
            cpu.analyze_root_moves(&position, &[illegal], limits(1), &AtomicBool::new(false))
                .is_err()
        );
    }
}
