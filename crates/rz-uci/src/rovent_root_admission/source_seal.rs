//! Original go/Rules/clock metadata before a live root capability is issued.
//!
//! Preparing a source preserves the existing `/1` descriptor and its actual
//! deadlines even when that original root has already stopped. Only the final
//! live gate can construct `RoventRootAdmission`. A stopped source exposes no
//! context, live clock, worker authority or reverse conversion to admission.

use super::{
    MAX_CANONICAL_BYTES, RoventOriginalGoCoverage, RoventRootAdmission, RoventRootAdmissionError,
    RoventRootAdmissionFailure, RoventRootAdmissionPolicy, authority_descriptor,
    canonical_original_subset, canonical_subset, codec_error, deadline_descriptor, fail, hex,
    move_texts, original_go_descriptor, parse_digest, require,
};
use crate::{contracts::SessionScope, engine::ProcessClock, search_driver::SearchSessionContext};
use rz_contracts::{Deadline, Digest, ErrorCode};
use rz_experiments::{
    RoventInputLock, RoventResolvedGo, RoventRootAllocationClock, RoventRootAllocationInput,
    RoventRootClocks, RoventRootDeclaredCeilings, RoventRootDescriptor, RoventRootDescriptorInput,
};
use rz_position::{BoardMove, PlayStatus, Position, PositionSnapshot};
use rz_runtime::contracts::{ContractClock, ContractSystemClock};
use std::sync::{Arc, atomic::Ordering};

/// A reason observed from the original UCI owner at the live admission gate.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum InitialStopReason {
    RootReplaced,
    Canceled,
    Deadline,
}

/// A live admission and an initially stopped source carry different authority.
// This is one final-gate outcome per root, not a queue. Preserve the original
// admission/stopped-source custody and their distinct capabilities.
#[allow(clippy::large_enum_variant)]
#[derive(Debug)]
pub enum InitialAdmissionOutcome {
    Admitted(RoventRootAdmission),
    StoppedInitial(StoppedInitial),
}

/// Validated original source metadata, not a live admission capability.
/// The retained owner exists only for one final gate; it is never exposed.
#[derive(Debug)]
pub struct SourceSeal {
    policy: RoventRootAdmissionPolicy,
    startup_lock: Digest,
    state_and_history: Digest,
    rules_profile: Digest,
    snapshot: PositionSnapshot,
    context: SearchSessionContext,
    runtime_clock: ContractSystemClock,
    root_moves: Option<Arc<[BoardMove]>>,
    allowed_moves: Arc<[BoardMove]>,
    fallback: Option<BoardMove>,
    hard: Deadline,
    soft: Deadline,
    admission: Deadline,
    cleanup: Deadline,
    descriptor: RoventRootDescriptor,
    root_budget: Digest,
    deadline_seal: Digest,
    digest: Digest,
    canonical: Vec<u8>,
    allocation_clock: RoventRootAllocationClock,
    allocation_clock_digest: Digest,
    allocation_clock_canonical: Vec<u8>,
    rejected: Option<InitialStopReason>,
}

/// Read-only source evidence for an original root stopped before live work.
/// It contains no context, live clock, worker or physical-completion proof.
#[derive(Debug)]
pub struct StoppedInitial {
    startup_lock: Digest,
    state_and_history: Digest,
    rules_profile: Digest,
    descriptor: RoventRootDescriptor,
    root_budget: Digest,
    deadline_seal: Digest,
    digest: Digest,
    canonical: Vec<u8>,
    allocation_clock: RoventRootAllocationClock,
    allocation_clock_digest: Digest,
    allocation_clock_canonical: Vec<u8>,
    allowed_moves: Arc<[BoardMove]>,
    fallback: Option<BoardMove>,
    stop_reason: InitialStopReason,
}

/// A /2 source prepared from the actual Rules/Start/clock inputs. It retains
/// the exact original lock owner and context for one final-gate decision.
#[derive(Debug)]
pub struct PreparedSourceV2 {
    owner: rz_experiments::RoventOriginalInputLock,
    core: PreparedRootCore,
    descriptor: rz_experiments::RoventRootDescriptorV2,
    allocation_clock: rz_experiments::RoventRootAllocationClockV2,
    rejected: Option<InitialStopReason>,
}

/// A /2 live admission, issued only while the original current-owner mutex is
/// held. The original /2 lock and revocation context remain owned; this is
/// never projected into a /1 admission.
#[derive(Debug)]
pub struct RoventRootAdmissionV2 {
    owner: rz_experiments::RoventOriginalInputLock,
    policy: RoventRootAdmissionPolicy,
    startup_lock: Digest,
    state_and_history: Digest,
    rules_profile: Digest,
    snapshot: PositionSnapshot,
    context: SearchSessionContext,
    runtime_clock: ContractSystemClock,
    root_moves: Option<Arc<[BoardMove]>>,
    allowed_moves: Arc<[BoardMove]>,
    hard: Deadline,
    soft: Deadline,
    admission: Deadline,
    cleanup: Deadline,
    descriptor: rz_experiments::RoventRootDescriptorV2,
    allocation_clock: rz_experiments::RoventRootAllocationClockV2,
}

/// An originally stopped /2 source retains metadata but no live context.
#[derive(Debug)]
pub struct StoppedInitialV2 {
    owner: rz_experiments::RoventOriginalInputLock,
    startup_lock: Digest,
    state_and_history: Digest,
    rules_profile: Digest,
    descriptor: rz_experiments::RoventRootDescriptorV2,
    allocation_clock: rz_experiments::RoventRootAllocationClockV2,
    allowed_moves: Arc<[BoardMove]>,
    fallback: Option<BoardMove>,
    stop_reason: InitialStopReason,
}

#[allow(clippy::large_enum_variant)]
#[derive(Debug)]
pub enum InitialAdmissionOutcomeV2 {
    Admitted(RoventRootAdmissionV2),
    StoppedInitial(StoppedInitialV2),
}

impl PreparedSourceV2 {
    pub fn prepare(
        owner: &rz_experiments::RoventOriginalInputLock,
        policy: RoventRootAdmissionPolicy,
        position: &Position,
        context: &SearchSessionContext,
        clock: &ProcessClock,
    ) -> Result<Self, RoventRootAdmissionError> {
        owner.validate().map_err(|_| {
            fail(
                RoventRootAdmissionFailure::InvalidLock,
                "original startup input lock",
            )
        })?;
        let lock = owner.v2().ok_or_else(|| {
            fail(
                RoventRootAdmissionFailure::InvalidLock,
                "original /2 startup input lock required",
            )
        })?;
        let declarations = SourceDeclarations::from_v2(lock);
        let core = prepare_root_core(&declarations, policy, position, context, clock)?;
        let descriptor =
            rz_experiments::RoventRootDescriptorV2::seal(core.descriptor_input.clone(), lock)
                .map_err(codec_error)?;
        let digest = descriptor.digest().map_err(codec_error)?;
        let allocation_clock = rz_experiments::RoventRootAllocationClockV2::seal(
            core.allocation_input(digest, &declarations),
            &descriptor,
            lock,
        )
        .map_err(codec_error)?;
        Ok(Self {
            owner: owner.clone(),
            core,
            descriptor,
            allocation_clock,
            rejected: None,
        })
    }

    /// These are declarations and owner identity, not an execution receipt.
    pub fn owner(&self) -> &rz_experiments::RoventOriginalInputLock {
        &self.owner
    }
    pub fn descriptor(&self) -> &rz_experiments::RoventRootDescriptorV2 {
        &self.descriptor
    }
    pub fn allocation_clock(&self) -> &rz_experiments::RoventRootAllocationClockV2 {
        &self.allocation_clock
    }
    pub fn startup_lock_digest(&self) -> Digest {
        self.core.startup_lock
    }

    /// A rejected source retains its original owner and cannot later mint live
    /// authority after cancellation, root replacement, or expiry is reversed.
    #[allow(clippy::result_large_err)]
    pub fn try_into_live(self) -> Result<RoventRootAdmissionV2, (Self, RoventRootAdmissionError)> {
        if let Some(reason) = self.rejected {
            return Err((self, stopped_error(reason)));
        }
        let context = self.core.context.clone();
        let runtime_clock = self.core.runtime_clock.clone();
        let hard = self.core.hard;
        let mut source = Some(self);
        match with_live_gate(&context, &runtime_clock, hard, || {
            let source = source.take().expect("one original /2 source");
            RoventRootAdmissionV2 {
                owner: source.owner,
                policy: source.core.policy,
                startup_lock: source.core.startup_lock,
                state_and_history: source.core.state_and_history,
                rules_profile: source.core.rules_profile,
                snapshot: source.core.snapshot,
                context: source.core.context,
                runtime_clock: source.core.runtime_clock,
                root_moves: source.core.root_moves,
                allowed_moves: source.core.allowed_moves,
                hard: source.core.hard,
                soft: source.core.soft,
                admission: source.core.admission,
                cleanup: source.core.cleanup,
                descriptor: source.descriptor,
                allocation_clock: source.allocation_clock,
            }
        }) {
            Ok(admitted) => Ok(admitted),
            Err((reason, error)) => {
                let mut source = source.expect("unissued original /2 source");
                source.rejected = reason;
                Err((source, error))
            }
        }
    }

    pub fn into_initial(self) -> Result<InitialAdmissionOutcomeV2, RoventRootAdmissionError> {
        match self.try_into_live() {
            Ok(admitted) => Ok(InitialAdmissionOutcomeV2::Admitted(admitted)),
            Err((source, error)) => match source.rejected {
                Some(stop_reason) => Ok(InitialAdmissionOutcomeV2::StoppedInitial(
                    StoppedInitialV2 {
                        owner: source.owner,
                        startup_lock: source.core.startup_lock,
                        state_and_history: source.core.state_and_history,
                        rules_profile: source.core.rules_profile,
                        descriptor: source.descriptor,
                        allocation_clock: source.allocation_clock,
                        allowed_moves: source.core.allowed_moves,
                        fallback: source.core.fallback,
                        stop_reason,
                    },
                )),
                None => Err(error),
            },
        }
    }
}

impl RoventRootAdmissionV2 {
    pub fn owner(&self) -> &rz_experiments::RoventOriginalInputLock {
        &self.owner
    }
    pub fn descriptor(&self) -> &rz_experiments::RoventRootDescriptorV2 {
        &self.descriptor
    }
    pub fn allocation_clock(&self) -> &rz_experiments::RoventRootAllocationClockV2 {
        &self.allocation_clock
    }
    pub fn startup_lock_digest(&self) -> Digest {
        self.startup_lock
    }
    pub fn policy(&self) -> RoventRootAdmissionPolicy {
        self.policy
    }
    pub fn state_and_history_digest(&self) -> Digest {
        self.state_and_history
    }
    pub fn rules_profile_digest(&self) -> Digest {
        self.rules_profile
    }
    pub fn authority(&self) -> rz_contracts::pals::SearchAuthority {
        self.context.authority
    }
    pub fn original_go(&self) -> &crate::parser::GoLimits {
        self.context
            .original_go
            .as_deref()
            .expect("admitted original go")
    }
    pub fn requested_nodes(&self) -> Option<u64> {
        self.context.requested_nodes
    }
    pub fn goal(&self) -> rz_contracts::rovent::SearchGoal {
        self.context.goal
    }
    pub fn declared_ceilings(&self) -> &RoventRootDeclaredCeilings {
        &self.descriptor.input.declared_ceilings
    }
    pub fn original_context(&self) -> &SearchSessionContext {
        &self.context
    }
    pub fn root_snapshot(&self) -> &PositionSnapshot {
        &self.snapshot
    }
    pub fn allowed_root_moves(&self) -> &Arc<[BoardMove]> {
        &self.allowed_moves
    }
    pub fn root_moves(&self) -> Option<&Arc<[BoardMove]>> {
        self.root_moves.as_ref()
    }
    pub fn runtime_clock(&self) -> &ContractSystemClock {
        &self.runtime_clock
    }
    pub fn hard_deadline(&self) -> Deadline {
        self.hard
    }
    pub fn soft_deadline(&self) -> Deadline {
        self.soft
    }
    pub fn admission_deadline(&self) -> Deadline {
        self.admission
    }
    pub fn cleanup_deadline(&self) -> Deadline {
        self.cleanup
    }
}

impl StoppedInitialV2 {
    pub fn owner(&self) -> &rz_experiments::RoventOriginalInputLock {
        &self.owner
    }
    pub fn startup_lock_digest(&self) -> Digest {
        self.startup_lock
    }
    pub fn state_and_history_digest(&self) -> Digest {
        self.state_and_history
    }
    pub fn rules_profile_digest(&self) -> Digest {
        self.rules_profile
    }
    pub fn descriptor(&self) -> &rz_experiments::RoventRootDescriptorV2 {
        &self.descriptor
    }
    pub fn allocation_clock(&self) -> &rz_experiments::RoventRootAllocationClockV2 {
        &self.allocation_clock
    }
    pub fn allowed_root_moves(&self) -> &Arc<[BoardMove]> {
        &self.allowed_moves
    }
    pub fn fallback(&self) -> Option<BoardMove> {
        self.fallback
    }
    pub fn stop_reason(&self) -> InitialStopReason {
        self.stop_reason
    }
}

impl SourceSeal {
    /// Validate and seal the actual original source without granting live work.
    /// No start, deadline or clock origin is created or inferred here.
    pub fn prepare(
        lock: &RoventInputLock,
        policy: RoventRootAdmissionPolicy,
        position: &Position,
        context: &SearchSessionContext,
        clock: &ProcessClock,
    ) -> Result<Self, RoventRootAdmissionError> {
        lock.validate().map_err(|_| {
            fail(
                RoventRootAdmissionFailure::InvalidLock,
                "original startup input lock",
            )
        })?;
        let declarations = SourceDeclarations::from_v1(lock);
        let core = prepare_root_core(&declarations, policy, position, context, clock)?;
        let descriptor =
            RoventRootDescriptor::seal(core.descriptor_input.clone()).map_err(codec_error)?;
        descriptor.verify_against(lock).map_err(codec_error)?;
        let root_budget = descriptor.root_budget_digest().map_err(codec_error)?;
        let deadline_seal = descriptor.deadline_seal_digest().map_err(codec_error)?;
        let canonical = descriptor.canonical_bytes().map_err(codec_error)?;
        let digest = descriptor.digest().map_err(codec_error)?;
        let allocation_clock = RoventRootAllocationClock::seal(
            core.allocation_input(digest, &declarations),
            &descriptor,
            lock,
        )
        .map_err(codec_error)?;
        let allocation_clock_digest = allocation_clock.digest().map_err(codec_error)?;
        let allocation_clock_canonical = allocation_clock.canonical_bytes().map_err(codec_error)?;
        let PreparedRootCore {
            policy,
            startup_lock,
            state_and_history,
            rules_profile,
            snapshot,
            context,
            runtime_clock,
            root_moves,
            allowed_moves,
            fallback,
            hard,
            soft,
            admission,
            cleanup,
            ..
        } = core;
        Ok(Self {
            policy,
            startup_lock,
            state_and_history,
            rules_profile,
            snapshot,
            context,
            runtime_clock,
            root_moves,
            allowed_moves,
            fallback,
            hard,
            soft,
            admission,
            cleanup,
            descriptor,
            root_budget,
            deadline_seal,
            digest,
            canonical,
            allocation_clock,
            allocation_clock_digest,
            allocation_clock_canonical,
            rejected: None,
        })
    }

    /// Issue live authority only after both original acceptance and hard gate.
    /// Every rejection retains this exact source for a typed stopped outcome.
    // Return the original seal and error without a fresh failure allocation;
    // the seal's rejected state still prevents a later live-admission retry.
    #[allow(clippy::result_large_err)]
    pub fn try_into_live(self) -> Result<RoventRootAdmission, (Self, RoventRootAdmissionError)> {
        // A rejected source cannot be retried into a live capability, even if a
        // caller later changes shared state to resemble the original root.
        if let Some(reason) = self.rejected {
            return Err((self, stopped_error(reason)));
        }
        let context = self.context.clone();
        let runtime_clock = self.runtime_clock.clone();
        let hard = self.hard;
        let mut source = Some(self);
        match with_live_gate(&context, &runtime_clock, hard, || {
            let source = source.take().expect("one original /1 source");
            RoventRootAdmission {
                policy: source.policy,
                startup_lock: source.startup_lock,
                state_and_history: source.state_and_history,
                rules_profile: source.rules_profile,
                snapshot: source.snapshot,
                context: source.context,
                runtime_clock: source.runtime_clock,
                root_moves: source.root_moves,
                allowed_moves: source.allowed_moves,
                hard: source.hard,
                soft: source.soft,
                admission: source.admission,
                cleanup: source.cleanup,
                descriptor: source.descriptor,
                root_budget: source.root_budget,
                deadline_seal: source.deadline_seal,
                digest: source.digest,
                canonical: source.canonical,
                allocation_clock: source.allocation_clock,
                allocation_clock_digest: source.allocation_clock_digest,
                allocation_clock_canonical: source.allocation_clock_canonical,
            }
        }) {
            Ok(admitted) => Ok(admitted),
            Err((reason, error)) => {
                let mut source = source.expect("unissued original /1 source");
                source.rejected = reason;
                Err((source, error))
            }
        }
    }

    /// Resolve one original source; only a captured control stop is normal.
    /// A deadline crossed during preparation or the final gate retains source.
    pub fn into_initial(self) -> Result<InitialAdmissionOutcome, RoventRootAdmissionError> {
        match self.try_into_live() {
            Ok(admitted) => Ok(InitialAdmissionOutcome::Admitted(admitted)),
            Err((source, error)) => match source.rejected {
                Some(stop_reason) => Ok(InitialAdmissionOutcome::StoppedInitial(StoppedInitial {
                    startup_lock: source.startup_lock,
                    state_and_history: source.state_and_history,
                    rules_profile: source.rules_profile,
                    descriptor: source.descriptor,
                    root_budget: source.root_budget,
                    deadline_seal: source.deadline_seal,
                    digest: source.digest,
                    canonical: source.canonical,
                    allocation_clock: source.allocation_clock,
                    allocation_clock_digest: source.allocation_clock_digest,
                    allocation_clock_canonical: source.allocation_clock_canonical,
                    allowed_moves: source.allowed_moves,
                    fallback: source.fallback,
                    stop_reason,
                })),
                None => Err(error),
            },
        }
    }

    pub fn allocation_clock(&self) -> &RoventRootAllocationClock {
        &self.allocation_clock
    }
    pub fn allocation_clock_digest(&self) -> Digest {
        self.allocation_clock_digest
    }
    pub fn allocation_clock_canonical_bytes(&self) -> &[u8] {
        &self.allocation_clock_canonical
    }
}

struct SourceDeclarations<'a> {
    canonical_sha256: &'a str,
    search: &'a rz_experiments::RoventSearchDeclaration,
    coordinator: &'a rz_experiments::RoventCoordinatorDeclaration,
    resources: &'a rz_experiments::RoventResourcePlan,
    wall_time_max_ms: u64,
    cleanup_max_ms: u64,
    output_bytes_max: u64,
}

impl<'a> SourceDeclarations<'a> {
    fn from_v1(lock: &'a RoventInputLock) -> Self {
        let m = &lock.manifest;
        Self {
            canonical_sha256: &lock.canonical_sha256,
            search: &m.engine.search,
            coordinator: &m.coordinator,
            resources: &m.resources,
            wall_time_max_ms: m.wall_time_max_ms,
            cleanup_max_ms: m.cleanup_max_ms,
            output_bytes_max: m.output_bytes_max,
        }
    }
    fn from_v2(lock: &'a rz_experiments::RoventInputLockV2) -> Self {
        let m = &lock.manifest;
        Self {
            canonical_sha256: &lock.canonical_sha256,
            search: &m.engine.search,
            coordinator: &m.coordinator,
            resources: &m.resources,
            wall_time_max_ms: m.wall_time_max_ms,
            cleanup_max_ms: m.cleanup_max_ms,
            output_bytes_max: m.output_bytes_max,
        }
    }
}

#[derive(Debug)]
struct PreparedRootCore {
    policy: RoventRootAdmissionPolicy,
    startup_lock: Digest,
    state_and_history: Digest,
    rules_profile: Digest,
    snapshot: PositionSnapshot,
    context: SearchSessionContext,
    runtime_clock: ContractSystemClock,
    root_moves: Option<Arc<[BoardMove]>>,
    allowed_moves: Arc<[BoardMove]>,
    fallback: Option<BoardMove>,
    hard: Deadline,
    soft: Deadline,
    admission: Deadline,
    cleanup: Deadline,
    descriptor_input: RoventRootDescriptorInput,
    allocation_start: Deadline,
    allocation_output: Deadline,
    allocation_ns: u64,
}

impl PreparedRootCore {
    fn allocation_input(
        &self,
        descriptor_digest: Digest,
        declarations: &SourceDeclarations<'_>,
    ) -> RoventRootAllocationInput {
        RoventRootAllocationInput {
            root_descriptor_sha256: hex(descriptor_digest),
            startup_lock_sha256: hex(self.startup_lock),
            authority: authority_descriptor(self.context.authority),
            start: deadline_descriptor(self.allocation_start),
            output: deadline_descriptor(self.allocation_output),
            allocation_ns: self.allocation_ns,
            wall_time_max_ms: declarations.wall_time_max_ms,
            cleanup_max_ms: declarations.cleanup_max_ms,
        }
    }
}

fn prepare_root_core(
    declarations: &SourceDeclarations<'_>,
    policy: RoventRootAdmissionPolicy,
    position: &Position,
    context: &SearchSessionContext,
    clock: &ProcessClock,
) -> Result<PreparedRootCore, RoventRootAdmissionError> {
    let original_go = context.original_go.as_deref().ok_or_else(|| {
        fail(
            RoventRootAdmissionFailure::OriginalGoMissing,
            "actual admission requires the original parsed UCI go",
        )
    })?;
    // Preserve the current bounded `/1` metadata path. Raw repetition is
    // retained in the parsed go and is not a distinct-move limit.
    if let Some(moves) = &original_go.searchmoves {
        let bytes = moves.iter().try_fold(0usize, |bytes, movement| {
            bytes.checked_add(movement.len())?.checked_add(3)
        });
        require(
            bytes.is_some_and(|bytes| bytes <= MAX_CANONICAL_BYTES / 2),
            RoventRootAdmissionFailure::CanonicalEncoding,
            "original go exceeds the bounded root descriptor budget",
        )?;
    }
    require(
        original_go.nodes == context.requested_nodes
            && original_go.depth_plies == context.goal.depth_plies
            && original_go.mate_moves == context.goal.mate_moves,
        RoventRootAdmissionFailure::OriginalGoMismatch,
        "original parsed go nodes or goal differ from resolved search context",
    )?;
    let runtime_clock = clock.runtime_clock();
    require(
        clock.epoch().0 != 0
            && context.authority.epoch == clock.epoch()
            && runtime_clock.domain() == clock.domain()
            && context.authority.implementation != Digest([0; 32]),
        RoventRootAdmissionFailure::Clock,
        "original authority and process clock domains differ",
    )?;
    let convert = |instant| {
        clock.deadline(instant).map_err(|_| {
            fail(
                RoventRootAdmissionFailure::Clock,
                "deadline predates or overflows the original process origin",
            )
        })
    };
    require(
        convert(runtime_clock.origin())?.at.0 == 0,
        RoventRootAdmissionFailure::Clock,
        "runtime and UCI process origins differ",
    )?;
    let hard = convert(context.control.deadline)?;
    let soft = convert(context.control.soft_deadline)?;
    let admission = convert(context.control.admission_deadline)?;
    let cleanup = convert(context.shutdown_deadline)?;
    require(
        soft.at <= admission.at && admission.at <= hard.at && hard.at <= cleanup.at,
        RoventRootAdmissionFailure::DeadlineOrder,
        "original soft/admission/hard/cleanup deadline order",
    )?;
    let cleanup_limit = declarations
        .cleanup_max_ms
        .checked_mul(1_000_000)
        .ok_or_else(|| {
            fail(
                RoventRootAdmissionFailure::DeclaredCeiling,
                "declared cleanup ceiling overflows nanoseconds",
            )
        })?;
    require(
        cleanup.at.0 - hard.at.0 <= cleanup_limit,
        RoventRootAdmissionFailure::DeclaredCeiling,
        "original cleanup exceeds the declared cleanup ceiling",
    )?;
    // The producer copies this once from the actual binding. The source
    // never reconstructs start or allocation from a deadline or sampled now.
    let budget = context.original_time_budget.as_ref().ok_or_else(|| {
        fail(
            RoventRootAdmissionFailure::Clock,
            "actual admission requires the original allocated TimeBudget",
        )
    })?;
    require(
        budget.soft_deadline == context.control.soft_deadline
            && budget.admission_deadline == context.control.admission_deadline
            && budget.hard_deadline == context.control.deadline,
        RoventRootAdmissionFailure::Clock,
        "original TimeBudget boundaries differ from the actual search control",
    )?;
    require(
        budget.start <= budget.soft_deadline
            && budget.soft_deadline <= budget.admission_deadline
            && budget.admission_deadline <= budget.hard_deadline
            && budget.hard_deadline <= budget.output_deadline
            && budget.output_deadline <= context.shutdown_deadline,
        RoventRootAdmissionFailure::DeadlineOrder,
        "original start/soft/admission/hard/output/cleanup deadline order",
    )?;
    require(
        budget.output_deadline.checked_duration_since(budget.start) == Some(budget.allocation),
        RoventRootAdmissionFailure::Clock,
        "original TimeBudget allocation differs from output minus actual start",
    )?;
    let allocation_ns = u64::try_from(budget.allocation.as_nanos()).map_err(|_| {
        fail(
            RoventRootAdmissionFailure::Clock,
            "original allocation overflows process-clock nanoseconds",
        )
    })?;
    let wall_limit = declarations
        .wall_time_max_ms
        .checked_mul(1_000_000)
        .ok_or_else(|| {
            fail(
                RoventRootAdmissionFailure::DeclaredCeiling,
                "declared wall ceiling overflows nanoseconds",
            )
        })?;
    require(
        allocation_ns <= wall_limit,
        RoventRootAdmissionFailure::DeclaredCeiling,
        "original whole allocation exceeds the declared wall ceiling",
    )?;
    let allocation_start = convert(budget.start)?;
    let allocation_output = convert(budget.output_deadline)?;
    let registered = declarations.search;
    context
        .goal
        .validate(registered.registered_max_chess_plies)
        .map_err(|_| {
            fail(
                RoventRootAdmissionFailure::Goal,
                "go exceeds registered chess-ply capability",
            )
        })?;
    let snapshot = position.snapshot();
    let state_and_history = rz_position::contracts::rules_snapshot_semantic_digest(&snapshot);
    let rules_profile = rz_position::contracts::profile_digest();
    require(
        registered.rules_profile_sha256 == hex(rules_profile),
        RoventRootAdmissionFailure::RulesProfile,
        "registered Rules profile differs from actual Rules",
    )?;
    let legal = position.ordered_legal_moves();
    let root_moves = canonical_subset(legal.moves(), context.root_moves.as_deref())?;
    let parsed_root_moves =
        canonical_original_subset(legal.moves(), original_go.searchmoves.as_deref())?;
    require(
        parsed_root_moves.as_deref() == root_moves.as_deref(),
        RoventRootAdmissionFailure::OriginalGoMismatch,
        "original parsed go root subset differs from resolved search context",
    )?;
    let allowed_moves = root_moves
        .clone()
        .unwrap_or_else(|| Arc::from(legal.moves()));
    let fallback = match position.play_status_from_view(&legal).map_err(|_| {
        fail(
            RoventRootAdmissionFailure::RootMoves,
            "original Rules view cannot classify its own root",
        )
    })? {
        PlayStatus::Ongoing => allowed_moves.first().copied(),
        PlayStatus::Terminal { .. } => None,
    };
    if policy == RoventRootAdmissionPolicy::ExactRegistered {
        let declared = registered
            .searchmoves
            .as_ref()
            .map(|moves| {
                moves
                    .iter()
                    .map(|movement| {
                        BoardMove::from_uci(movement).map_err(|_| {
                            fail(
                                RoventRootAdmissionFailure::RegisteredWorkload,
                                "registered root move decoding",
                            )
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()
            })
            .transpose()?;
        let declared = canonical_subset(legal.moves(), declared.as_deref())?;
        require(
            registered.root_state_and_history_sha256 == hex(state_and_history)
                && registered.requested_nodes == context.requested_nodes
                && registered.goal() == context.goal
                && declared.as_deref() == root_moves.as_deref(),
            RoventRootAdmissionFailure::RegisteredWorkload,
            "actual root/history/go/subset differs from exact registered workload",
        )?;
    }
    let original_start = context.original_start.as_ref().ok_or_else(|| {
        fail(
            RoventRootAdmissionFailure::Clock,
            "actual Start source missing",
        )
    })?;
    original_start
        .verify(position, context, clock)
        .map_err(|error| {
            let mut failure = fail(
                RoventRootAdmissionFailure::Clock,
                "actual Start source mismatch",
            );
            failure.metadata_error = Some(error.to_string());
            failure
        })?;
    let startup_lock = parse_digest(declarations.canonical_sha256)?;
    let declared_ceilings = RoventRootDeclaredCeilings {
        resource_plan: declarations.resources.clone(),
        registered_max_chess_plies: registered.registered_max_chess_plies,
        cpu_credit_limit: declarations.coordinator.cpu_credit_limit,
        questions_max: declarations.coordinator.questions_max,
        ancestor_depth_max: declarations.coordinator.ancestor_depth_max,
        waiting_jobs_max: declarations.coordinator.waiting_jobs_max,
        role_tickets_max: declarations.coordinator.role_tickets_max,
        wall_time_max_ms: declarations.wall_time_max_ms,
        cleanup_max_ms: declarations.cleanup_max_ms,
        output_bytes_max: declarations.output_bytes_max,
    };
    // The descriptor remains pure metadata. It cannot mint a live owner.
    let descriptor_input = RoventRootDescriptorInput {
        policy,
        startup_lock_sha256: hex(startup_lock),
        state_and_history_sha256: hex(state_and_history),
        rules_profile_sha256: hex(rules_profile),
        authority: authority_descriptor(context.authority),
        node_unit: registered.node_unit.clone(),
        search_configuration_sha256: registered.search_configuration_sha256.clone(),
        coordinator_policy_sha256: declarations.coordinator.policy_sha256.clone(),
        original_go: RoventResolvedGo {
            coverage: RoventOriginalGoCoverage::FullOriginalGo,
            requested_nodes: context.requested_nodes,
            depth_plies: context.goal.depth_plies,
            mate_moves: context.goal.mate_moves,
            searchmoves: root_moves.as_ref().map(|moves| move_texts(moves)),
            transport_simulation_cap: context.control.max_simulations,
            original_parsed_limits: Some(original_go_descriptor(original_go)),
        },
        allowed_root_moves: move_texts(&allowed_moves),
        clock: RoventRootClocks {
            hard: deadline_descriptor(hard),
            soft: deadline_descriptor(soft),
            admission: deadline_descriptor(admission),
            cleanup: deadline_descriptor(cleanup),
        },
        declared_ceilings,
    };
    Ok(PreparedRootCore {
        policy,
        startup_lock,
        state_and_history,
        rules_profile,
        snapshot,
        context: context.clone(),
        runtime_clock,
        root_moves,
        allowed_moves,
        fallback,
        hard,
        soft,
        admission,
        cleanup,
        descriptor_input,
        allocation_start,
        allocation_output,
        allocation_ns,
    })
}

/// Both wire versions pass this same final gate. The original current-owner
/// mutex is held until `issue` has constructed the typed live admission.
fn with_live_gate<T>(
    context: &SearchSessionContext,
    runtime_clock: &ContractSystemClock,
    hard: Deadline,
    issue: impl FnOnce() -> T,
) -> Result<T, (Option<InitialStopReason>, RoventRootAdmissionError)> {
    let accepted = context.accepts().map_err(|error| {
        let mut failure = fail(
            RoventRootAdmissionFailure::RevokedAuthority,
            "original UCI authority unavailable",
        );
        failure.metadata_error = Some(error.to_string());
        (None, failure)
    })?;
    let current_owner = Arc::clone(&context.current);
    let current = current_owner.lock().map_err(|_| {
        (
            None,
            fail(
                RoventRootAdmissionFailure::RevokedAuthority,
                "original UCI authority mutex poisoned",
            ),
        )
    })?;
    if let Some(reason) =
        source_stop_reason(context, runtime_clock, hard, *current).map_err(|error| (None, error))?
    {
        return Err((Some(reason), stopped_error(reason)));
    }
    if !accepted {
        return Err((
            None,
            fail(
                RoventRootAdmissionFailure::RevokedAuthority,
                "original UCI authority rejected without a stable control stop",
            ),
        ));
    }
    Ok(issue())
}

/// Called while holding this source's actual current-owner mutex. Scope takes
/// precedence over both original cancellation owners and then hard expiry.
fn source_stop_reason(
    context: &SearchSessionContext,
    runtime_clock: &ContractSystemClock,
    hard: Deadline,
    current: SessionScope,
) -> Result<Option<InitialStopReason>, RoventRootAdmissionError> {
    if current != SessionScope::Search(context.authority) {
        return Ok(Some(InitialStopReason::RootReplaced));
    }
    if context.cancellation.is_canceled() || context.control.cancellation.load(Ordering::Acquire) {
        return Ok(Some(InitialStopReason::Canceled));
    }
    let now = runtime_clock
        .try_now()
        .map_err(|_| fail(RoventRootAdmissionFailure::Clock, "original process now"))?;
    match hard.accepts(runtime_clock.domain(), now) {
        Ok(()) => Ok(None),
        Err(error) if error.code == ErrorCode::Expired => Ok(Some(InitialStopReason::Deadline)),
        Err(_) => Err(fail(
            RoventRootAdmissionFailure::Clock,
            "original hard deadline clock domain differs",
        )),
    }
}

fn stopped_error(reason: InitialStopReason) -> RoventRootAdmissionError {
    fail(
        RoventRootAdmissionFailure::RevokedAuthority,
        match reason {
            InitialStopReason::RootReplaced => "original UCI root scope replaced",
            InitialStopReason::Canceled => "original UCI root canceled",
            InitialStopReason::Deadline => "original hard deadline expired",
        },
    )
}

impl StoppedInitial {
    pub fn descriptor(&self) -> &RoventRootDescriptor {
        &self.descriptor
    }
    pub fn canonical_bytes(&self) -> &[u8] {
        &self.canonical
    }
    pub fn startup_lock_digest(&self) -> Digest {
        self.startup_lock
    }
    pub fn state_and_history_digest(&self) -> Digest {
        self.state_and_history
    }
    pub fn rules_profile_digest(&self) -> Digest {
        self.rules_profile
    }
    pub fn root_budget_digest(&self) -> Digest {
        self.root_budget
    }
    pub fn deadline_seal_digest(&self) -> Digest {
        self.deadline_seal
    }
    pub fn digest(&self) -> Digest {
        self.digest
    }
    pub fn allowed_root_moves(&self) -> &Arc<[BoardMove]> {
        &self.allowed_moves
    }
    pub fn fallback(&self) -> Option<BoardMove> {
        self.fallback
    }
    pub fn stop_reason(&self) -> InitialStopReason {
        self.stop_reason
    }
    pub fn allocation_clock(&self) -> &RoventRootAllocationClock {
        &self.allocation_clock
    }
    pub fn allocation_clock_digest(&self) -> Digest {
        self.allocation_clock_digest
    }
    pub fn allocation_clock_canonical_bytes(&self) -> &[u8] {
        &self.allocation_clock_canonical
    }
}

#[cfg(test)]
#[path = "source_seal/pure_tests.rs"]
mod pure_tests;
