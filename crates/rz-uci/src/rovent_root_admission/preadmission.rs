//! Pure go checks before a live root clock or search authority is issued.
//!
//! The caller supplies the original already-admitted startup lock. Verifying
//! its canonical source/config declarations here is not a second observation
//! of executable bytes, loaded model assets or applied resources. This draft
//! cannot construct RoventRootAdmission, a deadline, a worker or a live owner.

use super::{
    MAX_CANONICAL_BYTES, RoventRootAdmissionError, RoventRootAdmissionFailure,
    RoventRootAdmissionPolicy, canonical_original_subset, fail, hex, parse_digest, require,
};
use crate::parser::GoLimits;
use rz_contracts::{Digest, rovent::SearchGoal};
use rz_experiments::{RoventInputLock, RoventInputLockV2, RoventSearchDeclaration};
use rz_position::{BoardMove, Position, PositionSnapshot};
use std::{num::NonZeroU16, sync::Arc};

/// A pure observation of one position/go pair, not a live admission capability.
/// No clock, authority, root budget or deadline seal can be read from it.
#[derive(Clone, Debug)]
pub struct RoventGoPrevalidation {
    policy: RoventRootAdmissionPolicy,
    startup_lock: Digest,
    snapshot: PositionSnapshot,
    state_and_history: Digest,
    rules_profile: Digest,
    requested_nodes: Option<u64>,
    goal: SearchGoal,
    maximum_goal_plies: NonZeroU16,
    root_moves: Option<Arc<[BoardMove]>>,
    allowed_moves: Arc<[BoardMove]>,
}
impl RoventGoPrevalidation {
    pub fn policy(&self) -> RoventRootAdmissionPolicy {
        self.policy
    }
    pub fn startup_lock_digest(&self) -> Digest {
        self.startup_lock
    }
    pub fn root_snapshot(&self) -> &PositionSnapshot {
        &self.snapshot
    }
    pub fn state_and_history_digest(&self) -> Digest {
        self.state_and_history
    }
    pub fn rules_profile_digest(&self) -> Digest {
        self.rules_profile
    }
    pub fn requested_nodes(&self) -> Option<u64> {
        self.requested_nodes
    }
    pub fn goal(&self) -> SearchGoal {
        self.goal
    }
    pub fn maximum_goal_plies(&self) -> NonZeroU16 {
        self.maximum_goal_plies
    }
    pub fn root_moves(&self) -> Option<&Arc<[BoardMove]>> {
        self.root_moves.as_ref()
    }
    pub fn allowed_root_moves(&self) -> &Arc<[BoardMove]> {
        &self.allowed_moves
    }
}

/// Checks exact registered workload or explicit DynamicGo inheritance without
/// allocating a new root authority or deriving deadlines from a sampled now.
/// Some(0) nodes is preserved in this new root domain; the legacy manifest
/// validator still rejects a registered zero-node workload.
pub fn validate_go(
    position: &Position,
    go: &GoLimits,
    lock: &RoventInputLock,
    policy: RoventRootAdmissionPolicy,
    max_goal_plies: NonZeroU16,
) -> Result<RoventGoPrevalidation, RoventRootAdmissionError> {
    lock.validate().map_err(|_| {
        fail(
            RoventRootAdmissionFailure::InvalidLock,
            "original startup input lock before go",
        )
    })?;
    validate_go_for_search(
        position,
        go,
        &lock.manifest.engine.search,
        &lock.canonical_sha256,
        policy,
        max_goal_plies,
    )
}

/// Applies the same pure Rules and workload checks to the original /2 lock.
/// This does not convert the lock or create a root, worker, clock or authority.
pub fn validate_go_v2(
    position: &Position,
    go: &GoLimits,
    lock: &RoventInputLockV2,
    policy: RoventRootAdmissionPolicy,
    max_goal_plies: NonZeroU16,
) -> Result<RoventGoPrevalidation, RoventRootAdmissionError> {
    lock.validate().map_err(|_| {
        fail(
            RoventRootAdmissionFailure::InvalidLock,
            "original startup input lock before go",
        )
    })?;
    validate_go_for_search(
        position,
        go,
        &lock.manifest.engine.search,
        &lock.canonical_sha256,
        policy,
        max_goal_plies,
    )
}

fn validate_go_for_search(
    position: &Position,
    go: &GoLimits,
    registered: &RoventSearchDeclaration,
    canonical_sha256: &str,
    policy: RoventRootAdmissionPolicy,
    max_goal_plies: NonZeroU16,
) -> Result<RoventGoPrevalidation, RoventRootAdmissionError> {
    let goal = SearchGoal {
        depth_plies: go.depth_plies,
        mate_moves: go.mate_moves,
    };
    let maximum_goal_plies = registered.registered_max_chess_plies.min(max_goal_plies);
    goal.validate(maximum_goal_plies).map_err(|_| {
        fail(
            RoventRootAdmissionFailure::Goal,
            "go exceeds the startup or selected driver's chess-ply capability",
        )
    })?;
    if let Some(moves) = &go.searchmoves {
        let bytes = moves.iter().try_fold(0usize, |bytes, movement| {
            bytes.checked_add(movement.len())?.checked_add(3)
        });
        require(
            bytes.is_some_and(|bytes| bytes <= MAX_CANONICAL_BYTES / 2),
            RoventRootAdmissionFailure::CanonicalEncoding,
            "original go exceeds the bounded root descriptor budget",
        )?;
    }
    let snapshot = position.snapshot();
    let state_and_history = rz_position::contracts::rules_snapshot_semantic_digest(&snapshot);
    let rules_profile = rz_position::contracts::profile_digest();
    require(
        registered.rules_profile_sha256 == hex(rules_profile),
        RoventRootAdmissionFailure::RulesProfile,
        "registered Rules profile differs from actual Rules before go",
    )?;
    let legal = position.ordered_legal_moves();
    // This is the existing Session-compatible raw→distinct→Rules-order path.
    // Raw repetition/order remains in the caller's original GoLimits.
    let root_moves = canonical_original_subset(legal.moves(), go.searchmoves.as_deref())?;
    let allowed_moves = root_moves
        .clone()
        .unwrap_or_else(|| Arc::from(legal.moves()));
    if policy == RoventRootAdmissionPolicy::ExactRegistered {
        let declared = canonical_original_subset(legal.moves(), registered.searchmoves.as_deref())?;
        require(
            registered.root_state_and_history_sha256 == hex(state_and_history)
                && registered.requested_nodes == go.nodes
                && registered.goal() == goal
                && declared.as_deref() == root_moves.as_deref(),
            RoventRootAdmissionFailure::RegisteredWorkload,
            "position/history/go/subset differs from the exact registered workload",
        )?;
    }
    Ok(RoventGoPrevalidation {
        policy,
        startup_lock: parse_digest(canonical_sha256)?,
        snapshot,
        state_and_history,
        rules_profile,
        requested_nodes: go.nodes,
        goal,
        maximum_goal_plies,
        root_moves,
        allowed_moves,
    })
}
