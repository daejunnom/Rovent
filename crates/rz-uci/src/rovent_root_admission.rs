//! Per-go Rovent root admission below an already admitted startup process.
//!
//! DynamicGo inherits the original source/assets/resources/profile declaration;
//! it never relabels another root as the startup's exact registered workload.
//! The original UCI owner supplies both its ProcessClock clone and private
//! immutable Start custody. Metadata or a caller-created clock with the same
//! epoch cannot replace those original instances. No new origin or deadline is
//! created here, and no raw Instant is serialized.

use crate::{engine::ProcessClock, parser::GoLimits, search_driver::SearchSessionContext};
use rz_contracts::{Deadline, Digest, pals::SearchAuthority, rovent::SearchGoal};
pub use rz_experiments::{
    ROVENT_ROOT_ADMISSION_DOMAIN, ROVENT_ROOT_BUDGET_DOMAIN, ROVENT_ROOT_DEADLINE_DOMAIN,
    RoventOriginalGoCoverage, RoventRootAdmissionPolicy,
};
use rz_experiments::{
    RoventInputLock, RoventParsedGo, RoventRootAllocationClock, RoventRootAuthority,
    RoventRootDeadline, RoventRootDeclaredCeilings, RoventRootDescriptor,
};
use rz_position::{BoardMove, Position, PositionSnapshot};
use rz_runtime::contracts::ContractSystemClock;
use std::{fmt, sync::Arc};

pub mod preadmission;
pub mod source_seal;

const MAX_ROOT_MOVES: usize = 218;
const MAX_CANONICAL_BYTES: usize = 64 << 10;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RoventRootAdmissionFailure {
    InvalidLock,
    RevokedAuthority,
    OriginalGoMissing,
    OriginalGoMismatch,
    Clock,
    DeadlineOrder,
    DeclaredCeiling,
    Goal,
    RootMoves,
    RulesProfile,
    RegisteredWorkload,
    CanonicalEncoding,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RoventRootAdmissionError {
    pub kind: RoventRootAdmissionFailure,
    pub detail: &'static str,
    pub metadata_error: Option<String>,
}
impl fmt::Display for RoventRootAdmissionError {
    fn fmt(&self, out: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            out,
            "Rovent root admission {:?}: {}",
            self.kind, self.detail
        )?;
        if let Some(source) = &self.metadata_error {
            write!(out, ": {source}")?;
        }
        Ok(())
    }
}
impl std::error::Error for RoventRootAdmissionError {}
fn fail(kind: RoventRootAdmissionFailure, detail: &'static str) -> RoventRootAdmissionError {
    RoventRootAdmissionError {
        kind,
        detail,
        metadata_error: None,
    }
}
fn require(
    valid: bool,
    kind: RoventRootAdmissionFailure,
    detail: &'static str,
) -> Result<(), RoventRootAdmissionError> {
    if valid {
        Ok(())
    } else {
        Err(fail(kind, detail))
    }
}

/// No public fields or deserializer can mint root authority from a digest.
/// Original Rules history, shared revocation and runtime clock owners remain
/// retained separately from the canonical descriptor bytes.
#[derive(Clone, Debug)]
pub struct RoventRootAdmission {
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
    descriptor: RoventRootDescriptor,
    root_budget: Digest,
    deadline_seal: Digest,
    digest: Digest,
    canonical: Vec<u8>,
    allocation_clock: RoventRootAllocationClock,
    allocation_clock_digest: Digest,
    allocation_clock_canonical: Vec<u8>,
}
impl RoventRootAdmission {
    pub fn admit(
        lock: &RoventInputLock,
        policy: RoventRootAdmissionPolicy,
        position: &Position,
        context: &SearchSessionContext,
        clock: &ProcessClock,
    ) -> Result<Self, RoventRootAdmissionError> {
        source_seal::SourceSeal::prepare(lock, policy, position, context, clock)?
            .try_into_live()
            .map_err(|(_, error)| error)
    }
    pub fn policy(&self) -> RoventRootAdmissionPolicy {
        self.policy
    }
    pub fn original_go_coverage(&self) -> RoventOriginalGoCoverage {
        RoventOriginalGoCoverage::FullOriginalGo
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
    pub fn root_snapshot(&self) -> &PositionSnapshot {
        &self.snapshot
    }
    pub fn authority(&self) -> SearchAuthority {
        self.context.authority
    }
    pub fn original_context(&self) -> &SearchSessionContext {
        &self.context
    }
    pub fn original_go(&self) -> &GoLimits {
        self.context
            .original_go
            .as_deref()
            .expect("admitted original go")
    }
    pub fn requested_nodes(&self) -> Option<u64> {
        self.context.requested_nodes
    }
    pub fn goal(&self) -> SearchGoal {
        self.context.goal
    }
    pub fn root_moves(&self) -> Option<&Arc<[BoardMove]>> {
        self.root_moves.as_ref()
    }
    pub fn allowed_root_moves(&self) -> &Arc<[BoardMove]> {
        &self.allowed_moves
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
    pub fn declared_ceilings(&self) -> &RoventRootDeclaredCeilings {
        &self.descriptor.input.declared_ceilings
    }
    pub fn descriptor(&self) -> &RoventRootDescriptor {
        &self.descriptor
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
    pub fn canonical_bytes(&self) -> &[u8] {
        &self.canonical
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

fn canonical_subset(
    legal: &[BoardMove],
    requested: Option<&[BoardMove]>,
) -> Result<Option<Arc<[BoardMove]>>, RoventRootAdmissionError> {
    let Some(requested) = requested else {
        return Ok(None);
    };
    require(
        !requested.is_empty()
            && requested.len() <= MAX_ROOT_MOVES
            && requested.iter().enumerate().all(|(at, movement)| {
                legal.contains(movement) && !requested[..at].contains(movement)
            }),
        RoventRootAdmissionFailure::RootMoves,
        "root restriction is empty, duplicated, oversized or not Rules legal",
    )?;
    Ok(Some(
        legal
            .iter()
            .copied()
            .filter(|movement| requested.contains(movement))
            .collect::<Vec<_>>()
            .into(),
    ))
}
fn canonical_original_subset(
    legal: &[BoardMove],
    raw: Option<&[String]>,
) -> Result<Option<Arc<[BoardMove]>>, RoventRootAdmissionError> {
    let Some(raw) = raw else {
        return Ok(None);
    };
    // Match Session::seal_root_moves: raw order/repetition is retained in the
    // wire, while only legal distinct moves form the resolved Rules subset.
    let mut distinct = Vec::new();
    for text in raw {
        let movement = BoardMove::from_uci(text).map_err(|_| {
            fail(
                RoventRootAdmissionFailure::OriginalGoMismatch,
                "original go root move decoding",
            )
        })?;
        require(
            legal.contains(&movement),
            RoventRootAdmissionFailure::RootMoves,
            "original go includes a Rules-illegal root move",
        )?;
        if !distinct.contains(&movement) {
            require(
                distinct.len() < MAX_ROOT_MOVES,
                RoventRootAdmissionFailure::RootMoves,
                "original go exceeds the distinct Rules root move bound",
            )?;
            distinct.push(movement);
        }
    }
    canonical_subset(legal, Some(&distinct))
}
fn move_texts(moves: &[BoardMove]) -> Vec<String> {
    moves.iter().map(ToString::to_string).collect()
}
fn original_go_descriptor(go: &GoLimits) -> RoventParsedGo {
    RoventParsedGo {
        searchmoves: go.searchmoves.clone(),
        depth_plies: go.depth_plies,
        mate_moves: go.mate_moves,
        movetime_ms: go.movetime_ms,
        white_time_ms: go.white_time_ms,
        black_time_ms: go.black_time_ms,
        white_increment_ms: go.white_increment_ms,
        black_increment_ms: go.black_increment_ms,
        moves_to_go: go.moves_to_go,
        nodes: go.nodes,
        infinite: go.infinite,
        ponder: go.ponder,
    }
}
fn authority_descriptor(authority: SearchAuthority) -> RoventRootAuthority {
    RoventRootAuthority {
        epoch: authority.epoch.0,
        game: authority.game.0,
        root: authority.root.0,
        implementation_sha256: hex(authority.implementation),
    }
}
fn deadline_descriptor(deadline: Deadline) -> RoventRootDeadline {
    RoventRootDeadline {
        clock_epoch: deadline.clock.0.0,
        at_ns: deadline.at.0,
    }
}
fn codec_error(error: rz_experiments::ManifestError) -> RoventRootAdmissionError {
    let mut failure = fail(
        RoventRootAdmissionFailure::CanonicalEncoding,
        "pure root metadata descriptor validation",
    );
    failure.metadata_error = Some(error.to_string());
    failure
}
fn parse_digest(value: &str) -> Result<Digest, RoventRootAdmissionError> {
    rz_eval::asset::parse_sha256(value)
        .map(Digest)
        .map_err(|_| {
            fail(
                RoventRootAdmissionFailure::InvalidLock,
                "validated startup lock digest",
            )
        })
}
fn hex(digest: Digest) -> String {
    digest.0.iter().map(|byte| format!("{byte:02x}")).collect()
}

#[cfg(test)]
#[path = "rovent_root_admission/tests.rs"]
mod tests;
