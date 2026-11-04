//! Rovent search envelopes, independent of model roles and legacy execution purpose.
//! Rules, the manager and the physical owners attest their own facts; these
//! descriptors neither prove chess outcomes nor issue budget or completion.

use crate::pals::{Move16, SearchAuthority};
use crate::pals_adaptive::ReviewViewSeal;
use crate::{
    ClockDomain, ContractError, Deadline, Digest, ErrorCode, MonotonicTick, Stage, StateIdentity,
};
use std::num::NonZeroU16;

pub const ROVENT_SEARCH_REVISION: &str = "rovent-search/0.1";

/// This is a search recipient, not a neural P/C/V artifact role.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum SearchRecipient {
    Proposer,
    Critic,
    Researcher,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum PcCompute {
    Cpu,
    Accelerator,
}

/// Orthogonal to the existing Collection/Deployment purpose.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum ExecutionMode {
    HybridPcr,
    HybridPc,
    CpuPcr,
    CpuPc,
}

impl ExecutionMode {
    pub const fn pc_compute(self) -> PcCompute {
        match self {
            Self::HybridPcr | Self::HybridPc => PcCompute::Accelerator,
            Self::CpuPcr | Self::CpuPc => PcCompute::Cpu,
        }
    }

    pub const fn has_native_r(self) -> bool {
        matches!(self, Self::HybridPcr | Self::CpuPcr)
    }

    pub const fn supports(self, recipient: SearchRecipient) -> bool {
        !matches!(recipient, SearchRecipient::Researcher) || self.has_native_r()
    }

    pub fn validate_assembly(
        self,
        actual_pc: PcCompute,
        native_r: bool,
    ) -> Result<(), ContractError> {
        if self.pc_compute() != actual_pc || self.has_native_r() != native_r {
            return Err(invalid(
                "mode does not match actual P/C device and native R assembly",
            ));
        }
        Ok(())
    }
}

/// Actual legal root-relative chess transitions; zero is valid for an anchor.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct ChessPly(pub u16);

impl ChessPly {
    pub fn advance(self, count: u16) -> Result<Self, ContractError> {
        self.0
            .checked_add(count)
            .map(Self)
            .ok_or_else(|| invalid("root-relative chess ply overflow"))
    }
}

/// Search objectives are separate from nodes, clocks, physical work and output ownership.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct SearchGoal {
    pub depth_plies: Option<NonZeroU16>,
    pub mate_moves: Option<NonZeroU16>,
}

impl SearchGoal {
    pub const fn has_goal(self) -> bool {
        self.depth_plies.is_some() || self.mate_moves.is_some()
    }

    pub fn validate(self, registered_max_plies: NonZeroU16) -> Result<(), ContractError> {
        if self
            .depth_plies
            .is_some_and(|depth| depth > registered_max_plies)
        {
            return Err(invalid("depth exceeds the registered chess-ply capability"));
        }
        if let Some(moves) = self.mate_moves {
            let loss = moves
                .get()
                .checked_mul(2)
                .ok_or_else(|| invalid("mate move-to-ply conversion overflow"))?;
            if loss > registered_max_plies.get() {
                return Err(invalid(
                    "mate horizon exceeds the registered chess-ply capability",
                ));
            }
        }
        Ok(())
    }

    /// None means no depth admission limit, not a fabricated completed depth.
    pub fn normal_remaining(self, anchor: ChessPly) -> Option<ChessPly> {
        self.depth_plies
            .map(|depth| ChessPly(depth.get().saturating_sub(anchor.0)))
    }

    pub fn mate_horizon(self, root_wins: bool) -> Result<Option<ChessPly>, ContractError> {
        self.mate_moves
            .map(|moves| {
                let plies = moves
                    .get()
                    .checked_mul(2)
                    .ok_or_else(|| invalid("mate move-to-ply conversion overflow"))?;
                Ok(ChessPly(if root_wins { plies - 1 } else { plies }))
            })
            .transpose()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct ReciprocalQuestionId {
    pub authority: SearchAuthority,
    pub sequence: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum QuestionPurpose {
    Propose,
    Counter,
    Repair,
    TacticalReview,
    Continue,
    FreshValue,
}

/// The manager owns the actual Store pin, exact Rules and budget ledger.
/// A line digest is a descriptor, never a replacement for a Store capability.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct QuestionEnvelope {
    pub id: ReciprocalQuestionId,
    pub parent: Option<ReciprocalQuestionId>,
    pub sender: SearchRecipient,
    pub recipient: SearchRecipient,
    pub purpose: QuestionPurpose,
    pub anchor: StateIdentity,
    pub anchor_root_ply: ChessPly,
    pub original_first_move: Option<Move16>,
    pub prefix_digest: Digest,
    pub line_digest: Option<Digest>,
    pub line_offset: ChessPly,
    pub question_digest: Digest,
    pub view: ReviewViewSeal,
    pub evidence_revision: u64,
    /// Identity of the original root ledger; no fresh credits are issued here.
    pub root_budget: Digest,
    pub original_deadline: Deadline,
}

impl QuestionEnvelope {
    pub fn validate(
        &self,
        mode: ExecutionMode,
        clock: ClockDomain,
        now: MonotonicTick,
        token_budget: u16,
    ) -> Result<(), ContractError> {
        if self.id.sequence == 0 || self.sender == self.recipient {
            return Err(invalid(
                "question requires a positive id and distinct recipients",
            ));
        }
        if !mode.supports(self.sender) || !mode.supports(self.recipient) {
            return Err(ContractError::new(
                ErrorCode::UnsupportedContract,
                Stage::Admission,
                "R-free mode cannot admit an R question",
            ));
        }
        if self.original_deadline.clock.0 != self.id.authority.epoch {
            return Err(identity("question deadline and authority epochs differ"));
        }
        if let Some(parent) = self.parent {
            if parent.authority != self.id.authority
                || parent.sequence == 0
                || parent.sequence >= self.id.sequence
            {
                return Err(identity(
                    "question parent is not an earlier question in this root",
                ));
            }
        }
        if self.line_digest.is_none() && self.line_offset.0 != 0 {
            return Err(invalid("line offset requires an explicit line descriptor"));
        }
        if let Some(movement) = self.original_first_move {
            movement.decode()?;
        }
        self.view.validate(self.id.authority, token_budget)?;
        self.original_deadline.accepts(clock, now)
    }
}

fn invalid(detail: &'static str) -> ContractError {
    ContractError::new(ErrorCode::InvalidInput, Stage::Admission, detail)
}

fn identity(detail: &'static str) -> ContractError {
    ContractError::new(ErrorCode::IdentityMismatch, Stage::Admission, detail)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mode_preserves_device_and_r_independently() {
        for mode in [
            ExecutionMode::HybridPcr,
            ExecutionMode::HybridPc,
            ExecutionMode::CpuPcr,
            ExecutionMode::CpuPc,
        ] {
            assert!(mode
                .validate_assembly(mode.pc_compute(), mode.has_native_r())
                .is_ok());
            assert!(mode
                .validate_assembly(mode.pc_compute(), !mode.has_native_r())
                .is_err());
            assert_eq!(
                mode.supports(SearchRecipient::Researcher),
                mode.has_native_r()
            );
            assert!(mode.supports(SearchRecipient::Proposer));
            assert!(mode.supports(SearchRecipient::Critic));
        }
    }

    #[test]
    fn goal_uses_chess_ply_without_conflating_combined_limits() {
        let goal = SearchGoal {
            depth_plies: NonZeroU16::new(3),
            mate_moves: NonZeroU16::new(2),
        };
        assert!(goal.validate(NonZeroU16::new(4).unwrap()).is_ok());
        assert_eq!(goal.normal_remaining(ChessPly(2)), Some(ChessPly(1)));
        assert_eq!(goal.normal_remaining(ChessPly(5)), Some(ChessPly(0)));
        assert_eq!(goal.mate_horizon(true).unwrap(), Some(ChessPly(3)));
        assert_eq!(goal.mate_horizon(false).unwrap(), Some(ChessPly(4)));
        assert_eq!(SearchGoal::default().normal_remaining(ChessPly(2)), None);
        assert!(goal.validate(NonZeroU16::new(3).unwrap()).is_err());
        assert!(SearchGoal {
            mate_moves: NonZeroU16::new(u16::MAX),
            ..SearchGoal::default()
        }
        .mate_horizon(false)
        .is_err());
        assert!(ChessPly(u16::MAX).advance(1).is_err());
    }

    fn question() -> QuestionEnvelope {
        use crate::{GameGeneration, OwnerId, ProcessEpoch, RootGeneration, StateRevision};
        let authority = SearchAuthority {
            epoch: ProcessEpoch(1),
            game: GameGeneration(1),
            root: RootGeneration(1),
            implementation: Digest([1; 32]),
        };
        let anchor = StateIdentity {
            owner: OwnerId(1),
            revision: StateRevision(0),
            semantic: Digest([2; 32]),
        };
        QuestionEnvelope {
            id: ReciprocalQuestionId {
                authority,
                sequence: 1,
            },
            parent: None,
            sender: SearchRecipient::Proposer,
            recipient: SearchRecipient::Critic,
            purpose: QuestionPurpose::Counter,
            anchor,
            anchor_root_ply: ChessPly(0),
            original_first_move: None,
            prefix_digest: Digest([3; 32]),
            line_digest: None,
            line_offset: ChessPly(0),
            question_digest: Digest([4; 32]),
            view: ReviewViewSeal {
                authority,
                view_sequence: 1,
                context_policy: Digest([5; 32]),
                view_digest: Digest([6; 32]),
                root_state: anchor,
                actual_tokens: 1,
                selected_records: 0,
                anchors: 1,
                length_bucket: 8,
            },
            evidence_revision: 0,
            root_budget: Digest([7; 32]),
            original_deadline: Deadline {
                clock: ClockDomain(authority.epoch),
                at: MonotonicTick(10),
            },
        }
    }

    #[test]
    fn six_directions_keep_original_root_view_clock_and_mode() {
        let mut q = question();
        for sender in [
            SearchRecipient::Proposer,
            SearchRecipient::Critic,
            SearchRecipient::Researcher,
        ] {
            for recipient in [
                SearchRecipient::Proposer,
                SearchRecipient::Critic,
                SearchRecipient::Researcher,
            ] {
                q.sender = sender;
                q.recipient = recipient;
                let expected = sender != recipient;
                assert_eq!(
                    q.validate(
                        ExecutionMode::CpuPcr,
                        q.original_deadline.clock,
                        MonotonicTick(1),
                        512
                    )
                    .is_ok(),
                    expected
                );
                let pc_expected = expected
                    && sender != SearchRecipient::Researcher
                    && recipient != SearchRecipient::Researcher;
                assert_eq!(
                    q.validate(
                        ExecutionMode::CpuPc,
                        q.original_deadline.clock,
                        MonotonicTick(1),
                        512
                    )
                    .is_ok(),
                    pc_expected
                );
            }
        }
        q = question();
        q.parent = Some(q.id);
        assert!(q
            .validate(
                ExecutionMode::CpuPcr,
                q.original_deadline.clock,
                MonotonicTick(1),
                512
            )
            .is_err());
        q.parent = None;
        assert!(q
            .validate(
                ExecutionMode::CpuPcr,
                q.original_deadline.clock,
                MonotonicTick(10),
                512
            )
            .is_err());
        q.line_offset = ChessPly(1);
        assert!(q
            .validate(
                ExecutionMode::CpuPcr,
                q.original_deadline.clock,
                MonotonicTick(1),
                512
            )
            .is_err());
    }
}
