//! Root-relative selective horizon accounting, shared by P-C and P-C-R.
//!
//! The manager declares its actual candidate/counter/Repair obligations before
//! sealing a horizon. This is not full-width search: neither intermediate
//! 1..D barriers nor an extra pass for time-only `go` are introduced here.
//! Generated line length and an empty worker queue never complete an obligation.

use super::value::{ModelValueIdentity, ModelValueOutput};
use rz_contracts::Digest;
use rz_contracts::rovent::{ChessPly, SearchGoal};
use rz_position::{BoardMove, PlayStatus, Position, PositionIdentity, PositionSnapshot};
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_HORIZON_OWNER: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HorizonError {
    Disabled,
    Capacity,
    ForeignRoot,
    IllegalTransition,
    TerminalTransition,
    PlyOverflow,
    Sealed,
    InvalidObligation,
    AlreadyFinished,
    WrongBoundary,
    InvalidEvaluation,
}

/// An anchor is created at the exact root or by a checked Rules transition.
/// Context offsets and caller-declared generated lengths cannot construct it.
#[derive(Clone, Debug)]
pub struct RootPlyAnchor {
    owner: u64,
    root: PositionIdentity,
    snapshot: PositionSnapshot,
    ply: ChessPly,
}

impl RootPlyAnchor {
    pub fn snapshot(&self) -> &PositionSnapshot {
        &self.snapshot
    }

    pub fn ply(&self) -> ChessPly {
        self.ply
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct HorizonObligationId {
    owner: u64,
    slot: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HorizonObligationKind {
    Candidate,
    Counter,
    Repair,
}

/// Evidence that this exact boundary was evaluated, rather than merely
/// generated. Its constructors validate Rules or an actual model value output.
/// CPU owners may use the Fresh boundary path after their selected local work;
/// a raw CPU score/depth alone is deliberately not a boundary-state identity.
#[derive(Clone, Debug)]
pub struct HorizonBoundary {
    snapshot: PositionSnapshot,
    terminal: bool,
}

impl HorizonBoundary {
    pub fn rules_terminal(position: &Position) -> Result<Self, HorizonError> {
        let legal = position.ordered_legal_moves();
        if !matches!(
            position.play_status_from_view(&legal),
            Ok(PlayStatus::Terminal { .. })
        ) {
            return Err(HorizonError::InvalidEvaluation);
        }
        Ok(Self {
            snapshot: position.snapshot(),
            terminal: true,
        })
    }

    pub fn fresh_value(
        position: &Position,
        value: &ModelValueOutput,
        expected: &ModelValueIdentity,
    ) -> Result<Self, HorizonError> {
        value
            .validate(position, expected)
            .map_err(|_| HorizonError::InvalidEvaluation)?;
        Ok(Self {
            snapshot: position.snapshot(),
            terminal: false,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HorizonInterruption {
    Stop,
    Deadline,
    WorkBudget,
    RoleBudget,
    Capacity,
    EvidenceReclaimed,
    Failed,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HorizonSnapshot {
    pub nominal_target: Option<ChessPly>,
    /// Exact terminals may satisfy the target without inventing future plies.
    pub target_satisfied: bool,
    pub completed_depth: Option<ChessPly>,
    pub maximum_examined_ply: ChessPly,
    pub declared_obligations: usize,
    pub completed_obligations: usize,
    pub policy_sealed: bool,
    pub interruption: Option<HorizonInterruption>,
    pub selective_policy: Digest,
}

#[derive(Debug)]
struct Obligation {
    anchor: RootPlyAnchor,
    kind: HorizonObligationKind,
    completion: Option<RootPlyAnchor>,
}

/// One root and one nominal horizon; reopening a new root issues a new owner.
/// This owns only small exact Rules handles, not the Store's archive leases or
/// GPU buffers. The engine keeps those existing owners alive independently.
#[derive(Debug)]
pub struct SelectiveHorizon {
    owner: u64,
    root: PositionSnapshot,
    target: Option<ChessPly>,
    policy: Digest,
    limit: usize,
    obligations: Vec<Obligation>,
    sealed: bool,
    examined: ChessPly,
    interruption: Option<HorizonInterruption>,
}

impl SelectiveHorizon {
    pub fn new(
        root: &Position,
        goal: SearchGoal,
        selective_policy: Digest,
        max_obligations: usize,
    ) -> Result<Self, HorizonError> {
        if max_obligations == 0 {
            return Err(HorizonError::Capacity);
        }
        let owner = NEXT_HORIZON_OWNER
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
                value.checked_add(1)
            })
            .map_err(|_| HorizonError::Capacity)?;
        Ok(Self {
            owner,
            root: root.snapshot(),
            target: goal.depth_plies.map(|depth| ChessPly(depth.get())),
            policy: selective_policy,
            limit: max_obligations,
            obligations: Vec::new(),
            sealed: false,
            examined: ChessPly(0),
            interruption: None,
        })
    }

    pub fn root_anchor(&self) -> RootPlyAnchor {
        RootPlyAnchor {
            owner: self.owner,
            root: self.root.position_identity(),
            snapshot: self.root.clone(),
            ply: ChessPly(0),
        }
    }

    fn check_anchor(&self, anchor: &RootPlyAnchor) -> Result<(), HorizonError> {
        if anchor.owner != self.owner || anchor.root != self.root.position_identity() {
            return Err(HorizonError::ForeignRoot);
        }
        Ok(())
    }

    /// Call at the existing Rules transition, before moving a generated prefix
    /// into the Store. The returned child includes the complete known history.
    pub fn advance(
        &mut self,
        anchor: &RootPlyAnchor,
        position: &Position,
        movement: BoardMove,
    ) -> Result<RootPlyAnchor, HorizonError> {
        self.check_anchor(anchor)?;
        if !anchor.snapshot.same_state(&position.snapshot()) {
            return Err(HorizonError::ForeignRoot);
        }
        let legal = position.ordered_legal_moves();
        if !matches!(
            position.play_status_from_view(&legal),
            Ok(PlayStatus::Ongoing)
        ) {
            return Err(HorizonError::TerminalTransition);
        }
        let ply = anchor
            .ply
            .advance(1)
            .map_err(|_| HorizonError::PlyOverflow)?;
        let mut child = Position::from_snapshot(&anchor.snapshot, Default::default())
            .map_err(|_| HorizonError::IllegalTransition)?;
        // A fresh mutation owner requires its own exact legal view.
        child
            .make_move(movement)
            .map_err(|_| HorizonError::IllegalTransition)?;
        self.examined = self.examined.max(ply);
        Ok(RootPlyAnchor {
            owner: self.owner,
            root: anchor.root.clone(),
            snapshot: child.snapshot(),
            ply,
        })
    }

    /// None preserves the old time-only admission path. Zero is a real reached
    /// horizon, not permission to generate another normal chunk.
    pub fn remaining(&self, anchor: &RootPlyAnchor) -> Result<Option<ChessPly>, HorizonError> {
        self.check_anchor(anchor)?;
        Ok(self
            .target
            .map(|target| ChessPly(target.0.saturating_sub(anchor.ply.0))))
    }

    pub fn chunk_limit(
        &self,
        anchor: &RootPlyAnchor,
        configured_chunk: u16,
    ) -> Result<u16, HorizonError> {
        Ok(self
            .remaining(anchor)?
            .map_or(configured_chunk, |remaining| {
                configured_chunk.min(remaining.0)
            }))
    }

    /// Declare at actual candidate/counter/Repair frame admission. Different
    /// responsibilities at the same state remain separate required work.
    pub fn declare(
        &mut self,
        anchor: &RootPlyAnchor,
        kind: HorizonObligationKind,
    ) -> Result<HorizonObligationId, HorizonError> {
        self.check_anchor(anchor)?;
        if self.target.is_none() {
            return Err(HorizonError::Disabled);
        }
        if self.sealed || self.interruption.is_some() {
            return Err(HorizonError::Sealed);
        }
        if self.obligations.len() >= self.limit {
            return Err(HorizonError::Capacity);
        }
        self.obligations
            .try_reserve(1)
            .map_err(|_| HorizonError::Capacity)?;
        let id = HorizonObligationId {
            owner: self.owner,
            slot: self.obligations.len(),
        };
        self.obligations.push(Obligation {
            anchor: anchor.clone(),
            kind,
            completion: None,
        });
        Ok(id)
    }

    /// The manager seals only after its selective policy has declared all
    /// mandatory work. A temporarily empty queue does not call this method.
    pub fn seal_policy(&mut self) -> Result<(), HorizonError> {
        if self.target.is_none() {
            return Err(HorizonError::Disabled);
        }
        if self.sealed || self.interruption.is_some() {
            return Err(HorizonError::Sealed);
        }
        self.sealed = true;
        Ok(())
    }

    /// Finish at accepted endpoint evaluation, not at line publication. A
    /// Rules terminal can close early without fabricating the missing plies.
    pub fn finish(
        &mut self,
        id: HorizonObligationId,
        endpoint: &RootPlyAnchor,
        boundary: &HorizonBoundary,
    ) -> Result<(), HorizonError> {
        self.check_anchor(endpoint)?;
        if id.owner != self.owner {
            return Err(HorizonError::InvalidObligation);
        }
        let task = self
            .obligations
            .get_mut(id.slot)
            .ok_or(HorizonError::InvalidObligation)?;
        if task.completion.is_some() {
            return Err(HorizonError::AlreadyFinished);
        }
        if endpoint.ply < task.anchor.ply || !endpoint.snapshot.same_state(&boundary.snapshot) {
            return Err(HorizonError::WrongBoundary);
        }
        // An exact endpoint must actually descend from this declared anchor;
        // equal ply and an unrelated root branch cannot discharge its work.
        let ancestry = endpoint
            .snapshot
            .uci_replay(rz_position::MAX_UCI_REPLAY_PLIES)
            .map_err(|_| HorizonError::WrongBoundary)?;
        let prefix = task
            .anchor
            .snapshot
            .uci_replay(rz_position::MAX_UCI_REPLAY_PLIES)
            .map_err(|_| HorizonError::WrongBoundary)?;
        if ancestry.origin != prefix.origin
            || ancestry.completeness != prefix.completeness
            || ancestry.start_fen != prefix.start_fen
            || !ancestry.moves.starts_with(&prefix.moves)
            || ancestry.moves.len().checked_sub(prefix.moves.len())
                != Some(usize::from(endpoint.ply.0 - task.anchor.ply.0))
            || (!boundary.terminal && self.target.is_none_or(|target| endpoint.ply < target))
        {
            return Err(HorizonError::WrongBoundary);
        }
        task.completion = Some(endpoint.clone());
        Ok(())
    }

    /// Evidence eviction must revoke its closure, even after a previous
    /// snapshot reported completion. The new task must be declared/reissued by
    /// the owning manager; cancellation is never a successful completion.
    pub fn revoke(&mut self, id: HorizonObligationId) -> Result<(), HorizonError> {
        if id.owner != self.owner {
            return Err(HorizonError::InvalidObligation);
        }
        let task = self
            .obligations
            .get_mut(id.slot)
            .ok_or(HorizonError::InvalidObligation)?;
        task.completion = None;
        self.interruption = Some(HorizonInterruption::EvidenceReclaimed);
        Ok(())
    }

    pub fn interrupt(&mut self, reason: HorizonInterruption) {
        self.interruption = Some(reason);
    }

    pub fn obligation_kind(&self, id: HorizonObligationId) -> Option<HorizonObligationKind> {
        (id.owner == self.owner)
            .then(|| self.obligations.get(id.slot).map(|task| task.kind))
            .flatten()
    }

    pub fn snapshot(&self) -> HorizonSnapshot {
        let completed = self
            .obligations
            .iter()
            .filter(|task| task.completion.is_some())
            .count();
        let achieved = self.sealed
            && self.interruption.is_none()
            && !self.obligations.is_empty()
            && completed == self.obligations.len();
        HorizonSnapshot {
            nominal_target: self.target,
            target_satisfied: achieved,
            completed_depth: achieved.then(|| {
                let reached = self
                    .obligations
                    .iter()
                    .filter_map(|task| task.completion.as_ref().map(|endpoint| endpoint.ply))
                    .max()
                    .unwrap_or(ChessPly(0));
                self.target.map_or(reached, |target| reached.min(target))
            }),
            maximum_examined_ply: self.examined,
            declared_obligations: self.obligations.len(),
            completed_obligations: completed,
            policy_sealed: self.sealed,
            interruption: self.interruption,
            selective_policy: self.policy,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::num::NonZeroU16;

    fn horizon(position: &Position, depth: u16) -> SelectiveHorizon {
        SelectiveHorizon::new(
            position,
            SearchGoal {
                depth_plies: NonZeroU16::new(depth),
                mate_moves: None,
            },
            Digest([3; 32]),
            16,
        )
        .unwrap()
    }

    fn fresh(position: &Position) -> HorizonBoundary {
        let identity = ModelValueIdentity {
            semantics: "actual-boundary-only".into(),
            model: "deterministic-mock".into(),
            encoding: "test/1".into(),
            precision: "FP32".into(),
            model_epoch: [4; 32],
        };
        let value = ModelValueOutput {
            identity: identity.clone(),
            input_sha256: [5; 32],
            state: position.position_identity(),
            perspective: position.side_to_move(),
            wdl: [0.2, 0.3, 0.5],
        };
        HorizonBoundary::fresh_value(position, &value, &identity).unwrap()
    }

    #[test]
    fn generated_endpoint_requires_actual_evaluation_and_all_declared_work() {
        let root = Position::startpos();
        let mut h = horizon(&root, 2);
        let anchor = h.root_anchor();
        let candidate = h
            .declare(&anchor, HorizonObligationKind::Candidate)
            .unwrap();
        let counter = h.declare(&anchor, HorizonObligationKind::Counter).unwrap();
        let mut played = Position::from_snapshot(&root.snapshot(), Default::default()).unwrap();
        let a = h
            .advance(&anchor, &played, "e2e4".parse().unwrap())
            .unwrap();
        played.make_uci("e2e4").unwrap();
        let b = h.advance(&a, &played, "e7e5".parse().unwrap()).unwrap();
        played.make_uci("e7e5").unwrap();
        assert_eq!(h.chunk_limit(&b, 4).unwrap(), 0);
        h.seal_policy().unwrap();
        assert_eq!(h.snapshot().completed_depth, None);
        assert_eq!(h.snapshot().maximum_examined_ply, ChessPly(2));
        h.finish(candidate, &b, &fresh(&played)).unwrap();
        assert_eq!(h.snapshot().completed_depth, None);
        assert_eq!(
            h.finish(candidate, &b, &fresh(&played)),
            Err(HorizonError::AlreadyFinished)
        );
        h.finish(counter, &b, &fresh(&played)).unwrap();
        assert_eq!(h.snapshot().completed_depth, Some(ChessPly(2)));
        h.revoke(counter).unwrap();
        assert_eq!(h.snapshot().completed_depth, None);
    }

    #[test]
    fn time_only_does_not_create_depth_pass_and_empty_queue_does_not_finish() {
        let root = Position::startpos();
        let mut timed = horizon(&root, 0);
        let anchor = timed.root_anchor();
        assert_eq!(timed.remaining(&anchor).unwrap(), None);
        assert_eq!(timed.chunk_limit(&anchor, 4).unwrap(), 4);
        assert_eq!(
            timed.declare(&anchor, HorizonObligationKind::Candidate),
            Err(HorizonError::Disabled)
        );
        let mut fixed = horizon(&root, 8);
        fixed.seal_policy().unwrap();
        assert_eq!(fixed.snapshot().completed_depth, None);
    }

    #[test]
    fn wrong_branch_and_cancellation_cannot_complete_depth() {
        let root = Position::startpos();
        let mut h = horizon(&root, 2);
        let anchor = h.root_anchor();
        let a = h.advance(&anchor, &root, "e2e4".parse().unwrap()).unwrap();
        let b = h.advance(&anchor, &root, "d2d4".parse().unwrap()).unwrap();
        let task = h.declare(&a, HorizonObligationKind::Repair).unwrap();
        let mut d = Position::from_snapshot(b.snapshot(), Default::default()).unwrap();
        let child = h.advance(&b, &d, "d7d5".parse().unwrap()).unwrap();
        d.make_uci("d7d5").unwrap();
        assert_eq!(
            h.finish(task, &child, &fresh(&d)),
            Err(HorizonError::WrongBoundary)
        );
        h.seal_policy().unwrap();
        h.interrupt(HorizonInterruption::Stop);
        assert_eq!(h.snapshot().completed_depth, None);
        let mut other = horizon(&root, 2);
        assert_eq!(
            other.declare(&anchor, HorizonObligationKind::Repair),
            Err(HorizonError::ForeignRoot)
        );
    }

    #[test]
    fn rules_terminal_closes_early_without_advancing_seldepth_to_target() {
        let root = Position::from_fen("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1").unwrap();
        let mut h = horizon(&root, 10);
        let anchor = h.root_anchor();
        let task = h
            .declare(&anchor, HorizonObligationKind::Candidate)
            .unwrap();
        let child = h.advance(&anchor, &root, "f7g7".parse().unwrap()).unwrap();
        let mut mate = Position::from_snapshot(&root.snapshot(), Default::default()).unwrap();
        mate.make_uci("f7g7").unwrap();
        h.finish(
            task,
            &child,
            &HorizonBoundary::rules_terminal(&mate).unwrap(),
        )
        .unwrap();
        h.seal_policy().unwrap();
        assert!(h.snapshot().target_satisfied);
        assert_eq!(h.snapshot().completed_depth, Some(ChessPly(1)));
        assert_eq!(h.snapshot().maximum_examined_ply, ChessPly(1));
    }
}
