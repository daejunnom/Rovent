//! Rules-only, bounded OR/AND mate evidence, independent of P/C/R executors.
//!
//! Only checked Rules terminals and checked child edges create conclusions.
//! Neither WDL, CPU/external `score mate`, nor a `RestrictedRulesLine` is an
//! input. Missing defenses remain work for the existing P/C/R question queue;
//! this reducer executes no alpha-beta, NNUE, model, external process or worker.

use super::rovent_frontier::CheckedFrontierLine;
use super::store::{LineId, PalsStores};
use rz_contracts::rovent::ChessPly;
use rz_position::{
    Availability, BoardMove, Color, PlayStatus, Position, PositionSnapshot, RepetitionIdentity,
};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_PROOF_OWNER: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MateProofError {
    Capacity,
    ForeignHandle,
    InvalidRestriction,
    IllegalMove,
    AlreadyClosed,
    HorizonExhausted,
    Rules,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct MateNodeId {
    owner: u64,
    slot: usize,
}

/// Remaining-ply goal only. `NoMateWithin` is not a draw, a negative WDL, or
/// proof that the attacker cannot mate at a larger horizon.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MateConclusion {
    Unknown,
    Draw,
    NoMateWithin,
    ForcedMate(MateDistance),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MateDistance {
    /// Existence and upper bound; a shorter unexamined attacking move may win.
    Within(ChessPly),
    /// Min at attacker and max at defender are both closed at this horizon.
    Exact(ChessPly),
}

impl MateDistance {
    pub fn plies(self) -> ChessPly {
        match self {
            Self::Within(plies) | Self::Exact(plies) => plies,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MateScope {
    FullRoot,
    /// A root `searchmoves` subset can prove the opponent's strategy only
    /// against that subset. Never promote it to an unrestricted root loss.
    RestrictedRoot,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RootMateConclusion {
    pub attacker: Color,
    pub root_side_to_move: Color,
    pub conclusion: MateConclusion,
    pub scope: MateScope,
}

/// An actual checked root strategy, issued only by this Rules coverage owner.
/// Keeping the selected edge and its exact root domain with the conclusion
/// prevents a caller from attaching this score to an unrelated model choice.
#[derive(Clone, Debug)]
pub struct RootMateStrategy {
    root: PositionSnapshot,
    allowed_root_moves: Vec<BoardMove>,
    root_restricted: bool,
    movement: BoardMove,
    child: PositionSnapshot,
    conclusion: RootMateConclusion,
}

impl RootMateStrategy {
    pub fn best_move(&self) -> BoardMove {
        self.movement
    }

    pub fn conclusion(&self) -> RootMateConclusion {
        self.conclusion
    }

    /// Whether the original root domain was restricted, even when a winning
    /// allowed move also certifies existence for the unrestricted root.
    pub fn root_restricted(&self) -> bool {
        self.root_restricted
    }

    pub fn validates_root(&self, root: &PositionSnapshot, allowed: &[BoardMove]) -> bool {
        self.root.same_state(root) && self.allowed_root_moves == allowed
    }

    pub fn checked_child_snapshot(&self) -> &PositionSnapshot {
        &self.child
    }
}

impl RootMateConclusion {
    /// Existence is enough to stop `go mate`; exact shortest distance is not
    /// required. The caller preserves `scope` in its result/receipt.
    pub fn goal_satisfied(self) -> bool {
        matches!(self.conclusion, MateConclusion::ForcedMate(_))
    }

    /// Conservative initial UCI output: only exact distance, never an upper
    /// bound presented as an exact `score mate`. An already-mated root returns
    /// None and follows the terminal bestmove-0000 path.
    pub fn exact_uci_mate(self) -> Option<i32> {
        let MateConclusion::ForcedMate(MateDistance::Exact(ChessPly(plies))) = self.conclusion
        else {
            return None;
        };
        if plies == 0 {
            return None;
        }
        let moves = i32::from(plies.div_ceil(2));
        Some(if self.attacker == self.root_side_to_move {
            moves
        } else {
            -moves
        })
    }
}

#[derive(Clone, Debug)]
struct MateNode {
    snapshot: PositionSnapshot,
    remaining: ChessPly,
    legal: Vec<BoardMove>,
    children: Vec<Option<MateNodeId>>,
    parents: Vec<usize>,
    terminal: Option<MateConclusion>,
    defender_claim: Availability,
    conclusion: MateConclusion,
}

/// Bounded sidecar over exact Rules snapshots. It can be retained alongside
/// existing Store StateId/LineId handles and their existing archive pins. One
/// owner fixes root/history, attacker, remaining horizon and root restriction;
/// its buckets merely accelerate lookup, never replace exact history equality.
#[derive(Debug)]
pub struct MateProofStore {
    owner: u64,
    attacker: Color,
    root_side: Color,
    root_restricted: bool,
    max_nodes: usize,
    nodes: Vec<MateNode>,
    index: HashMap<RepetitionIdentity, Vec<usize>>,
}

impl MateProofStore {
    /// `root_moves=None` reuses Rules' full root domain. A nonempty subset is
    /// legal only at root; every descendant defender still enumerates all its
    /// legal choices. The order is retained for the existing executor policy.
    pub fn new(
        root: &Position,
        attacker: Color,
        remaining: ChessPly,
        root_moves: Option<&[BoardMove]>,
        max_nodes: usize,
    ) -> Result<Self, MateProofError> {
        if max_nodes == 0 {
            return Err(MateProofError::Capacity);
        }
        let owner = NEXT_PROOF_OWNER
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
                value.checked_add(1)
            })
            .map_err(|_| MateProofError::Capacity)?;
        let legal = root.ordered_legal_moves();
        let selected = match root_moves {
            None => legal.moves().to_vec(),
            Some(moves) => {
                if moves.is_empty()
                    || moves.iter().enumerate().any(|(at, movement)| {
                        !legal.moves().contains(movement) || moves[..at].contains(movement)
                    })
                {
                    return Err(MateProofError::InvalidRestriction);
                }
                moves.to_vec()
            }
        };
        let root_restricted = selected.len() != legal.moves().len();
        let mut result = Self {
            owner,
            attacker,
            root_side: root.side_to_move(),
            root_restricted,
            max_nodes,
            nodes: Vec::new(),
            index: HashMap::new(),
        };
        let root_node = result.make_node(root, remaining, selected)?;
        result.push_node(root_node)?;
        Ok(result)
    }

    pub fn root(&self) -> MateNodeId {
        MateNodeId {
            owner: self.owner,
            slot: 0,
        }
    }

    fn check(&self, id: MateNodeId) -> Result<usize, MateProofError> {
        if id.owner != self.owner || id.slot >= self.nodes.len() {
            return Err(MateProofError::ForeignHandle);
        }
        Ok(id.slot)
    }

    pub fn conclusion(&self, node: MateNodeId) -> Result<MateConclusion, MateProofError> {
        Ok(self.nodes[self.check(node)?].conclusion)
    }

    pub fn snapshot(&self, node: MateNodeId) -> Result<&PositionSnapshot, MateProofError> {
        Ok(&self.nodes[self.check(node)?].snapshot)
    }

    pub fn remaining(&self, node: MateNodeId) -> Result<ChessPly, MateProofError> {
        Ok(self.nodes[self.check(node)?].remaining)
    }

    pub fn root_conclusion(&self) -> RootMateConclusion {
        let mut conclusion = self.nodes[0].conclusion;
        // An allowed winning move is also a valid unrestricted winning
        // strategy. The opponent's AND proof against a subset stays conditional.
        let conditional = self.root_restricted
            && !(self.attacker == self.root_side
                && matches!(conclusion, MateConclusion::ForcedMate(_)));
        // A restricted attacker can certify existence for the full root, but
        // an excluded move could mate sooner. Only mate in one is globally
        // minimal without covering those omitted attacking alternatives.
        if self.root_restricted && self.attacker == self.root_side {
            if let MateConclusion::ForcedMate(MateDistance::Exact(plies)) = conclusion {
                if plies.0 > 1 {
                    conclusion = MateConclusion::ForcedMate(MateDistance::Within(plies));
                }
            }
        }
        RootMateConclusion {
            attacker: self.attacker,
            root_side_to_move: self.root_side,
            conclusion,
            scope: if conditional {
                MateScope::RestrictedRoot
            } else {
                MateScope::FullRoot
            },
        }
    }

    /// Read only already closed Rules coverage. At an attacking root this is
    /// a proven winning move; at a defending root every allowed move is proved
    /// lost and the chosen move maximizes the observed forced distance. No
    /// missing branch is searched or inferred by this accessor.
    pub fn root_strategy(&self) -> Option<RootMateStrategy> {
        let conclusion = self.root_conclusion();
        if !conclusion.goal_satisfied() {
            return None;
        }
        let (movement, child) = self.strategy_child(0)?;
        Some(RootMateStrategy {
            root: self.nodes[0].snapshot.clone(),
            allowed_root_moves: self.nodes[0].legal.clone(),
            root_restricted: self.root_restricted,
            movement,
            child: self.nodes[child.slot].snapshot.clone(),
            conclusion,
        })
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Existing P/C/R ordering may select any returned move. Quiet moves and
    /// zero-prior defenses remain here until a real Rules edge is supplied.
    pub fn missing_moves(&self, node: MateNodeId) -> Result<Vec<BoardMove>, MateProofError> {
        let node = &self.nodes[self.check(node)?];
        if node.terminal.is_some() || node.remaining.0 == 0 {
            return Ok(Vec::new());
        }
        Ok(node
            .legal
            .iter()
            .zip(&node.children)
            .filter_map(|(movement, child)| child.is_none().then_some(*movement))
            .collect())
    }

    pub fn child(
        &self,
        parent: MateNodeId,
        movement: BoardMove,
    ) -> Result<Option<MateNodeId>, MateProofError> {
        let parent = &self.nodes[self.check(parent)?];
        let slot = parent
            .legal
            .iter()
            .position(|held| *held == movement)
            .ok_or(MateProofError::IllegalMove)?;
        Ok(parent.children[slot])
    }

    /// Consume an existing Store line without treating its restricted value as
    /// a proof. The entire line first passes exact-start and Rules replay; only
    /// then are its legal edges supplied to this independent coverage reducer.
    /// The manager retains the Store's existing state/line/archive pins while
    /// calling this method; this read does not mint another archive capability.
    pub fn observe_stored_line(
        &mut self,
        parent: MateNodeId,
        stores: &PalsStores,
        line: LineId,
    ) -> Result<MateNodeId, MateProofError> {
        let at = self.check(parent)?;
        let record = stores.lines.get(line).map_err(|_| MateProofError::Rules)?;
        let start = stores
            .states
            .get(record.start_state)
            .map_err(|_| MateProofError::Rules)?;
        if !start.same_state(&self.nodes[at].snapshot) {
            return Err(MateProofError::ForeignHandle);
        }
        let movements = stores
            .lines
            .moves(line)
            .map_err(|_| MateProofError::Rules)?;
        if movements.len() > usize::from(self.nodes[at].remaining.0) {
            return Err(MateProofError::HorizonExhausted);
        }
        let mut position = Position::from_snapshot(start, Default::default())
            .map_err(|_| MateProofError::Rules)?;
        for (index, movement) in movements.iter().copied().enumerate() {
            let legal = position.ordered_legal_moves();
            if !matches!(
                position.play_status_from_view(&legal),
                Ok(PlayStatus::Ongoing)
            ) {
                return Err(MateProofError::AlreadyClosed);
            }
            if index == 0 && !self.nodes[at].legal.contains(&movement) {
                return Err(MateProofError::IllegalMove);
            }
            position
                .make_from_view(&legal, movement)
                .map_err(|_| MateProofError::IllegalMove)?;
        }
        let mut endpoint = parent;
        for movement in movements {
            endpoint = self.open_child(endpoint, movement)?;
        }
        Ok(endpoint)
    }

    /// Called at an actual checked continuation or to issue a missing-defense
    /// question. This applies one Rules move, never searches the child. Existing
    /// results are reused only with exact history and remaining-ply equality.
    /// Duplicate delivery returns the same edge without charging coverage twice.
    pub fn open_child(
        &mut self,
        parent: MateNodeId,
        movement: BoardMove,
    ) -> Result<MateNodeId, MateProofError> {
        let parent_slot = self.check(parent)?;
        let node = &self.nodes[parent_slot];
        let move_slot = node
            .legal
            .iter()
            .position(|held| *held == movement)
            .ok_or(MateProofError::IllegalMove)?;
        if let Some(child) = node.children[move_slot] {
            return Ok(child);
        }
        if node.terminal.is_some() {
            return Err(MateProofError::AlreadyClosed);
        }
        if node.remaining.0 == 0 {
            return Err(MateProofError::HorizonExhausted);
        }
        let remaining = ChessPly(node.remaining.0 - 1);
        let mut position = Position::from_snapshot(&node.snapshot, Default::default())
            .map_err(|_| MateProofError::Rules)?;
        position
            .make_move(movement)
            .map_err(|_| MateProofError::IllegalMove)?;
        self.link_checked_position(parent_slot, move_slot, remaining, &position)
    }

    /// Project one already charged controller edge into passive coverage.
    /// The immutable line is issued only after that owner's actual Rules
    /// transition. A snapshot declaration, raw score or unchecked move cannot
    /// enter this path, and no additional Rules mutation is performed here.
    pub(crate) fn observe_checked_child(
        &mut self,
        parent: MateNodeId,
        line: &CheckedFrontierLine,
        step_index: usize,
    ) -> Result<MateNodeId, MateProofError> {
        let parent_slot = self.check(parent)?;
        let step = line
            .steps()
            .get(step_index)
            .ok_or(MateProofError::ForeignHandle)?;
        let parent_snapshot = if step_index == 0 {
            line.root_snapshot()
        } else {
            line.steps()[step_index - 1].snapshot()
        };
        let root_offset =
            u16::try_from(step_index).map_err(|_| MateProofError::HorizonExhausted)?;
        let node = &self.nodes[parent_slot];
        if !self.nodes[0].snapshot.same_state(line.root_snapshot())
            || !node.snapshot.same_state(parent_snapshot)
            || self.nodes[0].remaining.0.checked_sub(root_offset) != Some(node.remaining.0)
        {
            return Err(MateProofError::ForeignHandle);
        }
        let move_slot = node
            .legal
            .iter()
            .position(|held| *held == step.movement())
            .ok_or(MateProofError::IllegalMove)?;
        if let Some(child) = node.children[move_slot] {
            return if self.nodes[child.slot].snapshot.same_state(step.snapshot()) {
                Ok(child)
            } else {
                Err(MateProofError::ForeignHandle)
            };
        }
        if node.terminal.is_some() {
            return Err(MateProofError::AlreadyClosed);
        }
        if node.remaining.0 == 0 {
            return Err(MateProofError::HorizonExhausted);
        }
        let remaining = ChessPly(node.remaining.0 - 1);
        let position = Position::from_snapshot(step.snapshot(), Default::default())
            .map_err(|_| MateProofError::Rules)?;
        self.link_checked_position(parent_slot, move_slot, remaining, &position)
    }

    fn link_checked_position(
        &mut self,
        parent_slot: usize,
        move_slot: usize,
        remaining: ChessPly,
        position: &Position,
    ) -> Result<MateNodeId, MateProofError> {
        let snapshot = position.snapshot();
        let existing = self
            .index
            .get(&snapshot.repetition_identity())
            .and_then(|bucket| {
                bucket.iter().copied().find(|slot| {
                    self.nodes[*slot].remaining == remaining
                        && self.nodes[*slot].snapshot.same_state(&snapshot)
                })
            });
        let child = if let Some(slot) = existing {
            MateNodeId {
                owner: self.owner,
                slot,
            }
        } else {
            let legal = position.ordered_legal_moves().moves().to_vec();
            let child = self.make_node(position, remaining, legal)?;
            self.push_node(child)?
        };
        self.nodes[child.slot]
            .parents
            .try_reserve(1)
            .map_err(|_| MateProofError::Capacity)?;
        self.nodes[child.slot].parents.push(parent_slot);
        self.nodes[parent_slot].children[move_slot] = Some(child);
        self.propagate(parent_slot);
        Ok(child)
    }

    /// Retract unfinished/reclaimed edge evidence. Conclusions are recomputed
    /// through affected ancestors; a past complete counter never survives the
    /// loss of one of its required defense proofs. Native physical leases are
    /// still owned and drained by the existing runtime, not by this sidecar.
    pub fn retract_child(
        &mut self,
        parent: MateNodeId,
        movement: BoardMove,
    ) -> Result<(), MateProofError> {
        let parent = self.check(parent)?;
        let at = self.nodes[parent]
            .legal
            .iter()
            .position(|held| *held == movement)
            .ok_or(MateProofError::IllegalMove)?;
        if let Some(child) = self.nodes[parent].children[at].take() {
            self.nodes[child.slot]
                .parents
                .retain(|held| *held != parent);
            self.propagate(parent);
        }
        Ok(())
    }

    /// A strategy move is an actual checked edge. This is enough for early win
    /// selection; its entire printed PV must still be replayed by Rules. A
    /// failed or missing child is never fabricated to match the distance bound.
    pub fn strategy_move(&self, node: MateNodeId) -> Result<Option<BoardMove>, MateProofError> {
        Ok(self
            .strategy_child(self.check(node)?)
            .map(|(movement, _)| movement))
    }

    fn strategy_child(&self, slot: usize) -> Option<(BoardMove, MateNodeId)> {
        let node = &self.nodes[slot];
        if !matches!(node.conclusion, MateConclusion::ForcedMate(_)) {
            return None;
        }
        let attack_turn = node.snapshot.side_to_move() == self.attacker;
        let mut best: Option<(BoardMove, MateNodeId, u16)> = None;
        for (movement, child) in node.legal.iter().zip(&node.children) {
            let Some(child) = child else { continue };
            let MateConclusion::ForcedMate(distance) = self.nodes[child.slot].conclusion else {
                continue;
            };
            let distance = distance.plies().0;
            if best.is_none_or(|(_, _, held)| {
                if attack_turn {
                    distance < held
                } else {
                    distance > held
                }
            }) {
                best = Some((*movement, *child, distance));
            }
        }
        best.map(|(movement, child, _)| (movement, child))
    }

    fn make_node(
        &self,
        position: &Position,
        remaining: ChessPly,
        legal: Vec<BoardMove>,
    ) -> Result<MateNode, MateProofError> {
        let classification = position
            .classify_position()
            .map_err(|_| MateProofError::Rules)?;
        let terminal = match classification.play_status {
            PlayStatus::Ongoing => None,
            PlayStatus::Terminal { winner: None, .. } => Some(MateConclusion::Draw),
            PlayStatus::Terminal {
                winner: Some(winner),
                ..
            } if winner == self.attacker => {
                Some(MateConclusion::ForcedMate(MateDistance::Exact(ChessPly(0))))
            }
            PlayStatus::Terminal { .. } => Some(MateConclusion::NoMateWithin),
        };
        // A defender may exercise a Rules-owned current or intended claim.
        // Unknown prefix evidence cannot be assumed unavailable for a proof.
        let defender_claim = if position.side_to_move() == self.attacker {
            Availability::Unavailable
        } else if classification
            .claim_availability
            .iter()
            .any(|claim| claim.availability == Availability::Available)
        {
            Availability::Available
        } else if classification
            .claim_availability
            .iter()
            .any(|claim| claim.availability == Availability::Unknown)
        {
            Availability::Unknown
        } else {
            Availability::Unavailable
        };
        let conclusion = terminal.unwrap_or(
            if remaining.0 == 0 || defender_claim == Availability::Available {
                MateConclusion::NoMateWithin
            } else {
                MateConclusion::Unknown
            },
        );
        let mut children = Vec::new();
        children
            .try_reserve_exact(legal.len())
            .map_err(|_| MateProofError::Capacity)?;
        children.resize(legal.len(), None);
        Ok(MateNode {
            snapshot: position.snapshot(),
            remaining,
            legal,
            children,
            parents: Vec::new(),
            terminal,
            defender_claim,
            conclusion,
        })
    }

    fn push_node(&mut self, node: MateNode) -> Result<MateNodeId, MateProofError> {
        if self.nodes.len() >= self.max_nodes {
            return Err(MateProofError::Capacity);
        }
        self.nodes
            .try_reserve(1)
            .map_err(|_| MateProofError::Capacity)?;
        self.index
            .try_reserve(1)
            .map_err(|_| MateProofError::Capacity)?;
        let bucket = self
            .index
            .entry(node.snapshot.repetition_identity())
            .or_default();
        bucket
            .try_reserve(1)
            .map_err(|_| MateProofError::Capacity)?;
        let id = MateNodeId {
            owner: self.owner,
            slot: self.nodes.len(),
        };
        bucket.push(id.slot);
        self.nodes.push(node);
        Ok(id)
    }

    fn reduced(&self, slot: usize) -> MateConclusion {
        let node = &self.nodes[slot];
        if let Some(terminal) = node.terminal {
            return terminal;
        }
        if node.remaining.0 == 0 || node.defender_claim == Availability::Available {
            return MateConclusion::NoMateWithin;
        }
        let attack_turn = node.snapshot.side_to_move() == self.attacker;
        let mut forced = 0;
        let mut closed = 0;
        let mut draws = 0;
        let mut all_exact = true;
        let mut best = if attack_turn { u16::MAX } else { 0 };
        for child in &node.children {
            let outcome = child.map_or(MateConclusion::Unknown, |child| {
                self.nodes[child.slot].conclusion
            });
            match outcome {
                MateConclusion::Unknown => all_exact = false,
                MateConclusion::Draw => {
                    closed += 1;
                    draws += 1;
                }
                MateConclusion::NoMateWithin => {
                    closed += 1;
                }
                MateConclusion::ForcedMate(distance) => {
                    forced += 1;
                    closed += 1;
                    all_exact &= matches!(distance, MateDistance::Exact(_));
                    let distance = distance.plies().0;
                    best = if attack_turn {
                        best.min(distance)
                    } else {
                        best.max(distance)
                    };
                }
            }
        }
        if attack_turn && forced > 0 {
            let plies = ChessPly(best + 1); // Every child has remaining = parent-1.
            // Mate in one has no faster legal alternative. Otherwise every
            // alternative must be closed before the minimum is exact.
            return MateConclusion::ForcedMate(
                if best == 0 || (all_exact && closed == node.legal.len()) {
                    MateDistance::Exact(plies)
                } else {
                    MateDistance::Within(plies)
                },
            );
        }
        if !attack_turn {
            if draws == node.legal.len() && draws > 0 {
                return MateConclusion::Draw;
            }
            if draws > 0 || closed > forced {
                return MateConclusion::NoMateWithin;
            }
            if node.defender_claim == Availability::Unknown {
                return MateConclusion::Unknown;
            }
            if forced == node.legal.len() && forced > 0 {
                let plies = ChessPly(best + 1);
                return MateConclusion::ForcedMate(if all_exact {
                    MateDistance::Exact(plies)
                } else {
                    MateDistance::Within(plies)
                });
            }
        }
        if closed == node.legal.len() && closed > 0 {
            if draws == closed {
                MateConclusion::Draw
            } else {
                MateConclusion::NoMateWithin
            }
        } else {
            MateConclusion::Unknown
        }
    }

    fn propagate(&mut self, first: usize) {
        // Remaining ply strictly decreases along every edge, so the parent
        // graph has no cycle even when the chess board repeats. Only affected
        // ancestors are visited, never every NN result or the complete Store.
        let mut pending = vec![first];
        while let Some(slot) = pending.pop() {
            let conclusion = self.reduced(slot);
            if conclusion != self.nodes[slot].conclusion {
                self.nodes[slot].conclusion = conclusion;
                pending.extend_from_slice(&self.nodes[slot].parents);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mv(text: &str) -> BoardMove {
        text.parse().unwrap()
    }

    #[test]
    fn both_colors_mate_in_one_are_rules_proofs_without_r_or_model() {
        for (fen, winner, movement) in [
            ("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1", Color::White, "f7g7"),
            ("8/8/8/8/8/6k1/5q2/7K b - - 0 1", Color::Black, "f2g2"),
        ] {
            let position = Position::from_fen(fen).unwrap();
            let mut proof = MateProofStore::new(&position, winner, ChessPly(1), None, 100).unwrap();
            let root = proof.root();
            let child = proof.open_child(root, mv(movement)).unwrap();
            assert_eq!(
                proof.conclusion(child).unwrap(),
                MateConclusion::ForcedMate(MateDistance::Exact(ChessPly(0)))
            );
            assert_eq!(
                proof.root_conclusion().conclusion,
                MateConclusion::ForcedMate(MateDistance::Exact(ChessPly(1)))
            );
            assert_eq!(proof.root_conclusion().exact_uci_mate(), Some(1));
            assert!(proof.root_conclusion().goal_satisfied());
            let strategy = proof.root_strategy().unwrap();
            assert_eq!(strategy.best_move(), mv(movement));
            assert_eq!(strategy.conclusion(), proof.root_conclusion());
            assert!(
                strategy
                    .validates_root(&position.snapshot(), position.ordered_legal_moves().moves())
            );
            assert!(
                strategy
                    .checked_child_snapshot()
                    .same_state(proof.snapshot(child).unwrap())
            );
            assert!(!strategy.validates_root(&position.snapshot(), &[mv(movement)]));
            let count = proof.len();
            assert_eq!(proof.open_child(root, mv(movement)).unwrap(), child);
            assert_eq!(proof.len(), count);
            proof.retract_child(root, mv(movement)).unwrap();
            assert_eq!(proof.root_conclusion().conclusion, MateConclusion::Unknown);
        }
    }

    #[test]
    fn terminal_precedes_zero_horizon_and_stalemate_is_not_mate() {
        let mate = Position::from_fen("7k/6Q1/5K2/8/8/8/8/8 b - - 0 1").unwrap();
        let proof = MateProofStore::new(&mate, Color::White, ChessPly(0), None, 8).unwrap();
        assert!(proof.root_conclusion().goal_satisfied());
        assert_eq!(proof.root_conclusion().exact_uci_mate(), None);
        let stale = Position::from_fen("7k/5K2/6Q1/8/8/8/8/8 b - - 0 1").unwrap();
        let proof = MateProofStore::new(&stale, Color::White, ChessPly(0), None, 8).unwrap();
        assert_eq!(proof.root_conclusion().conclusion, MateConclusion::Draw);
        let proof =
            MateProofStore::new(&Position::startpos(), Color::White, ChessPly(0), None, 8).unwrap();
        assert_eq!(
            proof.root_conclusion().conclusion,
            MateConclusion::NoMateWithin
        );
    }

    #[test]
    fn cooperative_line_does_not_close_unexamined_quiet_defense() {
        let position = Position::from_fen("6k1/8/5K2/8/8/8/8/6Q1 b - - 0 1").unwrap();
        let mut proof =
            MateProofStore::new(&position, Color::White, ChessPly(2), None, 100).unwrap();
        let root = proof.root();
        let cooperative = proof.open_child(root, mv("g8h8")).unwrap();
        proof.open_child(cooperative, mv("g1g7")).unwrap();
        assert!(matches!(
            proof.conclusion(cooperative).unwrap(),
            MateConclusion::ForcedMate(_)
        ));
        assert_eq!(proof.root_conclusion().conclusion, MateConclusion::Unknown);
        assert!(proof.missing_moves(root).unwrap().contains(&mv("g8f8")));
        let quiet = proof.open_child(root, mv("g8f8")).unwrap();
        let missing = proof.missing_moves(quiet).unwrap();
        for movement in missing {
            proof.open_child(quiet, movement).unwrap();
        }
        assert_eq!(
            proof.conclusion(quiet).unwrap(),
            MateConclusion::NoMateWithin
        );
        assert_eq!(
            proof.root_conclusion().conclusion,
            MateConclusion::NoMateWithin
        );
    }

    #[test]
    fn restricted_root_loss_stays_conditional_and_foreign_handles_are_rejected() {
        // A pawn move makes repetition evidence complete before the quiet
        // defense; a FEN-only unknown prefix cannot certify absent claims.
        let mut position = Position::from_fen("6k1/p7/5K2/8/8/8/P7/5Q2 w - - 0 1").unwrap();
        for text in ["a2a3", "a7a6", "f1g1"] {
            position.make_uci(text).unwrap();
        }
        let roots = [mv("g8h8")];
        let mut proof =
            MateProofStore::new(&position, Color::White, ChessPly(2), Some(&roots), 100).unwrap();
        let root = proof.root();
        let child = proof.open_child(root, roots[0]).unwrap();
        proof.open_child(child, mv("g1g7")).unwrap();
        // a7a6/a7a5 are still legal defenses inside the restricted child, so
        // the OR attack needs only its checked mate, while root is conditional.
        assert!(proof.root_conclusion().goal_satisfied());
        assert_eq!(proof.root_conclusion().scope, MateScope::RestrictedRoot);
        assert_eq!(proof.root_conclusion().exact_uci_mate(), Some(-1));
        let foreign = MateProofStore::new(&position, Color::White, ChessPly(2), None, 8).unwrap();
        assert_eq!(
            proof.conclusion(foreign.root()),
            Err(MateProofError::ForeignHandle)
        );
        proof.retract_child(root, roots[0]).unwrap();
        assert_eq!(proof.root_conclusion().conclusion, MateConclusion::Unknown);
    }

    #[test]
    fn known_claim_prevents_forced_mate_and_same_fen_keeps_different_history() {
        let mut position = Position::startpos();
        for text in [
            "g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1", "f6g8",
        ] {
            position.make_uci(text).unwrap();
        }
        let proof = MateProofStore::new(&position, Color::Black, ChessPly(4), None, 100).unwrap();
        assert_eq!(
            proof.root_conclusion().conclusion,
            MateConclusion::NoMateWithin
        );
        let imported = Position::from_fen(&position.to_fen()).unwrap();
        assert!(
            !proof
                .snapshot(proof.root())
                .unwrap()
                .same_state(&imported.snapshot())
        );
        let imported =
            MateProofStore::new(&imported, Color::Black, ChessPly(4), None, 100).unwrap();
        assert_eq!(
            imported.root_conclusion().conclusion,
            MateConclusion::Unknown
        );
    }

    #[test]
    fn forced_upper_bound_can_stop_before_exact_shortest_distance() {
        let mut position = Position::from_fen("7k/p4Q2/6K1/8/8/8/P7/8 b - - 0 1").unwrap();
        // Known irreversible boundary closes the unavailable repetition claim.
        position.make_uci("a7a6").unwrap();
        let mut proof =
            MateProofStore::new(&position, Color::White, ChessPly(3), None, 100).unwrap();
        let root = proof.root();
        let check = proof.open_child(root, mv("f7f6")).unwrap();
        assert_eq!(proof.missing_moves(check).unwrap(), [mv("h8g8")]);
        let response = proof.open_child(check, mv("h8g8")).unwrap();
        proof.open_child(response, mv("f6d8")).unwrap();
        assert_eq!(
            proof.root_conclusion().conclusion,
            MateConclusion::ForcedMate(MateDistance::Within(ChessPly(3)))
        );
        assert!(proof.root_conclusion().goal_satisfied());
        assert_eq!(proof.root_conclusion().exact_uci_mate(), None);
        let strategy = proof.root_strategy().unwrap();
        assert_eq!(strategy.best_move(), mv("f7f6"));
        assert_eq!(strategy.conclusion().exact_uci_mate(), None);
        proof.open_child(root, mv("f7g7")).unwrap();
        assert_eq!(
            proof.root_conclusion().conclusion,
            MateConclusion::ForcedMate(MateDistance::Exact(ChessPly(1)))
        );
        assert_eq!(proof.strategy_move(root).unwrap(), Some(mv("f7g7")));
        assert_eq!(proof.root_strategy().unwrap().best_move(), mv("f7g7"));
    }

    #[test]
    fn restricted_attacker_win_preserves_existence_but_not_global_shortest_distance() {
        let mut position = Position::from_fen("7k/p4Q2/6K1/8/8/8/P7/8 b - - 0 1").unwrap();
        position.make_uci("a7a6").unwrap();
        let roots = [mv("f7f6")];
        let mut proof =
            MateProofStore::new(&position, Color::White, ChessPly(3), Some(&roots), 100).unwrap();
        let root = proof.root();
        let check = proof.open_child(root, roots[0]).unwrap();
        let response = proof.open_child(check, mv("h8g8")).unwrap();
        proof.open_child(response, mv("f6d8")).unwrap();
        assert_eq!(
            proof.conclusion(root).unwrap(),
            MateConclusion::ForcedMate(MateDistance::Exact(ChessPly(3)))
        );
        assert_eq!(proof.root_conclusion().scope, MateScope::FullRoot);
        assert_eq!(
            proof.root_conclusion().conclusion,
            MateConclusion::ForcedMate(MateDistance::Within(ChessPly(3)))
        );
        assert_eq!(proof.root_conclusion().exact_uci_mate(), None);
        let strategy = proof.root_strategy().unwrap();
        assert!(strategy.root_restricted());
        assert_eq!(strategy.best_move(), roots[0]);
        assert!(strategy.validates_root(&position.snapshot(), &roots));
        assert!(!strategy.validates_root(
            &Position::from_fen(&position.to_fen()).unwrap().snapshot(),
            &roots,
        ));
    }

    #[test]
    fn root_loss_strategy_delays_within_the_original_proved_move_domain() {
        let mut position = Position::from_fen("6k1/p7/5K2/8/8/8/P7/5Q2 w - - 0 1").unwrap();
        for movement in ["a2a3", "a7a6", "f1g1"] {
            position.make_uci(movement).unwrap();
        }
        // Qg1 checks the king: these three king evasions are the full legal
        // root domain. Passing all of them via Some would remain unrestricted.
        let legal = position.legal_moves();
        assert_eq!(legal.len(), 3);
        assert!(
            [mv("g8h8"), mv("g8f8"), mv("g8h7")]
                .iter()
                .all(|movement| legal.contains(movement))
        );
        let allowed = [mv("g8h8"), mv("g8f8")];
        assert!(allowed.len() < legal.len());
        assert!(legal.contains(&mv("g8h7")) && !allowed.contains(&mv("g8h7")));
        let mut proof =
            MateProofStore::new(&position, Color::White, ChessPly(4), Some(&allowed), 32).unwrap();
        let root = proof.root();
        assert!(proof.root_strategy().is_none());
        for branch in [&["g8h8", "g1g7"][..], &["g8f8", "g1g7", "f8e8", "g7e7"][..]] {
            let mut parent = root;
            for movement in branch {
                parent = proof.open_child(parent, mv(movement)).unwrap();
            }
        }
        let strategy = proof.root_strategy().unwrap();
        assert_eq!(strategy.best_move(), mv("g8f8"));
        assert!(strategy.root_restricted());
        assert_eq!(strategy.conclusion().scope, MateScope::RestrictedRoot);
        assert_eq!(
            strategy.conclusion().conclusion,
            MateConclusion::ForcedMate(MateDistance::Within(ChessPly(4)))
        );
        assert!(strategy.validates_root(&position.snapshot(), &allowed));
        assert_eq!(strategy.conclusion().exact_uci_mate(), None);
    }

    #[test]
    fn capacity_failure_does_not_fill_a_defense_slot() {
        let position = Position::startpos();
        let mut proof = MateProofStore::new(&position, Color::White, ChessPly(3), None, 1).unwrap();
        let root = proof.root();
        assert_eq!(
            proof.open_child(root, mv("e2e4")),
            Err(MateProofError::Capacity)
        );
        assert!(proof.child(root, mv("e2e4")).unwrap().is_none());
        assert_eq!(proof.root_conclusion().conclusion, MateConclusion::Unknown);
    }

    #[test]
    fn store_line_requires_exact_start_and_full_rules_replay_before_publication() {
        let position = Position::from_fen("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1").unwrap();
        let mut stores = PalsStores::new(Default::default());
        let state = stores.states.insert(position.snapshot()).unwrap();
        let prefix = stores.lines.root(state).unwrap();
        let mate_line = stores.lines.append(prefix, &[mv("f7g7")]).unwrap();
        // A raw line container does not promise that play stops at terminal.
        let invalid_tail = stores.lines.append(mate_line, &[mv("h8h7")]).unwrap();
        let mut proof =
            MateProofStore::new(&position, Color::White, ChessPly(2), None, 32).unwrap();
        let root = proof.root();
        assert_eq!(
            proof.observe_stored_line(root, &stores, invalid_tail),
            Err(MateProofError::AlreadyClosed)
        );
        assert_eq!(proof.len(), 1);
        assert_eq!(proof.root_conclusion().conclusion, MateConclusion::Unknown);
        proof.observe_stored_line(root, &stores, mate_line).unwrap();
        assert!(proof.root_conclusion().goal_satisfied());
        let mut changed = Position::startpos();
        changed.make_uci("e2e4").unwrap();
        let another = stores.states.insert(changed.snapshot()).unwrap();
        let foreign_line = stores.lines.root(another).unwrap();
        assert_eq!(
            proof.observe_stored_line(root, &stores, foreign_line),
            Err(MateProofError::ForeignHandle)
        );
    }
}
