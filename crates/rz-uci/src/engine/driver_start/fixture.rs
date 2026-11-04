//! Test-only actual Session Start + binding custody. No raw source issuer.
//!
//! The fixed PositionPort answers from the actual Rules position, preserving
//! its history rather than rebuilding it from FEN or inventing legal/terminal
//! flags. A terminal Rules reply stays a Session publisher result with no Start.

#![cfg(test)]

use super::DriverStartSource;
use crate::bridge::{BuildSearchSettings, NodeBudgetSemantics, PreparedSearchBinding, SideToMove};
use crate::contracts::SessionScope;
use crate::engine::ProcessClock;
use crate::parser::{Command, GoLimits, ParserLimits, PositionSpec};
use crate::search_driver::SearchSessionContext;
use crate::{Effect, EngineIdentity, PositionPort, PreparedPosition, Session, SessionResult};
use rz_contracts::pals::SearchAuthority;
use rz_position::{BoardMove, Color, PlayStatus, Position};
use rz_search::contract_time::ContractCancellation;
use std::{
    num::NonZeroU16,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

struct FixturePositionPort {
    position: Position,
}
impl PositionPort for FixturePositionPort {
    type Snapshot = Position;
    type Error = String;

    fn prepare(&self, spec: &PositionSpec) -> Result<PreparedPosition<Position>, String> {
        if spec != &PositionSpec::default() {
            return Err("fixed Rules fixture accepts only its initial position request".into());
        }
        let legal = self.position.ordered_legal_moves();
        let status = self
            .position
            .play_status_from_view(&legal)
            .map_err(|error| error.to_string())?;
        Ok(PreparedPosition {
            snapshot: self.position.clone(),
            legal_moves: legal.moves().iter().map(ToString::to_string).collect(),
            exact_terminal: matches!(status, PlayStatus::Terminal { .. }),
        })
    }
}

// One fixture invocation returns its actual Start context or Rules-only result.
// Neither variant is queued or reconstructs the other's admission authority.
#[allow(clippy::large_enum_variant)]
pub(crate) enum FixtureStart {
    Started(SearchSessionContext),
    RulesOnly(SessionResult<Position>),
}

/// Mirrors the actual engine's raw-go preservation before ponder normalization.
/// The Session still owns its real ticket and canonical root restriction.
#[allow(clippy::too_many_arguments)]
pub(crate) fn issue_start(
    position: &Position,
    original_go: GoLimits,
    authority: SearchAuthority,
    clock: &ProcessClock,
    start: Instant,
    settings: BuildSearchSettings,
    cleanup_margin: Duration,
) -> Result<FixtureStart, String> {
    let mut session = Session::new(
        FixturePositionPort {
            position: position.clone(),
        },
        EngineIdentity {
            name: "Actual Rules Start fixture".into(),
            author: "Rovent source tests".into(),
        },
        Vec::new(),
        ParserLimits::default(),
    )
    .map_err(|error| error.to_string())?;
    session
        .set_search_goal_capability(NonZeroU16::new(64))
        .map_err(|error| error.to_string())?;
    session
        .set_node_budget_semantics(NodeBudgetSemantics::RoventPositionVisits)
        .map_err(|error| error.to_string())?;
    let mut normalized = original_go.clone();
    if normalized.ponder {
        // This is the production engine's bounded ponder normalization. Raw
        // clocks remain in original_go; the actual binding uses its wall limit.
        normalized.ponder = false;
        normalized.infinite = true;
        normalized.movetime_ms = None;
        normalized.white_time_ms = None;
        normalized.black_time_ms = None;
        normalized.white_increment_ms = None;
        normalized.black_increment_ms = None;
        normalized.moves_to_go = None;
    }
    let mut out = session.handle_command(Command::Go(normalized.clone()));
    if !out.accepted {
        return Err(format!(
            "Session rejected actual Start: {:?}",
            out.diagnostics
        ));
    }
    let start_effect = out
        .effects
        .iter()
        .position(|effect| matches!(effect, Effect::Start { .. }));
    let Some(index) = start_effect else {
        // In particular, exact terminal Rules never gets a fabricated ticket.
        return Ok(FixtureStart::RulesOnly(out));
    };
    let Effect::Start {
        ticket,
        snapshot,
        limits,
        ..
    } = out.effects.remove(index)
    else {
        return Err("Session Start effect changed within a local fixture".into());
    };
    if !snapshot.snapshot().same_state(&position.snapshot()) {
        return Err("Session Start changed its actual Rules position".into());
    }
    let side = match snapshot.side_to_move() {
        Color::White => SideToMove::White,
        Color::Black => SideToMove::Black,
    };
    let binding = PreparedSearchBinding::prepare(
        &normalized,
        side,
        start,
        settings,
        NonZeroU16::new(64),
        NodeBudgetSemantics::RoventPositionVisits,
    )
    .map_err(|error| error.to_string())?
    .bind(ticket.clone());
    let root_moves = limits
        .searchmoves
        .as_ref()
        .map(|moves| {
            moves
                .iter()
                .map(|movement| BoardMove::from_uci(movement).map_err(|error| error.to_string()))
                .collect::<Result<Vec<_>, _>>()
                .map(Arc::from)
        })
        .transpose()?;
    let cancellation = ContractCancellation::from_control(binding.control());
    let shutdown_deadline = binding
        .control()
        .deadline
        .checked_add(cleanup_margin)
        .ok_or_else(|| "actual Start fixture cleanup deadline overflow".to_owned())?;
    let mut context = SearchSessionContext {
        authority,
        original_go: Some(Arc::new(original_go)),
        original_time_budget: Some(*binding.budget()),
        original_start: None,
        effective_clock: None,
        root_moves,
        goal: binding.goal(),
        requested_nodes: limits.nodes,
        control: binding.control().clone(),
        cancellation: cancellation.token(),
        shutdown_deadline,
        current: Arc::new(Mutex::new(SessionScope::Search(authority))),
    };
    let source = DriverStartSource::from_start(&ticket, &snapshot, &binding, &context, clock)
        .map_err(|error| error.to_string())?;
    context.original_start = Some(source);
    Ok(FixtureStart::Started(context))
}
