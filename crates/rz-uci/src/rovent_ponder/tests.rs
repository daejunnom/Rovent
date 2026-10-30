use super::*;
use crate::bridge::{BuildSearchSettings, SideToMove};
use crate::parser::{Command, GoLimits, ParserLimits, PositionSpec};
use crate::{Effect, EngineIdentity, PositionPort, PreparedPosition, Session};
use rz_contracts::{Digest, GameGeneration, ProcessEpoch, RootGeneration};
use rz_position::{PlayStatus, Position};
use rz_search::time::TimeBudgetConfig;

struct RulesPort;
impl PositionPort for RulesPort {
    type Snapshot = Position;
    type Error = String;

    fn prepare(&self, spec: &PositionSpec) -> Result<PreparedPosition<Position>, String> {
        if spec != &PositionSpec::default() {
            return Err("ponder clock fixture accepts its original Rules root only".into());
        }
        let position = Position::startpos();
        let legal = position.ordered_legal_moves();
        let status = position
            .play_status_from_view(&legal)
            .map_err(|error| error.to_string())?;
        Ok(PreparedPosition {
            snapshot: position.clone(),
            legal_moves: legal.moves().iter().map(ToString::to_string).collect(),
            exact_terminal: matches!(status, PlayStatus::Terminal { .. }),
        })
    }
}

fn ticket() -> SearchTicket {
    let mut session = Session::new(
        RulesPort,
        EngineIdentity {
            name: "Ponder clock source fixture".into(),
            author: "Rovent".into(),
        },
        Vec::new(),
        ParserLimits::default(),
    )
    .unwrap();
    session
        .set_node_budget_semantics(NodeBudgetSemantics::RoventPositionVisits)
        .unwrap();
    let out = session.handle_command(Command::Go(GoLimits {
        infinite: true,
        ..GoLimits::default()
    }));
    assert!(out.accepted);
    out.effects
        .into_iter()
        .find_map(|effect| match effect {
            Effect::Start { ticket, .. } => Some(ticket),
            _ => None,
        })
        .expect("actual Session must issue the Start ticket")
}

fn authority(clock: &ProcessClock) -> SearchAuthority {
    SearchAuthority {
        epoch: clock.epoch(),
        game: GameGeneration(1),
        root: RootGeneration(1),
        // This fixture selects an implementation identity, while the actual
        // Session alone issues the opaque Start ticket used by the owner.
        implementation: Digest([29; 32]),
    }
}

fn prepared(
    start: Instant,
    limits: GoLimits,
    semantics: NodeBudgetSemantics,
) -> PreparedSearchBinding {
    PreparedSearchBinding::prepare(
        &limits,
        SideToMove::White,
        start,
        BuildSearchSettings {
            time_config: TimeBudgetConfig::default(),
            max_simulations: 1024,
            untimed_limit: Duration::from_secs(60),
        },
        None,
        semantics,
    )
    .unwrap()
}

fn original(start: Instant) -> TimeBudget {
    *prepared(
        start,
        GoLimits {
            infinite: true,
            ..GoLimits::default()
        },
        NodeBudgetSemantics::RoventPositionVisits,
    )
    .budget()
}

fn normal(start: Instant, milliseconds: u64) -> PreparedSearchBinding {
    prepared(
        start,
        GoLimits {
            movetime_ms: Some(milliseconds),
            ..GoLimits::default()
        },
        NodeBudgetSemantics::RoventPositionVisits,
    )
}

#[test]
fn hit_is_same_ticket_once_and_caps_every_original_boundary() {
    let clock = ProcessClock::new(ProcessEpoch(9201));
    let start = Instant::now();
    let original = original(start);
    let cleanup = original.output_deadline + Duration::from_secs(1);
    let authority = authority(&clock);
    let ticket = ticket();
    let (mut owner, handle) =
        RoventPonderClockOwner::new(ticket.clone(), authority, original, cleanup, &clock).unwrap();
    let hit_at = start + Duration::from_secs(1);
    let normal = normal(hit_at, 1000);
    let before = handle.snapshot(hit_at);
    let observed = handle.version();
    assert_eq!(
        owner
            .hit(
                &ticket,
                SessionScope::Search(authority),
                hit_at,
                &normal,
                hit_at,
            )
            .unwrap(),
        HitDisposition::Live
    );
    assert!(handle.wait_changed(observed, Duration::ZERO));
    let snapshot = handle.snapshot(hit_at);
    let effective = snapshot.budget();
    assert_eq!(effective.start, original.start);
    assert_eq!(effective.soft_deadline, normal.budget().soft_deadline);
    assert_eq!(
        effective.admission_deadline,
        normal.budget().admission_deadline
    );
    assert_eq!(effective.hard_deadline, normal.budget().hard_deadline);
    assert_eq!(effective.output_deadline, normal.budget().output_deadline);
    assert_eq!(
        effective.allocation,
        effective.output_deadline.duration_since(original.start)
    );
    assert_eq!(*before.budget(), original);
    assert!(!before.has_hit());
    assert_eq!(snapshot.hit().unwrap().hit_at(), hit_at);
    assert!(snapshot.can_admit(hit_at));
    assert!(!snapshot.can_admit(effective.admission_deadline));
    assert!(snapshot.soft_expired(effective.soft_deadline));
    assert!(snapshot.hard_expired(effective.hard_deadline));
    let expired = handle.snapshot(effective.hard_deadline);
    assert!(expired.hard_expired(start));
    assert!(!expired.can_admit(start));
    let regressed = handle.snapshot(start);
    assert!(regressed.hard_expired(start));
    assert!(!regressed.can_admit(start));
    assert_eq!(
        owner
            .hit(
                &ticket,
                SessionScope::Search(authority),
                hit_at,
                &normal,
                hit_at,
            )
            .unwrap_err()
            .code,
        ErrorCode::Stale
    );
}

#[test]
fn normal_allocation_cannot_extend_the_absolute_ponder_wall() {
    let clock = ProcessClock::new(ProcessEpoch(9202));
    let start = Instant::now();
    let original = original(start);
    let ticket = ticket();
    let authority = authority(&clock);
    let (mut owner, handle) = RoventPonderClockOwner::new(
        ticket.clone(),
        authority,
        original,
        original.output_deadline,
        &clock,
    )
    .unwrap();
    let hit_at = start + Duration::from_secs(2);
    let normal = normal(hit_at, 60_000);
    assert_eq!(
        owner
            .hit(
                &ticket,
                SessionScope::Search(authority),
                hit_at,
                &normal,
                hit_at,
            )
            .unwrap(),
        HitDisposition::Live
    );
    assert_eq!(*handle.snapshot(hit_at).budget(), original);
}

#[test]
fn hit_at_original_hard_wall_keeps_no_live_work_sticky() {
    let clock = ProcessClock::new(ProcessEpoch(9203));
    let start = Instant::now();
    let original = original(start);
    let ticket = ticket();
    let authority = authority(&clock);
    let (mut owner, handle) = RoventPonderClockOwner::new(
        ticket.clone(),
        authority,
        original,
        original.output_deadline,
        &clock,
    )
    .unwrap();
    let hit_at = original.hard_deadline;
    assert_eq!(
        owner
            .hit(
                &ticket,
                SessionScope::Search(authority),
                hit_at,
                &normal(hit_at, 1000),
                hit_at,
            )
            .unwrap(),
        HitDisposition::NoLiveWork
    );
    let snapshot = handle.snapshot(start);
    assert!(!snapshot.is_closed());
    assert!(snapshot.has_hit());
    assert!(!snapshot.can_admit(start));
    assert!(!snapshot.can_accept(start));
    assert!(snapshot.hard_expired(start));
    assert_eq!(snapshot.budget().output_deadline, original.output_deadline);
    assert_eq!(snapshot.budget().hard_deadline, original.hard_deadline);
}

#[test]
fn foreign_ticket_scope_and_wrong_preparation_do_not_consume_hit() {
    let clock = ProcessClock::new(ProcessEpoch(9204));
    let start = Instant::now();
    let original = original(start);
    let ticket = ticket();
    let foreign = self::ticket();
    let authority = authority(&clock);
    let (mut owner, handle) = RoventPonderClockOwner::new(
        ticket.clone(),
        authority,
        original,
        original.output_deadline,
        &clock,
    )
    .unwrap();
    let hit_at = start + Duration::from_secs(1);
    let normal = normal(hit_at, 1000);
    let scope = SessionScope::Search(authority);
    assert_eq!(
        owner
            .hit(&foreign, scope, hit_at, &normal, hit_at)
            .unwrap_err()
            .code,
        ErrorCode::IdentityMismatch
    );
    let foreign_scope = SessionScope::Search(SearchAuthority {
        root: RootGeneration(2),
        ..authority
    });
    assert_eq!(
        owner
            .hit(&ticket, foreign_scope, hit_at, &normal, hit_at)
            .unwrap_err()
            .code,
        ErrorCode::IdentityMismatch
    );
    assert_eq!(
        owner
            .hit(&ticket, scope, start, &normal, hit_at)
            .unwrap_err()
            .code,
        ErrorCode::IdentityMismatch
    );
    let canceled = self::normal(hit_at, 1000);
    canceled.control().cancel();
    assert_eq!(
        owner
            .hit(&ticket, scope, hit_at, &canceled, hit_at)
            .unwrap_err()
            .code,
        ErrorCode::Canceled
    );
    let legacy = prepared(
        hit_at,
        GoLimits {
            movetime_ms: Some(1000),
            ..GoLimits::default()
        },
        NodeBudgetSemantics::LegacySimulations,
    );
    assert_eq!(
        owner
            .hit(&ticket, scope, hit_at, &legacy, hit_at)
            .unwrap_err()
            .code,
        ErrorCode::IdentityMismatch
    );
    let before_origin = clock.runtime_clock().origin() - Duration::from_secs(1);
    let invalid_observation = handle.snapshot(before_origin);
    assert!(!invalid_observation.can_admit(start));
    assert!(invalid_observation.hard_expired(start));
    assert_eq!(
        owner
            .hit(&ticket, scope, before_origin, &normal, hit_at)
            .unwrap_err()
            .code,
        ErrorCode::InvalidInput
    );
    assert!(!handle.snapshot(hit_at).has_hit());
    assert_eq!(
        owner.hit(&ticket, scope, hit_at, &normal, hit_at).unwrap(),
        HitDisposition::Live
    );
}

#[test]
fn registration_requires_the_same_authority_domain_origin_and_boundaries() {
    let clock = ProcessClock::new(ProcessEpoch(9205));
    let start = Instant::now();
    let original = original(start);
    let cleanup = original.output_deadline;
    let authority = authority(&clock);
    let (_owner, handle) =
        RoventPonderClockOwner::new(ticket(), authority, original, cleanup, &clock).unwrap();
    let hard = clock.deadline(original.hard_deadline).unwrap();
    assert!(handle.matches_registration(authority, &clock.runtime_clock(), hard, cleanup));
    let foreign_origin =
        ContractSystemClock::with_origin(clock.epoch(), start + Duration::from_secs(1));
    assert!(!handle.matches_registration(authority, &foreign_origin, hard, cleanup));
    assert!(!handle.matches_registration(
        authority,
        &clock.runtime_clock(),
        Deadline {
            at: rz_contracts::MonotonicTick(hard.at.0 + 1),
            ..hard
        },
        cleanup,
    ));
    assert!(!handle.matches_registration(
        authority,
        &clock.runtime_clock(),
        hard,
        cleanup + Duration::from_secs(1)
    ));
    assert!(handle.shares_clock_with(&handle.clone()));
    let (_other_owner, other) =
        RoventPonderClockOwner::new(ticket(), authority, original, cleanup, &clock).unwrap();
    assert!(!handle.shares_clock_with(&other));
}

#[test]
fn close_and_drop_are_sticky_and_notify_before_wait() {
    let clock = ProcessClock::new(ProcessEpoch(9206));
    let start = Instant::now();
    let original = original(start);
    let authority = authority(&clock);
    let ticket = ticket();
    let (mut owner, handle) = RoventPonderClockOwner::new(
        ticket.clone(),
        authority,
        original,
        original.output_deadline,
        &clock,
    )
    .unwrap();
    let observed = handle.version();
    owner.close();
    assert!(handle.wait_changed(observed, Duration::ZERO));
    let closed = handle.snapshot(start);
    assert!(closed.is_closed());
    assert!(!closed.can_admit(start));
    assert!(closed.hard_expired(start));
    let closed_version = handle.version();
    owner.close();
    assert_eq!(handle.version(), closed_version);
    assert_eq!(
        owner
            .hit(
                &ticket,
                SessionScope::Search(authority),
                start,
                &normal(start, 1000),
                start
            )
            .unwrap_err()
            .code,
        ErrorCode::Canceled
    );
    let (other_owner, other) = RoventPonderClockOwner::new(
        self::ticket(),
        authority,
        original,
        original.output_deadline,
        &clock,
    )
    .unwrap();
    let observed = other.version();
    let signal = other.signal();
    drop(other_owner);
    assert!(other.snapshot(start).is_closed());
    assert!(signal.wait_changed(observed, Duration::ZERO));
}

#[test]
fn original_origin_allocation_order_and_cleanup_are_checked_without_renewal() {
    let clock = ProcessClock::new(ProcessEpoch(9207));
    let start = Instant::now();
    let original = original(start);
    let authority = authority(&clock);
    let invalid = TimeBudget {
        allocation: original.allocation + Duration::from_nanos(1),
        ..original
    };
    assert_eq!(
        RoventPonderClockOwner::new(
            ticket(),
            authority,
            invalid,
            original.output_deadline,
            &clock
        )
        .unwrap_err()
        .code,
        ErrorCode::InvalidInput
    );
    let invalid = TimeBudget {
        soft_deadline: original.hard_deadline,
        ..original
    };
    assert_eq!(
        RoventPonderClockOwner::new(
            ticket(),
            authority,
            invalid,
            original.output_deadline,
            &clock
        )
        .unwrap_err()
        .code,
        ErrorCode::InvalidInput
    );
    assert_eq!(
        RoventPonderClockOwner::new(
            ticket(),
            authority,
            original,
            original.hard_deadline,
            &clock
        )
        .unwrap_err()
        .code,
        ErrorCode::InvalidInput
    );
    let before_origin = clock.runtime_clock().origin() - Duration::from_secs(1);
    let invalid = TimeBudget {
        start: before_origin,
        ..original
    };
    assert_eq!(
        RoventPonderClockOwner::new(
            ticket(),
            authority,
            invalid,
            original.output_deadline,
            &clock
        )
        .unwrap_err()
        .code,
        ErrorCode::InvalidInput
    );
}
