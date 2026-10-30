use super::*;
use crate::contracts::{ContractSessionOwner, SessionScope};
use crate::engine::{EngineSettings, OwnerRegistry, RulesUciPort};
use crate::parser::ParserLimits;
use crate::{Effect, EngineIdentity, Event, SearchCompletion, Session};
use rz_contracts::{Color, Digest, GameGeneration, ProcessEpoch, RootGeneration};
use rz_position::PositionLimits;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Duration;

// Construct the concrete Start from the Session's real effect and the same
// binding and cancellation owner used by ContractSessionOwner. The hook's
// atomics are observables only; no physical work is started by this fixture.
fn started() -> (
    ContractSessionOwner,
    Session<RulesUciPort>,
    SearchSessionContext,
    Arc<AtomicU64>,
    Arc<AtomicBool>,
) {
    let clock = ProcessClock::new(ProcessEpoch(42));
    let authority = SearchAuthority {
        epoch: clock.epoch(),
        game: GameGeneration(1),
        root: RootGeneration(0),
        implementation: Digest([7; 32]),
    };
    let mut owner = ContractSessionOwner::new_scope(
        SessionScope::Search(authority),
        clock.epoch(),
        clock.b_clock(),
        EngineSettings::default().search,
    )
    .unwrap();
    let mut session = Session::new(
        RulesUciPort::new(Arc::new(OwnerRegistry::default()), PositionLimits::default()),
        EngineIdentity { name: "original start test".into(), author: "RoveZero".into() },
        Vec::new(),
        ParserLimits::default(),
    )
    .unwrap();
    let mut out = owner.handle_line(&mut session, "go nodes 1", |_| Ok(Color::White));
    assert!(out.session.accepted);
    let start_index = out.session.effects.iter().position(|e| matches!(e, Effect::Start { .. })).unwrap();
    let Effect::Start { ticket, snapshot, limits, .. } = out.session.effects.remove(start_index) else { unreachable!() };
    let authority = match owner.lifecycle_scope() {
        SessionScope::Search(authority) => authority,
        _ => panic!("start did not preserve search scope"),
    };
    let binding = owner.active_binding().unwrap();
    let control = binding.control().clone();
    let mut context = SearchSessionContext {
        authority,
        original_go: Some(Arc::new(limits.clone())),
        original_time_budget: Some(*binding.budget()),
        original_start: None,
        effective_clock: None,
        root_moves: None,
        goal: binding.goal(),
        requested_nodes: limits.nodes,
        control: control.clone(),
        cancellation: owner.cancellation().unwrap().token(),
        shutdown_deadline: control.deadline + Duration::from_secs(2),
        current: Arc::new(Mutex::new(SessionScope::Search(authority))),
    };
    let start = DriverStartSource::from_start(&ticket, snapshot.rules_position(), binding, &context, &clock).unwrap();
    context.original_start = Some(Arc::clone(&start));
    owner.bind_original_start(&ticket, start).unwrap();
    let game = Arc::new(AtomicU64::new(0));
    let root = Arc::new(AtomicBool::new(false));
    context.bind_original_collection_abort(Arc::clone(&game), Arc::clone(&root)).unwrap();
    (owner, session, context, game, root)
}

#[test]
fn accepted_stop_and_new_game_abort_exact_start_but_rejected_position_does_not() {
    for command in ["stop", "ucinewgame", "position startpos", "quit"] {
        let (mut owner, mut session, context, game, root) = started();
        let bad = owner.handle_line(&mut session, "position startpos moves e2e5", |_| Ok(Color::White));
        assert!(!bad.session.accepted);
        assert!(!root.load(Ordering::Acquire));
        assert_eq!(game.load(Ordering::Acquire), 0);
        assert!(!context.cancellation.is_canceled());
        let out = owner.handle_line(&mut session, command, |_| Ok(Color::White));
        assert!(out.session.accepted, "{command}");
        assert!(root.load(Ordering::Acquire), "{command}");
        assert_ne!(game.load(Ordering::Acquire) & crate::search_driver::ROVENT_COLLECTION_ABORT_BIT, 0);
        assert!(context.cancellation.is_canceled());
    }
}

#[test]
fn natural_completion_closes_common_token_without_aborting_collection() {
    let (mut owner, mut session, context, game, root) = started();
    let ticket = owner.active_ticket().unwrap().clone();
    let out = owner.handle_event(&mut session, Event::Complete {
        ticket,
        completion: SearchCompletion::Completed { bestmove: Some("e2e4".into()) },
    });
    assert!(out.session.accepted);
    assert!(context.cancellation.is_canceled());
    assert!(!root.load(Ordering::Acquire));
    assert_eq!(game.load(Ordering::Acquire), 0);
    drop(owner);
    assert!(!root.load(Ordering::Acquire));
}

// Exercise the actual accepted stop effect against the same atomic transition
// used for publication permission. This does not perform filesystem I/O.
#[test]
fn accepted_stop_and_publication_grant_have_one_atomic_order() {
    use std::sync::Barrier;
    const GRANTED: u64 = 1u64 << 62;
    const ABORTED: u64 = crate::search_driver::ROVENT_COLLECTION_ABORT_BIT;

    for _ in 0..24 {
        let (mut owner, mut session, context, game, root) = started();
        let barrier = Barrier::new(2);
        std::thread::scope(|scope| {
            let stopped = scope.spawn(|| {
                barrier.wait();
                owner.handle_line(&mut session, "stop", |_| Ok(Color::White))
            });
            barrier.wait();
            let granted = game.compare_exchange(0, GRANTED, Ordering::AcqRel, Ordering::Acquire).is_ok();
            assert!(stopped.join().expect("stop worker").session.accepted);
            assert!(context.cancellation.is_canceled());
            assert!(root.load(Ordering::Acquire));
            assert_eq!(game.load(Ordering::Acquire), ABORTED | if granted { GRANTED } else { 0 });
        });
    }
}
