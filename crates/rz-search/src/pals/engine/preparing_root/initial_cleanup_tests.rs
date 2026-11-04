//! Actual initial request/actor cleanup; no native R search or training.
use super::*;
use crate::contract_time::InstantClock;
use crate::cpu_value::nnuec::v2::retirement::{NnuecCacheInvalidation, NnuecCacheReleaseAck};
use crate::cpu_value::nnuec::v2::review::{PreparationActor, PreparationError, PreparationStatus};
use crate::pals::engine::adaptive_port::AdaptivePortModel;
use crate::pals::rovent_frontier::{FrontierLimits, FrontierStop};
use rz_contracts::rovent::{ExecutionMode, SearchGoal};
use rz_contracts::{
    ClockDomain, Digest, GameGeneration, MonotonicTick, OwnerId, ProcessEpoch, RootGeneration,
};
use std::num::NonZeroU16;

const WINDOW: Duration = Duration::from_secs(5);
type Root = ResearcherPreparingRoot<AdaptivePortModel>;
fn root() -> Root {
    let owner = super::super::tests::owner();
    let authority = SearchAuthority {
        epoch: ProcessEpoch(91),
        game: GameGeneration(0),
        root: RootGeneration(7),
        implementation: Digest([91; 32]),
    };
    let clock = InstantClock::new(ClockDomain(authority.epoch), Instant::now());
    let deadline = Deadline {
        clock: clock.domain(),
        at: MonotonicTick(20_000_000_000),
    };
    let hard = clock.instant_at(deadline.at).unwrap();
    let mut root = owner
        .bind_root(
            Position::startpos(),
            authority,
            deadline,
            hard,
            Arc::new(AtomicBool::new(false)),
        )
        .unwrap_or_else(|_| panic!("same semantic root registration"));
    let next = Arc::new(std::sync::atomic::AtomicU64::new(1));
    let issuer: adaptive_port::AdaptiveRulesOwnerIssuer =
        Arc::new(move || Ok(OwnerId(next.fetch_add(1, Ordering::Relaxed))));
    root.attach_original_controls(
        ResearcherRootStartup {
            mode: ExecutionMode::CpuPcr,
            root_budget: Digest([92; 32]),
            goal: SearchGoal {
                depth_plies: NonZeroU16::new(5),
                mate_moves: None,
            },
            nodes: Some(2048),
            selective_policy: Digest([93; 32]),
            frontier_limits: FrontierLimits {
                maximum_goal_plies: NonZeroU16::new(16).unwrap(),
                maximum_frames: 16,
                maximum_paths: 64,
                maximum_mate_nodes: 64,
                maximum_native_receipts: 8,
            },
            clock,
            limits: PalsLimits {
                deadline: hard,
                max_rounds: 4,
                max_cpu_nodes: 256,
                cpu_depth: 2,
            },
            cleanup_deadline: hard + WINDOW,
        },
        issuer,
        None,
    )
    .unwrap();
    root.prepare_original_root_rules().unwrap();
    root
}
fn cache_release(
    actor: &PreparationActor,
    ticket: &NnuecCacheInvalidation,
) -> NnuecCacheReleaseAck {
    let until = Instant::now() + WINDOW;
    loop {
        if let Some(ack) = actor.try_context_cache_release(ticket).unwrap() {
            return ack;
        }
        assert!(Instant::now() < until, "actual cache acknowledgement");
        std::thread::yield_now();
    }
}
fn stop_actor(actor: &mut PreparationActor) {
    actor.request_shutdown();
    let until = Instant::now() + WINDOW;
    loop {
        if actor.try_join().unwrap().is_some() {
            return;
        }
        assert!(Instant::now() < until, "actual actor join");
        std::thread::yield_now();
    }
}
#[test]
fn unsubmitted_initial_request_is_consumed_without_a_false_terminal_or_view_pin() {
    let mut root = root();
    let same_store = root.stores().read_owner();
    let authority = root.authority();
    let request = root.prepare_initial_nnuec_review().unwrap();
    let (original, seal) = request.into_parts();
    // Identical public metadata and the SAME View are not the original Request.
    let foreign = Request::admit(
        root.source.model(),
        Arc::clone(root.initial_view.as_ref().unwrap()),
        root.stores(),
        root.root_state_id(),
        authority.root.0,
        root.hard_deadline,
        Arc::clone(&root.cancellation),
    )
    .unwrap();
    assert_eq!(original.provenance(), foreign.provenance());
    root.cancel();
    let mut actor = PreparationActor::spawn(Arc::clone(root.source.model())).unwrap();
    let invalidation = actor.invalidate_context_cache().unwrap();
    let cache = cache_release(&actor, &invalidation);
    let rejected = match root.retire_after_initial_preparation(
        seal,
        InitialResearcherRelease::Unsubmitted(foreign),
        cache,
    ) {
        Err(rejected) => rejected,
        Ok(_) => panic!("new same-view Request must not replace the original"),
    };
    assert!(same_store.matches(rejected.root.stores()));
    assert_eq!(rejected.root.stores().review_pin_count().unwrap(), 2);
    let InitialPreparingReleaseCustody::Unreleased(InitialResearcherRelease::Unsubmitted(foreign)) =
        rejected.release
    else {
        panic!("exact failed input retained")
    };
    drop(foreign);
    let retired = rejected
        .root
        .retire_after_initial_preparation(
            rejected.original_review_seal.unwrap(),
            InitialResearcherRelease::Unsubmitted(original),
            rejected.cache_release,
        )
        .unwrap_or_else(|_| panic!("exact unsubmitted request cleanup"));
    assert!(matches!(
        retired.release,
        InitialResearcherReleaseReceipt::Unsubmitted(_)
    ));
    assert!(same_store.matches(retired.retired.owner.stores()));
    assert_eq!(
        retired.retired.owner.stores().review_pin_count().unwrap(),
        0
    );
    assert_eq!(
        retired.retired.owner.last_adaptive_authority(),
        Some(authority)
    );
    assert_eq!(
        retired.retired.frontier.as_ref().unwrap().snapshot.goal,
        FrontierGoalStatus::Stopped(FrontierStop::Canceled)
    );
    let observed = retired.retired.ledger.registration_observation().unwrap();
    assert_eq!(observed.spent_native_nodes, 0);
    assert_eq!(observed.active_jobs, 0);
    // An actual reusable actor is still alive. Only the test finally joins it.
    let followup_invalidation = actor.invalidate_context_cache().unwrap();
    assert!(cache_release(&actor, &followup_invalidation).epoch() > retired.cache_release.epoch());
    let owner = retired.retired.owner;
    let next = SearchAuthority {
        root: RootGeneration(8),
        ..authority
    };
    let next_root = owner
        .bind_root(
            Position::startpos(),
            next,
            Deadline {
                clock: ClockDomain(authority.epoch),
                at: MonotonicTick(30_000_000_000),
            },
            Instant::now() + WINDOW,
            Arc::new(AtomicBool::new(false)),
        )
        .unwrap_or_else(|_| panic!("same owner accepts its next root"));
    assert!(same_store.matches(next_root.stores()));
    drop(next_root);
    stop_actor(&mut actor);
}
#[test]
fn cancelled_initial_actor_terminal_rejects_foreign_cache_and_returns_same_owner() {
    let mut root = root();
    let same_store = root.stores().read_owner();
    let request = root.prepare_initial_nnuec_review().unwrap();
    let (request, seal) = request.into_parts();
    let model = Arc::clone(root.source.model());
    let mut actor = PreparationActor::spawn(Arc::clone(&model)).unwrap();
    let mut foreign = PreparationActor::spawn(model).unwrap();
    root.cancel();
    actor
        .try_submit(request)
        .unwrap_or_else(|_| panic!("actual canceled request submission"));
    let until = Instant::now() + WINDOW;
    let completion = loop {
        if let Some(completion) = actor.try_complete().unwrap() {
            break completion;
        }
        assert!(Instant::now() < until, "actual preparation terminal");
        std::thread::yield_now();
    };
    assert!(matches!(
        completion.result,
        Err(PreparationError::Cancelled)
    ));
    assert_eq!(
        completion.terminal_ack.status(),
        PreparationStatus::Cancelled
    );
    let ack = completion.terminal_ack;
    let foreign_cache = cache_release(&foreign, &foreign.invalidate_context_cache().unwrap());
    let rejected = match root.retire_after_initial_preparation(
        seal,
        InitialResearcherRelease::Preparation(ack),
        foreign_cache,
    ) {
        Err(rejected) => rejected,
        Ok(_) => panic!("foreign actor cache cannot release original preparation"),
    };
    assert!(same_store.matches(rejected.root.stores()));
    assert_eq!(rejected.root.stores().review_pin_count().unwrap(), 2);
    let InitialPreparingReleaseCustody::Unreleased(InitialResearcherRelease::Preparation(ack)) =
        rejected.release
    else {
        panic!("actual terminal preserved")
    };
    let cache = cache_release(&actor, &actor.invalidate_context_cache().unwrap());
    let retired = rejected
        .root
        .retire_after_initial_preparation(
            rejected.original_review_seal.unwrap(),
            InitialResearcherRelease::Preparation(ack),
            cache,
        )
        .unwrap_or_else(|_| panic!("same actor terminal/cache cleanup"));
    assert!(same_store.matches(retired.retired.owner.stores()));
    assert_eq!(
        retired.retired.owner.stores().review_pin_count().unwrap(),
        0
    );
    assert_eq!(
        retired
            .retired
            .ledger
            .registration_observation()
            .unwrap()
            .spent_native_nodes,
        0
    );
    assert!(matches!(
        retired.release,
        InitialResearcherReleaseReceipt::Preparation(_)
    ));
    stop_actor(&mut foreign);
    stop_actor(&mut actor);
}
