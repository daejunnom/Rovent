//! Actual /3 root admission and archive worker lifecycle, run inside the
//! existing transfer fixture's original child process and owned directories.
//! No model inference, product session or GPU completion is claimed here.

use super::*;
use crate::bridge::BuildSearchSettings;
use crate::engine::ProcessClock;
use crate::engine::driver_start_fixture::{FixtureStart, issue_start};
use crate::parser::GoLimits;
use crate::rovent_root_admission::RoventRootAdmissionPolicy;
use crate::rovent_root_admission::source_seal::PreparedSourceV3;
use rz_contracts::{Digest, GameGeneration, PrecisionProfile, RootGeneration};
use rz_experiments::{RoventInputLockV3, RoventOriginalInputLock};
use rz_position::Position;
use rz_search::pals::engine::adaptive_port::AdaptiveModelNamespace;
use rz_search::pals::engine::{LegalOrderRoleMock, PalsSemanticConfig, RoleModel};
use rz_search::pals::review_context::{ReviewIndexLimits, ReviewSelectionPolicy};
use rz_search::time::TimeBudgetConfig;
use std::num::NonZeroU16;
use std::sync::Arc;
use std::time::Duration;

fn pc_owner(authority: SearchAuthority, until: Instant) -> Box<PcSemanticOwner<AdaptivePortModel>> {
    let value = LegalOrderRoleMock.value_identity().unwrap().clone();
    let namespace = AdaptiveModelNamespace::new(
        value.clone(),
        Digest([72; 32]),
        Digest([73; 32]),
        PrecisionProfile::Fp32,
        1,
    )
    .unwrap();
    let model =
        AdaptivePortModel::new_with_namespace(value.model.clone(), Some(value), namespace).unwrap();
    let mut owner = Box::new(
        PcSemanticOwner::new_selected(
            PalsSemanticConfig {
                beam_width: 1,
                line_plies: 2,
                max_nodes: 257,
                max_records: 128,
                max_role_calls: 128,
            },
            model,
            ReviewSelectionPolicy::default(),
            ReviewIndexLimits::default(),
        )
        .unwrap(),
    );
    owner
        .try_adaptive_new_game(authority, until, &AtomicBool::new(false))
        .unwrap();
    owner
}

fn admission(
    lock: Arc<RoventInputLockV3>,
    clock: &ProcessClock,
    implementation: Digest,
    game: u64,
    root: u64,
) -> Arc<RoventRootAdmissionV3> {
    let position = Position::startpos();
    let authority = SearchAuthority {
        epoch: clock.epoch(),
        game: GameGeneration(game),
        root: RootGeneration(root),
        implementation,
    };
    let go = GoLimits {
        nodes: Some(7),
        depth_plies: NonZeroU16::new(2),
        ..GoLimits::default()
    };
    let settings = BuildSearchSettings {
        time_config: TimeBudgetConfig::default(),
        max_simulations: 4096,
        untimed_limit: Duration::from_secs(90),
    };
    let context = match issue_start(
        &position,
        go,
        authority,
        clock,
        clock.runtime_clock().origin(),
        settings,
        Duration::from_secs(10),
    )
    .unwrap()
    {
        FixtureStart::Started(context) => context,
        FixtureStart::RulesOnly(_) => panic!("start position unexpectedly terminal"),
    };
    let owner = RoventOriginalInputLock::from_v3(lock).unwrap();
    Arc::new(
        PreparedSourceV3::prepare(
            &owner,
            RoventRootAdmissionPolicy::DynamicGo,
            &position,
            &context,
            clock,
        )
        .unwrap()
        .try_into_live()
        .unwrap(),
    )
}

pub(crate) fn run_original_v3_lifecycle(
    lock: Arc<RoventInputLockV3>,
    clock: &ProcessClock,
    implementation: Digest,
    shared: Arc<SharedRoventStorageRegistration>,
    empty_close: bool,
    new_game_close: bool,
) {
    let original = admission(Arc::clone(&lock), clock, implementation, 1, 1);
    let scope = original.authority();
    let until = original.original_context().shutdown_deadline;
    let signal = CompletionSignal::default();
    let owner = pc_owner(scope, until);
    let original_ptr = std::ptr::from_ref(owner.as_ref());
    let original_store = owner.stores().read_owner();
    let mut lifecycle = SessionStorageLifecycle::new(Arc::clone(&shared));
    let returned = lifecycle
        .prepare_v3(
            StorageSemanticOwner::Pc(owner),
            Arc::clone(&original),
            true,
            &Position::startpos(),
            &signal,
        )
        .unwrap_or_else(|failure| panic!("V3 archive root prepare: {:?}", failure.failure));
    let StorageSemanticOwner::Pc(mut returned) = returned else {
        panic!("wrong returned owner")
    };
    assert_eq!(std::ptr::from_ref(returned.as_ref()), original_ptr);
    assert!(original_store.matches(returned.stores()));
    assert!(
        lifecycle
            .core
            .setup_receipt
            .as_ref()
            .unwrap()
            .configuration_complete
    );
    assert!(lifecycle.core.phase_joins.back().unwrap().1.worker_joined);
    assert!(!lifecycle.core.finish_completed);
    let mut observation = lifecycle.observation();
    observation.require_after_root(scope);
    assert!(observation.complete);
    let empty_owner = lifecycle
        .core
        .empty_game_phases
        .back()
        .unwrap()
        .owner
        .owner_id;
    if !empty_close || new_game_close {
        let root = returned
            .register_adaptive_root(&Position::startpos(), scope)
            .unwrap();
        returned.abandon_unconsumed_adaptive_root(scope).unwrap();
        drop(root);
    }
    let (final_scope, final_owner, final_until, returned) = if new_game_close {
        let next = admission(Arc::clone(&lock), clock, implementation, 2, 2);
        let next_scope = next.authority();
        let next_until = next.original_context().shutdown_deadline;
        let returned_after_new_game = lifecycle
            .prepare_v3(
                StorageSemanticOwner::Pc(returned),
                next,
                true,
                &Position::startpos(),
                &signal,
            )
            .unwrap_or_else(|failure| panic!("V3 archive NewGame: {:?}", failure.failure));
        let StorageSemanticOwner::Pc(next_owner) = returned_after_new_game else {
            panic!("wrong returned owner after NewGame")
        };
        assert!(!original_store.matches(next_owner.stores()));
        let phase = lifecycle.core.empty_game_phases.back().unwrap();
        assert!(matches!(phase.kind, EmptyGamePhaseKind::NewGame));
        assert_eq!(phase.requested_scope, next_scope);
        assert_ne!(phase.owner.owner_id, empty_owner);
        (next_scope, phase.owner.owner_id, next_until, next_owner)
    } else {
        (scope, empty_owner, until, returned)
    };
    lifecycle
        .finish(Some(StorageSemanticOwner::Pc(returned)), final_until, &signal)
        .unwrap_or_else(|failure| panic!("V3 archive final worker: {:?}", failure.failure));
    let mut observation = lifecycle.observation();
    observation.require_after_finish();
    assert!(observation.complete);
    if empty_close {
        let receipt = lifecycle.core.empty_close_receipt.as_ref().unwrap();
        assert_eq!(receipt.scope, final_scope);
        let closed = receipt.archive_owner.unwrap();
        assert_eq!(closed.owner_id, final_owner);
        assert!(closed.complete && closed.admission_closed);
        assert!(!lifecycle.core.finish_completed);
    } else {
        assert!(lifecycle.core.finish_completed);
        assert!(lifecycle.core.empty_close_receipt.is_none());
    }
    assert!(lifecycle.core.shutdown.as_ref().unwrap().worker_joined);

    // Equal lock content in a different allocation is not the admitted owner.
    // Rejection occurs before starting any worker and retains this exact PC.
    let different = admission(Arc::new(lock.as_ref().clone()), clock, implementation, 1, 3);
    let owner = pc_owner(different.authority(), until);
    let rejected_ptr = std::ptr::from_ref(owner.as_ref());
    let rejected_store = owner.stores().read_owner();
    let mut rejected = SessionStorageLifecycle::new(shared);
    let failure = rejected
        .prepare_v3(
            StorageSemanticOwner::Pc(owner),
            different,
            true,
            &Position::startpos(),
            &signal,
        )
        .err()
        .expect("different original lock must be rejected");
    assert!(failure.owner.is_none());
    assert_eq!(failure.failure.code, "RoventStorageOriginalAdmission");
    assert!(rejected.core.worker.is_none());
    // adopt() moves the exact PC owner into its Idle wrapper before the
    // admission check; rejection retains that wrapper, not core.original.
    assert!(rejected.core.original.is_none());
    let retained = rejected
        .core
        .idle
        .take()
        .expect("original idle owner retained")
        .into_pc()
        .unwrap_or_else(|_| panic!("original idle owner had a different kind"));
    assert_eq!(std::ptr::from_ref(retained.as_ref()), rejected_ptr);
    assert!(rejected_store.matches(retained.stores()));
    assert!(rejected.core.first_failure.is_some());
}
