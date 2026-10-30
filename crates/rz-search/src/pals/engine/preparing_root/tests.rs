//! Initial quota/owner regressions. These never prepare C, materialize a TT,
//! start an R worker, search, train or execute a P/C neural model.
use super::*;
use crate::contract_time::InstantClock;
use crate::cpu::{CpuConfig, CpuOrderingPolicy};
use crate::cpu_value::CpuTrainingState;
use crate::cpu_value::nnuec::v2::{Checkpoint, Model, RelationParameters, SCHEMA, SEMANTICS};
use crate::cpu_value::nnuec::{
    NNUEC_SCHEMA, NNUEC_SEMANTICS, NnuecCheckpoint, NnuecConfig, NnuecParameters,
};
use crate::cpu_worker::DeferredNnuecCpuPlan;
use crate::pals::engine::adaptive_port::{AdaptiveModelNamespace, AdaptivePortModel};
use crate::pals::rovent_frontier::{
    FrontierError, FrontierGoalStatus, FrontierLimits, FrontierStop,
};
use rz_contracts::rovent::{ExecutionMode, SearchGoal};
use rz_contracts::{
    ClockDomain, Digest as ContractDigest, GameGeneration, MonotonicTick, OwnerId,
    PrecisionProfile, ProcessEpoch, RootGeneration,
};
use std::num::NonZeroU16;

pub(in crate::pals::engine) fn source() -> DeferredNnuecRSource {
    let checkpoint = Checkpoint {
        schema: SCHEMA.into(),
        semantics: SEMANTICS.into(),
        base: NnuecCheckpoint {
            schema: NNUEC_SCHEMA.into(),
            semantics: NNUEC_SEMANTICS.into(),
            training: CpuTrainingState::Untrained,
            output_scale: 1.0,
            config: NnuecConfig {
                board_width: 1,
                line_width: 1,
                context_width: 1,
                local_width: 1,
                hidden_width: 1,
                causal_window: 3,
                max_lines: 8,
                max_plies: 16,
            },
            parameters: NnuecParameters {
                board_weights: vec![0.0; 800],
                board_bias: vec![0.0],
                anchor_weights: vec![0.0; 800],
                anchor_bias: vec![0.0],
                from_weights: vec![0.0; 64],
                to_weights: vec![0.0; 64],
                promotion_weights: vec![0.0; 5],
                ply_weights: vec![0.0; 16],
                token_bias: vec![0.0],
                causal_weights: vec![0.0; 3],
                causal_bias: vec![0.0],
                context_projection: vec![0.0],
                context_bias: vec![0.0],
                local_projection: vec![0.0],
                local_bias: vec![0.0],
                hidden_weights: vec![0.0; 4],
                hidden_bias: vec![0.0],
                output_weights: vec![0.0; 11],
                output_bias: vec![0.0; 11],
            },
        },
        parameters: RelationParameters {
            kind_embeddings: vec![0.0; 4],
            pair_parent: vec![0.0; 2],
            pair_child: vec![0.0; 2],
            pair_bias: vec![0.0; 2],
            pair_projection: vec![0.0],
        },
    };
    let plan = DeferredNnuecCpuPlan::new(
        CpuConfig {
            tt_entries: 16,
            max_depth: 2,
            quiescence_ply: 0,
            ..CpuConfig::default()
        },
        CpuOrderingPolicy::LegacyMvvLvaV1,
        128 * 1024,
        false,
        3,
        None,
    )
    .unwrap();
    DeferredNnuecRSource::new(Arc::new(Model::new(checkpoint).unwrap()), plan)
}
pub(in crate::pals::engine) fn owner() -> ResearcherPreparingOwner<AdaptivePortModel> {
    let value = LegalOrderRoleMock.value_identity().unwrap().clone();
    let namespace = AdaptiveModelNamespace::new(
        value.clone(),
        ContractDigest([5; 32]),
        ContractDigest([6; 32]),
        PrecisionProfile::Fp32,
        1,
    )
    .unwrap();
    let model =
        AdaptivePortModel::new_with_namespace(value.model.clone(), Some(value), namespace).unwrap();
    ResearcherPreparingOwner::new_selected(
        ResearcherSemanticProfile {
            config: PalsConfig {
                line_plies: 2,
                max_nodes: 257,
                max_records: 32,
                max_role_calls: 32,
                beam_width: 1,
                cpu_nodes_per_task: 256,
            },
            maximum_root_prefix_plies: 16,
        },
        model,
        ResolverPolicy::ModelWdlRestricted,
        PostRepairRecheckPolicy::Disabled,
        ReviewSelectionPolicy::default(),
        ReviewIndexLimits::default(),
        source(),
    )
    .unwrap()
}

#[test]
fn initial_zero_and_one_node_roots_return_the_same_preparing_owner_without_c_or_r() {
    for nodes in [0, 1] {
        let owner = owner();
        let store = owner.stores().read_owner();
        let authority = SearchAuthority {
            epoch: ProcessEpoch(71),
            game: GameGeneration(0),
            root: RootGeneration(7),
            implementation: ContractDigest([71; 32]),
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
            .map_err(|(_, error)| error)
            .unwrap();
        let next = Arc::new(std::sync::atomic::AtomicU64::new(1));
        let issuer: adaptive_port::AdaptiveRulesOwnerIssuer =
            Arc::new(move || Ok(OwnerId(next.fetch_add(1, Ordering::Relaxed))));
        root.attach_original_controls(
            ResearcherRootStartup {
                mode: ExecutionMode::CpuPcr,
                root_budget: ContractDigest([72; 32]),
                goal: SearchGoal {
                    depth_plies: NonZeroU16::new(5),
                    mate_moves: None,
                },
                nodes: Some(nodes),
                selective_policy: ContractDigest([73; 32]),
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
                cleanup_deadline: hard + Duration::from_secs(1),
            },
            issuer,
            None,
        )
        .unwrap();
        assert!(matches!(
            root.prepare_original_root_rules(),
            Err(ResearcherPreparingError::Frontier(
                FrontierError::NodeBudget
            ))
        ));
        assert!(root.initial_review_source().is_err());
        assert!(root.initial_view.is_none());
        assert_eq!(root.stores().review_pin_count().unwrap(), 1);
        let retired = root
            .retire_before_materialization()
            .map_err(|(_, error)| error)
            .unwrap();
        assert!(store.matches(retired.owner.stores()));
        assert_eq!(retired.owner.stores().review_pin_count().unwrap(), 0);
        assert_eq!(retired.owner.last_authority(), Some(authority));
        let receipt = retired.frontier.unwrap();
        assert_eq!(
            receipt.snapshot.goal,
            FrontierGoalStatus::Stopped(FrontierStop::WorkBudget)
        );
        assert_eq!(receipt.snapshot.work.total_node_visits(), Some(nodes));
        assert_eq!(receipt.snapshot.work.checked_transitions, 0);
        assert_eq!(
            retired
                .ledger
                .registration_observation()
                .unwrap()
                .active_jobs,
            0
        );
        let next = SearchAuthority {
            root: RootGeneration(8),
            ..authority
        };
        let again = retired
            .owner
            .bind_root(
                Position::startpos(),
                next,
                deadline,
                hard,
                Arc::new(AtomicBool::new(false)),
            )
            .map_err(|(_, error)| error)
            .unwrap();
        assert!(store.matches(again.stores()));
        assert_eq!(again.authority(), next);
    }
}

#[test]
fn terminal_initial_root_does_not_prepare_an_unneeded_c_view() {
    let owner = owner();
    let authority = SearchAuthority {
        epoch: ProcessEpoch(81),
        game: GameGeneration(0),
        root: RootGeneration(9),
        implementation: ContractDigest([81; 32]),
    };
    let clock = InstantClock::new(ClockDomain(authority.epoch), Instant::now());
    let deadline = Deadline {
        clock: clock.domain(),
        at: MonotonicTick(20_000_000_000),
    };
    let hard = clock.instant_at(deadline.at).unwrap();
    let terminal = Position::from_fen("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1").unwrap();
    let mut root = owner
        .bind_root(
            terminal,
            authority,
            deadline,
            hard,
            Arc::new(AtomicBool::new(false)),
        )
        .map_err(|(_, error)| error)
        .unwrap();
    let next = Arc::new(std::sync::atomic::AtomicU64::new(1));
    let issuer: adaptive_port::AdaptiveRulesOwnerIssuer =
        Arc::new(move || Ok(OwnerId(next.fetch_add(1, Ordering::Relaxed))));
    root.attach_original_controls(
        ResearcherRootStartup {
            mode: ExecutionMode::CpuPcr,
            root_budget: ContractDigest([82; 32]),
            goal: SearchGoal::default(),
            nodes: Some(8),
            selective_policy: ContractDigest([83; 32]),
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
            cleanup_deadline: hard + Duration::from_secs(1),
        },
        issuer,
        None,
    )
    .unwrap();
    root.prepare_original_root_rules().unwrap();
    let before = root.original_frontier_receipt().unwrap().snapshot.work;
    assert!(root.initial_review_source().is_err());
    assert!(root.initial_view.is_none());
    assert_eq!(root.stores().review_pin_count().unwrap(), 1);
    assert_eq!(
        root.original_frontier_receipt().unwrap().snapshot.work,
        before
    );
    assert_eq!(
        root.original_ledger()
            .unwrap()
            .registration_observation()
            .unwrap()
            .active_jobs,
        0
    );
}
