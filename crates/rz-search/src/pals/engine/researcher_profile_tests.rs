//! Constructor/Rules-only compatibility checks. No native search or NN runs.
use super::*;
use rz_contracts::pals::SearchAuthority;
use rz_contracts::{Digest, GameGeneration, ProcessEpoch, RootGeneration};

fn selected_engine(rovent: bool) -> PalsEngine<LegalOrderRoleMock> {
    let config = PalsConfig {
        line_plies: 2,
        ..PalsConfig::default()
    };
    let cpu = Box::new(
        OwnedCpuChecker::new(
            CpuEngine::new(crate::cpu::CpuConfig {
                tt_entries: 0,
                max_depth: 2,
                quiescence_ply: 0,
                ..crate::cpu::CpuConfig::default()
            })
            .unwrap(),
        )
        .unwrap(),
    );
    let engine = if rovent {
        PalsEngine::new_with_boxed_checker_and_rovent_profile(
            ResearcherSemanticProfile {
                config,
                maximum_root_prefix_plies: 32,
            },
            LegalOrderRoleMock,
            cpu,
            ResolverPolicy::ModelWdlRestricted,
            PostRepairRecheckPolicy::Disabled,
        )
    } else {
        PalsEngine::new_with_boxed_checker_and_policies(
            config,
            LegalOrderRoleMock,
            cpu,
            ResolverPolicy::ModelWdlRestricted,
            PostRepairRecheckPolicy::Disabled,
        )
    }
    .unwrap();
    engine
        .with_selected_review_context(
            ReviewSelectionPolicy::default(),
            ReviewIndexLimits::default(),
        )
        .unwrap()
}

#[test]
fn opt_in_registered_researcher_preserves_long_root_prefix_without_changing_chunk() {
    let prefix: Vec<_> = ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6"]
        .into_iter()
        .map(|text| BoardMove::from_uci(text).unwrap())
        .collect();
    for rovent in [false, true] {
        let mut engine = selected_engine(rovent);
        let position = Position::startpos();
        let authority = SearchAuthority {
            epoch: ProcessEpoch(41),
            game: GameGeneration(0),
            root: RootGeneration(1),
            implementation: Digest([41; 32]),
        };
        let registration = engine.register_adaptive_root(&position, authority).unwrap();
        engine
            .consume_adaptive_root_registration(registration, &position, authority)
            .unwrap();
        let config = engine.semantic_sequence_config();
        let mut node = engine.semantic.intern_hot(&config, position).unwrap();
        for movement in &prefix {
            node = engine
                .semantic
                .connect_hot(&config, node, *movement)
                .unwrap()
                .0;
        }
        let question = RoleQuestion {
            purpose: RoleQueryPurpose::ValueFresh,
            prefix: &prefix,
            proposal: &prefix,
            refutation: None,
            divergences: &[],
        };
        let legal = engine.semantic.nodes[node].position.legal_moves();
        let view = engine.prepare_review_view(node, question, &legal);
        assert_eq!(engine.config.line_plies, 2);
        assert_eq!(engine.rovent_semantic_profile().is_some(), rovent);
        if rovent {
            let view = view.unwrap().unwrap();
            assert_eq!(view.query().prefix, prefix);
            assert!(view.validates(engine.stores()));
            assert_eq!(engine.maximum_root_prefix_plies(), 32);
        } else {
            assert!(matches!(
                view,
                Err(PalsError::Review(ReviewError::ViewCapacity))
            ));
            assert_eq!(engine.maximum_root_prefix_plies(), 2);
        }
    }
}
