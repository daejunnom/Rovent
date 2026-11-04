//! Authored historical-data checks over the existing hot Store fixture.
//!
//! The fixture completes and consumes Store CPU observations; it does not run
//! a CPU/NN worker or issue a utility witness, known target, accepted head, or
//! `UtilityConditionalWitnessBasis` capability. Private DTO construction below
//! supplies historical parser input, including deliberately malformed input.

use super::*;
use crate::pals::engine::adaptive_port::conditional_evidence::tests::{
    accepted_raw_source, port, port_with_line_plies,
};
use crate::pals::engine::adaptive_port::validator_port::{
    BoundedObservationPreimage, ObservationSink,
};

type Port = AdaptiveEnginePort<LegalOrderRoleMock>;

const SOURCE_QUESTION: ContractDigest = ContractDigest([13; 32]);
const SOURCE_FRAME: u64 = 7;

struct HistoricalCase {
    basis: Basis,
    root: PositionSnapshot,
    actual: Vec<BoardMove>,
    kind: ActionWitnessKind,
    identity: ContractDigest,
    preimage: Vec<u8>,
}

impl HistoricalCase {
    fn validate(&self, basis: &Basis) -> Result<UtilityWitnessBasisMetadata, BasisError> {
        let bytes = encoded(basis, MAX_BYTES).unwrap();
        validate_utility_witness_basis(
            &bytes,
            UtilityWitnessBasisExpectation {
                witness: UtilityWitnessPreimageExpectation {
                    root: &self.root,
                    kind: self.kind,
                    line: &self.actual,
                    source_question: SOURCE_QUESTION,
                    source_frame_sequence: SOURCE_FRAME,
                    identity: self.identity,
                    raw_evidence: ContractDigest(Sha256::digest(&self.preimage).into()),
                },
                observation: self.basis.observation.id,
                basis_sha256: ContractDigest(Sha256::digest(&bytes).into()),
                preimage: &self.preimage,
            },
        )
    }

    fn copied_basis(&self) -> Basis {
        serde_json::from_slice(&encoded(&self.basis, MAX_BYTES).unwrap()).unwrap()
    }
}

fn line(text: &str) -> Vec<BoardMove> {
    text.split_whitespace()
        .map(|movement| movement.parse().unwrap())
        .collect()
}

fn cpu_endpoint(port: &mut Port, moves: &[BoardMove], score: i32) -> Endpoint {
    let (observation_id, original_consumer) = accepted_raw_source(port, moves, score, 2);
    let stores = &port.engine.semantic.stores;
    let observation = stores.observations.get(observation_id).unwrap();
    let execution = observation.execution.unwrap();
    let original_task = stores.tasks.get(execution).unwrap();
    // Use the original complete TaskRecord serde representation, including its
    // private consumer flags; no synthetic consumed row replaces that record.
    let task: Task = serde_json::from_slice(&serde_json::to_vec(original_task).unwrap()).unwrap();
    assert!(matches!(task.status, TaskStatus::Completed(id) if id == observation_id));
    assert!(
        task.consumers
            .iter()
            .any(|row| { row.consumer == original_consumer && row.consumed && !row.cancelled })
    );
    let EvidenceScope::DepthLimited {
        depth,
        profile,
        condition,
    } = observation.scope
    else {
        panic!("original own CPU scope")
    };
    let base_condition = original_task.key.cpu_condition.clone().unwrap();
    Endpoint::Cpu {
        line: pack(moves).unwrap(),
        snapshot: Snapshot::capture(stores.states.get(observation.state).unwrap()).unwrap(),
        root_value: root_value(score, moves.len()).unwrap(),
        depth,
        plies: moves.len(),
        identity: observation.value_identity.clone().unwrap(),
        base_condition: base_condition.clone(),
        profile,
        condition,
        observation: ObservationSource::capture(stores, observation_id).unwrap(),
        task,
        original_consumer,
        situation: original_consumer.situation,
        namespace: CpuNamespace::Legacy {
            condition: base_condition,
        },
    }
}

fn summary(
    port: &mut Port,
    kind: ActionWitnessKind,
    moves: &[BoardMove],
    value: i32,
    supersedes: Option<ObservationId>,
) -> ObservationId {
    let root = &port.engine.semantic.nodes[port.root];
    let state = root.state;
    let perspective = root.position.side_to_move();
    let focus = port
        .engine
        .semantic
        .stores
        .situations
        .get(root.situation)
        .unwrap()
        .focus;
    let stored_line = port
        .engine
        .semantic
        .stores
        .hot_append_line(focus, moves)
        .unwrap();
    let observation = Observation {
        state,
        line: Some(stored_line),
        source: stable_id(port.engine.search_identity()),
        epoch: 0,
        value_identity: None,
        checker_identity: None,
        checker_work: None,
        external_report: None,
        model_value_identity: None,
        model_value_input: None,
        cpu_condition: None,
        cpu_pv: None,
        scope: EvidenceScope::Model {
            model: stable_id(port.engine.semantic.model.identity()),
            encoding: stable_id("pals-restricted-summary-v1"),
            input: port.engine.semantic.revision,
        },
        score: RawScore::Estimate {
            value: value as f32,
            perspective,
        },
        budget: 0,
        kind: match kind {
            ActionWitnessKind::Counter => ObservationKind::Refutation,
            ActionWitnessKind::Repair => ObservationKind::Repair,
        },
        supersedes,
        execution: None,
    };
    port.engine
        .semantic
        .stores
        .hot_append_observation(observation)
        .unwrap()
}

fn historical_case(
    port: &Port,
    observation: ObservationId,
    kind: ActionWitnessKind,
    actual: &[BoardMove],
    recipe: ComparisonRecipe,
) -> HistoricalCase {
    let identity = port.validator_witness_identity(kind, actual).unwrap();
    let mut writer = BoundedObservationPreimage::new(MAX_BYTES);
    writer.write(b"rz-rovent-utility-actual-conditional-witness/1\0");
    writer.write(SOURCE_QUESTION.0);
    writer.write(SOURCE_FRAME.to_le_bytes());
    writer.write(identity.0);
    port.validator_write_observation_into(&mut writer, observation, 0)
        .unwrap();
    let preimage = writer.into_bytes().unwrap();
    let root = port.engine.semantic.nodes[port.root].position.snapshot();
    HistoricalCase {
        basis: Basis {
            schema: DOMAIN.into(),
            domain: DOMAIN.into(),
            root: Snapshot::capture(&root).unwrap(),
            authority: port.authority.into(),
            root_state: port.root_state_id(),
            kind: kind as u8,
            line: pack(actual).unwrap(),
            source_question: SOURCE_QUESTION.0,
            source_frame_sequence: SOURCE_FRAME,
            identity: identity.0,
            raw_evidence: Sha256::digest(&preimage).into(),
            root_budget: [17; 32],
            original_store_byte_limit: MAX_BYTES as u64,
            observation: ObservationSource::capture(&port.engine.semantic.stores, observation)
                .unwrap(),
            comparison: Comparison::Finite { recipe },
        },
        root,
        actual: actual.to_vec(),
        kind,
        identity,
        preimage,
    }
}

fn counter_and_repair() -> (HistoricalCase, HistoricalCase) {
    let mut port = port(Position::startpos());
    let proposal = line("e2e4 e7e5 g1f3 b8c6");
    let counter = line("e2e4 c7c5 g1f3 b8c6");
    let repaired = line("e2e4 c7c5 d2d4 b8c6");
    let before = cpu_endpoint(&mut port, &proposal, 80);
    let after = cpu_endpoint(&mut port, &counter, -40);
    let repair_after = cpu_endpoint(&mut port, &repaired, 20);
    let root =
        Snapshot::capture(&port.engine.semantic.nodes[port.root].position.snapshot()).unwrap();
    let counter_recipe = ComparisonRecipe {
        root: root.clone(),
        before,
        after: after.clone(),
        superseded: None,
        previous: None,
    };
    let counter_id = summary(&mut port, ActionWitnessKind::Counter, &proposal, -40, None);
    let counter_case = historical_case(
        &port,
        counter_id,
        ActionWitnessKind::Counter,
        &counter,
        counter_recipe.clone(),
    );
    let repair_recipe = ComparisonRecipe {
        root,
        before: after,
        after: repair_after,
        superseded: Some(
            ObservationSource::capture(&port.engine.semantic.stores, counter_id).unwrap(),
        ),
        previous: Some(Box::new(counter_recipe)),
    };
    let repair_id = summary(
        &mut port,
        ActionWitnessKind::Repair,
        &repaired,
        20,
        Some(counter_id),
    );
    let repair_case = historical_case(
        &port,
        repair_id,
        ActionWitnessKind::Repair,
        &repaired,
        repair_recipe,
    );
    (counter_case, repair_case)
}

fn before_mut(basis: &mut Basis) -> &mut Endpoint {
    let Comparison::Finite { recipe } = &mut basis.comparison else {
        panic!("finite comparison fixture")
    };
    &mut recipe.before
}

#[test]
fn stored_completed_consumed_cpu_rows_validate_historical_counter_then_repair() {
    let (counter, repair) = counter_and_repair();
    for case in [&counter, &repair] {
        let decoded = case.validate(&case.basis).unwrap();
        assert_eq!(decoded.kind(), case.kind);
        assert_eq!(decoded.line(), case.actual);
        assert!(!decoded.before().is_rules_terminal());
        assert!(!decoded.after().is_rules_terminal());
        for endpoint in [decoded.before(), decoded.after()] {
            let cpu = endpoint.consumed_cpu().unwrap();
            assert_eq!(
                cpu.original_consumer().generation,
                decoded.authority().root.0
            );
            assert_eq!(cpu.task().question, TaskQuestion::AnalyzePosition);
            assert!(cpu.context_authority().is_none());
        }
    }
    assert!(repair.basis.observation.supersedes == Some(counter.basis.observation.id));
}

#[test]
fn historical_cpu_basis_rejects_namespace_epoch_zero_budget_and_original_condition_changes() {
    let (case, _) = counter_and_repair();
    for mutation in 0..4 {
        let mut changed = case.copied_basis();
        let Endpoint::Cpu {
            namespace,
            task,
            observation,
            base_condition,
            ..
        } = before_mut(&mut changed)
        else {
            panic!("CPU fixture")
        };
        match mutation {
            0 => {
                let CpuNamespace::Legacy { condition } = namespace else {
                    panic!("legacy fixture")
                };
                condition.push_str(";different-namespace");
            }
            1 => task.key.epoch += 1,
            2 => {
                task.key.node_budget = 0;
                observation.budget = 0;
            }
            3 => {
                base_condition.push_str(";changed-original-condition");
                task.key.cpu_condition = Some(base_condition.clone());
                observation.condition = Some(base_condition.clone());
            }
            _ => unreachable!(),
        }
        assert!(case.validate(&changed).is_err(), "mutation {mutation}");
    }
}

#[test]
fn historical_cpu_basis_requires_the_exact_original_consumed_consumer() {
    let (case, _) = counter_and_repair();
    for mutation in 0..5 {
        let mut changed = case.copied_basis();
        let Endpoint::Cpu {
            task,
            original_consumer,
            ..
        } = before_mut(&mut changed)
        else {
            panic!("CPU fixture")
        };
        match mutation {
            0 => task.consumers[0].consumed = false,
            1 => task.consumers[0].cancelled = true,
            2 => original_consumer.id += 100,
            3 => task.consumers.clear(),
            4 => task.consumers.push(task.consumers[0].clone()),
            _ => unreachable!(),
        }
        assert!(
            case.validate(&changed).is_err(),
            "consumer mutation {mutation}"
        );
    }
}

#[test]
fn historical_cpu_basis_rejects_a_cpu_tag_on_an_actual_rules_terminal() {
    // The Store derives its line limit at engine construction. Changing only
    // config afterwards would leave its original four-ply admission intact.
    let mut port = port_with_line_plies(Position::startpos(), 8);
    let proposal = line("e2e4 e7e5 g1f3 b8c6 f1c4");
    let terminal = line("e2e4 f7f6 f1c4 g7g5 d1h5");
    // The negative fixture deliberately stores a CPU-tagged terminal snapshot.
    // Rules created this exact endpoint; real source admission excludes it.
    let before = cpu_endpoint(&mut port, &proposal, -80);
    let after = cpu_endpoint(&mut port, &terminal, 40);
    let endpoint_snapshot = match &after {
        Endpoint::Cpu { snapshot, .. } => snapshot.decode().unwrap(),
        _ => unreachable!(),
    };
    let endpoint_position = Position::from_snapshot(
        &endpoint_snapshot,
        PositionLimits {
            max_history_positions: endpoint_snapshot.known_history_len(),
            ..Default::default()
        },
    )
    .unwrap();
    let legal = endpoint_position.ordered_legal_moves();
    assert!(matches!(
        endpoint_position.play_status_from_view(&legal).unwrap(),
        PlayStatus::Terminal { .. }
    ));
    let recipe = ComparisonRecipe {
        root: Snapshot::capture(&port.engine.semantic.nodes[port.root].position.snapshot())
            .unwrap(),
        before,
        after,
        superseded: None,
        previous: None,
    };
    let observation = summary(&mut port, ActionWitnessKind::Counter, &proposal, -40, None);
    let case = historical_case(
        &port,
        observation,
        ActionWitnessKind::Counter,
        &terminal,
        recipe,
    );
    assert!(matches!(
        case.validate(&case.basis),
        Err(BasisError::Invalid("CPU endpoint is Rules terminal"))
    ));
}

#[test]
fn historical_basis_rejects_original_shared_payload_budget_overrun() {
    let (case, _) = counter_and_repair();
    let mut changed = case.copied_basis();
    changed.original_store_byte_limit = 1;
    assert!(matches!(
        case.validate(&changed),
        Err(BasisError::Limit("original shared payload allowance"))
    ));
}

#[test]
fn conditional_history_admission_uses_actual_backing_and_preserves_original_limit() {
    let mut port = port(Position::startpos());
    let path = line("e2e4 e7e5 g1f3 b8c6");
    let (observation, _) = accepted_raw_source(&mut port, &path, 80, 2);
    let before = port
        .utility_conditional_known_retained_host_bytes()
        .unwrap();
    assert!(before > 0, "the original Store consumer vector is retained");
    let mut original = Vec::with_capacity(4096);
    original.push(1);
    let mut stored_line = Vec::with_capacity(32);
    stored_line.extend_from_slice(&path);
    let source = ConditionalComparisonSource {
        original,
        unavailable: Unavailable::InvalidOriginalBasis,
    };
    let added = source.owned_host_bytes().unwrap()
        + (stored_line.capacity() * std::mem::size_of::<BoardMove>()) as u64;
    port.conditional_witnesses.insert(
        observation,
        (ObservationKind::Refutation, stored_line, Some(source)),
    );
    let known = port
        .utility_conditional_known_retained_host_bytes()
        .unwrap();
    assert_eq!(known, before + added);
    assert!(
        added > 4096,
        "spare Vec capacity and fixed source are charged"
    );
    port.utility_continuations.enable_bounded_payload(known);
    assert!(port.check_utility_conditional_known_backing(0).is_ok());
    assert!(matches!(
        port.check_utility_conditional_known_backing(1),
        Err(PalsError::Capacity)
    ));
    assert!(matches!(
        port.check_utility_conditional_known_backing(u64::MAX),
        Err(PalsError::Capacity)
    ));
    assert_eq!(
        port.utility_continuations.witness_payload_limit() as u64,
        known
    );
    assert_eq!(
        port.utility_conditional_known_retained_host_bytes(),
        Some(known)
    );
}
