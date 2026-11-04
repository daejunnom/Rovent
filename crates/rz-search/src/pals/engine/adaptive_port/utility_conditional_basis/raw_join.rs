//! Exact join of a validated old raw projection with its separately retained
//! Store-source DTO. This creates no live or known authority. Header/SHA and
//! Rules-line validation belong to the one historical parser; metadata absent
//! from that framing is validated by the parent basis verifier.

use super::*;

/// Join only fields the historical writer actually serialized. In particular,
/// tag 4 cannot attest ContextWdl revision/input, and CPU training declarations,
/// Store IDs, source/epoch/scope/work/consumer metadata remain parent checks.
pub(super) fn validate_raw_join(
    source: &ObservationSource,
    raw: &HistoricalWitnessObservation<'_>,
    before: Option<&ObservationSource>,
    after: Option<&ObservationSource>,
) -> Result<(), BasisError> {
    let mut nodes = 0;
    join(source, raw, before, after, 0, &mut nodes)
}

fn join(
    source: &ObservationSource,
    raw: &HistoricalWitnessObservation<'_>,
    before: Option<&ObservationSource>,
    after: Option<&ObservationSource>,
    depth: u8,
    nodes: &mut usize,
) -> Result<(), BasisError> {
    if depth >= 8 {
        return Err(BasisError::Limit("raw join depth"));
    }
    *nodes = nodes
        .checked_add(1)
        .ok_or(BasisError::Limit("raw join nodes"))?;
    if *nodes > 255 {
        return Err(BasisError::Limit("raw join nodes"));
    }
    let snapshot = source.state.decode()?;
    if source.kind != raw.kind
        || rz_position::contracts::rules_snapshot_semantic_digest(&snapshot) != raw.state_digest
        || source.budget != raw.budget
        || source.condition.as_deref().unwrap_or("") != raw.cpu_condition
        || raw.profiled_scope.is_some()
        || source.checker.is_some()
        || source.external.is_some()
    {
        return Err(BasisError::Mismatch("raw observation header/source"));
    }
    join_line(source.line.as_ref(), raw.line.as_deref())?;
    join_line(source.cpu_pv.as_ref(), raw.cpu_pv.as_deref())?;
    match (&source.value, raw.cpu_identity) {
        (None, None) => {}
        (Some(source), Some(raw))
            if source.semantics == raw.semantics
                && source.weights_sha256.as_deref().unwrap_or("") == raw.weights =>
        {
            source
                .validate()
                .map_err(|_| BasisError::Invalid("CPU value namespace"))?;
        }
        _ => return Err(BasisError::Mismatch("raw CPU value namespace")),
    }
    match (&source.model, raw.model_identity) {
        (None, None) => {}
        (Some(source), Some(raw))
            if source.semantics == raw.semantics
                && source.model == raw.model
                && source.encoding == raw.encoding
                && source.precision == raw.precision
                && source.epoch == raw.epoch => {}
        _ => return Err(BasisError::Mismatch("raw model value namespace")),
    }

    let conditional = matches!(
        source.score,
        Score::ConditionalWdl { .. } | Score::ConditionalRepairWdl { .. }
    );
    if !conditional && (before.is_some() || after.is_some()) {
        return Err(BasisError::Mismatch(
            "unexpected raw observation dependencies",
        ));
    }
    let valid = match (&source.score, &raw.score) {
        (Score::Unknown, HistoricalWitnessRawScore::Unknown) => true,
        (
            Score::Estimate { bits, side },
            HistoricalWitnessRawScore::Estimate { value, perspective },
        ) => *bits == value.to_bits() && color(*side)? == *perspective,
        (
            Score::Cpu { value, side, bound },
            HistoricalWitnessRawScore::Cpu {
                value: actual,
                perspective,
                bound: actual_bound,
            },
        ) => value == actual && bound == actual_bound && color(*side)? == *perspective,
        (
            Score::Wdl { bits, side } | Score::ContextWdl { bits, side, .. },
            HistoricalWitnessRawScore::Wdl {
                win,
                draw,
                loss,
                perspective,
            },
        ) => {
            *bits == [win.to_bits(), draw.to_bits(), loss.to_bits()]
                && color(*side)? == *perspective
        }
        (
            Score::ConditionalWdl {
                bits,
                side,
                before: before_id,
                after: after_id,
                ..
            },
            HistoricalWitnessRawScore::ConditionalWdl {
                expectation,
                perspective,
                repaired,
                counter,
            },
        ) => {
            join_children(
                before, after, *before_id, *after_id, repaired, counter, depth, nodes,
            )?;
            *bits == expectation.to_bits() && color(*side)? == *perspective
        }
        (
            Score::ConditionalRepairWdl {
                bits,
                side,
                before: before_id,
                after: after_id,
                ..
            },
            HistoricalWitnessRawScore::ConditionalRepairWdl {
                expectation,
                perspective,
                before: raw_before,
                after: raw_after,
            },
        ) => {
            join_children(
                before, after, *before_id, *after_id, raw_before, raw_after, depth, nodes,
            )?;
            *bits == expectation.to_bits() && color(*side)? == *perspective
        }
        (Score::Terminal { winner }, HistoricalWitnessRawScore::Terminal { winner: actual }) => {
            winner.map(color).transpose()? == *actual
        }
        _ => false,
    };
    if !valid {
        return Err(BasisError::Mismatch("raw score fields/bits"));
    }
    Ok(())
}

fn join_line(source: Option<&StoredLine>, raw: Option<&[BoardMove]>) -> Result<(), BasisError> {
    match (source, raw) {
        (None, None) => Ok(()),
        (Some(source), Some(raw)) => {
            if source.moves.len() > MAX_LINE || raw.len() > MAX_LINE {
                return Err(BasisError::Limit("raw line plies"));
            }
            if unpack(&source.moves)?.as_slice() == raw {
                Ok(())
            } else {
                Err(BasisError::Mismatch("raw ordered line"))
            }
        }
        _ => Err(BasisError::Mismatch("raw optional line")),
    }
}

#[allow(clippy::too_many_arguments)]
fn join_children(
    before: Option<&ObservationSource>,
    after: Option<&ObservationSource>,
    before_id: ObservationId,
    after_id: ObservationId,
    raw_before: &HistoricalWitnessObservation<'_>,
    raw_after: &HistoricalWitnessObservation<'_>,
    depth: u8,
    nodes: &mut usize,
) -> Result<(), BasisError> {
    let before = before.ok_or(BasisError::Invalid("missing original raw before source"))?;
    let after = after.ok_or(BasisError::Invalid("missing original raw after source"))?;
    if before.id != before_id || after.id != after_id || before.id == after.id {
        return Err(BasisError::Mismatch(
            "raw conditional original observation IDs",
        ));
    }
    // Store's actual model-comparison dependencies are raw ContextWdl leaves.
    // A nested derived DTO cannot acquire missing grandchildren by recursion.
    join(before, raw_before, None, None, depth + 1, nodes)?;
    join(after, raw_after, None, None, depth + 1, nodes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pals::engine::adaptive_port::conditional_evidence::tests::port;
    use crate::pals::engine::adaptive_port::validator_port::{
        BoundedObservationPreimage, ObservationSink,
    };
    use sha2::Sha256;

    struct Fixture {
        source: ObservationSource,
        before: Option<ObservationSource>,
        after: Option<ObservationSource>,
        root: PositionSnapshot,
        actual: Vec<BoardMove>,
        identity: ContractDigest,
        bytes: Vec<u8>,
    }
    impl Fixture {
        fn expectation(&self) -> UtilityWitnessPreimageExpectation<'_> {
            UtilityWitnessPreimageExpectation {
                root: &self.root,
                kind: ActionWitnessKind::Counter,
                line: &self.actual,
                source_question: ContractDigest([13; 32]),
                source_frame_sequence: 7,
                identity: self.identity,
                raw_evidence: ContractDigest(Sha256::digest(&self.bytes).into()),
            }
        }
    }
    fn line(text: &str) -> Vec<BoardMove> {
        text.split_whitespace()
            .map(|text| text.parse().unwrap())
            .collect()
    }
    fn raw_model(
        stores: &mut PalsStores,
        state: StateId,
        identity: &ModelValueIdentity,
        values: [f32; 3],
        input: [u8; 32],
    ) -> ObservationId {
        stores
            .hot_append_observation(Observation {
                state,
                line: None,
                source: 1,
                epoch: 0,
                value_identity: None,
                checker_identity: None,
                checker_work: None,
                external_report: None,
                model_value_identity: Some(identity.clone()),
                model_value_input: Some(input),
                cpu_condition: None,
                cpu_pv: None,
                scope: EvidenceScope::Model {
                    model: 1,
                    encoding: 1,
                    input: 11,
                },
                score: RawScore::ContextWdl {
                    win: values[0],
                    draw: values[1],
                    loss: values[2],
                    perspective: Color::White,
                    context_revision: 11,
                },
                budget: 0,
                kind: ObservationKind::Proposal,
                supersedes: None,
                execution: None,
            })
            .unwrap()
    }
    fn fixture(conditional: bool) -> Fixture {
        // Reuse the authored hot-Store fixture and the original raw writer.
        // This is a historical codec join, not NN/CPU execution or utility
        // producer acceptance; no live witness/basis capability is created.
        let root = Position::startpos().snapshot();
        let mut port = port(Position::startpos());
        let proposal = line("e2e4 e7e5");
        let actual = line("e2e4 c7c5");
        let root_node = port.root;
        let state = port.engine.semantic.nodes[root_node].state;
        let situation = port.engine.semantic.nodes[root_node].situation;
        let focus = port
            .engine
            .semantic
            .stores
            .situations
            .get(situation)
            .unwrap()
            .focus;
        let stored_line = port
            .engine
            .semantic
            .stores
            .hot_append_line(focus, &proposal)
            .unwrap();
        let (score, before, after) = if conditional {
            let before_node = port.replay_node(&proposal).unwrap();
            let after_node = port.replay_node(&actual).unwrap();
            let before_state = port.engine.semantic.nodes[before_node].state;
            let after_state = port.engine.semantic.nodes[after_node].state;
            let identity = port.engine.semantic.model.value_identity().unwrap().clone();
            let before = raw_model(
                &mut port.engine.semantic.stores,
                before_state,
                &identity,
                [0.75, 0.0, 0.25],
                [1; 32],
            );
            let after = raw_model(
                &mut port.engine.semantic.stores,
                after_state,
                &identity,
                [0.25, 0.0, 0.75],
                [2; 32],
            );
            (
                RawScore::ConditionalWdl {
                    expectation: -0.5,
                    perspective: Color::White,
                    repaired: before,
                    counter: after,
                    context_revision: 11,
                },
                Some(before),
                Some(after),
            )
        } else {
            (
                RawScore::Estimate {
                    value: -0.0,
                    perspective: Color::White,
                },
                None,
                None,
            )
        };
        let observation = port
            .engine
            .semantic
            .stores
            .hot_append_observation(Observation {
                state,
                line: Some(stored_line),
                source: 1,
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
                    model: 1,
                    encoding: 1,
                    input: 11,
                },
                score,
                budget: 0,
                kind: ObservationKind::Refutation,
                supersedes: None,
                execution: None,
            })
            .unwrap();
        let source = ObservationSource::capture(&port.engine.semantic.stores, observation).unwrap();
        let before =
            before.map(|id| ObservationSource::capture(&port.engine.semantic.stores, id).unwrap());
        let after =
            after.map(|id| ObservationSource::capture(&port.engine.semantic.stores, id).unwrap());
        let identity = port
            .validator_witness_identity(ActionWitnessKind::Counter, &actual)
            .unwrap();
        let mut bytes = BoundedObservationPreimage::new(MAX_BYTES);
        bytes.write(b"rz-rovent-utility-actual-conditional-witness/1\0");
        bytes.write([13; 32]);
        bytes.write(7_u64.to_le_bytes());
        bytes.write(identity.0);
        port.validator_write_observation_into(&mut bytes, observation, 0)
            .unwrap();
        Fixture {
            source,
            before,
            after,
            root,
            actual,
            identity,
            bytes: bytes.into_bytes().unwrap(),
        }
    }

    #[test]
    fn original_writer_projection_joins_exact_lines_metadata_and_float_bits() {
        let fixture = fixture(false);
        let raw = parse_validated_utility_witness_preimage(&fixture.bytes, fixture.expectation())
            .unwrap();
        validate_raw_join(&fixture.source, &raw, None, None).unwrap();
        let mut changed = fixture.source.clone();
        changed.budget += 1;
        assert!(validate_raw_join(&changed, &raw, None, None).is_err());
        let mut changed = fixture.source.clone();
        changed.line.as_mut().unwrap().moves.reverse();
        assert!(validate_raw_join(&changed, &raw, None, None).is_err());
        let mut changed = fixture.source.clone();
        changed.line = None;
        assert!(validate_raw_join(&changed, &raw, None, None).is_err());
        let mut changed = fixture.source.clone();
        changed.condition = Some("another evaluator".into());
        assert!(validate_raw_join(&changed, &raw, None, None).is_err());
        let mut changed = fixture.source.clone();
        // Float equality would equate these zeros; original wire bits differ.
        changed.score = Score::Estimate {
            bits: 0.0_f32.to_bits(),
            side: 0,
        };
        assert_eq!(
            validate_raw_join(&changed, &raw, None, None),
            Err(BasisError::Mismatch("raw score fields/bits"))
        );
        assert!(validate_raw_join(&fixture.source, &raw, Some(&fixture.source), None).is_err());
    }

    #[test]
    fn original_recursive_writer_joins_exact_dependency_ids_and_model_contents() {
        let fixture = fixture(true);
        let raw = parse_validated_utility_witness_preimage(&fixture.bytes, fixture.expectation())
            .unwrap();
        validate_raw_join(
            &fixture.source,
            &raw,
            fixture.before.as_ref(),
            fixture.after.as_ref(),
        )
        .unwrap();
        assert!(
            validate_raw_join(
                &fixture.source,
                &raw,
                fixture.after.as_ref(),
                fixture.before.as_ref()
            )
            .is_err()
        );
        assert!(validate_raw_join(&fixture.source, &raw, None, fixture.after.as_ref()).is_err());
        let mut changed = fixture.after.as_ref().unwrap().clone();
        changed.model.as_mut().unwrap().epoch[0] ^= 1;
        assert!(
            validate_raw_join(
                &fixture.source,
                &raw,
                fixture.before.as_ref(),
                Some(&changed)
            )
            .is_err()
        );
        let mut changed = fixture.after.as_ref().unwrap().clone();
        let Score::ContextWdl { bits, .. } = &mut changed.score else {
            panic!("original ContextWdl DTO")
        };
        bits[0] ^= 1;
        assert!(
            validate_raw_join(
                &fixture.source,
                &raw,
                fixture.before.as_ref(),
                Some(&changed)
            )
            .is_err()
        );
    }
}
