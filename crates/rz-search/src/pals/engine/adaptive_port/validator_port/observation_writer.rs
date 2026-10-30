//! One byte framing for historical observation hashes and bounded preimages.
//! These writers issue neither an observation nor conditional-proof authority.
use super::*;

pub(in crate::pals::engine::adaptive_port) trait ObservationSink {
    fn write(&mut self, bytes: impl AsRef<[u8]>);
    fn check(&self) -> Result<(), PalsError>;
}

pub(super) struct ObservationShaSink<'a>(pub &'a mut Sha256);
impl ObservationSink for ObservationShaSink<'_> {
    fn write(&mut self, bytes: impl AsRef<[u8]>) {
        sha2::Digest::update(self.0, bytes.as_ref());
    }
    fn check(&self) -> Result<(), PalsError> {
        Ok(())
    }
}

pub(in crate::pals::engine::adaptive_port) struct BoundedObservationPreimage {
    bytes: Vec<u8>,
    maximum: usize,
    failed: bool,
}
impl BoundedObservationPreimage {
    pub(in crate::pals::engine::adaptive_port) fn new(maximum: usize) -> Self {
        Self {
            bytes: Vec::new(),
            maximum,
            failed: false,
        }
    }
    pub(in crate::pals::engine::adaptive_port) fn into_bytes(self) -> Result<Vec<u8>, PalsError> {
        self.check()?;
        Ok(self.bytes)
    }
}
impl ObservationSink for BoundedObservationPreimage {
    fn write(&mut self, bytes: impl AsRef<[u8]>) {
        if self.failed {
            return;
        }
        let bytes = bytes.as_ref();
        if self
            .bytes
            .len()
            .checked_add(bytes.len())
            .is_none_or(|length| length > self.maximum)
            || self.bytes.try_reserve_exact(bytes.len()).is_err()
        {
            self.failed = true;
            return;
        }
        self.bytes.extend_from_slice(bytes);
    }
    fn check(&self) -> Result<(), PalsError> {
        if self.failed {
            Err(PalsError::Capacity)
        } else {
            Ok(())
        }
    }
}

pub(super) fn write_bytes<S: ObservationSink>(sink: &mut S, bytes: &[u8]) {
    sink.write((bytes.len() as u64).to_le_bytes());
    sink.write(bytes);
}
pub(super) fn write_moves<S: ObservationSink>(
    sink: &mut S,
    moves: &[BoardMove],
) -> Result<(), PalsError> {
    sink.check()?;
    sink.write((moves.len() as u64).to_le_bytes());
    for movement in moves {
        sink.write(Move16::pack(*movement)?.bits().to_le_bytes());
        sink.check()?;
    }
    sink.check()
}
pub(super) fn write_profiled_constraint<S: ObservationSink>(
    sink: &mut S,
    constraint: crate::pals::store::ProfiledExternalConstraint,
) {
    use crate::pals::store::ProfiledExternalConstraint as Limit;
    match constraint {
        Limit::Combined {
            depth,
            reported_nodes,
        } => {
            sink.write([0]);
            sink.write(depth.to_le_bytes());
            sink.write(reported_nodes.to_le_bytes());
        }
        Limit::Nodes { reported_nodes } => {
            sink.write([1]);
            sink.write(reported_nodes.to_le_bytes());
        }
        Limit::Depth { depth } => {
            sink.write([2]);
            sink.write(depth.to_le_bytes());
        }
        Limit::MoveTime { milliseconds } => {
            sink.write([3]);
            sink.write(milliseconds.to_le_bytes());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pals::engine::adaptive_port::conditional_evidence::tests::{accepted_raw, port};
    use crate::pals::store::ProfiledExternalConstraint;

    fn expected_bytes(bytes: &mut Vec<u8>, value: &[u8]) {
        bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
        bytes.extend_from_slice(value);
    }

    #[test]
    fn resident_cpu_observation_keeps_original_hash_bytes_and_exact_capacity_boundary() {
        let mut port = port(Position::startpos());
        let line = [
            BoardMove::from_uci("e2e4").unwrap(),
            BoardMove::from_uci("e7e5").unwrap(),
        ];
        // Reuse the existing checked Hot Store fixture. These facts claim no
        // physical CPU execution and cannot construct an action/body cap.
        let id = accepted_raw(&mut port, &line, 21, 2);
        let observation = port.engine.semantic.stores.observations.get(id).unwrap();
        assert!(observation.line.is_none());
        assert!(observation.cpu_pv.is_none());
        assert!(observation.model_value_identity.is_none());
        let identity = observation.value_identity.as_ref().unwrap();
        let RawScore::Cpu {
            value,
            perspective,
            bound,
        } = observation.score
        else {
            panic!("original CPU raw fixture")
        };
        // Independent spelling of the existing raw CPU preimage, including
        // absence tags, string lengths and numeric widths. No codec is used.
        let mut expected = vec![ObservationKind::CpuAnalysis as u8];
        expected.extend_from_slice(
            &rz_position::contracts::rules_snapshot_semantic_digest(
                port.engine
                    .semantic
                    .stores
                    .states
                    .get(observation.state)
                    .unwrap(),
            )
            .0,
        );
        expected.extend_from_slice(&[0, 0]);
        expected.extend_from_slice(&observation.budget.to_le_bytes());
        expected_bytes(
            &mut expected,
            observation
                .cpu_condition
                .as_deref()
                .unwrap_or("")
                .as_bytes(),
        );
        expected.push(1);
        expected_bytes(&mut expected, identity.semantics.as_bytes());
        expected_bytes(
            &mut expected,
            identity.weights_sha256.as_deref().unwrap_or("").as_bytes(),
        );
        expected.push(0);
        expected.extend_from_slice(&[2, u8::from(perspective == Color::Black), bound as u8]);
        expected.extend_from_slice(&value.to_le_bytes());
        let observations = port.engine.semantic.stores.observations.len();
        let mut body = BoundedObservationPreimage::new(expected.len());
        port.validator_write_observation_into(&mut body, id, 0)
            .unwrap();
        let bytes = body.into_bytes().unwrap();
        assert_eq!(bytes, expected);
        let mut raw = Sha256::new();
        port.validator_write_observation(&mut raw, id, 0).unwrap();
        let expected_hash: [u8; 32] = Sha256::digest(&expected).into();
        let actual_hash: [u8; 32] = raw.finalize().into();
        assert_eq!(actual_hash, expected_hash);

        let mut short = BoundedObservationPreimage::new(expected.len() - 1);
        assert!(matches!(
            port.validator_write_observation_into(&mut short, id, 0),
            Err(PalsError::Capacity)
        ));
        assert!(matches!(short.into_bytes(), Err(PalsError::Capacity)));
        assert_eq!(port.engine.semantic.stores.observations.len(), observations);
        assert_eq!(port.counters.role_calls, 0);
        assert_eq!(port.counters.cpu_nodes, 0);
    }

    #[test]
    fn single_limit_tags_keep_old_widths_and_failed_buffer_cannot_resume() {
        let cases: [(ProfiledExternalConstraint, Vec<u8>); 4] = [
            (
                ProfiledExternalConstraint::Combined {
                    depth: 3,
                    reported_nodes: 17,
                },
                [
                    vec![0],
                    3_u16.to_le_bytes().to_vec(),
                    17_u64.to_le_bytes().to_vec(),
                ]
                .concat(),
            ),
            (
                ProfiledExternalConstraint::Nodes { reported_nodes: 19 },
                [vec![1], 19_u64.to_le_bytes().to_vec()].concat(),
            ),
            (
                ProfiledExternalConstraint::Depth { depth: 5 },
                [vec![2], 5_u16.to_le_bytes().to_vec()].concat(),
            ),
            (
                ProfiledExternalConstraint::MoveTime { milliseconds: 23 },
                [vec![3], 23_u64.to_le_bytes().to_vec()].concat(),
            ),
        ];
        for (constraint, expected) in cases {
            let mut body = BoundedObservationPreimage::new(expected.len());
            write_profiled_constraint(&mut body, constraint);
            assert_eq!(body.into_bytes().unwrap(), expected);
        }
        let mut failed = BoundedObservationPreimage::new(1);
        failed.write([1, 2]);
        failed.write([]);
        assert!(matches!(failed.into_bytes(), Err(PalsError::Capacity)));
    }
}
