//! Opt-in codec for a reported same-go exact-rank observation.
//!
//! This is not a target, a proof of physical completion, or a substitute for
//! the original native/teacher owner. The caller must obtain `expected` and
//! the legal list from those owners; decoding these bytes cannot create them.
use super::{ensure, generation_sha, validate_generation_moves};
use crate::ManifestError;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

pub const TEACHER_EXACT_RANK_OBSERVATION_DOMAIN: &str = "rz-pals-teacher-exact-rank-observation/1";

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TeacherExactGoIdentity {
    pub input_sha256: String,
    pub view_sha256: String,
    pub rules_state_sha256: String,
    pub rules_history_sha256: String,
    pub input_lock_sha256: String,
    pub evaluation_namespace_sha256: String,
    pub root_restriction_sha256: String,
    pub teacher_binary_sha256: String,
    pub teacher_profile_sha256: String,
    pub process_id: u32,
    pub process_group: u32,
    pub process_start_ticks: u64,
    pub execution_epoch: u64,
    pub clock_epoch: u64,
    pub original_deadline_ns: u64,
    pub root_generation: u64,
    pub collection_owner: u32,
    pub job_sequence: u64,
    pub request_id: u64,
    pub requested_multipv: u16,
    /// The actual original root restriction; None means all legal candidates.
    #[serde(deserialize_with = "deserialize_required_root_moves")]
    pub root_moves: Option<Vec<u16>>,
}

fn deserialize_required_root_moves<'de, D>(deserializer: D) -> Result<Option<Vec<u16>>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    Option::<Vec<u16>>::deserialize(deserializer)
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TeacherExactRankOpinion {
    pub rank: u16,
    pub depth: u16,
    pub bound: String,
    pub pv: Vec<u16>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TeacherExactRankObservation {
    pub schema: String,
    pub policy: String,
    pub identity: TeacherExactGoIdentity,
    pub observation_sequence: u64,
    pub opinions: Vec<TeacherExactRankOpinion>,
    pub final_bestmove: u16,
    pub final_bestmove_matches_rank1: bool,
    /// These are actual same-clock observations, not UTC reconstruction.
    pub go_ns: u64,
    pub bestmove_ns: u64,
    pub ready_ns: u64,
    pub returned_ns: u64,
    pub cancel_sequence: u64,
    pub yield_sequence: u64,
    pub stop_sent: bool,
}

/// Validate a detached observation against the supplied original owner scope.
/// Success is only an integrity result. It does not admit a row or authorize
/// a model target; callers must separately retain and join the real owners.
pub fn validate_teacher_exact_rank_observation(
    observation: &TeacherExactRankObservation,
    expected: &TeacherExactGoIdentity,
    legal_moves: &[u16],
) -> Result<(), ManifestError> {
    let id = &observation.identity;
    let hashes = [
        &id.input_sha256,
        &id.view_sha256,
        &id.rules_state_sha256,
        &id.rules_history_sha256,
        &id.input_lock_sha256,
        &id.evaluation_namespace_sha256,
        &id.root_restriction_sha256,
        &id.teacher_binary_sha256,
        &id.teacher_profile_sha256,
    ];
    ensure(
        observation.schema == TEACHER_EXACT_RANK_OBSERVATION_DOMAIN
            && observation.policy == "completed_exact_rank_set"
            && id == expected
            && hashes.iter().all(|hash| generation_sha(hash))
            && id.process_id > 0
            && id.process_group > 0
            && id.process_start_ticks > 0
            && id.job_sequence > 0
            && id.request_id > 0,
        "exact-rank observation differs from supplied original owner scope",
    )?;
    validate_generation_moves(legal_moves, 256)?;
    ensure(
        !legal_moves.is_empty()
            && legal_moves.iter().collect::<BTreeSet<_>>().len() == legal_moves.len(),
        "exact-rank original legal list must be nonempty and unique",
    )?;
    if let Some(roots) = &id.root_moves {
        validate_generation_moves(roots, 256)?;
        ensure(
            !roots.is_empty()
                && roots.iter().collect::<BTreeSet<_>>().len() == roots.len()
                && roots.iter().all(|movement| legal_moves.contains(movement)),
            "exact-rank original root restriction invalid",
        )?;
    }
    ensure(
        observation.go_ns <= observation.bestmove_ns
            && observation.bestmove_ns <= observation.ready_ns
            && observation.ready_ns <= observation.returned_ns
            && observation.returned_ns < id.original_deadline_ns
            && observation.cancel_sequence == 0
            && observation.yield_sequence == 0
            && !observation.stop_sent
            && observation.observation_sequence > 0,
        "exact-rank original go did not finish naturally before its deadline",
    )?;
    let count = usize::from(id.requested_multipv);
    ensure(
        (1..=4).contains(&count)
            && observation.opinions.len() == count
            && legal_moves.contains(&observation.final_bestmove)
            && id
                .root_moves
                .as_ref()
                .is_none_or(|moves| moves.contains(&observation.final_bestmove)),
        "exact-rank count or final bestmove is invalid",
    )?;
    let depth = observation
        .opinions
        .first()
        .map(|opinion| opinion.depth)
        .unwrap_or(0);
    let mut first_moves = BTreeSet::new();
    for (index, opinion) in observation.opinions.iter().enumerate() {
        validate_generation_moves(&opinion.pv, 256)?;
        let first = opinion
            .pv
            .first()
            .copied()
            .ok_or_else(|| ManifestError::Integrity("exact-rank PV missing".into()))?;
        ensure(
            opinion.rank as usize == index + 1
                && depth > 0
                && opinion.depth == depth
                && opinion.bound == "ExactReported"
                && first_moves.insert(first)
                && legal_moves.contains(&first)
                && id
                    .root_moves
                    .as_ref()
                    .is_none_or(|moves| moves.contains(&first)),
            "exact-rank rank/depth/bound/PV differs from original legal candidates",
        )?;
    }
    ensure(
        observation.final_bestmove_matches_rank1
            == (observation.opinions[0].pv[0] == observation.final_bestmove),
        "exact-rank final bestmove comparison differs",
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    // Codec-only data: these values cannot issue a real owner or target.
    fn fixture() -> (
        TeacherExactRankObservation,
        TeacherExactGoIdentity,
        Vec<u16>,
    ) {
        let id = TeacherExactGoIdentity {
            input_sha256: "a".repeat(64),
            view_sha256: "b".repeat(64),
            rules_state_sha256: "c".repeat(64),
            rules_history_sha256: "d".repeat(64),
            input_lock_sha256: "e".repeat(64),
            evaluation_namespace_sha256: "f".repeat(64),
            root_restriction_sha256: "1".repeat(64),
            teacher_binary_sha256: "2".repeat(64),
            teacher_profile_sha256: "3".repeat(64),
            process_id: 7,
            process_group: 7,
            process_start_ticks: 8,
            execution_epoch: 9,
            clock_epoch: 9,
            original_deadline_ns: 23,
            root_generation: 1,
            collection_owner: 1,
            job_sequence: 2,
            request_id: 3,
            requested_multipv: 2,
            root_moves: Some(vec![64, 128]),
        };
        let observation = TeacherExactRankObservation {
            schema: TEACHER_EXACT_RANK_OBSERVATION_DOMAIN.into(),
            policy: "completed_exact_rank_set".into(),
            identity: id.clone(),
            observation_sequence: 5,
            opinions: vec![
                TeacherExactRankOpinion {
                    rank: 1,
                    depth: 8,
                    bound: "ExactReported".into(),
                    pv: vec![64],
                },
                TeacherExactRankOpinion {
                    rank: 2,
                    depth: 8,
                    bound: "ExactReported".into(),
                    pv: vec![128],
                },
            ],
            final_bestmove: 128,
            final_bestmove_matches_rank1: false,
            go_ns: 10,
            bestmove_ns: 20,
            ready_ns: 21,
            returned_ns: 22,
            cancel_sequence: 0,
            yield_sequence: 0,
            stop_sent: false,
        };
        (observation, id, vec![64, 128])
    }

    #[test]
    fn separate_exact_rank_codec_keeps_changed_best_and_rejects_wrong_scope() {
        let (valid, expected, legal) = fixture();
        assert!(validate_teacher_exact_rank_observation(&valid, &expected, &legal).is_ok());
        let mut invalid = valid.clone();
        invalid.identity.request_id += 1;
        assert!(validate_teacher_exact_rank_observation(&invalid, &expected, &legal).is_err());
        for change in 0..6 {
            let mut invalid = valid.clone();
            match change {
                0 => invalid.opinions[0].bound = "Lower".into(),
                1 => invalid.opinions[1].depth += 1,
                2 => invalid.opinions[1].pv = vec![64],
                3 => invalid.cancel_sequence = 1,
                4 => invalid.returned_ns = invalid.identity.original_deadline_ns,
                _ => invalid.final_bestmove_matches_rank1 = true,
            }
            assert!(validate_teacher_exact_rank_observation(&invalid, &expected, &legal).is_err());
        }
    }

    #[test]
    fn root_moves_wire_key_is_required_even_when_null() {
        let (_, mut identity, _) = fixture();
        identity.root_moves = None;
        let mut wire = serde_json::to_value(identity).unwrap();
        assert!(serde_json::from_value::<TeacherExactGoIdentity>(wire.clone()).is_ok());
        wire.as_object_mut().unwrap().remove("root_moves");
        assert!(serde_json::from_value::<TeacherExactGoIdentity>(wire).is_err());
    }
}
