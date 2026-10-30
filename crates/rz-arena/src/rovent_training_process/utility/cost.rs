//! Recorded utility learner expense, distinct from route/target utility cost.
//!
//! These are strict data observations and checked arithmetic only. No method
//! opens a checkpoint, verifies its source chain, restores an optimizer, proves
//! an old process closed, or grants a current execution/physical capability.
//! The actual parent must retain and verify every original chain operand.

use crate::ArenaError;
use rz_experiments::pals_data_v3::generation_sha;
use serde::de::{MapAccess, Visitor, value::MapAccessDeserializer};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const ROVENT_UTILITY_CHECKPOINT_COST: &str = "rz-rovent-utility-checkpoint-cost/1";
pub const ROVENT_UTILITY_CHECKPOINT_COST_SCOPE: &str =
    "recorded_cost_only_not_old_process_closure_or_new_window_authority";
pub const MAX_ROVENT_UTILITY_INHERITED_UNKNOWN_TAILS: usize = 262_144;

// Exact JSON objects only. A derived struct's sequence representation must not
// make arrays acceptable where the independent metadata decoder requires maps.
macro_rules! cost_object {
    ($(#[$meta:meta])* pub struct $name:ident {
        $($(#[$field_meta:meta])* pub $field:ident: $ty:ty,)*
    }) => {
        $(#[$meta])*
        #[derive(Clone, Debug, Eq, PartialEq, Serialize)]
        pub struct $name { $($(#[$field_meta])* pub $field: $ty,)* }
        impl<'de> Deserialize<'de> for $name {
            fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
                #[derive(Deserialize)]
                #[serde(deny_unknown_fields)]
                struct Fields { $($(#[$field_meta])* $field: $ty,)* }
                struct Object;
                impl<'input> Visitor<'input> for Object {
                    type Value = Fields;
                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        formatter.write_str("an exact utility checkpoint cost object")
                    }
                    fn visit_map<M: MapAccess<'input>>(self, map: M) -> Result<Fields, M::Error> {
                        Fields::deserialize(MapAccessDeserializer::new(map))
                    }
                }
                let fields = deserializer.deserialize_map(Object)?;
                Ok(Self { $($field: fields.$field,)* })
            }
        }
    };
}

fn invalid(reason: impl Into<String>) -> ArenaError {
    ArenaError::Integrity(format!("Rovent utility recorded cost: {}", reason.into()))
}
fn check(condition: bool, reason: &str) -> Result<(), ArenaError> {
    if condition {
        Ok(())
    } else {
        Err(invalid(reason))
    }
}
fn identifier(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 128
        && value.as_bytes()[0].is_ascii_alphanumeric()
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-'))
}

cost_object! {
/// Exactly the existing trainer's eight nonnegative u64 expense counters.
/// There is no route-risk, teacher score, utility target, or optimizer step
/// counter hidden in this map; those have independent units and contracts.
#[derive(Copy, Default)]
pub struct RoventUtilityRecordedCost {
    pub forward_calls: u64,
    pub backward_calls: u64,
    pub samples: u64,
    pub wall_time_ns: u64,
    pub partial_accumulation_ns: u64,
    pub replay_ns: u64,
    pub masked_groups: u64,
    pub output_bytes: u64,
}
}
impl RoventUtilityRecordedCost {
    /// Checked componentwise addition. A single overflowing counter rejects
    /// the whole observation; no saturation, wrapping, or partial mutation.
    pub fn checked_add(&self, other: &Self) -> Result<Self, ArenaError> {
        fn add(left: u64, right: u64) -> Result<u64, ArenaError> {
            left.checked_add(right)
                .ok_or_else(|| invalid("recorded expense counter overflow"))
        }
        Ok(Self {
            forward_calls: add(self.forward_calls, other.forward_calls)?,
            backward_calls: add(self.backward_calls, other.backward_calls)?,
            samples: add(self.samples, other.samples)?,
            wall_time_ns: add(self.wall_time_ns, other.wall_time_ns)?,
            partial_accumulation_ns: add(
                self.partial_accumulation_ns,
                other.partial_accumulation_ns,
            )?,
            replay_ns: add(self.replay_ns, other.replay_ns)?,
            masked_groups: add(self.masked_groups, other.masked_groups)?,
            output_bytes: add(self.output_bytes, other.output_bytes)?,
        })
    }
}

cost_object! {
/// Raw original relative evidence pin. Validation is lexical and performs no
/// filesystem access, hash verification, or old closure classification.
pub struct RoventUtilityCostEvidence {
    pub path: String,
    pub sha256: String,
    pub bytes: u64,
}
}
impl RoventUtilityCostEvidence {
    pub fn validate(&self) -> Result<(), ArenaError> {
        let path = &self.path;
        let relative = !path.is_empty()
            && path.len() <= 4096
            && !path.starts_with('/')
            && !path.contains('\\')
            && !path.chars().any(char::is_control)
            && path.split('/').all(|part| {
                let lower = part.to_ascii_lowercase();
                !part.is_empty()
                    && part != "."
                    && part != ".."
                    && !lower.starts_with(".env")
                    && !lower.ends_with(".key")
                    && !lower.ends_with(".pem")
                    && ![
                        "credential",
                        "service-account",
                        "service_account",
                        "serviceaccount",
                        "ssh-key",
                        "sshkey",
                        "api-key",
                        "api_key",
                        "apikey",
                        "private_key",
                        "client_secret",
                        "access_token",
                        "refresh_token",
                    ]
                    .iter()
                    .any(|word| lower.contains(word))
            });
        check(
            relative && generation_sha(&self.sha256) && self.bytes > 0,
            "bounded normalized relative nonsecret raw evidence pin required",
        )
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case", deny_unknown_fields)]
pub enum RoventUtilityClosureTail {
    Known {
        delta: RoventUtilityRecordedCost,
        evidence: RoventUtilityCostEvidence,
    },
    Unknown {
        reason: String,
        #[serde(deserialize_with = "required_optional_evidence")]
        evidence: Option<RoventUtilityCostEvidence>,
    },
}
fn required_optional_evidence<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<RoventUtilityCostEvidence>, D::Error> {
    Option::<RoventUtilityCostEvidence>::deserialize(deserializer)
}
impl RoventUtilityClosureTail {
    pub fn validate(&self) -> Result<(), ArenaError> {
        match self {
            Self::Known { evidence, .. } => evidence.validate(),
            Self::Unknown { reason, evidence } => {
                check(
                    identifier(reason),
                    "unknown original tail needs a bounded explicit identifier",
                )?;
                if let Some(evidence) = evidence {
                    evidence.validate()?;
                }
                Ok(())
            }
        }
    }
    pub fn is_unknown(&self) -> bool {
        matches!(self, Self::Unknown { .. })
    }
}

cost_object! {
/// An original unmeasured tail, with immutable publication/window identity.
/// A plain record is not proof that the parent traversed the full source chain.
pub struct RoventUtilityInheritedUnknownTail {
    pub checkpoint_publication_sha256: String,
    pub execution_id: String,
    pub window_id: String,
    pub tail: RoventUtilityClosureTail,
}
}
impl RoventUtilityInheritedUnknownTail {
    fn validate(&self) -> Result<(), ArenaError> {
        check(
            generation_sha(&self.checkpoint_publication_sha256)
                && identifier(&self.execution_id)
                && identifier(&self.window_id),
            "inherited utility tail requires its exact original source identity",
        )?;
        check(
            self.tail.is_unknown(),
            "inherited unknown tail cannot become known zero or another known cost",
        )?;
        self.tail.validate()
    }
    fn key(&self) -> (&str, &str, &str) {
        (
            &self.checkpoint_publication_sha256,
            &self.execution_id,
            &self.window_id,
        )
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RoventUtilityCostStatus {
    Known,
    LowerBoundOnly,
}

cost_object! {
/// New exact sidecar. All operands remain separate to prevent re-adding an
/// ancestor known delta that was already included in the selected snapshot.
pub struct RoventUtilityCheckpointCost {
    pub schema: String,
    pub ancestor_cost: RoventUtilityRecordedCost,
    pub snapshot_cost: RoventUtilityRecordedCost,
    pub post_snapshot_delta: RoventUtilityRecordedCost,
    pub closure_tail: RoventUtilityClosureTail,
    pub inherited_unknown_tails: Vec<RoventUtilityInheritedUnknownTail>,
    pub known_lower_bound: RoventUtilityRecordedCost,
    pub total_status: RoventUtilityCostStatus,
    pub scope: String,
    pub execution_authorized: bool,
}
}

/// A numeric lower bound of work after the chosen snapshot, plus every unknown
/// observation. It does not claim the unknown tail cost is zero or complete.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RoventUtilityKnownResumeDelta {
    pub known_delta: RoventUtilityRecordedCost,
    pub total_status: RoventUtilityCostStatus,
    pub closure_tail: RoventUtilityClosureTail,
    pub inherited_unknown_tails: Vec<RoventUtilityInheritedUnknownTail>,
}

/// Pure source operands for a new fork's cost journal. The actual tensor source
/// comparison, update counts and optimizer reset remain the parent's job.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RoventUtilityForkSourceCostProjection {
    pub ancestor_cost: RoventUtilityRecordedCost,
    pub current_cost: RoventUtilityRecordedCost,
    pub inherited_unknown_tails: Vec<RoventUtilityInheritedUnknownTail>,
    pub total_status: RoventUtilityCostStatus,
}

fn validate_inherited(tails: &[RoventUtilityInheritedUnknownTail]) -> Result<(), ArenaError> {
    check(
        tails.len() <= MAX_ROVENT_UTILITY_INHERITED_UNKNOWN_TAILS,
        "bounded inherited original utility tail list required",
    )?;
    check(
        tails.windows(2).all(|pair| pair[0].key() < pair[1].key()),
        "inherited utility tails must use sorted unique original source keys",
    )?;
    let mut publications = BTreeMap::new();
    let mut windows = BTreeSet::new();
    for tail in tails {
        tail.validate()?;
        check(
            publications
                .insert(
                    tail.checkpoint_publication_sha256.as_str(),
                    (tail.execution_id.as_str(), tail.window_id.as_str()),
                )
                .is_none(),
            "original utility publication cannot be duplicated or relabelled to another window",
        )?;
        check(
            windows.insert((tail.execution_id.as_str(), tail.window_id.as_str())),
            "original utility window tail cannot be repinned under another publication",
        )?;
    }
    Ok(())
}

impl RoventUtilityCheckpointCost {
    pub fn from_recorded_parts(
        ancestor_cost: RoventUtilityRecordedCost,
        snapshot_cost: RoventUtilityRecordedCost,
        post_snapshot_delta: RoventUtilityRecordedCost,
        closure_tail: RoventUtilityClosureTail,
        inherited_unknown_tails: Vec<RoventUtilityInheritedUnknownTail>,
    ) -> Result<Self, ArenaError> {
        closure_tail.validate()?;
        validate_inherited(&inherited_unknown_tails)?;
        let mut known_lower_bound = ancestor_cost
            .checked_add(&snapshot_cost)?
            .checked_add(&post_snapshot_delta)?;
        if let RoventUtilityClosureTail::Known { delta, .. } = &closure_tail {
            known_lower_bound = known_lower_bound.checked_add(delta)?;
        }
        let total_status = if closure_tail.is_unknown() || !inherited_unknown_tails.is_empty() {
            RoventUtilityCostStatus::LowerBoundOnly
        } else {
            RoventUtilityCostStatus::Known
        };
        let result = Self {
            schema: ROVENT_UTILITY_CHECKPOINT_COST.into(),
            ancestor_cost,
            snapshot_cost,
            post_snapshot_delta,
            closure_tail,
            inherited_unknown_tails,
            known_lower_bound,
            total_status,
            scope: ROVENT_UTILITY_CHECKPOINT_COST_SCOPE.into(),
            execution_authorized: false,
        };
        result.validate()?;
        Ok(result)
    }

    pub fn validate(&self) -> Result<(), ArenaError> {
        check(
            self.schema == ROVENT_UTILITY_CHECKPOINT_COST
                && self.scope == ROVENT_UTILITY_CHECKPOINT_COST_SCOPE
                && !self.execution_authorized,
            "separate recorded utility cost scope cannot issue execution authority",
        )?;
        self.closure_tail.validate()?;
        validate_inherited(&self.inherited_unknown_tails)?;
        let mut expected = self
            .ancestor_cost
            .checked_add(&self.snapshot_cost)?
            .checked_add(&self.post_snapshot_delta)?;
        if let RoventUtilityClosureTail::Known { delta, .. } = &self.closure_tail {
            expected = expected.checked_add(delta)?;
        }
        let status = if self.closure_tail.is_unknown() || !self.inherited_unknown_tails.is_empty() {
            RoventUtilityCostStatus::LowerBoundOnly
        } else {
            RoventUtilityCostStatus::Known
        };
        check(
            self.known_lower_bound == expected && self.total_status == status,
            "recorded utility lower bound/status differs; unknown cost is not a known zero",
        )
    }

    /// Return only known work after the selected snapshot, not ancestor or
    /// snapshot totals. Unknown observations travel with the numeric lower bound.
    pub fn known_resume_delta(&self) -> Result<RoventUtilityKnownResumeDelta, ArenaError> {
        self.validate()?;
        let mut known_delta = self.post_snapshot_delta;
        if let RoventUtilityClosureTail::Known { delta, .. } = &self.closure_tail {
            known_delta = known_delta.checked_add(delta)?;
        }
        Ok(RoventUtilityKnownResumeDelta {
            known_delta,
            total_status: self.total_status,
            closure_tail: self.closure_tail.clone(),
            inherited_unknown_tails: self.inherited_unknown_tails.clone(),
        })
    }

    /// Fold the selected source's lower bound into new ancestor expense once.
    /// Preserve all unknowns, adding the current one under the exact externally
    /// pinned source identity. No old known tail is appended or charged again.
    pub fn fork_source_projection(
        &self,
        checkpoint_publication_sha256: &str,
        execution_id: &str,
        window_id: &str,
    ) -> Result<RoventUtilityForkSourceCostProjection, ArenaError> {
        self.validate()?;
        check(
            generation_sha(checkpoint_publication_sha256)
                && identifier(execution_id)
                && identifier(window_id),
            "explicit original fork source cost identity required",
        )?;
        check(
            self.inherited_unknown_tails.iter().all(|tail| {
                tail.checkpoint_publication_sha256 != checkpoint_publication_sha256
                    && (tail.execution_id.as_str(), tail.window_id.as_str())
                        != (execution_id, window_id)
            }),
            "current fork cost source cannot reuse an original inherited publication or window",
        )?;
        let mut inherited_unknown_tails = self.inherited_unknown_tails.clone();
        if self.closure_tail.is_unknown() {
            inherited_unknown_tails.push(RoventUtilityInheritedUnknownTail {
                checkpoint_publication_sha256: checkpoint_publication_sha256.into(),
                execution_id: execution_id.into(),
                window_id: window_id.into(),
                tail: self.closure_tail.clone(),
            });
            inherited_unknown_tails.sort_by(|left, right| left.key().cmp(&right.key()));
        }
        validate_inherited(&inherited_unknown_tails)?;
        Ok(RoventUtilityForkSourceCostProjection {
            ancestor_cost: self.known_lower_bound,
            current_cost: RoventUtilityRecordedCost::default(),
            inherited_unknown_tails,
            total_status: self.total_status,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{Value, json};

    fn cost(number: u64) -> RoventUtilityRecordedCost {
        RoventUtilityRecordedCost {
            forward_calls: number,
            backward_calls: number,
            samples: number,
            wall_time_ns: number,
            partial_accumulation_ns: number,
            replay_ns: number,
            masked_groups: number,
            output_bytes: number,
        }
    }
    fn evidence() -> RoventUtilityCostEvidence {
        RoventUtilityCostEvidence {
            path: "receipts/original-tail.json".into(),
            sha256: "a".repeat(64),
            bytes: 128,
        }
    }
    fn known(number: u64) -> RoventUtilityClosureTail {
        RoventUtilityClosureTail::Known {
            delta: cost(number),
            evidence: evidence(),
        }
    }
    fn unknown() -> RoventUtilityClosureTail {
        RoventUtilityClosureTail::Unknown {
            reason: "original_window_unclosed".into(),
            evidence: None,
        }
    }
    fn inherited(sha: char, execution: &str, window: &str) -> RoventUtilityInheritedUnknownTail {
        RoventUtilityInheritedUnknownTail {
            checkpoint_publication_sha256: sha.to_string().repeat(64),
            execution_id: execution.into(),
            window_id: window.into(),
            tail: unknown(),
        }
    }
    fn recorded() -> RoventUtilityCheckpointCost {
        RoventUtilityCheckpointCost::from_recorded_parts(
            cost(10),
            cost(20),
            cost(3),
            known(4),
            vec![],
        )
        .unwrap()
    }
    #[test]
    fn exact_original_eight_counter_contract_and_checked_known_total() {
        let serialized = serde_json::to_value(cost(1)).unwrap();
        let actual = serialized
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect::<BTreeSet<_>>();
        let expected = crate::rovent_training_process::COST_FIELDS
            .into_iter()
            .collect::<BTreeSet<_>>();
        assert_eq!(actual, expected);
        let record = recorded();
        assert_eq!(record.known_lower_bound, cost(37));
        assert_eq!(record.total_status, RoventUtilityCostStatus::Known);
        assert!(!record.execution_authorized);
        let decoded: RoventUtilityCheckpointCost =
            serde_json::from_slice(&serde_json::to_vec(&record).unwrap()).unwrap();
        assert_eq!(decoded, record);
        decoded.validate().unwrap();
    }
    #[test]
    fn current_and_inherited_unknowns_preserve_lower_bound_without_fabricated_zero() {
        let record = RoventUtilityCheckpointCost::from_recorded_parts(
            cost(10),
            cost(20),
            cost(3),
            unknown(),
            vec![inherited('b', "old-execution", "old-window")],
        )
        .unwrap();
        assert_eq!(record.known_lower_bound, cost(33));
        assert_eq!(record.total_status, RoventUtilityCostStatus::LowerBoundOnly);
        assert!(record.closure_tail.is_unknown());
        assert_eq!(record.inherited_unknown_tails.len(), 1);
        let mut forged = record.clone();
        forged.total_status = RoventUtilityCostStatus::Known;
        assert!(forged.validate().is_err());
        let encoded = serde_json::to_value(record).unwrap();
        assert!(encoded["closure_tail"].get("delta").is_none());
        assert_eq!(encoded["closure_tail"]["status"], "unknown");
    }
    #[test]
    fn forged_lower_bound_scope_and_authority_are_rejected() {
        let mut record = recorded();
        record.known_lower_bound.output_bytes -= 1;
        assert!(record.validate().is_err());
        let mut record = recorded();
        record.scope = "old_process_closed".into();
        assert!(record.validate().is_err());
        let mut record = recorded();
        record.execution_authorized = true;
        assert!(record.validate().is_err());
        let mut record = recorded();
        record.schema = "rz-rovent-original-training-registration/1".into();
        assert!(record.validate().is_err());
    }
    #[test]
    fn any_counter_overflow_rejects_total_and_preserves_original_operands() {
        for key in crate::rovent_training_process::COST_FIELDS {
            let mut operand = serde_json::to_value(cost(0)).unwrap();
            operand[key] = json!(u64::MAX);
            let maximum: RoventUtilityRecordedCost =
                serde_json::from_value(operand.clone()).unwrap();
            assert!(maximum.checked_add(&cost(1)).is_err());
            assert_eq!(serde_json::to_value(maximum).unwrap(), operand);
            assert!(
                RoventUtilityCheckpointCost::from_recorded_parts(
                    maximum,
                    cost(1),
                    cost(0),
                    known(0),
                    vec![]
                )
                .is_err()
            );
        }
    }
    #[test]
    fn inherited_tail_order_duplicates_and_source_relabelling_are_rejected() {
        let a = inherited('a', "execution-a", "window-a");
        let b = inherited('b', "execution-b", "window-b");
        assert!(
            RoventUtilityCheckpointCost::from_recorded_parts(
                cost(0),
                cost(0),
                cost(0),
                known(0),
                vec![b.clone(), a.clone()]
            )
            .is_err()
        );
        for mut tails in [
            vec![a.clone(), a.clone()],
            vec![
                a.clone(),
                inherited('a', "another-execution", "another-window"),
            ],
            vec![a.clone(), inherited('b', "execution-a", "window-a")],
        ] {
            tails.sort_by(|left, right| left.key().cmp(&right.key()));
            assert!(
                RoventUtilityCheckpointCost::from_recorded_parts(
                    cost(0),
                    cost(0),
                    cost(0),
                    known(0),
                    tails
                )
                .is_err()
            );
        }
        let mut changed_kind = a;
        changed_kind.tail = known(0);
        assert!(
            RoventUtilityCheckpointCost::from_recorded_parts(
                cost(0),
                cost(0),
                cost(0),
                known(0),
                vec![changed_kind]
            )
            .is_err()
        );
        RoventUtilityCheckpointCost::from_recorded_parts(
            cost(0),
            cost(0),
            cost(0),
            known(0),
            vec![b],
        )
        .unwrap();
    }
    #[test]
    fn exact_objects_reject_sequences_missing_unknown_boolean_and_float_counters() {
        let serialized = serde_json::to_value(recorded()).unwrap();
        assert!(serde_json::from_value::<RoventUtilityCheckpointCost>(json!([])).is_err());
        assert!(
            serde_json::from_value::<RoventUtilityRecordedCost>(json!([0, 0, 0, 0, 0, 0, 0, 0]))
                .is_err()
        );
        for replacement in [json!(true), json!(1.0), json!(-1)] {
            let mut value = serialized.clone();
            value["snapshot_cost"]["samples"] = replacement;
            assert!(serde_json::from_value::<RoventUtilityCheckpointCost>(value).is_err());
        }
        let mut value = serialized.clone();
        value["snapshot_cost"]["route_wall_ns"] = json!(7);
        assert!(serde_json::from_value::<RoventUtilityCheckpointCost>(value).is_err());
        let mut value = serialized.clone();
        value["snapshot_cost"]
            .as_object_mut()
            .unwrap()
            .remove("samples");
        assert!(serde_json::from_value::<RoventUtilityCheckpointCost>(value).is_err());
        let mut value = serialized;
        value["execution_authorized"] = json!(0);
        assert!(serde_json::from_value::<RoventUtilityCheckpointCost>(value).is_err());
        let duplicate = "{\"forward_calls\":0,\"forward_calls\":0,\"backward_calls\":0,\"samples\":0,\"wall_time_ns\":0,\"partial_accumulation_ns\":0,\"replay_ns\":0,\"masked_groups\":0,\"output_bytes\":0}";
        assert!(serde_json::from_str::<RoventUtilityRecordedCost>(duplicate).is_err());
    }
    #[test]
    fn evidence_and_unknown_tail_must_retain_exact_normalized_explicit_forms() {
        for path in [
            "/outside.json",
            "receipts/../outside.json",
            "receipts//tail.json",
            "./tail.json",
            "receipts\\tail.json",
            "receipts/.env.local",
            "receipts/client_secret.json",
            "receipts/key.pem",
        ] {
            let mut pin = evidence();
            pin.path = path.into();
            assert!(pin.validate().is_err());
        }
        for reason in [
            "",
            "original tail unknown",
            "_starts_with_separator",
            "bad.reason",
        ] {
            assert!(
                RoventUtilityClosureTail::Unknown {
                    reason: reason.into(),
                    evidence: None
                }
                .validate()
                .is_err()
            );
        }
        let mut value = serde_json::to_value(unknown()).unwrap();
        value["delta"] = serde_json::to_value(cost(0)).unwrap();
        assert!(serde_json::from_value::<RoventUtilityClosureTail>(value).is_err());
        assert!(
            serde_json::from_value::<RoventUtilityClosureTail>(
                json!({"status":"known","delta":cost(0)})
            )
            .is_err()
        );
        assert!(
            serde_json::from_value::<RoventUtilityClosureTail>(
                json!({"status":"unknown","evidence":null})
            )
            .is_err()
        );
        assert!(
            serde_json::from_value::<RoventUtilityClosureTail>(
                json!({"status":"unknown","reason":"unclosed"})
            )
            .is_err()
        );
    }
    #[test]
    fn resume_delta_and_fork_projection_do_not_recharge_selected_snapshot() {
        let record = recorded();
        let resume = record.known_resume_delta().unwrap();
        assert_eq!(resume.known_delta, cost(7));
        assert_eq!(
            record
                .snapshot_cost
                .checked_add(&resume.known_delta)
                .unwrap(),
            cost(27)
        );
        assert_eq!(
            record.ancestor_cost.checked_add(&cost(27)).unwrap(),
            record.known_lower_bound
        );
        let fork = record
            .fork_source_projection(&"c".repeat(64), "source-execution", "source-window")
            .unwrap();
        assert_eq!(fork.ancestor_cost, cost(37));
        assert_eq!(fork.current_cost, cost(0));
        assert!(fork.inherited_unknown_tails.is_empty());
        let next = RoventUtilityCheckpointCost::from_recorded_parts(
            fork.ancestor_cost,
            cost(5),
            cost(2),
            known(1),
            fork.inherited_unknown_tails,
        )
        .unwrap();
        assert_eq!(next.known_lower_bound, cost(45));
    }
    #[test]
    fn fork_preserves_each_unknown_and_rejects_reusing_ancestor_identity() {
        let record = RoventUtilityCheckpointCost::from_recorded_parts(
            cost(10),
            cost(20),
            cost(3),
            unknown(),
            vec![inherited('b', "old-execution", "old-window")],
        )
        .unwrap();
        let resume = record.known_resume_delta().unwrap();
        assert_eq!(resume.known_delta, cost(3));
        assert_eq!(resume.total_status, RoventUtilityCostStatus::LowerBoundOnly);
        assert!(resume.closure_tail.is_unknown());
        assert_eq!(resume.inherited_unknown_tails.len(), 1);
        let fork = record
            .fork_source_projection(&"a".repeat(64), "source-execution", "source-window")
            .unwrap();
        assert_eq!(fork.ancestor_cost, cost(33));
        assert_eq!(fork.current_cost, cost(0));
        assert_eq!(fork.inherited_unknown_tails.len(), 2);
        assert!(
            fork.inherited_unknown_tails
                .windows(2)
                .all(|pair| pair[0].key() < pair[1].key())
        );
        assert_eq!(fork.total_status, RoventUtilityCostStatus::LowerBoundOnly);
        assert!(
            record
                .fork_source_projection(&"b".repeat(64), "new-execution", "new-window")
                .is_err()
        );
        assert!(
            record
                .fork_source_projection(&"c".repeat(64), "old-execution", "old-window")
                .is_err()
        );
        let encoded = serde_json::to_value(record).unwrap();
        assert_eq!(encoded["execution_authorized"], Value::Bool(false));
    }
}
