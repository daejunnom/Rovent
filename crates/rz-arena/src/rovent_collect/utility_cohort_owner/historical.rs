//! Detached verification of the original independently pinned cohort files.
//! No Store, native ticket, LoadedSource, Bound or execution capability is
//! reconstructed from these bytes. Numeric inputs are checked without forward.
use super::*;
use crate::rovent_collect::historical_scope::{
    RoventNativeOriginEvidence, RoventRootAdmissionEvidence,
};
use rz_contracts::pals::Move16;
use rz_experiments::rovent_utility_cohort::{
    RoventUtilityCohortRecipient, RoventUtilityCohortUsage, RoventUtilityComparisonCohort,
};
use rz_experiments::{RoventInputLock, RoventPurpose};
use rz_search::cpu_value::nnuec::{rules, v2};
use rz_uci::pals_adaptive::{UtilityCohortFrozenDecision, UtilityCohortSourcePins};
use serde::Deserialize;
use serde::de::DeserializeOwned;
use sha2::Sha256;
use std::collections::{BTreeMap, BTreeSet};

const SOURCE_DOMAIN: &str = "rz-rovent-utility-cohort-collection-source/1";
const TRIAL_DOMAIN: &str = "rz-rovent-utility-cohort-trial/1";
const MATERIALIZATION_DOMAIN: &str = "rz-rovent-utility-cohort-materialization/1";
const PAIR_DOMAIN: &str = "rz-rovent-utility-cohort-pair/1";

#[cfg(test)]
mod tests;

fn error(message: impl std::fmt::Display) -> ArenaError {
    invalid(format!("utility cohort historical: {message}"))
}
fn ensure(condition: bool, message: &str) -> Result<(), ArenaError> {
    if condition {
        Ok(())
    } else {
        Err(error(message))
    }
}
fn string<'a>(value: &'a Value, key: &str) -> Result<&'a str, ArenaError> {
    value
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| error(format!("missing string {key}")))
}
fn number(value: &Value, key: &str) -> Result<u64, ArenaError> {
    value
        .get(key)
        .and_then(Value::as_u64)
        .ok_or_else(|| error(format!("missing unsigned {key}")))
}
fn array<'a>(value: &'a Value, key: &str, maximum: usize) -> Result<&'a [Value], ArenaError> {
    let rows = value
        .get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| error(format!("missing array {key}")))?;
    ensure(rows.len() <= maximum, "bounded array")?;
    Ok(rows)
}
fn fields(value: &Value, keys: &[&str]) -> Result<(), ArenaError> {
    let body = value.as_object().ok_or_else(|| error("object required"))?;
    ensure(
        body.len() == keys.len() && keys.iter().all(|key| body.contains_key(*key)),
        "closed original object fields",
    )
}
fn decode<T: DeserializeOwned>(raw: &[u8]) -> Result<T, ArenaError> {
    rz_experiments::decode_json(std::str::from_utf8(raw).map_err(error)?)
        .map_err(ArenaError::Manifest)
}
fn sha(raw: &[u8]) -> String {
    format!("{:x}", Sha256::digest(raw))
}
fn digest_bytes(domain: &str, raw: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(domain.as_bytes());
    h.update([0]);
    h.update((raw.len() as u64).to_le_bytes());
    h.update(raw);
    format!("{:x}", h.finalize())
}
fn digest_value(domain: &str, value: &Value) -> Result<String, ArenaError> {
    let mut value = value.clone();
    value.sort_all_objects();
    Ok(digest_bytes(
        domain,
        &serde_json::to_vec(&value).map_err(error)?,
    ))
}
fn same<A: serde::Serialize, B: serde::Serialize>(a: &A, b: &B) -> Result<bool, ArenaError> {
    Ok(serde_json::to_value(a).map_err(error)? == serde_json::to_value(b).map_err(error)?)
}
fn unhex(value: &str) -> Result<Vec<u8>, ArenaError> {
    ensure(
        value.len().is_multiple_of(2)
            && value
                .bytes()
                .all(|v| v.is_ascii_digit() || (b'a'..=b'f').contains(&v)),
        "canonical hexadecimal bytes",
    )?;
    value
        .as_bytes()
        .chunks_exact(2)
        .map(|v| u8::from_str_radix(std::str::from_utf8(v).map_err(error)?, 16).map_err(error))
        .collect()
}
fn packed(bits: u16) -> Result<rz_position::BoardMove, ArenaError> {
    let shared = Move16::try_from_bits(bits)
        .and_then(Move16::decode)
        .map_err(error)?;
    rz_position::BoardMove::try_from(shared).map_err(error)
}
fn numerical_move(
    value: &rz_search::cpu_value::nnuec::ReferenceMove,
) -> Result<rz_position::BoardMove, ArenaError> {
    use rz_position::{BoardMove, PieceKind, Square};
    let promotion = match value.promotion {
        0 => None,
        1 => Some(PieceKind::Queen),
        2 => Some(PieceKind::Rook),
        3 => Some(PieceKind::Bishop),
        4 => Some(PieceKind::Knight),
        _ => return Err(error("numerical promotion token")),
    };
    BoardMove::new(
        Square::new(value.from_square).map_err(error)?,
        Square::new(value.to_square).map_err(error)?,
        promotion,
    )
    .map_err(error)
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RulesReplay {
    origin: String,
    completeness: String,
    start_fen: String,
    moves: Vec<u16>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RulesSnapshot {
    rules_version: String,
    variant: String,
    semantic_sha256: String,
    fen: String,
    history_origin: String,
    history_completeness: String,
    known_history: Vec<String>,
    observer_revision: u64,
    uci_replay: RulesReplay,
}
impl RulesSnapshot {
    fn restore(&self) -> Result<Position, ArenaError> {
        use rz_position::{
            HistoryCompleteness as C, HistoryOrigin as O, PositionLimits, PositionSnapshot,
            UciReplay,
        };
        ensure(
            self.uci_replay.moves.len() <= 16384 && self.known_history.len() <= 16385,
            "original Rules history capacity",
        )?;
        let origin = match self.uci_replay.origin.as_str() {
            "start_position" => O::StartPosition,
            "fen" => O::Fen,
            _ => return Err(error("original Rules origin")),
        };
        let completeness = match self.uci_replay.completeness.as_str() {
            "complete" => C::Complete,
            "unknown_prefix" => C::UnknownPrefix,
            _ => return Err(error("original Rules completeness")),
        };
        let replay = UciReplay {
            origin,
            completeness,
            start_fen: self.uci_replay.start_fen.clone(),
            moves: self
                .uci_replay
                .moves
                .iter()
                .copied()
                .map(packed)
                .collect::<Result<_, _>>()?,
        };
        let limits = PositionLimits {
            max_history_positions: 16385,
            ..Default::default()
        };
        let snapshot = PositionSnapshot::from_uci_replay(&replay, self.observer_revision, limits)
            .map_err(error)?;
        ensure(
            snapshot.rules_version() == self.rules_version
                && snapshot.variant() == self.variant
                && snapshot.to_fen() == self.fen
                && format!("{:?}", snapshot.history_origin()) == self.history_origin
                && format!("{:?}", snapshot.history_completeness()) == self.history_completeness
                && snapshot
                    .known_history()
                    .map(|v| v.to_fen())
                    .collect::<Vec<_>>()
                    == self.known_history,
            "original Rules snapshot metadata",
        )?;
        ensure(
            hex(rz_position::contracts::rules_snapshot_semantic_digest(
                &snapshot,
            )) == self.semantic_sha256,
            "original exact Rules semantic digest",
        )?;
        Position::from_snapshot(&snapshot, limits).map_err(error)
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RecordMetadata {
    anchor: u8,
    parent: Option<u16>,
    supersedes: Option<u16>,
    source_state: u64,
    source_revision: u64,
    source_line_offset: u16,
    original_first_move: Option<u16>,
}

pub(in crate::rovent_collect) struct CheckedInput {
    pub(in crate::rovent_collect) reference_json: String,
    pub(in crate::rovent_collect) input_sha256: String,
    pub(in crate::rovent_collect) target_sha256: String,
    pub(in crate::rovent_collect) submission_target_sha256: String,
    pub(in crate::rovent_collect) context_sha256: String,
}

fn boolean(value: &Value, key: &str) -> Result<bool, ArenaError> {
    value
        .get(key)
        .and_then(Value::as_bool)
        .ok_or_else(|| error(format!("missing boolean {key}")))
}
fn hash_field<'a>(value: &'a Value, key: &str) -> Result<&'a str, ArenaError> {
    let text = string(value, key)?;
    ensure(generation_sha(text), "original SHA256 syntax")?;
    Ok(text)
}
fn hash32(text: &str) -> Result<rz_contracts::Digest, ArenaError> {
    ensure(generation_sha(text), "original SHA256 syntax")?;
    Ok(rz_contracts::Digest(
        unhex(text)?
            .try_into()
            .map_err(|_| error("SHA256 length"))?,
    ))
}
fn authority(root: &RoventRootAdmissionEvidence) -> Value {
    serde_json::json!({"process_epoch":root.process_epoch,"game_generation":root.game_generation,
        "root_generation":root.root_generation,"implementation_sha256":root.implementation_sha256})
}
fn checked_authority(value: &Value) -> Result<(), ArenaError> {
    fields(
        value,
        &[
            "process_epoch",
            "game_generation",
            "root_generation",
            "implementation_sha256",
        ],
    )?;
    for key in ["process_epoch", "game_generation", "root_generation"] {
        ensure(
            number(value, key)? > 0,
            "original authority epoch/generation",
        )?;
    }
    hash_field(value, "implementation_sha256")?;
    Ok(())
}
pub(in crate::rovent_collect) fn checked_view(
    value: &Value,
    original: &Value,
) -> Result<(), ArenaError> {
    fields(
        value,
        &[
            "authority",
            "sequence",
            "context_policy_sha256",
            "view_digest",
            "root_state",
            "actual_tokens",
            "selected_records",
            "anchors",
            "length_bucket",
        ],
    )?;
    ensure(
        value["authority"] == *original && number(value, "sequence")? > 0,
        "original view authority/sequence",
    )?;
    for key in ["context_policy_sha256", "view_digest"] {
        hash_field(value, key)?;
    }
    fields(
        &value["root_state"],
        &["owner", "revision", "semantic_sha256"],
    )?;
    number(&value["root_state"], "owner")?;
    number(&value["root_state"], "revision")?;
    hash_field(&value["root_state"], "semantic_sha256")?;
    ensure(
        number(value, "selected_records")? <= 64
            && (1..=4).contains(&number(value, "anchors")?)
            && [8, 16, 32, 64, 128, 256].contains(&number(value, "length_bucket")?),
        "original selected view shape",
    )?;
    number(value, "actual_tokens")?;
    Ok(())
}
fn route(value: &str) -> Result<u8, ArenaError> {
    match value {
        "route_p" => Ok(4),
        "route_c" => Ok(5),
        "route_r" => Ok(6),
        _ => Err(error("original route recipient")),
    }
}
fn requested_routes(profile: &RoventUtilityComparisonCohort) -> Vec<u8> {
    use RoventUtilityCohortRecipient as R;
    profile
        .input
        .requested_recipients
        .iter()
        .map(|r| match r {
            R::Proposer => 4,
            R::Critic => 5,
            R::Researcher => 6,
        })
        .collect()
}
fn mask(value: &Value) -> Result<[bool; 7], ArenaError> {
    let values = value
        .as_array()
        .ok_or_else(|| error("original action mask"))?;
    ensure(values.len() == 7, "original mask length")?;
    let mut result = [false; 7];
    for (out, value) in result.iter_mut().zip(values) {
        *out = value
            .as_bool()
            .ok_or_else(|| error("original boolean mask"))?;
    }
    Ok(result)
}
fn sha_set(value: &Value, maximum: usize) -> Result<BTreeSet<String>, ArenaError> {
    let values = value.as_array().ok_or_else(|| error("original SHA set"))?;
    ensure(values.len() <= maximum, "original SHA set capacity")?;
    let mut set = BTreeSet::new();
    let mut prior = None;
    for value in values {
        let text = value
            .as_str()
            .ok_or_else(|| error("original SHA set item"))?;
        ensure(
            generation_sha(text) && prior.is_none_or(|v: &str| v < text) && set.insert(text.into()),
            "original unique sorted SHA set",
        )?;
        prior = Some(text);
    }
    Ok(set)
}
const COUNTERS: &[&str] = &[
    "native_preparations",
    "submitted_role_tickets",
    "physical_role_completions",
    "completed_nn_inputs",
    "nn_input_count_unknown",
    "consumed_role_outputs",
    "accepted_nn_inputs",
    "accepted_nn_input_count_unknown",
    "consumed_nn_inputs",
    "consumed_nn_input_count_unknown",
    "zero_nn_outputs_consumed",
    "unused_role_results",
    "canceled_preparations",
    "cpu_slice_receipts",
    "cpu_progress_publications",
    "cpu_yields",
    "cpu_yield_requests",
    "cpu_yield_acknowledgements",
    "cpu_natural_completions",
    "validator_ticket_admissions",
    "foreign_reports",
    "foreign_reported_nodes",
    "foreign_node_count_unknown",
    "foreign_yield_requests",
    "foreign_yield_acknowledgements",
    "rejected_events",
    "notification_waits",
];
fn counters(value: &Value) -> Result<(), ArenaError> {
    fields(value, COUNTERS)?;
    for key in COUNTERS {
        number(value, key)?;
    }
    Ok(())
}

fn verify_cost(
    value: &Value,
    profile: &RoventUtilityComparisonCohort,
) -> Result<rz_uci::pals_adaptive::UtilityCohortCostWire, ArenaError> {
    verify_cost_input(value, &profile.input)
}
pub(in crate::rovent_collect) fn verify_cost_input(
    value: &Value,
    input: &rz_experiments::rovent_utility_cohort::RoventUtilityComparisonCohortInput,
) -> Result<rz_uci::pals_adaptive::UtilityCohortCostWire, ArenaError> {
    verify_cost_window_input(value, input, true)
}
pub(in crate::rovent_collect) fn verify_cost_window_input(
    value: &Value,
    input: &rz_experiments::rovent_utility_cohort::RoventUtilityComparisonCohortInput,
    action_window_opened: bool,
) -> Result<rz_uci::pals_adaptive::UtilityCohortCostWire, ArenaError> {
    fields(
        value,
        &[
            "domain",
            "scope",
            "objective",
            "cost_class_sha256",
            "cost_epoch_sha256",
            "resource_plan_sha256",
            "resource_epoch",
            "clock_epoch",
            "started_tick",
            "root_fence_tick",
            "actor_release_tick",
            "physical_pin_zero_tick",
            "whole_wall_ns",
            "original_deadline_tick",
            "cohort_action_cutoff_tick",
            "cohort_cleanup_cutoff_tick",
            "baseline_counters",
            "final_counters",
            "native_credit_baseline",
            "native_credit_final",
            "cpu_thread_time_ns",
            "energy",
            "complete_scope",
        ],
    )?;
    let cost: rz_uci::pals_adaptive::UtilityCohortCostWire =
        serde_json::from_value(value.clone()).map_err(error)?;
    let p = input;
    ensure(
        same(&cost, value)?
            && cost.domain == "rz-rovent-utility-cohort-cost/1"
            && cost.scope == ROVENT_UTILITY_COHORT_COST_SCOPE
            && cost.objective == "exclusive_wall_nanoseconds"
            && cost.cost_class_sha256 == p.cost_class_sha256
            && cost.resource_plan_sha256 == p.resource_plan_sha256
            && cost.resource_epoch == p.resource_epoch
            && cost.cost_epoch_sha256.as_deref().is_none_or(generation_sha),
        "original cost profile",
    )?;
    ensure(
        cost.started_tick <= cost.root_fence_tick
            && cost.root_fence_tick <= cost.actor_release_tick
            && cost.actor_release_tick <= cost.physical_pin_zero_tick
            && cost.physical_pin_zero_tick - cost.started_tick == cost.whole_wall_ns
            && (!action_window_opened
                || (cost.started_tick.checked_add(p.action_ns)
                    == Some(cost.cohort_action_cutoff_tick)
                    && cost.cohort_action_cutoff_tick.checked_add(p.cleanup_ns)
                        == Some(cost.cohort_cleanup_cutoff_tick)))
            && cost.native_credit_baseline.0 <= cost.native_credit_final.0
            && cost.native_credit_baseline.1 == 0
            && cost.native_credit_final.1 == 0,
        "original whole wall cost/credit",
    )?;
    counters(&cost.baseline_counters)?;
    counters(&cost.final_counters)?;
    for key in COUNTERS {
        ensure(
            number(&cost.baseline_counters, key)? <= number(&cost.final_counters, key)?,
            "original counter reversal",
        )?;
    }
    Ok(cost)
}

fn runtime_resources_known(value: &Value) -> Result<bool, ArenaError> {
    fields(value, &["native", "own"])?;
    fn one(value: &Value) -> Result<bool, ArenaError> {
        if value.is_null() {
            return Ok(false);
        }
        if value
            .as_object()
            .is_some_and(|v| v.len() == 1 && v.contains_key("error"))
        {
            string(value, "error")?;
            return Ok(false);
        }
        fields(
            value,
            &[
                "actor",
                "affinity",
                "probe_method",
                "unknown_reason",
                "thread_id_present",
            ],
        )?;
        let actor = string(value, "actor")?;
        let indexed = ["OwnWorker(", "ExternalEngine(", "RuntimeHelper("]
            .into_iter()
            .any(|p| {
                actor
                    .strip_prefix(p)
                    .and_then(|v| v.strip_suffix(')'))
                    .is_some_and(|v| {
                        !v.is_empty()
                            && v.bytes().all(|b| b.is_ascii_digit())
                            && v.parse::<u16>().is_ok()
                    })
            });
        ensure(
            matches!(actor, "Manager" | "NativeCaller") || indexed,
            "original thread actor",
        )?;
        let present = boolean(value, "thread_id_present")?;
        if value["affinity"].is_null() {
            ensure(
                value["probe_method"].is_null()
                    && matches!(
                        string(value, "unknown_reason")?,
                        "UnsupportedPlatform"
                            | "ReadUnavailable"
                            | "InvalidFormat"
                            | "BudgetExceeded"
                            | "NamespaceOrMountAmbiguous"
                            | "ChangedDuringProbe"
                            | "HiddenAncestors"
                            | "WindowsJobHierarchyUnavailable"
                            | "WindowsGroupAffinityAmbiguous"
                            | "UnobservedThread"
                            | "UnobservedOrtSettings"
                    ),
                "explicit unknown affinity",
            )?;
            return Ok(false);
        }
        let cpus = array(value, "affinity", 65536)?;
        for pair in cpus {
            let p = pair.as_array().ok_or_else(|| error("CPU affinity pair"))?;
            ensure(
                p.len() == 2
                    && p[0].as_u64().is_some_and(|v| v <= u16::MAX as u64)
                    && p[1].as_u64().is_some_and(|v| v <= u32::MAX as u64),
                "CPU affinity range",
            )?;
        }
        ensure(
            value["unknown_reason"].is_null()
                && matches!(
                    string(value, "probe_method")?,
                    "LinuxProcThreadStatus"
                        | "LinuxProcProcessStatus"
                        | "LinuxSysfsSiblings"
                        | "LinuxCgroupV2Direct"
                        | "LinuxCgroupV2VisibleAncestors"
                        | "WindowsProcessAndThreadGroupAffinity"
                        | "WindowsLogicalProcessorInformationEx"
                        | "WindowsImmediateJobObject"
                        | "WorkerReadyBoundary"
                        | "OrtAppliedSessionSettings"
                ),
            "original known affinity method",
        )?;
        Ok(present && !cpus.is_empty())
    }
    let native = one(&value["native"])?;
    let own = array(value, "own", 4)?;
    let mut known = native && !own.is_empty();
    for row in own {
        known &= one(row)?;
    }
    Ok(known)
}

pub(in crate::rovent_collect) struct Prelude {
    pub(in crate::rovent_collect) ready: String,
    pub(in crate::rovent_collect) resource_known: bool,
    pub(in crate::rovent_collect) recipients: Vec<u8>,
}
pub(in crate::rovent_collect) fn verify_prelude(
    value: &Value,
    root: &RoventRootAdmissionEvidence,
) -> Result<Prelude, ArenaError> {
    fields(
        value,
        &[
            "captured_tick",
            "native_credit",
            "role_credit",
            "counters",
            "offers",
            "requested_common_nodes",
            "common_node_work",
            "ready_group_sha256",
            "actual_resource_observations",
            "native_thread_readback",
            "own_thread_readbacks",
        ],
    )?;
    ensure(
        number(value, "captured_tick")? < root.deadline_tick
            && value["requested_common_nodes"] == serde_json::json!(root.requested_nodes),
        "original prelude time/nodes",
    )?;
    let credit = &value["native_credit"];
    fields(
        credit,
        &[
            "limit",
            "spent",
            "reserved",
            "highest_job_sequence",
            "highest_execution_sequence",
        ],
    )?;
    for key in [
        "limit",
        "spent",
        "reserved",
        "highest_job_sequence",
        "highest_execution_sequence",
    ] {
        number(credit, key)?;
    }
    ensure(
        number(credit, "spent")?
            .checked_add(number(credit, "reserved")?)
            .is_some_and(|v| v <= number(credit, "limit").unwrap_or(0)),
        "native quota exceeded",
    )?;
    let roles = &value["role_credit"];
    fields(roles, &["spent", "reserved"])?;
    number(roles, "spent")?;
    number(roles, "reserved")?;
    counters(&value["counters"])?;
    let work = &value["common_node_work"];
    fields(
        work,
        &[
            "root_entries",
            "checked_transitions",
            "native_nodes",
            "native_qsearch_nodes",
            "reserved_native_nodes",
            "outstanding_native_executions",
            "native_usage_known",
        ],
    )?;
    for key in [
        "root_entries",
        "checked_transitions",
        "native_nodes",
        "native_qsearch_nodes",
        "reserved_native_nodes",
        "outstanding_native_executions",
    ] {
        number(work, key)?;
    }
    boolean(work, "native_usage_known")?;
    ensure(
        number(work, "native_qsearch_nodes")? <= number(work, "native_nodes")?
            && number(credit, "limit")?
                == root
                    .root_descriptor
                    .input
                    .declared_ceilings
                    .cpu_credit_limit,
        "original native budget/qsearch accounting",
    )?;
    let total = number(work, "root_entries")?
        .checked_add(number(work, "checked_transitions")?)
        .and_then(|v| v.checked_add(number(work, "native_nodes").ok()?))
        .and_then(|v| v.checked_add(number(work, "reserved_native_nodes").ok()?))
        .ok_or_else(|| error("original common node accounting overflow"))?;
    ensure(
        root.requested_nodes.is_none_or(|cap| total <= cap),
        "original common position visit quota",
    )?;
    let mut offers = BTreeMap::new();
    for offer in array(value, "offers", 3)? {
        fields(
            offer,
            &[
                "frame_sequence",
                "recipient",
                "stage",
                "original_first_move",
                "question_sha256",
                "view",
                "full_view_sha256",
                "public_context_sha256",
                "semantic_sha256",
                "budget",
                "deadline_clock",
                "deadline_tick",
            ],
        )?;
        let route = route(string(offer, "recipient")?)?;
        ensure(
            offers
                .insert(route, hash_field(offer, "semantic_sha256")?.to_owned())
                .is_none()
                && number(offer, "frame_sequence")? > 0,
            "unique ready route",
        )?;
        let stage = string(offer, "stage")?;
        let stage_ok = match route {
            4 => matches!(
                stage,
                "ProposerRanking"
                    | "ProposalContinuation"
                    | "Repair"
                    | "ProposalValueFresh"
                    | "CounterValueFresh"
                    | "RepairValueFresh"
                    | "FreshRepairedEndpoint"
                    | "FreshCounterEndpoint"
                    | "RecheckRepairedValueFresh"
            ),
            5 => matches!(
                stage,
                "CriticDivergence" | "CriticReply" | "CounterContinuation" | "CriticRecheck"
            ),
            _ => matches!(
                stage,
                "CpuValidation"
                    | "CpuDiscovery"
                    | "CounterValidation"
                    | "RepairValidation"
                    | "RecheckCounterValidation"
            ),
        };
        ensure(
            stage_ok
                && number(offer, "deadline_clock")? == root.deadline_clock
                && number(offer, "deadline_tick")? == root.deadline_tick,
            "original offer stage/deadline",
        )?;
        packed(u16::try_from(number(offer, "original_first_move")?).map_err(error)?)?;
        checked_view(&offer["view"], &authority(root))?;
        for key in [
            "question_sha256",
            "full_view_sha256",
            "public_context_sha256",
        ] {
            hash_field(offer, key)?;
        }
        fields(
            &offer["budget"],
            &[
                "role_submissions_remaining",
                "original_cpu_credit",
                "original_store_byte_limit",
            ],
        )?;
        for key in [
            "role_submissions_remaining",
            "original_cpu_credit",
            "original_store_byte_limit",
        ] {
            number(&offer["budget"], key)?;
        }
    }
    ensure(!offers.is_empty(), "nonempty actual ready offers")?;
    let resource_known = runtime_resources_known(&value["actual_resource_observations"])?;
    string(value, "native_thread_readback")?;
    string(value, "own_thread_readbacks")?;
    let mut h = Sha256::new();
    h.update(b"rz-rovent-utility-cohort-ready-offer-group/1\0");
    h.update((offers.len() as u64).to_le_bytes());
    for (role, digest) in &offers {
        h.update([*role]);
        h.update(unhex(digest)?);
    }
    h.update(unhex(&digest_value(
        "rz-rovent-utility-cohort-common-node-work/1",
        &serde_json::json!({"requested_nodes":value["requested_common_nodes"],"work":work}),
    )?)?);
    for number in [
        number(credit, "limit")?,
        number(credit, "spent")?,
        number(credit, "reserved")?,
        number(roles, "spent")?,
        number(roles, "reserved")?,
    ] {
        h.update(number.to_le_bytes());
    }
    let mut allocation = value["counters"].clone();
    allocation
        .as_object_mut()
        .unwrap()
        .remove("notification_waits");
    for (domain, v) in [
        ("rz-rovent-utility-cohort-prelude-counters/1", &allocation),
        (
            "rz-rovent-utility-cohort-actual-resource-observations/1",
            &value["actual_resource_observations"],
        ),
    ] {
        h.update(unhex(&digest_value(domain, v)?)?);
    }
    let ready = format!("{:x}", h.finalize());
    ensure(
        ready == hash_field(value, "ready_group_sha256")?,
        "original ready-group preimage",
    )?;
    Ok(Prelude {
        ready,
        resource_known,
        recipients: offers.keys().copied().collect(),
    })
}

fn verify_source(
    publication: &RoventUtilityCohortPublication,
    files: &BTreeMap<String, Vec<u8>>,
    profile: &RoventUtilityComparisonCohort,
) -> Result<
    (
        RoventUtilityCohortCollectionSource,
        RoventInputLock,
        v2::Model,
        Position,
    ),
    ArenaError,
> {
    let raw_source = files
        .get(&publication.source.path)
        .ok_or_else(|| error("source payload missing"))?;
    let source_value: Value = decode(raw_source)?;
    fields(
        &source_value,
        &[
            "schema",
            "source_pins",
            "nnuec_registration",
            "nnuec_weights_artifact",
            "nnuec_provenance_artifact",
            "native_origin",
            "registration_root",
            "collection_manifest",
            "collection_metadata",
            "rules_root",
            "transposition_sha256",
            "frozen_pc_sources",
            "ancestor_datasets",
            "resource_observation",
        ],
    )?;
    let source: RoventUtilityCohortCollectionSource =
        serde_json::from_value(source_value.clone()).map_err(error)?;
    ensure(
        same(&source_value, &source)? && source.schema == SOURCE_DOMAIN,
        "closed original collection source",
    )?;
    let lock = RoventInputLock::from_json(
        std::str::from_utf8(&files[&publication.input_lock.path]).map_err(error)?,
    )?;
    let pins = &source.source_pins;
    let original = &profile.input;
    ensure(
        pins.input_lock_sha256 == original.input_lock_sha256
            && pins.source_sha256 == original.source_sha256
            && pins.frozen_nnuec_model_sha256 == original.frozen_nnuec_model_sha256
            && pins.frozen_nnuec_namespace_sha256 == original.frozen_nnuec_namespace_sha256
            && pins.frozen_pc_model_sha256 == original.frozen_pc_model_sha256
            && pins.frozen_pc_namespace_sha256 == original.frozen_pc_namespace_sha256
            && pins.resource_plan_sha256 == original.resource_plan_sha256
            && pins.logical_budget_sha256 == original.logical_budget_sha256,
        "original profile/source pins",
    )?;
    ensure(
        lock.canonical_sha256 == pins.input_lock_sha256
            && lock.manifest.purpose == RoventPurpose::Collection
            && lock.manifest.engine.mode == original.mode.execution_mode()
            && lock.manifest.resources.canonical_sha256()? == pins.resource_plan_sha256
            && rz_experiments::rovent_source_aggregate_sha256(
                &lock.manifest.engine.source_commit,
                &lock.manifest.engine.source_components,
            )? == pins.source_sha256,
        "original full lock/source/mode/resource plan",
    )?;
    source.collection_manifest.validate()?;
    ensure(
        source.collection_manifest.source_sha256 == pins.source_sha256
            && source.collection_manifest.source_commit == lock.manifest.engine.source_commit,
        "original corpus source enrollment",
    )?;
    let registration: RoventRootAdmissionEvidence =
        serde_json::from_value(source.registration_root.clone()).map_err(error)?;
    ensure(
        same(&source.registration_root, &registration)?
            && source.native_origin == source.registration_root["native_origin"],
        "original registration root/native enrollment",
    )?;
    verify_root(&registration, &source.registration_root, &lock, pins)?;
    let root = source.rules_root.restore()?;
    verify_root_rules(&registration, &root)?;
    ensure(
        exact_rules_anchor(&root)?.rules_state_sha256 == registration.root_state.semantic
            && crate::pals_collect_v3::generation_transposition_sha256(&root)
                == source.transposition_sha256,
        "original registration Rules root",
    )?;
    let nnuec = lock
        .manifest
        .engine
        .nnuec
        .as_ref()
        .ok_or_else(|| error("original NNUEC pin missing"))?;
    ensure(
        source.nnuec_weights_artifact == nnuec.weights
            && source.nnuec_provenance_artifact == nnuec.provenance
            && nnuec.weights.sha256 == publication.nnuec_weights.sha256
            && nnuec.weights.bytes == publication.nnuec_weights.bytes
            && nnuec.provenance.sha256 == publication.nnuec_provenance.sha256
            && nnuec.provenance.bytes == publication.nnuec_provenance.bytes,
        "original NNUEC raw artifact pins",
    )?;
    // Parsing/parameter identity only; no board, context encoding or forward.
    let model =
        v2::Model::from_reader(files[&publication.nnuec_weights.path].as_slice()).map_err(error)?;
    ensure(
        model.identity().weights_sha256.as_deref() == Some(pins.frozen_nnuec_model_sha256.as_str()),
        "original NNUEC semantic parameters",
    )?;
    let registered = &source.nnuec_registration;
    fields(
        registered,
        &[
            "domain",
            "schema",
            "semantics",
            "encoding",
            "implementation_sha256",
            "configuration_sha256",
            "context_namespace_policy",
            "local_namespace_policy",
            "context_namespace_policy_sha256",
            "local_namespace_policy_sha256",
            "model_identity",
        ],
    )?;
    let calculated = model.registration().map_err(error)?;
    ensure(
        same(registered, &calculated)?
            && string(registered, "domain")? == v2::registration::REGISTRATION_SCHEMA
            && string(registered, "schema")? == nnuec.schema
            && string(registered, "semantics")? == nnuec.semantics
            && string(registered, "encoding")? == nnuec.encoding
            && string(registered, "implementation_sha256")? == nnuec.implementation_sha256
            && string(registered, "configuration_sha256")? == nnuec.configuration_sha256
            && nnuec.configuration_sha256 == calculated.configuration_sha256()
            && string(registered, "context_namespace_policy_sha256")?
                == nnuec.context_namespace_policy_sha256
            && string(registered, "local_namespace_policy_sha256")?
                == nnuec.local_namespace_policy_sha256
            && string(registered, "context_namespace_policy")?
                == calculated.context_namespace_policy()
            && string(registered, "local_namespace_policy")? == calculated.local_namespace_policy()
            && same(&registered["model_identity"], model.identity())?
            && digest_value("rz-rovent-utility-cohort-nnuec-namespace/1", registered)?
                == pins.frozen_nnuec_namespace_sha256,
        "original NNUEC registered namespace",
    )?;
    let provenance: Value = decode(&files[&publication.nnuec_provenance.path])?;
    fields(
        &provenance,
        &[
            "domain",
            "weights_raw_sha256",
            "semantic_weights_sha256",
            "schema",
            "semantics",
            "encoding",
            "implementation_sha256",
            "configuration_sha256",
            "context_namespace_policy_sha256",
            "local_namespace_policy_sha256",
            "training",
            "diagnostic",
            "provenance",
        ],
    )?;
    ensure(
        string(&provenance, "domain")? == "rz-rovent-nnuec-provenance/1"
            && string(&provenance, "weights_raw_sha256")? == nnuec.weights.sha256
            && string(&provenance, "semantic_weights_sha256")? == pins.frozen_nnuec_model_sha256
            && same(&provenance["training"], &model.identity().training)?
            && provenance["diagnostic"] == serde_json::json!(nnuec.diagnostic),
        "original NNUEC provenance identity",
    )?;
    for name in [
        "schema",
        "semantics",
        "encoding",
        "implementation_sha256",
        "configuration_sha256",
        "context_namespace_policy_sha256",
        "local_namespace_policy_sha256",
    ] {
        ensure(
            provenance[name] == registered[name],
            "original NNUEC provenance/registration",
        )?;
    }
    let mut seen = BTreeSet::new();
    for artifact in array(&provenance, "provenance", 32)? {
        let artifact: rz_experiments::ArtifactRef =
            serde_json::from_value(artifact.clone()).map_err(error)?;
        artifact.validate()?;
        ensure(
            seen.insert(artifact.path.clone()) && lock.manifest.provenance.contains(&artifact),
            "original NNUEC provenance enrollment",
        )?;
    }
    let origin: RoventNativeOriginEvidence =
        serde_json::from_value(source.native_origin.clone()).map_err(error)?;
    let namespace = serde_json::json!({"model_semantics":origin.model_semantics,"model_semantics_sha256":origin.model_semantics_sha256,"encoding":origin.encoding,
        "encoding_sha256":origin.encoding_sha256,"encoding_registry_sha256":origin.encoding_registry_sha256,"model_implementation_sha256":origin.model_implementation_sha256,
        "representation_model_domain":origin.representation_model_domain,"frozen_epoch":origin.frozen_epoch,"weights_sha256":origin.weights_sha256,
        "export_manifest_sha256":origin.export_manifest_sha256,"backend_sha256":origin.backend_sha256});
    ensure(
        digest_value("rz-rovent-utility-cohort-pc-namespace/1", &namespace)?
            == pins.frozen_pc_namespace_sha256
            && origin.weights_sha256 == pins.frozen_pc_model_sha256
            && source.frozen_pc_sources.len() == 2,
        "original frozen P/C namespace",
    )?;
    for ((role, declared), frozen) in [
        ("proposer", &lock.manifest.engine.pc.proposer),
        ("critic", &lock.manifest.engine.pc.critic),
    ]
    .into_iter()
    .zip(&source.frozen_pc_sources)
    {
        frozen.artifact.validate()?;
        ensure(
            frozen.role == role
                && frozen.artifact == declared.weights
                && !declared.diagnostic
                && frozen.model_semantics_sha256 == origin.model_semantics_sha256
                && frozen.model_identity_sha256 == origin.weights_sha256
                && frozen.encoding_sha256 == origin.encoding_sha256
                && declared.weights.sha256 == origin.weights_sha256
                && declared.semantics == origin.model_semantics
                && declared.encoding == origin.encoding
                && declared.implementation_sha256 == origin.model_implementation_sha256,
            "original frozen P/C artifact/source",
        )?;
    }
    let mut ancestors = BTreeSet::new();
    for ancestor in &source.ancestor_datasets {
        ancestor.validate()?;
        ensure(
            ancestors.insert(ancestor.publication_sha256.clone()),
            "duplicate original ancestor",
        )?;
    }
    ensure(
        source.ancestor_datasets.len() <= 256
            && source.resource_observation.len() == publication.trial_count as usize,
        "original ancestor/resource coverage",
    )?;
    for resource in &source.resource_observation {
        verify_resource_custody(resource)?;
    }
    ensure(
        source
            .collection_manifest
            .opening_splits
            .contains_key(&source.collection_metadata.opening_id),
        "original opening split enrollment",
    )?;
    Ok((source, lock, model, root))
}

fn verify_root(
    root: &RoventRootAdmissionEvidence,
    raw: &Value,
    lock: &RoventInputLock,
    pins: &UtilityCohortSourcePins,
) -> Result<(), ArenaError> {
    root.root_descriptor.verify_against(lock)?;
    let input = &root.root_descriptor.input;
    for digest in [
        &root.implementation_sha256,
        &root.selective_policy_sha256,
        &root.native_origin.model_semantics_sha256,
        &root.native_origin.encoding_sha256,
        &root.native_origin.encoding_registry_sha256,
        &root.native_origin.model_implementation_sha256,
        &root.native_origin.backend_sha256,
        &root.native_origin.runtime_library_sha256,
        &root.native_origin.export_manifest_sha256,
        &root.native_origin.native_configuration_sha256,
    ] {
        ensure(generation_sha(digest), "original registered source digest")?;
    }
    ensure(
        root.schema == "rz-rovent-collection-root-admission/1"
            && root.native_origin.schema == "rz-rovent-collection-native-origin/1"
            && generation_digest(&root.native_origin.schema, &raw["native_origin"])?
                == root.native_origin_sha256
            && root.native_origin.input_lock_sha256 == lock.canonical_sha256
            && root.native_origin.source_sha256 == pins.source_sha256
            && root.native_origin.source_commit == lock.manifest.engine.source_commit
            && root.native_origin.source_components == lock.manifest.engine.source_components
            && root.native_origin.mode == format!("{:?}", lock.manifest.engine.mode)
            && root.native_origin.purpose == "collection"
            && root.native_origin.weights_sha256 == pins.frozen_pc_model_sha256
            && root.native_origin.process_epoch == root.process_epoch
            && root.root_descriptor_sha256 == hex(root.root_descriptor.digest()?)
            && root.root_budget_sha256 == root.root_descriptor.root_budget_sha256
            && root.root_state.semantic == root.root_descriptor.input.state_and_history_sha256
            && root.root_moves == root.root_descriptor.input.allowed_root_moves
            && root.deadline_clock == root.root_descriptor.input.clock.hard.clock_epoch
            && root.deadline_tick == root.root_descriptor.input.clock.hard.at_ns
            && root.process_epoch == input.authority.epoch
            && root.game_generation == input.authority.game
            && root.root_generation == input.authority.root
            && root.implementation_sha256 == input.authority.implementation_sha256
            && root.depth_plies == input.original_go.depth_plies.map(|v| v.get())
            && root.mate_moves == input.original_go.mate_moves.map(|v| v.get())
            && root.requested_nodes == input.original_go.requested_nodes,
        "original root descriptor/source/budget/deadline",
    )?;
    ensure(
        digest_value(
            "rz-rovent-utility-cohort-logical-budget/1",
            &serde_json::to_value(&root.root_descriptor.input.declared_ceilings).map_err(error)?,
        )? == pins.logical_budget_sha256,
        "original root logical budget",
    )
}

fn verify_root_rules(
    root: &RoventRootAdmissionEvidence,
    position: &Position,
) -> Result<(), ArenaError> {
    let allowed = root
        .root_moves
        .iter()
        .map(|text| rz_position::BoardMove::from_uci(text).map_err(error))
        .collect::<Result<Vec<_>, _>>()?;
    ensure(
        !allowed.is_empty()
            && allowed.len() <= 218
            && allowed
                .iter()
                .enumerate()
                .all(|(i, m)| !allowed[..i].contains(m))
            && root.root_state.revision == position.revision()
            && root.root_state.semantic == exact_rules_anchor(position)?.rules_state_sha256
            && position
                .ordered_legal_moves()
                .moves()
                .iter()
                .copied()
                .filter(|m| allowed.contains(m))
                .collect::<Vec<_>>()
                == allowed,
        "original root exact history/revision/ordered legal restriction",
    )
}

fn verify_resource_custody(value: &Value) -> Result<(), ArenaError> {
    fields(
        value,
        &[
            "schema",
            "plan_pin",
            "evidence_directory",
            "profile",
            "admission_attempted",
            "finished_attempted",
            "original_plan_descriptor_retained",
            "original_delegated_descriptor_retained",
            "original_scope_retained",
            "resource_failure",
            "admission",
            "finished_live",
            "finished_live_error",
            "controlled_comparison_accepted",
            "automatic_cleanup",
        ],
    )?;
    ensure(
        string(value, "schema")? == "rz-rovent-resource-custody/1"
            && value["admission_attempted"] == true
            && value["original_plan_descriptor_retained"] == true
            && value["original_delegated_descriptor_retained"] == true
            && value["original_scope_retained"] == true
            && value["resource_failure"].is_null()
            && value["controlled_comparison_accepted"] == false
            && value["automatic_cleanup"] == false,
        "original actual resource custody",
    )?;
    let pin: RoventPublicationFile =
        serde_json::from_value(value["plan_pin"].clone()).map_err(error)?;
    pin.validate()?;
    ensure(
        same(&pin, &value["plan_pin"])?
            && pin.path.starts_with("inputs/")
            && pin.bytes <= 64 << 10
            && matches!(
                string(value, "profile")?,
                "functional_validation" | "strict_comparison"
            )
            && !value["admission"].is_null(),
        "original resource plan/admission",
    )?;
    let directory = string(value, "evidence_directory")?;
    ensure(
        !directory.is_empty()
            && !directory.starts_with('/')
            && !directory.contains('\\')
            && directory
                .split('/')
                .all(|v| !v.is_empty() && v != "." && v != ".." && !v.contains(':')),
        "relative resource evidence directory",
    )?;
    let finished = boolean(value, "finished_attempted")?;
    ensure(
        finished || (value["finished_live"].is_null() && value["finished_live_error"].is_null()),
        "resource finish attempt",
    )?;
    if !value["finished_live_error"].is_null() {
        string(value, "finished_live_error")?;
    }
    for key in ["admission", "finished_live"] {
        if !value[key].is_null() {
            let observation: rz_experiments::RoventResourceObservation =
                serde_json::from_value(value[key].clone()).map_err(error)?;
            ensure(
                same(&value[key], &observation)?,
                "closed typed resource observation",
            )?;
            observation.evidence.validate()?;
            // Resource-plan/whole-peak Unknown stays Unknown; this does not award
            // strict performance acceptance from a cohort's recorded declaration.
        }
    }
    Ok(())
}

fn verify_reference_input(
    input: &Value,
    model: &v2::Model,
    frozen: &UtilityCohortFrozenDecision,
    root: &Position,
) -> Result<CheckedInput, ArenaError> {
    let checked = verify_captured_utility_input(input, model, root)?;
    let metadata: Vec<RecordMetadata> =
        serde_json::from_value(input["records_metadata"].clone()).map_err(error)?;
    let target = usize::try_from(number(input, "target_record")?).map_err(error)?;
    let selected = metadata
        .get(target)
        .ok_or_else(|| error("original target record"))?;
    ensure(
        frozen.input_sha256 == checked.input_sha256
            && frozen.target_sha256 == checked.target_sha256
            && frozen.context_sha256 == checked.context_sha256
            && frozen.input_bytes == number(input, "bytes")?
            && same(&input["target_record_ids"], &frozen.target_record_ids)?
            && frozen.root_rules_sha256 == exact_rules_anchor(root)?.rules_state_sha256
            && selected.original_first_move == Some(frozen.original_first_move),
        "original utility input/target/submission/frozen seals",
    )?;
    Ok(checked)
}
/// Shared exact Rules/numerical/provenance checks only. No /1 cohort metadata,
/// frozen-decision authority or live capability is constructed by this leaf.
pub(in crate::rovent_collect) fn verify_captured_utility_input(
    input: &Value,
    model: &v2::Model,
    root: &Position,
) -> Result<CheckedInput, ArenaError> {
    fields(
        input,
        &[
            "reference_json",
            "bytes",
            "input_sha256",
            "target_sha256",
            "submission_target_sha256",
            "context_sha256",
            "target_record",
            "target_record_ids",
            "base_identity",
            "root_rules",
            "records_metadata",
            "source_rules",
            "provenance",
        ],
    )?;
    let reference_json = string(input, "reference_json")?;
    ensure(
        reference_json.len() <= v2::utility_capture::MAX_INPUT_BYTES
            && number(input, "bytes")? == reference_json.len() as u64,
        "original input bytes",
    )?;
    let reference: v2::utility::Reference = decode(reference_json.as_bytes())?;
    let current: RulesSnapshot =
        serde_json::from_value(input["root_rules"].clone()).map_err(error)?;
    let current = current.restore()?;
    ensure(
        current.snapshot().same_state(&root.snapshot()),
        "original utility Rules root",
    )?;
    ensure(
        reference.schema == v2::utility::INPUT_SCHEMA
            && reference.base_input.schema == v2::INPUT_SCHEMA
            && reference.base_input.board_features == rules::reference_board_features(&current)
            && reference.base_input.side_to_move
                == if current.side_to_move() == rz_position::Color::White {
                    "white"
                } else {
                    "black"
                }
            && same(
                &reference.base_input.local_line,
                &rules::reference_line_from_rules(&current, &[]).map_err(error)?,
            )?,
        "actual utility root board/local input",
    )?;
    let config = &model.base_checkpoint().config;
    let metadata: Vec<RecordMetadata> =
        serde_json::from_value(input["records_metadata"].clone()).map_err(error)?;
    let sources: Vec<RulesSnapshot> =
        serde_json::from_value(input["source_rules"].clone()).map_err(error)?;
    ensure(
        !metadata.is_empty()
            && metadata.len() <= 64
            && metadata.len() <= config.max_lines
            && metadata.len() == sources.len()
            && metadata.len() == reference.base_input.context_lines.len(),
        "original utility record/source coverage",
    )?;
    let mut edges = Vec::new();
    let mut anchors = BTreeMap::new();
    anchors.insert(0_u8, current.snapshot());
    for (index, ((metadata, source), line)) in metadata
        .iter()
        .zip(&sources)
        .zip(&reference.base_input.context_lines)
        .enumerate()
    {
        let source = source.restore()?;
        let selected = if metadata.anchor == 0 {
            current.snapshot()
        } else {
            ensure(
                metadata.anchor < 4 && metadata.source_line_offset == 0,
                "original selected old anchor requires unrebased source",
            )?;
            let snapshot = source.snapshot();
            if let Some(prior) = anchors.insert(metadata.anchor, snapshot.clone()) {
                ensure(
                    prior.same_state(&snapshot),
                    "same selected anchor refers to different Rules histories",
                )?;
            }
            snapshot
        };
        if metadata.anchor == 0 {
            let original = source.snapshot().uci_replay(16384).map_err(error)?;
            let actual = current.snapshot().uci_replay(16384).map_err(error)?;
            ensure(
                original.origin == actual.origin
                    && original.start_fen == actual.start_fen
                    && actual.moves.starts_with(&original.moves)
                    && actual.moves.len() - original.moves.len()
                        == metadata.source_line_offset as usize,
                "original source offset does not reach selected root",
            )?;
        }
        let selected = Position::from_snapshot(
            &selected,
            rz_position::PositionLimits {
                max_history_positions: 16385,
                ..Default::default()
            },
        )
        .map_err(error)?;
        // A reference move is an existing normalized numerical token. The
        // original typed Rules checker owns its inversion/legality below.
        ensure(
            line.id == index as u64
                && line.kind < 4
                && line.moves.len() <= config.max_plies
                && line.anchor_features
                    == rules::reference_board_features(&selected)[selected.side_to_move() as usize],
            "original utility typed line metadata",
        )?;
        let moves = line
            .moves
            .iter()
            .map(numerical_move)
            .collect::<Result<Vec<_>, _>>()?;
        let checked_line = rules::reference_line_from_rules(&selected, &moves).map_err(error)?;
        ensure(
            same(&line.moves, &checked_line.moves)? && line.perspective == checked_line.perspective,
            "original whole line Rules replay",
        )?;
        let first = metadata.original_first_move.map(packed).transpose()?;
        let expected_first = first
            .map(|movement| {
                rules::reference_line_from_rules(&source, &[movement]).map(|line| line.moves[0])
            })
            .transpose()
            .map_err(error)?;
        ensure(
            same(&line.original_first_move, &expected_first)?,
            "original source first-move token",
        )?;
        for (parent, kind) in [(metadata.parent, 0), (metadata.supersedes, 1)] {
            if let Some(parent) = parent {
                ensure(
                    usize::from(parent) < sources.len() && usize::from(parent) != index,
                    "original relation must name another selected record",
                )?;
                edges.push(v2::ReferenceEdge {
                    source: u64::from(parent),
                    target: index as u64,
                    kind,
                });
            }
        }
    }
    ensure(
        same(&reference.base_input.context_edges, &edges)?,
        "original selected relation order",
    )?;
    let target = usize::try_from(number(input, "target_record")?).map_err(error)?;
    let selected = metadata
        .get(target)
        .ok_or_else(|| error("original target record"))?;
    ensure(
        selected.original_first_move.is_some(),
        "target original first move",
    )?;
    let ids: Vec<_> = metadata
        .iter()
        .enumerate()
        .filter_map(|(i, item)| {
            (item.source_state == selected.source_state
                && item.original_first_move == selected.original_first_move)
                .then_some(i as u64)
        })
        .collect();
    ensure(
        reference.target_ids == ids && same(&input["target_record_ids"], &ids)?,
        "whole exact original target bundle",
    )?;
    for (index, item) in metadata.iter().enumerate() {
        for parent in [item.parent, item.supersedes].into_iter().flatten() {
            ensure(
                ids.contains(&(index as u64)) == ids.contains(&u64::from(parent)),
                "target relation crosses bundle",
            )?;
        }
    }
    let context_sha256 = v2::context_sha256(
        &reference.base_input.context_lines,
        &reference.base_input.context_edges,
    )
    .map_err(error)?;
    ensure(
        context_sha256 == string(input, "context_sha256")?
            && same(&input["base_identity"], model.identity())?,
        "original context/model identity",
    )?;
    let weights = model
        .identity()
        .weights_sha256
        .as_ref()
        .ok_or_else(|| error("model semantic weights identity"))?;
    let mut h = Sha256::new();
    h.update(v2::utility_capture::INPUT_CAPTURE_DOMAIN.as_bytes());
    h.update([0]);
    h.update(weights.as_bytes());
    h.update([0]);
    h.update((reference_json.len() as u64).to_le_bytes());
    h.update(reference_json.as_bytes());
    let input_sha256 = format!("{:x}", h.finalize());
    let mut h = Sha256::new();
    h.update(v2::utility_capture::TARGET_CAPTURE_DOMAIN.as_bytes());
    h.update([0]);
    h.update(context_sha256.as_bytes());
    h.update([0]);
    h.update((ids.len() as u32).to_le_bytes());
    for id in &ids {
        h.update(id.to_le_bytes());
    }
    let target_sha256 = format!("{:x}", h.finalize());
    let provenance = &input["provenance"];
    fields(
        provenance,
        &[
            "root_state",
            "generation",
            "sequence",
            "view_seal_sha256",
            "full_view_sha256",
            "record_scope_sha256",
            "policy_sha256",
            "model",
        ],
    )?;
    for key in [
        "view_seal_sha256",
        "full_view_sha256",
        "record_scope_sha256",
        "policy_sha256",
    ] {
        ensure(
            generation_sha(string(provenance, key)?),
            "original submitted provenance digest",
        )?;
    }
    ensure(
        same(&provenance["model"], model.identity())?,
        "original context provenance model",
    )?;
    verify_input_submission_seal(provenance, &metadata)?;
    let mut h = Sha256::new();
    h.update(v2::utility_capture::SUBMISSION_TARGET_CAPTURE_DOMAIN.as_bytes());
    h.update([0]);
    h.update(unhex(string(provenance, "view_seal_sha256")?)?);
    h.update(target_sha256.as_bytes());
    let submission_target_sha256 = format!("{:x}", h.finalize());
    ensure(
        input_sha256 == string(input, "input_sha256")?
            && target_sha256 == string(input, "target_sha256")?
            && submission_target_sha256 == string(input, "submission_target_sha256")?,
        "original utility input/target/submission/frozen seals",
    )?;
    Ok(CheckedInput {
        reference_json: reference_json.into(),
        input_sha256,
        target_sha256,
        submission_target_sha256,
        context_sha256,
    })
}

/// Preserve the producer's existing `view_seal_digest` framing. Source
/// revisions and rebasing offsets are historical metadata, not numerical NN
/// features, but they still distinguish the original immutable submission.
fn verify_input_submission_seal(
    provenance: &Value,
    metadata: &[RecordMetadata],
) -> Result<(), ArenaError> {
    let mut seal = Sha256::new();
    seal.update(b"rz-pals-review-view-seal/1\0");
    seal.update(unhex(hash_field(provenance, "full_view_sha256")?)?);
    for key in ["sequence", "generation", "root_state"] {
        seal.update(number(provenance, key)?.to_le_bytes());
    }
    for record in metadata {
        seal.update(record.source_revision.to_le_bytes());
        seal.update(record.source_state.to_le_bytes());
        seal.update(record.source_line_offset.to_le_bytes());
    }
    ensure(
        format!("{:x}", seal.finalize()) == hash_field(provenance, "view_seal_sha256")?,
        "original submitted view seal/source revision framing",
    )
}

/// Same submitted Request that bound the actual selected work. Numerical
/// input identity alone deliberately excludes these process-local fields;
/// retain and compare their original source instead of treating a matching
/// tensor as permission to relabel a different Ready view.
pub(in crate::rovent_collect) fn verify_submitted_input_source(
    input: &Value,
    selected_offer: &Value,
    root: &RoventRootAdmissionEvidence,
) -> Result<(), ArenaError> {
    let provenance = &input["provenance"];
    let view = &selected_offer["view"];
    ensure(
        number(provenance, "root_state")? == root.store_root_state as u64
            && number(provenance, "generation")? == root.root_generation
            && number(provenance, "sequence")? == number(view, "sequence")?
            && hash_field(provenance, "view_seal_sha256")? == hash_field(view, "view_digest")?
            && hash_field(provenance, "full_view_sha256")?
                == hash_field(selected_offer, "full_view_sha256")?
            && hash_field(provenance, "policy_sha256")?
                == hash_field(view, "context_policy_sha256")?,
        "submitted input differs from original selected Ready view/root",
    )
}

pub(in crate::rovent_collect) struct Physical {
    pub(in crate::rovent_collect) authority: Value,
    pub(in crate::rovent_collect) cache_known: bool,
    pub(in crate::rovent_collect) fresh_product: bool,
    pub(in crate::rovent_collect) high_water: u64,
}
pub(in crate::rovent_collect) fn verify_physical(
    trial: &Value,
    cost: &rz_uci::pals_adaptive::UtilityCohortCostWire,
    model: &v2::Model,
    root: Option<&RoventRootAdmissionEvidence>,
) -> Result<Physical, ArenaError> {
    let value = &trial["physical_closure"];
    fields(
        value,
        &[
            "root_fence_receipt",
            "review_pin_count",
            "engine_release",
            "context_cache_release",
            "input_cache_prelude",
            "native_product",
        ],
    )?;
    let r = &value["root_fence_receipt"];
    fields(
        r,
        &[
            "authority",
            "native_owner_joined",
            "native_owner_quiesced",
            "native_physical_shutdown_confirmed",
            "own_cpu_joined",
            "expected_own_cpu_owners",
            "own_cpu_panicked",
            "external_shutdown_confirmed",
            "external_owner_quiesced",
            "auxiliary_shutdown_confirmed",
            "native_cpu_spent",
            "native_cpu_reserved",
            "pending_tickets",
            "quarantined",
            "request_sequence_high_water",
            "permits_semantic_closure",
        ],
    )?;
    checked_authority(&r["authority"])?;
    if let Some(root) = root {
        ensure(
            r["authority"] == authority(root),
            "physical closure original authority",
        )?;
    }
    let external = match (
        &r["external_shutdown_confirmed"],
        &r["external_owner_quiesced"],
    ) {
        (Value::Null, Value::Null) => true,
        (a, b) => {
            ensure(
                (a.is_boolean() || a.is_null()) && (b.is_boolean() || b.is_null()),
                "external closure shape",
            )?;
            a == true || b == true
        }
    };
    ensure(
        ((boolean(r, "native_owner_joined")? && boolean(r, "native_physical_shutdown_confirmed")?)
            || boolean(r, "native_owner_quiesced")?)
            && number(r, "own_cpu_panicked")? == 0
            && number(r, "own_cpu_joined")? == number(r, "expected_own_cpu_owners")?
            && number(r, "native_cpu_reserved")? == 0
            && number(r, "pending_tickets")? == 0
            && !boolean(r, "quarantined")?
            && boolean(r, "auxiliary_shutdown_confirmed")?
            && external
            && boolean(r, "permits_semantic_closure")?
            && number(value, "review_pin_count")? == 0
            && cost.native_credit_final == (number(r, "native_cpu_spent")?, 0)
            && cost.clock_epoch == number(&r["authority"], "process_epoch")?,
        "actual root/actor/pin-zero closure",
    )?;
    let release = &value["engine_release"];
    fields(
        release,
        &[
            "owner",
            "actual_stack_owner_id",
            "profile",
            "binding_sha256",
            "model_identity",
            "preparation_epoch",
            "release_duration_ns",
            "provenance",
        ],
    )?;
    ensure(
        number(release, "owner")? <= u16::MAX as u64,
        "original released owner",
    )?;
    number(release, "actual_stack_owner_id")?;
    number(release, "preparation_epoch")?;
    let profile = &release["profile"];
    fields(
        profile,
        &[
            "owner",
            "max_depth",
            "evaluation_namespace_sha256",
            "search_conditions_sha256",
        ],
    )?;
    ensure(
        number(profile, "owner")? == number(release, "owner")?
            && number(profile, "max_depth")? <= u16::MAX as u64
            && hash_field(profile, "evaluation_namespace_sha256")?
                == hash_field(release, "binding_sha256")?,
        "original released value binding",
    )?;
    hash_field(profile, "search_conditions_sha256")?;
    ensure(
        same(&release["model_identity"], model.identity())?,
        "original released NNUEC identity",
    )?;
    let duration = string(release, "release_duration_ns")?;
    ensure(
        duration.len() <= 39
            && (duration == "0"
                || (!duration.starts_with('0') && duration.bytes().all(|b| b.is_ascii_digit())))
            && duration.parse::<u128>().is_ok(),
        "original release duration",
    )?;
    let p = &release["provenance"];
    fields(
        p,
        &[
            "root",
            "generation",
            "sequence",
            "full_view_sha256",
            "view_seal_sha256",
            "record_scope_sha256",
            "policy_sha256",
            "model",
        ],
    )?;
    number(p, "root")?;
    number(p, "sequence")?;
    ensure(
        number(p, "generation")? == number(&r["authority"], "root_generation")?
            && same(&p["model"], model.identity())?,
        "release provenance generation/model",
    )?;
    if let Some(root) = root {
        ensure(
            number(p, "root")? == root.store_root_state as u64,
            "released original Store root",
        )?;
    }
    for key in [
        "full_view_sha256",
        "view_seal_sha256",
        "record_scope_sha256",
        "policy_sha256",
    ] {
        hash_field(p, key)?;
    }
    let cache = &value["context_cache_release"];
    fields(cache, &["epoch", "matches_engine_release"])?;
    number(cache, "epoch")?;
    ensure(
        boolean(cache, "matches_engine_release")?,
        "actual cache release matches engine",
    )?;
    let prelude = &value["input_cache_prelude"];
    let cache_known = if prelude.is_null() {
        false
    } else {
        fields(prelude, &["epoch", "matches_input"])?;
        number(prelude, "epoch")?;
        boolean(prelude, "matches_input")?
    };
    let product = &value["native_product"];
    fields(
        product,
        &["original_lifetime_present", "first_product_root"],
    )?;
    Ok(Physical {
        authority: r["authority"].clone(),
        cache_known,
        fresh_product: boolean(product, "original_lifetime_present")?
            && boolean(product, "first_product_root")?,
        high_water: number(r, "request_sequence_high_water")?,
    })
}

fn verify_ticket(
    ticket: &Value,
    root: &RoventRootAdmissionEvidence,
    high_water: u64,
) -> Result<(), ArenaError> {
    fields(
        ticket,
        &[
            "authority",
            "request_epoch",
            "request_sequence",
            "role",
            "question_sha256",
            "view",
            "representation",
            "deadline_clock",
            "deadline_tick",
        ],
    )?;
    let original = authority(root);
    ensure(
        ticket["authority"] == original
            && number(ticket, "request_epoch")? == root.process_epoch
            && (1..=high_water).contains(&number(ticket, "request_sequence")?)
            && number(ticket, "deadline_clock")? == root.deadline_clock
            && number(ticket, "deadline_tick")? == root.deadline_tick,
        "original consumed ticket authority/deadline",
    )?;
    checked_view(&ticket["view"], &original)?;
    hash_field(ticket, "question_sha256")?;
    let r = &ticket["representation"];
    fields(
        r,
        &[
            "input",
            "history",
            "candidates",
            "mask_positions",
            "model",
            "encoding",
            "precision",
            "frozen_epoch",
            "record_revision",
            "role",
        ],
    )?;
    for key in [
        "input",
        "history",
        "candidates",
        "mask_positions",
        "model",
        "encoding",
    ] {
        hash_field(r, key)?;
    }
    ensure(
        hash_field(r, "model")? == root.native_origin.weights_sha256
            && hash_field(r, "encoding")? == root.native_origin.encoding_sha256
            && number(r, "frozen_epoch")? == root.native_origin.frozen_epoch,
        "original P/C model namespace",
    )?;
    number(r, "record_revision")?;
    let role = string(ticket, "role")?;
    ensure(
        matches!(role, "Proposer" | "Critic") && string(r, "role")? == role.to_ascii_lowercase(),
        "original consumed P/C role",
    )?;
    ensure(
        matches!(string(r, "precision")?, "Fp32" | "Fp16" | "Bf16"),
        "registered consumed precision",
    )?;
    Ok(())
}

fn verify_peer(peer: &Value, root: &RoventRootAdmissionEvidence) -> Result<(), ArenaError> {
    fields(
        peer,
        &[
            "authority",
            "id",
            "parent",
            "sender",
            "recipient",
            "purpose",
            "anchor",
            "anchor_root_ply",
            "original_first_move",
            "prefix_sha256",
            "line_sha256",
            "line_offset",
            "question_sha256",
            "view",
            "evidence_revision",
            "root_budget_sha256",
            "deadline_clock",
            "deadline_tick",
        ],
    )?;
    ensure(
        peer["authority"] == authority(root)
            && number(peer, "id")? > 0
            && number(peer, "anchor_root_ply")? <= 256
            && number(peer, "line_offset")? <= 256
            && hash_field(peer, "root_budget_sha256")? == root.root_budget_sha256
            && number(peer, "deadline_clock")? == root.deadline_clock
            && number(peer, "deadline_tick")? == root.deadline_tick,
        "original peer authority/deadline/root",
    )?;
    if !peer["parent"].is_null() {
        ensure(
            number(peer, "parent")? < number(peer, "id")?,
            "original peer parent chronology",
        )?;
    }
    for key in ["sender", "recipient"] {
        ensure(
            matches!(string(peer, key)?, "Proposer" | "Critic" | "Researcher"),
            "original peer recipient",
        )?;
    }
    ensure(
        matches!(
            string(peer, "purpose")?,
            "Propose" | "Counter" | "Repair" | "TacticalReview" | "Continue" | "FreshValue"
        ),
        "original peer purpose",
    )?;
    fields(&peer["anchor"], &["owner", "revision", "semantic_sha256"])?;
    number(&peer["anchor"], "owner")?;
    number(&peer["anchor"], "revision")?;
    hash_field(&peer["anchor"], "semantic_sha256")?;
    checked_view(&peer["view"], &authority(root))?;
    for key in ["prefix_sha256", "question_sha256"] {
        hash_field(peer, key)?;
    }
    if !peer["line_sha256"].is_null() {
        hash_field(peer, "line_sha256")?;
    }
    if !peer["original_first_move"].is_null() {
        packed(u16::try_from(number(peer, "original_first_move")?).map_err(error)?)?;
    }
    number(peer, "evidence_revision")?;
    Ok(())
}

fn verify_origin(
    origin: &Value,
    root: &RoventRootAdmissionEvidence,
    high_water: u64,
) -> Result<(), ArenaError> {
    fields(
        origin,
        &[
            "frame_sequence",
            "ticket",
            "kind",
            "peer",
            "output_sha256",
            "seal_sha256",
        ],
    )?;
    ensure(
        number(origin, "frame_sequence")? > 0,
        "original accepted frame",
    )?;
    let ticket = &origin["ticket"];
    verify_ticket(ticket, root, high_water)?;
    hash_field(origin, "output_sha256")?;
    hash_field(origin, "seal_sha256")?;
    let mut h = Sha256::new();
    h.update(b"rz-rovent-accepted-local-origin-seal/1\0");
    for value in [
        number(origin, "frame_sequence")?,
        number(ticket, "request_sequence")?,
        root.process_epoch,
        root.game_generation,
        root.root_generation,
    ] {
        h.update(value.to_le_bytes());
    }
    h.update(unhex(&root.implementation_sha256)?);
    h.update(unhex(hash_field(ticket, "question_sha256")?)?);
    h.update(unhex(hash_field(&ticket["view"], "view_digest")?)?);
    h.update(number(&ticket["view"], "sequence")?.to_le_bytes());
    h.update(unhex(hash_field(origin, "output_sha256")?)?);
    let repr = &ticket["representation"];
    for key in [
        "input",
        "history",
        "candidates",
        "mask_positions",
        "model",
        "encoding",
    ] {
        h.update(unhex(hash_field(repr, key)?)?);
    }
    for key in ["frozen_epoch", "record_revision"] {
        h.update(number(repr, key)?.to_le_bytes());
    }
    h.update([if string(ticket, "role")? == "Proposer" {
        0
    } else {
        1
    }]);
    h.update([match string(repr, "precision")? {
        "Fp32" => 0,
        "Fp16" => 1,
        "Bf16" => 2,
        _ => return Err(error("original precision seal")),
    }]);
    h.update(root.deadline_tick.to_le_bytes());
    match string(origin, "kind")? {
        "local_accepted" => {
            ensure(
                origin["peer"].is_null(),
                "local accepted origin is not a peer",
            )?;
            h.update([0]);
        }
        "peer_question" => {
            let peer = &origin["peer"];
            verify_peer(peer, root)?;
            ensure(
                peer["question_sha256"] == ticket["question_sha256"]
                    && peer["view"] == ticket["view"]
                    && string(peer, "recipient")? == string(ticket, "role")?,
                "original accepted peer ticket",
            )?;
            h.update([1]);
            h.update(number(peer, "id")?.to_le_bytes());
            h.update(unhex(hash_field(peer, "question_sha256")?)?);
        }
        _ => return Err(error("original accepted origin kind")),
    }
    ensure(
        format!("{:x}", h.finalize()) == string(origin, "seal_sha256")?,
        "original accepted origin seal",
    )
}

pub(in crate::rovent_collect) struct Semantic {
    pub(in crate::rovent_collect) witnesses: BTreeSet<String>,
    pub(in crate::rovent_collect) bodies_complete: bool,
}
fn verify_semantic(
    body: &Value,
    root: &RoventRootAdmissionEvidence,
    root_rules: &Position,
    physical: &Physical,
    model: &v2::Model,
) -> Result<Semantic, ArenaError> {
    let original_store_limit = array(&body["prelude"], "offers", 3)?
        .iter()
        .find(|offer| offer["recipient"] == body["recipient"])
        .ok_or_else(|| error("basis original ready source missing"))?;
    verify_semantic_with_store_limit(
        body,
        root,
        root_rules,
        physical,
        model,
        number(&original_store_limit["budget"], "original_store_byte_limit")?,
    )
}
pub(in crate::rovent_collect) fn verify_semantic_with_store_limit(
    body: &Value,
    root: &RoventRootAdmissionEvidence,
    root_rules: &Position,
    physical: &Physical,
    model: &v2::Model,
    original_store_byte_limit: u64,
) -> Result<Semantic, ArenaError> {
    let value = &body["semantic"];
    fields(
        value,
        &[
            "baseline_witnesses",
            "new_witnesses",
            "witness_preimages",
            "witness_basis",
            "pc_consumption",
            "dependencies",
            "r_selected_pc_consumption",
            "native_executions",
        ],
    )?;
    let baseline = sha_set(&value["baseline_witnesses"], 65536)?;
    let mut tickets = BTreeMap::new();
    let mut sources = BTreeSet::new();
    for c in array(value, "pc_consumption", 64)? {
        fields(
            c,
            &["origin", "question_sha256", "view", "full_view_sha256"],
        )?;
        verify_origin(&c["origin"], root, physical.high_water)?;
        let t = &c["origin"]["ticket"];
        ensure(
            hash_field(c, "question_sha256")? == hash_field(t, "question_sha256")?
                && c["view"] == t["view"],
            "actual P/C consumption original question/view",
        )?;
        hash_field(c, "full_view_sha256")?;
        ensure(
            tickets.insert(number(t, "request_sequence")?, c).is_none(),
            "duplicate original P/C consumption",
        )?;
        sources.insert((
            number(&c["origin"], "frame_sequence")?,
            string(c, "question_sha256")?.to_owned(),
        ));
    }
    let mut executions = BTreeSet::new();
    let mut previous = None;
    for execution in array(value, "native_executions", 64)? {
        fields(execution, &["epoch", "sequence"])?;
        let key = (number(execution, "epoch")?, number(execution, "sequence")?);
        ensure(
            key.0 == root.process_epoch
                && key.1 > 0
                && previous.is_none_or(|p| p < key)
                && executions.insert(key),
            "original ordered native executions",
        )?;
        previous = Some(key);
    }
    let mut dependencies = BTreeSet::new();
    for d in array(value, "dependencies", 64)? {
        fields(
            d,
            &[
                "frame_sequence",
                "question_sha256",
                "view",
                "full_view_sha256",
                "parent_frame_sequence",
                "alias",
                "canonical",
                "origin",
                "role_cause",
                "cpu_cause",
            ],
        )?;
        let frame = number(d, "frame_sequence")?;
        ensure(
            frame > 0 && dependencies.insert((frame, hash_field(d, "question_sha256")?.to_owned())),
            "unique actual dependency",
        )?;
        checked_view(&d["view"], &authority(root))?;
        hash_field(d, "full_view_sha256")?;
        if !d["parent_frame_sequence"].is_null() {
            ensure(
                number(d, "parent_frame_sequence")? < frame,
                "dependency parent chronology",
            )?;
        }
        ensure(
            d["alias"].is_null() == d["canonical"].is_null(),
            "dependency alias/canonical presence",
        )?;
        if !d["alias"].is_null() {
            verify_peer(&d["alias"], root)?;
            verify_peer(&d["canonical"], root)?;
        }
        if !d["origin"].is_null() {
            ensure(!d["alias"].is_null(), "peer source missing alias")?;
            verify_origin(&d["origin"], root, physical.high_water)?;
        }
        ensure(
            d["role_cause"].is_null() || d["cpu_cause"].is_null(),
            "single original continuation cause",
        )?;
        let cause = &d["role_cause"];
        if !cause.is_null() {
            fields(
                cause,
                &[
                    "request_epoch",
                    "request_sequence",
                    "frame_sequence",
                    "question_sha256",
                    "view_sha256",
                ],
            )?;
            let consumed = tickets
                .get(&number(cause, "request_sequence")?)
                .ok_or_else(|| error("role cause lacks actual P/C consumption"))?;
            ensure(
                number(cause, "request_epoch")? == root.process_epoch
                    && cause["frame_sequence"] == consumed["origin"]["frame_sequence"]
                    && cause["question_sha256"] == consumed["question_sha256"]
                    && cause["view_sha256"] == consumed["view"]["view_digest"],
                "actual role continuation source",
            )?;
        }
        let cause = &d["cpu_cause"];
        if !cause.is_null() {
            fields(
                cause,
                &[
                    "job_sequence",
                    "native_epoch",
                    "native_sequence",
                    "frame_sequence",
                    "question_sha256",
                    "observation",
                ],
            )?;
            ensure(
                number(cause, "job_sequence")? > 0
                    && number(cause, "frame_sequence")? > 0
                    && executions.contains(&(
                        number(cause, "native_epoch")?,
                        number(cause, "native_sequence")?,
                    )),
                "CPU continuation original execution",
            )?;
            hash_field(cause, "question_sha256")?;
            number(cause, "observation")?;
            sources.insert((
                number(cause, "frame_sequence")?,
                string(cause, "question_sha256")?.into(),
            ));
        }
    }
    let action = &body["action"];
    let mut allowed = dependencies;
    allowed.insert((
        number(action, "frame_sequence")?,
        hash_field(action, "question_sha256")?.into(),
    ));
    ensure(
        sources.iter().all(|v| allowed.contains(v)),
        "semantic source lacks original action/dependency",
    )?;
    let mut r_keys = BTreeSet::new();
    for r in array(value, "r_selected_pc_consumption", 64)? {
        fields(
            r,
            &[
                "stage",
                "authority",
                "frame_sequence",
                "question_sha256",
                "submitted_view",
                "submitted_full_view_sha256",
                "job",
                "owner",
                "store_execution",
                "observation",
                "native_execution_epoch",
                "native_execution_sequence",
                "capture_sequence",
                "binding_sha256",
                "capture_full_view_sha256",
                "capture_view_seal_sha256",
                "slice_work",
                "selected_pc_ticket",
                "selected_output_sha256",
                "selected_record",
                "selected_full_view_sha256",
            ],
        )?;
        let frame = number(r, "frame_sequence")?;
        let question = hash_field(r, "question_sha256")?;
        ensure(
            string(r, "stage")? == "SelectedPcViewConsumed"
                && r["authority"] == authority(root)
                && allowed.contains(&(frame, question.to_owned()))
                && executions.contains(&(
                    number(r, "native_execution_epoch")?,
                    number(r, "native_execution_sequence")?,
                ))
                && r_keys.insert((number(r, "store_execution")?, number(r, "observation")?)),
            "actual R consumption execution/lineage",
        )?;
        checked_view(&r["submitted_view"], &authority(root))?;
        for key in [
            "submitted_full_view_sha256",
            "binding_sha256",
            "capture_full_view_sha256",
            "capture_view_seal_sha256",
        ] {
            hash_field(r, key)?;
        }
        ensure(
            number(r, "capture_sequence")? > 0 && number(r, "owner")? <= u16::MAX as u64,
            "actual R owner/capture",
        )?;
        let job = &r["job"];
        fields(
            job,
            &[
                "authority",
                "sequence",
                "state_owner",
                "state_revision",
                "state_sha256",
                "question_sha256",
                "root_restriction_sha256",
                "evaluation_namespace_sha256",
                "search_conditions_sha256",
                "node_credit",
                "deadline_clock",
                "deadline_tick",
            ],
        )?;
        ensure(
            job["authority"] == authority(root)
                && number(job, "sequence")? > 0
                && job["question_sha256"] == r["question_sha256"]
                && job["evaluation_namespace_sha256"] == r["binding_sha256"]
                && number(job, "deadline_clock")? == root.deadline_clock
                && number(job, "deadline_tick")? == root.deadline_tick,
            "actual R job source/deadline/value binding",
        )?;
        for key in ["state_owner", "state_revision", "node_credit"] {
            number(job, key)?;
        }
        for key in [
            "state_sha256",
            "root_restriction_sha256",
            "search_conditions_sha256",
        ] {
            hash_field(job, key)?;
        }
        let work = &r["slice_work"];
        fields(work, &["nodes", "quiescence_nodes", "tt_hits"])?;
        ensure(
            number(work, "quiescence_nodes")? <= number(work, "nodes")?
                && number(work, "nodes")? <= number(job, "node_credit")?,
            "actual R work excludes double qsearch charge",
        )?;
        number(work, "tt_hits")?;
        let selected = &r["selected_pc_ticket"];
        ensure(
            !selected.is_null(),
            "actual R feedback requires selected P/C consumption",
        )?;
        if !selected.is_null() {
            verify_ticket(selected, root, physical.high_water)?;
            let consumed = tickets
                .get(&number(selected, "request_sequence")?)
                .ok_or_else(|| error("R selected P/C missing actual consumed head"))?;
            ensure(
                *selected == consumed["origin"]["ticket"]
                    && r["selected_output_sha256"] == consumed["origin"]["output_sha256"]
                    && r["selected_full_view_sha256"] == consumed["full_view_sha256"],
                "R actual selected P/C join",
            )?;
            fields(&r["selected_record"], &["revision", "origin_state"])?;
            number(&r["selected_record"], "revision")?;
            number(&r["selected_record"], "origin_state")?;
        } else {
            ensure(
                r["selected_output_sha256"].is_null()
                    && r["selected_record"].is_null()
                    && r["selected_full_view_sha256"].is_null(),
                "R selected head absence",
            )?;
        }
        sources.insert((frame, question.into()));
    }
    let recipient = route(string(body, "recipient")?)?;
    ensure(
        if recipient == 6 {
            !r_keys.is_empty()
        } else {
            !tickets.is_empty()
        },
        "route lacks actual semantic consumption",
    )?;
    let mut witnesses = BTreeSet::new();
    let mut by_id = BTreeMap::new();
    for witness in array(value, "new_witnesses", 64)? {
        fields(
            witness,
            &[
                "kind",
                "observation",
                "identity_sha256",
                "raw_evidence_sha256",
                "source_question_sha256",
                "source_frame_sequence",
                "line",
            ],
        )?;
        let id = hash_field(witness, "identity_sha256")?.to_owned();
        ensure(
            !baseline.contains(&id) && witnesses.insert(id.clone()),
            "new conditional witness duplicates baseline/another witness",
        )?;
        ensure(
            sources.contains(&(
                number(witness, "source_frame_sequence")?,
                hash_field(witness, "source_question_sha256")?.to_owned(),
            )),
            "conditional witness lacks original committed source",
        )?;
        ensure(
            matches!(string(witness, "kind")?, "Counter" | "Repair"),
            "actual conditional witness kind",
        )?;
        number(witness, "observation")?;
        hash_field(witness, "raw_evidence_sha256")?;
        let line = array(witness, "line", 256)?;
        ensure(!line.is_empty(), "actual conditional witness line")?;
        let moves = line
            .iter()
            .map(|v| {
                packed(
                    u16::try_from(v.as_u64().ok_or_else(|| error("witness Move16"))?)
                        .map_err(error)?,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        rules::reference_line_from_rules(root_rules, &moves).map_err(error)?;
        by_id.insert(id, (witness, moves));
    }
    let mut bodies = BTreeSet::new();
    if !value["witness_preimages"].is_null() {
        for b in array(value, "witness_preimages", 64)? {
            fields(
                b,
                &[
                    "observation",
                    "kind",
                    "identity_sha256",
                    "source_question_sha256",
                    "source_frame_sequence",
                    "raw_evidence_sha256",
                    "preimage_bytes",
                    "preimage_hex",
                ],
            )?;
            let id = hash_field(b, "identity_sha256")?;
            let (witness, moves) = by_id
                .get(id)
                .ok_or_else(|| error("body has no original witness"))?;
            ensure(bodies.insert(id.to_owned()), "duplicate conditional body")?;
            for key in [
                "observation",
                "kind",
                "identity_sha256",
                "source_question_sha256",
                "source_frame_sequence",
                "raw_evidence_sha256",
            ] {
                ensure(b[key] == witness[key], "body/original witness tuple")?;
            }
            let bytes = unhex(string(b, "preimage_hex")?)?;
            ensure(
                bytes.len() <= 16 << 20 && bytes.len() as u64 == number(b, "preimage_bytes")?,
                "bounded original witness body bytes",
            )?;
            use rz_search::pals::engine::adaptive_port::utility_route::{
                UtilityWitnessPreimageExpectation, validate_utility_witness_preimage,
            };
            use rz_search::pals::engine::adaptive_port::validator_port::ActionWitnessKind;
            validate_utility_witness_preimage(
                &bytes,
                UtilityWitnessPreimageExpectation {
                    root: &root_rules.snapshot(),
                    kind: if string(b, "kind")? == "Counter" {
                        ActionWitnessKind::Counter
                    } else {
                        ActionWitnessKind::Repair
                    },
                    line: moves,
                    source_question: hash32(string(b, "source_question_sha256")?)?,
                    source_frame_sequence: number(b, "source_frame_sequence")?,
                    identity: hash32(id)?,
                    raw_evidence: hash32(string(b, "raw_evidence_sha256")?)?,
                },
            )
            .map_err(error)?;
        }
    }
    // A framing-valid old preimage does not contain all ContextWDL/CPU
    // comparison facts. The additional original Store basis is mandatory for
    // a known comparison and is verified separately from the old raw body.
    let basis_complete = verify_witness_basis(
        value,
        &by_id,
        root_rules,
        root,
        model,
        original_store_byte_limit,
    )?;
    Ok(Semantic {
        witnesses,
        bodies_complete: bodies.len() == by_id.len() && basis_complete,
    })
}

fn verify_historical_input(
    value: &Value,
    model: &v2::Model,
    frozen: &UtilityCohortFrozenDecision,
    root: &Position,
) -> Result<CheckedInput, ArenaError> {
    fields(
        value,
        &[
            "reference_json",
            "input_sha256",
            "target_sha256",
            "submission_target_sha256",
            "context_sha256",
            "target_record_ids",
            "original_view_sha256",
            "original_full_view_sha256",
            "materialized_inside_complete_scope",
        ],
    )?;
    ensure(
        !boolean(value, "materialized_inside_complete_scope")?,
        "unbound historical input is not measured materialization",
    )?;
    hash_field(value, "original_full_view_sha256")?;
    let text = string(value, "reference_json")?;
    ensure(
        text.len() <= v2::utility_capture::MAX_INPUT_BYTES,
        "original historical input capacity",
    )?;
    let reference: v2::utility::Reference = decode(text.as_bytes())?;
    let config = &model.base_checkpoint().config;
    ensure(
        reference.schema == v2::utility::INPUT_SCHEMA
            && reference.base_input.schema == v2::INPUT_SCHEMA
            && reference.base_input.board_features == rules::reference_board_features(root)
            && reference.base_input.side_to_move
                == if root.side_to_move() == rz_position::Color::White {
                    "white"
                } else {
                    "black"
                }
            && same(
                &reference.base_input.local_line,
                &rules::reference_line_from_rules(root, &[]).map_err(error)?,
            )?
            && !reference.base_input.context_lines.is_empty()
            && reference.base_input.context_lines.len() <= 64
            && reference.base_input.context_lines.len() <= config.max_lines,
        "original unbound root/input schema",
    )?;
    let ids = &reference.target_ids;
    ensure(
        !ids.is_empty()
            && ids.len() <= 64
            && ids.windows(2).all(|p| p[0] < p[1])
            && ids
                .iter()
                .all(|id| *id < reference.base_input.context_lines.len() as u64)
            && same(&value["target_record_ids"], ids)?,
        "original unbound target ids",
    )?;
    for (index, line) in reference.base_input.context_lines.iter().enumerate() {
        ensure(
            line.id == index as u64 && line.moves.len() <= config.max_plies,
            "original unbound context line capacity/order",
        )?;
        for movement in &line.moves {
            numerical_move(movement)?;
        }
    }
    for edge in &reference.base_input.context_edges {
        ensure(
            edge.source < reference.base_input.context_lines.len() as u64
                && edge.target < reference.base_input.context_lines.len() as u64
                && edge.source != edge.target
                && ids.contains(&edge.source) == ids.contains(&edge.target),
            "original unbound target relation",
        )?;
    }
    let context = v2::context_sha256(
        &reference.base_input.context_lines,
        &reference.base_input.context_edges,
    )
    .map_err(error)?;
    let weights = model
        .identity()
        .weights_sha256
        .as_ref()
        .ok_or_else(|| error("original model semantic identity"))?;
    let mut h = Sha256::new();
    h.update(v2::utility_capture::INPUT_CAPTURE_DOMAIN.as_bytes());
    h.update([0]);
    h.update(weights.as_bytes());
    h.update([0]);
    h.update((text.len() as u64).to_le_bytes());
    h.update(text.as_bytes());
    let input = format!("{:x}", h.finalize());
    let mut h = Sha256::new();
    h.update(v2::utility_capture::TARGET_CAPTURE_DOMAIN.as_bytes());
    h.update([0]);
    h.update(context.as_bytes());
    h.update([0]);
    h.update((ids.len() as u32).to_le_bytes());
    for id in ids {
        h.update(id.to_le_bytes());
    }
    let target = format!("{:x}", h.finalize());
    let mut h = Sha256::new();
    h.update(v2::utility_capture::SUBMISSION_TARGET_CAPTURE_DOMAIN.as_bytes());
    h.update([0]);
    h.update(unhex(hash_field(value, "original_view_sha256")?)?);
    h.update(target.as_bytes());
    let submitted = format!("{:x}", h.finalize());
    ensure(
        input == hash_field(value, "input_sha256")?
            && target == hash_field(value, "target_sha256")?
            && context == hash_field(value, "context_sha256")?
            && submitted == hash_field(value, "submission_target_sha256")?
            && input == frozen.input_sha256
            && target == frozen.target_sha256
            && context == frozen.context_sha256
            && text.len() as u64 == frozen.input_bytes
            && *ids == frozen.target_record_ids
            && frozen.root_rules_sha256 == exact_rules_anchor(root)?.rules_state_sha256,
        "original unbound input/target/frozen seals",
    )?;
    // This historical-only capture lacks source anchors. It cannot satisfy
    // semantic consumption or issue known labels; the measured materialized
    // branch separately replays every source and line before it is comparable.
    Ok(CheckedInput {
        reference_json: text.into(),
        input_sha256: input,
        target_sha256: target,
        submission_target_sha256: submitted,
        context_sha256: context,
    })
}

fn verify_witness_basis(
    semantic: &Value,
    witnesses: &BTreeMap<String, (&Value, Vec<rz_position::BoardMove>)>,
    position: &Position,
    root: &RoventRootAdmissionEvidence,
    model: &v2::Model,
    original_store_byte_limit: u64,
) -> Result<bool, ArenaError> {
    let value = &semantic["witness_basis"];
    if value.is_null() {
        return Ok(false);
    }
    let mut seen = BTreeSet::new();
    let mut complete = true;
    for row in array(semantic, "witness_basis", 64)? {
        fields(
            row,
            &[
                "observation",
                "kind",
                "identity_sha256",
                "source_question_sha256",
                "source_frame_sequence",
                "raw_evidence_sha256",
                "basis",
                "unavailable_reason",
            ],
        )?;
        let id = hash_field(row, "identity_sha256")?;
        let (witness, moves) = witnesses
            .get(id)
            .ok_or_else(|| error("basis has no original witness"))?;
        ensure(seen.insert(id.to_owned()), "duplicate conditional basis")?;
        for key in [
            "observation",
            "kind",
            "identity_sha256",
            "source_question_sha256",
            "source_frame_sequence",
            "raw_evidence_sha256",
        ] {
            ensure(row[key] == witness[key], "basis/original witness tuple")?;
        }
        let basis = &row["basis"];
        if basis.is_null() {
            ensure(
                matches!(
                    string(row, "unavailable_reason")?,
                    "missing_original_comparison"
                        | "missing_resident_evidence"
                        | "incomplete_context"
                        | "namespace_mismatch"
                        | "capacity"
                        | "invalid_original_basis"
                ),
                "explicit missing basis reason",
            )?;
            complete = false;
            continue;
        }
        ensure(
            row["unavailable_reason"].is_null(),
            "present original basis cannot be unavailable",
        )?;
        fields(basis, &["domain", "sha256", "bytes", "original_json"])?;
        let text = string(basis, "original_json")?;
        ensure(
            text.len() <= 16 << 20
                && text.len() as u64 == number(basis, "bytes")?
                && string(basis, "domain")? == "rz-rovent-utility-cohort-conditional-basis/1",
            "bounded original conditional basis bytes/domain",
        )?;
        let body = array(semantic, "witness_preimages", 64)?
            .iter()
            .find(|b| b["identity_sha256"] == row["identity_sha256"])
            .ok_or_else(|| error("basis lacks original raw preimage"))?;
        let preimage = unhex(string(body, "preimage_hex")?)?;
        use rz_search::pals::engine::adaptive_port::utility_route::{
            UtilityWitnessBasisExpectation, UtilityWitnessPreimageExpectation,
            validate_utility_witness_basis,
        };
        use rz_search::pals::engine::adaptive_port::validator_port::ActionWitnessKind;
        let metadata = validate_utility_witness_basis(
            text.as_bytes(),
            UtilityWitnessBasisExpectation {
                witness: UtilityWitnessPreimageExpectation {
                    root: &position.snapshot(),
                    kind: if string(row, "kind")? == "Counter" {
                        ActionWitnessKind::Counter
                    } else {
                        ActionWitnessKind::Repair
                    },
                    line: moves,
                    source_question: hash32(string(row, "source_question_sha256")?)?,
                    source_frame_sequence: number(row, "source_frame_sequence")?,
                    identity: hash32(id)?,
                    raw_evidence: hash32(string(row, "raw_evidence_sha256")?)?,
                },
                observation: rz_search::pals::store::ObservationId(
                    usize::try_from(number(row, "observation")?).map_err(error)?,
                ),
                basis_sha256: hash32(hash_field(basis, "sha256")?)?,
                preimage: &preimage,
            },
        )
        .map_err(error)?;
        // The owning decoder above validates its complete recipe, actual
        // consumed Task/accepted-origin framing, arithmetic and Rules replay.
        // Read this same validated body solely to join the original trial;
        // no raw JSON here becomes a Store or current acceptance capability.
        let parsed: Value = decode(text.as_bytes())?;
        let expected = serde_json::json!({"epoch":root.process_epoch,"game":root.game_generation,"root":root.root_generation,
            "implementation":hash32(&root.implementation_sha256)?.0});
        ensure(
            parsed["authority"] == expected
                && parsed["root_state"] == serde_json::json!(root.store_root_state),
            "conditional basis original trial authority/Store root",
        )?;
        ensure(
            metadata.root_state().0 == root.store_root_state
                && metadata.root().same_state(&position.snapshot())
                && metadata.authority().epoch.0 == root.process_epoch
                && metadata.authority().game.0 == root.game_generation
                && metadata.authority().root.0 == root.root_generation
                && metadata.authority().implementation == hash32(&root.implementation_sha256)?
                && metadata.root_budget() == hash32(&root.root_budget_sha256)?
                && metadata.original_store_byte_limit() == original_store_byte_limit,
            "typed conditional basis original root/history/authority",
        )?;
        join_basis_comparison(&parsed["comparison"], &expected, root, model, semantic, 0)?;
    }
    Ok(complete && seen.len() == witnesses.len())
}

fn join_basis_comparison(
    comparison: &Value,
    expected: &Value,
    root: &RoventRootAdmissionEvidence,
    model: &v2::Model,
    semantic: &Value,
    depth: u8,
) -> Result<(), ArenaError> {
    ensure(depth <= 8, "conditional basis lineage bound")?;
    match string(comparison, "kind")? {
        "Model" => {
            for origin in array(comparison, "origins", 2)? {
                ensure(
                    origin["authority"] == *expected
                        && number(origin, "request_epoch")? == root.process_epoch
                        && origin["model"]
                            == serde_json::json!(hash32(&root.native_origin.weights_sha256)?.0)
                        && origin["encoding"]
                            == serde_json::json!(hash32(&root.native_origin.encoding_sha256)?.0)
                        && number(origin, "frozen_epoch")? == root.native_origin.frozen_epoch,
                    "basis accepted endpoint original P/C namespace",
                )?;
                // An original consumed endpoint belongs to this action's
                // explicit consumption journal, not an authored consumed bit.
                let consumed = array(semantic, "pc_consumption", 64)?
                    .iter()
                    .find(|c| {
                        number(&c["origin"]["ticket"], "request_sequence").ok()
                            == number(origin, "request_sequence").ok()
                    })
                    .ok_or_else(|| error("basis endpoint lacks actual consumed P/C source"))?;
                let ticket = &consumed["origin"]["ticket"];
                let representation = &ticket["representation"];
                for (basis_key, ticket_key) in [
                    ("question", "question_sha256"),
                    ("output", "output_sha256"),
                    ("origin_seal", "seal_sha256"),
                ] {
                    let source = if basis_key == "question" {
                        ticket
                    } else {
                        &consumed["origin"]
                    };
                    ensure(
                        origin[basis_key]
                            == serde_json::json!(hash32(hash_field(source, ticket_key)?)?.0),
                        "basis endpoint actual native source/output seal",
                    )?;
                }
                ensure(
                    origin["view"]
                        == serde_json::json!(
                            hash32(hash_field(&ticket["view"], "view_digest")?)?.0
                        )
                        && origin["full_view"]
                            == serde_json::json!(
                                hash32(hash_field(consumed, "full_view_sha256")?)?.0
                            )
                        && number(origin, "frame_sequence")?
                            == number(&consumed["origin"], "frame_sequence")?,
                    "basis endpoint original question/view lineage",
                )?;
                for key in [
                    "input",
                    "history",
                    "candidates",
                    "mask_positions",
                    "model",
                    "encoding",
                ] {
                    ensure(
                        origin[key]
                            == serde_json::json!(hash32(hash_field(representation, key)?)?.0),
                        "basis endpoint actual native representation",
                    )?;
                }
                ensure(
                    string(origin, "precision")?
                        == string(representation, "precision")?.to_ascii_lowercase()
                        && number(origin, "revision")?
                            == number(representation, "record_revision")?
                        && number(origin, "view_sequence")? == number(&ticket["view"], "sequence")?
                        && number(origin, "deadline_clock")? == root.deadline_clock
                        && number(origin, "deadline_at")? == root.deadline_tick,
                    "basis endpoint actual precision/revision/deadline/view sequence",
                )?;
            }
        }
        "Finite" => join_basis_recipe(
            &comparison["recipe"],
            expected,
            root,
            model,
            semantic,
            depth,
        )?,
        _ => return Err(error("conditional basis comparison variant")),
    }
    Ok(())
}
fn join_basis_recipe(
    recipe: &Value,
    expected: &Value,
    root: &RoventRootAdmissionEvidence,
    model: &v2::Model,
    semantic: &Value,
    depth: u8,
) -> Result<(), ArenaError> {
    ensure(depth <= 8, "conditional finite recipe lineage bound")?;
    for key in ["before", "after"] {
        let endpoint = &recipe[key];
        match string(endpoint, "kind")? {
            "Rules" => {}
            "Cpu" => {
                ensure(
                    same(&endpoint["identity"], model.identity())?,
                    "basis CPU endpoint original registered R value",
                )?;
                let context = &endpoint["namespace"];
                match string(context, "kind")? {
                    "Legacy" => {
                        string(context, "condition")?;
                    }
                    "Contextual" => {
                        let c = &context["context"];
                        ensure(
                            c["authority"] == *expected
                                && number(c, "generation")? == root.root_generation
                                && number(c, "root")? == root.store_root_state as u64
                                && number(c, "deadline_clock")? == root.deadline_clock
                                && number(c, "deadline_at")? == root.deadline_tick,
                            "basis CPU original contextual root/deadline",
                        )?;
                        let consumed = array(semantic, "r_selected_pc_consumption", 64)?
                            .iter()
                            .find(|r| {
                                number(&r["job"], "sequence").ok() == number(c, "job_sequence").ok()
                                    && number(r, "store_execution").ok()
                                        == number(c, "execution").ok()
                                    && number(r, "observation").ok()
                                        == number(c, "observation").ok()
                            })
                            .ok_or_else(|| {
                                error("basis CPU endpoint lacks actual consumed R source")
                            })?;
                        ensure(
                            number(c, "frame_sequence")? == number(consumed, "frame_sequence")?
                                && number(c, "owner")? == number(consumed, "owner")?
                                && c["question"]
                                    == serde_json::json!(
                                        hash32(hash_field(consumed, "question_sha256")?)?.0
                                    )
                                && c["evaluation_namespace"]
                                    == serde_json::json!(
                                        hash32(hash_field(consumed, "binding_sha256")?)?.0
                                    ),
                            "basis CPU actual action/job/value join",
                        )?;
                    }
                    _ => return Err(error("basis CPU namespace variant")),
                }
            }
            _ => return Err(error("basis finite endpoint variant")),
        }
    }
    if !recipe["previous"].is_null() {
        join_basis_recipe(
            &recipe["previous"],
            expected,
            root,
            model,
            semantic,
            depth + 1,
        )?;
    }
    Ok(())
}

struct CheckedTrial {
    cost: rz_uci::pals_adaptive::UtilityCohortCostWire,
    digest: String,
    recipient: u8,
    support: [bool; 7],
    comparable: bool,
    witnesses: BTreeSet<String>,
    input: Option<CheckedInput>,
    ready: String,
    ready_recipients: Vec<u8>,
    authority: Value,
}

// Each independent original operand remains explicit at this historical
// boundary; combining them must not replace their separate pin/Rules joins.
#[allow(clippy::too_many_arguments)]
fn verify_trial(
    ordinal: usize,
    publication: &RoventUtilityCohortPublication,
    files: &BTreeMap<String, Vec<u8>>,
    profile: &RoventUtilityComparisonCohort,
    source: &RoventUtilityCohortCollectionSource,
    lock: &RoventInputLock,
    model: &v2::Model,
    root_rules: &Position,
    frozen: &UtilityCohortFrozenDecision,
) -> Result<CheckedTrial, ArenaError> {
    let pin = &publication.trials[ordinal];
    let raw = &files[&pin.path];
    let wire: Value = decode(raw)?;
    fields(
        &wire,
        &[
            "schema",
            "cohort_id",
            "trial_sequence",
            "recipient",
            "profile",
            "profile_sha256",
            "source",
            "original_root",
            "frozen_context",
            "materialized_sha256",
            "materialized_json",
            "historical_unbound_input",
            "unresolved_source",
            "cost",
            "ready_prelude",
            "physical_closure",
            "previous_durable_ack",
            "resolved",
            "comparable",
            "unknown_reason",
            "support_mask",
            "target_known_mask",
        ],
    )?;
    let recipient = route(string(&wire, "recipient")?)?;
    ensure(
        string(&wire, "schema")? == TRIAL_DOMAIN
            && string(&wire, "cohort_id")? == publication.cohort_id
            && number(&wire, "trial_sequence")? == ordinal as u64 + 1
            && same(&wire["profile"], profile)?
            && hash_field(&wire, "profile_sha256")? == hex(profile.digest()?)
            && same(&wire["source"], &source.source_pins)?
            && requested_routes(profile).contains(&recipient),
        "original trial order/profile/source/recipient",
    )?;
    let root: RoventRootAdmissionEvidence =
        serde_json::from_value(wire["original_root"].clone()).map_err(error)?;
    ensure(
        same(&root, &wire["original_root"])?,
        "closed original per-trial root",
    )?;
    verify_root(&root, &wire["original_root"], lock, &source.source_pins)?;
    verify_root_rules(&root, root_rules)?;
    let mut native = wire["original_root"]["native_origin"].clone();
    let mut original = source.native_origin.clone();
    native
        .as_object_mut()
        .ok_or_else(|| error("trial original native source object"))?
        .remove("process_epoch");
    original
        .as_object_mut()
        .ok_or_else(|| error("registered native source object"))?
        .remove("process_epoch");
    ensure(
        native == original
            && root.root_state.semantic == exact_rules_anchor(root_rules)?.rules_state_sha256,
        "independent trial kept original frozen source/Rules",
    )?;
    if ordinal == 0 {
        ensure(
            wire["original_root"] == source.registration_root,
            "first trial original registration root",
        )?;
    }
    if !wire["frozen_context"].is_null() {
        let bound: UtilityCohortFrozenDecision =
            serde_json::from_value(wire["frozen_context"].clone()).map_err(error)?;
        ensure(
            same(&bound, &wire["frozen_context"])?
                && bound == *frozen
                && bound.first_trial_sequence <= ordinal as u64 + 1,
            "original frozen decision chronology",
        )?;
    } else {
        ensure(
            ordinal as u64 + 1 < frozen.first_trial_sequence,
            "bound trial lost original frozen decision",
        )?;
    }
    let comparable = boolean(&wire, "comparable")?;
    let resolved = boolean(&wire, "resolved")?;
    let support = mask(&wire["support_mask"])?;
    ensure(
        mask(&wire["target_known_mask"])? == [false; 7]
            && support
                .iter()
                .enumerate()
                .all(|(i, v)| !v || i == recipient as usize)
            && (!resolved || support[recipient as usize]),
        "trial cannot create a pair label/support",
    )?;
    let reason = &wire["unknown_reason"];
    ensure(
        comparable == reason.is_null()
            && (reason.is_null()
                || reason
                    .as_str()
                    .is_some_and(|s| !s.is_empty() && s.len() <= 256)),
        "original comparable/unknown observation",
    )?;
    let cost = verify_cost(&wire["cost"], profile)?;
    ensure(
        cost.clock_epoch == root.deadline_clock
            && cost.original_deadline_tick == root.deadline_tick
            && cost.cohort_action_cutoff_tick <= root.deadline_tick
            && cost.cohort_cleanup_cutoff_tick <= root.root_descriptor.input.clock.cleanup.at_ns,
        "original cost/root deadline",
    )?;
    let prelude = verify_prelude(&wire["ready_prelude"], &root)?;
    ensure(
        cost.started_tick >= number(&wire["ready_prelude"], "captured_tick")?
            && cost.baseline_counters == wire["ready_prelude"]["counters"]
            && cost.native_credit_baseline
                == (
                    number(&wire["ready_prelude"]["native_credit"], "spent")?,
                    number(&wire["ready_prelude"]["native_credit"], "reserved")?,
                ),
        "original cost/prelude projection",
    )?;
    let previous = &wire["previous_durable_ack"];
    if ordinal == 0 {
        ensure(previous.is_null(), "first trial has no prior durable ACK")?;
    } else {
        fields(previous, &["trial_sequence", "trial_sha256", "file_sha256"])?;
        let p = &publication.trials[ordinal - 1];
        ensure(
            number(previous, "trial_sequence")? == ordinal as u64
                && hash_field(previous, "trial_sha256")?
                    == digest_bytes(TRIAL_DOMAIN, &files[&p.path])
                && hash_field(previous, "file_sha256")? == p.sha256,
            "original preceding durable checkpoint chain",
        )?;
    }
    let physical = verify_physical(&wire, &cost, model, Some(&root))?;
    let materialization_path = format!("materializations/{ordinal:06}.json");
    let materialized_pin = publication
        .materializations
        .iter()
        .find(|p| p.path == materialization_path);
    let mut input = None;
    let mut semantic = None;
    if !wire["materialized_json"].is_null() {
        let text = string(&wire, "materialized_json")?;
        let pin = materialized_pin.ok_or_else(|| error("original materialization file missing"))?;
        ensure(
            text.as_bytes() == files[&pin.path]
                && hash_field(&wire, "materialized_sha256")?
                    == digest_bytes(MATERIALIZATION_DOMAIN, text.as_bytes()),
            "original embedded materialization raw/domain SHA",
        )?;
        let body: Value = decode(text.as_bytes())?;
        fields(
            &body,
            &[
                "schema",
                "cohort_id",
                "trial_sequence",
                "recipient",
                "profile_sha256",
                "source",
                "original_root",
                "prelude",
                "input",
                "action",
                "semantic",
            ],
        )?;
        ensure(
            string(&body, "schema")? == MATERIALIZATION_DOMAIN
                && body["original_root"] == wire["original_root"]
                && body["prelude"] == wire["ready_prelude"],
            "materialization original root/prelude",
        )?;
        for key in [
            "cohort_id",
            "trial_sequence",
            "recipient",
            "profile_sha256",
            "source",
        ] {
            ensure(
                body[key] == wire[key],
                "materialization original trial join",
            )?;
        }
        let captured = verify_reference_input(&body["input"], model, frozen, root_rules)?;
        let action = &body["action"];
        fields(
            action,
            &[
                "authority",
                "frame_sequence",
                "question_sha256",
                "root_budget_sha256",
                "original_first_move",
                "deadline_clock",
                "deadline_tick",
                "frozen_context_sha256",
                "cost_epoch_sha256",
                "cost_scope",
            ],
        )?;
        ensure(
            action["authority"] == authority(&root)
                && number(action, "frame_sequence")? > 0
                && hash_field(action, "root_budget_sha256")? == root.root_budget_sha256
                && number(action, "deadline_clock")? == root.deadline_clock
                && number(action, "deadline_tick")? == root.deadline_tick
                && number(action, "original_first_move")? == u64::from(frozen.original_first_move)
                && string(action, "cost_scope")? == ROVENT_UTILITY_COHORT_COST_SCOPE,
            "original materialized action/root/first/deadline",
        )?;
        hash_field(action, "frozen_context_sha256")?;
        hash_field(action, "question_sha256")?;
        let offers = array(&wire["ready_prelude"], "offers", 3)?;
        let selected = offers
            .iter()
            .filter(|o| o["recipient"] == wire["recipient"])
            .collect::<Vec<_>>();
        ensure(selected.len() == 1, "one original selected route offer")?;
        verify_submitted_input_source(&body["input"], selected[0], &root)?;
        for key in [
            "frame_sequence",
            "question_sha256",
            "original_first_move",
            "deadline_clock",
            "deadline_tick",
        ] {
            ensure(
                selected[0][key] == action[key],
                "actual action differs from original ready offer",
            )?;
        }
        let mut h = Sha256::new();
        h.update(b"rz-rovent-utility-cohort-cost-epoch/1\0");
        h.update(unhex(&publication.cohort_id)?);
        h.update(profile.digest()?.0);
        h.update(unhex(&prelude.ready)?);
        h.update(captured.input_sha256.as_bytes());
        h.update(captured.target_sha256.as_bytes());
        h.update(ROVENT_UTILITY_COHORT_COST_SCOPE.as_bytes());
        let epoch = format!("{:x}", h.finalize());
        ensure(
            hash_field(action, "cost_epoch_sha256")? == epoch
                && cost.cost_epoch_sha256.as_deref() == Some(epoch.as_str()),
            "original cost epoch",
        )?;
        semantic = Some(verify_semantic(&body, &root, root_rules, &physical, model)?);
        input = Some(captured);
    } else {
        ensure(
            wire["materialized_sha256"].is_null() && materialized_pin.is_none() && !resolved,
            "absent materialization stays unresolved",
        )?;
    }
    if !wire["historical_unbound_input"].is_null() {
        let unbound =
            verify_historical_input(&wire["historical_unbound_input"], model, frozen, root_rules)?;
        if let Some(actual) = &input {
            ensure(
                actual.reference_json == unbound.reference_json
                    && actual.input_sha256 == unbound.input_sha256
                    && actual.target_sha256 == unbound.target_sha256
                    && actual.submission_target_sha256 == unbound.submission_target_sha256
                    && actual.context_sha256 == unbound.context_sha256,
                "original duplicate historical capture",
            )?;
        } else {
            input = Some(unbound);
        }
    }
    if !wire["unresolved_source"].is_null() {
        verify_unresolved(&wire["unresolved_source"], recipient)?;
        ensure(
            !resolved && !comparable,
            "unresolved route cannot be completed/comparable",
        )?;
    }
    if comparable {
        ensure(
            profile.input.usage == RoventUtilityCohortUsage::CollectionCandidate
                && resolved
                && cost.complete_scope
                && semantic.as_ref().is_some_and(|s| s.bodies_complete)
                && prelude.resource_known
                && physical.cache_known
                && physical.fresh_product
                && cost.physical_pin_zero_tick <= cost.cohort_cleanup_cutoff_tick,
            "comparable trial lacks actual complete known scope",
        )?;
        for key in [
            "nn_input_count_unknown",
            "accepted_nn_input_count_unknown",
            "consumed_nn_input_count_unknown",
            "rejected_events",
        ] {
            ensure(
                cost.baseline_counters[key] == cost.final_counters[key],
                "comparable trial has unknown/rejected actual work",
            )?;
        }
    }
    Ok(CheckedTrial {
        cost,
        digest: digest_bytes(TRIAL_DOMAIN, raw),
        recipient,
        support,
        comparable,
        witnesses: semantic.map(|s| s.witnesses).unwrap_or_default(),
        input,
        ready: prelude.ready,
        ready_recipients: prelude.recipients,
        authority: physical.authority,
    })
}

pub(in crate::rovent_collect) fn verify_unresolved(
    value: &Value,
    recipient: u8,
) -> Result<(), ArenaError> {
    fields(
        value,
        &[
            "question_sha256",
            "frame_sequence",
            "recipient",
            "root_budget_sha256",
            "frozen_context_sha256",
            "reason",
            "actual_pc_consumptions",
            "actual_dependencies",
            "actual_r_pc_consumptions",
            "new_witnesses",
        ],
    )?;
    for key in [
        "question_sha256",
        "root_budget_sha256",
        "frozen_context_sha256",
    ] {
        hash_field(value, key)?;
    }
    ensure(
        number(value, "frame_sequence")? > 0
            && route(string(value, "recipient")?)? == recipient
            && matches!(
                string(value, "reason")?,
                "cancelled"
                    | "deadline"
                    | "replaced_root"
                    | "semantic_failure"
                    | "physical_unknown"
                    | "missing_actual_consumption"
                    | "missing_exclusive_whole_cost"
            ),
        "original unresolved route/reason",
    )?;
    for key in [
        "actual_pc_consumptions",
        "actual_dependencies",
        "actual_r_pc_consumptions",
    ] {
        number(value, key)?;
    }
    for witness in array(value, "new_witnesses", 64)? {
        fields(
            witness,
            &["observation", "identity_sha256", "raw_evidence_sha256"],
        )?;
        number(witness, "observation")?;
        hash_field(witness, "identity_sha256")?;
        hash_field(witness, "raw_evidence_sha256")?;
    }
    Ok(())
}

type PairEvidence = (Vec<[u8; 2]>, Vec<String>, Vec<String>);
fn verify_pairs(
    publication: &RoventUtilityCohortPublication,
    files: &BTreeMap<String, Vec<u8>>,
    trials: &[CheckedTrial],
) -> Result<PairEvidence, ArenaError> {
    let mut known = Vec::new();
    let mut known_sha = Vec::new();
    let mut all_sha = Vec::new();
    let mut seen = BTreeSet::new();
    for pin in &publication.pairs {
        let raw = &files[&pin.path];
        let pair: Value = decode(raw)?;
        fields(
            &pair,
            &[
                "schema",
                "cohort_id",
                "left_trial_sequence",
                "right_trial_sequence",
                "left_trial_sha256",
                "right_trial_sha256",
                "left_recipient",
                "right_recipient",
                "preferred",
                "unknown_reason",
                "support_mask",
                "target_known_mask",
                "left_new_witnesses",
                "right_new_witnesses",
                "left_whole_wall_ns",
                "right_whole_wall_ns",
                "cost_scope",
                "objective",
            ],
        )?;
        ensure(
            string(&pair, "schema")? == PAIR_DOMAIN
                && string(&pair, "cohort_id")? == publication.cohort_id
                && string(&pair, "cost_scope")? == ROVENT_UTILITY_COHORT_COST_SCOPE
                && string(&pair, "objective")? == "exclusive_wall_nanoseconds",
            "original pair domain/cost scope",
        )?;
        let l = usize::try_from(number(&pair, "left_trial_sequence")?).map_err(error)?;
        let r = usize::try_from(number(&pair, "right_trial_sequence")?).map_err(error)?;
        ensure(
            (1..=trials.len()).contains(&l)
                && (1..=trials.len()).contains(&r)
                && seen.insert((l.min(r), l.max(r))),
            "original unordered comparison coverage",
        )?;
        let left = &trials[l - 1];
        let right = &trials[r - 1];
        for (side, trial) in [("left", left), ("right", right)] {
            ensure(
                hash_field(&pair, &format!("{side}_trial_sha256"))? == trial.digest
                    && route(string(&pair, &format!("{side}_recipient"))?)? == trial.recipient
                    && number(&pair, &format!("{side}_whole_wall_ns"))? == trial.cost.whole_wall_ns
                    && sha_set(&pair[format!("{side}_new_witnesses")], 64)? == trial.witnesses,
                "pair changed original trial/route/cost/witnesses",
            )?;
        }
        let (preferred, reason) =
            if l == r || left.digest == right.digest || left.authority == right.authority {
                (None, Some("duplicate_independent_trial"))
            } else if left.recipient == right.recipient {
                (None, Some("same_recipient"))
            } else if !left.comparable || !right.comparable {
                (None, Some("unresolved_or_masked_trial"))
            } else if left.witnesses.is_empty() && right.witnesses.is_empty() {
                (None, Some("no_new_valid_witness"))
            } else if left.witnesses == right.witnesses
                && left.cost.whole_wall_ns == right.cost.whole_wall_ns
            {
                (None, Some("equal_observation"))
            } else if left.witnesses.is_superset(&right.witnesses)
                && left.cost.whole_wall_ns <= right.cost.whole_wall_ns
            {
                (Some(left.recipient), None)
            } else if right.witnesses.is_superset(&left.witnesses)
                && right.cost.whole_wall_ns <= left.cost.whole_wall_ns
            {
                (Some(right.recipient), None)
            } else {
                (None, Some("pareto_incomparable"))
            };
        ensure(
            pair["unknown_reason"] == serde_json::json!(reason)
                && match preferred {
                    Some(p) => {
                        !pair["preferred"].is_null() && route(string(&pair, "preferred")?)? == p
                    }
                    None => pair["preferred"].is_null(),
                },
            "original Pareto comparison outcome",
        )?;
        let mut support = left.support;
        for (a, b) in support.iter_mut().zip(right.support) {
            *a |= b;
        }
        let mut target = [false; 7];
        if preferred.is_some() {
            target[left.recipient as usize] = true;
            target[right.recipient as usize] = true;
        }
        ensure(
            mask(&pair["support_mask"])? == support && mask(&pair["target_known_mask"])? == target,
            "pair action support/known are distinct",
        )?;
        let digest = digest_bytes(PAIR_DOMAIN, raw);
        all_sha.push(digest.clone());
        if let Some(preferred) = preferred {
            known.push([
                preferred,
                if preferred == left.recipient {
                    right.recipient
                } else {
                    left.recipient
                },
            ]);
            known_sha.push(digest);
        }
    }
    Ok((known, known_sha, all_sha))
}

/// Verify detached, independently pinned original cohort payloads. This is a
/// data report only: no JSON path reconstructs Store/native/training authority.
pub(super) fn verify_historical_utility_cohort(
    publication: &RoventUtilityCohortPublication,
    files: &BTreeMap<String, Vec<u8>>,
) -> Result<RoventHistoricalUtilityCohortEvidence, ArenaError> {
    publication.validate()?;
    let mut paths = BTreeSet::new();
    for pin in publication.payloads() {
        let raw = files
            .get(&pin.path)
            .ok_or_else(|| error("pinned original payload missing"))?;
        ensure(
            paths.insert(pin.path.as_str())
                && raw.len() as u64 == pin.bytes
                && sha(raw) == pin.sha256,
            "original raw payload bytes/SHA/coverage",
        )?;
    }
    ensure(
        files.len() == paths.len() && files.keys().all(|p| paths.contains(p.as_str())),
        "undeclared historical payload",
    )?;
    let profile = RoventUtilityComparisonCohort::from_json(
        std::str::from_utf8(&files[&publication.cohort_profile.path]).map_err(error)?,
    )?;
    let frozen_value: Value = decode(&files[&publication.frozen_context.path])?;
    ensure(
        !frozen_value.is_null(),
        "cohort has no original submitted frozen context",
    )?;
    let frozen: UtilityCohortFrozenDecision =
        serde_json::from_value(frozen_value.clone()).map_err(error)?;
    ensure(
        same(&frozen_value, &frozen)?
            && (1..=publication.trial_count).contains(&frozen.first_trial_sequence)
            && frozen.input_bytes > 0
            && frozen.input_bytes <= v2::utility_capture::MAX_INPUT_BYTES as u64
            && !frozen.target_record_ids.is_empty()
            && frozen.target_record_ids.len() <= 64
            && frozen.target_record_ids.windows(2).all(|p| p[0] < p[1])
            && frozen.target_record_ids.iter().all(|v| *v < 64),
        "original frozen decision shape",
    )?;
    for hash in [
        &frozen.input_sha256,
        &frozen.target_sha256,
        &frozen.context_sha256,
        &frozen.root_rules_sha256,
    ] {
        ensure(generation_sha(hash), "original frozen digest")?;
    }
    packed(frozen.original_first_move)?;
    let (source, lock, model, root_rules) = verify_source(publication, files, &profile)?;
    let mut trials = Vec::new();
    let mut authorities = BTreeSet::new();
    for ordinal in 0..publication.trials.len() {
        let trial = verify_trial(
            ordinal,
            publication,
            files,
            &profile,
            &source,
            &lock,
            &model,
            &root_rules,
            &frozen,
        )?;
        ensure(
            authorities.insert(serde_json::to_string(&trial.authority).map_err(error)?),
            "trials reused original authority",
        )?;
        trials.push(trial);
    }
    let ready = &trials[0].ready;
    ensure(
        trials.iter().all(|trial| trial.ready == *ready),
        "independent trials changed original allocation",
    )?;
    let planned = requested_routes(&profile)
        .into_iter()
        .filter(|r| trials[0].ready_recipients.contains(r))
        .collect::<Vec<_>>();
    ensure(
        trials.len() <= planned.len()
            && trials
                .iter()
                .zip(&planned)
                .all(|(trial, planned)| trial.recipient == *planned),
        "independent trial original requested order",
    )?;
    let registered: RoventRootAdmissionEvidence =
        serde_json::from_value(source.registration_root.clone()).map_err(error)?;
    let mut h = Sha256::new();
    h.update(b"rz-rovent-utility-cohort-owner/1\0");
    h.update(profile.digest()?.0);
    h.update(registered.root_descriptor.digest()?.0);
    h.update(unhex(ready)?);
    h.update(&planned);
    ensure(
        format!("{:x}", h.finalize()) == publication.cohort_id,
        "original cohort profile/root/allocation preimage",
    )?;
    let canonical = trials[frozen.first_trial_sequence as usize - 1]
        .input
        .as_ref()
        .ok_or_else(|| error("original first frozen capture unavailable"))?;
    let mut h = Sha256::new();
    h.update(b"rz-rovent-utility-cohort-cost-epoch/1\0");
    h.update(unhex(&publication.cohort_id)?);
    h.update(profile.digest()?.0);
    h.update(unhex(ready)?);
    h.update(canonical.input_sha256.as_bytes());
    h.update(canonical.target_sha256.as_bytes());
    h.update(ROVENT_UTILITY_COHORT_COST_SCOPE.as_bytes());
    let cost_epoch = format!("{:x}", h.finalize());
    ensure(
        trials.iter().all(|trial| {
            trial
                .cost
                .cost_epoch_sha256
                .as_deref()
                .is_none_or(|v| v == cost_epoch)
        }),
        "trial changed original cost epoch",
    )?;
    let (known_pairs, known_pair_sha256, all_pair_sha256) =
        verify_pairs(publication, files, &trials)?;
    let metadata = &source.collection_metadata;
    let split = *source
        .collection_manifest
        .opening_splits
        .get(&metadata.opening_id)
        .ok_or_else(|| error("original opening split"))?;
    let mut ancestors = source
        .ancestor_datasets
        .iter()
        .map(|v| v.publication_sha256.clone())
        .collect::<Vec<_>>();
    ancestors.sort();
    Ok(RoventHistoricalUtilityCohortEvidence {
        cohort_id: publication.cohort_id.clone(),
        profile_sha256: hex(profile.digest()?),
        usage: match profile.input.usage {
            RoventUtilityCohortUsage::ValidationOnly => "validation_only",
            RoventUtilityCohortUsage::CollectionCandidate => "collection_candidate",
        }
        .into(),
        collection_source_sha256: source.source_pins.source_sha256.clone(),
        cost_epoch_sha256: cost_epoch,
        resource_profile_sha256: source.source_pins.resource_plan_sha256.clone(),
        frozen_base_weights_sha256: source.source_pins.frozen_nnuec_model_sha256.clone(),
        input_reference_json: canonical.reference_json.clone(),
        input_sha256: canonical.input_sha256.clone(),
        target_sha256: canonical.target_sha256.clone(),
        submission_target_sha256: canonical.submission_target_sha256.clone(),
        utility_context_sha256: canonical.context_sha256.clone(),
        known_pairs,
        known_pair_sha256,
        all_pair_sha256,
        frozen_pc_sources: source.frozen_pc_sources,
        split,
        opening_id: metadata.opening_id.clone(),
        game_id: metadata.game_id.clone(),
        line_genealogy_id: metadata.line_genealogy_id.clone(),
        transposition_sha256: source.transposition_sha256,
        rules_state_sha256: exact_rules_anchor(&root_rules)?.rules_state_sha256,
        ancestor_dataset_publication_sha256: ancestors,
        ancestor_datasets: source.ancestor_datasets,
    })
}

pub(in crate::rovent_collect) fn restore_rules_snapshot(
    value: &Value,
) -> Result<Position, ArenaError> {
    let snapshot: RulesSnapshot = serde_json::from_value(value.clone()).map_err(error)?;
    snapshot.restore()
}
