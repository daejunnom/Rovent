//! Actual frozen P/C collection. The native adapter owns encoding, inference,
//! physical fences and delivery; this observer only seals bounded evidence.
use super::*;
use rz_contracts::RequestId;
use rz_eval::error::BackendError;
use rz_eval::pals_model::PalsRawOutput;
use rz_eval::pals_onnx::PalsOnnxConfig;
use rz_eval::runtime_pin::RuntimeLibraryPin;
use rz_experiments::PalsSearchPolicyIdentityV3;
use rz_search::pals::engine::{
    PALS_SEARCH_VERSION, POST_REPAIR_RECHECK_OBSERVER_VERSION, PalsCounters, PalsError, PalsResult,
    PostRepairRecheckPolicy, RecheckEndpoint, RecheckEndpointEvidence, RecheckFinished,
    RecheckIdentity, RecheckPrepared, ResolverPolicy, RoleAcceptance, RoleError,
    RoleLogicalContext, RoleQueryPurpose,
};
use rz_search::pals::store::{EvidenceScope, Observation, RawScore, TaskStatus};
use rz_uci::pals_native::{
    NativePreparedContext, NativeQueryKind, NativeRoleFinishHandle, NativeRoleModel,
    NativeRoleObserver, NativeRoleRejection,
};
use std::sync::{Arc, Mutex};

#[path = "continuation.rs"]
mod continuation;
#[path = "divergence.rs"]
mod divergence;
pub use continuation::PalsNativeContinuationRegistration;

const REGISTRY_VERSION: &str = "rz-pals-native-collection-registry/1";
const REFINEMENT_REGISTRATION_VERSION: &str = "rz-pals-native-refinement-registration/1";
const MAX_REFINEMENT_REGISTRATION_BYTES: usize = 32 * 1024;
const RAW_RESERVE: u64 = 128 * 1024;
const STAGE_RESERVE: u64 = 32 * 1024;
const RECHECK_DOMAIN: &str = "rz-pals-native-post-repair-recheck/1";
const RECHECK_ARTIFACT: &str = "native-recheck-traces.jsonl";
const RECHECK_ROWS: usize = 3;
const MAX_RECHECK_ATTEMPTS: usize = 128;
const MAX_RECHECK_ROW_BYTES: usize = MAX_JSON_RECORD_BYTES;
const SEARCH_RETURN_DOMAIN: &str = "rz-pals-native-observed-search-return/1";
const SEARCH_RETURN_ARTIFACT: &str = "native-search-returns.jsonl";
const SEARCH_RETURN_ROW_BYTES: usize = 64 * 1024;
const SEARCH_SUMMARY_ROW_BYTES: usize = 8 * 1024;
const SEARCH_RETURN_RESERVED_BYTES: u64 =
    (SEARCH_RETURN_ROW_BYTES * 2 + SEARCH_SUMMARY_ROW_BYTES) as u64;
const SEARCH_RETURN_HELD_ROWS: usize = 3;
const SEARCH_RETURN_MAX_HISTORY_PLIES: usize = 4096;

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PalsCollectionGraphPin {
    pub role: String,
    pub sha256: String,
    pub serialized_bytes: u64,
}

/// Supplied independently of the loaded model. No collector output can enroll
/// its own hashes. Checkpoint bytes are separately verified: an export epoch
/// declaration alone is not evidence that the actual checkpoint was checked.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PalsNativeCollectionRegistry {
    pub version: String,
    pub collector_binary_sha256: String,
    pub model_configuration: PalsModelConfig,
    pub model_configuration_sha256: String,
    pub checkpoint_sha256: String,
    pub export_manifest_sha256: String,
    pub graphs: Vec<PalsCollectionGraphPin>,
    pub runtime_sha256: String,
    pub encoding_sha256: String,
    pub encoder_source_sha256: String,
    pub cpu_configuration_sha256: String,
    pub frozen_epoch: u64,
    pub training_state: String,
    pub provider: String,
    pub intra_threads: usize,
    pub cache_public_memory: bool,
}
fn valid_sha(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
impl PalsNativeCollectionRegistry {
    pub fn validate(&self) -> Result<(), ArenaError> {
        self.model_configuration
            .validate()
            .map_err(|e| invalid(e.to_string()))?;
        if self.version != REGISTRY_VERSION
            || self.provider != "cpu"
            || self.training_state != "untrained"
            || self.frozen_epoch == 0
            || !(1..=2).contains(&self.intra_threads)
            || self.graphs.is_empty()
            || self.graphs.len() > 4
            || self.graphs.iter().any(|g| {
                !ident(&g.role)
                    || !valid_sha(&g.sha256)
                    || g.serialized_bytes == 0
                    || g.serialized_bytes > 256 * 1024 * 1024
            })
            || self
                .graphs
                .iter()
                .map(|g| &g.role)
                .collect::<BTreeSet<_>>()
                .len()
                != self.graphs.len()
            || [
                &self.collector_binary_sha256,
                &self.model_configuration_sha256,
                &self.checkpoint_sha256,
                &self.export_manifest_sha256,
                &self.runtime_sha256,
                &self.encoding_sha256,
                &self.encoder_source_sha256,
                &self.cpu_configuration_sha256,
            ]
            .iter()
            .any(|s| !valid_sha(s))
            || canonical_sha256(&self.model_configuration)? != self.model_configuration_sha256
        {
            return Err(invalid("invalid independent CPU PALS collection registry"));
        }
        Ok(())
    }
    pub fn read_pinned(path: &Path, expected_sha256: &str) -> Result<Self, ArenaError> {
        if !path.is_absolute()
            || !valid_sha(expected_sha256)
            || path
                .file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n == ".env" || n.starts_with(".env."))
        {
            return Err(invalid(
                "native collection registry requires an absolute public data file and hash",
            ));
        }
        if file_sha(path, 256 * 1024)? != expected_sha256 {
            return Err(invalid(
                "native collection registry bytes differ from independent pin",
            ));
        }
        let file = File::open(path).map_err(io)?;
        let mut bytes = Vec::new();
        file.take(256 * 1024 + 1)
            .read_to_end(&mut bytes)
            .map_err(io)?;
        if bytes.len() > 256 * 1024 || format!("{:x}", Sha256::digest(&bytes)) != expected_sha256 {
            return Err(invalid(
                "native collection registry changed during admission",
            ));
        }
        let value: Self = serde_json::from_slice(&bytes).map_err(|e| invalid(e.to_string()))?;
        value.validate()?;
        Ok(value)
    }
    pub fn owned_sources(&self) -> Result<PalsOwnedSources, ArenaError> {
        self.validate()?;
        let sources = PalsOwnedSources {
            cpu_binary_sha256: BTreeSet::from([self.collector_binary_sha256.clone()]),
            input_sources: BTreeSet::from([PalsInputSource::OwnPals {
                model_configuration_sha256: self.model_configuration_sha256.clone(),
                model_weights_sha256: self.checkpoint_sha256.clone(),
            }]),
        };
        sources.validate()?;
        Ok(sources)
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct RefinementRegistrationWire {
    version: String,
    base_registry_canonical_sha256: String,
    collector_binary_sha256: String,
    search_policy: PalsSearchPolicyIdentityV3,
}

// Keep both closed tuples in this source owner: legacy source review must not
// acquire an unpinned policy-selection implementation in the new module.
fn registered_refinement_policy(
    wire: &RefinementRegistrationWire,
) -> Result<PostRepairRecheckPolicy, ArenaError> {
    if wire.version == REFINEMENT_REGISTRATION_VERSION
        && wire.search_policy == PalsSearchPolicyIdentityV3::expected_same_repaired_line_once_v1()
    {
        Ok(PostRepairRecheckPolicy::SameRepairedLineOnceV1)
    } else if wire.version == "rz-pals-native-continuation-registration/1"
        && wire.search_policy
            == PalsSearchPolicyIdentityV3::expected_actual_opponent_continuation_v1()
    {
        Ok(PostRepairRecheckPolicy::ActualOpponentContinuationV1)
    } else {
        Err(invalid(
            "native registration version and selected policy differ",
        ))
    }
}

/// Immutable independent registration, admitted from the exact same bounded
/// bytes that were hashed and decoded. No public mutable fields or Deserialize
/// implementation can attach a retained pin to caller-constructed contents.
/// This is policy/source registration only, never a Reply execution witness.
#[derive(Clone, Debug)]
pub struct PalsNativeRefinementRegistration {
    wire: RefinementRegistrationWire,
    raw_sha256: String,
}
impl Serialize for PalsNativeRefinementRegistration {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.wire.serialize(serializer)
    }
}
impl PalsNativeRefinementRegistration {
    pub fn from_registration_bytes(
        bytes: &[u8],
        expected_sha256: &str,
    ) -> Result<Self, ArenaError> {
        if bytes.is_empty()
            || bytes.len() > MAX_REFINEMENT_REGISTRATION_BYTES
            || !valid_sha(expected_sha256)
            || format!("{:x}", Sha256::digest(bytes)) != expected_sha256
        {
            return Err(invalid(
                "refinement registration bounded bytes differ from the independent raw pin",
            ));
        }
        let text = std::str::from_utf8(bytes)
            .map_err(|_| invalid("refinement registration is not UTF-8"))?;
        let wire: RefinementRegistrationWire = crate::decode_json(text)?;
        if wire.version != REFINEMENT_REGISTRATION_VERSION
            || wire.search_policy
                != PalsSearchPolicyIdentityV3::expected_same_repaired_line_once_v1()
            || !valid_sha(&wire.base_registry_canonical_sha256)
            || !valid_sha(&wire.collector_binary_sha256)
        {
            return Err(invalid("invalid closed native refinement registration"));
        }
        wire.search_policy
            .validate()
            .map_err(|e| invalid(e.to_string()))?;
        Ok(Self {
            wire,
            raw_sha256: expected_sha256.into(),
        })
    }
    pub fn read_pinned(path: &Path, expected_sha256: &str) -> Result<Self, ArenaError> {
        let bytes = Self::read_registration_file(path, expected_sha256)?;
        Self::from_registration_bytes(&bytes, expected_sha256)
    }
    fn read_registration_file(path: &Path, expected_sha256: &str) -> Result<Vec<u8>, ArenaError> {
        let private_path = path.components().any(|part| {
            let name = part.as_os_str().to_string_lossy().to_ascii_lowercase();
            name == ".env"
                || name.starts_with(".env.")
                || name.contains("credential")
                || name.contains("service-account")
                || name.contains("service_account")
                || name.starts_with("id_ed25519")
                || name.starts_with("id_rsa")
                || name.ends_with(".key")
                || name.ends_with(".pem")
        });
        if !path.is_absolute() || private_path || !valid_sha(expected_sha256) {
            return Err(invalid(
                "refinement registration requires an absolute public data path and independent SHA256",
            ));
        }
        let file = File::open(path).map_err(io)?;
        let metadata = file.metadata().map_err(io)?;
        if !metadata.is_file()
            || metadata.len() == 0
            || metadata.len() > MAX_REFINEMENT_REGISTRATION_BYTES as u64
        {
            return Err(invalid(
                "refinement registration handle is not a bounded nonempty regular file",
            ));
        }
        let mut bytes = Vec::new();
        file.take(MAX_REFINEMENT_REGISTRATION_BYTES as u64 + 1)
            .read_to_end(&mut bytes)
            .map_err(io)?;
        Ok(bytes)
    }
    pub fn validate_against(&self, base: &PalsNativeCollectionRegistry) -> Result<(), ArenaError> {
        base.validate()?;
        self.wire
            .search_policy
            .validate()
            .map_err(|e| invalid(e.to_string()))?;
        if registered_refinement_policy(&self.wire).is_err()
            || !valid_sha(&self.raw_sha256)
            || self.wire.base_registry_canonical_sha256 != canonical_sha256(base)?
            || self.wire.collector_binary_sha256 != base.collector_binary_sha256
        {
            return Err(invalid(
                "refinement registration base registry or collector binary binding differs",
            ));
        }
        Ok(())
    }
    pub fn raw_sha256(&self) -> &str {
        &self.raw_sha256
    }
}

// Registration admission and actual collection dispatch must derive the same
// closed tuple from the selected policy, then validate the live engine getters.
fn observed_refinement_identity(
    policy: PostRepairRecheckPolicy,
    actual_search_identity: &str,
    actual_conditions: Option<&str>,
) -> Result<PalsSearchPolicyIdentityV3, ArenaError> {
    let expected = match policy {
        PostRepairRecheckPolicy::SameRepairedLineOnceV1 => {
            PalsSearchPolicyIdentityV3::expected_same_repaired_line_once_v1()
        }
        PostRepairRecheckPolicy::ActualOpponentContinuationV1 => {
            PalsSearchPolicyIdentityV3::expected_actual_opponent_continuation_v1()
        }
        PostRepairRecheckPolicy::Disabled => {
            return Err(invalid("disabled search has no refinement policy identity"));
        }
        PostRepairRecheckPolicy::FrozenModelWdlV2
        | PostRepairRecheckPolicy::IterativeFrozenModelWdlV2 => {
            return Err(invalid(
                "V4 frozen WDL policy cannot use a legacy refinement registration",
            ));
        }
    };
    let conditions = actual_conditions
        .ok_or_else(|| invalid("actual selected engine refinement conditions are absent"))?;
    let observed = PalsSearchPolicyIdentityV3 {
        version: expected.version,
        policy: expected.policy,
        search_identity: actual_search_identity.into(),
        conditions_sha256: Sha256::digest(conditions.as_bytes()).into(),
    };
    observed.validate().map_err(|e| invalid(e.to_string()))?;
    Ok(observed)
}

fn observed_refinement_selection(
    registration: Option<&PalsNativeRefinementRegistration>,
    actual_policy: PostRepairRecheckPolicy,
    actual_search_identity: &str,
    actual_conditions: Option<&str>,
) -> Result<Option<PalsSearchPolicyIdentityV3>, ArenaError> {
    match (registration, actual_policy) {
        (None, PostRepairRecheckPolicy::Disabled)
            if actual_search_identity == PALS_SEARCH_VERSION && actual_conditions.is_none() =>
        {
            Ok(None)
        }
        (Some(registration), policy) if policy != PostRepairRecheckPolicy::Disabled => {
            if registered_refinement_policy(&registration.wire)? != policy {
                return Err(invalid(
                    "actual selected policy differs from its registration lane",
                ));
            }
            let observed =
                observed_refinement_identity(policy, actual_search_identity, actual_conditions)?;
            if observed != registration.wire.search_policy {
                return Err(invalid(
                    "actual engine refinement getters differ from independently registered policy",
                ));
            }
            Ok(Some(observed))
        }
        _ => Err(invalid(
            "actual engine refinement selection differs from explicit independent registration",
        )),
    }
}

struct Call {
    input_sha256: String,
    physical: bool,
    delivered: bool,
    accepted: bool,
    rejected: bool,
    physical_unknown: bool,
    logical: Option<CallLogical>,
    prepaid_rows: usize,
}
// Bounded copied metadata only: no extra PositionSnapshot, tensor, sidecar or
// lineage bytes are retained. Exact bytes remain in the existing sealed rows.
struct CallLogical {
    kind: NativeQueryKind,
    context: RoleLogicalContext,
    root_sha256: String,
    position_sha256: String,
    prefix: Vec<BoardMove>,
    proposal: Vec<BoardMove>,
    counterexample: Option<Vec<BoardMove>>,
    legal: Vec<BoardMove>,
    input_row_sha256: String,
    sidecar_row_sha256: String,
    sidecar_sha256: String,
    lineage_row_sha256: String,
    canonical_tensor_sha256: String,
    chosen_first: Option<BoardMove>,
    accepted_context_checked: bool,
    producer_metadata_admitted: bool,
}
struct PendingRecheck {
    identity: RecheckIdentity,
    descriptor_sha256: String,
    prepared_payload_sha256: String,
    reply_context: RoleLogicalContext,
    anchor_sha256: String,
    repaired: Vec<BoardMove>,
    refutation: Vec<BoardMove>,
    limits: PalsLimits,
    deadline_tick: u64,
    parent_epoch: u64,
    parent_last_sequence: u64,
    reply: Option<RequestId>,
    bound_payload_sha256: Option<String>,
}
// Exclusive descriptor/return/legacy-summary credit. Native callbacks retain
// their separate Call.prepaid_rows and cannot consume any of these three slots.
struct PendingSearchReturn {
    descriptor_sha256: String,
    capture_sequence: u64,
    limits: PalsLimits,
    descriptor_pending: bool,
    return_pending: bool,
    summary_pending: bool,
}
#[derive(Clone, Copy)]
enum SearchReturnSlot {
    Descriptor,
    Returned,
    Summary,
}
struct Sink {
    context: Option<PalsNativeCaptureContext>,
    source: PalsCollectionSourceDescription,
    trace: PalsNativeTrace,
    started: Instant,
    calls: BTreeMap<RequestId, Call>,
    raw_sources: BTreeSet<String>,
    total_rows: usize,
    producer: Option<RegisteredProducerHandle>,
    pending_recheck: Option<PendingRecheck>,
    pending_continuation: Option<continuation::PendingContinuation>,
    rejected_recheck: Option<RecheckIdentity>,
    closed_rechecks: Vec<RecheckIdentity>,
    prepaid_recheck_rows: usize,
    prepaid_native_rows: usize,
    pending_search_return: Option<PendingSearchReturn>,
}
fn role_error(error: impl std::fmt::Display) -> RoleError {
    RoleError::Backend(failure_text(error))
}
fn search_after_summary<T>(
    primary: Result<T, rz_search::pals::engine::PalsError>,
    summary: Result<(), ArenaError>,
) -> Result<T, ArenaError> {
    match primary {
        Err(error) => Err(invalid(error.to_string())),
        Ok(value) => {
            summary?;
            Ok(value)
        }
    }
}
fn native_trace_secondary(trace: &mut PalsNativeTrace, stage: &str, error: impl std::fmt::Display) {
    let secondary = failure_text(error);
    let previous = trace.failure.take();
    trace.failure = Some(match previous {
        Some(primary) => format!("{primary}; {stage} secondary: {secondary}"),
        None => format!("{stage} secondary: {secondary}"),
    });
}
fn native_trace_failure(trace: &mut PalsNativeTrace, stage: &str, error: impl std::fmt::Display) {
    if trace.failure.is_some() {
        native_trace_secondary(trace, stage, error);
    } else {
        // Retain the original diagnostic verbatim when there is no earlier primary.
        trace.failure = Some(failure_text(error));
    }
}
fn search_return_tick(started: Instant, at: Instant) -> Result<u64, RoleError> {
    let elapsed = at
        .checked_duration_since(started)
        .ok_or_else(|| role_error("search return clock precedes native capture origin"))?;
    u64::try_from(elapsed.as_nanos()).map_err(|_| role_error("search return clock tick overflow"))
}
fn search_return_deadline(
    started: Instant,
    deadline: Instant,
) -> Result<serde_json::Value, RoleError> {
    let (relation, distance) = if let Some(distance) = deadline.checked_duration_since(started) {
        ("at_or_after_origin", distance)
    } else {
        ("before_origin", started.duration_since(deadline))
    };
    let distance = u64::try_from(distance.as_nanos())
        .map_err(|_| role_error("original search deadline tick overflow"))?;
    Ok(
        serde_json::json!({"origin":"native_capture_start","unit":"nanoseconds",
        "relation":relation,"distance":distance,"deadline_replaced":false}),
    )
}
fn search_return_counters(c: &PalsCounters) -> serde_json::Value {
    // Each scalar has its own macro expansion; keep every original counter key.
    let fields = [
        ("rounds", serde_json::json!(c.rounds)),
        ("proposals", serde_json::json!(c.proposals)),
        ("refutations", serde_json::json!(c.refutations)),
        ("repairs", serde_json::json!(c.repairs)),
        (
            "supported_refutations",
            serde_json::json!(c.supported_refutations),
        ),
        ("supported_repairs", serde_json::json!(c.supported_repairs)),
        ("role_calls", serde_json::json!(c.role_calls)),
        ("proposer_calls", serde_json::json!(c.proposer_calls)),
        ("critic_calls", serde_json::json!(c.critic_calls)),
        ("repair_calls", serde_json::json!(c.repair_calls)),
        (
            "completed_proposer_calls",
            serde_json::json!(c.completed_proposer_calls),
        ),
        (
            "completed_critic_calls",
            serde_json::json!(c.completed_critic_calls),
        ),
        (
            "completed_repair_calls",
            serde_json::json!(c.completed_repair_calls),
        ),
        (
            "consumed_role_outputs",
            serde_json::json!(c.consumed_role_outputs),
        ),
        (
            "accepted_proposer_outputs",
            serde_json::json!(c.accepted_proposer_outputs),
        ),
        (
            "accepted_critic_outputs",
            serde_json::json!(c.accepted_critic_outputs),
        ),
        (
            "accepted_repair_outputs",
            serde_json::json!(c.accepted_repair_outputs),
        ),
        ("value_calls", serde_json::json!(c.value_calls)),
        (
            "completed_value_calls",
            serde_json::json!(c.completed_value_calls),
        ),
        (
            "accepted_value_outputs",
            serde_json::json!(c.accepted_value_outputs),
        ),
        (
            "external_checker_tasks",
            serde_json::json!(c.external_checker_tasks),
        ),
        (
            "external_checker_reports",
            serde_json::json!(c.external_checker_reports),
        ),
        (
            "external_checker_nodes_observed",
            serde_json::json!(c.external_checker_nodes_observed),
        ),
        (
            "external_checker_work_incomplete",
            serde_json::json!(c.external_checker_work_incomplete),
        ),
        (
            "external_checker_node_budget_reserved",
            serde_json::json!(c.external_checker_node_budget_reserved),
        ),
        (
            "consumed_external_checker_tasks",
            serde_json::json!(c.consumed_external_checker_tasks),
        ),
        (
            "cpu_tasks_requested",
            serde_json::json!(c.cpu_tasks_requested),
        ),
        ("cpu_tasks", serde_json::json!(c.cpu_tasks)),
        (
            "cpu_work_observation_incomplete",
            serde_json::json!(c.cpu_work_observation_incomplete),
        ),
        ("cpu_nodes", serde_json::json!(c.cpu_nodes)),
        (
            "cpu_quiescence_nodes",
            serde_json::json!(c.cpu_quiescence_nodes),
        ),
        ("cpu_tt_hits", serde_json::json!(c.cpu_tt_hits)),
        (
            "completed_cpu_tasks",
            serde_json::json!(c.completed_cpu_tasks),
        ),
        (
            "partial_cpu_iterations",
            serde_json::json!(c.partial_cpu_iterations),
        ),
        (
            "consumed_cpu_tasks",
            serde_json::json!(c.consumed_cpu_tasks),
        ),
        (
            "reused_completed_cpu_tasks_consumed",
            serde_json::json!(c.reused_completed_cpu_tasks_consumed),
        ),
        (
            "consumed_partial_cpu_values",
            serde_json::json!(c.consumed_partial_cpu_values),
        ),
        (
            "consumed_frontier_cpu_values",
            serde_json::json!(c.consumed_frontier_cpu_values),
        ),
        (
            "consumed_cached_cpu_values",
            serde_json::json!(c.consumed_cached_cpu_values),
        ),
        (
            "evidence_cache_hits",
            serde_json::json!(c.evidence_cache_hits),
        ),
        ("examined_edges", serde_json::json!(c.examined_edges)),
        (
            "retained_situations",
            serde_json::json!(c.retained_situations),
        ),
        (
            "root_scope_observation_complete",
            serde_json::json!(c.root_scope_observation_complete),
        ),
        (
            "unknown_root_children",
            serde_json::json!(c.unknown_root_children),
        ),
    ];
    serde_json::Value::Object(
        fields
            .into_iter()
            .map(|(name, value)| (name.into(), value))
            .collect(),
    )
}

fn search_return_result(
    result: &Result<PalsResult, PalsError>,
) -> Result<serde_json::Value, RoleError> {
    match result {
        Err(error) => {
            let variant = match error {
                PalsError::InvalidConfig => "InvalidConfig",
                PalsError::InvalidLimits => "InvalidLimits",
                PalsError::Rules(_) => "Rules",
                PalsError::Cpu(_) => "Cpu",
                PalsError::Checker(_) => "Checker",
                PalsError::Role(_) => "Role",
                PalsError::Store(_) => "Store",
                PalsError::Review(_) => "Review",
                PalsError::Rovent(_) => "Rovent",
                PalsError::Capacity => "Capacity",
                PalsError::RoleCallLimit => "RoleCallLimit",
            };
            Ok(
                serde_json::json!({"kind":"Err","pals_error_variant":variant,"primary_display":failure_text(error),
                "primary_display_scope":"original-PalsError-bounded-failure-text;not-unbounded-Debug-copy"}),
            )
        }
        Ok(result) => {
            let best_move = result
                .best_move
                .map(|m| Move16::pack(m).map(|m| m.bits()))
                .transpose()
                .map_err(role_error)?;
            let elapsed_ns = u64::try_from(result.elapsed.as_nanos())
                .map_err(|_| role_error("engine search elapsed tick overflow"))?;
            Ok(
                serde_json::json!({"kind":"Ok","best_move":best_move,"score":result.score,
                "checker_identity":result.checker_identity,"value_scope":format!("{:?}",result.value_scope),
                "terminal":result.terminal.map(|t|format!("{:?}",t)),"completion":format!("{:?}",result.completion),
                "counters":search_return_counters(&result.counters),"root_value_count":result.root_values.len(),
                "elapsed_ns":elapsed_ns,"elapsed_scope":"engine-reported-search-duration;not-whole-action-cost",
                "model_identity":result.model_identity,"resolver_version":result.resolver_version,
                "projection_scope":"explicit-return-fields;resolved-value-and-root-value-bodies-not-serialized;no-utility-or-label-authority"}),
            )
        }
    }
}
fn recheck_identity_json(identity: RecheckIdentity) -> serde_json::Value {
    serde_json::json!({"game_generation":identity.game_generation,"search_generation":identity.search_generation,
        "root":{"slot":identity.root.slot,"generation":identity.root.generation},"root_revision":identity.root_revision,
        "repair_record_revision":identity.repair_record_revision,"repaired_line":identity.repaired_line.0})
}
fn logical_json(context: &RoleLogicalContext) -> Result<serde_json::Value, RoleError> {
    Ok(
        serde_json::json!({"game_generation":context.game_generation,"search_generation":context.search_generation,
        "situation":{"slot":context.situation.slot,"generation":context.situation.generation},"state":context.state.0,
        "focus":context.focus.0,"purpose":format!("{:?}",context.purpose),"prefix":pack(&context.prefix).map_err(role_error)?,
        "focus_sha256":hex(context.focus_sha256),"prefix_sha256":hex(context.prefix_sha256),"proposal_sha256":hex(context.proposal_sha256),
        "refutation_sha256":context.refutation_sha256.map(hex),"divergence_sha256":hex(context.divergence_sha256),
        "public_revision":context.public_revision,"situation_revision":context.situation_revision}),
    )
}
fn recheck_observation_json(observation: &Observation) -> serde_json::Value {
    let scope = match observation.scope {
        EvidenceScope::RulesTerminal => serde_json::json!({"kind":"RulesTerminal"}),
        EvidenceScope::DepthLimited {
            depth,
            profile,
            condition,
        } => {
            serde_json::json!({"kind":"DepthLimited","depth":depth,"profile":profile,"condition":condition})
        }
        EvidenceScope::ExternalUci {
            requested_depth,
            reported_depth,
            seldepth,
            bound,
        } => {
            serde_json::json!({"kind":"ExternalUci","requested_depth":requested_depth,"reported_depth":reported_depth,"seldepth":seldepth,"bound":format!("{:?}",bound)})
        }
        EvidenceScope::ProfiledExternalUci {
            requested,
            issued,
            reported_depth,
            seldepth,
            bound,
        } => {
            serde_json::json!({"kind":"ProfiledExternalUci","requested":requested,"issued":issued,
                "reported_depth":reported_depth,"seldepth":seldepth,"bound":format!("{:?}",bound)})
        }
        EvidenceScope::Model {
            model,
            encoding,
            input,
        } => serde_json::json!({"kind":"Model","model":model,"encoding":encoding,"input":input}),
    };
    let score = match observation.score {
        RawScore::Unknown => serde_json::json!({"kind":"Unknown"}),
        RawScore::Cpu {
            value,
            perspective,
            bound,
        } => {
            serde_json::json!({"kind":"Cpu","value":value,"white_perspective":perspective==Color::White,"bound":format!("{:?}",bound)})
        }
        RawScore::Estimate { value, perspective } => {
            serde_json::json!({"kind":"Estimate","value_f32_bits":value.to_bits(),"white_perspective":perspective==Color::White})
        }
        RawScore::Wdl {
            win,
            draw,
            loss,
            perspective,
        } => {
            serde_json::json!({"kind":"Wdl","value_f32_bits":[win.to_bits(),draw.to_bits(),loss.to_bits()],"white_perspective":perspective==Color::White})
        }
        RawScore::ContextWdl {
            win,
            draw,
            loss,
            perspective,
            context_revision,
        } => {
            serde_json::json!({"kind":"ContextWdl","value_f32_bits":[win.to_bits(),draw.to_bits(),loss.to_bits()],"white_perspective":perspective==Color::White,"context_revision":context_revision})
        }
        RawScore::ConditionalWdl {
            expectation,
            perspective,
            repaired,
            counter,
            context_revision,
        } => {
            serde_json::json!({"kind":"ConditionalWdl","expectation_f32_bits":expectation.to_bits(),"white_perspective":perspective==Color::White,"repaired":repaired.0,"counter":counter.0,"context_revision":context_revision})
        }
        RawScore::ConditionalRepairWdl {
            expectation,
            perspective,
            before,
            after,
            context_revision,
        } => {
            serde_json::json!({"kind":"ConditionalRepairWdl","expectation_f32_bits":expectation.to_bits(),"white_perspective":perspective==Color::White,"before":before.0,"after":after.0,"context_revision":context_revision})
        }
        RawScore::Terminal { winner } => {
            serde_json::json!({"kind":"Terminal","white_winner":winner.map(|c|c==Color::White)})
        }
        RawScore::ExternalUci {
            value,
            bound,
            perspective,
            wdl_per_mille,
        } => {
            serde_json::json!({"kind":"ExternalUci","raw_value":format!("{:?}",value),"bound":format!("{:?}",bound),"white_perspective":perspective==Color::White,"wdl_per_mille":wdl_per_mille})
        }
    };
    serde_json::json!({"state":observation.state.0,"line":observation.line.map(|id|id.0),"source":observation.source,"epoch":observation.epoch,
        "value_identity":observation.value_identity,"checker_identity":observation.checker_identity,
        "checker_work":observation.checker_work.map(|w|serde_json::json!({"nodes":w.nodes,"qnodes":w.qnodes,"tt_hits":w.tt_hits})),
        "external_report_debug":observation.external_report.as_ref().map(|r|format!("{:?}",r)),
        "model_value_identity":observation.model_value_identity.as_ref().map(|v|serde_json::json!({"semantics":v.semantics,"model":v.model,"encoding":v.encoding,"precision":v.precision,"model_epoch":hex(v.model_epoch)})),
        "model_value_input":observation.model_value_input.map(hex),"cpu_condition":observation.cpu_condition,"cpu_pv":observation.cpu_pv.map(|id|id.0),
        "scope":scope,"score":score,"budget":observation.budget,"kind":format!("{:?}",observation.kind),
        "supersedes":observation.supersedes.map(|id|id.0),"execution":observation.execution.map(|id|id.0)})
}
fn recheck_endpoint_json(
    endpoint: &RecheckEndpoint<'_>,
    actual: &Position,
) -> Result<serde_json::Value, RoleError> {
    if !endpoint.snapshot.same_state(&actual.snapshot()) {
        return Err(role_error(
            "recheck endpoint snapshot differs from exact Rules replay",
        ));
    }
    let evidence = match &endpoint.evidence {
        RecheckEndpointEvidence::Unobserved => serde_json::json!({"kind":"Unobserved"}),
        RecheckEndpointEvidence::ModelWdl {
            observation_id,
            observation,
            output,
            context_revision,
        } => {
            output.validate(actual, &output.identity)?;
            serde_json::json!({"kind":"ModelWdl","observation_id":observation_id.0,"observation":recheck_observation_json(observation),
                "context_revision":context_revision,"model_identity":{"semantics":output.identity.semantics,"model":output.identity.model,
                    "encoding":output.identity.encoding,"precision":output.identity.precision,"model_epoch":hex(output.identity.model_epoch)},
                "input_sha256":hex(output.input_sha256),"wdl":output.wdl,"white_perspective":output.perspective==Color::White,
                "value_authority":"fresh-frozen-model-WDL;not-owned-raw;not-Rules-proof"})
        }
        RecheckEndpointEvidence::InvalidProvenance { error } => {
            serde_json::json!({"kind":"InvalidProvenance","error":failure_text(error)})
        }
        RecheckEndpointEvidence::RulesTerminal {
            reason,
            value,
            perspective,
        } => {
            serde_json::json!({"kind":"RulesTerminal","reason":format!("{:?}",reason),"value":value,"white_perspective":*perspective==Color::White})
        }
        RecheckEndpointEvidence::OwnCpu {
            observation_id,
            execution_id,
            observation,
            task,
            admitted_scope,
        } => {
            let status = match task.status {
                TaskStatus::InFlight => serde_json::json!({"kind":"InFlight"}),
                TaskStatus::CancellationRequested => {
                    serde_json::json!({"kind":"CancellationRequested"})
                }
                TaskStatus::Paused {
                    checkpoint,
                    evidence,
                } => {
                    serde_json::json!({"kind":"Paused","checkpoint":checkpoint,"evidence":evidence.map(|id|id.0)})
                }
                TaskStatus::RetiredPaused {
                    checkpoint,
                    evidence,
                } => {
                    serde_json::json!({"kind":"RetiredPaused","checkpoint":checkpoint,"evidence":evidence.map(|id|id.0)})
                }
                TaskStatus::Completed(id) => {
                    serde_json::json!({"kind":"Completed","observation_id":id.0})
                }
                TaskStatus::Failed => serde_json::json!({"kind":"Failed"}),
            };
            serde_json::json!({"kind":"OwnCpu","observation_id":observation_id.0,"execution_id":execution_id.0,
                "admitted_scope":format!("{:?}",admitted_scope),"observation":recheck_observation_json(observation),
                "task":{"status":status,"resumed_from":task.resumed_from.map(|id|id.0),"key":{
                    "state":task.key.state.0,"line":task.key.line.map(|id|id.0),"question":format!("{:?}",task.key.question),
                    "root_moves":task.key.root_moves.iter().map(|m|m.bits()).collect::<Vec<_>>(),"model":task.key.model,"epoch":task.key.epoch,
                    "value_identity":task.key.value_identity,"checker_identity":task.key.checker_identity,"cpu_condition":task.key.cpu_condition,
                    "profile":task.key.profile,"condition":task.key.condition,"input_revision":task.key.input_revision,
                    "requested_depth":task.key.requested_depth,"node_budget":task.key.node_budget}},
                "task_consumer_observation":"private-consumers-not-observed; engine-borrowed-completed-provenance-only"})
        }
    };
    Ok(
        serde_json::json!({"state":endpoint.state.0,"situation":{"slot":endpoint.situation.slot,"generation":endpoint.situation.generation},
        "rules_state_sha256":state_sha(actual).map_err(role_error)?,"board_fen":actual.to_fen(),"evidence":evidence}),
    )
}
impl Sink {
    fn prepare_search_return(
        &mut self,
        limits: PalsLimits,
        cancel: &AtomicBool,
        native_before: serde_json::Value,
    ) -> Result<(), RoleError> {
        if !self.selected_recheck() || self.pending_search_return.is_some() {
            return Err(role_error(
                "unselected or duplicate observed search return preparation",
            ));
        }
        self.producer
            .as_ref()
            .ok_or_else(|| role_error("observed search return requires strict producer"))?
            .verify_source(&self.source)
            .map_err(role_error)?;
        let capture = self.context()?;
        if capture.actual_moves.len() > SEARCH_RETURN_MAX_HISTORY_PLIES {
            return Err(role_error(
                "observed search return actual history exceeds 4096 plies",
            ));
        }
        let snapshot = capture.position.snapshot();
        let replay = snapshot
            .uci_replay(SEARCH_RETURN_MAX_HISTORY_PLIES)
            .map_err(role_error)?;
        let at = Instant::now();
        let descriptor = serde_json::json!({"domain":SEARCH_RETURN_DOMAIN,"stage":"before_search",
            "game_id":capture.game,"capture_sequence":capture.sequence,"opening_id":capture.opening.id,
            "root":{"rules_version":snapshot.rules_version(),"variant":snapshot.variant(),
                "board_fen":snapshot.to_fen(),"rules_state_sha256":state_sha(&capture.position).map_err(role_error)?,
                "history_origin":format!("{:?}",replay.origin),"history_completeness":format!("{:?}",replay.completeness),
                "known_history_max_plies":SEARCH_RETURN_MAX_HISTORY_PLIES,"known_history_start_fen":replay.start_fen,
                "known_history_moves":pack(&replay.moves).map_err(role_error)?,
                "capture_declared_actual_moves":pack(&capture.actual_moves).map_err(role_error)?,
                "unknown_prefix_invented":false},
            "checked_source_sha256":canonical_sha256(&("rz-pals-collector-checked-source/1",&self.source)).map_err(role_error)?,
            "refinement_registration_sha256":self.source.native.as_ref().and_then(|n|n.get("refinement_registration_sha256")),
            "pals_search_policy":self.source.native.as_ref().and_then(|n|n.get("pals_search_policy")),
            "limits":{"max_rounds":limits.max_rounds,"max_cpu_nodes":limits.max_cpu_nodes,"cpu_depth":limits.cpu_depth,
                "original_deadline":search_return_deadline(self.started,limits.deadline)?},
            "observer_tick":search_return_tick(self.started,at)?,"observer_tick_origin":"native_capture_start","observer_tick_unit":"nanoseconds",
            "cancelled_at_descriptor":cancel.load(Ordering::Acquire),"deadline_expired_at_descriptor":at>=limits.deadline,
            "native_owner_before":native_before,"native_owner_scope":"sequential-mutex-and-atomic-owner-snapshot;not-atomic-ledger-or-worker-Stats-ACK",
            "search_not_yet_called":true,"descriptor_persistence":"sealed-before-search;buffered-trace-not-durable-disk-commit",
            "reserved_output_bytes":SEARCH_RETURN_RESERVED_BYTES,"reserved_rows":SEARCH_RETURN_HELD_ROWS,
            "per_descriptor_or_return_row_bytes_including_newline":SEARCH_RETURN_ROW_BYTES,"summary_row_bytes_including_newline":SEARCH_SUMMARY_ROW_BYTES,
            "physical_nn_rows":null,"physical_nn_rows_observation":"unknown;role-callback-counts-are-not-graph-run-counts",
            "query_action_causal_authority":false,"whole_action_cost_authority":false,"utility_authority":false,"training_target_authority":false});
        // Both exact descriptor validation and all accounting checks precede the
        // atomic reservation. Failure cannot start search or leave partial credit.
        let json =
            bounded_json(&descriptor, SEARCH_RETURN_ROW_BYTES - 1, false).map_err(role_error)?;
        let descriptor_sha256 = format!("{:x}", Sha256::digest(&json));
        let capture_sequence = capture.sequence;
        let next_bytes = self
            .trace
            .reserved_bytes
            .checked_add(SEARCH_RETURN_RESERVED_BYTES)
            .filter(|n| *n <= capture.max_bytes)
            .ok_or_else(|| {
                role_error("observed search return byte credit exhausted before search")
            })?;
        let next_held = self
            .prepaid_native_rows
            .checked_add(SEARCH_RETURN_HELD_ROWS)
            .ok_or_else(|| role_error("observed search return row credit overflow"))?;
        if self
            .trace
            .rows
            .len()
            .checked_add(self.prepaid_recheck_rows)
            .and_then(|n| n.checked_add(next_held))
            .is_none_or(|n| n > MAX_ROWS * 8)
        {
            return Err(role_error(
                "observed search return row credit exhausted before search",
            ));
        }
        self.trace
            .rows
            .try_reserve(SEARCH_RETURN_HELD_ROWS)
            .map_err(|_| role_error("observed search return row allocation failed"))?;
        self.trace.reserved_bytes = next_bytes;
        self.prepaid_native_rows = next_held;
        self.pending_search_return = Some(PendingSearchReturn {
            descriptor_sha256,
            capture_sequence,
            limits,
            descriptor_pending: true,
            return_pending: true,
            summary_pending: true,
        });
        self.search_return_row(SearchReturnSlot::Descriptor, json)
    }
    fn search_return_row(
        &mut self,
        slot: SearchReturnSlot,
        json: Vec<u8>,
    ) -> Result<(), RoleError> {
        let pending = self
            .pending_search_return
            .as_ref()
            .ok_or_else(|| role_error("observed search row has no exclusive reservation"))?;
        if pending.capture_sequence != self.context()?.sequence {
            return Err(role_error("observed search row capture sequence changed"));
        }
        let (allowed, maximum, artifact) = match slot {
            SearchReturnSlot::Descriptor => (
                pending.descriptor_pending,
                SEARCH_RETURN_ROW_BYTES,
                SEARCH_RETURN_ARTIFACT,
            ),
            SearchReturnSlot::Returned => (
                !pending.descriptor_pending && pending.return_pending,
                SEARCH_RETURN_ROW_BYTES,
                SEARCH_RETURN_ARTIFACT,
            ),
            SearchReturnSlot::Summary => (
                !pending.descriptor_pending && pending.summary_pending,
                SEARCH_SUMMARY_ROW_BYTES,
                "native-work-summary.jsonl",
            ),
        };
        if !allowed || json.is_empty() || json.len() >= maximum {
            return Err(role_error(
                "observed search row exceeded or reused its own reserved slot",
            ));
        }
        let remaining = self
            .prepaid_native_rows
            .checked_sub(1)
            .ok_or_else(|| role_error("observed search held row accounting underflow"))?;
        let pending = self
            .pending_search_return
            .as_mut()
            .ok_or_else(|| role_error("observed search reservation disappeared"))?;
        match slot {
            SearchReturnSlot::Descriptor => pending.descriptor_pending = false,
            SearchReturnSlot::Returned => pending.return_pending = false,
            SearchReturnSlot::Summary => pending.summary_pending = false,
        }
        self.prepaid_native_rows = remaining;
        self.trace.rows.push(PalsNativeTraceRow { artifact, json });
        Ok(())
    }
    #[allow(clippy::too_many_arguments)]
    fn observed_search_return(
        &mut self,
        result: &Result<PalsResult, PalsError>,
        counters: Option<&PalsCounters>,
        observer_error: Option<&str>,
        returned_at: Instant,
        cancelled_at_search_return: bool,
        owner_snapshot_started: Instant,
        owner_snapshot_observed: Instant,
        native_after: serde_json::Value,
        cancel: &AtomicBool,
    ) -> Result<(), RoleError> {
        let pending = self
            .pending_search_return
            .as_ref()
            .ok_or_else(|| role_error("search returned without its before descriptor"))?;
        let mut prefix = Sha256::new();
        prefix.update(b"rz-pals-native-observed-search-trace-prefix/1");
        for row in &self.trace.rows {
            prefix.update((row.artifact.len() as u64).to_be_bytes());
            prefix.update(row.artifact.as_bytes());
            prefix.update((row.json.len() as u64).to_be_bytes());
            prefix.update(&row.json);
        }
        let construction_started = Instant::now();
        let mut value = serde_json::json!({"domain":SEARCH_RETURN_DOMAIN,"stage":"search_returned",
            "game_id":self.context()?.game,"capture_sequence":pending.capture_sequence,
            "before_search_row_sha256":pending.descriptor_sha256,"search_called_and_returned":true,
            "actual_result":search_return_result(result)?,"last_search_counters":counters.map(search_return_counters),
            "last_search_counters_observation":if counters.is_some(){"actual-engine-last-search-counters"}else{"None;unobserved-not-zero"},
            "recheck_observer_error":observer_error,"native_trace_failure":self.trace.failure,
            "trace_prefix_rows":self.trace.rows.len(),"trace_capture_sequence_at_return":self.trace.sequence,
            "trace_prefix_sha256":format!("{:x}",prefix.finalize()),
            "trace_prefix_scope":"ordered-artifact-names-and-exact-raw-JSON-bytes-before-this-row;not-durable-disk-commit",
            "clock":{"origin":"native_capture_start","unit":"nanoseconds",
                "search_returned_tick":search_return_tick(self.started,returned_at)?,
                "owner_snapshot_started_tick":search_return_tick(self.started,owner_snapshot_started)?,
                "owner_snapshot_observed_tick":search_return_tick(self.started,owner_snapshot_observed)?,
                "json_construction_started_tick":search_return_tick(self.started,construction_started)?,
                "serialization_started_tick":null,
                "serialization_completed_tick":null,"serialization_completed_observation":"not-self-attested-in-this-row"},
            "original_deadline":search_return_deadline(self.started,pending.limits.deadline)?,
            "deadline_expired_at_search_return":returned_at>=pending.limits.deadline,
            "deadline_expired_at_owner_snapshot":owner_snapshot_observed>=pending.limits.deadline,
            "deadline_expired_at_serialization_start":null,
            "cancelled_at_search_return_observation":cancelled_at_search_return,
            "cancelled_at_return_observer":cancel.load(Ordering::Acquire),
            "native_owner_after":native_after,"native_owner_scope":"sequential-mutex-and-atomic-owner-snapshot;not-atomic-ledger-or-worker-Stats-ACK",
            "search_return_is_worker_join":false,"search_return_is_process_shutdown":false,
            "physical_nn_rows":null,"physical_nn_rows_observation":"unknown;owner-role-callback-counts-and-search-role-counters-kept-separate",
            "query_action_causal_authority":false,"whole_action_cost_authority":false,"utility_authority":false,"training_target_authority":false});
        // Capture the byte-serialization boundary once, before serialization.
        // No fixed-point attempt invents a completed time inside its own row.
        let serialization_started = Instant::now();
        value["clock"]["serialization_started_tick"] =
            serde_json::json!(search_return_tick(self.started, serialization_started)?);
        value["deadline_expired_at_serialization_start"] =
            serde_json::json!(serialization_started >= pending.limits.deadline);
        let json = bounded_json(&value, SEARCH_RETURN_ROW_BYTES - 1, false).map_err(role_error)?;
        self.search_return_row(SearchReturnSlot::Returned, json)
    }
    fn search_summary_row(&mut self, summary: &serde_json::Value) -> Result<(), RoleError> {
        let json =
            bounded_json(summary, SEARCH_SUMMARY_ROW_BYTES - 1, false).map_err(role_error)?;
        self.search_return_row(SearchReturnSlot::Summary, json)
    }
    fn preserve_incomplete_return_failure(&mut self) {
        let missing = self
            .calls
            .iter()
            .find(|(_, c)| !c.physical && !c.rejected)
            .map(|(id, _)| {
                format!(
                    "native request {}:{} returned without physical/rejection evidence",
                    id.epoch.0, id.sequence
                )
            });
        if let Some(missing) = missing {
            if self.trace.failure.is_some() {
                native_trace_secondary(&mut self.trace, "missing physical/rejection", missing);
            } else {
                // Preserve the legacy no-primary diagnostic verbatim.
                self.trace.failure = Some(missing);
            }
        }
        if self.pending_recheck.is_some()
            || self.pending_continuation.is_some()
            || self.rejected_recheck.is_some()
        {
            native_trace_failure(
                &mut self.trace,
                "missing recheck finish",
                "native recheck returned without its prepaid finish or rejected-preparation close",
            );
        }
        if self
            .pending_search_return
            .as_ref()
            .is_some_and(|p| p.descriptor_pending || p.return_pending || p.summary_pending)
        {
            native_trace_secondary(
                &mut self.trace,
                "observed search return",
                "reserved return or summary row is missing",
            );
        }
    }
    fn context(&self) -> Result<&PalsNativeCaptureContext, RoleError> {
        self.context
            .as_ref()
            .ok_or_else(|| role_error("native capture context not admitted"))
    }
    fn row<T: Serialize>(&mut self, artifact: &'static str, value: &T) -> Result<(), RoleError> {
        let json = bounded_json(value, MAX_JSON_RECORD_BYTES, false).map_err(role_error)?;
        if self
            .trace
            .rows
            .len()
            .checked_add(self.prepaid_recheck_rows)
            .and_then(|n| n.checked_add(self.prepaid_native_rows))
            .is_none_or(|n| n >= MAX_ROWS * 8)
        {
            return Err(role_error("native trace row limit"));
        }
        self.trace.rows.push(PalsNativeTraceRow { artifact, json });
        Ok(())
    }
    fn event(
        &mut self,
        id: RequestId,
        stage: &str,
        detail: serde_json::Value,
    ) -> Result<(), RoleError> {
        let input = self
            .calls
            .get(&id)
            .ok_or_else(|| role_error("native event has no sealed request"))?;
        let value = serde_json::json!({
            "domain":"rz-pals-native-call-event/1", "game_id":self.context()?.game,
            "process_epoch":id.epoch.0,"request_sequence":id.sequence,
            "input_sha256":input.input_sha256, "stage":stage,
            "observer_elapsed_us":self.started.elapsed().as_micros().min(u64::MAX as u128) as u64,
            "detail":detail
        });
        if stage != "prepared" {
            self.consume_call_row(id)?;
        }
        self.row("native-events.jsonl", &value)
    }
    fn consume_call_row(&mut self, id: RequestId) -> Result<(), RoleError> {
        let call = self
            .calls
            .get_mut(&id)
            .ok_or_else(|| role_error("reserved native row has no call"))?;
        if call.logical.is_some() {
            call.prepaid_rows = call
                .prepaid_rows
                .checked_sub(1)
                .ok_or_else(|| role_error("native callback exhausted its prepaid rows"))?;
            self.prepaid_native_rows = self
                .prepaid_native_rows
                .checked_sub(1)
                .ok_or_else(|| role_error("native prepaid row accounting underflow"))?;
        }
        Ok(())
    }
    fn reserve(&mut self, bytes: u64) -> Result<(), RoleError> {
        let next = self
            .trace
            .reserved_bytes
            .checked_add(bytes)
            .filter(|n| *n <= self.context.as_ref().map_or(0, |c| c.max_bytes))
            .ok_or_else(|| {
                role_error("native collection output credit exhausted before dispatch")
            })?;
        self.trace.reserved_bytes = next;
        Ok(())
    }
    fn selected_recheck(&self) -> bool {
        self.source
            .native
            .as_ref()
            .and_then(|n| n.get("pals_search_policy"))
            .is_some()
    }
    fn recheck_row(
        &mut self,
        stage: &str,
        identity: RecheckIdentity,
        descriptor_sha256: &str,
        data: serde_json::Value,
    ) -> Result<String, RoleError> {
        if self.prepaid_recheck_rows == 0 {
            return Err(role_error("recheck close has no prepaid row"));
        }
        let identity_json = recheck_identity_json(identity);
        let payload_sha256 = canonical_sha256(&(
            RECHECK_DOMAIN,
            stage,
            &identity_json,
            descriptor_sha256,
            &data,
        ))
        .map_err(role_error)?;
        let value = serde_json::json!({"domain":RECHECK_DOMAIN,"stage":stage,"game_id":self.context()?.game,
            "identity":identity_json,"descriptor_sha256":descriptor_sha256,"payload_sha256":payload_sha256,
            "observer_elapsed_us":self.started.elapsed().as_micros().min(u64::MAX as u128) as u64,"data":data});
        let json = bounded_json(&value, MAX_RECHECK_ROW_BYTES, false).map_err(role_error)?;
        // Consume only this prepaid slot. Unused bound/finish credit remains
        // charged through drain; it cannot be borrowed by a later native call.
        self.prepaid_recheck_rows -= 1;
        self.trace.rows.push(PalsNativeTraceRow {
            artifact: RECHECK_ARTIFACT,
            json,
        });
        Ok(payload_sha256)
    }
    fn parent_repair_chain(
        &self,
        event: &RecheckPrepared<'_>,
        root_sha: &str,
    ) -> Result<(serde_json::Value, u64, u64), RoleError> {
        let revision = event
            .identity
            .repair_record_revision
            .checked_sub(1)
            .ok_or_else(|| role_error("accepted Repair revision cannot name a previous call"))?;
        let mut chain = BTreeMap::new();
        let mut epoch = None;
        let mut previous_sequence = None;
        let mut proposal = None;
        for (id, call) in &self.calls {
            let Some(meta) = &call.logical else {
                continue;
            };
            if meta.kind != NativeQueryKind::Repair
                || meta.context.public_revision != revision
                || meta.context.game_generation != event.identity.game_generation
                || meta.context.search_generation != event.identity.search_generation
                || meta.root_sha256 != root_sha
            {
                continue;
            }
            let ply = meta.prefix.len();
            if ply >= event.repaired.len()
                || meta.prefix.as_slice() != &event.repaired[..ply]
                || meta.counterexample.as_deref() != Some(event.refutation)
                || meta.chosen_first != Some(event.repaired[ply])
            {
                continue;
            }
            if !call.physical
                || !call.delivered
                || !call.accepted
                || call.rejected
                || call.physical_unknown
                || !meta.accepted_context_checked
                || !meta.producer_metadata_admitted
                || meta.context.purpose != RoleQueryPurpose::RepairPolicy
            {
                return Err(role_error(
                    "candidate parent Repair lacks actual completed/accepted/context/producer evidence",
                ));
            }
            if epoch.is_some_and(|e| e != id.epoch.0)
                || previous_sequence.is_some_and(|p| p >= id.sequence)
                || proposal
                    .as_ref()
                    .is_some_and(|p: &Vec<BoardMove>| p != &meta.proposal)
                || chain.contains_key(&ply)
            {
                return Err(role_error(
                    "parent Repair chain is ambiguous or has a changed epoch/order/proposal",
                ));
            }
            epoch = Some(id.epoch.0);
            previous_sequence = Some(id.sequence);
            proposal = Some(meta.proposal.clone());
            let context = logical_json(&meta.context)?;
            chain.insert(ply, serde_json::json!({"process_epoch":id.epoch.0,"request_sequence":id.sequence,
                "input_sha256":call.input_sha256,"input_row_sha256":meta.input_row_sha256,"sidecar_row_sha256":meta.sidecar_row_sha256,
                "sidecar_sha256":meta.sidecar_sha256,"canonical_tensor_sha256":meta.canonical_tensor_sha256,
                "lineage_row_sha256":meta.lineage_row_sha256,"prefix_len":ply,"prefix":pack(&meta.prefix).map_err(role_error)?,
                "chosen_move":Move16::pack(event.repaired[ply]).map_err(role_error)?.bits(),"position_rules_state_sha256":meta.position_sha256,
                "proposal":pack(&meta.proposal).map_err(role_error)?,"counterexample":pack(event.refutation).map_err(role_error)?,
                "logical_context_sha256":canonical_sha256(&context).map_err(role_error)?,"logical_context":context,
                "producer_metadata_admitted":meta.producer_metadata_admitted,
                "selection_observation":"actual-raw-policy-ranked-first-and-accepted-context-matches-repaired-ply"}));
        }
        let start = *chain
            .keys()
            .next()
            .ok_or_else(|| role_error("no actual accepted parent Repair chain"))?;
        if start == 0
            || start >= event.anchor_ply
            || chain.len() != event.repaired.len() - start
            || !chain.keys().copied().eq(start..event.repaired.len())
        {
            return Err(role_error(
                "parent Repair chain does not cover the repaired suffix through its anchor",
            ));
        }
        Ok((
            serde_json::json!({"initial_prefix":pack(&event.repaired[..start]).map_err(role_error)?,
            "initial_prefix_len":start,"full_repaired_line":pack(event.repaired).map_err(role_error)?,
            "calls":chain.into_values().collect::<Vec<_>>(),
            "journal_observation":"independent-exact-producer-journal-entry-required; raw-entry-SHA-not-returned-by-producer-API"}),
            epoch.ok_or_else(|| role_error("parent Repair epoch absent"))?,
            previous_sequence.ok_or_else(|| role_error("parent Repair sequence absent"))?,
        ))
    }
    fn prepare_recheck(&mut self, event: RecheckPrepared<'_>) -> Result<(), RoleError> {
        if self.pending_recheck.is_some()
            || self.pending_continuation.is_some()
            || self.rejected_recheck.is_some()
        {
            return Err(role_error(
                "recheck attempted while a prior descriptor is unclosed",
            ));
        }
        self.rejected_recheck = Some(event.identity);
        if event.policy != PostRepairRecheckPolicy::SameRepairedLineOnceV1
            || !self.selected_recheck()
            || self.closed_rechecks.contains(&event.identity)
            || self.closed_rechecks.len() >= MAX_RECHECK_ATTEMPTS
        {
            return Err(role_error(
                "unselected, duplicate or unbounded native recheck attempt",
            ));
        }
        self.producer
            .as_ref()
            .ok_or_else(|| role_error("recheck descriptor requires strict producer"))?
            .verify_source(&self.source)
            .map_err(role_error)?;
        event.config.validate().map_err(role_error)?;
        if event.cancel.load(Ordering::Acquire)
            || Instant::now() >= event.limits.deadline
            || event.repaired.len() > event.config.line_plies
            || event.refutation.len() > event.config.line_plies
            || event.anchor_ply == 0
            || event.anchor_ply >= event.repaired.len()
            || event.reply_context.purpose != RoleQueryPurpose::ReplyPolicy
            || event.reply_context.prefix.as_slice() != &event.repaired[..event.anchor_ply]
            || event.reply_context.game_generation != event.identity.game_generation
            || event.reply_context.search_generation != event.identity.search_generation
            || event.reply_context.state != event.anchor_state
            || event.reply_context.situation != event.anchor_situation
            || event.repair_record.kind != RecordKind::Repair
            || event.repair_record.revision != event.identity.repair_record_revision
            || event.repair_record.origin_state != event.root_state
            || event.repair_record.line.as_slice() != event.repaired
            || !matches!(
                event.checker_identity,
                rz_search::cpu_checker::CheckerIdentity::Owned(_)
            )
        {
            return Err(role_error(
                "recheck prepared invariants/controls/accepted Repair/OwnCPU namespace differ",
            ));
        }
        let root = &self.context()?.position;
        if !event.root_snapshot.same_state(&root.snapshot()) {
            return Err(role_error(
                "recheck root differs from exact native capture root/history",
            ));
        }
        let root_sha = state_sha(root).map_err(role_error)?;
        let (anchor, _) =
            replay_line(root, &event.repaired[..event.anchor_ply]).map_err(role_error)?;
        let (repaired, _) = replay_line(root, event.repaired).map_err(role_error)?;
        if !event.anchor_snapshot.same_state(&anchor.snapshot()) {
            return Err(role_error(
                "recheck anchor differs from exact repaired prefix Rules replay",
            ));
        }
        let anchor_sha = state_sha(&anchor).map_err(role_error)?;
        let (parent_chain, parent_epoch, parent_last_sequence) =
            self.parent_repair_chain(&event, &root_sha)?;
        let logical = logical_json(event.reply_context)?;
        let data = serde_json::json!({"engine_observer_version":POST_REPAIR_RECHECK_OBSERVER_VERSION,
            "checked_source_sha256":canonical_sha256(&("rz-pals-collector-checked-source/1",&self.source)).map_err(role_error)?,
            "refinement_registration_sha256":self.source.native.as_ref().and_then(|n|n.get("refinement_registration_sha256")),
            "root_state":event.root_state.0,"root_rules_state_sha256":root_sha,"anchor_state":event.anchor_state.0,
            "anchor_situation":{"slot":event.anchor_situation.slot,"generation":event.anchor_situation.generation},
            "anchor_rules_state_sha256":anchor_sha,"anchor_ply":event.anchor_ply,
            "anticipated_reply_context":logical,"anticipated_reply_context_sha256":canonical_sha256(&logical).map_err(role_error)?,
            "accepted_repair_record":{"revision":event.repair_record.revision,"origin_state":event.repair_record.origin_state.0,
                "kind":format!("{:?}",event.repair_record.kind),"line":pack(&event.repair_record.line).map_err(role_error)?,
                "value":event.repair_record.value,"completed_depth":event.repair_record.completed_depth,
                "score_scope":event.repair_record.score_scope.map(|s|format!("{:?}",s)),"cpu_observation":event.repair_record.cpu_observation.map(|id|id.0),
                "white_perspective":event.repair_record.perspective==Color::White,"critical":event.repair_record.critical},
            "repaired":pack(event.repaired).map_err(role_error)?,"refutation":pack(event.refutation).map_err(role_error)?,
            "parent_repair_chain":parent_chain,"repaired_endpoint":recheck_endpoint_json(&event.repaired_endpoint,&repaired)?,
            "limits":{"max_rounds":event.limits.max_rounds,"max_cpu_nodes":event.limits.max_cpu_nodes,"cpu_depth":event.limits.cpu_depth},
            "engine_deadline_tick":event.deadline_tick,"engine_deadline_tick_unit":"nanoseconds","engine_deadline_tick_origin":"engine_monotonic_clock_origin",
            "observer_elapsed_origin":"native_capture_start","observer_elapsed_unit":"microseconds",
            "cancelled_at_observer":event.cancel.load(Ordering::Acquire),"deadline_expired_at_observer":Instant::now()>=event.limits.deadline,
            "checker_identity":event.checker_identity,"cpu_condition":event.cpu_condition,
            "prepared_before_reply_submit":true,"trace_persistence":"seal-before-submit; prepaid-buffer-drain-after-search",
            "assurance":"conditional-search-observer; no-whole-game-proof; no-training-target; task-private-consumers-not-observed"});
        let descriptor_sha = canonical_sha256(&(
            RECHECK_DOMAIN,
            "prepared-descriptor",
            recheck_identity_json(event.identity),
            &data,
        ))
        .map_err(role_error)?;
        // Validate the exact row before mutating reservations or pending state.
        let _ = bounded_json(&serde_json::json!({"domain":RECHECK_DOMAIN,"stage":"prepared","game_id":self.context()?.game,
            "identity":recheck_identity_json(event.identity),"descriptor_sha256":descriptor_sha,
            "payload_sha256":"00".repeat(32),"observer_elapsed_us":u64::MAX,"data":data}),MAX_RECHECK_ROW_BYTES,false).map_err(role_error)?;
        if self
            .trace
            .rows
            .len()
            .checked_add(self.prepaid_recheck_rows)
            .and_then(|n| n.checked_add(self.prepaid_native_rows))
            .and_then(|n| n.checked_add(RECHECK_ROWS))
            .is_none_or(|n| n > MAX_ROWS * 8)
        {
            return Err(role_error(
                "recheck prepared/Reply-bound/finished row credit exhausted before Reply",
            ));
        }
        self.reserve(
            (MAX_RECHECK_ROW_BYTES as u64 + 1)
                .checked_mul(RECHECK_ROWS as u64)
                .ok_or_else(|| role_error("recheck reservation overflow"))?,
        )?;
        self.prepaid_recheck_rows += RECHECK_ROWS;
        let prepared_payload_sha256 =
            self.recheck_row("prepared", event.identity, &descriptor_sha, data)?;
        self.pending_recheck = Some(PendingRecheck {
            identity: event.identity,
            descriptor_sha256: descriptor_sha,
            prepared_payload_sha256,
            reply_context: event.reply_context.clone(),
            anchor_sha256: anchor_sha,
            repaired: event.repaired.to_vec(),
            refutation: event.refutation.to_vec(),
            limits: event.limits,
            deadline_tick: event.deadline_tick,
            parent_epoch,
            parent_last_sequence,
            reply: None,
            bound_payload_sha256: None,
        });
        self.rejected_recheck = None;
        Ok(())
    }
    #[allow(clippy::too_many_arguments)]
    fn check_pending_reply(
        &self,
        id: RequestId,
        kind: NativeQueryKind,
        position: &Position,
        prefix: &[BoardMove],
        proposal: &[BoardMove],
        counterexample: Option<&[BoardMove]>,
        logical: Option<&RoleLogicalContext>,
        until: Instant,
    ) -> Result<(), RoleError> {
        if self.pending_continuation.is_some() {
            return self.check_continuation_reply(
                id,
                kind,
                position,
                prefix,
                proposal,
                counterexample,
                logical,
                until,
            );
        }
        let Some(pending) = &self.pending_recheck else {
            return Ok(());
        };
        if kind != NativeQueryKind::Reply
            || pending.reply.is_some()
            || logical != Some(&pending.reply_context)
            || prefix != pending.reply_context.prefix.as_slice()
            || proposal != pending.repaired.as_slice()
            || counterexample != Some(pending.refutation.as_slice())
            || id.epoch.0 != pending.parent_epoch
            || id.sequence <= pending.parent_last_sequence
            || until != pending.limits.deadline
            || state_sha(position).map_err(role_error)? != pending.anchor_sha256
        {
            return Err(role_error(
                "next actual native input is not the exact anticipated recheck Reply",
            ));
        }
        Ok(())
    }
    fn bind_recheck_reply(&mut self, id: RequestId) -> Result<(), RoleError> {
        if self.pending_continuation.is_some() {
            return self.bind_continuation_reply(id);
        }
        let Some(pending) = self.pending_recheck.as_ref() else {
            return Ok(());
        };
        let call = self
            .calls
            .get(&id)
            .ok_or_else(|| role_error("recheck Reply has no actual sealed input"))?;
        let meta = call
            .logical
            .as_ref()
            .ok_or_else(|| role_error("recheck Reply lacks actual logical input"))?;
        let identity = pending.identity;
        let descriptor = pending.descriptor_sha256.clone();
        let data = serde_json::json!({"prepared_payload_sha256":pending.prepared_payload_sha256,
            "process_epoch":id.epoch.0,"request_sequence":id.sequence,"input_sha256":call.input_sha256,
            "input_row_sha256":meta.input_row_sha256,"sidecar_row_sha256":meta.sidecar_row_sha256,"sidecar_sha256":meta.sidecar_sha256,
            "canonical_tensor_sha256":meta.canonical_tensor_sha256,"lineage_row_sha256":meta.lineage_row_sha256,
            "logical_context":logical_json(&meta.context)?,"producer_metadata_admitted":meta.producer_metadata_admitted,
            "bound_before_submit":true,"physical_completion_observed":false,"delivery_observed":false,"search_consumption_observed":false});
        let bound = self.recheck_row("reply_bound", identity, &descriptor, data)?;
        let pending = self
            .pending_recheck
            .as_mut()
            .ok_or_else(|| role_error("recheck pending disappeared during bind"))?;
        pending.reply = Some(id);
        pending.bound_payload_sha256 = Some(bound);
        Ok(())
    }
    fn finish_recheck(&mut self, event: RecheckFinished<'_>) -> Result<(), RoleError> {
        if event.policy != PostRepairRecheckPolicy::SameRepairedLineOnceV1 {
            return Err(role_error(
                "single-Reply native witness does not admit a C continuation",
            ));
        }
        if self.closed_rechecks.contains(&event.identity) {
            return Err(role_error("duplicate recheck finish"));
        }
        let Some(pending) = self.pending_recheck.as_ref() else {
            if self
                .rejected_recheck
                .is_some_and(|identity| identity != event.identity)
            {
                return Err(role_error(
                    "unreserved recheck finish names another rejected preparation",
                ));
            }
            if event.prepared_accepted
                || event.reply_call_attempted
                || event.reply_accepted
                || event.publication.is_some()
            {
                return Err(role_error(
                    "unreserved recheck finish claims prepared/dispatched/accepted/publication evidence",
                ));
            }
            // Preparation may have failed inside the engine before any callback,
            // or its output reservation was rejected. No positive row is invented.
            self.trace.failure.get_or_insert_with(|| {
                format!(
                    "recheck unsupported-preserved without reservation/Reply input: {}",
                    event
                        .original_error
                        .map(failure_text)
                        .unwrap_or_else(|| "preparation not observed".into())
                )
            });
            self.rejected_recheck = None;
            if self.closed_rechecks.len() < MAX_RECHECK_ATTEMPTS {
                self.closed_rechecks.push(event.identity);
            }
            return Ok(());
        };
        if pending.identity != event.identity
            || !event.prepared_accepted
            || event.reply_context != &pending.reply_context
            || event.limits.deadline != pending.limits.deadline
            || event.limits.max_cpu_nodes != pending.limits.max_cpu_nodes
            || event.limits.max_rounds != pending.limits.max_rounds
            || event.limits.cpu_depth != pending.limits.cpu_depth
            || event.deadline_tick != pending.deadline_tick
        {
            return Err(role_error(
                "recheck finish drifted from its prepared identity/context/original limits",
            ));
        }
        let root = &self.context()?.position;
        let (repaired, _) = replay_line(root, &pending.repaired).map_err(role_error)?;
        let counter = event
            .counter_endpoint
            .as_ref()
            .map(|endpoint| {
                let (position, _) = replay_line(root, event.counterline).map_err(role_error)?;
                recheck_endpoint_json(endpoint, &position)
            })
            .transpose()?;
        let reply = match pending.reply {
            None => None,
            Some(id) => {
                let call = self
                    .calls
                    .get(&id)
                    .ok_or_else(|| role_error("bound recheck Reply disappeared"))?;
                if event.reply_accepted
                    && (!call.physical
                        || !call.delivered
                        || !call.accepted
                        || call.rejected
                        || call.physical_unknown
                        || !call
                            .logical
                            .as_ref()
                            .is_some_and(|m| m.accepted_context_checked))
                {
                    return Err(role_error(
                        "engine accepted recheck Reply lacks actual physical/delivery/consumption/context evidence",
                    ));
                }
                Some(
                    serde_json::json!({"process_epoch":id.epoch.0,"request_sequence":id.sequence,"input_sha256":call.input_sha256,
                    "physical":call.physical,"delivered":call.delivered,"search_consumed":call.accepted,"rejected":call.rejected,"physical_unknown":call.physical_unknown}),
                )
            }
        };
        if event.reply_accepted && pending.reply.is_none() {
            return Err(role_error("accepted recheck has no actual Reply RequestId"));
        }
        let publication=event.publication.as_ref().map(|p|serde_json::json!({"observation_id":p.observation_id.0,
            "observation":recheck_observation_json(p.observation),"conclusion":{"status":format!("{:?}",p.conclusion.status),
                "evidence":p.conclusion.evidence.map(|id|id.0),"revision":p.conclusion.revision}}));
        let identity = pending.identity;
        let descriptor = pending.descriptor_sha256.clone();
        let data = serde_json::json!({"prepared_payload_sha256":pending.prepared_payload_sha256,"bound_payload_sha256":pending.bound_payload_sha256,
            "prepared_accepted":event.prepared_accepted,"reply_call_attempted":event.reply_call_attempted,"reply_accepted":event.reply_accepted,
            "reply":reply,"reply_context":logical_json(event.reply_context)?,
            "selected_response":event.selected_response.map(|m|Move16::pack(m).map(|m|m.bits())).transpose().map_err(role_error)?,
            "counterline":pack(event.counterline).map_err(role_error)?,"full_suffix_replayed":event.full_suffix_replayed,
            "repaired_endpoint":recheck_endpoint_json(&event.repaired_endpoint,&repaired)?,"counter_endpoint":counter,
            "comparable":event.comparable,"publication":publication,"disposition":format!("{:?}",event.disposition),
            "original_error":event.original_error.map(failure_text),"engine_deadline_tick":event.deadline_tick,
            "engine_deadline_tick_unit":"nanoseconds","engine_deadline_tick_origin":"engine_monotonic_clock_origin",
            "observer_elapsed_origin":"native_capture_start","observer_elapsed_unit":"microseconds",
            "cancelled_at_observer":event.cancel.load(Ordering::Acquire),"deadline_expired_at_observer":Instant::now()>=event.limits.deadline,
            "limits":{"max_rounds":event.limits.max_rounds,"max_cpu_nodes":event.limits.max_cpu_nodes,"cpu_depth":event.limits.cpu_depth},
            "assurance":"conditional-search-observer; marker-is-not-refutation; no-training-target; physical-and-delivery-events-required"});
        self.recheck_row("finished", identity, &descriptor, data)?;
        self.pending_recheck = None;
        self.closed_rechecks.push(identity);
        Ok(())
    }
    fn capture(
        &mut self,
        id: RequestId,
        prepared: &PalsModelInput,
        context: NativePreparedContext<'_>,
    ) -> Result<(), RoleError> {
        self.capture_logical(id, prepared, context, None)
    }
    fn capture_logical(
        &mut self,
        id: RequestId,
        prepared: &PalsModelInput,
        context: NativePreparedContext<'_>,
        logical: Option<&RoleLogicalContext>,
    ) -> Result<(), RoleError> {
        if self.calls.contains_key(&id) || self.total_rows >= MAX_ROWS {
            return Err(role_error("duplicate native request or input row limit"));
        }
        let model_configuration = full_line::model_config(&self.source).map_err(role_error)?;
        prepared
            .validate(&model_configuration)
            .map_err(role_error)?;
        if prepared.model_epoch != self.source.model_epoch {
            return Err(role_error(
                "actual prepared model epoch differs from independent registry",
            ));
        }
        let (position, prefix, proposal, counterexample, records, revision, kind, divergences) =
            match context {
                NativePreparedContext::Role { query, kind } => (
                    query.position,
                    query.prefix,
                    query.proposal,
                    query.counterexample,
                    query.records,
                    query.revision,
                    kind,
                    Vec::new(),
                ),
                NativePreparedContext::Divergence { query } => (
                    query.root,
                    &[][..],
                    query.proposal,
                    None,
                    query.records,
                    query.revision,
                    NativeQueryKind::Divergence,
                    query.candidates.to_vec(),
                ),
            };
        let capture = self.context()?;
        let (replayed, _) = replay_line(&capture.position, prefix).map_err(role_error)?;
        if !replayed.snapshot().same_state(&position.snapshot())
            || prepared.situation_revision != revision
        {
            return Err(role_error(
                "native query prefix does not replay to its actual Rules state/revision",
            ));
        }
        let until = match context {
            NativePreparedContext::Role { query, .. } => query.deadline,
            NativePreparedContext::Divergence { query } => query.deadline,
        };
        self.check_pending_reply(
            id,
            kind,
            position,
            prefix,
            proposal,
            counterexample,
            logical,
            until,
        )?;
        if self.selected_recheck() && logical.is_none() {
            return Err(role_error(
                "selected recheck native input has no actual logical context",
            ));
        }
        if let Some(logical) = logical
            && (logical.public_revision != revision || logical.prefix.as_slice() != prefix)
        {
            return Err(role_error(
                "actual native logical context differs from query revision/prefix",
            ));
        }
        let eligible = kind == NativeQueryKind::Propose
            && prefix.is_empty()
            && capture.position.snapshot().same_state(&position.snapshot());
        let game = capture.game.clone();
        let opening = capture.opening.clone();
        let actual_moves = capture.actual_moves.clone();
        let mut complete_line = actual_moves.clone();
        complete_line.extend_from_slice(prefix);
        let sequence = self
            .trace
            .sequence
            .checked_add(1)
            .ok_or_else(|| role_error("native capture sequence exhausted"))?;
        let role = if matches!(kind, NativeQueryKind::Reply | NativeQueryKind::Divergence) {
            PalsDataRole::Critic
        } else {
            PalsDataRole::Proposer
        };
        let mut raw_rows = Vec::new();
        let mut selected = Vec::new();
        for token in &prepared.records {
            let index = usize::try_from(
                token
                    .record_id
                    .checked_sub(1)
                    .ok_or_else(|| role_error("native record ID zero"))?,
            )
            .map_err(role_error)?;
            let record = records
                .get(index)
                .ok_or_else(|| role_error("native public record is not in actual query"))?;
            if record.revision != token.revision || record.revision > revision {
                return Err(role_error("native public raw-source revision mismatch"));
            }
            let raw = serde_json::json!({"domain":"rz-pals-native-public-source/1","game_id":game,
                "record_index":token.record_id,"revision":record.revision,
                "origin_state_id":record.origin_state.0,"origin_state_id_is_advisory":true,
                "origin_rules_state_sha256":null,"origin_rules_identity_observation":"unknown",
                "kind":format!("{:?}",record.kind),"line":pack(&record.line).map_err(role_error)?,
                "value":record.value,"completed_depth":record.completed_depth,
                "scope":format!("{:?}",record.score_scope),"white_score_perspective":record.perspective==Color::White,
                "critical":token.critical,"source_cpu_profile_sha256":self.source.cpu_profile_sha256});
            let json = bounded_json(&raw, MAX_JSON_RECORD_BYTES, false).map_err(role_error)?;
            let digest = format!("{:x}", Sha256::digest(&json));
            selected.push(PalsPublicRecord {
                observation_sha256: digest.clone(),
                situation_revision: record.revision,
            });
            if !self.raw_sources.contains(&digest) {
                raw_rows.push((
                    digest,
                    PalsNativeTraceRow {
                        artifact: "public-record-sources.jsonl",
                        json,
                    },
                ));
            }
        }
        let input = PalsInputSnapshot {
            game_id: game.clone(),
            opening_id: opening.id.clone(),
            line_genealogy_id: format!("{game}-actual-and-conditional"),
            position_command: position_command(&opening, &complete_line),
            board_fen: position.to_fen(),
            actual_history: pack(&actual_moves).map_err(role_error)?,
            rules_state_sha256: state_sha(position).map_err(role_error)?,
            rules_history_sha256: hex(prepared.history_digest),
            transposition_sha256: transposition_sha(position),
            encoding_sha256: self.source.encoding_sha256.clone(),
            source: self.source.source.clone(),
            frozen_epoch: self.source.frozen_epoch,
            input_revision: revision,
            capture_sequence: sequence,
            white_to_move: position.side_to_move() == Color::White,
            role,
            legal_moves: pack(position.ordered_legal_moves().moves()).map_err(role_error)?,
            public_records: selected.clone(),
        }
        .seal()
        .map_err(role_error)?;
        if kind != NativeQueryKind::Divergence {
            let candidates = prepared
                .candidates
                .iter()
                .map(|c| c.packed())
                .collect::<Result<Vec<_>, _>>()
                .map_err(role_error)?;
            if candidates != input.snapshot().legal_moves {
                return Err(role_error(
                    "native prepared candidates differ from Rules legal order",
                ));
            }
        }
        let tensor_json = serde_json::to_string(prepared).map_err(role_error)?;
        let mut sidecar = PalsNativeInputSidecar {
            version: if model_configuration.profile.is_legacy() {
                "rz-pals-native-input-sidecar/1"
            } else {
                "rz-pals-native-input-sidecar/2"
            }
            .into(),
            input_sha256: input.sha256().into(),
            encoding_sha256: self.source.encoding_sha256.clone(),
            encoder_source_sha256: self.source.encoder_source_sha256.clone(),
            model_epoch_kind: "frozen_model_epoch".into(),
            canonical_tensor_sha256: hex(prepared
                .canonical_input_key(&model_configuration)
                .map_err(role_error)?),
            tensor_sha256: format!("{:x}", Sha256::digest(tensor_json.as_bytes())),
            tensor_json,
            record_sources: selected,
            model_profile: (!model_configuration.profile.is_legacy())
                .then(|| model_configuration.profile.as_str().into()),
            encoding_profile: (!model_configuration.profile.is_legacy())
                .then(|| model_configuration.profile.encoding_schema().into()),
            sha256: String::new(),
        };
        sidecar.sha256 = sidecar.digest().map_err(role_error)?;
        let full_line_row = full_line::prepare_context(
            &input,
            &sidecar,
            prepared,
            records,
            &self.source,
            id.epoch.0,
            id.sequence,
        )
        .map_err(role_error)?
        .map(|context| {
            bounded_json(&context, MAX_JSON_RECORD_BYTES, false)
                .map(|json| PalsNativeTraceRow {
                    artifact: full_line::CONTEXT_ARTIFACT,
                    json,
                })
                .map_err(role_error)
        })
        .transpose()?;
        let divergence = kind == NativeQueryKind::Divergence;
        let divergence_context = match context {
            NativePreparedContext::Divergence { query } => Some(divergence::prepare_context(
                id, &input, &sidecar, prepared, query,
            )?),
            NativePreparedContext::Role { .. } => None,
        };
        let mut lineage = serde_json::json!({"input_sha256":input.sha256(),"game_id":game,
            "process_epoch":id.epoch.0,"request_sequence":id.sequence,"native_query_kind":format!("{:?}",kind),
            "actual_played_history":pack(&actual_moves).map_err(role_error)?,"virtual_prefix":pack(prefix).map_err(role_error)?,
            "proposal":pack(proposal).map_err(role_error)?,"counterexample":counterexample.map(pack).transpose().map_err(role_error)?,
            "divergence_plies":divergences,"actual_outcome_eligible":eligible,
            "counterfactual_wdl":"masked","training_admission":if divergence{"deferred_divergence_head"}else{"ordinary_role"}});
        // Descriptor -> lineage -> exact prepared producer journal. The
        // descriptor never hashes the lineage or journal that references it.
        let divergence_row = divergence_context
            .as_ref()
            .map(|context| {
                lineage["native_divergence_context_sha256"] = serde_json::json!(context.sha256);
                bounded_json(context, MAX_JSON_RECORD_BYTES, false)
                    .map(|json| PalsNativeTraceRow {
                        artifact: divergence::CONTEXT_ARTIFACT,
                        json,
                    })
                    .map_err(role_error)
            })
            .transpose()?;
        let rows = [
            PalsNativeTraceRow {
                artifact: if divergence {
                    "native-divergence-inputs.jsonl"
                } else {
                    "inputs.jsonl"
                },
                json: bounded_json(&input, MAX_JSON_RECORD_BYTES, false).map_err(role_error)?,
            },
            PalsNativeTraceRow {
                artifact: if divergence {
                    "native-divergence-sidecars.jsonl"
                } else {
                    "native-inputs.jsonl"
                },
                json: bounded_json(&sidecar, MAX_JSON_RECORD_BYTES, false).map_err(role_error)?,
            },
            PalsNativeTraceRow {
                artifact: "input-lineage.jsonl",
                json: bounded_json(&lineage, MAX_JSON_RECORD_BYTES, false).map_err(role_error)?,
            },
        ];
        let call_logical = if self.selected_recheck() {
            let actual =
                logical.ok_or_else(|| role_error("selected input logical context missing"))?;
            Some(CallLogical {
                kind,
                context: actual.clone(),
                root_sha256: state_sha(&self.context()?.position).map_err(role_error)?,
                position_sha256: input.snapshot().rules_state_sha256.clone(),
                prefix: prefix.to_vec(),
                proposal: proposal.to_vec(),
                counterexample: counterexample.map(<[BoardMove]>::to_vec),
                legal: position.ordered_legal_moves().moves().to_vec(),
                input_row_sha256: format!("{:x}", Sha256::digest(&rows[0].json)),
                sidecar_row_sha256: format!("{:x}", Sha256::digest(&rows[1].json)),
                sidecar_sha256: sidecar.sha256.clone(),
                lineage_row_sha256: format!("{:x}", Sha256::digest(&rows[2].json)),
                canonical_tensor_sha256: sidecar.canonical_tensor_sha256.clone(),
                chosen_first: None,
                accepted_context_checked: false,
                producer_metadata_admitted: false,
            })
        } else {
            None
        };
        let exact_bytes = rows
            .iter()
            .chain(divergence_row.iter())
            .chain(full_line_row.iter())
            .chain(raw_rows.iter().map(|(_, r)| r))
            .try_fold(0_u64, |n, r| {
                n.checked_add(r.json.len() as u64 + 1)
                    .ok_or_else(|| role_error("prepared native bytes overflow"))
            })?;
        let added_rows = rows
            .len()
            .checked_add(raw_rows.len())
            .and_then(|n| n.checked_add(usize::from(divergence_row.is_some())))
            .and_then(|n| n.checked_add(usize::from(full_line_row.is_some())))
            .and_then(|n| n.checked_add(1)) // the prepaid prepared event
            .ok_or_else(|| role_error("native prepared row count overflow"))?;
        if self
            .trace
            .rows
            .len()
            .checked_add(added_rows)
            .and_then(|n| n.checked_add(self.prepaid_recheck_rows))
            .and_then(|n| n.checked_add(self.prepaid_native_rows))
            .and_then(|n| n.checked_add(if call_logical.is_some() { 4 } else { 0 }))
            .is_none_or(|n| n > MAX_ROWS * 8)
        {
            return Err(role_error("native prepared trace row limit"));
        }
        // Reject exhausted native credit before the producer can mutate its
        // exact prepared journal. A producer rejection after this reservation
        // still keeps the prepaid input/sidecar/lineage/descriptor below.
        self.reserve(
            exact_bytes
                .checked_add(RAW_RESERVE + STAGE_RESERVE)
                .ok_or_else(|| role_error("native dispatch reservation overflow"))?,
        )?;
        let producer_capture = if let Some(producer) = &self.producer {
            producer
                .verify_source(&self.source)
                .and_then(|()| {
                    producer.capture(
                        &input,
                        &rows[0].json,
                        &rows[1].json,
                        &rows[2].json,
                        Some((id.epoch.0, id.sequence)),
                        !divergence,
                    )
                })
                .map_err(role_error)
        } else {
            Ok(())
        };
        // The immutable snapshot and actual tensor are sealed before returning
        // to native submit. No later observation is admitted into these bytes.
        self.trace.sequence = sequence;
        self.total_rows += 1;
        let call_prepaid_rows = if call_logical.is_some() { 4 } else { 0 };
        self.prepaid_native_rows += call_prepaid_rows;
        for (digest, row) in raw_rows {
            self.raw_sources.insert(digest);
            self.trace.rows.push(row);
        }
        self.trace.rows.extend(rows);
        self.trace.rows.extend(divergence_row);
        self.trace.rows.extend(full_line_row);
        if !divergence {
            self.trace.inputs.push((input.clone(), eligible));
        }
        self.calls.insert(
            id,
            Call {
                input_sha256: input.sha256().into(),
                physical: false,
                delivered: false,
                accepted: false,
                rejected: false,
                physical_unknown: false,
                logical: call_logical.map(|mut logical| {
                    logical.producer_metadata_admitted =
                        producer_capture.is_ok() && self.producer.is_some();
                    logical
                }),
                prepaid_rows: call_prepaid_rows,
            },
        );
        self.bind_recheck_reply(id)?;
        let mut detail = serde_json::json!({"prepared_before_submit":true,"native_query_kind":format!("{:?}",kind)});
        if self.producer.is_some() {
            detail["producer_metadata_admitted"] = serde_json::json!(producer_capture.is_ok());
        }
        self.event(id, "prepared", detail)?;
        // A metadata rejection still preserves the prepaid exact prepared rows,
        // but the observer returns an error before the runtime can submit.
        producer_capture
    }
}
struct Observer(Arc<Mutex<Sink>>);

/// Factory/CLI failures can precede the main output receipt. Keep the native
/// owner behind the same finite cleanup and emit a small lifecycle receipt on
/// that failure stream; an explicit collection receipt marks it reported.
struct NativeFinishGuard {
    handle: NativeRoleFinishHandle,
    attempted: bool,
    reported: bool,
    error: Option<String>,
}
fn lifecycle_receipt(
    receipt: &rz_uci::pals_native::NativeRoleReceipt,
) -> Result<serde_json::Value, ArenaError> {
    let mut value = serde_json::to_value(receipt).map_err(|e| invalid(e.to_string()))?;
    if let Some(residency) = value.get_mut("residency").and_then(|v| v.as_object_mut()) {
        let bank = residency
            .remove("reader_initializer_bank")
            .unwrap_or(serde_json::Value::Null);
        let summary = if bank.is_null() {
            serde_json::Value::Null
        } else {
            let bytes = bounded_json(&bank, 256 * 1024, false)?;
            serde_json::json!({"canonical_json_sha256":canonical_sha256(&bank)?,"serialized_json_bytes":bytes.len(),
                "shared_parameters":bank.get("shared_parameters").and_then(|v|v.as_array()).map(Vec::len),
                "observation_scope":"export-manifest-serialized-ledger; native residency unknown"})
        };
        residency.insert("reader_initializer_bank_summary".into(), summary);
    }
    // The large serialized ledger is already bound by the export manifest SHA;
    // receipt reserve holds lifecycle facts and a small ledger identity only.
    let _ = bounded_json(&value, 48 * 1024, false)?;
    Ok(value)
}
impl NativeFinishGuard {
    fn new(handle: NativeRoleFinishHandle) -> Self {
        Self {
            handle,
            attempted: false,
            reported: false,
            error: None,
        }
    }
    fn receipt(&self) -> rz_uci::pals_native::NativeRoleReceipt {
        self.handle.receipt()
    }
    fn finish(&mut self, until: Instant) -> Result<(), RoleError> {
        self.attempted = true;
        let result = self.handle.finish(until);
        self.error = result.as_ref().err().map(failure_text);
        result.map(|_| ())
    }
}
impl Drop for NativeFinishGuard {
    fn drop(&mut self) {
        if self.reported {
            return;
        }
        if !self.attempted {
            if let Some(until) = Instant::now().checked_add(Duration::from_secs(30)) {
                let _ = self.finish(until);
            } else {
                self.error = Some("native cleanup deadline overflow".into());
            }
        }
        let native = self.receipt();
        let receipt = serde_json::json!({"domain":"rz-pals-collection-pre-receipt-shutdown/1",
            "receipt":lifecycle_receipt(&native).unwrap_or_else(|e|serde_json::json!({
                "physical_shutdown_confirmed":native.physical_shutdown_confirmed,"native_buffers_released":native.native_buffers_released,
                "quarantined":native.quarantined,"physical_runs_in_flight":native.physical_runs_in_flight,
                "compact_receipt_failure":failure_text(e)})),"finish_error":self.error,"actual_training_executed":false});
        match bounded_json(&receipt, 64 * 1024, false) {
            Ok(bytes) => eprintln!("{}", String::from_utf8_lossy(&bytes)),
            Err(_) => {
                eprintln!("pals_collect: native pre-receipt shutdown evidence serialization failed")
            }
        }
    }
}
impl Observer {
    fn with(
        &self,
        action: impl FnOnce(&mut Sink) -> Result<(), RoleError>,
    ) -> Result<(), RoleError> {
        let mut sink = self
            .0
            .lock()
            .map_err(|_| role_error("native collector sink poisoned"))?;
        let result = action(&mut sink);
        if let Err(error) = &result {
            sink.trace.failure = Some(failure_text(error));
        }
        result
    }
}
fn raw_bits(raw: &PalsRawOutput) -> serde_json::Value {
    serde_json::json!({"representation":"f32_ieee754_bits",
        "candidate_logits_bits":raw.candidate_logits.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),
        "wdl_logits_bits":raw.wdl_logits.map(f32::to_bits),
        "divergence_logits_bits":raw.divergence_logits.as_ref().map(|v|v.iter().map(|x|x.to_bits()).collect::<Vec<_>>()),
        "task_logits_bits":raw.task_logits.map(|v|v.map(f32::to_bits)),
        "private_latent_bits":raw.private_latent.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),
        "prediction_is_future_label":false})
}
impl NativeRoleObserver for Observer {
    fn prepared(
        &mut self,
        id: RequestId,
        input: &PalsModelInput,
        context: NativePreparedContext<'_>,
    ) -> Result<(), RoleError> {
        self.with(|s| s.capture(id, input, context))
    }
    fn prepared_with_logical_context(
        &mut self,
        id: RequestId,
        input: &PalsModelInput,
        context: NativePreparedContext<'_>,
        logical: Option<&RoleLogicalContext>,
    ) -> Result<(), RoleError> {
        self.with(|s| s.capture_logical(id, input, context, logical))
    }
    fn recheck_prepared(&mut self, event: RecheckPrepared<'_>) -> Result<(), RoleError> {
        self.with(|s| match event.policy {
            PostRepairRecheckPolicy::ActualOpponentContinuationV1 => s.prepare_continuation(event),
            _ => s.prepare_recheck(event),
        })
    }
    fn recheck_finished(&mut self, event: RecheckFinished<'_>) -> Result<(), RoleError> {
        self.with(|s| match event.policy {
            PostRepairRecheckPolicy::ActualOpponentContinuationV1 => s.finish_continuation(event),
            _ => s.finish_recheck(event),
        })
    }
    fn accepted_context(
        &mut self,
        id: RequestId,
        acceptance: &RoleAcceptance<'_>,
    ) -> Result<(), RoleError> {
        self.with(|s| {
            let call = s
                .calls
                .get(&id)
                .ok_or_else(|| role_error("accepted context has no actual RequestId"))?;
            let Some(meta) = &call.logical else {
                return Ok(());
            };
            let (position, _) =
                replay_line(&s.context()?.position, &meta.prefix).map_err(role_error)?;
            if !call.physical
                || !call.delivered
                || !call.accepted
                || call.rejected
                || call.physical_unknown
                || meta.accepted_context_checked
                || acceptance.context != &meta.context
                || !acceptance.snapshot.same_state(&position.snapshot())
            {
                return Err(role_error(
                    "actual accepted role context/snapshot differs from its prepared request",
                ));
            }
            s.calls
                .get_mut(&id)
                .and_then(|c| c.logical.as_mut())
                .ok_or_else(|| role_error("accepted logical call disappeared"))?
                .accepted_context_checked = true;
            Ok(())
        })
    }
    fn physically_completed(
        &mut self,
        id: RequestId,
        result: Result<&PalsRawOutput, &BackendError>,
    ) -> Result<(), RoleError> {
        self.with(|s| {
            let c=s.calls.get_mut(&id).ok_or_else(||role_error("physical result lacks prepared call"))?;
            if c.physical || c.physical_unknown {return Err(role_error("duplicate or unknown physical completion"));}
            c.physical=true;
            if let (Some(meta),Ok(raw))=(&mut c.logical,result)
                && matches!(meta.kind,NativeQueryKind::Repair|NativeQueryKind::Reply) {
                    if raw.candidate_logits.len()!=meta.legal.len() || raw.candidate_logits.iter().any(|v|!v.is_finite()) {
                        return Err(role_error("actual Repair/Reply raw policy shape/finite values differ from Rules legal order"));
                    }
                    let mut indices:Vec<_>=(0..meta.legal.len()).collect();
                    indices.sort_by(|&a,&b|raw.candidate_logits[b].total_cmp(&raw.candidate_logits[a]).then(a.cmp(&b)));
                    meta.chosen_first=indices.first().map(|&i|meta.legal[i]);
                    if let Some(pending) = s.pending_continuation.as_mut() {
                        pending.record_initial_ranking(id, &meta.legal, &indices)?;
                    }
            }
            let input_sha256=c.input_sha256.clone();
            let raw=match result {Ok(raw)=>raw_bits(raw),Err(error)=>serde_json::json!({"failure":failure_text(error),"kind":format!("{:?}",error)})};
            let json=bounded_json(&serde_json::json!({"domain":"rz-pals-native-physical-raw/1",
                "process_epoch":id.epoch.0,"request_sequence":id.sequence,"input_sha256":input_sha256,
                "physical_completion_confirmed":true,"success":result.is_ok(),"raw":raw}),RAW_RESERVE as usize-1,false).map_err(role_error)?;
            s.consume_call_row(id)?;
            if s.trace.rows.len().checked_add(s.prepaid_recheck_rows).and_then(|n|n.checked_add(s.prepaid_native_rows)).is_none_or(|n|n>=MAX_ROWS*8) {
                return Err(role_error("physical raw row exceeds held recheck/native row credit"));
            }
            s.trace.rows.push(PalsNativeTraceRow {artifact:"native-raw-outputs.jsonl",json});
            s.event(id,"physically_completed",serde_json::json!({"success":result.is_ok(),"logical_acceptance_inferred":false}))
        })
    }
    fn delivered(&mut self, id: RequestId) -> Result<(), RoleError> {
        self.with(|s| {
            let c = s
                .calls
                .get_mut(&id)
                .ok_or_else(|| role_error("delivery lacks prepared call"))?;
            if !c.physical || c.delivered || c.rejected || c.physical_unknown {
                return Err(role_error(
                    "delivery without known un-rejected physical output",
                ));
            }
            c.delivered = true;
            s.event(
                id,
                "delivered",
                serde_json::json!({"search_consumed":false}),
            )
        })
    }
    fn rejected(&mut self, id: RequestId, reason: NativeRoleRejection) -> Result<(), RoleError> {
        self.with(|s| {
            let c = s
                .calls
                .get_mut(&id)
                .ok_or_else(|| role_error("rejection lacks prepared call"))?;
            if c.accepted || c.rejected {
                return Err(role_error(
                    "duplicate rejection or rejected consumed output",
                ));
            }
            c.rejected = true;
            if c.physical && reason == NativeRoleRejection::PhysicalCompletionUnknown {
                return Err(role_error("known physical completion cannot become unknown due to observer/logical rejection"));
            }
            c.physical_unknown = reason == NativeRoleRejection::PhysicalCompletionUnknown;
            s.event(
                id,
                "logically_rejected",
                serde_json::json!({"reason":format!("{:?}",reason),"search_consumed":false}),
            )
        })
    }
    fn accepted(&mut self, id: RequestId) -> Result<(), RoleError> {
        self.with(|s| {
            let c = s
                .calls
                .get_mut(&id)
                .ok_or_else(|| role_error("consumption lacks prepared call"))?;
            if !c.physical || !c.delivered || c.accepted || c.rejected || c.physical_unknown {
                return Err(role_error("invalid/duplicate native search consumption"));
            }
            c.accepted = true;
            s.event(
                id,
                "search_consumed",
                serde_json::json!({"search_consumed":true}),
            )
        })
    }
}

pub struct OwnPalsOnnxCollectionDriver {
    engine: PalsEngine<NativeRoleModel>,
    finish: NativeFinishGuard,
    sink: Arc<Mutex<Sink>>,
    description: PalsCollectionSourceDescription,
    max_rounds: u64,
}
impl OwnPalsOnnxCollectionDriver {
    /// A separate explicit registration selects the actual multi-Reply lane.
    /// The ordinary/default and legacy refinement entry points keep their wire.
    #[allow(clippy::too_many_arguments)]
    pub fn load_cpu_with_continuation_policy(
        export: &Path,
        checkpoint: &Path,
        pin: &RuntimeLibraryPin,
        registry: &PalsNativeCollectionRegistry,
        cpu: CpuConfig,
        pals: PalsConfig,
        max_rounds: u64,
        registration: &PalsNativeContinuationRegistration,
    ) -> Result<Self, ArenaError> {
        registration.validate_against(registry)?;
        Self::load_cpu_selected(
            export,
            checkpoint,
            pin,
            registry,
            cpu,
            pals,
            max_rounds,
            Some(&registration.inner),
            None,
        )
    }
    pub fn load_cpu(
        export: &Path,
        checkpoint: &Path,
        pin: &RuntimeLibraryPin,
        registry: &PalsNativeCollectionRegistry,
        cpu: CpuConfig,
        pals: PalsConfig,
        max_rounds: u64,
    ) -> Result<Self, ArenaError> {
        Self::load_cpu_selected(
            export, checkpoint, pin, registry, cpu, pals, max_rounds, None, None,
        )
    }
    /// Explicit followup constructor. The independent registry selects the
    /// actual model profile; old load_cpu keeps its historical constructor.
    #[allow(clippy::too_many_arguments)]
    pub fn load_cpu_v4(
        export: &Path,
        checkpoint: &Path,
        pin: &RuntimeLibraryPin,
        registry: &PalsNativeCollectionRegistry,
        cpu: CpuConfig,
        pals: PalsConfig,
        max_rounds: u64,
        resolver: ResolverPolicy,
    ) -> Result<Self, ArenaError> {
        Self::load_cpu_selected(
            export,
            checkpoint,
            pin,
            registry,
            cpu,
            pals,
            max_rounds,
            None,
            Some(resolver),
        )
    }
    #[allow(clippy::too_many_arguments)]
    pub fn load_cpu_with_refinement_policy(
        export: &Path,
        checkpoint: &Path,
        pin: &RuntimeLibraryPin,
        registry: &PalsNativeCollectionRegistry,
        cpu: CpuConfig,
        pals: PalsConfig,
        max_rounds: u64,
        registration: &PalsNativeRefinementRegistration,
    ) -> Result<Self, ArenaError> {
        Self::load_cpu_selected(
            export,
            checkpoint,
            pin,
            registry,
            cpu,
            pals,
            max_rounds,
            Some(registration),
            None,
        )
    }
    #[allow(clippy::too_many_arguments)]
    fn load_cpu_selected(
        export: &Path,
        checkpoint: &Path,
        pin: &RuntimeLibraryPin,
        registry: &PalsNativeCollectionRegistry,
        cpu: CpuConfig,
        pals: PalsConfig,
        max_rounds: u64,
        registration: Option<&PalsNativeRefinementRegistration>,
        followup_resolver: Option<ResolverPolicy>,
    ) -> Result<Self, ArenaError> {
        registry.validate()?;
        // Validate the independent policy/base/binary registration before CPU
        // owner construction or runtime/model loading; no implicit fallback.
        if let Some(registration) = registration {
            registration.validate_against(registry)?;
        }
        pals.validate().map_err(|e| invalid(e.to_string()))?;
        if !(1..=16).contains(&max_rounds) {
            return Err(invalid("native collection max rounds must be 1..=16"));
        }
        let actual_binary = file_sha(&std::env::current_exe().map_err(io)?, 256 * 1024 * 1024)?;
        if !checkpoint.is_absolute()
            || file_sha(checkpoint, 256 * 1024 * 1024)? != registry.checkpoint_sha256
            || actual_binary != registry.collector_binary_sha256
            || hex(pin.binary_digest()) != registry.runtime_sha256
            || hex(
                rz_uci::pals_native::pals_fresh_encoding_semantic_digest_for_config(
                    &registry.model_configuration,
                ),
            ) != registry.encoding_sha256
            || native_source_sha()? != registry.encoder_source_sha256
        {
            return Err(invalid(
                "actual native collection assets differ from independent registry",
            ));
        }
        let cpu = CpuEngine::new(cpu).map_err(|e| invalid(e.to_string()))?;
        let cpu_source =
            source_description(&cpu, "own-pals-native-cpu-tasks", serde_json::Value::Null)?;
        if cpu_source.cpu_profile_sha256 != registry.cpu_configuration_sha256 {
            return Err(invalid(
                "actual own CPU task configuration differs from independent registry",
            ));
        }
        let native_config = PalsOnnxConfig {
            intra_threads: registry.intra_threads,
            cache_public_memory: registry.cache_public_memory,
            ..PalsOnnxConfig::cpu()
        };
        let mut model = NativeRoleModel::load_pinned(
            export,
            &registry.export_manifest_sha256,
            pin,
            native_config,
            Duration::from_secs(30),
        )
        .map_err(|e| invalid(e.to_string()))?;
        let mut finish = NativeFinishGuard::new(model.finish_handle());
        let identity = model.source_identity();
        let actual = finish.receipt();
        let graphs: Vec<_> = actual
            .residency
            .graphs
            .iter()
            .map(|g| PalsCollectionGraphPin {
                role: g.role.clone(),
                sha256: hex(g.sha256),
                serialized_bytes: g.serialized_bytes,
            })
            .collect();
        let mut expected_graphs = registry.graphs.clone();
        let mut actual_graphs = graphs.clone();
        expected_graphs.sort_by(|a, b| a.role.cmp(&b.role));
        actual_graphs.sort_by(|a, b| a.role.cmp(&b.role));
        if identity.trained
            || identity.frozen_epoch != registry.frozen_epoch
            || identity.model_configuration != registry.model_configuration
            || hex(identity.checkpoint_sha256) != registry.checkpoint_sha256
            || hex(identity.export_manifest_sha256) != registry.export_manifest_sha256
            || hex(identity.execution.runtime_sha256) != registry.runtime_sha256
            || identity.execution.provider != "cpu"
            || identity.execution.runtime_bundle_sha256.is_some()
            || actual_graphs != expected_graphs
        {
            let _ = finish.finish(Instant::now() + Duration::from_secs(30));
            return Err(invalid(
                "loaded native source/epoch/graphs/provider differs from registered untrained CPU source",
            ));
        }
        let configuration = serde_json::to_value(&identity.model_configuration)
            .map_err(|e| invalid(e.to_string()))?;
        let mut description = PalsCollectionSourceDescription {
            mode: "own-pals-onnx-untrained-cpu".into(),
            source: PalsInputSource::OwnPals {
                model_configuration_sha256: registry.model_configuration_sha256.clone(),
                model_weights_sha256: registry.checkpoint_sha256.clone(),
            },
            cpu_profile_sha256: cpu_source.cpu_profile_sha256.clone(),
            implementation_sha256: actual_binary,
            encoding_sha256: registry.encoding_sha256.clone(),
            encoder_source_sha256: registry.encoder_source_sha256.clone(),
            configuration,
            model_epoch: identity.checkpoint_sha256,
            model_epoch_kind: "frozen_model_epoch".into(),
            frozen_epoch: identity.frozen_epoch,
            native: Some(
                serde_json::json!({"independent_registry":registry,"loaded_source":identity,
                "cpu_task_source":cpu_source,"graphs":graphs,"search_version":PALS_SEARCH_VERSION,
                "search_configuration":{"max_rounds":max_rounds,"beam_width":pals.beam_width,"line_plies":pals.line_plies,"max_nodes":pals.max_nodes,
                    "max_records":pals.max_records,"max_role_calls":pals.max_role_calls,"cpu_nodes_per_task":pals.cpu_nodes_per_task},
                "provider":"cpu","precision":"fp32","training_state":"Untrained","actual_training_executed":false,
                "startup_probe_in_collection":false,"trace_persistence":"seal-before-submit; prepaid-buffer-drain-after-search"}),
            ),
        };
        checked_source(&description)?;
        // Split legacy search helpers are a compiled-source observation too.
        // A reader must still match an independent closed review profile and
        // the original build/manifest/binary pins; this fact enrolls nothing.
        if let Some(facts) = description
            .native
            .as_mut()
            .and_then(serde_json::Value::as_object_mut)
        {
            facts.insert(
                "search_implementation_sha256".into(),
                serde_json::json!(hex(
                    rz_search::pals::engine::compiled_search_implementation_sha256()
                )),
            );
        }
        let sink = Arc::new(Mutex::new(Sink {
            context: None,
            source: description.clone(),
            trace: PalsNativeTrace::default(),
            started: Instant::now(),
            calls: BTreeMap::new(),
            raw_sources: BTreeSet::new(),
            total_rows: 0,
            producer: None,
            pending_recheck: None,
            pending_continuation: None,
            rejected_recheck: None,
            closed_rechecks: Vec::new(),
            prepaid_recheck_rows: 0,
            prepaid_native_rows: 0,
            pending_search_return: None,
        }));
        model
            .set_observer(Box::new(Observer(Arc::clone(&sink))))
            .map_err(|e| invalid(e.to_string()))?;
        let engine = match (registration, followup_resolver) {
            (None, Some(resolver)) => {
                let checker = rz_search::cpu_checker::OwnedCpuChecker::new(cpu)
                    .map_err(|e| invalid(e.to_string()))?;
                PalsEngine::new_with_boxed_checker_and_policies(
                    pals,
                    model,
                    Box::new(checker),
                    resolver,
                    PostRepairRecheckPolicy::Disabled,
                )
            }
            (None, None) => PalsEngine::new(pals, model, cpu),
            (Some(registration), None) => PalsEngine::new_with_cpu_and_refinement_policy(
                pals,
                model,
                cpu,
                registered_refinement_policy(&registration.wire)?,
            ),
            (Some(_), Some(_)) => {
                return Err(invalid(
                    "V4 collection cannot reinterpret a legacy refinement registration",
                ));
            }
        }
        .map_err(|e| invalid(e.to_string()))?;
        if (followup_resolver.is_some() || !identity.model_configuration.profile.is_legacy())
            && let Some(facts) = description
                .native
                .as_mut()
                .and_then(serde_json::Value::as_object_mut)
        {
            facts.insert(
                "search_version".into(),
                serde_json::json!(engine.search_identity()),
            );
            facts.insert(
                "policy_conditions_sha256".into(),
                serde_json::json!(
                    engine
                        .refinement_conditions()
                        .map(|s| format!("{:x}", Sha256::digest(s.as_bytes())))
                ),
            );
            facts.insert(
                "followup_lane_explicit".into(),
                serde_json::json!(followup_resolver.is_some()),
            );
            facts.insert(
                "resolver_policy".into(),
                serde_json::json!(engine.resolver_policy().version()),
            );
            facts.insert(
                "checker_kind".into(),
                serde_json::json!(match engine.checker_identity() {
                    rz_search::cpu_checker::CheckerIdentity::Owned(_) => "own",
                    rz_search::cpu_checker::CheckerIdentity::ExternalUci(_) => "external_uci",
                }),
            );
            facts.insert(
                "checker_identity".into(),
                serde_json::to_value(engine.checker_identity())
                    .map_err(|e| invalid(e.to_string()))?,
            );
            facts.insert(
                "actual_model_identity".into(),
                serde_json::json!(engine.model_identity()),
            );
            facts.insert(
                "model_semantics_sha256".into(),
                serde_json::json!(hex(identity
                    .model_configuration
                    .profile
                    .registered_semantics_sha256())),
            );
            facts.insert(
                "model_implementation_sha256".into(),
                serde_json::json!(hex(rz_eval::pals_model::compiled_model_boundary_sha256())),
            );
            facts.insert(
                "search_implementation_sha256".into(),
                serde_json::json!(hex(
                    rz_search::pals::engine::compiled_search_implementation_sha256()
                )),
            );
            facts.insert(
                "resolver_implementation_sha256".into(),
                serde_json::json!(hex(
                    rz_search::pals::engine::compiled_resolver_implementation_sha256()
                )),
            );
            facts.insert(
                "resolver_semantics_sha256".into(),
                serde_json::json!(hex(engine.resolver_policy().semantics_sha256())),
            );
        }
        sink.lock()
            .map_err(|_| invalid("native collector sink poisoned during namespace binding"))?
            .source = description.clone();
        let observed = if followup_resolver.is_some() {
            if engine.post_repair_recheck_policy() != PostRepairRecheckPolicy::Disabled
                || engine.search_identity() != rz_search::pals::engine::PALS_FOLLOWUP_SEARCH_VERSION
                || engine.refinement_conditions()
                    != Some(rz_search::pals::engine::PALS_FOLLOWUP_CONDITIONS)
            {
                return Err(invalid(
                    "V4 actual collection policy/conditions differ from explicit baseline lane",
                ));
            }
            None
        } else {
            observed_refinement_selection(
                registration,
                engine.post_repair_recheck_policy(),
                engine.search_identity(),
                engine.refinement_conditions(),
            )?
        };
        if let (Some(registration), Some(observed)) = (registration, observed) {
            let facts = description
                .native
                .as_mut()
                .and_then(serde_json::Value::as_object_mut)
                .ok_or_else(|| invalid("loaded native source description is absent"))?;
            facts.insert(
                "refinement_registration".into(),
                serde_json::to_value(registration).map_err(|e| invalid(e.to_string()))?,
            );
            facts.insert(
                "refinement_registration_sha256".into(),
                serde_json::json!(registration.raw_sha256()),
            );
            facts.insert(
                "pals_search_policy".into(),
                serde_json::to_value(observed).map_err(|e| invalid(e.to_string()))?,
            );
            facts.insert(
                "search_version".into(),
                serde_json::json!(engine.search_identity()),
            );
            checked_source(&description)?;
            sink.lock()
                .map_err(|_| invalid("native collector sink poisoned during source binding"))?
                .source = description.clone();
        }
        Ok(Self {
            engine,
            finish,
            sink,
            description,
            max_rounds,
        })
    }
}
impl PalsCollectionDriver for OwnPalsOnnxCollectionDriver {
    fn description(&self) -> &PalsCollectionSourceDescription {
        &self.description
    }
    fn checked_producer_owner(&mut self) -> Result<Option<CheckedProducerOwner<'_>>, ArenaError> {
        // This description is created only after the constructor compares actual
        // loaded epoch, graphs, runtime and executable against independent pins.
        let source = self.description.clone();
        Ok(Some(CheckedProducerOwner::new(self, source)))
    }
    fn set_registered_producer(
        &mut self,
        producer: Option<RegisteredProducerHandle>,
    ) -> Result<(), ArenaError> {
        let mut sink = self
            .sink
            .lock()
            .map_err(|_| invalid("native collector sink poisoned"))?;
        if sink.context.is_some() || !sink.trace.rows.is_empty() {
            return Err(invalid(
                "producer handle must be fixed before native dispatch",
            ));
        }
        if let Some(handle) = &producer {
            handle.verify_source(&self.description)?;
        }
        sink.producer = producer;
        Ok(())
    }
    fn new_game(&mut self) {
        self.engine.new_game();
    }
    fn records_actual_native_calls(&self) -> bool {
        true
    }
    fn analyze(
        &mut self,
        _: &Position,
        _: &PalsFrozenInput,
        _: &PalsModelInput,
        _: CpuLimits,
        _: &AtomicBool,
    ) -> Result<PalsCollectionDecision, ArenaError> {
        Err(invalid(
            "real PALS collection requires actual native capture context",
        ))
    }
    fn analyze_native(
        &mut self,
        context: PalsNativeCaptureContext,
        limits: CpuLimits,
        cancel: &AtomicBool,
    ) -> Result<PalsCollectionDecision, ArenaError> {
        let position = context.position.clone();
        let selected_return =
            self.engine.post_repair_recheck_policy() != PostRepairRecheckPolicy::Disabled;
        {
            let mut s = self
                .sink
                .lock()
                .map_err(|_| invalid("native collector sink poisoned"))?;
            if s.context.is_some() || !s.trace.rows.is_empty() {
                return Err(invalid("prior native trace was not drained"));
            }
            if selected_return && s.producer.is_none() {
                return Err(invalid(
                    "explicit recheck cannot dispatch without a source-bound strict producer handle",
                ));
            }
            if selected_return != s.selected_recheck() {
                return Err(invalid(
                    "observed search return selection differs from registered source lane",
                ));
            }
            if selected_return {
                let actual = observed_refinement_identity(
                    self.engine.post_repair_recheck_policy(),
                    self.engine.search_identity(),
                    self.engine.refinement_conditions(),
                )?;
                let actual = serde_json::to_value(actual).map_err(|e| invalid(e.to_string()))?;
                if s.source
                    .native
                    .as_ref()
                    .and_then(|n| n.get("pals_search_policy"))
                    != Some(&actual)
                    || self
                        .description
                        .native
                        .as_ref()
                        .and_then(|n| n.get("pals_search_policy"))
                        != Some(&actual)
                {
                    return Err(invalid(
                        "actual search getters differ from registered observed-return lane",
                    ));
                }
            }
            s.trace.sequence = context.sequence;
            s.context = Some(context);
            s.started = Instant::now();
            s.calls.clear();
            s.raw_sources.clear();
            s.closed_rechecks.clear();
            s.rejected_recheck = None;
            s.prepaid_recheck_rows = 0;
            s.prepaid_native_rows = 0;
            s.pending_search_return = None;
        }
        let before = self
            .engine
            .records()
            .iter()
            .map(|r| r.revision)
            .max()
            .unwrap_or(0);
        let pals_limits = PalsLimits {
            deadline: limits
                .deadline
                .ok_or_else(|| invalid("native collection deadline missing"))?,
            max_rounds: self.max_rounds,
            max_cpu_nodes: limits.max_nodes,
            cpu_depth: limits.max_depth,
        };
        if selected_return {
            let native_before = lifecycle_receipt(&self.finish.receipt())?;
            self.sink
                .lock()
                .map_err(|_| invalid("native collector sink poisoned"))?
                .prepare_search_return(pals_limits, cancel, native_before)
                .map_err(|e| invalid(e.to_string()))?;
        }
        let result = self.engine.search(&position, pals_limits, cancel);
        // Freeze this before locks, counters, owner receipt cloning or JSON work.
        let search_returned_at = Instant::now();
        let cancelled_at_search_return = cancel.load(Ordering::Acquire);
        let counters = self.engine.last_search_counters();
        let recheck_observer_error = self.engine.last_recheck_observer_error().map(failure_text);
        let native_after = selected_return.then(|| {
            let snapshot_started = Instant::now();
            let snapshot = lifecycle_receipt(&self.finish.receipt());
            let snapshot_observed = Instant::now();
            (snapshot_started, snapshot_observed, snapshot)
        });
        let summary_result = (|| -> Result<(), ArenaError> {
            let mut s = self
                .sink
                .lock()
                .map_err(|_| invalid("native collector sink poisoned"))?;
            let c = counters.unwrap_or_default();
            s.trace.cpu_nodes = c.cpu_nodes;
            s.trace.cpu_jobs = c.cpu_tasks;
            s.trace.cpu_work_observation_incomplete =
                c.cpu_work_observation_incomplete || (result.is_err() && counters.is_none());
            if let Some(primary) = result.as_ref().err() {
                let previous = s.trace.failure.take();
                s.trace.failure = Some(format!(
                    "search primary: {}; native trace: {}",
                    failure_text(primary),
                    previous.as_deref().unwrap_or("none")
                ));
            }
            if let Some(error) = &recheck_observer_error {
                native_trace_failure(&mut s.trace, "recheck observer", error);
            }
            let envelope_result =
                if let Some((snapshot_started, snapshot_observed, snapshot)) = native_after {
                    match snapshot {
                        Ok(snapshot) => s
                            .observed_search_return(
                                &result,
                                counters.as_ref(),
                                recheck_observer_error.as_deref(),
                                search_returned_at,
                                cancelled_at_search_return,
                                snapshot_started,
                                snapshot_observed,
                                snapshot,
                                cancel,
                            )
                            .map_err(|e| invalid(e.to_string())),
                        Err(error) => Err(error),
                    }
                } else {
                    Ok(())
                };
            if let Err(error) = &envelope_result {
                native_trace_secondary(&mut s.trace, "observed search return", error);
            }
            if !selected_return {
                s.reserve(8192).map_err(|e| invalid(e.to_string()))?;
            }
            let mut summary = serde_json::json!({"domain":"rz-pals-native-search-work/1",
                "cpu_nodes":c.cpu_nodes,"cpu_tasks_requested":c.cpu_tasks_requested,"cpu_reports_returned":c.cpu_tasks,
                "cpu_work_observation_incomplete":s.trace.cpu_work_observation_incomplete,
                "cpu_task_configuration_sha256":self.description.cpu_profile_sha256,
                "search_result":result.as_ref().err().map(failure_text),"role_calls":c.role_calls,
                "search_consumed_role_outputs":c.consumed_role_outputs});
            if self.engine.post_repair_recheck_policy() != PostRepairRecheckPolicy::Disabled {
                summary["recheck_observer_error"] = serde_json::json!(recheck_observer_error);
            }
            let summary_result = if selected_return {
                s.search_summary_row(&summary)
            } else {
                s.row("native-work-summary.jsonl", &summary)
            }
            .map_err(|e| invalid(e.to_string()));
            if selected_return && let Err(error) = &summary_result {
                native_trace_secondary(&mut s.trace, "summary", error);
            }
            envelope_result?;
            summary_result?;
            Ok(())
        })();
        if let Err(error) = &summary_result
            && let Ok(mut s) = self.sink.lock()
            && !selected_return
        {
            let previous = s.trace.failure.take();
            s.trace.failure = Some(format!(
                "{}; summary secondary: {}",
                previous.as_deref().unwrap_or("search succeeded"),
                failure_text(error)
            ));
        }
        let r = search_after_summary(result, summary_result)?;
        let mut genealogy = Vec::new();
        for record in self.engine.records().iter().filter(|r| r.revision > before) {
            genealogy.push(PalsCollectedLine {
                kind: format!("{:?}", record.kind),
                moves: pack(&record.line)?,
                raw_value: record.value,
                completed_depth: record.completed_depth,
                raw_scope: format!("{:?}", record.score_scope),
                white_perspective: record.perspective == Color::White,
            });
        }
        Ok(PalsCollectionDecision {
            best_move: r.best_move,
            pv: r.best_move.into_iter().collect(),
            nodes: r.counters.cpu_nodes,
            completed_depth: 0,
            completed_estimate: false,
            genealogy,
            raw: serde_json::json!({"source":"actual_frozen_pals_onnx_cpu","resolver_version":r.resolver_version,"training_state":"Untrained",
                "completion":format!("{:?}",r.completion),"value_scope":format!("{:?}",r.value_scope),"raw_score":r.score,
                "cpu_nodes":r.counters.cpu_nodes,"cpu_tasks":r.counters.cpu_tasks,"cpu_task_configuration_sha256":self.description.cpu_profile_sha256,
                "role_calls":r.counters.role_calls,"search_consumed_role_outputs":r.counters.consumed_role_outputs,
                "model":r.model_identity,"raw_wdl_is_target":false}),
        })
    }
    fn take_native_trace(&mut self) -> Result<PalsNativeTrace, ArenaError> {
        let mut s = self
            .sink
            .lock()
            .map_err(|_| invalid("native collector sink poisoned"))?;
        s.preserve_incomplete_return_failure();
        let consumed: BTreeSet<_> = s
            .calls
            .values()
            .filter(|c| c.accepted)
            .map(|c| c.input_sha256.clone())
            .collect();
        for (input, eligible) in &mut s.trace.inputs {
            *eligible &= consumed.contains(input.sha256());
        }
        s.context = None;
        s.pending_recheck = None;
        s.pending_continuation = None;
        s.rejected_recheck = None;
        s.closed_rechecks.clear();
        s.prepaid_recheck_rows = 0;
        s.prepaid_native_rows = 0;
        s.pending_search_return = None;
        Ok(std::mem::take(&mut s.trace))
    }
    fn finish_collection(
        &mut self,
        until: Instant,
    ) -> Result<Option<serde_json::Value>, ArenaError> {
        let finish_error = self.finish.finish(until).err().map(failure_text);
        let receipt = self.finish.receipt();
        let failed = finish_error.is_some()
            || receipt.quarantined
            || !receipt.physical_shutdown_confirmed
            || !receipt.native_buffers_released
            || receipt.physical_runs_in_flight != 0
            || receipt.observer_failures != 0;
        let value = serde_json::json!({"receipt":lifecycle_receipt(&receipt)?,"finish_error":finish_error,
            "_collection_failure":if failed{Some("native collector did not confirm physical shutdown/buffer lifetime/observer evidence")}else{None},
            "collection_accepted":!failed});
        Ok(Some(value))
    }
    fn receipt_preserved(&mut self) {
        self.finish.reported = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rz_contracts::ProcessEpoch;
    use rz_search::pals::engine::{DivergenceQuery, RoleQuery};
    use rz_uci::pals_native::{prepare_divergence_input, prepare_role_input};

    fn refinement_base() -> PalsNativeCollectionRegistry {
        let model_configuration = PalsModelConfig::baseline();
        PalsNativeCollectionRegistry {
            version: REGISTRY_VERSION.into(),
            collector_binary_sha256: "01".repeat(32),
            model_configuration_sha256: canonical_sha256(&model_configuration).unwrap(),
            model_configuration,
            checkpoint_sha256: "02".repeat(32),
            export_manifest_sha256: "03".repeat(32),
            graphs: vec![PalsCollectionGraphPin {
                role: "public".into(),
                sha256: "04".repeat(32),
                serialized_bytes: 1024,
            }],
            runtime_sha256: "05".repeat(32),
            encoding_sha256: "06".repeat(32),
            encoder_source_sha256: "07".repeat(32),
            cpu_configuration_sha256: "08".repeat(32),
            frozen_epoch: 1,
            training_state: "untrained".into(),
            provider: "cpu".into(),
            intra_threads: 1,
            cache_public_memory: false,
        }
    }
    fn refinement_bytes(base: &PalsNativeCollectionRegistry) -> Vec<u8> {
        serde_json::to_vec(&RefinementRegistrationWire {
            version: REFINEMENT_REGISTRATION_VERSION.into(),
            base_registry_canonical_sha256: canonical_sha256(base).unwrap(),
            collector_binary_sha256: base.collector_binary_sha256.clone(),
            search_policy: PalsSearchPolicyIdentityV3::expected_same_repaired_line_once_v1(),
        })
        .unwrap()
    }
    fn admitted_refinement(bytes: &[u8]) -> Result<PalsNativeRefinementRegistration, ArenaError> {
        PalsNativeRefinementRegistration::from_registration_bytes(
            bytes,
            &format!("{:x}", Sha256::digest(bytes)),
        )
    }

    #[test]
    fn refinement_registration_retains_actual_raw_pin_and_preserves_base_registry_wire() {
        let base = refinement_base();
        let before = serde_json::to_vec(&base).unwrap();
        let canonical_before = canonical_sha256(&base).unwrap();
        let bytes = refinement_bytes(&base);
        let registration = admitted_refinement(&bytes).unwrap();
        registration.validate_against(&base).unwrap();
        assert_eq!(
            registration.raw_sha256(),
            format!("{:x}", Sha256::digest(&bytes))
        );
        assert_eq!(serde_json::to_vec(&registration).unwrap(), bytes);
        assert_eq!(serde_json::to_vec(&base).unwrap(), before);
        assert_eq!(canonical_sha256(&base).unwrap(), canonical_before);
        assert!(
            !serde_json::to_value(&base)
                .unwrap()
                .as_object()
                .unwrap()
                .contains_key("pals_search_policy")
        );
    }

    #[test]
    fn refinement_registration_rejects_raw_pin_size_duplicate_unknown_null_and_policy_forgery() {
        let base = refinement_base();
        let bytes = refinement_bytes(&base);
        assert!(
            PalsNativeRefinementRegistration::from_registration_bytes(&bytes, &"09".repeat(32))
                .is_err()
        );
        assert!(admitted_refinement(&[]).is_err());
        assert!(admitted_refinement(&vec![b' '; MAX_REFINEMENT_REGISTRATION_BYTES + 1]).is_err());
        let original: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        for changed in [
            {
                let mut v = original.clone();
                v["extra"] = serde_json::json!(true);
                v
            },
            {
                let mut v = original.clone();
                v["search_policy"] = serde_json::Value::Null;
                v
            },
            {
                let mut v = original.clone();
                v["search_policy"]["policy"] = serde_json::json!("disabled");
                v
            },
            {
                let mut v = original.clone();
                v["search_policy"]["conditions_sha256"] = serde_json::json!([0]);
                v
            },
            {
                let mut v = original.clone();
                v["collector_binary_sha256"] = serde_json::json!("AA".repeat(32));
                v
            },
        ] {
            assert!(admitted_refinement(&serde_json::to_vec(&changed).unwrap()).is_err());
        }
        let text = std::str::from_utf8(&bytes).unwrap();
        let duplicate = text.replacen(
            "\"version\":",
            "\"version\":\"rz-pals-native-refinement-registration/1\",\"version\":",
            1,
        );
        assert!(admitted_refinement(duplicate.as_bytes()).is_err());
        let duplicate_nested = text.replacen(
            "\"policy\":",
            "\"policy\":\"same_repaired_line_once_v1\",\"policy\":",
            1,
        );
        assert!(admitted_refinement(duplicate_nested.as_bytes()).is_err());
    }

    #[test]
    fn refinement_registration_binds_registry_canonical_identity_and_binary_separately() {
        let base = refinement_base();
        let registration = admitted_refinement(&refinement_bytes(&base)).unwrap();
        let mut changed = base.clone();
        changed.cache_public_memory = true;
        assert!(registration.validate_against(&changed).is_err());
        let mut wire: RefinementRegistrationWire =
            serde_json::from_slice(&refinement_bytes(&base)).unwrap();
        wire.collector_binary_sha256 = "09".repeat(32);
        let independent = admitted_refinement(&serde_json::to_vec(&wire).unwrap()).unwrap();
        assert!(independent.validate_against(&base).is_err());
    }

    #[test]
    fn single_reply_native_witness_refuses_actual_c_continuation_registration() {
        let base = refinement_base();
        let mut wire: RefinementRegistrationWire =
            serde_json::from_slice(&refinement_bytes(&base)).unwrap();
        wire.search_policy = PalsSearchPolicyIdentityV3::expected_actual_opponent_continuation_v1();
        wire.search_policy.validate().unwrap();
        assert!(admitted_refinement(&serde_json::to_vec(&wire).unwrap()).is_err());
        let actual = PostRepairRecheckPolicy::ActualOpponentContinuationV1;
        assert!(
            observed_refinement_selection(
                None,
                actual,
                actual.search_identity(),
                actual.conditions()
            )
            .is_err()
        );
    }

    #[test]
    fn selected_native_return_identity_keeps_each_closed_tuple_and_checks_actual_getters() {
        let legacy = PostRepairRecheckPolicy::SameRepairedLineOnceV1;
        let continuation = PostRepairRecheckPolicy::ActualOpponentContinuationV1;
        for (policy, expected, other) in [
            (
                legacy,
                PalsSearchPolicyIdentityV3::expected_same_repaired_line_once_v1(),
                continuation,
            ),
            (
                continuation,
                PalsSearchPolicyIdentityV3::expected_actual_opponent_continuation_v1(),
                legacy,
            ),
        ] {
            assert_eq!(
                observed_refinement_identity(policy, policy.search_identity(), policy.conditions())
                    .unwrap(),
                expected
            );
            assert!(
                observed_refinement_identity(policy, other.search_identity(), policy.conditions())
                    .is_err()
            );
            assert!(
                observed_refinement_identity(policy, policy.search_identity(), other.conditions())
                    .is_err()
            );
            assert!(observed_refinement_identity(policy, policy.search_identity(), None).is_err());
        }
        assert!(
            observed_refinement_identity(
                PostRepairRecheckPolicy::Disabled,
                PALS_SEARCH_VERSION,
                None
            )
            .is_err()
        );
    }

    #[test]
    fn refinement_selection_checks_actual_engine_identity_conditions_and_legacy_disabled() {
        use rz_search::pals::engine::{
            POST_REPAIR_RECHECK_CONDITIONS, POST_REPAIR_RECHECK_SEARCH_VERSION,
        };
        let base = refinement_base();
        let registration = admitted_refinement(&refinement_bytes(&base)).unwrap();
        assert_eq!(
            observed_refinement_selection(
                None,
                PostRepairRecheckPolicy::Disabled,
                PALS_SEARCH_VERSION,
                None
            )
            .unwrap(),
            None
        );
        let observed = observed_refinement_selection(
            Some(&registration),
            PostRepairRecheckPolicy::SameRepairedLineOnceV1,
            POST_REPAIR_RECHECK_SEARCH_VERSION,
            Some(POST_REPAIR_RECHECK_CONDITIONS),
        )
        .unwrap()
        .unwrap();
        assert_eq!(
            observed,
            PalsSearchPolicyIdentityV3::expected_same_repaired_line_once_v1()
        );
        assert!(
            observed_refinement_selection(
                None,
                PostRepairRecheckPolicy::SameRepairedLineOnceV1,
                POST_REPAIR_RECHECK_SEARCH_VERSION,
                Some(POST_REPAIR_RECHECK_CONDITIONS)
            )
            .is_err()
        );
        assert!(
            observed_refinement_selection(
                Some(&registration),
                PostRepairRecheckPolicy::Disabled,
                PALS_SEARCH_VERSION,
                None
            )
            .is_err()
        );
        assert!(
            observed_refinement_selection(
                Some(&registration),
                PostRepairRecheckPolicy::SameRepairedLineOnceV1,
                PALS_SEARCH_VERSION,
                Some(POST_REPAIR_RECHECK_CONDITIONS)
            )
            .is_err()
        );
        assert!(
            observed_refinement_selection(
                Some(&registration),
                PostRepairRecheckPolicy::SameRepairedLineOnceV1,
                POST_REPAIR_RECHECK_SEARCH_VERSION,
                Some("changed conditions")
            )
            .is_err()
        );
    }

    #[test]
    fn producer_checked_source_pin_covers_optional_refinement_registration_facts() {
        let shared = sink(&Position::startpos(), 1024 * 1024);
        let handle = fixture_producer(&shared, 32 * 1024);
        let mut source = shared.lock().unwrap().source.clone();
        handle.verify_source(&source).unwrap();
        let base = refinement_base();
        let registration = admitted_refinement(&refinement_bytes(&base)).unwrap();
        source.native = Some(serde_json::json!({
            "refinement_registration":registration,
            "refinement_registration_sha256":registration.raw_sha256(),
            "pals_search_policy":PalsSearchPolicyIdentityV3::expected_same_repaired_line_once_v1(),
            "search_version":rz_search::pals::engine::POST_REPAIR_RECHECK_SEARCH_VERSION,
        }));
        assert!(handle.verify_source(&source).is_err());
    }

    static NEXT_NATIVE_OUTPUT: AtomicU64 = AtomicU64::new(1);
    struct NativeTestOutput(PathBuf);
    impl NativeTestOutput {
        fn new() -> Self {
            let root = std::env::temp_dir().join(format!(
                "rz-pals-native-descriptor-test-{}-{}",
                std::process::id(),
                NEXT_NATIVE_OUTPUT.fetch_add(1, Ordering::Relaxed)
            ));
            std::fs::create_dir(&root).unwrap();
            Self(root)
        }
    }
    impl Drop for NativeTestOutput {
        fn drop(&mut self) {
            let (Ok(root), Ok(temp)) = (self.0.canonicalize(), std::env::temp_dir().canonicalize())
            else {
                return;
            };
            if root.parent() == Some(temp.as_path())
                && root.file_name().is_some_and(|name| {
                    name.to_string_lossy()
                        .starts_with("rz-pals-native-descriptor-test-")
                })
            {
                let _ = std::fs::remove_dir_all(root);
            }
        }
    }

    // Declared fake weight identities are confined to this observer fixture:
    // these tests never load, register, execute or claim an actual NN asset.
    fn sink(position: &Position, credit: u64) -> Arc<Mutex<Sink>> {
        let configuration = serde_json::to_value(PalsModelConfig::baseline()).unwrap();
        Arc::new(Mutex::new(Sink {
            context: Some(PalsNativeCaptureContext {
                position: position.clone(),
                opening: PalsCollectionOpening::default(),
                game: "fixture-g1".into(),
                actual_moves: Vec::new(),
                sequence: 4,
                max_bytes: credit,
            }),
            source: PalsCollectionSourceDescription {
                mode: "observer-fixture".into(),
                source: PalsInputSource::OwnPals {
                    model_configuration_sha256: canonical_sha256(&configuration).unwrap(),
                    model_weights_sha256: "07".repeat(32),
                },
                cpu_profile_sha256: "03".repeat(32),
                implementation_sha256: "04".repeat(32),
                encoding_sha256: native_encoding_sha().unwrap(),
                encoder_source_sha256: native_source_sha().unwrap(),
                configuration,
                model_epoch: [7; 32],
                model_epoch_kind: "frozen_model_epoch".into(),
                frozen_epoch: 1,
                native: None,
            },
            trace: PalsNativeTrace {
                sequence: 4,
                ..PalsNativeTrace::default()
            },
            started: Instant::now(),
            calls: BTreeMap::new(),
            raw_sources: BTreeSet::new(),
            total_rows: 0,
            producer: None,
            pending_recheck: None,
            pending_continuation: None,
            rejected_recheck: None,
            closed_rechecks: Vec::new(),
            prepaid_recheck_rows: 0,
            prepaid_native_rows: 0,
            pending_search_return: None,
        }))
    }
    fn raw(count: usize) -> PalsRawOutput {
        PalsRawOutput {
            candidate_logits: vec![0.; count],
            wdl_logits: [0.; 3],
            divergence_logits: None,
            task_logits: None,
            private_latent: vec![0.; PalsModelConfig::baseline().latent_elements()],
        }
    }
    fn id(sequence: u64) -> RequestId {
        RequestId::new(ProcessEpoch(17), sequence)
    }
    fn fixture_producer(shared: &Arc<Mutex<Sink>>, journal_bytes: u64) -> RegisteredProducerHandle {
        let source = shared.lock().unwrap().source.clone();
        let registration = serde_json::json!({"version":"rz-pals-collector-producer-registration/1",
            "producer_id":"observer-fixture-owner","source":source.source,"frozen_epoch":source.frozen_epoch,
            "encoding_policy":{"kind":"native_exact","encoding_sha256":source.encoding_sha256,
                "encoder_source_sha256":source.encoder_source_sha256,
                "native_model_epoch":{"kind":"frozen_model_epoch","sha256":hex(source.model_epoch)}},
            "checked_source_sha256":canonical_sha256(&("rz-pals-collector-checked-source/1",&source)).unwrap()});
        let bytes = serde_json::to_vec(&registration).unwrap();
        let config = PalsProducerCollectionConfig::from_registration_bytes(
            &bytes,
            &format!("{:x}", Sha256::digest(&bytes)),
        )
        .unwrap()
        .with_metadata_limits(journal_bytes, 16 * 1024)
        .unwrap();
        let handle = RegisteredProducerHandle::admit(
            &config,
            &PalsCollectionConfig {
                run_id: "fixture".into(),
                ..PalsCollectionConfig::default()
            },
            &source,
        )
        .unwrap();
        shared.lock().unwrap().producer = Some(handle.clone());
        handle
    }
    // These exercise actual collector seal/accounting functions with synthetic
    // owner JSON and an Err return supplied by the fixture. No engine.search,
    // model/CPU/NN execution, worker join or process success is claimed.
    fn return_fixture(credit: u64) -> (Arc<Mutex<Sink>>, PalsLimits) {
        let fixture = RecheckFixture::new();
        (fixture.admitted_sink(credit), fixture.limits)
    }
    fn fixture_owner_snapshot() -> serde_json::Value {
        serde_json::json!({"fixture_scope":"synthetic-owner-snapshot-not-actual-native-execution",
            "physical_shutdown_confirmed":false,"native_buffers_released":false,
            "physical_runs_in_flight":0,"backend_stats":null})
    }
    #[test]
    fn observed_return_reservation_checks_bytes_rows_and_descriptor_before_mutation() {
        for axis in ["bytes", "rows", "descriptor"] {
            let (shared, limits) = return_fixture(if axis == "bytes" {
                SEARCH_RETURN_RESERVED_BYTES - 1
            } else {
                1024 * 1024
            });
            let mut s = shared.lock().unwrap();
            if axis == "rows" {
                s.prepaid_recheck_rows = MAX_ROWS * 8 - 2;
            }
            let before = (
                s.trace.reserved_bytes,
                s.prepaid_native_rows,
                s.prepaid_recheck_rows,
                s.trace.rows.len(),
            );
            let owner = if axis == "descriptor" {
                serde_json::json!({"oversize":"x".repeat(SEARCH_RETURN_ROW_BYTES)})
            } else {
                fixture_owner_snapshot()
            };
            assert!(
                s.prepare_search_return(limits, &AtomicBool::new(false), owner)
                    .is_err(),
                "{axis}"
            );
            assert_eq!(
                (
                    s.trace.reserved_bytes,
                    s.prepaid_native_rows,
                    s.prepaid_recheck_rows,
                    s.trace.rows.len()
                ),
                before,
                "{axis}"
            );
            assert!(s.pending_search_return.is_none());
        }
    }
    #[test]
    fn observed_return_disabled_sink_keeps_legacy_summary_bytes_and_no_new_rows() {
        let shared = sink(&Position::startpos(), 1024 * 1024);
        let mut s = shared.lock().unwrap();
        let limits = RecheckFixture::new().limits;
        assert!(
            s.prepare_search_return(limits, &AtomicBool::new(false), fixture_owner_snapshot())
                .is_err()
        );
        assert!(s.pending_search_return.is_none());
        assert_eq!(s.prepaid_native_rows, 0);
        assert!(s.trace.rows.is_empty());
        let summary = serde_json::json!({"domain":"rz-pals-native-search-work/1","cpu_nodes":0,
            "cpu_tasks_requested":0,"cpu_reports_returned":0,"cpu_work_observation_incomplete":false,
            "cpu_task_configuration_sha256":s.source.cpu_profile_sha256,"search_result":null,
            "role_calls":0,"search_consumed_role_outputs":0});
        s.reserve(8192).unwrap();
        s.row("native-work-summary.jsonl", &summary).unwrap();
        assert_eq!(s.trace.rows.len(), 1);
        assert_eq!(s.trace.rows[0].artifact, "native-work-summary.jsonl");
        assert_eq!(s.trace.rows[0].json, serde_json::to_vec(&summary).unwrap());
        assert_eq!(s.trace.reserved_bytes, 8192);
    }
    #[test]
    fn observed_return_slots_are_exclusive_and_counters_none_stays_unknown() {
        let (shared, limits) = return_fixture(1024 * 1024);
        let mut s = shared.lock().unwrap();
        let cancel = AtomicBool::new(false);
        s.prepare_search_return(limits, &cancel, fixture_owner_snapshot())
            .unwrap();
        assert_eq!(s.trace.reserved_bytes, 139264);
        assert_eq!(s.prepaid_native_rows, 2);
        assert!(
            s.search_return_row(SearchReturnSlot::Returned, Vec::new())
                .is_err()
        );
        assert_eq!(s.prepaid_native_rows, 2);
        s.prepaid_recheck_rows = MAX_ROWS * 8 - 3;
        assert!(
            s.row(
                "native-events.jsonl",
                &serde_json::json!({"unrelated":true})
            )
            .is_err()
        );
        assert_eq!(s.prepaid_native_rows, 2);
        s.prepaid_recheck_rows = 0;
        let returned = Instant::now();
        let result = Err(PalsError::InvalidLimits);
        s.observed_search_return(
            &result,
            None,
            None,
            returned,
            false,
            returned,
            returned,
            fixture_owner_snapshot(),
            &cancel,
        )
        .unwrap();
        let raw: serde_json::Value = serde_json::from_slice(&s.trace.rows[1].json).unwrap();
        assert_eq!(raw["actual_result"]["kind"], "Err");
        assert_eq!(raw["actual_result"]["pals_error_variant"], "InvalidLimits");
        assert!(raw["last_search_counters"].is_null());
        assert!(raw["physical_nn_rows"].is_null());
        assert_eq!(
            raw["native_owner_after"]["physical_shutdown_confirmed"],
            false
        );
        assert_eq!(raw["whole_action_cost_authority"], false);
        assert_eq!(raw["search_return_is_worker_join"], false);
        assert_eq!(
            raw["before_search_row_sha256"],
            format!("{:x}", Sha256::digest(&s.trace.rows[0].json))
        );
        assert!(
            s.observed_search_return(
                &result,
                None,
                None,
                returned,
                false,
                returned,
                returned,
                fixture_owner_snapshot(),
                &cancel
            )
            .is_err()
        );
        let summary = serde_json::json!({"domain":"rz-pals-native-search-work/1","cpu_nodes":0});
        s.search_summary_row(&summary).unwrap();
        assert!(s.search_summary_row(&summary).is_err());
        assert_eq!(s.prepaid_native_rows, 0);
        assert_eq!(s.trace.rows.len(), 3);
        assert_eq!(s.trace.rows[2].json, serde_json::to_vec(&summary).unwrap());
        s.preserve_incomplete_return_failure();
        assert!(s.trace.failure.is_none());
    }
    #[test]
    fn observed_return_failure_keeps_summary_credit_and_is_unresolved_on_drain() {
        let (shared, limits) = return_fixture(1024 * 1024);
        let mut s = shared.lock().unwrap();
        let cancel = AtomicBool::new(false);
        s.prepare_search_return(limits, &cancel, fixture_owner_snapshot())
            .unwrap();
        let at = Instant::now();
        assert!(
            s.observed_search_return(
                &Err(PalsError::Capacity),
                None,
                None,
                at,
                false,
                at,
                at,
                serde_json::json!({"oversize":"x".repeat(SEARCH_RETURN_ROW_BYTES)}),
                &cancel
            )
            .is_err()
        );
        assert_eq!(s.prepaid_native_rows, 2);
        s.search_summary_row(&serde_json::json!({"domain":"rz-pals-native-search-work/1"}))
            .unwrap();
        assert_eq!(s.prepaid_native_rows, 1);
        s.preserve_incomplete_return_failure();
        assert!(
            s.trace
                .failure
                .as_ref()
                .unwrap()
                .contains("reserved return or summary row is missing")
        );
        assert_eq!(s.trace.rows.len(), 2);
    }
    #[test]
    fn observed_return_root_preserves_unknown_prefix_and_rejects_oversized_actual_extent() {
        let (shared, limits) = return_fixture(1024 * 1024);
        let mut s = shared.lock().unwrap();
        let start_fen = Position::startpos().to_fen();
        s.context.as_mut().unwrap().position = Position::from_fen(&start_fen).unwrap();
        s.prepare_search_return(limits, &AtomicBool::new(false), fixture_owner_snapshot())
            .unwrap();
        let raw: serde_json::Value = serde_json::from_slice(&s.trace.rows[0].json).unwrap();
        assert_eq!(raw["root"]["board_fen"], start_fen);
        assert_eq!(raw["root"]["history_origin"], "Fen");
        assert_eq!(raw["root"]["history_completeness"], "UnknownPrefix");
        assert_eq!(raw["root"]["known_history_moves"], serde_json::json!([]));
        drop(s);
        let (shared, limits) = return_fixture(1024 * 1024);
        let mut s = shared.lock().unwrap();
        s.context.as_mut().unwrap().actual_moves =
            vec![BoardMove::from_uci("e2e4").unwrap(); SEARCH_RETURN_MAX_HISTORY_PLIES + 1];
        assert!(
            s.prepare_search_return(limits, &AtomicBool::new(false), fixture_owner_snapshot())
                .is_err()
        );
        assert_eq!(s.trace.reserved_bytes, 0);
        assert!(s.trace.rows.is_empty());
    }
    #[test]
    fn observed_return_uses_original_clock_and_records_cancel_expiry_without_join_claim() {
        let (shared, mut limits) = return_fixture(1024 * 1024);
        let mut s = shared.lock().unwrap();
        // A one-nanosecond subtraction can round to the same Instant on Windows.
        limits.deadline = s.started.checked_sub(Duration::from_millis(1)).unwrap();
        let cancel = AtomicBool::new(true);
        s.prepare_search_return(limits, &cancel, fixture_owner_snapshot())
            .unwrap();
        let at = Instant::now();
        s.observed_search_return(
            &Err(PalsError::InvalidLimits),
            None,
            Some("synthetic observer failure"),
            at,
            true,
            at,
            at,
            fixture_owner_snapshot(),
            &cancel,
        )
        .unwrap();
        let before: serde_json::Value = serde_json::from_slice(&s.trace.rows[0].json).unwrap();
        let after: serde_json::Value = serde_json::from_slice(&s.trace.rows[1].json).unwrap();
        assert_eq!(
            before["limits"]["original_deadline"],
            after["original_deadline"]
        );
        assert_eq!(after["original_deadline"]["relation"], "before_origin");
        assert_eq!(after["deadline_expired_at_search_return"], true);
        assert_eq!(after["cancelled_at_return_observer"], true);
        assert_eq!(after["query_action_causal_authority"], false);
        assert_eq!(after["utility_authority"], false);
        assert_eq!(after["training_target_authority"], false);
        assert!(after["clock"]["serialization_completed_tick"].is_null());
        assert!(
            after["clock"]["serialization_started_tick"]
                .as_u64()
                .unwrap()
                >= after["clock"]["search_returned_tick"].as_u64().unwrap()
        );
    }
    #[test]
    fn missing_physical_secondary_preserves_search_primary_and_original_error_return() {
        let shared = sink(&Position::startpos(), 1024 * 1024);
        let mut s = shared.lock().unwrap();
        s.trace.failure = Some("search primary: PALS: InvalidConfig; native trace: none".into());
        native_trace_failure(
            &mut s.trace,
            "recheck observer",
            "synthetic observer failure",
        );
        s.rejected_recheck = Some(RecheckFixture::new().identity());
        native_trace_secondary(
            &mut s.trace,
            "observed search return",
            "synthetic envelope failure",
        );
        native_trace_secondary(&mut s.trace, "summary", "synthetic summary failure");
        s.calls.insert(
            id(7),
            Call {
                input_sha256: "07".repeat(32),
                physical: false,
                delivered: false,
                accepted: false,
                rejected: false,
                physical_unknown: false,
                logical: None,
                prepaid_rows: 0,
            },
        );
        s.preserve_incomplete_return_failure();
        let failure = s.trace.failure.as_ref().unwrap();
        assert!(failure.starts_with("search primary: PALS: InvalidConfig; native trace: none"));
        assert!(failure.contains("recheck observer secondary: synthetic observer failure"));
        assert!(failure.contains("missing recheck finish secondary: native recheck returned without its prepaid finish or rejected-preparation close"));
        assert!(failure.contains("observed search return secondary: synthetic envelope failure"));
        assert!(failure.contains("summary secondary: synthetic summary failure"));
        assert!(failure.contains("missing physical/rejection secondary: native request 17:7"));
        let returned = search_after_summary::<()>(
            Err(PalsError::InvalidConfig),
            Err(invalid("synthetic envelope failure")),
        )
        .unwrap_err();
        assert!(returned.to_string().contains("InvalidConfig"));
        assert!(!returned.to_string().contains("envelope failure"));
        s.trace.failure = None;
        s.rejected_recheck = None;
        s.preserve_incomplete_return_failure();
        assert_eq!(
            s.trace.failure.as_deref(),
            Some("native request 17:7 returned without physical/rejection evidence")
        );
    }
    // Observer-only fixture: Rules replay and real seal/credit functions run,
    // but store handles/policy logits are explicitly synthetic. No NN or CPU
    // task, learned asset, inference acceptance or training target is claimed.
    struct RecheckFixture {
        root: Position,
        repaired: Vec<BoardMove>,
        refutation: Vec<BoardMove>,
        root_snapshot: rz_position::PositionSnapshot,
        anchor_snapshot: rz_position::PositionSnapshot,
        repaired_snapshot: rz_position::PositionSnapshot,
        reply_context: RoleLogicalContext,
        record: rz_search::pals::engine::RoleRecord,
        checker: rz_search::cpu_checker::CheckerIdentity,
        config: PalsConfig,
        limits: PalsLimits,
        cancel: AtomicBool,
    }
    impl RecheckFixture {
        fn new() -> Self {
            let root = Position::startpos();
            let repaired = ["e2e4", "e7e5", "g1f3", "b8c6"]
                .map(|m| BoardMove::from_uci(m).unwrap())
                .to_vec();
            let refutation = ["e2e4", "e7e5", "f1c4", "b8c6"]
                .map(|m| BoardMove::from_uci(m).unwrap())
                .to_vec();
            let (anchor, _) = replay_line(&root, &repaired[..3]).unwrap();
            let (endpoint, _) = replay_line(&root, &repaired).unwrap();
            let root_snapshot = root.snapshot();
            Self {
                root,
                root_snapshot,
                anchor_snapshot: anchor.snapshot(),
                repaired_snapshot: endpoint.snapshot(),
                reply_context: Self::logical(&repaired[..3], NativeQueryKind::Reply, 8),
                record: rz_search::pals::engine::RoleRecord {
                    revision: 8,
                    origin_state: rz_search::pals::store::StateId(0),
                    kind: RecordKind::Repair,
                    line: repaired.clone(),
                    value: None,
                    completed_depth: 0,
                    score_scope: None,
                    cpu_observation: None,
                    parent_revision: None,
                    supersedes_revision: None,
                    perspective: Color::White,
                    critical: true,
                },
                repaired,
                refutation,
                checker: rz_search::cpu_checker::CheckerIdentity::Owned(
                    CpuEngine::new(CpuConfig::default())
                        .unwrap()
                        .value_identity()
                        .clone(),
                ),
                config: PalsConfig::default(),
                limits: PalsLimits {
                    deadline: Instant::now() + Duration::from_secs(30),
                    max_rounds: 2,
                    max_cpu_nodes: 16384,
                    cpu_depth: 2,
                },
                cancel: AtomicBool::new(false),
            }
        }
        fn logical(
            prefix: &[BoardMove],
            kind: NativeQueryKind,
            revision: u64,
        ) -> RoleLogicalContext {
            RoleLogicalContext {
                game_generation: 1,
                search_generation: 1,
                situation: rz_search::pals::store::SituationId {
                    slot: prefix.len(),
                    generation: 1,
                },
                state: rz_search::pals::store::StateId(prefix.len()),
                focus: rz_search::pals::store::LineId(prefix.len()),
                purpose: if kind == NativeQueryKind::Repair {
                    RoleQueryPurpose::RepairPolicy
                } else {
                    RoleQueryPurpose::ReplyPolicy
                },
                prefix: prefix.to_vec(),
                focus_sha256: [1; 32],
                prefix_sha256: [2; 32],
                proposal_sha256: [3; 32],
                refutation_sha256: Some([4; 32]),
                divergence_sha256: [5; 32],
                public_revision: revision,
                situation_revision: 1,
            }
        }
        fn identity(&self) -> RecheckIdentity {
            RecheckIdentity {
                game_generation: 1,
                search_generation: 1,
                root: rz_search::pals::store::SituationId {
                    slot: 0,
                    generation: 1,
                },
                root_revision: 1,
                repair_record_revision: 8,
                repaired_line: rz_search::pals::store::LineId(4),
            }
        }
        fn endpoint(&self) -> RecheckEndpoint<'_> {
            RecheckEndpoint {
                state: rz_search::pals::store::StateId(4),
                situation: rz_search::pals::store::SituationId {
                    slot: 4,
                    generation: 1,
                },
                snapshot: &self.repaired_snapshot,
                evidence: RecheckEndpointEvidence::Unobserved,
            }
        }
        fn prepared(&self) -> RecheckPrepared<'_> {
            RecheckPrepared {
                policy: PostRepairRecheckPolicy::SameRepairedLineOnceV1,
                identity: self.identity(),
                root_state: rz_search::pals::store::StateId(0),
                root_snapshot: &self.root_snapshot,
                anchor_state: self.reply_context.state,
                anchor_situation: self.reply_context.situation,
                anchor_snapshot: &self.anchor_snapshot,
                anchor_ply: 3,
                examined_responses: &[],
                reply_context: &self.reply_context,
                repair_record: &self.record,
                repaired: &self.repaired,
                refutation: &self.refutation,
                repaired_endpoint: self.endpoint(),
                limits: self.limits,
                deadline_tick: 1000000,
                cancel: &self.cancel,
                config: &self.config,
                checker_identity: &self.checker,
                cpu_condition: "observer-fixture-cpu-condition-not-executed",
            }
        }
        fn finished(
            &self,
            prepared_accepted: bool,
            reply_call_attempted: bool,
            reply_accepted: bool,
        ) -> RecheckFinished<'_> {
            RecheckFinished {
                policy: PostRepairRecheckPolicy::SameRepairedLineOnceV1,
                identity: self.identity(),
                prepared_accepted,
                reply_call_attempted,
                reply_accepted,
                reply_context: &self.reply_context,
                selected_response: None,
                counterline: &[],
                full_suffix_replayed: false,
                counterline_completed: false,
                repaired_endpoint: self.endpoint(),
                counter_endpoint: None,
                comparable: false,
                publication: None,
                disposition: if prepared_accepted {
                    rz_search::pals::engine::RecheckDisposition::NoAlternativeResponse
                } else {
                    rz_search::pals::engine::RecheckDisposition::PreparedRejected
                },
                original_error: None,
                limits: self.limits,
                deadline_tick: 1000000,
                cancel: &self.cancel,
            }
        }
        fn admitted_sink(&self, credit: u64) -> Arc<Mutex<Sink>> {
            let shared = sink(&self.root, credit);
            shared.lock().unwrap().source.native = Some(serde_json::json!({
                "pals_search_policy":PalsSearchPolicyIdentityV3::expected_same_repaired_line_once_v1(),
                "refinement_registration_sha256":"09".repeat(32)}));
            fixture_producer(&shared, 512 * 1024);
            shared
        }
        fn observe_call(
            &self,
            shared: &Arc<Mutex<Sink>>,
            sequence: u64,
            kind: NativeQueryKind,
            ply: usize,
            context: &RoleLogicalContext,
        ) -> Result<(), RoleError> {
            self.observe_prefix(
                shared,
                sequence,
                kind,
                &self.repaired[..ply],
                self.repaired[ply],
                context,
            )
        }
        fn observe_prefix(
            &self,
            shared: &Arc<Mutex<Sink>>,
            sequence: u64,
            kind: NativeQueryKind,
            prefix: &[BoardMove],
            chosen: BoardMove,
            context: &RoleLogicalContext,
        ) -> Result<(), RoleError> {
            self.observe_ranked_prefix(shared, sequence, kind, prefix, &[(chosen, 1.)], context)
        }
        fn observe_ranked_prefix(
            &self,
            shared: &Arc<Mutex<Sink>>,
            sequence: u64,
            kind: NativeQueryKind,
            prefix: &[BoardMove],
            logits: &[(BoardMove, f32)],
            context: &RoleLogicalContext,
        ) -> Result<(), RoleError> {
            let (position, _) = replay_line(&self.root, prefix).unwrap();
            let legal = position.ordered_legal_moves().moves().to_vec();
            let records = if kind == NativeQueryKind::Reply {
                vec![self.record.clone()]
            } else {
                Vec::new()
            };
            let query = RoleQuery {
                position: &position,
                legal: &legal,
                prefix,
                proposal: &self.repaired,
                counterexample: Some(&self.refutation),
                records: &records,
                revision: context.public_revision,
                deadline: self.limits.deadline,
                cancel: &self.cancel,
            };
            let input = prepare_role_input(&query, kind, [7; 32]).unwrap();
            let mut observer = Observer(Arc::clone(shared));
            observer.prepared_with_logical_context(
                id(sequence),
                &input,
                NativePreparedContext::Role {
                    query: &query,
                    kind,
                },
                Some(context),
            )?;
            let mut output = raw(legal.len());
            output.candidate_logits.fill(-1.);
            for (movement, logit) in logits {
                output.candidate_logits[legal.iter().position(|m| m == movement).unwrap()] = *logit;
            }
            observer.physically_completed(id(sequence), Ok(&output))?;
            observer.delivered(id(sequence))?;
            observer.accepted(id(sequence))?;
            observer.accepted_context(
                id(sequence),
                &RoleAcceptance {
                    snapshot: &position.snapshot(),
                    context,
                    deadline: self.limits.deadline,
                    cancel: &self.cancel,
                },
            )
        }
        fn parent_calls(&self, shared: &Arc<Mutex<Sink>>) {
            for (sequence, ply) in [(1, 2), (2, 3)] {
                self.observe_call(
                    shared,
                    sequence,
                    NativeQueryKind::Repair,
                    ply,
                    &Self::logical(&self.repaired[..ply], NativeQueryKind::Repair, 7),
                )
                .unwrap();
            }
        }
    }
    #[test]
    fn recheck_descriptor_actual_chain_reply_seals_and_finish_are_separate_prepaid_rows() {
        let f = RecheckFixture::new();
        let shared = f.admitted_sink(4 * 1024 * 1024);
        f.parent_calls(&shared);
        let mut observer = Observer(Arc::clone(&shared));
        observer.recheck_prepared(f.prepared()).unwrap();
        {
            let s = shared.lock().unwrap();
            assert_eq!(s.calls.len(), 2);
            assert_eq!(s.prepaid_recheck_rows, 2);
            assert!(s.pending_recheck.as_ref().unwrap().reply.is_none());
        }
        f.observe_call(&shared, 3, NativeQueryKind::Reply, 3, &f.reply_context)
            .unwrap();
        observer
            .recheck_finished(f.finished(true, true, true))
            .unwrap();
        let s = shared.lock().unwrap();
        assert!(s.pending_recheck.is_none());
        assert_eq!(s.prepaid_recheck_rows, 0);
        let rows: Vec<serde_json::Value> = s
            .trace
            .rows
            .iter()
            .filter(|r| r.artifact == RECHECK_ARTIFACT)
            .map(|r| serde_json::from_slice(&r.json).unwrap())
            .collect();
        assert_eq!(
            rows.iter()
                .map(|r| r["stage"].as_str().unwrap())
                .collect::<Vec<_>>(),
            ["prepared", "reply_bound", "finished"]
        );
        assert_eq!(
            rows[0]["data"]["parent_repair_chain"]["calls"]
                .as_array()
                .unwrap()
                .len(),
            2
        );
        assert_eq!(
            rows[1]["data"]["prepared_payload_sha256"],
            rows[0]["payload_sha256"]
        );
        assert_eq!(
            rows[2]["data"]["bound_payload_sha256"],
            rows[1]["payload_sha256"]
        );
        assert_eq!(
            rows[2]["data"]["repaired_endpoint"]["evidence"]["kind"],
            "Unobserved"
        );
        assert!(rows[2]["data"]["publication"].is_null());
        assert!(
            !s.trace
                .rows
                .iter()
                .find(|r| r.artifact == "input-lineage.jsonl")
                .map(|r| serde_json::from_slice::<serde_json::Value>(&r.json).unwrap())
                .unwrap()
                .as_object()
                .unwrap()
                .contains_key("descriptor_sha256")
        );
        let temp = NativeTestOutput::new();
        {
            let mut output =
                Output::new(&temp.0, "recheck-observer-storage", 8 * 1024 * 1024).unwrap();
            output.native_trace(&s.trace).unwrap();
            let stored = std::fs::read(
                temp.0
                    .join("recheck-observer-storage")
                    .join(RECHECK_ARTIFACT),
            )
            .unwrap();
            let expected = s
                .trace
                .rows
                .iter()
                .filter(|r| r.artifact == RECHECK_ARTIFACT)
                .flat_map(|r| r.json.iter().copied().chain(std::iter::once(b'\n')))
                .collect::<Vec<_>>();
            assert_eq!(stored, expected);
        }
    }

    fn continuation_fixture() -> RecheckFixture {
        let mut f = RecheckFixture::new();
        f.repaired = ["e2e4", "e7e5", "g1f3", "b8c6", "d2d3", "d7d6"]
            .map(|m| BoardMove::from_uci(m).unwrap())
            .to_vec();
        f.refutation = ["e2e4", "e7e5", "f1c4", "b8c6", "d2d3", "d7d6"]
            .map(|m| BoardMove::from_uci(m).unwrap())
            .to_vec();
        f.repaired_snapshot = replay_line(&f.root, &f.repaired).unwrap().0.snapshot();
        f.record.line = f.repaired.clone();
        f.config.line_plies = 6;
        f
    }
    fn continuation_sink(f: &RecheckFixture) -> Arc<Mutex<Sink>> {
        let shared = sink(&f.root, 8 * 1024 * 1024);
        shared.lock().unwrap().source.native = Some(serde_json::json!({
            "pals_search_policy":PalsSearchPolicyIdentityV3::expected_actual_opponent_continuation_v1(),
            "refinement_registration_sha256":"09".repeat(32)}));
        fixture_producer(&shared, 1024 * 1024);
        for ply in 2..f.repaired.len() {
            f.observe_call(
                &shared,
                (ply - 1) as u64,
                NativeQueryKind::Repair,
                ply,
                &RecheckFixture::logical(&f.repaired[..ply], NativeQueryKind::Repair, 7),
            )
            .unwrap();
        }
        shared
    }
    fn continuation_prepared(f: &RecheckFixture) -> RecheckPrepared<'_> {
        let mut event = f.prepared();
        event.policy = PostRepairRecheckPolicy::ActualOpponentContinuationV1;
        event
    }
    #[test]
    fn native_continuation_registration_is_separate_and_checks_observed_engine_selection() {
        let base = refinement_base();
        let legacy = refinement_bytes(&base);
        let mut wire: serde_json::Value = serde_json::from_slice(&legacy).unwrap();
        wire["version"] = serde_json::json!("rz-pals-native-continuation-registration/1");
        wire["search_policy"] = serde_json::to_value(
            PalsSearchPolicyIdentityV3::expected_actual_opponent_continuation_v1(),
        )
        .unwrap();
        let bytes = serde_json::to_vec(&wire).unwrap();
        let pin = format!("{:x}", Sha256::digest(&bytes));
        assert!(PalsNativeRefinementRegistration::from_registration_bytes(&bytes, &pin).is_err());
        assert!(
            PalsNativeContinuationRegistration::from_registration_bytes(
                &legacy,
                &format!("{:x}", Sha256::digest(&legacy))
            )
            .is_err()
        );
        assert!(
            PalsNativeContinuationRegistration::from_registration_bytes(&bytes, &"00".repeat(32))
                .is_err()
        );
        let registration =
            PalsNativeContinuationRegistration::from_registration_bytes(&bytes, &pin).unwrap();
        registration.validate_against(&base).unwrap();
        let policy = PostRepairRecheckPolicy::ActualOpponentContinuationV1;
        assert_eq!(
            observed_refinement_selection(
                Some(&registration.inner),
                policy,
                policy.search_identity(),
                policy.conditions()
            )
            .unwrap(),
            Some(PalsSearchPolicyIdentityV3::expected_actual_opponent_continuation_v1())
        );
        assert!(
            observed_refinement_selection(
                Some(&registration.inner),
                PostRepairRecheckPolicy::SameRepairedLineOnceV1,
                policy.search_identity(),
                policy.conditions()
            )
            .is_err()
        );
        assert!(
            observed_refinement_selection(
                Some(&registration.inner),
                policy,
                policy.search_identity(),
                Some("changed")
            )
            .is_err()
        );
        let mut changed = base.clone();
        changed.collector_binary_sha256 = "10".repeat(32);
        assert!(registration.validate_against(&changed).is_err());
    }
    #[test]
    fn native_continuation_seals_three_actual_prefixes_and_persists_a_distinct_trace() {
        let mut f = continuation_fixture();
        // Fresh CPU verification follows Repair and advances the public view;
        // the accepted Repair record and its parent chain remain historical.
        f.reply_context.public_revision = f.record.revision + 1;
        let shared = continuation_sink(&f);
        let counter = ["e2e4", "e7e5", "g1f3", "g8f6", "f1c4", "d7d5"]
            .map(|m| BoardMove::from_uci(m).unwrap());
        let mut observer = Observer(Arc::clone(&shared));
        observer
            .recheck_prepared(continuation_prepared(&f))
            .unwrap();
        for ply in 3..6 {
            let context = if ply == 3 {
                f.reply_context.clone()
            } else {
                RecheckFixture::logical(
                    &counter[..ply],
                    NativeQueryKind::Reply,
                    f.reply_context.public_revision,
                )
            };
            f.observe_prefix(
                &shared,
                ply as u64 + 2,
                NativeQueryKind::Reply,
                &counter[..ply],
                counter[ply],
                &context,
            )
            .unwrap();
        }
        let mut finish = f.finished(true, true, true);
        finish.policy = PostRepairRecheckPolicy::ActualOpponentContinuationV1;
        finish.selected_response = Some(counter[3]);
        finish.counterline = &counter;
        finish.counterline_completed = true;
        finish.disposition = rz_search::pals::engine::RecheckDisposition::IncomparableEvidence;
        observer.recheck_finished(finish).unwrap();
        let s = shared.lock().unwrap();
        assert!(s.pending_continuation.is_none());
        assert_eq!(s.prepaid_recheck_rows, 0);
        assert!(!s.trace.rows.iter().any(|r| r.artifact == RECHECK_ARTIFACT));
        let rows: Vec<serde_json::Value> = s
            .trace
            .rows
            .iter()
            .filter(|r| r.artifact == "native-continuation-traces.jsonl")
            .map(|r| serde_json::from_slice(&r.json).unwrap())
            .collect();
        assert_eq!(
            rows.iter()
                .map(|r| r["stage"].as_str().unwrap())
                .collect::<Vec<_>>(),
            [
                "prepared",
                "reply_bound",
                "reply_bound",
                "reply_bound",
                "finished"
            ]
        );
        let calls = rows[4]["data"]["calls"].as_array().unwrap();
        assert_eq!(calls.len(), 3);
        for (i, call) in calls.iter().enumerate() {
            assert_eq!(call["request_sequence"], serde_json::json!(i + 5));
            assert_eq!(call["bound_payload_sha256"], rows[i + 1]["payload_sha256"]);
            assert_eq!(call["search_consumed"], true);
            assert_eq!(call["accepted_context_checked"], true);
        }
        assert_eq!(rows[4]["data"]["counterline_completed"], true);
        assert_eq!(
            rows[0]["domain"],
            "rz-pals-native-post-repair-continuation/2"
        );
        assert_eq!(rows[0]["data"]["examined_responses"], serde_json::json!([]));
        assert_eq!(rows[0]["identity"]["repair_record_revision"], 8);
        assert_eq!(
            rows[0]["data"]["initial_reply_context"]["public_revision"],
            9
        );
        assert_eq!(rows[4]["data"]["initial_selection_checked"], true);
        assert_eq!(
            rows[4]["data"]["expected_initial_response"],
            serde_json::json!(Move16::pack(counter[3]).unwrap().bits())
        );
        assert_eq!(rows[4]["data"]["full_suffix_replayed"], false);
        assert!(rows[4]["data"]["publication"].is_null()); // No CPU comparison or actual NN fixture claim.
        let temp = NativeTestOutput::new();
        let mut output = Output::new(&temp.0, "continuation-storage", 16 * 1024 * 1024).unwrap();
        output.native_trace(&s.trace).unwrap();
        let stored = std::fs::read(
            temp.0
                .join("continuation-storage/native-continuation-traces.jsonl"),
        )
        .unwrap();
        let expected = s
            .trace
            .rows
            .iter()
            .filter(|r| r.artifact == "native-continuation-traces.jsonl")
            .flat_map(|r| r.json.iter().copied().chain(std::iter::once(b'\n')))
            .collect::<Vec<_>>();
        assert_eq!(stored, expected);
    }
    #[test]
    fn native_continuation_rejects_stale_initial_or_changed_reply_revision() {
        for invalid_revision in [7, 10] {
            let mut invalid = continuation_fixture();
            invalid.reply_context.public_revision = invalid_revision;
            let shared = continuation_sink(&invalid);
            let mut observer = Observer(Arc::clone(&shared));
            assert!(
                observer
                    .recheck_prepared(continuation_prepared(&invalid))
                    .is_err()
            );
            let s = shared.lock().unwrap();
            assert!(s.pending_continuation.is_none());
            assert!(!s.calls.contains_key(&id(5)));
            assert!(
                !s.trace
                    .rows
                    .iter()
                    .any(|r| r.artifact == "native-continuation-traces.jsonl")
            );
        }

        for changed_revision in [8, 10] {
            let mut f = continuation_fixture();
            f.reply_context.public_revision = f.record.revision + 1;
            let shared = continuation_sink(&f);
            let counter = ["e2e4", "e7e5", "g1f3", "g8f6", "f1c4", "d7d5"]
                .map(|m| BoardMove::from_uci(m).unwrap());
            let mut observer = Observer(Arc::clone(&shared));
            observer
                .recheck_prepared(continuation_prepared(&f))
                .unwrap();
            f.observe_prefix(
                &shared,
                5,
                NativeQueryKind::Reply,
                &counter[..3],
                counter[3],
                &f.reply_context,
            )
            .unwrap();
            let context =
                RecheckFixture::logical(&counter[..4], NativeQueryKind::Reply, changed_revision);
            assert!(
                f.observe_prefix(
                    &shared,
                    6,
                    NativeQueryKind::Reply,
                    &counter[..4],
                    counter[4],
                    &context,
                )
                .is_err()
            );
            assert!(!shared.lock().unwrap().calls.contains_key(&id(6)));
        }
    }
    #[test]
    fn native_continuation_rejects_unaccepted_previous_output_or_a_spliced_prefix() {
        let f = continuation_fixture();
        let counter = ["e2e4", "e7e5", "g1f3", "g8f6", "f1c4", "d7d5"]
            .map(|m| BoardMove::from_uci(m).unwrap());
        for mutation in 0..5 {
            let shared = continuation_sink(&f);
            let mut observer = Observer(Arc::clone(&shared));
            observer
                .recheck_prepared(continuation_prepared(&f))
                .unwrap();
            f.observe_prefix(
                &shared,
                5,
                NativeQueryKind::Reply,
                &counter[..3],
                counter[3],
                &f.reply_context,
            )
            .unwrap();
            let mut prefix = counter[..4].to_vec();
            if mutation == 4 {
                prefix[3] = f.repaired[3];
            } else {
                let mut s = shared.lock().unwrap();
                let c = s.calls.get_mut(&id(5)).unwrap();
                match mutation {
                    0 => c.physical = false,
                    1 => c.delivered = false,
                    2 => c.accepted = false,
                    _ => c.physical_unknown = true,
                }
            }
            let context = RecheckFixture::logical(&prefix, NativeQueryKind::Reply, 8);
            assert!(
                f.observe_prefix(
                    &shared,
                    6,
                    NativeQueryKind::Reply,
                    &prefix,
                    counter[4],
                    &context
                )
                .is_err()
            );
            assert!(!shared.lock().unwrap().calls.contains_key(&id(6)));
        }
    }
    #[test]
    fn native_continuation_partial_tail_and_failed_completion_never_claim_full_coverage() {
        let f = continuation_fixture();
        let shared = continuation_sink(&f);
        let counter = ["e2e4", "e7e5", "g1f3", "g8f6", "f1c4", "d7d5"]
            .map(|m| BoardMove::from_uci(m).unwrap());
        let mut observer = Observer(Arc::clone(&shared));
        observer
            .recheck_prepared(continuation_prepared(&f))
            .unwrap();
        f.observe_prefix(
            &shared,
            5,
            NativeQueryKind::Reply,
            &counter[..3],
            counter[3],
            &f.reply_context,
        )
        .unwrap();
        let context = RecheckFixture::logical(&counter[..4], NativeQueryKind::Reply, 8);
        f.observe_prefix(
            &shared,
            6,
            NativeQueryKind::Reply,
            &counter[..4],
            counter[4],
            &context,
        )
        .unwrap();
        let mut forged = f.finished(true, true, true);
        forged.policy = PostRepairRecheckPolicy::ActualOpponentContinuationV1;
        forged.selected_response = Some(counter[3]);
        forged.counterline = &counter;
        forged.counterline_completed = true;
        forged.disposition = rz_search::pals::engine::RecheckDisposition::IncomparableEvidence;
        assert!(observer.recheck_finished(forged).is_err());
        let mut finish = f.finished(true, true, true);
        finish.policy = PostRepairRecheckPolicy::ActualOpponentContinuationV1;
        finish.selected_response = Some(counter[3]);
        finish.counterline = &counter[..5];
        finish.disposition = rz_search::pals::engine::RecheckDisposition::Interrupted;
        observer.recheck_finished(finish).unwrap();
        let s = shared.lock().unwrap();
        assert!(s.trace.failure.is_some()); // Earlier false completion remains primary evidence.
        let end: serde_json::Value =
            serde_json::from_slice(&s.trace.rows.last().unwrap().json).unwrap();
        assert_eq!(end["data"]["counterline_completed"], false);
        assert!(end["data"]["publication"].is_null());
        assert_eq!(end["data"]["calls"].as_array().unwrap().len(), 2);
        assert_eq!(s.prepaid_recheck_rows, 1); // Unused bound slot remains charged through drain.
    }
    #[test]
    fn native_continuation_initial_response_replays_raw_ranking_examined_edges_and_ties() {
        let f = continuation_fixture();
        let (anchor, _) = replay_line(&f.root, &f.repaired[..3]).unwrap();
        let legal = anchor.ordered_legal_moves().moves().to_vec();
        let original = f.repaired[3];
        let first = BoardMove::from_uci("g8f6").unwrap();
        let second = BoardMove::from_uci("d7d6").unwrap();
        let tied = if legal.iter().position(|m| *m == first).unwrap()
            < legal.iter().position(|m| *m == second).unwrap()
        {
            first
        } else {
            second
        };
        let cases = [
            (Vec::new(), 9., 8., first),
            (vec![first], 9., 8., second),
            (legal.clone(), 9., 8., first),
            (Vec::new(), 9., 9., tied),
        ];
        for (examined, first_logit, second_logit, expected) in cases {
            let shared = continuation_sink(&f);
            let mut observer = Observer(Arc::clone(&shared));
            let mut prepared = continuation_prepared(&f);
            prepared.examined_responses = &examined;
            observer.recheck_prepared(prepared).unwrap();
            f.observe_ranked_prefix(
                &shared,
                5,
                NativeQueryKind::Reply,
                &f.repaired[..3],
                &[
                    (original, 100.),
                    (first, first_logit),
                    (second, second_logit),
                ],
                &f.reply_context,
            )
            .unwrap();
            let mut counter = f.repaired[..3].to_vec();
            counter.push(expected);
            let mut finish = f.finished(true, true, true);
            finish.policy = PostRepairRecheckPolicy::ActualOpponentContinuationV1;
            finish.selected_response = Some(expected);
            finish.counterline = &counter;
            finish.disposition = rz_search::pals::engine::RecheckDisposition::IncompleteCounterline;
            observer.recheck_finished(finish).unwrap();
            let s = shared.lock().unwrap();
            let prepared: serde_json::Value = serde_json::from_slice(
                &s.trace
                    .rows
                    .iter()
                    .find(|r| r.artifact == "native-continuation-traces.jsonl")
                    .unwrap()
                    .json,
            )
            .unwrap();
            let finished: serde_json::Value =
                serde_json::from_slice(&s.trace.rows.last().unwrap().json).unwrap();
            assert_eq!(
                prepared["data"]["examined_responses"],
                serde_json::json!(pack(&examined).unwrap())
            );
            assert_eq!(
                finished["data"]["expected_initial_response"],
                serde_json::json!(Move16::pack(expected).unwrap().bits())
            );
            assert_eq!(finished["data"]["initial_selection_checked"], true);
            assert!(finished["data"]["publication"].is_null());
        }
    }

    #[test]
    fn native_continuation_rejects_a_legal_but_wrong_initial_choice_before_tail_dispatch_and_finish()
     {
        let f = continuation_fixture();
        let first = BoardMove::from_uci("g8f6").unwrap();
        let wrong = BoardMove::from_uci("d7d6").unwrap();
        let shared = continuation_sink(&f);
        let mut observer = Observer(Arc::clone(&shared));
        observer
            .recheck_prepared(continuation_prepared(&f))
            .unwrap();
        f.observe_ranked_prefix(
            &shared,
            5,
            NativeQueryKind::Reply,
            &f.repaired[..3],
            &[(first, 9.), (wrong, 8.)],
            &f.reply_context,
        )
        .unwrap();
        let mut counter = f.repaired[..3].to_vec();
        counter.push(wrong);
        let context = RecheckFixture::logical(&counter, NativeQueryKind::Reply, 8);
        assert!(
            f.observe_prefix(
                &shared,
                6,
                NativeQueryKind::Reply,
                &counter,
                BoardMove::from_uci("f1c4").unwrap(),
                &context
            )
            .is_err()
        );
        assert!(!shared.lock().unwrap().calls.contains_key(&id(6)));
        let mut finish = f.finished(true, true, true);
        finish.policy = PostRepairRecheckPolicy::ActualOpponentContinuationV1;
        finish.selected_response = Some(wrong);
        finish.counterline = &counter;
        finish.disposition = rz_search::pals::engine::RecheckDisposition::IncompleteCounterline;
        assert!(observer.recheck_finished(finish).is_err());
        let mut no_alternative = f.finished(true, true, true);
        no_alternative.policy = PostRepairRecheckPolicy::ActualOpponentContinuationV1;
        assert!(observer.recheck_finished(no_alternative).is_err());
        assert!(!shared.lock().unwrap().trace.rows.iter().any(|r| {
            r.artifact == "native-continuation-traces.jsonl"
                && serde_json::from_slice::<serde_json::Value>(&r.json).unwrap()["stage"]
                    == "finished"
        }));
    }

    #[test]
    fn native_continuation_rejects_duplicate_or_illegal_examined_responses_before_submission() {
        let f = continuation_fixture();
        let movement = BoardMove::from_uci("g8f6").unwrap();
        for examined in [
            vec![movement, movement],
            vec![BoardMove::from_uci("e2e4").unwrap()],
        ] {
            let shared = continuation_sink(&f);
            let mut observer = Observer(Arc::clone(&shared));
            let mut prepared = continuation_prepared(&f);
            prepared.examined_responses = &examined;
            assert!(observer.recheck_prepared(prepared).is_err());
            let s = shared.lock().unwrap();
            assert!(s.pending_continuation.is_none());
            assert!(!s.calls.contains_key(&id(5)));
            assert!(
                !s.trace
                    .rows
                    .iter()
                    .any(|r| r.artifact == "native-continuation-traces.jsonl")
            );
        }
    }

    #[test]
    fn native_continuation_reserves_before_dispatch_and_keeps_original_control_context() {
        for cancel in [false, true] {
            let f = continuation_fixture();
            let shared = continuation_sink(&f);
            if cancel {
                f.cancel.store(true, Ordering::Release);
            } else {
                let mut s = shared.lock().unwrap();
                let reserved = s.trace.reserved_bytes;
                s.context.as_mut().unwrap().max_bytes = reserved;
            }
            let mut observer = Observer(Arc::clone(&shared));
            assert!(
                observer
                    .recheck_prepared(continuation_prepared(&f))
                    .is_err()
            );
            let s = shared.lock().unwrap();
            assert!(s.pending_continuation.is_none());
            assert!(!s.calls.contains_key(&id(5)));
        }
        for changed in 0..4 {
            let f = continuation_fixture();
            let shared = continuation_sink(&f);
            let mut observer = Observer(Arc::clone(&shared));
            observer
                .recheck_prepared(continuation_prepared(&f))
                .unwrap();
            let mut context = f.reply_context.clone();
            match changed {
                0 => context.game_generation += 1,
                1 => context.search_generation += 1,
                2 => context.public_revision += 1,
                _ => context.situation_revision += 1,
            }
            assert!(
                f.observe_prefix(
                    &shared,
                    5,
                    NativeQueryKind::Reply,
                    &f.repaired[..3],
                    BoardMove::from_uci("g8f6").unwrap(),
                    &context
                )
                .is_err()
            );
            assert!(!shared.lock().unwrap().calls.contains_key(&id(5)));
        }
    }
    #[test]
    fn recheck_duplicate_parent_candidate_is_not_resolved_by_last_repair() {
        let f = RecheckFixture::new();
        let shared = f.admitted_sink(4 * 1024 * 1024);
        f.parent_calls(&shared);
        f.observe_call(
            &shared,
            3,
            NativeQueryKind::Repair,
            2,
            &RecheckFixture::logical(&f.repaired[..2], NativeQueryKind::Repair, 7),
        )
        .unwrap();
        let mut observer = Observer(Arc::clone(&shared));
        assert!(observer.recheck_prepared(f.prepared()).is_err());
        observer
            .recheck_finished(f.finished(false, false, false))
            .unwrap();
        let s = shared.lock().unwrap();
        assert_eq!(s.calls.len(), 3);
        assert!(s.trace.failure.is_some());
        assert!(!s.trace.rows.iter().any(|r| r.artifact == RECHECK_ARTIFACT));
    }
    #[test]
    fn recheck_original_budget_rejects_before_pending_reply_and_preserves_unreserved_finish() {
        let f = RecheckFixture::new();
        let shared = f.admitted_sink(512 * 1024);
        f.parent_calls(&shared);
        let before = shared.lock().unwrap().trace.reserved_bytes;
        let mut observer = Observer(Arc::clone(&shared));
        assert!(observer.recheck_prepared(f.prepared()).is_err());
        observer
            .recheck_finished(f.finished(false, false, false))
            .unwrap();
        let s = shared.lock().unwrap();
        assert_eq!(s.trace.reserved_bytes, before);
        assert!(s.pending_recheck.is_none());
        assert_eq!(s.calls.len(), 2);
        assert!(s.trace.failure.is_some());
    }
    #[test]
    fn recheck_reply_context_epoch_sequence_and_prefix_drift_reject_before_input_publication() {
        let f = RecheckFixture::new();
        let shared = f.admitted_sink(4 * 1024 * 1024);
        f.parent_calls(&shared);
        Observer(Arc::clone(&shared))
            .recheck_prepared(f.prepared())
            .unwrap();
        let (anchor, _) = replay_line(&f.root, &f.repaired[..3]).unwrap();
        let s = shared.lock().unwrap();
        let rows = s.trace.rows.len();
        let bytes = s.trace.reserved_bytes;
        let mut changed = f.reply_context.clone();
        changed.search_generation += 1;
        for (request, logical, prefix) in [
            (
                RequestId::new(ProcessEpoch(18), 3),
                &f.reply_context,
                &f.repaired[..3],
            ),
            (id(2), &f.reply_context, &f.repaired[..3]),
            (id(3), &changed, &f.repaired[..3]),
            (id(3), &f.reply_context, &f.repaired[..2]),
        ] {
            assert!(
                s.check_pending_reply(
                    request,
                    NativeQueryKind::Reply,
                    &anchor,
                    prefix,
                    &f.repaired,
                    Some(&f.refutation),
                    Some(logical),
                    f.limits.deadline
                )
                .is_err()
            );
        }
        assert_eq!(s.trace.rows.len(), rows);
        assert_eq!(s.trace.reserved_bytes, bytes);
        assert_eq!(s.calls.len(), 2);
    }
    #[test]
    fn recheck_cancelled_prepaid_close_without_reply_keeps_unused_credit_and_duplicate_finish_fails()
     {
        let f = RecheckFixture::new();
        let shared = f.admitted_sink(4 * 1024 * 1024);
        f.parent_calls(&shared);
        let mut observer = Observer(Arc::clone(&shared));
        observer.recheck_prepared(f.prepared()).unwrap();
        let bytes = shared.lock().unwrap().trace.reserved_bytes;
        f.cancel.store(true, Ordering::Release);
        let mut close = f.finished(true, false, false);
        close.disposition = rz_search::pals::engine::RecheckDisposition::Interrupted;
        observer.recheck_finished(close).unwrap();
        {
            let s = shared.lock().unwrap();
            assert_eq!(s.trace.reserved_bytes, bytes);
            assert_eq!(s.prepaid_recheck_rows, 1);
            assert_eq!(s.calls.len(), 2);
            let row: serde_json::Value =
                serde_json::from_slice(&s.trace.rows.last().unwrap().json).unwrap();
            assert_eq!(row["data"]["cancelled_at_observer"], true);
            assert!(row["data"]["reply"].is_null());
        }
        assert!(
            observer
                .recheck_finished(f.finished(true, false, false))
                .is_err()
        );
    }
    #[test]
    fn recheck_invalid_provenance_remains_distinct_from_unobserved_without_positive_cpu_facts() {
        let f = RecheckFixture::new();
        let (actual, _) = replay_line(&f.root, &f.repaired).unwrap();
        let mut endpoint = f.endpoint();
        endpoint.evidence = RecheckEndpointEvidence::InvalidProvenance {
            error: rz_search::pals::store::StoreError::NotCompleted,
        };
        let raw = recheck_endpoint_json(&endpoint, &actual).unwrap();
        assert_eq!(raw["evidence"]["kind"], "InvalidProvenance");
        assert!(raw["evidence"].get("observation").is_none());
        assert!(recheck_endpoint_json(&endpoint, &f.root).is_err());
    }
    #[test]
    fn search_primary_physical_unknown_survives_summary_quota_and_success_requires_summary() {
        let shared = sink(&Position::startpos(), 0);
        let summary = shared
            .lock()
            .unwrap()
            .reserve(8192)
            .map_err(|e| invalid(e.to_string()));
        assert!(summary.is_err());
        let primary =
            rz_search::pals::engine::PalsError::Role(RoleError::PhysicalCompletionUnknown);
        let expected = primary.to_string();
        let error = search_after_summary::<()>(Err(primary), summary).unwrap_err();
        assert!(error.to_string().contains(&expected));
        assert!(!error.to_string().contains("output credit exhausted"));
        assert!(
            search_after_summary(Ok(()), Err(invalid("summary serialization failure"))).is_err()
        );
        search_after_summary(Ok(()), Ok(())).unwrap();
    }
    #[test]
    fn wrong_epoch_rejects_before_reservation_or_dispatch_evidence() {
        let position = Position::startpos();
        let legal = position.legal_moves();
        let cancel = AtomicBool::new(false);
        let query = RoleQuery {
            position: &position,
            legal: &legal,
            prefix: &[],
            proposal: &[],
            counterexample: None,
            records: &[],
            revision: 9,
            deadline: Instant::now() + Duration::from_secs(2),
            cancel: &cancel,
        };
        let prepared = prepare_role_input(&query, NativeQueryKind::Propose, [8; 32]).unwrap();
        let shared = sink(&position, 1024 * 1024);
        let mut observer = Observer(Arc::clone(&shared));
        assert!(
            observer
                .prepared(
                    id(1),
                    &prepared,
                    NativePreparedContext::Role {
                        query: &query,
                        kind: NativeQueryKind::Propose
                    }
                )
                .is_err()
        );
        let s = shared.lock().unwrap();
        assert_eq!(s.trace.reserved_bytes, 0);
        assert!(s.calls.is_empty());
        assert!(s.trace.rows.is_empty());
        assert!(s.trace.failure.is_some());
    }
    #[test]
    fn exact_tensor_seal_precedes_raw_delivery_and_exactly_once_consumption() {
        let position = Position::startpos();
        let legal = position.legal_moves();
        let cancel = AtomicBool::new(false);
        let query = RoleQuery {
            position: &position,
            legal: &legal,
            prefix: &[],
            proposal: &[],
            counterexample: None,
            records: &[],
            revision: 9,
            deadline: Instant::now() + Duration::from_secs(2),
            cancel: &cancel,
        };
        let prepared = prepare_role_input(&query, NativeQueryKind::Propose, [7; 32]).unwrap();
        let shared = sink(&position, 1024 * 1024);
        let mut observer = Observer(Arc::clone(&shared));
        observer
            .prepared(
                id(1),
                &prepared,
                NativePreparedContext::Role {
                    query: &query,
                    kind: NativeQueryKind::Propose,
                },
            )
            .unwrap();
        let frozen = shared.lock().unwrap().trace.inputs[0].0.clone();
        assert_eq!(frozen.snapshot().input_revision, 9);
        assert_eq!(frozen.snapshot().capture_sequence, 5);
        assert_eq!(frozen.snapshot().frozen_epoch, 1);
        let sidecar: PalsNativeInputSidecar = serde_json::from_slice(
            &shared
                .lock()
                .unwrap()
                .trace
                .rows
                .iter()
                .find(|r| r.artifact == "native-inputs.jsonl")
                .unwrap()
                .json,
        )
        .unwrap();
        assert_eq!(
            sidecar.tensor_json,
            serde_json::to_string(&prepared).unwrap()
        );
        let output = raw(legal.len());
        observer.physically_completed(id(1), Ok(&output)).unwrap();
        assert!(!shared.lock().unwrap().calls[&id(1)].accepted);
        observer.delivered(id(1)).unwrap();
        observer.accepted(id(1)).unwrap();
        assert!(observer.accepted(id(1)).is_err());
        let s = shared.lock().unwrap();
        let events: Vec<_> = s
            .trace
            .rows
            .iter()
            .filter(|r| r.artifact == "native-events.jsonl")
            .map(|r| {
                serde_json::from_slice::<serde_json::Value>(&r.json).unwrap()["stage"]
                    .as_str()
                    .unwrap()
                    .to_owned()
            })
            .collect();
        assert_eq!(
            events,
            vec![
                "prepared",
                "physically_completed",
                "delivered",
                "search_consumed"
            ]
        );
        assert_eq!(s.trace.inputs[0].0, frozen);
        assert!(
            s.trace.reserved_bytes >= s.trace.rows.iter().map(|r| r.json.len() as u64 + 1).sum()
        );
    }
    #[test]
    fn cancellation_after_delivery_never_consumes_or_rewrites_physical_evidence() {
        let position = Position::startpos();
        let legal = position.legal_moves();
        let cancel = AtomicBool::new(false);
        let query = RoleQuery {
            position: &position,
            legal: &legal,
            prefix: &[],
            proposal: &[],
            counterexample: None,
            records: &[],
            revision: 9,
            deadline: Instant::now() + Duration::from_secs(2),
            cancel: &cancel,
        };
        let prepared = prepare_role_input(&query, NativeQueryKind::Propose, [7; 32]).unwrap();
        let shared = sink(&position, 1024 * 1024);
        let mut observer = Observer(Arc::clone(&shared));
        observer
            .prepared(
                id(1),
                &prepared,
                NativePreparedContext::Role {
                    query: &query,
                    kind: NativeQueryKind::Propose,
                },
            )
            .unwrap();
        let mut output = raw(legal.len());
        output.candidate_logits[0] = f32::from_bits(0x7fc01234);
        observer.physically_completed(id(1), Ok(&output)).unwrap();
        observer.delivered(id(1)).unwrap();
        cancel.store(true, Ordering::Release);
        observer
            .rejected(id(1), NativeRoleRejection::Canceled)
            .unwrap();
        assert!(observer.accepted(id(1)).is_err());
        let s = shared.lock().unwrap();
        assert!(s.calls[&id(1)].physical);
        assert!(!s.calls[&id(1)].accepted);
        let raw: serde_json::Value = serde_json::from_slice(
            &s.trace
                .rows
                .iter()
                .find(|r| r.artifact == "native-raw-outputs.jsonl")
                .unwrap()
                .json,
        )
        .unwrap();
        assert_eq!(
            raw["raw"]["candidate_logits_bits"][0].as_u64(),
            Some(0x7fc01234)
        );
    }
    #[test]
    fn unknown_physical_completion_cannot_deliver_or_reuse_the_request() {
        let position = Position::startpos();
        let legal = position.legal_moves();
        let cancel = AtomicBool::new(false);
        let query = RoleQuery {
            position: &position,
            legal: &legal,
            prefix: &[],
            proposal: &[],
            counterexample: None,
            records: &[],
            revision: 9,
            deadline: Instant::now() + Duration::from_secs(2),
            cancel: &cancel,
        };
        let prepared = prepare_role_input(&query, NativeQueryKind::Propose, [7; 32]).unwrap();
        let shared = sink(&position, 1024 * 1024);
        let mut observer = Observer(Arc::clone(&shared));
        observer
            .prepared(
                id(1),
                &prepared,
                NativePreparedContext::Role {
                    query: &query,
                    kind: NativeQueryKind::Propose,
                },
            )
            .unwrap();
        observer
            .rejected(id(1), NativeRoleRejection::PhysicalCompletionUnknown)
            .unwrap();
        assert!(observer.delivered(id(1)).is_err());
        assert!(observer.accepted(id(1)).is_err());
        assert!(
            observer
                .prepared(
                    id(1),
                    &prepared,
                    NativePreparedContext::Role {
                        query: &query,
                        kind: NativeQueryKind::Propose
                    }
                )
                .is_err()
        );
        assert!(shared.lock().unwrap().calls[&id(1)].physical_unknown);
    }
    #[test]
    fn same_root_repair_and_divergence_never_receive_actual_game_wdl() {
        let position = Position::startpos();
        let legal = position.legal_moves();
        let cancel = AtomicBool::new(false);
        let response = [legal[0]];
        let query = RoleQuery {
            position: &position,
            legal: &legal,
            prefix: &[],
            proposal: &response,
            counterexample: Some(&response),
            records: &[],
            revision: 9,
            deadline: Instant::now() + Duration::from_secs(2),
            cancel: &cancel,
        };
        let prepared = prepare_role_input(&query, NativeQueryKind::Repair, [7; 32]).unwrap();
        let shared = sink(&position, 2 * 1024 * 1024);
        let producer = fixture_producer(&shared, 16 * 1024);
        let mut observer = Observer(Arc::clone(&shared));
        observer
            .prepared(
                id(1),
                &prepared,
                NativePreparedContext::Role {
                    query: &query,
                    kind: NativeQueryKind::Repair,
                },
            )
            .unwrap();
        assert!(!shared.lock().unwrap().trace.inputs[0].1);
        let divergence_proposal = [
            BoardMove::from_uci("e2e4").unwrap(),
            BoardMove::from_uci("e7e5").unwrap(),
        ];
        let dq = DivergenceQuery {
            root: &position,
            proposal: &divergence_proposal,
            candidates: &[1],
            records: &[],
            revision: 10,
            deadline: Instant::now() + Duration::from_secs(2),
            cancel: &cancel,
        };
        let divergence = prepare_divergence_input(&dq, [7; 32]).unwrap();
        observer
            .prepared(
                id(2),
                &divergence,
                NativePreparedContext::Divergence { query: &dq },
            )
            .unwrap();
        let s = shared.lock().unwrap();
        assert_eq!(s.trace.inputs.len(), 1);
        assert!(
            s.trace
                .rows
                .iter()
                .any(|r| r.artifact == "native-divergence-inputs.jsonl")
        );
        assert!(
            s.trace
                .rows
                .iter()
                .any(|r| r.artifact == "native-divergence-sidecars.jsonl")
        );
        let context_row = s
            .trace
            .rows
            .iter()
            .find(|r| r.artifact == divergence::CONTEXT_ARTIFACT)
            .unwrap();
        let context: divergence::NativeDivergenceContext =
            serde_json::from_slice(&context_row.json).unwrap();
        assert_eq!(context.native_request, [17, 2]);
        assert_eq!(context.captured_input_revision, 10);
        assert_eq!(context.divergence_sites.len(), 1);
        assert_eq!(context.divergence_sites[0].divergence_ply, 1);
        let lineage: Vec<_> = s
            .trace
            .rows
            .iter()
            .filter(|r| r.artifact == "input-lineage.jsonl")
            .map(|r| serde_json::from_slice::<serde_json::Value>(&r.json).unwrap())
            .collect();
        assert!(lineage[0].get("native_divergence_context_sha256").is_none());
        assert_eq!(
            lineage[1]["native_divergence_context_sha256"],
            context.sha256
        );
        let journal = producer.journal_snapshot();
        assert_eq!(journal.len(), 2);
        for (index, line) in journal.iter().enumerate() {
            let evidence: serde_json::Value = serde_json::from_slice(line).unwrap();
            assert_eq!(
                evidence["prepared"]["native_request"],
                serde_json::json!([17, index + 1])
            );
            assert_eq!(evidence["prepared"]["learning_input"], index == 0);
            assert_eq!(
                evidence["prepared"]["publication"],
                "seal-before-submit; prepaid-drain-after-search"
            );
            assert_eq!(
                evidence["sha256"],
                canonical_sha256(&(
                    "rz-pals-collector-prepared-producer/1",
                    &evidence["prepared"]
                ))
                .unwrap()
            );
        }
    }
    #[test]
    fn divergence_context_preserves_unsorted_slots_and_exact_journal_binding() {
        let position = Position::startpos();
        let proposal = ["e2e4", "e7e5", "g1f3", "b8c6"].map(|mv| BoardMove::from_uci(mv).unwrap());
        let cancel = AtomicBool::new(false);
        let query = DivergenceQuery {
            root: &position,
            proposal: &proposal,
            candidates: &[3, 1],
            records: &[],
            revision: 12,
            deadline: Instant::now() + Duration::from_secs(2),
            cancel: &cancel,
        };
        let prepared = prepare_divergence_input(&query, [7; 32]).unwrap();
        let shared = sink(&position, 2 * 1024 * 1024);
        let producer = fixture_producer(&shared, 16 * 1024);
        Observer(Arc::clone(&shared))
            .prepared(
                id(1),
                &prepared,
                NativePreparedContext::Divergence { query: &query },
            )
            .unwrap();
        let s = shared.lock().unwrap();
        let row = |artifact| {
            s.trace
                .rows
                .iter()
                .find(|r| r.artifact == artifact)
                .unwrap()
        };
        let context_row = row(divergence::CONTEXT_ARTIFACT);
        let context: divergence::NativeDivergenceContext =
            serde_json::from_slice(&context_row.json).unwrap();
        assert_eq!(context.version, divergence::CONTEXT_VERSION);
        assert_eq!(context.proposal_move16, pack(&proposal).unwrap());
        assert_eq!(
            context.challenged_line_sha256,
            canonical_sha256(&(
                "rz-pals-challenged-line/1",
                state_sha(&position).unwrap(),
                pack(&proposal).unwrap(),
            ))
            .unwrap()
        );
        for (slot, ply) in [3, 1].into_iter().enumerate() {
            let site = &context.divergence_sites[slot];
            let (prefix, _) = replay_line(&position, &proposal[..ply]).unwrap();
            assert_eq!(site.slot, u32::try_from(slot).unwrap());
            assert_eq!(site.divergence_ply, u32::try_from(ply).unwrap());
            assert_eq!(site.prefix_rules_state_sha256, state_sha(&prefix).unwrap());
            assert_eq!(
                site.prefix_rules_history_sha256,
                hex(rz_uci::pals_native::pals_history_digest(&prefix).unwrap())
            );
        }
        let sidecar: PalsNativeInputSidecar =
            serde_json::from_slice(&row("native-divergence-sidecars.jsonl").json).unwrap();
        assert_eq!(context.tensor_sidecar_sha256, sidecar.sha256);
        assert_eq!(context.input_sha256, sidecar.input_sha256);
        let mut body = serde_json::to_value(&context).unwrap();
        let _ = body.as_object_mut().unwrap().remove("sha256");
        assert_eq!(
            context.sha256,
            canonical_sha256(&(divergence::CONTEXT_VERSION, body)).unwrap()
        );
        let lineage_row = row("input-lineage.jsonl");
        let lineage: serde_json::Value = serde_json::from_slice(&lineage_row.json).unwrap();
        assert_eq!(lineage["native_divergence_context_sha256"], context.sha256);
        assert_eq!(lineage["training_admission"], "deferred_divergence_head");
        let journal = producer.journal_snapshot();
        let journal: serde_json::Value = serde_json::from_slice(&journal[0]).unwrap();
        assert_eq!(journal["prepared"]["learning_input"], false);
        assert_eq!(
            journal["prepared"]["lineage_json"]["sha256"],
            format!("{:x}", Sha256::digest(&lineage_row.json))
        );
        assert_eq!(
            journal["prepared"]["lineage_json"]["bytes"],
            lineage_row.json.len() as u64
        );
        assert!(!s.calls[&id(1)].physical);
        assert!(s.trace.inputs.is_empty());
    }
    #[test]
    fn divergence_descriptor_rejects_invalid_sites_and_feature_order_before_submit() {
        let position = Position::startpos();
        let proposal = ["e2e4", "e7e5", "g1f3", "b8c6"].map(|mv| BoardMove::from_uci(mv).unwrap());
        let cancel = AtomicBool::new(false);
        let deadline = Instant::now() + Duration::from_secs(2);
        let query = DivergenceQuery {
            root: &position,
            proposal: &proposal,
            candidates: &[3, 1],
            records: &[],
            revision: 12,
            deadline,
            cancel: &cancel,
        };
        let prepared = prepare_divergence_input(&query, [7; 32]).unwrap();
        for candidates in [&[1, 1][..], &[0][..], &[4][..]] {
            let invalid = DivergenceQuery {
                root: &position,
                proposal: &proposal,
                candidates,
                records: &[],
                revision: 12,
                deadline,
                cancel: &cancel,
            };
            let shared = sink(&position, 2 * 1024 * 1024);
            assert!(
                Observer(Arc::clone(&shared))
                    .prepared(
                        id(1),
                        &prepared,
                        NativePreparedContext::Divergence { query: &invalid },
                    )
                    .is_err()
            );
            let s = shared.lock().unwrap();
            assert!(s.trace.rows.is_empty());
            assert!(s.calls.is_empty());
        }
        let mut reordered = prepared;
        reordered.divergence_features.swap(0, 1);
        let shared = sink(&position, 2 * 1024 * 1024);
        assert!(
            Observer(Arc::clone(&shared))
                .prepared(
                    id(1),
                    &reordered,
                    NativePreparedContext::Divergence { query: &query },
                )
                .is_err()
        );
        assert!(shared.lock().unwrap().calls.is_empty());
    }
    #[test]
    fn divergence_descriptor_bytes_are_reserved_before_trace_commit() {
        let position = Position::startpos();
        let proposal = ["e2e4", "e7e5"].map(|mv| BoardMove::from_uci(mv).unwrap());
        let cancel = AtomicBool::new(false);
        let query = DivergenceQuery {
            root: &position,
            proposal: &proposal,
            candidates: &[1],
            records: &[],
            revision: 12,
            deadline: Instant::now() + Duration::from_secs(2),
            cancel: &cancel,
        };
        let prepared = prepare_divergence_input(&query, [7; 32]).unwrap();
        let full = sink(&position, 2 * 1024 * 1024);
        Observer(Arc::clone(&full))
            .prepared(
                id(1),
                &prepared,
                NativePreparedContext::Divergence { query: &query },
            )
            .unwrap();
        let s = full.lock().unwrap();
        let bytes: u64 = s
            .trace
            .rows
            .iter()
            .filter(|r| r.artifact != "native-events.jsonl")
            .map(|r| r.json.len() as u64 + 1)
            .sum();
        assert_eq!(s.trace.reserved_bytes, bytes + RAW_RESERVE + STAGE_RESERVE);
        let credit = s.trace.reserved_bytes - 1;
        let stored = s
            .trace
            .rows
            .iter()
            .find(|r| r.artifact == divergence::CONTEXT_ARTIFACT)
            .unwrap()
            .json
            .clone();
        let temp = NativeTestOutput::new();
        let mut output = Output::new(&temp.0, "descriptor-storage", 2 * 1024 * 1024).unwrap();
        output.native_trace(&s.trace).unwrap();
        assert_eq!(output.bytes, s.trace.reserved_bytes);
        let path = output.directory.join(divergence::CONTEXT_ARTIFACT);
        drop(output);
        let written = std::fs::read(path).unwrap();
        assert_eq!(written.len(), stored.len() + 1);
        assert_eq!(&written[..stored.len()], stored.as_slice());
        assert_eq!(written.last(), Some(&b'\n'));
        drop(s);
        let limited = sink(&position, credit);
        let producer = fixture_producer(&limited, 8 * 1024);
        assert!(
            Observer(Arc::clone(&limited))
                .prepared(
                    id(1),
                    &prepared,
                    NativePreparedContext::Divergence { query: &query },
                )
                .is_err()
        );
        let s = limited.lock().unwrap();
        assert!(s.trace.rows.is_empty());
        assert!(s.calls.is_empty());
        assert_eq!(s.total_rows, 0);
        assert_eq!(s.trace.sequence, 4);
        assert_eq!(s.trace.reserved_bytes, 0);
        assert!(producer.journal_snapshot().is_empty());
        drop(s);
        // Native credit rejection did not poison or append the producer. The
        // same undispatched request remains admissible with explicit credit.
        limited.lock().unwrap().context.as_mut().unwrap().max_bytes = 1024 * 1024;
        Observer(Arc::clone(&limited))
            .prepared(
                id(1),
                &prepared,
                NativePreparedContext::Divergence { query: &query },
            )
            .unwrap();
        assert_eq!(producer.journal_snapshot().len(), 1);
    }
    #[test]
    fn producer_quota_rejects_submit_but_preserves_exact_prepared_rows_and_journal() {
        let position = Position::startpos();
        let legal = position.legal_moves();
        let cancel = AtomicBool::new(false);
        let query = RoleQuery {
            position: &position,
            legal: &legal,
            prefix: &[],
            proposal: &[],
            counterexample: None,
            records: &[],
            revision: 9,
            deadline: Instant::now() + Duration::from_secs(2),
            cancel: &cancel,
        };
        let prepared = prepare_role_input(&query, NativeQueryKind::Propose, [7; 32]).unwrap();
        let shared = sink(&position, 16 * 1024 * 1024);
        let producer = fixture_producer(&shared, 8 * 1024);
        let mut observer = Observer(Arc::clone(&shared));
        let mut rejected = None;
        for request in 1..=32 {
            if observer
                .prepared(
                    id(request),
                    &prepared,
                    NativePreparedContext::Role {
                        query: &query,
                        kind: NativeQueryKind::Propose,
                    },
                )
                .is_err()
            {
                rejected = Some(id(request));
                break;
            }
        }
        let rejected = rejected.expect("finite producer journal must reject before a submit");
        let journal = producer.journal_snapshot();
        let last: serde_json::Value = serde_json::from_slice(journal.last().unwrap()).unwrap();
        assert_eq!(last["stage"], "prepared-rejected-before-submit");
        assert_eq!(
            last["prepared"]["native_request"],
            serde_json::json!([rejected.epoch.0, rejected.sequence])
        );
        let sink = shared.lock().unwrap();
        let call = &sink.calls[&rejected];
        assert!(!call.physical && !call.delivered && !call.accepted);
        let input = sink
            .trace
            .rows
            .iter()
            .rev()
            .find(|row| row.artifact == "inputs.jsonl")
            .unwrap();
        assert_eq!(
            last["prepared"]["input_json"]["sha256"],
            format!("{:x}", Sha256::digest(&input.json))
        );
        assert_eq!(
            last["prepared"]["input_json"]["bytes"],
            input.json.len() as u64
        );
        assert!(sink.trace.failure.is_some());
    }
    #[test]
    fn delegated_credit_reserves_raw_and_close_events_before_any_dispatch() {
        let position = Position::startpos();
        let legal = position.legal_moves();
        let cancel = AtomicBool::new(false);
        let query = RoleQuery {
            position: &position,
            legal: &legal,
            prefix: &[],
            proposal: &[],
            counterexample: None,
            records: &[],
            revision: 9,
            deadline: Instant::now() + Duration::from_secs(2),
            cancel: &cancel,
        };
        let prepared = prepare_role_input(&query, NativeQueryKind::Propose, [7; 32]).unwrap();
        let shared = sink(&position, RAW_RESERVE + STAGE_RESERVE - 1);
        let mut observer = Observer(Arc::clone(&shared));
        assert!(
            observer
                .prepared(
                    id(1),
                    &prepared,
                    NativePreparedContext::Role {
                        query: &query,
                        kind: NativeQueryKind::Propose
                    }
                )
                .is_err()
        );
        let s = shared.lock().unwrap();
        assert!(s.calls.is_empty());
        assert_eq!(s.trace.reserved_bytes, 0);
        assert!(s.trace.rows.is_empty());
    }
}
