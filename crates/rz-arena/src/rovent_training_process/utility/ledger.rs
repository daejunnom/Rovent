//! Utility-only protocol bookkeeping. Original datasets/checkpoints are owned
//! by the parent runtime; JSON observations never create those authorities.
//! A completed child step is prepared first and committed only after runtime
//! returns the exact durable receipt pin. No filesystem or tensor work is here.

use super::cost::{RoventUtilityCheckpointCost, RoventUtilityRecordedCost};
use super::execution::ROVENT_UTILITY_MAX_UPDATES;
use super::numerical_parts::ValidatedUtilityNumericalParts;
use super::runtime_family::{
    RuntimeUtilityContext as UtilityContextAuthority,
    RuntimeUtilityDataset as VerifiedUtilityDataset,
    RuntimeUtilityWindow as RoventUtilityTrainWindow, UtilityRuntimeFamily,
};
use crate::ArenaError;
use crate::rovent_training_process::RoventTrainingFilePin;
use rz_experiments::pals_data_v3::{generation_digest, generation_sha};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest as _, Sha256};
use std::collections::BTreeSet;

const STATE_DOMAIN: &str = "rz-rovent-utility-derived-training-state/1";
const SAMPLER_DOMAIN: &str = "rz-rovent-utility-sampler/1";
const COMPLETION_DOMAIN: &str = "rz-rovent-utility-learner-completion/1";
const ACCUMULATION: usize = 8;
const MAX_COMPLETION_RECEIPT_BYTES: usize = 16 * 1024 * 1024;
const STATE_FIELDS: [&str; 5] = [
    "model_state_sha256",
    "optimizer_state_sha256",
    "scheduler_state_sha256",
    "rng_state_sha256",
    "sampler_state_sha256",
];
const COST_FIELDS: [&str; 8] = [
    "forward_calls",
    "backward_calls",
    "samples",
    "wall_time_ns",
    "partial_accumulation_ns",
    "replay_ns",
    "masked_groups",
    "output_bytes",
];

fn invalid(reason: impl Into<String>) -> ArenaError {
    ArenaError::Integrity(format!("Rovent utility ledger: {}", reason.into()))
}
fn check(condition: bool, reason: &str) -> Result<(), ArenaError> {
    if condition {
        Ok(())
    } else {
        Err(invalid(reason))
    }
}
fn fields(value: &Value, expected: &[&str]) -> Result<(), ArenaError> {
    let object = value
        .as_object()
        .ok_or_else(|| invalid("exact object required"))?;
    check(
        object.len() == expected.len() && expected.iter().all(|key| object.contains_key(*key)),
        "missing or unknown protocol field",
    )
}
fn count(value: &Value) -> Result<u64, ArenaError> {
    value
        .as_u64()
        .ok_or_else(|| invalid("exact unsigned integer required"))
}
fn sha(value: &Value) -> Result<&str, ArenaError> {
    value
        .as_str()
        .filter(|value| generation_sha(value))
        .ok_or_else(|| invalid("exact lowercase SHA256 required"))
}
fn digest(domain: &str, value: &impl Serialize) -> Result<String, ArenaError> {
    generation_digest(domain, value).map_err(ArenaError::Manifest)
}
fn add(left: u64, right: u64) -> Result<u64, ArenaError> {
    left.checked_add(right)
        .ok_or_else(|| invalid("counter overflow"))
}
fn cost(value: &Value) -> Result<RoventUtilityRecordedCost, ArenaError> {
    fields(value, &COST_FIELDS)?;
    serde_json::from_value(value.clone()).map_err(|error| invalid(error.to_string()))
}
fn cost_monotone(
    before: RoventUtilityRecordedCost,
    after: RoventUtilityRecordedCost,
) -> Result<(), ArenaError> {
    let before = serde_json::to_value(before).map_err(|error| invalid(error.to_string()))?;
    let after = serde_json::to_value(after).map_err(|error| invalid(error.to_string()))?;
    check(
        COST_FIELDS
            .iter()
            .all(|key| after[*key].as_u64() >= before[*key].as_u64()),
        "recorded cost decreased; unknown expense cannot be refunded",
    )
}
fn pin(value: &Value) -> Result<RoventTrainingFilePin, ArenaError> {
    fields(value, &["path", "sha256", "bytes"])?;
    let pin: RoventTrainingFilePin =
        serde_json::from_value(value.clone()).map_err(|error| invalid(error.to_string()))?;
    check(
        pin.path.is_absolute() && generation_sha(&pin.sha256) && pin.bytes > 0,
        "original absolute nonempty file pin required",
    )?;
    Ok(pin)
}
fn source_declaration(value: &Value) -> Result<(), ArenaError> {
    match value["kind"].as_str() {
        Some("untrained") => fields(value, &["kind"]),
        Some("learned") => {
            fields(value, &["kind", "run_id", "steps", "dataset_sha256"])?;
            check(
                value["run_id"].as_str().is_some_and(|run| {
                    !run.trim().is_empty() && run.len() <= 128 && !run.chars().any(char::is_control)
                }) && count(&value["steps"])? > 0
                    && generation_sha(sha(&value["dataset_sha256"])?),
                "bounded original numerical source declaration required",
            )
        }
        _ => Err(invalid("unsupported original numerical source declaration")),
    }
}
fn accumulation_denominator(contexts: &[ConsumedContext]) -> Result<u64, ArenaError> {
    check(
        contexts.len() == ACCUMULATION,
        "exactly eight original context occurrences required",
    )?;
    contexts
        .iter()
        .try_fold(0u64, |total, row| add(total, row.unique_known_pairs))
}

fn zero_update_closure(
    family: UtilityRuntimeFamily,
    payload: &Value,
    completed_delta: u64,
    dispatched_delta: u64,
) -> Result<bool, ArenaError> {
    if payload["status"] != "no_update_checkpoint_not_published" {
        return Ok(false);
    }
    check(
        family == UtilityRuntimeFamily::InterventionV2
            && completed_delta == 0
            && dispatched_delta == 0
            && payload["checkpoint"].is_null()
            && payload["quarantined"] == false
            && payload["physical_completion"] == "confirmed"
            && payload["pending_microbatches"] == 0,
        "zero-update closure cannot issue a checkpoint or hide dispatched/physical work",
    )?;
    Ok(true)
}

/// Hashes are observations of five distinct original state families. They are
/// never a deserializable permission to restore or execute a tensor owner.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct UtilityStateSeals {
    pub(super) model_state_sha256: String,
    pub(super) optimizer_state_sha256: String,
    pub(super) scheduler_state_sha256: String,
    pub(super) rng_state_sha256: String,
    pub(super) sampler_state_sha256: String,
}
impl UtilityStateSeals {
    pub(super) fn from_payload(value: &Value) -> Result<Self, ArenaError> {
        Ok(Self {
            model_state_sha256: sha(&value[STATE_FIELDS[0]])?.into(),
            optimizer_state_sha256: sha(&value[STATE_FIELDS[1]])?.into(),
            scheduler_state_sha256: sha(&value[STATE_FIELDS[2]])?.into(),
            rng_state_sha256: sha(&value[STATE_FIELDS[3]])?.into(),
            sampler_state_sha256: sha(&value[STATE_FIELDS[4]])?.into(),
        })
    }
    fn validate(&self) -> Result<(), ArenaError> {
        check(
            [
                &self.model_state_sha256,
                &self.optimizer_state_sha256,
                &self.scheduler_state_sha256,
                &self.rng_state_sha256,
                &self.sampler_state_sha256,
            ]
            .into_iter()
            .all(|value| generation_sha(value)),
            "five original state seals required",
        )
    }
    fn same_parameters_and_moments(&self, other: &Self) -> bool {
        self.model_state_sha256 == other.model_state_sha256
            && self.optimizer_state_sha256 == other.optimizer_state_sha256
            && self.scheduler_state_sha256 == other.scheduler_state_sha256
    }
}

/// Constructed only by the sibling runtime after complete original
/// checkpoint/receipt-chain validation. Deliberately no
/// Deserialize implementation and no `verified`/`authorized` boolean.
pub(super) struct RestoredUtilityState {
    pub(super) states: UtilityStateSeals,
    pub(super) sampler: Value,
    pub(super) completed_updates: u64,
    pub(super) dispatched_updates: u64,
    pub(super) ancestor_updates: u64,
    pub(super) cost: RoventUtilityRecordedCost,
    pub(super) ancestor_cost: RoventUtilityRecordedCost,
    pub(super) completion_receipts: Vec<RoventTrainingFilePin>,
    pub(super) ancestor_completion_sha256: Vec<String>,
    pub(super) initial_source_declaration: Value,
    pub(super) frozen_base_weights_sha256: String,
    pub(super) frozen_base_mapping_sha256: String,
    pub(super) source_checkpoint: Option<RoventTrainingFilePin>,
    pub(super) source_payload_state_sha256: Option<String>,
    pub(super) source_snapshot_cost_observation: Option<Value>,
    pub(super) source_snapshot_cost: Option<RoventUtilityRecordedCost>,
    pub(super) source_snapshot_ancestor_cost: Option<RoventUtilityRecordedCost>,
    pub(super) checkpoint_cost: Option<RoventUtilityCheckpointCost>,
}

/// Observed only from the actual initial child after original peer, runtime,
/// resource and raw numerical asset admission. This records the child's five
/// initial state seals; Rust does not independently reconstruct Torch states.
/// It has no update counts or learned receipt chain to promote.
pub(super) struct InitialUtilityMeasurement {
    pub(super) states: UtilityStateSeals,
    pub(super) sampler: Value,
    pub(super) initial_source_declaration: Value,
    pub(super) frozen_base_weights_sha256: String,
    pub(super) frozen_base_mapping_sha256: String,
}

#[derive(Clone, Copy)]
enum UtilityStateAdmission {
    InitialPeerMeasurement,
    OriginalCheckpoint,
    ForkPeerMeasurementFromOriginalCheckpoint,
}
impl UtilityStateAdmission {
    fn scope(self) -> &'static str {
        match self {
            Self::InitialPeerMeasurement => {
                "actual_original_child_initial_observation_not_independent_torch_recomputation"
            }
            Self::OriginalCheckpoint => "original_held_checkpoint_and_full_completion_chain",
            Self::ForkPeerMeasurementFromOriginalCheckpoint => {
                "original_checkpoint_model_and_chain_with_actual_child_reset_state_observation"
            }
        }
    }
}
#[derive(Clone, Copy, Eq, PartialEq)]
enum UtilityLedgerMode {
    OriginalRuntime,
    HistoricalReplay,
}

#[derive(Clone)]
struct UtilitySampler {
    run_id: String,
    pool_sha256: String,
    seed: u32,
    epoch: u64,
    permutation: Vec<usize>,
    cursor: usize,
}
impl UtilitySampler {
    fn from_original(
        value: &Value,
        window: &RoventUtilityTrainWindow,
        dataset: &VerifiedUtilityDataset,
    ) -> Result<Self, ArenaError> {
        fields(
            value,
            &[
                "schema",
                "run_id",
                "pool_sha256",
                "context_ids",
                "seed",
                "epoch",
                "permutation",
                "cursor",
            ],
        )?;
        let ids = dataset.context_ids();
        check(
            !ids.is_empty()
                && ids.len() <= 8192
                && ids.iter().all(|value| generation_sha(value))
                && ids.iter().collect::<BTreeSet<_>>().len() == ids.len(),
            "bounded original context inventory required",
        )?;
        check(
            value["schema"] == SAMPLER_DOMAIN
                && value["run_id"] == window.run_id
                && value["pool_sha256"] == dataset.pool_sha256()
                && value["context_ids"] == json!(ids)
                && count(&value["seed"])? == u64::from(window.training_seed),
            "sampler run/pool/context order/seed differs",
        )?;
        let epoch = count(&value["epoch"])?;
        let cursor = usize::try_from(count(&value["cursor"])?)
            .map_err(|_| invalid("sampler cursor overflow"))?;
        let permutation = Self::permutation(ids, window.training_seed, epoch)?;
        check(
            cursor <= ids.len() && value["permutation"] == json!(permutation),
            "sampler permutation/cursor differs from exact framing",
        )?;
        Ok(Self {
            run_id: window.run_id.clone(),
            pool_sha256: dataset.pool_sha256().into(),
            seed: window.training_seed,
            epoch,
            permutation,
            cursor,
        })
    }
    fn permutation(ids: &[String], seed: u32, epoch: u64) -> Result<Vec<usize>, ArenaError> {
        let mut values = Vec::with_capacity(ids.len());
        for (index, id) in ids.iter().enumerate() {
            let mut hash = Sha256::new();
            hash.update(SAMPLER_DOMAIN.as_bytes());
            hash.update([0]);
            hash.update(u64::from(seed).to_le_bytes());
            hash.update(epoch.to_le_bytes());
            hash.update(
                u32::try_from(index)
                    .map_err(|_| invalid("sampler index overflow"))?
                    .to_le_bytes(),
            );
            let mut raw = [0u8; 32];
            for (offset, byte) in raw.iter_mut().enumerate() {
                *byte = u8::from_str_radix(&id[offset * 2..offset * 2 + 2], 16)
                    .map_err(|_| invalid("sampler context SHA"))?;
            }
            hash.update(raw);
            values.push((<[u8; 32]>::from(hash.finalize()), index));
        }
        values.sort_unstable();
        Ok(values.into_iter().map(|(_, index)| index).collect())
    }
    fn next(&mut self, ids: &[String]) -> Result<usize, ArenaError> {
        if self.cursor == ids.len() {
            self.epoch = add(self.epoch, 1)?;
            self.permutation = Self::permutation(ids, self.seed, self.epoch)?;
            self.cursor = 0;
        }
        let index = *self
            .permutation
            .get(self.cursor)
            .ok_or_else(|| invalid("original sampler exhausted"))?;
        self.cursor += 1;
        Ok(index)
    }
    fn projection(&self, ids: &[String]) -> Value {
        json!({"schema":SAMPLER_DOMAIN,"run_id":self.run_id,"pool_sha256":self.pool_sha256,"context_ids":ids,"seed":self.seed,"epoch":self.epoch,"permutation":self.permutation,"cursor":self.cursor})
    }
    fn state_sha256(&self, ids: &[String]) -> Result<String, ArenaError> {
        typed_plain_state_sha256(&self.projection(ids))
    }
}

/// Exact `_state_digest` framing for the plain ASCII utility sampler only.
/// Tensor/RNG/optimizer digesting remains with the verified numerical owner.
fn typed_plain_state_sha256(value: &Value) -> Result<String, ArenaError> {
    fn visit(value: &Value, hash: &mut Sha256, depth: usize) -> Result<(), ArenaError> {
        check(depth <= 16, "bounded sampler state nesting required")?;
        match value {
            Value::Object(object) => {
                hash.update(b"dict");
                let mut keys = object.keys().collect::<Vec<_>>();
                keys.sort_unstable();
                for key in keys {
                    visit(&json!(key), hash, depth + 1)?;
                    visit(&object[key], hash, depth + 1)?;
                }
            }
            Value::Array(values) => {
                hash.update(format!("list{}", values.len()).as_bytes());
                for value in values {
                    visit(value, hash, depth + 1)?;
                }
            }
            Value::String(text) => {
                check(
                    text.is_ascii(),
                    "sampler strings must retain exact ASCII framing",
                )?;
                hash.update(
                    serde_json::to_vec(&json!(["str", text]))
                        .map_err(|error| invalid(error.to_string()))?,
                );
            }
            Value::Number(_) => {
                count(value)?;
                hash.update(
                    serde_json::to_vec(&json!(["int", value]))
                        .map_err(|error| invalid(error.to_string()))?,
                );
            }
            _ => return Err(invalid("unexpected non-sampler scalar type")),
        }
        hash.update(b";");
        Ok(())
    }
    let mut hash = Sha256::new();
    visit(value, &mut hash, 0)?;
    Ok(format!("{:x}", hash.finalize()))
}

#[derive(Clone)]
struct ConsumedContext {
    sha256: String,
    source_projection: Value,
    pair_sha256: Vec<String>,
    unique_known_pairs: u64,
}
impl ConsumedContext {
    fn from_original(context: UtilityContextAuthority<'_>) -> Result<Self, ArenaError> {
        let pairs = context.pair_sha256s().to_vec();
        check(
            generation_sha(context.sha256())
                && pairs.len() <= 64
                && pairs.iter().all(|value| generation_sha(value))
                && pairs.iter().collect::<BTreeSet<_>>().len() == pairs.len()
                && context.unique_known_pairs() == pairs.len() as u64,
            "original unique known-pair authority differs",
        )?;
        Ok(Self {
            sha256: context.sha256().into(),
            source_projection: context.source_projection().clone(),
            pair_sha256: pairs,
            unique_known_pairs: context.unique_known_pairs(),
        })
    }
}
struct PendingUtilityUpdate {
    body: Value,
    bytes: Vec<u8>,
    sha256: String,
    states: UtilityStateSeals,
    cost: RoventUtilityRecordedCost,
    completed_updates: u64,
    dispatched_updates: u64,
}

/// Single original process/window ledger. A prepared update is retained until
/// its matching durable receipt arrives; it is never retried or counted twice.
pub(super) struct UtilityLedger {
    dataset: VerifiedUtilityDataset,
    window: RoventUtilityTrainWindow,
    _preparation: Value,
    numerical: ValidatedUtilityNumericalParts,
    restored: RestoredUtilityState,
    sampler: UtilitySampler,
    states: UtilityStateSeals,
    cost: RoventUtilityRecordedCost,
    completed_updates: u64,
    dispatched_updates: u64,
    initial_dispatched_updates: u64,
    initial_completed_updates: u64,
    initial_cost: RoventUtilityRecordedCost,
    completion_receipts: Vec<RoventTrainingFilePin>,
    consumed: u64,
    groups: u64,
    masked_groups: u64,
    pending_contexts: Vec<ConsumedContext>,
    pending_update: Option<PendingUtilityUpdate>,
    begun: bool,
    state_admission: UtilityStateAdmission,
    begin_payload: Option<Value>,
    mode: UtilityLedgerMode,
}

/// A borrow of the actual current ledger boundary. Only the ledger constructs
/// this type; a JSON checkpoint projection cannot substitute it. The runtime
/// still owns/verifies every original file descriptor and publication write.
pub(super) struct UtilityCheckpointBoundary<'a> {
    ledger: &'a UtilityLedger,
    original_tensor_pin: RoventTrainingFilePin,
    original_tensor_projection_pin: RoventTrainingFilePin,
    snapshot_cost: RoventUtilityRecordedCost,
    post_snapshot_cost: RoventUtilityRecordedCost,
    // checkpoint_projection already checked all five states, the sampler,
    // original file pins and costs. Keep that exact sealed projection instead
    // of maintaining a second state copy that publication does not consume.
    projection: Value,
}
impl UtilityCheckpointBoundary<'_> {
    pub(super) fn original_tensor_pin(&self) -> &RoventTrainingFilePin {
        &self.original_tensor_pin
    }
    pub(super) fn original_tensor_projection_pin(&self) -> &RoventTrainingFilePin {
        &self.original_tensor_projection_pin
    }
    pub(super) fn completion_receipts(&self) -> &[RoventTrainingFilePin] {
        &self.ledger.completion_receipts
    }
    pub(super) fn ancestor_completion_sha256(&self) -> &[String] {
        &self.ledger.restored.ancestor_completion_sha256
    }
    pub(super) fn snapshot_cost(&self) -> RoventUtilityRecordedCost {
        self.snapshot_cost
    }
    pub(super) fn post_snapshot_cost(&self) -> RoventUtilityRecordedCost {
        self.post_snapshot_cost
    }
    pub(super) fn ancestor_cost(&self) -> RoventUtilityRecordedCost {
        self.ledger.restored.ancestor_cost
    }
    pub(super) fn initial_ledger_projection(&self) -> Value {
        self.ledger.initial_ledger_projection()
    }
    pub(super) fn projection(&self) -> &Value {
        &self.projection
    }
}
impl UtilityLedger {
    pub(super) fn initial_sampler_projection(
        window: &RoventUtilityTrainWindow,
        dataset: &VerifiedUtilityDataset,
    ) -> Result<Value, ArenaError> {
        window.validate()?;
        check(
            window.family() == dataset.family(),
            "sampler source family mismatch",
        )?;
        check(
            dataset.context_ids().iter().all(|sha| generation_sha(sha))
                && !dataset.context_ids().is_empty()
                && dataset.context_ids().len() <= 8192
                && generation_sha(dataset.pool_sha256()),
            "original initial sampler inventory/pool required",
        )?;
        Ok(UtilitySampler {
            run_id: window.run_id.clone(),
            pool_sha256: dataset.pool_sha256().into(),
            seed: window.training_seed,
            epoch: 0,
            permutation: UtilitySampler::permutation(
                dataset.context_ids(),
                window.training_seed,
                0,
            )?,
            cursor: 0,
        }
        .projection(dataset.context_ids()))
    }
    pub(super) fn from_initial_measurement(
        window: RoventUtilityTrainWindow,
        preparation: Value,
        dataset: VerifiedUtilityDataset,
        measurement: InitialUtilityMeasurement,
    ) -> Result<Self, ArenaError> {
        check(
            measurement.sampler == Self::initial_sampler_projection(&window, &dataset)?,
            "initial child changed the original fresh sampler metadata",
        )?;
        let restored = RestoredUtilityState {
            states: measurement.states,
            sampler: measurement.sampler,
            completed_updates: 0,
            dispatched_updates: 0,
            ancestor_updates: 0,
            cost: RoventUtilityRecordedCost::default(),
            ancestor_cost: RoventUtilityRecordedCost::default(),
            completion_receipts: Vec::new(),
            ancestor_completion_sha256: Vec::new(),
            initial_source_declaration: measurement.initial_source_declaration,
            frozen_base_weights_sha256: measurement.frozen_base_weights_sha256,
            frozen_base_mapping_sha256: measurement.frozen_base_mapping_sha256,
            source_checkpoint: None,
            source_payload_state_sha256: None,
            source_snapshot_cost_observation: None,
            source_snapshot_cost: None,
            source_snapshot_ancestor_cost: None,
            checkpoint_cost: None,
        };
        Self::new_inner(
            window,
            preparation,
            dataset,
            restored,
            UtilityStateAdmission::InitialPeerMeasurement,
            UtilityLedgerMode::OriginalRuntime,
        )
    }
    pub(super) fn from_fork_measurement(
        window: RoventUtilityTrainWindow,
        preparation: Value,
        dataset: VerifiedUtilityDataset,
        source: &RestoredUtilityState,
        measurement: InitialUtilityMeasurement,
    ) -> Result<Self, ArenaError> {
        check(
            measurement.sampler == Self::initial_sampler_projection(&window, &dataset)?,
            "fork must reset sampler to its exact new run/seed inventory",
        )?;
        check(
            source.source_checkpoint.is_some()
                && source
                    .source_payload_state_sha256
                    .as_ref()
                    .is_some_and(|value| generation_sha(value))
                && source.checkpoint_cost.is_some(),
            "fork requires the original verified checkpoint source",
        )?;
        source.states.validate()?;
        check(
            measurement.states.model_state_sha256 == source.states.model_state_sha256
                && measurement.initial_source_declaration == source.initial_source_declaration
                && measurement.frozen_base_weights_sha256 == source.frozen_base_weights_sha256
                && measurement.frozen_base_mapping_sha256 == source.frozen_base_mapping_sha256,
            "fork changed original utility model/source/frozen identities",
        )?;
        check(
            source.completed_updates == source.completion_receipts.len() as u64
                && source.ancestor_updates == source.ancestor_completion_sha256.len() as u64,
            "fork source lacks complete original receipt chain",
        )?;
        let ancestor_updates = add(source.completed_updates, source.ancestor_updates)?;
        let mut ancestor_completion_sha256 = source.ancestor_completion_sha256.clone();
        ancestor_completion_sha256.extend(
            source
                .completion_receipts
                .iter()
                .map(|pin| pin.sha256.clone()),
        );
        let restored = RestoredUtilityState {
            states: measurement.states,
            sampler: measurement.sampler,
            completed_updates: 0,
            dispatched_updates: 0,
            ancestor_updates,
            cost: RoventUtilityRecordedCost::default(),
            ancestor_cost: source.ancestor_cost.checked_add(&source.cost)?,
            completion_receipts: Vec::new(),
            ancestor_completion_sha256,
            initial_source_declaration: source.initial_source_declaration.clone(),
            frozen_base_weights_sha256: source.frozen_base_weights_sha256.clone(),
            frozen_base_mapping_sha256: source.frozen_base_mapping_sha256.clone(),
            source_checkpoint: source.source_checkpoint.clone(),
            source_payload_state_sha256: source.source_payload_state_sha256.clone(),
            source_snapshot_cost_observation: source.source_snapshot_cost_observation.clone(),
            source_snapshot_cost: source.source_snapshot_cost,
            source_snapshot_ancestor_cost: source.source_snapshot_ancestor_cost,
            checkpoint_cost: source.checkpoint_cost.clone(),
        };
        Self::new_inner(
            window,
            preparation,
            dataset,
            restored,
            UtilityStateAdmission::ForkPeerMeasurementFromOriginalCheckpoint,
            UtilityLedgerMode::OriginalRuntime,
        )
    }
    pub(super) fn new(
        window: RoventUtilityTrainWindow,
        preparation: Value,
        dataset: VerifiedUtilityDataset,
        restored: RestoredUtilityState,
    ) -> Result<Self, ArenaError> {
        check(
            restored.source_checkpoint.is_some()
                && restored
                    .source_payload_state_sha256
                    .as_ref()
                    .is_some_and(|sha| generation_sha(sha))
                && restored.checkpoint_cost.is_some()
                && restored.source_snapshot_cost.is_some()
                && restored.source_snapshot_ancestor_cost.is_some(),
            "restored utility state requires actual original checkpoint, payload and cost sidecar",
        )?;
        Self::new_inner(
            window,
            preparation,
            dataset,
            restored,
            UtilityStateAdmission::OriginalCheckpoint,
            UtilityLedgerMode::OriginalRuntime,
        )
    }
    /// Metadata replay after the checkpoint owner verified original held
    /// projection/journal/closure files. No InitialUtilityMeasurement is
    /// issued, and this ledger cannot return a live RPC ACK or learned lease.
    pub(super) fn for_historical_replay(
        window: RoventUtilityTrainWindow,
        preparation: Value,
        dataset: VerifiedUtilityDataset,
        restored: RestoredUtilityState,
        original_admission_scope: &str,
    ) -> Result<Self, ArenaError> {
        let admission = [
            UtilityStateAdmission::InitialPeerMeasurement,
            UtilityStateAdmission::OriginalCheckpoint,
            UtilityStateAdmission::ForkPeerMeasurementFromOriginalCheckpoint,
        ]
        .into_iter()
        .find(|kind| kind.scope() == original_admission_scope)
        .ok_or_else(|| invalid("unknown original utility state admission scope"))?;
        Self::new_inner(
            window,
            preparation,
            dataset,
            restored,
            admission,
            UtilityLedgerMode::HistoricalReplay,
        )
    }
    fn new_inner(
        window: RoventUtilityTrainWindow,
        preparation: Value,
        dataset: VerifiedUtilityDataset,
        restored: RestoredUtilityState,
        state_admission: UtilityStateAdmission,
        mode: UtilityLedgerMode,
    ) -> Result<Self, ArenaError> {
        window.join_cohort_preparation(&preparation)?;
        check(
            window.family() == dataset.family(),
            "window/dataset family mismatch",
        )?;
        let numerical = window.family().numerical(&preparation)?;
        check(
            numerical.preparation_sha256() == window.preparation_sha256,
            "original family preparation seal differs",
        )?;
        check(
            window.dataset_sha256 == dataset.dataset_sha256(),
            "window differs from original held utility dataset",
        )?;
        restored.states.validate()?;
        source_declaration(&restored.initial_source_declaration)?;
        restored.ancestor_cost.checked_add(&restored.cost)?;
        check(
            restored.completed_updates == restored.completion_receipts.len() as u64
                && restored.ancestor_updates == restored.ancestor_completion_sha256.len() as u64
                && restored.completed_updates <= restored.dispatched_updates
                && restored.dispatched_updates <= ROVENT_UTILITY_MAX_UPDATES
                && add(restored.completed_updates, restored.ancestor_updates)?
                    <= ROVENT_UTILITY_MAX_UPDATES
                && add(
                    add(restored.completed_updates, restored.ancestor_updates)?,
                    window.max_optimizer_updates,
                )? <= ROVENT_UTILITY_MAX_UPDATES,
            "original completion-chain/count/finite credit differs",
        )?;
        let mut seen = BTreeSet::new();
        for hash in restored
            .ancestor_completion_sha256
            .iter()
            .map(String::as_str)
            .chain(
                restored
                    .completion_receipts
                    .iter()
                    .map(|pin| pin.sha256.as_str()),
            )
        {
            check(
                generation_sha(hash) && seen.insert(hash),
                "original completion chain contains a repeated or invalid receipt",
            )?;
        }
        for pin in &restored.completion_receipts {
            check(
                pin.path.is_absolute()
                    && pin.bytes > 0
                    && pin.bytes <= MAX_COMPLETION_RECEIPT_BYTES as u64,
                "original held completion receipt pin required",
            )?;
        }
        check(
            restored.frozen_base_weights_sha256 == numerical.frozen_base_weights_sha256()
                && generation_sha(&restored.frozen_base_mapping_sha256),
            "separate original frozen semantic/mapping identity differs",
        )?;
        if let Some(observation) = &restored.checkpoint_cost {
            observation.validate()?;
        }
        let sampler = UtilitySampler::from_original(&restored.sampler, &window, &dataset)?;
        check(
            sampler.state_sha256(dataset.context_ids())? == restored.states.sampler_state_sha256,
            "original sampler state seal differs from exact typed framing",
        )?;
        Ok(Self {
            dataset,
            window,
            _preparation: preparation,
            numerical,
            states: restored.states.clone(),
            sampler,
            cost: restored.cost,
            completed_updates: restored.completed_updates,
            dispatched_updates: restored.dispatched_updates,
            initial_dispatched_updates: restored.dispatched_updates,
            initial_completed_updates: restored.completed_updates,
            initial_cost: restored.cost,
            completion_receipts: restored.completion_receipts.clone(),
            restored,
            consumed: 0,
            groups: 0,
            masked_groups: 0,
            pending_contexts: Vec::new(),
            pending_update: None,
            begun: false,
            state_admission,
            begin_payload: None,
            mode,
        })
    }
    fn chain_sha256(&self) -> Result<String, ArenaError> {
        let hashes = self
            .restored
            .ancestor_completion_sha256
            .iter()
            .chain(self.completion_receipts.iter().map(|pin| &pin.sha256))
            .collect::<Vec<_>>();
        digest(self.window.family().domain(STATE_DOMAIN), &hashes)
    }
    pub(super) fn initial_ledger_projection(&self) -> Value {
        json!({"schema":self.window.family().domain("rz-rovent-utility-initial-ledger-projection/1"),
            "state_admission_scope":self.state_admission.scope(),"states":self.restored.states,
            "sampler":self.restored.sampler,"completed_updates":self.restored.completed_updates,
            "dispatched_updates":self.restored.dispatched_updates,"ancestor_updates":self.restored.ancestor_updates,
            "cost":self.restored.cost,"ancestor_cost":self.restored.ancestor_cost,
            "completion_receipts":self.restored.completion_receipts,
            "ancestor_completion_sha256":self.restored.ancestor_completion_sha256,
            "initial_source_declaration":self.restored.initial_source_declaration,
            "frozen_base_weights_sha256":self.restored.frozen_base_weights_sha256,
            "frozen_base_mapping_sha256":self.restored.frozen_base_mapping_sha256,
            "source_checkpoint":self.restored.source_checkpoint,
            "source_payload_state_sha256":self.restored.source_payload_state_sha256,
            "source_snapshot_cost_observation":self.restored.source_snapshot_cost_observation,
            "source_snapshot_cost":self.restored.source_snapshot_cost,
            "source_snapshot_ancestor_cost":self.restored.source_snapshot_ancestor_cost,
            "checkpoint_cost":self.restored.checkpoint_cost,"historical_evidence_only":true,
            "execution_authorized":false})
    }
    /// Observed dispatch/cost are retained even while durable completion is
    /// uncertain. They cannot grant learned eligibility before the matching
    /// original receipt or be refunded into an apparent unused reservation.
    pub(super) fn observation(&self) -> Value {
        let observed_dispatched = self
            .pending_update
            .as_ref()
            .map_or(self.dispatched_updates, |pending| {
                pending.dispatched_updates
            });
        json!({"schema":self.window.family().domain("rz-rovent-utility-parent-ledger-observation/1"),
            "window_id":self.window.window_id,"run_id":self.window.run_id,"dataset_sha256":self.dataset.dataset_sha256(),
            "completed_updates":self.completed_updates,"dispatched_updates":self.dispatched_updates,
            "observed_dispatched_updates":observed_dispatched,
            "window_completed_updates":self.completed_updates-self.initial_completed_updates,
            "window_dispatched_updates":observed_dispatched-self.initial_dispatched_updates,
            "ancestor_updates":self.restored.ancestor_updates,"committed_cost":self.cost,
            "pending_observed_cost":self.pending_update.as_ref().map(|pending|pending.cost),
            "consumed_contexts":self.consumed,"groups":self.groups,"masked_groups":self.masked_groups,
            "unupdated_context_sha256":self.pending_contexts.iter().map(|row|&row.sha256).collect::<Vec<_>>(),
            "pending_receipt":self.pending_update.as_ref().map(|pending|json!({"sha256":pending.sha256,"bytes":pending.bytes.len(),"original_body":pending.body})),
            "initial_state_admission_scope":self.state_admission.scope(),"historical_evidence_only":true,
            "execution_authorized":false})
    }
    fn derived_state(&self) -> Result<Value, ArenaError> {
        let total = add(self.completed_updates, self.restored.ancestor_updates)?;
        Ok(
            json!({"schema":self.window.family().domain(STATE_DOMAIN),"initial_source_declaration":self.restored.initial_source_declaration,"current_tensor_kind":if total>0 {"learned"} else {"initial_source_only"},"completed_updates":self.completed_updates,"ancestor_updates":self.restored.ancestor_updates,"total_updates":total,"completion_chain_sha256":self.chain_sha256()?,"recipe_sha256":self.numerical.recipe_sha256(),"registration_sha256":self.numerical.source_registration_sha256(),"frozen_base_weights_sha256":self.restored.frozen_base_weights_sha256,"frozen_base_mapping_sha256":self.restored.frozen_base_mapping_sha256,"live_training_authority_reissued":false}),
        )
    }
    fn current(&self) -> Result<(), ArenaError> {
        check(
            self.begun && self.pending_update.is_none(),
            "original window not begun or durable update receipt still pending",
        )
    }
    fn validate_counts(&self, payload: &Value) -> Result<(), ArenaError> {
        check(
            count(&payload["completed_updates"])? == self.completed_updates
                && count(&payload["ancestor_updates"])? == self.restored.ancestor_updates,
            "current/ancestor completion counts differ",
        )
    }
    fn validate_cost(&self, observed: RoventUtilityRecordedCost) -> Result<(), ArenaError> {
        cost_monotone(self.cost, observed)?;
        check(
            observed.samples == add(self.initial_cost.samples, self.consumed)?
                && observed.masked_groups
                    == add(self.initial_cost.masked_groups, self.masked_groups)?
                && observed
                    .output_bytes
                    .checked_sub(self.initial_cost.output_bytes)
                    .is_some_and(|used| used <= self.window.max_output_bytes),
            "sample/masked/current-window output cost differs",
        )
    }
    pub(super) fn begin_window(&mut self, payload: &Value) -> Result<Value, ArenaError> {
        check(
            self.mode == UtilityLedgerMode::OriginalRuntime,
            "historical replay cannot begin a live peer window",
        )?;
        self.begin_inner(payload)
    }
    fn begin_inner(&mut self, payload: &Value) -> Result<Value, ArenaError> {
        let mut names = vec![
            "schema",
            "preparation_sha256",
            "window_spec_sha256",
            "dataset_sha256",
            "initial_source_declaration",
            "frozen_base_mapping_sha256",
            "frozen_base_weights_sha256",
            "completed_updates",
            "ancestor_updates",
            "derived_training_state_sha256",
        ];
        names.extend(STATE_FIELDS);
        fields(payload, &names)?;
        check(
            !self.begun
                && self.pending_update.is_none()
                && payload["schema"]
                    == self
                        .window
                        .family()
                        .domain("rz-rovent-utility-trainer-begin/1")
                && sha(&payload["preparation_sha256"])? == self.numerical.preparation_sha256()
                && sha(&payload["window_spec_sha256"])? == self.window.canonical_sha256()?
                && sha(&payload["dataset_sha256"])? == self.dataset.dataset_sha256(),
            "one original begin/source/window/dataset binding required",
        )?;
        self.validate_counts(payload)?;
        check(
            UtilityStateSeals::from_payload(payload)? == self.states
                && payload["initial_source_declaration"]
                    == self.restored.initial_source_declaration
                && sha(&payload["frozen_base_mapping_sha256"])?
                    == self.restored.frozen_base_mapping_sha256
                && sha(&payload["frozen_base_weights_sha256"])?
                    == self.restored.frozen_base_weights_sha256
                && sha(&payload["derived_training_state_sha256"])?
                    == digest(
                        self.window.family().domain(STATE_DOMAIN),
                        &self.derived_state()?,
                    )?,
            "begin changed original five-state or completion-chain seal",
        )?;
        self.begun = true;
        self.begin_payload = Some(payload.clone());
        Ok(json!({}))
    }
    pub(super) fn consume_pair_context(&mut self, payload: &Value) -> Result<Value, ArenaError> {
        check(
            self.mode == UtilityLedgerMode::OriginalRuntime,
            "historical replay cannot issue a live consume acknowledgement",
        )?;
        self.consume_inner(payload)
    }
    fn consume_inner(&mut self, payload: &Value) -> Result<Value, ArenaError> {
        self.current()?;
        fields(payload, &["schema", "source_projection"])?;
        check(
            payload["schema"]
                == self
                    .window
                    .family()
                    .domain("rz-rovent-utility-trainer-consume/1")
                && self.pending_contexts.len() < ACCUMULATION
                && self.consumed < u64::from(self.window.max_contexts)
                && self.groups < u64::from(self.window.max_groups)
                && self.completed_updates - self.initial_completed_updates
                    < self.window.max_optimizer_updates,
            "utility context/group admission exhausted or update unacknowledged",
        )?;
        let context = self
            .dataset
            .validate_context_projection(&payload["source_projection"])?;
        check(
            context.source_projection() == &payload["source_projection"],
            "original source projection differs",
        )?;
        let row = ConsumedContext::from_original(context)?;
        let mut sampler = self.sampler.clone();
        let index = sampler.next(self.dataset.context_ids())?;
        check(
            self.dataset.context_ids()[index] == row.sha256,
            "context is not next original sampler occurrence",
        )?;
        let consumed = add(self.consumed, 1)?;
        self.sampler = sampler;
        self.consumed = consumed;
        self.pending_contexts.push(row);
        if self.pending_contexts.len() == ACCUMULATION {
            self.groups = add(self.groups, 1)?;
            if self
                .pending_contexts
                .iter()
                .all(|context| context.unique_known_pairs == 0)
            {
                self.masked_groups = add(self.masked_groups, 1)?;
                self.pending_contexts.clear();
            }
        }
        Ok(json!({}))
    }
    pub(super) fn prepare_update(&mut self, payload: &Value) -> Result<&Value, ArenaError> {
        check(
            self.mode == UtilityLedgerMode::OriginalRuntime,
            "historical replay cannot prepare a live learner update",
        )?;
        self.prepare_update_inner(payload)
    }
    fn prepare_update_inner(&mut self, payload: &Value) -> Result<&Value, ArenaError> {
        self.current()?;
        let mut names = vec![
            "schema",
            "event",
            "cost",
            "pending_microbatches",
            "frozen_base_weights_sha256",
            "frozen_base_mapping_sha256",
            "prior_completion_chain_sha256",
        ];
        names.extend(STATE_FIELDS);
        fields(payload, &names)?;
        check(
            payload["schema"]
                == self
                    .window
                    .family()
                    .domain("rz-rovent-utility-trainer-update/1")
                && self.pending_contexts.len() == ACCUMULATION
                && count(&payload["pending_microbatches"])? == 0
                && sha(&payload["frozen_base_weights_sha256"])?
                    == self.restored.frozen_base_weights_sha256
                && sha(&payload["frozen_base_mapping_sha256"])?
                    == self.restored.frozen_base_mapping_sha256
                && sha(&payload["prior_completion_chain_sha256"])? == self.chain_sha256()?,
            "update changed source/frozen identity/chain or lacks complete accumulation",
        )?;
        let event = &payload["event"];
        fields(
            event,
            &[
                "schema",
                "completed_updates",
                "dispatched_updates",
                "unique_known_pair_denominator",
                "consumed_context_sha256",
                "groups",
                "contexts",
            ],
        )?;
        let completed = add(self.completed_updates, 1)?;
        let dispatched = add(self.dispatched_updates, 1)?;
        let denominator = accumulation_denominator(&self.pending_contexts)?;
        let contexts = self
            .pending_contexts
            .iter()
            .map(|context| &context.sha256)
            .collect::<Vec<_>>();
        check(
            event["schema"]
                == self
                    .window
                    .family()
                    .domain("rz-rovent-utility-learner-update/1")
                && count(&event["completed_updates"])? == completed
                && count(&event["dispatched_updates"])? == dispatched
                && completed - self.initial_completed_updates <= self.window.max_optimizer_updates
                && denominator > 0
                && count(&event["unique_known_pair_denominator"])? == denominator
                && event["consumed_context_sha256"] == json!(contexts)
                && count(&event["groups"])? == 1
                && count(&event["contexts"])? == ACCUMULATION as u64,
            "completed update differs from actual eight ordered known-pair occurrences",
        )?;
        let states = UtilityStateSeals::from_payload(payload)?;
        check(
            states.sampler_state_sha256 == self.sampler.state_sha256(self.dataset.context_ids())?
                && states.model_state_sha256 != self.states.model_state_sha256
                && states.optimizer_state_sha256 != self.states.optimizer_state_sha256
                && states.scheduler_state_sha256 != self.states.scheduler_state_sha256,
            "successful utility step must change original parameter/moment/scheduler state and preserve exact sampler",
        )?;
        let observed = cost(&payload["cost"])?;
        self.validate_cost(observed)?;
        let active_contexts = self
            .pending_contexts
            .iter()
            .filter(|row| row.unique_known_pairs > 0)
            .count() as u64;
        check(
            observed.forward_calls == add(self.cost.forward_calls, active_contexts)?
                && observed.backward_calls == add(self.cost.backward_calls, active_contexts)?,
            "known-only forward/backward count differs from admitted accumulation",
        )?;
        let body = json!({"schema":self.window.family().domain(COMPLETION_DOMAIN),"window_id":self.window.window_id,"run_id":self.window.run_id,"dataset_sha256":self.dataset.dataset_sha256(),"preparation_sha256":self.window.preparation_sha256,"source_registration_sha256":self.numerical.source_registration_sha256(),"optimizer_owner":self.numerical.optimizer_owner(),"initial_state_admission_scope":self.state_admission.scope(),"completed_updates":completed,"dispatched_updates":dispatched,"ancestor_updates":self.restored.ancestor_updates,"window_completed_updates":completed-self.initial_completed_updates,"window_dispatched_updates":dispatched-self.initial_dispatched_updates,"consumed_contexts":self.pending_contexts.iter().map(|row|json!({"context_sha256":row.sha256,"source_projection":row.source_projection,"pair_sha256":row.pair_sha256,"unique_known_pairs":row.unique_known_pairs})).collect::<Vec<_>>(),"original_request":payload,"historical_evidence_only":true,"execution_authorized":false});
        let mut canonical = body.clone();
        canonical.sort_all_objects();
        let mut bytes =
            serde_json::to_vec(&canonical).map_err(|error| invalid(error.to_string()))?;
        bytes.push(b'\n');
        check(
            bytes.len() <= MAX_COMPLETION_RECEIPT_BYTES,
            "completion receipt exceeds its separate bounded durable body allowance",
        )?;
        let sha256 = format!("{:x}", Sha256::digest(&bytes));
        self.pending_update = Some(PendingUtilityUpdate {
            body,
            bytes,
            sha256,
            states,
            cost: observed,
            completed_updates: completed,
            dispatched_updates: dispatched,
        });
        Ok(&self.pending_update.as_ref().expect("just prepared").body)
    }
    pub(super) fn pending_receipt_bytes(&self) -> Option<&[u8]> {
        self.pending_update
            .as_ref()
            .map(|pending| pending.bytes.as_slice())
    }
    pub(super) fn commit_update(
        &mut self,
        original_receipt: RoventTrainingFilePin,
    ) -> Result<Value, ArenaError> {
        check(
            self.mode == UtilityLedgerMode::OriginalRuntime,
            "historical replay cannot return a live durable completion acknowledgement",
        )?;
        self.commit_update_inner(original_receipt)
    }
    fn commit_update_inner(
        &mut self,
        original_receipt: RoventTrainingFilePin,
    ) -> Result<Value, ArenaError> {
        let pending = self
            .pending_update
            .as_ref()
            .ok_or_else(|| invalid("no original prepared utility update"))?;
        check(
            original_receipt.path.is_absolute()
                && original_receipt.sha256 == pending.sha256
                && original_receipt.bytes == pending.bytes.len() as u64
                && !self
                    .completion_receipts
                    .iter()
                    .any(|pin| pin.sha256 == original_receipt.sha256)
                && !self
                    .restored
                    .ancestor_completion_sha256
                    .contains(&original_receipt.sha256),
            "durable original receipt pin differs from prepared bytes or repeats completion",
        )?;
        let pending = self.pending_update.take().expect("checked pending update");
        self.completed_updates = pending.completed_updates;
        self.dispatched_updates = pending.dispatched_updates;
        self.states = pending.states;
        self.cost = pending.cost;
        let ack = json!({"sha256":original_receipt.sha256,"bytes":original_receipt.bytes});
        self.completion_receipts.push(original_receipt);
        self.pending_contexts.clear();
        Ok(ack)
    }
    pub(super) fn validate_checkpoint_state(&self, payload: &Value) -> Result<Value, ArenaError> {
        check(
            self.mode == UtilityLedgerMode::OriginalRuntime,
            "historical replay cannot authorize current checkpoint admission",
        )?;
        let mut names = vec![
            "schema",
            "checkpoint",
            "preparation_sha256",
            "completed_updates",
            "ancestor_updates",
            "snapshot_cost",
            "ancestor_cost",
            "source_payload_state_sha256",
            "source_snapshot_cost_observation",
            "derived_training_state_sha256",
        ];
        names.extend(STATE_FIELDS);
        fields(payload, &names)?;
        check(
            !self.begun
                && payload["schema"]
                    == self
                        .window
                        .family()
                        .domain("rz-rovent-utility-trainer-resume/1")
                && sha(&payload["preparation_sha256"])? == self.window.preparation_sha256,
            "resume admission must precede original window",
        )?;
        self.validate_counts(payload)?;
        let source = self
            .restored
            .source_checkpoint
            .as_ref()
            .ok_or_else(|| invalid("actual verified original checkpoint missing"))?;
        check(
            pin(&payload["checkpoint"])? == *source
                && UtilityStateSeals::from_payload(payload)? == self.states
                && Some(cost(&payload["snapshot_cost"])?) == self.restored.source_snapshot_cost
                && Some(cost(&payload["ancestor_cost"])?)
                    == self.restored.source_snapshot_ancestor_cost
                && payload["source_payload_state_sha256"]
                    == json!(self.restored.source_payload_state_sha256)
                && payload["source_snapshot_cost_observation"]
                    == json!(self.restored.source_snapshot_cost_observation)
                && sha(&payload["derived_training_state_sha256"])?
                    == digest(
                        self.window.family().domain(STATE_DOMAIN),
                        &self.derived_state()?,
                    )?,
            "resume changed original checkpoint/state/cost/derived chain",
        )?;
        let observation = self
            .restored
            .checkpoint_cost
            .as_ref()
            .ok_or_else(|| invalid("original checkpoint cost sidecar missing"))?;
        observation.validate()?;
        serde_json::to_value(observation).map_err(|error| invalid(error.to_string()))
    }
    pub(super) fn checkpoint_projection(
        &self,
        payload: &Value,
        original_checkpoint: &RoventTrainingFilePin,
        original_tensor_projection: &RoventTrainingFilePin,
    ) -> Result<Value, ArenaError> {
        self.current()?;
        let mut names = vec![
            "schema",
            "checkpoint",
            "tensor_projection",
            "completed_updates",
            "ancestor_updates",
            "snapshot_cost",
            "post_snapshot_cost",
            "pending_microbatches",
            "derived_training_state_sha256",
        ];
        names.extend(STATE_FIELDS);
        fields(payload, &names)?;
        check(
            payload["schema"]
                == self
                    .window
                    .family()
                    .domain("rz-rovent-utility-trainer-checkpoint/1")
                && pin(&payload["checkpoint"])? == *original_checkpoint
                && pin(&payload["tensor_projection"])? == *original_tensor_projection
                && count(&payload["pending_microbatches"])? == 0
                && self.pending_contexts.len() <= ACCUMULATION,
            "checkpoint requires original held file at a no-gradient boundary",
        )?;
        self.validate_counts(payload)?;
        let states = UtilityStateSeals::from_payload(payload)?;
        check(
            states.same_parameters_and_moments(&self.states)
                && states.sampler_state_sha256
                    == self.sampler.state_sha256(self.dataset.context_ids())?
                && sha(&payload["derived_training_state_sha256"])?
                    == digest(
                        self.window.family().domain(STATE_DOMAIN),
                        &self.derived_state()?,
                    )?,
            "checkpoint changed active states, exact sampler or completion chain",
        )?;
        let snapshot = cost(&payload["snapshot_cost"])?;
        let post = cost(&payload["post_snapshot_cost"])?;
        self.validate_cost(snapshot)?;
        cost_monotone(snapshot, post)?;
        check(
            post.output_bytes.checked_sub(snapshot.output_bytes)
                == Some(add(
                    original_checkpoint.bytes,
                    original_tensor_projection.bytes,
                )?)
                && post
                    .output_bytes
                    .checked_sub(self.initial_cost.output_bytes)
                    .is_some_and(|used| used <= self.window.max_output_bytes),
            "checkpoint output bytes must be charged once to this original window",
        )?;
        Ok(
            json!({"schema":self.window.family().domain("rz-rovent-utility-parent-checkpoint-projection/1"),"checkpoint":original_checkpoint,"tensor_projection":original_tensor_projection,"states":states,"completed_updates":self.completed_updates,"ancestor_updates":self.restored.ancestor_updates,"snapshot_cost":snapshot,"post_snapshot_cost":post,"unupdated_context_sha256":self.pending_contexts.iter().map(|row|&row.sha256).collect::<Vec<_>>(),"pending_receipt":false,"historical_evidence_only":true,"execution_authorized":false}),
        )
    }
    pub(super) fn prepare_checkpoint_boundary(
        &self,
        payload: &Value,
        original_checkpoint: &RoventTrainingFilePin,
        original_tensor_projection: &RoventTrainingFilePin,
    ) -> Result<UtilityCheckpointBoundary<'_>, ArenaError> {
        check(
            self.mode == UtilityLedgerMode::OriginalRuntime,
            "historical replay cannot issue a current checkpoint boundary",
        )?;
        let projection =
            self.checkpoint_projection(payload, original_checkpoint, original_tensor_projection)?;
        Ok(UtilityCheckpointBoundary {
            ledger: self,
            original_tensor_pin: original_checkpoint.clone(),
            original_tensor_projection_pin: original_tensor_projection.clone(),
            snapshot_cost: cost(&payload["snapshot_cost"])?,
            post_snapshot_cost: cost(&payload["post_snapshot_cost"])?,
            projection,
        })
    }

    /// Replay only after the checkpoint owner has verified original receipt
    /// FDs and its complete ordered consumption journal. No peer ACK is issued.
    /// A compact receipt shortcut can recover its eight sources only when no
    /// prior consumption is pending; missing masked history still fails the
    /// exact sampler/cost join instead of manufacturing a successful chain.
    pub(super) fn replay_historical_completion(
        &mut self,
        original_body: &Value,
        original_held_pin: &RoventTrainingFilePin,
    ) -> Result<(), ArenaError> {
        check(
            self.mode == UtilityLedgerMode::HistoricalReplay,
            "original runtime cannot reinterpret a historical completion as current",
        )?;
        self.current()?;
        let contexts = original_body["consumed_contexts"]
            .as_array()
            .ok_or_else(|| invalid("original historical context roster missing"))?;
        check(
            contexts.len() == ACCUMULATION,
            "historical completion must retain original eight contexts",
        )?;
        if self.pending_contexts.is_empty() {
            for context in contexts {
                fields(
                    context,
                    &[
                        "context_sha256",
                        "source_projection",
                        "pair_sha256",
                        "unique_known_pairs",
                    ],
                )?;
                self.consume_inner(&json!({"schema":self.window.family().domain("rz-rovent-utility-trainer-consume/1"),"source_projection":context["source_projection"]}))?;
            }
        }
        check(
            self.pending_contexts.len() == ACCUMULATION,
            "historical journal has a partial or masked group at a completed boundary",
        )?;
        let request = original_body
            .get("original_request")
            .ok_or_else(|| invalid("original completed request missing"))?;
        let regenerated = self.prepare_update_inner(request)?;
        check(
            regenerated == original_body,
            "historical receipt differs from actual original source/state/count/chain replay",
        )?;
        self.commit_update_inner(original_held_pin.clone())?;
        Ok(())
    }
    pub(super) fn replay_historical_begin(
        &mut self,
        original_payload: &Value,
    ) -> Result<(), ArenaError> {
        check(
            self.mode == UtilityLedgerMode::HistoricalReplay,
            "historical begin requires separate metadata replay ledger",
        )?;
        self.begin_inner(original_payload)?;
        Ok(())
    }
    pub(super) fn replay_historical_consumption(
        &mut self,
        original_payload: &Value,
    ) -> Result<(), ArenaError> {
        check(
            self.mode == UtilityLedgerMode::HistoricalReplay,
            "historical consume requires separate metadata replay ledger",
        )?;
        self.consume_inner(original_payload)?;
        Ok(())
    }
    /// Read-only replay operands, never a peer ACK, tensor restore, current
    /// checkpoint boundary or learned export admission.
    pub(super) fn historical_sampler_projection(&self) -> Value {
        self.sampler.projection(self.dataset.context_ids())
    }
    pub(super) fn historical_derived_training_state(&self) -> Result<Value, ArenaError> {
        self.derived_state()
    }
    pub(super) fn historical_dispatched_updates(&self) -> u64 {
        self.dispatched_updates
    }
    pub(super) fn historical_rng_state_sha256(&self) -> &str {
        &self.states.rng_state_sha256
    }
    pub(super) fn learned_export_projection(&self, payload: &Value) -> Result<Value, ArenaError> {
        check(
            self.mode == UtilityLedgerMode::OriginalRuntime,
            "historical replay cannot issue learned export admission",
        )?;
        self.current()?;
        let mut names = vec![
            "schema",
            "kind",
            "preparation_sha256",
            "completed_updates",
            "ancestor_updates",
            "derived_training_state_sha256",
        ];
        names.extend(STATE_FIELDS);
        fields(payload, &names)?;
        let total = add(self.completed_updates, self.restored.ancestor_updates)?;
        check(
            payload["schema"]
                == self
                    .window
                    .family()
                    .domain("rz-rovent-utility-trainer-export/1")
                && matches!(payload["kind"].as_str(), Some("collection" | "deployment"))
                && sha(&payload["preparation_sha256"])? == self.window.preparation_sha256
                && total > 0
                && self.pending_contexts.is_empty()
                && UtilityStateSeals::from_payload(payload)? == self.states
                && sha(&payload["derived_training_state_sha256"])?
                    == digest(
                        self.window.family().domain(STATE_DOMAIN),
                        &self.derived_state()?,
                    )?,
            "learned export requires positive original held completion chain, frozen state and no partial work",
        )?;
        self.validate_counts(payload)?;
        Ok(
            json!({"kind":"learned","steps":total,"run_id":self.window.run_id,"dataset_sha256":self.dataset.dataset_sha256()}),
        )
    }
    pub(super) fn ended_projection(&self, payload: &Value) -> Result<Value, ArenaError> {
        self.current()?;
        fields(
            payload,
            &[
                "schema",
                "status",
                "completed_updates",
                "dispatched_updates",
                "window_dispatched_updates",
                "ancestor_updates",
                "cost",
                "pending_microbatches",
                "quarantined",
                "physical_completion",
                "checkpoint",
                "result",
            ],
        )?;
        self.validate_counts(payload)?;
        let observed = cost(&payload["cost"])?;
        self.validate_cost(observed)?;
        let dispatched = count(&payload["dispatched_updates"])?;
        let window_dispatched = count(&payload["window_dispatched_updates"])?;
        let no_update = zero_update_closure(
            self.window.family(),
            payload,
            self.completed_updates - self.initial_completed_updates,
            self.dispatched_updates - self.initial_dispatched_updates,
        )?;
        check(
            payload["schema"]
                == self
                    .window
                    .family()
                    .domain("rz-rovent-utility-trainer-end/1")
                && dispatched == self.dispatched_updates
                && window_dispatched == self.dispatched_updates - self.initial_dispatched_updates
                && count(&payload["pending_microbatches"])? == 0
                && payload["quarantined"].is_boolean()
                && matches!(
                    payload["physical_completion"].as_str(),
                    Some("confirmed" | "unknown")
                )
                && (no_update
                    || matches!(
                        payload["status"].as_str(),
                        Some(
                            "window_complete"
                                | "window_limit"
                                | "partial_accumulation_discarded"
                                | "export_complete"
                                | "export_failed"
                        )
                    )),
            "original end observation/count/status differs",
        )?;
        check(
            self.pending_contexts.is_empty()
                || payload["status"] == "partial_accumulation_discarded"
                || no_update,
            "unupdated original contexts require an explicit partial-accumulation closure observation",
        )?;
        Ok(
            json!({"schema":self.window.family().domain("rz-rovent-utility-parent-ended-projection/1"),"original_end":payload,"window_completed_updates":self.completed_updates-self.initial_completed_updates,"window_dispatched_updates":window_dispatched,"consumed_contexts":self.consumed,"groups":self.groups,"masked_groups":self.masked_groups,"unupdated_context_sha256":self.pending_contexts.iter().map(|row|&row.sha256).collect::<Vec<_>>(),"pending_durable_receipt":false,"historical_evidence_only":true,"execution_authorized":false}),
        )
    }
}

#[cfg(test)]
mod tests {
    //! Plain metadata/framing only. No fixture constructs a verified dataset,
    //! live peer, tensor owner, durable receipt or learned capability.
    use super::*;

    #[test]
    fn zero_update_closure_is_new_family_only_and_never_grants_publication() {
        let original = json!({
            "status":"no_update_checkpoint_not_published",
            "checkpoint":null,"quarantined":false,
            "physical_completion":"confirmed","pending_microbatches":0
        });
        assert!(
            zero_update_closure(UtilityRuntimeFamily::InterventionV2, &original, 0, 0).unwrap()
        );
        assert!(zero_update_closure(UtilityRuntimeFamily::CohortV1, &original, 0, 0).is_err());
        for (completed, dispatched) in [(1, 1), (0, 1)] {
            assert!(
                zero_update_closure(
                    UtilityRuntimeFamily::InterventionV2,
                    &original,
                    completed,
                    dispatched
                )
                .is_err()
            );
        }
        for (field, value) in [
            ("checkpoint", json!({"path":"/unused/checkpoint.pt"})),
            ("quarantined", json!(true)),
            ("physical_completion", json!("unknown")),
            ("pending_microbatches", json!(1)),
        ] {
            let mut changed = original.clone();
            changed[field] = value;
            assert!(
                zero_update_closure(UtilityRuntimeFamily::InterventionV2, &changed, 0, 0).is_err()
            );
        }
    }

    #[test]
    fn plain_sampler_digest_uses_typed_recursive_framing_not_json_digest() {
        let value = json!({"b":[2],"a":"x"});
        let exact = b"dict[\"str\",\"a\"];[\"str\",\"x\"];[\"str\",\"b\"];list1[\"int\",2];;;";
        assert_eq!(
            typed_plain_state_sha256(&value).unwrap(),
            format!("{:x}", Sha256::digest(exact))
        );
        assert_ne!(
            typed_plain_state_sha256(&value).unwrap(),
            format!("{:x}", Sha256::digest(serde_json::to_vec(&value).unwrap()))
        );
        for value in [
            json!(true),
            json!(1.0),
            json!(-1),
            json!(null),
            json!("한글"),
        ] {
            assert!(typed_plain_state_sha256(&value).is_err());
        }
    }

    #[test]
    fn sampler_binary_order_keeps_original_index_namespace_and_epoch() {
        let ids = vec!["a".repeat(64), "b".repeat(64), "c".repeat(64)];
        let order = UtilitySampler::permutation(&ids, 31, 0).unwrap();
        let mut expected = Vec::new();
        for (index, id) in ids.iter().enumerate() {
            let mut framed = b"rz-rovent-utility-sampler/1\0".to_vec();
            framed.extend(31u64.to_le_bytes());
            framed.extend(0u64.to_le_bytes());
            framed.extend((index as u32).to_le_bytes());
            for offset in 0..32 {
                framed.push(u8::from_str_radix(&id[offset * 2..offset * 2 + 2], 16).unwrap());
            }
            expected.push((<[u8; 32]>::from(Sha256::digest(framed)), index));
        }
        expected.sort_unstable();
        assert_eq!(
            order,
            expected
                .into_iter()
                .map(|(_, index)| index)
                .collect::<Vec<_>>()
        );
        let mut sampler = UtilitySampler {
            run_id: "run".into(),
            pool_sha256: "d".repeat(64),
            seed: 31,
            epoch: 0,
            permutation: order.clone(),
            cursor: 0,
        };
        for index in order {
            assert_eq!(sampler.next(&ids).unwrap(), index);
        }
        let next = UtilitySampler::permutation(&ids, 31, 1).unwrap();
        assert_eq!(sampler.next(&ids).unwrap(), next[0]);
        assert_eq!((sampler.epoch, sampler.cursor), (1, 1));
    }

    #[test]
    fn denominator_sums_unique_pairs_per_original_context_occurrence() {
        let row = ConsumedContext {
            sha256: "a".repeat(64),
            source_projection: json!({}),
            pair_sha256: vec!["b".repeat(64)],
            unique_known_pairs: 1,
        };
        // Repeated occurrences remain eight consumed contexts; pair IDs are
        // unique within each source context, not globally deduplicated here.
        assert_eq!(accumulation_denominator(&vec![row.clone(); 8]).unwrap(), 8);
        assert!(accumulation_denominator(&vec![row.clone(); 7]).is_err());
        let mut masked = row;
        masked.unique_known_pairs = 0;
        masked.pair_sha256.clear();
        assert_eq!(accumulation_denominator(&vec![masked; 8]).unwrap(), 0);
    }

    #[test]
    fn original_cost_observations_cannot_decrease_or_overflow() {
        let before = RoventUtilityRecordedCost {
            samples: 8,
            output_bytes: 11,
            ..Default::default()
        };
        let after = RoventUtilityRecordedCost {
            samples: 9,
            output_bytes: 12,
            ..Default::default()
        };
        cost_monotone(before, after).unwrap();
        assert!(cost_monotone(after, before).is_err());
        assert!(add(u64::MAX, 1).is_err());
        for forged in [json!(true), json!(1.0), json!(-1)] {
            let mut value = serde_json::to_value(before).unwrap();
            value["samples"] = forged;
            assert!(cost(&value).is_err());
        }
    }

    #[test]
    fn initial_declaration_is_exact_metadata_and_never_a_completion_chain() {
        source_declaration(&json!({"kind":"untrained"})).unwrap();
        source_declaration(&json!({"kind":"learned","run_id":"historical","steps":3,"dataset_sha256":"a".repeat(64)})).unwrap();
        for value in [
            json!({"kind":"untrained","steps":0}),
            json!({"kind":"learned","run_id":"old","steps":true,"dataset_sha256":"a".repeat(64)}),
            json!({"kind":"prepared_metadata_only","execution_authorized":true}),
        ] {
            assert!(source_declaration(&value).is_err());
        }
        assert!(fields(&json!({"a":1,"b":2}), &["a"]).is_err());
    }
}
