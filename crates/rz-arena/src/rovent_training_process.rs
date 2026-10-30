//! Original Rovent learner process ownership, separate from the fixed PALS G0
//! trainer. Parsing a registration or recovered corpus never issues authority.
//! Native row custody originates in `PublishedRoventRecord`; source/runtime,
//! held input descriptors, one local RPC peer, update journal, checkpoint and
//! process closure stay with the original owner on success and failure.
#![cfg_attr(not(target_os = "linux"), allow(dead_code))]

use crate::rovent_collect::{PublishedRoventRecord, PublishedRoventRecordV2};
use crate::{ArenaError, OriginalProcessOutput, ProcessLimits};
use rz_experiments::pals_data_v3::{
    GenerationRoleV3, GenerationSplitV3, generation_derive_tasks, generation_digest,
    generation_identifier, generation_sha,
};
use rz_experiments::rovent_corpus::{RoventCorpusIndex, RoventCorpusManifest, RoventCorpusOrigin};
use rz_experiments::rovent_corpus_v2::{
    RoventCorpusIndexV2, RoventCorpusManifestV2, RoventTrainingRecordV2,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::File,
    path::PathBuf,
    sync::{Arc, atomic::AtomicBool},
    time::Instant,
};

#[cfg(target_os = "linux")]
mod linux;
#[cfg(target_os = "linux")]
mod numerical_child;
#[cfg(target_os = "linux")]
pub mod recovery;
#[cfg(target_os = "linux")]
mod resources;
#[cfg(target_os = "linux")]
pub mod utility;
#[cfg(target_os = "linux")]
pub use linux::compiled_rovent_trainer_source_sha256;
#[cfg(target_os = "linux")]
pub use resources::{
    RoventNativeProcessEntry, RoventNativeProcessSnapshot, RoventNativeTreeObservation,
    RoventTrainingResourceFailure, RoventTrainingResourcePolicy, RoventTrainingResources,
    RoventTrainingTotalParentScopeObservation,
};

pub const ROVENT_TRAINING_REGISTRATION: &str = "rz-rovent-original-training-registration/1";
pub const ROVENT_TRAINER_STARTUP: &str = "rz-rovent-original-trainer-startup/1";
pub const ROVENT_TRAINER_RPC: &str = "rz-rovent-original-trainer-rpc/1";
pub const ROVENT_TRAINER_ACK: &str = "rz-rovent-original-trainer-ack/1";
pub const ROVENT_TRAINER_RESULT: &str = "rz-rovent-original-trainer-result/1";
pub const ROVENT_ORIGINAL_UPDATE: &str = "rz-rovent-original-learner-completion/1";
pub const ROVENT_ORIGINAL_CHECKPOINT: &str = "rz-rovent-original-checkpoint-retirement/1";
pub const MAX_TRAINER_RPC_BYTES: usize = 64 * 1024;
pub const MAX_TRAINER_ROWS: usize = 262_144;
pub const MAX_TRAINER_OUTPUT_BYTES: u64 = 18 << 30;
const HEADS: [&str; 5] = ["policy", "wdl", "divergence", "task", "nnuec_wdl"];
const ROLES: [&str; 3] = ["proposer", "critic", "validator"];
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
    ArenaError::Integrity(format!("Rovent trainer: {}", reason.into()))
}
fn check(ok: bool, reason: &str) -> Result<(), ArenaError> {
    if ok { Ok(()) } else { Err(invalid(reason)) }
}
fn role_name(role: GenerationRoleV3) -> &'static str {
    match role {
        GenerationRoleV3::Proposer => "proposer",
        GenerationRoleV3::Critic => "critic",
        GenerationRoleV3::Validator => "validator",
    }
}
fn sha(value: &Value) -> Result<&str, ArenaError> {
    value
        .as_str()
        .filter(|s| generation_sha(s))
        .ok_or_else(|| invalid("missing exact SHA256"))
}
fn count(value: &Value) -> Result<u64, ArenaError> {
    value
        .as_u64()
        .ok_or_else(|| invalid("nonnegative exact integer required"))
}
fn fields(value: &Value, expected: &[&str]) -> Result<(), ArenaError> {
    let object = value
        .as_object()
        .ok_or_else(|| invalid("strict object required"))?;
    check(
        object.len() == expected.len() && expected.iter().all(|k| object.contains_key(*k)),
        "unknown/missing object fields",
    )
}
fn digest_value(domain: &str, value: &impl Serialize) -> Result<String, ArenaError> {
    generation_digest(domain, value).map_err(|e| invalid(e.to_string()))
}
fn validate_cost(value: &Value) -> Result<(), ArenaError> {
    fields(value, &COST_FIELDS)?;
    for name in COST_FIELDS {
        count(&value[name])?;
    }
    Ok(())
}

/// Small immutable authority derived from the producer's original private row.
/// It retains no tensor, board clone or generated checkpoint payload.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RoventPcRecordFamily {
    LegacyRecordV1,
    SubmittedInputV1,
}
impl RoventPcRecordFamily {
    pub fn record_domain(self) -> &'static str {
        match self {
            Self::LegacyRecordV1 => rz_experiments::rovent_corpus::RECORD_DOMAIN,
            Self::SubmittedInputV1 => {
                rz_experiments::rovent_submitted_input::ROVENT_TRAINING_RECORD_V2_DOMAIN
            }
        }
    }
    pub fn input_domain(self) -> &'static str {
        match self {
            Self::LegacyRecordV1 => rz_experiments::pals_data_v3::GENERATION_INPUT_DOMAIN,
            Self::SubmittedInputV1 => {
                rz_experiments::rovent_submitted_input::ROVENT_SUBMITTED_INPUT_DOMAIN
            }
        }
    }
}

#[derive(Clone, Debug)]
struct NativeRowAuthority {
    family: RoventPcRecordFamily,
    /// Bounded source descriptor only. Full submitted bytes/tensor remain in
    /// the actual PublishedRecord or the historical held chunk, never here.
    input_descriptor: Value,
    origin: RoventCorpusOrigin,
    role: String,
    retirement_sha256: String,
    head_raw_sha256: BTreeMap<String, String>,
    valid_heads: BTreeMap<String, u64>,
}
impl NativeRowAuthority {
    fn validate_source_descriptor(&self) -> Result<(), ArenaError> {
        let mut expected = vec![
            "schema",
            "source_domain",
            "input_domain",
            "native_raw_domain",
            "input_sha256",
        ];
        if self.family == RoventPcRecordFamily::SubmittedInputV1 {
            expected.push("attack_mapping");
        }
        fields(&self.input_descriptor, &expected)?;
        check(
            self.input_descriptor["schema"] == "rz-rovent-training-input-source-descriptor/1"
                && self.input_descriptor["source_domain"] == self.family.record_domain()
                && self.input_descriptor["input_domain"] == self.family.input_domain()
                && self.input_descriptor["input_sha256"] == self.origin.input_sha256,
            "opaque sample changed its original record/input family descriptor",
        )?;
        match self.family {
            RoventPcRecordFamily::LegacyRecordV1 => check(
                matches!(
                    self.input_descriptor["native_raw_domain"].as_str(),
                    Some(
                        rz_experiments::rovent_corpus::NATIVE_RAW_DOMAIN
                            | rz_experiments::rovent_corpus::NATIVE_RAW_RULES_V2_DOMAIN
                    )
                ),
                "legacy source descriptor requires its original raw domain",
            ),
            RoventPcRecordFamily::SubmittedInputV1 => check(
                self.input_descriptor["native_raw_domain"]
                    == rz_experiments::rovent_submitted_input::ROVENT_NATIVE_RAW_V3_DOMAIN
                    && self.input_descriptor["attack_mapping"]
                        .as_array()
                        .is_some_and(|rows| rows.len() <= 128),
                "submitted source descriptor requires original bounded raw3 attack slots",
            ),
        }
    }
}

fn native_row_authority_legacy(
    row: &rz_experiments::rovent_corpus::RoventTrainingRecord,
) -> Result<NativeRowAuthority, ArenaError> {
    let mut valid_heads = BTreeMap::from(HEADS.map(|name| (name.to_owned(), 0)));
    if row.role == GenerationRoleV3::Validator {
        if let Some(task) = &row.targets.task {
            valid_heads.insert(
                "task".into(),
                generation_derive_tasks(&row.input, &task.outcomes)
                    .map_err(|error| invalid(error.to_string()))?
                    .pairs
                    .len() as u64,
            );
        }
    } else {
        valid_heads.insert("policy".into(), u64::from(row.targets.policy.is_some()));
        valid_heads.insert("wdl".into(), u64::from(row.targets.wdl.is_some()));
        valid_heads.insert(
            "divergence".into(),
            u64::from(row.role == GenerationRoleV3::Critic && row.targets.divergence.is_some()),
        );
        valid_heads.insert("nnuec_wdl".into(), u64::from(row.nnuec_wdl.is_some()));
    }
    let authority = NativeRowAuthority {
        family: RoventPcRecordFamily::LegacyRecordV1,
        input_descriptor: json!({"schema":"rz-rovent-training-input-source-descriptor/1",
            "source_domain":rz_experiments::rovent_corpus::RECORD_DOMAIN,
            "input_domain":rz_experiments::pals_data_v3::GENERATION_INPUT_DOMAIN,
            "native_raw_domain":row.native.raw_submission["schema"],"input_sha256":row.input_sha256}),
        origin: RoventCorpusOrigin::from_record(row),
        role: role_name(row.role).into(),
        retirement_sha256: row.native.raw_retirement_sha256.clone(),
        head_raw_sha256: row
            .head_sources
            .iter()
            .map(|(name, source)| (name.clone(), source.raw_receipt_sha256.clone()))
            .collect(),
        valid_heads,
    };
    authority.validate_source_descriptor()?;
    Ok(authority)
}

fn native_row_authority_v2(
    row: &RoventTrainingRecordV2,
    manifest: &RoventCorpusManifestV2,
) -> Result<NativeRowAuthority, ArenaError> {
    row.validate(manifest)
        .map_err(|error| invalid(error.to_string()))?;
    check(
        row.role != GenerationRoleV3::Validator,
        "submitted-input family is V-free",
    )?;
    let mut valid_heads = BTreeMap::from(HEADS.map(|name| (name.to_owned(), 0)));
    valid_heads.insert("policy".into(), u64::from(row.targets.policy.is_some()));
    valid_heads.insert("wdl".into(), u64::from(row.targets.wdl.is_some()));
    valid_heads.insert(
        "divergence".into(),
        u64::from(row.role == GenerationRoleV3::Critic && row.targets.divergence.is_some()),
    );
    let authority = NativeRowAuthority {
        family: RoventPcRecordFamily::SubmittedInputV1,
        input_descriptor: json!({"schema":"rz-rovent-training-input-source-descriptor/1",
            "source_domain":rz_experiments::rovent_submitted_input::ROVENT_TRAINING_RECORD_V2_DOMAIN,
            "input_domain":rz_experiments::rovent_submitted_input::ROVENT_SUBMITTED_INPUT_DOMAIN,
            "native_raw_domain":row.native.raw_submission["schema"],"input_sha256":row.input_sha256,"attack_mapping":row.input.attack_mapping}),
        origin: RoventCorpusOrigin {
            record_sha256: row.record_sha256.clone(),
            input_sha256: row.input_sha256.clone(),
            origin_sha256: row.origin_sha256.clone(),
            split: row.split,
            game_id: row.input.game_id.clone(),
            opening_id: row.input.opening_id.clone(),
            line_genealogy_id: row.input.line_genealogy_id.clone(),
            transposition_sha256: row.input.transposition_sha256.clone(),
            actor_id: row.lineage.actor_id.clone(),
            teacher_actor_id: row.lineage.teacher_actor_id.clone(),
            ancestor_origin_sha256: row.lineage.ancestor_origin_sha256.clone(),
        },
        role: role_name(row.role).into(),
        retirement_sha256: row.native.raw_retirement_sha256.clone(),
        head_raw_sha256: row
            .head_sources
            .iter()
            .map(|(name, source)| (name.clone(), source.raw_receipt_sha256.clone()))
            .collect(),
        valid_heads,
    };
    authority.validate_source_descriptor()?;
    Ok(authority)
}

/// Original producer custody. No Deserialize, wire constructor or boolean
/// admission exists. Recovering corpus files cannot reconstruct this type.
pub struct OriginalRoventCorpusCustody {
    manifest: RoventCorpusManifest,
    submitted_manifest: Option<RoventCorpusManifestV2>,
    manifest_sha256: String,
    rows: BTreeMap<String, NativeRowAuthority>,
    ancestors: Vec<Arc<OriginalRoventCorpusCustody>>,
}
impl OriginalRoventCorpusCustody {
    /// Creates an empty sink only. It has no training rows until real published
    /// capabilities are retained, and a launch must match every indexed row.
    pub fn new(
        manifest: RoventCorpusManifest,
        ancestors: Vec<Arc<Self>>,
    ) -> Result<Self, ArenaError> {
        manifest.validate().map_err(|e| invalid(e.to_string()))?;
        check(ancestors.len() <= 256, "ancestor custody capacity")?;
        let manifest_sha256 = manifest.sha256().map_err(|e| invalid(e.to_string()))?;
        let mut seen = BTreeSet::new();
        for ancestor in &ancestors {
            check(
                seen.insert(&ancestor.manifest_sha256)
                    && ancestor.manifest.recipe_sha256 == manifest.recipe_sha256
                    && ancestor.manifest.student == manifest.student
                    && ancestor.manifest.learner == manifest.learner
                    && ancestor.manifest.opening_splits == manifest.opening_splits,
                "duplicate or foreign original ancestor corpus",
            )?;
        }
        Ok(Self {
            manifest,
            submitted_manifest: None,
            manifest_sha256,
            rows: BTreeMap::new(),
            ancestors,
        })
    }
    pub fn retain_published(&mut self, original: &PublishedRoventRecord) -> Result<(), ArenaError> {
        self.retain_published_explicit(original, false)
    }
    /// An empty sink for the distinct submitted-input/1 corpus family. Only an
    /// actual PublishedRecordV2 can add rows; old /1 launch parsers stay closed.
    pub fn new_submitted_v3(
        manifest: RoventCorpusManifestV2,
        ancestors: Vec<Arc<Self>>,
    ) -> Result<Self, ArenaError> {
        manifest
            .validate()
            .map_err(|error| invalid(error.to_string()))?;
        check(
            ancestors.len() <= 256,
            "submitted ancestor custody capacity",
        )?;
        let manifest_sha256 = manifest
            .sha256()
            .map_err(|error| invalid(error.to_string()))?;
        let mut seen = BTreeSet::new();
        for ancestor in &ancestors {
            check(
                seen.insert(&ancestor.manifest_sha256)
                    && ancestor.manifest.recipe_sha256 == manifest.recipe_sha256
                    && ancestor.manifest.student == manifest.student
                    && ancestor.manifest.learner == manifest.learner
                    && ancestor.manifest.opening_splits == manifest.opening_splits,
                "duplicate or foreign original submitted ancestor",
            )?;
        }
        Ok(Self {
            manifest: manifest.common_registration_projection(),
            submitted_manifest: Some(manifest),
            manifest_sha256,
            rows: BTreeMap::new(),
            ancestors,
        })
    }
    pub fn retain_published_submitted_v3(
        &mut self,
        original: &PublishedRoventRecordV2,
    ) -> Result<(), ArenaError> {
        let manifest = self.submitted_manifest.as_ref().ok_or_else(|| {
            invalid("submitted published row requires its explicit manifest2 custody")
        })?;
        let row = original.record();
        let authority = native_row_authority_v2(row, manifest)?;
        check(
            original.manifest_sha256() == self.manifest_sha256
                && self.rows.len() < MAX_TRAINER_ROWS
                && !self.rows.contains_key(&row.origin_sha256),
            "submitted original row manifest/duplicate/capacity",
        )?;
        self.rows.insert(row.origin_sha256.clone(), authority);
        Ok(())
    }
    pub fn pc_record_family(&self) -> RoventPcRecordFamily {
        if self.submitted_manifest.is_some() {
            RoventPcRecordFamily::SubmittedInputV1
        } else {
            RoventPcRecordFamily::LegacyRecordV1
        }
    }
    pub fn join_index_submitted_v3(&self, index: &RoventCorpusIndexV2) -> Result<(), ArenaError> {
        let manifest = self
            .submitted_manifest
            .as_ref()
            .ok_or_else(|| invalid("submitted index requires actual manifest2 custody"))?;
        index
            .validate(manifest)
            .map_err(|error| invalid(error.to_string()))?;
        check(
            !self.rows.is_empty()
                && index.manifest_sha256 == self.manifest_sha256
                && index.origins.len() == self.rows.len()
                && index.origins.iter().all(|origin| {
                    self.rows.get(&origin.origin_sha256).is_some_and(|row| {
                        row.family == RoventPcRecordFamily::SubmittedInputV1
                            && row.origin == *origin
                    })
                }),
            "submitted index must cover its complete actual published2 custody",
        )
    }
    /// New exact Rules-replay format is selected explicitly. The original
    /// /1 intake above is preserved and never automatically retags old rows.
    pub fn retain_published_rules_replay_v2(
        &mut self,
        original: &PublishedRoventRecord,
    ) -> Result<(), ArenaError> {
        self.retain_published_explicit(original, true)
    }
    fn retain_published_explicit(
        &mut self,
        original: &PublishedRoventRecord,
        rules_replay_v2: bool,
    ) -> Result<(), ArenaError> {
        check(
            self.submitted_manifest.is_none(),
            "legacy published row cannot enter submitted manifest2 custody",
        )?;
        let row = original.record();
        (if rules_replay_v2 {
            row.validate_rules_replay_v2(&self.manifest)
        } else {
            row.validate(&self.manifest)
        })
        .map_err(|e| invalid(e.to_string()))?;
        check(
            original.manifest_sha256() == self.manifest_sha256
                && self.rows.len() < MAX_TRAINER_ROWS
                && !self.rows.contains_key(&row.origin_sha256),
            "retained original row manifest/duplicate/capacity",
        )?;
        let authority = native_row_authority_legacy(row)?;
        self.rows.insert(row.origin_sha256.clone(), authority);
        Ok(())
    }
    pub fn manifest_sha256(&self) -> &str {
        &self.manifest_sha256
    }
    pub fn retained_rows(&self) -> usize {
        self.rows.len()
    }
    fn join_index(&self, index: &RoventCorpusIndex) -> Result<(), ArenaError> {
        check(
            self.submitted_manifest.is_none(),
            "original /1 launch cannot admit submitted manifest2/index2 custody",
        )?;
        index.validate().map_err(|e| invalid(e.to_string()))?;
        check(
            !self.rows.is_empty()
                && index.manifest_sha256 == self.manifest_sha256
                && index.origins.len() == self.rows.len(),
            "complete original corpus/index custody required",
        )?;
        for origin in &index.origins {
            check(
                self.rows
                    .get(&origin.origin_sha256)
                    .is_some_and(|r| r.origin == *origin),
                "wire row was not issued by this original producer",
            )?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoventTrainingFilePin {
    pub path: PathBuf,
    pub sha256: String,
    pub bytes: u64,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoventTrainingCorpusFiles {
    pub manifest: RoventTrainingFilePin,
    pub index: RoventTrainingFilePin,
    pub root: PathBuf,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoventTrainingRegistration {
    pub schema: String,
    pub execution_id: String,
    pub action: String,
    pub export_request: Option<RoventTrainingFilePin>,
    pub recipe: RoventTrainingFilePin,
    pub membership: RoventTrainingFilePin,
    pub window_spec: RoventTrainingFilePin,
    pub trainer_source_sha256: String,
    pub python_runtime_sha256: String,
    pub model_semantics_sha256: String,
    pub current_corpus: RoventTrainingCorpusFiles,
    pub ancestor_corpora: Vec<RoventTrainingCorpusFiles>,
    pub checkpoint: Option<RoventTrainingFilePin>,
    pub pc_initial: Option<RoventTrainingFilePin>,
    pub nnuec_initial: Option<RoventTrainingFilePin>,
    pub rules_profile: RoventTrainingFilePin,
    pub checkpoint_mode: String,
    pub expected_initial_completed: BTreeMap<String, u64>,
    pub expected_ancestor_completed: BTreeMap<String, u64>,
    pub resources_policy: Value,
    /// Strict profiles require observed parent-and-child aggregate cgroup caps.
    /// False permits a conditional validation run, never a total-resource claim.
    pub require_total_quota: bool,
    pub output_root: PathBuf,
    pub control_root: PathBuf,
    pub checkpoint_directory: PathBuf,
    pub receipt_path: PathBuf,
    pub max_rpc_messages: u64,
    pub max_rpc_bytes: u64,
    pub max_read_bytes: u64,
    pub max_output_bytes: u64,
}

/// Original files and process-local deadlines are moved into the owner. This
/// cannot be created by deserializing a path or a saved launch receipt.
pub struct RoventTrainingLaunch {
    pub interpreter: File,
    pub interpreter_pin: RoventTrainingFilePin,
    pub interpreter_version: String,
    pub registration: File,
    pub registration_pin: RoventTrainingFilePin,
    pub dependency_directories: Vec<PathBuf>,
    pub runtime_module_pins: Vec<RoventTrainingFilePin>,
    pub started: Instant,
    pub execution_deadline: Instant,
    pub whole_deadline: Instant,
    pub limits: ProcessLimits,
    #[cfg(target_os = "linux")]
    pub resources: Arc<RoventTrainingResources>,
}

/// Successful owner, not a serializer or a constructor from a checkpoint pin.
pub struct VerifiedRoventTrainingWindow {
    registration: RoventTrainingRegistration,
    summary: Value,
    journal: TrainerLedger,
    process: OriginalProcessOutput,
    _launch: RoventTrainingLaunch,
    _corpus: Arc<OriginalRoventCorpusCustody>,
    previous: Option<Arc<Self>>,
    #[cfg(target_os = "linux")]
    held: linux::LinuxHeld,
}
impl VerifiedRoventTrainingWindow {
    pub fn summary(&self) -> &Value {
        &self.summary
    }
    pub fn process(&self) -> &OriginalProcessOutput {
        &self.process
    }
    pub fn committed_updates(&self) -> &BTreeMap<String, u64> {
        &self.journal.completed
    }
    pub fn execution_id(&self) -> &str {
        &self.registration.execution_id
    }
    pub fn action(&self) -> &str {
        &self.registration.action
    }
    #[cfg(target_os = "linux")]
    pub fn original_checkpoint_files(
        &self,
    ) -> impl Iterator<Item = (&RoventTrainingFilePin, &File)> {
        self.held.checkpoint_files()
    }
    #[cfg(target_os = "linux")]
    pub fn original_export_files(&self) -> impl Iterator<Item = (&RoventTrainingFilePin, &File)> {
        self.held.export_files()
    }
    pub fn previous(&self) -> Option<&Arc<Self>> {
        self.previous.as_ref()
    }
    pub fn durable_checkpoint_available(&self) -> bool {
        self.journal.ended
            && self
                .journal
                .checkpoint_states
                .iter()
                .any(|state| state["completed_updates"] == json!(self.journal.completed))
    }
}

/// Failure preserves source/input/checkpoint/cgroup/IPC/process custody. It is
/// not retry authority, a refund or permission to release an unknown GPU run.
pub struct RoventTrainingFailure {
    pub error: ArenaError,
    pub process: Option<OriginalProcessOutput>,
    pub summary: Option<Value>,
    pub charged_reads: u64,
    /// When true, charged_reads is a conservative reserved upper bound.
    pub read_accounting_unknown: bool,
    pub rpc_journal: Value,
    pub launch: RoventTrainingLaunch,
    pub corpus: Arc<OriginalRoventCorpusCustody>,
    pub previous: Option<Arc<VerifiedRoventTrainingWindow>>,
    #[cfg(target_os = "linux")]
    _retained_held: Option<linux::LinuxHeld>,
    #[cfg(target_os = "linux")]
    _retained_pending_broker: Option<linux::PendingBroker>,
    #[cfg(target_os = "linux")]
    _retained_shared_held: Option<Arc<std::sync::Mutex<Option<linux::LinuxHeld>>>>,
}
impl std::fmt::Debug for RoventTrainingFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RoventTrainingFailure")
            .field("error", &self.error)
            .field("process", &self.process)
            .finish_non_exhaustive()
    }
}

pub fn run_rovent_training_original(
    launch: RoventTrainingLaunch,
    corpus: Arc<OriginalRoventCorpusCustody>,
    previous: Option<Arc<VerifiedRoventTrainingWindow>>,
    cancel: Arc<AtomicBool>,
) -> Result<VerifiedRoventTrainingWindow, Box<RoventTrainingFailure>> {
    #[cfg(target_os = "linux")]
    {
        linux::run(launch, corpus, previous, cancel)
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = cancel;
        Err(Box::new(RoventTrainingFailure {
            error: invalid("original Rovent learner requires Linux"),
            process: None,
            summary: None,
            charged_reads: 0,
            read_accounting_unknown: false,
            rpc_journal: json!({}),
            launch,
            corpus,
            previous,
        }))
    }
}

#[derive(Clone, Debug)]
struct TrainerLedger {
    window: Value,
    recipe: Value,
    membership_sha256: String,
    profile_sha256: String,
    stage_limits: BTreeMap<String, u64>,
    corpus_sha256: String,
    rows: BTreeMap<String, NativeRowAuthority>,
    initial: BTreeMap<String, u64>,
    completed: BTreeMap<String, u64>,
    ancestors: BTreeMap<String, u64>,
    pending: Vec<String>,
    consumed_inputs: BTreeSet<String>,
    consumed_origins: BTreeSet<String>,
    updates: Vec<Value>,
    checkpoints: Vec<Value>,
    checkpoint_states: Vec<Value>,
    checkpoint_receipts: Vec<Value>,
    end_cost: Option<Value>,
    last_cost: Option<Value>,
    begun: bool,
    ended: bool,
    next_sequence: u64,
    bytes_spent: u64,
    samples_admitted: u64,
    groups_started: u64,
    masked_groups: u64,
    checkpoint_checked: bool,
    export_admitted: bool,
    export_admission: Option<Value>,
}
impl TrainerLedger {
    fn new(
        window: Value,
        recipe: Value,
        membership_sha256: String,
        corpus_sha256: String,
        rows: BTreeMap<String, NativeRowAuthority>,
        initial: BTreeMap<String, u64>,
        ancestors: BTreeMap<String, u64>,
    ) -> Result<Self, ArenaError> {
        check(
            ROLES
                .iter()
                .all(|role| initial.contains_key(*role) && ancestors.contains_key(*role))
                && initial.len() == 3
                && ancestors.len() == 3,
            "exact P/C/V update ownership",
        )?;
        let stage = recipe["stages"]
            .as_array()
            .and_then(|stages| stages.iter().find(|s| s["stage_id"] == window["stage_id"]))
            .ok_or_else(|| invalid("registered recipe stage missing"))?;
        let stage_limits: BTreeMap<String, u64> =
            serde_json::from_value(stage["role_updates"].clone())
                .map_err(|e| invalid(e.to_string()))?;
        check(
            stage_limits.len() == 3
                && ROLES.iter().all(|r| {
                    stage_limits
                        .get(*r)
                        .is_some_and(|limit| initial[*r] <= *limit)
                }),
            "initial update counts exceed frozen stage declaration",
        )?;
        let profile_sha256 = training_profile_digest(&window)?;
        Ok(Self {
            window,
            recipe,
            membership_sha256,
            profile_sha256,
            stage_limits,
            corpus_sha256,
            rows,
            completed: initial.clone(),
            initial,
            ancestors,
            pending: Vec::new(),
            consumed_inputs: BTreeSet::new(),
            consumed_origins: BTreeSet::new(),
            updates: Vec::new(),
            checkpoints: Vec::new(),
            checkpoint_states: Vec::new(),
            checkpoint_receipts: Vec::new(),
            end_cost: None,
            last_cost: None,
            begun: false,
            ended: false,
            next_sequence: 1,
            bytes_spent: 0,
            samples_admitted: 0,
            groups_started: 0,
            masked_groups: 0,
            checkpoint_checked: false,
            export_admitted: false,
            export_admission: None,
        })
    }
    fn begin(&mut self, p: &Value) -> Result<Value, ArenaError> {
        fields(
            p,
            &[
                "schema",
                "recipe_sha256",
                "stage_id",
                "training_profile_sha256",
                "corpus_sha256",
                "source_sha256",
                "runtime_sha256",
                "completed_updates",
                "ancestor_updates",
                "frozen_pc_sha256",
            ],
        )?;
        check(
            !self.begun
                && !self.ended
                && p["schema"] == "rz-rovent-trainer-begin/1"
                && p["recipe_sha256"] == self.window["recipe_sha256"]
                && p["stage_id"] == self.window["stage_id"]
                && p["source_sha256"] == self.window["source_sha256"]
                && p["runtime_sha256"] == self.window["runtime_sha256"]
                && p["corpus_sha256"] == self.corpus_sha256
                && p["frozen_pc_sha256"] == self.window["frozen_pc_sha256"]
                && p["completed_updates"] == json!(self.initial)
                && p["ancestor_updates"] == json!(self.ancestors),
            "original window/source/corpus/count begin mismatch",
        )?;
        check(
            p["training_profile_sha256"] == self.profile_sha256,
            "training profile differs from original fixed recipe",
        )?;
        self.begun = true;
        Ok(json!({}))
    }
    fn consume(&mut self, p: &Value) -> Result<Value, ArenaError> {
        self.consume_with_schema(p, "rz-rovent-trainer-consume/1")
    }
    fn consume_with_schema(&mut self, p: &Value, schema: &str) -> Result<Value, ArenaError> {
        fields(
            p,
            &[
                "schema",
                "input_sha256",
                "origin_sha256",
                "record_sha256",
                "native_retirement_sha256",
                "head_raw_sha256",
                "role",
                "split",
            ],
        )?;
        check(
            self.begun && !self.ended && p["schema"] == schema && self.pending.len() < 8,
            "closed or overfull original sample admission",
        )?;
        let origin = sha(&p["origin_sha256"])?;
        let row = self
            .rows
            .get(origin)
            .ok_or_else(|| invalid("sample has no original native retirement owner"))?;
        row.validate_source_descriptor()?;
        check(
            row.origin.split == GenerationSplitV3::Train
                && p["split"] == "train"
                && p["role"] == row.role
                && p["input_sha256"] == row.origin.input_sha256
                && p["record_sha256"] == row.origin.record_sha256
                && p["native_retirement_sha256"] == row.retirement_sha256
                && p["head_raw_sha256"] == json!(row.head_raw_sha256),
            "sample changed original train/role/input/retirement/head provenance",
        )?;
        if let Some(first) = self.pending.first() {
            check(
                self.rows[first].role == row.role,
                "accumulation cannot mix roles",
            )?;
        }
        check(
            self.samples_admitted < count(&self.window["max_samples"])?
                && (!self.pending.is_empty()
                    || self.groups_started < count(&self.window["max_groups"])?),
            "original group/sample budget",
        )?;
        if self.pending.is_empty() {
            self.groups_started += 1;
        }
        self.samples_admitted += 1;
        self.pending.push(origin.into());
        if self.pending.len() == 8
            && !self.pending.iter().any(|id| {
                self.rows[id]
                    .valid_heads
                    .iter()
                    .any(|(head, n)| *n > 0 && self.head_weight(head) > 0.0)
            })
        {
            self.pending.clear();
            self.masked_groups += 1;
        }
        Ok(json!({}))
    }
    fn update(&mut self, p: &Value) -> Result<Value, ArenaError> {
        self.update_with_schema(
            p,
            "rz-rovent-trainer-update/1",
            "rz-rovent-original-learner-update/1",
        )
    }
    fn update_with_schema(
        &mut self,
        p: &Value,
        schema: &str,
        event_schema: &str,
    ) -> Result<Value, ArenaError> {
        fields(
            p,
            &[
                "schema",
                "event",
                "model_state_sha256",
                "optimizer_state_sha256",
                "scheduler_state_sha256",
                "rng_state_sha256",
                "sampler_state_sha256",
                "cost",
                "pending_microbatches",
            ],
        )?;
        let event = &p["event"];
        let mut event_fields = vec![
            "schema",
            "window_id",
            "stage_id",
            "role",
            "completed_role_updates",
            "profile_sha256",
            "corpus_sha256",
            "phase",
            "sample_origins",
            "sample_inputs",
            "head_counts",
            "actual_model_state_sha256",
        ];
        if schema == "rz-rovent-trainer-update/2"
            && self.window["nnuec_r_auxiliary"].is_object()
            && event["role"] != "validator"
        {
            event_fields.push("r_auxiliary");
        }
        fields(event, &event_fields)?;
        check(
            self.begun
                && !self.ended
                && self.pending.len() == 8
                && p["schema"] == schema
                && p["pending_microbatches"] == 0
                && event["schema"] == event_schema
                && event["window_id"] == self.window["window_id"]
                && event["stage_id"] == self.window["stage_id"]
                && event["phase"] == self.window["phase"]
                && event["profile_sha256"] == self.profile_sha256
                && event["corpus_sha256"] == self.corpus_sha256
                && event["actual_model_state_sha256"] == p["model_state_sha256"],
            "original update window/phase/state/boundary mismatch",
        )?;
        for name in [
            "model_state_sha256",
            "optimizer_state_sha256",
            "scheduler_state_sha256",
            "rng_state_sha256",
            "sampler_state_sha256",
        ] {
            sha(&p[name])?;
        }
        self.observe_cost(&p["cost"])?;
        let role = event["role"]
            .as_str()
            .filter(|r| ROLES.contains(r))
            .ok_or_else(|| invalid("update role"))?;
        let next = self.completed[role]
            .checked_add(1)
            .ok_or_else(|| invalid("update count overflow"))?;
        check(
            count(&event["completed_role_updates"])? == next
                && next <= self.stage_limits[role]
                && self.pending.iter().all(|id| self.rows[id].role == role),
            "update delta/role/stage cap mismatch",
        )?;
        check(
            (self.window["phase"] == "v") == (role == "validator"),
            "P/C freeze or role phase violation",
        )?;
        let inputs = self
            .pending
            .iter()
            .map(|id| self.rows[id].origin.input_sha256.clone())
            .collect::<Vec<_>>();
        check(
            event["sample_origins"] == json!(self.pending)
                && event["sample_inputs"] == json!(inputs),
            "actual update did not consume its eight admitted rows",
        )?;
        let mut heads = BTreeMap::from(HEADS.map(|h| (h.to_string(), 0_u64)));
        for origin in &self.pending {
            for (name, n) in &self.rows[origin].valid_heads {
                let old = heads[name];
                heads.insert(
                    name.clone(),
                    old.checked_add(*n)
                        .ok_or_else(|| invalid("head count overflow"))?,
                );
            }
        }
        check(
            event["head_counts"] == json!(heads)
                && heads
                    .iter()
                    .any(|(h, n)| *n > 0 && self.head_weight(h) > 0.0),
            "all-masked or fabricated supervised head update",
        )?;
        let used = self.updates.len() as u64;
        check(
            used < count(&self.window["max_updates"])? && self.updates.len() < 65_536,
            "original update window quota",
        )?;
        self.completed.insert(role.into(), next);
        for origin in self.pending.drain(..) {
            self.consumed_inputs
                .insert(self.rows[&origin].origin.input_sha256.clone());
            self.consumed_origins.insert(origin);
        }
        self.updates.push(p.clone());
        Ok(json!({}))
    }
    fn summary(&self) -> Value {
        json!({"begun":self.begun,"ended":self.ended,"completed_updates":self.completed,
        "ancestor_updates":self.ancestors,"committed_updates":self.updates.len(),"pending_samples":self.pending.len(),
        "retained_input_sha256":self.consumed_inputs,"retained_origin_sha256":self.consumed_origins,
        "checkpoints":self.checkpoints,"next_sequence":self.next_sequence,"rpc_bytes":self.bytes_spent,
        "samples_admitted":self.samples_admitted,"groups_started":self.groups_started,"masked_groups":self.masked_groups})
    }
    fn head_weight(&self, head: &str) -> f64 {
        if head == "nnuec_wdl" {
            // Enabled R auxiliary retains P/C NNUEC evidence counts but
            // supervises that private optimizer from the separate R stream.
            // The absent/null path preserves the original P/C behavior.
            if self.window["schema"] == "rz-rovent-train-window/2"
                && self.window["nnuec_r_auxiliary"].is_object()
            {
                0.0
            } else {
                self.window["nnuec_loss"]["weight"].as_f64().unwrap_or(0.0)
            }
        } else {
            self.window["loss_weights"][head].as_f64().unwrap_or(0.0)
        }
    }
    fn observe_cost(&mut self, value: &Value) -> Result<(), ArenaError> {
        validate_cost(value)?;
        if let Some(previous) = &self.last_cost {
            for name in COST_FIELDS {
                check(
                    count(&value[name])? >= count(&previous[name])?,
                    "actual training cost cannot regress/refund work",
                )?;
            }
        }
        self.last_cost = Some(value.clone());
        Ok(())
    }
}

fn training_profile_digest(window: &Value) -> Result<String, ArenaError> {
    let renewal = [
        "window_id",
        "max_updates",
        "max_groups",
        "max_samples",
        "max_wall_ns",
        "max_output_bytes",
        "checkpoint_interval",
        "phase",
        "model_epoch",
        "frozen_pc_sha256",
        "nnuec_collection_model_sha256",
    ];
    let fixed = window
        .as_object()
        .ok_or_else(|| invalid("window object"))?
        .iter()
        .filter(|(k, _)| !renewal.contains(&k.as_str()))
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect::<BTreeMap<_, _>>();
    digest_value("rz-rovent-fixed-training-profile/1", &fixed)
}

#[cfg(test)]
mod tests;
