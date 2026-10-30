//! Finite own-data collection: checked Rules -> sealed input -> own analysis ->
//! later labels. No external teacher, optimizer, GPU or cloud launch lives here.
use crate::{ArenaError, canonical_sha256};
#[cfg(feature = "pals-collection")]
use rz_eval::pals_model::PALS_ENCODING_SCHEMA;
use rz_eval::pals_model::{PalsModelConfig, PalsModelInput};
use rz_experiments::{
    PalsConditionalValidity, PalsCounterexampleTarget, PalsDataRole, PalsDatasetAudit,
    PalsDatasetSplit, PalsFrozenInput, PalsFutureLabel, PalsGameEnd, PalsGameResult,
    PalsInputSnapshot, PalsInputSource, PalsLearningRecord, PalsOutcome, PalsOwnedSources,
    PalsPolicyTarget, PalsPublicRecord, PalsSplit, PalsTargetProvenance,
};
use rz_position::contracts::ContractPosition;
use rz_position::{BoardMove, Color, PlayStatus, Position, PositionLimits, TerminalReason};
use rz_search::cpu::{
    BOOTSTRAP_SCORE_VERSION, CPU_SEARCH_VERSION, CpuConfig, CpuEngine, CpuLimits, CpuReport,
    CpuScoreScope,
};
use rz_search::pals::engine::{
    LegalOrderRoleMock, PalsConfig, PalsEngine, PalsLimits, RecordKind, RoleQuery, RoleRecord,
};
use rz_search::pals::store::{Move16, StateId};
#[cfg(feature = "pals-collection")]
use rz_uci::pals_native::{
    NativeQueryKind, pals_native_source_digest, pals_rules_encoding_semantic_digest,
    prepare_role_input,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{File, Metadata, OpenOptions};
use std::hash::{Hash, Hasher};
use std::io::{Read, Write};
#[cfg(unix)]
use std::os::unix::fs::MetadataExt;
#[cfg(windows)]
use std::os::windows::fs::{MetadataExt, OpenOptionsExt};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

mod full_line;
#[cfg(feature = "pals-collection-onnx")]
mod native;
mod producer;
pub use full_line::{
    PalsCheckerValueNamespaceV2, PalsFullLineContextV2, PalsObservedCountV2,
    PalsTargetCoverageReceiptV2,
};
#[cfg(feature = "pals-collection-onnx")]
pub use native::{
    OwnPalsOnnxCollectionDriver, PalsCollectionGraphPin, PalsNativeCollectionRegistry,
    PalsNativeContinuationRegistration, PalsNativeRefinementRegistration,
};
pub use producer::{
    CheckedProducerOwner, PalsProducerCollectionConfig, RegisteredProducerHandle,
    pals_producer_registration_description,
};

pub const PALS_COLLECT_VERSION: &str = "rz-pals-own-collector/1";
const MAX_ROWS: usize = 65_536;
const RECEIPT_RESERVE: u64 = 256 * 1024;
const MAX_JSON_RECORD_BYTES: usize = 256 * 1024;
static NEXT_CAPTURE_OWNER: AtomicU64 = AtomicU64::new(1);

fn invalid(message: impl Into<String>) -> ArenaError {
    ArenaError::Invalid(message.into())
}
fn io(error: std::io::Error) -> ArenaError {
    ArenaError::Io(error.to_string())
}
fn rules(error: rz_position::PositionError) -> ArenaError {
    ArenaError::Contract(error.into())
}
fn ident(text: &str) -> bool {
    !text.is_empty()
        && text.len() <= 96
        && text
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PalsCollectionOpening {
    pub id: String,
    pub initial_fen: Option<String>,
    pub moves: Vec<String>,
    pub split: PalsSplit,
}
impl Default for PalsCollectionOpening {
    fn default() -> Self {
        Self {
            id: "startpos".into(),
            initial_fen: None,
            moves: Vec::new(),
            split: PalsSplit::Train,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PalsCollectionConfig {
    pub run_id: String,
    pub games: u32,
    pub max_plies: u32,
    pub cpu_depth: u16,
    pub nodes_per_job: u64,
    pub max_total_nodes: u64,
    pub max_wall_time_ms: u64,
    pub max_output_bytes: u64,
    pub seed: u64,
    /// Seeded legal plies before collection. These are actual opening moves,
    /// never virtual counterexample moves or unseen teacher choices.
    pub exploration_plies: u16,
    pub collect_conditional_repair: bool,
    pub openings: Vec<PalsCollectionOpening>,
}
impl Default for PalsCollectionConfig {
    fn default() -> Self {
        Self {
            run_id: "own-cpu-pilot".into(),
            games: 1,
            max_plies: 32,
            cpu_depth: 2,
            nodes_per_job: 2048,
            max_total_nodes: 196_608,
            max_wall_time_ms: 60_000,
            max_output_bytes: 8 * 1024 * 1024,
            seed: 1,
            exploration_plies: 2,
            collect_conditional_repair: true,
            openings: vec![PalsCollectionOpening::default()],
        }
    }
}
impl PalsCollectionConfig {
    pub fn validate(&self) -> Result<(), ArenaError> {
        if !ident(&self.run_id)
            || self.games == 0
            || self.games > 128
            || self.max_plies == 0
            || self.max_plies > 1024
            || self.cpu_depth == 0
            || self.cpu_depth > 16
            || self.nodes_per_job == 0
            || self.nodes_per_job > 10_000_000
            || self.max_total_nodes == 0
            || self.max_wall_time_ms == 0
            || self.max_wall_time_ms > 3_600_000
            || self.max_output_bytes <= RECEIPT_RESERVE * 2
            || self.max_output_bytes > 1024 * 1024 * 1024
            || self.openings.is_empty()
            || self.openings.len() > 128
            || self.exploration_plies > 32
            || u64::from(self.games) * (u64::from(self.max_plies) + 1) * 8 > MAX_ROWS as u64
        {
            return Err(invalid(
                "collector needs finite game/ply/depth/node/time/row/output limits",
            ));
        }
        let mut ids = BTreeSet::new();
        for opening in &self.openings {
            if !ident(&opening.id)
                || !ids.insert(&opening.id)
                || opening.moves.len() > 256
                || opening
                    .initial_fen
                    .as_ref()
                    .is_some_and(|f| f.len() > 512 || f.chars().any(char::is_control))
                || opening.moves.iter().any(|m| m.len() > 5)
            {
                return Err(invalid("invalid or duplicate collector opening"));
            }
        }
        if serde_json::to_vec(self)
            .map_err(|e| invalid(e.to_string()))?
            .len()
            > 64 * 1024
        {
            return Err(invalid("collector configuration byte limit"));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct PalsCollectionSourceDescription {
    pub mode: String,
    pub source: PalsInputSource,
    pub cpu_profile_sha256: String,
    pub implementation_sha256: String,
    pub encoding_sha256: String,
    pub encoder_source_sha256: String,
    pub configuration: serde_json::Value,
    /// For weightless CPU data this is an opaque encoding-only zero epoch.
    /// A real PALS provider must supply its actual registered epoch.
    pub model_epoch: [u8; 32],
    pub model_epoch_kind: String,
    pub frozen_epoch: u64,
    /// Actual native registration and execution configuration, absent for CPU/mock.
    /// This is evidence metadata, not part of the model configuration digest.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub native: Option<serde_json::Value>,
}
#[derive(Clone, Debug, Serialize)]
pub struct PalsCollectedLine {
    pub kind: String,
    pub moves: Vec<u16>,
    pub raw_value: Option<i32>,
    pub completed_depth: u16,
    pub raw_scope: String,
    pub white_perspective: bool,
}
#[derive(Clone, Debug)]
pub struct PalsCollectionDecision {
    pub best_move: Option<BoardMove>,
    pub pv: Vec<BoardMove>,
    pub nodes: u64,
    pub completed_depth: u16,
    pub completed_estimate: bool,
    pub raw: serde_json::Value,
    pub genealogy: Vec<PalsCollectedLine>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PalsNativeInputSidecar {
    pub version: String,
    pub input_sha256: String,
    pub encoding_sha256: String,
    pub encoder_source_sha256: String,
    pub model_epoch_kind: String,
    pub canonical_tensor_sha256: String,
    /// Preserve Rust serialization exactly; Python must not reserialize f32 to
    /// guess its byte identity.
    pub tensor_json: String,
    pub tensor_sha256: String,
    /// Exactly the selected tensor-record order, with immutable raw sources.
    pub record_sources: Vec<PalsPublicRecord>,
    /// Present only in sidecar/2; historical sidecar/1 bytes are unchanged.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_profile: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub encoding_profile: Option<String>,
    pub sha256: String,
}
impl PalsNativeInputSidecar {
    fn digest(&self) -> Result<String, ArenaError> {
        match (
            &self.model_profile,
            &self.encoding_profile,
            self.version.as_str(),
        ) {
            (None, None, "rz-pals-native-input-sidecar/1") => canonical_sha256(&(
                &self.version,
                &self.input_sha256,
                &self.encoding_sha256,
                &self.encoder_source_sha256,
                &self.model_epoch_kind,
                &self.canonical_tensor_sha256,
                &self.tensor_json,
                &self.tensor_sha256,
                &self.record_sources,
            )),
            (Some(profile), Some(encoding), "rz-pals-native-input-sidecar/2") => {
                canonical_sha256(&(
                    &self.version,
                    &self.input_sha256,
                    &self.encoding_sha256,
                    &self.encoder_source_sha256,
                    &self.model_epoch_kind,
                    &self.canonical_tensor_sha256,
                    &self.tensor_json,
                    &self.tensor_sha256,
                    &self.record_sources,
                    profile,
                    encoding,
                ))
            }
            _ => Err(invalid("sidecar version/profile fields differ")),
        }
    }
}

/// Model-backed PALS drivers may supply the same sealed-input boundary. Their
/// exact configuration/weights must already be enrolled in the independent
/// registry; a returned row cannot register itself. Only builtin CPU/mock
/// drivers are exposed by the executable example.
pub trait PalsCollectionDriver {
    fn description(&self) -> &PalsCollectionSourceDescription;
    /// Opt-in only from an actual checked constructor, not from a row or label.
    /// Legacy/custom/mock drivers do not acquire strict producer authority.
    fn checked_producer_owner(&mut self) -> Result<Option<CheckedProducerOwner<'_>>, ArenaError> {
        Ok(None)
    }
    fn set_registered_producer(
        &mut self,
        producer: Option<RegisteredProducerHandle>,
    ) -> Result<(), ArenaError> {
        if producer.is_some() && self.records_actual_native_calls() {
            return Err(invalid(
                "native driver cannot install strict producer observer",
            ));
        }
        Ok(())
    }
    fn new_game(&mut self);
    fn analyze(
        &mut self,
        position: &Position,
        input: &PalsFrozenInput,
        prepared: &PalsModelInput,
        limits: CpuLimits,
        cancel: &AtomicBool,
    ) -> Result<PalsCollectionDecision, ArenaError>;
    /// Real role models seal their exact prepared inputs through the native
    /// observer. The synthetic CPU/mock root encoder is never their producer.
    fn records_actual_native_calls(&self) -> bool {
        false
    }
    fn analyze_native(
        &mut self,
        _context: PalsNativeCaptureContext,
        _limits: CpuLimits,
        _cancel: &AtomicBool,
    ) -> Result<PalsCollectionDecision, ArenaError> {
        Err(invalid("driver has no actual native collection path"))
    }
    /// Drain even when analyze_native returned an error. Already sealed input
    /// and physically returned raw evidence must survive logical rejection.
    fn take_native_trace(&mut self) -> Result<PalsNativeTrace, ArenaError> {
        Ok(PalsNativeTrace::default())
    }
    fn finish_collection(
        &mut self,
        _until: Instant,
    ) -> Result<Option<serde_json::Value>, ArenaError> {
        Ok(None)
    }
    fn receipt_preserved(&mut self) {}
}

pub struct PalsNativeCaptureContext {
    pub position: Position,
    pub opening: PalsCollectionOpening,
    pub game: String,
    pub actual_moves: Vec<BoardMove>,
    pub sequence: u64,
    pub max_bytes: u64,
}
#[derive(Default)]
pub struct PalsNativeTrace {
    pub sequence: u64,
    /// Exclusive credit delegated from Output before dispatch; includes the
    /// worst-case physically completed raw output and close-stage evidence.
    pub reserved_bytes: u64,
    pub cpu_nodes: u64,
    pub cpu_jobs: u64,
    pub cpu_work_observation_incomplete: bool,
    pub rows: Vec<PalsNativeTraceRow>,
    pub inputs: Vec<(PalsFrozenInput, bool)>,
    pub failure: Option<String>,
}
pub struct PalsNativeTraceRow {
    pub artifact: &'static str,
    /// Exact bytes serialized and budget-admitted by the observer. Persisting
    /// these does not regenerate a deadline feature or canonical float form.
    pub json: Vec<u8>,
}

fn hex(value: [u8; 32]) -> String {
    value.iter().map(|b| format!("{b:02x}")).collect()
}
#[cfg(feature = "pals-collection")]
fn native_encoding_sha() -> Result<String, ArenaError> {
    let mut hash = Sha256::new();
    hash.update(PALS_ENCODING_SCHEMA);
    hash.update(pals_rules_encoding_semantic_digest());
    Ok(format!("{:x}", hash.finalize()))
}
#[cfg(feature = "pals-collection")]
fn source_encoding_sha(source: &PalsCollectionSourceDescription) -> Result<String, ArenaError> {
    let config = full_line::model_config(source)?;
    Ok(hex(
        rz_uci::pals_native::pals_fresh_encoding_semantic_digest_for_config(&config),
    ))
}
#[cfg(not(feature = "pals-collection"))]
fn source_encoding_sha(_: &PalsCollectionSourceDescription) -> Result<String, ArenaError> {
    native_encoding_sha()
}
#[cfg(not(feature = "pals-collection"))]
fn native_encoding_sha() -> Result<String, ArenaError> {
    Err(invalid(
        "actual own-data collection requires pals-collection feature and native Rust Rules encoder",
    ))
}

fn file_sha(path: &Path, max_bytes: u64) -> Result<String, ArenaError> {
    let metadata = std::fs::symlink_metadata(path).map_err(io)?;
    if !metadata.is_file() || metadata.file_type().is_symlink() || metadata.len() > max_bytes {
        return Err(invalid("source asset is not a bounded regular file"));
    }
    let mut file = File::open(path).map_err(io)?;
    hash_open_file(&mut file, metadata.len(), max_bytes)
}

fn hash_open_file(
    file: &mut File,
    expected_bytes: u64,
    max_bytes: u64,
) -> Result<String, ArenaError> {
    let mut hash = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    let mut total = 0_u64;
    loop {
        let read = file.read(&mut buffer).map_err(io)?;
        if read == 0 {
            break;
        }
        total = total
            .checked_add(read as u64)
            .filter(|n| *n <= max_bytes)
            .ok_or_else(|| invalid("source asset grew beyond limit"))?;
        hash.update(&buffer[..read]);
    }
    if total != expected_bytes {
        return Err(invalid("source asset changed during verification"));
    }
    Ok(format!("{:x}", hash.finalize()))
}

/// An independently hashed executable observation, owned through its open FD.
/// Metadata checks detect observed changes; this is not a kernel file seal.
struct VerifiedExecutableIdentity {
    path: PathBuf,
    file: File,
    metadata: Metadata,
    sha256: String,
}
impl VerifiedExecutableIdentity {
    fn current() -> Result<Self, ArenaError> {
        Self::open(&std::env::current_exe().map_err(io)?, 256 * 1024 * 1024)
    }

    fn open(path: &Path, max_bytes: u64) -> Result<Self, ArenaError> {
        let named = std::fs::symlink_metadata(path).map_err(io)?;
        if !named.is_file() || named.file_type().is_symlink() || named.len() > max_bytes {
            return Err(invalid("executable is not a bounded regular file"));
        }
        let mut options = OpenOptions::new();
        options.read(true);
        // A held Windows handle permits readers but denies write/delete opens.
        // Unix retains the FD and compares inode/change metadata at reuse gates.
        #[cfg(windows)]
        options.share_mode(0x0000_0001);
        let mut file = options.open(path).map_err(io)?;
        let held = file.metadata().map_err(io)?;
        if !Self::same_metadata(&named, &held)? {
            return Err(invalid("executable path/FD changed before verification"));
        }
        let sha256 = hash_open_file(&mut file, held.len(), max_bytes)?;
        let verified = Self {
            path: path.to_owned(),
            file,
            metadata: held,
            sha256,
        };
        verified.validate_stability()?;
        Ok(verified)
    }

    fn same_metadata(before: &Metadata, after: &Metadata) -> Result<bool, ArenaError> {
        if !before.is_file()
            || !after.is_file()
            || before.file_type().is_symlink()
            || after.file_type().is_symlink()
            || before.len() != after.len()
            || before.modified().map_err(io)? != after.modified().map_err(io)?
        {
            return Ok(false);
        }
        #[cfg(unix)]
        if (
            before.dev(),
            before.ino(),
            before.mode(),
            before.nlink(),
            before.ctime(),
            before.ctime_nsec(),
        ) != (
            after.dev(),
            after.ino(),
            after.mode(),
            after.nlink(),
            after.ctime(),
            after.ctime_nsec(),
        ) {
            return Ok(false);
        }
        #[cfg(windows)]
        if (before.creation_time(), before.file_attributes())
            != (after.creation_time(), after.file_attributes())
        {
            return Ok(false);
        }
        Ok(true)
    }

    fn validate_stability(&self) -> Result<(), ArenaError> {
        let held = self.file.metadata().map_err(io)?;
        let named = std::fs::symlink_metadata(&self.path).map_err(io)?;
        if !Self::same_metadata(&self.metadata, &held)?
            || !Self::same_metadata(&self.metadata, &named)?
        {
            return Err(invalid("verified executable path/FD metadata changed"));
        }
        Ok(())
    }
}

fn source_description(
    engine: &CpuEngine,
    mode: &str,
    extra: serde_json::Value,
) -> Result<PalsCollectionSourceDescription, ArenaError> {
    engine
        .value_identity()
        .validate()
        .map_err(|e| invalid(e.to_string()))?;
    let executable = VerifiedExecutableIdentity::current()?;
    source_description_from_verified(engine, mode, extra, &executable)
}

fn source_description_from_verified(
    engine: &CpuEngine,
    mode: &str,
    extra: serde_json::Value,
    executable: &VerifiedExecutableIdentity,
) -> Result<PalsCollectionSourceDescription, ArenaError> {
    executable.validate_stability()?;
    let config = engine.config();
    engine
        .value_identity()
        .validate()
        .map_err(|e| invalid(e.to_string()))?;
    let implementation = executable.sha256.clone();
    let configuration = serde_json::json!({"domain":"rz-own-cpu-configuration/1","mode":mode,
        "search":CPU_SEARCH_VERSION,"evaluator":BOOTSTRAP_SCORE_VERSION,"profile":config.profile.identity(),
        "tt_entries":config.tt_entries,"max_depth":config.max_depth,"quiescence_ply":config.quiescence_ply,
        "value_identity":engine.value_identity(),"value_precision":if engine.value_identity().weights_sha256.is_some(){"fp32"}else{"integer_cp"},"extra":extra});
    let profile = canonical_sha256(&configuration)?;
    let description = PalsCollectionSourceDescription {
        mode: mode.into(),
        source: PalsInputSource::OwnCpu {
            cpu_binary_sha256: implementation.clone(),
            evaluator_configuration_sha256: profile.clone(),
            model_weights_sha256: engine.value_identity().weights_sha256.clone(),
        },
        cpu_profile_sha256: profile,
        implementation_sha256: implementation,
        encoding_sha256: native_encoding_sha()?,
        encoder_source_sha256: native_source_sha()?,
        configuration,
        model_epoch: [0; 32],
        model_epoch_kind: "encoding_only_zero".into(),
        frozen_epoch: 0,
        native: None,
    };
    executable.validate_stability()?;
    Ok(description)
}

/// Read-only registration facts from the actual executable and Rust encoder.
/// This does not load/enroll a model or claim any NN execution or training.
pub fn pals_collection_registration_description() -> Result<serde_json::Value, ArenaError> {
    let cpu = CpuEngine::new(CpuConfig::default()).map_err(|e| invalid(e.to_string()))?;
    let source = source_description(&cpu, "own-pals-native-cpu-tasks", serde_json::Value::Null)?;
    let model = PalsModelConfig::baseline();
    Ok(
        serde_json::json!({"version":"rz-pals-collection-registration-description/1",
        "collector_binary_sha256":source.implementation_sha256,"model_configuration":model,
        "model_configuration_sha256":canonical_sha256(&model)?,"encoding_sha256":source.encoding_sha256,
        "encoder_source_sha256":source.encoder_source_sha256,"cpu_configuration_sha256":source.cpu_profile_sha256,
        "cpu_task_source":source,"native_model_loaded":false,"actual_training_executed":false}),
    )
}
/// Explicit V4 registration facts. It preserves the V1 description and performs
/// no model load, graph forward, enrollment or label generation.
#[cfg(feature = "pals-collection")]
pub fn pals_collection_registration_description_v4(
    profile: rz_eval::pals_model::PalsModelProfile,
) -> Result<serde_json::Value, ArenaError> {
    let mut facts = pals_collection_registration_description()?;
    let model = PalsModelConfig::for_profile(profile);
    model.validate().map_err(|e| invalid(e.to_string()))?;
    facts["version"] = serde_json::json!("rz-pals-collection-registration-description/2");
    facts["model_configuration"] =
        serde_json::to_value(&model).map_err(|e| invalid(e.to_string()))?;
    facts["model_configuration_sha256"] = serde_json::json!(canonical_sha256(&model)?);
    facts["encoding_sha256"] = serde_json::json!(hex(
        rz_uci::pals_native::pals_fresh_encoding_semantic_digest_for_config(&model)
    ));
    facts["model_profile"] = serde_json::json!(profile.as_str());
    facts["model_semantics"] = serde_json::json!(profile.model_semantics());
    facts["model_semantics_sha256"] = serde_json::json!(hex(profile.registered_semantics_sha256()));
    facts["model_implementation_sha256"] =
        serde_json::json!(hex(rz_eval::pals_model::compiled_model_boundary_sha256()));
    facts["search_implementation_sha256"] = serde_json::json!(hex(
        rz_search::pals::engine::compiled_search_implementation_sha256()
    ));
    facts["resolver_implementation_sha256"] = serde_json::json!(hex(
        rz_search::pals::engine::compiled_resolver_implementation_sha256()
    ));
    facts["own_raw_resolver_semantics_sha256"] = serde_json::json!(hex(
        rz_search::pals::engine::ResolverPolicy::OwnRawRestricted.semantics_sha256()
    ));
    facts["model_wdl_resolver_semantics_sha256"] = serde_json::json!(hex(
        rz_search::pals::engine::ResolverPolicy::ModelWdlRestricted.semantics_sha256()
    ));
    facts["implementation_pin_contract"] = serde_json::json!({
        "model_implementation_sha256": {"manifest_field":"base.model.implementation_sha256", "domain":rz_eval::pals_model::PALS_COMPILED_MODEL_BOUNDARY_DOMAIN,
            "scope":"compiled Rust/Python model boundary sources; weights, adapter, provider and successful NN execution are separate"},
        "encoder_source_sha256": {"manifest_field":"native recipe.adapter_source_sha256", "scope":"compiled native adapter source; independent of model boundary SHA"},
        "model_semantics_sha256": {"manifest_field":"model_v2.model_semantics_sha256", "domain":rz_eval::pals_model::PALS_REGISTERED_SEMANTICS_DOMAIN,
            "scope":"registered profile/configuration/math; source provenance and binary pin are separate"}
    });
    facts["encoding_profile"] = serde_json::json!(profile.encoding_schema());
    Ok(facts)
}

/// Reject unsafe output placement/name before expensive native initialization.
/// Output::new repeats the authoritative checks when creating its fresh slot.
pub fn validate_pals_collection_output(root: &Path, run_id: &str) -> Result<(), ArenaError> {
    if !ident(run_id)
        || !root.is_absolute()
        || root
            .components()
            .any(|c| matches!(c, std::path::Component::ParentDir))
    {
        return Err(invalid(
            "collection output requires absolute path and bounded fresh run ID",
        ));
    }
    let checkout = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .ok_or_else(|| invalid("checkout location unavailable"))?
        .canonicalize()
        .map_err(io)?;
    let mut ancestor = root;
    while !ancestor.exists() {
        ancestor = ancestor
            .parent()
            .ok_or_else(|| invalid("output root has no existing ancestor"))?;
    }
    let resolved = ancestor.canonicalize().map_err(io)?;
    if !resolved.is_dir() || resolved.starts_with(&checkout) || inside_git(&resolved) {
        return Err(invalid(
            "collection outputs must remain outside Git checkout",
        ));
    }
    if root.exists() && root.canonicalize().map_err(io)?.join(run_id).exists() {
        return Err(invalid(
            "collection run already exists; overwrite forbidden",
        ));
    }
    Ok(())
}

pub struct OwnCpuCollectionDriver {
    engine: CpuEngine,
    description: PalsCollectionSourceDescription,
}
impl OwnCpuCollectionDriver {
    pub fn new(config: CpuConfig) -> Result<Self, ArenaError> {
        let engine = CpuEngine::new(config).map_err(|e| invalid(e.to_string()))?;
        let description =
            source_description(&engine, "own-cpu-bootstrap", serde_json::Value::Null)?;
        Ok(Self {
            engine,
            description,
        })
    }
}
fn cpu_raw(report: &CpuReport) -> serde_json::Value {
    serde_json::json!({"search_version":report.search_version,"profile":report.profile.identity(),
        "score_provenance":report.score_provenance,"raw_score":report.score,"score_scope":format!("{:?}",report.score_scope),
        "completed_depth":report.completed_depth,"nodes":report.nodes,"quiescence_nodes":report.quiescence_nodes,
        "tt_hits":report.tt_hits,"completion":format!("{:?}",report.completion),"root_restricted":report.root_restricted,"value_identity":report.value_identity,"value_precision":if report.value_identity.weights_sha256.is_some(){"fp32"}else{"integer_cp"},
        "pv":report.pv.iter().map(ToString::to_string).collect::<Vec<_>>(),"elapsed_us":report.elapsed.as_micros().min(u64::MAX as u128) as u64})
}
impl PalsCollectionDriver for OwnCpuCollectionDriver {
    fn description(&self) -> &PalsCollectionSourceDescription {
        &self.description
    }
    fn checked_producer_owner(&mut self) -> Result<Option<CheckedProducerOwner<'_>>, ArenaError> {
        // Measure the actual owned evaluator, including its immutable value
        // identity. Declared teacher metadata is not consulted here.
        let source =
            source_description(&self.engine, "own-cpu-bootstrap", serde_json::Value::Null)?;
        Ok(Some(CheckedProducerOwner::new(self, source)))
    }
    fn new_game(&mut self) {
        self.engine.clear();
    }
    fn analyze(
        &mut self,
        position: &Position,
        _input: &PalsFrozenInput,
        _prepared: &PalsModelInput,
        limits: CpuLimits,
        cancel: &AtomicBool,
    ) -> Result<PalsCollectionDecision, ArenaError> {
        let r = self
            .engine
            .analyze(position, limits, cancel)
            .map_err(|e| invalid(e.to_string()))?;
        Ok(PalsCollectionDecision {
            best_move: r.best_move,
            pv: r.pv.clone(),
            nodes: r.nodes,
            completed_depth: r.completed_depth,
            completed_estimate: r.score_scope == CpuScoreScope::CompletedIteration,
            raw: cpu_raw(&r),
            genealogy: vec![PalsCollectedLine {
                kind: "cpu_proposal".into(),
                moves: pack(&r.pv)?,
                raw_value: Some(r.score),
                completed_depth: r.completed_depth,
                raw_scope: format!("{:?}", r.score_scope),
                white_perspective: position.side_to_move() == Color::White,
            }],
        })
    }
}

/// CPU-only PALS lineage collection. Legal-order role mocks have no learned
/// parameters; this mode is explicitly not evidence of trained P/C strength.
pub struct OwnPalsMockCollectionDriver {
    engine: PalsEngine<LegalOrderRoleMock>,
    description: PalsCollectionSourceDescription,
}
impl OwnPalsMockCollectionDriver {
    pub fn new(cpu: CpuConfig, config: PalsConfig) -> Result<Self, ArenaError> {
        let extra = serde_json::json!({"beam":config.beam_width,"line_plies":config.line_plies,"max_nodes":config.max_nodes,
            "max_records":config.max_records,"max_role_calls":config.max_role_calls,"cpu_nodes_per_task":config.cpu_nodes_per_task});
        let cpu = CpuEngine::new(cpu).map_err(|e| invalid(e.to_string()))?;
        let description = source_description(&cpu, "own-pals-legal-order-cpu-mock", extra)?;
        let engine =
            PalsEngine::new(config, LegalOrderRoleMock, cpu).map_err(|e| invalid(e.to_string()))?;
        Ok(Self {
            engine,
            description,
        })
    }
}
impl PalsCollectionDriver for OwnPalsMockCollectionDriver {
    fn description(&self) -> &PalsCollectionSourceDescription {
        &self.description
    }
    fn new_game(&mut self) {
        self.engine.new_game();
    }
    fn analyze(
        &mut self,
        position: &Position,
        _input: &PalsFrozenInput,
        _prepared: &PalsModelInput,
        limits: CpuLimits,
        cancel: &AtomicBool,
    ) -> Result<PalsCollectionDecision, ArenaError> {
        let before = self
            .engine
            .records()
            .iter()
            .map(|r| r.revision)
            .max()
            .unwrap_or(0);
        let r = self
            .engine
            .search(
                position,
                PalsLimits {
                    deadline: limits
                        .deadline
                        .ok_or_else(|| invalid("PALS collection needs a deadline"))?,
                    max_rounds: 2,
                    max_cpu_nodes: limits.max_nodes,
                    cpu_depth: limits.max_depth,
                },
                cancel,
            )
            .map_err(|e| invalid(e.to_string()))?;
        let mut genealogy = Vec::new();
        for record in self
            .engine
            .records()
            .iter()
            .filter(|record| record.revision > before)
        {
            if !self
                .engine
                .stores()
                .states
                .get(record.origin_state)
                .map_err(|e| invalid(e.to_string()))?
                .same_state(&position.snapshot())
            {
                continue;
            }
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
            // A mock PALS choice is a restricted-engine decision, not a single
            // completed CPU task. Preserve its real task lineage without
            // fabricating a completed-depth/policy label for the choice.
            completed_depth: 0,
            completed_estimate: false,
            raw: serde_json::json!({"source":"pals_cpu_mock","resolver_version":r.resolver_version,"completion":format!("{:?}",r.completion),"value_scope":format!("{:?}",r.value_scope),
                "raw_score":r.score,"cpu_nodes":r.counters.cpu_nodes,"cpu_tasks":r.counters.cpu_tasks,"completed_cpu_tasks":r.counters.completed_cpu_tasks,
                "role_calls":r.counters.role_calls,"model":r.model_identity,"elapsed_us":r.elapsed.as_micros().min(u64::MAX as u128) as u64}),
            genealogy,
        })
    }
}

fn pack(moves: &[BoardMove]) -> Result<Vec<u16>, ArenaError> {
    moves
        .iter()
        .map(|mv| {
            Move16::pack(*mv)
                .map(|m| m.bits())
                .map_err(|e| invalid(e.to_string()))
        })
        .collect()
}
fn state_sha(position: &Position) -> Result<String, ArenaError> {
    let owner = NEXT_CAPTURE_OWNER
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |v| v.checked_add(1))
        .map_err(|_| invalid("capture owner registry exhausted"))?;
    let exported =
        ContractPosition::new(rz_contracts::OwnerId(owner), position.clone()).export()?;
    Ok(exported
        .snapshot()
        .identity()
        .semantic
        .0
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect())
}
struct IdentityHasher(Sha256);
impl Hasher for IdentityHasher {
    fn finish(&self) -> u64 {
        let bytes = self.0.clone().finalize();
        u64::from_le_bytes(bytes[..8].try_into().expect("eight digest bytes"))
    }
    fn write(&mut self, bytes: &[u8]) {
        self.0.update(bytes);
    }
}
fn transposition_sha(position: &Position) -> String {
    let mut hasher = IdentityHasher(Sha256::new());
    hasher.write(b"rz-rules-transposition-group/1");
    position.snapshot().repetition_identity().hash(&mut hasher);
    format!("{:x}", hasher.0.finalize())
}

struct Output {
    directory: PathBuf,
    files: BTreeMap<&'static str, File>,
    max_bytes: u64,
    bytes: u64,
    failed_artifacts: BTreeSet<&'static str>,
    producer: Option<RegisteredProducerHandle>,
    producer_prepaid: u64,
}
fn inside_git(path: &Path) -> bool {
    path.ancestors()
        .any(|ancestor| ancestor.join(".git").exists())
}
struct BoundedJson {
    bytes: Vec<u8>,
    limit: usize,
}
impl Write for BoundedJson {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        if bytes.len() > self.limit.saturating_sub(self.bytes.len()) {
            return Err(std::io::Error::other(
                "collector serialized record byte limit",
            ));
        }
        self.bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}
fn bounded_json<T: Serialize>(
    value: &T,
    limit: usize,
    pretty: bool,
) -> Result<Vec<u8>, ArenaError> {
    let mut writer = BoundedJson {
        bytes: Vec::new(),
        limit,
    };
    if pretty {
        serde_json::to_writer_pretty(&mut writer, value)
    } else {
        serde_json::to_writer(&mut writer, value)
    }
    .map_err(|e| {
        if e.is_io() {
            ArenaError::Budget(e.to_string())
        } else {
            invalid(e.to_string())
        }
    })?;
    Ok(writer.bytes)
}
impl Output {
    fn new(root: &Path, run_id: &str, max_bytes: u64) -> Result<Self, ArenaError> {
        if !root.is_absolute()
            || root
                .components()
                .any(|c| matches!(c, std::path::Component::ParentDir))
        {
            return Err(invalid(
                "output root must be an absolute path without parent traversal",
            ));
        }
        let checkout = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(Path::parent)
            .ok_or_else(|| invalid("checkout location unavailable"))?
            .canonicalize()
            .map_err(io)?;
        let mut existing = root;
        while !existing.exists() {
            existing = existing
                .parent()
                .ok_or_else(|| invalid("output root has no existing ancestor"))?;
        }
        let resolved_ancestor = existing.canonicalize().map_err(io)?;
        if resolved_ancestor.starts_with(&checkout) || inside_git(&resolved_ancestor) {
            return Err(invalid(
                "collection outputs must remain outside Git checkout",
            ));
        }
        std::fs::create_dir_all(root).map_err(io)?;
        let resolved = root.canonicalize().map_err(io)?;
        if resolved.starts_with(&checkout) || inside_git(&resolved) {
            return Err(invalid("resolved output root is inside checkout"));
        }
        let directory = resolved.join(run_id);
        std::fs::create_dir(&directory).map_err(io)?;
        Ok(Self {
            directory,
            files: BTreeMap::new(),
            max_bytes,
            bytes: 0,
            failed_artifacts: BTreeSet::new(),
            producer: None,
            producer_prepaid: 0,
        })
    }
    fn write(
        &mut self,
        name: &'static str,
        bytes: &[u8],
        final_record: bool,
    ) -> Result<(), ArenaError> {
        let ceiling = if final_record {
            self.max_bytes
        } else {
            self.max_bytes - RECEIPT_RESERVE
        };
        let new = self
            .bytes
            .checked_add(bytes.len() as u64)
            .filter(|n| *n <= ceiling)
            .ok_or_else(|| ArenaError::Budget("collector output byte limit".into()))?;
        // Reserve before attempting an I/O operation: a failed write_all may
        // already have written a prefix. Failed attempts never release quota.
        self.bytes = new;
        if !self.files.contains_key(name) {
            let file = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(self.directory.join(name))
                .map_err(io)?;
            self.files.insert(name, file);
        }
        #[cfg(all(test, feature = "pals-collection"))]
        if let Some(prefix) = tests::FAIL_ARTIFACT.with(|flag| {
            if flag.get().is_some_and(|(target, _)| target == name) {
                let (_, prefix) = flag.get().expect("checked fault target");
                flag.set(None);
                Some(prefix)
            } else {
                None
            }
        }) {
            self.files
                .get_mut(name)
                .ok_or_else(|| invalid("injected failure artifact not opened"))?
                .write_all(&bytes[..prefix.min(bytes.len())])
                .map_err(io)?;
            self.failed_artifacts.insert(name);
            return Err(ArenaError::Budget(
                "injected partial artifact write failure".into(),
            ));
        }
        let written = self
            .files
            .get_mut(name)
            .expect("opened artifact")
            .write_all(bytes);
        if let Err(error) = written {
            self.failed_artifacts.insert(name);
            return Err(io(error));
        }
        Ok(())
    }
    fn json<T: Serialize>(&mut self, name: &'static str, value: &T) -> Result<String, ArenaError> {
        let bytes = bounded_json(value, MAX_JSON_RECORD_BYTES, false)?;
        self.json_payload(name, &bytes)
    }
    fn json_payload(&mut self, name: &'static str, bytes: &[u8]) -> Result<String, ArenaError> {
        if bytes.len() > MAX_JSON_RECORD_BYTES {
            return Err(invalid("collector JSON payload byte bound"));
        }
        let digest = format!("{:x}", Sha256::digest(bytes));
        let mut line = bytes.to_vec();
        line.push(b'\n');
        self.write(name, &line, false)?;
        Ok(digest)
    }
    fn reserve_producer(&mut self, bytes: u64) -> Result<(), ArenaError> {
        if self.producer_prepaid != 0 {
            return Err(invalid("producer metadata credit already delegated"));
        }
        self.bytes = self
            .bytes
            .checked_add(bytes)
            .filter(|n| *n <= self.max_bytes - RECEIPT_RESERVE)
            .ok_or_else(|| {
                ArenaError::Budget(
                    "producer metadata close credit unavailable before dispatch".into(),
                )
            })?;
        self.producer_prepaid = bytes;
        Ok(())
    }
    fn write_prepaid(
        &mut self,
        name: &'static str,
        bytes: &[u8],
        newline: bool,
    ) -> Result<(), ArenaError> {
        let admitted = bytes.len() as u64 + u64::from(newline);
        if admitted > self.producer_prepaid || self.failed_artifacts.contains(name) {
            return Err(invalid(
                "producer publication exceeds remaining prepaid credit",
            ));
        }
        // Never refund a failed/partial write reservation. This credit was
        // charged before any dispatch and is not charged a second time here.
        self.producer_prepaid -= admitted;
        if !self.files.contains_key(name) {
            let file = match OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(self.directory.join(name))
            {
                Ok(file) => file,
                Err(error) => {
                    self.failed_artifacts.insert(name);
                    return Err(io(error));
                }
            };
            self.files.insert(name, file);
        }
        let file = self
            .files
            .get_mut(name)
            .ok_or_else(|| invalid("producer artifact file missing"))?;
        let result = file.write_all(bytes).and_then(|_| {
            if newline {
                file.write_all(b"\n")
            } else {
                Ok(())
            }
        });
        if let Err(error) = result {
            self.failed_artifacts.insert(name);
            return Err(io(error));
        }
        Ok(())
    }
    fn native_credit(&self) -> u64 {
        (self.max_bytes - RECEIPT_RESERVE).saturating_sub(self.bytes)
    }
    fn native_trace(&mut self, trace: &PalsNativeTrace) -> Result<(), ArenaError> {
        if trace.reserved_bytes > self.native_credit()
            || trace.rows.len() > MAX_ROWS * 8
            || trace.inputs.len() > MAX_ROWS
        {
            return Err(ArenaError::Budget(
                "native collection delegated output credit".into(),
            ));
        }
        let written = trace.rows.iter().try_fold(0_u64, |sum, row| {
            if !matches!(
                row.artifact,
                "inputs.jsonl"
                    | "native-inputs.jsonl"
                    | "native-full-line-contexts.v2.jsonl"
                    | "input-lineage.jsonl"
                    | "native-events.jsonl"
                    | "native-raw-outputs.jsonl"
                    | "public-record-sources.jsonl"
                    | "native-divergence-inputs.jsonl"
                    | "native-divergence-sidecars.jsonl"
                    | "native-divergence-contexts.jsonl"
                    | "native-work-summary.jsonl"
                    | "native-recheck-traces.jsonl"
                    | "native-continuation-traces.jsonl"
                    | "native-search-returns.jsonl"
            ) || row.json.len() > MAX_JSON_RECORD_BYTES
            {
                return Err(invalid("native collector artifact/record limit"));
            }
            sum.checked_add(row.json.len() as u64 + 1)
                .ok_or_else(|| invalid("native collector byte overflow"))
        })?;
        if written > trace.reserved_bytes {
            return Err(invalid(
                "native collector exceeded its dispatch reservation",
            ));
        }
        // The producer had exclusive delegated credit during analysis. Charge
        // that reservation exactly once, including unused/failure allowances.
        self.bytes = self
            .bytes
            .checked_add(trace.reserved_bytes)
            .ok_or_else(|| invalid("native collector reservation overflow"))?;
        let mut first_failure = None;
        for row in &trace.rows {
            let name = row.artifact;
            if self.failed_artifacts.contains(name) {
                continue;
            }
            if !self.files.contains_key(name) {
                let opened = OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(self.directory.join(name));
                match opened {
                    Ok(file) => {
                        self.files.insert(name, file);
                    }
                    Err(error) => {
                        self.failed_artifacts.insert(name);
                        if first_failure.is_none() {
                            first_failure = Some(io(error));
                        }
                        continue;
                    }
                }
            }
            let file = self
                .files
                .get_mut(name)
                .ok_or_else(|| invalid("native output file missing"))?;
            if let Err(error) = file
                .write_all(&row.json)
                .and_then(|_| file.write_all(b"\n"))
            {
                self.failed_artifacts.insert(name);
                if first_failure.is_none() {
                    first_failure = Some(io(error));
                }
            }
        }
        match first_failure {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }
    fn final_json<T: Serialize>(
        &mut self,
        name: &'static str,
        value: &T,
    ) -> Result<(), ArenaError> {
        let mut bytes = bounded_json(value, MAX_JSON_RECORD_BYTES, false)?;
        bytes.push(b'\n');
        self.write(name, &bytes, true)
    }
    fn finish(&mut self, receipt: &PalsCollectionReceipt) -> Result<(), ArenaError> {
        // Compact admitted source+run config and bounded result/failure text
        // fit the receipt reserve; pretty indentation must not multiply them.
        let mut bytes = bounded_json(receipt, RECEIPT_RESERVE as usize - 1, false)?;
        bytes.push(b'\n');
        self.write("receipt.json", &bytes, true)?;
        for file in self.files.values() {
            file.sync_all().map_err(io)?;
        }
        Ok(())
    }
    fn artifacts(&self) -> Result<BTreeMap<String, PalsCollectionArtifact>, ArenaError> {
        let mut result = BTreeMap::new();
        for name in self.files.keys() {
            let path = self.directory.join(name);
            let bytes = std::fs::metadata(&path).map_err(io)?.len();
            result.insert(
                (*name).into(),
                PalsCollectionArtifact {
                    bytes,
                    sha256: file_sha(&path, self.max_bytes)?,
                },
            );
        }
        Ok(result)
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PalsCollectionArtifact {
    pub bytes: u64,
    pub sha256: String,
}
#[derive(Clone, Debug, Serialize)]
pub struct PalsCollectionReceipt {
    pub version: String,
    pub run_id: String,
    pub source: PalsCollectionSourceDescription,
    pub config: PalsCollectionConfig,
    pub complete: bool,
    pub failure: Option<String>,
    pub games_finished: u32,
    pub cpu_nodes: u64,
    pub cpu_jobs: u64,
    pub cpu_work_observation_incomplete: bool,
    pub wall_time_ms: u64,
    /// Payload written before this receipt; the receipt reserve is separately declared.
    pub data_output_bytes: u64,
    /// Includes full reservations for failed/partial writes, conservatively
    /// ensuring the artifact bytes cannot exceed the declared output ceiling.
    pub reserved_output_bytes: u64,
    pub receipt_reserve_bytes: u64,
    pub results: Vec<PalsGameResult>,
    pub audit: Option<PalsDatasetAudit>,
    /// Exact admitted artifact bytes before receipt creation. The receipt never
    /// declares its own digest or reads files named by supplied dataset rows.
    pub artifacts: BTreeMap<String, PalsCollectionArtifact>,
    pub actual_training_executed: bool,
    pub external_teacher_used: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub native_finish: Option<serde_json::Value>,
}

struct Collection {
    sequence: u64,
    nodes: u64,
    jobs: u64,
    work_incomplete: bool,
    rows: Vec<PalsLearningRecord>,
    /// Only successfully appended labels enter this bounded predecessor index.
    /// Unlabeled raw rows never replace the latest whole label.
    latest_labels: BTreeMap<String, (u64, String)>,
    inputs: Vec<CapturedInput>,
    assignments: BTreeMap<String, PalsSplit>,
    results: Vec<PalsGameResult>,
    public: Vec<PalsPublicRecord>,
    role_records: Vec<RoleRecord>,
}
#[derive(Clone)]
struct CapturedInput {
    input: PalsFrozenInput,
    outcome_eligible: bool,
}

/// Borrowed state and role context available before an input is sealed.
struct InputCapture<'a> {
    position: &'a Position,
    opening: &'a PalsCollectionOpening,
    game: &'a str,
    position_command: String,
    actual_moves: &'a [BoardMove],
    role: PalsDataRole,
    source: &'a PalsCollectionSourceDescription,
    outcome_eligible: bool,
    prefix: &'a [BoardMove],
    proposal: &'a [BoardMove],
    counterexample: Option<&'a [BoardMove]>,
    deadline: Instant,
    cancel: &'a AtomicBool,
}

/// One immutable public observation and its exact Rules anchor.
struct PublicObservation<'a> {
    position: &'a Position,
    hash: String,
    kind: RecordKind,
    line: &'a [BoardMove],
    value: Option<i32>,
    completed_depth: u16,
    scope: Option<CpuScoreScope>,
}

/// Completed CPU evidence, kept separate from the previously sealed input.
struct CpuLabelEvidence {
    task_sha: String,
    raw_sha: String,
    depth: u16,
    nodes: u64,
    sequence: u64,
    movement: BoardMove,
    counterexample: Option<PalsCounterexampleTarget>,
}

impl Collection {
    fn next(&mut self) -> Result<u64, ArenaError> {
        self.sequence = self
            .sequence
            .checked_add(1)
            .ok_or_else(|| invalid("capture sequence exhausted"))?;
        Ok(self.sequence)
    }
    fn add_row(
        &mut self,
        output: &mut Output,
        mut row: PalsLearningRecord,
    ) -> Result<(), ArenaError> {
        if self.rows.len() >= MAX_ROWS {
            return Err(ArenaError::Budget("collection row limit".into()));
        }
        let input_sha = row.input.sha256().to_owned();
        let next_sequence = if let Some(label) = &mut row.future_label {
            match self.latest_labels.get(&input_sha) {
                Some((sequence, digest)) => {
                    if label.observed_sequence <= *sequence {
                        return Err(invalid("collector label sequence must strictly increase"));
                    }
                    if label
                        .supersedes_label_sha256
                        .as_ref()
                        .is_some_and(|prior| prior != digest)
                    {
                        return Err(invalid(
                            "collector label predecessor differs from latest persisted label",
                        ));
                    }
                    label.supersedes_label_sha256 = Some(digest.clone());
                }
                None => {
                    if label.supersedes_label_sha256.is_some() {
                        return Err(invalid(
                            "collector label predecessor was not persisted for this input",
                        ));
                    }
                    if self.latest_labels.len() >= MAX_ROWS {
                        return Err(ArenaError::Budget("collection label index limit".into()));
                    }
                }
            }
            Some(label.observed_sequence)
        } else {
            None
        };
        // The public digest validates the complete row after the producer has
        // explicitly linked its actual persisted predecessor. No old targets
        // are merged into a newer whole label, including masked ActualGame rows.
        let label_digest = row.label_digest()?;
        output.json("records.jsonl", &row)?;
        if let (Some(sequence), Some(digest)) = (next_sequence, label_digest) {
            self.latest_labels.insert(input_sha, (sequence, digest));
        }
        self.rows.push(row);
        Ok(())
    }
    fn take_native(
        &mut self,
        output: &mut Output,
        trace: PalsNativeTrace,
        limits: CpuLimits,
        secondary: &mut NativeRetentionFailures,
    ) -> Result<(), ArenaError> {
        if self
            .inputs
            .len()
            .checked_add(trace.inputs.len())
            .is_none_or(|n| n > MAX_ROWS)
            || trace.sequence < self.sequence
        {
            return Err(ArenaError::Budget("native input/sequence limit".into()));
        }
        for (input, _) in &trace.inputs {
            input.verify()?;
            if input.snapshot().capture_sequence <= self.sequence
                || input.snapshot().capture_sequence > trace.sequence
                || !matches!(input.snapshot().source, PalsInputSource::OwnPals { .. })
            {
                return Err(invalid(
                    "native collector sealed input sequence/source mismatch",
                ));
            }
        }
        self.nodes = self
            .nodes
            .checked_add(trace.cpu_nodes)
            .ok_or_else(|| invalid("native CPU node counter overflow"))?;
        self.jobs = self
            .jobs
            .checked_add(trace.cpu_jobs)
            .ok_or_else(|| invalid("native CPU task counter overflow"))?;
        self.work_incomplete |= trace.cpu_work_observation_incomplete;
        // Both bounded persistence attempts remain possible even when the
        // first fails. Neither can erase the already observed native failure.
        let persisted = output.native_trace(&trace);
        let journal = if let Some(producer) = output.producer.clone() {
            producer.flush_journal(output)
        } else {
            Ok(())
        };
        let exceeded_nodes = trace.cpu_nodes > limits.max_nodes;
        let observed = if exceeded_nodes {
            Err(invalid(
                "native CPU work exceeded the actual requested node budget",
            ))
        } else if let Some(failure) = &trace.failure {
            Err(invalid(format!(
                "native collector observer failed: {failure}"
            )))
        } else {
            Ok(())
        };
        // Inputs/sequence are admitted only after persistence succeeds, as in
        // the original path. A failed persistence never becomes admission.
        let persistence_ok = persisted.is_ok() && journal.is_ok();
        let retained = native_after_retention(
            observed,
            persisted,
            NativeRetentionStage::Persistence,
            secondary,
        );
        let retained =
            native_after_retention(retained, journal, NativeRetentionStage::Journal, secondary);
        if !persistence_ok {
            return retained;
        }
        self.sequence = trace.sequence;
        self.inputs
            .extend(
                trace
                    .inputs
                    .into_iter()
                    .map(|(input, outcome_eligible)| CapturedInput {
                        input,
                        outcome_eligible,
                    }),
            );
        retained
    }
    fn capture(
        &mut self,
        output: &mut Output,
        context: InputCapture<'_>,
    ) -> Result<(PalsFrozenInput, PalsModelInput), ArenaError> {
        let InputCapture {
            position,
            opening,
            game,
            position_command,
            actual_moves,
            role,
            source,
            outcome_eligible,
            prefix,
            proposal,
            counterexample,
            deadline,
            cancel,
        } = context;
        let sequence = self.next()?;
        let legal = position.legal_moves();
        let query = RoleQuery {
            position,
            legal: &legal,
            prefix,
            proposal,
            counterexample,
            records: &self.role_records,
            revision: sequence,
            deadline,
            cancel,
        };
        let prepared = encode_native(&query, role, source)?;
        let selected = prepared
            .records
            .iter()
            .map(|record| {
                let index = usize::try_from(
                    record
                        .record_id
                        .checked_sub(1)
                        .ok_or_else(|| invalid("native record ID zero"))?,
                )
                .map_err(|_| invalid("native record ID overflow"))?;
                let raw = self
                    .public
                    .get(index)
                    .ok_or_else(|| invalid("native record lacks immutable raw source"))?;
                if raw.situation_revision != record.revision {
                    return Err(invalid("native record/source revision mismatch"));
                }
                Ok(raw.clone())
            })
            .collect::<Result<Vec<_>, ArenaError>>()?;
        let input = PalsInputSnapshot {
            game_id: game.into(),
            opening_id: opening.id.clone(),
            line_genealogy_id: format!("{}-actual-and-conditional", game),
            position_command,
            board_fen: position.to_fen(),
            actual_history: pack(actual_moves)?,
            rules_state_sha256: state_sha(position)?,
            rules_history_sha256: hex(prepared.history_digest),
            transposition_sha256: transposition_sha(position),
            encoding_sha256: source.encoding_sha256.clone(),
            source: source.source.clone(),
            frozen_epoch: source.frozen_epoch,
            input_revision: sequence,
            capture_sequence: sequence,
            white_to_move: position.side_to_move() == Color::White,
            role,
            legal_moves: pack(position.ordered_legal_moves().moves())?,
            public_records: selected.clone(),
        }
        .seal()?;
        let input_json = bounded_json(&input, MAX_JSON_RECORD_BYTES, false)?;
        output.json_payload("inputs.jsonl", &input_json)?;
        let tensor_json = serde_json::to_string(&prepared).map_err(|e| invalid(e.to_string()))?;
        let tensor_sha256 = format!("{:x}", Sha256::digest(tensor_json.as_bytes()));
        let mut sidecar = PalsNativeInputSidecar {
            version: "rz-pals-native-input-sidecar/1".into(),
            input_sha256: input.sha256().into(),
            encoding_sha256: source.encoding_sha256.clone(),
            encoder_source_sha256: native_source_sha()?,
            model_epoch_kind: source.model_epoch_kind.clone(),
            canonical_tensor_sha256: hex(prepared
                .canonical_input_key(&PalsModelConfig::baseline())
                .map_err(|e| invalid(e.to_string()))?),
            tensor_json,
            tensor_sha256,
            record_sources: selected,
            model_profile: None,
            encoding_profile: None,
            sha256: String::new(),
        };
        sidecar.sha256 = canonical_sha256(&(
            sidecar.version.as_str(),
            sidecar.input_sha256.as_str(),
            sidecar.encoding_sha256.as_str(),
            sidecar.encoder_source_sha256.as_str(),
            sidecar.model_epoch_kind.as_str(),
            sidecar.canonical_tensor_sha256.as_str(),
            sidecar.tensor_json.as_str(),
            sidecar.tensor_sha256.as_str(),
            &sidecar.record_sources,
        ))?;
        let sidecar_json = bounded_json(&sidecar, MAX_JSON_RECORD_BYTES, false)?;
        output.json_payload("native-inputs.jsonl", &sidecar_json)?;
        let lineage_json = bounded_json(
            &serde_json::json!({"input_sha256":input.sha256(),"game_id":game,"actual_played_history":pack(actual_moves)?,"virtual_branch":!outcome_eligible,"actual_outcome_eligible":outcome_eligible}),
            MAX_JSON_RECORD_BYTES,
            false,
        )?;
        output.json_payload("input-lineage.jsonl", &lineage_json)?;
        if let Some(producer) = output.producer.clone() {
            producer.verify_source(source)?;
            producer.capture(
                &input,
                &input_json,
                &sidecar_json,
                &lineage_json,
                None,
                true,
            )?;
            producer.flush_journal(output)?;
        }
        self.inputs.push(CapturedInput {
            input: input.clone(),
            outcome_eligible,
        });
        Ok((input, prepared))
    }
    fn publish(
        &mut self,
        output: &mut Output,
        observation: PublicObservation<'_>,
    ) -> Result<(), ArenaError> {
        let PublicObservation {
            position,
            hash,
            kind,
            line,
            value,
            completed_depth,
            scope,
        } = observation;
        if self.role_records.len() >= 16_384 {
            return Err(ArenaError::Budget("public source-record scan limit".into()));
        }
        let record = RoleRecord {
            revision: self.sequence,
            origin_state: StateId(
                usize::try_from(self.sequence)
                    .map_err(|_| invalid("advisory state ID exceeds target integer range"))?,
            ),
            kind,
            line: line.to_vec(),
            value,
            completed_depth,
            score_scope: scope,
            cpu_observation: None,
            parent_revision: None,
            supersedes_revision: None,
            perspective: position.side_to_move(),
            critical: false,
        };
        // Every immutable record retains its actual Rules anchor. The numeric
        // local StateId is advisory to the encoder, never exact state identity.
        let source=output.json("public-record-sources.jsonl",&serde_json::json!({"record_index":self.role_records.len()+1,"origin_state_id":record.origin_state.0,"revision":self.sequence,
            "origin_rules_state_sha256":state_sha(position)?,"kind":format!("{:?}",kind),"line":pack(line)?,"value":value,"completed_depth":completed_depth,
            "scope":format!("{:?}",scope),"white_score_perspective":position.side_to_move()==Color::White,"observation_sha256":hash,"critical":false}))?;
        self.public.push(PalsPublicRecord {
            observation_sha256: source,
            situation_revision: self.sequence,
        });
        self.role_records.push(record);
        Ok(())
    }
}

#[cfg(feature = "pals-collection")]
fn native_source_sha() -> Result<String, ArenaError> {
    Ok(hex(pals_native_source_digest()))
}
#[cfg(not(feature = "pals-collection"))]
fn native_source_sha() -> Result<String, ArenaError> {
    Err(invalid("native input encoder feature is unavailable"))
}
#[cfg(feature = "pals-collection")]
fn encode_native(
    query: &RoleQuery<'_>,
    role: PalsDataRole,
    source: &PalsCollectionSourceDescription,
) -> Result<PalsModelInput, ArenaError> {
    let kind = if role == PalsDataRole::Critic {
        NativeQueryKind::Reply
    } else if query.counterexample.is_some() {
        NativeQueryKind::Repair
    } else {
        NativeQueryKind::Propose
    };
    if role == PalsDataRole::Verifier {
        return Err(invalid(
            "verifier-private inputs require a distinct collection path",
        ));
    }
    prepare_role_input(query, kind, source.model_epoch).map_err(|e| invalid(e.to_string()))
}
#[cfg(not(feature = "pals-collection"))]
fn encode_native(
    _: &RoleQuery<'_>,
    _: PalsDataRole,
    _: &PalsCollectionSourceDescription,
) -> Result<PalsModelInput, ArenaError> {
    Err(invalid("native input encoder feature is unavailable"))
}

fn position_command(opening: &PalsCollectionOpening, moves: &[BoardMove]) -> String {
    let mut result = opening.initial_fen.as_ref().map_or_else(
        || "position startpos".into(),
        |fen| format!("position fen {fen}"),
    );
    if !moves.is_empty() {
        result.push_str(" moves");
        for mv in moves {
            result.push(' ');
            result.push_str(&mv.to_string());
        }
    }
    result
}
fn replay_line(position: &Position, line: &[BoardMove]) -> Result<(Position, String), ArenaError> {
    let mut next = position.clone();
    let start = state_sha(position)?;
    let mut transitions = Vec::new();
    for mv in line {
        if termination(&next)?.is_some() {
            return Err(invalid(
                "continuation contains a move after an exact Rules terminal",
            ));
        }
        let before = state_sha(&next)?;
        next.make_move(*mv).map_err(rules)?;
        transitions.push((
            before,
            Move16::pack(*mv)
                .map_err(|e| invalid(e.to_string()))?
                .bits(),
            state_sha(&next)?,
        ));
    }
    let evidence = canonical_sha256(&("rz-rules-checked-line/1", start, transitions))?;
    Ok((next, evidence))
}
fn job_limits(
    config: &PalsCollectionConfig,
    collection: &Collection,
    deadline: Instant,
) -> Result<CpuLimits, ArenaError> {
    let remaining = config.max_total_nodes.saturating_sub(collection.nodes);
    if remaining == 0 {
        return Err(ArenaError::Budget("total CPU node limit".into()));
    }
    if Instant::now() >= deadline {
        return Err(ArenaError::Budget("collector wall-time limit".into()));
    }
    Ok(CpuLimits {
        max_depth: config.cpu_depth,
        max_nodes: config.nodes_per_job.min(remaining),
        deadline: Some(deadline),
    })
}
fn account(collection: &mut Collection, limits: CpuLimits, nodes: u64) -> Result<(), ArenaError> {
    if nodes > limits.max_nodes {
        return Err(invalid("CPU source exceeded requested node budget"));
    }
    collection.nodes = collection
        .nodes
        .checked_add(nodes)
        .ok_or_else(|| invalid("CPU node counter overflow"))?;
    collection.jobs += 1;
    Ok(())
}
fn policy(input: &PalsFrozenInput, movement: BoardMove) -> Result<PalsPolicyTarget, ArenaError> {
    let packed = Move16::pack(movement)
        .map_err(|e| invalid(e.to_string()))?
        .bits();
    let at = input
        .snapshot()
        .legal_moves
        .iter()
        .position(|mv| *mv == packed)
        .ok_or_else(|| invalid("source bestmove is illegal at captured state"))?;
    let mut probabilities = vec![0.0; input.snapshot().legal_moves.len()];
    probabilities[at] = 1.0;
    Ok(PalsPolicyTarget {
        moves: input.snapshot().legal_moves.clone(),
        probabilities,
    })
}
fn owned_label(
    input: &PalsFrozenInput,
    description: &PalsCollectionSourceDescription,
    evidence: CpuLabelEvidence,
) -> Result<PalsLearningRecord, ArenaError> {
    let CpuLabelEvidence {
        task_sha,
        raw_sha,
        depth,
        nodes,
        sequence,
        movement,
        counterexample,
    } = evidence;
    Ok(PalsLearningRecord {
        input: input.clone(),
        future_label: Some(PalsFutureLabel {
            observed_sequence: sequence,
            provenance: PalsTargetProvenance::OwnedCpu {
                engine_sha256: description.implementation_sha256.clone(),
                profile_sha256: description.cpu_profile_sha256.clone(),
                task_sha256: task_sha,
                completed_depth: u32::from(depth),
                nodes,
                raw_evidence_sha256: raw_sha,
            },
            policy: Some(policy(input, movement)?),
            value_wdl: None,
            white_to_move: input.snapshot().white_to_move,
            counterexample,
            verifier_tasks: None,
            supersedes_label_sha256: None,
        }),
        verifier_private: None,
    })
}
fn termination(position: &Position) -> Result<Option<(PalsOutcome, PalsGameEnd)>, ArenaError> {
    let status = position
        .play_status_from_view(&position.ordered_legal_moves())
        .map_err(rules)?;
    Ok(match status {
        PlayStatus::Ongoing => None,
        PlayStatus::Terminal { reason, winner } => Some((
            match winner {
                Some(Color::White) => PalsOutcome::WhiteWin,
                Some(Color::Black) => PalsOutcome::BlackWin,
                None => PalsOutcome::Draw,
            },
            match reason {
                TerminalReason::Checkmate => PalsGameEnd::Checkmate,
                TerminalReason::Stalemate => PalsGameEnd::Stalemate,
                TerminalReason::DeadPosition => PalsGameEnd::DeadPosition,
                TerminalReason::FivefoldRepetition => PalsGameEnd::Repetition,
                TerminalReason::SeventyFiveMove => PalsGameEnd::SeventyFiveMove,
            },
        )),
    })
}

struct ActiveGame {
    game: String,
    opening: PalsCollectionOpening,
    movetext: String,
    position_sha: String,
    moves: Vec<String>,
    terminal: Option<(PalsOutcome, PalsGameEnd)>,
    resolved: Option<PalsGameResult>,
    result_written: bool,
    pgn_written: bool,
}
impl ActiveGame {
    fn new(
        game: &str,
        opening: &PalsCollectionOpening,
        position: &Position,
    ) -> Result<Self, ArenaError> {
        Ok(Self {
            game: game.into(),
            opening: opening.clone(),
            movetext: String::new(),
            position_sha: state_sha(position)?,
            moves: Vec::new(),
            terminal: termination(position)?,
            resolved: None,
            result_written: false,
            pgn_written: false,
        })
    }
    fn advance(
        &mut self,
        position: &Position,
        movetext: &str,
        moves: &[BoardMove],
    ) -> Result<(), ArenaError> {
        self.movetext = movetext.into();
        self.position_sha = state_sha(position)?;
        self.moves = moves.iter().map(ToString::to_string).collect();
        self.terminal = termination(position)?;
        Ok(())
    }
}
fn checked_source(description: &PalsCollectionSourceDescription) -> Result<(), ArenaError> {
    description.source.validate()?;
    let _ = bounded_json(&description.configuration, 64 * 1024, false)?;
    if !ident(&description.mode)
        || description.cpu_profile_sha256.len() != 64
        || !description
            .cpu_profile_sha256
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        || description.encoding_sha256 != source_encoding_sha(description)?
        || description.encoder_source_sha256 != native_source_sha()?
    {
        return Err(invalid("collector source/actual encoder mismatch"));
    }
    let digest = canonical_sha256(&description.configuration)?;
    match &description.source {
        PalsInputSource::OwnCpu {
            cpu_binary_sha256,
            evaluator_configuration_sha256,
            model_weights_sha256,
        } => {
            let value_identity: rz_search::cpu_value::CpuValueIdentity = serde_json::from_value(
                description
                    .configuration
                    .get("value_identity")
                    .cloned()
                    .ok_or_else(|| invalid("CPU source lacks exact evaluator value identity"))?,
            )
            .map_err(|e| invalid(e.to_string()))?;
            value_identity
                .validate()
                .map_err(|e| invalid(e.to_string()))?;
            if cpu_binary_sha256 != &description.implementation_sha256
                || evaluator_configuration_sha256 != &digest
                || digest != description.cpu_profile_sha256
                || model_weights_sha256 != &value_identity.weights_sha256
                || description.model_epoch != [0; 32]
                || description.model_epoch_kind != "encoding_only_zero"
                || description.frozen_epoch != 0
            {
                return Err(invalid(
                    "weightless own-CPU source does not match actual configuration or encoding-only epoch",
                ));
            }
        }
        PalsInputSource::OwnPals {
            model_configuration_sha256,
            ..
        } => {
            if model_configuration_sha256 != &digest
                || description.model_epoch_kind != "frozen_model_epoch"
                || description.model_epoch == [0; 32]
                || description.frozen_epoch == 0
            {
                return Err(invalid(
                    "neural PALS source needs verified model configuration and actual frozen epoch",
                ));
            }
        }
    }
    Ok(())
}
fn failure_text(error: impl std::fmt::Display) -> String {
    bounded_failure_text(error, 8192)
}
fn bounded_failure_text(error: impl std::fmt::Display, max_bytes: usize) -> String {
    let text = error.to_string();
    if bounded_json(&text, max_bytes, false).is_ok() {
        return text;
    }
    let suffix = format!(
        " [message truncated; bytes={}; sha256={:x}]",
        text.len(),
        Sha256::digest(text.as_bytes())
    );
    let mut end = max_bytes.min(text.len());
    loop {
        while !text.is_char_boundary(end) {
            end -= 1;
        }
        let candidate = format!("{}{suffix}", &text[..end]);
        if bounded_json(&candidate, max_bytes, false).is_ok() {
            return candidate;
        }
        if end == 0 {
            return suffix;
        }
        end = end.saturating_sub((end / 8).max(1));
    }
}

#[derive(Clone, Copy)]
enum NativeRetentionStage {
    Recovery,
    Persistence,
    Journal,
}
impl NativeRetentionStage {
    fn index(self) -> usize {
        match self {
            Self::Recovery => 0,
            Self::Persistence => 1,
            Self::Journal => 2,
        }
    }
}
#[derive(Default)]
struct NativeRetentionFailures {
    // One stopped analysis has at most these three retention boundaries. Each
    // note has a 1 KiB JSON bound; no unbounded per-call failure list is kept.
    notes: [Option<String>; 3],
}
impl NativeRetentionFailures {
    fn record(&mut self, stage: NativeRetentionStage, error: &ArenaError) {
        self.notes[stage.index()] = Some(bounded_failure_text(error, 1024));
    }
    fn failure_text(&self, primary: impl std::fmt::Display) -> String {
        if self.notes.iter().all(Option::is_none) {
            return failure_text(primary);
        }
        // Reserve room for every bounded secondary in the existing 8 KiB
        // failure field. The primary stays first, with its truncation SHA when
        // necessary; receipt/failures.jsonl schemas and quotas do not change.
        let mut text = bounded_failure_text(primary, 4096);
        for (stage, note) in ["trace-recovery", "trace-persistence", "producer-journal"]
            .into_iter()
            .zip(&self.notes)
        {
            if let Some(note) = note {
                text.push_str("; native-retention-secondary[");
                text.push_str(stage);
                text.push_str("]=");
                text.push_str(note);
            }
        }
        failure_text(text)
    }
}
fn native_after_retention<T>(
    primary: Result<T, ArenaError>,
    retention: Result<(), ArenaError>,
    stage: NativeRetentionStage,
    secondary: &mut NativeRetentionFailures,
) -> Result<T, ArenaError> {
    match primary {
        Err(error) => {
            if let Err(retention_error) = retention {
                secondary.record(stage, &retention_error);
            }
            Err(error)
        }
        Ok(value) => {
            retention?;
            Ok(value)
        }
    }
}

/// All output is append-only in a fresh directory outside the checkout. A
/// failed collection returns a receipt with complete=false and preserves the
/// admitted raw records. Split leakage is an explicit failed audit, never an
/// automatic reassignment of a user-declared holdout game.
pub fn collect_pals_own_data(
    config: PalsCollectionConfig,
    output_root: &Path,
    driver: &mut dyn PalsCollectionDriver,
    registry: &PalsOwnedSources,
    cancel: &AtomicBool,
) -> Result<PalsCollectionReceipt, ArenaError> {
    collect_pals_own_data_with_producer(config, output_root, driver, registry, cancel, None)
}

/// Optional strict producer capture. A requested registration failure never
/// falls back to legacy collection. Existing receipt fields/domains are intact.
pub fn collect_pals_own_data_with_producer(
    config: PalsCollectionConfig,
    output_root: &Path,
    driver: &mut dyn PalsCollectionDriver,
    registry: &PalsOwnedSources,
    cancel: &AtomicBool,
    producer_config: Option<&PalsProducerCollectionConfig>,
) -> Result<PalsCollectionReceipt, ArenaError> {
    let started = Instant::now();
    config.validate()?;
    registry.validate()?;
    // Strict dispatch uses the private capability's actual owner. A custom
    // wrapper cannot borrow a real owner's proof and then run its own analyze.
    let (driver, producer_source) = if producer_config.is_some() {
        let owner = driver
            .checked_producer_owner()?
            .ok_or_else(|| invalid("strict producer requires checked actual driver constructor"))?;
        let (actual_driver, source) = owner.into_parts();
        (actual_driver, Some(source))
    } else {
        (driver, None)
    };
    let description = driver.description().clone();
    checked_source(&description)?;
    if !registry.input_sources.contains(&description.source)
        || !registry
            .cpu_binary_sha256
            .contains(&description.implementation_sha256)
    {
        return Err(invalid("collector source is not independently registered"));
    }
    let actual_binary = VerifiedExecutableIdentity::current()?;
    if description.implementation_sha256 != actual_binary.sha256 {
        return Err(invalid(
            "collector executable differs from registered CPU implementation",
        ));
    }
    let producer = if let Some(pin) = producer_config {
        let actual = producer_source
            .ok_or_else(|| invalid("strict producer has no actual owner capability"))?;
        let handle = RegisteredProducerHandle::admit(pin, &config, &actual)?;
        handle.verify_source(&description)?;
        Some(handle)
    } else {
        None
    };
    driver.set_registered_producer(producer.clone())?;
    let mut output = Output::new(output_root, &config.run_id, config.max_output_bytes)?;
    if let (Some(handle), Some(pin)) = (&producer, producer_config) {
        handle.start(pin, &mut output)?;
    }
    output.producer = producer.clone();
    output.json("source-registry.jsonl", registry)?;
    output.json("run-spec.jsonl", &config)?;
    let deadline = started
        .checked_add(Duration::from_millis(config.max_wall_time_ms))
        .ok_or_else(|| invalid("collector deadline overflow"))?;
    let mut state = Collection {
        sequence: 0,
        nodes: 0,
        jobs: 0,
        work_incomplete: false,
        rows: Vec::new(),
        latest_labels: BTreeMap::new(),
        inputs: Vec::new(),
        assignments: BTreeMap::new(),
        results: Vec::new(),
        public: Vec::new(),
        role_records: Vec::new(),
    };
    let mut failure = None;
    let mut native_secondary = NativeRetentionFailures::default();
    let mut active: Option<ActiveGame> = None;
    let mut conditional_cpu =
        CpuEngine::new(CpuConfig::default()).map_err(|e| invalid(e.to_string()))?;
    let conditional_description = source_description_from_verified(
        &conditional_cpu,
        "own-cpu-conditional-verification",
        serde_json::Value::Null,
        &actual_binary,
    )?;
    output.json("conditional-source.jsonl", &conditional_description)?;
    let run_result = (|| -> Result<(), ArenaError> {
        for game_number in 0..config.games {
            if cancel.load(Ordering::Acquire) || Instant::now() >= deadline {
                return Err(ArenaError::Budget(
                    "collection cancelled or wall-time exhausted before next game".into(),
                ));
            }
            let opening = &config.openings[game_number as usize % config.openings.len()];
            let game = format!("{}-g{}", config.run_id, game_number + 1);
            driver.new_game();
            conditional_cpu.clear();
            state.public.clear();
            state.role_records.clear();
            let input_start = state.inputs.len();
            // This bounded Rules ceiling also covers hypothetical CPU depth,
            // quiescence and PALS continuation plies. Restricting it to the
            // actual game budget would reject valid deeper CPU queries.
            // Actual openings+game are separately bounded below 1,313 plies.
            let limits = PositionLimits::default();
            let mut position = match &opening.initial_fen {
                Some(fen) => Position::from_fen_with_limits(fen, limits).map_err(rules)?,
                None => Position::startpos_with_limits(limits).map_err(rules)?,
            };
            let mut moves = Vec::new();
            let mut movetext = String::new();
            active = Some(ActiveGame::new(&game, opening, &position)?);
            for text in &opening.moves {
                if termination(&position)?.is_some() {
                    return Err(invalid(
                        "declared opening contains a move after an exact Rules terminal",
                    ));
                }
                let mv = BoardMove::from_uci(text).map_err(rules)?;
                let before = state_sha(&position)?;
                append_san(&mut movetext, &position, mv)?;
                position.make_move(mv).map_err(rules)?;
                moves.push(mv);
                active
                    .as_mut()
                    .ok_or_else(|| invalid("active game missing after opening move"))?
                    .advance(&position, &movetext, &moves)?;
                output.json("transitions.jsonl",&serde_json::json!({"game_id":game,"from":before,"move":Move16::pack(mv).map_err(|e|invalid(e.to_string()))?.bits(),"to":state_sha(&position)?,"actual":true,"declared_opening":true}))?;
            }
            let mut rng = config.seed ^ u64::from(game_number).wrapping_mul(0x9e3779b97f4a7c15);
            for _ in 0..config.exploration_plies {
                if termination(&position)?.is_some() {
                    break;
                }
                let legal = position.legal_moves();
                rng = rng.wrapping_add(0x9e3779b97f4a7c15);
                let mut value = rng;
                value = (value ^ (value >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
                value = (value ^ (value >> 27)).wrapping_mul(0x94d049bb133111eb);
                value ^= value >> 31;
                let movement = legal[value as usize % legal.len()];
                let from = state_sha(&position)?;
                append_san(&mut movetext, &position, movement)?;
                position.make_move(movement).map_err(rules)?;
                moves.push(movement);
                active
                    .as_mut()
                    .ok_or_else(|| invalid("active game missing after exploration move"))?
                    .advance(&position, &movetext, &moves)?;
                output.json("transitions.jsonl",&serde_json::json!({"game_id":game,"from":from,"move":Move16::pack(movement).map_err(|e|invalid(e.to_string()))?.bits(),"to":state_sha(&position)?,"actual":true,"seeded_opening":true,"rng":"splitmix64-modulo-legal-order/1"}))?;
            }
            let mut end = None;
            for ply in 0..=config.max_plies {
                if cancel.load(Ordering::Acquire) {
                    end = Some((PalsOutcome::Unknown, PalsGameEnd::UserStop));
                    break;
                }
                if Instant::now() >= deadline {
                    end = Some((PalsOutcome::Unknown, PalsGameEnd::WallTimeLimit));
                    break;
                }
                let native_calls = driver.records_actual_native_calls();
                let root_capture = if native_calls {
                    None
                } else {
                    Some(state.capture(
                        &mut output,
                        InputCapture {
                            position: &position,
                            opening,
                            game: &game,
                            position_command: position_command(opening, &moves),
                            actual_moves: &moves,
                            role: PalsDataRole::Proposer,
                            source: &description,
                            outcome_eligible: true,
                            prefix: &[],
                            proposal: &[],
                            counterexample: None,
                            deadline,
                            cancel,
                        },
                    )?)
                };
                if let Some(terminal) = termination(&position)? {
                    end = Some(terminal);
                    break;
                }
                if ply == config.max_plies {
                    end = Some((PalsOutcome::Unknown, PalsGameEnd::PlyLimit));
                    break;
                }
                if cancel.load(Ordering::Acquire) {
                    end = Some((PalsOutcome::Unknown, PalsGameEnd::UserStop));
                    break;
                }
                if Instant::now() >= deadline {
                    end = Some((PalsOutcome::Unknown, PalsGameEnd::WallTimeLimit));
                    break;
                }
                let job = job_limits(&config, &state, deadline)?;
                let task_sha = canonical_sha256(&(
                    "rz-pals-own-task/1",
                    root_capture.as_ref().map(|(input, _)| input.sha256()),
                    state_sha(&position)?,
                    description.cpu_profile_sha256.as_str(),
                    job.max_depth,
                    job.max_nodes,
                    "propose",
                ))?;
                let decision = if native_calls {
                    let analyzed = driver.analyze_native(
                        PalsNativeCaptureContext {
                            position: position.clone(),
                            opening: opening.clone(),
                            game: game.clone(),
                            actual_moves: moves.clone(),
                            sequence: state.sequence,
                            max_bytes: output.native_credit(),
                        },
                        job,
                        cancel,
                    );
                    // Always preserve prepared/raw/rejected events before
                    // propagating the native search error or attempting a move.
                    let retained = driver.take_native_trace().and_then(|trace| {
                        state.take_native(&mut output, trace, job, &mut native_secondary)
                    });
                    native_after_retention(
                        analyzed,
                        retained,
                        NativeRetentionStage::Recovery,
                        &mut native_secondary,
                    )?
                } else {
                    let (input, prepared) = root_capture
                        .as_ref()
                        .ok_or_else(|| invalid("CPU root capture missing"))?;
                    driver.analyze(&position, input, prepared, job, cancel)?
                };
                if !native_calls {
                    account(&mut state, job, decision.nodes)?;
                }
                let Some(movement) = decision.best_move else {
                    return Err(invalid(
                        "own source returned no move for an ongoing Rules state",
                    ));
                };
                let (_, legality) = replay_line(&position, &decision.pv)?;
                let sequence = state.next()?;
                let raw_sha=output.json("observations.jsonl",&serde_json::json!({"domain":"rz-own-analysis/1","sequence":sequence,"input_sha256":root_capture.as_ref().map(|(input,_)|input.sha256()),"task_sha256":task_sha,"raw":decision.raw,"line_legality_sha256":legality,"genealogy":decision.genealogy,"actual_native_call_inputs":native_calls}))?;
                if !native_calls && decision.completed_estimate && decision.completed_depth > 0 {
                    let input = &root_capture
                        .as_ref()
                        .ok_or_else(|| invalid("CPU label input missing"))?
                        .0;
                    let row = owned_label(
                        input,
                        &description,
                        CpuLabelEvidence {
                            task_sha,
                            raw_sha: raw_sha.clone(),
                            depth: decision.completed_depth,
                            nodes: decision.nodes,
                            sequence,
                            movement,
                            counterexample: None,
                        },
                    )?;
                    state.add_row(&mut output, row)?;
                }
                let record = decision.genealogy.first();
                state.publish(
                    &mut output,
                    PublicObservation {
                        position: &position,
                        hash: raw_sha,
                        kind: RecordKind::Proposal,
                        line: &decision.pv,
                        value: record
                            .and_then(|r| r.raw_value)
                            .filter(|_| decision.completed_estimate),
                        completed_depth: decision.completed_depth,
                        scope: decision
                            .completed_estimate
                            .then_some(CpuScoreScope::CompletedIteration),
                    },
                )?;
                if cancel.load(Ordering::Acquire) {
                    end = Some((PalsOutcome::Unknown, PalsGameEnd::UserStop));
                    break;
                }
                if Instant::now() >= deadline {
                    end = Some((PalsOutcome::Unknown, PalsGameEnd::WallTimeLimit));
                    break;
                }
                if !native_calls
                    && config.collect_conditional_repair
                    && decision.pv.len() >= 2
                    && config.max_total_nodes > state.nodes
                    && Instant::now() < deadline
                    && !cancel.load(Ordering::Acquire)
                {
                    conditional(
                        &config,
                        &mut state,
                        &mut output,
                        &position,
                        opening,
                        &game,
                        &moves,
                        &decision.pv,
                        &description,
                        &conditional_description,
                        &mut conditional_cpu,
                        deadline,
                        cancel,
                    )?;
                }
                if cancel.load(Ordering::Acquire) {
                    end = Some((PalsOutcome::Unknown, PalsGameEnd::UserStop));
                    break;
                }
                if Instant::now() >= deadline {
                    end = Some((PalsOutcome::Unknown, PalsGameEnd::WallTimeLimit));
                    break;
                }
                append_san(&mut movetext, &position, movement)?;
                let before = state_sha(&position)?;
                position.make_move(movement).map_err(rules)?;
                moves.push(movement);
                active
                    .as_mut()
                    .ok_or_else(|| invalid("active game missing after source move"))?
                    .advance(&position, &movetext, &moves)?;
                output.json("transitions.jsonl",&serde_json::json!({"game_id":game,"from":before,"move":Move16::pack(movement).map_err(|e|invalid(e.to_string()))?.bits(),"to":state_sha(&position)?,"actual":true}))?;
            }
            let (outcome, ending) = end
                .ok_or_else(|| invalid("finite game ended without a termination classification"))?;
            let result = PalsGameResult {
                game_id: game.clone(),
                outcome,
                ending,
                raw_evidence_sha256: canonical_sha256(&(
                    "rz-own-game-end/1",
                    game.as_str(),
                    state_sha(&position)?,
                    &moves.iter().map(ToString::to_string).collect::<Vec<_>>(),
                    outcome,
                    ending,
                ))?,
            };
            result.validate()?;
            active
                .as_mut()
                .ok_or_else(|| invalid("active game missing at actual end"))?
                .resolved = Some(result.clone());
            state.assignments.insert(game.clone(), opening.split);
            state.results.push(result.clone());
            output.json("game-results.jsonl", &result)?;
            active
                .as_mut()
                .ok_or_else(|| invalid("active game missing after result write"))?
                .result_written = true;
            // Each iteration releases the input borrow before mutating the
            // sequence/rows; no second Vec of every sealed input is allocated.
            let input_end = state.inputs.len();
            let mut input_index = input_start;
            while input_index < input_end {
                let captured = state.inputs[input_index].clone();
                input_index += 1;
                let input = captured.input;
                let sequence = state.next()?;
                let future_label = if captured.outcome_eligible {
                    Some(PalsFutureLabel {
                        observed_sequence: sequence,
                        provenance: PalsTargetProvenance::ActualGame {
                            result: result.clone(),
                        },
                        policy: None,
                        value_wdl: outcome_wdl(outcome, input.snapshot().white_to_move),
                        white_to_move: input.snapshot().white_to_move,
                        counterexample: None,
                        verifier_tasks: None,
                        supersedes_label_sha256: None,
                    })
                } else {
                    None
                };
                state.add_row(
                    &mut output,
                    PalsLearningRecord {
                        input,
                        future_label,
                        verifier_private: None,
                    },
                )?;
            }
            let pgn = game_pgn(&game, opening, &description, &movetext, &result);
            output.write("games.pgn", pgn.as_bytes(), false)?;
            active
                .as_mut()
                .ok_or_else(|| invalid("active game missing after PGN write"))?
                .pgn_written = true;
            active = None;
        }
        let split = PalsDatasetSplit {
            games: state.assignments.clone(),
        };
        let audit = registry.audit(&state.rows, &split)?;
        output.json("split.jsonl", &split)?;
        output.json("audit.jsonl", &audit)?;
        Ok(())
    })();
    if let Err(error) = run_result {
        failure = Some(native_secondary.failure_text(error));
        if let Some(active) = active.take() {
            let fallback = active
                .terminal
                .unwrap_or((PalsOutcome::Unknown, PalsGameEnd::InfrastructureFailure));
            let result = if let Some(result) = active.resolved {
                result
            } else {
                PalsGameResult {
                    game_id: active.game.clone(),
                    outcome: fallback.0,
                    ending: fallback.1,
                    raw_evidence_sha256: canonical_sha256(&(
                        "rz-own-failed-game/1",
                        active.game.as_str(),
                        active.position_sha.as_str(),
                        &active.moves,
                        &failure,
                    ))?,
                }
            };
            let pgn = game_pgn(
                &active.game,
                &active.opening,
                &description,
                &active.movetext,
                &result,
            );
            if !active.pgn_written {
                let name = if output.failed_artifacts.contains("games.pgn") {
                    "recovered-games.pgn"
                } else {
                    "games.pgn"
                };
                let _ = output.write(name, pgn.as_bytes(), true);
            }
            if !active.result_written {
                let name = if output.failed_artifacts.contains("game-results.jsonl") {
                    "recovered-game-results.jsonl"
                } else {
                    "game-results.jsonl"
                };
                let _ = output.final_json(name, &result);
            }
            if !state.results.iter().any(|r| r.game_id == result.game_id) {
                state.results.push(result);
            }
        }
        let _=output.final_json("failures.jsonl",&serde_json::json!({"sequence":state.sequence,"cause":failure,"nodes":state.nodes,"rows":state.rows.len(),"inputs":state.inputs.len(),"outcome_targets_for_unfinished_game":"masked"}));
    }
    let cleanup_until = Instant::now()
        .checked_add(Duration::from_secs(30))
        .ok_or_else(|| invalid("collector cleanup deadline overflow"))?;
    let native_finish = match driver.finish_collection(cleanup_until) {
        Ok(value) => {
            if let Some(reason) = value
                .as_ref()
                .and_then(|v| v.get("_collection_failure"))
                .and_then(|v| v.as_str())
            {
                failure = Some(failure_text(format!(
                    "native finish failed: {reason}; prior={failure:?}"
                )));
            }
            value
        }
        Err(error) => {
            failure = Some(failure_text(format!(
                "native finish failed: {error}; prior={failure:?}"
            )));
            None
        }
    };
    if let Err(error) = actual_binary.validate_stability() {
        failure = Some(failure_text(format!(
            "collector executable stability failed: {error}; prior={failure:?}"
        )));
    }
    if let Some(handle) = &producer {
        let checked = driver
            .checked_producer_owner()
            .and_then(|source| {
                source.ok_or_else(|| invalid("producer lost checked source after collection"))
            })
            .and_then(|owner| handle.verify_source(owner.source()));
        if let Err(error) = checked {
            failure = Some(failure_text(format!(
                "producer source stability failed: {error}; prior={failure:?}"
            )));
        }
        let split = PalsDatasetSplit {
            games: state.assignments.clone(),
        };
        if let Err(error) = handle.close(
            &mut output,
            &state.rows,
            &split,
            registry,
            failure.is_none(),
        ) {
            failure = Some(failure_text(format!(
                "producer evidence/envelope preservation failed: {error}; prior={failure:?}"
            )));
        }
    }
    let audit = if failure.is_none() {
        Some(registry.audit(
            &state.rows,
            &PalsDatasetSplit {
                games: state.assignments.clone(),
            },
        )?)
    } else {
        None
    };
    if let Ok(model) = full_line::model_config(&description)
        && !model.profile.is_legacy()
    {
        match full_line::coverage(&state.rows, model.profile)
            .and_then(|coverage| output.final_json("target-coverage.v2.jsonl", &coverage))
        {
            Ok(()) => {}
            Err(error) => {
                failure = Some(failure_text(format!(
                    "target coverage preservation failed: {error}; prior={failure:?}"
                )))
            }
        }
    }
    let artifacts = match output.artifacts() {
        Ok(value) => value,
        Err(error) => {
            failure = Some(failure_text(format!(
                "artifact preservation failed: {error}; prior={failure:?}"
            )));
            BTreeMap::new()
        }
    };
    let receipt = PalsCollectionReceipt {
        version: PALS_COLLECT_VERSION.into(),
        run_id: config.run_id.clone(),
        source: description,
        config,
        complete: failure.is_none(),
        failure,
        games_finished: state.results.len() as u32,
        cpu_nodes: state.nodes,
        cpu_jobs: state.jobs,
        cpu_work_observation_incomplete: state.work_incomplete,
        wall_time_ms: started.elapsed().as_millis().min(u64::MAX as u128) as u64,
        data_output_bytes: artifacts.values().map(|a| a.bytes).sum(),
        reserved_output_bytes: output.bytes,
        receipt_reserve_bytes: RECEIPT_RESERVE,
        results: state.results,
        audit,
        artifacts,
        actual_training_executed: false,
        external_teacher_used: false,
        native_finish,
    };
    output.finish(&receipt)?;
    driver.receipt_preserved();
    Ok(receipt)
}

fn append_san(
    text: &mut String,
    position: &Position,
    movement: BoardMove,
) -> Result<(), ArenaError> {
    if position.side_to_move() == Color::White {
        text.push_str(&format!("{}. ", position.fullmove_number()));
    } else if text.is_empty() {
        text.push_str(&format!("{}... ", position.fullmove_number()));
    }
    text.push_str(&crate::pgn::san(position, movement)?);
    text.push(' ');
    Ok(())
}
fn game_pgn(
    game: &str,
    opening: &PalsCollectionOpening,
    source: &PalsCollectionSourceDescription,
    movetext: &str,
    result: &PalsGameResult,
) -> String {
    let outcome = match result.outcome {
        PalsOutcome::WhiteWin => "1-0",
        PalsOutcome::BlackWin => "0-1",
        PalsOutcome::Draw => "1/2-1/2",
        PalsOutcome::Unknown => "*",
    };
    let engine = format!("{}-{}", source.mode, &source.implementation_sha256[..16]);
    let mut pgn = format!(
        "[Event \"RoveZero own-data collection\"]\n[Round \"{game}\"]\n[White \"{engine}\"]\n[Black \"{engine}\"]\n[Result \"{outcome}\"]\n[Termination \"{:?}\"]\n",
        result.ending
    );
    if let Some(fen) = &opening.initial_fen {
        pgn.push_str(&format!("[SetUp \"1\"]\n[FEN \"{fen}\"]\n"));
    }
    pgn.push_str(&format!("\n{movetext}{outcome}\n\n"));
    pgn
}
fn outcome_wdl(outcome: PalsOutcome, white: bool) -> Option<[f64; 3]> {
    match outcome {
        PalsOutcome::Unknown => None,
        PalsOutcome::Draw => Some([0.0, 1.0, 0.0]),
        PalsOutcome::WhiteWin if white => Some([1.0, 0.0, 0.0]),
        PalsOutcome::BlackWin if !white => Some([1.0, 0.0, 0.0]),
        _ => Some([0.0, 0.0, 1.0]),
    }
}

#[allow(clippy::too_many_arguments)]
fn conditional(
    config: &PalsCollectionConfig,
    state: &mut Collection,
    output: &mut Output,
    root: &Position,
    opening: &PalsCollectionOpening,
    game: &str,
    actual: &[BoardMove],
    proposal: &[BoardMove],
    source: &PalsCollectionSourceDescription,
    cpu_source: &PalsCollectionSourceDescription,
    cpu: &mut CpuEngine,
    deadline: Instant,
    cancel: &AtomicBool,
) -> Result<(), ArenaError> {
    let mut divergence = root.clone();
    divergence.make_move(proposal[0]).map_err(rules)?;
    if termination(&divergence)?.is_some() {
        return Ok(());
    }
    let alternatives = divergence
        .legal_moves()
        .into_iter()
        .filter(|mv| *mv != proposal[1])
        .collect::<Vec<_>>();
    if alternatives.is_empty() {
        return Ok(());
    }
    let mut history = actual.to_vec();
    history.push(proposal[0]);
    let (input, _prepared) = state.capture(
        output,
        InputCapture {
            position: &divergence,
            opening,
            game,
            position_command: position_command(opening, &history),
            actual_moves: actual,
            role: PalsDataRole::Critic,
            source,
            outcome_eligible: false,
            prefix: &proposal[..1],
            proposal,
            counterexample: None,
            deadline,
            cancel,
        },
    )?;
    let job = job_limits(config, state, deadline)?;
    let challenged = canonical_sha256(&(
        "rz-pals-challenged-line/1",
        state_sha(root)?,
        pack(proposal)?,
    ))?;
    let task = canonical_sha256(&(
        "rz-own-conditional-task/1",
        input.sha256(),
        &challenged,
        pack(&alternatives)?,
        job.max_depth,
        job.max_nodes,
        &cpu_source.cpu_profile_sha256,
    ))?;
    let report = cpu
        .analyze_root_moves(&divergence, &alternatives, job, cancel)
        .map_err(|e| invalid(e.to_string()))?;
    account(state, job, report.nodes)?;
    let Some(response) = report.best_move else {
        return Ok(());
    };
    let response_line = if report.pv.is_empty() {
        vec![response]
    } else {
        report.pv.clone()
    };
    let (_, legality) = replay_line(&divergence, &response_line)?;
    let sequence = state.next()?;
    let raw=output.json("observations.jsonl",&serde_json::json!({"kind":"cpu_conditional_counterexample","sequence":sequence,"input":input.sha256(),"challenged_line":challenged,"raw":cpu_raw(&report),"legality":legality,"validity":"not_examined","claim":"conditional continuation only"}))?;
    let mut counter = PalsCounterexampleTarget {
        challenged_line_sha256: challenged.clone(),
        divergence_ply: 1,
        response_line: pack(&response_line)?,
        repair_line: None,
        validity: PalsConditionalValidity::NotExamined,
        input_revision: input.snapshot().input_revision,
        legality_evidence_sha256: legality,
    };
    state.publish(
        output,
        PublicObservation {
            position: &divergence,
            hash: raw.clone(),
            kind: RecordKind::Counterexample,
            line: &response_line,
            value: (report.score_scope == CpuScoreScope::CompletedIteration)
                .then_some(report.score),
            completed_depth: report.completed_depth,
            scope: Some(report.score_scope),
        },
    )?;
    if report.score_scope == CpuScoreScope::CompletedIteration && report.completed_depth > 0 {
        state.add_row(
            output,
            owned_label(
                &input,
                cpu_source,
                CpuLabelEvidence {
                    task_sha: task,
                    raw_sha: raw,
                    depth: report.completed_depth,
                    nodes: report.nodes,
                    sequence,
                    movement: response,
                    counterexample: Some(counter.clone()),
                },
            )?,
        )?;
    }
    if Instant::now() >= deadline
        || cancel.load(Ordering::Acquire)
        || state.nodes >= config.max_total_nodes
    {
        return Ok(());
    }
    let mut repair_position = divergence.clone();
    repair_position.make_move(response).map_err(rules)?;
    if termination(&repair_position)?.is_some() {
        return Ok(());
    }
    history.push(response);
    let prefix = [proposal[0], response];
    let (repair_input, _prepared) = state.capture(
        output,
        InputCapture {
            position: &repair_position,
            opening,
            game,
            position_command: position_command(opening, &history),
            actual_moves: actual,
            role: PalsDataRole::Proposer,
            source,
            outcome_eligible: false,
            prefix: &prefix,
            proposal,
            counterexample: Some(&response_line),
            deadline,
            cancel,
        },
    )?;
    let job = job_limits(config, state, deadline)?;
    let report = cpu
        .analyze(&repair_position, job, cancel)
        .map_err(|e| invalid(e.to_string()))?;
    account(state, job, report.nodes)?;
    let Some(repair_move) = report.best_move else {
        return Ok(());
    };
    let repair_line = if report.pv.is_empty() {
        vec![repair_move]
    } else {
        report.pv.clone()
    };
    let (_, repair_legality) = replay_line(&repair_position, &repair_line)?;
    let sequence = state.next()?;
    let task = canonical_sha256(&(
        "rz-own-repair-task/1",
        repair_input.sha256(),
        &challenged,
        job.max_depth,
        job.max_nodes,
        &cpu_source.cpu_profile_sha256,
    ))?;
    let raw=output.json("observations.jsonl",&serde_json::json!({"kind":"cpu_conditional_repair","sequence":sequence,"input":repair_input.sha256(),"challenged_line":challenged,"response":pack(&response_line)?,"repair":pack(&repair_line)?,"raw":cpu_raw(&report),"legality":repair_legality,"validity":"not_examined"}))?;
    if report.score_scope == CpuScoreScope::CompletedIteration && report.completed_depth > 0 {
        state.add_row(
            output,
            owned_label(
                &repair_input,
                cpu_source,
                CpuLabelEvidence {
                    task_sha: task,
                    raw_sha: raw.clone(),
                    depth: report.completed_depth,
                    nodes: report.nodes,
                    sequence,
                    movement: repair_move,
                    counterexample: None,
                },
            )?,
        )?;
    }
    state.publish(
        output,
        PublicObservation {
            position: &repair_position,
            hash: raw,
            kind: RecordKind::Repair,
            line: &repair_line,
            value: (report.score_scope == CpuScoreScope::CompletedIteration)
                .then_some(report.score),
            completed_depth: report.completed_depth,
            scope: Some(report.score_scope),
        },
    )?;
    // This is a lineage update, not a strategic success label. Finding a legal
    // repair alone does not prove that it refuted the counterexample.
    counter.repair_line = Some(pack(&repair_line)?);
    output.json("genealogy.jsonl",&serde_json::json!({"input_sha256":input.sha256(),"observed_sequence":sequence,"conditional":counter}))?;
    Ok(())
}

/// The source registry is constructed from verified loaded executable bytes,
/// not from declarations in returned learning rows.
pub fn own_collection_registry(
    driver: &dyn PalsCollectionDriver,
) -> Result<PalsOwnedSources, ArenaError> {
    let description = driver.description();
    checked_source(description)?;
    if !matches!(
        &description.source,
        PalsInputSource::OwnCpu {
            model_weights_sha256: None,
            ..
        }
    ) {
        return Err(invalid(
            "automatic source enrollment is only available for verified weightless CPU; NN assets need an independent verified registry",
        ));
    }
    let actual = file_sha(&std::env::current_exe().map_err(io)?, 256 * 1024 * 1024)?;
    if actual != description.implementation_sha256 {
        return Err(invalid("loaded source implementation hash mismatch"));
    }
    let registry = PalsOwnedSources {
        cpu_binary_sha256: BTreeSet::from([actual]),
        input_sources: BTreeSet::from([description.source.clone()]),
    };
    registry.validate()?;
    Ok(registry)
}

#[cfg(all(test, feature = "pals-collection"))]
mod tests {
    use super::*;
    std::thread_local! {pub(super) static FAIL_ARTIFACT:std::cell::Cell<Option<(&'static str,usize)>>=const {std::cell::Cell::new(None)};}
    static NEXT_TEMP: AtomicU64 = AtomicU64::new(1);
    static EXECUTABLE_FIXTURE_ADMISSION: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn admit_executable_fixture() -> std::sync::MutexGuard<'static, ()> {
        // These tests independently hash the actual debug test executable at
        // the driver, registry and collection boundaries. Admit one such
        // fixture before any of its clocks start, rather than letting the
        // test harness turn every 10-second collection into a competing
        // executable-hashing workload. This changes neither production
        // admission/deadlines nor any explicit timeout/cancellation case.
        // The mutex owns no source state, so a failed test may release its
        // admission without suppressing the next test's independent checks.
        EXECUTABLE_FIXTURE_ADMISSION
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
    struct OwnedTemp(PathBuf);
    impl OwnedTemp {
        fn new() -> Self {
            let path = std::env::temp_dir().join(format!(
                "rz-pals-collector-test-{}-{}",
                std::process::id(),
                NEXT_TEMP.fetch_add(1, Ordering::Relaxed)
            ));
            std::fs::create_dir(&path).unwrap();
            Self(path)
        }
    }
    impl Drop for OwnedTemp {
        fn drop(&mut self) {
            let Ok(resolved) = self.0.canonicalize() else {
                return;
            };
            let Ok(temp) = std::env::temp_dir().canonicalize() else {
                return;
            };
            if resolved.parent() == Some(temp.as_path())
                && resolved
                    .file_name()
                    .is_some_and(|v| v.to_string_lossy().starts_with("rz-pals-collector-test-"))
            {
                let _ = std::fs::remove_dir_all(&resolved);
            }
        }
    }
    fn small_config() -> PalsCollectionConfig {
        PalsCollectionConfig {
            max_plies: 1,
            cpu_depth: 1,
            nodes_per_job: 256,
            max_total_nodes: 1024,
            max_wall_time_ms: 10_000,
            max_output_bytes: 1024 * 1024,
            exploration_plies: 0,
            collect_conditional_repair: false,
            ..PalsCollectionConfig::default()
        }
    }
    fn driver() -> OwnCpuCollectionDriver {
        OwnCpuCollectionDriver::new(CpuConfig {
            tt_entries: 64,
            quiescence_ply: 2,
            ..CpuConfig::default()
        })
        .unwrap()
    }
    fn rows<T: serde::de::DeserializeOwned>(path: &Path) -> Vec<T> {
        std::fs::read_to_string(path)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect()
    }
    fn label_chain_fixture() -> (
        OwnedTemp,
        Output,
        Collection,
        PalsFrozenInput,
        PalsCollectionSourceDescription,
    ) {
        let temp = OwnedTemp::new();
        let executable_path = temp.0.join("synthetic-executable");
        std::fs::write(&executable_path, b"bounded label-chain fixture").unwrap();
        let executable = VerifiedExecutableIdentity::open(&executable_path, 1024).unwrap();
        let cpu = CpuEngine::new(CpuConfig {
            tt_entries: 64,
            ..CpuConfig::default()
        })
        .unwrap();
        let description = source_description_from_verified(
            &cpu,
            "own-cpu-bootstrap",
            serde_json::Value::Null,
            &executable,
        )
        .unwrap();
        let mut output = Output::new(&temp.0, "label-chain", 1024 * 1024).unwrap();
        let mut state = Collection {
            sequence: 0,
            nodes: 0,
            jobs: 0,
            work_incomplete: false,
            rows: Vec::new(),
            latest_labels: BTreeMap::new(),
            inputs: Vec::new(),
            assignments: BTreeMap::new(),
            results: Vec::new(),
            public: Vec::new(),
            role_records: Vec::new(),
        };
        let (input, _) = state
            .capture(
                &mut output,
                InputCapture {
                    position: &Position::startpos(),
                    opening: &PalsCollectionOpening::default(),
                    game: "label-chain-game",
                    position_command: position_command(&PalsCollectionOpening::default(), &[]),
                    actual_moves: &[],
                    role: PalsDataRole::Proposer,
                    source: &description,
                    outcome_eligible: true,
                    prefix: &[],
                    proposal: &[],
                    counterexample: None,
                    deadline: Instant::now() + Duration::from_secs(10),
                    cancel: &AtomicBool::new(false),
                },
            )
            .unwrap();
        (temp, output, state, input, description)
    }
    #[test]
    fn native_analysis_primary_survives_actual_trace_credit_rejection() {
        let (_temp, mut output, mut state, _, _) = label_chain_fixture();
        let sequence = state.sequence;
        let inputs = state.inputs.len();
        let bytes = output.bytes;
        let mut secondary = NativeRetentionFailures::default();
        let trace = PalsNativeTrace {
            sequence: sequence + 1,
            reserved_bytes: output.native_credit() + 1,
            ..PalsNativeTrace::default()
        };
        let limits = job_limits(
            &small_config(),
            &state,
            Instant::now() + Duration::from_secs(10),
        )
        .unwrap();
        let retained = state.take_native(&mut output, trace, limits, &mut secondary);
        assert!(matches!(&retained, Err(ArenaError::Budget(_))));
        let primary = invalid("PhysicalCompletionUnknown: original native analysis");
        let expected = primary.to_string();
        let error = native_after_retention::<()>(
            Err(primary),
            retained,
            NativeRetentionStage::Recovery,
            &mut secondary,
        )
        .unwrap_err();
        assert_eq!(error.to_string(), expected);
        let cause = secondary.failure_text(error);
        assert!(cause.starts_with(&expected));
        assert!(cause.contains("native-retention-secondary[trace-recovery]"));
        assert!(cause.contains("delegated output credit"));
        output
            .final_json("failures.jsonl", &serde_json::json!({"cause":cause}))
            .unwrap();
        let persisted: Vec<serde_json::Value> = rows(&output.directory.join("failures.jsonl"));
        assert_eq!(persisted[0]["cause"], cause);
        assert_eq!(state.sequence, sequence);
        assert_eq!(state.inputs.len(), inputs);
        // The rejected delegated credit was never charged or silently raised.
        assert_eq!(
            output.bytes - bytes,
            std::fs::metadata(output.directory.join("failures.jsonl"))
                .unwrap()
                .len()
        );
    }
    #[test]
    fn native_observer_primary_survives_io_failure_with_raw_sibling_attempted() {
        let (_temp, mut output, mut state, _, _) = label_chain_fixture();
        let sequence = state.sequence;
        let inputs = state.inputs.len();
        let bytes = output.bytes;
        // An exact owned-directory collision fails the first open. The second
        // row must still be written by the production native_trace path.
        std::fs::create_dir(output.directory.join("native-events.jsonl")).unwrap();
        let raw = br#"{"raw":"retained-sibling"}"#.to_vec();
        let rejected = br#"{"event":"physically-completion-unknown"}"#.to_vec();
        let reserved = (raw.len() + rejected.len() + 2) as u64;
        let trace = PalsNativeTrace {
            sequence: sequence + 1,
            reserved_bytes: reserved,
            cpu_nodes: 2,
            cpu_jobs: 1,
            cpu_work_observation_incomplete: true,
            rows: vec![
                PalsNativeTraceRow {
                    artifact: "native-events.jsonl",
                    json: rejected,
                },
                PalsNativeTraceRow {
                    artifact: "native-raw-outputs.jsonl",
                    json: raw.clone(),
                },
            ],
            failure: Some("PhysicalCompletionUnknown: observer retained original".into()),
            ..PalsNativeTrace::default()
        };
        let limits = job_limits(
            &small_config(),
            &state,
            Instant::now() + Duration::from_secs(10),
        )
        .unwrap();
        let mut secondary = NativeRetentionFailures::default();
        let error = state
            .take_native(&mut output, trace, limits, &mut secondary)
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("PhysicalCompletionUnknown: observer retained original")
        );
        let cause = secondary.failure_text(error);
        assert!(cause.contains("native-retention-secondary[trace-persistence]"));
        assert!(output.failed_artifacts.contains("native-events.jsonl"));
        let mut expected = raw;
        expected.push(b'\n');
        assert_eq!(
            std::fs::read(output.directory.join("native-raw-outputs.jsonl")).unwrap(),
            expected
        );
        assert_eq!(output.bytes, bytes + reserved);
        assert_eq!(
            (state.nodes, state.jobs, state.work_incomplete),
            (2, 1, true)
        );
        assert_eq!(state.sequence, sequence);
        assert_eq!(state.inputs.len(), inputs);
    }

    #[test]
    fn native_search_return_artifact_preserves_raw_bytes_and_unused_dispatch_credit() {
        // The production output boundary, with synthetic rows only. This does
        // not manufacture a native search result or a causal action witness.
        let temp = OwnedTemp::new();
        let mut output = Output::new(&temp.0, "native-search-return", 1024 * 1024).unwrap();
        let before = output.bytes;
        let raw =
            br#"{"domain":"output-path-fixture","native_action_causal_bridge_observed":false}"#
                .to_vec();
        let reserved = raw.len() as u64 + 1 + 4096;
        let trace = PalsNativeTrace {
            reserved_bytes: reserved,
            rows: vec![PalsNativeTraceRow {
                artifact: "native-search-returns.jsonl",
                json: raw.clone(),
            }],
            ..PalsNativeTrace::default()
        };
        output.native_trace(&trace).unwrap();
        let mut expected = raw;
        expected.push(b'\n');
        assert_eq!(
            std::fs::read(output.directory.join("native-search-returns.jsonl")).unwrap(),
            expected
        );
        assert_eq!(output.bytes, before + reserved);
        assert!(output.files.contains_key("native-search-returns.jsonl"));
        assert!(output.failed_artifacts.is_empty());
    }

    #[test]
    fn native_search_return_open_failure_keeps_raw_sibling_and_charged_credit() {
        let temp = OwnedTemp::new();
        let mut output = Output::new(&temp.0, "native-search-return-failed", 1024 * 1024).unwrap();
        std::fs::create_dir(output.directory.join("native-search-returns.jsonl")).unwrap();
        let before = output.bytes;
        let raw = br#"{"scope":"synthetic-raw-sibling"}"#.to_vec();
        let closure = br#"{"scope":"synthetic-return-output-failure"}"#.to_vec();
        let reserved = (raw.len() + closure.len() + 2) as u64 + 4096;
        let trace = PalsNativeTrace {
            reserved_bytes: reserved,
            rows: vec![
                PalsNativeTraceRow {
                    artifact: "native-search-returns.jsonl",
                    json: closure,
                },
                PalsNativeTraceRow {
                    artifact: "native-raw-outputs.jsonl",
                    json: raw.clone(),
                },
            ],
            failure: Some("original search primary retained".into()),
            ..PalsNativeTrace::default()
        };
        assert!(output.native_trace(&trace).is_err());
        let mut expected = raw;
        expected.push(b'\n');
        assert_eq!(
            std::fs::read(output.directory.join("native-raw-outputs.jsonl")).unwrap(),
            expected
        );
        assert_eq!(output.bytes, before + reserved);
        assert!(
            output
                .failed_artifacts
                .contains("native-search-returns.jsonl")
        );
        assert_eq!(
            trace.failure.as_deref(),
            Some("original search primary retained")
        );
    }
    #[test]
    fn native_retention_failure_bounds_and_success_gate_keep_all_three_stages() {
        let mut secondary = NativeRetentionFailures::default();
        let error = native_after_retention::<()>(
            Err(ArenaError::Budget("original-primary-variant".into())),
            Err(invalid("journal-failed")),
            NativeRetentionStage::Journal,
            &mut secondary,
        )
        .unwrap_err();
        assert!(
            matches!(error, ArenaError::Budget(ref cause) if cause == "original-primary-variant")
        );
        let error = native_after_retention(
            Ok(7_u8),
            Err(ArenaError::Budget("trace-recovery-failed".into())),
            NativeRetentionStage::Recovery,
            &mut secondary,
        )
        .unwrap_err();
        assert!(matches!(error, ArenaError::Budget(ref cause) if cause == "trace-recovery-failed"));
        assert_eq!(
            native_after_retention(
                Ok(7_u8),
                Ok(()),
                NativeRetentionStage::Recovery,
                &mut secondary
            )
            .unwrap(),
            7
        );
        for stage in [
            NativeRetentionStage::Recovery,
            NativeRetentionStage::Persistence,
            NativeRetentionStage::Journal,
        ] {
            secondary.record(stage, &invalid("한글\\\"\n".repeat(10_000)));
        }
        assert!(
            secondary
                .notes
                .iter()
                .all(|note| bounded_json(note.as_ref().unwrap(), 1024, false).is_ok())
        );
        let cause =
            secondary.failure_text(format!("original-primary: {}", "큰\\\"\n".repeat(10_000)));
        assert!(bounded_json(&cause, 8192, false).is_ok());
        assert!(cause.starts_with("original-primary:"));
        for stage in ["trace-recovery", "trace-persistence", "producer-journal"] {
            assert!(cause.contains(&format!("native-retention-secondary[{stage}]")));
        }
        assert_eq!(cause.matches("message truncated; bytes=").count(), 4);
    }
    fn fixture_cpu_label(
        input: &PalsFrozenInput,
        description: &PalsCollectionSourceDescription,
        sequence: u64,
    ) -> PalsLearningRecord {
        owned_label(
            input,
            description,
            CpuLabelEvidence {
                task_sha: "a".repeat(64),
                raw_sha: "b".repeat(64),
                depth: 1,
                nodes: 1,
                sequence,
                movement: BoardMove::from_uci("e2e4").unwrap(),
                counterexample: None,
            },
        )
        .unwrap()
    }
    fn fixture_unknown_game_label(input: &PalsFrozenInput, sequence: u64) -> PalsLearningRecord {
        PalsLearningRecord {
            input: input.clone(),
            future_label: Some(PalsFutureLabel {
                observed_sequence: sequence,
                provenance: PalsTargetProvenance::ActualGame {
                    result: PalsGameResult {
                        game_id: input.snapshot().game_id.clone(),
                        outcome: PalsOutcome::Unknown,
                        ending: PalsGameEnd::PlyLimit,
                        raw_evidence_sha256: "c".repeat(64),
                    },
                },
                policy: None,
                value_wdl: None,
                white_to_move: input.snapshot().white_to_move,
                counterexample: None,
                verifier_tasks: None,
                supersedes_label_sha256: None,
            }),
            verifier_private: None,
        }
    }
    #[test]
    fn persisted_cpu_policy_is_superseded_by_the_whole_unknown_game_label() {
        let (_temp, mut output, mut state, input, description) = label_chain_fixture();
        let sequence = state.next().unwrap();
        let owned = fixture_cpu_label(&input, &description, sequence);
        let owned_digest = owned.label_digest().unwrap().unwrap();
        state.add_row(&mut output, owned.clone()).unwrap();
        let prior_index = state.latest_labels.clone();
        state
            .add_row(
                &mut output,
                PalsLearningRecord {
                    input: input.clone(),
                    future_label: None,
                    verifier_private: None,
                },
            )
            .unwrap();
        assert_eq!(state.latest_labels, prior_index);
        let sequence = state.next().unwrap();
        state
            .add_row(&mut output, fixture_unknown_game_label(&input, sequence))
            .unwrap();
        let persisted: Vec<PalsLearningRecord> = rows(&output.directory.join("records.jsonl"));
        assert_eq!(persisted, state.rows);
        assert_eq!(persisted[0], owned);
        assert!(persisted[0].future_label.as_ref().unwrap().policy.is_some());
        assert!(persisted[1].future_label.is_none());
        let latest = persisted[2].future_label.as_ref().unwrap();
        assert_eq!(latest.supersedes_label_sha256.as_ref(), Some(&owned_digest));
        assert!(latest.policy.is_none());
        assert!(latest.value_wdl.is_none());
        assert_eq!(
            state.latest_labels.get(input.sha256()),
            Some(&(sequence, persisted[2].label_digest().unwrap().unwrap()))
        );
        assert!(state.latest_labels.len() <= state.rows.len());
        let registry = PalsOwnedSources {
            cpu_binary_sha256: BTreeSet::from([description.implementation_sha256]),
            input_sources: BTreeSet::from([description.source]),
        };
        let split = PalsDatasetSplit {
            games: BTreeMap::from([(input.snapshot().game_id.clone(), PalsSplit::Train)]),
        };
        let (audit, view) = registry
            .audit_with_current_view(&persisted, &split)
            .unwrap();
        assert_eq!(audit.records, 3);
        assert_eq!(audit.labeled_records, 2);
        assert_eq!(view.indices(), &[2]);
        assert!(
            persisted[view.indices()[0]]
                .future_label
                .as_ref()
                .unwrap()
                .policy
                .is_none()
        );
        let mut two_roots = persisted;
        two_roots[2]
            .future_label
            .as_mut()
            .unwrap()
            .supersedes_label_sha256 = None;
        assert!(registry.current_training_view(&two_roots, &split).is_err());
    }
    #[test]
    fn collector_rejects_nonincreasing_or_unpersisted_label_predecessors() {
        let (_temp, mut output, mut state, input, description) = label_chain_fixture();
        let first_sequence = state.next().unwrap();
        let first = fixture_cpu_label(&input, &description, first_sequence);
        let first_digest = first.label_digest().unwrap().unwrap();
        let path = output.directory.join("records.jsonl");
        let mut unpersisted = first.clone();
        unpersisted
            .future_label
            .as_mut()
            .unwrap()
            .supersedes_label_sha256 = Some("d".repeat(64));
        assert!(state.add_row(&mut output, unpersisted).is_err());
        assert!(state.rows.is_empty());
        assert!(state.latest_labels.is_empty());
        assert!(!path.exists());
        state.add_row(&mut output, first.clone()).unwrap();
        let persisted = std::fs::read(&path).unwrap();
        let prior_index = state.latest_labels.clone();
        for sequence in [first_sequence, first_sequence - 1] {
            let mut invalid_sequence = first.clone();
            invalid_sequence
                .future_label
                .as_mut()
                .unwrap()
                .observed_sequence = sequence;
            assert!(state.add_row(&mut output, invalid_sequence).is_err());
            assert_eq!(state.rows, vec![first.clone()]);
            assert_eq!(state.latest_labels, prior_index);
            assert_eq!(std::fs::read(&path).unwrap(), persisted);
        }
        let next_sequence = state.next().unwrap();
        let mut successor = fixture_unknown_game_label(&input, next_sequence);
        successor
            .future_label
            .as_mut()
            .unwrap()
            .supersedes_label_sha256 = Some("d".repeat(64));
        assert!(state.add_row(&mut output, successor.clone()).is_err());
        assert_eq!(state.rows, vec![first.clone()]);
        assert_eq!(state.latest_labels, prior_index);
        assert_eq!(std::fs::read(&path).unwrap(), persisted);
        successor
            .future_label
            .as_mut()
            .unwrap()
            .supersedes_label_sha256 = Some(first_digest.clone());
        state.add_row(&mut output, successor.clone()).unwrap();
        assert_eq!(state.rows, vec![first, successor]);
        let prior_index = state.latest_labels.clone();
        let persisted = std::fs::read(&path).unwrap();
        let mut stale = fixture_unknown_game_label(&input, state.next().unwrap());
        stale.future_label.as_mut().unwrap().supersedes_label_sha256 = Some(first_digest);
        assert!(state.add_row(&mut output, stale).is_err());
        assert_eq!(state.rows.len(), 2);
        assert_eq!(state.latest_labels, prior_index);
        assert_eq!(std::fs::read(&path).unwrap(), persisted);
    }
    #[test]
    fn partial_label_append_does_not_advance_rows_or_predecessor_index() {
        for has_prior in [false, true] {
            let (_temp, mut output, mut state, input, description) = label_chain_fixture();
            let sequence = state.next().unwrap();
            let first = fixture_cpu_label(&input, &description, sequence);
            let path = output.directory.join("records.jsonl");
            let next = if has_prior {
                state.add_row(&mut output, first).unwrap();
                fixture_unknown_game_label(&input, state.next().unwrap())
            } else {
                first
            };
            let prior_rows = state.rows.clone();
            let prior_index = state.latest_labels.clone();
            let prior_bytes = if path.exists() {
                std::fs::read(&path).unwrap()
            } else {
                Vec::new()
            };
            let mut linked = next.clone();
            if let Some((_, digest)) = prior_index.get(input.sha256()) {
                linked
                    .future_label
                    .as_mut()
                    .unwrap()
                    .supersedes_label_sha256 = Some(digest.clone());
            }
            let attempted = serde_json::to_vec(&linked).unwrap();
            FAIL_ARTIFACT.with(|flag| flag.set(Some(("records.jsonl", 13))));
            assert!(state.add_row(&mut output, next).is_err());
            assert_eq!(state.rows, prior_rows);
            assert_eq!(state.latest_labels, prior_index);
            assert!(output.failed_artifacts.contains("records.jsonl"));
            let bytes = std::fs::read(path).unwrap();
            assert_eq!(bytes.len(), prior_bytes.len() + 13);
            assert_eq!(&bytes[..prior_bytes.len()], prior_bytes);
            assert_eq!(&bytes[prior_bytes.len()..], &attempted[..13]);
        }
    }
    #[test]
    fn conditional_description_reuses_the_owned_verified_executable_identity() {
        let output = OwnedTemp::new();
        let path = output.0.join("synthetic-executable");
        let bytes = b"bounded fixture executable bytes";
        std::fs::write(&path, bytes).unwrap();
        let executable = VerifiedExecutableIdentity::open(&path, 1024).unwrap();
        let cpu = CpuEngine::new(CpuConfig {
            tt_entries: 64,
            ..CpuConfig::default()
        })
        .unwrap();
        let original = source_description_from_verified(
            &cpu,
            "own-cpu-bootstrap",
            serde_json::Value::Null,
            &executable,
        )
        .unwrap();
        let conditional = source_description_from_verified(
            &cpu,
            "own-cpu-conditional-verification",
            serde_json::Value::Null,
            &executable,
        )
        .unwrap();
        let actual = format!("{:x}", Sha256::digest(bytes));
        assert_eq!(original.implementation_sha256, actual);
        assert_eq!(conditional.implementation_sha256, actual);
        assert_ne!(original.cpu_profile_sha256, conditional.cpu_profile_sha256);
        assert_eq!(executable.metadata.len(), bytes.len() as u64);
        assert!(matches!(&conditional.source, PalsInputSource::OwnCpu {
            cpu_binary_sha256, model_weights_sha256: None, ..
        } if cpu_binary_sha256 == &actual));
        executable.validate_stability().unwrap();
        assert!(VerifiedExecutableIdentity::open(&path, bytes.len() as u64 - 1).is_err());
    }
    #[cfg(unix)]
    #[test]
    fn verified_executable_identity_rejects_growth_and_same_size_writes() {
        let output = OwnedTemp::new();
        for (name, changed) in [
            ("growing", b"longer".as_slice()),
            ("same-size", b"edit".as_slice()),
        ] {
            let path = output.0.join(name);
            std::fs::write(&path, b"base").unwrap();
            let executable = VerifiedExecutableIdentity::open(&path, 1024).unwrap();
            std::fs::write(&path, changed).unwrap();
            let writer = OpenOptions::new().write(true).open(&path).unwrap();
            writer
                .set_modified(std::time::SystemTime::UNIX_EPOCH + Duration::from_secs(1))
                .unwrap();
            writer.sync_all().unwrap();
            assert!(executable.validate_stability().is_err());
        }
    }
    #[cfg(unix)]
    #[test]
    fn verified_executable_identity_rejects_equal_length_path_replacement() {
        let output = OwnedTemp::new();
        let path = output.0.join("original");
        let replacement = output.0.join("replacement");
        std::fs::write(&path, b"base").unwrap();
        let executable = VerifiedExecutableIdentity::open(&path, 1024).unwrap();
        std::fs::write(&replacement, b"edit").unwrap();
        let writer = OpenOptions::new().write(true).open(&replacement).unwrap();
        writer
            .set_modified(executable.metadata.modified().unwrap())
            .unwrap();
        writer.sync_all().unwrap();
        drop(writer);
        std::fs::rename(&replacement, &path).unwrap();
        let named = std::fs::metadata(&path).unwrap();
        assert_eq!(named.len(), executable.metadata.len());
        assert_eq!(
            named.modified().unwrap(),
            executable.metadata.modified().unwrap()
        );
        assert_ne!(named.ino(), executable.metadata.ino());
        assert!(executable.validate_stability().is_err());
    }
    #[cfg(windows)]
    #[test]
    fn verified_executable_identity_denies_windows_write_handles() {
        let output = OwnedTemp::new();
        let path = output.0.join("protected");
        std::fs::write(&path, b"base").unwrap();
        let executable = VerifiedExecutableIdentity::open(&path, 1024).unwrap();
        assert!(OpenOptions::new().write(true).open(&path).is_err());
        executable.validate_stability().unwrap();
    }
    #[test]
    fn collection_rejects_a_consistently_forged_driver_and_registry_binary_digest() {
        let _admission = admit_executable_fixture();
        let output = OwnedTemp::new();
        let path = output.0.join("unregistered-fixture-executable");
        std::fs::write(&path, b"not the actual collector executable").unwrap();
        let executable = VerifiedExecutableIdentity::open(&path, 1024).unwrap();
        let engine = CpuEngine::new(CpuConfig {
            tt_entries: 64,
            ..CpuConfig::default()
        })
        .unwrap();
        let description = source_description_from_verified(
            &engine,
            "own-cpu-bootstrap",
            serde_json::Value::Null,
            &executable,
        )
        .unwrap();
        let registry = PalsOwnedSources {
            cpu_binary_sha256: BTreeSet::from([description.implementation_sha256.clone()]),
            input_sources: BTreeSet::from([description.source.clone()]),
        };
        let mut cpu = OwnCpuCollectionDriver {
            engine,
            description,
        };
        let error = collect_pals_own_data(
            small_config(),
            &output.0,
            &mut cpu,
            &registry,
            &AtomicBool::new(false),
        )
        .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("collector executable differs from registered CPU implementation")
        );
        assert!(!output.0.join(small_config().run_id).exists());
    }
    #[test]
    fn actual_cpu_collection_seals_native_inputs_and_masks_ply_limit() {
        let _admission = admit_executable_fixture();
        let output = OwnedTemp::new();
        let mut cpu = driver();
        let registry = own_collection_registry(&cpu).unwrap();
        let receipt = collect_pals_own_data(
            small_config(),
            &output.0,
            &mut cpu,
            &registry,
            &AtomicBool::new(false),
        )
        .unwrap();
        assert!(receipt.complete, "{:?}", receipt.failure);
        assert_eq!(receipt.results.len(), 1);
        assert_eq!(receipt.results[0].outcome, PalsOutcome::Unknown);
        assert_eq!(receipt.results[0].ending, PalsGameEnd::PlyLimit);
        let run = output.0.join(&receipt.run_id);
        let records: Vec<PalsLearningRecord> = rows(&run.join("records.jsonl"));
        assert!(
            records
                .iter()
                .any(|r| r.future_label.as_ref().is_some_and(|l| l.policy.is_some()))
        );
        for record in &records {
            record.validate().unwrap();
            assert!(
                record
                    .future_label
                    .as_ref()
                    .is_none_or(|l| l.value_wdl.is_none())
            );
        }
        let inputs: Vec<PalsFrozenInput> = rows(&run.join("inputs.jsonl"));
        let sidecars: Vec<PalsNativeInputSidecar> = rows(&run.join("native-inputs.jsonl"));
        assert_eq!(inputs.len(), sidecars.len());
        for (input, sidecar) in inputs.iter().zip(&sidecars) {
            input.verify().unwrap();
            assert_eq!(sidecar.input_sha256, input.sha256());
            assert_eq!(
                sidecar.tensor_sha256,
                format!("{:x}", Sha256::digest(sidecar.tensor_json.as_bytes()))
            );
            let tensor: PalsModelInput = serde_json::from_str(&sidecar.tensor_json).unwrap();
            tensor.validate(&PalsModelConfig::baseline()).unwrap();
            assert_eq!(
                hex(tensor.history_digest),
                input.snapshot().rules_history_sha256
            );
            assert_eq!(tensor.model_epoch, [0; 32]);
            assert_eq!(sidecar.model_epoch_kind, "encoding_only_zero");
            assert_eq!(
                tensor
                    .candidates
                    .iter()
                    .map(|c| c.packed().unwrap())
                    .collect::<Vec<_>>(),
                input.snapshot().legal_moves
            );
            assert_eq!(sidecar.record_sources, input.snapshot().public_records);
            assert!(matches!(
                &input.snapshot().source,
                PalsInputSource::OwnCpu {
                    model_weights_sha256: None,
                    ..
                }
            ));
        }
        let pgn = std::fs::read_to_string(run.join("games.pgn")).unwrap();
        assert!(pgn.contains("[Termination \"PlyLimit\"]"));
        assert!(pgn.contains("[Result \"*\"]"));
    }
    #[test]
    fn exact_terminal_and_stalemate_are_rules_labels_without_cpu_jobs() {
        let _admission = admit_executable_fixture();
        for (fen, outcome) in [
            ("7k/6Q1/5K2/8/8/8/8/8 b - - 0 1", PalsOutcome::WhiteWin),
            ("8/8/8/8/8/5k2/6q1/7K w - - 0 1", PalsOutcome::BlackWin),
            ("7k/5Q2/5K2/8/8/8/8/8 b - - 0 1", PalsOutcome::Draw),
        ] {
            let output = OwnedTemp::new();
            let mut cpu = driver();
            let registry = own_collection_registry(&cpu).unwrap();
            let mut config = small_config();
            config.openings[0].initial_fen = Some(fen.into());
            let receipt = collect_pals_own_data(
                config,
                &output.0,
                &mut cpu,
                &registry,
                &AtomicBool::new(false),
            )
            .unwrap();
            assert!(receipt.complete, "{:?}", receipt.failure);
            assert_eq!(receipt.cpu_jobs, 0);
            assert_eq!(receipt.results[0].outcome, outcome);
            let records: Vec<PalsLearningRecord> =
                rows(&output.0.join(receipt.run_id).join("records.jsonl"));
            assert_eq!(records.len(), 1);
            let label = records[0].future_label.as_ref().unwrap();
            assert_eq!(
                label.value_wdl,
                outcome_wdl(outcome, records[0].input.snapshot().white_to_move)
            );
        }
    }
    struct BrokenDriver(OwnCpuCollectionDriver);
    impl PalsCollectionDriver for BrokenDriver {
        fn description(&self) -> &PalsCollectionSourceDescription {
            self.0.description()
        }
        fn new_game(&mut self) {
            self.0.new_game();
        }
        fn analyze(
            &mut self,
            _: &Position,
            _: &PalsFrozenInput,
            _: &PalsModelInput,
            _: CpuLimits,
            _: &AtomicBool,
        ) -> Result<PalsCollectionDecision, ArenaError> {
            Err(invalid("injected own-driver failure"))
        }
    }
    #[test]
    fn failure_preserves_seal_and_unknown_partial_pgn() {
        let _admission = admit_executable_fixture();
        let output = OwnedTemp::new();
        let mut cpu = BrokenDriver(driver());
        let registry = own_collection_registry(&cpu).unwrap();
        let receipt = collect_pals_own_data(
            small_config(),
            &output.0,
            &mut cpu,
            &registry,
            &AtomicBool::new(false),
        )
        .unwrap();
        assert!(!receipt.complete);
        assert!(receipt.failure.as_ref().unwrap().contains("injected"));
        assert_eq!(receipt.results[0].outcome, PalsOutcome::Unknown);
        let run = output.0.join(receipt.run_id);
        assert!(run.join("inputs.jsonl").is_file());
        assert!(run.join("native-inputs.jsonl").is_file());
        let pgn = std::fs::read_to_string(run.join("games.pgn")).unwrap();
        assert!(pgn.contains("InfrastructureFailure"));
        assert!(pgn.contains("[Result \"*\"]"));
        assert!(run.join("receipt.json").is_file());
    }
    #[test]
    fn source_mismatch_and_existing_directory_are_rejected_without_overwrite() {
        let _admission = admit_executable_fixture();
        let output = OwnedTemp::new();
        let mut cpu = driver();
        let mut registry = own_collection_registry(&cpu).unwrap();
        registry.input_sources.clear();
        let replacement = PalsInputSource::OwnCpu {
            cpu_binary_sha256: cpu.description().implementation_sha256.clone(),
            evaluator_configuration_sha256: "a".repeat(64),
            model_weights_sha256: None,
        };
        registry.input_sources.insert(replacement);
        assert!(
            collect_pals_own_data(
                small_config(),
                &output.0,
                &mut cpu,
                &registry,
                &AtomicBool::new(false)
            )
            .is_err()
        );
        assert!(!output.0.join(small_config().run_id).exists());
        let registry = own_collection_registry(&cpu).unwrap();
        let receipt = collect_pals_own_data(
            small_config(),
            &output.0,
            &mut cpu,
            &registry,
            &AtomicBool::new(false),
        )
        .unwrap();
        assert!(receipt.complete);
        let receipt_path = output.0.join(receipt.run_id).join("receipt.json");
        let before = std::fs::read(&receipt_path).unwrap();
        assert!(
            collect_pals_own_data(
                small_config(),
                &output.0,
                &mut cpu,
                &registry,
                &AtomicBool::new(false)
            )
            .is_err()
        );
        assert_eq!(std::fs::read(receipt_path).unwrap(), before);
    }
    #[test]
    fn overlapping_declared_splits_fail_audit_and_preserve_games() {
        let _admission = admit_executable_fixture();
        let output = OwnedTemp::new();
        let mut cpu = driver();
        let registry = own_collection_registry(&cpu).unwrap();
        let mut config = small_config();
        config.games = 2;
        config.openings.push(PalsCollectionOpening {
            id: "same-board-holdout".into(),
            split: PalsSplit::Holdout,
            ..PalsCollectionOpening::default()
        });
        let receipt = collect_pals_own_data(
            config,
            &output.0,
            &mut cpu,
            &registry,
            &AtomicBool::new(false),
        )
        .unwrap();
        assert!(!receipt.complete);
        assert_eq!(receipt.results.len(), 2);
        assert!(receipt.audit.is_none());
        let pgn = std::fs::read_to_string(output.0.join(receipt.run_id).join("games.pgn")).unwrap();
        assert_eq!(pgn.matches("[Event ").count(), 2);
    }
    #[test]
    fn conditional_repair_inputs_keep_actual_history_separate_and_no_game_outcome() {
        let _admission = admit_executable_fixture();
        let output = OwnedTemp::new();
        let source = driver().description().clone();
        let config = small_config();
        let mut disk = Output::new(&output.0, "conditional", config.max_output_bytes).unwrap();
        let mut state = Collection {
            sequence: 0,
            nodes: 0,
            jobs: 0,
            work_incomplete: false,
            rows: Vec::new(),
            latest_labels: BTreeMap::new(),
            inputs: Vec::new(),
            assignments: BTreeMap::new(),
            results: Vec::new(),
            public: Vec::new(),
            role_records: Vec::new(),
        };
        let root = Position::startpos();
        let opening = PalsCollectionOpening::default();
        let cancel = AtomicBool::new(false);
        let proposal = [
            BoardMove::from_uci("e2e4").unwrap(),
            BoardMove::from_uci("e7e5").unwrap(),
        ];
        let mut cpu = CpuEngine::new(CpuConfig {
            tt_entries: 64,
            quiescence_ply: 2,
            ..CpuConfig::default()
        })
        .unwrap();
        let cpu_source = source_description(
            &cpu,
            "own-cpu-conditional-verification",
            serde_json::Value::Null,
        )
        .unwrap();
        // All source preparation, including executable verification, precedes
        // this conditional operation's own finite admission window.
        let deadline = Instant::now() + Duration::from_secs(10);
        conditional(
            &config,
            &mut state,
            &mut disk,
            &root,
            &opening,
            "g1",
            &[],
            &proposal,
            &source,
            &cpu_source,
            &mut cpu,
            deadline,
            &cancel,
        )
        .unwrap();
        assert!(!state.inputs.is_empty());
        assert_eq!(state.inputs[0].input.snapshot().role, PalsDataRole::Critic);
        for captured in &state.inputs {
            assert!(!captured.outcome_eligible);
            assert!(captured.input.snapshot().actual_history.is_empty());
            assert!(
                captured
                    .input
                    .snapshot()
                    .position_command
                    .contains(" moves e2e4")
            );
        }
        for row in &state.rows {
            row.validate().unwrap();
            let label = row.future_label.as_ref().unwrap();
            assert!(label.value_wdl.is_none());
            assert!(
                label
                    .counterexample
                    .as_ref()
                    .is_none_or(|c| c.validity == PalsConditionalValidity::NotExamined)
            );
        }
    }
    struct VirtualDepthProbe(OwnCpuCollectionDriver);
    impl PalsCollectionDriver for VirtualDepthProbe {
        fn description(&self) -> &PalsCollectionSourceDescription {
            self.0.description()
        }
        fn new_game(&mut self) {
            self.0.new_game();
        }
        fn analyze(
            &mut self,
            position: &Position,
            input: &PalsFrozenInput,
            prepared: &PalsModelInput,
            limits: CpuLimits,
            cancel: &AtomicBool,
        ) -> Result<PalsCollectionDecision, ArenaError> {
            let mut virtual_position = position.clone();
            for text in ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "f8c5"] {
                virtual_position
                    .make_move(BoardMove::from_uci(text).map_err(rules)?)
                    .map_err(rules)?;
            }
            self.0.analyze(position, input, prepared, limits, cancel)
        }
    }
    #[test]
    fn one_actual_ply_budget_keeps_capacity_for_deeper_virtual_cpu_lines() {
        let _admission = admit_executable_fixture();
        let output = OwnedTemp::new();
        let mut cpu = VirtualDepthProbe(driver());
        let registry = own_collection_registry(&cpu).unwrap();
        let receipt = collect_pals_own_data(
            small_config(),
            &output.0,
            &mut cpu,
            &registry,
            &AtomicBool::new(false),
        )
        .unwrap();
        assert!(receipt.complete, "{:?}", receipt.failure);
        assert_eq!(receipt.results[0].ending, PalsGameEnd::PlyLimit);
    }
    #[test]
    fn output_failure_after_rules_result_never_rewrites_outcome_to_unknown() {
        let _admission = admit_executable_fixture();
        let output = OwnedTemp::new();
        let mut cpu = driver();
        let registry = own_collection_registry(&cpu).unwrap();
        let mut config = small_config();
        config.openings[0].initial_fen = Some("7k/6Q1/5K2/8/8/8/8/8 b - - 0 1".into());
        // A terminal collection's first records write occurs after its actual
        // result is known. Simulate dataset serialization/storage failure there.
        FAIL_ARTIFACT.with(|flag| flag.set(Some(("records.jsonl", 0))));
        let receipt = collect_pals_own_data(
            config,
            &output.0,
            &mut cpu,
            &registry,
            &AtomicBool::new(false),
        )
        .unwrap();
        assert!(!receipt.complete);
        assert_eq!(receipt.results.len(), 1);
        assert_eq!(receipt.results[0].outcome, PalsOutcome::WhiteWin);
        assert_eq!(receipt.results[0].ending, PalsGameEnd::Checkmate);
        let run = output.0.join(receipt.run_id);
        let results: Vec<PalsGameResult> = rows(&run.join("game-results.jsonl"));
        assert_eq!(results.len(), 1);
        assert_eq!(results[0], receipt.results[0]);
        let pgn = std::fs::read_to_string(run.join("games.pgn")).unwrap();
        assert!(pgn.contains("[Result \"1-0\"]"));
        assert!(pgn.contains("Checkmate"));
        assert!(receipt.data_output_bytes <= receipt.reserved_output_bytes);
        assert!(receipt.reserved_output_bytes <= receipt.config.max_output_bytes);
    }
    #[test]
    fn partial_opening_failure_keeps_every_successfully_played_move() {
        let _admission = admit_executable_fixture();
        let output = OwnedTemp::new();
        let mut cpu = driver();
        let registry = own_collection_registry(&cpu).unwrap();
        let mut config = small_config();
        config.openings[0].moves = vec!["e2e4".into(), "e2e5".into()];
        let receipt = collect_pals_own_data(
            config,
            &output.0,
            &mut cpu,
            &registry,
            &AtomicBool::new(false),
        )
        .unwrap();
        assert!(!receipt.complete);
        assert_eq!(receipt.results.len(), 1);
        assert_eq!(receipt.results[0].outcome, PalsOutcome::Unknown);
        let pgn = std::fs::read_to_string(output.0.join(receipt.run_id).join("games.pgn")).unwrap();
        assert!(pgn.contains("1. e4 *"));
    }
    #[test]
    fn foreign_git_checkout_and_terminal_continuations_are_rejected() {
        let _admission = admit_executable_fixture();
        let output = OwnedTemp::new();
        let foreign = output.0.join("another-repository");
        std::fs::create_dir(&foreign).unwrap();
        std::fs::write(foreign.join(".git"), "gitdir: irrelevant\n").unwrap();
        assert!(Output::new(&foreign.join("raw-data"), "outside-required", 1024 * 1024).is_err());
        assert!(!foreign.join("raw-data").exists());
        let drawn = Position::from_fen("4k3/8/8/8/8/8/4P3/4K3 w - - 150 1").unwrap();
        let movement = BoardMove::from_uci("e2e4").unwrap();
        assert!(replay_line(&drawn, &[movement]).is_err());
        let mut cpu = driver();
        let registry = own_collection_registry(&cpu).unwrap();
        let mut config = small_config();
        config.openings[0].initial_fen = Some(drawn.to_fen());
        config.openings[0].moves = vec!["e2e4".into()];
        let receipt = collect_pals_own_data(
            config,
            &output.0,
            &mut cpu,
            &registry,
            &AtomicBool::new(false),
        )
        .unwrap();
        assert!(!receipt.complete);
        assert_eq!(receipt.cpu_jobs, 0);
        assert_eq!(receipt.results[0].outcome, PalsOutcome::Draw);
    }
    struct CancelAfterAnalysis(OwnCpuCollectionDriver);
    impl PalsCollectionDriver for CancelAfterAnalysis {
        fn description(&self) -> &PalsCollectionSourceDescription {
            self.0.description()
        }
        fn new_game(&mut self) {
            self.0.new_game();
        }
        fn analyze(
            &mut self,
            position: &Position,
            input: &PalsFrozenInput,
            prepared: &PalsModelInput,
            limits: CpuLimits,
            cancel: &AtomicBool,
        ) -> Result<PalsCollectionDecision, ArenaError> {
            let result = self.0.analyze(position, input, prepared, limits, cancel)?;
            cancel.store(true, Ordering::Release);
            Ok(result)
        }
    }
    #[test]
    fn cancellation_before_actual_consumption_preserves_observation_without_moving() {
        let _admission = admit_executable_fixture();
        let output = OwnedTemp::new();
        let mut cpu = CancelAfterAnalysis(driver());
        let registry = own_collection_registry(&cpu).unwrap();
        let receipt = collect_pals_own_data(
            small_config(),
            &output.0,
            &mut cpu,
            &registry,
            &AtomicBool::new(false),
        )
        .unwrap();
        assert!(receipt.complete, "{:?}", receipt.failure);
        assert_eq!(receipt.results[0].ending, PalsGameEnd::UserStop);
        let pgn = std::fs::read_to_string(output.0.join(receipt.run_id).join("games.pgn")).unwrap();
        assert!(pgn.contains("\n\n*\n"));
        assert!(!pgn.contains("\n\n1."));
    }
    #[test]
    fn serialization_and_failed_writes_keep_finite_reserved_byte_accounting() {
        let output = OwnedTemp::new();
        let mut disk = Output::new(&output.0, "bounded", 1024 * 1024).unwrap();
        assert!(
            disk.json("huge.jsonl", &"x".repeat(MAX_JSON_RECORD_BYTES + 1))
                .is_err()
        );
        assert_eq!(disk.bytes, 0);
        let existing = disk.directory.join("failed.jsonl");
        std::fs::write(&existing, b"seed").unwrap();
        disk.files
            .insert("failed.jsonl", File::open(&existing).unwrap());
        disk.bytes = 4;
        assert!(
            disk.write("failed.jsonl", b"reserved_attempt", false)
                .is_err()
        );
        assert_eq!(disk.bytes, 4 + "reserved_attempt".len() as u64);
        assert_eq!(std::fs::metadata(existing).unwrap().len(), 4);
    }
    #[test]
    fn partial_result_and_pgn_writes_preserve_raw_prefix_and_valid_separate_recovery() {
        let _admission = admit_executable_fixture();
        for (artifact, recovery) in [
            ("game-results.jsonl", "recovered-game-results.jsonl"),
            ("games.pgn", "recovered-games.pgn"),
        ] {
            let output = OwnedTemp::new();
            let mut cpu = driver();
            let registry = own_collection_registry(&cpu).unwrap();
            let mut config = small_config();
            config.openings[0].initial_fen = Some("7k/6Q1/5K2/8/8/8/8/8 b - - 0 1".into());
            FAIL_ARTIFACT.with(|flag| flag.set(Some((artifact, 13))));
            let receipt = collect_pals_own_data(
                config,
                &output.0,
                &mut cpu,
                &registry,
                &AtomicBool::new(false),
            )
            .unwrap();
            assert!(!receipt.complete);
            assert_eq!(receipt.results[0].outcome, PalsOutcome::WhiteWin);
            let run = output.0.join(receipt.run_id);
            assert_eq!(std::fs::metadata(run.join(artifact)).unwrap().len(), 13);
            if recovery.ends_with(".jsonl") {
                let recovered: Vec<PalsGameResult> = rows(&run.join(recovery));
                assert_eq!(recovered, receipt.results);
            } else {
                let pgn = std::fs::read_to_string(run.join(recovery)).unwrap();
                assert!(pgn.contains("[Result \"1-0\"]"));
                assert!(pgn.ends_with("1-0\n\n"));
            }
            assert!(receipt.artifacts.contains_key(artifact));
            assert!(receipt.artifacts.contains_key(recovery));
            assert!(receipt.data_output_bytes <= receipt.reserved_output_bytes);
        }
    }
    #[test]
    fn admitted_compact_source_config_fits_compact_receipt_reserve() {
        let _admission = admit_executable_fixture();
        let output = OwnedTemp::new();
        let mut description = driver().description().clone();
        description
            .configuration
            .as_object_mut()
            .unwrap()
            .insert("padding".into(), serde_json::json!(vec![0_u8; 32000]));
        let digest = canonical_sha256(&description.configuration).unwrap();
        description.cpu_profile_sha256 = digest.clone();
        if let PalsInputSource::OwnCpu {
            evaluator_configuration_sha256,
            ..
        } = &mut description.source
        {
            *evaluator_configuration_sha256 = digest;
        }
        checked_source(&description).unwrap();
        let config = small_config();
        let mut disk = Output::new(&output.0, "large-descriptor", config.max_output_bytes).unwrap();
        let receipt = PalsCollectionReceipt {
            version: PALS_COLLECT_VERSION.into(),
            run_id: config.run_id.clone(),
            source: description,
            config,
            complete: false,
            failure: Some(failure_text("x".repeat(64 * 1024))),
            games_finished: 0,
            cpu_nodes: 0,
            cpu_jobs: 0,
            cpu_work_observation_incomplete: false,
            wall_time_ms: 0,
            data_output_bytes: 0,
            reserved_output_bytes: 0,
            receipt_reserve_bytes: RECEIPT_RESERVE,
            results: Vec::new(),
            audit: None,
            artifacts: BTreeMap::new(),
            actual_training_executed: false,
            external_teacher_used: false,
            native_finish: None,
        };
        disk.finish(&receipt).unwrap();
        let bytes = std::fs::read(disk.directory.join("receipt.json")).unwrap();
        assert!(bytes.len() < RECEIPT_RESERVE as usize);
        assert!(serde_json::from_slice::<serde_json::Value>(&bytes).is_ok());
        let escaped = failure_text("\0".repeat(8192));
        assert!(bounded_json(&escaped, 8192, false).is_ok());
        assert!(escaped.contains("message truncated"));
    }
    fn producer_config(cpu: &mut OwnCpuCollectionDriver) -> PalsProducerCollectionConfig {
        let bytes = serde_json::to_vec(
            &pals_producer_registration_description(cpu, "primary-owner").unwrap(),
        )
        .unwrap();
        let pin = format!("{:x}", Sha256::digest(&bytes));
        PalsProducerCollectionConfig::from_registration_bytes(&bytes, &pin)
            .unwrap()
            .with_metadata_limits(16 * 1024, 16 * 1024)
            .unwrap()
    }
    #[test]
    fn strict_cpu_producer_pins_actual_bytes_and_preserves_raw_label_history() {
        let _admission = admit_executable_fixture();
        let output = OwnedTemp::new();
        let mut cpu = driver();
        let registration = producer_config(&mut cpu);
        let registry = own_collection_registry(&cpu).unwrap();
        let mut config = small_config();
        // This correctness fixture hashes the actual test executable while
        // other CI tests run concurrently. Keep a finite admission window,
        // without treating executable hashing/scheduling as a speed assertion.
        // Production defaults and explicit timeout/failure fixtures stay fixed.
        config.max_wall_time_ms = 120_000;
        let receipt = collect_pals_own_data_with_producer(
            config,
            &output.0,
            &mut cpu,
            &registry,
            &AtomicBool::new(false),
            Some(&registration),
        )
        .unwrap();
        assert!(receipt.complete, "{:?}", receipt.failure);
        let run = output.0.join(&receipt.run_id);
        for artifact in [
            "producer-registration.json",
            "producer-source.json",
            "producer-roster.json",
            "producer-prepared.jsonl",
            "producer-captures.json",
            "producer-envelope.json",
            "producer-audit.json",
        ] {
            let bytes = std::fs::read(run.join(artifact)).unwrap();
            let pin = &receipt.artifacts[artifact];
            assert_eq!(pin.bytes, bytes.len() as u64);
            assert_eq!(pin.sha256, format!("{:x}", Sha256::digest(&bytes)));
        }
        let records: Vec<PalsLearningRecord> = rows(&run.join("records.jsonl"));
        let view = rz_experiments::current_label_view(&records).unwrap();
        assert!(records.len() > view.indices().len());
        let raw_inputs: BTreeSet<_> = records
            .iter()
            .map(|row| row.input.sha256().to_owned())
            .collect();
        let current_inputs: BTreeSet<_> = view
            .indices()
            .iter()
            .map(|index| records[*index].input.sha256().to_owned())
            .collect();
        assert_eq!(current_inputs, raw_inputs);
        assert_eq!(view.indices().len(), current_inputs.len());
        // Capture occurs before the ply-limit check: this one-move fixture
        // preserves both the initial input and the post-move final input.
        assert_eq!(receipt.results[0].ending, PalsGameEnd::PlyLimit);
        let raw_byte_rows = |artifact: &str, frozen_input: bool| {
            let bytes = std::fs::read(run.join(artifact)).unwrap();
            let mut result = BTreeMap::new();
            for line in bytes
                .split(|byte| *byte == b'\n')
                .filter(|line| !line.is_empty())
            {
                let input_sha256 = if frozen_input {
                    let input: PalsFrozenInput = serde_json::from_slice(line).unwrap();
                    input.verify().unwrap();
                    input.sha256().to_owned()
                } else {
                    let row: serde_json::Value = serde_json::from_slice(line).unwrap();
                    row["input_sha256"].as_str().unwrap().to_owned()
                };
                assert!(
                    result.insert(input_sha256, line.to_vec()).is_none(),
                    "duplicate actual byte row in {artifact}"
                );
            }
            result
        };
        let byte_rows = [
            ("input_json", raw_byte_rows("inputs.jsonl", true)),
            (
                "tensor_sidecar_json",
                raw_byte_rows("native-inputs.jsonl", false),
            ),
            ("lineage_json", raw_byte_rows("input-lineage.jsonl", false)),
        ];
        for (_, rows) in &byte_rows {
            assert_eq!(rows.keys().cloned().collect::<BTreeSet<_>>(), raw_inputs);
        }
        let actual_inputs: BTreeMap<_, PalsFrozenInput> = byte_rows[0]
            .1
            .iter()
            .map(|(sha, bytes)| (sha.clone(), serde_json::from_slice(bytes).unwrap()))
            .collect();
        assert_eq!(
            actual_inputs
                .values()
                .map(|input| input.snapshot().actual_history.len())
                .collect::<BTreeSet<_>>(),
            BTreeSet::from([0, 1])
        );
        let capture = rz_experiments::frozen_producer::ProducerCaptureArtifact::from_json(
            &std::fs::read_to_string(run.join("producer-captures.json")).unwrap(),
        )
        .unwrap();
        let mut bindings = BTreeMap::new();
        for binding in &capture.body().bindings {
            assert!(
                bindings
                    .insert(binding.input_sha256.clone(), binding)
                    .is_none()
            );
        }
        assert_eq!(
            bindings.keys().cloned().collect::<BTreeSet<_>>(),
            raw_inputs
        );
        let journal: Vec<serde_json::Value> = rows(&run.join("producer-prepared.jsonl"));
        let mut prepared_by_input = BTreeMap::new();
        for entry in &journal {
            let input_sha = entry["prepared"]["input_sha256"]
                .as_str()
                .unwrap()
                .to_owned();
            assert!(prepared_by_input.insert(input_sha, entry).is_none());
        }
        assert_eq!(
            prepared_by_input.keys().cloned().collect::<BTreeSet<_>>(),
            raw_inputs
        );
        let roster = rz_experiments::frozen_producer::FrozenProducerRoster::from_json(
            &std::fs::read_to_string(run.join("producer-roster.json")).unwrap(),
        )
        .unwrap();
        for (input_sha, input) in &actual_inputs {
            let entry = prepared_by_input[input_sha];
            let prepared = &entry["prepared"];
            let binding = bindings[input_sha];
            assert_eq!(prepared["producer_id"], "primary-owner");
            assert_eq!(prepared["native_request"], serde_json::Value::Null);
            assert_eq!(prepared["learning_input"], true);
            assert_eq!(
                prepared["publication"],
                "append-before-analysis; sync-at-receipt-close"
            );
            assert_eq!(
                prepared["capture_sequence"],
                input.snapshot().capture_sequence
            );
            assert_eq!(prepared["game_id"], input.snapshot().game_id);
            assert_eq!(prepared["roster_sha256"], roster.sha256());
            assert_eq!(
                prepared["registration_sha256"],
                receipt.artifacts["producer-registration.json"].sha256
            );
            assert_eq!(
                prepared["checked_source_sha256"],
                receipt.artifacts["producer-source.json"].sha256
            );
            assert_eq!(
                entry["sha256"],
                canonical_sha256(&("rz-pals-collector-prepared-producer/1", prepared)).unwrap()
            );
            assert_eq!(
                binding.capture_evidence_sha256,
                entry["sha256"].as_str().unwrap()
            );
            assert_eq!(binding.capture_sequence, input.snapshot().capture_sequence);
            assert_eq!(binding.game_id, input.snapshot().game_id);
            assert_eq!(binding.producer_id, "primary-owner");
            for (field, rows) in &byte_rows {
                let line = &rows[input_sha];
                assert_eq!(prepared[*field]["bytes"], line.len() as u64);
                assert_eq!(
                    prepared[*field]["sha256"],
                    format!("{:x}", Sha256::digest(line))
                );
            }
        }
        let envelope = rz_experiments::frozen_producer::FrozenProducerEnvelope::from_json(
            &std::fs::read_to_string(run.join("producer-envelope.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(envelope.body().raw_records, records.len() as u64);
        assert_eq!(envelope.body().unique_inputs, raw_inputs.len() as u64);
        assert_eq!(envelope.body().current_view_sha256, view.sha256());
        assert_eq!(
            envelope.body().raw_dataset_sha256,
            receipt.audit.as_ref().unwrap().canonical_dataset_sha256
        );
        assert_eq!(envelope.body().capture_sha256, capture.sha256());
        assert_eq!(envelope.body().roster_sha256, roster.sha256());
        assert_eq!(
            envelope.body().capture_artifact.bytes,
            receipt.artifacts["producer-captures.json"].bytes
        );
        assert_eq!(
            envelope.body().capture_artifact.sha256,
            receipt.artifacts["producer-captures.json"].sha256
        );
        let audit: serde_json::Value =
            serde_json::from_slice(&std::fs::read(run.join("producer-audit.json")).unwrap())
                .unwrap();
        assert_eq!(audit["scope"], "metadata_only");
        assert_eq!(audit["raw_records"], records.len() as u64);
        assert_eq!(audit["unique_inputs"], raw_inputs.len() as u64);
        assert_eq!(audit["envelope_sha256"], envelope.sha256());
        assert_eq!(audit["capture_sha256"], capture.sha256());
        assert_eq!(audit["roster_sha256"], roster.sha256());
        assert!(!receipt.actual_training_executed);
        assert!(receipt.data_output_bytes <= receipt.reserved_output_bytes);
    }
    #[test]
    fn strict_registration_mismatch_and_unchecked_mock_never_fall_back() {
        let _admission = admit_executable_fixture();
        let output = OwnedTemp::new();
        let mut registered = driver();
        let registration = producer_config(&mut registered);
        let mut changed = OwnCpuCollectionDriver::new(CpuConfig {
            tt_entries: 128,
            ..CpuConfig::default()
        })
        .unwrap();
        let registry = own_collection_registry(&changed).unwrap();
        assert!(
            collect_pals_own_data_with_producer(
                small_config(),
                &output.0,
                &mut changed,
                &registry,
                &AtomicBool::new(false),
                Some(&registration)
            )
            .is_err()
        );
        assert!(!output.0.join(small_config().run_id).exists());
        let mut mock =
            OwnPalsMockCollectionDriver::new(CpuConfig::default(), PalsConfig::default()).unwrap();
        let registry = own_collection_registry(&mock).unwrap();
        assert!(
            collect_pals_own_data_with_producer(
                small_config(),
                &output.0,
                &mut mock,
                &registry,
                &AtomicBool::new(false),
                Some(&registration)
            )
            .is_err()
        );
        assert!(!output.0.join(small_config().run_id).exists());
    }
    #[test]
    fn producer_close_credit_is_reserved_before_first_capture() {
        let _admission = admit_executable_fixture();
        let output = OwnedTemp::new();
        let mut cpu = driver();
        let bytes = serde_json::to_vec(
            &pals_producer_registration_description(&mut cpu, "primary-owner").unwrap(),
        )
        .unwrap();
        let registration = PalsProducerCollectionConfig::from_registration_bytes(
            &bytes,
            &format!("{:x}", Sha256::digest(&bytes)),
        )
        .unwrap();
        let registry = own_collection_registry(&cpu).unwrap();
        assert!(
            collect_pals_own_data_with_producer(
                small_config(),
                &output.0,
                &mut cpu,
                &registry,
                &AtomicBool::new(false),
                Some(&registration)
            )
            .is_err()
        );
        let run = output.0.join(small_config().run_id);
        assert!(!run.join("inputs.jsonl").exists());
        assert!(!run.join("producer-prepared.jsonl").exists());
    }
    #[test]
    fn prior_registration_requires_actual_pin_unique_keys_and_integer_epoch() {
        let _admission = admit_executable_fixture();
        let mut cpu = driver();
        let description =
            pals_producer_registration_description(&mut cpu, "primary-owner").unwrap();
        let bytes = serde_json::to_vec(&description).unwrap();
        assert!(
            PalsProducerCollectionConfig::from_registration_bytes(&bytes, &"0".repeat(64)).is_err()
        );
        let duplicate = format!(
            "{{\"producer_id\":\"another-owner\",{}",
            std::str::from_utf8(&bytes)
                .unwrap()
                .strip_prefix('{')
                .unwrap()
        );
        assert!(
            PalsProducerCollectionConfig::from_registration_bytes(
                duplicate.as_bytes(),
                &format!("{:x}", Sha256::digest(duplicate.as_bytes()))
            )
            .is_err()
        );
        let mut fractional = description;
        fractional["frozen_epoch"] = serde_json::json!(0.5);
        let bytes = serde_json::to_vec(&fractional).unwrap();
        assert!(
            PalsProducerCollectionConfig::from_registration_bytes(
                &bytes,
                &format!("{:x}", Sha256::digest(&bytes))
            )
            .is_err()
        );
    }
    struct CustomOwnerWrapper {
        inner: OwnCpuCollectionDriver,
        delegate_owner: bool,
        custom_dispatches: usize,
    }
    impl PalsCollectionDriver for CustomOwnerWrapper {
        fn description(&self) -> &PalsCollectionSourceDescription {
            self.inner.description()
        }
        fn checked_producer_owner(
            &mut self,
        ) -> Result<Option<CheckedProducerOwner<'_>>, ArenaError> {
            if self.delegate_owner {
                self.inner.checked_producer_owner()
            } else {
                Ok(None)
            }
        }
        fn new_game(&mut self) {
            self.custom_dispatches += 1;
        }
        fn analyze(
            &mut self,
            _: &Position,
            _: &PalsFrozenInput,
            _: &PalsModelInput,
            _: CpuLimits,
            _: &AtomicBool,
        ) -> Result<PalsCollectionDecision, ArenaError> {
            self.custom_dispatches += 1;
            Err(invalid(
                "unverified custom dispatch must not receive strict authority",
            ))
        }
        fn finish_collection(
            &mut self,
            _: Instant,
        ) -> Result<Option<serde_json::Value>, ArenaError> {
            self.custom_dispatches += 1;
            Err(invalid(
                "unverified custom finish must not receive strict authority",
            ))
        }
    }
    #[test]
    fn opaque_owner_rejects_declaration_and_routes_delegated_proof_to_actual_dispatcher() {
        let _admission = admit_executable_fixture();
        let output = OwnedTemp::new();
        let mut wrapper = CustomOwnerWrapper {
            inner: driver(),
            delegate_owner: false,
            custom_dispatches: 0,
        };
        // Matching bytes and source facts alone cannot enroll this custom owner.
        let registration = producer_config(&mut wrapper.inner);
        let registry = own_collection_registry(&wrapper).unwrap();
        assert!(
            collect_pals_own_data_with_producer(
                small_config(),
                &output.0,
                &mut wrapper,
                &registry,
                &AtomicBool::new(false),
                Some(&registration)
            )
            .is_err()
        );
        assert_eq!(wrapper.custom_dispatches, 0);
        assert!(!output.0.join(small_config().run_id).exists());
        // Borrowing the actual owner capability redirects every dispatch and
        // lifecycle hook to that owner; it does not authorize wrapper methods.
        wrapper.delegate_owner = true;
        let mut config = small_config();
        // This positive owner-routing fixture also pins the actual test
        // executable under parallel CI; its wall-time is not a speed assertion.
        config.max_wall_time_ms = 120_000;
        let receipt = collect_pals_own_data_with_producer(
            config,
            &output.0,
            &mut wrapper,
            &registry,
            &AtomicBool::new(false),
            Some(&registration),
        )
        .unwrap();
        assert!(receipt.complete, "{:?}", receipt.failure);
        assert_eq!(wrapper.custom_dispatches, 0);
        assert!(receipt.artifacts.contains_key("producer-envelope.json"));
    }
}
