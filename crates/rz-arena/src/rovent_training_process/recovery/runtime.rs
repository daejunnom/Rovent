//! A new process owner for exact historical data. Historical byte evidence is
//! never cast into original native custody. Only this current peer, supervisor,
//! resource scope and FD owner can admit current work or current completion.
use super::super::numerical_child::{
    UnixFrameReadErrors, UnixFrameReadScope, charge_unix_ack, is_direct_parent_group_leader,
    prepare_unix_ack, read_unix_frame, unix_peer_credentials, write_unix_ack,
};
use super::super::{TrainerLedger, linux};
use super::r_auxiliary_ledger::{RoventAuxiliaryLedger, valid_pc_rows};
use super::*;
use crate::pals_collect_v3::training_process::{directory_file, open_directory};
use crate::process::{
    OriginalNumericalEnvironment, OriginalProcessInput, OriginalProcessWindow,
    RoventTrainingArtifactTreeWatch, supervise_rovent_training_numerical_input_in_original_window,
};
use sha2::{Digest as _, Sha256};
use std::{
    ffi::OsString,
    fs,
    io::Read,
    os::unix::{
        fs::MetadataExt,
        net::{UnixListener, UnixStream},
    },
    sync::Mutex,
    thread::{self, JoinHandle},
    time::Duration,
};

pub const ROVENT_RECOVERED_RPC: &str = "rz-rovent-recovered-trainer-rpc/1";
pub const ROVENT_RECOVERED_ACK: &str = "rz-rovent-recovered-trainer-ack/1";
pub const ROVENT_RECOVERED_RESULT: &str = "rz-rovent-recovered-trainer-result/1";
pub const ROVENT_RECOVERED_UPDATE: &str = "rz-rovent-recovered-learner-completion/1";
pub const ROVENT_RECOVERED_RETIREMENT: &str = "rz-rovent-recovered-checkpoint-retirement/1";
const MAX_CHECKPOINT: u64 = 2 << 30;
const MAX_SOURCE: u64 = 4 << 20;
static NEXT_SOCKET: AtomicU64 = AtomicU64::new(1);

pub struct RoventRecoveredTrainingLaunch {
    pub interpreter: File,
    pub interpreter_pin: RoventTrainingFilePin,
    pub interpreter_version: String,
    pub registration: File,
    pub registration_pin: RoventTrainingFilePin,
    pub dependency_directories: Vec<PathBuf>,
    pub started: Instant,
    pub execution_deadline: Instant,
    pub whole_deadline: Instant,
    pub limits: ProcessLimits,
    pub resources: Arc<RoventTrainingResources>,
    /// The CLI's admission reads and all current parent reads share this owner.
    pub read_budget: Arc<RoventRecoveryReadBudget>,
}

pub struct VerifiedRoventRecoveredTrainingWindow {
    pub(super) registration: RoventRecoveredTrainingRegistration,
    pub(super) summary: Value,
    pub(super) journal: TrainerLedger,
    pub(super) process: OriginalProcessOutput,
    pub(super) launch: RoventRecoveredTrainingLaunch,
    pub(super) dataset: Arc<VerifiedRoventDataset>,
    pub(super) checkpoint: Option<Arc<VerifiedRoventCheckpointSource>>,
    pub(super) _retained_reclamation: Option<VerifiedRoventOldRunReclamation>,
    pub(super) held: linux::LinuxHeld,
    pub(super) child_identity: RoventHistoricalChildIdentity,
    pub(super) _retained_boundary_archives: Vec<PublishedRoventCheckpointArchive>,
    pub(super) r_auxiliary_ledger: Option<RoventAuxiliaryLedger>,
    _retained_r_dataset: Option<Arc<VerifiedRoventRAuxiliaryDataset>>,
}
impl VerifiedRoventRecoveredTrainingWindow {
    pub fn summary(&self) -> &Value {
        &self.summary
    }
    pub fn observation(&self) -> Value {
        self.summary.clone()
    }
    pub fn process(&self) -> &OriginalProcessOutput {
        &self.process
    }
    pub fn committed_updates(&self) -> &BTreeMap<String, u64> {
        &self.journal.completed
    }
    pub fn original_checkpoint_files(
        &self,
    ) -> impl Iterator<Item = (&RoventTrainingFilePin, &File)> {
        self.held.checkpoint_files()
    }
    pub fn original_export_files(&self) -> impl Iterator<Item = (&RoventTrainingFilePin, &File)> {
        self.held.export_files()
    }
}

/// Failed owners retain the real child/group/FD/broker/resource scope. Dropping
/// metadata is not reclamation; the caller must preserve this owner on failure.
pub struct RoventRecoveredTrainingFailure {
    pub error: ArenaError,
    pub process: Option<OriginalProcessOutput>,
    pub summary: Option<Value>,
    pub rpc_journal: Value,
    pub launch: RoventRecoveredTrainingLaunch,
    pub dataset: Arc<VerifiedRoventDataset>,
    pub checkpoint: Option<Arc<VerifiedRoventCheckpointSource>>,
    pub reclamation: Option<VerifiedRoventOldRunReclamation>,
    pub(super) _retained_held: Option<linux::LinuxHeld>,
    _retained_broker: Option<PendingBroker>,
    _retained_shared_held: Arc<Mutex<Option<linux::LinuxHeld>>>,
    boundary_archives: Arc<Mutex<Vec<PublishedRoventCheckpointArchive>>>,
    // Retain each publisher-returned failure owner, including its original
    // descriptors and partial archive custody, without unpacking that owner.
    #[allow(clippy::vec_box)]
    boundary_archive_failures: Arc<Mutex<Vec<Box<RoventCheckpointArchiveFailure>>>>,
    r_auxiliary_ledger: Arc<Mutex<Option<RoventAuxiliaryLedger>>>,
    _retained_r_dataset: Option<Arc<VerifiedRoventRAuxiliaryDataset>>,
}
impl std::fmt::Debug for RoventRecoveredTrainingFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RoventRecoveredTrainingFailure")
            .field("error", &self.error)
            .field("process", &self.process)
            .finish_non_exhaustive()
    }
}
impl RoventRecoveredTrainingFailure {
    pub fn observation(&self) -> Value {
        let mut value = json!({"schema":"rz-rovent-recovered-trainer-failure/1",
        "status":"failed","read_accounting":self.launch.read_budget.observation(),
        "rpc_journal":self.rpc_journal,"summary":self.summary,
        "old_cost_refunded":false,"retry_authorized":false,"physical_completion":"unknown",
        "past_native_authority_reissued":false,"past_physical_completion_reissued":false,
        "boundary_archives":self.boundary_archives.try_lock().ok().map(|owners|owners.iter().map(|owner|owner.observation()).collect::<Vec<_>>()),
        "boundary_archive_failures":self.boundary_archive_failures.try_lock().ok().map(|owners|owners.iter().map(|owner|owner.observation()).collect::<Vec<_>>())});
        if let Ok(auxiliary) = self.r_auxiliary_ledger.try_lock()
            && let Some(auxiliary) = auxiliary.as_ref()
        {
            value["r_auxiliary"] = auxiliary.observation();
        }
        value
    }
}

fn own(
    pin: &RoventTrainingFilePin,
    max: u64,
    retain: bool,
    budget: &RoventRecoveryReadBudget,
) -> Result<(linux::HeldFile, Vec<u8>), ArenaError> {
    let (f, raw) = budget.read(pin, max, retain)?;
    Ok((
        linux::HeldFile {
            file: f.file,
            pin: f.pin,
            identity: f.identity,
        },
        raw,
    ))
}
fn load(
    pin: &RoventTrainingFilePin,
    owner: &mut linux::LinuxHeld,
    budget: &RoventRecoveryReadBudget,
) -> Result<Value, ArenaError> {
    let (f, raw) = own(pin, MAX_METADATA_BYTES, true, budget)?;
    owner.inputs.push(f);
    decode(&raw)
}

fn dataset_ancestors(
    current: &VerifiedRoventDataset,
) -> Result<BTreeMap<String, &VerifiedRoventDataset>, ArenaError> {
    fn visit<'a>(
        current: &'a VerifiedRoventDataset,
        out: &mut BTreeMap<String, &'a VerifiedRoventDataset>,
        depth: usize,
    ) -> Result<(), ArenaError> {
        check(depth <= 256, "historical dataset ancestor depth")?;
        for ancestor in &current.ancestors {
            if out
                .insert(ancestor.registration.publication.sha256.clone(), ancestor)
                .is_none()
            {
                visit(ancestor, out, depth + 1)?;
            }
        }
        Ok(())
    }
    let mut out = BTreeMap::new();
    visit(current, &mut out, 0)?;
    check(
        out.len() <= 256 && !out.contains_key(&current.registration.publication.sha256),
        "historical dataset cycle/capacity",
    )?;
    Ok(out)
}

/// Original R publications can reference PC evidence outside the current PC
/// dataset's own ancestry. Retain and join those owners too, without changing
/// the separate collection-admission projection for the current PC closure.
fn all_dataset_inputs<'a>(
    current: &'a VerifiedRoventDataset,
    r_dataset: Option<&'a VerifiedRoventRAuxiliaryDataset>,
    historical: &'a [Arc<VerifiedRoventDataset>],
) -> Result<BTreeMap<String, &'a VerifiedRoventDataset>, ArenaError> {
    fn add_pc<'a>(
        current: &VerifiedRoventDataset,
        source: &'a VerifiedRoventDataset,
        out: &mut BTreeMap<String, &'a VerifiedRoventDataset>,
    ) -> Result<(), ArenaError> {
        if source.registration.publication.sha256 == current.registration.publication.sha256 {
            check(
                source.registration == current.registration,
                "same current PC publication has conflicting original registration",
            )?;
        } else if let Some(old) = out.insert(source.registration.publication.sha256.clone(), source)
        {
            check(
                old.registration == source.registration,
                "same PC source publication has conflicting retained owners",
            )?;
        }
        for (_, ancestor) in dataset_ancestors(source)? {
            if ancestor.registration.publication.sha256 == current.registration.publication.sha256 {
                check(
                    ancestor.registration == current.registration,
                    "R-referenced PC ancestry changed current registration",
                )?;
            } else if let Some(old) =
                out.insert(ancestor.registration.publication.sha256.clone(), ancestor)
            {
                check(
                    old.registration == ancestor.registration,
                    "R-referenced PC ancestry has conflicting retained owners",
                )?;
            }
        }
        check(out.len() <= 256, "combined original PC source capacity")
    }
    fn add_r<'a>(
        current: &VerifiedRoventDataset,
        source: &'a VerifiedRoventRAuxiliaryDataset,
        out: &mut BTreeMap<String, &'a VerifiedRoventDataset>,
        seen: &mut BTreeSet<String>,
        depth: usize,
    ) -> Result<(), ArenaError> {
        check(depth <= 256, "R-referenced original PC ancestry depth")?;
        if !seen.insert(source.publication_pin().sha256.clone()) {
            return Ok(());
        }
        check(seen.len() <= 257, "R-referenced original dataset capacity")?;
        for pc in source.pc_datasets() {
            add_pc(current, pc, out)?;
        }
        for pc in source.joint_pc_datasets() {
            add_pc(current, pc, out)?;
        }
        for ancestor in source.ancestors() {
            add_r(current, ancestor, out, seen, depth + 1)?;
        }
        Ok(())
    }
    let mut out = dataset_ancestors(current)?;
    if let Some(source) = r_dataset {
        add_r(current, source, &mut out, &mut BTreeSet::new(), 0)?;
    }
    for source in historical {
        add_pc(current, source, &mut out)?;
    }
    Ok(out)
}
fn source_bundle() -> Result<BTreeMap<&'static str, &'static [u8]>, ArenaError> {
    let mut files = linux::source_bundle()?;
    macro_rules! source {
        ($name:literal) => {
            (
                concat!("experiments/model-research/pals/src/rz_pals_model/", $name),
                include_bytes!(concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/../../experiments/model-research/pals/src/rz_pals_model/",
                    $name
                ))
                .as_slice(),
            )
        };
    }
    for (path, bytes) in [
        source!("rovent_training_recovery.py"),
        source!("rovent_historical_checkpoint.py"),
        source!("rovent_recovered_training_child.py"),
        source!("rovent_recovered_training_authority.py"),
        source!("rovent_recovered_training_window.py"),
        source!("rovent_training_window_v2.py"),
        source!("rovent_historical_corpus.py"),
        source!("rovent_r_auxiliary.py"),
        source!("rovent_r_corpus.py"),
        source!("rovent_historical_r_corpus.py"),
        source!("rovent_r_readout.py"),
        source!("rovent_lineage_union.py"),
        source!("rovent_joint_corpus.py"),
        source!("rovent_r_ancestry.py"),
        source!("rovent_collection_teacher.py"),
        source!("rovent_root_descriptor.py"),
        source!("rovent_submitted_input.py"),
        source!("rovent_corpus_codec.py"),
        source!("rovent_native_v3.py"),
        source!("rovent_external_root.py"),
        source!("rovent_divergence.py"),
    ] {
        files.insert(path, bytes);
    }
    check(
        files.len() <= 128
            && files.values().all(|v| v.len() as u64 <= MAX_SOURCE)
            && files.values().map(|v| v.len() as u64).sum::<u64>() <= 32 << 20,
        "recovered compiled source closure bound",
    )?;
    Ok(files)
}
pub fn compiled_rovent_recovered_trainer_source_sha256() -> Result<String, ArenaError> {
    let inventory = source_bundle()?
        .iter()
        .map(|(p, b)| {
            (
                p.to_string(),
                json!({"bytes":b.len(),"sha256":format!("{:x}",Sha256::digest(b))}),
            )
        })
        .collect::<BTreeMap<_, _>>();
    digest_value("rz-rovent-recovered-trainer-source-bundle/1", &inventory)
}
fn install_sources(
    owner: &mut linux::LinuxHeld,
    registration: &RoventRecoveredTrainingRegistration,
    until: Instant,
    cancel: &AtomicBool,
) -> Result<Value, ArenaError> {
    check(
        registration.trainer_source_sha256 == compiled_rovent_recovered_trainer_source_sha256()?,
        "new recovered compiled source closure differs",
    )?;
    let files = source_bundle()?;
    check(
        files.len() <= registration.max_artifact_files
            && files
                .values()
                .try_fold(0_u64, |total, bytes| total.checked_add(bytes.len() as u64))
                .is_some_and(|bytes| {
                    registration
                        .max_output_bytes
                        .checked_sub(owner.control_limit)
                        .is_some_and(|remaining| bytes <= remaining)
                }),
        "compiled source installation exceeds the original file/byte output profile",
    )?;
    let mut inventory = Vec::new();
    for (path, bytes) in files {
        check(
            Instant::now() < until && !cancel.load(Ordering::Acquire),
            "source installation original deadline/cancel",
        )?;
        let target = owner.source.join(path);
        linux::ensure_child_directory(
            &owner.output_path,
            target.parent().ok_or_else(|| invalid("source parent"))?,
        )?;
        let file = linux::write_new(&target, bytes, owner, false)?;
        inventory
            .push(json!({"relative_path":path,"sha256":file.pin.sha256,"bytes":file.pin.bytes}));
        owner.sources.push(file);
    }
    Ok(json!(inventory))
}

fn new_ledger(
    window: &RoventTrainWindowV2,
    recipe: Value,
    membership_sha256: String,
    dataset: &VerifiedRoventDataset,
    initial: BTreeMap<String, u64>,
    ancestors: BTreeMap<String, u64>,
    checkpoint: Option<&VerifiedRoventCheckpointSource>,
) -> Result<TrainerLedger, ArenaError> {
    check(
        dataset.family() != RoventPcRecordFamily::SubmittedInputV1 || window.phase == "pc",
        "submitted-input corpus family is V-free; legacy V corpus remains a distinct source",
    )?;
    if let Some(auxiliary) = window.nnuec_r_auxiliary.enabled() {
        let owners = &recipe["parameter_ownership"];
        fields(
            owners,
            &[
                "pc_shared",
                "p_private",
                "c_private",
                "v_private",
                "nnuec",
                "membership_sha256",
            ],
        )?;
        check(
            owners["membership_sha256"] == membership_sha256
                && owners["nnuec"] == auxiliary.optimizer_owner
                && ["pc_shared", "p_private", "c_private", "v_private"]
                    .iter()
                    .all(|key| {
                        owners[*key].as_str().is_some_and(|owner| {
                            generation_identifier(owner) && owner != auxiliary.optimizer_owner
                        })
                    }),
            "R auxiliary needs its actual pinned membership and exclusive NNUEC optimizer owner",
        )?;
    }
    let stage = recipe["stages"]
        .as_array()
        .and_then(|stages| stages.iter().find(|s| s["stage_id"] == window.stage_id))
        .ok_or_else(|| invalid("registered stage missing"))?;
    let stage_limits: BTreeMap<String, u64> = serde_json::from_value(stage["role_updates"].clone())
        .map_err(|e| invalid(e.to_string()))?;
    check(
        initial.len() == 3
            && ancestors.len() == 3
            && stage_limits.len() == 3
            && ROLES.iter().all(|r| {
                initial
                    .get(*r)
                    .zip(stage_limits.get(*r))
                    .is_some_and(|(i, l)| i <= l)
                    && ancestors.contains_key(*r)
            }),
        "new exact bounded P/C/V stage counts",
    )?;
    let mut ledger = TrainerLedger {
        window: serde_json::to_value(window).map_err(|e| invalid(e.to_string()))?,
        recipe,
        membership_sha256,
        profile_sha256: window.profile_sha256()?,
        stage_limits,
        corpus_sha256: dataset.manifest_file_sha256().to_owned(),
        rows: dataset.rows.clone(),
        completed: initial.clone(),
        initial,
        ancestors,
        pending: Vec::new(),
        consumed_inputs: dataset
            .manifest
            .inherited_input_sha256
            .iter()
            .cloned()
            .collect(),
        consumed_origins: dataset
            .manifest
            .inherited_origin_sha256
            .iter()
            .cloned()
            .collect(),
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
    };
    if let Some(previous) = checkpoint {
        ledger
            .consumed_inputs
            .extend(previous.retained_inputs().iter().cloned());
        ledger
            .consumed_origins
            .extend(previous.retained_origins().iter().cloned());
    }
    Ok(ledger)
}

fn new_auxiliary_ledger(
    registration: &RoventRecoveredTrainingRegistration,
    window: &RoventTrainWindowV2,
    dataset: Option<Arc<VerifiedRoventRAuxiliaryDataset>>,
    checkpoint: Option<&VerifiedRoventCheckpointSource>,
) -> Result<Option<RoventAuxiliaryLedger>, ArenaError> {
    let Some(recipe) = window.nnuec_r_auxiliary.enabled() else {
        check(
            dataset.is_none(),
            "disabled/null R auxiliary cannot carry an opaque dataset owner",
        )?;
        return Ok(None);
    };
    let dataset = dataset
        .ok_or_else(|| invalid("enabled R auxiliary needs its original opaque dataset owner"))?;
    check(
        dataset.manifest_file_sha256() == recipe.manifest_file_sha256
            && dataset.index_file_sha256() == recipe.index_file_sha256,
        "R auxiliary recipe changed exact manifest/index file pins",
    )?;
    let mut result = match &registration.initialization_source {
        RoventTrainingInputSource::Initialize { .. } => {
            RoventAuxiliaryLedger::new(dataset, recipe.clone(), BTreeSet::new(), BTreeSet::new(), 0)
                .map(Some)
        }
        RoventTrainingInputSource::Resume { .. } => {
            let mut previous = checkpoint
                .and_then(|source| source.r_auxiliary_ledger())
                .cloned()
                .ok_or_else(|| {
                    invalid("enabled R resume has no replayed original prefix ledger")
                })?;
            previous.bind_current_dataset(dataset, recipe)?;
            Ok(Some(previous))
        }
        RoventTrainingInputSource::Fork { .. } => {
            let source = checkpoint
                .ok_or_else(|| invalid("fork lacks its original historical checkpoint source"))?;
            match source.r_auxiliary_ledger() {
                Some(previous) => previous.fork(dataset, recipe.clone()).map(Some),
                None => RoventAuxiliaryLedger::from_non_auxiliary_checkpoint(
                    dataset,
                    recipe.clone(),
                    source,
                )
                .map(Some),
            }
        }
    }?;
    if let Some(ledger) = result.as_mut() {
        let preserved = if matches!(
            registration.initialization_source,
            RoventTrainingInputSource::Resume { .. }
        ) {
            checkpoint
        } else {
            None
        };
        ledger.set_execution_phase(&window.phase, preserved)?;
    }
    Ok(result)
}

fn validate_auxiliary_fields(
    p: &Value,
    base: &[&str],
    auxiliary: Option<&RoventAuxiliaryLedger>,
) -> Result<(), ArenaError> {
    let mut expected = base.to_vec();
    if auxiliary.is_some() {
        expected.push("r_auxiliary");
    }
    fields(p, &expected)?;
    if let Some(auxiliary) = auxiliary {
        auxiliary.validate_checkpoint_projection(&p["r_auxiliary"])?;
    }
    Ok(())
}

fn auxiliary_admission(dataset: &VerifiedRoventRAuxiliaryDataset) -> Result<Value, ArenaError> {
    fn visit(
        dataset: &VerifiedRoventRAuxiliaryDataset,
        out: &mut BTreeSet<String>,
        depth: usize,
    ) -> Result<(), ArenaError> {
        check(depth <= 256, "R auxiliary ancestry depth")?;
        for ancestor in dataset.ancestors() {
            if out.insert(ancestor.publication_pin().sha256.clone()) {
                visit(ancestor, out, depth + 1)?;
            }
        }
        check(out.len() <= 256, "R auxiliary ancestor capacity")
    }
    let mut ancestors = BTreeSet::new();
    visit(dataset, &mut ancestors, 0)?;
    check(
        !ancestors.contains(&dataset.publication_pin().sha256),
        "R auxiliary ancestry cycle",
    )?;
    Ok(json!({"schema":"rz-rovent-recovered-r-dataset-set/1",
        "current_publication_sha256":dataset.publication_pin().sha256,"ancestor_publication_sha256":ancestors,
        "manifest_file_sha256":dataset.manifest_file_sha256(),"index_file_sha256":dataset.index_file_sha256(),
        "nnuec_parameter_sha256":dataset.registration().nnuec_parameter_sha256,"record_count":dataset.observation()["record_count"],
        "past_native_authority_reissued":false,"past_physical_completion_reissued":false,
        "historical_evidence_only":true,"execution_authorized":false}))
}

/// The parent joins declared ownership to its pinned descriptor inventory. The
/// child separately checks these descriptors against the real model before any
/// optimizer construction; this metadata check does not claim a model forward.
fn validate_auxiliary_membership(
    window: &RoventTrainWindowV2,
    recipe: &Value,
    membership: &Value,
    membership_sha256: &str,
) -> Result<(), ArenaError> {
    let Some(auxiliary) = window.nnuec_r_auxiliary.enabled() else {
        return Ok(());
    };
    check(
        recipe["parameter_ownership"]["membership_sha256"] == membership_sha256
            && recipe["parameter_ownership"]["nnuec"] == auxiliary.optimizer_owner
            && membership["model_semantics_sha256"] == window.training_model_semantics_sha256,
        "R optimizer inventory differs from exact recipe/model membership",
    )?;
    let parameters = membership["parameters"]
        .as_array()
        .filter(|p| !p.is_empty() && p.len() <= 4096)
        .ok_or_else(|| invalid("bounded actual parameter membership required"))?;
    let mut names = BTreeSet::new();
    let mut has_nnuec = false;
    for parameter in parameters {
        let name = parameter["name"]
            .as_str()
            .filter(|name| generation_identifier(name))
            .ok_or_else(|| invalid("actual parameter name"))?;
        check(
            names.insert(name) && parameter["dtype"] == "float32" && parameter["trainable"] == true,
            "duplicate or nontrainable FP32 actual parameter",
        )?;
        let component = if name.starts_with("nnuec.") {
            has_nnuec = true;
            "nnuec"
        } else if name.starts_with("public_encoder.")
            || name.starts_with("reader_blocks.")
            || name.starts_with("move_embedding.")
        {
            "pc_shared"
        } else if name.starts_with("experts.proposer.") {
            "p_private"
        } else if name.starts_with("experts.critic.") {
            "c_private"
        } else if name.starts_with("experts.validator.") {
            "v_private"
        } else {
            return Err(invalid("unmapped actual parameter component"));
        };
        check(
            (component == "nnuec")
                == (recipe["parameter_ownership"][component] == auxiliary.optimizer_owner),
            "R optimizer may own NNUEC parameters only",
        )?;
    }
    check(
        has_nnuec,
        "enabled R auxiliary has no actual NNUEC parameter membership",
    )
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    schema: String,
    window_id: String,
    startup_sha256: String,
    sequence: u64,
    operation: String,
    payload: Value,
}
struct Broker {
    listener: UnixListener,
    parent_pid: u32,
    parent_start: u64,
    interpreter: RoventTrainingFilePin,
    startup_sha256: String,
    registration: RoventRecoveredTrainingRegistration,
    window: RoventTrainWindowV2,
    until: Instant,
    whole_until: Instant,
    cancel: Arc<AtomicBool>,
    stop: Arc<AtomicBool>,
    image_verified: Arc<AtomicBool>,
    ledger: Arc<Mutex<TrainerLedger>>,
    held: Arc<Mutex<Option<linux::LinuxHeld>>>,
    budget: Arc<RoventRecoveryReadBudget>,
    resources: Arc<RoventTrainingResources>,
    checkpoint: Option<Arc<VerifiedRoventCheckpointSource>>,
    child_identity: Arc<Mutex<Option<RoventHistoricalChildIdentity>>>,
    reclamation: Option<Value>,
    dataset: Arc<VerifiedRoventDataset>,
    boundary_archives: Arc<Mutex<Vec<PublishedRoventCheckpointArchive>>>,
    // Share custody of the original boxed publication failures with the
    // returned failure owner; logical failure does not release their FDs.
    #[allow(clippy::vec_box)]
    boundary_archive_failures: Arc<Mutex<Vec<Box<RoventCheckpointArchiveFailure>>>>,
    _retained_r_dataset: Option<Arc<VerifiedRoventRAuxiliaryDataset>>,
    r_auxiliary_ledger: Arc<Mutex<Option<RoventAuxiliaryLedger>>>,
}

/// Borrowed only inside the current authenticated checkpoint RPC after the
/// tensor FD and parent retirement have been retained and fsynced. A report,
/// path, or saved publication cannot construct this source. It proves a current
/// durable update boundary; it does not claim process exit or GPU completion.
pub(super) struct ActualRoventCheckpointBoundary<'a> {
    ctx: &'a Broker,
    ledger: &'a TrainerLedger,
    held: &'a linux::LinuxHeld,
    checkpoint_index: usize,
}
impl ActualRoventCheckpointBoundary<'_> {
    pub(super) fn registration(&self) -> &RoventRecoveredTrainingRegistration {
        &self.ctx.registration
    }
    pub(super) fn window(&self) -> &RoventTrainWindowV2 {
        &self.ctx.window
    }
    pub(super) fn journal(&self) -> &TrainerLedger {
        self.ledger
    }
    pub(super) fn held(&self) -> &linux::LinuxHeld {
        self.held
    }
    pub(super) fn dataset(&self) -> &Arc<VerifiedRoventDataset> {
        &self.ctx.dataset
    }
    pub(super) fn checkpoint_history(&self) -> Option<&Arc<VerifiedRoventCheckpointSource>> {
        self.ctx.checkpoint.as_ref()
    }
    pub(super) fn checkpoint_index(&self) -> usize {
        self.checkpoint_index
    }
    pub(super) fn r_auxiliary_ledger(&self) -> Result<Option<RoventAuxiliaryLedger>, ArenaError> {
        Ok(self
            .ctx
            .r_auxiliary_ledger
            .lock()
            .map_err(|_| invalid("R auxiliary custody poisoned"))?
            .clone())
    }
    pub(super) fn child_identity(&self) -> Result<RoventHistoricalChildIdentity, ArenaError> {
        self.ctx
            .child_identity
            .lock()
            .map_err(|_| invalid("current boundary child identity poisoned"))?
            .clone()
            .ok_or_else(|| invalid("current boundary has no authenticated child identity"))
    }
}
struct PendingBroker {
    handle: JoinHandle<Result<(), ArenaError>>,
    stop: Arc<AtomicBool>,
}
fn kernel_text(
    path: &Path,
    maximum: u64,
    budget: &RoventRecoveryReadBudget,
) -> Result<String, ArenaError> {
    budget.reserve(
        maximum
            .checked_add(1)
            .ok_or_else(|| invalid("kernel read credit overflow"))?,
    )?;
    let mut bytes = Vec::new();
    File::open(path)
        .map_err(|_| invalid("current kernel text open"))?
        .take(maximum + 1)
        .read_to_end(&mut bytes)
        .map_err(|_| invalid("current kernel text read"))?;
    check(
        bytes.len() as u64 <= maximum,
        "current kernel text exceeded reserved read bound",
    )?;
    String::from_utf8(bytes).map_err(|_| invalid("current kernel text UTF8"))
}
fn process_start(pid: u32, budget: &RoventRecoveryReadBudget) -> Result<u64, ArenaError> {
    let value = kernel_text(Path::new(&format!("/proc/{pid}/stat")), 8192, budget)?;
    value
        .rsplit_once(") ")
        .and_then(|(_, tail)| tail.split_ascii_whitespace().nth(19))
        .and_then(|v| v.parse().ok())
        .ok_or_else(|| invalid("bounded current process start ticks"))
}
fn peer_admit(stream: &UnixStream, ctx: &Broker) -> Result<(u32, u64), ArenaError> {
    let peer = unix_peer_credentials(stream, "recovered SO_PEERCRED", "peer PID")?;
    let pid = peer.pid;
    check(
        peer.uid == nix::unistd::geteuid().as_raw()
            && ctx.image_verified.load(Ordering::Acquire)
            && process_start(ctx.parent_pid, &ctx.budget)? == ctx.parent_start,
        "current parent/image/UID mismatch",
    )?;
    let status = kernel_text(
        Path::new(&format!("/proc/{pid}/status")),
        16 << 10,
        &ctx.budget,
    )?;
    check(
        is_direct_parent_group_leader(pid, ctx.parent_pid, &status),
        "peer is not the current direct group leader",
    )?;
    let image =
        File::open(format!("/proc/{pid}/exe")).map_err(|_| invalid("current peer image"))?;
    ctx.budget.verify_held(&image, &ctx.interpreter, false)?;
    let membership = ctx.resources.verify_child_membership(pid)?;
    let boot = kernel_text(
        Path::new("/proc/sys/kernel/random/boot_id"),
        4096,
        &ctx.budget,
    )?;
    let resources = ctx.resources.startup_json()?;
    let identity = RoventHistoricalChildIdentity {
        boot_id: boot.trim().into(),
        pid,
        start_ticks: membership.process_start_ticks,
        cgroup_path: resources["path"]
            .as_str()
            .ok_or_else(|| invalid("resource path"))?
            .into(),
        cgroup_device: count(&resources["device"])?,
        cgroup_inode: count(&resources["inode"])?,
    };
    *ctx.child_identity
        .lock()
        .map_err(|_| invalid("peer identity poisoned"))? = Some(identity);
    Ok((pid, membership.process_start_ticks))
}
fn read_frame(
    stream: &mut UnixStream,
    ctx: &Broker,
    total: &mut u64,
) -> Result<Option<Vec<u8>>, ArenaError> {
    read_unix_frame(
        stream,
        UnixFrameReadScope {
            stop: &ctx.stop,
            cancel: &ctx.cancel,
            whole_until: ctx.whole_until,
            max_total_bytes: ctx.registration.max_rpc_bytes,
            max_frame_bytes: MAX_TRAINER_RPC_BYTES,
            errors: UnixFrameReadErrors {
                partial_stop: "partial RPC at stop",
                deadline_cancel: "RPC original whole deadline/cancel",
                partial_eof: "partial RPC EOF",
                byte_overflow: "RPC overflow",
                byte_frame_bound: "RPC shared byte/frame credit",
                pipelined: "pipelined RPC unsupported",
                read: "current peer RPC read",
            },
        },
        total,
    )
}
fn serve(ctx: Broker) -> Result<(), ArenaError> {
    ctx.listener
        .set_nonblocking(true)
        .map_err(|_| invalid("RPC nonblocking"))?;
    let (mut stream, _) = loop {
        if ctx.stop.load(Ordering::Acquire) {
            return Ok(());
        }
        check(
            Instant::now() < ctx.until && !ctx.cancel.load(Ordering::Acquire),
            "RPC admission deadline/cancel",
        )?;
        match ctx.listener.accept() {
            Ok(v) => break v,
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                thread::park_timeout(Duration::from_millis(2))
            }
            Err(_) => return Err(invalid("RPC accept")),
        }
    };
    let (pid, start) = peer_admit(&stream, &ctx)?;
    stream
        .set_read_timeout(Some(Duration::from_millis(25)))
        .and_then(|_| stream.set_write_timeout(Some(Duration::from_millis(25))))
        .map_err(|_| invalid("finite RPC timeouts"))?;
    let mut total = 0;
    while !ctx.stop.load(Ordering::Acquire) {
        let Some(raw) = read_frame(&mut stream, &ctx, &mut total)? else {
            break;
        };
        let request: Request = decode(&raw)?;
        check(
            request.schema == ROVENT_RECOVERED_RPC
                && request.window_id == ctx.window.window_id
                && request.startup_sha256 == ctx.startup_sha256
                && request.sequence > 0
                && request.sequence <= ctx.registration.max_rpc_messages
                && process_start(pid, &ctx.budget)? == start
                && (Instant::now() < ctx.until || request.operation == "end_window"),
            "RPC current window/peer/sequence/execution deadline",
        )?;
        let mut ledger = ctx.ledger.lock().map_err(|_| invalid("ledger poisoned"))?;
        check(
            request.sequence == ledger.next_sequence,
            "RPC duplicate/skipped sequence",
        )?;
        let payload = dispatch(&ctx, &mut ledger, &request.operation, &request.payload)?;
        ledger.next_sequence = ledger
            .next_sequence
            .checked_add(1)
            .ok_or_else(|| invalid("sequence overflow"))?;
        let ack=serde_json::to_vec(&json!({"schema":ROVENT_RECOVERED_ACK,"window_id":ctx.window.window_id,
            "sequence":request.sequence,"operation":request.operation,"accepted":true,"payload":payload})).map_err(|e|invalid(e.to_string()))?;
        let ack = prepare_unix_ack(ack, MAX_TRAINER_RPC_BYTES, "ack frame credit")?;
        charge_unix_ack(
            &mut total,
            &ack,
            ctx.registration.max_rpc_bytes,
            "RPC outbound byte overflow",
            "combined inbound/outbound RPC byte lease exhausted",
        )?;
        // Charge the full write before issuing it. Partial/lost ACKs never
        // refund bytes or the update that already completed.
        ledger.bytes_spent = total;
        write_unix_ack(&mut stream, &ack, "ack lost; actual work remains charged")?;
    }
    Ok(())
}

fn publish_current_boundary(
    ctx: &Broker,
    ledger: &TrainerLedger,
    owner: &linux::LinuxHeld,
) -> Result<(), ArenaError> {
    check(
        Instant::now() < ctx.until && !ctx.cancel.load(Ordering::Acquire),
        "checkpoint archive admission expired/cancelled",
    )?;
    let child = ctx
        .child_identity
        .lock()
        .map_err(|_| invalid("boundary identity poisoned"))?
        .clone()
        .ok_or_else(|| invalid("checkpoint boundary lacks its actual current child"))?;
    let observed = ctx.resources.verify_child_membership(child.pid)?;
    check(
        observed.process_start_ticks == child.start_ticks,
        "checkpoint peer identity changed",
    )?;
    let total = ctx.resources.observe_total_parent_scope()?;
    check(
        !ctx.registration.require_total_quota || total.verified,
        "boundary parent/child resource scope became unknown",
    )?;
    let mut reserved = owner.control_limit;
    for file in owner
        .sources
        .iter()
        .chain(&owner.checkpoints)
        .chain(&owner.artifacts)
    {
        reserved = reserved
            .checked_add(file.pin.bytes)
            .ok_or_else(|| invalid("owned output byte sum overflow"))?;
    }
    {
        let archives = ctx
            .boundary_archives
            .lock()
            .map_err(|_| invalid("boundary archive custody poisoned"))?;
        for archive in archives.iter() {
            reserved = reserved
                .checked_add(count(&archive.observation()["output_reserved_bytes"])?)
                .ok_or_else(|| invalid("archive output byte sum overflow"))?;
        }
    }
    let maximum_output_bytes = ctx
        .registration
        .max_output_bytes
        .checked_sub(reserved)
        .filter(|bytes| *bytes > 0)
        .ok_or_else(|| {
            invalid("shared original output budget cannot hold a durable checkpoint archive")
        })?;
    let checkpoint_index = ledger
        .checkpoint_states
        .len()
        .checked_sub(1)
        .ok_or_else(|| invalid("boundary checkpoint absent"))?;
    let boundary = ActualRoventCheckpointBoundary {
        ctx,
        ledger,
        held: owner,
        checkpoint_index,
    };
    let target = RoventCheckpointArchiveTarget {
        parent_directory: owner
            .output
            .try_clone()
            .map_err(|_| invalid("boundary output parent FD clone"))?,
        parent_path: ctx.registration.output_root.clone(),
        leaf_name: format!("checkpoint-archive-{:06}", checkpoint_index + 1),
        maximum_output_bytes,
        deadline: ctx.until,
        cancel: Arc::clone(&ctx.cancel),
        read_budget: Arc::clone(&ctx.budget),
    };
    match super::checkpoint_publication::publish_rovent_training_checkpoint_boundary(
        &boundary, target,
    ) {
        Ok(archive) => match ctx.boundary_archives.lock() {
            Ok(mut owners) => {
                owners.push(archive);
                Ok(())
            }
            Err(poisoned) => {
                poisoned.into_inner().push(archive);
                Err(invalid(
                    "boundary archive custody poisoned after publication; retained",
                ))
            }
        },
        Err(failure) => {
            match ctx.boundary_archive_failures.lock() {
                Ok(mut owners) => owners.push(failure),
                Err(poisoned) => poisoned.into_inner().push(failure),
            };
            Err(invalid(
                "atomic checkpoint boundary publication failed; original partial archive custody retained",
            ))
        }
    }
}

fn dispatch(
    ctx: &Broker,
    ledger: &mut TrainerLedger,
    operation: &str,
    p: &Value,
) -> Result<Value, ArenaError> {
    let train = ctx.registration.action == RoventRecoveredTrainingAction::Train;
    check(
        train
            || matches!(
                operation,
                "begin_window"
                    | "end_window"
                    | "validate_checkpoint_state"
                    | "validate_learned_export"
            ),
        "export owner cannot consume samples/update/write a training checkpoint",
    )?;
    match operation {
        "begin_window" => {
            let auxiliary = ctx
                .r_auxiliary_ledger
                .lock()
                .map_err(|_| invalid("R auxiliary custody poisoned"))?;
            validate_auxiliary_fields(
                p,
                &[
                    "schema",
                    "recipe_sha256",
                    "stage_id",
                    "training_profile_sha256",
                    "corpus_sha256",
                    "collection_source_sha256",
                    "collection_runtime_library_sha256",
                    "trainer_source_sha256",
                    "trainer_python_executable_sha256",
                    "trainer_dependency_set_sha256",
                    "training_model_semantics_sha256",
                    "completed_updates",
                    "ancestor_updates",
                    "frozen_pc_sha256",
                ],
                auxiliary.as_ref(),
            )?;
            check(
                !ledger.begun
                    && !ledger.ended
                    && p["schema"] == "rz-rovent-trainer-begin/2"
                    && (matches!(
                        ctx.registration.initialization_source,
                        RoventTrainingInputSource::Initialize { .. }
                    ) || ledger.checkpoint_checked)
                    && p["recipe_sha256"] == ctx.window.recipe_sha256
                    && p["stage_id"] == ctx.window.stage_id
                    && p["training_profile_sha256"] == ledger.profile_sha256
                    && p["corpus_sha256"] == ledger.corpus_sha256
                    && p["collection_source_sha256"] == ctx.window.collection_source_sha256
                    && p["collection_runtime_library_sha256"]
                        == ctx.window.collection_runtime_library_sha256
                    && p["trainer_source_sha256"] == ctx.window.trainer_source_sha256
                    && p["trainer_python_executable_sha256"]
                        == ctx.window.trainer_python_executable_sha256
                    && p["trainer_dependency_set_sha256"]
                        == ctx.window.trainer_dependency_set_sha256
                    && p["training_model_semantics_sha256"]
                        == ctx.window.training_model_semantics_sha256
                    && p["completed_updates"] == json!(ledger.initial)
                    && p["ancestor_updates"] == json!(ledger.ancestors)
                    && p["frozen_pc_sha256"] == json!(ctx.window.frozen_pc_sha256),
                "new explicit trainer/collection identity begin mismatch",
            )?;
            ledger.begun = true;
            Ok(json!({}))
        }
        "consume_sample" => {
            if let Some(auxiliary) = ctx
                .r_auxiliary_ledger
                .lock()
                .map_err(|_| invalid("R auxiliary custody poisoned"))?
                .as_ref()
            {
                check(
                    !auxiliary.is_pending(),
                    "new P/C sample cannot replace uncommitted R auxiliary work",
                )?;
                check(
                    auxiliary
                        .dataset()
                        .joint_eligible_current_pc_origins()?
                        .contains(sha(&p["origin_sha256"])?),
                    "P/C/V sample is excluded by the original joint PC/R leakage boundary",
                )?;
            }
            ledger.consume_with_schema(p, "rz-rovent-trainer-consume/2")
        }
        "consume_auxiliary" => {
            fields(p, &["schema", "evidence"])?;
            check(
                p["schema"] == "rz-rovent-trainer-consume-auxiliary/1"
                    && ctx.window.phase == "pc"
                    && ledger.begun
                    && !ledger.ended
                    && ledger.pending.len() == 8
                    && valid_pc_rows(ledger)? > 0
                    && ledger.samples_admitted < count(&ledger.window["max_samples"])?,
                "R auxiliary requires a valid pending P/C group and remaining shared sample credit",
            )?;
            let mut auxiliary = ctx
                .r_auxiliary_ledger
                .lock()
                .map_err(|_| invalid("R auxiliary custody poisoned"))?;
            auxiliary
                .as_mut()
                .ok_or_else(|| invalid("R auxiliary disabled or missing original dataset owner"))?
                .consume(&p["evidence"])?;
            ledger.samples_admitted = ledger
                .samples_admitted
                .checked_add(1)
                .ok_or_else(|| invalid("shared PC/R sample count overflow"))?;
            Ok(json!({}))
        }
        "completed_update" => {
            let mut auxiliary = ctx
                .r_auxiliary_ledger
                .lock()
                .map_err(|_| invalid("R auxiliary custody poisoned"))?;
            let is_pc = p["event"]["role"] != "validator";
            if let Some(auxiliary) = auxiliary.as_ref() {
                auxiliary.validate_pc_origins(ledger)?;
                if is_pc {
                    auxiliary
                        .validate_update(&p["event"]["r_auxiliary"], valid_pc_rows(ledger)?)?;
                } else {
                    check(
                        !auxiliary.is_pending()
                            && p["event"]
                                .as_object()
                                .is_some_and(|event| !event.contains_key("r_auxiliary")),
                        "V cannot consume or report auxiliary optimization",
                    )?;
                }
            }
            ledger.update_with_schema(
                p,
                "rz-rovent-trainer-update/2",
                "rz-rovent-recovered-learner-update/1",
            )?;
            if is_pc && let Some(auxiliary) = auxiliary.as_mut() {
                auxiliary.commit_update()?;
            }
            if let Some(auxiliary) = auxiliary.as_ref() {
                auxiliary.validate_joint_retained(ledger)?;
            }
            let raw=serde_json::to_vec(&json!({"schema":ROVENT_RECOVERED_UPDATE,"execution_id":ctx.registration.execution_id,
                "window_id":ctx.window.window_id,"sequence":ledger.updates.len(),"source_kind":ROVENT_RECOVERY_SOURCE_KIND,
                "past_native_authority_reissued":false,"update":p})).map_err(|e|invalid(e.to_string()))?;
            let mut owner = ctx.held.lock().map_err(|_| invalid("custody poisoned"))?;
            let owner = owner.as_mut().ok_or_else(|| invalid("custody missing"))?;
            let file = linux::write_new(
                &ctx.registration
                    .control_root
                    .join(format!("update-{:06}.json", ledger.updates.len())),
                &raw,
                owner,
                true,
            )?;
            let ack = json!({"sha256":file.pin.sha256,"bytes":file.pin.bytes});
            owner.journals.push(file);
            Ok(ack)
        }
        "checkpoint_published" => {
            {
                let auxiliary = ctx
                    .r_auxiliary_ledger
                    .lock()
                    .map_err(|_| invalid("R auxiliary custody poisoned"))?;
                validate_auxiliary_fields(
                    p,
                    &[
                        "schema",
                        "checkpoint",
                        "completed_updates",
                        "sampler_state_sha256",
                        "model_state_sha256",
                        "optimizer_state_sha256",
                        "scheduler_state_sha256",
                        "rng_state_sha256",
                        "cost",
                        "pending_microbatches",
                    ],
                    auxiliary.as_ref(),
                )?;
                if let Some(auxiliary) = auxiliary.as_ref() {
                    auxiliary.validate_joint_retained(ledger)?;
                }
            }
            check(
                ledger.begun
                    && !ledger.ended
                    && ledger.pending.is_empty()
                    && p["schema"] == "rz-rovent-trainer-checkpoint/2"
                    && p["completed_updates"] == json!(ledger.completed)
                    && p["pending_microbatches"] == 0,
                "new committed checkpoint boundary",
            )?;
            for key in [
                "model_state_sha256",
                "optimizer_state_sha256",
                "scheduler_state_sha256",
                "rng_state_sha256",
                "sampler_state_sha256",
            ] {
                sha(&p[key])?;
            }
            ledger.observe_cost(&p["cost"])?;
            if let Some(update) = ledger.updates.last() {
                for key in [
                    "model_state_sha256",
                    "optimizer_state_sha256",
                    "scheduler_state_sha256",
                    "rng_state_sha256",
                ] {
                    check(
                        update[key] == p[key],
                        "checkpoint changed committed update state",
                    )?;
                }
            }
            let pin: RoventTrainingFilePin = serde_json::from_value(p["checkpoint"].clone())
                .map_err(|e| invalid(e.to_string()))?;
            check(
                pin.path.parent() == Some(ctx.registration.checkpoint_directory.as_path())
                    && pin.path.extension().is_some_and(|e| e == "pt")
                    && ctx
                        .checkpoint
                        .as_ref()
                        .is_none_or(|source| source.tensor_file().0.path != pin.path)
                    && !ledger
                        .checkpoints
                        .iter()
                        .any(|old| old["path"] == p["checkpoint"]["path"]),
                "checkpoint path aliases input/escapes new owner",
            )?;
            let (file, _) = own(&pin, MAX_CHECKPOINT, false, &ctx.budget)?;
            let mut owner = ctx.held.lock().map_err(|_| invalid("custody poisoned"))?;
            let owner = owner.as_mut().ok_or_else(|| invalid("custody missing"))?;
            owner.checkpoints.push(file);
            let raw=serde_json::to_vec(&json!({"schema":ROVENT_RECOVERED_RETIREMENT,"execution_id":ctx.registration.execution_id,
                "window_id":ctx.window.window_id,"source_kind":ROVENT_RECOVERY_SOURCE_KIND,"past_native_authority_reissued":false,
                "checkpoint":p})).map_err(|e|invalid(e.to_string()))?;
            let receipt = linux::write_new(
                &ctx.registration.control_root.join(format!(
                    "checkpoint-{:06}.json",
                    ledger.checkpoints.len() + 1
                )),
                &raw,
                owner,
                true,
            )?;
            ledger.checkpoint_receipts.push(json!(receipt.pin));
            owner.journals.push(receipt);
            ledger.checkpoints.push(p["checkpoint"].clone());
            ledger.checkpoint_states.push(p.clone());
            // This must commit before the empty ACK lets the child resume.
            // Its archive records an unresolved current process scope, not a
            // fabricated successful terminal/supervisor receipt.
            publish_current_boundary(ctx, ledger, owner)?;
            Ok(json!({}))
        }
        "end_window" => {
            check(
                ctx.r_auxiliary_ledger
                    .lock()
                    .map_err(|_| invalid("R auxiliary custody poisoned"))?
                    .as_ref()
                    .is_none_or(|auxiliary| !auxiliary.is_pending()),
                "end cannot discard pending R auxiliary work",
            )?;
            fields(
                p,
                &[
                    "schema",
                    "status",
                    "completed_updates",
                    "dispatched_updates",
                    "cost",
                    "pending_microbatches",
                    "quarantined",
                    "physical_completion",
                ],
            )?;
            check(
                ledger.begun
                    && !ledger.ended
                    && p["schema"] == "rz-rovent-trainer-end/2"
                    && p["completed_updates"] == json!(ledger.completed)
                    && p["dispatched_updates"] == json!(ledger.completed)
                    && p["pending_microbatches"] == 0
                    && p["quarantined"] == false
                    && p["physical_completion"] == "confirmed"
                    && ledger.pending.is_empty()
                    && if train {
                        !ledger.checkpoints.is_empty()
                            && matches!(
                                p["status"].as_str(),
                                Some(
                                    "window_expired_at_boundary"
                                        | "await_new_frozen_v_corpus"
                                        | "declared_stage_updates_complete"
                                        | "sample_group_budget_exhausted"
                                        | "window_update_budget_complete"
                                )
                            )
                    } else {
                        p["status"] == "export_complete"
                            && ledger.export_admitted
                            && ledger.updates.is_empty()
                            && ctx.checkpoint.is_some()
                    },
                "current window end must confirm actual fence/boundary, not historical closure",
            )?;
            ledger.observe_cost(&p["cost"])?;
            ledger.ended = true;
            ledger.end_cost = Some(p["cost"].clone());
            Ok(json!({}))
        }
        "validate_checkpoint_state" | "validate_learned_export" => {
            let result = checkpoint_admission(ctx, ledger, operation, p)?;
            if operation == "validate_checkpoint_state" {
                check(
                    !ledger.checkpoint_checked,
                    "checkpoint admission single use",
                )?;
                ledger.checkpoint_checked = true;
            } else {
                check(!ledger.export_admitted, "export admission single use")?;
                ledger.export_admitted = true;
                ledger.export_admission = Some(result.clone());
            }
            Ok(result)
        }
        _ => Err(invalid("unknown recovered trainer RPC operation")),
    }
}

fn checkpoint_admission(
    ctx: &Broker,
    ledger: &TrainerLedger,
    operation: &str,
    p: &Value,
) -> Result<Value, ArenaError> {
    let source = ctx
        .checkpoint
        .as_ref()
        .ok_or_else(|| invalid("current peer lacks admitted historical checkpoint"))?;
    validate_auxiliary_fields(
        p,
        &[
            "schema",
            "projection",
            "recipe_sha256",
            "stage_id",
            "membership_sha256",
            "ancestor_updates",
            "retained_input_set_sha256",
            "retained_input_count",
            "retained_origin_set_sha256",
            "retained_origin_count",
        ],
        source.r_auxiliary_ledger(),
    )?;
    if let Some(original) = source.r_auxiliary_checkpoint_projection() {
        check(
            p["r_auxiliary"] == *original,
            "current scratch changed the exact original R pool/sampler checkpoint projection",
        )?;
    }
    let publication = source.publication();
    check(
        source.original_window_schema() == ROVENT_TRAIN_WINDOW_V2,
        "unseparated legacy checkpoint identities are historical metadata only",
    )?;
    let boundary = &publication.boundary;
    let resume = operation == "validate_checkpoint_state";
    check(
        (resume && !ledger.begun && !ledger.checkpoint_checked
            || !resume
                && ctx.registration.action == RoventRecoveredTrainingAction::Export
                && ledger.begun
                && !ledger.ended
                && ledger.checkpoint_checked
                && !ledger.export_admitted)
            && p["schema"]
                == if resume {
                    "rz-rovent-trainer-resume/2"
                } else {
                    "rz-rovent-trainer-export/2"
                }
            && p["recipe_sha256"] == publication.recipe_sha256
            && p["stage_id"] == publication.stage_id
            && p["membership_sha256"] == publication.membership_sha256
            && p["ancestor_updates"] == json!(boundary.ancestor_updates)
            && p["retained_input_set_sha256"] == boundary.retained_input_set_sha256
            && p["retained_input_count"] == boundary.retained_input_count
            && p["retained_origin_set_sha256"] == boundary.retained_origin_set_sha256
            && p["retained_origin_count"] == boundary.retained_origin_count,
        "current scratch request differs from pinned historical checkpoint source",
    )?;
    let projection: RoventHistoricalTensorProjection =
        serde_json::from_value(p["projection"].clone()).map_err(|e| invalid(e.to_string()))?;
    let compared = source.compare_tensor_projection(&projection)?;
    let historical_tail_cost_records = source.historical_tail_records();
    let inherited_unknown_tails = historical_tail_cost_records
        .iter()
        .filter(|record| {
            record["checkpoint_publication_sha256"] != source.publication_pin().sha256
                && record["tail"]["status"] == "unknown"
        })
        .cloned()
        .collect::<Vec<_>>();
    let mut result = json!({"schema":if resume{ROVENT_RECOVERED_CHECKPOINT}else{"rz-rovent-recovered-learned-export-admission/1"},
        "source_kind":ROVENT_RECOVERY_SOURCE_KIND,"past_native_authority_reissued":false,"past_physical_completion_reissued":false,
        "historical_physical_scope":"checkpoint_does_not_prove_old_run_closure",
        "current_execution_id":ctx.registration.execution_id,"current_window_id":ctx.window.window_id,
        "checkpoint_publication_sha256":source.publication_pin().sha256,
        "checkpoint_sha256":source.tensor_file().0.sha256,"checkpoint_retirement_sha256":publication.checkpoint_retirement.sha256,
        "original_update_chain_sha256":source.observation()["update_chain_sha256"],
        "closure_sha256":publication.recorded_closure.evidence.sha256,"new_os_reclamation":ctx.reclamation,
        "recipe_sha256":publication.recipe_sha256,"stage_id":publication.stage_id,
        "training_profile_sha256":publication.training_profile_sha256,"completed_updates":boundary.completed_updates,
        "ancestor_updates":boundary.ancestor_updates,"snapshot_cost":boundary.snapshot_cost,
        "recorded_tail_cost":publication.recorded_tail_cost,"inherited_unknown_tails":inherited_unknown_tails,
        "retained_input_set_sha256":boundary.retained_input_set_sha256,"retained_input_count":boundary.retained_input_count,
        "retained_origin_set_sha256":boundary.retained_origin_set_sha256,"retained_origin_count":boundary.retained_origin_count,
        "historical_tail_cost_records":historical_tail_cost_records,"current_peer_tensor_projection":compared});
    if let Some(original) = source.r_auxiliary_checkpoint_projection() {
        result["r_auxiliary"] = original.clone();
    }
    if !resume {
        let mut totals = BTreeMap::new();
        for role in ROLES {
            totals.insert(
                role,
                boundary.completed_updates[role]
                    .checked_add(boundary.ancestor_updates[role])
                    .ok_or_else(|| invalid("ancestry count overflow"))?,
            );
        }
        check(
            boundary.completed_updates == ledger.stage_limits && totals.values().any(|n| *n > 0),
            "export needs complete actual learned stage",
        )?;
        result["completed_updates"] = json!(totals);
        result["pc_training_steps"] = json!(
            totals["proposer"]
                .checked_add(totals["critic"])
                .ok_or_else(|| invalid("PC count overflow"))?
        );
        result["collection_source_sha256"] = json!(publication.collection_source_sha256);
        result["collection_runtime_library_sha256"] =
            json!(publication.collection_runtime_library_sha256);
        result["trainer_python_executable_sha256"] =
            json!(publication.trainer_python_executable_sha256);
        result["training_model_semantics_sha256"] =
            json!(publication.training_model_semantics_sha256);
        result["trainer_source_sha256"] = json!(publication.trainer_source_sha256);
        result["trainer_dependency_set_sha256"] = json!(publication.trainer_dependency_set_sha256);
        result["lineage_teacher_actor_ids"] = json!(linux::lineage_teachers(
            &ledger.recipe,
            &json!(publication.stage_id)
        )?);
        result["nnuec_training"] = source.learned_nnuec_lineage()?;
    }
    Ok(result)
}

fn validate_history(
    reg: &RoventRecoveredTrainingRegistration,
    window: &RoventTrainWindowV2,
    recipe: &Value,
    source: Option<&VerifiedRoventCheckpointSource>,
    reclamation: Option<&VerifiedRoventOldRunReclamation>,
) -> Result<(), ArenaError> {
    let counts_zero = |counts: &BTreeMap<String, u64>| counts.values().all(|v| *v == 0);
    match &reg.initialization_source {
        RoventTrainingInputSource::Initialize { .. } => check(
            source.is_none()
                && reclamation.is_none()
                && counts_zero(&reg.expected_initial_completed)
                && counts_zero(&reg.expected_ancestor_completed),
            "initialize cannot inherit historical counts or authority",
        ),
        RoventTrainingInputSource::Resume { checkpoint }
        | RoventTrainingInputSource::Fork { checkpoint, .. } => {
            let source =
                source.ok_or_else(|| invalid("explicit historical checkpoint source absent"))?;
            let reclaimed = reclamation
                .ok_or_else(|| invalid("old execution lacks actual reclamation owner"))?;
            check(
                source.original_window_schema() == ROVENT_TRAIN_WINDOW_V2
                    && source.publication_pin() == &checkpoint.publication
                    && reclaimed.checkpoint_publication_sha256() == checkpoint.publication.sha256,
                "historical source and current reclamation differ from requested exact pins",
            )?;
            reclaimed.revalidate()?;
            let p = source.publication();
            check(
                p.recipe_sha256 == window.recipe_sha256
                    && p.training_model_semantics_sha256 == window.training_model_semantics_sha256
                    && p.collection_source_sha256 == window.collection_source_sha256
                    && p.collection_runtime_library_sha256
                        == window.collection_runtime_library_sha256
                    && p.trainer_source_sha256 == window.trainer_source_sha256
                    && p.trainer_python_executable_sha256
                        == window.trainer_python_executable_sha256
                    && p.trainer_dependency_set_sha256 == window.trainer_dependency_set_sha256,
                "new trainer/source/runtime or collection identities changed across historical load",
            )?;
            if let RoventTrainingInputSource::Fork {
                parent_stage_id, ..
            } = &reg.initialization_source
            {
                let stages = recipe["stages"]
                    .as_array()
                    .ok_or_else(|| invalid("stage graph"))?;
                let stage = stages
                    .iter()
                    .find(|s| s["stage_id"] == window.stage_id)
                    .ok_or_else(|| invalid("fork stage"))?;
                let old = stages
                    .iter()
                    .find(|s| s["stage_id"] == *parent_stage_id)
                    .ok_or_else(|| invalid("fork parent stage"))?;
                let old_limits: BTreeMap<String, u64> =
                    serde_json::from_value(old["role_updates"].clone())
                        .map_err(|e| invalid(e.to_string()))?;
                check(
                    p.stage_id == *parent_stage_id
                        && stage["parent_stage_id"] == *parent_stage_id
                        && p.boundary.completed_updates == old_limits
                        && counts_zero(&reg.expected_initial_completed),
                    "fork requires a complete declared direct parent stage",
                )?;
                for role in ROLES {
                    check(
                        reg.expected_ancestor_completed[role]
                            == p.boundary.completed_updates[role]
                                .checked_add(p.boundary.ancestor_updates[role])
                                .ok_or_else(|| invalid("ancestor count overflow"))?,
                        "fork ancestry changed",
                    )?;
                }
            } else {
                check(
                    p.stage_id == window.stage_id
                        && p.training_model_epoch == window.training_model_epoch
                        && p.training_profile_sha256 == window.profile_sha256()?
                        && reg.expected_initial_completed == p.boundary.completed_updates
                        && reg.expected_ancestor_completed == p.boundary.ancestor_updates,
                    "resume must preserve historical fixed profile and exact selected snapshot counts",
                )?;
            }
            Ok(())
        }
    }
}

/// Runs only after the caller has built opaque dataset/checkpoint/reclamation
/// owners. The pinned registration is still reread and joined to those owners.
/// No branch reconstructs OriginalRoventCorpusCustody or a past process owner.
pub fn run_rovent_training_recovered(
    launch: RoventRecoveredTrainingLaunch,
    dataset: Arc<VerifiedRoventDataset>,
    checkpoint: Option<Arc<VerifiedRoventCheckpointSource>>,
    reclamation: Option<VerifiedRoventOldRunReclamation>,
    r_dataset: Option<Arc<VerifiedRoventRAuxiliaryDataset>>,
    cancel: Arc<AtomicBool>,
) -> Result<VerifiedRoventRecoveredTrainingWindow, Box<RoventRecoveredTrainingFailure>> {
    let mut process = None;
    let mut summary = None;
    let mut held = None;
    let mut broker = None;
    let shared_held = Arc::new(Mutex::new(None));
    let mut final_journal = None;
    let mut journal_slot = None;
    let child_identity = Arc::new(Mutex::new(None));
    let boundary_archives = Arc::new(Mutex::new(Vec::new()));
    let boundary_archive_failures = Arc::new(Mutex::new(Vec::new()));
    let r_auxiliary_ledger = Arc::new(Mutex::new(None));
    let result = (|| -> Result<RoventRecoveredTrainingRegistration, ArenaError> {
        check(
            launch.started < launch.execution_deadline
                && launch.execution_deadline < launch.whole_deadline
                && Instant::now() < launch.execution_deadline
                && launch.interpreter_version.starts_with("3.")
                && (1..=8).contains(&launch.dependency_directories.len())
                && launch.limits.max_child_processes == 1
                && launch.limits.max_output_bytes <= 128 << 10
                && launch.read_budget.until == launch.whole_deadline
                && Arc::ptr_eq(&launch.read_budget.cancel, &cancel),
            "new finite process/runtime scope",
        )?;
        let window_scope = OriginalProcessWindow::new(
            launch.started,
            launch.execution_deadline,
            launch.whole_deadline,
        )?;
        let raw =
            launch
                .read_budget
                .verify_held(&launch.registration, &launch.registration_pin, true)?;
        let reg: RoventRecoveredTrainingRegistration = decode(&raw)?;
        reg.validate()?;
        let historical_pc_sources = checkpoint
            .as_ref()
            .map(|source| source.historical_pc_datasets())
            .transpose()?
            .unwrap_or_default();
        let all_pc_sources =
            all_dataset_inputs(&dataset, r_dataset.as_deref(), &historical_pc_sources)?;
        check(
            reg.trainer_python_executable == launch.interpreter_pin
                && reg.trainer_python_version == launch.interpreter_version
                && reg.dependency_directories == launch.dependency_directories
                && reg.current_dataset == dataset.registration
                && reg.ancestor_datasets.len() == all_pc_sources.len()
                && reg.ancestor_datasets.iter().all(|declared| {
                    all_pc_sources
                        .get(&declared.publication.sha256)
                        .is_some_and(|owner| declared == &owner.registration)
                })
                && launch.read_budget.observation()["maximum_bytes"] == reg.max_read_bytes,
            "recovered registration differs from current opaque input/runtime/read-budget owners",
        )?;
        launch
            .read_budget
            .verify_held(&launch.interpreter, &launch.interpreter_pin, false)?;
        dataset.revalidate(&launch.read_budget)?;
        for ancestor in all_pc_sources.values() {
            ancestor.revalidate(&launch.read_budget)?;
        }
        if let Some(source) = &checkpoint {
            source.revalidate(&launch.read_budget)?;
        }
        let output = linux::empty_owned_directory(&reg.output_root)?;
        let control = linux::empty_owned_directory(&reg.control_root)?;
        linux::ensure_child_directory(&reg.output_root, &reg.checkpoint_directory)?;
        check(
            reg.receipt_path.parent() == Some(reg.output_root.as_path())
                && reg.receipt_path.extension().is_some_and(|v| v == "json"),
            "receipt path is not a direct owned JSON artifact",
        )?;
        let control_limit = (reg.max_output_bytes / 8).min(64 << 20);
        held = Some(linux::LinuxHeld {
            output,
            control,
            output_path: reg.output_root.clone(),
            control_path: reg.control_root.clone(),
            source: reg.output_root.join("source"),
            inputs: Vec::new(),
            sources: Vec::new(),
            checkpoints: Vec::new(),
            journals: Vec::new(),
            artifacts: Vec::new(),
            partial_outputs: Vec::new(),
            dependency_directories: Vec::new(),
            control_written: 0,
            control_limit,
            device_admission: None,
        });
        let owner = held.as_mut().ok_or_else(|| invalid("new FD owner"))?;
        let recipe = load(&reg.recipe, owner, &launch.read_budget)?;
        let membership = load(&reg.membership, owner, &launch.read_budget)?;
        let spec_raw = load(&reg.window_spec, owner, &launch.read_budget)?;
        let spec: RoventTrainWindowV2 =
            serde_json::from_value(spec_raw).map_err(|e| invalid(e.to_string()))?;
        reg.join_window(&spec)?;
        check(
            reg.r_auxiliary_dataset
                .enabled()
                .map(|declared| {
                    r_dataset
                        .as_ref()
                        .is_some_and(|owner| owner.registration() == declared)
                })
                .unwrap_or(r_dataset.is_none()),
            "R registration differs from the retained original dataset owner",
        )?;
        if let Some(source) = &r_dataset {
            check(
                source.joint_current_pc_publication_sha256()?
                    == dataset.registration.publication.sha256,
                "enabled R owner is not joint-bound to this exact current P/C publication",
            )?;
            source.revalidate(&launch.read_budget)?;
        }
        *r_auxiliary_ledger
            .lock()
            .map_err(|_| invalid("R auxiliary custody poisoned"))? = new_auxiliary_ledger(
            &reg,
            &spec,
            r_dataset.as_ref().map(Arc::clone),
            checkpoint.as_deref(),
        )?;
        check(
            recipe["schema"] == "rz-rovent-generation-series/1"
                && digest_value("rz-rovent-generation-series/1", &recipe)? == spec.recipe_sha256
                && recipe["source_sha256"] == spec.collection_source_sha256
                && recipe["model_semantics_sha256"] == spec.training_model_semantics_sha256
                && launch
                    .execution_deadline
                    .duration_since(launch.started)
                    .as_nanos()
                    <= u128::from(spec.max_wall_ns),
            "registered recipe or original elapsed training window changed",
        )?;
        let _ = load(&reg.rules_profile, owner, &launch.read_budget)?;
        check(
            launch.resources.policy() == &reg.resources_policy,
            "current applied resource policy differs from registration",
        )?;
        let total_before = launch.resources.observe_total_parent_scope()?;
        check(
            !reg.require_total_quota || total_before.verified,
            "strict aggregate parent/child resources unknown",
        )?;
        validate_history(
            &reg,
            &spec,
            &recipe,
            checkpoint.as_deref(),
            reclamation.as_ref(),
        )?;
        let mut export_request = None;
        if let Some(pin) = &reg.export_request {
            let request = load(pin, owner, &launch.read_budget)?;
            fields(
                &request,
                &["schema", "kind", "directory", "max_output_bytes"],
            )?;
            let directory = PathBuf::from(
                request["directory"]
                    .as_str()
                    .ok_or_else(|| invalid("export directory"))?,
            );
            check(
                request["schema"] == "rz-rovent-recovered-export-request/1"
                    && matches!(request["kind"].as_str(), Some("collection" | "deployment"))
                    && linux::path_ok(&directory)
                    && directory.parent() == Some(reg.output_root.as_path())
                    && directory != reg.checkpoint_directory
                    && directory != owner.source
                    && fs::symlink_metadata(&directory)
                        .is_err_and(|e| e.kind() == std::io::ErrorKind::NotFound)
                    && count(&request["max_output_bytes"])?
                        <= reg
                            .max_output_bytes
                            .checked_sub(control_limit)
                            .ok_or_else(|| invalid("output reserve"))?,
                "new export declaration/path/output bound",
            )?;
            export_request = Some(request);
        }
        match &reg.initialization_source {
            RoventTrainingInputSource::Initialize {
                pc_checkpoint,
                nnuec_checkpoint,
            } => {
                for pin in [pc_checkpoint, nnuec_checkpoint] {
                    let (file, _) = own(pin, MAX_CHECKPOINT, false, &launch.read_budget)?;
                    owner.inputs.push(file);
                }
            }
            RoventTrainingInputSource::Resume { .. } | RoventTrainingInputSource::Fork { .. } => {
                let source = checkpoint
                    .as_ref()
                    .ok_or_else(|| invalid("checkpoint owner missing"))?;
                launch.read_budget.verify_held(
                    source.tensor_file().1,
                    source.tensor_file().0,
                    false,
                )?;
            }
        }
        for endpoint in &reg.runtime_endpoints {
            let (file, _) = own(
                &endpoint.artifact,
                MAX_CHECKPOINT,
                false,
                &launch.read_budget,
            )?;
            owner.inputs.push(file);
        }
        let sources = install_sources(owner, &reg, launch.execution_deadline, &cancel)?;
        let mut dependencies = Vec::new();
        for path in &launch.dependency_directories {
            check(linux::path_ok(path), "dependency directory path")?;
            let file = directory_file(&open_directory(path)?)?;
            let meta = file
                .metadata()
                .map_err(|_| invalid("dependency identity"))?;
            dependencies.push(json!({"path":path,"device":meta.dev(),"inode":meta.ino()}));
            owner
                .dependency_directories
                .push((file, path.clone(), meta.dev(), meta.ino()));
        }
        let membership_domain = membership["schema"]
            .as_str()
            .filter(|v| {
                matches!(
                    *v,
                    "rz-rovent-trainable-parameter-membership/1"
                        | "rz-rovent-trainable-parameter-membership/2"
                )
            })
            .ok_or_else(|| invalid("membership codec"))?;
        let membership_sha256 = digest_value(membership_domain, &membership)?;
        validate_auxiliary_membership(&spec, &recipe, &membership, &membership_sha256)?;
        let ledger = Arc::new(Mutex::new(new_ledger(
            &spec,
            recipe,
            membership_sha256,
            &dataset,
            reg.expected_initial_completed.clone(),
            reg.expected_ancestor_completed.clone(),
            checkpoint.as_deref(),
        )?));
        journal_slot = Some(Arc::clone(&ledger));
        {
            let common = ledger
                .lock()
                .map_err(|_| invalid("ledger custody poisoned"))?;
            let auxiliary = r_auxiliary_ledger
                .lock()
                .map_err(|_| invalid("R auxiliary custody poisoned"))?;
            if let Some(auxiliary) = auxiliary.as_ref() {
                auxiliary.validate_joint_retained(&common)?;
            }
        }
        let sequence = NEXT_SOCKET
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |v| v.checked_add(1))
            .map_err(|_| invalid("socket identity overflow"))?;
        let socket = reg
            .control_root
            .join(format!("recovered-{}-{sequence}.sock", std::process::id()));
        check(socket.as_os_str().len() < 100, "Unix endpoint bound")?;
        let listener =
            UnixListener::bind(&socket).map_err(|_| invalid("current private socket bind"))?;
        let parent_start = process_start(std::process::id(), &launch.read_budget)?;
        let meta = owner
            .control
            .metadata()
            .map_err(|_| invalid("control identity"))?;
        let (start_ns, execution_ns, whole_ns) = linux::original_ticks(
            launch.started,
            launch.execution_deadline,
            launch.whole_deadline,
        )?;
        // One conservative child read lease is charged before stdin can reach
        // Python. Child accounting cannot refund it or mint another allowance.
        launch.read_budget.reserve(reg.max_child_read_bytes)?;
        let ancestor_pins = dataset_ancestors(&dataset)?
            .keys()
            .cloned()
            .collect::<Vec<_>>();
        let startup = json!({"schema":ROVENT_RECOVERY_STARTUP,"execution_id":reg.execution_id,"window_id":spec.window_id,
            "registration_sha256":launch.registration_pin.sha256,"registration":reg,
            "control":{"socket_path":socket,"parent_pid":std::process::id(),"parent_start_ticks":parent_start,"control_device":meta.dev(),"control_inode":meta.ino()},
            "original_started_ns":start_ns,"execution_until_ns":execution_ns,"whole_until_ns":whole_ns,
            "source_root":owner.source,"package_root":owner.source.join(linux::PACKAGE),"source_files":sources,"dependency_directories":dependencies,
            "runtime_module_pins":reg.runtime_endpoints.iter().map(|e|e.artifact.clone()).collect::<Vec<_>>(),
            "python_runtime":{"version":launch.interpreter_version,"executable":launch.interpreter_pin},"resources":launch.resources.startup_json()?,
            "collection_admission":{"schema":"rz-rovent-recovered-dataset-set/1","current_publication_sha256":dataset.registration.publication.sha256,
                "ancestor_publication_sha256":ancestor_pins,"past_native_authority_reissued":false,"past_physical_completion_reissued":false,
                "historical_evidence_only":true,"execution_authorized":false},
            "checkpoint_admission":checkpoint.as_ref().map(|s|s.observation())});
        let mut startup = startup;
        if let Some(source) = &r_dataset {
            startup["r_auxiliary_admission"] = auxiliary_admission(source)?;
        }
        startup["child_read_credit"] = json!({"reserved_bytes":reg.max_child_read_bytes,
            "scope":"charged_in_parent_shared_budget_no_refund"});
        startup["artifact_watch_profile"] = json!({"schema":"rz-rovent-training-artifact-watch/1",
            "maximum_total_bytes":reg.max_output_bytes.checked_sub(control_limit).ok_or_else(||invalid("shared output control reserve"))?,
            "maximum_file_bytes":MAX_CHECKPOINT,"maximum_files":reg.max_artifact_files,
            "scope":"new_trainer_owned_output_tree_only"});
        let raw = serde_json::to_vec(&startup).map_err(|e| invalid(e.to_string()))?;
        check(
            raw.len() <= crate::MAX_ORIGINAL_INPUT_BYTES,
            "startup stdin bound",
        )?;
        let startup_sha = format!("{:x}", Sha256::digest(&raw));
        let startup_file =
            linux::write_new(&reg.control_root.join("startup.json"), &raw, owner, true)?;
        owner.journals.push(startup_file);
        *shared_held
            .lock()
            .map_err(|_| invalid("custody poisoned"))? = held.take();
        let image_verified = Arc::new(AtomicBool::new(false));
        let stop = Arc::new(AtomicBool::new(false));
        let ctx = Broker {
            listener,
            parent_pid: std::process::id(),
            parent_start,
            interpreter: launch.interpreter_pin.clone(),
            startup_sha256: startup_sha.clone(),
            registration: reg.clone(),
            window: spec.clone(),
            until: launch.execution_deadline,
            whole_until: launch.whole_deadline,
            cancel: Arc::clone(&cancel),
            stop: Arc::clone(&stop),
            image_verified: Arc::clone(&image_verified),
            ledger: Arc::clone(&ledger),
            held: Arc::clone(&shared_held),
            budget: Arc::clone(&launch.read_budget),
            resources: Arc::clone(&launch.resources),
            checkpoint: checkpoint.as_ref().map(Arc::clone),
            child_identity: Arc::clone(&child_identity),
            reclamation: reclamation.as_ref().map(|r| r.observation()),
            dataset: Arc::clone(&dataset),
            boundary_archives: Arc::clone(&boundary_archives),
            boundary_archive_failures: Arc::clone(&boundary_archive_failures),
            _retained_r_dataset: r_dataset.as_ref().map(Arc::clone),
            r_auxiliary_ledger: Arc::clone(&r_auxiliary_ledger),
        };
        let broker_cancel = Arc::clone(&cancel);
        let handle = thread::Builder::new()
            .name("rovent-recovered-rpc".into())
            .spawn(move || {
                let result = serve(ctx);
                if result.is_err() {
                    broker_cancel.store(true, Ordering::Release);
                }
                result
            })
            .map_err(|_| invalid("current broker spawn"))?;
        broker = Some(PendingBroker { handle, stop });
        let args = [
            OsString::from("-I"),
            OsString::from("-S"),
            OsString::from("-B"),
            OsString::from("-c"),
            OsString::from(linux::BOOTSTRAP),
            reg.output_root
                .join("source")
                .join(linux::PACKAGE)
                .into_os_string(),
            OsString::from("rz_pals_model.rovent_recovered_training_child"),
            OsString::from("--execute"),
            OsString::from("--startup-sha256"),
            OsString::from(startup_sha),
            OsString::from("--startup-bytes"),
            OsString::from(raw.len().to_string()),
            OsString::from("--execution-until-ns"),
            OsString::from(execution_ns.to_string()),
            OsString::from("--whole-until-ns"),
            OsString::from(whole_ns.to_string()),
        ];
        let mut verify_image = |image: &File| {
            launch
                .read_budget
                .verify_held(image, &launch.interpreter_pin, false)
                .map(|_| {
                    image_verified.store(true, Ordering::Release);
                })
                .map_err(|_| ())
        };
        let cwd = shared_held
            .lock()
            .map_err(|_| invalid("custody poisoned"))?
            .as_ref()
            .ok_or_else(|| invalid("custody missing"))?
            .output
            .try_clone()
            .map_err(|_| invalid("output FD clone"))?;
        let artifact_watch = RoventTrainingArtifactTreeWatch::registered(
            reg.max_output_bytes
                .checked_sub(control_limit)
                .ok_or_else(|| invalid("output reserve"))?,
            MAX_CHECKPOINT,
            reg.max_artifact_files,
        )?;
        // The broker owns metadata/RPC and checkpoint publication, not GPU work.
        // Reserve only after preflight and keep the same owner through any
        // supervisor error: process=None is not proof that no child was issued.
        {
            let mut slot = shared_held
                .lock()
                .map_err(|_| invalid("custody poisoned before device dispatch"))?;
            let owner = slot
                .as_mut()
                .ok_or_else(|| invalid("custody missing before device dispatch"))?;
            check(
                owner.device_admission.is_none(),
                "current device reservation already present before dispatch",
            )?;
            owner.device_admission = linux::ProcessDeviceAdmission::acquire_for_dispatch(
                &spec.device,
                &cancel,
                launch.execution_deadline,
                launch.whole_deadline,
            )?;
        }
        process = Some(
            supervise_rovent_training_numerical_input_in_original_window(
                &launch.interpreter,
                &args,
                &cwd,
                launch.limits,
                Some(&cancel),
                &artifact_watch,
                OriginalProcessInput {
                    bytes: &raw,
                    window: window_scope,
                },
                &mut verify_image,
                OriginalNumericalEnvironment::Fp32Tf32Off,
            )?,
        );
        if process
            .as_ref()
            .is_none_or(|owner| !owner.transport_complete())
        {
            // Stop an in-progress parent archive as soon as the child/tree
            // supervisor has failed, including output file/byte exhaustion.
            cancel.store(true, Ordering::Release);
        }
        if let Some(b) = broker.as_ref() {
            b.stop.store(true, Ordering::Release);
        }
        while broker.as_ref().is_some_and(|b| !b.handle.is_finished())
            && Instant::now() < launch.whole_deadline
        {
            thread::park_timeout(Duration::from_millis(2));
        }
        if broker.as_ref().is_some_and(|b| b.handle.is_finished()) {
            broker
                .take()
                .ok_or_else(|| invalid("broker absent"))?
                .handle
                .join()
                .map_err(|_| invalid("broker panic"))??;
        } else {
            return Err(invalid("current broker closure unknown"));
        }
        held = shared_held
            .lock()
            .map_err(|_| invalid("custody poisoned"))?
            .take();
        let journal = ledger
            .lock()
            .map_err(|_| invalid("ledger poisoned"))?
            .clone();
        final_journal = Some(journal.clone());
        let original = process
            .as_ref()
            .ok_or_else(|| invalid("original current process absent"))?;
        original.checked_loaded_image()?;
        original.checked_timing()?;
        check(
            original.transport_complete()
                && journal.ended
                && journal.pending.is_empty()
                && (reg.action != RoventRecoveredTrainingAction::Export
                    || journal.checkpoint_checked
                        && journal.export_admitted
                        && journal.updates.is_empty()),
            "current process/physical/ledger closure incomplete",
        )?;
        let mut terminal_summary: Value = decode(&original.process().stdout)?;
        summary = Some(terminal_summary.clone());
        let terminal_pin: RoventTrainingFilePin =
            serde_json::from_value(terminal_summary["receipt"].clone())
                .map_err(|e| invalid(e.to_string()))?;
        check(
            terminal_pin.path == reg.receipt_path,
            "terminal receipt escaped current owner",
        )?;
        let (terminal_file, terminal_raw) =
            own(&terminal_pin, MAX_METADATA_BYTES, true, &launch.read_budget)?;
        let owner = held
            .as_mut()
            .ok_or_else(|| invalid("terminal FD owner absent"))?;
        owner.journals.push(terminal_file);
        let terminal: Value = decode(&terminal_raw)?;
        for value in [&terminal_summary, &terminal] {
            check(
                value["schema"] == ROVENT_RECOVERED_RESULT
                    && value["execution_id"] == reg.execution_id
                    && value["window_id"] == spec.window_id
                    && value["completed_updates"] == json!(journal.completed)
                    && value["physical_completion"] == "confirmed"
                    && value["arena_eligible"] == false
                    && value["status"] == "completed",
                "new child terminal identity/physical state mismatch",
            )?;
            fields(
                &value["child_read_accounting"],
                &["charged_bytes", "unknown", "maximum_bytes"],
            )?;
            check(
                value["child_read_accounting"]["maximum_bytes"] == reg.max_child_read_bytes
                    && count(&value["child_read_accounting"]["charged_bytes"])?
                        <= reg.max_child_read_bytes
                    && value["child_read_accounting"]["unknown"].is_boolean(),
                "child exceeded/lost its single reserved read lease",
            )?;
            if value["child_read_accounting"]["unknown"] == true {
                launch.read_budget.uncertain.store(true, Ordering::Release);
            }
        }
        check(
            terminal_summary["child_read_accounting"] == terminal["child_read_accounting"],
            "child terminal read accounting differs from pinned receipt",
        )?;
        if let Some(request) = &export_request {
            verify_export(
                &reg,
                request,
                &terminal,
                &journal,
                owner,
                &launch.read_budget,
            )?;
        }
        let closed = launch.resources.observe_closed()?;
        let total_after = launch.resources.observe_total_parent_scope()?;
        check(
            !reg.require_total_quota || total_after.verified,
            "aggregate resource observation changed",
        )?;
        for (file, pin) in [
            (&launch.registration, &launch.registration_pin),
            (&launch.interpreter, &launch.interpreter_pin),
        ] {
            launch.read_budget.verify_held(file, pin, false)?;
        }
        for file in owner
            .inputs
            .iter()
            .chain(&owner.sources)
            .chain(&owner.checkpoints)
            .chain(&owner.journals)
            .chain(&owner.artifacts)
        {
            check(
                file.identity
                    == FileIdentity::from(
                        &file
                            .file
                            .metadata()
                            .map_err(|_| invalid("retained FD metadata"))?,
                    ),
                "retained inode changed",
            )?;
            launch
                .read_budget
                .verify_held(&file.file, &file.pin, false)?;
        }
        for (fd, path, device, inode) in &owner.dependency_directories {
            let retained = fd
                .metadata()
                .map_err(|_| invalid("retained dependency identity"))?;
            let named = directory_file(&open_directory(path)?)?
                .metadata()
                .map_err(|_| invalid("named dependency identity"))?;
            check(
                retained.dev() == *device
                    && retained.ino() == *inode
                    && named.dev() == *device
                    && named.ino() == *inode,
                "dependency directory replaced",
            )?;
        }
        dataset.revalidate(&launch.read_budget)?;
        for ancestor in dataset_ancestors(&dataset)?.values() {
            ancestor.revalidate(&launch.read_budget)?;
        }
        if let Some(source) = &checkpoint {
            source.revalidate(&launch.read_budget)?;
        }
        if let Some(source) = &r_dataset {
            source.revalidate(&launch.read_budget)?;
        }
        if let Some(reclaimed) = &reclamation {
            reclaimed.revalidate()?;
        }
        for archive in boundary_archives
            .lock()
            .map_err(|_| invalid("boundary archive custody poisoned"))?
            .iter()
        {
            archive.revalidate(&launch.read_budget)?;
        }
        check(
            Instant::now() < launch.whole_deadline,
            "original preservation deadline expired",
        )?;
        terminal_summary["read_accounting"] = launch.read_budget.observation();
        terminal_summary["artifact_watch_profile"] = startup["artifact_watch_profile"].clone();
        terminal_summary["read_accounting_scope"] = json!({"parent":"registered_pinned_descriptors_and_bounded_kernel_reads",
            "child":"declared_pinned_source_and_artifact_endpoints",
            "torch_private_snapshot_loader_io":"unobserved","transitive_import_and_mapped_io":"unobserved",
            "global_os_io_enforced":false,"reserved_child_credit_refunded":false});
        terminal_summary["parent_process_receipt"] =
            serde_json::to_value(&original.process().receipt)
                .map_err(|e| invalid(e.to_string()))?;
        terminal_summary["parent_process_observation"] =
            serde_json::to_value(original.observation()).map_err(|e| invalid(e.to_string()))?;
        terminal_summary["resource_readback"] =
            serde_json::to_value(closed).map_err(|e| invalid(e.to_string()))?;
        terminal_summary["total_resource_scope"] = json!({"required":reg.require_total_quota,"before":total_before,"after":total_after,"controlled_performance_eligible":false});
        terminal_summary["historical_tail_cost_records"] = checkpoint
            .as_ref()
            .map(|s| s.historical_tail_records())
            .unwrap_or_default()
            .into();
        let has_unknown = terminal_summary["historical_tail_cost_records"]
            .as_array()
            .is_some_and(|records| {
                records
                    .iter()
                    .any(|record| record["tail"]["status"] == "unknown")
            });
        terminal_summary["parent_cost_observation"] = json!({"current_stage_cost":journal.end_cost,
            "aggregate_total_status":if has_unknown{"lower_bound_only"}else{"known_registered_windows"},
            "historical_unknown_preserved":has_unknown,"cumulative_snapshots_summed":false,
            "tail_cost_refunded":false});
        terminal_summary["checkpoint_boundary_archives"] = json!(
            boundary_archives
                .lock()
                .map_err(|_| invalid("boundary archive custody poisoned"))?
                .iter()
                .map(|owner| owner.observation())
                .collect::<Vec<_>>()
        );
        if let Some(auxiliary) = r_auxiliary_ledger
            .lock()
            .map_err(|_| invalid("R auxiliary custody poisoned"))?
            .as_ref()
        {
            terminal_summary["r_auxiliary"] = auxiliary.observation();
        }
        terminal_summary["historical_evidence_scope"] = json!({"past_native_authority_reissued":false,"past_physical_completion_reissued":false,"source_kind":ROVENT_RECOVERY_SOURCE_KIND});
        terminal_summary["parent_device_admission"] = json!({"scope":"process_local_logical_device_only","host_global_exclusivity":"unknown","physical_sharing":"unknown"});
        if let Some(device) = owner.device_admission.as_mut() {
            device.release_confirmed()?;
        }
        summary = Some(terminal_summary);
        Ok(reg)
    })();
    match result {
        Ok(registration) => Ok(VerifiedRoventRecoveredTrainingWindow {
            registration,
            summary: summary.expect("validated summary"),
            journal: final_journal.expect("validated ledger"),
            process: process.expect("validated current process"),
            launch,
            dataset,
            checkpoint,
            _retained_reclamation: reclamation,
            held: held.expect("retained current FDs"),
            child_identity: child_identity
                .lock()
                .expect("validated child identity mutex")
                .take()
                .expect("admitted current child"),
            _retained_boundary_archives: std::mem::take(
                &mut *boundary_archives
                    .lock()
                    .expect("validated boundary archive custody"),
            ),
            r_auxiliary_ledger: r_auxiliary_ledger
                .lock()
                .expect("validated R auxiliary custody")
                .take(),
            _retained_r_dataset: r_dataset,
        }),
        Err(error) => {
            cancel.store(true, Ordering::Release);
            if let Some(b) = broker.as_ref() {
                b.stop.store(true, Ordering::Release);
            }
            let journal = final_journal
                .as_ref()
                .map(TrainerLedger::summary)
                .or_else(|| {
                    journal_slot
                        .as_ref()
                        .and_then(|s| s.try_lock().ok().map(|v| v.summary()))
                })
                .unwrap_or_else(|| json!({"state":"preserved_partial_or_unknown"}));
            Err(Box::new(RoventRecoveredTrainingFailure {
                error,
                process,
                summary,
                rpc_journal: journal,
                launch,
                dataset,
                checkpoint,
                reclamation,
                _retained_held: held,
                _retained_broker: broker,
                _retained_shared_held: shared_held,
                boundary_archives,
                boundary_archive_failures,
                r_auxiliary_ledger,
                _retained_r_dataset: r_dataset,
            }))
        }
    }
}

fn verify_export(
    reg: &RoventRecoveredTrainingRegistration,
    request: &Value,
    terminal: &Value,
    ledger: &TrainerLedger,
    owner: &mut linux::LinuxHeld,
    budget: &RoventRecoveryReadBudget,
) -> Result<(), ArenaError> {
    let directory = PathBuf::from(
        request["directory"]
            .as_str()
            .ok_or_else(|| invalid("export directory"))?,
    );
    let pin: RoventTrainingFilePin = serde_json::from_value(terminal["export_receipt"].clone())
        .map_err(|e| invalid(e.to_string()))?;
    check(
        pin.path == directory.join("rovent-export.json"),
        "export terminal receipt escaped current output directory",
    )?;
    let (file, raw) = own(&pin, MAX_METADATA_BYTES, true, budget)?;
    owner.artifacts.push(file);
    let receipt: Value = decode(&raw)?;
    verify_learned_export_proof(
        &receipt["historical_training_admission"],
        ledger
            .export_admission
            .as_ref()
            .ok_or_else(|| invalid("new learned admission absent"))?,
    )?;
    check(
        terminal["result"] == receipt
            && receipt["schema"] == "rz-rovent-tensor-lifecycle-export/2"
            && receipt["kind"] == request["kind"]
            && receipt["recipe_sha256"] == ledger.window["recipe_sha256"]
            && receipt["stage_id"] == ledger.window["stage_id"]
            && receipt["checkpoint"]["sha256"]
                == ledger
                    .export_admission
                    .as_ref()
                    .ok_or_else(|| invalid("learned checkpoint admission absent"))?["checkpoint_sha256"]
            && receipt["actual_export_written"] == true
            && receipt["training_provenance_verified"] == true
            && receipt["optimizer_updates_executed"] == 0
            && receipt["numeric_status"] == "not_run"
            && receipt["cuda_status"] == "not_run"
            && receipt["arena_eligible"] == false,
        "actual export changed historical source/current admission or promoted numeric/GPU status",
    )?;
    let roles = receipt["pc"]["roles"]
        .as_array()
        .ok_or_else(|| invalid("export roles"))?;
    check(
        roles == &vec![json!("proposer"), json!("critic")]
            && receipt["pc"]["validator_present"] == false
            || request["kind"] == "collection"
                && roles == &vec![json!("proposer"), json!("critic"), json!("validator")]
                && receipt["pc"]["validator_present"] == true,
        "deployment must be V-free",
    )?;
    let graphs = receipt["pc"]["graphs"]
        .as_array()
        .filter(|g| (2..=3).contains(&g.len()))
        .ok_or_else(|| invalid("bounded export graphs"))?;
    let mut paths = BTreeSet::from([pin.path.clone()]);
    let mut total = pin.bytes;
    for item in std::iter::once(&receipt["pc"]["manifest"])
        .chain(graphs)
        .chain(std::iter::once(&receipt["nnuec"]))
    {
        let relative = Path::new(
            item["file"]
                .as_str()
                .ok_or_else(|| invalid("export file"))?,
        );
        check(
            !relative.is_absolute()
                && relative
                    .components()
                    .all(|c| matches!(c, Component::Normal(_))),
            "export relative path escaped",
        )?;
        let pin = RoventTrainingFilePin {
            path: directory.join(relative),
            sha256: sha(&item["sha256"])?.into(),
            bytes: count(&item["bytes"])?,
        };
        total = total
            .checked_add(pin.bytes)
            .ok_or_else(|| invalid("export bytes overflow"))?;
        check(
            paths.insert(pin.path.clone())
                && total <= count(&request["max_output_bytes"])?
                && total <= reg.max_output_bytes,
            "export duplicate/output quota",
        )?;
        let (file, _) = own(&pin, MAX_CHECKPOINT, false, budget)?;
        owner.artifacts.push(file);
    }
    Ok(())
}

fn verify_learned_export_proof(proof: &Value, original: &Value) -> Result<(), ArenaError> {
    fields(
        proof,
        &[
            "schema",
            "source_kind",
            "original_parent_admission",
            "completed_updates",
            "pc_training_steps",
            "nnuec_training",
            "historical_nnuec_lineage",
            "cost_state",
            "past_native_authority_reissued",
            "past_physical_completion_reissued",
            "scope",
        ],
    )?;
    check(
        proof["schema"] == "rz-rovent-recovered-learned-export-proof/1"
            && proof["source_kind"] == ROVENT_RECOVERY_SOURCE_KIND
            && proof["original_parent_admission"] == *original
            && proof["completed_updates"] == original["completed_updates"]
            && proof["pc_training_steps"] == original["pc_training_steps"]
            && proof["historical_nnuec_lineage"] == original["nnuec_training"]
            && proof["past_native_authority_reissued"] == false
            && proof["past_physical_completion_reissued"] == false
            && proof["scope"]
                == "current_peer_checked_historical_updates_not_numeric_gpu_or_arena_acceptance",
        "new export proof changed the actual current peer's exact admission",
    )?;
    let lineage = &original["nnuec_training"];
    fields(
        lineage,
        &[
            "schema",
            "completed_nnuec_updates",
            "dataset_manifest_sha256",
            "last_execution_id",
            "last_window_id",
            "historical_metadata_only",
            "past_native_authority_reissued",
            "execution_authorized",
        ],
    )?;
    check(
        lineage["schema"] == "rz-rovent-historical-learned-nnuec-lineage/1"
            && lineage["historical_metadata_only"] == true
            && lineage["past_native_authority_reissued"] == false
            && lineage["execution_authorized"] == false,
        "historical NNUEC lineage scope changed",
    )?;
    let steps = count(&lineage["completed_nnuec_updates"])?;
    check(
        steps <= count(&original["pc_training_steps"])?,
        "NNUEC count exceeds actual P/C steps",
    )?;
    let datasets = lineage["dataset_manifest_sha256"]
        .as_array()
        .filter(|v| v.len() <= 256)
        .ok_or_else(|| invalid("bounded NNUEC corpus set"))?;
    for value in datasets {
        sha(value)?;
    }
    check(
        datasets
            .windows(2)
            .all(|pair| pair[0].as_str() < pair[1].as_str()),
        "NNUEC corpus set is not sorted/unique",
    )?;
    let expected = if steps == 0 {
        check(
            datasets.is_empty()
                && lineage["last_execution_id"].is_null()
                && lineage["last_window_id"].is_null(),
            "untrained NNUEC cannot retain trained lineage",
        )?;
        json!({"kind":"untrained"})
    } else {
        check(
            !datasets.is_empty()
                && lineage["last_execution_id"]
                    .as_str()
                    .is_some_and(generation_identifier)
                && lineage["last_window_id"]
                    .as_str()
                    .is_some_and(generation_identifier),
            "learned NNUEC requires its original update identity",
        )?;
        let dataset = if datasets.len() == 1 {
            sha(&datasets[0])?.to_owned()
        } else {
            digest_value("rz-rovent-nnuec-training-corpus-set/1", datasets)?
        };
        json!({"kind":"learned","steps":steps,"run_id":lineage["last_execution_id"],"dataset_sha256":dataset})
    };
    check(
        proof["nnuec_training"] == expected,
        "writer NNUEC projection changed exact historical update lineage",
    )?;
    let cost = &proof["cost_state"];
    fields(
        cost,
        &[
            "schema",
            "historical_snapshot_cost",
            "historical_tail_cost",
            "inherited_unknown_tails",
            "new_window_deltas",
            "known_lower_bound",
            "total_status",
            "scope",
            "past_native_authority_reissued",
            "execution_authorized",
        ],
    )?;
    validate_cost(&original["snapshot_cost"])?;
    let mut lower = original["snapshot_cost"].clone();
    let known = original["recorded_tail_cost"]["status"] == "known";
    if known {
        validate_cost(&original["recorded_tail_cost"]["delta"])?;
        for name in COST_FIELDS {
            lower[name] = json!(
                count(&lower[name])?
                    .checked_add(count(&original["recorded_tail_cost"]["delta"][name])?)
                    .ok_or_else(|| invalid("recorded known cost overflow"))?
            );
        }
    }
    let unknown = !known
        || original["inherited_unknown_tails"]
            .as_array()
            .is_some_and(|v| !v.is_empty());
    check(
        cost["schema"] == "rz-rovent-recovered-cost-state/1"
            && cost["historical_snapshot_cost"] == original["snapshot_cost"]
            && cost["historical_tail_cost"] == original["recorded_tail_cost"]
            && cost["inherited_unknown_tails"] == original["inherited_unknown_tails"]
            && cost["new_window_deltas"] == json!([])
            && cost["known_lower_bound"] == lower
            && cost["total_status"]
                == if unknown {
                    "lower_bound_only"
                } else {
                    "recorded_scope_known"
                }
            && cost["scope"]
                == "recorded_cost_only_not_old_process_closure_or_new_window_authority"
            && cost["past_native_authority_reissued"] == false
            && cost["execution_authorized"] == false,
        "writer export cost proof erased uncertainty, duplicated a snapshot, or altered recorded costs",
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    fn unknown_cost_export_fixture() -> (Value, Value) {
        let cost = json!({"forward_calls":0,"backward_calls":0,"samples":0,"wall_time_ns":0,
            "partial_accumulation_ns":0,"replay_ns":0,"masked_groups":0,"output_bytes":0});
        let tail = json!({"status":"unknown","reason":"original_window_unclosed_after_boundary","evidence":null});
        let lineage = json!({"schema":"rz-rovent-historical-learned-nnuec-lineage/1","completed_nnuec_updates":0,
            "dataset_manifest_sha256":[],"last_execution_id":null,"last_window_id":null,
            "historical_metadata_only":true,"past_native_authority_reissued":false,"execution_authorized":false});
        let completed = json!({"proposer":4,"critic":4,"validator":0});
        let original = json!({"completed_updates":completed,"pc_training_steps":8,"nnuec_training":lineage,
            "snapshot_cost":cost,"recorded_tail_cost":tail,"inherited_unknown_tails":[]});
        let proof = json!({"schema":"rz-rovent-recovered-learned-export-proof/1","source_kind":"historical_exact_pins",
            "original_parent_admission":original,"completed_updates":completed,"pc_training_steps":8,
            "nnuec_training":{"kind":"untrained"},"historical_nnuec_lineage":lineage,
            "cost_state":{"schema":"rz-rovent-recovered-cost-state/1","historical_snapshot_cost":cost,
                "historical_tail_cost":tail,"inherited_unknown_tails":[],"new_window_deltas":[],"known_lower_bound":cost,
                "total_status":"lower_bound_only","scope":"recorded_cost_only_not_old_process_closure_or_new_window_authority",
                "past_native_authority_reissued":false,"execution_authorized":false},
            "past_native_authority_reissued":false,"past_physical_completion_reissued":false,
            "scope":"current_peer_checked_historical_updates_not_numeric_gpu_or_arena_acceptance"});
        (original, proof)
    }
    #[test]
    fn export_proof_cannot_erase_unknown_boundary_cost_or_mutate_current_ack() {
        let (original, proof) = unknown_cost_export_fixture();
        assert!(verify_learned_export_proof(&proof, &original).is_ok());
        let mut erased = proof.clone();
        erased["cost_state"]["total_status"] = json!("recorded_scope_known");
        assert!(verify_learned_export_proof(&erased, &original).is_err());
        let mut changed = proof;
        changed["original_parent_admission"]["completed_updates"]["critic"] = json!(5);
        assert!(verify_learned_export_proof(&changed, &original).is_err());
    }
    #[test]
    fn export_proof_cannot_reissue_native_authority_or_invent_nnuec_training() {
        let (original, proof) = unknown_cost_export_fixture();
        let mut authority = proof.clone();
        authority["past_native_authority_reissued"] = json!(true);
        assert!(verify_learned_export_proof(&authority, &original).is_err());
        let mut invented = proof;
        invented["nnuec_training"] =
            json!({"kind":"learned","steps":1,"run_id":"invented","dataset_sha256":"a".repeat(64)});
        assert!(verify_learned_export_proof(&invented, &original).is_err());
    }
}
