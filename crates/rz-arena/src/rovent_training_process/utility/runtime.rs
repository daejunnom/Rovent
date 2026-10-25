//! Actual utility-only subprocess owner. Metadata describes historical bytes;
//! only the current held descriptors, authenticated peer, supervisor and kernel
//! resource scope can authorize this window. No P/C/V training lease is reused.
#![cfg(target_os = "linux")]

use super::ancestry::VerifiedUtilityAncestors;
use super::cli::{RoventUtilityCliAction, RoventUtilityCliAdmissionRequest};
use super::execution::{RoventUtilityExecutionAction, RoventUtilityInitializationSource};
#[path = "ipc_endpoint.rs"]
mod ipc_endpoint;

use self::ipc_endpoint::{OwnedUtilityEndpoint, UtilityListener};
use super::ledger::{
    InitialUtilityMeasurement, UtilityCheckpointBoundary, UtilityLedger, UtilityStateSeals,
};
use super::runtime_family::{
    RuntimePublishedUtilityCheckpointArchive as PublishedUtilityCheckpointArchive,
    RuntimeUtilityCheckpointArchiveFailure as UtilityCheckpointArchiveFailure,
    RuntimeUtilityCheckpointSource as VerifiedUtilityCheckpointSource,
    RuntimeUtilityDataset as VerifiedUtilityDataset, RuntimeUtilityDatasetFailure,
    RuntimeUtilityOldRunReclamation as VerifiedUtilityOldRunReclamation,
    RuntimeUtilityRegistration as RoventUtilityExecutionRegistration, RuntimeUtilitySourceFailure,
    RuntimeUtilityWindow as RoventUtilityTrainWindow, UtilityRuntimeFamily,
    publish_closed as publish_closed_utility_checkpoint,
    publish_current as publish_current_utility_checkpoint,
};
use crate::ArenaError;
use crate::pals_collect_v3::training_process::{FileIdentity, directory_file, open_directory};
use crate::process::{
    OriginalNumericalEnvironment, OriginalProcessInput, OriginalProcessOutput,
    OriginalProcessWindow, ProcessLimits, ProcessStop, RoventTrainingArtifactTreeWatch,
    supervise_rovent_training_numerical_input_in_original_window,
};
use crate::rovent_training_process::numerical_child::{
    UnixFrameReadErrors, UnixFrameReadScope, charge_unix_ack, is_direct_parent_group_leader,
    prepare_unix_ack, read_unix_frame, unix_peer_credentials, write_unix_ack,
};
use crate::rovent_training_process::recovery::{
    RoventCheckpointArchiveTarget, RoventHistoricalChildIdentity, RoventRecoveryReadBudget,
};
use crate::rovent_training_process::{RoventTrainingFilePin, RoventTrainingResources, linux};
use rz_experiments::pals_data_v3::{generation_digest, generation_sha};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_json::{Value, json};
use sha2::{Digest as _, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    ffi::OsString,
    fs::File,
    io::{Read, Write},
    os::unix::{fs::MetadataExt, net::UnixStream},
    path::{Path, PathBuf},
    sync::{
        Arc, LazyLock, Mutex,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

const STARTUP: &str = "rz-rovent-utility-trainer-startup/1";
const SOURCE: &str = "rz-rovent-utility-trainer-source-bundle/1";
const RESULT: &str = "rz-rovent-utility-trainer-result/1";
const RPC: &str = "rz-rovent-utility-trainer-rpc/1";
const ACK: &str = "rz-rovent-utility-trainer-ack/1";
const MAX_JSON: u64 = 16 << 20;
const MAX_TENSOR: u64 = 2 << 30;
const MAX_FRAME: usize = 65536;
static NEXT_SOCKET: AtomicU64 = AtomicU64::new(1);
static ACTIVE_WINDOW: AtomicBool = AtomicBool::new(false);
struct OriginalWindowAdmission {
    quarantined: bool,
}
impl OriginalWindowAdmission {
    fn acquire() -> Result<Self, ArenaError> {
        require(
            ACTIVE_WINDOW
                .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
                .is_ok(),
            "another utility window is active or quarantined in this process",
        )?;
        Ok(Self { quarantined: false })
    }
}
impl Drop for OriginalWindowAdmission {
    fn drop(&mut self) {
        if !self.quarantined {
            ACTIVE_WINDOW.store(false, Ordering::Release);
        }
    }
}

fn invalid(reason: impl Into<String>) -> ArenaError {
    ArenaError::Integrity(format!("Rovent utility runtime: {}", reason.into()))
}
fn require(condition: bool, reason: &str) -> Result<(), ArenaError> {
    if condition {
        Ok(())
    } else {
        Err(invalid(reason))
    }
}
fn original_forward_deadline(
    execution_deadline: Instant,
    checkpoint_reserve_ns: u64,
) -> Result<Instant, ArenaError> {
    require(
        checkpoint_reserve_ns > 0,
        "utility checkpoint reserve is required",
    )?;
    execution_deadline
        .checked_sub(Duration::from_nanos(checkpoint_reserve_ns))
        .ok_or_else(|| invalid("utility checkpoint reserve deadline underflow"))
}
fn encode(value: &impl Serialize) -> Result<Vec<u8>, ArenaError> {
    serde_json::to_vec(value).map_err(|error| invalid(error.to_string()))
}
fn encode_startup(family: UtilityRuntimeFamily, mut startup: Value) -> Result<Vec<u8>, ArenaError> {
    let fields = [
        "schema",
        "execution_id",
        "window_id",
        "registration_sha256",
        "registration",
        "control",
        "original_started_ns",
        "execution_until_ns",
        "whole_until_ns",
        "source_root",
        "package_root",
        "source_files",
        "dependency_directories",
        "runtime_module_pins",
        "python_runtime",
        "resources",
        "artifact_watch_profile",
        "child_read_credit",
    ];
    let object = startup
        .as_object_mut()
        .ok_or_else(|| invalid("utility startup object"))?;
    require(
        object.len() == fields.len() && fields.iter().all(|field| object.contains_key(*field)),
        "utility startup original fields",
    )?;
    require(
        object.get("schema").and_then(Value::as_str) == Some(family.domain(STARTUP)),
        "utility startup original family domain",
    )?;
    // The family comes from the validated original registration, never an
    // optional JSON flag. Legacy bytes keep their exact eighteen-field shape.
    if family == UtilityRuntimeFamily::InterventionV2 {
        object.insert("data_family".into(), json!("intervention_v2"));
    }
    encode(&startup)
}
fn startup_registration_sha256(
    registration: &RoventUtilityExecutionRegistration,
    raw_file_sha256: &str,
) -> Result<String, ArenaError> {
    // The legacy startup records its retained file pin. Intervention /2 instead
    // binds the embedded /3 declaration by its typed canonical digest; the raw
    // file pin remains independently retained by read_json and the broker.
    match registration.family() {
        UtilityRuntimeFamily::CohortV1 => Ok(raw_file_sha256.to_owned()),
        UtilityRuntimeFamily::InterventionV2 => registration.as_intervention()?.canonical_sha256(),
    }
}
fn decode<T: DeserializeOwned>(raw: &[u8]) -> Result<T, ArenaError> {
    crate::decode_json(std::str::from_utf8(raw).map_err(|_| invalid("strict UTF8 JSON"))?)
}
fn pin(value: &Value) -> Result<RoventTrainingFilePin, ArenaError> {
    let object = value
        .as_object()
        .ok_or_else(|| invalid("exact original file pin object"))?;
    require(
        object.len() == 3
            && ["path", "sha256", "bytes"]
                .iter()
                .all(|k| object.contains_key(*k)),
        "exact file pin fields",
    )?;
    let pin: RoventTrainingFilePin =
        serde_json::from_value(value.clone()).map_err(|e| invalid(e.to_string()))?;
    require(
        linux::path_ok(&pin.path) && generation_sha(&pin.sha256) && pin.bytes > 0,
        "absolute original nonempty pin",
    )?;
    Ok(pin)
}
fn own(
    pin: &RoventTrainingFilePin,
    max: u64,
    retain: bool,
    budget: &RoventRecoveryReadBudget,
) -> Result<(linux::HeldFile, Vec<u8>), ArenaError> {
    let (held, raw) = budget.read(pin, max, retain)?;
    Ok((
        linux::HeldFile {
            file: held.file,
            pin: held.pin,
            identity: held.identity,
        },
        raw,
    ))
}
fn kernel_text(
    path: &Path,
    max: u64,
    budget: &RoventRecoveryReadBudget,
) -> Result<String, ArenaError> {
    budget.reserve(
        max.checked_add(1)
            .ok_or_else(|| invalid("kernel credit overflow"))?,
    )?;
    let mut raw = Vec::new();
    File::open(path)
        .map_err(|_| invalid("kernel text open"))?
        .take(max + 1)
        .read_to_end(&mut raw)
        .map_err(|_| invalid("kernel text read"))?;
    require(raw.len() as u64 <= max, "kernel text capacity")?;
    String::from_utf8(raw).map_err(|_| invalid("kernel text UTF8"))
}
fn process_start(pid: u32, budget: &RoventRecoveryReadBudget) -> Result<u64, ArenaError> {
    kernel_text(Path::new(&format!("/proc/{pid}/stat")), 8192, budget)?
        .rsplit_once(") ")
        .and_then(|(_, v)| v.split_ascii_whitespace().nth(19))
        .and_then(|v| v.parse().ok())
        .ok_or_else(|| invalid("process start ticks"))
}

/// Distinct utility closure; sharing source bytes never shares a training lease.
pub fn compiled_rovent_utility_trainer_source_sha256() -> Result<String, ArenaError> {
    let map = utility_sources()?
        .iter()
        .map(|(p, b)| {
            (
                p.to_string(),
                json!({"bytes":b.len(),"sha256":format!("{:x}",Sha256::digest(b))}),
            )
        })
        .collect::<BTreeMap<_, _>>();
    generation_digest(SOURCE, &map).map_err(ArenaError::Manifest)
}
fn utility_sources() -> Result<BTreeMap<&'static str, &'static [u8]>, ArenaError> {
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
    for (path, raw) in [
        source!("rovent_utility_training_child.py"),
        source!("rovent_utility_training_bootstrap.py"),
        source!("rovent_utility_training_authority.py"),
        source!("rovent_utility_training_window.py"),
        source!("rovent_utility_training.py"),
        source!("rovent_utility_training_run.py"),
        source!("rovent_utility_tensor_training.py"),
        source!("nnuec_utility_trainable.py"),
        source!("rovent_utility_input.py"),
        source!("rovent_utility_route.py"),
        source!("rovent_utility_cohort_files.py"),
        source!("rovent_utility_cohort_payload.py"),
        source!("rovent_utility_root_local.py"),
        source!("rovent_utility_historical_checkpoint.py"),
        source!("rovent_utility_intervention.py"),
        source!("rovent_utility_intervention_files.py"),
        source!("rovent_utility_intervention_payload.py"),
        source!("rovent_utility_intervention_preparation.py"),
        source!("rovent_utility_intervention_numerical.py"),
        source!("rovent_utility_intervention_dataset.py"),
        source!("rovent_utility_intervention_execution.py"),
        source!("rovent_utility_intervention_authority.py"),
        source!("rovent_utility_intervention_training.py"),
        source!("rovent_utility_intervention_checkpoint.py"),
        source!("rovent_utility_intervention_child.py"),
        source!("rovent_utility_training_family.py"),
    ] {
        files.insert(path, raw);
    }
    require(
        files.len() <= 128
            && files.values().all(|raw| raw.len() <= 4 << 20)
            && files.values().map(|raw| raw.len()).sum::<usize>() <= 32 << 20,
        "utility source closure bounds",
    )?;
    Ok(files)
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
struct PendingBroker {
    handle: JoinHandle<Result<(), ArenaError>>,
    stop: Arc<AtomicBool>,
}
struct Journal {
    ledger: Option<UtilityLedger>,
    source_checked: bool,
    next_sequence: u64,
    events: Vec<Value>,
    raw: Vec<u8>,
    file: Option<linux::HeldFile>,
    ended: Option<Value>,
    end_receipt: Option<RoventTrainingFilePin>,
    export: Option<Value>,
    cost_high_water: Value,
}
impl Journal {
    fn new() -> Self {
        Self {
            ledger: None,
            source_checked: false,
            next_sequence: 1,
            events: Vec::new(),
            raw: Vec::new(),
            file: None,
            ended: None,
            end_receipt: None,
            export: None,
            cost_high_water: Value::Null,
        }
    }
    fn durable_event(
        &mut self,
        ctx: &Broker,
        held: &mut linux::LinuxHeld,
        kind: &str,
        request: &Value,
        receipt: Option<&RoventTrainingFilePin>,
        ack: &Value,
    ) -> Result<(), ArenaError> {
        let event = json!({"schema":ctx.window.family().domain("rz-rovent-utility-ledger-event/1"),"sequence":self.events.len()+1,"kind":kind,"request":request,"completion_receipt":receipt,"ack":ack});
        let mut raw = encode(&event)?;
        raw.push(b'\n');
        require(
            self.raw
                .len()
                .checked_add(raw.len())
                .is_some_and(|n| n <= 32 << 20),
            "bounded utility RPC journal",
        )?;
        if let Some(file) = self.file.as_mut() {
            held.control_written = held
                .control_written
                .checked_add(raw.len() as u64)
                .ok_or_else(|| invalid("journal output overflow"))?;
            require(
                held.control_written <= held.control_limit,
                "shared control output credit",
            )?;
            file.file
                .write_all(&raw)
                .and_then(|_| file.file.sync_all())
                .map_err(|_| invalid("durable utility event append"))?;
            self.raw.extend_from_slice(&raw);
            file.pin.bytes = self.raw.len() as u64;
            file.pin.sha256 = format!("{:x}", Sha256::digest(&self.raw));
            file.identity = FileIdentity::from(
                &file
                    .file
                    .metadata()
                    .map_err(|_| invalid("journal metadata"))?,
            );
        } else {
            self.file = Some(linux::write_new(
                &ctx.registration.control_root.join("utility-rpc.jsonl"),
                &raw,
                held,
                true,
            )?);
            self.raw = raw;
        }
        self.events.push(event);
        Ok(())
    }
    fn observe_cost(&mut self, payload: &Value) -> Result<(), ArenaError> {
        for key in ["cost", "snapshot_cost", "post_snapshot_cost"] {
            if let Some(cost) = payload.get(key) {
                let object = cost.as_object().ok_or_else(|| invalid("cost object"))?;
                for (name, value) in object {
                    let now = value
                        .as_u64()
                        .ok_or_else(|| invalid("finite unsigned recorded cost"))?;
                    if let Some(old) = self.cost_high_water.get(name) {
                        require(
                            now >= old.as_u64().ok_or_else(|| invalid("prior cost"))?,
                            "recorded post-snapshot/end expense decreased",
                        )?;
                    }
                }
                self.cost_high_water = cost.clone();
            }
        }
        Ok(())
    }
}
struct Broker {
    listener: UtilityListener,
    parent_pid: u32,
    parent_start: u64,
    interpreter: RoventTrainingFilePin,
    startup_sha256: String,
    registration: RoventUtilityExecutionRegistration,
    window: RoventUtilityTrainWindow,
    preparation: Value,
    export_request: Option<Value>,
    /// Derived once from the original hard deadline. This early admission
    /// boundary never renews the execution or cleanup clock.
    forward_admission_until: Instant,
    until: Instant,
    whole_until: Instant,
    cancel: Arc<AtomicBool>,
    stop: Arc<AtomicBool>,
    image_verified: Arc<AtomicBool>,
    held: Arc<Mutex<Option<linux::LinuxHeld>>>,
    journal: Arc<Mutex<Journal>>,
    budget: Arc<RoventRecoveryReadBudget>,
    resources: Arc<RoventTrainingResources>,
    dataset: VerifiedUtilityDataset,
    ancestors: Arc<VerifiedUtilityAncestors>,
    checkpoint: Option<VerifiedUtilityCheckpointSource>,
    child_identity: Arc<Mutex<Option<RoventHistoricalChildIdentity>>>,
    fresh_identity: Option<InitialSourceIdentity>,
    registration_file: Arc<File>,
    registration_pin: RoventTrainingFilePin,
    archives: Arc<Mutex<Vec<PublishedUtilityCheckpointArchive>>>,
    // The publisher returns the original boxed partial-I/O owner. Preserve
    // that allocation when the retention registry grows instead of moving its
    // descriptor family out of the failure during quarantine.
    #[allow(clippy::vec_box)]
    archive_failures: Arc<Mutex<Vec<UtilityCheckpointArchiveFailure>>>,
}
#[derive(Clone)]
struct InitialSourceIdentity {
    declaration: Value,
    frozen_weights: String,
}

/// A borrow from the current authenticated checkpoint RPC, never deserializable.
pub(super) struct ActualUtilityCheckpointBoundary<'a> {
    ctx: &'a Broker,
    boundary: UtilityCheckpointBoundary<'a>,
    held: &'a linux::LinuxHeld,
    tensor_index: usize,
    journal_index: usize,
    projection_index: usize,
    original_request: &'a Value,
}
impl ActualUtilityCheckpointBoundary<'_> {
    pub(super) fn checkpoint_request(&self) -> &Value {
        self.original_request
    }
    pub(super) fn registration(&self) -> &RoventUtilityExecutionRegistration {
        &self.ctx.registration
    }
    pub(super) fn window(&self) -> &RoventUtilityTrainWindow {
        &self.ctx.window
    }
    pub(super) fn ledger_boundary(&self) -> &UtilityCheckpointBoundary<'_> {
        &self.boundary
    }
    pub(super) fn dataset(&self) -> &VerifiedUtilityDataset {
        &self.ctx.dataset
    }
    pub(super) fn ancestors(&self) -> &Arc<VerifiedUtilityAncestors> {
        &self.ctx.ancestors
    }
    pub(super) fn checkpoint_source(&self) -> Option<&VerifiedUtilityCheckpointSource> {
        self.ctx.checkpoint.as_ref()
    }
    pub(super) fn registration_file(&self) -> (&RoventTrainingFilePin, &File) {
        (&self.ctx.registration_pin, &self.ctx.registration_file)
    }
    pub(super) fn tensor_file(&self) -> (&RoventTrainingFilePin, &File) {
        let held = &self.held.checkpoints[self.tensor_index];
        (&held.pin, &held.file)
    }
    pub(super) fn journal_file(&self) -> (&RoventTrainingFilePin, &File) {
        let held = &self.held.journals[self.journal_index];
        (&held.pin, &held.file)
    }
    pub(super) fn tensor_projection_file(&self) -> (&RoventTrainingFilePin, &File) {
        let held = &self.held.journals[self.projection_index];
        (&held.pin, &held.file)
    }
    pub(super) fn read_budget(&self) -> &Arc<RoventRecoveryReadBudget> {
        &self.ctx.budget
    }
    pub(super) fn child_identity(&self) -> Result<RoventHistoricalChildIdentity, ArenaError> {
        self.ctx
            .child_identity
            .lock()
            .map_err(|_| invalid("child identity poisoned"))?
            .clone()
            .ok_or_else(|| invalid("authenticated child absent"))
    }
    pub(super) fn held_files(&self) -> impl Iterator<Item = (&RoventTrainingFilePin, &File)> {
        self.held
            .inputs
            .iter()
            .chain(&self.held.sources)
            .chain(&self.held.checkpoints)
            .chain(&self.held.journals)
            .chain(&self.held.artifacts)
            .map(|h| (&h.pin, &h.file))
    }
    pub(super) fn archive_target(&self) -> Result<RoventCheckpointArchiveTarget, ArenaError> {
        let root = &self.ctx.registration.checkpoint_directory;
        let fd = directory_file(&open_directory(root)?)?;
        let mut reserved = self
            .held
            .sources
            .iter()
            .chain(&self.held.checkpoints)
            .chain(&self.held.artifacts)
            .try_fold(self.held.control_limit, |a, h| {
                a.checked_add(h.pin.bytes)
                    .ok_or_else(|| invalid("archive output capacity"))
            })?;
        for archive in self
            .ctx
            .archives
            .lock()
            .map_err(|_| invalid("archive quota custody poisoned"))?
            .iter()
        {
            reserved = reserved
                .checked_add(archive.reserved_output_bytes())
                .ok_or_else(|| invalid("prior archive reservation overflow"))?;
        }
        let remaining = self
            .ctx
            .registration
            .max_output_bytes
            .checked_sub(reserved)
            .ok_or_else(|| invalid("archive remaining output credit"))?;
        Ok(RoventCheckpointArchiveTarget {
            parent_directory: fd,
            parent_path: root.clone(),
            leaf_name: format!("utility-archive-{:06}", self.tensor_index + 1),
            maximum_output_bytes: remaining,
            deadline: self.ctx.until,
            cancel: Arc::clone(&self.ctx.cancel),
            read_budget: Arc::clone(&self.ctx.budget),
        })
    }
}

/// Current normal-close borrow, constructed only after the actual supervisor
/// fence, original terminal FD and empty resource-leaf observation.
pub(super) struct ActualUtilityClosedCheckpoint<'a> {
    latest: &'a PublishedUtilityCheckpointArchive,
    end: &'a Value,
    process: &'a OriginalProcessOutput,
    terminal: &'a linux::HeldFile,
    child: &'a RoventHistoricalChildIdentity,
    registration: &'a RoventUtilityExecutionRegistration,
    window: &'a RoventUtilityTrainWindow,
    preparation: &'a Value,
    dataset: &'a VerifiedUtilityDataset,
    ancestors: &'a Arc<VerifiedUtilityAncestors>,
    ledger: &'a UtilityLedger,
    held: &'a linux::LinuxHeld,
    budget: &'a Arc<RoventRecoveryReadBudget>,
    deadline: Instant,
    cancel: &'a Arc<AtomicBool>,
    prior_archive_reserved: u64,
}
impl ActualUtilityClosedCheckpoint<'_> {
    pub(super) fn latest_archive(&self) -> &PublishedUtilityCheckpointArchive {
        self.latest
    }
    pub(super) fn original_end(&self) -> &Value {
        self.end
    }
    pub(super) fn process(&self) -> &OriginalProcessOutput {
        self.process
    }
    pub(super) fn terminal_file(&self) -> (&RoventTrainingFilePin, &File) {
        (&self.terminal.pin, &self.terminal.file)
    }
    pub(super) fn child_identity(&self) -> Result<RoventHistoricalChildIdentity, ArenaError> {
        Ok(self.child.clone())
    }
    pub(super) fn registration(&self) -> &RoventUtilityExecutionRegistration {
        self.registration
    }
    pub(super) fn window(&self) -> &RoventUtilityTrainWindow {
        self.window
    }
    pub(super) fn preparation(&self) -> &Value {
        self.preparation
    }
    pub(super) fn dataset(&self) -> &VerifiedUtilityDataset {
        self.dataset
    }
    pub(super) fn ancestors(&self) -> &Arc<VerifiedUtilityAncestors> {
        self.ancestors
    }
    pub(super) fn read_budget(&self) -> &Arc<RoventRecoveryReadBudget> {
        self.budget
    }
    pub(super) fn ledger(&self) -> &UtilityLedger {
        self.ledger
    }
    pub(super) fn archive_target(&self) -> Result<RoventCheckpointArchiveTarget, ArenaError> {
        let reserved = self
            .held
            .sources
            .iter()
            .chain(&self.held.checkpoints)
            .chain(&self.held.artifacts)
            .try_fold(
                self.held
                    .control_limit
                    .checked_add(self.prior_archive_reserved)
                    .ok_or_else(|| invalid("closed archive reservation overflow"))?,
                |sum, file| {
                    sum.checked_add(file.pin.bytes)
                        .ok_or_else(|| invalid("closed archive output overflow"))
                },
            )?;
        Ok(RoventCheckpointArchiveTarget {
            parent_directory: directory_file(&open_directory(
                &self.registration.checkpoint_directory,
            )?)?,
            parent_path: self.registration.checkpoint_directory.clone(),
            leaf_name: format!("utility-closed-{}", self.window.window_id),
            maximum_output_bytes: self
                .registration
                .max_output_bytes
                .checked_sub(reserved)
                .ok_or_else(|| invalid("closed archive remaining output"))?,
            deadline: self.deadline,
            cancel: Arc::clone(self.cancel),
            read_budget: Arc::clone(self.budget),
        })
    }
}

fn admit_peer(stream: &UnixStream, ctx: &Broker) -> Result<(u32, u64), ArenaError> {
    let peer = unix_peer_credentials(stream, "utility SO_PEERCRED", "utility peer PID")?;
    require(
        peer.uid == nix::unistd::geteuid().as_raw()
            && ctx.image_verified.load(Ordering::Acquire)
            && process_start(ctx.parent_pid, &ctx.budget)? == ctx.parent_start,
        "original parent/image/UID",
    )?;
    let status = kernel_text(
        Path::new(&format!("/proc/{}/status", peer.pid)),
        16 << 10,
        &ctx.budget,
    )?;
    require(
        is_direct_parent_group_leader(peer.pid, ctx.parent_pid, &status),
        "utility peer direct parent/group leader",
    )?;
    let image =
        File::open(format!("/proc/{}/exe", peer.pid)).map_err(|_| invalid("peer executable"))?;
    ctx.budget.verify_held(&image, &ctx.interpreter, false)?;
    let membership = ctx.resources.verify_child_membership(peer.pid)?;
    let r = ctx.resources.startup_json()?;
    let child = RoventHistoricalChildIdentity {
        boot_id: kernel_text(
            Path::new("/proc/sys/kernel/random/boot_id"),
            4096,
            &ctx.budget,
        )?
        .trim()
        .into(),
        pid: peer.pid,
        start_ticks: membership.process_start_ticks,
        cgroup_path: r["path"]
            .as_str()
            .ok_or_else(|| invalid("resource path"))?
            .into(),
        cgroup_device: r["device"]
            .as_u64()
            .ok_or_else(|| invalid("resource device"))?,
        cgroup_inode: r["inode"]
            .as_u64()
            .ok_or_else(|| invalid("resource inode"))?,
    };
    *ctx.child_identity
        .lock()
        .map_err(|_| invalid("child identity custody"))? = Some(child);
    Ok((peer.pid, membership.process_start_ticks))
}
fn measurement(ctx: &Broker, payload: &Value) -> Result<InitialUtilityMeasurement, ArenaError> {
    let frozen_mapping = payload["frozen_base_mapping_sha256"]
        .as_str()
        .filter(|sha| generation_sha(sha))
        .ok_or_else(|| invalid("current child observed canonical frozen mapping"))?
        .to_string();
    let (declaration, frozen_weights) = if let Some(source) = &ctx.fresh_identity {
        require(
            payload["initial_source_declaration"] == source.declaration
                && payload["frozen_base_weights_sha256"] == source.frozen_weights,
            "actual begin changed original held scalar source",
        )?;
        (source.declaration.clone(), source.frozen_weights.clone())
    } else {
        let source = ctx
            .checkpoint
            .as_ref()
            .ok_or_else(|| invalid("original fork source absent"))?
            .restored()?;
        require(
            payload["initial_source_declaration"] == source.initial_source_declaration
                && payload["frozen_base_weights_sha256"] == source.frozen_base_weights_sha256
                && frozen_mapping == source.frozen_base_mapping_sha256,
            "fork changed original training declaration or frozen mapping",
        )?;
        (
            source.initial_source_declaration,
            source.frozen_base_weights_sha256,
        )
    };
    Ok(InitialUtilityMeasurement {
        states: UtilityStateSeals::from_payload(payload)?,
        sampler: UtilityLedger::initial_sampler_projection(&ctx.window, &ctx.dataset)?,
        initial_source_declaration: declaration,
        frozen_base_weights_sha256: frozen_weights,
        frozen_base_mapping_sha256: frozen_mapping,
    })
}
fn dispatch(
    ctx: &Broker,
    j: &mut Journal,
    operation: &str,
    payload: &Value,
) -> Result<Value, ArenaError> {
    require(j.ended.is_none(), "no RPC after utility end")?;
    require(
        ctx.registration.action == RoventUtilityExecutionAction::Train
            || matches!(
                operation,
                "validate_checkpoint_state"
                    | "begin_window"
                    | "validate_learned_export"
                    | "end_window"
            ),
        "export cannot train or publish checkpoint",
    )?;
    match operation {
        "validate_checkpoint_state" => {
            require(
                j.ledger.is_none() && !j.source_checked,
                "source scratch check occurs once before begin",
            )?;
            let source = ctx
                .checkpoint
                .as_ref()
                .ok_or_else(|| invalid("original utility checkpoint missing"))?;
            let original = UtilityLedger::new(
                source.window().clone(),
                source.preparation().clone(),
                source.dataset(),
                source.restored()?,
            )?;
            let ack = original.validate_checkpoint_state(payload)?;
            j.source_checked = true;
            Ok(ack)
        }
        "begin_window" => {
            require(j.ledger.is_none(), "one actual utility window")?;
            let mut ledger = match &ctx.registration.initialization_source {
                RoventUtilityInitializationSource::Initialize { .. } => {
                    UtilityLedger::from_initial_measurement(
                        ctx.window.clone(),
                        ctx.preparation.clone(),
                        ctx.dataset.clone(),
                        measurement(ctx, payload)?,
                    )?
                }
                RoventUtilityInitializationSource::Resume { .. } => {
                    require(j.source_checked, "original scratch tensor check required")?;
                    UtilityLedger::new(
                        ctx.window.clone(),
                        ctx.preparation.clone(),
                        ctx.dataset.clone(),
                        ctx.checkpoint
                            .as_ref()
                            .ok_or_else(|| invalid("resume source"))?
                            .restored()?,
                    )?
                }
                RoventUtilityInitializationSource::Fork { .. } => {
                    require(
                        j.source_checked,
                        "original fork scratch tensor check required",
                    )?;
                    UtilityLedger::from_fork_measurement(
                        ctx.window.clone(),
                        ctx.preparation.clone(),
                        ctx.dataset.clone(),
                        &ctx.checkpoint
                            .as_ref()
                            .ok_or_else(|| invalid("fork source"))?
                            .restored()?,
                        measurement(ctx, payload)?,
                    )?
                }
            };
            let ack = ledger.begin_window(payload)?;
            let mut owner = ctx.held.lock().map_err(|_| invalid("custody poisoned"))?;
            let held = owner.as_mut().ok_or_else(|| invalid("custody absent"))?;
            let initial = linux::write_new(
                &ctx.registration
                    .control_root
                    .join("utility-initial-ledger.json"),
                &encode(&ledger.initial_ledger_projection())?,
                held,
                true,
            )?;
            held.journals.push(initial);
            j.durable_event(ctx, held, "begin_window", payload, None, &ack)?;
            j.ledger = Some(ledger);
            Ok(ack)
        }
        "consume_pair_context" => {
            require(
                Instant::now() < ctx.forward_admission_until,
                "utility forward admission closed for the reserved checkpoint window",
            )?;
            let ack = j
                .ledger
                .as_mut()
                .ok_or_else(|| invalid("utility begin required"))?
                .consume_pair_context(payload)?;
            let mut owner = ctx.held.lock().map_err(|_| invalid("custody poisoned"))?;
            j.durable_event(
                ctx,
                owner.as_mut().ok_or_else(|| invalid("custody absent"))?,
                "consume_pair_context",
                payload,
                None,
                &ack,
            )?;
            Ok(ack)
        }
        "completed_update" => {
            j.observe_cost(payload)?;
            let ledger = j
                .ledger
                .as_mut()
                .ok_or_else(|| invalid("utility begin required"))?;
            ledger.prepare_update(payload)?;
            let raw = ledger
                .pending_receipt_bytes()
                .ok_or_else(|| invalid("prepared actual update absent"))?
                .to_vec();
            let mut owner = ctx.held.lock().map_err(|_| invalid("custody poisoned"))?;
            let held = owner.as_mut().ok_or_else(|| invalid("custody absent"))?;
            let receipt = linux::write_new(
                &ctx.registration
                    .control_root
                    .join(format!("utility-update-{:08}.json", j.next_sequence)),
                &raw,
                held,
                true,
            )?;
            let receipt_pin = receipt.pin.clone();
            held.journals.push(receipt);
            let original_receipt = held
                .journals
                .last()
                .ok_or_else(|| invalid("original receipt custody absent"))?;
            ctx.budget
                .verify_held(&original_receipt.file, &original_receipt.pin, false)?;
            let ack = ledger.commit_update(receipt_pin.clone())?;
            j.durable_event(
                ctx,
                held,
                "committed_update",
                payload,
                Some(&receipt_pin),
                &ack,
            )?;
            Ok(ack)
        }
        "checkpoint_published" => {
            j.observe_cost(payload)?;
            let checkpoint = pin(&payload["checkpoint"])?;
            require(
                checkpoint
                    .path
                    .starts_with(&ctx.registration.checkpoint_directory)
                    && checkpoint.path.extension().is_some_and(|s| s == "pt"),
                "checkpoint escaped new output scope",
            )?;
            let mut owner = ctx.held.lock().map_err(|_| invalid("custody poisoned"))?;
            let held = owner.as_mut().ok_or_else(|| invalid("custody absent"))?;
            let (file, _) = own(&checkpoint, MAX_TENSOR, false, &ctx.budget)?;
            let tensor_index = held.checkpoints.len();
            held.checkpoints.push(file);
            // The current child measures its own tensor mapping; no historical
            // JSON or parent-only hash is treated as an independent Torch load.
            let projection = pin(&payload["tensor_projection"])?;
            require(
                projection
                    .path
                    .starts_with(&ctx.registration.checkpoint_directory)
                    && projection.path != checkpoint.path
                    && projection.path.extension().is_some_and(|s| s == "json"),
                "separate tensor projection escaped original checkpoint scope",
            )?;
            let (tensor_projection, raw) = own(&projection, MAX_JSON, true, &ctx.budget)?;
            let _: Value = decode(&raw)?;
            let projection_index = held.journals.len();
            held.journals.push(tensor_projection);
            let prefix = linux::write_new(
                &ctx.registration
                    .control_root
                    .join(format!("utility-rpc-prefix-{tensor_index:06}.jsonl")),
                &j.raw,
                held,
                true,
            )?;
            let journal_index = held.journals.len();
            held.journals.push(prefix);
            let ledger = j
                .ledger
                .as_ref()
                .ok_or_else(|| invalid("utility begin required"))?;
            let boundary = ledger.prepare_checkpoint_boundary(payload, &checkpoint, &projection)?;
            let source = ActualUtilityCheckpointBoundary {
                ctx,
                boundary,
                held,
                tensor_index,
                journal_index,
                projection_index,
                original_request: payload,
            };
            match publish_current_utility_checkpoint(&source) {
                Ok(archive) => ctx
                    .archives
                    .lock()
                    .map_err(|_| invalid("archive custody poisoned"))?
                    .push(archive),
                Err(failure) => {
                    let failure_observation = failure.observation();
                    ctx.archive_failures
                        .lock()
                        .map_err(|_| invalid("archive failure custody poisoned"))?
                        .push(failure);
                    return Err(invalid(format!(
                        "utility checkpoint publication failed; original bytes retained: {failure_observation}"
                    )));
                }
            }
            j.durable_event(ctx, held, "checkpoint_published", payload, None, &json!({}))?;
            Ok(json!({}))
        }
        "validate_learned_export" => {
            require(
                ctx.registration.action == RoventUtilityExecutionAction::Export
                    && ctx
                        .export_request
                        .as_ref()
                        .is_some_and(|request| request["kind"] == payload["kind"]),
                "export kind/action differs from original pinned request",
            )?;
            require(j.export.is_none(), "one original utility export")?;
            let ack = j
                .ledger
                .as_ref()
                .ok_or_else(|| invalid("utility begin required"))?
                .learned_export_projection(payload)?;
            j.export = Some(ack.clone());
            Ok(ack)
        }
        "end_window" => {
            j.observe_cost(payload)?;
            let ack = j
                .ledger
                .as_ref()
                .ok_or_else(|| invalid("utility begin required"))?
                .ended_projection(payload)?;
            require(
                ctx.registration.action != RoventUtilityExecutionAction::Export
                    || j.export.is_some(),
                "learned export ACK absent",
            )?;
            j.ended = Some(ack);
            let raw = encode(
                &json!({"schema":ctx.window.family().domain("rz-rovent-utility-parent-end/1"),"execution_id":ctx.registration.execution_id,"window_id":ctx.window.window_id,"original_end":payload,"parent_projection":j.ended,"cost_high_water":j.cost_high_water,"cost_scope":"original_peer_report_validated_by_current_ledger_not_independent_OS_cost","execution_authorized":false}),
            )?;
            let mut slot = ctx
                .held
                .lock()
                .map_err(|_| invalid("end receipt custody poisoned"))?;
            let held = slot
                .as_mut()
                .ok_or_else(|| invalid("end receipt custody missing"))?;
            let file = linux::write_new(
                &ctx.registration
                    .control_root
                    .join("utility-parent-end.json"),
                &raw,
                held,
                true,
            )?;
            j.end_receipt = Some(file.pin.clone());
            held.journals.push(file);
            let file = held
                .journals
                .last()
                .ok_or_else(|| invalid("original end receipt missing"))?;
            ctx.budget.verify_held(&file.file, &file.pin, false)?;
            Ok(json!({}))
        }
        _ => Err(invalid("unknown utility RPC operation")),
    }
}
fn serve(ctx: Broker) -> Result<(), ArenaError> {
    ctx.listener
        .set_nonblocking(true)
        .map_err(|_| invalid("utility nonblocking socket"))?;
    let (mut stream, _) = loop {
        if ctx.stop.load(Ordering::Acquire) {
            return Ok(());
        }
        require(
            Instant::now() < ctx.until && !ctx.cancel.load(Ordering::Acquire),
            "utility admission deadline/cancel",
        )?;
        match ctx.listener.accept() {
            Ok(v) => break v,
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                thread::park_timeout(Duration::from_millis(2))
            }
            Err(_) => return Err(invalid("utility accept")),
        }
    };
    let (pid, start) = admit_peer(&stream, &ctx)?;
    stream
        .set_read_timeout(Some(Duration::from_millis(25)))
        .and_then(|_| stream.set_write_timeout(Some(Duration::from_millis(25))))
        .map_err(|_| invalid("utility finite socket timeout"))?;
    let mut total = 0;
    loop {
        let raw = read_unix_frame(
            &mut stream,
            UnixFrameReadScope {
                stop: &ctx.stop,
                cancel: &ctx.cancel,
                whole_until: ctx.whole_until,
                max_total_bytes: ctx.registration.max_rpc_bytes,
                max_frame_bytes: MAX_FRAME,
                errors: UnixFrameReadErrors {
                    partial_stop: "utility partial stop",
                    deadline_cancel: "utility deadline/cancel",
                    partial_eof: "utility partial EOF",
                    byte_overflow: "utility byte overflow",
                    byte_frame_bound: "utility RPC bound",
                    pipelined: "utility pipelined RPC",
                    read: "utility RPC read",
                },
            },
            &mut total,
        )?;
        let Some(raw) = raw else { break };
        let r: Request = decode(&raw)?;
        require(
            r.schema == ctx.window.family().domain(RPC)
                && r.window_id == ctx.window.window_id
                && r.startup_sha256 == ctx.startup_sha256
                && r.sequence > 0
                && r.sequence <= ctx.registration.max_rpc_messages
                && process_start(pid, &ctx.budget)? == start
                && (Instant::now() < ctx.until || r.operation == "end_window"),
            "utility original peer/window/deadline",
        )?;
        let m = ctx.resources.verify_child_membership(pid)?;
        require(
            m.process_start_ticks == start,
            "utility peer cgroup/start changed",
        )?;
        let mut journal = ctx
            .journal
            .lock()
            .map_err(|_| invalid("utility ledger poisoned"))?;
        require(
            r.sequence == journal.next_sequence,
            "utility skipped/duplicate sequence",
        )?;
        let payload = dispatch(&ctx, &mut journal, &r.operation, &r.payload)?;
        let acknowledgement_deadline = if r.operation == "end_window" {
            ctx.whole_until
        } else {
            ctx.until
        };
        require(
            Instant::now() < acknowledgement_deadline,
            "utility operation/readback missed its original deadline; original custody retained",
        )?;
        journal.next_sequence = journal
            .next_sequence
            .checked_add(1)
            .ok_or_else(|| invalid("utility sequence overflow"))?;
        let ack = prepare_unix_ack(
            encode(
                &json!({"schema":ctx.window.family().domain(ACK),"window_id":ctx.window.window_id,"startup_sha256":ctx.startup_sha256,"sequence":r.sequence,"operation":r.operation,"status":"ok","payload":payload}),
            )?,
            MAX_FRAME,
            "utility ACK frame",
        )?;
        charge_unix_ack(
            &mut total,
            &ack,
            ctx.registration.max_rpc_bytes,
            "utility ACK credit overflow",
            "utility total request/ACK byte lease exhausted",
        )?;
        let remaining = acknowledgement_deadline.saturating_duration_since(Instant::now());
        require(
            !remaining.is_zero(),
            "utility original ACK deadline expired",
        )?;
        stream
            .set_write_timeout(Some(remaining.min(Duration::from_millis(25))))
            .map_err(|_| invalid("utility original finite ACK write timeout"))?;
        write_unix_ack(
            &mut stream,
            &ack,
            "utility lost ACK; work and bytes remain charged",
        )?;
        require(
            Instant::now() < acknowledgement_deadline,
            "utility ACK completion missed original deadline; charged custody retained",
        )?;
        if journal.ended.is_some() {
            break;
        }
    }
    Ok(())
}

struct Custody {
    held: Arc<Mutex<Option<linux::LinuxHeld>>>,
    journal: Arc<Mutex<Journal>>,
    process: Option<OriginalProcessOutput>,
    broker: Option<PendingBroker>,
    endpoint: Option<OwnedUtilityEndpoint>,
    // Secondary endpoint I/O failure stays with the original sticky owner.
    // It cannot replace the primary error or alter the receipt wire schema.
    _endpoint_cleanup_error: Option<String>,
    resources: Option<Arc<RoventTrainingResources>>,
    dataset: Option<VerifiedUtilityDataset>,
    ancestors: Option<Arc<VerifiedUtilityAncestors>>,
    datasets: Vec<VerifiedUtilityDataset>,
    checkpoint: Option<VerifiedUtilityCheckpointSource>,
    registration: Option<(RoventTrainingFilePin, Arc<File>)>,
    archives: Arc<Mutex<Vec<PublishedUtilityCheckpointArchive>>>,
    // Shared with Broker: retain each publisher-returned partial-I/O allocation
    // intact through failure custody, including registry growth and shutdown.
    #[allow(clippy::vec_box)]
    archive_failures: Arc<Mutex<Vec<UtilityCheckpointArchiveFailure>>>,
    admission_files: Vec<linux::HeldFile>,
    old_run_reclamation: Option<VerifiedUtilityOldRunReclamation>,
    dataset_failures: Vec<RuntimeUtilityDatasetFailure>,
    source_failures: Vec<RuntimeUtilitySourceFailure>,
    family: Option<UtilityRuntimeFamily>,
}
impl Custody {
    fn new() -> Self {
        Self {
            held: Arc::new(Mutex::new(None)),
            journal: Arc::new(Mutex::new(Journal::new())),
            process: None,
            broker: None,
            endpoint: None,
            _endpoint_cleanup_error: None,
            resources: None,
            dataset: None,
            ancestors: None,
            datasets: Vec::new(),
            checkpoint: None,
            registration: None,
            archives: Arc::new(Mutex::new(Vec::new())),
            archive_failures: Arc::new(Mutex::new(Vec::new())),
            admission_files: Vec::new(),
            old_run_reclamation: None,
            dataset_failures: Vec::new(),
            source_failures: Vec::new(),
            family: None,
        }
    }
}
// Sticky custody survives an outcome drop. It is not an automatic retry,
// process reaper, device unlock, old-run reclamation or permission to execute.
static FAILED_CUSTODY: LazyLock<Mutex<Option<Box<Custody>>>> = LazyLock::new(|| Mutex::new(None));
pub struct RoventUtilityTrainingOutcome {
    success: bool,
    observation: Value,
    _owner: Option<Box<Custody>>,
}
impl RoventUtilityTrainingOutcome {
    pub fn success(&self) -> bool {
        self.success
    }
    pub fn observation(&self) -> Value {
        self.observation.clone()
    }
}

fn read_json<T: DeserializeOwned>(
    pin: &RoventTrainingFilePin,
    custody: &mut Custody,
    budget: &RoventRecoveryReadBudget,
) -> Result<T, ArenaError> {
    let (held, raw) = own(pin, MAX_JSON, true, budget)?;
    custody.admission_files.push(held);
    decode(&raw)
}
fn initial_source_identity(
    base_raw: &[u8],
    utility_raw: &[u8],
    preparation: &Value,
    family: UtilityRuntimeFamily,
) -> Result<InitialSourceIdentity, ArenaError> {
    use rz_search::cpu_value::nnuec::v2::{Checkpoint, Model, utility};
    let numerical = family.numerical(preparation)?;
    let base: Checkpoint = decode(base_raw)?;
    base.validate().map_err(|e| invalid(e.to_string()))?;
    let frozen_weights = base.weights_sha256().map_err(|e| invalid(e.to_string()))?;
    require(
        numerical.frozen_base_weights_sha256() == frozen_weights,
        "original frozen semantic weights changed",
    )?;
    require(
        serde_json::to_value(&base.base.config).map_err(|e| invalid(e.to_string()))?
            == serde_json::to_value(numerical.frozen_base_config())
                .map_err(|e| invalid(e.to_string()))?,
        "original frozen base configuration changed",
    )?;
    let model = Model::new(base).map_err(|e| invalid(e.to_string()))?;
    let initial: utility::Checkpoint = decode(utility_raw)?;
    let identity = initial
        .weights_sha256(&model)
        .map_err(|e| invalid(e.to_string()))?;
    require(
        numerical.initial_utility_weights_sha256() == identity,
        "original utility semantic parameters differ",
    )?;
    let declaration =
        serde_json::to_value(&initial.training).map_err(|e| invalid(e.to_string()))?;
    Ok(InitialSourceIdentity {
        declaration,
        frozen_weights,
    })
}

fn install_sources(
    held: &mut linux::LinuxHeld,
    registration: &RoventUtilityExecutionRegistration,
) -> Result<Value, ArenaError> {
    require(
        registration.trainer_source_sha256 == compiled_rovent_utility_trainer_source_sha256()?,
        "registered utility source closure differs from compiled source",
    )?;
    let mut inventory = Vec::new();
    for (relative, raw) in utility_sources()? {
        let path = held.source.join(relative);
        linux::ensure_child_directory(
            &registration.output_root,
            path.parent().ok_or_else(|| invalid("source parent"))?,
        )?;
        let file = linux::write_new(&path, raw, held, false)?;
        inventory.push(
            json!({"relative_path":relative,"sha256":file.pin.sha256,"bytes":file.pin.bytes}),
        );
        held.sources.push(file);
    }
    Ok(json!(inventory))
}
fn verify_retained(
    held: &linux::LinuxHeld,
    budget: &RoventRecoveryReadBudget,
) -> Result<(), ArenaError> {
    for file in held
        .inputs
        .iter()
        .chain(&held.sources)
        .chain(&held.checkpoints)
        .chain(&held.journals)
        .chain(&held.artifacts)
    {
        require(
            FileIdentity::from(
                &file
                    .file
                    .metadata()
                    .map_err(|_| invalid("retained metadata"))?,
            ) == file.identity,
            "retained original descriptor changed",
        )?;
        budget.verify_held(&file.file, &file.pin, false)?;
        let (named, _) = budget.read(&file.pin, MAX_TENSOR, false)?;
        require(
            named.identity == file.identity,
            "original file name replaced",
        )?;
    }
    for (fd, path, device, inode) in &held.dependency_directories {
        let actual = fd.metadata().map_err(|_| invalid("dependency FD"))?;
        let named = directory_file(&open_directory(path)?)?
            .metadata()
            .map_err(|_| invalid("dependency path"))?;
        require(
            actual.dev() == *device
                && actual.ino() == *inode
                && named.dev() == *device
                && named.ino() == *inode,
            "dependency directory replaced",
        )?;
    }
    Ok(())
}
fn preserve_transport(
    process: &OriginalProcessOutput,
    held: &mut linux::LinuxHeld,
) -> Result<(), ArenaError> {
    for (name, raw) in [
        (
            "utility-child-stdout.txt",
            process.process().stdout.as_slice(),
        ),
        (
            "utility-child-stderr.txt",
            process.process().stderr.as_slice(),
        ),
    ] {
        let target = held.control_path.join(name);
        if !raw.is_empty() && !held.journals.iter().any(|file| file.pin.path == target) {
            let file = linux::write_new(&target, raw, held, true)?;
            held.journals.push(file);
        }
    }
    Ok(())
}
fn verify_import_report(
    report: &Value,
    registration: &RoventUtilityExecutionRegistration,
) -> Result<(), ArenaError> {
    // These are current child readbacks joined to actual pinned files. They
    // do not observe transitive mappings, all OS I/O or an independent forward.
    let runtime = &report["imported_runtime"];
    let observed = runtime["endpoints"]
        .as_array()
        .ok_or_else(|| invalid("imported runtime module readback missing"))?;
    require(
        observed.len() == 2
            && runtime["schema"] == "rz-rovent-imported-runtime-readback/1"
            && runtime["torch_extension_mapping"] == "confirmed_device_inode",
        "actual imported Torch Python/extension readback differs",
    )?;
    let mut names = BTreeSet::new();
    for actual in observed {
        let name = actual["module"]
            .as_str()
            .filter(|name| matches!(*name, "torch" | "torch._C"))
            .ok_or_else(|| invalid("runtime module name"))?;
        require(names.insert(name), "duplicate actual runtime endpoint")?;
        let expected = registration
            .runtime_endpoints
            .iter()
            .find(|p| p.name == format!("{name}.__file__"))
            .ok_or_else(|| invalid("unexpected imported endpoint"))?;
        require(
            actual["path"] == json!(expected.artifact.path)
                && actual["sha256"] == expected.artifact.sha256
                && actual["bytes"] == expected.artifact.bytes
                && actual["device"].as_u64().is_some()
                && actual["inode"].as_u64().is_some(),
            "imported runtime endpoint raw file changed",
        )?;
    }
    Ok(())
}

/// Concrete entry from the thin binary. Inspect remains read-only. Execute
/// creates one fresh owned tree and one original finite supervised process.
pub fn run_rovent_utility_from_cli(
    request: RoventUtilityCliAdmissionRequest,
    cancel: Arc<AtomicBool>,
) -> Result<RoventUtilityTrainingOutcome, ArenaError> {
    let mut admission = OriginalWindowAdmission::acquire()?;
    request.ensure_admission_open()?;
    let command = request.command().clone();
    let budget = Arc::new(RoventRecoveryReadBudget::new(
        command.scope.max_read_bytes,
        request.whole_deadline(),
        Arc::clone(&cancel),
    )?);
    let mut custody = Custody::new();
    let result = (|| -> Result<Value, ArenaError> {
        let registration: RoventUtilityExecutionRegistration =
            read_json(&command.registration, &mut custody, &budget)?;
        registration.validate()?;
        custody.family = Some(registration.family());
        require(
            command.scope.max_read_bytes == registration.max_read_bytes
                && command.scope.max_output_bytes == registration.max_output_bytes,
            "CLI and original registration limits must agree exactly",
        )?;
        let preparation: Value = read_json(&registration.preparation, &mut custody, &budget)?;
        let window: RoventUtilityTrainWindow =
            read_json(&registration.window_spec, &mut custody, &budget)?;
        registration.join_cohort_window(&window, &preparation)?;
        require(
            window.max_wall_time_ns
                == command
                    .scope
                    .wall_ms
                    .checked_mul(1_000_000)
                    .ok_or_else(|| invalid("wall overflow"))?
                && window.cleanup_time_ns
                    == command
                        .scope
                        .cleanup_ms
                        .checked_mul(1_000_000)
                        .ok_or_else(|| invalid("cleanup overflow"))?,
            "CLI cannot renew or change original utility window",
        )?;
        require(
            window.cpu_threads <= registration.resources_policy.affinity.len() as u32
                && window.interop_threads <= registration.resources_policy.affinity.len() as u32,
            "utility thread plan exceeds admitted CPU affinity",
        )?;
        let dataset = match VerifiedUtilityDataset::open(&registration, &preparation, &budget) {
            Ok(dataset) => dataset,
            Err(failure) => {
                let error = invalid(format!(
                    "original utility dataset admission: {}",
                    failure.error()
                ));
                custody.dataset_failures.push(failure);
                return Err(error);
            }
        };
        custody.dataset = Some(dataset.clone());
        custody.datasets.push(dataset.clone());
        dataset.validate_execution_sources(&registration)?;
        require(
            dataset.dataset_sha256() == window.dataset_sha256,
            "window dataset semantic seal differs",
        )?;
        // Historical cohorts must use their own held original preparation.
        // The checkpoint reader opens these exact enrollments while replaying
        // its original chain; a current cost epoch is never applied to them.
        let ancestors = Arc::new(match registration.open_ancestors(&budget) {
            Ok(owners) => owners,
            Err(failure) => {
                let error = invalid(format!("original utility ancestors: {failure}"));
                custody.source_failures.push(failure);
                return Err(error);
            }
        });
        custody.ancestors = Some(Arc::clone(&ancestors));
        let references = std::iter::once(registration.current_dataset.clone())
            .chain(ancestors.utility_references())
            .chain(
                ancestors
                    .intervention_datasets()
                    .iter()
                    .map(|dataset| dataset.reference().clone()),
            )
            .collect::<Vec<_>>();
        if let RoventUtilityInitializationSource::Resume { checkpoint }
        | RoventUtilityInitializationSource::Fork { checkpoint } =
            &registration.initialization_source
        {
            let source = match registration.open_checkpoint(checkpoint, &references, &budget) {
                Ok(source) => source,
                Err(failure) => {
                    let error = invalid(format!("original utility checkpoint: {failure}"));
                    custody.source_failures.push(failure);
                    return Err(error);
                }
            };
            custody.checkpoint = Some(source.clone());
            if matches!(
                registration.initialization_source,
                RoventUtilityInitializationSource::Resume { .. }
            ) {
                require(
                    source.preparation() == &preparation
                        && source.window().run_id == window.run_id
                        && source.window().training_seed == window.training_seed
                        && source.dataset().dataset_sha256() == dataset.dataset_sha256(),
                    "resume changed original model/recipe/run/seed/cohort",
                )?;
            } else {
                let source_numerical = source.window().family().numerical(source.preparation())?;
                let current_numerical = window.family().numerical(&preparation)?;
                require(
                    source.window().family() == window.family()
                        && source_numerical.recipe_sha256() == current_numerical.recipe_sha256()
                        && source_numerical.frozen_base_weights_sha256()
                            == current_numerical.frozen_base_weights_sha256()
                        && source_numerical.frozen_base_config()
                            == current_numerical.frozen_base_config()
                        && source_numerical.parameter_spec_sha256()
                            == current_numerical.parameter_spec_sha256(),
                    "fork changed original family/frozen base/utility parameter or optimizer semantics",
                )?;
            }
            custody.datasets.extend(source.datasets());
        }
        let mut required_checkpoint_publications = custody
            .checkpoint
            .as_ref()
            .map(|source| source.required_publication_sha256s())
            .unwrap_or_default();
        required_checkpoint_publications.remove(&registration.current_dataset.publication.sha256);
        dataset.admit_ancestry(&ancestors, &required_checkpoint_publications)?;
        let (interpreter, _) = own(
            &registration.trainer_python_executable,
            MAX_TENSOR,
            false,
            &budget,
        )?;
        let interpreter_fd = interpreter
            .file
            .try_clone()
            .map_err(|_| invalid("interpreter FD clone"))?;
        custody.admission_files.push(interpreter);
        for endpoint in &registration.runtime_endpoints {
            let (file, _) = own(&endpoint.artifact, MAX_TENSOR, false, &budget)?;
            custody.admission_files.push(file);
        }
        let (base, base_raw) = own(
            &registration.frozen_nnuec_base,
            rz_search::cpu_value::nnuec::NNUEC_MAX_BYTES as u64,
            true,
            &budget,
        )?;
        custody.admission_files.push(base);
        for source in &registration.frozen_pc_sources {
            let (file, _) = own(&source.artifact, MAX_TENSOR, false, &budget)?;
            custody.admission_files.push(file);
        }
        let fresh_identity = if let RoventUtilityInitializationSource::Initialize { utility } =
            &registration.initialization_source
        {
            let (file, raw) = own(
                utility,
                rz_search::cpu_value::nnuec::v2::utility::MAX_BYTES as u64,
                true,
                &budget,
            )?;
            custody.admission_files.push(file);
            Some(initial_source_identity(
                &base_raw,
                &raw,
                &preparation,
                registration.family(),
            )?)
        } else {
            None
        };
        let export_request: Option<Value> = registration
            .export_request
            .as_ref()
            .map(|p| read_json(p, &mut custody, &budget))
            .transpose()?;
        if let Some(export) = &export_request {
            let object = export
                .as_object()
                .ok_or_else(|| invalid("strict original utility export request"))?;
            require(
                object.len() == 4
                    && ["schema", "kind", "path", "max_output_bytes"]
                        .iter()
                        .all(|key| object.contains_key(*key))
                    && export["schema"]
                        == registration
                            .family()
                            .domain("rz-rovent-utility-export-request/1")
                    && matches!(export["kind"].as_str(), Some("collection" | "deployment")),
                "utility export request domain/fields",
            )?;
            let path = PathBuf::from(
                export["path"]
                    .as_str()
                    .ok_or_else(|| invalid("export path"))?,
            );
            require(
                linux::path_ok(&path)
                    && path.parent() == Some(registration.output_root.as_path())
                    && path != registration.receipt_path
                    && path.extension().is_some_and(|suffix| suffix == "json")
                    && export["max_output_bytes"]
                        .as_u64()
                        .is_some_and(|limit| limit > 0 && limit <= registration.max_output_bytes)
                    && !path.exists(),
                "new bounded utility export target required",
            )?;
        }
        require(
            registration.trainer_source_sha256 == compiled_rovent_utility_trainer_source_sha256()?,
            "compiled utility source pin mismatch",
        )?;
        request.ensure_admission_open()?;
        if command.action == RoventUtilityCliAction::Inspect {
            return Ok(
                json!({"schema":registration.family().domain("rz-rovent-utility-trainer-inspection/1"),"status":"inspected","execution_id":registration.execution_id,"registration_sha256":command.registration.sha256,"preparation_sha256":window.preparation_sha256,"window_sha256":window.canonical_sha256()?,"dataset":dataset.observation(),"read_accounting":budget.observation(),"execution_authorized":false,"historical_evidence_only":true,"model_or_tensor_forward_executed":false}),
            );
        }
        if let Some(source) = &custody.checkpoint {
            custody.old_run_reclamation =
                Some(source.admit_old_run_reclamation(Arc::clone(&budget))?);
        }
        let delegated = directory_file(&open_directory(&command.scope.delegated_cgroup)?)?;
        let resources = match RoventTrainingResources::prepare(
            delegated,
            &command.scope.delegated_cgroup,
            &command.scope.cgroup_leaf,
            registration.resources_policy.clone(),
        ) {
            Ok(resources) => Arc::new(resources),
            Err(failure) => {
                let observation = failure.observation();
                // Failure owns its original partially created cgroup; do not
                // discard it as an alleged clean unexecuted metadata error.
                // Retain the producer-returned allocation intact rather than
                // unboxing and moving that partial resource owner on growth.
                #[allow(clippy::vec_box)]
                static RESOURCE_FAILURES: LazyLock<
                    Mutex<Vec<Box<crate::rovent_training_process::RoventTrainingResourceFailure>>>,
                > = LazyLock::new(|| Mutex::new(Vec::new()));
                if let Ok(mut owners) = RESOURCE_FAILURES.lock() {
                    owners.push(failure);
                } else {
                    std::mem::forget(failure)
                }
                return Err(invalid(format!(
                    "resource preparation failed: {observation}"
                )));
            }
        };
        custody.resources = Some(Arc::clone(&resources));
        let total_before = resources.observe_total_parent_scope()?;
        require(
            !registration.require_total_quota || total_before.verified,
            "original aggregate quota unverified",
        )?;
        let output = linux::empty_owned_directory(&registration.output_root)?;
        let control = linux::empty_owned_directory(&registration.control_root)?;
        let control_limit = (registration.max_output_bytes / 8).min(64 << 20);
        let held = linux::LinuxHeld {
            output,
            control,
            output_path: registration.output_root.clone(),
            control_path: registration.control_root.clone(),
            source: registration.output_root.join("source"),
            inputs: std::mem::take(&mut custody.admission_files),
            sources: Vec::new(),
            checkpoints: Vec::new(),
            journals: Vec::new(),
            artifacts: Vec::new(),
            partial_outputs: Vec::new(),
            dependency_directories: Vec::new(),
            control_written: 0,
            control_limit,
            device_admission: None,
        };
        *custody
            .held
            .lock()
            .map_err(|_| invalid("custody poisoned"))? = Some(held);
        linux::ensure_child_directory(
            &registration.output_root,
            &registration.checkpoint_directory,
        )?;
        let (registration_file, _) = own(&command.registration, MAX_JSON, false, &budget)?;
        let registration_file = Arc::new(registration_file.file);
        custody.registration = Some((command.registration.clone(), Arc::clone(&registration_file)));
        let (sources, dependencies, control_device, control_inode, cwd) = {
            let mut slot = custody
                .held
                .lock()
                .map_err(|_| invalid("custody poisoned"))?;
            let held = slot.as_mut().ok_or_else(|| invalid("custody absent"))?;
            let sources = install_sources(held, &registration)?;
            let mut dependencies = Vec::new();
            for path in &registration.dependency_directories {
                require(
                    !path.starts_with(&registration.output_root)
                        && !path.starts_with(&registration.control_root),
                    "dependency aliases current output/control owner",
                )?;
                let fd = directory_file(&open_directory(path)?)?;
                let m = fd.metadata().map_err(|_| invalid("dependency identity"))?;
                dependencies.push(json!({"path":path,"device":m.dev(),"inode":m.ino()}));
                held.dependency_directories
                    .push((fd, path.clone(), m.dev(), m.ino()));
            }
            let m = held
                .control
                .metadata()
                .map_err(|_| invalid("control identity"))?;
            (
                sources,
                json!(dependencies),
                m.dev(),
                m.ino(),
                held.output.try_clone().map_err(|_| invalid("cwd FD"))?,
            )
        };
        let sequence = NEXT_SOCKET
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |v| v.checked_add(1))
            .map_err(|_| invalid("socket sequence overflow"))?;
        let socket = registration
            .control_root
            .join(format!("utility-{}-{sequence}.sock", std::process::id()));
        require(socket.as_os_str().len() < 100, "utility Unix path bound")?;
        let control = {
            let slot = custody
                .held
                .lock()
                .map_err(|_| invalid("endpoint custody poisoned"))?;
            slot.as_ref()
                .ok_or_else(|| invalid("endpoint control custody absent"))?
                .control
                .try_clone()
                .map_err(|_| invalid("endpoint control FD clone"))?
        };
        custody.endpoint = Some(OwnedUtilityEndpoint::prepare(
            control,
            &registration.control_root,
            &socket,
            request.whole_deadline(),
        )?);
        let listener = custody
            .endpoint
            .as_mut()
            .ok_or_else(|| invalid("original endpoint owner absent"))?
            .bind()?;
        let parent_start = process_start(std::process::id(), &budget)?;
        let (start_ns, execution_ns, whole_ns) = linux::original_ticks(
            request.original_started(),
            request.execution_deadline(),
            request.whole_deadline(),
        )?;
        budget.reserve(registration.max_child_read_bytes)?;
        let registration_sha256 =
            startup_registration_sha256(&registration, &command.registration.sha256)?;
        let startup = json!({"schema":registration.family().domain(STARTUP),"execution_id":registration.execution_id,"window_id":window.window_id,"registration_sha256":registration_sha256,"registration":registration,"control":{"socket_path":socket,"parent_pid":std::process::id(),"parent_start_ticks":parent_start,"control_device":control_device,"control_inode":control_inode},"original_started_ns":start_ns,"execution_until_ns":execution_ns,"whole_until_ns":whole_ns,"source_root":registration.output_root.join("source"),"package_root":registration.output_root.join("source").join(linux::PACKAGE),"source_files":sources,"dependency_directories":dependencies,"runtime_module_pins":registration.runtime_endpoints.iter().map(|p|p.artifact.clone()).collect::<Vec<_>>(),"python_runtime":{"version":registration.trainer_python_version,"executable":registration.trainer_python_executable},"resources":resources.startup_json()?,"artifact_watch_profile":{"schema":"rz-rovent-training-artifact-watch/1","maximum_total_bytes":registration.max_output_bytes-control_limit,"maximum_file_bytes":MAX_TENSOR,"maximum_files":registration.max_artifact_files,"scope":"new_trainer_owned_output_tree_only"},"child_read_credit":{"reserved_bytes":registration.max_child_read_bytes,"scope":"charged_in_parent_shared_budget_no_refund"}});
        let startup_raw = encode_startup(registration.family(), startup)?;
        require(
            startup_raw.len() <= crate::MAX_ORIGINAL_INPUT_BYTES,
            "utility startup stdin bound",
        )?;
        let startup_sha = format!("{:x}", Sha256::digest(&startup_raw));
        {
            let mut slot = custody
                .held
                .lock()
                .map_err(|_| invalid("custody poisoned"))?;
            let held = slot.as_mut().ok_or_else(|| invalid("custody absent"))?;
            let file = linux::write_new(
                &registration.control_root.join("utility-startup.json"),
                &startup_raw,
                held,
                true,
            )?;
            held.journals.push(file);
        }
        let image_verified = Arc::new(AtomicBool::new(false));
        let stop = Arc::new(AtomicBool::new(false));
        let child_identity = Arc::new(Mutex::new(None));
        let ctx = Broker {
            listener,
            parent_pid: std::process::id(),
            parent_start,
            interpreter: registration.trainer_python_executable.clone(),
            startup_sha256: startup_sha.clone(),
            registration: registration.clone(),
            window: window.clone(),
            preparation: preparation.clone(),
            export_request: export_request.clone(),
            forward_admission_until: original_forward_deadline(
                request.execution_deadline(),
                window.checkpoint_reserve_ns,
            )?,
            until: request.execution_deadline(),
            whole_until: request.whole_deadline(),
            cancel: Arc::clone(&cancel),
            stop: Arc::clone(&stop),
            image_verified: Arc::clone(&image_verified),
            held: Arc::clone(&custody.held),
            journal: Arc::clone(&custody.journal),
            budget: Arc::clone(&budget),
            resources: Arc::clone(&resources),
            dataset: dataset.clone(),
            ancestors: Arc::clone(&ancestors),
            checkpoint: custody.checkpoint.clone(),
            child_identity: Arc::clone(&child_identity),
            fresh_identity,
            registration_file,
            registration_pin: command.registration.clone(),
            archives: Arc::clone(&custody.archives),
            archive_failures: Arc::clone(&custody.archive_failures),
        };
        let broker_cancel = Arc::clone(&cancel);
        let handle = thread::Builder::new()
            .name("rovent-utility-rpc".into())
            .spawn(move || {
                let result = serve(ctx);
                if result.is_err() {
                    broker_cancel.store(true, Ordering::Release);
                }
                result
            })
            .map_err(|_| invalid("utility broker spawn"))?;
        custody.broker = Some(PendingBroker { handle, stop });
        let args = [
            OsString::from("-I"),
            OsString::from("-S"),
            OsString::from("-B"),
            OsString::from("-c"),
            OsString::from(linux::BOOTSTRAP),
            registration
                .output_root
                .join("source")
                .join(linux::PACKAGE)
                .into_os_string(),
            OsString::from(match registration.family() {
                UtilityRuntimeFamily::CohortV1 => "rz_pals_model.rovent_utility_training_child",
                UtilityRuntimeFamily::InterventionV2 => {
                    "rz_pals_model.rovent_utility_intervention_child"
                }
            }),
            OsString::from("--execute"),
            OsString::from("--startup-sha256"),
            OsString::from(startup_sha),
            OsString::from("--startup-bytes"),
            OsString::from(startup_raw.len().to_string()),
            OsString::from("--execution-until-ns"),
            OsString::from(execution_ns.to_string()),
            OsString::from("--whole-until-ns"),
            OsString::from(whole_ns.to_string()),
        ];
        let mut verify_image = |image: &File| {
            budget
                .verify_held(image, &registration.trainer_python_executable, false)
                .map(|_| {
                    image_verified.store(true, Ordering::Release);
                })
                .map_err(|_| ())
        };
        let artifact_watch = RoventTrainingArtifactTreeWatch::registered(
            registration.max_output_bytes - control_limit,
            MAX_TENSOR,
            registration.max_artifact_files,
        )?;
        let original_input = OriginalProcessInput {
            bytes: &startup_raw,
            window: OriginalProcessWindow::new(
                request.original_started(),
                request.execution_deadline(),
                request.whole_deadline(),
            )?,
        };
        // The broker only authenticates metadata/RPC and checkpoint publication.
        // Keep preparation failures outside device admission. After this first
        // GPU-capable supervisor call is attempted, process=None cannot prove
        // that no child was issued, so the original custody retains the slot.
        {
            let mut slot = custody
                .held
                .lock()
                .map_err(|_| invalid("custody poisoned before device dispatch"))?;
            let held = slot
                .as_mut()
                .ok_or_else(|| invalid("custody absent before device dispatch"))?;
            require(
                held.device_admission.is_none(),
                "utility device reservation already present before dispatch",
            )?;
            held.device_admission = linux::ProcessDeviceAdmission::acquire_for_dispatch(
                &window.device,
                &cancel,
                request.execution_deadline(),
                request.whole_deadline(),
            )?;
        }
        custody.process = Some(
            supervise_rovent_training_numerical_input_in_original_window(
                &interpreter_fd,
                &args,
                &cwd,
                ProcessLimits {
                    wall_ms: command.scope.wall_ms,
                    shutdown_grace_ms: command.scope.cleanup_ms,
                    max_output_bytes: command.scope.max_stdio_bytes,
                    max_child_processes: 1,
                },
                Some(&cancel),
                &artifact_watch,
                original_input,
                &mut verify_image,
                OriginalNumericalEnvironment::Fp32Tf32Off,
            )?,
        );
        if custody.process.as_ref().is_none_or(|process| {
            !process.transport_complete() || process.process().receipt.exit_code != Some(0)
        }) {
            cancel.store(true, Ordering::Release);
        }
        if let Some(broker) = &custody.broker {
            broker.stop.store(true, Ordering::Release);
        }
        while custody
            .broker
            .as_ref()
            .is_some_and(|b| !b.handle.is_finished())
            && Instant::now() < request.whole_deadline()
        {
            thread::park_timeout(Duration::from_millis(2));
        }
        require(
            custody
                .broker
                .as_ref()
                .is_some_and(|b| b.handle.is_finished()),
            "utility broker closure unknown",
        )?;
        custody
            .broker
            .take()
            .ok_or_else(|| invalid("broker missing"))?
            .handle
            .join()
            .map_err(|_| invalid("utility broker panic"))??;
        // The broker has joined, so these writes cannot wait on its archive
        // mutex. Failure cleanup uses try_lock and retains any partial files.
        {
            let process = custody
                .process
                .as_ref()
                .ok_or_else(|| invalid("supervised utility process absent"))?;
            let mut slot = custody
                .held
                .lock()
                .map_err(|_| invalid("utility custody poisoned"))?;
            let held = slot
                .as_mut()
                .ok_or_else(|| invalid("utility custody absent"))?;
            preserve_transport(process, held)?;
        }
        let process = custody
            .process
            .as_ref()
            .ok_or_else(|| invalid("utility process missing"))?;
        process.checked_loaded_image()?;
        process.checked_timing()?;
        require(
            process.transport_complete()
                && process.process().receipt.stop == ProcessStop::Exited
                && process.process().receipt.exit_code == Some(0),
            "utility original process did not exit successfully",
        )?;
        let summary: Value = decode(&process.process().stdout)?;
        let terminal_pin = pin(&summary["result"]["receipt"])?;
        require(
            terminal_pin.path == registration.receipt_path,
            "utility terminal receipt escaped original scope",
        )?;
        let (terminal_file, terminal_raw) = own(&terminal_pin, MAX_JSON, true, &budget)?;
        let terminal: Value = decode(&terminal_raw)?;
        require(
            terminal["schema"] == window.family().domain(RESULT)
                && terminal["execution_id"] == registration.execution_id
                && terminal["window_id"] == window.window_id
                && terminal["status"] == "completed"
                && terminal["physical_completion"] == "confirmed"
                && terminal["arena_eligible"] == false
                && summary["result"]["status"] == "completed",
            "utility terminal identity/status/physical completion",
        )?;
        for key in [
            "schema",
            "execution_id",
            "window_id",
            "completed_updates",
            "window_completed_updates",
            "dispatched_updates",
            "window_dispatched_updates",
            "physical_completion",
            "arena_eligible",
        ] {
            require(
                summary[key] == terminal[key],
                "utility stdout/held terminal receipt differ",
            )?;
        }
        let journal = custody
            .journal
            .lock()
            .map_err(|_| invalid("utility journal poisoned"))?;
        let ended = journal
            .ended
            .as_ref()
            .ok_or_else(|| invalid("actual utility end RPC absent"))?;
        require(
            ended["original_end"]["physical_completion"] == "confirmed"
                && ended["original_end"]["quarantined"] == false
                && ended["original_end"]["completed_updates"] == terminal["completed_updates"]
                && ended["original_end"]["dispatched_updates"] == terminal["dispatched_updates"],
            "terminal differs from original parent ledger/physical state",
        )?;
        verify_import_report(&terminal["bootstrap"], &registration)?;
        let mut slot = custody
            .held
            .lock()
            .map_err(|_| invalid("utility custody poisoned"))?;
        let held = slot
            .as_mut()
            .ok_or_else(|| invalid("utility custody missing"))?;
        let terminal_index = held.journals.len();
        held.journals.push(terminal_file);
        if let Some(export) = &export_request {
            require(
                journal.export.is_some(),
                "original learned export authority missing",
            )?;
            let path = PathBuf::from(
                export["path"]
                    .as_str()
                    .ok_or_else(|| invalid("export destination"))?,
            );
            require(
                path.starts_with(&registration.output_root)
                    && path != registration.receipt_path
                    && path.extension().is_some_and(|s| s == "json"),
                "learned export destination escaped owner",
            )?;
            let exported = pin(&terminal["result"]["artifact"])?;
            require(
                exported.path == path
                    && exported.bytes
                        <= export["max_output_bytes"]
                            .as_u64()
                            .ok_or_else(|| invalid("export output limit"))?,
                "actual utility export file differs",
            )?;
            let (file, raw) = own(
                &exported,
                rz_search::cpu_value::nnuec::v2::utility::MAX_BYTES as u64,
                true,
                &budget,
            )?;
            let value: Value = decode(&raw)?;
            require(
                value["training"]
                    == *journal
                        .export
                        .as_ref()
                        .ok_or_else(|| invalid("actual learned projection absent"))?,
                "utility export differs from original acknowledged completion lineage",
            )?;
            let base: rz_search::cpu_value::nnuec::v2::Checkpoint = decode(&base_raw)?;
            let base = rz_search::cpu_value::nnuec::v2::Model::new(base)
                .map_err(|error| invalid(error.to_string()))?;
            let exported_model: rz_search::cpu_value::nnuec::v2::utility::Checkpoint =
                decode(&raw)?;
            exported_model
                .validate(&base)
                .map_err(|error| invalid(error.to_string()))?;
            held.artifacts.push(file);
        }
        let closed = resources.observe_closed()?;
        require(
            closed.processes_empty && !closed.populated,
            "utility cgroup not physically empty",
        )?;
        let total_after = resources.observe_total_parent_scope()?;
        require(
            !registration.require_total_quota || total_after.verified,
            "utility aggregate quota became unverified",
        )?;
        // Both the original broker and actual numerical scope are closed.
        // A cleanup error must enter the existing failed/unknown path before
        // normal parent terminal publication; Drop alone cannot attest this.
        custody
            .endpoint
            .as_mut()
            .ok_or_else(|| invalid("original endpoint owner absent at normal close"))?
            .cleanup()?;
        verify_retained(held, &budget)?;
        if let Some(file) = &journal.file {
            budget.verify_held(&file.file, &file.pin, false)?;
        }
        dataset.revalidate(&budget)?;
        ancestors.revalidate(&budget)?;
        for ancestor in &custody.datasets {
            ancestor.revalidate(&budget)?;
        }
        if let Some(source) = &custody.checkpoint {
            source.revalidate(&budget)?;
        }
        if let Some(reclaimed) = &custody.old_run_reclamation {
            reclaimed.revalidate()?;
        }
        for archive in custody
            .archives
            .lock()
            .map_err(|_| invalid("utility archive custody poisoned"))?
            .iter()
        {
            archive.revalidate(&budget)?;
        }
        require(
            Instant::now() < request.whole_deadline(),
            "original utility preservation deadline expired",
        )?;
        let no_update = window.family() == UtilityRuntimeFamily::InterventionV2
            && ended["original_end"]["status"] == "no_update_checkpoint_not_published";
        if no_update {
            require(
                registration.action == RoventUtilityExecutionAction::Train
                    && terminal["window_completed_updates"] == 0
                    && terminal["window_dispatched_updates"] == 0
                    && ended["window_completed_updates"] == 0
                    && ended["window_dispatched_updates"] == 0
                    && ended["original_end"]["checkpoint"].is_null()
                    && journal.export.is_none()
                    && custody
                        .archives
                        .lock()
                        .map_err(|_| invalid("zero-update archive custody poisoned"))?
                        .is_empty(),
                "zero-update close cannot publish a checkpoint, export or learned update",
            )?;
        }
        if registration.action == RoventUtilityExecutionAction::Train && !no_update {
            require(
                journal
                    .events
                    .last()
                    .is_some_and(|event| event["kind"] == "checkpoint_published"),
                "normal utility close requires a final actual checkpoint after all consumed work",
            )?;
            let mut archives = custody
                .archives
                .lock()
                .map_err(|_| invalid("closed checkpoint custody poisoned"))?;
            let prior_archive_reserved = archives.iter().try_fold(0u64, |sum, archive| {
                sum.checked_add(archive.reserved_output_bytes())
                    .ok_or_else(|| invalid("closed archive reservation overflow"))
            })?;
            let latest = archives
                .last()
                .ok_or_else(|| invalid("final actual utility checkpoint absent"))?;
            let child = child_identity
                .lock()
                .map_err(|_| invalid("closed child identity custody poisoned"))?
                .clone()
                .ok_or_else(|| invalid("closed authenticated child absent"))?;
            require(
                child.pid == process.process().receipt.pid,
                "closed checkpoint child differs from original supervisor",
            )?;
            let source = ActualUtilityClosedCheckpoint {
                latest,
                end: &ended["original_end"],
                process,
                terminal: &held.journals[terminal_index],
                child: &child,
                registration: &registration,
                window: &window,
                preparation: &preparation,
                dataset: &dataset,
                ancestors: &ancestors,
                ledger: journal
                    .ledger
                    .as_ref()
                    .ok_or_else(|| invalid("actual utility ledger missing"))?,
                held,
                budget: &budget,
                deadline: request.whole_deadline(),
                cancel: &cancel,
                prior_archive_reserved,
            };
            match publish_closed_utility_checkpoint(&source) {
                Ok(archive) => {
                    archive.revalidate(&budget)?;
                    archives.push(archive)
                }
                Err(failure) => {
                    let failure_observation = failure.observation();
                    custody
                        .archive_failures
                        .lock()
                        .map_err(|_| invalid("closed archive failure custody poisoned"))?
                        .push(failure);
                    return Err(invalid(format!(
                        "closed utility checkpoint publication failed; running publication retained: {failure_observation}"
                    )));
                }
            }
        }
        if let Some(device) = held.device_admission.as_mut() {
            device.release_confirmed()?;
        }
        let archive_summary = {
            let archives = custody
                .archives
                .lock()
                .map_err(|_| invalid("archive summary custody poisoned"))?;
            json!({"published_count":archives.len(),"latest_publication":archives.last().map(|archive|archive.publication_pin())})
        };
        let mut observation = json!({"schema":window.family().domain("rz-rovent-utility-parent-outcome/1"),"status":"completed","execution_id":registration.execution_id,"window_id":window.window_id,"terminal_receipt":terminal_pin,"summary":summary,"ledger_end":ended,"dataset":dataset.observation(),"read_accounting":budget.observation(),"read_accounting_scope":{"parent":"registered_original_FD_and_bounded_kernel_reads","child":"declared_pinned_endpoints_one_reserved_credit_no_refund","transitive_import_and_mapped_IO":"unknown","torch_private_snapshot_IO":"unknown","global_OS_IO_enforced":false},"process_receipt":process.process().receipt,"process_observation":process.observation(),"resources":closed,"aggregate_resource_scope":{"required":registration.require_total_quota,"before":total_before,"after":total_after},"past_native_authority_reissued":false,"past_physical_completion_reissued":false,"performance_claim":false,"tensor_state_assurance":"current_child_measurement_and_original_metadata_replay_not_independent_torch_recomputation"});
        observation["checkpoint_archives"] = archive_summary;
        if no_update {
            observation["training_status"] = json!("no_update_checkpoint_not_published");
            observation["training_complete"] = json!(false);
            observation["new_optimizer_updates"] = json!(0);
            observation["new_checkpoint_published"] = json!(false);
        }
        let receipt = linux::write_new(
            &registration
                .control_root
                .join("utility-parent-terminal.json"),
            &encode(&observation)?,
            held,
            true,
        )?;
        budget.verify_held(&receipt.file, &receipt.pin, false)?;
        let parent_receipt = receipt.pin.clone();
        held.journals.push(receipt);
        // Detailed kernel/lineage observations stay in the held report. The
        // CLI emits a complete bounded JSON reference instead of truncating a
        // large successful resource report into invalid JSON after training.
        let mut outcome = json!({"schema":window.family().domain("rz-rovent-utility-parent-outcome/1"),"status":"completed","execution_id":registration.execution_id,"window_id":window.window_id,"parent_receipt":parent_receipt,"terminal_receipt":terminal_pin,"checkpoint_archives":observation["checkpoint_archives"],"completed_updates":terminal["completed_updates"],"window_completed_updates":terminal["window_completed_updates"],"dispatched_updates":terminal["dispatched_updates"],"physical_completion":"confirmed","process_cleanup":"confirmed_original_supervisor","host_peak_bytes":observation["resources"]["memory_peak_bytes"],"read_accounting":budget.observation(),"performance_claim":false,"past_native_authority_reissued":false});
        if no_update {
            outcome["training_status"] = json!("no_update_checkpoint_not_published");
            outcome["training_complete"] = json!(false);
            outcome["new_optimizer_updates"] = json!(0);
            outcome["new_checkpoint_published"] = json!(false);
        }
        Ok(outcome)
    })();
    match result {
        Ok(observation) => Ok(RoventUtilityTrainingOutcome {
            success: true,
            observation,
            _owner: Some(Box::new(custody)),
        }),
        Err(error) => {
            admission.quarantined = true;
            cancel.store(true, Ordering::Release);
            if let Some(broker) = &custody.broker {
                broker.stop.store(true, Ordering::Release);
            }
            while custody
                .broker
                .as_ref()
                .is_some_and(|broker| !broker.handle.is_finished())
                && Instant::now() < request.whole_deadline()
            {
                thread::park_timeout(Duration::from_millis(2));
            }
            let broker_closed = if custody
                .broker
                .as_ref()
                .is_some_and(|broker| broker.handle.is_finished())
            {
                custody
                    .broker
                    .take()
                    .is_some_and(|broker| broker.handle.join().is_ok())
            } else {
                custody.broker.is_none()
            };
            // An ended listener can retire its pathname without issuing any
            // child/device physical fence. Unknown broker lifetime is retained.
            if let Some(endpoint) = custody.endpoint.as_mut() {
                if broker_closed && Instant::now() < request.whole_deadline() {
                    if let Err(cleanup_error) = endpoint.cleanup() {
                        custody._endpoint_cleanup_error = Some(cleanup_error.to_string());
                    }
                } else {
                    custody._endpoint_cleanup_error = Some(
                        "endpoint preserved: broker closure or original cleanup window unknown"
                            .into(),
                    );
                }
            }
            let observation = json!({"schema":custody.family.unwrap_or(UtilityRuntimeFamily::CohortV1).domain("rz-rovent-utility-parent-outcome/1"),"status":"failed","error":error.to_string(),"read_accounting":budget.observation(),"process_receipt":custody.process.as_ref().map(|p|&p.process().receipt),"physical_completion":"unknown","quarantined":true,"retry_authorized":false,"work_or_read_credit_refunded":false,"past_native_authority_reissued":false});
            let mut observation = observation;
            observation["broker_thread_joined"] = json!(broker_closed);
            if let Ok(archives) = custody.archives.try_lock() {
                observation["checkpoint_archives"] = json!({"published_count":archives.len(),"latest_publication":archives.last().map(|archive|archive.publication_pin())});
            }
            if let Ok(failures) = custody.archive_failures.try_lock() {
                observation["checkpoint_archive_failures"] = json!(failures.len());
            }
            if let Ok(journal) = custody.journal.try_lock() {
                observation["cost_high_water"] = journal.cost_high_water.clone();
                observation["end_receipt"] = json!(journal.end_receipt);
                observation["original_end_observation"] =
                    journal.ended.clone().unwrap_or(Value::Null);
                observation["cost_observation_scope"] =
                    json!("current_original_peer_reports_no_refund_not_independent_OS_measurement");
                if let Some(ledger) = &journal.ledger {
                    let observed = ledger.observation();
                    observation["ledger"] = json!({"completed_updates":observed["completed_updates"],"observed_dispatched_updates":observed["observed_dispatched_updates"],"window_completed_updates":observed["window_completed_updates"],"window_dispatched_updates":observed["window_dispatched_updates"],"ancestor_updates":observed["ancestor_updates"],"committed_cost":observed["committed_cost"],"pending_observed_cost":observed["pending_observed_cost"],"pending_receipt":!observed["pending_receipt"].is_null()});
                }
                observation["rpc_journal"] = journal
                    .file
                    .as_ref()
                    .map(|file| json!(file.pin))
                    .unwrap_or(Value::Null);
            }
            // Preserve a bounded actual failure report when this same owner
            // still has time and control credit; inability to preserve it is
            // itself recorded rather than manufacturing a successful receipt.
            if Instant::now() < request.whole_deadline()
                && let Ok(mut slot) = custody.held.try_lock()
                && let Some(held) = slot.as_mut()
            {
                if let Some(process) = custody.process.as_ref()
                    && preserve_transport(process, held).is_err()
                {
                    observation["transport_preservation"] =
                        json!({"status":"unknown","partial_files_retained":true});
                }
                match encode(&observation).and_then(|raw| {
                    linux::write_new(
                        &held.control_path.join("utility-parent-failure.json"),
                        &raw,
                        held,
                        true,
                    )
                }) {
                    Ok(file) => {
                        observation["parent_failure_receipt"] = json!(file.pin);
                        held.journals.push(file);
                    }
                    Err(_) => {
                        observation["parent_failure_receipt"] =
                            json!({"status":"unknown","partial_files_retained":true})
                    }
                }
            }
            match FAILED_CUSTODY.lock() {
                Ok(mut owner) => {
                    if owner.is_none() {
                        *owner = Some(Box::new(custody));
                    } else {
                        std::mem::forget(custody)
                    }
                }
                Err(_) => std::mem::forget(custody),
            }
            // The detailed end/partial result lives in the retained failure
            // receipt. Keep stdout bounded even if its original RPC used the
            // whole frame allowance; no known tail is silently discarded.
            let _ = observation
                .as_object_mut()
                .map(|value| value.remove("original_end_observation"));
            Ok(RoventUtilityTrainingOutcome {
                success: false,
                observation,
                _owner: None,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn startup_draft(family: UtilityRuntimeFamily) -> Value {
        // Declaration-only transport fixture. No FD, peer, lease or process
        // authority is created by building these original startup fields.
        json!({
            "schema": family.domain(STARTUP),
            "execution_id": "startup-test",
            "window_id": "startup-window",
            "registration_sha256": "12".repeat(32),
            "registration": {"fixture": "original-registration"},
            "control": {"fixture": "original-control"},
            "original_started_ns": 100,
            "execution_until_ns": 200,
            "whole_until_ns": 300,
            "source_root": "/owned/source",
            "package_root": "/owned/source/package",
            "source_files": [{"fixture": "original-source-pin"}],
            "dependency_directories": [],
            "runtime_module_pins": [],
            "python_runtime": {"fixture": "original-runtime"},
            "resources": {"fixture": "original-resource-observation"},
            "artifact_watch_profile": {"fixture": "original-output-bound"},
            "child_read_credit": {"fixture": "original-reserved-credit"}
        })
    }

    fn startup_registration(family: UtilityRuntimeFamily) -> RoventUtilityExecutionRegistration {
        use crate::rovent_training_process::recovery::{
            RoventTrainerRuntimeEndpoint, rovent_trainer_dependency_set_sha256,
        };
        let pin = |path: &str| json!({"path":path,"sha256":"1".repeat(64),"bytes":128});
        let endpoints: Vec<RoventTrainerRuntimeEndpoint> = serde_json::from_value(json!([
            {"name":"torch.__file__","artifact":pin("/deps/torch/__init__.py")},
            {"name":"torch._C.__file__","artifact":pin("/deps/torch/native.so")}
        ]))
        .unwrap();
        let mut value = json!({
            "schema":"rz-rovent-utility-training-registration/3","data_family":"intervention_v2",
            "execution_id":"startup-test","action":"train","export_request":null,
            "preparation":pin("/input/preparation.json"),"window_spec":pin("/input/window.json"),
            "trainer_source_sha256":"2".repeat(64),"trainer_python_executable":pin("/runtime/python3"),
            "trainer_python_version":"3.12.2","dependency_directories":["/deps"],
            "trainer_dependency_set_sha256":rovent_trainer_dependency_set_sha256(&endpoints).unwrap(),
            "runtime_endpoints":endpoints,"current_dataset":{"root":"/input/utility-cohort",
                "publication":pin("/input/utility-cohort/publication.json"),
                "durable_ack":pin("/input/utility-cohort/ack.json")},
            "ancestor_datasets":[],"initialization_source":{"mode":"initialize","utility":pin("/input/utility.json")},
            "frozen_nnuec_base":pin("/input/base.json"),"frozen_pc_sources":[
                {"role":"proposer","artifact":pin("/input/p.onnx"),"model_semantics_sha256":"3".repeat(64),"model_identity_sha256":"4".repeat(64),"encoding_sha256":"5".repeat(64)},
                {"role":"critic","artifact":pin("/input/c.onnx"),"model_semantics_sha256":"3".repeat(64),"model_identity_sha256":"4".repeat(64),"encoding_sha256":"5".repeat(64)}],
            "resources_policy":{"cpu_quota_millicpus":2000,"memory_high_bytes":6_u64<<30,"memory_max_bytes":12_u64<<30,"pids_max":64,"affinity":[0,2],"device":"cpu"},
            "require_total_quota":true,"output_root":"/run/output","control_root":"/run/control",
            "checkpoint_directory":"/run/output/checkpoints","receipt_path":"/run/output/receipt.json",
            "max_rpc_messages":100,"max_rpc_bytes":1<<20,"max_read_bytes":4<<20,
            "max_child_read_bytes":1<<20,"max_output_bytes":2<<20,"max_artifact_files":64
        });
        if family == UtilityRuntimeFamily::CohortV1 {
            value["schema"] = json!("rz-rovent-utility-training-registration/2");
            value.as_object_mut().unwrap().remove("data_family");
        }
        serde_json::from_value(value).unwrap()
    }

    #[test]
    fn utility_legacy_startup_preserves_original_raw_registration_pin() {
        let registration = startup_registration(UtilityRuntimeFamily::CohortV1);
        let raw = serde_json::to_vec_pretty(&registration).unwrap();
        let raw_sha256 = format!("{:x}", Sha256::digest(&raw));
        let selected = startup_registration_sha256(&registration, &raw_sha256).unwrap();
        assert_eq!(selected, raw_sha256);
        assert_ne!(
            selected,
            registration
                .as_cohort()
                .unwrap()
                .canonical_sha256()
                .unwrap()
        );
        assert!(registration.as_intervention().is_err());
        let mut draft = startup_draft(UtilityRuntimeFamily::CohortV1);
        draft["registration"] = json!(registration);
        draft["registration_sha256"] = json!(selected);
        assert_eq!(
            encode_startup(UtilityRuntimeFamily::CohortV1, draft.clone()).unwrap(),
            encode(&draft).unwrap()
        );
    }

    #[test]
    fn utility_intervention_startup_uses_typed_canonical_registration_digest() {
        let registration = startup_registration(UtilityRuntimeFamily::InterventionV2);
        let original_raw = serde_json::to_vec_pretty(&registration).unwrap();
        let raw_sha256 = format!("{:x}", Sha256::digest(&original_raw));
        let selected = startup_registration_sha256(&registration, &raw_sha256).unwrap();
        assert_eq!(
            selected,
            registration
                .as_intervention()
                .unwrap()
                .canonical_sha256()
                .unwrap()
        );
        assert_ne!(selected, raw_sha256);
        assert!(registration.as_cohort().is_err());
        let mut draft = startup_draft(UtilityRuntimeFamily::InterventionV2);
        draft["registration"] = json!(registration);
        draft["registration_sha256"] = json!(selected);
        let decoded: Value =
            decode(&encode_startup(UtilityRuntimeFamily::InterventionV2, draft).unwrap()).unwrap();
        assert_eq!(
            decoded["registration_sha256"],
            generation_digest(
                "rz-rovent-utility-training-registration/3",
                &decoded["registration"]
            )
            .unwrap()
        );
        assert_eq!(decoded["data_family"], "intervention_v2");
        assert_eq!(
            serde_json::to_vec_pretty(&registration).unwrap(),
            original_raw
        );
    }

    #[test]
    fn utility_intervention_startup_digest_cannot_use_legacy_domain_or_changed_declaration() {
        let registration = startup_registration(UtilityRuntimeFamily::InterventionV2);
        let selected = startup_registration_sha256(&registration, &"ab".repeat(32)).unwrap();
        let mut changed = json!(registration);
        assert_ne!(
            selected,
            generation_digest("rz-rovent-utility-training-registration/2", &changed).unwrap()
        );
        changed["execution_id"] = json!("changed-startup-test");
        let changed: RoventUtilityExecutionRegistration = serde_json::from_value(changed).unwrap();
        assert_ne!(
            selected,
            startup_registration_sha256(&changed, &"ab".repeat(32)).unwrap()
        );
    }

    #[test]
    fn utility_legacy_startup_keeps_exact_eighteen_fields_and_bytes() {
        let original = startup_draft(UtilityRuntimeFamily::CohortV1);
        let original_raw = encode(&original).unwrap();
        let raw = encode_startup(UtilityRuntimeFamily::CohortV1, original.clone()).unwrap();
        assert_eq!(raw, original_raw);
        let decoded: Value = decode(&raw).unwrap();
        assert_eq!(decoded, original);
        assert_eq!(decoded.as_object().unwrap().len(), 18);
        assert!(decoded.get("data_family").is_none());
        assert_eq!(decoded["schema"], STARTUP);
    }

    #[test]
    fn utility_intervention_startup_adds_original_family_before_encoding() {
        let original = startup_draft(UtilityRuntimeFamily::InterventionV2);
        let raw = encode_startup(UtilityRuntimeFamily::InterventionV2, original.clone()).unwrap();
        let decoded: Value = decode(&raw).unwrap();
        assert_eq!(decoded.as_object().unwrap().len(), 19);
        assert_eq!(decoded["schema"], "rz-rovent-utility-trainer-startup/2");
        assert_eq!(decoded["data_family"], "intervention_v2");
        for (key, value) in original.as_object().unwrap() {
            assert_eq!(decoded.get(key), Some(value));
        }
        let mut expected = original;
        expected["data_family"] = json!("intervention_v2");
        assert_eq!(raw, encode(&expected).unwrap());
    }

    #[test]
    fn utility_startup_rejects_foreign_or_pretagged_family_drafts() {
        let original = startup_draft(UtilityRuntimeFamily::CohortV1);
        assert!(encode_startup(UtilityRuntimeFamily::InterventionV2, original).is_err());
        let original = startup_draft(UtilityRuntimeFamily::InterventionV2);
        assert!(encode_startup(UtilityRuntimeFamily::CohortV1, original).is_err());
        for family in [
            UtilityRuntimeFamily::CohortV1,
            UtilityRuntimeFamily::InterventionV2,
        ] {
            let mut pretagged = startup_draft(family);
            pretagged["data_family"] = json!("intervention_v2");
            assert!(encode_startup(family, pretagged).is_err());
            let mut missing = startup_draft(family);
            missing
                .as_object_mut()
                .unwrap()
                .remove("original_started_ns");
            assert!(encode_startup(family, missing).is_err());
            let mut foreign = startup_draft(family);
            foreign["execution_authorized"] = json!(true);
            assert!(encode_startup(family, foreign).is_err());
        }
    }

    #[test]
    fn utility_checkpoint_reserve_closes_only_new_work_and_keeps_original_hard_limit() {
        let original_started = Instant::now();
        let hard = original_started + Duration::from_secs(20);
        let early = original_forward_deadline(hard, 5_000_000_000).unwrap();
        assert_eq!(
            early.duration_since(original_started),
            Duration::from_secs(15)
        );
        assert_eq!(hard.duration_since(early), Duration::from_secs(5));
        assert!(original_forward_deadline(hard, 0).is_err());
        // Reusing the same original hard limit cannot turn a larger reserve
        // into more time for updates or a later checkpoint deadline.
        let larger_reserve = original_forward_deadline(hard, 6_000_000_000).unwrap();
        assert!(larger_reserve < early);
        assert_eq!(hard, early + Duration::from_secs(5));
    }
    #[test]
    fn utility_decode_rejects_nested_duplicate_metadata() {
        assert!(decode::<Value>(br#"{"outer":{"epoch":1,"epoch":2}}"#).is_err());
    }
    #[test]
    fn utility_file_pin_has_no_unknown_authority_field() {
        assert!(pin(&json!({"path":"/safe/input.json","sha256":"a".repeat(64),"bytes":1,"execution_authorized":true})).is_err());
    }
    #[test]
    fn utility_checkpoint_post_cost_cannot_be_refunded_by_next_update() {
        let mut journal = Journal::new();
        journal.observe_cost(&json!({"snapshot_cost":{"output_bytes":10,"samples":8},"post_snapshot_cost":{"output_bytes":30,"samples":8}})).unwrap();
        assert!(
            journal
                .observe_cost(&json!({"cost":{"output_bytes":29,"samples":16}}))
                .is_err()
        );
    }
    #[test]
    fn utility_partial_end_cost_stays_monotone_without_optimizer_update() {
        let mut journal = Journal::new();
        journal
            .observe_cost(&json!({"cost":{"wall_time_ns":20,"partial_accumulation_ns":5}}))
            .unwrap();
        assert!(
            journal
                .observe_cost(&json!({"cost":{"wall_time_ns":21,"partial_accumulation_ns":4}}))
                .is_err()
        );
    }
}
