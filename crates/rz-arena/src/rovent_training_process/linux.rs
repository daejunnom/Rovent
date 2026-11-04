use super::numerical_child::{
    UnixFrameReadErrors, UnixFrameReadScope, is_direct_parent_group_leader, prepare_unix_ack,
    read_unix_frame, unix_peer_credentials, write_unix_ack,
};
use super::*;
use crate::pals_collect_v3::training_process::{
    FileIdentity, GenerationTrainingFilePinV3, ReadAccounting, ReadCredit, directory_file,
    open_directory, open_file, verify_bytes,
};
use crate::process::{
    OriginalNumericalEnvironment, OriginalProcessInput, OriginalProcessWindow,
    OwnedArtifactTreeWatch, supervise_numerical_input_in_original_window,
};
use rz_experiments::rovent_corpus::{MAX_CHUNK_BYTES, RoventCorpusReader};
use sha2::{Digest as _, Sha256};
use std::{
    ffi::OsString,
    fs::{self},
    io::Write,
    os::unix::{
        fs::{MetadataExt, PermissionsExt},
        net::{UnixListener, UnixStream},
    },
    path::{Component, Path},
    sync::{
        Mutex,
        atomic::{AtomicU64, Ordering},
    },
    thread::{self, JoinHandle},
    time::Duration,
};

const MAX_JSON: u64 = 16 << 20;
const MAX_CHECKPOINT: u64 = 2 << 30;
const MAX_SOURCE: u64 = 4 << 20;
pub(super) const PACKAGE: &str = "experiments/model-research/pals/src";
pub(super) const BOOTSTRAP: &str = "import sys,runpy; package=sys.argv.pop(1); module=sys.argv.pop(1); sys.path.insert(0,package); sys.argv[0]=module; runpy.run_module(module,run_name='__main__',alter_sys=True)";
static NEXT_CONTROL: AtomicU64 = AtomicU64::new(1);
static DEVICE_ADMISSIONS: std::sync::LazyLock<Mutex<BTreeSet<String>>> =
    std::sync::LazyLock::new(|| Mutex::new(BTreeSet::new()));
/// A process-local logical device admission, not a host-wide GPU lock. Unknown
/// closure deliberately keeps its slot reserved even if its receipt is dropped.
pub(super) struct ProcessDeviceAdmission {
    device: String,
    released: bool,
}
impl ProcessDeviceAdmission {
    pub(super) fn acquire_for_dispatch(
        device: &str,
        cancel: &AtomicBool,
        execution_deadline: Instant,
        whole_deadline: Instant,
    ) -> Result<Option<Self>, ArenaError> {
        check(
            !cancel.load(Ordering::Acquire)
                && Instant::now() < execution_deadline
                && Instant::now() < whole_deadline,
            "original device dispatch cancelled or expired before reservation",
        )?;
        Self::acquire(device)
    }

    pub(super) fn acquire(device: &str) -> Result<Option<Self>, ArenaError> {
        if device == "cpu" {
            return Ok(None);
        }
        let mut admitted = DEVICE_ADMISSIONS
            .try_lock()
            .map_err(|_| invalid("device admission busy/unknown"))?;
        check(
            admitted.insert(device.into()),
            "same process already owns or quarantines this logical CUDA device",
        )?;
        Ok(Some(Self {
            device: device.into(),
            released: false,
        }))
    }
    pub(super) fn release_confirmed(&mut self) -> Result<(), ArenaError> {
        check(!self.released, "device admission already released")?;
        let mut admitted = DEVICE_ADMISSIONS
            .try_lock()
            .map_err(|_| invalid("device release owner unknown"))?;
        check(
            admitted.remove(&self.device),
            "device admission slot missing",
        )?;
        self.released = true;
        Ok(())
    }
}

pub(super) struct HeldFile {
    pub(super) file: File,
    pub(super) pin: RoventTrainingFilePin,
    pub(super) identity: FileIdentity,
}
pub(super) struct LinuxHeld {
    pub(super) output: File,
    pub(super) control: File,
    pub(super) output_path: PathBuf,
    pub(super) control_path: PathBuf,
    pub(super) source: PathBuf,
    pub(super) inputs: Vec<HeldFile>,
    pub(super) sources: Vec<HeldFile>,
    pub(super) checkpoints: Vec<HeldFile>,
    pub(super) journals: Vec<HeldFile>,
    pub(super) artifacts: Vec<HeldFile>,
    pub(super) partial_outputs: Vec<File>,
    pub(super) dependency_directories: Vec<(File, PathBuf, u64, u64)>,
    pub(super) control_written: u64,
    pub(super) control_limit: u64,
    pub(super) device_admission: Option<ProcessDeviceAdmission>,
}
impl LinuxHeld {
    pub(super) fn checkpoint_files(&self) -> impl Iterator<Item = (&RoventTrainingFilePin, &File)> {
        self.checkpoints.iter().map(|held| (&held.pin, &held.file))
    }
    pub(super) fn export_files(&self) -> impl Iterator<Item = (&RoventTrainingFilePin, &File)> {
        self.artifacts.iter().map(|held| (&held.pin, &held.file))
    }
}
pub(super) struct PendingBroker {
    handle: JoinHandle<Result<(), ArenaError>>,
    stop: Arc<AtomicBool>,
}
struct Budget {
    maximum: u64,
    spent: Mutex<u64>,
    uncertain: AtomicBool,
}
impl Budget {
    fn read(
        &self,
        file: &File,
        pin: &RoventTrainingFilePin,
        retain: bool,
        until: Instant,
        cancel: &AtomicBool,
    ) -> Result<Vec<u8>, ArenaError> {
        let mut spent = self
            .spent
            .lock()
            .map_err(|_| invalid("read budget poisoned"))?;
        let remaining = self
            .maximum
            .checked_sub(*spent)
            .ok_or_else(|| invalid("read credit overspent"))?;
        let account = ReadAccounting::new();
        let mut credit = ReadCredit {
            remaining,
            maximum: remaining,
            until,
            cancel,
            account: Arc::clone(&account),
        };
        let result = verify_bytes(file, &pin.legacy(), retain, &mut credit);
        *spent = spent
            .checked_add(account.charged.load(Ordering::Acquire))
            .ok_or_else(|| invalid("read accounting overflow"))?;
        if account.uncertain.load(Ordering::Acquire) {
            self.uncertain.store(true, Ordering::Release);
        }
        result
    }
    fn charged(&self) -> u64 {
        match self.spent.try_lock() {
            Ok(value) => *value,
            Err(_) => {
                self.uncertain.store(true, Ordering::Release);
                self.maximum
            }
        }
    }
    fn unknown(&self) -> bool {
        self.uncertain.load(Ordering::Acquire) || self.spent.try_lock().is_err()
    }
}
impl RoventTrainingFilePin {
    fn legacy(&self) -> GenerationTrainingFilePinV3 {
        GenerationTrainingFilePinV3 {
            path: self.path.clone(),
            sha256: self.sha256.clone(),
            bytes: self.bytes,
        }
    }
    fn validate(&self, max: u64) -> Result<(), ArenaError> {
        self.legacy().validate(max)
    }
}
fn own_file(
    pin: &RoventTrainingFilePin,
    budget: &Budget,
    max: u64,
    retain: bool,
    until: Instant,
    cancel: &AtomicBool,
) -> Result<(HeldFile, Vec<u8>), ArenaError> {
    pin.validate(max)?;
    let file = open_file(&pin.path)?;
    let bytes = budget.read(&file, pin, retain, until, cancel)?;
    let identity = FileIdentity::from(
        &file
            .metadata()
            .map_err(|_| invalid("held input metadata"))?,
    );
    Ok((
        HeldFile {
            file,
            pin: pin.clone(),
            identity,
        },
        bytes,
    ))
}
fn parse<T: serde::de::DeserializeOwned>(raw: &[u8]) -> Result<T, ArenaError> {
    let text = std::str::from_utf8(raw).map_err(|_| invalid("UTF8 input"))?;
    // Persisted registrations use the same duplicate-key rejecting parser.
    rz_experiments::decode_json(text).map_err(|e| invalid(e.to_string()))
}
pub(super) fn path_ok(path: &Path) -> bool {
    path.is_absolute()
        && path.components().all(|p| match p {
            Component::RootDir => true,
            Component::Normal(v) => v.to_str().is_some_and(|s| {
                !s.is_empty()
                    && !s.chars().any(char::is_control)
                    && !s.eq_ignore_ascii_case(".env")
                    && !s.starts_with(".env.")
                    && !s.to_ascii_lowercase().contains("credential")
                    && !s.to_ascii_lowercase().contains("private_key")
                    && !s.to_ascii_lowercase().ends_with(".key")
            }),
            _ => false,
        })
}
pub(super) fn empty_owned_directory(path: &Path) -> Result<File, ArenaError> {
    check(
        path_ok(path),
        "owned directory must be a nonsecret normalized absolute path",
    )?;
    let dir = open_directory(path)?;
    let file = directory_file(&dir)?;
    let meta = file.metadata().map_err(|_| invalid("directory metadata"))?;
    check(
        meta.uid() == nix::unistd::geteuid().as_raw()
            && meta.permissions().mode() & 0o777 == 0o700
            && dir
                .entries()
                .map_err(|_| invalid("owned directory entries"))?
                .next()
                .is_none(),
        "exclusive empty owner0700 directory required",
    )?;
    Ok(file)
}
pub(super) fn ensure_child_directory(root: &Path, path: &Path) -> Result<(), ArenaError> {
    check(
        path_ok(path) && path != root && path.starts_with(root),
        "output child escapes owned root",
    )?;
    let mut directory = open_directory(root)?;
    for component in path
        .strip_prefix(root)
        .map_err(|_| invalid("output relative path"))?
        .components()
    {
        let Component::Normal(name) = component else {
            return Err(invalid("output path component"));
        };
        if directory.symlink_metadata(name).is_err() {
            directory
                .create_dir(name)
                .map_err(|_| invalid("output child create"))?;
        }
        use cap_fs_ext::DirExt;
        directory = directory
            .open_dir_nofollow(name)
            .map_err(|_| invalid("output child nofollow"))?;
    }
    Ok(())
}
pub(super) fn write_new(
    path: &Path,
    raw: &[u8],
    owner: &mut LinuxHeld,
    is_control: bool,
) -> Result<HeldFile, ArenaError> {
    check(
        !raw.is_empty() && raw.len() as u64 <= MAX_CHECKPOINT && path_ok(path),
        "bounded output write",
    )?;
    if is_control {
        owner.control_written = owner
            .control_written
            .checked_add(raw.len() as u64)
            .ok_or_else(|| invalid("control output byte overflow"))?;
        check(
            owner.control_written <= owner.control_limit,
            "control output reserve exhausted",
        )?;
    }
    let root_path = if is_control {
        &owner.control_path
    } else {
        &owner.output_path
    };
    let root_file = if is_control {
        &owner.control
    } else {
        &owner.output
    };
    let named_root = directory_file(&open_directory(root_path)?)?;
    check(
        FileIdentity::from(
            &named_root
                .metadata()
                .map_err(|_| invalid("named owner metadata"))?,
        )
        .same_inode(FileIdentity::from(
            &root_file
                .metadata()
                .map_err(|_| invalid("original owner metadata"))?,
        )),
        "original output/control directory was replaced",
    )?;
    let parent = if is_control {
        check(
            path.parent() == Some(owner.control_path.as_path()),
            "journal escapes original control FD",
        )?;
        cap_std::fs::Dir::from_std_file(
            owner
                .control
                .try_clone()
                .map_err(|_| invalid("control descriptor clone"))?,
        )
    } else {
        check(
            path.starts_with(&owner.output_path),
            "source escapes original output root",
        )?;
        open_directory(path.parent().ok_or_else(|| invalid("output parent"))?)?
    };
    use cap_fs_ext::{FollowSymlinks, OpenOptionsFollowExt};
    use cap_std::fs::OpenOptionsExt;
    let mut options = cap_std::fs::OpenOptions::new();
    options
        .write(true)
        .read(true)
        .create_new(true)
        .mode(0o600)
        .follow(FollowSymlinks::No)
        .custom_flags(libc::O_CLOEXEC);
    let file = parent
        .open_with(
            path.file_name().ok_or_else(|| invalid("output basename"))?,
            &options,
        )
        .map_err(|_| invalid("preserve existing output/create_new"))?
        .into_std();
    owner.partial_outputs.push(file);
    let file = owner
        .partial_outputs
        .last_mut()
        .expect("first original output FD already retained");
    let created = file
        .metadata()
        .map_err(|_| invalid("created original output metadata"))?;
    check(
        created.is_file()
            && created.uid() == nix::unistd::geteuid().as_raw()
            && created.nlink() == 1
            && created.len() == 0,
        "created original output is not an empty owned single-link file",
    )?;
    file.write_all(raw)
        .and_then(|_| file.sync_all())
        .map_err(|_| invalid("partial output write preserved"))?;
    directory_file(&parent)?
        .sync_all()
        .map_err(|_| invalid("original output parent sync; partial FD preserved"))?;
    let identity = FileIdentity::from(
        &file
            .metadata()
            .map_err(|_| invalid("published file metadata"))?,
    );
    let file = owner
        .partial_outputs
        .pop()
        .expect("published original output FD");
    Ok(HeldFile {
        file,
        pin: RoventTrainingFilePin {
            path: path.into(),
            sha256: format!("{:x}", Sha256::digest(raw)),
            bytes: raw.len() as u64,
        },
        identity,
    })
}

macro_rules! source {
    ($name:literal) => {
        (
            concat!("experiments/model-research/pals/src/rz_pals_model/", $name),
            include_bytes!(concat!(
                "../../../../experiments/model-research/pals/src/rz_pals_model/",
                $name
            ))
            .as_slice(),
        )
    };
}
pub(super) fn source_bundle() -> Result<BTreeMap<&'static str, &'static [u8]>, ArenaError> {
    let mut files = BTreeMap::new();
    for (_, component) in rz_experiments::SOURCE_COMPONENT_FILES_V5 {
        for (path, bytes) in *component {
            if path.starts_with(PACKAGE) && path.ends_with(".py") {
                files.insert(*path, *bytes);
            }
        }
    }
    for (path, bytes) in [
        source!("rovent_generation.py"),
        source!("rovent_training.py"),
        source!("rovent_corpus.py"),
        source!("rovent_external_root.py"),
        source!("rovent_tensor_training.py"),
        source!("rovent_training_window.py"),
        source!("rovent_training_authority.py"),
        source!("rovent_training_child.py"),
        source!("rovent_training_bootstrap.py"),
        source!("rovent_nnuec_training_state.py"),
        source!("nnuec_trainable.py"),
        source!("nnuec_reference.py"),
        source!("nnuec_v2_reference.py"),
        source!("nnuec_utility_reference.py"),
    ] {
        files.insert(path, bytes);
    }
    check(
        !files.is_empty()
            && files.len() <= 128
            && files.values().all(|v| v.len() as u64 <= MAX_SOURCE)
            && files.values().map(|v| v.len() as u64).sum::<u64>() <= 32 << 20,
        "compiled trainer source capacity",
    )?;
    Ok(files)
}
pub fn compiled_rovent_trainer_source_sha256() -> Result<String, ArenaError> {
    let inventory = source_bundle()?
        .iter()
        .map(|(path, bytes)| {
            (
                path.to_string(),
                json!({"bytes":bytes.len(),"sha256":format!("{:x}",Sha256::digest(bytes))}),
            )
        })
        .collect::<BTreeMap<_, _>>();
    digest_value("rz-rovent-trainer-source-bundle/1", &inventory)
}
fn install_sources(
    owner: &mut LinuxHeld,
    registration: &RoventTrainingRegistration,
) -> Result<Value, ArenaError> {
    check(
        registration.trainer_source_sha256 == compiled_rovent_trainer_source_sha256()?,
        "registered trainer source differs from compiled source",
    )?;
    let mut inventory = Vec::new();
    for (path, bytes) in source_bundle()? {
        let target = owner.source.join(path);
        let parent = target.parent().ok_or_else(|| invalid("source parent"))?;
        ensure_child_directory(&registration.output_root, parent)?;
        let held = write_new(&target, bytes, owner, false)?;
        inventory
            .push(json!({"relative_path":path,"sha256":held.pin.sha256,"bytes":held.pin.bytes}));
        owner.sources.push(held);
    }
    Ok(json!(inventory))
}
pub(super) fn proc_start(pid: u32) -> Result<u64, ArenaError> {
    let text = fs::read_to_string(format!("/proc/{pid}/stat"))
        .map_err(|_| invalid("original process stat"))?;
    let tail = text
        .rsplit_once(") ")
        .ok_or_else(|| invalid("process stat framing"))?
        .1;
    tail.split_ascii_whitespace()
        .nth(19)
        .and_then(|v| v.parse().ok())
        .ok_or_else(|| invalid("process start tick"))
}
pub(super) fn original_ticks(
    started: Instant,
    execution: Instant,
    whole: Instant,
) -> Result<(u64, u64, u64), ArenaError> {
    let ts = nix::time::clock_gettime(nix::time::ClockId::CLOCK_MONOTONIC)
        .map_err(|_| invalid("monotonic read"))?;
    let sampled = Instant::now();
    let tick = u64::try_from(ts.tv_sec())
        .ok()
        .and_then(|s| s.checked_mul(1_000_000_000))
        .and_then(|n| n.checked_add(ts.tv_nsec() as u64))
        .ok_or_else(|| invalid("monotonic overflow"))?;
    let ns = |d: Duration| u64::try_from(d.as_nanos()).map_err(|_| invalid("deadline overflow"));
    let start = tick
        .checked_sub(ns(sampled
            .checked_duration_since(started)
            .ok_or_else(|| invalid("future original start"))?)?)
        .ok_or_else(|| invalid("start underflow"))?;
    let future = |deadline: Instant| {
        tick.checked_add(ns(deadline
            .checked_duration_since(sampled)
            .ok_or_else(|| invalid("expired original deadline"))?)?)
            .ok_or_else(|| invalid("deadline overflow"))
    };
    Ok((start, future(execution)?, future(whole)?))
}
fn load_json(
    pin: &RoventTrainingFilePin,
    owner: &mut LinuxHeld,
    budget: &Budget,
    until: Instant,
    cancel: &AtomicBool,
) -> Result<Value, ArenaError> {
    let (held, raw) = own_file(pin, budget, MAX_JSON, true, until, cancel)?;
    owner.inputs.push(held);
    parse(&raw)
}
fn load_corpus(
    files: &RoventTrainingCorpusFiles,
    custody: &OriginalRoventCorpusCustody,
    owner: &mut LinuxHeld,
    budget: &Budget,
    until: Instant,
    cancel: &AtomicBool,
    ancestors: &[RoventCorpusIndex],
) -> Result<RoventCorpusIndex, ArenaError> {
    check(path_ok(&files.root), "corpus root")?;
    let m: RoventCorpusManifest =
        serde_json::from_value(load_json(&files.manifest, owner, budget, until, cancel)?)
            .map_err(|e| invalid(e.to_string()))?;
    let index: RoventCorpusIndex =
        serde_json::from_value(load_json(&files.index, owner, budget, until, cancel)?)
            .map_err(|e| invalid(e.to_string()))?;
    check(
        m == custody.manifest,
        "original native producer manifest differs from file",
    )?;
    custody.join_index(&index)?;
    let reader =
        RoventCorpusReader::new(m, index.clone(), ancestors).map_err(|e| invalid(e.to_string()))?;
    let mut seen = BTreeSet::new();
    for (slot, chunk) in index.chunks.iter().enumerate() {
        let pin = RoventTrainingFilePin {
            path: files.root.join(&chunk.path),
            sha256: chunk.sha256.clone(),
            bytes: chunk.bytes,
        };
        let (held, bytes) = own_file(&pin, budget, MAX_CHUNK_BYTES as u64, true, until, cancel)?;
        for row in reader
            .read_chunk(slot, &bytes)
            .map_err(|e| invalid(e.to_string()))?
        {
            let native = custody
                .rows
                .get(&row.origin_sha256)
                .ok_or_else(|| invalid("file row lacks original native custody"))?;
            check(
                seen.insert(row.origin_sha256.clone())
                    && row.record_sha256 == native.origin.record_sha256
                    && row.native.raw_retirement_sha256 == native.retirement_sha256
                    && row
                        .head_sources
                        .iter()
                        .map(|(k, v)| (k.clone(), v.raw_receipt_sha256.clone()))
                        .collect::<BTreeMap<_, _>>()
                        == native.head_raw_sha256,
                "original corpus row/head/retirement bytes changed",
            )?;
        }
        owner.inputs.push(held);
    }
    check(
        seen.len() == custody.rows.len(),
        "original corpus omitted a native row",
    )?;
    Ok(index)
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
struct BrokerContext {
    listener: UnixListener,
    parent_pid: u32,
    parent_start: u64,
    interpreter: RoventTrainingFilePin,
    startup_sha256: String,
    execution_id: String,
    window_id: String,
    until: Instant,
    whole_until: Instant,
    cancel: Arc<AtomicBool>,
    stop: Arc<AtomicBool>,
    image_verified: Arc<AtomicBool>,
    ledger: Arc<Mutex<TrainerLedger>>,
    held: Arc<Mutex<Option<LinuxHeld>>>,
    budget: Arc<Budget>,
    registration: RoventTrainingRegistration,
    max_messages: u64,
    max_bytes: u64,
    resources: Arc<RoventTrainingResources>,
    previous: Option<Arc<VerifiedRoventTrainingWindow>>,
}
fn read_frame(
    stream: &mut UnixStream,
    ctx: &BrokerContext,
    total: &mut u64,
) -> Result<Option<Vec<u8>>, ArenaError> {
    read_unix_frame(
        stream,
        UnixFrameReadScope {
            stop: &ctx.stop,
            cancel: &ctx.cancel,
            whole_until: ctx.whole_until,
            max_total_bytes: ctx.max_bytes,
            max_frame_bytes: MAX_TRAINER_RPC_BYTES,
            errors: UnixFrameReadErrors {
                partial_stop: "partial RPC frame at owner stop",
                deadline_cancel: "original RPC deadline/cancel",
                partial_eof: "partial RPC frame EOF",
                byte_overflow: "RPC byte overflow",
                byte_frame_bound: "RPC byte/frame budget",
                pipelined: "pipelined/trailing RPC frames are unsupported",
                read: "RPC read failed",
            },
        },
        total,
    )
}
fn peer_admit(stream: &UnixStream, ctx: &BrokerContext) -> Result<u32, ArenaError> {
    let peer = unix_peer_credentials(stream, "RPC SO_PEERCRED", "RPC peer PID")?;
    let pid = peer.pid;
    check(
        peer.uid == nix::unistd::geteuid().as_raw()
            && ctx.image_verified.load(Ordering::Acquire)
            && proc_start(ctx.parent_pid)? == ctx.parent_start,
        "RPC original parent/image/UID",
    )?;
    let status = fs::read_to_string(format!("/proc/{pid}/status"))
        .map_err(|_| invalid("RPC child status"))?;
    check(
        is_direct_parent_group_leader(pid, ctx.parent_pid, &status),
        "RPC peer is not the original direct process-group leader",
    )?;
    let image = File::open(format!("/proc/{pid}/exe")).map_err(|_| invalid("RPC child image"))?;
    ctx.budget
        .read(&image, &ctx.interpreter, false, ctx.until, &ctx.cancel)?;
    ctx.resources.verify_child_membership(pid)?;
    Ok(pid)
}
fn serve(ctx: BrokerContext) -> Result<(), ArenaError> {
    ctx.listener
        .set_nonblocking(true)
        .map_err(|_| invalid("RPC nonblocking listener"))?;
    let (mut stream, _) = loop {
        if ctx.stop.load(Ordering::Acquire) {
            return Ok(());
        }
        check(
            Instant::now() < ctx.until && !ctx.cancel.load(Ordering::Acquire),
            "RPC accept deadline/cancel",
        )?;
        match ctx.listener.accept() {
            Ok(v) => break v,
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                thread::park_timeout(Duration::from_millis(2))
            }
            Err(_) => return Err(invalid("RPC accept")),
        }
    };
    let peer = peer_admit(&stream, &ctx)?;
    let peer_start = proc_start(peer)?;
    stream
        .set_read_timeout(Some(Duration::from_millis(25)))
        .and_then(|_| stream.set_write_timeout(Some(Duration::from_millis(25))))
        .map_err(|_| invalid("RPC finite socket timeouts"))?;
    let mut total = 0_u64;
    while !ctx.stop.load(Ordering::Acquire) {
        let Some(raw) = read_frame(&mut stream, &ctx, &mut total)? else {
            break;
        };
        let request: Request = parse(&raw)?;
        check(
            Instant::now() < ctx.until || request.operation == "end_window",
            "execution deadline forbids further optimizer/checkpoint work",
        )?;
        check(
            request.schema == ROVENT_TRAINER_RPC
                && request.window_id == ctx.window_id
                && request.startup_sha256 == ctx.startup_sha256
                && request.sequence > 0
                && request.sequence <= ctx.max_messages
                && proc_start(peer)? == peer_start,
            "RPC window/sequence/peer lifetime",
        )?;
        let mut ledger = ctx
            .ledger
            .lock()
            .map_err(|_| invalid("RPC ledger poisoned"))?;
        check(
            request.sequence == ledger.next_sequence,
            "RPC duplicate/skipped sequence",
        )?;
        let payload = dispatch(&ctx, &mut ledger, &request.operation, &request.payload)?;
        ledger.next_sequence = ledger
            .next_sequence
            .checked_add(1)
            .ok_or_else(|| invalid("RPC sequence overflow"))?;
        ledger.bytes_spent = total;
        let ack=serde_json::to_vec(&json!({"schema":ROVENT_TRAINER_ACK,"window_id":ctx.window_id,"sequence":request.sequence,
            "operation":request.operation,"accepted":true,"payload":payload})).map_err(|e|invalid(e.to_string()))?;
        let ack = prepare_unix_ack(ack, MAX_TRAINER_RPC_BYTES, "RPC ack frame bound")?;
        write_unix_ack(
            &mut stream,
            &ack,
            "RPC ack write failed; original work remains charged",
        )?;
    }
    Ok(())
}

fn dispatch(
    ctx: &BrokerContext,
    ledger: &mut TrainerLedger,
    operation: &str,
    p: &Value,
) -> Result<Value, ArenaError> {
    check(
        ctx.registration.action == "train"
            || matches!(
                operation,
                "validate_checkpoint_state"
                    | "validate_learned_export"
                    | "begin_window"
                    | "end_window"
            ),
        "export-only child cannot begin/consume/update/checkpoint a training window",
    )?;
    match operation {
        "begin_window" => {
            check(
                ctx.registration.checkpoint_mode == "initialize" || ledger.checkpoint_checked,
                "resume/fork requires original previous checkpoint admission before begin",
            )?;
            ledger.begin(p)
        }
        "consume_sample" => ledger.consume(p),
        "completed_update" => {
            ledger.update(p)?;
            let receipt = json!({"schema":ROVENT_ORIGINAL_UPDATE,"execution_id":ctx.execution_id,"window_id":ctx.window_id,
                "sequence":ledger.updates.len(),"original_native_lineage_verified":true,"update":p});
            let raw = serde_json::to_vec(&receipt).map_err(|e| invalid(e.to_string()))?;
            let path = ctx
                .registration
                .control_root
                .join(format!("update-{:06}.json", ledger.updates.len()));
            let mut owner = ctx.held.lock().map_err(|_| invalid("custody poisoned"))?;
            let owner = owner.as_mut().ok_or_else(|| invalid("custody released"))?;
            let held = write_new(&path, &raw, owner, true)?;
            let ack = json!({"sha256":held.pin.sha256,"bytes":held.pin.bytes});
            owner.journals.push(held);
            Ok(ack)
        }
        "checkpoint_published" => {
            fields(
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
            )?;
            check(
                ledger.begun
                    && !ledger.ended
                    && ledger.pending.is_empty()
                    && p["schema"] == "rz-rovent-trainer-checkpoint/1"
                    && p["completed_updates"] == json!(ledger.completed)
                    && p["pending_microbatches"] == 0,
                "checkpoint is not an original committed update boundary",
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
            ledger.observe_cost(&p["cost"])?;
            if let Some(update) = ledger.updates.last() {
                for name in [
                    "model_state_sha256",
                    "optimizer_state_sha256",
                    "scheduler_state_sha256",
                    "rng_state_sha256",
                ] {
                    check(
                        update[name] == p[name],
                        "checkpoint changed model/optimizer/scheduler/RNG after last committed update",
                    )?;
                }
            }
            let pin: RoventTrainingFilePin = serde_json::from_value(p["checkpoint"].clone())
                .map_err(|e| invalid(e.to_string()))?;
            check(
                pin.path.parent() == Some(ctx.registration.checkpoint_directory.as_path())
                    && pin.path.extension().is_some_and(|s| s == "pt")
                    && ctx
                        .registration
                        .checkpoint
                        .as_ref()
                        .is_none_or(|i| i.path != pin.path),
                "checkpoint output aliases input or escapes original output",
            )?;
            let (held, _) = own_file(
                &pin,
                &ctx.budget,
                MAX_CHECKPOINT,
                false,
                ctx.until,
                &ctx.cancel,
            )?;
            check(
                !ledger
                    .checkpoints
                    .iter()
                    .any(|old| old["path"] == p["checkpoint"]["path"]),
                "duplicate checkpoint publication",
            )?;
            let mut owner = ctx.held.lock().map_err(|_| invalid("custody poisoned"))?;
            let owner = owner.as_mut().ok_or_else(|| invalid("custody released"))?;
            owner.checkpoints.push(held);
            let raw=serde_json::to_vec(&json!({"schema":ROVENT_ORIGINAL_CHECKPOINT,"execution_id":ctx.execution_id,"window_id":ctx.window_id,"checkpoint":p})).map_err(|e|invalid(e.to_string()))?;
            let retirement = write_new(
                &ctx.registration.control_root.join(format!(
                    "checkpoint-{:06}.json",
                    ledger.checkpoints.len() + 1
                )),
                &raw,
                owner,
                true,
            )?;
            ledger.checkpoint_receipts.push(json!(retirement.pin));
            owner.journals.push(retirement);
            ledger.checkpoints.push(p["checkpoint"].clone());
            ledger.checkpoint_states.push(p.clone());
            Ok(json!({}))
        }
        "end_window" => {
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
                    && p["schema"] == "rz-rovent-trainer-end/1"
                    && p["completed_updates"] == json!(ledger.completed),
                "original end scope/count mismatch",
            )?;
            check(
                p["quarantined"] == false
                    && p["physical_completion"] == "confirmed"
                    && p["pending_microbatches"] == 0
                    && ledger.pending.is_empty()
                    && (if ctx.registration.action == "train" {
                        !ledger.checkpoints.is_empty()
                    } else {
                        ledger.export_admitted
                            && ledger.updates.is_empty()
                            && ctx.previous.is_some()
                    }),
                "partial/uncertain training must retain failure custody",
            )?;
            ledger.observe_cost(&p["cost"])?;
            check(
                p["dispatched_updates"] == json!(ledger.completed)
                    && (if ctx.registration.action == "export" {
                        p["status"] == "export_complete"
                    } else {
                        matches!(
                            p["status"].as_str(),
                            Some(
                                "window_expired_at_boundary"
                                    | "await_new_frozen_v_corpus"
                                    | "declared_stage_updates_complete"
                                    | "sample_group_budget_exhausted"
                                    | "window_update_budget_complete"
                            )
                        )
                    }),
                "original terminal dispatch/status mismatch",
            )?;
            ledger.ended = true;
            ledger.end_cost = Some(p["cost"].clone());
            Ok(json!({}))
        }
        "validate_checkpoint_state" | "validate_learned_export" => {
            let result = validate_original_checkpoint_request(ctx, ledger, operation, p)?;
            if operation == "validate_checkpoint_state" {
                check(
                    !ledger.checkpoint_checked,
                    "checkpoint admission is single use",
                )?;
                ledger.checkpoint_checked = true;
            } else {
                check(
                    !ledger.export_admitted,
                    "learned export admission is single use",
                )?;
                ledger.export_admitted = true;
                ledger.export_admission = Some(result.clone());
            }
            Ok(result)
        }
        _ => Err(invalid("unsupported original trainer RPC operation")),
    }
}

fn seal_set(kind: &str, values: &BTreeSet<String>) -> Result<Value, ArenaError> {
    check(
        matches!(kind, "input" | "origin") && values.len() <= MAX_TRAINER_ROWS,
        "bounded retained seal set",
    )?;
    let mut hash = Sha256::new();
    hash.update(b"rz-rovent-training-seal-set/1\0");
    hash.update(kind.as_bytes());
    hash.update([0]);
    hash.update((values.len() as u64).to_le_bytes());
    for value in values {
        check(generation_sha(value), "retained source seal")?;
        let mut bytes = [0u8; 32];
        for (i, pair) in value.as_bytes().chunks_exact(2).enumerate() {
            let decode = |v: u8| if v <= b'9' { v - b'0' } else { v - b'a' + 10 };
            bytes[i] = (decode(pair[0]) << 4) | decode(pair[1]);
        }
        hash.update(bytes);
    }
    Ok(json!({"sha256":format!("{:x}",hash.finalize()),"count":values.len()}))
}
pub(super) fn lineage_teachers(recipe: &Value, stage: &Value) -> Result<Vec<String>, ArenaError> {
    let stages = recipe["stages"]
        .as_array()
        .filter(|s| s.len() <= 64)
        .ok_or_else(|| invalid("bounded recipe stages"))?;
    let mut current = stage.clone();
    let mut visited = BTreeSet::new();
    let mut teachers = BTreeSet::new();
    loop {
        let name = current
            .as_str()
            .ok_or_else(|| invalid("lineage stage id"))?;
        check(
            visited.insert(name.to_owned()),
            "cyclic original recipe lineage",
        )?;
        let declaration = stages
            .iter()
            .find(|s| s["stage_id"] == current)
            .ok_or_else(|| invalid("lineage stage absent"))?;
        for actor in declaration["teacher_actor_ids"]
            .as_array()
            .ok_or_else(|| invalid("teacher lineage array"))?
        {
            let id = actor
                .as_str()
                .filter(|v| generation_identifier(v))
                .ok_or_else(|| invalid("teacher actor"))?;
            teachers.insert(id.to_owned());
        }
        if declaration["parent_stage_id"].is_null() {
            break;
        }
        current = declaration["parent_stage_id"].clone();
    }
    Ok(teachers.into_iter().collect())
}
fn chain_digest(
    ledger: &TrainerLedger,
    previous: Option<&VerifiedRoventTrainingWindow>,
) -> Result<String, ArenaError> {
    let mut ancestors = Vec::new();
    let mut current = previous;
    let mut ids = BTreeSet::new();
    while let Some(owner) = current {
        check(
            ancestors.len() < 1024 && ids.insert(&owner.registration.execution_id),
            "original predecessor chain capacity/cycle",
        )?;
        ancestors.push(json!({"execution_id":owner.registration.execution_id,"updates_sha256":digest_value("rz-rovent-original-window-updates/1",&owner.journal.updates)?,"update_count":owner.journal.updates.len(),"completed":owner.journal.completed,"ancestors":owner.journal.ancestors}));
        current = owner.previous.as_deref();
    }
    digest_value(
        "rz-rovent-original-update-chain/1",
        &json!({"previous":ancestors,"updates_sha256":digest_value("rz-rovent-original-window-updates/1",&ledger.updates)?,"update_count":ledger.updates.len(),"completed":ledger.completed,"ancestors":ledger.ancestors}),
    )
}
fn post_snapshot_cost(original: &TrainerLedger, checkpoint: &Value) -> Result<Value, ArenaError> {
    let end = original
        .end_cost
        .as_ref()
        .ok_or_else(|| invalid("original post-snapshot cost unknown"))?;
    validate_cost(end)?;
    validate_cost(&checkpoint["cost"])?;
    let mut delta = BTreeMap::new();
    for name in COST_FIELDS {
        delta.insert(
            name,
            count(&end[name])?
                .checked_sub(count(&checkpoint["cost"][name])?)
                .ok_or_else(|| invalid("original post-snapshot cost regressed"))?,
        );
    }
    Ok(json!(delta))
}
fn validate_original_checkpoint_request(
    ctx: &BrokerContext,
    ledger: &TrainerLedger,
    operation: &str,
    p: &Value,
) -> Result<Value, ArenaError> {
    fields(
        p,
        &[
            "schema",
            "checkpoint",
            "recipe_sha256",
            "stage_id",
            "membership_sha256",
            "completed_updates",
            "ancestor_updates",
            "retained_input_set_sha256",
            "retained_input_count",
            "retained_origin_set_sha256",
            "retained_origin_count",
            "model_state_sha256",
            "optimizer_state_sha256",
            "scheduler_state_sha256",
            "rng_state_sha256",
            "sampler_state_sha256",
        ],
    )?;
    fields(&p["checkpoint"], &["sha256", "bytes"])?;
    let resume = operation == "validate_checkpoint_state";
    let original = if resume || ctx.registration.action == "export" {
        &ctx.previous
            .as_ref()
            .ok_or_else(|| invalid("resume has no original predecessor custody"))?
            .journal
    } else {
        ledger
    };
    let slot = original
        .checkpoint_states
        .iter()
        .position(|state| {
            state["checkpoint"]["sha256"] == p["checkpoint"]["sha256"]
                && state["checkpoint"]["bytes"] == p["checkpoint"]["bytes"]
        })
        .ok_or_else(|| invalid("checkpoint was not retired by original trainer owner"))?;
    let checkpoint = &original.checkpoint_states[slot];
    check(
        checkpoint["completed_updates"] == json!(original.completed),
        "old checkpoint cannot inherit later update counts",
    )?;
    check(
        (resume
            && !ledger.begun
            && matches!(ctx.registration.checkpoint_mode.as_str(), "resume" | "fork")
            || !resume
                && ctx.registration.action == "export"
                && ledger.checkpoint_checked
                && ledger.begun
                && !ledger.ended
                && !ledger.export_admitted)
            && p["schema"]
                == (if resume {
                    "rz-rovent-trainer-resume/1"
                } else {
                    "rz-rovent-trainer-export/1"
                })
            && p["recipe_sha256"] == original.window["recipe_sha256"]
            && p["stage_id"] == original.window["stage_id"]
            && p["membership_sha256"] == original.membership_sha256
            && p["completed_updates"] == json!(original.completed)
            && p["ancestor_updates"] == json!(original.ancestors),
        "original checkpoint/source/stage/count admission mismatch",
    )?;
    for name in [
        "model_state_sha256",
        "optimizer_state_sha256",
        "scheduler_state_sha256",
        "rng_state_sha256",
        "sampler_state_sha256",
    ] {
        sha(&p[name])?;
        check(
            p[name] == checkpoint[name],
            "resumed tensor/optimizer/scheduler/RNG/sampler state differs from original checkpoint",
        )?;
    }
    let input = seal_set("input", &original.consumed_inputs)?;
    let origin = seal_set("origin", &original.consumed_origins)?;
    check(
        p["retained_input_set_sha256"] == input["sha256"]
            && p["retained_input_count"] == input["count"]
            && p["retained_origin_set_sha256"] == origin["sha256"]
            && p["retained_origin_count"] == origin["count"],
        "resumed/export input-origin ownership set mismatch",
    )?;
    let checkpoint_sha = sha(&p["checkpoint"]["sha256"])?;
    let chain_previous = if resume || ctx.registration.action == "export" {
        ctx.previous
            .as_ref()
            .and_then(|owner| owner.previous.as_deref())
    } else {
        ctx.previous.as_deref()
    };
    let original_chain = chain_digest(original, chain_previous)?;
    let original_post_cost = post_snapshot_cost(original, checkpoint)?;
    if resume {
        check(
            original.ended
                && (ctx.registration.checkpoint_mode == "fork"
                    || original.profile_sha256 == ledger.profile_sha256),
            "original resume fixed training profile/closure",
        )?;
        let receipt = original
            .checkpoint_receipts
            .get(slot)
            .ok_or_else(|| invalid("original checkpoint FD retirement receipt absent"))?;
        return Ok(
            json!({"schema":"rz-rovent-original-learner-resume-admission/1","recipe_sha256":original.window["recipe_sha256"],"stage_id":original.window["stage_id"],
            "checkpoint_sha256":checkpoint_sha,"training_profile_sha256":original.profile_sha256,"original_receipt_sha256":receipt["sha256"],
            "original_update_chain_sha256":original_chain,"completed_updates":original.completed,
            "retained_input_set_sha256":input["sha256"],"retained_input_count":input["count"],"retained_origin_set_sha256":origin["sha256"],"retained_origin_count":origin["count"],
            "post_snapshot_cost":original_post_cost,"original_native_lineage_verified":true}),
        );
    }
    let ledger = original;
    check(
        ledger.completed == ledger.stage_limits,
        "learned export requires every declared stage update",
    )?;
    let mut totals = BTreeMap::new();
    for role in ROLES {
        totals.insert(
            role,
            ledger.completed[role]
                .checked_add(ledger.ancestors[role])
                .ok_or_else(|| invalid("learned ancestry update overflow"))?,
        );
    }
    let pc_steps = totals["proposer"]
        .checked_add(totals["critic"])
        .ok_or_else(|| invalid("P/C ancestry overflow"))?;
    check(
        totals.values().any(|n| *n > 0),
        "zero-step state is not learned",
    )?;
    let own_nnuec = |journal: &TrainerLedger| {
        journal
            .updates
            .iter()
            .filter(|update| {
                update["event"]["phase"] == "pc"
                    && update["event"]["role"] != "validator"
                    && update["event"]["head_counts"]["nnuec_wdl"]
                        .as_u64()
                        .is_some_and(|v| v > 0)
            })
            .count() as u64
    };
    let mut nnuec_steps = own_nnuec(ledger);
    let mut prior = chain_previous;
    let mut windows = 0;
    while let Some(owner) = prior {
        windows += 1;
        check(windows <= 1024, "NNUEC ancestry window capacity")?;
        nnuec_steps = nnuec_steps
            .checked_add(own_nnuec(&owner.journal))
            .ok_or_else(|| invalid("NNUEC update count overflow"))?;
        prior = owner.previous.as_deref();
    }
    let mut datasets = BTreeSet::new();
    if own_nnuec(ledger) > 0 {
        datasets.insert(ledger.corpus_sha256.clone());
    }
    let mut last_run = if own_nnuec(ledger) > 0 {
        Some(
            ctx.previous
                .as_ref()
                .ok_or_else(|| invalid("learned export original owner absent"))?
                .registration
                .execution_id
                .clone(),
        )
    } else {
        None
    };
    let mut prior = chain_previous;
    while let Some(owner) = prior {
        if own_nnuec(&owner.journal) > 0 {
            datasets.insert(owner.journal.corpus_sha256.clone());
            if last_run.is_none() {
                last_run = Some(owner.registration.execution_id.clone());
            }
        }
        prior = owner.previous.as_deref();
    }
    let dataset = if datasets.len() == 1 {
        datasets.first().expect("one actual NNUEC corpus").clone()
    } else {
        digest_value("rz-rovent-original-nnuec-datasets/1", &datasets)?
    };
    let nnuec = if nnuec_steps == 0 {
        json!({"kind":"untrained"})
    } else {
        json!({"kind":"learned","run_id":last_run.ok_or_else(||invalid("NNUEC update origin absent"))?,"steps":nnuec_steps,"dataset_sha256":dataset})
    };
    Ok(
        json!({"schema":"rz-rovent-original-learned-export-admission/1","recipe_sha256":ledger.window["recipe_sha256"],"stage_id":ledger.window["stage_id"],"checkpoint_sha256":checkpoint_sha,
        "source_sha256":ledger.window["source_sha256"],"runtime_sha256":ledger.window["runtime_sha256"],"model_semantics_sha256":ledger.window["model_semantics_sha256"],
        "training_profile_sha256":ledger.profile_sha256,"original_update_chain_sha256":original_chain,
        "retained_input_set_sha256":input["sha256"],"retained_input_count":input["count"],"retained_origin_set_sha256":origin["sha256"],"retained_origin_count":origin["count"],
        "lineage_teacher_actor_ids":lineage_teachers(&ledger.recipe,&ledger.window["stage_id"])? ,"completed_updates":totals,"pc_training_steps":pc_steps,
        "nnuec_training":nnuec,"post_snapshot_cost":original_post_cost,"original_native_lineage_verified":true}),
    )
}

pub(super) fn run(
    launch: RoventTrainingLaunch,
    corpus: Arc<OriginalRoventCorpusCustody>,
    previous: Option<Arc<VerifiedRoventTrainingWindow>>,
    cancel: Arc<AtomicBool>,
) -> Result<VerifiedRoventTrainingWindow, Box<RoventTrainingFailure>> {
    let mut process = None;
    let mut summary = None;
    let mut held = None;
    let mut pending_broker = None;
    let mut read_budget = Arc::new(Budget {
        maximum: MAX_JSON + 256 * (1 << 20),
        spent: Mutex::new(0),
        uncertain: AtomicBool::new(false),
    });
    let mut ledger_result = None;
    let mut ledger_slot = None;
    let shared_held = Arc::new(Mutex::new(None));
    let result = (|| -> Result<RoventTrainingRegistration, ArenaError> {
        launch.registration_pin.validate(MAX_JSON)?;
        launch.interpreter_pin.validate(256 << 20)?;
        check(
            launch.started < launch.execution_deadline
                && launch.execution_deadline < launch.whole_deadline
                && Instant::now() < launch.execution_deadline
                && launch.interpreter_version.starts_with("3.")
                && (1..=8).contains(&launch.dependency_directories.len())
                && (1..=64).contains(&launch.runtime_module_pins.len())
                && launch.limits.max_child_processes == 1
                && launch.limits.max_output_bytes <= 128 << 10,
            "original launch/runtime/process bounds",
        )?;
        let window = OriginalProcessWindow::new(
            launch.started,
            launch.execution_deadline,
            launch.whole_deadline,
        )?;
        let raw = read_budget.read(
            &launch.registration,
            &launch.registration_pin,
            true,
            launch.execution_deadline,
            &cancel,
        )?;
        let registration: RoventTrainingRegistration = parse(&raw)?;
        check(
            registration.schema == ROVENT_TRAINING_REGISTRATION
                && generation_identifier(&registration.execution_id)
                && matches!(registration.action.as_str(), "train" | "export")
                && ((registration.action == "export") == registration.export_request.is_some())
                && registration.max_read_bytes > 0
                && registration.max_read_bytes <= MAX_TRAINER_OUTPUT_BYTES * 260
                && registration.max_output_bytes > MAX_JSON
                && registration.max_output_bytes <= MAX_TRAINER_OUTPUT_BYTES
                && (1..=1_000_000).contains(&registration.max_rpc_messages)
                && (1..=1 << 30).contains(&registration.max_rpc_bytes)
                && registration.current_corpus.manifest.path
                    != registration.current_corpus.index.path
                && registration.ancestor_corpora.len() == corpus.ancestors.len(),
            "original Rovent registration budgets/corpus",
        )?;
        check(
            read_budget.charged() <= registration.max_read_bytes,
            "preparation read credit exceeds original registration",
        )?;
        let budget = Arc::new(Budget {
            maximum: registration.max_read_bytes,
            spent: Mutex::new(read_budget.charged()),
            uncertain: AtomicBool::new(false),
        });
        read_budget = Arc::clone(&budget);
        budget.read(
            &launch.interpreter,
            &launch.interpreter_pin,
            false,
            launch.execution_deadline,
            &cancel,
        )?;
        let output = empty_owned_directory(&registration.output_root)?;
        let control = empty_owned_directory(&registration.control_root)?;
        check(
            !registration
                .output_root
                .starts_with(&registration.control_root)
                && !registration
                    .control_root
                    .starts_with(&registration.output_root),
            "control and output roots must be separate owners",
        )?;
        ensure_child_directory(
            &registration.output_root,
            &registration.checkpoint_directory,
        )?;
        check(
            registration.receipt_path.parent() == Some(registration.output_root.as_path())
                && registration
                    .receipt_path
                    .extension()
                    .is_some_and(|s| s == "json"),
            "original receipt output path",
        )?;
        let control_limit = (registration.max_output_bytes / 8).min(64 << 20);
        held = Some(LinuxHeld {
            output,
            control,
            output_path: registration.output_root.clone(),
            control_path: registration.control_root.clone(),
            source: registration.output_root.join("source"),
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
        let owner = held.as_mut().expect("held original root");
        let recipe = load_json(
            &registration.recipe,
            owner,
            &budget,
            launch.execution_deadline,
            &cancel,
        )?;
        let membership = load_json(
            &registration.membership,
            owner,
            &budget,
            launch.execution_deadline,
            &cancel,
        )?;
        let mut spec = load_json(
            &registration.window_spec,
            owner,
            &budget,
            launch.execution_deadline,
            &cancel,
        )?;
        let _ = load_json(
            &registration.rules_profile,
            owner,
            &budget,
            launch.execution_deadline,
            &cancel,
        )?;
        let mut export_request = None;
        if let Some(pin) = &registration.export_request {
            let request = load_json(pin, owner, &budget, launch.execution_deadline, &cancel)?;
            fields(
                &request,
                &["schema", "kind", "directory", "max_output_bytes"],
            )?;
            let directory = PathBuf::from(
                request["directory"]
                    .as_str()
                    .ok_or_else(|| invalid("absolute export directory"))?,
            );
            check(
                request["schema"] == "rz-rovent-original-export-request/1"
                    && matches!(request["kind"].as_str(), Some("collection" | "deployment"))
                    && path_ok(&directory)
                    && directory.parent() == Some(registration.output_root.as_path())
                    && directory != registration.checkpoint_directory
                    && directory != owner.source
                    && fs::symlink_metadata(&directory)
                        .is_err_and(|error| error.kind() == std::io::ErrorKind::NotFound)
                    && count(&request["max_output_bytes"])? > 0
                    && count(&request["max_output_bytes"])?
                        <= registration.max_output_bytes - control_limit,
                "export request escapes the new bounded owner or changes declaration",
            )?;
            export_request = Some(request);
        }
        check(
            recipe["schema"] == "rz-rovent-generation-series/1"
                && digest_value("rz-rovent-generation-series/1", &recipe)?
                    == corpus.manifest.recipe_sha256
                && spec["recipe_sha256"] == corpus.manifest.recipe_sha256
                && spec["stage_id"] == corpus.manifest.stage_id
                && spec["source_sha256"] == corpus.manifest.source_sha256
                && spec["runtime_sha256"] == registration.python_runtime_sha256
                && spec["model_semantics_sha256"] == registration.model_semantics_sha256
                && recipe["model_semantics_sha256"] == spec["model_semantics_sha256"]
                && recipe["source_sha256"] == spec["source_sha256"]
                && membership.is_object(),
            "source recipe/window/native corpus/runtime/model mismatch",
        )?;
        validate_parent_window(&spec, &registration, &launch)?;
        check(
            serde_json::to_value(launch.resources.policy()).map_err(|e| invalid(e.to_string()))?
                == registration.resources_policy
                && registration.resources_policy["device"] == spec["device"],
            "configured resources differ from original declaration",
        )?;
        let total_scope = launch.resources.observe_total_parent_scope()?;
        check(
            !registration.require_total_quota || total_scope.verified,
            "strict total-resource profile lacks verified original parent-and-child enclosing cgroup caps",
        )?;
        validate_predecessor(&registration, &spec, &recipe, previous.as_deref())?;
        if registration.action == "export" {
            let original = previous
                .as_ref()
                .ok_or_else(|| invalid("export requires original previous window"))?;
            check(
                registration.checkpoint_mode == "resume"
                    && original.journal.completed == original.journal.stage_limits
                    && training_profile_digest(&spec)? == original.journal.profile_sha256
                    && corpus.manifest_sha256 == original._corpus.manifest_sha256,
                "export must preserve the complete original stage/profile/native corpus",
            )?;
        }
        let mut ancestor_indices: Vec<RoventCorpusIndex> = Vec::new();
        for (files, original) in registration.ancestor_corpora.iter().zip(&corpus.ancestors) {
            let dependencies = ancestor_indices
                .iter()
                .filter(|i| {
                    i.sha256()
                        .is_ok_and(|sha| original.manifest.ancestor_index_sha256.contains(&sha))
                })
                .cloned()
                .collect::<Vec<_>>();
            let index = load_corpus(
                files,
                original,
                owner,
                &budget,
                launch.execution_deadline,
                &cancel,
                &dependencies,
            )?;
            ancestor_indices.push(index);
        }
        load_corpus(
            &registration.current_corpus,
            &corpus,
            owner,
            &budget,
            launch.execution_deadline,
            &cancel,
            &ancestor_indices,
        )?;
        for pin in registration
            .checkpoint
            .iter()
            .chain(registration.pc_initial.iter())
            .chain(registration.nnuec_initial.iter())
            .chain(launch.runtime_module_pins.iter())
        {
            let (file, _) = own_file(
                pin,
                &budget,
                MAX_CHECKPOINT,
                false,
                launch.execution_deadline,
                &cancel,
            )?;
            owner.inputs.push(file);
        }
        let sources = install_sources(owner, &registration)?;
        let mut dependencies = Vec::new();
        for path in &launch.dependency_directories {
            check(path_ok(path), "registered dependency directory path")?;
            let directory = open_directory(path)?;
            let fd = directory_file(&directory)?;
            let metadata = fd
                .metadata()
                .map_err(|_| invalid("dependency directory metadata"))?;
            dependencies.push(json!({"path":path,"device":metadata.dev(),"inode":metadata.ino()}));
            owner
                .dependency_directories
                .push((fd, path.clone(), metadata.dev(), metadata.ino()));
        }
        let execution_id = registration.execution_id.clone();
        let window_id = spec["window_id"]
            .as_str()
            .filter(|s| generation_identifier(s))
            .ok_or_else(|| invalid("window ID"))?
            .to_owned();
        let sequence = NEXT_CONTROL
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |n| n.checked_add(1))
            .map_err(|_| invalid("control identity overflow"))?;
        let socket_path = registration
            .control_root
            .join(format!("trainer-{}-{sequence}.sock", std::process::id()));
        check(
            socket_path.as_os_str().len() < 100,
            "bounded local socket path",
        )?;
        let listener =
            UnixListener::bind(&socket_path).map_err(|_| invalid("original RPC socket bind"))?;
        let parent_start = proc_start(std::process::id())?;
        let control_meta = owner
            .control
            .metadata()
            .map_err(|_| invalid("control owner identity"))?;
        let (start_ns, execution_ns, whole_ns) = original_ticks(
            launch.started,
            launch.execution_deadline,
            launch.whole_deadline,
        )?;
        check(
            serde_json::to_value(launch.resources.policy()).map_err(|e| invalid(e.to_string()))?
                == registration.resources_policy
                && registration.resources_policy["device"] == spec["device"],
            "configured resources differ from original training registration",
        )?;
        let startup = json!({"schema":ROVENT_TRAINER_STARTUP,"execution_id":execution_id,"window_id":window_id,
            "registration_sha256":launch.registration_pin.sha256,"registration":registration,
            "control":{"socket_path":socket_path,"parent_pid":std::process::id(),"parent_start_ticks":parent_start,"control_device":control_meta.dev(),"control_inode":control_meta.ino()},
            "original_started_ns":start_ns,"execution_until_ns":execution_ns,"whole_until_ns":whole_ns,
            "source_root":owner.source,"package_root":owner.source.join(PACKAGE),"source_files":sources,
            "dependency_directories":dependencies,"runtime_module_pins":launch.runtime_module_pins,
            "python_runtime":{"version":launch.interpreter_version,"executable":launch.interpreter_pin},"resources":launch.resources.startup_json()?});
        let raw = serde_json::to_vec(&startup).map_err(|e| invalid(e.to_string()))?;
        check(
            raw.len() <= crate::MAX_ORIGINAL_INPUT_BYTES,
            "bounded original startup stdin",
        )?;
        let startup_sha256 = format!("{:x}", Sha256::digest(&raw));
        let startup_file = write_new(
            &registration.control_root.join("startup.json"),
            &raw,
            owner,
            true,
        )?;
        owner.journals.push(startup_file);
        let membership_domain = membership["schema"]
            .as_str()
            .filter(|d| {
                matches!(
                    *d,
                    "rz-rovent-trainable-parameter-membership/1"
                        | "rz-rovent-trainable-parameter-membership/2"
                )
            })
            .ok_or_else(|| invalid("new explicit trainable membership domain"))?;
        let membership_sha256 = digest_value(membership_domain, &membership)?;
        let mut ledger = TrainerLedger::new(
            spec.take(),
            recipe,
            membership_sha256,
            registration.current_corpus.manifest.sha256.clone(),
            corpus.rows.clone(),
            registration.expected_initial_completed.clone(),
            registration.expected_ancestor_completed.clone(),
        )?;
        ledger
            .consumed_inputs
            .extend(corpus.manifest.inherited_input_sha256.iter().cloned());
        ledger
            .consumed_origins
            .extend(corpus.manifest.inherited_origin_sha256.iter().cloned());
        if let Some(original) = previous.as_ref() {
            ledger
                .consumed_inputs
                .extend(original.journal.consumed_inputs.iter().cloned());
            ledger
                .consumed_origins
                .extend(original.journal.consumed_origins.iter().cloned());
        }
        let ledger = Arc::new(Mutex::new(ledger));
        ledger_slot = Some(Arc::clone(&ledger));
        *shared_held
            .lock()
            .map_err(|_| invalid("custody poisoned"))? = held.take();
        let owner = Arc::clone(&shared_held);
        let image_verified = Arc::new(AtomicBool::new(false));
        let stop = Arc::new(AtomicBool::new(false));
        let ctx = BrokerContext {
            listener,
            parent_pid: std::process::id(),
            parent_start,
            interpreter: launch.interpreter_pin.clone(),
            startup_sha256: startup_sha256.clone(),
            execution_id,
            window_id,
            until: launch.execution_deadline,
            whole_until: launch.whole_deadline,
            cancel: Arc::clone(&cancel),
            stop: Arc::clone(&stop),
            image_verified: Arc::clone(&image_verified),
            ledger: Arc::clone(&ledger),
            held: Arc::clone(&owner),
            budget: Arc::clone(&budget),
            registration: registration.clone(),
            max_messages: registration.max_rpc_messages,
            max_bytes: registration.max_rpc_bytes,
            resources: Arc::clone(&launch.resources),
            previous: previous.as_ref().map(Arc::clone),
        };
        let broker_cancel = Arc::clone(&cancel);
        let handle = thread::Builder::new()
            .name("rovent-original-trainer-rpc".into())
            .spawn(move || {
                let result = serve(ctx);
                if result.is_err() {
                    broker_cancel.store(true, Ordering::Release);
                }
                result
            })
            .map_err(|_| invalid("RPC worker spawn"))?;
        pending_broker = Some(PendingBroker { handle, stop });
        let args = [
            OsString::from("-I"),
            OsString::from("-S"),
            OsString::from("-B"),
            OsString::from("-c"),
            OsString::from(BOOTSTRAP),
            registration
                .output_root
                .join("source")
                .join(PACKAGE)
                .into_os_string(),
            OsString::from("rz_pals_model.rovent_training_child"),
            OsString::from("--execute"),
            OsString::from("--startup-sha256"),
            OsString::from(startup_sha256),
            OsString::from("--startup-bytes"),
            OsString::from(raw.len().to_string()),
            OsString::from("--execution-until-ns"),
            OsString::from(execution_ns.to_string()),
            OsString::from("--whole-until-ns"),
            OsString::from(whole_ns.to_string()),
        ];
        let mut verify_image = |image: &File| {
            budget
                .read(
                    image,
                    &launch.interpreter_pin,
                    false,
                    launch.execution_deadline,
                    &cancel,
                )
                .map(|_| {
                    image_verified.store(true, Ordering::Release);
                })
                .map_err(|_| ())
        };
        let cwd = owner
            .lock()
            .map_err(|_| invalid("owner poisoned"))?
            .as_ref()
            .ok_or_else(|| invalid("owner missing"))?
            .output
            .try_clone()
            .map_err(|_| invalid("output FD clone"))?;
        let output_limit = registration
            .max_output_bytes
            .checked_sub(control_limit)
            .ok_or_else(|| invalid("combined output reserve"))?;
        // The RPC broker only checks metadata and cannot execute GPU work.
        // Reserve after all external preflight, immediately before the first
        // GPU-capable supervisor invocation. Once that call is attempted, an
        // Err may follow a partial spawn: process=None is not release proof.
        {
            let mut slot = owner
                .lock()
                .map_err(|_| invalid("owner poisoned before device dispatch"))?;
            let held = slot
                .as_mut()
                .ok_or_else(|| invalid("owner missing before device dispatch"))?;
            check(
                held.device_admission.is_none(),
                "original device reservation already present before dispatch",
            )?;
            held.device_admission = ProcessDeviceAdmission::acquire_for_dispatch(
                registration.resources_policy["device"]
                    .as_str()
                    .ok_or_else(|| invalid("explicit training device"))?,
                &cancel,
                launch.execution_deadline,
                launch.whole_deadline,
            )?;
        }
        process = Some(supervise_numerical_input_in_original_window(
            &launch.interpreter,
            &args,
            &cwd,
            launch.limits,
            Some(&cancel),
            &OwnedArtifactTreeWatch::bounded_output(output_limit, MAX_CHECKPOINT),
            OriginalProcessInput {
                bytes: &raw,
                window,
            },
            &mut verify_image,
            OriginalNumericalEnvironment::Fp32Tf32Off,
        )?);
        if let Some(broker) = pending_broker.as_ref() {
            broker.stop.store(true, Ordering::Release);
        }
        while pending_broker
            .as_ref()
            .is_some_and(|b| !b.handle.is_finished())
            && Instant::now() < launch.whole_deadline
        {
            thread::park_timeout(Duration::from_millis(2));
        }
        if pending_broker
            .as_ref()
            .is_some_and(|b| b.handle.is_finished())
        {
            pending_broker
                .take()
                .expect("broker")
                .handle
                .join()
                .map_err(|_| invalid("original broker panic"))??;
        } else {
            return Err(invalid("original RPC owner join unknown"));
        }
        held = owner.lock().map_err(|_| invalid("owner poisoned"))?.take();
        let final_ledger = ledger
            .lock()
            .map_err(|_| invalid("ledger poisoned"))?
            .clone();
        ledger_result = Some(final_ledger.clone());
        let original = process.as_ref().expect("original process held");
        original.checked_loaded_image()?;
        original.checked_timing()?;
        check(
            original.transport_complete()
                && if registration.action == "train" {
                    final_ledger.ended
                } else {
                    final_ledger.ended
                        && final_ledger.checkpoint_checked
                        && final_ledger.export_admitted
                        && final_ledger.updates.is_empty()
                        && final_ledger.pending.is_empty()
                },
            "original child/RPC physical process closure incomplete",
        )?;
        let mut raw_summary: Value = parse(&original.process().stdout)?;
        summary = Some(raw_summary.clone());
        let receipt_pin: RoventTrainingFilePin =
            serde_json::from_value(raw_summary["receipt"].clone())
                .map_err(|e| invalid(e.to_string()))?;
        check(
            receipt_pin.path == registration.receipt_path,
            "terminal receipt escaped original registered path",
        )?;
        let (terminal_file, terminal_raw) = own_file(
            &receipt_pin,
            &budget,
            MAX_JSON,
            true,
            launch.whole_deadline,
            &AtomicBool::new(false),
        )?;
        held.as_mut()
            .ok_or_else(|| invalid("terminal original custody absent"))?
            .journals
            .push(terminal_file);
        let terminal: Value = parse(&terminal_raw)?;
        check(
            raw_summary["schema"] == ROVENT_TRAINER_RESULT
                && raw_summary["window_id"] == final_ledger.window["window_id"]
                && raw_summary["execution_id"] == registration.execution_id
                && raw_summary["completed_updates"] == json!(final_ledger.completed)
                && raw_summary["physical_completion"] == "confirmed"
                && raw_summary["arena_eligible"] == false
                && raw_summary["status"] == "completed"
                && terminal["schema"] == ROVENT_TRAINER_RESULT
                && terminal["execution_id"] == registration.execution_id
                && terminal["window_id"] == final_ledger.window["window_id"]
                && terminal["status"] == "completed"
                && terminal["completed_updates"] == json!(final_ledger.completed)
                && terminal["physical_completion"] == "confirmed"
                && terminal["arena_eligible"] == false,
            "original trainer terminal summary mismatch",
        )?;
        if let Some(request) = export_request.as_ref() {
            verify_export_outputs(
                &registration,
                request,
                &terminal,
                &final_ledger,
                held.as_mut()
                    .ok_or_else(|| invalid("export custody absent"))?,
                &budget,
                launch.whole_deadline,
            )?;
        }
        raw_summary["parent_device_admission"] = json!({"scope":"process_local_logical_device_only","host_global_exclusivity":"unknown","physical_sharing":"unknown"});
        raw_summary["resource_readback"] = serde_json::to_value(launch.resources.observe_closed()?)
            .map_err(|e| invalid(e.to_string()))?;
        let total_after = launch.resources.observe_total_parent_scope()?;
        check(
            !registration.require_total_quota || total_after.verified,
            "strict total-resource enclosing cgroup evidence changed or became unknown",
        )?;
        raw_summary["total_resource_scope"] = json!({"required":registration.require_total_quota,"before":total_scope,"after":total_after,
            "controlled_performance_eligible":false,"scope":"source_validation_and_bounded_observations_not_actual_training_or_performance_acceptance"});
        summary = Some(raw_summary);
        let preservation_cancel = AtomicBool::new(false);
        for (file, pin) in [
            (&launch.registration, &launch.registration_pin),
            (&launch.interpreter, &launch.interpreter_pin),
        ] {
            budget.read(
                file,
                pin,
                false,
                launch.whole_deadline,
                &preservation_cancel,
            )?;
        }
        if let Some(h) = held.as_ref() {
            for input in h
                .inputs
                .iter()
                .chain(h.sources.iter())
                .chain(h.checkpoints.iter())
                .chain(h.journals.iter())
                .chain(h.artifacts.iter())
            {
                check(
                    input.identity
                        == FileIdentity::from(
                            &input
                                .file
                                .metadata()
                                .map_err(|_| invalid("post child input metadata"))?,
                        ),
                    "original source/input inode changed",
                )?;
                budget.read(
                    &input.file,
                    &input.pin,
                    false,
                    launch.whole_deadline,
                    &preservation_cancel,
                )?;
            }
            for (fd, path, device, inode) in &h.dependency_directories {
                let metadata = fd
                    .metadata()
                    .map_err(|_| invalid("held dependency metadata"))?;
                let named = directory_file(&open_directory(path)?)?;
                let named = named
                    .metadata()
                    .map_err(|_| invalid("named dependency metadata"))?;
                check(
                    metadata.dev() == *device
                        && metadata.ino() == *inode
                        && named.dev() == *device
                        && named.ino() == *inode,
                    "dependency directory identity changed",
                )?;
            }
        }
        check(
            Instant::now() < launch.whole_deadline,
            "original post-closure preservation deadline",
        )?;
        if let Some(admission) = held.as_mut().and_then(|h| h.device_admission.as_mut()) {
            admission.release_confirmed()?;
        }
        Ok(registration)
    })();
    match result {
        Ok(registration) => Ok(VerifiedRoventTrainingWindow {
            registration,
            summary: summary.expect("validated terminal"),
            journal: ledger_result.expect("validated ledger"),
            process: process.expect("closed original"),
            _launch: launch,
            _corpus: corpus,
            previous,
            held: held.expect("held original descriptors"),
        }),
        Err(error) => {
            if let Some(broker) = pending_broker.as_ref() {
                broker.stop.store(true, Ordering::Release);
            }
            let journal = ledger_result
                .as_ref()
                .map(TrainerLedger::summary)
                .or_else(|| {
                    ledger_slot
                        .as_ref()
                        .and_then(|l| l.try_lock().ok().map(|v| v.summary()))
                })
                .unwrap_or_else(|| json!({"state":"preserved_partial_or_unknown"}));
            Err(Box::new(RoventTrainingFailure {
                error,
                process,
                summary,
                charged_reads: read_budget.charged(),
                read_accounting_unknown: read_budget.unknown(),
                rpc_journal: journal,
                launch,
                corpus,
                previous,
                _retained_held: held,
                _retained_pending_broker: pending_broker,
                _retained_shared_held: Some(shared_held),
            }))
        }
    }
}

fn verify_export_outputs(
    registration: &RoventTrainingRegistration,
    request: &Value,
    terminal: &Value,
    ledger: &TrainerLedger,
    owner: &mut LinuxHeld,
    budget: &Budget,
    until: Instant,
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
        "export receipt escaped registered directory",
    )?;
    let cancel = AtomicBool::new(false);
    let (held, raw) = own_file(&pin, budget, MAX_JSON, true, until, &cancel)?;
    owner.artifacts.push(held);
    let receipt: Value = parse(&raw)?;
    check(
        terminal["result"] == receipt
            && receipt["schema"] == "rz-rovent-tensor-lifecycle-export/1"
            && receipt["kind"] == request["kind"]
            && receipt["recipe_sha256"] == ledger.window["recipe_sha256"]
            && receipt["stage_id"] == ledger.window["stage_id"]
            && receipt["checkpoint"]["sha256"]
                == registration
                    .checkpoint
                    .as_ref()
                    .ok_or_else(|| invalid("export checkpoint"))?
                    .sha256
            && receipt["original_training_admission"]
                == ledger
                    .export_admission
                    .clone()
                    .ok_or_else(|| invalid("original learned export admission absent"))?
            && receipt["actual_export_written"] == true
            && receipt["training_provenance_verified"] == true
            && receipt["optimizer_updates_executed"] == 0
            && receipt["numeric_status"] == "not_run"
            && receipt["cuda_status"] == "not_run"
            && receipt["arena_eligible"] == false,
        "export writer receipt changed original learned source or promoted numerical/GPU status",
    )?;
    let roles = receipt["pc"]["roles"]
        .as_array()
        .ok_or_else(|| invalid("export roles"))?;
    let deployment = request["kind"] == "deployment";
    check(
        (roles == &vec![json!("proposer"), json!("critic")]
            && receipt["pc"]["validator_present"] == false)
            || (!deployment
                && roles == &vec![json!("proposer"), json!("critic"), json!("validator")]
                && receipt["pc"]["validator_present"] == true),
        "deployment export must remain V-free",
    )?;
    let graphs = receipt["pc"]["graphs"]
        .as_array()
        .filter(|v| (2..=3).contains(&v.len()))
        .ok_or_else(|| invalid("bounded original export graphs"))?;
    let mut paths = BTreeSet::from([pin.path.clone()]);
    let mut total = pin.bytes;
    for item in std::iter::once(&receipt["pc"]["manifest"])
        .chain(graphs.iter())
        .chain(std::iter::once(&receipt["nnuec"]))
    {
        let relative = Path::new(
            item["file"]
                .as_str()
                .ok_or_else(|| invalid("export file name"))?,
        );
        check(
            !relative.is_absolute()
                && relative
                    .components()
                    .all(|component| matches!(component, Component::Normal(_))),
            "export file path escaped original directory",
        )?;
        let file = RoventTrainingFilePin {
            path: directory.join(relative),
            sha256: sha(&item["sha256"])?.into(),
            bytes: count(&item["bytes"])?,
        };
        check(
            paths.insert(file.path.clone()),
            "duplicate original export artifact",
        )?;
        total = total
            .checked_add(file.bytes)
            .ok_or_else(|| invalid("export output byte overflow"))?;
        check(
            total <= count(&request["max_output_bytes"])?,
            "export artifacts exceed registered output quota",
        )?;
        let (retained, _) = own_file(&file, budget, MAX_CHECKPOINT, false, until, &cancel)?;
        owner.artifacts.push(retained);
    }
    Ok(())
}
fn validate_parent_window(
    spec: &Value,
    registration: &RoventTrainingRegistration,
    launch: &RoventTrainingLaunch,
) -> Result<(), ArenaError> {
    let device = spec["device"]
        .as_str()
        .ok_or_else(|| invalid("canonical training device"))?;
    let cuda = device
        .strip_prefix("cuda:")
        .is_some_and(|n| n.parse::<u8>().is_ok_and(|i| i < 16 && i.to_string() == n));
    check(
        (device == "cpu" || cuda)
            && spec["precision"] == "fp32"
            && spec["tf32"] == false
            && spec["deterministic"] == true
            && spec["microbatch"] == 1
            && spec["accumulation"] == 8
            && matches!(spec["phase"].as_str(), Some("pc" | "v")),
        "registered explicit FP32/TF32-off/device/phase/accumulation required",
    )?;
    for (name, maximum) in [
        ("max_updates", 65_536),
        ("max_groups", 65_536),
        ("max_samples", 524_288),
        ("max_wall_ns", 86_400_000_000_000),
        ("checkpoint_interval", 250),
    ] {
        let n = count(&spec[name])?;
        check(
            n > 0 && n <= maximum,
            "finite positive training window work/time limit",
        )?;
    }
    check(
        count(&spec["max_groups"])? >= count(&spec["max_updates"])?
            && count(&spec["max_samples"])? >= 8
            && count(&spec["max_output_bytes"])? > 0
            && count(&spec["max_output_bytes"])?
                <= registration.max_output_bytes
                    - (registration.max_output_bytes / 8).min(64 << 20),
        "training window cannot fit original group/sample/output reserve",
    )?;
    let elapsed = launch
        .execution_deadline
        .checked_duration_since(launch.started)
        .ok_or_else(|| invalid("original execution duration"))?
        .as_nanos();
    check(
        u128::from(count(&spec["max_wall_ns"])?) <= elapsed,
        "window cannot extend the original process execution deadline",
    )?;
    for name in ["cpu_threads", "interop_threads"] {
        let n = count(&spec[name])?;
        check(
            (1..=64).contains(&n),
            "bounded explicit Torch host thread count",
        )?;
    }
    let requested = count(&spec["cpu_threads"])?
        .checked_add(count(&spec["interop_threads"])?)
        .ok_or_else(|| invalid("thread sum overflow"))?;
    check(
        requested <= u64::from(launch.resources.policy().cpu_quota_millicpus).div_ceil(1000),
        "declared training host pools exceed child CPU quota",
    )?;
    Ok(())
}
fn validate_predecessor(
    registration: &RoventTrainingRegistration,
    spec: &Value,
    recipe: &Value,
    previous: Option<&VerifiedRoventTrainingWindow>,
) -> Result<(), ArenaError> {
    match registration.checkpoint_mode.as_str() {
        "initialize" => check(
            previous.is_none()
                && registration.checkpoint.is_none()
                && registration.pc_initial.is_some()
                && registration.nnuec_initial.is_some()
                && registration
                    .expected_initial_completed
                    .values()
                    .chain(registration.expected_ancestor_completed.values())
                    .all(|v| *v == 0),
            "fresh initialize cannot carry learned/resume ownership",
        ),
        "resume" | "fork" => {
            let previous = previous
                .ok_or_else(|| invalid("resume/fork needs original predecessor process custody"))?;
            let pin = registration
                .checkpoint
                .as_ref()
                .ok_or_else(|| invalid("resume/fork checkpoint missing"))?;
            check(
                registration.pc_initial.is_none()
                    && registration.nnuec_initial.is_none()
                    && previous.durable_checkpoint_available()
                    && previous
                        .held
                        .checkpoints
                        .iter()
                        .any(|file| file.pin == *pin)
                    && previous.journal.checkpoint_states.iter().any(|state| {
                        state["checkpoint"] == json!(pin)
                            && state["completed_updates"] == json!(previous.journal.completed)
                    })
                    && previous.registration.execution_id != registration.execution_id,
                // Selected checkpoint, not merely some prior checkpoint, must
                // cover the actual completed update boundary before spawn.
                "resume/fork checkpoint was not published by original predecessor",
            )?;
            check(
                spec["recipe_sha256"] == previous.journal.window["recipe_sha256"]
                    && spec["source_sha256"] == previous.journal.window["source_sha256"]
                    && spec["runtime_sha256"] == previous.journal.window["runtime_sha256"]
                    && spec["model_semantics_sha256"]
                        == previous.journal.window["model_semantics_sha256"],
                "resume/fork changes original recipe/source/runtime/model semantics",
            )?;
            if registration.checkpoint_mode == "resume" {
                check(
                    spec["stage_id"] == previous.journal.window["stage_id"]
                        && registration.expected_initial_completed == previous.journal.completed
                        && registration.expected_ancestor_completed == previous.journal.ancestors,
                    "same-stage exact resume ownership",
                )
            } else {
                let stage = recipe["stages"]
                    .as_array()
                    .and_then(|s| s.iter().find(|s| s["stage_id"] == spec["stage_id"]))
                    .ok_or_else(|| invalid("fork stage"))?;
                check(
                    previous.journal.completed == previous.journal.stage_limits
                        && stage["parent_stage_id"] == previous.journal.window["stage_id"]
                        && registration
                            .expected_initial_completed
                            .values()
                            .all(|v| *v == 0)
                        && ROLES.iter().all(|r| {
                            registration.expected_ancestor_completed.get(*r)
                                == previous.journal.completed[*r]
                                    .checked_add(previous.journal.ancestors[*r])
                                    .as_ref()
                        }),
                    "fork must use direct original parent, reset current state and preserve ancestry",
                )
            }
        }
        _ => Err(invalid("unsupported original checkpoint mode")),
    }
}

#[cfg(test)]
mod process_device_admission_tests {
    use super::*;

    static TEST_SLOTS: Mutex<()> = Mutex::new(());

    // These are private logical slots, never CUDA devices or runtime probes.
    #[test]
    fn cancelled_or_expired_dispatch_does_not_reserve_a_device() {
        let _serial = TEST_SLOTS.lock().unwrap();
        let cases = [
            ("test:rovent-training-cancelled", true, false, false),
            ("test:rovent-training-execution-expired", false, true, false),
            ("test:rovent-training-whole-expired", false, false, true),
        ];
        for (device, cancelled, execution_expired, whole_expired) in cases {
            let now = Instant::now();
            let future = now + Duration::from_secs(60);
            assert!(
                ProcessDeviceAdmission::acquire_for_dispatch(
                    device,
                    &AtomicBool::new(cancelled),
                    if execution_expired { now } else { future },
                    if whole_expired { now } else { future },
                )
                .is_err()
            );
            let mut available = ProcessDeviceAdmission::acquire(device)
                .unwrap()
                .expect("never-dispatched slot must remain available");
            available.release_confirmed().unwrap();
        }
    }

    #[test]
    fn attempted_dispatch_keeps_unknown_reservation_after_owner_drop() {
        let _serial = TEST_SLOTS.lock().unwrap();
        let device = "test:rovent-training-attempted-unknown";
        {
            let until = Instant::now() + Duration::from_secs(60);
            let _reservation = ProcessDeviceAdmission::acquire_for_dispatch(
                device,
                &AtomicBool::new(false),
                until,
                until,
            )
            .unwrap()
            .expect("logical dispatch reservation");
            assert!(ProcessDeviceAdmission::acquire(device).is_err());
        }
        assert!(ProcessDeviceAdmission::acquire(device).is_err());
    }

    #[test]
    fn only_confirmed_release_reopens_a_dispatched_slot() {
        let _serial = TEST_SLOTS.lock().unwrap();
        let device = "test:rovent-training-confirmed-release";
        let until = Instant::now() + Duration::from_secs(60);
        let mut reservation = ProcessDeviceAdmission::acquire_for_dispatch(
            device,
            &AtomicBool::new(false),
            until,
            until,
        )
        .unwrap()
        .unwrap();
        assert!(ProcessDeviceAdmission::acquire(device).is_err());
        reservation.release_confirmed().unwrap();
        assert!(reservation.release_confirmed().is_err());
        let mut next = ProcessDeviceAdmission::acquire(device).unwrap().unwrap();
        next.release_confirmed().unwrap();
        assert!(ProcessDeviceAdmission::acquire("cpu").unwrap().is_none());
    }
}
