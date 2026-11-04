//! Minimal library facade for an LC0-specific authenticated supervisor.
//!
//! The caller captures a ticket before publishing any endpoint, accepts a
//! connection from the original parent, and retains the resulting session and
//! process whenever completion is unknown. The legacy CLI guard remains in
//! place until a real parent/listener owner and termination path use this API.

use super::{
    control::Lc0ControlSupervisor,
    endpoint::Lc0ParentEndpoint,
    profile::{Lc0ProfileCreateError, Lc0ProfileDirectory},
    projection::{Lc0OriginalSessionObservation, PublicationObservations},
    *,
};
use std::{
    fs::File,
    io::Write,
    os::unix::{fs::MetadataExt, net::UnixListener},
};

use crate::{
    engine::{
        EngineProcess, PriorityCloseHandle, ProcessClock,
        async_engine::{
            AsyncEngineRun, AsyncEngineStartFailure, AsyncEngineStartSlot, EngineTimerJoinStatus,
            EngineWriterRecoveryStatus, InPlacePreflightStatus, InPlaceTimerStatus,
            InPlaceTransferStatus, UnstartedWriterCloseStatus, UnstartedWriterRecoveryStatus,
        },
    },
    native_profile::ProfiledWrite,
    runner::{
        Event,
        async_events::AsyncServeStatus,
        async_output::{AsyncOutputOwner, OutputRecovery},
    },
};
use std::sync::mpsc::{Receiver, SyncSender};

type Lc0Output = AsyncOutputOwner<ProfiledWrite<std::io::Stdout>, std::io::Stderr>;
type Lc0Run = AsyncEngineRun<ProfiledWrite<std::io::Stdout>, std::io::Stderr>;
type Lc0StartFailure = AsyncEngineStartFailure<ProfiledWrite<std::io::Stdout>, std::io::Stderr>;
type Lc0StartSlot = AsyncEngineStartSlot<ProfiledWrite<std::io::Stdout>, std::io::Stderr>;

/// Inputs transferred only after this original supervisor accepts admission.
/// The caller retains its original parent, listener, receipt owner and process.
#[must_use]
pub struct Lc0AsyncInputs {
    pub events: Receiver<Event>,
    pub sender: SyncSender<Event>,
    pub output: Lc0Output,
    pub max_output_bytes: usize,
    pub poll_interval: std::time::Duration,
}

enum Lc0AsyncCustody {
    // Installed before inputs move into the fallible engine start operation.
    // An unwind here is unknown, never a completed service or a retry grant.
    Starting,
    StartPanicked(Box<dyn std::any::Any + Send>),
    CloseSetupFailed {
        _process: EngineProcess,
        _inputs: Lc0AsyncInputs,
        _error: rz_contracts::ContractError,
    },
    StartFailed(Lc0StartFailure),
    Slot {
        slot: Lc0StartSlot,
        close: PriorityCloseHandle,
        stage: Lc0SlotStage,
    },
    Running {
        run: Lc0Run,
        close: PriorityCloseHandle,
    },
}

#[derive(Clone, Copy, Debug)]
enum Lc0SlotStage {
    Created,
    Preflight(InPlacePreflightStatus),
    Timer(InPlaceTimerStatus),
    Install(InPlaceTransferStatus),
    Running,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Lc0AsyncAdmission {
    Started,
    RetainedFailure,
    Unknown,
}

/// CPU, writer and native collection are separate observations. None of
/// these variants establishes device-wide physical completion.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Lc0AsyncFinishStatus {
    NoRun,
    ServicePending,
    ServiceFailed,
    CpuPending,
    CpuFailed,
    WriterPending,
    WriterFailed,
    WriterUnknown,
    NativeCollected,
    NativeCollectionFailed,
    AlreadyAttempted,
}

/// Exercise original parent and receipt plumbing without opening model assets,
/// loading ORT, creating a CUDA session, or granting runtime authorization.
/// The caller must already own the configured managed-output lease.
pub fn parent_preflight(
    config: &NativeConfig,
    epoch: u64,
    until: Instant,
) -> Result<(), NativeBootstrapError> {
    let cleanup_until = until
        .checked_add(std::time::Duration::from_secs(30))
        .ok_or(NativeBootstrapError::Config(
            "LC0 preflight cleanup deadline overflow",
        ))?;
    let pending = Lc0PendingSupervisor::capture(config, epoch, until)?;
    let receipts = Lc0ReceiptOwner::open_for_config(config)?;
    let mut listener = pending.prepare_parent_listener(&receipts).map_err(|_| {
        NativeBootstrapError::Config("LC0 parent preflight listener preparation failed")
    })?;
    if listener.bind(until).is_err() {
        if listener.endpoint.cleanup(cleanup_until).is_err() {
            eprintln!(
                "LC0 preflight endpoint cleanup unknown after bind failure; native work did not start"
            );
        }
        return Err(NativeBootstrapError::Config(
            "LC0 parent preflight bind failed",
        ));
    }
    let outcome = (|| {
        let notice = pending.admission_notice(&listener)?;
        let mut stderr = std::io::stderr().lock();
        writeln!(stderr, "{notice}").map_err(|_| {
            NativeBootstrapError::Config("LC0 parent preflight notice could not be written")
        })?;
        drop(stderr);
        let stream = listener
            .accept_parent(until.min(pending.control_until))
            .map_err(|_| NativeBootstrapError::Config("LC0 parent preflight accept failed"))?;
        let parent = pending
            .pending
            .authenticate(stream, until.min(pending.control_until))
            .map_err(|_| {
                NativeBootstrapError::Config("LC0 parent preflight authentication failed")
            })?;
        parent
            .ensure_live()
            .map_err(|_| NativeBootstrapError::Config("LC0 parent preflight peer disconnected"))?;
        let observation = serde_json::json!({
            "domain": "rz-lc0-parent-preflight/1", "process_id": std::process::id(),
            "process_epoch": pending.epoch, "lock_sha256": pending.lock.canonical_sha256,
            "parent_authenticated": true, "model_opened": false, "native_started": false,
            "physical_completion": "not_applicable_to_this_preflight"
        });
        let bytes = serde_json::to_vec(&observation).map_err(|_| {
            NativeBootstrapError::Config("LC0 parent preflight observation encoding failed")
        })?;
        if receipts
            .writer
            .write_bytes("lc0-parent-preflight.v1.json", &bytes)
            .is_err()
        {
            eprintln!("LC0 parent preflight receipt unavailable; native work did not start");
        }
        Ok(())
    })();
    let cleanup = listener.endpoint.cleanup(cleanup_until);
    match (outcome, cleanup) {
        (Err(error), _) => Err(error),
        (Ok(()), Err(_)) => Err(NativeBootstrapError::Config(
            "LC0 preflight endpoint cleanup is unconfirmed",
        )),
        (Ok(()), Ok(())) => Ok(()),
    }
}

/// Original private output capability; creation alone starts no native work.
pub struct Lc0ReceiptOwner {
    writer: SystemReceiptWriter,
    owner_token: Arc<()>,
    root_identity: (u64, u64, u32),
}

/// Owns the original private listener and socket inode. Dropping this value
/// does not unlink the endpoint; cleanup needs a separate evidence gate.
#[must_use]
pub struct Lc0ParentListener {
    endpoint: Lc0ParentEndpoint,
    ticket_token: Arc<()>,
    receipt_token: Arc<()>,
}

impl Lc0ParentListener {
    /// A failure leaves this exact endpoint owner with the caller.
    pub fn bind(&mut self, until: Instant) -> io::Result<()> {
        self.endpoint.bind(until)
    }

    /// The route is meaningful only while this process keeps the endpoint FD.
    pub fn route(&self) -> std::path::PathBuf {
        self.endpoint.route()
    }

    pub fn accept_parent(&self, until: Instant) -> io::Result<UnixStream> {
        self.endpoint.accept_parent(until)
    }

    /// Retire a bind attempt only while the original pre-authentication
    /// ticket is still held. A failed cleanup remains in this endpoint owner;
    /// it is not a native-session or physical-completion observation.
    pub fn retire_before_auth(
        &mut self,
        pending: &Lc0PendingSupervisor,
        until: Instant,
    ) -> io::Result<()> {
        if !Arc::ptr_eq(&self.ticket_token, &pending.ticket_token) {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "LC0 listener belongs to another pending ticket",
            ));
        }
        self.endpoint.cleanup(until)
    }
}

impl Lc0ReceiptOwner {
    /// The caller owns and validates the parent output directory and lease.
    /// The writer exclusively creates its per-process leaf without following
    /// links and retains that directory for every subsequent publication.
    pub fn open(root: &cap_std::fs::Dir, relative: &std::path::Path) -> io::Result<Self> {
        use nix::{
            fcntl::{OFlag, openat},
            sys::stat::{Mode, fchmod},
            unistd::geteuid,
        };
        let root_metadata = root.try_clone()?.into_std_file().metadata()?;
        let root_identity = (
            root_metadata.dev(),
            root_metadata.ino(),
            root_metadata.uid(),
        );
        let writer = SystemReceiptWriter::open(root, relative)?;
        let original = writer.original_directory().try_clone()?.into_std_file();
        let metadata = original.metadata()?;
        if !metadata.is_dir() || metadata.uid() != geteuid().as_raw() {
            return Err(io::Error::other("LC0 receipt directory is not user owned"));
        }
        let reopened = File::from(
            openat(
                &original,
                ".",
                OFlag::O_RDONLY | OFlag::O_DIRECTORY | OFlag::O_NOFOLLOW | OFlag::O_CLOEXEC,
                Mode::empty(),
            )
            .map_err(io::Error::other)?,
        );
        let observed = reopened.metadata()?;
        if !observed.is_dir()
            || observed.dev() != metadata.dev()
            || observed.ino() != metadata.ino()
            || observed.uid() != metadata.uid()
        {
            return Err(io::Error::other("LC0 receipt directory identity changed"));
        }
        fchmod(&reopened, Mode::from_bits_truncate(0o700)).map_err(io::Error::other)?;
        Ok(Self {
            writer,
            owner_token: Arc::new(()),
            root_identity,
        })
    }

    /// The caller is responsible for the original managed output lease. This
    /// opens only the explicitly configured private output root.
    pub fn open_for_config(config: &NativeConfig) -> Result<Self, NativeBootstrapError> {
        let path = super::super::private_output_root(&config.output_root)?;
        let root = cap_std::fs::Dir::open_ambient_dir(path, cap_std::ambient_authority()).map_err(
            |_| NativeBootstrapError::Config("LC0 private receipt root cannot be opened"),
        )?;
        Self::open(&root, std::path::Path::new("")).map_err(|_| {
            NativeBootstrapError::Config("LC0 private receipt owner cannot be created")
        })
    }
}

/// Captured before endpoint publication. This contains no loaded ORT runtime.
pub struct Lc0PendingSupervisor {
    pending: Lc0PendingParent,
    lock: SystemRuntimeLockV6,
    config: NativeConfig,
    epoch: u64,
    control_until: Instant,
    ticket_token: Arc<()>,
}

/// Pre-native failure keeps the same pending ticket and, if authentication
/// completed, the same authenticated parent connection. An authentication
/// error closes its accepted stream and preserves the original I/O error.
pub enum Lc0PendingFailureCause {
    TicketMismatch,
    Accept(io::Error),
    Authentication(io::Error),
    Asset(NativeBootstrapError),
    Custody(NativeBootstrapError),
}

#[must_use]
pub struct Lc0PendingFailure {
    pending: Lc0PendingSupervisor,
    cause: Lc0PendingFailureCause,
    authenticated_parent: Option<Arc<ConnectedParent>>,
}

impl std::fmt::Debug for Lc0PendingFailure {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("Lc0PendingFailure(pre-native; original error retained)")
    }
}

impl std::fmt::Display for Lc0PendingFailure {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("LC0 parent admission failed before native entry")
    }
}

impl std::error::Error for Lc0PendingFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match &self.cause {
            Lc0PendingFailureCause::TicketMismatch => None,
            Lc0PendingFailureCause::Accept(error)
            | Lc0PendingFailureCause::Authentication(error) => Some(error),
            Lc0PendingFailureCause::Asset(error) | Lc0PendingFailureCause::Custody(error) => {
                Some(error)
            }
        }
    }
}

impl Lc0PendingFailure {
    pub fn cause(&self) -> &Lc0PendingFailureCause {
        &self.cause
    }

    pub fn parent_authenticated(&self) -> bool {
        self.authenticated_parent.is_some()
    }

    /// Operates on the original listener and ticket. On failure the original
    /// failure object and listener remain owned by the caller.
    pub fn retire_listener_before_native(
        &self,
        listener: &mut Lc0ParentListener,
        until: Instant,
    ) -> io::Result<()> {
        listener.retire_before_auth(&self.pending, until)
    }
}

/// Owns the original authenticated parent, startup custody and worker.
/// This is not an OS-process lifetime manager; dropping it or exiting on an
/// unknown physical result is not safe. No resource is copied to a cache.
#[must_use]
pub struct Lc0SupervisedSession {
    supervisor: Lc0ControlSupervisor,
    async_custody: Option<Lc0AsyncCustody>,
    slot_start_panic: Option<Box<dyn std::any::Any + Send>>,
    async_finish_attempted: bool,
    lock: SystemRuntimeLockV6,
    config: NativeConfig,
    owners: Arc<OwnerRegistry>,
    ticket_token: Arc<()>,
    receipt_token: Option<Arc<()>>,
    profile: Option<Lc0ProfileDirectory>,
    profile_failure: Option<Lc0ProfileCreateError>,
    profile_verification_failure: Option<BackendError>,
    endpoint_retirement_started: bool,
    epoch: u64,
    publications: PublicationObservations,
}

impl Lc0PendingSupervisor {
    /// Read the bounded, explicitly pinned V6 lock and capture this process's
    /// actual parent before the caller creates or publishes an endpoint.
    pub fn capture(
        config: &NativeConfig,
        epoch: u64,
        control_until: Instant,
    ) -> Result<Self, NativeBootstrapError> {
        let path = config
            .system_cuda_lock
            .as_ref()
            .ok_or(NativeBootstrapError::Config(
                "LC0 system CUDA lock path is absent",
            ))?;
        let bytes = rz_eval::asset::read_bounded(path, 256 * 1024)?;
        if Some(rz_eval::asset::sha256(&bytes)) != config.system_cuda_lock_sha256 {
            return Err(NativeBootstrapError::Config(
                "LC0 system CUDA lock file digest differs",
            ));
        }
        let text = std::str::from_utf8(&bytes)
            .map_err(|_| NativeBootstrapError::Config("LC0 system CUDA lock is not UTF-8"))?;
        let lock = SystemRuntimeLockV6::from_json(text)
            .map_err(|_| NativeBootstrapError::Config("LC0 system CUDA lock is invalid"))?;
        if control_until <= Instant::now() || config.parallelism != 1 {
            return Err(NativeBootstrapError::Config(
                "LC0 system control window or B1 is invalid",
            ));
        }
        let pending = Lc0PendingParent::capture(epoch, &lock)?;
        Ok(Self {
            pending,
            lock,
            config: config.clone(),
            epoch,
            control_until,
            ticket_token: Arc::new(()),
        })
    }

    /// Install the listener owner before binding or publishing its route.
    /// Authentication on the accepted stream is a separate mandatory step.
    pub fn prepare_parent_listener(
        &self,
        receipts: &Lc0ReceiptOwner,
    ) -> io::Result<Lc0ParentListener> {
        let root = super::super::private_output_root(&self.config.output_root)
            .map_err(|_| io::Error::other("LC0 configured output root is invalid"))?;
        let metadata = std::fs::metadata(root)?;
        if (metadata.dev(), metadata.ino(), metadata.uid()) != receipts.root_identity {
            return Err(io::Error::other(
                "LC0 receipt root differs from configured output root",
            ));
        }
        Ok(Lc0ParentListener {
            endpoint: Lc0ParentEndpoint::prepare(
                receipts.writer.original_directory(),
                self.pending.parent_pid(),
            )?,
            ticket_token: Arc::clone(&self.ticket_token),
            receipt_token: Arc::clone(&receipts.owner_token),
        })
    }

    /// Exact registered identifiers for the launcher's parent handshake; this
    /// observes no CUDA provider, inference, or physical completion.
    pub fn admission_notice(
        &self,
        listener: &Lc0ParentListener,
    ) -> Result<serde_json::Value, NativeBootstrapError> {
        if !Arc::ptr_eq(&self.ticket_token, &listener.ticket_token) {
            return Err(NativeBootstrapError::Config(
                "LC0 listener belongs to another ticket",
            ));
        }
        let ordinal = match &self.lock.runtime {
            NativeRuntimeSourceV6::SystemCuda(spec) => spec.cuda_device_id,
            _ => return Err(NativeBootstrapError::Config("LC0 admission requires CUDA")),
        };
        Ok(serde_json::json!({
            "schema": "rz-lc0-parent-endpoint/1",
            "process_id": std::process::id(), "process_epoch": self.epoch,
            "lock_sha256": self.lock.canonical_sha256, "cuda_ordinal": ordinal,
            "control_path": listener.route(), "native_started": false,
        }))
    }

    /// Authenticate only a connection accepted by the listener bound from
    /// this same pending ticket; no native operation has started on failure.
    pub fn authenticate_parent(
        self,
        listener: &Lc0ParentListener,
        until: Instant,
    ) -> Result<Lc0SupervisedSession, NativeBootstrapError> {
        if !Arc::ptr_eq(&self.ticket_token, &listener.ticket_token) {
            return Err(NativeBootstrapError::Config(
                "LC0 listener belongs to another ticket",
            ));
        }
        let stream = listener
            .accept_parent(until.min(self.control_until))
            .map_err(|_| NativeBootstrapError::Config("LC0 original parent accept failed"))?;
        self.authenticate_inner(stream, until, Some(Arc::clone(&listener.receipt_token)))
    }

    /// Additive pre-native entry that retains the original ticket and error
    /// after failed admission. An accepted stream may close on authentication
    /// error; once authenticated, the same parent is retained even if asset
    /// validation or custody construction fails.
    pub fn authenticate_parent_preserving_failure(
        self,
        listener: &Lc0ParentListener,
        until: Instant,
    ) -> Result<Lc0SupervisedSession, Lc0PendingFailure> {
        if !Arc::ptr_eq(&self.ticket_token, &listener.ticket_token) {
            return Err(Lc0PendingFailure {
                pending: self,
                cause: Lc0PendingFailureCause::TicketMismatch,
                authenticated_parent: None,
            });
        }
        let stream = match listener.accept_parent(until.min(self.control_until)) {
            Ok(stream) => stream,
            Err(error) => {
                return Err(Lc0PendingFailure {
                    pending: self,
                    cause: Lc0PendingFailureCause::Accept(error),
                    authenticated_parent: None,
                });
            }
        };
        let parent = match self
            .pending
            .authenticate(stream, until.min(self.control_until))
        {
            Ok(parent) => parent,
            Err(error) => {
                return Err(Lc0PendingFailure {
                    pending: self,
                    cause: Lc0PendingFailureCause::Authentication(error),
                    authenticated_parent: None,
                });
            }
        };
        let asset = match MaiaAsset::load(
            &self.config.source_weights,
            &self.config.onnx_model,
            &self.config.export_manifest,
        ) {
            Ok(asset) => asset,
            Err(error) => {
                return Err(Lc0PendingFailure {
                    pending: self,
                    cause: Lc0PendingFailureCause::Asset(error.into()),
                    authenticated_parent: Some(parent),
                });
            }
        };
        let custody = match Lc0SystemStartupCustody::new(
            asset,
            Arc::clone(&parent),
            self.epoch,
            &self.lock,
            &self.config,
        ) {
            Ok(custody) => custody,
            Err(error) => {
                return Err(Lc0PendingFailure {
                    pending: self,
                    cause: Lc0PendingFailureCause::Custody(error.into()),
                    authenticated_parent: Some(parent),
                });
            }
        };
        Ok(self.into_supervised_session(custody, Some(Arc::clone(&listener.receipt_token))))
    }

    /// This stream must come from the listener owned by the original process
    /// supervisor. Any error here occurs before ORT/session native entry.
    pub fn authenticate(
        self,
        stream: UnixStream,
        until: Instant,
    ) -> Result<Lc0SupervisedSession, NativeBootstrapError> {
        self.authenticate_inner(stream, until, None)
    }

    fn authenticate_inner(
        self,
        stream: UnixStream,
        until: Instant,
        receipt_token: Option<Arc<()>>,
    ) -> Result<Lc0SupervisedSession, NativeBootstrapError> {
        let parent = self
            .pending
            .authenticate(stream, until.min(self.control_until))
            .map_err(|_| NativeBootstrapError::Config("LC0 original parent admission failed"))?;
        let asset = MaiaAsset::load(
            &self.config.source_weights,
            &self.config.onnx_model,
            &self.config.export_manifest,
        )?;
        let custody =
            Lc0SystemStartupCustody::new(asset, parent, self.epoch, &self.lock, &self.config)?;
        Ok(self.into_supervised_session(custody, receipt_token))
    }

    fn into_supervised_session(
        self,
        custody: Lc0SystemStartupCustody,
        receipt_token: Option<Arc<()>>,
    ) -> Lc0SupervisedSession {
        Lc0SupervisedSession {
            supervisor: Lc0ControlSupervisor::new(
                Lc0SystemSessionOwner::new(custody),
                self.control_until,
            ),
            async_custody: None,
            slot_start_panic: None,
            async_finish_attempted: false,
            lock: self.lock,
            config: self.config,
            owners: Arc::new(OwnerRegistry::default()),
            ticket_token: self.ticket_token,
            receipt_token,
            profile: None,
            profile_failure: None,
            profile_verification_failure: None,
            endpoint_retirement_started: false,
            epoch: self.epoch,
            publications: PublicationObservations::default(),
        }
    }
}

impl Lc0SupervisedSession {
    /// The supervised CLI must check this before opening assets or entering
    /// CUDA. Enable only when in-place start, normal finish and caller-owned
    /// pending/drop custody are all implemented and independently verified.
    pub const fn async_service_available() -> bool {
        false
    }

    /// A projection of this exact owner and its own publication attempts.
    /// It cannot authorize process exit or establish physical completion.
    pub fn original_session_observation(
        &self,
    ) -> Result<Lc0OriginalSessionObservation, NativeBootstrapError> {
        let ordinal = match &self.lock.runtime {
            NativeRuntimeSourceV6::SystemCuda(spec) => spec.cuda_device_id,
            _ => {
                return Err(NativeBootstrapError::Config(
                    "LC0 observation requires CUDA",
                ));
            }
        };
        let source = self.supervisor.project_original();
        Ok(Lc0OriginalSessionObservation {
            schema: "rz-lc0-original-session-observation/1",
            process_id: std::process::id(),
            process_epoch: self.epoch,
            lock_sha256: self.lock.canonical_sha256.clone(),
            cuda_ordinal: ordinal,
            ort_entry_observed: source.ort_entry_observed,
            service: source.service,
            original_release: source.original_release,
            latest_release: source.latest_release,
            scope_settled: source.scope_settled,
            runtime_start_receipt: self.publications.runtime_start,
            runtime_finish_receipt: self.publications.runtime_finish,
            original_failure_receipt: self.publications.original_failure,
            finish_receipt: self.publications.finish,
            reobservation_receipt: self.publications.reobservation,
            endpoint_retirement: self.publications.endpoint_retirement,
            output_preservation: "unknown",
            physical_completion: "unknown",
        })
    }
    fn check_receipts(&self, receipts: &Lc0ReceiptOwner) -> Result<(), String> {
        if !self
            .receipt_token
            .as_ref()
            .is_some_and(|token| Arc::ptr_eq(token, &receipts.owner_token))
        {
            return Err("LC0 receipts do not belong to the original listener".into());
        }
        Ok(())
    }
    /// Whether this exact session was safely settled or never entered native
    /// code. This is not a GPU-wide completion or acceptance claim and alone
    /// does not authorize process exit: original failures and outputs must
    /// first be recovered by the owning supervisor.
    pub fn original_scope_settled(&self) -> bool {
        self.supervisor.original_scope_settled()
    }

    /// Reobserve only this original worker's pending release. The operation
    /// never changes a failure into success or asserts device-wide fencing.
    pub fn reobserve_original_release(&mut self) {
        self.supervisor.reobserve();
    }

    /// Service outcome remains independent of same-session cleanup.
    pub fn original_service_succeeded(&self) -> bool {
        self.supervisor.original_service_succeeded()
    }

    /// Renew a finite control/observation window on this original session.
    /// The original parent must still be live, the runtime lock and process
    /// identity must match, and an already retired endpoint stays retired.
    /// This does not retry native work or authorize process/session release.
    pub fn renew_control_window(&mut self, window: std::time::Duration) -> io::Result<Instant> {
        if self.endpoint_retirement_started {
            return Err(io::Error::other(
                "LC0 listener retirement closed control renewal",
            ));
        }
        self.supervisor.renew_control_window(&self.lock, window)
    }

    /// An individual bad control frame must not be promoted to parent loss.
    pub fn original_parent_liveness(&self) -> io::Result<()> {
        self.supervisor.original_parent_liveness()
    }

    /// Publish a bounded, non-authoritative classification even when startup
    /// failed before any finish receipt existed. Original errors, descriptors,
    /// and the same native owner remain in this process on every outcome.
    pub fn publish_original_failure_observation(
        &mut self,
        receipts: &Lc0ReceiptOwner,
    ) -> Result<(), String> {
        self.check_receipts(receipts)?;
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let profile = match self.profile_failure.as_ref() {
                Some(Lc0ProfileCreateError::BeforeCreate(error)) => Some(serde_json::json!({
                    "stage": "before_create", "error": control::backend_failure(error),
                })),
                Some(Lc0ProfileCreateError::AfterCreate { cause, .. }) => Some(serde_json::json!({
                    "stage": "after_create", "error": control::backend_failure(cause),
                    "original_directory_owner_retained": true,
                })),
                None => self.profile_verification_failure.as_ref().map(|error| {
                    serde_json::json!({
                        "stage": "verify_original", "error": control::backend_failure(error),
                        "original_directory_owner_retained": true,
                    })
                }),
            };
            let value = serde_json::json!({
                "domain": "rz-lc0-original-failure-observation/1",
                "process_id": std::process::id(),
                "process_epoch": self.epoch,
                "lock_sha256": self.lock.canonical_sha256,
                "profile": profile,
                "supervisor": self.supervisor.original_failure_observation(),
                "scope": "bounded classifications only; original owner and errors remain authoritative",
                "physical_completion": "unknown",
            });
            let bytes = serde_json::to_vec(&value).map_err(|error| error.to_string())?;
            receipts
                .writer
                .write_bytes("lc0-original-failure.v1.json", &bytes)
        }));
        let outcome = match result {
            Ok(result) => result,
            Err(payload) => {
                self.supervisor.retain_publication_panic(payload);
                Err("LC0 original failure observation panicked; owner retained".into())
            }
        };
        self.publications.original_failure.record(outcome.is_ok());
        outcome
    }

    /// Publication failure never changes the retained service or physical
    /// outcome. The caller must preserve and report this independent error.
    pub fn publish_runtime_receipt(
        &mut self,
        receipts: &Lc0ReceiptOwner,
        finish: bool,
    ) -> Result<(), String> {
        self.check_receipts(receipts)?;
        let outcome = self
            .supervisor
            .publish_runtime_receipt(&receipts.writer, &self.lock, finish);
        let observed = if finish {
            &mut self.publications.runtime_finish
        } else {
            &mut self.publications.runtime_start
        };
        observed.record(outcome.is_ok());
        outcome
    }

    pub fn publish_finish_observation(&mut self, receipts: &Lc0ReceiptOwner) -> Result<(), String> {
        self.check_receipts(receipts)?;
        let outcome = self
            .supervisor
            .publish_finish_observation(&receipts.writer, &self.lock);
        self.publications.finish.record(outcome.is_ok());
        outcome
    }

    pub fn publish_reobservation(&mut self, receipts: &Lc0ReceiptOwner) -> Result<(), String> {
        self.check_receipts(receipts)?;
        let outcome = self
            .supervisor
            .publish_reobservation(&receipts.writer, &self.lock);
        self.publications.reobservation.record(outcome.is_ok());
        outcome
    }
    /// This only returns the same registry used to construct the factory.
    pub fn owners(&self) -> Arc<OwnerRegistry> {
        Arc::clone(&self.owners)
    }

    /// Construct the B1 backend configuration internally and retain the
    /// exclusive procfd-backed profile directory with this same session. A
    /// failed directory creation is kept as original evidence, not retried.
    pub fn prepare_private_until(
        &mut self,
        until: Instant,
        cancelled: &AtomicBool,
    ) -> Result<(), &'static str> {
        if self.endpoint_retirement_started
            || self.receipt_token.is_none()
            || self.profile.is_some()
            || self.profile_failure.is_some()
        {
            return Err(
                "LC0 private preparation requires its original listener and a fresh profile",
            );
        }
        let profile = match Lc0ProfileDirectory::create(&self.config, self.epoch) {
            Ok(profile) => profile,
            Err(error) => {
                self.profile_failure = Some(error);
                return Err("LC0 private profile creation failed; original state retained");
            }
        };
        if let Err(error) = profile.verify_original() {
            let kind = error.kind;
            self.profile = Some(profile);
            self.profile_verification_failure = Some(error);
            return Err(if kind == FailureKind::IdentityMismatch {
                "LC0 private profile identity differs; original owner retained"
            } else {
                "LC0 private profile observation failed; original owner retained"
            });
        }
        let spec = match &self.lock.runtime {
            NativeRuntimeSourceV6::SystemCuda(spec) => spec,
            _ => {
                self.profile = Some(profile);
                return Err("LC0 private preparation requires a system CUDA lock");
            }
        };
        let arena_bytes = match usize::try_from(spec.cuda_arena_bytes) {
            Ok(value) => value,
            Err(_) => {
                self.profile = Some(profile);
                return Err("LC0 CUDA arena size exceeds this process");
            }
        };
        let mut backend = BackendConfig::cpu();
        backend.provider = Provider::Cuda {
            device_id: spec.cuda_device_id,
            arena_bytes,
        };
        backend.experiments = self.config.execution_experiments;
        backend.max_batch = 1;
        backend.intra_threads = 1;
        backend.host_io_bytes = IO_BYTES_PER_ITEM;
        backend.profiling_prefix = Some(profile.prefix().to_path_buf());
        // Store before the first native call, including every failed/unwound call.
        self.profile = Some(profile);
        self.supervisor.prepare_until(
            &self.lock,
            until,
            cancelled,
            backend,
            &self.owners,
            &self.config,
        )
    }

    /// Only the original descriptor-owned profiling prefix is accepted. New
    /// callers should use `prepare_private_until` to create it and prepare in
    /// one operation. All B1 settings are checked before native entry.
    pub fn prepare_until(
        &mut self,
        until: Instant,
        cancelled: &AtomicBool,
        backend: BackendConfig,
    ) -> Result<(), &'static str> {
        if self.endpoint_retirement_started || self.receipt_token.is_none() {
            return Err(
                "LC0 native startup requires the original owned listener and receipt directory",
            );
        }
        let profile = self
            .profile
            .as_ref()
            .ok_or("LC0 original private profile owner is absent")?;
        if let Err(error) = profile.verify_original() {
            self.profile_verification_failure.get_or_insert(error);
            return Err("LC0 original profile identity is unknown");
        }
        if backend.profiling_prefix.as_deref() != Some(profile.prefix()) {
            return Err("LC0 profiling prefix differs from the original private owner");
        }
        self.supervisor.prepare_until(
            &self.lock,
            until,
            cancelled,
            backend,
            &self.owners,
            &self.config,
        )
    }

    /// Setup, UCI serving and its failures must be inside `serve`. Return from
    /// this function is not proof of process-wide or device-wide completion.
    pub fn serve_until(
        &mut self,
        until: Instant,
        serve: impl FnOnce(Arc<NativeCudaFactory>) -> Result<(), EngineError>,
    ) -> Result<(), NativeBootstrapError> {
        if self.endpoint_retirement_started {
            return Err(NativeBootstrapError::Config(
                "LC0 listener retirement closed native admission",
            ));
        }
        self.supervisor.serve_until(until, serve)
    }

    /// Currently refuses async admission and returns the original inputs.
    /// Actual start is held until the original owner survives every start
    /// unwind and pending run; the CLI guard is independent.
    pub fn start_async_service(
        &mut self,
        inputs: Lc0AsyncInputs,
    ) -> Result<Lc0AsyncAdmission, (NativeBootstrapError, Lc0AsyncInputs)> {
        // `start_async` can unwind after taking its input owners. Until the
        // same-owner in-place start can retain them, do not enter native
        // service through a public library caller either.
        if !Self::async_service_available() {
            return Err((
                NativeBootstrapError::Config("LC0 async start custody is not available"),
                inputs,
            ));
        }
        self.start_async_service_with_pending_custody(inputs)
    }

    // Prepared implementation; intentionally not called while start-unwind
    // custody remains unknown. It must not become an alternate public entry.
    fn start_async_service_with_pending_custody(
        &mut self,
        inputs: Lc0AsyncInputs,
    ) -> Result<Lc0AsyncAdmission, (NativeBootstrapError, Lc0AsyncInputs)> {
        if self.endpoint_retirement_started
            || self.receipt_token.is_none()
            || self.async_custody.is_some()
        {
            return Err((
                NativeBootstrapError::Config("LC0 async admission is unavailable"),
                inputs,
            ));
        }
        let factory = match self.supervisor.begin_service() {
            Ok(factory) => factory,
            Err(error) => return Err((error, inputs)),
        };
        let clock = ProcessClock::new(rz_contracts::ProcessEpoch(self.epoch));
        let mut process = EngineProcess::new(factory, Arc::clone(&self.owners), clock)
            .with_identity(crate::EngineIdentity {
                name: "RoveZero LC0 weights ONNX CUDA integration".into(),
                author: "RoveZero contributors".into(),
            });
        let close = match process.install_priority_close() {
            Ok(close) => close,
            Err(error) => {
                self.async_custody = Some(Lc0AsyncCustody::CloseSetupFailed {
                    _process: process,
                    _inputs: inputs,
                    _error: error,
                });
                return Ok(Lc0AsyncAdmission::RetainedFailure);
            }
        };
        let max_output_bytes = inputs.max_output_bytes;
        let poll_interval = inputs.poll_interval;
        let slot = AsyncEngineStartSlot::new(
            process,
            inputs.events,
            inputs.sender,
            inputs.output,
            close.clone(),
            self.config.engine_settings(),
        );
        self.async_custody = Some(Lc0AsyncCustody::Slot {
            slot,
            close,
            stage: Lc0SlotStage::Created,
        });
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let Some(Lc0AsyncCustody::Slot { slot, stage, .. }) = self.async_custody.as_mut()
            else {
                return false;
            };
            let status = slot.preflight();
            *stage = Lc0SlotStage::Preflight(status);
            if status != InPlacePreflightStatus::Ready {
                return false;
            }
            let status = slot.try_start_timer();
            *stage = Lc0SlotStage::Timer(status);
            if status != InPlaceTimerStatus::Started {
                return false;
            }
            match slot.try_install_run(max_output_bytes, poll_interval) {
                Ok(()) => {
                    *stage = Lc0SlotStage::Running;
                    true
                }
                Err(status) => {
                    *stage = Lc0SlotStage::Install(status);
                    false
                }
            }
        }));
        match outcome {
            Ok(true) => Ok(Lc0AsyncAdmission::Started),
            Ok(false) => Ok(Lc0AsyncAdmission::RetainedFailure),
            Err(payload) => {
                self.slot_start_panic = Some(payload);
                Ok(Lc0AsyncAdmission::Unknown)
            }
        }
    }

    /// A bounded tick on the same UCI owner. OutputQueued is not a writer
    /// join, session release, or physical-completion observation.
    pub fn drive_async_until(&mut self, until: Instant) -> Option<AsyncServeStatus> {
        match self.async_custody.as_mut()? {
            Lc0AsyncCustody::Running { run, .. } => run.drive_until(until),
            Lc0AsyncCustody::Slot { slot, .. } => slot.original_run_mut()?.drive_until(until),
            _ => None,
        }
    }

    /// An interrupted transfer is not a completed admission. The marker does
    /// not assert that values moved into a panicking start were recovered.
    pub fn async_start_outcome_unknown(&self) -> bool {
        self.slot_start_panic.is_some()
            || matches!(
                self.async_custody.as_ref(),
                Some(Lc0AsyncCustody::Starting | Lc0AsyncCustody::StartPanicked(_))
            )
    }

    /// Borrow the exact engine setup failure and its retained input owners.
    pub(crate) fn async_start_failure(&self) -> Option<&Lc0StartFailure> {
        match self.async_custody.as_ref()? {
            Lc0AsyncCustody::StartFailed(error) => Some(error),
            _ => None,
        }
    }

    /// Request writer close only on this session's original unstarted error.
    /// Prepared startup custody remains untouched and physical state unknown.
    pub(crate) fn request_unstarted_writer_close(&mut self) -> Option<UnstartedWriterCloseStatus> {
        match self.async_custody.as_mut()? {
            Lc0AsyncCustody::StartFailed(error) => Some(error.request_unstarted_writer_close()),
            Lc0AsyncCustody::Slot { slot, .. } => Some(slot.request_idle_writer_close()),
            _ => None,
        }
    }

    /// Poll the original unstarted writer without blocking or replacing the
    /// start-failure owner. Recovered does not assert successful delivery.
    pub(crate) fn try_recover_unstarted_writer(&mut self) -> Option<UnstartedWriterRecoveryStatus> {
        match self.async_custody.as_mut()? {
            Lc0AsyncCustody::StartFailed(error) => Some(error.try_recover_unstarted_writer()),
            Lc0AsyncCustody::Slot { slot, .. } => Some(slot.try_recover_idle_writer()),
            _ => None,
        }
    }

    /// The original writer recovery or panic remains owned by StartFailed.
    pub(crate) fn unstarted_writer_recovery(
        &self,
    ) -> Option<
        &Result<
            OutputRecovery<ProfiledWrite<std::io::Stdout>, std::io::Stderr>,
            Box<dyn std::any::Any + Send>,
        >,
    > {
        match self.async_custody.as_ref()? {
            Lc0AsyncCustody::StartFailed(error) => error.unstarted_writer_recovery(),
            Lc0AsyncCustody::Slot { slot, .. } => slot.idle_writer_recovery(),
            _ => None,
        }
    }

    /// The same original priority handle, not an acknowledgement of shutdown.
    pub fn async_close_handle(&self) -> Option<PriorityCloseHandle> {
        match self.async_custody.as_ref()? {
            Lc0AsyncCustody::Running { close, .. } => Some(close.clone()),
            Lc0AsyncCustody::Slot { close, .. } => Some(close.clone()),
            _ => None,
        }
    }

    /// Advance the original normal UCI run's final collection once. All
    /// output and CPU checks refer to that same run and its joined writer.
    /// A failed or unknown stage remains retained and is never treated as
    /// success. NativeCollected is not a physical-completion witness.
    pub fn try_finish_async(&mut self, until: Instant) -> Lc0AsyncFinishStatus {
        if self.async_finish_attempted {
            return Lc0AsyncFinishStatus::AlreadyAttempted;
        }
        if self.slot_start_panic.is_some() {
            return Lc0AsyncFinishStatus::ServiceFailed;
        }
        let run = match self.async_custody.as_mut() {
            Some(Lc0AsyncCustody::Running { run, .. }) => run,
            Some(Lc0AsyncCustody::Slot { slot, .. }) => match slot.original_run_mut() {
                Some(run) => run,
                None => return Lc0AsyncFinishStatus::NoRun,
            },
            _ => return Lc0AsyncFinishStatus::NoRun,
        };
        match run.last_exit() {
            None => return Lc0AsyncFinishStatus::ServicePending,
            Some(exit)
                if exit.status == AsyncServeStatus::OutputQueued
                    && exit.failure.is_none()
                    && exit.cleanup_failures.is_empty() => {}
            Some(exit) if exit.status == AsyncServeStatus::Yielded => {
                return Lc0AsyncFinishStatus::ServicePending;
            }
            Some(_) => return Lc0AsyncFinishStatus::ServiceFailed,
        }
        if !run.session_closed() || run.pending_count() != 0 {
            return Lc0AsyncFinishStatus::ServicePending;
        }
        match run.try_join_timer_after_workers() {
            EngineTimerJoinStatus::Joined => {}
            EngineTimerJoinStatus::NotReady
            | EngineTimerJoinStatus::WaitingWorkers
            | EngineTimerJoinStatus::WaitingTimer => return Lc0AsyncFinishStatus::CpuPending,
            _ => return Lc0AsyncFinishStatus::CpuFailed,
        }
        match run.try_recover_writer() {
            EngineWriterRecoveryStatus::Recovered => {}
            EngineWriterRecoveryStatus::DriverNotSealed | EngineWriterRecoveryStatus::Pending => {
                return Lc0AsyncFinishStatus::WriterPending;
            }
            EngineWriterRecoveryStatus::WorkerPanicked => {
                return Lc0AsyncFinishStatus::WriterFailed;
            }
            EngineWriterRecoveryStatus::OriginalRecoveryMissing => {
                return Lc0AsyncFinishStatus::WriterUnknown;
            }
        }
        let Some(Ok(recovery)) = run.output_recovery() else {
            return Lc0AsyncFinishStatus::WriterUnknown;
        };
        if recovery.failure.is_some() || recovery.partial.is_some() {
            return Lc0AsyncFinishStatus::WriterFailed;
        }
        let Some(snapshot) = run.final_output_observation() else {
            return Lc0AsyncFinishStatus::WriterUnknown;
        };
        if snapshot.failed || snapshot.call_completion_unknown {
            return Lc0AsyncFinishStatus::WriterFailed;
        }
        if !snapshot.worker_finished
            || !snapshot.close_requested
            || snapshot.accepted_batches != snapshot.completed_batches
        {
            return Lc0AsyncFinishStatus::WriterUnknown;
        }
        self.async_finish_attempted = true;
        match self.supervisor.finish_service(until, Ok(())) {
            Ok(()) => Lc0AsyncFinishStatus::NativeCollected,
            Err(_) => Lc0AsyncFinishStatus::NativeCollectionFailed,
        }
    }

    /// Joins only confirmed-finished engine CPU workers and their timer. The
    /// original writer, native worker and physical completion remain separate.
    pub(crate) fn try_join_async_timer(&mut self) -> Option<EngineTimerJoinStatus> {
        match self.async_custody.as_mut()? {
            Lc0AsyncCustody::Running { run, .. } => Some(run.try_join_timer_after_workers()),
            Lc0AsyncCustody::Slot { slot, .. } => {
                Some(slot.original_run_mut()?.try_join_timer_after_workers())
            }
            _ => None,
        }
    }

    /// Poll the original writer without blocking or replacing the run. A
    /// recovered writer is independent of native session and GPU completion.
    pub(crate) fn try_recover_async_writer(&mut self) -> Option<EngineWriterRecoveryStatus> {
        match self.async_custody.as_mut()? {
            Lc0AsyncCustody::Running { run, .. } => Some(run.try_recover_writer()),
            Lc0AsyncCustody::Slot { slot, .. } => {
                Some(slot.original_run_mut()?.try_recover_writer())
            }
            _ => None,
        }
    }

    /// Keep the exact output, queued batches, partial bytes, errors or writer
    /// panic in the original run. Merely borrowing them does not replay output.
    pub(crate) fn async_writer_recovery(
        &self,
    ) -> Option<
        &Result<
            OutputRecovery<ProfiledWrite<std::io::Stdout>, std::io::Stderr>,
            Box<dyn std::any::Any + Send>,
        >,
    > {
        match self.async_custody.as_ref()? {
            Lc0AsyncCustody::Running { run, .. } => run.output_recovery(),
            Lc0AsyncCustody::Slot { slot, .. } => slot.original_run()?.output_recovery(),
            _ => None,
        }
    }

    /// Required after setup failure before any service has begun. A session
    /// that never reached a worker remains explicitly unknown and retained.
    pub fn close_without_service(&mut self) {
        self.supervisor.close_without_service();
    }

    /// The listener belongs to the external process supervisor; its creation,
    /// path and cleanup are outside this facade. This bounded loop processes
    /// one request at a time only after the caller has returned from serving.
    pub fn control_loop(&mut self, listener: &UnixListener) -> io::Result<()> {
        self.supervisor.control_loop(listener)
    }

    /// Accept and handle one request on the original owned listener. This is
    /// synchronous and must not be used to claim service-thread concurrency.
    pub fn handle_parent_control(
        &mut self,
        listener: &Lc0ParentListener,
        until: Instant,
    ) -> io::Result<()> {
        if !Arc::ptr_eq(&self.ticket_token, &listener.ticket_token) {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "LC0 listener belongs to another ticket",
            ));
        }
        let stream = listener.accept_parent(until)?;
        self.supervisor.handle_control(stream, until)
    }

    /// Authenticate the original parent and dispatch a single V1 control or
    /// read-only V2 projection frame. The V2 snapshot is taken before accept
    /// and authentication and is explicitly labelled as such in the reply.
    /// This synchronous method does not run during UCI service.
    pub fn handle_parent_control_versioned(
        &mut self,
        listener: &Lc0ParentListener,
        until: Instant,
    ) -> io::Result<()> {
        if !Arc::ptr_eq(&self.ticket_token, &listener.ticket_token) {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "LC0 listener belongs to another ticket",
            ));
        }
        let snapshot = self
            .original_session_observation()
            .map_err(|_| io::Error::other("LC0 original session observation unavailable"))?;
        let stream = listener.accept_parent(until)?;
        self.supervisor
            .handle_control_versioned(stream, until, &snapshot)
    }

    /// Retire only this session's original listener after the caller has
    /// finished using its control endpoint. Same-session settlement is a
    /// necessary condition; it is not a process-wide or device-wide fence.
    /// A failed cleanup remains recorded by the endpoint and is not retried.
    /// The original parent stream, session owner and receipts remain held.
    pub fn retire_parent_listener(
        &mut self,
        listener: &mut Lc0ParentListener,
        until: Instant,
    ) -> io::Result<()> {
        if !self.original_scope_settled() {
            return Err(io::Error::other(
                "LC0 original session is not settled; listener retained",
            ));
        }
        self.retire_original_listener(listener, until)
    }

    /// Async collection has already checked the same run's CPU and writer
    /// owners. Allow endpoint retirement after this original worker's release
    /// even if its retained report records a service or collection failure.
    /// Neither that failure nor device-wide physical status is changed.
    pub fn retire_listener_after_async_release(
        &mut self,
        listener: &mut Lc0ParentListener,
        until: Instant,
    ) -> io::Result<()> {
        if !self.async_finish_attempted
            || self.supervisor.project_original().latest_release != Lc0ObservedRelease::Released
        {
            return Err(io::Error::other(
                "LC0 async original worker release is not confirmed",
            ));
        }
        self.retire_original_listener(listener, until)
    }

    fn retire_original_listener(
        &mut self,
        listener: &mut Lc0ParentListener,
        until: Instant,
    ) -> io::Result<()> {
        if !Arc::ptr_eq(&self.ticket_token, &listener.ticket_token)
            || !self
                .receipt_token
                .as_ref()
                .is_some_and(|token| Arc::ptr_eq(token, &listener.receipt_token))
        {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "LC0 listener belongs to another session",
            ));
        }
        if self.endpoint_retirement_started {
            return Err(io::Error::other(
                "LC0 listener retirement already attempted",
            ));
        }
        // `original_scope_settled` is also true before native entry. Close
        // admission first, including on cleanup failure or unwind, so a
        // retired endpoint cannot be followed by a new CUDA startup.
        self.endpoint_retirement_started = true;
        let outcome = listener.endpoint.cleanup(until);
        self.publications
            .endpoint_retirement
            .record(outcome.is_ok());
        outcome
    }
}
