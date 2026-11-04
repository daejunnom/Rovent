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

/// Owns the original authenticated parent, startup custody and worker.
/// This is not an OS-process lifetime manager; dropping it or exiting on an
/// unknown physical result is not safe. No resource is copied to a cache.
#[must_use]
pub struct Lc0SupervisedSession {
    supervisor: Lc0ControlSupervisor,
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
        Ok(Lc0SupervisedSession {
            supervisor: Lc0ControlSupervisor::new(
                Lc0SystemSessionOwner::new(custody),
                self.control_until,
            ),
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
        })
    }
}

impl Lc0SupervisedSession {
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
        if !self.original_scope_settled() {
            return Err(io::Error::other(
                "LC0 original session is not settled; listener retained",
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
