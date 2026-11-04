//! Explicit /3 P-C Collection input preparation.
//!
//! These are original pinned declarations, not evidence that native work,
//! teachers, physical cleanup, or publication have completed.
use super::*;
use crate::rovent_collect::capture_router_v3::CaptureRouterV3;
use crate::rovent_collect::pc_hooks_lock_v3::RoventPcCollectionBridgeV3;
use crate::rovent_collect::pc_worker_v3::{
    PcDriverEvidenceErrorV3, PcDriverPollV3, PcWorkerSupervisorV3, SupervisorStartCustodyV3,
};
use crate::rovent_collect::teacher::v3_session::TeacherRosterEntryV3;
use rz_experiments::rovent_corpus_lock_v3::record::RoventCorpusManifestLockV3;
use rz_experiments::{RoventActor, RoventInputLockV3, RoventPurpose, RoventResearcher};
use std::os::unix::fs::{FileExt, MetadataExt};

pub const ROVENT_COLLECTION_LOCK_V3_LAUNCH_PLAN: &str = "rz-rovent-collection-lock-v3-launch/1";
pub const ROVENT_OWN_COLLECTION_LOCK_V3_LAUNCH_PLAN: &str =
    "rz-rovent-own-collection-lock-v3-launch/1";
const PLAN_MAX_BYTES: u64 = 256 << 10;
const MANIFEST_MAX_BYTES: u64 = 16 << 20;

/// The selector consumes only the /3 caller options; all native arguments
/// retain their original order and still pass through the native parser.
pub struct PcCollectionCliV3 {
    pub plan_pin: RoventPublicationFile,
    pub native_arguments: Vec<String>,
}

pub fn parse_pc_collection_cli_v3(
    arguments: Vec<String>,
) -> Result<Option<PcCollectionCliV3>, ArenaError> {
    if !arguments.iter().any(|arg| arg.starts_with("--v3-")) {
        return Ok(None);
    }
    let mut fields = BTreeMap::new();
    let mut native_arguments = Vec::new();
    for argument in arguments {
        if argument.starts_with("--v3-") {
            let (key, value) = argument
                .split_once('=')
                .ok_or_else(|| invalid("Collection /3 options require =value"))?;
            check(
                matches!(
                    key,
                    "--v3-mode" | "--v3-plan" | "--v3-plan-sha256" | "--v3-plan-bytes"
                ) && !value.is_empty()
                    && fields.insert(key.to_owned(), value.to_owned()).is_none(),
                "unknown, empty or duplicate Collection /3 option",
            )?;
        } else {
            check(
                !argument.starts_with("--v2-")
                    && !argument.starts_with("--collection-plan")
                    && !argument.starts_with("--utility-cohort-plan")
                    && !argument.starts_with("--utility-intervention-plan")
                    && !argument.starts_with("--prepare-utility-cohort-pins"),
                "Collection /3 and other collection selectors cannot be mixed",
            )?;
            native_arguments.push(argument);
        }
    }
    let mut take = |key: &str| {
        fields
            .remove(key)
            .ok_or_else(|| invalid(format!("missing {key}")))
    };
    check(take("--v3-mode")? == "pc", "unsupported Collection /3 mode")?;
    let plan_pin = RoventPublicationFile {
        path: take("--v3-plan")?,
        sha256: take("--v3-plan-sha256")?,
        bytes: take("--v3-plan-bytes")?
            .parse()
            .map_err(|_| invalid("Collection /3 plan bytes"))?,
    };
    plan_pin.validate()?;
    check(
        plan_pin.path.starts_with("inputs/")
            && valid_relative(&plan_pin.path)
            && plan_pin.bytes <= PLAN_MAX_BYTES,
        "Collection /3 original plan pin",
    )?;
    Ok(Some(PcCollectionCliV3 {
        plan_pin,
        native_arguments,
    }))
}

/// Separate, explicitly pinned Collection resource and roster declaration.
/// Worker limits, mailbox capacity and native role tickets are distinct.
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoventCollectionLockV3LaunchPlan {
    pub schema: String,
    pub manifest: RoventPublicationFile,
    pub resource_plan: RoventPublicationFile,
    pub metadata: RoventCollectionMetadata,
    pub teacher_sources: Vec<RoventTeacherSourceSelection>,
    pub teacher_slots: Vec<RoventCollectionTeacherSlot>,
    pub teacher_constraint: ExternalGoConstraint,
    pub teacher_soft_quantum_ns: u64,
    pub maximum_roots: u16,
    /// Maximum simultaneously owned collection root workers, including ones
    /// whose physical retirement remains unresolved.
    pub maximum_root_workers: u16,
    /// Separate teacher concurrency. The hook must reserve this capacity
    /// before starting actors and retain it until physical retirement.
    pub maximum_simultaneous_teacher_roots: u16,
    pub teacher_mailbox_capacity: u32,
    pub teacher_tickets_per_root: u32,
    pub maximum_records: u64,
    pub output_directory: PathBuf,
    pub publication_timeout_ms: u64,
}

/// The actual file descriptor and exact verified bytes remain alive with the
/// decoded object. Descriptor retention is not a claim that another process
/// cannot mutate the file; the product's private-input rules still apply.
pub(super) struct OriginalPinnedFile {
    _file: cap_std::fs::File,
    bytes: Vec<u8>,
    pin: RoventPublicationFile,
}

impl OriginalPinnedFile {
    pub(super) fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub(super) fn pin(&self) -> &RoventPublicationFile {
        &self.pin
    }

    pub(super) fn read(
        directory: &Dir,
        pin: &RoventPublicationFile,
        maximum: u64,
    ) -> Result<Self, ArenaError> {
        pin.validate()?;
        check(
            pin.path.starts_with("inputs/") && valid_relative(&pin.path) && pin.bytes <= maximum,
            "Collection /3 original private input path/size",
        )?;
        let mut options = cap_std::fs::OpenOptions::new();
        options.read(true).follow(FollowSymlinks::No);
        // `follow(No)` applies only to the final component. Walk all parent
        // directories as no-follow capabilities before opening that leaf.
        let mut parent = directory.try_clone().map_err(io)?;
        let mut parts = Path::new(&pin.path).components().peekable();
        let leaf = loop {
            let part = match parts.next() {
                Some(Component::Normal(part)) => part,
                _ => return Err(invalid("Collection /3 invalid input path component")),
            };
            if parts.peek().is_none() {
                break part;
            }
            parent = parent.open_dir_nofollow(part).map_err(io)?;
        };
        let mut file = parent.open_with(leaf, &options).map_err(io)?;
        let metadata = file.metadata().map_err(io)?;
        check(
            metadata.is_file() && metadata.len() == pin.bytes,
            "Collection /3 original file type/size",
        )?;
        let mut bytes = Vec::new();
        Read::take(&mut file, pin.bytes + 1)
            .read_to_end(&mut bytes)
            .map_err(io)?;
        check(
            bytes.len() as u64 == pin.bytes
                && hex(Digest(Sha256::digest(&bytes).into())) == pin.sha256,
            "Collection /3 original input bytes/SHA",
        )?;
        Ok(Self {
            _file: file,
            bytes,
            pin: pin.clone(),
        })
    }

    /// Consume a duplicate of the already inherited original plan FD; never
    /// reopen the plan pathname or change the shared file-description offset.
    fn from_original_plan(
        file: std::fs::File,
        pin: &RoventPublicationFile,
        maximum: u64,
        deadline: Instant,
    ) -> Result<Self, ArenaError> {
        pin.validate()?;
        check(
            pin.path.starts_with("inputs/") && valid_relative(&pin.path) && pin.bytes <= maximum,
            "Collection /3 original plan pin",
        )?;
        let before = file.metadata().map_err(io)?;
        check(
            before.is_file() && before.len() == pin.bytes,
            "Collection /3 original plan type/size",
        )?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(
                usize::try_from(pin.bytes).map_err(|_| invalid("Collection /3 plan length"))?,
            )
            .map_err(|_| invalid("Collection /3 plan allocation"))?;
        let mut buffer = [0_u8; 65536];
        let mut offset = 0_u64;
        while offset < pin.bytes {
            check(
                Instant::now() < deadline,
                "Collection /3 original plan deadline",
            )?;
            let count = (pin.bytes - offset).min(buffer.len() as u64) as usize;
            match file.read_at(&mut buffer[..count], offset) {
                Ok(0) => return Err(invalid("Collection /3 original plan truncated")),
                Ok(n) => {
                    bytes.extend_from_slice(&buffer[..n]);
                    offset += n as u64;
                }
                Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(error) => return Err(io(error)),
            }
        }
        let after = file.metadata().map_err(io)?;
        check(
            (
                before.dev(),
                before.ino(),
                before.len(),
                before.mtime(),
                before.mtime_nsec(),
                before.ctime(),
                before.ctime_nsec(),
            ) == (
                after.dev(),
                after.ino(),
                after.len(),
                after.mtime(),
                after.mtime_nsec(),
                after.ctime(),
                after.ctime_nsec(),
            ) && hex(Digest(Sha256::digest(&bytes).into())) == pin.sha256,
            "Collection /3 original plan bytes/identity changed",
        )?;
        Ok(Self {
            _file: cap_std::fs::File::from_std(file),
            bytes,
            pin: pin.clone(),
        })
    }
}

/// Retains the original plan and manifest owners across resource admission.
pub struct PreparedPcCollectionInputsV3 {
    original_directory: std::fs::File,
    original_plan: OriginalPinnedFile,
    original_manifest: OriginalPinnedFile,
    plan: RoventCollectionLockV3LaunchPlan,
    manifest: Arc<RoventCorpusManifestLockV3>,
    purpose: CollectionInputsPurposeV3,
    managed_binding: Option<super::managed_output_fd_v2::OriginalInputBindingV2>,
}

/// Original owners prepared before resource admission. Constructing this
/// bundle starts no thread, model, GPU or teacher process. The supervisor is
/// only a finite-step state owner; a separately verified resource-admitted
/// caller must retain it before starting any worker.
#[allow(dead_code)]
pub(crate) struct PcCollectionOwnersV3 {
    inputs: Arc<PreparedPcCollectionInputsV3>,
    router: Arc<CaptureRouterV3>,
    bridge: Arc<RoventPcCollectionBridgeV3>,
    native: Arc<rz_uci::rovent_bootstrap::launch::RoventPcCollectionLaunchOwnerV3>,
    supervisor: Arc<Mutex<PcWorkerSupervisorV3>>,
    driver_custody: Arc<Mutex<SupervisorStartCustodyV3>>,
    execution_ceiling: Instant,
    cleanup_ceiling: Instant,
    launch_claimed: std::sync::atomic::AtomicBool,
}

#[allow(dead_code)]
impl PcCollectionOwnersV3 {
    pub(crate) fn prepare(
        inputs: Arc<PreparedPcCollectionInputsV3>,
        execution_ceiling: Instant,
        cleanup_ceiling: Instant,
    ) -> Result<Self, ArenaError> {
        check(
            inputs.purpose == CollectionInputsPurposeV3::Pc,
            "P-C V3 plan purpose differs",
        )?;
        let (router, observer) = CaptureRouterV3::channel();
        let bridge = RoventPcCollectionBridgeV3::new(Arc::clone(&inputs), router.registration())
            .map_err(invalid)?;
        let hooks: Arc<dyn rz_uci::rovent_bootstrap::pc_collection_v3::RoventPcCollectionHooksV3> =
            bridge.clone();
        let native = Arc::new(
            rz_uci::rovent_bootstrap::launch::RoventPcCollectionLaunchOwnerV3::new(
                inputs.plan().teacher_sources.clone(),
                Vec::new(),
                hooks,
                Box::new(observer),
            ),
        );
        bridge.bind_native_owner(&native).map_err(invalid)?;
        let supervisor = PcWorkerSupervisorV3::new(
            Arc::clone(&bridge),
            Arc::clone(&router),
            execution_ceiling,
            cleanup_ceiling,
        )
        .map_err(invalid)?;
        let supervisor = Arc::new(Mutex::new(supervisor));
        bridge.bind_supervisor(&supervisor).map_err(invalid)?;
        // The recovery registry retains this exact owner before any callback
        // can start a worker. It does not itself start a thread or teacher.
        let driver_custody =
            SupervisorStartCustodyV3::register(&supervisor, cleanup_ceiling).map_err(invalid)?;
        Ok(Self {
            inputs,
            router,
            bridge,
            native,
            supervisor,
            driver_custody,
            execution_ceiling,
            cleanup_ceiling,
            launch_claimed: std::sync::atomic::AtomicBool::new(false),
        })
    }

    pub(crate) fn inputs(&self) -> &Arc<PreparedPcCollectionInputsV3> {
        &self.inputs
    }
    pub(crate) fn execution_ceiling(&self) -> Instant {
        self.execution_ceiling
    }
    pub(crate) fn cleanup_ceiling(&self) -> Instant {
        self.cleanup_ceiling
    }
    pub(crate) fn native(
        &self,
    ) -> &Arc<rz_uci::rovent_bootstrap::launch::RoventPcCollectionLaunchOwnerV3> {
        &self.native
    }
    pub(crate) fn supervisor(&self) -> &Arc<Mutex<PcWorkerSupervisorV3>> {
        &self.supervisor
    }

    pub(crate) fn driver_custody(&self) -> &Arc<Mutex<SupervisorStartCustodyV3>> {
        &self.driver_custody
    }

    /// Resource validation is separate and precedes any worker start.
    pub(crate) fn validate_original_boundary(
        &self,
        lock: &Arc<RoventInputLockV3>,
        directory: &Dir,
        lock_file: &std::fs::File,
    ) -> Result<(), ArenaError> {
        self.inputs
            .validate_original_boundary(lock, directory, lock_file)
    }

    /// Only an authenticated UCI gate and the same admitted arena resource
    /// owner can start the already retained original supervisor driver.
    pub(crate) fn start_verified(
        &self,
        gate: &rz_uci::rovent_bootstrap::launch::RoventPcCollectionStartGateV3,
        issuer: &crate::rovent_resource_scope::launch::PcCollectionResourceStartIssuerV3,
    ) -> Result<(), &'static str> {
        if Instant::now() >= self.execution_ceiling {
            return Err("original collection work deadline expired");
        }
        let control = self
            .bridge
            .startup()
            .ok_or("original startup control missing")?;
        let proof = issuer.issue(gate, control)?;
        proof.with_start_lease(gate, &self.native, control, |lease| {
            if Instant::now() >= self.execution_ceiling {
                return Err("original collection work deadline expired");
            }
            // The resource lease above has checked this exact authenticated
            // gate, native owner, scope and lock. Publish that same lock
            // identity before any driver can admit a root.
            self.bridge
                .bind_input_lock_identity(gate.input_lock_identity())?;
            SupervisorStartCustodyV3::start_verified(
                &self.driver_custody,
                &lease,
                gate,
                &self.native,
                control,
            )
        })?
    }

    /// Invoke the original resource supervisor with the same pre-retained
    /// inputs, bridge and driver. This is intentionally unreachable without
    /// an original managed output owner; a path or an input hash cannot
    /// manufacture that capability. The caller must retain this bundle and
    /// the returned outcome while completing physical settlement.
    pub(crate) fn run_with_original_output(
        self: &Arc<Self>,
        output: &Arc<ManagedOutputOwnerV3>,
        arguments: Vec<String>,
    ) -> Result<PcCollectionRunV3, ArenaError> {
        check(
            output.matches_inputs(&self.inputs),
            "Collection output owner differs",
        )?;
        let pin = self.inputs.resource_plan_pin()?;
        // The original supervisor is already retained in the recovery registry.
        // Retain this same output owner there before any launch callback.
        self.supervisor
            .try_lock()
            .map_err(|_| invalid("Collection supervisor unavailable"))?
            .register_output(output)
            .map_err(invalid)?;
        check(
            !self
                .launch_claimed
                .load(std::sync::atomic::Ordering::Acquire),
            "Collection original owner already used",
        )?;
        // Both composite and outcome recovery owners exist before any
        // resource preparation or native callback can begin.
        let outcome_key =
            crate::rovent_resource_scope::launch::RoventPcCollectionRecoveryKeyV3::prepare_for(
                &self.native,
            );
        let outcome_reservation = outcome_key
            .reserve()
            .ok_or_else(|| invalid("Collection original outcome already reserved"))?;
        let recovery = super::pc_recovery_v3::PcRunRecoveryKeyV3::prepare(
            Arc::clone(self),
            Arc::clone(output),
            outcome_key.clone(),
        )
        .map_err(|_| invalid("Collection original recovery registration failed"))?;
        check(
            self.launch_claimed
                .compare_exchange(
                    false,
                    true,
                    std::sync::atomic::Ordering::AcqRel,
                    std::sync::atomic::Ordering::Acquire,
                )
                .is_ok(),
            "Collection original owner already used",
        )?;
        recovery.begin().map_err(invalid)?;
        let validate_owner = Arc::clone(self);
        let start_owner = Arc::clone(self);
        let validate_output = Arc::clone(output);
        let resource_recovery = recovery.clone();
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            crate::rovent_resource_scope::launch::run_pc_collection_with_resources_v3_preowned(
                pin,
                arguments,
                &self.native,
                Box::new(move |lock, directory, lock_file| {
                    validate_owner
                        .validate_original_boundary(lock, directory, lock_file)
                        .map_err(
                            |error| rz_uci::rovent_bootstrap::launch::RoventLaunchError {
                                stage: "collection input validation",
                                detail: error.to_string(),
                            },
                        )?;
                    validate_output
                        .authorize_original_boundary(&validate_owner.inputs)
                        .map_err(
                            |detail| rz_uci::rovent_bootstrap::launch::RoventLaunchError {
                                stage: "collection output authorization",
                                detail: detail.to_owned(),
                            },
                        )
                }),
                Box::new(move |gate, issuer| {
                    start_owner.start_verified(gate, issuer).map_err(|error| {
                        rz_uci::rovent_bootstrap::launch::RoventLaunchError {
                            stage: "collection supervisor start",
                            detail: error.to_owned(),
                        }
                    })
                }),
                Box::new(move |resource| {
                    resource_recovery
                        .register_resource(Arc::clone(resource))
                        .map_err(|_| "Collection original resource retention failed")
                }),
                outcome_reservation,
            )
        }));
        let outcome = match outcome {
            Ok(Ok(outcome)) => outcome,
            Ok(Err(_reservation)) => {
                return Err(invalid("Collection outcome reservation owner differs"));
            }
            Err(payload) => match recovery.record_panic(payload) {
                Ok(()) => {
                    return Err(invalid(
                        "Collection original launch panicked; custody retained",
                    ));
                }
                Err(payload) => std::panic::resume_unwind(payload),
            },
        };
        // The identical outcome key was already registered before execution;
        // failure to update this observation does not discard the outcome.
        let (recovery_recorded, recovery_error) = match recovery.try_record_outcome_key(&outcome) {
            Ok(()) => (true, None),
            Err(super::pc_recovery_v3::PcRunRecoveryRecordErrorV3::Busy) => (false, None),
            Err(super::pc_recovery_v3::PcRunRecoveryRecordErrorV3::Poisoned) => {
                (false, Some("Collection recovery slot poisoned"))
            }
            Err(super::pc_recovery_v3::PcRunRecoveryRecordErrorV3::Invalid) => {
                (false, Some("Collection recovery outcome differs"))
            }
        };
        Ok(PcCollectionRunV3 {
            owners: Arc::clone(self),
            output: Arc::clone(output),
            outcome,
            recovery,
            recovery_recorded,
            recovery_error,
            settled: false,
        })
    }
}

/// The result retains the same original owners; it is not a settlement,
/// durable-publication receipt, or GPU completion proof.
#[allow(dead_code)]
pub(crate) struct PcCollectionRunV3 {
    owners: Arc<PcCollectionOwnersV3>,
    output: Arc<ManagedOutputOwnerV3>,
    outcome: crate::rovent_resource_scope::launch::RoventPcCollectionResourceOutcomeV3,
    recovery: super::pc_recovery_v3::PcRunRecoveryKeyV3,
    recovery_recorded: bool,
    recovery_error: Option<&'static str>,
    settled: bool,
}

impl PcCollectionRunV3 {
    /// Poll this supervisor's original recovery without conflating contention,
    /// a missing outcome, and poisoned custody. None is not a no-worker proof.
    pub(crate) fn recover_original_poll(
        owners: &Arc<PcCollectionOwnersV3>,
        output: &Arc<ManagedOutputOwnerV3>,
    ) -> Result<PcDriverPollV3<Option<Self>>, PcDriverEvidenceErrorV3> {
        use super::pc_recovery_v3::{PcRunRecoveryKeyV3, PcRunRecoveryStateV3};
        use crate::rovent_resource_scope::launch::RoventPcCollectionRecoveryErrorV3;
        let snapshot = match PcRunRecoveryKeyV3::try_recovery_snapshot_for_supervisor(
            owners.supervisor(),
        )? {
            PcDriverPollV3::Pending => return Ok(PcDriverPollV3::Pending),
            PcDriverPollV3::Ready(None) => return Ok(PcDriverPollV3::Ready(None)),
            PcDriverPollV3::Ready(Some(snapshot)) => snapshot,
        };
        if !Arc::ptr_eq(&snapshot.owners, owners) || !Arc::ptr_eq(&snapshot.output, output) {
            return Err(PcDriverEvidenceErrorV3::Invalid("original recovery owners differ"));
        }
        let outcome = match snapshot.outcome_key.try_recover_typed() {
            Ok(Some(outcome)) => outcome,
            Ok(None) => return Ok(PcDriverPollV3::Ready(None)),
            Err(RoventPcCollectionRecoveryErrorV3::Busy) => return Ok(PcDriverPollV3::Pending),
            Err(RoventPcCollectionRecoveryErrorV3::Poisoned) => {
                return Err(PcDriverEvidenceErrorV3::Poisoned("original outcome"));
            }
        };
        if !outcome.matches_original_collection(owners.native()) {
            return Err(PcDriverEvidenceErrorV3::Invalid("original outcome differs"));
        }
        let recovery_recorded = match snapshot.state {
            PcRunRecoveryStateV3::Returned | PcRunRecoveryStateV3::Panicked => true,
            PcRunRecoveryStateV3::Executing => false,
            PcRunRecoveryStateV3::Prepared => {
                return Err(PcDriverEvidenceErrorV3::Invalid("outcome was not executing"));
            }
        };
        Ok(PcDriverPollV3::Ready(Some(Self {
            owners: Arc::clone(owners),
            output: Arc::clone(output),
            outcome,
            recovery: snapshot.recovery,
            recovery_recorded,
            recovery_error: None,
            settled: false,
        })))
    }

    /// Reclaim only the outcome held by this supervisor's original recovery
    /// key. None does not prove that no worker or resource exists.
    pub(crate) fn recover_original(
        owners: &Arc<PcCollectionOwnersV3>,
        output: &Arc<ManagedOutputOwnerV3>,
    ) -> Result<Option<Self>, &'static str> {
        use super::pc_recovery_v3::PcRunRecoveryStateV3;
        let Some(recovery) = super::pc_recovery_v3::PcRunRecoveryKeyV3::recover_for_supervisor(
            owners.supervisor(),
        )? else {
            return Ok(None);
        };
        let (original_owners, original_output) = recovery.original_owners()?;
        if !Arc::ptr_eq(&original_owners, owners) || !Arc::ptr_eq(&original_output, output) {
            return Err("Collection original recovery owners differ");
        }
        let Some(key) = recovery.original_outcome_key()? else {
            return Ok(None);
        };
        let Some(outcome) = key.try_recover()? else {
            return Ok(None);
        };
        if !outcome.matches_original_collection(owners.native()) {
            return Err("Collection recovered outcome owner differs");
        }
        let (state, _, _) = recovery.observation()?;
        let recovery_recorded = match state {
            PcRunRecoveryStateV3::Returned | PcRunRecoveryStateV3::Panicked => true,
            PcRunRecoveryStateV3::Executing => false,
            PcRunRecoveryStateV3::Prepared => {
                return Err("Collection recovered outcome was not executing");
            }
        };
        Ok(Some(Self {
            owners: Arc::clone(owners),
            output: Arc::clone(output),
            outcome,
            recovery,
            recovery_recorded,
            recovery_error: None,
            settled: false,
        }))
    }

    /// This key names the retained original owner; it is not completion.
    pub(crate) fn recovery_key(&self) -> super::pc_recovery_v3::PcRunRecoveryKeyV3 {
        self.recovery.clone()
    }

    /// Observe this run's game join independently of physical settlement.
    /// The same recovery key can poll retained settled custody after this run
    /// is dropped; no game result is a GPU or publication completion proof.
    pub(crate) fn poll_game_join(
        &self,
    ) -> Result<super::pc_recovery_v3::PcGameJoinObservationV3, PcDriverEvidenceErrorV3> {
        let outcome = (!self.settled).then_some(&self.outcome);
        self.recovery.poll_game_join(outcome, &self.owners.bridge)
    }

    /// Nonblocking progress for this original run. Pre-admission driver
    /// retirement is independent of native/resource/output settlement. The
    /// original before-native recovery may wait only until `poll_until`,
    /// additionally bounded by the original scope cleanup deadline. A true
    /// result means settled owners are retained in the original slot.
    pub(crate) fn try_settle(&mut self, poll_until: Instant) -> Result<bool, &'static str> {
        if Instant::now() >= poll_until {
            return Ok(false);
        }
        if self.settled {
            return Ok(true);
        }
        if let Some(error) = self.recovery_error {
            return Err(error);
        }
        if !self.recovery_recorded {
            match self.recovery.try_record_outcome_key(&self.outcome) {
                Ok(()) => self.recovery_recorded = true,
                Err(super::pc_recovery_v3::PcRunRecoveryRecordErrorV3::Busy) => return Ok(false),
                Err(super::pc_recovery_v3::PcRunRecoveryRecordErrorV3::Poisoned) => {
                    self.recovery_error = Some("Collection recovery slot poisoned");
                    return Err("Collection recovery slot poisoned");
                }
                Err(super::pc_recovery_v3::PcRunRecoveryRecordErrorV3::Invalid) => {
                    self.recovery_error = Some("Collection recovery outcome differs");
                    return Err("Collection recovery outcome differs");
                }
            }
        }
        match self
            .recovery
            .poll_pre_admission_driver()
            .map_err(|error| match error {
                PcDriverEvidenceErrorV3::Poisoned(reason)
                | PcDriverEvidenceErrorV3::Invalid(reason) => reason,
            })? {
            PcDriverPollV3::Pending => return Ok(false),
            PcDriverPollV3::Ready(true) => {
                // This only progresses the original native failure's own
                // typed reservation recovery. It does not close resources or
                // mark the failed run as successful.
                let _ = self
                    .outcome
                    .try_recover_original_before_native_until(poll_until);
                return Ok(false);
            }
            PcDriverPollV3::Ready(false) => {}
        }
        let Some(fence) = self.owners.native.native_session_fence() else {
            return Ok(false);
        };
        if self
            .recovery
            .try_prepare_stop(&self.owners.bridge, &fence)?
            == super::pc_recovery_v3::PcStopStepV3::Pending
        {
            return Ok(false);
        }
        if self.recovery.try_handoff_stop()? == super::pc_recovery_v3::PcStopStepV3::Pending {
            return Ok(false);
        }
        match self
            .recovery
            .poll_joined_driver(&mut self.outcome)
            .map_err(|error| match error {
                PcDriverEvidenceErrorV3::Poisoned(reason)
                | PcDriverEvidenceErrorV3::Invalid(reason) => reason,
            })? {
            PcDriverPollV3::Pending | PcDriverPollV3::Ready(false) => Ok(false),
            PcDriverPollV3::Ready(true) => {
                match self.recovery.try_settle_outcome(&mut self.outcome)? {
                    PcDriverPollV3::Pending => Ok(false),
                    PcDriverPollV3::Ready(settled) => {
                        self.settled = settled;
                        Ok(settled)
                    }
                }
            }
        }
    }
}

/// Original live supervisor/lease and plan owner; authority is withheld until
/// the actual native input-lock callback has accepted the same input owner.
#[allow(dead_code)]
pub(crate) struct ManagedOutputOwnerV3 {
    inputs: Arc<PreparedPcCollectionInputsV3>,
    state: Mutex<ManagedOutputStateV3>,
    work_until: Instant,
    deadline: Instant,
}

struct ManagedOutputStateV3 {
    observation: super::managed_output_fd_v2::ObservedManagedPlanLeaseV2,
    authorized: bool,
    directory: Option<Dir>,
    chunks: Option<Dir>,
    uncertain: bool,
    completed_writes: u64,
    publication_attempted: bool,
    original_writer: Option<crate::rovent_collect::writer_lock_v3::WriterIdentityV3>,
    closed: Option<ManagedOutputCloseStatusV3>,
}

/// Facts about calls through this one output owner, not corpus, producer,
/// parent lease, worker, publication, or GPU completion evidence.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ManagedOutputCloseStatusV3 {
    NoWrites,
    WritesAcknowledged { count: u64 },
    Unknown,
    UnknownAfterAttempt,
}

/// Retains the exact original output owner along with its frozen close state.
pub(crate) struct ManagedOutputCloseEvidenceV3 {
    owner: Arc<ManagedOutputOwnerV3>,
    status: ManagedOutputCloseStatusV3,
}

impl ManagedOutputCloseEvidenceV3 {
    pub(crate) fn matches_original(&self, owner: &Arc<ManagedOutputOwnerV3>) -> bool {
        Arc::ptr_eq(&self.owner, owner)
    }

    pub(crate) fn status(&self) -> ManagedOutputCloseStatusV3 {
        self.status
    }
}

impl ManagedOutputStateV3 {
    /// Compare the retained chunks capability with the current no-follow
    /// entry under the retained output root. This is an observation, not a
    /// claim that unrelated writers cannot race or mutate that directory.
    fn check_chunks_binding(&self) -> std::io::Result<()> {
        let directory = self
            .directory
            .as_ref()
            .ok_or_else(|| std::io::Error::other("managed output root missing"))?;
        let held = self
            .chunks
            .as_ref()
            .ok_or_else(|| std::io::Error::other("managed chunks missing"))?
            .try_clone()?
            .into_std_file()
            .metadata()?;
        let current = directory
            .open_dir_nofollow("chunks")?
            .into_std_file()
            .metadata()?;
        if !held.is_dir()
            || !current.is_dir()
            || (held.dev(), held.ino()) != (current.dev(), current.ino())
        {
            return Err(std::io::Error::other("managed chunks binding changed"));
        }
        Ok(())
    }

    /// Observe the same retained file under the current leaf without
    /// following a replacement link. This does not exclude later mutation.
    fn check_chunk_leaf_binding(
        &self,
        file: &cap_std::fs::File,
        leaf: &std::ffi::OsStr,
    ) -> std::io::Result<()> {
        let held = file.metadata()?;
        let named = self
            .chunks
            .as_ref()
            .ok_or_else(|| std::io::Error::other("managed chunks missing"))?
            .symlink_metadata(leaf)?;
        if !held.is_file()
            || !named.is_file()
            || (
                cap_fs_ext::MetadataExt::dev(&held),
                cap_fs_ext::MetadataExt::ino(&held),
            ) != (
                cap_fs_ext::MetadataExt::dev(&named),
                cap_fs_ext::MetadataExt::ino(&named),
            )
        {
            return Err(std::io::Error::other("managed chunk leaf binding changed"));
        }
        Ok(())
    }
}

pub(crate) enum ManagedOutputWriteErrorV3 {
    Busy,
    BeforeWrite(String),
    PublicationBeforeWrite {
        reason: String,
        evidence: ManagedPublicationUnstartedEvidenceV3,
    },
    /// The output has not been mutated by this call, but the original game
    /// authority cannot be observed. This is not an ordinary refusal.
    AuthorityUnknown(String),
    AfterAttempt(String),
}

/// Nonclone evidence that one matched game/output metadata attempt returned
/// before its first mutation. Prior chunk writes are a separate scope.
pub(crate) struct ManagedPublicationUnstartedEvidenceV3 {
    output: Arc<ManagedOutputOwnerV3>,
    game: crate::rovent_collect::game_authority_v3::GamePublicationOriginV3,
}

impl ManagedPublicationUnstartedEvidenceV3 {
    pub(crate) fn matches_original<S>(
        &self,
        output: &Arc<ManagedOutputOwnerV3>,
        guard: &crate::rovent_collect::game_authority_v3::GamePublishGuardV3<S>,
    ) -> bool {
        Arc::ptr_eq(&self.output, output) && guard.matches_original_game(&self.game)
    }
}

#[allow(dead_code)]
impl ManagedOutputOwnerV3 {
    /// `deadline` is the original collection cleanup ceiling, not the
    /// shorter startup deadline; expiry preserves any unfinished packet.
    pub(super) fn from_observation(
        inputs: Arc<PreparedPcCollectionInputsV3>,
        observation: super::managed_output_fd_v2::ObservedManagedPlanLeaseV2,
        work_until: Instant,
        deadline: Instant,
    ) -> Result<
        Self,
        (
            ArenaError,
            Arc<PreparedPcCollectionInputsV3>,
            super::managed_output_fd_v2::ObservedManagedPlanLeaseV2,
        ),
    > {
        if work_until > deadline
            || Instant::now() >= work_until
            || Instant::now() >= deadline
            || !inputs
                .managed_binding
                .as_ref()
                .is_some_and(|binding| binding.same_original(&observation.original_binding()))
        {
            return Err((
                invalid("Collection output original observation differs or expired"),
                inputs,
                observation,
            ));
        }
        Ok(Self {
            inputs,
            state: Mutex::new(ManagedOutputStateV3 {
                observation,
                authorized: false,
                directory: None,
                chunks: None,
                uncertain: false,
                completed_writes: 0,
                publication_attempted: false,
                original_writer: None,
                closed: None,
            }),
            work_until,
            deadline,
        })
    }

    /// Only called by the same resource supervisor after its native original
    /// lock and directory validation. It creates no files and starts no work.
    fn authorize_original_boundary(
        &self,
        inputs: &Arc<PreparedPcCollectionInputsV3>,
    ) -> Result<(), &'static str> {
        if !self.matches_inputs(inputs) {
            return Err("Collection output input owner differs");
        }
        let mut state = self
            .state
            .try_lock()
            .map_err(|_| "Collection output state unavailable")?;
        if state.authorized || state.uncertain || state.closed.is_some() {
            return Err("Collection output authority unavailable");
        }
        state.authorized = true;
        Ok(())
    }

    pub(crate) fn inputs(&self) -> &Arc<PreparedPcCollectionInputsV3> {
        &self.inputs
    }

    pub(crate) fn matches_inputs(&self, inputs: &Arc<PreparedPcCollectionInputsV3>) -> bool {
        Arc::ptr_eq(&self.inputs, inputs)
    }

    /// Register only the pipeline's retained original writer before output
    /// starts. Busy is retryable and does not modify either owner.
    pub(crate) fn try_register_original_writer(
        &self,
        writer: &crate::rovent_collect::writer_lock_v3::WriterIdentityV3,
    ) -> Result<bool, &'static str> {
        let mut state = match self.state.try_lock() {
            Ok(state) => state,
            Err(std::sync::TryLockError::WouldBlock) => return Ok(false),
            Err(std::sync::TryLockError::Poisoned(_)) => {
                return Err("managed output state poisoned");
            }
        };
        if state.closed.is_some() || state.uncertain {
            return Err("managed output writer registration unavailable");
        }
        if let Some(original) = &state.original_writer {
            return if original.same_original(writer) {
                Ok(true)
            } else {
                Err("managed output writer differs")
            };
        }
        if state.publication_attempted || state.completed_writes != 0 {
            return Err("managed output writer registration is too late");
        }
        state.original_writer = Some(writer.clone());
        Ok(true)
    }

    /// Close admission to further writes only after the caller has stopped its
    /// producers. `None` is a live writer holding this same owner's mutex;
    /// callers may poll without blocking any other owner. Closing records the
    /// completed calls already observed and preserves uncertainty on panic or
    /// an attempted mutation. It does not close the original lease/channel.
    pub(crate) fn try_close(self: &Arc<Self>) -> Option<ManagedOutputCloseEvidenceV3> {
        let mut state = match self.state.try_lock() {
            Ok(state) => state,
            Err(std::sync::TryLockError::WouldBlock) => return None,
            Err(std::sync::TryLockError::Poisoned(error)) => {
                let mut state = error.into_inner();
                if state.closed.is_none() {
                    state.closed = Some(if state.uncertain {
                        ManagedOutputCloseStatusV3::UnknownAfterAttempt
                    } else {
                        ManagedOutputCloseStatusV3::Unknown
                    });
                }
                state.uncertain = true;
                state
            }
        };
        let status = state.closed.unwrap_or_else(|| {
            if state.uncertain {
                ManagedOutputCloseStatusV3::UnknownAfterAttempt
            } else if state.completed_writes == 0 {
                ManagedOutputCloseStatusV3::NoWrites
            } else {
                ManagedOutputCloseStatusV3::WritesAcknowledged {
                    count: state.completed_writes,
                }
            }
        });
        state.closed = Some(status);
        Some(ManagedOutputCloseEvidenceV3 {
            owner: Arc::clone(self),
            status,
        })
    }

    /// The packet owner is retained by the caller on every error. Once any
    /// filesystem mutation is attempted, failures remain uncertain and may
    /// not be retried as if no output had been created.
    pub(crate) fn checked_write_chunk(
        &self,
        leaf: &std::ffi::OsStr,
        bytes: &[u8],
    ) -> Result<(), ManagedOutputWriteErrorV3> {
        use std::io::Write;
        let mut parts = std::path::Path::new(leaf).components();
        if !matches!(
            (parts.next(), parts.next()),
            (Some(Component::Normal(_)), None)
        ) {
            return Err(ManagedOutputWriteErrorV3::BeforeWrite(
                "invalid chunk leaf".into(),
            ));
        }
        let mut state = match self.state.try_lock() {
            Ok(state) => state,
            Err(std::sync::TryLockError::WouldBlock) => {
                return Err(ManagedOutputWriteErrorV3::Busy);
            }
            Err(std::sync::TryLockError::Poisoned(_)) => {
                return Err(ManagedOutputWriteErrorV3::AfterAttempt(
                    "managed output state poisoned".into(),
                ));
            }
        };
        if state.uncertain {
            return Err(ManagedOutputWriteErrorV3::AfterAttempt(
                "managed output prior mutation uncertain".into(),
            ));
        }
        if state.closed.is_some() {
            return Err(ManagedOutputWriteErrorV3::BeforeWrite(
                "managed output write admission closed".into(),
            ));
        }
        if !state.authorized {
            return Err(ManagedOutputWriteErrorV3::BeforeWrite(
                "managed output not authorized".into(),
            ));
        }
        if Instant::now() >= self.work_until {
            return Err(ManagedOutputWriteErrorV3::BeforeWrite(
                "collection work deadline expired".into(),
            ));
        }
        let mut attempted = false;
        let result = (|| -> std::io::Result<()> {
            state.observation.verify_live(self.deadline)?;
            if state.chunks.is_none() {
                let directory =
                    Dir::from_std_file(state.observation.clone_original_output(self.deadline)?);
                state.observation.verify_live(self.deadline)?;
                if Instant::now() >= self.work_until {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::TimedOut,
                        "collection work deadline expired",
                    ));
                }
                attempted = true;
                state.uncertain = true;
                directory.create_dir("chunks")?;
                let chunks = directory.open_dir_nofollow("chunks")?;
                state.observation.verify_live(self.deadline)?;
                directory.try_clone()?.into_std_file().sync_all()?;
                state.observation.verify_live(self.deadline)?;
                state.directory = Some(directory);
                state.chunks = Some(chunks);
            }
            state.observation.verify_live(self.deadline)?;
            state.check_chunks_binding()?;
            if !attempted && Instant::now() >= self.work_until {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::TimedOut,
                    "collection work deadline expired",
                ));
            }
            attempted = true;
            state.uncertain = true;
            let mut options = cap_std::fs::OpenOptions::new();
            options
                .write(true)
                .create_new(true)
                .follow(FollowSymlinks::No);
            let mut file = state
                .chunks
                .as_ref()
                .expect("chunks established")
                .open_with(leaf, &options)?;
            state.observation.verify_live(self.deadline)?;
            state.check_chunks_binding()?;
            state.check_chunk_leaf_binding(&file, leaf)?;
            file.write_all(bytes)?;
            state.observation.verify_live(self.deadline)?;
            state.check_chunks_binding()?;
            state.check_chunk_leaf_binding(&file, leaf)?;
            file.sync_all()?;
            state.observation.verify_live(self.deadline)?;
            state.check_chunks_binding()?;
            state
                .chunks
                .as_ref()
                .expect("chunks established")
                .try_clone()?
                .into_std_file()
                .sync_all()?;
            state.observation.verify_live(self.deadline)?;
            state.check_chunks_binding()?;
            state.check_chunk_leaf_binding(&file, leaf)?;
            Ok(())
        })();
        match result {
            Ok(()) => {
                let Some(completed) = state.completed_writes.checked_add(1) else {
                    return Err(ManagedOutputWriteErrorV3::AfterAttempt(
                        "managed output write count overflow".into(),
                    ));
                };
                state.completed_writes = completed;
                state.uncertain = false;
                Ok(())
            }
            Err(error) if attempted => {
                Err(ManagedOutputWriteErrorV3::AfterAttempt(error.to_string()))
            }
            Err(error) => Err(ManagedOutputWriteErrorV3::BeforeWrite(error.to_string())),
        }
    }

    /// Publish the finished writer through this same live output owner. The
    /// original game guard is supplied by the game owner, never reconstructed
    /// here. Partial output and the caller's original finished writer remain
    /// retained on any failure; this proves neither lease nor GPU completion.
    pub(crate) fn checked_publish_corpus<S>(
        self: &Arc<Self>,
        finished: &crate::rovent_collect::writer_lock_v3::FinishedRoventCorpusLockV3,
        pipeline: &crate::rovent_collect::policy_row_pipeline_v3::PolicyRowPipelineOwnerV3,
        game_guard: &mut crate::rovent_collect::game_authority_v3::GamePublishGuardV3<S>,
        until: Instant,
        cancel: &std::sync::atomic::AtomicBool,
    ) -> Result<
        rz_experiments::rovent_corpus_lock_v3::publication::RoventLockV3Publication,
        ManagedOutputWriteErrorV3,
    > {
        use crate::rovent_collect::publication::RoventPublicationHookPhaseV3;
        match game_guard.try_matches_output_pipeline(self, pipeline) {
            Ok(Some(true)) => {}
            Ok(None) => return Err(ManagedOutputWriteErrorV3::Busy),
            Ok(Some(false)) => {
                return Err(ManagedOutputWriteErrorV3::BeforeWrite(
                    "managed publication original game or pipeline differs".into(),
                ));
            }
            Err(error) => return Err(ManagedOutputWriteErrorV3::AuthorityUnknown(error.into())),
        }
        let game_origin = game_guard.original_game_identity();
        let deadline = until.min(self.deadline);
        let mut state = match self.state.try_lock() {
            Ok(state) => state,
            Err(std::sync::TryLockError::WouldBlock) => {
                return Err(ManagedOutputWriteErrorV3::Busy);
            }
            Err(std::sync::TryLockError::Poisoned(_)) => {
                return Err(ManagedOutputWriteErrorV3::AfterAttempt(
                    "managed output state poisoned".into(),
                ));
            }
        };
        if state.uncertain || state.publication_attempted {
            return Err(ManagedOutputWriteErrorV3::AfterAttempt(
                "managed publication already attempted or uncertain".into(),
            ));
        }
        let unstarted = |reason: String| ManagedOutputWriteErrorV3::PublicationBeforeWrite {
            reason,
            evidence: ManagedPublicationUnstartedEvidenceV3 {
                output: Arc::clone(self),
                game: game_origin,
            },
        };
        if Instant::now() >= self.work_until {
            return Err(unstarted("collection work deadline expired before publication".into()));
        }
        if !finished.matches_original_manifest(self.inputs.manifest()) {
            return Err(unstarted(
                "managed publication original writer differs".into(),
            ));
        }
        if !state
            .original_writer
            .as_ref()
            .is_some_and(|writer| finished.matches_original_writer(writer))
        {
            return Err(unstarted(
                "managed publication original writer owner differs".into(),
            ));
        }
        if !state.authorized || state.closed.is_some() {
            return Err(unstarted(
                "managed publication not authorized or closed".into(),
            ));
        }
        let directory = match state.directory.as_ref() {
            Some(directory) => match directory.try_clone() {
                Ok(directory) => directory,
                Err(error) => return Err(unstarted(error.to_string())),
            },
            None => {
                return Err(unstarted(
                    "managed publication original output directory missing".into(),
                ));
            }
        };
        let result = finished.publish_atomic_managed_with_authorize(
            &directory,
            deadline,
            cancel,
            || match game_guard.try_grant() {
                crate::rovent_collect::game_authority_v3::GameStepV3::Ready => Ok(()),
                step => Err(invalid(format!(
                    "managed publication original game authority: {step:?}"
                ))),
            },
            |event, phase| {
                state.observation.verify_live(deadline).map_err(io)?;
                state.check_chunks_binding().map_err(io)?;
                if phase == RoventPublicationHookPhaseV3::Before && event.is_mutation() {
                    if !state.publication_attempted && Instant::now() >= self.work_until {
                        return Err(invalid("collection work deadline expired before publication"));
                    }
                    state.publication_attempted = true;
                    state.uncertain = true;
                }
                Ok(())
            },
        );
        match result {
            Ok(receipt) => {
                state.uncertain = false;
                Ok(receipt)
            }
            Err(error) if state.publication_attempted => {
                Err(ManagedOutputWriteErrorV3::AfterAttempt(error.to_string()))
            }
            Err(error) => Err(unstarted(error.to_string())),
        }
    }
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum CollectionInputsPurposeV3 {
    Pc,
    Own,
}

impl PreparedPcCollectionInputsV3 {
    /// Build the original owner from the same /2-inherited directory and plan
    /// descriptors. This is not native lock acceptance or output permission.
    pub(super) fn open_from_managed_observation(
        observation: &mut super::managed_output_fd_v2::ObservedManagedPlanLeaseV2,
        plan_pin: &RoventPublicationFile,
        deadline: Instant,
    ) -> Result<Self, ArenaError> {
        check(
            observation.matches_plan(&plan_pin.path, plan_pin.bytes, &plan_pin.sha256),
            "Collection /3 original observed plan differs",
        )?;
        observation.verify_live(deadline).map_err(io)?;
        let (original_directory, original_plan_file) =
            observation.clone_original_input().map_err(io)?;
        let directory = Dir::from_std_file(original_directory.try_clone().map_err(io)?);
        let original_plan = OriginalPinnedFile::from_original_plan(
            original_plan_file,
            plan_pin,
            PLAN_MAX_BYTES,
            deadline,
        )?;
        let text = std::str::from_utf8(&original_plan.bytes)
            .map_err(|_| invalid("Collection /3 plan UTF8"))?;
        let plan: RoventCollectionLockV3LaunchPlan = rz_experiments::decode_json(text)?;
        check(
            plan.schema == ROVENT_COLLECTION_LOCK_V3_LAUNCH_PLAN,
            "Collection /3 explicit launch plan domain",
        )?;
        let original_manifest =
            OriginalPinnedFile::read(&directory, &plan.manifest, MANIFEST_MAX_BYTES)?;
        let text = std::str::from_utf8(&original_manifest.bytes)
            .map_err(|_| invalid("Collection /3 manifest UTF8"))?;
        let manifest = Arc::new(RoventCorpusManifestLockV3::from_json(text)?);
        observation.verify_live(deadline).map_err(io)?;
        Ok(Self {
            original_directory,
            original_plan,
            original_manifest,
            plan,
            manifest,
            purpose: CollectionInputsPurposeV3::Pc,
            managed_binding: Some(observation.original_binding()),
        })
    }
    pub fn open(directory: &Dir, plan_pin: &RoventPublicationFile) -> Result<Self, ArenaError> {
        Self::open_for(directory, plan_pin, CollectionInputsPurposeV3::Pc)
    }

    /// Read an explicitly Own-domain plan while retaining the same original
    /// directory, file descriptors and byte pins used by later validation.
    pub fn open_own(directory: &Dir, plan_pin: &RoventPublicationFile) -> Result<Self, ArenaError> {
        Self::open_for(directory, plan_pin, CollectionInputsPurposeV3::Own)
    }

    fn open_for(
        directory: &Dir,
        plan_pin: &RoventPublicationFile,
        purpose: CollectionInputsPurposeV3,
    ) -> Result<Self, ArenaError> {
        let original_directory = directory.try_clone().map_err(io)?.into_std_file();
        let original_plan = OriginalPinnedFile::read(directory, plan_pin, PLAN_MAX_BYTES)?;
        let text = std::str::from_utf8(&original_plan.bytes)
            .map_err(|_| invalid("Collection /3 plan UTF8"))?;
        let plan: RoventCollectionLockV3LaunchPlan = rz_experiments::decode_json(text)?;
        check(
            plan.schema
                == match purpose {
                    CollectionInputsPurposeV3::Pc => ROVENT_COLLECTION_LOCK_V3_LAUNCH_PLAN,
                    CollectionInputsPurposeV3::Own => ROVENT_OWN_COLLECTION_LOCK_V3_LAUNCH_PLAN,
                },
            "Collection /3 explicit launch plan domain",
        )?;
        let original_manifest =
            OriginalPinnedFile::read(directory, &plan.manifest, MANIFEST_MAX_BYTES)?;
        let text = std::str::from_utf8(&original_manifest.bytes)
            .map_err(|_| invalid("Collection /3 manifest UTF8"))?;
        let manifest = Arc::new(RoventCorpusManifestLockV3::from_json(text)?);
        Ok(Self {
            original_directory,
            original_plan,
            original_manifest,
            plan,
            manifest,
            purpose,
            managed_binding: None,
        })
    }

    pub fn plan(&self) -> &RoventCollectionLockV3LaunchPlan {
        &self.plan
    }
    pub fn manifest(&self) -> &Arc<RoventCorpusManifestLockV3> {
        &self.manifest
    }
    pub fn original_plan_bytes(&self) -> &[u8] {
        &self.original_plan.bytes
    }
    pub fn original_manifest_bytes(&self) -> &[u8] {
        &self.original_manifest.bytes
    }
    pub fn original_plan_pin(&self) -> &RoventPublicationFile {
        &self.original_plan.pin
    }
    pub fn original_manifest_pin(&self) -> &RoventPublicationFile {
        &self.original_manifest.pin
    }

    /// Exact declared resource-plan pin. The original supervisor still opens
    /// and validates its own resource owner before calling lock admission.
    pub fn resource_plan_pin(
        &self,
    ) -> Result<crate::rovent_resource_scope::launch::RoventNativeResourcePlanPin, ArenaError> {
        let pin = &self.plan.resource_plan;
        pin.validate()?;
        check(
            pin.path.starts_with("inputs/") && valid_relative(&pin.path),
            "Collection /3 original resource-plan path",
        )?;
        Ok(
            crate::rovent_resource_scope::launch::RoventNativeResourcePlanPin {
                path: PathBuf::from(&pin.path),
                sha256: pin.sha256.clone(),
                bytes: pin.bytes,
            },
        )
    }

    /// Validate against the same /3 lock supplied by native resource
    /// admission. This registers no capability and starts no worker.
    pub fn validate_against(&self, lock: &RoventInputLockV3) -> Result<(), ArenaError> {
        self.validate_for(lock, CollectionInputsPurposeV3::Pc)
    }

    /// The Own-only path is distinct from the existing P-C validation and
    /// requires the original lock's actual Own researcher declaration.
    pub fn validate_own_against(&self, lock: &RoventInputLockV3) -> Result<(), ArenaError> {
        self.validate_for(lock, CollectionInputsPurposeV3::Own)
    }

    fn validate_for(
        &self,
        lock: &RoventInputLockV3,
        purpose: CollectionInputsPurposeV3,
    ) -> Result<(), ArenaError> {
        check(
            self.purpose == purpose,
            "Collection /3 plan purpose differs",
        )?;
        self.manifest.validate(lock)?;
        self.plan.resource_plan.validate()?;
        let mode_matches = match purpose {
            CollectionInputsPurposeV3::Pc => !lock.manifest.engine.mode.has_native_r(),
            CollectionInputsPurposeV3::Own => {
                lock.manifest.engine.mode.has_native_r()
                    && matches!(
                        &lock.manifest.engine.researcher,
                        Some(RoventResearcher::Own { .. })
                    )
            }
        };
        check(
            lock.manifest.purpose == RoventPurpose::Collection
                && mode_matches
                && self.plan.resource_plan.path.starts_with("inputs/")
                && valid_relative(&self.plan.resource_plan.path)
                && (1..=64).contains(&self.plan.maximum_roots)
                && (1..=self.plan.maximum_roots).contains(&self.plan.maximum_root_workers)
                && (1..=self.plan.maximum_root_workers)
                    .contains(&self.plan.maximum_simultaneous_teacher_roots)
                && self.plan.teacher_mailbox_capacity > 0
                && self.plan.teacher_mailbox_capacity
                    <= rz_experiments::rovent_corpus::MAX_CORPUS_ROWS as u32
                && self.plan.teacher_tickets_per_root > 0
                && self.plan.teacher_tickets_per_root
                    <= rz_experiments::rovent_corpus::MAX_CORPUS_ROWS as u32
                && (1..=rz_experiments::rovent_corpus::MAX_CORPUS_ROWS as u64)
                    .contains(&self.plan.maximum_records)
                && (1..=30_000).contains(&self.plan.publication_timeout_ms)
                && self.plan.output_directory.is_absolute()
                && self.plan.output_directory.components().all(|part| {
                    matches!(
                        part,
                        Component::RootDir | Component::Prefix(_) | Component::Normal(_)
                    )
                })
                && !self.plan.teacher_slots.is_empty()
                && self.plan.teacher_slots.len() == self.manifest.registration.teachers.len()
                && self.plan.teacher_sources.len() == self.manifest.registration.teachers.len(),
            "Collection /3 finite plan, mode or teacher coverage",
        )?;
        check(
            Duration::from_nanos(self.plan.teacher_soft_quantum_ns)
                >= rz_search::external_cpu::EXTERNAL_MIN_QUANTUM
                && self.plan.teacher_soft_quantum_ns <= 1_000_000_000,
            "Collection /3 teacher soft quantum",
        )?;
        match self.plan.teacher_constraint {
            ExternalGoConstraint::Depth { depth } => check(depth > 0, "teacher depth")?,
            ExternalGoConstraint::Nodes { reported_nodes } => {
                check(reported_nodes > 0, "teacher nodes")?
            }
            ExternalGoConstraint::Combined {
                depth,
                reported_nodes,
            } => check(depth > 0 && reported_nodes > 0, "teacher combined limit")?,
            ExternalGoConstraint::MoveTime { milliseconds } => {
                check(milliseconds > 0, "teacher time")?
            }
        }
        check(
            generation_identifier(&self.plan.metadata.game_id)
                && generation_identifier(&self.plan.metadata.opening_id)
                && generation_identifier(&self.plan.metadata.line_genealogy_id)
                && self
                    .manifest
                    .registration
                    .opening_splits
                    .contains_key(&self.plan.metadata.opening_id),
            "Collection /3 metadata/opening not registered",
        )?;
        let mut actors = BTreeSet::new();
        let mut indices = BTreeSet::new();
        let mut owners = BTreeSet::new();
        for slot in &self.plan.teacher_slots {
            check(
                self.manifest
                    .registration
                    .teachers
                    .contains_key(&slot.actor_id)
                    && actors.insert(&slot.actor_id)
                    && indices.insert(slot.source_index)
                    && slot.collection_owner > 0
                    && owners.insert(slot.collection_owner)
                    && slot.source_index < self.plan.teacher_sources.len()
                    && self.plan.teacher_sources[slot.source_index].index == slot.source_index,
                "Collection /3 actor/source/owner coverage",
            )?;
            let teacher = &self.manifest.registration.teachers[&slot.actor_id];
            let source = lock
                .manifest
                .engine
                .teachers
                .get(slot.source_index)
                .ok_or_else(|| invalid("Collection /3 teacher source not in original lock"))?;
            check(
                source.source_commit == teacher.source_commit
                    && source.binary.sha256 == teacher.binary_sha256
                    && source.binary.bytes == teacher.binary_bytes
                    && source.profile.sha256 == teacher.profile_sha256
                    && source.canonical_sha256 == teacher.canonical_sha256
                    && source.capability_sha256 == teacher.capability_sha256,
                "Collection /3 actor maps to a different original teacher source",
            )?;
        }
        let requested_threads = lock
            .manifest
            .engine
            .teachers
            .iter()
            .try_fold(0u32, |sum, teacher| {
                sum.checked_add(teacher.requested_threads)
            })
            .ok_or_else(|| invalid("Collection /3 teacher thread total overflow"))?;
        let simultaneous_threads = requested_threads
            .checked_mul(u32::from(self.plan.maximum_simultaneous_teacher_roots))
            .ok_or_else(|| invalid("Collection /3 simultaneous teacher threads overflow"))?;
        check(
            lock.manifest.resources.pools.iter().any(|pool| {
                pool.actors.contains(&RoventActor::Teacher)
                    && pool.max_threads >= simultaneous_threads
            }),
            "Collection /3 teacher workers exceed the original physical pool",
        )?;
        Ok(())
    }

    /// Compare the original open directory owner with the resource callback's
    /// original directory; equal path strings or equal hashes do not establish
    /// that both consumers accessed the same directory object.
    pub fn validate_original_boundary(
        &self,
        lock: &RoventInputLockV3,
        directory: &Dir,
        original_lock_file: &std::fs::File,
    ) -> Result<(), ArenaError> {
        self.validate_boundary_for(
            lock,
            directory,
            original_lock_file,
            CollectionInputsPurposeV3::Pc,
        )
    }

    /// Own-specific original boundary verification; a PC-domain plan cannot
    /// be relabelled as Own even when the serialized fields otherwise match.
    pub fn validate_original_own_boundary(
        &self,
        lock: &RoventInputLockV3,
        directory: &Dir,
        original_lock_file: &std::fs::File,
    ) -> Result<(), ArenaError> {
        self.validate_boundary_for(
            lock,
            directory,
            original_lock_file,
            CollectionInputsPurposeV3::Own,
        )
    }

    fn validate_boundary_for(
        &self,
        lock: &RoventInputLockV3,
        directory: &Dir,
        original_lock_file: &std::fs::File,
        purpose: CollectionInputsPurposeV3,
    ) -> Result<(), ArenaError> {
        let held = self.original_directory.metadata().map_err(io)?;
        let supplied = directory
            .try_clone()
            .map_err(io)?
            .into_std_file()
            .metadata()
            .map_err(io)?;
        let lock_file = original_lock_file.metadata().map_err(io)?;
        check(
            held.is_dir()
                && supplied.is_dir()
                && (held.dev(), held.ino()) == (supplied.dev(), supplied.ino())
                && lock_file.is_file(),
            "Collection /3 original resource boundary differs",
        )?;
        self.validate_for(lock, purpose)
    }

    pub(crate) fn roster(&self) -> Vec<TeacherRosterEntryV3> {
        self.plan
            .teacher_slots
            .iter()
            .map(|slot| TeacherRosterEntryV3 {
                actor_id: slot.actor_id.clone(),
                source_index: slot.source_index,
                collection_owner: slot.collection_owner,
            })
            .collect()
    }
}

#[cfg(test)]
mod cli_tests {
    use super::*;

    #[test]
    fn v3_selector_keeps_native_arguments_and_rejects_mixed_or_duplicate_inputs() {
        let digest = format!("{:x}", Sha256::digest(b"plan input"));
        let arguments = vec![
            "--v3-mode=pc".to_owned(),
            "--v3-plan=inputs/plan.json".to_owned(),
            format!("--v3-plan-sha256={digest}"),
            "--v3-plan-bytes=1".to_owned(),
            "--process-epoch=9".to_owned(),
        ];
        let parsed = parse_pc_collection_cli_v3(arguments.clone())
            .unwrap()
            .unwrap();
        assert_eq!(parsed.native_arguments, ["--process-epoch=9"]);
        let mut mixed = arguments.clone();
        mixed.push("--v2-mode=pc".into());
        assert!(parse_pc_collection_cli_v3(mixed).is_err());
        let mut duplicate = arguments;
        duplicate.push("--v3-mode=pc".into());
        assert!(parse_pc_collection_cli_v3(duplicate).is_err());
        assert!(
            parse_pc_collection_cli_v3(vec!["--process-epoch=9".into()])
                .unwrap()
                .is_none()
        );
    }
}
