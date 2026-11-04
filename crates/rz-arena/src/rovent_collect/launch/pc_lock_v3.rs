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
struct OriginalPinnedFile {
    _file: cap_std::fs::File,
    bytes: Vec<u8>,
    pin: RoventPublicationFile,
}

impl OriginalPinnedFile {
    fn read(
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
            launch_claimed: std::sync::atomic::AtomicBool::new(false),
        })
    }

    pub(crate) fn inputs(&self) -> &Arc<PreparedPcCollectionInputsV3> {
        &self.inputs
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
        let control = self
            .bridge
            .startup()
            .ok_or("original startup control missing")?;
        let proof = issuer.issue(gate, control)?;
        proof.with_start_lease(gate, &self.native, control, |lease| {
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
    /// This key names the retained original owner; it is not completion.
    pub(crate) fn recovery_key(&self) -> super::pc_recovery_v3::PcRunRecoveryKeyV3 {
        self.recovery.clone()
    }

    /// Nonblocking progress for this original run. Pre-admission driver
    /// retirement is independent of native/resource/output settlement. A
    /// true result means settled owners are retained in the original slot.
    pub(crate) fn try_settle(&mut self) -> Result<bool, &'static str> {
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
            PcDriverPollV3::Pending | PcDriverPollV3::Ready(true) => return Ok(false),
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
    deadline: Instant,
}

struct ManagedOutputStateV3 {
    observation: super::managed_output_fd_v2::ObservedManagedPlanLeaseV2,
    authorized: bool,
    directory: Option<Dir>,
    chunks: Option<Dir>,
    uncertain: bool,
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
    AfterAttempt(String),
}

#[allow(dead_code)]
impl ManagedOutputOwnerV3 {
    /// `deadline` is the original collection cleanup ceiling, not the
    /// shorter startup deadline; expiry preserves any unfinished packet.
    pub(super) fn from_observation(
        inputs: Arc<PreparedPcCollectionInputsV3>,
        observation: super::managed_output_fd_v2::ObservedManagedPlanLeaseV2,
        deadline: Instant,
    ) -> Result<
        Self,
        (
            ArenaError,
            Arc<PreparedPcCollectionInputsV3>,
            super::managed_output_fd_v2::ObservedManagedPlanLeaseV2,
        ),
    > {
        if Instant::now() >= deadline
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
            }),
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
        if state.authorized || state.uncertain {
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
        if !state.authorized {
            return Err(ManagedOutputWriteErrorV3::BeforeWrite(
                "managed output not authorized".into(),
            ));
        }
        let mut attempted = false;
        let result = (|| -> std::io::Result<()> {
            state.observation.verify_live(self.deadline)?;
            if state.chunks.is_none() {
                let directory =
                    Dir::from_std_file(state.observation.clone_original_output(self.deadline)?);
                state.observation.verify_live(self.deadline)?;
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
                state.uncertain = false;
                Ok(())
            }
            Err(error) if attempted => {
                Err(ManagedOutputWriteErrorV3::AfterAttempt(error.to_string()))
            }
            Err(error) => Err(ManagedOutputWriteErrorV3::BeforeWrite(error.to_string())),
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
