//! Opt-in storage admission and original one-shot setup custody.
//!
//! The original lock, verified profile/source descriptors and actual directory
//! capability stay together. This starts no model, worker or Store archive IO.
//! CLI/session attachment retains these same owners. Archive setup is deferred
//! until an admitted go, before registering its root. Byte admission and the
//! checked-compiled conversion do not attest worker, native-root or physical
//! completion readiness.

use crate::rovent_bootstrap::RoventStartupControl;
use crate::rovent_bootstrap::process_boundary::{
    PROCESS_SOURCE_SCOPE, ProcessBoundaryError, VerifiedOwnedRoventProcessBoundary,
    VerifiedRoventProcessBoundary, compiled_process_source_components,
};
use crate::{engine::ProcessClock, rovent_root_admission::RoventRootAdmission};
use cap_fs_ext::DirExt;
use cap_std::fs::Dir;
use rz_contracts::{ContractError, Digest};
use rz_experiments::{
    ArtifactRef, CompiledSourceInventoryV5, ManifestError, RoventInputLock, RoventStorageProfile,
};
use rz_runtime::contracts::ContractClock;
use rz_search::pals::engine::idle_archive::{
    UnusedIdleArchiveSetup, UnusedIdleArchiveSetupControl, UnusedIdleArchiveSetupIdentity,
};
use rz_search::pals::store::{
    ArchiveConfig, ArchiveRootCapability, ArchiveRuntimeLimits, StoreError,
};
use sha2::{Digest as _, Sha256};
use std::{
    collections::BTreeMap,
    error::Error,
    fmt,
    fs::File,
    path::{Component, Path, PathBuf},
    sync::{Arc, atomic::Ordering},
    time::Instant,
};

// Reuse the existing persisted-metadata bound, not an archive runtime selector.
const MAX_INPUT_BYTES: u64 = rz_experiments::MAX_MANIFEST_BYTES as u64;
const MAX_ROOT_PATH_BYTES: usize = 4096;
const ARCHIVE: usize = 0;
const CHECKOUT: usize = 1;
const PRIVATE: usize = 2;
const PROTECTED: usize = 3;
// These are supported native-bridge limits, not codec/default selectors.
const NATIVE_MAINTENANCE_IO_BYTES: u64 = 16 * 1024 * 1024;
const RUNTIME_REGISTRATION_DOMAIN: &[u8] = b"rz-rovent-storage-runtime-registration/1";
// Role order is the profile's source/codec/launch order, never its snapshot
// artifact path. Snapshot paths remain inputs/... and do not become repo paths.
const COMPILED_STORAGE_SOURCE_PATHS: [&str; 3] = [
    "crates/rz-search/src/pals/store/archive.rs",
    "crates/rz-experiments/src/rovent_storage.rs",
    "crates/rz-uci/src/rovent_bootstrap/launch/storage.rs",
];

/// Explicit borrowed invocation inputs. No path, raw pin or deadline defaults.
pub struct RoventStorageAdmissionRequest<'a> {
    pub profile_path: &'a str,
    pub profile_sha256: &'a str,
    pub profile_bytes: u64,
    pub archive_root: &'a Path,
    pub checkout_root: &'a Path,
    pub private_snapshot_root: &'a Path,
    pub protected_execution_root: &'a Path,
}

/// Read-only admission custody, not a physical fence or an enabled archive.
/// This owner is intentionally not Clone. Partial failures retain the same
/// original successful descriptor/directory opens rather than reopening them.
pub struct RoventStorageAdmissionOwners {
    input_lock: Arc<RoventInputLock>,
    control: RoventStartupControl,
    artifact: Option<ArtifactRef>,
    profile: Option<RoventStorageProfile>,
    profile_file: Option<VerifiedInput>,
    source_files: [Option<VerifiedInput>; 3],
    directories: [Option<HeldDirectory>; 4],
    capability: Option<ArchiveRootCapability>,
}

impl RoventStorageAdmissionOwners {
    pub fn input_lock(&self) -> &Arc<RoventInputLock> {
        &self.input_lock
    }
    pub fn profile_artifact(&self) -> Option<&ArtifactRef> {
        self.artifact.as_ref()
    }
    pub fn profile(&self) -> Option<&RoventStorageProfile> {
        self.profile.as_ref()
    }
    pub fn profile_file(&self) -> Option<&File> {
        self.profile_file.as_ref().map(|held| &held.file)
    }
    /// 0=storage source, 1=codec, 2=launch. An absent file is never verified.
    pub fn source_file(&self, index: usize) -> Option<&File> {
        self.source_files
            .get(index)
            .and_then(Option::as_ref)
            .map(|held| &held.file)
    }
    pub fn verified_source_files(&self) -> usize {
        self.source_files
            .iter()
            .flatten()
            .filter(|held| held.revalidated)
            .count()
    }
    pub fn retains_root_capability(&self) -> bool {
        self.capability.is_some()
    }
}

/// Move-only byte/directory custody only, with compiled namespace still
/// unverified. This cannot configure/start an archive: a separate Root-owned
/// final compiled/source comparison and checked runtime bridge are required.
/// No such runtime bridge is provided by this independent library entry.
pub struct RoventStorageByteCustody {
    owners: RoventStorageAdmissionOwners,
}

impl RoventStorageByteCustody {
    pub fn owners(&self) -> &RoventStorageAdmissionOwners {
        &self.owners
    }
    pub fn input_lock_sha256(&self) -> &str {
        &self.owners.input_lock.canonical_sha256
    }
    pub fn profile(&self) -> &RoventStorageProfile {
        self.owners.profile.as_ref().expect("admitted profile")
    }
    pub fn retains_root_capability(&self) -> bool {
        self.owners.capability.is_some()
    }
    /// Recheck only within the retained original startup control. This cannot
    /// renew admission or replace Search's later per-operation root checks.
    pub fn revalidate_startup(&self) -> Result<(), RoventStorageAdmissionError> {
        check(&self.owners.control)?;
        for input in self
            .owners
            .profile_file
            .iter()
            .chain(self.owners.source_files.iter().flatten())
        {
            read_rechecked(input, &self.owners.control, false)?;
        }
        for directory in self.owners.directories.iter().flatten() {
            directory.revalidate(&self.owners.control)?;
        }
        require_separation(&self.owners)?;
        check(&self.owners.control)
    }
    pub fn into_owners(self) -> RoventStorageAdmissionOwners {
        self.owners
    }
}

/// Move-only source/byte custody tied to the actual opaque process observation.
/// The borrowed observation keeps its original binary and private-root FDs
/// alive. This is not archive execution admission: no capability is exposed,
/// worker is started, Store is configured, or physical fence is produced.
pub struct RoventStorageCompiledCustody<'boundary, 'process> {
    custody: RoventStorageByteCustody,
    boundary: &'boundary VerifiedRoventProcessBoundary<'process>,
}

impl RoventStorageCompiledCustody<'_, '_> {
    pub fn owners(&self) -> &RoventStorageAdmissionOwners {
        self.custody.owners()
    }
    pub fn input_lock_sha256(&self) -> &str {
        self.custody.input_lock_sha256()
    }
    pub fn profile(&self) -> &RoventStorageProfile {
        self.custody.profile()
    }
    pub fn retains_root_capability(&self) -> bool {
        self.custody.retains_root_capability()
    }
    /// Reuses the original startup cancel/deadline and held descriptors. This
    /// does not renew process admission or authorize a later archive command.
    pub fn revalidate_startup(&self) -> Result<(), RoventStorageCompiledError> {
        verify_compiled_custody(&self.custody, self.boundary)
    }
    /// Explicitly discard compiled proof while keeping the exact byte owners.
    /// No capability or public constructor can turn these owners back into a
    /// checked value without the original process-boundary comparison.
    pub fn into_byte_custody(self) -> RoventStorageByteCustody {
        self.custody
    }
}

/// Original registration FD custody retained by the launcher, not the archive
/// worker. The caller MUST retain this owner through setup completion and the
/// actual worker join/physical shutdown. Dropping it attests neither event.
///
/// The clock/implementation captured here are expected namespace selections,
/// not authority: only the later opaque original RoventRootAdmission can admit
/// the first go. No CLI/session currently creates or consumes this entry.
pub struct RoventStorageRuntimeRegistration<'boundary, 'process> {
    custody: RoventStorageCompiledCustody<'boundary, 'process>,
    state: StorageRuntimeState,
}

struct StorageRuntimeState {
    expected_clock: ProcessClock,
    expected_implementation: Digest,
    identity: UnusedIdleArchiveSetupIdentity,
    issued_admission: Option<Arc<RoventRootAdmission>>,
}

impl RoventStorageRuntimeRegistration<'_, '_> {
    pub fn owners(&self) -> &RoventStorageAdmissionOwners {
        self.custody.owners()
    }
    pub fn profile(&self) -> &RoventStorageProfile {
        self.custody.profile()
    }
    /// Attribution only; this digest is not a worker/physical completion proof.
    pub fn identity(&self) -> UnusedIdleArchiveSetupIdentity {
        self.state.identity
    }
    pub fn issued_admission(&self) -> Option<&Arc<RoventRootAdmission>> {
        self.state.issued_admission.as_ref()
    }
    pub fn retains_root_capability(&self) -> bool {
        self.custody.retains_root_capability()
    }
    /// Take the original capability once, after every fallible validation.
    /// Failure leaves this exact registration/capability with the caller and
    /// returns its exact admission. Success keeps all other original FD owners
    /// here and retains that admission; this owner must outlive worker closure.
    ///
    /// No startup deadline is renewed/reused here. The finite control is copied
    /// only from an existing live original-go admission. Search still checks
    /// its actual unused Idle/new_game_ready seal on submission. This makes no
    /// worker, performs no archive IO and does not claim physical readiness.
    pub fn take_unused_idle_setup(
        &mut self,
        admission: Arc<RoventRootAdmission>,
    ) -> Result<UnusedIdleArchiveSetup, RoventStorageSetupRejection> {
        split_unused_setup(&mut self.custody.custody, &mut self.state, admission)
    }
}

/// Lifetime-free runtime registration retaining the same original FD owners
/// and actual opaque process observation. No descriptor is reopened or cloned
/// in the conversion, and the borrowed proof is not stored self-referentially.
///
/// This is a library ownership boundary only: CLI/static drivers do not yet
/// retain it. Callers MUST keep this owner through setup completion and actual
/// worker join, including unknown/quarantine shutdown. Drop issues no physical
/// completion and does not return a capability already moved to Search.
pub struct RoventStorageOwnedRuntimeRegistration {
    custody: RoventStorageByteCustody,
    boundary: Arc<VerifiedOwnedRoventProcessBoundary>,
    state: StorageRuntimeState,
}

impl RoventStorageOwnedRuntimeRegistration {
    /// Namespace/owner selection only. This neither renews startup nor admits
    /// a go; setup still requires the original opaque live root admission.
    pub(crate) fn validate_session_attachment(
        &self,
        lock: &RoventInputLock,
        clock: &ProcessClock,
        implementation: Digest,
    ) -> Result<(), RoventStorageRuntimeError> {
        let original = self.state.expected_clock.runtime_clock();
        let actual = clock.runtime_clock();
        if !std::ptr::eq(self.custody.owners.input_lock.as_ref(), lock)
            || clock.epoch() != self.state.expected_clock.epoch()
            || actual.domain() != original.domain()
            || actual.origin() != original.origin()
            || implementation != self.state.expected_implementation
        {
            return Err(RoventStorageRuntimeError::Namespace(
                "session and original storage lock/clock/implementation differ",
            ));
        }
        Ok(())
    }

    pub fn owners(&self) -> &RoventStorageAdmissionOwners {
        self.custody.owners()
    }
    pub fn profile(&self) -> &RoventStorageProfile {
        self.custody.profile()
    }
    pub fn process_boundary(&self) -> &Arc<VerifiedOwnedRoventProcessBoundary> {
        &self.boundary
    }
    /// Attribution only; neither this nor the owned process proof is a fence.
    pub fn identity(&self) -> UnusedIdleArchiveSetupIdentity {
        self.state.identity
    }
    pub fn issued_admission(&self) -> Option<&Arc<RoventRootAdmission>> {
        self.state.issued_admission.as_ref()
    }
    pub fn retains_root_capability(&self) -> bool {
        self.custody.retains_root_capability()
    }
    /// Same one-shot split and independently admitted original go as the
    /// borrowed registration. No startup window is renewed or used as go
    /// authority. The original non-capability FD owners stay here.
    pub fn take_unused_idle_setup(
        &mut self,
        admission: Arc<RoventRootAdmission>,
    ) -> Result<UnusedIdleArchiveSetup, RoventStorageSetupRejection> {
        split_unused_setup(&mut self.custody, &mut self.state, admission)
    }
}

/// A failed ownership conversion returns both exact original owners. Nothing
/// is enabled, consumed or substituted with another lock/descriptor/proof.
pub struct RoventStorageRuntimeOwnershipFailure<'boundary, 'process> {
    pub error: RoventStorageRuntimeError,
    pub registration: RoventStorageRuntimeRegistration<'boundary, 'process>,
    pub candidate: Arc<VerifiedOwnedRoventProcessBoundary>,
}
impl fmt::Debug for RoventStorageRuntimeOwnershipFailure<'_, '_> {
    fn fmt(&self, output: &mut fmt::Formatter<'_>) -> fmt::Result {
        output
            .debug_struct("RoventStorageRuntimeOwnershipFailure")
            .field("error", &self.error)
            .field(
                "root_capability_retained",
                &self.registration.retains_root_capability(),
            )
            .finish_non_exhaustive()
    }
}

/// Transfer only from a view issued by this exact owned actual boundary and
/// using the same Arc lock and original startup control. A separately observed
/// process with equal bytes/digests, or another File/Dir handle to the same
/// inode, is rejected. This must complete during the original startup window;
/// later setup still requires its own opaque original-go admission.
// Preserve complete partial FD custody directly on error, with no extra Box
// allocation just to reduce the error's representation.
#[allow(clippy::result_large_err)]
pub fn own_rovent_storage_runtime_registration<'boundary, 'process>(
    registration: RoventStorageRuntimeRegistration<'boundary, 'process>,
    candidate: Arc<VerifiedOwnedRoventProcessBoundary>,
) -> Result<
    RoventStorageOwnedRuntimeRegistration,
    RoventStorageRuntimeOwnershipFailure<'boundary, 'process>,
> {
    let checked = (|| {
        candidate
            .matches_original_view(
                registration.custody.boundary,
                &registration.custody.custody.owners.input_lock,
                &registration.custody.custody.owners.control,
            )
            .map_err(RoventStorageCompiledError::CompiledBoundary)
            .map_err(RoventStorageRuntimeError::Compiled)?;
        registration
            .custody
            .revalidate_startup()
            .map_err(RoventStorageRuntimeError::Compiled)?;
        candidate
            .revalidate_startup()
            .map_err(RoventStorageCompiledError::CompiledBoundary)
            .map_err(RoventStorageRuntimeError::Compiled)
    })();
    match checked {
        Ok(()) => Ok(RoventStorageOwnedRuntimeRegistration {
            custody: registration.custody.custody,
            boundary: candidate,
            state: registration.state,
        }),
        Err(error) => Err(RoventStorageRuntimeOwnershipFailure {
            error,
            registration,
            candidate,
        }),
    }
}

#[derive(Debug)]
pub enum RoventStorageRuntimeError {
    Compiled(RoventStorageCompiledError),
    Namespace(&'static str),
    GoControl(ContractError),
    GoNoLongerLive,
    AlreadyIssued,
    MissingCapability,
}
impl fmt::Display for RoventStorageRuntimeError {
    fn fmt(&self, output: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Compiled(error) => write!(output, "storage runtime registration: {error}"),
            Self::Namespace(reason) => write!(output, "storage expected namespace: {reason}"),
            Self::GoControl(error) => write!(output, "storage original go control: {error}"),
            Self::GoNoLongerLive => write!(output, "storage original go is stopped or expired"),
            Self::AlreadyIssued => write!(output, "storage setup capability was already issued"),
            Self::MissingCapability => write!(output, "storage original capability is absent"),
        }
    }
}
impl Error for RoventStorageRuntimeError {}

/// Rejected startup-to-runtime conversion retains every original input owner
/// plus the caller's exact expected namespace values. It issues no go authority.
pub struct RoventStorageRuntimeRegistrationFailure<'boundary, 'process> {
    pub error: RoventStorageRuntimeError,
    pub custody: RoventStorageCompiledCustody<'boundary, 'process>,
    pub expected_process_clock: ProcessClock,
    pub expected_session_implementation: Digest,
}
impl fmt::Debug for RoventStorageRuntimeRegistrationFailure<'_, '_> {
    fn fmt(&self, output: &mut fmt::Formatter<'_>) -> fmt::Result {
        output
            .debug_struct("RoventStorageRuntimeRegistrationFailure")
            .field("error", &self.error)
            .field(
                "root_capability_retained",
                &self.custody.retains_root_capability(),
            )
            .finish_non_exhaustive()
    }
}

/// The registration remains borrowed in-place on rejection. Only this exact
/// original admission is returned; no replacement control/deadline is minted.
#[derive(Debug)]
pub struct RoventStorageSetupRejection {
    pub error: RoventStorageRuntimeError,
    pub admission: Arc<RoventRootAdmission>,
}

/// Freeze successful startup custody while its original startup control is
/// still live. Expected clock/implementation must later come from the original
/// driver constructor, not from the first admission being compared with them.
/// They are selections, not source/physical authority, and cannot issue setup
/// without an independently admitted original go. No worker/IO starts here.
#[allow(clippy::result_large_err)]
pub fn register_rovent_storage_runtime<'boundary, 'process>(
    custody: RoventStorageCompiledCustody<'boundary, 'process>,
    expected_process_clock: ProcessClock,
    expected_session_implementation: Digest,
) -> Result<
    RoventStorageRuntimeRegistration<'boundary, 'process>,
    RoventStorageRuntimeRegistrationFailure<'boundary, 'process>,
> {
    let prepared = (|| {
        custody
            .revalidate_startup()
            .map_err(RoventStorageRuntimeError::Compiled)?;
        let state = prepare_runtime_state(
            &custody.custody,
            &expected_process_clock,
            expected_session_implementation,
        )?;
        // Identity framing/allocation must not cross the original startup
        // boundary and then publish a new runtime registration as if live.
        check(&custody.custody.owners.control)
            .map_err(RoventStorageCompiledError::Admission)
            .map_err(RoventStorageRuntimeError::Compiled)?;
        Ok(state)
    })();
    match prepared {
        Ok(state) => Ok(RoventStorageRuntimeRegistration { custody, state }),
        Err(error) => Err(RoventStorageRuntimeRegistrationFailure {
            error,
            custody,
            expected_process_clock,
            expected_session_implementation,
        }),
    }
}

fn prepare_runtime_state(
    custody: &RoventStorageByteCustody,
    expected_clock: &ProcessClock,
    expected_implementation: Digest,
) -> Result<StorageRuntimeState, RoventStorageRuntimeError> {
    if expected_clock.epoch().0 == 0 || expected_implementation == Digest([0; 32]) {
        return Err(RoventStorageRuntimeError::Namespace(
            "nonzero expected clock/implementation required",
        ));
    }
    validate_runtime_profile(custody)?;
    if !custody.retains_root_capability() {
        return Err(RoventStorageRuntimeError::MissingCapability);
    }
    let identity = runtime_identity(custody, expected_clock, expected_implementation)?;
    Ok(StorageRuntimeState {
        expected_clock: expected_clock.clone(),
        expected_implementation,
        identity,
        issued_admission: None,
    })
}

fn validate_runtime_profile(
    custody: &RoventStorageByteCustody,
) -> Result<(), RoventStorageRuntimeError> {
    verify_native_allowance(custody.profile().maintenance_io_bytes_max)
        .map_err(RoventStorageRuntimeError::Compiled)?;
    custody
        .profile()
        .validate_for_manifest(&custody.owners.input_lock.manifest)
        .map_err(RoventStorageCompiledError::Input)
        .map_err(RoventStorageRuntimeError::Compiled)
}

fn runtime_identity(
    custody: &RoventStorageByteCustody,
    clock: &ProcessClock,
    implementation: Digest,
) -> Result<UnusedIdleArchiveSetupIdentity, RoventStorageRuntimeError> {
    let fail = RoventStorageRuntimeError::Compiled;
    let profile_text = custody
        .profile()
        .canonical_sha256()
        .map_err(RoventStorageCompiledError::Input)
        .map_err(fail)?;
    let profile = compiled_digest(&profile_text).map_err(fail)?;
    let input_lock = compiled_digest(custody.input_lock_sha256()).map_err(fail)?;
    let aggregate = rz_experiments::rovent_source_aggregate_sha256(
        &custody.owners.input_lock.manifest.engine.source_commit,
        &custody.owners.input_lock.manifest.engine.source_components,
    )
    .map_err(RoventStorageCompiledError::Input)
    .map_err(fail)?;
    let mut hash = Sha256::new();
    hash_frame(&mut hash, RUNTIME_REGISTRATION_DOMAIN);
    hash.update(profile.0);
    hash.update(input_lock.0);
    hash.update(compiled_digest(&aggregate).map_err(fail)?.0);
    hash_frame(&mut hash, PROCESS_SOURCE_SCOPE.as_bytes());
    hash_frame(
        &mut hash,
        custody
            .owners
            .input_lock
            .manifest
            .engine
            .source_commit
            .as_bytes(),
    );
    // Expected epoch/implementation are framed as selections, not proof. The
    // raw Instant origin is never serialized; the retained clock is compared
    // directly with the actual admitted clock on first use.
    hash.update(clock.epoch().0.to_le_bytes());
    hash.update(implementation.0);
    for (role, directory) in [b"archive".as_slice(), b"checkout", b"private", b"protected"]
        .into_iter()
        .zip(&custody.owners.directories)
    {
        let directory = directory
            .as_ref()
            .ok_or(RoventStorageRuntimeError::Namespace(
                "original runtime directory is absent",
            ))?;
        hash_frame(&mut hash, role);
        hash.update(directory.identity.0.to_le_bytes());
        hash.update(directory.identity.1.to_le_bytes());
    }
    let artifact = custody
        .owners
        .artifact
        .as_ref()
        .ok_or(RoventStorageRuntimeError::Namespace(
            "original profile artifact is absent",
        ))?;
    hash.update(artifact.bytes.to_le_bytes());
    hash.update(compiled_digest(&artifact.sha256).map_err(fail)?.0);
    Ok(UnusedIdleArchiveSetupIdentity {
        profile,
        input_lock,
        registration: Digest(hash.finalize().into()),
    })
}

fn hash_frame(hash: &mut Sha256, value: &[u8]) {
    hash.update((value.len() as u64).to_le_bytes());
    hash.update(value);
}

struct PreparedSetupValues {
    config: ArchiveConfig,
    limits: ArchiveRuntimeLimits,
    control: UnusedIdleArchiveSetupControl,
}

fn validate_original_setup_go(
    custody: &RoventStorageByteCustody,
    state: &StorageRuntimeState,
    admission: &RoventRootAdmission,
) -> Result<(), RoventStorageRuntimeError> {
    if state.issued_admission.is_some() {
        return Err(RoventStorageRuntimeError::AlreadyIssued);
    }
    validate_runtime_profile(custody)?;
    let actual = admission.runtime_clock();
    let expected = state.expected_clock.runtime_clock();
    let context = admission.original_context();
    if admission.startup_lock_digest() != state.identity.input_lock
        || compiled_digest(custody.input_lock_sha256())
            .map_err(RoventStorageRuntimeError::Compiled)?
            != state.identity.input_lock
        || admission.authority().epoch != state.expected_clock.epoch()
        || admission.authority().implementation != state.expected_implementation
        || actual.domain() != expected.domain()
        || actual.origin() != expected.origin()
    {
        return Err(RoventStorageRuntimeError::Namespace(
            "original go lock/epoch/implementation/clock differs",
        ));
    }
    for (instant, sealed) in [
        (context.control.deadline, admission.hard_deadline()),
        (context.control.soft_deadline, admission.soft_deadline()),
        (
            context.control.admission_deadline,
            admission.admission_deadline(),
        ),
        (context.shutdown_deadline, admission.cleanup_deadline()),
    ] {
        if state
            .expected_clock
            .deadline(instant)
            .map_err(RoventStorageRuntimeError::GoControl)?
            != sealed
        {
            return Err(RoventStorageRuntimeError::Namespace(
                "original go clock/control seal differs",
            ));
        }
    }
    if !context
        .try_accepts()
        .map_err(RoventStorageRuntimeError::GoControl)?
    {
        return Err(RoventStorageRuntimeError::GoNoLongerLive);
    }
    Ok(())
}

fn prepare_setup_values(
    custody: &RoventStorageByteCustody,
    state: &StorageRuntimeState,
    admission: &RoventRootAdmission,
) -> Result<PreparedSetupValues, RoventStorageRuntimeError> {
    validate_original_setup_go(custody, state, admission)?;
    if !custody.retains_root_capability() {
        return Err(RoventStorageRuntimeError::MissingCapability);
    }
    let selected = custody.profile().archive_limits;
    let limits = ArchiveRuntimeLimits {
        game_bytes_max: selected.game_bytes_max,
        global_bytes_max: selected.global_bytes_max,
        index_entries_max: selected.index_entries_max,
        index_bytes_max: selected.index_bytes_max,
        load_bytes_max: selected.load_bytes_max,
        load_deadline_max_ms: selected.load_deadline_max_ms,
        record_payload_bytes_max: selected.record_payload_bytes_max,
        max_load_pins: selected.max_load_pins,
    };
    let root = custody.owners.directories[ARCHIVE].as_ref().ok_or(
        RoventStorageRuntimeError::Namespace("original archive directory is absent"),
    )?;
    let checkout = custody.owners.directories[CHECKOUT].as_ref().ok_or(
        RoventStorageRuntimeError::Namespace("original checkout directory is absent"),
    )?;
    // validate_for_manifest checked max_load_pins <= 16, which fits usize on
    // every supported native target; no runtime selector/default is invented.
    let max_load_pins = selected.max_load_pins as usize;
    Ok(PreparedSetupValues {
        config: ArchiveConfig {
            root: root.path.clone(),
            repository_root: checkout.path.clone(),
            game_bytes: selected.game_bytes_max,
            global_bytes: selected.global_bytes_max,
            record_bytes: selected.record_payload_bytes_max,
            max_load_pins,
        },
        limits,
        control: UnusedIdleArchiveSetupControl {
            deadline: admission.original_context().control.deadline,
            cancellation: Arc::clone(&admission.original_context().control.cancellation),
            max_io_bytes: custody.profile().maintenance_io_bytes_max,
        },
    })
}

fn split_unused_setup(
    custody: &mut RoventStorageByteCustody,
    state: &mut StorageRuntimeState,
    admission: Arc<RoventRootAdmission>,
) -> Result<UnusedIdleArchiveSetup, RoventStorageSetupRejection> {
    let prepared = match prepare_setup_values(custody, state, &admission).and_then(|prepared| {
        // Last fallible/liveness check precedes the sole capability take.
        validate_original_setup_go(custody, state, &admission)?;
        Ok(prepared)
    }) {
        Ok(prepared) => prepared,
        Err(error) => return Err(RoventStorageSetupRejection { error, admission }),
    };
    let Some(capability) = custody.owners.capability.take() else {
        return Err(RoventStorageSetupRejection {
            error: RoventStorageRuntimeError::MissingCapability,
            admission,
        });
    };
    // Every following operation is an infallible move/Arc increment. The
    // original startup control and all other original FDs remain untouched.
    let setup = UnusedIdleArchiveSetup::new(
        prepared.config,
        capability,
        prepared.limits,
        state.identity,
        prepared.control,
    );
    state.issued_admission = Some(admission);
    Ok(setup)
}

#[derive(Debug)]
pub enum RoventStorageCompiledError {
    Admission(RoventStorageAdmissionError),
    CompiledBoundary(ProcessBoundaryError),
    Input(ManifestError),
    Mismatch(&'static str),
    /// The codec may describe smaller positive values; the first native
    /// bridge supports exactly this one allowance and never rounds it up.
    UnsupportedMaintenanceIoBytes {
        requested: u64,
        supported: u64,
    },
}
impl fmt::Display for RoventStorageCompiledError {
    fn fmt(&self, output: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Admission(error) => write!(output, "storage compiled admission: {error}"),
            Self::CompiledBoundary(error) => write!(output, "storage compiled boundary: {error}"),
            Self::Input(error) => write!(output, "storage compiled input: {error}"),
            Self::Mismatch(reason) => write!(output, "storage compiled identity: {reason}"),
            Self::UnsupportedMaintenanceIoBytes {
                requested,
                supported,
            } => write!(
                output,
                "UnsupportedCapability: native archive maintenance requires {supported} bytes; declared {requested}"
            ),
        }
    }
}
impl Error for RoventStorageCompiledError {}

/// Every rejection returns the same complete byte custody. No FD, directory,
/// capability, original lock or startup cancellation owner is replaced/dropped.
pub struct RoventStorageCompiledFailure {
    pub error: RoventStorageCompiledError,
    pub custody: RoventStorageByteCustody,
}
impl fmt::Debug for RoventStorageCompiledFailure {
    fn fmt(&self, output: &mut fmt::Formatter<'_>) -> fmt::Result {
        output
            .debug_struct("RoventStorageCompiledFailure")
            .field("error", &self.error)
            .field(
                "source_fds_retained",
                &self.custody.owners().verified_source_files(),
            )
            .field(
                "root_capability_retained",
                &self.custody.retains_root_capability(),
            )
            .finish()
    }
}

/// Separate opt-in library conversion. Only the existing actual process
/// observation can provide `boundary`; declarations and callers cannot create
/// it. This checks raw source roles plus every original registered component,
/// but does not advertise CLI/session/worker or archive execution support.
#[allow(clippy::result_large_err)]
pub fn check_rovent_storage_compiled_custody<'boundary, 'process>(
    custody: RoventStorageByteCustody,
    boundary: &'boundary VerifiedRoventProcessBoundary<'process>,
) -> Result<RoventStorageCompiledCustody<'boundary, 'process>, RoventStorageCompiledFailure> {
    match verify_compiled_custody(&custody, boundary) {
        Ok(()) => Ok(RoventStorageCompiledCustody { custody, boundary }),
        Err(error) => Err(RoventStorageCompiledFailure { error, custody }),
    }
}

fn verify_compiled_custody(
    custody: &RoventStorageByteCustody,
    boundary: &VerifiedRoventProcessBoundary<'_>,
) -> Result<(), RoventStorageCompiledError> {
    let owners = &custody.owners;
    check(&owners.control).map_err(RoventStorageCompiledError::Admission)?;
    verify_native_allowance(custody.profile().maintenance_io_bytes_max)?;
    custody
        .revalidate_startup()
        .map_err(RoventStorageCompiledError::Admission)?;
    let profile = custody.profile();
    profile
        .validate_for_manifest(&owners.input_lock.manifest)
        .map_err(RoventStorageCompiledError::Input)?;
    let profile_artifact = owners
        .artifact
        .as_ref()
        .ok_or_else(|| compiled_mismatch("original profile artifact is absent"))?;
    if Some(profile_artifact) != owners.profile_file.as_ref().map(|held| &held.artifact)
        || unique_provenance(&owners.input_lock, &profile_artifact.path)
            .map_err(RoventStorageCompiledError::Admission)?
            != profile_artifact
    {
        return Err(compiled_mismatch(
            "profile differs from original held provenance",
        ));
    }
    let original_private = owners.directories[PRIVATE]
        .as_ref()
        .ok_or_else(|| compiled_mismatch("original private directory is not held"))?;
    let binding = StorageProcessBinding {
        input_lock: boundary.input_lock_digest(),
        source_aggregate: boundary.source_aggregate_digest(),
        source_commit: boundary.registered_source_commit(),
        source_scope: boundary.source_scope(),
        private_root: directory_identity(boundary.original_root())
            .map_err(RoventStorageCompiledError::Admission)?,
        executable_bytes: boundary.executable_identity().bytes,
        executable_sha256: &boundary.executable_identity().sha256,
    };
    let actual_components = compiled_process_source_components()
        .map_err(RoventStorageCompiledError::CompiledBoundary)?;
    verify_original_process_binding(
        &owners.input_lock,
        original_private.identity,
        &binding,
        &actual_components,
    )?;
    let artifacts = [
        &profile.source_pins.source,
        &profile.source_pins.codec,
        &profile.source_pins.launch,
    ];
    for (index, artifact) in artifacts.iter().enumerate() {
        let held = owners.source_files[index]
            .as_ref()
            .ok_or_else(|| compiled_mismatch("original source descriptor is absent"))?;
        if !held.revalidated
            || &held.artifact != *artifact
            || unique_provenance(&owners.input_lock, &artifact.path)
                .map_err(RoventStorageCompiledError::Admission)?
                != *artifact
        {
            return Err(compiled_mismatch(
                "source differs from original held provenance",
            ));
        }
    }
    let inventory = rz_experiments::compiled_source_inventory_v5()
        .map_err(RoventStorageCompiledError::Input)?;
    verify_compiled_storage_sources(artifacts, &inventory)?;
    // The accounting aggregate is deliberately absent from the raw comparison:
    // it frames several Store files and is not SHA256(raw archive.rs).
    check(&owners.control).map_err(RoventStorageCompiledError::Admission)
}

fn verify_native_allowance(requested: u64) -> Result<(), RoventStorageCompiledError> {
    if requested != NATIVE_MAINTENANCE_IO_BYTES {
        return Err(RoventStorageCompiledError::UnsupportedMaintenanceIoBytes {
            requested,
            supported: NATIVE_MAINTENANCE_IO_BYTES,
        });
    }
    Ok(())
}

// Private comparison values extracted only from the opaque actual boundary in
// production. Tests may exercise this pure comparison without issuing a proof
// or constructing a VerifiedRoventProcessBoundary/checked custody.
struct StorageProcessBinding<'a> {
    input_lock: Digest,
    source_aggregate: Digest,
    source_commit: &'a str,
    source_scope: &'static str,
    private_root: (u64, u64),
    executable_bytes: u64,
    executable_sha256: &'a str,
}

fn verify_original_process_binding(
    lock: &RoventInputLock,
    original_private: (u64, u64),
    binding: &StorageProcessBinding<'_>,
    actual_components: &BTreeMap<String, String>,
) -> Result<(), RoventStorageCompiledError> {
    lock.validate().map_err(RoventStorageCompiledError::Input)?;
    let aggregate = rz_experiments::rovent_source_aggregate_sha256(
        &lock.manifest.engine.source_commit,
        &lock.manifest.engine.source_components,
    )
    .map_err(RoventStorageCompiledError::Input)?;
    if compiled_digest(&lock.canonical_sha256)? != binding.input_lock
        || compiled_digest(&aggregate)? != binding.source_aggregate
        || lock.manifest.engine.source_commit != binding.source_commit
        || binding.source_scope != PROCESS_SOURCE_SCOPE
        || original_private != binding.private_root
        || lock.manifest.engine.binary.bytes != binding.executable_bytes
        || lock.manifest.engine.binary.sha256 != binding.executable_sha256
    {
        return Err(compiled_mismatch(
            "original lock/source/executable/private-root binding differs",
        ));
    }
    // The opaque boundary already enforces the mode-specific required set.
    // Compare its entire original declaration again with the complete actual
    // compiled map, including extra declared keys; no cherry-picked role set.
    if lock
        .manifest
        .engine
        .source_components
        .iter()
        .any(|(name, digest)| actual_components.get(name) != Some(digest))
    {
        return Err(compiled_mismatch(
            "registered component differs from compiled process source",
        ));
    }
    Ok(())
}

fn compiled_digest(value: &str) -> Result<Digest, RoventStorageCompiledError> {
    rz_eval::asset::parse_sha256(value)
        .map(Digest)
        .map_err(|_| compiled_mismatch("validated source/lock digest is not exact SHA256"))
}

fn verify_compiled_storage_sources(
    artifacts: [&ArtifactRef; 3],
    inventory: &CompiledSourceInventoryV5,
) -> Result<(), RoventStorageCompiledError> {
    for (artifact, expected_path) in artifacts.into_iter().zip(COMPILED_STORAGE_SOURCE_PATHS) {
        let mut found = false;
        for rows in inventory.values() {
            for (path, bytes, digest) in rows {
                if path == expected_path {
                    found = true;
                    if *bytes != artifact.bytes
                        || Digest(*digest) != compiled_digest(&artifact.sha256)?
                    {
                        return Err(compiled_mismatch(
                            "source role raw bytes differ from embedded inventory",
                        ));
                    }
                }
            }
        }
        if !found {
            return Err(compiled_mismatch(
                "closed source role is absent from embedded inventory",
            ));
        }
    }
    // include_bytes! inventory rows are exact raw length/hash. Component
    // sorting/framing occurs later; neither CRLF nor LF is normalized here.
    Ok(())
}

fn compiled_mismatch(reason: &'static str) -> RoventStorageCompiledError {
    RoventStorageCompiledError::Mismatch(reason)
}

#[derive(Debug)]
pub enum RoventStorageAdmissionError {
    UnsupportedPlatform,
    Cancelled,
    Expired,
    Invalid(&'static str),
    Allocation,
    Input(ManifestError),
    Root(StoreError),
    Io {
        stage: &'static str,
        source: std::io::Error,
    },
}

impl fmt::Display for RoventStorageAdmissionError {
    fn fmt(&self, output: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedPlatform => write!(
                output,
                "UnsupportedCapability: native storage admission requires Linux"
            ),
            Self::Cancelled => write!(output, "storage admission canceled"),
            Self::Expired => write!(output, "original storage startup deadline expired"),
            Self::Invalid(reason) => write!(output, "storage admission: {reason}"),
            Self::Allocation => write!(output, "storage admission allocation failed"),
            Self::Input(error) => write!(output, "storage original input: {error}"),
            Self::Root(error) => write!(output, "storage root capability: {error}"),
            Self::Io { stage, source } => write!(output, "storage {stage}: {source}"),
        }
    }
}
impl Error for RoventStorageAdmissionError {}

/// The exact original lock and each successfully opened owner survive failure.
pub struct RoventStorageAdmissionFailure {
    pub error: RoventStorageAdmissionError,
    pub owners: RoventStorageAdmissionOwners,
}

impl fmt::Debug for RoventStorageAdmissionFailure {
    fn fmt(&self, output: &mut fmt::Formatter<'_>) -> fmt::Result {
        output
            .debug_struct("RoventStorageAdmissionFailure")
            .field("error", &self.error)
            .field("profile_fd_retained", &self.owners.profile_file.is_some())
            .field("source_fds_retained", &self.owners.verified_source_files())
            .field(
                "root_capability_retained",
                &self.owners.capability.is_some(),
            )
            .finish()
    }
}

/// Library entry only: no existing CLI, startup assembly or session selects it.
/// The caller supplies the original Arc lock, actual private Dir and original
/// startup control. No renewed or cleanup deadline is used for admission reads.
#[allow(clippy::result_large_err)]
pub fn open_rovent_storage_byte_custody(
    input_lock: Arc<RoventInputLock>,
    private_root: &Dir,
    request: RoventStorageAdmissionRequest<'_>,
    control: RoventStartupControl,
) -> Result<RoventStorageByteCustody, RoventStorageAdmissionFailure> {
    let mut owners = RoventStorageAdmissionOwners {
        input_lock,
        control,
        artifact: None,
        profile: None,
        profile_file: None,
        source_files: [None, None, None],
        directories: [None, None, None, None],
        capability: None,
    };
    let result = admit(&mut owners, private_root, request);
    match result {
        Ok(()) => Ok(RoventStorageByteCustody { owners }),
        Err(error) => Err(RoventStorageAdmissionFailure { error, owners }),
    }
}

fn admit(
    owners: &mut RoventStorageAdmissionOwners,
    private_root: &Dir,
    request: RoventStorageAdmissionRequest<'_>,
) -> Result<(), RoventStorageAdmissionError> {
    if !cfg!(target_os = "linux") {
        return Err(RoventStorageAdmissionError::UnsupportedPlatform);
    }
    check(&owners.control)?;
    if owners.control.cleanup_until < owners.control.until
        || !rz_experiments::pals_private_snapshot_path_v5(request.profile_path)
        || request.profile_bytes == 0
        || request.profile_bytes > MAX_INPUT_BYTES
    {
        return Err(invalid(
            "original bounded profile/control selection is required",
        ));
    }
    owners.input_lock.validate().map_err(input)?;
    let artifact = unique_provenance(&owners.input_lock, request.profile_path)?.clone();
    owners.artifact = Some(artifact.clone());
    if artifact.sha256 != request.profile_sha256 || artifact.bytes != request.profile_bytes {
        return Err(invalid(
            "explicit profile raw pin differs from original provenance",
        ));
    }
    let raw = retain_verified_input(
        &mut owners.profile_file,
        private_root,
        artifact,
        &owners.control,
        true,
    )?;
    let text = std::str::from_utf8(&raw).map_err(|_| invalid("profile is not UTF-8 JSON"))?;
    let profile = RoventStorageProfile::from_json(text).map_err(input)?;
    owners.profile = Some(profile);
    owners
        .profile
        .as_ref()
        .unwrap()
        .validate_for_manifest(&owners.input_lock.manifest)
        .map_err(input)?;

    let declared_sources = &owners.profile.as_ref().unwrap().source_pins;
    let sources = [
        declared_sources.source.clone(),
        declared_sources.codec.clone(),
        declared_sources.launch.clone(),
    ];
    for (index, artifact) in sources.into_iter().enumerate() {
        check(&owners.control)?;
        if unique_provenance(&owners.input_lock, &artifact.path)? != &artifact {
            return Err(invalid(
                "source pin metadata differs from original provenance",
            ));
        }
        retain_verified_input(
            &mut owners.source_files[index],
            private_root,
            artifact,
            &owners.control,
            false,
        )?;
    }
    // Capture every named boundary without following a component. These are
    // metadata/root capabilities only, never a new archive game or segment.
    for (index, path) in [
        request.archive_root,
        request.checkout_root,
        request.private_snapshot_root,
        request.protected_execution_root,
    ]
    .into_iter()
    .enumerate()
    {
        owners.directories[index] = Some(open_directory(path, &owners.control)?);
    }
    if owners.directories[PRIVATE].as_ref().unwrap().identity != directory_identity(private_root)? {
        return Err(invalid(
            "private snapshot path differs from the original held Dir",
        ));
    }
    require_separation(owners)?;
    let archive = owners.directories[ARCHIVE].as_ref().unwrap();
    let directory = archive
        .directory
        .try_clone()
        .map_err(|source| io("archive capability clone", source))?;
    owners.capability = Some(
        ArchiveRootCapability::from_directory(archive.path.clone(), directory)
            .map_err(RoventStorageAdmissionError::Root)?,
    );
    // Revalidate the same successfully held inputs after all opens, including
    // growth/EOF and root-name replacement races. No compiled match is emitted.
    for held in owners
        .profile_file
        .iter()
        .chain(owners.source_files.iter().flatten())
    {
        read_rechecked(held, &owners.control, false)?;
    }
    for directory in owners.directories.iter().flatten() {
        directory.revalidate(&owners.control)?;
    }
    check(&owners.control)
}

fn unique_provenance<'a>(
    lock: &'a RoventInputLock,
    path: &str,
) -> Result<&'a ArtifactRef, RoventStorageAdmissionError> {
    let mut matching = lock
        .manifest
        .provenance
        .iter()
        .filter(|artifact| artifact.path == path);
    let artifact = matching
        .next()
        .ok_or_else(|| invalid("input is absent from original provenance"))?;
    if matching.next().is_some() {
        return Err(invalid("input path is not unique in original provenance"));
    }
    Ok(artifact)
}

struct VerifiedInput {
    artifact: ArtifactRef,
    file: File,
    identity: Option<FileIdentity>,
    revalidated: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
struct FileIdentity {
    device: u64,
    inode: u64,
    bytes: u64,
    links: u64,
    modified: (i64, i64),
    changed: (i64, i64),
}

fn retain_verified_input(
    destination: &mut Option<VerifiedInput>,
    root: &Dir,
    artifact: ArtifactRef,
    control: &RoventStartupControl,
    collect: bool,
) -> Result<Vec<u8>, RoventStorageAdmissionError> {
    check(control)?;
    let file = artifact
        .open_verified_at(root, MAX_INPUT_BYTES)
        .map_err(input)?;
    // ArtifactRef has returned the actual verified descriptor. Retain it
    // before every further check, so cancellation/partial rejection cannot
    // silently discard a successfully acquired original owner.
    *destination = Some(VerifiedInput {
        artifact,
        file,
        identity: None,
        revalidated: false,
    });
    let held = destination.as_mut().unwrap();
    held.identity = Some(file_identity(&held.file)?);
    let raw = read_rechecked(held, control, collect)?;
    held.revalidated = true;
    Ok(raw)
}

#[cfg(target_os = "linux")]
fn file_identity(file: &File) -> Result<FileIdentity, RoventStorageAdmissionError> {
    use std::os::unix::fs::MetadataExt;
    let metadata = file
        .metadata()
        .map_err(|source| io("original input metadata", source))?;
    if !metadata.is_file() {
        return Err(invalid("original input descriptor is not regular"));
    }
    Ok(FileIdentity {
        device: metadata.dev(),
        inode: metadata.ino(),
        bytes: metadata.len(),
        links: metadata.nlink(),
        modified: (metadata.mtime(), metadata.mtime_nsec()),
        changed: (metadata.ctime(), metadata.ctime_nsec()),
    })
}
#[cfg(not(target_os = "linux"))]
fn file_identity(_: &File) -> Result<FileIdentity, RoventStorageAdmissionError> {
    Err(RoventStorageAdmissionError::UnsupportedPlatform)
}

#[cfg(target_os = "linux")]
fn read_rechecked(
    input: &VerifiedInput,
    control: &RoventStartupControl,
    collect: bool,
) -> Result<Vec<u8>, RoventStorageAdmissionError> {
    use std::os::unix::fs::FileExt;
    check(control)?;
    let before = file_identity(&input.file)?;
    if Some(before) != input.identity
        || before.bytes != input.artifact.bytes
        || before.bytes == 0
        || before.bytes > MAX_INPUT_BYTES
    {
        return Err(invalid(
            "original descriptor identity or byte bound changed",
        ));
    }
    let mut raw = Vec::new();
    if collect {
        raw.try_reserve_exact(
            usize::try_from(before.bytes).map_err(|_| RoventStorageAdmissionError::Allocation)?,
        )
        .map_err(|_| RoventStorageAdmissionError::Allocation)?;
    }
    let mut hash = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    let mut offset = 0u64;
    while offset < before.bytes {
        check(control)?;
        let length = (before.bytes - offset).min(buffer.len() as u64) as usize;
        let read = input
            .file
            .read_at(&mut buffer[..length], offset)
            .map_err(|source| io("original descriptor read", source))?;
        if read == 0 {
            return Err(invalid("original descriptor ended before its raw pin"));
        }
        hash.update(&buffer[..read]);
        if collect {
            raw.extend_from_slice(&buffer[..read]);
        }
        offset += read as u64;
    }
    check(control)?;
    let mut tail = [0u8; 1];
    let tail_len = input
        .file
        .read_at(&mut tail, offset)
        .map_err(|source| io("original descriptor EOF", source))?;
    if tail_len != 0
        || file_identity(&input.file)? != before
        || format!("{:x}", hash.finalize()) != input.artifact.sha256
    {
        return Err(invalid("original descriptor bytes/identity/EOF changed"));
    }
    check(control)?;
    Ok(raw)
}
#[cfg(not(target_os = "linux"))]
fn read_rechecked(
    input: &VerifiedInput,
    _: &RoventStartupControl,
    _: bool,
) -> Result<Vec<u8>, RoventStorageAdmissionError> {
    let _ = (&input.artifact, &input.identity);
    Err(RoventStorageAdmissionError::UnsupportedPlatform)
}

struct HeldDirectory {
    path: PathBuf,
    directory: Dir,
    identity: (u64, u64),
    ancestors: Vec<(u64, u64)>,
}
impl HeldDirectory {
    fn revalidate(
        &self,
        control: &RoventStartupControl,
    ) -> Result<(), RoventStorageAdmissionError> {
        let named = open_directory(&self.path, control)?;
        if named.identity != self.identity
            || named.ancestors != self.ancestors
            || directory_identity(&self.directory)? != self.identity
        {
            return Err(invalid("held directory was renamed or replaced"));
        }
        Ok(())
    }
}

#[cfg(target_os = "linux")]
fn directory_identity(directory: &Dir) -> Result<(u64, u64), RoventStorageAdmissionError> {
    use std::os::unix::fs::MetadataExt;
    let metadata = directory
        .try_clone()
        .map_err(|source| io("directory clone", source))?
        .into_std_file()
        .metadata()
        .map_err(|source| io("directory metadata", source))?;
    if !metadata.is_dir() {
        return Err(invalid("held root is not a directory"));
    }
    Ok((metadata.dev(), metadata.ino()))
}
#[cfg(not(target_os = "linux"))]
fn directory_identity(_: &Dir) -> Result<(u64, u64), RoventStorageAdmissionError> {
    Err(RoventStorageAdmissionError::UnsupportedPlatform)
}

fn open_directory(
    path: &Path,
    control: &RoventStartupControl,
) -> Result<HeldDirectory, RoventStorageAdmissionError> {
    check(control)?;
    if !cfg!(target_os = "linux") {
        return Err(RoventStorageAdmissionError::UnsupportedPlatform);
    }
    if !path.is_absolute()
        || path.as_os_str().as_encoded_bytes().len() > MAX_ROOT_PATH_BYTES
        || !path
            .components()
            .any(|part| matches!(part, Component::Normal(_)))
        || path
            .components()
            .any(|part| !matches!(part, Component::RootDir | Component::Normal(_)))
    {
        return Err(invalid(
            "explicit bounded absolute roots without traversal are required",
        ));
    }
    let mut directory = Dir::open_ambient_dir("/", cap_std::ambient_authority())
        .map_err(|source| io("directory filesystem anchor", source))?;
    let mut ancestors = vec![directory_identity(&directory)?];
    for part in path.components() {
        check(control)?;
        if let Component::Normal(name) = part {
            directory = directory
                .open_dir_nofollow(name)
                .map_err(|source| io("no-follow directory component", source))?;
            ancestors.push(directory_identity(&directory)?);
        }
    }
    let identity = directory_identity(&directory)?;
    Ok(HeldDirectory {
        path: path.to_path_buf(),
        directory,
        identity,
        ancestors,
    })
}

fn require_separation(
    owners: &RoventStorageAdmissionOwners,
) -> Result<(), RoventStorageAdmissionError> {
    let archive = owners.directories[ARCHIVE]
        .as_ref()
        .ok_or_else(|| invalid("archive root is not held"))?;
    for index in [CHECKOUT, PRIVATE, PROTECTED] {
        let protected = owners.directories[index]
            .as_ref()
            .ok_or_else(|| invalid("protected directory is not held"))?;
        if archive.path.starts_with(&protected.path)
            || protected.path.starts_with(&archive.path)
            || archive.ancestors.contains(&protected.identity)
            || protected.ancestors.contains(&archive.identity)
        {
            return Err(invalid(
                "archive root overlaps checkout/private/protected execution root",
            ));
        }
    }
    Ok(())
}

fn check(control: &RoventStartupControl) -> Result<(), RoventStorageAdmissionError> {
    if control.cancellation.load(Ordering::Acquire) {
        return Err(RoventStorageAdmissionError::Cancelled);
    }
    if Instant::now() >= control.until {
        return Err(RoventStorageAdmissionError::Expired);
    }
    Ok(())
}
fn invalid(reason: &'static str) -> RoventStorageAdmissionError {
    RoventStorageAdmissionError::Invalid(reason)
}
fn input(error: ManifestError) -> RoventStorageAdmissionError {
    RoventStorageAdmissionError::Input(error)
}
fn io(stage: &'static str, source: std::io::Error) -> RoventStorageAdmissionError {
    RoventStorageAdmissionError::Io { stage, source }
}

#[cfg(test)]
mod compiled_source_tests {
    use super::*;
    use sha2::Sha256;

    fn artifact(path: &str, bytes: &[u8]) -> ArtifactRef {
        ArtifactRef {
            path: path.into(),
            bytes: bytes.len() as u64,
            sha256: format!("{:x}", <Sha256 as sha2::Digest>::digest(bytes)),
            source: "https://example.org/compiled-storage-comparison-fixture".into(),
            license: "fixture-only".into(),
        }
    }
    fn sources() -> [ArtifactRef; 3] {
        [
            artifact("inputs/retained-source.rs", b"store raw source\n"),
            artifact("inputs/retained-codec.rs", b"codec raw source\n"),
            artifact("inputs/retained-launch.rs", b"launch raw source\n"),
        ]
    }
    fn inventory(sources: &[ArtifactRef; 3]) -> CompiledSourceInventoryV5 {
        let rows: Vec<_> = sources
            .iter()
            .zip(COMPILED_STORAGE_SOURCE_PATHS)
            .map(|(artifact, path)| {
                (
                    path.into(),
                    artifact.bytes,
                    compiled_digest(&artifact.sha256).unwrap().0,
                )
            })
            .collect();
        BTreeMap::from([("first".into(), rows.clone()), ("second".into(), rows)])
    }
    fn refs(sources: &[ArtifactRef; 3]) -> [&ArtifactRef; 3] {
        [&sources[0], &sources[1], &sources[2]]
    }

    #[test]
    fn raw_roles_use_repo_rows_and_require_all_component_occurrences_to_agree() {
        let sources = sources();
        let original = inventory(&sources);
        verify_compiled_storage_sources(refs(&sources), &original).unwrap();
        // Snapshot artifact names deliberately differ from the repo paths.
        assert_ne!(sources[0].path, COMPILED_STORAGE_SOURCE_PATHS[0]);
        let mut conflicting = original.clone();
        conflicting.get_mut("second").unwrap()[0].2[31] ^= 1;
        assert!(verify_compiled_storage_sources(refs(&sources), &conflicting).is_err());
        let mut missing = original;
        for rows in missing.values_mut() {
            rows.retain(|(path, _, _)| path != COMPILED_STORAGE_SOURCE_PATHS[2]);
        }
        assert!(verify_compiled_storage_sources(refs(&sources), &missing).is_err());
        assert!(
            verify_compiled_storage_sources(
                [&sources[1], &sources[0], &sources[2]],
                &inventory(&sources)
            )
            .is_err()
        );
    }

    #[test]
    fn raw_length_line_endings_and_component_framing_are_not_interchangeable() {
        let sources = sources();
        let inventory = inventory(&sources);
        let mut changed = sources.clone();
        changed[0].bytes += 1;
        assert!(verify_compiled_storage_sources(refs(&changed), &inventory).is_err());
        changed = sources.clone();
        changed[0] = artifact(&sources[0].path, b"store raw source\r\n");
        assert!(verify_compiled_storage_sources(refs(&changed), &inventory).is_err());
        changed = sources;
        let framed = rz_experiments::source_component_digest_v5(&inventory["first"]).unwrap();
        changed[0].sha256 = format!("{:x}", <Sha256 as sha2::Digest>::digest(framed));
        assert!(verify_compiled_storage_sources(refs(&changed), &inventory).is_err());
        // Even an exact framed digest is not a raw archive.rs digest.
        changed[0].sha256 = framed.iter().map(|byte| format!("{byte:02x}")).collect();
        assert!(verify_compiled_storage_sources(refs(&changed), &inventory).is_err());
    }

    #[test]
    fn compiled_inventory_raw_rows_are_reused_without_normalization() {
        let inventory = rz_experiments::compiled_source_inventory_v5().unwrap();
        let mut sources = sources();
        for (artifact, path) in sources.iter_mut().zip(COMPILED_STORAGE_SOURCE_PATHS) {
            let row = inventory
                .values()
                .flatten()
                .find(|(candidate, _, _)| candidate == path)
                .expect("storage source is registered by its owning components");
            artifact.bytes = row.1;
            artifact.sha256 = row.2.iter().map(|byte| format!("{byte:02x}")).collect();
        }
        verify_compiled_storage_sources(refs(&sources), &inventory).unwrap();
    }

    #[test]
    fn first_native_bridge_rejects_smaller_allowance_without_rounding() {
        verify_native_allowance(NATIVE_MAINTENANCE_IO_BYTES).unwrap();
        let requested = NATIVE_MAINTENANCE_IO_BYTES - 1;
        assert!(matches!(
            verify_native_allowance(requested),
            Err(RoventStorageCompiledError::UnsupportedMaintenanceIoBytes {
                requested: actual,
                supported: NATIVE_MAINTENANCE_IO_BYTES,
            }) if actual == requested
        ));
        assert!(verify_native_allowance(0).is_err());
        assert!(verify_native_allowance(NATIVE_MAINTENANCE_IO_BYTES + 1).is_err());
    }
}

#[cfg(all(test, target_os = "linux"))]
mod tests {
    use super::open_rovent_storage_byte_custody as admit_rovent_storage;
    use super::*;
    use rz_experiments::{
        ROVENT_STORAGE_MEASUREMENT_CONTRACT, ROVENT_STORAGE_PROFILE_DOMAIN,
        RoventArchiveRuntimeLimits, RoventExecutionManifest, RoventStorageHelpers,
        RoventStoragePolicy, RoventStorageSourcePins,
    };
    use std::{
        fs,
        sync::atomic::{AtomicBool, AtomicU64},
        time::Duration,
    };

    static NEXT: AtomicU64 = AtomicU64::new(1);

    #[test]
    fn owned_runtime_registration_has_no_process_borrow_or_missing_thread_bound() {
        fn assert_static<T: Send + Sync + 'static>() {}
        assert_static::<RoventStorageOwnedRuntimeRegistration>();
        // This is a type boundary assertion only. Do not fabricate an actual
        // process proof to claim public owned registration/driver acceptance.
    }

    struct Scratch(PathBuf);
    impl Scratch {
        fn new() -> Self {
            let path = std::env::temp_dir().join(format!(
                "rovent-storage-admission-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir(&path).unwrap();
            Self(path)
        }
    }
    impl Drop for Scratch {
        fn drop(&mut self) {
            // Exactly this fixture-created tree; no source/cache/foreign tree.
            if self.0.parent() == Some(std::env::temp_dir().as_path())
                && self.0.file_name().is_some_and(|name| {
                    name.to_string_lossy()
                        .starts_with("rovent-storage-admission-")
                })
                && fs::symlink_metadata(&self.0)
                    .is_ok_and(|metadata| metadata.is_dir() && !metadata.file_type().is_symlink())
            {
                let _ = fs::remove_dir_all(&self.0);
            }
        }
    }

    struct Fixture {
        _scratch: Scratch,
        private_path: PathBuf,
        checkout: PathBuf,
        protected: PathBuf,
        archive: PathBuf,
        private: Dir,
        profile: ArtifactRef,
        lock: Arc<RoventInputLock>,
    }
    impl Fixture {
        fn new() -> Self {
            Self::with_maintenance(1 << 20)
        }
        fn with_maintenance(maintenance_io_bytes_max: u64) -> Self {
            let scratch = Scratch::new();
            let private_path = scratch.0.join("private");
            let checkout = scratch.0.join("checkout");
            let protected = scratch.0.join("execution");
            let archive = scratch.0.join("archive");
            for directory in [&private_path, &checkout, &protected, &archive] {
                fs::create_dir(directory).unwrap();
            }
            fs::create_dir(private_path.join("inputs")).unwrap();
            let mut pins = Vec::new();
            for (name, body) in [
                ("source.json", "source-fixture/1\n"),
                ("codec.rs", "codec-fixture/1\n"),
                ("launch.rs", "launch-fixture/1\n"),
            ] {
                let path = format!("inputs/{name}");
                fs::write(private_path.join(&path), body).unwrap();
                pins.push(artifact(&path, body.as_bytes()));
            }
            let declaration = RoventStorageProfile {
                schema: ROVENT_STORAGE_PROFILE_DOMAIN.into(),
                policy: RoventStoragePolicy::ArchiveOnV1,
                archive_limits: RoventArchiveRuntimeLimits {
                    game_bytes_max: 1 << 20,
                    global_bytes_max: 2 << 20,
                    index_entries_max: 16,
                    index_bytes_max: 1024,
                    load_bytes_max: 1024,
                    load_deadline_max_ms: 100,
                    record_payload_bytes_max: 1024,
                    max_load_pins: 1,
                },
                maintenance_io_bytes_max,
                worker_threads: 1,
                command_capacity: 1,
                completion_capacity: 1,
                helpers: RoventStorageHelpers {
                    shared_nnuec: 0,
                    archive: 1,
                },
                source_pins: RoventStorageSourcePins {
                    source: pins[0].clone(),
                    codec: pins[1].clone(),
                    launch: pins[2].clone(),
                },
                measurement_contract: ROVENT_STORAGE_MEASUREMENT_CONTRACT.into(),
            };
            let raw = serde_json::to_vec(&declaration).unwrap();
            fs::write(private_path.join("inputs/storage-profile.json"), &raw).unwrap();
            let profile = artifact("inputs/storage-profile.json", &raw);
            let mut provenance = vec![profile.clone()];
            provenance.extend(pins);
            let manifest = manifest(provenance);
            let lock = Arc::new(manifest.lock().unwrap());
            let private =
                Dir::open_ambient_dir(&private_path, cap_std::ambient_authority()).unwrap();
            Self {
                _scratch: scratch,
                private_path,
                checkout,
                protected,
                archive,
                private,
                profile,
                lock,
            }
        }
        fn request(&self) -> RoventStorageAdmissionRequest<'_> {
            RoventStorageAdmissionRequest {
                profile_path: &self.profile.path,
                profile_sha256: &self.profile.sha256,
                profile_bytes: self.profile.bytes,
                archive_root: &self.archive,
                checkout_root: &self.checkout,
                private_snapshot_root: &self.private_path,
                protected_execution_root: &self.protected,
            }
        }
        // Match the library entry: keep partial original FD/capability custody
        // directly in the error instead of adding a failure-path allocation.
        #[allow(clippy::result_large_err)]
        fn admit(&self) -> Result<RoventStorageByteCustody, RoventStorageAdmissionFailure> {
            admit_rovent_storage(
                Arc::clone(&self.lock),
                &self.private,
                self.request(),
                control(),
            )
        }
    }

    fn runtime_fixture() -> Fixture {
        let mut fixture = Fixture::with_maintenance(NATIVE_MAINTENANCE_IO_BYTES);
        let mut manifest = fixture.lock.manifest.clone();
        manifest.engine.search.rules_profile_sha256 =
            crate::rovent_bootstrap::hex(rz_position::contracts::profile_digest().0);
        fixture.lock = Arc::new(manifest.lock().unwrap());
        fixture
    }

    fn original_go_admission(
        lock: &RoventInputLock,
        clock: &ProcessClock,
        implementation: Digest,
    ) -> Arc<RoventRootAdmission> {
        use crate::engine::driver_start_fixture::{FixtureStart, issue_start};
        let position = rz_position::Position::startpos();
        let start = clock.runtime_clock().origin();
        let context = match issue_start(
            &position,
            // Match the registered fixture work bound; the production start
            // validator correctly rejects an otherwise unbounded default go.
            crate::parser::GoLimits {
                nodes: Some(4096),
                ..crate::parser::GoLimits::default()
            },
            rz_contracts::pals::SearchAuthority {
                epoch: clock.epoch(),
                game: rz_contracts::GameGeneration(1),
                root: rz_contracts::RootGeneration(1),
                implementation,
            },
            clock,
            start,
            crate::bridge::BuildSearchSettings {
                time_config: rz_search::time::TimeBudgetConfig::default(),
                max_simulations: 4096,
                untimed_limit: Duration::from_secs(10),
            },
            Duration::from_millis(500),
        )
        .unwrap()
        {
            FixtureStart::Started(context) => context,
            FixtureStart::RulesOnly(_) => panic!("ongoing actual Rules Start is required"),
        };
        Arc::new(
            RoventRootAdmission::admit(
                lock,
                crate::rovent_root_admission::RoventRootAdmissionPolicy::DynamicGo,
                &position,
                &context,
                clock,
            )
            .unwrap(),
        )
    }

    fn artifact(path: &str, raw: &[u8]) -> ArtifactRef {
        ArtifactRef {
            path: path.into(),
            sha256: format!("{:x}", Sha256::digest(raw)),
            bytes: raw.len() as u64,
            source: "https://example.org/storage-admission-fixture".into(),
            license: "MIT".into(),
        }
    }
    fn control() -> RoventStartupControl {
        let until = Instant::now() + Duration::from_secs(10);
        RoventStartupControl {
            until,
            cleanup_until: until + Duration::from_secs(1),
            cancellation: Arc::new(AtomicBool::new(false)),
        }
    }
    fn manifest(provenance: Vec<ArtifactRef>) -> RoventExecutionManifest {
        let pin = serde_json::json!({
            "semantics":"storage-fixture-model/1","encoding":"storage-fixture-input/1",
            "policy_head":"legal-candidates/1","value_head":"side-to-move-wdl/1",
            "implementation_sha256":"b".repeat(64),"weights":artifact("inputs/weights.bin",b"x"),
            "provenance":artifact("inputs/model-provenance.json",b"x"),"diagnostic":false
        });
        let value = serde_json::json!({
            "domain":"rz-rovent-execution/1","contract_revision":rz_contracts::rovent::ROVENT_SEARCH_REVISION,
            "run_id":"storage-fd-admission-fixture","purpose":"validation",
            "engine":{
                "source_commit":"a".repeat(40),"source_components":{"storage":"c".repeat(64)},
                "binary":artifact("inputs/rovent",b"x"),"mode":"cpu_pc",
                "pc":{"proposer":pin.clone(),"critic":pin,"validator":null,"backend":"rust_cpu",
                    "device":{"device":"cpu"},"backend_configuration_sha256":"d".repeat(64),
                    "runtime_artifacts":[],"fp32":true,"tf32":false,"batch_width":1,
                    "ort_intra_threads":null,"ort_inter_threads":null,"ort_spinning":null},
                "researcher":null,"teachers":[],"nnuec":null,
                "search":{"node_unit":"rovent-position-visits/1","requested_nodes":null,
                    "depth_plies":null,"mate_moves":null,"registered_max_chess_plies":64,"searchmoves":null,
                    "search_configuration_sha256":"e".repeat(64),"root_state_and_history_sha256":"f".repeat(64),
                    "rules_profile_sha256":"1".repeat(64)}
            },
            "resources":{"cpu_quota_millicpus":2000,"pools":[
                {"actors":["manager","pc_caller"],"max_threads":1,"cpu_charge_millicpus":1000},
                {"actors":["runtime_helper"],"max_threads":1,"cpu_charge_millicpus":1000}],
                "memory_high_bytes":3u64<<30,"memory_max_bytes":4u64<<30,"gpu_devices":0,
                "physical_nn_concurrency":1,"topology_policy":"shared-control/1"},
            "coordinator":{"policy_sha256":"2".repeat(64),"questions_max":64,"ancestor_depth_max":64,
                "waiting_jobs_max":64,"role_tickets_max":16,"cpu_credit_limit":0,
                "original_root_budget_sha256":"3".repeat(64),"original_deadline_seal_sha256":"4".repeat(64)},
            "wall_time_max_ms":120000,"cleanup_max_ms":30000,"output_bytes_max":1u64<<20,
            "agent_renewable":true,"provenance":provenance
        });
        RoventExecutionManifest::from_json(&value.to_string()).unwrap()
    }

    #[test]
    fn runtime_split_is_one_shot_and_uses_original_go_control_after_startup_ends() {
        use std::os::fd::AsRawFd;
        let fixture = runtime_fixture();
        let clock = ProcessClock::new(rz_contracts::ProcessEpoch(7201));
        let implementation = Digest([9; 32]);
        let mut custody = fixture.admit().unwrap();
        // Test only the private split core; do not fabricate a compiled/process
        // proof or a public RuntimeRegistration for the synthetic source files.
        let mut state = prepare_runtime_state(&custody, &clock, implementation).unwrap();
        let original_startup_until = custody.owners.control.until;
        let original_profile_fd = custody.owners().profile_file().unwrap().as_raw_fd();
        let original_source_fd = custody.owners().source_file(0).unwrap().as_raw_fd();
        let admission = original_go_admission(&fixture.lock, &clock, implementation);
        let before = prepare_setup_values(&custody, &state, &admission).unwrap();
        assert_eq!(
            before.control.deadline,
            admission.original_context().control.deadline
        );
        assert_eq!(before.control.max_io_bytes, NATIVE_MAINTENANCE_IO_BYTES);
        assert!(Arc::ptr_eq(
            &before.control.cancellation,
            &admission.original_context().control.cancellation
        ));
        // Mark the retained startup-only control stopped, without renewing it.
        // The independent live original go still owns its different cancel Arc.
        custody.owners.control.until = Instant::now();
        custody
            .owners
            .control
            .cancellation
            .store(true, Ordering::Release);
        assert!(custody.revalidate_startup().is_err());
        let setup = split_unused_setup(&mut custody, &mut state, Arc::clone(&admission)).unwrap();
        assert_eq!(setup.identity(), state.identity);
        assert_eq!(setup.runtime_limits(), before.limits);
        assert!(Arc::ptr_eq(
            state.issued_admission.as_ref().unwrap(),
            &admission
        ));
        assert!(!custody.retains_root_capability());
        assert_eq!(
            custody.owners().profile_file().unwrap().as_raw_fd(),
            original_profile_fd
        );
        assert_eq!(
            custody.owners().source_file(0).unwrap().as_raw_fd(),
            original_source_fd
        );
        assert!(custody.owners.control.until <= original_startup_until);
        assert!(custody.owners.control.cancellation.load(Ordering::Acquire));
        let duplicate = split_unused_setup(&mut custody, &mut state, Arc::clone(&admission))
            .err()
            .unwrap();
        assert!(matches!(
            duplicate.error,
            RoventStorageRuntimeError::AlreadyIssued
        ));
        assert!(Arc::ptr_eq(&duplicate.admission, &admission));
        // Creating/moving the payload starts no worker and creates no game.
        assert_eq!(fs::read_dir(&fixture.archive).unwrap().count(), 0);
        drop(setup);
    }

    #[test]
    fn runtime_split_rejects_other_opaque_lock_implementation_epoch_and_clock_origin() {
        let fixture = runtime_fixture();
        let clock = ProcessClock::new(rz_contracts::ProcessEpoch(7202));
        let implementation = Digest([10; 32]);
        let mut custody = fixture.admit().unwrap();
        let mut state = prepare_runtime_state(&custody, &clock, implementation).unwrap();
        std::thread::sleep(Duration::from_millis(1));
        let same_epoch_other_clock = ProcessClock::new(clock.epoch());
        assert_ne!(
            same_epoch_other_clock.runtime_clock().origin(),
            clock.runtime_clock().origin()
        );
        let other_epoch = ProcessClock::new(rz_contracts::ProcessEpoch(7203));
        let mut other_lock = fixture.lock.manifest.clone();
        other_lock.run_id = "other-original-startup-lock".into();
        let other_lock = other_lock.lock().unwrap();
        let admissions = [
            original_go_admission(&other_lock, &clock, implementation),
            original_go_admission(&fixture.lock, &clock, Digest([11; 32])),
            original_go_admission(&fixture.lock, &same_epoch_other_clock, implementation),
            original_go_admission(&fixture.lock, &other_epoch, implementation),
        ];
        for admission in admissions {
            let original_cap = custody.owners.capability.as_ref().unwrap() as *const _;
            let rejected = split_unused_setup(&mut custody, &mut state, Arc::clone(&admission))
                .err()
                .unwrap();
            assert!(matches!(
                rejected.error,
                RoventStorageRuntimeError::Namespace(_)
            ));
            assert!(Arc::ptr_eq(&rejected.admission, &admission));
            assert!(Arc::ptr_eq(custody.owners().input_lock(), &fixture.lock));
            assert_eq!(
                custody.owners.capability.as_ref().unwrap() as *const _,
                original_cap
            );
            assert!(state.issued_admission.is_none());
        }
        assert_eq!(fs::read_dir(&fixture.archive).unwrap().count(), 0);
    }

    #[test]
    fn runtime_split_stopped_original_go_preserves_the_complete_unissued_custody() {
        use std::os::fd::AsRawFd;
        let fixture = runtime_fixture();
        let clock = ProcessClock::new(rz_contracts::ProcessEpoch(7204));
        let implementation = Digest([12; 32]);
        let mut custody = fixture.admit().unwrap();
        let mut state = prepare_runtime_state(&custody, &clock, implementation).unwrap();
        let admission = original_go_admission(&fixture.lock, &clock, implementation);
        let fd = custody.owners().source_file(2).unwrap().as_raw_fd();
        admission
            .original_context()
            .control
            .cancellation
            .store(true, Ordering::Release);
        let rejected = split_unused_setup(&mut custody, &mut state, Arc::clone(&admission))
            .err()
            .unwrap();
        assert!(matches!(
            rejected.error,
            RoventStorageRuntimeError::GoNoLongerLive
        ));
        assert!(Arc::ptr_eq(&rejected.admission, &admission));
        assert!(state.issued_admission.is_none());
        assert!(custody.retains_root_capability());
        assert_eq!(custody.owners().source_file(2).unwrap().as_raw_fd(), fd);
        assert!(custody.revalidate_startup().is_ok());
        assert_eq!(fs::read_dir(&fixture.archive).unwrap().count(), 0);
    }

    #[test]
    fn runtime_state_keeps_exact_profile_and_rejects_smaller_native_maintenance() {
        let fixture = Fixture::new();
        let custody = fixture.admit().unwrap();
        let clock = ProcessClock::new(rz_contracts::ProcessEpoch(7205));
        let original = custody.profile().clone();
        let error = prepare_runtime_state(&custody, &clock, Digest([13; 32]))
            .err()
            .unwrap();
        assert!(matches!(
            error,
            RoventStorageRuntimeError::Compiled(
                RoventStorageCompiledError::UnsupportedMaintenanceIoBytes {
                    requested: 1048576,
                    ..
                }
            )
        ));
        assert_eq!(custody.profile(), &original);
        assert!(custody.retains_root_capability());
        assert!(Arc::ptr_eq(custody.owners().input_lock(), &fixture.lock));
        let fixture = runtime_fixture();
        let mut custody = fixture.admit().unwrap();
        let mut manifest = custody.owners.input_lock.manifest.clone();
        manifest.resources.pools.retain(|pool| {
            !pool
                .actors
                .contains(&rz_experiments::RoventActor::RuntimeHelper)
        });
        custody.owners.input_lock = Arc::new(manifest.lock().unwrap());
        assert!(prepare_runtime_state(&custody, &clock, Digest([13; 32])).is_err());
        assert!(custody.retains_root_capability());
    }

    #[test]
    fn compiled_binding_requires_original_lock_source_binary_and_private_root() {
        let fixture = Fixture::new();
        let lock = &fixture.lock;
        let aggregate = rz_experiments::rovent_source_aggregate_sha256(
            &lock.manifest.engine.source_commit,
            &lock.manifest.engine.source_components,
        )
        .unwrap();
        let private = directory_identity(&fixture.private).unwrap();
        // Pure comparison inputs only: no actual process observation or
        // checked-custody constructor is fabricated by this test.
        let mut binding = StorageProcessBinding {
            input_lock: compiled_digest(&lock.canonical_sha256).unwrap(),
            source_aggregate: compiled_digest(&aggregate).unwrap(),
            source_commit: &lock.manifest.engine.source_commit,
            source_scope: PROCESS_SOURCE_SCOPE,
            private_root: private,
            executable_bytes: lock.manifest.engine.binary.bytes,
            executable_sha256: &lock.manifest.engine.binary.sha256,
        };
        let components = lock.manifest.engine.source_components.clone();
        verify_original_process_binding(lock, private, &binding, &components).unwrap();
        binding.private_root.1 ^= 1;
        assert!(verify_original_process_binding(lock, private, &binding, &components).is_err());
        binding.private_root = private;
        binding.source_aggregate.0[31] ^= 1;
        assert!(verify_original_process_binding(lock, private, &binding, &components).is_err());
        binding.source_aggregate.0[31] ^= 1;
        binding.executable_bytes += 1;
        assert!(verify_original_process_binding(lock, private, &binding, &components).is_err());
        binding.executable_bytes -= 1;
        binding.source_commit = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
        assert!(verify_original_process_binding(lock, private, &binding, &components).is_err());
        binding.source_commit = &lock.manifest.engine.source_commit;
        let mut changed = lock.manifest.clone();
        changed.run_id = "different-original-lock".into();
        let changed = changed.lock().unwrap();
        assert!(verify_original_process_binding(&changed, private, &binding, &components).is_err());
        let mut components = components;
        components.insert("storage".into(), "0".repeat(64));
        assert!(verify_original_process_binding(lock, private, &binding, &components).is_err());
        components.clear();
        assert!(verify_original_process_binding(lock, private, &binding, &components).is_err());
    }

    #[test]
    fn failed_raw_compiled_comparison_keeps_original_verified_byte_owners() {
        use std::os::fd::AsRawFd;
        let fixture = Fixture::new();
        let admitted = fixture.admit().unwrap();
        let fd = admitted.owners().source_file(0).unwrap().as_raw_fd();
        let until = admitted.owners.control.until;
        let profile = admitted.profile();
        let error = verify_compiled_storage_sources(
            [
                &profile.source_pins.source,
                &profile.source_pins.codec,
                &profile.source_pins.launch,
            ],
            &rz_experiments::compiled_source_inventory_v5().unwrap(),
        )
        .unwrap_err();
        assert!(matches!(error, RoventStorageCompiledError::Mismatch(_)));
        assert_eq!(admitted.owners().source_file(0).unwrap().as_raw_fd(), fd);
        assert!(Arc::ptr_eq(admitted.owners().input_lock(), &fixture.lock));
        assert_eq!(admitted.owners.control.until, until);
        assert!(admitted.retains_root_capability());
        assert!(admitted.revalidate_startup().is_ok());
        assert_eq!(fs::read_dir(&fixture.archive).unwrap().count(), 0);
    }

    #[test]
    fn admission_holds_original_lock_fds_and_empty_root_without_enabling_archive() {
        let fixture = Fixture::new();
        let original_control = control();
        let admitted = admit_rovent_storage(
            Arc::clone(&fixture.lock),
            &fixture.private,
            fixture.request(),
            original_control.clone(),
        )
        .unwrap();
        assert!(Arc::ptr_eq(admitted.owners().input_lock(), &fixture.lock));
        assert_eq!(admitted.input_lock_sha256(), fixture.lock.canonical_sha256);
        assert_eq!(admitted.owners().verified_source_files(), 3);
        assert!(admitted.owners().profile_file().is_some());
        assert!(admitted.owners().retains_root_capability());
        assert_eq!(admitted.owners.control.until, original_control.until);
        assert!(Arc::ptr_eq(
            &admitted.owners.control.cancellation,
            &original_control.cancellation
        ));
        assert!(admitted.revalidate_startup().is_ok());
        assert_eq!(fs::read_dir(&fixture.archive).unwrap().count(), 0);
    }

    #[test]
    fn wrong_lock_raw_pin_and_duplicate_provenance_reject_without_profile_open() {
        let fixture = Fixture::new();
        let mut changed = (*fixture.lock).clone();
        changed.canonical_sha256 = "b".repeat(64);
        let wrong = Arc::new(changed);
        let failed = admit_rovent_storage(
            Arc::clone(&wrong),
            &fixture.private,
            fixture.request(),
            control(),
        )
        .err()
        .unwrap();
        assert!(Arc::ptr_eq(failed.owners.input_lock(), &wrong));
        assert!(failed.owners.profile_file().is_none());
        let mut request = fixture.request();
        request.profile_sha256 = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
        let failed = admit_rovent_storage(
            Arc::clone(&fixture.lock),
            &fixture.private,
            request,
            control(),
        )
        .err()
        .unwrap();
        assert!(failed.owners.profile_artifact().is_some());
        assert!(failed.owners.profile_file().is_none());
        let mut repeated = fixture.lock.manifest.clone();
        repeated.provenance.push(fixture.profile.clone());
        let repeated = Arc::new(repeated.lock().unwrap());
        let failed = admit_rovent_storage(repeated, &fixture.private, fixture.request(), control())
            .err()
            .unwrap();
        assert!(failed.owners.profile_file().is_none());
    }

    #[test]
    fn missing_second_source_preserves_profile_and_first_verified_fd() {
        let fixture = Fixture::new();
        fs::remove_file(fixture.private_path.join("inputs/codec.rs")).unwrap();
        let failed = fixture.admit().err().unwrap();
        assert!(failed.owners.profile_file().is_some());
        assert!(failed.owners.profile().is_some());
        assert_eq!(failed.owners.verified_source_files(), 1);
        assert!(failed.owners.source_file(0).is_some());
        assert!(failed.owners.source_file(1).is_none());
        assert!(!failed.owners.retains_root_capability());
        assert_eq!(fs::read_dir(&fixture.archive).unwrap().count(), 0);
    }

    #[test]
    fn exact_source_metadata_and_actual_private_directory_are_required() {
        let fixture = Fixture::new();
        let mut changed = fixture.lock.manifest.clone();
        changed
            .provenance
            .iter_mut()
            .find(|a| a.path == "inputs/codec.rs")
            .unwrap()
            .source = "https://example.org/different-origin".into();
        let failed = admit_rovent_storage(
            Arc::new(changed.lock().unwrap()),
            &fixture.private,
            fixture.request(),
            control(),
        )
        .err()
        .unwrap();
        assert_eq!(failed.owners.verified_source_files(), 1);
        let mut request = fixture.request();
        request.private_snapshot_root = &fixture.checkout;
        // The actual private capability still reads the original files; a
        // different declared private root must not impersonate that same Dir.
        let failed = admit_rovent_storage(
            Arc::clone(&fixture.lock),
            &fixture.private,
            request,
            control(),
        )
        .err()
        .unwrap();
        assert!(matches!(
            failed.error,
            RoventStorageAdmissionError::Invalid(_)
        ));
        assert_eq!(failed.owners.verified_source_files(), 3);
    }

    #[test]
    fn overlapping_roots_and_symlink_components_never_make_a_capability() {
        use std::os::unix::fs::symlink;
        let fixture = Fixture::new();
        for root in [&fixture.checkout, &fixture.private_path, &fixture.protected] {
            let mut request = fixture.request();
            request.archive_root = root;
            let failed = admit_rovent_storage(
                Arc::clone(&fixture.lock),
                &fixture.private,
                request,
                control(),
            )
            .err()
            .unwrap();
            assert!(matches!(
                failed.error,
                RoventStorageAdmissionError::Invalid(_)
            ));
            assert!(!failed.owners.retains_root_capability());
        }
        let alias = fixture._scratch.0.join("archive-alias");
        symlink(&fixture.archive, &alias).unwrap();
        let mut request = fixture.request();
        request.archive_root = &alias;
        let failed = admit_rovent_storage(
            Arc::clone(&fixture.lock),
            &fixture.private,
            request,
            control(),
        )
        .err()
        .unwrap();
        assert!(matches!(
            failed.error,
            RoventStorageAdmissionError::Io { .. }
        ));
        assert!(!failed.owners.retains_root_capability());
    }

    #[test]
    fn root_name_replacement_is_rejected_while_original_capability_is_retained() {
        let fixture = Fixture::new();
        let admitted = fixture.admit().unwrap();
        let old = fixture._scratch.0.join("original-archive");
        fs::rename(&fixture.archive, &old).unwrap();
        fs::create_dir(&fixture.archive).unwrap();
        assert!(matches!(
            admitted.revalidate_startup(),
            Err(RoventStorageAdmissionError::Invalid(_))
        ));
        assert!(admitted.owners().retains_root_capability());
        assert_eq!(fs::read_dir(&old).unwrap().count(), 0);
        assert_eq!(fs::read_dir(&fixture.archive).unwrap().count(), 0);
    }

    #[test]
    fn original_file_mutation_and_original_cancellation_are_not_renewed() {
        let fixture = Fixture::new();
        let original = control();
        let admitted = admit_rovent_storage(
            Arc::clone(&fixture.lock),
            &fixture.private,
            fixture.request(),
            original.clone(),
        )
        .unwrap();
        fs::write(
            fixture.private_path.join("inputs/source.json"),
            b"changed-source\n",
        )
        .unwrap();
        assert!(matches!(
            admitted.revalidate_startup(),
            Err(RoventStorageAdmissionError::Invalid(_))
        ));
        original.cancellation.store(true, Ordering::Release);
        assert!(matches!(
            admitted.revalidate_startup(),
            Err(RoventStorageAdmissionError::Cancelled)
        ));
        let failed = admit_rovent_storage(
            Arc::clone(&fixture.lock),
            &fixture.private,
            fixture.request(),
            original,
        )
        .err()
        .unwrap();
        assert!(matches!(
            failed.error,
            RoventStorageAdmissionError::Cancelled
        ));
        assert!(failed.owners.profile_file().is_none());
        let mut expired = control();
        expired.until = Instant::now();
        let failed = admit_rovent_storage(
            Arc::clone(&fixture.lock),
            &fixture.private,
            fixture.request(),
            expired,
        )
        .err()
        .unwrap();
        assert!(matches!(failed.error, RoventStorageAdmissionError::Expired));
        assert!(failed.owners.profile_file().is_none());
    }
}
