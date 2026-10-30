//! Actual Rovent startup source/binary boundary, independent of V5 receipts.
//!
//! This admits only the original process and registered embedded components.
//! It does not prove whole-engine semantic source coverage, create a model or
//! worker, attest neural readiness, or authorize functional acceptance. The
//! final session/CLI owner must register its own additional source component.
//! Participating owners keep private snapshots immutable; these observations
//! are not a hostile same-UID filesystem sandbox.

use super::{RoventStartupControl, compiled_bootstrap_source_digest};
#[cfg(target_os = "linux")]
use crate::process_receipts::ExecutableIdentitySourceV1;
use crate::process_receipts::ExecutableIdentityV1;
use cap_std::fs::Dir;
use rz_contracts::Digest;
use rz_experiments::{
    ArtifactRef, RoventEngineDeclaration, RoventInputLock, RoventInputLockV2, RoventResearcher,
};
use sha2::{Digest as _, Sha256};
use std::{
    collections::BTreeMap,
    fmt,
    fs::File,
    path::{Path, PathBuf},
    sync::{Arc, atomic::Ordering},
};

pub const PROCESS_SOURCE_COMPONENT: &str = "rovent_process_boundary";
pub const PROCESS_SOURCE_SCOPE: &str = "startup-boundary-and-registered-components/1";
// Same bounded executable profile as ExecutableIdentityV1::observe().
#[cfg(target_os = "linux")]
const MAX_EXECUTABLE_BYTES: u64 = 128 << 20;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProcessBoundaryFailure {
    UnsupportedMappedExecutable,
    Cancelled,
    Expired,
    InvalidInputLock,
    SourceCommitRegistration,
    SourceComponent,
    BinaryDeclaration,
    BinaryDescriptor,
    MappedExecutable,
    WorkingDirectory,
    SourceCheckout,
    Io,
}

#[derive(Debug)]
pub struct ProcessBoundaryError {
    pub kind: ProcessBoundaryFailure,
    pub stage: &'static str,
    pub io_kind: Option<std::io::ErrorKind>,
}
impl fmt::Display for ProcessBoundaryError {
    fn fmt(&self, out: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            out,
            "Rovent process boundary {:?}: {}",
            self.kind, self.stage
        )
    }
}
impl std::error::Error for ProcessBoundaryError {}

fn fail(kind: ProcessBoundaryFailure, stage: &'static str) -> ProcessBoundaryError {
    ProcessBoundaryError {
        kind,
        stage,
        io_kind: None,
    }
}
#[cfg(target_os = "linux")]
fn io(stage: &'static str, error: std::io::Error) -> ProcessBoundaryError {
    ProcessBoundaryError {
        kind: ProcessBoundaryFailure::Io,
        stage,
        io_kind: Some(error.kind()),
    }
}
fn require(
    valid: bool,
    kind: ProcessBoundaryFailure,
    stage: &'static str,
) -> Result<(), ProcessBoundaryError> {
    if valid {
        Ok(())
    } else {
        Err(fail(kind, stage))
    }
}
fn check_control(control: &RoventStartupControl) -> Result<(), ProcessBoundaryError> {
    require(
        !control.cancellation.load(Ordering::Acquire),
        ProcessBoundaryFailure::Cancelled,
        "original startup cancellation",
    )?;
    require(
        std::time::Instant::now() < control.until,
        ProcessBoundaryFailure::Expired,
        "original startup deadline",
    )
}

fn require_mapped_executable_profile() -> Result<(), ProcessBoundaryError> {
    #[cfg(target_os = "linux")]
    {
        Ok(())
    }
    #[cfg(not(target_os = "linux"))]
    {
        Err(fail(
            ProcessBoundaryFailure::UnsupportedMappedExecutable,
            "this platform does not provide the Linux mapped-executable proof",
        ))
    }
}

#[cfg(target_os = "linux")]
fn verify_mapped_executable(
    artifact: &ArtifactRef,
    executable: &ExecutableIdentityV1,
) -> Result<(), ProcessBoundaryError> {
    require_mapped_executable_profile()?;
    require(
        matches!(
            executable.identity_source,
            ExecutableIdentitySourceV1::LinuxProcSelfExe
        ) && executable.bytes == artifact.bytes
            && executable.sha256 == artifact.sha256,
        ProcessBoundaryFailure::MappedExecutable,
        "actual mapped executable differs from the retained binary bytes",
    )
}

pub fn compiled_process_boundary_source_digest() -> [u8; 32] {
    let bytes = include_bytes!("process_boundary.rs");
    let mut hash = Sha256::new();
    hash.update(b"rz-rovent-process-boundary-source/1\0");
    hash.update((bytes.len() as u64).to_le_bytes());
    hash.update(bytes);
    hash.finalize().into()
}

/// Actual embedded bytes only. An execution receipt must separately establish
/// which components were invoked and final CLI/session source coverage.
pub fn compiled_process_source_components() -> Result<BTreeMap<String, String>, ProcessBoundaryError>
{
    let mut actual = rz_experiments::compiled_source_components_v5().map_err(|_| {
        fail(
            ProcessBoundaryFailure::SourceComponent,
            "embedded source inventory",
        )
    })?;
    for (name, digest) in [
        ("rovent_bootstrap", compiled_bootstrap_source_digest()),
        (
            PROCESS_SOURCE_COMPONENT,
            compiled_process_boundary_source_digest(),
        ),
    ] {
        require(
            actual.insert(name.into(), hex(digest)).is_none(),
            ProcessBoundaryFailure::SourceComponent,
            "ambiguous embedded source component",
        )?;
    }
    Ok(actual)
}

fn verify_component_declaration(
    declared: &BTreeMap<String, String>,
    researcher: Option<&RoventResearcher>,
    nnuec_present: bool,
) -> Result<(), ProcessBoundaryError> {
    let actual = compiled_process_source_components()?;
    let mut required = vec![
        "rules",
        "store",
        "review_view",
        "coordinator",
        "role_tickets",
        "adaptive_costs",
        "resource_boundary",
        "uci",
        "model_native",
        "model_encoding",
        "nn_runtime",
        "rovent_bootstrap",
        PROCESS_SOURCE_COMPONENT,
    ];
    match researcher {
        Some(RoventResearcher::Own { .. }) => {
            required.extend(["cpu_search", "cpu_worker", "cpu_credit"]);
        }
        Some(RoventResearcher::External { .. }) => required.push("external_cpu"),
        None => {}
    }
    if nnuec_present && !required.contains(&"cpu_search") {
        // A PC-only shared readout registers its source without claiming R use.
        required.push("cpu_search");
    }
    require(
        (1..=64).contains(&declared.len())
            && required.iter().all(|key| declared.contains_key(*key))
            && declared
                .iter()
                .all(|(key, value)| actual.get(key) == Some(value)),
        ProcessBoundaryFailure::SourceComponent,
        "registered source components differ from actual embedded bytes",
    )
}

fn verify_commit_registration(declared: &str) -> Result<(), ProcessBoundaryError> {
    let registered = env!("RZ_PALS_COMPILED_SOURCE_COMMIT");
    let valid = |text: &str| {
        text.len() == 40
            && text
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    };
    require(
        valid(registered) && valid(declared) && declared == registered,
        ProcessBoundaryFailure::SourceCommitRegistration,
        "source commit differs from the build registration",
    )
}

/// Only the actual observation function below can create this admission.
/// Borrows keep the original immutable lock, binary FD and root FD alive.
pub struct VerifiedRoventProcessBoundary<'a> {
    lock: &'a RoventInputLock,
    original_binary: &'a File,
    original_root: &'a Dir,
    observation: Arc<ProcessBoundaryObservation>,
}

// Exactly one actual verification issues this private observation. Owned views
// share it, so equal digest fields from a different observation are not a
// transferable process proof.
struct ProcessBoundaryObservation {
    executable: ExecutableIdentityV1,
    input_lock: Digest,
    source_aggregate: Digest,
}
impl fmt::Debug for VerifiedRoventProcessBoundary<'_> {
    fn fmt(&self, out: &mut fmt::Formatter<'_>) -> fmt::Result {
        out.debug_struct("VerifiedRoventProcessBoundary")
            .field("input_lock", &self.observation.input_lock)
            .field("executable", &self.observation.executable)
            .field("scope", &PROCESS_SOURCE_SCOPE)
            .finish_non_exhaustive()
    }
}
impl VerifiedRoventProcessBoundary<'_> {
    pub fn input_lock_digest(&self) -> Digest {
        self.observation.input_lock
    }
    pub fn executable_identity(&self) -> &ExecutableIdentityV1 {
        &self.observation.executable
    }
    pub fn source_aggregate_digest(&self) -> Digest {
        self.observation.source_aggregate
    }
    pub fn registered_source_commit(&self) -> &str {
        &self.lock.manifest.engine.source_commit
    }
    pub fn source_scope(&self) -> &'static str {
        PROCESS_SOURCE_SCOPE
    }
    pub fn original_binary(&self) -> &File {
        self.original_binary
    }
    pub fn original_root(&self) -> &Dir {
        self.original_root
    }
}

/// Original launch inputs, moved together before actual verification. The owned
/// wrapper adds no descriptor reopen/clone to the existing actual observer.
/// These public inputs are declarations/owners only; constructing them does
/// not issue a boundary.
pub struct RoventProcessBoundaryInputs {
    pub lock: Arc<RoventInputLock>,
    pub original_binary: File,
    pub original_root: Dir,
    pub private_cwd: PathBuf,
    pub control: RoventStartupControl,
}

/// Owns the actual original lock, binary/root descriptors and startup control.
/// Only verify_actual_owned_process_boundary can issue this opaque value.
/// Keeping or dropping it proves neither worker completion nor physical drain.
/// CLI/session retention and unknown/quarantine ownership remain caller duties.
pub struct VerifiedOwnedRoventProcessBoundary {
    inputs: RoventProcessBoundaryInputs,
    observation: Arc<ProcessBoundaryObservation>,
}

impl fmt::Debug for VerifiedOwnedRoventProcessBoundary {
    fn fmt(&self, out: &mut fmt::Formatter<'_>) -> fmt::Result {
        out.debug_struct("VerifiedOwnedRoventProcessBoundary")
            .field("input_lock", &self.observation.input_lock)
            .field("executable", &self.observation.executable)
            .field("scope", &PROCESS_SOURCE_SCOPE)
            .finish_non_exhaustive()
    }
}

impl VerifiedOwnedRoventProcessBoundary {
    pub fn input_lock(&self) -> &Arc<RoventInputLock> {
        &self.inputs.lock
    }
    pub fn original_binary(&self) -> &File {
        &self.inputs.original_binary
    }
    pub fn original_root(&self) -> &Dir {
        &self.inputs.original_root
    }
    pub fn private_cwd(&self) -> &Path {
        &self.inputs.private_cwd
    }
    /// A borrowed view of this exact prior observation and these same original
    /// objects. It issues no fresh observation, deadline or runtime authority.
    pub fn startup_view(&self) -> Result<VerifiedRoventProcessBoundary<'_>, ProcessBoundaryError> {
        check_control(&self.inputs.control)?;
        Ok(VerifiedRoventProcessBoundary {
            lock: &self.inputs.lock,
            original_binary: &self.inputs.original_binary,
            original_root: &self.inputs.original_root,
            observation: Arc::clone(&self.observation),
        })
    }
    /// Repeat actual verification only inside the original startup window and
    /// using these held inputs. This does not replace the sealed observation.
    pub fn revalidate_startup(&self) -> Result<(), ProcessBoundaryError> {
        let actual = verify_actual_process_boundary(
            &self.inputs.lock,
            &self.inputs.lock.manifest.engine.binary,
            &self.inputs.original_binary,
            &self.inputs.private_cwd,
            &self.inputs.original_root,
            &self.inputs.control,
        )?;
        require(
            same_observed_values(&self.observation, &actual.observation),
            ProcessBoundaryFailure::MappedExecutable,
            "owned process observation changed during original startup",
        )
    }
    /// Require both exact owner objects and the same privately issued actual
    /// observation. Same bytes, inode numbers or independently verified values
    /// alone do not permit borrowed-to-owned storage registration transfer.
    pub(crate) fn matches_original_view(
        &self,
        view: &VerifiedRoventProcessBoundary<'_>,
        lock: &Arc<RoventInputLock>,
        control: &RoventStartupControl,
    ) -> Result<(), ProcessBoundaryError> {
        verify_original_startup_control(&self.inputs.control, control)?;
        verify_original_owner_objects(
            &self.inputs,
            view.lock,
            view.original_binary,
            view.original_root,
            lock,
        )?;
        verify_original_observation(&self.observation, &view.observation)
    }
}

fn verify_original_startup_control(
    original: &RoventStartupControl,
    registration: &RoventStartupControl,
) -> Result<(), ProcessBoundaryError> {
    check_control(original)?;
    require(
        original.until == registration.until
            && original.cleanup_until == registration.cleanup_until
            && Arc::ptr_eq(&original.cancellation, &registration.cancellation),
        ProcessBoundaryFailure::InvalidInputLock,
        "storage and owned process do not retain the original startup control",
    )
}

fn verify_original_owner_objects(
    original: &RoventProcessBoundaryInputs,
    lock: &RoventInputLock,
    binary: &File,
    root: &Dir,
    registration_lock: &Arc<RoventInputLock>,
) -> Result<(), ProcessBoundaryError> {
    require(
        Arc::ptr_eq(&original.lock, registration_lock)
            && std::ptr::eq(original.lock.as_ref(), lock),
        ProcessBoundaryFailure::InvalidInputLock,
        "owned process and storage do not retain the same original lock object",
    )?;
    require(
        std::ptr::eq(&original.original_binary, binary),
        ProcessBoundaryFailure::BinaryDescriptor,
        "storage view does not borrow the original owned binary object",
    )?;
    require(
        std::ptr::eq(&original.original_root, root),
        ProcessBoundaryFailure::WorkingDirectory,
        "storage view does not borrow the original owned root object",
    )
}

fn verify_original_observation(
    original: &Arc<ProcessBoundaryObservation>,
    view: &Arc<ProcessBoundaryObservation>,
) -> Result<(), ProcessBoundaryError> {
    require(
        Arc::ptr_eq(original, view),
        ProcessBoundaryFailure::MappedExecutable,
        "storage view is not the original owned process observation",
    )
}

fn same_observed_values(
    first: &ProcessBoundaryObservation,
    next: &ProcessBoundaryObservation,
) -> bool {
    first.input_lock == next.input_lock
        && first.source_aggregate == next.source_aggregate
        && first.executable.bytes == next.executable.bytes
        && first.executable.sha256 == next.executable.sha256
        && std::mem::discriminant(&first.executable.identity_source)
            == std::mem::discriminant(&next.executable.identity_source)
}

/// Failure returns every original input, including partial verification's FDs
/// and the unchanged cancel/deadline. No fallback proof or replacement owner.
pub struct RoventOwnedProcessBoundaryFailure {
    pub error: ProcessBoundaryError,
    pub inputs: RoventProcessBoundaryInputs,
}
impl fmt::Debug for RoventOwnedProcessBoundaryFailure {
    fn fmt(&self, out: &mut fmt::Formatter<'_>) -> fmt::Result {
        out.debug_struct("RoventOwnedProcessBoundaryFailure")
            .field("error", &self.error)
            .finish_non_exhaustive()
    }
}

/// Owned sibling of the existing borrowed verifier. Only that actual verifier
/// can create the sealed observation, with its existing OS probes. The wrapper
/// never reopens/replaces the retained inputs or recreates authority from a
/// numeric receipt.
// Preserve original FD/control custody directly on failure, without allocating
// an extra Box solely to shrink the error representation.
#[allow(clippy::result_large_err)]
pub fn verify_actual_owned_process_boundary(
    inputs: RoventProcessBoundaryInputs,
) -> Result<Arc<VerifiedOwnedRoventProcessBoundary>, RoventOwnedProcessBoundaryFailure> {
    let observed = verify_actual_process_boundary(
        &inputs.lock,
        &inputs.lock.manifest.engine.binary,
        &inputs.original_binary,
        &inputs.private_cwd,
        &inputs.original_root,
        &inputs.control,
    )
    .map(|actual| actual.observation);
    match observed {
        Ok(observation) => Ok(Arc::new(VerifiedOwnedRoventProcessBoundary {
            inputs,
            observation,
        })),
        Err(error) => Err(RoventOwnedProcessBoundaryFailure { error, inputs }),
    }
}

/// /2-owned inputs. The original lock, descriptors, and control are retained
/// together; constructing this value does not issue process authority.
pub struct RoventProcessBoundaryInputsV2 {
    pub lock: Arc<RoventInputLockV2>,
    pub original_binary: File,
    pub original_root: Dir,
    pub private_cwd: PathBuf,
    pub control: RoventStartupControl,
}

/// A borrowed view of one actual /2 process observation and its original
/// inputs. It cannot be reconstructed from a digest or a /1 observation.
pub struct VerifiedRoventProcessBoundaryV2<'a> {
    lock: &'a RoventInputLockV2,
    original_binary: &'a File,
    original_root: &'a Dir,
    observation: Arc<ProcessBoundaryObservation>,
}

impl VerifiedRoventProcessBoundaryV2<'_> {
    pub fn input_lock_digest(&self) -> Digest {
        self.observation.input_lock
    }
    pub fn executable_identity(&self) -> &ExecutableIdentityV1 {
        &self.observation.executable
    }
    pub fn source_aggregate_digest(&self) -> Digest {
        self.observation.source_aggregate
    }
    pub fn registered_source_commit(&self) -> &str {
        &self.lock.manifest.engine.source_commit
    }
    pub fn source_scope(&self) -> &'static str {
        PROCESS_SOURCE_SCOPE
    }
    pub fn original_binary(&self) -> &File {
        self.original_binary
    }
    pub fn original_root(&self) -> &Dir {
        self.original_root
    }
}

/// Owns the original /2 lock and actual process observation. This says nothing
/// about a model, worker, neural completion, or process teardown.
pub struct VerifiedOwnedRoventProcessBoundaryV2 {
    inputs: RoventProcessBoundaryInputsV2,
    observation: Arc<ProcessBoundaryObservation>,
}

impl fmt::Debug for VerifiedOwnedRoventProcessBoundaryV2 {
    fn fmt(&self, out: &mut fmt::Formatter<'_>) -> fmt::Result {
        out.debug_struct("VerifiedOwnedRoventProcessBoundaryV2")
            .field("input_lock", &self.observation.input_lock)
            .field("executable", &self.observation.executable)
            .field("scope", &PROCESS_SOURCE_SCOPE)
            .finish_non_exhaustive()
    }
}

impl VerifiedOwnedRoventProcessBoundaryV2 {
    /// Exact original cancellation allocation and the unrenewed deadlines.
    pub(crate) fn matches_startup_control(&self, other: &RoventStartupControl) -> bool {
        Arc::ptr_eq(&self.inputs.control.cancellation, &other.cancellation)
            && self.inputs.control.until == other.until
            && self.inputs.control.cleanup_until == other.cleanup_until
    }
    pub fn input_lock(&self) -> &Arc<RoventInputLockV2> {
        &self.inputs.lock
    }
    pub fn original_binary(&self) -> &File {
        &self.inputs.original_binary
    }
    pub fn original_root(&self) -> &Dir {
        &self.inputs.original_root
    }
    pub fn private_cwd(&self) -> &Path {
        &self.inputs.private_cwd
    }
    pub fn startup_view(
        &self,
    ) -> Result<VerifiedRoventProcessBoundaryV2<'_>, ProcessBoundaryError> {
        check_control(&self.inputs.control)?;
        Ok(VerifiedRoventProcessBoundaryV2 {
            lock: &self.inputs.lock,
            original_binary: &self.inputs.original_binary,
            original_root: &self.inputs.original_root,
            observation: Arc::clone(&self.observation),
        })
    }
    /// Re-observe only with the same held inputs, cancellation, and deadline.
    pub fn revalidate_startup(&self) -> Result<(), ProcessBoundaryError> {
        let actual = verify_actual_process_boundary_v2(
            &self.inputs.lock,
            &self.inputs.lock.manifest.engine.binary,
            &self.inputs.original_binary,
            &self.inputs.private_cwd,
            &self.inputs.original_root,
            &self.inputs.control,
        )?;
        require(
            same_observed_values(&self.observation, &actual.observation),
            ProcessBoundaryFailure::MappedExecutable,
            "owned /2 process observation changed during original startup",
        )
    }
}

/// A rejected /2 observation returns the same owner and every held input.
pub struct RoventOwnedProcessBoundaryFailureV2 {
    pub error: ProcessBoundaryError,
    pub inputs: RoventProcessBoundaryInputsV2,
}
impl fmt::Debug for RoventOwnedProcessBoundaryFailureV2 {
    fn fmt(&self, out: &mut fmt::Formatter<'_>) -> fmt::Result {
        out.debug_struct("RoventOwnedProcessBoundaryFailureV2")
            .field("error", &self.error)
            .finish_non_exhaustive()
    }
}

#[allow(clippy::result_large_err)]
pub fn verify_actual_owned_process_boundary_v2(
    inputs: RoventProcessBoundaryInputsV2,
) -> Result<Arc<VerifiedOwnedRoventProcessBoundaryV2>, RoventOwnedProcessBoundaryFailureV2> {
    let observed = verify_actual_process_boundary_v2(
        &inputs.lock,
        &inputs.lock.manifest.engine.binary,
        &inputs.original_binary,
        &inputs.private_cwd,
        &inputs.original_root,
        &inputs.control,
    )
    .map(|actual| actual.observation);
    match observed {
        Ok(observation) => Ok(Arc::new(VerifiedOwnedRoventProcessBoundaryV2 {
            inputs,
            observation,
        })),
        Err(error) => Err(RoventOwnedProcessBoundaryFailureV2 { error, inputs }),
    }
}

/// Observe the current process against the actual /2 lock and held inputs.
/// This is a typed sibling of /1, not a projection into a /1 lock.
pub fn verify_actual_process_boundary_v2<'a>(
    lock: &'a RoventInputLockV2,
    binary: &ArtifactRef,
    original_binary: &'a File,
    private_cwd: &Path,
    original_root: &'a Dir,
    control: &RoventStartupControl,
) -> Result<VerifiedRoventProcessBoundaryV2<'a>, ProcessBoundaryError> {
    check_control(control)?;
    require_mapped_executable_profile()?;
    lock.validate().map_err(|_| {
        fail(
            ProcessBoundaryFailure::InvalidInputLock,
            "original /2 input lock",
        )
    })?;
    let observation = verify_actual_process_observation(
        &lock.manifest.engine,
        &lock.canonical_sha256,
        binary,
        original_binary,
        private_cwd,
        original_root,
        control,
    )?;
    Ok(VerifiedRoventProcessBoundaryV2 {
        lock,
        original_binary,
        original_root,
        observation,
    })
}

/// Source/startup evidence only. The exact original startup deadline and cancel
/// owner are checked throughout bounded hashing; neither is renewed here.
pub fn verify_actual_process_boundary<'a>(
    lock: &'a RoventInputLock,
    binary: &ArtifactRef,
    original_binary: &'a File,
    private_cwd: &Path,
    original_root: &'a Dir,
    control: &RoventStartupControl,
) -> Result<VerifiedRoventProcessBoundary<'a>, ProcessBoundaryError> {
    check_control(control)?;
    require_mapped_executable_profile()?;
    lock.validate().map_err(|_| {
        fail(
            ProcessBoundaryFailure::InvalidInputLock,
            "original input lock",
        )
    })?;
    let observation = verify_actual_process_observation(
        &lock.manifest.engine,
        &lock.canonical_sha256,
        binary,
        original_binary,
        private_cwd,
        original_root,
        control,
    )?;
    Ok(VerifiedRoventProcessBoundary {
        lock,
        original_binary,
        original_root,
        observation,
    })
}

/// Actual process observation shared by typed lock callers after they have
/// validated their original lock. This private helper does not itself issue a
/// public boundary or replace the original lock/descriptor/control owners.
fn verify_actual_process_observation(
    engine: &RoventEngineDeclaration,
    canonical_sha256: &str,
    binary: &ArtifactRef,
    original_binary: &File,
    private_cwd: &Path,
    original_root: &Dir,
    control: &RoventStartupControl,
) -> Result<Arc<ProcessBoundaryObservation>, ProcessBoundaryError> {
    require(
        binary == &engine.binary,
        ProcessBoundaryFailure::BinaryDeclaration,
        "retained binary declaration differs from the original lock",
    )?;
    verify_commit_registration(&engine.source_commit)?;
    verify_component_declaration(
        &engine.source_components,
        engine.researcher.as_ref(),
        engine.nnuec.is_some(),
    )?;
    check_control(control)?;
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (
            canonical_sha256,
            original_binary,
            private_cwd,
            original_root,
        );
        Err(fail(
            ProcessBoundaryFailure::UnsupportedMappedExecutable,
            "this platform does not provide the Linux mapped-executable proof",
        ))
    }
    #[cfg(target_os = "linux")]
    {
        let root_identity = verify_private_cwd(private_cwd, original_root)?;
        let binary_identity = verify_binary_descriptor(binary, original_binary, control)?;
        let executable = ExecutableIdentityV1::observe().map_err(|error| ProcessBoundaryError {
            kind: ProcessBoundaryFailure::MappedExecutable,
            stage: "actual mapped executable observation",
            io_kind: error.io_kind,
        })?;
        check_control(control)?;
        verify_mapped_executable(binary, &executable)?;
        require(
            descriptor_identity(original_binary)? == binary_identity,
            ProcessBoundaryFailure::BinaryDescriptor,
            "original binary descriptor changed during process observation",
        )?;
        require(
            verify_private_cwd(private_cwd, original_root)? == root_identity,
            ProcessBoundaryFailure::WorkingDirectory,
            "original process working directory changed",
        )?;
        let source_aggregate = rz_experiments::rovent_source_aggregate_sha256(
            &engine.source_commit,
            &engine.source_components,
        )
        .map_err(|_| fail(ProcessBoundaryFailure::SourceComponent, "source aggregate"))?;
        let parse = |value: &str| {
            rz_eval::asset::parse_sha256(value)
                .map(Digest)
                .map_err(|_| {
                    fail(
                        ProcessBoundaryFailure::InvalidInputLock,
                        "validated identity digest",
                    )
                })
        };
        let input_lock = parse(canonical_sha256)?;
        let source_aggregate = parse(&source_aggregate)?;
        check_control(control)?;
        Ok(Arc::new(ProcessBoundaryObservation {
            executable,
            input_lock,
            source_aggregate,
        }))
    }
}

fn hex(bytes: [u8; 32]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

#[cfg(target_os = "linux")]
use std::os::unix::fs::{FileExt, MetadataExt};

#[cfg(target_os = "linux")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct DescriptorIdentity {
    device: u64,
    inode: u64,
    bytes: u64,
    modified: (i64, i64),
    changed: (i64, i64),
}
#[cfg(target_os = "linux")]
fn descriptor_identity(file: &File) -> Result<DescriptorIdentity, ProcessBoundaryError> {
    let metadata = file
        .metadata()
        .map_err(|error| io("original binary metadata", error))?;
    require(
        metadata.is_file(),
        ProcessBoundaryFailure::BinaryDescriptor,
        "original binary is not regular",
    )?;
    Ok(DescriptorIdentity {
        device: metadata.dev(),
        inode: metadata.ino(),
        bytes: metadata.len(),
        modified: (metadata.mtime(), metadata.mtime_nsec()),
        changed: (metadata.ctime(), metadata.ctime_nsec()),
    })
}
#[cfg(target_os = "linux")]
fn verify_binary_descriptor(
    artifact: &ArtifactRef,
    file: &File,
    control: &RoventStartupControl,
) -> Result<DescriptorIdentity, ProcessBoundaryError> {
    check_control(control)?;
    artifact.validate().map_err(|_| {
        fail(
            ProcessBoundaryFailure::BinaryDeclaration,
            "original binary artifact",
        )
    })?;
    let before = descriptor_identity(file)?;
    require(
        before.bytes == artifact.bytes && (1..=MAX_EXECUTABLE_BYTES).contains(&before.bytes),
        ProcessBoundaryFailure::BinaryDescriptor,
        "original binary exceeds its exact bounded declaration",
    )?;
    let mut hash = Sha256::new();
    let mut buffer = [0u8; 64 << 10];
    let mut offset = 0;
    while offset < before.bytes {
        check_control(control)?;
        let length = (before.bytes - offset).min(buffer.len() as u64) as usize;
        let count = file
            .read_at(&mut buffer[..length], offset)
            .map_err(|error| io("original binary positional read", error))?;
        require(
            count > 0,
            ProcessBoundaryFailure::BinaryDescriptor,
            "original binary ended early",
        )?;
        hash.update(&buffer[..count]);
        offset += count as u64;
    }
    require(
        file.read_at(&mut buffer[..1], offset)
            .map_err(|error| io("original binary end check", error))?
            == 0
            && hex(hash.finalize().into()) == artifact.sha256
            && descriptor_identity(file)? == before,
        ProcessBoundaryFailure::BinaryDescriptor,
        "original binary SHA or descriptor identity changed",
    )?;
    check_control(control)?;
    Ok(before)
}

#[cfg(target_os = "linux")]
fn directory_identity(file: &File) -> Result<(u64, u64), ProcessBoundaryError> {
    let metadata = file
        .metadata()
        .map_err(|error| io("working directory metadata", error))?;
    require(
        metadata.is_dir(),
        ProcessBoundaryFailure::WorkingDirectory,
        "working directory capability is not a directory",
    )?;
    Ok((metadata.dev(), metadata.ino()))
}
#[cfg(target_os = "linux")]
fn verify_private_cwd(private_cwd: &Path, root: &Dir) -> Result<(u64, u64), ProcessBoundaryError> {
    let actual =
        std::env::current_dir().map_err(|error| io("actual process working directory", error))?;
    let cwd = File::open(".").map_err(|error| io("actual cwd capability", error))?;
    verify_private_directory_at(private_cwd, &actual, &cwd, root)
}
#[cfg(target_os = "linux")]
fn verify_private_directory_at(
    private_cwd: &Path,
    actual: &Path,
    actual_cwd: &File,
    root: &Dir,
) -> Result<(u64, u64), ProcessBoundaryError> {
    require(
        private_cwd.is_absolute() && actual == private_cwd,
        ProcessBoundaryFailure::WorkingDirectory,
        "actual working directory differs from original private cwd",
    )?;
    // Preserve the legacy source-checkout exclusion using marker metadata only.
    for parent in actual.ancestors() {
        match std::fs::symlink_metadata(parent.join(".git")) {
            Ok(_) => {
                return Err(fail(
                    ProcessBoundaryFailure::SourceCheckout,
                    "private snapshot inside source checkout",
                ));
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(io("source checkout exclusion", error)),
        }
    }
    let held = root
        .try_clone()
        .map_err(|error| io("original root capability", error))?
        .into_std_file();
    let identity = directory_identity(&held)?;
    // No credential or repository contents are read. The real process cwd FD
    // and the named cwd must both still identify the original retained root.
    let named = File::open(actual).map_err(|error| io("named cwd capability", error))?;
    require(
        directory_identity(actual_cwd)? == identity && directory_identity(&named)? == identity,
        ProcessBoundaryFailure::WorkingDirectory,
        "actual cwd or named root differs from original root capability",
    )?;
    Ok(identity)
}

#[cfg(test)]
#[path = "process_boundary/tests.rs"]
mod tests;
