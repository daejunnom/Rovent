//! Parent-owned normalization of a completed pair. All four original session
//! observations and their native Rules/clock/exit receipt remain linked.
use super::*;
use std::time::Instant;

#[derive(Clone, Debug)]
pub(super) struct PalsAdaptiveRunTimingV5 {
    started: Instant,
    finished: Option<Instant>,
}
pub(super) fn begin_pals_run_timing_v5(lock: &LockedPalsArenaLaunchV5) -> Result<(), ArenaError> {
    let mut t = lock
        .timing
        .lock()
        .map_err(|_| invalid("run timing owner poisoned"))?;
    require(
        t.is_none(),
        "the original V5 run window cannot be restarted or retried",
    )?;
    *t = Some(PalsAdaptiveRunTimingV5 {
        started: Instant::now(),
        finished: None,
    });
    Ok(())
}
pub(super) fn finish_pals_run_timing_v5(
    t: &std::sync::Arc<std::sync::Mutex<Option<PalsAdaptiveRunTimingV5>>>,
) {
    if let Ok(mut t) = t.lock()
        && let Some(t) = t.as_mut()
    {
        t.finished = Some(Instant::now());
    }
}
pub(super) fn pals_original_wall_deadline_v5(
    lock: &LockedPalsArenaLaunchV5,
) -> Result<Instant, ArenaError> {
    let timing = lock
        .timing
        .lock()
        .map_err(|_| invalid("original run timing poisoned"))?;
    let timing = timing
        .as_ref()
        .ok_or_else(|| invalid("original finite run window not started"))?;
    require(
        timing.finished.is_none(),
        "completed run cannot dispatch a child",
    )?;
    timing
        .started
        .checked_add(std::time::Duration::from_millis(
            lock.input.semantic_lock.manifest.window.wall_time_max_ms,
        ))
        .ok_or_else(|| invalid("original run deadline overflow"))
}

#[derive(Clone, Debug, Serialize)]
pub struct PalsNormalizedEndpointScopeV5 {
    pub endpoint_id: String,
    pub aggregate_coordinator_evidence: ArtifactRef,
    pub process_ids: Vec<u32>,
    pub process_run_ids: Vec<String>,
    /// Resource/model readback is one exact representative process snapshot;
    /// every process snapshot remains in `sessions`, and all were admitted.
    pub representative_process_id: u32,
    pub accounting_scope: String,
}
#[derive(Clone, Debug, Serialize)]
pub struct PalsNormalizedPairReceiptV5 {
    pub domain: String,
    pub normalized: PalsRunReceiptV5,
    pub native_receipt: ArtifactRef,
    pub sessions: Vec<PalsAdaptiveSessionAuditV5>,
    pub endpoint_scopes: Vec<PalsNormalizedEndpointScopeV5>,
    pub artifacts: Vec<ArtifactRef>,
    pub elapsed_scope: String,
    pub raw_game_failures_preserved: bool,
    pub training_executed: bool,
}

/// Failed evidence never grants acceptance. The original borrowed failure
/// continues to own its process/input/physical quarantine scope; this DTO is
/// only a diagnostic projection. A missing stored receipt stays unknown.
#[derive(Clone, Debug, Serialize)]
pub struct PalsNormalizedFailedPairReceiptV5 {
    pub domain: String,
    pub normalized: PalsRunReceiptV5,
    pub original_cause: String,
    pub raw_native_receipt: Option<crate::NativePairReceipt<PalsAdaptiveSessionAuditV5>>,
    pub native_receipt_proof: PalsObservedV3<ArtifactRef>,
    pub supervisor_process: PalsObservedV3<crate::ProcessReceipt>,
    pub observed_run_time_ms: PalsObservedV3<u64>,
    pub observed_cleanup_time_ms: PalsObservedV3<u64>,
    pub artifacts: Vec<ArtifactRef>,
    pub original_owner_not_consumed: bool,
    pub physical_disposition: String,
    pub normalization_notes: Vec<String>,
    pub persistence_not_permitted_reason: Option<String>,
    pub training_executed: bool,
}

#[cfg(target_os = "linux")]
fn native_receipt_belongs_to_owner(
    native: &crate::NativePairReceipt<PalsAdaptiveSessionAuditV5>,
    owner: &NativeLaunchOwner<LockedPalsArenaLaunchV5>,
) -> bool {
    native.receipt_version == owner.spec.receipt_version()
        && native.input_sha256 == owner.spec.input_sha256()
        && native.pair_id == owner.spec.input.semantic_lock.manifest.run_id
        && native.contract_revision == owner.spec.contract_revision()
        && native.runner_source_commit == owner.spec.input.runner.source_commit
        && native.runner_binary_sha256 == owner.spec.input.runner.binary.sha256
}

#[cfg(target_os = "linux")]
type PalsFailurePartsV5<'a> = (
    &'a NativeLaunchOwner<LockedPalsArenaLaunchV5>,
    &'a ArenaError,
    Option<&'a crate::NativePairReceipt<PalsAdaptiveSessionAuditV5>>,
    Option<&'a ArtifactRef>,
    Option<&'a crate::ProcessOutput>,
    bool,
);

#[cfg(target_os = "linux")]
fn failure_parts(
    failure: &PalsAdaptivePairFailureV5,
) -> Result<PalsFailurePartsV5<'_>, ArenaError> {
    match failure {
        PalsAdaptivePairFailureV5::Admission { cause, owner } => {
            Ok((owner, cause, None, None, None, true))
        }
        PalsAdaptivePairFailureV5::Execution(failure) => Ok((
            failure
                .launch_owner()
                .ok_or_else(|| invalid("original failed launch owner unavailable"))?,
            &failure.cause,
            failure.receipt.as_ref(),
            failure.receipt_artifact.as_ref(),
            failure.process(),
            false,
        )),
    }
}

/// Reads only parent-owned bounded receipt bytes and passive retained process
/// observations. It never drops, joins, signals, retries or clears a failure.
/// Unknown timing is explicitly retained outside the legacy numeric envelope;
/// its zero placeholder cannot be presented as a measured zero-cost execution.
#[cfg(target_os = "linux")]
pub fn build_pals_failed_run_receipt_v5(
    failure: &PalsAdaptivePairFailureV5,
) -> Result<PalsNormalizedFailedPairReceiptV5, ArenaError> {
    let (owner, cause, native, artifact, process, admission_failure) = failure_parts(failure)?;
    let lock = &owner.spec.input.semantic_lock;
    let mut receipt = PalsRunReceiptV5::pending(lock)?;
    receipt.failures.insert(PalsFailureV5::Infrastructure);
    let mut notes = Vec::new();
    let mut proof = PalsObservedV3::Unknown;
    if let (Some(native), Some(artifact)) = (native, artifact) {
        let identity_matches = native_receipt_belongs_to_owner(native, owner);
        if !identity_matches {
            receipt
                .failures
                .insert(PalsFailureV5::SourceIdentityViolation);
        }
        let checked = (|| -> Result<(), ArenaError> {
            artifact.validate()?;
            require(
                artifact.path
                    == format!(
                        "{}/{}",
                        owner.snapshot.output_directory,
                        owner.spec.receipt_filename()
                    ),
                "failed native receipt ArtifactRef does not name the original owner file",
            )?;
            require(
                identity_matches,
                "failed native receipt belongs to a different original input/pair/runner/contract",
            )?;
            let bytes = crate::native_runner::linux::read_file(
                &owner.snapshot.directory,
                owner.spec.receipt_filename(),
                owner.spec.pair_metadata_cap(),
            )?;
            require(
                bytes.len() as u64 == artifact.bytes
                    && format!("{:x}", Sha256::digest(&bytes)) == artifact.sha256,
                "stored failure receipt byte identity differs",
            )?;
            let actual: serde_json::Value = crate::decode_json(
                std::str::from_utf8(&bytes).map_err(|_| invalid("failure receipt not UTF8"))?,
            )?;
            require(
                actual
                    == serde_json::to_value(native)
                        .map_err(|_| invalid("failure projection encode failed"))?,
                "retained failure projection differs from stored receipt",
            )
        })();
        match checked {
            Ok(()) => proof = PalsObservedV3::Observed {
                value: artifact.clone(),
                method: "actual stored parent failure receipt hash/length/typed projection checked".into(),
            },
            Err(error) => notes.push(format!("stored native proof unavailable: {error}; raw projection retained without stored-file authority")),
        }
    }
    let mut elapsed = PalsObservedV3::Unknown;
    if let Ok(timing) = owner.spec.timing.lock()
        && let Some(timing) = timing.as_ref()
        && let Some(end) = timing.finished
    {
        let ms = millis_ceil(end.duration_since(timing.started).as_nanos())?;
        receipt.wall_time_ms = ms;
        elapsed = PalsObservedV3::Observed {
            value: ms, method: "original run-entry Instant through failed executor return; excludes snapshot preparation".into(),
        };
        if ms > lock.manifest.window.wall_time_max_ms {
            receipt.failures.insert(PalsFailureV5::Expired);
        }
    }
    let mut cleanup = PalsObservedV3::Unknown;
    if let Some(p) = process {
        if let Some(ns) = p.receipt.cleanup_elapsed_ns {
            let ms = millis_ceil(ns as u128)?;
            receipt.cleanup_time_ms = ms;
            cleanup = PalsObservedV3::Observed {
                value: ms,
                method: "retained parent process supervisor cleanup interval".into(),
            };
            if ms > lock.manifest.window.cleanup_max_ms {
                receipt.failures.insert(PalsFailureV5::ProcessCleanup);
            }
        }
        match p.receipt.stop {
            crate::ProcessStop::Cancelled => {
                receipt.failures.insert(PalsFailureV5::Cancelled);
            }
            crate::ProcessStop::WallLimit => {
                receipt.failures.insert(PalsFailureV5::Expired);
            }
            crate::ProcessStop::OutputLimit | crate::ProcessStop::ArtifactLimit => {
                receipt.failures.insert(PalsFailureV5::OutputQuota);
            }
            crate::ProcessStop::ChildLimit => {
                receipt.failures.insert(PalsFailureV5::ResourceAdmission);
            }
            _ => {}
        }
    }
    let tree_closed = process.is_some_and(|p| {
        p.receipt.group_cleanup == crate::CleanupStatus::Gone && p.pending_child.is_none()
    });
    let provider_closed = observed(&proof).is_some()
        && native.is_some_and(|native| {
            native.provider_sessions.len() == 4
                && native.provider_sessions.iter().all(|s| {
                    observed(&s.endpoint.coordinator).is_some_and(|c| {
                        c.physical_state == PalsPhysicalStateV3::Completed
                            && c.physical_completion_unknown == 0
                            && c.tickets_quarantined == 0
                            && c.physical_in_flight == 0
                            && c.cpu_outstanding_reserved == 0
                            && c.tickets_queued == 0
                            && c.tickets_released == c.tickets_submitted
                            && c.owner_joined
                            && c.archive_closed
                    })
                })
        });
    if !admission_failure && !provider_closed {
        receipt
            .failures
            .insert(PalsFailureV5::PhysicalCompletionUnknown);
    }
    let disposition = if admission_failure {
        "no child dispatch was admitted"
    } else if tree_closed {
        "parent process tree cleanup observed; provider closure still has no acceptance"
    } else {
        receipt
            .failures
            .insert(PalsFailureV5::PhysicalCompletionUnknown);
        "original owner/process/input scope retained; physical completion unknown"
    };
    let admission = owner
        .spec
        .admission
        .lock()
        .map_err(|_| invalid("failure source admission poisoned"))?
        .clone();
    if let Some(admission) = &admission {
        receipt.source_verification = PalsObservedV3::Observed {
            value: admission.source_verification.clone(),
            method: "actual prechild FD verification retained independently of failed execution"
                .into(),
        };
    }
    // No partial native result is aggregated into an endpoint accepted record.
    // Complete audited games may be represented only from an exact stored proof;
    // otherwise the raw native failure (including engine offender) remains linked.
    if observed(&proof).is_some()
        && let Some(native) = native
        && let (Some(pgn), Some(audit), Some(clock), Some(pair)) = (
            native
                .artifacts
                .iter()
                .find(|a| a.path.ends_with("/match.pgn")),
            native.pgn_audit.as_ref(),
            native.clock_audit.as_ref(),
            lock.manifest.pair.as_ref(),
        )
        && native.pgn_audit_error.is_none()
        && native.clock_audit_error.is_none()
        && audit.games.len() <= 2
        && audit.games.len() == clock.games.len()
    {
        for (index, game) in audit.games.iter().enumerate() {
            if game.game_id != clock.games[index].game_id {
                notes.push("Rules/clock game identity differed; raw game retained without normalized result".into());
                continue;
            }
            match game_receipt(game, index, pair, pgn, artifact.unwrap()) {
                Ok(game) => receipt.games.push(game),
                Err(error) => notes.push(format!("unresolved game remains raw: {error}")),
            }
        }
    }
    if receipt.games.len() != 2
        || receipt
            .games
            .iter()
            .any(|g| g.result == PalsResultV3::Incomplete)
    {
        receipt.failures.insert(PalsFailureV5::IncompleteGame);
    }
    let mut artifacts = BTreeMap::new();
    if let Some(native) = native {
        for a in &native.artifacts {
            artifacts.insert(a.path.clone(), a.clone());
        }
    }
    if let Some(a) = observed(&proof) {
        artifacts.insert(a.path.clone(), a.clone());
    }
    if let Some(admission) = admission {
        let a = admission.source_verification.evidence;
        artifacts.insert(a.path.clone(), a);
    }
    if admission_failure || tree_closed {
        match owner_output_bytes(owner) {
            Ok(n) => {
                receipt.output_bytes = PalsObservedV3::Observed {
                    value:n,method:"bounded no-follow owned output length scan after no-child admission failure or observed process cleanup; inputs excluded".into(),
                };
                if n > lock.manifest.window.output_bytes_max {
                    receipt.failures.insert(PalsFailureV5::OutputQuota);
                }
            }
            Err(error) => notes.push(format!("output length remains unknown: {error}")),
        }
    }
    receipt.validate_against(lock)?;
    Ok(PalsNormalizedFailedPairReceiptV5 {
        domain:match owner.spec.storage_policy { PalsOwnedStoragePolicy::StrictV5 => "rz-pals-normalized-pair-failure-v5/1", PalsOwnedStoragePolicy::AdvisoryV1 => "rz-pals-normalized-pair-failure-storage-advisory-v1/1" }.into(), normalized:receipt,
        original_cause:cause.to_string(),raw_native_receipt:native.cloned(),native_receipt_proof:proof,
        supervisor_process:process.map(|p|PalsObservedV3::Observed { value:p.receipt.clone(),method:"passive observation of original retained supervisor output; no cleanup operation performed".into() }).unwrap_or(PalsObservedV3::Unknown),
        observed_run_time_ms:elapsed,observed_cleanup_time_ms:cleanup,artifacts:artifacts.into_values().collect(),
        original_owner_not_consumed:true,physical_disposition:disposition.into(),normalization_notes:notes,
        persistence_not_permitted_reason:(!admission_failure && !tree_closed).then(||"physical/process completion unknown; keep memory projection and original quarantine owner, do not write into changing child scope".into()),
        training_executed:false,
    })
}

/// Persists only when the original process scope is closed and output quota
/// remains. A missing optional artifact means the typed failed memory receipt
/// and original owner remain available; it is never a retry or success.
#[cfg(target_os = "linux")]
pub fn save_pals_failed_run_receipt_v5(
    failure: &PalsAdaptivePairFailureV5,
) -> Result<(PalsNormalizedFailedPairReceiptV5, Option<ArtifactRef>), ArenaError> {
    let mut wrapper = build_pals_failed_run_receipt_v5(failure)?;
    if wrapper.persistence_not_permitted_reason.is_some() {
        return Ok((wrapper, None));
    }
    let (owner, _, _, _, _, _) = failure_parts(failure)?;
    let current = match owner_output_bytes(owner) {
        Ok(n) => n,
        Err(error) => {
            wrapper.persistence_not_permitted_reason = Some(error.to_string());
            return Ok((wrapper, None));
        }
    };
    let mut total = current;
    let mut bytes = Vec::new();
    for _ in 0..4 {
        wrapper.normalized.output_bytes=PalsObservedV3::Observed { value:total,method:"actual owned output bytes plus this parent failure receipt; original inputs excluded".into() };
        if total
            > owner
                .spec
                .input
                .semantic_lock
                .manifest
                .window
                .output_bytes_max
        {
            wrapper
                .normalized
                .failures
                .insert(PalsFailureV5::OutputQuota);
        }
        wrapper
            .normalized
            .validate_against(&owner.spec.input.semantic_lock)?;
        bytes =
            serde_json::to_vec(&wrapper).map_err(|_| invalid("failure wrapper encode failed"))?;
        let next = current
            .checked_add(bytes.len() as u64)
            .ok_or_else(|| invalid("failure receipt size overflow"))?;
        if next == total {
            break;
        }
        total = next;
    }
    if observed(&wrapper.normalized.output_bytes)
        != Some(
            &current
                .checked_add(bytes.len() as u64)
                .ok_or_else(|| invalid("failure size overflow"))?,
        )
    {
        wrapper.persistence_not_permitted_reason =
            Some("failure receipt length fixpoint unavailable; original evidence unchanged".into());
        wrapper.normalized.output_bytes = PalsObservedV3::Observed {
            value: current, method:"actual owned output before unpersisted failed normalization; no proposed receipt bytes counted".into(),
        };
        return Ok((wrapper, None));
    }
    let filename = match owner.spec.storage_policy {
        PalsOwnedStoragePolicy::StrictV5 => "pals-run-failure.v5.json",
        PalsOwnedStoragePolicy::AdvisoryV1 => "pals-run-failure.storage-advisory-v1.json",
    };
    match save_parent(owner, filename, &bytes) {
        Ok(artifact) => Ok((wrapper, Some(artifact))),
        Err(error) => {
            wrapper.persistence_not_permitted_reason = Some(error.to_string());
            wrapper.normalized.output_bytes = match owner_output_bytes(owner) {
                Ok(n)=>PalsObservedV3::Observed { value:n,method:"actual owned output after failed parent save, including any retained partial file; no proposed full receipt size counted".into() },
                Err(_)=>PalsObservedV3::Unknown,
            };
            Ok((wrapper, None))
        }
    }
}

#[cfg(not(target_os = "linux"))]
pub fn build_pals_failed_run_receipt_v5(
    _: &PalsAdaptivePairFailureV5,
) -> Result<PalsNormalizedFailedPairReceiptV5, ArenaError> {
    Err(invalid(
        "actual failed V5 evidence normalization requires Linux-owned runner scope",
    ))
}
#[cfg(not(target_os = "linux"))]
pub fn save_pals_failed_run_receipt_v5(
    _: &PalsAdaptivePairFailureV5,
) -> Result<(PalsNormalizedFailedPairReceiptV5, Option<ArtifactRef>), ArenaError> {
    Err(invalid(
        "actual failed V5 evidence persistence requires Linux-owned runner scope",
    ))
}

#[cfg(any(target_os = "linux", test))]
fn millis_ceil(ns: u128) -> Result<u64, ArenaError> {
    let value = ns
        .checked_add(999_999)
        .ok_or_else(|| invalid("duration overflow"))?
        / 1_000_000;
    u64::try_from(value).map_err(|_| invalid("duration exceeds u64"))
}
#[cfg(target_os = "linux")]
fn observed<T>(v: &PalsObservedV3<T>) -> Option<&T> {
    if let PalsObservedV3::Observed { value, .. } = v {
        Some(value)
    } else {
        None
    }
}

/// A bounded length-only walk of the already registered, exclusively owned
/// attempt. The read-only `inputs` copies have their own input reservation;
/// every runtime/cache/archive/log/receipt file outside that subtree is counted.
/// Artifact lists are references, not an authority to omit unlisted files.
#[cfg(target_os = "linux")]
fn owned_output_bytes(directory: &cap_std::fs::Dir) -> Result<u64, ArenaError> {
    use cap_fs_ext::{DirExt, FollowSymlinks, OpenOptionsFollowExt};
    use cap_std::fs::{OpenOptions, OpenOptionsExt};
    use std::os::unix::fs::MetadataExt;
    fn walk(dir: &cap_std::fs::Dir, depth: usize, entries: &mut usize) -> Result<u64, ArenaError> {
        require(depth <= 64, "normalized output depth exceeds bounded scope")?;
        let mut bytes = 0u64;
        for entry in dir
            .entries()
            .map_err(|_| invalid("owned output enumeration failed"))?
        {
            *entries = entries
                .checked_add(1)
                .ok_or_else(|| invalid("output entry overflow"))?;
            require(
                *entries <= 100_000,
                "normalized output entry bound exceeded",
            )?;
            let entry = entry.map_err(|_| invalid("owned output entry unavailable"))?;
            let name = entry.file_name();
            let metadata = dir
                .symlink_metadata(&name)
                .map_err(|_| invalid("owned output entry changed during observation"))?;
            if depth == 0 && name == "inputs" {
                require(
                    metadata.is_dir(),
                    "registered input subtree became a link or file",
                )?;
                continue;
            }
            let amount = if metadata.is_dir() {
                let child = dir
                    .open_dir_nofollow(&name)
                    .map_err(|_| invalid("owned output directory changed"))?;
                walk(&child, depth + 1, entries)?
            } else {
                require(
                    metadata.is_file(),
                    "owned output has an unsupported file kind",
                )?;
                let mut opts = OpenOptions::new();
                opts.read(true)
                    .follow(FollowSymlinks::No)
                    .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
                let file = dir
                    .open_with(&name, &opts)
                    .map_err(|_| invalid("owned output regular file changed during pin"))?
                    .into_std();
                let metadata = file
                    .metadata()
                    .map_err(|_| invalid("owned output pinned length unavailable"))?;
                require(
                    metadata.is_file() && metadata.nlink() == 1,
                    "owned output contains a link, alias or unsupported entry",
                )?;
                metadata.len()
            };
            bytes = bytes
                .checked_add(amount)
                .ok_or_else(|| invalid("output length overflow"))?;
            require(
                bytes <= PALS_STAGING_BYTES_V5,
                "owned output exceeds registered staging cap",
            )?;
        }
        Ok(bytes)
    }
    walk(directory, 0, &mut 0)
}

#[cfg(target_os = "linux")]
fn owner_output_bytes(
    owner: &NativeLaunchOwner<LockedPalsArenaLaunchV5>,
) -> Result<u64, ArenaError> {
    crate::native_launch::linux::verify_input_directories(
        &owner.snapshot.directory,
        &owner.snapshot.input_directory,
        &owner.snapshot.input_subdirectories,
    )?;
    owned_output_bytes(&owner.snapshot.directory)
}

#[cfg(any(target_os = "linux", test))]
fn merge_peak(
    a: &serde_json::Value,
    b: &serde_json::Value,
) -> Result<serde_json::Value, ArenaError> {
    if let (Some(a), Some(b)) = (a.as_u64(), b.as_u64()) {
        return Ok(a.max(b).into());
    }
    let known = |v: &serde_json::Value| -> Option<u64> {
        (v["status"] == "observed")
            .then(|| v["value"].as_u64())
            .flatten()
    };
    match (known(a), known(b)) {
        (Some(a), Some(b)) => Ok(
            serde_json::json!({"status":"observed","value":a.max(b),"method":"maximum of two independently observed game-session peaks"}),
        ),
        _ if a["status"] == "unknown" || b["status"] == "unknown" => {
            Ok(serde_json::json!({"status":"unknown"}))
        }
        _ => Err(invalid(
            "coordinator peak has an unsupported observation type",
        )),
    }
}
#[cfg(target_os = "linux")]
fn aggregate_coordinator(
    a: &PalsCoordinatorObservationV5,
    b: &PalsCoordinatorObservationV5,
    evidence: ArtifactRef,
) -> Result<PalsCoordinatorObservationV5, ArenaError> {
    require(
        a.owner_joined
            && b.owner_joined
            && a.archive_closed
            && b.archive_closed
            && a.physical_state == PalsPhysicalStateV3::Completed
            && b.physical_state == PalsPhysicalStateV3::Completed,
        "unclosed physical session cannot be aggregated",
    )?;
    let mut x =
        serde_json::to_value(a).map_err(|_| invalid("coordinator aggregate encode failed"))?;
    let y = serde_json::to_value(b).map_err(|_| invalid("coordinator aggregate encode failed"))?;
    for (key, value) in x
        .as_object_mut()
        .ok_or_else(|| invalid("coordinator is not an object"))?
    {
        if key == "evidence" {
            *value =
                serde_json::to_value(&evidence).map_err(|_| invalid("evidence encode failed"))?;
        } else if key.ends_with("_peak") {
            *value = merge_peak(value, &y[key])?;
        } else if matches!(
            key.as_str(),
            "physical_state" | "owner_joined" | "archive_closed"
        ) {
            require(*value == y[key], "coordinator shutdown values differ")?;
        } else {
            let left = value
                .as_u64()
                .ok_or_else(|| invalid("unregistered coordinator aggregate field type"))?;
            let right = y[key]
                .as_u64()
                .ok_or_else(|| invalid("unregistered coordinator aggregate field type"))?;
            *value = left
                .checked_add(right)
                .ok_or_else(|| invalid("coordinator aggregate count overflow"))?
                .into();
        }
    }
    serde_json::from_value(x).map_err(|_| invalid("aggregate coordinator schema differs"))
}

#[cfg(target_os = "linux")]
fn save_parent(
    owner: &NativeLaunchOwner<LockedPalsArenaLaunchV5>,
    name: &str,
    bytes: &[u8],
) -> Result<ArtifactRef, ArenaError> {
    use cap_fs_ext::{FollowSymlinks, OpenOptionsFollowExt};
    use cap_std::fs::{OpenOptions, OpenOptionsExt};
    use std::io::Write;
    require(
        !name.contains('/') && name.ends_with(".json") && bytes.len() <= 2 << 20,
        "normalized parent artifact exceeds reserved metadata scope",
    )?;
    require(
        owner.spec.storage_lease.is_some(),
        "parent writer has no owned storage lease",
    )?;
    let current = owner_output_bytes(owner)?;
    require(
        current.checked_add(bytes.len() as u64).is_some_and(|n| {
            n <= owner
                .spec
                .input
                .semantic_lock
                .manifest
                .window
                .output_bytes_max
        }),
        "parent normalization has no remaining registered output quota; original evidence retained",
    )?;
    let mut opts = OpenOptions::new();
    opts.write(true)
        .create_new(true)
        .follow(FollowSymlinks::No)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
        .mode(0o600);
    let mut file = owner
        .snapshot
        .directory
        .open_with(name, &opts)
        .map_err(|_| invalid("parent normalization artifact exists or cannot be created"))?;
    file.write_all(bytes)
        .and_then(|()| file.sync_all())
        .map_err(|_| invalid("parent normalization artifact save failed"))?;
    Ok(ArtifactRef {
        path: format!("{}/{name}", owner.snapshot.output_directory),
        sha256: format!("{:x}", Sha256::digest(bytes)),
        bytes: bytes.len() as u64,
        source: "https://github.com/daejunnom/RoveZero".into(),
        license: "MIT".into(),
    })
}
#[cfg(target_os = "linux")]
fn game_receipt(
    g: &crate::GamePgnAudit,
    index: usize,
    pair: &PalsPairPlanV5,
    pgn: &ArtifactRef,
    audit: &ArtifactRef,
) -> Result<PalsGameReceiptV5, ArenaError> {
    let (result, termination, failed) = match g.classification.as_str() {
        "rules_terminal" | "accepted_claim" => {
            let result = match g.result {
                crate::GameResult::WhiteWin => PalsResultV3::WhiteWin,
                crate::GameResult::BlackWin => PalsResultV3::BlackWin,
                crate::GameResult::Draw => PalsResultV3::Draw,
            };
            let termination = if g.terminal_reason.as_deref() == Some("checkmate") {
                PalsTerminationV3::Checkmate
            } else {
                require(
                    result == PalsResultV3::Draw,
                    "non-mate Rules ending must be a draw",
                )?;
                PalsTerminationV3::RulesDraw
            };
            (result, termination, None)
        }
        "engine_loss" => {
            let result = match g.result {
                crate::GameResult::WhiteWin => PalsResultV3::WhiteWin,
                crate::GameResult::BlackWin => PalsResultV3::BlackWin,
                _ => return Err(invalid("engine loss cannot become draw")),
            };
            let termination = match g.engine_failure {
                Some(crate::EngineFailureKind::IllegalMove) => PalsTerminationV3::IllegalMove,
                Some(crate::EngineFailureKind::Crash) => PalsTerminationV3::EngineCrash,
                Some(crate::EngineFailureKind::Timeout) => PalsTerminationV3::TimeForfeit,
                None => return Err(invalid("engine loss has no actual failure kind")),
            };
            (
                result,
                termination,
                Some(
                    g.loser_engine
                        .clone()
                        .ok_or_else(|| invalid("engine loss lacks offender"))?,
                ),
            )
        }
        "incomplete" => {
            require(
                g.uci_moves.len() as u32 == pair.max_plies,
                "unsupported incomplete game cannot be a ply cutoff",
            )?;
            (PalsResultV3::Incomplete, PalsTerminationV3::PlyLimit, None)
        }
        _ => {
            return Err(invalid(
                "unregistered game classification cannot be normalized",
            ));
        }
    };
    Ok(PalsGameReceiptV5 {
        game_index: index as u32,
        white_endpoint: g.white_engine.clone(),
        black_endpoint: g.black_engine.clone(),
        base_ms: pair.base_ms,
        increment_ms: pair.increment_ms,
        plies: g.uci_moves.len() as u32,
        result,
        termination,
        failed_endpoint: failed,
        pgn: pgn.clone(),
        rules_clock_pgn_check: PalsObservedV3::Observed {
            value: audit.clone(),
            method: "parent actual Rules replay and pinned-runner per-move Fischer clock audit"
                .into(),
        },
    })
}

/// Builds a V5 semantic receipt only from a real common-runner output. It
/// checks the actual stored native receipt bytes before reading its public
/// projection, preserves all four session observations, and writes new parent
/// evidence without overwriting raw native/PGN/failure artifacts.
#[cfg(target_os = "linux")]
pub fn build_pals_run_receipt_v5(
    output: &crate::NativePairOutput<LockedPalsArenaLaunchV5>,
) -> Result<(PalsNormalizedPairReceiptV5, ArtifactRef), ArenaError> {
    let owner = output.launch_owner();
    let lock = &owner.spec.input.semantic_lock;
    let native = &output.receipt;
    output.receipt_artifact.validate()?;
    require(
        output.receipt_artifact.path
            == format!(
                "{}/{}",
                owner.snapshot.output_directory,
                owner.spec.receipt_filename()
            ),
        "native receipt ArtifactRef does not name the actual owner file",
    )?;
    let original = crate::native_runner::linux::read_file(
        &owner.snapshot.directory,
        owner.spec.receipt_filename(),
        owner.spec.pair_metadata_cap(),
    )?;
    require(
        original.len() as u64 == output.receipt_artifact.bytes
            && format!("{:x}", Sha256::digest(&original)) == output.receipt_artifact.sha256,
        "actual stored native receipt byte identity differs",
    )?;
    let actual: serde_json::Value = crate::decode_json(
        std::str::from_utf8(&original).map_err(|_| invalid("native receipt not UTF8"))?,
    )?;
    require(
        actual
            == serde_json::to_value(native)
                .map_err(|_| invalid("native receipt projection encode failed"))?,
        "mutable native receipt projection differs from actual stored evidence",
    )?;
    require(
        native_receipt_belongs_to_owner(native, owner)
            && native.integration_checks_passed
            && native.cleanup_verified
            && !native.unresolved_owner_retained
            && native.provider_sessions.len() == 4
            && output.process().receipt.exit_code == Some(0)
            && output.process().pending_child.is_none(),
        "partial/fabricated/unclosed pair cannot become normalized acceptance",
    )?;
    let admission = owner
        .spec
        .admission
        .lock()
        .map_err(|_| invalid("parent admission poisoned"))?
        .clone()
        .ok_or_else(|| invalid("actual parent source/domain/numerical admission missing"))?;
    let pair = lock
        .manifest
        .pair
        .as_ref()
        .ok_or_else(|| invalid("pair lock missing"))?;
    let pgn = native
        .artifacts
        .iter()
        .find(|a| a.path.ends_with("/match.pgn"))
        .ok_or_else(|| invalid("actual pair PGN artifact missing"))?;
    let pgn_audit = native
        .pgn_audit
        .as_ref()
        .ok_or_else(|| invalid("actual Rules PGN audit missing"))?;
    let clock = native
        .clock_audit
        .as_ref()
        .ok_or_else(|| invalid("actual clock audit missing"))?;
    require(
        pgn_audit.games.len() == 2
            && clock.games.len() == 2
            && native.clock_audit_error.is_none()
            && native.pgn_audit_error.is_none(),
        "complete Rules/clock evidence missing",
    )?;
    let timing = owner
        .spec
        .timing
        .lock()
        .map_err(|_| invalid("original run timing poisoned"))?
        .clone()
        .ok_or_else(|| invalid("original finite run timing missing"))?;
    let end = timing
        .finished
        .ok_or_else(|| invalid("run has not completed"))?;
    let mut receipt = PalsRunReceiptV5::pending(lock)?;
    receipt.wall_time_ms = millis_ceil(end.duration_since(timing.started).as_nanos())?;
    receipt.cleanup_time_ms = millis_ceil(
        native
            .process
            .cleanup_elapsed_ns
            .ok_or_else(|| invalid("actual process cleanup duration unknown"))? as u128,
    )?;
    receipt.source_verification = PalsObservedV3::Observed {
        value: admission.source_verification.clone(),
        method: "parent retained FD content verification before any child".into(),
    };
    let mut produced = Vec::new();
    let mut scopes = Vec::new();
    for endpoint in &lock.manifest.engines {
        let sessions: Vec<_> = native
            .provider_sessions
            .iter()
            .filter(|s| s.endpoint_id == endpoint.id)
            .collect();
        require(
            sessions.len() == 2 && sessions[0].process_id != sessions[1].process_id,
            "each normalized endpoint requires two distinct actual sessions",
        )?;
        for s in &sessions {
            validate_pals_endpoint_observation_v5(&s.endpoint, endpoint, true, false, false)?;
        }
        let mut aggregate = sessions[0].endpoint.clone();
        aggregate.runtime_owner = format!("v5-aggregate-{}-{}", endpoint.id, owner.spec.sha256);
        let first = observed(&sessions[0].endpoint.coordinator)
            .ok_or_else(|| invalid("coordinator absent"))?;
        let second = observed(&sessions[1].endpoint.coordinator)
            .ok_or_else(|| invalid("coordinator absent"))?;
        let filename = format!("aggregate-{}-coordinator.v5.json", receipt.endpoints.len());
        let placeholder = ArtifactRef {
            path: format!("{}/{filename}", owner.snapshot.output_directory),
            sha256: "0".repeat(64),
            bytes: 1,
            source: "https://github.com/daejunnom/RoveZero".into(),
            license: "MIT".into(),
        };
        let mut counter = aggregate_coordinator(first, second, placeholder)?;
        let mut counter_value =
            serde_json::to_value(&counter).map_err(|_| invalid("counter encode failed"))?;
        counter_value.as_object_mut().unwrap().remove("evidence");
        let payload = serde_json::json!({"domain":"rz-pals-coordinator-pair-aggregate-v5/1",
            "endpoint_id":endpoint.id,"native_receipt":output.receipt_artifact,
            "process_ids":sessions.iter().map(|s|s.process_id).collect::<Vec<_>>(),
            "input_evidence":sessions.iter().map(|s|observed(&s.endpoint.coordinator).unwrap().evidence.clone()).collect::<Vec<_>>(),
            "observation_without_evidence":counter_value});
        let artifact = save_parent(
            owner,
            &filename,
            &serde_json::to_vec(&payload)
                .map_err(|_| invalid("aggregate evidence encode failed"))?,
        )?;
        counter.evidence = artifact.clone();
        aggregate.coordinator=PalsObservedV3::Observed{value:counter,
            method:"sum of distinct game-session counters; maximum of independently observed peaks; no native work invented".into()};
        // Both actual per-game peaks stay in raw sessions. A missing observation
        // in either process keeps the run-level peak unknown.
        for (out, a, b) in [
            (
                &mut aggregate.host_peak_bytes,
                &sessions[0].endpoint.host_peak_bytes,
                &sessions[1].endpoint.host_peak_bytes,
            ),
            (
                &mut aggregate.vram_peak_bytes,
                &sessions[0].endpoint.vram_peak_bytes,
                &sessions[1].endpoint.vram_peak_bytes,
            ),
        ] {
            *out=match(observed(a),observed(b)){(Some(a),Some(b))=>PalsObservedV3::Observed{value:(*a).max(*b),method:"maximum of two actual per-game observations; original scope retained in sessions".into()},_=>PalsObservedV3::Unknown};
        }
        scopes.push(PalsNormalizedEndpointScopeV5{endpoint_id:endpoint.id.clone(),aggregate_coordinator_evidence:artifact.clone(),
            process_ids:sessions.iter().map(|s|s.process_id).collect(),process_run_ids:sessions.iter().map(|s|s.process_run_id.clone()).collect(),
            representative_process_id:sessions[0].process_id,
            accounting_scope:"two-game sum for cumulative counters; per-session maxima for peaks; representative exact model/resource snapshot; every session independently admitted".into()});
        produced.push(artifact);
        receipt.endpoints.push(aggregate);
    }
    for (index, g) in pgn_audit.games.iter().enumerate() {
        require(
            g.game_id == clock.games[index].game_id,
            "Rules PGN and clock game identities differ",
        )?;
        receipt
            .games
            .push(game_receipt(g, index, pair, pgn, &output.receipt_artifact)?);
    }
    receipt.accepted = receipt
        .games
        .iter()
        .all(|g| g.result != PalsResultV3::Incomplete);
    // Functional pair acceptance is distinct from controlled performance or
    // formal strength authority. The original native integration scope is retained.
    receipt.performance_eligible = false;
    receipt.strength_eligible = false;
    if receipt.wall_time_ms > lock.manifest.window.wall_time_max_ms {
        receipt.failures.insert(PalsFailureV5::Expired);
        receipt.accepted = false;
    }
    if receipt.cleanup_time_ms > lock.manifest.window.cleanup_max_ms {
        receipt.failures.insert(PalsFailureV5::ProcessCleanup);
        receipt.accepted = false;
    }
    let mut paths = BTreeMap::new();
    for a in native
        .artifacts
        .iter()
        .chain(std::iter::once(&output.receipt_artifact))
        .chain(std::iter::once(&admission.source_verification.evidence))
        .chain(produced.iter())
    {
        if let Some(old) = paths.insert(a.path.clone(), a.clone()) {
            require(
                old.sha256 == a.sha256 && old.bytes == a.bytes,
                "output artifact alias differs",
            )?;
        }
    }
    let base_bytes = owner_output_bytes(owner)?;
    let mut wrapper=PalsNormalizedPairReceiptV5{domain:match owner.spec.storage_policy { PalsOwnedStoragePolicy::StrictV5 => "rz-pals-normalized-pair-receipt-v5/1", PalsOwnedStoragePolicy::AdvisoryV1 => "rz-pals-normalized-pair-receipt-storage-advisory-v1/1" }.into(),normalized:receipt,
        native_receipt:output.receipt_artifact.clone(),sessions:native.provider_sessions.clone(),endpoint_scopes:scopes,
        artifacts:paths.into_values().collect(),elapsed_scope:"original run entry includes input admission, metadata probes, runner, Rules/provider/clock verification; snapshot preparation is separately scoped".into(),
        raw_game_failures_preserved:true,training_executed:false};
    // Only a bounded pure JSON-length fixpoint; this never repeats model/game work.
    let mut bytes = Vec::new();
    let mut total = base_bytes;
    for _ in 0..4 {
        wrapper.normalized.output_bytes=PalsObservedV3::Observed{value:total,method:"bounded no-follow length scan of every owned output/runtime/archive file, plus this parent receipt; excludes only registered inputs".into()};
        if total > lock.manifest.window.output_bytes_max {
            wrapper.normalized.accepted = false;
            wrapper
                .normalized
                .failures
                .insert(PalsFailureV5::OutputQuota);
        }
        wrapper.normalized.validate_against(lock)?;
        bytes = serde_json::to_vec(&wrapper)
            .map_err(|_| invalid("normalized receipt encode failed"))?;
        let next = base_bytes
            .checked_add(bytes.len() as u64)
            .ok_or_else(|| invalid("normalized output byte overflow"))?;
        if next == total {
            break;
        }
        total = next;
    }
    require(
        observed(&wrapper.normalized.output_bytes)
            == Some(
                &base_bytes
                    .checked_add(bytes.len() as u64)
                    .ok_or_else(|| invalid("output overflow"))?,
            ),
        "normalized receipt byte fixpoint did not converge",
    )?;
    require(
        total <= lock.manifest.window.output_bytes_max,
        "normalized receipt cannot be persisted outside registered output quota",
    )?;
    let filename = match owner.spec.storage_policy {
        PalsOwnedStoragePolicy::StrictV5 => "pals-run-receipt.v5.json",
        PalsOwnedStoragePolicy::AdvisoryV1 => "pals-run-receipt.storage-advisory-v1.json",
    };
    let artifact = save_parent(owner, filename, &bytes)?;
    Ok((wrapper, artifact))
}
#[cfg(not(target_os = "linux"))]
pub fn build_pals_run_receipt_v5(
    _: &crate::NativePairOutput<LockedPalsArenaLaunchV5>,
) -> Result<(PalsNormalizedPairReceiptV5, ArtifactRef), ArenaError> {
    Err(invalid(
        "actual V5 normalization requires Linux-owned runner evidence",
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn peak_merge_preserves_unknown_and_distinguishes_sum() {
        assert_eq!(
            merge_peak(&3.into(), &7.into()).unwrap(),
            serde_json::json!(7)
        );
        assert_eq!(
            merge_peak(
                &serde_json::json!({"status":"unknown"}),
                &serde_json::json!({"status":"observed","value":7,"method":"fixture"})
            )
            .unwrap(),
            serde_json::json!({"status":"unknown"})
        );
        assert!(merge_peak(&true.into(), &7.into()).is_err());
        assert_eq!(millis_ceil(1).unwrap(), 1);
    }
    #[cfg(target_os = "linux")]
    #[test]
    fn output_scan_counts_unlisted_runtime_and_rejects_aliases() {
        use cap_std::fs::Dir;
        let root = std::env::temp_dir().join(format!(
            "rz-v5-output-observation-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir(&root).unwrap();
        std::fs::create_dir(root.join("inputs")).unwrap();
        std::fs::write(root.join("inputs/model.bin"), vec![1; 1024]).unwrap();
        std::fs::write(root.join("raw.log"), b"raw").unwrap();
        std::fs::create_dir(root.join("runtime")).unwrap();
        std::fs::write(root.join("runtime/unlisted.archive"), b"archive").unwrap();
        let dir = Dir::open_ambient_dir(&root, cap_std::ambient_authority()).unwrap();
        assert_eq!(owned_output_bytes(&dir).unwrap(), 10);
        std::os::unix::fs::symlink("runtime/unlisted.archive", root.join("bad-link")).unwrap();
        assert!(owned_output_bytes(&dir).is_err());
        std::fs::remove_file(root.join("bad-link")).unwrap();
        std::fs::hard_link(root.join("raw.log"), root.join("hard-alias")).unwrap();
        assert!(owned_output_bytes(&dir).is_err());
        std::fs::remove_file(root.join("hard-alias")).unwrap();
        assert_eq!(owned_output_bytes(&dir).unwrap(), 10);
        drop(dir);
        for path in ["inputs/model.bin", "raw.log", "runtime/unlisted.archive"] {
            std::fs::remove_file(root.join(path)).unwrap();
        }
        std::fs::remove_dir(root.join("inputs")).unwrap();
        std::fs::remove_dir(root.join("runtime")).unwrap();
        std::fs::remove_dir(root).unwrap();
    }
}
