//! V6 metadata-only preflight observations under the original role owner.
//!
//! A preflight record is written before the child serves UCI. Only the common
//! supervisor's own process and protocol observations can prove readiness or
//! the rules-only stop/bestmove exchange. These records never prove inference.

use super::LockedPalsSystemArenaPairV6;
use crate::{ArenaError, ExternalUciPreflight};
use cap_fs_ext::DirExt;
use cap_std::fs::Dir;
use rz_experiments::{NativeEngineRole, PalsSystemSourcePolicyV6, PalsSystemStartupLockV6};
use serde::Deserialize;
use std::collections::BTreeSet;

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
enum SourceObservation {
    Verified,
    Unknown,
    Mismatch,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct PreflightWireV6 {
    domain: String,
    lock_sha256: String,
    system_runtime_lock_sha256: String,
    endpoint_id: String,
    process_id: u32,
    process_generation: u64,
    mapped_executable_sha256: String,
    mapped_executable_bytes: u64,
    source_commit_observation: SourceObservation,
    source_components_observation: SourceObservation,
    source_components_scope: String,
    system_v6_source_observation: SourceObservation,
    system_v6_source_scope: String,
    assets_held: usize,
    sessions_created: u64,
    provider_initialized: bool,
    neural_execution: bool,
    physical_completion: String,
}

impl LockedPalsSystemArenaPairV6 {
    /// Preserve metadata from a single original supervisor probe. A matching
    /// child report and OS process exit do not independently prove that the
    /// child never used a GPU. Callers must retain that observation separately.
    pub(crate) fn observe_system_probe_v6(
        &self,
        role: NativeEngineRole,
        process: &crate::ProcessOutput,
        role_directory: &Dir,
        inputs: &crate::NativeProviderInputContext<'_>,
    ) -> Result<(String, Vec<u8>), ArenaError> {
        use crate::{CleanupStatus, ProcessStop};

        let endpoint = self.endpoint(role);
        let pid = process.receipt.pid;
        if pid == 0
            || process.receipt.stop != ProcessStop::Exited
            || process.receipt.exit_code != Some(0)
            || process.receipt.group_cleanup != CleanupStatus::Gone
            || !process.receipt.errors.is_empty()
            || process.pending_child.is_some()
        {
            return Err(invalid(
                "system preflight supervisor process did not close normally",
            ));
        }
        // These are the original common owner's verified snapshot FDs. The
        // child report's executable observation is still a child claim.
        for expected in [
            &endpoint.startup_lock_artifact,
            &endpoint.startup_lock.config.endpoint.binary,
        ] {
            let matches: Vec<_> = inputs
                .pins
                .iter()
                .filter(|pin| pin.artifact == expected)
                .collect();
            if matches.len() != 1 {
                return Err(invalid(
                    "system preflight lacks its original lock or binary pin",
                ));
            }
            let metadata = matches[0]
                .file
                .metadata()
                .map_err(|_| invalid("system preflight input pin unavailable"))?;
            if !metadata.is_file() || metadata.len() != expected.bytes {
                return Err(invalid("system preflight input pin changed"));
            }
        }
        let name = format!("preflight-process-{pid}");
        let directory = role_directory
            .open_dir_nofollow(&name)
            .map_err(|_| invalid("owned system preflight directory unavailable"))?;
        let bytes = crate::native_runner::linux::read_file(
            &directory,
            "pals-system-preflight.v6.json",
            64 << 10,
        )?;
        validate_preflight_record(&endpoint.startup_lock, pid, &bytes)?;
        Ok((format!("{name}/pals-system-preflight.v6.json"), bytes))
    }

    /// Bind both endpoints to the original supervisor's four distinct probe
    /// PIDs. This verifies their metadata reports, not model or GPU execution.
    pub fn validate_system_pair_preflights_v6(
        &self,
        baseline: (&ExternalUciPreflight, &Dir),
        candidate: (&ExternalUciPreflight, &Dir),
    ) -> Result<Vec<(NativeEngineRole, Vec<(String, Vec<u8>)>)>, ArenaError> {
        let pids = [
            baseline.0.identification_process.pid,
            baseline.0.readiness_process.pid,
            candidate.0.identification_process.pid,
            candidate.0.readiness_process.pid,
        ];
        validate_preflight_pids(pids)?;
        Ok(vec![
            (
                NativeEngineRole::Baseline,
                self.validate_system_preflight_v6(
                    NativeEngineRole::Baseline,
                    baseline.0,
                    baseline.1,
                )?,
            ),
            (
                NativeEngineRole::Candidate,
                self.validate_system_preflight_v6(
                    NativeEngineRole::Candidate,
                    candidate.0,
                    candidate.1,
                )?,
            ),
        ])
    }

    /// Check raw metadata from the two exact processes seen by the existing
    /// preflight supervisor and return their original bytes for retention.
    /// Source observations and zero-session claims remain child reports;
    /// the supervisor separately validates exit, readiness, and legal bestmove.
    pub fn validate_system_preflight_v6(
        &self,
        role: NativeEngineRole,
        preflight: &ExternalUciPreflight,
        role_directory: &Dir,
    ) -> Result<Vec<(String, Vec<u8>)>, ArenaError> {
        let endpoint = self.endpoint(role);
        let config = &endpoint.startup_lock.config;
        let pids = [
            preflight.identification_process.pid,
            preflight.readiness_process.pid,
        ];
        if preflight.engine_id != config.endpoint.id
            || !preflight.stop_and_legal_bestmove_observed
            || pids[0] == 0
            || pids[1] == 0
            || pids[0] == pids[1]
        {
            return Err(invalid(
                "preflight endpoint, stop witness, or process identities differ",
            ));
        }
        let mut evidence = Vec::with_capacity(2);
        for pid in pids {
            let name = format!("preflight-process-{pid}");
            let dir = role_directory
                .open_dir_nofollow(&name)
                .map_err(|_| invalid("owned system preflight directory unavailable"))?;
            let bytes = crate::native_runner::linux::read_file(
                &dir,
                "pals-system-preflight.v6.json",
                64 << 10,
            )?;
            validate_preflight_record(&endpoint.startup_lock, pid, &bytes)?;
            evidence.push((format!("{name}/pals-system-preflight.v6.json"), bytes));
        }
        Ok(evidence)
    }
}

pub(super) fn validate_preflight_pids(pids: [u32; 4]) -> Result<(), ArenaError> {
    if pids.contains(&0) || pids.into_iter().collect::<BTreeSet<_>>().len() != pids.len() {
        return Err(invalid(
            "PALS system preflight process identity is reused across roles",
        ));
    }
    Ok(())
}

/// Validate child-authored bytes without interpreting source observations as
/// independent provenance. The caller retains the unchanged raw bytes.
pub(super) fn validate_preflight_record(
    lock: &PalsSystemStartupLockV6,
    pid: u32,
    bytes: &[u8],
) -> Result<(), ArenaError> {
    lock.verify()?;
    if pid == 0 || bytes.len() > 64 << 10 {
        return Err(invalid("invalid PALS system preflight identity or size"));
    }
    let text =
        std::str::from_utf8(bytes).map_err(|_| invalid("system preflight record is not UTF-8"))?;
    let record: PreflightWireV6 = crate::decode_json(text)?;
    let config = &lock.config;
    let source_matches = config.source_policy != PalsSystemSourcePolicyV6::Strict
        || (record.source_commit_observation == SourceObservation::Verified
            && record.source_components_observation == SourceObservation::Verified
            && record.system_v6_source_observation == SourceObservation::Verified);
    if record.domain != "rz-pals-system-preflight-v6/1"
        || record.lock_sha256 != lock.canonical_sha256
        || record.system_runtime_lock_sha256 != config.system_runtime_lock.canonical_sha256
        || record.endpoint_id != config.endpoint.id
        || record.process_id != pid
        || record.process_generation != config.process_generation
        || record.mapped_executable_sha256 != config.endpoint.binary.sha256
        || record.mapped_executable_bytes != config.endpoint.binary.bytes
        || record.source_components_scope != "declared-v5-registry-subset"
        || record.system_v6_source_scope != "rz-pals-system-v6-boundary-source/1"
        || record.assets_held != config.assets.len()
        || record.sessions_created != 0
        || record.provider_initialized
        || record.neural_execution
        || record.physical_completion != "not_applicable_no_gpu_execution"
        || (config.endpoint.source_commit.is_none()
            && record.source_commit_observation != SourceObservation::Unknown)
        || (!config
            .source_components
            .keys()
            .any(|key| key != "pals_system_v6" && key != "pals_system_v6_execution")
            && record.source_components_observation != SourceObservation::Unknown)
        || (!config.source_components.contains_key("pals_system_v6")
            && record.system_v6_source_observation != SourceObservation::Unknown)
        || !source_matches
    {
        return Err(invalid(
            "system preflight record differs from the owned child or lock",
        ));
    }
    Ok(())
}

fn invalid(reason: &str) -> ArenaError {
    ArenaError::Integrity(reason.into())
}
