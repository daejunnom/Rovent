//! Standalone selected-endpoint NoGame smoke plan and original clock.
//! No Fastchess, paired budget, opposite endpoint or paired canonical is used.

use super::{super::PalsSourceFileV5, PalsSystemEndpointLaunchV6};
use crate::ArenaError;
use rz_experiments::{
    ArtifactRef, EngineEnvironmentV2, NativeEngineRole, PalsSystemSourcePolicyV6,
    PalsSystemStartupLockV6, pals_private_snapshot_path_v5, source_component_digest_v5,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};
use std::{
    collections::BTreeMap,
    time::{Duration, Instant},
};

pub(crate) const NO_GAME_SMOKE_PLAN_DOMAIN: &str = "rz-pals-no-game-smoke-plan/1";

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct NoGameSmokeBudgetV1 {
    pub max_input_bytes: u64,
    pub max_runtime_bytes: u64,
    pub max_output_bytes: u64,
    pub max_artifact_bytes: u64,
    /// Includes the direct proxy and its native child; no game runner.
    pub max_child_processes: u32,
    pub max_runtime_files: u32,
    pub max_runtime_depth: u16,
    pub address_space_per_process_bytes: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct NoGameSmokeEndpointV1 {
    pub startup_lock: PalsSystemStartupLockV6,
    pub startup_lock_artifact: ArtifactRef,
    pub startup_snapshot_path: String,
    pub binary_snapshot_path: String,
    pub environment: Option<EngineEnvironmentV2>,
    pub source_files: BTreeMap<String, Vec<PalsSourceFileV5>>,
}

impl From<&PalsSystemEndpointLaunchV6> for NoGameSmokeEndpointV1 {
    fn from(endpoint: &PalsSystemEndpointLaunchV6) -> Self {
        Self {
            startup_lock: endpoint.startup_lock.clone(),
            startup_lock_artifact: endpoint.startup_lock_artifact.clone(),
            startup_snapshot_path: endpoint.startup_snapshot_path.clone(),
            binary_snapshot_path: endpoint.binary_snapshot_path.clone(),
            environment: endpoint.environment.clone(),
            source_files: endpoint.source_files.clone(),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct NoGameSmokePlanV1 {
    pub domain: String,
    pub declared_role: NativeEngineRole,
    pub endpoint: NoGameSmokeEndpointV1,
    pub proxy_binary: ArtifactRef,
    pub proxy_snapshot_path: String,
    pub budget: NoGameSmokeBudgetV1,
    /// Optional provenance only; no paired declaration is required or inferred.
    pub paired_source_sha256: Option<String>,
}

#[derive(Clone, Debug)]
pub(crate) struct LockedNoGameSmokePlanV1 {
    plan: NoGameSmokePlanV1,
    sha256: String,
}

fn invalid(message: &'static str) -> ArenaError {
    ArenaError::Invalid(message.into())
}

fn decode_sha(value: &str) -> Result<[u8; 32], ArenaError> {
    if value.len() != 64 {
        return Err(invalid("NoGame source digest invalid"));
    }
    let mut result = [0; 32];
    for (i, pair) in value.as_bytes().chunks_exact(2).enumerate() {
        let nibble = |byte: u8| -> Option<u8> {
            match byte {
                b'0'..=b'9' => Some(byte - b'0'),
                b'a'..=b'f' => Some(byte - b'a' + 10),
                _ => None,
            }
        };
        result[i] = (nibble(pair[0]).ok_or_else(|| invalid("NoGame source digest invalid"))? << 4)
            | nibble(pair[1]).ok_or_else(|| invalid("NoGame source digest invalid"))?;
    }
    Ok(result)
}

impl NoGameSmokePlanV1 {
    pub(crate) fn lock(self) -> Result<LockedNoGameSmokePlanV1, ArenaError> {
        if self.domain != NO_GAME_SMOKE_PLAN_DOMAIN
            || !pals_private_snapshot_path_v5(&self.endpoint.startup_snapshot_path)
            || !pals_private_snapshot_path_v5(&self.endpoint.binary_snapshot_path)
            || !pals_private_snapshot_path_v5(&self.proxy_snapshot_path)
        {
            return Err(invalid("NoGame plan domain or snapshot path invalid"));
        }
        self.endpoint.startup_lock.verify()?;
        self.endpoint.startup_lock_artifact.validate()?;
        self.proxy_binary.validate()?;
        if let Some(source) = &self.paired_source_sha256 {
            decode_sha(source)?;
        }
        let lock_bytes = self.endpoint.startup_lock.canonical_bytes()?;
        if self.endpoint.startup_lock_artifact.bytes != lock_bytes.len() as u64
            || self.endpoint.startup_lock_artifact.sha256
                != format!("{:x}", Sha256::digest(&lock_bytes))
        {
            return Err(invalid("NoGame startup artifact does not bind its lock"));
        }
        if let Some(environment) = &self.endpoint.environment {
            environment.validate_native()?;
        }
        let budget = &self.budget;
        if budget.max_input_bytes == 0
            || budget.max_runtime_bytes == 0
            || budget.max_output_bytes == 0
            || budget.max_artifact_bytes == 0
            || budget.max_child_processes < 2
            || budget.max_child_processes > i32::MAX as u32
            || budget.max_runtime_files == 0
            || budget.max_runtime_depth == 0
            || budget.address_space_per_process_bytes == 0
        {
            return Err(invalid("NoGame finite resource budget invalid"));
        }
        let config = &self.endpoint.startup_lock.config;
        let expected_runtime = match self.declared_role {
            NativeEngineRole::Baseline => "baseline-runtime",
            NativeEngineRole::Candidate => "candidate-runtime",
        };
        if config.runtime_directory != expected_runtime {
            return Err(invalid("NoGame selected runtime directory differs"));
        }
        if config.source_policy == PalsSystemSourcePolicyV6::Strict {
            if self.endpoint.source_files.len() != config.source_components.len()
                || self
                    .endpoint
                    .source_files
                    .keys()
                    .any(|key| !config.source_components.contains_key(key))
            {
                return Err(invalid("NoGame strict source rows differ"));
            }
            for (component, files) in &self.endpoint.source_files {
                if files.is_empty() || files.len() > 128 {
                    return Err(invalid("NoGame strict source row count invalid"));
                }
                let tuples = files
                    .iter()
                    .map(|file| {
                        Ok((
                            file.logical_source_path.clone(),
                            file.artifact.bytes,
                            decode_sha(&file.artifact.sha256)?,
                        ))
                    })
                    .collect::<Result<Vec<_>, ArenaError>>()?;
                let actual = source_component_digest_v5(&tuples)?;
                let expected = decode_sha(
                    config
                        .source_components
                        .get(component)
                        .ok_or_else(|| invalid("NoGame strict source component missing"))?,
                )?;
                if actual != expected {
                    return Err(invalid("NoGame strict source digest differs"));
                }
            }
        }
        let sha256 = crate::canonical_sha256(&(NO_GAME_SMOKE_PLAN_DOMAIN, &self))?;
        let locked = LockedNoGameSmokePlanV1 { plan: self, sha256 };
        super::smoke_preparation::validate_selection(&locked)?;
        Ok(locked)
    }
}

impl LockedNoGameSmokePlanV1 {
    pub(crate) fn input(&self) -> &NoGameSmokePlanV1 {
        &self.plan
    }
    pub(crate) fn sha256(&self) -> &str {
        &self.sha256
    }
}

/// Created before preparation and owned unchanged through the direct smoke.
/// Its start and deadlines are process-local and never reconstructed from IPC.
pub(crate) struct NoGameExecutionStartV1 {
    plan_sha256: String,
    owner_id: [u8; 16],
    owner_epoch: u64,
    started: Instant,
    run_until: Instant,
    cleanup_until: Instant,
}

impl NoGameExecutionStartV1 {
    pub(crate) fn begin(
        plan: &LockedNoGameSmokePlanV1,
        owner_id: [u8; 16],
        owner_epoch: u64,
    ) -> Result<Self, ArenaError> {
        if owner_id == [0; 16] || owner_epoch == 0 {
            return Err(invalid("NoGame original execution identity invalid"));
        }
        let started = Instant::now();
        let window = &plan.input().endpoint.startup_lock.config.window;
        let run_until = started
            .checked_add(Duration::from_millis(window.wall_time_max_ms))
            .ok_or_else(|| invalid("NoGame run deadline overflow"))?;
        let cleanup_until = run_until
            .checked_add(Duration::from_millis(window.cleanup_max_ms))
            .ok_or_else(|| invalid("NoGame cleanup deadline overflow"))?;
        if run_until <= started || cleanup_until <= run_until {
            return Err(invalid("NoGame execution window invalid"));
        }
        Ok(Self {
            plan_sha256: plan.sha256().into(),
            owner_id,
            owner_epoch,
            started,
            run_until,
            cleanup_until,
        })
    }
    pub(crate) fn matches(
        &self,
        plan: &LockedNoGameSmokePlanV1,
        owner_id: [u8; 16],
        owner_epoch: u64,
    ) -> bool {
        self.plan_sha256 == plan.sha256()
            && self.owner_id == owner_id
            && self.owner_epoch == owner_epoch
    }
    pub(crate) fn matches_plan(&self, plan: &LockedNoGameSmokePlanV1) -> bool {
        self.plan_sha256 == plan.sha256()
    }
    pub(crate) fn require_running(&self) -> Result<(), ArenaError> {
        if Instant::now() >= self.run_until {
            Err(invalid("NoGame original execution expired"))
        } else {
            Ok(())
        }
    }
    pub(crate) fn started(&self) -> Instant {
        self.started
    }
    pub(crate) fn run_until(&self) -> Instant {
        self.run_until
    }
    pub(crate) fn cleanup_until(&self) -> Instant {
        self.cleanup_until
    }
}
