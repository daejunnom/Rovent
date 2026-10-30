//! 원 외부 worker 관측의 역사 검증이다. JSON/PID/profile로 live source,
//! native node credit, physical 완료 또는 새 target 권한을 발급하지 않는다.
use super::*;
use rz_experiments::{ArtifactRef, RoventExternalPurpose, RoventInputLock, RoventResearcher};
use rz_search::external_cpu::{ExternalEngineProfile, ExternalEnginePurpose};
use rz_uci::pals_checker_profile::rovent::RoventExternalProfile;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoventHistoricalExternalAuthority {
    pub process_epoch: u64,
    pub game_generation: u64,
    pub root_generation: u64,
    pub implementation_sha256: String,
}
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoventHistoricalExternalProcess {
    pub pid: u32,
    pub process_group: u32,
    pub proc_start_ticks: u64,
}
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoventHistoricalObservedUci {
    pub name: String,
    pub author: Option<String>,
}

/// 원 file SHA와 profile canonical/capability SHA는 서로 다른 관측이다.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoventHistoricalExternalSource {
    pub schema: String,
    pub root_source_sequence: u64,
    pub worker_owner: u16,
    pub clock_epoch: u64,
    pub authority: RoventHistoricalExternalAuthority,
    pub engine_profile: ExternalEngineProfile,
    pub purpose: ExternalEnginePurpose,
    pub process: RoventHistoricalExternalProcess,
    pub profile: RoventExternalProfile,
    pub profile_file_sha256: String,
    pub profile_file_bytes: u64,
    pub profile_canonical_sha256: String,
    pub profile_capability_sha256: String,
    pub binary: ArtifactRef,
    pub startup_input_lock_sha256: String,
    pub observed_uci: RoventHistoricalObservedUci,
    pub options_application_observed: Option<bool>,
    pub model_loading_observed: Option<bool>,
}
impl RoventHistoricalExternalSource {
    pub(super) fn from_original_root(raw: &Value) -> Result<Self, ArenaError> {
        let source: Self = serde_json::from_value(raw["external_source"].clone())
            .map_err(|error| invalid(format!("explicit external source evidence: {error}")))?;
        source
            .profile
            .validate()
            .map_err(|error| invalid(error.to_string()))?;
        check(
            source.schema == rz_experiments::rovent_corpus_v2::ROVENT_PROFILED_EXTERNAL_SOURCE
                && source.purpose == ExternalEnginePurpose::ResearchChecker
                && source.profile.purpose == source.purpose
                && source.profile.engine == source.engine_profile
                && source
                    .profile
                    .canonical_sha256()
                    .map_err(|error| invalid(error.to_string()))?
                    == source.profile_canonical_sha256
                && source
                    .profile
                    .capability_sha256()
                    .map_err(|error| invalid(error.to_string()))?
                    == source.profile_capability_sha256
                && source.profile.identity.binary_sha256 == source.binary.sha256
                && source.observed_uci.name == source.profile.expected_uci_name
                && source.authority.process_epoch
                    == raw["process_epoch"].as_u64().unwrap_or(u64::MAX)
                && source.authority.game_generation
                    == raw["game_generation"].as_u64().unwrap_or(u64::MAX)
                && source.authority.root_generation
                    == raw["root_generation"].as_u64().unwrap_or(u64::MAX)
                && source.authority.implementation_sha256 == raw["implementation_sha256"]
                && source.clock_epoch == raw["deadline_clock"].as_u64().unwrap_or(u64::MAX)
                && source.options_application_observed.is_none()
                && source.model_loading_observed.is_none(),
            "historical external source official profile/root mismatch",
        )?;
        Ok(source)
    }

    fn verify_input_lock(&self, lock: &RoventInputLock) -> Result<(), ArenaError> {
        let Some(RoventResearcher::External { profile: pin }) = &lock.manifest.engine.researcher
        else {
            return Err(invalid(
                "external collection evidence requires original external researcher",
            ));
        };
        let option = |name: &str| {
            self.profile
                .identity
                .options
                .get(name)
                .and_then(|value| value.parse::<u32>().ok())
        };
        check(
            self.startup_input_lock_sha256 == lock.canonical_sha256
                && pin.domain == rz_experiments::ROVENT_EXTERNAL_PROFILE_DOMAIN
                && pin.purpose == RoventExternalPurpose::ResearchChecker
                && pin.profile.sha256 == self.profile_file_sha256
                && pin.profile.bytes == self.profile_file_bytes
                && pin.canonical_sha256 == self.profile_canonical_sha256
                && pin.capability_sha256 == self.profile_capability_sha256
                && pin.binary == self.binary
                && pin.source_commit == self.engine_profile.source_commit()
                && Some(pin.requested_threads) == option("Threads")
                && Some(pin.requested_hash_mib) == option("Hash"),
            "historical external profile/file/binary/resources differ from original whole lock",
        )
    }
}

/// 전체 lock을 보유한 역사 loader가 호출한다. 현재 프로세스 실행 권한이나
/// file-open 권한을 발급하지 않고 원 admission 관측과 strict lock만 대조한다.
pub fn verify_historical_external_source_input_lock(
    report: &RoventHistoricalRecordEvidence,
    lock: &RoventInputLock,
) -> Result<(), ArenaError> {
    if let Some(source) = &report.external_source {
        source.verify_input_lock(lock)?;
    }
    Ok(())
}
