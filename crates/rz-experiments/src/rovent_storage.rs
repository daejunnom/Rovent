//! Explicit, opt-in storage registration for the native Rovent PC/PCR owners.
//!
//! This independent artifact does not extend Rovent V1 or startup V1-V3. Its
//! absence means archive-off at the launcher; there is no default/on fallback.
//! Validation is declarative only: it opens no root, starts no helper, grants no
//! Store/physical authority and does not attest source bytes or execution.

use crate::{
    ArtifactRef, ManifestError, RoventActor, RoventExecutionManifest, RoventResearcher,
    RoventResourcePlan, decode_json, digest,
};
use rz_contracts::rovent::ExecutionMode;
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

pub const ROVENT_STORAGE_PROFILE_DOMAIN: &str = "rz-rovent-storage-profile/1";
pub const ROVENT_STORAGE_CANONICAL_DOMAIN: &str = "rz-rovent-storage-profile-canonical/1";
pub const ROVENT_STORAGE_MEASUREMENT_CONTRACT: &str = "rz-pals-cold-archive-owner-accounting-v4/1";

// Same value boundaries as Search store/archive.rs; these are format maxima,
// not chosen/default selectors. Keep this metadata crate independent of Search.
const GAME_BYTES_MAX: u64 = 256 * 1024 * 1024;
const GLOBAL_BYTES_MAX: u64 = 4 * 1024 * 1024 * 1024;
const RECORD_BYTES_MAX: u64 = 16 * 1024 * 1024;
const INDEX_ENTRIES_MAX: u32 = 1_048_576;
const LOAD_PINS_MAX: u32 = 16;
const MAINTENANCE_IO_BYTES_MAX: u64 = 16 * 1024 * 1024;

/// Only this explicit selection belongs to the first storage artifact domain.
/// Archive-off is absence of the artifact selection, not a default enum case.
#[derive(Clone, Copy, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum RoventStoragePolicy {
    #[serde(rename = "archive_on_v1")]
    ArchiveOnV1,
}

/// All eight Search runtime selectors must be supplied by the caller. No
/// serde/default construction fills missing limits or converts a legacy policy.
#[derive(Clone, Copy, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RoventArchiveRuntimeLimits {
    pub game_bytes_max: u64,
    pub global_bytes_max: u64,
    pub index_entries_max: u32,
    pub index_bytes_max: u64,
    pub load_bytes_max: u64,
    pub load_deadline_max_ms: u64,
    pub record_payload_bytes_max: u64,
    pub max_load_pins: u32,
}

impl RoventArchiveRuntimeLimits {
    /// Matches the value-only part of Search's one-shot selector admission.
    /// The actual Store's unused/owner/ledger checks remain Search's job.
    pub fn validate(&self) -> Result<(), ManifestError> {
        require(
            self.game_bytes_max > 0
                && self.game_bytes_max <= GAME_BYTES_MAX
                && self.global_bytes_max > 0
                && self.global_bytes_max <= GLOBAL_BYTES_MAX
                && self.game_bytes_max <= self.global_bytes_max
                && self.record_payload_bytes_max > 0
                && self.record_payload_bytes_max <= RECORD_BYTES_MAX
                && self.record_payload_bytes_max <= self.game_bytes_max
                && self.max_load_pins > 0
                && self.max_load_pins <= LOAD_PINS_MAX
                && self.index_entries_max > 0
                && self.index_entries_max <= INDEX_ENTRIES_MAX
                && self.index_bytes_max > 0
                && self.index_bytes_max <= RECORD_BYTES_MAX
                && self.load_bytes_max > 0
                && self.load_bytes_max <= self.game_bytes_max
                && self.load_bytes_max <= RECORD_BYTES_MAX
                && self.load_deadline_max_ms > 0
                && Instant::now()
                    .checked_add(Duration::from_millis(self.load_deadline_max_ms))
                    .is_some(),
            "archive runtime selectors exceed Search bounds or disagree",
        )
    }
}

/// Physical helper subroles sharing the single declared RuntimeHelper pool.
/// Counts describe actual concurrent threads, not idle/go CPU credit.
#[derive(Clone, Copy, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RoventStorageHelpers {
    pub shared_nnuec: u32,
    pub archive: u32,
}

impl RoventStorageHelpers {
    pub fn validate(&self) -> Result<(), ManifestError> {
        require(
            self.shared_nnuec <= 1 && self.archive == 1,
            "archive needs one worker and zero or one shared NNUEC helper",
        )
    }

    pub fn required_threads(&self) -> Result<u32, ManifestError> {
        self.validate()?;
        // The preceding closed counts make this addition at most two.
        Ok(self.shared_nnuec + self.archive)
    }
}

/// Exact declared source artifacts. A launcher must separately verify their
/// original bytes and compare them with the compiled storage/codec/launch pins.
/// No input-lock digest occurs here: this artifact may itself be provenance of
/// that lock, so embedding the final lock's digest would create a hash cycle.
#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RoventStorageSourcePins {
    pub source: ArtifactRef,
    pub codec: ArtifactRef,
    pub launch: ArtifactRef,
}

impl RoventStorageSourcePins {
    pub fn validate(&self) -> Result<(), ManifestError> {
        self.source.validate()?;
        self.codec.validate()?;
        self.launch.validate()
    }
}

/// An immutable registration, not an archive directory or mutable segment pin.
/// Actual CLI/raw-pin/provenance admission and held directory custody belong to
/// the native launcher, which must retain the original owner on every failure.
#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RoventStorageProfile {
    pub schema: String,
    pub policy: RoventStoragePolicy,
    pub archive_limits: RoventArchiveRuntimeLimits,
    pub maintenance_io_bytes_max: u64,
    pub worker_threads: u32,
    pub command_capacity: u32,
    pub completion_capacity: u32,
    pub helpers: RoventStorageHelpers,
    pub source_pins: RoventStorageSourcePins,
    pub measurement_contract: String,
}

impl RoventStorageProfile {
    /// Uses the existing bounded, recursively duplicate-key rejecting decoder.
    pub fn from_json(input: &str) -> Result<Self, ManifestError> {
        let profile: Self = decode_json(input)?;
        profile.validate()?;
        Ok(profile)
    }

    pub fn validate(&self) -> Result<(), ManifestError> {
        require(
            self.schema == ROVENT_STORAGE_PROFILE_DOMAIN
                && self.policy == RoventStoragePolicy::ArchiveOnV1
                && self.maintenance_io_bytes_max > 0
                && self.maintenance_io_bytes_max <= MAINTENANCE_IO_BYTES_MAX
                && self.worker_threads == 1
                && self.command_capacity == 1
                && self.completion_capacity == 1
                && self.measurement_contract == ROVENT_STORAGE_MEASUREMENT_CONTRACT,
            "storage domain/policy/bounded worker/measurement declaration mismatch",
        )?;
        self.archive_limits.validate()?;
        self.helpers.validate()?;
        self.source_pins.validate()
    }

    /// Check this declaration and the complete original V1 manifest, then their
    /// declared session/resource relationship. This enables no IO or authority.
    pub fn validate_for_manifest(
        &self,
        manifest: &RoventExecutionManifest,
    ) -> Result<(), ManifestError> {
        self.validate()?;
        manifest.validate()?;
        let engine = &manifest.engine;
        let expected_shared_nnuec = match engine.mode {
            ExecutionMode::CpuPc | ExecutionMode::HybridPc => {
                require(
                    engine.researcher.is_none(),
                    "PC storage requires a researcher-free declaration",
                )?;
                u32::from(engine.nnuec.is_some())
            }
            ExecutionMode::CpuPcr | ExecutionMode::HybridPcr => {
                require(
                    matches!(&engine.researcher, Some(RoventResearcher::Own { .. })),
                    "storage requires an explicit native own researcher; external R is unsupported",
                )?;
                0
            }
        };
        require(
            self.helpers.shared_nnuec == expected_shared_nnuec,
            "shared NNUEC helper subrole differs from the selected native session",
        )?;
        self.validate_resource_helpers(&manifest.resources)
    }

    /// Storage-specific pool check only. Use validate_for_manifest for complete
    /// quota/device/role validation; this cannot attest OS resource application.
    pub fn validate_resource_helpers(
        &self,
        resources: &RoventResourcePlan,
    ) -> Result<(), ManifestError> {
        self.validate()?;
        let required_threads = self.helpers.required_threads()?;
        let mut matches = resources
            .pools
            .iter()
            .filter(|pool| pool.actors.contains(&RoventActor::RuntimeHelper));
        let helper = matches.next().ok_or_else(|| {
            ManifestError::Integrity("Rovent storage: RuntimeHelper pool is missing".into())
        })?;
        require(
            matches.next().is_none()
                && helper.actors.len() == 1
                && helper.max_threads >= required_threads
                && helper.max_threads <= 1024
                && helper.cpu_charge_millicpus > 0,
            "archive and shared NNUEC helpers need one separate fully charged physical pool",
        )
    }

    pub fn canonical_bytes(&self) -> Result<Vec<u8>, ManifestError> {
        self.validate()?;
        // Same sorted-object encoding convention as Rovent V1, in this separate
        // artifact's own domain. Existing V1 bytes and digests are untouched.
        let value = serde_json::to_value((ROVENT_STORAGE_CANONICAL_DOMAIN, self))
            .map_err(|error| ManifestError::Integrity(error.to_string()))?;
        let bytes = serde_json::to_vec(&value)
            .map_err(|error| ManifestError::Integrity(error.to_string()))?;
        require(
            bytes.len() <= crate::MAX_MANIFEST_BYTES,
            "canonical storage profile exceeds the shared metadata bound",
        )?;
        Ok(bytes)
    }

    pub fn canonical_sha256(&self) -> Result<String, ManifestError> {
        Ok(digest(&self.canonical_bytes()?))
    }
}

fn require(ok: bool, reason: &str) -> Result<(), ManifestError> {
    if ok {
        Ok(())
    } else {
        Err(ManifestError::Integrity(format!(
            "Rovent storage: {reason}"
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ROVENT_EXECUTION_DOMAIN, ROVENT_EXTERNAL_PROFILE_DOMAIN, ROVENT_POSITION_VISIT_UNIT,
        RoventBackend, RoventCoordinatorDeclaration, RoventDevice, RoventEngineDeclaration,
        RoventExternalProfilePin, RoventExternalPurpose, RoventModelPin, RoventNnuecPin,
        RoventPcConfiguration, RoventPurpose, RoventSearchDeclaration, RoventThreadPool,
    };
    use rz_contracts::rovent::ROVENT_SEARCH_REVISION;
    use std::collections::{BTreeMap, BTreeSet};
    use std::num::NonZeroU16;

    fn artifact(name: &str) -> ArtifactRef {
        ArtifactRef {
            path: format!("fixture/{name}"),
            sha256: "a".repeat(64),
            bytes: 32,
            source: "https://example.org/storage-contract-fixture".into(),
            license: "MIT".into(),
        }
    }

    fn profile() -> RoventStorageProfile {
        RoventStorageProfile {
            schema: ROVENT_STORAGE_PROFILE_DOMAIN.into(),
            policy: RoventStoragePolicy::ArchiveOnV1,
            archive_limits: RoventArchiveRuntimeLimits {
                game_bytes_max: 256 << 20,
                global_bytes_max: 4 << 30,
                index_entries_max: 128,
                index_bytes_max: 4096,
                load_bytes_max: 1 << 20,
                load_deadline_max_ms: 500,
                record_payload_bytes_max: 1 << 20,
                max_load_pins: 2,
            },
            maintenance_io_bytes_max: 16 << 20,
            worker_threads: 1,
            command_capacity: 1,
            completion_capacity: 1,
            helpers: RoventStorageHelpers {
                shared_nnuec: 0,
                archive: 1,
            },
            source_pins: RoventStorageSourcePins {
                source: artifact("storage-source.json"),
                codec: artifact("archive-codec.rs"),
                launch: artifact("storage-launch.rs"),
            },
            measurement_contract: ROVENT_STORAGE_MEASUREMENT_CONTRACT.into(),
        }
    }

    fn model() -> RoventModelPin {
        RoventModelPin {
            semantics: "storage-contract-model/1".into(),
            encoding: "storage-contract-input/1".into(),
            policy_head: "legal-candidates/1".into(),
            value_head: "side-to-move-wdl/1".into(),
            implementation_sha256: "b".repeat(64),
            weights: artifact("weights.bin"),
            provenance: artifact("weights-provenance.json"),
            diagnostic: false,
        }
    }

    fn nnuec() -> RoventNnuecPin {
        RoventNnuecPin {
            schema: "nnuec-fixture/2".into(),
            semantics: "nnuec-fixture-semantics/2".into(),
            encoding: "nnuec-fixture-encoding/2".into(),
            implementation_sha256: "1".repeat(64),
            configuration_sha256: "2".repeat(64),
            weights: artifact("nnuec.bin"),
            provenance: artifact("nnuec-provenance.json"),
            diagnostic: false,
            context_namespace_policy_sha256: "3".repeat(64),
            local_namespace_policy_sha256: "4".repeat(64),
            search_conditions_sha256: "5".repeat(64),
        }
    }

    fn manifest(mode: ExecutionMode) -> RoventExecutionManifest {
        let hybrid = matches!(mode, ExecutionMode::HybridPc | ExecutionMode::HybridPcr);
        let researcher = mode.has_native_r().then(|| RoventResearcher::Own {
            implementation_sha256: "8".repeat(64),
            search_configuration_sha256: "9".repeat(64),
        });
        let mut pools = vec![
            RoventThreadPool {
                actors: BTreeSet::from([RoventActor::Manager, RoventActor::PcCaller]),
                max_threads: 1,
                cpu_charge_millicpus: 500,
            },
            RoventThreadPool {
                actors: BTreeSet::from([RoventActor::RuntimeHelper]),
                max_threads: 1,
                cpu_charge_millicpus: 500,
            },
        ];
        if mode.has_native_r() {
            pools.push(RoventThreadPool {
                actors: BTreeSet::from([RoventActor::OwnR]),
                max_threads: 1,
                cpu_charge_millicpus: 500,
            });
        }
        RoventExecutionManifest {
            domain: ROVENT_EXECUTION_DOMAIN.into(),
            contract_revision: ROVENT_SEARCH_REVISION.into(),
            run_id: "storage-contract-fixture".into(),
            purpose: RoventPurpose::Validation,
            engine: RoventEngineDeclaration {
                source_commit: "a".repeat(40),
                source_components: BTreeMap::from([("storage".into(), "c".repeat(64))]),
                binary: artifact("rovent"),
                mode,
                pc: RoventPcConfiguration {
                    proposer: model(),
                    critic: model(),
                    validator: None,
                    backend: if hybrid {
                        RoventBackend::OrtCuda
                    } else {
                        RoventBackend::RustCpu
                    },
                    device: if hybrid {
                        RoventDevice::Cuda {
                            ordinal: 0,
                            registered_device: "declared-fixture-device".into(),
                        }
                    } else {
                        RoventDevice::Cpu
                    },
                    backend_configuration_sha256: "d".repeat(64),
                    runtime_artifacts: if hybrid {
                        vec![artifact("runtime.json")]
                    } else {
                        vec![]
                    },
                    fp32: true,
                    tf32: false,
                    batch_width: 1,
                    ort_intra_threads: hybrid.then_some(1),
                    ort_inter_threads: hybrid.then_some(1),
                    ort_spinning: hybrid.then_some(false),
                },
                researcher,
                teachers: vec![],
                nnuec: mode.has_native_r().then(nnuec),
                search: RoventSearchDeclaration {
                    node_unit: ROVENT_POSITION_VISIT_UNIT.into(),
                    requested_nodes: None,
                    depth_plies: None,
                    mate_moves: None,
                    registered_max_chess_plies: NonZeroU16::new(64).unwrap(),
                    searchmoves: None,
                    search_configuration_sha256: "e".repeat(64),
                    root_state_and_history_sha256: "f".repeat(64),
                    rules_profile_sha256: "1".repeat(64),
                },
            },
            resources: RoventResourcePlan {
                cpu_quota_millicpus: 2000,
                pools,
                memory_high_bytes: 3 << 30,
                memory_max_bytes: 4 << 30,
                gpu_devices: u32::from(hybrid),
                physical_nn_concurrency: 1,
                topology_policy: "shared-control/1".into(),
            },
            coordinator: RoventCoordinatorDeclaration {
                policy_sha256: "2".repeat(64),
                questions_max: 64,
                ancestor_depth_max: 64,
                waiting_jobs_max: 64,
                role_tickets_max: 16,
                cpu_credit_limit: 100_000,
                original_root_budget_sha256: "3".repeat(64),
                original_deadline_seal_sha256: "4".repeat(64),
            },
            wall_time_max_ms: 120_000,
            cleanup_max_ms: 30_000,
            output_bytes_max: 256 << 20,
            agent_renewable: true,
            provenance: vec![artifact("storage-profile.json")],
        }
    }

    #[test]
    fn explicit_profile_roundtrips_and_has_independent_canonical_domain() {
        let profile = profile();
        let raw = serde_json::to_string_pretty(&profile).unwrap();
        let decoded = RoventStorageProfile::from_json(&raw).unwrap();
        assert_eq!(
            serde_json::to_value(&profile).unwrap()["policy"],
            "archive_on_v1"
        );
        assert_eq!(decoded, profile);
        assert_eq!(
            profile.canonical_bytes().unwrap(),
            decoded.canonical_bytes().unwrap()
        );
        let bytes = profile.canonical_bytes().unwrap();
        let value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(value[0], ROVENT_STORAGE_CANONICAL_DOMAIN);
        assert_eq!(profile.canonical_sha256().unwrap(), digest(&bytes));
        let mut changed = profile.clone();
        changed.archive_limits.load_bytes_max -= 1;
        assert_ne!(
            profile.canonical_sha256().unwrap(),
            changed.canonical_sha256().unwrap()
        );
        changed = profile.clone();
        changed.source_pins.launch.sha256 = "b".repeat(64);
        assert_ne!(
            profile.canonical_sha256().unwrap(),
            changed.canonical_sha256().unwrap()
        );
    }

    #[test]
    fn parser_rejects_missing_unknown_duplicate_legacy_and_cyclic_lock_fields() {
        let value = serde_json::to_value(profile()).unwrap();
        for key in [
            "schema",
            "archive_limits",
            "helpers",
            "source_pins",
            "measurement_contract",
        ] {
            let mut changed = value.clone();
            changed.as_object_mut().unwrap().remove(key);
            assert!(
                RoventStorageProfile::from_json(&changed.to_string()).is_err(),
                "{key}"
            );
        }
        let mut changed = value.clone();
        changed["input_lock_sha256"] = "a".repeat(64).into();
        assert!(RoventStorageProfile::from_json(&changed.to_string()).is_err());
        changed = value.clone();
        changed["archive_limits"]["unknown_limit"] = 1.into();
        assert!(RoventStorageProfile::from_json(&changed.to_string()).is_err());
        changed = value.clone();
        changed["policy"] = "disabled".into();
        assert!(RoventStorageProfile::from_json(&changed.to_string()).is_err());
        changed = value.clone();
        changed["schema"] = crate::ROVENT_EXECUTION_DOMAIN.into();
        assert!(RoventStorageProfile::from_json(&changed.to_string()).is_err());
        let raw = value.to_string();
        for (original, duplicate) in [
            (
                "\"worker_threads\":1",
                "\"worker_threads\":1,\"worker_threads\":1",
            ),
            ("\"archive\":1", "\"archive\":1,\"archive\":1"),
        ] {
            let changed = raw.replacen(original, duplicate, 1);
            assert_ne!(changed, raw);
            assert!(RoventStorageProfile::from_json(&changed).is_err());
        }
        assert!(
            RoventStorageProfile::from_json(&" ".repeat(crate::MAX_MANIFEST_BYTES + 1)).is_err()
        );
    }

    #[test]
    fn every_runtime_selector_is_required_and_search_bounds_are_preserved() {
        let raw = serde_json::to_value(profile()).unwrap();
        let limits = profile().archive_limits;
        for key in [
            "game_bytes_max",
            "global_bytes_max",
            "index_entries_max",
            "index_bytes_max",
            "load_bytes_max",
            "load_deadline_max_ms",
            "record_payload_bytes_max",
            "max_load_pins",
        ] {
            let mut changed = raw.clone();
            changed["archive_limits"]
                .as_object_mut()
                .unwrap()
                .remove(key);
            assert!(
                RoventStorageProfile::from_json(&changed.to_string()).is_err(),
                "{key}"
            );
            changed = raw.clone();
            changed["archive_limits"][key] = 0.into();
            assert!(
                RoventStorageProfile::from_json(&changed.to_string()).is_err(),
                "{key}"
            );
        }
        for mutate in [
            (|v: &mut RoventArchiveRuntimeLimits| v.game_bytes_max = GAME_BYTES_MAX + 1)
                as fn(&mut RoventArchiveRuntimeLimits),
            |v| v.global_bytes_max = GLOBAL_BYTES_MAX + 1,
            |v| v.global_bytes_max = v.game_bytes_max - 1,
            |v| v.index_entries_max = INDEX_ENTRIES_MAX + 1,
            |v| v.index_bytes_max = RECORD_BYTES_MAX + 1,
            |v| v.load_bytes_max = RECORD_BYTES_MAX + 1,
            |v| v.record_payload_bytes_max = RECORD_BYTES_MAX + 1,
            |v| v.max_load_pins = LOAD_PINS_MAX + 1,
            |v| {
                v.game_bytes_max = 1024;
                v.record_payload_bytes_max = 1025;
            },
            |v| {
                v.game_bytes_max = 1024;
                v.record_payload_bytes_max = 512;
                v.load_bytes_max = 1025;
            },
        ] {
            let mut changed = limits;
            mutate(&mut changed);
            assert!(changed.validate().is_err(), "{changed:?}");
        }
        let mut maximum = limits;
        maximum.index_entries_max = INDEX_ENTRIES_MAX;
        maximum.index_bytes_max = RECORD_BYTES_MAX;
        maximum.record_payload_bytes_max = RECORD_BYTES_MAX;
        maximum.load_bytes_max = RECORD_BYTES_MAX;
        maximum.max_load_pins = LOAD_PINS_MAX;
        assert!(maximum.validate().is_ok());
    }

    #[test]
    fn worker_and_helper_counts_are_closed_and_source_pins_are_validated() {
        for mutate in [
            (|v: &mut RoventStorageProfile| v.maintenance_io_bytes_max = 0)
                as fn(&mut RoventStorageProfile),
            |v| v.maintenance_io_bytes_max = MAINTENANCE_IO_BYTES_MAX + 1,
            |v| v.worker_threads = 2,
            |v| v.command_capacity = 2,
            |v| v.completion_capacity = 0,
            |v| v.helpers.archive = 0,
            |v| v.helpers.shared_nnuec = 2,
            |v| v.source_pins.source.bytes = 0,
            |v| v.source_pins.codec.sha256 = "unknown".into(),
            |v| v.source_pins.launch.path = "../foreign.rs".into(),
            |v| v.measurement_contract = "unknown".into(),
        ] {
            let mut changed = profile();
            mutate(&mut changed);
            assert!(changed.validate().is_err(), "{changed:?}");
        }
    }

    #[test]
    fn all_four_native_modes_and_shared_helper_total_are_checked() {
        for mode in [
            ExecutionMode::CpuPc,
            ExecutionMode::HybridPc,
            ExecutionMode::CpuPcr,
            ExecutionMode::HybridPcr,
        ] {
            let m = manifest(mode);
            assert!(m.validate().is_ok());
            assert!(profile().validate_for_manifest(&m).is_ok());
        }
        let mut m = manifest(ExecutionMode::CpuPc);
        m.engine.nnuec = Some(nnuec());
        let mut selected = profile();
        assert!(selected.validate_for_manifest(&m).is_err());
        selected.helpers.shared_nnuec = 1;
        assert!(selected.validate_for_manifest(&m).is_err());
        m.resources.pools[1].max_threads = 2;
        assert!(selected.validate_for_manifest(&m).is_ok());
        let mut pcr = manifest(ExecutionMode::CpuPcr);
        pcr.resources.pools[1].max_threads = 2;
        assert!(selected.validate_for_manifest(&pcr).is_err());
    }

    #[test]
    fn helper_pool_must_be_unique_separate_sized_and_charged() {
        for mutate in [
            (|v: &mut RoventResourcePlan| {
                v.pools.remove(1);
            }) as fn(&mut RoventResourcePlan),
            |v| {
                v.pools[1].actors.insert(RoventActor::Manager);
            },
            |v| v.pools[1].max_threads = 0,
            |v| v.pools[1].cpu_charge_millicpus = 0,
            |v| v.pools.push(v.pools[1].clone()),
        ] {
            let mut resources = manifest(ExecutionMode::CpuPc).resources;
            mutate(&mut resources);
            assert!(profile().validate_resource_helpers(&resources).is_err());
        }
        let mut overcharged = manifest(ExecutionMode::CpuPc);
        overcharged.resources.pools[1].cpu_charge_millicpus = 2000;
        assert!(profile().validate_for_manifest(&overcharged).is_err());
    }

    #[test]
    fn own_board_storage_is_explicit_and_external_researcher_stays_unsupported() {
        let mut own_board = manifest(ExecutionMode::CpuPcr);
        own_board.engine.nnuec = None;
        assert!(own_board.validate().is_ok());
        assert!(profile().validate_for_manifest(&own_board).is_ok());
        let mut wrong_helpers = profile();
        wrong_helpers.helpers.shared_nnuec = 1;
        assert!(wrong_helpers.validate_for_manifest(&own_board).is_err());
        let mut external = manifest(ExecutionMode::CpuPcr);
        external.engine.researcher = Some(RoventResearcher::External {
            profile: Box::new(RoventExternalProfilePin {
                domain: ROVENT_EXTERNAL_PROFILE_DOMAIN.into(),
                profile: artifact("external-profile.json"),
                canonical_sha256: "8".repeat(64),
                capability_sha256: "9".repeat(64),
                purpose: RoventExternalPurpose::ResearchChecker,
                binary: artifact("external-engine"),
                source_commit: "b".repeat(40),
                requested_threads: 1,
                requested_hash_mib: 256,
            }),
        });
        external.resources.pools[2].actors = BTreeSet::from([RoventActor::ExternalR]);
        assert!(external.validate().is_ok());
        assert!(profile().validate_for_manifest(&external).is_err());
    }

    #[test]
    fn validating_storage_never_changes_legacy_manifest_or_lock_bytes() {
        let m = manifest(ExecutionMode::CpuPc);
        let old_bytes = m.canonical_bytes().unwrap();
        let lock = m.clone().lock().unwrap();
        let old_lock = serde_json::to_vec(&lock).unwrap();
        profile().validate_for_manifest(&m).unwrap();
        assert_eq!(m.canonical_bytes().unwrap(), old_bytes);
        assert_eq!(serde_json::to_vec(&lock).unwrap(), old_lock);
        let mut selected = m;
        selected
            .provenance
            .push(artifact("new-explicit-storage-profile.json"));
        assert_ne!(
            selected.lock().unwrap().canonical_sha256,
            lock.canonical_sha256
        );
    }
}
