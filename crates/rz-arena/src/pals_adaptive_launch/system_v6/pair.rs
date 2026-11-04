//! Independent two-endpoint declaration for PALS system runtime V6.
//!
//! A validated declaration records only planned inputs. It is not a process
//! owner, a verified provider, or proof of GPU completion.
use super::super::{PalsSourceFileV5, opening};
use crate::ArenaError;
use rz_experiments::{
    ArtifactRef, EngineEnvironmentV2, ExternalSourceV2, ExternalUciEndpointV2, ManifestError,
    NativeEngineRole, NativeResourceBudgetV1, OpeningSpec, PalsPairPlanV5,
    PalsSystemSourcePolicyV6, PalsSystemStartupLockV6, ToolIdentity, pals_private_snapshot_path_v5,
    source_component_digest_v5,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};
use std::collections::BTreeMap;

pub const PALS_SYSTEM_ARENA_PAIR_V6_DOMAIN: &str = "rz-pals-system-arena-pair-v6/1";

fn invalid(message: &str) -> ArenaError {
    ArenaError::Integrity(message.into())
}

fn require(condition: bool, message: &str) -> Result<(), ArenaError> {
    if condition {
        Ok(())
    } else {
        Err(invalid(message))
    }
}

fn text(value: &str, maximum: usize) -> bool {
    !value.is_empty() && value.len() <= maximum && !value.chars().any(char::is_control)
}

fn system_opening(pair: &PalsPairPlanV5) -> Result<OpeningSpec, ArenaError> {
    let mut value = opening(pair)?;
    value.history_origin = "registered V6 position command; FEN prefix remains unknown".into();
    Ok(value)
}

fn decode_sha(value: &str) -> Result<[u8; 32], ArenaError> {
    require(
        value.len() == 64
            && value
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)),
        "invalid source SHA256",
    )?;
    let mut bytes = [0u8; 32];
    for (i, byte) in bytes.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&value[2 * i..2 * i + 2], 16)
            .map_err(|_| invalid("invalid source SHA256"))?;
    }
    Ok(bytes)
}

/// Artifact paths and process metadata for one system-runtime endpoint.
/// Source rows are optional in observe mode; declaration alone is not evidence.
#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PalsSystemEndpointLaunchV6 {
    pub startup_lock: PalsSystemStartupLockV6,
    pub startup_lock_artifact: ArtifactRef,
    pub startup_snapshot_path: String,
    pub binary_snapshot_path: String,
    pub expected_uci_name: String,
    pub version: String,
    pub handshake_max_ms: u64,
    pub environment: Option<EngineEnvironmentV2>,
    pub source_files: BTreeMap<String, Vec<PalsSourceFileV5>>,
}

/// A fresh V6 pair; V5 canonical bytes and records are not converted.
#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PalsSystemArenaPairV6 {
    pub domain: String,
    pub pair_id: String,
    pub pair: PalsPairPlanV5,
    pub runner: ToolIdentity,
    pub opening_artifact: ArtifactRef,
    pub endpoints: [PalsSystemEndpointLaunchV6; 2],
    pub budget: NativeResourceBudgetV1,
}

#[derive(Clone, Debug)]
pub struct LockedPalsSystemArenaPairV6 {
    input: PalsSystemArenaPairV6,
    sha256: String,
    opening: OpeningSpec,
    endpoint_views: [ExternalUciEndpointV2; 2],
}

impl PalsSystemArenaPairV6 {
    pub fn from_json(input: &str) -> Result<Self, ArenaError> {
        let value: Self = crate::decode_json(input)?;
        value.validate()?;
        Ok(value)
    }

    pub fn validate(&self) -> Result<(), ArenaError> {
        require(
            self.domain == PALS_SYSTEM_ARENA_PAIR_V6_DOMAIN && text(&self.pair_id, 256),
            "invalid V6 pair identity",
        )?;
        self.runner.binary.validate()?;
        self.opening_artifact.validate()?;
        require(
            text(&self.runner.version, 256)
                && text(&self.runner.source_url, 2048)
                && text(&self.runner.source_commit, 256)
                && text(&self.runner.build_mode, 256)
                && text(&self.runner.compiler, 256)
                && text(&self.runner.target, 256)
                && text(&self.runner.isa, 256)
                && self.runner.dirty == self.runner.dirty_patch.is_some(),
            "runner identity or source/build evidence is incomplete",
        )?;
        if let Some(patch) = &self.runner.dirty_patch {
            patch.validate()?;
        }
        let p = &self.pair;
        let ids = [
            &self.endpoints[0].startup_lock.config.endpoint.id,
            &self.endpoints[1].startup_lock.config.endpoint.id,
        ];
        require(
            ids[0] != ids[1]
                && p.white_order[0] != p.white_order[1]
                && ids.contains(&&p.white_order[0])
                && ids.contains(&&p.white_order[1])
                && text(&p.opening_id, 256)
                && (1..=86_400_000).contains(&p.base_ms)
                && p.increment_ms <= 86_400_000
                && (1..=4095).contains(&p.max_plies)
                && p.concurrent_games == 1
                && p.restart_processes_each_game
                && !p.ponder
                && !p.score_adjudication,
            "pair colors, finite clock, game limit or execution policy differ",
        )?;
        let opening = system_opening(p)?;
        crate::validate_opening_artifact_for_spec(&opening, p.max_plies, &self.opening_artifact)?;
        let window = &self.endpoints[0].startup_lock.config.window;
        require(
            self.endpoints[0].startup_lock.config.run_id
                == self.endpoints[1].startup_lock.config.run_id
                && *window == self.endpoints[1].startup_lock.config.window
                && window.cleanup_max_ms >= 2
                && self.budget.max_output_bytes <= window.output_bytes_max,
            "pair endpoints must share the original run window and output bound",
        )?;

        let mut snapshots = BTreeMap::<String, ArtifactRef>::new();
        let mut sources = BTreeMap::<String, String>::new();
        let mut bind = |artifact: &ArtifactRef, snapshot: &str| -> Result<(), ArenaError> {
            artifact.validate()?;
            require(
                pals_private_snapshot_path_v5(snapshot),
                "unsafe V6 private snapshot path",
            )?;
            if let Some(previous) = snapshots.insert(snapshot.into(), artifact.clone()) {
                require(
                    previous == *artifact,
                    "different inputs alias a V6 snapshot",
                )?;
            }
            if let Some(previous) = sources.insert(artifact.path.clone(), snapshot.into()) {
                require(
                    previous == snapshot,
                    "one source has contradictory V6 snapshot names",
                )?;
            }
            Ok(())
        };
        for (index, endpoint) in self.endpoints.iter().enumerate() {
            endpoint.startup_lock.verify()?;
            let config = &endpoint.startup_lock.config;
            let expected_directory = if index == 0 {
                "baseline-runtime"
            } else {
                "candidate-runtime"
            };
            require(
                config.runtime_directory == expected_directory
                    && text(&endpoint.expected_uci_name, 512)
                    && text(&endpoint.version, 256)
                    && (1..=180_000).contains(&endpoint.handshake_max_ms)
                    && endpoint.handshake_max_ms <= window.wall_time_max_ms,
                "endpoint runtime, identification or deadline differs",
            )?;
            let bytes = endpoint.startup_lock.canonical_bytes()?;
            require(
                endpoint.startup_lock_artifact.bytes == bytes.len() as u64
                    && endpoint.startup_lock_artifact.sha256
                        == format!("{:x}", Sha256::digest(&bytes)),
                "startup lock artifact must bind its actual canonical bytes",
            )?;
            bind(
                &endpoint.startup_lock_artifact,
                &endpoint.startup_snapshot_path,
            )?;
            bind(&config.endpoint.binary, &endpoint.binary_snapshot_path)?;
            for asset in &config.assets {
                bind(&asset.artifact, &asset.snapshot_path)?;
            }
            if let Some(environment) = &endpoint.environment {
                environment.validate_native()?;
            }

            let declared = &config.source_components;
            require(
                endpoint.source_files.len() <= declared.len()
                    && endpoint
                        .source_files
                        .keys()
                        .all(|key| declared.contains_key(key))
                    && (config.source_policy != PalsSystemSourcePolicyV6::Strict
                        || endpoint.source_files.len() == declared.len()),
                "source rows differ from the selected source policy",
            )?;
            for (component, files) in &endpoint.source_files {
                require(
                    !files.is_empty() && files.len() <= 128,
                    "source component row count invalid",
                )?;
                let mut tuples = Vec::with_capacity(files.len());
                for file in files {
                    bind(&file.artifact, &file.snapshot_path)?;
                    tuples.push((
                        file.logical_source_path.clone(),
                        file.artifact.bytes,
                        decode_sha(&file.artifact.sha256)?,
                    ));
                }
                let actual = source_component_digest_v5(&tuples)?;
                let actual_hex: String = actual.iter().map(|byte| format!("{byte:02x}")).collect();
                require(
                    actual_hex == declared[component],
                    "source component digest differs",
                )?;
            }
        }
        let unique = self.unique_input_bytes()?;
        let b = self.budget;
        require(
            unique <= b.max_input_bytes
                && b.max_output_bytes > 0
                && b.max_runtime_bytes > 0
                && b.max_child_processes >= 5
                && b.max_child_processes <= 16
                && (16..=4096).contains(&b.max_runtime_files)
                && (1..=8).contains(&b.max_runtime_depth)
                && b.address_space_per_process_bytes > 0
                && b.max_artifact_bytes
                    == b.max_input_bytes
                        .checked_add(b.max_output_bytes)
                        .and_then(|x| x.checked_add(b.max_runtime_bytes))
                        .ok_or_else(|| invalid("artifact reservation overflow"))?,
            "invalid or insufficient bounded resource budget",
        )?;
        crate::native_launch::pair_stream_cap(b.max_output_bytes, 2 << 20)?;
        Ok(())
    }

    pub fn declared_artifacts(&self) -> Vec<&ArtifactRef> {
        let mut values = vec![&self.runner.binary, &self.opening_artifact];
        if let Some(patch) = &self.runner.dirty_patch {
            values.push(patch);
        }
        for endpoint in &self.endpoints {
            values.push(&endpoint.startup_lock_artifact);
            values.push(&endpoint.startup_lock.config.endpoint.binary);
            values.extend(
                endpoint
                    .startup_lock
                    .config
                    .assets
                    .iter()
                    .map(|asset| &asset.artifact),
            );
            values.extend(
                endpoint
                    .source_files
                    .values()
                    .flatten()
                    .map(|file| &file.artifact),
            );
            if let Some(environment) = &endpoint.environment {
                values.push(&environment.launcher);
            }
        }
        values
    }

    pub fn unique_input_bytes(&self) -> Result<u64, ManifestError> {
        let mut unique = BTreeMap::new();
        for artifact in self.declared_artifacts() {
            artifact.validate()?;
            if let Some(previous) = unique.insert(&artifact.path, artifact)
                && previous != artifact
            {
                return Err(ManifestError::Integrity(
                    "contradictory source artifact identities".into(),
                ));
            }
        }
        unique
            .values()
            .try_fold(0u64, |n, a| n.checked_add(a.bytes))
            .ok_or_else(|| ManifestError::Integrity("unique input byte overflow".into()))
    }

    pub fn lock(&self) -> Result<LockedPalsSystemArenaPairV6, ArenaError> {
        self.validate()?;
        let endpoint_views = std::array::from_fn(|i| {
            let e = &self.endpoints[i];
            let cfg = &e.startup_lock.config;
            ExternalUciEndpointV2 {
                id: cfg.endpoint.id.clone(),
                role: if i == 0 {
                    NativeEngineRole::Baseline
                } else {
                    NativeEngineRole::Candidate
                },
                family: "pals-system-v6".into(),
                version: e.version.clone(),
                expected_uci_name: e.expected_uci_name.clone(),
                binary: cfg.endpoint.binary.clone(),
                source: cfg
                    .endpoint
                    .source_commit
                    .as_ref()
                    .map(|commit| ExternalSourceV2 {
                        url: cfg.endpoint.binary.source.clone(),
                        commit: commit.clone(),
                        license: cfg.endpoint.binary.license.clone(),
                    }),
                arguments: vec![],
                assets: cfg.assets.iter().map(|a| a.artifact.clone()).collect(),
                requested_options: BTreeMap::new(),
                environment: e.environment.clone(),
                handshake_timeout_ms: e.handshake_max_ms,
            }
        });
        Ok(LockedPalsSystemArenaPairV6 {
            input: self.clone(),
            sha256: crate::canonical_sha256(&(PALS_SYSTEM_ARENA_PAIR_V6_DOMAIN, self))?,
            opening: system_opening(&self.pair)?,
            endpoint_views,
        })
    }
}

impl LockedPalsSystemArenaPairV6 {
    pub fn input(&self) -> &PalsSystemArenaPairV6 {
        &self.input
    }
    pub fn sha256(&self) -> &str {
        &self.sha256
    }
    pub fn opening(&self) -> &OpeningSpec {
        &self.opening
    }
    pub fn endpoint(&self, role: NativeEngineRole) -> &PalsSystemEndpointLaunchV6 {
        &self.input.endpoints[if role == NativeEngineRole::Baseline {
            0
        } else {
            1
        }]
    }
    pub fn endpoint_view(&self, role: NativeEngineRole) -> &ExternalUciEndpointV2 {
        &self.endpoint_views[if role == NativeEngineRole::Baseline {
            0
        } else {
            1
        }]
    }
    pub fn snapshot_relative_path(&self, artifact: &ArtifactRef) -> Option<String> {
        for endpoint in &self.input.endpoints {
            if artifact == &endpoint.startup_lock_artifact {
                return Some(endpoint.startup_snapshot_path.clone());
            }
            if artifact == &endpoint.startup_lock.config.endpoint.binary {
                return Some(endpoint.binary_snapshot_path.clone());
            }
            if let Some(asset) = endpoint
                .startup_lock
                .config
                .assets
                .iter()
                .find(|a| &a.artifact == artifact)
            {
                return Some(asset.snapshot_path.clone());
            }
            if let Some(file) = endpoint
                .source_files
                .values()
                .flatten()
                .find(|f| &f.artifact == artifact)
            {
                return Some(file.snapshot_path.clone());
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rz_experiments::*;
    use std::collections::BTreeSet;

    // Synthetic declarations are not provider, filesystem or GPU evidence.
    fn artifact(path: &str) -> ArtifactRef {
        ArtifactRef {
            path: path.into(),
            sha256: "a".repeat(64),
            bytes: 10,
            source: "https://example.org/test-only".into(),
            license: "MIT".into(),
        }
    }
    fn component(name: &str) -> PalsComponentV3 {
        PalsComponentV3 {
            semantic_id: name.into(),
            implementation_sha256: "b".repeat(64),
            options: BTreeMap::new(),
        }
    }
    fn endpoint(index: usize) -> PalsSystemEndpointLaunchV6 {
        let runtime = SystemRuntimeSpecV6 {
            platform: SystemPlatformV6::Linux,
            libraries: ["ort-core", "ort-provider-cuda"]
                .into_iter()
                .enumerate()
                .map(|(i, role)| SystemLibrarySpecV6 {
                    role: role.into(),
                    requested_path: format!("/test/lib{i}.so"),
                    canonical_path: format!("/test/lib{i}.so"),
                    version: None,
                    sha256: if i == 0 {
                        "1".repeat(64)
                    } else {
                        "2".repeat(64)
                    },
                    bytes: 100,
                    device: 1,
                    inode: i as u64 + 1,
                    required_mapping: true,
                })
                .collect(),
            system_identity_sha256: "3".repeat(64),
            cuda_device_id: 0,
            cuda_arena_bytes: 1024,
            startup_ms: 1000,
            drain_ms: 200,
            capture_max_libraries: 2,
            capture_max_file_bytes: 200,
            capture_max_total_bytes: 400,
            capture_timeout_ms: 100,
        };
        let options = PalsSystemNativeOptionsV6 {
            cache_public_memory: false,
            device_public_memory: false,
            cudnn_conv_search: "legacy_implicit_exhaustive".into(),
            approximate_warm: false,
            cpu_placement: PalsSystemCpuPlacementV6::StrictCudaOnly,
            startup_ms: 1000,
            drain_ms: 200,
        };
        let mut rt_component = component("system-runtime/test");
        rt_component.options = options.canonical_runtime_options(&runtime);
        let mut resources = PalsResourcePlanV5::default();
        resources.gpu_count = 1;
        resources.gpu_model = Some("declared-test-gpu".into());
        let model = PalsModelSpecificationV5 {
            profile: "virtual_review_v1".into(),
            model_schema: PALS_V5_MODEL_SCHEMA.into(),
            model_semantics: PALS_V5_MODEL_SEMANTICS.into(),
            encoding_schema: PALS_V5_ENCODING_SCHEMA.into(),
            semantic_sha256: "c".repeat(64),
            encoding_sha256: "d".repeat(64),
            implementation_sha256: "e".repeat(64),
            weights: artifact("test/weights"),
            artifact_domain: PalsArtifactDomainV5::Untrained,
            provenance: None,
            checkpoint_domain: None,
            epoch: 1,
            backend: PalsModelBackendV3::OrtCuda,
            precision: PalsPrecisionV3::Fp32,
            tf32: false,
            exported_roles: BTreeSet::from([PalsRoleV3::Proposer, PalsRoleV3::Critic]),
            width: 384,
            latent_count: 16,
            temporal_width: 64,
            candidate_head_width: 128,
            record_feature_axes: 8,
            query_feature_axes: 8,
            max_records: 64,
            max_anchors: 4,
            line_buckets: PALS_V5_LINE_BUCKETS.to_vec(),
            one_dynamic_line_graph: true,
            ids_as_numeric_features: false,
            fresh_warm_separate_domains: true,
        };
        use PalsStartupAssetPurposeV5 as P;
        let assets = [
            (P::Weights, "weights"),
            (P::ExportManifest, "export"),
            (P::RulesDescriptor, "rules"),
            (P::PublicGraph, "public"),
            (P::SharedPcGraph, "shared"),
            (P::InitializationEvidence, "initialization"),
            (P::InitializationCheckpointMetadata, "checkpoint"),
        ]
        .into_iter()
        .map(|(purpose, name)| PalsStartupAssetV5 {
            artifact: artifact(&format!("test/{name}")),
            snapshot_path: format!("inputs/test/{name}"),
            purpose,
        })
        .collect();
        let config = PalsSystemStartupConfigV6 {
            domain: PALS_SYSTEM_STARTUP_CONFIG_V6_DOMAIN.into(),
            run_id: "test-run".into(),
            endpoint: PalsSystemEndpointV6 {
                id: format!("engine-{index}"),
                binary: artifact("test/engine"),
                source_commit: None,
                mode: PalsEngineModeV5::Deployment,
                model,
                cpu: PalsCpuEndpointsV5 {
                    cpu_t: PalsCpuSelectionV5::Disabled,
                    cpu_r: PalsCpuSelectionV5::Own,
                    own: component("own/test"),
                    external: None,
                },
                context: PalsReviewContextV5::default(),
                coordinator: PalsCoordinatorV5::default(),
                resources,
                search: component("search/test"),
                runtime: rt_component,
                v_generate: false,
                v_load: false,
                v_call: false,
            },
            window: PalsExecutionWindowV5 {
                window_id: "test-window".into(),
                wall_time_max_ms: 5000,
                cleanup_max_ms: 1000,
                output_bytes_max: 64 << 20,
                agent_renewable: true,
            },
            source_components: BTreeMap::new(),
            source_policy: PalsSystemSourcePolicyV6::Observe,
            assets,
            system_runtime_lock: SystemRuntimeLockV6::new(NativeRuntimeSourceV6::SystemCuda(
                runtime,
            ))
            .unwrap(),
            native_options: options,
            process_generation: index as u64 + 1,
            runtime_directory: if index == 0 {
                "baseline-runtime"
            } else {
                "candidate-runtime"
            }
            .into(),
        };
        let startup_lock = PalsSystemStartupLockV6::new(config).unwrap();
        let bytes = startup_lock.canonical_bytes().unwrap();
        let mut startup_lock_artifact = artifact(&format!("test/startup-{index}"));
        startup_lock_artifact.bytes = bytes.len() as u64;
        startup_lock_artifact.sha256 = format!("{:x}", Sha256::digest(&bytes));
        PalsSystemEndpointLaunchV6 {
            startup_lock,
            startup_lock_artifact,
            startup_snapshot_path: format!("inputs/test/startup-{index}"),
            binary_snapshot_path: "inputs/test/engine".into(),
            expected_uci_name: "test-engine".into(),
            version: "test-version".into(),
            handshake_max_ms: 1500,
            environment: None,
            source_files: BTreeMap::new(),
        }
    }
    fn fixture() -> PalsSystemArenaPairV6 {
        let pair = PalsPairPlanV5 {
            opening_id: "test-opening".into(),
            position_command: "position startpos".into(),
            white_order: ["engine-0".into(), "engine-1".into()],
            base_ms: 1000,
            increment_ms: 100,
            max_plies: 16,
            seed: 1,
            concurrent_games: 1,
            restart_processes_each_game: true,
            ponder: false,
            score_adjudication: false,
        };
        let opening = system_opening(&pair).unwrap();
        let pgn = crate::opening_pgn_for_spec(&opening, pair.max_plies).unwrap();
        let mut opening_artifact = artifact("test/opening");
        opening_artifact.bytes = pgn.len() as u64;
        opening_artifact.sha256 = format!("{:x}", Sha256::digest(pgn.as_bytes()));
        PalsSystemArenaPairV6 {
            domain: PALS_SYSTEM_ARENA_PAIR_V6_DOMAIN.into(),
            pair_id: "test-pair".into(),
            pair,
            runner: ToolIdentity {
                version: "test".into(),
                source_url: "https://example.org".into(),
                source_commit: "test-source".into(),
                binary: artifact("test/runner"),
                dirty: false,
                dirty_patch: None,
                build_mode: "test".into(),
                compiler: "test".into(),
                target: "test".into(),
                isa: "test".into(),
            },
            opening_artifact,
            endpoints: [endpoint(0), endpoint(1)],
            budget: NativeResourceBudgetV1 {
                max_input_bytes: 1 << 30,
                max_output_bytes: 4 << 20,
                max_runtime_bytes: 4 << 20,
                max_artifact_bytes: (1 << 30) + (8 << 20),
                max_child_processes: 5,
                max_runtime_files: 16,
                max_runtime_depth: 4,
                address_space_per_process_bytes: 1 << 30,
            },
        }
    }
    #[test]
    fn startup_artifact_must_match_each_endpoint_lock_bytes() {
        for index in 0..2 {
            let mut pair = fixture();
            pair.validate().unwrap();
            pair.endpoints[index].startup_lock_artifact.bytes += 1;
            assert!(pair.validate().is_err());
            let mut pair = fixture();
            pair.endpoints[index].startup_lock_artifact.sha256 = "f".repeat(64);
            assert!(pair.validate().is_err());
        }
    }
    #[test]
    fn observation_keeps_source_unknown() {
        let locked = fixture().lock().unwrap();
        for role in [NativeEngineRole::Baseline, NativeEngineRole::Candidate] {
            assert!(
                locked
                    .endpoint(role)
                    .startup_lock
                    .config
                    .endpoint
                    .source_commit
                    .is_none()
            );
            assert!(locked.endpoint_view(role).source.is_none());
        }
    }
    #[test]
    fn endpoints_cannot_have_distinct_windows() {
        let mut pair = fixture();
        let config = &mut pair.endpoints[1].startup_lock.config;
        config.window.wall_time_max_ms += 1;
        let lock = PalsSystemStartupLockV6::new(config.clone()).unwrap();
        let bytes = lock.canonical_bytes().unwrap();
        pair.endpoints[1].startup_lock = lock;
        pair.endpoints[1].startup_lock_artifact.bytes = bytes.len() as u64;
        pair.endpoints[1].startup_lock_artifact.sha256 = format!("{:x}", Sha256::digest(&bytes));
        assert!(pair.validate().is_err());
    }
    #[test]
    fn endpoint_specific_clock_cannot_override_shared_pair_clock() {
        let mut input = serde_json::to_value(fixture()).unwrap();
        input["endpoints"][1]["clock"] = serde_json::json!({"base_ms": 2000});
        assert!(PalsSystemArenaPairV6::from_json(&input.to_string()).is_err());
    }

    #[cfg(target_os = "linux")]
    fn preflight_wire(lock: &PalsSystemStartupLockV6, pid: u32) -> serde_json::Value {
        let config = &lock.config;
        serde_json::json!({
            "domain": "rz-pals-system-preflight-v6/1",
            "lock_sha256": lock.canonical_sha256,
            "system_runtime_lock_sha256": config.system_runtime_lock.canonical_sha256,
            "endpoint_id": config.endpoint.id,
            "process_id": pid,
            "process_generation": config.process_generation,
            "mapped_executable_sha256": config.endpoint.binary.sha256,
            "mapped_executable_bytes": config.endpoint.binary.bytes,
            "source_commit_observation": "unknown",
            "source_components_observation": "unknown",
            "source_components_scope": "declared-v5-registry-subset",
            "system_v6_source_observation": "unknown",
            "system_v6_source_scope": "rz-pals-system-v6-boundary-source/1",
            "assets_held": config.assets.len(),
            "sessions_created": 0,
            "provider_initialized": false,
            "neural_execution": false,
            "physical_completion": "not_applicable_no_gpu_execution"
        })
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn preflight_pids_are_distinct_across_both_roles() {
        use super::super::preflight::validate_preflight_pids;
        assert!(validate_preflight_pids([11, 12, 21, 22]).is_ok());
        assert!(validate_preflight_pids([11, 12, 11, 22]).is_err());
        assert!(validate_preflight_pids([11, 12, 21, 0]).is_err());
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn preflight_raw_report_is_bound_to_original_lock_endpoint_and_generation() {
        use super::super::preflight::validate_preflight_record;
        let pair = fixture();
        let original = &pair.endpoints[0].startup_lock;
        let other = &pair.endpoints[1].startup_lock;
        let wire = preflight_wire(original, 11);
        let raw = serde_json::to_vec(&wire).unwrap();
        assert!(validate_preflight_record(original, 11, &raw).is_ok());
        assert!(validate_preflight_record(other, 11, &raw).is_err());
        for field in ["lock_sha256", "endpoint_id", "process_generation"] {
            let mut changed = wire.clone();
            let replacement = preflight_wire(other, 11);
            changed[field] = replacement[field].clone();
            assert!(
                validate_preflight_record(original, 11, &serde_json::to_vec(&changed).unwrap())
                    .is_err(),
                "preflight must not relabel {field}"
            );
        }
        // Validation borrows and retains the exact wire bytes, including the
        // explicit source observations; it does not promote them to verified.
        assert_eq!(raw, serde_json::to_vec(&wire).unwrap());
        assert_eq!(wire["source_commit_observation"], "unknown");
        assert_eq!(wire["source_components_observation"], "unknown");
        assert_eq!(wire["system_v6_source_observation"], "unknown");
        for field in [
            "source_components_scope",
            "system_v6_source_observation",
            "system_v6_source_scope",
        ] {
            let mut missing = wire.clone();
            missing.as_object_mut().unwrap().remove(field);
            assert!(
                validate_preflight_record(original, 11, &serde_json::to_vec(&missing).unwrap())
                    .is_err()
            );
        }
        for field in ["source_components_scope", "system_v6_source_scope"] {
            let mut forged_scope = wire.clone();
            forged_scope[field] = serde_json::json!("different_scope");
            assert!(
                validate_preflight_record(
                    original,
                    11,
                    &serde_json::to_vec(&forged_scope).unwrap()
                )
                .is_err()
            );
        }
        let mut forged_boundary = wire.clone();
        forged_boundary["system_v6_source_observation"] = serde_json::json!("verified");
        assert!(
            validate_preflight_record(original, 11, &serde_json::to_vec(&forged_boundary).unwrap())
                .is_err()
        );
        let mut forged = wire;
        forged["source_commit_observation"] = serde_json::json!("verified");
        assert!(
            validate_preflight_record(original, 11, &serde_json::to_vec(&forged).unwrap()).is_err()
        );
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn preflight_source_scopes_are_independent_in_observe_and_strict_modes() {
        use super::super::preflight::validate_preflight_record;
        let pair = fixture();
        let mut config = pair.endpoints[0].startup_lock.config.clone();
        config
            .source_components
            .insert("pals_system_v6".into(), "a".repeat(64));
        let observe = PalsSystemStartupLockV6::new(config.clone()).unwrap();
        let mut wire = preflight_wire(&observe, 11);
        wire["system_v6_source_observation"] = serde_json::json!("verified");
        assert!(
            validate_preflight_record(&observe, 11, &serde_json::to_vec(&wire).unwrap()).is_ok()
        );
        wire["source_components_observation"] = serde_json::json!("verified");
        assert!(
            validate_preflight_record(&observe, 11, &serde_json::to_vec(&wire).unwrap()).is_err()
        );

        config.source_policy = PalsSystemSourcePolicyV6::Strict;
        config.endpoint.source_commit = Some("b".repeat(40));
        config
            .source_components
            .insert("v5_test_component".into(), "c".repeat(64));
        let strict = PalsSystemStartupLockV6::new(config).unwrap();
        let mut wire = preflight_wire(&strict, 11);
        for field in [
            "source_commit_observation",
            "source_components_observation",
            "system_v6_source_observation",
        ] {
            wire[field] = serde_json::json!("verified");
        }
        assert!(
            validate_preflight_record(&strict, 11, &serde_json::to_vec(&wire).unwrap()).is_ok()
        );
        for field in [
            "source_commit_observation",
            "source_components_observation",
            "system_v6_source_observation",
        ] {
            for status in ["unknown", "mismatch"] {
                let mut bad = wire.clone();
                bad[field] = serde_json::json!(status);
                assert!(
                    validate_preflight_record(&strict, 11, &serde_json::to_vec(&bad).unwrap())
                        .is_err()
                );
            }
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn preflight_execution_pin_alone_is_not_a_v5_source_observation() {
        use super::super::preflight::validate_preflight_record;
        let pair = fixture();
        let mut config = pair.endpoints[0].startup_lock.config.clone();
        config.source_components.insert(
            "pals_system_v6_execution".into(),
            format!("{:x}", Sha256::digest(b"fixture execution source")),
        );
        let lock = PalsSystemStartupLockV6::new(config).unwrap();
        let mut wire = preflight_wire(&lock, 11);
        assert!(validate_preflight_record(&lock, 11, &serde_json::to_vec(&wire).unwrap()).is_ok());

        wire["source_components_observation"] = serde_json::json!("verified");
        assert!(validate_preflight_record(&lock, 11, &serde_json::to_vec(&wire).unwrap()).is_err());

        let mut both = lock.config.clone();
        both.source_components.insert(
            "pals_system_v6".into(),
            format!("{:x}", Sha256::digest(b"fixture boundary source")),
        );
        let lock = PalsSystemStartupLockV6::new(both).unwrap();
        let mut wire = preflight_wire(&lock, 11);
        assert!(validate_preflight_record(&lock, 11, &serde_json::to_vec(&wire).unwrap()).is_ok());
        wire["source_components_observation"] = serde_json::json!("verified");
        assert!(validate_preflight_record(&lock, 11, &serde_json::to_vec(&wire).unwrap()).is_err());
    }
    #[test]
    fn environment_rejects_loader_injection_and_unbound_device_visibility() {
        for name in ["LD_PRELOAD", "LD_LIBRARY_PATH", "CUDA_VISIBLE_DEVICES"] {
            let mut pair = fixture();
            pair.endpoints[0].environment = Some(EngineEnvironmentV2 {
                launcher: artifact("test/env-launcher"),
                variables: BTreeMap::from([(name.into(), "test-value".into())]),
            });
            assert!(
                pair.validate().is_err(),
                "{name} must not override the registered runtime"
            );
        }
        let mut pair = fixture();
        pair.endpoints[0].environment = Some(EngineEnvironmentV2 {
            launcher: artifact("test/env-launcher"),
            variables: BTreeMap::from([
                ("LANG".into(), "C.UTF-8".into()),
                ("MALLOC_ARENA_MAX".into(), "2".into()),
            ]),
        });
        assert!(pair.validate().is_ok());
    }
    #[test]
    fn declaration_does_not_enforce_an_eighteen_gib_ceiling() {
        let mut pair = fixture();
        pair.budget.max_input_bytes = 19u64 << 30;
        pair.budget.max_artifact_bytes = pair.budget.max_input_bytes
            + pair.budget.max_output_bytes
            + pair.budget.max_runtime_bytes;
        assert!(pair.validate().is_ok());
    }

    fn reported_session() -> (
        super::super::PalsSystemArenaEndpointV6,
        super::super::PalsSystemChildReceiptClaimsV6,
    ) {
        use super::super::{
            PalsSystemArenaEndpointV6, PalsSystemChildReceiptClaimsV6, PalsSystemReceiptClaimV6,
        };
        let launch = endpoint(0);
        let config = &launch.startup_lock.config;
        let lock = &config.system_runtime_lock;
        let NativeRuntimeSourceV6::SystemCuda(system) = &lock.runtime else {
            panic!("synthetic fixture must declare system CUDA");
        };
        let mapped: Vec<SystemLibraryObservationV6> = system
            .libraries
            .iter()
            .map(|library| SystemLibraryObservationV6 {
                role: library.role.clone(),
                canonical_path: library.canonical_path.clone(),
                sha256: library.sha256.clone(),
                bytes: library.bytes,
                device: library.device,
                inode: library.inode,
                mapped: Some(true),
            })
            .collect();
        let startup = SystemRuntimeReceiptV6 {
            domain: SYSTEM_RUNTIME_RECEIPT_V6_DOMAIN.into(),
            lock_sha256: lock.canonical_sha256.clone(),
            process_id: 123,
            process_generation: config.process_generation,
            observed_identity_sha256: Some(system.system_identity_sha256.clone()),
            observed_libraries: mapped,
            unregistered_native_mappings: Some(vec![]),
            provider_ready: ProviderReadinessV6::Ready,
            physical_completion: PhysicalCompletionV6::Unknown,
        };
        let mut termination = startup.clone();
        termination.physical_completion = PhysicalCompletionV6::Confirmed {
            completion_id: "synthetic-only".into(),
        };
        let reported = |receipt: SystemRuntimeReceiptV6| PalsSystemReceiptClaimV6::Reported {
            bytes: serde_json::to_vec(&receipt).unwrap(),
            receipt,
        };
        let claims = PalsSystemChildReceiptClaimsV6 {
            endpoint_id: config.endpoint.id.clone(),
            process_id: 123,
            process_generation: config.process_generation,
            startup: reported(startup),
            termination: reported(termination),
        };
        (
            PalsSystemArenaEndpointV6::new(launch.startup_lock).unwrap(),
            claims,
        )
    }

    #[test]
    fn session_reports_are_bound_to_the_preserved_raw_bytes() {
        use super::super::PalsSystemReceiptClaimV6;
        let (endpoint, mut claims) = reported_session();
        assert_eq!(
            endpoint
                .validate_reported_session(&claims)
                .unwrap()
                .strict_inventory_empty,
            Some(true)
        );
        if let PalsSystemReceiptClaimV6::Reported { receipt, .. } = &mut claims.startup {
            receipt.provider_ready = ProviderReadinessV6::Failed;
        } else {
            panic!("synthetic report expected");
        }
        assert!(endpoint.validate_reported_session(&claims).is_err());
    }

    #[test]
    fn session_requires_registered_mappings_ready_and_confirmed_child_claim() {
        use super::super::PalsSystemReceiptClaimV6;
        for case in 0..3 {
            let (endpoint, mut claims) = reported_session();
            let target = if case == 2 {
                &mut claims.termination
            } else {
                &mut claims.startup
            };
            if let PalsSystemReceiptClaimV6::Reported { bytes, receipt } = target {
                match case {
                    0 => receipt.observed_libraries[0].mapped = None,
                    1 => receipt.provider_ready = ProviderReadinessV6::Failed,
                    _ => receipt.physical_completion = PhysicalCompletionV6::Unknown,
                }
                *bytes = serde_json::to_vec(receipt).unwrap();
            } else {
                panic!("synthetic report expected");
            }
            assert!(endpoint.validate_reported_session(&claims).is_err());
        }
    }

    #[test]
    fn unregistered_mapping_inventory_is_reported_without_promotion() {
        use super::super::PalsSystemReceiptClaimV6;
        for (mappings, expected) in [
            (Some(vec!["/test/extra.so".into()]), Some(false)),
            (None, None),
        ] {
            let (endpoint, mut claims) = reported_session();
            if let PalsSystemReceiptClaimV6::Reported { bytes, receipt } = &mut claims.startup {
                receipt.unregistered_native_mappings = mappings;
                *bytes = serde_json::to_vec(receipt).unwrap();
            } else {
                panic!("synthetic report expected");
            }
            assert_eq!(
                endpoint
                    .validate_reported_session(&claims)
                    .unwrap()
                    .strict_inventory_empty,
                expected
            );
        }
    }

    #[test]
    fn optional_lazy_mapping_unknown_is_observed_without_blocking_function() {
        use super::super::{PalsSystemArenaEndpointV6, PalsSystemReceiptClaimV6};
        let (endpoint, mut claims) = reported_session();
        let mut config = endpoint.startup_lock().config.clone();
        let NativeRuntimeSourceV6::SystemCuda(mut runtime) =
            config.system_runtime_lock.runtime.clone()
        else {
            panic!("synthetic fixture must declare system CUDA");
        };
        let optional = SystemLibrarySpecV6 {
            role: "optional-lazy".into(),
            requested_path: "/test/optional.so".into(),
            canonical_path: "/test/optional.so".into(),
            version: None,
            sha256: "4".repeat(64),
            bytes: 100,
            device: 1,
            inode: 3,
            required_mapping: false,
        };
        runtime.libraries.push(optional.clone());
        runtime.capture_max_libraries = 3;
        config.endpoint.runtime.options = config.native_options.canonical_runtime_options(&runtime);
        config.system_runtime_lock =
            SystemRuntimeLockV6::new(NativeRuntimeSourceV6::SystemCuda(runtime)).unwrap();
        let lock_sha256 = config.system_runtime_lock.canonical_sha256.clone();
        let endpoint =
            PalsSystemArenaEndpointV6::new(PalsSystemStartupLockV6::new(config).unwrap()).unwrap();
        for claim in [&mut claims.startup, &mut claims.termination] {
            if let PalsSystemReceiptClaimV6::Reported { bytes, receipt } = claim {
                receipt.lock_sha256 = lock_sha256.clone();
                receipt.observed_libraries.push(SystemLibraryObservationV6 {
                    role: optional.role.clone(),
                    canonical_path: optional.canonical_path.clone(),
                    sha256: optional.sha256.clone(),
                    bytes: optional.bytes,
                    device: optional.device,
                    inode: optional.inode,
                    mapped: None,
                });
                *bytes = serde_json::to_vec(receipt).unwrap();
            } else {
                panic!("synthetic report expected");
            }
        }
        let observation = endpoint.validate_reported_session(&claims).unwrap();
        assert!(!observation.optional_mappings_observed);
        assert_eq!(observation.strict_inventory_empty, Some(true));
    }

    #[test]
    fn unconnected_native_owner_is_rejected_before_execution() {
        let locked = fixture().lock().unwrap();
        let result = crate::NativeLaunchDeclaration::validate_execution(&locked);
        assert!(result.is_err());
        assert!(crate::NativeProviderDeclaration::requires_physical_lifetime_guard(&locked));
        assert!(
            crate::NativeProviderDeclaration::uses_provider_records(
                &locked,
                NativeEngineRole::Baseline,
            )
            .unwrap()
        );
    }
}
