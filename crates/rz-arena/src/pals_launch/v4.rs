//! PALS V4 execution through the pinned arena owner and physical lifecycle.
//! V3 recipe DTOs below describe unchanged executable resources only. The V4
//! semantic lock, argv, observed producer policies and receipts retain V4 IDs.
//! V4 base.model.implementation_sha256 pins compiled model-boundary sources;
//! the native recipe's adapter_source_sha256 independently pins the native
//! adapter. Runtime getters attest each against its own declared scope.
use super::*;
use rz_eval::pals_model::{PalsModelConfig, PalsModelProfile};
use rz_experiments::*;

mod followup;
pub use followup::*;
#[path = "v4_resources.rs"]
mod resources;
pub use resources::*;
#[cfg(any(feature = "pals-collection-onnx", feature = "native-cuda"))]
mod lifecycle;

pub const PALS_ARENA_V4_DOMAIN: &str = "rz-pals-arena-launch-v4/1";
pub const PALS_NATIVE_STARTUP_V4_DOMAIN: &str = "rz-pals-native-startup-v4/1";
pub const PALS_NATIVE_TERMINATION_V4_DOMAIN: &str = "rz-pals-native-termination-v4/1";
const V4_METADATA_CAP: u64 = 4 * 1024 * 1024;
pub const PALS_SEARCH_WORK_STARTUP_V4_DOMAIN: &str = "rz-pals-search-work-startup-v4/1";
pub const PALS_SEARCH_WORK_TERMINATION_V4_DOMAIN: &str = "rz-pals-search-work-termination-v4/1";

fn hex(bytes: [u8; 32]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}
#[cfg(any(feature = "pals-collection", feature = "native-cuda"))]
fn validate_actual_encoding(e: &PalsEndpointV4, n: &PalsOnnxLaunchV3) -> Result<(), ArenaError> {
    let cfg = config(e);
    require(
        n.encoding_semantic_sha256
            == hex(rz_uci::pals_native::pals_rules_encoding_semantic_digest_for_config(&cfg))
            && e.model_v2.encoding_sha256
                == hex(rz_uci::pals_native::pals_fresh_encoding_semantic_digest_for_config(&cfg)),
        "V4 actual model/Rules encoding differs from declared profile",
    )
}
#[cfg(not(any(feature = "pals-collection", feature = "native-cuda")))]
fn validate_actual_encoding(_e: &PalsEndpointV4, _n: &PalsOnnxLaunchV3) -> Result<(), ArenaError> {
    Err(invalid(
        "native V4 encoding verification requires the explicitly selected PALS feature",
    ))
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct PalsArenaLaunchV4 {
    pub domain: String,
    pub semantic_lock: PalsInputLockV4,
    pub runner: ToolIdentity,
    pub opening_artifact: ArtifactRef,
    pub endpoints: [PalsEndpointLaunchV3; 2],
    pub budget: NativeResourceBudgetV1,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub followup_execution: Option<PalsFollowupExecutionBindingV4>,
    /// Absence preserves the strict inherited-resource admission and its wire bytes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resource_profile: Option<PalsArenaResourceProfileV4>,
}
#[derive(Clone, Debug)]
pub struct LockedPalsArenaLaunchV4 {
    input: PalsArenaLaunchV4,
    sha256: String,
    opening: OpeningSpec,
    endpoint_views: [ExternalUciEndpointV2; 2],
    #[cfg(target_os = "linux")]
    archive_roots: std::sync::Arc<std::sync::Mutex<BTreeMap<usize, followup::ArchiveRootPins>>>,
    #[cfg(target_os = "linux")]
    resource_observations: std::sync::Arc<std::sync::Mutex<Vec<PalsArenaResourceObservationV4>>>,
}
fn requested_options(engine: &PalsEngineV4) -> &BTreeMap<String, String> {
    match engine {
        PalsEngineV4::Pals(e) => &e.base.requested_options,
        PalsEngineV4::OwnCpu(e) => &e.requested_options,
        PalsEngineV4::ReferenceUci(e) => &e.requested_options,
    }
}
fn config(e: &PalsEndpointV4) -> PalsModelConfig {
    let profile = match e.model_v2.profile {
        PalsModelProfileV4::LegacySummaryV1 => PalsModelProfile::LegacySummaryV1,
        PalsModelProfileV4::FullLineV2 => PalsModelProfile::FullLineV2,
        PalsModelProfileV4::InteractionHeadV2 => PalsModelProfile::InteractionHeadV2,
        PalsModelProfileV4::FullLineInteractionV2 => PalsModelProfile::FullLineInteractionV2,
    };
    PalsModelConfig::for_profile(profile)
}
fn policy_arguments(
    e: &PalsEndpointV4,
    binding: Option<&PalsEndpointExecutionBindingV4>,
) -> Result<Vec<String>, ArenaError> {
    if let PalsIterativeRepairPolicyV4::Bounded(l) = &e.policies.iterative_repair {
        require(
            l.max_repairs_per_first_move == 3 && l.max_pending_questions == 64,
            "native queue CLI registers exactly three Repairs and 64 pending questions; lower declared limits need their own actual producer selector",
        )?;
    }
    if let PalsPausedStackPolicyV4::Bounded(l) = &e.policies.paused_stack {
        require(
            l.tokens_max == 1 && l.bytes_max == 8 * 1024 * 1024,
            "native PausedStack selector registers one token and 8MiB; declared limits must equal the actual selected owner",
        )?;
    }
    let mut args = vec![
        "--pals-arena-wire=v4".into(),
        format!("--pals-model-profile={}", config(e).profile.as_str()),
        format!(
            "--pals-resolver-policy={}",
            match e.policies.resolver {
                PalsResolverPolicyV4::OwnRaw => "own-raw-restricted",
                PalsResolverPolicyV4::ModelWdl => "model-wdl-restricted",
            }
        ),
        format!(
            "--pals-cuda-private-warm={}",
            matches!(e.policies.cuda_warm, PalsCudaWarmPolicyV4::ApproxWarm(_))
        ),
    ];
    if e.base.cpu_r.is_own() {
        args.push(format!(
            "--pals-cpu-resume={}",
            match e.policies.paused_stack {
                PalsPausedStackPolicyV4::Disabled => "completed-iteration",
                PalsPausedStackPolicyV4::Bounded(_) => "paused-stack",
            }
        ));
    }
    match (&e.policies.recheck, &e.policies.iterative_repair) {
        (PalsRecheckPolicyV4::Disabled, PalsIterativeRepairPolicyV4::Disabled) => {}
        (PalsRecheckPolicyV4::FrozenWdl, PalsIterativeRepairPolicyV4::Disabled) => {
            args.push("--pals-post-repair-recheck=frozen-model-wdl-v2".into())
        }
        (PalsRecheckPolicyV4::FrozenWdl, PalsIterativeRepairPolicyV4::Bounded(_)) => {
            args.push("--pals-post-repair-recheck=iterative-frozen-model-wdl-v2".into())
        }
        _ => return Err(invalid("iterative repair requires frozen WDL recheck")),
    }
    args.extend(followup::selected_policy_arguments(e, binding)?);
    Ok(args)
}

impl PalsArenaValidationContext for LockedPalsArenaLaunchV4 {
    #[cfg(all(
        target_os = "linux",
        any(feature = "pals-collection-onnx", feature = "native-cuda")
    ))]
    fn archive_runtime_root_selected(&self, role: NativeEngineRole) -> bool {
        self.input
            .execution_binding(role_index(role))
            .is_some_and(|b| b.cold_archive.is_some())
    }
    #[cfg(target_os = "linux")]
    fn verify_semantic(&self) -> Result<(), ArenaError> {
        self.input.semantic_lock.verify().map_err(ArenaError::from)
    }
    fn launch_sha256(&self) -> &str {
        &self.sha256
    }
    fn native(&self, role: NativeEngineRole) -> Result<&PalsOnnxLaunchV3, ArenaError> {
        LockedPalsArenaLaunchV4::native(self, role)
    }
    fn native_receipt_encoding_identity(
        &self,
        role: NativeEngineRole,
    ) -> Result<NativeReceiptEncodingIdentity<'_>, ArenaError> {
        self.native(role)?;
        match &self.input.semantic_lock.manifest.engines[role_index(role)] {
            PalsEngineV4::Pals(endpoint) => Ok(NativeReceiptEncodingIdentity::FreshModelInput(
                &endpoint.model_v2.encoding_sha256,
            )),
            _ => Err(invalid(
                "native V4 requires actual Fresh model/input identity",
            )),
        }
    }
    fn recipe(&self, role: NativeEngineRole) -> &PalsEndpointLaunchV3 {
        &self.input.endpoints[role_index(role)]
    }
    fn endpoint_view(&self, role: NativeEngineRole) -> &ExternalUciEndpointV2 {
        &self.endpoint_views[role_index(role)]
    }
    fn pals_endpoint(&self, role: NativeEngineRole) -> Result<&PalsEndpointV3, ArenaError> {
        match &self.input.semantic_lock.manifest.engines[role_index(role)] {
            PalsEngineV4::Pals(e) => Ok(&e.base),
            _ => Err(invalid("native V4 requires actual PALS model identity")),
        }
    }
    fn resource(&self, role: NativeEngineRole) -> &PalsResourcePolicyV3 {
        &self.input.semantic_lock.manifest.resources[role_index(role)]
    }
    fn external_binding(
        &self,
        role: NativeEngineRole,
    ) -> Result<Option<(&PalsExternalCpuRV3, &PalsExternalCpuRLaunchV3)>, ArenaError> {
        self.input.external_cpu_r_binding(role_index(role))
    }
    fn native_wire_version(&self) -> u32 {
        4
    }
    fn native_receipt_byte_cap(&self, role: NativeEngineRole) -> usize {
        if self.input.execution_binding(role_index(role)).is_some() {
            256 * 1024
        } else {
            128 * 1024
        }
    }
    fn device_public_memory_selected(&self, role: NativeEngineRole) -> bool {
        self.input
            .execution_binding(role_index(role))
            .and_then(|b| b.cuda_warm.as_ref())
            .is_some()
    }
    fn shared_role_layout(&self, role: NativeEngineRole) -> &str {
        self.input
            .execution_binding(role_index(role))
            .and_then(|b| b.cuda_warm.as_ref())
            .map_or("shared_pc_if", |w| w.layout.as_str())
    }
    fn validate_selected_private_execution(
        &self,
        role: NativeEngineRole,
        record: &serde_json::Value,
    ) -> Result<(), ArenaError> {
        followup::validate_selected_private_execution(self, role, record)
    }
    fn native_wire_domains(&self) -> (&'static str, &'static str) {
        (
            PALS_NATIVE_STARTUP_V4_DOMAIN,
            PALS_NATIVE_TERMINATION_V4_DOMAIN,
        )
    }
    #[cfg(target_os = "linux")]
    fn native_wire_filenames(&self) -> (&'static str, &'static str) {
        (
            "pals-native-startup.v4.json",
            "pals-native-termination.v4.json",
        )
    }
    fn validate_policy_pair(
        &self,
        role: NativeEngineRole,
        startup: &serde_json::Value,
        termination: &serde_json::Value,
    ) -> Result<(), ArenaError> {
        validate_actual_policy_work(self, role, startup, termination)
    }
    fn validate_wire_pair(
        &self,
        role: NativeEngineRole,
        startup: &serde_json::Value,
        termination: &serde_json::Value,
    ) -> Result<(), ArenaError> {
        validate_actual_policy_work(
            self,
            role,
            &startup["search_work"],
            &termination["search_work"],
        )?;
        validate_followup_marker_pair(self, role, startup, termination)?;
        followup::validate_lifecycle_pair(self, role, startup, termination).map(|_| ())
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct PalsActualFollowupMarkerV4 {
    pub model_identity: String,
    pub model_profile: Option<String>,
    pub model_semantics: String,
    pub model_semantics_sha256: Option<[u8; 32]>,
    pub model_implementation_sha256: Option<[u8; 32]>,
    pub encoding_schema: Option<String>,
    pub model_configuration: Option<PalsModelConfig>,
    pub model_epoch: Option<[u8; 32]>,
    pub resolver_policy: String,
    pub cpu_resume: Option<String>,
    pub post_repair_recheck: String,
    pub policy_identity: String,
    pub policy_conditions_sha256: Option<[u8; 32]>,
    pub search_implementation_sha256: [u8; 32],
    pub resolver_implementation_sha256: [u8; 32],
    pub resolver_semantics_sha256: [u8; 32],
    pub cuda_warm: bool,
}

fn expected_recheck(e: &PalsEndpointV4) -> &'static str {
    match (&e.policies.recheck, &e.policies.iterative_repair) {
        (PalsRecheckPolicyV4::Disabled, _) => "disabled",
        (_, PalsIterativeRepairPolicyV4::Disabled) => "frozen-model-wdl-v2",
        _ => "iterative-frozen-model-wdl-v2",
    }
}
fn validate_actual_policy_work(
    lock: &LockedPalsArenaLaunchV4,
    role: NativeEngineRole,
    s: &serde_json::Value,
    t: &serde_json::Value,
) -> Result<(), ArenaError> {
    let PalsEngineV4::Pals(e) = &lock.input.semantic_lock.manifest.engines[role_index(role)] else {
        return require(
            s["pals_resolver"].is_null()
                && t["pals_resolver"].is_null()
                && s["pals_search_policy"].is_null()
                && t["pals_search_policy"].is_null(),
            "non-PALS work cannot claim a PALS policy",
        );
    };
    let (version, semantics) = match e.policies.resolver {
        PalsResolverPolicyV4::OwnRaw => (
            rz_search::pals::engine::PALS_VALUE_RESOLVER_VERSION,
            rz_search::pals::engine::PALS_VALUE_RESOLVER_SEMANTICS,
        ),
        PalsResolverPolicyV4::ModelWdl => (
            rz_search::pals::value::MODEL_WDL_RESOLVER_VERSION,
            rz_search::pals::value::MODEL_WDL_RESOLVER_SEMANTICS,
        ),
    };
    let semantic_sha: [u8; 32] = Sha256::digest(semantics.as_bytes()).into();
    require(
        e.policies.resolver_identity.semantic_id == version
            && e.policies.resolver_semantics_sha256 == hex(semantic_sha),
        "V4 resolver source registration differs",
    )?;
    for work in [s, t] {
        require(
            work["pals_resolver"]["version"] == version
                && work["pals_resolver"]["semantics_sha256"] == serde_json::json!(semantic_sha),
            "actual V4 resolver differs from locked resolver",
        )?;
        match e.policies.recheck {
            PalsRecheckPolicyV4::Disabled => require(
                work["pals_search_policy"].is_null(),
                "disabled V4 recheck cannot inherit another policy",
            )?,
            PalsRecheckPolicyV4::FrozenWdl => {
                let selected: PalsPolicyIdentityV4 =
                    serde_json::from_value(work["pals_search_policy"].clone())
                        .map_err(|e| invalid(format!("actual V4 policy marker: {e}")))?;
                require(
                    selected == e.policy_identity,
                    "actual V4 recheck registration/conditions differ",
                )?;
            }
        }
    }
    Ok(())
}
fn validate_followup_marker_pair(
    lock: &LockedPalsArenaLaunchV4,
    role: NativeEngineRole,
    s: &serde_json::Value,
    t: &serde_json::Value,
) -> Result<Option<PalsActualFollowupMarkerV4>, ArenaError> {
    let PalsEngineV4::Pals(e) = &lock.input.semantic_lock.manifest.engines[role_index(role)] else {
        require(
            s["v4"].is_null() && t["v4"].is_null(),
            "non-PALS source cannot inherit a followup model marker",
        )?;
        return Ok(None);
    };
    require(
        s["v4"].is_object() && s["v4"] == t["v4"],
        "actual V4 selection marker absent or changed",
    )?;
    let actual: PalsActualFollowupMarkerV4 = serde_json::from_value(s["v4"].clone())
        .map_err(|e| invalid(format!("V4 producer marker: {e}")))?;
    require(
        actual.resolver_policy
            == match e.policies.resolver {
                PalsResolverPolicyV4::OwnRaw => "own-raw-restricted",
                PalsResolverPolicyV4::ModelWdl => "model-wdl-restricted",
            }
            && actual.cpu_resume.as_deref()
                == if e.base.cpu_r.is_own() {
                    Some(match e.policies.paused_stack {
                        PalsPausedStackPolicyV4::Disabled => "completed-iteration",
                        _ => "paused-stack",
                    })
                } else {
                    None
                }
            && actual.post_repair_recheck == expected_recheck(e)
            && actual.policy_identity == e.policy_identity.search_identity
            && actual.policy_conditions_sha256 == Some(e.policy_identity.conditions_sha256)
            && actual.search_implementation_sha256
                == rz_search::pals::engine::compiled_search_implementation_sha256()
            && hex(actual.search_implementation_sha256) == e.base.search.implementation_sha256
            && actual.resolver_implementation_sha256
                == rz_search::pals::engine::compiled_resolver_implementation_sha256()
            && hex(actual.resolver_implementation_sha256)
                == e.policies.resolver_identity.implementation_sha256
            && hex(actual.resolver_semantics_sha256) == e.policies.resolver_semantics_sha256
            && actual.cuda_warm
                == matches!(e.policies.cuda_warm, PalsCudaWarmPolicyV4::ApproxWarm(_)),
        "actual V4 resolver/stack/recheck/conditions/CUDA Warm selection differs",
    )?;
    if lock.input.endpoints[role_index(role)]
        .native_model()
        .is_some()
    {
        let checkpoint = match &e.base.model.weights {
            PalsWeightIdentityV3::Untrained { artifact, .. }
            | PalsWeightIdentityV3::Trained { artifact, .. } => artifact,
            _ => return Err(invalid("native V4 model has no checkpoint")),
        };
        require(
            actual.model_identity == e.model_v2.model_identity
                && actual.model_profile.as_deref() == Some(config(e).profile.as_str())
                && actual.model_semantics == e.model_v2.model_semantics
                && actual.model_semantics_sha256
                    == Some(config(e).profile.registered_semantics_sha256())
                && actual
                    .model_semantics_sha256
                    .is_some_and(|sha| hex(sha) == e.model_v2.model_semantics_sha256)
                && actual.model_implementation_sha256
                    == Some(rz_eval::pals_model::compiled_model_boundary_sha256())
                && actual
                    .model_implementation_sha256
                    .is_some_and(|sha| hex(sha) == e.base.model.implementation_sha256)
                && actual.encoding_schema.as_deref() == Some(e.model_v2.encoding_schema.as_str())
                && actual.model_configuration.as_ref() == Some(&config(e))
                && actual
                    .model_epoch
                    .is_some_and(|epoch| hex(epoch) == checkpoint.sha256),
            "actual loaded V4 model/profile/encoding/checkpoint differs",
        )?;
    } else {
        require(
            actual.model_profile.is_none()
                && actual.model_configuration.is_none()
                && actual.model_epoch.is_none()
                && actual.model_semantics_sha256.is_none()
                && actual.model_implementation_sha256.is_none()
                && actual.encoding_schema.is_none(),
            "explicit legal-order mock cannot claim a loaded neural model",
        )?;
    }
    Ok(Some(actual))
}

fn verify_external_profiles(
    input: &PalsArenaLaunchV4,
    source_root: &Path,
) -> Result<(), ArenaError> {
    for role in [NativeEngineRole::Baseline, NativeEngineRole::Candidate] {
        if let Some((_, binding)) = input.external_cpu_r_binding(role_index(role))? {
            #[cfg(all(
                target_os = "linux",
                any(feature = "pals-collection-onnx", feature = "native-cuda")
            ))]
            verify_external_profile(
                &input.semantic_lock,
                role,
                source_root,
                Path::new(&binding.registered_program),
                Path::new(&binding.registered_working_directory),
            )?;
            #[cfg(not(all(
                target_os = "linux",
                any(feature = "pals-collection-onnx", feature = "native-cuda")
            )))]
            {
                let _ = (binding, source_root);
                return Err(invalid(
                    "external V4 checker profile verification requires the explicit native feature",
                ));
            }
        }
    }
    Ok(())
}
impl PalsArenaLaunchV4 {
    fn external_cpu_r_binding(
        &self,
        i: usize,
    ) -> Result<Option<(&PalsExternalCpuRV3, &PalsExternalCpuRLaunchV3)>, ArenaError> {
        let binding = self.endpoints[i]
            .native_model()
            .and_then(|native| native.external_cpu_r.as_ref());
        match &self.semantic_lock.manifest.engines[i] {
            PalsEngineV4::Pals(endpoint) => match &endpoint.base.cpu_r {
                PalsCpuRSelectionV3::Own => {
                    require(
                        binding.is_none(),
                        "own CPU_R cannot inherit an external launch binding",
                    )?;
                    Ok(None)
                }
                PalsCpuRSelectionV3::ExternalUci(declaration) => {
                    let binding = binding.ok_or_else(|| {
                        invalid("external CPU_R requires an explicit native P/C launch binding")
                    })?;
                    binding.validate()?;
                    Ok(Some((declaration, binding)))
                }
            },
            _ => {
                require(
                    binding.is_none(),
                    "non-PALS endpoint cannot inherit an external CPU_R launch binding",
                )?;
                Ok(None)
            }
        }
    }
    pub fn from_json(text: &str) -> Result<Self, ArenaError> {
        require(text.len() <= MAX_JSON_BYTES, "launch JSON budget exceeded")?;
        let input: Self = crate::decode_json(text)?;
        input.validate()?;
        Ok(input)
    }
    pub fn validate(&self) -> Result<(), ArenaError> {
        self.semantic_lock.verify()?;
        require(self.domain == PALS_ARENA_V4_DOMAIN, "wrong V4 arena domain")?;
        validate_cudnn_conv_pair(&self.endpoints)?;
        resources::validate_profile(self)?;
        let m = &self.semantic_lock.manifest;
        require(
            m.purpose == PalsRunPurposeV4::ArenaPilot,
            "diagnostic declarations cannot launch arena",
        )?;
        require(
            m.pilot.max_plies.is_multiple_of(2),
            "paired runner requires even max ply count",
        )?;
        self.require_reap_status_runner()?;
        self.runner.binary.validate()?;
        self.opening_artifact.validate()?;
        for a in self.declared_artifacts() {
            a.validate()?;
            require_public_artifact(a)?;
        }
        let b = self.budget;
        let unique = self.unique_input_bytes()?;
        let output_cap = self.validate_followup_execution()?;
        require(
            unique <= b.max_input_bytes
                && b.max_input_bytes > 0
                && b.max_output_bytes > V4_METADATA_CAP
                && b.max_output_bytes <= output_cap
                && b.max_output_bytes <= m.output_bytes_max
                && b.max_runtime_bytes > 0
                && b.max_runtime_bytes <= 10 * 1024 * 1024 * 1024
                && (3..=16).contains(&b.max_child_processes)
                && (16..=4096).contains(&b.max_runtime_files)
                && (2..=8).contains(&b.max_runtime_depth)
                && b.address_space_per_process_bytes == 0
                && unique
                    .checked_add(b.max_output_bytes)
                    .and_then(|x| x.checked_add(b.max_runtime_bytes))
                    .is_some_and(|x| x <= b.max_artifact_bytes),
            "V4 finite input/output/runtime/process budget invalid",
        )?;
        let a = &m.resources[0];
        let c = &m.resources[1];
        require(
            a.cpu_affinity == c.cpu_affinity
                && a.memory_high_bytes == c.memory_high_bytes
                && a.memory_max_bytes == c.memory_max_bytes
                && a.swap_max_bytes == c.swap_max_bytes,
            "V4 sequential engines require one actual inherited resource policy",
        )?;
        for i in 0..2 {
            self.endpoint(i)?;
            if let PalsEngineV4::Pals(e) = &m.engines[i] {
                policy_arguments(e, self.execution_binding(i))?;
                require(
                    e.model_v2.arena_candidate(),
                    "diagnostic checkpoint/export cannot enter arena",
                )?;
                require(
                    e.base.pools.host_bytes == m.resources[i].memory_max_bytes,
                    "V4 host pool differs from inherited memory cap",
                )?;
                if let Some(n) = self.endpoints[i].native_model() {
                    require(
                        n.search.post_repair_recheck.is_none(),
                        "V4 typed policy cannot carry a legacy recheck selector",
                    )?;
                    require(
                        e.base.model.frozen_epoch == 1,
                        "V4 production model requires frozen epoch 1",
                    )?;
                    validate_actual_encoding(e, n)?;
                }
                if let Some(cuda) = self.endpoints[i].cuda_model() {
                    require(
                        cuda.cuda_record_pages.is_none(),
                        "V2 full-line model does not register legacy resident packing",
                    )?;
                    require(
                        m.resources[i].requested_gpu.is_some()
                            && e.base.pools.device_bytes
                                == m.resources[i].device_allocation_max_bytes
                            && cuda
                                .session_arena_bytes
                                .checked_mul(cuda.model.graphs.len() as u64)
                                .is_some_and(|n| n < e.base.pools.device_bytes),
                        "V4 CUDA device/arena reservation differs",
                    )?;
                } else {
                    require(
                        m.resources[i].requested_gpu.is_none() && e.base.pools.device_bytes == 0,
                        "CPU/mock V4 cannot silently inherit CUDA",
                    )?;
                }
            }
            if self.external_cpu_r_binding(i)?.is_some() {
                require(
                    cfg!(all(
                        target_os = "linux",
                        any(feature = "pals-collection-onnx", feature = "native-cuda")
                    )),
                    "external CPU_R needs actual native feature; no fallback",
                )?;
            }
        }
        let mut names = BTreeMap::new();
        for (a, name) in self.named_assets() {
            if let Some(prior) = names.insert(name, a) {
                require(prior == a, "V4 snapshot destination collision")?;
            }
        }
        Ok(())
    }
    pub fn lock(&self) -> Result<LockedPalsArenaLaunchV4, ArenaError> {
        self.validate()?;
        Ok(LockedPalsArenaLaunchV4 {
            input: self.clone(),
            sha256: crate::canonical_sha256(self)?,
            opening: self.opening(),
            endpoint_views: [self.endpoint(0)?, self.endpoint(1)?],
            #[cfg(target_os = "linux")]
            archive_roots: Default::default(),
            #[cfg(target_os = "linux")]
            resource_observations: Default::default(),
        })
    }
    pub fn runner_status_patch(&self) -> Result<PalsRunnerStatusPatchV3, ArenaError> {
        let patch = self
            .runner
            .dirty_patch
            .as_ref()
            .ok_or_else(|| invalid("runner patch absent"))?;
        match (patch.sha256.as_str(), patch.bytes) {
            (
                rz_experiments::FASTCHESS_CLOCK_PATCH_SHA256,
                rz_experiments::FASTCHESS_CLOCK_PATCH_BYTES,
            ) => Ok(PalsRunnerStatusPatchV3::LegacyClockOnly),
            (PALS_CLOCK_REAP_PATCH_SHA256, PALS_CLOCK_REAP_PATCH_BYTES) => {
                Ok(PalsRunnerStatusPatchV3::ClockAndReapStatus)
            }
            _ => Err(invalid("runner patch hash/bytes not registered")),
        }
    }
    fn require_reap_status_runner(&self) -> Result<(), ArenaError> {
        require(
            self.runner_status_patch()? == PalsRunnerStatusPatchV3::ClockAndReapStatus
                && registered_reap_runner(
                    &self.runner.version,
                    &self.runner.binary.sha256,
                    self.runner.binary.bytes,
                ),
            "new PALS execution/Core requires the registered clock+reap-status patch and actual runner binary; legacy evidence is preserved without promotion",
        )
    }
    #[cfg(target_os = "linux")]
    fn require_actual_native_epoch(&self) -> Result<(), ArenaError> {
        for (engine, recipe) in self
            .semantic_lock
            .manifest
            .engines
            .iter()
            .zip(&self.endpoints)
        {
            if recipe.native_model().is_some() {
                require(
                    matches!(engine, PalsEngineV4::Pals(e) if e.base.model.frozen_epoch == 1),
                    "new native PALS execution/Core requires deployment frozen epoch 1; historical epoch 0 locks remain unchanged",
                )?;
            }
        }
        Ok(())
    }
    fn opening(&self) -> OpeningSpec {
        OpeningSpec {
            id: "pals-standard-start".into(),
            initial: InitialPosition::Startpos,
            fen: None,
            moves: vec![],
            history: HistoryCompleteness::Complete,
            history_origin: "startpos-complete-trace".into(),
        }
    }
    fn named_assets(&self) -> Vec<(&ArtifactRef, String)> {
        let mut result = Vec::new();
        for engine in &self.semantic_lock.manifest.engines {
            if let PalsEngineV4::Pals(endpoint) = engine
                && let PalsCpuRSelectionV3::ExternalUci(helper) = &endpoint.base.cpu_r
            {
                result.push((
                    &helper.profile,
                    format!("pals-helper-profile-{}/profile.json", helper.profile.sha256),
                ));
                result.push((
                    &helper.binary,
                    format!("pals-helper-binary-{}/binary", helper.binary.sha256),
                ));
            }
        }
        for recipe in &self.endpoints {
            if let Some(n) = recipe.native_model() {
                let namespace = format!("pals-{}", n.export.sha256);
                result.push((&n.export, format!("{namespace}/{}", n.export_file)));
                result.extend(
                    n.graphs
                        .iter()
                        .map(|g| (&g.artifact, format!("{namespace}/{}", g.file))),
                );
            }
            if let Some(control) = recipe
                .cuda_model()
                .and_then(|cuda| cuda.cuda_control.as_ref())
            {
                result.push((
                    &control.inventory,
                    format!(
                        "pals-control-{}/inventory.v2.json",
                        control.inventory.sha256
                    ),
                ));
            }
            if let Some(pages) = recipe
                .cuda_model()
                .and_then(|cuda| cuda.cuda_record_pages.as_ref())
            {
                let namespace = format!("pals-packing-{}", pages.manifest.sha256);
                result.push((&pages.manifest, format!("{namespace}/device-packing.json")));
                result.push((&pages.graph, format!("{namespace}/device_public_pack.onnx")));
            }
        }
        result
    }
    pub fn declared_artifacts(&self) -> Vec<&ArtifactRef> {
        let mut result = vec![&self.runner.binary, &self.opening_artifact];
        if let Some(p) = &self.runner.dirty_patch {
            result.push(p);
        }
        for (engine, recipe) in self
            .semantic_lock
            .manifest
            .engines
            .iter()
            .zip(&self.endpoints)
        {
            match engine {
                PalsEngineV4::Pals(e) => {
                    let e = &e.base;
                    result.push(&e.binary);
                    if let PalsCpuRSelectionV3::ExternalUci(helper) = &e.cpu_r {
                        result.extend([&helper.profile, &helper.binary]);
                    }
                    if let PalsWeightIdentityV3::Untrained { artifact, .. }
                    | PalsWeightIdentityV3::Trained { artifact, .. } = &e.model.weights
                    {
                        result.push(artifact);
                    }
                }
                PalsEngineV4::OwnCpu(e) => result.push(&e.binary),
                PalsEngineV4::ReferenceUci(e) => {
                    result.push(&e.binary);
                    result.extend(&e.assets);
                }
            }
            match recipe {
                PalsEndpointLaunchV3::OnnxCpu(n) => {
                    result.extend([&n.export, &n.runtime]);
                    result.extend(n.graphs.iter().map(|g| &g.artifact));
                }
                PalsEndpointLaunchV3::OnnxCuda(cuda) => {
                    let n = &cuda.model;
                    result.extend([&n.export, &n.runtime, &cuda.cuda_bundle.manifest]);
                    result.extend(n.graphs.iter().map(|g| &g.artifact));
                    result.extend(cuda.cuda_bundle.files.iter().map(|f| &f.artifact));
                    if let Some(control) = &cuda.cuda_control {
                        result.push(&control.inventory);
                    }
                    if let Some(pages) = &cuda.cuda_record_pages {
                        result.extend([&pages.manifest, &pages.graph]);
                    }
                }
                PalsEndpointLaunchV3::ReferenceUci {
                    environment: Some(e),
                    ..
                } => result.push(&e.launcher),
                _ => {}
            }
        }
        result
    }
    fn unique_input_bytes(&self) -> Result<u64, ManifestError> {
        let mut by_path = BTreeMap::new();
        let mut total = 0u64;
        for a in self.declared_artifacts() {
            a.validate()?;
            require_public_artifact(a).map_err(|e| ManifestError::Integrity(e.to_string()))?;
            if let Some(prior) = by_path.insert(&a.path, a) {
                if prior != a {
                    return Err(ManifestError::Integrity(
                        "PALS path has inconsistent artifact identity".into(),
                    ));
                }
            } else {
                total = total
                    .checked_add(a.bytes)
                    .ok_or_else(|| ManifestError::Integrity("PALS input byte overflow".into()))?;
            }
        }
        Ok(total)
    }
    fn endpoint(&self, i: usize) -> Result<ExternalUciEndpointV2, ArenaError> {
        let helper = self.external_cpu_r_binding(i)?;
        let m = &self.semantic_lock.manifest;
        let role = if i == 0 {
            NativeEngineRole::Baseline
        } else {
            NativeEngineRole::Candidate
        };

        let (binary, family, version, uci_name, source, mut arguments, assets, environment) = match (
            &m.engines[i],
            &self.endpoints[i],
        ) {
            (PalsEngineV4::Pals(v4), PalsEndpointLaunchV3::LegalOrderMock(s)) => {
                let e = &v4.base;
                require(
                    e.model.backend == PalsModelBackendV3::DeterministicMock
                        && matches!(
                            e.model.weights,
                            PalsWeightIdentityV3::DeterministicMock { seed: 0 }
                        )
                        && e.model.architecture == "explicit-legal-order-role-mock-v1"
                        && e.model.precision == PalsPrecisionV3::Fp32
                        && e.model.max_batch_width == 1
                        && e.model.frozen_epoch == 0,
                    "explicit legal-order mock identity/seed/epoch differs",
                )?;
                validate_search(e, s, m.pilot.wall_time_max_ms)?;
                (
                    &e.binary,
                    "rovezero-pals",
                    "pals/0.1",
                    "RoveZero PALS explicit legal-order CPU mock + own CPU_R".to_string(),
                    Some(ExternalSourceV2 {
                        url: e.binary.source.clone(),
                        commit: e.source_commit.clone(),
                        license: e.binary.license.clone(),
                    }),
                    pals_search_arguments(s, "legal-order-mock"),
                    vec![],
                    None,
                )
            }
            (
                PalsEngineV4::Pals(e),
                recipe @ (PalsEndpointLaunchV3::OnnxCpu(_) | PalsEndpointLaunchV3::OnnxCuda(_)),
            ) => {
                let v4 = e;
                let e = &v4.base;
                let n = recipe.native_model().expect("native pattern selected");
                let cuda = recipe.cuda_model();
                let shutdown_ms = m.pilot.cleanup_max_ms.div_ceil(2);
                require(
                    (1..=30_000).contains(&shutdown_ms),
                    "PALS native shutdown requires half the finite cleanup reservation within 1..30000 milliseconds",
                )?;
                require(
                    e.model.backend
                        == if cuda.is_some() {
                            PalsModelBackendV3::OrtCuda
                        } else {
                            PalsModelBackendV3::OrtCpu
                        }
                        && e.model.precision == PalsPrecisionV3::Fp32
                        && e.model.max_batch_width == 1
                        && matches!(e.model.frozen_epoch, 0 | 1),
                    "native metadata requires explicit provider, historical/deployment frozen epoch 0/1 and FP32 B1",
                )?;
                require(
                    e.model.architecture == "pals-width384-latent16-iterations2"
                        && e.model.input_schema == v4.model_v2.encoding_schema
                        && e.model.policy_head == "candidate-policy/1"
                        && e.model.value_head == "stm-wdl/1"
                        && e.model.implementation_sha256
                            == hex(rz_eval::pals_model::compiled_model_boundary_sha256()),
                    "native model topology/input/head/compiled model boundary declaration differs from supported recipe",
                )?;
                require(
                    m.resources[i].cpu_threads == 2,
                    "native CPU model uses two registered ORT intra-op threads",
                )?;
                require(
                    matches!(
                        e.model.weights,
                        PalsWeightIdentityV3::Untrained { .. }
                            | PalsWeightIdentityV3::Trained { .. }
                    ),
                    "native P/C requires actual checkpoint identity",
                )?;
                require(
                    hash(&n.encoding_semantic_sha256)
                        && hash(&n.adapter_source_sha256)
                        && name(&n.export_file)
                        && n.export_file.ends_with(".json")
                        && n.export.bytes <= 64 * 1024
                        && matches!(n.graphs.len(), 2 | 3),
                    "P/C export/adapter identities invalid",
                )?;
                let roles: BTreeSet<_> = n.graphs.iter().map(|g| g.role.as_str()).collect();
                require(
                    (roles == BTreeSet::from(["public", "proposer", "critic"])
                        || roles == BTreeSet::from(["public", "shared_pc"]))
                        && n.graphs.iter().all(|g| {
                            name(&g.file) && g.file.ends_with(".onnx") && g.file != n.export_file
                        }),
                    "only exact public/P/C graphs may execute; V is excluded",
                )?;
                require(
                    n.graphs
                        .iter()
                        .map(|g| &g.file)
                        .collect::<BTreeSet<_>>()
                        .len()
                        == n.graphs.len(),
                    "duplicate graph filename",
                )?;
                validate_search(e, &n.search, m.pilot.wall_time_max_ms)?;
                let mut args = pals_search_arguments(&n.search, "onnx");
                args.extend([
                    format!("--pals-shutdown-timeout-ms={shutdown_ms}"),
                    format!(
                        "--pals-provider={}",
                        if cuda.is_some() { "cuda" } else { "cpu" }
                    ),
                    "--pals-export-manifest={{asset:0}}".into(),
                    format!("--pals-export-sha256={}", n.export.sha256),
                    "--pals-runtime-path={{asset:1}}".into(),
                    format!("--pals-runtime-sha256={}", n.runtime.sha256),
                ]);
                let mut assets = vec![n.export.clone(), n.runtime.clone()];
                assets.extend(n.graphs.iter().map(|g| g.artifact.clone()));
                if let Some(cuda) = cuda {
                    cuda.cuda_bundle.validate()?;
                    if let Some(probe_ms) = cuda.startup_probe_timeout_ms {
                        require(
                            (1..=MAX_STARTUP_PROBE_TIMEOUT_MS).contains(&probe_ms)
                                && probe_ms <= m.pilot.handshake_max_ms,
                            "explicit CUDA startup probe requires 1..180000ms within the cold-model readiness window",
                        )?;
                    }
                    require(
                        cuda.device_id == 0
                            && cuda.session_arena_bytes == 2 * 1024 * 1024 * 1024
                            && roles == BTreeSet::from(["public", "shared_pc"]),
                        "first 6GiB CUDA recipe requires explicit device 0, per-session 2GiB and two shared-P/C graphs with room for tensor reservations",
                    )?;
                    require(
                        cuda.cuda_bundle.file(
                            rz_experiments::CudaBundleFileRoleV1::Core,
                            "libonnxruntime.so.1.22.0",
                        )? == &n.runtime,
                        "CUDA bundle core identity differs from declared runtime",
                    )?;
                    let index = assets.len();
                    assets.push(cuda.cuda_bundle.manifest.clone());
                    assets.extend(
                        cuda.cuda_bundle
                            .files
                            .iter()
                            .filter(|f| f.artifact != n.runtime)
                            .map(|f| f.artifact.clone()),
                    );
                    args.extend([
                        format!("--pals-cuda-bundle={{{{asset:{index}}}}}"),
                        format!(
                            "--pals-cuda-bundle-sha256={}",
                            cuda.cuda_bundle.manifest.sha256
                        ),
                        format!("--pals-cuda-device={}", cuda.device_id),
                        format!(
                            "--pals-cuda-session-arena-bytes={}",
                            cuda.session_arena_bytes
                        ),
                        format!(
                            "--pals-device-public-memory={}",
                            matches!(v4.policies.cuda_warm, PalsCudaWarmPolicyV4::ApproxWarm(_))
                        ),
                    ]);
                    if let Some(control) = &cuda.cuda_control {
                        control.inventory.validate()?;
                        require_public_artifact(&control.inventory)?;
                        require(
                            (1..=MAX_CONTROL_INVENTORY_BYTES).contains(&control.inventory.bytes),
                            "CUDA control inventory must have a finite one-MiB input bound",
                        )?;
                        let index = assets.len();
                        assets.push(control.inventory.clone());
                        args.extend([
                            "--pals-cuda-control-mode=inventory-v2".into(),
                            format!("--pals-cuda-control-inventory={{{{asset:{index}}}}}"),
                            format!(
                                "--pals-cuda-control-inventory-sha256={}",
                                control.inventory.sha256
                            ),
                        ]);
                        if let Some(loading) = &control.loading_profile {
                            loading.validate()?;
                            args.extend([
                                format!(
                                    "--pals-cuda-loading-profile={}",
                                    loading.profile.native().identifier()
                                ),
                                format!(
                                    "--pals-cuda-loading-profile-sha256={}",
                                    loading.canonical_sha256
                                ),
                            ]);
                        }
                        if let Some(conv) = &control.cudnn_conv_search {
                            require(
                                m.engines.iter().all(|e| matches!(e, PalsEngineV4::Pals(_)))
                                    && matches!(
                                        v4.policies.cuda_warm,
                                        PalsCudaWarmPolicyV4::Disabled
                                    )
                                    && self.execution_binding(i).is_none(),
                                "DEFAULT host runtime candidate requires the two-PALS V4 pilot with no Warm/followup mode",
                            )?;
                            let checkpoint = match &e.model.weights {
                                PalsWeightIdentityV3::Untrained { artifact, .. }
                                | PalsWeightIdentityV3::Trained { artifact, .. } => {
                                    &artifact.sha256
                                }
                                _ => {
                                    return Err(invalid(
                                        "DEFAULT host requires an actual checkpoint",
                                    ));
                                }
                            };
                            conv.validate(cuda, checkpoint)?;
                            args.push("--pals-cudnn-conv-search=default-host-control-v1".into());
                        }
                    }
                    if let Some(probe_ms) = cuda.startup_probe_timeout_ms {
                        args.push(format!("--pals-startup-probe-timeout-ms={probe_ms}"));
                    }
                    if let Some(pages) = &cuda.cuda_record_pages {
                        pages.validate(cuda)?;
                        require(
                            helper.is_none(),
                            "resident CUDA cannot launch an external helper",
                        )?;
                        let manifest_index = assets.len();
                        assets.extend([pages.manifest.clone(), pages.graph.clone()]);
                        args.extend([
                            "--pals-cuda-record-pages=registered-packing-v1".into(),
                            format!("--pals-packing-manifest={{{{asset:{manifest_index}}}}}"),
                            format!("--pals-packing-manifest-sha256={}", pages.manifest.sha256),
                            format!("--pals-packing-graph={{{{asset:{}}}}}", manifest_index + 1),
                            format!("--pals-packing-graph-sha256={}", pages.graph.sha256),
                            format!(
                                "--pals-cuda-record-pages-resources={}",
                                pages.resources_json()?
                            ),
                        ]);
                    }
                }
                if let Some((helper, _)) = helper {
                    let index = assets.len();
                    assets.extend([helper.profile.clone(), helper.binary.clone()]);
                    args.extend([
                        "--pals-cpu-checker=external-uci".into(),
                        format!("--pals-cpu-profile={{{{asset:{index}}}}}"),
                        format!("--pals-cpu-profile-sha256={}", helper.profile.sha256),
                    ]);
                }
                (
                    &e.binary,
                    "rovezero-pals",
                    "pals/0.1",
                    format!(
                        "RoveZero PALS P/C ONNX {} + {} CPU_R",
                        if cuda.is_some() { "CUDA" } else { "CPU" },
                        if helper.is_some() {
                            "external UCI"
                        } else {
                            "own"
                        }
                    ),
                    Some(ExternalSourceV2 {
                        url: e.binary.source.clone(),
                        commit: e.source_commit.clone(),
                        license: e.binary.license.clone(),
                    }),
                    args,
                    assets,
                    None,
                )
            }
            (
                PalsEngineV4::OwnCpu(e),
                PalsEndpointLaunchV3::OwnCpu {
                    max_depth,
                    max_nodes,
                    tt_entries,
                },
            ) => {
                require(
                    *max_depth > 0
                        && *max_depth <= 64
                        && u32::from(*max_depth) <= e.cpu.max_depth
                        && *max_nodes > 0
                        && *max_nodes <= e.cpu.max_nodes_per_task
                        && (1..=1_048_576).contains(tt_entries),
                    "own CPU executable limits exceed declaration",
                )?;
                validate_cpu_profile(&e.cpu, m.pilot.wall_time_max_ms)?;
                let config = rz_search::cpu::CpuConfig {
                    max_depth: *max_depth,
                    tt_entries: *tt_entries as usize,
                    ..Default::default()
                };
                require(
                    config
                        .tt_allocation_bytes()
                        .map_err(|e| invalid(e.to_string()))?
                        <= e.cpu.max_tt_bytes,
                    "own CPU TT slot allocation exceeds declared byte limit",
                )?;
                require(
                    e.requested_options.is_empty(),
                    "own CPU closed recipe accepts no unverified UCI overrides",
                )?;
                let args = vec![
                    "--search=cpu".into(),
                    format!("--cpu-max-depth={max_depth}"),
                    format!("--cpu-max-nodes={max_nodes}"),
                    format!("--cpu-tt-entries={tt_entries}"),
                ];
                (
                    &e.binary,
                    "rovezero-own-cpu",
                    "rz-cpu-pvs/0.1",
                    "RoveZero own Rust CPU_R".to_string(),
                    Some(ExternalSourceV2 {
                        url: e.binary.source.clone(),
                        commit: e.source_commit.clone(),
                        license: e.binary.license.clone(),
                    }),
                    args,
                    vec![],
                    None,
                )
            }
            (
                PalsEngineV4::ReferenceUci(e),
                PalsEndpointLaunchV3::ReferenceUci {
                    expected_uci_name,
                    arguments,
                    environment,
                },
            ) => {
                require(
                    !expected_uci_name.is_empty()
                        && expected_uci_name.len() <= 256
                        && !expected_uci_name.chars().any(char::is_control),
                    "reference UCI identity invalid",
                )?;
                (
                    &e.binary,
                    e.family.as_str(),
                    e.version.as_str(),
                    expected_uci_name.clone(),
                    Some(ExternalSourceV2 {
                        url: e.source_url.clone(),
                        commit: e.source_commit.clone(),
                        license: e.binary.license.clone(),
                    }),
                    arguments.clone(),
                    e.assets.clone(),
                    environment.clone(),
                )
            }
            _ => return Err(invalid("semantic endpoint and executable recipe differ")),
        };
        if let PalsEngineV4::Pals(e) = &m.engines[i] {
            arguments.extend(policy_arguments(e, self.execution_binding(i))?);
        } else if matches!(self.endpoints[i], PalsEndpointLaunchV3::OwnCpu { .. }) {
            arguments.push("--pals-arena-wire=v4".into());
        }
        require(
            arguments.len() <= 48
                && arguments
                    .iter()
                    .all(|a| !a.is_empty() && a.len() <= 4096 && !a.chars().any(char::is_control)),
            "executable argv budget/injection violation",
        )?;
        Ok(ExternalUciEndpointV2 {
            id: m.engines[i].id().into(),
            role,
            family: family.into(),
            version: version.into(),
            expected_uci_name: uci_name,
            binary: binary.clone(),
            source,
            arguments,
            assets,
            requested_options: requested_options(&m.engines[i]).clone(),
            environment,
            handshake_timeout_ms: m.pilot.handshake_max_ms,
        })
    }
}
fn validate_search(
    e: &rz_experiments::PalsEndpointV3,
    s: &PalsSearchLaunchV3,
    wall: u64,
) -> Result<(), ArenaError> {
    require(
        s.post_repair_recheck.is_none(),
        "V4 executable policy is separately typed",
    )?;
    let (max_depth, max_nodes_per_task) = match &e.cpu_r {
        PalsCpuRSelectionV3::Own => {
            validate_cpu_profile(&e.cpu, wall)?;
            require(
                rz_search::cpu::CpuConfig::default()
                    .tt_allocation_bytes()
                    .map_err(|e| invalid(e.to_string()))?
                    <= e.cpu.max_tt_bytes,
                "PALS own CPU default TT slot allocation exceeds declared byte limit",
            )?;
            (e.cpu.max_depth, e.cpu.max_nodes_per_task)
        }
        PalsCpuRSelectionV3::ExternalUci(helper) => {
            // CPU_T/legacy own declarations remain in the lock; they are not
            // runtime authority for the explicitly selected foreign CPU_R.
            (helper.policy.max_depth, helper.policy.max_nodes_per_task)
        }
    };
    require(
        (1..=1_000_000).contains(&s.max_rounds)
            && s.max_cpu_nodes > 0
            && s.cpu_depth > 0
            && s.cpu_depth <= 16
            && u32::from(s.cpu_depth) <= max_depth
            && (257..=65_536).contains(&s.max_situations)
            && max_nodes_per_task
                >= if e.cpu_r.is_own() {
                    4096
                } else {
                    rz_search::pals::engine::PalsConfig::default()
                        .cpu_nodes_per_task
                        .min(s.max_cpu_nodes)
                },
        "PALS executable work/profile bounds invalid",
    )?;
    let p = &e.pools;
    let n = s.max_situations;
    require(
        p.states >= n
            && p.situations >= n
            && p.line_chunks >= n * 16
            && p.observations >= n * 16
            && p.tasks >= n * 8
            && p.role_states >= 1
            && p.memory_pages >= 1
            && p.queue_requests >= 1,
        "declared pool bounds do not cover the closed PALS executable recipe",
    )?;
    require(
        e.runtime.semantic_id == "single-owner/1" && e.runtime.options.is_empty(),
        "unsupported PALS search/runtime semantic option override",
    )?;
    require(
        e.requested_options.is_empty(),
        "PALS fixed executable recipe accepts no unverified UCI overrides",
    )
}
impl LockedPalsArenaLaunchV4 {
    pub fn input(&self) -> &PalsArenaLaunchV4 {
        &self.input
    }
    pub fn sha256(&self) -> &str {
        &self.sha256
    }
    pub fn to_json(&self) -> Result<String, ArenaError> {
        serde_json::to_string_pretty(&serde_json::json!({"domain":PALS_ARENA_V4_DOMAIN,"sha256":self.sha256,"input":self.input})).map_err(|e|invalid(e.to_string()))
    }
    pub fn from_json(text: &str) -> Result<Self, ArenaError> {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Envelope {
            domain: String,
            sha256: String,
            input: PalsArenaLaunchV4,
        }
        require(
            text.len() <= MAX_JSON_BYTES,
            "locked launch JSON budget exceeded",
        )?;
        let envelope: Envelope = crate::decode_json(text)?;
        let lock = envelope.input.lock()?;
        require(
            envelope.domain == PALS_ARENA_V4_DOMAIN && envelope.sha256 == lock.sha256,
            "arena lock digest/domain differs",
        )?;
        Ok(lock)
    }
    fn native(&self, role: NativeEngineRole) -> Result<&PalsOnnxLaunchV3, ArenaError> {
        self.input.endpoints[role_index(role)]
            .native_model()
            .ok_or_else(|| invalid("endpoint has no native P/C session"))
    }
    fn validate_export(&self, role: NativeEngineRole, bytes: &[u8]) -> Result<(), ArenaError> {
        require(bytes.len() <= 128 * 1024, "P/C export JSON exceeds bound")?;
        let n = self.native(role)?;
        require(digest(bytes) == n.export.sha256, "P/C export bytes differ")?;
        let value = json(bytes)?;
        let PalsEngineV4::Pals(e) = &self.input.semantic_lock.manifest.engines[role_index(role)]
        else {
            return Err(invalid("native recipe has no PALS identity"));
        };
        let v4 = e;
        let e = &v4.base;
        require(
            v4.model_v2.model_semantics_sha256
                == hex(config(v4).profile.registered_semantics_sha256())
                && e.model.implementation_sha256
                    == hex(rz_eval::pals_model::compiled_model_boundary_sha256()),
            "V4 model semantic/source registration differs from compiled boundary",
        )?;
        let (weights, trained) = match &e.model.weights {
            PalsWeightIdentityV3::Untrained { artifact, .. } => (artifact, false),
            PalsWeightIdentityV3::Trained { artifact, .. } => (artifact, true),
            _ => return Err(invalid("no native checkpoint identity")),
        };
        let shared = n.graphs.iter().any(|g| g.role == "shared_pc");
        let warm = self
            .input
            .execution_binding(role_index(role))
            .and_then(|b| b.cuda_warm.as_ref());
        let layout_valid = if let Some(warm) = warm {
            shared
                && value["schema"] == warm.export_schema
                && value["layout"] == warm.layout
                && value["layout_revision"] == warm.layout_revision
                && value["query_semantics"] == warm.query_semantics
                && value["execution_domain"] == warm.execution_domain
                && value["model_semantics"] == v4.model_v2.model_semantics
                && value["batch_mode"] == "one_scalar_role_and_one_scalar_mode_per_physical_batch"
        } else if shared {
            value["schema"] == "rovezero.pals-model.v2"
                && value["layout"] == "shared_pc_if"
                && value["layout_revision"] == 1
                && value["model_semantics"] == v4.model_v2.model_semantics
                && value["role_batching"] == "one_scalar_role_per_physical_batch"
        } else {
            value["schema"] == "rovezero.pals-model.v1"
                && (value["layout"].is_null() || value["layout"] == "separate_pc")
        };
        require(
            layout_valid
                && value["schema"] == v4.model_v2.model_schema
                && value["config"]
                    == serde_json::to_value(config(v4)).map_err(|e| invalid(e.to_string()))?
                && (config(v4).profile.is_legacy()
                    || (value["model_profile"] == config(v4).profile.as_str()
                        && value["encoding_schema"] == v4.model_v2.encoding_schema))
                && value["artifact_purpose"] != "diagnostic"
                && value["diagnostic"] != true
                && value["checkpoint_domain"] != PALS_V4_NONZERO_CHECKPOINT_DOMAIN
                && value["checkpoint_sha256"] == weights.sha256
                && value["trained"] == trained
                && value["validator_present"] == false
                && value["roles"] == serde_json::json!(["proposer", "critic"]),
            "P/C export model/checkpoint/V-free identity differs",
        )?;
        let graphs = value["graphs"]
            .as_array()
            .ok_or_else(|| invalid("missing P/C graph list"))?;
        require(graphs.len() == n.graphs.len(), "export graph count differs")?;
        for declared in &n.graphs {
            require(
                graphs
                    .iter()
                    .filter(|g| {
                        g["role"] == declared.role
                            && g["file"] == declared.file
                            && g["sha256"] == declared.artifact.sha256
                    })
                    .count()
                    == 1,
                "export graph identity differs from pinned input",
            )?;
        }
        Ok(())
    }
    fn validate_cuda_bundle(&self, role: NativeEngineRole, bytes: &[u8]) -> Result<(), ArenaError> {
        let cuda = self.input.endpoints[role_index(role)]
            .cuda_model()
            .ok_or_else(|| invalid("endpoint has no CUDA bundle"))?;
        require(
            bytes.len() <= 64 * 1024 && digest(bytes) == cuda.cuda_bundle.manifest.sha256,
            "CUDA descriptor byte/hash differs",
        )?;
        let parsed = rz_eval::runtime_pin::CudaRuntimeBundleSpec::from_json(
            std::str::from_utf8(bytes).map_err(|_| invalid("CUDA descriptor not UTF-8"))?,
        )
        .map_err(|e| invalid(format!("CUDA descriptor rejected: {:?}", e.kind)))?;
        let actual = parsed
            .digest()
            .map_err(|e| invalid(format!("CUDA descriptor digest rejected: {:?}", e.kind)))?;
        require(
            actual
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect::<String>()
                == cuda.cuda_bundle.canonical_sha256,
            "CUDA descriptor canonical digest differs from pinned nineteen-file identity",
        )?;
        Ok(())
    }
    fn validate_control_inventory(
        &self,
        role: NativeEngineRole,
        bytes: &[u8],
    ) -> Result<(), ArenaError> {
        let cuda = self.input.endpoints[role_index(role)]
            .cuda_model()
            .ok_or_else(|| invalid("endpoint has no CUDA recipe"))?;
        let control = cuda
            .cuda_control
            .as_ref()
            .ok_or_else(|| invalid("endpoint has no registered control inventory"))?;
        require(
            bytes.len() as u64 == control.inventory.bytes
                && bytes.len() as u64 <= MAX_CONTROL_INVENTORY_BYTES
                && digest(bytes) == control.inventory.sha256,
            "control inventory byte/hash differs",
        )?;
        let value = json(bytes)?;
        require(
            value["schema"] == "rovezero.pals-static-control-inventory.v2"
                && value["scope"] == "read_only_serialized_graph_metadata_no_runtime"
                && value["manifest_sha256"] == cuda.model.export.sha256
                && value["actual_provider_placement"] == "not_observed"
                && value["cpu_allowlist"] == "not_created"
                && (1..=5000).contains(&count(&value, "total_nodes")?),
            "registered control inventory metadata differs",
        )?;
        let graphs = value["graphs"]
            .as_array()
            .ok_or_else(|| invalid("control inventory graph list missing"))?;
        require(
            graphs.len() == 2
                && cuda.model.graphs.iter().all(|declared| {
                    graphs
                        .iter()
                        .filter(|g| {
                            g["role"] == declared.role && g["sha256"] == declared.artifact.sha256
                        })
                        .count()
                        == 1
                }),
            "control inventory is not the pinned public/shared-P/C graph pair",
        )
    }
    /// Only the V4 own DEFAULT lane requires native probe receipts; legacy
    /// implicit CUDA and the shared external-helper selector remain unchanged.
    fn uses_owned_default_preflight(&self, role: NativeEngineRole) -> bool {
        matches!(
            &self.input.semantic_lock.manifest.engines[role_index(role)],
            PalsEngineV4::Pals(e) if e.base.cpu_r.is_own()
        ) && self.input.endpoints[role_index(role)]
            .cuda_model()
            .and_then(cudnn_conv_binding)
            .is_some()
    }
    fn cuda_profile_argument(
        &self,
        role: NativeEngineRole,
        runtime_root: &Path,
    ) -> Result<Option<OsString>, ArenaError> {
        if self.input.endpoints[role_index(role)]
            .cuda_model()
            .and_then(|cuda| cuda.cuda_control.as_ref())
            .is_none()
        {
            return Ok(None);
        }
        let profile_parent = if self.uses_owned_default_preflight(role) {
            let (game, preflight) = match role {
                NativeEngineRole::Baseline => {
                    ("baseline-runtime", "external-baseline-preflight-runtime")
                }
                NativeEngineRole::Candidate => {
                    ("candidate-runtime", "external-candidate-preflight-runtime")
                }
            };
            match runtime_root.file_name().and_then(|name| name.to_str()) {
                Some(name) if name == preflight => runtime_root.to_path_buf(),
                Some(name) if name == game => runtime_root
                    .parent()
                    .ok_or_else(|| invalid("DEFAULT owner runtime parent absent"))?
                    .join(preflight),
                _ => {
                    return Err(invalid(
                        "DEFAULT profile parent must be the owner-selected role root",
                    ));
                }
            }
        } else {
            runtime_root.to_path_buf()
        };
        let path = profile_parent
            .to_str()
            .ok_or_else(|| invalid("CUDA profile parent is not UTF-8"))?;
        require(
            runtime_root.is_absolute()
                && profile_parent.is_absolute()
                && path.len() <= 4096
                && !path.chars().any(char::is_control),
            "CUDA profile parent must be a bounded owned absolute runtime root",
        )?;
        // DEFAULT profiles from probe/game PIDs share only this separately
        // owner-created/pinned role root. Game receipt roots remain exclusive
        // to their two game process slots; profiles receive exact PID/nonce,
        // inode/ownership, producer-file and witness-digest validation.
        // Fastchess reuses argv between games. The UCI child derives an
        // exclusive PID+nonce directory here for each preflight/game process.
        Ok(Some(format!("--pals-cuda-profile-parent={path}").into()))
    }
}
#[cfg(target_os = "linux")]
fn default_control_profile_pid(name: &str) -> Result<u32, ArenaError> {
    let rest = name
        .strip_prefix("pals-cuda-profile-")
        .ok_or_else(|| invalid("DEFAULT CUDA profile name is not producer-scoped"))?;
    let (pid, nonce) = rest
        .split_once('-')
        .ok_or_else(|| invalid("DEFAULT CUDA profile PID/nonce absent"))?;
    require(
        !pid.is_empty()
            && pid.bytes().all(|b| b.is_ascii_digit())
            && nonce.len() == 16
            && nonce
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)),
        "DEFAULT CUDA profile PID/nonce is not the exact producer format",
    )?;
    let pid: u32 = pid
        .parse()
        .map_err(|_| invalid("DEFAULT CUDA profile PID invalid"))?;
    require(
        pid > 0 && pid <= i32::MAX as u32 && name == format!("pals-cuda-profile-{pid}-{nonce}"),
        "DEFAULT CUDA profile PID is not canonical or supervised",
    )?;
    Ok(pid)
}

#[cfg(target_os = "linux")]
fn default_control_directory_pin(
    directory: &cap_std::fs::Dir,
) -> Result<std::fs::File, ArenaError> {
    use cap_fs_ext::{FollowSymlinks, OpenOptionsExt, OpenOptionsFollowExt};
    let mut options = cap_std::fs::OpenOptions::new();
    options
        .read(true)
        .follow(FollowSymlinks::No)
        .custom_flags(libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_NONBLOCK);
    directory
        .open_with(".", &options)
        .map(cap_std::fs::File::into_std)
        .map_err(|_| invalid("DEFAULT CUDA profile directory capability pin unavailable"))
}

#[cfg(target_os = "linux")]
fn validate_owned_default_profile_slots(
    root: &cap_std::fs::Dir,
    audits: &[PalsNativeSessionAuditV3],
    native_slots: &BTreeSet<String>,
    entry_limit: u32,
) -> Result<(), ArenaError> {
    use cap_fs_ext::DirExt;
    use std::os::unix::fs::MetadataExt;
    let root_pin = default_control_directory_pin(root)?;
    let root_meta = root_pin
        .metadata()
        .map_err(|_| invalid("DEFAULT CUDA profile parent pin metadata unavailable"))?;
    require(
        root_meta.is_dir() && root_meta.mode() & 0o7777 == 0o700,
        "DEFAULT CUDA profile parent is not the owner-created private root",
    )?;
    let mut expected = BTreeMap::new();
    for audit in audits {
        require(
            audit.process_id > 0
                && audit.physical_shutdown_confirmed
                && audit.native_buffers_released
                && audit.external_cpu_r_session.is_none()
                && audit.cuda_placement.is_some()
                && expected.insert(audit.process_id, audit).is_none(),
            "DEFAULT CUDA profile requires distinct already-validated physical owners",
        )?;
    }
    require(
        (2..=4).contains(&expected.len()),
        "DEFAULT role profile scope must be two probes and optional two games",
    )?;
    let mut actual_profiles = BTreeSet::new();
    let mut actual_native = BTreeSet::new();
    for (index, entry) in root
        .entries()
        .map_err(|e| ArenaError::Io(e.to_string()))?
        .enumerate()
    {
        require(
            index < entry_limit as usize,
            "DEFAULT profile root entry cap exceeded",
        )?;
        let name = entry
            .map_err(|e| ArenaError::Io(e.to_string()))?
            .file_name();
        let name = name
            .to_str()
            .ok_or_else(|| invalid("DEFAULT profile slot is not UTF-8"))?;
        if native_slots.contains(name) {
            require(
                actual_native.insert(name.to_owned()),
                "DEFAULT native slot duplicated",
            )?;
            root.open_dir_nofollow(name)
                .map_err(|_| invalid("DEFAULT native slot is not an owned real directory"))?;
            continue;
        }
        let pid = default_control_profile_pid(name)?;
        let audit = expected
            .get(&pid)
            .ok_or_else(|| invalid("DEFAULT CUDA profile PID was not independently supervised"))?;
        require(
            actual_profiles.insert(pid),
            "DEFAULT CUDA profile PID/nonce slot repeated",
        )?;
        let profile = root
            .open_dir_nofollow(name)
            .map_err(|_| invalid("DEFAULT CUDA profile slot is not an owned real directory"))?;
        let profile_pin = default_control_directory_pin(&profile)?;
        let before = profile_pin
            .metadata()
            .map_err(|_| invalid("DEFAULT profile inode pin metadata unavailable"))?;
        require(
            before.is_dir() && before.uid() == root_meta.uid() && before.mode() & 0o022 == 0,
            "DEFAULT profile directory owner/type/write permissions differ",
        )?;
        let placement = audit
            .cuda_placement
            .as_ref()
            .expect("validated placement above");
        let mut seen = BTreeSet::new();
        for (file_index, entry) in profile
            .entries()
            .map_err(|e| ArenaError::Io(e.to_string()))?
            .enumerate()
        {
            require(file_index < 4, "DEFAULT CUDA profile has excess files")?;
            let filename = entry
                .map_err(|e| ArenaError::Io(e.to_string()))?
                .file_name();
            let filename = filename
                .to_str()
                .ok_or_else(|| invalid("DEFAULT CUDA profile filename is not UTF-8"))?;
            require(
                filename.len() <= 128
                    && filename
                        .bytes()
                        .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.')),
                "DEFAULT CUDA profile filename is not a bounded basename",
            )?;
            let (role, kernel, cap) = if filename == "public-initial-placement.json" {
                ("public", false, 4 * 1024 * 1024)
            } else if filename == "shared_pc-initial-placement.json" {
                ("shared_pc", false, 4 * 1024 * 1024)
            } else if filename.starts_with("pals-public-kernels_") && filename.ends_with(".json") {
                ("public", true, 4 * 1024 * 1024)
            } else if filename.starts_with("pals-shared_pc-kernels_") && filename.ends_with(".json")
            {
                ("shared_pc", true, 4 * 1024 * 1024)
            } else {
                return Err(invalid(
                    "DEFAULT CUDA profile contains an undeclared producer file",
                ));
            };
            require(
                seen.insert((role, kernel)),
                "DEFAULT CUDA profile role/file duplicated",
            )?;
            let bytes = crate::native_runner::linux::read_file(&profile, filename, cap)?;
            if kernel {
                let expected_digest = if role == "public" {
                    &placement.public_profile_sha256
                } else {
                    &placement.shared_pc_profile_sha256
                };
                require(
                    digest(&bytes) == *expected_digest,
                    "DEFAULT actual kernel profile differs from the validated startup witness",
                )?;
            } else {
                let value = json(&bytes)?;
                require(
                    value["schema"] == "rovezero.pals-initial-placement.v2"
                        && value["source"]
                            == "ort-1.22-finalize-recursive-plus-memcpy-origin-metadata"
                        && value["optimization"] == "disable"
                        && value["role"] == role
                        && value["session_created"] == true
                        && value["before_first_neural_run"] == true
                        && value["gate"] == "passed"
                        && value["gate_detail"].is_null()
                        && value["collector_overflow"] == false,
                    "DEFAULT initial placement producer scope or pre-Run gate differs",
                )?;
            }
        }
        require(
            seen.len() == 4,
            "DEFAULT CUDA profile lacks one of four bounded producer files",
        )?;
        let named = root
            .open_dir_nofollow(name)
            .map_err(|_| invalid("DEFAULT profile pathname changed after verification"))?;
        let named_pin = default_control_directory_pin(&named)?;
        let after = profile_pin
            .metadata()
            .map_err(|_| invalid("DEFAULT profile held inode recheck unavailable"))?;
        let named_after = named_pin
            .metadata()
            .map_err(|_| invalid("DEFAULT profile named inode recheck unavailable"))?;
        for metadata in [&after, &named_after] {
            require(
                metadata.is_dir()
                    && (
                        metadata.dev(),
                        metadata.ino(),
                        metadata.uid(),
                        metadata.mode(),
                    ) == (before.dev(), before.ino(), before.uid(), before.mode()),
                "DEFAULT profile capability/name/inode changed during verification",
            )?;
        }
    }
    require(
        actual_native == *native_slots
            && actual_profiles == expected.keys().copied().collect::<BTreeSet<_>>(),
        "DEFAULT exact native/profile process set is incomplete",
    )
}

#[cfg(target_os = "linux")]
fn audit_owned_default_preflight(
    lock: &LockedPalsArenaLaunchV4,
    role: NativeEngineRole,
    preflight: &crate::ExternalUciPreflight,
    root: &cap_std::fs::Dir,
    games: &[PalsNativeSessionAuditV3],
) -> Result<PalsPreflightHelperEvidence, ArenaError> {
    use cap_fs_ext::DirExt;
    require(
        lock.uses_owned_default_preflight(role)
            && lock
                .input
                .external_cpu_r_binding(role_index(role))?
                .is_none()
            && preflight.engine_id == lock.endpoint_views[role_index(role)].id,
        "DEFAULT own preflight requires the exact registered V4 role",
    )?;
    let mut native_slots = BTreeSet::new();
    let mut result = PalsPreflightHelperEvidence {
        sessions: vec![],
        artifacts: vec![],
    };
    for process in [
        &preflight.identification_process,
        &preflight.readiness_process,
    ] {
        require(
            process.pid > 0
                && native_slots.insert(format!("native-process-{}", process.pid))
                && process.stop == crate::ProcessStop::Exited
                && process.exit_code == Some(0)
                && process.exit_signal.is_none()
                && process.group_cleanup == crate::CleanupStatus::Gone
                && process.errors.is_empty(),
            "DEFAULT preflight requires two independently supervised normal exits",
        )?;
        let name = format!("native-process-{}", process.pid);
        let directory = root.open_dir_nofollow(&name).map_err(|_| {
            invalid("DEFAULT preflight receipt slot is not an owned real directory")
        })?;
        let actual = directory
            .entries()
            .map_err(|e| ArenaError::Io(e.to_string()))?
            .take(3)
            .map(|entry| {
                entry
                    .map(|e| e.file_name())
                    .map_err(|e| ArenaError::Io(e.to_string()))
            })
            .collect::<Result<BTreeSet<_>, _>>()?;
        require(
            actual
                == BTreeSet::from([
                    OsString::from(lock.native_wire_filenames().0),
                    OsString::from(lock.native_wire_filenames().1),
                ]),
            "DEFAULT preflight requires exactly startup/termination receipt files",
        )?;
        let start = read_work_file(&directory, lock.native_wire_filenames().0)?;
        let end = read_work_file(&directory, lock.native_wire_filenames().1)?;
        let (audit, pid) =
            validate_pals_native_records_inner(lock, role, &start, &end, &name, None, false)?;
        require(
            pid == process.pid,
            "DEFAULT physical receipt PID differs from its supervised probe",
        )?;
        result.sessions.push(audit);
        result
            .artifacts
            .push((format!("{name}/{}", lock.native_wire_filenames().0), start));
        result
            .artifacts
            .push((format!("{name}/{}", lock.native_wire_filenames().1), end));
    }
    let mut profiles = result.sessions.clone();
    for game in games {
        require(
            game.endpoint_id == lock.endpoint_views[role_index(role)].id
                && game.external_cpu_r_session.is_none(),
            "DEFAULT game profile endpoint/helper origin differs",
        )?;
        profiles.push(game.clone());
    }
    require(
        games.is_empty() || games.len() == 2,
        "DEFAULT profile scope must contain exactly two game owners",
    )?;
    validate_owned_default_profile_slots(
        root,
        &profiles,
        &native_slots,
        lock.input.budget.max_runtime_files,
    )?;
    Ok(result)
}

impl crate::native_launch::sealed::Sealed for LockedPalsArenaLaunchV4 {}
impl NativeLaunchDeclaration for LockedPalsArenaLaunchV4 {
    fn argument_limit(&self) -> usize {
        48
    }
    fn input_sha256(&self) -> &str {
        &self.sha256
    }
    fn view(&self) -> NativePairView<'_> {
        let p = &self.input.semantic_lock.manifest.pilot;
        let ids = &self.input.semantic_lock.manifest.engines;
        let white_order = p.white_order.each_ref().map(|id| {
            if id == ids[0].id() {
                NativeEngineRole::Baseline
            } else {
                NativeEngineRole::Candidate
            }
        });
        NativePairView {
            pair_id: &self.input.semantic_lock.manifest.pair_id,
            white_order,
            opening: &self.opening,
            opening_artifact: &self.input.opening_artifact,
            runner: &self.input.runner,
            clock: NativePairClock::Game(NativeGameClockV3 {
                base_ms: p.base_ms,
                increment_ms: p.increment_ms,
            }),
            max_plies: p.max_plies,
            timeouts: NativeTimeoutsV1 {
                startup_ms: p.handshake_max_ms,
                handshake_ms: p.handshake_max_ms,
                runtime_ms: p.wall_time_max_ms,
                drain_ms: p.cleanup_max_ms,
                shutdown_ms: p.cleanup_max_ms.div_ceil(2),
            },
            budget: self.input.budget,
        }
    }
    fn declared_inputs(&self) -> Vec<&ArtifactRef> {
        self.input.declared_artifacts()
    }
    fn unique_bytes(&self) -> Result<u64, ManifestError> {
        self.input.unique_input_bytes()
    }
    fn engine_view(&self, role: NativeEngineRole) -> Result<NativeEngineView<'_>, ManifestError> {
        let e = &self.endpoint_views[role_index(role)];
        Ok(NativeEngineView {
            engine_id: &e.id,
            artifacts: &[],
            cuda_bundle: self.input.endpoints[role_index(role)]
                .cuda_model()
                .map(|cuda| &cuda.cuda_bundle),
            search: None,
            batch_experiment: None,
            external: Some(e),
            environment: e.environment.as_ref(),
        })
    }
    fn provider_name(&self) -> &'static str {
        "PALS-V4"
    }
    fn pair_metadata_cap(&self) -> u64 {
        V4_METADATA_CAP
            + if self.input.resource_profile.is_some() {
                resources::FAILURE_METADATA_CAP
            } else {
                0
            }
    }
    fn amend_execution_limitations(&self, limitations: &mut Vec<String>) {
        limitations.push("V4 frozen P/C paired pilot; model/profile, resolver, checker, recheck, stack and CUDA flags are separately locked and checked against actual producer getters; no training or Elo promotion".into());
        limitations.push("Shared V3 physical resource DTOs preserve their exact meaning; the V4 semantic lock and raw domains are never coerced to historical IDs".into());
        if self.input.resource_profile.is_some() {
            limitations.push("Explicit Colab-observed functional pilot: inherited cgroup values are observations separate from requested local targets; no strict-local resource, performance, pair or strength eligibility".into());
        }
    }
    fn seed(&self) -> u64 {
        self.input.semantic_lock.manifest.pilot.seed
    }
    fn validate_execution(&self) -> Result<(), ArenaError> {
        self.input.validate()?;
        crate::native_launch::pair_stream_cap(
            self.input.budget.max_output_bytes,
            self.pair_metadata_cap(),
        )?;
        Ok(())
    }
    fn advise_drop_input_cache(&self) -> bool {
        true
    }
    fn snapshot_relative_path(&self, a: &ArtifactRef) -> Option<String> {
        self.input
            .named_assets()
            .into_iter()
            .find_map(|(artifact, path)| (artifact == a).then_some(path))
    }
    fn runtime_arguments(
        &self,
        role: NativeEngineRole,
        runtime_root: &Path,
    ) -> Result<Vec<OsString>, ArenaError> {
        if matches!(
            &self.input.endpoints[role_index(role)],
            PalsEndpointLaunchV3::ReferenceUci { .. }
        ) {
            return Ok(vec![]);
        }
        let root = runtime_root
            .to_str()
            .ok_or_else(|| invalid("native output root is not UTF-8"))?;
        require(
            runtime_root.is_absolute(),
            "native output root must be absolute",
        )?;
        if !matches!(
            &self.input.endpoints[role_index(role)],
            PalsEndpointLaunchV3::OnnxCpu(_) | PalsEndpointLaunchV3::OnnxCuda(_)
        ) {
            return Ok(vec![
                format!("--search-work-output-root={root}").into(),
                format!("--search-work-launch-sha256={}", self.sha256).into(),
                format!(
                    "--search-work-endpoint-id={}",
                    self.endpoint_views[role_index(role)].id
                )
                .into(),
            ]);
        }
        let mut arguments = vec![
            format!("--pals-output-root={root}").into(),
            format!("--pals-launch-sha256={}", self.sha256).into(),
            format!(
                "--pals-endpoint-id={}",
                self.endpoint_views[role_index(role)].id
            )
            .into(),
            pals_shared_runtime_argument()?,
        ];
        if let Some(profile) = self.cuda_profile_argument(role, runtime_root)? {
            arguments.push(profile);
        }
        arguments.extend(self.archive_runtime_arguments(role, runtime_root)?);
        Ok(arguments)
    }
    fn uses_separate_preflight_runtime_root(
        &self,
        role: NativeEngineRole,
    ) -> Result<bool, ArenaError> {
        Ok(self
            .input
            .external_cpu_r_binding(role_index(role))?
            .is_some()
            || self.uses_owned_default_preflight(role))
    }
    fn preflight_arguments(
        &self,
        role: NativeEngineRole,
        runtime_root: &Path,
    ) -> Result<Vec<OsString>, ArenaError> {
        if !matches!(
            &self.input.endpoints[role_index(role)],
            PalsEndpointLaunchV3::OnnxCpu(_) | PalsEndpointLaunchV3::OnnxCuda(_)
        ) {
            return Ok(vec![]);
        }
        require(
            runtime_root.is_absolute(),
            "preflight cache root must be absolute",
        )?;
        let isolated = self.uses_separate_preflight_runtime_root(role)?;
        // The Native capability owner creates and pins this exact root. Do not
        // derive an unowned sibling from a game path at the argv boundary.
        if isolated {
            let expected = match role {
                NativeEngineRole::Baseline => "external-baseline-preflight-runtime",
                NativeEngineRole::Candidate => "external-candidate-preflight-runtime",
            };
            require(
                runtime_root
                    .file_name()
                    .is_some_and(|name| name == expected),
                "external CPU_R preflight requires the owner-selected role root",
            )?;
        }
        let evidence_root = runtime_root;
        let mut arguments = vec![pals_shared_runtime_argument()?];
        if let Some(profile) = self.cuda_profile_argument(role, evidence_root)? {
            arguments.push(profile);
        }
        if isolated {
            let root = evidence_root
                .to_str()
                .ok_or_else(|| invalid("external CPU_R preflight evidence root is not UTF-8"))?;
            arguments.extend([
                format!("--pals-output-root={root}").into(),
                format!("--pals-launch-sha256={}", self.sha256).into(),
                format!(
                    "--pals-endpoint-id={}",
                    self.endpoint_views[role_index(role)].id
                )
                .into(),
            ]);
        }
        arguments.extend(self.archive_existing_arguments(role)?);
        Ok(arguments)
    }
}
impl NativeProviderDeclaration for LockedPalsArenaLaunchV4 {
    type Audit = PalsNativeSessionAuditV3;
    fn scope(&self) -> &'static str {
        "pals_v4_pc_native_or_explicit_mock_cpu_uci_rules_clocks_process_pilot_no_elo"
    }
    fn receipt_filename(&self) -> &'static str {
        "pals-arena-pair-receipt.v4.json"
    }
    fn receipt_version(&self) -> u32 {
        4
    }
    fn contract_revision(&self) -> &str {
        &self.input.semantic_lock.manifest.contract_revision
    }
    #[cfg(target_os = "linux")]
    fn validate_runtime_admission(&self) -> Result<(), ArenaError> {
        self.input.require_reap_status_runner()?;
        self.input.require_actual_native_epoch()?;
        resources::admit(self, "before_native_spawn").map(|_| ())
    }
    #[cfg(target_os = "linux")]
    fn validate_external_preflight(
        &self,
        role: NativeEngineRole,
        preflight: &crate::ExternalUciPreflight,
        runtime_directory: &cap_std::fs::Dir,
        inputs: &crate::native_runner::NativeProviderInputContext<'_>,
    ) -> Result<Vec<(String, Vec<u8>)>, ArenaError> {
        if self.uses_owned_default_preflight(role) {
            return Ok(audit_owned_default_preflight(
                self,
                role,
                preflight,
                runtime_directory,
                &[],
            )?
            .artifacts);
        }
        Ok(
            audit_pals_helper_preflight(self, role, preflight, runtime_directory, inputs)?
                .artifacts,
        )
    }
    #[cfg(target_os = "linux")]
    fn validate_records_with_context(
        &self,
        role: NativeEngineRole,
        startup: &[u8],
        termination: &[u8],
        session: &str,
        inputs: &crate::native_runner::NativeProviderInputContext<'_>,
        supervisor_stdout: &[u8],
    ) -> Result<(Self::Audit, u32), ArenaError> {
        if self
            .input
            .external_cpu_r_binding(role_index(role))?
            .is_none()
        {
            return self.validate_records(role, startup, termination, session);
        }
        #[cfg(not(any(feature = "pals-collection-onnx", feature = "native-cuda")))]
        {
            let _ = (inputs, supervisor_stdout);
            Err(invalid(
                "unsupported external CPU_R game validation: native PALS feature is required; no fallback",
            ))
        }
        #[cfg(any(feature = "pals-collection-onnx", feature = "native-cuda"))]
        {
            let loaded = load_owner_helper_profile(self, role, inputs)?;
            // These role/game IDs originate in the pinned supervisor trace,
            // before any native JSON supplies a purported PID.
            let mapping = validate_pals_game_process_trace_context(
                self,
                supervisor_stdout,
                &[BTreeSet::new(), BTreeSet::new()],
            )?;
            let expected = mapping[role_index(role)]
                .iter()
                .copied()
                .find(|pid| session == format!("native-process-{pid}"))
                .ok_or_else(|| {
                    invalid("native session absent from supervised role/game PID mapping")
                })?;
            let helper = validate_pals_external_cpu_r_session_context(
                self,
                role,
                &loaded,
                startup,
                termination,
                expected,
                PalsExternalCpuRSessionPurposeV3::Game,
            )?;
            validate_pals_native_records_inner(
                self,
                role,
                startup,
                termination,
                session,
                Some(helper),
                true,
            )
        }
    }
    #[cfg(target_os = "linux")]
    fn validate_provider_session_set(
        &self,
        sessions: &[Self::Audit],
        preflight: &[crate::ExternalUciPreflight],
        runtime_root: &cap_std::fs::Dir,
        inputs: &crate::native_runner::NativeProviderInputContext<'_>,
        supervisor_stdout: &[u8],
        artifacts: &[ArtifactRef],
    ) -> Result<(), ArenaError> {
        // The historical helper branch below is unchanged. The DEFAULT own
        // branch binds probe originals again after all four game owners close.
        if [NativeEngineRole::Baseline, NativeEngineRole::Candidate]
            .into_iter()
            .any(|role| self.uses_owned_default_preflight(role))
        {
            use cap_fs_ext::DirExt;
            let mut leaders = BTreeSet::new();
            for game in sessions {
                require(
                    game.process_id > 0 && leaders.insert(game.process_id),
                    "DEFAULT game PID reused",
                )?;
            }
            for role in [NativeEngineRole::Baseline, NativeEngineRole::Candidate] {
                if !self.uses_owned_default_preflight(role) {
                    continue;
                }
                let index = role_index(role);
                let mut matching = preflight
                    .iter()
                    .filter(|p| p.engine_id == self.endpoint_views[index].id);
                let receipt = matching
                    .next()
                    .ok_or_else(|| invalid("DEFAULT preflight receipt absent"))?;
                require(
                    matching.next().is_none(),
                    "DEFAULT preflight receipt duplicated",
                )?;
                for process in [&receipt.identification_process, &receipt.readiness_process] {
                    require(leaders.insert(process.pid), "DEFAULT probe/game PID reused")?;
                }
                let games: Vec<_> = sessions
                    .iter()
                    .filter(|s| s.endpoint_id == self.endpoint_views[index].id)
                    .cloned()
                    .collect();
                require(
                    games.len() == 2,
                    "DEFAULT requires exactly two restarted game owners per role",
                )?;
                let name = if index == 0 {
                    "external-baseline-preflight-runtime"
                } else {
                    "external-candidate-preflight-runtime"
                };
                let root = runtime_root
                    .open_dir_nofollow(name)
                    .map_err(|_| invalid("DEFAULT owner preflight root absent"))?;
                let evidence = audit_owned_default_preflight(self, role, receipt, &root, &games)?;
                for (relative, bytes) in evidence.artifacts {
                    let suffix = format!("/{name}/{relative}");
                    require(
                        artifacts
                            .iter()
                            .filter(|a| {
                                a.path.ends_with(&suffix)
                                    && a.sha256 == digest(&bytes)
                                    && a.bytes == bytes.len() as u64
                            })
                            .count()
                            == 1,
                        "DEFAULT probe evidence changed from its retained original",
                    )?;
                }
            }
        }
        let mut has_external = false;
        for index in 0..2 {
            has_external |= self.input.external_cpu_r_binding(index)?.is_some();
        }
        if !has_external {
            return Ok(());
        }
        use cap_fs_ext::DirExt;
        let mapping = validate_pals_game_process_trace_context(
            self,
            supervisor_stdout,
            &[BTreeSet::new(), BTreeSet::new()],
        )?;
        let mut leaders: BTreeSet<_> = mapping.iter().flatten().copied().collect();
        for receipt in preflight {
            for process in [&receipt.identification_process, &receipt.readiness_process] {
                require(
                    leaders.insert(process.pid),
                    "preflight/game supervisor PID reused",
                )?;
            }
        }
        let mut audited = sessions.to_vec();
        for role in [NativeEngineRole::Baseline, NativeEngineRole::Candidate] {
            let index = role_index(role);
            if self.input.external_cpu_r_binding(index)?.is_none() {
                continue;
            }
            let mut matching = preflight
                .iter()
                .filter(|p| p.engine_id == self.endpoint_views[index].id);
            let receipt = matching
                .next()
                .ok_or_else(|| invalid("external helper preflight receipt absent"))?;
            require(
                matching.next().is_none(),
                "external helper preflight receipt duplicated",
            )?;
            let name = if index == 0 {
                "external-baseline-preflight-runtime"
            } else {
                "external-candidate-preflight-runtime"
            };
            let root = runtime_root
                .open_dir_nofollow(name)
                .map_err(|_| invalid("owner helper preflight root missing"))?;
            let evidence = audit_pals_helper_preflight(self, role, receipt, &root, inputs)?;
            for (relative, bytes) in &evidence.artifacts {
                let suffix = format!("/{name}/{relative}");
                require(
                    artifacts
                        .iter()
                        .filter(|a| {
                            a.path.ends_with(&suffix)
                                && a.sha256 == digest(bytes)
                                && a.bytes == bytes.len() as u64
                        })
                        .count()
                        == 1,
                    "helper preflight bytes differ from retained original evidence",
                )?;
            }
            audited.extend(evidence.sessions);
            let selected: Vec<_> = sessions
                .iter()
                .filter(|s| s.endpoint_id == self.endpoint_views[index].id)
                .collect();
            require(
                selected.len() == 2
                    && selected.iter().all(|s| {
                        mapping[index].contains(&s.process_id)
                            && s.external_cpu_r_session.as_ref().is_some_and(|h| {
                                h.purpose == PalsExternalCpuRSessionPurposeV3::Game
                            })
                    }),
                "two mapped external helper game sessions required",
            )?;
        }
        validate_pals_helper_identity_set(&audited, &leaders)
    }
    fn expected_provider_sessions(&self) -> usize {
        2 * self
            .input
            .endpoints
            .iter()
            .filter(|e| e.native_model().is_some())
            .count()
    }
    fn uses_provider_records(&self, role: NativeEngineRole) -> Result<bool, ArenaError> {
        Ok(matches!(
            &self.input.endpoints[role_index(role)],
            PalsEndpointLaunchV3::OnnxCpu(_) | PalsEndpointLaunchV3::OnnxCuda(_)
        ))
    }
    #[cfg(target_os = "linux")]
    fn external_game_process_ids(
        &self,
        role: NativeEngineRole,
        external_ids: &[u32],
        root_output: &cap_std::fs::Dir,
    ) -> Result<Vec<u32>, ArenaError> {
        require(
            !self.uses_provider_records(role)?,
            "native role has its own physical record audit",
        )?;
        let own = collect_process_work(self, role, root_output)?;
        let selected: Vec<u32> = if own.is_empty() {
            let other = if role_index(role) == 0 {
                NativeEngineRole::Candidate
            } else {
                NativeEngineRole::Baseline
            };
            let other_records = if self.uses_provider_records(other)? {
                vec![]
            } else {
                collect_process_work(self, other, root_output)?
            };
            // Native counterpart IDs have already been excluded by the generic
            // native auditor. A CPU/mock counterpart remains in external_ids.
            external_ids
                .iter()
                .copied()
                .filter(|pid| !other_records.iter().any(|w| w.process_id == *pid))
                .collect()
        } else {
            require(
                own.iter().all(|w| external_ids.contains(&w.process_id)),
                "own work PID missing from supervised normal exit trace",
            )?;
            external_ids
                .iter()
                .copied()
                .filter(|pid| own.iter().any(|w| w.process_id == *pid))
                .collect()
        };
        require(
            selected.len() == 2 && selected[0] != selected[1],
            "two role-specific external exits required",
        )?;
        Ok(selected)
    }
    fn provider_export_artifact(
        &self,
        role: NativeEngineRole,
    ) -> Result<Option<&ArtifactRef>, ArenaError> {
        if self.uses_provider_records(role)? {
            Ok(Some(&self.native(role)?.export))
        } else {
            Ok(None)
        }
    }
    fn startup_filename(&self) -> &'static str {
        "pals-native-startup.v4.json"
    }
    fn termination_filename(&self) -> &'static str {
        "pals-native-termination.v4.json"
    }
    fn claim_policy(&self) -> rz_experiments::ClaimPolicy {
        rz_experiments::ClaimPolicy::AutomaticAcceptance
    }
    fn validate_clock_trace(
        &self,
        stdout: &[u8],
        pgn: &crate::PairPgnAudit,
    ) -> Result<Option<crate::NativePilotClockAudit>, ArenaError> {
        let p = &self.input.semantic_lock.manifest.pilot;
        crate::native_pilot::validate_pilot_clock_trace(
            stdout,
            pgn,
            &self.opening,
            NativeGameClockV3 {
                base_ms: p.base_ms,
                increment_ms: p.increment_ms,
            },
        )
        .map(Some)
    }
    fn audit_failure_trace(
        &self,
        stdout: &[u8],
        pair: &crate::PairSpec,
    ) -> Result<Option<crate::NativePilotFailureAudit>, ArenaError> {
        crate::native_pilot::audit_pilot_startup_failure_trace(stdout, pair)
    }
    #[cfg(target_os = "linux")]
    fn validate_records(
        &self,
        role: NativeEngineRole,
        startup: &[u8],
        termination: &[u8],
        session: &str,
    ) -> Result<(Self::Audit, u32), ArenaError> {
        validate_pals_native_records_inner(self, role, startup, termination, session, None, true)
    }
    #[cfg(target_os = "linux")]
    fn validate_conversion_manifest(
        &self,
        role: NativeEngineRole,
        startup: &[u8],
        manifest: &[u8],
        batch: bool,
    ) -> Result<(), ArenaError> {
        require(
            !batch,
            "PALS native B1 recipe does not authorize experimental batch",
        )?;
        self.validate_export(role, manifest)?;
        let s = json(startup)?;
        require(
            array_hash(&s["native"]["export_manifest_sha256"])? == self.native(role)?.export.sha256,
            "startup export digest differs",
        )
    }
    #[cfg(target_os = "linux")]
    fn external_exit_ids(&self, stdout: &[u8], pids: &[u32]) -> Result<Vec<u32>, ArenaError> {
        crate::native_exit::validate_endpoint_exit_trace(
            stdout,
            pids,
            4 - self.expected_provider_sessions(),
        )
    }
    #[cfg(target_os = "linux")]
    fn validate_process_exit_trace(&self, stdout: &[u8], pids: &[u32]) -> Result<(), ArenaError> {
        self.external_exit_ids(stdout, pids).map(|_| ())
    }
}
pub fn prepare_pals_pair_launch_v4(
    spec: &LockedPalsArenaLaunchV4,
    source_root: &Path,
    output_root: &Path,
    label: &str,
) -> Result<NativeLaunchOwner<LockedPalsArenaLaunchV4>, Box<crate::NativePreparationFailure>> {
    if let Err(cause) = verify_pals_launch_exports_v4(spec, source_root) {
        return Err(Box::new(crate::NativePreparationFailure {
            receipt:crate::NativePreparationReceipt {receipt_version:4,policy_envelope:None,execution_ready:false,input_sha256:spec.sha256.clone(),
                output_directory:"not-created".into(),attempt_created:false,attempted_snapshot_relative_paths:vec![],completed_snapshots:vec![],
                child_spawned:false,writers_closed:true,input_pins_required:false,original_error:cause.to_string(),
                subsequent_file_owner:"source assets retained; descriptor verification failed before snapshot creation or child launch".into(),automatic_retry:false},
            cause,receipt_artifact:None,persistence_error:None,
        }));
    }
    crate::native_launch::prepare_native_launch_for(spec, source_root, output_root, label)
}
/// Uses the existing bounded runner, clocks, Rules replay and cleanup receipts.
pub fn run_pals_pair_v4(
    owner: NativeLaunchOwner<LockedPalsArenaLaunchV4>,
    cancel: Option<&std::sync::atomic::AtomicBool>,
) -> Result<
    crate::NativePairOutput<LockedPalsArenaLaunchV4>,
    Box<crate::NativePairFailure<LockedPalsArenaLaunchV4>>,
> {
    crate::native_runner::run_native_pair_for(owner, cancel)
}

/// Time scope starts before UCI preflight and ends after mandatory arena
/// postchecks and receipt persistence. Snapshot preparation remains separate.
pub struct PalsObservedPairOutputV4 {
    pub native: crate::NativePairOutput<LockedPalsArenaLaunchV4>,
    pub wall_time_ms: u64,
}

#[derive(Clone, Debug, Serialize)]
pub struct PalsProcessWorkAuditV4 {
    #[serde(flatten)]
    pub base: PalsProcessWorkAuditV3,
    pub marker: Option<PalsActualFollowupMarkerV4>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub followup_lifecycle: Option<serde_json::Value>,
}
impl std::ops::Deref for PalsProcessWorkAuditV4 {
    type Target = PalsProcessWorkAuditV3;
    fn deref(&self) -> &Self::Target {
        &self.base
    }
}
#[cfg(target_os = "linux")]
fn collect_process_work(
    lock: &LockedPalsArenaLaunchV4,
    role: NativeEngineRole,
    root: &cap_std::fs::Dir,
) -> Result<Vec<PalsProcessWorkAuditV4>, ArenaError> {
    collect_process_work_with_audits(lock, role, root, &[])
}

#[cfg(target_os = "linux")]
fn metadata_output_bytes(
    output: &PalsObservedPairOutputV4,
    work: &[PalsProcessWorkAuditV4],
) -> Result<u64, ArenaError> {
    let p = &output.native.receipt.process;
    let owner = output.native.launch_owner();
    let lock = owner.spec();
    let mut files = BTreeMap::<String, (String, u64)>::new();
    let mut admit = |path: String, sha256: String, bytes: u64| -> Result<(), ArenaError> {
        if let Some(previous) = files.insert(path, (sha256.clone(), bytes)) {
            require(
                previous == (sha256, bytes),
                "V4 retained evidence path has conflicting bytes",
            )?;
        }
        Ok(())
    };
    for artifact in output
        .native
        .receipt
        .artifacts
        .iter()
        .chain(std::iter::once(&output.native.receipt_artifact))
    {
        admit(
            artifact.path.clone(),
            artifact.sha256.clone(),
            artifact.bytes,
        )?;
    }
    for work in work {
        let index = lock
            .endpoint_views
            .iter()
            .position(|endpoint| endpoint.id == work.endpoint_id)
            .ok_or_else(|| invalid("V4 work output has unknown endpoint"))?;
        let root = if index == 0 {
            "baseline-runtime"
        } else {
            "candidate-runtime"
        };
        let filenames = if lock.input.endpoints[index].native_model().is_some() {
            (
                "pals-native-startup.v4.json",
                "pals-native-termination.v4.json",
            )
        } else {
            (
                "search-work-startup.v4.json",
                "search-work-termination.v4.json",
            )
        };
        for (name, sha256, bytes) in [
            (filenames.0, &work.startup_sha256, work.startup_bytes),
            (
                filenames.1,
                &work.termination_sha256,
                work.termination_bytes,
            ),
        ] {
            admit(
                format!(
                    "{}/{root}/native-process-{}/{name}",
                    owner.snapshot.output_directory, work.process_id
                ),
                sha256.clone(),
                bytes,
            )?;
        }
    }
    files
        .values()
        .try_fold(p.observed_output_bytes, |total, (_, bytes)| {
            total.checked_add(*bytes)
        })
        .ok_or_else(|| invalid("V4 output byte observation overflow"))
}
#[cfg(target_os = "linux")]
fn observed_output_bytes(
    output: &PalsObservedPairOutputV4,
    work: &[PalsProcessWorkAuditV4],
) -> Result<Option<u64>, ArenaError> {
    #[cfg(any(feature = "pals-collection-onnx", feature = "native-cuda"))]
    let mut total = metadata_output_bytes(output, work)?;
    #[cfg(not(any(feature = "pals-collection-onnx", feature = "native-cuda")))]
    let total = metadata_output_bytes(output, work)?;
    let lock = output.native.launch_owner().spec();
    #[cfg(any(feature = "pals-collection-onnx", feature = "native-cuda"))]
    {
        let mut sessions = BTreeSet::new();
        for session in work {
            let index = lock
                .endpoint_views
                .iter()
                .position(|e| e.id == session.endpoint_id)
                .ok_or_else(|| invalid("archive output belongs to unknown endpoint"))?;
            require(
                sessions.insert((index, session.process_id)),
                "archive output process counted twice",
            )?;
            let role = if index == 0 {
                NativeEngineRole::Baseline
            } else {
                NativeEngineRole::Candidate
            };
            let Some(extent) =
                lifecycle::archive_output_extent(lock, role, session.followup_lifecycle.as_ref())?
            else {
                return Ok(None);
            };
            total = total
                .checked_add(extent)
                .ok_or_else(|| invalid("archive write output overflow"))?;
        }
        for index in 0..2 {
            if lock
                .input
                .execution_binding(index)
                .is_some_and(|b| b.cold_archive.is_some())
                && sessions.iter().filter(|(i, _)| *i == index).count() != 2
            {
                return Ok(None);
            }
        }
    }
    #[cfg(not(any(feature = "pals-collection-onnx", feature = "native-cuda")))]
    {
        let _ = lock;
        require(
            output
                .native
                .launch_owner()
                .spec()
                .input
                .followup_execution
                .is_none(),
            "actual archive output requires the selected Native producer",
        )?;
    }
    Ok(Some(total))
}
#[cfg(any(target_os = "linux", test))]
const MODEL_OBSERVATION_METHOD_V4: &str = "loaded profile/config/epoch, registered semantics and compiled model SHA checked against pinned manifest; original graph/runtime/checkpoint/export bytes verified; diagnostic export rejected before spawn";
#[cfg(target_os = "linux")]
const PROCESS_CLEANUP_METHOD_V4: &str = "exact renderer/normal-exit game PID matches work/native owner; supervisor group Gone/reaped and readers joined; shared pair streams charged once to first baseline game, other zero charges are not per-child measurements; interval is whole-pair cleanup";

#[cfg(target_os = "linux")]
fn observed<T>(value: T, method: &str) -> PalsObservedV3<T> {
    PalsObservedV3::Observed {
        value,
        method: method.into(),
    }
}
#[cfg(target_os = "linux")]
fn endpoint_admitted(e: &PalsEndpointReceiptV4) -> bool {
    let model_ok = matches!(e.checks.model, PalsObservedV3::Observed { .. })
        || matches!(e.checks.policies, PalsObservedV3::Unknown);
    model_ok && matches!(e.checks.artifacts,PalsObservedV3::Observed{..})
        && matches!(e.checks.resource_limits,PalsObservedV3::Observed{..})
        && matches!(e.process_cleanup,PalsObservedV3::Observed{..})
        // Enabled lanes require their separately captured lifecycle/trace.
        && match &e.checks.policies {
            PalsObservedV3::Observed{value,..}=> {
                let p=&value.policies;
                let selected=p.recheck!=PalsRecheckPolicyV4::Disabled
                    || !matches!(p.cold_archive,PalsColdArchivePolicyV4::Disabled)
                    || !matches!(p.paused_stack,PalsPausedStackPolicyV4::Disabled)
                    || !matches!(p.cuda_warm,PalsCudaWarmPolicyV4::Disabled);
                (!selected || e.followup_lifecycle.as_ref().is_some_and(|scope|
                    scope.process_epoch.is_some() && scope.capture_complete && scope.owner_shutdown_complete
                    && scope.capture_failure.is_none() && scope.partial_facts.is_none()))
                &&
                (matches!(value.policies.cold_archive,PalsColdArchivePolicyV4::Disabled)||matches!(e.archive,PalsObservedV3::Observed{..}))
                && (matches!(value.policies.paused_stack,PalsPausedStackPolicyV4::Disabled)||matches!(e.paused_stack,PalsObservedV3::Observed{..}))
                && (matches!(value.policies.cuda_warm,PalsCudaWarmPolicyV4::Disabled)||matches!(e.cuda_warm,PalsObservedV3::Observed{..}))
                && (matches!(value.policies.recheck,PalsRecheckPolicyV4::Disabled)||matches!(e.repair_trace_total,PalsObservedV3::Observed{value,..} if value==e.repair_traces.len() as u64))
                && (matches!(p.iterative_repair,PalsIterativeRepairPolicyV4::Disabled)
                    || matches!(e.repair_admissions_peak,Some(PalsObservedV3::Observed{..})))
            },
            PalsObservedV3::Unknown=>true,
        }
}
#[cfg(target_os = "linux")]
#[derive(Clone, Copy)]
struct PalsGameCleanupScopeV4 {
    // Comes from the exact renderer/color/normal-exit mapping, never a game
    // ordinal or the shared Fastchess supervisor PID.
    process_id: u32,
    // The retained pair streams are shared and must be charged only once.
    charge_pair_streams: bool,
}
#[cfg(target_os = "linux")]
fn cleanup_game_owner_v4(
    supervisor_pid: u32,
    game_pid: u32,
    launch_sha256: &str,
    work_pid: Option<u32>,
    native_pid: Option<u32>,
) -> Result<String, ArenaError> {
    require(
        game_pid > 0
            && game_pid <= i32::MAX as u32
            && game_pid != supervisor_pid
            && work_pid.is_none_or(|pid| pid == game_pid)
            && native_pid.is_none_or(|pid| pid == game_pid),
        "V4 cleanup game PID differs from audited renderer/work/native owner",
    )?;
    Ok(format!("arena-game-process-{game_pid}-{launch_sha256}"))
}
#[cfg(target_os = "linux")]
fn cleanup_stream_bytes_v4(
    scope: PalsGameCleanupScopeV4,
    stdout_bytes: u64,
    stderr_bytes: u64,
) -> (u64, u64) {
    if scope.charge_pair_streams {
        (stdout_bytes, stderr_bytes)
    } else {
        (0, 0)
    }
}
#[cfg(target_os = "linux")]
fn endpoint_v4_receipt(
    output: &PalsObservedPairOutputV4,
    role: NativeEngineRole,
    work: Option<&PalsProcessWorkAuditV4>,
    native: Option<&PalsNativeSessionAuditV3>,
    preflight: &crate::ExternalUciPreflight,
    cleanup_ms: u64,
    game_scope: PalsGameCleanupScopeV4,
) -> Result<PalsEndpointReceiptV4, ArenaError> {
    let owner = output.native.launch_owner();
    let lock = owner.spec();
    let index = role_index(role);
    let engine = &lock.input.semantic_lock.manifest.engines[index];
    let base = endpoint_shared_work_receipt(lock, role, work.map(|w| &w.base), native, preflight)?;
    let mut out = PalsEndpointReceiptV4::with_unknown_checks(base);
    let artifacts: Vec<&ArtifactRef> = match engine {
        PalsEngineV4::Pals(e) => {
            let mut a = vec![&e.base.binary];
            if let PalsWeightIdentityV3::Untrained { artifact, .. }
            | PalsWeightIdentityV3::Trained { artifact, .. } = &e.base.model.weights
            {
                a.push(artifact);
            }
            if let PalsCpuRSelectionV3::ExternalUci(c) = &e.base.cpu_r {
                a.extend([&c.profile, &c.binary]);
            }
            a
        }
        PalsEngineV4::OwnCpu(e) => vec![&e.binary],
        PalsEngineV4::ReferenceUci(e) => {
            let mut a = vec![&e.binary];
            a.extend(&e.assets);
            a
        }
    };
    let verified = artifacts
        .into_iter()
        .map(|a| {
            let snapshot = owner
                .snapshots()
                .iter()
                .find(|s| s.artifact == *a)
                .ok_or_else(|| invalid("V4 required artifact absent from pinned snapshots"))?;
            require(
                snapshot.sha256 == a.sha256
                    && snapshot.bytes == a.bytes
                    && snapshot.distinct_source_inode
                    && snapshot.closed_writer_read_only,
                "V4 pinned snapshot artifact verification differs",
            )?;
            Ok((
                a.path.clone(),
                PalsVerifiedArtifactV4 {
                    sha256: snapshot.sha256.clone(),
                    bytes: snapshot.bytes,
                    pinned_handle_consumed: output.native.receipt.integration_checks_passed,
                },
            ))
        })
        .collect::<Result<BTreeMap<_, _>, ArenaError>>()?;
    out.checks.artifacts = observed(
        verified,
        "owner copied verified original handle into distinct synced read-only inode; pinned executable/input handles consumed by audited runner/native loading and retired after process/physical cleanup",
    );
    let affinity = resources::admit(lock, "endpoint_receipt_after_process_exit")?;
    out.base.actual_affinity = observed(
        affinity,
        "actual /proc/self affinity after inherited native admission; child affinity is separately attested for external checker",
    );
    out.checks.resource_limits = if lock.input.resource_profile.is_some() {
        // Actual numeric/max/unknown values are retained in the separate assembly
        // observation. Requested local limits cannot become observed matches.
        PalsObservedV3::Unknown
    } else {
        observed(
            lock.input.semantic_lock.manifest.resources[index].clone(),
            "actual /proc/self affinity and cgroup-v2 memory.high/max/swap.max independently rechecked; requested device index only, no VRAM peak claim",
        )
    };
    if work.is_some() {
        out.checks.physical_work_accounting = observed(
            true,
            "validated actual producer invocation/CPU task counters and native physical B1 inputs/calls separately; current process/game mapping and no unknown owner",
        );
    }
    if let PalsEngineV4::Pals(e) = engine {
        let actual = work.and_then(|w| w.marker.as_ref()).ok_or_else(|| {
            invalid("V4 policy receipt requires an observed actual immutable marker")
        })?;
        let identity = PalsPolicyIdentityV4 {
            version: e.policy_identity.version.clone(),
            policy: e.policy_identity.policy.clone(),
            search_identity: actual.policy_identity.clone(),
            conditions_sha256: actual
                .policy_conditions_sha256
                .ok_or_else(|| invalid("V4 actual policy conditions unknown"))?,
        };
        let mut actual_policies = e.policies.clone();
        actual_policies.resolver_identity.implementation_sha256 =
            hex(actual.resolver_implementation_sha256);
        actual_policies.resolver_semantics_sha256 = hex(actual.resolver_semantics_sha256);
        out.checks.policies = observed(
            PalsPolicyCheckV4 {
                search: PalsComponentV3 {
                    semantic_id: actual.policy_identity.clone(),
                    implementation_sha256: hex(actual.search_implementation_sha256),
                    options: e.base.search.options.clone(),
                },
                policies: actual_policies,
                identity,
            },
            "actual driver getters + resolver semantic digest, exact recheck condition hash, checker profile/own registration, profile/stack/CUDA argv and loaded marker compared independently with lock; disabled optional lanes remain off",
        );
        if let Some(n) = native {
            require(
                actual.model_configuration.as_ref() == Some(&config(e))
                    && array_hash(&n.raw_native["model_epoch"])?
                        == match &e.base.model.weights {
                            PalsWeightIdentityV3::Untrained { artifact, .. }
                            | PalsWeightIdentityV3::Trained { artifact, .. } => {
                                artifact.sha256.clone()
                            }
                            _ => return Err(invalid("V4 actual checkpoint absent")),
                        },
                "V4 actual model proof drifted",
            )?;
            let mut model = e.base.model.clone();
            model.implementation_sha256 = hex(actual
                .model_implementation_sha256
                .ok_or_else(|| invalid("V4 actual model source pin unknown"))?);
            let mut specification = e.model_v2.clone();
            specification.model_semantics_sha256 = hex(actual
                .model_semantics_sha256
                .ok_or_else(|| invalid("V4 actual model semantics pin unknown"))?);
            out.checks.model = observed(
                PalsModelCheckV4 {
                    model,
                    specification,
                },
                MODEL_OBSERVATION_METHOD_V4,
            );
        }
    }
    let process = &output.native.receipt.process;
    let exited = out.base.process_exited
        && process.group_cleanup == crate::CleanupStatus::Gone
        && process.exit_code == Some(0)
        && output.native.receipt.cleanup_verified
        && !output.native.receipt.unresolved_owner_retained;
    require(
        exited,
        "V4 supervisor/game-process ownership cleanup incomplete",
    )?;
    let owner_identity = cleanup_game_owner_v4(
        process.pid,
        game_scope.process_id,
        &lock.sha256,
        work.map(|w| w.process_id),
        native.map(|n| n.process_id),
    )?;
    let (stdout_bytes, stderr_bytes) =
        cleanup_stream_bytes_v4(game_scope, process.stdout_bytes, process.stderr_bytes);
    out.process_cleanup = observed(
        PalsProcessCleanupV4 {
            owner_identity,
            process_exited: true,
            exit_code: process.exit_code,
            exit_signal: process.exit_signal,
            child_processes_remaining: 0,
            stdout_drained: true,
            stderr_drained: true,
            stdout_bytes,
            stderr_bytes,
            elapsed_ms: cleanup_ms,
            cleanup_complete: true,
            ownership_lost: false,
        },
        PROCESS_CLEANUP_METHOD_V4,
    );
    #[cfg(any(feature = "pals-collection-onnx", feature = "native-cuda"))]
    if let Some(work) = work {
        lifecycle::project(lock, role, work, native, &mut out)?;
    }
    Ok(out)
}
pub fn run_pals_pair_observed_v4(
    owner: NativeLaunchOwner<LockedPalsArenaLaunchV4>,
    cancel: Option<&std::sync::atomic::AtomicBool>,
) -> Result<PalsObservedPairOutputV4, Box<crate::NativePairFailure<LockedPalsArenaLaunchV4>>> {
    let started = std::time::Instant::now();
    run_pals_pair_v4(owner, cancel).map(|native| PalsObservedPairOutputV4 {
        native,
        wall_time_ms: u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX),
    })
}

pub fn verify_pals_launch_exports_v4(
    spec: &LockedPalsArenaLaunchV4,
    source_root: &Path,
) -> Result<(), ArenaError> {
    verify_external_profiles(&spec.input, source_root)?;
    for role in [NativeEngineRole::Baseline, NativeEngineRole::Candidate] {
        if spec.uses_provider_records(role)? {
            let n = spec.native(role)?;
            let mut file = n.export.open_verified(source_root, 128 * 1024)?;
            let mut bytes = Vec::with_capacity(n.export.bytes as usize);
            file.read_to_end(&mut bytes)
                .map_err(|e| ArenaError::Io(e.to_string()))?;
            spec.validate_export(role, &bytes)?;
        }
        if let Some(cuda) = spec.input.endpoints[role_index(role)].cuda_model() {
            let mut file = cuda
                .cuda_bundle
                .manifest
                .open_verified(source_root, 64 * 1024)?;
            let mut bytes = Vec::with_capacity(cuda.cuda_bundle.manifest.bytes as usize);
            file.read_to_end(&mut bytes)
                .map_err(|e| ArenaError::Io(e.to_string()))?;
            spec.validate_cuda_bundle(role, &bytes)?;
            if let Some(control) = &cuda.cuda_control {
                let mut file = control
                    .inventory
                    .open_verified(source_root, MAX_CONTROL_INVENTORY_BYTES)?;
                let mut bytes = Vec::with_capacity(control.inventory.bytes as usize);
                file.read_to_end(&mut bytes)
                    .map_err(|e| ArenaError::Io(e.to_string()))?;
                spec.validate_control_inventory(role, &bytes)?;
            }
        }
    }
    Ok(())
}

#[cfg(target_os = "linux")]
pub fn verify_pals_inherited_resources_v4(
    spec: &LockedPalsArenaLaunchV4,
) -> Result<Vec<u32>, ArenaError> {
    fn read(path: &Path) -> Result<String, ArenaError> {
        let mut f = std::fs::File::open(path).map_err(|e| ArenaError::Io(e.to_string()))?;
        let mut bytes = vec![];
        f.by_ref()
            .take(64 * 1024 + 1)
            .read_to_end(&mut bytes)
            .map_err(|e| ArenaError::Io(e.to_string()))?;
        require(
            bytes.len() <= 64 * 1024,
            "resource observation budget exceeded",
        )?;
        String::from_utf8(bytes).map_err(|_| invalid("resource observation is not UTF-8"))
    }
    let status = read(Path::new("/proc/self/status"))?;
    let cpus = status
        .lines()
        .find_map(|s| s.strip_prefix("Cpus_allowed_list:"))
        .ok_or_else(|| invalid("missing affinity evidence"))?
        .trim();
    let actual = parse_pals_affinity(cpus)?;
    let r = &spec.input.semantic_lock.manifest.resources[0];
    let mut wanted = r.cpu_affinity.clone();
    wanted.sort_unstable();
    require(
        actual == wanted,
        "inherited affinity differs from PALS launch",
    )?;
    let membership = read(Path::new("/proc/self/cgroup"))?;
    let relative = membership
        .lines()
        .find_map(|s| s.strip_prefix("0::"))
        .ok_or_else(|| invalid("cgroup v2 is required"))?;
    require(
        relative.starts_with('/') && !relative.split('/').any(|c| c == ".." || c == "."),
        "cgroup membership path invalid",
    )?;
    let root = Path::new("/sys/fs/cgroup").join(relative.trim_start_matches('/'));
    for (file, value) in [
        ("memory.high", r.memory_high_bytes),
        ("memory.max", r.memory_max_bytes),
        ("memory.swap.max", r.swap_max_bytes),
    ] {
        require(
            read(&root.join(file))?.trim() == value.to_string(),
            "inherited cgroup memory limit differs from PALS launch",
        )?;
    }
    Ok(actual)
}

/// Explicit profile-aware admission. The historical verifier above stays strict.
#[cfg(target_os = "linux")]
pub fn verify_pals_runtime_resources_v4(
    spec: &LockedPalsArenaLaunchV4,
) -> Result<Vec<u32>, ArenaError> {
    resources::admit(spec, "explicit_execute_before_preparation")
}

#[cfg(target_os = "linux")]
fn collect_process_work_with_audits(
    lock: &LockedPalsArenaLaunchV4,
    role: NativeEngineRole,
    root: &cap_std::fs::Dir,
    native_audits: &[PalsNativeSessionAuditV3],
) -> Result<Vec<PalsProcessWorkAuditV4>, ArenaError> {
    use cap_fs_ext::DirExt;
    if matches!(
        &lock.input.endpoints[role_index(role)],
        PalsEndpointLaunchV3::ReferenceUci { .. }
    ) {
        return Ok(vec![]);
    }
    let path = if role_index(role) == 0 {
        "baseline-runtime"
    } else {
        "candidate-runtime"
    };
    let runtime = root
        .open_dir_nofollow(path)
        .map_err(|e| ArenaError::Io(e.to_string()))?;
    let native = matches!(
        &lock.input.endpoints[role_index(role)],
        PalsEndpointLaunchV3::OnnxCpu(_) | PalsEndpointLaunchV3::OnnxCuda(_)
    );
    let (start_name, end_name) = if native {
        (
            "pals-native-startup.v4.json",
            "pals-native-termination.v4.json",
        )
    } else {
        (
            "search-work-startup.v4.json",
            "search-work-termination.v4.json",
        )
    };
    let mut records = vec![];
    for (index, entry) in runtime
        .entries()
        .map_err(|e| ArenaError::Io(e.to_string()))?
        .enumerate()
    {
        require(
            index < lock.input.budget.max_runtime_files as usize,
            "work runtime entry bound exceeded",
        )?;
        let entry = entry.map_err(|e| ArenaError::Io(e.to_string()))?;
        let filename = entry.file_name();
        let name = filename
            .to_str()
            .ok_or_else(|| invalid("non-UTF8 runtime slot"))?;
        if name == "runtime-cache" && native {
            continue;
        }
        if name == "cold-archive-managed"
            && lock
                .input
                .execution_binding(role_index(role))
                .and_then(|b| b.cold_archive.as_ref())
                .is_some()
        {
            continue;
        }
        require(
            name.starts_with("native-process-") && records.len() < 2,
            "unexpected or excessive work runtime slot",
        )?;
        let directory = runtime
            .open_dir_nofollow(&filename)
            .map_err(|e| ArenaError::Io(e.to_string()))?;
        let start = read_work_file(&directory, start_name)?;
        let end = read_work_file(&directory, end_name)?;
        let mut matching = native_audits.iter().filter(|audit| {
            audit.endpoint_id == lock.endpoint_views[role_index(role)].id
                && name == format!("native-process-{}", audit.process_id)
        });
        let audit = matching.next();
        require(matching.next().is_none(), "native work audit duplicated")?;
        records.push(validate_work_records(
            lock, role, &start, &end, name, audit,
        )?);
    }
    require(
        records.len() == 2 && records[0].process_id != records[1].process_id,
        "two distinct restarted game work processes required",
    )?;
    Ok(records)
}

#[cfg(target_os = "linux")]
fn endpoint_shared_work_receipt(
    lock: &LockedPalsArenaLaunchV4,
    role: NativeEngineRole,
    work: Option<&PalsProcessWorkAuditV3>,
    native: Option<&PalsNativeSessionAuditV3>,
    preflight: &crate::ExternalUciPreflight,
) -> Result<PalsEndpointReceiptV3, ArenaError> {
    let index = role_index(role);
    let engine = &lock.input.semantic_lock.manifest.engines[index];
    require(
        preflight.engine_id == engine.id(),
        "work/preflight endpoint differs",
    )?;
    let mut options = BTreeMap::new();
    for (key, requested) in requested_options(engine) {
        let value = preflight
            .options
            .get(key)
            .ok_or_else(|| invalid("requested option missing from preflight"))?;
        require(
            value.value_supported && value.sent_to_preflight && value.readiness_barrier_observed,
            "requested option/preflight barrier unsupported",
        )?;
        options.insert(
            key.clone(),
            PalsOptionReceiptV3 {
                requested: requested.clone(),
                advertised_supported: true,
                observed: value
                    .actual_value
                    .as_ref()
                    .map_or(PalsObservedV3::Unknown, |value| PalsObservedV3::Observed {
                        value: value.clone(),
                        method: "independent option value observation from preflight".into(),
                    }),
            },
        );
    }
    let mut receipt = PalsEndpointReceiptV3 {
        endpoint_id: engine.id().into(),
        external_cpu_r: None,
        uci_ready_observed: true,
        search_failed_go_count: None,
        pals_search_policy: None,
        options,
        actual_affinity: PalsObservedV3::Unknown,
        memory_peak_bytes: PalsObservedV3::Unknown,
        gpu_device: PalsObservedV3::Unknown,
        vram_peak_bytes: PalsObservedV3::Unknown,
        nn_inputs_completed: 0,
        nn_calls_completed: 0,
        nn_inputs_consumed: 0,
        cached_evaluations_consumed: 0,
        proposer_tasks_completed: 0,
        critic_tasks_completed: 0,
        cpu_tasks_requested: 0,
        cpu_tasks_completed: 0,
        cpu_tasks_reused_consumed: 0,
        cpu_tasks_consumed: 0,
        cpu_nodes: 0,
        physical_state: PalsPhysicalStateV3::NotRequired,
        buffers_released: true,
        process_exited: true,
    };
    match engine {
        PalsEngineV4::ReferenceUci(_) => require(
            work.is_none() && native.is_none(),
            "reference UCI cannot inherit RoveZero work",
        )?,
        PalsEngineV4::OwnCpu(_) => {
            let w = &work
                .ok_or_else(|| invalid("own CPU work unobserved"))?
                .search_work;
            validate_pals_process_counter_units(w, "cpu", false, false, false)?;
            receipt.search_failed_go_count = Some(work_count(w, "failed_returns")?);
            let cpu = &w["cpu"];
            receipt.cpu_tasks_requested = work_count(cpu, "tasks_requested")?;
            receipt.cpu_tasks_completed = work_count(cpu, "requested_depth_completed")?
                .checked_add(work_count(cpu, "rules_terminal_reports")?)
                .ok_or_else(|| invalid("CPU completed task overflow"))?;
            // Partial/frontier estimates are retained in raw work evidence; the
            // core completed-task consumer count has the stricter unit.
            receipt.cpu_tasks_consumed =
                work_count(cpu, "completed_reports_accepted_for_uci_output")?;
            receipt.cpu_nodes = work_count(cpu, "nodes")?;
            require(native.is_none(), "own CPU cannot claim native NN evidence")?;
        }
        PalsEngineV4::Pals(endpoint) => {
            let p = &endpoint.base;
            let w = &work
                .ok_or_else(|| invalid("PALS search work unobserved"))?
                .search_work;
            let external = !p.cpu_r.is_own();
            validate_pals_process_counter_units(w, "pals", false, external, false)?;
            receipt.search_failed_go_count = Some(work_count(w, "failed_returns")?);
            let t = &w["pals"];
            receipt.proposer_tasks_completed = work_count(t, "completed_proposer_calls")?
                .checked_add(work_count(t, "completed_repair_calls")?)
                .ok_or_else(|| invalid("proposer task overflow"))?;
            receipt.critic_tasks_completed = work_count(t, "completed_critic_calls")?;
            receipt.cpu_tasks_requested = work_count(t, "cpu_tasks_requested")?;
            receipt.cpu_tasks_completed = work_count(t, "completed_cpu_tasks")?;
            receipt.cpu_tasks_reused_consumed =
                work_count(t, "reused_completed_cpu_tasks_consumed")?;
            receipt.cpu_tasks_consumed = work_count(t, "consumed_cpu_tasks")?;
            receipt.cpu_nodes = work_count(t, "cpu_nodes")?;
            if external {
                let work = work.unwrap();
                let n = native.ok_or_else(|| {
                    invalid("external CPU_R lacks production native session audit")
                })?;
                let helper = n.external_cpu_r_session.as_ref().ok_or_else(|| {
                    invalid("external CPU_R lacks validated independent helper projection")
                })?;
                require(
                    helper.purpose == PalsExternalCpuRSessionPurposeV3::Game
                        && helper.endpoint_id == receipt.endpoint_id
                        && helper.parent_process_id == work.process_id
                        && n.launch_sha256 == lock.sha256
                        && n.startup_sha256 == work.startup_sha256
                        && n.termination_sha256 == work.termination_sha256
                        && n.raw_search_work == work.search_work
                        && helper.startup_sha256 == work.startup_sha256
                        && helper.termination_sha256 == work.termination_sha256
                        && receipt.cpu_tasks_requested == 0
                        && receipt.cpu_tasks_completed == 0
                        && receipt.cpu_tasks_reused_consumed == 0
                        && receipt.cpu_tasks_consumed == 0
                        && receipt.cpu_nodes == 0,
                    "external helper/game/work projection differs or claims Own work",
                )?;
                helper
                    .receipt
                    .validate_against(p, &lock.input.semantic_lock.manifest.resources[index])?;
                receipt.external_cpu_r = Some(helper.receipt.clone());
            } else {
                require(
                    native.is_none_or(|n| n.external_cpu_r_session.is_none()),
                    "own CPU_R cannot inherit validated foreign projection",
                )?;
            }
            if p.model.backend == PalsModelBackendV3::DeterministicMock {
                require(native.is_none(), "explicit mock cannot claim NN inputs")?;
            } else {
                let n =
                    native.ok_or_else(|| invalid("PALS native work missing completion receipt"))?;
                require(
                    n.process_id == work.unwrap().process_id
                        && n.endpoint_id == receipt.endpoint_id
                        && n.physical_shutdown_confirmed
                        && n.native_buffers_released,
                    "native/work process or physical release differs",
                )?;
                require(
                    p.model.frozen_epoch == 1 && count(&n.raw_native, "frozen_epoch")? == 1,
                    "new Core requires actual deployment frozen epoch 1",
                )?;
                require(
                    n.raw_native["backend_stats_observation"] == "exclusive_worker_before_shutdown",
                    "native graph work observation missing",
                )?;
                require(
                    count(&n.raw_native, "observer_failures")? == 0
                        && n.raw_native["last_observer_failure"].is_null(),
                    "Core requires actual successful observer completion",
                )?;
                let stats = &n.raw_native["backend_stats"];
                let (all_inputs, all_calls, all_roles) = graph_stats(stats)?;
                receipt.nn_inputs_completed = all_inputs
                    .checked_sub(n.startup_nn_inputs_completed)
                    .ok_or_else(|| invalid("NN cumulative inputs below startup snapshot"))?;
                receipt.nn_calls_completed = all_calls
                    .checked_sub(n.startup_nn_calls_completed)
                    .ok_or_else(|| invalid("NN cumulative calls below startup snapshot"))?;
                require(
                    receipt.nn_inputs_completed == receipt.nn_calls_completed
                        && all_roles.checked_sub(n.startup_role_inputs_completed)
                            == Some(n.completed_role_inputs),
                    "physical B1 graph inputs/completion or role completion differs",
                )?;
                receipt.nn_inputs_consumed = n.search_consumed_role_inputs;
                require(
                    work_count(t, "consumed_role_outputs")? == receipt.nn_inputs_consumed
                        && receipt
                            .proposer_tasks_completed
                            .checked_add(receipt.critic_tasks_completed)
                            .is_some_and(|completed| completed <= n.completed_role_inputs),
                    "native role completion/search consumption differs from observed search work",
                )?;
                receipt.physical_state = PalsPhysicalStateV3::Completed;
                if let Some(cuda) = lock.input.endpoints[index].cuda_model() {
                    let observed = validate_cuda_placement_witness(
                        cuda,
                        &n.raw_native["startup_probe"]["cuda_placement_witness"],
                    )?;
                    require(
                        observed == n.cuda_placement,
                        "Core CUDA placement projection differs from raw startup evidence",
                    )?;
                    if let Some(placement) = observed {
                        receipt.gpu_device=PalsObservedV3::Observed {
                            value:format!("cuda:{}",placement.device_id),
                            method:"pinned producer CUDA EP index exercised by exact pre-Run coverage and actual public/P/C neural kernel profiles; physical GPU UUID/model and VRAM peak unobserved".into(),
                        };
                    }
                }
            }
        }
    }
    Ok(receipt)
}

#[cfg(all(
    target_os = "linux",
    any(feature = "pals-collection-onnx", feature = "native-cuda")
))]
fn verify_external_profile(
    semantic_lock: &PalsInputLockV4,
    role: NativeEngineRole,
    source_root: &Path,
    registered_program: &Path,
    registered_cwd: &Path,
) -> Result<PalsExternalCpuRProfileAuditV3, ArenaError> {
    use rz_experiments::PalsCpuRSelectionV3;
    use rz_uci::pals_checker_profile::{PalsCheckerProfile, PalsCheckerSelection};

    semantic_lock.verify()?;
    let manifest = &semantic_lock.manifest;
    let PalsEngineV4::Pals(endpoint) = &manifest.engines[role_index(role)] else {
        return Err(invalid("external CPU_R profile requires a PALS endpoint"));
    };
    let endpoint = &endpoint.base;
    let PalsCpuRSelectionV3::ExternalUci(declaration) = &endpoint.cpu_r else {
        return Err(invalid(
            "own CPU_R selection cannot inherit an external profile",
        ));
    };
    require_public_artifact(&declaration.profile)?;
    require_public_artifact(&declaration.binary)?;
    require(
        source_root.is_absolute()
            && registered_program.is_absolute()
            && registered_cwd.is_absolute(),
        "external CPU_R profile root and registered program/cwd must be absolute",
    )?;
    let program = registered_program
        .to_str()
        .ok_or_else(|| invalid("registered CPU_R program is not UTF-8"))?;
    let cwd = registered_cwd
        .to_str()
        .ok_or_else(|| invalid("registered CPU_R cwd is not UTF-8"))?;
    // Validate only path names here. These are not executable/cwd opens; the
    // owner performs its independent pinned-FD verification before spawn.
    let mut registered_name = declaration.binary.clone();
    registered_name.path = program.trim_start_matches('/').into();
    require_public_artifact(&registered_name)?;
    let loaded = PalsCheckerProfile::load(
        &source_root.join(&declaration.profile.path),
        &declaration.profile.sha256,
    )
    .map_err(|error| invalid(format!("external CPU_R profile load: {error}")))?;
    let profile = loaded.profile();
    let registration = loaded.registration();
    require(
        loaded.canonical_sha256() == declaration.profile_canonical_sha256
            && registration.file_bytes == declaration.profile.bytes
            && profile.selection == PalsCheckerSelection::StockfishEmbeddedNnue
            && profile.program == program
            && profile.working_directory == cwd,
        "external CPU_R actual profile canonical/size/selection or registered paths differ",
    )?;
    require(
        profile.identity.binary_sha256 == declaration.binary.sha256
            && profile.identity.declared_source == declaration.binary.source
            && profile.identity.declared_license == declaration.binary.license,
        "external CPU_R loaded profile differs from independently registered binary/source/license",
    )?;
    require(
        declaration.resolver.version == rz_search::pals::value::MODEL_WDL_RESOLVER_VERSION
            && declaration.resolver.semantics_sha256
                == digest(rz_search::pals::value::MODEL_WDL_RESOLVER_SEMANTICS.as_bytes()),
        "external CPU_R resolver declaration differs from actual model-WDL semantics",
    )?;
    let policy = &declaration.policy;
    let option_number = |name: &str| -> Result<u32, ArenaError> {
        profile
            .identity
            .options
            .get(name)
            .and_then(|value| value.parse::<u32>().ok())
            .ok_or_else(|| invalid(format!("external CPU_R profile {name} cap missing")))
    };
    require(
        option_number("Threads")? <= policy.threads_max
            && option_number("Hash")? <= policy.hash_mib_max
            && u32::from(profile.max_depth) <= policy.max_depth
            && profile.max_prefix_plies as u64 <= u64::from(policy.max_prefix_plies)
            && profile.handshake_timeout_ms <= policy.handshake_max_ms
            && profile.max_task_wall_time_ms <= policy.task_wall_time_max_ms
            && profile.stop_grace_ms <= policy.stop_grace_max_ms
            && profile.shutdown_grace_ms <= policy.shutdown_grace_max_ms
            && profile.max_output_bytes as u64 <= policy.lifetime_output_bytes_max
            && profile.max_line_bytes as u64 <= policy.line_bytes_max,
        "external CPU_R actual profile exceeds registered inherited task/time/output ceilings",
    )?;
    // The loader enforces the exact first-selection option whitelist, empty
    // argv/assets, and unknown embedded model metadata. Preserve that complete
    // declaration with unknown observations; readyok cannot fill those fields.
    let profile_registration = serde_json::to_value(registration)
        .map_err(|error| invalid(format!("external CPU_R profile audit encoding: {error}")))?;
    Ok(PalsExternalCpuRProfileAuditV3 {
        domain: "rz-pals-external-cpu-r-profile-audit-v4/1",
        endpoint_id: endpoint.id.clone(),
        semantic_lock_sha256: semantic_lock.canonical_sha256.clone(),
        profile_file_sha256: loaded.file_sha256().into(),
        profile_canonical_sha256: loaded.canonical_sha256().into(),
        profile_file_bytes: declaration.profile.bytes,
        registered_binary_sha256: declaration.binary.sha256.clone(),
        registered_program: program.into(),
        registered_working_directory: cwd.into(),
        profile_registration,
        profile_file_verification_performed: true,
        binary_file_verification_performed: false,
        execution_admission_completed: false,
        child_spawned: false,
        options_application_observed: None,
        model_loading_observed: None,
        inherited_resource_join_observed: None,
        physical_shutdown_observed: None,
    })
}

#[cfg(target_os = "linux")]
fn validate_work_records(
    lock: &LockedPalsArenaLaunchV4,
    role: NativeEngineRole,
    startup: &[u8],
    termination: &[u8],
    session: &str,
    audited_native: Option<&PalsNativeSessionAuditV3>,
) -> Result<PalsProcessWorkAuditV4, ArenaError> {
    require(
        startup.len() <= MAX_JSON_BYTES && termination.len() <= MAX_JSON_BYTES,
        "work envelope byte budget exceeded",
    )?;
    let (s, t) = (json(startup)?, json(termination)?);
    lock.validate_wire_pair(role, &s, &t)?;
    let marker = validate_followup_marker_pair(lock, role, &s, &t)?;
    let followup_lifecycle = followup::validate_lifecycle_pair(lock, role, &s, &t)?;
    let engine = &lock.endpoint_views[role_index(role)];
    let external = lock
        .input
        .external_cpu_r_binding(role_index(role))?
        .is_some();
    let native = matches!(
        &lock.input.endpoints[role_index(role)],
        PalsEndpointLaunchV3::OnnxCpu(_) | PalsEndpointLaunchV3::OnnxCuda(_)
    );
    let (sdomain, tdomain, kind) = match &lock.input.endpoints[role_index(role)] {
        PalsEndpointLaunchV3::OnnxCpu(_) | PalsEndpointLaunchV3::OnnxCuda(_) => (
            PALS_NATIVE_STARTUP_V4_DOMAIN,
            PALS_NATIVE_TERMINATION_V4_DOMAIN,
            "pals",
        ),
        PalsEndpointLaunchV3::LegalOrderMock(_) => (
            PALS_SEARCH_WORK_STARTUP_V4_DOMAIN,
            PALS_SEARCH_WORK_TERMINATION_V4_DOMAIN,
            "pals",
        ),
        PalsEndpointLaunchV3::OwnCpu { .. } => (
            PALS_SEARCH_WORK_STARTUP_V4_DOMAIN,
            PALS_SEARCH_WORK_TERMINATION_V4_DOMAIN,
            "cpu",
        ),
        PalsEndpointLaunchV3::ReferenceUci { .. } => {
            return Err(invalid(
                "external reference has no invented RoveZero work envelope",
            ));
        }
    };
    let pid = s["process_id"]
        .as_u64()
        .filter(|v| *v > 0 && *v <= i32::MAX as u64)
        .ok_or_else(|| invalid("work process PID invalid"))? as u32;
    require(
        session == format!("native-process-{pid}"),
        "work directory/PID mismatch",
    )?;
    for (value, domain, ending) in [(&s, sdomain, false), (&t, tdomain, true)] {
        require(
            value["schema_version"] == 4
                && value["domain"] == domain
                && value["endpoint_id"] == engine.id
                && value["launch_sha256"] == lock.sha256
                && value["process_id"] == pid
                && value["binary_sha256"] == engine.binary.sha256
                && value["service_exit_success"] == ending,
            "work envelope identity/start/end differs",
        )?;
        validate_pals_process_counter_units(&value["search_work"], kind, !ending, external, false)?;
    }
    if native {
        // Native worker counters have their own accepted/delivered lifecycle.
        if external {
            let audit = audited_native.ok_or_else(|| {
                invalid("external work requires production owner/supervisor native audit")
            })?;
            require(
                audit.endpoint_id == engine.id
                    && audit.process_id == pid
                    && audit.launch_sha256 == lock.sha256
                    && audit.startup_sha256 == digest(startup)
                    && audit.termination_sha256 == digest(termination)
                    && audit.raw_search_work == t["search_work"]
                    && audit.raw_native == t["native"]
                    && audit.physical_shutdown_confirmed
                    && audit.native_buffers_released
                    && audit.external_cpu_r_session.as_ref().is_some_and(|helper| {
                        helper.purpose == PalsExternalCpuRSessionPurposeV3::Game
                            && helper.parent_process_id == pid
                            && helper.startup_sha256 == audit.startup_sha256
                            && helper.termination_sha256 == audit.termination_sha256
                    }),
                "external work and validated native/helper session differ",
            )?;
        } else {
            validate_pals_native_records_inner(
                lock,
                role,
                startup,
                termination,
                session,
                None,
                true,
            )?;
        }
    }
    Ok(PalsProcessWorkAuditV4 {
        base: PalsProcessWorkAuditV3 {
            endpoint_id: engine.id.clone(),
            process_id: pid,
            startup_sha256: digest(startup),
            termination_sha256: digest(termination),
            startup_bytes: startup.len() as u64,
            termination_bytes: termination.len() as u64,
            search_work: t["search_work"].clone(),
        },
        marker,
        followup_lifecycle,
    })
}

#[cfg(target_os = "linux")]
fn project_core_pgn(
    lock: &LockedPalsArenaLaunchV4,
    native_artifact: &ArtifactRef,
) -> Result<(ArtifactRef, PalsPgnProvenanceV3), ArenaError> {
    // Generated native artifacts use source as an execution description. Core's
    // ArtifactRef contract requires a public producer URL instead; retain the
    // original reference and description without changing the recorded bytes.
    let runner = &lock.input.runner;
    let mut artifact = native_artifact.clone();
    artifact.source = runner.source_url.clone();
    artifact.validate()?;
    let provenance = PalsPgnProvenanceV3 {
        native_artifact: native_artifact.clone(),
        producer_source_url: runner.source_url.clone(),
        producer_source_commit: runner.source_commit.clone(),
        producer_binary_sha256: runner.binary.sha256.clone(),
        description: "Generated match PGN from the pinned Fastchess execution; the public URL identifies the producer's source repository, not a public download location for this local PGN. The original native reference preserves its execution description, path, digest, size and license.".into(),
    };
    Ok((artifact, provenance))
}

#[derive(Clone, Debug, Serialize)]
pub struct PalsCoreAssemblyV4 {
    pub domain: String,
    pub arena_launch_sha256: String,
    pub native_receipt: ArtifactRef,
    pub wall_time_ms: Option<u64>,
    pub cleanup_time_ms: Option<u64>,
    pub timing_scope: String,
    pub game_process_correspondence: String,
    pub work: Vec<PalsProcessWorkAuditV4>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pgn_provenance: Option<PalsPgnProvenanceV3>,
    pub core: Option<PalsRunReceiptV4>,
    pub assembly_error: Option<String>,
    pub training_executed: bool,
    /// Colab observations never replace Core's strict resource-limit checks.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resource_observations: Option<Vec<PalsArenaResourceObservationV4>>,
}

#[cfg(target_os = "linux")]
pub fn assemble_pals_core_receipt_v4(
    output: &PalsObservedPairOutputV4,
    cleanup_time_ms: Option<u64>,
) -> PalsCoreAssemblyV4 {
    let native = &output.native;
    let owner = native.launch_owner();
    let lock = owner.spec();
    let mut assembly = PalsCoreAssemblyV4 { domain: "rz-pals-core-assembly-v4/1".into(), arena_launch_sha256: lock.sha256.clone(),
        native_receipt: native.receipt_artifact.clone(), wall_time_ms: Some(output.wall_time_ms), cleanup_time_ms,
        timing_scope: "UCI preflight through mandatory arena postcheck/receipt; snapshot preparation separate; cleanup is supervisor-owned measured interval".into(),
        game_process_correspondence: "pinned Fastchess 1.8.2 source: one synchronous worker, restart=on, recover=false; TRACE/renderer game id and color plus finished-white-exit-black-exit windows; separate native/owned-group completion gates required".into(),
        work: vec![], pgn_provenance: None, core: None, assembly_error: None, training_executed: false,
        resource_observations: None };
    let result: Result<PalsRunReceiptV4, ArenaError> = (|| {
        lock.input.require_reap_status_runner()?;
        lock.input.require_actual_native_epoch()?;
        for role in [NativeEngineRole::Baseline, NativeEngineRole::Candidate] {
            assembly.work.extend(collect_process_work_with_audits(
                lock,
                role,
                &owner.snapshot.directory,
                &native.receipt.provider_sessions,
            )?);
        }
        let cleanup = cleanup_time_ms.ok_or_else(|| {
            invalid("cleanup interval unobserved; original arena receipt retained")
        })?;
        let receipts = &native.receipt;
        require(
            receipts.integration_checks_passed
                && receipts.cleanup_verified
                && !receipts.unresolved_owner_retained,
            "arena integration/process/physical gates failed; original failure is preserved",
        )?;
        let pgn = receipts
            .pgn_audit
            .as_ref()
            .ok_or_else(|| invalid("Rules PGN audit absent"))?;
        let clock = receipts
            .clock_audit
            .as_ref()
            .ok_or_else(|| invalid("whole-game clock observation absent"))?;
        require(
            pgn.games.len() == 2 && clock.games.len() == 2,
            "two exchanged audited games required",
        )?;
        let exits =
            crate::native_exit::validate_endpoint_exit_trace(&native.process().stdout, &[], 4)?;
        let mut known = BTreeSet::new();
        for work in &assembly.work {
            require(
                known.insert(work.process_id),
                "work process reused across endpoints",
            )?;
        }
        let known_work_pids: [BTreeSet<u32>; 2] = std::array::from_fn(|index| {
            let id = lock.input.semantic_lock.manifest.engines[index].id();
            assembly
                .work
                .iter()
                .filter(|w| w.endpoint_id == id)
                .map(|w| w.process_id)
                .collect()
        });
        let mapped = validate_pals_game_process_trace_context(
            lock,
            &native.process().stdout,
            &known_work_pids,
        )?;
        let per_role = mapped.map(|pids| pids.to_vec());
        for role in [NativeEngineRole::Baseline, NativeEngineRole::Candidate] {
            let index = role_index(role);
            require(
                per_role[index].len() == 2 && per_role[index].iter().all(|pid| exits.contains(pid)),
                "work identity/normal exit game correspondence incomplete",
            )?;
            let view = &lock.endpoint_views[index];
            let options = view
                .requested_options
                .iter()
                .map(|(key, value)| {
                    crate::external_uci::resolve_asset_tokens(value, view, &owner.snapshot.pins)
                        .map(|v| (key.clone(), v))
                })
                .collect::<Result<BTreeMap<_, _>, _>>()?;
            crate::external_uci::audit_external_game_protocol(
                &native.process().stdout,
                view,
                &options,
                &per_role[index],
                crate::external_uci::preflight_advertisement(
                    &receipts.external_preflight,
                    &view.id,
                )?,
            )?;
        }
        let native_pgn_artifact = receipts
            .artifacts
            .iter()
            .find(|a| a.path.ends_with("/match.pgn"))
            .ok_or_else(|| invalid("retained PGN artifact absent"))?;
        let (pgn_artifact, provenance) = project_core_pgn(lock, native_pgn_artifact)?;
        assembly.pgn_provenance = Some(provenance);
        let m = &lock.input.semantic_lock.manifest;
        let mut failures = BTreeSet::new();
        if output.wall_time_ms > m.pilot.wall_time_max_ms {
            failures.insert(PalsRunFailureV3::WallDeadline);
        }
        if cleanup > m.pilot.cleanup_max_ms {
            failures.insert(PalsRunFailureV3::CleanupTimeout);
        }
        let mut games = vec![];
        for (index, game) in pgn.games.iter().enumerate() {
            require(
                clock.games.iter().any(|c| c.game_id == game.game_id),
                "clock/game identity differs",
            )?;
            let mut engines = vec![];
            for role in [NativeEngineRole::Baseline, NativeEngineRole::Candidate] {
                let ri = role_index(role);
                let id = m.engines[ri].id();
                let pid = per_role[ri][index];
                let work = assembly
                    .work
                    .iter()
                    .find(|w| w.endpoint_id == id && w.process_id == pid);
                if game.classification != "engine_loss"
                    && let Some(work) = work
                {
                    let plies = game.uci_moves.len() as u64;
                    let expected_go = if game.white_engine == id {
                        plies / 2 + plies % 2
                    } else {
                        plies / 2
                    };
                    require(
                        work_count(&work.search_work, "go_invocations")? == expected_go,
                        "per-game work invocation count differs from audited moves",
                    )?;
                }
                let nn = receipts
                    .provider_sessions
                    .iter()
                    .find(|n| n.endpoint_id == id && n.process_id == pid);
                let preflight = receipts
                    .external_preflight
                    .iter()
                    .find(|p| p.engine_id == id)
                    .ok_or_else(|| invalid("pinned UCI preflight absent"))?;
                let endpoint = endpoint_v4_receipt(
                    output,
                    role,
                    work,
                    nn,
                    preflight,
                    cleanup,
                    PalsGameCleanupScopeV4 {
                        process_id: pid,
                        charge_pair_streams: index == 0 && ri == 0,
                    },
                )?;
                record_endpoint_search_failure(&endpoint.base, &mut failures);
                engines.push(endpoint);
            }
            let (result, termination, failed_endpoint) = core_game_outcome(game)?;
            if result == PalsResultV3::Incomplete {
                failures.insert(PalsRunFailureV3::IncompleteGame);
            }
            games.push(PalsGameReceiptV4 {
                game_index: index as u32,
                white_endpoint: game.white_engine.clone(),
                black_endpoint: game.black_engine.clone(),
                base_ms: m.pilot.base_ms,
                increment_ms: m.pilot.increment_ms,
                plies: u32::try_from(game.uci_moves.len())
                    .map_err(|_| invalid("ply count overflow"))?,
                result,
                termination,
                failed_endpoint,
                engines: engines
                    .try_into()
                    .map_err(|_| invalid("two endpoint work receipts required"))?,
                pgn: pgn_artifact.clone(),
            });
        }
        let output_bytes = observed_output_bytes(output, &assembly.work)?;
        let core = PalsRunReceiptV4 {
            schema_version:4,
            domain: PALS_RECEIPT_V4_DOMAIN.into(),
            run_id: m.run_id.clone(),
            pair_id: m.pair_id.clone(),
            lock_sha256: lock.input.semantic_lock.canonical_sha256.clone(),
            training_executed: false,
            training: PalsObservedV3::Unknown,
            output_bytes: output_bytes.map_or(PalsObservedV3::Unknown,|value|PalsObservedV3::Observed {value, method: if lock.input.followup_execution.as_ref().is_some_and(|b|b.archive_output.is_some()) {"actual supervisor streams plus unique retained evidence lengths and actual archive write extents, each process-lifetime owner counted once; commit retention, integrity reads and runtime/input snapshots are distinct".into()} else {"actual observed supervisor stream bytes plus retained receipt/evidence file lengths counted once per exact path, including separately read CPU/mock work envelopes; pinned inputs and separately capped runtime caches are outside this output-evidence scope".into()}}),
            policy_failures:BTreeSet::new(),
            wall_time_ms: output.wall_time_ms,
            cleanup_time_ms: cleanup,
            pair_eligible: lock.input.resource_profile.is_none() && output_bytes.is_some() && failures.is_empty() && games.iter().all(|game|game.engines.iter().all(endpoint_admitted)),
            failures,
            games,
        };
        core.validate_against(&lock.input.semantic_lock)?;
        Ok(core)
    })();
    match result {
        Ok(core) => assembly.core = Some(core),
        Err(error) => assembly.assembly_error = Some(error.to_string()),
    }
    if lock.input.resource_profile.is_some() {
        match lock.resource_observations.lock() {
            Ok(observations) => assembly.resource_observations = Some(observations.clone()),
            Err(_) => {
                assembly.core = None;
                assembly.assembly_error = Some(
                    "Colab resource observation owner poisoned; original evidence retained".into(),
                );
            }
        }
    }
    assembly
}

#[cfg(target_os = "linux")]
pub fn save_pals_core_assembly_v4(
    output: &PalsObservedPairOutputV4,
    assembly: &PalsCoreAssemblyV4,
) -> Result<ArtifactRef, ArenaError> {
    use cap_fs_ext::{FollowSymlinks, OpenOptionsFollowExt};
    use std::io::Write;
    let owner = output.native.launch_owner();
    require(
        assembly.arena_launch_sha256 == owner.spec().sha256
            && assembly.native_receipt == output.native.receipt_artifact,
        "assembly provenance changed",
    )?;
    let bytes = serde_json::to_vec(assembly).map_err(|e| invalid(e.to_string()))?;
    require(
        bytes.len() <= V4_METADATA_CAP as usize,
        "V4 core assembly reservation exceeds 4MiB",
    )?;
    require(
        output
            .native
            .receipt
            .process
            .watched_artifact_bytes
            .is_some_and(|used| {
                used.checked_add(output.native.receipt_artifact.bytes)
                    .and_then(|n| n.checked_add(bytes.len() as u64))
                    .is_some_and(|n| n <= owner.spec().input.budget.max_artifact_bytes)
            }),
        "core assembly output budget unavailable",
    )?;
    // Original provider artifacts may already include their metadata. Reserve
    // all work metadata again conservatively rather than undercount CPU/mock
    // files absent from the generic NN-only artifact list.
    let required = output
        .native
        .receipt
        .artifacts
        .iter()
        .try_fold(output.native.receipt_artifact.bytes, |sum, a| {
            sum.checked_add(a.bytes)
        })
        .and_then(|sum| {
            assembly.work.iter().try_fold(sum, |n, w| {
                n.checked_add(w.startup_bytes)?
                    .checked_add(w.termination_bytes)
            })
        })
        .and_then(|sum| sum.checked_add(bytes.len() as u64));
    require(
        required.is_some_and(|n| n <= owner.spec().input.budget.max_output_bytes),
        "core assembly conservative evidence reservation exceeds output budget",
    )?;
    // Unknown write extent is retained as Unknown. The declared per-session
    // prewrite reservation is used only to save bounded failure evidence.
    let output_reservation = match observed_output_bytes(output, &assembly.work)? {
        Some(bytes) => bytes,
        None => metadata_output_bytes(output, &assembly.work)?
            .checked_add(
                owner
                    .spec()
                    .input
                    .followup_execution
                    .as_ref()
                    .and_then(|b| b.archive_output.as_ref())
                    .ok_or_else(|| invalid("unknown archive extent lacks an original reservation"))?
                    .archive_write_bytes_max_per_endpoint
                    .iter()
                    .try_fold(0u64, |total, bytes| total.checked_add(*bytes))
                    .ok_or_else(|| invalid("archive output reservation overflow"))?,
            )
            .ok_or_else(|| invalid("failure evidence output reservation overflow"))?,
    };
    require(
        output_reservation
            .checked_add(bytes.len() as u64)
            .is_some_and(|n| n <= owner.spec().input.budget.max_output_bytes),
        "V4 observed streams and retained evidence plus core assembly exceed output budget",
    )?;
    let mut options = cap_std::fs::OpenOptions::new();
    options
        .write(true)
        .create_new(true)
        .follow(FollowSymlinks::No);
    let mut file = owner
        .snapshot
        .directory
        .open_with("pals-core-assembly.v4.json", &options)
        .map_err(|e| ArenaError::Io(e.to_string()))?;
    file.write_all(&bytes)
        .and_then(|()| file.sync_all())
        .map_err(|e| ArenaError::Io(e.to_string()))?;
    Ok(ArtifactRef { path: format!("{}/pals-core-assembly.v4.json", owner.snapshot.output_directory), sha256: digest(&bytes), bytes: bytes.len() as u64,
        source: "PALS V4 actual arena work/clock/physical-lifetime assembly; raw native receipt retained".into(), license: "MIT execution evidence; external asset rights remain separate".into() })
}

#[cfg(test)]
mod tests {
    use super::*;
    pub(super) fn fixture() -> PalsArenaLaunchV4 {
        let old = super::super::tests::fixture();
        let mut base = match &old.semantic_lock.manifest.engines[0] {
            PalsEngineV3::Pals(e) => e.as_ref().clone(),
            _ => unreachable!(),
        };
        let policies = PalsPoliciesV4 {
            resolver: PalsResolverPolicyV4::OwnRaw,
            resolver_identity: PalsComponentV3 {
                semantic_id: PALS_V4_OWN_RAW_RESOLVER_SEMANTICS.into(),
                implementation_sha256: hex(
                    rz_search::pals::engine::compiled_resolver_implementation_sha256(),
                ),
                options: BTreeMap::new(),
            },
            resolver_semantics_sha256: PalsResolverPolicyV4::OwnRaw.expected_semantics_sha256(),
            checker: PalsCheckerPolicyV4::Own,
            recheck: PalsRecheckPolicyV4::Disabled,
            iterative_repair: PalsIterativeRepairPolicyV4::Disabled,
            cold_archive: PalsColdArchivePolicyV4::Disabled,
            paused_stack: PalsPausedStackPolicyV4::Disabled,
            cuda_warm: PalsCudaWarmPolicyV4::Disabled,
        };
        let identity = PalsPolicyIdentityV4::expected_for_policies(&policies).unwrap();
        base.search.semantic_id = identity.search_identity.clone();
        base.search.implementation_sha256 =
            hex(rz_search::pals::engine::compiled_search_implementation_sha256());
        base.model.input_schema = PalsModelProfile::LegacySummaryV1.encoding_schema().into();
        let endpoint = PalsEndpointV4 {
            base,
            policies,
            policy_identity: identity,
            model_v2: PalsModelSpecificationV4 {
                profile: PalsModelProfileV4::LegacySummaryV1,
                model_schema: "rovezero.pals-model.v1".into(),
                model_semantics: "rovezero.pals-model.v1".into(),
                encoding_schema: "rovezero.pals-board-records.v1".into(),
                model_identity: "explicit-legal-order-role-mock-v1".into(),
                model_semantics_sha256: "d".repeat(64),
                encoding_sha256: "e".repeat(64),
                max_records: 128,
                max_line_plies: 0,
                provenance: PalsModelProvenanceV4 {
                    purpose: PalsArtifactPurposeV4::ArenaCandidate,
                    training: PalsTrainingProvenanceV4::Untrained,
                },
            },
        };
        let own = match &old.semantic_lock.manifest.engines[1] {
            PalsEngineV3::OwnCpu(e) => e.clone(),
            _ => unreachable!(),
        };
        let mut manifest = PalsRunManifestV4 {
            schema_version: 4,
            run_id: old.semantic_lock.manifest.run_id.clone(),
            pair_id: old.semantic_lock.manifest.pair_id.clone(),
            contract_revision: PALS_V4_CONTRACT_REVISION.into(),
            rules_profile: old.semantic_lock.manifest.rules_profile.clone(),
            purpose: PalsRunPurposeV4::ArenaPilot,
            comparison: PalsComparisonV3::System,
            declared_changes: BTreeSet::new(),
            training_executed: false,
            engines: [
                PalsEngineV4::Pals(Box::new(endpoint)),
                PalsEngineV4::OwnCpu(own),
            ],
            resources: old.semantic_lock.manifest.resources.clone(),
            pilot: old.semantic_lock.manifest.pilot.clone(),
            output_bytes_max: 64 * 1024 * 1024,
        };
        manifest.declared_changes = manifest.actual_change_axes();
        let mut runner = old.runner;
        runner.binary.sha256 = PALS_CLOCK_REAP_RUNNER_SHA256.into();
        runner.binary.bytes = PALS_CLOCK_REAP_RUNNER_BYTES;
        let patch = runner.dirty_patch.as_mut().unwrap();
        patch.sha256 = PALS_CLOCK_REAP_PATCH_SHA256.into();
        patch.bytes = PALS_CLOCK_REAP_PATCH_BYTES;
        PalsArenaLaunchV4 {
            domain: PALS_ARENA_V4_DOMAIN.into(),
            semantic_lock: manifest.lock().unwrap(),
            runner,
            opening_artifact: old.opening_artifact,
            endpoints: old.endpoints,
            budget: old.budget,
            followup_execution: None,
            resource_profile: None,
        }
    }
    #[test]
    #[cfg(target_os = "linux")]
    fn v4_cleanup_owner_uses_four_audited_game_pids_and_rejects_reuse() {
        // Authored CPU fixture only: the same production renderer/exit reader
        // binds PID to endpoint and restarted game before receipt projection.
        let lock = fixture().lock().unwrap();
        let pids = [[62406_u32, 63287], [62556_u32, 63437]];
        let trace = |message: String| {
            format!(
                "[TRACE ] [12:34:56.123456] <{:>20}> fastchess --- {message}",
                1
            )
        };
        let render = |pids: [[u32; 2]; 2]| {
            let mut rows = vec![];
            for (game, white) in lock
                .input
                .semantic_lock
                .manifest
                .pilot
                .white_order
                .iter()
                .enumerate()
            {
                let white_role = lock
                    .input
                    .semantic_lock
                    .manifest
                    .engines
                    .iter()
                    .position(|engine| engine.id() == white)
                    .unwrap();
                let black = lock.input.semantic_lock.manifest.engines[1 - white_role].id();
                let number = game + 1;
                rows.extend([
                    trace(format!(
                        "Game {number} between {white} and {black} starting"
                    )),
                    format!("Started game {number} of 2 ({white} vs {black})"),
                    trace(format!(
                        "Game {number} between {white} and {black} finished"
                    )),
                    trace(format!("Game {number} finished with result 1/2-1/2")),
                    format!("Finished game {number} ({white} vs {black}): 1/2-1/2 {{Draw}}"),
                    trace(format!(
                        "Process with pid: {} terminated with status: 0",
                        pids[white_role][game]
                    )),
                    trace(format!(
                        "Process with pid: {} terminated with status: 0",
                        pids[1 - white_role][game]
                    )),
                ]);
            }
            format!("{}\n", rows.join("\n")).into_bytes()
        };
        let known = pids.map(BTreeSet::from);
        let stdout = render(pids);
        let mapped = validate_pals_game_process_trace_context(&lock, &stdout, &known).unwrap();
        let exited = crate::native_exit::validate_endpoint_exit_trace(&stdout, &[], 4).unwrap();
        assert_eq!(mapped, pids);
        let mut owners = BTreeSet::new();
        for role_pids in &mapped {
            for &pid in role_pids {
                assert!(exited.contains(&pid));
                let owner =
                    cleanup_game_owner_v4(60000, pid, &lock.sha256, Some(pid), Some(pid)).unwrap();
                assert!(owners.insert(owner.clone()));
                // ReferenceUci has no invented work/native envelope: it still
                // receives this exact renderer/normal-exit PID from the caller.
                assert_eq!(
                    owner,
                    cleanup_game_owner_v4(60000, pid, &lock.sha256, None, None).unwrap()
                );
            }
        }
        assert_eq!(owners.len(), 4);
        let mut duplicate = pids;
        duplicate[0][1] = duplicate[0][0];
        assert!(
            validate_pals_game_process_trace_context(&lock, &render(duplicate), &known).is_err()
        );
        for (pid, work, native) in [
            (0, None, None),
            (60000, None, None),
            (i32::MAX as u32 + 1, None, None),
            (62406, Some(62556), Some(62406)),
            (62406, Some(62406), Some(62556)),
        ] {
            assert!(cleanup_game_owner_v4(60000, pid, &lock.sha256, work, native).is_err());
        }
    }
    #[test]
    #[cfg(target_os = "linux")]
    fn v4_cleanup_shared_pair_streams_are_charged_once() {
        let mut observed = (0_u64, 0_u64);
        for game in 0..2 {
            for role in 0..2 {
                let charged = cleanup_stream_bytes_v4(
                    PalsGameCleanupScopeV4 {
                        process_id: 100 + (game * 2 + role) as u32,
                        charge_pair_streams: game == 0 && role == 0,
                    },
                    100_777,
                    13,
                );
                observed.0 += charged.0;
                observed.1 += charged.1;
                if game != 0 || role != 0 {
                    assert_eq!(charged, (0, 0));
                }
            }
        }
        assert_eq!(observed, (100_777, 13));
        assert!(PROCESS_CLEANUP_METHOD_V4.len() <= 256);
        assert!(!PROCESS_CLEANUP_METHOD_V4.contains('\n'));
    }
    #[test]
    fn v4_model_observation_method_passes_shared_receipt_validation() {
        // No model/game is executed. The public incomplete-receipt reader uses
        // the same generic observation gate as PalsRuntimeChecksV4::model.
        let lock = fixture().semantic_lock;
        let mut receipt = PalsRunReceiptV4 {
            schema_version: 4,
            domain: PALS_RECEIPT_V4_DOMAIN.into(),
            run_id: lock.manifest.run_id.clone(),
            pair_id: lock.manifest.pair_id.clone(),
            lock_sha256: lock.canonical_sha256.clone(),
            training_executed: false,
            training: PalsObservedV3::Unknown,
            output_bytes: PalsObservedV3::Observed {
                value: 0,
                method: MODEL_OBSERVATION_METHOD_V4.into(),
            },
            wall_time_ms: 1,
            cleanup_time_ms: 1,
            pair_eligible: false,
            failures: BTreeSet::new(),
            policy_failures: BTreeSet::new(),
            games: vec![],
        };
        assert!(receipt.validate_against(&lock).is_ok());
        let old = "loaded backend profile/configuration/epoch plus actual registered semantics and compiled model boundary SHA getters compared with independently pinned manifest; original graph/runtime/checkpoint/export bytes validated; diagnostic export rejected before spawn";
        assert_eq!(old.len(), 258);
        for invalid in [
            old.to_string(),
            "readyok".into(),
            "ReAdYoK".into(),
            "x".repeat(257),
            "é".repeat(129),
            "independent\nmethod".into(),
        ] {
            receipt.output_bytes = PalsObservedV3::Observed {
                value: 0,
                method: invalid,
            };
            let error = receipt.validate_against(&lock).unwrap_err();
            assert!(
                error.to_string().contains(
                    "observation requires an independent measurement method, not readyok"
                )
            );
        }
        for boundary in ["x".repeat(256), "é".repeat(128)] {
            receipt.output_bytes = PalsObservedV3::Observed {
                value: 0,
                method: boundary,
            };
            assert!(receipt.validate_against(&lock).is_ok());
        }
    }
    pub(super) fn colab_fixture() -> PalsArenaLaunchV4 {
        let mut input = fixture();
        let mut manifest = input.semantic_lock.manifest.clone();
        let mut candidate = manifest.engines[0].clone();
        let PalsEngineV4::Pals(endpoint) = &mut candidate else {
            unreachable!()
        };
        endpoint.base.id = manifest.engines[1].id().into();
        manifest.engines[1] = candidate;
        manifest.declared_changes = manifest.actual_change_axes();
        input.semantic_lock = manifest.lock().unwrap();
        input.endpoints[1] = input.endpoints[0].clone();
        input.resource_profile = Some(PalsArenaResourceProfileV4::ColabObserved {});
        input
    }
    #[cfg(all(target_os = "linux", feature = "native-cuda"))]
    fn cudnn_conv_pilot_fixture() -> PalsArenaLaunchV4 {
        let (native, _) = super::super::tests::cuda_control_fixture();
        let old = match &native.semantic_lock.manifest.engines[0] {
            PalsEngineV3::Pals(e) => e,
            _ => unreachable!(),
        };
        let mut input = colab_fixture();
        input.budget = native.budget;
        for i in 0..2 {
            let config = if i == 0 {
                PalsModelConfig::baseline()
            } else {
                PalsModelConfig::full_line_interaction_v2()
            };
            let PalsEngineV4::Pals(e) = &mut input.semantic_lock.manifest.engines[i] else {
                unreachable!()
            };
            e.base.model = old.model.clone();
            e.base.model.input_schema = config.profile.encoding_schema().into();
            e.base.model.implementation_sha256 =
                hex(rz_eval::pals_model::compiled_model_boundary_sha256());
            e.base.pools = old.pools.clone();
            e.model_v2.profile = if i == 0 {
                PalsModelProfileV4::LegacySummaryV1
            } else {
                PalsModelProfileV4::FullLineInteractionV2
            };
            e.model_v2.model_schema = if i == 0 {
                "rovezero.pals-model.v1"
            } else {
                "rovezero.pals-model.v2"
            }
            .into();
            e.model_v2.model_semantics = config.profile.model_semantics().into();
            e.model_v2.model_semantics_sha256 = hex(config.profile.registered_semantics_sha256());
            e.model_v2.encoding_schema = config.profile.encoding_schema().into();
            e.model_v2.encoding_sha256 =
                hex(rz_uci::pals_native::pals_fresh_encoding_semantic_digest_for_config(&config));
            e.model_v2.max_line_plies = if i == 0 { 0 } else { 256 };
            let mut cuda = native.endpoints[0].cuda_model().unwrap().clone();
            cuda.model.encoding_semantic_sha256 =
                hex(rz_uci::pals_native::pals_rules_encoding_semantic_digest_for_config(&config));
            cuda.model.adapter_source_sha256 =
                hex(rz_uci::pals_native::pals_native_source_digest());
            e.model_v2.model_identity = format!("pals-onnx-pc-fp32-{}", cuda.model.export.sha256);
            let mut conv = PalsCuDnnConvBindingV1 {
                profile: PalsCuDnnConvProfileV1::DefaultHostControlV1,
                backend_source_sha256: digest(include_bytes!("../../../rz-eval/src/pals_onnx.rs")),
                descriptor_sha256: "0".repeat(64),
            };
            let checkpoint = match &e.base.model.weights {
                PalsWeightIdentityV3::Untrained { artifact, .. }
                | PalsWeightIdentityV3::Trained { artifact, .. } => &artifact.sha256,
                _ => unreachable!(),
            };
            conv.descriptor_sha256 =
                digest(&serde_json::to_vec(&conv.descriptor(&cuda, checkpoint)).unwrap());
            cuda.cuda_control.as_mut().unwrap().cudnn_conv_search = Some(conv);
            input.endpoints[i] = PalsEndpointLaunchV3::OnnxCuda(Box::new(cuda));
            input.semantic_lock.manifest.resources[i] =
                native.semantic_lock.manifest.resources[0].clone();
        }
        input.semantic_lock.manifest.declared_changes =
            input.semantic_lock.manifest.actual_change_axes();
        input.semantic_lock = input.semantic_lock.manifest.lock().unwrap();
        input
    }
    #[cfg(all(target_os = "linux", feature = "native-cuda"))]
    #[test]
    fn pals_cudnn_conv_v4_argv_roundtrip_and_mode_mixing() {
        let input = cudnn_conv_pilot_fixture();
        let locked = input.lock().unwrap();
        let encoded = serde_json::to_string(&input).unwrap();
        let roundtrip = PalsArenaLaunchV4::from_json(&encoded)
            .unwrap()
            .lock()
            .unwrap();
        assert_eq!(locked.sha256(), roundtrip.sha256());
        for i in 0..2 {
            let args = &locked.endpoint_views[i].arguments;
            assert_eq!(args, &roundtrip.endpoint_views[i].arguments);
            assert_eq!(
                args.iter()
                    .filter(|a| a.starts_with("--pals-cudnn-conv-search="))
                    .count(),
                1
            );
            assert!(
                args.iter()
                    .any(|a| a == "--pals-cudnn-conv-search=default-host-control-v1")
            );
            assert!(
                args.iter()
                    .any(|a| a == "--pals-device-public-memory=false")
            );
        }
        let mut none = input.clone();
        for endpoint in &mut none.endpoints {
            let PalsEndpointLaunchV3::OnnxCuda(c) = endpoint else {
                unreachable!()
            };
            c.cuda_control.as_mut().unwrap().cudnn_conv_search = None;
        }
        let old = none.lock().unwrap();
        assert_ne!(old.sha256(), locked.sha256());
        assert!(
            !serde_json::to_string(&none)
                .unwrap()
                .contains("cudnn_conv_search")
        );
        for i in 0..2 {
            let expected: Vec<_> = locked.endpoint_views[i]
                .arguments
                .iter()
                .filter(|a| !a.starts_with("--pals-cudnn-conv-search="))
                .collect();
            assert_eq!(
                old.endpoint_views[i].arguments.iter().collect::<Vec<_>>(),
                expected
            );
        }
        let mut one = input.clone();
        let PalsEndpointLaunchV3::OnnxCuda(c) = &mut one.endpoints[1] else {
            unreachable!()
        };
        c.cuda_control.as_mut().unwrap().cudnn_conv_search = None;
        assert!(one.lock().is_err());
        let mut mixed = input.clone();
        mixed.followup_execution = Some(PalsFollowupExecutionBindingV4 {
            domain: "rz-pals-followup-execution-v4/1".into(),
            schema_version: 4,
            archive_output: None,
            endpoints: [
                Some(PalsEndpointExecutionBindingV4 {
                    owner_methods: PalsFollowupOwnerMethodsV4 {
                        producer_domain: "metadata-only".into(),
                        producer_schema_version: 4,
                        producer_implementation_sha256: "a".repeat(64),
                        trace_capacity: 1,
                        trace_bytes_max: 1024,
                        repair: None,
                        paused_stack: None,
                        cold_archive: None,
                        cuda_warm: None,
                    },
                    cold_archive: None,
                    cuda_warm: None,
                }),
                None,
            ],
        });
        assert!(
            mixed
                .endpoint(0)
                .unwrap_err()
                .to_string()
                .contains("DEFAULT host runtime candidate")
        );
        let mut namespace = json(encoded.as_bytes()).unwrap();
        namespace["endpoints"][0]["configuration"]["cuda_control"]["cudnn_conv_search"]["profile"] =
            "default-device-warm-v1".into();
        assert!(PalsArenaLaunchV4::from_json(&namespace.to_string()).is_err());
    }
    #[test]
    fn colab_profile_is_explicit_and_absent_strict_wire_bytes_are_preserved() {
        let input = fixture();
        let canonical = crate::canonical_sha256(&input).unwrap();
        let mut value = serde_json::to_value(&input).unwrap();
        assert!(value.get("resource_profile").is_none());
        let decoded = PalsArenaLaunchV4::from_json(&value.to_string()).unwrap();
        assert_eq!(crate::canonical_sha256(&decoded).unwrap(), canonical);
        value["resource_profile"] = serde_json::Value::Null;
        let decoded = PalsArenaLaunchV4::from_json(&value.to_string()).unwrap();
        assert_eq!(crate::canonical_sha256(&decoded).unwrap(), canonical);
        let cloud = colab_fixture();
        cloud.validate().unwrap();
        let cloud_bytes = serde_json::to_value(&cloud).unwrap();
        assert_eq!(
            cloud_bytes["resource_profile"],
            serde_json::json!({"profile":"colab_observed"})
        );
        let cloud_lock = cloud.lock().unwrap();
        assert_eq!(
            LockedPalsArenaLaunchV4::from_json(&cloud_lock.to_json().unwrap())
                .unwrap()
                .sha256(),
            cloud_lock.sha256()
        );
        assert_ne!(cloud_lock.sha256(), canonical);
    }
    #[test]
    fn v4_runner_requires_registered_version_hash_and_bytes_together() {
        let original = fixture();
        let old_lock = original.lock().unwrap();
        let mut cloud = colab_fixture();
        cloud.runner.version = PALS_CLOCK_REAP_COLAB_RUNNER_VERSION.into();
        cloud.runner.binary.sha256 = PALS_CLOCK_REAP_COLAB_RUNNER_SHA256.into();
        cloud.runner.binary.bytes = PALS_CLOCK_REAP_COLAB_RUNNER_BYTES;
        let lock = cloud.lock().unwrap();
        assert_eq!(
            LockedPalsArenaLaunchV4::from_json(&lock.to_json().unwrap())
                .unwrap()
                .sha256(),
            lock.sha256()
        );
        let mut wrong_version = cloud.clone();
        wrong_version.runner.version = crate::FASTCHESS_VERSION.into();
        assert!(wrong_version.lock().is_err());
        let mut wrong_hash = cloud.clone();
        wrong_hash.runner.binary.sha256 = PALS_CLOCK_REAP_RUNNER_SHA256.into();
        assert!(wrong_hash.lock().is_err());
        let mut wrong_bytes = cloud.clone();
        wrong_bytes.runner.binary.bytes = PALS_CLOCK_REAP_RUNNER_BYTES;
        assert!(wrong_bytes.lock().is_err());
        let mut wrong_patch = cloud;
        wrong_patch.runner.dirty_patch.as_mut().unwrap().bytes -= 1;
        assert!(wrong_patch.lock().is_err());
        assert_eq!(
            original.lock().unwrap().to_json().unwrap(),
            old_lock.to_json().unwrap()
        );
    }
    #[test]
    fn cloud_lane_does_not_expand_external_or_inconsistent_resource_admission() {
        let mut old = fixture();
        old.resource_profile = Some(PalsArenaResourceProfileV4::ColabObserved {});
        assert!(old.validate().is_err()); // The second endpoint is OwnCpu, not PALS.
        let mut value = serde_json::to_value(colab_fixture()).unwrap();
        value["resource_profile"] = serde_json::json!({"profile":"strict_local"});
        assert!(PalsArenaLaunchV4::from_json(&value.to_string()).is_err());
        value["resource_profile"] =
            serde_json::json!({"profile":"colab_observed","skip_physical_completion":true});
        assert!(PalsArenaLaunchV4::from_json(&value.to_string()).is_err());
        let mut cloud = colab_fixture();
        let mut manifest = cloud.semantic_lock.manifest.clone();
        manifest.resources[0].swap_max_bytes = 1;
        manifest.resources[1].swap_max_bytes = 1;
        cloud.semantic_lock = manifest.lock().unwrap();
        assert!(cloud.validate().is_err());
    }
    #[test]
    fn v4_executes_actual_explicit_flags_and_preserves_v3_lock_bytes() {
        let old = super::super::tests::fixture();
        let historical = serde_json::to_string(&old.semantic_lock).unwrap();
        let input = fixture();
        let lock = input.lock().unwrap();
        let args = &lock.endpoint_views[0].arguments;
        for flag in [
            "--pals-arena-wire=v4",
            "--pals-model-profile=legacy_summary_v1",
            "--pals-resolver-policy=own-raw-restricted",
            "--pals-cpu-resume=completed-iteration",
            "--pals-cuda-private-warm=false",
        ] {
            assert!(args.iter().any(|arg| arg == flag));
        }
        assert!(
            !args
                .iter()
                .any(|arg| arg.starts_with("--pals-post-repair-recheck="))
        );
        assert_eq!(lock.argument_limit(), 48);
        assert_eq!(
            LockedPalsArenaLaunchV4::from_json(&lock.to_json().unwrap())
                .unwrap()
                .sha256(),
            lock.sha256()
        );
        assert_ne!(lock.sha256(), old.lock().unwrap().sha256());
        assert_eq!(
            serde_json::to_string(&old.semantic_lock).unwrap(),
            historical
        );
    }
    #[test]
    fn v4_diagnostic_and_undeclared_enabled_owner_fail_before_preparation() {
        let mut input = fixture();
        input.semantic_lock.manifest.purpose = PalsRunPurposeV4::Diagnostic;
        input.semantic_lock = input.semantic_lock.manifest.lock().unwrap();
        assert!(input.lock().is_err());
        let mut input = fixture();
        let PalsEngineV4::Pals(e) = &mut input.semantic_lock.manifest.engines[0] else {
            unreachable!()
        };
        e.model_v2.provenance.purpose = PalsArtifactPurposeV4::Diagnostic;
        assert!(input.semantic_lock.manifest.validate().is_err());
    }
    #[cfg(feature = "pals-collection")]
    #[test]
    fn native_v2_model_and_adapter_keep_independent_source_scopes() {
        let native = super::super::tests::native_fixture();
        let mut input = fixture();
        let PalsEngineV3::Pals(old) = &native.semantic_lock.manifest.engines[0] else {
            unreachable!()
        };
        let PalsEngineV4::Pals(e) = &mut input.semantic_lock.manifest.engines[0] else {
            unreachable!()
        };
        e.base.model = old.model.clone();
        let config = PalsModelConfig::full_line_interaction_v2();
        e.base.model.input_schema = config.profile.encoding_schema().into();
        e.base.model.implementation_sha256 =
            hex(rz_eval::pals_model::compiled_model_boundary_sha256());
        e.model_v2.profile = PalsModelProfileV4::FullLineInteractionV2;
        e.model_v2.model_schema = "rovezero.pals-model.v2".into();
        e.model_v2.model_semantics = config.profile.model_semantics().into();
        e.model_v2.model_semantics_sha256 = hex(config.profile.registered_semantics_sha256());
        e.model_v2.encoding_schema = config.profile.encoding_schema().into();
        e.model_v2.encoding_sha256 =
            hex(rz_uci::pals_native::pals_fresh_encoding_semantic_digest_for_config(&config));
        e.model_v2.max_line_plies = 256;
        let PalsEndpointLaunchV3::OnnxCpu(mut n) = native.endpoints[0].clone() else {
            unreachable!()
        };
        n.adapter_source_sha256 = hex(rz_uci::pals_native::pals_native_source_digest());
        n.encoding_semantic_sha256 =
            hex(rz_uci::pals_native::pals_rules_encoding_semantic_digest_for_config(&config));
        n.graphs.retain(|g| g.role != "proposer");
        n.graphs[1].role = "shared_pc".into();
        n.graphs[1].file = "shared_pc_if.onnx".into();
        e.model_v2.model_identity = format!("pals-onnx-pc-fp32-{}", n.export.sha256);
        assert_ne!(e.base.model.implementation_sha256, n.adapter_source_sha256);
        input.endpoints[0] = PalsEndpointLaunchV3::OnnxCpu(n.clone());
        input.semantic_lock.manifest.declared_changes =
            input.semantic_lock.manifest.actual_change_axes();
        input.semantic_lock = input.semantic_lock.manifest.lock().unwrap();
        let lock = input.lock().unwrap();
        assert!(
            lock.endpoint_views[0]
                .arguments
                .iter()
                .any(|a| a == "--pals-model-profile=full_line_interaction_v2")
        );
        let PalsEngineV4::Pals(e) = &lock.input.semantic_lock.manifest.engines[0] else {
            unreachable!()
        };
        let mut marker = PalsActualFollowupMarkerV4 {
            model_identity: e.model_v2.model_identity.clone(),
            model_profile: Some(config.profile.as_str().into()),
            model_semantics: config.profile.model_semantics().into(),
            model_semantics_sha256: Some(config.profile.registered_semantics_sha256()),
            model_implementation_sha256: Some(rz_eval::pals_model::compiled_model_boundary_sha256()),
            encoding_schema: Some(config.profile.encoding_schema().into()),
            model_configuration: Some(config),
            model_epoch: Some([0xaau8; 32]),
            resolver_policy: "own-raw-restricted".into(),
            cpu_resume: Some("completed-iteration".into()),
            post_repair_recheck: "disabled".into(),
            policy_identity: e.policy_identity.search_identity.clone(),
            policy_conditions_sha256: Some(e.policy_identity.conditions_sha256),
            search_implementation_sha256:
                rz_search::pals::engine::compiled_search_implementation_sha256(),
            resolver_implementation_sha256:
                rz_search::pals::engine::compiled_resolver_implementation_sha256(),
            resolver_semantics_sha256: rz_search::pals::engine::ResolverPolicy::OwnRawRestricted
                .semantics_sha256(),
            cuda_warm: false,
        };
        let PalsWeightIdentityV3::Untrained { artifact, .. } = &e.base.model.weights else {
            unreachable!()
        };
        let bytes: Vec<u8> = (0..64)
            .step_by(2)
            .map(|i| u8::from_str_radix(&artifact.sha256[i..i + 2], 16).unwrap())
            .collect();
        marker.model_epoch = Some(bytes.try_into().unwrap());
        let actual = serde_json::json!({"v4":marker});
        assert!(
            validate_followup_marker_pair(&lock, NativeEngineRole::Baseline, &actual, &actual)
                .is_ok()
        );
        let mut bad = actual.clone();
        bad["v4"]["model_implementation_sha256"] = serde_json::to_value([0_u8; 32]).unwrap();
        assert!(
            validate_followup_marker_pair(&lock, NativeEngineRole::Baseline, &bad, &bad).is_err()
        );
    }
    #[cfg(any(feature = "pals-collection", feature = "native-cuda"))]
    fn native_encoding_cpu_fixture() -> PalsArenaLaunchV4 {
        let native = super::super::tests::native_fixture();
        let PalsEngineV3::Pals(old) = &native.semantic_lock.manifest.engines[0] else {
            unreachable!()
        };
        let mut input = colab_fixture();
        for i in 0..2 {
            let cfg = if i == 0 {
                PalsModelConfig::baseline()
            } else {
                PalsModelConfig::full_line_interaction_v2()
            };
            let PalsEngineV4::Pals(e) = &mut input.semantic_lock.manifest.engines[i] else {
                unreachable!()
            };
            e.base.model = old.model.clone();
            e.base.model.input_schema = cfg.profile.encoding_schema().into();
            e.base.model.implementation_sha256 =
                hex(rz_eval::pals_model::compiled_model_boundary_sha256());
            e.model_v2.profile = if i == 0 {
                PalsModelProfileV4::LegacySummaryV1
            } else {
                PalsModelProfileV4::FullLineInteractionV2
            };
            e.model_v2.model_schema = if i == 0 {
                "rovezero.pals-model.v1"
            } else {
                "rovezero.pals-model.v2"
            }
            .into();
            e.model_v2.model_semantics = cfg.profile.model_semantics().into();
            e.model_v2.model_semantics_sha256 = hex(cfg.profile.registered_semantics_sha256());
            e.model_v2.encoding_schema = cfg.profile.encoding_schema().into();
            e.model_v2.encoding_sha256 =
                hex(rz_uci::pals_native::pals_fresh_encoding_semantic_digest_for_config(&cfg));
            e.model_v2.max_line_plies = if i == 0 { 0 } else { 256 };
            let PalsEndpointLaunchV3::OnnxCpu(mut n) = native.endpoints[0].clone() else {
                unreachable!()
            };
            n.adapter_source_sha256 = hex(rz_uci::pals_native::pals_native_source_digest());
            n.encoding_semantic_sha256 =
                hex(rz_uci::pals_native::pals_rules_encoding_semantic_digest_for_config(&cfg));
            if i == 1 {
                n.graphs.retain(|g| g.role != "proposer");
                n.graphs[1].role = "shared_pc".into();
                n.graphs[1].file = "shared_pc_if.onnx".into();
            }
            e.model_v2.model_identity = format!("pals-onnx-pc-fp32-{}", n.export.sha256);
            input.endpoints[i] = PalsEndpointLaunchV3::OnnxCpu(n);
        }
        input.semantic_lock.manifest.declared_changes =
            input.semantic_lock.manifest.actual_change_axes();
        input.semantic_lock = input.semantic_lock.manifest.lock().unwrap();
        input
    }
    #[cfg(any(feature = "pals-collection", feature = "native-cuda"))]
    fn native_encoding_fixtures() -> Vec<PalsArenaLaunchV4> {
        #[cfg(all(target_os = "linux", feature = "native-cuda"))]
        {
            vec![native_encoding_cpu_fixture(), cudnn_conv_pilot_fixture()]
        }
        #[cfg(not(all(target_os = "linux", feature = "native-cuda")))]
        {
            vec![native_encoding_cpu_fixture()]
        }
    }
    fn encoding_array(value: &str) -> Vec<u8> {
        (0..64)
            .step_by(2)
            .map(|i| u8::from_str_radix(&value[i..i + 2], 16).unwrap())
            .collect()
    }
    fn native_identity_record<L: PalsArenaValidationContext>(
        lock: &L,
        role: NativeEngineRole,
    ) -> serde_json::Value {
        let n = lock.native(role).unwrap();
        let (checkpoint, trained) = match &lock.pals_endpoint(role).unwrap().model.weights {
            PalsWeightIdentityV3::Untrained { artifact, .. } => (artifact, false),
            PalsWeightIdentityV3::Trained { artifact, .. } => (artifact, true),
            _ => unreachable!(),
        };
        serde_json::json!({
            "model_epoch":encoding_array(&checkpoint.sha256),
            "export_manifest_sha256":encoding_array(&n.export.sha256),
            "encoding_semantic_sha256":encoding_array(lock.native_receipt_encoding_identity(role).unwrap().sha256()),
            "adapter_source_sha256":encoding_array(&n.adapter_source_sha256),
            "trained":trained
        })
    }
    fn check_native_identity<L: PalsArenaValidationContext>(
        lock: &L,
        role: NativeEngineRole,
        record: &serde_json::Value,
    ) -> Result<(), ArenaError> {
        let (checkpoint, trained) = match &lock.pals_endpoint(role)?.model.weights {
            PalsWeightIdentityV3::Untrained { artifact, .. } => (artifact, false),
            PalsWeightIdentityV3::Trained { artifact, .. } => (artifact, true),
            _ => return Err(invalid("missing actual weights")),
        };
        validate_native_model_identity(
            lock.native(role)?,
            checkpoint,
            trained,
            lock.native_receipt_encoding_identity(role)?,
            record,
        )
    }
    #[cfg(any(feature = "pals-collection", feature = "native-cuda"))]
    #[test]
    fn native_v4_fresh_encoding_accepts_both_profiles_and_providers() {
        for input in native_encoding_fixtures() {
            let lock = input.lock().unwrap();
            for role in [NativeEngineRole::Baseline, NativeEngineRole::Candidate] {
                let fresh = lock.native_receipt_encoding_identity(role).unwrap();
                assert!(matches!(
                    fresh,
                    NativeReceiptEncodingIdentity::FreshModelInput(_)
                ));
                let rules = &lock.native(role).unwrap().encoding_semantic_sha256;
                assert_ne!(fresh.sha256(), rules);
                let record = native_identity_record(&lock, role);
                assert!(check_native_identity(&lock, role, &record).is_ok());
                let mut wrong = record;
                wrong["encoding_semantic_sha256"] = serde_json::json!(encoding_array(rules));
                assert!(check_native_identity(&lock, role, &wrong).is_err());
            }
        }
    }
    #[cfg(any(feature = "pals-collection", feature = "native-cuda"))]
    #[test]
    fn native_v4_rules_and_fresh_declaration_domains_cannot_be_swapped() {
        for input in native_encoding_fixtures() {
            for i in 0..2 {
                let PalsEngineV4::Pals(e) = &input.semantic_lock.manifest.engines[i] else {
                    unreachable!()
                };
                let n = input.endpoints[i].native_model().unwrap();
                assert!(validate_actual_encoding(e, n).is_ok());
                let mut swapped_e = e.as_ref().clone();
                let mut swapped_n = n.clone();
                std::mem::swap(
                    &mut swapped_e.model_v2.encoding_sha256,
                    &mut swapped_n.encoding_semantic_sha256,
                );
                assert!(validate_actual_encoding(&swapped_e, &swapped_n).is_err());
                assert!(validate_actual_encoding(&swapped_e, n).is_err());
                assert!(validate_actual_encoding(e, &swapped_n).is_err());
            }
        }
    }
    #[cfg(any(feature = "pals-collection", feature = "native-cuda"))]
    #[test]
    fn native_v4_encoding_rejects_history_profile_and_malformed_namespaces() {
        let exact = rz_position::Position::startpos();
        let unknown = rz_position::Position::from_fen(&exact.to_fen()).unwrap();
        let exact_history = rz_uci::pals_native::pals_history_digest(&exact).unwrap();
        let unknown_history = rz_uci::pals_native::pals_history_digest(&unknown).unwrap();
        assert_ne!(exact_history, unknown_history);
        for input in native_encoding_fixtures() {
            let lock = input.lock().unwrap();
            for role in [NativeEngineRole::Baseline, NativeEngineRole::Candidate] {
                let other = match role {
                    NativeEngineRole::Baseline => NativeEngineRole::Candidate,
                    NativeEngineRole::Candidate => NativeEngineRole::Baseline,
                };
                let record = native_identity_record(&lock, role);
                let different_profile = lock
                    .native_receipt_encoding_identity(other)
                    .unwrap()
                    .sha256();
                assert_ne!(
                    lock.native_receipt_encoding_identity(role)
                        .unwrap()
                        .sha256(),
                    different_profile
                );
                for bytes in [
                    exact_history.to_vec(),
                    unknown_history.to_vec(),
                    encoding_array(different_profile),
                    vec![0_u8; 31],
                ] {
                    let mut wrong = record.clone();
                    wrong["encoding_semantic_sha256"] = serde_json::json!(bytes);
                    assert!(check_native_identity(&lock, role, &wrong).is_err());
                }
            }
        }
    }
    #[cfg(any(feature = "pals-collection", feature = "native-cuda"))]
    #[test]
    fn native_v4_fresh_encoding_preserves_other_frozen_identity_guards() {
        for input in native_encoding_fixtures() {
            let lock = input.lock().unwrap();
            for role in [NativeEngineRole::Baseline, NativeEngineRole::Candidate] {
                let record = native_identity_record(&lock, role);
                for field in [
                    "model_epoch",
                    "export_manifest_sha256",
                    "adapter_source_sha256",
                ] {
                    let mut wrong = record.clone();
                    let bytes = wrong[field].as_array_mut().unwrap();
                    bytes[0] = (bytes[0].as_u64().unwrap() ^ 1).into();
                    assert!(check_native_identity(&lock, role, &wrong).is_err());
                }
                let mut wrong = record.clone();
                wrong["trained"] = true.into();
                assert!(check_native_identity(&lock, role, &wrong).is_err());
                let mut wrong = record;
                wrong
                    .as_object_mut()
                    .unwrap()
                    .remove("adapter_source_sha256");
                assert!(check_native_identity(&lock, role, &wrong).is_err());
            }
        }
    }
    #[test]
    fn native_v3_receipt_encoding_preserves_legacy_recipe_domain() {
        let lock = super::super::tests::native_fixture().lock().unwrap();
        let role = NativeEngineRole::Baseline;
        let encoding = lock.native_receipt_encoding_identity(role).unwrap();
        assert!(matches!(
            encoding,
            NativeReceiptEncodingIdentity::LegacyRecipe(_)
        ));
        assert_eq!(
            encoding.sha256(),
            lock.native(role).unwrap().encoding_semantic_sha256
        );
        let mut record = native_identity_record(&lock, role);
        assert!(check_native_identity(&lock, role, &record).is_ok());
        record["encoding_semantic_sha256"] = serde_json::json!(vec![0xee_u8; 32]);
        assert!(check_native_identity(&lock, role, &record).is_err());
    }
    #[test]
    fn native_v4_mock_cannot_claim_a_native_fresh_encoding_domain() {
        let lock = fixture().lock().unwrap();
        for role in [NativeEngineRole::Baseline, NativeEngineRole::Candidate] {
            assert!(lock.native_receipt_encoding_identity(role).is_err());
        }
    }
    #[test]
    fn actual_marker_conditions_resolver_and_stack_drift_reject() {
        let lock = fixture().lock().unwrap();
        let PalsEngineV4::Pals(e) = &lock.input.semantic_lock.manifest.engines[0] else {
            unreachable!()
        };
        let marker = PalsActualFollowupMarkerV4 {
            model_identity: "explicit-legal-order-role-mock-v1".into(),
            model_profile: None,
            model_semantics: "explicit-legal-order-role-mock-v1".into(),
            model_semantics_sha256: None,
            model_implementation_sha256: None,
            encoding_schema: None,
            model_configuration: None,
            model_epoch: None,
            resolver_policy: "own-raw-restricted".into(),
            cpu_resume: Some("completed-iteration".into()),
            post_repair_recheck: "disabled".into(),
            policy_identity: e.policy_identity.search_identity.clone(),
            policy_conditions_sha256: Some(e.policy_identity.conditions_sha256),
            search_implementation_sha256:
                rz_search::pals::engine::compiled_search_implementation_sha256(),
            resolver_implementation_sha256:
                rz_search::pals::engine::compiled_resolver_implementation_sha256(),
            resolver_semantics_sha256: rz_search::pals::engine::ResolverPolicy::OwnRawRestricted
                .semantics_sha256(),
            cuda_warm: false,
        };
        let good = serde_json::json!({"v4":marker});
        assert!(
            validate_followup_marker_pair(&lock, NativeEngineRole::Baseline, &good, &good).is_ok()
        );
        for field in ["resolver_policy", "cpu_resume", "policy_identity"] {
            let mut bad = good.clone();
            bad["v4"][field] = "different".into();
            assert!(
                validate_followup_marker_pair(&lock, NativeEngineRole::Baseline, &bad, &bad)
                    .is_err()
            );
        }
        let mut bad = good.clone();
        bad["v4"]["policy_conditions_sha256"] = serde_json::Value::Null;
        assert!(
            validate_followup_marker_pair(&lock, NativeEngineRole::Baseline, &bad, &bad).is_err()
        );
        for field in [
            "search_implementation_sha256",
            "resolver_implementation_sha256",
            "resolver_semantics_sha256",
        ] {
            let mut bad = good.clone();
            bad["v4"][field] = serde_json::to_value([0_u8; 32]).unwrap();
            assert!(
                validate_followup_marker_pair(&lock, NativeEngineRole::Baseline, &bad, &bad)
                    .is_err()
            );
        }
    }
    #[cfg(all(target_os = "linux", feature = "native-cuda"))]
    #[test]
    fn own_default_preflight_uses_exact_receipts_and_owner_roots() {
        let lock = cudnn_conv_pilot_fixture().lock().unwrap();
        assert_eq!(lock.expected_provider_sessions(), 4);
        for role in [NativeEngineRole::Baseline, NativeEngineRole::Candidate] {
            let (game_name, probe_name) = match role {
                NativeEngineRole::Baseline => {
                    ("baseline-runtime", "external-baseline-preflight-runtime")
                }
                NativeEngineRole::Candidate => {
                    ("candidate-runtime", "external-candidate-preflight-runtime")
                }
            };
            let game = Path::new("/owned/attempt").join(game_name);
            let probe = Path::new("/owned/attempt").join(probe_name);
            assert!(lock.uses_owned_default_preflight(role));
            assert!(lock.uses_separate_preflight_runtime_root(role).unwrap());
            assert!(lock.preflight_arguments(role, &game).is_err());
            let preflight = lock.preflight_arguments(role, &probe).unwrap();
            let actual = lock.runtime_arguments(role, &game).unwrap();
            let shutdown = OsString::from(format!(
                "--pals-shutdown-timeout-ms={}",
                lock.input
                    .semantic_lock
                    .manifest
                    .pilot
                    .cleanup_max_ms
                    .div_ceil(2)
            ));
            // The launcher combines endpoint base arguments with either extension.
            // Check that each resulting argv has one owner budget, rather than
            // requiring an endpoint option in the receipt-only extension.
            let base = lock
                .endpoint_view(role)
                .arguments
                .iter()
                .map(OsString::from);
            for extension in [&preflight, &actual] {
                assert_eq!(
                    base.clone()
                        .chain(extension.iter().cloned())
                        .filter(|argument| argument == &shutdown)
                        .count(),
                    1
                );
            }
            let receipt_flags = [
                format!("--pals-output-root={}", probe.display()),
                format!("--pals-launch-sha256={}", lock.sha256()),
                format!(
                    "--pals-endpoint-id={}",
                    lock.endpoint_views[role_index(role)].id
                ),
            ];
            for flag in receipt_flags {
                assert!(preflight.contains(&OsString::from(flag)));
            }
            let profile = OsString::from(format!("--pals-cuda-profile-parent={}", probe.display()));
            assert!(preflight.contains(&profile) && actual.contains(&profile));
            assert!(actual.contains(&OsString::from(format!(
                "--pals-output-root={}",
                game.display()
            ))));
            assert!(!game.exists() && !probe.exists()); // argv preparation creates no unowned directory
            let wrong = Path::new("/owned/attempt").join(if role == NativeEngineRole::Baseline {
                "candidate-runtime"
            } else {
                "baseline-runtime"
            });
            assert!(lock.cuda_profile_argument(role, &wrong).is_err());
        }
        let historical_mock = fixture().lock().unwrap();
        assert!(!historical_mock.uses_owned_default_preflight(NativeEngineRole::Baseline));
        let mut non_default = cudnn_conv_pilot_fixture();
        for endpoint in &mut non_default.endpoints {
            let PalsEndpointLaunchV3::OnnxCuda(cuda) = endpoint else {
                unreachable!()
            };
            cuda.cuda_control.as_mut().unwrap().cudnn_conv_search = None;
        }
        let non_default = non_default.lock().unwrap();
        let game = Path::new("/owned/attempt/baseline-runtime");
        assert!(!non_default.uses_owned_default_preflight(NativeEngineRole::Baseline));
        assert!(
            !non_default
                .uses_separate_preflight_runtime_root(NativeEngineRole::Baseline)
                .unwrap()
        );
        let args = non_default
            .preflight_arguments(NativeEngineRole::Baseline, game)
            .unwrap();
        assert_eq!(args.len(), 2); // unchanged shared-cache + existing explicit inventory profile argv
        assert_eq!(
            args.last(),
            Some(&OsString::from(
                "--pals-cuda-profile-parent=/owned/attempt/baseline-runtime"
            ))
        );
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn default_profile_pid_nonce_is_closed_and_canonical() {
        assert_eq!(
            default_control_profile_pid("pals-cuda-profile-10-0000000000000001").unwrap(),
            10
        );
        assert_eq!(
            default_control_profile_pid("pals-cuda-profile-2147483647-ffffffffffffffff").unwrap(),
            2147483647
        );
        for value in [
            "pals-cuda-profile-0-0000000000000001",
            "pals-cuda-profile-010-0000000000000001",
            "pals-cuda-profile-2147483648-0000000000000001",
            "pals-cuda-profile-4294967296-0000000000000001",
            "pals-cuda-profile--10-0000000000000001",
            "pals-cuda-profile-10-1",
            "pals-cuda-profile-10-FFFFFFFFFFFFFFFF",
            "pals-cuda-profile-10-000000000000000g",
            "pals-cuda-profile-10-0000000000000001-extra",
            "pals-cuda-profile-10-0000000000000001/child",
            "native-process-10",
        ] {
            assert!(default_control_profile_pid(value).is_err(), "{value}");
        }
    }

    #[cfg(target_os = "linux")]
    struct DefaultProfileCpuFixture {
        root: std::path::PathBuf,
        native_slots: BTreeSet<String>,
        audits: Vec<PalsNativeSessionAuditV3>,
    }
    #[cfg(target_os = "linux")]
    impl DefaultProfileCpuFixture {
        fn new() -> Self {
            use std::os::unix::fs::PermissionsExt;
            let nonce = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let root = std::env::temp_dir().join(format!(
                "rovezero-own-default-cpu-{}-{nonce}",
                std::process::id()
            ));
            std::fs::create_dir(&root).unwrap();
            std::fs::set_permissions(&root, std::fs::Permissions::from_mode(0o700)).unwrap();
            let native_slots =
                BTreeSet::from(["native-process-10".into(), "native-process-12".into()]);
            for slot in &native_slots {
                std::fs::create_dir(root.join(slot)).unwrap();
            }
            let mut fixture = Self {
                root,
                native_slots,
                audits: vec![],
            };
            fixture.add_profile(10);
            fixture.add_profile(12);
            fixture
        }
        fn profile(&self, pid: u32) -> std::path::PathBuf {
            self.root
                .join(format!("pals-cuda-profile-{pid}-0000000000000001"))
        }
        fn add_profile(&mut self, pid: u32) {
            // Tiny authored wire/file fixture; never a claim of a CUDA run or physical fence.
            let profile = self.profile(pid);
            std::fs::create_dir(&profile).unwrap();
            let public = format!("authored-public-kernels-{pid}").into_bytes();
            let pc = format!("authored-shared-pc-kernels-{pid}").into_bytes();
            std::fs::write(
                profile.join("pals-public-kernels_2026-10-12_00-00-00.json"),
                &public,
            )
            .unwrap();
            std::fs::write(
                profile.join("pals-shared_pc-kernels_2026-10-12_00-00-00.json"),
                &pc,
            )
            .unwrap();
            for role in ["public", "shared_pc"] {
                let value = serde_json::json!({
                    "schema":"rovezero.pals-initial-placement.v2",
                    "source":"ort-1.22-finalize-recursive-plus-memcpy-origin-metadata",
                    "optimization":"disable","role":role,"session_created":true,
                    "before_first_neural_run":true,"gate":"passed","gate_detail":null,
                    "collector_overflow":false
                });
                std::fs::write(
                    profile.join(format!("{role}-initial-placement.json")),
                    serde_json::to_vec(&value).unwrap(),
                )
                .unwrap();
            }
            self.audits.push(PalsNativeSessionAuditV3 {
                endpoint_id: "authored-default-profile-only".into(),
                process_id: pid,
                launch_sha256: "a".repeat(64),
                startup_sha256: "b".repeat(64),
                termination_sha256: "c".repeat(64),
                external_cpu_r_session: None,
                completed_role_inputs: 0,
                search_consumed_role_inputs: 0,
                completed_new_game_resets: 0,
                physical_shutdown_confirmed: true,
                native_buffers_released: true,
                startup_nn_inputs_completed: 0,
                startup_nn_calls_completed: 0,
                startup_role_inputs_completed: 0,
                startup_probe_timeout_ms: None,
                startup_probe: None,
                execution: None,
                cuda_placement: Some(PalsCudaPlacementAuditV3 {
                    inventory_sha256: "d".repeat(64),
                    public_profile_sha256: digest(&public),
                    shared_pc_profile_sha256: digest(&pc),
                    public_neural_kernels: 0,
                    proposer_private_kernels: 0,
                    critic_private_kernels: 0,
                    device_id: 0,
                }),
                cuda_loading: None,
                cuda_record_pages: None,
                raw_native: serde_json::Value::Null,
                raw_search_work: serde_json::Value::Null,
            });
        }
        fn validate(&self, audits: &[PalsNativeSessionAuditV3]) -> Result<(), ArenaError> {
            let directory =
                cap_std::fs::Dir::open_ambient_dir(&self.root, cap_std::ambient_authority())
                    .unwrap();
            validate_owned_default_profile_slots(&directory, audits, &self.native_slots, 2048)
        }
    }
    #[cfg(target_os = "linux")]
    impl Drop for DefaultProfileCpuFixture {
        fn drop(&mut self) {
            // Only this newly created, test-owned regular subtree is disposable.
            assert_eq!(self.root.parent(), Some(std::env::temp_dir().as_path()));
            assert!(
                self.root
                    .file_name()
                    .unwrap()
                    .to_string_lossy()
                    .starts_with("rovezero-own-default-cpu-")
            );
            std::fs::remove_dir_all(&self.root).unwrap();
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn default_profile_scope_rejects_foreign_pid_duplicate_nonce_and_digest() {
        let fixture = DefaultProfileCpuFixture::new();
        fixture.validate(&fixture.audits).unwrap();
        let foreign = fixture.root.join("pals-cuda-profile-99-0000000000000001");
        std::fs::create_dir(&foreign).unwrap();
        assert!(fixture.validate(&fixture.audits).is_err());
        std::fs::remove_dir(foreign).unwrap();
        let repeated = fixture.root.join("pals-cuda-profile-10-0000000000000002");
        std::fs::create_dir(&repeated).unwrap();
        assert!(fixture.validate(&fixture.audits).is_err());
        std::fs::remove_dir(repeated).unwrap();
        let kernel = fixture
            .profile(10)
            .join("pals-public-kernels_2026-10-12_00-00-00.json");
        let original = std::fs::read(&kernel).unwrap();
        std::fs::write(&kernel, b"changed digest").unwrap();
        assert!(fixture.validate(&fixture.audits).is_err());
        std::fs::write(&kernel, original).unwrap();
        fixture.validate(&fixture.audits).unwrap();
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn default_profile_scope_requires_physical_closure_and_nofollow_owned_files() {
        use std::os::unix::fs::{PermissionsExt, symlink};
        let fixture = DefaultProfileCpuFixture::new();
        fixture.validate(&fixture.audits).unwrap();
        let mut unclosed = fixture.audits.clone();
        unclosed[0].physical_shutdown_confirmed = false;
        assert!(fixture.validate(&unclosed).is_err());
        unclosed[0].physical_shutdown_confirmed = true;
        unclosed[0].native_buffers_released = false;
        assert!(fixture.validate(&unclosed).is_err());
        unclosed[0].native_buffers_released = true;
        unclosed[0].cuda_placement = None;
        assert!(fixture.validate(&unclosed).is_err());
        let kernel = fixture
            .profile(10)
            .join("pals-public-kernels_2026-10-12_00-00-00.json");
        let original = std::fs::read(&kernel).unwrap();
        std::fs::remove_file(&kernel).unwrap();
        symlink(
            fixture
                .profile(12)
                .join("pals-public-kernels_2026-10-12_00-00-00.json"),
            &kernel,
        )
        .unwrap();
        assert!(fixture.validate(&fixture.audits).is_err());
        std::fs::remove_file(&kernel).unwrap();
        std::fs::write(&kernel, original).unwrap();
        let held = fixture.profile(10);
        std::fs::set_permissions(&held, std::fs::Permissions::from_mode(0o777)).unwrap();
        assert!(fixture.validate(&fixture.audits).is_err());
        std::fs::set_permissions(held, std::fs::Permissions::from_mode(0o755)).unwrap();
        fixture.validate(&fixture.audits).unwrap();
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn default_profile_scope_keeps_probe_slots_and_accepts_bounded_four_mib_placement_json() {
        let mut fixture = DefaultProfileCpuFixture::new();
        fixture.validate(&fixture.audits).unwrap();
        let placement = fixture.profile(10).join("public-initial-placement.json");
        let mut value: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&placement).unwrap()).unwrap();
        // The producer's bounded JSON writer permits 4MiB, independently of its
        // 1MiB logger input cap. This authored case catches a mistaken 1MiB reader.
        value["authored_diagnostic_padding"] = "X".repeat(1024 * 1024).into();
        std::fs::write(&placement, serde_json::to_vec(&value).unwrap()).unwrap();
        fixture.validate(&fixture.audits).unwrap();
        fixture.add_profile(100);
        fixture.add_profile(102);
        fixture.validate(&fixture.audits).unwrap(); // two probe slots + four profile PIDs; no game receipt in probe root
        let forbidden_game_slot = fixture.root.join("native-process-100");
        std::fs::create_dir(&forbidden_game_slot).unwrap();
        assert!(fixture.validate(&fixture.audits).is_err());
        std::fs::remove_dir(forbidden_game_slot).unwrap();
        let overflow = fixture.profile(10).join("shared_pc-initial-placement.json");
        let original = std::fs::read(&overflow).unwrap();
        std::fs::write(&overflow, vec![b'X'; 4 * 1024 * 1024 + 1]).unwrap();
        assert!(fixture.validate(&fixture.audits).is_err());
        std::fs::write(overflow, original).unwrap();
        fixture.validate(&fixture.audits).unwrap();
    }
}
