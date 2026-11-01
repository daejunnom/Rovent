//! PALS V3 launch envelope over the existing pinned arena lifecycle.
//!
//! V2 UCI endpoint values below are executable views, never a conversion of the
//! V3 manifest or its digest. Native P/C evidence, UCI observations, Rules PGN
//! replay, clocks, process exit and input retirement remain separate gates.
use crate::{
    ArenaError, NativeEngineView, NativeLaunchDeclaration, NativeLaunchOwner, NativePairView,
    NativeProviderDeclaration,
};
use rz_eval::pals_device_resources::NativeCudaRecordPagesResourceInput;
use rz_experiments::{
    ArtifactRef, EngineEnvironmentV2, ExternalSourceV2, ExternalUciEndpointV2, HistoryCompleteness,
    InitialPosition, ManifestError, NativeEngineRole, NativeGameClockV3, NativePairClock,
    NativeResourceBudgetV1, NativeTimeoutsV1, OpeningSpec, PalsCpuRSelectionV3, PalsEngineV3,
    PalsExternalCpuRV3, PalsInputLockV3, PalsModelBackendV3, PalsPostRepairRecheckPolicyV3,
    PalsPrecisionV3, PalsRunReceiptV3, PalsSearchPolicyIdentityV3, PalsWeightIdentityV3,
    ToolIdentity,
};
#[cfg(target_os = "linux")]
use rz_experiments::{PALS_RECEIPT_V3_DOMAIN, PalsGameReceiptV3, PalsResultV3, PalsTerminationV3};
#[cfg(any(target_os = "linux", test))]
use rz_experiments::{
    PalsEndpointReceiptV3, PalsObservedV3, PalsOptionReceiptV3, PalsPhysicalStateV3,
    PalsRunFailureV3,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    ffi::OsString,
    io::Read,
    path::{Component, Path, PathBuf},
};

pub mod v4;

pub const PALS_ARENA_V3_DOMAIN: &str = "rz-pals-arena-launch-v3/1";
pub const PALS_NATIVE_STARTUP_V3_DOMAIN: &str = "rz-pals-native-startup-v3/1";
pub const PALS_NATIVE_TERMINATION_V3_DOMAIN: &str = "rz-pals-native-termination-v3/1";
pub const PALS_SEARCH_WORK_STARTUP_V3_DOMAIN: &str = "rz-pals-search-work-startup-v3/1";
pub const PALS_SEARCH_WORK_TERMINATION_V3_DOMAIN: &str = "rz-pals-search-work-termination-v3/1";
pub const PALS_CLOCK_REAP_PATCH_SHA256: &str =
    "23bc4abfa79fcde2b07bebcd70dba2adc112c6152ae3346188f693dc1b5e9ec5";
pub const PALS_CLOCK_REAP_PATCH_BYTES: u64 = 9907;
pub const PALS_CLOCK_REAP_RUNNER_SHA256: &str =
    "29e89312bc4ec16a32ec185d8b53c60b0cdbad17eaac294f987491e36b8d29ff";
pub const PALS_CLOCK_REAP_RUNNER_BYTES: u64 = 2466608;
/// Separately registered baseline-ISA release build of the same source and patch.
/// Release mode reports the short version; its exact binary pin retains provenance.
pub const PALS_CLOCK_REAP_COLAB_RUNNER_VERSION: &str = "fastchess alpha 1.8.2";
pub const PALS_CLOCK_REAP_COLAB_RUNNER_SHA256: &str =
    "9e70bcf2c619836e2c438e31a30f56aacfac8199aebc24007191b106605b3f7e";
pub const PALS_CLOCK_REAP_COLAB_RUNNER_BYTES: u64 = 2438832;

pub(crate) fn registered_reap_runner(version: &str, sha256: &str, bytes: u64) -> bool {
    matches!(
        (version, sha256, bytes),
        (
            crate::FASTCHESS_VERSION,
            PALS_CLOCK_REAP_RUNNER_SHA256,
            PALS_CLOCK_REAP_RUNNER_BYTES
        ) | (
            PALS_CLOCK_REAP_COLAB_RUNNER_VERSION,
            PALS_CLOCK_REAP_COLAB_RUNNER_SHA256,
            PALS_CLOCK_REAP_COLAB_RUNNER_BYTES
        )
    )
}
const MAX_JSON_BYTES: usize = 256 * 1024;
const MAX_CONTROL_INVENTORY_BYTES: u64 = 1024 * 1024;
const MAX_STARTUP_PROBE_TIMEOUT_MS: u64 = 180_000;
const MAX_CUDA_RECORD_PAGES_RESOURCES_BYTES: usize = 8192;

/// An actual bounded profile-file observation, separate from process startup.
/// Registered paths are compared exactly; the hashed profile is never rewritten
/// to point at an arena snapshot. Binary execution still requires the checker's
/// own ELF/FD/hash verification on its finite startup clock.
#[derive(Clone, Debug, Serialize)]
pub struct PalsExternalCpuRProfileAuditV3 {
    pub domain: &'static str,
    pub endpoint_id: String,
    pub semantic_lock_sha256: String,
    pub profile_file_sha256: String,
    pub profile_canonical_sha256: String,
    pub profile_file_bytes: u64,
    pub registered_binary_sha256: String,
    pub registered_program: String,
    pub registered_working_directory: String,
    pub profile_registration: serde_json::Value,
    pub profile_file_verification_performed: bool,
    /// The binary hash above is independently registered metadata compared with
    /// the loaded profile identity, not an observation of executable bytes.
    pub binary_file_verification_performed: bool,
    pub execution_admission_completed: bool,
    pub child_spawned: bool,
    pub options_application_observed: Option<bool>,
    pub model_loading_observed: Option<bool>,
    pub inherited_resource_join_observed: Option<bool>,
    pub physical_shutdown_observed: Option<bool>,
}

#[derive(Clone, Copy, Debug, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PalsExternalCpuRSessionPurposeV3 {
    Identification,
    Readiness,
    Game,
}
/// Validation of one actual producer pair, separate from execution admission.
/// Original registration and byte hashes remain available; scoped resources
/// and foreign work are never projected into Own CPU counters.
/// Parent start ticks/cgroup come from the helper producer's ready snapshot;
/// the supervisor independently joins PID/role/session and owns its reap.
/// Native resets do not observe a foreign ucinewgame/ready barrier history.
#[derive(Clone, Debug, Serialize)]
pub struct PalsExternalCpuRSessionAuditV3 {
    pub purpose: PalsExternalCpuRSessionPurposeV3,
    pub endpoint_id: String,
    pub parent_process_id: u32,
    pub startup_sha256: String,
    pub termination_sha256: String,
    pub registration: serde_json::Value,
    pub receipt: rz_experiments::PalsExternalCpuRReceiptV3,
}

#[cfg(all(
    target_os = "linux",
    any(feature = "pals-collection-onnx", feature = "native-cuda")
))]
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ExternalCheckerWireV3 {
    schema_version: u32,
    domain: String,
    profile_file_sha256: String,
    profile_canonical_sha256: String,
    registration_sha256: String,
    registration: serde_json::Value,
    startup_handshake_completed: bool,
    observed_uci: Option<ExternalCheckerUciWireV3>,
    startup_resource_observation: Option<rz_experiments::PalsCheckerReadyResourcesV3>,
    startup_resource_unavailable: Option<String>,
    requested_option_checks_completed: bool,
    applied_option_values: String,
    latest_attempt: Option<serde_json::Value>,
    shutdown: Option<rz_experiments::PalsHelperShutdownV3>,
    cleanup_complete: bool,
    started_owner_exit_and_drains_confirmed: bool,
}
#[cfg(all(
    target_os = "linux",
    any(feature = "pals-collection-onnx", feature = "native-cuda")
))]
#[derive(Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct ExternalCheckerUciWireV3 {
    name: String,
    author: Option<String>,
}

/// Actual profile loading must precede this API. It creates only an unstarted
/// checker to obtain immutable identity/conditions/capabilities; it starts no
/// model or process and does not itself admit an arena run or Core projection.
/// The caller must join `expected_parent_pid` to its supervisor/game evidence;
/// this helper audit does not replace the four UCI exits, NN graph or PGN audit.
#[cfg(all(
    target_os = "linux",
    any(feature = "pals-collection-onnx", feature = "native-cuda")
))]
pub fn validate_pals_external_cpu_r_session(
    lock: &LockedPalsArenaLaunchV3,
    role: NativeEngineRole,
    loaded: &rz_uci::pals_checker_profile::LoadedPalsCheckerProfile,
    startup: &[u8],
    termination: &[u8],
    expected_parent_pid: u32,
    purpose: PalsExternalCpuRSessionPurposeV3,
) -> Result<PalsExternalCpuRSessionAuditV3, ArenaError> {
    validate_pals_external_cpu_r_session_context(
        lock,
        role,
        loaded,
        startup,
        termination,
        expected_parent_pid,
        purpose,
    )
}
#[cfg(all(
    target_os = "linux",
    any(feature = "pals-collection-onnx", feature = "native-cuda")
))]
fn validate_pals_external_cpu_r_session_context<L: PalsArenaValidationContext>(
    lock: &L,
    role: NativeEngineRole,
    loaded: &rz_uci::pals_checker_profile::LoadedPalsCheckerProfile,
    startup: &[u8],
    termination: &[u8],
    expected_parent_pid: u32,
    purpose: PalsExternalCpuRSessionPurposeV3,
) -> Result<PalsExternalCpuRSessionAuditV3, ArenaError> {
    use rz_experiments::{
        PalsExternalCpuRReceiptV3, PalsExternalCpuRWorkV3, PalsModelValueIdentityV3, PalsObservedV3,
    };
    use rz_search::cpu_checker::CpuChecker;
    lock.verify_semantic()?;
    require(
        startup.len() <= 128 * 1024 && termination.len() <= 128 * 1024,
        "external helper native envelope byte budget exceeded",
    )?;
    let s = unique_helper_json(startup)?;
    let t = unique_helper_json(termination)?;
    let endpoint = lock.pals_endpoint(role)?;
    let Some((declaration, binding)) = lock.external_binding(role)? else {
        return Err(invalid(
            "own CPU_R cannot consume an external helper session",
        ));
    };
    let native = lock.native(role)?;
    let profile = loaded.profile();
    require(
        loaded.file_sha256() == declaration.profile.sha256
            && loaded.canonical_sha256() == declaration.profile_canonical_sha256
            && loaded.registration().file_bytes == declaration.profile.bytes
            && profile.program == binding.registered_program
            && profile.working_directory == binding.registered_working_directory
            && profile.identity.binary_sha256 == declaration.binary.sha256
            && profile.identity.declared_source == declaration.binary.source
            && profile.identity.declared_license == declaration.binary.license,
        "external helper actual loaded profile/pins/registered paths differ",
    )?;
    let p = &declaration.policy;
    require(
        profile.identity.options["Threads"]
            .parse::<u32>()
            .is_ok_and(|v| v <= p.threads_max)
            && profile.identity.options["Hash"]
                .parse::<u32>()
                .is_ok_and(|v| v <= p.hash_mib_max)
            && u32::from(profile.max_depth) <= p.max_depth
            && profile.max_prefix_plies as u64 <= u64::from(p.max_prefix_plies)
            && profile.handshake_timeout_ms <= p.handshake_max_ms
            && profile.max_task_wall_time_ms <= p.task_wall_time_max_ms
            && profile.stop_grace_ms <= p.stop_grace_max_ms
            && profile.shutdown_grace_ms <= p.shutdown_grace_max_ms
            && profile.max_output_bytes as u64 <= p.lifetime_output_bytes_max
            && profile.max_line_bytes as u64 <= p.line_bytes_max,
        "external helper loaded task/time/output profile exceeds declaration",
    )?;
    require(
        declaration.resolver.version == rz_search::pals::value::MODEL_WDL_RESOLVER_VERSION
            && declaration.resolver.semantics_sha256
                == digest(rz_search::pals::value::MODEL_WDL_RESOLVER_SEMANTICS.as_bytes()),
        "external helper model-WDL resolver differs from production semantics",
    )?;
    let checker = loaded
        .create_checker()
        .map_err(|e| invalid(format!("external helper unstarted checker: {e}")))?;
    let caps = checker.capabilities();
    let checkpoint = match &endpoint.model.weights {
        PalsWeightIdentityV3::Untrained { artifact, .. }
        | PalsWeightIdentityV3::Trained { artifact, .. } => artifact,
        _ => return Err(invalid("external helper model-WDL checkpoint missing")),
    };
    let mut epoch = [0u8; 32];
    for (i, byte) in epoch.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&checkpoint.sha256[i * 2..i * 2 + 2], 16)
            .map_err(|_| invalid("external helper checkpoint digest malformed"))?;
    }
    let mut model = format!("pals-onnx-pc-fp32-{}", native.export.sha256);
    if let Some(cuda) = lock.recipe(role).cuda_model() {
        model.push_str(&format!(
            "-cuda-device{}-arena{}",
            cuda.device_id, cuda.session_arena_bytes
        ));
    }
    let model_value = PalsModelValueIdentityV3 {
        semantics: rz_search::pals::value::MODEL_WDL_VALUE_SEMANTICS.into(),
        model: model.clone(),
        encoding: native.encoding_semantic_sha256.clone(),
        precision: "fp32".into(),
        model_epoch: epoch,
    };
    let registration = serde_json::json!({
        "identity": checker.identity(), "conditions": checker.conditions(),
        "capabilities": {"max_depth":caps.max_depth,"max_prefix_plies":caps.max_prefix_plies,
            "max_root_moves":caps.max_root_moves,"root_moves":caps.root_moves,"divergence":caps.divergence,
            "resume":caps.resume,"selective_search":caps.selective_search},
        "role_model": model, "model_value": model_value,
        "resolver_version":rz_search::pals::value::MODEL_WDL_RESOLVER_VERSION,
        "resolver_semantics":rz_search::pals::value::MODEL_WDL_RESOLVER_SEMANTICS,
    });
    let mut registered_hash = Sha256::new();
    registered_hash.update(b"rz-pals-checker-registration/1\0");
    registered_hash.update(serde_json::to_vec(&registration).map_err(|e| invalid(e.to_string()))?);
    let registration_sha256 = format!("{:x}", registered_hash.finalize());
    let (sw, tw): (ExternalCheckerWireV3, ExternalCheckerWireV3) = (
        serde_json::from_value(s["cpu_checker"].clone())
            .map_err(|e| invalid(format!("external startup checker wire: {e}")))?,
        serde_json::from_value(t["cpu_checker"].clone())
            .map_err(|e| invalid(format!("external termination checker wire: {e}")))?,
    );
    require(
        sw.startup_resource_observation == tw.startup_resource_observation
            && sw.startup_resource_unavailable.is_none()
            && tw.startup_resource_unavailable.is_none()
            && sw.observed_uci == tw.observed_uci,
        "external helper historical ready resources/name changed or unavailable",
    )?;
    for (v, wire, ending) in [(&s, &sw, false), (&t, &tw, true)] {
        require(
            v["schema_version"] == lock.native_wire_version()
                && v["domain"]
                    == if ending {
                        lock.native_wire_domains().1
                    } else {
                        lock.native_wire_domains().0
                    }
                && v["endpoint_id"] == endpoint.id
                && v["launch_sha256"] == lock.launch_sha256()
                && v["process_id"] == expected_parent_pid
                && expected_parent_pid > 0
                && expected_parent_pid <= i32::MAX as u32
                && v["binary_sha256"] == endpoint.binary.sha256
                && v["runtime_sha256"] == native.runtime.sha256
                && v["precision"] == "fp32"
                && v["service_exit_success"] == ending
                && v["provider"]
                    == if lock.recipe(role).cuda_model().is_some() {
                        "cuda"
                    } else {
                        "cpu"
                    },
            "external helper native envelope identity differs",
        )?;
        let n = &v["native"];
        require(
            array_hash(&n["model_epoch"])? == checkpoint.sha256
                && array_hash(&n["export_manifest_sha256"])? == native.export.sha256
                && array_hash(&n["encoding_semantic_sha256"])? == native.encoding_semantic_sha256
                && array_hash(&n["adapter_source_sha256"])? == native.adapter_source_sha256
                && n["trained"]
                    == matches!(endpoint.model.weights, PalsWeightIdentityV3::Trained { .. })
                && (n["frozen_epoch"].as_u64() == Some(endpoint.model.frozen_epoch)
                    || (endpoint.model.frozen_epoch == 0 && n["frozen_epoch"].is_null()))
                && count(n, "process_epoch")? > 0
                && n["execution"]["host_record_pages"].is_null()
                && n["host_record_page_observation"].is_null(),
            "external helper native checkpoint/model/encoding/epoch or undeclared host mode differs",
        )?;
        require_no_undeclared_private_warm(n)?;
        require_no_cuda_record_pages(n)?;
        let execution = &n["execution"];
        require(
            execution.is_object()
                && array_hash(&execution["runtime_sha256"])? == native.runtime.sha256,
            "external helper native execution/runtime observation missing",
        )?;
        match lock.recipe(role).cuda_model() {
            Some(cuda) => require(
                execution["provider"] == "cuda"
                    && execution["device_id"] == cuda.device_id
                    && execution["session_arena_bytes"] == cuda.session_arena_bytes
                    && v["runtime_bundle_sha256"] == cuda.cuda_bundle.canonical_sha256
                    && array_hash(&execution["runtime_bundle_sha256"])?
                        == cuda.cuda_bundle.canonical_sha256,
                "external helper native CUDA model execution differs",
            )?,
            None => require(
                execution["provider"] == "cpu"
                    && execution["device_id"].is_null()
                    && execution["session_arena_bytes"].is_null()
                    && execution["runtime_bundle_sha256"].is_null()
                    && v["runtime_bundle_sha256"].is_null(),
                "external helper native CPU execution differs",
            )?,
        }
        require(
            wire.schema_version == 1
                && wire.domain == "rz-pals-checker-process/1"
                && wire.profile_file_sha256 == loaded.file_sha256()
                && wire.profile_canonical_sha256 == loaded.canonical_sha256()
                && wire.registration == registration
                && wire.registration_sha256 == registration_sha256
                && wire.startup_handshake_completed
                && wire.requested_option_checks_completed
                && wire.applied_option_values == "unknown",
            "external helper full registration/profile/resolver/option barrier differs",
        )?;
        let observed = wire
            .observed_uci
            .as_ref()
            .ok_or_else(|| invalid("external helper UCI identity unobserved"))?;
        require(
            observed.name == profile.expected_uci_name
                && observed.name.len() <= 1024
                && !observed.name.chars().any(char::is_control)
                && observed
                    .author
                    .as_ref()
                    .is_none_or(|a| a.len() <= 1024 && !a.chars().any(char::is_control)),
            "external helper observed UCI name/author differs or is unbounded",
        )?;
        require(
            wire.latest_attempt
                .as_ref()
                .is_none_or(serde_json::Value::is_object),
            "external helper latest diagnostic attempt malformed",
        )?;
        validate_pals_process_counter_units(
            &v["search_work"],
            "pals",
            !ending,
            true,
            lock.native_wire_version() == 3,
        )?;
        for field in [
            "cpu_tasks_requested",
            "cpu_tasks",
            "completed_cpu_tasks",
            "reused_completed_cpu_tasks_consumed",
            "consumed_cpu_tasks",
            "cpu_nodes",
        ] {
            require(
                work_count(&v["search_work"]["pals"], field)? == 0,
                "external helper contains own CPU work",
            )?;
        }
    }
    require(
        s["native"]["process_epoch"] == t["native"]["process_epoch"]
            && s["native"]["execution"] == t["native"]["execution"],
        "external helper native process epoch/execution changed",
    )?;
    require(
        !sw.cleanup_complete
            && !sw.started_owner_exit_and_drains_confirmed
            && sw.shutdown.is_none()
            && tw.cleanup_complete
            && tw.started_owner_exit_and_drains_confirmed,
        "external helper startup/final owner closure boundary invalid",
    )?;
    let ready_resources = tw
        .startup_resource_observation
        .ok_or_else(|| invalid("external helper actual ready resource snapshot missing"))?;
    let shutdown = tw
        .shutdown
        .ok_or_else(|| invalid("external helper final shutdown observation missing"))?;
    let helper = shutdown
        .process_identity
        .as_ref()
        .ok_or_else(|| invalid("external helper actual historical spawn identity missing"))?;
    ready_resources.validate_against(expected_parent_pid, helper, lock.resource(role), p)?;
    require(
        t["native"]["physical_shutdown_confirmed"] == true
            && t["native"]["native_buffers_released"] == true
            && t["native"]["quarantined"] == false
            && count(&t["native"], "physical_runs_in_flight")? == 0,
        "external helper closure cannot replace independent native physical shutdown",
    )?;
    if purpose == PalsExternalCpuRSessionPurposeV3::Game {
        require(
            count(&t["native"], "completed_new_game_resets")? >= 1
                && count(&t["native"], "game_generation")?
                    > count(&s["native"], "game_generation")?,
            "external helper game session lacks native game reset boundary",
        )?;
    }
    let totals = &t["search_work"]["pals"];
    let work = PalsExternalCpuRWorkV3 {
        tasks_dispatched: optional_foreign_count(totals, "external_checker_tasks")?,
        reports_returned: optional_foreign_count(totals, "external_checker_reports")?,
        node_budget_reserved: optional_foreign_count(
            totals,
            "external_checker_node_budget_reserved",
        )?,
        nodes_observed: optional_foreign_count(totals, "external_checker_nodes_observed")?,
        consumed_completed_tasks: optional_foreign_count(
            totals,
            "consumed_external_checker_tasks",
        )?,
        work_incomplete: totals["external_checker_work_incomplete"].as_bool(),
        completed_tasks: PalsObservedV3::Unknown,
        reused_completed_task_consumptions: PalsObservedV3::Unknown,
    };
    let receipt = PalsExternalCpuRReceiptV3 {
        profile_file_sha256: loaded.file_sha256().into(),
        profile_canonical_sha256: loaded.canonical_sha256().into(),
        registered_binary_sha256: declaration.binary.sha256.clone(),
        registration_sha256,
        resolver: declaration.resolver.clone(),
        model_value,
        ready_resources,
        shutdown,
        work,
        applied_option_values: PalsObservedV3::Unknown,
        model_loading: PalsObservedV3::Unknown,
    };
    receipt.validate_against(endpoint, lock.resource(role))?;
    Ok(PalsExternalCpuRSessionAuditV3 {
        purpose,
        endpoint_id: endpoint.id.clone(),
        parent_process_id: expected_parent_pid,
        startup_sha256: digest(startup),
        termination_sha256: digest(termination),
        registration,
        receipt,
    })
}

#[cfg(all(
    target_os = "linux",
    any(feature = "pals-collection-onnx", feature = "native-cuda")
))]
fn optional_foreign_count(
    value: &serde_json::Value,
    field: &str,
) -> Result<Option<u64>, ArenaError> {
    let v = value
        .get(field)
        .ok_or_else(|| invalid(format!("foreign work field missing: {field}")))?;
    if v.is_null() {
        Ok(None)
    } else {
        v.as_u64()
            .map(Some)
            .ok_or_else(|| invalid(format!("foreign work field type invalid: {field}")))
    }
}

// serde's struct duplicate checks do not cover nested Value/BTreeMap fields.
// Keep actual evidence bytes unambiguous before full registration comparison.
#[cfg(all(
    target_os = "linux",
    any(feature = "pals-collection-onnx", feature = "native-cuda")
))]
fn unique_helper_json(bytes: &[u8]) -> Result<serde_json::Value, ArenaError> {
    struct Unique(serde_json::Value);
    impl<'de> Deserialize<'de> for Unique {
        fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
            struct V;
            impl<'de> serde::de::Visitor<'de> for V {
                type Value = Unique;
                fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                    f.write_str("JSON with unique object keys")
                }
                fn visit_bool<E: serde::de::Error>(self, v: bool) -> Result<Unique, E> {
                    Ok(Unique(v.into()))
                }
                fn visit_u64<E: serde::de::Error>(self, v: u64) -> Result<Unique, E> {
                    Ok(Unique(v.into()))
                }
                fn visit_i64<E: serde::de::Error>(self, v: i64) -> Result<Unique, E> {
                    Ok(Unique(v.into()))
                }
                fn visit_f64<E: serde::de::Error>(self, v: f64) -> Result<Unique, E> {
                    serde_json::Number::from_f64(v)
                        .map(|v| Unique(v.into()))
                        .ok_or_else(|| E::custom("nonfinite JSON"))
                }
                fn visit_str<E: serde::de::Error>(self, v: &str) -> Result<Unique, E> {
                    Ok(Unique(v.into()))
                }
                fn visit_string<E: serde::de::Error>(self, v: String) -> Result<Unique, E> {
                    Ok(Unique(v.into()))
                }
                fn visit_unit<E: serde::de::Error>(self) -> Result<Unique, E> {
                    Ok(Unique(serde_json::Value::Null))
                }
                fn visit_none<E: serde::de::Error>(self) -> Result<Unique, E> {
                    self.visit_unit()
                }
                fn visit_seq<A: serde::de::SeqAccess<'de>>(
                    self,
                    mut a: A,
                ) -> Result<Unique, A::Error> {
                    let mut v = Vec::new();
                    while let Some(Unique(item)) = a.next_element()? {
                        v.push(item);
                    }
                    Ok(Unique(v.into()))
                }
                fn visit_map<A: serde::de::MapAccess<'de>>(
                    self,
                    mut a: A,
                ) -> Result<Unique, A::Error> {
                    let mut v = serde_json::Map::new();
                    while let Some(key) = a.next_key::<String>()? {
                        if v.contains_key(&key) {
                            return Err(serde::de::Error::custom(
                                "duplicate external helper JSON key",
                            ));
                        }
                        let Unique(item) = a.next_value()?;
                        v.insert(key, item);
                    }
                    Ok(Unique(v.into()))
                }
            }
            d.deserialize_any(V)
        }
    }
    serde_json::from_slice::<Unique>(bytes)
        .map(|v| v.0)
        .map_err(|e| invalid(format!("external helper JSON: {e}")))
}

/// Verify actual profile bytes/canonical identity against a prepared semantic
/// lock and explicit registered Linux CAS program/cwd paths. This starts no
/// checker or NN, reads no executable/model, and grants no arena launch/Core
/// eligibility. Time, cgroup, preflight and owner closure remain separate gates.
#[cfg(all(
    target_os = "linux",
    any(feature = "pals-collection-onnx", feature = "native-cuda")
))]
pub fn verify_pals_external_cpu_r_profile(
    semantic_lock: &PalsInputLockV3,
    role: NativeEngineRole,
    source_root: &Path,
    registered_program: &Path,
    registered_cwd: &Path,
) -> Result<PalsExternalCpuRProfileAuditV3, ArenaError> {
    use rz_experiments::PalsCpuRSelectionV3;
    use rz_uci::pals_checker_profile::{PalsCheckerProfile, PalsCheckerSelection};

    semantic_lock.verify()?;
    let manifest = &semantic_lock.manifest;
    let PalsEngineV3::Pals(endpoint) = &manifest.engines[role_index(role)] else {
        return Err(invalid("external CPU_R profile requires a PALS endpoint"));
    };
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
        domain: "rz-pals-external-cpu-r-profile-audit-v3/1",
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

#[cfg(not(all(
    target_os = "linux",
    any(feature = "pals-collection-onnx", feature = "native-cuda")
)))]
pub fn verify_pals_external_cpu_r_profile(
    _semantic_lock: &PalsInputLockV3,
    _role: NativeEngineRole,
    _source_root: &Path,
    _registered_program: &Path,
    _registered_cwd: &Path,
) -> Result<PalsExternalCpuRProfileAuditV3, ArenaError> {
    Err(invalid(
        "unsupported external CPU_R profile verification: Linux and pals-collection-onnx or native-cuda are required; no fallback or process was started",
    ))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PalsRunnerStatusPatchV3 {
    LegacyClockOnly,
    ClockAndReapStatus,
}

fn invalid(reason: impl Into<String>) -> ArenaError {
    ArenaError::Integrity(format!("PALS arena V3: {}", reason.into()))
}
fn require(ok: bool, reason: &str) -> Result<(), ArenaError> {
    if ok { Ok(()) } else { Err(invalid(reason)) }
}
fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn json(bytes: &[u8]) -> Result<serde_json::Value, ArenaError> {
    let text = std::str::from_utf8(bytes).map_err(|_| invalid("JSON evidence is not UTF-8"))?;
    crate::decode_json(text)
}
fn hash(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
fn name(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 128
        && value
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"._-".contains(&b))
        && matches!(
            Path::new(value).components().collect::<Vec<_>>().as_slice(),
            [Component::Normal(_)]
        )
}
fn role_index(role: NativeEngineRole) -> usize {
    match role {
        NativeEngineRole::Baseline => 0,
        NativeEngineRole::Candidate => 1,
    }
}
/// Resolve the parent user's existing cache before the engine's cleared child
/// environment. A cache open never supplies NN-ready or placement evidence.
fn pals_shared_runtime_argument() -> Result<OsString, ArenaError> {
    let cache = rz_eval::runtime_pin::RuntimeCache::for_user()
        .map_err(|e| invalid(format!("shared runtime cache preparation:{:?}", e.kind)))?;
    let path = cache
        .root()
        .to_str()
        .ok_or_else(|| invalid("shared runtime root is not UTF-8"))?;
    Ok(format!("--pals-runtime-cache-root={path}").into())
}

/// These are actual accepted CLI limits, separately pinned from declarations.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct PalsSearchLaunchV3 {
    pub max_rounds: u64,
    pub max_cpu_nodes: u64,
    pub cpu_depth: u16,
    pub max_situations: u32,
    /// Explicit search semantics only; omission preserves the legacy recipe.
    /// A present null is invalid and cannot silently select the old policy.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "deserialize_post_repair_recheck"
    )]
    pub post_repair_recheck: Option<PalsPostRepairRecheckPolicyV3>,
}
fn deserialize_post_repair_recheck<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<PalsPostRepairRecheckPolicyV3>, D::Error> {
    PalsPostRepairRecheckPolicyV3::deserialize(deserializer).map(Some)
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct PalsGraphAssetV3 {
    /// Original single filename in the hashed export descriptor.
    pub file: String,
    pub role: String,
    pub artifact: ArtifactRef,
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct PalsOnnxLaunchV3 {
    pub export: ArtifactRef,
    pub export_file: String,
    pub graphs: Vec<PalsGraphAssetV3>,
    pub runtime: ArtifactRef,
    /// Semantic and source identities expected from the actual native producer.
    pub encoding_semantic_sha256: String,
    pub adapter_source_sha256: String,
    pub search: PalsSearchLaunchV3,
    /// Exact paths in the independently registered Linux CAS. The unchanged
    /// profile is retained in the private input snapshot and still names this
    /// program/cwd; the retained binary copy is evidence, not the executed path.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub external_cpu_r: Option<PalsExternalCpuRLaunchV3>,
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct PalsExternalCpuRLaunchV3 {
    pub registered_program: String,
    pub registered_working_directory: String,
}
impl PalsExternalCpuRLaunchV3 {
    fn validate(&self) -> Result<(), ArenaError> {
        for path in [&self.registered_program, &self.registered_working_directory] {
            require(
                path.len() <= 4096
                    && path.starts_with('/')
                    && !path.contains('\\')
                    && !path.chars().any(char::is_control)
                    && path
                        .split('/')
                        .skip(1)
                        .all(|part| !part.is_empty() && part != "." && part != ".."),
                "external CPU_R registered program/cwd requires a bounded exact absolute Linux path",
            )?;
            require_public_artifact_path(path)?;
        }
        require(
            Path::new(&self.registered_program).parent()
                == Some(Path::new(&self.registered_working_directory)),
            "external CPU_R registered program must reside in its exact registered CAS cwd",
        )
    }
}
/// Explicit CUDA execution identity; the session arena is a declaration, never
/// an observed device peak. Host K/V and physical B1 remain the first recipe.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct PalsOnnxCudaLaunchV3 {
    pub model: PalsOnnxLaunchV3,
    pub cuda_bundle: rz_experiments::CudaBundleBindingV1,
    pub device_id: i32,
    pub session_arena_bytes: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cuda_control: Option<Box<PalsCudaControlBindingV3>>,
    /// One logical probe deadline after cold model construction, shared by all
    /// P/C and NN-zero startup commands. None preserves the existing 15s
    /// default/argv/canonical. This is separate from the whole external
    /// readiness window and never extends physical drain or the pair wall cap.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub startup_probe_timeout_ms: Option<u64>,
    /// Explicit auxiliary packing registration; omission preserves the legacy
    /// two-model-graph recipe. Declaration is never a device Run/fence witness.
    /// The selected configuration is owned once at launch; unselected endpoints
    /// do not carry its full resource DTO inline. Box is transparent on the wire.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "deserialize_present_cuda_record_pages"
    )]
    pub cuda_record_pages: Option<Box<PalsCudaRecordPagesBindingV1>>,
}
#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub enum PalsCudaRecordPagesModeV1 {
    #[serde(rename = "registered-packing-v1")]
    RegisteredPackingV1,
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct PalsCudaRecordPagesBindingV1 {
    pub mode: PalsCudaRecordPagesModeV1,
    /// Independently registered implementation provenance, distinct from the
    /// model adapter/encoding and from observed Run/physical completion.
    pub implementation_sha256: String,
    pub manifest: ArtifactRef,
    pub graph: ArtifactRef,
    pub resources: NativeCudaRecordPagesResourceInput,
}
fn deserialize_present_cuda_record_pages<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<Box<PalsCudaRecordPagesBindingV1>>, D::Error> {
    Box::<PalsCudaRecordPagesBindingV1>::deserialize(deserializer).map(Some)
}
impl PalsCudaRecordPagesBindingV1 {
    fn resources_json(&self) -> Result<String, ArenaError> {
        self.resources
            .validate()
            .map_err(|error| invalid(error.to_string()))?;
        let text =
            serde_json::to_string(&self.resources).map_err(|error| invalid(error.to_string()))?;
        require(
            text.len() <= MAX_CUDA_RECORD_PAGES_RESOURCES_BYTES,
            "resident CUDA resources require bounded inline JSON",
        )?;
        Ok(text)
    }
    fn validate(&self, cuda: &PalsOnnxCudaLaunchV3) -> Result<(), ArenaError> {
        self.manifest.validate()?;
        self.graph.validate()?;
        require_public_artifact(&self.manifest)?;
        require_public_artifact(&self.graph)?;
        self.resources_json()?;
        let r = &self.resources;
        let retained = self
            .manifest
            .bytes
            .checked_add(self.graph.bytes)
            .ok_or_else(|| invalid("resident packing artifact byte overflow"))?;
        require(
            hash(&self.implementation_sha256)
                && self.implementation_sha256 != "0".repeat(64)
                && hash(&self.manifest.sha256)
                && hash(&self.graph.sha256)
                && (1..=512 * 1024).contains(&self.manifest.bytes)
                && (1..=2 * 1024 * 1024).contains(&self.graph.bytes)
                && self.manifest.path != self.graph.path
                && retained <= r.packing_artifact.max_owned_artifact_host_bytes
                && retained
                    .checked_add(r.packing_artifact.additional_owner_metadata_host_bytes)
                    .is_some_and(|n| n <= r.packing_artifact.max_declared_host_bytes)
                && cuda.cuda_control.is_some()
                && r.invocation.public_session_bytes == cuda.session_arena_bytes
                && r.invocation.private_session_bytes == cuda.session_arena_bytes,
            "resident packing pins/resources require separate registered artifacts, model control gate and exact model arenas",
        )
    }
    fn declared_resources(&self) -> serde_json::Value {
        let l = &self.resources.limits;
        let i = &self.resources.invocation;
        serde_json::json!({"max_blocks":l.max_blocks,"max_bank_whole_payload_bytes":l.max_bank_whole_payload_bytes,
            "max_registry_entries":l.max_registry_entries,"max_container_host_bytes":l.max_container_host_bytes,
            "max_invocation_host_bytes":l.max_invocation_host_bytes,"max_invocation_device_bytes":l.max_invocation_device_bytes,
            "public_session_bytes":i.public_session_bytes,"packing_session_bytes":i.packing_session_bytes,
            "private_session_bytes":i.private_session_bytes,"private_output_payload":i.private_output_payload,
            "original_input_and_transfer_payload":i.original_input_and_transfer_payload,
            "additional_owner_metadata_payload":i.additional_owner_metadata_payload})
    }
}
/// Explicit reviewed metadata inventory; it does not declare GPU success.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct PalsCudaControlBindingV3 {
    pub inventory: ArtifactRef,
    /// Explicit experimental loading policy, independent of the unchanged
    /// nineteen-file binary bundle and the control/NN placement inventory.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub loading_profile: Option<PalsCudaLoadingBindingV1>,
    /// Separate experimental host-runtime selection; omission preserves the
    /// historical control envelope, canonical digest and executable argv.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "deserialize_present_cudnn_conv"
    )]
    pub cudnn_conv_search: Option<PalsCuDnnConvBindingV1>,
}
#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub enum PalsCuDnnConvProfileV1 {
    #[serde(rename = "default-host-control-v1")]
    DefaultHostControlV1,
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct PalsCuDnnConvBindingV1 {
    pub profile: PalsCuDnnConvProfileV1,
    pub backend_source_sha256: String,
    pub descriptor_sha256: String,
}
fn deserialize_present_cudnn_conv<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<PalsCuDnnConvBindingV1>, D::Error> {
    PalsCuDnnConvBindingV1::deserialize(deserializer).map(Some)
}
fn cudnn_conv_binding(cuda: &PalsOnnxCudaLaunchV3) -> Option<&PalsCuDnnConvBindingV1> {
    cuda.cuda_control.as_ref()?.cudnn_conv_search.as_ref()
}
impl PalsCuDnnConvBindingV1 {
    fn descriptor(&self, cuda: &PalsOnnxCudaLaunchV3, checkpoint: &str) -> serde_json::Value {
        serde_json::json!({
            "schema":"rz-pals-cudnn-default-loaded-execution/1",
            "scope":"loaded_backend_request_not_selected_plan_or_process_exit",
            "selected_profile":"default-host-control-v1",
            "manifest_sha256":cuda.model.export.sha256,"model_epoch":checkpoint,
            "backend_source_sha256":self.backend_source_sha256,
            "runtime_sha256":cuda.model.runtime.sha256,
            "runtime_bundle_sha256":cuda.cuda_bundle.canonical_sha256,
            "control_inventory_sha256":cuda.cuda_control.as_ref().map(|c|&c.inventory.sha256),
            "provider":"CUDAExecutionProvider","device_id":cuda.device_id,
            "session_arena_bytes":cuda.session_arena_bytes,
            "intra_threads":2,"inter_threads":1,"parallel_execution":false,
            "cache_public_memory":true,"public_memory":"host",
            "native_loading_profile":"eager-cuda12-cudnn9-v1","graph_optimization":"disable",
            "cudnn_conv_algo_search_requested":"DEFAULT","tf32_requested":false,"cuda_graph_requested":false,
            "nvidia_tf32_override_environment_request":"absent"
        })
    }
    fn validate(&self, cuda: &PalsOnnxCudaLaunchV3, checkpoint: &str) -> Result<(), ArenaError> {
        let control = cuda
            .cuda_control
            .as_ref()
            .ok_or_else(|| invalid("DEFAULT requires inventory control"))?;
        require(
            hash(&self.backend_source_sha256)
                && hash(&self.descriptor_sha256)
                && self.backend_source_sha256
                    == digest(include_bytes!("../../rz-eval/src/pals_onnx.rs"))
                && hash(checkpoint)
                && control.loading_profile.is_none()
                && cuda.cuda_record_pages.is_none()
                && cuda.session_arena_bytes == 2u64 << 30
                && cuda.model.graphs.len() == 2
                && cuda
                    .model
                    .graphs
                    .iter()
                    .map(|g| g.role.as_str())
                    .collect::<BTreeSet<_>>()
                    == BTreeSet::from(["public", "shared_pc"]),
            "DEFAULT host candidate requires exact compiled backend source, eager inventory, two public/shared graphs and unchanged two-GiB model arenas",
        )?;
        control.inventory.validate()?;
        require_public_artifact(&control.inventory)?;
        require(
            (1..=MAX_CONTROL_INVENTORY_BYTES).contains(&control.inventory.bytes),
            "DEFAULT inventory bound differs",
        )?;
        let bytes = serde_json::to_vec(&self.descriptor(cuda, checkpoint))
            .map_err(|e| invalid(e.to_string()))?;
        require(
            digest(&bytes) == self.descriptor_sha256,
            "DEFAULT declared loaded-execution descriptor hash differs",
        )
    }
}
fn validate_cudnn_conv_pair(endpoints: &[PalsEndpointLaunchV3; 2]) -> Result<(), ArenaError> {
    let selected = endpoints
        .iter()
        .filter_map(PalsEndpointLaunchV3::cuda_model)
        .filter(|c| cudnn_conv_binding(c).is_some())
        .count();
    require(
        selected == 0 || selected == 2,
        "DEFAULT host runtime candidate must be selected for both PALS CUDA model endpoints",
    )
}
#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub enum PalsCudaLoadingProfileV1 {
    #[serde(rename = "experimental-cudnn-shim-lazy-v1")]
    ExperimentalCudnnShimLazyV1,
}
impl PalsCudaLoadingProfileV1 {
    fn native(self) -> rz_native_loader::NativeLoadingProfile {
        match self {
            Self::ExperimentalCudnnShimLazyV1 => {
                rz_native_loader::NativeLoadingProfile::CuDnnShimLazyV1
            }
        }
    }
    pub fn canonical_sha256(self) -> String {
        digest(self.native().canonical_descriptor().as_bytes())
    }
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct PalsCudaLoadingBindingV1 {
    pub profile: PalsCudaLoadingProfileV1,
    /// SHA-256 of NativeLoadingProfile's canonical descriptor bytes, never an
    /// inventory file hash or a replacement CUDA library bundle digest.
    pub canonical_sha256: String,
}
impl PalsCudaLoadingBindingV1 {
    fn validate(&self) -> Result<(), ArenaError> {
        require(
            hash(&self.canonical_sha256)
                && self.canonical_sha256 == self.profile.canonical_sha256(),
            "experimental CUDA loading descriptor canonical hash differs",
        )
    }
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(
    tag = "recipe",
    content = "configuration",
    rename_all = "snake_case",
    deny_unknown_fields
)]
pub enum PalsEndpointLaunchV3 {
    LegalOrderMock(PalsSearchLaunchV3),
    OnnxCpu(PalsOnnxLaunchV3),
    OnnxCuda(Box<PalsOnnxCudaLaunchV3>),
    OwnCpu {
        max_depth: u16,
        max_nodes: u64,
        tt_entries: u32,
    },
    ReferenceUci {
        expected_uci_name: String,
        arguments: Vec<String>,
        environment: Option<EngineEnvironmentV2>,
    },
}
impl PalsEndpointLaunchV3 {
    fn native_model(&self) -> Option<&PalsOnnxLaunchV3> {
        match self {
            Self::OnnxCpu(model) => Some(model),
            Self::OnnxCuda(cuda) => Some(&cuda.model),
            _ => None,
        }
    }
    fn cuda_model(&self) -> Option<&PalsOnnxCudaLaunchV3> {
        if let Self::OnnxCuda(cuda) = self {
            Some(cuda)
        } else {
            None
        }
    }
}

/// A distinct arena envelope pins runner, opening, executable recipes and I/O
/// budgets. A V3 semantic lock alone does not authorize a child process.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct PalsArenaLaunchV3 {
    pub domain: String,
    pub semantic_lock: PalsInputLockV3,
    pub runner: ToolIdentity,
    pub opening_artifact: ArtifactRef,
    pub endpoints: [PalsEndpointLaunchV3; 2],
    pub budget: NativeResourceBudgetV1,
}
#[derive(Clone, Debug)]
pub struct LockedPalsArenaLaunchV3 {
    input: PalsArenaLaunchV3,
    sha256: String,
    opening: OpeningSpec,
    endpoint_views: [ExternalUciEndpointV2; 2],
}
impl PalsArenaLaunchV3 {
    fn external_cpu_r_binding(
        &self,
        i: usize,
    ) -> Result<Option<(&PalsExternalCpuRV3, &PalsExternalCpuRLaunchV3)>, ArenaError> {
        let binding = self.endpoints[i]
            .native_model()
            .and_then(|native| native.external_cpu_r.as_ref());
        match &self.semantic_lock.manifest.engines[i] {
            PalsEngineV3::Pals(endpoint) => match &endpoint.cpu_r {
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
        require(
            self.endpoints
                .iter()
                .filter_map(PalsEndpointLaunchV3::cuda_model)
                .all(|c| cudnn_conv_binding(c).is_none()),
            "DEFAULT host runtime candidate requires a separately registered V4 arena pilot",
        )?;
        self.semantic_lock.verify()?;
        for index in 0..2 {
            validate_launch_search_policy(self, index)?;
            if let Some(cuda) = self.endpoints[index].cuda_model()
                && let Some(pages) = &cuda.cuda_record_pages
            {
                pages.validate(cuda)?;
                require(
                    matches!(&self.semantic_lock.manifest.engines[index],
                    PalsEngineV3::Pals(e) if matches!(&e.cpu_r, PalsCpuRSelectionV3::Own)
                        && e.model.frozen_epoch == 1),
                    "resident CUDA registration requires deployment epoch 1 and own CPU_R; external helper is unsupported",
                )?;
            }
            if self.external_cpu_r_binding(index)?.is_some() {
                require(
                    cfg!(all(
                        target_os = "linux",
                        any(feature = "pals-collection-onnx", feature = "native-cuda")
                    )),
                    "unsupported external CPU_R launch: Linux and pals-collection-onnx or native-cuda are required; no fallback",
                )?;
                require(
                    self.endpoints[index].native_model().is_some(),
                    "external CPU_R launch requires an explicitly registered native PALS endpoint",
                )?;
            }
        }
        require(
            self.domain == PALS_ARENA_V3_DOMAIN,
            "wrong arena launch domain",
        )?;
        let pilot = &self.semantic_lock.manifest.pilot;
        require(
            pilot.max_plies.is_multiple_of(2),
            "Fastchess max-plies must be even",
        )?;
        require(
            self.runner.source_url == crate::FASTCHESS_SOURCE_URL
                && self.runner.source_commit == crate::FASTCHESS_SOURCE_COMMIT
                && (self.runner.version == crate::FASTCHESS_VERSION
                    || registered_reap_runner(
                        &self.runner.version,
                        &self.runner.binary.sha256,
                        self.runner.binary.bytes,
                    ))
                && self.runner.dirty
                && self.runner_status_patch().is_ok(),
            "clock-audited pinned Fastchess and an exact registered patch are required",
        )?;
        self.runner.binary.validate()?;
        self.runner
            .dirty_patch
            .as_ref()
            .ok_or_else(|| invalid("missing runner patch"))?
            .validate()?;
        self.opening_artifact.validate()?;
        let b = self.budget;
        let unique = self.unique_input_bytes()?;
        let cuda_count = self
            .endpoints
            .iter()
            .filter(|e| e.cuda_model().is_some())
            .count() as u64;
        let cpu_native_count = self
            .endpoints
            .iter()
            .filter(|e| matches!(e, PalsEndpointLaunchV3::OnnxCpu(_)))
            .count() as u64;
        // Historical PALS CUDA locks with a finite per-role cache reservation
        // remain readable. New children share the bounded, content-addressed
        // user runtime cache outside this attempt; no per-run bundle copy is
        // included in the owned runtime tree.
        let runtime_ceiling = if cuda_count == 0 {
            2 * 1024 * 1024 * 1024
        } else {
            (4 * cuda_count + cpu_native_count) * 1024 * 1024 * 1024 + 64 * 1024 * 1024
        };
        require(
            b.max_input_bytes > 0
                && unique <= b.max_input_bytes
                && b.max_output_bytes > 0
                && b.max_output_bytes <= 64 * 1024 * 1024
                && b.max_runtime_bytes > 0
                && b.max_runtime_bytes <= runtime_ceiling
                && b.max_child_processes >= 3
                && b.max_child_processes <= 16
                && b.max_runtime_files >= 16
                && b.max_runtime_files <= 4096
                && (2..=8).contains(&b.max_runtime_depth)
                && b.address_space_per_process_bytes == 0,
            "finite I/O/process budgets required; RAM is enforced by the inherited cgroup",
        )?;
        require(
            unique
                .checked_add(b.max_output_bytes)
                .and_then(|n| n.checked_add(b.max_runtime_bytes))
                .is_some_and(|n| n <= b.max_artifact_bytes),
            "artifact budget does not cover inputs and bounded output",
        )?;
        let resources = &self.semantic_lock.manifest.resources;
        require(
            resources[0].cpu_affinity == resources[1].cpu_affinity
                && resources[0].memory_high_bytes == resources[1].memory_high_bytes
                && resources[0].memory_max_bytes == resources[1].memory_max_bytes
                && resources[0].swap_max_bytes == resources[1].swap_max_bytes,
            "this executor supports one identical inherited CPU/memory policy for the two sequential engines",
        )?;
        for (i, resource) in resources.iter().enumerate() {
            self.endpoint(i)?;
            if let PalsEngineV3::Pals(e) = &self.semantic_lock.manifest.engines[i] {
                require(
                    e.pools.host_bytes == resource.memory_max_bytes,
                    "PALS pool host ceiling differs from verified memory cgroup",
                )?;
                if let Some(cuda) = self.endpoints[i].cuda_model() {
                    let declared_arenas = cuda
                        .session_arena_bytes
                        .checked_mul(cuda.model.graphs.len() as u64)
                        .ok_or_else(|| invalid("CUDA session declaration overflow"))?;
                    require(
                        resource.requested_gpu.is_some()
                            && e.pools.device_bytes > 0
                            && e.pools.device_bytes == resource.device_allocation_max_bytes
                            && declared_arenas < e.pools.device_bytes
                            && e.pools.device_bytes <= 6 * 1024 * 1024 * 1024,
                        "CUDA explicit device budget must cover all declared session arenas; observed peak remains unknown",
                    )?;
                    if let Some(pages) = &cuda.cuda_record_pages {
                        require(
                            pages.resources.limits.max_invocation_host_bytes <= e.pools.host_bytes
                                && pages.resources.limits.max_invocation_device_bytes
                                    <= e.pools.device_bytes,
                            "resident combined owner invocation bounds exceed admitted host/device pools",
                        )?;
                    }
                } else {
                    require(
                        e.pools.device_bytes == 0
                            && resource.requested_gpu.is_none()
                            && resource.device_allocation_max_bytes == 0,
                        "closed CPU/mock recipe must not declare GPU allocation",
                    )?;
                }
            }
        }
        let cuda_bundles: Vec<_> = self
            .endpoints
            .iter()
            .filter_map(PalsEndpointLaunchV3::cuda_model)
            .map(|c| &c.cuda_bundle)
            .collect();
        require(
            cuda_bundles.windows(2).all(|pair| pair[0] == pair[1]),
            "first CUDA executor requires one exact shared runtime bundle identity for both roles",
        )?;
        let mut names = BTreeMap::new();
        let mut destinations = BTreeMap::new();
        for (artifact, target) in self.named_assets() {
            if let Some(prior) = destinations.insert(&artifact.path, target.clone()) {
                require(
                    prior == target,
                    "one source artifact cannot be placed in multiple export namespaces by this executor",
                )?;
            }
            if let Some(prior) = names.insert(target, artifact) {
                require(prior == artifact, "snapshot destination collision")?;
            }
        }
        Ok(())
    }
    pub fn lock(&self) -> Result<LockedPalsArenaLaunchV3, ArenaError> {
        self.validate()?;
        Ok(LockedPalsArenaLaunchV3 {
            input: self.clone(),
            sha256: crate::canonical_sha256(self)?,
            opening: self.opening(),
            endpoint_views: [self.endpoint(0)?, self.endpoint(1)?],
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
    #[cfg(any(target_os = "linux", test))]
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
    #[cfg(any(target_os = "linux", test))]
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
                    matches!(engine, PalsEngineV3::Pals(e) if e.model.frozen_epoch == 1),
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
            if let PalsEngineV3::Pals(endpoint) = engine
                && let PalsCpuRSelectionV3::ExternalUci(helper) = &endpoint.cpu_r
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
                PalsEngineV3::Pals(e) => {
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
                PalsEngineV3::OwnCpu(e) => result.push(&e.binary),
                PalsEngineV3::ReferenceUci(e) => {
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
        require(
            self.endpoints[i]
                .cuda_model()
                .and_then(cudnn_conv_binding)
                .is_none(),
            "V3 cannot generate DEFAULT host runtime candidate argv",
        )?;
        let helper = self.external_cpu_r_binding(i)?;
        let m = &self.semantic_lock.manifest;
        let role = if i == 0 {
            NativeEngineRole::Baseline
        } else {
            NativeEngineRole::Candidate
        };
        let selected_policy = validate_launch_search_policy(self, i)?;
        let (binary, family, version, uci_name, source, mut arguments, assets, environment) = match (
            &m.engines[i],
            &self.endpoints[i],
        ) {
            (PalsEngineV3::Pals(e), PalsEndpointLaunchV3::LegalOrderMock(s)) => {
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
                validate_pals_search(e, s, m.pilot.wall_time_max_ms)?;
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
                PalsEngineV3::Pals(e),
                recipe @ (PalsEndpointLaunchV3::OnnxCpu(_) | PalsEndpointLaunchV3::OnnxCuda(_)),
            ) => {
                let n = recipe.native_model().expect("native pattern selected");
                let cuda = recipe.cuda_model();
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
                        && e.model.input_schema == "rz-pals-rules-fields-v1"
                        && e.model.policy_head == "candidate-policy/1"
                        && e.model.value_head == "stm-wdl/1"
                        && e.model.implementation_sha256 == n.adapter_source_sha256,
                    "native model topology/input/head/adapter declaration differs from supported recipe",
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
                validate_pals_search(e, &n.search, m.pilot.wall_time_max_ms)?;
                let mut args = pals_search_arguments(&n.search, "onnx");
                args.extend([
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
                        "--pals-device-public-memory=false".into(),
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
                PalsEngineV3::OwnCpu(e),
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
                PalsEngineV3::ReferenceUci(e),
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
        if let Some(policy) = &selected_policy {
            // Append exactly once after all registered native/provider options.
            let option =
                if policy == &PalsSearchPolicyIdentityV3::expected_same_repaired_line_once_v1() {
                    PalsPostRepairRecheckPolicyV3::SameRepairedLineOnceV1.option_value()
                } else if policy
                    == &PalsSearchPolicyIdentityV3::expected_actual_opponent_continuation_v1()
                {
                    PalsPostRepairRecheckPolicyV3::ActualOpponentContinuationV1.option_value()
                } else {
                    return Err(invalid("unsupported selected post-Repair launch identity"));
                };
            arguments.push(format!("--pals-post-repair-recheck={option}"));
        }
        require(
            arguments.len() <= 32
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
            requested_options: m.engines[i].requested_options().clone(),
            environment,
            handshake_timeout_ms: m.pilot.handshake_max_ms,
        })
    }
}
fn validate_cpu_profile(cpu: &rz_experiments::PalsOwnCpuV3, wall: u64) -> Result<(), ArenaError> {
    require(
        cpu.core.semantic_id == "rz-cpu-pvs/0.1"
            && cpu.runtime_profile.semantic_id == "cpu-plan-assisted-conservative-v1"
            && cpu.evaluation.semantic_id == "bootstrap-material-pst-v1"
            && cpu.max_task_ms >= wall
            && cpu.core.options.is_empty()
            && cpu.runtime_profile.options.is_empty()
            && cpu.evaluation.options.is_empty(),
        "first executable recipe is PlanAssisted/bootstrap; task time declaration must cover its inherited finite deadline",
    )
}
fn validate_pals_search(
    e: &rz_experiments::PalsEndpointV3,
    s: &PalsSearchLaunchV3,
    wall: u64,
) -> Result<(), ArenaError> {
    validate_search_policy_selection(e, s)?;
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
fn validate_search_policy_selection(
    engine: &rz_experiments::PalsEndpointV3,
    search: &PalsSearchLaunchV3,
) -> Result<Option<PalsSearchPolicyIdentityV3>, ArenaError> {
    match search.post_repair_recheck {
        None => {
            require(
                engine.search.semantic_id == "pals" && engine.search.options.is_empty(),
                "legacy PALS recipe requires the unchanged pals/options-empty semantic lane",
            )?;
            Ok(None)
        }
        Some(selected) => {
            require(
                engine.cpu_r.is_own()
                    && engine.search.semantic_id == selected.version()
                    && engine.search.options.len() == 1
                    && engine
                        .search
                        .options
                        .get("post_repair_recheck")
                        .map(String::as_str)
                        == Some(selected.option_value()),
                "post-Repair recheck requires exact manifest/recipe selection and own CPU_R; external dispatch is unsupported",
            )?;
            Ok(Some(selected.expected_identity()))
        }
    }
}
fn validate_launch_search_policy(
    launch: &PalsArenaLaunchV3,
    index: usize,
) -> Result<Option<PalsSearchPolicyIdentityV3>, ArenaError> {
    let search = match &launch.endpoints[index] {
        PalsEndpointLaunchV3::LegalOrderMock(search) => Some(search),
        endpoint => endpoint.native_model().map(|native| &native.search),
    };
    match (&launch.semantic_lock.manifest.engines[index], search) {
        (PalsEngineV3::Pals(engine), Some(search)) => {
            validate_search_policy_selection(engine, search)
        }
        (_, None) => Ok(None),
        _ => Err(invalid(
            "search policy recipe has no matching PALS semantic endpoint",
        )),
    }
}
fn require_public_artifact(a: &ArtifactRef) -> Result<(), ArenaError> {
    require_public_artifact_path(&a.path)
}
fn require_public_artifact_path(path: &str) -> Result<(), ArenaError> {
    for part in path.split('/') {
        let p = part.to_ascii_lowercase();
        require(
            !p.starts_with(".env")
                && !p.ends_with(".key")
                && !p.ends_with(".pem")
                && !p.starts_with("id_rsa")
                && !p.starts_with("id_ed25519")
                && !p.contains("credential")
                && !p.contains("service-account")
                && !p.contains("service_account"),
            "secret artifacts cannot be launch inputs",
        )?;
    }
    Ok(())
}
fn pals_search_arguments(s: &PalsSearchLaunchV3, model: &str) -> Vec<String> {
    vec![
        "--search=pals".into(),
        format!("--pals-model={model}"),
        format!("--pals-max-rounds={}", s.max_rounds),
        format!("--pals-max-cpu-nodes={}", s.max_cpu_nodes),
        format!("--pals-cpu-depth={}", s.cpu_depth),
        format!("--pals-max-situations={}", s.max_situations),
    ]
}
impl LockedPalsArenaLaunchV3 {
    pub fn input(&self) -> &PalsArenaLaunchV3 {
        &self.input
    }
    pub fn sha256(&self) -> &str {
        &self.sha256
    }
    pub fn to_json(&self) -> Result<String, ArenaError> {
        serde_json::to_string_pretty(&serde_json::json!({"domain":PALS_ARENA_V3_DOMAIN,"sha256":self.sha256,"input":self.input})).map_err(|e|invalid(e.to_string()))
    }
    pub fn from_json(text: &str) -> Result<Self, ArenaError> {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Envelope {
            domain: String,
            sha256: String,
            input: PalsArenaLaunchV3,
        }
        require(
            text.len() <= MAX_JSON_BYTES,
            "locked launch JSON budget exceeded",
        )?;
        let envelope: Envelope = crate::decode_json(text)?;
        let lock = envelope.input.lock()?;
        require(
            envelope.domain == PALS_ARENA_V3_DOMAIN && envelope.sha256 == lock.sha256,
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
        let PalsEngineV3::Pals(e) = &self.input.semantic_lock.manifest.engines[role_index(role)]
        else {
            return Err(invalid("native recipe has no PALS identity"));
        };
        let (weights, trained) = match &e.model.weights {
            PalsWeightIdentityV3::Untrained { artifact, .. } => (artifact, false),
            PalsWeightIdentityV3::Trained { artifact, .. } => (artifact, true),
            _ => return Err(invalid("no native checkpoint identity")),
        };
        let shared = n.graphs.iter().any(|g| g.role == "shared_pc");
        let layout_valid = if shared {
            value["schema"] == "rovezero.pals-model.v2"
                && value["layout"] == "shared_pc_if"
                && value["layout_revision"] == 1
                && value["model_semantics"] == "rovezero.pals-model.v1"
                && value["role_batching"] == "one_scalar_role_per_physical_batch"
        } else {
            value["schema"] == "rovezero.pals-model.v1"
                && (value["layout"].is_null() || value["layout"] == "separate_pc")
        };
        require(
            layout_valid
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
        let path = runtime_root
            .to_str()
            .ok_or_else(|| invalid("CUDA profile parent is not UTF-8"))?;
        require(
            runtime_root.is_absolute() && path.len() <= 4096 && !path.chars().any(char::is_control),
            "CUDA profile parent must be a bounded owned absolute runtime root",
        )?;
        // Fastchess reuses argv between games. The UCI child derives an
        // exclusive PID+nonce directory here for each preflight/game process.
        Ok(Some(format!("--pals-cuda-profile-parent={path}").into()))
    }
}
impl crate::native_launch::sealed::Sealed for LockedPalsArenaLaunchV3 {}
// Fixed admission reservation, not a measured peak or an unbounded grow-on-save
// retry. PALS retains nested native/work/mapping evidence for up to four game
// sessions. Individual provider records remain bounded separately at 256 KiB.
const PALS_PAIR_METADATA_CAP: u64 = 2 * 1024 * 1024;
impl NativeLaunchDeclaration for LockedPalsArenaLaunchV3 {
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
                shutdown_ms: p.cleanup_max_ms / 2,
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
        "PALS-V3"
    }
    fn pair_metadata_cap(&self) -> u64 {
        PALS_PAIR_METADATA_CAP
    }
    fn amend_execution_limitations(&self, limitations: &mut Vec<String>) {
        if let Some(scope) = limitations.get_mut(0) {
            *scope="PALS V3 whole-system or explicitly controlled paired pilot; P/C native sessions and CPU/reference UCI processes are distinct; no Elo, training or model promotion".into();
        }
        if let Some(cache) = limitations.get_mut(8) {
            *cache="PALS native runtime uses the existing bounded content-addressed shared user cache outside the attempt; hits rehash and retain read-only native file pins through physical completion. Private input snapshots and per-attempt evidence remain owned and bounded separately; cache reuse is not NN inference evidence".into();
        }
        limitations.push("CPU TT byte admission covers the compiled inline slot layout; retained identity heaps, allocator overhead and total peak are separately bounded by the verified inherited memory cgroup".into());
        limitations.push("CPU_T profile is preserved as declared identity only; this pilot executes CPU_R and P/C, with V absent and training_executed=false".into());
        limitations.push("PALS reserves a fixed 2MiB pair receipt before input copying/spawn within the declared output budget; insufficient admission and oversized final metadata remain errors. Legacy V1/V2 keep their original 64KiB reservation".into());
        limitations.push("PALS handshake_max_ms bounds external process readiness, including cold runtime/model construction, the selected logical CUDA startup probe and protocol. The optional probe budget starts only after model construction and is shared across all startup commands; a probe not exceeding readiness is only a necessary condition, not guaranteed cold-start slack. Identification/readiness and each game restart may repeat this startup work. Preflight elapsed time, including separately bounded cleanup, is still deducted from the registered total pair window; no physical drain, quarantine or wall-time cap is extended".into());
        limitations.push(format!("PALS runner status patch: {:?}; legacy clock-only records remain readable, while new execution/Core requires the separately registered clock+reap-status runner", self.input.runner_status_patch()));
        if self
            .input
            .endpoints
            .iter()
            .any(|e| e.cuda_model().is_some())
        {
            limitations.push("PALS CUDA recipe fixes FP32/TF32 off, physical B1, host K/V, device 0 and per-session 2GiB arena declaration; CPU fallback/I/O binding/CUDA Graph are disabled; arena declarations and summed model device budget do not attest observed VRAM peak".into());
        }
        if self
            .input
            .endpoints
            .iter()
            .filter_map(PalsEndpointLaunchV3::cuda_model)
            .any(|cuda| {
                cuda.cuda_control
                    .as_ref()
                    .is_some_and(|control| control.loading_profile.is_some())
            })
        {
            limitations.push("Explicit experimental cuDNN shim loading changes native loading order only; the full nineteen-file bundle/cache pins remain unchanged. Mapping witnesses prove bounded origin observations, not NN/kernel placement, VRAM, normal exit, root cause or default adoption".into());
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
        Ok(arguments)
    }
    fn uses_separate_preflight_runtime_root(
        &self,
        role: NativeEngineRole,
    ) -> Result<bool, ArenaError> {
        Ok(self
            .input
            .external_cpu_r_binding(role_index(role))?
            .is_some())
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
        let isolated = self
            .input
            .external_cpu_r_binding(role_index(role))?
            .is_some();
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
        Ok(arguments)
    }
}

/// Prospective preflight evidence directory, separate from the exactly two
/// game process slots. This is path planning only: the Native capability owner
/// must create, pin and budget this sibling before the external launch guard
/// can be removed. No directory, process or cache is created here.
pub fn pals_external_cpu_r_preflight_root(
    role: NativeEngineRole,
    runtime_root: &Path,
) -> Result<PathBuf, ArenaError> {
    let name = match role {
        NativeEngineRole::Baseline => "baseline-runtime",
        NativeEngineRole::Candidate => "candidate-runtime",
    };
    require(
        runtime_root.is_absolute() && runtime_root.file_name().is_some_and(|file| file == name),
        "external CPU_R preflight requires the bound role game-runtime root",
    )?;
    let isolated = match role {
        NativeEngineRole::Baseline => "external-baseline-preflight-runtime",
        NativeEngineRole::Candidate => "external-candidate-preflight-runtime",
    };
    Ok(runtime_root
        .parent()
        .ok_or_else(|| invalid("external CPU_R preflight attempt parent missing"))?
        .join(isolated))
}

/// Accepted native evidence is recorded in the existing arena receipt. The
/// original producer fields and hashes remain available without MCTS relabeling.
#[derive(Clone, Debug, Serialize)]
pub struct PalsNativeSessionAuditV3 {
    pub endpoint_id: String,
    pub process_id: u32,
    pub launch_sha256: String,
    pub startup_sha256: String,
    pub termination_sha256: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub external_cpu_r_session: Option<PalsExternalCpuRSessionAuditV3>,
    pub completed_role_inputs: u64,
    pub search_consumed_role_inputs: u64,
    pub completed_new_game_resets: u64,
    pub physical_shutdown_confirmed: bool,
    pub native_buffers_released: bool,
    pub startup_nn_inputs_completed: u64,
    pub startup_nn_calls_completed: u64,
    pub startup_role_inputs_completed: u64,
    /// Exact selected explicit budget from the execution receipt. Missing
    /// historical/default metadata remains absent, not measured elapsed time.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub startup_probe_timeout_ms: Option<u64>,
    pub startup_probe: Option<serde_json::Value>,
    pub execution: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cuda_placement: Option<PalsCudaPlacementAuditV3>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cuda_loading: Option<PalsCudaLoadingAuditV1>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cuda_record_pages: Option<PalsCudaRecordPagesAuditV1>,
    pub raw_native: serde_json::Value,
    pub raw_search_work: serde_json::Value,
}
/// Exact raw producer observations validated against an explicit launch
/// registration. These DTOs grant no CUDA owner, completion fence or peak.
#[derive(Clone, Debug, Serialize)]
pub struct PalsCudaRecordPagesAuditV1 {
    pub selection: serde_json::Value,
    pub startup_observation: serde_json::Value,
    pub termination_observation: serde_json::Value,
    pub termination_observation_scope: &'static str,
}
/// Compact validated linkage; the original typed producer witness remains in
/// startup_probe. Device index is the exercised producer's configured index,
/// never a physical GPU UUID/model-name or VRAM-peak observation.
#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct PalsCudaPlacementAuditV3 {
    pub inventory_sha256: String,
    pub public_profile_sha256: String,
    pub shared_pc_profile_sha256: String,
    pub public_neural_kernels: u64,
    pub proposer_private_kernels: u64,
    pub critic_private_kernels: u64,
    pub device_id: i32,
}
/// Two actual origin observations at exclusive worker boundaries. These are
/// neither kernel/NN counters nor allocator/VRAM/normal-exit evidence.
#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct PalsCudaLoadingAuditV1 {
    pub profile: PalsCudaLoadingProfileV1,
    pub canonical_sha256: String,
    pub startup: PalsNativeMappingAuditV1,
    pub final_mapping: PalsNativeMappingAuditV1,
}
#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct PalsNativeMappingAuditV1 {
    pub required_nvidia_files: Vec<String>,
    pub mapped_nvidia_files: Vec<String>,
    pub deferred_nvidia_not_mapped: Vec<String>,
    pub mapped_ort_files: Vec<String>,
}
fn mapping_names(v: &serde_json::Value, allowed: &[&str]) -> Result<Vec<String>, ArenaError> {
    let values = v
        .as_array()
        .ok_or_else(|| invalid("runtime mapping filenames unobserved"))?;
    require(
        values.len() <= allowed.len(),
        "runtime mapping filename count exceeds profile",
    )?;
    let mut names = Vec::with_capacity(values.len());
    let mut unique = BTreeSet::new();
    for value in values {
        let filename = value
            .as_str()
            .ok_or_else(|| invalid("runtime mapping filename malformed"))?;
        require(
            allowed.contains(&filename) && unique.insert(filename),
            "runtime mapping filename unknown or duplicated",
        )?;
        names.push(filename.to_owned());
    }
    Ok(names)
}
fn validate_native_loading_mapping(
    cuda: &PalsOnnxCudaLaunchV3,
    loading: &PalsCudaLoadingBindingV1,
    witness: &serde_json::Value,
) -> Result<PalsNativeMappingAuditV1, ArenaError> {
    loading.validate()?;
    require(
        witness.is_object()
            && witness["schema"] == "rovezero.pals-native-mapping-witness.v1"
            && array_hash(&witness["runtime_sha256"])? == cuda.model.runtime.sha256
            && array_hash(&witness["runtime_bundle_sha256"])? == cuda.cuda_bundle.canonical_sha256
            && witness["loading_profile"] == loading.profile.native().identifier()
            && array_hash(&witness["loading_profile_sha256"])? == loading.canonical_sha256
            && witness["scope"] == "exclusive_physical_worker_full_runtime_origin"
            && count(witness, "declared_nvidia_files")?
                == rz_native_loader::NVIDIA_LOAD_ORDER.len() as u64,
        "experimental loading mapping identity/scope differs",
    )?;
    let nvidia = &rz_native_loader::NVIDIA_LOAD_ORDER;
    let required = mapping_names(&witness["required_nvidia_files"], nvidia)?;
    let mapped = mapping_names(&witness["mapped_nvidia_files"], nvidia)?;
    let deferred = mapping_names(&witness["deferred_nvidia_not_mapped"], nvidia)?;
    let ort = mapping_names(
        &witness["mapped_ort_files"],
        &rz_native_loader::ORT_LIBRARY_NAMES,
    )?;
    let required_set: BTreeSet<_> = required.iter().map(String::as_str).collect();
    let expected_required: BTreeSet<_> = loading
        .profile
        .native()
        .eager_indices()
        .iter()
        .map(|&i| nvidia[i])
        .collect();
    let mapped_set: BTreeSet<_> = mapped.iter().map(String::as_str).collect();
    let deferred_set: BTreeSet<_> = deferred.iter().map(String::as_str).collect();
    let all: BTreeSet<_> = nvidia.iter().copied().collect();
    require(
        required_set == expected_required
            && required_set.is_subset(&mapped_set)
            && mapped_set.is_disjoint(&deferred_set)
            && mapped_set
                .union(&deferred_set)
                .copied()
                .collect::<BTreeSet<_>>()
                == all
            && ort.iter().map(String::as_str).collect::<BTreeSet<_>>()
                == rz_native_loader::ORT_LIBRARY_NAMES.into_iter().collect(),
        "experimental runtime mapping lacks exact required/deferred NVIDIA partition or full ORT3",
    )?;
    Ok(PalsNativeMappingAuditV1 {
        required_nvidia_files: required,
        mapped_nvidia_files: mapped,
        deferred_nvidia_not_mapped: deferred,
        mapped_ort_files: ort,
    })
}
fn validate_native_loading_evidence(
    cuda: &PalsOnnxCudaLaunchV3,
    execution: &serde_json::Value,
    probe: &serde_json::Value,
    final_mapping: &serde_json::Value,
) -> Result<Option<PalsCudaLoadingAuditV1>, ArenaError> {
    let loading = cuda
        .cuda_control
        .as_ref()
        .and_then(|control| control.loading_profile.as_ref());
    let actual = &execution["cuda_loading_profile"];
    let startup_mapping = &probe["runtime_loading_mapping"];
    let Some(loading) = loading else {
        require(
            actual.is_null() && startup_mapping.is_null() && final_mapping.is_null(),
            "legacy/default CUDA loading cannot claim an unregistered experimental profile or mapping",
        )?;
        return Ok(None);
    };
    require(
        actual.is_object()
            && actual["profile"] == loading.profile.native().identifier()
            && array_hash(&actual["canonical_sha256"])? == loading.canonical_sha256,
        "actual experimental CUDA loading profile/hash differs",
    )?;
    Ok(Some(PalsCudaLoadingAuditV1 {
        profile: loading.profile,
        canonical_sha256: loading.canonical_sha256.clone(),
        startup: validate_native_loading_mapping(cuda, loading, startup_mapping)?,
        final_mapping: validate_native_loading_mapping(cuda, loading, final_mapping)?,
    }))
}
fn validate_native_cudnn_conv(
    cuda: Option<&PalsOnnxCudaLaunchV3>,
    wire_version: u32,
    checkpoint: &str,
    record: &serde_json::Value,
) -> Result<(), ArenaError> {
    let actual = &record["execution"]["cudnn_conv_search_diagnostic"];
    let selected = cuda.and_then(cudnn_conv_binding);
    let Some(binding) = selected else {
        return require(
            actual.is_null(),
            "undeclared cuDNN DEFAULT execution diagnostic",
        );
    };
    let cuda = cuda.expect("binding requires CUDA");
    require(
        wire_version == 4,
        "DEFAULT host candidate cannot attest a legacy native wire",
    )?;
    binding.validate(cuda, checkpoint)?;
    require(
        record["execution"]["device_public_memory"] == false
            && record["execution"]["private_warm"].is_null()
            && record["execution"]["host_record_pages"].is_null()
            && record["execution"]["cuda_record_pages"].is_null(),
        "DEFAULT host candidate cannot mix device/Warm/page execution",
    )?;
    let descriptor = binding.descriptor(cuda, checkpoint);
    require(
        actual.as_object().is_some_and(|v| v.len() == 10)
            && actual["descriptor"] == descriptor
            && array_hash(&actual["descriptor_sha256"])? == binding.descriptor_sha256
            && actual["descriptor_encoding"] == "serde_json_utf8_fixed_descriptor_object"
            && actual["actual_cudnn_plan"] == "unknown"
            && actual["numerical_note"] == "unknown"
            && actual["tf32_actual"] == "unknown"
            && actual["tf32_retry"] == "unknown"
            && actual["production_default_changed"] == false
            && actual["warm_success_claim"] == false
            && actual["pilot_success_claim"] == false,
        "actual DEFAULT descriptor/schema/hash/source/runtime/bundle/inventory/config differs from the registered host candidate",
    )
}
fn validate_native_cudnn_conv_pair(
    cuda: Option<&PalsOnnxCudaLaunchV3>,
    wire_version: u32,
    checkpoint: &str,
    startup: &serde_json::Value,
    termination: &serde_json::Value,
) -> Result<(), ArenaError> {
    if cuda.and_then(cudnn_conv_binding).is_some() {
        require(
            startup["execution"] == termination["execution"],
            "DEFAULT host execution changed between startup and final receipt",
        )?;
    }
    for record in [startup, termination] {
        validate_native_cudnn_conv(cuda, wire_version, checkpoint, record)?;
    }
    Ok(())
}
fn validate_native_startup_budget(
    cuda: &PalsOnnxCudaLaunchV3,
    execution: &serde_json::Value,
) -> Result<Option<u64>, ArenaError> {
    let actual = &execution["startup_probe_timeout_ms"];
    match cuda.startup_probe_timeout_ms {
        None => {
            require(
                actual.is_null(),
                "default CUDA startup cannot claim an unregistered explicit probe budget",
            )?;
            Ok(None)
        }
        Some(timeout_ms) => {
            require(
                (1..=MAX_STARTUP_PROBE_TIMEOUT_MS).contains(&timeout_ms)
                    && actual.as_u64() == Some(timeout_ms),
                "actual selected CUDA startup probe budget differs from the launch declaration",
            )?;
            Ok(Some(timeout_ms))
        }
    }
}
fn validate_cuda_placement_witness(
    cuda: &PalsOnnxCudaLaunchV3,
    witness: &serde_json::Value,
) -> Result<Option<PalsCudaPlacementAuditV3>, ArenaError> {
    let Some(control) = &cuda.cuda_control else {
        require(
            witness.is_null(),
            "strict CUDA cannot claim an unregistered control-policy witness",
        )?;
        return Ok(None);
    };
    require(
        witness.is_object()
            && witness["schema"] == "rovezero.pals-cuda-metadata-control.v2"
            && witness["optimization"] == "disable"
            && witness["category_provenance"]
                == "rc10-category-unavailable-id-location-message-used"
            && array_hash(&witness["inventory_sha256"])? == control.inventory.sha256
            && array_hash(&witness["manifest_sha256"])? == cuda.model.export.sha256
            && array_hash(&witness["runtime_sha256"])? == cuda.model.runtime.sha256
            && array_hash(&witness["runtime_bundle_sha256"])? == cuda.cuda_bundle.canonical_sha256,
        "CUDA placement witness does not match the registered control/model/runtime source",
    )?;
    let graphs = witness["initialization"]
        .as_array()
        .ok_or_else(|| invalid("CUDA pre-Run graph coverage absent"))?;
    require(
        graphs.len() == 2,
        "CUDA pre-Run witness requires two physical graphs",
    )?;
    for declared in &cuda.model.graphs {
        let entries: Vec<_> = graphs
            .iter()
            .filter(|g| g["role"] == declared.role)
            .collect();
        require(
            entries.len() == 1,
            "CUDA graph placement duplicated or missing",
        )?;
        let g = entries[0];
        let assigned = count(g, "assigned_nodes")?;
        let gpu = count(g, "cuda_nodes")?;
        let cpu = count(g, "approved_cpu_control_nodes")?;
        require(
            array_hash(&g["graph_sha256"])? == declared.artifact.sha256
                && hash(&array_hash(&g["log_sha256"])?)
                && (1..=5000).contains(&assigned)
                && gpu > 0
                && gpu.checked_add(cpu) == Some(assigned)
                && g["optimization"] == "disable"
                && g["recursive_coverage"]
                    == "ort-1.22-finalize-recursive-exact-named-provider-coverage-before-first-run",
            "CUDA placement lacks complete pinned pre-Run node coverage",
        )?;
        let transfers = g["approved_transfers"]
            .as_array()
            .ok_or_else(|| invalid("CUDA placement transfer declaration absent"))?;
        require(
            transfers.len() <= if declared.role == "shared_pc" { 1 } else { 0 },
            "CUDA placement permits only the pinned shared-P/C scalar transfer",
        )?;
        for transfer in transfers {
            let conditions = transfer["host_condition_nodes"]
                .as_array()
                .ok_or_else(|| invalid("CUDA transfer condition source absent"))?;
            let names: Option<BTreeSet<_>> = conditions.iter().map(|v| v.as_str()).collect();
            require(
                transfer["kind"] == "bool_scalar_host_to_device"
                    && array_hash(&transfer["graph_sha256"])? == declared.artifact.sha256
                    && transfer["source_scope"] == "shared_pc"
                    && transfer["runtime_graph_name"]
                        .as_str()
                        .is_some_and(|s| !s.is_empty() && s.len() <= 256)
                    && transfer["node_name"] == "Memcpy"
                    && transfer["opcode"] == "MemcpyFromHost"
                    && transfer["provider"] == "CUDAExecutionProvider"
                    && transfer["source_value"] == "role_is_critic"
                    && transfer["source_dtype"] == "BOOL"
                    && transfer["source_shape"]
                        .as_array()
                        .is_some_and(Vec::is_empty)
                    && transfer["destination_node"] == "output_is_critic"
                    && transfer["destination_opcode"] == "Identity"
                    && transfer["destination_input_index"] == 0
                    && transfer["destination_output"] == "is_critic"
                    && names.is_some_and(|n| {
                        n.len() == 6 && n.iter().all(|s| !s.is_empty() && s.len() <= 256)
                    })
                    && conditions.len() == 6
                    && transfer["provenance"]
                        == "ort-1.22-disable-source-io-addcopy-origin-single-recursive-cuda-transfer",
                "CUDA generated transfer differs from pinned scalar source evidence",
            )?;
        }
    }
    let public = &witness["public"];
    let pc = &witness["shared_pc"];
    for kernels in [public, pc] {
        require(
            (1..=20_000).contains(&count(kernels, "cuda_kernels")?)
                && count(kernels, "cuda_transfer_kernels")? <= count(kernels, "cuda_kernels")?
                && count(kernels, "approved_cpu_control_kernels")? <= 20_000,
            "CUDA selected-kernel counts are absent or unbounded",
        )?;
    }
    let public_neural = count(public, "neural_kernels")?;
    let proposer = count(pc, "proposer_private_kernels")?;
    let critic = count(pc, "critic_private_kernels")?;
    let pc_cuda = count(pc, "cuda_kernels")?;
    let pc_neural = count(pc, "neural_kernels")?;
    require(
        public_neural > 0
            && count(public, "cuda_transfer_kernels")? == 0
            && public_neural <= count(public, "cuda_kernels")?
            && proposer > 0
            && critic > 0
            && pc_neural > 0
            && (count(pc, "cuda_transfer_kernels")? == 0
                || graphs.iter().any(|g| {
                    g["role"] == "shared_pc"
                        && g["approved_transfers"]
                            .as_array()
                            .is_some_and(|v| v.len() == 1)
                }))
            && pc_neural
                .checked_add(count(pc, "cuda_transfer_kernels")?)
                .is_some_and(|n| n <= pc_cuda)
            && proposer.checked_add(critic).is_some_and(|n| n <= pc_neural),
        "CUDA witness must exercise public NN and both selected private P/C branches",
    )?;
    Ok(Some(PalsCudaPlacementAuditV3 {
        inventory_sha256: control.inventory.sha256.clone(),
        public_profile_sha256: array_hash(&public["profile_sha256"])?,
        shared_pc_profile_sha256: array_hash(&pc["profile_sha256"])?,
        public_neural_kernels: public_neural,
        proposer_private_kernels: proposer,
        critic_private_kernels: critic,
        device_id: cuda.device_id,
    }))
}
#[cfg(all(
    target_os = "linux",
    any(feature = "pals-collection-onnx", feature = "native-cuda")
))]
fn load_owner_helper_profile<L: PalsArenaValidationContext>(
    lock: &L,
    role: NativeEngineRole,
    inputs: &crate::native_runner::NativeProviderInputContext<'_>,
) -> Result<rz_uci::pals_checker_profile::LoadedPalsCheckerProfile, ArenaError> {
    use cap_fs_ext::{DirExt, FollowSymlinks, OpenOptionsFollowExt};
    use std::os::unix::fs::MetadataExt;
    let (declaration, _) = lock
        .external_binding(role)?
        .ok_or_else(|| invalid("own CPU_R has no external profile pin"))?;
    let mut matches = inputs
        .pins
        .iter()
        .filter(|pin| pin.artifact == &declaration.profile);
    let pin = matches
        .next()
        .ok_or_else(|| invalid("owner profile snapshot pin missing"))?;
    require(
        matches.next().is_none(),
        "owner profile snapshot pin duplicated",
    )?;
    let relative = lock
        .snapshot_relative_path(&declaration.profile)
        .ok_or_else(|| invalid("owner profile snapshot topology missing"))?;
    let relative_path = Path::new(&relative);
    let namespace = relative_path
        .parent()
        .and_then(Path::to_str)
        .ok_or_else(|| invalid("profile snapshot namespace missing"))?;
    let name = relative_path
        .file_name()
        .ok_or_else(|| invalid("profile snapshot filename missing"))?;
    require(
        relative_path.components().count() == 2
            && relative_path
                .components()
                .all(|part| matches!(part, Component::Normal(_)))
            && pin.path.ends_with(relative_path),
        "owner profile pin topology differs",
    )?;
    let directory = inputs
        .directory
        .open_dir_nofollow(namespace)
        .map_err(|_| invalid("owner profile namespace unavailable"))?;
    let identity = |metadata: &std::fs::Metadata| {
        (
            metadata.dev(),
            metadata.ino(),
            metadata.uid(),
            metadata.mode(),
            metadata.len(),
            metadata.mtime(),
            metadata.mtime_nsec(),
            metadata.ctime(),
            metadata.ctime_nsec(),
        )
    };
    let named = || -> Result<std::fs::Metadata, ArenaError> {
        let mut options = cap_std::fs::OpenOptions::new();
        options.read(true).follow(FollowSymlinks::No);
        directory
            .open_with(name, &options)
            .map_err(|_| invalid("owner profile name unavailable"))?
            .into_std()
            .metadata()
            .map_err(|_| invalid("owner profile name metadata missing"))
    };
    let before = pin
        .file
        .metadata()
        .map_err(|_| invalid("owner profile pin metadata missing"))?;
    require(
        before.is_file()
            && before.len() == declaration.profile.bytes
            && before.len() <= 64 * 1024
            && before.mode() & 0o222 == 0
            && before.nlink() == 1
            && identity(&before) == identity(&named()?),
        "owner profile snapshot is not its sealed read-only inode",
    )?;
    let loaded = rz_uci::pals_checker_profile::PalsCheckerProfile::load(
        pin.path,
        &declaration.profile.sha256,
    )
    .map_err(|e| invalid(format!("owner profile load failed: {e}")))?;
    let after = pin
        .file
        .metadata()
        .map_err(|_| invalid("owner profile pin recheck missing"))?;
    require(
        identity(&before) == identity(&after) && identity(&after) == identity(&named()?),
        "owner profile snapshot changed during actual byte loading",
    )?;
    Ok(loaded)
}

#[cfg(target_os = "linux")]
struct PalsPreflightHelperEvidence {
    sessions: Vec<PalsNativeSessionAuditV3>,
    artifacts: Vec<(String, Vec<u8>)>,
}

#[cfg(target_os = "linux")]
fn audit_pals_helper_preflight<L: PalsArenaValidationContext>(
    lock: &L,
    role: NativeEngineRole,
    preflight: &crate::ExternalUciPreflight,
    root: &cap_std::fs::Dir,
    inputs: &crate::native_runner::NativeProviderInputContext<'_>,
) -> Result<PalsPreflightHelperEvidence, ArenaError> {
    if lock.external_binding(role)?.is_none() {
        return Ok(PalsPreflightHelperEvidence {
            sessions: vec![],
            artifacts: vec![],
        });
    }
    #[cfg(not(any(feature = "pals-collection-onnx", feature = "native-cuda")))]
    {
        let _ = (preflight, root, inputs);
        Err(invalid(
            "unsupported external CPU_R preflight: native PALS feature is required; no fallback",
        ))
    }
    #[cfg(any(feature = "pals-collection-onnx", feature = "native-cuda"))]
    {
        use cap_fs_ext::DirExt;
        let loaded = load_owner_helper_profile(lock, role, inputs)?;
        require(
            preflight.engine_id == lock.endpoint_view(role).id,
            "helper preflight endpoint differs",
        )?;
        let probes = [
            (
                &preflight.identification_process,
                PalsExternalCpuRSessionPurposeV3::Identification,
            ),
            (
                &preflight.readiness_process,
                PalsExternalCpuRSessionPurposeV3::Readiness,
            ),
        ];
        let mut expected = BTreeSet::new();
        for (process, _) in probes {
            require(
                process.pid > 0
                    && expected.insert(format!("native-process-{}", process.pid))
                    && process.stop == crate::ProcessStop::Exited
                    && process.exit_code == Some(0)
                    && process.exit_signal.is_none()
                    && process.group_cleanup == crate::CleanupStatus::Gone
                    && process.errors.is_empty(),
                "helper preflight requires two independently supervised normal exits",
            )?;
        }
        let mut actual = BTreeSet::new();
        for (index, entry) in root
            .entries()
            .map_err(|e| ArenaError::Io(e.to_string()))?
            .enumerate()
        {
            require(
                index < lock.view().budget.max_runtime_files as usize,
                "helper preflight runtime entry cap exceeded",
            )?;
            let entry = entry.map_err(|e| ArenaError::Io(e.to_string()))?;
            let name = entry.file_name();
            let name = name
                .to_str()
                .ok_or_else(|| invalid("helper preflight name is not UTF-8"))?;
            if name == "runtime-cache" {
                continue;
            }
            if name == "cold-archive-managed" && lock.archive_runtime_root_selected(role) {
                root.open_dir_nofollow(name).map_err(|_| {
                    invalid("selected archive root is not a real no-link directory")
                })?;
                continue;
            }
            require(
                expected.contains(name) && actual.insert(name.to_owned()),
                "unexpected or duplicate helper preflight session slot",
            )?;
        }
        require(actual == expected, "helper preflight session slot missing")?;
        let mut result = PalsPreflightHelperEvidence {
            sessions: vec![],
            artifacts: vec![],
        };
        for (process, purpose) in probes {
            let name = format!("native-process-{}", process.pid);
            let directory = root
                .open_dir_nofollow(&name)
                .map_err(|_| invalid("helper preflight session is not a real directory"))?;
            let filenames = directory
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
                filenames
                    == BTreeSet::from([
                        OsString::from(lock.native_wire_filenames().0),
                        OsString::from(lock.native_wire_filenames().1),
                    ]),
                "helper preflight session requires exactly its startup/termination evidence",
            )?;
            let start = read_work_file(&directory, lock.native_wire_filenames().0)?;
            let end = read_work_file(&directory, lock.native_wire_filenames().1)?;
            let helper = validate_pals_external_cpu_r_session_context(
                lock,
                role,
                &loaded,
                &start,
                &end,
                process.pid,
                purpose,
            )?;
            let (audit, _) = validate_pals_native_records_inner(
                lock,
                role,
                &start,
                &end,
                &name,
                Some(helper),
                false,
            )?;
            result.sessions.push(audit);
            result
                .artifacts
                .push((format!("{name}/{}", lock.native_wire_filenames().0), start));
            result
                .artifacts
                .push((format!("{name}/{}", lock.native_wire_filenames().1), end));
        }
        let leaders = probes.iter().map(|(process, _)| process.pid).collect();
        validate_pals_helper_identity_set(&result.sessions, &leaders)?;
        Ok(result)
    }
}

// Historical identities are compared as observations, not as currently live
// PIDs or a proof of an escaped descendant/security sandbox boundary.
#[cfg(target_os = "linux")]
fn validate_pals_helper_identity_set(
    sessions: &[PalsNativeSessionAuditV3],
    leaders: &BTreeSet<u32>,
) -> Result<(), ArenaError> {
    let (mut pids, mut groups, mut tuples) = (BTreeSet::new(), BTreeSet::new(), BTreeSet::new());
    for session in sessions {
        if let Some(helper) = &session.external_cpu_r_session {
            let identity = helper
                .receipt
                .shutdown
                .process_identity
                .as_ref()
                .ok_or_else(|| invalid("validated helper historical process identity missing"))?;
            require(
                helper.parent_process_id == session.process_id
                    && leaders.contains(&session.process_id)
                    && !leaders.contains(&identity.pid)
                    && !leaders.contains(&identity.process_group)
                    && pids.insert(identity.pid)
                    && groups.insert(identity.process_group)
                    && tuples.insert((
                        identity.pid,
                        identity.process_group,
                        identity.proc_start_ticks,
                    )),
                "helper historical identity reused or collides with supervised UCI leaders",
            )?;
        }
    }
    Ok(())
}

impl NativeProviderDeclaration for LockedPalsArenaLaunchV3 {
    type Audit = PalsNativeSessionAuditV3;
    fn scope(&self) -> &'static str {
        "pals_v3_pc_native_or_explicit_mock_cpu_uci_rules_clocks_process_pilot_no_elo"
    }
    fn receipt_filename(&self) -> &'static str {
        "pals-arena-pair-receipt.v3.json"
    }
    fn receipt_version(&self) -> u32 {
        3
    }
    fn contract_revision(&self) -> &str {
        &self.input.semantic_lock.manifest.contract_revision
    }
    #[cfg(target_os = "linux")]
    fn validate_runtime_admission(&self) -> Result<(), ArenaError> {
        self.input.require_reap_status_runner()?;
        self.input.require_actual_native_epoch()?;
        verify_pals_inherited_resources(self).map(|_| ())
    }
    #[cfg(target_os = "linux")]
    fn validate_external_preflight(
        &self,
        role: NativeEngineRole,
        preflight: &crate::ExternalUciPreflight,
        runtime_directory: &cap_std::fs::Dir,
        inputs: &crate::native_runner::NativeProviderInputContext<'_>,
    ) -> Result<Vec<(String, Vec<u8>)>, ArenaError> {
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
            let mapping = validate_pals_game_process_trace(
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
            let helper = validate_pals_external_cpu_r_session(
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
        let mut has_external = false;
        for index in 0..2 {
            has_external |= self.input.external_cpu_r_binding(index)?.is_some();
        }
        if !has_external {
            return Ok(());
        }
        use cap_fs_ext::DirExt;
        let mapping = validate_pals_game_process_trace(
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
        let own = collect_pals_process_work(self, role, root_output)?;
        let selected: Vec<u32> = if own.is_empty() {
            let other = if role_index(role) == 0 {
                NativeEngineRole::Candidate
            } else {
                NativeEngineRole::Baseline
            };
            let other_records = if self.uses_provider_records(other)? {
                vec![]
            } else {
                collect_pals_process_work(self, other, root_output)?
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
        "pals-native-startup.v3.json"
    }
    fn termination_filename(&self) -> &'static str {
        "pals-native-termination.v3.json"
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
        validate_pals_native_records(self, role, startup, termination, session)
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
fn array_hash(v: &serde_json::Value) -> Result<String, ArenaError> {
    let bytes: [u8; 32] = serde_json::from_value(v.clone())
        .map_err(|_| invalid("native digest is not exactly 32 bytes"))?;
    Ok(bytes.iter().map(|b| format!("{b:02x}")).collect())
}
fn count(v: &serde_json::Value, k: &str) -> Result<u64, ArenaError> {
    v[k].as_u64()
        .ok_or_else(|| invalid(format!("missing native counter {k}")))
}
/// All graph executions, including public memory, have explicit completion and
/// validation counters. The caller keeps startup and search snapshots distinct.
fn graph_stats(v: &serde_json::Value) -> Result<(u64, u64, u64), ArenaError> {
    let public = count(v, "public_nn_runs_completed")?;
    let role = count(v, "role_nn_runs_completed")?;
    let inputs = count(v, "completed_nn_inputs")?;
    let calls = public
        .checked_add(role)
        .ok_or_else(|| invalid("NN graph counters overflow"))?;
    require(
        inputs == calls
            && count(v, "public_nn_runs_failed_known")? == 0
            && count(v, "role_nn_runs_failed_known")? == 0
            && public <= count(v, "public_nn_runs_attempted")?
            && role <= count(v, "role_nn_runs_attempted")?
            && count(v, "validated_public_outputs")? <= public
            && count(v, "validated_role_outputs")? <= role
            && count(v, "live_public_cache_entries")? <= 1,
        "native graph completion/failure/validation counters inconsistent",
    )?;
    Ok((inputs, calls, role))
}
// This launch revision registers Fresh whole-input execution. Optional Warm
// producer fields remain readable only as absent/null; observations or an
// unavailable marker cannot authorize an undeclared private execution mode.
fn require_no_undeclared_private_warm(record: &serde_json::Value) -> Result<(), ArenaError> {
    require(
        record["execution"]["private_warm"].is_null()
            && record["private_warm_observation"].is_null()
            && record["private_warm_observation_unavailable"].is_null(),
        "unsupported undeclared private warm execution/observation: the arena lock registers Fresh whole-input execution only",
    )
}
fn require_no_cuda_record_pages(record: &serde_json::Value) -> Result<(), ArenaError> {
    let present =
        |v: &serde_json::Value, key: &str| v.as_object().is_some_and(|o| o.contains_key(key));
    require(
        !present(&record["execution"], "cuda_record_pages")
            && ![
                "cuda_record_page_observation",
                "cuda_record_page_observation_error",
                "cuda_record_page_observation_scope",
            ]
            .iter()
            .any(|key| present(record, key))
            && ![
                "cuda_record_page_observation",
                "cuda_record_page_observation_error",
            ]
            .iter()
            .any(|key| present(&record["startup_probe"], key)),
        "unregistered resident CUDA declaration/observation/marker is forbidden",
    )
}
fn exact_cuda_record_page_fields(v: &serde_json::Value, fields: &[&str]) -> Result<(), ArenaError> {
    let object = v
        .as_object()
        .ok_or_else(|| invalid("resident CUDA evidence must be an object"))?;
    require(
        object.len() == fields.len() && fields.iter().all(|key| object.contains_key(*key)),
        "resident CUDA evidence has missing/unknown fields",
    )
}
fn resident_encoding_sha256(native: &serde_json::Value) -> Result<String, ArenaError> {
    let semantics: [u8; 32] = serde_json::from_value(native["encoding_semantic_sha256"].clone())
        .map_err(|_| invalid("resident encoding semantic digest is not exact bytes"))?;
    let mut hash = Sha256::new();
    hash.update(rz_eval::pals_model::PALS_ENCODING_SCHEMA);
    hash.update(semantics);
    Ok(format!("{:x}", hash.finalize()))
}
fn validate_cuda_record_page_observation(
    v: &serde_json::Value,
    selection: &serde_json::Value,
    pages: &PalsCudaRecordPagesBindingV1,
    backend_stats: &serde_json::Value,
) -> Result<(), ArenaError> {
    exact_cuda_record_page_fields(
        v,
        &[
            "schema",
            "initialized_scope",
            "initialized_provider_count",
            "packing_native_sessions",
            "initialized_log_sha256",
            "initialized_log_digest_domain",
            "packing_graph_sha256",
            "process_epoch",
            "game_generation",
            "model_manifest_sha256",
            "model_epoch",
            "public_graph_sha256",
            "encoding_sha256",
            "runtime_sha256",
            "runtime_identity_scope",
            "device_id",
            "live_blocks",
            "certified_projections",
            "whole_owner_host_bytes",
            "whole_owner_device_bytes",
            "active_invocation",
            "physical_completion_unknown",
            "quarantined",
            "stats",
        ],
    )?;
    require(
        v["schema"] == "rz-pals-resident-cuda-record-pages-observation/1"
            && count(v, "game_generation")? >= count(selection, "installation_game_generation")?
            && v["active_invocation"] == false
            && v["physical_completion_unknown"] == false
            && v["quarantined"] == false,
        "resident CUDA observation is stale/active/unknown/quarantined",
    )?;
    for field in [
        "initialized_scope",
        "initialized_provider_count",
        "packing_native_sessions",
        "initialized_log_sha256",
        "initialized_log_digest_domain",
        "packing_graph_sha256",
        "process_epoch",
        "model_manifest_sha256",
        "model_epoch",
        "public_graph_sha256",
        "encoding_sha256",
        "runtime_sha256",
        "runtime_identity_scope",
        "device_id",
    ] {
        require(
            v[field] == selection[field],
            "resident CUDA observed identity changed",
        )?;
    }
    let r = &pages.resources;
    let whole = count(v, "whole_owner_host_bytes")?
        .checked_add(count(v, "whole_owner_device_bytes")?)
        .ok_or_else(|| invalid("resident whole-owner byte overflow"))?;
    require(
        count(v, "live_blocks")? <= r.limits.max_blocks as u64
            && count(v, "certified_projections")? <= r.limits.max_registry_entries as u64
            && whole <= r.limits.max_bank_whole_payload_bytes,
        "resident actual whole owners/index exceed their registered bounds",
    )?;
    let stats = &v["stats"];
    exact_cuda_record_page_fields(
        stats,
        &[
            "admitted_views",
            "public_subset_runs_attempted",
            "public_subset_runs_completed",
            "packing_runs_attempted",
            "packing_runs_completed",
            "private_joined_runs_attempted",
            "private_joined_runs_completed",
            "certified_board_slices",
            "certified_record_slices",
            "published_blocks",
            "last_unique_whole_pins",
            "last_reserved_host_bytes",
            "last_reserved_device_bytes",
        ],
    )?;
    let public = count(stats, "public_subset_runs_completed")?;
    let packing = count(stats, "packing_runs_completed")?;
    let private = count(stats, "private_joined_runs_completed")?;
    require(
        public == count(stats, "public_subset_runs_attempted")?
            && packing == count(stats, "packing_runs_attempted")?
            && private == count(stats, "private_joined_runs_attempted")?
            && packing == private
            && private == count(stats, "admitted_views")?
            && public == count(backend_stats, "public_nn_runs_completed")?
            && private == count(backend_stats, "role_nn_runs_completed")?
            && count(stats, "certified_board_slices")? <= public
            && count(stats, "published_blocks")? <= public
            && count(stats, "certified_record_slices")?
                <= public
                    .checked_mul(128)
                    .ok_or_else(|| invalid("resident slice counter overflow"))?
            && count(stats, "last_unique_whole_pins")? <= r.limits.max_blocks as u64
            && count(stats, "last_reserved_host_bytes")? <= r.limits.max_invocation_host_bytes
            && count(stats, "last_reserved_device_bytes")? <= r.limits.max_invocation_device_bytes,
        "resident subset/packing/private/owner counters or combined reservations differ",
    )
}
fn validate_cuda_record_pages_pair(
    cuda: Option<&PalsOnnxCudaLaunchV3>,
    startup: &serde_json::Value,
    termination: &serde_json::Value,
) -> Result<Option<PalsCudaRecordPagesAuditV1>, ArenaError> {
    let Some((cuda, pages)) =
        cuda.and_then(|cuda| cuda.cuda_record_pages.as_ref().map(|p| (cuda, p)))
    else {
        require_no_cuda_record_pages(startup)?;
        require_no_cuda_record_pages(termination)?;
        return Ok(None);
    };
    pages.validate(cuda)?;
    let selection = &startup["execution"]["cuda_record_pages"];
    exact_cuda_record_page_fields(
        selection,
        &[
            "schema",
            "mode",
            "process_epoch",
            "installation_game_generation",
            "frozen_epoch",
            "model_manifest_sha256",
            "model_epoch",
            "public_graph_sha256",
            "encoding_sha256",
            "runtime_sha256",
            "runtime_identity_scope",
            "device_id",
            "packing_graph_sha256",
            "packing_graph_bytes",
            "packing_manifest_sha256",
            "packing_manifest_bytes",
            "known_retained_artifact_host_bytes",
            "implementation_sha256",
            "initialized_scope",
            "initialized_provider_count",
            "initialized_log_sha256",
            "initialized_log_digest_domain",
            "model_native_sessions",
            "packing_native_sessions",
            "resource_accounting_scope",
            "declared_resources",
        ],
    )?;
    let public = cuda
        .model
        .graphs
        .iter()
        .find(|g| g.role == "public")
        .ok_or_else(|| invalid("resident public graph missing"))?;
    let retained = count(selection, "known_retained_artifact_host_bytes")?;
    let raw = pages
        .manifest
        .bytes
        .checked_add(pages.graph.bytes)
        .ok_or_else(|| invalid("resident artifact byte overflow"))?;
    require(
        selection == &termination["execution"]["cuda_record_pages"]
            && selection["schema"] == "rz-pals-native-cuda-record-pages-selection/1"
            && selection["mode"] == "registered-packing-v1"
            && array_hash(&selection["implementation_sha256"])? == pages.implementation_sha256
            && count(selection, "process_epoch")? == count(startup, "process_epoch")?
            && count(selection, "process_epoch")? == count(termination, "process_epoch")?
            && selection["frozen_epoch"] == 1
            && startup["frozen_epoch"] == 1
            && termination["frozen_epoch"] == 1
            && selection["model_manifest_sha256"] == startup["export_manifest_sha256"]
            && selection["model_epoch"] == startup["model_epoch"]
            && array_hash(&selection["public_graph_sha256"])? == public.artifact.sha256
            && array_hash(&selection["encoding_sha256"])? == resident_encoding_sha256(startup)?
            && array_hash(&selection["runtime_sha256"])? == cuda.cuda_bundle.canonical_sha256
            && selection["runtime_identity_scope"] == "closed_cuda_library_bundle"
            && selection["device_id"] == cuda.device_id
            && array_hash(&selection["packing_graph_sha256"])? == pages.graph.sha256
            && selection["packing_graph_bytes"] == pages.graph.bytes
            && array_hash(&selection["packing_manifest_sha256"])? == pages.manifest.sha256
            && selection["packing_manifest_bytes"] == pages.manifest.bytes
            && retained >= raw
            && retained
                <= pages
                    .resources
                    .invocation
                    .additional_owner_metadata_payload
                    .host
            && selection["initialized_scope"] == "initialized_provider_count"
            && selection["initialized_provider_count"] == 275
            && array_hash(&selection["initialized_log_sha256"])? != "0".repeat(64)
            && selection["initialized_log_digest_domain"]
                == "rz-pals-fixed-packing-initialized-provider-count/1"
            && selection["model_native_sessions"] == 2
            && selection["packing_native_sessions"] == 1
            && startup["residency"]["native_sessions"] == 2
            && selection["resource_accounting_scope"]
                == "combined_resident_owner_invocation_including_sessions_and_backing"
            && selection["declared_resources"] == pages.declared_resources(),
        "resident CUDA registered selection/actual installed identity or resources differ",
    )?;
    // Only the startup probe owns the post-probe/reset ACK. The final ACK is
    // separate; an initial installed DTO is never accepted in its place.
    let absent =
        |v: &serde_json::Value, key: &str| !v.as_object().is_some_and(|o| o.contains_key(key));
    require(
        absent(startup, "cuda_record_page_observation")
            && absent(startup, "cuda_record_page_observation_error")
            && absent(startup, "cuda_record_page_observation_scope")
            && absent(
                &startup["startup_probe"],
                "cuda_record_page_observation_error",
            )
            && absent(termination, "cuda_record_page_observation_error")
            && termination["cuda_record_page_observation_scope"]
                == "exclusive_worker_before_shutdown",
        "resident snapshot ACK provenance is missing/failed/duplicated",
    )?;
    let first = &startup["startup_probe"]["cuda_record_page_observation"];
    let last = &termination["cuda_record_page_observation"];
    validate_cuda_record_page_observation(
        first,
        selection,
        pages,
        &startup["startup_probe"]["backend_stats"],
    )?;
    validate_cuda_record_page_observation(last, selection, pages, &termination["backend_stats"])?;
    require(
        count(first, "game_generation")? == count(startup, "game_generation")?
            && count(last, "game_generation")? == count(termination, "game_generation")?
            && count(last, "game_generation")? >= count(first, "game_generation")?,
        "resident final snapshot generation differs from the actual native owner",
    )?;
    for key in [
        "admitted_views",
        "public_subset_runs_attempted",
        "public_subset_runs_completed",
        "packing_runs_attempted",
        "packing_runs_completed",
        "private_joined_runs_attempted",
        "private_joined_runs_completed",
        "certified_board_slices",
        "certified_record_slices",
        "published_blocks",
    ] {
        require(
            count(&last["stats"], key)? >= count(&first["stats"], key)?,
            "resident cumulative counter regressed",
        )?;
    }
    Ok(Some(PalsCudaRecordPagesAuditV1 {
        selection: selection.clone(),
        startup_observation: first.clone(),
        termination_observation: last.clone(),
        termination_observation_scope: "exclusive_worker_before_shutdown",
    }))
}

/// Pure wire validation. No self-reported field replaces process exit or PGN.
pub fn validate_pals_native_records(
    lock: &LockedPalsArenaLaunchV3,
    role: NativeEngineRole,
    startup: &[u8],
    termination: &[u8],
    session: &str,
) -> Result<(PalsNativeSessionAuditV3, u32), ArenaError> {
    validate_pals_native_records_inner(lock, role, startup, termination, session, None, true)
}

/// Shared physical admission context. Implementations keep their own semantic
/// lock and native wire domain; no historical manifest is synthesized.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum NativeReceiptEncodingIdentity<'a> {
    // Historical V3 receipts retain their declared recipe domain.
    LegacyRecipe(&'a str),
    // V4 loader metadata still names Rules; the physical owner names Fresh.
    FreshModelInput(&'a str),
}
impl<'a> NativeReceiptEncodingIdentity<'a> {
    fn sha256(self) -> &'a str {
        match self {
            Self::LegacyRecipe(value) | Self::FreshModelInput(value) => value,
        }
    }
}
trait PalsArenaValidationContext: NativeLaunchDeclaration {
    #[cfg(target_os = "linux")]
    fn verify_semantic(&self) -> Result<(), ArenaError>;
    fn launch_sha256(&self) -> &str;
    fn native(&self, role: NativeEngineRole) -> Result<&PalsOnnxLaunchV3, ArenaError>;
    fn native_receipt_encoding_identity(
        &self,
        role: NativeEngineRole,
    ) -> Result<NativeReceiptEncodingIdentity<'_>, ArenaError> {
        Ok(NativeReceiptEncodingIdentity::LegacyRecipe(
            &self.native(role)?.encoding_semantic_sha256,
        ))
    }
    fn recipe(&self, role: NativeEngineRole) -> &PalsEndpointLaunchV3;
    fn endpoint_view(&self, role: NativeEngineRole) -> &ExternalUciEndpointV2;
    fn pals_endpoint(
        &self,
        role: NativeEngineRole,
    ) -> Result<&rz_experiments::PalsEndpointV3, ArenaError>;
    fn resource(&self, role: NativeEngineRole) -> &rz_experiments::PalsResourcePolicyV3;
    fn external_binding(
        &self,
        role: NativeEngineRole,
    ) -> Result<Option<(&PalsExternalCpuRV3, &PalsExternalCpuRLaunchV3)>, ArenaError>;
    fn native_wire_version(&self) -> u32;
    fn native_wire_domains(&self) -> (&'static str, &'static str);
    fn native_receipt_byte_cap(&self, _role: NativeEngineRole) -> usize {
        128 * 1024
    }
    fn device_public_memory_selected(&self, _role: NativeEngineRole) -> bool {
        false
    }
    fn shared_role_layout(&self, _role: NativeEngineRole) -> &str {
        "shared_pc_if"
    }
    #[cfg(all(
        target_os = "linux",
        any(feature = "pals-collection-onnx", feature = "native-cuda")
    ))]
    fn archive_runtime_root_selected(&self, _role: NativeEngineRole) -> bool {
        false
    }
    fn validate_selected_private_execution(
        &self,
        _role: NativeEngineRole,
        record: &serde_json::Value,
    ) -> Result<(), ArenaError> {
        require_no_undeclared_private_warm(record)
    }
    #[cfg(target_os = "linux")]
    fn native_wire_filenames(&self) -> (&'static str, &'static str);
    fn validate_policy_pair(
        &self,
        role: NativeEngineRole,
        startup: &serde_json::Value,
        termination: &serde_json::Value,
    ) -> Result<(), ArenaError>;
    fn validate_wire_pair(
        &self,
        role: NativeEngineRole,
        startup: &serde_json::Value,
        termination: &serde_json::Value,
    ) -> Result<(), ArenaError> {
        self.validate_policy_pair(role, &startup["search_work"], &termination["search_work"])
    }
}
impl PalsArenaValidationContext for LockedPalsArenaLaunchV3 {
    #[cfg(target_os = "linux")]
    fn verify_semantic(&self) -> Result<(), ArenaError> {
        self.input.semantic_lock.verify().map_err(ArenaError::from)
    }
    fn launch_sha256(&self) -> &str {
        &self.sha256
    }
    fn native(&self, role: NativeEngineRole) -> Result<&PalsOnnxLaunchV3, ArenaError> {
        LockedPalsArenaLaunchV3::native(self, role)
    }
    fn recipe(&self, role: NativeEngineRole) -> &PalsEndpointLaunchV3 {
        &self.input.endpoints[role_index(role)]
    }
    fn endpoint_view(&self, role: NativeEngineRole) -> &ExternalUciEndpointV2 {
        &self.endpoint_views[role_index(role)]
    }
    fn pals_endpoint(
        &self,
        role: NativeEngineRole,
    ) -> Result<&rz_experiments::PalsEndpointV3, ArenaError> {
        match &self.input.semantic_lock.manifest.engines[role_index(role)] {
            PalsEngineV3::Pals(e) => Ok(e),
            _ => Err(invalid("missing native model identity")),
        }
    }
    fn resource(&self, role: NativeEngineRole) -> &rz_experiments::PalsResourcePolicyV3 {
        &self.input.semantic_lock.manifest.resources[role_index(role)]
    }
    fn external_binding(
        &self,
        role: NativeEngineRole,
    ) -> Result<Option<(&PalsExternalCpuRV3, &PalsExternalCpuRLaunchV3)>, ArenaError> {
        self.input.external_cpu_r_binding(role_index(role))
    }
    fn native_wire_version(&self) -> u32 {
        3
    }
    #[cfg(target_os = "linux")]
    fn native_wire_filenames(&self) -> (&'static str, &'static str) {
        (
            "pals-native-startup.v3.json",
            "pals-native-termination.v3.json",
        )
    }
    fn native_wire_domains(&self) -> (&'static str, &'static str) {
        (
            PALS_NATIVE_STARTUP_V3_DOMAIN,
            PALS_NATIVE_TERMINATION_V3_DOMAIN,
        )
    }
    fn validate_policy_pair(
        &self,
        role: NativeEngineRole,
        startup: &serde_json::Value,
        termination: &serde_json::Value,
    ) -> Result<(), ArenaError> {
        validate_search_policy_pair(self, role, startup, termination)
    }
}

fn validate_native_model_identity(
    native: &PalsOnnxLaunchV3,
    checkpoint: &ArtifactRef,
    trained: bool,
    encoding: NativeReceiptEncodingIdentity<'_>,
    record: &serde_json::Value,
) -> Result<(), ArenaError> {
    require(
        array_hash(&record["model_epoch"])? == checkpoint.sha256
            && array_hash(&record["export_manifest_sha256"])? == native.export.sha256
            && array_hash(&record["encoding_semantic_sha256"])? == encoding.sha256()
            && array_hash(&record["adapter_source_sha256"])? == native.adapter_source_sha256
            && record["trained"] == trained,
        "native model/adapter/epoch identity differs",
    )
}

fn validate_pals_native_records_inner<L: PalsArenaValidationContext>(
    lock: &L,
    role: NativeEngineRole,
    startup: &[u8],
    termination: &[u8],
    session: &str,
    helper: Option<PalsExternalCpuRSessionAuditV3>,
    game: bool,
) -> Result<(PalsNativeSessionAuditV3, u32), ArenaError> {
    require(
        startup.len() <= lock.native_receipt_byte_cap(role)
            && termination.len() <= lock.native_receipt_byte_cap(role),
        "native receipt budget exceeded",
    )?;
    let s = json(startup)?;
    let t = json(termination)?;
    let native = lock.native(role)?;
    lock.validate_wire_pair(role, &s, &t)?;
    let cuda = lock.recipe(role).cuda_model();
    let engine = lock.endpoint_view(role);
    require(
        s["schema_version"] == lock.native_wire_version()
            && t["schema_version"] == lock.native_wire_version()
            && s["domain"] == lock.native_wire_domains().0
            && t["domain"] == lock.native_wire_domains().1,
        "native wire revision/domain differs",
    )?;
    for field in [
        "endpoint_id",
        "launch_sha256",
        "process_id",
        "binary_sha256",
        "runtime_sha256",
        "runtime_bundle_sha256",
        "provider",
        "precision",
    ] {
        require(
            s[field] == t[field],
            "native startup/termination identity differs",
        )?;
    }
    require(
        s["endpoint_id"] == engine.id
            && s["launch_sha256"] == lock.launch_sha256()
            && s["binary_sha256"] == engine.binary.sha256
            && s["runtime_sha256"] == native.runtime.sha256
            && s["provider"] == if cuda.is_some() { "cuda" } else { "cpu" }
            && s["precision"] == "fp32"
            && s["service_exit_success"] == false
            && t["service_exit_success"] == true,
        "native executable/runtime/provider identity differs",
    )?;
    let pid = count(&s, "process_id")?;
    require(
        pid > 0 && pid <= i32::MAX as u64 && session == format!("native-process-{pid}"),
        "native PID/session mismatch",
    )?;
    let sn = &s["native"];
    let tn = &t["native"];
    // Host pages and private Warm remain unregistered. The separately explicit
    // resident CUDA binding cannot authorize either unrelated optional mode.
    for record in [sn, tn] {
        require(
            record["execution"]["host_record_pages"].is_null()
                && record["host_record_page_observation"].is_null(),
            "unsupported undeclared host record-page execution/observation: the arena lock registers the whole-input public graph only",
        )?;
        lock.validate_selected_private_execution(role, record)?;
    }
    let cuda_record_pages = validate_cuda_record_pages_pair(cuda, sn, tn)?;
    if lock.external_binding(role)?.is_some() {
        let audit = helper.as_ref().ok_or_else(|| invalid(
            "unsupported external CPU_R native receipt: owner profile and independent supervisor context required"))?;
        require(
            audit.endpoint_id == engine.id
                && u64::from(audit.parent_process_id) == pid
                && audit.startup_sha256 == digest(startup)
                && audit.termination_sha256 == digest(termination)
                && (audit.purpose == PalsExternalCpuRSessionPurposeV3::Game) == game,
            "helper/native audit pair or purpose differs",
        )?;
    } else {
        require(
            helper.is_none() && s["cpu_checker"].is_null() && t["cpu_checker"].is_null(),
            "unsupported external CPU_R native receipt: own selection cannot inherit helper evidence",
        )?;
    }
    require(
        sn["final_runtime_loading_mapping"].is_null(),
        "startup cannot claim a later final loading observation",
    )?;
    // The native owner allocates a checked, domain-scoped process epoch; it is
    // neither an OS PID nor a fixed first-owner value. Its paired records are
    // already bound to the same executable/launch/PID and session directory.
    let process_epoch = count(sn, "process_epoch")?;
    require(
        process_epoch > 0 && process_epoch == count(tn, "process_epoch")?,
        "native process epoch must be nonzero and match the bound startup/termination pair",
    )?;
    let mut startup_nn = (0, 0, 0);
    let mut cuda_placement = None;
    let mut cuda_loading = None;
    let mut startup_probe_timeout_ms = None;
    if let Some(cuda) = cuda {
        require(
            s["runtime_bundle_sha256"] == cuda.cuda_bundle.canonical_sha256,
            "CUDA envelope canonical runtime bundle differs",
        )?;
        require(
            sn["execution"] == tn["execution"] && sn["startup_probe"] == tn["startup_probe"],
            "CUDA execution/probe identity changed",
        )?;
        let execution = &sn["execution"];
        startup_probe_timeout_ms = validate_native_startup_budget(cuda, execution)?;
        match &cuda.cuda_control {
            Some(control) => require(
                array_hash(&execution["cuda_control_inventory_sha256"])?
                    == control.inventory.sha256,
                "CUDA loaded control-policy identity differs",
            )?,
            None => require(
                execution["cuda_control_inventory_sha256"].is_null(),
                "strict CUDA cannot claim a control-policy identity",
            )?,
        }
        require(
            execution["provider"] == "cuda"
                && execution["device_id"] == cuda.device_id
                && execution["session_arena_bytes"] == cuda.session_arena_bytes
                && execution["device_public_memory"] == lock.device_public_memory_selected(role)
                && array_hash(&execution["runtime_sha256"])? == native.runtime.sha256
                && array_hash(&execution["runtime_bundle_sha256"])?
                    == cuda.cuda_bundle.canonical_sha256,
            "CUDA provider/device/session/runtime bundle identity differs",
        )?;
        if let Some(pages) = &cuda.cuda_record_pages {
            // This authoritative invocation bound already contains all three
            // sessions, backing/join/intermediate/input/metadata reservations.
            // Legacy transient+model arena arithmetic would count them twice.
            require(
                pages.resources.limits.max_invocation_device_bytes
                    <= lock.resource(role).device_allocation_max_bytes
                    && pages.resources.limits.max_invocation_host_bytes
                        <= lock.resource(role).memory_max_bytes
                    && count(execution, "pinned_request_bytes")? == 0,
                "resident combined owner reservation exceeds actual admitted pools",
            )?;
        } else {
            let transient = count(execution, "transient_request_device_bytes")?
                .checked_add(count(execution, "transient_execution_device_bytes")?)
                .ok_or_else(|| invalid("CUDA transient reservation overflow"))?;
            let arenas = cuda
                .session_arena_bytes
                .checked_mul(native.graphs.len() as u64)
                .ok_or_else(|| invalid("CUDA session reservation overflow"))?;
            require(
                transient > 0
                    && arenas
                        .checked_add(transient)
                        .is_some_and(|n| n <= lock.resource(role).device_allocation_max_bytes)
                    && count(execution, "pinned_request_bytes")? == 0,
                "CUDA declared budget does not cover actual tensor reservations plus session declarations",
            )?;
        }
        let probe = &sn["startup_probe"];
        cuda_loading = validate_native_loading_evidence(
            cuda,
            execution,
            probe,
            &tn["final_runtime_loading_mapping"],
        )?;
        cuda_placement = validate_cuda_placement_witness(cuda, &probe["cuda_placement_witness"])?;
        require(
            count(probe, "completed_proposer_calls")? == 1
                && count(probe, "completed_critic_calls")? == 1
                && probe["runtime_mapping_confirmed"] == true
                && probe["reset_completed"] == true,
            "CUDA startup requires actual P/C callbacks, full runtime audit and reset",
        )?;
        startup_nn = graph_stats(&probe["backend_stats"])?;
        require(
            startup_nn.0 > 0
                && startup_nn.2 == 2
                && count(&probe["backend_stats"], "new_game_resets")? >= 1
                && tn["final_runtime_mapping_confirmed"] == true,
            "CUDA startup NN work/reset unavailable or inconsistent",
        )?;
    } else {
        require(
            sn["startup_probe"].is_null()
                && tn["startup_probe"].is_null()
                && s["runtime_bundle_sha256"].is_null()
                && tn["final_runtime_loading_mapping"].is_null(),
            "CPU recipe cannot claim CUDA initialization work",
        )?;
        require(
            (sn["execution"].is_null() && tn["execution"].is_null())
                || (sn["execution"].is_object() && tn["execution"].is_object()),
            "CPU execution identity is malformed or present in only one record",
        )?;
        if sn["execution"].is_object() {
            require(
                sn["execution"] == tn["execution"]
                    && sn["execution"]["provider"] == "cpu"
                    && sn["execution"]["device_id"].is_null()
                    && sn["execution"]["session_arena_bytes"].is_null()
                    && sn["execution"]["runtime_bundle_sha256"].is_null()
                    && sn["execution"]["cuda_control_inventory_sha256"].is_null()
                    && sn["execution"]["cuda_loading_profile"].is_null()
                    && sn["execution"]["startup_probe_timeout_ms"].is_null()
                    && count(&sn["execution"], "transient_request_device_bytes")? == 0
                    && count(&sn["execution"], "transient_execution_device_bytes")? == 0
                    && count(&sn["execution"], "pinned_request_bytes")? == 0
                    && sn["execution"]["device_public_memory"] == false
                    && array_hash(&sn["execution"]["runtime_sha256"])? == native.runtime.sha256,
                "CPU native execution metadata differs",
            )?;
        }
    }
    let e = lock.pals_endpoint(role)?;
    let (checkpoint, trained) = match &e.model.weights {
        PalsWeightIdentityV3::Untrained { artifact, .. } => (artifact, false),
        PalsWeightIdentityV3::Trained { artifact, .. } => (artifact, true),
        _ => return Err(invalid("missing actual weights")),
    };
    validate_native_cudnn_conv_pair(cuda, lock.native_wire_version(), &checkpoint.sha256, sn, tn)?;
    for field in [
        "model_epoch",
        "export_manifest_sha256",
        "encoding_semantic_sha256",
        "adapter_source_sha256",
        "trained",
        "residency",
        "process_epoch",
        "frozen_epoch",
    ] {
        require(
            sn[field] == tn[field],
            "native frozen identity changed during game",
        )?;
    }
    validate_native_model_identity(
        native,
        checkpoint,
        trained,
        lock.native_receipt_encoding_identity(role)?,
        sn,
    )?;
    require(
        (e.model.frozen_epoch == 0 && sn["frozen_epoch"].is_null() && tn["frozen_epoch"].is_null())
            || sn["frozen_epoch"].as_u64() == Some(e.model.frozen_epoch),
        "native frozen deployment epoch differs from immutable semantic lock",
    )?;
    for field in [
        "physically_completed_role_calls",
        "completed_role_inputs",
        "failed_physical_role_calls",
        "invalid_role_outputs",
        "delivered_role_inputs",
        "search_consumed_role_inputs",
        "canceled_requests",
        "expired_requests",
        "completed_new_game_resets",
        "physical_runs_in_flight",
        "request_high_water",
        "execution_high_water",
    ] {
        require(
            count(sn, field)? == 0,
            "startup contains work from another game/run",
        )?;
    }
    require(
        sn["last_failure"].is_null(),
        "startup retained an earlier failure",
    )?;
    require(
        sn["quarantined"] == false
            && sn["physical_shutdown_confirmed"] == false
            && sn["native_buffers_released"] == false,
        "startup owner is not live and fresh",
    )?;
    let physical = count(tn, "physically_completed_role_calls")?;
    let completed = count(tn, "completed_role_inputs")?;
    let failed = count(tn, "failed_physical_role_calls")?;
    let delivered = count(tn, "delivered_role_inputs")?;
    let consumed = count(tn, "search_consumed_role_inputs")?;
    // Historical CPU producer records may omit this additive observation;
    // preserve their wire reader. New Core assembly requires the actual count.
    if !tn["observer_failures"].is_null() {
        require(
            count(tn, "observer_failures")? == 0 && tn["last_observer_failure"].is_null(),
            "native observer failed; physical counters are preserved without Core acceptance",
        )?;
    }
    require(
        physical
            == completed
                .checked_add(failed)
                .ok_or_else(|| invalid("native count overflow"))?
            && failed == 0
            && count(tn, "invalid_role_outputs")? == 0
            && consumed <= delivered
            && delivered <= completed
            && count(tn, "request_high_water")? >= physical,
        "native completion/delivery/consumption counters inconsistent or failed",
    )?;
    require(
        count(tn, "execution_high_water")? >= physical,
        "native physical execution high-water is below completed work",
    )?;
    require(
        (!game
            || (count(tn, "completed_new_game_resets")? >= 1
                && count(tn, "game_generation")? > count(sn, "game_generation")?))
            && count(tn, "physical_runs_in_flight")? == 0
            && tn["quarantined"] == false
            && tn["physical_shutdown_confirmed"] == true
            && tn["native_buffers_released"] == true,
        "native game reset or physical completion/join/release is unconfirmed",
    )?;
    let graphs = sn["residency"]["graphs"]
        .as_array()
        .ok_or_else(|| invalid("missing native residency graph evidence"))?;
    let shared = native.graphs.iter().any(|g| g.role == "shared_pc");
    require(
        sn["residency"] == tn["residency"]
            && sn["residency"]["native_sessions"] == native.graphs.len()
            && graphs.len() == native.graphs.len()
            && if shared {
                sn["residency"]["layout"] == lock.shared_role_layout(role)
                    && sn["residency"]["role_reader_weights_shared"].is_null()
            } else {
                sn["residency"]["role_reader_weights_shared"] == false
                    && (sn["residency"]["layout"].is_null()
                        || sn["residency"]["layout"] == "separate_pc")
            },
        "actual P/C graph layout/residency differs; optimized parameter sharing remains unknown",
    )?;
    for g in &native.graphs {
        require(
            graphs
                .iter()
                .filter(|observed| {
                    observed["role"] == g.role
                        && array_hash(&observed["sha256"]).is_ok_and(|h| h == g.artifact.sha256)
                        && observed["serialized_bytes"] == g.artifact.bytes
                })
                .count()
                == 1,
            "native graph hash/bytes differ",
        )?;
    }
    Ok((
        PalsNativeSessionAuditV3 {
            endpoint_id: engine.id.clone(),
            process_id: pid as u32,
            launch_sha256: lock.launch_sha256().to_owned(),
            startup_sha256: digest(startup),
            termination_sha256: digest(termination),
            external_cpu_r_session: helper,
            completed_role_inputs: completed,
            search_consumed_role_inputs: consumed,
            completed_new_game_resets: count(tn, "completed_new_game_resets")?,
            physical_shutdown_confirmed: true,
            native_buffers_released: true,
            startup_nn_inputs_completed: startup_nn.0,
            startup_nn_calls_completed: startup_nn.1,
            startup_role_inputs_completed: startup_nn.2,
            startup_probe_timeout_ms,
            startup_probe: sn["startup_probe"]
                .as_object()
                .map(|_| sn["startup_probe"].clone()),
            execution: sn["execution"].as_object().map(|_| sn["execution"].clone()),
            cuda_placement,
            cuda_loading,
            cuda_record_pages,
            raw_native: tn.clone(),
            raw_search_work: t["search_work"].clone(),
        },
        pid as u32,
    ))
}

/// Snapshot/pin preparation starts no engines and grants no NN-ready claim.
pub fn prepare_pals_pair_launch(
    spec: &LockedPalsArenaLaunchV3,
    source_root: &Path,
    output_root: &Path,
    label: &str,
) -> Result<NativeLaunchOwner<LockedPalsArenaLaunchV3>, Box<crate::NativePreparationFailure>> {
    if let Err(cause) = verify_pals_launch_exports(spec, source_root) {
        return Err(Box::new(crate::NativePreparationFailure {
            receipt:crate::NativePreparationReceipt {receipt_version:3,policy_envelope:None,execution_ready:false,input_sha256:spec.sha256.clone(),
                output_directory:"not-created".into(),attempt_created:false,attempted_snapshot_relative_paths:vec![],completed_snapshots:vec![],
                child_spawned:false,writers_closed:true,input_pins_required:false,original_error:cause.to_string(),
                subsequent_file_owner:"source assets retained; descriptor verification failed before snapshot creation or child launch".into(),automatic_retry:false},
            cause,receipt_artifact:None,persistence_error:None,
        }));
    }
    crate::native_launch::prepare_native_launch_for(spec, source_root, output_root, label)
}
/// Uses the existing bounded runner, clocks, Rules replay and cleanup receipts.
pub fn run_pals_pair(
    owner: NativeLaunchOwner<LockedPalsArenaLaunchV3>,
    cancel: Option<&std::sync::atomic::AtomicBool>,
) -> Result<
    crate::NativePairOutput<LockedPalsArenaLaunchV3>,
    Box<crate::NativePairFailure<LockedPalsArenaLaunchV3>>,
> {
    crate::native_runner::run_native_pair_for(owner, cancel)
}

/// Time scope starts before UCI preflight and ends after mandatory arena
/// postchecks and receipt persistence. Snapshot preparation remains separate.
pub struct PalsObservedPairOutputV3 {
    pub native: crate::NativePairOutput<LockedPalsArenaLaunchV3>,
    pub wall_time_ms: u64,
}
pub fn run_pals_pair_observed(
    owner: NativeLaunchOwner<LockedPalsArenaLaunchV3>,
    cancel: Option<&std::sync::atomic::AtomicBool>,
) -> Result<PalsObservedPairOutputV3, Box<crate::NativePairFailure<LockedPalsArenaLaunchV3>>> {
    let started = std::time::Instant::now();
    run_pals_pair(owner, cancel).map(|native| PalsObservedPairOutputV3 {
        native,
        wall_time_ms: u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX),
    })
}

/// Process-wide work is observed once per restarted game. CPU task reuse and
/// native NN execution have different units; this record preserves both wires.
#[derive(Clone, Debug, Serialize)]
pub struct PalsProcessWorkAuditV3 {
    pub endpoint_id: String,
    pub process_id: u32,
    pub startup_sha256: String,
    pub termination_sha256: String,
    pub startup_bytes: u64,
    pub termination_bytes: u64,
    pub search_work: serde_json::Value,
}

fn work_count(work: &serde_json::Value, field: &str) -> Result<u64, ArenaError> {
    work.get(field)
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| {
            invalid(format!(
                "work observation missing, unknown or invalid: {field}"
            ))
        })
}

/// Extract only a validated producer observation. An expected declaration is
/// used for comparison, never copied into the observed Core receipt.
fn validate_search_policy_marker(
    work: &serde_json::Value,
    expected: Option<&PalsSearchPolicyIdentityV3>,
) -> Result<Option<PalsSearchPolicyIdentityV3>, ArenaError> {
    match work.get("pals_search_policy") {
        None => {
            require(
                expected.is_none(),
                "selected post-Repair recheck producer marker is missing",
            )?;
            Ok(None)
        }
        Some(value) => {
            let observed: PalsSearchPolicyIdentityV3 = serde_json::from_value(value.clone())
                .map_err(|e| invalid(format!("search policy marker wire: {e}")))?;
            observed.validate().map_err(|e| invalid(e.to_string()))?;
            require(
                expected == Some(&observed),
                "search policy marker is undeclared, foreign, or differs from the selected manifest/recipe",
            )?;
            Ok(Some(observed))
        }
    }
}

fn validate_search_policy_pair(
    lock: &LockedPalsArenaLaunchV3,
    role: NativeEngineRole,
    startup_work: &serde_json::Value,
    termination_work: &serde_json::Value,
) -> Result<(), ArenaError> {
    let expected = validate_launch_search_policy(&lock.input, role_index(role))?;
    let startup = validate_search_policy_marker(startup_work, expected.as_ref())?;
    let termination = validate_search_policy_marker(termination_work, expected.as_ref())?;
    require(
        startup == termination,
        "startup/termination search policy drift",
    )
}

/// Validate the cumulative work snapshot independently of root-coverage gauges.
/// A failed go can retain observed work. Missing work never becomes zero.
pub fn validate_pals_process_work(
    work: &serde_json::Value,
    kind: &str,
    startup: bool,
) -> Result<(), ArenaError> {
    validate_pals_process_work_inner(work, kind, startup, false)
}
fn validate_pals_process_work_inner(
    work: &serde_json::Value,
    kind: &str,
    startup: bool,
    external: bool,
) -> Result<(), ArenaError> {
    validate_pals_process_counter_units(work, kind, startup, external, true)
}
fn validate_pals_process_counter_units(
    work: &serde_json::Value,
    kind: &str,
    startup: bool,
    external: bool,
    legacy_identity_checks: bool,
) -> Result<(), ArenaError> {
    require(
        work["schema_version"] == 1 && work["search_kind"] == kind,
        "work schema/search kind differs",
    )?;
    // This context-free counter check permits only a structurally exact Own
    // PALS selector. Record/launch audit separately requires its declaration
    // and both snapshots; the marker never advertises execution success.
    if legacy_identity_checks && work.get("pals_search_policy").is_some() {
        require(
            kind == "pals" && !external,
            "non-PALS or external work cannot inherit a post-Repair recheck marker",
        )?;
        let observed: PalsSearchPolicyIdentityV3 =
            serde_json::from_value(work["pals_search_policy"].clone())
                .map_err(|e| invalid(format!("search policy marker wire: {e}")))?;
        observed.validate().map_err(|e| invalid(e.to_string()))?;
    }
    // Historical v1 receipts omit this additive identity. Keep their absence
    // observable; an explicit identity must match the registered semantics.
    if legacy_identity_checks && external {
        require(
            kind == "pals"
                && work["pals_resolver"]["version"]
                    == rz_search::pals::value::MODEL_WDL_RESOLVER_VERSION
                && work["pals_resolver"]["semantics_sha256"]
                    == serde_json::json!(
                        Sha256::digest(
                            rz_search::pals::value::MODEL_WDL_RESOLVER_SEMANTICS.as_bytes()
                        )
                        .to_vec()
                    ),
            "external work model-WDL resolver missing or differs",
        )?;
        let values = &work["pals"];
        require(
            work_count(values, "completed_value_calls")? <= work_count(values, "value_calls")?
                && work_count(values, "accepted_value_outputs")?
                    <= work_count(values, "completed_value_calls")?,
            "external model-WDL completion/acceptance counters inconsistent",
        )?;
    } else if legacy_identity_checks && let Some(resolver) = work.get("pals_resolver") {
        require(
            kind == "pals"
                && resolver["version"] == rz_search::pals::engine::PALS_VALUE_RESOLVER_VERSION
                && resolver["semantics_sha256"]
                    == serde_json::json!(
                        Sha256::digest(
                            rz_search::pals::engine::PALS_VALUE_RESOLVER_SEMANTICS.as_bytes()
                        )
                        .to_vec()
                    ),
            "work resolver identity belongs to a different policy/search path",
        )?;
    }
    let go = work_count(work, "go_invocations")?;
    let success = work_count(work, "successful_returns")?;
    let failed = work_count(work, "failed_returns")?;
    let active = work_count(work, "active_invocations")?;
    require(
        active == 0
            && go
                == success
                    .checked_add(failed)
                    .ok_or_else(|| invalid("work return count overflow"))?,
        "work returns/active invocation inconsistent",
    )?;
    require(
        work_count(work, "unobserved_work_invocations")? == 0
            && work_count(work, "physical_unknown_returns")? == 0,
        "work or physical completion observation unknown",
    )?;
    for field in ["canceled_returns", "deadline_returns"] {
        require(
            work_count(work, field)? <= go,
            "work outcome counter exceeds invocations",
        )?;
    }
    let (totals, absent) = if kind == "pals" {
        ("pals", "cpu")
    } else if kind == "cpu" {
        ("cpu", "pals")
    } else {
        return Err(invalid("unsupported work kind"));
    };
    require(
        work[absent].is_null() && work[totals].is_object(),
        "work totals belong to a different search path",
    )?;
    // Only the Core's mandatory units require numeric observations. Partial
    // output ACKs and root gauges can be unknown while completed-task consumers
    // are observed; raw optional counters retain that distinction.
    let mandatory: &[&str] = if kind == "pals" {
        &[
            "completed_proposer_calls",
            "completed_repair_calls",
            "completed_critic_calls",
            "cpu_tasks_requested",
            "completed_cpu_tasks",
            "reused_completed_cpu_tasks_consumed",
            "consumed_cpu_tasks",
            "cpu_nodes",
            "consumed_role_outputs",
        ]
    } else {
        &[
            "tasks_requested",
            "requested_depth_completed",
            "rules_terminal_reports",
            "completed_reports_accepted_for_uci_output",
            "nodes",
        ]
    };
    for field in mandatory {
        work_count(&work[totals], field)?;
    }
    for (field, value) in work[totals].as_object().unwrap() {
        if field == "external_checker_work_incomplete" {
            require(
                kind == "pals" && (value.is_boolean() || value.is_null()),
                "external checker work completeness type invalid",
            )?;
            if startup {
                require(
                    value.as_bool() != Some(true),
                    "startup already observed incomplete external checker work",
                )?;
            }
            continue;
        }
        if !matches!(
            field.as_str(),
            "retained_situations_peak" | "unknown_root_children" | "max_completed_depth"
        ) {
            if let Some(observed) = value.as_u64() {
                if startup {
                    require(observed == 0, "startup work is not zero")?;
                }
            } else {
                require(value.is_null(), "work total type invalid")?;
            }
        }
    }
    if startup {
        require(
            go == 0 && failed == 0 && success == 0,
            "startup invocation already executed",
        )?;
    }
    Ok(())
}

pub fn validate_pals_search_work_records(
    lock: &LockedPalsArenaLaunchV3,
    role: NativeEngineRole,
    startup: &[u8],
    termination: &[u8],
    session: &str,
) -> Result<PalsProcessWorkAuditV3, ArenaError> {
    validate_pals_search_work_records_inner(lock, role, startup, termination, session, None)
}

fn validate_pals_search_work_records_inner(
    lock: &LockedPalsArenaLaunchV3,
    role: NativeEngineRole,
    startup: &[u8],
    termination: &[u8],
    session: &str,
    audited_native: Option<&PalsNativeSessionAuditV3>,
) -> Result<PalsProcessWorkAuditV3, ArenaError> {
    require(
        startup.len() <= MAX_JSON_BYTES && termination.len() <= MAX_JSON_BYTES,
        "work envelope byte budget exceeded",
    )?;
    let (s, t) = (json(startup)?, json(termination)?);
    validate_search_policy_pair(lock, role, &s["search_work"], &t["search_work"])?;
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
            PALS_NATIVE_STARTUP_V3_DOMAIN,
            PALS_NATIVE_TERMINATION_V3_DOMAIN,
            "pals",
        ),
        PalsEndpointLaunchV3::LegalOrderMock(_) => (
            PALS_SEARCH_WORK_STARTUP_V3_DOMAIN,
            PALS_SEARCH_WORK_TERMINATION_V3_DOMAIN,
            "pals",
        ),
        PalsEndpointLaunchV3::OwnCpu { .. } => (
            PALS_SEARCH_WORK_STARTUP_V3_DOMAIN,
            PALS_SEARCH_WORK_TERMINATION_V3_DOMAIN,
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
            value["schema_version"] == 3
                && value["domain"] == domain
                && value["endpoint_id"] == engine.id
                && value["launch_sha256"] == lock.sha256
                && value["process_id"] == pid
                && value["binary_sha256"] == engine.binary.sha256
                && value["service_exit_success"] == ending,
            "work envelope identity/start/end differs",
        )?;
        validate_pals_process_work_inner(&value["search_work"], kind, !ending, external)?;
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
            validate_pals_native_records(lock, role, startup, termination, session)?;
        }
    }
    Ok(PalsProcessWorkAuditV3 {
        endpoint_id: engine.id.clone(),
        process_id: pid,
        startup_sha256: digest(startup),
        termination_sha256: digest(termination),
        startup_bytes: startup.len() as u64,
        termination_bytes: termination.len() as u64,
        search_work: t["search_work"].clone(),
    })
}

#[cfg(target_os = "linux")]
fn read_work_file(directory: &cap_std::fs::Dir, filename: &str) -> Result<Vec<u8>, ArenaError> {
    use cap_fs_ext::{FollowSymlinks, OpenOptionsExt, OpenOptionsFollowExt};
    let metadata = directory
        .symlink_metadata(filename)
        .map_err(|e| ArenaError::Io(e.to_string()))?;
    require(
        metadata.is_file() && metadata.len() > 0 && metadata.len() <= MAX_JSON_BYTES as u64,
        "work file type/byte bound invalid",
    )?;
    let mut options = cap_std::fs::OpenOptions::new();
    options
        .read(true)
        .follow(FollowSymlinks::No)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
    let mut file = directory
        .open_with(filename, &options)
        .map_err(|e| ArenaError::Io(e.to_string()))?;
    require(
        file.metadata()
            .map_err(|e| ArenaError::Io(e.to_string()))?
            .is_file(),
        "pinned work handle is not regular",
    )?;
    let mut bytes = Vec::new();
    file.by_ref()
        .take(MAX_JSON_BYTES as u64 + 1)
        .read_to_end(&mut bytes)
        .map_err(|e| ArenaError::Io(e.to_string()))?;
    require(
        bytes.len() <= MAX_JSON_BYTES,
        "work file grew past byte bound",
    )?;
    Ok(bytes)
}

/// Only dedicated runtime receipt slots are read; cache/model directories are
/// never scanned as search work and no process is started by this collector.
#[cfg(target_os = "linux")]
pub fn collect_pals_process_work(
    lock: &LockedPalsArenaLaunchV3,
    role: NativeEngineRole,
    root: &cap_std::fs::Dir,
) -> Result<Vec<PalsProcessWorkAuditV3>, ArenaError> {
    collect_pals_process_work_with_audits(lock, role, root, &[])
}

#[cfg(target_os = "linux")]
fn collect_pals_process_work_with_audits(
    lock: &LockedPalsArenaLaunchV3,
    role: NativeEngineRole,
    root: &cap_std::fs::Dir,
    native_audits: &[PalsNativeSessionAuditV3],
) -> Result<Vec<PalsProcessWorkAuditV3>, ArenaError> {
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
            "pals-native-startup.v3.json",
            "pals-native-termination.v3.json",
        )
    } else {
        (
            "search-work-startup.v3.json",
            "search-work-termination.v3.json",
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
        records.push(validate_pals_search_work_records_inner(
            lock, role, &start, &end, name, audit,
        )?);
    }
    require(
        records.len() == 2 && records[0].process_id != records[1].process_id,
        "two distinct restarted game work processes required",
    )?;
    Ok(records)
}

#[cfg(any(target_os = "linux", test))]
fn endpoint_work_receipt(
    lock: &LockedPalsArenaLaunchV3,
    role: NativeEngineRole,
    work: Option<&PalsProcessWorkAuditV3>,
    native: Option<&PalsNativeSessionAuditV3>,
    preflight: &crate::ExternalUciPreflight,
) -> Result<PalsEndpointReceiptV3, ArenaError> {
    let index = role_index(role);
    let engine = &lock.input.semantic_lock.manifest.engines[index];
    let expected_policy = validate_launch_search_policy(&lock.input, index)?;
    require(
        preflight.engine_id == engine.id(),
        "work/preflight endpoint differs",
    )?;
    let mut options = BTreeMap::new();
    for (key, requested) in engine.requested_options() {
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
        PalsEngineV3::ReferenceUci(_) => require(
            work.is_none() && native.is_none(),
            "reference UCI cannot inherit RoveZero work",
        )?,
        PalsEngineV3::OwnCpu(_) => {
            let w = &work
                .ok_or_else(|| invalid("own CPU work unobserved"))?
                .search_work;
            validate_pals_process_work(w, "cpu", false)?;
            receipt.pals_search_policy =
                validate_search_policy_marker(w, expected_policy.as_ref())?;
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
        PalsEngineV3::Pals(p) => {
            let w = &work
                .ok_or_else(|| invalid("PALS search work unobserved"))?
                .search_work;
            let external = !p.cpu_r.is_own();
            validate_pals_process_work_inner(w, "pals", false, external)?;
            receipt.pals_search_policy =
                validate_search_policy_marker(w, expected_policy.as_ref())?;
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
                    validate_search_policy_marker(&n.raw_search_work, expected_policy.as_ref())?
                        == receipt.pals_search_policy,
                    "native/work validated search policy projection differs",
                )?;
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

#[cfg(any(target_os = "linux", test))]
fn record_endpoint_search_failure(
    endpoint: &PalsEndpointReceiptV3,
    failures: &mut BTreeSet<PalsRunFailureV3>,
) {
    if endpoint
        .search_failed_go_count
        .is_some_and(|failed| failed > 0)
    {
        failures.insert(PalsRunFailureV3::SearchFailure);
    }
}

#[cfg(target_os = "linux")]
fn core_game_outcome(
    game: &crate::GamePgnAudit,
) -> Result<(PalsResultV3, PalsTerminationV3, Option<String>), ArenaError> {
    let result = match game.result {
        crate::GameResult::WhiteWin => PalsResultV3::WhiteWin,
        crate::GameResult::BlackWin => PalsResultV3::BlackWin,
        crate::GameResult::Draw => PalsResultV3::Draw,
    };
    match game.classification.as_str() {
        "rules_terminal" | "accepted_claim" => {
            let reason = game
                .terminal_reason
                .as_deref()
                .ok_or_else(|| invalid("Rules termination observation absent"))?;
            Ok((
                result,
                if reason == "checkmate" {
                    PalsTerminationV3::Checkmate
                } else {
                    PalsTerminationV3::RulesDraw
                },
                None,
            ))
        }
        "engine_loss" => {
            let reason = match game
                .engine_failure
                .ok_or_else(|| invalid("engine loss cause absent"))?
            {
                crate::EngineFailureKind::IllegalMove => PalsTerminationV3::IllegalMove,
                crate::EngineFailureKind::Crash => PalsTerminationV3::EngineCrash,
                crate::EngineFailureKind::Timeout => PalsTerminationV3::TimeForfeit,
            };
            Ok((
                result,
                reason,
                Some(
                    game.loser_engine
                        .clone()
                        .ok_or_else(|| invalid("engine failure offender absent"))?,
                ),
            ))
        }
        "incomplete" => Ok((PalsResultV3::Incomplete, PalsTerminationV3::PlyLimit, None)),
        _ => Err(invalid("unsupported PGN outcome classification")),
    }
}

/// The pinned source uses one synchronous game worker, restart=on and recover
/// off: after Finished it destroys white then black before starting game 2.
/// Check those exact game/color windows; exit order alone is insufficient.
/// This codec still requires the separate owned-group/native completion gates.
pub fn validate_pals_game_process_trace(
    lock: &LockedPalsArenaLaunchV3,
    stdout: &[u8],
    known_work_pids: &[BTreeSet<u32>; 2],
) -> Result<[[u32; 2]; 2], ArenaError> {
    validate_pals_game_process_trace_context(lock, stdout, known_work_pids)
}
fn validate_pals_game_process_trace_context<L: PalsArenaValidationContext>(
    lock: &L,
    stdout: &[u8],
    known_work_pids: &[BTreeSet<u32>; 2],
) -> Result<[[u32; 2]; 2], ArenaError> {
    require(
        !stdout.is_empty() && stdout.len() <= 64 * 1024 * 1024 && stdout.ends_with(b"\n"),
        "game process trace byte/newline bound invalid",
    )?;
    let text =
        std::str::from_utf8(stdout).map_err(|_| invalid("game process trace is not UTF-8"))?;
    let view = lock.view();
    let engines = [
        lock.endpoint_view(NativeEngineRole::Baseline),
        lock.endpoint_view(NativeEngineRole::Candidate),
    ];
    let white_order = view
        .white_order
        .map(|role| lock.endpoint_view(role).id.clone());
    let mut result = [[0u32; 2]; 2];
    let mut seen = BTreeSet::new();
    let mut game = 0usize;
    // trace start, renderer start, running, trace result, renderer finish,
    // white exit, black exit. A later game cannot begin before both exits.
    let mut phase = 0u8;
    for (line_number, line) in text.lines().enumerate() {
        require(
            line_number < 131_072 && line.len() <= 4096,
            "game process trace line bound exceeded",
        )?;
        let trace = crate::native_exit::cuda_exit_trace_message(line);
        let relevant = line.starts_with("Started game ")
            || line.starts_with("Finished game ")
            || trace.is_some_and(|m| {
                m.starts_with("Game ")
                    || m.starts_with("Process with pid:")
                    || m.starts_with("Force terminating process with pid:")
            });
        if !relevant {
            continue;
        }
        require(game < 2, "extra game/process trace after two game windows")?;
        let number = game + 1;
        let white = &white_order[game];
        let black = engines
            .iter()
            .find(|e| e.id.as_str() != white)
            .ok_or_else(|| invalid("black endpoint absent"))?
            .id
            .as_str();
        let white_role = engines
            .iter()
            .position(|e| e.id.as_str() == white)
            .ok_or_else(|| invalid("white endpoint absent"))?;
        let black_role = 1 - white_role;
        let trace_start = format!("Game {number} between {white} and {black} starting");
        let trace_finish = format!("Game {number} between {white} and {black} finished");
        let renderer_start = format!("Started game {number} of 2 ({white} vs {black})");
        let renderer_finish = format!("Finished game {number} ({white} vs {black}): ");
        if trace == Some(trace_start.as_str()) {
            require(phase == 0, "game start outside completed prior window")?;
            phase = 1;
        } else if line == renderer_start {
            require(phase == 1, "renderer start lacks matching TRACE game/color")?;
            phase = 2;
        } else if trace == Some(trace_finish.as_str()) {
            require(phase == 2, "TRACE finished outside running game")?;
            phase = 3;
        } else if trace
            .is_some_and(|m| m.starts_with(&format!("Game {number} finished with result ")))
        {
            require(phase == 3, "result TRACE outside matching finished game")?;
            phase = 4;
        } else if let Some(outcome) = line.strip_prefix(&renderer_finish) {
            require(
                phase == 4
                    && outcome.split_once(' ').is_some_and(|(r, a)| {
                        matches!(r, "1-0" | "0-1" | "1/2-1/2" | "*")
                            && a.starts_with('{')
                            && a.ends_with('}')
                    }),
                "renderer finish/result window invalid",
            )?;
            phase = 5;
        } else if let Some(message) = trace.and_then(|m| m.strip_prefix("Process with pid: ")) {
            let (pid, status) = message
                .split_once(" terminated with status: ")
                .ok_or_else(|| invalid("game exit renderer invalid"))?;
            let parsed = pid
                .parse::<u32>()
                .map_err(|_| invalid("game exit PID invalid"))?;
            require(
                matches!(phase, 5 | 6)
                    && pid == parsed.to_string()
                    && parsed > 0
                    && parsed <= i32::MAX as u32
                    && status == "0"
                    && seen.insert(parsed),
                "game exit outside finished window, failed, or duplicate",
            )?;
            let role = if phase == 5 { white_role } else { black_role };
            require(
                known_work_pids[role].is_empty() || known_work_pids[role].contains(&parsed),
                "white/black exit does not match actual work endpoint PID",
            )?;
            result[role][game] = parsed;
            if phase == 5 {
                phase = 6;
            } else {
                game += 1;
                phase = 0;
            }
        } else {
            return Err(invalid("foreign/unsupported game or exit renderer"));
        }
    }
    require(
        game == 2 && phase == 0 && seen.len() == 4,
        "game process correspondence incomplete; no relative-order fallback",
    )?;
    Ok(result)
}

/// Separates the generated PGN's native description from its public producer
/// source URL. This URL does not claim that the local PGN is publicly hosted.
#[derive(Clone, Debug, Serialize)]
pub struct PalsPgnProvenanceV3 {
    pub native_artifact: ArtifactRef,
    pub producer_source_url: String,
    pub producer_source_commit: String,
    pub producer_binary_sha256: String,
    pub description: String,
}

#[cfg(any(target_os = "linux", test))]
fn project_pals_core_pgn(
    lock: &LockedPalsArenaLaunchV3,
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

/// A missing observation leaves the numeric Core receipt absent. The original
/// native receipt and PGN remain authoritative evidence of failures, never an
/// empty-success or an omitted engine loss.
#[derive(Clone, Debug, Serialize)]
pub struct PalsCoreAssemblyV3 {
    pub domain: String,
    pub arena_launch_sha256: String,
    pub native_receipt: ArtifactRef,
    pub wall_time_ms: Option<u64>,
    pub cleanup_time_ms: Option<u64>,
    pub timing_scope: String,
    pub game_process_correspondence: String,
    pub work: Vec<PalsProcessWorkAuditV3>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pgn_provenance: Option<PalsPgnProvenanceV3>,
    pub core: Option<PalsRunReceiptV3>,
    pub assembly_error: Option<String>,
    pub training_executed: bool,
}

#[cfg(target_os = "linux")]
pub fn assemble_pals_core_receipt(
    output: &PalsObservedPairOutputV3,
    cleanup_time_ms: Option<u64>,
) -> PalsCoreAssemblyV3 {
    let native = &output.native;
    let owner = native.launch_owner();
    let lock = owner.spec();
    let mut assembly = PalsCoreAssemblyV3 { domain: "rz-pals-core-assembly-v3/1".into(), arena_launch_sha256: lock.sha256.clone(),
        native_receipt: native.receipt_artifact.clone(), wall_time_ms: Some(output.wall_time_ms), cleanup_time_ms,
        timing_scope: "UCI preflight through mandatory arena postcheck/receipt; snapshot preparation separate; cleanup is supervisor-owned measured interval".into(),
        game_process_correspondence: "pinned Fastchess 1.8.2 source: one synchronous worker, restart=on, recover=false; TRACE/renderer game id and color plus finished-white-exit-black-exit windows; separate native/owned-group completion gates required".into(),
        work: vec![], pgn_provenance: None, core: None, assembly_error: None, training_executed: false };
    let result: Result<PalsRunReceiptV3, ArenaError> = (|| {
        lock.input.require_reap_status_runner()?;
        lock.input.require_actual_native_epoch()?;
        for role in [NativeEngineRole::Baseline, NativeEngineRole::Candidate] {
            assembly.work.extend(collect_pals_process_work_with_audits(
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
        let mapped =
            validate_pals_game_process_trace(lock, &native.process().stdout, &known_work_pids)?;
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
            )?;
        }
        let native_pgn_artifact = receipts
            .artifacts
            .iter()
            .find(|a| a.path.ends_with("/match.pgn"))
            .ok_or_else(|| invalid("retained PGN artifact absent"))?;
        let (pgn_artifact, provenance) = project_pals_core_pgn(lock, native_pgn_artifact)?;
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
                let endpoint = endpoint_work_receipt(lock, role, work, nn, preflight)?;
                record_endpoint_search_failure(&endpoint, &mut failures);
                engines.push(endpoint);
            }
            let (result, termination, failed_endpoint) = core_game_outcome(game)?;
            if result == PalsResultV3::Incomplete {
                failures.insert(PalsRunFailureV3::IncompleteGame);
            }
            games.push(PalsGameReceiptV3 {
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
        let core = PalsRunReceiptV3 {
            domain: PALS_RECEIPT_V3_DOMAIN.into(),
            run_id: m.run_id.clone(),
            pair_id: m.pair_id.clone(),
            lock_sha256: lock.input.semantic_lock.canonical_sha256.clone(),
            training_executed: false,
            wall_time_ms: output.wall_time_ms,
            cleanup_time_ms: cleanup,
            pair_eligible: failures.is_empty(),
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
    assembly
}

#[cfg(target_os = "linux")]
pub fn save_pals_core_assembly(
    output: &PalsObservedPairOutputV3,
    assembly: &PalsCoreAssemblyV3,
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
        bytes.len() <= 64 * 1024,
        "core assembly reservation exceeds 64KiB",
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
    let mut options = cap_std::fs::OpenOptions::new();
    options
        .write(true)
        .create_new(true)
        .follow(FollowSymlinks::No);
    let mut file = owner
        .snapshot
        .directory
        .open_with("pals-core-assembly.v3.json", &options)
        .map_err(|e| ArenaError::Io(e.to_string()))?;
    file.write_all(&bytes)
        .and_then(|()| file.sync_all())
        .map_err(|e| ArenaError::Io(e.to_string()))?;
    Ok(ArtifactRef { path: format!("{}/pals-core-assembly.v3.json", owner.snapshot.output_directory), sha256: digest(&bytes), bytes: bytes.len() as u64,
        source: "PALS V3 actual arena work/clock/physical-lifetime assembly; raw native receipt retained".into(), license: "MIT execution evidence; external asset rights remain separate".into() })
}

/// Independently checks descriptor contents before preparation, using the same
/// bounded no-follow artifact verifier as every actual launch input.
///
/// External profile verification is intentionally first. It is still only a
/// preparation observation; unsupported execution/resource/closure admission
/// remains fail-closed in `PalsArenaLaunchV3::validate`.
pub fn verify_pals_launch_exports(
    spec: &LockedPalsArenaLaunchV3,
    source_root: &Path,
) -> Result<(), ArenaError> {
    verify_pals_launch_external_cpu_r_profiles(&spec.input, source_root)?;
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

/// Read-only preparation entry point for a not-yet-executable external launch.
/// Compares the unchanged source profile with its independently registered
/// Linux program/cwd and semantic pins before any child admission is attempted.
/// This deliberately does not call `lock()` or authorize execution/Core output.
pub fn verify_pals_launch_external_cpu_r_profiles(
    input: &PalsArenaLaunchV3,
    source_root: &Path,
) -> Result<Vec<PalsExternalCpuRProfileAuditV3>, ArenaError> {
    input.semantic_lock.verify()?;
    let mut observations = Vec::new();
    for role in [NativeEngineRole::Baseline, NativeEngineRole::Candidate] {
        if let Some((_, binding)) = input.external_cpu_r_binding(role_index(role))? {
            observations.push(verify_pals_external_cpu_r_profile(
                &input.semantic_lock,
                role,
                source_root,
                Path::new(&binding.registered_program),
                Path::new(&binding.registered_working_directory),
            )?);
        }
    }
    Ok(observations)
}

/// Checks inherited limits; this does not install limits or claim a GPU peak.
/// Both engines are in the same serial-game cgroup in this first executor.
#[cfg(target_os = "linux")]
pub fn verify_pals_inherited_resources(
    spec: &LockedPalsArenaLaunchV3,
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
#[cfg(any(target_os = "linux", test))]
fn parse_pals_affinity(cpus: &str) -> Result<Vec<u32>, ArenaError> {
    require(
        !cpus.is_empty() && cpus.len() <= 4096,
        "affinity observation is empty or oversized",
    )?;
    let mut result = BTreeSet::new();
    for segment in cpus.split(',') {
        if let Some((lo, hi)) = segment.split_once('-') {
            let lo = lo
                .parse::<u32>()
                .map_err(|_| invalid("affinity range invalid"))?;
            let hi = hi
                .parse::<u32>()
                .map_err(|_| invalid("affinity range invalid"))?;
            require(hi >= lo && hi - lo < 64, "affinity range budget exceeded")?;
            for cpu in lo..=hi {
                require(result.insert(cpu), "duplicate CPU affinity observation")?;
            }
        } else {
            require(
                result.insert(
                    segment
                        .parse::<u32>()
                        .map_err(|_| invalid("affinity value invalid"))?,
                ),
                "duplicate CPU affinity observation",
            )?;
        }
        require(result.len() <= 64, "affinity count budget exceeded")?;
    }
    Ok(result.into_iter().collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rz_experiments::*;
    fn asset(path: &str) -> ArtifactRef {
        ArtifactRef {
            path: path.into(),
            sha256: "a".repeat(64),
            bytes: 1000,
            source: "https://example.org/public-source".into(),
            license: "MIT".into(),
        }
    }
    fn component(id: &str) -> PalsComponentV3 {
        PalsComponentV3 {
            semantic_id: id.into(),
            implementation_sha256: "b".repeat(64),
            options: BTreeMap::new(),
        }
    }
    fn cpu() -> PalsOwnCpuV3 {
        PalsOwnCpuV3 {
            core: component("rz-cpu-pvs/0.1"),
            training_profile: component("cpu-independent-conservative-v1"),
            runtime_profile: component("cpu-plan-assisted-conservative-v1"),
            evaluation: component("bootstrap-material-pst-v1"),
            max_nodes_per_task: 100_000,
            max_depth: 16,
            max_task_ms: 900_000,
            max_tt_bytes: 16 * 1024 * 1024,
        }
    }
    pub(super) fn fixture() -> PalsArenaLaunchV3 {
        let pals = PalsEndpointV3 {
            id: "pals".into(),
            binary: asset("bin/rove"),
            source_commit: "c".repeat(40),
            model: PalsModelIdentityV3 {
                architecture: "explicit-legal-order-role-mock-v1".into(),
                input_schema: "rz-pals-rules-fields-v1".into(),
                policy_head: "candidate-policy/1".into(),
                value_head: "stm-wdl/1".into(),
                implementation_sha256: "b".repeat(64),
                weights: PalsWeightIdentityV3::DeterministicMock { seed: 0 },
                frozen_epoch: 0,
                backend: PalsModelBackendV3::DeterministicMock,
                precision: PalsPrecisionV3::Fp32,
                max_batch_width: 1,
                exported_roles: BTreeSet::from([PalsRoleV3::Proposer, PalsRoleV3::Critic]),
            },
            cpu: cpu(),
            cpu_r: PalsCpuRSelectionV3::Own,
            search: component("pals"),
            runtime: component("single-owner/1"),
            pools: PalsPoolLimitsV3 {
                states: 4096,
                line_chunks: 65536,
                situations: 4096,
                observations: 65536,
                tasks: 32768,
                role_states: 1,
                memory_pages: 1,
                queue_requests: 1,
                host_bytes: 12 << 30,
                device_bytes: 0,
            },
            requested_options: BTreeMap::new(),
        };
        let resources = PalsResourcePolicyV3 {
            cpu_threads: 2,
            cpu_affinity: vec![0, 2],
            memory_high_bytes: 6 << 30,
            memory_max_bytes: 12 << 30,
            swap_max_bytes: 0,
            requested_gpu: None,
            device_allocation_max_bytes: 0,
        };
        let semantic = PalsRunManifestV3 {
            schema_version: 3,
            run_id: "pals-test".into(),
            pair_id: "pair-test".into(),
            contract_revision: "pals/0.1".into(),
            rules_profile: "standard-complete-history/1".into(),
            comparison: PalsComparisonV3::System,
            declared_changes: BTreeSet::from([PalsChangeAxisV3::Endpoint]),
            training_executed: false,
            engines: [
                PalsEngineV3::Pals(Box::new(pals)),
                PalsEngineV3::OwnCpu(Box::new(PalsCpuEndpointV3 {
                    id: "cpu".into(),
                    binary: asset("bin/rove"),
                    source_commit: "c".repeat(40),
                    cpu: cpu(),
                    requested_options: BTreeMap::new(),
                })),
            ],
            resources: [resources.clone(), resources],
            pilot: PalsPilotV3 {
                position_command: "position startpos".into(),
                games: 2,
                white_order: ["pals".into(), "cpu".into()],
                base_ms: 120_000,
                increment_ms: 1000,
                max_plies: 256,
                seed: 1,
                wall_time_max_ms: 900_000,
                cleanup_max_ms: 30_000,
                handshake_max_ms: 30_000,
                concurrent_games: 1,
                restart_processes_each_game: true,
                ponder: false,
                score_adjudication: false,
                elo_claim: false,
            },
        };
        let mut patch = asset("runner/clock.patch");
        patch.sha256 = FASTCHESS_CLOCK_PATCH_SHA256.into();
        patch.bytes = FASTCHESS_CLOCK_PATCH_BYTES;
        PalsArenaLaunchV3 {
            domain: PALS_ARENA_V3_DOMAIN.into(),
            semantic_lock: semantic.lock().unwrap(),
            runner: ToolIdentity {
                version: crate::FASTCHESS_VERSION.into(),
                source_url: crate::FASTCHESS_SOURCE_URL.into(),
                source_commit: crate::FASTCHESS_SOURCE_COMMIT.into(),
                binary: asset("runner/fastchess"),
                dirty: true,
                dirty_patch: Some(patch),
                build_mode: "release".into(),
                compiler: "clang".into(),
                target: "x86_64-unknown-linux-gnu".into(),
                isa: "x86-64".into(),
            },
            opening_artifact: asset("opening/startpos.pgn"),
            endpoints: [
                PalsEndpointLaunchV3::LegalOrderMock(PalsSearchLaunchV3 {
                    max_rounds: 16,
                    max_cpu_nodes: 100_000,
                    cpu_depth: 2,
                    max_situations: 4096,
                    post_repair_recheck: None,
                }),
                PalsEndpointLaunchV3::OwnCpu {
                    max_depth: 8,
                    max_nodes: 10_000,
                    tt_entries: 128,
                },
            ],
            budget: NativeResourceBudgetV1 {
                max_input_bytes: 32 * 1024 * 1024,
                max_output_bytes: 8 * 1024 * 1024,
                max_runtime_bytes: 1024 * 1024 * 1024,
                max_artifact_bytes: 2 * 1024 * 1024 * 1024,
                max_child_processes: 3,
                max_runtime_files: 64,
                max_runtime_depth: 4,
                address_space_per_process_bytes: 0,
            },
        }
    }
    #[test]
    fn declared_external_cpu_r_cannot_silently_launch_as_own() {
        for mut f in [fixture(), native_fixture()] {
            assert!(f.clone().lock().is_ok());
            let mut binary = asset("helper/stockfish");
            binary.source = "https://github.com/official-stockfish/Stockfish".into();
            binary.license = "GPL-3.0-or-later".into();
            let PalsEngineV3::Pals(e) = &mut f.semantic_lock.manifest.engines[0] else {
                unreachable!()
            };
            e.cpu_r = PalsCpuRSelectionV3::ExternalUci(Box::new(PalsExternalCpuRV3 {
                selection: PalsExternalCpuRSelectionV3::StockfishEmbeddedNnue,
                profile: asset("helper/profile.json"),
                profile_canonical_sha256: "a".repeat(64),
                binary,
                resolver: PalsExternalCpuRResolverV3 {
                    version: "pals-model-wdl-restricted/0.1".into(),
                    semantics_sha256: "a".repeat(64),
                },
                policy: PalsExternalCpuRPolicyV3 {
                    resource_scope: PalsExternalCpuRResourceScopeV3::InheritedParentCgroup,
                    max_owners: 1,
                    max_active_tasks: 1,
                    max_process_leaders: 1,
                    inherited_kernel_tasks_max: 128,
                    threads_max: 2,
                    hash_mib_max: 16,
                    max_depth: 8,
                    max_prefix_plies: 64,
                    max_nodes_per_task: 10_000,
                    handshake_max_ms: 1000,
                    task_wall_time_max_ms: 1000,
                    stop_grace_max_ms: 100,
                    shutdown_grace_max_ms: 100,
                    lifetime_output_bytes_max: 4096,
                    line_bytes_max: 1024,
                },
            }));
            // Semantic preparation is permitted. No child/profile/model is
            // started, copied or read by this declaration-only fixture.
            f.semantic_lock = f.semantic_lock.manifest.lock().unwrap();
            let error = f.validate().unwrap_err();
            assert!(matches!(&error, ArenaError::Integrity(_)));
            assert!(error.to_string().contains("external CPU_R"));
            // Direct closed-lane preflight refuses a foreign checker before
            // helper assets/profile/model dispatch, even in a native recipe.
            let mut selected = f.clone();
            let PalsEngineV3::Pals(p) = &mut selected.semantic_lock.manifest.engines[0] else {
                unreachable!()
            };
            p.search.semantic_id = "pals-post-repair-recheck/1".into();
            p.search.options.insert(
                "post_repair_recheck".into(),
                "same-repaired-line-once-v1".into(),
            );
            match &mut selected.endpoints[0] {
                PalsEndpointLaunchV3::LegalOrderMock(s) => {
                    s.post_repair_recheck =
                        Some(PalsPostRepairRecheckPolicyV3::SameRepairedLineOnceV1)
                }
                PalsEndpointLaunchV3::OnnxCpu(n) => {
                    n.search.post_repair_recheck =
                        Some(PalsPostRepairRecheckPolicyV3::SameRepairedLineOnceV1)
                }
                _ => unreachable!(),
            }
            assert!(validate_launch_search_policy(&selected, 0).is_err());
            assert!(f.lock().is_err());
        }
    }
    #[cfg(all(
        target_os = "linux",
        any(feature = "pals-collection-onnx", feature = "native-cuda")
    ))]
    struct ExternalProfileFixture {
        root: std::path::PathBuf,
        program: std::path::PathBuf,
        cwd: std::path::PathBuf,
        lock: PalsInputLockV3,
    }
    #[cfg(all(
        target_os = "linux",
        any(feature = "pals-collection-onnx", feature = "native-cuda")
    ))]
    impl ExternalProfileFixture {
        fn new() -> Self {
            use std::sync::atomic::{AtomicU64, Ordering};
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let root = std::env::temp_dir().join(format!(
                "rz-pals-profile-crosscheck-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed),
            ));
            std::fs::create_dir(&root).unwrap();
            // Neither the program nor cwd exists. This fixture verifies only
            // profile bytes/declared identities and must not open or spawn them.
            let cwd = root.join("registered-cas");
            let program = cwd.join("stockfish");
            let wire = serde_json::json!({
                "schema":"rz-pals-external-checker-profile/1",
                "selection":"stockfish_embedded_nnue",
                "program":program,
                "working_directory":cwd,
                "arguments":[],
                "identity":{
                    "adapter_semantics":rz_search::external_cpu::EXTERNAL_UCI_SCORE_SEMANTICS,
                    "binary_sha256":"a".repeat(64),
                    "launch_arguments_sha256":rz_search::external_cpu::arguments_sha256(&[]),
                    "declared_name":"Stockfish profile fixture",
                    "declared_version":"fixture-1",
                    "declared_source":"https://github.com/official-stockfish/Stockfish",
                    "declared_license":"GPL-3.0-or-later",
                    "options":{
                        "Threads":"2","Hash":"16","Ponder":"false",
                        "UCI_Chess960":"false","MultiPV":"1",
                        "SyzygyPath":"","SyzygyProbeLimit":"0"
                    },
                    "assets":[],
                    "model_metadata":{
                        "weights_sha256":null,"training":"unknown",
                        "declared_rights":null,"precision":null
                    }
                },
                "expected_uci_name":"Stockfish profile fixture",
                "max_depth":8,"max_prefix_plies":64,"handshake_timeout_ms":1000,
                "max_task_wall_time_ms":1000,"stop_grace_ms":100,"shutdown_grace_ms":100,
                "max_output_bytes":4096,"max_line_bytes":1024
            });
            let bytes = serde_json::to_vec(&wire).unwrap();
            let profile_path = root.join("profile.json");
            std::fs::write(&profile_path, &bytes).unwrap();
            let file_sha256 = digest(&bytes);
            let loaded =
                rz_uci::pals_checker_profile::PalsCheckerProfile::load(&profile_path, &file_sha256)
                    .unwrap();
            let mut f = fixture();
            let PalsEngineV3::Pals(e) = &mut f.semantic_lock.manifest.engines[0] else {
                unreachable!()
            };
            let mut profile = asset("profile.json");
            profile.sha256 = file_sha256;
            profile.bytes = bytes.len() as u64;
            let mut binary = asset("helper/stockfish");
            binary.source = "https://github.com/official-stockfish/Stockfish".into();
            binary.license = "GPL-3.0-or-later".into();
            e.cpu_r = PalsCpuRSelectionV3::ExternalUci(Box::new(PalsExternalCpuRV3 {
                selection: PalsExternalCpuRSelectionV3::StockfishEmbeddedNnue,
                profile,
                profile_canonical_sha256: loaded.canonical_sha256().into(),
                binary,
                resolver: PalsExternalCpuRResolverV3 {
                    version: rz_search::pals::value::MODEL_WDL_RESOLVER_VERSION.into(),
                    semantics_sha256: digest(
                        rz_search::pals::value::MODEL_WDL_RESOLVER_SEMANTICS.as_bytes(),
                    ),
                },
                policy: PalsExternalCpuRPolicyV3 {
                    resource_scope: PalsExternalCpuRResourceScopeV3::InheritedParentCgroup,
                    max_owners: 1,
                    max_active_tasks: 1,
                    max_process_leaders: 1,
                    inherited_kernel_tasks_max: 128,
                    threads_max: 2,
                    hash_mib_max: 16,
                    max_depth: 8,
                    max_prefix_plies: 64,
                    max_nodes_per_task: 10_000,
                    handshake_max_ms: 1000,
                    task_wall_time_max_ms: 1000,
                    stop_grace_max_ms: 100,
                    shutdown_grace_max_ms: 100,
                    lifetime_output_bytes_max: 4096,
                    line_bytes_max: 1024,
                },
            }));
            Self {
                root,
                program,
                cwd,
                lock: f.semantic_lock.manifest.lock().unwrap(),
            }
        }
        fn verify(&self) -> Result<PalsExternalCpuRProfileAuditV3, ArenaError> {
            verify_pals_external_cpu_r_profile(
                &self.lock,
                NativeEngineRole::Baseline,
                &self.root,
                &self.program,
                &self.cwd,
            )
        }
        fn native_launch(&self) -> PalsArenaLaunchV3 {
            let mut input = native_fixture();
            let PalsEngineV3::Pals(source) = &self.lock.manifest.engines[0] else {
                unreachable!()
            };
            let PalsEngineV3::Pals(target) = &mut input.semantic_lock.manifest.engines[0] else {
                unreachable!()
            };
            target.cpu_r = source.cpu_r.clone();
            input.semantic_lock = input.semantic_lock.manifest.lock().unwrap();
            let PalsEndpointLaunchV3::OnnxCpu(model) = &mut input.endpoints[0] else {
                unreachable!()
            };
            model.external_cpu_r = Some(PalsExternalCpuRLaunchV3 {
                registered_program: self.program.to_str().unwrap().into(),
                registered_working_directory: self.cwd.to_str().unwrap().into(),
            });
            input
        }
        fn mutate_declaration(&mut self, change: fn(&mut PalsExternalCpuRV3)) {
            let PalsEngineV3::Pals(e) = &mut self.lock.manifest.engines[0] else {
                unreachable!()
            };
            let PalsCpuRSelectionV3::ExternalUci(c) = &mut e.cpu_r else {
                unreachable!()
            };
            change(c);
            self.lock = self.lock.manifest.lock().unwrap();
        }
    }
    #[cfg(all(
        target_os = "linux",
        any(feature = "pals-collection-onnx", feature = "native-cuda")
    ))]
    impl Drop for ExternalProfileFixture {
        fn drop(&mut self) {
            std::fs::remove_dir_all(&self.root).unwrap();
        }
    }
    #[cfg(all(
        target_os = "linux",
        any(feature = "pals-collection-onnx", feature = "native-cuda")
    ))]
    #[test]
    fn external_launch_preparation_retains_pins_without_rewriting_or_admitting_a_child() {
        let fixture = ExternalProfileFixture::new();
        let input = fixture.native_launch();
        let observed = verify_pals_launch_external_cpu_r_profiles(&input, &fixture.root).unwrap();
        assert_eq!(observed.len(), 1);
        assert_eq!(
            observed[0].registered_program,
            fixture.program.to_str().unwrap()
        );
        assert!(!observed[0].child_spawned && !observed[0].execution_admission_completed);
        let (helper, _) = input.external_cpu_r_binding(0).unwrap().unwrap();
        assert!(input.declared_artifacts().contains(&&helper.profile));
        assert!(input.declared_artifacts().contains(&&helper.binary));
        assert!(
            input
                .named_assets()
                .iter()
                .any(|(artifact, path)| *artifact == &helper.profile
                    && path
                        == &format!("pals-helper-profile-{}/profile.json", helper.profile.sha256))
        );
        let view = input.endpoint(0).unwrap();
        assert_eq!(
            view.expected_uci_name,
            "RoveZero PALS P/C ONNX CPU + external UCI CPU_R"
        );
        let index = view
            .assets
            .iter()
            .position(|asset| asset == &helper.profile)
            .unwrap();
        assert!(
            view.arguments
                .contains(&format!("--pals-cpu-profile={{{{asset:{index}}}}}"))
        );
        assert!(view.arguments.contains(&format!(
            "--pals-cpu-profile-sha256={}",
            helper.profile.sha256
        )));
        assert!(view.assets.contains(&helper.binary));
        assert!(
            !view
                .arguments
                .iter()
                .any(|argument| argument.contains(fixture.program.to_str().unwrap()))
        );
        assert!(input.lock().is_ok());
        let mut wrong = input.clone();
        let PalsEndpointLaunchV3::OnnxCpu(model) = &mut wrong.endpoints[0] else {
            unreachable!()
        };
        model.external_cpu_r.as_mut().unwrap().registered_program =
            fixture.cwd.join("another-program").to_str().unwrap().into();
        assert!(verify_pals_launch_external_cpu_r_profiles(&wrong, &fixture.root).is_err());
        let mut own = native_fixture();
        let PalsEndpointLaunchV3::OnnxCpu(model) = &mut own.endpoints[0] else {
            unreachable!()
        };
        model.external_cpu_r = input.endpoints[0]
            .native_model()
            .unwrap()
            .external_cpu_r
            .clone();
        assert!(own.lock().is_err());
        assert!(verify_pals_launch_external_cpu_r_profiles(&own, &fixture.root).is_err());
    }
    #[cfg(all(
        target_os = "linux",
        any(feature = "pals-collection-onnx", feature = "native-cuda")
    ))]
    #[test]
    fn external_preflight_evidence_is_separate_from_two_game_process_slots() {
        let fixture = ExternalProfileFixture::new();
        let input = fixture.native_launch();
        // Construct only a prospective argv view. Actual admission still needs
        // owner-pinned preflight evidence; this fixture starts no process.
        let view = LockedPalsArenaLaunchV3 {
            sha256: crate::canonical_sha256(&input).unwrap(),
            opening: input.opening(),
            endpoint_views: [input.endpoint(0).unwrap(), input.endpoint(1).unwrap()],
            input,
        };
        assert!(view.validate_execution().is_ok());
        let game_root = fixture.root.join("baseline-runtime");
        let preflight_root = fixture.root.join("external-baseline-preflight-runtime");
        assert!(
            view.uses_separate_preflight_runtime_root(NativeEngineRole::Baseline)
                .unwrap()
        );
        assert!(
            !view
                .uses_separate_preflight_runtime_root(NativeEngineRole::Candidate)
                .unwrap()
        );
        assert!(
            view.preflight_arguments(NativeEngineRole::Baseline, &game_root)
                .is_err()
        );
        let preflight = view
            .preflight_arguments(NativeEngineRole::Baseline, &preflight_root)
            .unwrap();
        let runtime = view
            .runtime_arguments(NativeEngineRole::Baseline, &game_root)
            .unwrap();
        assert!(
            preflight.contains(&format!("--pals-output-root={}", preflight_root.display()).into())
        );
        assert!(runtime.contains(&format!("--pals-output-root={}", game_root.display()).into()));
        for arguments in [&preflight, &runtime] {
            assert!(arguments.contains(&format!("--pals-launch-sha256={}", view.sha256).into()));
            assert!(arguments.contains(&OsString::from("--pals-endpoint-id=pals")));
        }
        assert!(!preflight.contains(&format!("--pals-output-root={}", game_root.display()).into()));
        assert!(
            pals_external_cpu_r_preflight_root(NativeEngineRole::Candidate, &game_root).is_err()
        );
        assert!(!game_root.exists() && !preflight_root.exists());
    }
    #[cfg(all(
        target_os = "linux",
        any(feature = "pals-collection-onnx", feature = "native-cuda")
    ))]
    fn helper_pair_fixture(
        fixture: &ExternalProfileFixture,
    ) -> (
        LockedPalsArenaLaunchV3,
        rz_uci::pals_checker_profile::LoadedPalsCheckerProfile,
        serde_json::Value,
        serde_json::Value,
    ) {
        use rz_search::cpu_checker::CpuChecker;
        let input = fixture.native_launch();
        let lock = LockedPalsArenaLaunchV3 {
            sha256: crate::canonical_sha256(&input).unwrap(),
            opening: input.opening(),
            endpoint_views: [input.endpoint(0).unwrap(), input.endpoint(1).unwrap()],
            input,
        };
        let (declaration, _) = lock.input.external_cpu_r_binding(0).unwrap().unwrap();
        let loaded = rz_uci::pals_checker_profile::PalsCheckerProfile::load(
            &fixture.root.join("profile.json"),
            &declaration.profile.sha256,
        )
        .unwrap();
        let checker = loaded.create_checker().unwrap();
        let caps = checker.capabilities();
        let n = lock.native(NativeEngineRole::Baseline).unwrap();
        let PalsEngineV3::Pals(e) = &lock.input.semantic_lock.manifest.engines[0] else {
            unreachable!()
        };
        let PalsWeightIdentityV3::Untrained { artifact, .. } = &e.model.weights else {
            unreachable!()
        };
        let model_value = serde_json::json!({"semantics":rz_search::pals::value::MODEL_WDL_VALUE_SEMANTICS,
            "model":format!("pals-onnx-pc-fp32-{}",n.export.sha256),"encoding":n.encoding_semantic_sha256,
            "precision":"fp32","model_epoch":hash_array(&artifact.sha256)});
        let registration = serde_json::json!({"identity":checker.identity(),"conditions":checker.conditions(),
            "capabilities":{"max_depth":caps.max_depth,"max_prefix_plies":caps.max_prefix_plies,"max_root_moves":caps.max_root_moves,
                "root_moves":caps.root_moves,"divergence":caps.divergence,"resume":caps.resume,"selective_search":caps.selective_search},
            "role_model":model_value["model"],"model_value":model_value,
            "resolver_version":rz_search::pals::value::MODEL_WDL_RESOLVER_VERSION,
            "resolver_semantics":rz_search::pals::value::MODEL_WDL_RESOLVER_SEMANTICS});
        let mut sha = Sha256::new();
        sha.update(b"rz-pals-checker-registration/1\0");
        sha.update(serde_json::to_vec(&registration).unwrap());
        let limits = serde_json::json!({"mount_point":"/sys/fs/cgroup","mount_root":"/","resolved_directory":"/sys/fs/cgroup/test-only",
            "memory_high":{"kind":"numeric","value":6u64<<30},"memory_max":{"kind":"numeric","value":12u64<<30},
            "memory_swap_max":{"kind":"numeric","value":0},"pids_max":{"kind":"numeric","value":128}});
        let process = |pid, parent, ticks| {
            serde_json::json!({"pid":pid,"parent_pid":parent,"process_group":pid,
            "proc_start_ticks_before":ticks,"proc_start_ticks_after":ticks,"cgroup_v2_membership":"/test-only",
            "membership_path_view":"observer_procfs_and_cgroup2_mount_view","cgroup_namespace_inode":777,"cpu_allowed_list":"0,2",
            "threads":[{"tid":pid,"proc_start_ticks_before":ticks,"proc_start_ticks_after":ticks,"cpu_allowed_list":"0,2"}],
            "thread_observation_scope":"bounded_ready_boundary_thread_snapshot_not_lifetime_enforcement","cgroup_limits":limits})
        };
        let ready = serde_json::json!({"schema_version":1,"domain":"rz-pals-checker-ready-resources/1","scope":"linux_ready_boundary_snapshot",
            "observed_unix_us":10,"observation_elapsed_us":1,"parent":process(100,99,500),"helper":process(101,100,600),
            "cgroup_membership_equal":true,"cgroup_namespace_equal":true,"cpu_allowed_list_equal":true,
            "additional_allocation":"none_declared_not_an_enforcement_proof",
            "read_consistency":"identity_membership_affinity_bracketed_limits_sequential_not_atomic"});
        let helper = serde_json::json!({"schema_version":1,"domain":"rz-pals-checker-process/1","profile_file_sha256":loaded.file_sha256(),
            "profile_canonical_sha256":loaded.canonical_sha256(),"registration_sha256":format!("{:x}",sha.finalize()),"registration":registration,
            "startup_handshake_completed":true,"observed_uci":{"name":loaded.profile().expected_uci_name,"author":null},
            "startup_resource_observation":ready,"requested_option_checks_completed":true,"applied_option_values":"unknown",
            "latest_attempt":null,"shutdown":null,"cleanup_complete":false,"started_owner_exit_and_drains_confirmed":false});
        let mut totals = serde_json::json!({"external_checker_tasks":0,"external_checker_reports":0,"external_checker_nodes_observed":0,
            "external_checker_node_budget_reserved":0,"consumed_external_checker_tasks":0,"external_checker_work_incomplete":false,
            "value_calls":0,"completed_value_calls":0,"accepted_value_outputs":0});
        for field in [
            "completed_proposer_calls",
            "completed_repair_calls",
            "completed_critic_calls",
            "cpu_tasks_requested",
            "cpu_tasks",
            "completed_cpu_tasks",
            "reused_completed_cpu_tasks_consumed",
            "consumed_cpu_tasks",
            "cpu_nodes",
            "consumed_role_outputs",
        ] {
            totals[field] = 0.into();
        }
        let work = serde_json::json!({"schema_version":1,"search_kind":"pals","go_invocations":0,"successful_returns":0,"failed_returns":0,
            "active_invocations":0,"unobserved_work_invocations":0,"physical_unknown_returns":0,"canceled_returns":0,"deadline_returns":0,
            "pals_resolver":{"version":rz_search::pals::value::MODEL_WDL_RESOLVER_VERSION,
                "semantics_sha256":Sha256::digest(rz_search::pals::value::MODEL_WDL_RESOLVER_SEMANTICS.as_bytes()).to_vec()},"cpu":null,"pals":totals});
        let native = serde_json::json!({"model_epoch":hash_array(&artifact.sha256),"export_manifest_sha256":hash_array(&n.export.sha256),
            "encoding_semantic_sha256":hash_array(&n.encoding_semantic_sha256),"adapter_source_sha256":hash_array(&n.adapter_source_sha256),
            "trained":false,"frozen_epoch":e.model.frozen_epoch,"process_epoch":77,
            "execution":{"provider":"cpu","runtime_sha256":hash_array(&n.runtime.sha256),"device_id":null,"session_arena_bytes":null,"runtime_bundle_sha256":null},
            "physical_shutdown_confirmed":false,"native_buffers_released":false,"quarantined":false,"physical_runs_in_flight":0,
            "game_generation":0,"completed_new_game_resets":0,
            "physically_completed_role_calls":0,"completed_role_inputs":0,"failed_physical_role_calls":0,"invalid_role_outputs":0,
            "delivered_role_inputs":0,"search_consumed_role_inputs":0,"canceled_requests":0,"expired_requests":0,
            "request_high_water":0,"execution_high_water":0,"last_failure":null,
            "residency":{"native_sessions":n.graphs.len(),"role_reader_weights_shared":false,
                "graphs":n.graphs.iter().map(|g|serde_json::json!({"role":g.role,"sha256":hash_array(&g.artifact.sha256),"serialized_bytes":g.artifact.bytes})).collect::<Vec<_>>()}});
        let mut native = native;
        for field in [
            "transient_request_device_bytes",
            "transient_execution_device_bytes",
            "pinned_request_bytes",
        ] {
            native["execution"][field] = 0.into();
        }
        native["execution"]["device_public_memory"] = false.into();
        let start = serde_json::json!({"schema_version":3,"domain":PALS_NATIVE_STARTUP_V3_DOMAIN,"endpoint_id":e.id,"launch_sha256":lock.sha256,
            "process_id":100,"binary_sha256":e.binary.sha256,"runtime_sha256":n.runtime.sha256,"provider":"cpu","precision":"fp32",
            "service_exit_success":false,"cpu_checker":helper,"native":native,"search_work":work});
        let mut end = start.clone();
        end["domain"] = PALS_NATIVE_TERMINATION_V3_DOMAIN.into();
        end["service_exit_success"] = true.into();
        end["native"]["physical_shutdown_confirmed"] = true.into();
        end["native"]["native_buffers_released"] = true.into();
        end["native"]["game_generation"] = 1.into();
        end["native"]["completed_new_game_resets"] = 1.into();
        for field in [
            "physically_completed_role_calls",
            "completed_role_inputs",
            "delivered_role_inputs",
            "search_consumed_role_inputs",
            "request_high_water",
            "execution_high_water",
        ] {
            end["native"][field] = 3.into();
        }
        end["native"]["observer_failures"] = 0.into();
        end["native"]["backend_stats_observation"] = "exclusive_worker_before_shutdown".into();
        end["native"]["backend_stats"] = stats_fixture(1, 3);
        let h = &mut end["cpu_checker"];
        h["cleanup_complete"] = true.into();
        h["started_owner_exit_and_drains_confirmed"] = true.into();
        h["shutdown"] = serde_json::json!({"process_identity":{"pid":101,"process_group":101,"proc_start_ticks":600,"scope":"linux_spawn_observed_identity"},
            "stop_sent":false,"quit_sent":true,"exit_observed":true,"stdout_drained":true,"stderr_drained":true,"exit_code":0,"exit_signal":null,
            "cleanup_complete":true,"quarantined":false,"ownership_lost":false,"stdout_bytes":100,"stderr_bytes":0});
        end["search_work"]["go_invocations"] = 1.into();
        end["search_work"]["successful_returns"] = 1.into();
        for (key, value) in [
            ("external_checker_tasks", 1),
            ("external_checker_reports", 1),
            ("external_checker_nodes_observed", 20),
            ("external_checker_node_budget_reserved", 256),
            ("consumed_external_checker_tasks", 3),
            ("value_calls", 3),
            ("completed_value_calls", 3),
            ("accepted_value_outputs", 3),
            ("consumed_role_outputs", 3),
        ] {
            end["search_work"]["pals"][key] = value.into();
        }
        end["search_work"]["pals"]["external_checker_work_incomplete"] = true.into();
        (lock, loaded, start, end)
    }
    #[cfg(all(
        target_os = "linux",
        any(feature = "pals-collection-onnx", feature = "native-cuda")
    ))]
    #[test]
    fn production_helper_context_joins_profile_supervisor_native_and_foreign_core() {
        use crate::native_runner::{NativeProviderInputContext, NativeProviderInputView};
        use std::os::unix::fs::PermissionsExt;
        let fixture = ExternalProfileFixture::new();
        let (lock, _, start, end) = helper_pair_fixture(&fixture);
        let (declaration, _) = lock.input.external_cpu_r_binding(0).unwrap().unwrap();
        let input_path = fixture.root.join("private-inputs");
        let relative = lock.snapshot_relative_path(&declaration.profile).unwrap();
        let profile_path = input_path.join(&relative);
        std::fs::create_dir_all(profile_path.parent().unwrap()).unwrap();
        std::fs::copy(fixture.root.join("profile.json"), &profile_path).unwrap();
        std::fs::set_permissions(&profile_path, std::fs::Permissions::from_mode(0o444)).unwrap();
        let file = std::fs::File::open(&profile_path).unwrap();
        let input_directory =
            cap_std::fs::Dir::open_ambient_dir(&input_path, cap_std::ambient_authority()).unwrap();
        let pins = [NativeProviderInputView {
            artifact: &declaration.profile,
            path: &profile_path,
            file: &file,
        }];
        let inputs = NativeProviderInputContext {
            directory: &input_directory,
            pins: &pins,
        };
        let trace = synthetic_helper_game_trace(&lock, [[100, 102], [200, 202]]);
        let start_bytes = serde_json::to_vec(&start).unwrap();
        let end_bytes = serde_json::to_vec(&end).unwrap();
        assert!(
            validate_pals_native_records(
                &lock,
                NativeEngineRole::Baseline,
                &start_bytes,
                &end_bytes,
                "native-process-100"
            )
            .is_err()
        );
        assert!(
            validate_pals_search_work_records(
                &lock,
                NativeEngineRole::Baseline,
                &start_bytes,
                &end_bytes,
                "native-process-100"
            )
            .is_err()
        );
        let (audit, pid) = lock
            .validate_records_with_context(
                NativeEngineRole::Baseline,
                &start_bytes,
                &end_bytes,
                "native-process-100",
                &inputs,
                &trace,
            )
            .unwrap();
        assert_eq!(pid, 100);
        assert!(
            lock.validate_records_with_context(
                NativeEngineRole::Candidate,
                &start_bytes,
                &end_bytes,
                "native-process-100",
                &inputs,
                &trace
            )
            .is_err()
        );
        let wrong_trace = synthetic_helper_game_trace(&lock, [[110, 112], [200, 202]]);
        assert!(
            lock.validate_records_with_context(
                NativeEngineRole::Baseline,
                &start_bytes,
                &end_bytes,
                "native-process-100",
                &inputs,
                &wrong_trace
            )
            .is_err()
        );
        let work = validate_pals_search_work_records_inner(
            &lock,
            NativeEngineRole::Baseline,
            &start_bytes,
            &end_bytes,
            "native-process-100",
            Some(&audit),
        )
        .unwrap();
        assert!(
            endpoint_work_receipt(
                &lock,
                NativeEngineRole::Baseline,
                Some(&work),
                None,
                &preflight_fixture("pals")
            )
            .is_err()
        );
        let projected = endpoint_work_receipt(
            &lock,
            NativeEngineRole::Baseline,
            Some(&work),
            Some(&audit),
            &preflight_fixture("pals"),
        )
        .unwrap();
        assert_eq!(
            (
                projected.cpu_tasks_requested,
                projected.cpu_tasks_completed,
                projected.cpu_tasks_consumed,
                projected.cpu_nodes
            ),
            (0, 0, 0, 0)
        );
        assert_eq!(
            (projected.nn_inputs_completed, projected.nn_inputs_consumed),
            (4, 3)
        );
        let foreign = projected.external_cpu_r.unwrap();
        assert_eq!(
            (
                foreign.work.reports_returned,
                foreign.work.nodes_observed,
                foreign.work.consumed_completed_tasks
            ),
            (Some(1), Some(20), Some(3))
        );
        assert!(matches!(
            foreign.work.completed_tasks,
            PalsObservedV3::Unknown
        ));
        let mut changed_work = work.clone();
        changed_work.termination_sha256 = "f".repeat(64);
        assert!(
            endpoint_work_receipt(
                &lock,
                NativeEngineRole::Baseline,
                Some(&changed_work),
                Some(&audit),
                &preflight_fixture("pals")
            )
            .is_err()
        );
        let mut broken = end.clone();
        broken["native"]["residency"]["graphs"][0]["serialized_bytes"] = 999.into();
        assert!(
            lock.validate_records_with_context(
                NativeEngineRole::Baseline,
                &start_bytes,
                &serde_json::to_vec(&broken).unwrap(),
                "native-process-100",
                &inputs,
                &trace
            )
            .is_err()
        ); // whole NN audit remains mandatory
        let leaders = BTreeSet::from([100, 102, 200, 202]);
        assert!(validate_pals_helper_identity_set(std::slice::from_ref(&audit), &leaders).is_ok());
        assert!(
            validate_pals_helper_identity_set(&[audit.clone(), audit.clone()], &leaders).is_err()
        );
        assert!(
            validate_pals_helper_identity_set(
                std::slice::from_ref(&audit),
                &BTreeSet::from([100, 101, 200, 202])
            )
            .is_err()
        );
        // Byte-identical replacement is still a different inode from the owner.
        std::fs::rename(&profile_path, profile_path.with_extension("old")).unwrap();
        std::fs::copy(fixture.root.join("profile.json"), &profile_path).unwrap();
        std::fs::set_permissions(&profile_path, std::fs::Permissions::from_mode(0o444)).unwrap();
        assert!(load_owner_helper_profile(&lock, NativeEngineRole::Baseline, &inputs).is_err());
    }
    #[cfg(all(
        target_os = "linux",
        any(feature = "pals-collection-onnx", feature = "native-cuda")
    ))]
    fn synthetic_helper_game_trace(lock: &LockedPalsArenaLaunchV3, pids: [[u32; 2]; 2]) -> Vec<u8> {
        let trace = |message: String| {
            format!(
                "[TRACE ] [12:34:56.123456] <{:>20}> fastchess --- {message}",
                1
            )
        };
        let mut rows = vec![];
        for (game, (&baseline_pid, &candidate_pid)) in pids[0].iter().zip(&pids[1]).enumerate() {
            let game_pids = [baseline_pid, candidate_pid];
            let number = game + 1;
            let white = &lock.input.semantic_lock.manifest.pilot.white_order[game];
            let wr = lock
                .input
                .semantic_lock
                .manifest
                .engines
                .iter()
                .position(|e| e.id() == white)
                .unwrap();
            let black = lock.input.semantic_lock.manifest.engines[1 - wr].id();
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
                    game_pids[wr]
                )),
                trace(format!(
                    "Process with pid: {} terminated with status: 0",
                    game_pids[1 - wr]
                )),
            ]);
        }
        format!("{}\n", rows.join("\n")).into_bytes()
    }
    #[cfg(all(
        target_os = "linux",
        any(feature = "pals-collection-onnx", feature = "native-cuda")
    ))]
    fn synthetic_helper_reidentify(
        start: &mut serde_json::Value,
        end: &mut serde_json::Value,
        parent: u32,
        helper: u32,
    ) {
        for value in [&mut *start, &mut *end] {
            value["process_id"] = parent.into();
            let ready = &mut value["cpu_checker"]["startup_resource_observation"];
            ready["parent"]["pid"] = parent.into();
            ready["parent"]["process_group"] = parent.into();
            ready["parent"]["threads"][0]["tid"] = parent.into();
            ready["helper"]["pid"] = helper.into();
            ready["helper"]["process_group"] = helper.into();
            ready["helper"]["parent_pid"] = parent.into();
            ready["helper"]["threads"][0]["tid"] = helper.into();
        }
        end["cpu_checker"]["shutdown"]["process_identity"]["pid"] = helper.into();
        end["cpu_checker"]["shutdown"]["process_identity"]["process_group"] = helper.into();
    }
    #[cfg(all(
        target_os = "linux",
        any(feature = "pals-collection-onnx", feature = "native-cuda")
    ))]
    #[test]
    fn helper_preflight_consumes_exact_supervised_slots_and_keeps_game_root_separate() {
        use crate::native_runner::{NativeProviderInputContext, NativeProviderInputView};
        use std::os::unix::fs::PermissionsExt;
        let fixture = ExternalProfileFixture::new();
        let (lock, _, start, end) = helper_pair_fixture(&fixture);
        let (declaration, _) = lock.input.external_cpu_r_binding(0).unwrap().unwrap();
        let input_path = fixture.root.join("private-inputs");
        let profile_path =
            input_path.join(lock.snapshot_relative_path(&declaration.profile).unwrap());
        std::fs::create_dir_all(profile_path.parent().unwrap()).unwrap();
        std::fs::copy(fixture.root.join("profile.json"), &profile_path).unwrap();
        std::fs::set_permissions(&profile_path, std::fs::Permissions::from_mode(0o444)).unwrap();
        let file = std::fs::File::open(&profile_path).unwrap();
        let directory =
            cap_std::fs::Dir::open_ambient_dir(&input_path, cap_std::ambient_authority()).unwrap();
        let pins = [NativeProviderInputView {
            artifact: &declaration.profile,
            path: &profile_path,
            file: &file,
        }];
        let inputs = NativeProviderInputContext {
            directory: &directory,
            pins: &pins,
        };
        let root_path = fixture.root.join("external-baseline-preflight-runtime");
        std::fs::create_dir(&root_path).unwrap();
        let root =
            cap_std::fs::Dir::open_ambient_dir(&root_path, cap_std::ambient_authority()).unwrap();
        let mut preflight = preflight_fixture("pals");
        preflight.identification_process.pid = 10;
        preflight.readiness_process.pid = 12;
        for (parent, helper, identification) in [(10, 11, true), (12, 13, false)] {
            let (mut s, mut t) = (start.clone(), end.clone());
            for value in [&mut s, &mut t] {
                value["process_id"] = parent.into();
                let ready = &mut value["cpu_checker"]["startup_resource_observation"];
                ready["parent"]["pid"] = parent.into();
                ready["parent"]["process_group"] = parent.into();
                ready["parent"]["threads"][0]["tid"] = parent.into();
                ready["helper"]["pid"] = helper.into();
                ready["helper"]["process_group"] = helper.into();
                ready["helper"]["parent_pid"] = parent.into();
                ready["helper"]["threads"][0]["tid"] = helper.into();
            }
            t["cpu_checker"]["shutdown"]["process_identity"]["pid"] = helper.into();
            t["cpu_checker"]["shutdown"]["process_identity"]["process_group"] = helper.into();
            if identification {
                t["search_work"] = s["search_work"].clone();
                for field in [
                    "physically_completed_role_calls",
                    "completed_role_inputs",
                    "delivered_role_inputs",
                    "search_consumed_role_inputs",
                    "request_high_water",
                    "execution_high_water",
                    "completed_new_game_resets",
                    "game_generation",
                ] {
                    t["native"][field] = 0.into();
                }
                t["native"]["backend_stats"] = stats_fixture(0, 0);
            }
            let path = root_path.join(format!("native-process-{parent}"));
            std::fs::create_dir(&path).unwrap();
            std::fs::write(
                path.join("pals-native-startup.v3.json"),
                serde_json::to_vec(&s).unwrap(),
            )
            .unwrap();
            std::fs::write(
                path.join("pals-native-termination.v3.json"),
                serde_json::to_vec(&t).unwrap(),
            )
            .unwrap();
        }
        let evidence = lock
            .validate_external_preflight(NativeEngineRole::Baseline, &preflight, &root, &inputs)
            .unwrap();
        assert_eq!(evidence.len(), 4);
        assert!(!fixture.root.join("baseline-runtime").exists());
        let trace = synthetic_helper_game_trace(&lock, [[100, 102], [200, 202]]);
        let mut games = Vec::new();
        for (parent, helper) in [(100, 101), (102, 103)] {
            let (mut s, mut t) = (start.clone(), end.clone());
            synthetic_helper_reidentify(&mut s, &mut t, parent, helper);
            games.push(
                lock.validate_records_with_context(
                    NativeEngineRole::Baseline,
                    &serde_json::to_vec(&s).unwrap(),
                    &serde_json::to_vec(&t).unwrap(),
                    &format!("native-process-{parent}"),
                    &inputs,
                    &trace,
                )
                .unwrap()
                .0,
            );
        }
        let artifacts: Vec<ArtifactRef> = evidence
            .iter()
            .map(|(relative, bytes)| ArtifactRef {
                path: format!("attempt/external-baseline-preflight-runtime/{relative}"),
                sha256: digest(bytes),
                bytes: bytes.len() as u64,
                source: "synthetic retained producer evidence".into(),
                license: "MIT".into(),
            })
            .collect();
        let runtime_root =
            cap_std::fs::Dir::open_ambient_dir(&fixture.root, cap_std::ambient_authority())
                .unwrap();
        lock.validate_provider_session_set(
            &games,
            std::slice::from_ref(&preflight),
            &runtime_root,
            &inputs,
            &trace,
            &artifacts,
        )
        .unwrap();
        let mut changed_history = artifacts.clone();
        changed_history[0].sha256 = "f".repeat(64);
        assert!(
            lock.validate_provider_session_set(
                &games,
                std::slice::from_ref(&preflight),
                &runtime_root,
                &inputs,
                &trace,
                &changed_history,
            )
            .is_err()
        );
        assert!(
            lock.validate_provider_session_set(
                &games[..1],
                std::slice::from_ref(&preflight),
                &runtime_root,
                &inputs,
                &trace,
                &artifacts,
            )
            .is_err()
        );
        let mut wrong = preflight.clone();
        wrong.readiness_process.pid = 14;
        assert!(
            lock.validate_external_preflight(NativeEngineRole::Baseline, &wrong, &root, &inputs)
                .is_err()
        );
        let mut failed = preflight.clone();
        failed.identification_process.exit_code = Some(1);
        assert!(
            lock.validate_external_preflight(NativeEngineRole::Baseline, &failed, &root, &inputs)
                .is_err()
        );
        let absent = NativeProviderInputContext {
            directory: &directory,
            pins: &[],
        };
        assert!(
            lock.validate_external_preflight(
                NativeEngineRole::Baseline,
                &preflight,
                &root,
                &absent
            )
            .is_err()
        );
        std::fs::create_dir(root_path.join("native-process-99")).unwrap();
        assert!(
            lock.validate_external_preflight(
                NativeEngineRole::Baseline,
                &preflight,
                &root,
                &inputs
            )
            .is_err()
        );
    }
    #[cfg(all(
        target_os = "linux",
        any(feature = "pals-collection-onnx", feature = "native-cuda")
    ))]
    #[test]
    fn external_helper_session_validates_registered_model_resources_and_unknown_work() {
        let fixture = ExternalProfileFixture::new();
        let (lock, loaded, start, end) = helper_pair_fixture(&fixture);
        let audit = validate_pals_external_cpu_r_session(
            &lock,
            NativeEngineRole::Baseline,
            &loaded,
            &serde_json::to_vec(&start).unwrap(),
            &serde_json::to_vec(&end).unwrap(),
            100,
            PalsExternalCpuRSessionPurposeV3::Game,
        )
        .unwrap();
        assert_eq!(audit.receipt.work.reports_returned, Some(1));
        assert_eq!(audit.receipt.work.consumed_completed_tasks, Some(3));
        assert_eq!(audit.receipt.work.work_incomplete, Some(true));
        assert!(matches!(
            audit.receipt.work.completed_tasks,
            PalsObservedV3::Unknown
        ));
        assert!(matches!(
            audit.receipt.work.reused_completed_task_consumptions,
            PalsObservedV3::Unknown
        ));
        assert!(matches!(
            audit.receipt.applied_option_values,
            PalsObservedV3::Unknown
        ));
        assert_eq!(
            audit
                .receipt
                .shutdown
                .process_identity
                .as_ref()
                .unwrap()
                .pid,
            101
        );
        assert!(lock.validate_execution().is_ok());
    }
    #[cfg(all(
        target_os = "linux",
        any(feature = "pals-collection-onnx", feature = "native-cuda")
    ))]
    #[test]
    fn external_helper_session_rejects_identity_resource_and_closure_fabrication() {
        let fixture = ExternalProfileFixture::new();
        let (lock, loaded, start, end) = helper_pair_fixture(&fixture);
        let mutations: &[fn(&mut serde_json::Value)] = &[
            |v| {
                v["cpu_checker"]["registration"]["identity"]["identity"]["options"]["Threads"] =
                    "1".into()
            },
            |v| v["cpu_checker"]["registration"]["model_value"]["model_epoch"][0] = 0.into(),
            |v| v["native"]["model_epoch"][0] = 0.into(),
            |v| v["search_work"]["pals_resolver"]["version"] = "own".into(),
            |v| {
                v["cpu_checker"]["startup_resource_observation"]["helper"]["cgroup_v2_membership"] =
                    "/other".into()
            },
            |v| v["cpu_checker"]["shutdown"]["process_identity"]["proc_start_ticks"] = 601.into(),
            |v| v["cpu_checker"]["shutdown"]["stderr_drained"] = false.into(),
            |v| v["cpu_checker"]["shutdown"]["exit_code"] = 1.into(),
            |v| v["cpu_checker"]["shutdown"]["ownership_lost"] = true.into(),
            |v| v["native"]["physical_shutdown_confirmed"] = false.into(),
            |v| v["native"]["execution"]["private_warm"] = serde_json::json!({}),
            |v| v["native"]["private_warm_observation"] = serde_json::json!({}),
            |v| v["native"]["private_warm_observation_unavailable"] = false.into(),
            |v| v["search_work"]["pals"]["cpu_nodes"] = 20.into(),
        ];
        for (i, mutate) in mutations.iter().enumerate() {
            let mut bad = end.clone();
            mutate(&mut bad);
            assert!(
                validate_pals_external_cpu_r_session(
                    &lock,
                    NativeEngineRole::Baseline,
                    &loaded,
                    &serde_json::to_vec(&start).unwrap(),
                    &serde_json::to_vec(&bad).unwrap(),
                    100,
                    PalsExternalCpuRSessionPurposeV3::Readiness
                )
                .is_err(),
                "mutation {i}"
            );
        }
        assert!(
            unique_helper_json(br#"{"identity":{"options":{"Threads":"1","Threads":"2"}}}"#)
                .is_err()
        );
    }
    #[cfg(all(
        target_os = "linux",
        any(feature = "pals-collection-onnx", feature = "native-cuda")
    ))]
    #[test]
    fn actual_external_profile_verification_is_separate_from_execution_observation() {
        let f = ExternalProfileFixture::new();
        let original = std::fs::read(f.root.join("profile.json")).unwrap();
        let audit = f.verify().unwrap();
        assert!(audit.profile_file_verification_performed);
        assert!(!audit.binary_file_verification_performed);
        assert!(!audit.execution_admission_completed);
        assert!(!audit.child_spawned);
        assert_eq!(audit.options_application_observed, None);
        assert_eq!(audit.model_loading_observed, None);
        assert_eq!(audit.inherited_resource_join_observed, None);
        assert_eq!(audit.physical_shutdown_observed, None);
        assert_eq!(audit.profile_file_bytes, original.len() as u64);
        assert_eq!(audit.profile_file_sha256, digest(&original));
        assert_eq!(audit.registered_program, f.program.to_str().unwrap());
        assert_eq!(audit.profile_registration["process_start_performed"], false);
        assert!(audit.profile_registration["model_loading_observed"].is_null());
        assert!(audit.profile_registration["options_application_observed"].is_null());
        assert_eq!(
            std::fs::read(f.root.join("profile.json")).unwrap(),
            original
        );
        assert!(!f.program.exists() && !f.cwd.exists());
    }
    #[cfg(all(
        target_os = "linux",
        any(feature = "pals-collection-onnx", feature = "native-cuda")
    ))]
    #[test]
    fn actual_external_profile_must_match_independent_lock_paths_and_caps() {
        let changes: &[fn(&mut PalsExternalCpuRV3)] = &[
            |c| c.profile.sha256 = "f".repeat(64),
            |c| c.profile.bytes += 1,
            |c| c.profile_canonical_sha256 = "f".repeat(64),
            |c| c.binary.sha256 = "f".repeat(64),
            |c| c.binary.source = "https://github.com/official-stockfish/Stockfish/tree/pin".into(),
            |c| c.resolver.semantics_sha256 = "f".repeat(64),
            |c| c.policy.threads_max = 1,
            |c| c.policy.hash_mib_max = 8,
            |c| c.policy.max_depth = 4,
            |c| c.policy.max_prefix_plies = 32,
            |c| c.policy.handshake_max_ms = 500,
            |c| c.policy.task_wall_time_max_ms = 500,
            |c| c.policy.stop_grace_max_ms = 50,
            |c| c.policy.shutdown_grace_max_ms = 50,
            |c| c.policy.lifetime_output_bytes_max = 2048,
            |c| c.policy.line_bytes_max = 512,
        ];
        for (index, change) in changes.iter().enumerate() {
            let mut f = ExternalProfileFixture::new();
            f.mutate_declaration(*change);
            assert!(
                f.verify().is_err(),
                "accepted actual profile mismatch {index}"
            );
        }
        let mut f = ExternalProfileFixture::new();
        f.program = f.program.with_file_name("different-program");
        assert!(f.verify().is_err());
        let mut f = ExternalProfileFixture::new();
        f.cwd = f.root.join("different-cwd");
        assert!(f.verify().is_err());
        let mut f = ExternalProfileFixture::new();
        let PalsEngineV3::Pals(e) = &mut f.lock.manifest.engines[0] else {
            unreachable!()
        };
        e.cpu_r = PalsCpuRSelectionV3::Own;
        f.lock = f.lock.manifest.lock().unwrap();
        assert!(
            f.verify()
                .unwrap_err()
                .to_string()
                .contains("own CPU_R selection")
        );
    }
    #[cfg(not(all(
        target_os = "linux",
        any(feature = "pals-collection-onnx", feature = "native-cuda")
    )))]
    #[test]
    fn external_profile_verification_has_no_feature_or_platform_fallback() {
        let lock = fixture().semantic_lock;
        let error = verify_pals_external_cpu_r_profile(
            &lock,
            NativeEngineRole::Baseline,
            Path::new("/missing-source"),
            Path::new("/missing-program"),
            Path::new("/missing-cwd"),
        )
        .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("unsupported external CPU_R profile verification")
        );
    }
    pub(super) fn native_fixture() -> PalsArenaLaunchV3 {
        let mut f = fixture();
        let PalsEngineV3::Pals(e) = &mut f.semantic_lock.manifest.engines[0] else {
            unreachable!()
        };
        e.model.architecture = "pals-width384-latent16-iterations2".into();
        e.model.backend = PalsModelBackendV3::OrtCpu;
        e.model.frozen_epoch = 1;
        e.model.weights = PalsWeightIdentityV3::Untrained {
            artifact: asset("model/checkpoint.pt"),
            initialization_seed: 1,
        };
        f.semantic_lock = f.semantic_lock.manifest.lock().unwrap();
        f.endpoints[0] = PalsEndpointLaunchV3::OnnxCpu(PalsOnnxLaunchV3 {
            export: asset("model/export.json"),
            export_file: "export.json".into(),
            graphs: ["public", "proposer", "critic"]
                .into_iter()
                .map(|r| PalsGraphAssetV3 {
                    file: format!("{r}.onnx"),
                    role: r.into(),
                    artifact: asset(&format!("model/{r}.onnx")),
                })
                .collect(),
            runtime: asset("runtime/libonnxruntime.so"),
            encoding_semantic_sha256: "d".repeat(64),
            adapter_source_sha256: "b".repeat(64),
            search: PalsSearchLaunchV3 {
                max_rounds: 16,
                max_cpu_nodes: 100_000,
                cpu_depth: 2,
                max_situations: 4096,
                post_repair_recheck: None,
            },
            external_cpu_r: None,
        });
        f
    }
    #[test]
    fn own_native_launch_omits_external_binding_and_rejects_unregistered_paths() {
        let input = native_fixture();
        let text = serde_json::to_string(&input).unwrap();
        let legacy: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert!(
            legacy["endpoints"][0]["configuration"]
                .get("external_cpu_r")
                .is_none()
        );
        let decoded = PalsArenaLaunchV3::from_json(&text).unwrap();
        assert_eq!(
            crate::canonical_sha256(&decoded).unwrap(),
            crate::canonical_sha256(&input).unwrap()
        );
        let valid = PalsExternalCpuRLaunchV3 {
            registered_program: "/registered-cas/stockfish".into(),
            registered_working_directory: "/registered-cas".into(),
        };
        assert!(valid.validate().is_ok());
        for path in [
            "relative/stockfish",
            "/registered-cas/../stockfish",
            "/registered-cas/./stockfish",
            "/registered-cas//stockfish",
            "/registered-cas/.env",
            "/registered-cas/id_ed25519",
            "/registered-cas/stockfish\n",
            "C:\\registered-cas\\stockfish",
        ] {
            let mut bad = valid.clone();
            bad.registered_program = path.into();
            assert!(bad.validate().is_err());
        }
        let mut mismatched = valid.clone();
        mismatched.registered_working_directory = "/other-cas".into();
        assert!(mismatched.validate().is_err());
    }
    fn cuda_fixture() -> (PalsArenaLaunchV3, Vec<u8>) {
        let mut f = native_fixture();
        let PalsEndpointLaunchV3::OnnxCpu(mut model) = f.endpoints[0].clone() else {
            unreachable!()
        };
        model.graphs.retain(|g| g.role == "public");
        model.graphs.push(PalsGraphAssetV3 {
            file: "shared_pc_if.onnx".into(),
            role: "shared_pc".into(),
            artifact: asset("model/shared_pc_if.onnx"),
        });
        model.runtime = asset("runtime/libonnxruntime.so.1.22.0");
        let files: Vec<_> = CUDA_BUNDLE_FILENAMES
            .iter()
            .map(|(filename, role)| CudaBundleFileBindingV1 {
                filename: (*filename).into(),
                role: *role,
                artifact: asset(&format!("runtime/{filename}")),
            })
            .collect();
        let descriptor = serde_json::to_vec(&serde_json::json!({"schema_version":1,
            "files":files.iter().map(|f|serde_json::json!({"role":f.role,"filename":f.filename,
                "bytes":f.artifact.bytes,"sha256":f.artifact.sha256})).collect::<Vec<_>>()}))
        .unwrap();
        let mut bundle = CudaBundleBindingV1 {
            schema_version: None,
            loader_aliases: None,
            manifest: asset("runtime/bundle.json"),
            canonical_sha256: "0".repeat(64),
            files,
        };
        bundle.manifest.sha256 = digest(&descriptor);
        bundle.manifest.bytes = descriptor.len() as u64;
        bundle.canonical_sha256 = bundle.canonical_digest().unwrap();
        let PalsEngineV3::Pals(e) = &mut f.semantic_lock.manifest.engines[0] else {
            unreachable!()
        };
        e.model.backend = PalsModelBackendV3::OrtCuda;
        e.pools.device_bytes = 6 << 30;
        f.semantic_lock.manifest.resources[0].requested_gpu = Some("RTX 4050 6GB".into());
        f.semantic_lock.manifest.resources[0].device_allocation_max_bytes = 6 << 30;
        f.semantic_lock = f.semantic_lock.manifest.lock().unwrap();
        f.endpoints[0] = PalsEndpointLaunchV3::OnnxCuda(Box::new(PalsOnnxCudaLaunchV3 {
            model,
            cuda_bundle: bundle,
            device_id: 0,
            session_arena_bytes: 2 << 30,
            cuda_control: None,
            startup_probe_timeout_ms: None,
            cuda_record_pages: None,
        }));
        f.budget.max_runtime_bytes = 4 << 30;
        f.budget.max_artifact_bytes = 8 << 30;
        (f, descriptor)
    }
    fn stats_fixture(public: u64, role: u64) -> serde_json::Value {
        serde_json::json!({"public_nn_runs_completed":public,"role_nn_runs_completed":role,
            "completed_nn_inputs":public+role,"public_nn_runs_attempted":public,"role_nn_runs_attempted":role,
            "public_nn_runs_failed_known":0,"role_nn_runs_failed_known":0,"validated_public_outputs":public,
            "validated_role_outputs":role,"live_public_cache_entries":0,"new_game_resets":1})
    }
    pub(super) fn cuda_control_fixture() -> (PalsArenaLaunchV3, Vec<u8>) {
        let (mut f, _) = cuda_fixture();
        let cuda = f.endpoints[0].cuda_model().unwrap();
        // Static metadata/receipt linkage fixture, never native CUDA evidence.
        let bytes=serde_json::to_vec(&serde_json::json!({
            "schema":"rovezero.pals-static-control-inventory.v2",
            "scope":"read_only_serialized_graph_metadata_no_runtime",
            "manifest_sha256":cuda.model.export.sha256,
            "actual_provider_placement":"not_observed","cpu_allowlist":"not_created","total_nodes":6,
            "graphs":cuda.model.graphs.iter().map(|g|serde_json::json!({"role":g.role,"sha256":g.artifact.sha256})).collect::<Vec<_>>()
        })).unwrap();
        let mut inventory = asset("research/control-inventory.v2.json");
        inventory.sha256 = digest(&bytes);
        inventory.bytes = bytes.len() as u64;
        let PalsEndpointLaunchV3::OnnxCuda(cuda) = &mut f.endpoints[0] else {
            unreachable!()
        };
        cuda.cuda_control = Some(Box::new(PalsCudaControlBindingV3 {
            inventory,
            loading_profile: None,
            cudnn_conv_search: None,
        }));
        (f, bytes)
    }
    fn attach_control_witness(
        lock: &LockedPalsArenaLaunchV3,
        start: &mut serde_json::Value,
        end: &mut serde_json::Value,
    ) {
        let cuda = lock.input.endpoints[0].cuda_model().unwrap();
        let inventory = &cuda.cuda_control.as_ref().unwrap().inventory;
        let kernels = serde_json::json!({"profile_sha256":hash_array(&"d".repeat(64)),"cuda_kernels":3,"cuda_transfer_kernels":0,
            "approved_cpu_control_kernels":1,"neural_kernels":3,"proposer_private_kernels":1,"critic_private_kernels":1});
        let witness = serde_json::json!({"schema":"rovezero.pals-cuda-metadata-control.v2",
            "optimization":"disable",
            "category_provenance":"rc10-category-unavailable-id-location-message-used",
            "inventory_sha256":hash_array(&inventory.sha256),"manifest_sha256":hash_array(&cuda.model.export.sha256),
            "runtime_sha256":hash_array(&cuda.model.runtime.sha256),"runtime_bundle_sha256":hash_array(&cuda.cuda_bundle.canonical_sha256),
            "initialization":cuda.model.graphs.iter().map(|g|serde_json::json!({"role":g.role,"graph_sha256":hash_array(&g.artifact.sha256),
                "log_sha256":hash_array(&"e".repeat(64)),"assigned_nodes":3,"cuda_nodes":2,"approved_cpu_control_nodes":1,
                "optimization":"disable","approved_transfers":[],
                "recursive_coverage":"ort-1.22-finalize-recursive-exact-named-provider-coverage-before-first-run"})).collect::<Vec<_>>(),
            "public":kernels,"shared_pc":kernels});
        for record in [start, end] {
            record["native"]["execution"]["cuda_control_inventory_sha256"] =
                hash_array(&inventory.sha256).into();
            record["native"]["startup_probe"]["cuda_placement_witness"] = witness.clone();
        }
    }
    fn loading_binding_fixture() -> PalsCudaLoadingBindingV1 {
        let profile = PalsCudaLoadingProfileV1::ExperimentalCudnnShimLazyV1;
        PalsCudaLoadingBindingV1 {
            profile,
            canonical_sha256: profile.canonical_sha256(),
        }
    }
    fn cudnn_conv_fixture() -> (PalsOnnxCudaLaunchV3, PalsCuDnnConvBindingV1) {
        let (f, _) = cuda_control_fixture();
        let mut cuda = f.endpoints[0].cuda_model().unwrap().clone();
        let mut binding = PalsCuDnnConvBindingV1 {
            profile: PalsCuDnnConvProfileV1::DefaultHostControlV1,
            backend_source_sha256: digest(include_bytes!("../../rz-eval/src/pals_onnx.rs")),
            descriptor_sha256: "0".repeat(64),
        };
        binding.descriptor_sha256 =
            digest(&serde_json::to_vec(&binding.descriptor(&cuda, &"a".repeat(64))).unwrap());
        cuda.cuda_control.as_mut().unwrap().cudnn_conv_search = Some(binding.clone());
        (cuda, binding)
    }
    #[test]
    fn pals_cudnn_conv_absence_preserves_codec_digest_and_argv() {
        let (old, _) = cuda_control_fixture();
        let locked = old.lock().unwrap();
        let encoded = serde_json::to_string(&old).unwrap();
        assert!(!encoded.contains("cudnn_conv_search"));
        let roundtrip = PalsArenaLaunchV3::from_json(&encoded)
            .unwrap()
            .lock()
            .unwrap();
        assert_eq!(locked.sha256(), roundtrip.sha256());
        assert_eq!(
            locked.endpoint_views[0].arguments,
            roundtrip.endpoint_views[0].arguments
        );
        let mut bad = json(encoded.as_bytes()).unwrap();
        bad["endpoints"][0]["configuration"]["cuda_control"]["cudnn_conv_search"] =
            serde_json::Value::Null;
        assert!(PalsArenaLaunchV3::from_json(&bad.to_string()).is_err());
        bad["endpoints"][0]["configuration"]["cuda_control"]["cudnn_conv_search"] = serde_json::json!({
            "profile":"default-device-warm-v1","backend_source_sha256":"a".repeat(64),"descriptor_sha256":"b".repeat(64)});
        assert!(PalsArenaLaunchV3::from_json(&bad.to_string()).is_err());
    }
    #[test]
    fn pals_cudnn_conv_selection_requires_v4_eager_host_pair_and_exact_pins() {
        let (cuda, binding) = cudnn_conv_fixture();
        binding.validate(&cuda, &"a".repeat(64)).unwrap();
        let (mut old, _) = cuda_control_fixture();
        old.endpoints[0] = PalsEndpointLaunchV3::OnnxCuda(Box::new(cuda.clone()));
        assert!(old.lock().is_err());
        assert!(old.endpoint(0).is_err());
        assert!(validate_cudnn_conv_pair(&old.endpoints).is_err());
        old.endpoints[1] = old.endpoints[0].clone();
        validate_cudnn_conv_pair(&old.endpoints).unwrap();
        let mut mixed = cuda.clone();
        mixed.cuda_control.as_mut().unwrap().loading_profile = Some(loading_binding_fixture());
        assert!(binding.validate(&mixed, &"a".repeat(64)).is_err());
        let mut bad = binding.clone();
        bad.descriptor_sha256 = "f".repeat(64);
        assert!(bad.validate(&cuda, &"a".repeat(64)).is_err());
        bad = binding;
        bad.backend_source_sha256 = "f".repeat(64);
        assert!(bad.validate(&cuda, &"a".repeat(64)).is_err());
    }
    #[test]
    fn pals_cudnn_conv_loaded_descriptor_rejects_tamper_and_undeclared_modes() {
        let (cuda, binding) = cudnn_conv_fixture();
        let checkpoint = "a".repeat(64);
        let descriptor = binding.descriptor(&cuda, &checkpoint);
        let actual = serde_json::json!({"descriptor":descriptor,"descriptor_sha256":hash_array(&binding.descriptor_sha256),
            "descriptor_encoding":"serde_json_utf8_fixed_descriptor_object","actual_cudnn_plan":"unknown",
            "numerical_note":"unknown","tf32_actual":"unknown","tf32_retry":"unknown",
            "production_default_changed":false,"warm_success_claim":false,"pilot_success_claim":false});
        let record = serde_json::json!({"execution":{"device_public_memory":false,"cudnn_conv_search_diagnostic":actual}});
        validate_native_cudnn_conv(Some(&cuda), 4, &checkpoint, &record).unwrap();
        validate_native_cudnn_conv_pair(Some(&cuda), 4, &checkpoint, &record, &record.clone())
            .unwrap();
        let mut changed_final = record.clone();
        let first = changed_final["execution"]["cudnn_conv_search_diagnostic"]["descriptor_sha256"]
            [0]
        .as_u64()
        .unwrap();
        changed_final["execution"]["cudnn_conv_search_diagnostic"]["descriptor_sha256"][0] =
            (first ^ 1).into();
        assert!(
            validate_native_cudnn_conv_pair(Some(&cuda), 4, &checkpoint, &record, &changed_final)
                .is_err()
        );
        let mut missing_final = record.clone();
        missing_final["execution"]["cudnn_conv_search_diagnostic"] = serde_json::Value::Null;
        assert!(
            validate_native_cudnn_conv_pair(Some(&cuda), 4, &checkpoint, &record, &missing_final)
                .is_err()
        );
        assert!(validate_native_cudnn_conv(Some(&cuda), 3, &checkpoint, &record).is_err());
        assert!(validate_native_cudnn_conv(None, 4, &checkpoint, &record).is_err());
        let mut legacy = cuda.clone();
        legacy.cuda_control.as_mut().unwrap().cudnn_conv_search = None;
        assert!(validate_native_cudnn_conv(Some(&legacy), 4, &checkpoint, &record).is_err());
        for field in [
            "schema",
            "selected_profile",
            "backend_source_sha256",
            "runtime_sha256",
            "runtime_bundle_sha256",
            "control_inventory_sha256",
            "manifest_sha256",
            "model_epoch",
            "cache_public_memory",
            "public_memory",
            "native_loading_profile",
            "graph_optimization",
            "device_id",
            "session_arena_bytes",
            "intra_threads",
            "inter_threads",
            "parallel_execution",
            "tf32_requested",
            "cuda_graph_requested",
            "nvidia_tf32_override_environment_request",
        ] {
            let mut bad = record.clone();
            bad["execution"]["cudnn_conv_search_diagnostic"]["descriptor"][field] =
                serde_json::Value::Null;
            assert!(
                validate_native_cudnn_conv(Some(&cuda), 4, &checkpoint, &bad).is_err(),
                "{field}"
            );
        }
        for field in [
            "production_default_changed",
            "warm_success_claim",
            "pilot_success_claim",
        ] {
            let mut bad = record.clone();
            bad["execution"]["cudnn_conv_search_diagnostic"][field] = true.into();
            assert!(validate_native_cudnn_conv(Some(&cuda), 4, &checkpoint, &bad).is_err());
        }
        let mut bad = record;
        bad["execution"]["device_public_memory"] = true.into();
        assert!(validate_native_cudnn_conv(Some(&cuda), 4, &checkpoint, &bad).is_err());
    }
    fn mapping_fixture(
        cuda: &PalsOnnxCudaLaunchV3,
        loading: &PalsCudaLoadingBindingV1,
        additional: &[usize],
    ) -> serde_json::Value {
        let eager = loading.profile.native().eager_indices();
        let mapped: Vec<_> = rz_native_loader::NVIDIA_LOAD_ORDER
            .iter()
            .enumerate()
            .filter(|(i, _)| eager.contains(i) || additional.contains(i))
            .map(|(_, n)| *n)
            .collect();
        let absent: Vec<_> = rz_native_loader::NVIDIA_LOAD_ORDER
            .iter()
            .filter(|n| !mapped.contains(n))
            .copied()
            .collect();
        serde_json::json!({"schema":"rovezero.pals-native-mapping-witness.v1",
            "runtime_sha256":hash_array(&cuda.model.runtime.sha256),"runtime_bundle_sha256":hash_array(&cuda.cuda_bundle.canonical_sha256),
            "loading_profile":loading.profile.native().identifier(),"loading_profile_sha256":hash_array(&loading.canonical_sha256),
            "scope":"exclusive_physical_worker_full_runtime_origin","declared_nvidia_files":16,
            "required_nvidia_files":eager.iter().map(|&i|rz_native_loader::NVIDIA_LOAD_ORDER[i]).collect::<Vec<_>>(),
            "mapped_nvidia_files":mapped,"deferred_nvidia_not_mapped":absent,"mapped_ort_files":rz_native_loader::ORT_LIBRARY_NAMES})
    }
    fn hash_array(hex: &str) -> Vec<u8> {
        (0..hex.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap())
            .collect()
    }
    fn cuda_records_fixture(
        lock: &LockedPalsArenaLaunchV3,
    ) -> (serde_json::Value, serde_json::Value) {
        let cuda = lock.input.endpoints[0].cuda_model().unwrap();
        let model = &cuda.model;
        let native = serde_json::json!({"physically_completed_role_calls":0,"completed_role_inputs":0,
            "failed_physical_role_calls":0,"invalid_role_outputs":0,"delivered_role_inputs":0,"search_consumed_role_inputs":0,
            "canceled_requests":0,"expired_requests":0,"completed_new_game_resets":0,"process_epoch":1,"game_generation":0,
            "request_high_water":0,"execution_high_water":0,"physical_runs_in_flight":0,"quarantined":false,
            "physical_shutdown_confirmed":false,"native_buffers_released":false,"last_failure":null,
            "model_epoch":hash_array(&"a".repeat(64)),"export_manifest_sha256":hash_array(&model.export.sha256),
            "encoding_semantic_sha256":hash_array(&model.encoding_semantic_sha256),"adapter_source_sha256":hash_array(&model.adapter_source_sha256),
            "trained":false,"frozen_epoch":1,"residency":{"native_sessions":model.graphs.len(),"layout":"shared_pc_if",
                "role_reader_weights_shared":null,"native_resident_parameter_bytes":null,"vram_peak_bytes":null,
                "graphs":model.graphs.iter().map(|g|serde_json::json!({"role":g.role,"sha256":hash_array(&g.artifact.sha256),
                    "serialized_bytes":g.artifact.bytes})).collect::<Vec<_>>()},
            "execution":{"provider":"cuda","device_id":0,"session_arena_bytes":2u64<<30,
                "runtime_sha256":hash_array(&model.runtime.sha256),"runtime_bundle_sha256":hash_array(&cuda.cuda_bundle.canonical_sha256),
                "transient_request_device_bytes":8192,"transient_execution_device_bytes":8192,"pinned_request_bytes":0,"device_public_memory":false},
            "startup_probe":{"completed_proposer_calls":1,"completed_critic_calls":1,"runtime_mapping_confirmed":true,
                "reset_completed":true,"backend_stats":stats_fixture(1,2)}});
        let start = serde_json::json!({"schema_version":3,"domain":PALS_NATIVE_STARTUP_V3_DOMAIN,
            "endpoint_id":"pals","launch_sha256":lock.sha256(),"process_id":100,"binary_sha256":"a".repeat(64),
            "runtime_sha256":model.runtime.sha256,"runtime_bundle_sha256":cuda.cuda_bundle.canonical_sha256,
            "provider":"cuda","precision":"fp32","service_exit_success":false,"native":native});
        let mut end = start.clone();
        end["domain"] = PALS_NATIVE_TERMINATION_V3_DOMAIN.into();
        end["service_exit_success"] = true.into();
        for (key, value) in [
            ("physically_completed_role_calls", 3u64),
            ("completed_role_inputs", 3),
            ("delivered_role_inputs", 2),
            ("search_consumed_role_inputs", 1),
            ("completed_new_game_resets", 1),
            ("game_generation", 1),
            ("request_high_water", 5),
            ("execution_high_water", 3),
        ] {
            end["native"][key] = value.into();
        }
        end["native"]["physical_shutdown_confirmed"] = true.into();
        end["native"]["native_buffers_released"] = true.into();
        end["native"]["final_runtime_mapping_confirmed"] = true.into();
        end["native"]["observer_failures"] = 0.into();
        end["native"]["last_observer_failure"] = serde_json::Value::Null;
        end["native"]["backend_stats_observation"] = "exclusive_worker_before_shutdown".into();
        end["native"]["backend_stats"] = stats_fixture(2, 5);
        (start, end)
    }
    // These are CPU wire fixtures. Synthetic receipts exercise the acceptance
    // boundary; they do not attest an actual CUDA owner, Run, fence or peak.
    fn cuda_record_pages_fixture() -> PalsArenaLaunchV3 {
        let (mut input, _) = cuda_control_fixture();
        let resources = serde_json::from_value(serde_json::json!({
            "limits":{"max_blocks":4,"max_bank_whole_payload_bytes":4u64<<20,
                "max_registry_entries":512,"max_container_host_bytes":1u64<<20,
                "max_invocation_host_bytes":32u64<<20,"max_invocation_device_bytes":5u64<<30},
            "invocation":{"public_session_bytes":2u64<<30,"packing_session_bytes":64u64<<20,
                "private_session_bytes":2u64<<30,"private_output_payload":{"host":0,"device":1u64<<20},
                "original_input_and_transfer_payload":{"host":4u64<<20,"device":4u64<<20},
                "additional_owner_metadata_payload":{"host":8u64<<20,"device":0}},
            "packing_artifact":{"packing_session_bytes":64u64<<20,"additional_owner_metadata_host_bytes":2u64<<20,
                "max_owned_artifact_host_bytes":4u64<<20,"max_declared_host_bytes":8u64<<20,"max_declared_device_bytes":128u64<<20},
            "graph_inspection":{"max_inspection_host_bytes":1u64<<20,"max_wire_fields":65536}
        })).unwrap();
        let mut manifest = asset("packing/device-packing.json");
        manifest.sha256 = "c".repeat(64);
        let mut graph = asset("packing/device_public_pack.onnx");
        graph.sha256 = "d".repeat(64);
        let PalsEndpointLaunchV3::OnnxCuda(cuda) = &mut input.endpoints[0] else {
            unreachable!()
        };
        cuda.cuda_record_pages = Some(Box::new(PalsCudaRecordPagesBindingV1 {
            mode: PalsCudaRecordPagesModeV1::RegisteredPackingV1,
            implementation_sha256: "e".repeat(64),
            manifest,
            graph,
            resources,
        }));
        input
    }
    fn cuda_record_page_wire_fixture(
        lock: &LockedPalsArenaLaunchV3,
    ) -> (serde_json::Value, serde_json::Value) {
        let (mut start, mut end) = cuda_records_fixture(lock);
        let cuda = lock.input.endpoints[0].cuda_model().unwrap();
        let pages = cuda.cuda_record_pages.as_ref().unwrap();
        let native = &start["native"];
        let public = cuda
            .model
            .graphs
            .iter()
            .find(|g| g.role == "public")
            .unwrap();
        let selection = serde_json::json!({
            "schema":"rz-pals-native-cuda-record-pages-selection/1","mode":"registered-packing-v1",
            "process_epoch":1,"installation_game_generation":0,"frozen_epoch":1,
            "model_manifest_sha256":native["export_manifest_sha256"],"model_epoch":native["model_epoch"],
            "public_graph_sha256":hash_array(&public.artifact.sha256),
            "encoding_sha256":hash_array(&resident_encoding_sha256(native).unwrap()),
            "runtime_sha256":hash_array(&cuda.cuda_bundle.canonical_sha256),"runtime_identity_scope":"closed_cuda_library_bundle",
            "device_id":0,"packing_graph_sha256":hash_array(&pages.graph.sha256),"packing_graph_bytes":pages.graph.bytes,
            "packing_manifest_sha256":hash_array(&pages.manifest.sha256),"packing_manifest_bytes":pages.manifest.bytes,
            "known_retained_artifact_host_bytes":4096,"implementation_sha256":hash_array(&pages.implementation_sha256),
            "initialized_scope":"initialized_provider_count","initialized_provider_count":275,
            "initialized_log_sha256":hash_array(&"f".repeat(64)),"initialized_log_digest_domain":"rz-pals-fixed-packing-initialized-provider-count/1",
            "model_native_sessions":2,"packing_native_sessions":1,
            "resource_accounting_scope":"combined_resident_owner_invocation_including_sessions_and_backing",
            "declared_resources":pages.declared_resources()
        });
        let snapshot = |generation: u64, public: u64, private: u64, live: u64| {
            let mut fields = serde_json::Map::new();
            for key in [
                "initialized_scope",
                "initialized_provider_count",
                "packing_native_sessions",
                "initialized_log_sha256",
                "initialized_log_digest_domain",
                "packing_graph_sha256",
                "process_epoch",
                "model_manifest_sha256",
                "model_epoch",
                "public_graph_sha256",
                "encoding_sha256",
                "runtime_sha256",
                "runtime_identity_scope",
                "device_id",
            ] {
                fields.insert(key.into(), selection[key].clone());
            }
            let mut value = serde_json::Value::Object(fields);
            value["schema"] = "rz-pals-resident-cuda-record-pages-observation/1".into();
            value["game_generation"] = generation.into();
            value["live_blocks"] = live.into();
            value["certified_projections"] = live.into();
            value["whole_owner_host_bytes"] = (live * 128).into();
            value["whole_owner_device_bytes"] = (live * 512).into();
            value["active_invocation"] = false.into();
            value["physical_completion_unknown"] = false.into();
            value["quarantined"] = false.into();
            value["stats"] = serde_json::json!({"admitted_views":private,
                "public_subset_runs_attempted":public,"public_subset_runs_completed":public,
                "packing_runs_attempted":private,"packing_runs_completed":private,
                "private_joined_runs_attempted":private,"private_joined_runs_completed":private,
                "certified_board_slices":public,"certified_record_slices":public,"published_blocks":public,
                "last_unique_whole_pins":1,"last_reserved_host_bytes":16u64<<20,"last_reserved_device_bytes":5u64<<30});
            value
        };
        let first = snapshot(1, 1, 2, 0);
        let last = snapshot(2, 2, 5, 2);
        // Fresh UCI owner generation is 1; the installed backend was generation
        // 0 before the actual startup NewGame reset ACK recorded below.
        start["native"]["game_generation"] = 1.into();
        // The game adds its own ucinewgame after the startup reset.
        end["native"]["game_generation"] = 2.into();
        for record in [&mut start, &mut end] {
            record["native"]["execution"]["cuda_record_pages"] = selection.clone();
            record["native"]["startup_probe"]["cuda_record_page_observation"] = first.clone();
        }
        end["native"]["cuda_record_page_observation"] = last;
        end["native"]["cuda_record_page_observation_scope"] =
            "exclusive_worker_before_shutdown".into();
        attach_control_witness(lock, &mut start, &mut end);
        (start, end)
    }
    fn audit_cuda_record_page_wire(
        lock: &LockedPalsArenaLaunchV3,
        start: &serde_json::Value,
        end: &serde_json::Value,
    ) -> Result<(PalsNativeSessionAuditV3, u32), ArenaError> {
        validate_pals_native_records(
            lock,
            NativeEngineRole::Baseline,
            &serde_json::to_vec(start).unwrap(),
            &serde_json::to_vec(end).unwrap(),
            "native-process-100",
        )
    }
    #[test]
    fn resident_cuda_omission_preserves_legacy_lock_arguments_and_audit_projection() {
        for old in [cuda_fixture().0, cuda_control_fixture().0] {
            let text = serde_json::to_string(&old).unwrap();
            assert!(!text.contains("cuda_record_pages"));
            let decoded = PalsArenaLaunchV3::from_json(&text).unwrap();
            assert_eq!(
                crate::canonical_sha256(&decoded).unwrap(),
                crate::canonical_sha256(&old).unwrap()
            );
            let old_lock = old.lock().unwrap();
            let decoded_lock = decoded.lock().unwrap();
            assert_eq!(old_lock.sha256(), decoded_lock.sha256());
            for index in 0..2 {
                assert_eq!(
                    old_lock.endpoint_views[index].arguments,
                    decoded_lock.endpoint_views[index].arguments
                );
                assert_eq!(
                    old_lock.endpoint_views[index].assets,
                    decoded_lock.endpoint_views[index].assets
                );
            }
            let (mut start, mut end) = cuda_records_fixture(&old_lock);
            if old_lock.input.endpoints[0]
                .cuda_model()
                .unwrap()
                .cuda_control
                .is_some()
            {
                attach_control_witness(&old_lock, &mut start, &mut end);
            }
            let audit = audit_cuda_record_page_wire(&old_lock, &start, &end)
                .unwrap()
                .0;
            assert!(
                serde_json::to_value(audit)
                    .unwrap()
                    .get("cuda_record_pages")
                    .is_none()
            );
            let mut invalid = serde_json::to_value(&old_lock.input).unwrap();
            invalid["endpoints"][0]["configuration"]["cuda_record_pages"] = serde_json::Value::Null;
            assert!(
                PalsArenaLaunchV3::from_json(&serde_json::to_string(&invalid).unwrap()).is_err()
            );
        }
    }
    #[test]
    fn resident_cuda_separate_assets_and_six_flags_fit_maximum_registered_own_recipe() {
        let mut input = cuda_record_pages_fixture();
        let PalsEndpointLaunchV3::OnnxCuda(cuda) = &mut input.endpoints[0] else {
            unreachable!()
        };
        cuda.cuda_control.as_mut().unwrap().loading_profile = Some(loading_binding_fixture());
        cuda.startup_probe_timeout_ms = Some(30_000);
        let input = select_post_repair_recheck(input);
        let mut legacy = input.clone();
        let PalsEndpointLaunchV3::OnnxCuda(cuda) = &mut legacy.endpoints[0] else {
            unreachable!()
        };
        cuda.cuda_record_pages = None;
        let legacy = legacy.lock().unwrap();
        let lock = input.lock().unwrap();
        let args = &lock.endpoint_views[0].arguments;
        assert_eq!(legacy.endpoint_views[0].arguments.len(), 23);
        assert_eq!(args.len(), 29);
        assert!(args.len() <= 32 && args.iter().all(|arg| arg.len() <= 4096));
        let prefixes = [
            "--pals-cuda-record-pages=",
            "--pals-packing-manifest=",
            "--pals-packing-manifest-sha256=",
            "--pals-packing-graph=",
            "--pals-packing-graph-sha256=",
            "--pals-cuda-record-pages-resources=",
        ];
        for prefix in prefixes {
            assert_eq!(args.iter().filter(|arg| arg.starts_with(prefix)).count(), 1);
        }
        let without_selected: Vec<_> = args
            .iter()
            .filter(|arg| !prefixes.iter().any(|p| arg.starts_with(p)))
            .cloned()
            .collect();
        assert_eq!(without_selected, legacy.endpoint_views[0].arguments);
        let resource_text = args
            .iter()
            .find_map(|arg| arg.strip_prefix("--pals-cuda-record-pages-resources="))
            .unwrap();
        assert!(resource_text.starts_with('{') && resource_text.len() <= 8192);
        let cuda = lock.input.endpoints[0].cuda_model().unwrap();
        let pages = cuda.cuda_record_pages.as_ref().unwrap();
        assert_eq!(
            crate::decode_json::<NativeCudaRecordPagesResourceInput>(resource_text).unwrap(),
            pages.resources
        );
        assert_eq!(cuda.model.graphs.len(), 2);
        assert!(cuda.model.graphs.iter().all(|g| g.role != "packing"));
        assert_eq!(
            lock.endpoint_views[0].assets.len(),
            legacy.endpoint_views[0].assets.len() + 2
        );
        assert_eq!(
            lock.input.declared_artifacts().len(),
            legacy.input.declared_artifacts().len() + 2
        );
        assert_eq!(
            lock.snapshot_relative_path(&pages.manifest).unwrap(),
            format!("pals-packing-{}/device-packing.json", pages.manifest.sha256)
        );
        assert_eq!(
            lock.snapshot_relative_path(&pages.graph).unwrap(),
            format!(
                "pals-packing-{}/device_public_pack.onnx",
                pages.manifest.sha256
            )
        );
    }
    #[test]
    fn resident_cuda_registration_rejects_partial_null_unknown_and_incompatible_declarations() {
        let input = cuda_record_pages_fixture();
        input.clone().lock().unwrap();
        let raw = serde_json::to_value(&input).unwrap();
        for mutation in 0..9 {
            let mut bad = raw.clone();
            let pages = &mut bad["endpoints"][0]["configuration"]["cuda_record_pages"];
            match mutation {
                0 => *pages = serde_json::Value::Null,
                1 => {
                    pages.as_object_mut().unwrap().remove("graph");
                }
                2 => pages["mode"] = "registered-packing-v2".into(),
                3 => pages["extra"] = true.into(),
                4 => pages["implementation_sha256"] = "E".repeat(64).into(),
                5 => pages["resources"]["limits"]["max_blocks"] = serde_json::json!(4.0),
                6 => {
                    pages["resources"]["invocation"]["packing_session_bytes"] =
                        serde_json::json!(u64::MAX)
                }
                7 => pages["graph"]["bytes"] = serde_json::json!(2u64 * 1024 * 1024 + 1),
                _ => {
                    pages["resources"]["limits"]["max_invocation_device_bytes"] =
                        serde_json::json!(7u64 << 30)
                }
            }
            assert!(
                PalsArenaLaunchV3::from_json(&serde_json::to_string(&bad).unwrap()).is_err(),
                "mutation {mutation}"
            );
        }
        let duplicate = serde_json::to_string(&raw).unwrap().replacen(
            "\"mode\":\"registered-packing-v1\"",
            "\"mode\":\"registered-packing-v1\",\"mode\":\"registered-packing-v1\"",
            1,
        );
        assert!(PalsArenaLaunchV3::from_json(&duplicate).is_err());
        // Fields cannot cross to CPU/reference/OwnCpu endpoint schemas.
        let declaration = raw["endpoints"][0]["configuration"]["cuda_record_pages"].clone();
        for other in [
            serde_json::to_value(native_fixture()).unwrap(),
            serde_json::to_value(fixture()).unwrap(),
        ] {
            for index in 0..2 {
                let mut bad = other.clone();
                bad["endpoints"][index]["configuration"]["cuda_record_pages"] = declaration.clone();
                assert!(
                    PalsArenaLaunchV3::from_json(&serde_json::to_string(&bad).unwrap()).is_err()
                );
            }
        }
        let mut reference = serde_json::to_value(PalsEndpointLaunchV3::ReferenceUci {
            expected_uci_name: "reference".into(),
            arguments: vec![],
            environment: None,
        })
        .unwrap();
        reference["configuration"]["cuda_record_pages"] = declaration;
        assert!(
            crate::decode_json::<PalsEndpointLaunchV3>(&serde_json::to_string(&reference).unwrap())
                .is_err()
        );
        let mut no_control = input.clone();
        let PalsEndpointLaunchV3::OnnxCuda(cuda) = &mut no_control.endpoints[0] else {
            unreachable!()
        };
        cuda.cuda_control = None;
        assert!(no_control.lock().is_err());
        let mut declared_helper = input.clone();
        let PalsEngineV3::Pals(engine) = &mut declared_helper.semantic_lock.manifest.engines[0]
        else {
            unreachable!()
        };
        let mut binary = asset("helper/stockfish");
        binary.source = "https://github.com/official-stockfish/Stockfish".into();
        binary.license = "GPL-3.0-or-later".into();
        engine.cpu_r = PalsCpuRSelectionV3::ExternalUci(Box::new(PalsExternalCpuRV3 {
            selection: PalsExternalCpuRSelectionV3::StockfishEmbeddedNnue,
            profile: asset("helper/profile.json"),
            profile_canonical_sha256: "a".repeat(64),
            binary,
            resolver: PalsExternalCpuRResolverV3 {
                version: rz_search::pals::value::MODEL_WDL_RESOLVER_VERSION.into(),
                semantics_sha256: digest(
                    rz_search::pals::value::MODEL_WDL_RESOLVER_SEMANTICS.as_bytes(),
                ),
            },
            policy: PalsExternalCpuRPolicyV3 {
                resource_scope: PalsExternalCpuRResourceScopeV3::InheritedParentCgroup,
                max_owners: 1,
                max_active_tasks: 1,
                max_process_leaders: 1,
                inherited_kernel_tasks_max: 128,
                threads_max: 2,
                hash_mib_max: 16,
                max_depth: 8,
                max_prefix_plies: 64,
                max_nodes_per_task: 10_000,
                handshake_max_ms: 1000,
                task_wall_time_max_ms: 1000,
                stop_grace_max_ms: 100,
                shutdown_grace_max_ms: 100,
                lifetime_output_bytes_max: 4096,
                line_bytes_max: 1024,
            },
        }));
        declared_helper.semantic_lock = declared_helper.semantic_lock.manifest.lock().unwrap();
        let error = declared_helper.validate().unwrap_err();
        assert!(
            error
                .to_string()
                .contains("resident CUDA registration requires")
        );
        let mut helper = input;
        let PalsEndpointLaunchV3::OnnxCuda(cuda) = &mut helper.endpoints[0] else {
            unreachable!()
        };
        cuda.model.external_cpu_r = Some(PalsExternalCpuRLaunchV3 {
            registered_program: "/registered-cas/stockfish".into(),
            registered_working_directory: "/registered-cas".into(),
        });
        assert!(helper.lock().is_err());
    }
    #[test]
    fn resident_cuda_audit_preserves_distinct_actual_selection_dynamic_ack_and_model_counts() {
        let lock = cuda_record_pages_fixture().lock().unwrap();
        let (start, end) = cuda_record_page_wire_fixture(&lock);
        let (audit, pid) = audit_cuda_record_page_wire(&lock, &start, &end).unwrap();
        assert_eq!(pid, 100);
        let resident = audit.cuda_record_pages.unwrap();
        assert_eq!(
            resident.selection,
            start["native"]["execution"]["cuda_record_pages"]
        );
        assert_eq!(
            resident.startup_observation,
            start["native"]["startup_probe"]["cuda_record_page_observation"]
        );
        assert_eq!(
            resident.termination_observation,
            end["native"]["cuda_record_page_observation"]
        );
        assert_ne!(
            resident.startup_observation,
            resident.termination_observation
        );
        assert_eq!(
            array_hash(&resident.selection["implementation_sha256"]).unwrap(),
            "e".repeat(64)
        );
        assert_ne!(
            resident.selection["implementation_sha256"],
            start["native"]["adapter_source_sha256"]
        );
        assert_eq!(audit.startup_nn_calls_completed, 3);
        assert_eq!(end["native"]["backend_stats"]["completed_nn_inputs"], 7);
        assert_eq!(
            resident.termination_observation["stats"]["packing_runs_completed"],
            5
        );
        // Retained CheckedFixedPackingGraph wrapper bytes may exceed the raw
        // artifact sub-budget, but must fit additional owner metadata.
        let mut wrapper_start = start.clone();
        let mut wrapper_end = end.clone();
        for r in [&mut wrapper_start, &mut wrapper_end] {
            r["native"]["execution"]["cuda_record_pages"]["known_retained_artifact_host_bytes"] =
                serde_json::json!(5u64 << 20);
        }
        assert!(audit_cuda_record_page_wire(&lock, &wrapper_start, &wrapper_end).is_ok());
    }
    #[test]
    fn resident_cuda_audit_rejects_borrowed_ack_drift_unknown_and_missing_physical_close() {
        let lock = cuda_record_pages_fixture().lock().unwrap();
        let (start, end) = cuda_record_page_wire_fixture(&lock);
        for mutation in 0..15 {
            let mut s = start.clone();
            let mut t = end.clone();
            match mutation {
                0 => {
                    for r in [&mut s, &mut t] {
                        r["native"]["execution"]["cuda_record_pages"]["implementation_sha256"] =
                            hash_array(&"b".repeat(64)).into();
                    }
                }
                1 => {
                    t["native"]["cuda_record_page_observation"] =
                        s["native"]["startup_probe"]["cuda_record_page_observation"].clone()
                }
                2 => t["native"]["cuda_record_page_observation"]["game_generation"] = 0.into(),
                3 => {
                    t["native"]["cuda_record_page_observation"]["physical_completion_unknown"] =
                        true.into()
                }
                4 => t["native"]["cuda_record_page_observation"]["quarantined"] = true.into(),
                5 => {
                    t["native"]["cuda_record_page_observation"]["stats"]["packing_runs_completed"] =
                        4.into()
                }
                6 => {
                    t["native"]["cuda_record_page_observation"]["stats"]["certified_board_slices"] =
                        0.into()
                }
                7 => {
                    t["native"]["cuda_record_page_observation_scope"] = "installed_snapshot".into()
                }
                8 => t["native"]["cuda_record_page_observation_error"] = serde_json::Value::Null,
                9 => {
                    s["native"]["cuda_record_page_observation"] =
                        end["native"]["cuda_record_page_observation"].clone()
                }
                10 => t["native"]["cuda_record_page_observation"]["extra"] = true.into(),
                11 => t["native"]["physical_shutdown_confirmed"] = false.into(),
                12 => {
                    t["native"]
                        .as_object_mut()
                        .unwrap()
                        .remove("cuda_record_page_observation");
                }
                13 => {
                    t["native"]["cuda_record_page_observation"]["stats"]["last_reserved_device_bytes"] =
                        serde_json::json!(6u64 << 30)
                }
                _ => {
                    for r in [&mut s, &mut t] {
                        r["native"]["startup_probe"]["cuda_record_page_observation"]["game_generation"] =
                            0.into();
                    }
                }
            }
            assert!(
                audit_cuda_record_page_wire(&lock, &s, &t).is_err(),
                "mutation {mutation}"
            );
        }
        let bytes = serde_json::to_string(&end).unwrap().replacen(
            "\"packing_runs_completed\":5",
            "\"packing_runs_completed\":5,\"packing_runs_completed\":5",
            1,
        );
        assert!(
            validate_pals_native_records(
                &lock,
                NativeEngineRole::Baseline,
                &serde_json::to_vec(&start).unwrap(),
                bytes.as_bytes(),
                "native-process-100"
            )
            .is_err()
        );
    }
    #[test]
    fn resident_cuda_unregistered_markers_are_rejected_even_when_null_or_declaration_only() {
        let lock = cuda_fixture().0.lock().unwrap();
        let (start, end) = cuda_records_fixture(&lock);
        for key in [
            "cuda_record_page_observation",
            "cuda_record_page_observation_error",
            "cuda_record_page_observation_scope",
        ] {
            let mut bad = end.clone();
            bad["native"][key] = serde_json::Value::Null;
            assert!(audit_cuda_record_page_wire(&lock, &start, &bad).is_err());
        }
        for value in [
            serde_json::Value::Null,
            serde_json::json!({"mode":"registered-packing-v1"}),
        ] {
            let mut bad = end.clone();
            bad["native"]["execution"]["cuda_record_pages"] = value;
            assert!(audit_cuda_record_page_wire(&lock, &start, &bad).is_err());
            assert!(validate_cuda_record_pages_pair(None, &bad["native"], &bad["native"]).is_err());
            assert!(require_no_cuda_record_pages(&bad["native"]).is_err());
        }
        let selected = cuda_record_pages_fixture().lock().unwrap();
        let (s, t) = cuda_records_fixture(&selected);
        assert!(audit_cuda_record_page_wire(&selected, &s, &t).is_err());
    }
    #[test]
    fn undeclared_host_record_pages_cannot_be_accepted_as_whole_input_native_execution() {
        let (input, _) = cuda_fixture();
        let lock = input.lock().unwrap();
        let (start, end) = cuda_records_fixture(&lock);
        let validate = |s: &serde_json::Value, t: &serde_json::Value| {
            validate_pals_native_records(
                &lock,
                NativeEngineRole::Baseline,
                &serde_json::to_vec(s).unwrap(),
                &serde_json::to_vec(t).unwrap(),
                "native-process-100",
            )
        };
        assert!(validate(&start, &end).is_ok());
        let mut legacy_start = start.clone();
        let mut legacy_end = end.clone();
        for record in [&mut legacy_start, &mut legacy_end] {
            record["native"]["execution"]["host_record_pages"] = serde_json::Value::Null;
            record["native"]["host_record_page_observation"] = serde_json::Value::Null;
        }
        assert!(validate(&legacy_start, &legacy_end).is_ok());
        for startup in [true, false] {
            for declaration in [true, false] {
                for selected in [
                    serde_json::json!({"schema":"rz-pals-host-record-page-observation/1"}),
                    serde_json::json!({}),
                    serde_json::json!(false),
                ] {
                    let mut s = start.clone();
                    let mut t = end.clone();
                    let record = if startup { &mut s } else { &mut t };
                    if declaration {
                        record["native"]["execution"]["host_record_pages"] = selected;
                    } else {
                        record["native"]["host_record_page_observation"] = selected;
                    }
                    let error = validate(&s, &t).unwrap_err();
                    assert!(
                        matches!(error, ArenaError::Integrity(ref detail) if detail.contains("unsupported undeclared host record-page"))
                    );
                }
            }
        }
    }
    #[test]
    fn undeclared_private_warm_cannot_be_accepted_as_fresh_whole_input_execution() {
        let (input, _) = cuda_fixture();
        let lock = input.lock().unwrap();
        let (start, end) = cuda_records_fixture(&lock);
        let validate = |s: &serde_json::Value, t: &serde_json::Value| {
            validate_pals_native_records(
                &lock,
                NativeEngineRole::Baseline,
                &serde_json::to_vec(s).unwrap(),
                &serde_json::to_vec(t).unwrap(),
                "native-process-100",
            )
        };
        assert!(validate(&start, &end).is_ok());
        let fields = [
            "private_warm",
            "private_warm_observation",
            "private_warm_observation_unavailable",
        ];
        let set = |record: &mut serde_json::Value, field: &str, value: serde_json::Value| {
            if field == "private_warm" {
                record["native"]["execution"][field] = value;
            } else {
                record["native"][field] = value;
            }
        };
        let mut null_start = start.clone();
        let mut null_end = end.clone();
        for record in [&mut null_start, &mut null_end] {
            for field in fields {
                set(record, field, serde_json::Value::Null);
            }
        }
        assert!(validate(&null_start, &null_end).is_ok());
        for startup in [true, false] {
            for field in fields {
                for value in [
                    serde_json::json!({"schema_version": 1}),
                    serde_json::json!({}),
                    serde_json::json!(false),
                    serde_json::json!(true),
                    serde_json::json!(0),
                    serde_json::json!("unavailable"),
                ] {
                    let mut s = start.clone();
                    let mut t = end.clone();
                    set(if startup { &mut s } else { &mut t }, field, value);
                    let error = validate(&s, &t).unwrap_err();
                    assert!(
                        matches!(error, ArenaError::Integrity(ref detail) if detail.contains("unsupported undeclared private warm")),
                        "startup={startup}, field={field}"
                    );
                }
            }
        }
    }
    #[test]
    fn independent_helper_evidence_cannot_silently_attest_default_own_cpu_r() {
        let (input, _) = cuda_fixture();
        let lock = input.lock().unwrap();
        let (start, end) = cuda_records_fixture(&lock);
        for startup in [true, false] {
            let mut s = start.clone();
            let mut t = end.clone();
            let record = if startup { &mut s } else { &mut t };
            record["cpu_checker"] =
                serde_json::json!({"schema_version":1,"domain":"rz-pals-checker-process/1"});
            let error = validate_pals_native_records(
                &lock,
                NativeEngineRole::Baseline,
                &serde_json::to_vec(&s).unwrap(),
                &serde_json::to_vec(&t).unwrap(),
                "native-process-100",
            )
            .unwrap_err();
            assert!(
                matches!(error, ArenaError::Integrity(ref detail) if detail.contains("unsupported external CPU_R native receipt"))
            );
        }
    }
    #[test]
    fn pals_v3_launch_lock_has_own_identity_and_actual_endpoint_cli() {
        let lock = fixture().lock().unwrap();
        let decoded = LockedPalsArenaLaunchV3::from_json(&lock.to_json().unwrap()).unwrap();
        assert_eq!(decoded.sha256(), lock.sha256());
        assert_eq!(decoded.pair_metadata_cap(), PALS_PAIR_METADATA_CAP);
        let pals = decoded.engine_view(NativeEngineRole::Baseline).unwrap();
        let cpu = decoded.engine_view(NativeEngineRole::Candidate).unwrap();
        assert!(
            pals.external
                .unwrap()
                .arguments
                .contains(&"--search=pals".into())
        );
        assert!(
            cpu.external
                .unwrap()
                .arguments
                .contains(&"--search=cpu".into())
        );
        assert!(
            !pals
                .external
                .unwrap()
                .arguments
                .iter()
                .any(|a| a.contains("onnx") || a.contains("attestation") || a.contains("weights"))
        );
        assert_eq!(decoded.expected_provider_sessions(), 0);
        let output_root = std::env::temp_dir().join("rovezero-pals-launch-mock-run");
        let arguments = decoded
            .runtime_arguments(NativeEngineRole::Baseline, &output_root)
            .unwrap();
        assert_eq!(arguments.len(), 3);
        assert!(arguments.contains(&OsString::from(format!(
            "--search-work-output-root={}",
            output_root.display()
        ))));
        assert!(
            decoded
                .preflight_arguments(NativeEngineRole::Baseline, &output_root)
                .unwrap()
                .is_empty()
        );
        assert_eq!(
            decoded.view().clock,
            NativePairClock::Game(NativeGameClockV3 {
                base_ms: 120_000,
                increment_ms: 1000
            })
        );
        let mut value: serde_json::Value = serde_json::from_str(&lock.to_json().unwrap()).unwrap();
        value["input"]["endpoints"][0]["configuration"]["max_rounds"] = 17.into();
        assert!(LockedPalsArenaLaunchV3::from_json(&value.to_string()).is_err());
    }

    #[test]
    fn pals_receipt_reservation_is_admitted_before_snapshot_or_process_work() {
        let mut input = fixture();
        input.budget.max_output_bytes = PALS_PAIR_METADATA_CAP;
        // Small historical declarations remain parseable. They cannot launch
        // under the new fixed metadata reservation or silently grow on save.
        let lock = input.lock().unwrap();
        let decoded = LockedPalsArenaLaunchV3::from_json(&lock.to_json().unwrap()).unwrap();
        assert!(matches!(
            decoded.validate_execution(),
            Err(ArenaError::Budget(_))
        ));
        assert!(crate::native_launch::pair_stream_cap(65_536, 65_536).is_err());
        assert!(crate::native_launch::pair_stream_cap(65_537, 65_536).is_err());
        assert!(crate::native_launch::pair_stream_cap(u64::MAX, 0).is_err());
        let output = 8 * 1024 * 1024;
        let legacy_stream = crate::native_launch::pair_stream_cap(
            output,
            crate::native_launch::NATIVE_PAIR_METADATA_CAP,
        )
        .unwrap();
        let pals_stream =
            crate::native_launch::pair_stream_cap(output, PALS_PAIR_METADATA_CAP).unwrap();
        assert_eq!(legacy_stream, (output - 65_536) / 2);
        assert_eq!(pals_stream, 3 * 1024 * 1024);
        assert_eq!(2 * pals_stream + decoded.pair_metadata_cap(), output);
        assert!(fixture().lock().unwrap().validate_execution().is_ok());
    }
    #[test]
    fn pals_runner_registers_new_patch_and_binary_without_rewriting_legacy_locks() {
        let legacy = fixture();
        let old = legacy.lock().unwrap();
        assert_eq!(
            legacy.runner_status_patch().unwrap(),
            PalsRunnerStatusPatchV3::LegacyClockOnly
        );
        assert!(legacy.require_reap_status_runner().is_err());
        let mut current = legacy.clone();
        let patch = current.runner.dirty_patch.as_mut().unwrap();
        patch.sha256 = PALS_CLOCK_REAP_PATCH_SHA256.into();
        patch.bytes = PALS_CLOCK_REAP_PATCH_BYTES;
        current.runner.binary.sha256 = PALS_CLOCK_REAP_RUNNER_SHA256.into();
        current.runner.binary.bytes = PALS_CLOCK_REAP_RUNNER_BYTES;
        let new = current.lock().unwrap();
        assert!(current.require_reap_status_runner().is_ok());
        assert_ne!(old.sha256(), new.sha256());
        assert_eq!(
            LockedPalsArenaLaunchV3::from_json(&old.to_json().unwrap())
                .unwrap()
                .sha256(),
            old.sha256()
        );
        assert_eq!(
            current.semantic_lock.canonical_sha256,
            legacy.semantic_lock.canonical_sha256
        );
        let mut invalid_patch = current.clone();
        invalid_patch.runner.dirty_patch.as_mut().unwrap().bytes -= 1;
        assert!(invalid_patch.lock().is_err());
        let mut unregistered_binary = current;
        unregistered_binary.runner.binary.sha256 = "f".repeat(64);
        assert!(unregistered_binary.lock().is_ok()); // metadata-only declaration remains explicit.
        assert!(unregistered_binary.require_reap_status_runner().is_err());
    }
    #[test]
    fn pals_runner_colab_release_version_and_binary_are_one_registration() {
        let legacy = fixture();
        let historical = legacy.lock().unwrap().to_json().unwrap();
        let mut current = legacy.clone();
        let patch = current.runner.dirty_patch.as_mut().unwrap();
        patch.sha256 = PALS_CLOCK_REAP_PATCH_SHA256.into();
        patch.bytes = PALS_CLOCK_REAP_PATCH_BYTES;
        let registrations = [
            (
                crate::FASTCHESS_VERSION,
                PALS_CLOCK_REAP_RUNNER_SHA256,
                PALS_CLOCK_REAP_RUNNER_BYTES,
            ),
            (
                PALS_CLOCK_REAP_COLAB_RUNNER_VERSION,
                PALS_CLOCK_REAP_COLAB_RUNNER_SHA256,
                PALS_CLOCK_REAP_COLAB_RUNNER_BYTES,
            ),
        ];
        for (v, version) in registrations.iter().enumerate() {
            for (h, hash) in registrations.iter().enumerate() {
                for (b, bytes) in registrations.iter().enumerate() {
                    current.runner.version = version.0.into();
                    current.runner.binary.sha256 = hash.1.into();
                    current.runner.binary.bytes = bytes.2;
                    let accepted = v == h && h == b;
                    assert_eq!(current.require_reap_status_runner().is_ok(), accepted);
                    if accepted {
                        let lock = current.lock().unwrap();
                        assert_eq!(
                            LockedPalsArenaLaunchV3::from_json(&lock.to_json().unwrap())
                                .unwrap()
                                .sha256(),
                            lock.sha256()
                        );
                    } else if v == 1 {
                        assert!(current.lock().is_err());
                    }
                }
            }
        }
        current.runner.version = PALS_CLOCK_REAP_COLAB_RUNNER_VERSION.into();
        current.runner.binary.sha256 = PALS_CLOCK_REAP_COLAB_RUNNER_SHA256.into();
        current.runner.binary.bytes = PALS_CLOCK_REAP_COLAB_RUNNER_BYTES + 1;
        assert!(current.require_reap_status_runner().is_err());
        assert!(current.lock().is_err());
        assert_eq!(legacy.lock().unwrap().to_json().unwrap(), historical);
    }
    #[test]
    fn pals_native_recipe_assets_and_dynamic_arguments_do_not_leak_lc0_flags() {
        let lock = native_fixture().lock().unwrap();
        assert_eq!(lock.expected_provider_sessions(), 2);
        let view = lock.engine_view(NativeEngineRole::Baseline).unwrap();
        let e = view.external.unwrap();
        assert!(
            e.arguments
                .contains(&"--pals-export-manifest={{asset:0}}".into())
        );
        assert_eq!(e.assets.len(), 5);
        assert!(!e.arguments.iter().any(|a| a.starts_with("--onnx-")
            || a == "--attestation"
            || a.starts_with("--source-weights")));
        for a in &e.assets[2..] {
            assert!(lock.snapshot_relative_path(a).unwrap().starts_with("pals-"));
        }
        let output_root = std::env::temp_dir().join("rovezero-pals-launch-native-runtime");
        let other_output_root = std::env::temp_dir().join("rovezero-pals-launch-second-attempt");
        let arguments = lock
            .runtime_arguments(NativeEngineRole::Baseline, &output_root)
            .unwrap();
        assert_eq!(arguments.len(), 4);
        let preflight = lock
            .preflight_arguments(NativeEngineRole::Baseline, &output_root)
            .unwrap();
        assert_eq!(preflight, vec![pals_shared_runtime_argument().unwrap()]);
        let other_runtime = lock
            .runtime_arguments(NativeEngineRole::Baseline, &other_output_root)
            .unwrap();
        let other_preflight = lock
            .preflight_arguments(NativeEngineRole::Baseline, &other_output_root)
            .unwrap();
        assert_eq!(arguments.last(), other_runtime.last());
        assert_eq!(preflight, other_preflight);
        let cache_argument = arguments.last().unwrap().to_str().unwrap();
        assert!(cache_argument.starts_with("--pals-runtime-cache-root="));
        assert!(
            !cache_argument.contains(output_root.to_str().unwrap())
                && !cache_argument.contains("second-attempt")
        );
    }
    #[test]
    fn pals_cuda_alias_v2_descriptor_matches_native_codec_without_rewriting_v1() {
        let (mut input, old_descriptor) = cuda_fixture();
        let original = input.lock().unwrap().to_json().unwrap();
        let PalsEndpointLaunchV3::OnnxCuda(cuda) = &mut input.endpoints[0] else {
            unreachable!()
        };
        let bundle = &mut cuda.cuda_bundle;
        bundle.schema_version = Some(2);
        bundle.loader_aliases = Some(
            bundle
                .files
                .iter()
                .filter(|file| file.filename.starts_with("libcudnn"))
                .map(|file| rz_experiments::CudaBundleAliasBindingV2 {
                    filename: format!("{}.8.0", file.filename),
                    target: file.filename.clone(),
                })
                .collect(),
        );
        let descriptor = serde_json::to_vec(&serde_json::json!({"schema_version":2,
            "loader_aliases":bundle.loader_aliases,
            "files":bundle.files.iter().map(|f|serde_json::json!({"role":f.role,
                "filename":f.filename,"bytes":f.artifact.bytes,"sha256":f.artifact.sha256})).collect::<Vec<_>>()})).unwrap();
        bundle.manifest.sha256 = digest(&descriptor);
        bundle.manifest.bytes = descriptor.len() as u64;
        bundle.canonical_sha256 = bundle.canonical_digest().unwrap();
        let native = rz_eval::runtime_pin::CudaRuntimeBundleSpec::from_json(
            std::str::from_utf8(&descriptor).unwrap(),
        )
        .unwrap();
        assert_eq!(
            bundle.canonical_sha256,
            native
                .digest()
                .unwrap()
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect::<String>()
        );
        let lock = input.lock().unwrap();
        lock.validate_cuda_bundle(NativeEngineRole::Baseline, &descriptor)
            .unwrap();
        assert!(
            lock.validate_cuda_bundle(NativeEngineRole::Baseline, &old_descriptor)
                .is_err()
        );
        let (old_input, _) = cuda_fixture();
        assert_eq!(old_input.lock().unwrap().to_json().unwrap(), original);
    }
    #[test]
    fn pals_cuda_recipe_pins_nineteen_libraries_without_cpu_or_lc0_fallback() {
        let (f, descriptor) = cuda_fixture();
        let lock = f.lock().unwrap();
        assert_eq!(lock.expected_provider_sessions(), 2);
        lock.validate_cuda_bundle(NativeEngineRole::Baseline, &descriptor)
            .unwrap();
        let view = lock.engine_view(NativeEngineRole::Baseline).unwrap();
        assert_eq!(view.cuda_bundle.unwrap().files.len(), 19);
        let endpoint = view.external.unwrap();
        assert_eq!(endpoint.assets.len(), 23); // export, core, two graphs, descriptor, eighteen sibling libs.
        assert!(endpoint.arguments.contains(&"--pals-provider=cuda".into()));
        assert!(
            endpoint
                .arguments
                .contains(&"--pals-cuda-bundle={{asset:4}}".into())
        );
        assert!(
            endpoint
                .arguments
                .contains(&"--pals-device-public-memory=false".into())
        );
        assert!(
            !endpoint
                .arguments
                .iter()
                .any(|arg| arg == "--pals-provider=cpu"
                    || arg.starts_with("--onnx-")
                    || arg.starts_with("--source-weights"))
        );
        let serialized = lock.to_json().unwrap();
        assert_eq!(
            LockedPalsArenaLaunchV3::from_json(&serialized)
                .unwrap()
                .sha256(),
            lock.sha256()
        );
        let mut bad = f.clone();
        let PalsEndpointLaunchV3::OnnxCuda(n) = &mut bad.endpoints[0] else {
            unreachable!()
        };
        n.cuda_bundle.files.pop();
        assert!(bad.lock().is_err());
        let mut bad = f.clone();
        let PalsEndpointLaunchV3::OnnxCuda(n) = &mut bad.endpoints[0] else {
            unreachable!()
        };
        n.model.runtime.sha256 = "f".repeat(64);
        assert!(bad.lock().is_err());
        let mut bad = f.clone();
        let PalsEndpointLaunchV3::OnnxCuda(n) = &mut bad.endpoints[0] else {
            unreachable!()
        };
        n.device_id = 1;
        assert!(bad.lock().is_err());
        let mut bad = f.clone();
        let PalsEndpointLaunchV3::OnnxCuda(n) = &mut bad.endpoints[0] else {
            unreachable!()
        };
        n.model.graphs = native_fixture().endpoints[0]
            .native_model()
            .unwrap()
            .graphs
            .clone();
        assert!(bad.lock().is_err());
        let mut bad = f;
        let PalsEngineV3::Pals(e) = &mut bad.semantic_lock.manifest.engines[0] else {
            unreachable!()
        };
        e.model.backend = PalsModelBackendV3::OrtCpu;
        bad.semantic_lock = bad.semantic_lock.manifest.lock().unwrap();
        assert!(bad.lock().is_err());
        let mut wrong = descriptor;
        wrong[0] = b'[';
        assert!(
            lock.validate_cuda_bundle(NativeEngineRole::Baseline, &wrong)
                .is_err()
        );
    }
    #[test]
    fn pals_cuda_control_binding_preserves_strict_lock_and_owns_profile_parent() {
        let (strict, _) = cuda_fixture();
        let strict_lock = strict.lock().unwrap();
        let encoded = serde_json::to_string(&strict).unwrap();
        assert!(!encoded.contains("cuda_control"));
        assert_eq!(
            PalsArenaLaunchV3::from_json(&encoded)
                .unwrap()
                .lock()
                .unwrap()
                .sha256(),
            strict_lock.sha256()
        );
        let (f, bytes) = cuda_control_fixture();
        let lock = f.lock().unwrap();
        assert_ne!(lock.sha256(), strict_lock.sha256());
        assert_eq!(
            lock.input.semantic_lock.canonical_sha256,
            strict.semantic_lock.canonical_sha256
        );
        lock.validate_control_inventory(NativeEngineRole::Baseline, &bytes)
            .unwrap();
        let external = lock
            .engine_view(NativeEngineRole::Baseline)
            .unwrap()
            .external
            .unwrap();
        assert_eq!(external.assets.len(), 24);
        assert!(
            external
                .arguments
                .contains(&"--pals-cuda-control-mode=inventory-v2".into())
        );
        assert!(
            external
                .arguments
                .contains(&"--pals-cuda-control-inventory={{asset:23}}".into())
        );
        let inventory = &f.endpoints[0]
            .cuda_model()
            .unwrap()
            .cuda_control
            .as_ref()
            .unwrap()
            .inventory;
        assert!(lock.declared_inputs().contains(&inventory));
        assert_eq!(
            lock.snapshot_relative_path(inventory).unwrap(),
            format!("pals-control-{}/inventory.v2.json", inventory.sha256)
        );
        let root = std::env::temp_dir().join("rovezero-pals-control-owned-parent");
        let expected = OsString::from(format!("--pals-cuda-profile-parent={}", root.display()));
        let runtime = lock
            .runtime_arguments(NativeEngineRole::Baseline, &root)
            .unwrap();
        let preflight = lock
            .preflight_arguments(NativeEngineRole::Baseline, &root)
            .unwrap();
        assert_eq!(runtime.len(), 5);
        assert_eq!(preflight.len(), 2);
        assert_eq!(runtime.last(), Some(&expected));
        assert_eq!(preflight.last(), Some(&expected));
        assert_eq!(runtime[3], preflight[0]); // Runtime cache stays shared outside the profile tree.
        let mut wrong = json(&bytes).unwrap();
        wrong["graphs"][0]["sha256"] = "0".repeat(64).into();
        assert!(
            lock.validate_control_inventory(
                NativeEngineRole::Baseline,
                &serde_json::to_vec(&wrong).unwrap()
            )
            .is_err()
        );
        let mut oversized = f;
        let PalsEndpointLaunchV3::OnnxCuda(c) = &mut oversized.endpoints[0] else {
            unreachable!()
        };
        c.cuda_control.as_mut().unwrap().inventory.bytes = MAX_CONTROL_INVENTORY_BYTES + 1;
        assert!(oversized.lock().is_err());
    }
    #[test]
    fn pals_explicit_startup_probe_budget_preserves_default_canonical_and_fits_cold_readiness() {
        let (legacy, _) = cuda_fixture();
        let old = legacy.lock().unwrap();
        let old_json = serde_json::to_string(&legacy).unwrap();
        assert!(!old_json.contains("startup_probe_timeout_ms"));
        let mut explicit_none = json(old_json.as_bytes()).unwrap();
        explicit_none["endpoints"][0]["configuration"]["startup_probe_timeout_ms"] =
            serde_json::Value::Null;
        assert_eq!(
            PalsArenaLaunchV3::from_json(&explicit_none.to_string())
                .unwrap()
                .lock()
                .unwrap()
                .sha256(),
            old.sha256()
        );
        let mut explicit = legacy.clone();
        let PalsEndpointLaunchV3::OnnxCuda(cuda) = &mut explicit.endpoints[0] else {
            unreachable!()
        };
        cuda.startup_probe_timeout_ms = Some(120_000);
        // Declaring a 120s post-load probe does not silently enlarge the
        // existing 30s cold-model/process readiness window.
        assert!(explicit.lock().is_err());
        explicit.semantic_lock.manifest.pilot.handshake_max_ms = 180_000;
        explicit.semantic_lock = explicit.semantic_lock.manifest.lock().unwrap();
        let new = explicit.lock().unwrap();
        assert_ne!(old.sha256(), new.sha256());
        assert_eq!(new.declared_inputs(), old.declared_inputs());
        assert_eq!(new.view().timeouts.startup_ms, 180_000);
        assert_eq!(new.view().timeouts.handshake_ms, 180_000);
        assert_eq!(
            new.view().timeouts.runtime_ms,
            old.view().timeouts.runtime_ms
        );
        assert_eq!(new.view().timeouts.drain_ms, old.view().timeouts.drain_ms);
        let old_view = old.engine_view(NativeEngineRole::Baseline).unwrap();
        let new_view = new.engine_view(NativeEngineRole::Baseline).unwrap();
        assert_eq!(old_view.cuda_bundle, new_view.cuda_bundle);
        let old_engine = old_view.external.unwrap();
        let new_engine = new_view.external.unwrap();
        assert_eq!(old_engine.assets, new_engine.assets);
        assert_eq!(old_engine.environment, new_engine.environment);
        assert_eq!(
            new_engine.arguments[..old_engine.arguments.len()],
            old_engine.arguments
        );
        assert_eq!(
            &new_engine.arguments[old_engine.arguments.len()..],
            &["--pals-startup-probe-timeout-ms=120000".to_owned()]
        );
        for timeout in [0, MAX_STARTUP_PROBE_TIMEOUT_MS + 1, u64::MAX] {
            let mut bad = explicit.clone();
            let PalsEndpointLaunchV3::OnnxCuda(cuda) = &mut bad.endpoints[0] else {
                unreachable!()
            };
            cuda.startup_probe_timeout_ms = Some(timeout);
            assert!(bad.lock().is_err());
        }
        for timeout in [1, MAX_STARTUP_PROBE_TIMEOUT_MS] {
            let mut valid = explicit.clone();
            let PalsEndpointLaunchV3::OnnxCuda(cuda) = &mut valid.endpoints[0] else {
                unreachable!()
            };
            cuda.startup_probe_timeout_ms = Some(timeout);
            assert!(valid.lock().is_ok());
        }
        let mut cpu_json =
            json(serde_json::to_string(&native_fixture()).unwrap().as_bytes()).unwrap();
        cpu_json["endpoints"][0]["configuration"]["startup_probe_timeout_ms"] = 120_000.into();
        assert!(PalsArenaLaunchV3::from_json(&cpu_json.to_string()).is_err());
    }
    #[test]
    fn pals_startup_probe_budget_requires_actual_selected_value_without_reinterpreting_nn_work() {
        let (mut declared, _) = cuda_fixture();
        declared.semantic_lock.manifest.pilot.handshake_max_ms = 180_000;
        declared.semantic_lock = declared.semantic_lock.manifest.lock().unwrap();
        let PalsEndpointLaunchV3::OnnxCuda(cuda) = &mut declared.endpoints[0] else {
            unreachable!()
        };
        cuda.startup_probe_timeout_ms = Some(120_000);
        let lock = declared.lock().unwrap();
        let (mut start, mut end) = cuda_records_fixture(&lock);
        for record in [&mut start, &mut end] {
            record["native"]["execution"]["startup_probe_timeout_ms"] = 120_000.into();
        }
        let validate = |s: &serde_json::Value, t: &serde_json::Value| {
            validate_pals_native_records(
                &lock,
                NativeEngineRole::Baseline,
                &serde_json::to_vec(s).unwrap(),
                &serde_json::to_vec(t).unwrap(),
                "native-process-100",
            )
        };
        let (observed, _) = validate(&start, &end).unwrap();
        assert_eq!(observed.startup_probe_timeout_ms, Some(120_000));
        assert_eq!(
            (
                observed.startup_nn_inputs_completed,
                observed.completed_role_inputs,
                observed.search_consumed_role_inputs,
            ),
            (3, 3, 1)
        );
        for wrong in [
            serde_json::Value::Null,
            119_999.into(),
            0.into(),
            u64::MAX.into(),
            "120000".into(),
        ] {
            let mut a = start.clone();
            let mut b = end.clone();
            for record in [&mut a, &mut b] {
                record["native"]["execution"]["startup_probe_timeout_ms"] = wrong.clone();
            }
            assert!(validate(&a, &b).is_err());
        }
        let mut changed = end.clone();
        changed["native"]["execution"]["startup_probe_timeout_ms"] = 120_001.into();
        assert!(validate(&start, &changed).is_err());
        let (legacy, _) = cuda_fixture();
        let old = legacy.lock().unwrap();
        let old_cuda = legacy.endpoints[0].cuda_model().unwrap();
        assert!(validate_native_startup_budget(old_cuda, &start["native"]["execution"]).is_err());
        let (old_start, old_end) = cuda_records_fixture(&old);
        let (audit, _) = validate_pals_native_records(
            &old,
            NativeEngineRole::Baseline,
            &serde_json::to_vec(&old_start).unwrap(),
            &serde_json::to_vec(&old_end).unwrap(),
            "native-process-100",
        )
        .unwrap();
        assert_eq!(audit.startup_probe_timeout_ms, None);
    }
    #[test]
    fn pals_experimental_loading_binding_preserves_legacy_canonical_and_all_binary_pins() {
        let (legacy, _) = cuda_control_fixture();
        let old = legacy.lock().unwrap();
        let old_json = serde_json::to_string(&legacy).unwrap();
        assert!(!old_json.contains("loading_profile"));
        let mut explicit_none = json(old_json.as_bytes()).unwrap();
        explicit_none["endpoints"][0]["configuration"]["cuda_control"]["loading_profile"] =
            serde_json::Value::Null;
        assert_eq!(
            PalsArenaLaunchV3::from_json(&explicit_none.to_string())
                .unwrap()
                .lock()
                .unwrap()
                .sha256(),
            old.sha256()
        );
        let binding = loading_binding_fixture();
        assert_eq!(
            binding.canonical_sha256,
            "3084f27e678d6a6750713265bfb921e90687f199fb71484ff2c475000982ee6d"
        );
        let mut explicit = legacy.clone();
        let PalsEndpointLaunchV3::OnnxCuda(cuda) = &mut explicit.endpoints[0] else {
            unreachable!()
        };
        cuda.cuda_control.as_mut().unwrap().loading_profile = Some(binding.clone());
        let new = explicit.lock().unwrap();
        assert_ne!(new.sha256(), old.sha256());
        assert_eq!(new.input.semantic_lock, old.input.semantic_lock);
        assert_eq!(new.declared_inputs(), old.declared_inputs());
        let old_view = old.engine_view(NativeEngineRole::Baseline).unwrap();
        let new_view = new.engine_view(NativeEngineRole::Baseline).unwrap();
        assert_eq!(new_view.cuda_bundle, old_view.cuda_bundle);
        let old_engine = old_view.external.unwrap();
        let new_engine = new_view.external.unwrap();
        assert_eq!(new_engine.assets, old_engine.assets);
        assert_eq!(new_engine.environment, old_engine.environment);
        assert_eq!(
            new_engine.arguments[..old_engine.arguments.len()],
            old_engine.arguments
        );
        assert_eq!(
            &new_engine.arguments[old_engine.arguments.len()..],
            &[
                "--pals-cuda-loading-profile=experimental-cudnn-shim-lazy-v1".to_owned(),
                format!(
                    "--pals-cuda-loading-profile-sha256={}",
                    binding.canonical_sha256
                )
            ]
        );
        let mut bad = explicit.clone();
        let PalsEndpointLaunchV3::OnnxCuda(cuda) = &mut bad.endpoints[0] else {
            unreachable!()
        };
        cuda.cuda_control
            .as_mut()
            .unwrap()
            .loading_profile
            .as_mut()
            .unwrap()
            .canonical_sha256 = "f".repeat(64);
        assert!(bad.lock().is_err());
        let mut unsupported = json(serde_json::to_string(&explicit).unwrap().as_bytes()).unwrap();
        unsupported["endpoints"][0]["configuration"]["cuda_control"]["loading_profile"]["profile"] =
            "automatic-fallback".into();
        assert!(PalsArenaLaunchV3::from_json(&unsupported.to_string()).is_err());
    }
    #[test]
    fn pals_loading_mapping_is_actual_bounded_origin_evidence_not_invented_nn_work() {
        let (mut f, _) = cuda_control_fixture();
        let binding = loading_binding_fixture();
        let PalsEndpointLaunchV3::OnnxCuda(cuda) = &mut f.endpoints[0] else {
            unreachable!()
        };
        cuda.cuda_control.as_mut().unwrap().loading_profile = Some(binding.clone());
        let lock = f.lock().unwrap();
        let cuda = lock.input.endpoints[0].cuda_model().unwrap();
        let (mut start, mut end) = cuda_records_fixture(&lock);
        attach_control_witness(&lock, &mut start, &mut end);
        let execution = serde_json::json!({"profile":binding.profile,"canonical_sha256":hash_array(&binding.canonical_sha256)});
        for record in [&mut start, &mut end] {
            record["native"]["execution"]["cuda_loading_profile"] = execution.clone();
            record["native"]["startup_probe"]["runtime_loading_mapping"] =
                mapping_fixture(cuda, &binding, &[]);
        }
        end["native"]["final_runtime_loading_mapping"] = mapping_fixture(cuda, &binding, &[10, 11]);
        let validate = |s: &serde_json::Value, t: &serde_json::Value| {
            validate_pals_native_records(
                &lock,
                NativeEngineRole::Baseline,
                &serde_json::to_vec(s).unwrap(),
                &serde_json::to_vec(t).unwrap(),
                "native-process-100",
            )
        };
        let (audit, _) = validate(&start, &end).unwrap();
        let loading = audit.cuda_loading.as_ref().unwrap();
        assert_eq!(
            (
                loading.startup.mapped_nvidia_files.len(),
                loading.final_mapping.mapped_nvidia_files.len()
            ),
            (9, 11)
        );
        assert_eq!(
            (
                audit.startup_nn_inputs_completed,
                audit.completed_role_inputs,
                audit.search_consumed_role_inputs
            ),
            (3, 3, 1)
        );
        let mut wrong = end.clone();
        wrong["native"]["final_runtime_loading_mapping"] = serde_json::Value::Null;
        assert!(validate(&start, &wrong).is_err());
        let mut wrong = end.clone();
        wrong["native"]["final_runtime_loading_mapping"]["mapped_ort_files"] =
            serde_json::json!([]);
        assert!(validate(&start, &wrong).is_err());
        let mut wrong = end.clone();
        wrong["native"]["final_runtime_loading_mapping"]["mapped_nvidia_files"][0] =
            "untrusted.so".into();
        assert!(validate(&start, &wrong).is_err());
        let mut wrong = end.clone();
        wrong["native"]["final_runtime_loading_mapping"]["required_nvidia_files"][0] =
            rz_native_loader::NVIDIA_LOAD_ORDER[8].into();
        assert!(validate(&start, &wrong).is_err());
        let mut wrong = end.clone();
        wrong["native"]["final_runtime_loading_mapping"]["loading_profile_sha256"] =
            hash_array(&"f".repeat(64)).into();
        assert!(validate(&start, &wrong).is_err());
        let (old, _) = cuda_control_fixture();
        let old_cuda = old.endpoints[0].cuda_model().unwrap();
        assert!(
            validate_native_loading_evidence(
                old_cuda,
                &start["native"]["execution"],
                &start["native"]["startup_probe"],
                &end["native"]["final_runtime_loading_mapping"]
            )
            .is_err()
        );
        assert!(
            validate_native_loading_evidence(
                old_cuda,
                &serde_json::json!({}),
                &serde_json::json!({}),
                &serde_json::Value::Null
            )
            .unwrap()
            .is_none()
        );
    }
    #[test]
    fn pals_cuda_control_witness_requires_exact_source_and_both_selected_branches() {
        let (f, _) = cuda_control_fixture();
        let lock = f.lock().unwrap();
        let (mut start, mut end) = cuda_records_fixture(&lock);
        let validate = |s: &serde_json::Value, t: &serde_json::Value| {
            validate_pals_native_records(
                &lock,
                NativeEngineRole::Baseline,
                &serde_json::to_vec(s).unwrap(),
                &serde_json::to_vec(t).unwrap(),
                "native-process-100",
            )
        };
        assert!(validate(&start, &end).is_err()); // Origin/probe alone cannot authorize control mode.
        attach_control_witness(&lock, &mut start, &mut end);
        let (audit, _) = validate(&start, &end).unwrap();
        let mut work = work_fixture("pals");
        work["pals"]["consumed_role_outputs"] = 1.into();
        work["pals"]["completed_proposer_calls"] = 1.into();
        let work = PalsProcessWorkAuditV3 {
            endpoint_id: "pals".into(),
            process_id: 100,
            startup_sha256: "a".repeat(64),
            termination_sha256: "b".repeat(64),
            startup_bytes: 100,
            termination_bytes: 200,
            search_work: work,
        };
        let core = endpoint_work_receipt(
            &lock,
            NativeEngineRole::Baseline,
            Some(&work),
            Some(&audit),
            &preflight_fixture("pals"),
        )
        .unwrap();
        assert!(
            matches!(core.gpu_device,PalsObservedV3::Observed{ref value,..} if value=="cuda:0")
        );
        assert_eq!(core.vram_peak_bytes, PalsObservedV3::Unknown);
        assert_eq!(core.nn_inputs_completed, 4); // Three startup inputs are still subtracted.
        for field in [
            "inventory_sha256",
            "manifest_sha256",
            "runtime_sha256",
            "runtime_bundle_sha256",
        ] {
            let mut s = start.clone();
            let mut t = end.clone();
            s["native"]["startup_probe"]["cuda_placement_witness"][field] =
                hash_array(&"0".repeat(64)).into();
            t["native"]["startup_probe"] = s["native"]["startup_probe"].clone();
            assert!(validate(&s, &t).is_err());
        }
        let mut s = start.clone();
        let mut t = end.clone();
        s["native"]["startup_probe"]["cuda_placement_witness"]["optimization"] = "level1".into();
        t["native"]["startup_probe"] = s["native"]["startup_probe"].clone();
        assert!(validate(&s, &t).is_err());
        let mut s = start.clone();
        let mut t = end.clone();
        s["native"]["startup_probe"]["cuda_placement_witness"]["shared_pc"]["cuda_transfer_kernels"] =
            1.into();
        t["native"]["startup_probe"] = s["native"]["startup_probe"].clone();
        assert!(validate(&s, &t).is_err()); // No source transfer was registered in this fixture.
        let mut s = start.clone();
        let mut t = end.clone();
        s["native"]["startup_probe"]["cuda_placement_witness"]["shared_pc"]["critic_private_kernels"] =
            0.into();
        t["native"]["startup_probe"] = s["native"]["startup_probe"].clone();
        assert!(validate(&s, &t).is_err());
        let mut forged = audit;
        forged.cuda_placement = None;
        assert!(
            endpoint_work_receipt(
                &lock,
                NativeEngineRole::Baseline,
                Some(&work),
                Some(&forged),
                &preflight_fixture("pals")
            )
            .is_err()
        );
    }
    #[test]
    fn pals_cuda_startup_work_is_not_search_work_or_cuda_placement_evidence() {
        let (f, _) = cuda_fixture();
        let lock = f.lock().unwrap();
        let (start, end) = cuda_records_fixture(&lock);
        let validate = |s: &serde_json::Value, t: &serde_json::Value| {
            validate_pals_native_records(
                &lock,
                NativeEngineRole::Baseline,
                &serde_json::to_vec(s).unwrap(),
                &serde_json::to_vec(t).unwrap(),
                "native-process-100",
            )
        };
        let (audit, pid) = validate(&start, &end).unwrap();
        assert_eq!(pid, 100);
        assert_eq!(
            (
                audit.startup_nn_inputs_completed,
                audit.startup_nn_calls_completed,
                audit.startup_role_inputs_completed
            ),
            (3, 3, 2)
        );
        assert_eq!(
            (
                audit.completed_role_inputs,
                audit.search_consumed_role_inputs
            ),
            (3, 1)
        );
        let mut work = work_fixture("pals");
        work["pals"]["consumed_role_outputs"] = 1.into();
        work["pals"]["completed_proposer_calls"] = 1.into();
        let search_work = PalsProcessWorkAuditV3 {
            endpoint_id: "pals".into(),
            process_id: 100,
            startup_sha256: "a".repeat(64),
            termination_sha256: "b".repeat(64),
            startup_bytes: 100,
            termination_bytes: 200,
            search_work: work,
        };
        let receipt = endpoint_work_receipt(
            &lock,
            NativeEngineRole::Baseline,
            Some(&search_work),
            Some(&audit),
            &preflight_fixture("pals"),
        )
        .unwrap();
        assert_eq!(
            (
                receipt.nn_inputs_completed,
                receipt.nn_calls_completed,
                receipt.nn_inputs_consumed
            ),
            (4, 4, 1)
        );
        assert_eq!(receipt.gpu_device, PalsObservedV3::Unknown);
        let mut incomplete_snapshot = audit.clone();
        incomplete_snapshot.raw_native["backend_stats"] = stats_fixture(0, 1);
        assert!(
            endpoint_work_receipt(
                &lock,
                NativeEngineRole::Baseline,
                Some(&search_work),
                Some(&incomplete_snapshot),
                &preflight_fixture("pals")
            )
            .is_err()
        );
        let mut failed_observer = end.clone();
        failed_observer["native"]["observer_failures"] = 1.into();
        assert!(validate(&start, &failed_observer).is_err());
        let mut bad = end.clone();
        bad["runtime_bundle_sha256"] = "f".repeat(64).into();
        assert!(validate(&start, &bad).is_err());
        let mut bad_start = start.clone();
        let mut bad_end = end.clone();
        bad_start["native"]["startup_probe"]["runtime_mapping_confirmed"] = false.into();
        bad_end["native"]["startup_probe"] = bad_start["native"]["startup_probe"].clone();
        assert!(validate(&bad_start, &bad_end).is_err());
        let mut bad_start = start.clone();
        let mut bad_end = end;
        bad_start["native"]["execution"]["transient_execution_device_bytes"] = (3u64 << 30).into();
        bad_end["native"]["execution"] = bad_start["native"]["execution"].clone();
        assert!(validate(&bad_start, &bad_end).is_err());
    }
    #[test]
    fn pals_launch_rejects_unsupported_options_small_tt_and_secret_inputs() {
        let mut f = fixture();
        let PalsEngineV3::Pals(e) = &mut f.semantic_lock.manifest.engines[0] else {
            unreachable!()
        };
        e.search.options.insert("beam".into(), "100".into());
        f.semantic_lock = f.semantic_lock.manifest.lock().unwrap();
        assert!(f.lock().is_err());
        let mut f = fixture();
        let PalsEngineV3::OwnCpu(e) = &mut f.semantic_lock.manifest.engines[1] else {
            unreachable!()
        };
        e.cpu.max_tt_bytes = 1;
        f.semantic_lock = f.semantic_lock.manifest.lock().unwrap();
        assert!(f.lock().is_err());
        let mut f = fixture();
        f.opening_artifact.path = "secret/ssh.key".into();
        assert!(f.lock().is_err());
        let mut f = native_fixture();
        let PalsEndpointLaunchV3::OnnxCpu(n) = &mut f.endpoints[0] else {
            unreachable!()
        };
        n.export.bytes = 64 * 1024 + 1;
        assert!(f.lock().is_err());
    }
    #[test]
    fn pals_launch_rejects_same_source_graph_in_two_export_namespaces() {
        let mut f = native_fixture();
        let mut e = f.semantic_lock.manifest.engines[0].clone();
        let PalsEngineV3::Pals(p) = &mut e else {
            unreachable!()
        };
        p.id = "second-pals".into();
        f.semantic_lock.manifest.engines[1] = e;
        f.semantic_lock.manifest.pilot.white_order[1] = "second-pals".into();
        f.semantic_lock.manifest.declared_changes.clear();
        f.semantic_lock = f.semantic_lock.manifest.lock().unwrap();
        f.endpoints[1] = f.endpoints[0].clone();
        let PalsEndpointLaunchV3::OnnxCpu(n) = &mut f.endpoints[1] else {
            unreachable!()
        };
        n.export.path = "model/second-export.json".into();
        n.export.sha256 = "e".repeat(64);
        assert!(f.lock().is_err());
    }
    #[test]
    fn pals_native_digest_and_resource_parsing_fail_closed() {
        assert!(array_hash(&serde_json::json!([0, 1])).is_err());
        assert!(
            json(br#"{"native":{"completed_role_inputs":1,"completed_role_inputs":2}}"#).is_err()
        );
        assert_eq!(parse_pals_affinity("2,0").unwrap(), vec![0, 2]);
        assert_eq!(parse_pals_affinity("0-2,4").unwrap(), vec![0, 1, 2, 4]);
        for v in ["", "0,0", "2-1", "0-1000", "0-2,1"] {
            assert!(parse_pals_affinity(v).is_err());
        }
    }
    #[test]
    fn pals_native_records_require_exact_identity_fence_and_real_consumption() {
        let lock = native_fixture().lock().unwrap();
        let graph_hash = [170u8; 32];
        let encoding_hash = [221u8; 32];
        let adapter_hash = [187u8; 32];
        let zero = serde_json::json!({"physically_completed_role_calls":0,"completed_role_inputs":0,"failed_physical_role_calls":0,"invalid_role_outputs":0,"delivered_role_inputs":0,"search_consumed_role_inputs":0,"canceled_requests":0,"expired_requests":0,"completed_new_game_resets":0,"process_epoch":1,"game_generation":0,"request_high_water":0,"execution_high_water":0,"physical_runs_in_flight":0,"quarantined":false,"physical_shutdown_confirmed":false,"native_buffers_released":false,"last_failure":null,
            "model_epoch":graph_hash,"export_manifest_sha256":graph_hash,"encoding_semantic_sha256":encoding_hash,"adapter_source_sha256":adapter_hash,"trained":false,
            "frozen_epoch":1,"residency":{"native_sessions":3,"role_reader_weights_shared":false,"native_resident_parameter_bytes":null,"vram_peak_bytes":null,"graphs":[{"role":"public","sha256":graph_hash,"serialized_bytes":1000},{"role":"proposer","sha256":graph_hash,"serialized_bytes":1000},{"role":"critic","sha256":graph_hash,"serialized_bytes":1000}]}});
        let start = serde_json::json!({"schema_version":3,"domain":PALS_NATIVE_STARTUP_V3_DOMAIN,"endpoint_id":"pals","launch_sha256":lock.sha256(),"process_id":100,"binary_sha256":"a".repeat(64),"runtime_sha256":"a".repeat(64),"provider":"cpu","precision":"fp32","service_exit_success":false,"native":zero});
        let mut end = start.clone();
        end["domain"] = PALS_NATIVE_TERMINATION_V3_DOMAIN.into();
        end["service_exit_success"] = true.into();
        for (key, value) in [
            ("physically_completed_role_calls", 3u64),
            ("completed_role_inputs", 3),
            ("delivered_role_inputs", 2),
            ("search_consumed_role_inputs", 1),
            ("completed_new_game_resets", 1),
            ("game_generation", 1),
            ("request_high_water", 5),
            ("execution_high_water", 3),
        ] {
            end["native"][key] = value.into();
        }
        end["native"]["physical_shutdown_confirmed"] = true.into();
        end["native"]["native_buffers_released"] = true.into();
        let validate = |v: &serde_json::Value| {
            validate_pals_native_records(
                &lock,
                NativeEngineRole::Baseline,
                &serde_json::to_vec(&start).unwrap(),
                &serde_json::to_vec(v).unwrap(),
                "native-process-100",
            )
        };
        let (audit, pid) = validate(&end).unwrap();
        assert_eq!(pid, 100);
        assert_eq!(audit.completed_role_inputs, 3);
        assert_eq!(audit.search_consumed_role_inputs, 1);
        let validate_epoch_pair = |s: &serde_json::Value, t: &serde_json::Value| {
            validate_pals_native_records(
                &lock,
                NativeEngineRole::Baseline,
                &serde_json::to_vec(s).unwrap(),
                &serde_json::to_vec(t).unwrap(),
                "native-process-100",
            )
        };
        // Actual native domain issuance and later owners are not epoch 1 or
        // PID 100. Legacy synthetic epoch 1 above remains a valid wire.
        let domain = 0x5041_4c53_0000_0000u64;
        let mut epoch_start = start.clone();
        let mut epoch_end = end.clone();
        for epoch in [domain, domain + 8] {
            epoch_start["native"]["process_epoch"] = epoch.into();
            epoch_end["native"]["process_epoch"] = epoch.into();
            let (observed, pid) = validate_epoch_pair(&epoch_start, &epoch_end).unwrap();
            assert_eq!(pid, 100);
            assert_eq!(observed.raw_native["process_epoch"], epoch);
        }
        epoch_start["native"]["process_epoch"] = 0.into();
        epoch_end["native"]["process_epoch"] = 0.into();
        assert!(validate_epoch_pair(&epoch_start, &epoch_end).is_err());
        epoch_start["native"]["process_epoch"] = domain.into();
        epoch_end["native"]["process_epoch"] = (domain + 1).into();
        assert!(validate_epoch_pair(&epoch_start, &epoch_end).is_err());
        epoch_start["native"]["process_epoch"] = "forged counter".into();
        epoch_end["native"]["process_epoch"] = "forged counter".into();
        assert!(validate_epoch_pair(&epoch_start, &epoch_end).is_err());
        epoch_start["native"]
            .as_object_mut()
            .unwrap()
            .remove("process_epoch");
        epoch_end["native"]
            .as_object_mut()
            .unwrap()
            .remove("process_epoch");
        assert!(validate_epoch_pair(&epoch_start, &epoch_end).is_err());
        epoch_start["native"]["process_epoch"] = domain.into();
        epoch_end["native"]["process_epoch"] = domain.into();
        epoch_end["process_id"] = 101.into();
        assert!(validate_epoch_pair(&epoch_start, &epoch_end).is_err());
        epoch_start["process_id"] = 101.into();
        assert!(validate_epoch_pair(&epoch_start, &epoch_end).is_err());
        assert!(lock.input.require_actual_native_epoch().is_ok());
        let mut historical = native_fixture();
        let PalsEngineV3::Pals(e) = &mut historical.semantic_lock.manifest.engines[0] else {
            unreachable!()
        };
        e.model.frozen_epoch = 0;
        historical.semantic_lock = historical.semantic_lock.manifest.lock().unwrap();
        let old = historical.lock().unwrap();
        assert!(old.input.require_actual_native_epoch().is_err());
        let mut old_start = start.clone();
        let mut old_end = end.clone();
        old_start["launch_sha256"] = old.sha256().into();
        old_end["launch_sha256"] = old.sha256().into();
        old_start["native"]
            .as_object_mut()
            .unwrap()
            .remove("frozen_epoch");
        old_end["native"]
            .as_object_mut()
            .unwrap()
            .remove("frozen_epoch");
        assert!(
            validate_pals_native_records(
                &old,
                NativeEngineRole::Baseline,
                &serde_json::to_vec(&old_start).unwrap(),
                &serde_json::to_vec(&old_end).unwrap(),
                "native-process-100"
            )
            .is_ok()
        );
        assert_eq!(
            LockedPalsArenaLaunchV3::from_json(&old.to_json().unwrap())
                .unwrap()
                .sha256(),
            old.sha256()
        );
        let mut missing_start = start.clone();
        let mut missing_end = end.clone();
        missing_start["native"]
            .as_object_mut()
            .unwrap()
            .remove("frozen_epoch");
        missing_end["native"]
            .as_object_mut()
            .unwrap()
            .remove("frozen_epoch");
        assert!(
            validate_pals_native_records(
                &lock,
                NativeEngineRole::Baseline,
                &serde_json::to_vec(&missing_start).unwrap(),
                &serde_json::to_vec(&missing_end).unwrap(),
                "native-process-100"
            )
            .is_err()
        );
        let mut bad = end.clone();
        bad["native"]["search_consumed_role_inputs"] = 4.into();
        assert!(validate(&bad).is_err());
        let mut bad = end.clone();
        bad["native"]["physical_runs_in_flight"] = 1.into();
        assert!(validate(&bad).is_err());
        let mut bad = end.clone();
        bad["native"]["physical_shutdown_confirmed"] = false.into();
        assert!(validate(&bad).is_err());
        let mut bad = end.clone();
        bad["native"]["quarantined"] = true.into();
        assert!(validate(&bad).is_err());
        let mut bad = end.clone();
        bad["runtime_sha256"] = "f".repeat(64).into();
        assert!(validate(&bad).is_err());
        let mut bad = end.clone();
        bad["native"]["execution"] = serde_json::json!({"provider":"cuda"});
        assert!(validate(&bad).is_err());
        let mut bad = end;
        bad["native"]["execution"] = "not an object".into();
        assert!(validate(&bad).is_err());
    }
    fn select_post_repair_recheck(input: PalsArenaLaunchV3) -> PalsArenaLaunchV3 {
        select_refinement(input, PalsPostRepairRecheckPolicyV3::SameRepairedLineOnceV1)
    }
    fn select_refinement(
        mut input: PalsArenaLaunchV3,
        selected: PalsPostRepairRecheckPolicyV3,
    ) -> PalsArenaLaunchV3 {
        let PalsEngineV3::Pals(p) = &mut input.semantic_lock.manifest.engines[0] else {
            unreachable!()
        };
        p.search.semantic_id = selected.version().into();
        p.search.options =
            BTreeMap::from([("post_repair_recheck".into(), selected.option_value().into())]);
        match &mut input.endpoints[0] {
            PalsEndpointLaunchV3::LegalOrderMock(s) => s.post_repair_recheck = Some(selected),
            PalsEndpointLaunchV3::OnnxCpu(n) => n.search.post_repair_recheck = Some(selected),
            PalsEndpointLaunchV3::OnnxCuda(n) => {
                n.model.search.post_repair_recheck = Some(selected)
            }
            _ => unreachable!(),
        }
        input.semantic_lock = input.semantic_lock.manifest.lock().unwrap();
        input
    }

    #[test]
    fn actual_c_continuation_contract_pin_and_launch_are_distinct_for_mock_cpu_and_cuda() {
        use rz_search::pals::engine::PostRepairRecheckPolicy;
        let actual = PostRepairRecheckPolicy::ActualOpponentContinuationV1;
        let declaration = PalsPostRepairRecheckPolicyV3::ActualOpponentContinuationV1;
        let expected = declaration.expected_identity();
        assert_eq!(
            Some(expected.version.as_str()),
            actual.registration_version()
        );
        assert_eq!(Some(expected.policy.as_str()), actual.registration_policy());
        assert_eq!(expected.search_identity, actual.search_identity());
        assert_eq!(
            expected.conditions_sha256,
            <[u8; 32]>::from(Sha256::digest(actual.conditions().unwrap().as_bytes()))
        );
        assert_eq!(actual.conditions().unwrap().len(), 468);
        expected.validate().unwrap();
        for old in [fixture(), native_fixture(), cuda_fixture().0] {
            let legacy = old.clone().lock().unwrap();
            let input = select_refinement(old, declaration);
            let selected = input.clone().lock().unwrap();
            let args = &selected.endpoint_views[0].arguments;
            assert_eq!(
                args.last().unwrap(),
                "--pals-post-repair-recheck=actual-opponent-continuation-v1"
            );
            assert_eq!(
                &args[..args.len() - 1],
                legacy.endpoint_views[0].arguments.as_slice()
            );
            assert_eq!(
                args.iter()
                    .filter(|arg| arg.starts_with("--pals-post-repair-recheck="))
                    .count(),
                1
            );
            assert_ne!(selected.sha256(), legacy.sha256());
            let mut mismatch = input;
            let PalsEngineV3::Pals(p) = &mut mismatch.semantic_lock.manifest.engines[0] else {
                unreachable!()
            };
            p.search.semantic_id = PalsPostRepairRecheckPolicyV3::SameRepairedLineOnceV1
                .version()
                .into();
            p.search.options.insert(
                "post_repair_recheck".into(),
                PalsPostRepairRecheckPolicyV3::SameRepairedLineOnceV1
                    .option_value()
                    .into(),
            );
            mismatch.semantic_lock = mismatch.semantic_lock.manifest.lock().unwrap();
            assert!(mismatch.lock().is_err());
        }
    }

    #[test]
    fn post_repair_contract_pin_matches_actual_engine_identity_and_conditions() {
        let actual = rz_search::pals::engine::PostRepairRecheckPolicy::SameRepairedLineOnceV1;
        let expected = PalsSearchPolicyIdentityV3::expected_same_repaired_line_once_v1();
        assert_eq!(
            expected.version,
            rz_experiments::PALS_POST_REPAIR_RECHECK_V3_VERSION
        );
        assert_eq!(
            expected.policy,
            rz_experiments::PALS_POST_REPAIR_RECHECK_V3_POLICY
        );
        assert_eq!(
            actual.search_identity(),
            rz_search::pals::engine::POST_REPAIR_RECHECK_SEARCH_VERSION
        );
        assert_eq!(expected.search_identity, actual.search_identity());
        assert_eq!(
            expected.search_identity,
            rz_experiments::PALS_POST_REPAIR_RECHECK_V3_SEARCH_IDENTITY
        );
        let conditions = actual.conditions().unwrap();
        assert_eq!(
            conditions,
            rz_search::pals::engine::POST_REPAIR_RECHECK_CONDITIONS
        );
        assert_eq!(conditions.len(), 425);
        let actual_digest: [u8; 32] = Sha256::digest(conditions.as_bytes()).into();
        assert_eq!(
            actual_digest,
            rz_experiments::PALS_POST_REPAIR_RECHECK_V3_CONDITIONS_SHA256
        );
        assert_eq!(expected.conditions_sha256, actual_digest);
        expected.validate().unwrap();
    }

    #[test]
    fn post_repair_recipe_omission_preserves_legacy_and_selected_flag_is_last_once() {
        for old in [fixture(), native_fixture(), cuda_fixture().0] {
            let old_text = serde_json::to_string(&old).unwrap();
            assert!(!old_text.contains("post_repair_recheck"));
            let old_digest = crate::canonical_sha256(&old).unwrap();
            let roundtrip = PalsArenaLaunchV3::from_json(&old_text).unwrap();
            assert_eq!(crate::canonical_sha256(&roundtrip).unwrap(), old_digest);
            let legacy = old.clone().lock().unwrap();
            let selected_input = select_post_repair_recheck(old.clone());
            let selected = selected_input.clone().lock().unwrap();
            let args = &selected.endpoint_views[0].arguments;
            assert_eq!(
                args.last().unwrap(),
                "--pals-post-repair-recheck=same-repaired-line-once-v1"
            );
            assert_eq!(
                &args[..args.len() - 1],
                legacy.endpoint_views[0].arguments.as_slice()
            );
            assert_eq!(
                args.iter()
                    .filter(|a| a.starts_with("--pals-post-repair-recheck="))
                    .count(),
                1
            );
            assert!(args.len() <= 32);
            assert_eq!(
                selected.endpoint_views[0].assets,
                legacy.endpoint_views[0].assets
            );
            assert_ne!(selected.sha256(), legacy.sha256());
            let mut missing_recipe = selected_input;
            match &mut missing_recipe.endpoints[0] {
                PalsEndpointLaunchV3::LegalOrderMock(s) => s.post_repair_recheck = None,
                PalsEndpointLaunchV3::OnnxCpu(n) => n.search.post_repair_recheck = None,
                PalsEndpointLaunchV3::OnnxCuda(n) => n.model.search.post_repair_recheck = None,
                _ => unreachable!(),
            }
            assert!(missing_recipe.lock().is_err());
            let mut missing_manifest = old;
            match &mut missing_manifest.endpoints[0] {
                PalsEndpointLaunchV3::LegalOrderMock(s) => {
                    s.post_repair_recheck =
                        Some(PalsPostRepairRecheckPolicyV3::SameRepairedLineOnceV1)
                }
                PalsEndpointLaunchV3::OnnxCpu(n) => {
                    n.search.post_repair_recheck =
                        Some(PalsPostRepairRecheckPolicyV3::SameRepairedLineOnceV1)
                }
                PalsEndpointLaunchV3::OnnxCuda(n) => {
                    n.model.search.post_repair_recheck =
                        Some(PalsPostRepairRecheckPolicyV3::SameRepairedLineOnceV1)
                }
                _ => unreachable!(),
            }
            assert!(missing_manifest.lock().is_err());
        }
        let recipe = match fixture().endpoints[0].clone() {
            PalsEndpointLaunchV3::LegalOrderMock(s) => s,
            _ => unreachable!(),
        };
        let raw = serde_json::to_value(recipe).unwrap();
        assert!(raw.get("post_repair_recheck").is_none());
        for invalid in [
            serde_json::Value::Null,
            true.into(),
            1.into(),
            "disabled".into(),
            "same_repaired_line_once_v1".into(),
        ] {
            let mut bad = raw.clone();
            bad["post_repair_recheck"] = invalid;
            assert!(
                crate::decode_json::<PalsSearchLaunchV3>(&serde_json::to_string(&bad).unwrap())
                    .is_err()
            );
        }
        assert!(crate::decode_json::<PalsSearchLaunchV3>(r#"{"max_rounds":16,"max_cpu_nodes":100000,"cpu_depth":2,"max_situations":4096,"post_repair_recheck":"same-repaired-line-once-v1","post_repair_recheck":"same-repaired-line-once-v1"}"#).is_err());
    }

    #[test]
    fn post_repair_maximum_registered_own_recipe_keeps_the_original_argv_cap() {
        // All optional argv-producing Own CUDA declarations selected. This is
        // metadata-only; external CPU_R is incompatible with the new lane.
        let (mut input, _) = cuda_control_fixture();
        let PalsEndpointLaunchV3::OnnxCuda(cuda) = &mut input.endpoints[0] else {
            unreachable!()
        };
        cuda.cuda_control.as_mut().unwrap().loading_profile = Some(loading_binding_fixture());
        cuda.startup_probe_timeout_ms = Some(30_000);
        let legacy = input.clone().lock().unwrap();
        let selected = select_post_repair_recheck(input).lock().unwrap();
        let args = &selected.endpoint_views[0].arguments;
        assert_eq!(legacy.endpoint_views[0].arguments.len(), 22);
        assert_eq!(args.len(), 23);
        assert!(args.len() <= 32);
        assert_eq!(&args[..22], legacy.endpoint_views[0].arguments.as_slice());
        assert_eq!(
            args[22],
            "--pals-post-repair-recheck=same-repaired-line-once-v1"
        );
        assert!(
            args.iter()
                .any(|a| a == "--pals-startup-probe-timeout-ms=30000")
        );
        assert!(
            args.iter()
                .any(|a| a.starts_with("--pals-cuda-loading-profile="))
        );
        assert_eq!(
            selected.endpoint_views[0].assets,
            legacy.endpoint_views[0].assets
        );
    }

    fn work_fixture(kind: &str) -> serde_json::Value {
        let totals = if kind == "pals" {
            serde_json::json!({"completed_proposer_calls":0,"completed_repair_calls":0,"completed_critic_calls":0,"consumed_role_outputs":0,
                "cpu_tasks_requested":0,"completed_cpu_tasks":0,"reused_completed_cpu_tasks_consumed":0,
                "consumed_cpu_tasks":0,"cpu_nodes":0,"consumed_cached_cpu_values":0,"unknown_root_children":null})
        } else {
            serde_json::json!({"tasks_requested":0,"requested_depth_completed":0,"rules_terminal_reports":0,
                "completed_reports_accepted_for_uci_output":0,"nodes":0,"max_completed_depth":null})
        };
        let mut work = serde_json::json!({"schema_version":1,"search_kind":kind,"go_invocations":0,"successful_returns":0,
            "failed_returns":0,"canceled_returns":0,"deadline_returns":0,"physical_unknown_returns":0,"active_invocations":0,
            "unobserved_work_invocations":0,"cpu":null,"pals":null});
        work[kind] = totals;
        work
    }
    fn preflight_fixture(id: &str) -> crate::ExternalUciPreflight {
        let process: crate::ProcessReceipt = serde_json::from_value(serde_json::json!({"supervisor_version":1,"pid":1,
            "elapsed_ns":1,"stop":"exited","exit_code":0,"exit_signal":null,"group_cleanup":"gone","stdout_bytes":1,"stderr_bytes":0,
            "observed_output_bytes":1,"descendant_cleanup_required":false,"child_limit_enforcement":"fixture-only",
            "watched_artifact_bytes":1,"artifact_limit_enforcement":null,"errors":[]})).unwrap();
        crate::ExternalUciPreflight {
            schema_version: 1,
            engine_id: id.into(),
            advertisement: crate::parse_uci_advertisement(b"id name fixture\nuciok\n").unwrap(),
            options: BTreeMap::new(),
            environment: None,
            identification_process: process.clone(),
            readiness_process: process,
            stop_and_legal_bestmove_observed: true,
            compiler_information: vec![],
            selected_isa: None,
            scope: "independent test wire fixture",
        }
    }

    #[test]
    fn actual_c_continuation_work_marker_is_structural_until_both_launch_snapshots_match() {
        let lock = select_refinement(
            fixture(),
            PalsPostRepairRecheckPolicyV3::ActualOpponentContinuationV1,
        )
        .lock()
        .unwrap();
        let mut work = work_fixture("pals");
        work["pals_search_policy"] = serde_json::to_value(
            PalsSearchPolicyIdentityV3::expected_actual_opponent_continuation_v1(),
        )
        .unwrap();
        validate_pals_process_work(&work, "pals", true).unwrap();
        validate_search_policy_pair(&lock, NativeEngineRole::Baseline, &work, &work).unwrap();
        let mut old = work.clone();
        old["pals_search_policy"] =
            serde_json::to_value(PalsSearchPolicyIdentityV3::expected_same_repaired_line_once_v1())
                .unwrap();
        // Each identity is structurally valid; the independently locked launch
        // must reject substitution at either startup or termination.
        validate_pals_process_work(&old, "pals", true).unwrap();
        assert!(
            validate_search_policy_pair(&lock, NativeEngineRole::Baseline, &old, &work).is_err()
        );
        assert!(
            validate_search_policy_pair(&lock, NativeEngineRole::Baseline, &work, &old).is_err()
        );
    }

    #[test]
    fn post_repair_work_pair_requires_both_exact_markers_and_rejects_raw_duplicates() {
        let lock = select_post_repair_recheck(fixture()).lock().unwrap();
        let expected = PalsSearchPolicyIdentityV3::expected_same_repaired_line_once_v1();
        let mut work = work_fixture("pals");
        work["pals_search_policy"] = serde_json::to_value(&expected).unwrap();
        let start = serde_json::json!({"schema_version":3,"domain":PALS_SEARCH_WORK_STARTUP_V3_DOMAIN,
            "endpoint_id":"pals","launch_sha256":lock.sha256(),"process_id":101,"binary_sha256":"a".repeat(64),
            "service_exit_success":false,"search_work":work});
        let mut end = start.clone();
        end["domain"] = PALS_SEARCH_WORK_TERMINATION_V3_DOMAIN.into();
        end["service_exit_success"] = true.into();
        end["search_work"]["go_invocations"] = 1.into();
        end["search_work"]["successful_returns"] = 1.into();
        let verify = |s: &serde_json::Value, t: &serde_json::Value| {
            validate_pals_search_work_records(
                &lock,
                NativeEngineRole::Baseline,
                &serde_json::to_vec(s).unwrap(),
                &serde_json::to_vec(t).unwrap(),
                "native-process-101",
            )
        };
        let audit = verify(&start, &end).unwrap();
        assert_eq!(
            validate_search_policy_marker(&audit.search_work, Some(&expected)).unwrap(),
            Some(expected.clone())
        );
        for startup in [true, false] {
            let mut s = start.clone();
            let mut t = end.clone();
            let record = if startup { &mut s } else { &mut t };
            record["search_work"]
                .as_object_mut()
                .unwrap()
                .remove("pals_search_policy");
            assert!(verify(&s, &t).is_err());
            let mut invalid_markers = vec![serde_json::Value::Null, true.into(), "unknown".into()];
            for (field, value) in [
                ("version", "pals-post-repair-recheck/2".into()),
                ("policy", "same-repaired-line-once-v1".into()),
                ("search_identity", "pals-restricted-refinement/0.1".into()),
                ("conditions_sha256", serde_json::json!(vec![0u8; 31])),
                (
                    "conditions_sha256",
                    "bea44b7e9ab59f32dcffb1b4c597fd36a4b75037803aeeacd6c18785ce838166".into(),
                ),
            ] {
                let mut marker = serde_json::to_value(&expected).unwrap();
                marker[field] = value;
                invalid_markers.push(marker);
            }
            let mut out_of_range = serde_json::to_value(&expected).unwrap();
            out_of_range["conditions_sha256"][0] = 256.into();
            invalid_markers.push(out_of_range);
            let mut unknown = serde_json::to_value(&expected).unwrap();
            unknown["claimed_reply_success"] = true.into();
            invalid_markers.push(unknown);
            for marker in invalid_markers {
                let mut s = start.clone();
                let mut t = end.clone();
                let record = if startup { &mut s } else { &mut t };
                record["search_work"]["pals_search_policy"] = marker;
                assert!(verify(&s, &t).is_err());
            }
        }
        let raw = serde_json::to_string(&start).unwrap();
        let needle = "\"policy\":\"same_repaired_line_once_v1\"";
        assert!(raw.contains(needle));
        let duplicate = raw.replace(needle, &format!("{needle},{needle}"));
        assert!(
            validate_pals_search_work_records(
                &lock,
                NativeEngineRole::Baseline,
                duplicate.as_bytes(),
                &serde_json::to_vec(&end).unwrap(),
                "native-process-101"
            )
            .is_err()
        );
        let legacy = fixture().lock().unwrap();
        assert!(
            validate_search_policy_pair(&legacy, NativeEngineRole::Baseline, &work, &work).is_err()
        );
        assert!(
            validate_search_policy_pair(&lock, NativeEngineRole::Candidate, &work, &work).is_err()
        );
        assert!(validate_pals_process_work_inner(&work, "pals", true, true).is_err());
        let mut cpu = work_fixture("cpu");
        cpu["pals_search_policy"] = work["pals_search_policy"].clone();
        assert!(validate_pals_process_work(&cpu, "cpu", true).is_err());
    }

    #[test]
    fn post_repair_public_native_audit_requires_markers_without_weakening_physical_gates() {
        // Serialized wire fixture only: no CUDA session/device is constructed.
        let legacy = cuda_fixture().0.lock().unwrap();
        let selected = select_post_repair_recheck(cuda_fixture().0).lock().unwrap();
        let (mut start, mut end) = cuda_records_fixture(&selected);
        let verify = |s: &serde_json::Value, t: &serde_json::Value| {
            validate_pals_native_records(
                &selected,
                NativeEngineRole::Baseline,
                &serde_json::to_vec(s).unwrap(),
                &serde_json::to_vec(t).unwrap(),
                "native-process-100",
            )
        };
        assert!(verify(&start, &end).is_err());
        let mut work = work_fixture("pals");
        work["pals_search_policy"] =
            serde_json::to_value(PalsSearchPolicyIdentityV3::expected_same_repaired_line_once_v1())
                .unwrap();
        start["search_work"] = work.clone();
        end["search_work"] = work;
        assert!(verify(&start, &end).is_ok());
        for startup in [true, false] {
            let mut s = start.clone();
            let mut t = end.clone();
            let record = if startup { &mut s } else { &mut t };
            record["search_work"]["pals_search_policy"] = serde_json::Value::Null;
            assert!(verify(&s, &t).is_err());
        }
        for (field, value) in [
            ("physical_shutdown_confirmed", false.into()),
            ("native_buffers_released", false.into()),
            ("physical_runs_in_flight", 1.into()),
            ("quarantined", true.into()),
        ] {
            let mut invalid = end.clone();
            invalid["native"][field] = value;
            assert!(verify(&start, &invalid).is_err());
        }
        start["launch_sha256"] = legacy.sha256().into();
        end["launch_sha256"] = legacy.sha256().into();
        assert!(
            validate_pals_native_records(
                &legacy,
                NativeEngineRole::Baseline,
                &serde_json::to_vec(&start).unwrap(),
                &serde_json::to_vec(&end).unwrap(),
                "native-process-100"
            )
            .is_err()
        );
    }

    #[test]
    fn post_repair_core_projects_observed_marker_and_preserves_failed_go_and_nn_requirements() {
        let lock = select_post_repair_recheck(fixture()).lock().unwrap();
        let expected = PalsSearchPolicyIdentityV3::expected_same_repaired_line_once_v1();
        let mut work = work_fixture("pals");
        work["pals_search_policy"] = serde_json::to_value(&expected).unwrap();
        work["go_invocations"] = 6.into();
        work["failed_returns"] = 6.into();
        let mut audit = PalsProcessWorkAuditV3 {
            endpoint_id: "pals".into(),
            process_id: 101,
            startup_sha256: "a".repeat(64),
            termination_sha256: "b".repeat(64),
            startup_bytes: 100,
            termination_bytes: 200,
            search_work: work,
        };
        let preflight = preflight_fixture("pals");
        assert!(
            endpoint_work_receipt(&lock, NativeEngineRole::Baseline, None, None, &preflight)
                .is_err()
        );
        let observed = endpoint_work_receipt(
            &lock,
            NativeEngineRole::Baseline,
            Some(&audit),
            None,
            &preflight,
        )
        .unwrap();
        assert_eq!(observed.pals_search_policy, Some(expected));
        assert_eq!(observed.search_failed_go_count, Some(6));
        assert_eq!(observed.nn_inputs_completed, 0);
        let mut failures = BTreeSet::new();
        record_endpoint_search_failure(&observed, &mut failures);
        assert!(failures.contains(&PalsRunFailureV3::SearchFailure));
        let native = select_post_repair_recheck(native_fixture()).lock().unwrap();
        assert!(
            endpoint_work_receipt(
                &native,
                NativeEngineRole::Baseline,
                Some(&audit),
                None,
                &preflight
            )
            .is_err()
        );
        let legacy = fixture().lock().unwrap();
        assert!(
            endpoint_work_receipt(
                &legacy,
                NativeEngineRole::Baseline,
                Some(&audit),
                None,
                &preflight
            )
            .is_err()
        );
        audit
            .search_work
            .as_object_mut()
            .unwrap()
            .remove("pals_search_policy");
        assert!(
            endpoint_work_receipt(
                &lock,
                NativeEngineRole::Baseline,
                Some(&audit),
                None,
                &preflight
            )
            .is_err()
        );
    }
    #[test]
    fn pals_work_records_reject_missing_unknown_and_cross_process_observations() {
        let lock = fixture().lock().unwrap();
        let work = work_fixture("pals");
        let start = serde_json::json!({"schema_version":3,"domain":PALS_SEARCH_WORK_STARTUP_V3_DOMAIN,
            "endpoint_id":"pals","launch_sha256":lock.sha256(),"process_id":101,"binary_sha256":"a".repeat(64),
            "service_exit_success":false,"search_work":work});
        let mut end = start.clone();
        end["domain"] = PALS_SEARCH_WORK_TERMINATION_V3_DOMAIN.into();
        end["service_exit_success"] = true.into();
        end["search_work"]["go_invocations"] = 1.into();
        end["search_work"]["successful_returns"] = 1.into();
        let verify = |end: &serde_json::Value| {
            validate_pals_search_work_records(
                &lock,
                NativeEngineRole::Baseline,
                &serde_json::to_vec(&start).unwrap(),
                &serde_json::to_vec(end).unwrap(),
                "native-process-101",
            )
        };
        assert!(verify(&end).is_ok()); // unknown root gauge does not erase observed work.
        for (key, value) in [
            ("active_invocations", 1u64),
            ("unobserved_work_invocations", 1),
            ("physical_unknown_returns", 1),
        ] {
            let mut bad = end.clone();
            bad["search_work"][key] = value.into();
            assert!(verify(&bad).is_err());
        }
        let mut bad = end.clone();
        bad["process_id"] = 102.into();
        assert!(verify(&bad).is_err());
        let mut bad = end.clone();
        bad["search_work"]["pals"]["cpu_nodes"] = serde_json::Value::Null;
        assert!(verify(&bad).is_err());
        let mut bad = end;
        bad["search_work"]["successful_returns"] = 2.into();
        assert!(verify(&bad).is_err());
    }

    #[test]
    fn additive_resolver_identity_checks_semantics_without_rewriting_legacy_work() {
        let mut work = work_fixture("pals");
        assert!(validate_pals_process_work(&work, "pals", true).is_ok());
        assert!(work.get("pals_resolver").is_none());
        work["pals_resolver"] = serde_json::json!({
            "version": rz_search::pals::engine::PALS_VALUE_RESOLVER_VERSION,
            "semantics_sha256": Sha256::digest(
                rz_search::pals::engine::PALS_VALUE_RESOLVER_SEMANTICS.as_bytes()
            ).to_vec()
        });
        assert!(validate_pals_process_work(&work, "pals", true).is_ok());
        let mut cpu = work_fixture("cpu");
        cpu["pals_resolver"] = work["pals_resolver"].clone();
        assert!(validate_pals_process_work(&cpu, "cpu", true).is_err());
        work["pals_resolver"]["semantics_sha256"][0] = 256.into();
        assert!(validate_pals_process_work(&work, "pals", true).is_err());
    }
    #[test]
    fn additive_external_work_completeness_keeps_boolean_and_unknown_distinct() {
        let mut work = work_fixture("pals");
        // Historical receipts omit the field; current own-checker receipts
        // explicitly observe that no incomplete foreign work exists.
        assert!(validate_pals_process_work(&work, "pals", true).is_ok());
        for value in [serde_json::Value::Null, false.into()] {
            work["pals"]["external_checker_work_incomplete"] = value;
            assert!(validate_pals_process_work(&work, "pals", true).is_ok());
        }
        work["pals"]["external_checker_work_incomplete"] = true.into();
        assert!(validate_pals_process_work(&work, "pals", true).is_err());
        assert!(validate_pals_process_work(&work, "pals", false).is_ok());
        for value in [0.into(), "false".into(), serde_json::json!({})] {
            work["pals"]["external_checker_work_incomplete"] = value;
            assert!(validate_pals_process_work(&work, "pals", false).is_err());
        }
        let mut cpu = work_fixture("cpu");
        cpu["cpu"]["external_checker_work_incomplete"] = false.into();
        assert!(validate_pals_process_work(&cpu, "cpu", false).is_err());
        work["pals"]["external_checker_work_incomplete"] = false.into();
        work["pals"]["completed_cpu_tasks"] = serde_json::Value::Null;
        assert!(validate_pals_process_work(&work, "pals", false).is_err());
    }
    #[test]
    fn pals_core_projection_separates_cpu_evidence_reuse_from_nn_and_actual_consumption() {
        let lock = fixture().lock().unwrap();
        let mut work = work_fixture("pals");
        for (field, value) in [
            ("cpu_tasks_requested", 1u64),
            ("completed_cpu_tasks", 1),
            ("reused_completed_cpu_tasks_consumed", 2),
            ("consumed_cpu_tasks", 3),
            ("consumed_cached_cpu_values", 4),
            ("completed_proposer_calls", 2),
            ("completed_repair_calls", 1),
            ("completed_critic_calls", 2),
        ] {
            work["pals"][field] = value.into();
        }
        let audit = PalsProcessWorkAuditV3 {
            endpoint_id: "pals".into(),
            process_id: 1,
            startup_sha256: "a".repeat(64),
            termination_sha256: "a".repeat(64),
            startup_bytes: 100,
            termination_bytes: 200,
            search_work: work,
        };
        let receipt = endpoint_work_receipt(
            &lock,
            NativeEngineRole::Baseline,
            Some(&audit),
            None,
            &preflight_fixture("pals"),
        )
        .unwrap();
        assert_eq!(
            (
                receipt.cpu_tasks_completed,
                receipt.cpu_tasks_reused_consumed,
                receipt.cpu_tasks_consumed
            ),
            (1, 2, 3)
        );
        assert_eq!(
            (
                receipt.nn_inputs_completed,
                receipt.cached_evaluations_consumed
            ),
            (0, 0)
        );
        assert_eq!(
            (
                receipt.proposer_tasks_completed,
                receipt.critic_tasks_completed
            ),
            (3, 2)
        );
        let mut cpu = work_fixture("cpu");
        cpu["cpu"]["tasks_requested"] = 2.into();
        cpu["cpu"]["requested_depth_completed"] = 2.into();
        cpu["cpu"]["completed_reports_accepted_for_uci_output"] = 1.into();
        let cpu_audit = PalsProcessWorkAuditV3 {
            endpoint_id: "cpu".into(),
            search_work: cpu,
            ..audit
        };
        let receipt = endpoint_work_receipt(
            &lock,
            NativeEngineRole::Candidate,
            Some(&cpu_audit),
            None,
            &preflight_fixture("cpu"),
        )
        .unwrap();
        assert_eq!(
            (receipt.cpu_tasks_completed, receipt.cpu_tasks_consumed),
            (2, 1)
        );
        let mut missing = cpu_audit;
        missing.search_work["cpu"]
            .as_object_mut()
            .unwrap()
            .remove("completed_reports_accepted_for_uci_output");
        assert!(
            endpoint_work_receipt(
                &lock,
                NativeEngineRole::Candidate,
                Some(&missing),
                None,
                &preflight_fixture("cpu")
            )
            .is_err()
        );
    }
    #[test]
    fn pals_core_search_failed_go_count_preserves_failure_and_rejects_unknown_work() {
        let lock = fixture().lock().unwrap();
        for (role, kind, id) in [
            (NativeEngineRole::Baseline, "pals", "pals"),
            (NativeEngineRole::Candidate, "cpu", "cpu"),
        ] {
            let mut audit = PalsProcessWorkAuditV3 {
                endpoint_id: id.into(),
                process_id: 1,
                startup_sha256: "a".repeat(64),
                termination_sha256: "b".repeat(64),
                startup_bytes: 100,
                termination_bytes: 200,
                search_work: work_fixture(kind),
            };
            let preflight = preflight_fixture(id);
            let zero = endpoint_work_receipt(&lock, role, Some(&audit), None, &preflight).unwrap();
            assert_eq!(zero.search_failed_go_count, Some(0));
            let mut failures = BTreeSet::new();
            record_endpoint_search_failure(&zero, &mut failures);
            assert!(failures.is_empty());

            audit.search_work["go_invocations"] = 6.into();
            audit.search_work["failed_returns"] = 6.into();
            let failed =
                endpoint_work_receipt(&lock, role, Some(&audit), None, &preflight).unwrap();
            assert_eq!(failed.search_failed_go_count, Some(6));
            assert_eq!(failed.cpu_tasks_completed, zero.cpu_tasks_completed);
            assert_eq!(failed.cpu_nodes, zero.cpu_nodes);
            record_endpoint_search_failure(&failed, &mut failures);
            assert_eq!(failures, BTreeSet::from([PalsRunFailureV3::SearchFailure]));
            assert!(!failures.is_empty()); // The assembler derives pair eligibility here.

            for invalid in [serde_json::Value::Null, true.into(), "6".into()] {
                let mut unknown = audit.clone();
                unknown.search_work["failed_returns"] = invalid;
                assert!(
                    endpoint_work_receipt(&lock, role, Some(&unknown), None, &preflight).is_err()
                );
            }
            audit
                .search_work
                .as_object_mut()
                .unwrap()
                .remove("failed_returns");
            assert!(endpoint_work_receipt(&lock, role, Some(&audit), None, &preflight).is_err());
            assert!(endpoint_work_receipt(&lock, role, None, None, &preflight).is_err());
        }
    }

    #[test]
    fn pals_core_nn_completion_includes_public_graph_without_counting_cache_reuse() {
        let lock = native_fixture().lock().unwrap();
        let mut work = work_fixture("pals");
        work["pals"]["consumed_role_outputs"] = 1.into();
        work["pals"]["completed_proposer_calls"] = 1.into();
        let audit = PalsProcessWorkAuditV3 {
            endpoint_id: "pals".into(),
            process_id: 10,
            startup_sha256: "a".repeat(64),
            termination_sha256: "b".repeat(64),
            startup_bytes: 100,
            termination_bytes: 200,
            search_work: work,
        };
        let native = PalsNativeSessionAuditV3 {
            endpoint_id: "pals".into(),
            process_id: 10,
            launch_sha256: lock.sha256().into(),
            startup_sha256: "a".repeat(64),
            termination_sha256: "b".repeat(64),
            external_cpu_r_session: None,
            completed_role_inputs: 3,
            search_consumed_role_inputs: 1,
            completed_new_game_resets: 1,
            physical_shutdown_confirmed: true,
            native_buffers_released: true,
            startup_nn_inputs_completed: 0,
            startup_nn_calls_completed: 0,
            startup_role_inputs_completed: 0,
            startup_probe_timeout_ms: None,
            startup_probe: None,
            execution: None,
            cuda_placement: None,
            cuda_loading: None,
            cuda_record_pages: None,
            raw_native: serde_json::json!({"backend_stats_observation":"exclusive_worker_before_shutdown","observer_failures":0,"last_observer_failure":null,"frozen_epoch":1,
                "backend_stats":{"public_nn_runs_completed":1,"role_nn_runs_completed":3,"completed_nn_inputs":4,"public_cache_hits":2,
                    "public_nn_runs_failed_known":0,"role_nn_runs_failed_known":0,"public_nn_runs_attempted":1,"role_nn_runs_attempted":3,
                    "validated_public_outputs":1,"validated_role_outputs":3,"live_public_cache_entries":1}}),
            raw_search_work: audit.search_work.clone(),
        };
        let receipt = endpoint_work_receipt(
            &lock,
            NativeEngineRole::Baseline,
            Some(&audit),
            Some(&native),
            &preflight_fixture("pals"),
        )
        .unwrap();
        assert_eq!(
            (
                receipt.nn_inputs_completed,
                receipt.nn_calls_completed,
                receipt.nn_inputs_consumed
            ),
            (4, 4, 1)
        );
        assert_eq!(receipt.cached_evaluations_consumed, 0);
        let mut missing_observer = native.clone();
        missing_observer
            .raw_native
            .as_object_mut()
            .unwrap()
            .remove("observer_failures");
        assert!(
            endpoint_work_receipt(
                &lock,
                NativeEngineRole::Baseline,
                Some(&audit),
                Some(&missing_observer),
                &preflight_fixture("pals")
            )
            .is_err()
        );
        let mut unknown = native;
        unknown.raw_native["backend_stats"] = serde_json::Value::Null;
        assert!(
            endpoint_work_receipt(
                &lock,
                NativeEngineRole::Baseline,
                Some(&audit),
                Some(&unknown),
                &preflight_fixture("pals")
            )
            .is_err()
        );
    }
    #[test]
    fn pals_shared_pc_recipe_uses_exact_two_graph_layout() {
        let mut f = native_fixture();
        let PalsEndpointLaunchV3::OnnxCpu(n) = &mut f.endpoints[0] else {
            unreachable!()
        };
        n.graphs.retain(|g| g.role == "public");
        n.graphs.push(PalsGraphAssetV3 {
            file: "shared_pc_if.onnx".into(),
            role: "shared_pc".into(),
            artifact: asset("model/shared_pc_if.onnx"),
        });
        let lock = f.lock().unwrap();
        assert_eq!(lock.expected_provider_sessions(), 2); // process count, not graph/session count.
        let mut bad = f.clone();
        let PalsEndpointLaunchV3::OnnxCpu(n) = &mut bad.endpoints[0] else {
            unreachable!()
        };
        n.graphs[1].role = "validator".into();
        assert!(bad.lock().is_err());
    }
    #[test]
    fn pals_core_projects_generated_pgn_provenance_without_relaxing_artifact_validation() {
        // Contract fixture only: no engine, model, game or physical execution.
        let lock = fixture().lock().unwrap();
        let mut native_pgn = asset("attempt-01/match.pgn");
        native_pgn.source = "RoveZero PALS-V3 NN integration execution evidence".into();
        let original = native_pgn.clone();
        assert!(native_pgn.validate().is_err());
        let (projected, provenance) = project_pals_core_pgn(&lock, &native_pgn).unwrap();
        assert_eq!(native_pgn, original);
        assert_eq!(provenance.native_artifact, original);
        assert_eq!(projected.source, lock.input.runner.source_url);
        assert_eq!(provenance.producer_source_url, projected.source);
        assert_eq!(
            provenance.producer_source_commit,
            lock.input.runner.source_commit
        );
        assert_eq!(
            provenance.producer_binary_sha256,
            lock.input.runner.binary.sha256
        );
        assert!(
            provenance
                .description
                .contains("not a public download location")
        );
        let mut only_source_changed = projected.clone();
        only_source_changed.source = original.source.clone();
        assert_eq!(only_source_changed, original);
        projected.validate().unwrap();
        let engines: [PalsEndpointReceiptV3; 2] = std::array::from_fn(|index| {
            let id = lock.input.semantic_lock.manifest.engines[index].id();
            let work = PalsProcessWorkAuditV3 {
                endpoint_id: id.into(),
                process_id: 101 + index as u32,
                startup_sha256: "d".repeat(64),
                termination_sha256: "e".repeat(64),
                startup_bytes: 1,
                termination_bytes: 1,
                search_work: work_fixture(if index == 0 { "pals" } else { "cpu" }),
            };
            endpoint_work_receipt(
                &lock,
                if index == 0 {
                    NativeEngineRole::Baseline
                } else {
                    NativeEngineRole::Candidate
                },
                Some(&work),
                None,
                &preflight_fixture(id),
            )
            .unwrap()
        });
        let manifest = &lock.input.semantic_lock.manifest;
        let games = (0..2)
            .map(|index| PalsGameReceiptV3 {
                game_index: index,
                white_endpoint: manifest.pilot.white_order[index as usize].clone(),
                black_endpoint: manifest.pilot.white_order[1 - index as usize].clone(),
                base_ms: manifest.pilot.base_ms,
                increment_ms: manifest.pilot.increment_ms,
                plies: 0,
                result: PalsResultV3::Draw,
                termination: PalsTerminationV3::RulesDraw,
                failed_endpoint: None,
                engines: engines.clone(),
                pgn: projected.clone(),
            })
            .collect();
        let core = PalsRunReceiptV3 {
            domain: PALS_RECEIPT_V3_DOMAIN.into(),
            run_id: manifest.run_id.clone(),
            pair_id: manifest.pair_id.clone(),
            lock_sha256: lock.input.semantic_lock.canonical_sha256.clone(),
            training_executed: false,
            wall_time_ms: 1,
            cleanup_time_ms: 1,
            pair_eligible: true,
            failures: BTreeSet::new(),
            games,
        };
        core.validate_against(&lock.input.semantic_lock).unwrap();
        let mut old_projection = core.clone();
        old_projection.games[0].pgn = original.clone();
        assert!(
            old_projection
                .validate_against(&lock.input.semantic_lock)
                .is_err()
        );
        for source in [
            "http://example.org/source",
            "https://user@example.org/source",
            "https://example.org/source?query",
            "https://example.org/source#fragment",
        ] {
            let mut invalid_producer = lock.clone();
            invalid_producer.input.runner.source_url = source.into();
            assert!(project_pals_core_pgn(&invalid_producer, &original).is_err());
        }
    }
    #[test]
    fn pals_game_process_mapping_requires_game_color_and_synchronous_exit_windows() {
        let lock = fixture().lock().unwrap();
        let trace = |message: &str| {
            format!(
                "[TRACE ] [12:34:56.123456] <{:>20}> fastchess --- {message}",
                1
            )
        };
        let exit = |pid| {
            trace(&format!(
                "Process with pid: {pid} terminated with status: 0"
            ))
        };
        let rows = vec![
            "[Engine] fixture pals ---> Started game 1 of 2 (pals vs cpu)".into(),
            trace("Game 1 between pals and cpu starting"),
            "Started game 1 of 2 (pals vs cpu)".into(),
            trace("Game 1 between pals and cpu finished"),
            trace("Game 1 finished with result 0-1"),
            "Finished game 1 (pals vs cpu): 0-1 {Black wins}".into(),
            exit(101),
            exit(201),
            trace("Game 2 between cpu and pals starting"),
            "Started game 2 of 2 (cpu vs pals)".into(),
            trace("Game 2 between cpu and pals finished"),
            trace("Game 2 finished with result 1-0"),
            "Finished game 2 (cpu vs pals): 1-0 {White wins}".into(),
            exit(202),
            exit(102),
        ];
        let pids = [BTreeSet::from([101, 102]), BTreeSet::from([201, 202])];
        let encode = |rows: &[String]| format!("{}\n", rows.join("\n")).into_bytes();
        assert_eq!(
            validate_pals_game_process_trace(&lock, &encode(&rows), &pids).unwrap(),
            [[101, 102], [201, 202]]
        );
        let mut swapped = rows.clone();
        swapped.swap(6, 7);
        assert!(validate_pals_game_process_trace(&lock, &encode(&swapped), &pids).is_err());
        let mut missing = rows.clone();
        missing.remove(7);
        assert!(validate_pals_game_process_trace(&lock, &encode(&missing), &pids).is_err());
        let mut early = rows.clone();
        early.swap(7, 8);
        assert!(validate_pals_game_process_trace(&lock, &encode(&early), &pids).is_err());
        let mut failed = rows.clone();
        failed[6] = trace("Process with pid: 101 terminated with status: 256");
        assert!(validate_pals_game_process_trace(&lock, &encode(&failed), &pids).is_err());
        let mut unknown = rows;
        unknown.remove(3);
        assert!(validate_pals_game_process_trace(&lock, &encode(&unknown), &pids).is_err());
    }
}
