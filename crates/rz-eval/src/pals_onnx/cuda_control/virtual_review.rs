//! Opt-in V3 selected-review control policy. The legacy inventory, decoder,
//! placement policy and fixed pins are unchanged. Static metadata admits only
//! exact named shape/BOOL control; actual placement and kernels remain separate.
use super::*;
use serde_json::Value;
mod inspection;

const SCHEMA: &str = "rovezero.pals-virtual-static-control-inventory.v1";
const WITNESS_SCHEMA: &str = "rovezero.pals-virtual-cuda-metadata-control.v1";
const FRESH_LAYOUT: &str = "virtual_review_shared_pc_if_v1";
const WARM_LAYOUT: &str = "virtual_review_shared_pc_if_approx_warm_v1";
const PUBLIC_GRAPH: &str = "virtual_review_public_v1";
const VALIDATOR_GRAPH: &str = "virtual_review_validator_fresh_v1";
const MAX_VIRTUAL_INVENTORY_BYTES: usize = 4 * 1024 * 1024;
const MAX_VIRTUAL_NODES: usize = 12000;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Kind {
    FreshPc,
    FreshCollection,
    ApproxWarm,
}
#[derive(Clone, Debug)]
pub struct VirtualReviewControlLayout {
    kind: Kind,
    validator_private: BTreeMap<String, String>,
    validator_task_head: BTreeMap<String, String>,
    weight_bindings: BTreeMap<String, Vec<WeightBindingV3>>,
}
impl VirtualReviewControlLayout {
    pub fn artifact_kind(&self) -> &'static str {
        if self.kind == Kind::ApproxWarm {
            "approx_warm"
        } else {
            "fresh"
        }
    }
    pub fn layout_name(&self) -> &'static str {
        if self.kind == Kind::ApproxWarm {
            WARM_LAYOUT
        } else {
            FRESH_LAYOUT
        }
    }
    pub fn graph_roles(&self) -> &'static [&'static str] {
        match self.kind {
            Kind::FreshPc => &["public", "shared_pc"],
            Kind::FreshCollection => &["public", "shared_pc", "validator"],
            Kind::ApproxWarm => &["public", "shared_warm"],
        }
    }
}
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct VirtualReviewCudaPlacementWitness {
    pub schema: String,
    pub inventory_sha256: [u8; 32],
    pub manifest_sha256: [u8; 32],
    pub runtime_sha256: [u8; 32],
    pub runtime_bundle_sha256: [u8; 32],
    pub layout: String,
    pub artifact_kind: String,
    pub optimization: PalsGraphOptimization,
    pub initialization: Vec<PalsGraphPlacement>,
    pub kernels: BTreeMap<String, PalsKernelWitness>,
    pub validator_private_kernels: usize,
    pub validator_task_head_kernels: usize,
    pub warm_seed_identity_kernels: usize,
    pub category_provenance: String,
}

/// The system-runtime domain deliberately has no legacy bundle field.  This
/// proves only the supplied placement profiles and current process mappings;
/// GPU physical completion remains the native worker owner's responsibility.
#[cfg(target_os = "linux")]
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct SystemVirtualReviewCudaPlacementWitness {
    pub schema: &'static str,
    pub inventory_sha256: [u8; 32],
    pub manifest_sha256: [u8; 32],
    pub ort_sha256: [u8; 32],
    pub system_runtime_identity_sha256: [u8; 32],
    pub declared_device_id: i32,
    pub mapped_roles: Vec<String>,
    pub missing_optional_roles: Vec<String>,
    pub layout: String,
    pub artifact_kind: String,
    pub optimization: PalsGraphOptimization,
    pub initialization: Vec<PalsGraphPlacement>,
    pub kernels: BTreeMap<String, PalsKernelWitness>,
    pub validator_private_kernels: usize,
    pub validator_task_head_kernels: usize,
    pub warm_seed_identity_kernels: usize,
    pub physical_completion: &'static str,
    pub category_provenance: String,
}

struct VerifiedVirtualProfiles {
    layout: String,
    artifact_kind: String,
    initialization: Vec<PalsGraphPlacement>,
    kernels: BTreeMap<String, PalsKernelWitness>,
    validator_private_kernels: usize,
    validator_task_head_kernels: usize,
    warm_seed_identity_kernels: usize,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct InventoryV3 {
    schema: String,
    scope: String,
    manifest_sha256: String,
    actual_provider_placement: String,
    cpu_allowlist: String,
    total_nodes: usize,
    artifact_kind: String,
    layout: String,
    producer_source_sha256: String,
    admission_limits: BTreeMap<String, usize>,
    graphs: Vec<InventoryGraphV3>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct InventoryGraphV3 {
    role: String,
    file: String,
    sha256: String,
    serialized_graph_name: String,
    inventory: InventoryScope,
    weight_bindings: Vec<WeightBindingV3>,
    private_neural_dependencies: Vec<PrivateNeuralDependencyV3>,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct WeightBindingV3 {
    initializer: String,
    parameter: String,
    dtype: String,
    shape: Vec<u64>,
    bytes: u64,
    sha256: String,
    source_parameter_sha256: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PrivateNeuralDependencyV3 {
    scope_path: String,
    node_index: usize,
    node_name: String,
    op: String,
    initializer: String,
    parameter: String,
}
fn strict_sha(value: &str) -> Result<[u8; 32], BackendError> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    {
        return Err(policy_error("V3 control pin must be lowercase SHA-256"));
    }
    parse_sha256(value)
}
impl PalsCudaControlPolicy {
    pub fn from_virtual_review_inventory(
        path: &Path,
        expected_sha256: &str,
    ) -> Result<Self, BackendError> {
        if !path.is_absolute() {
            return Err(policy_error("V3 control inventory must be absolute"));
        }
        let bytes = asset::read_bounded(path, MAX_VIRTUAL_INVENTORY_BYTES)?;
        Self::from_virtual_review_inventory_bytes(&bytes, expected_sha256)
    }
    /// Bootstrap-owned inventory bytes; no path reopening or schema rewrite.
    /// This verifies actual bytes before strict V3 policy construction and is
    /// not evidence of a session, Run, kernel or physical completion.
    pub fn from_virtual_review_inventory_bytes(
        bytes: &[u8],
        expected_sha256: &str,
    ) -> Result<Self, BackendError> {
        if bytes.is_empty() || bytes.len() > MAX_VIRTUAL_INVENTORY_BYTES {
            return Err(policy_error("V3 inventory exceeds bounded input"));
        }
        if asset::sha256(bytes) != strict_sha(expected_sha256)? {
            return Err(policy_error("V3 control inventory digest differs"));
        }
        Self::from_virtual_review_bytes(bytes)
    }
    fn from_virtual_review_bytes(bytes: &[u8]) -> Result<Self, BackendError> {
        if bytes.is_empty() || bytes.len() > MAX_VIRTUAL_INVENTORY_BYTES {
            return Err(policy_error("V3 inventory exceeds bounded input"));
        }
        let unique: UniqueJson = serde_json::from_slice(bytes).map_err(|e| {
            policy_error("cannot decode unique V3 inventory")
                .with_external_cause(CauseCode::ProfileParse, &e)
        })?;
        validate_nested_keys(&unique.0)?;
        let source: InventoryV3 = serde_json::from_value(unique.0).map_err(|e| {
            policy_error("cannot decode strict V3 inventory")
                .with_external_cause(CauseCode::ProfileParse, &e)
        })?;
        let expected_limits = BTreeMap::from([
            ("max_manifest_bytes".into(), 256 * 1024),
            ("max_graph_bytes".into(), 256 * 1024 * 1024),
            ("max_inventory_bytes".into(), MAX_VIRTUAL_INVENTORY_BYTES),
            ("max_nodes".into(), MAX_VIRTUAL_NODES),
            ("max_graph_nodes".into(), MAX_NODES),
            ("max_initializers_per_scope".into(), 5000),
            ("max_attributes_per_node".into(), 32),
            ("max_recursive_depth".into(), 32),
            ("max_path_bytes".into(), 4096),
        ]);
        strict_sha(&source.producer_source_sha256)?;
        if source.schema != SCHEMA
            || source.scope != "read_only_serialized_graph_metadata_no_runtime"
            || source.actual_provider_placement != "not_observed"
            || source.cpu_allowlist != "not_created"
            || !(1..=MAX_VIRTUAL_NODES).contains(&source.total_nodes)
            || source.admission_limits != expected_limits
        {
            return Err(policy_error(
                "V3 control inventory declaration is unsupported",
            ));
        }
        let roles = source
            .graphs
            .iter()
            .map(|g| g.role.as_str())
            .collect::<Vec<_>>();
        let kind = match (
            source.artifact_kind.as_str(),
            source.layout.as_str(),
            roles.as_slice(),
        ) {
            ("fresh", FRESH_LAYOUT, ["public", "shared_pc"]) => Kind::FreshPc,
            ("fresh", FRESH_LAYOUT, ["public", "shared_pc", "validator"]) => Kind::FreshCollection,
            ("approx_warm", WARM_LAYOUT, ["public", "shared_warm"]) => Kind::ApproxWarm,
            _ => return Err(policy_error("V3 graph roles/layout/artifact domain differ")),
        };
        let mut graphs = BTreeMap::new();
        let mut count = 0;
        let mut validator_private = BTreeMap::new();
        let mut validator_task_head = BTreeMap::new();
        let mut weight_bindings = BTreeMap::new();
        for graph in &source.graphs {
            let (file, name) = match graph.role.as_str() {
                "public" => ("public_memory.onnx", PUBLIC_GRAPH),
                "shared_pc" => ("shared_pc_if.onnx", FRESH_LAYOUT),
                "validator" => ("role_validator.onnx", VALIDATOR_GRAPH),
                "shared_warm" => ("shared_pc_if_approx_warm.onnx", WARM_LAYOUT),
                _ => return Err(policy_error("V3 graph role is unsupported")),
            };
            if graph.file != file
                || graph.serialized_graph_name != name
                || graph.inventory.path != graph.role
            {
                return Err(policy_error("V3 serialized graph/file/scope differs"));
            }
            strict_sha(&graph.sha256)?;
            validate_root_abi(graph)?;
            validate_weight_bindings(graph)?;
            let (policy, private, task) = inspection::inspect(graph, &mut count)?;
            if graph.role == "validator" {
                validator_private = private;
                validator_task_head = task;
            }
            weight_bindings.insert(graph.role.clone(), graph.weight_bindings.clone());
            if graphs.insert(graph.role.clone(), policy).is_some() {
                return Err(policy_error("V3 graph is duplicated"));
            }
        }
        if count != source.total_nodes {
            return Err(policy_error("V3 inventory node coverage is incomplete"));
        }
        Ok(Self {
            manifest: strict_sha(&source.manifest_sha256)?,
            inventory: asset::sha256(bytes),
            graphs,
            virtual_layout: Some(VirtualReviewControlLayout {
                kind,
                validator_private,
                validator_task_head,
                weight_bindings,
            }),
        })
    }
    pub fn virtual_review_layout(&self) -> Option<&VirtualReviewControlLayout> {
        self.virtual_layout.as_ref()
    }
    /// Actual manifest bytes bind the declaration ledger. This does not hash
    /// TensorProto values or claim native parameter residency/sharing.
    pub(crate) fn validate_virtual_manifest_weight_bindings(
        &self,
        bytes: &[u8],
    ) -> Result<(), BackendError> {
        if bytes.is_empty() || bytes.len() > 256 * 1024 || asset::sha256(bytes) != self.manifest {
            return Err(policy_error("V3 weight ledger manifest byte pin differs"));
        }
        let layout = self
            .virtual_layout
            .as_ref()
            .ok_or_else(|| policy_error("legacy inventory has no V3 ledger"))?;
        let unique: UniqueJson = serde_json::from_slice(bytes).map_err(|e| {
            policy_error("cannot decode V3 pinned manifest ledger")
                .with_external_cause(CauseCode::ProfileParse, &e)
        })?;
        let banks = unique
            .0
            .get("initializer_banks")
            .and_then(Value::as_object)
            .ok_or_else(|| policy_error("V3 pinned initializer bank required"))?;
        if banks.len() != layout.weight_bindings.len() {
            return Err(policy_error("V3 initializer ledger graph coverage differs"));
        }
        for (role, bindings) in &layout.weight_bindings {
            let rows = banks
                .get(role)
                .and_then(|bank| bank.get("parameters"))
                .and_then(Value::as_array)
                .ok_or_else(|| policy_error("V3 actual manifest parameter ledger absent"))?;
            if rows.len() != bindings.len() {
                return Err(policy_error(
                    "V3 manifest parameter ledger coverage differs",
                ));
            }
            for (row, binding) in rows.iter().zip(bindings) {
                keys(
                    row,
                    &[
                        "initializer",
                        "parameter",
                        "shape",
                        "bytes",
                        "sha256",
                        "source_parameter_sha256",
                    ],
                )?;
                let mut expected = serde_json::to_value(binding)
                    .map_err(|_| policy_error("V3 weight ledger serialization failed"))?;
                expected
                    .as_object_mut()
                    .expect("typed binding object")
                    .remove("dtype");
                if row != &expected {
                    return Err(policy_error(
                        "V3 source metadata differs from pinned manifest parameter ledger",
                    ));
                }
            }
        }
        Ok(())
    }
    pub fn verify_virtual_profiles(
        &self,
        profiles: &BTreeMap<String, Vec<u8>>,
        placements: Vec<PalsGraphPlacement>,
        runtime: [u8; 32],
        bundle: [u8; 32],
    ) -> Result<VirtualReviewCudaPlacementWitness, BackendError> {
        let verified = self.verify_virtual_profiles_common(profiles, placements)?;
        Ok(VirtualReviewCudaPlacementWitness {
            schema: WITNESS_SCHEMA.into(),
            inventory_sha256: self.inventory,
            manifest_sha256: self.manifest,
            runtime_sha256: runtime,
            runtime_bundle_sha256: bundle,
            layout: verified.layout,
            artifact_kind: verified.artifact_kind,
            optimization: PalsGraphOptimization::Disable,
            initialization: verified.initialization,
            kernels: verified.kernels,
            validator_private_kernels: verified.validator_private_kernels,
            validator_task_head_kernels: verified.validator_task_head_kernels,
            warm_seed_identity_kernels: verified.warm_seed_identity_kernels,
            category_provenance: "rc10-category-unavailable-id-location-message-used".into(),
        })
    }

    /// System-library evidence has a distinct domain and never masquerades as
    /// the historical nineteen-file bundle. The caller must use profiles from
    /// its actual owned sessions and separately establish physical completion.
    #[cfg(target_os = "linux")]
    pub(crate) fn verify_system_virtual_profiles(
        &self,
        profiles: &BTreeMap<String, Vec<u8>>,
        placements: Vec<PalsGraphPlacement>,
        runtime: &OrtRuntime,
        declared_device_id: i32,
    ) -> Result<SystemVirtualReviewCudaPlacementWitness, BackendError> {
        if declared_device_id < 0 {
            return Err(policy_error("system CUDA declared device is invalid"));
        }
        let system = runtime
            .system_runtime()
            .ok_or_else(|| policy_error("system CUDA placement requires a system runtime owner"))?;
        let observed = system.verify_mappings().map_err(|cause| {
            policy_error("system CUDA runtime mappings could not be verified")
                .with_external_cause(CauseCode::RuntimeLibraryLoad, &cause)
        })?;
        if observed.identity_sha256 != system.identity_sha256()
            || !observed.missing_required_roles.is_empty()
            || !observed.unregistered_native_mappings.is_empty()
        {
            return Err(policy_error("system CUDA mapping identity differs"));
        }
        let verified = self.verify_virtual_profiles_common(profiles, placements)?;
        Ok(SystemVirtualReviewCudaPlacementWitness {
            schema: "rovezero.pals-virtual-system-cuda-placement.v1",
            inventory_sha256: self.inventory,
            manifest_sha256: self.manifest,
            ort_sha256: runtime.binary_digest(),
            system_runtime_identity_sha256: observed.identity_sha256,
            declared_device_id,
            mapped_roles: observed.mapped_roles,
            missing_optional_roles: observed.missing_optional_roles,
            layout: verified.layout,
            artifact_kind: verified.artifact_kind,
            optimization: PalsGraphOptimization::Disable,
            initialization: verified.initialization,
            kernels: verified.kernels,
            validator_private_kernels: verified.validator_private_kernels,
            validator_task_head_kernels: verified.validator_task_head_kernels,
            warm_seed_identity_kernels: verified.warm_seed_identity_kernels,
            physical_completion: "not_inferred",
            category_provenance: "rc10-category-unavailable-id-location-message-used".into(),
        })
    }

    fn verify_virtual_profiles_common(
        &self,
        profiles: &BTreeMap<String, Vec<u8>>,
        placements: Vec<PalsGraphPlacement>,
    ) -> Result<VerifiedVirtualProfiles, BackendError> {
        let layout = self
            .virtual_layout
            .as_ref()
            .ok_or_else(|| policy_error("legacy control cannot issue V3 witness"))?;
        let expected = layout
            .graph_roles()
            .iter()
            .copied()
            .collect::<BTreeSet<_>>();
        if profiles.keys().map(String::as_str).collect::<BTreeSet<_>>() != expected
            || placements.len() != expected.len()
        {
            return Err(policy_error(
                "V3 actual evidence is missing a registered graph role",
            ));
        }
        let mut seen = BTreeSet::new();
        for placement in &placements {
            let graph = self
                .graphs
                .get(&placement.role)
                .ok_or_else(|| policy_error("V3 placement has unknown role"))?;
            if !seen.insert(placement.role.as_str())
                || placement.graph_sha256 != graph.digest
                || placement.assigned_nodes == 0
                || placement.optimization != PalsGraphOptimization::Disable
                || placement.assigned_nodes > graph.nodes.len() + placement.approved_transfers.len()
                || !valid_transfers(graph, &placement.approved_transfers)
                || placement.cuda_nodes == 0
                || placement
                    .cuda_nodes
                    .checked_add(placement.approved_cpu_control_nodes)
                    != Some(placement.assigned_nodes)
                || placement.recursive_coverage
                    != "ort-1.22-finalize-recursive-exact-named-provider-coverage-before-first-run"
            {
                return Err(policy_error(
                    "V3 pre-Run source/provider coverage is invalid",
                ));
            }
        }
        if seen != expected {
            return Err(policy_error("V3 pre-Run graph role coverage is incomplete"));
        }
        let mut kernels = BTreeMap::new();
        for role in layout.graph_roles() {
            let placement = placements
                .iter()
                .find(|p| p.role == *role)
                .expect("complete registered roles");
            let witness = verify_profile(
                &profiles[*role],
                &self.graphs[*role],
                &placement.approved_transfers,
            )?;
            if witness.neural_kernels == 0 {
                return Err(policy_error(
                    "V3 role profile has no completed neural kernel",
                ));
            }
            kernels.insert((*role).into(), witness);
        }
        let pc = if layout.kind == Kind::ApproxWarm {
            "shared_warm"
        } else {
            "shared_pc"
        };
        if kernels[pc].proposer_private_kernels == 0 || kernels[pc].critic_private_kernels == 0 {
            return Err(policy_error(
                "V3 actual profiles do not prove both selected P/C private branches",
            ));
        }
        let (validator_private_kernels, validator_task_head_kernels) =
            if layout.kind == Kind::FreshCollection {
                let data = &profiles["validator"];
                let private = count_cuda_named(data, &layout.validator_private)?;
                let task = count_cuda_named(data, &layout.validator_task_head)?;
                if private == 0 || task == 0 {
                    return Err(policy_error(
                        "V3 V profile lacks its own private/task-head kernel evidence",
                    ));
                }
                (private, task)
            } else {
                (0, 0)
            };
        let warm_seed_identity_kernels = if layout.kind == Kind::ApproxWarm {
            let seed = BTreeMap::from([(
                "consume_complete_virtual_private_seed".into(),
                "Identity".into(),
            )]);
            let count = count_cuda_named(&profiles["shared_warm"], &seed)?;
            if count == 0 {
                return Err(policy_error(
                    "V3 Warm profile lacks actual accepted-seed branch execution",
                ));
            }
            count
        } else {
            0
        };
        Ok(VerifiedVirtualProfiles {
            layout: layout.layout_name().into(),
            artifact_kind: layout.artifact_kind().into(),
            initialization: placements,
            kernels,
            validator_private_kernels,
            validator_task_head_kernels,
            warm_seed_identity_kernels,
        })
    }
    pub(crate) fn finish_virtual_profile(
        &self,
        session: &mut Session,
        root: &Path,
        role: &str,
    ) -> Result<Vec<u8>, BackendError> {
        if self.virtual_layout.is_none() || !self.graphs.contains_key(role) {
            return Err(policy_error("V3 profile role is unregistered"));
        }
        let name = session.end_profiling().map_err(|e| {
            native(
                CauseCode::ProfilingFinish,
                "cannot finish V3 actual profile",
                e,
            )
        })?;
        let path = Path::new(&name);
        let metadata = std::fs::symlink_metadata(path).map_err(|e| {
            policy_error("V3 profile unavailable")
                .with_external_cause(CauseCode::ProfilingFinish, &e)
        })?;
        let actual = path.canonicalize().map_err(|e| {
            policy_error("cannot resolve V3 profile")
                .with_external_cause(CauseCode::ProfilingFinish, &e)
        })?;
        if !path.is_absolute()
            || !metadata.is_file()
            || metadata.file_type().is_symlink()
            || actual.parent() != Some(root)
            || !actual
                .file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| {
                    n.starts_with(&format!("pals-{role}-kernels_")) && n.ends_with(".json")
                })
        {
            return Err(policy_error("V3 profile escapes its exclusive role owner"));
        }
        asset::read_bounded(&actual, MAX_PROFILE_BYTES)
    }
    /// V3-only descriptor owner route. The ORT prefix is the held procfd
    /// spelling and the final file is opened once with no-follow semantics.
    /// Never canonicalize it back into an ordinary mutable parent pathname.
    pub(in crate::pals_onnx) fn finish_virtual_profile_owned(
        &self,
        session: &mut Session,
        directory: &crate::pals_onnx::virtual_review::VirtualReviewProfileDirectory,
        role: &str,
    ) -> Result<Vec<u8>, BackendError> {
        if self.virtual_layout.is_none() || !self.graphs.contains_key(role) {
            return Err(policy_error("V3 descriptor profile role is unregistered"));
        }
        directory.validate_live()?;
        let name = session.end_profiling().map_err(|error| {
            native(
                CauseCode::ProfilingFinish,
                "cannot finish V3 descriptor profile",
                error,
            )
        })?;
        directory.read_finished_profile(Path::new(&name), role, MAX_PROFILE_BYTES)
    }
}

fn count_cuda_named(bytes: &[u8], names: &BTreeMap<String, String>) -> Result<usize, BackendError> {
    let events: Vec<Value> = serde_json::from_slice(bytes).map_err(|e| {
        policy_error("cannot decode V3 private kernel evidence")
            .with_external_cause(CauseCode::ProfileParse, &e)
    })?;
    Ok(events
        .iter()
        .filter(|event| {
            event.get("cat").and_then(Value::as_str) == Some("Node")
                && event.pointer("/args/provider").and_then(Value::as_str) == Some(CUDA)
                && event
                    .get("name")
                    .and_then(Value::as_str)
                    .and_then(|n| n.strip_suffix("_kernel_time"))
                    .is_some_and(|n| {
                        names.get(n).map(String::as_str)
                            == event.pointer("/args/op_name").and_then(Value::as_str)
                    })
        })
        .count())
}
fn keys(value: &Value, expected: &[&str]) -> Result<(), BackendError> {
    let object = value
        .as_object()
        .ok_or_else(|| policy_error("V3 inventory object required"))?;
    if object.len() != expected.len() || expected.iter().any(|key| !object.contains_key(*key)) {
        return Err(policy_error("V3 inventory has unknown or absent fields"));
    }
    Ok(())
}
fn validate_nested_keys(value: &Value) -> Result<(), BackendError> {
    fn array<'a>(value: &'a Value, key: &str) -> Result<&'a Vec<Value>, BackendError> {
        value
            .get(key)
            .and_then(Value::as_array)
            .ok_or_else(|| policy_error("V3 inventory array required"))
    }
    fn seed(value: &Value) -> Result<(), BackendError> {
        keys(value, &["name", "dtype", "shape", "source"])
    }
    fn scope(value: &Value, depth: usize) -> Result<(), BackendError> {
        if depth > 32 {
            return Err(policy_error("V3 inventory recursion exceeds bound"));
        }
        keys(
            value,
            &[
                "path",
                "inputs",
                "outputs",
                "integral_initializers",
                "initializer_count",
                "nodes",
                "subgraphs",
            ],
        )?;
        for kind in ["inputs", "outputs"] {
            for descriptor in array(value, kind)? {
                keys(descriptor, &["name", "dtype", "shape"])?;
            }
        }
        for item in array(value, "integral_initializers")? {
            seed(item)?;
        }
        for node in array(value, "nodes")? {
            keys(
                node,
                &[
                    "index",
                    "name",
                    "domain",
                    "op",
                    "inputs",
                    "outputs",
                    "input_origins",
                    "constant_integral_seeds",
                    "static_annotation",
                    "actual_provider",
                    "attribute_names",
                ],
            )?;
            for item in array(node, "constant_integral_seeds")? {
                seed(item)?;
            }
        }
        for child in array(value, "subgraphs")? {
            scope(child, depth + 1)?;
        }
        Ok(())
    }
    keys(
        value,
        &[
            "schema",
            "scope",
            "manifest_sha256",
            "actual_provider_placement",
            "cpu_allowlist",
            "total_nodes",
            "artifact_kind",
            "layout",
            "producer_source_sha256",
            "admission_limits",
            "graphs",
        ],
    )?;
    for graph in array(value, "graphs")? {
        keys(
            graph,
            &[
                "role",
                "file",
                "sha256",
                "serialized_graph_name",
                "inventory",
                "weight_bindings",
                "private_neural_dependencies",
            ],
        )?;
        for binding in array(graph, "weight_bindings")? {
            keys(
                binding,
                &[
                    "initializer",
                    "parameter",
                    "dtype",
                    "shape",
                    "bytes",
                    "sha256",
                    "source_parameter_sha256",
                ],
            )?;
        }
        for dependency in array(graph, "private_neural_dependencies")? {
            keys(
                dependency,
                &[
                    "scope_path",
                    "node_index",
                    "node_name",
                    "op",
                    "initializer",
                    "parameter",
                ],
            )?;
        }
        scope(&graph["inventory"], 0)?;
    }
    Ok(())
}
fn validate_root_abi(graph: &InventoryGraphV3) -> Result<(), BackendError> {
    use serde_json::json;
    let public_inputs = vec![
        ("anchor_board", "INT64", json!(["batch", "anchors", 64])),
        ("anchor_metadata", "FLOAT", json!(["batch", "anchors", 16])),
        ("anchor_mask", "BOOL", json!(["batch", "anchors"])),
        ("records", "FLOAT", json!(["batch", "records", 8])),
        ("record_anchor", "INT64", json!(["batch", "records"])),
        (
            "record_line_tokens",
            "INT64",
            json!(["batch", "records", "line_bucket", 3]),
        ),
        (
            "record_line_mask",
            "BOOL",
            json!(["batch", "records", "line_bucket"]),
        ),
        ("record_mask", "BOOL", json!(["batch", "records"])),
    ];
    let memory = vec![
        (
            "memory_key",
            "FLOAT",
            json!(["batch", 2, "memory_tokens", 64]),
        ),
        (
            "memory_value",
            "FLOAT",
            json!(["batch", 2, "memory_tokens", 64]),
        ),
        ("memory_mask", "BOOL", json!(["batch", "memory_tokens"])),
        ("anchor_context", "FLOAT", json!(["batch", "anchors", 384])),
    ];
    let mut role_inputs = vec![
        ("role_is_critic", "BOOL", json!([])),
        memory[0].clone(),
        memory[1].clone(),
        memory[2].clone(),
        ("candidates", "INT64", json!(["batch", "candidates", 3])),
        ("candidate_mask", "BOOL", json!(["batch", "candidates"])),
        (
            "divergence_features",
            "FLOAT",
            json!(["batch", "divergences", 8]),
        ),
        ("divergence_mask", "BOOL", json!(["batch", "divergences"])),
        ("query", "FLOAT", json!(["batch", 8])),
        memory[3].clone(),
        ("query_anchor", "INT64", json!(["batch", 3])),
        (
            "query_line_tokens",
            "INT64",
            json!(["batch", 3, "line_bucket", 3]),
        ),
        (
            "query_line_mask",
            "BOOL",
            json!(["batch", 3, "line_bucket"]),
        ),
        ("record_relations", "INT64", json!(["batch", "records", 2])),
    ];
    let mut role_outputs = vec![
        ("candidate_logits", "FLOAT", json!(["batch", "candidates"])),
        ("wdl_logits", "FLOAT", json!(["batch", 3])),
        ("private_latent", "FLOAT", json!(["batch", 16, 384])),
        (
            "divergence_logits",
            "FLOAT",
            json!(["batch", "divergences"]),
        ),
        ("is_critic", "BOOL", json!([])),
    ];
    let (inputs, outputs) = match graph.role.as_str() {
        "public" => (public_inputs, memory),
        "shared_pc" => (role_inputs, role_outputs),
        "shared_warm" => {
            role_inputs.extend([
                ("initial_latent", "FLOAT", json!(["batch", 16, 384])),
                ("warm_start", "BOOL", json!([])),
            ]);
            (role_inputs, role_outputs)
        }
        "validator" => {
            role_inputs.remove(0);
            role_outputs.truncate(3);
            role_outputs.push(("task_logits", "FLOAT", json!(["batch", 7])));
            (role_inputs, role_outputs)
        }
        _ => return Err(policy_error("V3 root ABI role unsupported")),
    };
    for (actual, expected) in [
        (&graph.inventory.inputs, inputs),
        (&graph.inventory.outputs, outputs),
    ] {
        if actual.len() != expected.len()
            || actual
                .iter()
                .zip(&expected)
                .any(|(v, (n, t, s))| v.name != *n || v.dtype != *t || &v.shape != s)
        {
            return Err(policy_error("V3 root exact dtype/shape/name ABI differs"));
        }
    }
    Ok(())
}
fn validate_weight_bindings(graph: &InventoryGraphV3) -> Result<(), BackendError> {
    if graph.weight_bindings.is_empty()
        || graph.weight_bindings.len() > 5000
        || graph.private_neural_dependencies.len() > MAX_VIRTUAL_NODES * 8
    {
        return Err(policy_error(
            "V3 source parameter/dependency ledger exceeds bound",
        ));
    }
    let mut names = BTreeSet::new();
    let mut bytes = 0u64;
    for binding in &graph.weight_bindings {
        let elements = binding
            .shape
            .iter()
            .try_fold(1u64, |n, d| n.checked_mul(*d))
            .and_then(|n| n.checked_mul(4));
        if binding.initializer.is_empty()
            || binding.initializer.len() > 256
            || binding.parameter.is_empty()
            || binding.parameter.len() > 256
            || !names.insert(binding.initializer.as_str())
            || binding.dtype != "FLOAT"
            || binding.shape.is_empty()
            || binding.shape.len() > 8
            || binding.shape.iter().any(|d| *d == 0 || *d > 1048576)
            || elements != Some(binding.bytes)
            || binding.bytes > 256 * 1024 * 1024
            || (binding.initializer != binding.parameter
                && binding.initializer != format!("{}.linear_transpose", binding.parameter))
        {
            return Err(policy_error(
                "V3 source parameter ledger identity/dtype/shape/bytes differs",
            ));
        }
        strict_sha(&binding.sha256)?;
        strict_sha(&binding.source_parameter_sha256)?;
        bytes = bytes
            .checked_add(binding.bytes)
            .filter(|n| *n <= 256 * 1024 * 1024)
            .ok_or_else(|| policy_error("V3 graph weight ledger exceeds serialization bound"))?;
        if binding.parameter.starts_with("experts.") {
            let allowed = match graph.role.as_str() {
                "validator" => binding.parameter.starts_with("experts.validator."),
                "shared_pc" | "shared_warm" => {
                    binding.parameter.starts_with("experts.proposer.")
                        || binding.parameter.starts_with("experts.critic.")
                }
                _ => false,
            };
            if !allowed {
                return Err(policy_error(
                    "V3 source weight ledger includes a different private role",
                ));
            }
        }
    }
    Ok(())
}
struct UniqueJson(Value);
impl<'de> Deserialize<'de> for UniqueJson {
    fn deserialize<D: serde::Deserializer<'de>>(decoder: D) -> Result<Self, D::Error> {
        struct Visitor;
        impl<'de> serde::de::Visitor<'de> for Visitor {
            type Value = UniqueJson;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("unique bounded JSON")
            }
            fn visit_bool<E: serde::de::Error>(self, v: bool) -> Result<Self::Value, E> {
                Ok(UniqueJson(v.into()))
            }
            fn visit_i64<E: serde::de::Error>(self, v: i64) -> Result<Self::Value, E> {
                Ok(UniqueJson(v.into()))
            }
            fn visit_u64<E: serde::de::Error>(self, v: u64) -> Result<Self::Value, E> {
                Ok(UniqueJson(v.into()))
            }
            fn visit_f64<E: serde::de::Error>(self, v: f64) -> Result<Self::Value, E> {
                serde_json::Number::from_f64(v)
                    .map(|n| UniqueJson(n.into()))
                    .ok_or_else(|| E::custom("nonfinite JSON"))
            }
            fn visit_str<E: serde::de::Error>(self, v: &str) -> Result<Self::Value, E> {
                Ok(UniqueJson(v.into()))
            }
            fn visit_string<E: serde::de::Error>(self, v: String) -> Result<Self::Value, E> {
                Ok(UniqueJson(v.into()))
            }
            fn visit_unit<E: serde::de::Error>(self) -> Result<Self::Value, E> {
                Ok(UniqueJson(Value::Null))
            }
            fn visit_seq<A: serde::de::SeqAccess<'de>>(
                self,
                mut seq: A,
            ) -> Result<Self::Value, A::Error> {
                let mut values = Vec::new();
                while let Some(v) = seq.next_element::<UniqueJson>()? {
                    values.push(v.0);
                }
                Ok(UniqueJson(values.into()))
            }
            fn visit_map<A: serde::de::MapAccess<'de>>(
                self,
                mut map: A,
            ) -> Result<Self::Value, A::Error> {
                let mut values = serde_json::Map::new();
                while let Some(key) = map.next_key::<String>()? {
                    if values.contains_key(&key) {
                        return Err(serde::de::Error::custom("duplicate JSON key"));
                    }
                    values.insert(key, map.next_value::<UniqueJson>()?.0);
                }
                Ok(UniqueJson(values.into()))
            }
        }
        decoder.deserialize_any(Visitor)
    }
}

impl PalsCudaControlPolicy {
    pub(in crate::pals_onnx) fn record_initial_log_virtual(
        &self,
        log: &Arc<Mutex<PlacementLog>>,
        path: &Path,
        role: &str,
        session_created: bool,
        gate: Option<&Result<PalsGraphPlacement, BackendError>>,
    ) -> Result<(), BackendError> {
        use std::io::Write;
        let log = log
            .lock()
            .map_err(|_| policy_error("PALS placement collector was poisoned"))?;
        if self.virtual_layout.is_none() || !self.graphs.contains_key(role) || !path.is_absolute() {
            return Err(policy_error("PALS placement metadata owner is invalid"));
        }
        let mut selected = None;
        let mut headers = Vec::new();
        let mut nodes = Vec::new();
        let mut unparsed = Vec::new();
        let mut transfer_metadata = Vec::new();
        let mut transfer_unparsed = Vec::new();
        let mut began = false;
        for (sequence, line) in log.lines.iter().enumerate() {
            let line = line.trim();
            if line == "Node placements" {
                began = true;
                continue;
            }
            let header = placement_header(line, "Node(s) placed on [")
                .map(|(provider, count)| (provider, count, false))
                .or_else(|| {
                    placement_header(line, "All nodes placed on [")
                        .map(|(provider, count)| (provider, count, true))
                });
            if let Some((provider, count, all_nodes_summary)) = header {
                if provider.len() <= 256 {
                    selected = Some(provider.clone());
                    headers.push(PlacementHeader {
                        sequence,
                        provider,
                        declared_nodes: count,
                        all_nodes_summary,
                    });
                    continue;
                }
            }
            if began {
                if let Some((op, name)) = line
                    .split_once(" (")
                    .and_then(|(op, rest)| rest.strip_suffix(')').map(|name| (op, name)))
                {
                    if op.len() <= 256 && name.len() <= 256 {
                        nodes.push(PlacementNode {
                            sequence,
                            provider: selected.clone(),
                            name,
                            op,
                        });
                        continue;
                    }
                }
            }
            // Preserve incomplete-stream identity without copying arbitrary logger
            // text or any values into the metadata-only file.
            unparsed.push(PlacementUnparsed {
                sequence,
                line_sha256: asset::sha256(line.as_bytes()),
            });
        }
        for (sequence, line) in log.transfer_lines.iter().enumerate() {
            if let Some(metadata) = copy_metadata(line.trim()) {
                transfer_metadata.push(metadata);
            } else {
                transfer_unparsed.push(PlacementUnparsed {
                    sequence,
                    line_sha256: asset::sha256(line.as_bytes()),
                });
            }
        }
        let evidence = PlacementFile {
            schema: "rovezero.pals-virtual-initial-placement.v1",
            source: "ort-1.22-finalize-recursive-plus-memcpy-origin-metadata",
            optimization: PalsGraphOptimization::Disable,
            role,
            session_created,
            before_first_neural_run: true,
            gate: match gate {
                Some(Ok(_)) => "passed",
                Some(Err(_)) => "rejected",
                None => "not_reached",
            },
            gate_detail: gate
                .and_then(|result| result.as_ref().err())
                .map(|error| error.detail),
            collected_bytes: log.bytes,
            collector_overflow: log.overflow,
            provider_headers: headers,
            nodes,
            unparsed,
            transfer_metadata,
            transfer_unparsed,
            approved_transfers: gate
                .and_then(|result| result.as_ref().ok())
                .map(|placement| placement.approved_transfers.as_slice()),
        };
        let mut json = BoundedPlacementJson { bytes: Vec::new() };
        serde_json::to_writer_pretty(&mut json, &evidence).map_err(|error| {
            policy_error("cannot serialize bounded PALS placement metadata")
                .with_external_cause(CauseCode::ProfilingFinish, &error)
        })?;
        json.write_all(b"\n").map_err(|error| {
            policy_error("PALS placement metadata exceeds bounded output")
                .with_external_cause(CauseCode::ProfilingFinish, &error)
        })?;
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .map_err(|error| {
                policy_error("cannot create exclusive PALS placement metadata")
                    .with_external_cause(CauseCode::ProfilingStart, &error)
            })?;
        file.write_all(&json.bytes)
            .and_then(|()| file.sync_all())
            .map_err(|error| {
                policy_error("cannot preserve bounded PALS placement metadata")
                    .with_external_cause(CauseCode::ProfilingFinish, &error)
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use sha2::{Digest as _, Sha256};

    fn binding() -> WeightBindingV3 {
        WeightBindingV3 {
            initializer: "experts.validator.task_head.weight.linear_transpose".into(),
            parameter: "experts.validator.task_head.weight".into(),
            dtype: "FLOAT".into(),
            shape: vec![384, 7],
            bytes: 384 * 7 * 4,
            sha256: "11".repeat(32),
            source_parameter_sha256: "22".repeat(32),
        }
    }
    fn graph() -> InventoryGraphV3 {
        InventoryGraphV3 {
            role: "validator".into(),
            file: "role_validator.onnx".into(),
            sha256: "33".repeat(32),
            serialized_graph_name: VALIDATOR_GRAPH.into(),
            inventory: serde_json::from_value(json!({"path":"validator","inputs":[],"outputs":[],
                "integral_initializers":[],"initializer_count":1,"nodes":[],"subgraphs":[]}))
            .unwrap(),
            weight_bindings: vec![binding()],
            private_neural_dependencies: vec![],
        }
    }
    fn policy(manifest: [u8; 32], bindings: Vec<WeightBindingV3>) -> PalsCudaControlPolicy {
        PalsCudaControlPolicy {
            manifest,
            inventory: [7; 32],
            graphs: BTreeMap::new(),
            virtual_layout: Some(VirtualReviewControlLayout {
                kind: Kind::FreshCollection,
                validator_private: BTreeMap::new(),
                validator_task_head: BTreeMap::new(),
                weight_bindings: BTreeMap::from([("validator".into(), bindings)]),
            }),
        }
    }
    #[test]
    fn virtual_inventory_rejects_legacy_unknown_and_duplicate_keys() {
        assert!(
            PalsCudaControlPolicy::from_virtual_review_bytes(br#"{"schema":"legacy"}"#).is_err()
        );
        assert!(
            PalsCudaControlPolicy::from_virtual_review_bytes(br#"{"schema":1,"schema":2}"#)
                .is_err()
        );
        assert!(PalsCudaControlPolicy::from_virtual_review_bytes(&vec![
            0;
            MAX_VIRTUAL_INVENTORY_BYTES
                + 1
        ])
        .is_err());
    }
    #[test]
    fn owned_inventory_bytes_require_actual_pin_before_strict_schema_admission() {
        let bytes = br#"{"schema":"legacy"}"#;
        let wrong =
            PalsCudaControlPolicy::from_virtual_review_inventory_bytes(bytes, &"00".repeat(32))
                .err()
                .expect("incorrect pin must fail before schema admission");
        assert_eq!(wrong.detail, "V3 control inventory digest differs");
        let actual = format!("{:x}", Sha256::digest(bytes));
        assert!(
            PalsCudaControlPolicy::from_virtual_review_inventory_bytes(bytes, &actual).is_err()
        );
        assert!(PalsCudaControlPolicy::from_virtual_review_inventory_bytes(b"", &actual).is_err());
    }
    #[test]
    fn source_weight_metadata_cannot_cross_private_roles_or_forge_byte_shape() {
        assert!(validate_weight_bindings(&graph()).is_ok());
        let mut value = graph();
        value.weight_bindings[0].bytes -= 4;
        assert!(validate_weight_bindings(&value).is_err());
        let mut value = graph();
        value.role = "public".into();
        assert!(validate_weight_bindings(&value).is_err());
        let mut value = graph();
        value.weight_bindings[0].sha256 = "AB".repeat(32);
        assert!(validate_weight_bindings(&value).is_err());
    }
    #[test]
    fn weight_ledger_requires_actual_pinned_manifest_bytes_and_exact_rows() {
        let binding = binding();
        let mut row = serde_json::to_value(&binding).unwrap();
        row.as_object_mut().unwrap().remove("dtype");
        let bytes =
            serde_json::to_vec(&json!({"initializer_banks":{"validator":{"parameters":[row]}}}))
                .unwrap();
        let value = policy(asset::sha256(&bytes), vec![binding.clone()]);
        assert!(value
            .validate_virtual_manifest_weight_bindings(&bytes)
            .is_ok());
        let mut changed = serde_json::from_slice::<Value>(&bytes).unwrap();
        changed["initializer_banks"]["validator"]["parameters"][0]["bytes"] = 1.into();
        let changed = serde_json::to_vec(&changed).unwrap();
        assert!(value
            .validate_virtual_manifest_weight_bindings(&changed)
            .is_err());
        let value = policy(asset::sha256(&changed), vec![binding]);
        assert!(value
            .validate_virtual_manifest_weight_bindings(&changed)
            .is_err());
    }
    #[test]
    fn missing_v_actual_evidence_cannot_issue_a_pc_only_witness() {
        let value = policy([0; 32], vec![binding()]);
        let profiles = BTreeMap::from([
            ("public".into(), b"[]".to_vec()),
            ("shared_pc".into(), b"[]".to_vec()),
        ]);
        assert!(value
            .verify_virtual_profiles(&profiles, vec![], [1; 32], [2; 32])
            .is_err());
    }
}
