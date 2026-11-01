//! Explicit metadata-only CPU control admission for a pinned PALS graph.
//! Strict CUDA remains the default. Initialization placement is checked before
//! any Run; later kernel profiles attest execution and cannot replace that gate.
use super::*;
use ort::logging::LogLevel;
use ort::session::builder::SessionBuilder;
use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

mod virtual_review;
#[cfg(target_os = "linux")]
pub use virtual_review::SystemVirtualReviewCudaPlacementWitness;
pub use virtual_review::{VirtualReviewControlLayout, VirtualReviewCudaPlacementWitness};

const POLICY_SCHEMA: &str = "rovezero.pals-cuda-metadata-control.v2";
const MAX_INVENTORY_BYTES: usize = 1024 * 1024;
const MAX_PLACEMENT_BYTES: usize = 1024 * 1024;
const MAX_PROFILE_BYTES: usize = 4 * 1024 * 1024;
const MAX_NODES: usize = 5000;
const CUDA: &str = "CUDAExecutionProvider";
const CPU: &str = "CPUExecutionProvider";
const CUDA_WARM_GRAPH_NAME: &str = "shared_pc_if_approx_cuda_warm_v2";
const CUDA_WARM_IF_ANNOTATION: &str = "cuda_warm_if_dispatch_candidate";

fn policy_error(detail: &'static str) -> BackendError {
    fail(K::BackendUnavailable, S::Backend, detail)
}

#[derive(Clone)]
pub struct PalsCudaControlPolicy {
    pub(super) manifest: [u8; 32],
    pub(super) inventory: [u8; 32],
    graphs: BTreeMap<String, GraphPolicy>,
    /// Explicit V3 admission domain. Historical inventory construction remains
    /// None and keeps its original P/C-only layout and witness checks.
    virtual_layout: Option<VirtualReviewControlLayout>,
}
#[derive(Clone)]
struct GraphPolicy {
    digest: [u8; 32],
    nodes: BTreeMap<String, String>,
    controls: BTreeMap<String, String>,
    major: BTreeMap<String, String>,
    proposer_private: BTreeMap<String, String>,
    critic_private: BTreeMap<String, String>,
    role_transfer: Option<RoleTransferPolicy>,
    required_cuda_dispatch: BTreeSet<String>,
}
#[derive(Clone)]
struct RoleTransferPolicy {
    scope: String,
    destination_node: String,
    host_condition_nodes: Vec<String>,
}
#[derive(Clone)]
struct CudaWarmControlRoute {
    root_scope: String,
    mode_node: String,
    role_transfer: RoleTransferPolicy,
    required_cuda_dispatch: BTreeSet<String>,
}
#[derive(Clone, Copy)]
enum PrivateRole {
    Proposer,
    Critic,
}
#[derive(Deserialize)]
struct Inventory {
    schema: String,
    scope: String,
    manifest_sha256: String,
    actual_provider_placement: String,
    cpu_allowlist: String,
    total_nodes: usize,
    graphs: Vec<InventoryGraph>,
}
#[derive(Deserialize)]
struct InventoryGraph {
    role: String,
    sha256: String,
    /// Optional for historical role-only inventories. The additional route is
    /// admitted only with the exact registered serialized CUDA Warm name.
    #[serde(default)]
    serialized_graph_name: Option<String>,
    inventory: InventoryScope,
}
#[derive(Deserialize)]
struct InventoryScope {
    path: String,
    inputs: Vec<InventoryInput>,
    #[serde(default)]
    outputs: Vec<InventoryInput>,
    integral_initializers: Vec<IntegralSeed>,
    /// Dense plus sparse serialized initializers; required to be zero in each
    /// new Warm branch. Historical inventories keep their existing contract.
    #[serde(default)]
    initializer_count: Option<usize>,
    nodes: Vec<InventoryNode>,
    subgraphs: Vec<InventoryScope>,
}
#[derive(Deserialize)]
struct InventoryInput {
    name: String,
    dtype: String,
    shape: serde_json::Value,
}
#[derive(Deserialize)]
struct IntegralSeed {
    name: String,
    dtype: String,
    shape: Vec<u64>,
    source: String,
}
impl IntegralSeed {
    fn validate(&self, source: &str) -> Result<(), BackendError> {
        if self.name.is_empty()
            || self.name.len() > 256
            || self.source != source
            || !matches!(self.dtype.as_str(), "INT64" | "INT32" | "BOOL")
            || self.shape.len() > 1
            || self.shape.iter().any(|dimension| *dimension > 64)
        {
            return Err(policy_error(
                "PALS metadata seed dtype/shape/provenance is invalid",
            ));
        }
        Ok(())
    }
}
#[derive(Clone, Copy, Eq, PartialEq)]
enum Origin {
    Metadata,
    Integral,
    Role,
    CudaWarmMode,
    Data,
}
impl Origin {
    fn label(self) -> &'static str {
        match self {
            Self::Metadata => "shape_metadata",
            Self::Integral => "small_integral_constant",
            Self::Role => "role_control",
            Self::CudaWarmMode => "cuda_warm_mode_control",
            Self::Data => "model_data_or_unknown",
        }
    }
}
#[derive(Deserialize)]
struct InventoryNode {
    index: usize,
    name: String,
    domain: String,
    op: String,
    inputs: Vec<String>,
    outputs: Vec<String>,
    input_origins: Vec<String>,
    constant_integral_seeds: Vec<IntegralSeed>,
    static_annotation: String,
    actual_provider: String,
    #[serde(default)]
    attribute_names: Option<Vec<String>>,
}
impl PalsCudaControlPolicy {
    /// Explicit reviewed source inventory, not runtime-discovered fallback.
    /// Hash pinning grants no GPU success: every session still needs its own
    /// complete pre-Run placement audit and actual selected P/C kernel witness.
    pub fn from_inventory(path: &Path, expected_sha256: &str) -> Result<Self, BackendError> {
        if !path.is_absolute() {
            return Err(policy_error("PALS control inventory must be absolute"));
        }
        let bytes = asset::read_bounded(path, MAX_INVENTORY_BYTES)?;
        let digest = asset::sha256(&bytes);
        if digest != parse_sha256(expected_sha256)? {
            return Err(policy_error("PALS control inventory hash differs"));
        }
        let inventory: Inventory = serde_json::from_slice(&bytes).map_err(|error| {
            policy_error("cannot decode pinned PALS control inventory")
                .with_external_cause(CauseCode::ProfileParse, &error)
        })?;
        if inventory.schema != "rovezero.pals-static-control-inventory.v2"
            || inventory.scope != "read_only_serialized_graph_metadata_no_runtime"
            || inventory.actual_provider_placement != "not_observed"
            || inventory.cpu_allowlist != "not_created"
            || !(1..=MAX_NODES).contains(&inventory.total_nodes)
            || inventory.graphs.len() != 2
        {
            return Err(policy_error(
                "PALS inventory scope is not the reviewed static graph schema",
            ));
        }
        let mut graphs = BTreeMap::new();
        let mut count = 0;
        for graph in inventory.graphs {
            if !matches!(graph.role.as_str(), "public" | "shared_pc")
                || graphs.contains_key(&graph.role)
            {
                return Err(policy_error(
                    "PALS control graph role is unsupported or duplicated",
                ));
            }
            let warm_route = if graph.serialized_graph_name.as_deref() == Some(CUDA_WARM_GRAPH_NAME)
            {
                Some(cuda_warm_control_route(&graph.role, &graph.inventory)?)
            } else {
                None
            };
            let mut policy = GraphPolicy {
                digest: parse_sha256(&graph.sha256)?,
                nodes: BTreeMap::new(),
                controls: BTreeMap::new(),
                major: BTreeMap::new(),
                proposer_private: BTreeMap::new(),
                critic_private: BTreeMap::new(),
                role_transfer: None,
                required_cuda_dispatch: warm_route
                    .as_ref()
                    .map(|route| route.required_cuda_dispatch.clone())
                    .unwrap_or_default(),
            };
            let mut names = BTreeSet::new();
            if let Some(route) = &warm_route {
                inspect_scope_with_layout(
                    &graph.inventory,
                    &BTreeMap::new(),
                    &mut policy,
                    &mut names,
                    &mut count,
                    Some(route),
                    None,
                )?;
            } else {
                inspect_scope(
                    &graph.inventory,
                    &BTreeMap::new(),
                    &mut policy,
                    &mut names,
                    &mut count,
                )?;
            }
            policy.role_transfer = warm_route
                .map(|route| route.role_transfer)
                .or_else(|| role_transfer_policy(&graph.role, &graph.inventory));
            if policy.major.is_empty() {
                return Err(policy_error(
                    "PALS control inventory has no neural computation",
                ));
            }
            graphs.insert(graph.role, policy);
        }
        if count != inventory.total_nodes
            || count > MAX_NODES
            || graphs["shared_pc"].proposer_private.is_empty()
            || graphs["shared_pc"].critic_private.is_empty()
        {
            return Err(policy_error(
                "PALS inventory has incomplete node or P/C branch coverage",
            ));
        }
        Ok(Self {
            manifest: parse_sha256(&inventory.manifest_sha256)?,
            inventory: digest,
            graphs,
            virtual_layout: None,
        })
    }
    pub(super) fn validate_pins(
        &self,
        manifest: [u8; 32],
        graph: &GraphManifest,
    ) -> Result<(), BackendError> {
        let expected = self.graphs.get(&graph.role).ok_or_else(|| {
            policy_error("PALS explicit control policy is P/C shared-layout only")
        })?;
        if self.manifest != manifest || expected.digest != parse_sha256(&graph.sha256)? {
            return Err(policy_error(
                "PALS explicit control policy graph/manifest pin differs",
            ));
        }
        Ok(())
    }
    pub(super) fn configure(
        &self,
        builder: SessionBuilder,
        role: &str,
        profile_prefix: &Path,
    ) -> Result<(SessionBuilder, Arc<Mutex<PlacementLog>>), BackendError> {
        if !self.graphs.contains_key(role) {
            return Err(policy_error(
                "PALS graph is absent from explicit CUDA control policy",
            ));
        }
        let log = Arc::new(Mutex::new(PlacementLog::default()));
        let captured = Arc::clone(&log);
        let id = format!("pals-placement-{role}");
        let expected_id = id.clone();
        let builder = builder
            .with_log_id(id)
            .and_then(|builder| builder.with_log_level(LogLevel::Verbose))
            .and_then(|builder| builder.with_log_verbosity(1))
            .and_then(|builder| {
                builder.with_logger(Box::new(move |level, _, id, location, message| {
                    // rc.10 mistakenly reads category from code_location. Only id,
                    // location and message are usable. Never panic across this FFI.
                    // The same ORT function emits a warning after mixed node
                    // lists. Only its Verbose placement stream is parseable.
                    if level == LogLevel::Verbose
                        && id == expected_id
                        && location.contains("VerifyEachNodeIsAssignedToAnEp")
                    {
                        if let Ok(mut log) = captured.lock() {
                            log.capture(message);
                        }
                    } else if id == expected_id
                        && location.contains("transformer_memcpy.cc")
                        && ((level == LogLevel::Info && location.contains("AddCopyNode"))
                            || (level == LogLevel::Warning && location.contains("ApplyImpl")))
                    {
                        if let Ok(mut log) = captured.lock() {
                            log.capture_transfer(message);
                        }
                    }
                }))
            })
            .and_then(|builder| builder.with_profiling(profile_prefix))
            .map_err(|error| {
                native(
                    CauseCode::SessionConfiguration,
                    "cannot enable explicit PALS placement audit",
                    error,
                )
            })?;
        Ok((builder, log))
    }
    pub(super) fn verify_initialization(
        &self,
        role: &str,
        log: &Arc<Mutex<PlacementLog>>,
    ) -> Result<PalsGraphPlacement, BackendError> {
        let log = log
            .lock()
            .map_err(|_| policy_error("PALS placement collector was poisoned"))?;
        let graph = self
            .graphs
            .get(role)
            .ok_or_else(|| policy_error("unknown placement graph"))?;
        verify_placement(role, graph, &log)
    }
    pub(super) fn verify_profiles(
        &self,
        public: &[u8],
        shared: &[u8],
        placements: Vec<PalsGraphPlacement>,
        runtime: [u8; 32],
        bundle: [u8; 32],
    ) -> Result<PalsCudaPlacementWitness, BackendError> {
        if self.virtual_layout.is_some() {
            return Err(policy_error(
                "V3 control policy requires its separate complete graph witness",
            ));
        }
        let mut roles = BTreeSet::new();
        if placements.len() != 2
            || placements.iter().any(|placement| {
                let Some(graph) = self.graphs.get(&placement.role) else {
                    return true;
                };
                !roles.insert(placement.role.clone())
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
            })
        {
            return Err(policy_error(
                "PALS kernel witness lacks both pinned pre-Run graph placements",
            ));
        }
        let transfers = |role: &str| {
            placements
                .iter()
                .find(|placement| placement.role == role)
                .map(|placement| placement.approved_transfers.as_slice())
                .unwrap_or(&[])
        };
        let public = verify_profile(public, &self.graphs["public"], transfers("public"))?;
        let shared = verify_profile(shared, &self.graphs["shared_pc"], transfers("shared_pc"))?;
        if public.neural_kernels == 0
            || shared.proposer_private_kernels == 0
            || shared.critic_private_kernels == 0
        {
            return Err(policy_error(
                "PALS profiles do not prove public and both selected P/C neural branches",
            ));
        }
        Ok(PalsCudaPlacementWitness {
            schema: POLICY_SCHEMA.into(),
            inventory_sha256: self.inventory,
            manifest_sha256: self.manifest,
            runtime_sha256: runtime,
            runtime_bundle_sha256: bundle,
            optimization: PalsGraphOptimization::Disable,
            initialization: placements,
            public,
            shared_pc: shared,
            category_provenance: "rc10-category-unavailable-id-location-message-used".into(),
        })
    }
}

fn inspect_scope(
    scope: &InventoryScope,
    inherited: &BTreeMap<String, Origin>,
    policy: &mut GraphPolicy,
    names: &mut BTreeSet<String>,
    count: &mut usize,
) -> Result<(), BackendError> {
    inspect_scope_with_layout(scope, inherited, policy, names, count, None, None)
}

fn inspect_scope_with_layout(
    scope: &InventoryScope,
    inherited: &BTreeMap<String, Origin>,
    policy: &mut GraphPolicy,
    names: &mut BTreeSet<String>,
    count: &mut usize,
    warm_route: Option<&CudaWarmControlRoute>,
    private_role: Option<PrivateRole>,
) -> Result<(), BackendError> {
    let mut origins = inherited.clone();
    for input in &scope.inputs {
        let origin = if input.name == "role_is_critic" {
            if input.dtype != "BOOL" || input.shape != serde_json::json!([]) {
                return Err(policy_error("PALS role control seed is not scalar BOOL"));
            }
            Origin::Role
        } else if input.name == "warm_start"
            && warm_route.is_some_and(|route| route.root_scope == scope.path)
        {
            if input.dtype != "BOOL" || input.shape != serde_json::json!([]) {
                return Err(policy_error("PALS CUDA Warm mode is not scalar BOOL"));
            }
            Origin::CudaWarmMode
        } else {
            Origin::Data
        };
        origins.insert(input.name.clone(), origin);
    }
    for seed in &scope.integral_initializers {
        seed.validate("graph_initializer_metadata")?;
        if origins
            .insert(seed.name.clone(), Origin::Integral)
            .is_some()
        {
            return Err(policy_error(
                "PALS metadata initializer shadows another input/seed",
            ));
        }
    }
    let mut child_count = 0;
    for (index, node) in scope.nodes.iter().enumerate() {
        *count += 1;
        if *count > MAX_NODES
            || node.index != index
            || node.name.is_empty()
            || node.name.len() > 256
            || !names.insert(node.name.clone())
            || node.actual_provider != "unknown"
            || !matches!(node.domain.as_str(), "" | "ai.onnx")
        {
            return Err(policy_error(
                "PALS inventory node identity is ambiguous or unsupported",
            ));
        }
        if node.inputs.len() > 64
            || node.outputs.is_empty()
            || node.outputs.len() > 16
            || node.input_origins.len() != node.inputs.len()
        {
            return Err(policy_error("PALS metadata node has invalid IO provenance"));
        }
        policy.nodes.insert(node.name.clone(), node.op.clone());
        let actual_origins: Vec<_> = node
            .inputs
            .iter()
            .map(|name| origins.get(name).copied().unwrap_or(Origin::Data))
            .collect();
        if node
            .inputs
            .iter()
            .zip(&actual_origins)
            .zip(&node.input_origins)
            .any(|((name, origin), recorded)| {
                recorded
                    != if name.is_empty() {
                        "absent_optional_input"
                    } else {
                        origin.label()
                    }
            })
        {
            return Err(policy_error(
                "PALS metadata input-flow provenance cannot be independently reproduced",
            ));
        }
        let mut output_origin = Origin::Data;
        if !node.constant_integral_seeds.is_empty() {
            if node.op != "Constant" || node.constant_integral_seeds.len() != node.outputs.len() {
                return Err(policy_error("PALS integral Constant source is malformed"));
            }
            for seed in &node.constant_integral_seeds {
                seed.validate("constant_node_tensor_metadata")?;
                if !node.outputs.contains(&seed.name) {
                    return Err(policy_error(
                        "PALS Constant seed does not identify its exact output",
                    ));
                }
            }
            output_origin = Origin::Integral;
        }
        let allowed = match node.static_annotation.as_str() {
            "shape_metadata_candidate" => {
                let valid = matches!(node.op.as_str(), "Shape" | "Size")
                    && node.inputs.len() == 1
                    && node.outputs.len() == 1;
                if valid {
                    output_origin = Origin::Metadata;
                }
                valid
            }
            "shape_metadata_chain_candidate" => {
                // Exact named flow, not a general opcode whitelist. The pinned
                // producer includes bounded integral initializer/Constant
                // seeds. Captured scope provenance is independently recomputed.
                matches!(
                    node.op.as_str(),
                    "Gather" | "Concat" | "Unsqueeze" | "Slice" | "Mul"
                ) && !node.inputs.is_empty()
                    && actual_origins.contains(&Origin::Metadata)
                    && actual_origins
                        .iter()
                        .all(|origin| matches!(origin, Origin::Metadata | Origin::Integral))
            }
            "role_scalar_identity_candidate" => {
                let valid = node.op == "Identity"
                    && node.inputs == ["role_is_critic"]
                    && actual_origins == [Origin::Role];
                if valid {
                    output_origin = Origin::Role;
                }
                valid
            }
            _ => false,
        };
        if allowed {
            policy.controls.insert(node.name.clone(), node.op.clone());
            if output_origin == Origin::Data {
                output_origin = Origin::Metadata;
            }
        } else if node.static_annotation == "shape_metadata_chain_candidate"
            || node.static_annotation == "shape_metadata_candidate"
            || node.static_annotation == "role_scalar_identity_candidate"
        {
            return Err(policy_error(
                "PALS metadata annotation conflicts with exact node flow",
            ));
        }
        if node.static_annotation == "major_nn_requires_cuda" {
            if !matches!(
                node.op.as_str(),
                "MatMul"
                    | "Gemm"
                    | "Conv"
                    | "Attention"
                    | "MultiHeadAttention"
                    | "Softmax"
                    | "LayerNormalization"
                    | "SimplifiedLayerNormalization"
                    | "SkipLayerNormalization"
                    | "Gelu"
                    | "FastGelu"
                    | "BiasGelu"
                    | "Relu"
                    | "Sigmoid"
                    | "Tanh"
            ) {
                return Err(policy_error(
                    "PALS neural annotation is not a reviewed neural opcode",
                ));
            }
            policy.major.insert(node.name.clone(), node.op.clone());
            match private_role {
                Some(PrivateRole::Critic) => {
                    policy
                        .critic_private
                        .insert(node.name.clone(), node.op.clone());
                }
                Some(PrivateRole::Proposer) => {
                    policy
                        .proposer_private
                        .insert(node.name.clone(), node.op.clone());
                }
                None => {}
            }
        }
        // If dispatch stays on CUDA EP. Its CPU condition does not grant a CPU
        // fallback permission to this node or to its private branch kernels.
        let role_if = node.op == "If"
            && node.static_annotation == "role_if_dispatch_candidate"
            && node.inputs == ["role_is_critic"]
            && actual_origins == [Origin::Role];
        let cuda_warm_if = node.op == "If"
            && node.static_annotation == CUDA_WARM_IF_ANNOTATION
            && node.inputs == ["warm_start"]
            && actual_origins == [Origin::CudaWarmMode]
            && warm_route.is_some_and(|route| {
                route.root_scope == scope.path && route.mode_node == node.name
            });
        if node.op == "If" && !role_if && !cuda_warm_if {
            return Err(policy_error(
                "PALS If control is not the exact registered role or CUDA Warm scalar",
            ));
        }
        // Capture the lexical outer scope at this exact node. Future outputs,
        // especially the If's own outputs, cannot seed its private branch.
        let prefix = format!("{}/node{}:{}/", scope.path, node.index, node.name);
        for child in scope
            .subgraphs
            .iter()
            .filter(|child| child.path.starts_with(&prefix))
        {
            if node.op != "If" {
                return Err(policy_error("PALS has an undeclared nested graph"));
            }
            let child_role = if role_if {
                match child.path.strip_prefix(&prefix) {
                    Some("then_branch") => Some(PrivateRole::Critic),
                    Some("else_branch") => Some(PrivateRole::Proposer),
                    _ => {
                        return Err(policy_error(
                            "PALS private role branch lacks its exact direct If scope",
                        ));
                    }
                }
            } else {
                private_role
            };
            inspect_scope_with_layout(
                child, &origins, policy, names, count, warm_route, child_role,
            )?;
            child_count += 1;
        }
        for name in &node.outputs {
            if name.is_empty()
                || name.len() > 256
                || origins.insert(name.clone(), output_origin).is_some()
            {
                return Err(policy_error(
                    "PALS metadata output aliases another source value",
                ));
            }
        }
    }
    if child_count != scope.subgraphs.len() {
        return Err(policy_error(
            "PALS source subgraph is missing its exact parent node",
        ));
    }
    Ok(())
}

fn descriptor(
    values: &[InventoryInput],
    name: &str,
    dtype: &str,
    shape: &serde_json::Value,
) -> bool {
    let mut matched = values.iter().filter(|value| value.name == name);
    matched
        .next()
        .is_some_and(|value| value.dtype == dtype && &value.shape == shape)
        && matched.next().is_none()
}

fn exported_serial_name(name: &str, stem: &str) -> bool {
    name.strip_prefix(stem).is_some_and(|serial| {
        !serial.is_empty()
            && serial.bytes().all(|byte| byte.is_ascii_digit())
            && serial.parse::<u64>().is_ok_and(|value| value > 0)
    })
}

fn exact_if_attributes(node: &InventoryNode) -> bool {
    node.attribute_names.as_ref().is_some_and(|names| {
        names.len() == 2
            && names.iter().filter(|name| *name == "then_branch").count() == 1
            && names.iter().filter(|name| *name == "else_branch").count() == 1
    })
}

fn direct_if_branches<'a>(
    scope: &'a InventoryScope,
    node: &InventoryNode,
) -> Result<(&'a InventoryScope, &'a InventoryScope), BackendError> {
    let prefix = format!("{}/node{}:{}/", scope.path, node.index, node.name);
    let then_path = format!("{prefix}then_branch");
    let else_path = format!("{prefix}else_branch");
    let nested: Vec<_> = scope
        .subgraphs
        .iter()
        .filter(|child| child.path.starts_with(&prefix))
        .collect();
    if nested.len() != 2 || !exact_if_attributes(node) {
        return Err(policy_error(
            "PALS CUDA Warm If needs exactly its two direct branches",
        ));
    }
    let then = nested.iter().find(|child| child.path == then_path).copied();
    let otherwise = nested.iter().find(|child| child.path == else_path).copied();
    match (then, otherwise) {
        (Some(then), Some(otherwise)) => Ok((then, otherwise)),
        _ => Err(policy_error(
            "PALS CUDA Warm branches differ from the exact If scope",
        )),
    }
}

fn consumers<'a>(
    scope: &'a InventoryScope,
    input: &str,
    result: &mut Vec<(&'a InventoryScope, &'a InventoryNode)>,
) {
    result.extend(
        scope
            .nodes
            .iter()
            .filter(|node| node.inputs.iter().any(|name| name == input))
            .map(|node| (scope, node)),
    );
    for child in &scope.subgraphs {
        consumers(child, input, result);
    }
}

fn cuda_warm_control_route(
    role: &str,
    scope: &InventoryScope,
) -> Result<CudaWarmControlRoute, BackendError> {
    let scalar = serde_json::json!([]);
    let latent = serde_json::json!(["batch", 16, 384]);
    if role != "shared_pc"
        || scope.path != "shared_pc"
        || !descriptor(&scope.inputs, "role_is_critic", "BOOL", &scalar)
        || !descriptor(&scope.inputs, "warm_start", "BOOL", &scalar)
        || !descriptor(&scope.inputs, "initial_latent", "FLOAT", &latent)
        || !descriptor(&scope.outputs, "is_critic", "BOOL", &scalar)
    {
        return Err(policy_error(
            "PALS registered CUDA Warm root ABI is invalid",
        ));
    }
    let mut modes = scope
        .nodes
        .iter()
        .filter(|node| node.name == "private_warm_mode");
    let mode = modes
        .next()
        .ok_or_else(|| policy_error("PALS CUDA Warm mode is absent"))?;
    if modes.next().is_some()
        || mode.op != "If"
        || !mode.domain.is_empty()
        || mode.inputs != ["warm_start"]
        || mode.outputs.len() != 1
        || mode.static_annotation != CUDA_WARM_IF_ANNOTATION
        || !mode.constant_integral_seeds.is_empty()
        || !exported_serial_name(&mode.outputs[0], "selected_private_initial_")
    {
        return Err(policy_error(
            "PALS CUDA Warm mode is not the exact initial selector",
        ));
    }
    let (warm, fresh) = direct_if_branches(scope, mode)?;
    let warm_output = format!("{}__warm_branch", mode.outputs[0]);
    let fresh_output = format!("{}__fresh_branch", mode.outputs[0]);
    for (branch, output) in [(warm, &warm_output), (fresh, &fresh_output)] {
        if !branch.inputs.is_empty()
            || branch.initializer_count != Some(0)
            || !branch.integral_initializers.is_empty()
            || branch.nodes.len() != 1
            || branch.outputs.len() != 1
            || !descriptor(&branch.outputs, output, "FLOAT", &latent)
        {
            return Err(policy_error(
                "PALS CUDA Warm branch has extra IO, weights, or computation",
            ));
        }
    }
    let seed = &warm.nodes[0];
    if seed.index != 0
        || seed.name != "consume_complete_private_seed"
        || seed.op != "Identity"
        || !seed.domain.is_empty()
        || seed.inputs != ["initial_latent"]
        || seed.outputs != [warm_output]
        || seed
            .attribute_names
            .as_ref()
            .is_none_or(|names| !names.is_empty())
        || !seed.constant_integral_seeds.is_empty()
        || seed.static_annotation != "none"
        || !warm.subgraphs.is_empty()
    {
        return Err(policy_error(
            "PALS CUDA Warm seed branch must be one complete latent Identity",
        ));
    }
    let initial = &fresh.nodes[0];
    if initial.index != 0
        || !exported_serial_name(&initial.name, "route_private_initial_")
        || initial.op != "If"
        || !initial.domain.is_empty()
        || initial.inputs != ["role_is_critic"]
        || initial.outputs != [fresh_output]
        || initial.static_annotation != "role_if_dispatch_candidate"
        || !initial.constant_integral_seeds.is_empty()
    {
        return Err(policy_error(
            "PALS CUDA Warm Fresh branch is not the original initial role If",
        ));
    }
    let (critic_initial, proposer_initial) = direct_if_branches(fresh, initial)?;
    for branch in [critic_initial, proposer_initial] {
        if !branch.inputs.is_empty()
            || branch.initializer_count != Some(0)
            || !branch.integral_initializers.is_empty()
            || branch.outputs.len() != 1
            || branch.outputs[0].dtype != "FLOAT"
            || branch.outputs[0].shape != latent
            || !branch.subgraphs.is_empty()
        {
            return Err(policy_error(
                "PALS original initial role branch has unexpected IO or weights",
            ));
        }
    }
    let mut role_consumers = Vec::new();
    consumers(scope, "role_is_critic", &mut role_consumers);
    let mut condition_nodes = Vec::new();
    let mut destination = None;
    let mut ffn_count = 0;
    let mut heads_count = 0;
    for (owner, node) in role_consumers {
        if node.inputs != ["role_is_critic"] {
            return Err(policy_error(
                "PALS CUDA Warm role has another data consumer",
            ));
        }
        if owner.path == scope.path && node.op == "Identity" {
            if node.name != "output_is_critic"
                || node.outputs != ["is_critic"]
                || node.static_annotation != "role_scalar_identity_candidate"
                || destination.replace(node.name.clone()).is_some()
            {
                return Err(policy_error(
                    "PALS CUDA Warm role destination is not unique",
                ));
            }
        } else if node.op == "If"
            && node.static_annotation == "role_if_dispatch_candidate"
            && ((owner.path == fresh.path && node.name == initial.name)
                || (owner.path == scope.path
                    && (exported_serial_name(&node.name, "route_private_ffn_")
                        || exported_serial_name(&node.name, "route_private_heads_"))))
        {
            let (then, otherwise) = direct_if_branches(owner, node)?;
            if [then, otherwise]
                .iter()
                .any(|branch| !branch.inputs.is_empty() || branch.initializer_count != Some(0))
            {
                return Err(policy_error(
                    "PALS CUDA Warm private role branch has copied inputs or weights",
                ));
            }
            if owner.path == scope.path {
                if exported_serial_name(&node.name, "route_private_ffn_") {
                    ffn_count += 1;
                }
                if exported_serial_name(&node.name, "route_private_heads_") {
                    heads_count += 1;
                }
            }
            condition_nodes.push(node.name.clone());
        } else {
            return Err(policy_error(
                "PALS CUDA Warm has an unregistered nested role consumer",
            ));
        }
    }
    let mut mode_consumers = Vec::new();
    consumers(scope, "warm_start", &mut mode_consumers);
    let mut seed_consumers = Vec::new();
    consumers(scope, "initial_latent", &mut seed_consumers);
    if condition_nodes.len() != 6
        || ffn_count != 4
        || heads_count != 1
        || mode_consumers.len() != 1
        || mode_consumers[0].0.path != scope.path
        || mode_consumers[0].1.name != mode.name
        || seed_consumers.len() != 1
        || seed_consumers[0].0.path != warm.path
        || seed_consumers[0].1.name != seed.name
    {
        return Err(policy_error(
            "PALS CUDA Warm control consumers differ from the registered route",
        ));
    }
    let destination =
        destination.ok_or_else(|| policy_error("PALS CUDA Warm role output is absent"))?;
    let mut required_cuda_dispatch: BTreeSet<_> = condition_nodes.iter().cloned().collect();
    required_cuda_dispatch.insert(mode.name.clone());
    required_cuda_dispatch.insert(seed.name.clone());
    required_cuda_dispatch.insert(destination.clone());
    Ok(CudaWarmControlRoute {
        root_scope: scope.path.clone(),
        mode_node: mode.name.clone(),
        role_transfer: RoleTransferPolicy {
            scope: scope.path.clone(),
            destination_node: destination,
            host_condition_nodes: condition_nodes,
        },
        required_cuda_dispatch,
    })
}

fn role_transfer_policy(role: &str, scope: &InventoryScope) -> Option<RoleTransferPolicy> {
    // This is one pinned root-scope scalar route, not a Memcpy/opcode exception.
    // With graph optimizations disabled, the only non-host role consumer is
    // the original Identity. The six CUDA If kernels require CPU input 0 in
    // the registered ORT 1.22 implementation; their private NN stays CUDA.
    if role != "shared_pc"
        || scope.path != "shared_pc"
        || scope
            .inputs
            .iter()
            .filter(|input| input.name == "role_is_critic")
            .count()
            != 1
        || !scope.inputs.iter().any(|input| {
            input.name == "role_is_critic"
                && input.dtype == "BOOL"
                && input.shape == serde_json::json!([])
        })
        || !scope.outputs.iter().any(|output| {
            output.name == "is_critic"
                && output.dtype == "BOOL"
                && output.shape == serde_json::json!([])
        })
    {
        return None;
    }
    let mut conditions = Vec::new();
    let mut destination = None;
    for node in &scope.nodes {
        if !node.inputs.iter().any(|input| input == "role_is_critic") {
            continue;
        }
        if node.inputs != ["role_is_critic"] {
            return None;
        }
        match node.op.as_str() {
            "If" if node.static_annotation == "role_if_dispatch_candidate" => {
                conditions.push(node.name.clone());
            }
            "Identity"
                if node.name == "output_is_critic"
                    && node.outputs == ["is_critic"]
                    && node.static_annotation == "role_scalar_identity_candidate"
                    && destination.is_none() =>
            {
                destination = Some(node.name.clone());
            }
            _ => return None,
        }
    }
    fn nested_role_consumer(scope: &InventoryScope) -> bool {
        scope
            .nodes
            .iter()
            .any(|node| node.inputs.iter().any(|input| input == "role_is_critic"))
            || scope.subgraphs.iter().any(nested_role_consumer)
    }
    if conditions.len() != 6 || scope.subgraphs.iter().any(nested_role_consumer) {
        return None;
    }
    Some(RoleTransferPolicy {
        scope: scope.path.clone(),
        destination_node: destination?,
        host_condition_nodes: conditions,
    })
}

#[derive(Default)]
pub(super) struct PlacementLog {
    lines: Vec<String>,
    transfer_lines: Vec<String>,
    bytes: usize,
    overflow: bool,
}
impl PlacementLog {
    pub(super) fn capture(&mut self, message: &str) {
        if self.reserve_message(message) {
            self.lines.push(message.to_owned());
        }
    }
    pub(super) fn capture_transfer(&mut self, message: &str) {
        if self.reserve_message(message) {
            self.transfer_lines.push(message.to_owned());
        }
    }
    /// Observation bytes only. This does not admit any graph or native Session.
    #[cfg(feature = "experimental-io-binding")]
    pub(super) fn bounded_lines(&self) -> Result<&[String], BackendError> {
        if self.overflow {
            return Err(policy_error("PALS placement collector overflowed"));
        }
        Ok(&self.lines)
    }
    #[cfg(feature = "experimental-io-binding")]
    pub(super) fn has_transfer_lines(&self) -> bool {
        !self.transfer_lines.is_empty()
    }
    /// Bounded diagnostic prefix; callers must retain the overflow flag.
    /// These bytes alone never authorize initialized placement.
    #[cfg(feature = "experimental-io-binding")]
    pub(super) fn diagnostic_lines(&self) -> &[String] {
        &self.lines
    }
    #[cfg(feature = "experimental-io-binding")]
    pub(super) fn overflowed(&self) -> bool {
        self.overflow
    }
    fn reserve_message(&mut self, message: &str) -> bool {
        if self.overflow {
            return false;
        }
        let Some(bytes) = self.bytes.checked_add(message.len()) else {
            self.overflow = true;
            return false;
        };
        if bytes > MAX_PLACEMENT_BYTES
            || self.lines.len() + self.transfer_lines.len() >= MAX_NODES + 32
        {
            self.overflow = true;
            return false;
        }
        self.bytes = bytes;
        true
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PalsGraphOptimization {
    Level1,
    Disable,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PalsControlTransferKind {
    BoolScalarHostToDevice,
}
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct PalsControlTransfer {
    pub kind: PalsControlTransferKind,
    pub graph_sha256: [u8; 32],
    pub source_scope: String,
    pub runtime_graph_name: String,
    pub node_name: String,
    pub opcode: String,
    pub provider: String,
    pub source_value: String,
    pub source_dtype: String,
    pub source_shape: Vec<usize>,
    pub destination_node: String,
    pub destination_opcode: String,
    pub destination_input_index: usize,
    pub destination_output: String,
    pub host_condition_nodes: Vec<String>,
    pub provenance: String,
}
const TRANSFER_PROVENANCE: &str =
    "ort-1.22-disable-source-io-addcopy-origin-single-recursive-cuda-transfer";
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct PalsGraphPlacement {
    pub role: String,
    pub graph_sha256: [u8; 32],
    pub log_sha256: [u8; 32],
    pub assigned_nodes: usize,
    pub cuda_nodes: usize,
    pub approved_cpu_control_nodes: usize,
    pub optimization: PalsGraphOptimization,
    pub approved_transfers: Vec<PalsControlTransfer>,
    pub recursive_coverage: String,
}
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct PalsKernelWitness {
    pub profile_sha256: [u8; 32],
    pub cuda_kernels: usize,
    /// Actual transfer kernel events, not NN inputs or CPU model computation.
    pub cuda_transfer_kernels: usize,
    pub approved_cpu_control_kernels: usize,
    pub neural_kernels: usize,
    pub proposer_private_kernels: usize,
    pub critic_private_kernels: usize,
}
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct PalsCudaPlacementWitness {
    pub schema: String,
    pub inventory_sha256: [u8; 32],
    pub manifest_sha256: [u8; 32],
    pub runtime_sha256: [u8; 32],
    pub runtime_bundle_sha256: [u8; 32],
    pub optimization: PalsGraphOptimization,
    pub initialization: Vec<PalsGraphPlacement>,
    pub public: PalsKernelWitness,
    pub shared_pc: PalsKernelWitness,
    pub category_provenance: String,
}
pub(super) fn placement_header(line: &str, prefix: &str) -> Option<(String, usize)> {
    let rest = line.strip_prefix(prefix)?;
    let (provider, count) = rest.split_once("]. Number of nodes: ")?;
    let count = count.parse::<usize>().ok()?;
    Some((provider.into(), count))
}
#[derive(Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum CopyMetadata {
    Origin {
        opcode: String,
        source_value: String,
        provider: String,
    },
    Summary {
        graph_name: String,
        count: usize,
        provider: String,
    },
}
fn copy_metadata(line: &str) -> Option<CopyMetadata> {
    if let Some(rest) = line.strip_prefix("Add ") {
        let (opcode, rest) = rest.split_once(" after ")?;
        let (source_value, provider) = rest.split_once(" for ")?;
        if opcode != "MemcpyFromHost"
            || source_value.is_empty()
            || source_value.len() > 256
            || provider != CUDA
        {
            return None;
        }
        return Some(CopyMetadata::Origin {
            opcode: opcode.into(),
            source_value: source_value.into(),
            provider: provider.into(),
        });
    }
    let (count, rest) = line.split_once(" Memcpy nodes are added to the graph ")?;
    let (graph_name, rest) = rest.split_once(" for ")?;
    let provider = rest.strip_suffix(". It might have negative impact on performance (including unable to run CUDA graph). Set session_options.log_severity_level=1 to see the detail logs before this message.")?;
    let count = count.parse::<usize>().ok()?;
    if graph_name.is_empty()
        || graph_name.len() > 256
        || count == 0
        || count > MAX_NODES
        || provider != CUDA
    {
        return None;
    }
    Some(CopyMetadata::Summary {
        graph_name: graph_name.into(),
        count,
        provider: provider.into(),
    })
}
fn valid_transfers(policy: &GraphPolicy, transfers: &[PalsControlTransfer]) -> bool {
    if transfers.is_empty() {
        return true;
    }
    let Some(source) = &policy.role_transfer else {
        return false;
    };
    if transfers.len() != 1 {
        return false;
    }
    let transfer = &transfers[0];
    transfer.kind == PalsControlTransferKind::BoolScalarHostToDevice
        && transfer.graph_sha256 == policy.digest
        && transfer.source_scope == source.scope
        && !transfer.runtime_graph_name.is_empty()
        && transfer.runtime_graph_name.len() <= 256
        && transfer.node_name == "Memcpy"
        && !policy.nodes.contains_key(&transfer.node_name)
        && transfer.opcode == "MemcpyFromHost"
        && transfer.provider == CUDA
        && transfer.source_value == "role_is_critic"
        && transfer.source_dtype == "BOOL"
        && transfer.source_shape.is_empty()
        && transfer.destination_node == source.destination_node
        && transfer.destination_opcode == "Identity"
        && transfer.destination_input_index == 0
        && transfer.destination_output == "is_critic"
        && transfer.host_condition_nodes == source.host_condition_nodes
        && transfer.provenance == TRANSFER_PROVENANCE
}
fn verify_control_transfer(
    policy: &GraphPolicy,
    log: &PlacementLog,
    nodes: &BTreeMap<String, (String, String)>,
    copies: &[String],
) -> Result<Vec<PalsControlTransfer>, BackendError> {
    if log.transfer_lines.is_empty() && copies.is_empty() {
        return Ok(Vec::new());
    }
    // The ORT origin INFO omits generated-node IO and scope. Admission therefore
    // requires Disable, the pinned unique root IO route, one recursive copy,
    // its one official graph summary, and every original consumer's exact CUDA
    // placement. Multiple copies/scopes or any unsupported source stay denied.
    let Some(source) = &policy.role_transfer else {
        return Err(policy_error(
            "PALS generated transfer lacks a pinned scalar source route",
        ));
    };
    if copies != ["Memcpy"] || log.transfer_lines.len() != 2 {
        return Err(policy_error(
            "PALS generated transfer has ambiguous copy count or origin",
        ));
    }
    let mut origin = None;
    let mut graph_name = None;
    for line in &log.transfer_lines {
        match copy_metadata(line.trim()) {
            Some(CopyMetadata::Origin {
                opcode,
                source_value,
                provider,
            }) if origin.is_none()
                && opcode == "MemcpyFromHost"
                && source_value == "role_is_critic"
                && provider == CUDA =>
            {
                origin = Some(());
            }
            Some(CopyMetadata::Summary {
                graph_name: name,
                count: 1,
                provider,
            }) if graph_name.is_none() && provider == CUDA => {
                graph_name = Some(name);
            }
            _ => {
                return Err(policy_error(
                    "PALS generated transfer origin/count metadata is unapproved",
                ));
            }
        }
    }
    if origin.is_none()
        || !nodes
            .get(&source.destination_node)
            .is_some_and(|(op, provider)| op == "Identity" && provider == CUDA)
        || source.host_condition_nodes.iter().any(|name| {
            !nodes
                .get(name)
                .is_some_and(|(op, provider)| op == "If" && provider == CUDA)
        })
    {
        return Err(policy_error(
            "PALS scalar transfer lacks exact CUDA destination/host-condition consumers",
        ));
    }
    let transfer = PalsControlTransfer {
        kind: PalsControlTransferKind::BoolScalarHostToDevice,
        graph_sha256: policy.digest,
        source_scope: source.scope.clone(),
        runtime_graph_name: graph_name
            .ok_or_else(|| policy_error("PALS transfer graph summary is missing"))?,
        node_name: copies[0].clone(),
        opcode: "MemcpyFromHost".into(),
        provider: CUDA.into(),
        source_value: "role_is_critic".into(),
        source_dtype: "BOOL".into(),
        source_shape: Vec::new(),
        destination_node: source.destination_node.clone(),
        destination_opcode: "Identity".into(),
        destination_input_index: 0,
        destination_output: "is_critic".into(),
        host_condition_nodes: source.host_condition_nodes.clone(),
        provenance: TRANSFER_PROVENANCE.into(),
    };
    if !valid_transfers(policy, std::slice::from_ref(&transfer)) {
        return Err(policy_error(
            "PALS transfer attestation differs from pinned source route",
        ));
    }
    Ok(vec![transfer])
}
fn verify_placement(
    role: &str,
    policy: &GraphPolicy,
    log: &PlacementLog,
) -> Result<PalsGraphPlacement, BackendError> {
    if log.overflow {
        return Err(policy_error("PALS placement collector overflowed"));
    }
    let mut began = false;
    let mut headers: BTreeMap<String, usize> = BTreeMap::new();
    let mut observed: BTreeMap<String, usize> = BTreeMap::new();
    let mut selected = None;
    let mut names = BTreeSet::new();
    let mut nodes = BTreeMap::new();
    let mut copies = Vec::new();
    for line in &log.lines {
        let line = line.trim();
        if line == "Node placements" {
            if began {
                return Err(policy_error(
                    "PALS placement has ambiguous multiple initialization sections",
                ));
            }
            began = true;
            continue;
        }
        if !began {
            continue;
        }
        if let Some((provider, count)) = placement_header(line, "All nodes placed on [") {
            let _ = (provider, count);
            // ORT's single-provider summary omits node identities. This exact
            // policy cannot prove that no unknown/fused node was admitted.
            return Err(policy_error(
                "PALS placement summary lacks exact compiled-node identities",
            ));
        }
        if let Some((provider, count)) = placement_header(line, "Node(s) placed on [") {
            if !matches!(provider.as_str(), CUDA | CPU)
                || count == 0
                || count > MAX_NODES
                || headers.insert(provider.clone(), count).is_some()
            {
                return Err(policy_error(
                    "PALS placement provider header is invalid or duplicated",
                ));
            }
            selected = Some(provider);
            continue;
        }
        if let Some(provider) = &selected {
            let (op, name) = line
                .split_once(" (")
                .and_then(|(op, rest)| rest.strip_suffix(')').map(|name| (op, name)))
                .ok_or_else(|| policy_error("PALS placement node record is incomplete"))?;
            let original = policy.nodes.get(name).map(String::as_str) == Some(op);
            let copy_candidate =
                !original && name == "Memcpy" && op == "MemcpyFromHost" && provider == CUDA;
            if !names.insert(name.to_owned())
                || name.len() > 256
                || op.is_empty()
                || (!original && !copy_candidate)
                || (provider == CPU && policy.controls.get(name).map(String::as_str) != Some(op))
            {
                return Err(policy_error(
                    "PALS unknown/fused or unapproved exact node placement",
                ));
            }
            if copy_candidate {
                copies.push(name.to_owned());
            }
            nodes.insert(name.to_owned(), (op.to_owned(), provider.clone()));
            *observed.entry(provider.clone()).or_default() += 1;
        } else {
            return Err(policy_error(
                "PALS placement record has no provider section",
            ));
        }
    }
    if headers != observed || headers.get(CUDA).copied().unwrap_or(0) == 0 || headers.len() != 2 {
        return Err(policy_error(
            "PALS placement lacks complete recursive CUDA/CPU node coverage",
        ));
    }
    let (cuda_nodes, cpu_nodes) = (headers[CUDA], headers[CPU]);
    if policy.required_cuda_dispatch.iter().any(|name| {
        !nodes
            .get(name)
            .is_some_and(|(op, provider)| policy.nodes.get(name) == Some(op) && provider == CUDA)
    }) {
        return Err(policy_error(
            "PALS CUDA Warm lacks exact pre-Run CUDA mode, seed, or role consumer placement",
        ));
    }
    let approved_transfers = verify_control_transfer(policy, log, &nodes, &copies)?;
    if !began || cuda_nodes + cpu_nodes > policy.nodes.len() + approved_transfers.len() {
        return Err(policy_error("PALS placement evidence missing"));
    }
    Ok(PalsGraphPlacement {
        role: role.into(),
        graph_sha256: policy.digest,
        log_sha256: asset::sha256(
            format!(
                "placement\n{}\ncopy\n{}",
                log.lines.join("\n"),
                log.transfer_lines.join("\n")
            )
            .as_bytes(),
        ),
        assigned_nodes: cuda_nodes + cpu_nodes,
        cuda_nodes,
        approved_cpu_control_nodes: cpu_nodes,
        optimization: PalsGraphOptimization::Disable,
        approved_transfers,
        recursive_coverage:
            "ort-1.22-finalize-recursive-exact-named-provider-coverage-before-first-run".into(),
    })
}
fn bounded_metadata_output(value: &serde_json::Value) -> bool {
    let Some(outputs) = value.as_array() else {
        return false;
    };
    !outputs.is_empty()
        && outputs.iter().all(|output| {
            let Some(types) = output.as_object() else {
                return false;
            };
            types.len() == 1
                && types.iter().all(|(kind, dims)| {
                    if !matches!(kind.as_str(), "int64" | "int32" | "bool") {
                        return false;
                    }
                    let Some(dims) = dims.as_array() else {
                        return false;
                    };
                    dims.len() <= 4
                        && dims
                            .iter()
                            .try_fold(1_u64, |n, dim| n.checked_mul(dim.as_u64()?))
                            .is_some_and(|elements| elements <= 64)
                })
        })
}
fn verify_profile(
    bytes: &[u8],
    graph: &GraphPolicy,
    transfers: &[PalsControlTransfer],
) -> Result<PalsKernelWitness, BackendError> {
    if bytes.len() > MAX_PROFILE_BYTES {
        return Err(policy_error("PALS kernel profile exceeds bounded output"));
    }
    let events: Vec<serde_json::Value> = serde_json::from_slice(bytes).map_err(|error| {
        policy_error("cannot decode actual PALS kernel profile")
            .with_external_cause(CauseCode::ProfileParse, &error)
    })?;
    let mut witness = PalsKernelWitness {
        profile_sha256: asset::sha256(bytes),
        cuda_kernels: 0,
        cuda_transfer_kernels: 0,
        approved_cpu_control_kernels: 0,
        neural_kernels: 0,
        proposer_private_kernels: 0,
        critic_private_kernels: 0,
    };
    for event in events {
        if event.get("cat").and_then(|v| v.as_str()) != Some("Node") {
            continue;
        }
        let name = event
            .get("name")
            .and_then(|v| v.as_str())
            .ok_or_else(|| policy_error("PALS node profile lacks name"))?;
        let provider = event.pointer("/args/provider").and_then(|v| v.as_str());
        if name.ends_with("_fence_before") || name.ends_with("_fence_after") {
            if provider.is_some_and(|provider| provider != CUDA && provider != CPU) {
                return Err(policy_error("PALS fence record has an unknown provider"));
            }
            continue;
        }
        let node = name
            .strip_suffix("_kernel_time")
            .ok_or_else(|| policy_error("unknown PALS Node profile record"))?;
        let op = event
            .pointer("/args/op_name")
            .and_then(|v| v.as_str())
            .ok_or_else(|| policy_error("PALS actual kernel record lacks op type"))?;
        match provider {
            Some(CUDA) => {
                if graph.nodes.get(node).map(String::as_str) != Some(op) {
                    if !valid_transfers(graph, transfers)
                        || !transfers
                            .iter()
                            .any(|transfer| transfer.node_name == node && transfer.opcode == op)
                        || event.pointer("/args/output_type_shape")
                            != Some(&serde_json::json!([{"bool":[]}]))
                    {
                        return Err(policy_error(
                            "PALS actual CUDA kernel has unknown/fused identity or transfer output",
                        ));
                    }
                    witness.cuda_transfer_kernels += 1;
                }
                witness.cuda_kernels += 1;
                if graph.major.get(node).map(String::as_str) == Some(op) {
                    witness.neural_kernels += 1;
                }
                if graph.proposer_private.get(node).map(String::as_str) == Some(op) {
                    witness.proposer_private_kernels += 1;
                }
                if graph.critic_private.get(node).map(String::as_str) == Some(op) {
                    witness.critic_private_kernels += 1;
                }
            }
            Some(CPU)
                if graph.controls.get(node).map(String::as_str) == Some(op)
                    && event
                        .pointer("/args/output_type_shape")
                        .is_some_and(bounded_metadata_output) =>
            {
                witness.approved_cpu_control_kernels += 1;
            }
            _ => {
                return Err(policy_error(
                    "PALS kernel is unapproved CPU inference or unknown placement",
                ));
            }
        }
    }
    if witness.cuda_kernels == 0 {
        return Err(policy_error("PALS profile contains no actual CUDA kernels"));
    }
    Ok(witness)
}

pub(super) fn owned_profile_root(path: &Path) -> Result<PathBuf, BackendError> {
    if !path.is_absolute() {
        return Err(policy_error(
            "PALS explicit audit output root must be absolute",
        ));
    }
    let parent = path
        .parent()
        .ok_or_else(|| policy_error("PALS audit output parent is absent"))?;
    let parent = parent.canonicalize().map_err(|error| {
        policy_error("PALS audit output parent is unavailable")
            .with_external_cause(CauseCode::ProfilingStart, &error)
    })?;
    for ancestor in parent.ancestors() {
        if ancestor.join(".git").exists() {
            return Err(policy_error("PALS native profiles must be outside Git"));
        }
    }
    let name = path
        .file_name()
        .ok_or_else(|| policy_error("PALS audit output root has no name"))?;
    let output = parent.join(name);
    std::fs::create_dir(&output).map_err(|error| {
        policy_error("PALS audit output root must be new and exclusive")
            .with_external_cause(CauseCode::ProfilingStart, &error)
    })?;
    Ok(output)
}

pub(super) fn record_initial_log(
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
    if !matches!(role, "public" | "shared_pc") || !path.is_absolute() {
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
        schema: "rovezero.pals-initial-placement.v2",
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

#[derive(Serialize)]
struct PlacementHeader {
    sequence: usize,
    provider: String,
    declared_nodes: usize,
    all_nodes_summary: bool,
}
#[derive(Serialize)]
struct PlacementNode<'a> {
    sequence: usize,
    provider: Option<String>,
    name: &'a str,
    op: &'a str,
}
#[derive(Serialize)]
struct PlacementUnparsed {
    sequence: usize,
    line_sha256: [u8; 32],
}
#[derive(Serialize)]
struct PlacementFile<'a> {
    schema: &'static str,
    source: &'static str,
    optimization: PalsGraphOptimization,
    role: &'a str,
    session_created: bool,
    before_first_neural_run: bool,
    gate: &'static str,
    gate_detail: Option<&'static str>,
    collected_bytes: usize,
    collector_overflow: bool,
    provider_headers: Vec<PlacementHeader>,
    nodes: Vec<PlacementNode<'a>>,
    unparsed: Vec<PlacementUnparsed>,
    transfer_metadata: Vec<CopyMetadata>,
    transfer_unparsed: Vec<PlacementUnparsed>,
    approved_transfers: Option<&'a [PalsControlTransfer]>,
}
struct BoundedPlacementJson {
    bytes: Vec<u8>,
}
impl std::io::Write for BoundedPlacementJson {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        let required = self
            .bytes
            .len()
            .checked_add(bytes.len())
            .filter(|required| *required <= MAX_PROFILE_BYTES)
            .ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "PALS placement JSON exceeds 4MiB",
                )
            })?;
        if required > self.bytes.capacity() {
            let target = self
                .bytes
                .capacity()
                .saturating_mul(2)
                .max(required)
                .clamp(1024, MAX_PROFILE_BYTES);
            self.bytes
                .try_reserve_exact(target - self.bytes.len())
                .map_err(|_| std::io::Error::other("PALS placement JSON allocation failed"))?;
        }
        self.bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

pub(super) fn finish_profile(session: &mut Session, root: &Path) -> Result<Vec<u8>, BackendError> {
    let name = session.end_profiling().map_err(|error| {
        native(
            CauseCode::ProfilingFinish,
            "cannot complete actual PALS kernel profile",
            error,
        )
    })?;
    let path = Path::new(&name);
    let metadata = std::fs::symlink_metadata(path).map_err(|error| {
        policy_error("actual PALS kernel profile is unavailable")
            .with_external_cause(CauseCode::ProfilingFinish, &error)
    })?;
    let actual = path.canonicalize().map_err(|error| {
        policy_error("cannot resolve PALS kernel profile")
            .with_external_cause(CauseCode::ProfilingFinish, &error)
    })?;
    if !path.is_absolute()
        || !metadata.is_file()
        || metadata.file_type().is_symlink()
        || actual.parent() != Some(root)
        || !actual
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| {
                (name.starts_with("pals-public-kernels_")
                    || name.starts_with("pals-shared_pc-kernels_"))
                    && name.ends_with(".json")
            })
    {
        return Err(policy_error(
            "PALS native profile escapes its exclusive owner root",
        ));
    }
    asset::read_bounded(&actual, MAX_PROFILE_BYTES)
}

#[cfg(test)]
mod tests {
    use super::*;
    struct IoRoot(PathBuf);
    impl Drop for IoRoot {
        fn drop(&mut self) {
            // Only the two exact files owned by this test; no recursive cleanup.
            for name in [
                "public-initial-placement.json",
                "shared_pc-initial-placement.json",
            ] {
                let _ = std::fs::remove_file(self.0.join(name));
            }
            let _ = std::fs::remove_dir(&self.0);
        }
    }
    fn io_root() -> IoRoot {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "rz-pals-placement-io-{}-{nonce}",
            std::process::id()
        ));
        IoRoot(owned_profile_root(&path).unwrap())
    }
    fn graph() -> GraphPolicy {
        GraphPolicy {
            digest: [1; 32],
            nodes: [
                ("shape_node".into(), "Shape".into()),
                ("nn".into(), "MatMul".into()),
                ("p_nn".into(), "Gemm".into()),
                ("c_nn".into(), "Gemm".into()),
            ]
            .into(),
            controls: [("shape_node".into(), "Shape".into())].into(),
            major: [
                ("nn".into(), "MatMul".into()),
                ("p_nn".into(), "Gemm".into()),
                ("c_nn".into(), "Gemm".into()),
            ]
            .into(),
            proposer_private: [("p_nn".into(), "Gemm".into())].into(),
            critic_private: [("c_nn".into(), "Gemm".into())].into(),
            role_transfer: None,
            required_cuda_dispatch: BTreeSet::new(),
        }
    }
    fn log(lines: &[&str]) -> PlacementLog {
        let mut result = PlacementLog::default();
        for line in lines {
            result.capture(line);
        }
        result
    }
    const COPY_SUMMARY: &str = "1 Memcpy nodes are added to the graph shared_role_graph for CUDAExecutionProvider. It might have negative impact on performance (including unable to run CUDA graph). Set session_options.log_severity_level=1 to see the detail logs before this message.";
    fn transfer_scope() -> InventoryScope {
        let mut nodes: Vec<_> = (0..6)
            .map(|i| {
                serde_json::json!({
                    "index":i,"name":format!("if_{i}"),"domain":"","op":"If",
                    "inputs":["role_is_critic"],"outputs":[format!("latent_{i}")],
                    "input_origins":["role_control"],"constant_integral_seeds":[],
                    "static_annotation":"role_if_dispatch_candidate","actual_provider":"unknown"
                })
            })
            .collect();
        nodes.push(serde_json::json!({"index":6,"name":"output_is_critic","domain":"","op":"Identity",
            "inputs":["role_is_critic"],"outputs":["is_critic"],"input_origins":["role_control"],
            "constant_integral_seeds":[],"static_annotation":"role_scalar_identity_candidate","actual_provider":"unknown"}));
        serde_json::from_value(serde_json::json!({"path":"shared_pc",
            "inputs":[{"name":"role_is_critic","dtype":"BOOL","shape":[]}],
            "outputs":[{"name":"is_critic","dtype":"BOOL","shape":[]}],
            "integral_initializers":[],"nodes":nodes,"subgraphs":[]}))
        .unwrap()
    }
    fn transfer_graph_log() -> (GraphPolicy, PlacementLog) {
        let scope = transfer_scope();
        let mut policy = graph();
        policy.nodes.remove("p_nn");
        policy.nodes.remove("c_nn");
        for node in &scope.nodes {
            policy.nodes.insert(node.name.clone(), node.op.clone());
        }
        policy
            .controls
            .insert("output_is_critic".into(), "Identity".into());
        policy.role_transfer = role_transfer_policy("shared_pc", &scope);
        let mut captured = log(&[
            "Node placements",
            "Node(s) placed on [CPUExecutionProvider]. Number of nodes: 1",
            "Shape (shape_node)",
            "Node(s) placed on [CUDAExecutionProvider]. Number of nodes: 9",
            "MatMul (nn)",
            "If (if_0)",
            "If (if_1)",
            "If (if_2)",
            "If (if_3)",
            "If (if_4)",
            "If (if_5)",
            "Identity (output_is_critic)",
            "MemcpyFromHost (Memcpy)",
        ]);
        captured
            .capture_transfer("Add MemcpyFromHost after role_is_critic for CUDAExecutionProvider");
        captured.capture_transfer(COPY_SUMMARY);
        (policy, captured)
    }
    fn warm_node(
        index: usize,
        name: &str,
        op: &str,
        input: &str,
        outputs: Vec<String>,
    ) -> serde_json::Value {
        let (annotation, origin) = match op {
            "If" if input == "warm_start" => (CUDA_WARM_IF_ANNOTATION, "cuda_warm_mode_control"),
            "If" => ("role_if_dispatch_candidate", "role_control"),
            "Identity" if input == "role_is_critic" => {
                ("role_scalar_identity_candidate", "role_control")
            }
            "Gemm" => ("major_nn_requires_cuda", "model_data_or_unknown"),
            "Shape" => ("shape_metadata_candidate", "model_data_or_unknown"),
            _ => ("none", "model_data_or_unknown"),
        };
        serde_json::json!({"index":index,"name":name,"domain":"","op":op,
            "inputs":[input],"outputs":outputs,"input_origins":[origin],
            "constant_integral_seeds":[],"static_annotation":annotation,"actual_provider":"unknown",
            "attribute_names":if op == "If" { vec!["then_branch","else_branch"] } else { vec![] }})
    }
    fn warm_scope_value() -> serde_json::Value {
        let latent = serde_json::json!(["batch", 16, 384]);
        let branch = |path: String, tag: &str| {
            let output = format!("{tag}_latent");
            serde_json::json!({"path":path,"inputs":[],"outputs":[
                {"name":output,"dtype":"FLOAT","shape":latent}],
                "integral_initializers":[],"initializer_count":0,
                "nodes":[warm_node(0,&format!("{tag}_nn"),"Gemm","query",vec![output])],"subgraphs":[]})
        };
        let selected = "selected_private_initial_10";
        let mode_path = "shared_pc/node0:private_warm_mode";
        let fresh_path = format!("{mode_path}/else_branch");
        let initial_name = "route_private_initial_9";
        let initial_path = format!("{fresh_path}/node0:{initial_name}");
        let warm = serde_json::json!({"path":format!("{mode_path}/then_branch"),"inputs":[],
            "outputs":[{"name":format!("{selected}__warm_branch"),"dtype":"FLOAT","shape":latent}],
            "integral_initializers":[],"initializer_count":0,
            "nodes":[warm_node(0,"consume_complete_private_seed","Identity","initial_latent",vec![format!("{selected}__warm_branch")])],
            "subgraphs":[]});
        let fresh = serde_json::json!({"path":fresh_path,"inputs":[],
            "outputs":[{"name":format!("{selected}__fresh_branch"),"dtype":"FLOAT","shape":latent}],
            "integral_initializers":[],"initializer_count":0,
            "nodes":[warm_node(0,initial_name,"If","role_is_critic",vec![format!("{selected}__fresh_branch")])],
            "subgraphs":[branch(format!("{initial_path}/then_branch"),"critic_initial"),
                branch(format!("{initial_path}/else_branch"),"proposer_initial")]});
        let mut nodes = vec![warm_node(
            0,
            "private_warm_mode",
            "If",
            "warm_start",
            vec![selected.into()],
        )];
        let mut subgraphs = vec![warm, fresh];
        for index in 1..=5 {
            let name = format!(
                "route_private_{}_{}",
                if index == 5 { "heads" } else { "ffn" },
                index + 10
            );
            nodes.push(warm_node(
                index,
                &name,
                "If",
                "role_is_critic",
                vec![format!("selected_{index}")],
            ));
            for (suffix, role) in [("then_branch", "critic"), ("else_branch", "proposer")] {
                subgraphs.push(branch(
                    format!("shared_pc/node{index}:{name}/{suffix}"),
                    &format!("{role}_route_{index}"),
                ));
            }
        }
        nodes.push(warm_node(
            6,
            "output_is_critic",
            "Identity",
            "role_is_critic",
            vec!["is_critic".into()],
        ));
        nodes.push(warm_node(
            7,
            "shape_node",
            "Shape",
            "query",
            vec!["query_shape".into()],
        ));
        nodes.push(warm_node(
            8,
            "nn",
            "Gemm",
            "query",
            vec!["shared_features".into()],
        ));
        serde_json::json!({"path":"shared_pc","inputs":[
            {"name":"role_is_critic","dtype":"BOOL","shape":[]},
            {"name":"warm_start","dtype":"BOOL","shape":[]},
            {"name":"initial_latent","dtype":"FLOAT","shape":latent},
            {"name":"query","dtype":"FLOAT","shape":["batch",16]}],
            "outputs":[{"name":"is_critic","dtype":"BOOL","shape":[]}],
            "integral_initializers":[],"initializer_count":1,"nodes":nodes,"subgraphs":subgraphs})
    }
    fn inspected_warm(scope: &InventoryScope) -> Result<GraphPolicy, BackendError> {
        let route = cuda_warm_control_route("shared_pc", scope)?;
        let mut policy = GraphPolicy {
            digest: [1; 32],
            nodes: BTreeMap::new(),
            controls: BTreeMap::new(),
            major: BTreeMap::new(),
            proposer_private: BTreeMap::new(),
            critic_private: BTreeMap::new(),
            role_transfer: Some(route.role_transfer.clone()),
            required_cuda_dispatch: route.required_cuda_dispatch.clone(),
        };
        inspect_scope_with_layout(
            scope,
            &BTreeMap::new(),
            &mut policy,
            &mut BTreeSet::new(),
            &mut 0,
            Some(&route),
            None,
        )?;
        Ok(policy)
    }
    fn warm_graph_log() -> (GraphPolicy, PlacementLog) {
        let scope = serde_json::from_value(warm_scope_value()).unwrap();
        let policy = inspected_warm(&scope).unwrap();
        let mut captured = log(&[
            "Node placements",
            "Node(s) placed on [CPUExecutionProvider]. Number of nodes: 1",
            "Shape (shape_node)",
        ]);
        captured.capture(&format!(
            "Node(s) placed on [{CUDA}]. Number of nodes: {}",
            policy.nodes.len()
        ));
        for (name, op) in &policy.nodes {
            if name != "shape_node" {
                captured.capture(&format!("{op} ({name})"));
            }
        }
        captured.capture("MemcpyFromHost (Memcpy)");
        captured
            .capture_transfer("Add MemcpyFromHost after role_is_critic for CUDAExecutionProvider");
        captured.capture_transfer(COPY_SUMMARY);
        (policy, captured)
    }
    #[test]
    fn cuda_warm_route_is_separate_from_legacy_and_classifies_only_actual_role_branches() {
        let scope: InventoryScope = serde_json::from_value(warm_scope_value()).unwrap();
        assert!(role_transfer_policy("shared_pc", &scope).is_none());
        assert!(inspect_scope(
            &scope,
            &BTreeMap::new(),
            &mut graph(),
            &mut BTreeSet::new(),
            &mut 0
        )
        .is_err());
        let policy = inspected_warm(&scope).unwrap();
        assert_eq!(policy.required_cuda_dispatch.len(), 9);
        assert_eq!(
            policy
                .role_transfer
                .as_ref()
                .unwrap()
                .host_condition_nodes
                .len(),
            6
        );
        assert!(policy.critic_private.contains_key("critic_initial_nn"));
        assert!(!policy.proposer_private.contains_key("critic_initial_nn"));
        assert!(policy.proposer_private.contains_key("proposer_initial_nn"));
        assert!(!policy.critic_private.contains_key("proposer_initial_nn"));
        assert!(!policy.critic_private.contains_key("nn"));
        assert!(!policy.proposer_private.contains_key("nn"));
        assert!(!policy.controls.contains_key("private_warm_mode"));
        assert!(!policy
            .controls
            .contains_key("consume_complete_private_seed"));
        assert!(cuda_warm_control_route("public", &scope).is_err());
    }
    #[test]
    fn cuda_warm_rejects_arbitrary_bool_if_seed_math_extra_branch_weights_and_nested_role_consumers(
    ) {
        type AlterWarmScope = Box<dyn Fn(&mut serde_json::Value)>;
        let cases: Vec<(&str, AlterWarmScope)> = vec![
            (
                "vector_mode",
                Box::new(|v| v["inputs"][1]["shape"] = serde_json::json!([1])),
            ),
            (
                "partial_seed",
                Box::new(|v| v["inputs"][2]["shape"] = serde_json::json!(["batch", 16, 383])),
            ),
            (
                "arbitrary_bool",
                Box::new(|v| v["nodes"][0]["inputs"] = serde_json::json!(["another_bool"])),
            ),
            (
                "other_mode",
                Box::new(|v| v["nodes"][0]["name"] = serde_json::json!("unregistered_mode")),
            ),
            (
                "extra_attribute",
                Box::new(|v| {
                    v["nodes"][0]["attribute_names"] =
                        serde_json::json!(["then_branch", "else_branch", "extra"])
                }),
            ),
            (
                "seed_math",
                Box::new(|v| v["subgraphs"][0]["nodes"][0]["op"] = serde_json::json!("Add")),
            ),
            (
                "query_double_add",
                Box::new(|v| {
                    v["subgraphs"][0]["nodes"][0]["inputs"] =
                        serde_json::json!(["initial_latent", "query"])
                }),
            ),
            (
                "seed_weights",
                Box::new(|v| v["subgraphs"][0]["initializer_count"] = serde_json::json!(1)),
            ),
            (
                "missing_seed_weight_attestation",
                Box::new(|v| {
                    v["subgraphs"][0]
                        .as_object_mut()
                        .unwrap()
                        .remove("initializer_count");
                }),
            ),
            (
                "extra_seed_node",
                Box::new(|v| {
                    let extra = v["subgraphs"][0]["nodes"][0].clone();
                    v["subgraphs"][0]["nodes"]
                        .as_array_mut()
                        .unwrap()
                        .push(extra);
                }),
            ),
            (
                "nonoriginal_fresh",
                Box::new(|v| {
                    v["subgraphs"][1]["nodes"][0]["name"] = serde_json::json!("other_role_initial")
                }),
            ),
            (
                "extra_fresh_node",
                Box::new(|v| {
                    let extra = v["subgraphs"][1]["nodes"][0].clone();
                    v["subgraphs"][1]["nodes"]
                        .as_array_mut()
                        .unwrap()
                        .push(extra);
                }),
            ),
            (
                "nested_role_data",
                Box::new(|v| {
                    v["subgraphs"][2]["nodes"][0]["inputs"] = serde_json::json!(["role_is_critic"])
                }),
            ),
            (
                "another_role_head",
                Box::new(|v| v["nodes"][1]["name"] = serde_json::json!("route_private_heads_99")),
            ),
            (
                "misstated_mode_flow",
                Box::new(|v| {
                    v["nodes"][0]["input_origins"] = serde_json::json!(["small_integral_constant"])
                }),
            ),
        ];
        for (name, alter) in cases {
            let mut value = warm_scope_value();
            alter(&mut value);
            let scope: InventoryScope = serde_json::from_value(value).unwrap();
            assert!(inspected_warm(&scope).is_err(), "{name}");
        }
    }
    #[test]
    fn cuda_warm_requires_actual_cuda_dispatch_and_original_single_role_copy_evidence() {
        let (policy, mut captured) = warm_graph_log();
        let placement = verify_placement("shared_pc", &policy, &captured).unwrap();
        assert_eq!(placement.approved_transfers.len(), 1);
        assert!(placement.approved_transfers[0]
            .host_condition_nodes
            .contains(&"route_private_initial_9".into()));
        captured
            .lines
            .retain(|line| line != "If (route_private_initial_9)");
        assert!(verify_placement("shared_pc", &policy, &captured).is_err());
        let (_, mut bad) = warm_graph_log();
        bad.transfer_lines[0] =
            "Add MemcpyFromHost after warm_start for CUDAExecutionProvider".into();
        assert!(verify_placement("shared_pc", &policy, &bad).is_err());
        let (_, mut bad) = warm_graph_log();
        bad.transfer_lines[1] = COPY_SUMMARY.replacen("1 Memcpy", "2 Memcpy", 1);
        assert!(verify_placement("shared_pc", &policy, &bad).is_err());
        let (_, mut bad) = warm_graph_log();
        bad.lines[2] = "If (private_warm_mode)".into();
        assert!(verify_placement("shared_pc", &policy, &bad).is_err());
        let (_, mut bad) = warm_graph_log();
        bad.lines[2] = "Gemm (critic_initial_nn)".into();
        assert!(verify_placement("shared_pc", &policy, &bad).is_err());
    }
    #[test]
    fn cuda_warm_profile_fresh_critic_scope_cannot_also_attest_proposer_or_allow_cpu_nn() {
        let (policy, captured) = warm_graph_log();
        let transfers = verify_placement("shared_pc", &policy, &captured)
            .unwrap()
            .approved_transfers;
        let event = |node: &str, provider: &str| serde_json::json!({"cat":"Node","name":format!("{node}_kernel_time"),"args":{"provider":provider,"op_name":"Gemm"}});
        let critic_only =
            serde_json::to_vec(&vec![event("critic_initial_nn", CUDA), event("nn", CUDA)]).unwrap();
        let witness = verify_profile(&critic_only, &policy, &transfers).unwrap();
        assert_eq!(witness.critic_private_kernels, 1);
        assert_eq!(witness.proposer_private_kernels, 0);
        let both = serde_json::to_vec(&vec![
            event("critic_initial_nn", CUDA),
            event("proposer_initial_nn", CUDA),
        ])
        .unwrap();
        let witness = verify_profile(&both, &policy, &transfers).unwrap();
        assert_eq!(witness.critic_private_kernels, 1);
        assert_eq!(witness.proposer_private_kernels, 1);
        let cpu = serde_json::to_vec(&vec![event("critic_initial_nn", CPU)]).unwrap();
        assert!(verify_profile(&cpu, &policy, &transfers).is_err());
    }
    #[test]
    fn role_transfer_source_requires_scalar_bool_unique_root_destination_and_six_host_conditions() {
        let mut scope = transfer_scope();
        assert!(role_transfer_policy("shared_pc", &scope).is_some());
        assert!(role_transfer_policy("public", &scope).is_none());
        scope.inputs[0].shape = serde_json::json!([1]);
        assert!(role_transfer_policy("shared_pc", &scope).is_none());
        scope.inputs[0].shape = serde_json::json!([]);
        scope.outputs[0].dtype = "FLOAT".into();
        assert!(role_transfer_policy("shared_pc", &scope).is_none());
        scope.outputs[0].dtype = "BOOL".into();
        scope.nodes[6].outputs = vec!["unrelated_tag".into()];
        assert!(role_transfer_policy("shared_pc", &scope).is_none());
        scope.nodes[6].outputs = vec!["is_critic".into()];
        scope.nodes.push(
            serde_json::from_value(serde_json::json!({"index":7,"name":"extra",
            "domain":"","op":"Cast","inputs":["role_is_critic"],"outputs":["float_role"],
            "input_origins":["role_control"],"constant_integral_seeds":[],
            "static_annotation":"none","actual_provider":"unknown"}))
            .unwrap(),
        );
        assert!(role_transfer_policy("shared_pc", &scope).is_none());
    }
    #[test]
    fn scalar_transfer_pre_run_gate_requires_actual_origin_count_and_cuda_consumers() {
        let (policy, captured) = transfer_graph_log();
        let placement = verify_placement("shared_pc", &policy, &captured).unwrap();
        assert_eq!(placement.assigned_nodes, policy.nodes.len() + 1);
        assert_eq!(placement.approved_transfers.len(), 1);
        assert_eq!(
            placement.approved_transfers[0].source_value,
            "role_is_critic"
        );
        assert_eq!(placement.optimization, PalsGraphOptimization::Disable);
        let (_, mut missing) = transfer_graph_log();
        missing.transfer_lines.clear();
        assert!(verify_placement("shared_pc", &policy, &missing).is_err());
        let (_, mut wrong) = transfer_graph_log();
        wrong.transfer_lines[0] = "Add MemcpyFromHost after query for CUDAExecutionProvider".into();
        assert!(verify_placement("shared_pc", &policy, &wrong).is_err());
        let (_, mut doubled) = transfer_graph_log();
        doubled
            .capture_transfer("Add MemcpyFromHost after role_is_critic for CUDAExecutionProvider");
        assert!(verify_placement("shared_pc", &policy, &doubled).is_err());
        let (_, mut wrong_count) = transfer_graph_log();
        wrong_count.transfer_lines[1] = COPY_SUMMARY.replacen("1 Memcpy", "2 Memcpy", 1);
        assert!(verify_placement("shared_pc", &policy, &wrong_count).is_err());
        let (_, mut unknown_name) = transfer_graph_log();
        *unknown_name.lines.last_mut().unwrap() = "MemcpyFromHost (Memcpy_token_1)".into();
        assert!(verify_placement("shared_pc", &policy, &unknown_name).is_err());
        let (_, mut missing_destination) = transfer_graph_log();
        missing_destination.lines[11] = "Identity (unregistered_destination)".into();
        assert!(verify_placement("shared_pc", &policy, &missing_destination).is_err());
        assert!(verify_placement("shared_pc", &graph(), &captured).is_err());
        let mut tampered = placement.approved_transfers;
        tampered[0].source_shape = vec![1];
        assert!(!valid_transfers(&policy, &tampered));
    }
    #[test]
    fn cuda_scalar_copy_kernel_is_separate_from_neural_work_and_requires_bool_output() {
        let (policy, captured) = transfer_graph_log();
        let transfers = verify_placement("shared_pc", &policy, &captured)
            .unwrap()
            .approved_transfers;
        let mut events = serde_json::json!([
            {"cat":"Node","name":"nn_kernel_time","args":{"provider":CUDA,"op_name":"MatMul"}},
            {"cat":"Node","name":"Memcpy_kernel_time","args":{"provider":CUDA,"op_name":"MemcpyFromHost","output_type_shape":[{"bool":[]}]}}
        ]);
        let bytes = serde_json::to_vec(&events).unwrap();
        let witness = verify_profile(&bytes, &policy, &transfers).unwrap();
        assert_eq!(witness.cuda_transfer_kernels, 1);
        assert_eq!(witness.cuda_kernels, 2);
        assert_eq!(witness.neural_kernels, 1);
        assert!(verify_profile(&bytes, &policy, &[]).is_err());
        events[1]["args"]["output_type_shape"] = serde_json::json!([{"float":[]}]);
        assert!(
            verify_profile(&serde_json::to_vec(&events).unwrap(), &policy, &transfers).is_err()
        );
        events[1]["args"]["output_type_shape"] = serde_json::json!([{"bool":[1]}]);
        assert!(
            verify_profile(&serde_json::to_vec(&events).unwrap(), &policy, &transfers).is_err()
        );
    }
    #[test]
    fn pre_run_copy_origin_is_bounded_and_saved_as_metadata_without_raw_logger_text() {
        let root = io_root();
        let (policy, captured) = transfer_graph_log();
        let gate = verify_placement("shared_pc", &policy, &captured);
        let captured = Arc::new(Mutex::new(captured));
        let path = root.0.join("shared_pc-initial-placement.json");
        record_initial_log(&captured, &path, "shared_pc", true, Some(&gate)).unwrap();
        let bytes = std::fs::read(path).unwrap();
        let file: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(file["optimization"], "disable");
        assert_eq!(
            file["transfer_metadata"][0]["source_value"],
            "role_is_critic"
        );
        assert_eq!(
            file["approved_transfers"][0]["kind"],
            "bool_scalar_host_to_device"
        );
        assert!(!std::str::from_utf8(&bytes)
            .unwrap()
            .contains("Add MemcpyFromHost after"));
        let mut full = PlacementLog {
            bytes: MAX_PLACEMENT_BYTES,
            ..Default::default()
        };
        full.capture_transfer("x");
        assert!(full.overflow);
        assert!(full.transfer_lines.is_empty());
    }
    #[test]
    fn rejected_pre_run_metadata_is_saved_exclusively_without_nn_values() {
        let root = io_root();
        let captured = Arc::new(Mutex::new(log(&[
            "Node placements",
            "Node(s) placed on [CPUExecutionProvider]. Number of nodes: 1",
            "MatMul (p_nn)",
            "Node(s) placed on [CUDAExecutionProvider]. Number of nodes: 1",
            "MatMul (nn)",
            "unrecognized callback record",
        ])));
        let gate = verify_placement("public", &graph(), &captured.lock().unwrap());
        let detail = gate.as_ref().unwrap_err().detail;
        let path = root.0.join("public-initial-placement.json");
        record_initial_log(&captured, &path, "public", true, Some(&gate)).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        let file: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(file["gate"], "rejected");
        assert_eq!(file["gate_detail"], detail);
        assert_eq!(file["before_first_neural_run"], true);
        assert_eq!(file["nodes"][0]["provider"], CPU);
        assert_eq!(file["nodes"][0]["name"], "p_nn");
        assert_eq!(file["nodes"][0]["op"], "MatMul");
        assert!(file.get("lines").is_none());
        assert!(!std::str::from_utf8(&bytes)
            .unwrap()
            .contains("unrecognized callback record"));
        assert!(record_initial_log(&captured, &path, "public", true, Some(&gate)).is_err());
        assert_eq!(std::fs::read(&path).unwrap(), bytes);
        assert!(record_initial_log(
            &captured,
            &root.0.join("absent").join("record.json"),
            "public",
            true,
            Some(&gate)
        )
        .is_err());
        assert_eq!(gate.as_ref().unwrap_err().detail, detail);
        let failure_path = root.0.join("shared_pc-initial-placement.json");
        record_initial_log(&captured, &failure_path, "shared_pc", false, None).unwrap();
        let failure: serde_json::Value =
            serde_json::from_slice(&std::fs::read(failure_path).unwrap()).unwrap();
        assert_eq!(failure["session_created"], false);
        assert_eq!(failure["gate"], "not_reached");
    }
    #[test]
    fn placement_json_byte_ceiling_rejects_before_extending_output() {
        use std::io::Write;
        let mut output = BoundedPlacementJson { bytes: Vec::new() };
        output.write_all(&vec![0; MAX_PROFILE_BYTES]).unwrap();
        assert!(output.write_all(b"x").is_err());
        assert_eq!(output.bytes.len(), MAX_PROFILE_BYTES);
    }
    #[test]
    fn metadata_candidate_cannot_hide_model_data_or_unbounded_seed() {
        let node = |index,
                    name: &str,
                    op: &str,
                    inputs: &[&str],
                    outputs: &[&str],
                    annotation: &str,
                    input_origins: &[&str]| InventoryNode {
            index,
            name: name.into(),
            op: op.into(),
            domain: String::new(),
            inputs: inputs.iter().map(|name| (*name).into()).collect(),
            outputs: outputs.iter().map(|name| (*name).into()).collect(),
            static_annotation: annotation.into(),
            actual_provider: "unknown".into(),
            attribute_names: None,
            input_origins: input_origins.iter().map(|name| (*name).into()).collect(),
            constant_integral_seeds: Vec::new(),
        };
        let mut scope = InventoryScope {
            path: "public".into(),
            outputs: Vec::new(),
            inputs: vec![InventoryInput {
                name: "data".into(),
                dtype: "FLOAT".into(),
                shape: serde_json::json!([1, 384]),
            }],
            integral_initializers: vec![IntegralSeed {
                name: "index".into(),
                dtype: "INT64".into(),
                shape: vec![],
                source: "graph_initializer_metadata".into(),
            }],
            initializer_count: None,
            nodes: vec![
                node(
                    0,
                    "shape",
                    "Shape",
                    &["data"],
                    &["dimensions"],
                    "shape_metadata_candidate",
                    &["model_data_or_unknown"],
                ),
                node(
                    1,
                    "leaky",
                    "Gather",
                    &["data", "index"],
                    &["output"],
                    "shape_metadata_chain_candidate",
                    &["model_data_or_unknown", "small_integral_constant"],
                ),
            ],
            subgraphs: vec![],
        };
        let inspect = |scope: &InventoryScope| {
            inspect_scope(
                scope,
                &BTreeMap::new(),
                &mut graph(),
                &mut BTreeSet::new(),
                &mut 0,
            )
        };
        assert!(inspect(&scope).is_err());
        scope.nodes[1].inputs[0] = "dimensions".into();
        scope.nodes[1].input_origins[0] = "shape_metadata".into();
        assert!(inspect(&scope).is_ok());
        scope.integral_initializers[0].shape = vec![65];
        assert!(inspect(&scope).is_err());
        scope.integral_initializers[0].shape = vec![];
        scope.integral_initializers[0].dtype = "FLOAT".into();
        assert!(inspect(&scope).is_err());
    }
    #[test]
    fn initialization_rejects_nn_cpu_missing_coverage_and_unknown_provider_before_run() {
        let valid = [
            "Node placements",
            " Node(s) placed on [CPUExecutionProvider]. Number of nodes: 1",
            "  Shape (shape_node)",
            " Node(s) placed on [CUDAExecutionProvider]. Number of nodes: 1",
            "  MatMul (nn)",
        ];
        assert_eq!(
            verify_placement("public", &graph(), &log(&valid))
                .unwrap()
                .assigned_nodes,
            2
        );
        let mut changed = valid;
        changed[2] = "  MatMul (shape_node)";
        assert!(verify_placement("public", &graph(), &log(&changed)).is_err());
        changed[2] = "  Shape (nn)";
        assert!(verify_placement("public", &graph(), &log(&changed)).is_err());
        changed = valid;
        changed[4] = "  Shape (nn)";
        assert!(verify_placement("public", &graph(), &log(&changed)).is_err());
        changed[4] = "  FusedMatMul (unregistered)";
        assert!(verify_placement("public", &graph(), &log(&changed)).is_err());
        assert!(verify_placement("public", &graph(), &log(&valid[..4])).is_err());
        assert!(verify_placement(
            "public",
            &graph(),
            &log(&[
                "Node placements",
                "All nodes placed on [CPUExecutionProvider]. Number of nodes: 2"
            ])
        )
        .is_err());
        assert!(verify_placement(
            "public",
            &graph(),
            &log(&[
                "Node placements",
                "All nodes placed on [CUDAExecutionProvider]. Number of nodes: 0"
            ])
        )
        .is_err());
        assert!(verify_placement(
            "public",
            &graph(),
            &PlacementLog {
                overflow: true,
                ..Default::default()
            }
        )
        .is_err());
    }
    #[test]
    fn kernel_controls_require_exact_identity_integral_bounded_output_and_real_cuda() {
        let cuda = serde_json::json!({"cat":"Node","name":"nn_kernel_time","args":{"provider":CUDA,"op_name":"MatMul"}});
        let control = serde_json::json!({"cat":"Node","name":"shape_node_kernel_time","args":{"provider":CPU,"op_name":"Shape","output_type_shape":[{"int64":[4]}]}});
        let profile = |events| serde_json::to_vec(&events).unwrap();
        assert_eq!(
            verify_profile(&profile(vec![cuda.clone(), control.clone()]), &graph(), &[])
                .unwrap()
                .approved_cpu_control_kernels,
            1
        );
        let mut bad = control.clone();
        bad["args"]["output_type_shape"] = serde_json::json!([{"float":[4]}]);
        assert!(verify_profile(&profile(vec![cuda.clone(), bad]), &graph(), &[]).is_err());
        let mut bad = control;
        bad["args"]["output_type_shape"] = serde_json::json!([{"int64":[65]}]);
        assert!(verify_profile(&profile(vec![cuda, bad]), &graph(), &[]).is_err());
        let fake_nn = serde_json::json!({"cat":"Node","name":"nn_kernel_time","args":{"provider":CUDA,"op_name":"Shape"}});
        assert!(verify_profile(&profile(vec![fake_nn]), &graph(), &[]).is_err());
        assert!(verify_profile(b"[]", &graph(), &[]).is_err());
    }

    #[test]
    fn kernel_witness_requires_both_exact_pre_run_placements() {
        let policy = PalsCudaControlPolicy {
            manifest: [2; 32],
            inventory: [3; 32],
            graphs: [("public".into(), graph()), ("shared_pc".into(), graph())].into(),
            virtual_layout: None,
        };
        let profile = |node: &str, op: &str| serde_json::json!({"cat":"Node","name":format!("{node}_kernel_time"),"args":{"provider":CUDA,"op_name":op}});
        let public = serde_json::to_vec(&vec![profile("nn", "MatMul")]).unwrap();
        let shared =
            serde_json::to_vec(&vec![profile("p_nn", "Gemm"), profile("c_nn", "Gemm")]).unwrap();
        assert!(policy
            .verify_profiles(&public, &shared, vec![], [4; 32], [5; 32])
            .is_err());
        let make = |role: &str| PalsGraphPlacement {
            role: role.into(),
            graph_sha256: [1; 32],
            log_sha256: [6; 32],
            assigned_nodes: 2,
            cuda_nodes: 1,
            approved_cpu_control_nodes: 1,
            optimization: PalsGraphOptimization::Disable,
            approved_transfers: Vec::new(),
            recursive_coverage:
                "ort-1.22-finalize-recursive-exact-named-provider-coverage-before-first-run".into(),
        };
        assert!(policy
            .verify_profiles(
                &public,
                &shared,
                vec![make("public"), make("shared_pc")],
                [4; 32],
                [5; 32]
            )
            .is_ok());
        assert!(policy
            .verify_profiles(
                &public,
                &shared,
                vec![make("public"), make("public")],
                [4; 32],
                [5; 32]
            )
            .is_err());
        let mut wrong = make("shared_pc");
        wrong.graph_sha256 = [7; 32];
        assert!(policy
            .verify_profiles(
                &public,
                &shared,
                vec![make("public"), wrong],
                [4; 32],
                [5; 32]
            )
            .is_err());
    }
}
