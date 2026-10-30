//! Exact historical checkpoint bytes and replayed metadata, never past process
//! authority. Tensor/optimizer deserialization belongs to a separately pinned
//! scratch validator; matching a JSON projection does not prove that it ran.
use super::r_auxiliary_ledger::{RoventAuxiliaryLedger, valid_pc_rows};
use super::*;

pub const ROVENT_CHECKPOINT_PUBLICATION: &str = "rz-rovent-training-checkpoint-publication/1";
pub const ROVENT_RECORDED_TAIL_COST: &str = "rz-rovent-recorded-tail-cost/1";
pub const ROVENT_ORIGINAL_TRAINING_CLOSURE: &str = "rz-rovent-original-training-closure/1";
pub const ROVENT_CHECKPOINT_BOUNDARY_CLOSURE: &str = "rz-rovent-checkpoint-boundary-closure/1";
pub const ROVENT_CHECKPOINT_BOUNDARY_REPORT: &str = "rz-rovent-checkpoint-boundary-report/1";
pub const ROVENT_HISTORICAL_TENSOR_PROJECTION: &str =
    "rz-rovent-historical-checkpoint-projection/1";
pub const ROVENT_CHECKPOINT_R_AUXILIARY: &str = "rz-rovent-checkpoint-r-auxiliary/1";
const MAX_TENSOR_BYTES: u64 = 2 << 30;
const MAX_CHAIN: usize = 64;
const MAX_UPDATES: usize = 65_536;
const STATE_HASHES: [&str; 5] = [
    "model_state_sha256",
    "optimizer_state_sha256",
    "scheduler_state_sha256",
    "rng_state_sha256",
    "sampler_state_sha256",
];

#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum RoventCheckpointInitializationSource {
    Initialize {
        pc_checkpoint_sha256: String,
        nnuec_checkpoint_sha256: String,
    },
    Resume {
        tensor_checkpoint_sha256: String,
    },
    Fork {
        tensor_checkpoint_sha256: String,
        parent_stage_id: String,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RoventCheckpointBoundary {
    pub completed_updates: BTreeMap<String, u64>,
    pub ancestor_updates: BTreeMap<String, u64>,
    pub model_state_sha256: String,
    pub optimizer_state_sha256: String,
    pub scheduler_state_sha256: String,
    pub rng_state_sha256: String,
    pub sampler_state_sha256: String,
    pub retained_input_set_sha256: String,
    pub retained_input_count: u64,
    pub retained_origin_set_sha256: String,
    pub retained_origin_count: u64,
    pub snapshot_cost: BTreeMap<String, u64>,
}
impl RoventCheckpointBoundary {
    fn validate(&self) -> Result<(), ArenaError> {
        role_counts(&self.completed_updates)?;
        role_counts(&self.ancestor_updates)?;
        cost_map(&self.snapshot_cost)?;
        check(
            [
                &self.model_state_sha256,
                &self.optimizer_state_sha256,
                &self.scheduler_state_sha256,
                &self.rng_state_sha256,
                &self.sampler_state_sha256,
                &self.retained_input_set_sha256,
                &self.retained_origin_set_sha256,
            ]
            .iter()
            .all(|v| generation_sha(v))
                && self.retained_input_count <= MAX_TRAINER_ROWS as u64
                && self.retained_origin_count <= MAX_TRAINER_ROWS as u64,
            "historical checkpoint boundary state/set pins",
        )
    }
    fn state_value(&self) -> Value {
        json!({"completed_updates":self.completed_updates,"model_state_sha256":self.model_state_sha256,
            "optimizer_state_sha256":self.optimizer_state_sha256,"scheduler_state_sha256":self.scheduler_state_sha256,
            "rng_state_sha256":self.rng_state_sha256,"sampler_state_sha256":self.sampler_state_sha256,
            "cost":self.snapshot_cost,"pending_microbatches":0})
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
#[serde(tag = "status", rename_all = "snake_case", deny_unknown_fields)]
pub enum RoventRecordedTailCost {
    Known {
        delta: BTreeMap<String, u64>,
        evidence: RoventPublicationFile,
    },
    Unknown {
        reason: String,
        evidence: Option<RoventPublicationFile>,
    },
}
#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RoventRecordedClosure {
    pub process_scope: String,
    pub physical_scope: String,
    pub evidence: RoventPublicationFile,
}

/// Exact data-owner identity plus a separate checkpoint-prefix projection.
/// The caller supplies the independently verified R dataset owner; these SHAs
/// never make the checkpoint directory an R corpus root or recreate a lease.
#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RoventCheckpointRAuxiliary {
    pub schema: String,
    pub dataset_publication_sha256: String,
    pub manifest_file_sha256: String,
    pub index_file_sha256: String,
    pub parent_ledger_projection: RoventPublicationFile,
}
impl RoventCheckpointRAuxiliary {
    fn validate(&self) -> Result<(), ArenaError> {
        check(
            self.schema == ROVENT_CHECKPOINT_R_AUXILIARY
                && [
                    &self.dataset_publication_sha256,
                    &self.manifest_file_sha256,
                    &self.index_file_sha256,
                ]
                .iter()
                .all(|value| generation_sha(value)),
            "R auxiliary checkpoint sidecar needs exact original data-owner pins",
        )?;
        self.parent_ledger_projection.validate()
    }
}

/// A durable report from the original parent, not current process custody.
/// An externally registered publication must pin these exact bytes. Neither
/// this DTO nor its classification grants numeric PID signal/reap authority.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoventOriginalTrainingClosure {
    pub schema: String,
    pub execution_id: String,
    pub window_id: String,
    pub terminal_receipt_sha256: String,
    pub terminal_receipt_bytes: u64,
    pub process_receipt: Option<crate::process::ProcessReceipt>,
    pub child_identity: RoventHistoricalChildIdentity,
    pub child_reap: RoventHistoricalChildReap,
    pub physical_completion: RoventHistoricalPhysicalCompletion,
}
#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RoventHistoricalChildIdentity {
    pub boot_id: String,
    pub pid: u32,
    pub start_ticks: u64,
    pub cgroup_path: String,
    pub cgroup_device: u64,
    pub cgroup_inode: u64,
}
#[derive(Clone, Copy, Debug, Serialize, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum RoventHistoricalChildReap {
    OwnedChildReaped,
    Unresolved,
}
#[derive(Clone, Copy, Debug, Serialize, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum RoventHistoricalPhysicalCompletion {
    Confirmed,
    Unknown,
}
#[derive(Clone, Copy, Debug, Serialize, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum RoventHistoricalClosureClassification {
    RecordedClosed,
    Unresolved,
}

#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RoventCheckpointPublication {
    pub schema: String,
    pub execution_id: String,
    pub window_id: String,
    pub collection_source_sha256: String,
    pub collection_runtime_library_sha256: String,
    pub trainer_source_sha256: String,
    pub trainer_python_executable_sha256: String,
    pub trainer_dependency_set_sha256: String,
    pub training_model_semantics_sha256: String,
    pub initialization_source: RoventCheckpointInitializationSource,
    pub collection_model_epoch: u64,
    pub collection_context_epoch: u64,
    /// Actual training/output epoch, independent of sealed collection input.
    pub training_model_epoch: u64,
    pub recipe_sha256: String,
    pub stage_id: String,
    pub training_profile_sha256: String,
    pub membership_sha256: String,
    pub current_dataset_publication_sha256: String,
    pub ancestor_dataset_publication_sha256: Vec<String>,
    pub tensor: RoventPublicationFile,
    pub checkpoint_retirement: RoventPublicationFile,
    pub update_receipts: Vec<RoventPublicationFile>,
    pub terminal_receipt: RoventPublicationFile,
    pub original_recipe: RoventPublicationFile,
    pub original_membership: RoventPublicationFile,
    pub original_window_spec: RoventPublicationFile,
    pub parent_publication: Option<RoventPublicationFile>,
    pub boundary: RoventCheckpointBoundary,
    pub recorded_tail_cost: RoventRecordedTailCost,
    pub recorded_closure: RoventRecordedClosure,
    #[serde(default, skip_serializing_if = "RoventRAuxiliaryPresence::is_absent")]
    pub r_auxiliary: RoventRAuxiliaryPresence<RoventCheckpointRAuxiliary>,
}
impl RoventCheckpointPublication {
    pub fn validate(&self) -> Result<(), ArenaError> {
        check(
            self.schema == ROVENT_CHECKPOINT_PUBLICATION
                && [&self.execution_id, &self.window_id, &self.stage_id]
                    .iter()
                    .all(|v| generation_identifier(v))
                && [
                    &self.collection_source_sha256,
                    &self.collection_runtime_library_sha256,
                    &self.trainer_source_sha256,
                    &self.trainer_python_executable_sha256,
                    &self.trainer_dependency_set_sha256,
                    &self.training_model_semantics_sha256,
                    &self.recipe_sha256,
                    &self.training_profile_sha256,
                    &self.membership_sha256,
                    &self.current_dataset_publication_sha256,
                ]
                .iter()
                .all(|v| generation_sha(v))
                && self.ancestor_dataset_publication_sha256.len() <= 256
                && self
                    .ancestor_dataset_publication_sha256
                    .iter()
                    .all(|v| generation_sha(v))
                && self
                    .ancestor_dataset_publication_sha256
                    .windows(2)
                    .all(|p| p[0] < p[1])
                && self.update_receipts.len() <= MAX_UPDATES,
            "historical checkpoint publication identity/source/lineage limits",
        )?;
        self.boundary.validate()?;
        if let Some(auxiliary) = self.r_auxiliary.enabled() {
            auxiliary.validate()?;
        }
        check(
            bounded_text(&self.recorded_closure.process_scope)
                && bounded_text(&self.recorded_closure.physical_scope),
            "recorded closure needs explicit historical scopes",
        )?;
        match &self.initialization_source {
            RoventCheckpointInitializationSource::Initialize {
                pc_checkpoint_sha256,
                nnuec_checkpoint_sha256,
            } => check(
                self.parent_publication.is_none()
                    && generation_sha(pc_checkpoint_sha256)
                    && generation_sha(nnuec_checkpoint_sha256),
                "historical initialize requires two explicit assets and no parent",
            )?,
            RoventCheckpointInitializationSource::Resume {
                tensor_checkpoint_sha256,
            } => check(
                self.parent_publication.is_some() && generation_sha(tensor_checkpoint_sha256),
                "historical resume parent pin",
            )?,
            RoventCheckpointInitializationSource::Fork {
                tensor_checkpoint_sha256,
                parent_stage_id,
            } => check(
                self.parent_publication.is_some()
                    && generation_sha(tensor_checkpoint_sha256)
                    && generation_identifier(parent_stage_id),
                "historical fork source/stage pin",
            )?,
        }
        let mut paths = BTreeSet::new();
        for file in self.artifacts() {
            file.validate()?;
            check(
                paths.insert(&file.path),
                "duplicate historical checkpoint artifact path",
            )?;
        }
        check(
            self.tensor.bytes <= MAX_TENSOR_BYTES,
            "historical tensor byte cap",
        )?;
        match &self.recorded_tail_cost {
            RoventRecordedTailCost::Known { delta, .. } => cost_map(delta)?,
            RoventRecordedTailCost::Unknown { reason, .. } => {
                check(bounded_text(reason), "unknown tail cost requires a reason")?
            }
        }
        Ok(())
    }
    pub(super) fn artifacts(&self) -> Vec<&RoventPublicationFile> {
        let mut files = vec![
            &self.tensor,
            &self.checkpoint_retirement,
            &self.terminal_receipt,
            &self.recorded_closure.evidence,
            &self.original_recipe,
            &self.original_membership,
            &self.original_window_spec,
        ];
        files.extend(&self.update_receipts);
        if let Some(parent) = &self.parent_publication {
            files.push(parent);
        }
        if let Some(auxiliary) = self.r_auxiliary.enabled() {
            files.push(&auxiliary.parent_ledger_projection);
        }
        match &self.recorded_tail_cost {
            RoventRecordedTailCost::Known { evidence, .. } => files.push(evidence),
            RoventRecordedTailCost::Unknown {
                evidence: Some(evidence),
                ..
            } => files.push(evidence),
            _ => (),
        }
        files
    }
}

/// Historical byte/metadata admission only. This has no live trainer/physical
/// completion constructor and is not sufficient for loading tensor state.
pub struct VerifiedRoventCheckpointSource {
    root: PathBuf,
    publication_pin: RoventTrainingFilePin,
    publication: RoventCheckpointPublication,
    tensor: HeldHistoricalFile,
    held: Vec<HeldHistoricalFile>,
    metadata_pins: Vec<RoventTrainingFilePin>,
    current_dataset: Arc<VerifiedRoventDataset>,
    ancestor_datasets: Vec<Arc<VerifiedRoventDataset>>,
    parent: Option<Arc<Self>>,
    retained_inputs: BTreeSet<String>,
    retained_origins: BTreeSet<String>,
    update_chain_sha256: String,
    tail_update_counts: BTreeMap<String, u64>,
    original_window_schema: String,
    nnuec_prefix_updates: u64,
    closure: RoventOriginalTrainingClosure,
    closure_classification: RoventHistoricalClosureClassification,
    r_auxiliary_ledger: Option<RoventAuxiliaryLedger>,
    r_auxiliary_checkpoint_projection: Option<Value>,
}
impl VerifiedRoventCheckpointSource {
    pub fn open(
        root: &Path,
        publication: RoventTrainingFilePin,
        current_dataset: Arc<VerifiedRoventDataset>,
        ancestor_datasets: Vec<Arc<VerifiedRoventDataset>>,
        budget: &RoventRecoveryReadBudget,
    ) -> Result<Self, ArenaError> {
        Self::open_with_r_auxiliary(
            root,
            publication,
            current_dataset,
            ancestor_datasets,
            Vec::new(),
            budget,
        )
    }
    /// Enabled publications require all exact current/ancestor R data owners.
    /// The compatibility `open` supplies none and therefore rejects enabled
    /// history rather than replacing its learning stream or inventing rows.
    pub fn open_with_r_auxiliary(
        root: &Path,
        publication: RoventTrainingFilePin,
        current_dataset: Arc<VerifiedRoventDataset>,
        ancestor_datasets: Vec<Arc<VerifiedRoventDataset>>,
        r_datasets: Vec<Arc<VerifiedRoventRAuxiliaryDataset>>,
        budget: &RoventRecoveryReadBudget,
    ) -> Result<Self, ArenaError> {
        check(
            root.is_absolute()
                && publication.path.starts_with(root)
                && publication.path != root
                && ancestor_datasets.len() <= 256
                && r_datasets.len() <= 257,
            "historical checkpoint root/ancestors",
        )?;
        let mut datasets = BTreeMap::new();
        collect_datasets(&current_dataset, &mut datasets, 0)?;
        for ancestor in &ancestor_datasets {
            collect_datasets(ancestor, &mut datasets, 0)?;
        }
        let mut auxiliary_datasets = BTreeMap::new();
        for dataset in r_datasets {
            collect_r_auxiliary_datasets(&dataset, &mut auxiliary_datasets, 0)?;
        }
        let mut visited = ChainVisited::default();
        let source = Self::open_inner(
            root,
            publication,
            &datasets,
            &auxiliary_datasets,
            &mut visited,
            0,
            budget,
        )?;
        let expected = sorted_dataset_pins(&ancestor_datasets)?;
        check(
            source.publication.current_dataset_publication_sha256
                == current_dataset.registration.publication.sha256
                && source.publication.ancestor_dataset_publication_sha256 == expected,
            "checkpoint dataset pins differ from externally registered current/ancestor owners",
        )?;
        Ok(source)
    }
    fn open_inner(
        root: &Path,
        pin: RoventTrainingFilePin,
        datasets: &BTreeMap<String, Arc<VerifiedRoventDataset>>,
        r_datasets: &BTreeMap<String, Arc<VerifiedRoventRAuxiliaryDataset>>,
        visited: &mut ChainVisited,
        depth: usize,
        budget: &RoventRecoveryReadBudget,
    ) -> Result<Self, ArenaError> {
        check(
            depth < MAX_CHAIN
                && visited.paths.insert(pin.path.clone())
                && visited.sha.insert(pin.sha256.clone()),
            "historical parent checkpoint depth/cycle/duplicate path or SHA",
        )?;
        let (publication_file, raw) = budget.read(&pin, MAX_METADATA_BYTES, true)?;
        let publication: RoventCheckpointPublication = decode(&raw)?;
        publication.validate()?;
        check(
            visited.windows.insert((
                publication.execution_id.clone(),
                publication.window_id.clone(),
            )),
            "historical checkpoint chain repeats an execution/window cost identity",
        )?;
        let dataset = Arc::clone(
            datasets
                .get(&publication.current_dataset_publication_sha256)
                .ok_or_else(|| {
                    invalid("checkpoint references an unregistered historical dataset")
                })?,
        );
        let ancestor_datasets = publication
            .ancestor_dataset_publication_sha256
            .iter()
            .map(|sha| {
                datasets
                    .get(sha)
                    .cloned()
                    .ok_or_else(|| invalid("checkpoint ancestor dataset is not retained"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        check(
            publication.ancestor_dataset_publication_sha256
                == sorted_dataset_pins(&dataset.ancestors)?,
            "checkpoint ancestor dataset list differs from the dataset's validated ancestry",
        )?;
        check(
            publication.collection_source_sha256 == dataset.registration.collection_source_sha256
                && publication.collection_runtime_library_sha256
                    == dataset.registration.collection_runtime_library_sha256
                && publication.collection_model_epoch == dataset.manifest.model_epoch
                && publication.collection_context_epoch == dataset.manifest.context_epoch
                && publication.recipe_sha256 == dataset.manifest.recipe_sha256
                && publication.stage_id == dataset.manifest.stage_id,
            "checkpoint changed sealed collection source/runtime/epochs/recipe/stage",
        )?;
        let parent = publication
            .parent_publication
            .as_ref()
            .map(|file| {
                Self::open_inner(
                    root,
                    archive_pin(root, file)?,
                    datasets,
                    r_datasets,
                    visited,
                    depth + 1,
                    budget,
                )
                .map(Arc::new)
            })
            .transpose()?;
        let mut held = vec![publication_file];
        let recipe = read_json(root, &publication.original_recipe, budget, &mut held)?;
        let membership = read_json(root, &publication.original_membership, budget, &mut held)?;
        let window = read_json(root, &publication.original_window_spec, budget, &mut held)?;
        let verified_profile = validate_documents(&publication, &recipe, &membership, &window)?;
        let original_window_schema = window["schema"]
            .as_str()
            .ok_or_else(|| invalid("historical window schema"))?
            .to_owned();
        let mut auxiliary = initial_r_auxiliary_ledger(
            &publication,
            &window,
            Some(&dataset),
            &recipe,
            parent.as_deref(),
            r_datasets,
            budget,
        )?;
        let (initial, ancestors) = initial_counts(&publication, &recipe, parent.as_deref())?;
        let mut rows = dataset.rows.clone();
        for ancestor in &ancestor_datasets {
            for (id, row) in &ancestor.rows {
                if let Some(old) = rows.get(id) {
                    check(same_row(old, row), "conflicting ancestor sample origin")?;
                } else {
                    rows.insert(id.clone(), row.clone());
                }
            }
        }
        check(
            rows.len() <= MAX_TRAINER_ROWS,
            "historical checkpoint row namespace capacity",
        )?;
        let mut ledger = TrainerLedger::new(
            window.clone(),
            recipe,
            publication.membership_sha256.clone(),
            dataset.publication.manifest.sha256.clone(),
            rows,
            initial,
            ancestors,
        )?;
        // The live /1 digest stays unchanged. Only this private historical
        // ledger consumes the separately validated /1 or /2 profile.
        ledger.profile_sha256 = verified_profile;
        // This is a private metadata replay, not begin_window or dispatch.
        ledger.begun = true;
        ledger
            .consumed_inputs
            .extend(dataset.manifest.inherited_input_sha256.iter().cloned());
        ledger
            .consumed_origins
            .extend(dataset.manifest.inherited_origin_sha256.iter().cloned());
        if let Some(parent) = &parent {
            ledger
                .consumed_inputs
                .extend(parent.retained_inputs.iter().cloned());
            ledger
                .consumed_origins
                .extend(parent.retained_origins.iter().cloned());
        }
        if let Some(auxiliary) = &auxiliary {
            auxiliary.validate_joint_retained(&ledger)?;
        }
        let mut prefix = boundary_snapshot(&mut ledger, &publication.boundary)?;
        let mut auxiliary_prefix = prefix.as_ref().and_then(|_| auxiliary.clone());
        let auxiliary_initial_steps = auxiliary
            .as_ref()
            .map_or(0, RoventAuxiliaryLedger::completed_nnuec_updates);
        let mut nnuec_seen = 0_u64;
        let mut nnuec_prefix_updates = prefix.as_ref().map(|_| 0_u64);
        let mut metadata_pins = Vec::new();
        let mut chain = Sha256::new();
        chain.update(b"rz-rovent-historical-update-chain/1\0");
        chain.update(pin.sha256.as_bytes());
        if let Some(parent) = &parent {
            chain.update(parent.update_chain_sha256.as_bytes());
        }
        for (index, file) in publication.update_receipts.iter().enumerate() {
            let absolute = archive_pin(root, file)?;
            let (receipt_file, raw) =
                budget.read(&absolute, MAX_TRAINER_RPC_BYTES as u64 + 4096, true)?;
            let receipt: Value = decode(&raw)?;
            validate_update_with_r_auxiliary(
                &publication,
                &mut ledger,
                &receipt,
                index,
                auxiliary.as_mut(),
            )?;
            if let Some(auxiliary) = &auxiliary {
                nnuec_seen = auxiliary
                    .completed_nnuec_updates()
                    .checked_sub(auxiliary_initial_steps)
                    .ok_or_else(|| invalid("historical R auxiliary steps regressed"))?;
            } else if receipt["update"]["event"]["phase"] == "pc"
                && receipt["update"]["event"]["role"] != "validator"
                && count(&receipt["update"]["event"]["head_counts"]["nnuec_wdl"])? > 0
            {
                nnuec_seen = nnuec_seen
                    .checked_add(1)
                    .ok_or_else(|| invalid("historical NNUEC update overflow"))?;
            }
            if prefix.is_none() {
                if ledger.completed == publication.boundary.completed_updates
                    && let Some(auxiliary) = &auxiliary
                {
                    auxiliary.validate_joint_retained(&ledger)?;
                }
                prefix = boundary_snapshot(&mut ledger, &publication.boundary)?;
                if prefix.is_some() {
                    nnuec_prefix_updates = Some(nnuec_seen);
                    auxiliary_prefix = auxiliary.clone();
                }
            }
            chain.update(file.sha256.as_bytes());
            metadata_pins.push(receipt_file.pin.clone());
            // Retain the exact pin rather than one FD per optimizer update.
            // These bytes are historical metadata, not a native capability.
        }
        let (retained_inputs, retained_origins, prefix_state) = prefix.ok_or_else(|| {
            invalid("checkpoint boundary count is absent from the original update chain")
        })?;
        check(
            ledger.pending.is_empty(),
            "historical replay ended inside accumulation",
        )?;
        validate_boundary_sets(&publication.boundary, &retained_inputs, &retained_origins)?;
        let retirement = read_json(root, &publication.checkpoint_retirement, budget, &mut held)?;
        validate_retirement_with_r_auxiliary(
            &publication,
            &retirement,
            prefix_state.as_ref(),
            &original_window_schema,
            auxiliary_prefix.as_ref(),
        )?;
        validate_r_auxiliary_prefix(
            root,
            &publication,
            auxiliary_prefix.as_ref(),
            budget,
            &mut held,
        )?;
        let r_auxiliary_checkpoint_projection = auxiliary_prefix
            .as_ref()
            .map(|_| retirement["checkpoint"]["r_auxiliary"].clone());
        if prefix_state.is_none()
            && matches!(
                publication.initialization_source,
                RoventCheckpointInitializationSource::Resume { .. }
            )
        {
            let previous = parent
                .as_ref()
                .ok_or_else(|| invalid("resume lacks historical parent"))?
                .publication
                .boundary
                .state_value();
            // Cost is window-local; the sampler journal may change at begin.
            // With no optimizer update, the four core states cannot change.
            for key in [
                "model_state_sha256",
                "optimizer_state_sha256",
                "scheduler_state_sha256",
                "rng_state_sha256",
            ] {
                check(
                    previous[key] == retirement["checkpoint"][key],
                    "zero-new-update resume changed core checkpoint state",
                )?;
            }
        }
        let terminal = read_json(root, &publication.terminal_receipt, budget, &mut held)?;
        let closure_value = read_json(
            root,
            &publication.recorded_closure.evidence,
            budget,
            &mut held,
        )?;
        let closure = parse_recorded_closure(closure_value)?;
        validate_terminal(&publication, &terminal, &ledger, &closure)?;
        let closure_classification = validate_closure(&publication, &closure, &terminal)?;
        validate_tail(root, &publication, budget, &mut held, &ledger)?;
        let (tensor, _) = budget.read(
            &archive_pin(root, &publication.tensor)?,
            MAX_TENSOR_BYTES,
            false,
        )?;
        let mut tail_update_counts = BTreeMap::new();
        for role in ROLES {
            tail_update_counts.insert(
                role.into(),
                ledger.completed[role]
                    .checked_sub(publication.boundary.completed_updates[role])
                    .ok_or_else(|| invalid("historical tail counts regressed"))?,
            );
        }
        check(
            retirement["checkpoint"]["checkpoint"]["sha256"] == tensor.pin.sha256
                && retirement["checkpoint"]["checkpoint"]["bytes"] == tensor.pin.bytes,
            "historical retirement does not bind relocated tensor bytes",
        )?;
        Ok(Self {
            root: root.into(),
            publication_pin: pin,
            publication,
            tensor,
            held,
            metadata_pins,
            current_dataset: dataset,
            ancestor_datasets,
            parent,
            retained_inputs,
            retained_origins,
            update_chain_sha256: format!("{:x}", chain.finalize()),
            tail_update_counts,
            original_window_schema,
            nnuec_prefix_updates: nnuec_prefix_updates
                .ok_or_else(|| invalid("historical NNUEC boundary was not captured"))?,
            closure,
            closure_classification,
            r_auxiliary_ledger: auxiliary_prefix,
            r_auxiliary_checkpoint_projection,
        })
    }
    pub fn publication(&self) -> &RoventCheckpointPublication {
        &self.publication
    }
    pub fn publication_pin(&self) -> &RoventTrainingFilePin {
        &self.publication_pin
    }
    pub(super) fn archive_root(&self) -> &Path {
        &self.root
    }
    pub fn tensor_file(&self) -> (&RoventTrainingFilePin, &File) {
        (&self.tensor.pin, &self.tensor.file)
    }
    pub fn parent(&self) -> Option<&Arc<Self>> {
        self.parent.as_ref()
    }
    /// A codec observation, not a trainer runtime capability. Legacy /1 source
    /// and runtime pins cannot establish the separated /2 trainer identities.
    pub fn original_window_schema(&self) -> &str {
        &self.original_window_schema
    }
    pub fn retained_inputs(&self) -> &BTreeSet<String> {
        &self.retained_inputs
    }
    pub fn retained_origins(&self) -> &BTreeSet<String> {
        &self.retained_origins
    }
    /// Exact original P/C data owners referenced by this checkpoint chain.
    /// These may differ from the new execution's current P/C or R graph; no
    /// unverified registry entry or similarly named publication is added.
    pub(super) fn historical_pc_datasets(
        &self,
    ) -> Result<Vec<Arc<VerifiedRoventDataset>>, ArenaError> {
        let mut all = BTreeMap::new();
        let mut cursor = Some(self);
        let mut depth = 0;
        while let Some(source) = cursor {
            depth += 1;
            check(depth <= MAX_CHAIN, "historical P/C checkpoint owner depth")?;
            collect_datasets(&source.current_dataset, &mut all, 0)?;
            for ancestor in &source.ancestor_datasets {
                collect_datasets(ancestor, &mut all, 0)?;
            }
            cursor = source.parent.as_deref();
        }
        Ok(all.into_values().collect())
    }
    /// Selected snapshot only; no later same-window R work is resumed.
    pub(super) fn r_auxiliary_ledger(&self) -> Option<&RoventAuxiliaryLedger> {
        self.r_auxiliary_ledger.as_ref()
    }
    pub(super) fn r_auxiliary_checkpoint_projection(&self) -> Option<&Value> {
        self.r_auxiliary_checkpoint_projection.as_ref()
    }
    /// R-only historical lineage, including R-enabled stages separated by a
    /// disabled stage. P/C consumed sets never enter this namespace. Returning
    /// recorded seals neither reconstructs sample custody nor grants a lease.
    pub(super) fn historical_r_auxiliary_seals(
        &self,
    ) -> Result<(BTreeSet<String>, BTreeSet<String>), ArenaError> {
        let mut inputs = BTreeSet::new();
        let mut origins = BTreeSet::new();
        let mut cursor = Some(self);
        let mut depth = 0;
        while let Some(source) = cursor {
            depth += 1;
            check(depth <= MAX_CHAIN, "historical R seal lineage depth")?;
            if let Some(auxiliary) = source.r_auxiliary_ledger() {
                let (retained_inputs, retained_origins, ancestor_inputs, ancestor_origins) =
                    auxiliary.separate_seal_sets();
                union_r_seals(&mut inputs, retained_inputs)?;
                union_r_seals(&mut inputs, ancestor_inputs)?;
                union_r_seals(&mut origins, retained_origins)?;
                union_r_seals(&mut origins, ancestor_origins)?;
            }
            cursor = source.parent.as_deref();
        }
        Ok((inputs, origins))
    }
    /// Selected snapshot lineage; later same-window updates are never counted.
    /// Actual export still requires the separate pinned tensor-state owner.
    pub fn learned_nnuec_lineage(&self) -> Result<Value, ArenaError> {
        let mut updates = 0_u64;
        let mut datasets = BTreeSet::new();
        let mut last_execution_id = None;
        let mut last_window_id = None;
        let mut cursor = Some(self);
        let mut depth = 0_usize;
        while let Some(source) = cursor {
            depth += 1;
            check(depth <= MAX_CHAIN, "historical NNUEC lineage depth")?;
            updates = updates
                .checked_add(source.nnuec_prefix_updates)
                .ok_or_else(|| invalid("historical NNUEC lineage overflow"))?;
            if source.nnuec_prefix_updates > 0 {
                datasets.insert(match &source.r_auxiliary_ledger {
                    Some(auxiliary) => auxiliary.dataset().manifest_file_sha256().to_owned(),
                    None => source.current_dataset.publication.manifest.sha256.clone(),
                });
                if last_execution_id.is_none() {
                    last_execution_id = Some(source.publication.execution_id.clone());
                    last_window_id = Some(source.publication.window_id.clone());
                }
            }
            cursor = source.parent.as_deref();
        }
        Ok(
            json!({"schema":"rz-rovent-historical-learned-nnuec-lineage/1","completed_nnuec_updates":updates,
            "dataset_manifest_sha256":datasets,"last_execution_id":last_execution_id,"last_window_id":last_window_id,
            "historical_metadata_only":true,"past_native_authority_reissued":false,"execution_authorized":false}),
        )
    }
    pub fn recorded_closure(
        &self,
    ) -> (
        &RoventOriginalTrainingClosure,
        RoventHistoricalClosureClassification,
    ) {
        (&self.closure, self.closure_classification)
    }
    pub fn historical_tail_records(&self) -> Vec<Value> {
        let mut records = self
            .parent
            .as_ref()
            .map(|p| p.historical_tail_records())
            .unwrap_or_default();
        records.push(
            json!({"checkpoint_publication_sha256":self.publication_pin.sha256,
            "execution_id":self.publication.execution_id,"window_id":self.publication.window_id,
            "tail":self.publication.recorded_tail_cost}),
        );
        records
    }
    pub fn observation(&self) -> Value {
        json!({"schema":"rz-rovent-historical-checkpoint-metadata-admission/1",
            "publication_sha256":self.publication_pin.sha256,"tensor_sha256":self.tensor.pin.sha256,
            "update_chain_sha256":self.update_chain_sha256,"snapshot_completed_updates":self.publication.boundary.completed_updates,
            "post_snapshot_optimizer_updates":self.tail_update_counts,"recorded_tail_cost":self.publication.recorded_tail_cost,
            "recorded_closure_classification":self.closure_classification,
            "historical_window_schema":self.original_window_schema,
            "historical_window_identity_scope":if self.original_window_schema==ROVENT_TRAIN_WINDOW_V2 {"explicit_v2_recorded_pins"} else {"original_v1_unseparated"},
            "historical_metadata_integrity_only":true,"tensor_state_validated":false,
            "past_native_authority_reissued":false,"past_physical_completion_reissued":false,"execution_authorized":false})
    }
    pub fn revalidate(&self, budget: &RoventRecoveryReadBudget) -> Result<(), ArenaError> {
        for original in self.held.iter().chain(std::iter::once(&self.tensor)) {
            let (now, _) = budget.read(&original.pin, original.pin.bytes, false)?;
            check(
                now.identity == original.identity,
                "retained historical checkpoint file identity changed",
            )?;
        }
        for pin in &self.metadata_pins {
            let _ = budget.read(pin, pin.bytes, false)?;
        }
        self.current_dataset.revalidate(budget)?;
        for dataset in &self.ancestor_datasets {
            dataset.revalidate(budget)?;
        }
        if let Some(auxiliary) = &self.r_auxiliary_ledger {
            auxiliary.dataset().revalidate(budget)?;
        }
        if let Some(parent) = &self.parent {
            parent.revalidate(budget)?;
        }
        Ok(())
    }
}
fn union_r_seals(
    destination: &mut BTreeSet<String>,
    original: &BTreeSet<String>,
) -> Result<(), ArenaError> {
    for seal in original {
        check(
            generation_sha(seal),
            "historical R lineage contains an invalid seal",
        )?;
        check(
            destination.contains(seal) || destination.len() < MAX_TRAINER_ROWS,
            "historical R seal lineage capacity",
        )?;
        destination.insert(seal.clone());
    }
    Ok(())
}

#[derive(Default)]
struct ChainVisited {
    paths: BTreeSet<PathBuf>,
    sha: BTreeSet<String>,
    windows: BTreeSet<(String, String)>,
}
fn bounded_text(text: &str) -> bool {
    !text.trim().is_empty() && text.len() <= 4096 && !text.chars().any(char::is_control)
}
fn role_counts(value: &BTreeMap<String, u64>) -> Result<(), ArenaError> {
    check(
        value.len() == 3 && ROLES.iter().all(|r| value.contains_key(*r)),
        "exact historical role count map",
    )
}
fn cost_map(value: &BTreeMap<String, u64>) -> Result<(), ArenaError> {
    check(
        value.len() == 8 && COST_FIELDS.iter().all(|r| value.contains_key(*r)),
        "exact historical cost map",
    )
}
fn read_json(
    root: &Path,
    file: &RoventPublicationFile,
    budget: &RoventRecoveryReadBudget,
    held: &mut Vec<HeldHistoricalFile>,
) -> Result<Value, ArenaError> {
    let (owner, raw) = budget.read(&archive_pin(root, file)?, MAX_METADATA_BYTES, true)?;
    held.push(owner);
    decode(&raw)
}
fn sorted_dataset_pins(datasets: &[Arc<VerifiedRoventDataset>]) -> Result<Vec<String>, ArenaError> {
    let mut result = datasets
        .iter()
        .map(|d| d.registration.publication.sha256.clone())
        .collect::<Vec<_>>();
    result.sort();
    check(
        result.windows(2).all(|v| v[0] < v[1]),
        "duplicate historical dataset owners",
    )?;
    Ok(result)
}
fn collect_datasets(
    owner: &Arc<VerifiedRoventDataset>,
    all: &mut BTreeMap<String, Arc<VerifiedRoventDataset>>,
    depth: usize,
) -> Result<(), ArenaError> {
    check(depth <= 256, "historical dataset ancestry depth")?;
    let pin = &owner.registration.publication.sha256;
    if let Some(previous) = all.get(pin) {
        check(
            previous.same_original_metadata(owner),
            "same publication pin has conflicting dataset owners",
        )?;
        return Ok(());
    }
    check(all.len() < 257, "historical dataset ancestry capacity")?;
    all.insert(pin.clone(), Arc::clone(owner));
    for ancestor in &owner.ancestors {
        collect_datasets(ancestor, all, depth + 1)?;
    }
    Ok(())
}
fn same_row(a: &NativeRowAuthority, b: &NativeRowAuthority) -> bool {
    a.family == b.family
        && a.input_descriptor == b.input_descriptor
        && a.origin == b.origin
        && a.role == b.role
        && a.retirement_sha256 == b.retirement_sha256
        && a.head_raw_sha256 == b.head_raw_sha256
        && a.valid_heads == b.valid_heads
}

fn collect_r_auxiliary_datasets(
    owner: &Arc<VerifiedRoventRAuxiliaryDataset>,
    all: &mut BTreeMap<String, Arc<VerifiedRoventRAuxiliaryDataset>>,
    depth: usize,
) -> Result<(), ArenaError> {
    check(depth <= 256, "historical R dataset ancestry depth")?;
    let pin = &owner.publication_pin().sha256;
    if let Some(previous) = all.get(pin) {
        check(
            previous.manifest_file_sha256() == owner.manifest_file_sha256()
                && previous.index_file_sha256() == owner.index_file_sha256()
                && previous.registration() == owner.registration(),
            "same R dataset publication has conflicting retained owners",
        )?;
        return Ok(());
    }
    check(all.len() < 257, "historical R dataset ancestry capacity")?;
    all.insert(pin.clone(), Arc::clone(owner));
    for ancestor in owner.ancestors() {
        collect_r_auxiliary_datasets(ancestor, all, depth + 1)?;
    }
    Ok(())
}

pub(super) fn initial_r_auxiliary_ledger(
    publication: &RoventCheckpointPublication,
    window: &Value,
    current_pc: Option<&Arc<VerifiedRoventDataset>>,
    source_recipe: &Value,
    parent: Option<&VerifiedRoventCheckpointSource>,
    datasets: &BTreeMap<String, Arc<VerifiedRoventRAuxiliaryDataset>>,
    budget: &RoventRecoveryReadBudget,
) -> Result<Option<RoventAuxiliaryLedger>, ArenaError> {
    if window["schema"] == "rz-rovent-train-window/1" {
        check(
            publication.r_auxiliary.is_absent(),
            "legacy /1 history cannot acquire an R auxiliary sidecar",
        )?;
        return Ok(None);
    }
    let typed: RoventTrainWindowV2 =
        serde_json::from_value(window.clone()).map_err(|e| invalid(e.to_string()))?;
    typed.validate()?;
    check(
        typed.nnuec_r_auxiliary.state() == publication.r_auxiliary.state(),
        "checkpoint R auxiliary presence differs from its original window",
    )?;
    let Some(sidecar) = publication.r_auxiliary.enabled() else {
        if matches!(
            publication.initialization_source,
            RoventCheckpointInitializationSource::Resume { .. }
        ) {
            check(
                parent
                    .and_then(|source| source.r_auxiliary_ledger())
                    .is_none(),
                "same-stage resume cannot discard its original R auxiliary ledger",
            )?;
        }
        return Ok(None);
    };
    let recipe = typed
        .nnuec_r_auxiliary
        .enabled()
        .ok_or_else(|| invalid("enabled checkpoint R recipe is absent"))?;
    let dataset = Arc::clone(
        datasets
            .get(&sidecar.dataset_publication_sha256)
            .ok_or_else(|| {
                invalid("enabled checkpoint requires its exact external R dataset owner")
            })?,
    );
    let registered = dataset.registration();
    check(
        dataset.manifest_file_sha256() == sidecar.manifest_file_sha256
            && dataset.index_file_sha256() == sidecar.index_file_sha256
            && recipe.manifest_file_sha256 == sidecar.manifest_file_sha256
            && recipe.index_file_sha256 == sidecar.index_file_sha256
            && registered.collection_recipe_sha256 == publication.recipe_sha256
            && registered.collection_stage_id == publication.stage_id
            && registered.collection_source_sha256 == publication.collection_source_sha256
            && registered.collection_runtime_library_sha256
                == publication.collection_runtime_library_sha256
            && registered.collection_model_semantics_sha256
                == publication.training_model_semantics_sha256
            && registered.collection_context_epoch == publication.collection_context_epoch,
        "R auxiliary sidecar changed original dataset/recipe/source/epoch pins",
    )?;
    dataset.revalidate(budget)?;
    let original_pc = current_pc.ok_or_else(|| {
        invalid("enabled historical R checkpoint requires its exact original P/C owner")
    })?;
    check(
        original_pc.registration.publication.sha256
            == publication.current_dataset_publication_sha256,
        "historical R pool cannot be rebound to another P/C publication",
    )?;
    // A shared R publication may have a different eligible pool at each old
    // P/C root. Bind a new immutable context; never alias the new run's pool.
    let dataset = dataset.for_current_pc(Arc::clone(original_pc), source_recipe, budget)?;
    let mut auxiliary = match (&publication.initialization_source, parent) {
        (RoventCheckpointInitializationSource::Initialize { .. }, None) => {
            RoventAuxiliaryLedger::new(
                dataset,
                recipe.clone(),
                BTreeSet::new(),
                BTreeSet::new(),
                0,
            )?
        }
        (RoventCheckpointInitializationSource::Resume { .. }, Some(parent)) => {
            let mut ledger = parent
                .r_auxiliary_ledger()
                .ok_or_else(|| invalid("resumed R checkpoint lacks its original prefix ledger"))?
                .clone();
            ledger.bind_current_dataset(dataset, recipe)?;
            ledger
        }
        (RoventCheckpointInitializationSource::Fork { .. }, Some(parent)) => {
            if let Some(ledger) = parent.r_auxiliary_ledger() {
                ledger.fork(dataset, recipe.clone())?
            } else {
                RoventAuxiliaryLedger::from_non_auxiliary_checkpoint(
                    dataset,
                    recipe.clone(),
                    parent,
                )?
            }
        }
        _ => {
            return Err(invalid(
                "R checkpoint initialization and parent chain disagree",
            ));
        }
    };
    auxiliary.set_execution_phase(
        &typed.phase,
        if matches!(
            publication.initialization_source,
            RoventCheckpointInitializationSource::Resume { .. }
        ) {
            parent
        } else {
            None
        },
    )?;
    check(
        !auxiliary.is_pending(),
        "historical R prefix starts inside accumulation",
    )?;
    Ok(Some(auxiliary))
}

fn validate_r_auxiliary_prefix(
    root: &Path,
    publication: &RoventCheckpointPublication,
    prefix: Option<&RoventAuxiliaryLedger>,
    budget: &RoventRecoveryReadBudget,
    held: &mut Vec<HeldHistoricalFile>,
) -> Result<(), ArenaError> {
    match (publication.r_auxiliary.enabled(), prefix) {
        (Some(sidecar), Some(prefix)) => {
            check(
                !prefix.is_pending(),
                "R checkpoint prefix has pending occurrences",
            )?;
            let recorded = read_json(root, &sidecar.parent_ledger_projection, budget, held)?;
            check(
                recorded == prefix.observation(),
                "R checkpoint prefix differs from the complete original update-chain replay",
            )
        }
        (None, None) => Ok(()),
        _ => Err(invalid("R checkpoint sidecar/prefix owner mismatch")),
    }
}

fn validate_documents(
    p: &RoventCheckpointPublication,
    recipe: &Value,
    membership: &Value,
    window: &Value,
) -> Result<String, ArenaError> {
    check(
        recipe["schema"] == "rz-rovent-generation-series/1"
            && digest_value("rz-rovent-generation-series/1", recipe)? == p.recipe_sha256
            && window["recipe_sha256"] == p.recipe_sha256
            && window["stage_id"] == p.stage_id
            && window["window_id"] == p.window_id
            && recipe["model_semantics_sha256"] == p.training_model_semantics_sha256,
        "historical original recipe/window/model/profile mismatch",
    )?;
    let profile = match window["schema"].as_str() {
        Some("rz-rovent-train-window/1") => {
            // Preserve original /1 bytes and its unseparated source/runtime
            // fields. Do not invent separate Python/ORT identity from them.
            check(
                window["model_semantics_sha256"] == p.training_model_semantics_sha256,
                "legacy historical training model semantics mismatch",
            )?;
            for key in ["source_sha256", "runtime_sha256"] {
                sha(&window[key])?;
            }
            training_profile_digest(window)?
        }
        Some(ROVENT_TRAIN_WINDOW_V2) => {
            let typed: RoventTrainWindowV2 =
                serde_json::from_value(window.clone()).map_err(|e| invalid(e.to_string()))?;
            typed.validate()?;
            check(
                typed.collection_source_sha256 == p.collection_source_sha256
                    && typed.collection_runtime_library_sha256
                        == p.collection_runtime_library_sha256
                    && typed.collection_model_epoch == p.collection_model_epoch
                    && typed.collection_context_epoch == p.collection_context_epoch
                    && typed.training_model_epoch == p.training_model_epoch
                    && typed.trainer_source_sha256 == p.trainer_source_sha256
                    && typed.trainer_python_executable_sha256 == p.trainer_python_executable_sha256
                    && typed.trainer_dependency_set_sha256 == p.trainer_dependency_set_sha256
                    && typed.training_model_semantics_sha256 == p.training_model_semantics_sha256,
                "explicit historical collection/trainer identities changed",
            )?;
            typed.profile_sha256()?
        }
        _ => return Err(invalid("historical training window domain unsupported")),
    };
    check(
        profile == p.training_profile_sha256,
        "historical training profile pin differs from original window codec",
    )?;
    let domain = membership["schema"]
        .as_str()
        .filter(|d| {
            matches!(
                *d,
                "rz-rovent-trainable-parameter-membership/1"
                    | "rz-rovent-trainable-parameter-membership/2"
            )
        })
        .ok_or_else(|| invalid("historical parameter membership domain unsupported"))?;
    check(
        digest_value(domain, membership)? == p.membership_sha256,
        "historical original parameter membership digest",
    )?;
    for key in ["max_updates", "max_groups", "max_samples"] {
        check(
            count(&window[key])? > 0,
            "historical original work budget is zero",
        )?;
    }
    let stages = recipe["stages"]
        .as_array()
        .filter(|s| !s.is_empty() && s.len() <= 64)
        .ok_or_else(|| invalid("historical bounded stage declarations"))?;
    let mut ids = BTreeSet::new();
    check(
        stages.iter().all(|s| {
            s["stage_id"]
                .as_str()
                .is_some_and(|id| generation_identifier(id) && ids.insert(id))
        }),
        "historical duplicate/invalid recipe stage",
    )?;
    let stage = stages
        .iter()
        .find(|s| s["stage_id"] == p.stage_id)
        .ok_or_else(|| invalid("historical recipe stage absent"))?;
    let limits: BTreeMap<String, u64> = serde_json::from_value(stage["role_updates"].clone())
        .map_err(|e| invalid(e.to_string()))?;
    role_counts(&limits)?;
    check(
        ROLES
            .iter()
            .all(|role| p.boundary.completed_updates[*role] <= limits[*role]),
        "historical snapshot exceeds stage updates",
    )?;
    Ok(profile)
}
/// Completed updates in the current stage and inherited ancestor updates.
type InitialRoleCounts = (BTreeMap<String, u64>, BTreeMap<String, u64>);

fn initial_counts(
    p: &RoventCheckpointPublication,
    recipe: &Value,
    parent: Option<&VerifiedRoventCheckpointSource>,
) -> Result<InitialRoleCounts, ArenaError> {
    let zero = || {
        ROLES
            .map(|r| (r.into(), 0))
            .into_iter()
            .collect::<BTreeMap<_, _>>()
    };
    match (&p.initialization_source, parent) {
        (RoventCheckpointInitializationSource::Initialize { .. }, None) => {
            let stage = recipe["stages"]
                .as_array()
                .and_then(|stages| stages.iter().find(|s| s["stage_id"] == p.stage_id))
                .ok_or_else(|| invalid("initialize recipe stage absent"))?;
            check(
                stage.get("parent_stage_id") == Some(&Value::Null),
                "initialize requires a recipe root stage",
            )?;
            check(
                p.boundary.ancestor_updates == zero(),
                "initialize cannot invent ancestor updates",
            )?;
            Ok((zero(), zero()))
        }
        (
            RoventCheckpointInitializationSource::Resume {
                tensor_checkpoint_sha256,
            },
            Some(parent),
        ) => {
            check(
                tensor_checkpoint_sha256 == &parent.tensor.pin.sha256
                    && p.stage_id == parent.publication.stage_id
                    && p.recipe_sha256 == parent.publication.recipe_sha256
                    && p.training_profile_sha256 == parent.publication.training_profile_sha256
                    && p.membership_sha256 == parent.publication.membership_sha256
                    && p.training_model_semantics_sha256
                        == parent.publication.training_model_semantics_sha256
                    && p.training_model_epoch == parent.publication.training_model_epoch
                    && p.boundary.ancestor_updates == parent.publication.boundary.ancestor_updates,
                "historical resume changed parent tensor/profile/stage/ancestry",
            )?;
            Ok((
                parent.publication.boundary.completed_updates.clone(),
                parent.publication.boundary.ancestor_updates.clone(),
            ))
        }
        (
            RoventCheckpointInitializationSource::Fork {
                tensor_checkpoint_sha256,
                parent_stage_id,
            },
            Some(parent),
        ) => {
            let stages = recipe["stages"]
                .as_array()
                .ok_or_else(|| invalid("fork recipe stages absent"))?;
            let stage = stages
                .iter()
                .find(|s| s["stage_id"] == p.stage_id)
                .ok_or_else(|| invalid("fork target stage absent"))?;
            let previous = stages
                .iter()
                .find(|s| s["stage_id"] == *parent_stage_id)
                .ok_or_else(|| invalid("fork parent stage absent"))?;
            let limits: BTreeMap<String, u64> =
                serde_json::from_value(previous["role_updates"].clone())
                    .map_err(|e| invalid(e.to_string()))?;
            role_counts(&limits)?;
            check(
                tensor_checkpoint_sha256 == &parent.tensor.pin.sha256
                    && parent_stage_id == &parent.publication.stage_id
                    && stage["parent_stage_id"] == *parent_stage_id
                    && p.recipe_sha256 == parent.publication.recipe_sha256
                    && parent.publication.boundary.completed_updates == limits
                    && p.training_model_semantics_sha256
                        == parent.publication.training_model_semantics_sha256,
                "historical fork needs an exact complete direct parent tensor/stage",
            )?;
            let mut ancestors = BTreeMap::new();
            for role in ROLES {
                ancestors.insert(
                    role.into(),
                    parent.publication.boundary.completed_updates[role]
                        .checked_add(parent.publication.boundary.ancestor_updates[role])
                        .ok_or_else(|| invalid("historical ancestor count overflow"))?,
                );
            }
            check(
                ancestors == p.boundary.ancestor_updates,
                "fork ancestor counters do not equal actual parent chain",
            )?;
            Ok((zero(), ancestors))
        }
        _ => Err(invalid(
            "historical initialization kind and parent chain disagree",
        )),
    }
}
type BoundaryReplay = (BTreeSet<String>, BTreeSet<String>, Option<Value>);
fn boundary_snapshot(
    ledger: &mut TrainerLedger,
    boundary: &RoventCheckpointBoundary,
) -> Result<Option<BoundaryReplay>, ArenaError> {
    if ledger.completed != boundary.completed_updates {
        return Ok(None);
    }
    // Checkpoint serialization itself spends work after its final update.
    // Insert the checkpoint's monotonic barrier before replaying later work;
    // update-to-update monotonicity alone would allow a cost regression here.
    ledger.observe_cost(&json!(boundary.snapshot_cost))?;
    Ok(Some((
        ledger.consumed_inputs.clone(),
        ledger.consumed_origins.clone(),
        ledger.updates.last().cloned(),
    )))
}
#[cfg(test)]
fn validate_update(
    p: &RoventCheckpointPublication,
    ledger: &mut TrainerLedger,
    receipt: &Value,
    index: usize,
) -> Result<(), ArenaError> {
    validate_update_with_r_auxiliary(p, ledger, receipt, index, None)
}
fn validate_update_with_r_auxiliary(
    p: &RoventCheckpointPublication,
    ledger: &mut TrainerLedger,
    receipt: &Value,
    index: usize,
    auxiliary: Option<&mut RoventAuxiliaryLedger>,
) -> Result<(), ArenaError> {
    let recovered = ledger.window["schema"] == ROVENT_TRAIN_WINDOW_V2;
    let (consume_schema, update_schema, event_schema) = if recovered {
        fields(
            receipt,
            &[
                "schema",
                "execution_id",
                "window_id",
                "sequence",
                "source_kind",
                "past_native_authority_reissued",
                "update",
            ],
        )?;
        check(
            receipt["schema"] == "rz-rovent-recovered-learner-completion/1"
                && receipt["source_kind"] == ROVENT_RECOVERY_SOURCE_KIND
                && receipt["past_native_authority_reissued"] == false,
            "historical recovered update domain/scope",
        )?;
        (
            "rz-rovent-trainer-consume/2",
            "rz-rovent-trainer-update/2",
            "rz-rovent-recovered-learner-update/1",
        )
    } else {
        check(
            ledger.window["schema"] == "rz-rovent-train-window/1",
            "historical update window domain",
        )?;
        fields(
            receipt,
            &[
                "schema",
                "execution_id",
                "window_id",
                "sequence",
                "original_native_lineage_verified",
                "update",
            ],
        )?;
        check(
            receipt["schema"] == ROVENT_ORIGINAL_UPDATE
                && receipt["original_native_lineage_verified"] == true,
            "historical original update domain/recorded scope",
        )?;
        (
            "rz-rovent-trainer-consume/1",
            "rz-rovent-trainer-update/1",
            "rz-rovent-original-learner-update/1",
        )
    };
    check(
        receipt["execution_id"] == p.execution_id
            && receipt["window_id"] == p.window_id
            && receipt["sequence"] == (index + 1) as u64,
        "historical update receipt chain identity/sequence",
    )?;
    let update = &receipt["update"];
    check(
        p.r_auxiliary.enabled().is_some() == auxiliary.is_some(),
        "enabled R update requires its retained dataset/ledger owner",
    )?;
    replay_update_body(
        ledger,
        update,
        consume_schema,
        update_schema,
        event_schema,
        auxiliary,
    )
}
pub(super) fn replay_update_body(
    ledger: &mut TrainerLedger,
    update: &Value,
    consume_schema: &str,
    update_schema: &str,
    event_schema: &str,
    mut auxiliary: Option<&mut RoventAuxiliaryLedger>,
) -> Result<(), ArenaError> {
    let origins = update["event"]["sample_origins"]
        .as_array()
        .filter(|v| v.len() == 8)
        .ok_or_else(|| invalid("historical optimizer update needs eight original microbatches"))?;
    for origin in origins {
        let id = sha(origin)?;
        let row = ledger.rows.get(id).ok_or_else(|| {
            invalid("historical update sample is outside retained dataset owners")
        })?;
        let consume = json!({"schema":consume_schema,"input_sha256":row.origin.input_sha256,
            "origin_sha256":id,"record_sha256":row.origin.record_sha256,"native_retirement_sha256":row.retirement_sha256,
            "head_raw_sha256":row.head_raw_sha256,"role":row.role,"split":"train"});
        ledger.consume_with_schema(&consume, consume_schema)?;
    }
    if let Some(auxiliary) = auxiliary.as_deref_mut() {
        auxiliary.validate_pc_origins(ledger)?;
        if update["event"]["role"] != "validator" {
            let projection = &update["event"]["r_auxiliary"];
            replay_r_auxiliary_update(ledger, auxiliary, projection)?;
        } else {
            check(
                !auxiliary.is_pending()
                    && update["event"]
                        .as_object()
                        .is_some_and(|event| !event.contains_key("r_auxiliary")),
                "V cannot consume or report R auxiliary optimization",
            )?;
        }
    }
    ledger.update_with_schema(update, update_schema, event_schema)?;
    if update["event"]["role"] != "validator"
        && let Some(auxiliary) = auxiliary.as_deref_mut()
    {
        auxiliary.commit_update()?;
    }
    if let Some(auxiliary) = auxiliary {
        auxiliary.validate_joint_retained(ledger)?;
    }
    Ok(())
}
pub(super) fn replay_r_auxiliary_update(
    ledger: &mut TrainerLedger,
    auxiliary: &mut RoventAuxiliaryLedger,
    projection: &Value,
) -> Result<(), ArenaError> {
    let pc_valid = valid_pc_rows(ledger)?;
    let occurrences = projection["occurrences"]
        .as_array()
        .filter(|rows| rows.len() == usize::from(auxiliary.recipe().rows_per_pc_update))
        .ok_or_else(|| invalid("R update needs its exact original occurrence count"))?;
    for occurrence in occurrences {
        check(
            ledger.samples_admitted < count(&ledger.window["max_samples"])?,
            "historical R samples exceed the shared PC/R sample credit",
        )?;
        let original = auxiliary
            .dataset()
            .original_auxiliary_evidence(sha(&occurrence["record_sha256"])?)?;
        auxiliary.consume(&original)?;
        ledger.samples_admitted = ledger
            .samples_admitted
            .checked_add(1)
            .ok_or_else(|| invalid("historical R sample credit overflow"))?;
    }
    auxiliary.validate_update(projection, pc_valid)
}
pub(super) fn set_seal(kind: &str, set: &BTreeSet<String>) -> Result<String, ArenaError> {
    check(
        matches!(kind, "input" | "origin") && set.len() <= MAX_TRAINER_ROWS,
        "historical retained set bounds",
    )?;
    let mut hash = Sha256::new();
    hash.update(b"rz-rovent-training-seal-set/1\0");
    hash.update(kind.as_bytes());
    hash.update([0]);
    hash.update((set.len() as u64).to_le_bytes());
    for item in set {
        check(
            generation_sha(item),
            "historical retained set contains invalid SHA",
        )?;
        let mut raw = [0_u8; 32];
        for (i, pair) in item.as_bytes().chunks_exact(2).enumerate() {
            let nibble = |v: u8| if v <= b'9' { v - b'0' } else { v - b'a' + 10 };
            raw[i] = (nibble(pair[0]) << 4) | nibble(pair[1]);
        }
        hash.update(raw);
    }
    Ok(format!("{:x}", hash.finalize()))
}
fn validate_boundary_sets(
    boundary: &RoventCheckpointBoundary,
    inputs: &BTreeSet<String>,
    origins: &BTreeSet<String>,
) -> Result<(), ArenaError> {
    check(
        boundary.retained_input_count == inputs.len() as u64
            && boundary.retained_origin_count == origins.len() as u64
            && boundary.retained_input_set_sha256 == set_seal("input", inputs)?
            && boundary.retained_origin_set_sha256 == set_seal("origin", origins)?,
        "historical checkpoint retained input/origin set mismatch",
    )
}
#[cfg(test)]
fn validate_retirement(
    p: &RoventCheckpointPublication,
    receipt: &Value,
    prefix_update: Option<&Value>,
    window_schema: &str,
) -> Result<(), ArenaError> {
    validate_retirement_with_r_auxiliary(p, receipt, prefix_update, window_schema, None)
}
fn validate_retirement_with_r_auxiliary(
    p: &RoventCheckpointPublication,
    receipt: &Value,
    prefix_update: Option<&Value>,
    window_schema: &str,
    auxiliary: Option<&RoventAuxiliaryLedger>,
) -> Result<(), ArenaError> {
    let checkpoint_schema = match window_schema {
        ROVENT_TRAIN_WINDOW_V2 => {
            fields(
                receipt,
                &[
                    "schema",
                    "execution_id",
                    "window_id",
                    "source_kind",
                    "past_native_authority_reissued",
                    "checkpoint",
                ],
            )?;
            check(
                receipt["schema"] == "rz-rovent-recovered-checkpoint-retirement/1"
                    && receipt["source_kind"] == ROVENT_RECOVERY_SOURCE_KIND
                    && receipt["past_native_authority_reissued"] == false,
                "historical recovered checkpoint domain/scope",
            )?;
            "rz-rovent-trainer-checkpoint/2"
        }
        "rz-rovent-train-window/1" => {
            fields(
                receipt,
                &["schema", "execution_id", "window_id", "checkpoint"],
            )?;
            check(
                receipt["schema"] == ROVENT_ORIGINAL_CHECKPOINT,
                "historical original checkpoint domain",
            )?;
            "rz-rovent-trainer-checkpoint/1"
        }
        _ => return Err(invalid("historical checkpoint window domain unsupported")),
    };
    check(
        receipt["execution_id"] == p.execution_id && receipt["window_id"] == p.window_id,
        "historical checkpoint retirement identity",
    )?;
    let state = &receipt["checkpoint"];
    let mut state_fields = vec![
        "schema",
        "checkpoint",
        "completed_updates",
        "sampler_state_sha256",
        "model_state_sha256",
        "optimizer_state_sha256",
        "scheduler_state_sha256",
        "rng_state_sha256",
        "cost",
        "pending_microbatches",
    ];
    if window_schema == ROVENT_TRAIN_WINDOW_V2 && p.r_auxiliary.enabled().is_some() {
        state_fields.push("r_auxiliary");
    }
    fields(state, &state_fields)?;
    match (&p.r_auxiliary, auxiliary) {
        (RoventRAuxiliaryPresence::Absent, None) => (),
        // Disabled RPC remains byte-for-byte on its original no-field path;
        // only window/publication retain absent versus explicit null.
        (RoventRAuxiliaryPresence::Null, None) => (),
        (RoventRAuxiliaryPresence::Enabled(_), Some(auxiliary)) => {
            auxiliary.validate_checkpoint_projection(&state["r_auxiliary"])?;
        }
        _ => {
            return Err(invalid(
                "R checkpoint retirement lacks its exact original prefix owner",
            ));
        }
    }
    fields(&state["checkpoint"], &["path", "sha256", "bytes"])?;
    check(
        state["schema"] == checkpoint_schema
            && state["checkpoint"]["sha256"] == p.tensor.sha256
            && state["checkpoint"]["bytes"] == p.tensor.bytes
            && state["pending_microbatches"] == 0,
        "historical checkpoint retirement tensor/accumulation boundary",
    )?;
    check(
        state["checkpoint"]["path"]
            .as_str()
            .is_some_and(bounded_text),
        "historical checkpoint path must remain descriptive text",
    )?;
    let boundary = p.boundary.state_value();
    for key in STATE_HASHES
        .into_iter()
        .chain(["completed_updates", "cost", "pending_microbatches"])
    {
        check(
            state[key] == boundary[key],
            "checkpoint retirement differs from declared actual snapshot",
        )?;
    }
    if let Some(update) = prefix_update {
        for key in [
            "model_state_sha256",
            "optimizer_state_sha256",
            "scheduler_state_sha256",
            "rng_state_sha256",
        ] {
            check(
                update[key] == state[key],
                "checkpoint state differs from its final original update",
            )?;
        }
        for key in COST_FIELDS {
            check(
                count(&state["cost"][key])? >= count(&update["cost"][key])?,
                "checkpoint snapshot cost regressed",
            )?;
        }
    }
    Ok(())
}
fn validate_terminal(
    p: &RoventCheckpointPublication,
    terminal: &Value,
    ledger: &TrainerLedger,
    closure: &RoventOriginalTrainingClosure,
) -> Result<(), ArenaError> {
    if closure.schema == ROVENT_CHECKPOINT_BOUNDARY_CLOSURE {
        fields(
            terminal,
            &[
                "schema",
                "execution_id",
                "window_id",
                "completed_updates",
                "physical_completion",
                "arena_eligible",
                "status",
                "checkpoint_retirement_sha256",
            ],
        )?;
        check(
            terminal["schema"] == ROVENT_CHECKPOINT_BOUNDARY_REPORT
                && terminal["execution_id"] == p.execution_id
                && terminal["window_id"] == p.window_id
                && terminal["completed_updates"] == json!(p.boundary.completed_updates)
                && terminal["completed_updates"] == json!(ledger.completed)
                && terminal["physical_completion"] == "unknown"
                && terminal["arena_eligible"] == false
                && terminal["status"] == "checkpoint_boundary_only"
                && terminal["checkpoint_retirement_sha256"] == p.checkpoint_retirement.sha256,
            "unclosed boundary report cannot claim terminal success or omit actual checkpoint retirement",
        )?;
        check(
            matches!(&p.recorded_tail_cost,RoventRecordedTailCost::Unknown{reason,..}if reason=="original_window_unclosed_after_boundary"),
            "unclosed checkpoint boundary must preserve unknown post-snapshot cost",
        )?;
        return Ok(());
    }
    check(
        terminal.is_object()
            && terminal["schema"]
                == if ledger.window["schema"] == ROVENT_TRAIN_WINDOW_V2 {
                    "rz-rovent-recovered-trainer-result/1"
                } else {
                    ROVENT_TRAINER_RESULT
                }
            && terminal["execution_id"] == p.execution_id
            && terminal["window_id"] == p.window_id
            && matches!(terminal["status"].as_str(), Some("completed" | "failed"))
            && terminal["arena_eligible"] == false,
        "historical terminal receipt identity/scope",
    )?;
    let completed: BTreeMap<String, u64> =
        serde_json::from_value(terminal["completed_updates"].clone())
            .map_err(|e| invalid(e.to_string()))?;
    role_counts(&completed)?;
    check(
        completed == ledger.completed,
        "historical terminal omitted/added an original update receipt",
    )
}
fn validate_closure(
    p: &RoventCheckpointPublication,
    closure: &RoventOriginalTrainingClosure,
    terminal: &Value,
) -> Result<RoventHistoricalClosureClassification, ArenaError> {
    let child = &closure.child_identity;
    let boot = child.boot_id.as_bytes();
    let boot_shape = boot.len() == 36
        && boot.iter().enumerate().all(|(i, b)| {
            if [8, 13, 18, 23].contains(&i) {
                *b == b'-'
            } else {
                b.is_ascii_digit() || (b'a'..=b'f').contains(b)
            }
        })
        && boot.iter().any(|b| matches!(*b,b'1'..=b'9'|b'a'..=b'f'));
    let cgroup = Path::new(&child.cgroup_path);
    check(
        matches!(
            closure.schema.as_str(),
            ROVENT_ORIGINAL_TRAINING_CLOSURE | ROVENT_CHECKPOINT_BOUNDARY_CLOSURE
        ) && closure.execution_id == p.execution_id
            && closure.window_id == p.window_id
            && closure.terminal_receipt_sha256 == p.terminal_receipt.sha256
            && closure.terminal_receipt_bytes == p.terminal_receipt.bytes
            && child.pid > 0
            && child.start_ticks > 0
            && child.cgroup_device > 0
            && child.cgroup_inode > 0
            && boot_shape
            && bounded_text(&child.cgroup_path)
            && child.cgroup_path.starts_with('/')
            && !child.cgroup_path.contains('\\')
            && cgroup
                .components()
                .all(|part| matches!(part, Component::RootDir | Component::Normal(_)))
            && !child
                .cgroup_path
                .split('/')
                .any(|part| matches!(part, "." | "..")),
        "original recorded closure identity/terminal/child/cgroup mismatch",
    )?;
    if closure.schema == ROVENT_CHECKPOINT_BOUNDARY_CLOSURE {
        check(
            closure.process_receipt.is_none()
                && closure.child_reap == RoventHistoricalChildReap::Unresolved
                && closure.physical_completion == RoventHistoricalPhysicalCompletion::Unknown,
            "boundary-only closure must not fabricate a process receipt or confirmed physical completion",
        )?;
        return Ok(RoventHistoricalClosureClassification::Unresolved);
    }
    let receipt = closure
        .process_receipt
        .as_ref()
        .ok_or_else(|| invalid("original final closure requires an actual process receipt"))?;
    check(
        receipt.pid == child.pid,
        "original final closure PID differs from original peer identity",
    )?;
    if closure.physical_completion == RoventHistoricalPhysicalCompletion::Confirmed {
        check(
            terminal["physical_completion"] == "confirmed",
            "recorded physical closure differs from original terminal receipt",
        )?;
    }
    let exited = receipt.exit_code.is_some() ^ receipt.exit_signal.is_some();
    let closed = closure.child_reap == RoventHistoricalChildReap::OwnedChildReaped
        && receipt.group_cleanup == crate::process::CleanupStatus::Gone
        && exited
        && closure.physical_completion == RoventHistoricalPhysicalCompletion::Confirmed;
    // A pinned report describes a past event only. Unresolved evidence must
    // pass a separate current-OS reclamation owner before new execution.
    Ok(if closed {
        RoventHistoricalClosureClassification::RecordedClosed
    } else {
        RoventHistoricalClosureClassification::Unresolved
    })
}
fn parse_recorded_closure(value: Value) -> Result<RoventOriginalTrainingClosure, ArenaError> {
    fields(
        &value,
        &[
            "schema",
            "execution_id",
            "window_id",
            "terminal_receipt_sha256",
            "terminal_receipt_bytes",
            "process_receipt",
            "child_identity",
            "child_reap",
            "physical_completion",
        ],
    )?;
    serde_json::from_value(value).map_err(|e| invalid(e.to_string()))
}
fn validate_tail(
    root: &Path,
    p: &RoventCheckpointPublication,
    budget: &RoventRecoveryReadBudget,
    held: &mut Vec<HeldHistoricalFile>,
    ledger: &TrainerLedger,
) -> Result<(), ArenaError> {
    match &p.recorded_tail_cost {
        RoventRecordedTailCost::Unknown { evidence, .. } => {
            if let Some(file) = evidence {
                let (owner, _) =
                    budget.read(&archive_pin(root, file)?, MAX_METADATA_BYTES, false)?;
                held.push(owner);
            }
            Ok(())
        }
        RoventRecordedTailCost::Known { delta, evidence } => {
            let value = read_json(root, evidence, budget, held)?;
            validate_tail_value(p, delta, &value, ledger.last_cost.as_ref())
        }
    }
}
fn validate_tail_value(
    p: &RoventCheckpointPublication,
    delta: &BTreeMap<String, u64>,
    value: &Value,
    last_cost: Option<&Value>,
) -> Result<(), ArenaError> {
    fields(
        value,
        &[
            "schema",
            "execution_id",
            "window_id",
            "snapshot_cost",
            "end_cost",
            "delta",
        ],
    )?;
    cost_map(delta)?;
    check(
        value["schema"] == ROVENT_RECORDED_TAIL_COST
            && value["execution_id"] == p.execution_id
            && value["window_id"] == p.window_id
            && value["snapshot_cost"] == json!(p.boundary.snapshot_cost)
            && value["delta"] == json!(delta),
        "known tail evidence changed snapshot/window/delta",
    )?;
    validate_cost(&value["end_cost"])?;
    for key in COST_FIELDS {
        check(
            count(&value["end_cost"][key])?.checked_sub(p.boundary.snapshot_cost[key])
                == Some(delta[key]),
            "known tail cost is not an exact nonrefunded difference",
        )?;
        if let Some(previous) = last_cost {
            check(
                count(&value["end_cost"][key])? >= count(&previous[key])?,
                "known tail end cost omits recorded work",
            )?;
        }
    }
    Ok(())
}

/// F's projection is plain comparison input until an original pinned scratch
/// execution owner is integrated. This DTO has no successful-run constructor.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoventHistoricalTensorProjection {
    pub schema: String,
    pub status: String,
    pub receipt_pin: RoventHistoricalDigestPin,
    pub historical_execution_id: String,
    pub historical_window_id: String,
    pub historical_checkpoint_pin: RoventTrainingFilePin,
    pub recovered_checkpoint_pin: RoventTrainingFilePin,
    pub sampler_state_sha256: String,
    pub model_state_sha256: String,
    pub optimizer_state_sha256: String,
    pub scheduler_state_sha256: String,
    pub rng_state_sha256: String,
    pub completed_updates: BTreeMap<String, u64>,
    pub snapshot_cost: BTreeMap<String, u64>,
    pub pending_microbatches: u64,
    pub past_native_authority_reissued: bool,
    pub execution_authorized: bool,
    pub physical_completion_scope: String,
    #[serde(default, skip_serializing_if = "RoventRAuxiliaryPresence::is_absent")]
    pub r_auxiliary: RoventRAuxiliaryPresence<Value>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoventHistoricalDigestPin {
    pub sha256: String,
    pub bytes: u64,
}
impl VerifiedRoventCheckpointSource {
    pub fn compare_tensor_projection(
        &self,
        projection: &RoventHistoricalTensorProjection,
    ) -> Result<Value, ArenaError> {
        check(
            projection.schema == ROVENT_HISTORICAL_TENSOR_PROJECTION
                && projection.status == "historical_integrity_only"
                && projection.receipt_pin.sha256 == self.publication.checkpoint_retirement.sha256
                && projection.receipt_pin.bytes == self.publication.checkpoint_retirement.bytes
                && projection.historical_execution_id == self.publication.execution_id
                && projection.historical_window_id == self.publication.window_id
                && projection.historical_checkpoint_pin.sha256 == self.tensor.pin.sha256
                && projection.historical_checkpoint_pin.bytes == self.tensor.pin.bytes
                && projection.recovered_checkpoint_pin == self.tensor.pin
                && projection.pending_microbatches == 0
                && !projection.past_native_authority_reissued
                && !projection.execution_authorized
                && projection.physical_completion_scope
                    == "checkpoint_does_not_prove_old_run_closure",
            "historical scratch projection identity/bytes/scope mismatch",
        )?;
        let actual = serde_json::to_value(projection).map_err(|e| invalid(e.to_string()))?;
        let expected = self.publication.boundary.state_value();
        for key in STATE_HASHES {
            check(
                actual[key] == expected[key],
                "scratch projection state differs from historical checkpoint",
            )?;
        }
        check(
            projection.completed_updates == self.publication.boundary.completed_updates
                && projection.snapshot_cost == self.publication.boundary.snapshot_cost,
            "scratch projection counters/cost mismatch",
        )?;
        match (self.r_auxiliary_ledger(), projection.r_auxiliary.enabled()) {
            (Some(auxiliary), Some(actual)) => {
                auxiliary.validate_checkpoint_projection(actual)?;
                check(
                    Some(actual) == self.r_auxiliary_checkpoint_projection(),
                    "scratch R pool/sampler projection differs from exact original retirement",
                )?;
            }
            (None, None) if projection.r_auxiliary.is_absent() => (),
            _ => {
                return Err(invalid(
                    "scratch projection changed enabled R checkpoint presence",
                ));
            }
        }
        Ok(
            json!({"schema":"rz-rovent-historical-checkpoint-projection-comparison/1",
            "publication_sha256":self.publication_pin.sha256,"tensor_sha256":self.tensor.pin.sha256,
            "metadata_matches":true,"tensor_state_validated":false,"freeze_nnuec_derived_validation":"requires_pinned_scratch_owner",
            "past_native_authority_reissued":false,"past_physical_completion_reissued":false,"execution_authorized":false}),
        )
    }
}

#[cfg(test)]
#[path = "checkpoint/tests.rs"]
mod tests;

use sha2::{Digest as _, Sha256};
