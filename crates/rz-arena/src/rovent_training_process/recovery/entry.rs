//! Actual product admission from a pinned CLI request. Inspection stops before
//! OS resource creation or child dispatch. Execution consumes historical data
//! owners, a separately admitted old-run scope, and the concrete current parent
//! runtime. Failure after dispatch retains the runtime's original ownership.
use super::*;
use cap_fs_ext::{DirExt, FollowSymlinks, OpenOptionsFollowExt};
use cap_std::fs::{Dir, OpenOptionsExt};
use cli::{RoventTrainerCliAction, RoventTrainerCliAdmissionRequest, RoventTrainerCliTotalQuota};
use std::os::unix::fs::MetadataExt;

/// Read-only inspection custody. Its public observation grants no execution
/// authority; private original descriptors and data owners stay retained for
/// this result's lifetime without exposing internal FD transport types.
pub struct RoventRecoveredTrainingInspection {
    observation: Value,
    _retained_registration: HeldHistoricalFile,
    _retained_metadata: Vec<HeldHistoricalFile>,
    _retained_dataset: Arc<VerifiedRoventDataset>,
    _retained_r_datasets: Vec<Arc<VerifiedRoventRAuxiliaryDataset>>,
    _retained_checkpoint: Option<Arc<VerifiedRoventCheckpointSource>>,
    _retained_budget: Arc<RoventRecoveryReadBudget>,
}
impl RoventRecoveredTrainingInspection {
    pub fn observation(&self) -> &Value {
        &self.observation
    }
}

// One execution returns one outcome retaining its original verified owner and
// descriptors. Preserve this public by-value custody API on success and failure;
// enum sizing alone must not change how callers receive the original owner.
#[allow(clippy::large_enum_variant)]
pub enum RoventRecoveredTrainingOutcome {
    Inspected(RoventRecoveredTrainingInspection),
    Completed {
        owner: VerifiedRoventRecoveredTrainingWindow,
        checkpoint_archive: Option<PublishedRoventCheckpointArchive>,
    },
    Failed(Box<RoventRecoveredTrainingFailure>),
    ResourcePreparationFailed(Box<RoventTrainingResourceFailure>),
    ResourceAdmissionFailed {
        error: ArenaError,
        resources: Arc<RoventTrainingResources>,
        read_budget: Arc<RoventRecoveryReadBudget>,
    },
    CheckpointPublicationFailed {
        owner: VerifiedRoventRecoveredTrainingWindow,
        failure: Box<RoventCheckpointArchiveFailure>,
    },
    CheckpointPublicationAdmissionFailed {
        owner: VerifiedRoventRecoveredTrainingWindow,
        error: ArenaError,
    },
}
impl RoventRecoveredTrainingOutcome {
    pub fn success(&self) -> bool {
        !matches!(
            self,
            Self::Failed(_)
                | Self::ResourcePreparationFailed(_)
                | Self::ResourceAdmissionFailed { .. }
                | Self::CheckpointPublicationFailed { .. }
                | Self::CheckpointPublicationAdmissionFailed { .. }
        )
    }
    pub fn observation(&self) -> Value {
        match self {
            Self::Inspected(owner) => owner.observation().clone(),
            Self::Completed {
                owner,
                checkpoint_archive,
            } => {
                let mut observation = owner.observation();
                observation["checkpoint_archive"] = checkpoint_archive
                    .as_ref()
                    .map(|archive| archive.observation())
                    .into();
                observation
            }
            Self::Failed(owner) => owner.observation(),
            Self::ResourcePreparationFailed(owner) => owner.observation(),
            Self::ResourceAdmissionFailed {
                error,
                resources,
                read_budget,
            } => json!({"schema":"rz-rovent-recovered-trainer-admission-failure/1",
                "status":"failed","error":error.to_string(),
                "read_accounting":read_budget.observation(),
                "resource_owner_retained":true,
                "resource_policy":resources.policy(),
                "child_started":false,"automatic_retry":false}),
            Self::CheckpointPublicationFailed { owner, failure } => {
                json!({"schema":"rz-rovent-recovered-checkpoint-publication-failure/1",
                "status":"failed","training_window":owner.observation(),"archive":failure.observation(),
                "actual_training_owner_retained":true,"automatic_retry":false})
            }
            Self::CheckpointPublicationAdmissionFailed { owner, error } => {
                json!({"schema":"rz-rovent-recovered-checkpoint-publication-failure/1",
                "status":"failed","training_window":owner.observation(),"archive_admission_error":error.to_string(),
                "actual_training_owner_retained":true,"automatic_retry":false})
            }
        }
    }
}

pub fn run_rovent_recovered_from_cli(
    request: RoventTrainerCliAdmissionRequest,
    cancel: Arc<AtomicBool>,
) -> Result<RoventRecoveredTrainingOutcome, ArenaError> {
    request.ensure_admission_open()?;
    let command = request.command();
    let budget = Arc::new(RoventRecoveryReadBudget::new(
        command.scope.max_read_bytes,
        request.whole_deadline(),
        Arc::clone(&cancel),
    )?);
    let (registration_file, raw) = budget.read(&command.registration, MAX_METADATA_BYTES, true)?;
    let registration: RoventRecoveredTrainingRegistration = decode(&raw)?;
    registration.validate()?;
    join_cli_scope(&request, &registration)?;
    let resources = if command.action == RoventTrainerCliAction::Execute {
        let delegated =
            super::reclamation::open_recovery_directory(&command.scope.delegated_cgroup)?;
        let owner = match RoventTrainingResources::prepare(
            delegated,
            &command.scope.delegated_cgroup,
            &command.scope.cgroup_leaf,
            registration.resources_policy.clone(),
        ) {
            Ok(owner) => Arc::new(owner),
            Err(failure) => {
                return Ok(RoventRecoveredTrainingOutcome::ResourcePreparationFailed(
                    failure,
                ));
            }
        };
        Some(owner)
    } else {
        None
    };
    let retained_resources = resources.as_ref().map(Arc::clone);
    let retained_budget = Arc::clone(&budget);
    let result = run_after_resource_preparation(
        request,
        registration_file,
        registration,
        budget,
        resources,
        cancel,
    );
    match (result, retained_resources) {
        (Err(error), Some(resources)) => {
            Ok(RoventRecoveredTrainingOutcome::ResourceAdmissionFailed {
                error,
                resources,
                read_budget: retained_budget,
            })
        }
        (result, _) => result,
    }
}

fn run_after_resource_preparation(
    request: RoventTrainerCliAdmissionRequest,
    registration_file: HeldHistoricalFile,
    registration: RoventRecoveredTrainingRegistration,
    budget: Arc<RoventRecoveryReadBudget>,
    resources: Option<Arc<RoventTrainingResources>>,
    cancel: Arc<AtomicBool>,
) -> Result<RoventRecoveredTrainingOutcome, ArenaError> {
    let command = request.command();
    if let Some(owner) = &resources {
        let total = owner.observe_total_parent_scope()?;
        check(
            !registration.require_total_quota || total.verified,
            "strict aggregate scope must include this parent before historical corpus/Rules admission",
        )?;
    }
    let (window_file, window_raw) =
        budget.read(&registration.window_spec, MAX_METADATA_BYTES, true)?;
    let window: RoventTrainWindowV2 = decode(&window_raw)?;
    registration.join_window(&window)?;
    let (recipe_file, recipe_raw) = budget.read(&registration.recipe, MAX_METADATA_BYTES, true)?;
    let recipe: Value = decode(&recipe_raw)?;
    check(
        recipe["schema"] == "rz-rovent-generation-series/1"
            && digest_value("rz-rovent-generation-series/1", &recipe)? == window.recipe_sha256
            && recipe["source_sha256"] == window.collection_source_sha256
            && recipe["model_semantics_sha256"] == window.training_model_semantics_sha256,
        "new window differs from the exact original collection recipe or training architecture",
    )?;
    let datasets = RegisteredDatasetGraph::new(&registration, &budget)?;
    let dataset = Arc::clone(&datasets.current);
    let r_datasets =
        RegisteredRAuxiliaryDatasetGraph::new(&registration, &datasets, &recipe, &budget)?;
    let live_r_dataset = r_datasets
        .current
        .as_ref()
        .map(|owner| owner.for_current_pc(Arc::clone(&dataset), &recipe, &budget))
        .transpose()?;
    let checkpoint_registration = match &registration.initialization_source {
        RoventTrainingInputSource::Initialize { .. } => None,
        RoventTrainingInputSource::Resume { checkpoint }
        | RoventTrainingInputSource::Fork { checkpoint, .. } => Some(checkpoint),
    };
    let checkpoint = checkpoint_registration
        .map(|source| {
            let (_, raw) = budget.read(&source.publication, MAX_METADATA_BYTES, true)?;
            let publication: RoventCheckpointPublication = decode(&raw)?;
            publication.validate()?;
            let current =
                datasets.by_publication(&publication.current_dataset_publication_sha256)?;
            let ancestors = publication
                .ancestor_dataset_publication_sha256
                .iter()
                .map(|sha| datasets.by_publication(sha))
                .collect::<Result<Vec<_>, _>>()?;
            VerifiedRoventCheckpointSource::open_with_r_auxiliary(
                &source.root,
                source.publication.clone(),
                current,
                ancestors,
                r_datasets.all(),
                &budget,
            )
            .map(Arc::new)
        })
        .transpose()?;
    // This publication's separated identities are required for actual new
    // windows; ambiguous old /1 reports remain inspectable historical bytes.
    if command.action == RoventTrainerCliAction::Execute
        && let Some(source) = &checkpoint
    {
        check(
            source.original_window_schema() == ROVENT_TRAIN_WINDOW_V2,
            "legacy unseparated checkpoint identities are metadata-only; explicit source recovery is required",
        )?;
        let previous = source.publication();
        check(
            previous.trainer_source_sha256 == registration.trainer_source_sha256
                && previous.trainer_python_executable_sha256
                    == registration.trainer_python_executable.sha256
                && previous.trainer_dependency_set_sha256
                    == registration.trainer_dependency_set_sha256
                && previous.training_model_semantics_sha256
                    == window.training_model_semantics_sha256,
            "checkpoint changed the explicit trainer/runtime/model profile",
        )?;
    }
    request.ensure_admission_open()?;
    if command.action == RoventTrainerCliAction::Inspect {
        return Ok(RoventRecoveredTrainingOutcome::Inspected(
            RoventRecoveredTrainingInspection {
                observation: json!({"schema":"rz-rovent-recovered-trainer-inspection/1",
                "execution_id":registration.execution_id,"source_kind":ROVENT_RECOVERY_SOURCE_KIND,
                "dataset":dataset.observation(),"checkpoint":checkpoint.as_ref().map(|owner|owner.observation()),
                "r_auxiliary_dataset":live_r_dataset.as_ref().map(|owner|owner.observation()),
                "window_profile_sha256":window.profile_sha256()?,"read_budget":budget.observation(),
                "resources_applied":false,"resource_scope":"inspection_read_and_time_budget_only_not_whole_cpu_memory_enforcement",
                "past_native_authority_reissued":false,"past_physical_completion_reissued":false,
                "execution_authorized":false,"child_started":false}),
                _retained_registration: registration_file,
                _retained_metadata: vec![window_file, recipe_file],
                _retained_dataset: dataset,
                _retained_r_datasets: r_datasets.all().into_iter().chain(live_r_dataset).collect(),
                _retained_checkpoint: checkpoint,
                _retained_budget: budget,
            },
        ));
    }
    let reclamation = match (&checkpoint, checkpoint_registration) {
        (Some(source), Some(requested)) => Some(VerifiedRoventOldRunReclamation::admit(
            Arc::clone(source),
            &requested.reclamation,
            Arc::clone(&budget),
        )?),
        (None, None) => None,
        _ => {
            return Err(invalid(
                "checkpoint source/reclamation pairing is inconsistent",
            ));
        }
    };
    let (interpreter, _) = budget.read(&registration.trainer_python_executable, 2 << 30, false)?;
    request.ensure_admission_open()?;
    let resources =
        resources.ok_or_else(|| invalid("execute resource owner missing after full admission"))?;
    let total = resources.observe_total_parent_scope()?;
    check(
        !registration.require_total_quota || total.verified,
        "strict total CPU/memory scope is unknown or excludes the actual parent",
    )?;
    request.ensure_admission_open()?;
    let launch = RoventRecoveredTrainingLaunch {
        interpreter: interpreter.file,
        interpreter_pin: registration.trainer_python_executable.clone(),
        interpreter_version: registration.trainer_python_version.clone(),
        registration: registration_file.file,
        registration_pin: command.registration.clone(),
        dependency_directories: registration.dependency_directories.clone(),
        started: request.original_started(),
        execution_deadline: request.execution_deadline(),
        whole_deadline: request.whole_deadline(),
        limits: ProcessLimits {
            wall_ms: command.scope.wall_ms,
            shutdown_grace_ms: command.scope.cleanup_ms,
            max_output_bytes: command.scope.max_stdio_bytes,
            max_child_processes: 1,
        },
        resources,
        read_budget: budget,
    };
    // In particular, do not map a post-spawn failure to ArenaError and discard
    // its original process/cgroup/IPC/checkpoint custody.
    Ok(
        match run_rovent_training_recovered(
            launch,
            dataset,
            checkpoint,
            reclamation,
            live_r_dataset,
            cancel,
        ) {
            Ok(owner) => publish_completed(owner),
            Err(failure) => RoventRecoveredTrainingOutcome::Failed(failure),
        },
    )
}

fn publish_completed(
    owner: VerifiedRoventRecoveredTrainingWindow,
) -> RoventRecoveredTrainingOutcome {
    if owner.registration.action == RoventRecoveredTrainingAction::Export
        || owner.journal.checkpoint_states.is_empty()
    {
        return RoventRecoveredTrainingOutcome::Completed {
            owner,
            checkpoint_archive: None,
        };
    }
    let prepare = (|| -> Result<RoventCheckpointArchiveTarget, ArenaError> {
        let mut visited = BTreeSet::new();
        let mut entries = 0usize;
        let root = Dir::from_std_file(
            owner
                .held
                .output
                .try_clone()
                .map_err(|_| invalid("completed output descriptor clone"))?,
        );
        let used = existing_output_bytes(
            &root,
            0,
            &mut entries,
            &mut visited,
            &owner.launch.read_budget,
        )?;
        let available = owner
            .registration
            .max_output_bytes
            .checked_sub(used)
            .and_then(|left| left.checked_sub(owner.held.control_written))
            .filter(|left| *left > 0)
            .ok_or_else(|| {
                invalid("completed trainer exhausted its original total artifact budget")
            })?;
        Ok(RoventCheckpointArchiveTarget {
            parent_directory: owner
                .held
                .output
                .try_clone()
                .map_err(|_| invalid("original archive parent descriptor"))?,
            parent_path: owner.registration.output_root.clone(),
            leaf_name: "checkpoint-final".into(),
            maximum_output_bytes: available,
            deadline: owner.launch.whole_deadline,
            cancel: Arc::clone(&owner.launch.read_budget.cancel),
            read_budget: Arc::clone(&owner.launch.read_budget),
        })
    })();
    let target = match prepare {
        Ok(target) => target,
        Err(error) => {
            return RoventRecoveredTrainingOutcome::CheckpointPublicationAdmissionFailed {
                owner,
                error,
            };
        }
    };
    let selected = owner.journal.checkpoint_states.len() - 1;
    match publish_rovent_training_checkpoint(&owner, selected, target) {
        Ok(archive) => RoventRecoveredTrainingOutcome::Completed {
            owner,
            checkpoint_archive: Some(archive),
        },
        Err(failure) => {
            RoventRecoveredTrainingOutcome::CheckpointPublicationFailed { owner, failure }
        }
    }
}
// Post-join inventory is separate from peak/disk-quota claims. The caller holds
// the original output root, traverses no links, and reserves metadata read
// credit. Content copies are independently checked by their retained FD/hash.
fn existing_output_bytes(
    directory: &Dir,
    depth: usize,
    entries: &mut usize,
    visited: &mut BTreeSet<(u64, u64)>,
    budget: &RoventRecoveryReadBudget,
) -> Result<u64, ArenaError> {
    check(depth <= 32, "completed output tree depth exceeded")?;
    let descriptor = crate::pals_collect_v3::training_process::directory_file(directory)?;
    let metadata = descriptor
        .metadata()
        .map_err(|_| invalid("completed output directory identity"))?;
    check(
        visited.insert((metadata.dev(), metadata.ino())),
        "completed output tree repeats a directory",
    )?;
    let mut total = 0u64;
    for entry in directory
        .entries()
        .map_err(|_| invalid("completed output enumeration"))?
    {
        *entries = entries
            .checked_add(1)
            .ok_or_else(|| invalid("completed output count overflow"))?;
        check(
            *entries <= 32768,
            "completed output finite entry bound exceeded",
        )?;
        budget.reserve(256)?;
        let entry = entry.map_err(|_| invalid("completed output directory entry"))?;
        let name = entry.file_name();
        let kind = entry
            .file_type()
            .map_err(|_| invalid("completed output entry type"))?;
        let bytes = if kind.is_dir() {
            existing_output_bytes(
                &directory
                    .open_dir_nofollow(&name)
                    .map_err(|_| invalid("completed nofollow directory"))?,
                depth + 1,
                entries,
                visited,
                budget,
            )?
        } else {
            check(
                kind.is_file() && !kind.is_symlink(),
                "completed output contains a link or special file",
            )?;
            let mut options = cap_std::fs::OpenOptions::new();
            options
                .read(true)
                .follow(FollowSymlinks::No)
                .custom_flags(libc::O_CLOEXEC | libc::O_NONBLOCK);
            let file = directory
                .open_with(&name, &options)
                .map(|file| file.into_std())
                .map_err(|_| invalid("completed nofollow file"))?;
            let metadata = file
                .metadata()
                .map_err(|_| invalid("completed file readback"))?;
            check(
                metadata.is_file() && metadata.nlink() == 1,
                "completed file identity is not one owned regular file",
            )?;
            metadata.len()
        };
        total = total
            .checked_add(bytes)
            .ok_or_else(|| invalid("completed output byte sum overflow"))?;
    }
    Ok(total)
}

fn join_cli_scope(
    request: &RoventTrainerCliAdmissionRequest,
    registration: &RoventRecoveredTrainingRegistration,
) -> Result<(), ArenaError> {
    let scope = &request.command().scope;
    let policy = &registration.resources_policy;
    check(
        scope.max_read_bytes == registration.max_read_bytes
            && scope.max_output_bytes == registration.max_output_bytes
            && scope.cpu_quota_millicpus == policy.cpu_quota_millicpus
            && scope.memory_high_bytes == policy.memory_high_bytes
            && scope.memory_max_bytes == policy.memory_max_bytes
            && scope.pids_max == policy.pids_max
            && scope.affinity == policy.affinity
            && scope.device == policy.device
            && (scope.total_quota == RoventTrainerCliTotalQuota::Strict)
                == registration.require_total_quota,
        "CLI request must match every pinned resource/read/output bound",
    )
}

struct DatasetProbe {
    registration: RoventHistoricalDatasetRegistration,
    family: RoventPcRecordFamily,
    index_sha256: String,
    ancestor_indices: Vec<String>,
}

struct RDatasetProbe {
    registration: RoventHistoricalRAuxiliaryDatasetRegistration,
    manifest: rz_experiments::rovent_r_readout::RoventRReadoutManifest,
}
struct RegisteredRAuxiliaryDatasetGraph {
    current: Option<Arc<VerifiedRoventRAuxiliaryDataset>>,
    ancestors: Vec<Arc<VerifiedRoventRAuxiliaryDataset>>,
}
impl RegisteredRAuxiliaryDatasetGraph {
    fn all(&self) -> Vec<Arc<VerifiedRoventRAuxiliaryDataset>> {
        self.current
            .iter()
            .chain(&self.ancestors)
            .map(Arc::clone)
            .collect()
    }
    fn new(
        registration: &RoventRecoveredTrainingRegistration,
        pc: &RegisteredDatasetGraph,
        recipe: &Value,
        budget: &RoventRecoveryReadBudget,
    ) -> Result<Self, ArenaError> {
        let current = registration.r_auxiliary_dataset.enabled();
        let ancestors = registration
            .ancestor_r_auxiliary_datasets
            .enabled()
            .map(Vec::as_slice)
            .unwrap_or(&[]);
        if current.is_none() && ancestors.is_empty() {
            return Ok(Self {
                current: None,
                ancestors: Vec::new(),
            });
        }
        let mut probes = BTreeMap::new();
        let mut current_index = None;
        for request in ancestors.iter().chain(current) {
            request.validate()?;
            let (_, raw) = budget.read(&request.publication, MAX_METADATA_BYTES, true)?;
            let publication: crate::rovent_collect::RoventRCorpusPublication = decode(&raw)?;
            publication.validate()?;
            let (_, raw) = budget.read(
                &archive_pin(&request.root, &publication.manifest)?,
                MAX_METADATA_BYTES,
                true,
            )?;
            let manifest: rz_experiments::rovent_r_readout::RoventRReadoutManifest = decode(&raw)?;
            check(
                manifest
                    .sha256()
                    .map_err(|error| invalid(error.to_string()))?
                    == publication.canonical_manifest_sha256,
                "R dataset graph original manifest canonical pin changed",
            )?;
            if current.is_some_and(|item| request.publication == item.publication) {
                current_index = Some(publication.canonical_index_sha256.clone());
            }
            check(
                probes
                    .insert(
                        publication.canonical_index_sha256,
                        RDatasetProbe {
                            registration: request.clone(),
                            manifest,
                        },
                    )
                    .is_none(),
                "duplicate R dataset canonical index registration",
            )?;
        }
        let mut done = BTreeMap::new();
        let mut active = BTreeSet::new();
        let current = current_index
            .map(|index| Self::open(&index, &probes, pc, recipe, &mut done, &mut active, budget))
            .transpose()?;
        // Earlier R sources are historical checkpoint inputs even when the
        // new window has no R work. Only the exact old chain opens these roots.
        for publication in registered_checkpoint_sources(registration, budget)?.1 {
            let index = probes
                .iter()
                .find(|(_, probe)| probe.registration.publication.sha256 == publication)
                .map(|(index, _)| index)
                .ok_or_else(|| {
                    invalid("original checkpoint requires an unregistered R publication")
                })?;
            Self::open(index, &probes, pc, recipe, &mut done, &mut active, budget)?;
        }
        check(
            done.len() == probes.len(),
            "extra unreachable R dataset registration",
        )?;
        let mut ancestors = done
            .into_values()
            .filter(|item| {
                current.as_ref().is_none_or(|current| {
                    item.publication_pin().sha256 != current.publication_pin().sha256
                })
            })
            .collect::<Vec<_>>();
        ancestors.sort_by(|a, b| a.publication_pin().sha256.cmp(&b.publication_pin().sha256));
        Ok(Self { current, ancestors })
    }
    fn open(
        index: &str,
        probes: &BTreeMap<String, RDatasetProbe>,
        pc: &RegisteredDatasetGraph,
        recipe: &Value,
        done: &mut BTreeMap<String, Arc<VerifiedRoventRAuxiliaryDataset>>,
        active: &mut BTreeSet<String>,
        budget: &RoventRecoveryReadBudget,
    ) -> Result<Arc<VerifiedRoventRAuxiliaryDataset>, ArenaError> {
        if let Some(owner) = done.get(index) {
            return Ok(Arc::clone(owner));
        }
        check(
            active.len() <= 256 && active.insert(index.to_owned()),
            "R dataset ancestor cycle/depth limit",
        )?;
        let probe = probes
            .get(index)
            .ok_or_else(|| invalid("missing original R ancestor dataset"))?;
        let ancestors = probe
            .manifest
            .r_ancestor_index_sha256
            .iter()
            .map(|sha| Self::open(sha, probes, pc, recipe, done, active, budget))
            .collect::<Result<Vec<_>, _>>()?;
        let pc_datasets = probe
            .manifest
            .pc_ancestor_index_sha256
            .iter()
            .map(|sha| {
                std::iter::once(&pc.current)
                    .chain(&pc.ancestors)
                    .find(|owner| owner.index_sha256() == sha.as_str())
                    .cloned()
                    .ok_or_else(|| invalid("R manifest references unregistered P/C ancestor index"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let owner = Arc::new(VerifiedRoventRAuxiliaryDataset::open(
            probe.registration.clone(),
            recipe,
            pc_datasets,
            ancestors,
            budget,
        )?);
        active.remove(index);
        done.insert(index.to_owned(), Arc::clone(&owner));
        Ok(owner)
    }
}
struct RegisteredDatasetGraph {
    current: Arc<VerifiedRoventDataset>,
    ancestors: Vec<Arc<VerifiedRoventDataset>>,
}
impl RegisteredDatasetGraph {
    fn by_publication(&self, sha: &str) -> Result<Arc<VerifiedRoventDataset>, ArenaError> {
        std::iter::once(&self.current)
            .chain(&self.ancestors)
            .find(|owner| owner.registration.publication.sha256 == sha)
            .cloned()
            .ok_or_else(|| {
                invalid("checkpoint references a dataset outside the full registered graph")
            })
    }
    fn new(
        registration: &RoventRecoveredTrainingRegistration,
        budget: &RoventRecoveryReadBudget,
    ) -> Result<Self, ArenaError> {
        let all = registration
            .ancestor_datasets
            .iter()
            .chain(std::iter::once(&registration.current_dataset));
        let mut probes = BTreeMap::new();
        let mut current_index = None;
        for item in all {
            // A family is selected only by the exact externally pinned
            // publication domain. The probe runs that family's strict parser
            // and canonical manifest join; it never re-tags closed bytes.
            let (family, key, ancestor_indices) = probe_registered_pc_dataset(item, budget)?;
            if item.publication == registration.current_dataset.publication {
                current_index = Some(key.clone());
            }
            check(
                probes
                    .insert(
                        key.clone(),
                        DatasetProbe {
                            registration: item.clone(),
                            family,
                            index_sha256: key,
                            ancestor_indices,
                        },
                    )
                    .is_none(),
                "duplicate historical dataset/index registration",
            )?;
        }
        let mut done = BTreeMap::new();
        let mut active = BTreeSet::new();
        let current = Self::open(
            &current_index.ok_or_else(|| invalid("current historical dataset is absent"))?,
            &probes,
            &mut done,
            &mut active,
            budget,
        )?;
        // R and P/C are different source domains. An R collection may retain
        // a pinned P/C genealogy that is outside the current P/C corpus's
        // own ancestor closure. Admit those explicitly declared index roots
        // through the same full P/C verifier; unrelated extras still fail.
        for index in registered_r_pc_indices(registration, budget)? {
            Self::open(&index, &probes, &mut done, &mut active, budget)?;
        }
        for publication in registered_checkpoint_sources(registration, budget)?.0 {
            let index = probes
                .iter()
                .find(|(_, probe)| probe.registration.publication.sha256 == publication)
                .map(|(index, _)| index)
                .ok_or_else(|| {
                    invalid("original checkpoint requires an unregistered P/C publication")
                })?;
            Self::open(index, &probes, &mut done, &mut active, budget)?;
        }
        check(
            done.len() == probes.len(),
            "extra unreachable historical dataset registration",
        )?;
        let mut ancestors = done
            .into_values()
            .filter_map(|owner| {
                (owner.registration.publication != current.registration.publication)
                    .then_some(owner)
            })
            .collect::<Vec<_>>();
        ancestors.sort_by(|a, b| {
            a.registration
                .publication
                .sha256
                .cmp(&b.registration.publication.sha256)
        });
        Ok(Self { current, ancestors })
    }
    fn open(
        key: &str,
        probes: &BTreeMap<String, DatasetProbe>,
        done: &mut BTreeMap<String, Arc<VerifiedRoventDataset>>,
        active: &mut BTreeSet<String>,
        budget: &RoventRecoveryReadBudget,
    ) -> Result<Arc<VerifiedRoventDataset>, ArenaError> {
        if let Some(owner) = done.get(key) {
            return Ok(Arc::clone(owner));
        }
        check(
            active.len() <= 256 && active.insert(key.into()),
            "historical dataset graph cycle/depth exceeded",
        )?;
        let probe = probes
            .get(key)
            .ok_or_else(|| invalid("missing exact historical ancestor dataset registration"))?;
        check(
            probe.index_sha256 == key,
            "historical dataset graph key changed",
        )?;
        let ancestors = probe
            .ancestor_indices
            .iter()
            .map(|parent| Self::open(parent, probes, done, active, budget))
            .collect::<Result<Vec<_>, _>>()?;
        let owner = Arc::new(match probe.family {
            RoventPcRecordFamily::LegacyRecordV1 => {
                VerifiedRoventDataset::open(probe.registration.clone(), ancestors, budget)?
            }
            RoventPcRecordFamily::SubmittedInputV1 => {
                VerifiedRoventDataset::open_submitted_input_v1(
                    probe.registration.clone(),
                    ancestors,
                    budget,
                )?
            }
        });
        check(
            owner.family() == probe.family && owner.index_sha256() == key,
            "historical dataset family or canonical index changed after probing",
        )?;
        active.remove(key);
        done.insert(key.into(), Arc::clone(&owner));
        Ok(owner)
    }
}

fn registered_r_pc_indices(
    registration: &RoventRecoveredTrainingRegistration,
    budget: &RoventRecoveryReadBudget,
) -> Result<BTreeSet<String>, ArenaError> {
    let mut needed = BTreeSet::new();
    for item in registration
        .r_auxiliary_dataset
        .enabled()
        .into_iter()
        .chain(
            registration
                .ancestor_r_auxiliary_datasets
                .enabled()
                .into_iter()
                .flatten(),
        )
    {
        let (_, raw) = budget.read(&item.publication, MAX_METADATA_BYTES, true)?;
        let publication: crate::rovent_collect::RoventRCorpusPublication = decode(&raw)?;
        publication.validate()?;
        let (_, raw) = budget.read(
            &archive_pin(&item.root, &publication.manifest)?,
            MAX_METADATA_BYTES,
            true,
        )?;
        let manifest: rz_experiments::rovent_r_readout::RoventRReadoutManifest = decode(&raw)?;
        check(
            manifest
                .sha256()
                .map_err(|error| invalid(error.to_string()))?
                == publication.canonical_manifest_sha256,
            "R reference source changed original manifest pin",
        )?;
        needed.extend(manifest.pc_ancestor_index_sha256);
        check(
            needed.len() <= 256,
            "R referenced P/C ancestor set exceeds its original bound",
        )?;
    }
    Ok(needed)
}

fn registered_checkpoint_sources(
    registration: &RoventRecoveredTrainingRegistration,
    budget: &RoventRecoveryReadBudget,
) -> Result<(BTreeSet<String>, BTreeSet<String>), ArenaError> {
    let source = match &registration.initialization_source {
        RoventTrainingInputSource::Initialize { .. } => {
            return Ok((BTreeSet::new(), BTreeSet::new()));
        }
        RoventTrainingInputSource::Resume { checkpoint }
        | RoventTrainingInputSource::Fork { checkpoint, .. } => checkpoint,
    };
    let mut pc = BTreeSet::new();
    let mut r = BTreeSet::new();
    let mut seen = BTreeSet::new();
    let mut next = Some(source.publication.clone());
    while let Some(pin) = next {
        check(
            seen.len() < 64 && seen.insert(pin.sha256.clone()),
            "original checkpoint metadata chain cycle/depth",
        )?;
        let (_, raw) = budget.read(&pin, MAX_METADATA_BYTES, true)?;
        let publication: RoventCheckpointPublication = decode(&raw)?;
        publication.validate()?;
        pc.insert(publication.current_dataset_publication_sha256);
        pc.extend(publication.ancestor_dataset_publication_sha256);
        if let Some(auxiliary) = publication.r_auxiliary.enabled() {
            r.insert(auxiliary.dataset_publication_sha256.clone());
        }
        check(
            pc.len() <= 257 && r.len() <= 257,
            "original checkpoint source graph capacity",
        )?;
        next = publication
            .parent_publication
            .as_ref()
            .map(|file| archive_pin(&source.root, file))
            .transpose()?;
    }
    Ok((pc, r))
}
