// Shared commit-last checkpoint machine. Each enclosing module supplies its
// exact family codecs, domains and opaque runtime borrows; public types stay
// distinct because this source is included in separate private namespaces.

fn invalid(reason: impl Into<String>) -> ArenaError {
    ArenaError::Integrity(format!("Rovent utility checkpoint: {}", reason.into()))
}
fn check(condition: bool, reason: &str) -> Result<(), ArenaError> {
    if condition {
        Ok(())
    } else {
        Err(invalid(reason))
    }
}
fn fields(value: &Value, names: &[&str]) -> Result<(), ArenaError> {
    let object = value
        .as_object()
        .ok_or_else(|| invalid("exact object required"))?;
    check(
        object.len() == names.len() && names.iter().all(|name| object.contains_key(*name)),
        "unknown or missing original field",
    )
}
fn count(value: &Value) -> Result<u64, ArenaError> {
    value
        .as_u64()
        .ok_or_else(|| invalid("unsigned integer required"))
}
fn digest(domain: &str, value: &impl Serialize) -> Result<String, ArenaError> {
    generation_digest(domain, value).map_err(ArenaError::Manifest)
}
fn required_nullable<'de, D: serde::Deserializer<'de>, T: Deserialize<'de>>(
    de: D,
) -> Result<Option<T>, D::Error> {
    Option::<T>::deserialize(de)
}
fn deserialize_object<'de, D: serde::Deserializer<'de>, T: Deserialize<'de>>(
    de: D,
) -> Result<T, D::Error> {
    struct Object<T>(std::marker::PhantomData<T>);
    impl<'de, T: Deserialize<'de>> Visitor<'de> for Object<T> {
        type Value = T;
        fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("an original object")
        }
        fn visit_map<M: MapAccess<'de>>(self, map: M) -> Result<T, M::Error> {
            T::deserialize(MapAccessDeserializer::new(map))
        }
    }
    de.deserialize_map(Object(std::marker::PhantomData))
}
fn nullable_object<'de, D: serde::Deserializer<'de>, T: Deserialize<'de>>(
    de: D,
) -> Result<Option<T>, D::Error> {
    struct Object<T>(std::marker::PhantomData<T>);
    impl<'de, T: Deserialize<'de>> Visitor<'de> for Object<T> {
        type Value = Option<T>;
        fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("explicit null or an original object")
        }
        fn visit_unit<E: serde::de::Error>(self) -> Result<Self::Value, E> {
            Ok(None)
        }
        fn visit_none<E: serde::de::Error>(self) -> Result<Self::Value, E> {
            Ok(None)
        }
        fn visit_map<M: MapAccess<'de>>(self, map: M) -> Result<Self::Value, M::Error> {
            T::deserialize(MapAccessDeserializer::new(map)).map(Some)
        }
    }
    de.deserialize_any(Object(std::marker::PhantomData))
}

macro_rules! object {
    ($(#[$meta:meta])* pub struct $name:ident { $($(#[$attr:meta])* pub $field:ident: $ty:ty,)* }) => {
        $(#[$meta])* #[derive(Clone, Debug, Serialize)]
        pub struct $name { $($(#[$attr])* pub $field: $ty,)* }
        impl<'de> Deserialize<'de> for $name {
            fn deserialize<D: serde::Deserializer<'de>>(de: D) -> Result<Self, D::Error> {
                #[derive(Deserialize)] #[serde(deny_unknown_fields)]
                struct Fields { $($(#[$attr])* $field: $ty,)* }
                struct Object;
                impl<'de> Visitor<'de> for Object {
                    type Value = Fields;
                    fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { f.write_str("an exact utility checkpoint object") }
                    fn visit_map<M: MapAccess<'de>>(self, map: M) -> Result<Fields, M::Error> { Fields::deserialize(MapAccessDeserializer::new(map)) }
                }
                let fields = de.deserialize_map(Object)?;
                Ok(Self { $($field: fields.$field,)* })
            }
        }
    };
}

object! {
#[derive(Eq, PartialEq)]
pub struct UtilityCheckpointFile {
    pub path: String,
    pub sha256: String,
    pub bytes: u64,
}
}
impl UtilityCheckpointFile {
    fn validate(&self) -> Result<(), ArenaError> {
        RoventUtilityCostEvidence {
            path: self.path.clone(),
            sha256: self.sha256.clone(),
            bytes: self.bytes,
        }
        .validate()
    }
    fn common(&self) -> RoventPublicationFile {
        RoventPublicationFile {
            path: self.path.clone(),
            sha256: self.sha256.clone(),
            bytes: self.bytes,
        }
    }
    fn from_common(value: RoventPublicationFile) -> Self {
        Self {
            path: value.path,
            sha256: value.sha256,
            bytes: value.bytes,
        }
    }
    fn absolute(&self, root: &Path) -> Result<RoventTrainingFilePin, ArenaError> {
        self.validate()?;
        check(
            root.is_absolute()
                && root
                    .components()
                    .all(|part| matches!(part, Component::RootDir | Component::Normal(_))),
            "normalized absolute archive root required",
        )?;
        Ok(RoventTrainingFilePin {
            path: root.join(&self.path),
            sha256: self.sha256.clone(),
            bytes: self.bytes,
        })
    }
}
object! {
#[derive(Eq, PartialEq)]
pub struct UtilityCompletionAck { pub sha256: String, pub bytes: u64, }
}
object! {
pub struct UtilityCheckpointPublication {
    pub schema: String,
    pub execution_id: String,
    pub window_id: String,
    pub run_id: String,
    pub preparation_sha256: String,
    pub dataset_sha256: String,
    pub window_spec_sha256: String,
    pub trainer_source_sha256: String,
    pub trainer_python_executable_sha256: String,
    pub trainer_dependency_set_sha256: String,
    pub tensor: UtilityCheckpointFile,
    pub retirement: UtilityCheckpointFile,
    pub tensor_projection: UtilityCheckpointFile,
    pub initial_ledger_state: UtilityCheckpointFile,
    pub original_preparation: UtilityCheckpointFile,
    pub original_window: UtilityCheckpointFile,
    pub original_registration: UtilityCheckpointFile,
    pub original_rpc_journal: UtilityCheckpointFile,
    pub completion_receipts: Vec<UtilityCheckpointFile>,
    pub ancestor_completion_receipts: Vec<UtilityCheckpointFile>,
    #[serde(deserialize_with="required_nullable")]
    pub parent_publication: Option<UtilityCheckpointFile>,
    pub cost: UtilityCheckpointFile,
    pub closure: UtilityCheckpointFile,
    pub historical_evidence_only: bool,
    pub execution_authorized: bool,
}
}
impl UtilityCheckpointPublication {
    pub fn validate(&self) -> Result<(), ArenaError> {
        check(
            self.schema == UTILITY_CHECKPOINT_PUBLICATION
                && self.historical_evidence_only
                && !self.execution_authorized,
            "utility-only historical publication domain required",
        )?;
        check(
            [&self.execution_id, &self.window_id, &self.run_id]
                .iter()
                .all(|id| generation_identifier(id))
                && [
                    &self.preparation_sha256,
                    &self.dataset_sha256,
                    &self.window_spec_sha256,
                    &self.trainer_source_sha256,
                    &self.trainer_python_executable_sha256,
                    &self.trainer_dependency_set_sha256,
                ]
                .iter()
                .all(|sha| generation_sha(sha)),
            "exact original utility identity/source pins required",
        )?;
        check(
            self.completion_receipts
                .len()
                .checked_add(self.ancestor_completion_receipts.len())
                .is_some_and(|n| n <= ROVENT_UTILITY_MAX_UPDATES as usize),
            "bounded full ordered utility completion inventory required",
        )?;
        let mut paths = BTreeMap::new();
        for file in self.artifacts() {
            file.validate()?;
            let maximum = if std::ptr::eq(file, &self.tensor) {
                MAX_TENSOR_BYTES
            } else if std::ptr::eq(file, &self.original_rpc_journal) {
                MAX_JOURNAL_BYTES
            } else {
                MAX_METADATA_BYTES
            };
            check(
                file.bytes <= maximum,
                "original utility artifact exceeds its registered format bound",
            )?;
            if let Some(old) = paths.insert(file.path.as_str(), (&file.sha256, file.bytes)) {
                check(
                    old == (&file.sha256, file.bytes),
                    "same relative utility file path has conflicting pins",
                )?;
            }
        }
        let mut completions = BTreeSet::new();
        for receipt in self
            .completion_receipts
            .iter()
            .chain(&self.ancestor_completion_receipts)
        {
            check(
                completions.insert(&receipt.sha256),
                "full utility publication repeats a completion SHA",
            )?;
        }
        Ok(())
    }
    fn artifacts(&self) -> Vec<&UtilityCheckpointFile> {
        let mut out = vec![
            &self.tensor,
            &self.retirement,
            &self.tensor_projection,
            &self.initial_ledger_state,
            &self.original_preparation,
            &self.original_window,
            &self.original_registration,
            &self.original_rpc_journal,
            &self.cost,
            &self.closure,
        ];
        out.extend(&self.completion_receipts);
        out.extend(&self.ancestor_completion_receipts);
        out.extend(&self.parent_publication);
        out
    }
}
object! {
pub struct UtilityTensorProjection {
    pub schema: String,
    pub payload_state_sha256: String,
    pub preparation: Value,
    pub metadata: Value,
    pub sampler: Value,
    pub cuda_devices: Vec<u32>,
    pub parameter_names: Vec<String>,
    pub completed_updates: u64,
    pub dispatched_updates: u64,
    pub ancestor_updates: u64,
    pub pending_microbatches: u64,
    pub cost: RoventUtilityRecordedCost,
    pub ancestor_cost: RoventUtilityRecordedCost,
    pub completion_receipts: Vec<UtilityCompletionAck>,
    pub ancestor_completion_sha256: Vec<String>,
    #[serde(deserialize_with="required_nullable")]
    pub parent_checkpoint_sha256: Option<String>,
    #[serde(deserialize_with="required_nullable")]
    pub source_cost_observation: Option<Value>,
    pub derived_training_state: Value,
    pub model_state_sha256: String,
    pub optimizer_state_sha256: String,
    pub scheduler_state_sha256: String,
    pub rng_state_sha256: String,
    pub sampler_state_sha256: String,
}
}
impl UtilityTensorProjection {
    fn states(&self) -> UtilityStateSeals {
        UtilityStateSeals {
            model_state_sha256: self.model_state_sha256.clone(),
            optimizer_state_sha256: self.optimizer_state_sha256.clone(),
            scheduler_state_sha256: self.scheduler_state_sha256.clone(),
            rng_state_sha256: self.rng_state_sha256.clone(),
            sampler_state_sha256: self.sampler_state_sha256.clone(),
        }
    }
    fn validate(&self) -> Result<(), ArenaError> {
        check(
            self.schema == UTILITY_TENSOR_PROJECTION
                && generation_sha(&self.payload_state_sha256)
                && self.pending_microbatches == 0
                && self.completed_updates <= self.dispatched_updates
                && self.completed_updates == self.completion_receipts.len() as u64
                && self.ancestor_updates == self.ancestor_completion_sha256.len() as u64
                && self
                    .completed_updates
                    .checked_add(self.ancestor_updates)
                    .is_some_and(|n| n <= ROVENT_UTILITY_MAX_UPDATES),
            "utility tensor projection domain/count/no-gradient boundary differs",
        )?;
        validate_checkpoint_preparation(&self.preparation)?;
        check(
            [
                &self.model_state_sha256,
                &self.optimizer_state_sha256,
                &self.scheduler_state_sha256,
                &self.rng_state_sha256,
                &self.sampler_state_sha256,
            ]
            .iter()
            .all(|sha| generation_sha(sha))
                && self.metadata.is_object()
                && self.sampler.is_object(),
            "five distinct original utility state families required",
        )?;
        check(
            self.cuda_devices.len() <= 1
                && self.cuda_devices.windows(2).all(|pair| pair[0] < pair[1])
                && self.cuda_devices.iter().all(|n| *n < 64)
                && !self.parameter_names.is_empty()
                && self.parameter_names.len() <= 64
                && self.parameter_names.iter().all(|name| {
                    !name.is_empty() && name.len() <= 256 && !name.chars().any(char::is_control)
                })
                && self.parameter_names.iter().collect::<BTreeSet<_>>().len()
                    == self.parameter_names.len(),
            "bounded exact utility device/parameter inventory required",
        )?;
        let mut seen = BTreeSet::new();
        for sha in self
            .ancestor_completion_sha256
            .iter()
            .chain(self.completion_receipts.iter().map(|ack| &ack.sha256))
        {
            check(
                generation_sha(sha) && seen.insert(sha),
                "duplicate or invalid full completion SHA",
            )?;
        }
        for ack in &self.completion_receipts {
            check(
                ack.bytes > 0 && ack.bytes <= MAX_METADATA_BYTES,
                "bounded original completion ACK required",
            )?;
        }
        if let Some(parent) = &self.parent_checkpoint_sha256 {
            check(
                generation_sha(parent),
                "original parent tensor SHA required",
            )?;
        }
        if let Some(cost) = &self.source_cost_observation {
            from_value::<RoventUtilityCheckpointCost>(cost.clone())?.validate()?;
        }
        Ok(())
    }
}
object! {
pub struct UtilityCheckpointRetirement {
    pub schema: String,
    #[serde(deserialize_with="deserialize_object")]
    pub checkpoint: RoventTrainingFilePin,
    pub original_request: Value,
    pub parent_projection: Value,
    pub tensor_projection_sha256: String,
    pub historical_evidence_only: bool,
    pub execution_authorized: bool,
}
}
object! {
pub struct UtilityOriginalClosure {
    pub schema: String,
    pub execution_id: String,
    pub window_id: String,
    pub run_id: String,
    pub kind: String,
    #[serde(deserialize_with="deserialize_object")]
    pub child_identity: RoventHistoricalChildIdentity,
    #[serde(deserialize_with="nullable_object")]
    pub process_receipt: Option<ProcessReceipt>,
    #[serde(deserialize_with="required_nullable")]
    pub terminal_receipt: Option<UtilityCheckpointFile>,
    #[serde(deserialize_with="required_nullable")]
    pub boundary_publication: Option<UtilityCheckpointFile>,
    pub child_reap: String,
    pub physical_completion: String,
    pub historical_evidence_only: bool,
    pub execution_authorized: bool,
}
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum UtilityClosureClassification {
    RecordedClosed,
    Unresolved,
}

/// All byte/metadata owners remain retained. The tensor body is not executed
/// or loaded here; its current child scratch check is an independent gate.
pub struct VerifiedUtilityCheckpointSource {
    root: PathBuf,
    publication_pin: RoventTrainingFilePin,
    publication: UtilityCheckpointPublication,
    tensor: HeldHistoricalFile,
    held: Vec<HeldHistoricalFile>,
    receipt_pins: Vec<RoventTrainingFilePin>,
    ancestor_receipt_pins: Vec<RoventTrainingFilePin>,
    preparation: Value,
    window: RoventUtilityTrainWindow,
    dataset: Arc<VerifiedUtilityDataset>,
    ancestors: Arc<VerifiedUtilityAncestors>,
    projection: UtilityTensorProjection,
    cost: RoventUtilityCheckpointCost,
    closure: UtilityOriginalClosure,
    classification: UtilityClosureClassification,
    parent: Option<Arc<Self>>,
}
#[derive(Default)]
struct Visited {
    publications: BTreeSet<String>,
    paths: BTreeMap<PathBuf, (String, u64)>,
    windows: BTreeSet<(String, String)>,
}

impl VerifiedUtilityCheckpointSource {
    pub fn open(
        reference: &RoventUtilityCheckpointReference,
        dataset_references: &[RoventUtilityDatasetReference],
        caller_mixed_registrations: &[RoventUtilityAncestorDatasetRegistration],
        budget: &Arc<RoventRecoveryReadBudget>,
    ) -> Result<Self, CheckpointReadFailure> {
        reference.validate()?;
        check(
            !dataset_references.is_empty() && dataset_references.len() <= 257,
            "bounded original utility dataset reference registry required",
        )?;
        let mut map = BTreeMap::new();
        for reference in dataset_references {
            reference.validate()?;
            check(
                map.insert(reference.publication.sha256.clone(), reference.clone())
                    .is_none(),
                "duplicate original utility dataset publication",
            )?;
        }
        check(
            caller_mixed_registrations.len() <= 256,
            "bounded caller mixed ancestor registry required",
        )?;
        let mut mixed = BTreeMap::new();
        for registration in caller_mixed_registrations {
            registration.validate()?;
            check(
                mixed
                    .insert(
                        registration.publication().sha256.clone(),
                        registration.clone(),
                    )
                    .is_none(),
                "caller mixed registry repeats an original raw publication",
            )?;
        }
        Self::open_inner(
            &reference.root,
            reference.publication.clone(),
            &map,
            &mixed,
            budget,
            &mut Visited::default(),
            0,
        )
    }
    fn open_inner(
        root: &Path,
        pin: RoventTrainingFilePin,
        datasets: &BTreeMap<String, RoventUtilityDatasetReference>,
        mixed: &BTreeMap<String, RoventUtilityAncestorDatasetRegistration>,
        budget: &Arc<RoventRecoveryReadBudget>,
        visited: &mut Visited,
        depth: usize,
    ) -> Result<Self, CheckpointReadFailure> {
        check(
            depth < MAX_CHAIN
                && pin.path.starts_with(root)
                && visited.publications.insert(pin.sha256.clone()),
            "utility checkpoint depth/cycle/original raw publication differs",
        )?;
        let (pub_file, raw) = budget.read(&pin, MAX_METADATA_BYTES, true)?;
        let publication: UtilityCheckpointPublication = decode(&raw)?;
        publication.validate()?;
        check(
            visited.windows.insert((
                publication.execution_id.clone(),
                publication.window_id.clone(),
            )),
            "utility chain repeats original execution/window",
        )?;
        for reference in publication.artifacts() {
            let absolute = reference.absolute(root)?;
            if let Some(old) = visited
                .paths
                .insert(absolute.path, (absolute.sha256.clone(), absolute.bytes))
            {
                check(
                    old == (absolute.sha256, absolute.bytes),
                    "utility ancestry changes a retained relative file pin",
                )?;
            }
        }
        let parent = publication
            .parent_publication
            .as_ref()
            .map(|parent| {
                Self::open_inner(
                    root,
                    parent.absolute(root)?,
                    datasets,
                    mixed,
                    budget,
                    visited,
                    depth + 1,
                )
                .map(Arc::new)
            })
            .transpose()?;
        let mut held = vec![pub_file];
        let preparation = read_json(root, &publication.original_preparation, budget, &mut held)?;
        let window: RoventUtilityTrainWindow = from_value(read_json(
            root,
            &publication.original_window,
            budget,
            &mut held,
        )?)?;
        window.join_cohort_preparation(&preparation)?;
        let registration: RoventUtilityExecutionRegistrationV2 = from_value(read_json(
            root,
            &publication.original_registration,
            budget,
            &mut held,
        )?)?;
        registration.validate()?;
        registration.join_cohort_window(&window, &preparation)?;
        check(
            publication.execution_id == registration.execution_id
                && publication.window_id == window.window_id
                && publication.run_id == window.run_id
                && publication.preparation_sha256 == window.preparation_sha256
                && publication.dataset_sha256 == window.dataset_sha256
                && publication.window_spec_sha256 == window.canonical_sha256()?
                && publication.trainer_source_sha256 == registration.trainer_source_sha256
                && publication.trainer_python_executable_sha256
                    == registration.trainer_python_executable.sha256
                && publication.trainer_dependency_set_sha256
                    == registration.trainer_dependency_set_sha256
                && registration.preparation.sha256 == publication.original_preparation.sha256
                && registration.preparation.bytes == publication.original_preparation.bytes
                && registration.window_spec.sha256 == publication.original_window.sha256
                && registration.window_spec.bytes == publication.original_window.bytes,
            "original utility registration/window/preparation/source join differs",
        )?;
        let reference = datasets
            .get(&registration.current_dataset.publication.sha256)
            .ok_or_else(|| {
                invalid("original utility dataset publication reference is not retained")
            })?;
        check(
            reference.publication.sha256 == registration.current_dataset.publication.sha256
                && reference.publication.bytes == registration.current_dataset.publication.bytes
                && reference.durable_ack.sha256 == registration.current_dataset.durable_ack.sha256
                && reference.durable_ack.bytes == registration.current_dataset.durable_ack.bytes,
            "original utility dataset raw publication/durable ACK pins differ",
        )?;
        if let Some(caller) = mixed.get(&reference.publication.sha256) {
            validate_current_dataset_ancestor(
                caller,
                reference,
                &publication.original_preparation,
            )?;
        }
        let dataset = Arc::new(open_checkpoint_dataset(reference, &preparation, budget)?);
        let relocated_ancestors = registration
            .ancestor_datasets
            .iter()
            .map(|original| {
                let caller = mixed.get(&original.publication().sha256).ok_or_else(|| {
                    invalid("original mixed ancestor lacks a caller-held typed registration")
                })?;
                check(
                    same_ancestor_semantics(original, caller),
                    "caller mixed ancestor kind/raw publication/semantic metadata differs",
                )?;
                Ok(caller.clone())
            })
            .collect::<Result<Vec<_>, ArenaError>>()?;
        let ancestors = Arc::new(open_checkpoint_ancestors(&relocated_ancestors, budget)?);
        let mut required = parent
            .as_ref()
            .map(|source| source.required_publication_sha256s())
            .unwrap_or_default();
        required.remove(&reference.publication.sha256);
        dataset.admit_ancestry(&ancestors, &required)?;
        check(
            dataset.dataset_sha256() == publication.dataset_sha256,
            "source dataset was not validated with its own original cohort preparation",
        )?;
        let projection: UtilityTensorProjection = from_value(read_json(
            root,
            &publication.tensor_projection,
            budget,
            &mut held,
        )?)?;
        projection.validate()?;
        let expected_devices = if window.device == "cpu" {
            Vec::new()
        } else {
            vec![
                window
                    .device
                    .strip_prefix("cuda:")
                    .ok_or_else(|| invalid("original CUDA device missing"))?
                    .parse::<u32>()
                    .map_err(|_| invalid("original CUDA device number differs"))?,
            ]
        };
        check(
            projection.preparation == preparation && projection.cuda_devices == expected_devices,
            "original tensor preparation/device roster differs",
        )?;
        match (&registration.initialization_source, parent.as_deref()) {
            (RoventUtilityInitializationSource::Initialize { .. }, None) => {
                check(
                    projection.source_cost_observation.is_none()
                        && projection.parent_checkpoint_sha256.is_none(),
                    "fresh utility tensor cannot invent a historical parent or cost sidecar",
                )?;
            }
            (RoventUtilityInitializationSource::Resume { .. }, Some(parent)) => {
                check(
                    projection.source_cost_observation == Some(json!(parent.cost))
                        && projection.parent_checkpoint_sha256
                            == parent.projection.parent_checkpoint_sha256,
                    "resume changed the original fork pointer or parent cost observation",
                )?;
            }
            (RoventUtilityInitializationSource::Fork { .. }, Some(parent)) => {
                check(
                    projection.source_cost_observation == Some(json!(parent.cost))
                        && projection.parent_checkpoint_sha256.as_ref()
                            == Some(&parent.tensor.pin.sha256),
                    "fork changed selected tensor ancestry or parent cost observation",
                )?;
            }
            _ => {
                return Err(checkpoint_read_failure(invalid(
                    "utility tensor initialization family/parent differs",
                )));
            }
        }
        let receipt_pins = read_receipts(root, &publication.completion_receipts, budget)?;
        let ancestor_receipt_pins =
            read_receipts(root, &publication.ancestor_completion_receipts, budget)?;
        check(
            projection.completion_receipts.len() == receipt_pins.len()
                && projection
                    .completion_receipts
                    .iter()
                    .zip(&receipt_pins)
                    .all(|(ack, pin)| ack.sha256 == pin.sha256 && ack.bytes == pin.bytes)
                && projection.ancestor_completion_sha256
                    == ancestor_receipt_pins
                        .iter()
                        .map(|pin| pin.sha256.clone())
                        .collect::<Vec<_>>(),
            "original .pt ACK roster differs from full retained receipt files",
        )?;
        let initial = read_json(root, &publication.initial_ledger_state, budget, &mut held)?;
        let restored = initial_restored(
            &initial,
            &receipt_pins,
            &ancestor_receipt_pins,
            parent.as_deref(),
            &registration,
        )?;
        let scope = initial["state_admission_scope"]
            .as_str()
            .ok_or_else(|| invalid("original initial admission scope missing"))?;
        let mut replay = historical_checkpoint_ledger(
            window.clone(),
            preparation.clone(),
            Arc::clone(&dataset),
            restored,
            scope,
        )?;
        let (journal, _) = budget.read(
            &publication.original_rpc_journal.absolute(root)?,
            MAX_JOURNAL_BYTES,
            false,
        )?;
        let journal_cost = replay_journal(&journal, &mut replay, &dataset, &receipt_pins, budget)?;
        held.push(journal);
        let retirement: UtilityCheckpointRetirement =
            from_value(read_json(root, &publication.retirement, budget, &mut held)?)?;
        check(
            retirement.schema == UTILITY_CHECKPOINT_RETIREMENT
                && retirement.historical_evidence_only
                && !retirement.execution_authorized
                && retirement.tensor_projection_sha256 == publication.tensor_projection.sha256
                && retirement.checkpoint.sha256 == publication.tensor.sha256
                && retirement.checkpoint.bytes == publication.tensor.bytes,
            "original utility retirement/raw tensor/projection join differs",
        )?;
        let original_projection_pin: RoventTrainingFilePin =
            from_value(retirement.original_request["tensor_projection"].clone())?;
        check(
            original_projection_pin.sha256 == publication.tensor_projection.sha256
                && original_projection_pin.bytes == publication.tensor_projection.bytes,
            "original utility projection request pin differs",
        )?;
        let expected_parent = replay.checkpoint_projection(
            &retirement.original_request,
            &retirement.checkpoint,
            &original_projection_pin,
        )?;
        subtract(
            from_value(retirement.original_request["snapshot_cost"].clone())?,
            journal_cost,
        )?;
        check(
            expected_parent == retirement.parent_projection,
            "historical parent boundary differs from full original ledger replay",
        )?;
        check(
            projection.sampler == replay.historical_sampler_projection()
                && projection.dispatched_updates == replay.historical_dispatched_updates()
                && projection.rng_state_sha256 == replay.historical_rng_state_sha256()
                && projection.derived_training_state
                    == replay.historical_derived_training_state()?,
            "original tensor sampler/derived contents differ from full journal replay",
        )?;
        validate_projection(
            &projection,
            &retirement,
            &receipt_pins,
            &ancestor_receipt_pins,
            &preparation,
        )?;
        let cost: RoventUtilityCheckpointCost =
            from_value(read_json(root, &publication.cost, budget, &mut held)?)?;
        cost.validate()?;
        let closure: UtilityOriginalClosure =
            from_value(read_json(root, &publication.closure, budget, &mut held)?)?;
        let boundary_unknown = validate_boundary_publication(
            root,
            &publication,
            &closure,
            &projection,
            &retirement,
            parent.as_deref(),
            budget,
            &mut held,
        )?;
        validate_cost(
            &cost,
            &projection,
            &retirement,
            parent.as_deref(),
            boundary_unknown,
        )?;
        let classification =
            validate_closure(root, &publication, &closure, &replay, budget, &mut held)?;
        validate_closure_cost(
            root,
            &cost,
            &retirement,
            &projection,
            &closure,
            budget,
            &mut held,
        )?;
        verify_cost_evidence(root, &cost, budget, &mut held)?;
        let (tensor, _) =
            budget.read(&publication.tensor.absolute(root)?, MAX_TENSOR_BYTES, false)?;
        Ok(Self {
            root: root.to_path_buf(),
            publication_pin: pin,
            publication,
            tensor,
            held,
            receipt_pins,
            ancestor_receipt_pins,
            preparation,
            window,
            dataset,
            ancestors,
            projection,
            cost,
            closure,
            classification,
            parent,
        })
    }
    pub fn publication_pin(&self) -> &RoventTrainingFilePin {
        &self.publication_pin
    }
    pub fn publication(&self) -> &UtilityCheckpointPublication {
        &self.publication
    }
    pub fn preparation(&self) -> &Value {
        &self.preparation
    }
    pub fn window(&self) -> &RoventUtilityTrainWindow {
        &self.window
    }
    pub fn dataset(&self) -> &Arc<VerifiedUtilityDataset> {
        &self.dataset
    }
    pub fn datasets(&self) -> Vec<Arc<VerifiedUtilityDataset>> {
        let mut out = BTreeMap::new();
        let mut current = Some(self);
        while let Some(source) = current {
            out.entry(source.dataset.reference().publication.sha256.clone())
                .or_insert_with(|| Arc::clone(&source.dataset));
            for dataset in checkpoint_utility_datasets(&source.ancestors) {
                out.entry(dataset.reference().publication.sha256.clone())
                    .or_insert_with(|| Arc::clone(dataset));
            }
            current = source.parent.as_deref();
        }
        out.into_values().collect()
    }
    /// The complete original PC/R/utility publication closure. The caller
    /// removes only its own current raw publication before ancestor admission.
    pub fn required_publication_sha256s(&self) -> BTreeSet<String> {
        let mut publications = self.ancestors.publication_sha256s();
        publications.insert(self.dataset.reference().publication.sha256.clone());
        if let Some(parent) = &self.parent {
            publications.extend(parent.required_publication_sha256s());
        }
        publications
    }
    pub fn tensor_file(&self) -> (&RoventTrainingFilePin, &File) {
        (&self.tensor.pin, &self.tensor.file)
    }
    pub fn recorded_closure(&self) -> (&UtilityOriginalClosure, UtilityClosureClassification) {
        (&self.closure, self.classification)
    }
    pub fn parent(&self) -> Option<&Arc<Self>> {
        self.parent.as_ref()
    }
    pub(super) fn restored(&self) -> Result<RestoredUtilityState, ArenaError> {
        let resume = self.cost.known_resume_delta()?;
        Ok(RestoredUtilityState {
            states: self.projection.states(),
            sampler: self.projection.sampler.clone(),
            completed_updates: self.projection.completed_updates,
            dispatched_updates: self.projection.dispatched_updates,
            ancestor_updates: self.projection.ancestor_updates,
            cost: self.projection.cost.checked_add(&resume.known_delta)?,
            ancestor_cost: self.projection.ancestor_cost,
            completion_receipts: self.receipt_pins.clone(),
            ancestor_completion_sha256: self.projection.ancestor_completion_sha256.clone(),
            initial_source_declaration:
                self.projection.derived_training_state["initial_source_declaration"].clone(),
            frozen_base_weights_sha256: text(
                &self.projection.derived_training_state,
                "frozen_base_weights_sha256",
            )?
            .into(),
            frozen_base_mapping_sha256: text(
                &self.projection.derived_training_state,
                "frozen_base_mapping_sha256",
            )?
            .into(),
            source_checkpoint: Some(self.tensor.pin.clone()),
            source_payload_state_sha256: Some(self.projection.payload_state_sha256.clone()),
            source_snapshot_cost_observation: self.projection.source_cost_observation.clone(),
            source_snapshot_cost: Some(self.projection.cost),
            source_snapshot_ancestor_cost: Some(self.projection.ancestor_cost),
            checkpoint_cost: Some(self.cost.clone()),
        })
    }
    pub fn revalidate(&self, budget: &RoventRecoveryReadBudget) -> Result<(), ArenaError> {
        for item in std::iter::once(&self.tensor).chain(&self.held) {
            budget.verify_held(&item.file, &item.pin, false)?;
        }
        for pin in self.receipt_pins.iter().chain(&self.ancestor_receipt_pins) {
            budget.read(pin, MAX_METADATA_BYTES, false)?;
        }
        self.dataset.revalidate(budget)?;
        self.ancestors.revalidate(budget)?;
        if let Some(parent) = &self.parent {
            parent.revalidate(budget)?;
        }
        Ok(())
    }
    pub fn admit_old_run_reclamation(
        self: &Arc<Self>,
        budget: Arc<RoventRecoveryReadBudget>,
    ) -> Result<VerifiedUtilityOldRunReclamation, ArenaError> {
        self.revalidate(&budget)?;
        let mut unresolved = Vec::new();
        let mut cursor = Some(self.as_ref());
        let mut closed = 0usize;
        while let Some(source) = cursor {
            match source.classification {
                UtilityClosureClassification::RecordedClosed => closed += 1,
                UtilityClosureClassification::Unresolved => {
                    unresolved.push(source.closure.child_identity.clone())
                }
            };
            cursor = source.parent.as_deref();
        }
        let scopes = VerifiedHistoricalUnresolvedScopes::admit(&unresolved, Arc::clone(&budget))?;
        Ok(VerifiedUtilityOldRunReclamation {
            _retained_source: Arc::clone(self),
            scopes,
            budget,
            closed,
        })
    }
}
pub struct VerifiedUtilityOldRunReclamation {
    _retained_source: Arc<VerifiedUtilityCheckpointSource>,
    scopes: VerifiedHistoricalUnresolvedScopes,
    budget: Arc<RoventRecoveryReadBudget>,
    closed: usize,
}
impl VerifiedUtilityOldRunReclamation {
    pub fn revalidate(&self) -> Result<(), ArenaError> {
        self._retained_source.revalidate(&self.budget)?;
        self.scopes.revalidate()
    }
    pub fn observation(&self) -> Value {
        json!({"schema":OLD_RUN_RECLAMATION,"recorded_closed_historical_windows":self.closed,"past_native_authority_reissued":false,"past_physical_completion_reissued":false,"execution_authorized":false})
    }
}

fn text<'a>(value: &'a Value, key: &str) -> Result<&'a str, ArenaError> {
    value[key]
        .as_str()
        .ok_or_else(|| invalid(format!("missing original text {key}")))
}
fn from_value<T: serde::de::DeserializeOwned>(value: Value) -> Result<T, ArenaError> {
    serde_json::from_value(value).map_err(|error| invalid(error.to_string()))
}
fn decode<T: serde::de::DeserializeOwned>(raw: &[u8]) -> Result<T, ArenaError> {
    let strict: StrictValue =
        serde_json::from_slice(raw).map_err(|error| invalid(error.to_string()))?;
    from_value(strict.0)
}
struct StrictValue(Value);
impl<'de> Deserialize<'de> for StrictValue {
    fn deserialize<D: serde::Deserializer<'de>>(de: D) -> Result<Self, D::Error> {
        struct Strict;
        impl<'de> Visitor<'de> for Strict {
            type Value = StrictValue;
            fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str("JSON with unique object keys")
            }
            fn visit_map<M: MapAccess<'de>>(self, mut map: M) -> Result<Self::Value, M::Error> {
                let mut out = serde_json::Map::new();
                while let Some(key) = map.next_key::<String>()? {
                    if out.contains_key(&key) {
                        return Err(serde::de::Error::custom("duplicate original JSON key"));
                    }
                    out.insert(key, map.next_value::<StrictValue>()?.0);
                }
                Ok(StrictValue(Value::Object(out)))
            }
            fn visit_seq<S: serde::de::SeqAccess<'de>>(
                self,
                mut seq: S,
            ) -> Result<Self::Value, S::Error> {
                let mut out = Vec::new();
                while let Some(value) = seq.next_element::<StrictValue>()? {
                    out.push(value.0);
                }
                Ok(StrictValue(Value::Array(out)))
            }
            fn visit_bool<E: serde::de::Error>(self, v: bool) -> Result<Self::Value, E> {
                Ok(StrictValue(Value::Bool(v)))
            }
            fn visit_u64<E: serde::de::Error>(self, v: u64) -> Result<Self::Value, E> {
                Ok(StrictValue(json!(v)))
            }
            fn visit_i64<E: serde::de::Error>(self, v: i64) -> Result<Self::Value, E> {
                Ok(StrictValue(json!(v)))
            }
            fn visit_f64<E: serde::de::Error>(self, v: f64) -> Result<Self::Value, E> {
                serde_json::Number::from_f64(v)
                    .map(|n| StrictValue(Value::Number(n)))
                    .ok_or_else(|| E::custom("nonfinite original JSON number"))
            }
            fn visit_str<E: serde::de::Error>(self, v: &str) -> Result<Self::Value, E> {
                Ok(StrictValue(Value::String(v.into())))
            }
            fn visit_string<E: serde::de::Error>(self, v: String) -> Result<Self::Value, E> {
                Ok(StrictValue(Value::String(v)))
            }
            fn visit_unit<E: serde::de::Error>(self) -> Result<Self::Value, E> {
                Ok(StrictValue(Value::Null))
            }
        }
        de.deserialize_any(Strict)
    }
}
fn read_json(
    root: &Path,
    reference: &UtilityCheckpointFile,
    budget: &RoventRecoveryReadBudget,
    held: &mut Vec<HeldHistoricalFile>,
) -> Result<Value, ArenaError> {
    let (file, raw) = budget.read(&reference.absolute(root)?, MAX_METADATA_BYTES, true)?;
    let value = decode(&raw)?;
    held.push(file);
    Ok(value)
}
fn read_receipts(
    root: &Path,
    refs: &[UtilityCheckpointFile],
    budget: &RoventRecoveryReadBudget,
) -> Result<Vec<RoventTrainingFilePin>, ArenaError> {
    let mut out = Vec::with_capacity(refs.len());
    for reference in refs {
        let pin = reference.absolute(root)?;
        budget.read(&pin, MAX_METADATA_BYTES, false)?;
        out.push(pin);
    }
    Ok(out)
}

fn same_pins(left: &[RoventTrainingFilePin], right: &[RoventTrainingFilePin]) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right)
            .all(|(a, b)| a.sha256 == b.sha256 && a.bytes == b.bytes)
}
fn initial_restored(
    initial: &Value,
    receipts: &[RoventTrainingFilePin],
    ancestors: &[RoventTrainingFilePin],
    parent: Option<&VerifiedUtilityCheckpointSource>,
    registration: &RoventUtilityExecutionRegistrationV2,
) -> Result<RestoredUtilityState, ArenaError> {
    fields(
        initial,
        &[
            "schema",
            "state_admission_scope",
            "states",
            "sampler",
            "completed_updates",
            "dispatched_updates",
            "ancestor_updates",
            "cost",
            "ancestor_cost",
            "completion_receipts",
            "ancestor_completion_sha256",
            "initial_source_declaration",
            "frozen_base_weights_sha256",
            "frozen_base_mapping_sha256",
            "source_checkpoint",
            "source_payload_state_sha256",
            "source_snapshot_cost_observation",
            "source_snapshot_cost",
            "source_snapshot_ancestor_cost",
            "checkpoint_cost",
            "historical_evidence_only",
            "execution_authorized",
        ],
    )?;
    check(
        initial["schema"] == UTILITY_INITIAL_LEDGER
            && initial["historical_evidence_only"] == true
            && initial["execution_authorized"] == false,
        "original utility initial projection domain required",
    )?;
    fields(
        &initial["states"],
        &[
            "model_state_sha256",
            "optimizer_state_sha256",
            "scheduler_state_sha256",
            "rng_state_sha256",
            "sampler_state_sha256",
        ],
    )?;
    if !initial["source_checkpoint"].is_null() {
        fields(&initial["source_checkpoint"], &["path", "sha256", "bytes"])?;
    }
    let completed = count(&initial["completed_updates"])?;
    let original_receipts: Vec<RoventTrainingFilePin> =
        from_value(initial["completion_receipts"].clone())?;
    for pin in initial["completion_receipts"]
        .as_array()
        .ok_or_else(|| invalid("original initial receipt list required"))?
    {
        fields(pin, &["path", "sha256", "bytes"])?;
    }
    check(
        completed == original_receipts.len() as u64
            && original_receipts.len() <= receipts.len()
            && same_pins(&original_receipts, &receipts[..original_receipts.len()]),
        "initial completion prefix differs from original held full chain",
    )?;
    let ancestor_sha: Vec<String> = from_value(initial["ancestor_completion_sha256"].clone())?;
    check(
        ancestor_sha
            == ancestors
                .iter()
                .map(|pin| pin.sha256.clone())
                .collect::<Vec<_>>(),
        "initial ancestor full completion chain differs",
    )?;
    let restored = RestoredUtilityState {
        states: from_value(initial["states"].clone())?,
        sampler: initial["sampler"].clone(),
        completed_updates: completed,
        dispatched_updates: count(&initial["dispatched_updates"])?,
        ancestor_updates: count(&initial["ancestor_updates"])?,
        cost: from_value(initial["cost"].clone())?,
        ancestor_cost: from_value(initial["ancestor_cost"].clone())?,
        completion_receipts: receipts[..original_receipts.len()].to_vec(),
        ancestor_completion_sha256: ancestor_sha,
        initial_source_declaration: initial["initial_source_declaration"].clone(),
        frozen_base_weights_sha256: text(initial, "frozen_base_weights_sha256")?.into(),
        frozen_base_mapping_sha256: text(initial, "frozen_base_mapping_sha256")?.into(),
        source_checkpoint: from_value(initial["source_checkpoint"].clone())?,
        source_payload_state_sha256: from_value(initial["source_payload_state_sha256"].clone())?,
        source_snapshot_cost_observation: from_value(
            initial["source_snapshot_cost_observation"].clone(),
        )?,
        source_snapshot_cost: from_value(initial["source_snapshot_cost"].clone())?,
        source_snapshot_ancestor_cost: from_value(
            initial["source_snapshot_ancestor_cost"].clone(),
        )?,
        checkpoint_cost: from_value(initial["checkpoint_cost"].clone())?,
    };
    match (&registration.initialization_source, parent) {
        (RoventUtilityInitializationSource::Initialize { .. }, None) => {
            check(
                initial["state_admission_scope"]
                    == "actual_original_child_initial_observation_not_independent_torch_recomputation",
                "fresh historical initial scope differs",
            )?;
            check(
                restored.completed_updates == 0
                    && restored.dispatched_updates == 0
                    && restored.ancestor_updates == 0
                    && restored.completion_receipts.is_empty()
                    && restored.ancestor_completion_sha256.is_empty()
                    && restored.cost == RoventUtilityRecordedCost::default()
                    && restored.ancestor_cost == RoventUtilityRecordedCost::default()
                    && restored.source_checkpoint.is_none()
                    && restored.source_payload_state_sha256.is_none()
                    && restored.checkpoint_cost.is_none()
                    && restored.source_snapshot_cost.is_none()
                    && restored.source_snapshot_ancestor_cost.is_none()
                    && restored.source_snapshot_cost_observation.is_none(),
                "fresh original initial projection has learned/resumed work",
            )?;
        }
        (RoventUtilityInitializationSource::Resume { checkpoint }, Some(parent))
        | (RoventUtilityInitializationSource::Fork { checkpoint }, Some(parent)) => {
            check(
                checkpoint.publication.sha256 == parent.publication_pin.sha256
                    && checkpoint.publication.bytes == parent.publication_pin.bytes,
                "original parent publication differs from registered source",
            )?;
            let source = restored
                .source_checkpoint
                .as_ref()
                .ok_or_else(|| invalid("original source checkpoint pin missing"))?;
            check(
                source.sha256 == parent.tensor.pin.sha256
                    && source.bytes == parent.tensor.pin.bytes
                    && restored.source_payload_state_sha256.as_deref()
                        == Some(parent.projection.payload_state_sha256.as_str())
                    && restored.source_snapshot_cost == Some(parent.projection.cost)
                    && restored.source_snapshot_ancestor_cost
                        == Some(parent.projection.ancestor_cost)
                    && restored.source_snapshot_cost_observation
                        == parent.projection.source_cost_observation
                    && restored.checkpoint_cost.as_ref() == Some(&parent.cost)
                    && restored.initial_source_declaration
                        == parent.projection.derived_training_state["initial_source_declaration"]
                    && restored.frozen_base_weights_sha256
                        == parent.projection.derived_training_state["frozen_base_weights_sha256"]
                    && restored.frozen_base_mapping_sha256
                        == parent.projection.derived_training_state["frozen_base_mapping_sha256"],
                "original parent tensor/cost/frozen/source projection changed",
            )?;
            match &registration.initialization_source {
                RoventUtilityInitializationSource::Resume { .. } => {
                    check(
                        initial["state_admission_scope"]
                            == "original_held_checkpoint_and_full_completion_chain",
                        "original resume admission scope differs",
                    )?;
                    let expected = parent.restored()?;
                    check(
                        restored.states == expected.states
                            && restored.sampler == expected.sampler
                            && restored.completed_updates == expected.completed_updates
                            && restored.dispatched_updates == expected.dispatched_updates
                            && restored.ancestor_updates == expected.ancestor_updates
                            && restored.cost == expected.cost
                            && restored.ancestor_cost == expected.ancestor_cost
                            && same_pins(
                                &restored.completion_receipts,
                                &expected.completion_receipts,
                            )
                            && restored.ancestor_completion_sha256
                                == expected.ancestor_completion_sha256,
                        "resume changed original states/counts/sampler/known lower cost",
                    )?;
                }
                RoventUtilityInitializationSource::Fork { .. } => {
                    check(
                        initial["state_admission_scope"]
                            == "original_checkpoint_model_and_chain_with_actual_child_reset_state_observation",
                        "original fork admission scope differs",
                    )?;
                    let expected_ancestors = parent
                        .ancestor_receipt_pins
                        .iter()
                        .chain(&parent.receipt_pins)
                        .map(|pin| pin.sha256.clone())
                        .collect::<Vec<_>>();
                    check(
                        restored.states.model_state_sha256 == parent.projection.model_state_sha256
                            && restored.completed_updates == 0
                            && restored.dispatched_updates == 0
                            && restored.completion_receipts.is_empty()
                            && restored.ancestor_updates == expected_ancestors.len() as u64
                            && restored.ancestor_completion_sha256 == expected_ancestors
                            && restored.cost == RoventUtilityRecordedCost::default()
                            && restored.ancestor_cost == parent.cost.known_lower_bound,
                        "fork lost original full chain/known costs or did not reset current work",
                    )?;
                }
                _ => unreachable!(),
            }
        }
        _ => {
            return Err(invalid(
                "utility initialize/resume/fork parent domain differs",
            ));
        }
    }
    Ok(restored)
}

struct OriginalRead<'a> {
    file: &'a File,
    offset: u64,
    maximum: u64,
}
impl std::io::Read for OriginalRead<'_> {
    fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
        let request = (self.maximum - self.offset).min(buffer.len() as u64) as usize;
        if request == 0 {
            return Ok(0);
        }
        let count = self.file.read_at(&mut buffer[..request], self.offset)?;
        self.offset += count as u64;
        Ok(count)
    }
}
fn replay_journal(
    journal: &HeldHistoricalFile,
    ledger: &mut UtilityLedger,
    dataset: &VerifiedUtilityDataset,
    receipts: &[RoventTrainingFilePin],
    budget: &RoventRecoveryReadBudget,
) -> Result<RoventUtilityRecordedCost, ArenaError> {
    budget.reserve(journal.pin.bytes)?;
    let mut reader = BufReader::with_capacity(
        65_536,
        OriginalRead {
            file: &journal.file,
            offset: 0,
            maximum: journal.pin.bytes,
        },
    );
    let mut sequence = 0_u64;
    let mut updates = BTreeSet::new();
    let mut saw_begin = false;
    let mut original_pending = 0_u8;
    let mut original_pending_known = false;
    let mut high_water: RoventUtilityRecordedCost =
        from_value(ledger.initial_ledger_projection()["cost"].clone())?;
    loop {
        budget.reserve(0)?;
        let mut line = Vec::new();
        loop {
            let available = reader
                .fill_buf()
                .map_err(|_| invalid("original utility journal read failed"))?;
            if available.is_empty() {
                break;
            }
            let end = available
                .iter()
                .position(|byte| *byte == b'\n')
                .map_or(available.len(), |offset| offset + 1);
            check(
                line.len()
                    .checked_add(end)
                    .is_some_and(|n| n <= MAX_JOURNAL_EVENT_BYTES),
                "utility journal event exceeds bounded frame",
            )?;
            line.extend_from_slice(&available[..end]);
            let complete = available[end - 1] == b'\n';
            reader.consume(end);
            if complete {
                break;
            }
        }
        if line.is_empty() {
            break;
        }
        check(
            line.last() == Some(&b'\n'),
            "original utility journal lacks a complete line",
        )?;
        let event: Value = decode(&line)?;
        fields(
            &event,
            &[
                "schema",
                "sequence",
                "kind",
                "request",
                "completion_receipt",
                "ack",
            ],
        )?;
        sequence = sequence
            .checked_add(1)
            .ok_or_else(|| invalid("journal sequence overflow"))?;
        check(
            event["schema"] == JOURNAL && count(&event["sequence"])? == sequence,
            "original dense utility journal sequence differs",
        )?;
        match event["kind"].as_str() {
            Some("begin_window") => {
                check(
                    !saw_begin && sequence == 1 && event["completion_receipt"].is_null(),
                    "original utility begin is missing/duplicated",
                )?;
                ledger.replay_historical_begin(&event["request"])?;
                check(event["ack"] == json!({}), "original begin ACK differs")?;
                saw_begin = true;
            }
            Some("consume_pair_context") => {
                check(
                    saw_begin && event["completion_receipt"].is_null(),
                    "original utility consume order differs",
                )?;
                ledger.replay_historical_consumption(&event["request"])?;
                let context =
                    dataset.validate_context_projection(&event["request"]["source_projection"])?;
                original_pending = original_pending
                    .checked_add(1)
                    .filter(|n| *n <= 8)
                    .ok_or_else(|| {
                        invalid("original journal accumulation exceeds eight sources")
                    })?;
                original_pending_known |= context.unique_known_pairs() > 0;
                if original_pending == 8 && !original_pending_known {
                    original_pending = 0;
                }
                check(event["ack"] == json!({}), "original consume ACK differs")?;
            }
            Some("committed_update") => {
                check(saw_begin, "original utility update precedes begin")?;
                check(
                    original_pending == 8 && original_pending_known,
                    "full original journal lacks the eight consumed sources of this update",
                )?;
                let original: RoventTrainingFilePin =
                    from_value(event["completion_receipt"].clone())?;
                let pin = receipts
                    .iter()
                    .find(|pin| pin.sha256 == original.sha256 && pin.bytes == original.bytes)
                    .ok_or_else(|| {
                        invalid("journal update lacks an original retained completion file")
                    })?;
                check(
                    updates.insert(pin.sha256.clone()),
                    "duplicate original committed update",
                )?;
                let (_, raw) = budget.read(pin, MAX_METADATA_BYTES, true)?;
                let body: Value = decode(&raw)?;
                check(
                    body["original_request"] == event["request"]
                        && event["ack"] == json!({"sha256":pin.sha256,"bytes":pin.bytes}),
                    "original update request/durable ACK differs",
                )?;
                let observed: RoventUtilityRecordedCost =
                    from_value(event["request"]["cost"].clone())?;
                subtract(observed, high_water)?;
                high_water = observed;
                ledger.replay_historical_completion(&body, pin)?;
                original_pending = 0;
                original_pending_known = false;
            }
            Some("checkpoint_published") => {
                check(
                    saw_begin && event["completion_receipt"].is_null() && event["ack"] == json!({}),
                    "original historical checkpoint journal order/ACK differs",
                )?;
                fields(
                    &event["request"]["checkpoint"],
                    &["path", "sha256", "bytes"],
                )?;
                fields(
                    &event["request"]["tensor_projection"],
                    &["path", "sha256", "bytes"],
                )?;
                let tensor: RoventTrainingFilePin =
                    from_value(event["request"]["checkpoint"].clone())?;
                let projection: RoventTrainingFilePin =
                    from_value(event["request"]["tensor_projection"].clone())?;
                ledger.checkpoint_projection(&event["request"], &tensor, &projection)?;
                check(
                    event["request"]["rng_state_sha256"] == ledger.historical_rng_state_sha256(),
                    "original checkpoint RNG differs from the fixed non-random utility profile ledger",
                )?;
                let snapshot: RoventUtilityRecordedCost =
                    from_value(event["request"]["snapshot_cost"].clone())?;
                let post: RoventUtilityRecordedCost =
                    from_value(event["request"]["post_snapshot_cost"].clone())?;
                subtract(snapshot, high_water)?;
                subtract(post, snapshot)?;
                high_water = post;
            }
            _ => return Err(invalid("unknown utility journal kind")),
        }
    }
    check(saw_begin, "utility journal has no original begin")?;
    budget.verify_held(&journal.file, &journal.pin, false)?;
    Ok(high_water)
}

fn validate_projection(
    projection: &UtilityTensorProjection,
    retirement: &UtilityCheckpointRetirement,
    receipts: &[RoventTrainingFilePin],
    ancestors: &[RoventTrainingFilePin],
    preparation: &Value,
) -> Result<(), ArenaError> {
    let parent = &retirement.parent_projection;
    let states =
        serde_json::to_value(projection.states()).map_err(|error| invalid(error.to_string()))?;
    check(
        parent["states"] == states
            && parent["completed_updates"] == projection.completed_updates
            && parent["ancestor_updates"] == projection.ancestor_updates
            && parent["snapshot_cost"] == json!(projection.cost)
            && retirement.original_request["model_state_sha256"] == projection.model_state_sha256
            && retirement.original_request["optimizer_state_sha256"]
                == projection.optimizer_state_sha256
            && retirement.original_request["scheduler_state_sha256"]
                == projection.scheduler_state_sha256
            && retirement.original_request["rng_state_sha256"] == projection.rng_state_sha256
            && retirement.original_request["sampler_state_sha256"]
                == projection.sampler_state_sha256,
        "original .pt five-state/count/cost projection differs from boundary",
    )?;
    let derived = &projection.derived_training_state;
    fields(
        derived,
        &[
            "schema",
            "initial_source_declaration",
            "current_tensor_kind",
            "completed_updates",
            "ancestor_updates",
            "total_updates",
            "completion_chain_sha256",
            "recipe_sha256",
            "registration_sha256",
            "frozen_base_weights_sha256",
            "frozen_base_mapping_sha256",
            "live_training_authority_reissued",
        ],
    )?;
    let chain = ancestors
        .iter()
        .chain(receipts)
        .map(|pin| &pin.sha256)
        .collect::<Vec<_>>();
    let total = projection
        .completed_updates
        .checked_add(projection.ancestor_updates)
        .ok_or_else(|| invalid("utility derived total overflow"))?;
    let (recipe_sha256, source_registration_sha256) = checkpoint_preparation_seals(preparation)?;
    check(
        derived["schema"] == DERIVED
            && derived["completed_updates"] == projection.completed_updates
            && derived["ancestor_updates"] == projection.ancestor_updates
            && derived["total_updates"] == total
            && derived["current_tensor_kind"]
                == if total > 0 {
                    "learned"
                } else {
                    "initial_source_only"
                }
            && derived["completion_chain_sha256"] == digest(DERIVED, &chain)?
            && derived["recipe_sha256"] == recipe_sha256
            && derived["registration_sha256"] == source_registration_sha256
            && derived["frozen_base_weights_sha256"]
                == preparation["registration"]["frozen_base"]["weights_sha256"]
            && derived["live_training_authority_reissued"] == false
            && generation_sha(text(derived, "frozen_base_mapping_sha256")?),
        "original derived utility state/full completion/frozen identity differs",
    )?;
    check(
        retirement.original_request["derived_training_state_sha256"] == digest(DERIVED, derived)?,
        "original derived training-state seal differs",
    )?;
    let registration = &preparation["registration"];
    let frozen = &registration["frozen_base"];
    check(
        projection.metadata
            == json!({
                "schema":"rz-nnuec-utility-torch-metadata/1",
                "model_schema":registration["utility"]["schema"],
                "model_semantics":registration["utility"]["semantics"],
                "base_schema":frozen["schema"], "base_semantics":frozen["semantics"],
                "base_weights_sha256":frozen["weights_sha256"],
                "frozen_base_mapping_sha256":derived["frozen_base_mapping_sha256"],
                "frozen_base_config":frozen["config"],
                "hidden_width":preparation["parameter_spec"]["hidden_width"],
                "input_width":preparation["parameter_spec"]["input_width"],
                "initial_training":derived["initial_source_declaration"]
            }),
        "original utility tensor metadata differs from its own preparation/frozen/source semantics",
    )?;
    let names = preparation["recipe"]["active_parameter_names"]
        .as_array()
        .ok_or_else(|| invalid("original active parameter order missing"))?
        .iter()
        .map(|wire| {
            let wire = wire
                .as_str()
                .ok_or_else(|| invalid("active parameter wire name must be text"))?;
            preparation["parameter_spec"]["wire_to_parameter"][wire]
                .as_str()
                .map(str::to_owned)
                .ok_or_else(|| invalid("original parameter mapping missing"))
        })
        .collect::<Result<Vec<_>, ArenaError>>()?;
    check(
        projection.parameter_names == names,
        "original utility parameter membership/order differs",
    )?;
    Ok(())
}
fn subtract(
    after: RoventUtilityRecordedCost,
    before: RoventUtilityRecordedCost,
) -> Result<RoventUtilityRecordedCost, ArenaError> {
    fn sub(a: u64, b: u64) -> Result<u64, ArenaError> {
        a.checked_sub(b)
            .ok_or_else(|| invalid("original utility cost regressed"))
    }
    Ok(RoventUtilityRecordedCost {
        forward_calls: sub(after.forward_calls, before.forward_calls)?,
        backward_calls: sub(after.backward_calls, before.backward_calls)?,
        samples: sub(after.samples, before.samples)?,
        wall_time_ns: sub(after.wall_time_ns, before.wall_time_ns)?,
        partial_accumulation_ns: sub(
            after.partial_accumulation_ns,
            before.partial_accumulation_ns,
        )?,
        replay_ns: sub(after.replay_ns, before.replay_ns)?,
        masked_groups: sub(after.masked_groups, before.masked_groups)?,
        output_bytes: sub(after.output_bytes, before.output_bytes)?,
    })
}
fn inherited_tails(
    parent: Option<&VerifiedUtilityCheckpointSource>,
) -> Result<Vec<RoventUtilityInheritedUnknownTail>, ArenaError> {
    let mut expected = parent.map_or_else(Vec::new, |parent| {
        parent.cost.inherited_unknown_tails.clone()
    });
    if let Some(parent) = parent
        && parent.cost.closure_tail.is_unknown()
    {
        expected.push(RoventUtilityInheritedUnknownTail {
            checkpoint_publication_sha256: parent.publication_pin.sha256.clone(),
            execution_id: parent.publication.execution_id.clone(),
            window_id: parent.publication.window_id.clone(),
            tail: parent.cost.closure_tail.clone(),
        });
    }
    expected.sort_by(|a, b| {
        (
            &a.checkpoint_publication_sha256,
            &a.execution_id,
            &a.window_id,
        )
            .cmp(&(
                &b.checkpoint_publication_sha256,
                &b.execution_id,
                &b.window_id,
            ))
    });
    check(
        expected.windows(2).all(|pair| {
            pair[0].checkpoint_publication_sha256 != pair[1].checkpoint_publication_sha256
        }),
        "original utility unknown tail repeated",
    )?;
    Ok(expected)
}
fn validate_cost(
    cost: &RoventUtilityCheckpointCost,
    projection: &UtilityTensorProjection,
    retirement: &UtilityCheckpointRetirement,
    parent: Option<&VerifiedUtilityCheckpointSource>,
    boundary_unknown: Option<RoventUtilityInheritedUnknownTail>,
) -> Result<(), ArenaError> {
    let post: RoventUtilityRecordedCost =
        from_value(retirement.parent_projection["post_snapshot_cost"].clone())?;
    let mut expected = inherited_tails(parent)?;
    expected.extend(boundary_unknown);
    sort_unknown_tails(&mut expected)?;
    check(
        cost.snapshot_cost == projection.cost
            && cost.ancestor_cost == projection.ancestor_cost
            && cost.post_snapshot_delta == subtract(post, projection.cost)?
            && cost.inherited_unknown_tails == expected,
        "original utility snapshot/postdelta/ancestor unknown-tail conservation differs",
    )
}
fn sort_unknown_tails(tails: &mut [RoventUtilityInheritedUnknownTail]) -> Result<(), ArenaError> {
    tails.sort_by(|a, b| {
        (
            &a.checkpoint_publication_sha256,
            &a.execution_id,
            &a.window_id,
        )
            .cmp(&(
                &b.checkpoint_publication_sha256,
                &b.execution_id,
                &b.window_id,
            ))
    });
    check(
        tails.windows(2).all(|pair| {
            pair[0].checkpoint_publication_sha256 != pair[1].checkpoint_publication_sha256
        }),
        "original unknown boundary tail repeated",
    )
}
fn same_boundary_publication(
    left: &UtilityCheckpointPublication,
    right: &UtilityCheckpointPublication,
) -> Result<bool, ArenaError> {
    let mut left = serde_json::to_value(left).map_err(|error| invalid(error.to_string()))?;
    let mut right = serde_json::to_value(right).map_err(|error| invalid(error.to_string()))?;
    for value in [&mut left, &mut right] {
        let object = value
            .as_object_mut()
            .ok_or_else(|| invalid("publication object required"))?;
        object.remove("cost");
        object.remove("closure");
    }
    Ok(left == right)
}
#[allow(clippy::too_many_arguments)]
fn validate_boundary_publication(
    root: &Path,
    publication: &UtilityCheckpointPublication,
    closure: &UtilityOriginalClosure,
    projection: &UtilityTensorProjection,
    retirement: &UtilityCheckpointRetirement,
    parent: Option<&VerifiedUtilityCheckpointSource>,
    budget: &RoventRecoveryReadBudget,
    held: &mut Vec<HeldHistoricalFile>,
) -> Result<Option<RoventUtilityInheritedUnknownTail>, ArenaError> {
    if closure.kind == "optimizer_boundary" {
        check(
            closure.boundary_publication.is_none(),
            "running boundary cannot replace an original boundary",
        )?;
        return Ok(None);
    }
    let reference = closure
        .boundary_publication
        .as_ref()
        .ok_or_else(|| invalid("terminal closure must retain its original running boundary"))?;
    let boundary: UtilityCheckpointPublication =
        from_value(read_json(root, reference, budget, held)?)?;
    boundary.validate()?;
    check(
        same_boundary_publication(publication, &boundary)?,
        "terminal publication changed original tensor/state/source/journal/receipt bytes",
    )?;
    let original: UtilityOriginalClosure =
        from_value(read_json(root, &boundary.closure, budget, held)?)?;
    check(
        original.schema == UTILITY_ORIGINAL_CLOSURE
            && original.kind == "optimizer_boundary"
            && original.boundary_publication.is_none()
            && original.terminal_receipt.is_none()
            && original.process_receipt.is_none()
            && original.child_reap == "unresolved"
            && original.physical_completion == "unknown"
            && original.historical_evidence_only
            && !original.execution_authorized
            && original.execution_id == closure.execution_id
            && original.window_id == closure.window_id
            && original.run_id == closure.run_id
            && json!(original.child_identity) == json!(closure.child_identity),
        "terminal closure changed original running identity or closure claim",
    )?;
    let cost: RoventUtilityCheckpointCost =
        from_value(read_json(root, &boundary.cost, budget, held)?)?;
    cost.validate()?;
    validate_cost(&cost, projection, retirement, parent, None)?;
    check(
        cost.closure_tail.is_unknown(),
        "original running boundary must retain its unknown tail",
    )?;
    verify_cost_evidence(root, &cost, budget, held)?;
    Ok(Some(RoventUtilityInheritedUnknownTail {
        checkpoint_publication_sha256: reference.sha256.clone(),
        execution_id: closure.execution_id.clone(),
        window_id: closure.window_id.clone(),
        tail: cost.closure_tail,
    }))
}
fn verify_cost_evidence(
    root: &Path,
    cost: &RoventUtilityCheckpointCost,
    budget: &RoventRecoveryReadBudget,
    held: &mut Vec<HeldHistoricalFile>,
) -> Result<(), ArenaError> {
    let mut evidence = Vec::new();
    match &cost.closure_tail {
        RoventUtilityClosureTail::Known { evidence: pin, .. } => evidence.push(pin),
        RoventUtilityClosureTail::Unknown {
            evidence: Some(pin),
            ..
        } => evidence.push(pin),
        _ => {}
    }
    for tail in &cost.inherited_unknown_tails {
        if let RoventUtilityClosureTail::Unknown {
            evidence: Some(pin),
            ..
        } = &tail.tail
        {
            evidence.push(pin);
        }
    }
    for pin in evidence {
        let reference = UtilityCheckpointFile {
            path: pin.path.clone(),
            sha256: pin.sha256.clone(),
            bytes: pin.bytes,
        };
        let item = budget
            .read(&reference.absolute(root)?, MAX_METADATA_BYTES, false)?
            .0;
        held.push(item);
    }
    Ok(())
}
fn validate_closure(
    root: &Path,
    publication: &UtilityCheckpointPublication,
    closure: &UtilityOriginalClosure,
    replay: &UtilityLedger,
    budget: &RoventRecoveryReadBudget,
    held: &mut Vec<HeldHistoricalFile>,
) -> Result<UtilityClosureClassification, ArenaError> {
    check(
        closure.schema == UTILITY_ORIGINAL_CLOSURE
            && closure.execution_id == publication.execution_id
            && closure.window_id == publication.window_id
            && closure.run_id == publication.run_id
            && closure.historical_evidence_only
            && !closure.execution_authorized
            && matches!(
                closure.child_reap.as_str(),
                "owned_child_reaped" | "unresolved"
            )
            && matches!(
                closure.physical_completion.as_str(),
                "confirmed" | "unknown"
            )
            && closure.child_identity.pid > 0
            && closure.child_identity.start_ticks > 0
            && !closure.child_identity.boot_id.is_empty()
            && closure.child_identity.boot_id.len() <= 128
            && Path::new(&closure.child_identity.cgroup_path).is_absolute(),
        "original utility closure identity/type differs",
    )?;
    match closure.kind.as_str() {
        "optimizer_boundary" => {
            check(
                closure.process_receipt.is_none()
                    && closure.terminal_receipt.is_none()
                    && closure.boundary_publication.is_none()
                    && closure.child_reap == "unresolved"
                    && closure.physical_completion == "unknown",
                "running boundary cannot assert an old process closed",
            )?;
            Ok(UtilityClosureClassification::Unresolved)
        }
        "process_terminal" => {
            let receipt = closure
                .process_receipt
                .as_ref()
                .ok_or_else(|| invalid("original process receipt absent"))?;
            let terminal = read_json(
                root,
                closure
                    .terminal_receipt
                    .as_ref()
                    .ok_or_else(|| invalid("original terminal evidence missing"))?,
                budget,
                held,
            )?;
            fields(
                &terminal,
                &[
                    "schema",
                    "execution_id",
                    "window_id",
                    "run_id",
                    "end_payload",
                    "process_receipt",
                    "child_identity",
                    "original_child_terminal",
                    "historical_evidence_only",
                    "execution_authorized",
                ],
            )?;
            check(
                terminal["schema"] == TERMINAL
                    && terminal["execution_id"] == closure.execution_id
                    && terminal["window_id"] == closure.window_id
                    && terminal["run_id"] == closure.run_id
                    && terminal["process_receipt"] == json!(receipt)
                    && terminal["child_identity"] == json!(closure.child_identity)
                    && terminal["historical_evidence_only"] == true
                    && terminal["execution_authorized"] == false
                    && receipt.pid == closure.child_identity.pid,
                "original terminal/process/child identity differs",
            )?;
            fields(
                &terminal["end_payload"],
                &[
                    "schema",
                    "status",
                    "completed_updates",
                    "dispatched_updates",
                    "window_dispatched_updates",
                    "ancestor_updates",
                    "cost",
                    "pending_microbatches",
                    "quarantined",
                    "physical_completion",
                    "checkpoint",
                    "result",
                ],
            )?;
            check(
                terminal["end_payload"]["schema"] == TRAINER_END,
                "original utility terminal domain differs",
            )?;
            let ended = replay.ended_projection(&terminal["end_payload"])?;
            let child_ref: UtilityCheckpointFile =
                from_value(terminal["original_child_terminal"].clone())?;
            let child_terminal = read_json(root, &child_ref, budget, held)?;
            validate_child_terminal(
                &child_terminal,
                &terminal["end_payload"],
                publication,
                count(&ended["window_completed_updates"])?,
            )?;
            // Includes the original partial-source discard outcome. The new
            // restored ledger starts with no pending source buffer afterward.
            Ok(
                if matches!(receipt.group_cleanup, CleanupStatus::Gone)
                    && receipt.stop == ProcessStop::Exited
                    && receipt.exit_code == Some(0)
                    && receipt.exit_signal.is_none()
                    && closure.child_reap == "owned_child_reaped"
                    && closure.physical_completion == "confirmed"
                    && terminal["end_payload"]["physical_completion"] == "confirmed"
                    && terminal["end_payload"]["quarantined"] == false
                {
                    UtilityClosureClassification::RecordedClosed
                } else {
                    UtilityClosureClassification::Unresolved
                },
            )
        }
        _ => Err(invalid("unknown original utility closure kind")),
    }
}
fn validate_child_terminal(
    terminal: &Value,
    end: &Value,
    publication: &UtilityCheckpointPublication,
    window_completed_updates: u64,
) -> Result<(), ArenaError> {
    let mut names = vec![
        "schema",
        "execution_id",
        "window_id",
        "status",
        "result",
        "bootstrap",
        "completed_updates",
        "window_completed_updates",
        "dispatched_updates",
        "window_dispatched_updates",
        "physical_completion",
        "physical_process_retirement",
        "arena_eligible",
    ];
    if terminal.get("error_type").is_some() || terminal.get("error_chain_types").is_some() {
        names.extend(["error_type", "error_chain_types"]);
        let error = text(terminal, "error_type")?;
        check(
            !error.is_empty() && error.len() <= 256 && !error.chars().any(char::is_control),
            "bounded original child error type required",
        )?;
        let chain = terminal["error_chain_types"]
            .as_array()
            .ok_or_else(|| invalid("child error type chain must be an array"))?;
        check(
            chain.len() <= 64
                && chain.iter().all(|value| {
                    value.as_str().is_some_and(|value| {
                        !value.is_empty()
                            && value.len() <= 256
                            && !value.chars().any(char::is_control)
                    })
                }),
            "bounded original child error chain required",
        )?;
    }
    fields(terminal, &names)?;
    check(
        terminal.is_object()
            && terminal["bootstrap"].is_object()
            && terminal["result"].is_object()
            && terminal["schema"] == TRAINER_RESULT
            && terminal["execution_id"] == publication.execution_id
            && terminal["window_id"] == publication.window_id
            && terminal["completed_updates"] == end["completed_updates"]
            && count(&terminal["window_completed_updates"])? == window_completed_updates
            && terminal["dispatched_updates"] == end["dispatched_updates"]
            && terminal["window_dispatched_updates"] == end["window_dispatched_updates"]
            && terminal["physical_process_retirement"] == "original_parent_join_required"
            && terminal["physical_completion"] == "confirmed"
            && terminal["status"] == "completed"
            && terminal["arena_eligible"] == false,
        "original child terminal identity/count/result differs",
    )
}

fn validate_closure_cost(
    root: &Path,
    cost: &RoventUtilityCheckpointCost,
    retirement: &UtilityCheckpointRetirement,
    projection: &UtilityTensorProjection,
    closure: &UtilityOriginalClosure,
    budget: &RoventRecoveryReadBudget,
    held: &mut Vec<HeldHistoricalFile>,
) -> Result<(), ArenaError> {
    let post: RoventUtilityRecordedCost =
        from_value(retirement.parent_projection["post_snapshot_cost"].clone())?;
    if let Some(terminal_ref) = &closure.terminal_receipt {
        let terminal = read_json(root, terminal_ref, budget, held)?;
        let final_cost: RoventUtilityRecordedCost =
            from_value(terminal["end_payload"]["cost"].clone())?;
        fields(
            &terminal["end_payload"]["checkpoint"],
            &[
                "path",
                "sha256",
                "bytes",
                "tensor_projection",
                "completed_updates",
                "state_sha256",
                "snapshot_cost",
                "post_snapshot_cost",
            ],
        )?;
        check(
            terminal["end_payload"]["checkpoint"]["path"] == json!(retirement.checkpoint.path)
                && terminal["end_payload"]["checkpoint"]["sha256"] == retirement.checkpoint.sha256
                && terminal["end_payload"]["checkpoint"]["bytes"] == retirement.checkpoint.bytes
                && terminal["end_payload"]["checkpoint"]["tensor_projection"]
                    == retirement.original_request["tensor_projection"]
                && terminal["end_payload"]["checkpoint"]["completed_updates"]
                    == projection.completed_updates
                && terminal["end_payload"]["checkpoint"]["state_sha256"]
                    == projection.payload_state_sha256
                && terminal["end_payload"]["checkpoint"]["snapshot_cost"] == json!(projection.cost)
                && terminal["end_payload"]["checkpoint"]["post_snapshot_cost"]
                    == retirement.parent_projection["post_snapshot_cost"],
            "original terminal does not select the same completed raw checkpoint/projection",
        )?;
        let actual_delta = subtract(final_cost, post)?;
        if let RoventUtilityClosureTail::Known { delta, evidence } = &cost.closure_tail {
            check(
                evidence.path == terminal_ref.path
                    && evidence.sha256 == terminal_ref.sha256
                    && evidence.bytes == terminal_ref.bytes
                    && *delta == actual_delta,
                "known utility closure tail does not match original terminal cost evidence",
            )?;
        } else {
            return Err(invalid(
                "terminal utility closure cannot discard its measured known tail",
            ));
        }
    } else {
        check(
            cost.closure_tail.is_unknown(),
            "running utility boundary cannot claim a known terminal expense",
        )?;
    }
    Ok(())
}

/// The runtime alone constructs the borrowed source from its authenticated
/// checkpoint RPC and actual ledger boundary. There is no raw-JSON publisher.
pub(super) fn publish_current_utility_checkpoint(
    source: &ActualUtilityCheckpointBoundary<'_>,
) -> Result<PublishedUtilityCheckpointArchive, Box<UtilityCheckpointArchiveFailure>> {
    let target = source.archive_target().map_err(|error| {
        Box::new(UtilityCheckpointArchiveFailure {
            error: checkpoint_read_failure(error),
            io: None,
        })
    })?;
    let mut io = CheckpointArchiveIo::new(target);
    let result: Result<_, CheckpointReadFailure> = (|| {
        // These are borrows of the current opaque runtime owner. Selecting a
        // namespace cannot relabel another family's declaration or source.
        let registration = checkpoint_registration(source.registration())?;
        let window = checkpoint_window(source.window())?;
        let dataset = checkpoint_dataset(source.dataset())?;
        let checkpoint_source = checkpoint_parent(source.checkpoint_source())?;
        io.prepare()?;
        let boundary = source.ledger_boundary();
        let prefix = format!("runs/{}-{}", registration.execution_id, window.window_id);
        let (tensor_pin, tensor_file) = source.tensor_file();
        check(
            tensor_pin == boundary.original_tensor_pin(),
            "runtime tensor differs from actual opaque ledger boundary",
        )?;
        let tensor = copy_original(&mut io, &prefix, "tensor.pt", tensor_pin, tensor_file)?;
        let (projection_pin, projection_file) = source.tensor_projection_file();
        check(
            projection_pin == boundary.original_tensor_projection_pin(),
            "runtime projection differs from opaque ledger boundary",
        )?;
        let tensor_projection = copy_original(
            &mut io,
            &prefix,
            "tensor-projection.json",
            projection_pin,
            projection_file,
        )?;
        let (journal_pin, journal_file) = source.journal_file();
        let original_rpc_journal = copy_original(
            &mut io,
            &prefix,
            "rpc-journal.jsonl",
            journal_pin,
            journal_file,
        )?;
        let (registration_pin, registration_file) = source.registration_file();
        let original_registration = copy_original(
            &mut io,
            &prefix,
            "registration.json",
            registration_pin,
            registration_file,
        )?;
        let find_input = |pin: &RoventTrainingFilePin| -> Result<&File, ArenaError> {
            source
                .held_files()
                .find(|(actual, _)| *actual == pin)
                .map(|(_, file)| file)
                .ok_or_else(|| invalid("actual runtime lost an original declared input descriptor"))
        };
        let original_preparation = copy_original(
            &mut io,
            &prefix,
            "preparation.json",
            &registration.preparation,
            find_input(&registration.preparation)?,
        )?;
        let original_window = copy_original(
            &mut io,
            &prefix,
            "window.json",
            &registration.window_spec,
            find_input(&registration.window_spec)?,
        )?;
        let mut completion_receipts = Vec::with_capacity(boundary.completion_receipts().len());
        for pin in boundary.completion_receipts() {
            let file = source
                .held_files()
                .find(|(actual, _)| actual.sha256 == pin.sha256 && actual.bytes == pin.bytes)
                .map(|(_, file)| file);
            let name = format!("completions/{}.json", pin.sha256);
            let reference = if let Some(file) = file {
                io.copy_file(&name, pin, file)?
            } else {
                let (held, _) = source.read_budget().read(pin, MAX_METADATA_BYTES, false)?;
                io.copy_file(&name, &held.pin, &held.file)?
            };
            completion_receipts.push(UtilityCheckpointFile::from_common(reference));
        }
        let parent_publication = checkpoint_source
            .map(|parent| {
                copy_history(
                    &mut io,
                    parent,
                    &format!("parents/{}.json", parent.publication_pin.sha256),
                    0,
                )
            })
            .transpose()?;
        let mut ancestor_completion_receipts =
            Vec::with_capacity(boundary.ancestor_completion_sha256().len());
        for sha in boundary.ancestor_completion_sha256() {
            let parent = checkpoint_source.ok_or_else(|| {
                invalid("actual ancestor completion lacks original checkpoint owner")
            })?;
            let pin = parent
                .ancestor_receipt_pins
                .iter()
                .chain(&parent.receipt_pins)
                .find(|pin| &pin.sha256 == sha)
                .ok_or_else(|| {
                    invalid("actual ancestor completion absent from retained original full chain")
                })?;
            let (held, _) = source.read_budget().read(pin, MAX_METADATA_BYTES, false)?;
            let reference =
                io.copy_file(&format!("completions/{sha}.json"), &held.pin, &held.file)?;
            ancestor_completion_receipts.push(UtilityCheckpointFile::from_common(reference));
        }
        let initial_ledger_state = write_json(
            &mut io,
            &format!("{prefix}/initial-ledger.json"),
            &boundary.initial_ledger_projection(),
        )?;
        let cost = RoventUtilityCheckpointCost::from_recorded_parts(
            boundary.ancestor_cost(),
            boundary.snapshot_cost(),
            subtract(boundary.post_snapshot_cost(), boundary.snapshot_cost())?,
            RoventUtilityClosureTail::Unknown {
                reason: "original_process_still_running".into(),
                evidence: None,
            },
            inherited_tails(checkpoint_source.map(Arc::as_ref))?,
        )?;
        let cost_file = write_json(&mut io, &format!("{prefix}/cost.json"), &cost)?;
        let closure = UtilityOriginalClosure {
            schema: UTILITY_ORIGINAL_CLOSURE.into(),
            execution_id: registration.execution_id.clone(),
            window_id: window.window_id.clone(),
            run_id: window.run_id.clone(),
            kind: "optimizer_boundary".into(),
            child_identity: source.child_identity()?,
            process_receipt: None,
            terminal_receipt: None,
            boundary_publication: None,
            child_reap: "unresolved".into(),
            physical_completion: "unknown".into(),
            historical_evidence_only: true,
            execution_authorized: false,
        };
        let closure_file = write_json(&mut io, &format!("{prefix}/closure.json"), &closure)?;
        let original_request = source.checkpoint_request().clone();
        let retirement = UtilityCheckpointRetirement {
            schema: UTILITY_CHECKPOINT_RETIREMENT.into(),
            checkpoint: tensor_pin.clone(),
            original_request,
            parent_projection: boundary.projection().clone(),
            tensor_projection_sha256: projection_pin.sha256.clone(),
            historical_evidence_only: true,
            execution_authorized: false,
        };
        let retirement_file =
            write_json(&mut io, &format!("{prefix}/retirement.json"), &retirement)?;
        let publication = UtilityCheckpointPublication {
            schema: UTILITY_CHECKPOINT_PUBLICATION.into(),
            execution_id: registration.execution_id.clone(),
            window_id: window.window_id.clone(),
            run_id: window.run_id.clone(),
            preparation_sha256: window.preparation_sha256.clone(),
            dataset_sha256: dataset.dataset_sha256().into(),
            window_spec_sha256: window.canonical_sha256()?,
            trainer_source_sha256: registration.trainer_source_sha256.clone(),
            trainer_python_executable_sha256: registration.trainer_python_executable.sha256.clone(),
            trainer_dependency_set_sha256: registration.trainer_dependency_set_sha256.clone(),
            tensor,
            retirement: retirement_file,
            tensor_projection,
            initial_ledger_state,
            original_preparation,
            original_window,
            original_registration,
            original_rpc_journal,
            completion_receipts,
            ancestor_completion_receipts,
            parent_publication,
            cost: cost_file,
            closure: closure_file,
            historical_evidence_only: true,
            execution_authorized: false,
        };
        publication.validate()?;
        let pending = write_json(&mut io, "publication.pending.json", &publication)?;
        let mut refs = vec![registration.current_dataset.clone()];
        refs.extend(checkpoint_ancestor_references(source.ancestors()));
        if let Some(parent) = checkpoint_source {
            refs.extend(
                parent
                    .datasets()
                    .iter()
                    .map(|dataset| dataset.reference().clone()),
            );
        }
        refs.sort_by(|a, b| a.publication.sha256.cmp(&b.publication.sha256));
        refs.dedup_by(|a, b| a == b);
        let reference = RoventUtilityCheckpointReference {
            root: io.root().to_path_buf(),
            publication: io.absolute(&pending.common())?,
        };
        // Re-open the destination and replay the complete new family before
        // its final name becomes visible. No tensor execution occurs here.
        io.named_root()?;
        let verified = VerifiedUtilityCheckpointSource::open(
            &reference,
            &refs,
            &registration.ancestor_datasets,
            io.read_budget(),
        )?;
        check(
            serde_json::to_value(verified.publication())
                .map_err(|error| invalid(error.to_string()))?
                == serde_json::to_value(&publication)
                    .map_err(|error| invalid(error.to_string()))?,
            "utility destination readback changed publication",
        )?;
        let pin = io.commit_publication(&pending.common())?;
        let verified = committed_source(verified, &pin)?;
        Ok((publication, pin, Arc::new(verified)))
    })();
    match result {
        Ok((publication, pin, verified_source)) => Ok(PublishedUtilityCheckpointArchive {
            publication,
            pin,
            io,
            verified_source,
        }),
        Err(error) => Err(Box::new(UtilityCheckpointArchiveFailure {
            error,
            io: Some(io),
        })),
    }
}

/// Only an actual normal-close borrow can add the original supervisor/terminal
/// evidence. It preserves the selected raw family and its unknown boundary,
/// while charging the measured terminal delta exactly once in a new sidecar.
pub(super) fn publish_closed_utility_checkpoint(
    source: &ActualUtilityClosedCheckpoint<'_>,
) -> Result<PublishedUtilityCheckpointArchive, Box<UtilityCheckpointArchiveFailure>> {
    let target = source.archive_target().map_err(|error| {
        Box::new(UtilityCheckpointArchiveFailure {
            error: checkpoint_read_failure(error),
            io: None,
        })
    })?;
    let mut io = CheckpointArchiveIo::new(target);
    let result: Result<_, CheckpointReadFailure> = (|| {
        let latest = checkpoint_archive(source.latest_archive())?;
        let registration = checkpoint_registration(source.registration())?;
        let window = checkpoint_window(source.window())?;
        let dataset = checkpoint_dataset(source.dataset())?;
        latest.revalidate(source.read_budget())?;
        let original = &latest.verified_source;
        original.revalidate(source.read_budget())?;
        check(
            original.classification == UtilityClosureClassification::Unresolved
                && original.closure.kind == "optimizer_boundary"
                && original.publication.execution_id == registration.execution_id
                && original.window.canonical_sha256()? == window.canonical_sha256()?
                && original.preparation == *source.preparation()
                && original.dataset.reference() == dataset.reference()
                && original.ancestors.publication_sha256s()
                    == source.ancestors().publication_sha256s(),
            "normal-close borrow differs from its actual latest running archive",
        )?;
        source.process().checked_timing()?;
        source.process().checked_loaded_image()?;
        check(
            matches!(
                source.process().process().receipt.group_cleanup,
                CleanupStatus::Gone
            ) && source.process().process().receipt.pid == source.child_identity()?.pid,
            "normal-close original process/identity proof differs",
        )?;
        let ended = source.ledger().ended_projection(source.original_end())?;
        let end = source.original_end();
        check(
            count(&end["completed_updates"])? == original.projection.completed_updates
                && count(&end["dispatched_updates"])? == original.projection.dispatched_updates
                && count(&end["ancestor_updates"])? == original.projection.ancestor_updates
                && end["physical_completion"] == "confirmed"
                && end["quarantined"] == false,
            "selected checkpoint omits later completed/dispatched work or physical closure",
        )?;
        let mut checked_files = Vec::new();
        let retirement: UtilityCheckpointRetirement = from_value(read_json(
            &original.root,
            &original.publication.retirement,
            source.read_budget(),
            &mut checked_files,
        )?)?;
        let projection_pin: RoventTrainingFilePin =
            from_value(retirement.original_request["tensor_projection"].clone())?;
        check(
            source.ledger().checkpoint_projection(
                &retirement.original_request,
                &retirement.checkpoint,
                &projection_pin,
            )? == retirement.parent_projection
                && original.projection.rng_state_sha256
                    == source.ledger().historical_rng_state_sha256()
                && original.projection.dispatched_updates
                    == source.ledger().historical_dispatched_updates(),
            "normal-close current ledger five-state/sampler/chain changed after selected boundary",
        )?;
        let (terminal_pin, terminal_file) = source.terminal_file();
        source
            .read_budget()
            .verify_held(terminal_file, terminal_pin, false)?;
        let (_, raw) = source
            .read_budget()
            .read(terminal_pin, MAX_METADATA_BYTES, true)?;
        let child_terminal: Value = decode(&raw)?;
        validate_child_terminal(
            &child_terminal,
            end,
            &original.publication,
            count(&ended["window_completed_updates"])?,
        )?;
        io.prepare()?;
        let boundary_publication = copy_history(
            &mut io,
            original,
            &format!("boundaries/{}.json", latest.pin.sha256),
            0,
        )?;
        let prefix = format!(
            "closed/{}-{}",
            original.publication.execution_id, original.publication.window_id
        );
        let original_child_terminal = copy_original(
            &mut io,
            &prefix,
            "child-terminal.json",
            terminal_pin,
            terminal_file,
        )?;
        let identity = source.child_identity()?;
        check(
            json!(identity) == json!(original.closure.child_identity),
            "normal-close child differs from selected original peer",
        )?;
        let terminal = json!({"schema":TERMINAL,"execution_id":original.publication.execution_id,
            "window_id":original.publication.window_id,"run_id":original.publication.run_id,
            "end_payload":end,"process_receipt":source.process().process().receipt,
            "child_identity":identity,"original_child_terminal":original_child_terminal,
            "historical_evidence_only":true,"execution_authorized":false});
        let terminal_ref = write_json(
            &mut io,
            &format!("{prefix}/parent-terminal.json"),
            &terminal,
        )?;
        let mut inherited = original.cost.inherited_unknown_tails.clone();
        check(
            original.cost.closure_tail.is_unknown(),
            "selected running boundary lost original unknown tail",
        )?;
        inherited.push(RoventUtilityInheritedUnknownTail {
            checkpoint_publication_sha256: boundary_publication.sha256.clone(),
            execution_id: original.publication.execution_id.clone(),
            window_id: original.publication.window_id.clone(),
            tail: original.cost.closure_tail.clone(),
        });
        sort_unknown_tails(&mut inherited)?;
        let post: RoventUtilityRecordedCost =
            from_value(retirement.parent_projection["post_snapshot_cost"].clone())?;
        let end_cost: RoventUtilityRecordedCost = from_value(end["cost"].clone())?;
        let cost = RoventUtilityCheckpointCost::from_recorded_parts(
            original.cost.ancestor_cost,
            original.cost.snapshot_cost,
            original.cost.post_snapshot_delta,
            RoventUtilityClosureTail::Known {
                delta: subtract(end_cost, post)?,
                evidence: RoventUtilityCostEvidence {
                    path: terminal_ref.path.clone(),
                    sha256: terminal_ref.sha256.clone(),
                    bytes: terminal_ref.bytes,
                },
            },
            inherited,
        )?;
        let cost_ref = write_json(&mut io, &format!("{prefix}/cost.json"), &cost)?;
        let closure = UtilityOriginalClosure {
            schema: UTILITY_ORIGINAL_CLOSURE.into(),
            execution_id: original.publication.execution_id.clone(),
            window_id: original.publication.window_id.clone(),
            run_id: original.publication.run_id.clone(),
            kind: "process_terminal".into(),
            child_identity: identity,
            process_receipt: Some(source.process().process().receipt.clone()),
            terminal_receipt: Some(terminal_ref),
            boundary_publication: Some(boundary_publication),
            child_reap: "owned_child_reaped".into(),
            physical_completion: "confirmed".into(),
            historical_evidence_only: true,
            execution_authorized: false,
        };
        let closure_ref = write_json(&mut io, &format!("{prefix}/closure.json"), &closure)?;
        let mut publication = original.publication.clone();
        publication.cost = cost_ref;
        publication.closure = closure_ref;
        publication.validate()?;
        let pending = write_json(&mut io, "publication.pending.json", &publication)?;
        let reference = RoventUtilityCheckpointReference {
            root: io.root().to_path_buf(),
            publication: io.absolute(&pending.common())?,
        };
        let references = original
            .datasets()
            .iter()
            .map(|dataset| dataset.reference().clone())
            .collect::<Vec<_>>();
        io.named_root()?;
        let verified = VerifiedUtilityCheckpointSource::open(
            &reference,
            &references,
            &registration.ancestor_datasets,
            io.read_budget(),
        )?;
        check(
            verified.classification == UtilityClosureClassification::RecordedClosed,
            "actual closed publication did not verify its terminal family",
        )?;
        let pin = io.commit_publication(&pending.common())?;
        Ok((
            publication,
            pin.clone(),
            Arc::new(committed_source(verified, &pin)?),
        ))
    })();
    match result {
        Ok((publication, pin, verified_source)) => Ok(PublishedUtilityCheckpointArchive {
            publication,
            pin,
            io,
            verified_source,
        }),
        Err(error) => Err(Box::new(UtilityCheckpointArchiveFailure {
            error,
            io: Some(io),
        })),
    }
}
fn committed_source(
    mut source: VerifiedUtilityCheckpointSource,
    pin: &RoventTrainingFilePin,
) -> Result<VerifiedUtilityCheckpointSource, ArenaError> {
    // The commit primitive itself verified the final name against this same
    // original descriptor; this changes no bytes and performs no new admission.
    let held = source
        .held
        .first_mut()
        .ok_or_else(|| invalid("verified publication descriptor missing"))?;
    check(
        held.pin.sha256 == pin.sha256 && held.pin.bytes == pin.bytes,
        "committed publication changed original verified bytes",
    )?;
    held.pin = pin.clone();
    source.publication_pin = pin.clone();
    Ok(source)
}

pub struct PublishedUtilityCheckpointArchive {
    publication: UtilityCheckpointPublication,
    pin: RoventTrainingFilePin,
    io: CheckpointArchiveIo,
    verified_source: Arc<VerifiedUtilityCheckpointSource>,
}
impl PublishedUtilityCheckpointArchive {
    pub fn reserved_output_bytes(&self) -> u64 {
        self.io.reserved_output_bytes()
    }
    pub fn publication(&self) -> &UtilityCheckpointPublication {
        &self.publication
    }
    pub fn publication_pin(&self) -> &RoventTrainingFilePin {
        &self.pin
    }
    pub fn observation(&self) -> Value {
        json!({"schema":ARCHIVE_OBSERVATION,"publication":self.pin,"io":self.io.observation(true),"tensor_body_scratch_validated":false,"past_native_authority_reissued":false,"past_physical_completion_reissued":false,"execution_authorized":false})
    }
    pub fn revalidate(&self, budget: &RoventRecoveryReadBudget) -> Result<(), ArenaError> {
        self.io.revalidate(budget)?;
        self.verified_source.revalidate(budget)
    }
}
pub struct UtilityCheckpointArchiveFailure {
    pub error: CheckpointReadFailure,
    io: Option<CheckpointArchiveIo>,
}
impl std::fmt::Debug for UtilityCheckpointArchiveFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UtilityCheckpointArchiveFailure")
            .field("error", &self.error)
            .field("observation", &self.observation())
            .finish_non_exhaustive()
    }
}
impl UtilityCheckpointArchiveFailure {
    pub fn observation(&self) -> Value {
        json!({"schema":ARCHIVE_FAILURE,"io":self.io.as_ref().map(|io|io.observation(false)),"failure_files_and_fds_retained":self.io.is_some(),"publication_acknowledged":false,"commit_state":"not_acknowledged_or_unknown","execution_authorized":false})
    }
}
fn write_json(
    io: &mut CheckpointArchiveIo,
    path: &str,
    value: &impl Serialize,
) -> Result<UtilityCheckpointFile, ArenaError> {
    let mut object = serde_json::to_value(value).map_err(|error| invalid(error.to_string()))?;
    object.sort_all_objects();
    let mut raw = BoundedMetadataBytes(Vec::new());
    serde_json::to_writer(&mut raw, &object).map_err(|error| invalid(error.to_string()))?;
    io.write_bytes(path, &raw.0)
        .map(UtilityCheckpointFile::from_common)
}
/// Bounded metadata only; tensor and journal bodies use held-descriptor copy.
/// This is a byte allowance, never a process-memory or host-peak measurement.
struct BoundedMetadataBytes(Vec<u8>);
impl std::io::Write for BoundedMetadataBytes {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        let required = self
            .0
            .len()
            .checked_add(bytes.len())
            .filter(|n| *n <= MAX_METADATA_BYTES as usize)
            .ok_or_else(|| {
                std::io::Error::other("utility generated metadata allowance exceeded")
            })?;
        if required > self.0.capacity() {
            let capacity = required
                .next_power_of_two()
                .min(MAX_METADATA_BYTES as usize);
            self.0
                .try_reserve_exact(capacity - self.0.len())
                .map_err(|_| std::io::Error::other("utility bounded metadata allocation failed"))?;
        }
        self.0.extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}
fn copy_original(
    io: &mut CheckpointArchiveIo,
    prefix: &str,
    name: &str,
    pin: &RoventTrainingFilePin,
    file: &File,
) -> Result<UtilityCheckpointFile, ArenaError> {
    io.copy_file(&format!("{prefix}/{name}"), pin, file)
        .map(UtilityCheckpointFile::from_common)
}
fn copy_history(
    io: &mut CheckpointArchiveIo,
    source: &VerifiedUtilityCheckpointSource,
    relative_publication: &str,
    depth: usize,
) -> Result<UtilityCheckpointFile, ArenaError> {
    check(
        depth < MAX_CHAIN,
        "utility portable ancestry depth exceeded",
    )?;
    source.revalidate(io.read_budget())?;
    let publication = source
        .held
        .first()
        .ok_or_else(|| invalid("source publication FD missing"))?;
    let copied = io.copy_file(relative_publication, &publication.pin, &publication.file)?;
    for held in std::iter::once(&source.tensor).chain(source.held.iter().skip(1)) {
        let relative = held
            .pin
            .path
            .strip_prefix(&source.root)
            .map_err(|_| invalid("historical original file escapes source root"))?
            .to_str()
            .ok_or_else(|| invalid("historical relative original encoding"))?;
        io.copy_file(relative, &held.pin, &held.file)?;
    }
    for pin in source
        .receipt_pins
        .iter()
        .chain(&source.ancestor_receipt_pins)
    {
        let (held, _) = io.read_budget().read(pin, MAX_METADATA_BYTES, false)?;
        let relative = pin
            .path
            .strip_prefix(&source.root)
            .map_err(|_| invalid("historical completion escapes original root"))?
            .to_str()
            .ok_or_else(|| invalid("historical completion path encoding"))?;
        io.copy_file(relative, &held.pin, &held.file)?;
    }
    if let Some(parent) = &source.parent {
        let reference = source
            .publication
            .parent_publication
            .as_ref()
            .ok_or_else(|| invalid("original parent owner lacks declaration"))?;
        let actual = copy_history(io, parent, &reference.path, depth + 1)?;
        check(actual == *reference, "copied utility parent bytes changed")?;
    }
    Ok(UtilityCheckpointFile::from_common(copied))
}
