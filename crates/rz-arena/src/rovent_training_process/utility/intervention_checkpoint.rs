//! Explicit intervention-v2 checkpoint history and commit-last publication. Retained raw
//! files and complete historical journals never recreate old native, process,
//! physical-completion or current tensor execution authority. The actual child
//! must still validate the full .pt in scratch before its checkpoint-checked
//! admission can begin a new numerical window.

use super::ancestry::VerifiedUtilityAncestors;
use super::cost::{
    RoventUtilityCheckpointCost, RoventUtilityClosureTail, RoventUtilityCostEvidence,
    RoventUtilityInheritedUnknownTail, RoventUtilityRecordedCost,
};
use super::execution::{
    ROVENT_UTILITY_MAX_UPDATES, RoventUtilityCheckpointReference, RoventUtilityDatasetReference,
    RoventUtilityInitializationSource,
};
use super::intervention_dataset::VerifiedUtilityInterventionDataset as VerifiedUtilityDataset;
use super::intervention_execution::{
    RoventUtilityInterventionAncestorRegistration as RoventUtilityAncestorDatasetRegistration,
    RoventUtilityInterventionExecutionRegistration as RoventUtilityExecutionRegistrationV2,
};
use super::intervention_window::RoventUtilityInterventionTrainWindow as RoventUtilityTrainWindow;
use super::ledger::{RestoredUtilityState, UtilityLedger, UtilityStateSeals};
use super::runtime::{ActualUtilityCheckpointBoundary, ActualUtilityClosedCheckpoint};
use crate::ArenaError;
use crate::process::{CleanupStatus, ProcessReceipt, ProcessStop};
use crate::rovent_collect::RoventPublicationFile;
use crate::rovent_training_process::RoventTrainingFilePin;
use crate::rovent_training_process::recovery::{
    CheckpointArchiveIo, HeldHistoricalFile, RoventHistoricalChildIdentity,
    RoventRecoveryReadBudget, VerifiedHistoricalUnresolvedScopes,
};
use rz_experiments::pals_data_v3::{generation_digest, generation_identifier, generation_sha};
use serde::de::{MapAccess, Visitor, value::MapAccessDeserializer};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::os::unix::fs::FileExt;
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;

pub const UTILITY_CHECKPOINT_PUBLICATION: &str = "rz-rovent-utility-checkpoint-publication/2";
pub const UTILITY_CHECKPOINT_RETIREMENT: &str = "rz-rovent-utility-checkpoint-retirement/2";
pub const UTILITY_TENSOR_PROJECTION: &str = "rz-rovent-utility-tensor-projection/2";
pub const UTILITY_INITIAL_LEDGER: &str = "rz-rovent-utility-initial-ledger-projection/2";
pub const UTILITY_ORIGINAL_CLOSURE: &str = "rz-rovent-utility-original-training-closure/2";
const TERMINAL: &str = "rz-rovent-utility-parent-terminal-receipt/2";
const JOURNAL: &str = "rz-rovent-utility-ledger-event/2";
const DERIVED: &str = "rz-rovent-utility-derived-training-state/2";
const OLD_RUN_RECLAMATION: &str = "rz-rovent-utility-old-run-reclamation/2";
const TRAINER_END: &str = "rz-rovent-utility-trainer-end/2";
const TRAINER_RESULT: &str = "rz-rovent-utility-trainer-result/2";
const ARCHIVE_OBSERVATION: &str = "rz-rovent-utility-checkpoint-archive-observation/2";
const ARCHIVE_FAILURE: &str = "rz-rovent-utility-checkpoint-archive-failure/2";
const MAX_METADATA_BYTES: u64 = 16 << 20;
const MAX_TENSOR_BYTES: u64 = 2 << 30;
const MAX_JOURNAL_BYTES: u64 = 2 << 30;
const MAX_JOURNAL_EVENT_BYTES: usize = 256 << 10;
const MAX_CHAIN: usize = 64;

/// A failed new-family source keeps any original Dataset/ancestor descriptors
/// returned by its reader. It is neither a source admission nor execution
/// authority and is never flattened into a success or a string-only error.
#[derive(Debug)]
pub enum UtilityCheckpointReadFailure {
    Metadata(ArenaError),
    Dataset(Box<super::intervention_dataset::RoventUtilityInterventionDatasetRejected>),
    Ancestors(Box<super::ancestry::RoventUtilityInterventionAncestorsRejected>),
}
impl UtilityCheckpointReadFailure {
    pub fn error(&self) -> &ArenaError {
        match self {
            Self::Metadata(error) => error,
            Self::Dataset(rejected) => &rejected.error,
            Self::Ancestors(rejected) => &rejected.error,
        }
    }
}
impl From<ArenaError> for UtilityCheckpointReadFailure {
    fn from(error: ArenaError) -> Self {
        Self::Metadata(error)
    }
}
impl std::fmt::Display for UtilityCheckpointReadFailure {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.error().fmt(formatter)
    }
}
impl std::error::Error for UtilityCheckpointReadFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Metadata(error) => Some(error),
            Self::Dataset(rejected) => Some(rejected.as_ref()),
            Self::Ancestors(rejected) => Some(rejected.as_ref()),
        }
    }
}
type CheckpointReadFailure = UtilityCheckpointReadFailure;

fn checkpoint_read_failure(error: ArenaError) -> CheckpointReadFailure {
    UtilityCheckpointReadFailure::Metadata(error)
}

fn open_checkpoint_dataset(
    reference: &RoventUtilityDatasetReference,
    preparation: &Value,
    budget: &Arc<RoventRecoveryReadBudget>,
) -> Result<VerifiedUtilityDataset, CheckpointReadFailure> {
    VerifiedUtilityDataset::open_retaining(reference, preparation, budget)
        .map_err(UtilityCheckpointReadFailure::Dataset)
}

fn validate_checkpoint_preparation(preparation: &Value) -> Result<(), ArenaError> {
    super::numerical_parts::ValidatedUtilityNumericalParts::from_intervention(preparation)?;
    Ok(())
}

fn checkpoint_preparation_seals(preparation: &Value) -> Result<(String, String), ArenaError> {
    let parts =
        super::numerical_parts::ValidatedUtilityNumericalParts::from_intervention(preparation)?;
    Ok((
        parts.recipe_sha256().to_owned(),
        parts.source_registration_sha256().to_owned(),
    ))
}

fn open_checkpoint_ancestors(
    registrations: &[RoventUtilityAncestorDatasetRegistration],
    budget: &Arc<RoventRecoveryReadBudget>,
) -> Result<VerifiedUtilityAncestors, CheckpointReadFailure> {
    VerifiedUtilityAncestors::open_intervention(registrations, budget)
        .map_err(UtilityCheckpointReadFailure::Ancestors)
}

fn checkpoint_utility_datasets(
    ancestors: &VerifiedUtilityAncestors,
) -> &[Arc<VerifiedUtilityDataset>] {
    ancestors.intervention_datasets()
}

fn checkpoint_ancestor_references(
    ancestors: &VerifiedUtilityAncestors,
) -> Vec<RoventUtilityDatasetReference> {
    ancestors.intervention_references()
}

fn historical_checkpoint_ledger(
    window: RoventUtilityTrainWindow,
    preparation: Value,
    dataset: Arc<VerifiedUtilityDataset>,
    restored: RestoredUtilityState,
    scope: &str,
) -> Result<UtilityLedger, ArenaError> {
    UtilityLedger::for_historical_replay(
        window.into(),
        preparation,
        dataset.into(),
        restored,
        scope,
    )
}

fn checkpoint_registration(
    registration: &super::runtime_family::RuntimeUtilityRegistration,
) -> Result<&RoventUtilityExecutionRegistrationV2, ArenaError> {
    registration.as_intervention()
}

fn checkpoint_window(
    window: &super::runtime_family::RuntimeUtilityWindow,
) -> Result<&RoventUtilityTrainWindow, ArenaError> {
    window.as_intervention()
}

fn checkpoint_dataset(
    dataset: &super::runtime_family::RuntimeUtilityDataset,
) -> Result<&Arc<VerifiedUtilityDataset>, ArenaError> {
    dataset.as_intervention()
}

fn checkpoint_parent(
    source: Option<&super::runtime_family::RuntimeUtilityCheckpointSource>,
) -> Result<Option<&Arc<VerifiedUtilityCheckpointSource>>, ArenaError> {
    source.map(|source| source.as_intervention()).transpose()
}

fn checkpoint_archive(
    archive: &super::runtime_family::RuntimePublishedUtilityCheckpointArchive,
) -> Result<&PublishedUtilityCheckpointArchive, ArenaError> {
    archive.as_intervention()
}

/// Relocation is the only difference permitted between the original source
/// declaration and a caller-held copy. Legacy arms retain their original
/// validators; an intervention arm never becomes a utility-cohort ancestor.
fn same_ancestor_semantics(
    original: &RoventUtilityAncestorDatasetRegistration,
    caller: &RoventUtilityAncestorDatasetRegistration,
) -> bool {
    use RoventUtilityAncestorDatasetRegistration as Kind;
    let mut normalized = caller.clone();
    match (original, &mut normalized) {
        (Kind::Legacy(old), Kind::Legacy(new)) => {
            return super::checkpoint::same_ancestor_semantics(old, new);
        }
        (
            Kind::Intervention {
                registration: old,
                preparation: old_preparation,
            },
            Kind::Intervention {
                registration: new,
                preparation: new_preparation,
            },
        ) => {
            new.root.clone_from(&old.root);
            new.publication.path.clone_from(&old.publication.path);
            new.durable_ack.path.clone_from(&old.durable_ack.path);
            new_preparation.path.clone_from(&old_preparation.path);
        }
        _ => return false,
    }
    original == &normalized
}

fn validate_current_dataset_ancestor(
    caller: &RoventUtilityAncestorDatasetRegistration,
    reference: &RoventUtilityDatasetReference,
    original_preparation: &UtilityCheckpointFile,
) -> Result<(), ArenaError> {
    let (declared, preparation) = caller.intervention().ok_or_else(|| {
        invalid("source-current intervention dataset was relabelled as another family")
    })?;
    check(
        declared.publication.sha256 == reference.publication.sha256
            && declared.publication.bytes == reference.publication.bytes
            && declared.durable_ack.sha256 == reference.durable_ack.sha256
            && declared.durable_ack.bytes == reference.durable_ack.bytes
            && preparation.sha256 == original_preparation.sha256
            && preparation.bytes == original_preparation.bytes,
        "caller source-current intervention ancestor changed its own original preparation/raw ACK",
    )
}

// The original FD, strict decoder, journal replay, five-state, cost, closure,
// and commit-last implementation is shared. Its public types are instantiated
// in this separate namespace and accept only the explicit family above.
include!("checkpoint/machine.rs");

#[cfg(test)]
mod tests {
    // Metadata operands only: these checks admit no files, tensor, native
    // process, completion authority or optimizer execution.
    use super::*;

    fn pin(name: &str) -> UtilityCheckpointFile {
        UtilityCheckpointFile {
            path: format!("test/{name}"),
            sha256: "ab".repeat(32),
            bytes: 1,
        }
    }

    fn publication() -> UtilityCheckpointPublication {
        UtilityCheckpointPublication {
            schema: UTILITY_CHECKPOINT_PUBLICATION.into(),
            execution_id: "test-intervention".into(),
            window_id: "test-window".into(),
            run_id: "test-run".into(),
            preparation_sha256: "ab".repeat(32),
            dataset_sha256: "bc".repeat(32),
            window_spec_sha256: "cd".repeat(32),
            trainer_source_sha256: "de".repeat(32),
            trainer_python_executable_sha256: "ef".repeat(32),
            trainer_dependency_set_sha256: "fa".repeat(32),
            tensor: pin("tensor.pt"),
            retirement: pin("retirement.json"),
            tensor_projection: pin("tensor-projection.json"),
            initial_ledger_state: pin("initial-ledger.json"),
            original_preparation: pin("preparation.json"),
            original_window: pin("window.json"),
            original_registration: pin("registration.json"),
            original_rpc_journal: pin("journal.jsonl"),
            completion_receipts: vec![],
            ancestor_completion_receipts: vec![],
            parent_publication: None,
            cost: pin("cost.json"),
            closure: pin("closure.json"),
            historical_evidence_only: true,
            execution_authorized: false,
        }
    }

    #[test]
    fn publication_has_an_exact_separate_domain_and_original_shape() {
        let publication = publication();
        publication.validate().unwrap();
        let mut value = serde_json::to_value(&publication).unwrap();
        value["schema"] = json!(super::super::checkpoint::UTILITY_CHECKPOINT_PUBLICATION);
        assert!(
            from_value::<UtilityCheckpointPublication>(value.clone())
                .unwrap()
                .validate()
                .is_err()
        );
        value["schema"] = json!(UTILITY_CHECKPOINT_PUBLICATION);
        value["data_family"] = json!("intervention_v2");
        assert!(from_value::<UtilityCheckpointPublication>(value).is_err());
        let raw = serde_json::to_string(&publication).unwrap();
        let duplicate = raw.replacen(
            "{\"schema\":",
            "{\"schema\":\"rz-rovent-utility-checkpoint-publication/1\",\"schema\":",
            1,
        );
        assert!(decode::<UtilityCheckpointPublication>(duplicate.as_bytes()).is_err());
    }

    fn intervention_ancestor(root: &str) -> RoventUtilityAncestorDatasetRegistration {
        let absolute = |name: &str| RoventTrainingFilePin {
            path: PathBuf::from(root).join(name),
            sha256: "ab".repeat(32),
            bytes: 1,
        };
        RoventUtilityAncestorDatasetRegistration::Intervention {
            registration: RoventUtilityDatasetReference {
                root: root.into(),
                publication: absolute("publication.json"),
                durable_ack: absolute("durable-ack.json"),
            },
            preparation: absolute("preparation.json"),
        }
    }

    #[test]
    fn ancestor_relocation_preserves_raw_publication_ack_and_preparation() {
        let original = intervention_ancestor("/test/original");
        let relocated = intervention_ancestor("/test/relocated");
        assert!(same_ancestor_semantics(&original, &relocated));
        let mut altered = relocated.clone();
        let RoventUtilityAncestorDatasetRegistration::Intervention { preparation, .. } =
            &mut altered
        else {
            unreachable!()
        };
        preparation.sha256 = "cd".repeat(32);
        assert!(!same_ancestor_semantics(&original, &altered));
        let mut altered = relocated;
        let RoventUtilityAncestorDatasetRegistration::Intervention { registration, .. } =
            &mut altered
        else {
            unreachable!()
        };
        registration.durable_ack.bytes = 2;
        assert!(!same_ancestor_semantics(&original, &altered));
    }
}
