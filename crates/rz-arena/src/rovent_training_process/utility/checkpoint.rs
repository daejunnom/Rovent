//! Utility-only checkpoint history and commit-last publication. Retained raw
//! files and complete historical journals never recreate old native, process,
//! physical-completion or current tensor execution authority. The actual child
//! must still validate the full .pt in scratch before its checkpoint-checked
//! admission can begin a new numerical window.

use super::ancestry::VerifiedUtilityAncestors;
use super::cost::{
    RoventUtilityCheckpointCost, RoventUtilityClosureTail, RoventUtilityCostEvidence,
    RoventUtilityInheritedUnknownTail, RoventUtilityRecordedCost,
};
use super::dataset::VerifiedUtilityDataset;
use super::execution::{
    ROVENT_UTILITY_MAX_UPDATES, RoventUtilityCheckpointReference, RoventUtilityDatasetReference,
    RoventUtilityInitializationSource, validate_utility_training_preparation,
};
use super::execution_v2::{
    RoventUtilityAncestorDatasetRegistration, RoventUtilityExecutionRegistrationV2,
};
use super::ledger::{RestoredUtilityState, UtilityLedger, UtilityStateSeals};
use super::runtime::{ActualUtilityCheckpointBoundary, ActualUtilityClosedCheckpoint};
use super::window_v2::RoventUtilityTrainWindowV2 as RoventUtilityTrainWindow;
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

pub const UTILITY_CHECKPOINT_PUBLICATION: &str = "rz-rovent-utility-checkpoint-publication/1";
pub const UTILITY_CHECKPOINT_RETIREMENT: &str = "rz-rovent-utility-checkpoint-retirement/1";
pub const UTILITY_TENSOR_PROJECTION: &str = "rz-rovent-utility-tensor-projection/1";
pub const UTILITY_INITIAL_LEDGER: &str = "rz-rovent-utility-initial-ledger-projection/1";
pub const UTILITY_ORIGINAL_CLOSURE: &str = "rz-rovent-utility-original-training-closure/1";
const TERMINAL: &str = "rz-rovent-utility-parent-terminal-receipt/1";
const JOURNAL: &str = "rz-rovent-utility-ledger-event/1";
const DERIVED: &str = "rz-rovent-utility-derived-training-state/1";
const OLD_RUN_RECLAMATION: &str = "rz-rovent-utility-old-run-reclamation/1";
const TRAINER_END: &str = "rz-rovent-utility-trainer-end/1";
const TRAINER_RESULT: &str = "rz-rovent-utility-trainer-result/1";
const ARCHIVE_OBSERVATION: &str = "rz-rovent-utility-checkpoint-archive-observation/1";
const ARCHIVE_FAILURE: &str = "rz-rovent-utility-checkpoint-archive-failure/1";
const MAX_METADATA_BYTES: u64 = 16 << 20;
const MAX_TENSOR_BYTES: u64 = 2 << 30;
const MAX_JOURNAL_BYTES: u64 = 2 << 30;
const MAX_JOURNAL_EVENT_BYTES: usize = 256 << 10;
const MAX_CHAIN: usize = 64;

// The legacy read error and public signature remain the original ArenaError.
// New source families may instead retain a failed original-file owner.
type CheckpointReadFailure = ArenaError;

fn checkpoint_read_failure(error: ArenaError) -> CheckpointReadFailure {
    error
}

fn open_checkpoint_dataset(
    reference: &RoventUtilityDatasetReference,
    preparation: &Value,
    budget: &Arc<RoventRecoveryReadBudget>,
) -> Result<VerifiedUtilityDataset, CheckpointReadFailure> {
    VerifiedUtilityDataset::open(reference, preparation, budget)
}

fn validate_checkpoint_preparation(preparation: &Value) -> Result<(), ArenaError> {
    validate_utility_training_preparation(preparation)?;
    check(
        preparation["schema"] == "rz-rovent-utility-training-preparation/2",
        "actual utility cohort preparation required",
    )
}

fn checkpoint_preparation_seals(preparation: &Value) -> Result<(String, String), ArenaError> {
    let parts = super::numerical_parts::ValidatedUtilityNumericalParts::from_cohort(preparation)?;
    Ok((
        parts.recipe_sha256().to_owned(),
        parts.source_registration_sha256().to_owned(),
    ))
}

fn open_checkpoint_ancestors(
    registrations: &[RoventUtilityAncestorDatasetRegistration],
    budget: &Arc<RoventRecoveryReadBudget>,
) -> Result<VerifiedUtilityAncestors, ArenaError> {
    VerifiedUtilityAncestors::open(registrations, budget)
}

fn checkpoint_utility_datasets(
    ancestors: &VerifiedUtilityAncestors,
) -> &[Arc<VerifiedUtilityDataset>] {
    ancestors.utility_datasets()
}

fn checkpoint_ancestor_references(
    ancestors: &VerifiedUtilityAncestors,
) -> Vec<RoventUtilityDatasetReference> {
    ancestors.utility_references()
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
    registration.as_cohort()
}

fn checkpoint_window(
    window: &super::runtime_family::RuntimeUtilityWindow,
) -> Result<&RoventUtilityTrainWindow, ArenaError> {
    window.as_cohort()
}

fn checkpoint_dataset(
    dataset: &super::runtime_family::RuntimeUtilityDataset,
) -> Result<&Arc<VerifiedUtilityDataset>, ArenaError> {
    dataset.as_cohort()
}

fn checkpoint_parent(
    source: Option<&super::runtime_family::RuntimeUtilityCheckpointSource>,
) -> Result<Option<&Arc<VerifiedUtilityCheckpointSource>>, ArenaError> {
    source.map(|source| source.as_cohort()).transpose()
}

fn checkpoint_archive(
    archive: &super::runtime_family::RuntimePublishedUtilityCheckpointArchive,
) -> Result<&PublishedUtilityCheckpointArchive, ArenaError> {
    archive.as_cohort()
}

/// Relocation changes where the caller retained the same original files, not
/// the historical family, raw identities or source/model/recipe declarations.
pub(super) fn same_ancestor_semantics(
    original: &RoventUtilityAncestorDatasetRegistration,
    caller: &RoventUtilityAncestorDatasetRegistration,
) -> bool {
    use RoventUtilityAncestorDatasetRegistration as Kind;
    let mut normalized = caller.clone();
    match (original, &mut normalized) {
        (Kind::PcLegacy { registration: old }, Kind::PcLegacy { registration: new })
        | (
            Kind::PcSubmittedInputV1 { registration: old },
            Kind::PcSubmittedInputV1 { registration: new },
        ) => {
            new.root.clone_from(&old.root);
            new.publication.path.clone_from(&old.publication.path);
        }
        (
            Kind::RAuxiliary {
                registration: old,
                recipe: old_recipe,
            },
            Kind::RAuxiliary {
                registration: new,
                recipe: new_recipe,
            },
        ) => {
            new.root.clone_from(&old.root);
            new.publication.path.clone_from(&old.publication.path);
            new_recipe.path.clone_from(&old_recipe.path);
        }
        (
            Kind::UtilityCohort {
                registration: old,
                preparation: old_prep,
            },
            Kind::UtilityCohort {
                registration: new,
                preparation: new_prep,
            },
        ) => {
            new.root.clone_from(&old.root);
            new.publication.path.clone_from(&old.publication.path);
            new.durable_ack.path.clone_from(&old.durable_ack.path);
            new_prep.path.clone_from(&old_prep.path);
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
    match caller {
        RoventUtilityAncestorDatasetRegistration::UtilityCohort {
            registration: declared,
            preparation: original,
        } => {
            check(
                declared.publication.sha256 == reference.publication.sha256
                    && declared.publication.bytes == reference.publication.bytes
                    && declared.durable_ack.sha256 == reference.durable_ack.sha256
                    && declared.durable_ack.bytes == reference.durable_ack.bytes
                    && original.sha256 == original_preparation.sha256
                    && original.bytes == original_preparation.bytes,
                "caller source-current utility ancestor changed its own original preparation/raw ACK",
            )?;
        }
        _ => {
            return Err(invalid(
                "source-current utility dataset was relabelled as another family",
            ));
        }
    }
    Ok(())
}

include!("checkpoint/machine.rs");

#[cfg(test)]
mod tests {
    //! Pure format/identity operands only. These fixtures issue no verified
    //! dataset, live peer, tensor, process closure or learned capability.
    use super::*;

    fn reference(name: &str) -> UtilityCheckpointFile {
        UtilityCheckpointFile {
            path: format!("test/{name}"),
            sha256: "ab".repeat(32),
            bytes: 1,
        }
    }
    fn publication() -> UtilityCheckpointPublication {
        UtilityCheckpointPublication {
            schema: UTILITY_CHECKPOINT_PUBLICATION.into(),
            execution_id: "test-execution".into(),
            window_id: "test-window".into(),
            run_id: "test-run".into(),
            preparation_sha256: "ab".repeat(32),
            dataset_sha256: "bc".repeat(32),
            window_spec_sha256: "cd".repeat(32),
            trainer_source_sha256: "de".repeat(32),
            trainer_python_executable_sha256: "ef".repeat(32),
            trainer_dependency_set_sha256: "fa".repeat(32),
            tensor: reference("tensor.pt"),
            retirement: reference("retirement.json"),
            tensor_projection: reference("tensor-projection.json"),
            initial_ledger_state: reference("initial.json"),
            original_preparation: reference("preparation.json"),
            original_window: reference("window.json"),
            original_registration: reference("registration.json"),
            original_rpc_journal: reference("journal.jsonl"),
            completion_receipts: vec![],
            ancestor_completion_receipts: vec![],
            parent_publication: None,
            cost: reference("cost.json"),
            closure: reference("closure.json"),
            historical_evidence_only: true,
            execution_authorized: false,
        }
    }
    #[test]
    fn original_object_decoder_rejects_duplicate_nested_keys_arrays_and_legacy_tags() {
        assert!(
            decode::<UtilityCheckpointFile>(
                br#"{"path":"test/a","sha256":"a","sha256":"b","bytes":1}"#
            )
            .is_err()
        );
        assert!(decode::<UtilityCheckpointFile>(br#"["test/a","hash",1]"#).is_err());
        let mut value = serde_json::to_value(publication()).unwrap();
        value["schema"] = json!("rz-rovent-checkpoint-publication/1");
        assert!(
            from_value::<UtilityCheckpointPublication>(value)
                .unwrap()
                .validate()
                .is_err()
        );
        assert!(decode::<Value>(br#"{"outer":{"state":0,"state":1}}"#).is_err());
    }
    #[test]
    fn original_publication_caps_and_entire_receipt_chain_are_checked_before_admission() {
        let mut value = publication();
        value.validate().unwrap();
        value.tensor.bytes = MAX_TENSOR_BYTES + 1;
        assert!(value.validate().is_err());
        value.tensor.bytes = 1;
        value.original_rpc_journal.bytes = MAX_JOURNAL_BYTES + 1;
        assert!(value.validate().is_err());
        value.original_rpc_journal.bytes = 1;
        value.tensor_projection.bytes = MAX_METADATA_BYTES + 1;
        assert!(value.validate().is_err());
        value.tensor_projection.bytes = 1;
        value
            .completion_receipts
            .push(reference("completion-a.json"));
        value
            .ancestor_completion_receipts
            .push(reference("completion-b.json"));
        assert!(value.validate().is_err());
    }
    #[test]
    fn closed_seal_can_change_only_cost_and_closure_from_its_original_boundary() {
        let original = publication();
        let mut closed = original.clone();
        closed.cost = reference("closed-cost.json");
        closed.closure = reference("closed-closure.json");
        assert!(same_boundary_publication(&original, &closed).unwrap());
        closed.tensor.sha256 = "11".repeat(32);
        assert!(!same_boundary_publication(&original, &closed).unwrap());
        closed = original.clone();
        closed.original_rpc_journal.bytes += 1;
        assert!(!same_boundary_publication(&original, &closed).unwrap());
        closed = original.clone();
        closed.parent_publication = Some(reference("invented-parent.json"));
        assert!(!same_boundary_publication(&original, &closed).unwrap());
    }
    #[test]
    fn mixed_source_relocation_never_erases_family_semantics_raw_ack_or_preparation() {
        use RoventUtilityAncestorDatasetRegistration as Kind;
        let pin = |path: &str| RoventTrainingFilePin {
            path: path.into(),
            sha256: "aa".repeat(32),
            bytes: 1,
        };
        let original = Kind::UtilityCohort {
            registration: RoventUtilityDatasetReference {
                root: "/old/corpus".into(),
                publication: pin("/old/corpus/publication.json"),
                durable_ack: pin("/old/corpus/ack.json"),
            },
            preparation: pin("/old/preparation.json"),
        };
        let mut relocated = original.clone();
        if let Kind::UtilityCohort {
            registration,
            preparation,
        } = &mut relocated
        {
            registration.root = "/current/corpus".into();
            registration.publication.path = "/current/corpus/publication.json".into();
            registration.durable_ack.path = "/current/corpus/ack.json".into();
            preparation.path = "/current/preparation.json".into();
        }
        assert!(same_ancestor_semantics(&original, &relocated));
        if let Kind::UtilityCohort { preparation, .. } = &mut relocated {
            preparation.bytes += 1;
        }
        assert!(!same_ancestor_semantics(&original, &relocated));
        relocated = original.clone();
        if let Kind::UtilityCohort { registration, .. } = &mut relocated {
            registration.durable_ack.sha256 = "bb".repeat(32);
        }
        assert!(!same_ancestor_semantics(&original, &relocated));
    }
    #[test]
    fn known_terminal_delta_is_counted_once_while_original_unknown_remains_explicit() {
        let snapshot = RoventUtilityRecordedCost {
            samples: 8,
            output_bytes: 100,
            ..Default::default()
        };
        let post = RoventUtilityRecordedCost {
            samples: 8,
            output_bytes: 140,
            ..Default::default()
        };
        let end = RoventUtilityRecordedCost {
            samples: 8,
            output_bytes: 160,
            ..Default::default()
        };
        let original_unknown = RoventUtilityInheritedUnknownTail {
            checkpoint_publication_sha256: "ab".repeat(32),
            execution_id: "test-execution".into(),
            window_id: "test-window".into(),
            tail: RoventUtilityClosureTail::Unknown {
                reason: "original_process_still_running".into(),
                evidence: None,
            },
        };
        let cost = RoventUtilityCheckpointCost::from_recorded_parts(
            Default::default(),
            snapshot,
            subtract(post, snapshot).unwrap(),
            RoventUtilityClosureTail::Known {
                delta: subtract(end, post).unwrap(),
                evidence: RoventUtilityCostEvidence {
                    path: "test/terminal.json".into(),
                    sha256: "cd".repeat(32),
                    bytes: 10,
                },
            },
            vec![original_unknown.clone()],
        )
        .unwrap();
        assert_eq!(cost.known_lower_bound, end);
        assert_eq!(
            cost.known_resume_delta().unwrap().known_delta.output_bytes,
            60
        );
        assert_eq!(cost.inherited_unknown_tails, vec![original_unknown]);
        assert_eq!(
            serde_json::to_value(cost.total_status).unwrap(),
            json!("lower_bound_only")
        );
        assert!(subtract(post, end).is_err());
    }
    #[test]
    fn original_child_terminal_uses_exact_window_completed_count_and_keeps_raw_receipt_shape() {
        let publication = publication();
        let end =
            json!({"completed_updates":7,"dispatched_updates":8,"window_dispatched_updates":3});
        let mut child = json!({"schema":"rz-rovent-utility-trainer-result/1",
            "execution_id":publication.execution_id,"window_id":publication.window_id,"status":"completed",
            "result":{},"bootstrap":{},"completed_updates":7,"window_completed_updates":2,
            "dispatched_updates":8,"window_dispatched_updates":3,"physical_completion":"confirmed",
            "physical_process_retirement":"original_parent_join_required","arena_eligible":false});
        validate_child_terminal(&child, &end, &publication, 2).unwrap();
        child["window_completed_updates"] = json!("2");
        assert!(validate_child_terminal(&child, &end, &publication, 2).is_err());
        child["window_completed_updates"] = json!(3);
        assert!(validate_child_terminal(&child, &end, &publication, 2).is_err());
        child["window_completed_updates"] = json!(2);
        child["run_id"] = json!("invented");
        assert!(validate_child_terminal(&child, &end, &publication, 2).is_err());
    }
}
