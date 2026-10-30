//! Serialize-only projection of original archive custody; never a capability.

use super::*;
use rz_search::pals::store::{ArchiveOwnerSnapshot, ArchiveStats};
use serde::Serialize;
use serde_json::{Value, json};

#[derive(Debug, Serialize)]
pub(crate) struct StorageLifecycleObservation {
    pub(crate) complete: bool,
    facts: Value,
    // Read-only coverage from this same borrowed custody. These fields never
    // enter the JSON schema or reconstruct a live root/finish capability.
    #[serde(skip)]
    after_root_scope: Option<SearchAuthority>,
    #[serde(skip)]
    after_finish_complete: bool,
}
impl StorageLifecycleObservation {
    pub(crate) fn require_after_root(&mut self, authority: SearchAuthority) {
        self.complete &= self.after_root_scope == Some(authority);
    }

    pub(crate) fn require_after_finish(&mut self) {
        self.complete &= self.after_finish_complete;
    }

    pub(crate) fn observe_hot_owner(
        &mut self,
        records: usize,
        stats: ArchiveStats,
        owner: Option<ArchiveOwnerSnapshot>,
    ) {
        self.facts["hot_semantic_owner"] = json!({
            "records":records,"hot":hot(stats),"archive_owner":owner.map(owner_observation),
        });
        if owner.is_none_or(|owner| !owner.complete) {
            self.complete = false;
        }
    }
}

fn scope(value: SearchAuthority) -> Value {
    json!({"epoch":value.epoch.0,"game":value.game.0,"root":value.root.0,
        "implementation_sha256":hex(value.implementation.0)})
}
fn hex(value: [u8; 32]) -> String {
    let mut result = String::with_capacity(64);
    for byte in value {
        use std::fmt::Write as _;
        let _ = write!(result, "{byte:02x}");
    }
    result
}
fn hot(value: ArchiveStats) -> Value {
    json!({"states":value.states,"lines":value.lines,"situations":value.situations,
        "observations":value.observations,"dependencies":value.dependencies,
        "executions":value.executions,"consumers":value.consumers,
        "history_bytes":value.history_bytes})
}
fn join(value: ArchiveShutdownReceipt) -> Value {
    json!({"worker_joined":value.worker_joined,"owners_returned":value.owners_returned,
        "admission_closed":value.admission_closed})
}
fn owner_observation(value: ArchiveOwnerSnapshot) -> Value {
    json!({"owner_id":value.owner_id,"generation":value.generation,
        "event_sequence":value.event_sequence,
        "compiled_source_sha256":hex(value.compiled_source_sha256),
        "lifecycle":format!("{:?}",value.lifecycle),"complete":value.complete,
        "committed_chunks":value.committed_chunks,"committed_bytes":value.committed_bytes,
        "integrity_verified_chunks":value.integrity_verified_chunks,
        "integrity_verified_bytes":value.integrity_verified_bytes,
        "last_committed_generation":value.last_committed_generation,
        "last_verified_generation":value.last_verified_generation,
        "ram_reclaim_events":value.ram_reclaim_events,
        "integrity_before_reclaim_events":value.integrity_before_reclaim_events,
        "unverified_reclaim_events":value.unverified_reclaim_events,
        "ram_released_bytes":value.ram_released_bytes,
        "ram_release_method":value.ram_release_method,
        "loads_requested":value.loads_requested,"loads_completed":value.loads_completed,
        "load_bytes_total":value.load_bytes_total,
        "owner_generation_checks":value.owner_generation_checks,
        "owner_generation_check_failures":value.owner_generation_check_failures,
        "quota_failures":value.quota_failures,"io_failures":value.io_failures,
        "pin_saturation_failures":value.pin_saturation_failures,
        "pending_commit_bytes":value.pending_commit_bytes,
        "pending_files_retained":value.pending_files_retained,
        "pending_buffers_retained":value.pending_buffers_retained,
        "admission_closed":value.admission_closed,"cleanup_complete":value.cleanup_complete})
}
fn operation(value: ArchiveOperationKind) -> &'static str {
    match value {
        ArchiveOperationKind::PrepareRoot => "prepare_root",
        ArchiveOperationKind::LoadRoot => "load_root",
        ArchiveOperationKind::RestoreRootIfArchived => "restore_root_if_archived",
        ArchiveOperationKind::ReclaimHot => "reclaim_hot",
        ArchiveOperationKind::PrepareForeign => "prepare_foreign",
        ArchiveOperationKind::NewGame => "new_game",
        ArchiveOperationKind::PrepareProfiledForeign => "prepare_profiled_foreign",
        ArchiveOperationKind::PersistProfiledRetirement => "persist_profiled_retirement",
        ArchiveOperationKind::FinishIdleGame => "finish_idle_game",
    }
}
fn receipt(value: &ArchiveWorkReceipt) -> Value {
    json!({"nonce":value.nonce,"authority":scope(value.authority),
        "operation":operation(value.operation),"generation_before":value.generation_before,
        "generation_after":value.generation_after,"hot_before":hot(value.hot_before),
        "hot_after":hot(value.hot_after),"io_bytes":value.io_bytes,
        "elapsed_ns":u64::try_from(value.elapsed.as_nanos()).ok(),
        "loaded_root":value.loaded_root,"archive_owner":value.archive_owner.map(owner_observation),
        "next_authority":value.next_authority.map(scope),
        "idle_checkpoint":value.idle_checkpoint.as_ref().map(|checkpoint|json!({
            "hot":hot(checkpoint.hot),"records":checkpoint.records,"nodes":checkpoint.nodes,
            "committed_bytes":checkpoint.committed_bytes,"io_bytes":checkpoint.io_bytes,
            "archive_owner":owner_observation(checkpoint.owner)}))})
}

pub(super) fn project<M: RoleModel + 'static>(
    core: &WorkerCustody<M>,
) -> StorageLifecycleObservation {
    let phase_count = core
        .receipt_sequence
        .saturating_sub(core.phase_receipt_start);
    let retained = usize::try_from(phase_count)
        .ok()
        .filter(|count| *count <= core.receipts.len());
    // The normal idle phase has at most PrepareRoot+Restore. A bounded unknown
    // row is preferable to copying the complete receipt ring after a bad phase.
    let selected = retained.filter(|count| *count <= 2);
    let receipts = selected.map(|count| {
        core.receipts
            .iter()
            .skip(core.receipts.len() - count)
            .map(receipt)
            .collect::<Vec<_>>()
    });
    let receipts_complete = selected.is_some()
        && core
            .receipts
            .iter()
            .rev()
            .take(selected.unwrap_or(0))
            .all(|receipt| {
                receipt.archive_owner.is_some_and(|owner| owner.complete)
                    && receipt.elapsed.as_nanos() <= u128::from(u64::MAX)
            });
    let receipt_at = |offset: usize| {
        selected.and_then(|count| {
            (offset < count)
                .then(|| core.receipts.get(core.receipts.len() - count + offset))
                .flatten()
        })
    };
    let empty = core
        .empty_game_phases
        .back()
        .filter(|empty| Some(empty.requested_scope) == core.phase_scope && empty.owner.complete);
    let root_phase_complete = match selected {
        Some(0) => empty.is_some_and(|empty| {
            matches!(empty.kind, EmptyGamePhaseKind::UnusedSetup)
                && core.setup_receipt.as_ref().is_some_and(|setup| {
                    Some(setup.scope) == core.phase_scope
                        && setup.configuration_complete
                        && setup.archive_owner.is_some_and(|owner| {
                            owner.complete && owner.owner_id == empty.owner.owner_id
                        })
                })
        }),
        Some(1) => empty.is_some_and(|empty| {
            matches!(empty.kind, EmptyGamePhaseKind::NewGame)
                && receipt_at(0).is_some_and(|receipt| {
                    receipt.operation == ArchiveOperationKind::NewGame
                        && receipt.next_authority == core.phase_scope
                        && receipt
                            .idle_checkpoint
                            .as_ref()
                            .is_some_and(|checkpoint| checkpoint.owner.complete)
                })
        }),
        Some(2) => {
            receipt_at(0)
                .is_some_and(|receipt| receipt.operation == ArchiveOperationKind::PrepareRoot)
                && receipt_at(1).is_some_and(|receipt| {
                    receipt.operation == ArchiveOperationKind::RestoreRootIfArchived
                })
        }
        _ => false,
    };
    let after_root_scope = core.phase_scope.filter(|_| root_phase_complete);
    let after_finish_complete = core.phase_scope.is_some()
        && selected == Some(1)
        && core.finish_completed
        && receipt_at(0).is_some_and(|receipt| {
            receipt.operation == ArchiveOperationKind::FinishIdleGame
                && receipt
                    .idle_checkpoint
                    .as_ref()
                    .is_some_and(|checkpoint| checkpoint.owner.complete)
                && receipt
                    .archive_owner
                    .is_some_and(|owner| owner.complete && owner.admission_closed)
        });
    let failure = core
        .first_failure
        .as_ref()
        .map(|failure| failure.code.chars().take(128).collect::<String>());
    let pending = core.pending.is_some();
    let quarantined = core.quarantine.is_some()
        || core.unexpected.is_some()
        || core.setup_rejected.is_some()
        || core.setup_admission_rejection.is_some()
        || core.rejected_work.is_some()
        || core.caller_panic.is_some();
    let complete = receipts_complete
        && (after_root_scope.is_some() || after_finish_complete)
        && !core.observation_sequence_exhausted
        && !pending
        && !quarantined
        && failure.is_none()
        && core.shutdown.is_some_and(|join| {
            join.worker_joined && join.owners_returned && join.admission_closed
        });
    StorageLifecycleObservation {
        complete,
        after_root_scope,
        after_finish_complete,
        facts: json!({"phase_scope":core.phase_scope.map(scope),
            "last_scope":core.last_scope.map(scope),
            "receipt_sequence":core.receipt_sequence,"phase_receipt_start":core.phase_receipt_start,
            "phase_receipts_known":selected.is_some(),"phase_receipts":receipts,
            "observation_sequence_exhausted":core.observation_sequence_exhausted,
            "setup_receipt_observed":core.setup_receipt.is_some(),
            "empty_game_phase":core.empty_game_phases.back()
                .filter(|empty|Some(empty.requested_scope)==core.phase_scope)
                .map(|empty|json!({
                    "kind":match empty.kind {EmptyGamePhaseKind::UnusedSetup=>"unused_setup",EmptyGamePhaseKind::NewGame=>"new_game"},
                    "requested_scope":scope(empty.requested_scope),
                    "archive_owner":owner_observation(empty.owner)})),
            "original_phase_join":core.shutdown.map(join),
            "last_join_observed":core.shutdown_observed.map(join),
            "last_join_observation_scope_known":core.shutdown.is_some(),
            "last_join_observation_scope":core.shutdown.and(core.phase_scope).map(scope),
            "finish_completed":core.finish_completed,"first_failure_code":failure,
            "pending":pending,"quarantined":quarantined,"hot_semantic_owner":null,
            "physical_completion_attested":false,"capabilities_reissued":false}),
    }
}
