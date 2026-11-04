//! Read-only projection of actual adaptive /2 root and initial-stop receipts.
//!
//! These values are diagnostic snapshots, never live admission or physical
//! completion authority. In particular the root receipt is sampled before the
//! manager returns its owners and is not the final process fence.

use super::PcrDriver;
use crate::rovent_bootstrap::process_boundary::VerifiedOwnedRoventProcessBoundaryV2;
use crate::search_driver::SearchSessionContext;
use rz_contracts::{Digest, pals::SearchAuthority};
use serde_json::{Value, json};
use std::sync::Arc;

fn hex(value: Digest) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut result = String::with_capacity(64);
    for byte in value.0 {
        result.push(DIGITS[(byte >> 4) as usize] as char);
        result.push(DIGITS[(byte & 15) as usize] as char);
    }
    result
}

fn authority(value: SearchAuthority) -> Value {
    json!({"epoch": value.epoch.0, "game": value.game.0, "root": value.root.0,
           "implementation_sha256": hex(value.implementation)})
}

fn descriptor_authority_matches(
    declared: &rz_experiments::RoventRootAuthority,
    actual: SearchAuthority,
) -> bool {
    declared.epoch == actual.epoch.0
        && declared.game == actual.game.0
        && declared.root == actual.root.0
        && declared.implementation_sha256 == hex(actual.implementation)
}

fn same_original_context(
    admission: &crate::rovent_root_admission::source_seal::RoventRootAdmissionV2,
    context: &SearchSessionContext,
) -> bool {
    let original = admission.original_context();
    original.authority == context.authority
        && original
            .original_start
            .as_ref()
            .zip(context.original_start.as_ref())
            .is_some_and(|(a, b)| Arc::ptr_eq(a, b))
        && original
            .original_go
            .as_ref()
            .zip(context.original_go.as_ref())
            .is_some_and(|(a, b)| Arc::ptr_eq(a, b))
        && Arc::ptr_eq(&original.current, &context.current)
        && Arc::ptr_eq(
            &original.control.cancellation,
            &context.control.cancellation,
        )
        && original.control.deadline == context.control.deadline
        && original.control.soft_deadline == context.control.soft_deadline
        && original.control.admission_deadline == context.control.admission_deadline
        && original.shutdown_deadline == context.shutdown_deadline
        && original.goal == context.goal
        && original.requested_nodes == context.requested_nodes
}

/// Read the exact current root from the original driver. An older receipt is
/// not relabelled as the current root; it is treated as absent. The caller
/// must independently record its run result and final session completion.
pub(super) fn project_adaptive(
    driver: &PcrDriver,
    context: &SearchSessionContext,
    process: &Arc<VerifiedOwnedRoventProcessBoundaryV2>,
) -> Result<Option<Value>, &'static str> {
    let root = driver
        .try_observe_rovent_root_v2()
        .map_err(|_| "adaptive_root_receipt_unavailable")?
        .filter(|receipt| receipt.authority == context.authority);
    let initial = driver
        .try_observe_rovent_initial_v2()
        .map_err(|_| "adaptive_initial_receipt_unavailable")?
        .filter(|receipt| receipt.authority() == context.authority);
    if root.is_some() && initial.is_some() {
        return Err("adaptive_duplicate_current_receipt");
    }
    let original_lock = process.input_lock();
    if let Some(receipt) = root {
        let admission = &receipt.root_admission;
        let descriptor = admission.descriptor();
        if !admission
            .owner()
            .v2()
            .is_some_and(|lock| Arc::ptr_eq(lock, original_lock))
            || descriptor.verify_against(original_lock).is_err()
            || !same_original_context(admission, context)
            || admission.authority() != context.authority
            || receipt.input_lock != admission.startup_lock_digest()
            || descriptor.input.startup_lock_sha256 != hex(receipt.input_lock)
            || receipt.mode != original_lock.manifest.engine.mode
            || !descriptor_authority_matches(&descriptor.input.authority, context.authority)
        {
            return Err("adaptive_root_scope_mismatch");
        }
        let budget = descriptor
            .root_budget_digest()
            .map_err(|_| "adaptive_root_budget_invalid")?;
        let deadline = descriptor
            .deadline_seal_digest()
            .map_err(|_| "adaptive_root_deadline_invalid")?;
        if receipt.root_budget != budget {
            return Err("adaptive_root_budget_mismatch");
        }
        if receipt.frontier.as_ref().is_some_and(|frontier| {
            frontier.authority != context.authority
                || frontier.original_deadline != admission.hard_deadline()
                || frontier.requested_nodes != context.requested_nodes
        }) || receipt
            .physical
            .as_ref()
            .is_some_and(|physical| physical.authority != context.authority)
            || receipt
                .profiled_physical
                .as_ref()
                .is_some_and(|physical| physical.authority() != context.authority)
        {
            return Err("adaptive_observation_scope_mismatch");
        }
        let expected_external = matches!(
            &original_lock.manifest.engine.researcher,
            Some(rz_experiments::RoventResearcher::External { .. })
        );
        let expected_domain = if expected_external {
            "rz-rovent-profiled-external-root/2"
        } else {
            crate::pals_adaptive::ROVENT_NATIVE_ROOT_V2_DOMAIN
        };
        if receipt.domain != expected_domain {
            return Err("adaptive_receipt_domain_mismatch");
        }
        // The shared manager can also produce a native-only accounting
        // snapshot for an external researcher. It is not evidence that the
        // external process drained; use only the profiled receipt for that
        // branch, and leave a missing profiled receipt unknown.
        if !expected_external && receipt.profiled_physical.is_some() {
            return Err("adaptive_physical_kind_mismatch");
        }
        let frontier = receipt.frontier.as_ref().map(|value| json!({
            "goal":format!("{:?}", value.goal),
            "completed_depth":value.horizon.completed_depth.map(|depth|depth.0),
            "maximum_examined_ply":value.horizon.maximum_examined_ply.0,
            "native_usage_known":value.work.native_usage_known,
            "native_nodes":value.work.native_usage_known.then_some(value.work.native_nodes),
            "native_qsearch_nodes":value.work.native_usage_known.then_some(value.work.native_qsearch_nodes),
            "reserved_native_nodes":value.work.reserved_native_nodes,
            "outstanding_native_executions":value.work.outstanding_native_executions,
        }));
        let physical =
            if expected_external {
                receipt.profiled_physical.as_ref().map(|value| json!({
                "kind":"profiled_external", "permits_root_reuse":value.permits_root_reuse(),
                "quarantined":value.quarantined(), "pending_tickets":value.pending_tickets(),
                "native_process_joined":value.native_process_joined(),
                "native_physical_shutdown_confirmed":value.native_physical_shutdown_confirmed(),
                "external_process_joined":value.external_process_joined(),
            }))
            } else {
                receipt.physical.as_ref().map(|value| json!({
                "kind":"own", "permits_root_reuse":value.permits_root_reuse(),
                "quarantined":value.quarantined, "pending_tickets":value.pending_tickets,
                "native_owner_joined":value.native_owner_joined,
                "native_owner_quiesced":value.native_owner_quiesced,
                "native_physical_shutdown_confirmed":value.native_physical_shutdown_confirmed,
            }))
            };
        return Ok(Some(json!({
            "kind":"root", "receipt_domain":receipt.domain,
            "observation_scope":"original_root_before_owner_return_not_final_process_fence",
            "authority":authority(context.authority),
            "input_lock_sha256":hex(receipt.input_lock),
            "root_admission_sha256":hex(descriptor.digest().map_err(|_| "adaptive_descriptor_invalid")?),
            "root_budget_sha256":hex(budget), "deadline_seal_sha256":hex(deadline),
            "state_and_history_sha256":hex(admission.state_and_history_digest()),
            "frontier":frontier, "physical":physical,
            "physical_completion_attested":false, "execution_authorized":false,
        })));
    }
    if let Some(receipt) = initial {
        let source = receipt.source();
        let descriptor = source.descriptor();
        if !source
            .owner()
            .v2()
            .is_some_and(|lock| Arc::ptr_eq(lock, original_lock))
            || descriptor.verify_against(original_lock).is_err()
            || descriptor.input.startup_lock_sha256 != hex(source.startup_lock_digest())
            || !descriptor_authority_matches(&descriptor.input.authority, context.authority)
        {
            return Err("adaptive_initial_scope_mismatch");
        }
        return Ok(Some(json!({
            "kind":"initial_stop", "receipt_domain":receipt.domain(),
            "observation_scope":"original_initial_stop_no_current_root_fence",
            "authority":authority(context.authority),
            "input_lock_sha256":hex(source.startup_lock_digest()),
            "root_admission_sha256":hex(descriptor.digest().map_err(|_| "adaptive_initial_descriptor_invalid")?),
            "root_budget_sha256":hex(descriptor.root_budget_digest().map_err(|_| "adaptive_initial_budget_invalid")?),
            "deadline_seal_sha256":hex(descriptor.deadline_seal_digest().map_err(|_| "adaptive_initial_deadline_invalid")?),
            "state_and_history_sha256":hex(source.state_and_history_digest()),
            "original_start_identity_observed":false,
            "stop_reason":format!("{:?}", source.stop_reason()),
            "used_original_startup_ready":receipt.used_original_startup_ready(),
            "previous_clean_root_present":receipt.previous_clean_root().is_some(),
            "previous_clean_profiled_root_present":receipt.previous_clean_profiled_root().is_some(),
            "physical":null,
            "physical_completion_attested":false, "execution_authorized":false,
        })));
    }
    Ok(None)
}
