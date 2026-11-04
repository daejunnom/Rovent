//! Read-only projection of the original V2 NNUEC receipt.
//!
//! This cannot issue a root capability, accept work, or turn a missing
//! manager fence into a physical-completion claim.

use crate::rovent_bootstrap::nnuec_session::RoventNnuecSessionDriver;
use crate::rovent_bootstrap::process_boundary::VerifiedOwnedRoventProcessBoundaryV2;
use crate::rovent_root_admission::source_seal::RoventRootAdmissionV2;
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
    json!({
        "epoch": value.epoch.0,
        "game": value.game.0,
        "root": value.root.0,
        "implementation_sha256": hex(value.implementation),
    })
}

fn same_original_context(root: &RoventRootAdmissionV2, context: &SearchSessionContext) -> bool {
    let original = root.original_context();
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

fn original_v2_owner(
    owner: &rz_experiments::RoventOriginalInputLock,
    boundary: &VerifiedOwnedRoventProcessBoundaryV2,
) -> bool {
    owner
        .v2()
        .is_some_and(|lock| Arc::ptr_eq(lock, boundary.input_lock()))
}

/// Projects only a current original receipt. A stale prior-root receipt is not
/// evidence for this context; failure to read a receipt stays an observation
/// error rather than a search or cleanup error.
pub(super) fn project_nnuec(
    driver: &RoventNnuecSessionDriver,
    context: &SearchSessionContext,
    boundary: &Arc<VerifiedOwnedRoventProcessBoundaryV2>,
) -> Result<Option<Value>, &'static str> {
    let lock = boundary.input_lock();
    let root_receipt = driver
        .try_observe_root_v2()
        .map_err(|_| "nnuec_root_receipt_read_failed")?
        .filter(|receipt| receipt.authority == context.authority);
    let initial_receipt = driver
        .try_observe_initial_v2()
        .map_err(|_| "nnuec_initial_receipt_read_failed")?
        .filter(|receipt| receipt.authority == context.authority);
    if root_receipt.is_some() && initial_receipt.is_some() {
        return Err("nnuec_duplicate_current_receipt");
    }
    if let Some(receipt) = root_receipt {
        let root = &receipt.root_admission;
        let declared = &root.descriptor().input.authority;
        if receipt.domain != "rz-rovent-nnuec-native-root/2"
            || receipt.input_lock != root.startup_lock_digest()
            || hex(receipt.input_lock) != lock.canonical_sha256
            || root.authority() != context.authority
            || !original_v2_owner(root.owner(), boundary)
            || !same_original_context(root, context)
            || root.descriptor().verify_against(lock).is_err()
            || declared.epoch != context.authority.epoch.0
            || declared.game != context.authority.game.0
            || declared.root != context.authority.root.0
            || declared.implementation_sha256 != hex(context.authority.implementation)
        {
            return Err("nnuec_root_original_scope_mismatch");
        }
        if let Some(frontier) = &receipt.frontier {
            if frontier.authority != context.authority
                || frontier.original_deadline != root.hard_deadline()
                || frontier.requested_nodes != context.requested_nodes
            {
                return Err("nnuec_frontier_original_scope_mismatch");
            }
        }
        if receipt
            .physical
            .as_ref()
            .is_some_and(|physical| physical.authority != context.authority)
        {
            return Err("nnuec_physical_original_scope_mismatch");
        }
        let descriptor = root.descriptor();
        let admission = descriptor
            .digest()
            .map_err(|_| "nnuec_root_digest_failed")?;
        let budget = descriptor
            .root_budget_digest()
            .map_err(|_| "nnuec_budget_digest_failed")?;
        let deadline = descriptor
            .deadline_seal_digest()
            .map_err(|_| "nnuec_deadline_digest_failed")?;
        let frontier = receipt.frontier.as_ref();
        let physical = receipt.physical.as_ref();
        return Ok(Some(json!({
            "source_status": "original_current_root",
            "receipt_domain": receipt.domain,
            "authority": authority(context.authority),
            "input_lock_sha256": hex(receipt.input_lock),
            "root_admission_sha256": hex(admission),
            "root_budget_sha256": hex(budget),
            "deadline_seal_sha256": hex(deadline),
            "original_start_identity_matched": true,
            "frontier_present": frontier.is_some(),
            "goal": frontier.map(|value| format!("{:?}", value.goal)),
            "completed_depth": frontier.and_then(|value| value.horizon.completed_depth.map(|depth| depth.0)),
            "maximum_examined_ply": frontier.map(|value| value.horizon.maximum_examined_ply.0),
            "native_nodes": frontier.and_then(|value| value.work.native_usage_known.then_some(value.work.native_nodes)),
            "native_usage_known": frontier.map(|value| value.work.native_usage_known),
            "physical_receipt_present": physical.is_some(),
            "native_root_reuse_permitted": physical.map(|value| value.permits_root_reuse()),
            "native_shutdown_confirmed": physical.map(|value| value.native_physical_shutdown_confirmed),
            "initial_retirement_present": receipt.initial_retirement.is_some(),
            "pre_materialization_retirement_matches": receipt.pre_materialization_retirement_matches(),
            "stopped_before_research_execution": receipt.stopped_before_research_execution,
            "deadline_before_frontier": receipt.deadline_before_frontier,
            "physical_completion_attested": false,
            "execution_authorized": false,
        })));
    }

    let Some(receipt) = initial_receipt else {
        return Ok(None);
    };
    let source = &receipt.source;
    let descriptor = source.descriptor();
    let declared = &descriptor.input.authority;
    if receipt.input_lock != source.startup_lock_digest()
        || hex(receipt.input_lock) != lock.canonical_sha256
        || !original_v2_owner(source.owner(), boundary)
        || descriptor.verify_against(lock).is_err()
        || declared.epoch != context.authority.epoch.0
        || declared.game != context.authority.game.0
        || declared.root != context.authority.root.0
        || declared.implementation_sha256 != hex(context.authority.implementation)
    {
        return Err("nnuec_initial_original_scope_mismatch");
    }
    let admission = descriptor
        .digest()
        .map_err(|_| "nnuec_initial_digest_failed")?;
    let budget = descriptor
        .root_budget_digest()
        .map_err(|_| "nnuec_initial_budget_digest_failed")?;
    let deadline = descriptor
        .deadline_seal_digest()
        .map_err(|_| "nnuec_initial_deadline_digest_failed")?;
    Ok(Some(json!({
        "source_status": "original_initial_stop",
        "authority": authority(context.authority),
        "input_lock_sha256": hex(receipt.input_lock),
        "root_admission_sha256": hex(admission),
        "root_budget_sha256": hex(budget),
        "deadline_seal_sha256": hex(deadline),
        "stop_reason": format!("{:?}", source.stop_reason()),
        "previous_clean_root_present": receipt.previous_clean_root().is_some(),
        "used_original_startup_ready": receipt.used_original_startup_ready(),
        "original_start_identity_matched": null,
        "physical_receipt_present": null,
        "native_root_reuse_permitted": null,
        "native_shutdown_confirmed": null,
        "physical_completion_attested": false,
        "execution_authorized": false,
    })))
}
