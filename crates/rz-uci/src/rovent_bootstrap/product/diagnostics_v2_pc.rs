//! Observational projection of the original PC /2 producer. A JSON value is
//! neither a root admission nor a native fence and must never be read back as
//! one. The caller bounds and delivers it independently of product cleanup.

use crate::rovent_bootstrap::{
    pc_session::RoventPcSessionDriver, process_boundary::VerifiedOwnedRoventProcessBoundaryV2,
};
use crate::search_driver::SearchSessionContext;
use rz_contracts::Digest;
use rz_search::pals::rovent_frontier::{FrontierGoalStatus, FrontierStop};
use serde_json::{Value, json};
use std::sync::Arc;

fn hex(value: Digest) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(64);
    for byte in value.0 {
        output.push(DIGITS[(byte >> 4) as usize] as char);
        output.push(DIGITS[(byte & 15) as usize] as char);
    }
    output
}

fn goal(value: FrontierGoalStatus) -> &'static str {
    match value {
        FrontierGoalStatus::Pending => "pending",
        FrontierGoalStatus::DepthSatisfied => "depth_satisfied",
        FrontierGoalStatus::MateSatisfied => "mate_satisfied",
        FrontierGoalStatus::SelectiveFrontier => "selective_frontier",
        FrontierGoalStatus::GoalUnresolved => "goal_unresolved",
        FrontierGoalStatus::Stopped(reason) => match reason {
            FrontierStop::Canceled => "stopped_canceled",
            FrontierStop::Deadline => "stopped_deadline",
            FrontierStop::WorkBudget => "stopped_work_budget",
            FrontierStop::RoleBudget => "stopped_role_budget",
            FrontierStop::Capacity => "stopped_capacity",
            FrontierStop::NativeUnknown => "stopped_native_unknown",
            FrontierStop::EvidenceReclaimed => "stopped_evidence_reclaimed",
            FrontierStop::Failed => "stopped_failed",
        },
    }
}

fn same_deadline(
    declared: &rz_experiments::RoventRootDeadline,
    actual: rz_contracts::Deadline,
) -> bool {
    declared.clock_epoch == actual.clock.0.0 && declared.at_ns == actual.at.0
}

/// The latest slot can contain an older root while this Start stopped before
/// admission. A stale slot is never attributed to the current context. A
/// current slot that fails the owner/scope checks is an observation error.
pub(super) fn project_pc(
    driver: &RoventPcSessionDriver,
    context: &SearchSessionContext,
    process: &Arc<VerifiedOwnedRoventProcessBoundaryV2>,
) -> Result<Option<Value>, &'static str> {
    let original_lock = process.input_lock();
    let current = driver
        .try_observe_root_v2()
        .map_err(|_| "pc_v2_root_receipt_unavailable")?;
    let initial = driver
        .try_observe_initial_v2()
        .map_err(|_| "pc_v2_initial_receipt_unavailable")?;
    let current = current.filter(|receipt| receipt.authority == context.authority);
    let initial = initial.filter(|receipt| receipt.authority == context.authority);
    if current.is_some() && initial.is_some() {
        return Err("pc_v2_duplicate_current_receipt");
    }
    if let Some(receipt) = current {
        let admission = &receipt.root_admission;
        let source_lock = admission.owner().v2().ok_or("pc_v2_root_lock_version")?;
        let descriptor = admission.descriptor();
        descriptor
            .verify_against(original_lock)
            .map_err(|_| "pc_v2_root_descriptor_invalid")?;
        admission
            .allocation_clock()
            .verify_against(descriptor, original_lock)
            .map_err(|_| "pc_v2_root_allocation_invalid")?;
        let budget = descriptor
            .root_budget_digest()
            .map_err(|_| "pc_v2_root_budget_invalid")?;
        let same_start = admission
            .original_context()
            .original_start
            .as_ref()
            .zip(context.original_start.as_ref())
            .is_some_and(|(a, b)| Arc::ptr_eq(a, b));
        let original = admission.original_context();
        let same_go = original
            .original_go
            .as_ref()
            .zip(context.original_go.as_ref())
            .is_some_and(|(a, b)| Arc::ptr_eq(a, b));
        let declared_authority = &descriptor.input.authority;
        if receipt.domain != "rz-rovent-pc-native-root/2"
            || !Arc::ptr_eq(source_lock, original_lock)
            || receipt.input_lock != admission.startup_lock_digest()
            || hex(receipt.input_lock) != original_lock.canonical_sha256
            || admission.authority() != context.authority
            || receipt.frontier.authority != context.authority
            || receipt.frontier.original_deadline != admission.hard_deadline()
            || receipt.physical.authority != context.authority
            || receipt.physical.root_budget != budget
            || receipt.physical.mode != original_lock.manifest.engine.mode
            || !same_deadline(&descriptor.input.clock.hard, admission.hard_deadline())
            || !same_deadline(&descriptor.input.clock.soft, admission.soft_deadline())
            || !same_deadline(
                &descriptor.input.clock.admission,
                admission.admission_deadline(),
            )
            || !same_deadline(
                &descriptor.input.clock.cleanup,
                admission.cleanup_deadline(),
            )
            || descriptor.input.state_and_history_sha256
                != hex(admission.state_and_history_digest())
            || descriptor.input.rules_profile_sha256 != hex(admission.rules_profile_digest())
            || declared_authority.epoch != context.authority.epoch.0
            || declared_authority.game != context.authority.game.0
            || declared_authority.root != context.authority.root.0
            || declared_authority.implementation_sha256 != hex(context.authority.implementation)
            || !same_start
            || !same_go
            || !Arc::ptr_eq(&original.current, &context.current)
            || !Arc::ptr_eq(
                &original.control.cancellation,
                &context.control.cancellation,
            )
            || original.control.deadline != context.control.deadline
            || original.control.soft_deadline != context.control.soft_deadline
            || original.control.admission_deadline != context.control.admission_deadline
            || original.shutdown_deadline != context.shutdown_deadline
            || original.goal != context.goal
            || original.requested_nodes != context.requested_nodes
            || receipt.frontier.requested_nodes != context.requested_nodes
        {
            return Err("pc_v2_root_scope_mismatch");
        }
        let fence = receipt.physical.root_fence();
        if fence.is_some_and(|fence| fence.authority() != context.authority) {
            return Err("pc_v2_foreign_native_fence");
        }
        let work = &receipt.frontier.work;
        let counters = &receipt.physical.counters;
        return Ok(Some(json!({
            "kind": "original_current_root",
            "receipt_domain": receipt.domain,
            "input_lock_sha256": hex(receipt.input_lock),
            "root_admission_sha256": hex(descriptor.digest().map_err(|_| "pc_v2_root_digest_invalid")?),
            "root_budget_sha256": hex(budget),
            "deadline_seal_sha256": hex(descriptor.deadline_seal_digest().map_err(|_| "pc_v2_deadline_seal_invalid")?),
            "state_and_history_sha256": hex(admission.state_and_history_digest()),
            "original_start_matches": true,
            "goal_status": goal(receipt.frontier.goal),
            "completed_depth": receipt.frontier.horizon.completed_depth.map(|depth| depth.0),
            "maximum_examined_ply": receipt.frontier.horizon.maximum_examined_ply.0,
            "native_usage_known": work.native_usage_known,
            "native_nodes": work.native_usage_known.then_some(work.native_nodes),
            "native_qsearch_nodes": work.native_usage_known.then_some(work.native_qsearch_nodes),
            "reserved_native_nodes": work.reserved_native_nodes,
            "outstanding_native_executions": work.outstanding_native_executions,
            "submitted_role_tickets": counters.submitted_role_tickets,
            "completed_nn_inputs": (counters.nn_input_count_unknown == 0).then_some(counters.completed_nn_inputs),
            "consumed_role_outputs": counters.consumed_role_outputs,
            "consumed_nn_inputs": (counters.consumed_nn_input_count_unknown == 0).then_some(counters.consumed_nn_inputs),
            "root_fence_permits_reuse": fence.map(|fence| fence.permits_reuse()),
            "semantic_closure_permitted": receipt.physical.permits_semantic_closure(),
            "native_process_joined": receipt.physical.native_process_joined,
            "native_physical_shutdown_confirmed": receipt.physical.native_physical_shutdown_confirmed,
            "quarantined": receipt.physical.quarantined,
            "pending_tickets": receipt.physical.pending_tickets,
            "retained_outputs": receipt.physical.retained_outputs,
            "repair_observed": receipt.repair.is_some(),
        })));
    }

    let Some(receipt) = initial else {
        return Ok(None);
    };
    let source = &receipt.source;
    let source_lock = source.owner().v2().ok_or("pc_v2_initial_lock_version")?;
    let descriptor = source.descriptor();
    descriptor
        .verify_against(original_lock)
        .map_err(|_| "pc_v2_initial_descriptor_invalid")?;
    let authority = &descriptor.input.authority;
    if !Arc::ptr_eq(source_lock, original_lock)
        || receipt.input_lock != source.startup_lock_digest()
        || hex(receipt.input_lock) != original_lock.canonical_sha256
        || descriptor.input.state_and_history_sha256 != hex(source.state_and_history_digest())
        || descriptor.input.rules_profile_sha256 != hex(source.rules_profile_digest())
        || authority.epoch != context.authority.epoch.0
        || authority.game != context.authority.game.0
        || authority.root != context.authority.root.0
        || authority.implementation_sha256 != hex(context.authority.implementation)
    {
        return Err("pc_v2_initial_scope_mismatch");
    }
    Ok(Some(json!({
        "kind": "original_initial_stop",
        "input_lock_sha256": hex(receipt.input_lock),
        "root_admission_sha256": hex(descriptor.digest().map_err(|_| "pc_v2_initial_digest_invalid")?),
        "root_budget_sha256": hex(descriptor.root_budget_digest().map_err(|_| "pc_v2_initial_budget_invalid")?),
        "deadline_seal_sha256": hex(descriptor.deadline_seal_digest().map_err(|_| "pc_v2_initial_deadline_invalid")?),
        "state_and_history_sha256": hex(source.state_and_history_digest()),
        "original_start_matches": null,
        "root_fence_permits_reuse": null,
        "stop_reason": format!("{:?}", source.stop_reason()),
    })))
}

#[cfg(test)]
mod tests {
    use super::same_deadline;
    use rz_contracts::{ClockDomain, Deadline, MonotonicTick, ProcessEpoch};
    use rz_experiments::RoventRootDeadline;

    #[test]
    fn root_clock_epoch_and_tick_must_both_match() {
        let declared = RoventRootDeadline {
            clock_epoch: 7,
            at_ns: 10,
        };
        let actual = Deadline {
            clock: ClockDomain(ProcessEpoch(7)),
            at: MonotonicTick(10),
        };
        assert!(same_deadline(&declared, actual));
        assert!(!same_deadline(
            &declared,
            Deadline {
                clock: ClockDomain(ProcessEpoch(8)),
                ..actual
            }
        ));
        assert!(!same_deadline(
            &declared,
            Deadline {
                at: MonotonicTick(11),
                ..actual
            }
        ));
    }
}
