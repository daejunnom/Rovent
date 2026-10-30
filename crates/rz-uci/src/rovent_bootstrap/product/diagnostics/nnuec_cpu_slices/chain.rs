//! Separate, Serialize-only observations. No existing receipt/domain changes.
use super::*;
use crate::pals_adaptive::{NnuecChainEvent, NnuecChainObservation};
use rz_contracts::pals_adaptive::{ReviewViewSeal, RoleTicketEnvelope};
use rz_contracts::{Deadline, StateIdentity};
use rz_search::pals::coordinator::FrameId;
use serde_json::{Value, json};

const CHAIN_DOMAIN: &str = "rz-rovent-nnuec-accepted-critic-chain-observation/1";

#[derive(Serialize)]
struct ChainRow<'a> {
    domain: &'static str,
    event: &'static str,
    input_lock_sha256: String,
    implementation_sha256: String,
    observed_after_authority: AuthorityObservation,
    root_admission_sha256: Option<&'a str>,
    source_status: &'a str,
    snapshot_read_failure: Option<&'a str>,
    retained_prefix_records: Option<usize>,
    dropped_records: Option<u64>,
    original: Option<Value>,
    // Raw events permit a consumer to check joins; no boolean substitutes for
    // a complete chain or for a worker-computed scalar readout.
    accepted_critic_chain_observed: Option<bool>,
    conditional_scalar_value_observed: Option<bool>,
    physical_completion_attested: bool,
    execution_authorized: bool,
    token_or_job_authority_issued: bool,
    performance_evidence: bool,
    full_acceptance_issued: bool,
}

impl RoventLifecycleDiagnostics {
    pub(super) fn observe_nnuec_chain(
        &self,
        context: &SearchSessionContext,
        status: &str,
        failure: Option<&str>,
        snapshot: Option<&CpuSliceOwnerObservation>,
    ) {
        let mut row = ChainRow {
            domain: CHAIN_DOMAIN,
            event: "root_summary",
            input_lock_sha256: hex(self.lock),
            implementation_sha256: hex(self.implementation),
            observed_after_authority: context.authority.into(),
            root_admission_sha256: snapshot.map(|s| s.summary.root_admission_sha256.as_str()),
            source_status: status,
            snapshot_read_failure: failure,
            retained_prefix_records: snapshot
                .and_then(|s| s.nnuec_chain_dropped.map(|_| s.nnuec_chain.len())),
            dropped_records: snapshot.and_then(|s| s.nnuec_chain_dropped),
            original: None,
            accepted_critic_chain_observed: None,
            conditional_scalar_value_observed: None,
            physical_completion_attested: false,
            execution_authorized: false,
            token_or_job_authority_issued: false,
            performance_evidence: false,
            full_acceptance_issued: false,
        };
        let Ok(mut state) = self.state.lock() else {
            self.poisoned.store(true, Ordering::Release);
            return;
        };
        if state.root_runs > MAX_ROOT_ROWS {
            state.scope_incomplete = true;
            state.first_issue.get_or_insert("nnuec_chain_root_limit");
            return;
        }
        let result = (|| {
            write_slice_row(&mut io::stderr().lock(), &row, &mut state)?;
            if let Some(snapshot) = snapshot {
                for original in &snapshot.nnuec_chain {
                    row.event = match original.event {
                        NnuecChainEvent::AcceptedCriticChild(_) => {
                            "accepted_critic_researcher_child"
                        }
                        NnuecChainEvent::NnuecJobBound(_) => "acknowledged_nnuec_job_bound",
                    };
                    row.original = Some(project(original));
                    write_slice_row(&mut io::stderr().lock(), &row, &mut state)?;
                }
            }
            Ok::<(), &'static str>(())
        })();
        if let Err(cause) = result {
            state.scope_incomplete = true;
            state.first_issue.get_or_insert(cause);
        }
    }
}

fn authority(value: SearchAuthority) -> Value {
    json!({"epoch":value.epoch.0,"game":value.game.0,"root":value.root.0,
        "implementation_sha256":hex(value.implementation)})
}
fn frame(value: FrameId) -> Value {
    json!({"authority":authority(value.authority),"sequence":value.sequence})
}
fn state_identity(value: StateIdentity) -> Value {
    json!({"owner":value.owner.0,"revision":value.revision.0,"semantic_sha256":hex(value.semantic)})
}
fn deadline(value: Deadline) -> Value {
    json!({"clock_epoch":value.clock.0.0,"tick_ns":value.at.0})
}
fn view(value: ReviewViewSeal) -> Value {
    json!({"authority":authority(value.authority),"view_sequence":value.view_sequence,
        "view_sha256":hex(value.view_digest),"root_state":state_identity(value.root_state),
        "context_policy_sha256":hex(value.context_policy),"actual_tokens":value.actual_tokens,
        "selected_records":value.selected_records,"anchors":value.anchors,"length_bucket":value.length_bucket})
}
fn ticket(value: RoleTicketEnvelope) -> Value {
    let rep = value.representation;
    json!({"request_epoch":value.id.epoch.0,"request_sequence":value.id.sequence,
        "authority":authority(value.authority),"question_sha256":hex(value.question),
        "view":view(value.view),"role":format!("{:?}",value.role),
        "original_deadline":deadline(value.original_deadline),
        "representation":{"input_sha256":hex(rep.input),"history_sha256":hex(rep.history),
            "candidates_sha256":hex(rep.candidates),"mask_positions_sha256":hex(rep.mask_positions),
            "model_sha256":hex(rep.model),"encoding_sha256":hex(rep.encoding),
            "precision":format!("{:?}",rep.precision),"frozen_epoch":rep.frozen_epoch,
            "record_revision":rep.record_revision,"role":format!("{:?}",rep.role)}})
}
fn project(original: &NnuecChainObservation) -> Value {
    match original.event {
        NnuecChainEvent::AcceptedCriticChild(value) => {
            let q = value.question;
            json!({"sequence":original.sequence,"source_frame":frame(value.source_frame),
                "accepted_ticket":ticket(value.ticket),"accepted_output_sha256":hex(value.output),
                "accepted_origin_seal_sha256":hex(value.origin_seal),
                "child_frame":frame(value.child),"joined_existing_child":value.joined,
                "question":{"authority":authority(q.id.authority),"sequence":q.id.sequence,
                    "parent_sequence":q.parent.map(|p|p.sequence),
                    "parent_authority":q.parent.map(|p|authority(p.authority)),
                    "sender":format!("{:?}",q.sender),"recipient":format!("{:?}",q.recipient),
                    "purpose":format!("{:?}",q.purpose),"anchor":state_identity(q.anchor),
                    "anchor_root_ply":q.anchor_root_ply.0,
                    "original_first_move_bits":q.original_first_move.map(|movement|movement.bits()),
                    "prefix_sha256":hex(q.prefix_digest),
                    "line_sha256":q.line_digest.map(hex),"line_offset":q.line_offset.0,
                    "question_sha256":hex(q.question_digest),"view":view(q.view),
                    "evidence_revision":q.evidence_revision,"root_budget_sha256":hex(q.root_budget),
                    "original_deadline":deadline(q.original_deadline)},
                "actual_child_view":value.actual_child_view.map(view),
                "actual_child_public_context_sha256":value.actual_child_public_context.map(hex),
                "published_record_revision":value.published_record_revision,
                "selected_record_index":value.selected_record_index,
                "selected_record_count":value.selected_record_count,
                "scalar_readout":null})
        }
        NnuecChainEvent::NnuecJobBound(value) => {
            let job = value.job;
            let grant = value.first_grant;
            json!({"sequence":original.sequence,"frame":frame(value.frame),
                "question_sha256":hex(value.question),"view":view(value.view),
                "full_view_sha256":hex(value.full_view),
                "job":{"authority":authority(job.id.authority),"sequence":job.id.sequence,
                    "state":state_identity(job.state_identity),"question_sha256":hex(job.question),
                    "root_restriction_sha256":hex(job.root_restriction),
                    "evaluation_namespace_sha256":hex(job.evaluation_namespace),
                    "search_conditions_sha256":hex(job.search_conditions),
                    "total_native_node_credit":job.total_native_node_credit,
                    "hard_deadline":deadline(job.hard_deadline)},
                "first_grant":{"job_sequence":grant.job.sequence,
                    "job_authority":authority(grant.job.authority),
                    "execution_epoch":grant.execution.epoch.0,"execution_sequence":grant.execution.sequence,
                    "owner":grant.owner.0,"node_credit":grant.node_credit,
                    "soft_lease_ns":u64::try_from(grant.soft_lease.as_nanos()).ok(),
                    "control_sequence":grant.control_sequence},
                "installation_sequence":value.installation_sequence,
                "installation_succeeded":value.installation_succeeded,
                "binding_sha256":hex(value.binding),
                "installed_owner":value.installed_owner.map(|owner|owner.0),
                "actual_stack_owner_id":value.actual_stack_owner_id,
                "begin_enqueued_and_frame_bound":true,"worker_invocation_observed":null,
                "scalar_readout":null,"same_job_resume_observed":null})
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn missing_original_chain_and_scalar_do_not_become_success() {
        let mut state = State::default();
        let mut output = Vec::new();
        let raw = json!({"domain":CHAIN_DOMAIN,"original":null,
            "accepted_critic_chain_observed":null,"conditional_scalar_value_observed":null});
        write_slice_row(&mut output, &raw, &mut state).unwrap();
        assert!(output.len() <= MAX_ROW_BYTES);
        let encoded: Value = serde_json::from_slice(&output).unwrap();
        assert!(encoded["original"].is_null());
        assert!(encoded["conditional_scalar_value_observed"].is_null());
    }
    #[test]
    fn chain_writer_keeps_the_existing_final_row_reservation() {
        let mut state = State {
            bytes: MAX_STREAM_BYTES - MAX_ROW_BYTES,
            ..State::default()
        };
        let mut output = Vec::new();
        assert!(write_slice_row(&mut output, &json!({"domain":CHAIN_DOMAIN}), &mut state).is_err());
        assert!(output.is_empty());
        assert_eq!(state.bytes, MAX_STREAM_BYTES - MAX_ROW_BYTES);
    }
}
