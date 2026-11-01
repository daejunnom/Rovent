//! Bounded opt-in observation of actual P/C ticket inputs and raw outputs.
//!
//! This is not numerical, tensor-byte, provider-placement, or strength
//! acceptance. No file I/O or waiting for a contended lock occurs on the
//! native ticket owner. The caller alone publishes the returned JSON bytes.
use crate::pals_native::ticketed::{
    NativeTicketQuestion, TicketExecutionEvidence, TicketPreparation, VirtualReviewCaptureFailure,
    VirtualReviewCollectionObserver, VirtualReviewPhysicalInvocationEvidence,
    VirtualReviewPreparedTicket,
};
use rz_contracts::pals::{Role, RoleOutput};
use rz_contracts::pals_adaptive::RoleTicketEnvelope;
use rz_eval::pals_model::{PalsRawOutput, PalsRole};
use rz_search::pals::coordinator::{PhysicalRoleCompletion, RoleFailure};
use rz_search::pals::review_context::ReviewView;
use serde::Serialize;
use std::io::{self, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

const DOMAIN: &str = "rz-rovent-system-numeric-observation/1";
const MAX_PENDING: usize = 16;
const MAX_BYTES: usize = 4 * 1024 * 1024;

struct LimitedBuffer {
    bytes: Vec<u8>,
    limit: usize,
}
impl Write for LimitedBuffer {
    fn write(&mut self, input: &[u8]) -> io::Result<usize> {
        if input.len() > self.limit.saturating_sub(self.bytes.len()) {
            return Err(io::Error::other("numeric observation byte limit"));
        }
        self.bytes.extend_from_slice(input);
        Ok(input.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
fn encode_bounded<T: Serialize>(value: &T, limit: usize) -> Result<Vec<u8>, serde_json::Error> {
    let mut buffer = LimitedBuffer {
        bytes: Vec::new(),
        limit,
    };
    serde_json::to_writer(&mut buffer, value)?;
    Ok(buffer.bytes)
}
fn hex(bytes: &[u8; 32]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}
fn role_index(role: Role) -> Option<usize> {
    match role {
        Role::Proposer => Some(0),
        Role::Critic => Some(1),
        Role::Validator => None,
    }
}
fn role_matches(role: Role, input: PalsRole) -> bool {
    matches!(
        (role, input),
        (Role::Proposer, PalsRole::Proposer) | (Role::Critic, PalsRole::Critic)
    )
}
fn raw_is_finite(raw: &PalsRawOutput) -> bool {
    raw.candidate_logits
        .iter()
        .chain(raw.wdl_logits.iter())
        .chain(raw.divergence_logits.iter().flatten())
        .chain(raw.task_logits.iter().flatten())
        .chain(raw.private_latent.iter())
        .all(|value| value.is_finite())
}

struct Pending {
    ticket: RoleTicketEnvelope,
    model_epoch: [u8; 32],
    input: Vec<u8>,
}
struct Capture {
    pending: Pending,
    completed_nn_inputs: u64,
    invocation_key: [u8; 32],
    raw: Vec<u8>,
}
#[derive(Default)]
struct State {
    pending: Vec<Pending>,
    captures: [Option<Capture>; 2],
    retained_bytes: usize,
}
#[derive(Default)]
struct Shared {
    state: Mutex<State>,
    unavailable: AtomicBool,
}
struct Observer(Arc<Shared>);

/// The handle belongs to the caller; dropping it never acknowledges a ticket.
pub(super) struct Handle(Arc<Shared>);

pub(super) fn attach() -> (Box<dyn VirtualReviewCollectionObserver>, Handle) {
    let shared = Arc::new(Shared::default());
    (Box::new(Observer(Arc::clone(&shared))), Handle(shared))
}

impl VirtualReviewCollectionObserver for Observer {
    fn submitted(
        &mut self,
        ticket: &RoleTicketEnvelope,
        _: &TicketPreparation<ReviewView, NativeTicketQuestion>,
        prepared: &VirtualReviewPreparedTicket,
    ) -> Result<(), RoleFailure> {
        let Some(index) = role_index(ticket.role) else {
            return Ok(());
        };
        if !role_matches(ticket.role, prepared.input.role) {
            self.0.unavailable.store(true, Ordering::Relaxed);
            return Ok(());
        }
        let Ok(mut state) = self.0.state.try_lock() else {
            self.0.unavailable.store(true, Ordering::Relaxed);
            return Ok(());
        };
        if state.captures[index].is_some() {
            return Ok(());
        }
        if state.pending.len() >= MAX_PENDING
            || state.pending.iter().any(|p| p.ticket.id == ticket.id)
        {
            self.0.unavailable.store(true, Ordering::Relaxed);
            return Ok(());
        }
        let remaining = MAX_BYTES.saturating_sub(state.retained_bytes);
        match encode_bounded(&prepared.input, remaining) {
            Ok(input) => {
                state.retained_bytes += input.len();
                state.pending.push(Pending {
                    ticket: *ticket,
                    model_epoch: prepared.identity.model_epoch,
                    input,
                });
            }
            Err(_) => self.0.unavailable.store(true, Ordering::Relaxed),
        }
        Ok(())
    }

    fn completed(
        &mut self,
        _: &RoleTicketEnvelope,
        _: PhysicalRoleCompletion,
        _: Option<&RoleOutput>,
        _: Option<TicketExecutionEvidence>,
        _: Option<VirtualReviewPhysicalInvocationEvidence>,
    ) -> Result<(), RoleFailure> {
        Ok(())
    }

    fn completed_with_raw_classified(
        &mut self,
        ticket: &RoleTicketEnvelope,
        physical: PhysicalRoleCompletion,
        output: Option<&RoleOutput>,
        evidence: Option<TicketExecutionEvidence>,
        invocation: Option<VirtualReviewPhysicalInvocationEvidence>,
        raw: Option<&PalsRawOutput>,
    ) -> Result<(), VirtualReviewCaptureFailure> {
        let Ok(mut state) = self.0.state.try_lock() else {
            self.0.unavailable.store(true, Ordering::Relaxed);
            return Ok(());
        };
        let Some(position) = state
            .pending
            .iter()
            .position(|pending| pending.ticket == *ticket)
        else {
            return Ok(());
        };
        let pending = state.pending.swap_remove(position);
        let Some(index) = role_index(ticket.role) else {
            state.retained_bytes -= pending.input.len();
            return Ok(());
        };
        let eligible = evidence.filter(|e| {
            e.measurement_complete
                && e.physical_fresh == Some(true)
                && e.completed_nn_inputs.is_some_and(|n| n > 0)
        });
        if physical != PhysicalRoleCompletion::Confirmed
            || output.is_none_or(|value| {
                value.id != ticket.id
                    || value.authority != ticket.authority
                    || value.key != ticket.representation
            })
            || raw.is_none()
            || invocation.is_none_or(|i| !i.physical_fresh)
            || eligible.is_none()
            || state.captures[index].is_some()
        {
            state.retained_bytes -= pending.input.len();
            return Ok(());
        }
        let raw = raw.expect("checked above");
        if !raw_is_finite(raw) {
            state.retained_bytes -= pending.input.len();
            self.0.unavailable.store(true, Ordering::Relaxed);
            return Ok(());
        }
        match encode_bounded(raw, MAX_BYTES.saturating_sub(state.retained_bytes)) {
            Ok(raw) => {
                state.retained_bytes += raw.len();
                state.captures[index] = Some(Capture {
                    pending,
                    completed_nn_inputs: eligible
                        .and_then(|e| e.completed_nn_inputs)
                        .expect("checked above"),
                    invocation_key: invocation.expect("checked above").invocation_key.0,
                    raw,
                });
            }
            Err(_) => {
                state.retained_bytes -= pending.input.len();
                self.0.unavailable.store(true, Ordering::Relaxed);
            }
        }
        Ok(())
    }

    fn retired(&mut self, ticket: &RoleTicketEnvelope, _: bool) -> Result<(), RoleFailure> {
        let Ok(mut state) = self.0.state.try_lock() else {
            self.0.unavailable.store(true, Ordering::Relaxed);
            return Ok(());
        };
        if let Some(index) = state.pending.iter().position(|p| p.ticket == *ticket) {
            let pending = state.pending.swap_remove(index);
            state.retained_bytes -= pending.input.len();
        }
        Ok(())
    }
}

impl Handle {
    /// Copies at most two bounded samples after caller-observed session finish.
    /// A busy/poisoned lock yields an explicit unavailable record, never wait.
    pub(super) fn to_json(
        &self,
        input_lock_sha256: &str,
        process_epoch: u64,
        session_physical_confirmed: bool,
    ) -> Result<Vec<u8>, serde_json::Error> {
        let valid_lock = input_lock_sha256.len() == 64
            && input_lock_sha256
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte));
        let mut unavailable = self.0.unavailable.load(Ordering::Relaxed) || !valid_lock;
        let mut samples = Vec::new();
        if let Ok(state) = self.0.state.try_lock() {
            for (index, item) in state.captures.iter().enumerate() {
                let Some(capture) = item else { continue };
                if capture.pending.ticket.authority.epoch.0 != process_epoch {
                    unavailable = true;
                    continue;
                }
                let ticket = &capture.pending.ticket;
                samples.push(serde_json::json!({
                    "role": if index == 0 { "proposer" } else { "critic" },
                    "ticket": {"epoch": ticket.id.epoch.0, "sequence": ticket.id.sequence},
                    "authority": {"epoch": ticket.authority.epoch.0, "game": ticket.authority.game.0,
                        "root": ticket.authority.root.0, "implementation_sha256": hex(&ticket.authority.implementation.0)},
                    "model_epoch_sha256": hex(&capture.pending.model_epoch),
                    "physical_fresh": true,
                    "invocation_key_sha256": hex(&capture.invocation_key),
                    "completed_nn_inputs": capture.completed_nn_inputs,
                    "input": serde_json::from_slice::<serde_json::Value>(&capture.pending.input)?,
                    "raw_output": serde_json::from_slice::<serde_json::Value>(&capture.raw)?,
                }));
            }
        } else {
            unavailable = true;
        }
        let status = if unavailable {
            "unavailable"
        } else if samples.is_empty() {
            "unknown"
        } else if samples.len() == 2 {
            "complete"
        } else {
            "partial"
        };
        encode_bounded(
            &serde_json::json!({
            "domain": DOMAIN, "input_lock_sha256": valid_lock.then_some(input_lock_sha256), "process_epoch": process_epoch,
                "session_physical_confirmed": session_physical_confirmed, "status": status,
                "unavailable": unavailable, "captures": samples,
                "scope": {"tensor_bytes_verified": false, "placement_verified": false, "numerical_acceptance": false},
            }),
            MAX_BYTES + 16 * 1024,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rz_contracts::pals::{RepresentationKey, RolePayload, SearchAuthority};
    use rz_contracts::pals_adaptive::ReviewViewSeal;
    use rz_contracts::{
        ClockDomain, Deadline, Digest, GameGeneration, MonotonicTick, OwnerId, PrecisionProfile,
        ProcessEpoch, RequestId, RootGeneration, StateIdentity, StateRevision, Wdl,
    };

    fn ticket(sequence: u64) -> RoleTicketEnvelope {
        let epoch = ProcessEpoch(7);
        let digest = Digest([3; 32]);
        let authority = SearchAuthority {
            epoch,
            game: GameGeneration(1),
            root: RootGeneration(2),
            implementation: digest,
        };
        let representation = RepresentationKey {
            input: digest,
            history: digest,
            candidates: digest,
            mask_positions: digest,
            model: digest,
            encoding: digest,
            precision: PrecisionProfile::Fp32,
            frozen_epoch: 1,
            record_revision: 1,
            role: Role::Proposer,
        };
        RoleTicketEnvelope {
            id: RequestId::new(epoch, sequence),
            authority,
            question: digest,
            view: ReviewViewSeal {
                authority,
                view_sequence: 1,
                context_policy: digest,
                view_digest: digest,
                root_state: StateIdentity {
                    owner: OwnerId(1),
                    revision: StateRevision(1),
                    semantic: digest,
                },
                actual_tokens: 1,
                selected_records: 0,
                anchors: 1,
                length_bucket: 8,
            },
            representation,
            role: Role::Proposer,
            original_deadline: Deadline {
                clock: ClockDomain(epoch),
                at: MonotonicTick(1),
            },
        }
    }

    fn pending(shared: &Arc<Shared>, ticket: RoleTicketEnvelope) {
        let input = br#"{"schema":"test-input"}"#.to_vec();
        let mut state = shared.state.lock().unwrap();
        state.retained_bytes += input.len();
        state.pending.push(Pending {
            ticket,
            model_epoch: [4; 32],
            input,
        });
    }

    #[test]
    fn only_same_confirmed_fresh_completed_ticket_captures_raw() {
        let shared = Arc::new(Shared::default());
        let mut observer = Observer(Arc::clone(&shared));
        let first = ticket(1);
        let other = ticket(2);
        let output = RoleOutput {
            id: first.id,
            authority: first.authority,
            key: first.representation,
            payload: RolePayload::Proposal {
                candidate_logits: vec![0.25],
                wdl: Wdl::try_new(0.0, 1.0, 0.0, 0.001).unwrap(),
            },
            private_latent: vec![],
        };
        let evidence = TicketExecutionEvidence {
            execution: None,
            completed_nn_inputs: Some(1),
            measurement_complete: true,
            physical_fresh: Some(true),
            accounting: None,
        };
        let invocation = VirtualReviewPhysicalInvocationEvidence {
            invocation_key: Digest([5; 32]),
            accepted_seed_digest: None,
            physical_fresh: true,
        };
        let raw = PalsRawOutput {
            candidate_logits: vec![0.25],
            wdl_logits: [0.1, 0.2, 0.3],
            divergence_logits: None,
            task_logits: None,
            private_latent: vec![0.5],
        };
        pending(&shared, first);
        observer
            .completed_with_raw_classified(
                &other,
                PhysicalRoleCompletion::Confirmed,
                Some(&output),
                Some(evidence),
                Some(invocation),
                Some(&raw),
            )
            .unwrap();
        assert!(shared.state.lock().unwrap().captures[0].is_none());
        observer
            .completed_with_raw_classified(
                &first,
                PhysicalRoleCompletion::Unknown,
                Some(&output),
                Some(evidence),
                Some(invocation),
                Some(&raw),
            )
            .unwrap();
        assert!(shared.state.lock().unwrap().captures[0].is_none());

        pending(&shared, first);
        let mut warm = invocation;
        warm.physical_fresh = false;
        observer
            .completed_with_raw_classified(
                &first,
                PhysicalRoleCompletion::Confirmed,
                Some(&output),
                Some(evidence),
                Some(warm),
                Some(&raw),
            )
            .unwrap();
        assert!(shared.state.lock().unwrap().captures[0].is_none());

        pending(&shared, first);
        let mut no_inputs = evidence;
        no_inputs.completed_nn_inputs = Some(0);
        observer
            .completed_with_raw_classified(
                &first,
                PhysicalRoleCompletion::Confirmed,
                Some(&output),
                Some(no_inputs),
                Some(invocation),
                Some(&raw),
            )
            .unwrap();
        assert!(shared.state.lock().unwrap().captures[0].is_none());

        pending(&shared, first);
        observer
            .completed_with_raw_classified(
                &first,
                PhysicalRoleCompletion::Confirmed,
                Some(&output),
                Some(evidence),
                Some(invocation),
                Some(&raw),
            )
            .unwrap();
        let handle = Handle(shared);
        let bytes = handle.to_json(&"a".repeat(64), 7, false).unwrap();
        let value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(value["captures"][0]["ticket"]["sequence"], 1);
        assert_eq!(value["captures"][0]["input"]["schema"], "test-input");
        assert_eq!(
            value["captures"][0]["raw_output"]["candidate_logits"][0],
            0.25
        );
        assert_eq!(value["session_physical_confirmed"], false);
        assert_eq!(value["scope"]["numerical_acceptance"], false);
    }

    #[test]
    fn absent_and_busy_capture_are_distinct_and_never_assert_numerical_acceptance() {
        let (_, handle) = attach();
        let bytes = handle.to_json(&"a".repeat(64), 7, false).unwrap();
        let value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(value["status"], "unknown");
        assert_eq!(value["session_physical_confirmed"], false);
        assert_eq!(value["scope"]["numerical_acceptance"], false);
        let _held = handle.0.state.lock().unwrap();
        let busy = handle.to_json(&"a".repeat(64), 7, false).unwrap();
        let value: serde_json::Value = serde_json::from_slice(&busy).unwrap();
        assert_eq!(value["status"], "unavailable");
        assert!(value["captures"].as_array().unwrap().is_empty());
    }

    #[test]
    fn capture_serialization_refuses_oversized_or_nonfinite_raw() {
        assert!(encode_bounded(&vec![0u8; 64], 16).is_err());
        let raw = PalsRawOutput {
            candidate_logits: vec![f32::NAN],
            wdl_logits: [0.0; 3],
            divergence_logits: None,
            task_logits: None,
            private_latent: vec![],
        };
        assert!(!raw_is_finite(&raw));
    }
}
