//! Explicit validation only. No callbacks, I/O, blocking wait or model policy.
//! A recorded interval is a host ORT call envelope, never kernel-only timing.
use rz_contracts::pals::SearchAuthority;
use rz_contracts::pals_adaptive::RoleTicketEnvelope;
use rz_contracts::{Digest, ExecutionId};
use serde_json::{json, Value};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex, TryLockError,
};
use std::time::Instant;

const CAPACITY: usize = 256;
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RunGraph {
    Public,
    Private,
}
#[derive(Clone, Copy)]
enum Event {
    Admitted(u64, RoleTicketEnvelope, Instant),
    Cancel(u64, Instant, Instant, bool),
    Enter(RoleTicketEnvelope, ExecutionId, RunGraph, Instant),
    Return(
        RoleTicketEnvelope,
        ExecutionId,
        RunGraph,
        Instant,
        Instant,
        Option<bool>,
    ),
    Backend(
        RoleTicketEnvelope,
        ExecutionId,
        Option<Digest>,
        Option<Digest>,
        bool,
        Option<u64>,
        bool,
        Instant,
    ),
    Logical(RoleTicketEnvelope, bool, bool, Instant),
    Retired(RoleTicketEnvelope, bool, bool, Instant),
    Fence(SearchAuthority, bool, Instant),
    Placement(Instant),
    Finished(bool, Instant),
    Joined(Instant),
}
type ActualCudaOwner = ([u8; 32], [u8; 32], [u8; 32], [u8; 32]);
struct State {
    active: Option<(RoleTicketEnvelope, ExecutionId)>,
    owner: Option<ActualCudaOwner>,
    events: Vec<Event>,
}
struct Inner {
    state: Mutex<State>,
    incomplete: AtomicBool,
    origin: Instant,
}
#[derive(Clone)]
pub struct GpuLifecycleTrace(Arc<Inner>);
pub struct TicketScope {
    trace: GpuLifecycleTrace,
    ticket: RoleTicketEnvelope,
    execution: ExecutionId,
}
pub struct OrtRunScope {
    trace: GpuLifecycleTrace,
    ticket: RoleTicketEnvelope,
    execution: ExecutionId,
    graph: RunGraph,
    entered: Instant,
    returned: bool,
}
impl Default for GpuLifecycleTrace {
    fn default() -> Self {
        Self::new()
    }
}
impl GpuLifecycleTrace {
    pub fn new() -> Self {
        Self(Arc::new(Inner {
            state: Mutex::new(State {
                active: None,
                owner: None,
                events: Vec::with_capacity(CAPACITY),
            }),
            incomplete: AtomicBool::new(false),
            origin: Instant::now(),
        }))
    }
    fn update(&self, action: impl FnOnce(&mut State)) {
        match self.0.state.try_lock() {
            Ok(mut state) => action(&mut state),
            Err(_) => self.0.incomplete.store(true, Ordering::Release),
        }
    }
    fn push(&self, event: Event) {
        self.update(|state| {
            if state.events.len() == CAPACITY {
                self.0.incomplete.store(true, Ordering::Release);
            } else {
                state.events.push(event);
            }
        });
    }
    pub fn bind_actual_cuda_owner(
        &self,
        weights: [u8; 32],
        backend: [u8; 32],
        export: [u8; 32],
        runtime: [u8; 32],
    ) {
        self.update(|state| {
            if state.owner.is_some() {
                self.0.incomplete.store(true, Ordering::Release);
            } else {
                state.owner = Some((weights, backend, export, runtime));
            }
        });
    }
    pub fn admitted(&self, preparation: u64, ticket: RoleTicketEnvelope, at: Instant) {
        self.push(Event::Admitted(preparation, ticket, at));
    }
    pub fn cancel_flag_store(
        &self,
        preparation: u64,
        before: Instant,
        after: Instant,
        first: bool,
    ) {
        self.push(Event::Cancel(preparation, before, after, first));
    }
    pub fn enter_ticket(&self, ticket: RoleTicketEnvelope, execution: ExecutionId) -> TicketScope {
        self.update(|state| {
            if state.active.is_some()
                || execution.epoch != ticket.authority.epoch
                || execution.sequence == 0
            {
                self.0.incomplete.store(true, Ordering::Release);
            } else {
                state.active = Some((ticket, execution));
            }
        });
        TicketScope {
            trace: self.clone(),
            ticket,
            execution,
        }
    }
    pub fn enter_ort(&self, graph: RunGraph) -> Option<OrtRunScope> {
        let mut active = None;
        self.update(|state| active = state.active);
        let Some((ticket, execution)) = active else {
            self.0.incomplete.store(true, Ordering::Release);
            return None;
        };
        let entered = Instant::now();
        self.push(Event::Enter(ticket, execution, graph, entered));
        Some(OrtRunScope {
            trace: self.clone(),
            ticket,
            execution,
            graph,
            entered,
            returned: false,
        })
    }
    pub fn has_open_ort_run(&self, ticket: &RoleTicketEnvelope) -> bool {
        match self.0.state.try_lock() {
            Ok(state) => state.events.iter().any(|event| match *event {
                Event::Enter(t, e, g, at) if t == *ticket => !state.events.iter().any(|end| {
                    matches!(*end,
                    Event::Return(t2,e2,g2,at2,_,_) if t2==t && e2==e && g2==g && at2==at)
                }),
                _ => false,
            }),
            // A read-side miss loses no event. Retry observation on a later
            // tick without poisoning an otherwise complete diagnostic trace.
            Err(TryLockError::WouldBlock) => false,
            Err(TryLockError::Poisoned(_)) => {
                self.0.incomplete.store(true, Ordering::Release);
                false
            }
        }
    }
    #[allow(
        clippy::too_many_arguments,
        reason = "records one backend completion event without changing its evidence fields"
    )]
    pub fn backend_completed(
        &self,
        ticket: RoleTicketEnvelope,
        execution: ExecutionId,
        invocation: Option<Digest>,
        seed: Option<Digest>,
        fresh: bool,
        completed: Option<u64>,
        known: bool,
    ) {
        self.push(Event::Backend(
            ticket,
            execution,
            invocation,
            seed,
            fresh,
            completed,
            known,
            Instant::now(),
        ));
    }
    pub fn logical_completed(&self, ticket: RoleTicketEnvelope, physical: bool, canceled: bool) {
        self.push(Event::Logical(ticket, physical, canceled, Instant::now()));
    }
    pub fn retired(&self, ticket: RoleTicketEnvelope, accepted: bool, ok: bool) {
        self.push(Event::Retired(ticket, accepted, ok, Instant::now()));
    }
    pub fn root_fence(&self, authority: SearchAuthority, reusable: bool) {
        self.push(Event::Fence(authority, reusable, Instant::now()));
    }
    pub fn actual_placement_verified(&self) {
        self.push(Event::Placement(Instant::now()));
    }
    pub fn finished(&self, confirmed: bool) {
        self.push(Event::Finished(confirmed, Instant::now()));
    }
    pub fn joined(&self) {
        self.push(Event::Joined(Instant::now()));
    }

    pub fn receipt(&self) -> Value {
        let mut owner = None;
        let mut events = Vec::new();
        let mut active = false;
        self.update(|state| {
            owner = state.owner;
            events = state.events.clone();
            active = state.active.is_some();
        });
        let paired_spans = events.iter().all(|event| match *event {
            Event::Enter(t, x, g, b) => {
                events
                    .iter()
                    .filter(|other| {
                        matches!(**other,
                Event::Return(t2,x2,g2,b2,_,_) if t2==t && x2==x && g2==g && b2==b)
                    })
                    .count()
                    == 1
            }
            Event::Return(t, x, g, b, _, _) => {
                events
                    .iter()
                    .filter(|other| {
                        matches!(**other,
                Event::Enter(t2,x2,g2,b2) if t2==t && x2==x && g2==g && b2==b)
                    })
                    .count()
                    == 1
                    && events
                        .iter()
                        .filter(|other| {
                            matches!(**other,
                    Event::Return(t2,x2,g2,b2,_,_) if t2==t && x2==x && g2==g && b2==b)
                        })
                        .count()
                        == 1
            }
            _ => true,
        });
        let incomplete =
            self.0.incomplete.load(Ordering::Acquire) || active || owner.is_none() || !paired_spans;
        let placements: Vec<_> = events
            .iter()
            .filter_map(|e| match *e {
                Event::Placement(at) => Some(at),
                _ => None,
            })
            .collect();
        let finishes: Vec<_> = events
            .iter()
            .filter_map(|e| match *e {
                Event::Finished(ok, at) => Some((ok, at)),
                _ => None,
            })
            .collect();
        let joins: Vec<_> = events
            .iter()
            .filter_map(|e| match *e {
                Event::Joined(at) => Some(at),
                _ => None,
            })
            .collect();
        let drained = placements.len() == 1
            && finishes.len() == 1
            && joins.len() == 1
            && finishes[0].0
            && placements[0] <= finishes[0].1
            && finishes[0].1 <= joins[0];
        let mut cases = Vec::new();
        for event in &events {
            let Event::Return(ticket, execution, graph, entered, returned, Some(true)) = *event
            else {
                continue;
            };
            let admissions: Vec<_> = events
                .iter()
                .filter_map(|e| match *e {
                    Event::Admitted(id, t, at) if t == ticket => Some((id, at)),
                    _ => None,
                })
                .collect();
            if admissions.len() != 1 {
                continue;
            }
            let (id, admitted) = admissions[0];
            let cancels: Vec<_> = events
                .iter()
                .filter_map(|e| match *e {
                    Event::Cancel(p, b, a, true) if p == id => Some((b, a)),
                    _ => None,
                })
                .collect();
            if cancels.len() != 1 {
                continue;
            }
            let (cancel_before, cancel_after) = cancels[0];
            let ordered = admitted <= entered
                && entered < cancel_before
                && cancel_before <= cancel_after
                && cancel_after < returned;
            let backends: Vec<_> = events
                .iter()
                .filter_map(|e| match *e {
                    Event::Backend(t, x, key, seed, fresh, n, known, at) if t == ticket => {
                        Some((x, key, seed, fresh, n, known, at))
                    }
                    _ => None,
                })
                .collect();
            let logicals: Vec<_> = events
                .iter()
                .filter_map(|e| match *e {
                    Event::Logical(t, physical, canceled, at) if t == ticket => {
                        Some((physical, canceled, at))
                    }
                    _ => None,
                })
                .collect();
            let retirements: Vec<_> = events
                .iter()
                .filter_map(|e| match *e {
                    Event::Retired(t, accepted, ok, at) if t == ticket => Some((accepted, ok, at)),
                    _ => None,
                })
                .collect();
            let fences: Vec<_> = events
                .iter()
                .filter_map(|e| match *e {
                    Event::Fence(a, reusable, at) if a == ticket.authority => Some((reusable, at)),
                    _ => None,
                })
                .collect();
            let backend = backends.len() == 1
                && backends[0].0 == execution
                && backends[0].1 == Some(ticket.representation.input)
                && backends[0].2.is_none()
                && backends[0].3
                && backends[0].4.is_some_and(|n| n > 0)
                && backends[0].5;
            let logical = logicals.len() == 1 && logicals[0].0 && logicals[0].1;
            let retire = retirements.len() == 1 && !retirements[0].0 && retirements[0].1;
            let fence = fences.len() == 1 && fences[0].0;
            let terminal_order = backend
                && logical
                && retire
                && fence
                && drained
                && returned <= backends[0].6
                && backends[0].6 <= logicals[0].2
                && logicals[0].2 <= retirements[0].2
                && retirements[0].2 <= fences[0].1
                && fences[0].1 <= placements[0]
                && placements[0] <= finishes[0].1
                && finishes[0].1 <= joins[0];
            let owner_match =
                owner.is_some_and(|o| o.0 == ticket.representation.model.0 && o.1 != [0; 32]);
            let known_clock = [admitted, entered, cancel_before, cancel_after, returned]
                .iter()
                .all(|at| ns(self.0.origin, *at).is_some());
            if ordered && terminal_order && known_clock && owner_match && !incomplete {
                cases.push(json!({"ticket":lineage(&ticket),"execution":{"epoch":execution.epoch.0,"sequence":execution.sequence},
                    "graph":format!("{graph:?}"),"run_enter_ns":ns(self.0.origin,entered),
                    "cancel_before_ns":ns(self.0.origin,cancel_before),"cancel_after_ns":ns(self.0.origin,cancel_after),
                    "run_return_ns":ns(self.0.origin,returned),"native_retired_accepted":false}));
            }
        }
        let status = if incomplete {
            "TraceIncomplete"
        } else if !drained {
            "DrainUnconfirmed"
        } else if cases.is_empty() {
            "NotObserved"
        } else {
            "Observed"
        };
        json!({"schema":"rz-current-gpu-inflight-validation/1","scope":"same_ticket_host_CUDA_ORT_envelope_cancel_then_unaccepted_completion_and_drain",
            "status":status,"cases":cases,"trace_complete":!incomplete,"native_drain_and_join_confirmed":drained,
            "owner":owner.map(|(w,b,e,r)|json!({"weights":w,"backend":b,"export":e,"runtime":r})),
            "events":events.iter().map(|event|event_json(event,self.0.origin)).collect::<Vec<_>>(),
            "gpu_kernel_timing_attested":false,"performance_evidence":false,"numerical_acceptance":false,
            "new_execution_authority_issued":false})
    }
}
impl Drop for TicketScope {
    fn drop(&mut self) {
        self.trace.update(|state| {
            if state.active == Some((self.ticket, self.execution)) {
                state.active = None;
            } else {
                self.trace.0.incomplete.store(true, Ordering::Release);
            }
        });
    }
}
impl OrtRunScope {
    pub fn returned(mut self, ok: bool) {
        self.trace.push(Event::Return(
            self.ticket,
            self.execution,
            self.graph,
            self.entered,
            Instant::now(),
            Some(ok),
        ));
        self.returned = true;
    }
}
impl Drop for OrtRunScope {
    fn drop(&mut self) {
        if !self.returned {
            self.trace.push(Event::Return(
                self.ticket,
                self.execution,
                self.graph,
                self.entered,
                Instant::now(),
                None,
            ));
            self.trace.0.incomplete.store(true, Ordering::Release);
        }
    }
}
fn ns(origin: Instant, value: Instant) -> Option<u64> {
    value
        .checked_duration_since(origin)
        .and_then(|v| v.as_nanos().try_into().ok())
}
fn lineage(t: &RoleTicketEnvelope) -> Value {
    json!({"request":{"epoch":t.id.epoch.0,"sequence":t.id.sequence},
        "authority":{"epoch":t.authority.epoch.0,"game":t.authority.game.0,"root":t.authority.root.0,"implementation":t.authority.implementation.0},
        "question":t.question.0,"view":{"sequence":t.view.view_sequence,"digest":t.view.view_digest.0,"policy":t.view.context_policy.0,
            "root_state":{"owner":t.view.root_state.owner.0,"revision":t.view.root_state.revision.0,"semantic":t.view.root_state.semantic.0},
            "actual_tokens":t.view.actual_tokens,"selected_records":t.view.selected_records,"anchors":t.view.anchors,"length_bucket":t.view.length_bucket},
        "representation":{"input":t.representation.input.0,"history":t.representation.history.0,"candidates":t.representation.candidates.0,
            "mask":t.representation.mask_positions.0,"model":t.representation.model.0,"encoding":t.representation.encoding.0,
            "precision":format!("{:?}",t.representation.precision),"frozen_epoch":t.representation.frozen_epoch,"record_revision":t.representation.record_revision},
        "role":format!("{:?}",t.role),"deadline":{"clock_epoch":t.original_deadline.clock.0.0,"at":t.original_deadline.at.0}})
}
fn event_json(e: &Event, origin: Instant) -> Value {
    match *e {
        Event::Admitted(p, t, at) => {
            json!({"kind":"admitted","preparation":p,"ticket":lineage(&t),"at_ns":ns(origin,at)})
        }
        Event::Cancel(p, b, a, first) => {
            json!({"kind":"cancel_flag_store","preparation":p,"before_ns":ns(origin,b),"after_ns":ns(origin,a),"first_transition":first})
        }
        Event::Enter(t, x, g, at) => {
            json!({"kind":"ort_enter","ticket":lineage(&t),"execution":[x.epoch.0,x.sequence],"graph":format!("{g:?}"),"at_ns":ns(origin,at)})
        }
        Event::Return(t, x, g, b, a, ok) => {
            json!({"kind":"ort_return","ticket":lineage(&t),"execution":[x.epoch.0,x.sequence],"graph":format!("{g:?}"),"begin_ns":ns(origin,b),"at_ns":ns(origin,a),"ok":ok})
        }
        Event::Backend(t, x, k, seed, f, n, known, at) => {
            json!({"kind":"backend_completed","ticket":lineage(&t),"execution":[x.epoch.0,x.sequence],"invocation":k.map(|v|v.0),"accepted_seed":seed.map(|v|v.0),"fresh":f,"completed_nn_inputs":n,"measurement_complete":known,"at_ns":ns(origin,at)})
        }
        Event::Logical(t, p, c, at) => {
            json!({"kind":"logical_completed","ticket":lineage(&t),"physical_confirmed":p,"canceled":c,"at_ns":ns(origin,at)})
        }
        Event::Retired(t, a, ok, at) => {
            json!({"kind":"retired","ticket":lineage(&t),"accepted":a,"ok":ok,"at_ns":ns(origin,at)})
        }
        Event::Fence(a, r, at) => {
            json!({"kind":"root_fence","authority":[a.epoch.0,a.game.0,a.root.0],"reusable":r,"at_ns":ns(origin,at)})
        }
        Event::Placement(at) => json!({"kind":"actual_placement_verified","at_ns":ns(origin,at)}),
        Event::Finished(c, at) => {
            json!({"kind":"finished","physical_shutdown_confirmed":c,"at_ns":ns(origin,at)})
        }
        Event::Joined(at) => json!({"kind":"actual_native_thread_joined","at_ns":ns(origin,at)}),
    }
}

#[cfg(test)]
mod tests;
