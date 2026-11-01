//! Opt-in, selection-call-only capture of raw owned-CPU observations.
//!
//! This deliberately narrow projection is not a complete provenance sidecar,
//! a teacher/physical-completion proof, or permission to publish/train. A
//! selected model evidence body, external checker, or unsupported observation
//! leaves this capture unavailable; the original view remains usable.
use super::*;
use crate::cpu_checker::CheckerIdentity;
use crate::pals::store::{LineChunk, LineId, Observation, RawScore};
use serde::Serialize;
use std::io::{self, Write};
use std::time::Instant;

#[derive(Clone, Copy, Debug)]
pub struct CaptureLimits {
    pub max_observations: usize,
    pub max_line_chunks: usize,
    pub max_states: usize,
    /// Requested bytes for buffers and fixed-size retained entries. Does not
    /// measure allocator overhead, Arc sharing, or process RSS.
    pub max_owned_bytes: usize,
    /// Conservative per-snapshot history charge, including shared histories.
    pub max_retained_history_bytes: usize,
}
pub struct CaptureControl<'a> {
    /// The caller must pass the original root deadline and cancellation flag.
    /// These values themselves do not attest that provenance.
    pub deadline: Instant,
    pub canceled: &'a AtomicBool,
    pub limits: CaptureLimits,
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CaptureUnavailable {
    MissingOriginalControl,
    InvalidLimits,
    Canceled,
    Deadline,
    Capacity,
    UnsupportedSelectedEvidence,
    UnsupportedObservation,
    MissingSource,
    DuplicateObservation,
    Store(StoreError),
    Serialization,
}

/// Detached raw data only; no selection authority, replay acceptance, CPU
/// physical-completion claim, or right to publish is constructed here.
pub struct SelectedCpuRawCapture {
    source: Arc<CohortCpu7OriginalBundle>,
    binding: [u8; 32],
    observations_json: Vec<u8>,
    states: Vec<(StateId, PositionSnapshot)>,
    lines: Vec<(LineId, LineChunk)>,
    requested_owned_bytes: usize,
    retained_history_bytes: usize,
}
impl SelectedCpuRawCapture {
    pub fn original_bundle(&self) -> &CohortCpu7OriginalBundle {
        &self.source
    }
    pub fn view_binding(&self) -> [u8; 32] {
        self.binding
    }
    pub fn observations_json(&self) -> &[u8] {
        &self.observations_json
    }
    pub fn states(&self) -> &[(StateId, PositionSnapshot)] {
        &self.states
    }
    pub fn lines(&self) -> &[(LineId, LineChunk)] {
        &self.lines
    }
    pub fn requested_owned_bytes(&self) -> usize {
        self.requested_owned_bytes
    }
    pub fn retained_history_bytes(&self) -> usize {
        self.retained_history_bytes
    }
}

struct BudgetWriter<'a> {
    bytes: Vec<u8>,
    charged: usize,
    control: &'a CaptureControl<'a>,
    error: Option<CaptureUnavailable>,
}
impl BudgetWriter<'_> {
    fn check(&self) -> Result<(), CaptureUnavailable> {
        if self.control.canceled.load(Ordering::Acquire) {
            return Err(CaptureUnavailable::Canceled);
        }
        if Instant::now() >= self.control.deadline {
            return Err(CaptureUnavailable::Deadline);
        }
        Ok(())
    }
    fn charge(&mut self, bytes: usize) -> Result<(), CaptureUnavailable> {
        self.check()?;
        self.charged = self
            .charged
            .checked_add(bytes)
            .filter(|sum| *sum <= self.control.limits.max_owned_bytes)
            .ok_or(CaptureUnavailable::Capacity)?;
        Ok(())
    }
    fn raw(&mut self, bytes: &[u8]) -> Result<(), CaptureUnavailable> {
        self.write_all(bytes).map_err(|_| {
            self.error
                .take()
                .unwrap_or(CaptureUnavailable::Serialization)
        })
    }
    fn value<T: Serialize>(&mut self, value: &T) -> Result<(), CaptureUnavailable> {
        serde_json::to_writer(&mut *self, value).map_err(|_| {
            self.error
                .take()
                .unwrap_or(CaptureUnavailable::Serialization)
        })
    }
}
impl Write for BudgetWriter<'_> {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if let Err(error) = self.check() {
            self.error = Some(error);
            return Err(io::Error::other("CPU7 capture unavailable"));
        }
        let Some(needed) = self.bytes.len().checked_add(bytes.len()) else {
            self.error = Some(CaptureUnavailable::Capacity);
            return Err(io::Error::other("CPU7 capture overflow"));
        };
        if needed > self.bytes.capacity() {
            let old = self.bytes.capacity();
            let remaining = self
                .control
                .limits
                .max_owned_bytes
                .saturating_sub(self.charged);
            let target = old
                .saturating_mul(2)
                .max(1024)
                .min(old.saturating_add(remaining));
            if target < needed {
                self.error = Some(CaptureUnavailable::Capacity);
                return Err(io::Error::other("CPU7 capture budget"));
            }
            if let Err(error) = self.charge(target - old) {
                self.error = Some(error);
                return Err(io::Error::other("CPU7 capture unavailable"));
            }
            if self
                .bytes
                .try_reserve_exact(target - self.bytes.len())
                .is_err()
            {
                self.error = Some(CaptureUnavailable::Capacity);
                return Err(io::Error::other("CPU7 capture allocation"));
            }
            let extra = self.bytes.capacity().saturating_sub(target);
            if let Err(error) = self.charge(extra) {
                self.error = Some(error);
                return Err(io::Error::other("CPU7 capture budget"));
            }
        }
        self.bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[derive(Serialize)]
struct BorrowedCpuObservation<'a> {
    schema: &'static str,
    kind: &'static str,
    scope: &'static str,
    score_units: &'static str,
    id: ObservationId,
    state: StateId,
    line: Option<LineId>,
    cpu_pv: Option<LineId>,
    source: u64,
    epoch: u64,
    value_identity: &'a crate::cpu_value::CpuValueIdentity,
    checker_identity: Option<&'a CheckerIdentity>,
    cpu_condition: &'a str,
    depth: u16,
    profile: u64,
    condition: u64,
    score: i32,
    perspective: u8,
    bound: crate::pals::store::BoundKind,
    work: Option<(Option<u64>, Option<u64>, Option<u64>)>,
    budget: u64,
    supersedes: Option<ObservationId>,
    execution: Option<crate::pals::store::ExecutionId>,
}
fn borrowed_cpu(
    id: ObservationId,
    o: &Observation,
) -> Result<BorrowedCpuObservation<'_>, CaptureUnavailable> {
    let (depth, profile, condition) = match o.scope {
        EvidenceScope::DepthLimited {
            depth,
            profile,
            condition,
        } if o.kind == ObservationKind::CpuAnalysis => (depth, profile, condition),
        _ => return Err(CaptureUnavailable::UnsupportedObservation),
    };
    let (score, perspective, bound) = match o.score {
        RawScore::Cpu {
            value,
            perspective,
            bound,
        } => (value, u8::from(perspective == Color::Black), bound),
        _ => return Err(CaptureUnavailable::UnsupportedObservation),
    };
    if o.external_report.is_some()
        || o.model_value_identity.is_some()
        || o.model_value_input.is_some()
        || matches!(
            o.checker_identity.as_ref(),
            Some(CheckerIdentity::ExternalUci(_))
        )
    {
        return Err(CaptureUnavailable::UnsupportedObservation);
    }
    Ok(BorrowedCpuObservation {
        schema: "rz-pals-cpu7-owned-observation/1",
        kind: "cpu_analysis",
        scope: "depth_limited",
        score_units: "owned_cpu",
        id,
        state: o.state,
        line: o.line,
        cpu_pv: o.cpu_pv,
        source: o.source,
        epoch: o.epoch,
        value_identity: o
            .value_identity
            .as_ref()
            .ok_or(CaptureUnavailable::UnsupportedObservation)?,
        checker_identity: o.checker_identity.as_ref(),
        cpu_condition: o
            .cpu_condition
            .as_deref()
            .ok_or(CaptureUnavailable::UnsupportedObservation)?,
        depth,
        profile,
        condition,
        score,
        perspective,
        bound,
        work: o.checker_work.map(|w| (w.nodes, w.qnodes, w.tt_hits)),
        budget: o.budget,
        supersedes: o.supersedes,
        execution: o.execution,
    })
}

fn push<T>(
    items: &mut Vec<T>,
    item: T,
    budget: &mut BudgetWriter<'_>,
) -> Result<(), CaptureUnavailable> {
    budget.check()?;
    if items.len() == items.capacity() {
        let unit = std::mem::size_of::<T>();
        if unit == 0 {
            return Err(CaptureUnavailable::InvalidLimits);
        }
        let old = items.capacity();
        let allowed = (budget
            .control
            .limits
            .max_owned_bytes
            .saturating_sub(budget.charged))
            / unit;
        let target = old
            .saturating_mul(2)
            .max(4)
            .min(old.saturating_add(allowed));
        if target <= old {
            return Err(CaptureUnavailable::Capacity);
        }
        budget.charge(
            (target - old)
                .checked_mul(unit)
                .ok_or(CaptureUnavailable::Capacity)?,
        )?;
        items
            .try_reserve_exact(target - items.len())
            .map_err(|_| CaptureUnavailable::Capacity)?;
        let extra = items
            .capacity()
            .saturating_sub(target)
            .checked_mul(unit)
            .ok_or(CaptureUnavailable::Capacity)?;
        budget.charge(extra)?;
    }
    items.push(item);
    Ok(())
}

fn charge_history(
    snapshot: &PositionSnapshot,
    total: &mut usize,
    maximum: usize,
) -> Result<(), CaptureUnavailable> {
    let bytes = PositionSnapshot::retained_history_bytes_for_len(snapshot.known_history_len())
        .ok_or(CaptureUnavailable::Capacity)?;
    *total = total
        .checked_add(bytes)
        .filter(|sum| *sum <= maximum)
        .ok_or(CaptureUnavailable::Capacity)?;
    Ok(())
}

pub(super) fn capture_selected_cpu_raw(
    view: &ReviewView,
    stores: &PalsStores,
    control: Option<&CaptureControl<'_>>,
) -> Result<SelectedCpuRawCapture, CaptureUnavailable> {
    let control = control.ok_or(CaptureUnavailable::MissingOriginalControl)?;
    let limits = control.limits;
    if !(7..=64).contains(&limits.max_observations)
        || limits.max_line_chunks > 4096
        || limits.max_states == 0
        || limits.max_states > 4096
        || limits.max_owned_bytes == 0
        || limits.max_owned_bytes > 16 * 1024 * 1024
        || limits.max_retained_history_bytes == 0
    {
        return Err(CaptureUnavailable::InvalidLimits);
    }
    let source = view
        .cohort_cpu7
        .as_ref()
        .ok_or(CaptureUnavailable::MissingSource)?;
    let binding = view
        .cohort_cpu7_binding
        .ok_or(CaptureUnavailable::MissingSource)?;
    let mut out = BudgetWriter {
        bytes: Vec::new(),
        charged: 0,
        control,
        error: None,
    };
    out.check()?;
    let mut history_charge = 0;
    charge_history(
        source.root_snapshot(),
        &mut history_charge,
        limits.max_retained_history_bytes,
    )?;
    charge_history(
        source.anchor(),
        &mut history_charge,
        limits.max_retained_history_bytes,
    )?;
    if source.generation != stores.generation() || source.root != view.root_state {
        return Err(CaptureUnavailable::MissingSource);
    }
    let selected = source
        .records
        .get(..source.selected_len)
        .ok_or(CaptureUnavailable::MissingSource)?;
    let mut pending = Vec::new();
    for record in selected {
        out.check()?;
        if record.evidence.is_some() {
            return Err(CaptureUnavailable::UnsupportedSelectedEvidence);
        }
        if record.record.kind == RecordKind::CpuVerification {
            let id = record
                .record
                .cpu_observation
                .ok_or(CaptureUnavailable::MissingSource)?;
            if pending.contains(&id) {
                return Err(CaptureUnavailable::DuplicateObservation);
            }
            if pending.len() >= limits.max_observations {
                return Err(CaptureUnavailable::Capacity);
            }
            push(&mut pending, id, &mut out)?;
        }
    }
    if pending.len() != 7 {
        return Err(CaptureUnavailable::MissingSource);
    }
    let mut seen = Vec::new();
    let mut line_pending = Vec::new();
    let mut lines = Vec::new();
    let mut states = Vec::new();
    out.raw(b"[")?;
    while let Some(id) = pending.pop() {
        out.check()?;
        if seen.contains(&id) {
            continue;
        }
        if seen.len() >= limits.max_observations {
            return Err(CaptureUnavailable::Capacity);
        }
        let o = stores
            .observations
            .get(id)
            .map_err(CaptureUnavailable::Store)?;
        let raw = borrowed_cpu(id, o)?;
        if !seen.is_empty() {
            out.raw(b",")?;
        }
        out.value(&raw)?;
        push(&mut seen, id, &mut out)?;
        if !states.iter().any(|(s, _)| *s == o.state) {
            if states.len() >= limits.max_states {
                return Err(CaptureUnavailable::Capacity);
            }
            let snapshot = stores
                .states
                .get(o.state)
                .map_err(CaptureUnavailable::Store)?;
            charge_history(
                snapshot,
                &mut history_charge,
                limits.max_retained_history_bytes,
            )?;
            push(&mut states, (o.state, snapshot.clone()), &mut out)?;
        }
        for line in [o.line, o.cpu_pv].into_iter().flatten() {
            if !line_pending.contains(&line) && !lines.iter().any(|(id, _)| *id == line) {
                if line_pending.len() + lines.len() >= limits.max_line_chunks {
                    return Err(CaptureUnavailable::Capacity);
                }
                push(&mut line_pending, line, &mut out)?;
            }
        }
        if let Some(previous) = o.supersedes {
            if !pending.contains(&previous) && !seen.contains(&previous) {
                if pending.len() + seen.len() >= limits.max_observations {
                    return Err(CaptureUnavailable::Capacity);
                }
                push(&mut pending, previous, &mut out)?;
            }
        }
    }
    out.raw(b"]")?;
    while let Some(id) = line_pending.pop() {
        out.check()?;
        if lines.iter().any(|(old, _)| *old == id) {
            continue;
        }
        let chunk = stores.lines.get(id).map_err(CaptureUnavailable::Store)?;
        if !states.iter().any(|(state, _)| *state == chunk.start_state) {
            if states.len() >= limits.max_states {
                return Err(CaptureUnavailable::Capacity);
            }
            let snapshot = stores
                .states
                .get(chunk.start_state)
                .map_err(CaptureUnavailable::Store)?;
            charge_history(
                snapshot,
                &mut history_charge,
                limits.max_retained_history_bytes,
            )?;
            push(&mut states, (chunk.start_state, snapshot.clone()), &mut out)?;
        }
        if let Some(parent) = chunk.parent {
            if !line_pending.contains(&parent) && !lines.iter().any(|(old, _)| *old == parent) {
                if line_pending.len() + lines.len() + 1 >= limits.max_line_chunks {
                    return Err(CaptureUnavailable::Capacity);
                }
                push(&mut line_pending, parent, &mut out)?;
            }
        }
        if lines.len() >= limits.max_line_chunks {
            return Err(CaptureUnavailable::Capacity);
        }
        push(&mut lines, (id, chunk.clone()), &mut out)?;
    }
    out.check()?;
    Ok(SelectedCpuRawCapture {
        source: Arc::clone(source),
        binding,
        observations_json: out.bytes,
        states,
        lines,
        requested_owned_bytes: out.charged,
        retained_history_bytes: history_charge,
    })
}
