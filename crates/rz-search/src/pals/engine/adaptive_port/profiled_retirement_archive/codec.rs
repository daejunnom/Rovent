//! Streaming historical projection of the original retired external owner.
//!
//! The stream contains no constructors for worker, Store, view, publication or
//! acknowledgement capabilities. Its byte digest and durable readback belong
//! to the IO owner. Source execution failures are data; an incomplete Store
//! closure, a write failure, cancellation or expiry cannot emit a footer.

use super::super::*;
use crate::cpu_checker::{
    CheckerAttempt, CheckerError, CheckerShutdown, ExternalBound, ExternalCheckerReport,
    ExternalProcessIdentity, ExternalRawScore,
};
use crate::external_cpu::{
    ExternalControlAction, ExternalCpuConfig, ExternalGoIssue, ExternalPolicyOpinion,
    ProfiledExternalCompletion, ProfiledExternalDispatch, ProfiledExternalJob,
    ProfiledExternalOutcome, ProfiledExternalRootSource, ProfiledExternalScope,
};
use crate::pals::store::StorePins;
use rz_contracts::rovent::{QuestionEnvelope, ReciprocalQuestionId};
use rz_position::{MAX_UCI_REPLAY_PLIES, PositionSnapshot};
use serde::Serialize;
use std::fmt::{Debug, Display};
use std::io::{self, Write};

pub(super) const STREAM_SCHEMA: &str = "rz-rovent-profiled-external-retirement-archive/1";
pub(super) const STREAM_DOMAIN: &str = "rz-rovent-profiled-external-retirement-stream/1";
const MAX_PACKETS: usize = 64;
const MAX_VIEW_RECORDS: usize = 64;
const MAX_VIEW_ANCHORS: usize = 4;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct CodecSummary {
    pub packet_count: u64,
    pub store_records: u64,
    pub stream_records: u64,
    pub source_error_count: u64,
}

/// The caller supplies the retained original owner and a finite Store record
/// bound. Packet ordering is a bounded pointer index, never a cloned graph.
pub(super) fn encode_retirement(
    writer: &mut dyn Write,
    stores: &PalsStores,
    retirement: &ProfiledExternalRootRetirement,
    registration: &super::ProfiledRetirementSinkRegistration,
    root_cleanup_deadline: Instant,
    until: Instant,
    cancel: &AtomicBool,
) -> Result<CodecSummary, PalsError> {
    check_control(until, cancel)?;
    let registration_sha256 = registration
        .canonical_sha256()
        .map_err(|_| PalsError::InvalidConfig)?;
    let max_records = registration.max_store_records as usize;
    if retirement.original_cleanup_deadline() != Some(root_cleanup_deadline) {
        return Err(StoreError::InvalidEvidence(
            "profiled retirement cleanup deadline differs from original owner",
        )
        .into());
    }
    if until > root_cleanup_deadline {
        return Err(StoreError::ArchiveDeadline.into());
    }
    if max_records == 0 || retirement.retained_questions() > MAX_PACKETS {
        return Err(StoreError::Capacity("profiled retirement record bound").into());
    }
    retirement.validate_owner(stores)?;
    let original_pins = retirement.protected_pins()?;
    let mut closure_control = || store_control(until, cancel);
    let closure = stores.retirement_dependency_closure_controlled(
        original_pins,
        max_records,
        &mut closure_control,
    )?;
    let mut store_count = pin_count(&closure)?;
    for (observation, situations) in stores.retirement_dependency_edges() {
        check_control(until, cancel)?;
        for situation in situations {
            check_control(until, cancel)?;
            if closure.observations.contains(&observation) && closure.situations.contains(situation)
            {
                store_count = store_count.checked_add(1).ok_or(PalsError::Capacity)?;
                if store_count > max_records {
                    return Err(
                        StoreError::Capacity("profiled retirement dependency records").into(),
                    );
                }
            }
        }
    }
    if store_count > max_records {
        return Err(StoreError::Capacity("profiled retirement Store records").into());
    }

    let mut packets: [Option<ProfiledExternalRetiredPacket<'_>>; MAX_PACKETS] =
        std::array::from_fn(|_| None);
    let mut packet_count = 0usize;
    for packet in retirement.packets() {
        check_control(until, cancel)?;
        let slot = packets.get_mut(packet_count).ok_or(PalsError::Capacity)?;
        *slot = Some(packet);
        packet_count = packet_count.checked_add(1).ok_or(PalsError::Capacity)?;
    }
    if packet_count != retirement.retained_questions() {
        return Err(StoreError::ArchiveIntegrity("profiled retirement packet count").into());
    }
    packets[..packet_count].sort_unstable_by_key(|packet| {
        let packet = packet.as_ref().expect("filled bounded packet index");
        (
            packet.work().input.question_digest.0,
            packet.job().question.id.sequence,
        )
    });
    let source = retirement.source();
    let clock = source.clock();
    let mut encoder = Encoder {
        writer: CheckedWriter {
            inner: writer,
            until,
            cancel,
        },
        stores,
        origin: clock.origin,
        summary: CodecSummary::default(),
    };
    encoder.record("header", |object| {
        object.field("domain", STREAM_DOMAIN)?;
        object.field("sink_registration", registration)?;
        object.field("sink_registration_sha256", registration_sha256)?;
        object.field("sink_window_grants_search_authority", false)?;
        object.field("root_cleanup_window_is_search_window", false)?;
        object.nested(
            "original_root_cleanup_deadline_relative_to_original_clock",
            |e| e.instant(root_cleanup_deadline),
        )?;
        object.nested("archive_until_relative_to_original_clock", |e| {
            e.instant(until)
        })?;
        object.field("historical_evidence_only", true)?;
        object.field("execution_authorized", false)?;
        object.field("capabilities_reissued", false)?;
        object.field("rules_version", rz_position::RULES_VERSION)?;
        object.field("rules_variant", rz_position::RULES_VARIANT)?;
        object.field("packet_count", packet_count)?;
        object.field("packet_order", "question_digest_then_reciprocal_sequence")?;
        object.field(
            "source_error_count_unit",
            "packets_with_completion_or_publication_error",
        )?;
        object.field("max_store_records", max_records)?;
        object.field("declared_store_records", store_count)?;
        object.field("store_generation", stores.generation())?;
        object.nested("authority", |e| authority(e, retirement.authority()))?;
        object.field(
            "clock_encoding",
            "signed_duration_from_original_monotonic_origin",
        )?;
        object.nested("source", |e| external_source(e, source))?;
        object.nested("boundary", |e| boundary(e, retirement))?;
        object.nested("protected_pins", |e| pins(e, original_pins))?;
        object.nested("dependency_closure", |e| pins(e, &closure))
    })?;

    for packet in packets[..packet_count].iter().flatten() {
        encoder.summary.packet_count = encoder
            .summary
            .packet_count
            .checked_add(1)
            .ok_or(PalsError::Capacity)?;
        if packet.publication_error().is_some()
            || packet
                .completion()
                .is_some_and(|value| value.result().is_err())
        {
            encoder.summary.source_error_count = encoder
                .summary
                .source_error_count
                .checked_add(1)
                .ok_or(PalsError::Capacity)?;
        }
        encoder.record("packet", |o| packet_record(o, packet))?;
    }
    for &id in &closure.states {
        encoder.store_record("state", |o| {
            o.field("id", id)?;
            o.nested("value", |e| {
                e.store_wire(|s, w| s.write_retirement_state(id, w))
            })
        })?;
    }
    for &id in &closure.lines {
        encoder.store_record("line", |o| {
            o.field("id", id)?;
            o.nested("value", |e| {
                e.store_wire(|s, w| s.write_retirement_line(id, w))
            })
        })?;
    }
    for &id in &closure.observations {
        encoder.store_record("observation", |o| {
            o.field("id", id)?;
            o.nested("value", |e| {
                e.store_wire(|s, w| s.write_retirement_observation(id, w))
            })
        })?;
    }
    for &id in &closure.situations {
        encoder.store_record("situation", |o| {
            o.field("id", id)?;
            o.field("value", stores.situations.get(id)?)
        })?;
    }
    for &id in &closure.executions {
        encoder.store_record("task", |o| {
            o.field("id", id)?;
            o.field("value", stores.tasks.get(id)?)
        })?;
    }
    for (observation, situations) in stores.retirement_dependency_edges() {
        for &situation in situations {
            encoder.check()?;
            if closure.observations.contains(&observation)
                && closure.situations.contains(&situation)
            {
                encoder.store_record("dependency", |o| {
                    o.field("observation", observation)?;
                    o.field("situation", situation)
                })?;
            }
        }
    }
    if usize::try_from(encoder.summary.store_records).ok() != Some(store_count) {
        return Err(StoreError::ArchiveIntegrity("profiled retirement Store record count").into());
    }
    // This footer certifies only complete encoding of the bounded historical
    // source. It is not a physical fence, durable IO result or caller ack.
    let mut final_summary = encoder.summary;
    final_summary.stream_records = final_summary
        .stream_records
        .checked_add(1)
        .ok_or(PalsError::Capacity)?;
    encoder.record("footer", |o| {
        o.field("packet_count", final_summary.packet_count)?;
        o.field("store_records", final_summary.store_records)?;
        o.field("stream_records", final_summary.stream_records)?;
        o.field("source_error_count", final_summary.source_error_count)?;
        o.field("historical_encoding_complete", true)?;
        o.field("execution_authorized", false)?;
        o.field("capabilities_reissued", false)
    })?;
    encoder.check()?;
    Ok(encoder.summary)
}

fn pin_count(pins: &StorePins) -> Result<usize, PalsError> {
    [
        pins.states.len(),
        pins.lines.len(),
        pins.observations.len(),
        pins.situations.len(),
        pins.executions.len(),
    ]
    .into_iter()
    .try_fold(0usize, |total, count| total.checked_add(count))
    .ok_or(PalsError::Capacity)
}

fn check_control(until: Instant, cancel: &AtomicBool) -> Result<(), PalsError> {
    store_control(until, cancel).map_err(Into::into)
}

fn store_control(until: Instant, cancel: &AtomicBool) -> Result<(), StoreError> {
    if cancel.load(Ordering::Acquire) {
        Err(StoreError::ArchiveCanceled)
    } else if Instant::now() >= until {
        Err(StoreError::ArchiveDeadline)
    } else {
        Ok(())
    }
}

struct CheckedWriter<'a> {
    inner: &'a mut dyn Write,
    until: Instant,
    cancel: &'a AtomicBool,
}
impl Write for CheckedWriter<'_> {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if self.cancel.load(Ordering::Acquire) {
            return Err(io::Error::new(
                io::ErrorKind::ConnectionAborted,
                "retirement canceled",
            ));
        }
        if Instant::now() >= self.until {
            return Err(io::Error::new(
                io::ErrorKind::TimedOut,
                "retirement deadline",
            ));
        }
        self.inner.write(bytes)
    }
    fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }
}

struct Encoder<'a> {
    writer: CheckedWriter<'a>,
    stores: &'a PalsStores,
    origin: Instant,
    summary: CodecSummary,
}
impl Encoder<'_> {
    fn check(&self) -> Result<(), PalsError> {
        check_control(self.writer.until, self.writer.cancel)
    }
    fn io_error(&self, error: io::Error) -> PalsError {
        self.check().err().unwrap_or_else(|| {
            StoreError::ArchiveIo {
                stage: "profiled retirement stream write",
                kind: error.kind(),
            }
            .into()
        })
    }
    fn raw(&mut self, bytes: &[u8]) -> Result<(), PalsError> {
        self.check()?;
        self.writer.write_all(bytes).map_err(|e| self.io_error(e))
    }
    fn json<T: Serialize>(&mut self, value: T) -> Result<(), PalsError> {
        self.check()?;
        serde_json::to_writer(&mut self.writer, &value).map_err(|error| {
            self.check().err().unwrap_or_else(|| {
                StoreError::ArchiveIo {
                    stage: "profiled retirement JSON encode",
                    kind: error.io_error_kind().unwrap_or(io::ErrorKind::InvalidData),
                }
                .into()
            })
        })
    }
    fn object(
        &mut self,
        body: impl FnOnce(&mut Object<'_, '_>) -> Result<(), PalsError>,
    ) -> Result<(), PalsError> {
        self.raw(b"{")?;
        body(&mut Object {
            encoder: self,
            first: true,
        })?;
        self.raw(b"}")
    }
    fn array<I>(
        &mut self,
        values: I,
        mut item: impl FnMut(&mut Self, I::Item) -> Result<(), PalsError>,
    ) -> Result<(), PalsError>
    where
        I: IntoIterator,
    {
        self.raw(b"[")?;
        let mut first = true;
        for value in values {
            if !first {
                self.raw(b",")?;
            }
            first = false;
            item(self, value)?;
        }
        self.raw(b"]")
    }
    fn option<T>(
        &mut self,
        value: Option<T>,
        encode: impl FnOnce(&mut Self, T) -> Result<(), PalsError>,
    ) -> Result<(), PalsError> {
        match value {
            Some(value) => encode(self, value),
            None => self.raw(b"null"),
        }
    }
    fn record(
        &mut self,
        kind: &'static str,
        body: impl FnOnce(&mut Object<'_, '_>) -> Result<(), PalsError>,
    ) -> Result<(), PalsError> {
        let next = self
            .summary
            .stream_records
            .checked_add(1)
            .ok_or(PalsError::Capacity)?;
        self.object(|o| {
            o.field("schema", STREAM_SCHEMA)?;
            o.field("kind", kind)?;
            body(o)
        })?;
        self.raw(b"\n")?;
        self.summary.stream_records = next;
        Ok(())
    }
    fn store_record(
        &mut self,
        kind: &'static str,
        body: impl FnOnce(&mut Object<'_, '_>) -> Result<(), PalsError>,
    ) -> Result<(), PalsError> {
        self.record(kind, body)?;
        self.summary.store_records = self
            .summary
            .store_records
            .checked_add(1)
            .ok_or(PalsError::Capacity)?;
        Ok(())
    }
    fn store_wire(
        &mut self,
        write: impl FnOnce(&PalsStores, &mut dyn Write) -> Result<(), StoreError>,
    ) -> Result<(), PalsError> {
        self.check()?;
        let result = write(self.stores, &mut self.writer);
        match result {
            Ok(()) => self.check(),
            Err(error) => Err(self.check().err().unwrap_or_else(|| error.into())),
        }
    }
    fn instant(&mut self, value: Instant) -> Result<(), PalsError> {
        let (negative, duration) = match value.checked_duration_since(self.origin) {
            Some(duration) => (false, duration),
            None => (true, self.origin.duration_since(value)),
        };
        self.object(|o| {
            o.field("negative", negative)?;
            o.field("seconds", duration.as_secs())?;
            o.field("nanoseconds", duration.subsec_nanos())
        })
    }
    fn moves(&mut self, moves: &[BoardMove]) -> Result<(), PalsError> {
        if moves.len() > MAX_UCI_REPLAY_PLIES {
            return Err(StoreError::ArchiveIntegrity("retirement move sequence bound").into());
        }
        self.array(moves, |e, movement| e.json(Move16::pack(*movement)?))
    }
    fn snapshot(&mut self, id: StateId, value: &PositionSnapshot) -> Result<(), PalsError> {
        self.store_wire(|s, w| s.write_retirement_snapshot(id, value, w))
    }
}

struct Object<'a, 'b> {
    encoder: &'a mut Encoder<'b>,
    first: bool,
}
impl Object<'_, '_> {
    fn key(&mut self, name: &str) -> Result<(), PalsError> {
        if !self.first {
            self.encoder.raw(b",")?;
        }
        self.first = false;
        self.encoder.json(name)?;
        self.encoder.raw(b":")
    }
    fn field<T: Serialize>(&mut self, name: &str, value: T) -> Result<(), PalsError> {
        self.key(name)?;
        self.encoder.json(value)
    }
    fn nested(
        &mut self,
        name: &str,
        body: impl FnOnce(&mut Encoder<'_>) -> Result<(), PalsError>,
    ) -> Result<(), PalsError> {
        self.key(name)?;
        body(self.encoder)
    }
}

struct DebugText<'a, T: ?Sized>(&'a T);
impl<T: Debug + ?Sized> Display for DebugText<'_, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(self.0, f)
    }
}
impl<T: Debug + ?Sized> Serialize for DebugText<'_, T> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_str(self)
    }
}
struct DisplayText<'a, T: ?Sized>(&'a T);
impl<T: Display + ?Sized> Serialize for DisplayText<'_, T> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_str(self.0)
    }
}

fn authority(e: &mut Encoder<'_>, value: SearchAuthority) -> Result<(), PalsError> {
    e.object(|o| {
        o.field("process_epoch", value.epoch.0)?;
        o.field("game_generation", value.game.0)?;
        o.field("root_generation", value.root.0)?;
        o.field("implementation", value.implementation.0)
    })
}
fn frame(e: &mut Encoder<'_>, value: FrameId) -> Result<(), PalsError> {
    e.object(|o| {
        o.nested("authority", |e| authority(e, value.authority))?;
        o.field("sequence", value.sequence)
    })
}
fn deadline(e: &mut Encoder<'_>, value: Deadline) -> Result<(), PalsError> {
    e.object(|o| {
        o.field("clock_domain", value.clock.0.0)?;
        o.field("tick", value.at.0)
    })
}
fn state_identity(
    e: &mut Encoder<'_>,
    value: rz_contracts::StateIdentity,
) -> Result<(), PalsError> {
    e.object(|o| {
        o.field("owner", value.owner.0)?;
        o.field("revision", value.revision.0)?;
        o.field("semantic", value.semantic.0)
    })
}
fn view_seal(e: &mut Encoder<'_>, value: ReviewViewSeal) -> Result<(), PalsError> {
    e.object(|o| {
        o.nested("authority", |e| authority(e, value.authority))?;
        o.field("view_sequence", value.view_sequence)?;
        o.field("context_policy", value.context_policy.0)?;
        o.field("view_digest", value.view_digest.0)?;
        o.nested("root_state", |e| state_identity(e, value.root_state))?;
        o.field("actual_tokens", value.actual_tokens)?;
        o.field("selected_records", value.selected_records)?;
        o.field("anchors", value.anchors)?;
        o.field("length_bucket", value.length_bucket)
    })
}
fn reciprocal(e: &mut Encoder<'_>, value: ReciprocalQuestionId) -> Result<(), PalsError> {
    e.object(|o| {
        o.nested("authority", |e| authority(e, value.authority))?;
        o.field("sequence", value.sequence)
    })
}
fn envelope(e: &mut Encoder<'_>, value: QuestionEnvelope) -> Result<(), PalsError> {
    e.object(|o| {
        o.nested("id", |e| reciprocal(e, value.id))?;
        o.nested("parent", |e| e.option(value.parent, reciprocal))?;
        o.field("sender", DebugText(&value.sender))?;
        o.field("recipient", DebugText(&value.recipient))?;
        o.field("purpose", DebugText(&value.purpose))?;
        o.nested("anchor", |e| state_identity(e, value.anchor))?;
        o.field("anchor_root_ply", value.anchor_root_ply.0)?;
        o.field(
            "original_first_move",
            value.original_first_move.map(|m| m.bits()),
        )?;
        o.field("prefix_digest", value.prefix_digest.0)?;
        o.field("line_digest", value.line_digest.map(|d| d.0))?;
        o.field("line_offset", value.line_offset.0)?;
        o.field("question_digest", value.question_digest.0)?;
        o.nested("view", |e| view_seal(e, value.view))?;
        o.field("evidence_revision", value.evidence_revision)?;
        o.field("root_budget", value.root_budget.0)?;
        o.nested("original_deadline", |e| {
            deadline(e, value.original_deadline)
        })
    })
}
fn scope(e: &mut Encoder<'_>, value: ProfiledExternalScope) -> Result<(), PalsError> {
    e.object(|o| {
        o.nested("question", |e| reciprocal(e, value.question))?;
        o.field("execution_epoch", value.execution.epoch.0)?;
        o.field("execution_sequence", value.execution.sequence)?;
        o.field("owner", value.owner.0)
    })
}
fn process(e: &mut Encoder<'_>, value: ExternalProcessIdentity) -> Result<(), PalsError> {
    e.object(|o| {
        o.field("pid", value.pid)?;
        o.field("process_group", value.process_group)?;
        o.field("proc_start_ticks", value.proc_start_ticks)
    })
}
fn external_source(
    e: &mut Encoder<'_>,
    value: &ProfiledExternalRootSource,
) -> Result<(), PalsError> {
    e.object(|o| {
        o.nested("authority", |e| authority(e, value.authority()))?;
        o.field("source_sequence", value.source_sequence())?;
        o.field("worker_owner", value.owner().0)?;
        o.field("clock_domain", value.clock().domain.0.0)?;
        o.nested("origin_relative_to_retirement_source", |e| {
            e.instant(value.clock().origin)
        })?;
        o.field("profile", value.profile())?;
        o.field("purpose", value.purpose())?;
        o.nested("process", |e| process(e, value.process_identity()))
    })
}
fn boundary(e: &mut Encoder<'_>, value: &ProfiledExternalRootRetirement) -> Result<(), PalsError> {
    match (value.root_fence(), value.process_close_proof()) {
        (Some(fence), None) => e.object(|o| {
            o.field("kind", "CurrentRootFence")?;
            o.nested("source", |e| external_source(e, fence.source()))?;
            o.nested("authority", |e| authority(e, fence.authority()))?;
            o.field("sequence", fence.sequence())?;
            o.field("execution_high_water", fence.execution_high_water())?;
            o.field(
                "native_request_high_water",
                fence.native_request_high_water(),
            )?;
            o.field("accepted_jobs", fence.accepted_jobs())?;
            o.field("terminal_controls", fence.terminal_controls())?;
            o.field("drained_terminals", fence.drained_terminals())?;
            o.nested("process", |e| shutdown(e, fence.shutdown_receipt()))
        }),
        (None, Some(proof)) => e.object(|o| {
            o.field("kind", "ProcessCloseProof")?;
            o.nested("source", |e| external_source(e, proof.source()))?;
            o.nested("authority", |e| authority(e, proof.authority()))?;
            o.field("accepted_jobs", proof.accepted_jobs())?;
            o.field("terminal_controls", proof.terminal_controls())?;
            o.field("drained_terminals", proof.drained_terminals())?;
            o.nested("process", |e| shutdown(e, proof.shutdown_receipt()))
        }),
        _ => Err(StoreError::ArchiveIntegrity("profiled retirement original boundary").into()),
    }
}
fn duration(e: &mut Encoder<'_>, value: Duration) -> Result<(), PalsError> {
    e.object(|o| {
        o.field("seconds", value.as_secs())?;
        o.field("nanoseconds", value.subsec_nanos())
    })
}
fn pins(e: &mut Encoder<'_>, value: &StorePins) -> Result<(), PalsError> {
    e.object(|o| {
        o.field("states", &value.states)?;
        o.field("lines", &value.lines)?;
        o.field("observations", &value.observations)?;
        o.field("situations", &value.situations)?;
        o.field("executions", &value.executions)
    })
}

fn packet_record(
    o: &mut Object<'_, '_>,
    packet: &ProfiledExternalRetiredPacket<'_>,
) -> Result<(), PalsError> {
    o.nested("work", |e| work(e, packet.work()))?;
    o.nested("job", |e| job(e, packet.job(), packet.task_key().state))?;
    o.nested("config", |e| config(e, packet.config()))?;
    o.field("task_key", packet.task_key())?;
    o.field("store_execution", packet.execution())?;
    o.field("consumer", packet.consumer())?;
    o.nested("dispatch", |e| e.option(packet.dispatch(), dispatch))?;
    o.nested("completion", |e| {
        e.option(packet.completion().map(Arc::as_ref), completion)
    })?;
    o.nested("prepared_observation", |e| {
        e.option(packet.prepared_observation(), |e, value| {
            e.store_wire(|s, w| s.write_retirement_observation_value(value, w))
        })
    })?;
    o.field("published_observation", packet.observation())?;
    o.field("consumed_observation", packet.actual_consumed_observation())?;
    o.field("search_consumed", packet.search_consumed())?;
    o.field("feedback_published", packet.feedback_published())?;
    o.field("terminal", packet.terminal())?;
    o.field("accounted", packet.accounted())?;
    o.nested("publication_error", |e| {
        e.option(packet.publication_error(), pals_error)
    })
}

fn work(
    e: &mut Encoder<'_>,
    value: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
) -> Result<(), PalsError> {
    e.object(|o| {
        o.nested("frame", |e| frame(e, value.frame))?;
        o.field("original_first_move", value.original_first_move.bits())?;
        o.nested("original_deadline", |e| {
            deadline(e, value.original_deadline)
        })?;
        o.nested("budget", |e| {
            e.object(|o| {
                o.field(
                    "role_submissions_remaining",
                    value.budget.role_submissions_remaining,
                )?;
                o.field("original_cpu_credit", value.budget.original_cpu_credit)?;
                o.field(
                    "original_store_byte_limit",
                    value.budget.original_store_byte_limit,
                )
            })
        })?;
        o.field("stage", DebugText(&value.input.stage))?;
        o.field("question_digest", value.input.question_digest.0)?;
        o.field("public_context", value.input.public_context.0)?;
        o.nested("view_seal", |e| view_seal(e, value.input.view_seal))?;
        o.nested("question", |e| question(e, &value.input.question))?;
        o.nested("view", |e| view(e, &value.input.view))
    })
}

fn context(e: &mut Encoder<'_>, value: &RoleLogicalContext) -> Result<(), PalsError> {
    e.object(|o| {
        o.field("game_generation", value.game_generation)?;
        o.field("search_generation", value.search_generation)?;
        o.field("situation", value.situation)?;
        o.field("state", value.state)?;
        o.field("focus", value.focus)?;
        o.field("purpose", DebugText(&value.purpose))?;
        o.nested("prefix", |e| e.moves(&value.prefix))?;
        o.field("focus_sha256", value.focus_sha256)?;
        o.field("prefix_sha256", value.prefix_sha256)?;
        o.field("proposal_sha256", value.proposal_sha256)?;
        o.field("refutation_sha256", value.refutation_sha256)?;
        o.field("divergence_sha256", value.divergence_sha256)?;
        o.field("public_revision", value.public_revision)?;
        o.field("situation_revision", value.situation_revision)
    })
}

fn question(e: &mut Encoder<'_>, value: &AdaptivePortQuestion) -> Result<(), PalsError> {
    e.object(|o| {
        o.nested("position", |e| {
            e.snapshot(value.context.state, &value.position.snapshot())
        })?;
        o.nested("context", |e| context(e, &value.context))?;
        o.nested("deadline", |e| e.instant(value.deadline))?;
        o.field("node", value.node)?;
        o.field("stage", DebugText(&value.stage))?;
        o.nested("first", |e| {
            e.option(value.first, |e, m| e.json(Move16::pack(m)?))
        })?;
        o.nested("proposal", |e| e.moves(&value.proposal))?;
        o.nested("counter", |e| {
            e.option(value.counter.as_deref(), |e, v| e.moves(v))
        })?;
        o.nested("repaired", |e| {
            e.option(value.repaired.as_deref(), |e, v| e.moves(v))
        })?;
        o.field("attack_ply", value.attack_ply)?;
        o.field("discovery_restricted", value.discovery_restricted)?;
        o.field("counter_source", value.counter_source)?;
        o.nested("validator_root_moves", |e| {
            e.option(value.validator_root_moves.as_deref(), |e, v| e.moves(v))
        })?;
        o.field("recheck", value.recheck)?;
        o.nested("frozen_counter", |e| {
            e.option(value.frozen_counter.as_deref(), view)
        })?;
        o.field("paired_repaired", value.paired_repaired)?;
        o.nested("paired_fresh_seal", |e| {
            e.option(value.paired_fresh_seal, fresh_pair)
        })?;
        o.field("question_digest", value.question_digest.0)?;
        o.nested("legacy_full_records", |e| {
            e.option(value.legacy_full_records.as_deref(), legacy_records)
        })?;
        o.nested("frontier_source", |e| frontier_source(e, value))?;
        o.nested("goal_local", |e| {
            e.option(value.goal_local.as_deref(), |e, v| {
                e.object(|o| {
                    o.field(
                        "kind",
                        match v.source_key() {
                            GoalLocalQuestionKey::Mate(_) => "Mate",
                            GoalLocalQuestionKey::Horizon(_) => "Horizon",
                        },
                    )?;
                    o.field("opaque_original_key", DebugText(&v.source_key()))?;
                    o.field("digest", v.digest().0)?;
                    o.nested("prefix", |e| e.moves(v.prefix()))?;
                    o.nested("snapshot", |e| {
                        e.snapshot(value.context.state, v.snapshot())
                    })?;
                    o.nested("first", |e| e.json(Move16::pack(v.first_move())?))?;
                    o.nested("original_deadline", |e| deadline(e, v.original_deadline()))?;
                    o.field("purpose", DebugText(&v.purpose()))?;
                    o.field("stage", DebugText(&v.stage()))
                })
            })
        })?;
        o.nested("diagnostic_repair_line", |e| {
            e.option(value.diagnostic_repair_line, repair_line)
        })
    })
}

fn frontier_source(e: &mut Encoder<'_>, value: &AdaptivePortQuestion) -> Result<(), PalsError> {
    let source = value.retirement_frontier_source();
    e.object(|o| {
        o.nested("authority", |e| authority(e, source.authority))?;
        o.nested("original_deadline", |e| {
            deadline(e, source.original_deadline)
        })?;
        o.nested("deadline", |e| e.instant(source.deadline))?;
        o.nested("position", |e| {
            e.snapshot(value.context.state, source.position)
        })?;
        o.field("context", source.context.0)?;
        o.nested("view", |e| view_seal(e, source.view))?;
        o.nested("legacy_digest_or_expired", |e| {
            e.option(source.legacy_digest_or_expired, |e, value| {
                e.object(|o| match value {
                    Ok(digest) => {
                        o.field("kind", "Retained")?;
                        o.field("digest", digest.0)
                    }
                    Err(()) => o.field("kind", "Expired"),
                })
            })
        })?;
        o.field("goal_origin", source.goal_origin.map(|v| v.0))
    })
}

fn legacy_records(e: &mut Encoder<'_>, value: &LegacyFullRoleRecords) -> Result<(), PalsError> {
    if value.records().len() > 16_384 {
        return Err(StoreError::ArchiveIntegrity("retirement legacy row bound").into());
    }
    e.object(|o| {
        o.nested("authority", |e| authority(e, value.authority()))?;
        o.field("revision", value.revision())?;
        o.field("generation", value.generation())?;
        o.field("root_state", value.root_state())?;
        o.field("digest", value.digest())?;
        o.nested("records", |e| {
            e.array(value.records(), |e, record| {
                e.store_wire(|s, w| s.write_retirement_role_record(record, w))
            })
        })
    })
}

fn view(e: &mut Encoder<'_>, value: &ReviewView) -> Result<(), PalsError> {
    if value.records().len() > MAX_VIEW_RECORDS
        || value.records_metadata().len() != value.records().len()
        || value.source_snapshots().len() != value.records().len()
        || value.anchors().len() > MAX_VIEW_ANCHORS
        || value.selected_bundle_ids().len() > MAX_VIEW_RECORDS
    {
        return Err(
            StoreError::ArchiveIntegrity("retirement original selected view bounds").into(),
        );
    }
    let policy = value.selection_policy();
    e.object(|o| {
        o.field("sequence", value.sequence())?;
        o.field("root_state", value.root_state())?;
        o.field("generation", value.generation())?;
        o.nested("policy", |e| {
            e.object(|o| {
                o.field("mode", DebugText(&policy.mode))?;
                o.field("token_budget", policy.token_budget)?;
                o.field("max_records", policy.max_records)?;
                o.field("max_anchors", policy.max_anchors)?;
                o.field("required_bundle", policy.required_bundle)
            })
        })?;
        o.field("bundles", value.selected_bundle_ids())?;
        o.field("tokens", value.token_count())?;
        o.field("bucket", value.length_bucket())?;
        o.field("input_digest", value.digest())?;
        o.field("view_seal_digest", value.seal_digest())?;
        o.field("record_scope_digest", value.record_scope_digest())?;
        o.field("policy_digest", value.policy_digest())?;
        o.field("consumption_observed", value.consumption_observed())?;
        o.nested("anchors", |e| {
            e.array(value.anchors(), |e, anchor| {
                e.snapshot(anchor.state, &anchor.snapshot)
            })
        })?;
        o.nested("source_snapshots", |e| {
            e.array(
                value
                    .source_snapshots()
                    .iter()
                    .zip(value.records_metadata()),
                |e, (snapshot, metadata)| e.snapshot(metadata.source_state, snapshot),
            )
        })?;
        o.nested("records", |e| {
            e.array(value.records(), |e, record| {
                e.store_wire(|s, w| s.write_retirement_role_record(record, w))
            })
        })?;
        o.nested("metadata", |e| {
            e.array(value.records_metadata(), |e, metadata| {
                e.object(|o| {
                    o.field("anchor", metadata.anchor)?;
                    o.field("parent", metadata.parent)?;
                    o.field("supersedes", metadata.supersedes)?;
                    o.field("source_state", metadata.source_state)?;
                    o.field("source_revision", metadata.source_revision)?;
                    o.field("source_line_offset", metadata.source_line_offset)?;
                    o.nested("original_first_move", |e| {
                        e.option(metadata.original_first_move, |e, m| {
                            e.json(Move16::pack(m)?)
                        })
                    })
                })
            })
        })?;
        o.nested("query", |e| {
            e.object(|o| {
                let query = value.query();
                o.field("purpose", DebugText(&query.purpose))?;
                o.nested("prefix", |e| e.moves(&query.prefix))?;
                o.nested("proposal", |e| e.moves(&query.proposal))?;
                o.nested("counterexample", |e| {
                    e.option(query.counterexample.as_deref(), |e, v| e.moves(v))
                })?;
                o.nested("legal", |e| e.moves(&query.legal))?;
                o.field("divergences", &query.divergences)
            })
        })
    })
}

fn fresh_pair(e: &mut Encoder<'_>, value: FreshPairSeal) -> Result<(), PalsError> {
    e.object(|o| {
        o.field("public_context", value.public_context.0)?;
        o.nested("authority", |e| authority(e, value.authority))?;
        o.nested("root_state", |e| state_identity(e, value.root_state))?;
        o.field("context_policy", value.context_policy.0)?;
        o.field("selected_records", value.selected_records)?;
        o.field("anchors", value.anchors)?;
        o.field("model", value.model.0)?;
        o.field("encoding", value.encoding.0)?;
        o.nested("precision", |e| precision(e, value.precision))?;
        o.field("frozen_epoch", value.frozen_epoch)?;
        o.field("record_revision", value.record_revision)?;
        o.field("role", DebugText(&value.role))?;
        o.field("perspective", DebugText(&value.perspective))
    })
}
fn precision(e: &mut Encoder<'_>, value: rz_contracts::PrecisionProfile) -> Result<(), PalsError> {
    e.object(|o| {
        let tag = match value {
            rz_contracts::PrecisionProfile::Fp32 => "Fp32",
            rz_contracts::PrecisionProfile::Fp16 => "Fp16",
            rz_contracts::PrecisionProfile::Bf16 => "Bf16",
            rz_contracts::PrecisionProfile::Quantized(_) => "Quantized",
        };
        o.field("kind", tag)?;
        o.field(
            "quantization",
            match value {
                rz_contracts::PrecisionProfile::Quantized(d) => Some(d.0),
                _ => None,
            },
        )
    })
}
fn repair_line(e: &mut Encoder<'_>, value: AdaptiveRepairLineProvenance) -> Result<(), PalsError> {
    e.object(|o| {
        o.nested("producer_frame", |e| frame(e, value.producer_frame))?;
        o.field("producer_request_epoch", value.producer_request.epoch.0)?;
        o.field("producer_request_sequence", value.producer_request.sequence)?;
        o.field("submitted_question", value.submitted_question.0)?;
        o.field("submitted_view", value.submitted_view.0)?;
        o.field("completed_evidence", value.completed_evidence.0)?;
        o.nested("original_deadline", |e| {
            deadline(e, value.original_deadline)
        })?;
        o.nested("published_line", |e| {
            e.object(|o| {
                let line = value.published_line;
                o.field("line", line.line)?;
                o.field("observation", line.observation)?;
                o.field("record_revision", line.record_revision)?;
                o.field("parent_revision", line.parent_revision)?;
                o.field("supersedes_revision", line.supersedes_revision)?;
                o.field("origin_state", line.origin_state)?;
                o.field("committed_plies", line.committed_plies)?;
                o.nested("first_move", |e| {
                    e.option(line.first_move, |e, m| e.json(Move16::pack(m)?))
                })
            })
        })
    })
}

fn job(e: &mut Encoder<'_>, value: &ProfiledExternalJob, state: StateId) -> Result<(), PalsError> {
    e.object(|o| {
        o.nested("question", |e| envelope(e, value.question))?;
        o.field("mode", DebugText(&value.mode))?;
        o.nested("scope", |e| scope(e, value.scope()))?;
        o.nested("position", |e| {
            e.snapshot(state, &value.position.snapshot())
        })?;
        o.nested("root_moves", |e| {
            e.option(value.root_moves.as_deref(), |e, v| e.moves(v))
        })?;
        o.field("constraint", value.constraint)?;
        o.nested("soft_quantum", |e| duration(e, value.soft_quantum))?;
        o.field("context_token_budget", value.context_token_budget)?;
        o.nested("controls", |e| controls(e, value.control.snapshot()))
    })
}
fn config(e: &mut Encoder<'_>, value: &ExternalCpuConfig) -> Result<(), PalsError> {
    e.object(|o| {
        // Encoded OS bytes are exact on the source platform. They are history,
        // not paths to reopen, commands to execute or launch capabilities.
        o.field("os", std::env::consts::OS)?;
        o.field(
            "program_os_bytes",
            value.program.as_os_str().as_encoded_bytes(),
        )?;
        o.nested("arguments_os_bytes", |e| {
            e.array(&value.arguments, |e, arg| e.json(arg.as_encoded_bytes()))
        })?;
        o.field(
            "working_directory_os_bytes",
            value.working_directory.as_os_str().as_encoded_bytes(),
        )?;
        o.field("identity", &value.identity)?;
        o.field("expected_uci_name", &value.expected_uci_name)?;
        o.field("max_depth", value.max_depth)?;
        o.field("max_prefix_plies", value.max_prefix_plies)?;
        o.nested("handshake_timeout", |e| {
            duration(e, value.handshake_timeout)
        })?;
        o.nested("max_task_wall_time", |e| {
            duration(e, value.max_task_wall_time)
        })?;
        o.nested("stop_grace", |e| duration(e, value.stop_grace))?;
        o.nested("shutdown_grace", |e| duration(e, value.shutdown_grace))?;
        o.field("max_output_bytes", value.max_output_bytes)?;
        o.field("max_line_bytes", value.max_line_bytes)
    })
}
fn dispatch(e: &mut Encoder<'_>, value: &ProfiledExternalDispatch) -> Result<(), PalsError> {
    e.object(|o| {
        o.nested("question", |e| envelope(e, value.question()))?;
        o.nested("scope", |e| scope(e, value.scope()))?;
        o.field("constraint", value.constraint())?;
        o.field("profile", value.profile())?;
        o.field("purpose", value.purpose())?;
        o.field("evaluation_namespace", value.evaluation_namespace().0)?;
        o.nested("process", |e| process(e, value.process_identity()))?;
        o.field("root_restriction", value.root_restriction().0)
    })
}
fn controls(
    e: &mut Encoder<'_>,
    value: rz_runtime::adaptive::ControlSnapshot,
) -> Result<(), PalsError> {
    e.object(|o| {
        o.field("cancel_sequence", value.cancel_sequence)?;
        o.field("yield_sequence", value.yield_sequence)?;
        o.field("publish_sequence", value.publish_sequence)?;
        o.field("cancel_ack", value.cancel_ack)?;
        o.field("yield_ack", value.yield_ack)?;
        o.field("publish_ack", value.publish_ack)?;
        o.field("terminal", value.terminal)
    })
}
fn completion(e: &mut Encoder<'_>, value: &ProfiledExternalCompletion) -> Result<(), PalsError> {
    e.object(|o| {
        o.nested("question", |e| envelope(e, value.question()))?;
        o.nested("scope", |e| scope(e, value.scope()))?;
        o.field("requested", value.requested())?;
        o.field("profile", value.profile())?;
        o.field("purpose", value.purpose())?;
        o.field("evaluation_namespace", value.evaluation_namespace().0)?;
        o.nested("process", |e| e.option(value.process_identity(), process))?;
        o.field(
            "go_issue",
            match value.go_issue() {
                ExternalGoIssue::NotIssued => "NotIssued",
                ExternalGoIssue::Unknown => "Unknown",
                ExternalGoIssue::Issued => "Issued",
            },
        )?;
        o.field("root_restriction", value.root_restriction().0)?;
        o.nested("controls", |e| controls(e, value.controls()))?;
        o.nested("timing", |e| {
            e.object(|o| {
                let time = value.timing();
                for (name, at) in [
                    ("admitted_at", time.admitted_at),
                    ("run_started_at", time.run_started_at),
                    ("go_issued_at", time.go_issued_at),
                    ("quantum_deadline", time.quantum_deadline),
                    ("stop_sent_at", time.stop_sent_at),
                    ("bestmove_drained_at", time.bestmove_drained_at),
                    ("ready_completed_at", time.ready_completed_at),
                    ("returned_at", time.returned_at),
                ] {
                    o.nested(name, |e| e.option(at, |e, at| e.instant(at)))?;
                }
                Ok(())
            })
        })?;
        o.nested("attempt", |e| e.option(value.attempt(), attempt))?;
        o.nested("result", |e| {
            e.object(|o| match value.result() {
                Ok(value) => {
                    o.field("kind", "Ok")?;
                    o.nested("value", |e| outcome(e, value))
                }
                Err(error) => {
                    o.field("kind", "Err")?;
                    o.nested("error", |e| checker_error(e, error))
                }
            })
        })
    })
}
fn outcome(e: &mut Encoder<'_>, value: &ProfiledExternalOutcome) -> Result<(), PalsError> {
    if value.opinions.len() > 4 {
        return Err(StoreError::ArchiveIntegrity("retirement original MultiPV bound").into());
    }
    let report = &value.report;
    e.object(|o| {
        o.nested("report", |e| {
            e.object(|o| {
                o.field("profile", report.profile)?;
                o.field("purpose", report.purpose)?;
                o.field("requested", report.requested)?;
                o.field("issued", report.issued)?;
                o.field("issued_go", &report.issued_go)?;
                o.nested("raw", |e| raw_report(e, &report.raw))?;
                o.field("reported_node_overshoot", report.reported_node_overshoot)?;
                o.field("strict_native_nodes", report.strict_native_nodes)?;
                o.field(
                    "option_application_observed",
                    report.option_application_observed,
                )?;
                o.field(
                    "readiness_barrier_completed",
                    report.readiness_barrier_completed,
                )
            })
        })?;
        o.nested("opinions", |e| e.array(&value.opinions, opinion))?;
        o.field("requested_multipv", value.requested_multipv)?;
        o.field(
            "conditional_policy_target_known",
            value.conditional_policy_target_known,
        )?;
        o.nested("stop_control", |e| {
            e.option(value.stop_control, |e, control| {
                e.object(|o| {
                    o.field("sequence", control.sequence)?;
                    o.field(
                        "action",
                        match control.action {
                            ExternalControlAction::Continue => "Continue",
                            ExternalControlAction::Yield => "Yield",
                            ExternalControlAction::Cancel => "Cancel",
                            ExternalControlAction::Revoked => "Revoked",
                        },
                    )
                })
            })
        })
    })
}
fn raw_report(e: &mut Encoder<'_>, value: &ExternalCheckerReport) -> Result<(), PalsError> {
    if value.pv.len() > MAX_UCI_REPLAY_PLIES
        || value.observed_uci.name.len() > 65_536
        || value
            .observed_uci
            .author
            .as_ref()
            .is_some_and(|author| author.len() > 65_536)
    {
        return Err(StoreError::ArchiveIntegrity("retirement original raw report bounds").into());
    }
    value.identity.validate()?;
    e.store_wire(|s, w| s.write_retirement_external_report(value, w))
}
fn opinion(e: &mut Encoder<'_>, value: &ExternalPolicyOpinion) -> Result<(), PalsError> {
    e.object(|o| {
        o.field("rank", value.rank)?;
        o.field("depth", value.depth)?;
        o.field("seldepth", value.seldepth)?;
        o.nested("pv", |e| e.moves(&value.pv))?;
        o.nested("score", |e| score(e, value.score))?;
        o.field(
            "bound",
            match value.bound {
                ExternalBound::ExactReported => "ExactReported",
                ExternalBound::Lower => "Lower",
                ExternalBound::Upper => "Upper",
                ExternalBound::Unknown => "Unknown",
            },
        )?;
        o.field("reported_wdl", value.reported_wdl)
    })
}
fn score(e: &mut Encoder<'_>, value: ExternalRawScore) -> Result<(), PalsError> {
    e.object(|o| match value {
        ExternalRawScore::Unknown => o.field("kind", "Unknown"),
        ExternalRawScore::Centipawns(value) => {
            o.field("kind", "Centipawns")?;
            o.field("value", value)
        }
        ExternalRawScore::MateMoves(value) => {
            o.field("kind", "MateMoves")?;
            o.field("value", value)
        }
    })
}
fn attempt(e: &mut Encoder<'_>, value: &CheckerAttempt) -> Result<(), PalsError> {
    e.object(|o| {
        o.nested("work", |e| checker_work(e, value.work))?;
        o.nested("elapsed", |e| duration(e, value.elapsed))?;
        o.nested("external", |e| {
            e.option(value.external.as_ref(), |e, external| {
                e.object(|o| {
                    o.field("request_id", external.request_id)?;
                    o.nested("partial_report", |e| {
                        e.option(external.partial_report.as_ref(), raw_report)
                    })?;
                    o.nested("process", |e| shutdown(e, external.process))
                })
            })
        })
    })
}
fn checker_work(
    e: &mut Encoder<'_>,
    value: crate::cpu_checker::CheckerWork,
) -> Result<(), PalsError> {
    e.object(|o| {
        o.field("nodes", value.nodes)?;
        o.field("qnodes", value.qnodes)?;
        o.field("tt_hits", value.tt_hits)
    })
}
fn shutdown(e: &mut Encoder<'_>, value: CheckerShutdown) -> Result<(), PalsError> {
    e.object(|o| {
        o.nested("process_identity", |e| {
            e.option(value.process_identity, process)
        })?;
        o.field("stop_sent", value.stop_sent)?;
        o.field("quit_sent", value.quit_sent)?;
        o.field("exit_observed", value.exit_observed)?;
        o.field("stdout_drained", value.stdout_drained)?;
        o.field("stderr_drained", value.stderr_drained)?;
        o.field("exit_code", value.exit_code)?;
        o.field("exit_signal", value.exit_signal)?;
        o.field("cleanup_complete", value.cleanup_complete)?;
        o.field("quarantined", value.quarantined)?;
        o.field("ownership_lost", value.ownership_lost)?;
        o.field("stdout_bytes", value.stdout_bytes)?;
        o.field("stderr_bytes", value.stderr_bytes)
    })
}

fn pals_error(e: &mut Encoder<'_>, value: &PalsError) -> Result<(), PalsError> {
    e.object(|o| {
        o.field("domain", "PalsError")?;
        o.field("message", DisplayText(value))?;
        match value {
            PalsError::InvalidConfig => o.field("variant", "InvalidConfig"),
            PalsError::InvalidLimits => o.field("variant", "InvalidLimits"),
            PalsError::Capacity => o.field("variant", "Capacity"),
            PalsError::RoleCallLimit => o.field("variant", "RoleCallLimit"),
            PalsError::Checker(error) => {
                o.field("variant", "Checker")?;
                o.nested("cause", |e| checker_error(e, error))
            }
            PalsError::Store(error) => {
                o.field("variant", "Store")?;
                o.nested("cause", |e| store_error(e, error))
            }
            PalsError::Rules(error) => {
                o.field("variant", "Rules")?;
                o.nested("cause", |e| position_error(e, error))
            }
            PalsError::Cpu(error) => {
                o.field("variant", "Cpu")?;
                o.nested("cause", |e| cpu_error(e, error))
            }
            PalsError::Role(error) => {
                o.field("variant", "Role")?;
                o.nested("cause", |e| role_error(e, error))
            }
            PalsError::Review(error) => {
                o.field("variant", "Review")?;
                o.nested("cause", |e| review_error(e, error))
            }
            PalsError::Rovent(error) => {
                o.field("variant", "Rovent")?;
                o.nested("cause", |e| frontier_error(e, error))
            }
        }
    })
}
fn checker_error(e: &mut Encoder<'_>, value: &CheckerError) -> Result<(), PalsError> {
    e.object(|o| {
        o.field("domain", "CheckerError")?;
        o.field("message", DisplayText(value))?;
        match value {
            CheckerError::Owned(error) => {
                o.field("variant", "Owned")?;
                o.nested("cause", |e| cpu_error(e, error))
            }
            CheckerError::OwnedReportRejected { reason } => {
                o.field("variant", "OwnedReportRejected")?;
                o.field("reason", reason)
            }
            CheckerError::Invalid(detail) => {
                o.field("variant", "Invalid")?;
                o.field("detail", detail)
            }
            CheckerError::Unsupported(detail) => {
                o.field("variant", "Unsupported")?;
                o.field("detail", detail)
            }
            CheckerError::External { stage, code } => {
                o.field("variant", "External")?;
                o.field("stage", stage)?;
                o.field("code", code)
            }
        }
    })
}
fn store_error(e: &mut Encoder<'_>, value: &StoreError) -> Result<(), PalsError> {
    e.object(|o| {
        o.field("domain", "StoreError")?;
        o.field("message", DisplayText(value))?;
        let variant = match value {
            StoreError::Capacity(detail) => {
                o.field("detail", detail)?;
                "Capacity"
            }
            StoreError::InvalidHandle(detail) => {
                o.field("detail", detail)?;
                "InvalidHandle"
            }
            StoreError::InvalidEvidence(detail) => {
                o.field("detail", detail)?;
                "InvalidEvidence"
            }
            StoreError::InvalidConditions(detail) => {
                o.field("detail", detail)?;
                "InvalidConditions"
            }
            StoreError::ArchiveQuota(detail) => {
                o.field("detail", detail)?;
                "ArchiveQuota"
            }
            StoreError::ArchiveIntegrity(detail) => {
                o.field("detail", detail)?;
                "ArchiveIntegrity"
            }
            StoreError::PinSaturated(detail) => {
                o.field("detail", detail)?;
                "PinSaturated"
            }
            StoreError::ColdRecord(detail) => {
                o.field("detail", detail)?;
                "ColdRecord"
            }
            StoreError::ArchiveIo { stage, kind } => {
                o.field("stage", stage)?;
                o.field("kind", DebugText(kind))?;
                "ArchiveIo"
            }
            StoreError::ReviewLeaseBackpressure {
                current,
                capacity,
                required_new,
                replacement_reserve,
            } => {
                o.field("current", current)?;
                o.field("capacity", capacity)?;
                o.field("required_new", required_new)?;
                o.field("replacement_reserve", replacement_reserve)?;
                "ReviewLeaseBackpressure"
            }
            StoreError::InvalidMove => "InvalidMove",
            StoreError::ExpiredConsumer => "ExpiredConsumer",
            StoreError::StaleConsumer => "StaleConsumer",
            StoreError::CancelledConsumer => "CancelledConsumer",
            StoreError::AlreadyConsumed => "AlreadyConsumed",
            StoreError::NotCompleted => "NotCompleted",
            StoreError::RevisionExhausted => "RevisionExhausted",
            StoreError::ArchiveDisabled => "ArchiveDisabled",
            StoreError::ArchiveBudgetRequired => "ArchiveBudgetRequired",
            StoreError::ArchiveCanceled => "ArchiveCanceled",
            StoreError::ArchiveDeadline => "ArchiveDeadline",
            StoreError::ArchiveByteBudget => "ArchiveByteBudget",
            StoreError::ForeignPermitStale => "ForeignPermitStale",
        };
        o.field("variant", variant)
    })
}
fn position_error(e: &mut Encoder<'_>, value: &PositionError) -> Result<(), PalsError> {
    e.object(|o| {
        o.field("domain", "PositionError")?;
        o.field("message", DisplayText(value))?;
        let variant = match value {
            PositionError::InvalidFen(detail) => {
                o.field("detail", detail)?;
                "InvalidFen"
            }
            PositionError::InvalidMove(detail) => {
                o.field("detail", detail)?;
                "InvalidMove"
            }
            PositionError::ResourceLimit(detail) => {
                o.field("detail", detail)?;
                "ResourceLimit"
            }
            PositionError::IllegalMove => "IllegalMove",
            PositionError::CounterOverflow => "CounterOverflow",
            PositionError::RevisionExhausted => "RevisionExhausted",
            PositionError::UndoMismatch => "UndoMismatch",
            PositionError::StaleView => "StaleView",
        };
        o.field("variant", variant)
    })
}
fn cpu_error(e: &mut Encoder<'_>, value: &crate::cpu::CpuError) -> Result<(), PalsError> {
    use crate::cpu::CpuError;
    e.object(|o| {
        o.field("domain", "CpuError")?;
        o.field("message", DisplayText(value))?;
        let variant = match value {
            CpuError::InvalidConfig(detail) => {
                o.field("detail", detail)?;
                "InvalidConfig"
            }
            CpuError::InvalidLimits(detail) => {
                o.field("detail", detail)?;
                "InvalidLimits"
            }
            CpuError::InvalidRootMoves(detail) => {
                o.field("detail", detail)?;
                "InvalidRootMoves"
            }
            CpuError::ResumeMismatch(detail) => {
                o.field("detail", detail)?;
                "ResumeMismatch"
            }
            CpuError::Unsupported(detail) => {
                o.field("detail", detail)?;
                "Unsupported"
            }
            CpuError::PausedStackMemoryLimit {
                retained_bytes,
                maximum_bytes,
            } => {
                o.field("retained_bytes", retained_bytes)?;
                o.field("maximum_bytes", maximum_bytes)?;
                "PausedStackMemoryLimit"
            }
            CpuError::Allocation => "Allocation",
            CpuError::Rules(error) => {
                o.nested("cause", |e| position_error(e, error))?;
                "Rules"
            }
            CpuError::Value(error) => {
                o.nested("cause", |e| cpu_value_error(e, error))?;
                "Value"
            }
        };
        o.field("variant", variant)
    })
}
fn cpu_value_error(
    e: &mut Encoder<'_>,
    value: &crate::cpu_value::CpuValueError,
) -> Result<(), PalsError> {
    use crate::cpu_value::{CpuValueAuthorityRevocation, CpuValueError};
    e.object(|o| {
        o.field("domain", "CpuValueError")?;
        o.field("message", DisplayText(value))?;
        let variant = match value {
            CpuValueError::CheckpointSize => "CheckpointSize",
            CpuValueError::InvalidCheckpoint(detail) => {
                o.field("detail", detail)?;
                "InvalidCheckpoint"
            }
            CpuValueError::Io(detail) => {
                o.field("detail", detail)?;
                "Io"
            }
            CpuValueError::Json(detail) => {
                o.field("detail", detail)?;
                "Json"
            }
            CpuValueError::StateMismatch => "StateMismatch",
            CpuValueError::AuthorityRevoked(reason) => {
                o.field(
                    "reason",
                    match reason {
                        CpuValueAuthorityRevocation::Cancelled => "Cancelled",
                        CpuValueAuthorityRevocation::Expired => "Expired",
                        CpuValueAuthorityRevocation::ActorStopped => "ActorStopped",
                        CpuValueAuthorityRevocation::Invalidated => "Invalidated",
                    },
                )?;
                "AuthorityRevoked"
            }
            CpuValueError::NonFiniteForward => "NonFiniteForward",
            CpuValueError::ScoreOutsideFrontierNamespace => "ScoreOutsideFrontierNamespace",
            CpuValueError::AccumulatorOverflow => "AccumulatorOverflow",
        };
        o.field("variant", variant)
    })
}
fn role_error(e: &mut Encoder<'_>, value: &RoleError) -> Result<(), PalsError> {
    e.object(|o| {
        o.field("domain", "RoleError")?;
        o.field("message", DisplayText(value))?;
        let variant = match value {
            RoleError::Unavailable => "Unavailable",
            RoleError::Unsupported(detail) => {
                o.field("detail", detail)?;
                "Unsupported"
            }
            RoleError::InvalidOutput => "InvalidOutput",
            RoleError::Canceled => "Canceled",
            RoleError::Deadline => "Deadline",
            RoleError::PhysicalCompletionUnknown => "PhysicalCompletionUnknown",
            RoleError::Backend(detail) => {
                o.field("detail", detail)?;
                "Backend"
            }
        };
        o.field("variant", variant)
    })
}
fn review_error(
    e: &mut Encoder<'_>,
    value: &crate::pals::review_context::ReviewError,
) -> Result<(), PalsError> {
    use crate::pals::review_context::ReviewError;
    e.object(|o| {
        o.field("domain", "ReviewError")?;
        o.field("message", DisplayText(value))?;
        let variant = match value {
            ReviewError::InvalidPolicy => "InvalidPolicy",
            ReviewError::OwnerMismatch => "OwnerMismatch",
            ReviewError::StaleGeneration => "StaleGeneration",
            ReviewError::DuplicateRevision => "DuplicateRevision",
            ReviewError::MissingRelation(id) => {
                o.field("relation", id)?;
                "MissingRelation"
            }
            ReviewError::RelationMismatch => "RelationMismatch",
            ReviewError::InvalidLine => "InvalidLine",
            ReviewError::AnchorMismatch => "AnchorMismatch",
            ReviewError::ViewCapacity => "ViewCapacity",
            ReviewError::IndexCapacity => "IndexCapacity",
            ReviewError::BundleRecordCapacity => "BundleRecordCapacity",
            ReviewError::AlreadyConsumed => "AlreadyConsumed",
            ReviewError::Store(error) => {
                o.nested("cause", |e| store_error(e, error))?;
                "Store"
            }
        };
        o.field("variant", variant)
    })
}
fn frontier_error(
    e: &mut Encoder<'_>,
    value: &crate::pals::rovent_frontier::FrontierError,
) -> Result<(), PalsError> {
    use crate::pals::rovent_frontier::FrontierError;
    e.object(|o| {
        o.field("domain", "FrontierError")?;
        o.field("message", DisplayText(value))?;
        let variant = match value {
            FrontierError::Invalid(detail) => {
                o.field("detail", detail)?;
                "Invalid"
            }
            FrontierError::Capacity(detail) => {
                o.field("detail", detail)?;
                "Capacity"
            }
            FrontierError::PublicRecordCapacity => "PublicRecordCapacity",
            FrontierError::PublicReviewIndexCapacity => "PublicReviewIndexCapacity",
            FrontierError::ForeignRoot => "ForeignRoot",
            FrontierError::ForeignHandle => "ForeignHandle",
            FrontierError::Duplicate => "Duplicate",
            FrontierError::PolicySealed => "PolicySealed",
            FrontierError::DepthHorizon => "DepthHorizon",
            FrontierError::NodeBudget => "NodeBudget",
            FrontierError::NodesReserved => "NodesReserved",
            FrontierError::NativeUnknown => "NativeUnknown",
            FrontierError::MissingAcceptance => "MissingAcceptance",
            FrontierError::Deadline => "Deadline",
            FrontierError::Revoked => "Revoked",
            FrontierError::StoreUnavailable => "StoreUnavailable",
            FrontierError::Horizon(error) => {
                o.nested("cause", |e| {
                    e.object(|o| {
                        o.field("domain", "HorizonError")?;
                        o.field("variant", DebugText(error))
                    })
                })?;
                "Horizon"
            }
            FrontierError::Mate(error) => {
                o.nested("cause", |e| {
                    e.object(|o| {
                        o.field("domain", "MateProofError")?;
                        o.field("variant", DebugText(error))
                    })
                })?;
                "Mate"
            }
        };
        o.field("variant", variant)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    struct FixedOutput {
        bytes: [u8; 8192],
        used: usize,
    }
    impl Default for FixedOutput {
        fn default() -> Self {
            Self {
                bytes: [0; 8192],
                used: 0,
            }
        }
    }
    impl Write for FixedOutput {
        fn write(&mut self, value: &[u8]) -> io::Result<usize> {
            let end = self
                .used
                .checked_add(value.len())
                .filter(|end| *end <= self.bytes.len())
                .ok_or_else(|| io::Error::new(io::ErrorKind::WriteZero, "fixed fixture output"))?;
            self.bytes[self.used..end].copy_from_slice(value);
            self.used = end;
            Ok(value.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
    fn projection_bytes(body: impl FnOnce(&mut Encoder<'_>) -> Result<(), PalsError>) -> Vec<u8> {
        let stores = PalsStores::new(StoreLimits::default());
        let cancel = AtomicBool::new(false);
        let origin = Instant::now();
        let mut output = FixedOutput::default();
        {
            let mut encoder = Encoder {
                writer: CheckedWriter {
                    inner: &mut output,
                    until: origin + Duration::from_secs(30),
                    cancel: &cancel,
                },
                stores: &stores,
                origin,
                summary: CodecSummary::default(),
            };
            body(&mut encoder).unwrap();
        }
        output.bytes[..output.used].to_vec()
    }
    fn projection(
        body: impl FnOnce(&mut Encoder<'_>) -> Result<(), PalsError>,
    ) -> serde_json::Value {
        serde_json::from_slice(&projection_bytes(body)).unwrap()
    }

    #[test]
    fn public_record_capacity_projection_preserves_its_distinct_cause() {
        use crate::pals::rovent_frontier::FrontierError;
        let value =
            projection(|encoder| frontier_error(encoder, &FrontierError::PublicRecordCapacity));
        assert_eq!(value["domain"], "FrontierError");
        assert_eq!(value["variant"], "PublicRecordCapacity");
        assert!(value.get("detail").is_none());
        let legacy = projection(|encoder| {
            frontier_error(encoder, &FrontierError::Capacity("original limit"))
        });
        assert_eq!(legacy["variant"], "Capacity");
        assert_eq!(legacy["detail"], "original limit");
        // Historical JSON records the cause; it cannot recreate the private
        // live controller marker that authorizes graceful quiescence.
    }

    #[test]
    fn public_review_bundle_capacity_projection_preserves_its_distinct_cause() {
        use crate::pals::review_context::ReviewError;
        use crate::pals::rovent_frontier::FrontierError;
        let review =
            projection(|encoder| review_error(encoder, &ReviewError::BundleRecordCapacity));
        assert_eq!(review["domain"], "ReviewError");
        assert_eq!(review["variant"], "BundleRecordCapacity");
        assert!(review.get("detail").is_none());
        let frontier = projection(|encoder| {
            frontier_error(encoder, &FrontierError::PublicReviewIndexCapacity)
        });
        assert_eq!(frontier["domain"], "FrontierError");
        assert_eq!(frontier["variant"], "PublicReviewIndexCapacity");
        assert!(frontier.get("detail").is_none());
        let generic = projection(|encoder| review_error(encoder, &ReviewError::IndexCapacity));
        assert_eq!(generic["variant"], "IndexCapacity");
        // Saved cause JSON cannot recreate either private live-controller marker.
    }

    #[test]
    fn cpu_value_revocation_roundtrip_preserves_each_original_typed_reason() {
        use crate::cpu::{CpuCompletion, CpuError};
        use crate::cpu_value::{CpuValueAuthorityRevocation, CpuValueError};
        for (reason, expected, stop) in [
            (
                CpuValueAuthorityRevocation::Cancelled,
                "Cancelled",
                Some(CpuCompletion::Canceled),
            ),
            (
                CpuValueAuthorityRevocation::Expired,
                "Expired",
                Some(CpuCompletion::Deadline),
            ),
            (
                CpuValueAuthorityRevocation::ActorStopped,
                "ActorStopped",
                None,
            ),
            (
                CpuValueAuthorityRevocation::Invalidated,
                "Invalidated",
                None,
            ),
        ] {
            let error = CpuError::Value(CpuValueError::AuthorityRevoked(reason));
            let value = projection(|encoder| cpu_error(encoder, &error));
            assert_eq!(value["domain"], "CpuError");
            assert_eq!(value["variant"], "Value");
            assert_eq!(value["cause"]["domain"], "CpuValueError");
            assert_eq!(value["cause"]["variant"], "AuthorityRevoked");
            assert_eq!(value["cause"]["reason"], expected);
            assert_eq!(value["cause"].as_object().unwrap().len(), 4);
            assert_eq!(
                value["cause"]["message"],
                format!("original CPU NNUEC authority revoked: {expected}"),
            );
            assert_eq!(value["message"], error.to_string());
            // JSON readback is historical data only. It creates no value,
            // live error/owner, cancellation or cleanup acknowledgement.
            assert_eq!(error.value_authority_stop(), stop);
        }
    }

    #[test]
    fn legacy_cpu_value_error_projection_bytes_remain_unchanged() {
        use crate::cpu_value::CpuValueError;
        for (error, expected) in [
            (CpuValueError::CheckpointSize,
                br#"{"domain":"CpuValueError","message":"CPU checkpoint exceeds 8 MiB","variant":"CheckpointSize"}"#.as_slice()),
            (CpuValueError::InvalidCheckpoint("frozen source"),
                br#"{"domain":"CpuValueError","message":"CPU checkpoint: frozen source","detail":"frozen source","variant":"InvalidCheckpoint"}"#.as_slice()),
            (CpuValueError::Io("disk original".into()),
                br#"{"domain":"CpuValueError","message":"CPU checkpoint read: disk original","detail":"disk original","variant":"Io"}"#.as_slice()),
            (CpuValueError::Json("syntax original".into()),
                br#"{"domain":"CpuValueError","message":"CPU checkpoint JSON: syntax original","detail":"syntax original","variant":"Json"}"#.as_slice()),
            (CpuValueError::StateMismatch,
                br#"{"domain":"CpuValueError","message":"CPU accumulator does not match exact Rules state or value namespace","variant":"StateMismatch"}"#.as_slice()),
            (CpuValueError::NonFiniteForward,
                br#"{"domain":"CpuValueError","message":"CPU FP32 forward produced a non-finite value","variant":"NonFiniteForward"}"#.as_slice()),
            (CpuValueError::ScoreOutsideFrontierNamespace,
                br#"{"domain":"CpuValueError","message":"CPU nonterminal evaluator score entered the Rules mate namespace","variant":"ScoreOutsideFrontierNamespace"}"#.as_slice()),
            (CpuValueError::AccumulatorOverflow,
                br#"{"domain":"CpuValueError","message":"CPU signed192 accumulator overflow","variant":"AccumulatorOverflow"}"#.as_slice()),
        ] {
            assert_eq!(projection_bytes(|encoder| cpu_value_error(encoder, &error)), expected);
        }
    }

    #[test]
    fn failed_source_keeps_typed_checker_stage_code_and_full_message() {
        let value = projection(|encoder| {
            pals_error(
                encoder,
                &PalsError::Checker(CheckerError::External {
                    stage: "stop/drain\n원 단계",
                    code: "pipe_\"original\"_failure",
                }),
            )
        });
        assert_eq!(value["domain"], "PalsError");
        assert_eq!(value["variant"], "Checker");
        assert_eq!(value["cause"]["domain"], "CheckerError");
        assert_eq!(value["cause"]["variant"], "External");
        assert_eq!(value["cause"]["stage"], "stop/drain\n원 단계");
        assert_eq!(value["cause"]["code"], "pipe_\"original\"_failure");
        assert!(
            value["cause"]["message"]
                .as_str()
                .unwrap()
                .contains("원 단계")
        );
    }

    #[test]
    fn raw_store_failure_keeps_large_integer_fields() {
        let Ok(large) = usize::try_from((1u64 << 53) + 17) else {
            // This exact integer cannot exist in the source type on 32-bit.
            return;
        };
        let value = projection(|encoder| {
            store_error(
                encoder,
                &StoreError::ReviewLeaseBackpressure {
                    current: large,
                    capacity: large + 1,
                    required_new: 3,
                    replacement_reserve: 1,
                },
            )
        });
        assert_eq!(value["variant"], "ReviewLeaseBackpressure");
        assert_eq!(value["current"].as_u64(), Some(large as u64));
        assert_eq!(value["capacity"].as_u64(), Some((large + 1) as u64));
        assert_eq!(value["required_new"], 3);
    }

    #[test]
    fn monotonic_before_origin_is_signed_and_missing_time_stays_null() {
        let value = projection(|encoder| {
            let before = encoder.origin - Duration::new(1, 7);
            encoder.object(|o| {
                o.nested("observed", |e| e.instant(before))?;
                o.nested("unknown", |e| {
                    e.option(None::<Instant>, |e, value| e.instant(value))
                })
            })
        });
        assert_eq!(value["observed"]["negative"], true);
        assert_eq!(value["observed"]["seconds"], 1);
        assert_eq!(value["observed"]["nanoseconds"], 7);
        assert_eq!(value["unknown"], serde_json::Value::Null);
    }

    #[test]
    fn canceled_writer_returns_non_retryable_error_and_writes_nothing() {
        let cancel = AtomicBool::new(true);
        let mut output = FixedOutput::default();
        let mut writer = CheckedWriter {
            inner: &mut output,
            until: Instant::now() + Duration::from_secs(30),
            cancel: &cancel,
        };
        assert_eq!(
            writer.write_all(b"footer").unwrap_err().kind(),
            io::ErrorKind::ConnectionAborted
        );
        assert_eq!(output.used, 0);
    }
}
