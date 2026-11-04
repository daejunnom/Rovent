//! Same-owner LC0 service and bounded control observations.
//!
//! This is a retaining component for a future executable supervisor, not a
//! process-level supervisor or permission to exit on an unknown result. The
//! original authenticated admission channel and the physical owner remain held
//! by `Lc0SystemSessionOwner`. Control arrives on a different authenticated
//! connection and cannot authorize termination, replace a GPU fence, or
//! silently retry native initialization.

use super::projection::{Lc0ObservedRelease, Lc0ObservedService};
use super::*;
use serde::Deserialize;
use std::{
    any::Any,
    io::{Read, Write},
    os::unix::net::UnixListener,
    panic::{AssertUnwindSafe, catch_unwind},
    time::Duration,
};

const CONTROL_SCHEMA: &str = "rz-lc0-system-control/1";
const RESPONSE_SCHEMA: &str = "rz-lc0-system-control-response/1";
const MAX_CONTROL_FRAME: usize = 4096;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ServiceState {
    Prepared,
    Serving,
    DrainRequested,
    /// Only means the original session owner reported session release.
    SessionReleased,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ControlRequest {
    schema: String,
    operation: ControlOperation,
    sequence: u64,
    process_epoch: u64,
    lock_sha256: String,
    cuda_ordinal: i32,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
enum ControlOperation {
    Status,
    Reobserve,
}

/// Owns the actual session and service outcome together. Caller must retain
/// this value and its process while release is pending or unknown, including
/// if the parent disconnects or a diagnostic receipt cannot be written.
pub(super) struct Lc0ControlSupervisor {
    owner: Lc0SystemSessionOwner,
    state: ServiceState,
    service_started: bool,
    service_failure_projection: Option<&'static str>,
    // Held independently until the final report has accepted ownership.
    service_failure: Option<EngineError>,
    finish: Option<Lc0SystemFinish>,
    // Preserve the original panic payload without rendering possibly sensitive
    // details or treating it as a normally completed protocol service.
    service_panic: Option<Box<dyn Any + Send>>,
    finish_panic: Option<Box<dyn Any + Send>>,
    setup_attempted: bool,
    setup_failure: Option<NativeBootstrapError>,
    setup_panic: Option<Box<dyn Any + Send>>,
    finish_failure: Option<NativeBootstrapError>,
    publication_panic: Option<Box<dyn Any + Send>>,
    unstarted_release: Option<Poll<Result<SessionReleased, PhysicalFailure>>>,
    followup_release: Option<Poll<Result<SessionReleased, PhysicalFailure>>>,
    followup_count: u64,
    control_until: Instant,
    last_control: Option<(u64, ControlOperation, serde_json::Value)>,
}

pub(super) struct ControlProjection {
    pub(super) ort_entry_observed: bool,
    pub(super) service: Lc0ObservedService,
    pub(super) original_release: Lc0ObservedRelease,
    pub(super) latest_release: Lc0ObservedRelease,
    pub(super) scope_settled: bool,
}

fn remaining(until: Instant) -> io::Result<Duration> {
    until
        .checked_duration_since(Instant::now())
        .filter(|remaining| !remaining.is_zero())
        .ok_or_else(|| io::Error::new(io::ErrorKind::TimedOut, "LC0 control deadline expired"))
}

fn read_frame(stream: &mut UnixStream, until: Instant) -> io::Result<ControlRequest> {
    let mut frame = Vec::with_capacity(256);
    loop {
        stream.set_read_timeout(Some(remaining(until)?))?;
        let mut byte = [0];
        match stream.read(&mut byte) {
            Ok(0) => {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "LC0 control closed",
                ));
            }
            Ok(_) if byte[0] == b'\n' => break,
            Ok(_) if frame.len() < MAX_CONTROL_FRAME - 1 => frame.push(byte[0]),
            Ok(_) => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "LC0 control frame too large",
                ));
            }
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            Err(error) => return Err(error),
        }
    }
    serde_json::from_slice(&frame)
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "invalid LC0 control frame"))
}

fn write_frame(
    stream: &mut UnixStream,
    value: &serde_json::Value,
    until: Instant,
) -> io::Result<()> {
    let mut bytes = serde_json::to_vec(value).map_err(io::Error::other)?;
    if bytes.len() >= MAX_CONTROL_FRAME {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "LC0 control response too large",
        ));
    }
    bytes.push(b'\n');
    let mut offset = 0;
    while offset < bytes.len() {
        stream.set_write_timeout(Some(remaining(until)?))?;
        match stream.write(&bytes[offset..]) {
            Ok(0) => {
                return Err(io::Error::new(
                    io::ErrorKind::WriteZero,
                    "LC0 control write failed",
                ));
            }
            Ok(n) => offset += n,
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            Err(error) => return Err(error),
        }
    }
    Ok(())
}

// These projections deliberately omit external error text and panic payloads.
// The original errors and physical owner remain in the supervisor.
fn contract_failure(error: &ContractError) -> serde_json::Value {
    serde_json::json!({
        "code": format!("{:?}", error.code),
        "stage": format!("{:?}", error.stage),
    })
}

pub(super) fn backend_failure(error: &BackendError) -> serde_json::Value {
    serde_json::json!({
        "kind": format!("{:?}", error.kind),
        "stage": format!("{:?}", error.stage),
        "native_detail_retained_by_owner": error.native.is_some(),
    })
}

fn physical_failure(error: &PhysicalFailure) -> serde_json::Value {
    serde_json::json!({
        "contract": contract_failure(&error.contract),
        "backend": error.backend.as_ref().map(backend_failure),
    })
}

fn bootstrap_failure(error: &NativeBootstrapError) -> serde_json::Value {
    match error {
        NativeBootstrapError::Config(_) => serde_json::json!({"type": "config"}),
        NativeBootstrapError::Backend(error) => {
            serde_json::json!({"type": "backend", "error": backend_failure(error)})
        }
        NativeBootstrapError::Physical(error) => {
            serde_json::json!({"type": "physical", "error": physical_failure(error)})
        }
        NativeBootstrapError::Contract(error) => {
            serde_json::json!({"type": "contract", "error": contract_failure(error)})
        }
    }
}

fn service_failure(error: &EngineError) -> &'static str {
    match error {
        EngineError::Contract(_) => "contract",
        EngineError::SearchDriverClose(_) => "search_driver_close",
        EngineError::Session(_) => "session",
        EngineError::Transport(_) => "transport",
        EngineError::WorkerPanic => "worker_panic",
        EngineError::DrainTimeout => "drain_timeout",
        EngineError::UndeliveredDiagnostics(_) => "undelivered_diagnostics",
        EngineError::UndeliveredWorkerFailure(_) => "undelivered_worker_failure",
        EngineError::Cleanup { .. } => "cleanup",
    }
}

impl Lc0ControlSupervisor {
    pub(super) fn project_original(&self) -> ControlProjection {
        let release = |value: Option<&Poll<Result<SessionReleased, PhysicalFailure>>>| match value {
            None => Lc0ObservedRelease::NotObserved,
            Some(Poll::Pending) => Lc0ObservedRelease::Pending,
            Some(Poll::Ready(Ok(_))) => Lc0ObservedRelease::Released,
            Some(Poll::Ready(Err(_))) => Lc0ObservedRelease::Failed,
        };
        let original = self
            .finish
            .as_ref()
            .and_then(|v| v.session_release.as_ref())
            .or(self.unstarted_release.as_ref());
        let service = if !self.service_started {
            Lc0ObservedService::NotStarted
        } else if self.state == ServiceState::Serving {
            Lc0ObservedService::Serving
        } else if self.service_panic.is_some() || self.service_failure_projection.is_some() {
            Lc0ObservedService::Failed
        } else if matches!(
            self.state,
            ServiceState::DrainRequested | ServiceState::SessionReleased
        ) {
            // This state is reached only after the UCI service returned.
            // Session/receipt failures are separate observations below.
            Lc0ObservedService::Succeeded
        } else {
            Lc0ObservedService::Unknown
        };
        ControlProjection {
            ort_entry_observed: self.owner.custody.ort_entry_observed(),
            service,
            original_release: release(original),
            latest_release: release(self.followup_release.as_ref().or(original)),
            scope_settled: self.original_scope_settled(),
        }
    }
    pub(super) fn new(owner: Lc0SystemSessionOwner, control_until: Instant) -> Self {
        Self {
            owner,
            state: ServiceState::Prepared,
            service_started: false,
            service_failure_projection: None,
            service_failure: None,
            finish: None,
            service_panic: None,
            finish_panic: None,
            setup_attempted: false,
            setup_failure: None,
            setup_panic: None,
            finish_failure: None,
            publication_panic: None,
            unstarted_release: None,
            followup_release: None,
            followup_count: 0,
            control_until,
            last_control: None,
        }
    }

    /// This only determines whether this original process session is settled;
    /// it is not a device-wide GPU fence or V6 physical-completion claim.
    pub(super) fn original_scope_settled(&self) -> bool {
        if !self.service_started
            && self.state != ServiceState::Serving
            && self.owner.custody.phase() == Lc0SystemStartupPhase::BeforeOrtCall
            && !self.owner.custody.ort_entry_observed()
            && self.owner.custody.runtime.is_none()
            && self.owner.custody.model.is_none()
            && self.owner.custody.worker.is_none()
        {
            return true;
        }
        if self.state != ServiceState::SessionReleased
            || self.setup_panic.is_some()
            || self.finish_panic.is_some()
            || self.finish_failure.is_some()
        {
            return false;
        }
        match &self.finish {
            Some(finish) => match &finish.run {
                Ok(_) => true,
                Err(error) => {
                    error.service.is_none()
                        && error.collection_error.is_none()
                        && error.worker_shutdown_error.is_none()
                        && error.report.is_some()
                }
            },
            None => matches!(&self.unstarted_release, Some(Poll::Ready(Ok(_)))),
        }
    }

    /// Only the recorded service/factory result, not session release,
    /// publication success, or process-wide completion.
    pub(super) fn original_service_succeeded(&self) -> bool {
        self.setup_failure.is_none()
            && self.setup_panic.is_none()
            && self.service_panic.is_none()
            && self.finish_panic.is_none()
            && self.finish_failure.is_none()
            && self
                .finish
                .as_ref()
                .is_some_and(|finish| finish.run.is_ok())
    }

    /// Read-only, bounded classification of the first failures. This is not a
    /// completion witness and must never replace the retained original errors.
    pub(super) fn original_failure_observation(&self) -> serde_json::Value {
        let failure = self
            .finish
            .as_ref()
            .and_then(|finish| finish.run.as_ref().err());
        let original_session = self
            .finish
            .as_ref()
            .and_then(|finish| finish.session_release.as_ref())
            .or(self.unstarted_release.as_ref());
        let release_error =
            |release: Option<&Poll<Result<SessionReleased, PhysicalFailure>>>| match release {
                Some(Poll::Ready(Err(error))) => Some(physical_failure(error)),
                _ => None,
            };
        let service_original_retained = self.service_failure_projection.is_none()
            || self.service_failure.is_some()
            || failure.is_some_and(|error| error.service.is_some());
        serde_json::json!({
            "startup_attempted": self.setup_attempted,
            "startup": self.setup_failure.as_ref().map(bootstrap_failure),
            "startup_panic": self.setup_panic.is_some(),
            "service_started": self.service_started,
            "service": self.service_failure_projection,
            "service_original_retained": service_original_retained,
            "service_panic": self.service_panic.is_some(),
            "finish_observed": self.finish.is_some(),
            "finish": self.finish_failure.as_ref().map(bootstrap_failure),
            "finish_panic": self.finish_panic.is_some(),
            "collection": failure.and_then(|error| error.collection_error.as_ref()).map(contract_failure),
            "worker_shutdown": failure.and_then(|error| error.worker_shutdown_error.as_deref()).map(physical_failure),
            "original_session_release_error": release_error(original_session),
            "followup_session_release_error": release_error(self.followup_release.as_ref()),
            "runtime_mapping": failure.and_then(|error| error.runtime_mapping_error.as_deref()).map(backend_failure),
            "publication_panic": self.publication_panic.is_some(),
            "physical_completion": "unknown",
        })
    }

    pub(super) fn retain_publication_panic(&mut self, payload: Box<dyn Any + Send>) {
        self.publication_panic.get_or_insert(payload);
    }

    /// A missing or failed report is separate from the service and session
    /// outcome. These methods catch publisher unwinds while retaining owner,
    /// and return the original ordinary publication error to the caller.
    pub(super) fn publish_runtime_receipt(
        &mut self,
        writer: &SystemReceiptWriter,
        lock: &SystemRuntimeLockV6,
        finish: bool,
    ) -> Result<(), String> {
        match catch_unwind(AssertUnwindSafe(|| {
            self.owner.publish_runtime_receipt(writer, lock, finish)
        })) {
            Ok(result) => result,
            Err(payload) => {
                self.publication_panic.get_or_insert(payload);
                Err("LC0 runtime receipt publication panicked; owner retained".into())
            }
        }
    }

    pub(super) fn publish_finish_observation(
        &mut self,
        writer: &SystemReceiptWriter,
        lock: &SystemRuntimeLockV6,
    ) -> Result<(), String> {
        let finish = self.finish.as_ref().ok_or("LC0 finish is unavailable")?;
        match catch_unwind(AssertUnwindSafe(|| {
            self.owner.publish_finish_observation(writer, lock, finish)
        })) {
            Ok(result) => result,
            Err(payload) => {
                self.publication_panic.get_or_insert(payload);
                Err("LC0 finish observation publication panicked; owner retained".into())
            }
        }
    }

    /// Keep the original owner on every failure or unwind, including an ORT
    /// or session entry with unknown completion. No second startup is issued.
    pub(super) fn prepare_until(
        &mut self,
        lock: &SystemRuntimeLockV6,
        until: Instant,
        cancelled: &AtomicBool,
        backend: BackendConfig,
        owners: &OwnerRegistry,
        config: &NativeConfig,
    ) -> Result<(), &'static str> {
        if self.state != ServiceState::Prepared || self.setup_attempted {
            return Err("LC0 original startup already attempted or closed");
        }
        self.setup_attempted = true;
        let outcome = catch_unwind(AssertUnwindSafe(|| {
            self.owner.custody.load(lock, until, cancelled, backend)?;
            self.owner.connect(owners, config)
        }));
        match outcome {
            Ok(Ok(())) => Ok(()),
            Ok(Err(error)) => {
                self.setup_failure = Some(error);
                Err("LC0 original startup failed; owner retained")
            }
            Err(payload) => {
                self.setup_panic.get_or_insert(payload);
                Err("LC0 original startup panicked; owner retained")
            }
        }
    }

    /// Setup and protocol service must be inside `serve`; this prevents a
    /// setup error or unwind after factory creation from skipping the original
    /// owner's bounded finish path. The original panic is retained here.
    pub(super) fn serve_until(
        &mut self,
        until: Instant,
        serve: impl FnOnce(Arc<NativeCudaFactory>) -> Result<(), EngineError>,
    ) -> Result<(), NativeBootstrapError> {
        if self.state != ServiceState::Prepared {
            return Err(NativeBootstrapError::Config(
                "LC0 service already started or closed",
            ));
        }
        let factory = self
            .owner
            .factory()
            .cloned()
            .ok_or(NativeBootstrapError::Config(
                "LC0 system session is not connected",
            ))?;
        self.state = ServiceState::Serving;
        self.service_started = true;
        let result = catch_unwind(AssertUnwindSafe(|| serve(factory)));
        let served = match result {
            Ok(result) => result,
            Err(payload) => {
                self.service_panic.get_or_insert(payload);
                Err(EngineError::WorkerPanic)
            }
        };
        // Keep the original error in the supervisor until the finish report
        // takes it. A panic during collection must not drop the service error.
        self.service_failure_projection = served.as_ref().err().map(service_failure);
        self.service_failure = served.err();
        self.state = ServiceState::DrainRequested;
        let result = catch_unwind(AssertUnwindSafe(|| {
            self.owner.finish_until(&mut self.service_failure, until)
        }));
        let finish = match result {
            Ok(Ok(finish)) => finish,
            Ok(Err(error)) => {
                self.finish_failure = Some(error);
                return Err(NativeBootstrapError::Config(
                    "LC0 final collection failed; original owner retained",
                ));
            }
            Err(payload) => {
                self.finish_panic.get_or_insert(payload);
                return Err(NativeBootstrapError::Config(
                    "LC0 final collection panicked; original owner retained",
                ));
            }
        };
        // Keep the original run report (including any service failure) in
        // the supervisor before the potentially unwinding release call.
        self.finish = Some(finish);
        let release = match catch_unwind(AssertUnwindSafe(|| self.owner.try_release())) {
            Ok(release) => release,
            Err(payload) => {
                self.finish_panic.get_or_insert(payload);
                return Err(NativeBootstrapError::Config(
                    "LC0 session release panicked; original report and owner retained",
                ));
            }
        };
        if matches!(&release, Poll::Ready(Ok(_))) {
            self.state = ServiceState::SessionReleased;
        }
        self.finish
            .as_mut()
            .expect("report retained above")
            .session_release = Some(release);
        Ok(())
    }

    /// Close admission after a pre-service setup failure. If no worker was
    /// installed, completion stays unknown and the original custody is kept.
    pub(super) fn close_without_service(&mut self) {
        if self.state != ServiceState::Prepared {
            return;
        }
        self.state = ServiceState::DrainRequested;
        if self.owner.custody.phase() == Lc0SystemStartupPhase::WorkerOwned {
            let release = match catch_unwind(AssertUnwindSafe(|| self.owner.try_release())) {
                Ok(release) => release,
                Err(payload) => {
                    self.finish_panic.get_or_insert(payload);
                    return;
                }
            };
            if matches!(&release, Poll::Ready(Ok(_))) {
                self.state = ServiceState::SessionReleased;
            }
            self.unstarted_release = Some(release);
        }
    }

    fn reobserve(&mut self) {
        if self.state != ServiceState::DrainRequested || self.finish_panic.is_some() {
            return;
        }
        let previous = self.followup_release.as_ref().or_else(|| {
            self.finish
                .as_ref()
                .and_then(|value| value.session_release.as_ref())
                .or(self.unstarted_release.as_ref())
        });
        if previous.is_some_and(|value| matches!(value, Poll::Pending)) {
            let Some(count) = self.followup_count.checked_add(1) else {
                return;
            };
            let release = match catch_unwind(AssertUnwindSafe(|| self.owner.try_release())) {
                Ok(release) => release,
                Err(payload) => {
                    self.finish_panic.get_or_insert(payload);
                    return;
                }
            };
            if matches!(&release, Poll::Ready(Ok(_))) {
                self.state = ServiceState::SessionReleased;
            }
            self.followup_release = Some(release);
            self.followup_count = count;
        }
    }

    /// Preserve the original finish receipt and append a distinct, numbered
    /// session-only follow-up. Publication failure is an evidence failure;
    /// the caller must not replace the retained physical or service result.
    pub(super) fn publish_reobservation(
        &mut self,
        writer: &SystemReceiptWriter,
        lock: &SystemRuntimeLockV6,
    ) -> Result<(), String> {
        match catch_unwind(AssertUnwindSafe(|| {
            self.publish_reobservation_inner(writer, lock)
        })) {
            Ok(result) => result,
            Err(payload) => {
                self.publication_panic.get_or_insert(payload);
                Err("LC0 reobservation publication panicked; owner retained".into())
            }
        }
    }

    fn publish_reobservation_inner(
        &self,
        writer: &SystemReceiptWriter,
        lock: &SystemRuntimeLockV6,
    ) -> Result<(), String> {
        let spec = match &lock.runtime {
            NativeRuntimeSourceV6::SystemCuda(spec) if lock.verify().is_ok() => spec,
            _ => return Err("LC0 reobservation requires a valid CUDA lock".into()),
        };
        if !self.owner.custody.parent.as_ref().is_some_and(|parent| {
            parent.matches(
                self.owner.custody.process_epoch,
                &lock.canonical_sha256,
                spec.cuda_device_id,
            )
        }) {
            return Err("LC0 reobservation lock differs from the original owner".into());
        }
        let followup = self
            .followup_release
            .as_ref()
            .ok_or("LC0 reobservation is absent")?;
        let (status, completed_runs) = match followup {
            Poll::Pending => ("pending", None),
            Poll::Ready(Err(_)) => ("failed", None),
            Poll::Ready(Ok(release)) => ("released", Some(release.completed_synchronous_runs)),
        };
        let observation = serde_json::json!({
            "domain": "rz-lc0-session-reobservation/1",
            "process_id": std::process::id(),
            "process_epoch": self.owner.custody.process_epoch,
            "lock_sha256": lock.canonical_sha256,
            "sequence": self.followup_count,
            "session_release": { "status": status, "completed_synchronous_runs": completed_runs },
            "physical_completion": "unknown",
            "scope": "original process-local LC0 session only; prior finish and errors remain authoritative"
        });
        let bytes = serde_json::to_vec(&observation).map_err(|error| error.to_string())?;
        let name = format!(
            "lc0-session-reobservation-{:020}.v1.json",
            self.followup_count
        );
        writer.write_bytes(&name, &bytes)
    }

    /// Handle exactly one bounded request on an independently authenticated
    /// connection. The parent cannot issue stop/exit or assert completion.
    pub(super) fn handle_control(&mut self, stream: UnixStream, until: Instant) -> io::Result<()> {
        let until = until.min(self.control_until);
        let mut stream = self.owner.custody.authenticate_control(stream, until)?;
        let request = read_frame(&mut stream, until)?;
        let parent = self
            .owner
            .custody
            .parent
            .as_ref()
            .ok_or_else(|| io::Error::other("LC0 parent missing"))?;
        parent.ensure_live()?;
        if request.schema != CONTROL_SCHEMA
            || !parent.matches(
                request.process_epoch,
                &request.lock_sha256,
                request.cuda_ordinal,
            )
        {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "LC0 control binding differs",
            ));
        }
        if let Some((sequence, operation, response)) = &self.last_control {
            if request.sequence == *sequence {
                if request.operation != *operation {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "LC0 replay operation differs",
                    ));
                }
                let mut replay = response.clone();
                replay["observation"] = serde_json::json!("historical_replay");
                return write_frame(&mut stream, &replay, until);
            }
        }
        let expected = self
            .last_control
            .as_ref()
            .map_or(Some(1), |value| value.0.checked_add(1));
        if Some(request.sequence) != expected {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "LC0 control sequence differs",
            ));
        }
        if matches!(request.operation, ControlOperation::Reobserve) {
            self.reobserve();
        }
        let original_session = if let Some(finish) = &self.finish {
            finish.session_release.as_ref()
        } else {
            self.unstarted_release.as_ref()
        };
        let describe_session =
            |session: Option<&Poll<Result<SessionReleased, PhysicalFailure>>>| match session {
                Some(Poll::Ready(Ok(_))) => "released",
                Some(Poll::Ready(Err(_))) => "failed",
                Some(Poll::Pending) => "pending",
                None => "unknown",
            };
        let response = serde_json::json!({
            "schema": RESPONSE_SCHEMA,
            "process_id": std::process::id(),
            "process_epoch": self.owner.custody.process_epoch,
            "sequence": request.sequence,
            "observation": "current_at_sequence",
            "lock_sha256": request.lock_sha256,
            "startup_phase": format!("{:?}", self.owner.custody.phase()),
            "service_state": format!("{:?}", self.state),
            "service_panic": self.service_panic.is_some(),
            "finish_panic": self.finish_panic.is_some(),
            "finish_failure": self.finish_failure.is_some(),
            "startup_failure": self.setup_failure.is_some(),
            "startup_panic": self.setup_panic.is_some(),
            "publication_panic": self.publication_panic.is_some(),
            "service_error": self.finish.as_ref().map(|value| value.run.as_ref().err().is_some_and(|error| error.service.is_some())),
            "collection_error": self.finish.as_ref().map(|value| value.run.as_ref().err().is_some_and(|error| error.collection_error.is_some())),
            "worker_shutdown_error": self.finish.as_ref().map(|value| value.run.as_ref().err().is_some_and(|error| error.worker_shutdown_error.is_some())),
            "runtime_mapping": self.finish.as_ref().map(|value| match value.mapping {
                Some(CudaRuntimeMappingEvidence::Observed) => "observed",
                Some(CudaRuntimeMappingEvidence::Unknown) => "unknown",
                None if value.run.as_ref().err().is_some_and(|error| error.runtime_mapping_error.is_some()) => "failed",
                None => "not_observed",
            }),
            "original_session_release": describe_session(original_session),
            "latest_session_release": describe_session(self.followup_release.as_ref().or(original_session)),
            "reobservation_count": self.followup_count,
            "physical_completion": "unknown",
            "scope": "authenticated original-parent connection; not per-command sender proof or a device-wide GPU fence",
        });
        self.last_control = Some((request.sequence, request.operation, response.clone()));
        write_frame(&mut stream, &response, until)
    }

    /// Serve a caller-owned, already-bound private listener for this original
    /// control window. A timeout, bad request, or parent disconnect returns
    /// with `self` intact; none permits process exit or custody release. The
    /// caller must keep retaining this supervisor when completion is unknown.
    /// This synchronous loop is entered after service returns and does not
    /// claim to handle commands concurrently with the protocol service.
    pub(super) fn control_loop(&mut self, listener: &UnixListener) -> io::Result<()> {
        listener.set_nonblocking(true)?;
        loop {
            remaining(self.control_until)?;
            match listener.accept() {
                Ok((stream, _)) => self.handle_control(stream, self.control_until)?,
                Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
                Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                    std::thread::sleep(
                        remaining(self.control_until)?.min(Duration::from_millis(10)),
                    );
                }
                Err(error) => return Err(error),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn failure_projection_preserves_types_without_native_text() {
        let backend = BackendError::new(
            FailureKind::BackendUnavailable,
            FailureStage::Backend,
            "do not expose this detail",
        )
        .with_diagnostic("code", "do not expose this native message");
        let projected = bootstrap_failure(&NativeBootstrapError::Backend(Box::new(backend)));
        assert_eq!(projected["type"], "backend");
        assert_eq!(projected["error"]["kind"], "BackendUnavailable");
        assert_eq!(projected["error"]["stage"], "Backend");
        assert_eq!(projected["error"]["native_detail_retained_by_owner"], true);
        assert!(!projected.to_string().contains("do not expose"));
        assert_eq!(service_failure(&EngineError::DrainTimeout), "drain_timeout");
    }

    #[test]
    fn control_frame_is_bounded_and_closed_schema_rejects_extra_fields() {
        let (mut reader, mut writer) = UnixStream::pair().unwrap();
        writer.write_all(br#"{"schema":"rz-lc0-system-control/1","operation":"status","sequence":1,"process_epoch":1,"lock_sha256":"x","cuda_ordinal":0,"extra":true}
"#).unwrap();
        assert_eq!(
            read_frame(&mut reader, Instant::now() + Duration::from_secs(1))
                .unwrap_err()
                .kind(),
            io::ErrorKind::InvalidData
        );
        let (mut reader, mut writer) = UnixStream::pair().unwrap();
        writer.write_all(&vec![b'x'; MAX_CONTROL_FRAME]).unwrap();
        assert_eq!(
            read_frame(&mut reader, Instant::now() + Duration::from_secs(1))
                .unwrap_err()
                .kind(),
            io::ErrorKind::InvalidData
        );
    }
}
