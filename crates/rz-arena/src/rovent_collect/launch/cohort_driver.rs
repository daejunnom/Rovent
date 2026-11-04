//! 유한 cohort caller는 실제 UCI 이벤트와 실제 출력만 관측한다.
//! root/완료 권한은 EngineProcess와 원 Product가 발급하며 이 파일은 만들지 않는다.
use rz_uci::engine::{
    self, EngineError, EngineProcess, EngineSettings, OwnerRegistry, ProcessClock,
};
use rz_uci::rovent_bootstrap::launch::RoventLaunchError;
use rz_uci::rovent_bootstrap::product::RoventNativeProductDriver;
use rz_uci::{Command, Event};
use std::any::Any;
use std::fmt::{self, Write as _};
use std::io::{self, Write};
use std::sync::mpsc::{SyncSender, TrySendError};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;
use std::time::{Duration, Instant};

const CAPTURE_BYTES: usize = 64 << 10;
const LINE_BYTES: usize = 16 << 10;
const ERROR_BYTES: usize = 4096;

#[derive(Default)]
struct CapturedText {
    prefix: Vec<u8>,
    total_bytes: u64,
    truncated: bool,
}
impl CapturedText {
    fn append(&mut self, bytes: &[u8]) -> io::Result<()> {
        self.total_bytes = self
            .total_bytes
            .checked_add(u64::try_from(bytes.len()).map_err(|_| io::ErrorKind::OutOfMemory)?)
            .ok_or(io::ErrorKind::OutOfMemory)?;
        let keep = bytes.len().min(CAPTURE_BYTES - self.prefix.len());
        self.prefix.extend_from_slice(&bytes[..keep]);
        self.truncated |= keep != bytes.len();
        Ok(())
    }
    fn utf8_prefix(&self) -> &str {
        match std::str::from_utf8(&self.prefix) {
            Ok(text) => text,
            Err(error) => std::str::from_utf8(&self.prefix[..error.valid_up_to()])
                .expect("validated UTF-8 prefix"),
        }
    }
}

/// 원 오류와 bounded 출력 prefix를 caller가 마지막까지 보존한다.
/// 어떤 bool도 RootAdmission 또는 물리 완료 권한으로 사용하지 않는다.
#[derive(Default)]
pub(super) struct FiniteTrialObservation {
    protocol: CapturedText,
    diagnostics: CapturedText,
    pub(super) bestmove: Option<String>,
    pub(super) bestmove_count: u64,
    pub(super) uciok: bool,
    pub(super) readyok: bool,
    pub(super) maximum_wall_reached: bool,
    pub(super) stop_sent: bool,
    pub(super) quit_sent: bool,
    pub(super) transport_aborted: bool,
    pub(super) watchdog_joined: bool,
    pub(super) engine_error: Option<EngineError>,
    pub(super) engine_panic: Option<Box<dyn Any + Send>>,
    pub(super) watchdog_spawn_error: Option<io::Error>,
    pub(super) watchdog_failed: bool,
    pub(super) input_rejected: Option<&'static str>,
}
impl FiniteTrialObservation {
    pub(super) fn protocol_utf8_prefix(&self) -> &str {
        self.protocol.utf8_prefix()
    }
    pub(super) fn diagnostics_utf8_prefix(&self) -> &str {
        self.diagnostics.utf8_prefix()
    }
    pub(super) fn protocol_bytes(&self) -> u64 {
        self.protocol.total_bytes
    }
    pub(super) fn diagnostics_bytes(&self) -> u64 {
        self.diagnostics.total_bytes
    }
    pub(super) fn protocol_truncated(&self) -> bool {
        self.protocol.truncated
    }
    pub(super) fn diagnostics_truncated(&self) -> bool {
        self.diagnostics.truncated
    }
    /// Bounded original observations only; no text is promoted to an outcome.
    pub(super) fn diagnostic_value(&self) -> serde_json::Value {
        serde_json::json!({
            "protocol_utf8_prefix":self.protocol_utf8_prefix(),
            "diagnostics_utf8_prefix":self.diagnostics_utf8_prefix(),
            "protocol_bytes":self.protocol_bytes(),
            "diagnostics_bytes":self.diagnostics_bytes(),
            "protocol_truncated":self.protocol_truncated(),
            "diagnostics_truncated":self.diagnostics_truncated(),
            "bestmove":self.bestmove,"bestmove_count":self.bestmove_count,
            "uciok":self.uciok,"readyok":self.readyok,
            "maximum_wall_reached":self.maximum_wall_reached,
            "stop_sent":self.stop_sent,"quit_sent":self.quit_sent,
            "transport_aborted":self.transport_aborted,
            "watchdog_joined":self.watchdog_joined,
            "watchdog_failed":self.watchdog_failed,
            "watchdog_spawn_error":self.watchdog_spawn_error.as_ref().map(bounded_error),
            "input_rejected":self.input_rejected,
            "engine_error":self.engine_error.as_ref().map(bounded_error),
            "engine_error_retained":self.engine_error.is_some(),
            "engine_panic_retained":self.engine_panic.is_some()
        })
    }
}

#[derive(Default)]
struct WatchState {
    bestmove: bool,
    engine_returned: bool,
    abort_transport: bool,
}
type WatchControl = Arc<(Mutex<WatchState>, Condvar)>;

struct ProtocolOutput {
    original: Arc<Mutex<FiniteTrialObservation>>,
    control: WatchControl,
    line: Vec<u8>,
}
impl ProtocolOutput {
    fn check_transport(&self) -> io::Result<()> {
        if self
            .control
            .0
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .abort_transport
        {
            Err(io::Error::new(
                io::ErrorKind::TimedOut,
                "finite UCI control delivery expired",
            ))
        } else {
            Ok(())
        }
    }
}
impl Write for ProtocolOutput {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.check_transport()?;
        let mut observation = self.original.lock().unwrap_or_else(|p| p.into_inner());
        observation.protocol.append(bytes)?;
        let mut bestmove = false;
        for &byte in bytes {
            if byte != b'\n' {
                if self.line.len() >= LINE_BYTES {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "bounded UCI output line exceeded",
                    ));
                }
                self.line.push(byte);
                continue;
            }
            let text = std::str::from_utf8(&self.line)
                .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "UCI output is not UTF-8"))?
                .trim_end_matches('\r');
            if text == "uciok" {
                observation.uciok = true;
            } else if text == "readyok" {
                observation.readyok = true;
            } else if text.split_whitespace().next() == Some("bestmove") {
                observation.bestmove_count = observation
                    .bestmove_count
                    .checked_add(1)
                    .ok_or(io::ErrorKind::InvalidData)?;
                if observation.bestmove_count != 1 {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "duplicate actual UCI bestmove",
                    ));
                }
                observation.bestmove = Some(text.to_owned());
                bestmove = true;
            }
            self.line.clear();
        }
        drop(observation);
        if bestmove {
            self.control
                .0
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .bestmove = true;
            self.control.1.notify_all();
        }
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        self.check_transport()
    }
}
struct DiagnosticOutput(Arc<Mutex<FiniteTrialObservation>>);
impl Write for DiagnosticOutput {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.0
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .diagnostics
            .append(bytes)?;
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

fn watchdog(
    events: SyncSender<Event>,
    control: WatchControl,
    original: Arc<Mutex<FiniteTrialObservation>>,
    wall_until: Instant,
    delivery_until: Instant,
) {
    let mut state = control.0.lock().unwrap_or_else(|p| p.into_inner());
    while !state.bestmove && !state.engine_returned && Instant::now() < wall_until {
        state = control
            .1
            .wait_timeout(state, wall_until.saturating_duration_since(Instant::now()))
            .unwrap_or_else(|p| p.into_inner())
            .0;
    }
    if state.engine_returned {
        return;
    }
    let expired = !state.bestmove;
    drop(state);
    if expired {
        original
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .maximum_wall_reached = true;
    }
    let commands: &[&str] = if expired {
        &["stop", "quit"]
    } else {
        &["quit"]
    };
    for &command in commands {
        let mut event = Event::Line(command.to_owned());
        loop {
            if control
                .0
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .engine_returned
            {
                return;
            }
            match events.try_send(event) {
                Ok(()) => {
                    let mut observation = original.lock().unwrap_or_else(|p| p.into_inner());
                    if command == "stop" {
                        observation.stop_sent = true;
                    } else {
                        observation.quit_sent = true;
                    }
                    break;
                }
                Err(TrySendError::Disconnected(_)) => return,
                Err(TrySendError::Full(returned)) => event = returned,
            }
            if Instant::now() >= delivery_until {
                original
                    .lock()
                    .unwrap_or_else(|p| p.into_inner())
                    .transport_aborted = true;
                control
                    .0
                    .lock()
                    .unwrap_or_else(|p| p.into_inner())
                    .abort_transport = true;
                // A full event queue already has a wakeup. If it became empty,
                // this ordinary diagnostic event wakes the same output guard.
                let _ = events.try_send(Event::RejectedInput {
                    code: "FiniteTrialControlExpired",
                    message: String::new(),
                });
                return;
            }
            let state = control.0.lock().unwrap_or_else(|p| p.into_inner());
            drop(
                control
                    .1
                    .wait_timeout(state, Duration::from_millis(1))
                    .unwrap_or_else(|p| p.into_inner()),
            );
        }
    }
}

struct BoundedDisplay(String);
impl fmt::Write for BoundedDisplay {
    fn write_str(&mut self, text: &str) -> fmt::Result {
        let mut count = text.len().min(ERROR_BYTES - self.0.len());
        while !text.is_char_boundary(count) {
            count -= 1;
        }
        self.0.push_str(&text[..count]);
        if count != text.len() {
            Err(fmt::Error)
        } else {
            Ok(())
        }
    }
}
pub(super) fn bounded_error(error: &impl fmt::Display) -> String {
    let mut text = BoundedDisplay(String::new());
    let _ = write!(&mut text, "{error}");
    text.0
}

/// 실제 bestmove 뒤 quit한다. watchdog 종료도 stop/quit와 원 Engine cleanup을
/// 거치며, 하위 launch wrapper가 actual Product physical-close 관측을 확인한다.
#[allow(clippy::too_many_arguments)] // 원 Product/clock/owner/settings와 한 trial의 명시 입력이다.
pub(super) fn run_finite_collection_trial(
    product: RoventNativeProductDriver,
    clock: ProcessClock,
    owners: Arc<OwnerRegistry>,
    settings: EngineSettings,
    position_command: String,
    go_command: String,
    maximum_wall: Duration,
    original: Arc<Mutex<FiniteTrialObservation>>,
) -> Result<(), RoventLaunchError> {
    let start = Instant::now();
    let window = start.checked_add(maximum_wall).and_then(|wall| {
        wall.checked_add(settings.shutdown_limit)
            .map(|cleanup| (wall, cleanup))
    });
    let rejected =
        if maximum_wall.is_zero() || settings.shutdown_limit.is_zero() || window.is_none() {
            Some("finite trial wall/cleanup window is invalid")
        } else if !matches!(
            rz_uci::parse(&position_command, settings.parser),
            Ok(Command::Position(_))
        ) || go_command.len() > settings.parser.max_line_bytes
            // The actual product parser owns its registered node domain,
            // including nodes 0. Do not apply the legacy go parser here.
            || go_command.split_whitespace().next() != Some("go")
            || position_command.contains(['\r', '\n', '\0'])
            || go_command.contains(['\r', '\n', '\0'])
        {
            Some("finite trial requires one bounded original position and go command")
        } else {
            None
        };
    original
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .input_rejected = rejected;
    let (events, receiver) = engine::event_channel();
    let control = Arc::new((Mutex::new(WatchState::default()), Condvar::new()));
    let timer = if rejected.is_none() {
        let timer_events = events.clone();
        let timer_control = Arc::clone(&control);
        let timer_original = Arc::clone(&original);
        let (wall, cleanup) = window.expect("checked finite watchdog window");
        match thread::Builder::new()
            .name("rovent-cohort-watchdog".into())
            .spawn(move || watchdog(timer_events, timer_control, timer_original, wall, cleanup))
        {
            Ok(thread) => Some(thread),
            Err(error) => {
                original
                    .lock()
                    .unwrap_or_else(|p| p.into_inner())
                    .watchdog_spawn_error = Some(error);
                None
            }
        }
    } else {
        None
    };
    let commands = if timer.is_some() {
        vec![
            "uci".into(),
            "ucinewgame".into(),
            "isready".into(),
            position_command,
            go_command,
        ]
    } else {
        vec!["quit".into()]
    };
    for command in commands {
        if events.try_send(Event::Line(command)).is_err() {
            original
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .watchdog_failed = true;
            break;
        }
    }
    let mut protocol = ProtocolOutput {
        original: Arc::clone(&original),
        control: Arc::clone(&control),
        line: Vec::new(),
    };
    let mut diagnostics = DiagnosticOutput(Arc::clone(&original));
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let process = EngineProcess::with_search_driver(Arc::new(product), owners, clock);
        engine::serve(
            receiver,
            events,
            &mut protocol,
            &mut diagnostics,
            process,
            settings,
        )
    }));
    control
        .0
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .engine_returned = true;
    control.1.notify_all();
    if let Some(timer) = timer {
        let joined = timer.join().is_ok();
        let mut observation = original.lock().unwrap_or_else(|p| p.into_inner());
        observation.watchdog_joined = joined;
        observation.watchdog_failed |= !joined;
    }
    let mut observation = original.lock().unwrap_or_else(|p| p.into_inner());
    let detail = match result {
        Err(payload) => {
            observation.engine_panic = Some(payload);
            Some("actual finite UCI engine panicked".into())
        }
        Ok(Err(error)) => {
            observation.engine_error = Some(error);
            Some(bounded_error(
                observation
                    .engine_error
                    .as_ref()
                    .expect("original engine error retained"),
            ))
        }
        Ok(Ok(())) if rejected.is_some() => rejected.map(str::to_owned),
        Ok(Ok(())) if observation.watchdog_spawn_error.is_some() || observation.watchdog_failed => {
            Some("finite trial watchdog did not start or join normally".into())
        }
        Ok(Ok(())) if observation.maximum_wall_reached || observation.transport_aborted => {
            Some("finite trial maximum wall reached before original bestmove".into())
        }
        Ok(Ok(()))
            if !observation.uciok || !observation.readyok || observation.bestmove_count != 1 =>
        {
            Some("finite trial has no unique actual bestmove and UCI readiness".into())
        }
        Ok(Ok(())) => None,
    };
    match detail {
        Some(detail) => Err(RoventLaunchError {
            stage: "FiniteCollectionTrial",
            detail,
        }),
        None => Ok(()),
    }
}

#[cfg(test)]
mod diagnostic_tests {
    use super::*;
    #[test]
    fn original_diagnostic_projection_retains_failure_and_truncation() {
        let mut original = FiniteTrialObservation {
            engine_error: Some(EngineError::DrainTimeout),
            input_rejected: Some("original position/go rejected"),
            ..FiniteTrialObservation::default()
        };
        original
            .diagnostics
            .append(b"RoventNnuecManagerFailure: original typed error\n")
            .unwrap();
        original.diagnostics.append(&[b'x'; CAPTURE_BYTES]).unwrap();
        let value = original.diagnostic_value();
        assert!(
            value["diagnostics_utf8_prefix"]
                .as_str()
                .unwrap()
                .starts_with("RoventNnuecManagerFailure: original typed error\n")
        );
        assert_eq!(value["diagnostics_truncated"], true);
        assert!(value["diagnostics_bytes"].as_u64().unwrap() > CAPTURE_BYTES as u64);
        assert_eq!(value["engine_error"], "DrainTimeout");
        assert_eq!(value["engine_error_retained"], true);
        assert!(matches!(
            original.engine_error.as_ref(),
            Some(EngineError::DrainTimeout)
        ));
        assert_eq!(value["bestmove_count"], 0);
        assert!(
            value.get("preferred").is_none() && value.get("physical_completion_attested").is_none()
        );
    }
    #[test]
    fn original_empty_diagnostics_stays_empty_without_an_invented_failure() {
        let value = FiniteTrialObservation::default().diagnostic_value();
        assert_eq!(value["diagnostics_utf8_prefix"], "");
        assert_eq!(value["diagnostics_bytes"], 0);
        assert!(value["engine_error"].is_null() && value["input_rejected"].is_null());
        assert_eq!(value["engine_error_retained"], false);
        assert_eq!(value["engine_panic_retained"], false);
    }
}
