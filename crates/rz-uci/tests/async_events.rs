use rz_uci::runner::async_events::{
    AsyncControl, AsyncServeDriver, AsyncServeFailure, AsyncServeStatus, OutputRecoveryPoll,
};
use rz_uci::runner::async_output::{AsyncOutputOwner, OutputBatch, OutputFailure};
use rz_uci::{
    Diagnostic, Effect, EngineIdentity, Event, ParserLimits, PositionPort, PositionSpec,
    PreparedPosition, Session, handle_event,
};
use std::cell::Cell;
use std::io::{self, Write};
use std::sync::{Arc, Condvar, Mutex, mpsc};
use std::thread;
use std::time::{Duration, Instant};

struct Fixture;
impl PositionPort for Fixture {
    type Snapshot = ();
    type Error = &'static str;
    fn prepare(&self, _: &PositionSpec) -> Result<PreparedPosition<()>, Self::Error> {
        Ok(PreparedPosition {
            snapshot: (),
            legal_moves: vec!["e2e4".into()],
            exact_terminal: false,
        })
    }
}

fn session() -> Session<Fixture> {
    Session::new(
        Fixture,
        EngineIdentity {
            name: "Fixture".into(),
            author: "B".into(),
        },
        vec![],
        ParserLimits::default(),
    )
    .unwrap()
}

fn finish_empty(mut owner: AsyncOutputOwner<Vec<u8>, Vec<u8>>) {
    owner.request_close();
    let until = Instant::now() + Duration::from_secs(3);
    loop {
        owner = match owner.try_recover() {
            Ok(Ok(recovered)) => {
                assert!(recovered.failure.is_none());
                break;
            }
            Ok(Err(_)) => panic!("fixture output worker panicked"),
            Err(owner) if Instant::now() < until => {
                thread::yield_now();
                owner
            }
            Err(_) => panic!("fixture output worker did not finish"),
        };
    }
}

struct GatedWriter {
    gate: Arc<(Mutex<(bool, bool)>, Condvar)>,
    bytes: Vec<u8>,
}
impl Write for GatedWriter {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        let (mutex, condition) = &*self.gate;
        let mut state = mutex.lock().unwrap();
        state.0 = true;
        condition.notify_all();
        while !state.1 {
            state = condition.wait(state).unwrap();
        }
        self.bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[test]
fn full_output_retains_current_and_close_output_in_original_order() {
    let gate = Arc::new((Mutex::new((false, false)), Condvar::new()));
    let writer = GatedWriter {
        gate: Arc::clone(&gate),
        bytes: Vec::new(),
    };
    let owner = match AsyncOutputOwner::start(writer, Vec::<u8>::new(), 1, 4096) {
        Ok(owner) => owner,
        Err(_) => panic!("cannot create fixture output owner"),
    };
    owner
        .try_submit(OutputBatch {
            protocol: b"first\n".to_vec(),
            diagnostics: vec![],
        })
        .unwrap();
    {
        let (mutex, condition) = &*gate;
        let state = mutex.lock().unwrap();
        let (state, timeout) = condition
            .wait_timeout_while(state, Duration::from_secs(2), |s| !s.0)
            .unwrap();
        assert!(!timeout.timed_out() && state.0);
    }
    owner
        .try_submit(OutputBatch {
            protocol: b"second\n".to_vec(),
            diagnostics: vec![],
        })
        .unwrap();
    let (sender, events) = mpsc::sync_channel(2);
    sender.send(Event::Line("uci".into())).unwrap();
    let calls = Cell::new(0);
    let shutdown_seen = Cell::new(false);
    let mut driver = AsyncServeDriver::new(
        session(),
        events,
        owner,
        4096,
        Duration::from_millis(1),
        |session, event| {
            if matches!(event, Event::EndOfInput) {
                let mut result = handle_event(session, event);
                result.protocol.push("close-evidence".into());
                result.diagnostics.push(Diagnostic {
                    code: "Close",
                    message: "retained".into(),
                });
                result
            } else {
                handle_event(session, event)
            }
        },
        |effect| {
            if matches!(effect, Effect::Shutdown) {
                shutdown_seen.set(true);
            }
            Ok::<_, &'static str>(())
        },
    );
    let exit = driver.drive(|| {
        let count = calls.get();
        calls.set(count + 1);
        if count >= 2 {
            AsyncControl::ReturnPending
        } else {
            AsyncControl::Continue
        }
    });
    {
        let (mutex, condition) = &*gate;
        mutex.lock().unwrap().1 = true;
        condition.notify_all();
    }
    assert_eq!(exit.status, AsyncServeStatus::Yielded);
    assert!(driver.session_closed() && shutdown_seen.get());
    assert_eq!(driver.pending().len(), 2);
    assert_eq!(
        driver.pending()[0].original.protocol.last().unwrap(),
        "uciok"
    );
    assert!(driver.pending()[0].encoded.is_some());
    assert_eq!(
        driver.pending()[1].original.protocol,
        vec!["close-evidence"]
    );
    assert_eq!(
        driver.pending()[1].original.diagnostics[0].message,
        "retained"
    );
    let until = Instant::now() + Duration::from_secs(3);
    let resumed = driver.drive(|| AsyncControl::Continue);
    assert_eq!(resumed.status, AsyncServeStatus::OutputQueued);
    assert!(driver.pending().is_empty());
    let parts = driver.into_parts();
    assert!(parts.pending.is_empty());
    let owner = parts.output.unwrap();
    let mut owner = owner;
    let result = loop {
        owner = match owner.try_recover() {
            Ok(Ok(result)) => break result,
            Ok(Err(_)) => panic!("fixture worker panic"),
            Err(owner) if Instant::now() < until => {
                thread::yield_now();
                owner
            }
            Err(_) => panic!("fixture worker did not drain"),
        };
    };
    assert!(result.failure.is_none());
    assert_eq!(
        String::from_utf8(result.protocol.bytes).unwrap(),
        "first\nsecond\nid name Fixture\nid author B\nuciok\nclose-evidence\n"
    );
    assert_eq!(
        String::from_utf8(result.diagnostics).unwrap(),
        "Close: retained\n"
    );
}

#[test]
fn control_panic_retains_session_without_implicit_shutdown_retry() {
    let owner = match AsyncOutputOwner::start(Vec::<u8>::new(), Vec::<u8>::new(), 1, 4096) {
        Ok(owner) => owner,
        Err(_) => panic!("cannot create fixture output owner"),
    };
    let (_sender, events) = mpsc::sync_channel(1);
    let mut driver = AsyncServeDriver::new(
        session(),
        events,
        owner,
        4096,
        Duration::from_millis(1),
        handle_event,
        |effect| -> Result<(), &'static str> {
            if matches!(effect, Effect::Shutdown) {
                Err("fixture shutdown failure")
            } else {
                Ok(())
            }
        },
    );
    let exit = driver.drive(|| panic!("fixture control panic"));
    assert_eq!(exit.status, AsyncServeStatus::Failed);
    assert!(matches!(
        exit.failure,
        Some(AsyncServeFailure::ControlPanic(_))
    ));
    assert!(exit.cleanup_failures.is_empty());
    assert!(driver.pending().is_empty());
    assert!(!driver.session_closed());
    let parts = driver.into_parts();
    assert!(parts.sealed);
    assert!(!parts.shutdown_requested);
    finish_empty(parts.output.unwrap());
}

#[test]
fn explicit_unknown_control_retains_session_without_reducer_or_dispatch() {
    let owner = match AsyncOutputOwner::start(Vec::<u8>::new(), Vec::<u8>::new(), 1, 4096) {
        Ok(owner) => owner,
        Err(_) => panic!("cannot create fixture output owner"),
    };
    let (_sender, events) = mpsc::sync_channel(1);
    let mut driver = AsyncServeDriver::new(
        session(),
        events,
        owner,
        4096,
        Duration::from_millis(1),
        |_, _| -> rz_uci::SessionResult<()> { panic!("reducer must not run") },
        |_| -> Result<(), &'static str> { panic!("dispatch must not run") },
    );
    let exit = driver.drive(|| AsyncControl::RetainWithoutClose);
    assert_eq!(exit.status, AsyncServeStatus::RetainedWithoutClose);
    assert!(driver.pending().is_empty());
    assert!(!driver.session_closed());
    let retry: rz_uci::runner::async_events::AsyncServeExit<&'static str> =
        driver.drive(|| AsyncControl::Continue);
    assert!(matches!(
        retry.failure,
        Some(AsyncServeFailure::OriginalStateSealed)
    ));
    let parts = driver.into_parts();
    assert!(parts.sealed);
    finish_empty(parts.output.unwrap());
}

#[test]
fn cooperative_yield_keeps_live_session_and_resumes_same_driver() {
    let owner = match AsyncOutputOwner::start(Vec::<u8>::new(), Vec::<u8>::new(), 2, 4096) {
        Ok(owner) => owner,
        Err(_) => panic!("cannot create fixture output owner"),
    };
    let (sender, events) = mpsc::sync_channel(2);
    sender.send(Event::Line("uci".into())).unwrap();
    sender.send(Event::Line("quit".into())).unwrap();
    let mut driver = AsyncServeDriver::new(
        session(),
        events,
        owner,
        4096,
        Duration::from_millis(1),
        handle_event,
        |_| Ok::<_, &'static str>(()),
    );
    assert_eq!(
        driver.drive(|| AsyncControl::Yield).status,
        AsyncServeStatus::Yielded
    );
    assert!(!driver.session_closed());
    assert!(!driver.shutdown_requested());
    assert!(driver.pending().is_empty());
    assert_eq!(
        driver.drive(|| AsyncControl::Continue).status,
        AsyncServeStatus::OutputQueued
    );
    assert!(driver.session_closed());
    let parts = driver.into_parts();
    assert!(parts.sealed);
    finish_empty(parts.output.unwrap());
}

#[test]
fn deferred_owner_close_is_not_forced_or_reissued_after_yield() {
    let owner = match AsyncOutputOwner::start(Vec::<u8>::new(), Vec::<u8>::new(), 1, 4096) {
        Ok(owner) => owner,
        Err(_) => panic!("cannot create fixture output owner"),
    };
    let (_sender, events) = mpsc::sync_channel(1);
    let eof_count = Cell::new(0);
    let mut driver = AsyncServeDriver::new(
        session(),
        events,
        owner,
        4096,
        Duration::from_millis(1),
        |session, event| {
            if matches!(event, Event::EndOfInput) {
                eof_count.set(eof_count.get() + 1);
                // Stand in for an Owner waiting on its original physical fence.
                rz_uci::SessionResult::default()
            } else {
                handle_event(session, event)
            }
        },
        |_| Ok::<_, &'static str>(()),
    );
    assert_eq!(
        driver.drive(|| AsyncControl::ReturnPending).status,
        AsyncServeStatus::Yielded
    );
    assert!(driver.shutdown_requested());
    assert!(!driver.session_closed());
    assert_eq!(eof_count.get(), 1);
    assert_eq!(
        driver.drive(|| AsyncControl::ReturnPending).status,
        AsyncServeStatus::Yielded
    );
    assert!(!driver.session_closed());
    assert_eq!(eof_count.get(), 1);
    let parts = driver.into_parts();
    assert_eq!(parts.pending.len(), 1);
    assert!(parts.shutdown_requested);
    assert!(!parts.sealed);
    finish_empty(parts.output.unwrap());
}

#[test]
fn encoding_limit_preserves_original_and_closes_original_session() {
    let owner = match AsyncOutputOwner::start(Vec::<u8>::new(), Vec::<u8>::new(), 1, 4096) {
        Ok(owner) => owner,
        Err(_) => panic!("cannot create fixture output owner"),
    };
    let (sender, events) = mpsc::sync_channel(1);
    sender.send(Event::Line("uci".into())).unwrap();
    let mut driver = AsyncServeDriver::new(
        session(),
        events,
        owner,
        1,
        Duration::from_millis(1),
        handle_event,
        |_| Ok::<_, &'static str>(()),
    );
    let exit = driver.drive(|| AsyncControl::Continue);
    assert_eq!(exit.status, AsyncServeStatus::Failed);
    assert!(matches!(exit.failure, Some(AsyncServeFailure::Encoding(_))));
    assert_eq!(driver.pending().len(), 2);
    assert_eq!(
        driver.pending()[0].original.protocol.last().unwrap(),
        "uciok"
    );
    assert!(driver.session_closed());
    let parts = driver.into_parts();
    assert!(parts.sealed);
    finish_empty(parts.output.unwrap());
}

#[test]
fn output_failure_does_not_force_close_while_owner_defers_eof() {
    let owner = match AsyncOutputOwner::start(Vec::<u8>::new(), Vec::<u8>::new(), 1, 4096) {
        Ok(owner) => owner,
        Err(_) => panic!("cannot create fixture output owner"),
    };
    let (sender, events) = mpsc::sync_channel(1);
    sender.send(Event::Line("uci".into())).unwrap();
    let mut driver = AsyncServeDriver::new(
        session(),
        events,
        owner,
        1,
        Duration::from_millis(1),
        |session, event| {
            if matches!(event, Event::EndOfInput) {
                rz_uci::SessionResult::default()
            } else {
                handle_event(session, event)
            }
        },
        |_| Ok::<_, &'static str>(()),
    );
    let exit = driver.drive(|| AsyncControl::Continue);
    assert_eq!(exit.status, AsyncServeStatus::Failed);
    assert!(matches!(exit.failure, Some(AsyncServeFailure::Encoding(_))));
    assert!(!driver.session_closed());
    assert_eq!(driver.pending().len(), 2);
    let parts = driver.into_parts();
    assert!(parts.sealed);
    assert!(parts.shutdown_requested);
    finish_empty(parts.output.unwrap());
}

#[test]
fn nonblocking_writer_recovery_keeps_original_driver_when_pending() {
    let gate = Arc::new((Mutex::new((false, false)), Condvar::new()));
    let writer = GatedWriter {
        gate: Arc::clone(&gate),
        bytes: Vec::new(),
    };
    let owner = match AsyncOutputOwner::start(writer, Vec::<u8>::new(), 2, 4096) {
        Ok(owner) => owner,
        Err(_) => panic!("cannot create fixture output owner"),
    };
    owner
        .try_submit(OutputBatch {
            protocol: b"first".to_vec(),
            diagnostics: vec![],
        })
        .unwrap();
    {
        let (mutex, condition) = &*gate;
        let state = mutex.lock().unwrap();
        let (state, timeout) = condition
            .wait_timeout_while(state, Duration::from_secs(2), |s| !s.0)
            .unwrap();
        assert!(!timeout.timed_out() && state.0);
    }
    let (sender, events) = mpsc::sync_channel(1);
    sender.send(Event::Line("quit".into())).unwrap();
    let mut driver = AsyncServeDriver::new(
        session(),
        events,
        owner,
        4096,
        Duration::from_millis(1),
        handle_event,
        |_| Ok::<_, &'static str>(()),
    );
    assert!(matches!(
        driver.try_finish_output(),
        OutputRecoveryPoll::DriverNotSealed
    ));
    assert_eq!(
        driver.drive(|| AsyncControl::Continue).status,
        AsyncServeStatus::OutputQueued
    );
    assert!(matches!(
        driver.try_finish_output(),
        OutputRecoveryPoll::Pending
    ));
    assert!(driver.final_output_observation().is_none());
    assert!(driver.output().is_some());
    assert!(driver.session_closed());
    assert!(driver.shutdown_requested());
    assert!(driver.pending().is_empty());
    {
        let (mutex, condition) = &*gate;
        mutex.lock().unwrap().1 = true;
        condition.notify_all();
    }
    let until = Instant::now() + Duration::from_secs(3);
    let recovered = loop {
        match driver.try_finish_output() {
            OutputRecoveryPoll::Pending if Instant::now() < until => thread::yield_now(),
            OutputRecoveryPoll::Recovered(recovered) => break recovered,
            _ => panic!("fixture original writer did not finish"),
        }
    };
    assert!(Arc::ptr_eq(&recovered.protocol.gate, &gate));
    assert_eq!(recovered.protocol.bytes, b"first");
    assert!(recovered.failure.is_none());
    let final_observation = driver.final_output_observation().unwrap();
    assert!(final_observation.worker_finished);
    assert!(!final_observation.failed);
    assert_eq!(
        final_observation.accepted_batches,
        final_observation.completed_batches
    );
    assert!(driver.output().is_none());
    assert!(matches!(
        driver.try_finish_output(),
        OutputRecoveryPoll::AlreadyRecovered
    ));
    assert!(driver.session_closed());
    assert!(driver.pending().is_empty());
}

enum FailingWriter {
    Io,
    Panic,
}
impl Write for FailingWriter {
    fn write(&mut self, _: &[u8]) -> io::Result<usize> {
        match self {
            Self::Io => Err(io::Error::other("fixture output failure")),
            Self::Panic => panic!("fixture output panic"),
        }
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[test]
fn writer_recovery_keeps_io_and_panic_as_distinct_original_results() {
    for panic_expected in [false, true] {
        let writer = if panic_expected {
            FailingWriter::Panic
        } else {
            FailingWriter::Io
        };
        let owner = match AsyncOutputOwner::start(writer, Vec::<u8>::new(), 1, 4096) {
            Ok(owner) => owner,
            Err(_) => panic!("cannot create fixture output owner"),
        };
        owner
            .try_submit(OutputBatch {
                protocol: b"data".to_vec(),
                diagnostics: vec![],
            })
            .unwrap();
        let until = Instant::now() + Duration::from_secs(3);
        while !owner.observation().worker_finished && Instant::now() < until {
            thread::yield_now();
        }
        assert!(owner.observation().worker_finished);
        let (_sender, events) = mpsc::sync_channel(1);
        let mut driver = AsyncServeDriver::new(
            session(),
            events,
            owner,
            4096,
            Duration::from_millis(1),
            handle_event,
            |_| Ok::<_, &'static str>(()),
        );
        assert_eq!(
            driver.drive(|| AsyncControl::Continue).status,
            AsyncServeStatus::Failed
        );
        let recovered = loop {
            match driver.try_finish_output() {
                OutputRecoveryPoll::Pending if Instant::now() < until => thread::yield_now(),
                OutputRecoveryPoll::Recovered(recovered) => break recovered,
                _ => panic!("fixture writer recovery unavailable"),
            }
        };
        assert!(if panic_expected {
            matches!(recovered.failure, Some(OutputFailure::Panic(_)))
        } else {
            matches!(recovered.failure, Some(OutputFailure::Io(_)))
        });
        let final_observation = driver.final_output_observation().unwrap();
        assert!(final_observation.worker_finished);
        assert!(final_observation.failed);
        assert_eq!(recovered.partial.unwrap().batch.protocol, b"data");
        assert!(driver.output().is_none());
    }
}
