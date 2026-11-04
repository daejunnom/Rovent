use rz_uci::runner::async_events::{
    AsyncControl, AsyncServeFailure, AsyncServeStatus, serve_events_with_async_handler,
};
use rz_uci::runner::async_output::{
    AsyncOutputOwner, OutputBatch, SubmitError, encode_event_output,
};
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
    let mut session = session();
    let mut shutdown_seen = false;
    let mut exit = serve_events_with_async_handler(
        &mut session,
        &events,
        &owner,
        4096,
        Duration::from_millis(1),
        || {
            let count = calls.get();
            calls.set(count + 1);
            if count >= 2 {
                AsyncControl::ReturnPending
            } else {
                AsyncControl::Continue
            }
        },
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
                shutdown_seen = true;
            }
            Ok::<_, &'static str>(())
        },
    );
    {
        let (mutex, condition) = &*gate;
        mutex.lock().unwrap().1 = true;
        condition.notify_all();
    }
    assert_eq!(exit.status, AsyncServeStatus::Yielded);
    assert!(session.is_closed() && shutdown_seen);
    assert_eq!(exit.pending.len(), 2);
    assert_eq!(exit.pending[0].original.protocol.last().unwrap(), "uciok");
    assert!(exit.pending[0].encoded.is_some());
    assert_eq!(exit.pending[1].original.protocol, vec!["close-evidence"]);
    assert_eq!(exit.pending[1].original.diagnostics[0].message, "retained");
    let until = Instant::now() + Duration::from_secs(3);
    while let Some(mut item) = exit.pending.pop_front() {
        let mut batch = item
            .encoded
            .take()
            .unwrap_or_else(|| encode_event_output(&item.original, 4096).unwrap());
        loop {
            match owner.try_submit(batch) {
                Ok(()) => break,
                Err(SubmitError::Full(original)) if Instant::now() < until => {
                    batch = original;
                    thread::yield_now();
                }
                _ => panic!("fixture could not drain pending output"),
            }
        }
    }
    owner.request_close();
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
fn control_panic_preserves_close_dispatch_failure_and_output() {
    let owner = match AsyncOutputOwner::start(Vec::<u8>::new(), Vec::<u8>::new(), 1, 4096) {
        Ok(owner) => owner,
        Err(_) => panic!("cannot create fixture output owner"),
    };
    let (_sender, events) = mpsc::sync_channel(1);
    let mut session = session();
    let exit = serve_events_with_async_handler(
        &mut session,
        &events,
        &owner,
        4096,
        Duration::from_millis(1),
        || panic!("fixture control panic"),
        handle_event,
        |effect| -> Result<(), &'static str> {
            if matches!(effect, Effect::Shutdown) {
                Err("fixture shutdown failure")
            } else {
                Ok(())
            }
        },
    );
    assert_eq!(exit.status, AsyncServeStatus::Failed);
    assert!(matches!(
        exit.failure,
        Some(AsyncServeFailure::ControlPanic(_))
    ));
    assert_eq!(exit.cleanup_failures.len(), 1);
    assert_eq!(exit.pending.len(), 1);
    assert!(session.is_closed());
    finish_empty(owner);
}

#[test]
fn encoding_limit_preserves_original_and_closes_original_session() {
    let owner = match AsyncOutputOwner::start(Vec::<u8>::new(), Vec::<u8>::new(), 1, 4096) {
        Ok(owner) => owner,
        Err(_) => panic!("cannot create fixture output owner"),
    };
    let (sender, events) = mpsc::sync_channel(1);
    sender.send(Event::Line("uci".into())).unwrap();
    let mut session = session();
    let exit = serve_events_with_async_handler(
        &mut session,
        &events,
        &owner,
        1,
        Duration::from_millis(1),
        || AsyncControl::Continue,
        handle_event,
        |_| Ok::<_, &'static str>(()),
    );
    assert_eq!(exit.status, AsyncServeStatus::Failed);
    assert!(matches!(exit.failure, Some(AsyncServeFailure::Encoding(_))));
    assert_eq!(exit.pending.len(), 2);
    assert_eq!(exit.pending[0].original.protocol.last().unwrap(), "uciok");
    assert!(session.is_closed());
    finish_empty(owner);
}
