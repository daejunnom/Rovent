use rz_uci::runner::async_output::{
    AsyncOutputOwner, OutputBatch, OutputFailure, OutputPhase, OutputRecovery, SubmitError,
};
use std::io::{self, Write};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;
use std::time::{Duration, Instant};

fn batch(protocol: &[u8], diagnostics: &[u8]) -> OutputBatch {
    OutputBatch {
        protocol: protocol.to_vec(),
        diagnostics: diagnostics.to_vec(),
    }
}

fn recover<O: Write + Send + 'static, D: Write + Send + 'static>(
    mut owner: AsyncOutputOwner<O, D>,
) -> OutputRecovery<O, D> {
    let until = Instant::now() + Duration::from_secs(3);
    loop {
        owner = match owner.try_recover() {
            Ok(Ok(result)) => return result,
            Ok(Err(_)) => panic!("output worker panicked outside its retained failure boundary"),
            Err(owner) if Instant::now() < until => {
                thread::yield_now();
                owner
            }
            Err(_) => panic!("output worker did not finish"),
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
fn blocked_writer_retains_full_batch_and_allows_cancel_and_late_drain() {
    let gate = Arc::new((Mutex::new((false, false)), Condvar::new()));
    let writer = GatedWriter {
        gate: Arc::clone(&gate),
        bytes: Vec::new(),
    };
    let owner = match AsyncOutputOwner::start(writer, Vec::<u8>::new(), 1, 128) {
        Ok(owner) => owner,
        Err(_) => panic!("bounded output owner could not start"),
    };
    owner.try_submit(batch(b"first\n", b"diag\n")).unwrap();
    {
        let (mutex, condition) = &*gate;
        let state = mutex.lock().unwrap();
        let (state, timeout) = condition
            .wait_timeout_while(state, Duration::from_secs(2), |s| !s.0)
            .unwrap();
        assert!(!timeout.timed_out() && state.0);
    }
    owner.try_submit(batch(b"second\n", b"")).unwrap();
    match owner.try_submit(batch(b"third\n", b"")) {
        Err(SubmitError::Full(original)) => assert_eq!(original, batch(b"third\n", b"")),
        _ => panic!("full queue must return the original batch"),
    }
    // This short owner operation can run even while the real writer blocks.
    owner.request_close();
    let observation = owner.observation();
    assert!(observation.close_requested);
    assert_eq!(observation.phase, Some(OutputPhase::ProtocolWrite));
    assert!(observation.call_completion_unknown);
    {
        let (mutex, condition) = &*gate;
        mutex.lock().unwrap().1 = true;
        condition.notify_all();
    }
    let finished = recover(owner);
    assert!(finished.failure.is_none());
    assert!(finished.close_requested);
    assert!(finished.partial.is_none());
    assert!(finished.queued.try_recv().is_err());
    assert_eq!(finished.protocol.bytes, b"first\nsecond\n");
    assert_eq!(finished.diagnostics, b"diag\n");
}

struct PartialFailure {
    bytes: Vec<u8>,
}
impl Write for PartialFailure {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if !self.bytes.is_empty() {
            return Err(io::Error::other("fixture write failure"));
        }
        let count = bytes.len().min(2);
        self.bytes.extend_from_slice(&bytes[..count]);
        Ok(count)
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[test]
fn partial_write_failure_returns_original_writer_and_exact_remaining_batch() {
    let owner = match AsyncOutputOwner::start(
        PartialFailure { bytes: Vec::new() },
        Vec::<u8>::new(),
        1,
        128,
    ) {
        Ok(owner) => owner,
        Err(_) => panic!("bounded output owner could not start"),
    };
    owner.try_submit(batch(b"hello", b"diag")).unwrap();
    let finished = recover(owner);
    assert!(matches!(finished.failure, Some(OutputFailure::Io(_))));
    assert_eq!(finished.protocol.bytes, b"he");
    let partial = finished.partial.unwrap();
    assert_eq!(partial.phase, OutputPhase::ProtocolWrite);
    assert_eq!(partial.offset, 2);
    assert!(!partial.call_completion_unknown);
    assert_eq!(partial.batch, batch(b"hello", b"diag"));
}

struct PanicWriter;
impl Write for PanicWriter {
    fn write(&mut self, _: &[u8]) -> io::Result<usize> {
        panic!("fixture writer panic")
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[test]
fn panicked_write_retains_batch_and_marks_call_progress_unknown() {
    let owner = match AsyncOutputOwner::start(PanicWriter, Vec::<u8>::new(), 1, 128) {
        Ok(owner) => owner,
        Err(_) => panic!("bounded output owner could not start"),
    };
    owner.try_submit(batch(b"hello", b"")).unwrap();
    let finished = recover(owner);
    assert!(matches!(finished.failure, Some(OutputFailure::Panic(_))));
    let partial = finished.partial.unwrap();
    assert_eq!(partial.offset, 0);
    assert!(partial.call_completion_unknown);
    assert_eq!(partial.batch, batch(b"hello", b""));
}
