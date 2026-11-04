//! Best-effort UCI quit request within a finite supervisor window.
//!
//! Enqueueing a command is neither a UCI acknowledgement nor physical
//! completion. A full queue, blocked service, or stuck native operation must
//! remain visible to the original supervisor and cannot authorize process exit.

use rz_uci::Event;
use std::{
    io,
    sync::mpsc::{self, RecvTimeoutError, SyncSender, TrySendError},
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum QuitRequestObservation {
    Cancelled,
    Enqueued,
    QueueFullUntilDeadline,
    ConsumerClosed,
}

/// The caller owns this worker and must join it before settling the session.
/// Dropping this value only asks the worker to stop; it is not join evidence.
pub(super) struct TimedQuitRequest {
    cancel: mpsc::Sender<()>,
    worker: Option<JoinHandle<QuitRequestObservation>>,
}

impl TimedQuitRequest {
    pub(super) fn spawn(
        events: SyncSender<Event>,
        request_at: Instant,
        enqueue_until: Instant,
    ) -> io::Result<Self> {
        if request_at > enqueue_until {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "quit window inverted",
            ));
        }
        let (cancel, cancelled) = mpsc::channel();
        let worker = thread::Builder::new()
            .name("lc0-timed-quit".into())
            .spawn(move || {
                match cancelled.recv_timeout(request_at.saturating_duration_since(Instant::now())) {
                    Ok(()) | Err(RecvTimeoutError::Disconnected) => {
                        return QuitRequestObservation::Cancelled;
                    }
                    Err(RecvTimeoutError::Timeout) => {}
                }
                loop {
                    if Instant::now() > enqueue_until {
                        return QuitRequestObservation::QueueFullUntilDeadline;
                    }
                    match events.try_send(Event::Line("quit".into())) {
                        Ok(()) => return QuitRequestObservation::Enqueued,
                        Err(TrySendError::Disconnected(_)) => {
                            return QuitRequestObservation::ConsumerClosed;
                        }
                        Err(TrySendError::Full(_)) => {}
                    }
                    let wait = enqueue_until
                        .saturating_duration_since(Instant::now())
                        .min(Duration::from_millis(10));
                    match cancelled.recv_timeout(wait) {
                        Ok(()) | Err(RecvTimeoutError::Disconnected) => {
                            return QuitRequestObservation::Cancelled;
                        }
                        Err(RecvTimeoutError::Timeout) => {}
                    }
                }
            })?;
        Ok(Self {
            cancel,
            worker: Some(worker),
        })
    }

    pub(super) fn cancel_and_join(mut self) -> thread::Result<QuitRequestObservation> {
        let _ = self.cancel.send(());
        self.worker.take().expect("timed quit worker exists").join()
    }
}

impl Drop for TimedQuitRequest {
    fn drop(&mut self) {
        let _ = self.cancel.send(());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timed_quit_can_be_cancelled_before_its_window() {
        let (events, receiver) = mpsc::sync_channel(1);
        let start = Instant::now() + Duration::from_secs(1);
        let worker =
            TimedQuitRequest::spawn(events, start, start + Duration::from_secs(1)).unwrap();
        assert_eq!(
            worker.cancel_and_join().unwrap(),
            QuitRequestObservation::Cancelled
        );
        assert!(receiver.try_recv().is_err());
    }

    #[test]
    fn queue_acceptance_does_not_claim_ack_or_completion() {
        let (events, receiver) = mpsc::sync_channel(1);
        let now = Instant::now();
        let mut worker =
            TimedQuitRequest::spawn(events, now, now + Duration::from_secs(1)).unwrap();
        assert!(
            matches!(receiver.recv_timeout(Duration::from_secs(1)).unwrap(), Event::Line(line) if line == "quit")
        );
        assert_eq!(
            worker.worker.take().unwrap().join().unwrap(),
            QuitRequestObservation::Enqueued
        );
    }

    #[test]
    fn full_queue_does_not_get_reported_as_a_stop() {
        let (events, _receiver) = mpsc::sync_channel(1);
        events.send(Event::Line("occupied".into())).unwrap();
        let now = Instant::now();
        let mut worker =
            TimedQuitRequest::spawn(events, now, now + Duration::from_millis(25)).unwrap();
        assert_eq!(
            worker.worker.take().unwrap().join().unwrap(),
            QuitRequestObservation::QueueFullUntilDeadline
        );
    }
}
