use crate::CompletionSignal;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, Weak};
use std::time::Duration;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ControlError {
    Closed,
    StaleSequence,
    RequestOutstanding,
    MismatchedAck,
    SequenceExhausted,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ControlSnapshot {
    pub cancel_sequence: u64,
    pub yield_sequence: u64,
    pub publish_sequence: u64,
    pub cancel_ack: u64,
    pub yield_ack: u64,
    pub publish_ack: u64,
    pub terminal: bool,
}
impl ControlSnapshot {
    pub fn is_cancelled(self) -> bool {
        self.cancel_sequence != 0
    }
    pub fn pending_yield(self) -> Option<u64> {
        (!self.terminal && !self.is_cancelled() && self.yield_sequence > self.yield_ack)
            .then_some(self.yield_sequence)
    }
    pub fn pending_publish(self) -> Option<u64> {
        (!self.terminal && !self.is_cancelled() && self.publish_sequence > self.publish_ack)
            .then_some(self.publish_sequence)
    }
}

#[derive(Debug)]
struct Controls<S> {
    scope: S,
    highest: AtomicU64,
    cancel: AtomicU64,
    yield_request: AtomicU64,
    publish: AtomicU64,
    cancel_ack: AtomicU64,
    yield_ack: AtomicU64,
    publish_ack: AtomicU64,
    // u64::MAX is never an admissible control sequence. It represents an
    // unsealed terminal snapshot, not an acknowledged cancellation.
    terminal_cancel_ack: AtomicU64,
    signal: CompletionSignal,
}

/// Immutable per-Job/owner scope. Construct a new mailbox for every new Job.
/// The coordinator is the single control writer; the owning worker is the
/// single ack writer. Atomic reads bypass the ordinary command queue. There is
/// at most one outstanding Yield and one PublishProgress per Job; ack facts
/// belong in the already-reserved slice report, not an unbounded ack queue.
#[derive(Debug)]
pub struct ControlMailbox<S>(Arc<Controls<S>>);
impl<S> Clone for ControlMailbox<S> {
    fn clone(&self) -> Self {
        Self(Arc::clone(&self.0))
    }
}
impl<S> ControlMailbox<S> {
    pub fn new(scope: S, signal: CompletionSignal) -> Self {
        Self(Arc::new(Controls {
            scope,
            highest: AtomicU64::new(0),
            cancel: AtomicU64::new(0),
            yield_request: AtomicU64::new(0),
            publish: AtomicU64::new(0),
            cancel_ack: AtomicU64::new(0),
            yield_ack: AtomicU64::new(0),
            publish_ack: AtomicU64::new(0),
            terminal_cancel_ack: AtomicU64::new(u64::MAX),
            signal,
        }))
    }
    pub fn scope(&self) -> &S {
        &self.0.scope
    }
    pub fn signal(&self) -> CompletionSignal {
        self.0.signal.clone()
    }
    pub fn snapshot(&self) -> ControlSnapshot {
        let live_cancel_ack = self.0.cancel_ack.load(Ordering::Acquire);
        let sealed_cancel_ack = self.0.terminal_cancel_ack.load(Ordering::Acquire);
        ControlSnapshot {
            cancel_sequence: self.0.cancel.load(Ordering::Acquire),
            yield_sequence: self.0.yield_request.load(Ordering::Acquire),
            publish_sequence: self.0.publish.load(Ordering::Acquire),
            cancel_ack: if sealed_cancel_ack == u64::MAX {
                live_cancel_ack
            } else {
                sealed_cancel_ack
            },
            yield_ack: self.0.yield_ack.load(Ordering::Acquire),
            publish_ack: self.0.publish_ack.load(Ordering::Acquire),
            terminal: sealed_cancel_ack != u64::MAX,
        }
    }
    fn request(
        &self,
        sequence: u64,
        request: &AtomicU64,
        ack: &AtomicU64,
        cancellation: bool,
    ) -> Result<(), ControlError> {
        self.request_with_publish_hook(sequence, request, ack, cancellation, || {})
    }
    fn request_with_publish_hook(
        &self,
        sequence: u64,
        request: &AtomicU64,
        ack: &AtomicU64,
        cancellation: bool,
        after_publish: impl FnOnce(),
    ) -> Result<(), ControlError> {
        if self.0.terminal_cancel_ack.load(Ordering::Acquire) != u64::MAX {
            return Err(ControlError::Closed);
        }
        if sequence == u64::MAX {
            return Err(ControlError::SequenceExhausted);
        }
        if sequence == 0 || sequence <= self.0.highest.load(Ordering::Acquire) {
            return Err(ControlError::StaleSequence);
        }
        if !cancellation {
            if self.0.cancel.load(Ordering::Acquire) != 0 {
                return Err(ControlError::Closed);
            }
            if request.load(Ordering::Acquire) > ack.load(Ordering::Acquire) {
                return Err(ControlError::RequestOutstanding);
            }
        }
        // Only the coordinator writes requests. This CAS also rejects callers
        // violating monotonic sequence allocation; it is not a work queue.
        let previous = self.0.highest.load(Ordering::Acquire);
        if previous >= sequence {
            return Err(ControlError::StaleSequence);
        }
        self.0
            .highest
            .compare_exchange(previous, sequence, Ordering::AcqRel, Ordering::Acquire)
            .map_err(|_| ControlError::StaleSequence)?;
        request.store(sequence, Ordering::Release);
        after_publish();
        self.0.signal.notify();
        let terminal_cancel_ack = self.0.terminal_cancel_ack.load(Ordering::Acquire);
        if terminal_cancel_ack != u64::MAX {
            // A fast owner may acknowledge this cancellation and seal the
            // job before the coordinator returns. Only the acknowledgement
            // captured at terminal sealing proves that this request was seen;
            // an acknowledgement racing after sealing is not sufficient.
            if cancellation && terminal_cancel_ack == sequence {
                return Ok(());
            }
            return Err(ControlError::Closed);
        }
        Ok(())
    }
    pub fn request_cancel(&self, sequence: u64) -> Result<(), ControlError> {
        self.request(sequence, &self.0.cancel, &self.0.cancel_ack, true)
    }
    pub fn request_yield(&self, sequence: u64) -> Result<(), ControlError> {
        self.request(sequence, &self.0.yield_request, &self.0.yield_ack, false)
    }
    pub fn request_publish_progress(&self, sequence: u64) -> Result<(), ControlError> {
        self.request(sequence, &self.0.publish, &self.0.publish_ack, false)
    }
    fn acknowledge(
        &self,
        sequence: u64,
        request: &AtomicU64,
        ack: &AtomicU64,
    ) -> Result<(), ControlError> {
        if sequence == 0 || request.load(Ordering::Acquire) != sequence {
            return Err(ControlError::MismatchedAck);
        }
        let previous = ack.load(Ordering::Acquire);
        if previous >= sequence {
            return Err(ControlError::StaleSequence);
        }
        ack.compare_exchange(previous, sequence, Ordering::AcqRel, Ordering::Acquire)
            .map_err(|_| ControlError::StaleSequence)?;
        self.0.signal.notify();
        Ok(())
    }
    /// Call only after the worker retains its owned checkpoint and report.
    pub fn acknowledge_yield(&self, sequence: u64) -> Result<(), ControlError> {
        self.acknowledge(sequence, &self.0.yield_request, &self.0.yield_ack)
    }
    /// Call only after publishing a consistent completed iteration.
    pub fn acknowledge_publish_progress(&self, sequence: u64) -> Result<(), ControlError> {
        self.acknowledge(sequence, &self.0.publish, &self.0.publish_ack)
    }
    /// Records observation of the current cancel request. `Ok` after terminal
    /// sealing does not change the sealed snapshot and does not prove the
    /// completion cause or the release of any physical resources.
    pub fn acknowledge_cancel(&self, sequence: u64) -> Result<(), ControlError> {
        self.acknowledge(sequence, &self.0.cancel, &self.0.cancel_ack)
    }
    fn freeze_terminal_cancel_ack(&self) -> u64 {
        let observed_ack = self.0.cancel_ack.load(Ordering::Acquire);
        match self.0.terminal_cancel_ack.compare_exchange(
            u64::MAX,
            observed_ack,
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            Ok(_) => observed_ack,
            Err(existing) => existing,
        }
    }
    /// Revokes further control admission and freezes the first terminal cancel
    /// acknowledgement. As with acknowledgement, the owning worker is the
    /// single writer: it acknowledges before sealing. Later acknowledgements
    /// cannot change that snapshot. This does not attest a completion cause
    /// or free a GPU lease/token.
    pub fn seal_terminal(&self) -> ControlSnapshot {
        self.freeze_terminal_cancel_ack();
        self.0.signal.notify();
        self.snapshot()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MailboxError {
    InvalidCapacity,
    ResourceLimit,
    Closed,
    QueueFull,
    ReliableFull,
    StaleProgress,
    NoPermit,
}

#[derive(Debug)]
pub struct MailboxRejected<T> {
    pub reason: MailboxError,
    pub value: T,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MailboxOccupancy {
    pub commands: usize,
    pub reliable_queued: usize,
    pub reliable_reserved: usize,
    pub progress_present: bool,
    pub closed: bool,
}

#[derive(Debug)]
struct MailboxState<C, E, P> {
    commands: VecDeque<WorkerCommand<C, E, P>>,
    reliable: VecDeque<E>,
    progress: Option<(u64, P)>,
    progress_sequence: u64,
    reserved: usize,
    closed: bool,
}
#[derive(Debug)]
struct SharedMailbox<C, E, P> {
    state: Mutex<MailboxState<C, E, P>>,
    command_capacity: usize,
    reliable_capacity: usize,
    signal: CompletionSignal,
}

/// A per-worker scoped mailbox. The caller validates immutable Job/owner/
/// ExecutionId in C/E/P; this generic runtime primitive does not parse search.
#[derive(Debug)]
pub struct WorkerMailbox<C, E, P>(Arc<SharedMailbox<C, E, P>>);
impl<C, E, P> Clone for WorkerMailbox<C, E, P> {
    fn clone(&self) -> Self {
        Self(Arc::clone(&self.0))
    }
}
#[derive(Debug)]
pub struct WorkerCommand<C, E, P> {
    pub command: C,
    pub reliable: ReliablePermit<C, E, P>,
}

/// Reserved event storage only. Drop returns unused storage, never CPU credit,
/// Job authority or physical buffers. A supervisor must publish worker failure
/// before abandoning accepted work; a dropped permit is not a terminal fact.
#[derive(Debug)]
pub struct ReliablePermit<C, E, P> {
    mailbox: Weak<SharedMailbox<C, E, P>>,
    remaining: usize,
}
impl<C, E, P> ReliablePermit<C, E, P> {
    pub fn remaining(&self) -> usize {
        self.remaining
    }
    /// Transfer part of an existing storage reservation to a supervisor before
    /// handing work to a thread. No new capacity or credit is minted. The
    /// supervisor can retain a failure-report slot through a worker panic.
    pub fn split_off(&mut self, slots: usize) -> Result<Self, MailboxError> {
        if slots == 0 || slots > self.remaining {
            return Err(MailboxError::NoPermit);
        }
        self.remaining -= slots;
        Ok(Self {
            mailbox: self.mailbox.clone(),
            remaining: slots,
        })
    }
    pub fn publish(&mut self, event: E) -> Result<(), MailboxRejected<E>> {
        if self.remaining == 0 {
            return Err(MailboxRejected {
                reason: MailboxError::NoPermit,
                value: event,
            });
        }
        let Some(mailbox) = self.mailbox.upgrade() else {
            return Err(MailboxRejected {
                reason: MailboxError::Closed,
                value: event,
            });
        };
        let mut state = mailbox
            .state
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        // Admission can be closed during cancellation; accepted reports remain
        // publishable and drainable until the owner disposes of this mailbox.
        assert!(state.reserved > 0);
        assert!(state.reliable.len() < mailbox.reliable_capacity);
        state.reserved -= 1;
        self.remaining -= 1;
        state.reliable.push_back(event);
        drop(state);
        mailbox.signal.notify();
        Ok(())
    }
}
impl<C, E, P> Drop for ReliablePermit<C, E, P> {
    fn drop(&mut self) {
        if self.remaining == 0 {
            return;
        }
        if let Some(mailbox) = self.mailbox.upgrade() {
            let mut state = mailbox
                .state
                .lock()
                .unwrap_or_else(|poison| poison.into_inner());
            state.reserved = state
                .reserved
                .checked_sub(self.remaining)
                .expect("permit owns each released reliable slot");
            self.remaining = 0;
            drop(state);
            mailbox.signal.notify();
        }
    }
}

impl<C, E, P> WorkerMailbox<C, E, P> {
    pub fn new(
        command_capacity: usize,
        reliable_capacity: usize,
        signal: CompletionSignal,
    ) -> Result<Self, MailboxError> {
        if command_capacity == 0
            || command_capacity > 8
            || reliable_capacity == 0
            || reliable_capacity > 256
        {
            return Err(MailboxError::InvalidCapacity);
        }
        let mut commands = VecDeque::new();
        let mut reliable = VecDeque::new();
        commands
            .try_reserve_exact(command_capacity)
            .map_err(|_| MailboxError::ResourceLimit)?;
        reliable
            .try_reserve_exact(reliable_capacity)
            .map_err(|_| MailboxError::ResourceLimit)?;
        Ok(Self(Arc::new(SharedMailbox {
            state: Mutex::new(MailboxState {
                commands,
                reliable,
                progress: None,
                progress_sequence: 0,
                reserved: 0,
                closed: false,
            }),
            command_capacity,
            reliable_capacity,
            signal,
        })))
    }
    pub fn signal(&self) -> CompletionSignal {
        self.0.signal.clone()
    }
    /// Atomic command admission + reliable space reservation. Reserving >1
    /// slots permits separate error/ack/fence facts when the caller needs them.
    pub fn admit(&self, command: C, reliable_slots: usize) -> Result<(), MailboxRejected<C>> {
        let mut state = self
            .0
            .state
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let reason = if state.closed {
            Some(MailboxError::Closed)
        } else if reliable_slots == 0 || reliable_slots > self.0.reliable_capacity {
            Some(MailboxError::InvalidCapacity)
        } else if state.commands.len() == self.0.command_capacity {
            Some(MailboxError::QueueFull)
        } else if reliable_slots > self.0.reliable_capacity - state.reliable.len() - state.reserved
        {
            Some(MailboxError::ReliableFull)
        } else {
            None
        };
        if let Some(reason) = reason {
            return Err(MailboxRejected {
                reason,
                value: command,
            });
        }
        state.reserved += reliable_slots;
        state.commands.push_back(WorkerCommand {
            command,
            reliable: ReliablePermit {
                mailbox: Arc::downgrade(&self.0),
                remaining: reliable_slots,
            },
        });
        drop(state);
        self.0.signal.notify();
        Ok(())
    }
    pub fn take_command(&self) -> Option<WorkerCommand<C, E, P>> {
        let result = self
            .0
            .state
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .commands
            .pop_front();
        if result.is_some() {
            self.0.signal.notify();
        }
        result
    }
    pub fn take_reliable(&self) -> Option<E> {
        let result = self
            .0
            .state
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .reliable
            .pop_front();
        if result.is_some() {
            self.0.signal.notify();
        }
        result
    }
    /// Latest progress only, with strictly increasing per-mailbox sequence.
    /// Construct a new scoped mailbox instead of resetting sequence for a Job.
    pub fn publish_progress(&self, sequence: u64, progress: P) -> Result<(), MailboxRejected<P>> {
        let mut state = self
            .0
            .state
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let reason = if state.closed {
            Some(MailboxError::Closed)
        } else if sequence == 0 || sequence <= state.progress_sequence {
            Some(MailboxError::StaleProgress)
        } else {
            None
        };
        if let Some(reason) = reason {
            return Err(MailboxRejected {
                reason,
                value: progress,
            });
        }
        state.progress_sequence = sequence;
        let old = state.progress.replace((sequence, progress));
        drop(state);
        self.0.signal.notify();
        drop(old);
        Ok(())
    }
    pub fn take_progress(&self) -> Option<(u64, P)> {
        let result = self
            .0
            .state
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .progress
            .take();
        if result.is_some() {
            self.0.signal.notify();
        }
        result
    }
    /// Stops admission/progress without discarding accepted reliable events.
    /// Queued commands remain available for the worker's cancellation reports.
    pub fn close(&self) {
        let mut state = self
            .0
            .state
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        state.closed = true;
        let old = state.progress.take();
        drop(state);
        self.0.signal.notify();
        drop(old);
    }
    pub fn occupancy(&self) -> MailboxOccupancy {
        let state = self
            .0
            .state
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        MailboxOccupancy {
            commands: state.commands.len(),
            reliable_queued: state.reliable.len(),
            reliable_reserved: state.reserved,
            progress_present: state.progress.is_some(),
            closed: state.closed,
        }
    }
    /// Read signal version BEFORE checking all predicates, including the atomic
    /// controls sharing this signal. This never waits while holding state locks.
    pub fn wait_changed(&self, observed: u64, timeout: Duration) -> bool {
        self.0.signal.wait_changed(observed, timeout)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cancellation_acknowledged_before_return_is_not_rejected_as_closed() {
        let mailbox = ControlMailbox::new("job", CompletionSignal::default());
        let owner = mailbox.clone();
        let result = mailbox.request_with_publish_hook(
            1,
            &mailbox.0.cancel,
            &mailbox.0.cancel_ack,
            true,
            move || {
                std::thread::spawn(move || {
                    assert_eq!(owner.snapshot().cancel_sequence, 1);
                    owner.acknowledge_cancel(1).unwrap();
                    owner.seal_terminal();
                })
                .join()
                .unwrap();
            },
        );
        assert_eq!(result, Ok(()));
        let snapshot = mailbox.snapshot();
        assert!(snapshot.terminal);
        assert_eq!((snapshot.cancel_sequence, snapshot.cancel_ack), (1, 1));

        // An independently finished owner does not authorize claiming that
        // this cancellation was processed, even if publication raced closure.
        let naturally_finished = ControlMailbox::new("other", CompletionSignal::default());
        let owner = naturally_finished.clone();
        let result = naturally_finished.request_with_publish_hook(
            1,
            &naturally_finished.0.cancel,
            &naturally_finished.0.cancel_ack,
            true,
            move || {
                std::thread::spawn(move || {
                    assert_eq!(owner.seal_terminal().cancel_ack, 0);
                    // A public late ack cannot change what the owner sealed.
                    owner.acknowledge_cancel(1).unwrap();
                    assert_eq!(owner.seal_terminal().cancel_ack, 0);
                    assert_eq!(owner.snapshot().cancel_ack, 0);
                })
                .join()
                .unwrap();
            },
        );
        assert_eq!(result, Err(ControlError::Closed));
        assert_eq!(naturally_finished.0.cancel_ack.load(Ordering::Acquire), 1);
        assert_eq!(naturally_finished.snapshot().cancel_ack, 0);

        // The first seal CAS is the admission boundary even before the
        // sealer has returned or sent its completion notification.
        let staged = ControlMailbox::new("staged", CompletionSignal::default());
        let owner = staged.clone();
        let result = staged.request_with_publish_hook(
            1,
            &staged.0.cancel,
            &staged.0.cancel_ack,
            true,
            move || {
                std::thread::spawn(move || {
                    assert_eq!(owner.freeze_terminal_cancel_ack(), 0);
                    assert!(owner.snapshot().terminal);
                })
                .join()
                .unwrap();
            },
        );
        assert_eq!(result, Err(ControlError::Closed));
        assert_eq!(staged.seal_terminal().cancel_ack, 0);

        // An earlier acknowledged sequence does not accept a later request.
        let prior = ControlMailbox::new("prior", CompletionSignal::default());
        prior.request_cancel(1).unwrap();
        prior.acknowledge_cancel(1).unwrap();
        let owner = prior.clone();
        let result = prior.request_with_publish_hook(
            2,
            &prior.0.cancel,
            &prior.0.cancel_ack,
            true,
            move || {
                std::thread::spawn(move || owner.seal_terminal())
                    .join()
                    .unwrap();
            },
        );
        assert_eq!(result, Err(ControlError::Closed));
        assert_eq!(prior.snapshot().cancel_ack, 1);
    }
    #[test]
    fn concurrent_duplicate_terminal_seals_keep_one_acknowledgement() {
        let mailbox = ControlMailbox::new("job", CompletionSignal::default());
        mailbox.request_cancel(1).unwrap();
        mailbox.acknowledge_cancel(1).unwrap();
        let barrier = Arc::new(std::sync::Barrier::new(3));
        let handles: Vec<_> = (0..2)
            .map(|_| {
                let owner = mailbox.clone();
                let barrier = Arc::clone(&barrier);
                std::thread::spawn(move || {
                    barrier.wait();
                    owner.seal_terminal()
                })
            })
            .collect();
        barrier.wait();
        for handle in handles {
            let snapshot = handle.join().unwrap();
            assert!(snapshot.terminal);
            assert_eq!(snapshot.cancel_ack, 1);
        }
        assert_eq!(mailbox.snapshot().cancel_ack, 1);
    }
    #[test]
    fn full_command_queue_cannot_block_yield_or_sticky_cancel() {
        let signal = CompletionSignal::default();
        let mailbox = WorkerMailbox::<u8, u8, u8>::new(1, 2, signal.clone()).unwrap();
        let control = ControlMailbox::new((7_u64, 11_u64), signal);
        mailbox.admit(1, 1).unwrap();
        assert_eq!(
            mailbox.admit(2, 1).unwrap_err().reason,
            MailboxError::QueueFull
        );
        control.request_yield(1).unwrap();
        assert_eq!(control.snapshot().pending_yield(), Some(1));
        assert_eq!(
            control.request_yield(2),
            Err(ControlError::RequestOutstanding)
        );
        control.acknowledge_yield(1).unwrap();
        control.request_yield(2).unwrap();
        control.request_cancel(3).unwrap();
        control.acknowledge_cancel(3).unwrap();
        assert!(control.snapshot().is_cancelled());
        assert!(control.snapshot().pending_yield().is_none());
        assert_eq!(
            control.request_publish_progress(4),
            Err(ControlError::Closed)
        );
        control.seal_terminal();
        assert_eq!(control.request_cancel(5), Err(ControlError::Closed));
        let new_job = ControlMailbox::new((8_u64, 11_u64), CompletionSignal::default());
        assert!(!new_job.snapshot().is_cancelled());
    }
    #[test]
    fn reliable_reservations_survive_close_and_progress_coalesce() {
        let mailbox =
            WorkerMailbox::<u8, &'static str, u8>::new(2, 2, CompletionSignal::default()).unwrap();
        mailbox.admit(1, 2).unwrap();
        assert_eq!(
            mailbox.admit(2, 1).unwrap_err().reason,
            MailboxError::ReliableFull
        );
        for seq in 1..=100 {
            mailbox.publish_progress(seq, seq as u8).unwrap();
        }
        assert_eq!(mailbox.take_progress(), Some((100, 100)));
        assert_eq!(
            mailbox.publish_progress(100, 0).unwrap_err().reason,
            MailboxError::StaleProgress
        );
        let mut command = mailbox.take_command().unwrap();
        mailbox.close();
        command.reliable.publish("yield_ack").unwrap();
        command.reliable.publish("terminal").unwrap();
        assert_eq!(mailbox.occupancy().reliable_reserved, 0);
        assert_eq!(mailbox.occupancy().reliable_queued, 2);
        assert_eq!(mailbox.take_reliable(), Some("yield_ack"));
        assert_eq!(mailbox.take_reliable(), Some("terminal"));
        assert_eq!(
            command.reliable.publish("duplicate").unwrap_err().reason,
            MailboxError::NoPermit
        );
    }
    #[test]
    fn unused_permits_release_storage_and_do_not_form_arc_cycles() {
        let mailbox = WorkerMailbox::<u8, u8, u8>::new(2, 3, CompletionSignal::default()).unwrap();
        mailbox.admit(1, 2).unwrap();
        let command = mailbox.take_command().unwrap();
        assert_eq!(mailbox.occupancy().reliable_reserved, 2);
        drop(command);
        assert_eq!(mailbox.occupancy().reliable_reserved, 0);
        mailbox.admit(2, 3).unwrap();
        let weak = Arc::downgrade(&mailbox.0);
        drop(mailbox);
        assert!(weak.upgrade().is_none());
    }
    #[test]
    fn supervisor_retains_failure_slot_when_worker_permit_is_dropped() {
        let mailbox =
            WorkerMailbox::<u8, &'static str, u8>::new(1, 2, CompletionSignal::default()).unwrap();
        mailbox.admit(1, 2).unwrap();
        let mut command = mailbox.take_command().unwrap();
        let mut supervisor = command.reliable.split_off(1).unwrap();
        drop(command);
        assert_eq!(mailbox.occupancy().reliable_reserved, 1);
        supervisor.publish("worker_failed").unwrap();
        assert_eq!(mailbox.take_reliable(), Some("worker_failed"));
    }
    #[test]
    fn sequence_and_ack_misuse_and_late_progress_are_rejected() {
        let mailbox = ControlMailbox::new("job", CompletionSignal::default());
        assert_eq!(mailbox.request_yield(0), Err(ControlError::StaleSequence));
        assert_eq!(
            mailbox.request_yield(u64::MAX),
            Err(ControlError::SequenceExhausted)
        );
        mailbox.request_publish_progress(10).unwrap();
        assert_eq!(
            mailbox.acknowledge_publish_progress(9),
            Err(ControlError::MismatchedAck)
        );
        mailbox.acknowledge_publish_progress(10).unwrap();
        assert_eq!(
            mailbox.acknowledge_publish_progress(10),
            Err(ControlError::StaleSequence)
        );
        assert_eq!(mailbox.request_yield(9), Err(ControlError::StaleSequence));
        let final_snapshot = mailbox.seal_terminal();
        assert!(final_snapshot.terminal);
        assert_eq!(
            mailbox.request_publish_progress(11),
            Err(ControlError::Closed)
        );
        let messages = WorkerMailbox::<u8, u8, u8>::new(1, 1, CompletionSignal::default()).unwrap();
        messages.publish_progress(1, 1).unwrap();
        messages.close();
        assert!(messages.take_progress().is_none());
        assert_eq!(
            messages.publish_progress(2, 2).unwrap_err().reason,
            MailboxError::Closed
        );
    }
    #[test]
    fn version_recheck_covers_notifications_before_and_during_wait() {
        let signal = CompletionSignal::default();
        let mailbox = WorkerMailbox::<u8, u8, u8>::new(1, 1, signal.clone()).unwrap();
        let controls = ControlMailbox::new(1, signal.clone());
        let observed = signal.version();
        assert!(mailbox.take_command().is_none());
        controls.request_cancel(1).unwrap();
        assert!(mailbox.wait_changed(observed, Duration::from_secs(1)));
        let observed = signal.version();
        let waiter = mailbox.clone();
        let thread =
            std::thread::spawn(move || waiter.wait_changed(observed, Duration::from_secs(1)));
        mailbox.admit(1, 1).unwrap();
        assert!(thread.join().unwrap());
        assert!(!mailbox.wait_changed(signal.version(), Duration::ZERO));
    }
}
