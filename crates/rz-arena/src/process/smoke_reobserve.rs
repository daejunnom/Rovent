//! Bounded observation of the original NoGame child and its original pipes.
//!
//! This module issues no signal, stdin command, search grant or cleanup grant.
//! An observation window does not extend the original execution window.

use std::io::{self, Read};
use std::os::fd::AsFd;
use std::sync::Arc;
use std::time::Instant;

use nix::errno::Errno;
use nix::fcntl::{FcntlArg, OFlag, fcntl};
use nix::sys::wait::{Id, WaitPidFlag, WaitStatus, waitid};
use nix::unistd::Pid;

use super::{OriginalSmokeUciObservation, SpawnCustody};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ChildObservation {
    Alive,
    Exited(i32),
    Signaled(i32),
    Interrupted,
    Unknown,
    /// The original wait relationship can no longer be established. Never
    /// infer permission to signal or reap from the numeric PID afterwards.
    OwnershipLost,
    WindowElapsed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum PipeObservation {
    Absent,
    Eof,
    Pending,
    OutputCap,
    /// The original FD is retained; the failure is not interpreted as EOF.
    Unknown,
    WindowElapsed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct SmokeReobserve {
    pub(super) child: ChildObservation,
    pub(super) stdout: PipeObservation,
    pub(super) stderr: PipeObservation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum SmokeReobserveError {
    NotRetained,
    DifferentOwner,
    MissingChild,
}

/// One finite non-reaping tick over the original custody and observation.
/// The original capture cap remains in force. Retained bytes are a prefix; a
/// full cap can leave the pipe backpressured. Child exit is CPU process
/// evidence, not a GPU completion fence.
pub(super) fn observe_once(
    custody: &mut SpawnCustody,
    observation: &mut OriginalSmokeUciObservation,
    until: Instant,
) -> Result<SmokeReobserve, SmokeReobserveError> {
    let pipes = custody
        .retained_smoke
        .as_mut()
        .ok_or(SmokeReobserveError::NotRetained)?;
    let token = observation
        .retained_token
        .as_ref()
        .ok_or(SmokeReobserveError::NotRetained)?;
    if !Arc::ptr_eq(&pipes.token, token) {
        return Err(SmokeReobserveError::DifferentOwner);
    }
    let child = custody
        .child
        .as_ref()
        .ok_or(SmokeReobserveError::MissingChild)?;
    let output = &mut observation.output;
    let cap = pipes.max_output_bytes;
    let child = if pipes.ownership_lost {
        ChildObservation::OwnershipLost
    } else if Instant::now() >= until {
        ChildObservation::WindowElapsed
    } else if let Ok(raw_pid) = i32::try_from(child.id()) {
        if raw_pid <= 0 {
            pipes.ownership_lost = true;
            ChildObservation::OwnershipLost
        } else {
            match waitid(
                Id::Pid(Pid::from_raw(raw_pid)),
                WaitPidFlag::WEXITED | WaitPidFlag::WNOHANG | WaitPidFlag::WNOWAIT,
            ) {
                Ok(WaitStatus::StillAlive) => ChildObservation::Alive,
                Ok(WaitStatus::Exited(_, status)) => ChildObservation::Exited(status),
                Ok(WaitStatus::Signaled(_, signal, _)) => ChildObservation::Signaled(signal as i32),
                Err(Errno::EINTR) => ChildObservation::Interrupted,
                Err(_) => {
                    pipes.ownership_lost = true;
                    ChildObservation::OwnershipLost
                }
                Ok(_) => ChildObservation::Unknown,
            }
        }
    } else {
        pipes.ownership_lost = true;
        ChildObservation::OwnershipLost
    };

    let stdout = observe_pipe(
        &mut pipes.stdout,
        &mut pipes.stdout_eof,
        &mut output.stdout,
        output.stderr.len(),
        cap,
        until,
    );
    let stderr = observe_pipe(
        &mut pipes.stderr,
        &mut pipes.stderr_eof,
        &mut output.stderr,
        output.stdout.len(),
        cap,
        until,
    );
    pipes.output_at_capacity |=
        stdout == PipeObservation::OutputCap || stderr == PipeObservation::OutputCap;
    Ok(SmokeReobserve {
        child,
        stdout,
        stderr,
    })
}

fn observe_pipe<T: Read + AsFd>(
    pipe: &mut Option<T>,
    eof: &mut bool,
    output: &mut Vec<u8>,
    other_len: usize,
    cap: u64,
    until: Instant,
) -> PipeObservation {
    if *eof {
        return PipeObservation::Eof;
    }
    let Some(pipe) = pipe.as_mut() else {
        return PipeObservation::Absent;
    };
    if Instant::now() >= until {
        return PipeObservation::WindowElapsed;
    }
    // Verify rather than change the descriptor's flags. Never block the owner
    // on a pipe whose nonblocking setup failed before the retained handoff.
    let Ok(flags) = fcntl(&*pipe, FcntlArg::F_GETFL) else {
        return PipeObservation::Unknown;
    };
    if !OFlag::from_bits_truncate(flags).contains(OFlag::O_NONBLOCK) {
        return PipeObservation::Unknown;
    }
    let mut buf = [0_u8; 8192];
    for _ in 0..16 {
        if Instant::now() >= until {
            return PipeObservation::WindowElapsed;
        }
        let Some(remaining) = remaining_capacity(output.len(), other_len, cap) else {
            return PipeObservation::OutputCap;
        };
        let request = remaining.min(buf.len());
        // Do not consume original bytes unless their retained destination is
        // already available; allocation failure leaves the pipe untouched.
        if output.try_reserve(request).is_err() {
            return PipeObservation::Unknown;
        }
        match pipe.read(&mut buf[..request]) {
            Ok(0) => {
                *eof = true;
                return PipeObservation::Eof;
            }
            Ok(n) => output.extend_from_slice(&buf[..n]),
            Err(e) if e.kind() == io::ErrorKind::WouldBlock => return PipeObservation::Pending,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
            Err(_) => return PipeObservation::Unknown,
        }
    }
    PipeObservation::Pending
}

fn remaining_capacity(len: usize, other_len: usize, cap: u64) -> Option<usize> {
    let total = u64::try_from(len)
        .ok()?
        .checked_add(u64::try_from(other_len).ok()?)?;
    let remaining = cap.checked_sub(total)?;
    if remaining == 0 {
        return None;
    }
    Some(usize::try_from(remaining).unwrap_or(usize::MAX))
}

#[cfg(test)]
mod tests {
    use super::remaining_capacity;

    #[test]
    fn combined_output_capacity_never_reads_past_the_original_limit() {
        assert_eq!(remaining_capacity(4, 3, 10), Some(3));
        assert_eq!(remaining_capacity(4, 6, 10), None);
        assert_eq!(remaining_capacity(4, 7, 10), None);
    }
}
