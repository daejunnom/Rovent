//! Bounded, observational UCI relay state for one directly owned child.
//!
//! This module never writes to a pipe, creates a reply, starts a clock, sends a
//! signal or establishes GPU completion. The executor forwards the original
//! bytes unchanged and alone owns their ordering and resource limits. State
//! below may guide diagnostics; it is never authority to fabricate `bestmove`.

/// Reject an individual line before allocating or forwarding it when it is
/// larger than the registered relay limit. The executor retains the original
/// child and records the transport failure separately from the engine result.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum RelayObservationError {
    InvalidLimit,
    LineTooLong,
    IncompleteLine,
    MultipleLines,
}

/// UCI requires commands to be handled in order. An overlapping `go` is
/// observable but its following bestmove cannot be assigned to either search.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum PendingSearch {
    None,
    One { sequence: u64, stop_seen: bool },
    Ambiguous,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum InputObservation {
    Other,
    GoStarted { sequence: u64 },
    OverlappingGo,
    Stop,
    Quit,
    InputAfterShutdown,
    SequenceExhausted,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum OutputObservation {
    Other,
    /// A syntactically present line while one request is pending. UCI does
    /// not tag responses: this temporal observation alone cannot establish
    /// authority, legality, or exclude a late/duplicate older response.
    BestmoveWhilePending {
        sequence: u64,
    },
    BestmoveWithoutPending,
    BestmoveAmbiguous,
}

/// A lifecycle observation supplied by the original process owner, not by a
/// UCI text line. The executor must obtain the two positive cases from the
/// same native-entry/session owner; process exit or an EOF is insufficient.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum OriginalOwnerPhysical {
    ProvenBeforeNative,
    ConfirmedSessionRelease,
    Unknown,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum ReleaseDecision {
    VerifyOriginalOwnerEvidence,
    PreserveOriginalOwner,
}

/// A finite per-line view only. No entire game transcript or model bytes are
/// retained. The upper bound is supplied by the registered executor profile.
pub(super) struct RelayState {
    max_line_bytes: usize,
    sequence: u64,
    pending: PendingSearch,
    quit_seen: bool,
    input_eof: bool,
    output_eof: bool,
}

impl RelayState {
    pub(super) fn new(max_line_bytes: usize) -> Result<Self, RelayObservationError> {
        if max_line_bytes == 0 {
            return Err(RelayObservationError::InvalidLimit);
        }
        Ok(Self {
            max_line_bytes,
            sequence: 0,
            pending: PendingSearch::None,
            quit_seen: false,
            input_eof: false,
            output_eof: false,
        })
    }

    pub(super) fn pending(&self) -> PendingSearch {
        self.pending
    }

    /// Observe a complete original line without rewriting or retaining it.
    /// Unknown/invalid UCI text remains the engine's responsibility. Caller
    /// may forward the same bytes even when a protocol anomaly is observed.
    pub(super) fn observe_input(
        &mut self,
        raw: &[u8],
    ) -> Result<InputObservation, RelayObservationError> {
        let mut tokens = Self::tokens(raw, self.max_line_bytes)?;
        if self.quit_seen || self.input_eof {
            return Ok(InputObservation::InputAfterShutdown);
        }
        match tokens.next() {
            Some(b"go") => {
                if self.pending != PendingSearch::None {
                    self.pending = PendingSearch::Ambiguous;
                    return Ok(InputObservation::OverlappingGo);
                }
                let Some(sequence) = self.sequence.checked_add(1) else {
                    self.pending = PendingSearch::Ambiguous;
                    return Ok(InputObservation::SequenceExhausted);
                };
                self.sequence = sequence;
                self.pending = PendingSearch::One {
                    sequence,
                    stop_seen: false,
                };
                Ok(InputObservation::GoStarted { sequence })
            }
            Some(b"stop") if tokens.next().is_none() => {
                if let PendingSearch::One { sequence, .. } = self.pending {
                    self.pending = PendingSearch::One {
                        sequence,
                        stop_seen: true,
                    };
                }
                Ok(InputObservation::Stop)
            }
            Some(b"quit") if tokens.next().is_none() => {
                self.quit_seen = true;
                Ok(InputObservation::Quit)
            }
            _ => Ok(InputObservation::Other),
        }
    }

    /// The actual child bytes always remain the only source of a bestmove.
    /// Legal-move and search correctness are checked by the existing Rules and
    /// arena paths, not by this relay.
    pub(super) fn observe_output(
        &mut self,
        raw: &[u8],
    ) -> Result<OutputObservation, RelayObservationError> {
        let mut tokens = Self::tokens(raw, self.max_line_bytes)?;
        if tokens.next() != Some(&b"bestmove"[..]) || tokens.next().is_none() {
            return Ok(OutputObservation::Other);
        }
        match self.pending {
            PendingSearch::None => Ok(OutputObservation::BestmoveWithoutPending),
            PendingSearch::Ambiguous => Ok(OutputObservation::BestmoveAmbiguous),
            PendingSearch::One { sequence, .. } => {
                self.pending = PendingSearch::None;
                Ok(OutputObservation::BestmoveWhilePending { sequence })
            }
        }
    }

    pub(super) fn note_input_eof(&mut self) {
        self.input_eof = true;
    }

    pub(super) fn note_output_eof(&mut self) {
        self.output_eof = true;
    }

    pub(super) fn transport_closed(&self) -> bool {
        self.input_eof && self.output_eof
    }

    /// This decision only covers native ownership. The original child and
    /// other workers still need their own explicit lifecycle/exit evidence.
    pub(super) fn release_decision(physical: OriginalOwnerPhysical) -> ReleaseDecision {
        match physical {
            OriginalOwnerPhysical::ProvenBeforeNative
            | OriginalOwnerPhysical::ConfirmedSessionRelease => {
                ReleaseDecision::VerifyOriginalOwnerEvidence
            }
            OriginalOwnerPhysical::Unknown => ReleaseDecision::PreserveOriginalOwner,
        }
    }

    fn tokens(
        raw: &[u8],
        max_line_bytes: usize,
    ) -> Result<impl Iterator<Item = &[u8]>, RelayObservationError> {
        if raw.len() > max_line_bytes {
            return Err(RelayObservationError::LineTooLong);
        }
        if !raw.ends_with(b"\n") {
            return Err(RelayObservationError::IncompleteLine);
        }
        if raw[..raw.len() - 1].contains(&b'\n') {
            return Err(RelayObservationError::MultipleLines);
        }
        Ok(raw
            .split(|byte| byte.is_ascii_whitespace())
            .filter(|part| !part.is_empty()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn input_and_output_are_observations_of_original_bytes() {
        let mut state = RelayState::new(1024).unwrap();
        let input = b"go wtime 1000 btime 2000\r\n";
        assert_eq!(
            state.observe_input(input),
            Ok(InputObservation::GoStarted { sequence: 1 })
        );
        assert_eq!(input, b"go wtime 1000 btime 2000\r\n");
        assert_eq!(
            state.observe_output(b"info depth 1\n"),
            Ok(OutputObservation::Other)
        );
        assert_eq!(
            state.observe_output(b"bestmove e2e4 ponder e7e5\n"),
            Ok(OutputObservation::BestmoveWhilePending { sequence: 1 })
        );
        assert_eq!(
            state.observe_output(b"bestmove d2d4\n"),
            Ok(OutputObservation::BestmoveWithoutPending)
        );
    }

    #[test]
    fn overlapping_go_never_claims_a_later_bestmove() {
        let mut state = RelayState::new(128).unwrap();
        assert_eq!(
            state.observe_input(b"go nodes 10\n"),
            Ok(InputObservation::GoStarted { sequence: 1 })
        );
        assert_eq!(
            state.observe_input(b"go nodes 20\n"),
            Ok(InputObservation::OverlappingGo)
        );
        assert_eq!(
            state.observe_output(b"bestmove e2e4\n"),
            Ok(OutputObservation::BestmoveAmbiguous)
        );
        assert_eq!(state.pending(), PendingSearch::Ambiguous);
    }

    #[test]
    fn stop_quit_and_eof_do_not_synthesize_completion() {
        let mut state = RelayState::new(128).unwrap();
        state.observe_input(b"go infinite\n").unwrap();
        assert_eq!(state.observe_input(b"stop\n"), Ok(InputObservation::Stop));
        assert_eq!(
            state.pending(),
            PendingSearch::One {
                sequence: 1,
                stop_seen: true
            }
        );
        assert_eq!(state.observe_input(b"quit\n"), Ok(InputObservation::Quit));
        state.note_input_eof();
        state.note_output_eof();
        assert!(state.transport_closed());
        assert_eq!(
            state.pending(),
            PendingSearch::One {
                sequence: 1,
                stop_seen: true
            }
        );
        assert_eq!(
            RelayState::release_decision(OriginalOwnerPhysical::Unknown),
            ReleaseDecision::PreserveOriginalOwner
        );
    }

    #[test]
    fn positive_release_requires_original_owner_observation() {
        assert_eq!(
            RelayState::release_decision(OriginalOwnerPhysical::ProvenBeforeNative),
            ReleaseDecision::VerifyOriginalOwnerEvidence
        );
        assert_eq!(
            RelayState::release_decision(OriginalOwnerPhysical::ConfirmedSessionRelease),
            ReleaseDecision::VerifyOriginalOwnerEvidence
        );
    }

    #[test]
    fn size_incomplete_and_unknown_text_do_not_imply_engine_failure() {
        assert_eq!(
            RelayState::new(0).err(),
            Some(RelayObservationError::InvalidLimit)
        );
        let mut state = RelayState::new(8).unwrap();
        assert_eq!(
            state.observe_input(b"go nodes 100\n"),
            Err(RelayObservationError::LineTooLong)
        );
        assert_eq!(
            state.observe_input(b"go"),
            Err(RelayObservationError::IncompleteLine)
        );
        assert_eq!(
            state.observe_input(b"a\nb\n"),
            Err(RelayObservationError::MultipleLines)
        );
        assert_eq!(state.observe_input(b"\xff\n"), Ok(InputObservation::Other));
        assert_eq!(state.pending(), PendingSearch::None);
    }
}
