//! Same-owner NoGame proxy preparation without native or asset execution.
//! The caller supplies original stdio and already-connected role channels;
//! these are not proof of argv provenance, listener authority or GPU finish.

use super::{
    proxy_entry_no_game_smoke_v1::{
        OriginalNoGameSmokeGrants, ProxyEntryNoGameAdmittedBundleV1, ProxyEntryNoGameSmokeReceiptV1,
    },
    proxy_entry_smoke_channel_v1::{
        ObservedSmokeChallenge, OriginalDirectSmokeGrant, PendingDirectSmokeChallenge,
        PendingDirectSmokeGrant, SmokeChallengePoll, SmokeGrantPoll,
    },
    proxy_entry_smoke_lock_v1::OriginalSmokeLockAdmission,
    proxy_entry_smoke_v1::ProxyEntryFunctionalSmokeV1,
    smoke_expected_argv_v1::ParsedSmokeExpectedArgvV1,
};
use rz_contracts::pals::proxy_entry_smoke_grant_v1::ProxyEntrySmokePhase;
use std::{
    io,
    os::{fd::OwnedFd, unix::net::UnixStream},
    sync::{Arc, atomic::AtomicBool},
    time::Instant,
};

#[cfg(test)]
#[path = "smoke_prepare_tests.rs"]
mod tests;

fn invalid(message: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

/// Caller-owned descriptors are retained verbatim. This wrapper makes no
/// assertion that the caller has authenticated or validated their provenance.
pub(super) struct OriginalProxyStdioV1 {
    pub incoming: OwnedFd,
    pub outgoing: OwnedFd,
    pub diagnostics: OwnedFd,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum SmokePreparationStageV1 {
    AwaitRunChannel,
    RunChallenge,
    RunGrant,
    AwaitCleanupChannel,
    CleanupChallenge,
    CleanupGrant,
    GrantsPrepared,
    Failed,
}

fn next_phase(stage: SmokePreparationStageV1) -> Option<ProxyEntrySmokePhase> {
    match stage {
        SmokePreparationStageV1::AwaitRunChannel
        | SmokePreparationStageV1::RunChallenge
        | SmokePreparationStageV1::RunGrant => Some(ProxyEntrySmokePhase::Run),
        SmokePreparationStageV1::AwaitCleanupChannel
        | SmokePreparationStageV1::CleanupChallenge
        | SmokePreparationStageV1::CleanupGrant => Some(ProxyEntrySmokePhase::Cleanup),
        _ => None,
    }
}

/// This owner keeps all transferred resources, including any partially read
/// channel and first error. It does not execute a child or open model assets.
/// Dropping this owner closes its held descriptors; callers must retain or
/// explicitly settle it after failure rather than treating failure as cleanup.
#[must_use = "retain or settle the original smoke preparation owner"]
pub(super) struct NoGameProxyPrepareOwnerV1 {
    argv: Option<ParsedSmokeExpectedArgvV1>,
    entry: Option<ProxyEntryFunctionalSmokeV1>,
    lock: Option<OriginalSmokeLockAdmission>,
    stdio: Option<OriginalProxyStdioV1>,
    cancelled: Arc<AtomicBool>,
    cleanup_cancelled: Arc<AtomicBool>,
    stage: SmokePreparationStageV1,
    challenge: Option<PendingDirectSmokeChallenge>,
    run_challenge: Option<ObservedSmokeChallenge>,
    pending_grant: Option<PendingDirectSmokeGrant>,
    run: Option<OriginalDirectSmokeGrant>,
    cleanup: Option<OriginalDirectSmokeGrant>,
    ready: Option<(
        ProxyEntryNoGameAdmittedBundleV1,
        ProxyEntryNoGameSmokeReceiptV1,
    )>,
    first_failure: Option<(io::ErrorKind, String, Option<i32>)>,
}

#[must_use = "retain or settle the original smoke grant and stdio owners"]
pub(super) struct PreparedNoGameGrantOwnersV1 {
    pub argv: ParsedSmokeExpectedArgvV1,
    pub stdio: OriginalProxyStdioV1,
    pub bundle: ProxyEntryNoGameAdmittedBundleV1,
    pub receipt: ProxyEntryNoGameSmokeReceiptV1,
}

impl NoGameProxyPrepareOwnerV1 {
    /// Failure returns the same four input owners. The caller retains both
    /// cancellation Arcs because this function only borrows them.
    pub(super) fn new(
        argv: ParsedSmokeExpectedArgvV1,
        entry: ProxyEntryFunctionalSmokeV1,
        lock: OriginalSmokeLockAdmission,
        stdio: OriginalProxyStdioV1,
        cancelled: &Arc<AtomicBool>,
        cleanup_cancelled: &Arc<AtomicBool>,
    ) -> Result<
        Self,
        (
            io::Error,
            ParsedSmokeExpectedArgvV1,
            ProxyEntryFunctionalSmokeV1,
            OriginalSmokeLockAdmission,
            OriginalProxyStdioV1,
        ),
    > {
        let result = (|| {
            if entry.local_entry() != lock.entry()
                || !lock.matches_token(cancelled)
                || !argv.matches_local_lock(
                    &lock.lock().canonical_sha256,
                    lock.lock().config.process_generation,
                )
            {
                return Err(invalid(
                    "NoGame initial lock and expected declaration differ",
                ));
            }
            lock.verify_held()
        })();
        if let Err(error) = result {
            return Err((error, argv, entry, lock, stdio));
        }
        Ok(Self {
            argv: Some(argv),
            entry: Some(entry),
            lock: Some(lock),
            stdio: Some(stdio),
            cancelled: Arc::clone(cancelled),
            cleanup_cancelled: Arc::clone(cleanup_cancelled),
            stage: SmokePreparationStageV1::AwaitRunChannel,
            challenge: None,
            run_challenge: None,
            pending_grant: None,
            run: None,
            cleanup: None,
            ready: None,
            first_failure: None,
        })
    }
    pub(super) fn stage(&self) -> SmokePreparationStageV1 {
        self.stage
    }
    pub(super) fn first_failure(&self) -> Option<(io::ErrorKind, &str, Option<i32>)> {
        self.first_failure
            .as_ref()
            .map(|(kind, message, raw)| (*kind, message.as_str(), *raw))
    }
    fn fail(&mut self, error: io::Error) -> io::Error {
        if self.first_failure.is_none() {
            self.first_failure = Some((error.kind(), error.to_string(), error.raw_os_error()));
        }
        self.stage = SmokePreparationStageV1::Failed;
        error
    }
    /// Accept one caller-owned connection for the currently expected phase.
    /// Its peer is observed by the receiver; its listener origin still needs
    /// the original arena owner's independent proof.
    pub(super) fn attach_channel(
        &mut self,
        stream: UnixStream,
        until: Instant,
    ) -> Result<(), (io::Error, UnixStream)> {
        let stage = match self.stage {
            SmokePreparationStageV1::AwaitRunChannel => SmokePreparationStageV1::RunChallenge,
            SmokePreparationStageV1::AwaitCleanupChannel => {
                SmokePreparationStageV1::CleanupChallenge
            }
            _ => return Err((invalid("NoGame smoke channel is not expected"), stream)),
        };
        let Some(lock) = self.lock.as_ref() else {
            return Err((invalid("NoGame original lock owner missing"), stream));
        };
        if let Err(error) = lock.verify_held() {
            return Err((self.fail(error), stream));
        }
        let until = until.min(lock.local_until());
        let challenge = match PendingDirectSmokeChallenge::begin(stream, until) {
            Ok(challenge) => challenge,
            Err((error, stream)) => return Err((self.fail(error), stream)),
        };
        self.challenge = Some(challenge);
        self.stage = stage;
        Ok(())
    }
    /// Performs one nonblocking challenge or grant poll. On any failure the
    /// same channel/grant/lock/stdio owners stay in this object.
    pub(super) fn poll(&mut self) -> io::Result<SmokePreparationStageV1> {
        if let Some((kind, message, _)) = &self.first_failure {
            return Err(io::Error::new(*kind, message.clone()));
        }
        match self.stage {
            SmokePreparationStageV1::AwaitRunChannel
            | SmokePreparationStageV1::AwaitCleanupChannel
            | SmokePreparationStageV1::GrantsPrepared => return Ok(self.stage),
            SmokePreparationStageV1::Failed => {
                return Err(self.fail(invalid("NoGame preparation failed")));
            }
            _ => {}
        }
        if let Some(lock) = self.lock.as_ref() {
            if let Err(error) = lock.check_control() {
                return Err(self.fail(error));
            }
        } else {
            return Err(self.fail(invalid("NoGame original lock owner missing")));
        }
        let Some(phase) = next_phase(self.stage) else {
            return Err(self.fail(invalid("NoGame phase unavailable")));
        };
        match self.stage {
            SmokePreparationStageV1::RunChallenge | SmokePreparationStageV1::CleanupChallenge => {
                let Some(challenge) = self.challenge.as_mut() else {
                    return Err(self.fail(invalid("NoGame challenge owner missing")));
                };
                match challenge.poll() {
                    Ok(SmokeChallengePoll::Pending) => return Ok(self.stage),
                    Ok(SmokeChallengePoll::Complete) => {}
                    Err(error) => return Err(self.fail(error)),
                }
                let Some(challenge) = self.challenge.take() else {
                    return Err(self.fail(invalid("NoGame challenge owner missing")));
                };
                let Some(argv) = self.argv.as_ref() else {
                    self.challenge = Some(challenge);
                    return Err(self.fail(invalid("NoGame argv owner missing")));
                };
                let prior = if phase == ProxyEntrySmokePhase::Cleanup {
                    self.run_challenge.as_ref()
                } else {
                    None
                };
                match argv.compare_original_challenge(challenge, phase, prior) {
                    Ok((grant, observed)) => {
                        if phase == ProxyEntrySmokePhase::Run {
                            self.run_challenge = Some(observed);
                        }
                        self.pending_grant = Some(grant);
                        self.stage = if phase == ProxyEntrySmokePhase::Run {
                            SmokePreparationStageV1::RunGrant
                        } else {
                            SmokePreparationStageV1::CleanupGrant
                        };
                    }
                    Err((error, challenge)) => {
                        self.challenge = Some(challenge);
                        return Err(self.fail(error));
                    }
                }
            }
            SmokePreparationStageV1::RunGrant | SmokePreparationStageV1::CleanupGrant => {
                let Some(grant) = self.pending_grant.as_mut() else {
                    return Err(self.fail(invalid("NoGame pending grant owner missing")));
                };
                match grant.poll() {
                    Ok(SmokeGrantPoll::Pending) => return Ok(self.stage),
                    Ok(SmokeGrantPoll::Complete) => {}
                    Err(error) => return Err(self.fail(error)),
                }
                let Some(grant) = self.pending_grant.take() else {
                    return Err(self.fail(invalid("NoGame pending grant owner missing")));
                };
                let grant = match grant.into_completed() {
                    Ok(grant) => grant,
                    Err(grant) => {
                        self.pending_grant = Some(grant);
                        return Err(self.fail(invalid("NoGame original grant incomplete")));
                    }
                };
                if phase == ProxyEntrySmokePhase::Run {
                    self.run = Some(grant);
                    self.stage = SmokePreparationStageV1::AwaitCleanupChannel;
                } else {
                    self.cleanup = Some(grant);
                    self.finish_projection()?;
                }
            }
            _ => return Err(self.fail(invalid("NoGame preparation stage invalid"))),
        }
        Ok(self.stage)
    }
    fn finish_projection(&mut self) -> io::Result<()> {
        let Some(entry) = self.entry.take() else {
            return Err(self.fail(invalid("NoGame entry owner missing")));
        };
        let Some(lock) = self.lock.take() else {
            self.entry = Some(entry);
            return Err(self.fail(invalid("NoGame original lock owner missing")));
        };
        let Some(run) = self.run.take() else {
            self.entry = Some(entry);
            self.lock = Some(lock);
            return Err(self.fail(invalid("NoGame run grant owner missing")));
        };
        let Some(cleanup) = self.cleanup.take() else {
            self.entry = Some(entry);
            self.lock = Some(lock);
            self.run = Some(run);
            return Err(self.fail(invalid("NoGame cleanup grant owner missing")));
        };
        match entry.bind_admitted_no_game_original(
            OriginalNoGameSmokeGrants { run, cleanup },
            lock,
            &self.cancelled,
            &self.cleanup_cancelled,
        ) {
            Ok(ready) => {
                self.ready = Some(ready);
                self.stage = SmokePreparationStageV1::GrantsPrepared;
                Ok(())
            }
            Err((error, entry, inputs, lock)) => {
                self.entry = Some(entry);
                self.lock = Some(lock);
                self.run = Some(inputs.run);
                self.cleanup = Some(inputs.cleanup);
                Err(self.fail(error))
            }
        }
    }
    /// Grant preparation retains argv, caller-provided stdio, lock and both
    /// grant FDs. It is not a validated CLI or native-execution readiness.
    pub(super) fn into_prepared_grants(mut self) -> Result<PreparedNoGameGrantOwnersV1, Self> {
        if self.stage != SmokePreparationStageV1::GrantsPrepared
            || self.argv.is_none()
            || self.stdio.is_none()
            || self.ready.is_none()
        {
            return Err(self);
        }
        let Some(argv) = self.argv.take() else {
            return Err(self);
        };
        let Some(stdio) = self.stdio.take() else {
            self.argv = Some(argv);
            return Err(self);
        };
        let Some((bundle, receipt)) = self.ready.take() else {
            self.argv = Some(argv);
            self.stdio = Some(stdio);
            return Err(self);
        };
        Ok(PreparedNoGameGrantOwnersV1 {
            argv,
            stdio,
            bundle,
            receipt,
        })
    }
}
