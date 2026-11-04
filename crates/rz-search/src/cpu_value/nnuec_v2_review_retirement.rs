//! Cleanup commands for the same C-preparation owner. This is a child of
//! review, so actual private actor custody does not become a public setter.

use super::{Command, PreparationActor, PreparationError};
use crate::cpu_value::nnuec::v2::retirement::{NnuecCacheInvalidation, NnuecCacheReleaseAck};
use crate::cpu_worker::{
    MaterializedNnuecEngine, NnuecEngineRetirementCompletion, NnuecEngineRetirementId,
    NnuecMaterializationReleaseCompletion, NnuecMaterializationReleaseId, RecoveredNnuecEngine,
};
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::sync::mpsc::{TryRecvError, TrySendError};

impl PreparationActor {
    /// This distinct command owns a never-started materialization, not a
    /// joined worker. Every rejection returns that same unique engine/Bound.
    #[allow(clippy::result_large_err)]
    pub fn try_release_materialized(
        &self,
        engine: MaterializedNnuecEngine,
    ) -> Result<NnuecMaterializationReleaseId, (PreparationError, MaterializedNnuecEngine)> {
        if self.stopping.load(Ordering::Acquire) || self.sender.is_none() {
            return Err((PreparationError::WorkerStopped, engine));
        }
        if self
            .outstanding
            .compare_exchange(0, 4, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return Err((PreparationError::Busy, engine));
        }
        let sequence = match self.retirement_sequence.fetch_update(
            Ordering::AcqRel,
            Ordering::Acquire,
            |value| value.checked_add(1),
        ) {
            Ok(previous) => previous + 1,
            Err(_) => {
                self.outstanding.store(0, Ordering::Release);
                return Err((
                    PreparationError::InvalidView("release sequence exhausted"),
                    engine,
                ));
            }
        };
        let id =
            NnuecMaterializationReleaseId::new(Arc::clone(&self.retirement_authority), sequence);
        match self
            .sender
            .as_ref()
            .expect("live actor sender")
            .try_send(Command::ReleaseMaterialized(id.clone(), engine))
        {
            Ok(()) => Ok(id),
            Err(error) => {
                self.outstanding.store(0, Ordering::Release);
                match error {
                    TrySendError::Full(Command::ReleaseMaterialized(_, engine)) => {
                        Err((PreparationError::Busy, engine))
                    }
                    TrySendError::Disconnected(Command::ReleaseMaterialized(_, engine)) => {
                        Err((PreparationError::WorkerStopped, engine))
                    }
                    _ => unreachable!("only an unstarted release was submitted"),
                }
            }
        }
    }
    pub fn try_released_materialized(
        &self,
    ) -> Result<Option<NnuecMaterializationReleaseCompletion>, PreparationError> {
        match self.released_materialized.try_recv() {
            Ok(completion) => {
                self.outstanding.store(0, Ordering::Release);
                Ok(Some(completion))
            }
            Err(TryRecvError::Empty) => Ok(None),
            Err(TryRecvError::Disconnected) => Err(PreparationError::WorkerStopped),
        }
    }

    /// Admission shares the single actor seat with preparation and initial
    /// materialization. On every failure the same unique engine is returned.
    /// The manager neither destroys its TT nor replaces its evaluator.
    #[allow(clippy::result_large_err)]
    pub fn try_retire_engine(
        &self,
        owner: RecoveredNnuecEngine,
    ) -> Result<NnuecEngineRetirementId, (PreparationError, RecoveredNnuecEngine)> {
        if self.stopping.load(Ordering::Acquire) || self.sender.is_none() {
            return Err((PreparationError::WorkerStopped, owner));
        }
        if self
            .outstanding
            .compare_exchange(0, 3, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return Err((PreparationError::Busy, owner));
        }
        let sequence = match self.retirement_sequence.fetch_update(
            Ordering::AcqRel,
            Ordering::Acquire,
            |value| value.checked_add(1),
        ) {
            Ok(previous) => previous + 1,
            Err(_) => {
                self.outstanding.store(0, Ordering::Release);
                return Err((
                    PreparationError::InvalidView("retirement sequence exhausted"),
                    owner,
                ));
            }
        };
        let id = NnuecEngineRetirementId::new(Arc::clone(&self.retirement_authority), sequence);
        match self
            .sender
            .as_ref()
            .expect("live actor sender")
            .try_send(Command::RetireEngine(id.clone(), owner))
        {
            Ok(()) => Ok(id),
            Err(error) => {
                self.outstanding.store(0, Ordering::Release);
                match error {
                    TrySendError::Full(Command::RetireEngine(_, owner)) => {
                        Err((PreparationError::Busy, owner))
                    }
                    TrySendError::Disconnected(Command::RetireEngine(_, owner)) => {
                        Err((PreparationError::WorkerStopped, owner))
                    }
                    _ => unreachable!("only a retirement command was submitted"),
                }
            }
        }
    }

    /// The original one-use terminal retains the actor seat until collected.
    /// A failed terminal still owns the same engine and can be quarantined or
    /// retried on its original actor without inventing a release receipt.
    pub fn try_retired_engine(
        &self,
    ) -> Result<Option<NnuecEngineRetirementCompletion>, PreparationError> {
        match self.retired.try_recv() {
            Ok(completion) => {
                self.outstanding.store(0, Ordering::Release);
                Ok(Some(completion))
            }
            Err(TryRecvError::Empty) => Ok(None),
            Err(TryRecvError::Disconnected) => Err(PreparationError::WorkerStopped),
        }
    }

    /// Revokes original Bound authority immediately, but cache release is
    /// acknowledged only after that exact actor actually observes the epoch.
    pub fn invalidate_context_cache(&self) -> Result<NnuecCacheInvalidation, PreparationError> {
        let epoch = self.invalidate_cache()?;
        Ok(NnuecCacheInvalidation::new(
            Arc::clone(&self.invalidations),
            epoch,
        ))
    }

    pub fn try_context_cache_release(
        &self,
        ticket: &NnuecCacheInvalidation,
    ) -> Result<Option<NnuecCacheReleaseAck>, PreparationError> {
        if !ticket.matches_actor(&self.invalidations) {
            return Err(PreparationError::InvalidView("foreign cache invalidation"));
        }
        if self.cache_invalidated_through() < ticket.epoch() {
            return Ok(None);
        }
        ticket.claim().map(Some)
    }
}
