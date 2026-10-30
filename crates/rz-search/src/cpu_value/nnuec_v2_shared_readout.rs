//! Optional R-free readout on the existing context preparation actor.
//! No CPU engine, TT, node ledger or search stack is constructed here.

use super::*;
use sha2::{Digest as _, Sha256};

pub(super) struct SharedReadoutTarget {
    endpoint: PositionSnapshot,
}

/// Actual actor output over one admitted immutable view and its exact query.
/// This is a conditional numerical observation, never a role WDL, mate proof,
/// evaluator installation, retention policy or learned target.
pub struct SharedNnuecReadout {
    request: RequestIdentity,
    endpoint: PositionSnapshot,
    output: super::super::Output,
    input_sha256: String,
}
impl SharedNnuecReadout {
    pub fn output(&self) -> &super::super::Output {
        &self.output
    }
    pub fn input_sha256(&self) -> &str {
        &self.input_sha256
    }
    pub fn endpoint(&self) -> &PositionSnapshot {
        &self.endpoint
    }
    pub(crate) fn matches_request(&self, original: &RequestIdentity) -> bool {
        self.request.same_request(original)
    }
}

impl Request {
    /// Semantic owners choose the exact endpoint. Replay/encoding/forward do
    /// not run on the manager; the original actor checks the entire prefix.
    #[allow(clippy::result_large_err)]
    pub(crate) fn with_shared_readout(
        mut self,
        endpoint: &PositionSnapshot,
    ) -> Result<Self, (PreparationError, Self)> {
        let checked = (|| {
            check_control(self.deadline, &self.cancel)?;
            if self.shared_readout.is_some()
                || self.utility_target.is_some()
                || self.utility_cache_prelude.is_some()
                || self.view.query().prefix.len() > MAX_UCI_REPLAY_PLIES
            {
                return Err(PreparationError::InvalidView(
                    "exclusive bounded shared readout",
                ));
            }
            Ok(())
        })();
        if let Err(error) = checked {
            return Err((error, self));
        }
        self.shared_readout = Some(SharedReadoutTarget {
            endpoint: endpoint.clone(),
        });
        Ok(self)
    }
    pub(crate) fn has_shared_readout(&self) -> bool {
        self.shared_readout.is_some()
    }
}

impl BoundNnuecReviewContext {
    pub fn shared_readout(&self) -> Option<&SharedNnuecReadout> {
        self.shared_readout.as_ref()
    }
}

impl Factory {
    fn shared_readout_control(
        &self,
        bound: &BoundNnuecReviewContext,
    ) -> Result<(), PreparationError> {
        match bound.authority_revocation() {
            Some(NnuecAuthorityRevocation::Cancelled) => Err(PreparationError::Cancelled),
            Some(NnuecAuthorityRevocation::Expired) => Err(PreparationError::Expired),
            Some(NnuecAuthorityRevocation::ActorStopped) => Err(PreparationError::WorkerStopped),
            Some(NnuecAuthorityRevocation::Invalidated) => Err(PreparationError::StaleView),
            None => Ok(()),
        }
    }
    pub(super) fn shared_readout(
        &self,
        bound: &BoundNnuecReviewContext,
        target: SharedReadoutTarget,
    ) -> Result<SharedNnuecReadout, PreparationError> {
        self.shared_readout_control(bound)?;
        if !bound.matches_preparation_actor(&self.stopping, &self.invalidations)
            || bound.provenance().model != *self.model.identity()
        {
            return Err(PreparationError::StaleView);
        }
        let root = &bound.view().anchors()[0].snapshot;
        let prefix = &bound.view().query().prefix;
        let mut position = Position::from_snapshot(
            root,
            rz_position::PositionLimits {
                max_history_positions: root
                    .known_history_len()
                    .checked_add(prefix.len())
                    .ok_or(PreparationError::InvalidView(
                        "shared readout history capacity",
                    ))?
                    .max(1),
                ..Default::default()
            },
        )
        .map_err(|_| PreparationError::InvalidView("shared readout root Rules"))?;
        let mut local = self.model.local(&position)?;
        for movement in prefix {
            self.shared_readout_control(bound)?;
            self.model.append_local(&mut local, *movement)?;
            position
                .make_move(*movement)
                .map_err(|_| PreparationError::InvalidView("shared readout query prefix"))?;
        }
        if !position.snapshot().same_state(&target.endpoint) {
            return Err(PreparationError::InvalidView(
                "shared readout endpoint Rules mismatch",
            ));
        }
        self.shared_readout_control(bound)?;
        let board = self.model.board(&position)?;
        let output = self
            .model
            .forward(&board, bound.context(), &local, &position)?;
        let input = self
            .model
            .reference_input(bound.context(), &local, &position)?;
        let bytes = serde_json::to_vec(&input)
            .map_err(|error| PreparationError::Neural(NnuecError::Json(error.to_string())))?;
        let mut hash = Sha256::new();
        hash.update(b"rz-rovent-pc-shared-nnuec-readout-input/1\0");
        hash.update((bytes.len() as u64).to_le_bytes());
        hash.update(bytes);
        self.shared_readout_control(bound)?;
        Ok(SharedNnuecReadout {
            request: bound.request_identity(),
            endpoint: target.endpoint,
            output,
            input_sha256: format!("{:x}", hash.finalize()),
        })
    }
}

impl PreparationActor {
    /// Read-only thread observation; it is not a source or execution capability.
    pub fn worker_thread_id(&self) -> Option<std::thread::ThreadId> {
        self.worker.as_ref().map(|worker| worker.thread().id())
    }
    pub(crate) fn verifies_shared_cache_release(
        &self,
        model: &Arc<Model>,
        release: &super::super::retirement::NnuecCacheReleaseAck,
    ) -> bool {
        std::sync::Weak::ptr_eq(&self.original_model, &Arc::downgrade(model))
            && self.outstanding.load(Ordering::Acquire) == 0
            && release.prepares(
                &self.invalidations,
                self.invalidations.load(Ordering::Acquire),
            )
            && self.cache_invalidated_through() >= release.epoch()
    }
}
