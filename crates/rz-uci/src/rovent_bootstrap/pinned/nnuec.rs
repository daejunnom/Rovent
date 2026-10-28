//! NNUEC raw artifacts held by the same original startup directory owner.
//! These are file-byte pins; parsed model/semantic identities are separate.

use super::*;
use crate::rovent_bootstrap::nnuec_source::{NNUEC_PROVENANCE_MAX_BYTES, NNUEC_WEIGHTS_MAX_BYTES};

/// Original weights declaration/bytes followed by provenance declaration/bytes.
/// This publication preserves raw source identities, not loaded model authority.
pub(crate) type NnuecWeightsAndProvenanceBytes = (ArtifactRef, Vec<u8>, ArtifactRef, Vec<u8>);

pub(super) struct PinnedNnuecArtifacts {
    weights: PinnedBlob,
    provenance: PinnedBlob,
}
impl PinnedNnuecArtifacts {
    pub(super) fn open(
        root: &Dir,
        pin: Option<&rz_experiments::RoventNnuecPin>,
        control: &RoventStartupControl,
    ) -> Result<Option<Self>, BackendError> {
        let Some(pin) = pin else {
            return Ok(None);
        };
        control.check()?;
        if pin.weights.path == pin.provenance.path {
            return Err(fail(
                K::IdentityMismatch,
                "NNUEC weights/provenance cannot alias one input path",
            ));
        }
        let weights = open_blob(root, pin.weights.clone(), NNUEC_WEIGHTS_MAX_BYTES as u64)?;
        control.check()?;
        let provenance = open_blob(
            root,
            pin.provenance.clone(),
            NNUEC_PROVENANCE_MAX_BYTES as u64,
        )?;
        control.check()?;
        Ok(Some(Self {
            weights,
            provenance,
        }))
    }

    pub(super) fn read(
        &self,
        pin: &rz_experiments::RoventNnuecPin,
    ) -> Result<NnuecWeightsAndProvenanceBytes, BackendError> {
        if self.weights.declaration != pin.weights || self.provenance.declaration != pin.provenance
        {
            return Err(fail(
                K::IdentityMismatch,
                "retained NNUEC artifacts differ from the original lock",
            ));
        }
        // The existing positional reader verifies original metadata, bounds,
        // byte count and raw SHA before returning either serialized artifact.
        let weights = read_blob(&self.weights, NNUEC_WEIGHTS_MAX_BYTES)?;
        let provenance = read_blob(&self.provenance, NNUEC_PROVENANCE_MAX_BYTES)?;
        Ok((
            self.weights.declaration.clone(),
            weights,
            self.provenance.declaration.clone(),
            provenance,
        ))
    }
}

impl RoventPinnedNativeSources {
    /// Bounded publication/load read from the original retained descriptors.
    /// A named replacement cannot supply a different model or provenance.
    pub(crate) fn nnuec_artifact_bytes(
        &self,
    ) -> Result<Option<NnuecWeightsAndProvenanceBytes>, BackendError> {
        match (&self.nnuec, &self.lock.manifest.engine.nnuec) {
            (None, None) => Ok(None),
            (Some(held), Some(pin)) => held.read(pin).map(Some),
            _ => Err(fail(
                K::IdentityMismatch,
                "NNUEC retained source presence differs from lock",
            )),
        }
    }
}
