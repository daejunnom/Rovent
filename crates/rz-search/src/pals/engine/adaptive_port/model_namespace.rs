//! Frozen explicit bridge between a native producer's representation namespace
//! and Search's model-WDL namespace. Digest fields are not derived from display
//! strings: legacy V2 and virtual-review V3 use different model/encoding keys.
//!
//! The actual startup owner constructs this registration from retained loaded
//! metadata. This pure declaration checks consistency at consumption; it does
//! not attest weights, provider readiness, tensor contents or a physical Run.

use crate::pals::engine::RoleError;
use crate::pals::value::{MODEL_WDL_VALUE_SEMANTICS, ModelValueIdentity};
use rz_contracts::pals::{RepresentationKey, RoleOutput};
use rz_contracts::pals_adaptive::RoleTicketEnvelope;
use rz_contracts::{Digest, PrecisionProfile};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AdaptiveModelNamespace {
    value: ModelValueIdentity,
    model: Digest,
    encoding: Digest,
    precision: PrecisionProfile,
    /// Representation/cache epoch. This is neither process epoch nor the V5
    /// manifest's registered model epoch; each producer supplies its own value.
    frozen_epoch: u64,
}

impl AdaptiveModelNamespace {
    /// Register the actual producer's exact mapping once at startup. In V3,
    /// representation `model` is the loaded checkpoint hash, while its encoding
    /// key hashes the encoding name. The value identity's composite strings and
    /// registered encoding digest must not be substituted for those keys. V2
    /// instead uses an export-manifest model key. Never infer either mapping.
    pub fn new(
        value: ModelValueIdentity,
        model: Digest,
        encoding: Digest,
        precision: PrecisionProfile,
        frozen_epoch: u64,
    ) -> Result<Self, RoleError> {
        value.validate()?;
        if value.semantics != MODEL_WDL_VALUE_SEMANTICS {
            return Err(RoleError::InvalidOutput);
        }
        // These are the two existing explicit native FP32 value profiles.
        // PrecisionProfile does not encode TF32, so the complete string stays
        // pinned in `value` and is compared without normalization on delivery.
        if precision != PrecisionProfile::Fp32
            || !matches!(value.precision.as_str(), "fp32" | "fp32_tf32_off")
        {
            return Err(RoleError::Unsupported(
                "adaptive registered model namespace requires a supported FP32 profile",
            ));
        }
        Ok(Self {
            value,
            model,
            encoding,
            precision,
            frozen_epoch,
        })
    }

    pub fn value_identity(&self) -> &ModelValueIdentity {
        &self.value
    }

    /// Read-only representation namespace recorded by this owner. These
    /// values are not a ticket, native acceptance or physical completion.
    pub fn representation_identity(&self) -> (Digest, Digest, PrecisionProfile, u64) {
        (self.model, self.encoding, self.precision, self.frozen_epoch)
    }

    /// Comparison-only identity of the actual registered mapping. This is not
    /// a representation key, cache hit or authority to issue a model request.
    pub(super) fn utility_cohort_digest(&self) -> Digest {
        use sha2::{Digest as _, Sha256};
        let mut hash = Sha256::new();
        hash.update(b"rz-rovent-utility-cohort-model-namespace/1\0");
        for text in [
            &self.value.semantics,
            &self.value.model,
            &self.value.encoding,
            &self.value.precision,
        ] {
            hash.update((text.len() as u64).to_le_bytes());
            hash.update(text.as_bytes());
        }
        hash.update(self.value.model_epoch);
        hash.update(self.model.0);
        hash.update(self.encoding.0);
        match self.precision {
            PrecisionProfile::Fp32 => hash.update([0]),
            PrecisionProfile::Fp16 => hash.update([1]),
            PrecisionProfile::Bf16 => hash.update([2]),
            PrecisionProfile::Quantized(key) => {
                hash.update([3]);
                hash.update(key.0);
            }
        }
        hash.update(self.frozen_epoch.to_le_bytes());
        Digest(hash.finalize().into())
    }

    /// Used by the immutable metadata mirror's constructor and again before a
    /// store commit. Equal ticket/output keys alone cannot certify which loaded
    /// model or encoding issued them.
    pub fn verify_identity(
        &self,
        role_model: &str,
        registered: Option<&ModelValueIdentity>,
    ) -> Result<(), RoleError> {
        self.value.validate()?;
        if role_model != self.value.model || registered != Some(&self.value) {
            return Err(RoleError::InvalidOutput);
        }
        Ok(())
    }

    pub fn verify(
        &self,
        ticket: &RoleTicketEnvelope,
        output: &RoleOutput,
        registered: Option<&ModelValueIdentity>,
    ) -> Result<(), RoleError> {
        if output.id != ticket.id
            || output.authority != ticket.authority
            || output.key != ticket.representation
        {
            return Err(RoleError::InvalidOutput);
        }
        self.verify_representation(&ticket.representation, registered)
    }

    /// A coordinator ranking publication already retains the private native
    /// acceptance proof and the original output key. Validate that original
    /// ticket's registered mapping without constructing another role output.
    pub(super) fn verify_ticket(
        &self,
        ticket: &RoleTicketEnvelope,
        registered: Option<&ModelValueIdentity>,
    ) -> Result<(), RoleError> {
        self.verify_representation(&ticket.representation, registered)
    }

    fn verify_representation(
        &self,
        representation: &RepresentationKey,
        registered: Option<&ModelValueIdentity>,
    ) -> Result<(), RoleError> {
        self.value.validate()?;
        if registered != Some(&self.value)
            || representation.model != self.model
            || representation.encoding != self.encoding
            || representation.precision != self.precision
            || representation.frozen_epoch != self.frozen_epoch
        {
            return Err(RoleError::InvalidOutput);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rz_contracts::pals::Role;

    fn value() -> ModelValueIdentity {
        ModelValueIdentity {
            semantics: MODEL_WDL_VALUE_SEMANTICS.into(),
            model: "actual-loaded-export-and-backend-namespace".into(),
            encoding: "actual-registered-encoding-name-and-digest".into(),
            precision: "fp32_tf32_off".into(),
            model_epoch: [9; 32],
        }
    }

    fn key() -> RepresentationKey {
        RepresentationKey {
            input: Digest([1; 32]),
            history: Digest([2; 32]),
            candidates: Digest([3; 32]),
            mask_positions: Digest([4; 32]),
            model: Digest([5; 32]),
            encoding: Digest([6; 32]),
            precision: PrecisionProfile::Fp32,
            frozen_epoch: 1,
            record_revision: 7,
            role: Role::Proposer,
        }
    }

    fn namespace() -> AdaptiveModelNamespace {
        let representation = key();
        AdaptiveModelNamespace::new(
            value(),
            representation.model,
            representation.encoding,
            representation.precision,
            representation.frozen_epoch,
        )
        .unwrap()
    }

    #[test]
    fn explicit_mapping_preserves_distinct_v2_manifest_and_checkpoint_namespaces() {
        let namespace = namespace();
        // The generic bridge must not assume every producer uses V3's weights
        // hash as representation.model or hash a display string to replace it.
        assert_ne!(key().model.0, namespace.value_identity().model_epoch);
        namespace
            .verify_identity(&value().model, Some(&value()))
            .unwrap();
        namespace
            .verify_representation(&key(), Some(&value()))
            .unwrap();
    }

    #[test]
    fn all_registered_representation_fields_reject_equal_but_wrong_ticket_keys() {
        let namespace = namespace();
        for field in 0..4 {
            let mut representation = key();
            match field {
                0 => representation.model = Digest([99; 32]),
                1 => representation.encoding = Digest([99; 32]),
                2 => representation.precision = PrecisionProfile::Fp16,
                _ => representation.frozen_epoch += 1,
            }
            assert!(
                namespace
                    .verify_representation(&representation, Some(&value()))
                    .is_err()
            );
        }
    }

    #[test]
    fn delivery_compares_loaded_namespace_even_when_ticket_and_output_agree() {
        use super::super::port_tests::{output, setup, work};
        use crate::pals::coordinator::RefinementStage;
        use rz_contracts::RequestId;
        use rz_position::Position;

        let mut port = setup(Position::startpos(), 1);
        let work = work(&mut port, "e2e4", RefinementStage::ProposalValueFresh, 1);
        let (mut ticket, mut result) = output(&work, 1);
        let namespace = namespace();
        namespace.verify(&ticket, &result, Some(&value())).unwrap();

        ticket.representation.model = Digest([99; 32]);
        result.key = ticket.representation;
        assert_eq!(result.key, ticket.representation);
        assert!(namespace.verify(&ticket, &result, Some(&value())).is_err());

        ticket.representation = key();
        result.key = key();
        result.id = RequestId::new(ticket.id.epoch, ticket.id.sequence + 1);
        assert!(namespace.verify(&ticket, &result, Some(&value())).is_err());
    }

    #[test]
    fn full_value_identity_and_mirror_name_are_pinned_without_precision_aliasing() {
        let namespace = namespace();
        assert!(
            namespace
                .verify_identity("invented-mirror", Some(&value()))
                .is_err()
        );
        assert!(namespace.verify_representation(&key(), None).is_err());
        for field in 0..5 {
            let mut wrong = value();
            match field {
                0 => wrong.semantics.push_str("-other"),
                1 => wrong.model.push_str("-other"),
                2 => wrong.encoding.push_str("-other"),
                3 => wrong.precision = "fp32".into(),
                _ => wrong.model_epoch = [99; 32],
            }
            assert!(
                namespace
                    .verify_representation(&key(), Some(&wrong))
                    .is_err()
            );
        }
    }

    #[test]
    fn registration_rejects_unsupported_precision_and_wrong_value_semantics() {
        let representation = key();
        let mut wrong = value();
        wrong.semantics = "own-raw-cp".into();
        assert!(
            AdaptiveModelNamespace::new(
                wrong,
                representation.model,
                representation.encoding,
                representation.precision,
                representation.frozen_epoch
            )
            .is_err()
        );
        assert!(
            AdaptiveModelNamespace::new(
                value(),
                representation.model,
                representation.encoding,
                PrecisionProfile::Fp16,
                representation.frozen_epoch
            )
            .is_err()
        );
    }
}
