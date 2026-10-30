//! Explicit, separately versioned target-bundle utility research head.
//! These seven raw observations grant no retention, search, routing, WDL or
//! Rules authority. Inference cannot create a training target-known label.

use super::review::BoundNnuecReviewContext;
use super::{Context, Model, ReferenceInput, change, project, rounded};
use crate::cpu_value::nnuec::{NnuecBoard, NnuecError, NnuecLocal};
use crate::cpu_value::{CpuTrainingState, CpuValueIdentity};
use crate::pals::store::PalsStores;
use rz_position::{Color, Position};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::io::Read;
use std::sync::Arc;

pub const SCHEMA: &str = "rz-nnuec-utility-fp32-v1";
pub const SEMANTICS: &str = "nnuec-v2-targetbundle-risk-route-shadow-fp32-v1";
pub const INPUT_SCHEMA: &str = "rz-nnuec-utility-input/1";
pub const OUTPUT_SCHEMA: &str = "rz-nnuec-utility-output/1";
pub const OUTPUTS: usize = 7;
pub const MAX_BYTES: usize = 8 * 1024 * 1024;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Parameters {
    pub hidden_weights: Vec<f32>,
    pub hidden_bias: Vec<f32>,
    pub output_weights: Vec<f32>,
    pub output_bias: Vec<f32>,
}
impl Parameters {
    fn arrays(&self) -> [&Vec<f32>; 4] {
        [
            &self.hidden_weights,
            &self.hidden_bias,
            &self.output_weights,
            &self.output_bias,
        ]
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Checkpoint {
    pub schema: String,
    pub semantics: String,
    pub base_weights_sha256: String,
    /// Declared provenance, not independent proof of any optimizer execution.
    pub training: CpuTrainingState,
    pub hidden_width: usize,
    pub parameters: Parameters,
}
impl Checkpoint {
    pub fn validate(&self, base: &Model) -> Result<(), NnuecError> {
        if self.schema != SCHEMA
            || self.semantics != SEMANTICS
            || self.schema.capacity() > 256
            || self.semantics.capacity() > 256
            || self.base_weights_sha256.len() != 64
            || self.base_weights_sha256.capacity() > 128
            || base.identity().weights_sha256.as_deref() != Some(self.base_weights_sha256.as_str())
            || !(1..=64).contains(&self.hidden_width)
        {
            return Err(super::invalid(
                "utility schema/base identity/required width",
            ));
        }
        CpuValueIdentity {
            semantics: SEMANTICS.into(),
            weights_sha256: Some(self.base_weights_sha256.clone()),
            training: self.training.clone(),
        }
        .validate()
        .map_err(|_| super::invalid("utility training declaration"))?;
        let config = &base.base_checkpoint().config;
        let input = 2 * config.board_width + 2 * config.context_width + config.local_width;
        for (values, expected) in self.parameters.arrays().into_iter().zip([
            self.hidden_width * input,
            self.hidden_width,
            OUTPUTS * self.hidden_width,
            OUTPUTS,
        ]) {
            if values.len() != expected || values.capacity() > (expected * 2).max(4) {
                return Err(super::invalid("utility parameter shape/allocation"));
            }
            if values
                .iter()
                .any(|value| !value.is_finite() || value.abs() > 1000.0)
            {
                return Err(super::invalid("utility parameter finite/magnitude"));
            }
        }
        Ok(())
    }
    pub fn weights_sha256(&self, base: &Model) -> Result<String, NnuecError> {
        self.validate(base)?;
        let mut hash = Sha256::new();
        hash.update(SCHEMA.as_bytes());
        hash.update([0]);
        hash.update(SEMANTICS.as_bytes());
        hash.update([0]);
        hash.update(self.base_weights_sha256.as_bytes());
        hash.update([0]);
        hash.update((self.hidden_width as u32).to_le_bytes());
        for values in self.parameters.arrays() {
            for value in values {
                hash.update(value.to_bits().to_le_bytes());
            }
        }
        Ok(format!("{:x}", hash.finalize()))
    }
}

pub struct UtilityModel {
    base: Arc<Model>,
    checkpoint: Checkpoint,
    weights_sha256: String,
}
impl UtilityModel {
    pub fn new(base: Arc<Model>, checkpoint: Checkpoint) -> Result<Self, NnuecError> {
        let weights_sha256 = checkpoint.weights_sha256(&base)?;
        Ok(Self {
            base,
            checkpoint,
            weights_sha256,
        })
    }
    pub fn from_reader(base: Arc<Model>, reader: impl Read) -> Result<Self, NnuecError> {
        let mut bytes = Vec::new();
        reader
            .take((MAX_BYTES + 1) as u64)
            .read_to_end(&mut bytes)
            .map_err(|e| NnuecError::Io(e.to_string()))?;
        if bytes.len() > MAX_BYTES {
            return Err(NnuecError::Capacity("utility checkpoint bytes"));
        }
        let checkpoint =
            serde_json::from_slice(&bytes).map_err(|e| NnuecError::Json(e.to_string()))?;
        Self::new(base, checkpoint)
    }
    pub fn checkpoint(&self) -> &Checkpoint {
        &self.checkpoint
    }
    pub fn weights_sha256(&self) -> &str {
        &self.weights_sha256
    }

    /// Only named boundaries may request a shadow observation. No per-node
    /// policy integration or hidden selected-line replacement is performed.
    #[allow(clippy::too_many_arguments)] // Preserve the independently checked original input authorities.
    pub fn observe(
        &self,
        bound: &BoundNnuecReviewContext,
        stores: &PalsStores,
        board: &NnuecBoard,
        local: &NnuecLocal,
        position: &Position,
        target_record: u16,
        scope: ObservationScope,
    ) -> Result<Observation, NnuecError> {
        scope.validate()?;
        let provenance = bound.provenance();
        if !bound.validates(
            stores,
            provenance.root,
            provenance.generation,
            provenance.view_seal,
        ) || &provenance.model != self.base.identity()
        {
            return Err(NnuecError::StateMismatch);
        }
        // This observation is a distinct root utility question, not a neural
        // result for an arbitrary CPU suffix or the view's policy query.
        let root = bound.view().anchors()[0].snapshot.position_identity();
        if position.position_identity() != root || local.anchor != root || !local.moves.is_empty() {
            return Err(NnuecError::StateMismatch);
        }
        self.base.check_context(bound.context())?;
        self.base.cache_context(bound.context(), local, position)?;
        if board.model != self.base.base.identity || board.position != position.position_identity()
        {
            return Err(NnuecError::StateMismatch);
        }
        let metadata = bound.view().records_metadata();
        let target = metadata
            .get(usize::from(target_record))
            .ok_or(super::invalid("utility target record"))?;
        if target.original_first_move.is_none() {
            return Err(super::invalid("utility target has no original bundle move"));
        }
        let ids: Vec<_> = metadata
            .iter()
            .enumerate()
            .filter_map(|(index, record)| {
                (record.source_state == target.source_state
                    && record.original_first_move == target.original_first_move)
                    .then_some(index as u64)
            })
            .collect();
        let numeric = self.forward_reference(board, bound.context(), local, position, &ids)?;
        let context_sha256 = bound.context().sha256()?;
        let submission_target_sha256 =
            submission_target_seal(provenance.view_seal, &numeric.target_sha256);
        if !bound.validates(
            stores,
            provenance.root,
            provenance.generation,
            provenance.view_seal,
        ) {
            return Err(NnuecError::StateMismatch);
        }
        Ok(Observation {
            numeric,
            scope,
            view_seal: provenance.view_seal,
            full_view_digest: provenance.full_view_digest,
            context_sha256,
            submission_target_sha256,
            target_record_ids: ids,
            training_declaration: self.checkpoint.training.clone(),
        })
    }
    pub fn reference_input(
        &self,
        context: &Context,
        local: &NnuecLocal,
        position: &Position,
        target_ids: &[u64],
    ) -> Result<Reference, NnuecError> {
        self.base.check_context(context)?;
        target_contribution(context, target_ids)?;
        Ok(Reference {
            schema: INPUT_SCHEMA.into(),
            base_input: self.base.reference_input(context, local, position)?,
            target_ids: target_ids.to_vec(),
        })
    }
    /// CPU numerical comparison only. This checks exact Rust input ownership,
    /// but does not grant submitted-view, target-label or policy authority.
    pub fn forward_reference(
        &self,
        board: &NnuecBoard,
        context: &Context,
        local: &NnuecLocal,
        position: &Position,
        target_ids: &[u64],
    ) -> Result<NumericOutput, NnuecError> {
        self.base.check_context(context)?;
        self.base.cache_context(context, local, position)?;
        if board.model != self.base.base.identity || board.position != position.position_identity()
        {
            return Err(NnuecError::StateMismatch);
        }
        let target = target_contribution(context, target_ids)?;
        let mut output = self.numeric(
            &board.values,
            context.values(),
            &local.projected,
            &target,
            position.side_to_move(),
        )?;
        let context_sha256 = context.sha256()?;
        output.target_sha256 = target_seal(&context_sha256, target_ids);
        output.input_sha256 = numeric_input_seal(
            &self.reference_input(context, local, position, target_ids)?,
            &self.checkpoint.base_weights_sha256,
            &self.weights_sha256,
            &context_sha256,
        );
        Ok(output)
    }
    fn numeric(
        &self,
        board: &[Vec<f32>; 2],
        context: &[f32],
        local: &[f32],
        target: &[f32],
        side: Color,
    ) -> Result<NumericOutput, NnuecError> {
        let config = &self.base.base_checkpoint().config;
        if board
            .iter()
            .any(|values| values.len() != config.board_width)
            || context.len() != config.context_width
            || target.len() != config.context_width
            || local.len() != config.local_width
            || board
                .iter()
                .flatten()
                .chain(context)
                .chain(local)
                .chain(target)
                .any(|value| !value.is_finite())
        {
            return Err(NnuecError::StateMismatch);
        }
        let side = side as usize;
        let input: Vec<_> = board[side]
            .iter()
            .chain(&board[1 - side])
            .chain(context)
            .chain(local)
            .chain(target)
            .map(|value| value.clamp(0.0, 1.0))
            .collect();
        let mut hidden = project(
            &input,
            &self.checkpoint.parameters.hidden_weights,
            &self.checkpoint.parameters.hidden_bias,
        )?;
        for value in &mut hidden {
            *value = value.max(0.0);
        }
        let raw_outputs: [f32; OUTPUTS] = project(
            &hidden,
            &self.checkpoint.parameters.output_weights,
            &self.checkpoint.parameters.output_bias,
        )?
        .try_into()
        .map_err(|_| NnuecError::StateMismatch)?;
        Ok(NumericOutput {
            schema: OUTPUT_SCHEMA.into(),
            weights_sha256: self.weights_sha256.clone(),
            base_weights_sha256: self.checkpoint.base_weights_sha256.clone(),
            hidden,
            raw_outputs,
            target_contribution: target.to_vec(),
            policy_actions_authorized: false,
            target_known_mask: [false; OUTPUTS],
            target_sha256: String::new(),
            input_sha256: String::new(),
        })
    }
}

/// Caller-supplied provenance is carried verbatim after bounded validation.
/// Unknown teacher identity remains None; inference does not invent one.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObservationScope {
    pub checker_profile: String,
    pub collection_profile: String,
    pub teacher_profile: Option<String>,
    pub boundary: Boundary,
}
impl ObservationScope {
    fn validate(&self) -> Result<(), NnuecError> {
        for value in [&self.checker_profile, &self.collection_profile]
            .into_iter()
            .chain(self.teacher_profile.iter())
        {
            if value.trim().is_empty()
                || value.len() > 128
                || value.capacity() > 256
                || value.chars().any(char::is_control)
            {
                return Err(super::invalid("utility provenance namespace"));
            }
        }
        Ok(())
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Boundary {
    ImportantResult,
    ContextPressure,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Reference {
    pub schema: String,
    pub base_input: ReferenceInput,
    pub target_ids: Vec<u64>,
}
#[derive(Clone, Debug, Serialize)]
pub struct NumericOutput {
    pub schema: String,
    pub weights_sha256: String,
    pub base_weights_sha256: String,
    pub hidden: Vec<f32>,
    /// retain, drop, revisit, missing-response risk, route P, route C, route R.
    pub raw_outputs: [f32; OUTPUTS],
    pub target_contribution: Vec<f32>,
    pub target_sha256: String,
    /// Actual B/C/L/target input identity, distinct from the parent view/query.
    pub input_sha256: String,
    pub policy_actions_authorized: bool,
    pub target_known_mask: [bool; OUTPUTS],
}
#[derive(Clone, Debug, Serialize)]
pub struct Observation {
    #[serde(flatten)]
    pub numeric: NumericOutput,
    pub scope: ObservationScope,
    pub view_seal: [u8; 32],
    pub full_view_digest: [u8; 32],
    pub context_sha256: String,
    pub submission_target_sha256: String,
    pub target_record_ids: Vec<u64>,
    pub training_declaration: CpuTrainingState,
}

fn target_contribution(context: &Context, ids: &[u64]) -> Result<Vec<f32>, NnuecError> {
    if ids.is_empty() || ids.len() > context.lines.len() {
        return Err(super::invalid("utility target capacity"));
    }
    let selected: BTreeSet<_> = ids.iter().copied().collect();
    if selected.len() != ids.len() {
        return Err(super::invalid("utility duplicate target record"));
    }
    let first = context
        .lines
        .get(&ids[0])
        .ok_or(super::invalid("utility missing target record"))?;
    if first.original_first_move.is_none() {
        return Err(super::invalid("utility target bundle move"));
    }
    for id in ids {
        let line = context
            .lines
            .get(id)
            .ok_or(super::invalid("utility missing target record"))?;
        if line.anchor() != first.anchor() || line.original_first_move != first.original_first_move
        {
            return Err(super::invalid("utility target mixed bundle"));
        }
    }
    let mut sum = vec![super::Exact256::default(); context.values.len()];
    for id in ids {
        change(&mut sum, &context.lines[id].contribution, false)?;
    }
    for (edge, pair) in &context.edges {
        let source = selected.contains(&edge.source);
        let target = selected.contains(&edge.target);
        if source != target {
            return Err(super::invalid("utility target crosses relation"));
        }
        if source {
            change(&mut sum, &pair.contribution, false)?;
        }
    }
    rounded(&sum)
}
fn target_seal(context_sha256: &str, ids: &[u64]) -> String {
    let mut hash = Sha256::new();
    hash.update(INPUT_SCHEMA.as_bytes());
    hash.update(b"\0target\0");
    hash.update(context_sha256.as_bytes());
    hash.update([0]);
    hash.update((ids.len() as u32).to_le_bytes());
    for id in ids {
        hash.update(id.to_le_bytes());
    }
    format!("{:x}", hash.finalize())
}
fn submission_target_seal(view: [u8; 32], target_sha256: &str) -> String {
    let mut hash = Sha256::new();
    hash.update(b"rz-nnuec-shadow-submission-target/1\0");
    hash.update(view);
    hash.update(target_sha256.as_bytes());
    format!("{:x}", hash.finalize())
}
fn numeric_input_seal(input: &Reference, base: &str, utility: &str, context: &str) -> String {
    let mut hash = Sha256::new();
    hash.update(INPUT_SCHEMA.as_bytes());
    hash.update(b"\0numeric\0");
    for digest in [base, utility, context] {
        hash.update(digest.as_bytes());
        hash.update([0]);
    }
    let features = |hash: &mut Sha256, values: &[usize]| {
        hash.update((values.len() as u32).to_le_bytes());
        for value in values {
            hash.update((*value as u16).to_le_bytes());
        }
    };
    for values in &input.base_input.board_features {
        features(&mut hash, values);
    }
    hash.update([u8::from(input.base_input.side_to_move == "black")]);
    features(&mut hash, &input.base_input.local_line.anchor_features);
    hash.update([u8::from(input.base_input.local_line.perspective == "black")]);
    hash.update((input.base_input.local_line.moves.len() as u32).to_le_bytes());
    for movement in &input.base_input.local_line.moves {
        hash.update([movement.from_square, movement.to_square, movement.promotion]);
    }
    hash.update((input.target_ids.len() as u32).to_le_bytes());
    for id in &input.target_ids {
        hash.update(id.to_le_bytes());
    }
    format!("{:x}", hash.finalize())
}

#[cfg(test)]
mod tests {
    use super::super::{EdgeKind, LineKind};
    use super::*;
    fn base() -> Arc<Model> {
        Arc::new(Model::new(super::super::tests::checkpoint()).unwrap())
    }
    fn checkpoint(base: &Model) -> Checkpoint {
        Checkpoint {
            schema: SCHEMA.into(),
            semantics: SEMANTICS.into(),
            base_weights_sha256: base.identity().weights_sha256.clone().unwrap(),
            training: CpuTrainingState::Untrained,
            hidden_width: 1,
            parameters: Parameters {
                hidden_weights: vec![0.0, 0.0, 0.0, 0.0, 1.0],
                hidden_bias: vec![0.0],
                output_weights: vec![1.0; 7],
                output_bias: vec![0.0; 7],
            },
        }
    }
    #[test]
    fn independent_utility_digest_shapes_and_base_model_are_required() {
        let base = base();
        let original = checkpoint(&base);
        let hash = original.weights_sha256(&base).unwrap();
        let mut changed = original.clone();
        changed.parameters.output_bias[3] = 0.5;
        assert_ne!(hash, changed.weights_sha256(&base).unwrap());
        changed = original.clone();
        changed.base_weights_sha256 = "0".repeat(64);
        assert!(changed.validate(&base).is_err());
        changed = original.clone();
        changed.hidden_width = 0;
        assert!(changed.validate(&base).is_err());
        changed = original;
        changed.parameters.output_weights.pop();
        assert!(changed.validate(&base).is_err());
    }
    #[test]
    fn target_input_and_missing_risk_are_raw_shadow_without_training_targets() {
        let base = base();
        let model = UtilityModel::new(Arc::clone(&base), checkpoint(&base)).unwrap();
        let low = model
            .numeric(
                &[vec![0.0], vec![0.0]],
                &[0.0],
                &[0.0],
                &[0.25],
                Color::White,
            )
            .unwrap();
        let high = model
            .numeric(
                &[vec![0.0], vec![0.0]],
                &[0.0],
                &[0.0],
                &[0.75],
                Color::White,
            )
            .unwrap();
        assert_eq!(low.raw_outputs[3], 0.25);
        assert_eq!(high.raw_outputs[3], 0.75);
        assert!(!high.policy_actions_authorized);
        assert_eq!(high.target_known_mask, [false; 7]);
        assert!(
            model
                .numeric(
                    &[vec![f32::NAN], vec![0.0]],
                    &[0.0],
                    &[0.0],
                    &[0.0],
                    Color::White
                )
                .is_err()
        );
    }
    #[test]
    fn coherent_target_is_bias_free_and_incomplete_bundle_is_rejected() {
        let base = base();
        let root = Position::startpos();
        let mut context = base.empty_context();
        for (id, kind, text) in [
            (0, LineKind::Proposal, "e2e4"),
            (1, LineKind::Counterexample, "e2e4 e7e5"),
            (2, LineKind::Proposal, "d2d4"),
        ] {
            let moves = text
                .split_whitespace()
                .map(|s| s.parse().unwrap())
                .collect::<Vec<_>>();
            context = base
                .with_line(&context, id, base.encode_line(&root, &moves, kind).unwrap())
                .unwrap();
        }
        context = base
            .with_edge(
                &context,
                super::super::Edge {
                    source: 0,
                    target: 1,
                    kind: EdgeKind::Parent,
                },
            )
            .unwrap();
        let target = target_contribution(&context, &[0, 1]).unwrap();
        let removed = base.without_bundle(&context, &[0, 1]).unwrap();
        let restored = super::super::add(target[0], removed.values()[0]).unwrap();
        assert_eq!(restored.to_bits(), context.values()[0].to_bits());
        assert!(target_contribution(&context, &[0]).is_err());
        assert!(target_contribution(&context, &[0, 1, 2]).is_err());
        assert!(target_contribution(&context, &[0, 0]).is_err());
        assert!(target_contribution(&context, &vec![0; 129]).is_err());
    }
    #[test]
    fn namespace_does_not_invent_unknown_teacher_or_accept_blank_profiles() {
        let mut scope = ObservationScope {
            checker_profile: "own-r".into(),
            collection_profile: "untrained-shadow".into(),
            teacher_profile: None,
            boundary: Boundary::ImportantResult,
        };
        scope.validate().unwrap();
        assert!(scope.teacher_profile.is_none());
        scope.checker_profile = " ".into();
        assert!(scope.validate().is_err());
    }

    #[test]
    fn numeric_input_seal_distinguishes_actual_local_suffix_from_parent_view_scope() {
        let base = base();
        let model = UtilityModel::new(Arc::clone(&base), checkpoint(&base)).unwrap();
        let mut position = Position::startpos();
        let line = base
            .encode_line(&position, &["e2e4".parse().unwrap()], LineKind::Proposal)
            .unwrap();
        let context = base.with_line(&base.empty_context(), 0, line).unwrap();
        let mut local = base.local(&position).unwrap();
        let before = model
            .forward_reference(
                &base.board(&position).unwrap(),
                &context,
                &local,
                &position,
                &[0],
            )
            .unwrap();
        let movement = "e2e4".parse().unwrap();
        base.append_local(&mut local, movement).unwrap();
        position.make_move(movement).unwrap();
        let after = model
            .forward_reference(
                &base.board(&position).unwrap(),
                &context,
                &local,
                &position,
                &[0],
            )
            .unwrap();
        assert_eq!(before.target_sha256, after.target_sha256);
        assert_ne!(before.input_sha256, after.input_sha256);
        assert_eq!(before.input_sha256.len(), 64);
        assert!(!after.policy_actions_authorized);
    }
}
