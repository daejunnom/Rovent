//! 실제 NNUEC R capture만 별도 R 자료로 옮기는 producer다.
//! 원 readout·퇴역·loaded source를 빌리며 평가를 다시 수행하지 않는다.
//! P/C native receipt, 소비 권한, whole-action 비용, WDL/utility를 발명하지 않는다.
use super::*;
use rz_contracts::pals_adaptive::CpuJobEnvelope;
use rz_experiments::rovent_r_readout::*;
use rz_search::cpu_value::nnuec::v2::capture::{
    NnuecCaptureEnding, NnuecCaptureLineage, NnuecCaptureReservation, NnuecCaptureRetirement,
    NnuecCapturedReadout,
};
use rz_uci::rovent_bootstrap::nnuec_source::LoadedRoventNnuecSource;
use std::sync::Arc;

#[derive(Clone, Debug)]
pub struct RoventRCollectionContext {
    pub game_id: String,
    pub opening_id: String,
    pub line_genealogy_id: String,
    pub actor_id: String,
    pub teacher_actor_id: Option<String>,
    pub ancestor_origin_sha256: Vec<String>,
    pub pc_genealogy: Option<RoventRPcGenealogy>,
}

/// Public wire/flags cannot construct this original-retirement data custody.
/// A row without an actual Rules target is raw observation, not a valid NNUEC
/// optimizer sample. No live/physical source capability can be recovered from it.
pub struct PublishedRoventRReadout {
    record: RoventRReadoutRecord,
    admission: RoventCollectionRootAdmission,
    source: Arc<LoadedRoventNnuecSource>,
}
impl PublishedRoventRReadout {
    pub fn record(&self) -> &RoventRReadoutRecord {
        &self.record
    }
    pub fn admission(&self) -> &RoventCollectionRootAdmission {
        &self.admission
    }
    pub fn nnuec_source(&self) -> &Arc<LoadedRoventNnuecSource> {
        &self.source
    }
}

/// Empty/error/overflow terminal observations are retained separately; they
/// are never manufactured into a zero-input training row.
pub struct CapturedRoventRRetirement {
    lineage: NnuecCaptureLineage,
    manifest_sha256: String,
    retirement: RoventRCaptureRetirement,
    retirement_sha256: String,
    rows: Vec<PublishedRoventRReadout>,
    admission: RoventCollectionRootAdmission,
    source: Arc<LoadedRoventNnuecSource>,
}
impl CapturedRoventRRetirement {
    /// 원 worker가 활성화한 같은 예약의 작은 lineage만 대조한다.
    pub fn matches_reservation(&self, reservation: &NnuecCaptureReservation) -> bool {
        self.lineage.matches_reservation(reservation)
    }
    pub fn manifest_sha256(&self) -> &str {
        &self.manifest_sha256
    }
    pub fn retirement(&self) -> &RoventRCaptureRetirement {
        &self.retirement
    }
    pub fn retirement_sha256(&self) -> &str {
        &self.retirement_sha256
    }
    pub fn rows(&self) -> &[PublishedRoventRReadout] {
        &self.rows
    }
    pub fn admission(&self) -> &RoventCollectionRootAdmission {
        &self.admission
    }
    pub fn nnuec_source(&self) -> &Arc<LoadedRoventNnuecSource> {
        &self.source
    }
}

pub struct RoventRReadoutCollector {
    manifest: RoventRReadoutManifest,
    consumed: BTreeSet<(u64, u64, u64, u64, u64)>,
}
impl RoventRReadoutCollector {
    /// Declaration only: this constructor issues no observation/source owner.
    pub fn new(manifest: RoventRReadoutManifest) -> Result<Self, ArenaError> {
        manifest.validate()?;
        Ok(Self {
            manifest,
            consumed: BTreeSet::new(),
        })
    }
    pub fn manifest(&self) -> &RoventRReadoutManifest {
        &self.manifest
    }

    /// Called outside the manager after original worker terminal/ack. Missing
    /// NotStarted retirement is reported by the caller; it must not wait for an
    /// absent OnceLock or turn it into an empty successful R evaluation.
    pub fn capture_retirement(
        &mut self,
        admission: &RoventCollectionRootAdmission,
        source: Arc<LoadedRoventNnuecSource>,
        retirement: &NnuecCaptureRetirement,
        context: &RoventRCollectionContext,
    ) -> Result<CapturedRoventRRetirement, ArenaError> {
        let job = retirement.job();
        let frame = retirement.frame();
        let provenance = retirement.provenance();
        check(
            job.id.authority == admission.authority()
                && frame.authority == admission.authority()
                && job.question == retirement.question_digest()
                && source.input_lock_digest() == admission.root_admission().startup_lock_digest()
                && retirement.matches_model(source.model())
                && provenance.model == *source.model().identity()
                && provenance.generation == admission.authority().root.0
                && provenance.root.0 as u64
                    == admission.raw_receipt()["store_root_state"]
                        .as_u64()
                        .unwrap_or(u64::MAX)
                && retirement.context_view().seal_digest() == provenance.view_seal
                && retirement.context_view().digest() == provenance.full_view_digest
                && retirement.original_deadline().clock
                    == admission.root_admission().hard_deadline().clock
                && retirement.original_deadline().at
                    <= admission.root_admission().hard_deadline().at
                && job.hard_deadline.clock == retirement.original_deadline().clock
                && job.hard_deadline.at <= retirement.original_deadline().at,
            "actual R retirement does not join original root/loaded model/Job/Q/Bound",
        )?;
        check(
            self.manifest.collection.source_sha256
                == admission.native_origin().raw_receipt()["source_sha256"]
                && self.manifest.collection.runtime_sha256
                    == admission.native_origin().raw_receipt()["runtime_sha256"],
            "R corpus belongs to different original source/runtime",
        )?;
        let occurrence = (
            job.id.authority.epoch.0,
            job.id.authority.game.0,
            job.id.authority.root.0,
            job.id.sequence,
            retirement.capture_sequence(),
        );
        check(
            !self.consumed.contains(&occurrence) && self.consumed.len() < MAX_CORPUS_ROWS,
            "duplicate/unbounded original R retirement",
        )?;
        let mut rows = Vec::with_capacity(retirement.readouts().len());
        for readout in retirement.readouts() {
            check(
                readout.job() == job
                    && readout.owner() == retirement.owner()
                    && readout.frame() == frame
                    && readout.question_digest() == retirement.question_digest()
                    && readout.capture_sequence() == retirement.capture_sequence()
                    && readout.matches_model(source.model())
                    && Arc::ptr_eq(readout.context_view(), retirement.context_view()),
                "R readout differs from the original retirement/Bound",
            )?;
            let record = self.record(admission, &source, retirement, readout, context)?;
            rows.push(PublishedRoventRReadout {
                record,
                admission: admission.clone(),
                source: Arc::clone(&source),
            });
        }
        // Record original empty/failed terminal occurrence too. There is no
        // fabricated zero-input row and no target/optimizer count for it.
        self.consumed.insert(occurrence);
        let lineage = retirement.lineage();
        let retirement = retirement_wire(retirement)?;
        let retirement_sha256 = generation_digest(R_READOUT_RETIREMENT_DOMAIN, &retirement)?;
        Ok(CapturedRoventRRetirement {
            lineage,
            manifest_sha256: self.manifest.sha256()?,
            retirement,
            retirement_sha256,
            rows,
            admission: admission.clone(),
            source,
        })
    }

    fn record(
        &self,
        admission: &RoventCollectionRootAdmission,
        source: &LoadedRoventNnuecSource,
        retirement: &NnuecCaptureRetirement,
        readout: &NnuecCapturedReadout,
        context: &RoventRCollectionContext,
    ) -> Result<RoventRReadoutRecord, ArenaError> {
        let view = readout.context_view();
        let p = readout.provenance();
        let query = view.query();
        let policy = view.selection_policy();
        let rules =
            |snapshot: &rz_position::PositionSnapshot| -> Result<RoventRRulesEvidence, ArenaError> {
                let position = Position::from_snapshot(
                    snapshot,
                    rz_position::PositionLimits {
                        max_history_positions: 16_385,
                        ..Default::default()
                    },
                )
                .map_err(|e| invalid(e.to_string()))?;
                let replay = serde_json::to_value(RoventRulesPositionReplay::capture(&position)?)
                    .map_err(|e| invalid(e.to_string()))?;
                Ok(RoventRRulesEvidence {
                    anchor: exact_rules_anchor(&position)?,
                    replay_sha256: generation_digest(ROVENT_RULES_POSITION_REPLAY, &replay)?,
                    replay,
                })
            };
        let context_records = view
            .records()
            .iter()
            .zip(view.records_metadata())
            .enumerate()
            .map(|(i, (r, m))| {
                Ok(RoventRContextRecord {
                    local_id: i as u64,
                    record_id: format!("public-record-{}", r.revision),
                    record_revision: r.revision,
                    parent_revision: r.parent_revision,
                    supersedes_revision: r.supersedes_revision,
                    origin_state: r.origin_state.0 as u64,
                    kind: match r.kind {
                        rz_search::pals::engine::RecordKind::Proposal => 0,
                        rz_search::pals::engine::RecordKind::Counterexample => 1,
                        rz_search::pals::engine::RecordKind::Repair => 2,
                        rz_search::pals::engine::RecordKind::CpuVerification => 3,
                    },
                    anchor: m.anchor,
                    moves: packed(&r.line)?,
                    original_first_move: m
                        .original_first_move
                        .map(|mv| packed(&[mv]).map(|v| v[0]))
                        .transpose()?,
                    parent: m.parent,
                    supersedes: m.supersedes,
                    source_state: m.source_state.0 as u64,
                    source_revision: m.source_revision,
                    source_line_offset: m.source_line_offset,
                    record_perspective: match r.perspective {
                        rz_position::Color::White => "white",
                        rz_position::Color::Black => "black",
                    }
                    .into(),
                    encoded_perspective: readout
                        .input()
                        .context_lines
                        .get(i)
                        .ok_or_else(|| invalid("original R context line missing"))?
                        .perspective
                        .clone(),
                    completed_depth: r.completed_depth,
                    score_scope: r.score_scope.map(|v| format!("{v:?}")),
                    value: r.value,
                    cpu_observation: r.cpu_observation.map(|v| v.0 as u64),
                    critical: r.critical,
                })
            })
            .collect::<Result<Vec<_>, ArenaError>>()?;
        let context_anchors = readout
            .context_anchors()
            .iter()
            .map(rules)
            .collect::<Result<Vec<_>, _>>()?;
        let parameter = p
            .model
            .weights_sha256
            .clone()
            .ok_or_else(|| invalid("actual R semantic parameter identity missing"))?;
        let bound = RoventRBoundEvidence {
            model_semantics: p.model.semantics.clone(),
            parameter_sha256: parameter,
            training_state: serde_json::to_value(&p.model.training)
                .map_err(|e| invalid(e.to_string()))?,
            weights_file_sha256: source.weights_artifact().sha256.clone(),
            collection_model_epoch: self.manifest.collection.model_epoch,
            collection_context_epoch: self.manifest.collection.context_epoch,
            root_state: p.root.0 as u64,
            generation: p.generation,
            sequence: p.sequence,
            full_view_sha256: hex(Digest(p.full_view_digest)),
            view_seal_sha256: hex(Digest(p.view_seal)),
            record_scope_sha256: hex(Digest(p.record_scope_digest)),
            policy_sha256: hex(Digest(p.policy_digest)),
            context_sha256: readout.context_sha256().into(),
            context_query: RoventRContextQuery {
                purpose: format!("{:?}", query.purpose),
                prefix: packed(&query.prefix)?,
                proposal: packed(&query.proposal)?,
                counterexample: query.counterexample.as_deref().map(packed).transpose()?,
                legal: packed(&query.legal)?,
                divergences: query
                    .divergences
                    .iter()
                    .map(|v| u16::try_from(*v).map_err(|_| invalid("R query divergence index")))
                    .collect::<Result<Vec<_>, _>>()?,
            },
            selection_policy: RoventRSelectionPolicy {
                mode: match policy.mode {
                    rz_search::pals::review_context::ReviewAnchorMode::CurrentRootOnly => {
                        "current_root_only"
                    }
                    rz_search::pals::review_context::ReviewAnchorMode::AnchoredCompact => {
                        "anchored_compact"
                    }
                }
                .into(),
                token_budget: policy.token_budget,
                max_records: policy.max_records,
                max_anchors: policy.max_anchors,
                required_bundle: policy.required_bundle,
            },
            actual_tokens: view.token_count(),
            token_budget: view.token_budget(),
            length_bucket: view.length_bucket(),
            context_records,
            context_anchors,
            context_source_anchors: readout
                .context_source_snapshots()
                .iter()
                .map(rules)
                .collect::<Result<Vec<_>, _>>()?,
            context_anchor_states: view
                .anchors()
                .iter()
                .map(|anchor| anchor.state.0 as u64)
                .collect(),
            local_anchor: rules(readout.local_anchor())?,
            local_encoded_perspective: readout.input().local_line.perspective.clone(),
            current_rules: rules(readout.current_rules())?,
        };
        let input = RoventRInputEvidence::from_original_json(
            serde_json::to_string(readout.input()).map_err(|e| invalid(e.to_string()))?,
            &bound,
            &hex(readout.input_sha256()),
        )?;
        let output = readout.numeric();
        let bits = |values: &[f32]| values.iter().map(|v| v.to_bits()).collect();
        let mut record = RoventRReadoutRecord {
            schema: R_READOUT_RECORD_DOMAIN.into(),
            manifest_sha256: self.manifest.sha256()?,
            split: *self
                .manifest
                .collection
                .opening_splits
                .get(&context.opening_id)
                .ok_or_else(|| invalid("R opening absent from original split registration"))?,
            lineage: RoventRLineage {
                game_id: context.game_id.clone(),
                opening_id: context.opening_id.clone(),
                line_genealogy_id: context.line_genealogy_id.clone(),
                transposition_sha256: bound.current_rules.anchor.rules_state_sha256.clone(),
                actor_id: context.actor_id.clone(),
                teacher_actor_id: context.teacher_actor_id.clone(),
                ancestor_origin_sha256: context.ancestor_origin_sha256.clone(),
                pc_genealogy: context.pc_genealogy.clone(),
            },
            native_admission: admission.raw_receipt().clone(),
            native_admission_sha256: admission.raw_sha256().into(),
            root_descriptor: admission.root_admission().descriptor().clone(),
            root_descriptor_sha256: hex(admission.root_admission().digest()),
            startup_input_lock_sha256: hex(admission.root_admission().startup_lock_digest()),
            question: RoventRQuestion {
                job: job_wire(readout.job()),
                worker_owner: readout.owner().0,
                frame_sequence: readout.frame().sequence,
                capture_sequence: readout.capture_sequence(),
                readout_sequence: readout.readout_sequence(),
                observed_tick_ns: readout.observed_at().0,
            },
            bound,
            input,
            numeric: RoventRNumericBits {
                schema: output.schema.clone(),
                parameter_sha256: output.weights_sha256.clone(),
                board_accumulators_bits: output.board_accumulators.each_ref().map(|v| bits(v)),
                context_accumulator_bits: bits(&output.context_accumulator),
                local_accumulator_bits: bits(&output.local_accumulator),
                hidden_bits: bits(&output.hidden),
                raw_outputs_bits: output.raw_outputs.map(f32::to_bits),
                value_bits: output.value.to_bits(),
                wdl_logits_bits: output.wdl_logits.map(f32::to_bits),
                wdl_bits: output.wdl.map(f32::to_bits),
            },
            retirement: retirement_wire(retirement)?,
            retirement_sha256: String::new(),
            semantic_consumption: None,
            whole_action_cost: None,
            wdl: None,
            utility: RoventRUtilityState {
                support_mask: [false; 7],
                target_known_mask: [false; 7],
                unknown_reason: "MissingOriginalActionClosureAndWholeCost".into(),
            },
            origin_sha256: String::new(),
            record_sha256: String::new(),
        };
        record.seal(&self.manifest)?;
        Ok(record)
    }
}
fn retirement_wire(
    retirement: &NnuecCaptureRetirement,
) -> Result<RoventRCaptureRetirement, ArenaError> {
    let limits = retirement.limits();
    Ok(RoventRCaptureRetirement {
        job: job_wire(retirement.job()),
        worker_owner: retirement.owner().0,
        frame_sequence: retirement.frame().sequence,
        capture_sequence: retirement.capture_sequence(),
        ending: match retirement.ending() {
            NnuecCaptureEnding::CpuTerminal(c) => RoventRCaptureEnding::CpuTerminal {
                completion: format!("{c:?}"),
            },
            NnuecCaptureEnding::NotStarted => RoventRCaptureEnding::NotStarted,
            NnuecCaptureEnding::Failed => RoventRCaptureEnding::Failed,
            NnuecCaptureEnding::Unknown => RoventRCaptureEnding::Unknown,
        },
        limits: RoventRCaptureLimits {
            maximum_readouts: limits.maximum_readouts(),
            every_nth_score: limits.every_nth_score(),
            maximum_input_bytes: limits.maximum_input_bytes() as u64,
        },
        attempted: retirement.attempted(),
        selected: retirement.selected(),
        retained_readouts: retirement.readouts().len() as u16,
        overflow: retirement.overflow(),
        retained_input_bytes: retirement.input_bytes() as u64,
        selected_input_preparation_ns: u64::try_from(retirement.preparation_time().as_nanos())
            .map_err(|_| invalid("R capture preparation time overflow"))?,
        error: retirement.error().map(|e| format!("{e:?}")),
    })
}
fn job_wire(job: CpuJobEnvelope) -> RoventRJob {
    RoventRJob {
        authority: RoventRAuthority {
            process_epoch: job.id.authority.epoch.0,
            game_generation: job.id.authority.game.0,
            root_generation: job.id.authority.root.0,
            implementation_sha256: hex(job.id.authority.implementation),
        },
        job_sequence: job.id.sequence,
        state_owner: job.state_identity.owner.0,
        state_revision: job.state_identity.revision.0,
        state_semantic_sha256: hex(job.state_identity.semantic),
        question_sha256: hex(job.question),
        root_restriction_sha256: hex(job.root_restriction),
        evaluation_namespace_sha256: hex(job.evaluation_namespace),
        search_conditions_sha256: hex(job.search_conditions),
        total_native_node_credit: job.total_native_node_credit,
        hard_deadline_clock_epoch: job.hard_deadline.clock.0.0,
        hard_deadline_ns: job.hard_deadline.at.0,
    }
}
