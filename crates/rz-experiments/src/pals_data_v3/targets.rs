use super::*;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GenerationPolicyTargetV3 {
    pub observed_sequence: u64,
    pub question_id: String,
    pub view_sha256: String,
    pub raw_evidence_sha256: String,
    pub source: String,
    pub completed: bool,
    pub stale: bool,
    pub anchor: GenerationRulesAnchorV3,
    pub ranked_moves: Vec<u16>,
    pub teacher_binary_sha256: String,
    pub teacher_profile_sha256: String,
    pub go_sequence: u64,
    pub completed_depth: u32,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GenerationRulesEndingV3 {
    Checkmate,
    Stalemate,
    Repetition,
    FiftyMove,
    SeventyFiveMove,
    InsufficientMaterial,
    DeadPosition,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GenerationOutcomeV3 {
    WhiteWin,
    BlackWin,
    Draw,
}
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GenerationWdlTargetV3 {
    pub observed_sequence: u64,
    pub question_id: String,
    pub view_sha256: String,
    pub raw_evidence_sha256: String,
    pub source: String,
    pub game_id: String,
    pub anchor: GenerationRulesAnchorV3,
    pub trajectory_membership_sha256: String,
    pub rules_result_sha256: String,
    pub ending: GenerationRulesEndingV3,
    pub outcome: GenerationOutcomeV3,
}
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GenerationTrajectoryAuthorityV3 {
    pub game_id: String,
    pub ending: GenerationRulesEndingV3,
    pub outcome: GenerationOutcomeV3,
    pub anchors: Vec<GenerationRulesAnchorV3>,
    pub result_evidence_sha256: String,
}
impl GenerationTrajectoryAuthorityV3 {
    pub fn validate(&self) -> Result<(), ManifestError> {
        ensure(
            generation_identifier(&self.game_id)
                && generation_sha(&self.result_evidence_sha256)
                && !self.anchors.is_empty()
                && self.anchors.len() <= 4096,
            "actual Rules trajectory authority capacity/identity",
        )?;
        valid_outcome(self.ending, self.outcome)?;
        for anchor in &self.anchors {
            anchor.validate()?;
        }
        Ok(())
    }
    pub fn sha256(&self) -> Result<String, ManifestError> {
        self.validate()?;
        generation_digest("rz-pals-actual-trajectory-membership/1", self)
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GenerationDivergenceStatusV3 {
    SupportedAfterRepair,
    RefutedByRepair,
}
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GenerationDivergenceTargetV3 {
    pub observed_sequence: u64,
    pub question_id: String,
    pub view_sha256: String,
    pub raw_evidence_sha256: String,
    pub source: String,
    pub counter_record_id: String,
    pub repair_question_id: String,
    pub status: GenerationDivergenceStatusV3,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GenerationTaskV3 {
    DefendResponse,
    AttackRepair,
    WidenResponses,
    LowerSelectivity,
    ResumeTask,
    CrossProfileRecheck,
    Defer,
}
impl GenerationTaskV3 {
    pub fn index(self) -> usize {
        match self {
            Self::DefendResponse => 0,
            Self::AttackRepair => 1,
            Self::WidenResponses => 2,
            Self::LowerSelectivity => 3,
            Self::ResumeTask => 4,
            Self::CrossProfileRecheck => 5,
            Self::Defer => 6,
        }
    }
}
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GenerationWitnessV3 {
    pub witness_id: String,
    pub kind: GenerationRecordKindV3,
    pub observed_sequence: u64,
    pub raw_evidence_sha256: String,
}
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GenerationActionOutcomeV3 {
    pub action: GenerationTaskV3,
    pub execution_id: String,
    pub completed: bool,
    pub wall_cost_ns: Option<u64>,
    pub new_witnesses: Vec<GenerationWitnessV3>,
    pub causal_receipt_sha256: String,
}
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GenerationTaskTargetV3 {
    pub observed_sequence: u64,
    pub question_id: String,
    pub view_sha256: String,
    pub raw_evidence_sha256: String,
    pub source: String,
    pub outcomes: Vec<GenerationActionOutcomeV3>,
    pub target_known_mask: [bool; 7],
}
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GenerationTargetsV3 {
    pub policy: Option<GenerationPolicyTargetV3>,
    pub wdl: Option<GenerationWdlTargetV3>,
    pub divergence: Option<GenerationDivergenceTargetV3>,
    pub task: Option<GenerationTaskTargetV3>,
}
fn base(
    input: &GenerationInputV3,
    observed: u64,
    question: &str,
    view: &str,
    raw: &str,
) -> Result<(), ManifestError> {
    ensure(
        observed >= input.capture_sequence
            && question == input.question_id
            && view == input.view_sha256
            && generation_sha(raw),
        "head target predates/mismatches submitted question/view/evidence",
    )
}
fn valid_outcome(
    ending: GenerationRulesEndingV3,
    outcome: GenerationOutcomeV3,
) -> Result<(), ManifestError> {
    ensure(
        (ending != GenerationRulesEndingV3::Checkmate) == (outcome == GenerationOutcomeV3::Draw),
        "Rules terminal/outcome mismatch",
    )
}
impl GenerationPolicyTargetV3 {
    pub fn probabilities(&self, input: &GenerationInputV3) -> Result<Vec<f64>, ManifestError> {
        self.probabilities_for_source(input, "stockfish19_completed_multipv")
    }
    /// 조건부 순위 의견의 기존 계산을 공유한다. source 이름은 각 명시
    /// 형식의 caller가 고정하며 native/teacher 권한을 이 함수가 발급하지 않는다.
    pub(crate) fn probabilities_for_source(
        &self,
        input: &GenerationInputV3,
        source: &str,
    ) -> Result<Vec<f64>, ManifestError> {
        base(
            input,
            self.observed_sequence,
            &self.question_id,
            &self.view_sha256,
            &self.raw_evidence_sha256,
        )?;
        validate_generation_moves(&self.ranked_moves, 4)?;
        ensure(
            self.source == source
                && self.completed
                && !self.stale
                && self.anchor == input.rules_anchor
                && generation_sha(&self.teacher_binary_sha256)
                && generation_sha(&self.teacher_profile_sha256)
                && self.go_sequence > 0
                && self.completed_depth > 0
                && !self.ranked_moves.is_empty()
                && self.ranked_moves.iter().collect::<BTreeSet<_>>().len()
                    == self.ranked_moves.len()
                && self
                    .ranked_moves
                    .iter()
                    .all(|m| input.legal_moves.contains(m)),
            "policy requires completed nonstale same-anchor legal teacher opinion",
        )?;
        let total = (0..self.ranked_moves.len())
            .map(|rank| (-(rank as f64)).exp())
            .sum::<f64>();
        Ok(input
            .legal_moves
            .iter()
            .map(|m| {
                self.ranked_moves
                    .iter()
                    .position(|r| r == m)
                    .map_or(0.0, |rank| (-(rank as f64)).exp() / total)
            })
            .collect())
    }
}
impl GenerationWdlTargetV3 {
    pub fn probabilities(&self, input: &GenerationInputV3) -> Result<[f64; 3], ManifestError> {
        base(
            input,
            self.observed_sequence,
            &self.question_id,
            &self.view_sha256,
            &self.raw_evidence_sha256,
        )?;
        valid_outcome(self.ending, self.outcome)?;
        ensure(
            self.source == "rules_selfplay_trajectory"
                && self.game_id == input.game_id
                && self.anchor == input.rules_anchor
                && generation_sha(&self.trajectory_membership_sha256)
                && generation_sha(&self.rules_result_sha256),
            "WDL requires exact actual trajectory anchor and Rules terminal",
        )?;
        Ok(match self.outcome {
            GenerationOutcomeV3::Draw => [0., 1., 0.],
            outcome
                if (outcome == GenerationOutcomeV3::WhiteWin)
                    == input.rules_anchor.white_to_move =>
            {
                [1., 0., 0.]
            }
            _ => [0., 0., 1.],
        })
    }
}
impl GenerationDivergenceTargetV3 {
    pub fn value(&self, input: &GenerationInputV3) -> Result<f64, ManifestError> {
        base(
            input,
            self.observed_sequence,
            &self.question_id,
            &self.view_sha256,
            &self.raw_evidence_sha256,
        )?;
        ensure(
            self.source == "actual_repair_continuation"
                && generation_identifier(&self.repair_question_id)
                && input
                    .divergence_record_ids
                    .contains(&self.counter_record_id)
                && input.selected_view.records.iter().any(|r| {
                    r.record_id == self.counter_record_id
                        && r.kind == GenerationRecordKindV3::Counter
                }),
            "divergence counter absent from submitted critic input",
        )?;
        Ok(
            if self.status == GenerationDivergenceStatusV3::SupportedAfterRepair {
                1.
            } else {
                0.
            },
        )
    }
}
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct GenerationDerivedTasksV3 {
    pub target_known_mask: [bool; 7],
    pub pairs: Vec<[u8; 2]>,
    pub new_witness_context: bool,
}
pub fn generation_derive_tasks(
    input: &GenerationInputV3,
    outcomes: &[GenerationActionOutcomeV3],
) -> Result<GenerationDerivedTasksV3, ManifestError> {
    ensure(outcomes.len() <= 7, "action outcomes capacity")?;
    let visible: BTreeSet<_> = input
        .selected_view
        .records
        .iter()
        .map(|r| r.record_id.as_str())
        .collect();
    let mut actual = BTreeMap::new();
    let mut actions = BTreeSet::new();
    for outcome in outcomes {
        ensure(
            actions.insert(outcome.action)
                && input.action_support_mask[outcome.action.index()]
                && generation_identifier(&outcome.execution_id)
                && generation_sha(&outcome.causal_receipt_sha256)
                && outcome.new_witnesses.len() <= 256,
            "unsupported/duplicate action causal identity",
        )?;
        let mut witnesses = BTreeSet::new();
        for witness in &outcome.new_witnesses {
            ensure(
                generation_identifier(&witness.witness_id)
                    && generation_sha(&witness.raw_evidence_sha256)
                    && [
                        GenerationRecordKindV3::Counter,
                        GenerationRecordKindV3::Repair,
                    ]
                    .contains(&witness.kind)
                    && !visible.contains(witness.witness_id.as_str())
                    && witnesses.insert(witness.witness_id.as_str())
                    && witness.observed_sequence > input.capture_sequence,
                "duplicate/old/noncausal witness cannot reward V",
            )?;
        }
        if outcome.completed {
            if let Some(cost) = outcome.wall_cost_ns {
                actual.insert(outcome.action, (witnesses.len(), cost));
            }
        }
    }
    let mut result = GenerationDerivedTasksV3 {
        target_known_mask: [false; 7],
        pairs: vec![],
        new_witness_context: actual.values().any(|v| v.0 > 0),
    };
    for (preferred, a) in &actual {
        for (other, b) in &actual {
            if preferred != other && a.0.max(b.0) > 0 && a.0 >= b.0 && a.1 <= b.1 && a != b {
                result.target_known_mask[preferred.index()] = true;
                result.target_known_mask[other.index()] = true;
                result
                    .pairs
                    .push([preferred.index() as u8, other.index() as u8]);
            }
        }
    }
    Ok(result)
}
impl GenerationTargetsV3 {
    pub fn validate(&self, input: &GenerationInputV3) -> Result<(), ManifestError> {
        if input.role == GenerationRoleV3::Validator {
            ensure(
                self.policy.is_none() && self.wdl.is_none() && self.divergence.is_none(),
                "validator private head routing",
            )?;
        } else {
            ensure(
                self.task.is_none(),
                "P/C cannot receive V-private task target",
            )?;
        }
        if let Some(target) = &self.policy {
            target.probabilities(input)?;
        }
        if let Some(target) = &self.wdl {
            target.probabilities(input)?;
        }
        if let Some(target) = &self.divergence {
            ensure(
                input.role == GenerationRoleV3::Critic,
                "noncritic divergence label",
            )?;
            target.value(input)?;
        }
        if let Some(target) = &self.task {
            base(
                input,
                target.observed_sequence,
                &target.question_id,
                &target.view_sha256,
                &target.raw_evidence_sha256,
            )?;
            ensure(
                target.source == "actual_action_witness"
                    && target.target_known_mask
                        == generation_derive_tasks(input, &target.outcomes)?.target_known_mask,
                "V target-known mask differs from actual Pareto evidence",
            )?;
        }
        Ok(())
    }
}

/// Full head receipts, not a set of declared hashes. Caller must separately
/// verify producer registration and the Rules trajectory membership authority.
pub fn validate_generation_head_evidence(
    record: &GenerationRecordV3,
    evidence: &BTreeMap<String, serde_json::Value>,
) -> Result<(), ManifestError> {
    record.validate()?;
    for (head, target) in [
        (
            "policy",
            serde_json::to_value(&record.future_targets.policy),
        ),
        ("wdl", serde_json::to_value(&record.future_targets.wdl)),
        (
            "divergence",
            serde_json::to_value(&record.future_targets.divergence),
        ),
        ("task", serde_json::to_value(&record.future_targets.task)),
    ] {
        let mut target = target.map_err(|e| ManifestError::Integrity(e.to_string()))?;
        if target.is_null() {
            continue;
        }
        let fields = target.as_object_mut().unwrap();
        let raw = fields
            .remove("raw_evidence_sha256")
            .and_then(|v| v.as_str().map(str::to_owned))
            .ok_or_else(|| ManifestError::Integrity("missing head raw evidence seal".into()))?;
        let expected = serde_json::json!({"schema":"rz-pals-generation-head-evidence/1","input_sha256":record.input_sha256,"head":head,"target":target});
        ensure(
            evidence.get(&raw) == Some(&expected)
                && generation_digest("rz-pals-generation-head-evidence/1", &expected)? == raw,
            "head differs from recovered sealed execution evidence",
        )?;
    }
    if let Some(policy) = &record.future_targets.policy {
        validate_generation_teacher_evidence(record, policy, evidence)?;
    }
    if let Some(task) = &record.future_targets.task {
        for outcome in &task.outcomes {
            let original_input =
                super::frozen_action::causal_input_for_outcome(record, outcome, evidence)?;
            let mut content = serde_json::to_value(outcome).unwrap();
            content
                .as_object_mut()
                .unwrap()
                .remove("causal_receipt_sha256");
            let causal = serde_json::json!({"schema":"rz-pals-generation-causal-evidence/1","input_sha256":original_input,"outcome":content});
            ensure(
                evidence.get(&outcome.causal_receipt_sha256) == Some(&causal)
                    && generation_digest("rz-pals-generation-causal-evidence/1", &causal)?
                        == outcome.causal_receipt_sha256,
                "action differs from recovered causal execution",
            )?;
            for witness in &outcome.new_witnesses {
                let mut content = serde_json::to_value(witness).unwrap();
                content
                    .as_object_mut()
                    .unwrap()
                    .remove("raw_evidence_sha256");
                let raw = serde_json::json!({"schema":"rz-pals-generation-witness-evidence/1","input_sha256":original_input,"witness":content,"rules_validated":true});
                ensure(
                    evidence.get(&witness.raw_evidence_sha256) == Some(&raw)
                        && generation_digest("rz-pals-generation-witness-evidence/1", &raw)?
                            == witness.raw_evidence_sha256,
                    "witness differs from recovered Rules evidence",
                )?;
            }
        }
    }
    Ok(())
}

/// Join a policy head's canonical receipt with the original input and the
/// recovered private-owner teacher submission/terminal receipts. This is an
/// independent provenance check, never a conversion of CP/mate into WDL.
pub fn validate_generation_teacher_evidence(
    record: &GenerationRecordV3,
    policy: &GenerationPolicyTargetV3,
    evidence: &BTreeMap<String, serde_json::Value>,
) -> Result<(), ManifestError> {
    const BINDING: &str = "rz-pals-generation-teacher-input-binding-evidence/1";
    const SUBMITTED: &str = "rz-pals-generation-teacher-submission-evidence/1";
    const COMPLETED: &str = "rz-pals-generation-teacher-completion-evidence/1";
    let mut matches = evidence.iter().filter(|(_, raw)| {
        raw["schema"] == BINDING
            && raw["input_sha256"] == record.input_sha256
            && raw["policy_head_evidence_sha256"] == policy.raw_evidence_sha256
    });
    let (binding_sha, binding) = matches.next().ok_or_else(|| {
        ManifestError::Integrity("teacher policy missing actual input/source binding".into())
    })?;
    ensure(
        matches.next().is_none() && generation_digest(BINDING, binding)? == *binding_sha,
        "teacher input/source binding duplicate or digest mismatch",
    )?;
    let anchor = serde_json::to_value(&record.input.rules_anchor)
        .map_err(|e| ManifestError::Integrity(e.to_string()))?;
    ensure(
        binding["rules_anchor"] == anchor
            && binding["observed_sequence"].as_u64() == Some(policy.observed_sequence)
            && binding["interpretation"] == "conditional_exact_endpoint_opinion",
        "teacher binding changed original anchor/chronology/interpretation",
    )?;
    let recover = |field: &str, schema: &str| -> Result<&serde_json::Value, ManifestError> {
        let sha = binding[field]
            .as_str()
            .filter(|sha| generation_sha(sha))
            .ok_or_else(|| ManifestError::Integrity("teacher source receipt SHA missing".into()))?;
        let raw = evidence
            .get(sha)
            .ok_or_else(|| ManifestError::Integrity("teacher source receipt unrecovered".into()))?;
        ensure(
            raw["schema"] == schema && generation_digest(schema, raw)? == sha,
            "teacher source receipt schema/digest mismatch",
        )?;
        Ok(raw)
    };
    let submitted = recover("teacher_submission_sha256", SUBMITTED)?;
    let completed = recover("teacher_completion_sha256", COMPLETED)?;
    let submit_sequence = submitted["observed_sequence"]
        .as_u64()
        .ok_or_else(|| ManifestError::Integrity("teacher submission sequence".into()))?;
    let epoch = submitted["execution_epoch"]
        .as_u64()
        .ok_or_else(|| ManifestError::Integrity("teacher execution epoch".into()))?;
    ensure(
        submit_sequence > record.input.capture_sequence
            && completed["observed_sequence"].as_u64() == Some(policy.observed_sequence)
            && completed["submission_observed_sequence"].as_u64() == Some(submit_sequence)
            && policy.observed_sequence > submit_sequence
            && record.input.worker_id == format!("native-owner-{epoch}")
            && submitted["rules_anchor"] == anchor,
        "teacher source predates input or crossed exact Rules/process context",
    )?;
    for field in [
        "execution_epoch",
        "execution_sequence",
        "job_sequence",
        "root_generation",
    ] {
        ensure(
            submitted[field].as_u64().is_some() && submitted[field] == completed[field],
            "teacher original job/execution/root changed",
        )?;
    }
    for (field, expected) in [
        ("teacher_binary_sha256", &policy.teacher_binary_sha256),
        ("teacher_profile_sha256", &policy.teacher_profile_sha256),
    ] {
        ensure(
            submitted[field].as_str() == Some(expected.as_str())
                && completed[field].as_str() == Some(expected.as_str()),
            "teacher registered binary/profile changed",
        )?;
    }
    ensure(
        submitted["process"].is_object()
            && submitted["process"] == completed["physical_process_identity"]
            && submitted["process"]["pid"].as_u64().is_some_and(|n| n > 0)
            && submitted["process"]["proc_start_ticks"]
                .as_u64()
                .is_some_and(|n| n > 0)
            && submitted["evaluation_namespace"]
                .as_str()
                .is_some_and(generation_sha)
            && submitted["evaluation_namespace"] == completed["evaluation_namespace"]
            && completed["successful_owner_join"] == true
            && completed["source_scope_joined"] == true
            && completed["original_question_target_eligible"] == true
            && completed["cancel_sequence"].as_u64() == Some(0)
            && completed["exact_resume_supported"] == false
            && completed["go_issue"] == "Issued",
        "teacher actual owner/go source was not joined",
    )?;
    let result = &completed["result"];
    ensure(
        result["completion"] == "BestMove"
            && result["go_issue"] == "Issued"
            && result["go_sequence"].as_u64() == Some(policy.go_sequence)
            && result["readiness_barrier_completed"] == true
            && result["policy_target_known"] == true
            && result["identity"]["binary_sha256"].as_str()
                == Some(policy.teacher_binary_sha256.as_str())
            && submitted["uci_name"] == result["uci_name"],
        "teacher opinion did not complete its original go/readiness",
    )?;
    let mut opinions = result["opinions"]
        .as_array()
        .cloned()
        .ok_or_else(|| ManifestError::Integrity("teacher completed opinions absent".into()))?;
    ensure(
        !opinions.is_empty()
            && opinions.len() <= 4
            && result["requested_multipv"].as_u64() == Some(opinions.len() as u64),
        "teacher requested complete rank set differs",
    )?;
    opinions.sort_by_key(|o| o["rank"].as_u64().unwrap_or(0));
    let mut moves = Vec::with_capacity(opinions.len());
    for (rank, opinion) in opinions.iter().enumerate() {
        let pv = opinion["pv_move16"]
            .as_array()
            .ok_or_else(|| ManifestError::Integrity("teacher PV Move16 evidence absent".into()))?;
        let movement = pv
            .first()
            .and_then(|value| value.as_u64())
            .filter(|value| *value <= u64::from(u16::MAX))
            .ok_or_else(|| ManifestError::Integrity("teacher legal first move absent".into()))?
            as u16;
        ensure(
            opinion["rank"].as_u64() == Some(rank as u64 + 1)
                && opinion["depth"].as_u64() == Some(u64::from(policy.completed_depth))
                && opinion["bound"] == "ExactReported",
            "teacher rank/depth not one complete opinion set",
        )?;
        moves.push(movement);
    }
    ensure(
        moves == policy.ranked_moves
            && result["best_move_move16"].as_u64()
                == moves.first().map(|movement| u64::from(*movement)),
        "teacher policy differs from recovered ranked moves",
    )?;
    Ok(())
}
