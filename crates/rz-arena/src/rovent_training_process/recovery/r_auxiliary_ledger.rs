//! Separate R sample admission and successful-update accounting. The dataset
//! owner has already replayed original Rules/trajectory evidence; this ledger
//! never turns a caller supplied target flag or projection into that owner.
use super::*;

pub(super) const R_AUXILIARY_LEDGER: &str = "rz-rovent-r-auxiliary-parent-ledger/1";
const MAX_AUX_RETAINED: usize = MAX_TRAINER_ROWS;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum AuxiliaryPhase {
    Pc,
    V,
}

/// Issued only from an already verified checkpoint owner, or explicit empty
/// new-stage state. This is the preserved sampler's context, not the current
/// V corpus's joint eligibility context and not a deserializable capability.
#[derive(Clone, Debug)]
struct PreservedVSamplingContext {
    checkpoint_publication_sha256: Option<String>,
    pool_sha256: Option<String>,
    sampler_sha256: Option<String>,
}
impl PreservedVSamplingContext {
    fn observation(&self) -> Value {
        json!({"schema":"rz-rovent-preserved-v-r-sampling-context/1",
        "checkpoint_publication_sha256":self.checkpoint_publication_sha256,"pool_sha256":self.pool_sha256,
        "sampler_sha256":self.sampler_sha256,"scope":"preserved_sampler_only_current_joint_owner_still_checks_leakage",
        "execution_authorized":false})
    }
}

#[derive(Clone, Debug)]
struct AdmittedAuxiliaryOccurrence {
    input_sha256: String,
    origin_sha256: String,
    record_sha256: String,
    retirement_sha256: String,
    wdl_raw_sha256: Option<String>,
    valid: bool,
}
impl AdmittedAuxiliaryOccurrence {
    fn projection(&self) -> Value {
        json!({"input_sha256":self.input_sha256,"origin_sha256":self.origin_sha256,
        "record_sha256":self.record_sha256,"retirement_sha256":self.retirement_sha256,"wdl_raw_sha256":self.wdl_raw_sha256})
    }
}

/// This is a current ledger, not a Deserialize type. Cloning retains the same
/// opaque dataset owner. Snapshot values are data projections only.
#[derive(Clone)]
pub(super) struct RoventAuxiliaryLedger {
    dataset: Arc<VerifiedRoventRAuxiliaryDataset>,
    recipe: RoventRAuxiliaryRecipe,
    pending: Vec<AdmittedAuxiliaryOccurrence>,
    consumed_rows: u64,
    valid_rows: u64,
    completed_nnuec_updates: u64,
    retained_inputs: BTreeSet<String>,
    retained_origins: BTreeSet<String>,
    ancestor_inputs: BTreeSet<String>,
    ancestor_origins: BTreeSet<String>,
    ancestor_nnuec_updates: u64,
    non_auxiliary_nnuec_ancestry: Option<Value>,
    phase: AuxiliaryPhase,
    preserved_v_sampling: Option<PreservedVSamplingContext>,
}
impl RoventAuxiliaryLedger {
    pub(super) fn new(
        dataset: Arc<VerifiedRoventRAuxiliaryDataset>,
        recipe: RoventRAuxiliaryRecipe,
        ancestor_inputs: BTreeSet<String>,
        ancestor_origins: BTreeSet<String>,
        ancestor_nnuec_updates: u64,
    ) -> Result<Self, ArenaError> {
        recipe.validate()?;
        dataset.joint_current_pc_publication_sha256()?;
        check(
            dataset.manifest_file_sha256() == recipe.manifest_file_sha256
                && dataset.index_file_sha256() == recipe.index_file_sha256,
            "R auxiliary recipe differs from its exact opaque dataset source",
        )?;
        check(
            ancestor_inputs.len() <= MAX_AUX_RETAINED
                && ancestor_origins.len() <= MAX_AUX_RETAINED
                && ancestor_inputs
                    .iter()
                    .chain(&ancestor_origins)
                    .all(|v| generation_sha(v)),
            "bounded separate R auxiliary ancestry seals",
        )?;
        Ok(Self {
            dataset,
            recipe,
            pending: Vec::new(),
            consumed_rows: 0,
            valid_rows: 0,
            completed_nnuec_updates: 0,
            retained_inputs: BTreeSet::new(),
            retained_origins: BTreeSet::new(),
            ancestor_inputs,
            ancestor_origins,
            ancestor_nnuec_updates,
            non_auxiliary_nnuec_ancestry: None,
            phase: AuxiliaryPhase::Pc,
            preserved_v_sampling: None,
        })
    }
    pub(super) fn is_pending(&self) -> bool {
        !self.pending.is_empty()
    }
    pub(super) fn validate_pc_origins(&self, ledger: &TrainerLedger) -> Result<(), ArenaError> {
        let eligible = self.dataset.joint_eligible_current_pc_origins()?;
        check(
            ledger.pending.len() == 8
                && ledger
                    .pending
                    .iter()
                    .all(|origin| eligible.contains(origin)),
            "P/C/V pending rows violate the original joint PC/R leakage exclusion",
        )
    }
    pub(super) fn validate_joint_retained(&self, ledger: &TrainerLedger) -> Result<(), ArenaError> {
        let inputs = self
            .ancestor_inputs
            .union(&self.retained_inputs)
            .cloned()
            .collect();
        let origins = self
            .ancestor_origins
            .union(&self.retained_origins)
            .cloned()
            .collect();
        self.dataset.validate_joint_retained_training(
            &ledger.consumed_inputs,
            &ledger.consumed_origins,
            &inputs,
            &origins,
        )
    }
    pub(super) fn dataset(&self) -> &Arc<VerifiedRoventRAuxiliaryDataset> {
        &self.dataset
    }
    pub(super) fn recipe(&self) -> &RoventRAuxiliaryRecipe {
        &self.recipe
    }
    pub(super) fn completed_nnuec_updates(&self) -> u64 {
        self.completed_nnuec_updates
    }
    pub(super) fn ancestor_nnuec_updates(&self) -> u64 {
        self.ancestor_nnuec_updates
    }
    pub(super) fn separate_seal_sets(
        &self,
    ) -> (
        &BTreeSet<String>,
        &BTreeSet<String>,
        &BTreeSet<String>,
        &BTreeSet<String>,
    ) {
        (
            &self.retained_inputs,
            &self.retained_origins,
            &self.ancestor_inputs,
            &self.ancestor_origins,
        )
    }
    pub(super) fn bind_current_dataset(
        &mut self,
        dataset: Arc<VerifiedRoventRAuxiliaryDataset>,
        recipe: &RoventRAuxiliaryRecipe,
    ) -> Result<(), ArenaError> {
        check(
            self.recipe == *recipe
                && dataset.manifest_file_sha256() == recipe.manifest_file_sha256
                && dataset.index_file_sha256() == recipe.index_file_sha256,
            "resumed auxiliary source or recipe changed",
        )?;
        self.dataset = dataset;
        Ok(())
    }
    /// Every actual/replayed window declares its phase after source binding.
    /// V retains the exact old sampler state while its new joint owner still
    /// validates current V samples and all retained PC/R exposure.
    pub(super) fn set_execution_phase(
        &mut self,
        phase: &str,
        preserved_checkpoint: Option<&VerifiedRoventCheckpointSource>,
    ) -> Result<(), ArenaError> {
        check(
            !self.is_pending(),
            "phase binding cannot discard pending R work",
        )?;
        match phase {
            "pc" => {
                self.phase = AuxiliaryPhase::Pc;
                self.preserved_v_sampling = None;
            }
            "v" => {
                let context = if let Some(source) = preserved_checkpoint {
                    let original = source.r_auxiliary_checkpoint_projection().ok_or_else(|| {
                        invalid("V resume lacks its original compact R sampler projection")
                    })?;
                    source
                        .r_auxiliary_ledger()
                        .ok_or_else(|| invalid("V resume lacks its original R ledger owner"))?
                        .validate_checkpoint_projection(original)?;
                    PreservedVSamplingContext {
                        checkpoint_publication_sha256: Some(
                            source.publication_pin().sha256.clone(),
                        ),
                        pool_sha256: optional_sha(&original["pool_sha256"])?,
                        sampler_sha256: optional_sha(&original["sampler_sha256"])?,
                    }
                } else {
                    check(
                        self.consumed_rows == 0
                            && self.valid_rows == 0
                            && self.completed_nnuec_updates == 0
                            && self.retained_inputs.is_empty()
                            && self.retained_origins.is_empty(),
                        "new V stage cannot erase an already consumed current-stage R sampler",
                    )?;
                    PreservedVSamplingContext {
                        checkpoint_publication_sha256: None,
                        pool_sha256: None,
                        sampler_sha256: None,
                    }
                };
                self.phase = AuxiliaryPhase::V;
                self.preserved_v_sampling = Some(context);
            }
            _ => {
                return Err(invalid(
                    "explicit PC or V auxiliary execution phase required",
                ));
            }
        }
        Ok(())
    }
    pub(super) fn fork(
        &self,
        dataset: Arc<VerifiedRoventRAuxiliaryDataset>,
        recipe: RoventRAuxiliaryRecipe,
    ) -> Result<Self, ArenaError> {
        check(
            !self.is_pending(),
            "uncommitted auxiliary samples cannot fork",
        )?;
        let inputs = self
            .ancestor_inputs
            .union(&self.retained_inputs)
            .cloned()
            .collect();
        let origins = self
            .ancestor_origins
            .union(&self.retained_origins)
            .cloned()
            .collect();
        let mut result = Self::new(
            dataset,
            recipe,
            inputs,
            origins,
            self.ancestor_nnuec_updates
                .checked_add(self.completed_nnuec_updates)
                .ok_or_else(|| invalid("R ancestry steps overflow"))?,
        )?;
        result.non_auxiliary_nnuec_ancestry = self.non_auxiliary_nnuec_ancestry.clone();
        Ok(result)
    }
    /// First auxiliary-enabled stage can inherit an older, non-auxiliary
    /// learned NNUEC. Preserve that historical source separately, never label
    /// its updates or P/C input seals as newly consumed R observations.
    pub(super) fn from_non_auxiliary_checkpoint(
        dataset: Arc<VerifiedRoventRAuxiliaryDataset>,
        recipe: RoventRAuxiliaryRecipe,
        source: &VerifiedRoventCheckpointSource,
    ) -> Result<Self, ArenaError> {
        check(
            source.r_auxiliary_ledger().is_none(),
            "non-auxiliary ancestry constructor cannot reset an existing R ledger",
        )?;
        let lineage = source.learned_nnuec_lineage()?;
        let updates = count(&lineage["completed_nnuec_updates"])?;
        let (inputs, origins) = source.historical_r_auxiliary_seals()?;
        let mut result = Self::new(dataset, recipe, inputs, origins, updates)?;
        result.non_auxiliary_nnuec_ancestry = Some(
            json!({"schema":"rz-rovent-preserved-non-auxiliary-nnuec-ancestry/1",
            "checkpoint_publication_sha256":source.publication_pin().sha256,"original_lineage":lineage,
            "scope":"inherited_checkpoint_total_not_current_r_auxiliary_steps",
            "historical_metadata_only":true,"past_native_authority_reissued":false,"execution_authorized":false}),
        );
        Ok(result)
    }
    pub(super) fn consume(&mut self, evidence: &Value) -> Result<(), ArenaError> {
        check(
            self.phase == AuxiliaryPhase::Pc
                && self.pending.len() < usize::from(self.recipe.rows_per_pc_update),
            "R auxiliary pending occurrence quota exhausted",
        )?;
        let record = sha(&evidence["record_sha256"])?;
        let original = self.dataset.original_auxiliary_evidence(record)?;
        check(
            evidence == &original,
            "R auxiliary sample changed original occurrence/retirement/target evidence",
        )?;
        let row = self.dataset.auxiliary_row(record)?;
        check(
            row.semantic_input_sha256 == sha(&original["semantic_input_sha256"])?
                && row.record_sha256 == record
                && row.origin_sha256 == sha(&original["origin_sha256"])?
                && row.retirement_sha256 == sha(&original["retirement_sha256"])?
                && json!(row.wdl_raw_receipt_sha256) == original["wdl_raw_sha256"],
            "opaque R row evidence mapping differs",
        )?;
        self.pending.push(AdmittedAuxiliaryOccurrence {
            input_sha256: row.semantic_input_sha256.clone(),
            origin_sha256: row.origin_sha256.clone(),
            record_sha256: row.record_sha256.clone(),
            retirement_sha256: row.retirement_sha256.clone(),
            wdl_raw_sha256: row.wdl_raw_receipt_sha256.clone(),
            valid: row.valid_exact_trajectory,
        });
        Ok(())
    }
    /// Checks the exact R projection before the common actual P/C update owner
    /// commits. No pending sample is discarded if a later validation fails.
    pub(super) fn validate_update(
        &self,
        projection: &Value,
        pc_valid_rows: u64,
    ) -> Result<(), ArenaError> {
        fields(
            projection,
            &[
                "schema",
                "manifest_file_sha256",
                "index_file_sha256",
                "optimizer_owner",
                "occurrences",
                "valid_rows",
                "unique_valid_inputs",
                "pc_valid_rows",
                "normalization",
                "supervision_source",
                "pc_nnuec_evidence_preserved_not_optimized",
                "execution_authorized",
                "r_auxiliary_nnuec_optimizer_steps",
                "r_auxiliary_nnuec_scheduler_steps",
                "r_auxiliary_nnuec_moments_steps",
            ],
        )?;
        check(
            self.phase == AuxiliaryPhase::Pc
                && pc_valid_rows > 0
                && self.pending.len() == usize::from(self.recipe.rows_per_pc_update),
            "R stream cannot create an all-masked P/C or incomplete accumulation update",
        )?;
        let (valid, unique, steps) = sampled_auxiliary_counts(&self.pending);
        check(
            projection["schema"] == "rz-rovent-r-auxiliary-update/1"
                && projection["manifest_file_sha256"] == self.recipe.manifest_file_sha256
                && projection["index_file_sha256"] == self.recipe.index_file_sha256
                && projection["optimizer_owner"] == self.recipe.optimizer_owner
                && projection["normalization"] == self.recipe.normalization
                && projection["supervision_source"] == self.recipe.supervision_source
                && projection["pc_nnuec_evidence_preserved_not_optimized"] == true
                && projection["execution_authorized"] == false
                && projection["occurrences"]
                    == json!(
                        self.pending
                            .iter()
                            .map(|row| row.projection())
                            .collect::<Vec<_>>()
                    )
                && count(&projection["valid_rows"])? == valid
                && count(&projection["unique_valid_inputs"])? == unique
                && count(&projection["pc_valid_rows"])? == pc_valid_rows
                && count(&projection["r_auxiliary_nnuec_optimizer_steps"])? == steps
                && count(&projection["r_auxiliary_nnuec_scheduler_steps"])? == steps
                && count(&projection["r_auxiliary_nnuec_moments_steps"])? == steps,
            "R update fabricated a row, target, PC eligibility or private optimizer/scheduler/moments step",
        )?;
        check(
            self.consumed_rows
                .checked_add(self.pending.len() as u64)
                .is_some()
                && self.valid_rows.checked_add(valid).is_some()
                && self.completed_nnuec_updates.checked_add(steps).is_some(),
            "R counter overflow",
        )?;
        let inputs = self
            .retained_inputs
            .iter()
            .map(String::as_str)
            .chain(self.pending.iter().map(|row| row.input_sha256.as_str()))
            .collect::<BTreeSet<_>>();
        let origins = self
            .retained_origins
            .iter()
            .map(String::as_str)
            .chain(self.pending.iter().map(|row| row.origin_sha256.as_str()))
            .collect::<BTreeSet<_>>();
        check(
            inputs.len() <= MAX_AUX_RETAINED && origins.len() <= MAX_AUX_RETAINED,
            "separate R retained seal capacity exhausted",
        )
    }
    /// Called once only after the common P/C update is committed. A caller
    /// should validate first; checked additions also keep failure nonrefunding.
    pub(super) fn commit_update(&mut self) -> Result<(), ArenaError> {
        let (valid, _, steps) = sampled_auxiliary_counts(&self.pending);
        self.consumed_rows = self
            .consumed_rows
            .checked_add(self.pending.len() as u64)
            .ok_or_else(|| invalid("R consumed count overflow"))?;
        self.valid_rows = self
            .valid_rows
            .checked_add(valid)
            .ok_or_else(|| invalid("R valid count overflow"))?;
        self.completed_nnuec_updates = self
            .completed_nnuec_updates
            .checked_add(steps)
            .ok_or_else(|| invalid("R step count overflow"))?;
        for row in self.pending.drain(..) {
            self.retained_inputs.insert(row.input_sha256);
            self.retained_origins.insert(row.origin_sha256);
        }
        Ok(())
    }
    /// The child retains its complete sampler in the actual tensor state. This
    /// compact projection is correlated with the parent-admitted source,
    /// occurrence counts and seal sets; it is not aliased to our ledger JSON.
    pub(super) fn validate_checkpoint_projection(
        &self,
        projection: &Value,
    ) -> Result<(), ArenaError> {
        fields(
            projection,
            &[
                "schema",
                "recipe_sha256",
                "manifest_file_sha256",
                "index_file_sha256",
                "pool_sha256",
                "sampler_sha256",
                "current_consumed_rows",
                "current_valid_rows",
                "current_nnuec_updates",
                "ancestor_nnuec_updates",
                "retained_input",
                "retained_origin",
                "ancestor_input",
                "ancestor_origin",
                "pc_nnuec_evidence_optimized",
                "execution_authorized",
            ],
        )?;
        check(
            !self.is_pending()
                && projection["schema"] == "rz-rovent-r-auxiliary-checkpoint-projection/1"
                && projection["recipe_sha256"]
                    == digest_value(
                        "rz-rovent-r-auxiliary-recipe-seal/1",
                        &json!({"algorithm":"coefficient-f64-bits-v1","recipe":self.recipe.coefficient_projection()?}),
                    )?
                && projection["manifest_file_sha256"] == self.recipe.manifest_file_sha256
                && projection["index_file_sha256"] == self.recipe.index_file_sha256
                && count(&projection["current_consumed_rows"])? == self.consumed_rows
                && count(&projection["current_valid_rows"])? == self.valid_rows
                && count(&projection["current_nnuec_updates"])? == self.completed_nnuec_updates
                && count(&projection["ancestor_nnuec_updates"])? == self.ancestor_nnuec_updates()
                && projection["pc_nnuec_evidence_optimized"] == false
                && projection["execution_authorized"] == false,
            "R checkpoint projection changed the exact committed source or counters",
        )?;
        for (name, kind, values) in [
            ("retained_input", "input", &self.retained_inputs),
            ("retained_origin", "origin", &self.retained_origins),
            ("ancestor_input", "input", &self.ancestor_inputs),
            ("ancestor_origin", "origin", &self.ancestor_origins),
        ] {
            fields(&projection[name], &["sha256", "count"])?;
            check(
                projection[name]
                    == json!({"sha256":super::checkpoint::set_seal(kind,values)?,"count":values.len()}),
                "R checkpoint projection changed a separate retained/ancestor seal set",
            )?;
        }
        for name in ["pool_sha256", "sampler_sha256"] {
            if !projection[name].is_null() {
                sha(&projection[name])?;
            }
        }
        let current_pool =
            if self.phase == AuxiliaryPhase::Pc && !projection["pool_sha256"].is_null() {
                Some(self.dataset.auxiliary_pool_sha256()?)
            } else {
                None
            };
        validate_phase_sampling(
            self.phase,
            self.preserved_v_sampling.as_ref(),
            projection,
            current_pool.as_deref(),
        )?;
        check(
            (!projection["pool_sha256"].is_null() || projection["sampler_sha256"].is_null())
                && (self.consumed_rows == 0
                    || !projection["pool_sha256"].is_null()
                        && !projection["sampler_sha256"].is_null()),
            "R consumed occurrences require the original pool and actual sampler state seal",
        )
    }
    pub(super) fn observation(&self) -> Value {
        json!({"schema":R_AUXILIARY_LEDGER,"recipe":self.recipe,
        "current_consumed_rows":self.consumed_rows,"current_valid_rows":self.valid_rows,"current_nnuec_updates":self.completed_nnuec_updates,
        "retained_input_sha256":self.retained_inputs,"retained_origin_sha256":self.retained_origins,
        "ancestor_input_sha256":self.ancestor_inputs,"ancestor_origin_sha256":self.ancestor_origins,
        "ancestor_nnuec_updates":self.ancestor_nnuec_updates,"pending_occurrences":self.pending.iter().map(|row|row.projection()).collect::<Vec<_>>(),
        "non_auxiliary_nnuec_ancestry":self.non_auxiliary_nnuec_ancestry,
        "phase":if self.phase==AuxiliaryPhase::Pc{"pc"}else{"v"},
        "preserved_v_sampling_context":self.preserved_v_sampling.as_ref().map(|context|context.observation()),
        "pc_nnuec_evidence_optimized":false,"execution_authorized":false})
    }
}

fn optional_sha(value: &Value) -> Result<Option<String>, ArenaError> {
    if value.is_null() {
        Ok(None)
    } else {
        Ok(Some(sha(value)?.to_owned()))
    }
}

fn validate_phase_sampling(
    phase: AuxiliaryPhase,
    preserved: Option<&PreservedVSamplingContext>,
    projection: &Value,
    current_joint_pool: Option<&str>,
) -> Result<(), ArenaError> {
    let pool = optional_sha(&projection["pool_sha256"])?;
    let sampler = optional_sha(&projection["sampler_sha256"])?;
    match phase {
        AuxiliaryPhase::Pc => {
            check(
                preserved.is_none(),
                "PC cannot use a preserved V sampler context",
            )?;
            if let Some(pool) = pool {
                check(
                    Some(pool.as_str()) == current_joint_pool,
                    "PC R sampler pool differs from exact current joint eligibility",
                )?;
            }
        }
        AuxiliaryPhase::V => {
            let original = preserved
                .ok_or_else(|| invalid("V has no explicitly preserved R sampling context"))?;
            check(
                pool == original.pool_sha256 && sampler == original.sampler_sha256,
                "V changed the original R pool or sampler without consuming R",
            )?;
        }
    }
    Ok(())
}

fn sampled_auxiliary_counts(pending: &[AdmittedAuxiliaryOccurrence]) -> (u64, u64, u64) {
    let valid = pending.iter().filter(|row| row.valid).count() as u64;
    let unique = pending
        .iter()
        .filter(|row| row.valid)
        .map(|row| row.input_sha256.as_str())
        .collect::<BTreeSet<_>>()
        .len() as u64;
    (valid, unique, u64::from(valid > 0))
}

fn has_pc_supervision(valid_heads: &BTreeMap<String, u64>, weight: impl Fn(&str) -> f64) -> bool {
    ["policy", "wdl", "divergence"]
        .iter()
        .any(|head| valid_heads.get(*head).is_some_and(|count| *count > 0) && weight(head) > 0.0)
}

pub(super) fn valid_pc_rows(ledger: &TrainerLedger) -> Result<u64, ArenaError> {
    check(
        ledger.pending.len() == 8,
        "exact eight P/C rows required before R admission",
    )?;
    let mut valid = 0;
    for id in &ledger.pending {
        let row = ledger
            .rows
            .get(id)
            .ok_or_else(|| invalid("pending P/C row missing"))?;
        check(row.role != "validator", "V cannot consume R auxiliary")?;
        if has_pc_supervision(&row.valid_heads, |head| ledger.head_weight(head)) {
            valid += 1;
        }
    }
    Ok(valid)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn row(input: &str, valid: bool) -> AdmittedAuxiliaryOccurrence {
        AdmittedAuxiliaryOccurrence {
            input_sha256: input.repeat(64),
            origin_sha256: "a".repeat(64),
            record_sha256: "b".repeat(64),
            retirement_sha256: "c".repeat(64),
            wdl_raw_sha256: Some("d".repeat(64)),
            valid,
        }
    }
    #[test]
    fn raw_pc_nnuec_evidence_cannot_make_an_auxiliary_only_pc_update() {
        let heads = BTreeMap::from([
            ("policy".into(), 0),
            ("wdl".into(), 0),
            ("divergence".into(), 0),
            ("nnuec_wdl".into(), 1),
        ]);
        assert!(!has_pc_supervision(&heads, |_| 1.0));
        let mut policy = heads;
        policy.insert("policy".into(), 1);
        assert!(!has_pc_supervision(&policy, |head| if head == "policy" {
            0.0
        } else {
            1.0
        }));
        assert!(has_pc_supervision(&policy, |_| 1.0));
    }
    #[test]
    fn retained_raw_receipt_without_exact_trajectory_never_steps_nnuec() {
        let pending = [row("1", false), row("2", false)];
        assert!(pending.iter().all(|row| row.wdl_raw_sha256.is_some()));
        assert_eq!(sampled_auxiliary_counts(&pending), (0, 0, 0));
    }
    #[test]
    fn repeated_valid_samples_count_rows_but_one_unique_input_and_one_step() {
        let pending = [row("1", true), row("1", true), row("2", false)];
        assert_eq!(sampled_auxiliary_counts(&pending), (2, 1, 1));
    }
    #[test]
    fn frozen_v_preserves_sampler_when_only_unused_r_rows_leave_current_pool() {
        let old_pool = "1".repeat(64);
        let new_pool = "2".repeat(64);
        let sampler = "3".repeat(64);
        let projection = json!({"pool_sha256":old_pool,"sampler_sha256":sampler});
        let preserved = PreservedVSamplingContext {
            checkpoint_publication_sha256: Some("4".repeat(64)),
            pool_sha256: Some(old_pool.clone()),
            sampler_sha256: Some(sampler.clone()),
        };
        assert!(
            validate_phase_sampling(
                AuxiliaryPhase::V,
                Some(&preserved),
                &projection,
                Some(&new_pool)
            )
            .is_ok()
        );
        assert!(
            validate_phase_sampling(AuxiliaryPhase::Pc, None, &projection, Some(&new_pool))
                .is_err()
        );
        let mut changed = projection.clone();
        changed["sampler_sha256"] = json!("5".repeat(64));
        assert!(
            validate_phase_sampling(AuxiliaryPhase::V, Some(&preserved), &changed, None).is_err()
        );
        changed = projection;
        changed["pool_sha256"] = json!(new_pool);
        assert!(
            validate_phase_sampling(AuxiliaryPhase::V, Some(&preserved), &changed, None).is_err()
        );
        assert!(validate_phase_sampling(AuxiliaryPhase::V, None, &changed, None).is_err());
        let empty = PreservedVSamplingContext {
            checkpoint_publication_sha256: None,
            pool_sha256: None,
            sampler_sha256: None,
        };
        assert!(
            validate_phase_sampling(
                AuxiliaryPhase::V,
                Some(&empty),
                &json!({"pool_sha256":null,"sampler_sha256":null}),
                None
            )
            .is_ok()
        );
    }
}
