//! /2 trial은 원 독립 root와 실제 arm을 각각 검증한다.
use super::*;

const TRIAL_DOMAIN: &str = "rz-rovent-utility-cohort-trial/2";
const MATERIALIZATION_DOMAIN: &str = "rz-rovent-utility-cohort-materialization/2";

pub(super) struct CheckedTrial {
    pub cost: rz_uci::pals_adaptive::UtilityCohortCostWire,
    pub digest: String,
    pub recipient: u8,
    pub support: [bool; 7],
    pub comparable: bool,
    pub witnesses: BTreeSet<String>,
    pub neutral: Option<context::Neutral>,
    pub authority: Value,
    pub source_budget: Value,
    pub baseline: Option<BTreeSet<String>>,
    pub role_query: Option<
        rz_experiments::rovent_utility_intervention_preparation::RoventInterventionRoleQuery,
    >,
}

// These are independently pinned operands, not a replacement authority DTO.
#[allow(clippy::too_many_arguments)]
pub(super) fn verify_trial(
    ordinal: usize,
    publication: &RoventUtilityInterventionPublication,
    files: &BTreeMap<String, Vec<u8>>,
    profile: &RoventUtilityInterventionCohortV2,
    source: &RoventUtilityInterventionCollectionSource,
    lock: &RoventInputLock,
    model: &v2::Model,
    root_rules: &Position,
) -> Result<CheckedTrial, ArenaError> {
    let pin = &publication.trials[ordinal];
    let raw = &files[&pin.path];
    let wire: Value = decode(raw)?;
    fields(
        &wire,
        &[
            "schema",
            "cohort_id",
            "trial_sequence",
            "recipient",
            "profile",
            "profile_sha256",
            "source",
            "neutral_target",
            "neutral_target_sha256",
            "arm_query",
            "materialized_sha256",
            "materialized_json",
            "historical_unbound_input",
            "unresolved_source",
            "cost",
            "original_root",
            "preparation",
            "physical_closure",
            "previous_durable_ack",
            "resolved",
            "comparable",
            "unknown_reason",
            "support_mask",
            "target_known_mask",
        ],
    )?;
    let recipient = route(string(&wire, "recipient")?)?;
    ensure(
        string(&wire, "schema")? == TRIAL_DOMAIN
            && string(&wire, "cohort_id")? == publication.cohort_id
            && number(&wire, "trial_sequence")? == ordinal as u64 + 1
            && same(&wire["profile"], profile)?
            && hash_field(&wire, "profile_sha256")? == hex(profile.digest()?)
            && same(&wire["source"], &source.source_pins)?
            && requested(profile).get(ordinal) == Some(&recipient),
        "/2 original trial order/profile/source/recipient",
    )?;
    let root: RoventRootAdmissionEvidence =
        serde_json::from_value(wire["original_root"].clone()).map_err(error)?;
    ensure(
        same(&root, &wire["original_root"])?,
        "/2 closed original per-trial root",
    )?;
    verify_root(&root, &wire["original_root"], lock, &source.source_pins)?;
    verify_root_rules(&root, root_rules)?;
    let mut native = wire["original_root"]["native_origin"].clone();
    let mut original = source.native_origin.clone();
    native
        .as_object_mut()
        .ok_or_else(|| error("/2 original native source object"))?
        .remove("process_epoch");
    original
        .as_object_mut()
        .ok_or_else(|| error("/2 registered native source object"))?
        .remove("process_epoch");
    ensure(
        native == original,
        "/2 independent root changed original source",
    )?;
    if ordinal == 0 {
        ensure(
            wire["original_root"] == source.registration_root,
            "/2 first original registration root",
        )?;
    }
    let resolved = boolean(&wire, "resolved")?;
    let comparable = boolean(&wire, "comparable")?;
    let support = mask(&wire["support_mask"])?;
    let declared = requested(profile);
    ensure(
        mask(&wire["target_known_mask"])? == [false; 7]
            && support
                .iter()
                .enumerate()
                .all(|(i, yes)| !yes || declared.contains(&(i as u8)))
            && (!(resolved || comparable)
                || (support[recipient as usize] && support.iter().filter(|b| **b).count() >= 2)),
        "/2 actual support intersection/unknown target mask",
    )?;
    let reason = &wire["unknown_reason"];
    ensure(
        comparable == reason.is_null()
            && (reason.is_null()
                || reason
                    .as_str()
                    .is_some_and(|s| !s.is_empty() && s.len() <= 256)),
        "/2 comparable/unknown observation",
    )?;
    let preparation = &wire["preparation"];
    let prelude = shared::verify_prelude(preparation, &root)?;
    let source_offer = context::original_source_offer(preparation)?;
    let raw_cost = &wire["cost"];
    let opened = number(raw_cost, "started_tick")?.checked_add(profile.input.action_ns)
        == Some(number(raw_cost, "cohort_action_cutoff_tick")?)
        && number(raw_cost, "cohort_action_cutoff_tick")?.checked_add(profile.input.cleanup_ns)
            == Some(number(raw_cost, "cohort_cleanup_cutoff_tick")?);
    let cost = shared::verify_cost_window_input(raw_cost, &profile.input, opened)?;
    ensure(
        cost.clock_epoch == root.deadline_clock
            && cost.original_deadline_tick == root.deadline_tick
            && cost.cohort_action_cutoff_tick <= root.deadline_tick
            && cost.cohort_cleanup_cutoff_tick <= root.root_descriptor.input.clock.cleanup.at_ns
            && cost.started_tick >= number(preparation, "captured_tick")?
            && cost.baseline_counters == preparation["counters"]
            && cost.native_credit_baseline
                == (
                    number(&preparation["native_credit"], "spent")?,
                    number(&preparation["native_credit"], "reserved")?,
                ),
        "/2 cost retains original source baseline and clock",
    )?;
    if !opened {
        ensure(
            wire["neutral_target"].is_null()
                && wire["arm_query"].is_null()
                && wire["materialized_json"].is_null()
                && !resolved
                && !comparable
                && !cost.complete_scope
                && cost.started_tick == number(preparation, "captured_tick")?
                && cost.cohort_action_cutoff_tick == root.deadline_tick
                && cost.cohort_cleanup_cutoff_tick
                    == root.root_descriptor.input.clock.cleanup.at_ns,
            "/2 unstarted gate cannot become an action observation",
        )?;
    }
    let previous = &wire["previous_durable_ack"];
    if ordinal == 0 {
        ensure(previous.is_null(), "/2 first trial has no preceding ACK")?;
    } else {
        fields(previous, &["trial_sequence", "trial_sha256", "file_sha256"])?;
        let prior = &publication.trials[ordinal - 1];
        ensure(
            number(previous, "trial_sequence")? == ordinal as u64
                && hash_field(previous, "trial_sha256")?
                    == digest_bytes(TRIAL_DOMAIN, &files[&prior.path])
                && hash_field(previous, "file_sha256")? == prior.sha256,
            "/2 original durable ACK chain",
        )?;
    }
    let physical = shared::verify_physical(&wire, &cost, model, Some(&root))?;
    let neutral = if wire["neutral_target"].is_null() {
        ensure(
            wire["neutral_target_sha256"].is_null()
                && wire["arm_query"].is_null()
                && wire["materialized_json"].is_null()
                && !resolved
                && !comparable,
            "/2 missing neutral target stays unknown",
        )?;
        None
    } else {
        let n = context::verify_neutral(
            &wire["neutral_target"],
            source_offer,
            model,
            &root,
            root_rules,
            profile,
        )?;
        ensure(
            hash_field(&wire, "neutral_target_sha256")? == n.digest,
            "/2 original neutral digest",
        )?;
        Some(n)
    };
    let arm = &wire["arm_query"];
    if arm.is_null() {
        ensure(
            !resolved && !comparable && wire["materialized_json"].is_null(),
            "/2 absent actual arm stays unknown",
        )?;
    } else {
        let n = neutral
            .as_ref()
            .ok_or_else(|| error("/2 actual arm missing neutral source"))?;
        ensure(
            support[recipient as usize] && support.iter().filter(|b| **b).count() >= 2,
            "/2 actual arm lacks supported alternatives",
        )?;
        context::verify_arm(arm, n, source_offer, recipient, model, &root, root_rules)?;
    }
    let materialization_path = format!("materializations/{ordinal:06}.json");
    let materialized_pin = publication
        .materializations
        .iter()
        .find(|p| p.path == materialization_path);
    let mut semantic = None;
    let mut baseline = None;
    if !wire["materialized_json"].is_null() {
        let n = neutral
            .as_ref()
            .ok_or_else(|| error("/2 materialization missing neutral"))?;
        ensure(
            !arm.is_null() && !arm["prepared_input"].is_null(),
            "/2 materialization missing original prepared arm",
        )?;
        let text = string(&wire, "materialized_json")?;
        let pinned = materialized_pin.ok_or_else(|| error("/2 materialization file missing"))?;
        ensure(
            text.as_bytes() == files[&pinned.path]
                && hash_field(&wire, "materialized_sha256")?
                    == digest_bytes(MATERIALIZATION_DOMAIN, text.as_bytes()),
            "/2 embedded materialization original bytes/domain",
        )?;
        let body: Value = decode(text.as_bytes())?;
        fields(
            &body,
            &[
                "schema",
                "cohort_id",
                "trial_sequence",
                "recipient",
                "profile_sha256",
                "source",
                "original_root",
                "preparation",
                "neutral_target",
                "arm_query",
                "action",
                "semantic",
            ],
        )?;
        ensure(
            string(&body, "schema")? == MATERIALIZATION_DOMAIN,
            "/2 materialization domain",
        )?;
        for key in [
            "cohort_id",
            "trial_sequence",
            "recipient",
            "profile_sha256",
            "source",
            "original_root",
            "preparation",
            "neutral_target",
            "arm_query",
        ] {
            ensure(
                body[key] == wire[key],
                "/2 materialization original trial join",
            )?;
        }
        let action = &body["action"];
        fields(
            action,
            &[
                "authority",
                "frame_sequence",
                "question_sha256",
                "root_budget_sha256",
                "original_first_move",
                "deadline_clock",
                "deadline_tick",
                "frozen_context_sha256",
                "cost_epoch_sha256",
                "cost_scope",
            ],
        )?;
        ensure(
            action["authority"] == authority(&root)
                && action["frame_sequence"] == arm["frame_sequence"]
                && hash_field(action, "question_sha256")? == hash_field(arm, "arm_query_sha256")?
                && hash_field(action, "root_budget_sha256")? == root.root_budget_sha256
                && number(action, "original_first_move")? == u64::from(n.first)
                && number(action, "deadline_clock")? == root.deadline_clock
                && number(action, "deadline_tick")? == root.deadline_tick
                && hash_field(action, "frozen_context_sha256")? == n.digest
                && cost.cost_epoch_sha256.as_deref()
                    == Some(hash_field(action, "cost_epoch_sha256")?)
                && string(action, "cost_scope")? == ROVENT_UTILITY_COHORT_COST_SCOPE,
            "/2 actual action/root/question/neutral/original cost epoch",
        )?;
        baseline = Some(sha_set(&body["semantic"]["baseline_witnesses"], 65536)?);
        semantic = Some(shared::verify_semantic_with_store_limit(
            &body,
            &root,
            root_rules,
            &physical,
            model,
            number(&arm["budget"], "original_store_byte_limit")?,
        )?);
    } else {
        ensure(
            wire["materialized_sha256"].is_null() && materialized_pin.is_none() && !resolved,
            "/2 absent materialization stays unresolved",
        )?;
    }
    if !wire["historical_unbound_input"].is_null() {
        verify_unbound(&wire["historical_unbound_input"], neutral.as_ref(), arm)?;
    }
    if !wire["unresolved_source"].is_null() {
        let unresolved = &wire["unresolved_source"];
        shared::verify_unresolved(unresolved, recipient)?;
        ensure(
            !resolved
                && !comparable
                && hash_field(unresolved, "root_budget_sha256")? == root.root_budget_sha256,
            "/2 unresolved original root cannot become complete",
        )?;
        if let Some(n) = &neutral {
            ensure(
                hash_field(unresolved, "frozen_context_sha256")? == n.digest,
                "/2 unresolved neutral seal",
            )?;
        }
        if !arm.is_null() {
            ensure(
                unresolved["frame_sequence"] == arm["frame_sequence"]
                    && unresolved["question_sha256"] == arm["arm_query_sha256"],
                "/2 unresolved original arm",
            )?;
        }
    }
    if comparable {
        ensure(
            profile.input.usage == RoventUtilityCohortUsage::CollectionCandidate
                && resolved
                && cost.complete_scope
                && semantic.as_ref().is_some_and(|s| s.bodies_complete)
                && prelude.resource_known
                && physical.cache_known
                && physical.fresh_product
                && cost.physical_pin_zero_tick <= cost.cohort_cleanup_cutoff_tick,
            "/2 comparable trial lacks original complete known scope",
        )?;
        for key in [
            "nn_input_count_unknown",
            "accepted_nn_input_count_unknown",
            "consumed_nn_input_count_unknown",
            "rejected_events",
        ] {
            ensure(
                cost.baseline_counters[key] == cost.final_counters[key],
                "/2 comparable trial unknown/rejected work",
            )?;
        }
    }
    let role_query = if arm.is_null() {
        None
    } else {
        use rz_experiments::rovent_utility_intervention_preparation::{
            RoventInterventionRecipient, RoventInterventionRoleQuery,
        };
        Some(RoventInterventionRoleQuery {
            trial_sequence: ordinal as u64 + 1,
            recipient: match recipient {
                4 => RoventInterventionRecipient::Proposer,
                5 => RoventInterventionRecipient::Critic,
                6 => RoventInterventionRecipient::Researcher,
                _ => return Err(error("/2 original issued query recipient")),
            },
            question_sha256: hash_field(arm, "arm_query_sha256")?.into(),
            view_sha256: hash_field(&arm["view"], "view_digest")?.into(),
            full_view_sha256: hash_field(arm, "full_view_sha256")?.into(),
        })
    };
    Ok(CheckedTrial {
        cost,
        digest: digest_bytes(TRIAL_DOMAIN, raw),
        recipient,
        support,
        comparable,
        witnesses: semantic.map(|s| s.witnesses).unwrap_or_default(),
        neutral,
        authority: physical.authority,
        source_budget: source_offer["budget"].clone(),
        baseline,
        role_query,
    })
}

fn verify_unbound(
    value: &Value,
    neutral: Option<&context::Neutral>,
    arm: &Value,
) -> Result<(), ArenaError> {
    fields(
        value,
        &[
            "reference_json",
            "input_sha256",
            "target_sha256",
            "submission_target_sha256",
            "context_sha256",
            "target_record_ids",
            "original_view_sha256",
            "original_full_view_sha256",
            "materialized_inside_complete_scope",
        ],
    )?;
    let n = neutral.ok_or_else(|| error("/2 unbound history missing original neutral"))?;
    ensure(
        value["materialized_inside_complete_scope"] == false
            && string(value, "reference_json")? == n.input.reference_json
            && hash_field(value, "input_sha256")? == n.input.input_sha256
            && hash_field(value, "target_sha256")? == n.input.target_sha256
            && hash_field(value, "context_sha256")? == n.input.context_sha256
            && value["target_record_ids"] == n.wire["target_record_ids"],
        "/2 passive unbound original numerical input",
    )?;
    // A passive duplicate may be the original neutral capture or the explicitly
    // prepared arm. Neither path creates a semantic completion or live cap.
    let neutral_source = hash_field(value, "submission_target_sha256")?
        == n.input.submission_target_sha256
        && value["original_view_sha256"] == n.wire["original_view_sha256"]
        && value["original_full_view_sha256"] == n.wire["original_full_view_sha256"];
    let input = &arm["prepared_input"];
    let arm_source = !arm.is_null()
        && !input.is_null()
        && value["submission_target_sha256"] == input["submission_target_sha256"]
        && value["original_view_sha256"] == input["provenance"]["view_seal_sha256"]
        && value["original_full_view_sha256"] == input["provenance"]["full_view_sha256"];
    ensure(
        neutral_source || arm_source,
        "/2 unbound original capture provenance",
    )
}
