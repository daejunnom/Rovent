//! /2 pair는 독립 원 clock의 duration만 비교한다.
use super::*;
type PairEvidence = (Vec<[u8; 2]>, Vec<String>, Vec<String>);
pub(super) fn verify_pairs(
    publication: &RoventUtilityInterventionPublication,
    files: &BTreeMap<String, Vec<u8>>,
    trials: &[trial::CheckedTrial],
    neutral_digest: &str,
) -> Result<PairEvidence, ArenaError> {
    ensure(
        publication.pairs.len() == trials.len() * (trials.len() - 1) / 2,
        "/2 complete unordered pair coverage",
    )?;
    let mut known = Vec::new();
    let mut known_sha = Vec::new();
    let mut all_sha = Vec::new();
    let mut seen = BTreeSet::new();
    for pin in &publication.pairs {
        let raw = &files[&pin.path];
        let pair: Value = decode(raw)?;
        fields(
            &pair,
            &[
                "schema",
                "cohort_id",
                "neutral_target_sha256",
                "left_trial_sequence",
                "right_trial_sequence",
                "left_trial_sha256",
                "right_trial_sha256",
                "left_recipient",
                "right_recipient",
                "preferred",
                "unknown_reason",
                "support_mask",
                "target_known_mask",
                "left_new_witnesses",
                "right_new_witnesses",
                "left_whole_wall_ns",
                "right_whole_wall_ns",
                "cost_scope",
                "objective",
            ],
        )?;
        ensure(
            string(&pair, "schema")? == "rz-rovent-utility-cohort-pair/2"
                && string(&pair, "cohort_id")? == publication.cohort_id
                && string(&pair, "cost_scope")? == ROVENT_UTILITY_COHORT_COST_SCOPE
                && string(&pair, "objective")? == "exclusive_wall_nanoseconds",
            "/2 original pair domain/cost scope",
        )?;
        let l = usize::try_from(number(&pair, "left_trial_sequence")?).map_err(error)?;
        let r = usize::try_from(number(&pair, "right_trial_sequence")?).map_err(error)?;
        ensure(
            (1..=trials.len()).contains(&l)
                && (1..=trials.len()).contains(&r)
                && l != r
                && seen.insert((l.min(r), l.max(r))),
            "/2 unique independent pair coverage",
        )?;
        let left = &trials[l - 1];
        let right = &trials[r - 1];
        ensure(
            left.digest != right.digest
                && left.authority != right.authority
                && left.cost.clock_epoch != right.cost.clock_epoch
                && left.recipient != right.recipient
                && left.source_budget == right.source_budget,
            "/2 pair independent root/clock and same source budget",
        )?;
        if let (Some(a), Some(b)) = (&left.baseline, &right.baseline) {
            ensure(a == b, "/2 original semantic witness baselines differ")?;
        }
        ensure(
            if left.neutral.is_some() || right.neutral.is_some() {
                hash_field(&pair, "neutral_target_sha256")? == neutral_digest
            } else {
                pair["neutral_target_sha256"].is_null()
            },
            "/2 pair neutral seal",
        )?;
        for (side, t) in [("left", left), ("right", right)] {
            ensure(
                hash_field(&pair, &format!("{side}_trial_sha256"))? == t.digest
                    && route(string(&pair, &format!("{side}_recipient"))?)? == t.recipient
                    && number(&pair, &format!("{side}_whole_wall_ns"))? == t.cost.whole_wall_ns
                    && sha_set(&pair[format!("{side}_new_witnesses")], 64)? == t.witnesses,
                "/2 pair changed original trial/cost/witness",
            )?;
        }
        let (preferred, reason) = if !left.comparable
            || !right.comparable
            || left.neutral.is_none()
            || right.neutral.is_none()
        {
            (None, Some("unresolved_or_masked_trial"))
        } else if left.witnesses.is_empty() && right.witnesses.is_empty() {
            (None, Some("no_new_valid_witness"))
        } else if left.witnesses == right.witnesses
            && left.cost.whole_wall_ns == right.cost.whole_wall_ns
        {
            (None, Some("equal_observation"))
        } else if left.witnesses.is_superset(&right.witnesses)
            && left.cost.whole_wall_ns <= right.cost.whole_wall_ns
        {
            (Some(left.recipient), None)
        } else if right.witnesses.is_superset(&left.witnesses)
            && right.cost.whole_wall_ns <= left.cost.whole_wall_ns
        {
            (Some(right.recipient), None)
        } else {
            (None, Some("pareto_incomparable"))
        };
        ensure(
            pair["unknown_reason"] == serde_json::json!(reason)
                && match preferred {
                    Some(p) => {
                        !pair["preferred"].is_null() && route(string(&pair, "preferred")?)? == p
                    }
                    None => pair["preferred"].is_null(),
                },
            "/2 original Pareto outcome",
        )?;
        let mut support = left.support;
        for (a, b) in support.iter_mut().zip(right.support) {
            *a |= b;
        }
        let mut target = [false; 7];
        if preferred.is_some() {
            target[left.recipient as usize] = true;
            target[right.recipient as usize] = true;
        }
        ensure(
            mask(&pair["support_mask"])? == support && mask(&pair["target_known_mask"])? == target,
            "/2 action support/known pair mask",
        )?;
        let digest = digest_bytes("rz-rovent-utility-cohort-pair/2", raw);
        all_sha.push(digest.clone());
        if let Some(p) = preferred {
            known.push([
                p,
                if p == left.recipient {
                    right.recipient
                } else {
                    left.recipient
                },
            ]);
            known_sha.push(digest);
        }
    }
    Ok((known, known_sha, all_sha))
}
