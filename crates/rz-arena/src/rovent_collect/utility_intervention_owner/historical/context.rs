//! /2 neutral 원 입력과 실제 새 arm의 별도 문맥을 검증한다.
use super::*;
use rz_search::pals::engine::adaptive_port::utility_route::{
    UtilityInterventionHistoricalTargetContent, utility_intervention_historical_snapshot_digest,
    utility_intervention_historical_target_digest,
};
use rz_search::pals::{
    coordinator::RefinementStage, engine::adaptive_port::AdaptiveQuestionDigestPreimage,
};
pub(super) struct Neutral {
    pub wire: Value,
    pub digest: String,
    pub input: shared::CheckedInput,
    pub proposal: Vec<rz_position::BoardMove>,
    pub first: u16,
}
fn move_list(value: &Value) -> Result<Vec<rz_position::BoardMove>, ArenaError> {
    let rows = value.as_array().ok_or_else(|| error("/2 Move16 array"))?;
    ensure(rows.len() <= 256, "/2 bounded Move16 array")?;
    rows.iter()
        .map(|v| {
            packed(
                u16::try_from(v.as_u64().ok_or_else(|| error("/2 Move16 integer"))?)
                    .map_err(error)?,
            )
        })
        .collect()
}
fn replay(root: &Position, moves: &[rz_position::BoardMove]) -> Result<Position, ArenaError> {
    let mut p = Position::from_snapshot(
        &root.snapshot(),
        rz_position::PositionLimits {
            max_history_positions: 16385,
            ..Default::default()
        },
    )
    .map_err(error)?;
    for movement in moves {
        p.make_move(*movement).map_err(error)?;
    }
    Ok(p)
}
pub(super) fn original_source_offer(preparation: &Value) -> Result<&Value, ArenaError> {
    let offers = array(preparation, "offers", 3)?;
    ["route_p", "route_r", "route_c"]
        .into_iter()
        .find_map(|recipient| offers.iter().find(|v| v["recipient"] == recipient))
        .ok_or_else(|| error("/2 actual neutral source absent"))
}
/// Projection contains only fields derived from the independently validated
/// neutral payload and original source; it never creates /1 metadata or a cap.
pub(super) fn verify_neutral(
    value: &Value,
    source_offer: &Value,
    model: &v2::Model,
    root: &RoventRootAdmissionEvidence,
    root_rules: &Position,
    profile: &RoventUtilityInterventionCohortV2,
) -> Result<Neutral, ArenaError> {
    fields(
        value,
        &[
            "domain",
            "input_sha256",
            "target_sha256",
            "context_sha256",
            "target_record",
            "target_record_ids",
            "input_bytes",
            "reference_json",
            "root_snapshot_sha256",
            "source_snapshot_sha256",
            "record_scope_sha256",
            "model_identity",
            "original_first_move",
            "root_rules",
            "records_metadata",
            "source_rules",
            "original_view_sha256",
            "original_full_view_sha256",
        ],
    )?;
    ensure(
        string(value, "domain")? == "rz-rovent-utility-intervention-target/1"
            && number(value, "target_record")? == u64::from(profile.target_record)
            && value["model_identity"] == serde_json::to_value(model.identity()).map_err(error)?
            && value["original_view_sha256"] == source_offer["view"]["view_digest"]
            && value["original_full_view_sha256"] == source_offer["full_view_sha256"]
            && value["record_scope_sha256"] == source_offer["public_context_sha256"]
            && value["original_first_move"] == source_offer["original_first_move"],
        "/2 neutral differs from actual original source",
    )?;
    let target = usize::from(profile.target_record);
    let metadata = array(value, "records_metadata", 64)?;
    let first = u16::try_from(number(value, "original_first_move")?).map_err(error)?;
    let selected = metadata
        .get(target)
        .ok_or_else(|| error("/2 neutral target record"))?;
    ensure(
        number(selected, "anchor")? == 0
            && selected["original_first_move"] == serde_json::json!(first),
        "/2 current-root proposal target",
    )?;
    let reference: v2::utility::Reference = decode(string(value, "reference_json")?.as_bytes())?;
    let line = reference
        .base_input
        .context_lines
        .get(target)
        .ok_or_else(|| error("/2 original neutral line"))?;
    ensure(
        line.kind == 0 && !line.moves.is_empty(),
        "/2 proposal target kind/nonempty line",
    )?;
    let proposal = line
        .moves
        .iter()
        .map(|m| {
            let promotion = match m.promotion {
                0 => None,
                1 => Some(rz_position::PieceKind::Queen),
                2 => Some(rz_position::PieceKind::Rook),
                3 => Some(rz_position::PieceKind::Bishop),
                4 => Some(rz_position::PieceKind::Knight),
                _ => return Err(error("/2 numerical promotion")),
            };
            rz_position::BoardMove::new(
                rz_position::Square::new(m.from_square).map_err(error)?,
                rz_position::Square::new(m.to_square).map_err(error)?,
                promotion,
            )
            .map_err(error)
        })
        .collect::<Result<Vec<_>, _>>()?;
    ensure(
        proposal.first() == Some(&packed(first)?),
        "/2 proposal first move",
    )?;
    let mut submission = Sha256::new();
    submission.update(v2::utility_capture::SUBMISSION_TARGET_CAPTURE_DOMAIN.as_bytes());
    submission.update([0]);
    submission.update(hash32(hash_field(value, "original_view_sha256")?)?.0);
    submission.update(hash_field(value, "target_sha256")?.as_bytes());
    let projection = serde_json::json!({"reference_json":value["reference_json"],"bytes":value["input_bytes"],"input_sha256":value["input_sha256"],
        "target_sha256":value["target_sha256"],"submission_target_sha256":format!("{:x}",submission.finalize()),"context_sha256":value["context_sha256"],
        "target_record":value["target_record"],"target_record_ids":value["target_record_ids"],"base_identity":value["model_identity"],
        "root_rules":value["root_rules"],"records_metadata":value["records_metadata"],"source_rules":value["source_rules"],
        "provenance":{"root_state":root.store_root_state,"generation":root.root_generation,"sequence":source_offer["view"]["sequence"],
            "view_seal_sha256":value["original_view_sha256"],"full_view_sha256":value["original_full_view_sha256"],
            "record_scope_sha256":value["record_scope_sha256"],"policy_sha256":source_offer["view"]["context_policy_sha256"],"model":value["model_identity"]}});
    let input = shared::verify_captured_utility_input(&projection, model, root_rules)?;
    shared::verify_submitted_input_source(&projection, source_offer, root)?;
    let actual_root = shared::restore_rules_snapshot(&value["root_rules"])?;
    ensure(
        hex(
            utility_intervention_historical_snapshot_digest(&actual_root.snapshot())
                .map_err(error)?,
        ) == hash_field(value, "root_snapshot_sha256")?,
        "/2 exact root snapshot preimage",
    )?;
    let mut source_snapshots = Vec::new();
    let digests = array(value, "source_snapshot_sha256", 64)?;
    let source_rules = array(value, "source_rules", 64)?;
    ensure(
        digests.len() == source_rules.len(),
        "/2 source snapshot coverage",
    )?;
    for (source, digest) in source_rules.iter().zip(digests) {
        let snapshot = shared::restore_rules_snapshot(source)?.snapshot();
        ensure(
            digest.as_str()
                == Some(
                    hex(utility_intervention_historical_snapshot_digest(&snapshot)
                        .map_err(error)?)
                    .as_str(),
                ),
            "/2 original source snapshot digest",
        )?;
        source_snapshots.push(snapshot);
    }
    let digest = hex(utility_intervention_historical_target_digest(
        UtilityInterventionHistoricalTargetContent {
            root_snapshot: &actual_root.snapshot(),
            record_scope_digest: hash32(hash_field(value, "record_scope_sha256")?)?,
            target_record: profile.target_record,
            original_first_move: Move16::try_from_bits(first).map_err(error)?,
            reference_json: input.reference_json.as_bytes(),
            model_identity: model.identity(),
            source_snapshots: &source_snapshots,
        },
    )
    .map_err(error)?);
    Ok(Neutral {
        wire: value.clone(),
        digest,
        input,
        proposal,
        first,
    })
}
/// A fresh role query is checked separately; it is never a fabricated Ready offer.
pub(super) fn verify_arm(
    arm: &Value,
    neutral: &Neutral,
    source_offer: &Value,
    recipient: u8,
    model: &v2::Model,
    root: &RoventRootAdmissionEvidence,
    root_rules: &Position,
) -> Result<(), ArenaError> {
    fields(
        arm,
        &[
            "neutral_target_sha256",
            "arm_query_sha256",
            "recipient",
            "frame_sequence",
            "source_frame_sequence",
            "stage",
            "purpose",
            "original_first_move",
            "view",
            "full_view_sha256",
            "public_context_sha256",
            "budget",
            "deadline_clock",
            "deadline_tick",
            "query",
            "prepared_input",
        ],
    )?;
    let stage = match recipient {
        4 => RefinementStage::ProposalContinuation,
        5 => RefinementStage::CriticDivergence,
        6 => RefinementStage::CpuValidation,
        _ => return Err(error("/2 arm recipient")),
    };
    ensure(
        hash_field(arm, "neutral_target_sha256")? == neutral.digest
            && route(string(arm, "recipient")?)? == recipient
            && number(arm, "frame_sequence")? > number(source_offer, "frame_sequence")?
            && number(arm, "source_frame_sequence")? == number(source_offer, "frame_sequence")?
            && string(arm, "stage")? == format!("{stage:?}")
            && string(arm, "purpose")?
                == if recipient == 5 {
                    "DivergencePolicy"
                } else {
                    "ProposePolicy"
                }
            && number(arm, "original_first_move")? == u64::from(neutral.first)
            && arm["public_context_sha256"] == neutral.wire["record_scope_sha256"]
            && arm["budget"] == source_offer["budget"]
            && number(arm, "deadline_clock")? == root.deadline_clock
            && number(arm, "deadline_tick")? == root.deadline_tick,
        "/2 actual arm source/budget/deadline",
    )?;
    shared::checked_view(&arm["view"], &authority(root))?;
    ensure(
        number(&arm["view"]["root_state"], "owner")? == root.root_state.owner
            && number(&arm["view"]["root_state"], "revision")? == root.root_state.revision
            && hash_field(&arm["view"]["root_state"], "semantic_sha256")?
                == root.root_state.semantic,
        "/2 arm original root view",
    )?;
    let query = &arm["query"];
    fields(
        query,
        &[
            "endpoint_rules",
            "prefix",
            "proposal",
            "counterexample",
            "legal",
            "divergences",
            "question_preimage",
        ],
    )?;
    let prefix = move_list(&query["prefix"])?;
    let proposal = move_list(&query["proposal"])?;
    let legal = move_list(&query["legal"])?;
    let first = [packed(neutral.first)?];
    let expected_prefix = match recipient {
        4 => first.as_slice(),
        5 => &[][..],
        6 => neutral.proposal.as_slice(),
        _ => return Err(error("/2 arm prefix recipient")),
    };
    ensure(
        prefix == expected_prefix
            && proposal == neutral.proposal
            && query["counterexample"].is_null(),
        "/2 frozen-proposal arm sequences",
    )?;
    let endpoint = replay(root_rules, &prefix)?;
    if recipient != 5 {
        ensure(
            matches!(
                endpoint.classify_position().map_err(error)?.play_status,
                rz_position::PlayStatus::Ongoing
            ),
            "/2 issued P/R arm cannot start at a Rules terminal endpoint",
        )?;
    }
    ensure(
        if recipient == 6 {
            number(&arm["budget"], "original_cpu_credit")? > 0
        } else {
            number(&arm["budget"], "role_submissions_remaining")? > 0
        },
        "/2 issued arm requires its original declared credit",
    )?;
    ensure(
        shared::restore_rules_snapshot(&query["endpoint_rules"])?
            .snapshot()
            .same_state(&endpoint.snapshot()),
        "/2 actual arm endpoint Rules",
    )?;
    let expected_legal = if recipient == 5 {
        Vec::new()
    } else {
        endpoint.ordered_legal_moves().moves().to_vec()
    };
    ensure(legal == expected_legal, "/2 arm legal candidate order")?;
    let mut divergences = Vec::new();
    let mut state = Position::from_snapshot(
        &root_rules.snapshot(),
        rz_position::PositionLimits {
            max_history_positions: 16385,
            ..Default::default()
        },
    )
    .map_err(error)?;
    for (ply, movement) in neutral.proposal.iter().enumerate() {
        if recipient == 5
            && state.side_to_move() != root_rules.side_to_move()
            && matches!(
                state.classify_position().map_err(error)?.play_status,
                rz_position::PlayStatus::Ongoing
            )
        {
            divergences.push(ply);
        }
        state.make_move(*movement).map_err(error)?;
    }
    ensure(
        query["divergences"] == serde_json::json!(divergences),
        "/2 arm real opponent divergence indices",
    )?;
    ensure(
        recipient != 5 || !divergences.is_empty(),
        "/2 issued C arm requires an actual opponent divergence",
    )?;
    let p = &query["question_preimage"];
    fields(
        p,
        &[
            "sequence",
            "stage_code",
            "view_digest",
            "view_seal_digest",
            "focus_prefix_sha256",
            "goal_local_sha256",
            "legacy_full_records_sha256",
        ],
    )?;
    let optional = |key| -> Result<Option<Digest>, ArenaError> {
        if p[key].is_null() {
            Ok(None)
        } else {
            Ok(Some(hash32(hash_field(p, key)?)?))
        }
    };
    let preimage = AdaptiveQuestionDigestPreimage {
        sequence: number(p, "sequence")?,
        stage_code: u8::try_from(number(p, "stage_code")?).map_err(error)?,
        view_digest: hash32(hash_field(p, "view_digest")?)?,
        view_seal_digest: hash32(hash_field(p, "view_seal_digest")?)?,
        focus_prefix_sha256: hash32(hash_field(p, "focus_prefix_sha256")?)?,
        goal_local_sha256: optional("goal_local_sha256")?,
        legacy_full_records_sha256: optional("legacy_full_records_sha256")?,
    };
    ensure(
        preimage.sequence > 0
            && preimage.stage_code == stage as u8
            && hex(preimage.view_digest) == hash_field(arm, "full_view_sha256")?
            && hex(preimage.view_seal_digest) == hash_field(&arm["view"], "view_digest")?
            && hex(preimage.digest()) == hash_field(arm, "arm_query_sha256")?,
        "/2 arm actual question preimage",
    )?;
    if !arm["prepared_input"].is_null() {
        let input =
            shared::verify_captured_utility_input(&arm["prepared_input"], model, root_rules)?;
        shared::verify_submitted_input_source(&arm["prepared_input"], arm, root)?;
        ensure(
            input.reference_json == neutral.input.reference_json
                && input.input_sha256 == neutral.input.input_sha256
                && input.target_sha256 == neutral.input.target_sha256
                && input.context_sha256 == neutral.input.context_sha256
                && arm["prepared_input"]["provenance"]["record_scope_sha256"]
                    == neutral.wire["record_scope_sha256"],
            "/2 arm actual preparation changed neutral NN input",
        )?;
    }
    Ok(())
}
