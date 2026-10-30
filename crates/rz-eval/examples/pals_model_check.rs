//! Independent PyTorch reference -> frozen Rust/ORT P/C acceptance.
//! Generated fixtures and reports belong to caller-owned external roots.
use rz_eval::asset;
use rz_eval::error::{BackendError, FailureKind, FailureStage};
use rz_eval::onnx::{OrtRuntime, Provider};
use rz_eval::pals_model::{
    PalsModelConfig, PalsModelInput, PalsModelProfile, PalsRawOutput, PalsRecordLine, PalsRole,
};
use rz_eval::pals_onnx::{
    HostRecordPagePolicy, HostRecordPageSnapshot, PalsBackendStats, PalsCuDnnConvSearch,
    PalsCudaControlPolicy, PalsNativeCommand, PalsNativeResult, PalsOnnxBackend, PalsOnnxConfig,
    PalsPublicMemoryWitness,
};
use rz_eval::runtime_pin::{CudaRuntimeBundleSpec, RuntimeBundleFileRole, RuntimeCache};
use rz_eval::worker::PhysicalPoll;
use rz_native_loader::{NativeLoadingProfile, NativeMappingObservation};
use serde::Deserialize;
use serde_json::json;
use std::error::Error;
use std::io::Write;
use std::path::Path;
use std::task::Poll;
use std::time::{Duration, Instant};

#[path = "support/native_startup_trace.rs"]
mod native_startup_trace;
use native_startup_trace::NativeStartupTrace;

#[cfg(feature = "experimental-io-binding")]
#[path = "pals_model_check/resident_cuda.rs"]
mod resident_cuda;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Fixtures {
    schema: String,
    #[serde(default)]
    config: PalsModelConfig,
    #[serde(default)]
    model_profile: Option<PalsModelProfile>,
    #[serde(default)]
    encoding_schema: Option<String>,
    checkpoint_sha256: String,
    rules_certified: bool,
    trained: bool,
    reference: String,
    cases: Vec<Case>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Case {
    name: String,
    input: PalsModelInput,
    expected: PalsRawOutput,
    public_memory: PalsPublicMemoryWitness,
}

fn validate_fixture_coverage(fixtures: &Fixtures) -> Result<(), Box<dyn Error>> {
    fixtures.config.validate()?;
    let profile = fixtures.config.profile;
    if !profile.is_legacy()
        && (fixtures.model_profile != Some(profile)
            || fixtures.encoding_schema.as_deref() != Some(profile.encoding_schema()))
    {
        return Err("V2 reference profile/encoding declaration differs".into());
    }
    if profile.is_legacy()
        && (fixtures.model_profile.is_some_and(|p| !p.is_legacy())
            || fixtures
                .encoding_schema
                .as_deref()
                .is_some_and(|s| s != profile.encoding_schema()))
    {
        return Err("legacy reference cannot declare V2 tensor semantics".into());
    }
    let mut required = vec![
        ("empty_context", PalsRole::Proposer, 0, 0, 0),
        ("critic_divergence", PalsRole::Critic, 3, 7, 4),
        ("proposer_order", PalsRole::Proposer, 3, 7, 0),
        ("proposer_order_reversed", PalsRole::Proposer, 3, 7, 0),
        ("same_board_other_history", PalsRole::Proposer, 3, 7, 0),
        ("all_promotions", PalsRole::Critic, 1, 4, 1),
    ];
    if profile.uses_full_line() {
        required.extend([
            ("full_middle_record_changed", PalsRole::Critic, 3, 7, 4),
            ("full_middle_query_changed", PalsRole::Proposer, 3, 7, 0),
            ("record_relations_changed", PalsRole::Critic, 3, 7, 4),
            ("empty_line_padding", PalsRole::Proposer, 1, 1, 0),
        ]);
    }
    if fixtures.schema != profile.model_semantics()
        || fixtures.rules_certified
        || fixtures.reference != "pytorch_fp32_tf32_off"
        || fixtures.cases.len() != required.len()
    {
        return Err("expected registered six-case P/C numeric-only reference fixtures".into());
    }
    for (name, role, records, candidates, divergences) in required {
        let matching: Vec<_> = fixtures
            .cases
            .iter()
            .filter(|case| case.name == name)
            .collect();
        if matching.len() != 1 {
            return Err("P/C numeric reference case is missing or duplicated".into());
        }
        let input = &matching[0].input;
        input.validate(&fixtures.config)?;
        if input.role != role
            || input.records.len() != records
            || input.candidates.len() != candidates
            || input.divergence_features.len() != divergences
        {
            return Err("P/C numeric reference case does not cover its declared role/shape".into());
        }
        let witness = &matching[0].public_memory;
        let tokens = fixtures.config.public_memory_tokens(records)?;
        if witness.tokens != tokens
            || witness.memory_key.len() != 2 * tokens * 64
            || witness.memory_value.len() != 2 * tokens * 64
            || witness.memory_mask.len() != tokens
            || witness
                .memory_key
                .iter()
                .chain(&witness.memory_value)
                .any(|value| !value.is_finite())
            || witness.memory_mask[..66].iter().any(|value| !*value)
            || witness.memory_mask[66..]
                .iter()
                .filter(|value| **value)
                .count()
                != records
        {
            return Err("public-memory reference witness shape/finite/mask differs".into());
        }
    }
    let case = |name: &str| {
        fixtures
            .cases
            .iter()
            .find(|case| case.name == name)
            .expect("coverage verified")
    };
    let ordered = &case("proposer_order").input;
    let mut reversed = ordered.clone();
    reversed.candidates.reverse();
    if reversed != case("proposer_order_reversed").input {
        return Err("candidate order comparison changed another input condition".into());
    }
    let history = &case("same_board_other_history").input;
    let mut other_history = ordered.clone();
    other_history.history_digest = history.history_digest;
    if ordered.history_digest == history.history_digest || other_history != *history {
        return Err("history comparison did not preserve the same board/other inputs".into());
    }
    let promotions = &case("all_promotions").input.candidates;
    if !promotions.iter().enumerate().all(|(index, candidate)| {
        candidate.from == 48 && candidate.to == 56 && candidate.promotion == index as u8 + 1
    }) {
        return Err("promotion reference does not cover queen/rook/bishop/knight order".into());
    }
    if profile.uses_full_line() {
        let middle_changed =
            |before: &[rz_eval::pals_model::PalsCandidateToken],
             after: &[rz_eval::pals_model::PalsCandidateToken]| {
                before.len() >= 3
                    && before.len() == after.len()
                    && before.first() == after.first()
                    && before.last() == after.last()
                    && before[1..before.len() - 1] != after[1..after.len() - 1]
            };
        let record = &case("full_middle_record_changed").input;
        let base = &case("critic_divergence").input;
        let mut restored = record.clone();
        restored
            .full_line
            .as_mut()
            .ok_or("record mutation full_line missing")?
            .records = base
            .full_line
            .as_ref()
            .ok_or("base full_line missing")?
            .records
            .clone();
        if restored != *base
            || !record
                .full_line
                .as_ref()
                .unwrap()
                .records
                .iter()
                .zip(&base.full_line.as_ref().unwrap().records)
                .any(|(after, before)| middle_changed(&before.moves, &after.moves))
        {
            return Err(
                "record-middle comparison does not isolate equal-endpoint full sequence changes"
                    .into(),
            );
        }
        let query = &case("full_middle_query_changed").input;
        let base = &case("proposer_order").input;
        let after = query
            .full_line
            .as_ref()
            .ok_or("query mutation full_line missing")?;
        let before = base
            .full_line
            .as_ref()
            .ok_or("query base full_line missing")?;
        let mut restored = query.clone();
        let lines = restored.full_line.as_mut().unwrap();
        lines.query_prefix = before.query_prefix.clone();
        lines.query_proposal = before.query_proposal.clone();
        lines.query_counter = before.query_counter.clone();
        if restored != *base
            || ![
                (&before.query_prefix, &after.query_prefix),
                (&before.query_proposal, &after.query_proposal),
                (&before.query_counter, &after.query_counter),
            ]
            .into_iter()
            .any(|(before, after)| middle_changed(before, after))
            || case("full_middle_query_changed").public_memory
                != case("proposer_order").public_memory
        {
            return Err(
                "query-middle comparison changed public memory or another model input".into(),
            );
        }
        let relation = &case("record_relations_changed").input;
        let base = &case("critic_divergence").input;
        let mut restored = relation.clone();
        let changed_lines = restored
            .full_line
            .as_mut()
            .ok_or("relation mutation full_line missing")?;
        let base_lines = base.full_line.as_ref().unwrap();
        for (changed, before) in changed_lines.records.iter_mut().zip(&base_lines.records) {
            changed.parent = before.parent;
            changed.supersedes = before.supersedes;
            changed.parent_required = before.parent_required;
            changed.supersedes_required = before.supersedes_required;
        }
        if relation == base
            || restored != *base
            || case("record_relations_changed").public_memory
                != case("critic_divergence").public_memory
        {
            return Err(
                "private relationship mutation changed independent public encoder inputs".into(),
            );
        }
        let empty = case("empty_line_padding")
            .input
            .full_line
            .as_ref()
            .ok_or("padding full_line missing")?;
        if empty.records.iter().any(|r| !r.moves.is_empty())
            || !empty.query_prefix.is_empty()
            || !empty.query_proposal.is_empty()
            || !empty.query_counter.is_empty()
        {
            return Err("empty-line case does not exercise fully masked temporal padding".into());
        }
    }
    Ok(())
}

fn compare(actual: &[f32], expected: &[f32], atol: f64, rtol: f64) -> Result<f64, Box<dyn Error>> {
    if actual.len() != expected.len() {
        return Err("reference tensor shape differs".into());
    }
    let mut maximum: f64 = 0.;
    for (a, e) in actual.iter().zip(expected) {
        let delta = (f64::from(*a) - f64::from(*e)).abs();
        if !a.is_finite() || !e.is_finite() || delta > atol + rtol * f64::from(*e).abs() {
            return Err(
                format!("PALS numeric mismatch: actual={a}, expected={e}, abs={delta}").into(),
            );
        }
        maximum = maximum.max(delta);
    }
    Ok(maximum)
}
fn verify(
    raw: &PalsRawOutput,
    case: &Case,
    config: &PalsModelConfig,
) -> Result<serde_json::Value, Box<dyn Error>> {
    let actual = raw.decode(&case.input, config)?;
    let expected = case.expected.decode(&case.input, config)?;
    let candidate = compare(
        &raw.candidate_logits,
        &case.expected.candidate_logits,
        1e-4,
        1e-3,
    )?;
    let wdl_logits = compare(&raw.wdl_logits, &case.expected.wdl_logits, 1e-4, 1e-3)?;
    let latent = compare(
        &raw.private_latent,
        &case.expected.private_latent,
        1e-4,
        1e-3,
    )?;
    let policy = compare(
        &actual.candidate_policy,
        &expected.candidate_policy,
        1e-4,
        0.,
    )?;
    let wdl = compare(&actual.wdl, &expected.wdl, 1e-4, 0.)?;
    let divergence = match (&raw.divergence_logits, &case.expected.divergence_logits) {
        (Some(a), Some(e)) => Some(compare(a, e, 1e-4, 1e-3)?),
        (None, None) => None,
        _ => return Err("private critic head differs".into()),
    };
    Ok(
        json!({"case":case.name,"candidate_logit_max_abs":candidate,"wdl_logit_max_abs":wdl_logits,
        "latent_max_abs":latent,"policy_max_abs":policy,"wdl_max_abs":wdl,"divergence_logit_max_abs":divergence}),
    )
}
fn verify_public(
    backend: &PalsOnnxBackend,
    case: &Case,
) -> Result<serde_json::Value, Box<dyn Error>> {
    match backend.public_memory_witness()? {
        Some(actual) => {
            if actual.tokens != case.public_memory.tokens
                || actual.memory_mask != case.public_memory.memory_mask
            {
                return Err("independent public-memory shape/mask mismatch".into());
            }
            let key = compare(
                &actual.memory_key,
                &case.public_memory.memory_key,
                1e-4,
                1e-3,
            )?;
            let value = compare(
                &actual.memory_value,
                &case.public_memory.memory_value,
                1e-4,
                1e-3,
            )?;
            Ok(
                json!({"status":"passed","key_max_abs":key,"value_max_abs":value,"mask":"exact","source":"completed_host_cache"}),
            )
        }
        None if backend.config().device_public_memory => Ok(
            json!({"status":"not_run","cause":"device_KV_host_witness_unavailable","key_max_abs":null,"value_max_abs":null}),
        ),
        None => Err("completed public-memory cache witness missing".into()),
    }
}
fn outside_git(path: &Path) -> Result<(), Box<dyn Error>> {
    if !path.is_absolute() {
        return Err("external artifact path must be absolute".into());
    }
    for ancestor in path.canonicalize()?.ancestors() {
        if ancestor.join(".git").try_exists()? {
            return Err("generated artifacts must be outside Git".into());
        }
    }
    Ok(())
}
fn hex_digest(value: &[u8; 32]) -> String {
    value.iter().map(|v| format!("{v:02x}")).collect()
}
fn stats_json(stats: &PalsBackendStats) -> serde_json::Value {
    json!({"admitted_role_requests":stats.admitted_role_requests,
        "public_cache_hits":stats.public_cache_hits,"public_cache_misses":stats.public_cache_misses,
        "public_nn_runs_attempted":stats.public_nn_runs_attempted,"public_nn_runs_completed":stats.public_nn_runs_completed,
        "public_nn_runs_failed_known":stats.public_nn_runs_failed_known,
        "role_nn_runs_attempted":stats.role_nn_runs_attempted,"role_nn_runs_completed":stats.role_nn_runs_completed,
        "role_nn_runs_failed_known":stats.role_nn_runs_failed_known,"completed_nn_inputs":stats.completed_nn_inputs,
        "validated_public_outputs":stats.validated_public_outputs,"validated_role_outputs":stats.validated_role_outputs,
        "new_game_resets":stats.new_game_resets,"live_public_cache_entries":stats.live_public_cache_entries})
}

const HOST_PAGE_CHECK_FLAG: &str = "--check-host-record-pages";
const HOST_PAGE_CHECK_SECONDS: u64 = 300;

fn page_snapshot_json(snapshot: &HostRecordPageSnapshot) -> serde_json::Value {
    let stats = &snapshot.stats;
    json!({"policy":{"public_graph_sha256":hex_digest(&snapshot.policy.public_graph_sha256),
        "projection_semantics":snapshot.policy.projection_semantics,
        "max_page_entries":snapshot.policy.max_page_entries,
        "max_page_bytes":snapshot.policy.max_page_bytes,
        "max_transient_bytes":snapshot.policy.max_transient_bytes},
        "bank":{"entries":snapshot.bank.entries,"reserved_bytes":snapshot.bank.reserved_bytes,
            "entry_backing_bytes":snapshot.bank.entry_backing_bytes,
            "max_entries":snapshot.bank.max_entries,"max_value_bytes":snapshot.bank.max_bytes,
            "pinned_entries":snapshot.bank.pinned_entries,"pinned_bytes":snapshot.bank.pinned_bytes},
        "actual_page_counters":{"view_hits":stats.view_hits,"view_misses":stats.view_misses,
            "board_hits":stats.board_hits,"board_misses":stats.board_misses,
            "record_hits":stats.record_hits,"record_misses":stats.record_misses,
            "public_calls_attempted":stats.public_calls_attempted,
            "public_calls_completed":stats.public_calls_completed,
            "submitted_record_tokens":stats.submitted_record_tokens,
            "encoded_record_tokens":stats.encoded_record_tokens,
            "contextual_board_encodes_completed":stats.contextual_board_encodes_completed,
            "joins_completed":stats.joins_completed,"joined_bytes":stats.joined_bytes,
            "evicted_pages":stats.evicted_pages,
            "max_reserved_host_transient_bytes":stats.max_reserved_host_transient_bytes},
        "active_pin_count":snapshot.active_pin_count,"retained_join_bytes":snapshot.retained_join_bytes,
        "active_subset_bytes":snapshot.active_subset_bytes,
        "active_join_backing_bytes":snapshot.active_join_backing_bytes,
        "active_full_input_bytes":snapshot.active_full_input_bytes,
        "transient_reservation_bytes":snapshot.transient_reservation_bytes,
        "quarantined":snapshot.quarantined,
        "scope":"declared_and_actual_host_backing_not_native_allocator_or_vram_peak"})
}

fn completed_page_snapshot(
    backend: &PalsOnnxBackend,
    joined: bool,
) -> Result<HostRecordPageSnapshot, Box<dyn Error>> {
    let snapshot = backend
        .host_record_page_snapshot()
        .ok_or("host page mode is absent")?;
    if snapshot.quarantined
        || snapshot.active_pin_count != 0
        || snapshot.active_subset_bytes != 0
        || snapshot.active_join_backing_bytes != 0
        || snapshot.active_full_input_bytes != 0
        || snapshot.transient_reservation_bytes != 0
        || snapshot.bank.pinned_entries != 0
        || snapshot.bank.pinned_bytes != 0
        || snapshot.bank.entries > snapshot.bank.max_entries
        || snapshot.bank.reserved_bytes > snapshot.bank.max_bytes
        || snapshot
            .bank
            .reserved_bytes
            .checked_add(snapshot.bank.entry_backing_bytes)
            .is_none_or(|bytes| bytes > snapshot.policy.max_page_bytes)
        || snapshot.stats.max_reserved_host_transient_bytes > snapshot.policy.max_transient_bytes
        || (snapshot.retained_join_bytes != 0) != joined
    {
        return Err("host record page completion, pin release or backing budget differs".into());
    }
    let native = backend.snapshot_stats()?;
    if native.public_cache_hits != 0
        || native.public_cache_misses != 0
        || native.live_public_cache_entries != 0
    {
        return Err(
            "host record page counters leaked into the legacy whole-input cache counters".into(),
        );
    }
    Ok(snapshot)
}

fn numeric_window(start: Instant) -> Result<(), Box<dyn Error>> {
    if start.elapsed() >= Duration::from_secs(HOST_PAGE_CHECK_SECONDS) {
        return Err("finite CPU host-page correctness window exhausted before next Run".into());
    }
    Ok(())
}

fn reseal_numeric_input(
    input: &mut PalsModelInput,
    config: &PalsModelConfig,
) -> Result<(), Box<dyn Error>> {
    // Derived inputs are model-tensor cases, not fabricated Rules positions,
    // CPU observations or training targets. Retain the reference history/epoch.
    input.required_critical_records = input
        .records
        .iter()
        .filter(|record| record.critical)
        .map(|record| record.record_id)
        .collect();
    // These controlled page derivations isolate public feature reuse. Full-line
    // reference mutations above exercise actual nonempty complete sequences.
    if let Some(lines) = &mut input.full_line {
        lines.records = input
            .records
            .iter()
            .map(|_| PalsRecordLine {
                moves: vec![],
                parent: None,
                supersedes: None,
                parent_required: false,
                supersedes_required: false,
            })
            .collect();
    }
    input.validate(config)?;
    let prepared = input.prepare_tensors(config)?;
    if prepared.input_key != input.canonical_input_key(config)?
        || prepared.public_memory_key != input.public_memory_key(config)?
    {
        return Err("derived full canonical/encoding identity did not reseal".into());
    }
    Ok(())
}

fn derived_record_cases(
    fixtures: &Fixtures,
) -> Result<Vec<(&'static str, PalsModelInput)>, Box<dyn Error>> {
    let case = |name: &str| {
        fixtures
            .cases
            .iter()
            .find(|case| case.name == name)
            .expect("six-case coverage verified")
            .input
            .clone()
    };
    let base = case("proposer_order");
    let critic = case("critic_divergence");
    let seed = base.records[0].clone();
    let mut empty = base.clone();
    empty.records.clear();
    let mut zero = empty.clone();
    let mut zero_record = seed.clone();
    zero_record.record_id = 1;
    zero_record.revision = 1;
    zero_record.critical = false;
    zero_record.features = [0.; 16];
    zero.records.push(zero_record);
    let mut one = base.clone();
    one.records.truncate(1);
    one.records[0].critical = false;
    one.records[0].features[15] = 0.03125;
    let mut two = base.clone();
    two.records.truncate(2);
    for record in &mut two.records {
        record.critical = false;
    }
    two.records[0] = one.records[0].clone();
    two.records[1].features[15] = 0.09375;
    let mut corrected = two.clone();
    corrected.records[0].features[3] += 0.125;
    corrected.records[0].features[15] = 0.1875;
    corrected.records[0].revision = corrected.records[0]
        .revision
        .checked_add(1)
        .ok_or("derived record revision overflow")?;
    let mut reordered = corrected.clone();
    reordered.records.reverse();
    let mut critical = reordered.clone();
    critical.records[0].critical = true;
    let mut shifted = critical.clone();
    for (index, record) in shifted.records.iter_mut().enumerate() {
        record.record_id = index as u64 + 40;
    }
    let mut many = base.clone();
    many.records = (0..128)
        .map(|index| {
            let mut record = seed.clone();
            record.record_id = index + 1;
            record.revision = index + 1;
            record.critical = false;
            record.features[15] = index as f32 / 128.;
            record
        })
        .collect();
    let mut evicted = many.clone();
    evicted.records.remove(0);
    for (index, record) in evicted.records.iter_mut().enumerate() {
        record.record_id = index as u64 + 1;
    }
    let mut appended = evicted.clone();
    let mut added = seed;
    added.record_id = 128;
    added.revision = 200;
    added.critical = false;
    added.features[15] = 2.;
    appended.records.push(added);
    let mut critic_view = appended.clone();
    critic_view.role = PalsRole::Critic;
    critic_view.divergence_features = critic.divergence_features;
    critic_view.query = critic.query;
    let mut cases = vec![
        ("derived_empty_false_padding", empty.clone()),
        ("derived_actual_zero_feature_record", zero.clone()),
        ("derived_back_to_empty_padding", empty),
        ("derived_one_record", one),
        ("derived_two_records_append", two),
        ("derived_record_correction", corrected),
        ("derived_record_reorder", reordered),
        ("derived_critical_change", critical),
        ("derived_id_shift", shifted),
        ("derived_128_records", many),
        ("derived_eviction_and_remaining_id_shift", evicted),
        ("derived_append_after_eviction_128", appended.clone()),
        ("derived_role_order_p_to_c", critic_view.clone()),
        ("derived_role_order_c_to_p", appended),
        ("derived_role_order_p_to_c_again", critic_view),
    ];
    for (_, input) in &mut cases {
        reseal_numeric_input(input, &fixtures.config)?;
    }
    if cases.len() > 20 {
        return Err("derived correctness case cap exceeded".into());
    }
    Ok(cases)
}

fn delta(after: u64, before: u64) -> Result<u64, Box<dyn Error>> {
    after
        .checked_sub(before)
        .ok_or_else(|| "native/page counter moved backwards".into())
}

fn verify_page_run_accounting(
    before: &PalsBackendStats,
    after: &PalsBackendStats,
    page_before: &HostRecordPageSnapshot,
    page_after: &HostRecordPageSnapshot,
) -> Result<serde_json::Value, Box<dyn Error>> {
    let public = delta(
        after.public_nn_runs_completed,
        before.public_nn_runs_completed,
    )?;
    let misses = delta(
        page_after.stats.record_misses,
        page_before.stats.record_misses,
    )?;
    let encoded = delta(
        page_after.stats.encoded_record_tokens,
        page_before.stats.encoded_record_tokens,
    )?;
    if public > 1
        || delta(after.admitted_role_requests, before.admitted_role_requests)? != 1
        || delta(after.role_nn_runs_attempted, before.role_nn_runs_attempted)? != 1
        || delta(after.role_nn_runs_completed, before.role_nn_runs_completed)? != 1
        || delta(
            after.public_nn_runs_attempted,
            before.public_nn_runs_attempted,
        )? != public
        || delta(after.completed_nn_inputs, before.completed_nn_inputs)? != 1 + public
        || delta(
            page_after.stats.public_calls_attempted,
            page_before.stats.public_calls_attempted,
        )? != public
        || delta(
            page_after.stats.public_calls_completed,
            page_before.stats.public_calls_completed,
        )? != public
        || delta(
            page_after.stats.contextual_board_encodes_completed,
            page_before.stats.contextual_board_encodes_completed,
        )? != public
        || delta(
            page_after.stats.submitted_record_tokens,
            page_before.stats.submitted_record_tokens,
        )? != encoded
        || encoded != if public == 0 { 0 } else { misses.max(1) }
        || delta(
            page_after.stats.joins_completed,
            page_before.stats.joins_completed,
        )? != 1
    {
        return Err("derived host page run changed physical NN/token/join accounting".into());
    }
    Ok(
        json!({"public_calls_completed":public,"physical_nn_inputs_completed":1+public,
        "unique_record_misses":misses,"encoded_record_slots":encoded,
        "board_recomputed_by_existing_combined_graph":public!=0}),
    )
}

fn check_host_record_pages(
    fixtures: &Fixtures,
    runtime: OrtRuntime,
    export: &Path,
    export_sha: &str,
) -> Result<serde_json::Value, Box<dyn Error>> {
    let start = Instant::now();
    let config = PalsOnnxConfig::cpu();
    let mut whole = PalsOnnxBackend::load(export, export_sha, runtime.clone(), config)?;
    let mut pages = PalsOnnxBackend::load(export, export_sha, runtime, config)?;
    let graph_sha = pages
        .residency()
        .graphs
        .iter()
        .find(|graph| graph.role == "public")
        .ok_or("registered public graph identity missing")?
        .sha256;
    let policy =
        HostRecordPagePolicy::for_profile_registered_graph(fixtures.config.profile, graph_sha);
    pages.enable_host_record_pages(policy)?;
    let checkpoint = asset::parse_sha256(&fixtures.checkpoint_sha256)?;
    if !matches!(whole.config().provider, Provider::Cpu)
        || !matches!(pages.config().provider, Provider::Cpu)
        || whole.model_epoch() != checkpoint
        || pages.model_epoch() != checkpoint
        || whole.is_trained() != fixtures.trained
        || pages.is_trained() != fixtures.trained
    {
        return Err(
            "host record page correctness mode changed CPU/checkpoint/provenance conditions".into(),
        );
    }
    let mut anchors = Vec::new();
    for case in &fixtures.cases {
        numeric_window(start)?;
        let raw = pages.run(&case.input)?;
        let mut row = verify(&raw, case, &fixtures.config)?;
        row["public_memory"] = verify_public(&pages, case)?;
        row["page_ownership"] = page_snapshot_json(&completed_page_snapshot(&pages, true)?);
        anchors.push(row);
    }
    // These six anchor references remain independent PyTorch evidence. Derived
    // cases below use freshly run whole-input ORT as the explicit comparator.
    pages.clear_public_memory()?;
    let anchor_clear = completed_page_snapshot(&pages, false)?;
    if anchor_clear.bank.entries != 0 || anchor_clear.bank.reserved_bytes != 0 {
        return Err("page anchor reset did not retire its representations".into());
    }
    let derived = derived_record_cases(fixtures)?;
    let mut rows = Vec::new();
    let mut first_case = None;
    for (name, input) in &derived {
        numeric_window(start)?;
        whole.clear_public_memory()?;
        let whole_before = whole.snapshot_stats()?;
        let expected = whole.run(input)?;
        let witness = whole
            .public_memory_witness()?
            .ok_or("whole comparator witness missing")?;
        let case = Case {
            name: (*name).into(),
            input: input.clone(),
            expected,
            public_memory: witness,
        };
        let page_before = completed_page_snapshot(&pages, !rows.is_empty())?;
        let native_before = pages.snapshot_stats()?;
        numeric_window(start)?;
        let actual = pages.run(input)?;
        let native_after = pages.snapshot_stats()?;
        let page_after = completed_page_snapshot(&pages, true)?;
        let mut row = verify(&actual, &case, &fixtures.config)?;
        row["public_memory"] = verify_public(&pages, &case)?;
        row["input_key"] = json!(hex_digest(&input.canonical_input_key(&fixtures.config)?));
        row["full_public_key"] = json!(hex_digest(&input.public_memory_key(&fixtures.config)?));
        row["records"] = json!(input.records.len());
        row["role"] = json!(input.role);
        row["first_run_accounting"] =
            verify_page_run_accounting(&native_before, &native_after, &page_before, &page_after)?;
        let public_delta = delta(
            native_after.public_nn_runs_completed,
            native_before.public_nn_runs_completed,
        )?;
        let record_misses = delta(
            page_after.stats.record_misses,
            page_before.stats.record_misses,
        )?;
        // These controlled mutations must exercise actual page reuse/subset
        // execution, rather than merely compare another full re-encode.
        match *name {
            "derived_actual_zero_feature_record"
            | "derived_back_to_empty_padding"
            | "derived_record_reorder"
            | "derived_critical_change"
            | "derived_id_shift"
            | "derived_eviction_and_remaining_id_shift"
            | "derived_role_order_p_to_c"
            | "derived_role_order_c_to_p"
            | "derived_role_order_p_to_c_again"
                if public_delta != 0 || record_misses != 0 =>
            {
                return Err(
                    "feature-identical derived view failed to reuse existing public pages".into(),
                );
            }
            "derived_one_record"
            | "derived_two_records_append"
            | "derived_record_correction"
            | "derived_append_after_eviction_128"
                if public_delta != 1 || record_misses != 1 =>
            {
                return Err(
                    "single-record derived delta did not execute exactly one missing feature slot"
                        .into(),
                );
            }
            _ => {}
        }
        row["page_after_first"] = page_snapshot_json(&page_after);
        // Each repeat is a fresh private role invocation over exact reused K/V.
        numeric_window(start)?;
        let repeated = pages.run(input)?;
        let repeat_native = pages.snapshot_stats()?;
        let repeat_pages = completed_page_snapshot(&pages, true)?;
        verify(&repeated, &case, &fixtures.config)?;
        let repeat_counts =
            verify_page_run_accounting(&native_after, &repeat_native, &page_after, &repeat_pages)?;
        if repeat_native.public_nn_runs_completed != native_after.public_nn_runs_completed
            || repeat_pages.stats.record_misses != page_after.stats.record_misses
            || repeat_pages.stats.board_misses != page_after.stats.board_misses
            || repeat_pages.stats.view_hits != page_after.stats.view_hits + 1
        {
            return Err("repeated host page view unexpectedly recomputed a representation".into());
        }
        row["repeated_page_accounting"] = repeat_counts;
        let whole_after = whole.snapshot_stats()?;
        if whole_after.public_nn_runs_completed != whole_before.public_nn_runs_completed + 1
            || whole_after.completed_nn_inputs != whole_before.completed_nn_inputs + 2
        {
            return Err("fresh whole comparator did not physically execute both graphs".into());
        }
        numeric_window(start)?;
        verify(&whole.run(input)?, &case, &fixtures.config)?;
        let whole_repeat = whole.snapshot_stats()?;
        let whole_ownership = whole.host_public_page_snapshot();
        if whole_repeat.public_nn_runs_completed != whole_after.public_nn_runs_completed
            || whole_repeat.completed_nn_inputs != whole_after.completed_nn_inputs + 1
            || whole_repeat.public_cache_hits != whole_after.public_cache_hits + 1
            || whole_ownership.pinned_entries != 0
            || whole_ownership.pinned_bytes != 0
        {
            return Err("repeated whole comparator NN/cache/pin accounting differs".into());
        }
        row["whole_after_repeat"] = stats_json(&whole_repeat);
        if first_case.is_none() {
            first_case = Some(case);
        }
        rows.push(row);
    }
    let page_before_clear = completed_page_snapshot(&pages, true)?;
    let native_before_clear = pages.snapshot_stats()?;
    pages.clear_public_memory()?;
    let page_after_clear = completed_page_snapshot(&pages, false)?;
    let native_after_clear = pages.snapshot_stats()?;
    if page_after_clear.bank.entries != 0
        || page_after_clear.bank.reserved_bytes != 0
        || pages.public_memory_witness()?.is_some()
        || native_after_clear.completed_nn_inputs != native_before_clear.completed_nn_inputs
        || native_after_clear.new_game_resets != native_before_clear.new_game_resets + 1
    {
        return Err("host page clear did not retire representations without NN work".into());
    }
    numeric_window(start)?;
    let first = first_case.ok_or("derived correctness cases are empty")?;
    let fresh_after_clear = pages.run(&first.input)?;
    let after_fresh_native = pages.snapshot_stats()?;
    let after_fresh_pages = completed_page_snapshot(&pages, true)?;
    verify(&fresh_after_clear, &first, &fixtures.config)?;
    verify_public(&pages, &first)?;
    let fresh_counts = verify_page_run_accounting(
        &native_after_clear,
        &after_fresh_native,
        &page_after_clear,
        &after_fresh_pages,
    )?;
    if after_fresh_native.public_nn_runs_completed
        != native_after_clear.public_nn_runs_completed + 1
    {
        return Err("game clear reused an old generation's host page".into());
    }
    whole.verify_runtime()?;
    pages.verify_runtime()?;
    let whole_final = whole.snapshot_stats()?;
    drop(whole);
    drop(pages);
    Ok(
        json!({"schema":"rovezero.pals-host-record-pages-correctness.v1","status":"passed",
        "scope":"model_tensor_numeric_only_no_rules_or_training_certification_no_performance_claim",
        "provider":"CPUExecutionProvider","precision":"fp32","tf32":false,
        "max_window_seconds":HOST_PAGE_CHECK_SECONDS,"native_deadline_scope":"before_each_synchronous_Run_not_forced_cancellation",
        "encoding_schema":fixtures.config.profile.encoding_schema(),"manifest_sha256":export_sha,"checkpoint_sha256":fixtures.checkpoint_sha256,
        "independent_reference_anchor":{"source":"original_registered_six_case_pytorch_fp32_tf32_off",
            "whole_results":"top_level_cases_unchanged","page_cases":anchors},
        "derived_whole_page_equivalence":{"source":"fresh_whole_input_ORT_same_pinned_export_and_runtime",
            "input_source":"deterministic_derivatives_of_registered_fixtures","cases":rows},
        "tolerance":{"raw_atol":1e-4,"raw_rtol":1e-3,"policy_wdl_max_abs":1e-4,"mask":"exact"},
        "anchor_clear":page_snapshot_json(&anchor_clear),
        "final_clear":{"before":page_snapshot_json(&page_before_clear),"after":page_snapshot_json(&page_after_clear),
            "neural_runs":"zero","fresh_after_clear":fresh_counts},
        "final_page_stats":page_snapshot_json(&after_fresh_pages),"final_native_stats":stats_json(&after_fresh_native),
        "final_whole_stats":stats_json(&whole_final),
        "native_owner_drop":"returned_after_completed_CPU_runs","process_exit":"requires_external_process_exit_and_cleanup",
        "native_allocator_peak":"unknown","vram_peak":"unknown"}),
    )
}

fn mapping_json(phase: &str, observed: &NativeMappingObservation) -> serde_json::Value {
    json!({"phase":phase,"scope":"proc_self_maps_at_audit_boundary",
        "loading_profile":observed.profile.identifier(),
        "declared_nvidia_files":observed.declared_nvidia_files,
        "required_nvidia_files":observed.required_nvidia_files,
        "mapped_nvidia_files":observed.mapped_nvidia_files,
        "mapped_nvidia_count":observed.mapped_nvidia_files.len(),
        "deferred_nvidia_not_mapped":observed.deferred_nvidia_not_mapped,
        "deferred_absence_meaning":"not_mapped_at_observation_not_unused",
        "mapped_ort_files":observed.mapped_ort_files,
        "kernel_placement":"separate_cuda_placement_witness",
        "vram_residency":"unknown","normal_process_exit":"requires_external_supervisor"})
}

const COST_COMMON_SIX_FLAG: &str = "--cost-common-six";
const COST_COMMON_SIX_CONV_DEFAULT_FLAG: &str = "--cost-common-six-conv-default";
const COST_SECONDS: u64 = 45;
const COST_REPORT_BYTES: usize = 1024 * 1024;
const COST_WARMUP: usize = 3;
const COST_SAMPLES: usize = 5;
const COST_CASES: [&str; 6] = [
    "empty_context",
    "critic_divergence",
    "proposer_order",
    "proposer_order_reversed",
    "same_board_other_history",
    "all_promotions",
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CostProfile {
    CommonSix,
    CommonSixConvDefault,
}

impl CostProfile {
    fn take(args: &mut Vec<String>) -> Result<Option<Self>, &'static str> {
        let selected = match args.first().map(String::as_str) {
            Some(COST_COMMON_SIX_FLAG) => Some(Self::CommonSix),
            Some(COST_COMMON_SIX_CONV_DEFAULT_FLAG) => Some(Self::CommonSixConvDefault),
            _ => None,
        };
        if selected.is_some() {
            args.remove(0);
            if args.iter().any(|arg| arg.starts_with("--")) {
                return Err("common-six cost is independent;record-page and other optional modes cannot be combined");
            }
        }
        Ok(selected)
    }

    fn flag(self) -> &'static str {
        match self {
            Self::CommonSix => COST_COMMON_SIX_FLAG,
            Self::CommonSixConvDefault => COST_COMMON_SIX_CONV_DEFAULT_FLAG,
        }
    }

    fn validate_provider(self, provider: CheckerProvider) -> Result<(), &'static str> {
        match (self, provider) {
            (Self::CommonSix, CheckerProvider::CudaControlConvDefault) => Err(
                "cuDNN DEFAULT cost requires the separate --cost-common-six-conv-default opt-in",
            ),
            (Self::CommonSixConvDefault, CheckerProvider::CudaControlConvDefault) => Ok(()),
            (Self::CommonSixConvDefault, _) => Err(
                "--cost-common-six-conv-default requires exact cuda-control-conv-default with pinned bundle and inventory",
            ),
            (Self::CommonSix, _) => Ok(()),
        }
    }

    fn validate_tf32_override(self, value: Option<&std::ffi::OsStr>) -> Result<(), &'static str> {
        // The execution owner supplies the environment before native loading.
        // This checks an explicit request; it does not observe a cuDNN plan.
        if self == Self::CommonSixConvDefault && value != Some(std::ffi::OsStr::new("0")) {
            return Err("DEFAULT cost requires NVIDIA_TF32_OVERRIDE=0 before native loading");
        }
        Ok(())
    }
}

fn cost_ns(duration: Duration) -> Result<u64, Box<dyn Error>> {
    Ok(u64::try_from(duration.as_nanos())?)
}

fn cost_failure(error: &(dyn Error + 'static)) -> serde_json::Value {
    if let Some(error) = error.downcast_ref::<BackendError>() {
        json!({"kind":format!("{:?}",error.kind),"stage":format!("{:?}",error.stage),
            "detail":error.detail,"bounded_cause":error.cause.map(|cause|format!("{cause:?}"))})
    } else {
        json!({"kind":"example_cost_failure","detail":error.to_string()})
    }
}

fn cost_percentiles(samples: &[u64]) -> serde_json::Value {
    let mut ordered = samples.to_vec();
    ordered.sort_unstable();
    let rank = |percent: usize| (ordered.len() * percent).div_ceil(100) - 1;
    json!({"raw_ns":samples,"count":samples.len(),"p50_ns":ordered[rank(50)],
        "p95_ns":ordered[rank(95)],"quantile_method":"nearest_rank;five_samples_p95_is_max"})
}

fn cost_host_memory() -> serde_json::Value {
    #[cfg(target_os = "linux")]
    {
        let Ok(status) = std::fs::read_to_string("/proc/self/status") else {
            return json!({"status":"unknown","scope":"own_process_proc_status_unavailable"});
        };
        let field = |name: &str| {
            status.lines().find_map(|line| {
                let value = line.strip_prefix(name)?.trim();
                let mut parts = value.split_whitespace();
                let kib: u64 = parts.next()?.parse().ok()?;
                (parts.next() == Some("kB"))
                    .then(|| kib.checked_mul(1024))
                    .flatten()
            })
        };
        json!({"status":"observed_fields_only","rss_bytes":field("VmRSS:"),
            "process_hwm_bytes":field("VmHWM:"),
            "scope":"whole_process_lifetime_including_fixture_runtime_load_and_verification;not_per_call_allocator_peak"})
    }
    #[cfg(not(target_os = "linux"))]
    json!({"status":"unknown","scope":"own_process_proc_status_unsupported"})
}

fn cost_delta(
    before: &PalsBackendStats,
    after: &PalsBackendStats,
    cold: bool,
) -> Result<serde_json::Value, Box<dyn Error>> {
    let delta = |a: u64, b: u64| -> Result<u64, Box<dyn Error>> {
        b.checked_sub(a)
            .ok_or_else(|| "cost NN counter regressed".into())
    };
    let public = u64::from(cold);
    let admitted = delta(before.admitted_role_requests, after.admitted_role_requests)?;
    let public_attempted = delta(
        before.public_nn_runs_attempted,
        after.public_nn_runs_attempted,
    )?;
    let public_completed = delta(
        before.public_nn_runs_completed,
        after.public_nn_runs_completed,
    )?;
    let role_attempted = delta(before.role_nn_runs_attempted, after.role_nn_runs_attempted)?;
    let role_completed = delta(before.role_nn_runs_completed, after.role_nn_runs_completed)?;
    let physical = delta(before.completed_nn_inputs, after.completed_nn_inputs)?;
    let hits = delta(before.public_cache_hits, after.public_cache_hits)?;
    let misses = delta(before.public_cache_misses, after.public_cache_misses)?;
    let role_validated = delta(before.validated_role_outputs, after.validated_role_outputs)?;
    if admitted != 1
        || public_attempted != public
        || public_completed != public
        || role_attempted != 1
        || role_completed != 1
        || physical != public + 1
        || hits != u64::from(!cold)
        || misses != public
        || role_validated != 1
        || after.public_nn_runs_failed_known != before.public_nn_runs_failed_known
        || after.role_nn_runs_failed_known != before.role_nn_runs_failed_known
    {
        return Err("cost invocation physical NN work or cache lane differs".into());
    }
    Ok(
        json!({"admitted_role_requests":admitted,"public_attempted":public_attempted,
        "public_completed":public_completed,"role_attempted":role_attempted,
        "role_completed":role_completed,"completed_nn_inputs":physical,
        "public_cache_hits":hits,"public_cache_misses":misses,"validated_role_outputs":role_validated,
        "physical_completion":"synchronous_run_fence_and_idle_snapshot_confirmed"}),
    )
}

fn cost_common_six(
    args: &[String],
    fixtures: &Fixtures,
    fixture_digest: &[u8; 32],
    backend: &mut PalsOnnxBackend,
    started: Instant,
    backend_load_ns: u64,
    cost_profile: CostProfile,
) -> Result<(), Box<dyn Error>> {
    let mut receipt = json!({"schema":"rz-pals-common-six-api-cost/1","status":"pending",
        "model_profile":fixtures.config.profile,"checkpoint_sha256":fixtures.checkpoint_sha256,
        "manifest_sha256":args[1],"fixture_sha256":hex_digest(fixture_digest),
        "original_fixture_case_count":fixtures.cases.len(),"selected_cases":COST_CASES,
        "provider_argument":args[6],"precision":"fp32","tf32":false,"batch":1,
        "actual_ORT_intra_threads":backend.config().intra_threads,
        "thread_scope":"explicit_common_six_model_cost_profile;arena_default_intra2_is_a_separate_engine_profile",
        "graph_optimization":backend.graph_optimization(),"device_public_memory":backend.config().device_public_memory,
        "runtime_sha256":hex_digest(&backend.runtime_binary_digest()),
        "runtime_bundle_sha256":backend.runtime_bundle_digest().as_ref().map(hex_digest)});
    // DEFAULT is a separately registered cost request. The original wire and
    // common-six timing/verification lanes remain unchanged for legacy modes.
    if cost_profile == CostProfile::CommonSixConvDefault {
        cost_profile.validate_tf32_override(std::env::var_os("NVIDIA_TF32_OVERRIDE").as_deref())?;
        let evidence = conv_default_execution_evidence(backend)?
            .ok_or("DEFAULT cost loaded backend lacks the requested execution descriptor")?;
        receipt["cost_opt_in"] = json!(cost_profile.flag());
        receipt["cudnn_conv_search_diagnostic"] = evidence;
        receipt["tf32"] = json!("unknown");
        receipt["tf32_requested"] = json!(false);
        receipt["nvidia_tf32_override_environment_request"] = json!("0");
        receipt["experiment_scope"] = json!("explicit_DEFAULT_common_six_API_cost;colab_observed_resource_evidence_is_external;not_a_production_default_or_speedup_claim");
    }
    let serde_json::Value::Object(timing_fields) = json!({
        "warmup_calls_per_case_lane":COST_WARMUP,"measured_calls_per_case_lane":COST_SAMPLES,
        "soft_work_seconds_max":COST_SECONDS,"report_bytes_max":COST_REPORT_BYTES,
        "external_supervisor_required":"physical Run may outlive soft deadline;root bounds60seconds plus cleanup30seconds",
        "startup_before_ready_wall_ns":cost_ns(started.elapsed())?,
        "backend_load_api_ns":backend_load_ns,"host_memory_after_load":cost_host_memory(),
        "call_time_scope":"PalsOnnxBackend::run entry_to_return including preparation/cache/public_and_role_Run/transfer/fence/output_validation;load/reset/verify/stats/JSON excluded",
        "cache_hit_preparation":"inherits same input public cache from preceding final cold call;no extra priming Run",
        "cold_scope":"public_cache_cleared_before_each_call;model_loaded_and_three_preparation_calls_before_five_samples",
        "planned_role_api_calls":{"ready":2,"preparation":36,"measured":60},
        "expected_physical_nn_inputs_per_call":{"cold":2,"cache_hit":1}})
    else {
        return Err("cost timing metadata is not an object".into());
    };
    let serde_json::Value::Object(scope_fields) = json!({
        "engine_consumed_nn_inputs":"unknown;direct_model_cost_mode_has_no_engine_consumer",
        "timer_overhead":"raw;recorded_not_subtracted","cuda_kernel_event_elapsed":"unknown",
        "per_call_host_allocator_peak":"unknown","vram_peak":"unknown","device_sharing":"unknown",
        "physical_shutdown":"requires_external_exit_and_cleanup;not_claimed_by_direct_owner_snapshot",
        "five_percent_model_gate":false,"strength_or_full_engine_effect_claim":false,
        "ready":[],"rows":[]})
    else {
        return Err("cost scope metadata is not an object".into());
    };
    let receipt_fields = receipt
        .as_object_mut()
        .ok_or("cost receipt is not an object")?;
    receipt_fields.extend(timing_fields);
    receipt_fields.extend(scope_fields);
    let mut overhead = Vec::with_capacity(COST_SAMPLES);
    for _ in 0..COST_SAMPLES {
        let timer = Instant::now();
        overhead.push(cost_ns(timer.elapsed())?);
    }
    receipt["timer_pair_overhead"] = cost_percentiles(&overhead);
    let deadline = || -> Result<(), Box<dyn Error>> {
        if started.elapsed() >= Duration::from_secs(COST_SECONDS) {
            return Err("finite45second cost window exhausted;no next physical Run".into());
        }
        Ok(())
    };
    let result = (|| -> Result<(), Box<dyn Error>> {
        let selected: Vec<&Case> = COST_CASES
            .iter()
            .map(|name| {
                fixtures
                    .cases
                    .iter()
                    .find(|case| case.name == *name)
                    .ok_or_else(|| -> Box<dyn Error> { "common cost case missing".into() })
            })
            .collect::<Result<_, _>>()?;
        // Ready uses original P/C fixture inputs and is outside every cost lane.
        let ready_started = Instant::now();
        for role in [PalsRole::Proposer, PalsRole::Critic] {
            let case = selected
                .iter()
                .copied()
                .find(|case| case.input.role == role)
                .ok_or("common cost fixtures lack P/C readiness coverage")?;
            deadline()?;
            backend.clear_public_memory()?;
            let before = backend.snapshot_stats()?;
            let timer = Instant::now();
            let run = backend.run(&case.input);
            let elapsed = cost_ns(timer.elapsed())?;
            let after = backend.snapshot_stats();
            let mut ready = json!({"case":case.name,"role":role,"api_elapsed_ns":elapsed,
                "backend_before":stats_json(&before),"physical_completion":"unconfirmed"});
            let ready_result = (|| -> Result<(), Box<dyn Error>> {
                let raw = run?;
                let after = after?;
                ready["backend_after"] = stats_json(&after);
                ready["NN_delta"] = cost_delta(&before, &after, true)?;
                ready["physical_completion"] = json!("confirmed");
                ready["numeric"] = verify(&raw, case, &fixtures.config)?;
                Ok(())
            })();
            if let Err(error) = &ready_result {
                ready["failure"] = cost_failure(error.as_ref());
            }
            receipt["ready"]
                .as_array_mut()
                .ok_or("cost ready ledger invalid")?
                .push(ready);
            ready_result?;
            deadline()?;
        }
        if backend.cuda_control_inventory_digest().is_some() {
            receipt["startup_only_cuda_placement"] = json!(backend.verify_cuda_placement()?);
            receipt["profile_scope"] = json!("explicit_control_ready_P_C_only;profiling_stopped_before_cost_lanes;not_common_six_cost_profile");
        } else {
            receipt["profile_scope"] = json!("no_cost_profile_collected");
        }
        receipt["ready_elapsed_ns"] = json!(cost_ns(ready_started.elapsed())?);
        receipt["backend_stats_after_ready"] = stats_json(&backend.snapshot_stats()?);
        for case in selected {
            let original_input_sha = hex_digest(&asset::sha256(&serde_json::to_vec(&case.input)?));
            let input_key = hex_digest(&case.input.canonical_input_key(&fixtures.config)?);
            let public_key = hex_digest(&case.input.public_memory_key(&fixtures.config)?);
            for cold in [true, false] {
                let lane = if cold { "cold" } else { "cache_hit" };
                let mut row = json!({"case":case.name,"role":case.input.role,"lane":lane,
                    "original_input_sha256":original_input_sha,"canonical_input_key":input_key,
                    "public_memory_key":public_key,"records":case.input.records.len(),
                    "physical_completion":"pending","warmup":[],"measured":[]});
                let mut measured = Vec::with_capacity(COST_SAMPLES);
                let mut complete = 0;
                let lane_result = (|| -> Result<(), Box<dyn Error>> {
                    for ordinal in 0..COST_WARMUP + COST_SAMPLES {
                        deadline()?;
                        if cold {
                            // Reset and its cost are outside the exact Run timer.
                            backend.clear_public_memory()?;
                        }
                        let before = backend.snapshot_stats()?;
                        let timer = Instant::now();
                        let run = backend.run(&case.input);
                        let elapsed = cost_ns(timer.elapsed())?;
                        let after = backend.snapshot_stats();
                        let mut sample = json!({"ordinal":ordinal,"elapsed_ns":elapsed,
                            "backend_before":stats_json(&before),"physical_completion":"unconfirmed"});
                        match &after {
                            Ok(after) => {
                                sample["backend_after"] = stats_json(after);
                                sample["returned_physical_nn_inputs_delta"] = json!(after
                                    .completed_nn_inputs
                                    .checked_sub(before.completed_nn_inputs));
                                sample["owner_idle_after_call"] = json!(true);
                            }
                            Err(error) => sample["after_stats_failure"] = cost_failure(error),
                        }
                        if let Err(error) = &run {
                            sample["run_failure"] = cost_failure(error);
                        }
                        let sample_result = (|| -> Result<(), Box<dyn Error>> {
                            let raw = run?;
                            let after = after?;
                            sample["backend_after"] = stats_json(&after);
                            sample["NN_delta"] = cost_delta(&before, &after, cold)?;
                            sample["physical_completion"] = json!("confirmed");
                            sample["numeric"] = verify(&raw, case, &fixtures.config)?;
                            sample["raw_output_sha256"] =
                                json!(hex_digest(&asset::sha256(&serde_json::to_vec(&raw)?)));
                            complete += 1;
                            deadline()?;
                            Ok(())
                        })();
                        if let Err(error) = &sample_result {
                            sample["failure"] = cost_failure(error.as_ref());
                        }
                        let field = if ordinal < COST_WARMUP {
                            "warmup"
                        } else {
                            "measured"
                        };
                        row[field]
                            .as_array_mut()
                            .ok_or("cost sample ledger invalid")?
                            .push(sample);
                        sample_result?;
                        if ordinal >= COST_WARMUP {
                            measured.push(elapsed);
                        }
                    }
                    row["sample_summary"] = cost_percentiles(&measured);
                    row["physical_completion"] = json!("all8calls_confirmed");
                    Ok(())
                })();
                row["confirmed_calls"] = json!(complete);
                if let Err(error) = &lane_result {
                    row["failure"] = cost_failure(error.as_ref());
                }
                receipt["rows"]
                    .as_array_mut()
                    .ok_or("cost rows ledger invalid")?
                    .push(row);
                lane_result?;
            }
        }
        backend.verify_runtime()?;
        receipt["backend_stats_final"] = stats_json(&backend.snapshot_stats()?);
        receipt["drain_scope"] = json!("direct_Run_owner_idle_at_final_snapshot;no_async_worker;external_process_cleanup_not_observed");
        deadline()?;
        Ok(())
    })();
    receipt["status"] = json!(if result.is_ok() {
        "passed"
    } else {
        "failed_no_further_case_dispatched"
    });
    if let Err(error) = &result {
        receipt["failure"] = cost_failure(error.as_ref());
    }
    receipt["host_memory_final"] = cost_host_memory();
    receipt["program_wall_ns_before_report_write"] = json!(cost_ns(started.elapsed())?);
    let bytes = serde_json::to_vec_pretty(&receipt)?;
    if bytes
        .len()
        .checked_add(1)
        .is_none_or(|len| len > COST_REPORT_BYTES)
    {
        return Err("cost report exceeded1MiB;registered inputs retained".into());
    }
    let mut output = std::fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&args[5])?;
    output.write_all(&bytes)?;
    output.write_all(b"\n")?;
    output.sync_all()?;
    result
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CheckerProvider {
    Cpu,
    Cuda,
    CudaDevice,
    CudaControl,
    CudaControlShim,
    CudaControlConvDefault,
}
impl CheckerProvider {
    fn parse(provider: &str, argument_count: usize) -> Result<Self, &'static str> {
        match (provider, argument_count) {
            ("cpu", 8) => Ok(Self::Cpu),
            ("cuda", 9) => Ok(Self::Cuda),
            ("cuda-device", 9) => Ok(Self::CudaDevice),
            ("cuda-control", 11) => Ok(Self::CudaControl),
            ("cuda-control-shim", 11) => Ok(Self::CudaControlShim),
            ("cuda-control-conv-default", 11) => Ok(Self::CudaControlConvDefault),
            _ => Err("CUDA requires its explicit pinned bundle; CPU excludes it"),
        }
    }
    fn is_cuda(self) -> bool {
        self != Self::Cpu
    }
    fn explicit_control(self) -> bool {
        matches!(
            self,
            Self::CudaControl | Self::CudaControlShim | Self::CudaControlConvDefault
        )
    }
    fn loading_profile(self) -> Option<NativeLoadingProfile> {
        match self {
            Self::Cpu => None,
            Self::CudaControlShim => Some(NativeLoadingProfile::CuDnnShimLazyV1),
            _ => Some(NativeLoadingProfile::EagerCuda12Cudnn9V1),
        }
    }
    fn config(self, cost_profile: Option<CostProfile>) -> Result<PalsOnnxConfig, Box<dyn Error>> {
        if let Some(cost_profile) = cost_profile {
            cost_profile.validate_provider(self)?;
        }
        let mut config = PalsOnnxConfig::cpu();
        if cost_profile.is_some() {
            config.intra_threads = 1;
        }
        if self.is_cuda() {
            config.provider = Provider::Cuda {
                device_id: 0,
                arena_bytes: 2 * 1024 * 1024 * 1024,
            };
            config.device_public_memory = self == Self::CudaDevice;
        }
        if self == Self::CudaControlConvDefault {
            config.cudnn_conv_search = PalsCuDnnConvSearch::ExperimentalDefault;
        }
        config.validate_execution_selection(self.loading_profile(), self.explicit_control())?;
        Ok(config)
    }
}

/// Configuration provenance only. This separate digest never identifies an
/// observed cuDNN plan, numerical-note, retry, or a successful process exit.
fn conv_default_execution_evidence(
    backend: &PalsOnnxBackend,
) -> Result<Option<serde_json::Value>, Box<dyn Error>> {
    let config = backend.config();
    if config.cudnn_conv_search != PalsCuDnnConvSearch::ExperimentalDefault {
        return Ok(None);
    }
    config.validate_execution_selection(
        backend.native_loading_profile(),
        backend.cuda_control_inventory_digest().is_some(),
    )?;
    let Provider::Cuda {
        device_id,
        arena_bytes,
    } = config.provider
    else {
        return Err("cuDNN DEFAULT execution descriptor has no CUDA provider".into());
    };
    let bundle = backend
        .runtime_bundle_digest()
        .ok_or("CUDA bundle identity absent")?;
    // Record only the explicitly recognized process request. This environment
    // override does not observe the plan, numerical note, or cuDNN retry.
    let tf32_environment_override = match std::env::var_os("NVIDIA_TF32_OVERRIDE") {
        None => "absent",
        Some(value) if value == "0" => "0",
        Some(value) if value == "1" => "1",
        Some(_) => "unrecognized",
    };
    let descriptor = json!({
        "schema":"rz-pals-cuda-control-conv-default-execution/1",
        "scope":"loaded_backend_execution_request_not_observed_cudnn_plan",
        "manifest_sha256":hex_digest(&backend.manifest_digest()),
        "model_epoch":hex_digest(&backend.model_epoch()),
        "runtime_sha256":hex_digest(&backend.runtime_binary_digest()),
        "runtime_bundle_sha256":hex_digest(&bundle),
        "control_inventory_sha256":backend.cuda_control_inventory_digest().as_ref().map(hex_digest),
        "provider":"CUDAExecutionProvider","device_id":device_id,"session_arena_bytes":arena_bytes,
        "intra_threads":config.intra_threads,"inter_threads":1,"parallel_execution":false,
        "cache_public_memory":config.cache_public_memory,"public_memory":"host",
        "native_loading_profile":"eager-cuda12-cudnn9-v1","graph_optimization":backend.graph_optimization(),
        "cudnn_conv_algo_search_requested":"DEFAULT","tf32_requested":false,
        "nvidia_tf32_override_environment_request":tf32_environment_override,"cuda_graph_requested":false
    });
    let serialized = serde_json::to_vec(&descriptor)?;
    Ok(Some(json!({
        "descriptor":descriptor,"descriptor_sha256":hex_digest(&asset::sha256(&serialized)),
        "descriptor_encoding":"serde_json_utf8_fixed_descriptor_object",
        "actual_cudnn_plan":"unknown","numerical_note":"unknown",
        "tf32_actual":"unknown","tf32_retry":"unknown",
        "experiment_scope":"single_requested_option_numeric_and_lifetime_diagnostic",
        "native_termination_acceptance":"requires_external_process_exit_zero_and_cleanup",
        "production_default_changed":false,"warm_success_claim":false,"pilot_success_claim":false,
        "failure_cause":"not_proven"
    })))
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut args: Vec<_> = std::env::args().skip(1).collect();
    let cost_profile = CostProfile::take(&mut args)?;
    let cost_mode = cost_profile.is_some();
    let startup_trace = NativeStartupTrace::new(cost_mode)?;
    let cost_started = cost_mode.then(Instant::now);
    #[cfg(feature = "experimental-io-binding")]
    let resident_check = resident_cuda::Selection::take(&mut args)?;
    #[cfg(not(feature = "experimental-io-binding"))]
    if args.iter().any(|arg| arg == "--check-cuda-record-pages") {
        return Err(
            "resident CUDA correctness requires the explicit experimental-io-binding build feature"
                .into(),
        );
    }
    let check_record_pages = args.last().is_some_and(|arg| arg == HOST_PAGE_CHECK_FLAG);
    if check_record_pages {
        args.pop();
        if args.len() != 8 || args[6] != "cpu" {
            return Err("--check-host-record-pages is an explicit CPU-only optional flag after the existing eight CPU arguments".into());
        }
    }
    if args.len() == 3 && args[0] == "--check-cuda-control-policy" {
        // Pure bounded source-policy parsing: no runtime load, session, Run or
        // provider claim. The external owner preserves stdout as evidence.
        PalsCudaControlPolicy::from_inventory(Path::new(&args[1]), &args[2])?;
        println!("Pinned static PALS CUDA control v2 accepted; explicit graph optimization=disable; native placement and GPU execution not observed");
        return Ok(());
    }
    if !matches!(args.len(), 8 | 9 | 11) {
        return Err("usage: pals_model_check [--cost-common-six|--cost-common-six-conv-default] EXPORT.json EXPORT_SHA256 ORT_LIBRARY ORT_SHA256 FIXTURES.json REPORT.json cpu|cuda|cuda-device|cuda-control|cuda-control-shim|cuda-control-conv-default CACHE_ROOT [CUDA_BUNDLE.json] [INVENTORY_V2.json INVENTORY_SHA256]; DEFAULT cost requires its separate leading flag, exact cuda-control-conv-default and NVIDIA_TF32_OVERRIDE=0; CPU-only optional trailing --check-host-record-pages; Linux cuda-control-shim optional trailing --check-cuda-record-pages MANIFEST SHA256 GRAPH SHA256 RESOURCES.json".into());
    }
    let selected_provider = CheckerProvider::parse(&args[6], args.len())?;
    let is_cuda = selected_provider.is_cuda();
    // Candidate combinations are refused before fixture/model reads, cache
    // mutation, or native loading. Only the candidate's inventory moves earlier.
    let mut config = selected_provider.config(cost_profile)?;
    if let Some(cost_profile) = cost_profile {
        cost_profile.validate_tf32_override(std::env::var_os("NVIDIA_TF32_OVERRIDE").as_deref())?;
    }
    let preflight_control = if selected_provider == CheckerProvider::CudaControlConvDefault {
        Some(PalsCudaControlPolicy::from_inventory(
            Path::new(&args[9]),
            &args[10],
        )?)
    } else {
        None
    };
    let report = Path::new(&args[5]);
    outside_git(report.parent().ok_or("report has no existing parent")?)?;
    if report.exists() {
        return Err("refusing to overwrite registered PALS acceptance report".into());
    }
    let fixture_path = Path::new(&args[4]);
    if !fixture_path.is_absolute() {
        return Err("fixture must be absolute".into());
    }
    startup_trace.stage("fixtures_entered");
    let fixture_bytes = asset::read_bounded(fixture_path, 16 * 1024 * 1024)?;
    let fixture_digest = asset::sha256(&fixture_bytes);
    let fixtures: Fixtures = serde_json::from_slice(&fixture_bytes)?;
    validate_fixture_coverage(&fixtures)?;
    startup_trace.stage("fixtures_returned");
    startup_trace.stage("runtime_pin_entered");
    let cache = RuntimeCache::open(Path::new(&args[7]))?;
    let pin = if is_cuda {
        let spec_bytes = asset::read_bounded(Path::new(&args[8]), 64 * 1024)?;
        let spec = CudaRuntimeBundleSpec::from_json(std::str::from_utf8(&spec_bytes)?)?;
        let core = spec
            .files
            .iter()
            .find(|f| f.role == RuntimeBundleFileRole::Core)
            .ok_or("bundle core absent")?;
        let path = Path::new(&args[2]);
        if path.file_name().and_then(|n| n.to_str()) != Some(core.filename.as_str())
            || args[3] != core.sha256
        {
            return Err("ORT declaration differs from explicit CUDA bundle".into());
        }
        cache.cuda_bundle(path.parent().ok_or("runtime core has no parent")?, &spec)?
    } else {
        cache.library(Path::new(&args[2]), &args[3])?
    };
    startup_trace.stage("runtime_pin_returned");
    // Shim lazy is an explicitly registered diagnostic variable. Other modes,
    // including the existing cuda-control mode, retain eager16 bootstrap.
    startup_trace.stage("runtime_load_entered");
    let runtime = (if args[6] == "cuda-control-shim" {
        OrtRuntime::load_with_cuda_loading_profile(&pin, NativeLoadingProfile::CuDnnShimLazyV1)
    } else {
        OrtRuntime::load(&pin)
    })
    .map_err(|mut failure| {
        if is_cuda {
            failure.detail = "PALS CUDA runtime bootstrap failed before session creation";
        }
        failure
    })?;
    startup_trace.stage("runtime_load_returned");
    let runtime_digest = runtime.binary_digest();
    let runtime_bundle_digest = runtime.bundle_digest();
    let loading_profile = runtime.native_loading_profile();
    let loading_profile_digest = runtime.native_loading_profile_digest();
    let mut loading_observations = Vec::new();
    if is_cuda {
        startup_trace.stage("runtime_origin_audit_entered");
        loading_observations.push(mapping_json(
            "dependencies_before_sessions",
            &runtime.cuda_mapping_observation(false)?,
        ));
        startup_trace.stage("runtime_origin_audit_returned");
    }
    let control_policy = if preflight_control.is_some() {
        preflight_control
    } else if selected_provider.explicit_control() {
        Some(PalsCudaControlPolicy::from_inventory(
            Path::new(&args[9]),
            &args[10],
        )?)
    } else {
        None
    };
    let load = |runtime, config, suffix: &str| {
        startup_trace.stage("session_load_entered");
        let result = if let Some(policy) = &control_policy {
            PalsOnnxBackend::load_with_cuda_control_policy(
                Path::new(&args[0]),
                &args[1],
                runtime,
                config,
                policy.clone(),
                &report.with_extension(format!("cuda-control-{suffix}")),
            )
        } else {
            PalsOnnxBackend::load(Path::new(&args[0]), &args[1], runtime, config)
        };
        startup_trace.stage(if result.is_ok() {
            "session_load_returned_ok"
        } else {
            "session_load_returned_error"
        });
        result
    };
    let cost_load_started = cost_mode.then(Instant::now);
    let mut backend = load(runtime.clone(), config, "primary")?;
    let conv_default_execution = conv_default_execution_evidence(&backend)?;
    let cost_backend_load_ns = cost_load_started
        .map(|timer| cost_ns(timer.elapsed()))
        .transpose()?;
    let graph_optimization = backend.graph_optimization();
    let epoch = backend.model_epoch();
    if epoch != asset::parse_sha256(&fixtures.checkpoint_sha256)?
        || backend.is_trained() != fixtures.trained
        || backend.model_config() != &fixtures.config
    {
        return Err("reference weight epoch differs".into());
    }
    for case in &fixtures.cases {
        if case.name.is_empty() || case.name.len() > 128 || case.input.model_epoch != epoch {
            return Err("fixture identity differs".into());
        }
        case.input.validate(&fixtures.config)?;
    }
    if let Some(started) = cost_started {
        return cost_common_six(
            &args,
            &fixtures,
            &fixture_digest,
            &mut backend,
            started,
            cost_backend_load_ns.ok_or("cost load timer missing")?,
            cost_profile.ok_or("cost selection missing")?,
        );
    }
    // CUDA tests are bounded by the shell/process owner in addition to these
    // logical deadlines. This check never claims a deadline stopped native Run.
    let began = Instant::now();
    let mut reports = Vec::new();
    let mut raw_outputs = Vec::new();
    let mut proposer_seen = false;
    let mut critic_seen = false;
    let mut placement_witness = None;
    for (case_index, case) in fixtures.cases.iter().enumerate() {
        if began.elapsed() >= Duration::from_secs(120) {
            return Err("finite numeric window exhausted before next physical Run".into());
        }
        if case_index == 0 {
            startup_trace.stage("first_native_run_entered");
        }
        let raw = backend.run(&case.input)?;
        if case_index == 0 {
            startup_trace.stage("first_native_run_returned");
        }
        let pages = backend.host_public_page_snapshot();
        if pages.max_entries != 1
            || pages.pinned_entries != 0
            || pages.pinned_bytes != 0
            || pages.reserved_bytes > pages.max_bytes
            || pages.entries != usize::from(!config.device_public_memory)
        {
            return Err(
                "completed role retained an active host public-page pin or exceeded its bank"
                    .into(),
            );
        }
        let mut report = verify(&raw, case, &fixtures.config)?;
        report["public_memory"] = verify_public(&backend, case)?;
        reports.push(report);
        let repeated = backend.run(&case.input)?;
        if repeated != raw {
            return Err("fresh private-role repeat changed exact PALS result".into());
        }
        proposer_seen |= case.input.role == PalsRole::Proposer;
        critic_seen |= case.input.role == PalsRole::Critic;
        if control_policy.is_some() && proposer_seen && critic_seen && placement_witness.is_none() {
            placement_witness = Some(backend.verify_cuda_placement()?);
        }
        raw_outputs.push(raw);
    }
    startup_trace.stage("numeric_cases_returned");
    let mut validator = fixtures.cases[0].input.clone();
    validator.role = PalsRole::Validator;
    validator.divergence_features.clear();
    if backend.run(&validator).is_ok() {
        return Err("P/C product admitted private V role".into());
    }
    let public_encodes = backend.public_encodes;
    let public_cache_hits = backend.public_cache_hits;
    backend.verify_runtime()?;
    if is_cuda {
        loading_observations.push(mapping_json(
            "primary_numeric_physical_completion",
            &runtime.cuda_mapping_observation(true)?,
        ));
    }
    let trained = backend.is_trained();
    let residency = backend.residency().clone();
    let pages_before_reset = backend.host_public_page_snapshot();
    let encodes_before_new_game = backend.public_encodes;
    backend.clear_public_memory()?;
    let pages_after_reset = backend.host_public_page_snapshot();
    if pages_after_reset.entries != 0
        || pages_after_reset.reserved_bytes != 0
        || pages_after_reset.pinned_entries != 0
        || pages_after_reset.pinned_bytes != 0
    {
        return Err("NewGame did not retire the completed host public-memory bank".into());
    }
    if backend.public_encodes != encodes_before_new_game {
        return Err("new game cache reset executed a neural graph".into());
    }
    if backend.run(&fixtures.cases[0].input)? != raw_outputs[0]
        || backend.public_encodes != encodes_before_new_game + 1
    {
        return Err("new game did not force a fresh equivalent public-memory encode".into());
    }
    // Separate sessions verify cache eviction/fresh encode equivalence. No
    // measured performance improvement is inferred from these correctness calls.
    config.cache_public_memory = false;
    let mut fresh = load(runtime.clone(), config, "fresh")?;
    let fresh_conv_default_execution = conv_default_execution_evidence(&fresh)?;
    let mut fresh_proposer_seen = false;
    let mut fresh_critic_seen = false;
    let mut fresh_placement_witness = None;
    for (case, expected) in fixtures.cases.iter().zip(&raw_outputs) {
        if fresh.run(&case.input)? != *expected {
            return Err("public memory cache changed exact PALS output".into());
        }
        let pages = fresh.host_public_page_snapshot();
        if pages.entries != 0 || pages.reserved_bytes != 0 || pages.pinned_entries != 0 {
            return Err("disabled cache retained a completed host public page".into());
        }
        fresh_proposer_seen |= case.input.role == PalsRole::Proposer;
        fresh_critic_seen |= case.input.role == PalsRole::Critic;
        if control_policy.is_some()
            && fresh_proposer_seen
            && fresh_critic_seen
            && fresh_placement_witness.is_none()
        {
            fresh_placement_witness = Some(fresh.verify_cuda_placement()?);
        }
    }
    fresh.verify_runtime()?;
    drop(fresh);
    let before_worker_stats = backend.snapshot_stats()?;
    let mut worker = backend.controlled_worker()?;
    let mut reset = worker.submit(PalsNativeCommand::NewGame)?;
    let reset_deadline = Instant::now() + Duration::from_secs(30);
    loop {
        match reset.poll() {
            PhysicalPoll::Ready(Ok(PalsNativeResult::NewGame)) => break,
            PhysicalPoll::Ready(Ok(PalsNativeResult::ResetTo { .. })) => {
                return Err("legacy new game reset unexpectedly returned a target reset ACK".into())
            }
            PhysicalPoll::Ready(Err(error)) => return Err(error.into()),
            PhysicalPoll::Ready(Ok(PalsNativeResult::Evaluation(_))) => {
                return Err("cache reset unexpectedly returned a neural evaluation".into());
            }
            PhysicalPoll::Ready(Ok(PalsNativeResult::EvaluationWithEvidence { .. })) => {
                return Err(BackendError::new(
                    FailureKind::BackendFailure,
                    FailureStage::Backend,
                    "cache reset unexpectedly returned an evidence evaluation",
                )
                .into());
            }
            PhysicalPoll::Ready(Ok(PalsNativeResult::Stats(_))) => {
                return Err("cache reset unexpectedly returned native statistics".into())
            }
            #[cfg(feature = "experimental-io-binding")]
            PhysicalPoll::Ready(Ok(PalsNativeResult::CudaRecordPagesObserved(_))) => {
                return Err("cache reset unexpectedly returned resident CUDA metadata".into())
            }
            PhysicalPoll::Ready(Ok(PalsNativeResult::RuntimeVerified)) => {
                return Err("cache reset unexpectedly returned runtime verification".into())
            }
            PhysicalPoll::Ready(Ok(PalsNativeResult::CudaPlacementVerified(_))) => {
                return Err("cache reset unexpectedly returned CUDA placement verification".into())
            }
            PhysicalPoll::Ready(Ok(PalsNativeResult::RuntimeMappingsObserved(_))) => {
                return Err("cache reset unexpectedly returned runtime mapping observation".into())
            }
            PhysicalPoll::Quarantined => {
                return Err("new game cache reset remains quarantined".into())
            }
            PhysicalPoll::Consumed => return Err("new game cache reset completed twice".into()),
            PhysicalPoll::Pending if Instant::now() < reset_deadline => {
                std::thread::sleep(Duration::from_millis(1));
            }
            PhysicalPoll::Pending => {
                return Err("new game cache reset has not physically completed".into())
            }
        }
    }
    let mut lease = worker.submit(PalsNativeCommand::Evaluate(fixtures.cases[0].input.clone()))?;
    let physical_deadline = Instant::now() + Duration::from_secs(30);
    loop {
        match lease.poll() {
            PhysicalPoll::Ready(result) => {
                match result? {
                    PalsNativeResult::Evaluation(raw) => {
                        verify(&raw, &fixtures.cases[0], &fixtures.config)?;
                    }
                    PalsNativeResult::EvaluationWithEvidence { .. } => {
                        return Err(BackendError::new(
                            FailureKind::BackendFailure,
                            FailureStage::Backend,
                            "legacy evaluation unexpectedly returned an evidence evaluation",
                        )
                        .into());
                    }
                    PalsNativeResult::NewGame => {
                        return Err("evaluation unexpectedly returned a cache reset".into())
                    }
                    PalsNativeResult::ResetTo { .. } => {
                        return Err("evaluation unexpectedly returned a target reset ACK".into())
                    }
                    PalsNativeResult::Stats(_) => {
                        return Err("evaluation unexpectedly returned native statistics".into())
                    }
                    #[cfg(feature = "experimental-io-binding")]
                    PalsNativeResult::CudaRecordPagesObserved(_) => {
                        return Err("evaluation unexpectedly returned resident CUDA metadata".into())
                    }
                    PalsNativeResult::RuntimeVerified => {
                        return Err("evaluation unexpectedly returned runtime verification".into())
                    }
                    PalsNativeResult::CudaPlacementVerified(_) => {
                        return Err(
                            "evaluation unexpectedly returned CUDA placement verification".into(),
                        )
                    }
                    PalsNativeResult::RuntimeMappingsObserved(_) => {
                        return Err(
                            "evaluation unexpectedly returned runtime mapping observation".into(),
                        )
                    }
                }
                break;
            }
            PhysicalPoll::Quarantined => {
                return Err("physical completion is unknown; quarantined job retained".into())
            }
            PhysicalPoll::Consumed => return Err("physical lease completed twice".into()),
            PhysicalPoll::Pending => {}
        }
        if Instant::now() >= physical_deadline {
            return Err("physical lease still live at finite check deadline".into());
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    let mut verification = worker.submit(PalsNativeCommand::VerifyRuntime)?;
    let verification_deadline = Instant::now() + Duration::from_secs(30);
    loop {
        match verification.poll() {
            PhysicalPoll::Ready(Ok(PalsNativeResult::RuntimeVerified)) => break,
            PhysicalPoll::Ready(Err(error)) => return Err(error.into()),
            PhysicalPoll::Ready(Ok(_)) => {
                return Err("runtime verification returned another command response".into())
            }
            PhysicalPoll::Quarantined => {
                return Err(
                    "runtime verification cannot confirm unknown physical completion".into(),
                )
            }
            PhysicalPoll::Consumed => return Err("runtime verification completed twice".into()),
            PhysicalPoll::Pending if Instant::now() < verification_deadline => {
                std::thread::sleep(Duration::from_millis(1))
            }
            PhysicalPoll::Pending => {
                return Err("runtime verification has no physical ACK before deadline".into())
            }
        }
    }
    if let Some(expected) = &placement_witness {
        let mut verification = worker.submit(PalsNativeCommand::VerifyCudaPlacement)?;
        let deadline = Instant::now() + Duration::from_secs(30);
        loop {
            match verification.poll() {
                PhysicalPoll::Ready(Ok(PalsNativeResult::CudaPlacementVerified(witness)))
                    if *witness == *expected =>
                {
                    break
                }
                PhysicalPoll::Ready(Err(error)) => return Err(error.into()),
                PhysicalPoll::Ready(Ok(_)) => {
                    return Err(
                        "CUDA placement control returned a different immutable witness or response"
                            .into(),
                    )
                }
                PhysicalPoll::Quarantined => {
                    return Err("CUDA placement control has unknown physical completion".into())
                }
                PhysicalPoll::Consumed => {
                    return Err("CUDA placement control completed twice".into())
                }
                PhysicalPoll::Pending if Instant::now() < deadline => {
                    std::thread::sleep(Duration::from_millis(1))
                }
                PhysicalPoll::Pending => {
                    return Err("CUDA placement control has no finite physical ACK".into())
                }
            }
        }
    }
    let mut snapshot = worker.submit(PalsNativeCommand::SnapshotStats)?;
    let stats_deadline = Instant::now() + Duration::from_secs(30);
    let stats = loop {
        match snapshot.poll() {
            PhysicalPoll::Ready(Ok(PalsNativeResult::Stats(stats))) => break stats,
            PhysicalPoll::Ready(Err(error)) => return Err(error.into()),
            PhysicalPoll::Ready(Ok(_)) => {
                return Err("native statistics returned another command response".into())
            }
            PhysicalPoll::Quarantined => {
                return Err("native statistics cannot confirm unknown physical completion".into())
            }
            PhysicalPoll::Consumed => return Err("native statistics completed twice".into()),
            PhysicalPoll::Pending if Instant::now() < stats_deadline => {
                std::thread::sleep(Duration::from_millis(1))
            }
            PhysicalPoll::Pending => {
                return Err("native statistics have no physical ACK before deadline".into())
            }
        }
    };
    stats.validate()?;
    if stats.admitted_role_requests != before_worker_stats.admitted_role_requests + 1
        || stats.role_nn_runs_attempted != before_worker_stats.role_nn_runs_attempted + 1
        || stats.role_nn_runs_completed != before_worker_stats.role_nn_runs_completed + 1
        || stats.public_nn_runs_attempted != before_worker_stats.public_nn_runs_attempted + 1
        || stats.public_nn_runs_completed != before_worker_stats.public_nn_runs_completed + 1
        || stats.completed_nn_inputs != before_worker_stats.completed_nn_inputs + 2
        || stats.new_game_resets != before_worker_stats.new_game_resets + 1
        || stats.live_public_cache_entries != 1
    {
        return Err("NewGame/SnapshotStats control changed NN accounting or failed to force fresh public encode".into());
    }
    let shutdown_deadline = Instant::now() + Duration::from_secs(30);
    loop {
        match worker.try_shutdown() {
            Poll::Ready(result) => {
                result?;
                break;
            }
            Poll::Pending if Instant::now() < shutdown_deadline => {
                std::thread::sleep(Duration::from_millis(1))
            }
            Poll::Pending => return Err("physical owner has not completed shutdown".into()),
        }
    }
    if is_cuda {
        loading_observations.push(mapping_json(
            "final_worker_joined",
            &runtime.cuda_mapping_observation(true)?,
        ));
    }
    let host_page_evidence = json!({"scope":"direct_owner_after_numeric_cases_before_worker",
        "page_kind":"whole_input","frozen_numeric_epoch":0,
        "weight_epoch_identity":"full_checkpoint_digest_in_exact_content_key",
        "bank_max_entries":pages_before_reset.max_entries,"bank_max_bytes":pages_before_reset.max_bytes,
        "before_new_game":{"entries":pages_before_reset.entries,"reserved_bytes":pages_before_reset.reserved_bytes,
            "pinned_entries":pages_before_reset.pinned_entries,"pinned_bytes":pages_before_reset.pinned_bytes},
        "after_new_game":{"entries":pages_after_reset.entries,"reserved_bytes":pages_after_reset.reserved_bytes,
            "pinned_entries":pages_after_reset.pinned_entries,"pinned_bytes":pages_after_reset.pinned_bytes},
        "cache_off_completed_page_retirement":"passed","runtime_allocator_peak":"unknown","vram_peak":"unknown"});
    let cuda_evidence = json!({"cuda_mode":args[6],"graph_optimization":graph_optimization,
        "cuda_control_inventory_sha256":control_policy.as_ref().map(|_| args[10].as_str()),
        "cuda_placement_witness":placement_witness,"fresh_cuda_placement_witness":fresh_placement_witness});
    let mut receipt = json!({"schema": fixtures.config.profile.model_semantics(),"status":"passed","trained":trained,"rules_certified":false,
        "model_epoch":hex_digest(&epoch),"manifest_sha256":args[1],"fixture_sha256":hex_digest(&fixture_digest),
        "runtime_sha256":hex_digest(&runtime_digest),"runtime":"1.22.0","rust_ort":"2.0.0-rc.10",
        "provider":if is_cuda {"CUDAExecutionProvider"} else {"CPUExecutionProvider"},"precision":"fp32","tf32":false,
        "physical_completion":"confirmed","physical_shutdown":"confirmed","public_encodes":public_encodes,
        "public_cache_hits":public_cache_hits,"fresh_cache_equivalence":"passed","private_role_repeat":"passed",
        "public_counter_scope":"six_reference_cases_before_new_game_checks",
        "v_rejection":"passed","new_game_cache_reset":"passed","new_game_physical_ack":"confirmed",
        "stats_physical_ack":"confirmed","control_neural_runs":"zero","backend_stats":stats_json(&stats),
        "runtime_verify_physical_ack":"confirmed",
        "backend_stats_scope":"backend_lifetime_through_final_snapshot_ack",
        "device_public_memory":config.device_public_memory,"session_residency":residency,"vram_peak":"unknown","cases":reports});
    let fields = receipt
        .as_object_mut()
        .ok_or("acceptance receipt is not an object")?;
    if let Some(primary) = conv_default_execution {
        // The old baseline wire retains its historical field. This candidate
        // has no observation of actual TF32 use, even after numerical success.
        fields.insert("tf32".into(), json!("unknown"));
        fields.insert("tf32_requested".into(), json!(false));
        fields.insert(
            "cudnn_conv_search_diagnostic".into(),
            json!({
                "primary_backend_execution":primary,
                "fresh_backend_execution":fresh_conv_default_execution
                    .ok_or("fresh DEFAULT execution descriptor absent")?
            }),
        );
    }
    if !fixtures.config.profile.is_legacy() {
        fields.insert("model_profile".into(), json!(fixtures.config.profile));
        fields.insert(
            "encoding_schema".into(),
            json!(fixtures.config.profile.encoding_schema()),
        );
        fields.insert(
            "public_counter_scope".into(),
            json!("registered_profile_reference_cases_before_new_game_checks"),
        );
    }
    fields.insert("host_public_page_ownership".into(), host_page_evidence);
    fields.insert(
        "native_loading".into(),
        json!({
        "schema":"rovezero.native-loading-observation.v1",
        "profile":loading_profile.map(NativeLoadingProfile::identifier),
        "profile_sha256":loading_profile_digest.as_ref().map(hex_digest),
        "experimental_candidate":loading_profile == Some(NativeLoadingProfile::CuDnnShimLazyV1),
        "runtime_bundle_sha256":runtime_bundle_digest.as_ref().map(hex_digest),
        "declared_bundle_file_count":if is_cuda {19} else {1},
        "mapping_observations":loading_observations,
        "native_termination_acceptance":"requires_external_process_exit_zero_and_cleanup",
        "preload_cause":"not_proven"}),
    );
    fields.extend(
        cuda_evidence
            .as_object()
            .ok_or("CUDA evidence is not an object")?
            .clone(),
    );
    if check_record_pages {
        fields.insert(
            "host_record_page_correctness".into(),
            check_host_record_pages(&fixtures, runtime.clone(), Path::new(&args[0]), &args[1])?,
        );
    }
    #[cfg(feature = "experimental-io-binding")]
    if let Some(selection) = resident_check {
        fields.insert(
            "resident_cuda_record_correctness".into(),
            selection.check(resident_cuda::Context {
                fixtures: &fixtures,
                runtime: runtime.clone(),
                export: Path::new(&args[0]),
                export_sha: &args[1],
                policy: control_policy
                    .as_ref()
                    .ok_or("resident correctness has no explicit CUDA policy")?,
                config,
                report,
            })?,
        );
    }
    startup_trace.stage("report_write_entered");
    let mut output = std::fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(report)?;
    serde_json::to_writer_pretty(&mut output, &receipt)?;
    output.write_all(b"\n")?;
    output.sync_all()?;
    startup_trace.stage("report_write_returned");
    println!("PALS P/C numeric/physical checks passed; trained={trained}; GPU peak unknown");
    Ok(())
}

#[cfg(test)]
mod conv_selection_tests {
    use super::*;

    #[test]
    fn conv_default_selection_is_exact_control_host_and_eager() {
        let selected = CheckerProvider::parse("cuda-control-conv-default", 11).unwrap();
        assert!(selected.explicit_control());
        assert_eq!(
            selected.loading_profile(),
            Some(NativeLoadingProfile::EagerCuda12Cudnn9V1)
        );
        let config = selected.config(None).unwrap();
        assert_eq!(
            config.cudnn_conv_search,
            PalsCuDnnConvSearch::ExperimentalDefault
        );
        assert!(!config.device_public_memory);
        for arity in [0, 8, 9, 10, 12, 17] {
            assert!(CheckerProvider::parse("cuda-control-conv-default", arity).is_err());
        }
        assert!(selected.config(Some(CostProfile::CommonSix)).is_err());
        assert!(CheckerProvider::parse("cuda-control-conv-default-shim", 11).is_err());
    }

    #[test]
    fn original_provider_modes_keep_implicit_conv_and_loading_selection() {
        for (name, count) in [
            ("cpu", 8),
            ("cuda", 9),
            ("cuda-control", 11),
            ("cuda-control-shim", 11),
        ] {
            let selected = CheckerProvider::parse(name, count).unwrap();
            assert_eq!(
                selected.config(None).unwrap().cudnn_conv_search,
                PalsCuDnnConvSearch::LegacyImplicitExhaustive
            );
        }
        assert_eq!(
            CheckerProvider::parse("cuda-control-shim", 11)
                .unwrap()
                .loading_profile(),
            Some(NativeLoadingProfile::CuDnnShimLazyV1)
        );
        assert_eq!(
            CheckerProvider::parse("cpu", 8).unwrap().loading_profile(),
            None
        );
        assert_eq!(
            CheckerProvider::parse("cuda-device", 9).unwrap(),
            CheckerProvider::CudaDevice
        );
        assert!(!CheckerProvider::CudaDevice.explicit_control());
    }

    #[test]
    fn default_cost_needs_its_exact_leading_flag_and_provider() {
        let mut args = vec![
            COST_COMMON_SIX_CONV_DEFAULT_FLAG.into(),
            "registered-export".into(),
        ];
        assert_eq!(
            CostProfile::take(&mut args).unwrap(),
            Some(CostProfile::CommonSixConvDefault)
        );
        assert_eq!(args, ["registered-export"]);
        let config = CheckerProvider::CudaControlConvDefault
            .config(Some(CostProfile::CommonSixConvDefault))
            .unwrap();
        assert_eq!(
            config.cudnn_conv_search,
            PalsCuDnnConvSearch::ExperimentalDefault
        );
        assert_eq!(config.intra_threads, 1);
        assert!(!config.device_public_memory);
        for provider in [
            CheckerProvider::Cpu,
            CheckerProvider::Cuda,
            CheckerProvider::CudaDevice,
            CheckerProvider::CudaControl,
            CheckerProvider::CudaControlShim,
        ] {
            assert!(provider
                .config(Some(CostProfile::CommonSixConvDefault))
                .is_err());
        }
    }

    #[test]
    fn default_cost_tf32_request_is_required_without_mutating_process_environment() {
        let profile = CostProfile::CommonSixConvDefault;
        assert!(profile
            .validate_tf32_override(Some(std::ffi::OsStr::new("0")))
            .is_ok());
        for value in [
            None,
            Some(std::ffi::OsStr::new("1")),
            Some(std::ffi::OsStr::new("")),
            Some(std::ffi::OsStr::new("false")),
        ] {
            assert!(profile.validate_tf32_override(value).is_err());
        }
        assert!(CostProfile::CommonSix.validate_tf32_override(None).is_ok());
    }

    #[test]
    fn cost_flags_are_independent_and_ordinary_numeric_selection_is_unchanged() {
        for other in [
            COST_COMMON_SIX_FLAG,
            COST_COMMON_SIX_CONV_DEFAULT_FLAG,
            HOST_PAGE_CHECK_FLAG,
            "--check-cuda-control-policy",
        ] {
            let mut args = vec![COST_COMMON_SIX_CONV_DEFAULT_FLAG.into(), other.into()];
            assert!(CostProfile::take(&mut args).is_err());
        }
        let original = vec![
            "registered-export".into(),
            COST_COMMON_SIX_CONV_DEFAULT_FLAG.into(),
        ];
        let mut args = original.clone();
        assert_eq!(CostProfile::take(&mut args).unwrap(), None);
        assert_eq!(args, original);
        let config = CheckerProvider::CudaControlConvDefault
            .config(None)
            .unwrap();
        assert_eq!(config.intra_threads, 2);
        assert_eq!(
            config.cudnn_conv_search,
            PalsCuDnnConvSearch::ExperimentalDefault
        );
        let baseline = CheckerProvider::CudaControl
            .config(Some(CostProfile::CommonSix))
            .unwrap();
        assert_eq!(baseline.intra_threads, 1);
        assert_eq!(
            baseline.cudnn_conv_search,
            PalsCuDnnConvSearch::LegacyImplicitExhaustive
        );
    }
}
