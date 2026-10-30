//! 원 publication으로 회수한 데이터의 source·입력 scope 관측이다.
//! 이 함수는 live native/physical/optimizer 권한을 복원하지 않는다.
use super::*;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoventNativeOriginEvidence {
    pub schema: String,
    pub purpose: String,
    pub process_epoch: u64,
    pub mode: String,
    pub input_lock_sha256: String,
    pub source_commit: String,
    pub source_components: BTreeMap<String, String>,
    pub source_sha256: String,
    pub weights_sha256: String,
    pub export_manifest_sha256: String,
    pub backend_sha256: String,
    pub runtime_library_sha256: String,
    pub runtime_bundle_sha256: Option<String>,
    pub native_configuration_sha256: String,
    pub model_semantics: String,
    pub model_semantics_sha256: String,
    pub encoding: String,
    pub encoding_sha256: String,
    pub encoding_registry_sha256: String,
    pub model_implementation_sha256: String,
    pub frozen_epoch: u64,
    pub representation_model_domain: String,
    pub runtime_sha256_domain: String,
    pub runtime_sha256: String,
    pub teacher_profiles: Vec<rz_experiments::RoventExternalProfilePin>,
}
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoventRootStateEvidence {
    pub owner: u64,
    pub revision: u64,
    pub semantic: String,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoventRootAdmissionEvidence {
    pub schema: String,
    pub native_origin: RoventNativeOriginEvidence,
    pub native_origin_sha256: String,
    pub process_epoch: u64,
    pub game_generation: u64,
    pub root_generation: u64,
    pub implementation_sha256: String,
    pub deadline_clock: u64,
    pub deadline_tick: u64,
    pub root_budget_sha256: String,
    pub selective_policy_sha256: String,
    pub root_state: RoventRootStateEvidence,
    pub store_root_state: usize,
    pub root_moves: Vec<String>,
    pub depth_plies: Option<u16>,
    pub mate_moves: Option<u16>,
    pub requested_nodes: Option<u64>,
    pub root_descriptor: rz_experiments::RoventRootDescriptor,
    pub root_descriptor_sha256: String,
}

/// Historical data observations only. The original ticket-model and loaded
/// weight domains stay separately named even in V3's checkpoint-key mapping.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RoventHistoricalRecordEvidence {
    pub native_admission_sha256: String,
    pub collection_source_commit: String,
    pub collection_source_components: BTreeMap<String, String>,
    pub collection_source_sha256: String,
    pub collection_runtime_library_sha256: String,
    pub collection_runtime_bundle_sha256: Option<String>,
    pub collection_backend_sha256: String,
    pub collection_native_configuration_sha256: String,
    pub collection_input_lock_sha256: String,
    pub model_semantics_sha256: String,
    pub model_identity_sha256: String,
    pub weights_sha256: String,
    pub encoding_sha256: String,
    pub root_descriptor: rz_experiments::RoventRootDescriptor,
    pub external_source: Option<super::external_scope::RoventHistoricalExternalSource>,
}

/// Historical DTO source joins, full original Rules replay and exact typed
/// input/mask/native-key recalculation. It grants no current execution right.
pub fn verify_historical_record_evidence(
    record: &RoventTrainingRecord,
    manifest: &RoventCorpusManifest,
) -> Result<RoventHistoricalRecordEvidence, ArenaError> {
    record.validate_rules_replay_v2(manifest)?;
    super::rules_replay::verify_historical_record_rules(record)?;
    verify_historical_native_scope(record, manifest)
}

/// 각 명시 형식의 전체 검증 뒤에만 호출하는 공통 native/source 계산이다.
/// projection을 legacy record로 발행하거나 raw schema를 재태그하지 않는다.
pub(super) fn verify_historical_native_scope(
    record: &RoventTrainingRecord,
    manifest: &RoventCorpusManifest,
) -> Result<RoventHistoricalRecordEvidence, ArenaError> {
    verify_historical_native_fields(&record.input, &record.native, manifest)
}

/// 형식별 validator 뒤에 원 input/native 필드만 대조한다. 임시 전체 row나
/// target/lineage를 만들지 않으며 metadata에서 실행 권한을 복구하지 않는다.
pub(super) fn verify_historical_native_fields(
    generation_input: &GenerationInputV3,
    native: &RoventNativeLifecycle,
    manifest: &RoventCorpusManifest,
) -> Result<RoventHistoricalRecordEvidence, ArenaError> {
    verify_historical_native_fields_for_root(generation_input, native, manifest, false)
}

/// 새 raw3만 root2를 명시 선택한다. 원 root1/raw2 decoder를 확대하지 않는다.
pub(super) fn verify_historical_submitted_native_fields(
    generation_input: &GenerationInputV3,
    native: &RoventNativeLifecycle,
    manifest: &RoventCorpusManifest,
) -> Result<RoventHistoricalRecordEvidence, ArenaError> {
    rz_experiments::rovent_corpus_v2::validate_submitted_root_admission(
        &native.raw_retirement["root_admission"],
        &native.raw_retirement["root_admission_sha256"],
    )?;
    verify_historical_native_fields_for_root(generation_input, native, manifest, true)
}

fn verify_historical_native_fields_for_root(
    generation_input: &GenerationInputV3,
    native: &RoventNativeLifecycle,
    manifest: &RoventCorpusManifest,
    submitted: bool,
) -> Result<RoventHistoricalRecordEvidence, ArenaError> {
    let retirement = &native.raw_retirement;
    let raw = &retirement["root_admission"];
    let external = submitted
        && raw["schema"] == rz_experiments::rovent_corpus_v2::ROVENT_EXTERNAL_COLLECTION_ROOT;
    let external_source = if external {
        Some(super::external_scope::RoventHistoricalExternalSource::from_original_root(raw)?)
    } else {
        None
    };
    // 이 임시 값은 공통 metadata 필드만 읽는다. 원 schema와 실제 raw2/raw3
    // 본문·SHA를 유지하며 root1로 재태그하거나 별도 artifact로 발행하지 않는다.
    let mut fields = raw.clone();
    if external {
        let object = fields
            .as_object_mut()
            .ok_or_else(|| invalid("external root object"))?;
        object.remove("external_source");
        object.remove("external_source_sha256");
    }
    let root: RoventRootAdmissionEvidence = serde_json::from_value(fields)
        .map_err(|_| invalid("missing/malformed explicit Rovent root admission evidence"))?;
    let origin = &root.native_origin;
    let origin_raw = &raw["native_origin"];
    let origin_domain = "rz-rovent-collection-native-origin/1";
    let root_domain = if external {
        rz_experiments::rovent_corpus_v2::ROVENT_EXTERNAL_COLLECTION_ROOT
    } else {
        "rz-rovent-collection-root-admission/1"
    };
    let root_sha = generation_digest(root_domain, raw)?;
    let ticket = &native.raw_submission["ticket"];
    let source = rz_experiments::rovent_source_aggregate_sha256(
        &origin.source_commit,
        &origin.source_components,
    )?;
    root.root_descriptor.validate()?;
    let descriptor = &root.root_descriptor.input;
    let semantics = hex(Digest(
        rz_eval::pals_model::PalsModelProfile::VirtualReviewV1.registered_semantics_sha256(),
    ));
    let encoding = hex(Digest(
        Sha256::digest(rz_eval::pals_model::virtual_review::ENCODING.as_bytes()).into(),
    ));
    check(
        root.schema == root_domain
            && origin.schema == origin_domain
            && origin.purpose == "collection"
            && generation_digest(origin_domain, origin_raw)? == root.native_origin_sha256
            && retirement["root_admission_sha256"] == root_sha
            && root.process_epoch == native.process_epoch
            && origin.process_epoch == root.process_epoch
            && root.game_generation == native.game_generation
            && root.root_generation == native.root_generation
            && ticket["implementation_sha256"] == root.implementation_sha256
            && ticket["deadline_clock"].as_u64() == Some(root.deadline_clock)
            && ticket["deadline_tick"].as_u64() == Some(root.deadline_tick)
            && ticket["root_state"]
                == serde_json::to_value(&root.root_state)
                    .map_err(|error| invalid(error.to_string()))?
            && source == origin.source_sha256
            && source == manifest.source_sha256
            && origin.source_commit == manifest.source_commit
            && origin.runtime_sha256_domain == "loaded_runtime_library_file_sha256"
            && origin.runtime_sha256 == origin.runtime_library_sha256
            && origin.runtime_library_sha256 == manifest.runtime_sha256
            && origin.model_semantics == rz_eval::pals_model::virtual_review::MODEL_SEMANTICS
            && origin.model_semantics_sha256 == semantics
            && semantics == manifest.model_semantics_sha256
            && origin.encoding == rz_eval::pals_model::virtual_review::ENCODING
            && origin.encoding_sha256 == encoding
            && encoding == native.encoding_sha256
            && origin.encoding_registry_sha256
                == hex(Digest(
                    rz_eval::pals_model::virtual_review::registered_encoding_sha256(),
                ))
            && origin.representation_model_domain == "loaded_checkpoint_sha256"
            && origin.weights_sha256 == native.model_sha256
            && origin.frozen_epoch
                == ticket["representation"]["frozen_epoch"]
                    .as_u64()
                    .unwrap_or(0)
            && root.root_descriptor_sha256 == hex(root.root_descriptor.digest()?)
            && descriptor.startup_lock_sha256 == origin.input_lock_sha256
            && descriptor.authority.epoch == root.process_epoch
            && descriptor.authority.game == root.game_generation
            && descriptor.authority.root == root.root_generation
            && descriptor.authority.implementation_sha256 == root.implementation_sha256
            && root.root_descriptor.root_budget_sha256 == root.root_budget_sha256
            && descriptor.clock.hard.clock_epoch == root.deadline_clock
            && descriptor.clock.hard.at_ns == root.deadline_tick
            && descriptor.state_and_history_sha256 == root.root_state.semantic
            && descriptor.allowed_root_moves == root.root_moves
            && descriptor.original_go.depth_plies.map(|value| value.get()) == root.depth_plies
            && descriptor.original_go.mate_moves.map(|value| value.get()) == root.mate_moves
            && descriptor.original_go.requested_nodes == root.requested_nodes
            && [
                &root.native_origin_sha256,
                &root_sha,
                &root.implementation_sha256,
                &root.root_budget_sha256,
                &root.selective_policy_sha256,
                &origin.input_lock_sha256,
                &origin.weights_sha256,
                &origin.export_manifest_sha256,
                &origin.backend_sha256,
                &origin.runtime_library_sha256,
                &origin.native_configuration_sha256,
                &origin.model_implementation_sha256,
            ]
            .into_iter()
            .all(|value| generation_sha(value))
            && origin
                .runtime_bundle_sha256
                .as_ref()
                .is_none_or(|value| generation_sha(value)),
        "historical native/root/source/model/runtime evidence does not join original input",
    )?;
    check(
        matches!(
            origin.mode.as_str(),
            "HybridPcr" | "HybridPc" | "CpuPcr" | "CpuPc"
        ) && !root.root_moves.is_empty()
            && root.root_moves.len() <= 218,
        "historical collection execution mode/root restriction",
    )?;
    let submission: super::rules_replay::RoventSubmittedRulesReplay =
        serde_json::from_value(native.raw_submission["rules_replay"].clone())
            .map_err(|_| invalid("missing Rules submission replay"))?;
    let actual = submission.actual_root.restore()?;
    let root_moves = root
        .root_moves
        .iter()
        .map(|movement| BoardMove::from_uci(movement).map_err(|error| invalid(error.to_string())))
        .collect::<Result<Vec<_>, _>>()?;
    let legal = actual.legal_moves();
    check(
        root_moves.iter().enumerate().all(|(slot, movement)| {
            legal.contains(movement) && !root_moves[..slot].contains(movement)
        }),
        "historical root restriction contains an illegal/duplicate Rules move",
    )?;
    check(
        hex(rz_position::contracts::rules_snapshot_semantic_digest(
            &actual.snapshot(),
        )) == descriptor.state_and_history_sha256
            && actual.revision() == root.root_state.revision
            && actual
                .ordered_legal_moves()
                .moves()
                .iter()
                .copied()
                .filter(|movement| root_moves.contains(movement))
                .collect::<Vec<_>>()
                == root_moves,
        "historical actual go Rules history/ordered root domain differs from original descriptor",
    )?;
    for pin in manifest.teachers.values() {
        check(
            origin.teacher_profiles.iter().any(|profile| {
                profile.canonical_sha256 == pin.canonical_sha256
                    && profile.capability_sha256 == pin.capability_sha256
                    && profile.profile.sha256 == pin.profile_sha256
                    && profile.binary.sha256 == pin.binary_sha256
                    && profile.binary.bytes == pin.binary_bytes
                    && profile.source_commit == pin.source_commit
            }),
            "historical corpus teacher is absent from original loaded lock",
        )?;
    }
    verify_original_native_key(generation_input, native, &submission, origin)?;
    Ok(RoventHistoricalRecordEvidence {
        native_admission_sha256: root_sha,
        collection_source_commit: origin.source_commit.clone(),
        collection_source_components: origin.source_components.clone(),
        collection_source_sha256: source,
        collection_runtime_library_sha256: origin.runtime_library_sha256.clone(),
        collection_runtime_bundle_sha256: origin.runtime_bundle_sha256.clone(),
        collection_backend_sha256: origin.backend_sha256.clone(),
        collection_native_configuration_sha256: origin.native_configuration_sha256.clone(),
        collection_input_lock_sha256: origin.input_lock_sha256.clone(),
        model_semantics_sha256: semantics,
        model_identity_sha256: native.model_sha256.clone(),
        weights_sha256: origin.weights_sha256.clone(),
        encoding_sha256: encoding,
        root_descriptor: root.root_descriptor.clone(),
        external_source,
    })
}

fn typed_virtual_input(value: &Value) -> Result<PalsVirtualReviewInput, ArenaError> {
    fn tokens(value: &mut Value) -> Result<(), ArenaError> {
        let array = value
            .as_array_mut()
            .ok_or_else(|| invalid("virtual input move list"))?;
        for movement in array {
            let values = movement
                .as_array()
                .ok_or_else(|| invalid("virtual input move components"))?;
            check(values.len() == 3, "virtual input move component count")?;
            *movement = json!({"from":values[0],"to":values[1],"promotion":values[2]});
        }
        Ok(())
    }
    let mut value = value.clone();
    tokens(&mut value["candidates"])?;
    for record in value["records"]
        .as_array_mut()
        .ok_or_else(|| invalid("virtual records"))?
    {
        tokens(&mut record["line"])?;
    }
    for name in ["prefix", "proposal", "counter"] {
        tokens(&mut value["query"][name])?;
    }
    let typed: PalsVirtualReviewInput =
        serde_json::from_value(value).map_err(|error| invalid(error.to_string()))?;
    typed
        .validate()
        .map_err(|error| invalid(error.to_string()))?;
    Ok(typed)
}
fn verify_original_native_key(
    generation_input: &GenerationInputV3,
    native: &RoventNativeLifecycle,
    replay: &super::rules_replay::RoventSubmittedRulesReplay,
    origin: &RoventNativeOriginEvidence,
) -> Result<(), ArenaError> {
    verify_original_native_key_for_sources(
        generation_input,
        native,
        replay,
        &origin.weights_sha256,
        &origin.backend_sha256,
    )
}

/// Shared input/key calculation after the caller has verified its own exact
/// origin domain and file identities. This never reconstructs an authority.
pub(super) fn verify_original_native_key_for_sources(
    generation_input: &GenerationInputV3,
    native: &RoventNativeLifecycle,
    replay: &super::rules_replay::RoventSubmittedRulesReplay,
    weights_sha256: &str,
    backend_sha256: &str,
) -> Result<(), ArenaError> {
    let input = typed_virtual_input(&generation_input.encoded_model_input)?;
    let anchors = replay
        .selected_anchors
        .iter()
        .map(|anchor| anchor.restore())
        .collect::<Result<Vec<_>, _>>()?;
    let history = anchors
        .iter()
        .map(rz_uci::pals_native::pals_history_digest)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| invalid(error.to_string()))?;
    let identity = ReviewExecutionIdentity {
        model_epoch: rz_eval::asset::parse_sha256(weights_sha256)
            .map_err(|error| invalid(error.to_string()))?,
        backend: rz_eval::asset::parse_sha256(backend_sha256)
            .map_err(|error| invalid(error.to_string()))?,
        rules_history: *history
            .first()
            .ok_or_else(|| invalid("original Rules root absent"))?,
        anchor_history_seals: history,
        fp32: true,
    };
    let query = &generation_input.selected_view.query;
    let mut endpoint = anchors
        .get(query.anchors[0] as usize)
        .ok_or_else(|| invalid("query root absent"))?
        .clone();
    for movement in &query.prefix {
        endpoint
            .make_move(
                rz_contracts::pals::Move16::try_from_bits(*movement)
                    .and_then(|movement| movement.decode())
                    .map_err(|error| invalid(error.to_string()))?
                    .try_into()
                    .map_err(|error: rz_contracts::ContractError| invalid(error.to_string()))?,
            )
            .map_err(|error| invalid(error.to_string()))?;
    }
    let history = rz_uci::pals_native::pals_history_digest(&endpoint)
        .map_err(|error| invalid(error.to_string()))?;
    let original = &native.raw_submission["ticket"]["representation"];
    let revision = original["record_revision"]
        .as_u64()
        .ok_or_else(|| invalid("original native record revision absent"))?;
    let key = rz_uci::pals_native::ticketed::virtual_review_representation_for_input(
        &input, &identity, history, revision,
    )
    .map_err(|error| invalid(format!("{error:?}")))?;
    let expected = json!({"input":hex(key.input),"history":hex(key.history),"candidates":hex(key.candidates),
        "mask_positions":hex(key.mask_positions),"model":hex(key.model),"encoding":hex(key.encoding),
        "precision":format!("{:?}",key.precision),"frozen_epoch":key.frozen_epoch,"record_revision":key.record_revision,"role":role_name(key.role)});
    check(
        *original == expected && hex(key.input) == native.input_digest,
        "historical model fields/masks/Fresh input seal differ from original native representation",
    )
}
