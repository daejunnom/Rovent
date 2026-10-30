//! /2 독립 root 개입 자료의 원 FD payload/source/Rules 검증.
//! /1 Ready decoder나 JSON 재태깅으로 실행/퇴역 권한을 발급하지 않는다.
use super::*;
use crate::rovent_collect::historical_scope::{
    RoventNativeOriginEvidence, RoventRootAdmissionEvidence,
};
use rz_experiments::rovent_utility_cohort::{
    RoventUtilityCohortUsage, RoventUtilityInterventionCohortV2,
};
use rz_experiments::{RoventInputLock, RoventPurpose};
use rz_search::cpu_value::nnuec::v2;
use rz_uci::pals_adaptive::UtilityCohortSourcePins;
use serde::de::DeserializeOwned;
const SOURCE_DOMAIN: &str = "rz-rovent-utility-cohort-collection-source/2";
fn error(message: impl std::fmt::Display) -> ArenaError {
    invalid(format!("utility cohort historical: {message}"))
}
fn ensure(condition: bool, message: &str) -> Result<(), ArenaError> {
    if condition {
        Ok(())
    } else {
        Err(error(message))
    }
}
fn string<'a>(value: &'a Value, key: &str) -> Result<&'a str, ArenaError> {
    value
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| error(format!("missing string {key}")))
}
fn array<'a>(value: &'a Value, key: &str, maximum: usize) -> Result<&'a [Value], ArenaError> {
    let rows = value
        .get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| error(format!("missing array {key}")))?;
    ensure(rows.len() <= maximum, "bounded array")?;
    Ok(rows)
}
fn fields(value: &Value, keys: &[&str]) -> Result<(), ArenaError> {
    let body = value.as_object().ok_or_else(|| error("object required"))?;
    ensure(
        body.len() == keys.len() && keys.iter().all(|key| body.contains_key(*key)),
        "closed original object fields",
    )
}
fn decode<T: DeserializeOwned>(raw: &[u8]) -> Result<T, ArenaError> {
    rz_experiments::decode_json(std::str::from_utf8(raw).map_err(error)?)
        .map_err(ArenaError::Manifest)
}
fn sha(raw: &[u8]) -> String {
    format!("{:x}", Sha256::digest(raw))
}
fn digest_bytes(domain: &str, raw: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(domain.as_bytes());
    h.update([0]);
    h.update((raw.len() as u64).to_le_bytes());
    h.update(raw);
    format!("{:x}", h.finalize())
}
fn digest_value(domain: &str, value: &Value) -> Result<String, ArenaError> {
    let mut value = value.clone();
    value.sort_all_objects();
    Ok(digest_bytes(
        domain,
        &serde_json::to_vec(&value).map_err(error)?,
    ))
}
fn same<A: serde::Serialize, B: serde::Serialize>(a: &A, b: &B) -> Result<bool, ArenaError> {
    Ok(serde_json::to_value(a).map_err(error)? == serde_json::to_value(b).map_err(error)?)
}
fn boolean(value: &Value, key: &str) -> Result<bool, ArenaError> {
    value
        .get(key)
        .and_then(Value::as_bool)
        .ok_or_else(|| error(format!("missing boolean {key}")))
}
fn verify_source(
    publication: &RoventUtilityInterventionPublication,
    files: &BTreeMap<String, Vec<u8>>,
    profile: &RoventUtilityInterventionCohortV2,
) -> Result<
    (
        RoventUtilityInterventionCollectionSource,
        RoventInputLock,
        v2::Model,
        Position,
    ),
    ArenaError,
> {
    let raw_source = files
        .get(&publication.source.path)
        .ok_or_else(|| error("source payload missing"))?;
    let source_value: Value = decode(raw_source)?;
    fields(
        &source_value,
        &[
            "schema",
            "source_pins",
            "nnuec_registration",
            "nnuec_weights_artifact",
            "nnuec_provenance_artifact",
            "native_origin",
            "registration_root",
            "collection_manifest",
            "collection_metadata",
            "rules_root",
            "transposition_sha256",
            "frozen_pc_sources",
            "ancestor_datasets",
            "resource_observation",
        ],
    )?;
    let source: RoventUtilityInterventionCollectionSource =
        serde_json::from_value(source_value.clone()).map_err(error)?;
    ensure(
        same(&source_value, &source)? && source.schema == SOURCE_DOMAIN,
        "closed original collection source",
    )?;
    let lock = RoventInputLock::from_json(
        std::str::from_utf8(&files[&publication.input_lock.path]).map_err(error)?,
    )?;
    let pins = &source.source_pins;
    let original = &profile.input;
    ensure(
        pins.input_lock_sha256 == original.input_lock_sha256
            && pins.source_sha256 == original.source_sha256
            && pins.frozen_nnuec_model_sha256 == original.frozen_nnuec_model_sha256
            && pins.frozen_nnuec_namespace_sha256 == original.frozen_nnuec_namespace_sha256
            && pins.frozen_pc_model_sha256 == original.frozen_pc_model_sha256
            && pins.frozen_pc_namespace_sha256 == original.frozen_pc_namespace_sha256
            && pins.resource_plan_sha256 == original.resource_plan_sha256
            && pins.logical_budget_sha256 == original.logical_budget_sha256,
        "original profile/source pins",
    )?;
    ensure(
        lock.canonical_sha256 == pins.input_lock_sha256
            && lock.manifest.purpose == RoventPurpose::Collection
            && lock.manifest.engine.mode == original.mode.execution_mode()
            && lock.manifest.resources.canonical_sha256()? == pins.resource_plan_sha256
            && rz_experiments::rovent_source_aggregate_sha256(
                &lock.manifest.engine.source_commit,
                &lock.manifest.engine.source_components,
            )? == pins.source_sha256,
        "original full lock/source/mode/resource plan",
    )?;
    source.collection_manifest.validate()?;
    ensure(
        source.collection_manifest.source_sha256 == pins.source_sha256
            && source.collection_manifest.source_commit == lock.manifest.engine.source_commit,
        "original corpus source enrollment",
    )?;
    let registration: RoventRootAdmissionEvidence =
        serde_json::from_value(source.registration_root.clone()).map_err(error)?;
    ensure(
        same(&source.registration_root, &registration)?
            && source.native_origin == source.registration_root["native_origin"],
        "original registration root/native enrollment",
    )?;
    verify_root(&registration, &source.registration_root, &lock, pins)?;
    let root = source.rules_root.restore()?;
    verify_root_rules(&registration, &root)?;
    ensure(
        exact_rules_anchor(&root)?.rules_state_sha256 == registration.root_state.semantic
            && crate::pals_collect_v3::generation_transposition_sha256(&root)
                == source.transposition_sha256,
        "original registration Rules root",
    )?;
    let nnuec = lock
        .manifest
        .engine
        .nnuec
        .as_ref()
        .ok_or_else(|| error("original NNUEC pin missing"))?;
    ensure(
        source.nnuec_weights_artifact == nnuec.weights
            && source.nnuec_provenance_artifact == nnuec.provenance
            && nnuec.weights.sha256 == publication.nnuec_weights.sha256
            && nnuec.weights.bytes == publication.nnuec_weights.bytes
            && nnuec.provenance.sha256 == publication.nnuec_provenance.sha256
            && nnuec.provenance.bytes == publication.nnuec_provenance.bytes,
        "original NNUEC raw artifact pins",
    )?;
    // Parsing/parameter identity only; no board, context encoding or forward.
    let model =
        v2::Model::from_reader(files[&publication.nnuec_weights.path].as_slice()).map_err(error)?;
    ensure(
        model.identity().weights_sha256.as_deref() == Some(pins.frozen_nnuec_model_sha256.as_str()),
        "original NNUEC semantic parameters",
    )?;
    let registered = &source.nnuec_registration;
    fields(
        registered,
        &[
            "domain",
            "schema",
            "semantics",
            "encoding",
            "implementation_sha256",
            "configuration_sha256",
            "context_namespace_policy",
            "local_namespace_policy",
            "context_namespace_policy_sha256",
            "local_namespace_policy_sha256",
            "model_identity",
        ],
    )?;
    let calculated = model.registration().map_err(error)?;
    ensure(
        same(registered, &calculated)?
            && string(registered, "domain")? == v2::registration::REGISTRATION_SCHEMA
            && string(registered, "schema")? == nnuec.schema
            && string(registered, "semantics")? == nnuec.semantics
            && string(registered, "encoding")? == nnuec.encoding
            && string(registered, "implementation_sha256")? == nnuec.implementation_sha256
            && string(registered, "configuration_sha256")? == nnuec.configuration_sha256
            && nnuec.configuration_sha256 == calculated.configuration_sha256()
            && string(registered, "context_namespace_policy_sha256")?
                == nnuec.context_namespace_policy_sha256
            && string(registered, "local_namespace_policy_sha256")?
                == nnuec.local_namespace_policy_sha256
            && string(registered, "context_namespace_policy")?
                == calculated.context_namespace_policy()
            && string(registered, "local_namespace_policy")? == calculated.local_namespace_policy()
            && same(&registered["model_identity"], model.identity())?
            && digest_value("rz-rovent-utility-cohort-nnuec-namespace/1", registered)?
                == pins.frozen_nnuec_namespace_sha256,
        "original NNUEC registered namespace",
    )?;
    let provenance: Value = decode(&files[&publication.nnuec_provenance.path])?;
    fields(
        &provenance,
        &[
            "domain",
            "weights_raw_sha256",
            "semantic_weights_sha256",
            "schema",
            "semantics",
            "encoding",
            "implementation_sha256",
            "configuration_sha256",
            "context_namespace_policy_sha256",
            "local_namespace_policy_sha256",
            "training",
            "diagnostic",
            "provenance",
        ],
    )?;
    ensure(
        string(&provenance, "domain")? == "rz-rovent-nnuec-provenance/1"
            && string(&provenance, "weights_raw_sha256")? == nnuec.weights.sha256
            && string(&provenance, "semantic_weights_sha256")? == pins.frozen_nnuec_model_sha256
            && same(&provenance["training"], &model.identity().training)?
            && provenance["diagnostic"] == serde_json::json!(nnuec.diagnostic),
        "original NNUEC provenance identity",
    )?;
    for name in [
        "schema",
        "semantics",
        "encoding",
        "implementation_sha256",
        "configuration_sha256",
        "context_namespace_policy_sha256",
        "local_namespace_policy_sha256",
    ] {
        ensure(
            provenance[name] == registered[name],
            "original NNUEC provenance/registration",
        )?;
    }
    let mut seen = BTreeSet::new();
    for artifact in array(&provenance, "provenance", 32)? {
        let artifact: rz_experiments::ArtifactRef =
            serde_json::from_value(artifact.clone()).map_err(error)?;
        artifact.validate()?;
        ensure(
            seen.insert(artifact.path.clone()) && lock.manifest.provenance.contains(&artifact),
            "original NNUEC provenance enrollment",
        )?;
    }
    let origin: RoventNativeOriginEvidence =
        serde_json::from_value(source.native_origin.clone()).map_err(error)?;
    let namespace = serde_json::json!({"model_semantics":origin.model_semantics,"model_semantics_sha256":origin.model_semantics_sha256,"encoding":origin.encoding,
        "encoding_sha256":origin.encoding_sha256,"encoding_registry_sha256":origin.encoding_registry_sha256,"model_implementation_sha256":origin.model_implementation_sha256,
        "representation_model_domain":origin.representation_model_domain,"frozen_epoch":origin.frozen_epoch,"weights_sha256":origin.weights_sha256,
        "export_manifest_sha256":origin.export_manifest_sha256,"backend_sha256":origin.backend_sha256});
    ensure(
        digest_value("rz-rovent-utility-cohort-pc-namespace/1", &namespace)?
            == pins.frozen_pc_namespace_sha256
            && origin.weights_sha256 == pins.frozen_pc_model_sha256
            && source.frozen_pc_sources.len() == 2,
        "original frozen P/C namespace",
    )?;
    for ((role, declared), frozen) in [
        ("proposer", &lock.manifest.engine.pc.proposer),
        ("critic", &lock.manifest.engine.pc.critic),
    ]
    .into_iter()
    .zip(&source.frozen_pc_sources)
    {
        frozen.artifact.validate()?;
        ensure(
            frozen.role == role
                && frozen.artifact == declared.weights
                && !declared.diagnostic
                && frozen.model_semantics_sha256 == origin.model_semantics_sha256
                && frozen.model_identity_sha256 == origin.weights_sha256
                && frozen.encoding_sha256 == origin.encoding_sha256
                && declared.weights.sha256 == origin.weights_sha256
                && declared.semantics == origin.model_semantics
                && declared.encoding == origin.encoding
                && declared.implementation_sha256 == origin.model_implementation_sha256,
            "original frozen P/C artifact/source",
        )?;
    }
    let mut ancestors = BTreeSet::new();
    for ancestor in &source.ancestor_datasets {
        validate_intervention_ancestor(ancestor)?;
        ensure(
            ancestors.insert(ancestor.publication_sha256.clone()),
            "duplicate original ancestor",
        )?;
    }
    ensure(
        source.ancestor_datasets.len() <= 256
            && source.resource_observation.len() == publication.trial_count as usize,
        "original ancestor/resource coverage",
    )?;
    for resource in &source.resource_observation {
        verify_resource_custody(resource)?;
    }
    ensure(
        source
            .collection_manifest
            .opening_splits
            .contains_key(&source.collection_metadata.opening_id),
        "original opening split enrollment",
    )?;
    Ok((source, lock, model, root))
}

fn verify_root(
    root: &RoventRootAdmissionEvidence,
    raw: &Value,
    lock: &RoventInputLock,
    pins: &UtilityCohortSourcePins,
) -> Result<(), ArenaError> {
    root.root_descriptor.verify_against(lock)?;
    let input = &root.root_descriptor.input;
    for digest in [
        &root.implementation_sha256,
        &root.selective_policy_sha256,
        &root.native_origin.model_semantics_sha256,
        &root.native_origin.encoding_sha256,
        &root.native_origin.encoding_registry_sha256,
        &root.native_origin.model_implementation_sha256,
        &root.native_origin.backend_sha256,
        &root.native_origin.runtime_library_sha256,
        &root.native_origin.export_manifest_sha256,
        &root.native_origin.native_configuration_sha256,
    ] {
        ensure(generation_sha(digest), "original registered source digest")?;
    }
    ensure(
        root.schema == "rz-rovent-collection-root-admission/1"
            && root.native_origin.schema == "rz-rovent-collection-native-origin/1"
            && generation_digest(&root.native_origin.schema, &raw["native_origin"])?
                == root.native_origin_sha256
            && root.native_origin.input_lock_sha256 == lock.canonical_sha256
            && root.native_origin.source_sha256 == pins.source_sha256
            && root.native_origin.source_commit == lock.manifest.engine.source_commit
            && root.native_origin.source_components == lock.manifest.engine.source_components
            && root.native_origin.mode == format!("{:?}", lock.manifest.engine.mode)
            && root.native_origin.purpose == "collection"
            && root.native_origin.weights_sha256 == pins.frozen_pc_model_sha256
            && root.native_origin.process_epoch == root.process_epoch
            && root.root_descriptor_sha256 == hex(root.root_descriptor.digest()?)
            && root.root_budget_sha256 == root.root_descriptor.root_budget_sha256
            && root.root_state.semantic == root.root_descriptor.input.state_and_history_sha256
            && root.root_moves == root.root_descriptor.input.allowed_root_moves
            && root.deadline_clock == root.root_descriptor.input.clock.hard.clock_epoch
            && root.deadline_tick == root.root_descriptor.input.clock.hard.at_ns
            && root.process_epoch == input.authority.epoch
            && root.game_generation == input.authority.game
            && root.root_generation == input.authority.root
            && root.implementation_sha256 == input.authority.implementation_sha256
            && root.depth_plies == input.original_go.depth_plies.map(|v| v.get())
            && root.mate_moves == input.original_go.mate_moves.map(|v| v.get())
            && root.requested_nodes == input.original_go.requested_nodes,
        "original root descriptor/source/budget/deadline",
    )?;
    ensure(
        digest_value(
            "rz-rovent-utility-cohort-logical-budget/1",
            &serde_json::to_value(&root.root_descriptor.input.declared_ceilings).map_err(error)?,
        )? == pins.logical_budget_sha256,
        "original root logical budget",
    )
}

fn verify_root_rules(
    root: &RoventRootAdmissionEvidence,
    position: &Position,
) -> Result<(), ArenaError> {
    let allowed = root
        .root_moves
        .iter()
        .map(|text| rz_position::BoardMove::from_uci(text).map_err(error))
        .collect::<Result<Vec<_>, _>>()?;
    ensure(
        !allowed.is_empty()
            && allowed.len() <= 218
            && allowed
                .iter()
                .enumerate()
                .all(|(i, m)| !allowed[..i].contains(m))
            && root.root_state.revision == position.revision()
            && root.root_state.semantic == exact_rules_anchor(position)?.rules_state_sha256
            && position
                .ordered_legal_moves()
                .moves()
                .iter()
                .copied()
                .filter(|m| allowed.contains(m))
                .collect::<Vec<_>>()
                == allowed,
        "original root exact history/revision/ordered legal restriction",
    )
}

fn verify_resource_custody(value: &Value) -> Result<(), ArenaError> {
    fields(
        value,
        &[
            "schema",
            "plan_pin",
            "evidence_directory",
            "profile",
            "admission_attempted",
            "finished_attempted",
            "original_plan_descriptor_retained",
            "original_delegated_descriptor_retained",
            "original_scope_retained",
            "resource_failure",
            "admission",
            "finished_live",
            "finished_live_error",
            "controlled_comparison_accepted",
            "automatic_cleanup",
        ],
    )?;
    ensure(
        string(value, "schema")? == "rz-rovent-resource-custody/1"
            && value["admission_attempted"] == true
            && value["original_plan_descriptor_retained"] == true
            && value["original_delegated_descriptor_retained"] == true
            && value["original_scope_retained"] == true
            && value["resource_failure"].is_null()
            && value["controlled_comparison_accepted"] == false
            && value["automatic_cleanup"] == false,
        "original actual resource custody",
    )?;
    let pin: RoventPublicationFile =
        serde_json::from_value(value["plan_pin"].clone()).map_err(error)?;
    pin.validate()?;
    ensure(
        same(&pin, &value["plan_pin"])?
            && pin.path.starts_with("inputs/")
            && pin.bytes <= 64 << 10
            && matches!(
                string(value, "profile")?,
                "functional_validation" | "strict_comparison"
            )
            && !value["admission"].is_null(),
        "original resource plan/admission",
    )?;
    let directory = string(value, "evidence_directory")?;
    ensure(
        !directory.is_empty()
            && !directory.starts_with('/')
            && !directory.contains('\\')
            && directory
                .split('/')
                .all(|v| !v.is_empty() && v != "." && v != ".." && !v.contains(':')),
        "relative resource evidence directory",
    )?;
    let finished = boolean(value, "finished_attempted")?;
    ensure(
        finished || (value["finished_live"].is_null() && value["finished_live_error"].is_null()),
        "resource finish attempt",
    )?;
    if !value["finished_live_error"].is_null() {
        string(value, "finished_live_error")?;
    }
    for key in ["admission", "finished_live"] {
        if !value[key].is_null() {
            let observation: rz_experiments::RoventResourceObservation =
                serde_json::from_value(value[key].clone()).map_err(error)?;
            ensure(
                same(&value[key], &observation)?,
                "closed typed resource observation",
            )?;
            observation.evidence.validate()?;
            // Resource-plan/whole-peak Unknown stays Unknown; this does not award
            // strict performance acceptance from a cohort's recorded declaration.
        }
    }
    Ok(())
}

mod context;
mod pairs;
mod trial;
use super::super::utility_cohort_owner::historical as shared;
use rz_contracts::pals::Move16;
fn number(value: &Value, key: &str) -> Result<u64, ArenaError> {
    value
        .get(key)
        .and_then(Value::as_u64)
        .ok_or_else(|| error(format!("missing unsigned {key}")))
}
fn hash_field<'a>(value: &'a Value, key: &str) -> Result<&'a str, ArenaError> {
    let text = string(value, key)?;
    ensure(generation_sha(text), "original /2 digest")?;
    Ok(text)
}
fn hash32(text: &str) -> Result<Digest, ArenaError> {
    ensure(generation_sha(text), "original /2 hex32")?;
    let mut raw = [0u8; 32];
    for (out, part) in raw.iter_mut().zip(text.as_bytes().chunks_exact(2)) {
        *out = u8::from_str_radix(std::str::from_utf8(part).map_err(error)?, 16).map_err(error)?;
    }
    Ok(Digest(raw))
}
fn packed(bits: u16) -> Result<rz_position::BoardMove, ArenaError> {
    let value = Move16::try_from_bits(bits)
        .and_then(Move16::decode)
        .map_err(error)?;
    rz_position::BoardMove::try_from(value).map_err(error)
}
fn route(value: &str) -> Result<u8, ArenaError> {
    match value {
        "route_p" => Ok(4),
        "route_c" => Ok(5),
        "route_r" => Ok(6),
        _ => Err(error("original /2 recipient")),
    }
}
fn requested(profile: &RoventUtilityInterventionCohortV2) -> Vec<u8> {
    use rz_experiments::rovent_utility_cohort::RoventUtilityCohortRecipient as R;
    profile
        .input
        .requested_recipients
        .iter()
        .map(|r| match r {
            R::Proposer => 4,
            R::Critic => 5,
            R::Researcher => 6,
        })
        .collect()
}
fn mask(value: &Value) -> Result<[bool; 7], ArenaError> {
    let values = value
        .as_array()
        .ok_or_else(|| error("original /2 action mask"))?;
    ensure(values.len() == 7, "original /2 action mask length")?;
    let mut out = [false; 7];
    for (out, value) in out.iter_mut().zip(values) {
        *out = value
            .as_bool()
            .ok_or_else(|| error("original /2 boolean mask"))?;
    }
    Ok(out)
}
fn authority(root: &RoventRootAdmissionEvidence) -> Value {
    serde_json::json!({"process_epoch":root.process_epoch,"game_generation":root.game_generation,
        "root_generation":root.root_generation,"implementation_sha256":root.implementation_sha256})
}
fn sha_set(value: &Value, maximum: usize) -> Result<BTreeSet<String>, ArenaError> {
    let values = value
        .as_array()
        .ok_or_else(|| error("original /2 SHA list"))?;
    ensure(values.len() <= maximum, "original /2 SHA list bound")?;
    let mut out = BTreeSet::new();
    for value in values {
        let text = value
            .as_str()
            .ok_or_else(|| error("original /2 SHA string"))?;
        ensure(
            generation_sha(text) && out.insert(text.to_string()),
            "original /2 unique SHA list",
        )?;
    }
    ensure(
        values
            .iter()
            .filter_map(Value::as_str)
            .eq(out.iter().map(String::as_str)),
        "original /2 SHA list order",
    )?;
    Ok(out)
}
/// Detached history never reconstructs a live Target, arm, retirement or ACK.
pub(super) fn verify_historical_utility_intervention(
    publication: &RoventUtilityInterventionPublication,
    files: &BTreeMap<String, Vec<u8>>,
) -> Result<RoventHistoricalUtilityInterventionEvidence, ArenaError> {
    publication.validate()?;
    let mut paths = BTreeSet::new();
    for pin in publication.payloads() {
        let raw = files
            .get(&pin.path)
            .ok_or_else(|| error("pinned /2 payload missing"))?;
        ensure(
            paths.insert(pin.path.as_str())
                && raw.len() as u64 == pin.bytes
                && sha(raw) == pin.sha256,
            "original /2 raw payload SHA/coverage",
        )?;
    }
    ensure(
        files.len() == paths.len() && files.keys().all(|p| paths.contains(p.as_str())),
        "undeclared /2 historical payload",
    )?;
    let profile = RoventUtilityInterventionCohortV2::from_json(
        std::str::from_utf8(&files[&publication.cohort_profile.path]).map_err(error)?,
    )?;
    let neutral: Value = decode(&files[&publication.neutral_target.path])?;
    let (source, lock, model, root_rules) = verify_source(publication, files, &profile)?;
    ensure(
        publication.trials.len() == profile.input.requested_recipients.len(),
        "/2 complete declared arm coverage",
    )?;
    let mut trials = Vec::new();
    let mut authorities = BTreeSet::new();
    let mut clocks = BTreeSet::new();
    for ordinal in 0..publication.trials.len() {
        let row = trial::verify_trial(
            ordinal,
            publication,
            files,
            &profile,
            &source,
            &lock,
            &model,
            &root_rules,
        )?;
        ensure(
            authorities.insert(serde_json::to_string(&row.authority).map_err(error)?)
                && clocks.insert(row.cost.clock_epoch),
            "/2 reused original root/clock",
        )?;
        trials.push(row);
    }
    let canonical = trials
        .iter()
        .find_map(|row| row.neutral.as_ref())
        .ok_or_else(|| error("/2 original neutral target unavailable"))?;
    ensure(
        canonical.wire == neutral,
        "/2 original first neutral publication differs",
    )?;
    for row in &trials {
        if let Some(actual) = &row.neutral {
            ensure(
                actual.digest == canonical.digest
                    && actual.input.reference_json == canonical.input.reference_json
                    && actual.input.input_sha256 == canonical.input.input_sha256
                    && actual.input.target_sha256 == canonical.input.target_sha256
                    && actual.input.context_sha256 == canonical.input.context_sha256,
                "/2 independent roots changed neutral semantic input",
            )?;
        }
    }
    let registered: RoventRootAdmissionEvidence =
        serde_json::from_value(source.registration_root.clone()).map_err(error)?;
    let owner = serde_json::json!({"profile":hex(profile.digest()?),"original_root":hex(registered.root_descriptor.digest()?),"source":source.source_pins});
    ensure(
        digest_value("rz-rovent-utility-intervention-owner/2", &owner)? == publication.cohort_id,
        "/2 owner original preimage",
    )?;
    let (known_pairs, known_pair_sha256, all_pair_sha256) =
        pairs::verify_pairs(publication, files, &trials, &canonical.digest)?;
    let metadata = &source.collection_metadata;
    let split = *source
        .collection_manifest
        .opening_splits
        .get(&metadata.opening_id)
        .ok_or_else(|| error("original /2 opening split"))?;
    let mut ancestors = source
        .ancestor_datasets
        .iter()
        .map(|a| a.publication_sha256.clone())
        .collect::<Vec<_>>();
    ancestors.sort();
    let first_epoch = trials[0]
        .cost
        .cost_epoch_sha256
        .clone()
        .ok_or_else(|| error("/2 first trial original cost epoch unavailable"))?;
    Ok(RoventHistoricalUtilityInterventionEvidence {
        cohort_id: publication.cohort_id.clone(),
        profile_sha256: hex(profile.digest()?),
        usage: match profile.input.usage {
            RoventUtilityCohortUsage::ValidationOnly => "validation_only",
            RoventUtilityCohortUsage::CollectionCandidate => "collection_candidate",
        }
        .into(),
        collection_source_sha256: source.source_pins.source_sha256.clone(),
        cost_epoch_sha256: first_epoch,
        resource_profile_sha256: source.source_pins.resource_plan_sha256.clone(),
        frozen_base_weights_sha256: source.source_pins.frozen_nnuec_model_sha256.clone(),
        input_reference_json: canonical.input.reference_json.clone(),
        input_sha256: canonical.input.input_sha256.clone(),
        target_sha256: canonical.input.target_sha256.clone(),
        neutral_target_sha256: canonical.digest.clone(),
        role_queries: trials
            .iter()
            .filter_map(|trial| trial.role_query.clone())
            .collect(),
        submission_target_sha256: canonical.input.submission_target_sha256.clone(),
        utility_context_sha256: canonical.input.context_sha256.clone(),
        known_pairs,
        known_pair_sha256,
        all_pair_sha256,
        frozen_pc_sources: source.frozen_pc_sources,
        split,
        opening_id: metadata.opening_id.clone(),
        game_id: metadata.game_id.clone(),
        line_genealogy_id: metadata.line_genealogy_id.clone(),
        transposition_sha256: source.transposition_sha256,
        rules_state_sha256: exact_rules_anchor(&root_rules)?.rules_state_sha256,
        ancestor_dataset_publication_sha256: ancestors,
        ancestor_datasets: source.ancestor_datasets,
    })
}
