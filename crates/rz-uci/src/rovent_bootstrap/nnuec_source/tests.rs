//! Pure source-declaration fixtures. These construct tiny scalar checkpoints,
//! never run a forward, load a runtime, open an artifact or accept training.
use super::*;
use rz_search::cpu_value::nnuec::{
    NNUEC_SCHEMA, NNUEC_SEMANTICS, NnuecCheckpoint, NnuecConfig, NnuecParameters,
    v2::{self, Checkpoint, RelationParameters},
};
use serde_json::Value;
use std::{
    sync::atomic::{AtomicBool, Ordering},
    time::{Duration, Instant},
};

const CONDITIONS: [u8; 32] = [0x55; 32];

fn control() -> RoventStartupControl {
    let until = Instant::now() + Duration::from_secs(30);
    RoventStartupControl {
        until,
        cleanup_until: until + Duration::from_secs(1),
        cancellation: Arc::new(AtomicBool::new(false)),
    }
}

fn artifact(path: &str, bytes: &[u8]) -> ArtifactRef {
    ArtifactRef {
        path: path.into(),
        sha256: format!("{:x}", Sha256::digest(bytes)),
        bytes: bytes.len() as u64,
        source: "https://example.org/rovent/nnuec-source-fixture".into(),
        license: "MIT".into(),
    }
}

fn checkpoint(training: CpuTrainingState) -> Checkpoint {
    Checkpoint {
        schema: v2::SCHEMA.into(),
        semantics: v2::SEMANTICS.into(),
        base: NnuecCheckpoint {
            schema: NNUEC_SCHEMA.into(),
            semantics: NNUEC_SEMANTICS.into(),
            config: NnuecConfig {
                board_width: 1,
                line_width: 1,
                context_width: 1,
                local_width: 1,
                hidden_width: 1,
                causal_window: 3,
                max_lines: 8,
                max_plies: 16,
            },
            training,
            parameters: NnuecParameters {
                board_weights: vec![0.0; 800],
                board_bias: vec![0.0],
                anchor_weights: vec![0.0; 800],
                anchor_bias: vec![0.0],
                from_weights: vec![0.0; 64],
                to_weights: vec![0.0; 64],
                promotion_weights: vec![0.0; 5],
                ply_weights: vec![0.0; 16],
                token_bias: vec![0.0],
                causal_weights: vec![0.0; 3],
                causal_bias: vec![0.0],
                context_projection: vec![0.0],
                context_bias: vec![0.0],
                local_projection: vec![0.0],
                local_bias: vec![0.0],
                hidden_weights: vec![0.0; 4],
                hidden_bias: vec![0.0],
                output_weights: vec![0.0; 11],
                output_bias: vec![0.0; 11],
            },
            output_scale: 1.0,
        },
        parameters: RelationParameters {
            kind_embeddings: vec![0.0; 4],
            pair_parent: vec![0.0; 2],
            pair_child: vec![0.0; 2],
            pair_bias: vec![0.0; 2],
            pair_projection: vec![0.0],
        },
    }
}

struct Fixture {
    model: Model,
    registration: NnuecV2Registration,
    pin: RoventNnuecPin,
    body: NnuecProvenance,
    source_provenance: Vec<ArtifactRef>,
    raw_weights: Vec<u8>,
}
impl Fixture {
    fn new(training: CpuTrainingState, diagnostic: bool) -> Self {
        let checkpoint = checkpoint(training);
        let raw_weights = serde_json::to_vec(&checkpoint).unwrap();
        let model = Model::new(checkpoint).unwrap();
        let registration = model.registration().unwrap();
        let weights = artifact("nnuec/weights-v2.json", &raw_weights);
        let source_provenance = vec![artifact("evidence/source-record.json", b"source fixture")];
        let body = NnuecProvenance {
            domain: NNUEC_PROVENANCE_DOMAIN.into(),
            weights_raw_sha256: weights.sha256.clone(),
            semantic_weights_sha256: model.identity().weights_sha256.clone().unwrap(),
            schema: registration.schema().into(),
            semantics: registration.semantics().into(),
            encoding: registration.encoding().into(),
            implementation_sha256: registration.implementation_sha256().into(),
            configuration_sha256: registration.configuration_sha256().into(),
            context_namespace_policy_sha256: registration.context_namespace_policy_sha256().into(),
            local_namespace_policy_sha256: registration.local_namespace_policy_sha256().into(),
            training: model.identity().training.clone(),
            diagnostic,
            provenance: source_provenance.clone(),
        };
        let pin = RoventNnuecPin {
            schema: body.schema.clone(),
            semantics: body.semantics.clone(),
            encoding: body.encoding.clone(),
            implementation_sha256: body.implementation_sha256.clone(),
            configuration_sha256: body.configuration_sha256.clone(),
            weights,
            provenance: artifact(
                "nnuec/provenance-v1.json",
                &serde_json::to_vec(&body).unwrap(),
            ),
            diagnostic,
            context_namespace_policy_sha256: body.context_namespace_policy_sha256.clone(),
            local_namespace_policy_sha256: body.local_namespace_policy_sha256.clone(),
            search_conditions_sha256: hex(CONDITIONS),
        };
        Self {
            model,
            registration,
            pin,
            body,
            source_provenance,
            raw_weights,
        }
    }
    fn diagnostic() -> Self {
        Self::new(CpuTrainingState::Untrained, true)
    }
    fn verify(&self) -> Result<(), BackendError> {
        verify_declarations(
            &self.pin,
            &self.source_provenance,
            Some(CONDITIONS),
            &self.body,
            &self.model,
            &self.registration,
        )
    }
}

#[test]
fn strict_provenance_rejects_duplicate_unknown_and_nested_fields() {
    let fixture = Fixture::diagnostic();
    let bytes = serde_json::to_vec(&fixture.body).unwrap();
    assert_eq!(decode_provenance(&bytes).unwrap(), fixture.body);
    let text = String::from_utf8(bytes).unwrap();
    let duplicate = format!(
        "{{\"domain\":\"{}\",{}",
        NNUEC_PROVENANCE_DOMAIN,
        &text[1..]
    );
    assert!(decode_provenance(duplicate.as_bytes()).is_err());
    let duplicate_training = text.replace(
        "\"training\":{\"kind\":\"untrained\"}",
        "\"training\":{\"kind\":\"untrained\",\"kind\":\"untrained\"}",
    );
    assert_ne!(duplicate_training, text);
    assert!(decode_provenance(duplicate_training.as_bytes()).is_err());
    for location in ["root", "training", "artifact"] {
        let mut value = serde_json::to_value(&fixture.body).unwrap();
        match location {
            "root" => value["extra"] = Value::Bool(true),
            "training" => value["training"]["optimizer"] = Value::Bool(true),
            _ => value["provenance"][0]["extra"] = Value::Bool(true),
        }
        assert!(
            decode_provenance(&serde_json::to_vec(&value).unwrap()).is_err(),
            "{location}"
        );
    }
    // Empty Untrained is a strict shape, not permission to ignore fields
    // that would be meaningful only for a different training declaration.
    let mut value = serde_json::to_value(&fixture.body).unwrap();
    value["training"]["steps"] = Value::from(1);
    assert!(decode_provenance(&serde_json::to_vec(&value).unwrap()).is_err());
    let learned = CpuTrainingState::Learned {
        run_id: "authored-provenance-declaration".into(),
        steps: 1,
        dataset_sha256: "a".repeat(64),
    };
    let mut value = serde_json::to_value(&fixture.body).unwrap();
    value["training"] = serde_json::to_value(&learned).unwrap();
    assert_eq!(
        decode_provenance(&serde_json::to_vec(&value).unwrap())
            .unwrap()
            .training,
        learned
    );
    value["training"]["optimizer"] = Value::Bool(true);
    assert!(decode_provenance(&serde_json::to_vec(&value).unwrap()).is_err());
}

#[test]
fn provenance_byte_utf8_and_domain_boundaries_are_checked_before_consumption() {
    assert_eq!(
        decode_provenance(&[]).unwrap_err().kind,
        K::ResourceExhausted
    );
    assert_eq!(
        decode_provenance(&vec![b' '; NNUEC_PROVENANCE_MAX_BYTES + 1])
            .unwrap_err()
            .kind,
        K::ResourceExhausted,
    );
    assert_eq!(
        decode_provenance(&[0xff]).unwrap_err().kind,
        K::InvalidInput
    );
    let mut fixture = Fixture::diagnostic();
    fixture.body.domain = "rz-rovent-nnuec-provenance/2".into();
    assert!(decode_provenance(&serde_json::to_vec(&fixture.body).unwrap()).is_err());
}

#[test]
fn raw_bytes_and_semantic_weights_keep_independent_identities() {
    let mut fixture = Fixture::diagnostic();
    fixture.verify().unwrap();
    assert_ne!(
        fixture.body.weights_raw_sha256,
        fixture.body.semantic_weights_sha256
    );
    verify_raw_artifact(
        &fixture.pin.weights,
        &fixture.raw_weights,
        NNUEC_WEIGHTS_MAX_BYTES,
        &control(),
    )
    .unwrap();
    let pretty = serde_json::to_vec_pretty(&checkpoint(CpuTrainingState::Untrained)).unwrap();
    assert_ne!(fixture.raw_weights, pretty);
    let pretty_model = Model::from_reader(pretty.as_slice()).unwrap();
    assert_eq!(pretty_model.identity(), fixture.model.identity());
    fixture.pin.weights = artifact("nnuec/weights-v2.json", &pretty);
    assert_ne!(fixture.pin.weights.sha256, fixture.body.weights_raw_sha256);
    assert!(fixture.verify().is_err());
    fixture.body.weights_raw_sha256 = fixture.pin.weights.sha256.clone();
    fixture.verify().unwrap();
    fixture.body.semantic_weights_sha256 = fixture.pin.weights.sha256.clone();
    assert!(fixture.verify().is_err());
}

#[test]
fn raw_artifact_requires_exact_size_hash_budget_and_original_controls() {
    let bytes = b"bounded raw artifact";
    let original = artifact("nnuec/raw-fixture.json", bytes);
    verify_raw_artifact(&original, bytes, bytes.len(), &control()).unwrap();
    assert_eq!(
        verify_raw_artifact(&original, bytes, bytes.len() - 1, &control())
            .unwrap_err()
            .kind,
        K::ResourceExhausted
    );
    let mut changed = original.clone();
    changed.bytes += 1;
    assert!(verify_raw_artifact(&changed, bytes, bytes.len() + 1, &control()).is_err());
    changed = original.clone();
    changed.sha256 = "a".repeat(64);
    assert!(verify_raw_artifact(&changed, bytes, bytes.len(), &control()).is_err());
    changed = original.clone();
    changed.path = "../foreign.json".into();
    assert!(verify_raw_artifact(&changed, bytes, bytes.len(), &control()).is_err());
    let canceled = control();
    canceled.cancellation.store(true, Ordering::Release);
    assert!(verify_raw_artifact(&original, bytes, bytes.len(), &canceled).is_err());
    let mut expired = control();
    expired.until = Instant::now() - Duration::from_millis(1);
    assert!(verify_raw_artifact(&original, bytes, bytes.len(), &expired).is_err());
}

#[test]
fn each_compiled_pin_field_must_match_actual_registration() {
    let mutations: [fn(&mut RoventNnuecPin); 7] = [
        |pin| pin.schema.push_str("-other"),
        |pin| pin.semantics.push_str("-other"),
        |pin| pin.encoding.push_str("-other"),
        |pin| pin.implementation_sha256 = "a".repeat(64),
        |pin| pin.configuration_sha256 = "b".repeat(64),
        |pin| pin.context_namespace_policy_sha256 = "c".repeat(64),
        |pin| pin.local_namespace_policy_sha256 = "d".repeat(64),
    ];
    for (index, mutate) in mutations.into_iter().enumerate() {
        let mut fixture = Fixture::diagnostic();
        mutate(&mut fixture.pin);
        assert!(fixture.verify().is_err(), "compiled pin field {index}");
    }
}

#[test]
fn each_body_declaration_must_match_actual_model_and_registration() {
    for field in [
        "schema",
        "semantics",
        "encoding",
        "implementation_sha256",
        "configuration_sha256",
        "context_namespace_policy_sha256",
        "local_namespace_policy_sha256",
        "weights_raw_sha256",
        "semantic_weights_sha256",
    ] {
        let mut fixture = Fixture::diagnostic();
        let mut value = serde_json::to_value(&fixture.body).unwrap();
        value[field] = Value::String(if field.ends_with("sha256") {
            "e".repeat(64)
        } else {
            "other-profile".into()
        });
        fixture.body = decode_provenance(&serde_json::to_vec(&value).unwrap()).unwrap();
        assert!(fixture.verify().is_err(), "body field {field}");
    }
}

#[test]
fn configured_search_conditions_are_required_and_must_match_exactly() {
    let mut fixture = Fixture::diagnostic();
    assert!(
        verify_declarations(
            &fixture.pin,
            &fixture.source_provenance,
            None,
            &fixture.body,
            &fixture.model,
            &fixture.registration
        )
        .is_err()
    );
    fixture.pin.search_conditions_sha256 = hex([0x56; 32]);
    assert!(fixture.verify().is_err());
}

#[test]
fn provenance_membership_uses_the_full_original_artifact() {
    let fixture = Fixture::diagnostic();
    assert!(
        verify_declarations(
            &fixture.pin,
            &[],
            Some(CONDITIONS),
            &fixture.body,
            &fixture.model,
            &fixture.registration
        )
        .is_err()
    );
    for field in ["path", "source", "license", "sha256", "bytes"] {
        let mut changed = fixture.source_provenance.clone();
        match field {
            "path" => changed[0].path = "evidence/other.json".into(),
            "source" => changed[0].source = "https://example.org/other-source".into(),
            "license" => changed[0].license = "Apache-2.0".into(),
            "sha256" => changed[0].sha256 = "f".repeat(64),
            _ => changed[0].bytes += 1,
        }
        assert!(
            verify_declarations(
                &fixture.pin,
                &changed,
                Some(CONDITIONS),
                &fixture.body,
                &fixture.model,
                &fixture.registration
            )
            .is_err(),
            "provenance field {field}"
        );
    }
    let mut superset = fixture.source_provenance.clone();
    superset.push(artifact(
        "evidence/unrelated.json",
        b"unrelated original evidence",
    ));
    verify_declarations(
        &fixture.pin,
        &superset,
        Some(CONDITIONS),
        &fixture.body,
        &fixture.model,
        &fixture.registration,
    )
    .unwrap();
}

#[test]
fn provenance_roster_rejects_duplicate_paths_content_aliases_and_overflow() {
    let mut fixture = Fixture::diagnostic();
    let first = fixture.body.provenance[0].clone();
    fixture.body.provenance.push(first.clone());
    assert!(decode_provenance(&serde_json::to_vec(&fixture.body).unwrap()).is_err());
    fixture.body.provenance[1] = artifact(&first.path, b"different bytes same path");
    assert!(decode_provenance(&serde_json::to_vec(&fixture.body).unwrap()).is_err());
    fixture.body.provenance[1] = first;
    fixture.body.provenance[1].path = "evidence/alias.json".into();
    assert!(decode_provenance(&serde_json::to_vec(&fixture.body).unwrap()).is_err());
    fixture.body.provenance = (0..33)
        .map(|index| {
            artifact(
                &format!("evidence/{index}.json"),
                format!("unique source {index}").as_bytes(),
            )
        })
        .collect();
    assert_eq!(
        decode_provenance(&serde_json::to_vec(&fixture.body).unwrap())
            .unwrap_err()
            .kind,
        K::ResourceExhausted
    );
    fixture.body.provenance.clear();
    // An empty declaration adds no evidence; do not invent a nonempty requirement.
    decode_provenance(&serde_json::to_vec(&fixture.body).unwrap()).unwrap();
}

#[test]
fn training_and_diagnostic_declarations_are_exact_metadata_not_training_acceptance() {
    let training = CpuTrainingState::Learned {
        run_id: "declared-only-fixture".into(),
        steps: 1,
        dataset_sha256: "9".repeat(64),
    };
    let learned = Fixture::new(training.clone(), false);
    learned.verify().unwrap();
    assert_eq!(learned.model.identity().training, training);
    let mut fixture = Fixture::diagnostic();
    fixture.body.training = training;
    assert!(fixture.verify().is_err());
    fixture.body.training = CpuTrainingState::Untrained;
    fixture.pin.diagnostic = false;
    assert!(fixture.verify().is_err());
    fixture.pin.diagnostic = true;
    let mut value = serde_json::to_value(&fixture.body).unwrap();
    value["training"] = serde_json::json!({"kind":"learned", "run_id":"invalid-zero-step", "steps":0, "dataset_sha256":"9".repeat(64)});
    assert!(decode_provenance(&serde_json::to_vec(&value).unwrap()).is_err());
}

#[test]
fn a_registration_from_other_actual_parameters_cannot_validate_this_model() {
    let fixture = Fixture::diagnostic();
    let mut other = checkpoint(CpuTrainingState::Untrained);
    other.base.parameters.board_bias[0] = 0.25;
    let other = Model::new(other).unwrap();
    let other_registration = other.registration().unwrap();
    assert_eq!(
        other_registration.configuration_sha256(),
        fixture.registration.configuration_sha256()
    );
    assert_ne!(
        other_registration.model_identity(),
        fixture.model.identity()
    );
    assert!(
        verify_declarations(
            &fixture.pin,
            &fixture.source_provenance,
            Some(CONDITIONS),
            &fixture.body,
            &fixture.model,
            &other_registration
        )
        .is_err()
    );
}
