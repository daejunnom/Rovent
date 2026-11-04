//! Compile-time source identity for the separate system-runtime startup path.
//!
//! This is an identity for the embedded source inputs, not evidence of the
//! compiled binary, enabled features, native mappings, inference or GPU fence.
//! The historical V5 inventory and the V1/V2 bootstrap digest retain their
//! original framing and verification semantics.

use rz_experiments::{CompiledSourceInventoryV5, ManifestError, SOURCE_COMPONENT_FILES_V5};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

/// New files or build inputs used by the system-runtime path that are not
/// necessarily part of the independently maintained historical inventory.
/// Each value is the actual source bytes embedded in this executable; no
/// expected content or precomputed hash is used as an admission condition.
const SYSTEM_RUNTIME_V3_FILES: &[(&str, &[u8])] = &[
    ("Cargo.toml", include_bytes!("../../../../Cargo.toml")),
    (
        "rust-toolchain.toml",
        include_bytes!("../../../../rust-toolchain.toml"),
    ),
    (
        "crates/rz-uci/src/rovent_bootstrap/source_inventory_v3.rs",
        include_bytes!("source_inventory_v3.rs"),
    ),
    (
        "crates/rz-uci/src/rovent_bootstrap/configuration/v5.rs",
        include_bytes!("configuration/v5.rs"),
    ),
    (
        "crates/rz-uci/src/rovent_bootstrap/launch/v3.rs",
        include_bytes!("launch/v3.rs"),
    ),
    (
        "crates/rz-uci/src/rovent_bootstrap/launch/collection_v3.rs",
        include_bytes!("launch/collection_v3.rs"),
    ),
    (
        "crates/rz-uci/src/rovent_bootstrap/pc_collection_v3.rs",
        include_bytes!("pc_collection_v3.rs"),
    ),
    (
        "crates/rz-uci/src/rovent_bootstrap/pc_collection_v3_root.rs",
        include_bytes!("pc_collection_v3_root.rs"),
    ),
    (
        "crates/rz-uci/src/rovent_bootstrap/product/own_board_v3.rs",
        include_bytes!("product/own_board_v3.rs"),
    ),
    (
        "crates/rz-uci/src/pals_adaptive/session_driver/collection_v3.rs",
        include_bytes!("../pals_adaptive/session_driver/collection_v3.rs"),
    ),
    (
        "crates/rz-uci/src/pals_adaptive/session_driver/collection_v3_root.rs",
        include_bytes!("../pals_adaptive/session_driver/collection_v3_root.rs"),
    ),
    (
        "crates/rz-uci/src/rovent_bootstrap/launch/v3_scope.rs",
        include_bytes!("launch/v3_scope.rs"),
    ),
    (
        "crates/rz-uci/src/rovent_root_admission/source_seal/v3.rs",
        include_bytes!("../rovent_root_admission/source_seal/v3.rs"),
    ),
    (
        "crates/rz-uci/src/pals_adaptive_bootstrap/runtime_assets/system_cuda.rs",
        include_bytes!("../pals_adaptive_bootstrap/runtime_assets/system_cuda.rs"),
    ),
    (
        "crates/rz-native-loader/src/system.rs",
        include_bytes!("../../../rz-native-loader/src/system.rs"),
    ),
    (
        "crates/rz-experiments/src/system_runtime_v6.rs",
        include_bytes!("../../../rz-experiments/src/system_runtime_v6.rs"),
    ),
    (
        "crates/rz-experiments/src/rovent_manifest/v3.rs",
        include_bytes!("../../../rz-experiments/src/rovent_manifest/v3.rs"),
    ),
    (
        "crates/rz-eval/src/lib.rs",
        include_bytes!("../../../rz-eval/src/lib.rs"),
    ),
    (
        "crates/rz-eval/src/adapters/mod.rs",
        include_bytes!("../../../rz-eval/src/adapters/mod.rs"),
    ),
    (
        "crates/rz-eval/src/adapters/lc0/mod.rs",
        include_bytes!("../../../rz-eval/src/adapters/lc0/mod.rs"),
    ),
    (
        "crates/rz-eval/src/adapters/lc0/onnx.rs",
        include_bytes!("../../../rz-eval/src/adapters/lc0/onnx.rs"),
    ),
];

/// Actual V3 child source inputs selected from historical V5 components, plus
/// the separately named system-runtime inputs. Training, arena, offline model
/// export/reference, examples and tests are intentionally not admission
/// dependencies of this child. A caller must bind this observation to the
/// actual V3 execution; it does not confer execution or cleanup authority.
pub fn compiled_system_source_inventory_v3() -> Result<CompiledSourceInventoryV5, ManifestError> {
    let mut inventory = selected_runtime_inventory(SOURCE_COMPONENT_FILES_V5)?;
    let mut paths = BTreeSet::new();
    let mut entries = Vec::with_capacity(SYSTEM_RUNTIME_V3_FILES.len());
    for &(path, bytes) in SYSTEM_RUNTIME_V3_FILES {
        if !paths.insert(path) {
            return Err(ManifestError::Integrity(
                "duplicate V3 system-runtime source path".into(),
            ));
        }
        entries.push((
            path.to_owned(),
            bytes.len() as u64,
            Sha256::digest(bytes).into(),
        ));
    }
    entries.sort_by(|left, right| left.0.cmp(&right.0));
    if inventory
        .insert("system_runtime_v3".into(), entries)
        .is_some()
    {
        return Err(ManifestError::Integrity(
            "duplicate V3 system-runtime source component".into(),
        ));
    }
    Ok(inventory)
}

type EmbeddedSourceComponent<'a> = (&'a str, &'a [(&'a str, &'a [u8])]);

fn selected_runtime_inventory(
    components: &[EmbeddedSourceComponent<'_>],
) -> Result<CompiledSourceInventoryV5, ManifestError> {
    let mut inventory = CompiledSourceInventoryV5::new();
    for &(name, files) in components {
        if !runtime_component(name) {
            continue;
        }
        let mut entries: Vec<_> = files
            .iter()
            .filter(|(path, _)| runtime_file(path))
            .map(|(path, bytes)| {
                (
                    (*path).to_owned(),
                    bytes.len() as u64,
                    Sha256::digest(bytes).into(),
                )
            })
            .collect();
        entries.sort_by(|left, right| left.0.cmp(&right.0));
        if !entries.is_empty() && inventory.insert(name.to_owned(), entries).is_some() {
            return Err(ManifestError::Integrity(
                "duplicate V3 source component".into(),
            ));
        }
    }
    Ok(inventory)
}

fn runtime_component(name: &str) -> bool {
    matches!(
        name,
        "rules"
            | "store"
            | "review_view"
            | "cpu_search"
            | "cpu_worker"
            | "cpu_credit"
            | "coordinator"
            | "role_tickets"
            | "adaptive_costs"
            | "resource_boundary"
            | "uci"
            | "manifest_v5"
            | "model_native"
            | "model_encoding"
            | "nn_runtime"
            | "external_cpu"
            | "data_v3"
    )
}

fn runtime_file(path: &str) -> bool {
    !path.starts_with("crates/rz-arena/")
        && !path.contains("/examples/")
        && !path.contains("/tests/")
        && !path.ends_with("/tests.rs")
        && !path.ends_with("_tests.rs")
        // The V5 roster lists offline sources too. V3 hashes its selected
        // entries and its own framing, not the unrelated full roster text.
        && path != "crates/rz-experiments/src/pals_compiled_sources_v5.rs"
        && path != "crates/rz-experiments/src/pals_source_inventory_v5.json"
}

/// V3-only framing over source component names and actual file identities.
/// This does not change the existing V5 component digests or their consumers.
pub fn compiled_system_source_digest_v3() -> Result<[u8; 32], ManifestError> {
    let inventory = compiled_system_source_inventory_v3()?;
    Ok(inventory_digest(&inventory))
}

/// Identity for production model/encoding/runtime code used by system V3.
/// Unlike the historical `model_native` V5 component this deliberately
/// excludes offline examples, trainer/arena and unrelated UCI startup code.
pub fn compiled_system_model_native_digest_v3() -> Result<[u8; 32], ManifestError> {
    let inventory = compiled_system_source_inventory_v3()?;
    Ok(inventory_digest_for_domain(
        b"rz-rovent-system-model-native-source/3\0",
        &system_model_inventory(&inventory)?,
    ))
}

/// Child component identities for explicit V3 registration. Component names
/// shared with V5 have independently framed V3 digests; the V5 `model_native`
/// key is replaced with `model_native_v3`, and the complete child source
/// digest is separately published as `rovent_system_source_v3`.
pub fn compiled_system_source_components_v3() -> Result<BTreeMap<String, String>, ManifestError> {
    let inventory = compiled_system_source_inventory_v3()?;
    let mut result = BTreeMap::new();
    for (name, rows) in &inventory {
        if name == "model_native" {
            continue;
        }
        let component = BTreeMap::from([(name.clone(), rows.clone())]);
        result.insert(
            name.clone(),
            hex(&inventory_digest_for_domain(
                b"rz-rovent-system-component-source/3\0",
                &component,
            )),
        );
    }
    result.insert(
        "model_native_v3".into(),
        hex(&inventory_digest_for_domain(
            b"rz-rovent-system-model-native-source/3\0",
            &system_model_inventory(&inventory)?,
        )),
    );
    result.insert(
        "rovent_system_source_v3".into(),
        hex(&inventory_digest(&inventory)),
    );
    Ok(result)
}

fn system_model_inventory(
    inventory: &CompiledSourceInventoryV5,
) -> Result<CompiledSourceInventoryV5, ManifestError> {
    let mut selected = BTreeMap::new();
    for name in [
        "model_native",
        "model_encoding",
        "nn_runtime",
        "system_runtime_v3",
    ] {
        let files = inventory.get(name).ok_or_else(|| {
            ManifestError::Integrity("missing V3 model/runtime source component".into())
        })?;
        let files: Vec<_> = files
            .iter()
            .filter(|(path, _, _)| model_runtime_file(path))
            .cloned()
            .collect();
        if files.is_empty() {
            return Err(ManifestError::Integrity(
                "empty V3 model/runtime source component".into(),
            ));
        }
        selected.insert(name.to_owned(), files);
    }
    Ok(selected)
}

fn model_runtime_file(path: &str) -> bool {
    // These are the native model, encoding, scheduler and loader code paths,
    // not their offline Python export/reference or child startup diagnostics.
    (path.starts_with("crates/rz-eval/")
        || path.starts_with("crates/rz-native-loader/")
        || path.starts_with("crates/rz-runtime/")
        || path == "crates/rz-uci/src/pals_native.rs"
        || path.starts_with("crates/rz-uci/src/pals_native/"))
        && runtime_file(path)
        && !path.contains("diagnostic")
}

fn hex(bytes: &[u8; 32]) -> String {
    use std::fmt::Write;
    let mut output = String::with_capacity(64);
    for byte in bytes {
        write!(output, "{byte:02x}").expect("writing to String cannot fail");
    }
    output
}

fn inventory_digest(inventory: &CompiledSourceInventoryV5) -> [u8; 32] {
    inventory_digest_for_domain(b"rz-rovent-system-compiled-source/3\0", inventory)
}

fn inventory_digest_for_domain(domain: &[u8], inventory: &CompiledSourceInventoryV5) -> [u8; 32] {
    fn frame(hash: &mut Sha256, bytes: &[u8]) {
        hash.update((bytes.len() as u64).to_le_bytes());
        hash.update(bytes);
    }
    let mut hash = Sha256::new();
    hash.update(domain);
    hash.update((inventory.len() as u64).to_le_bytes());
    for (component, files) in inventory {
        frame(&mut hash, component.as_bytes());
        let mut sorted: Vec<_> = files.iter().collect();
        sorted.sort_by(|left, right| left.0.cmp(&right.0));
        hash.update((sorted.len() as u64).to_le_bytes());
        for (path, bytes, digest) in sorted {
            frame(&mut hash, path.as_bytes());
            hash.update(bytes.to_le_bytes());
            hash.update(digest);
        }
    }
    hash.finalize().into()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn system_inventory_is_separate_and_tracks_actual_embedded_sources() {
        let inventory = compiled_system_source_inventory_v3().unwrap();
        for (component, files) in &inventory {
            if component != "system_runtime_v3" {
                assert!(runtime_component(component));
                assert!(files.iter().all(|(path, _, _)| runtime_file(path)));
            }
        }
        let files = &inventory["system_runtime_v3"];
        for &(path, bytes) in SYSTEM_RUNTIME_V3_FILES {
            assert!(files.iter().any(|(p, size, digest)| {
                p == path
                    && *size == bytes.len() as u64
                    && *digest == <[u8; 32]>::from(Sha256::digest(bytes))
            }));
        }
        let baseline = inventory_digest(&inventory);
        let mut changed = inventory.clone();
        changed.get_mut("system_runtime_v3").unwrap()[0].2[0] ^= 1;
        assert_ne!(baseline, inventory_digest(&changed));
    }

    #[test]
    fn unrelated_trainer_and_arena_changes_do_not_alter_child_scope() {
        let rules: &[(&str, &[u8])] = &[("crates/rz-position/src/lib.rs", b"rules")];
        let rules_with_arena: &[(&str, &[u8])] = &[
            ("crates/rz-position/src/lib.rs", b"rules"),
            ("crates/rz-arena/src/native_launch.rs", b"unrelated parent"),
        ];
        let trainer: &[(&str, &[u8])] = &[("trainer.rs", b"unrelated training")];
        let baseline = selected_runtime_inventory(&[("rules", rules)]).unwrap();
        let with_unrelated =
            selected_runtime_inventory(&[("rules", rules_with_arena), ("trainer", trainer)])
                .unwrap();
        assert_eq!(baseline, with_unrelated);
        assert_eq!(
            inventory_digest(&baseline),
            inventory_digest(&with_unrelated)
        );
    }

    #[test]
    fn model_component_uses_v3_domain_and_ignores_offline_inputs() {
        let inventory = compiled_system_source_inventory_v3().unwrap();
        let components = compiled_system_source_components_v3().unwrap();
        assert!(!components.contains_key("model_native"));
        assert_eq!(
            components["model_native_v3"],
            hex(&compiled_system_model_native_digest_v3().unwrap())
        );
        assert_eq!(
            components["rovent_system_source_v3"],
            hex(&compiled_system_source_digest_v3().unwrap())
        );
        let baseline = system_model_inventory(&inventory).unwrap();
        let mut changed = inventory.clone();
        changed.get_mut("model_native").unwrap().push((
            "experiments/model-research/pals/trainer.py".into(),
            9,
            [1; 32],
        ));
        assert_eq!(baseline, system_model_inventory(&changed).unwrap());
        let model_row = changed
            .get_mut("model_native")
            .unwrap()
            .iter_mut()
            .find(|(path, _, _)| model_runtime_file(path))
            .unwrap();
        model_row.2[0] ^= 1;
        assert_ne!(baseline, system_model_inventory(&changed).unwrap());
    }
}
