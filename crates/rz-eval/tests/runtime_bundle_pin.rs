//! Authored, tiny fake-library fixtures. No test executes ORT, CUDA or native code.

use rz_eval::runtime_pin::{
    CudaRuntimeBundleSpec, RuntimeBundleAlias, RuntimeBundleFile, RuntimeBundleFileRole as Role,
    RuntimeLibraryPin,
};
use sha2::{Digest, Sha256};

const CORE: &str = "libonnxruntime.so.1.22.0";
const SHARED: &str = "libonnxruntime_providers_shared.so";
const CUDA: &str = "libonnxruntime_providers_cuda.so";
const CUDART: &str = "libcudart.so.12";
const NVIDIA: &[&str] = &[
    CUDART,
    "libnvJitLink.so.12",
    "libnvrtc-builtins.so.12.8",
    "libnvrtc.so.12",
    "libcublasLt.so.12",
    "libcublas.so.12",
    "libcurand.so.10",
    "libcufft.so.11",
    "libcudnn_graph.so.9",
    "libcudnn_ops.so.9",
    "libcudnn_adv.so.9",
    "libcudnn_cnn.so.9",
    "libcudnn_engines_precompiled.so.9",
    "libcudnn_engines_runtime_compiled.so.9",
    "libcudnn_heuristic.so.9",
    "libcudnn.so.9",
];
const FAKE_BYTES: &[u8] = b"authored fake bundle bytes; no native execution";

fn hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn entry(role: Role, filename: &str) -> RuntimeBundleFile {
    RuntimeBundleFile {
        role,
        filename: filename.into(),
        bytes: FAKE_BYTES.len() as u64,
        sha256: hex(FAKE_BYTES),
    }
}

fn spec() -> CudaRuntimeBundleSpec {
    let mut files = vec![
        entry(Role::Core, CORE),
        entry(Role::ProvidersShared, SHARED),
        entry(Role::ProvidersCuda, CUDA),
    ];
    files.extend(
        NVIDIA
            .iter()
            .map(|filename| entry(Role::NvidiaDependency, filename)),
    );
    CudaRuntimeBundleSpec {
        schema_version: 1,
        files,
        loader_aliases: None,
    }
}

fn manifest_json(spec: &CudaRuntimeBundleSpec) -> String {
    let files: Vec<_> = spec
        .files
        .iter()
        .map(|file| {
            let role = match file.role {
                Role::Core => "core",
                Role::ProvidersShared => "providers_shared",
                Role::ProvidersCuda => "providers_cuda",
                Role::NvidiaDependency => "nvidia_dependency",
            };
            format!(
                "{{\"role\":\"{role}\",\"filename\":\"{}\",\"bytes\":{},\"sha256\":\"{}\"}}",
                file.filename, file.bytes, file.sha256
            )
        })
        .collect();
    let alias_json = match &spec.loader_aliases {
        Some(aliases) => format!(
            ",\"loader_aliases\":[{}]",
            aliases
                .iter()
                .map(|alias| format!(
                    "{{\"filename\":\"{}\",\"target\":\"{}\"}}",
                    alias.filename, alias.target
                ))
                .collect::<Vec<_>>()
                .join(",")
        ),
        None => String::new(),
    };
    format!(
        "{{\"schema_version\":{},\"files\":[{}]{}}}",
        spec.schema_version,
        files.join(","),
        alias_json,
    )
}

fn alias_spec() -> CudaRuntimeBundleSpec {
    let mut spec = spec();
    spec.schema_version = 2;
    spec.loader_aliases = Some(
        NVIDIA
            .iter()
            .filter(|filename| filename.starts_with("libcudnn"))
            .map(|target| RuntimeBundleAlias {
                filename: format!("{target}.8.0"),
                target: (*target).into(),
            })
            .collect(),
    );
    spec
}

#[test]
fn alias_manifest_binds_closed_declarations_and_preserves_legacy_digest() {
    let legacy = spec();
    let legacy_digest = legacy.digest().unwrap();
    let legacy_hex: String = legacy_digest
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();
    // Independent binary-codec-v1 golden from the authored nineteen-file fixture.
    assert_eq!(
        legacy_hex,
        "8637ae81a64922e83091d99fcdf213f6d8712ab49c0111ef7a56c5441b0fc0a7"
    );
    let source = manifest_json(&legacy);
    assert!(!source.contains("loader_aliases"));
    assert_eq!(CudaRuntimeBundleSpec::from_json(&source).unwrap(), legacy);
    let aliases = alias_spec();
    let aliased_digest = aliases.digest().unwrap();
    assert_ne!(legacy_digest, aliased_digest);
    assert_eq!(
        CudaRuntimeBundleSpec::from_json(&manifest_json(&aliases)).unwrap(),
        aliases
    );
    let mut reordered = aliases.clone();
    reordered.files.reverse();
    reordered.loader_aliases.as_mut().unwrap().reverse();
    assert_eq!(aliased_digest, reordered.digest().unwrap());
    let mut subset = aliases.clone();
    subset.loader_aliases.as_mut().unwrap().pop();
    assert_ne!(aliased_digest, subset.digest().unwrap());
    let mut v1_with_aliases = aliases.clone();
    v1_with_aliases.schema_version = 1;
    assert!(v1_with_aliases.validate().is_err());
    let mut v2_without_aliases = legacy.clone();
    v2_without_aliases.schema_version = 2;
    assert!(v2_without_aliases.validate().is_err());
    let mut empty = aliases.clone();
    empty.loader_aliases = Some(vec![]);
    assert!(empty.validate().is_err());
    let mut duplicate = aliases.clone();
    let repeated = duplicate.loader_aliases.as_ref().unwrap()[0].clone();
    duplicate.loader_aliases.as_mut().unwrap()[1] = repeated;
    assert!(duplicate.validate().is_err());
    for (filename, target) in [
        ("../libcudnn_graph.so.9.8.0", "libcudnn_graph.so.9"),
        ("libcudnn_graph.so.9.8.0", "/lib/libcudnn_graph.so.9"),
        ("libcudnn_graph.so.9.8.0", "../libcudnn_graph.so.9"),
        ("libcudnn_graph.so.9.8.0", "./libcudnn_graph.so.9"),
        ("libcudnn_graph.so.9.8.0", "libcudnn_ops.so.9"),
        ("libcudnn_graph.so.9.9.0", "libcudnn_graph.so.9"),
        ("libcustom.so.9.8.0", "libcudnn_graph.so.9"),
    ] {
        let mut invalid = aliases.clone();
        invalid.loader_aliases.as_mut().unwrap()[0] = RuntimeBundleAlias {
            filename: filename.into(),
            target: target.into(),
        };
        assert!(invalid.validate().is_err());
        assert!(!format!("{invalid:?}").contains("../"));
        assert!(!format!("{invalid:?}").contains("/lib/"));
    }
    let json = manifest_json(&aliases);
    assert!(CudaRuntimeBundleSpec::from_json(&json.replacen(
        "\"target\":",
        "\"unknown\":0,\"target\":",
        1
    ))
    .is_err());
    assert!(CudaRuntimeBundleSpec::from_json(&json.replacen(
        "\"target\":",
        "\"filename\":\"duplicate\",\"target\":",
        1
    ))
    .is_err());
    assert_eq!(legacy.digest().unwrap(), legacy_digest);
}

#[test]
fn manifest_rejects_unknown_duplicate_fields_and_unbounded_json() {
    let source = manifest_json(&spec());
    assert!(CudaRuntimeBundleSpec::from_json(&source).is_ok());
    let rejected = [
        source.replacen(
            "\"schema_version\":1",
            "\"schema_version\":1,\"unknown\":0",
            1,
        ),
        source.replacen(
            "\"schema_version\":1",
            "\"schema_version\":1,\"schema_version\":1",
            1,
        ),
        source.replacen("\"bytes\":", "\"unknown\":0,\"bytes\":", 1),
        source.replacen(
            "\"role\":\"core\"",
            "\"role\":\"core\",\"role\":\"core\"",
            1,
        ),
        source.replacen("\"sha256\":", "\"bytes\":1,\"sha256\":", 1),
        format!("{source}{}", " ".repeat(64 * 1024)),
    ];
    for json in rejected {
        assert!(CudaRuntimeBundleSpec::from_json(&json).is_err());
    }
}

#[test]
fn canonical_identity_survives_order_and_binds_size_and_content() {
    let original = spec();
    let expected = original.digest().unwrap();
    let mut reordered = original.clone();
    reordered.files.reverse();
    assert_eq!(expected, reordered.digest().unwrap());
    assert_eq!(
        expected,
        CudaRuntimeBundleSpec::from_json(&manifest_json(&reordered))
            .unwrap()
            .digest()
            .unwrap()
    );
    let mut changed_size = original.clone();
    changed_size.files[0].bytes += 1;
    assert_ne!(expected, changed_size.digest().unwrap());
    let mut changed_content = original.clone();
    changed_content.files[0].sha256 = hex(b"another authored fixture");
    assert_ne!(expected, changed_content.digest().unwrap());
    let mut changed_content = original.clone();
    changed_content.files[3].sha256 = hex(b"another NVIDIA fixture");
    assert_ne!(expected, changed_content.digest().unwrap());
}

#[test]
fn manifest_rejects_unsafe_names_wrong_roles_missing_core_and_bad_hashes() {
    for filename in [
        "../libcudart.so.12",
        "/tmp/libcudart.so.12",
        "..\\libcudart.so.12",
        "libunknown.so.12",
    ] {
        let mut invalid = spec();
        invalid.files[3].filename = filename.into();
        assert!(invalid.validate().is_err());
        assert!(invalid.digest().is_err());
    }
    let mut invalid = spec();
    invalid.files[0].role = Role::NvidiaDependency;
    assert!(invalid.validate().is_err());
    let mut invalid = spec();
    invalid.files.remove(0);
    assert!(invalid.validate().is_err());
    let mut invalid = spec();
    invalid.files.pop();
    assert!(
        invalid.validate().is_err(),
        "the first GPU profile requires all 16 NVIDIA entries"
    );
    let mut invalid = spec();
    invalid.files.push(invalid.files[3].clone());
    assert!(invalid.validate().is_err());
    for hash in ["f".repeat(63), "G".repeat(64), "A".repeat(64)] {
        let mut invalid = spec();
        invalid.files[0].sha256 = hash;
        assert!(invalid.validate().is_err());
    }
    let mut invalid = spec();
    invalid.schema_version = 2;
    assert!(invalid.validate().is_err());
}

#[test]
fn manifest_enforces_file_count_individual_and_total_byte_budgets() {
    for bytes in [0, 1024 * 1024 * 1024 + 1, u64::MAX] {
        let mut invalid = spec();
        invalid.files[0].bytes = bytes;
        assert!(invalid.validate().is_err());
    }
    let mut excessive = spec();
    excessive.files.resize(33, excessive.files[3].clone());
    assert!(excessive.validate().is_err());
    let mut excessive = spec();
    for file in &mut excessive.files {
        file.bytes = 1;
    }
    for file in &mut excessive.files[..4] {
        file.bytes = 1024 * 1024 * 1024;
    }
    assert!(excessive.validate().is_err());
    let remaining_file_bytes = (excessive.files.len() - 4) as u64;
    excessive.files[3].bytes -= remaining_file_bytes;
    assert!(
        excessive.validate().is_ok(),
        "exactly 4 GiB is within the manifest budget"
    );
}

#[test]
fn debug_does_not_disclose_raw_hashes_or_unvalidated_paths() {
    let original = spec();
    let debug = format!("{original:?}");
    assert!(!debug.contains(&original.files[0].sha256));
    let mut invalid = original;
    invalid.files[0].filename = "/private/bootstrap/source.so".into();
    assert!(!format!("{invalid:?}").contains("/private/bootstrap"));
}

// Legacy manifest parsing remains available for historical receipts. New native
// library bundle copies are intentionally unavailable in all product paths.
#[test]
fn direct_legacy_bundle_copy_is_rejected_before_opening_paths() {
    use rz_eval::error::FailureKind;
    use std::time::{SystemTime, UNIX_EPOCH};
    let base = std::env::temp_dir().join(format!(
        "rovezero-disabled-direct-bundle-{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let source = base.join("not-created-source");
    let output = base.join("not-created-output");
    assert!(!base.exists());
    let error = RuntimeLibraryPin::copy_cuda_bundle(&source, &output, &spec()).unwrap_err();
    assert_eq!(error.kind, FailureKind::BackendUnavailable);
    assert!(
        !base.exists(),
        "rejected copy must not create output or read inputs"
    );
}

#[cfg(target_os = "linux")]
#[test]
fn legacy_bundle_cache_request_is_rejected_without_creating_a_bundle() {
    use rz_eval::{error::FailureKind, runtime_pin::RuntimeCache};
    use std::time::{SystemTime, UNIX_EPOCH};
    struct EmptyCache(std::path::PathBuf);
    impl Drop for EmptyCache {
        fn drop(&mut self) {
            // Only remove our own empty directory; unexpected contents survive.
            let _ = std::fs::remove_dir(&self.0);
        }
    }
    let base = std::env::temp_dir().canonicalize().unwrap().join(format!(
        "rovezero-disabled-bundle-cache-{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    assert!(!base.exists());
    let _cleanup = EmptyCache(base.clone());
    let cache = RuntimeCache::open(&base).unwrap();
    let source = base.join("not-created-source");
    let error = cache.cuda_bundle(&source, &spec()).unwrap_err();
    assert_eq!(error.kind, FailureKind::BackendUnavailable);
    let mut checkpoint_called = false;
    let error = cache
        .prepare_cuda_artifact_tree(&source, &spec(), &mut |_| {
            checkpoint_called = true;
            Ok(())
        })
        .unwrap_err();
    assert_eq!(error.kind, FailureKind::BackendUnavailable);
    assert!(
        !checkpoint_called,
        "rejected preparation must not consume credit"
    );
    assert!(!source.exists());
    assert_eq!(std::fs::read_dir(&base).unwrap().count(), 0);
    drop(cache);
}
