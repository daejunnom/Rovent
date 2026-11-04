//! Compile-time source snapshot for the separate V6 deadline integration.
//!
//! This identifies exactly the source bytes listed below, as embedded by the
//! arena build. In particular, embedding UCI source bytes does not attest an
//! independently built UCI executable or prove that either side executes the
//! included code. It is not execution, process-ownership, or GPU evidence.
//! The historical V6 source components and canonical lock are unchanged.

use sha2::{Digest as _, Sha256};

pub const PALS_V6_DEADLINE_SOURCE_SNAPSHOT_COMPONENT: &str = "pals_v6_deadline_source_snapshot";

const DOMAIN: &[u8] = b"rz-pals-v6-deadline-source-snapshot/1";

/// Names and bytes included in this snapshot, including the snapshot's own
/// definition. They are not a list of verified linked or executed modules.
fn sources() -> &'static [(&'static str, &'static [u8])] {
    &[
        (
            "deadline_source_snapshot",
            include_bytes!("deadline_sources.rs"),
        ),
        ("arena_deadline_grant", include_bytes!("deadline_grant.rs")),
        (
            "arena_execution_start",
            include_bytes!("execution_start.rs"),
        ),
        ("arena_v6_boundary", include_bytes!("../system_v6.rs")),
        ("arena_pair_adapter", include_bytes!("pair_native.rs")),
        ("arena_proxy_inputs", include_bytes!("proxy_inputs.rs")),
        (
            "arena_native_launch",
            include_bytes!("../../native_launch.rs"),
        ),
        (
            "arena_native_runner",
            include_bytes!("../../native_runner.rs"),
        ),
        (
            "deadline_wire",
            include_bytes!("../../../../rz-contracts/src/pals/deadline_grant_v1.rs"),
        ),
        (
            "uci_proxy_cli_source",
            include_bytes!("../../../../rz-uci/src/pals_system_proxy/cli_v6.rs"),
        ),
        (
            "uci_proxy_runner_source",
            include_bytes!("../../../../rz-uci/src/pals_system_proxy/runner_v6.rs"),
        ),
        (
            "uci_deadline_grant_source",
            include_bytes!("../../../../rz-uci/src/pals_system_proxy/deadline_grant.rs"),
        ),
    ]
}

/// Stable source labels for receipts or inspection. The precise scope is the
/// contents of `sources`, not all code transitively reachable at runtime.
pub fn pals_v6_deadline_source_snapshot_labels() -> Vec<&'static str> {
    sources().iter().map(|(name, _)| *name).collect()
}

/// SHA-256 of domain-separated, length-framed labels and compile-time bytes.
/// Do not compare this with the older V6 implementation digest.
pub fn compiled_pals_v6_deadline_source_snapshot_digest() -> String {
    digest(sources())
}

fn digest(files: &[(&str, &[u8])]) -> String {
    let mut hash = Sha256::new();
    hash.update(DOMAIN);
    for (name, bytes) in files {
        hash.update((name.len() as u64).to_le_bytes());
        hash.update(name.as_bytes());
        hash.update((bytes.len() as u64).to_le_bytes());
        hash.update(bytes);
    }
    format!("{:x}", hash.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snapshot_identity_changes_for_source_or_label_changes() {
        let original = [("one", b"abc".as_slice()), ("two", b"def".as_slice())];
        let changed_bytes = [("one", b"abd".as_slice()), ("two", b"def".as_slice())];
        let changed_label = [("one", b"abc".as_slice()), ("other", b"def".as_slice())];
        assert_ne!(digest(&original), digest(&changed_bytes));
        assert_ne!(digest(&original), digest(&changed_label));
        assert_ne!(digest(&original), digest(&[("one", b"abcdef".as_slice())]));
    }

    #[test]
    fn labels_are_unique_and_digest_is_sha256() {
        let labels = pals_v6_deadline_source_snapshot_labels();
        let unique: std::collections::BTreeSet<_> = labels.iter().collect();
        assert_eq!(labels.len(), unique.len());
        let digest = compiled_pals_v6_deadline_source_snapshot_digest();
        assert_eq!(digest.len(), 64);
        assert!(digest.bytes().all(|byte| byte.is_ascii_hexdigit()));
    }
}
