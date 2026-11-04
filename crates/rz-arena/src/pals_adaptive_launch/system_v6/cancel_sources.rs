//! Source-byte observation for the separate V6 cancel implementation.
//!
//! This is not a running executable identity, permission, ACK, or physical
//! completion evidence. Historical V6 component definitions remain intact.

use sha2::{Digest as _, Sha256};

const DOMAIN: &str = "rz-pals-v6-cancel-source-snapshot/1";

/// The exact reported scope is the labels below, not transitive code or the
/// independently built proxy executable.
pub(crate) struct CancelSourceObservation {
    pub(crate) domain: &'static str,
    pub(crate) labels: Vec<&'static str>,
    pub(crate) digest: String,
}

fn sources() -> &'static [(&'static str, &'static [u8])] {
    &[
        ("cancel_sources", include_bytes!("cancel_sources.rs")),
        ("v6_boundary", include_bytes!("../system_v6.rs")),
        ("v6_pair", include_bytes!("pair.rs")),
        ("v6_pair_native", include_bytes!("pair_native.rs")),
        ("v6_proxy_inputs", include_bytes!("proxy_inputs.rs")),
        (
            "v6_endpoint_snapshot",
            include_bytes!("endpoint_snapshot.rs"),
        ),
        ("v6_deadline_control", include_bytes!("deadline_control.rs")),
        ("v6_execution_start", include_bytes!("execution_start.rs")),
        ("arena_process", include_bytes!("../../process.rs")),
        (
            "arena_native_launch",
            include_bytes!("../../native_launch.rs"),
        ),
        (
            "arena_native_runner",
            include_bytes!("../../native_runner.rs"),
        ),
        (
            "cancel_wire",
            include_bytes!("../../../../rz-contracts/src/pals/cancel_control_v1.rs"),
        ),
        (
            "deadline_wire",
            include_bytes!("../../../../rz-contracts/src/pals/deadline_grant_v1.rs"),
        ),
        (
            "uci_cancel_receiver",
            include_bytes!("../../../../rz-uci/src/pals_system_proxy/cancel_control_v1.rs"),
        ),
        (
            "uci_proxy_cli",
            include_bytes!("../../../../rz-uci/src/pals_system_proxy/cli_v6.rs"),
        ),
    ]
}

pub(crate) fn compiled_cancel_source_observation() -> CancelSourceObservation {
    CancelSourceObservation {
        domain: DOMAIN,
        labels: sources().iter().map(|(name, _)| *name).collect(),
        digest: digest(sources()),
    }
}

fn digest(files: &[(&str, &[u8])]) -> String {
    let mut hash = Sha256::new();
    hash.update(DOMAIN.as_bytes());
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
    fn source_snapshot_frames_both_labels_and_bytes() {
        let original = [("one", b"abc".as_slice()), ("two", b"def".as_slice())];
        assert_ne!(digest(&original), digest(&[("one", b"abcdef".as_slice())]));
        assert_ne!(
            digest(&original),
            digest(&[("one", b"abc".as_slice()), ("other", b"def".as_slice())])
        );
        let labels: Vec<_> = sources().iter().map(|(name, _)| name).collect();
        let unique: std::collections::BTreeSet<_> = labels.iter().collect();
        assert_eq!(labels.len(), unique.len());
    }
}
