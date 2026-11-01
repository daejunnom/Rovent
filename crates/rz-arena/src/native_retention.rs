//! Retire only the launch owner's verified private copies after evidence is saved.
//! Shared runtimes, source assets, executable pins and JSON evidence are retained.
use crate::{
    ArenaError, NATIVE_PAIR_METADATA_CAP,
    native_launch::linux::{Snapshot, input_parent, verify_input_directories},
};
use cap_fs_ext::{DirExt, FollowSymlinks, OpenOptionsFollowExt};
use cap_std::fs::{Dir, OpenOptions, OpenOptionsExt};
use serde_json::json;
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{File, Permissions},
    io::Write,
    os::unix::fs::{MetadataExt, PermissionsExt},
    path::Path,
};

const JOURNAL: &str = "snapshot-retention.v1.jsonl";

#[cfg(test)]
pub(crate) fn eligible(cleanup_verified: bool, provider_ok: bool, sessions: usize) -> bool {
    eligible_expected(4, cleanup_verified, provider_ok, sessions)
}

pub(crate) fn eligible_expected(
    expected: usize,
    cleanup_verified: bool,
    provider_ok: bool,
    sessions: usize,
) -> bool {
    matches!(expected, 2 | 4) && cleanup_verified && provider_ok && sessions == expected
}

fn invalid(message: &str) -> ArenaError {
    ArenaError::Integrity(message.into())
}
fn readable_directory(directory: &Dir) -> Result<File, ArenaError> {
    let mut options = OpenOptions::new();
    options
        .read(true)
        .follow(FollowSymlinks::No)
        .custom_flags(libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_NONBLOCK);
    Ok(directory
        .open_with(".", &options)
        .map_err(|_| invalid("cannot pin retention directory"))?
        .into_std())
}

/// Strict admission for legacy profiles whose registered storage contract
/// includes an explicit root budget. This is not the advisory total-storage
/// warning used for ordinary native launches.
///
/// Admission counts lengths without opening file contents or following links.
/// Existing evidence is never evicted to make room. Concurrent independent
/// output-root writers need external serialization; this is not a disk quota.
pub(crate) fn admit_bounded(root: &Dir, reservation: u64, limit: u64) -> Result<(), ArenaError> {
    fn size(
        directory: &Dir,
        depth: usize,
        entries: &mut usize,
        limit: u64,
    ) -> Result<u64, ArenaError> {
        if depth > 64 {
            return Err(invalid("retention root depth exceeds bound"));
        }
        let mut bytes = 0_u64;
        for entry in directory
            .entries()
            .map_err(|_| invalid("cannot inspect retention root"))?
        {
            *entries += 1;
            if *entries > 100_000 {
                return Err(invalid("retention root entry count exceeds bound"));
            }
            let entry = entry.map_err(|_| invalid("cannot inspect retention entry"))?;
            let name = entry.file_name();
            let kind = entry
                .file_type()
                .map_err(|_| invalid("cannot inspect retention kind"))?;
            let amount = if kind.is_dir() {
                let child = directory
                    .open_dir_nofollow(&name)
                    .map_err(|_| invalid("retention directory changed"))?;
                size(&child, depth + 1, entries, limit)?
            } else if kind.is_file() {
                directory
                    .symlink_metadata(&name)
                    .map_err(|_| invalid("retention file changed"))?
                    .len()
            } else {
                return Err(invalid(
                    "retention root contains a link or unsupported entry",
                ));
            };
            bytes = bytes
                .checked_add(amount)
                .ok_or_else(|| invalid("retention size overflow"))?;
            if bytes > limit {
                return Err(ArenaError::Budget(format!(
                    "native output root exceeds {limit} bytes; preserve evidence and perform owner-aware maintenance"
                )));
            }
        }
        Ok(bytes)
    }
    if size(root, 0, &mut 0, limit)?
        .checked_add(reservation)
        .is_none_or(|n| n > limit)
    {
        return Err(ArenaError::Budget(format!(
            "native output root plus reserved attempt exceeds {limit} bytes"
        )));
    }
    Ok(())
}

fn write_event(journal: &mut File, event: &serde_json::Value) -> Result<(), ArenaError> {
    let bytes = serde_json::to_vec(event)
        .map_err(|_| invalid("cannot encode snapshot retirement event"))?;
    let length = journal
        .metadata()
        .map_err(|_| invalid("cannot measure retirement journal"))?
        .len();
    if length
        .checked_add(bytes.len() as u64)
        .and_then(|n| n.checked_add(1))
        .is_none_or(|n| n > NATIVE_PAIR_METADATA_CAP)
    {
        return Err(ArenaError::Budget(
            "snapshot retirement journal exceeds 64 KiB; remaining copies preserved".into(),
        ));
    }
    journal
        .write_all(&bytes)
        .and_then(|()| journal.write_all(b"\n"))
        .and_then(|()| journal.sync_all())
        .map_err(|_| ArenaError::Io("cannot persist snapshot retirement journal".into()))
}

pub(crate) fn retire(snapshot: &mut Snapshot) -> Result<(), ArenaError> {
    let runner = snapshot.pins[snapshot.runner_index].artifact.clone();
    let removed = retire_pins(
        &snapshot.directory,
        &snapshot.input_directory,
        &snapshot.input_subdirectories,
        &snapshot.path,
        &snapshot.pins,
    )?;
    // Unlink alone does not release disk while these read handles remain open.
    snapshot.pins.retain(|pin| !removed.contains(&pin.path));
    snapshot.runner_index = snapshot
        .pins
        .iter()
        .position(|p| p.artifact == runner)
        .ok_or_else(|| invalid("retirement removed the executable runner"))?;
    Ok(())
}

fn retire_pins(
    output: &Dir,
    inputs: &Dir,
    subdirectories: &BTreeMap<std::path::PathBuf, Dir>,
    root: &Path,
    pins: &[crate::native_launch::linux::InputPin],
) -> Result<BTreeSet<std::path::PathBuf>, ArenaError> {
    // All ancestors, including folders holding only preserved evidence, must
    // still name the exact no-follow capabilities before any chmod or unlink.
    verify_input_directories(output, inputs, subdirectories)?;
    let mut candidates = Vec::new();
    for pin in pins {
        let held = pin
            .file
            .metadata()
            .map_err(|_| invalid("retention pin metadata missing"))?;
        if held.mode() & 0o111 != 0
            || Path::new(&pin.artifact.path)
                .extension()
                .and_then(|s| s.to_str())
                .is_some_and(|s| s.eq_ignore_ascii_case("json"))
        {
            continue;
        }
        let relative = pin
            .path
            .strip_prefix(root)
            .map_err(|_| invalid("retention pin escaped root"))?;
        let (directory, name) = input_parent(inputs, subdirectories, relative)?;
        let named = directory
            .symlink_metadata(name)
            .map_err(|_| invalid("retention named pin absent"))?;
        if !named.is_file()
            || cap_fs_ext::MetadataExt::nlink(&named) != 1
            || (held.dev(), held.ino(), held.len())
                != (
                    cap_fs_ext::MetadataExt::dev(&named),
                    cap_fs_ext::MetadataExt::ino(&named),
                    named.len(),
                )
            || held.len() != pin.artifact.bytes
        {
            return Err(invalid(
                "retention inode, kind or size changed; all copies preserved",
            ));
        }
        candidates.push((pin, directory, name.to_os_string(), relative.to_path_buf()));
    }
    let mut options = OpenOptions::new();
    options
        .write(true)
        .create_new(true)
        .follow(FollowSymlinks::No)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
        .mode(0o600);
    let mut journal = output
        .open_with(JOURNAL, &options)
        .map_err(|_| invalid("retirement journal exists or cannot be created; copies preserved"))?
        .into_std();
    write_event(
        &mut journal,
        &json!({
            "schema": "rovezero.snapshot-retention.v1", "event": "planned",
            "eligibility": "saved_receipt_verified_process_cleanup_postcheck_and_declared_provider_drains",
            "copies": candidates.iter().map(|(pin, _, _, relative)| json!({
                "path": relative, "artifact": pin.artifact
            })).collect::<Vec<_>>(),
            "preserved": "original_assets_shared_runtime_executables_json_PGN_logs_receipts"
        }),
    )?;
    readable_directory(output)?
        .sync_all()
        .map_err(|_| ArenaError::Io("cannot sync retirement journal directory".into()))?;
    let directories: Vec<_> = std::iter::once(inputs)
        .chain(subdirectories.values())
        .map(readable_directory)
        .collect::<Result<_, _>>()?;
    let result = (|| {
        for directory in &directories {
            directory
                .set_permissions(Permissions::from_mode(0o700))
                .map_err(|_| ArenaError::Io("cannot make owned inputs removable".into()))?;
        }
        let mut removed = BTreeSet::new();
        for (pin, directory, name, relative) in candidates {
            directory
                .remove_file(&name)
                .map_err(|_| ArenaError::Io("cannot unlink verified private input copy".into()))?;
            removed.insert(pin.path.clone());
            write_event(
                &mut journal,
                &json!({"event": "unlinked", "path": relative, "bytes": pin.artifact.bytes}),
            )?;
        }
        for directory in &directories {
            directory
                .sync_all()
                .map_err(|_| ArenaError::Io("cannot sync retired input directory".into()))?;
        }
        write_event(
            &mut journal,
            &json!({"event": "complete", "unlinked_files": removed.len()}),
        )?;
        Ok(removed)
    })();
    let mut restore = Ok(());
    for directory in &directories {
        if directory
            .set_permissions(Permissions::from_mode(0o555))
            .is_err()
        {
            restore = Err(ArenaError::Io(
                "cannot restore readonly input directory".into(),
            ));
        }
    }
    restore?;
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::native_launch::linux::InputPin;
    use rz_experiments::ArtifactRef;
    use sha2::{Digest, Sha256};
    use std::{
        fs,
        path::PathBuf,
        sync::atomic::{AtomicU64, Ordering},
    };

    static NEXT: AtomicU64 = AtomicU64::new(0);
    struct Tree(PathBuf);
    impl Tree {
        fn new() -> Self {
            let path = std::env::temp_dir().join(format!(
                "rovezero-retention-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir(&path).unwrap();
            Self(path)
        }
        fn dir(&self) -> Dir {
            Dir::open_ambient_dir(&self.0, cap_std::ambient_authority()).unwrap()
        }
    }
    impl Drop for Tree {
        fn drop(&mut self) {
            for name in ["inputs", "inputs/cuda-bundle", "old-inputs"] {
                let path = self.0.join(name);
                if path.is_dir() {
                    let _ = fs::set_permissions(path, Permissions::from_mode(0o700));
                }
            }
            let _ = fs::remove_dir_all(&self.0);
        }
    }
    fn input(tree: &Tree, name: &str, executable: bool) -> InputPin {
        let path = tree.0.join("inputs").join(name);
        let bytes = b"synthetic private copy";
        fs::write(&path, bytes).unwrap();
        fs::set_permissions(
            &path,
            Permissions::from_mode(if executable { 0o555 } else { 0o444 }),
        )
        .unwrap();
        InputPin {
            artifact: ArtifactRef {
                path: name.into(),
                bytes: bytes.len() as u64,
                sha256: format!("{:x}", Sha256::digest(bytes)),
                source: "synthetic retirement fixture".into(),
                license: "MIT fixture".into(),
            },
            file: File::open(&path).unwrap(),
            path,
        }
    }
    #[test]
    fn retirement_requires_all_cleanup_and_provider_gates() {
        assert!(eligible(true, true, 4));
        for (gone, audited, count) in [
            (false, true, 4),
            (true, false, 4),
            (true, true, 0),
            (true, true, 3),
        ] {
            assert!(!eligible(gone, audited, count));
        }
    }
    #[test]
    fn retires_only_private_copies_and_keeps_evidence_source_and_executable() {
        let tree = Tree::new();
        let output = tree.dir();
        output.create_dir("inputs").unwrap();
        let inputs = output.open_dir_nofollow("inputs").unwrap();
        let original = tree.0.join("source-model.bin");
        fs::write(&original, b"original unchanged").unwrap();
        fs::write(tree.0.join("match.pgn"), b"saved PGN").unwrap();
        fs::write(tree.0.join("native-pair-receipt.v1.json"), b"saved receipt").unwrap();
        let pins = vec![
            input(&tree, "model.onnx", false),
            input(&tree, "libonnxruntime.so", false),
            input(&tree, "runner", true),
            input(&tree, "export.json", false),
            input(&tree, "extra.JSON", false),
        ];
        fs::set_permissions(tree.0.join("inputs"), Permissions::from_mode(0o555)).unwrap();
        let removed = retire_pins(&output, &inputs, &BTreeMap::new(), &tree.0, &pins).unwrap();
        assert_eq!(removed.len(), 2);
        assert!(!pins[0].path.exists());
        assert!(!pins[1].path.exists());
        assert!(pins[2].path.exists() && pins[3].path.exists() && pins[4].path.exists());
        assert_eq!(fs::read(original).unwrap(), b"original unchanged");
        assert_eq!(fs::read(tree.0.join("match.pgn")).unwrap(), b"saved PGN");
        let events = fs::read_to_string(tree.0.join(JOURNAL)).unwrap();
        assert!(
            events
                .lines()
                .last()
                .unwrap()
                .contains("\"event\":\"complete\"")
        );
        assert_eq!(
            fs::metadata(tree.0.join("inputs")).unwrap().mode() & 0o777,
            0o555
        );
    }
    #[test]
    fn nested_export_retirement_preserves_descriptor_and_rejects_replaced_evidence_directory() {
        for replace in [false, true] {
            let tree = Tree::new();
            let output = tree.dir();
            output.create_dir("inputs").unwrap();
            let inputs = output.open_dir_nofollow("inputs").unwrap();
            inputs.create_dir("pals-export").unwrap();
            inputs.create_dir("pals-descriptor").unwrap();
            let subdirectories: BTreeMap<_, _> = ["pals-export", "pals-descriptor"]
                .into_iter()
                .map(|name| (PathBuf::from(name), inputs.open_dir_nofollow(name).unwrap()))
                .collect();
            let pins = vec![
                input(&tree, "pals-export/public_memory.onnx", false),
                input(&tree, "pals-export/export.json", false),
                input(&tree, "pals-descriptor/evidence.json", false),
                input(&tree, "runner", true),
            ];
            if replace {
                // A directory containing only preserved JSON still participates
                // in ownership validation before any model copy is unlinked.
                fs::rename(
                    tree.0.join("inputs/pals-descriptor"),
                    tree.0.join("old-inputs"),
                )
                .unwrap();
                fs::create_dir(tree.0.join("inputs/pals-descriptor")).unwrap();
                assert!(retire_pins(&output, &inputs, &subdirectories, &tree.0, &pins).is_err());
                assert!(pins[0].path.exists());
                assert!(!tree.0.join(JOURNAL).exists());
            } else {
                let removed =
                    retire_pins(&output, &inputs, &subdirectories, &tree.0, &pins).unwrap();
                assert_eq!(removed.len(), 1);
                assert!(!pins[0].path.exists());
                assert!(pins[1].path.exists() && pins[2].path.exists() && pins[3].path.exists());
            }
            // Restore only this fixture's directory modes for its Drop cleanup.
            for name in ["pals-export", "pals-descriptor"] {
                fs::set_permissions(
                    tree.0.join("inputs").join(name),
                    Permissions::from_mode(0o700),
                )
                .unwrap();
            }
        }
    }
    #[test]
    fn replaced_pin_or_existing_journal_preserves_every_copy() {
        for replace in [true, false] {
            let tree = Tree::new();
            let output = tree.dir();
            output.create_dir("inputs").unwrap();
            let inputs = output.open_dir_nofollow("inputs").unwrap();
            let pins = vec![
                input(&tree, "first.onnx", false),
                input(&tree, "second.onnx", false),
            ];
            if replace {
                fs::rename(&pins[1].path, tree.0.join("old-copy")).unwrap();
                fs::write(&pins[1].path, b"replacement").unwrap();
            } else {
                fs::write(tree.0.join(JOURNAL), b"prior evidence").unwrap();
            }
            assert!(retire_pins(&output, &inputs, &BTreeMap::new(), &tree.0, &pins).is_err());
            assert!(pins[0].path.exists() && pins[1].path.exists());
        }
    }
    #[test]
    fn journal_budget_rejects_without_appending_partial_event() {
        let tree = Tree::new();
        let mut journal = File::create(tree.0.join(JOURNAL)).unwrap();
        journal.write_all(b"prior\n").unwrap();
        let event = json!({"oversize": "x".repeat(NATIVE_PAIR_METADATA_CAP as usize)});
        assert!(write_event(&mut journal, &event).is_err());
        assert_eq!(fs::read(tree.0.join(JOURNAL)).unwrap(), b"prior\n");
    }
    #[test]
    fn admission_rejects_oversize_history_links_and_reservation_without_deleting() {
        let tree = Tree::new();
        let root = tree.dir();
        let file = File::create(tree.0.join("historical-evidence")).unwrap();
        file.set_len(1024).unwrap();
        assert!(admit_bounded(&root, 1, 1024).is_err());
        assert!(admit_bounded(&root, 0, 1023).is_err());
        assert!(tree.0.join("historical-evidence").exists());
        file.set_len(0).unwrap();
        std::os::unix::fs::symlink("/unopened-external-target", tree.0.join("link")).unwrap();
        assert!(admit_bounded(&root, 0, 1024).is_err());
        fs::remove_file(tree.0.join("link")).unwrap();
        assert!(admit_bounded(&root, 1024, 1024).is_ok());
    }
}
