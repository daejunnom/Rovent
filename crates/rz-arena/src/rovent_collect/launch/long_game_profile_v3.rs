//! Opt-in, read-only preparation of a separately pinned long-game profile.
//!
//! Preparing this input does not apply it to the `/3` collector. In
//! particular it cannot authorize a slot release, a long-game run, or a
//! change to an existing launch plan or receipt.

use super::pc_lock_v3::OriginalPinnedFile;
use super::{ArenaError, Dir, RoventPublicationFile, invalid, io};
use crate::rovent_collect::long_game_profile_v3::LongGameProfileV3;
use sha2::{Digest, Sha256};
use std::os::unix::fs::MetadataExt;

/// Same input-size guard as the independent Collection launch declaration.
const PROFILE_MAX_BYTES: u64 = 256 << 10;

/// Retains both the original input-directory capability and profile file.
/// File descriptor retention does not prevent another writer from modifying
/// that file; private input ownership is a separate runtime requirement.
pub(crate) struct PreparedLongGameProfileV3 {
    original_directory: std::fs::File,
    original_profile: OriginalPinnedFile,
    profile: LongGameProfileV3,
    canonical: Vec<u8>,
    canonical_sha256: [u8; 32],
}

impl PreparedLongGameProfileV3 {
    /// Open only an explicitly supplied pin under the original inputs
    /// directory. The shared reader validates path components, regular file,
    /// size and raw SHA while retaining the original file handle.
    pub(crate) fn open(directory: &Dir, pin: &RoventPublicationFile) -> Result<Self, ArenaError> {
        let original_directory = directory.try_clone().map_err(io)?.into_std_file();
        let original_profile = OriginalPinnedFile::read(directory, pin, PROFILE_MAX_BYTES)?;
        let profile: LongGameProfileV3 = serde_json::from_slice(original_profile.bytes())
            .map_err(|error| invalid(format!("long-game profile JSON: {error}")))?;
        let canonical = profile.canonical_bytes().map_err(|error| {
            invalid(format!("long-game profile canonical validation: {error:?}"))
        })?;
        let canonical_sha256 = Sha256::digest(&canonical).into();
        Ok(Self {
            original_directory,
            original_profile,
            profile,
            canonical,
            canonical_sha256,
        })
    }

    pub(crate) fn profile(&self) -> &LongGameProfileV3 {
        &self.profile
    }

    pub(crate) fn original_bytes(&self) -> &[u8] {
        self.original_profile.bytes()
    }

    pub(crate) fn original_pin(&self) -> &RoventPublicationFile {
        self.original_profile.pin()
    }

    pub(crate) fn canonical_bytes(&self) -> &[u8] {
        &self.canonical
    }

    /// This semantic digest is distinct from the raw-file pin.
    pub(crate) fn canonical_sha256(&self) -> [u8; 32] {
        self.canonical_sha256
    }

    /// Compare directory object identity at the resource boundary. This is
    /// not a check that an external writer left the profile file unchanged.
    pub(crate) fn matches_original_directory(&self, directory: &Dir) -> Result<bool, ArenaError> {
        let original = self.original_directory.metadata().map_err(io)?;
        let current = directory
            .try_clone()
            .map_err(io)?
            .into_std_file()
            .metadata()
            .map_err(io)?;
        Ok(original.is_dir()
            && current.is_dir()
            && original.dev() == current.dev()
            && original.ino() == current.ino())
    }
}
