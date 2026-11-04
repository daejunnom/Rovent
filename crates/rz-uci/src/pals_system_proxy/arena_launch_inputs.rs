//! Explicit arena-to-proxy socket arguments. Paths are declarations, not
//! authentication, game identity, process ownership, or execution evidence.
//! The child-only PALS parent socket remains a separate input.

use super::controlled_path;
use std::{
    io,
    path::{Path, PathBuf},
};

pub(super) const ARENA_GRANT_SOCKET_ARG: &str = "--pals-arena-grant-socket=";
pub(super) const ARENA_CANCEL_SOCKET_ARG: &str = "--pals-arena-cancel-socket=";

fn invalid(detail: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, detail)
}

#[derive(Default)]
pub(super) struct ArenaSocketArgs {
    grant: Option<PathBuf>,
    cancel: Option<PathBuf>,
}

/// Declared absolute paths only. The actual connection must use capabilities
/// from each process's own pinned directory, verify the same original arena
/// and role, and preserve the sockets through the original supervisor.
pub(super) struct ArenaSocketPaths {
    pub grant: PathBuf,
    pub cancel: PathBuf,
}

impl ArenaSocketArgs {
    /// Returns false for arguments outside this namespace so the unchanged
    /// system-startup parser still handles them. Duplicate or invalid socket
    /// arguments are rejected before mutating the selected path.
    pub(super) fn consume(&mut self, argument: &str) -> io::Result<bool> {
        let (slot, text) = if let Some(value) = argument.strip_prefix(ARENA_GRANT_SOCKET_ARG) {
            (&mut self.grant, value)
        } else if let Some(value) = argument.strip_prefix(ARENA_CANCEL_SOCKET_ARG) {
            (&mut self.cancel, value)
        } else {
            return Ok(false);
        };
        let path = PathBuf::from(text);
        if slot.is_some() || !controlled_path(&path) {
            return Err(invalid("invalid or duplicate arena socket"));
        }
        *slot = Some(path);
        Ok(true)
    }

    /// Lexical path separation prevents accidentally using the child-only
    /// channel for arena control. This check does not establish inode identity
    /// or role/listener ownership; those require the original capabilities.
    pub(super) fn finish(self, child_parent: &Path) -> io::Result<ArenaSocketPaths> {
        let grant = self
            .grant
            .ok_or_else(|| invalid("arena grant socket required"))?;
        let cancel = self
            .cancel
            .ok_or_else(|| invalid("arena cancel socket required"))?;
        if !controlled_path(child_parent)
            || grant == cancel
            || grant.as_path() == child_parent
            || cancel.as_path() == child_parent
        {
            return Err(invalid("arena and child control paths must be distinct"));
        }
        Ok(ArenaSocketPaths { grant, cancel })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn parses_distinct_channels_without_touching_child_argument() {
        let mut args = ArenaSocketArgs::default();
        assert!(
            args.consume("--pals-arena-grant-socket=/owned/grant")
                .unwrap()
        );
        assert!(
            args.consume("--pals-arena-cancel-socket=/owned/cancel")
                .unwrap()
        );
        assert!(
            !args
                .consume("--pals-system-parent-socket=/owned/child")
                .unwrap()
        );
        let paths = args.finish(Path::new("/owned/child")).unwrap();
        assert_eq!(paths.grant, Path::new("/owned/grant"));
        assert_eq!(paths.cancel, Path::new("/owned/cancel"));
    }
    #[test]
    fn invalid_or_duplicate_paths_do_not_replace_original_selection() {
        let mut args = ArenaSocketArgs::default();
        assert!(
            args.consume("--pals-arena-grant-socket=/owned/grant")
                .unwrap()
        );
        assert!(
            args.consume("--pals-arena-grant-socket=/owned/other")
                .is_err()
        );
        assert!(args.consume("--pals-arena-cancel-socket=relative").is_err());
        assert!(
            args.consume("--pals-arena-cancel-socket=/owned/cancel")
                .unwrap()
        );
        let paths = args.finish(Path::new("/owned/child")).unwrap();
        assert_eq!(paths.grant, Path::new("/owned/grant"));
    }
    #[test]
    fn child_and_arena_paths_cannot_be_the_same() {
        let mut args = ArenaSocketArgs::default();
        args.consume("--pals-arena-grant-socket=/owned/grant")
            .unwrap();
        args.consume("--pals-arena-cancel-socket=/owned/child")
            .unwrap();
        assert!(args.finish(Path::new("/owned/child")).is_err());
    }
}
