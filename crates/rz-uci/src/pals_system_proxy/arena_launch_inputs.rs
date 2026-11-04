//! Explicit arena-to-proxy socket arguments. Paths are declarations, not
//! authentication, game identity, process ownership, or execution evidence.
//! The child-only PALS parent socket remains a separate input.

use super::controlled_path;
use std::{io, path::PathBuf};

pub(super) const ARENA_GRANT_SOCKET_ARG: &str = "--pals-arena-grant-socket=";
pub(super) const ARENA_CANCEL_SOCKET_ARG: &str = "--pals-arena-cancel-socket=";

fn invalid(detail: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, detail)
}

/// Linux abstract addresses are kernel namespace names, not filesystem paths.
/// Their identity is no substitute for peer or original listener authority.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) enum ArenaSocketAddress {
    Path(PathBuf),
    Abstract(String),
}

impl ArenaSocketAddress {
    pub(super) fn parse(text: &str) -> io::Result<Self> {
        if let Some(name) = text.strip_prefix("abstract:") {
            // Linux sockaddr_un has 108 bytes; the leading NUL uses one.
            if name.is_empty()
                || name.len() > 107
                || !name.is_ascii()
                || name.chars().any(char::is_control)
            {
                return Err(invalid("invalid abstract arena socket name"));
            }
            return Ok(Self::Abstract(name.to_owned()));
        }
        let path = PathBuf::from(text);
        if !controlled_path(&path) {
            return Err(invalid("invalid arena socket path"));
        }
        Ok(Self::Path(path))
    }
}

#[derive(Default)]
pub(super) struct ArenaSocketArgs {
    grant: Option<ArenaSocketAddress>,
    cancel: Option<ArenaSocketAddress>,
}

/// Declared names only. Pathname transport uses a directory pin owned by each
/// process. Abstract transport retains its original listener and stream FDs.
pub(super) struct ArenaSocketPaths {
    pub grant: ArenaSocketAddress,
    pub cancel: ArenaSocketAddress,
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
        if slot.is_some() {
            return Err(invalid("duplicate arena socket"));
        }
        let address = ArenaSocketAddress::parse(text)?;
        *slot = Some(address);
        Ok(true)
    }

    /// Address separation prevents accidentally sharing the child-only
    /// channel. This does not establish inode or role/listener ownership.
    pub(super) fn finish(self, child_parent: &ArenaSocketAddress) -> io::Result<ArenaSocketPaths> {
        let grant = self
            .grant
            .ok_or_else(|| invalid("arena grant socket required"))?;
        let cancel = self
            .cancel
            .ok_or_else(|| invalid("arena cancel socket required"))?;
        if grant == cancel || &grant == child_parent || &cancel == child_parent {
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
        let paths = args
            .finish(&ArenaSocketAddress::Path(PathBuf::from("/owned/child")))
            .unwrap();
        assert_eq!(
            paths.grant,
            ArenaSocketAddress::Path(PathBuf::from("/owned/grant"))
        );
        assert_eq!(
            paths.cancel,
            ArenaSocketAddress::Path(PathBuf::from("/owned/cancel"))
        );
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
        let paths = args
            .finish(&ArenaSocketAddress::Path(PathBuf::from("/owned/child")))
            .unwrap();
        assert_eq!(
            paths.grant,
            ArenaSocketAddress::Path(PathBuf::from("/owned/grant"))
        );
    }
    #[test]
    fn abstract_listener_fixture_preserves_socket_and_peer_credentials() {
        use nix::sys::socket::{getsockopt, sockopt::PeerCredentials};
        use std::{
            io::{Read, Write},
            os::linux::net::SocketAddrExt,
            os::unix::net::{SocketAddr, UnixListener, UnixStream},
            sync::atomic::{AtomicU64, Ordering},
        };
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let name = format!(
            "rovezero-test-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        );
        let address = SocketAddr::from_abstract_name(name.as_bytes()).unwrap();
        let listener = UnixListener::bind_addr(&address).unwrap();
        let mut client = UnixStream::connect_addr(&address).unwrap();
        let (mut peer, _) = listener.accept().unwrap();
        let credentials = getsockopt(&peer, PeerCredentials).unwrap();
        assert_eq!(credentials.pid() as u32, std::process::id());
        assert_eq!(credentials.uid(), nix::unistd::geteuid().as_raw());
        client.write_all(b"x").unwrap();
        let mut message = [0];
        peer.read_exact(&mut message).unwrap();
        assert_eq!(message, *b"x");
    }
    #[test]
    fn abstract_names_are_bounded_and_distinct_from_child_and_each_other() {
        let mut args = ArenaSocketArgs::default();
        args.consume("--pals-arena-grant-socket=abstract:grant")
            .unwrap();
        args.consume("--pals-arena-cancel-socket=abstract:cancel")
            .unwrap();
        let paths = args
            .finish(&ArenaSocketAddress::Abstract("child".into()))
            .unwrap();
        assert_eq!(paths.grant, ArenaSocketAddress::Abstract("grant".into()));
        assert_eq!(paths.cancel, ArenaSocketAddress::Abstract("cancel".into()));
        assert!(ArenaSocketAddress::parse("abstract:").is_err());
        assert!(ArenaSocketAddress::parse(&format!("abstract:{}", "a".repeat(108))).is_err());
        assert!(ArenaSocketAddress::parse("abstract:name\n").is_err());
        assert!(ArenaSocketAddress::parse("abstract:한글").is_err());
        assert!(ArenaSocketAddress::parse(&format!("abstract:{}", "a".repeat(107))).is_ok());
        let mut same = ArenaSocketArgs::default();
        same.consume("--pals-arena-grant-socket=abstract:same")
            .unwrap();
        same.consume("--pals-arena-cancel-socket=abstract:same")
            .unwrap();
        assert!(
            same.finish(&ArenaSocketAddress::Abstract("child".into()))
                .is_err()
        );
    }
    #[test]
    fn child_and_arena_paths_cannot_be_the_same() {
        let mut args = ArenaSocketArgs::default();
        args.consume("--pals-arena-grant-socket=/owned/grant")
            .unwrap();
        args.consume("--pals-arena-cancel-socket=/owned/child")
            .unwrap();
        assert!(
            args.finish(&ArenaSocketAddress::Path(PathBuf::from("/owned/child")))
                .is_err()
        );
    }
}
