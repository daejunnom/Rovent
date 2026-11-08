//! Candidate Own /3 owners assembled without admitting or starting work.
//!
//! This is deliberately not a launch entry: Own teacher effect claims and
//! same-owner settlement must be connected before these owners can run. The
//! prepared value itself is not a lifecycle custody or recovery registration.

use super::{ArenaError, invalid};
use super::pc_lock_v3::{ManagedOutputOwnerV3, PreparedPcCollectionInputsV3};
use crate::rovent_collect::capture_router_v3::CaptureRouterV3;
use crate::rovent_collect::game_authority_v3::{GameAuthorityV3, GameStepV3};
use crate::rovent_collect::own_hooks_lock_v3::RoventOwnCollectionBridgeV3;
use crate::rovent_collect::own_worker_v3::OwnWorkerSupervisorV3;
use rz_experiments::RoventInputLockV3;
use rz_uci::rovent_bootstrap::launch::{
    RoventOwnCollectionHooksV3, RoventOwnCollectionLaunchOwnerV3,
};
use std::os::unix::fs::{FileExt, MetadataExt};
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// Holds supplied owners strongly only while this value itself is retained.
/// It is not registered as custody across failure, cancellation, or unknown
/// physical completion. No publication, native or GPU completion can be
/// inferred from a successfully prepared value.
pub(crate) struct OwnCollectionPreparedV3 {
    inputs: Arc<PreparedPcCollectionInputsV3>,
    output: Arc<ManagedOutputOwnerV3>,
    lock: Arc<RoventInputLockV3>,
    lock_file: Arc<std::fs::File>,
    game: Arc<GameAuthorityV3<OwnWorkerSupervisorV3>>,
    router: Arc<CaptureRouterV3>,
    bridge: Arc<RoventOwnCollectionBridgeV3>,
    native: Arc<RoventOwnCollectionLaunchOwnerV3>,
    supervisor: Arc<Mutex<OwnWorkerSupervisorV3>>,
}

/// Observe the supplied lock FD without changing its shared offset. This does
/// not prove it is the native bootstrap's actual FD or that another process
/// cannot later modify the file.
fn read_supplied_lock_bytes(
    file: &std::fs::File,
    deadline: Instant,
) -> Result<Vec<u8>, ArenaError> {
    let before = file.metadata().map_err(super::io)?;
    if !before.is_file() || before.len() == 0 || before.len() > 1 << 20 {
        return Err(invalid("Own /3 original lock size or type differs"));
    }
    let len = usize::try_from(before.len()).map_err(|_| invalid("Own /3 lock length"))?;
    let mut bytes = Vec::new();
    bytes.try_reserve_exact(len).map_err(|_| invalid("Own /3 lock allocation"))?;
    let mut buffer = [0_u8; 65536];
    while bytes.len() < len {
        if Instant::now() >= deadline {
            return Err(invalid("Own /3 original lock read deadline"));
        }
        let count = (len - bytes.len()).min(buffer.len());
        match file.read_at(&mut buffer[..count], bytes.len() as u64) {
            Ok(0) => return Err(invalid("Own /3 original lock truncated")),
            Ok(count) => bytes.extend_from_slice(&buffer[..count]),
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(error) => return Err(super::io(error)),
        }
    }
    let after = file.metadata().map_err(super::io)?;
    if Instant::now() >= deadline {
        return Err(invalid("Own /3 original lock read deadline"));
    }
    let stamp = |m: &std::fs::Metadata| {
        (m.dev(), m.ino(), m.len(), m.mtime(), m.mtime_nsec(), m.ctime(), m.ctime_nsec())
    };
    if stamp(&before) != stamp(&after) {
        return Err(invalid("Own /3 original lock changed while reading"));
    }
    Ok(bytes)
}

fn verify_supplied_lock(
    lock: &RoventInputLockV3,
    file: &std::fs::File,
    deadline: Instant,
) -> Result<(), ArenaError> {
    let bytes = read_supplied_lock_bytes(file, deadline)?;
    let text = std::str::from_utf8(&bytes).map_err(|_| invalid("Own /3 original lock UTF8"))?;
    let decoded = RoventInputLockV3::from_json(text)
        .map_err(|error| invalid(format!("Own /3 original lock: {error}")))?;
    if &decoded != lock {
        return Err(invalid("Own /3 original lock FD differs from decoded owner"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Seek, SeekFrom};
    use std::time::Duration;

    #[test]
    fn supplied_fd_read_preserves_offset_and_rejects_expired_window() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/Cargo.toml");
        let mut original = std::fs::File::open(path).expect("tracked fixture source");
        original.seek(SeekFrom::Start(3)).unwrap();
        let expected = std::fs::read(path).unwrap();
        let bytes = read_supplied_lock_bytes(&original, Instant::now() + Duration::from_secs(5))
            .expect("bounded original FD read");
        assert_eq!(bytes, expected);
        assert_eq!(original.stream_position().unwrap(), 3);
        assert!(read_supplied_lock_bytes(&original, Instant::now()).is_err());
        assert_eq!(original.stream_position().unwrap(), 3);
    }
}

impl OwnCollectionPreparedV3 {
    /// Validate and retain the supplied input/lock and output owners.
    /// Failure leaves their original Arcs with the caller. The caller must
    /// retain the result before attempting any later admission; this function
    /// neither starts a worker nor authorizes output. Matching bytes and a
    /// supplied FD are not proof of the native bootstrap's original lock FD.
    /// A future launch must pass and validate that original FD directly and
    /// register this exact prepared owner in lifecycle custody before dispatch.
    pub(crate) fn prepare(
        inputs: &Arc<PreparedPcCollectionInputsV3>,
        output: &Arc<ManagedOutputOwnerV3>,
        lock: &Arc<RoventInputLockV3>,
        directory: &cap_std::fs::Dir,
        lock_file: &Arc<std::fs::File>,
        execution_ceiling: Instant,
        cleanup_ceiling: Instant,
    ) -> Result<Self, ArenaError> {
        if execution_ceiling > cleanup_ceiling || Instant::now() >= execution_ceiling {
            return Err(invalid("Own /3 original preparation deadline unavailable"));
        }
        verify_supplied_lock(lock, lock_file, execution_ceiling)?;
        // Do not reopen supplied descriptors by pathname. Native FD identity
        // is not yet observed or established here.
        inputs.validate_original_own_boundary(lock, directory, lock_file)?;
        if !output.matches_inputs(inputs) {
            return Err(invalid("Own /3 output original input owner differs"));
        }
        let game = GameAuthorityV3::new_candidate(Arc::clone(inputs));
        match game.try_bind_output(output) {
            GameStepV3::Ready | GameStepV3::Existing => {}
            _ => return Err(invalid("Own /3 original output binding unavailable")),
        }
        let (router, observer) = CaptureRouterV3::channel();
        let bridge = RoventOwnCollectionBridgeV3::new(
            router.registration(),
            Arc::clone(inputs),
            Arc::clone(&game),
            lock,
            directory,
            lock_file,
        )?;
        let supervisor = Arc::new(Mutex::new(
            OwnWorkerSupervisorV3::new(
                Arc::clone(&bridge),
                Arc::clone(&router),
                execution_ceiling,
                cleanup_ceiling,
            )
            .map_err(invalid)?,
        ));
        if !OwnWorkerSupervisorV3::try_bind_game(&supervisor).map_err(invalid)? {
            return Err(invalid("Own /3 original supervisor binding busy"));
        }
        bridge.bind_supervisor(&supervisor).map_err(invalid)?;
        let hooks: Arc<RoventOwnCollectionHooksV3> = bridge.clone();
        let native = Arc::new(RoventOwnCollectionLaunchOwnerV3::new(
            inputs.plan().teacher_sources.clone(),
            Vec::new(),
            hooks,
            Box::new(observer),
        ));
        bridge.bind_native_owner(&native).map_err(invalid)?;
        Ok(Self {
            inputs: Arc::clone(inputs),
            output: Arc::clone(output),
            lock: Arc::clone(lock),
            lock_file: Arc::clone(lock_file),
            game,
            router,
            bridge,
            native,
            supervisor,
        })
    }
}
