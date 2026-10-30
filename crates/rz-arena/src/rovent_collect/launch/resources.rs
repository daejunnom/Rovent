//! Collection keeps the exact same resource owner as the direct native upper.
//! No second cgroup preparation, policy, OS probe or shutdown implementation.
use crate::ArenaError;
use crate::rovent_collect::RoventPublicationFile;
use crate::rovent_resource_scope::launch::{
    NativeResourceCustody, RoventNativeResourcePlanPin, prepare_resource_custody,
};
use cap_std::fs::Dir;
use rz_experiments::{RoventInputLock, RoventInputLockV2};
use serde_json::Value;
use std::{path::PathBuf, sync::Arc};

pub(super) struct CollectionResourceCustody {
    original: NativeResourceCustody,
}
impl CollectionResourceCustody {
    pub(super) fn no_admission_started(&self) -> bool {
        self.original.no_admission_started()
    }
    pub(super) fn prepare(
        original_private_root: &Dir,
        pin: &RoventPublicationFile,
    ) -> Result<Self, ArenaError> {
        pin.validate()?;
        let pin = RoventNativeResourcePlanPin {
            path: PathBuf::from(&pin.path),
            sha256: pin.sha256.clone(),
            bytes: pin.bytes,
        };
        Ok(Self {
            original: prepare_resource_custody(original_private_root, pin)?,
        })
    }
    pub(super) fn admit(&mut self, lock: &Arc<RoventInputLock>) -> Result<(), ArenaError> {
        self.original.admit(lock)
    }
    pub(super) fn admit_v2(&mut self, lock: &Arc<RoventInputLockV2>) -> Result<(), ArenaError> {
        self.original.admit_v2(lock)
    }
    pub(super) fn finish_live(&mut self) -> Result<(), ArenaError> {
        self.original.finish_live()
    }
    pub(super) fn validate_success_v2(
        &self,
        lock: &Arc<RoventInputLockV2>,
    ) -> Result<(), ArenaError> {
        self.original.validate_success_v2(lock)
    }
    pub(super) fn observation(&self) -> Value {
        self.original.observation()
    }
}
