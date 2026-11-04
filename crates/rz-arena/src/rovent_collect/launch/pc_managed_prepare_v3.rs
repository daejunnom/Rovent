//! Same-owner input preparation for the opt-in managed V3 caller.
//!
//! Preparation neither enters native execution nor authorizes output writes.
//! The returned owner is not evidence of physical completion or publication.
use super::ArenaError;
use super::managed_output_fd_v2::{ManagedPlanLeaseRejectionV2, ObservedManagedPlanLeaseV2};
use super::pc_cli_managed_v3::{ManagedCollectionCliV3, ManagedDeadlineErrorV3};
use super::pc_lock_v3::{ManagedOutputOwnerV3, PreparedPcCollectionInputsV3};
use rz_platform::ManagedCollectionInheritedFds;
use std::sync::Arc;
use std::time::Instant;

pub(crate) struct PreparedManagedPcV3 {
    pub(crate) inputs: Arc<PreparedPcCollectionInputsV3>,
    pub(crate) output: Arc<ManagedOutputOwnerV3>,
    pub(crate) arguments: Vec<String>,
    pub(crate) work_until: Instant,
    pub(crate) cleanup_until: Instant,
}

/// Rejection carries the original /2 observation when it was acquired. It is
/// not safe to replace an owner with a matching path or a new file descriptor.
pub(crate) enum ManagedPcPrepareErrorV3 {
    /// The seal can have changed descriptor flags before failure. This is
    /// neither a no-mutation result nor a completed supervisor handoff.
    DescriptorUncertain(std::io::Error),
    Observation(ManagedPlanLeaseRejectionV2),
    Deadline {
        error: ManagedDeadlineErrorV3,
        observation: ObservedManagedPlanLeaseV2,
    },
    Inputs {
        error: ArenaError,
        observation: ObservedManagedPlanLeaseV2,
    },
    Output {
        error: ArenaError,
        inputs: Arc<PreparedPcCollectionInputsV3>,
        observation: ObservedManagedPlanLeaseV2,
    },
    WorkExpired {
        inputs: Arc<PreparedPcCollectionInputsV3>,
        output: Arc<ManagedOutputOwnerV3>,
    },
}

/// `bootstrap_until` is a finite local setup ceiling supplied independently
/// of the unverified parent timestamps. Seal at single-threaded child entry;
/// a successful namespace observation is only point-in-time evidence.
pub(crate) fn prepare_managed_pc_v3(
    selection: ManagedCollectionCliV3,
    bootstrap_until: Instant,
) -> Result<PreparedManagedPcV3, ManagedPcPrepareErrorV3> {
    let fds = ManagedCollectionInheritedFds::seal_exclusive(
        selection.channel,
        selection.output,
        selection.input,
        selection.plan,
    )
    .map_err(ManagedPcPrepareErrorV3::DescriptorUncertain)?;
    let pin = &selection.selection.plan_pin;
    let mut observation = ObservedManagedPlanLeaseV2::bind(
        fds,
        pin.path.clone(),
        pin.bytes,
        pin.sha256.clone(),
        pin.bytes,
        bootstrap_until,
    )
    .map_err(ManagedPcPrepareErrorV3::Observation)?;
    let (work_until, cleanup_until) =
        match selection.remaining_deadlines(&mut observation, bootstrap_until) {
            Ok(window) => window,
            Err(error) => return Err(ManagedPcPrepareErrorV3::Deadline { error, observation }),
        };
    let inputs = match PreparedPcCollectionInputsV3::open_from_managed_observation(
        &mut observation,
        pin,
        bootstrap_until,
    ) {
        Ok(inputs) => Arc::new(inputs),
        Err(error) => return Err(ManagedPcPrepareErrorV3::Inputs { error, observation }),
    };
    let output = match ManagedOutputOwnerV3::from_observation(
        Arc::clone(&inputs),
        observation,
        cleanup_until,
    ) {
        Ok(output) => Arc::new(output),
        Err((error, inputs, observation)) => {
            return Err(ManagedPcPrepareErrorV3::Output {
                error,
                inputs,
                observation,
            });
        }
    };
    if Instant::now() >= work_until {
        return Err(ManagedPcPrepareErrorV3::WorkExpired { inputs, output });
    }
    Ok(PreparedManagedPcV3 {
        inputs,
        output,
        arguments: selection.selection.native_arguments,
        work_until,
        cleanup_until,
    })
}
