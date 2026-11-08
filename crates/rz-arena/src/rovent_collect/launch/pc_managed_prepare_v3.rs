//! Same-owner input preparation for the opt-in managed V3 caller.
//!
//! Preparation neither enters native execution nor authorizes output writes.
//! The returned owner is not evidence of physical completion or publication.
use super::ArenaError;
use super::managed_output_fd_v2::{ManagedPlanLeaseRejectionV2, ObservedManagedPlanLeaseV2};
use super::pc_cli_managed_v3::{ManagedCollectionCliV3, ManagedDeadlineErrorV3};
use super::pc_lock_v3::{
    ManagedOutputOwnerV3, PcCollectionOwnersV3, PcCollectionRunV3, PreparedPcCollectionInputsV3,
};
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

/// The same managed transport, but locked to the Own plan domain. It cannot
/// call the PC-only `start` path.
pub(crate) struct PreparedManagedOwnV3 {
    pub(crate) inputs: Arc<PreparedPcCollectionInputsV3>,
    pub(crate) output: Arc<ManagedOutputOwnerV3>,
    pub(crate) arguments: Vec<String>,
    pub(crate) work_until: Instant,
    pub(crate) cleanup_until: Instant,
}

/// Start failures retain the original input/output and, if created, the same
/// supervisor owner. The optional registry key is custody, not completion.
pub(crate) enum ManagedPcStartErrorV3 {
    WorkExpired(PreparedManagedPcV3),
    Supervisor {
        error: ArenaError,
        prepared: PreparedManagedPcV3,
    },
    /// Original inputs and payload are retained; any partial internal
    /// registration remains unknown until its own owner is recovered.
    SupervisorPanic {
        payload: Box<dyn std::any::Any + Send>,
        prepared: PreparedManagedPcV3,
    },
    WorkExpiredAfterSupervisor {
        prepared: PreparedManagedPcV3,
        owners: Arc<PcCollectionOwnersV3>,
    },
    Launch {
        error: ArenaError,
        prepared: PreparedManagedPcV3,
        owners: Arc<PcCollectionOwnersV3>,
    },
    Panic {
        payload: Box<dyn std::any::Any + Send>,
        prepared: PreparedManagedPcV3,
        owners: Arc<PcCollectionOwnersV3>,
    },
}

impl PreparedManagedPcV3 {
    /// Enter only after the caller has arranged bounded recovery for the
    /// original supervisor and output. A return is not final settlement.
    pub(crate) fn start(self) -> Result<PcCollectionRunV3, ManagedPcStartErrorV3> {
        if Instant::now() >= self.work_until {
            return Err(ManagedPcStartErrorV3::WorkExpired(self));
        }
        let preparation = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            PcCollectionOwnersV3::prepare(
                Arc::clone(&self.inputs),
                self.work_until,
                self.cleanup_until,
            )
        }));
        let owners = match preparation {
            Ok(Ok(owners)) => Arc::new(owners),
            Ok(Err(error)) => {
                return Err(ManagedPcStartErrorV3::Supervisor {
                    error,
                    prepared: self,
                });
            }
            Err(payload) => {
                return Err(ManagedPcStartErrorV3::SupervisorPanic {
                    payload,
                    prepared: self,
                });
            }
        };
        if Instant::now() >= self.work_until {
            return Err(ManagedPcStartErrorV3::WorkExpiredAfterSupervisor {
                prepared: self,
                owners,
            });
        }
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            owners.run_with_original_output(&self.output, self.arguments.clone())
        }));
        match result {
            Ok(Ok(run)) => Ok(run),
            Ok(Err(error)) => Err(ManagedPcStartErrorV3::Launch {
                error,
                prepared: self,
                owners,
            }),
            Err(payload) => Err(ManagedPcStartErrorV3::Panic {
                payload,
                prepared: self,
                owners,
            }),
        }
    }
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

/// A read-only profile observation retaining the same authenticated /2
/// owner and the original profile file. It grants no execution authority.
pub(crate) struct PreparedLongGamePreflightV3 {
    _observation: ObservedManagedPlanLeaseV2,
    pub(crate) profile: super::long_game_profile_v3::PreparedLongGameProfileV3,
}

pub(crate) enum LongGamePreflightErrorV3 {
    Selection(ArenaError),
    Setup(ManagedPcPrepareErrorV3),
    Read {
        error: ArenaError,
        observation: ObservedManagedPlanLeaseV2,
        profile: Option<super::long_game_profile_v3::PreparedLongGameProfileV3>,
    },
}

/// Uses the same inherited /2 input owner; no native/teacher or output write
/// is started. The selected profile pin is not inserted into the launch plan.
pub(crate) fn prepare_long_game_preflight_v3(
    selection: ManagedCollectionCliV3,
    bootstrap_until: Instant,
) -> Result<PreparedLongGamePreflightV3, LongGamePreflightErrorV3> {
    let profile_pin = selection.long_game_preflight.as_ref().ok_or_else(|| {
        LongGamePreflightErrorV3::Selection(super::invalid("long-game preflight not selected"))
    })?;
    let fds = ManagedCollectionInheritedFds::seal_exclusive(
        selection.channel,
        selection.output,
        selection.input,
        selection.plan,
    )
    .map_err(|error| {
        LongGamePreflightErrorV3::Setup(ManagedPcPrepareErrorV3::DescriptorUncertain(error))
    })?;
    let pin = &selection.selection.plan_pin;
    let mut observation = ObservedManagedPlanLeaseV2::bind(
        fds,
        pin.path.clone(),
        pin.bytes,
        pin.sha256.clone(),
        pin.bytes,
        bootstrap_until,
    )
    .map_err(|error| {
        LongGamePreflightErrorV3::Setup(ManagedPcPrepareErrorV3::Observation(error))
    })?;
    if let Err(error) = selection.remaining_deadlines(&mut observation, bootstrap_until) {
        return Err(LongGamePreflightErrorV3::Setup(
            ManagedPcPrepareErrorV3::Deadline { error, observation },
        ));
    }
    let first_read = (|| -> Result<_, ArenaError> {
        observation
            .verify_live(bootstrap_until)
            .map_err(super::io)?;
        let (directory, _plan) = observation.clone_original_input().map_err(super::io)?;
        let directory = cap_std::fs::Dir::from_std_file(directory);
        super::long_game_profile_v3::PreparedLongGameProfileV3::open(&directory, profile_pin)
    })();
    let profile = match first_read {
        Ok(profile) => profile,
        Err(error) => {
            return Err(LongGamePreflightErrorV3::Read {
                error,
                observation,
                profile: None,
            });
        }
    };
    let after = if Instant::now() >= bootstrap_until {
        Err(super::invalid("long-game preflight deadline expired"))
    } else {
        observation.verify_live(bootstrap_until).map_err(super::io)
    };
    if let Err(error) = after {
        return Err(LongGamePreflightErrorV3::Read {
            error,
            observation,
            profile: Some(profile),
        });
    }
    Ok(PreparedLongGamePreflightV3 {
        _observation: observation,
        profile,
    })
}

/// `bootstrap_until` is a finite local setup ceiling supplied independently
/// of the unverified parent timestamps. Seal at single-threaded child entry;
/// a successful namespace observation is only point-in-time evidence.
pub(crate) fn prepare_managed_pc_v3(
    selection: ManagedCollectionCliV3,
    bootstrap_until: Instant,
) -> Result<PreparedManagedPcV3, ManagedPcPrepareErrorV3> {
    prepare_managed_collection_v3(selection, bootstrap_until, false)
}

/// Open Own inputs and output under the same inherited observation owner.
pub(crate) fn prepare_managed_own_v3(
    selection: ManagedCollectionCliV3,
    bootstrap_until: Instant,
) -> Result<PreparedManagedOwnV3, ManagedPcPrepareErrorV3> {
    let prepared = prepare_managed_collection_v3(selection, bootstrap_until, true)?;
    Ok(PreparedManagedOwnV3 {
        inputs: prepared.inputs,
        output: prepared.output,
        arguments: prepared.arguments,
        work_until: prepared.work_until,
        cleanup_until: prepared.cleanup_until,
    })
}

fn prepare_managed_collection_v3(
    selection: ManagedCollectionCliV3,
    bootstrap_until: Instant,
    own: bool,
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
    let inputs_result = if own {
        PreparedPcCollectionInputsV3::open_own_from_managed_observation(
            &mut observation, pin, bootstrap_until,
        )
    } else {
        PreparedPcCollectionInputsV3::open_from_managed_observation(
            &mut observation, pin, bootstrap_until,
        )
    };
    let inputs = match inputs_result {
        Ok(inputs) => Arc::new(inputs),
        Err(error) => return Err(ManagedPcPrepareErrorV3::Inputs { error, observation }),
    };
    let output = match ManagedOutputOwnerV3::from_observation(
        Arc::clone(&inputs),
        observation,
        work_until,
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
