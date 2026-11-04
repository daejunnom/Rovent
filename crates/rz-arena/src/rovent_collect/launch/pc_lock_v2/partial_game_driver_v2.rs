//! All-settled game partial output. Root owners and physical fences remain
//! original; a partial index never implies complete publication or training.
use super::*;
use crate::rovent_collect::launch::own_publisher_v2::{OwnPumpPollV2, PartialOfferOutcomeV2};
use crate::rovent_collect::launch::partial_collection_v1::{
    PartialChunkPin, PartialCollectionAccounting, PartialEndReason, account_flushed_chunks,
    create_partial_directory, write_partial_receipt,
};
use crate::rovent_collect::launch::partial_game_v2::{
    GAME_PARTIAL_INDEX_V2_DOMAIN, GamePartialIndexV2, GamePartialRootV2, pin_root_partial_receipt,
    validate_game_partial_index, write_game_partial_index,
};
use crate::rovent_collect::pc_hooks_lock_v2::PcRootSeparationV2;
use crate::rovent_collect::pc_worker_supervisor_v2::PcSecondaryWorkerStartV2;

type OriginalRoot = (
    u16,
    Arc<RoventPcCollectionBridgeV2>,
    Arc<Mutex<RoventPcCollectionWorkerV2>>,
    Arc<super::super::own_publisher_v2::OwnPumpConsumerV2>,
);

fn offer_partial_root(
    owner: &RoventCollectionOwnerLockV2,
    bridge: &RoventPcCollectionBridgeV2,
    worker: &Arc<Mutex<RoventPcCollectionWorkerV2>>,
    consumer: &super::super::own_publisher_v2::OwnPumpConsumerV2,
    assembly: &mut RoventCorpusAssemblyLockV2,
) -> Result<PartialOfferOutcomeV2, ArenaError> {
    let worker = match worker.try_lock() {
        Ok(worker) => worker,
        Err(std::sync::TryLockError::WouldBlock) => return Ok(PartialOfferOutcomeV2::Pending),
        Err(std::sync::TryLockError::Poisoned(_)) => {
            return Err(invalid("P/C original collection worker poisoned"));
        }
    };
    let Some(handoff) = worker.original_handoff() else {
        return Ok(PartialOfferOutcomeV2::Pending);
    };
    let mut handoff = match handoff.try_lock() {
        Ok(handoff) => handoff,
        Err(std::sync::TryLockError::WouldBlock) => return Ok(PartialOfferOutcomeV2::Pending),
        Err(std::sync::TryLockError::Poisoned(_)) => {
            return Err(invalid("P/C original handoff poisoned"));
        }
    };
    let admission = handoff.parts_mut().0.clone();
    let Some(gate) = bridge.try_begin_commit(&admission)? else {
        return Ok(PartialOfferOutcomeV2::Pending);
    };
    consumer.try_offer_front_partial_pc(owner, assembly, &gate, &mut handoff)
}

fn advance_partial_root(
    bridge: &RoventPcCollectionBridgeV2,
    worker: &Arc<Mutex<RoventPcCollectionWorkerV2>>,
    assembly: &mut RoventCorpusAssemblyLockV2,
) -> Result<(), ArenaError> {
    let attempt = |assembly: &mut RoventCorpusAssemblyLockV2| -> Result<bool, ArenaError> {
        let worker = match worker.try_lock() {
            Ok(worker) => worker,
            Err(std::sync::TryLockError::WouldBlock) => return Ok(false),
            Err(std::sync::TryLockError::Poisoned(_)) => {
                return Err(invalid("P/C original collection worker poisoned"));
            }
        };
        let Some(handoff) = worker.original_handoff() else {
            return Ok(false);
        };
        let mut handoff = match handoff.try_lock() {
            Ok(handoff) => handoff,
            Err(std::sync::TryLockError::WouldBlock) => return Ok(false),
            Err(std::sync::TryLockError::Poisoned(_)) => {
                return Err(invalid("P/C original handoff poisoned"));
            }
        };
        let admission = handoff.parts_mut().0.clone();
        let Some(gate) = bridge.try_begin_commit(&admission)? else {
            return Ok(false);
        };
        assembly.try_advance_fenced_pc(&gate, &mut handoff)
    };
    let advanced = match attempt(assembly) {
        Err(error) if assembly.chunk_flush_required()? => {
            assembly.flush()?;
            attempt(assembly).map_err(|retry| {
                invalid(format!(
                    "P/C game append before flush: {error}; retry: {retry}"
                ))
            })?
        }
        other => other?,
    };
    check(
        advanced,
        "P/C game partial original commit gate remained busy",
    )
}

impl RoventPcCollectionRunV2 {
    /// Collect each settled root into its own /1 partial directory, then bind
    /// them in a separate /2 index. All original worker, fence and custody
    /// checks remain mandatory. Unknown physical completion is not resolved.
    pub fn collect_partial_settled_game(&mut self, until: Instant) -> Result<(), ArenaError> {
        let inputs = self
            .verified_game_inputs
            .as_ref()
            .ok_or_else(|| invalid("P/C game partial requires pre-start verified plan and spec"))?
            .clone();
        let spec = &inputs.spec;
        let max_total_bytes = inputs.max_total_bytes;
        let index_max_bytes = inputs.index_max_bytes;
        spec.validate(&inputs.plan_sha256).map_err(io)?;
        let capacity = inputs.plan.maximum_roots;
        check(
            capacity > 0
                && capacity <= 64
                && self.secondary_slots.len() < capacity as usize
                && !self.publication_attempted
                && self.publication_assembly.is_none()
                && self.publication_dir.is_none()
                && self.partial_game_assemblies.is_empty()
                && self.supervisor.is_none()
                && self.supervisor_error.is_none()
                && self.pump_thread_joined
                && self.pump_thread_failure.is_none()
                && !self.worker_thread_panicked
                && (!self.speculative_teacher_requested || self.speculative_teacher_enabled),
            "P/C game partial original supervisor or custody unresolved",
        )?;
        self.closed_owner()?;
        let mut roots: Vec<OriginalRoot> = vec![(
            0,
            Arc::clone(&self.bridge),
            Arc::clone(&self.worker),
            Arc::clone(&self.pump_consumer),
        )];
        roots
            .try_reserve(self.secondary_slots.len())
            .map_err(|_| invalid("P/C game partial root capacity"))?;
        for (offset, slot) in self.secondary_slots.iter().enumerate() {
            check(
                matches!(slot.bridge.try_has_deferred()?, Some(false)),
                "P/C game partial secondary deferred",
            )?;
            if slot.bridge.installed_admission_for_recovery().is_none() {
                let unstarted = (slot.worker_join_observed
                    && slot.consumer_drained_after_join
                    && slot.worker_exit_kind == Some(PcWorkerExitKindV2::NoAdmission))
                    || (slot.worker_start == PcSecondaryWorkerStartV2::FailedBeforeStart
                        && slot.consumer_empty_before_worker_start);
                check(unstarted, "P/C game partial secondary start unknown")?;
                continue;
            }
            check(
                slot.speculative_setup_error.is_none()
                    && slot.worker_join_observed
                    && slot.consumer_drained_after_join
                    && matches!(
                        slot.worker_exit_kind,
                        Some(PcWorkerExitKindV2::Aggregate | PcWorkerExitKindV2::Retirement)
                    )
                    && slot
                        .bridge
                        .returned_handoff_settled()
                        .map_err(|e| invalid(format!("P/C secondary handoff: {e:?}")))?,
                "P/C game partial secondary owner unresolved",
            )?;
            roots.push((
                (offset + 1) as u16,
                Arc::clone(&slot.bridge),
                Arc::clone(&slot.worker),
                Arc::clone(&slot.consumer),
            ));
        }
        for (_, bridge, worker, _) in &roots {
            check(
                matches!(bridge.try_has_deferred()?, Some(false))
                    && worker.try_lock().is_ok_and(|worker| {
                        !worker.has_failure() && !worker.has_unpublished_pump_rows()
                    }),
                "P/C game partial root deferred or worker incomplete",
            )?;
        }
        check(
            self.late_rows.is_empty()
                && self.owner.try_status()?.is_some_and(|status| {
                    status.collection_disabled_first_ticket.is_none()
                        && status.prior_disabled_first_tickets.is_empty()
                        && status.incomplete_capture_count == 0
                        && status.failure.is_none()
                        && !status.callback_lock_failure
                        && status.game_reserved_slots == Some(0)
                }),
            "P/C game partial shared owner incomplete",
        )?;
        let (_, last, _, _) = roots.last().expect("primary root retained");
        let last_admission = last
            .installed_admission_for_recovery()
            .ok_or_else(|| invalid("P/C last original admission missing"))?
            .clone();
        loop {
            check(
                Instant::now() < until,
                "P/C game partial separation deadline",
            )?;
            match last.try_separate_root_for_next(
                &last_admission,
                self.publication_maximum_records as usize,
            )? {
                PcRootSeparationV2::Pending => std::thread::sleep(Duration::from_millis(1)),
                PcRootSeparationV2::Released { .. } => break,
            }
        }
        let mut index = GamePartialIndexV2 {
            schema: GAME_PARTIAL_INDEX_V2_DOMAIN.into(),
            collection_plan_sha256: inputs.plan_sha256.clone(),
            partial_spec_sha256: inputs.spec_sha256.clone(),
            status: "incomplete".into(),
            training_eligible: false,
            publication_authority: false,
            physical_completion: "not_attested".into(),
            root_capacity: capacity,
            max_total_rows: (spec.row_limit as u64).min(self.publication_maximum_records),
            max_total_bytes,
            index_max_bytes,
            observed_valid_rows: 0,
            observed_failed_rows: 0,
            roots: (0..capacity)
                .map(|slot| GamePartialRootV2::Unknown {
                    slot,
                    root_admission_sha256: None,
                    reason: "slot has no observed admission".into(),
                })
                .collect(),
        };
        let parent = self
            .publication_directory
            .parent()
            .ok_or_else(|| invalid("P/C game partial output parent"))?;
        let leaf = self
            .publication_directory
            .file_name()
            .ok_or_else(|| invalid("P/C game partial output leaf"))?;
        let root = Dir::open_ambient_dir(parent, cap_std::ambient_authority()).map_err(io)?;
        // With only unknown entries this validates the exact pinned source plan,
        // capacity and index bounds before any output or row is moved.
        validate_game_partial_index(&root, &index, &inputs.plan_bytes, &inputs.spec_bytes)
            .map_err(io)?;
        let mut remaining_bytes = max_total_bytes;
        self.partial_game_assemblies
            .try_reserve(roots.len())
            .map_err(|_| invalid("P/C game partial custody capacity"))?;
        self.publication_attempted = true;
        root.create_dir(leaf).map_err(io)?;
        let output = root.open_dir_nofollow(leaf).map_err(io)?;
        for (slot, bridge, worker, consumer) in roots {
            check(Instant::now() < until, "P/C game partial root deadline")?;
            let admission = bridge
                .installed_admission_for_recovery()
                .ok_or_else(|| invalid("P/C game partial root admission missing"))?
                .clone();
            let used_rows = index
                .observed_valid_rows
                .checked_add(index.observed_failed_rows)
                .ok_or_else(|| invalid("P/C game partial row overflow"))?;
            let left_rows = index
                .max_total_rows
                .checked_sub(used_rows)
                .ok_or_else(|| invalid("P/C game partial row limit"))?;
            check(left_rows > 0, "P/C game partial row budget exhausted")?;
            check(
                remaining_bytes > 0,
                "P/C game partial byte budget exhausted",
            )?;
            let mut root_spec = spec.clone();
            root_spec.row_limit = usize::try_from(left_rows)
                .unwrap_or(usize::MAX)
                .min(spec.row_limit);
            root_spec.byte_limit = remaining_bytes.min(spec.byte_limit);
            let path = format!("root-{slot}");
            output.create_dir(&path).map_err(io)?;
            let root_dir = output.open_dir_nofollow(&path).map_err(io)?;
            let partial = create_partial_directory(&root_dir).map_err(io)?;
            partial.create_dir("chunks").map_err(io)?;
            self.publication_dir = Some(partial);
            self.publication_assembly = Some(RoventCorpusAssemblyLockV2::new(
                &self.owner,
                self.publication_dir
                    .as_ref()
                    .expect("original partial directory"),
                &self.publication_manifest,
                &admission,
                left_rows,
            )?);
            self.publication_assembly
                .as_mut()
                .expect("partial assembly retained")
                .restrict_partial_chunk_bytes(root_spec.byte_limit)?;
            let owner = self.owner.clone();
            let assembly = self
                .publication_assembly
                .as_mut()
                .expect("partial assembly retained");
            let result = (|| -> Result<(u64, u64, u64, Option<super::super::partial_game_v2::GamePartialPinV2>), ArenaError> {
                loop {
                    check(Instant::now() < until, "P/C game partial consumer deadline")?;
                    let polled = consumer.poll()?;
                    match offer_partial_root(&owner, &bridge, &worker, &consumer, assembly)? {
                        PartialOfferOutcomeV2::Offered => advance_partial_root(&bridge, &worker, assembly)?,
                        PartialOfferOutcomeV2::PolicyUnavailableRetained => {},
                        PartialOfferOutcomeV2::BlockedFailureRetained => return Err(invalid("P/C game partial retains other failure")),
                        PartialOfferOutcomeV2::Pending => {
                            let Some((items, _, _, _)) = consumer.try_counts()? else { std::thread::sleep(Duration::from_millis(1)); continue; };
                            if items != 0 { std::thread::sleep(Duration::from_millis(1)); continue; }
                            if matches!(polled, OwnPumpPollV2::Empty | OwnPumpPollV2::Disconnected) { break; }
                            std::thread::sleep(Duration::from_millis(1));
                        }
                    }
                }
                loop {
                    check(Instant::now() < until, "P/C game partial separated row deadline")?;
                    match bridge.try_offer_separated_unlabeled(&admission, assembly)? {
                        None => std::thread::sleep(Duration::from_millis(1)),
                        Some(false) => break,
                        Some(true) => advance_partial_root(&bridge, &worker, assembly)?,
                    }
                }
                let failures = consumer.try_partial_failure_ids()?.ok_or_else(|| invalid("P/C game partial failure custody busy"))?;
                let valid = assembly.records();
                let failed = failures.len() as u64;
                if valid == 0 && failed == 0 { return Ok((0, 0, 0, None)); }
                let pins: Vec<PartialChunkPin> = if valid == 0 { Vec::new() } else {
                    assembly.flush()?;
                    assembly.finish()?.index().chunks.iter().map(|chunk| PartialChunkPin {
                        path: chunk.path.clone(), bytes: chunk.bytes, sha256: chunk.sha256.clone(), rows: chunk.records,
                    }).collect()
                };
                let partial_dir = self.publication_dir.as_ref().expect("partial directory retained");
                let mut accounting = PartialCollectionAccounting::new(root_spec.row_limit, root_spec.byte_limit)
                    .map_err(|e| invalid(format!("partial accounting: {e:?}")))?;
                account_flushed_chunks(partial_dir, &root_spec, &mut accounting, &pins).map_err(io)?;
                let chunk_bytes = pins.iter().try_fold(0u64, |sum, pin| sum.checked_add(pin.bytes))
                    .ok_or_else(|| invalid("P/C game partial chunk byte overflow"))?;
                let left = root_spec.byte_limit.checked_sub(chunk_bytes).ok_or_else(|| invalid("P/C game partial chunk limit"))?;
                let capture = consumer.capture_partial_failures(partial_dir, left)?;
                let captured: Vec<_> = capture.as_ref().map(|(_, rows)| rows.iter().map(|(id, _)| id.clone()).collect()).unwrap_or_default();
                check(captured == failures, "P/C game partial failure custody differs")?;
                if let Some((_, rows)) = capture.as_ref() {
                    for (id, bytes) in rows {
                        accounting.begin_row(id).map_err(|e| invalid(format!("partial failure accounting: {e:?}")))?;
                        accounting.record_failure(id, &format!("failure-evidence.jsonl#{id}"), *bytes)
                            .map_err(|e| invalid(format!("partial failure accounting: {e:?}")))?;
                    }
                }
                let failure_bytes = capture.as_ref().map_or(0, |(pin, _)| pin.bytes);
                let payload_bytes = chunk_bytes.checked_add(failure_bytes)
                    .ok_or_else(|| invalid("P/C game partial payload byte overflow"))?;
                root_spec.receipt_byte_limit = remaining_bytes.checked_sub(payload_bytes)
                    .filter(|value| *value > 0)
                    .ok_or_else(|| invalid("P/C game partial receipt byte limit"))?
                    .min(spec.receipt_byte_limit);
                let mut receipt = accounting.finish(PartialEndReason::InputExhausted)
                    .map_err(|(e, _)| invalid(format!("partial accounting: {e:?}")))?.with_chunks(pins);
                if let Some((pin, _)) = capture { receipt = receipt.with_failure_evidence(pin); }
                write_partial_receipt(partial_dir, &root_spec, &receipt).map_err(io)?;
                let receipt_pin = pin_root_partial_receipt(&output, &path, &root_spec).map_err(io)?;
                let bytes = chunk_bytes.checked_add(failure_bytes).and_then(|n| n.checked_add(receipt_pin.bytes))
                    .ok_or_else(|| invalid("P/C game partial output byte overflow"))?;
                Ok((valid, failed, bytes, Some(receipt_pin)))
            })();
            let (valid, failed, bytes, receipt) = result?;
            remaining_bytes = remaining_bytes
                .checked_sub(bytes)
                .ok_or_else(|| invalid("P/C game partial byte accounting"))?;
            index.observed_valid_rows = index
                .observed_valid_rows
                .checked_add(valid)
                .ok_or_else(|| invalid("P/C game partial valid count overflow"))?;
            index.observed_failed_rows = index
                .observed_failed_rows
                .checked_add(failed)
                .ok_or_else(|| invalid("P/C game partial failed count overflow"))?;
            index.roots[slot as usize] = match receipt {
                Some(receipt) => GamePartialRootV2::Available {
                    slot,
                    root_admission_sha256: admission.raw_sha256().to_owned(),
                    relative_path: path,
                    spec: root_spec,
                    receipt,
                },
                None => GamePartialRootV2::Unknown {
                    slot,
                    root_admission_sha256: Some(admission.raw_sha256().to_owned()),
                    reason: "settled root has no retained rows".into(),
                },
            };
            let directory = self
                .publication_dir
                .take()
                .expect("partial directory retained");
            let assembly = self
                .publication_assembly
                .take()
                .expect("partial assembly retained");
            self.partial_game_assemblies.push((directory, assembly));
        }
        check(
            (index.observed_valid_rows > 0 || index.observed_failed_rows > 0)
                && self.owner.try_status()?.is_some_and(|status| {
                    status.pending_retired_count == 0
                        && status.game_reserved_slots == Some(0)
                        && index
                            .observed_valid_rows
                            .checked_add(index.observed_failed_rows)
                            .is_some_and(|n| {
                                u64::try_from(status.game_retired_total).ok() == Some(n)
                            })
                }),
            "P/C game partial owner totals incomplete",
        )?;
        write_game_partial_index(&output, &index, &inputs.plan_bytes, &inputs.spec_bytes)
            .map_err(io)?;
        Ok(())
    }
}
