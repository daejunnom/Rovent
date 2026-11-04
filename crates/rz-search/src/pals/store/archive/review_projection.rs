//! Bounded, read-only historical projection of an existing live review pin.
//! This is not a cold archive, a physical completion receipt, or authority.
use super::*;

const PROJECTION_MAX_BYTES: usize = 16 * 1024 * 1024;
const PROJECTION_MAX_RECORDS: usize = 65_536;

fn validate_projection_bounds(max_records: usize, max_bytes: usize) -> Result<(), StoreError> {
    if max_records == 0
        || max_records > PROJECTION_MAX_RECORDS
        || max_bytes == 0
        || max_bytes > PROJECTION_MAX_BYTES
    {
        return Err(StoreError::InvalidConditions("review projection bounds"));
    }
    Ok(())
}

struct ProjectionWriter<'a> {
    bytes: Vec<u8>,
    limit: usize,
    check: &'a mut dyn FnMut() -> Result<(), StoreError>,
    failed: Option<StoreError>,
}

impl Write for ProjectionWriter<'_> {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        if let Err(error) = (self.check)() {
            self.failed = Some(error);
            return Err(std::io::Error::other("review projection control"));
        }
        let Some(needed) = self.bytes.len().checked_add(bytes.len()) else {
            self.failed = Some(StoreError::ArchiveByteBudget);
            return Err(std::io::Error::other("review projection bytes"));
        };
        if needed > self.limit {
            self.failed = Some(StoreError::ArchiveByteBudget);
            return Err(std::io::Error::other("review projection bytes"));
        }
        if self.bytes.try_reserve_exact(bytes.len()).is_err() {
            self.failed = Some(StoreError::Capacity("review projection allocation"));
            return Err(std::io::Error::other("review projection allocation"));
        }
        self.bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl ProjectionWriter<'_> {
    fn raw(&mut self, bytes: &[u8]) -> Result<(), StoreError> {
        self.write_all(bytes).map_err(|_| {
            self.failed
                .take()
                .unwrap_or(StoreError::ArchiveIntegrity("review projection write"))
        })
    }
    fn value<T: Serialize + ?Sized>(&mut self, value: &T) -> Result<(), StoreError> {
        serde_json::to_writer(&mut *self, value).map_err(|_| {
            self.failed
                .take()
                .unwrap_or(StoreError::ArchiveIntegrity("review projection encoding"))
        })
    }
}

impl StoreReadPin {
    fn projection_registry(
        &self,
    ) -> Result<std::sync::MutexGuard<'_, ReviewPinRegistry>, StoreError> {
        self.registry.try_lock().map_err(|error| match error {
            std::sync::TryLockError::WouldBlock => StoreError::ArchiveIo {
                stage: "review projection pin registry busy",
                kind: std::io::ErrorKind::WouldBlock,
            },
            std::sync::TryLockError::Poisoned(_) => {
                StoreError::InvalidConditions("review pin owner poisoned")
            }
        })
    }

    fn projection_validate(
        &self,
        stores: &PalsStores,
        check: &mut dyn FnMut() -> Result<(), StoreError>,
    ) -> Result<(), StoreError> {
        check()?;
        if !self.owner.matches(stores)
            || self.generation != stores.generation()
            || !Arc::ptr_eq(&self.registry, &stores.review_pins)
        {
            return Err(StoreError::StaleConsumer);
        }
        {
            let registry = self.projection_registry()?;
            if !registry.leases.contains_key(&self.serial) {
                return Err(StoreError::StaleConsumer);
            }
        }
        check()
    }

    fn projection_seed(
        &self,
        stores: &PalsStores,
        max_records: usize,
        check: &mut dyn FnMut() -> Result<(), StoreError>,
    ) -> Result<StorePins, StoreError> {
        check()?;
        if !self.owner.matches(stores)
            || self.generation != stores.generation()
            || !Arc::ptr_eq(&self.registry, &stores.review_pins)
        {
            return Err(StoreError::StaleConsumer);
        }
        let registry = self.projection_registry()?;
        // A caller's control callback may panic. Catch it while the registry
        // guard is live, release the guard normally, then resume the original
        // panic outside the lock so this diagnostic cannot poison product pins.
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let pins = registry
                .leases
                .get(&self.serial)
                .ok_or(StoreError::InvalidHandle("review pin lease"))?;
            if pins.count() > max_records {
                return Err(StoreError::Capacity("review projection seed"));
            }
            let mut controlled = Some(&mut *check as &mut dyn FnMut() -> Result<(), StoreError>);
            let result = StorePins {
                states: clone_closure_set(&pins.states, &mut controlled)?,
                lines: clone_closure_set(&pins.lines, &mut controlled)?,
                observations: clone_closure_set(&pins.observations, &mut controlled)?,
                situations: clone_closure_set(&pins.situations, &mut controlled)?,
                executions: clone_closure_set(&pins.executions, &mut controlled)?,
            };
            check_pin_closure(&mut controlled)?;
            Ok(result)
        }));
        drop(registry);
        let result = match result {
            Ok(result) => result?,
            Err(panic) => std::panic::resume_unwind(panic),
        };
        check()?;
        Ok(result)
    }
}

impl PalsStores {
    /// Detached evidence for one already-selected original CPU7 view. This
    /// does not claim when it was called, bind a trial, or persist anything.
    /// A caller must capture it at the selection boundary using the original
    /// cancellation and deadline, then bind and publish it separately.
    pub(crate) fn capture_cohort_cpu7_view(
        &self,
        view: &crate::pals::review_context::ReviewView,
        pin: &StoreReadPin,
        max_records: usize,
        max_bytes: usize,
        check: &mut dyn FnMut() -> Result<(), StoreError>,
    ) -> Result<Vec<u8>, StoreError> {
        check()?;
        let original = view
            .cohort_cpu7_original_bundle()
            .ok_or(StoreError::InvalidConditions("CPU7 view required"))?;
        let binding = view
            .cohort_cpu7_binding()
            .ok_or(StoreError::InvalidEvidence("CPU7 view binding missing"))?;
        if original.generation() != self.generation()
            || original.root() != view.root_state()
            || original.selected_len() > original.records().len()
        {
            return Err(StoreError::StaleConsumer);
        }
        let anchor = original
            .records()
            .first()
            .ok_or(StoreError::InvalidEvidence("CPU7 original bundle empty"))?
            .record()
            .origin_state;
        if original
            .records()
            .iter()
            .any(|r| r.record().origin_state != anchor)
        {
            return Err(StoreError::InvalidEvidence("CPU7 original anchor mismatch"));
        }
        validate_projection_bounds(max_records, max_bytes)?;
        pin.projection_validate(self, check)?;
        let mut out = ProjectionWriter {
            bytes: Vec::new(),
            limit: max_bytes,
            check,
            failed: None,
        };
        out.raw(b"{\"schema\":\"rz-pals-cpu7-view-evidence/1\",\"bundle_id\":")?;
        out.value(&original.bundle_id())?;
        out.raw(b",\"generation\":")?;
        out.value(&original.generation())?;
        out.raw(b",\"root\":")?;
        out.value(&StateWire::encode(
            original.root(),
            original.root_snapshot(),
        )?)?;
        out.raw(b",\"anchor\":")?;
        out.value(&StateWire::encode(anchor, original.anchor())?)?;
        out.raw(b",\"selected_len\":")?;
        out.value(&original.selected_len())?;
        out.raw(b",\"original_record_handle_digest\":")?;
        out.value(&original.digest())?;
        out.raw(b",\"view_binding\":")?;
        out.value(&binding)?;
        out.raw(b",\"view_digest\":")?;
        out.value(&view.digest())?;
        out.raw(b",\"view_seal\":")?;
        out.value(&view.seal_digest())?;
        out.raw(b",\"policy_digest\":")?;
        out.value(&view.policy_digest())?;
        out.raw(b",\"records\":[")?;
        for (index, record) in original.records().iter().enumerate() {
            if index != 0 {
                out.raw(b",")?;
            }
            out.raw(b"{\"record\":")?;
            out.value(&RoleRecordWire::encode(record.record())?)?;
            out.raw(b",\"evidence\":")?;
            out.value(&record.evidence())?;
            out.raw(b"}")?;
        }
        out.raw(b"],\"store\":")?;
        self.write_review_store_closure(pin, max_records, &mut out)?;
        out.raw(b"}")?;
        pin.projection_validate(self, &mut *out.check)?;
        (out.check)()?;
        Ok(out.bytes)
    }

    /// Only the pin's original owner and generation may project its existing
    /// dependency closure. No cold record is loaded, and no pin or authority
    /// is issued. `max_records` bounds the closure and `max_bytes` bounds the
    /// serialized output, not transient DTO allocations or process peak RSS.
    /// The supplied control checks must enforce the original deadline/cancel.
    pub(crate) fn project_review_store_closure(
        &self,
        pin: &StoreReadPin,
        max_records: usize,
        max_bytes: usize,
        check: &mut dyn FnMut() -> Result<(), StoreError>,
    ) -> Result<Vec<u8>, StoreError> {
        check()?;
        validate_projection_bounds(max_records, max_bytes)?;
        let mut out = ProjectionWriter {
            bytes: Vec::new(),
            limit: max_bytes,
            check,
            failed: None,
        };
        self.write_review_store_closure(pin, max_records, &mut out)?;
        Ok(out.bytes)
    }

    fn write_review_store_closure(
        &self,
        pin: &StoreReadPin,
        max_records: usize,
        out: &mut ProjectionWriter<'_>,
    ) -> Result<(), StoreError> {
        (out.check)()?;
        let seed = pin.projection_seed(self, max_records, &mut *out.check)?;
        if seed.observations.is_empty() {
            return Err(StoreError::InvalidEvidence(
                "review projection has no observations",
            ));
        }
        let closure =
            self.retirement_dependency_closure_controlled(&seed, max_records, &mut *out.check)?;
        out.raw(b"{\"schema\":\"rz-pals-review-store-projection/1\",\"generation\":")?;
        out.value(&self.generation())?;
        out.raw(b",\"states\":[")?;
        for (index, id) in closure.states.iter().enumerate() {
            if index != 0 {
                out.raw(b",")?;
            }
            out.value(&StateWire::encode(*id, self.states.get(*id)?)?)?;
        }
        out.raw(b"],\"lines\":[")?;
        for (index, id) in closure.lines.iter().enumerate() {
            if index != 0 {
                out.raw(b",")?;
            }
            out.value(&LineWire::encode(*id, self.lines.get(*id)?))?;
        }
        out.raw(b"],\"observations\":[")?;
        for (index, id) in closure.observations.iter().enumerate() {
            if index != 0 {
                out.raw(b",")?;
            }
            out.value(&(*id, ObservationWire::encode(self.observations.get(*id)?)?))?;
        }
        out.raw(b"],\"situations\":[")?;
        for (index, id) in closure.situations.iter().enumerate() {
            if index != 0 {
                out.raw(b",")?;
            }
            out.value(&(*id, self.situations.get(*id)?))?;
        }
        out.raw(b"],\"executions\":[")?;
        for (index, id) in closure.executions.iter().enumerate() {
            if index != 0 {
                out.raw(b",")?;
            }
            out.value(&(*id, self.tasks.get(*id)?))?;
        }
        out.raw(b"],\"dependencies\":[")?;
        let mut first = true;
        for (observation, situations) in self.retirement_dependency_edges() {
            (out.check)()?;
            for situation in situations {
                (out.check)()?;
                if closure.situations.contains(situation) {
                    if !closure.observations.contains(&observation) {
                        return Err(StoreError::InvalidEvidence("review dependency closure"));
                    }
                    if !first {
                        out.raw(b",")?;
                    }
                    first = false;
                    out.value(&(observation, *situation))?;
                }
            }
        }
        out.raw(b"]}")?;
        (out.check)()?;
        pin.projection_validate(self, &mut *out.check)?;
        (out.check)()?;
        Ok(())
    }
}
