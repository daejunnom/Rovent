//! Separate, bounded NNUEC owner readback. Existing lifecycle/root receipts
//! keep their original JSON shape and never acquire authority from this row.
use super::*;
use crate::rovent_bootstrap::nnuec_session::diagnostics::NnuecOwnerObservation;

const DOMAIN: &str = "rz-rovent-nnuec-owner-observation/1";

#[derive(Serialize)]
struct NnuecRow {
    domain: &'static str,
    event: &'static str,
    input_lock_sha256: String,
    implementation_sha256: String,
    observed_after_authority: AuthorityObservation,
    source_status: &'static str,
    original: Option<NnuecOwnerObservation>,
    observation_read_failure: Option<&'static str>,
    details_complete: bool,
    physical_completion_attested: bool,
    capabilities_reissued: bool,
    execution_authorized: bool,
    performance_evidence: bool,
    full_acceptance_issued: bool,
}

impl RoventLifecycleDiagnostics {
    pub(in crate::rovent_bootstrap::product) fn observe_nnuec_owner(
        &self,
        product: &RoventNativeProductDriver,
        context: &SearchSessionContext,
    ) {
        let ActualProductDriver::Nnuec(driver) = product.inner.as_ref() else {
            return;
        };
        let original = driver.nnuec_owner_observation();
        let (original, source_status, observation_read_failure) = match original {
            Ok(Some(value)) if value.authority.matches(context.authority) => {
                (Some(value), "original_same_root_scalar_receipts", None)
            }
            Ok(Some(_)) => (
                None,
                "original_root_scope_mismatch",
                Some("nnuec_root_scope_mismatch"),
            ),
            Ok(None) => (None, "original_receipts_not_observed", None),
            Err(cause) => (None, "original_observation_unavailable", Some(cause)),
        };
        let details_complete = original.as_ref().is_some_and(|value| {
            value.preparation_scope_observed
                && value.preparation_details_omitted == Some(0)
                && value.preparation_records_omitted_by_manager == Some(0)
        });
        let mut row = NnuecRow {
            domain: DOMAIN,
            event: "after_original_root",
            input_lock_sha256: hex(self.lock),
            implementation_sha256: hex(self.implementation),
            observed_after_authority: context.authority.into(),
            source_status,
            original,
            observation_read_failure,
            details_complete,
            physical_completion_attested: false,
            capabilities_reissued: false,
            execution_authorized: false,
            performance_evidence: false,
            full_acceptance_issued: false,
        };
        let Ok(mut state) = self.state.lock() else {
            self.poisoned.store(true, Ordering::Release);
            return;
        };
        if state.root_runs > MAX_ROOT_ROWS {
            state.scope_incomplete = true;
            state
                .first_issue
                .get_or_insert("nnuec_diagnostic_row_limit");
            return;
        }
        if let Err(cause) = write_nnuec_row(&mut io::stderr().lock(), &mut row, &mut state) {
            state.scope_incomplete = true;
            state.first_issue.get_or_insert(cause);
        }
        drop(state);
        self.observe_nnuec_cpu_slices(product, context);
    }
}

fn encode_row(row: &mut NnuecRow) -> Result<RowBuffer, &'static str> {
    let mut buffer = RowBuffer(Vec::new());
    buffer
        .0
        .try_reserve_exact(MAX_ROW_BYTES)
        .map_err(|_| "nnuec_diagnostic_allocation_failed")?;
    loop {
        buffer.0.clear();
        match serde_json::to_writer(&mut buffer, &row) {
            Ok(()) if buffer.write_all(b"\n").is_ok() => return Ok(buffer),
            Ok(()) => {}
            Err(error) if error.is_io() => {}
            Err(_) => return Err("nnuec_diagnostic_encoding_failed"),
        }
        let Some(original) = row.original.as_mut() else {
            return Err("nnuec_diagnostic_row_byte_limit");
        };
        if original.preparations.pop().is_none() {
            return Err("nnuec_diagnostic_row_byte_limit");
        }
        // Keep the original ordered prefix and report every omitted detail.
        // Never turn omitted records or unobserved receipts into zero work.
        original.preparation_details_omitted = original
            .preparation_details_omitted
            .and_then(|count| count.checked_add(1));
        row.details_complete = false;
    }
}

fn write_nnuec_row(
    output: &mut impl Write,
    row: &mut NnuecRow,
    state: &mut State,
) -> Result<(), &'static str> {
    let buffer = encode_row(row)?;
    if buffer.0.len() > (MAX_STREAM_BYTES - MAX_ROW_BYTES).saturating_sub(state.bytes) {
        return Err("nnuec_diagnostic_stream_byte_limit");
    }
    // Account a failed/partial OS write once. Do not retry a torn JSONL row.
    state.bytes += buffer.0.len();
    output
        .write_all(&buffer.0)
        .map_err(|_| "nnuec_diagnostic_write_failed")?;
    output.flush().map_err(|_| "nnuec_diagnostic_flush_failed")
}

#[cfg(test)]
mod tests {
    use super::*;
    fn row() -> NnuecRow {
        NnuecRow {
            domain: DOMAIN,
            event: "after_original_root",
            input_lock_sha256: "1".repeat(64),
            implementation_sha256: "2".repeat(64),
            observed_after_authority: AuthorityObservation {
                epoch: 1,
                game: 2,
                root: 3,
                implementation_sha256: "3".repeat(64),
            },
            source_status: "original_receipts_not_observed",
            original: None,
            observation_read_failure: None,
            details_complete: false,
            physical_completion_attested: false,
            capabilities_reissued: false,
            execution_authorized: false,
            performance_evidence: false,
            full_acceptance_issued: false,
        }
    }
    #[test]
    fn separate_domain_and_unknown_do_not_issue_physical_or_semantic_authority() {
        let encoded = encode_row(&mut row()).unwrap();
        let value: serde_json::Value = serde_json::from_slice(&encoded.0).unwrap();
        assert_eq!(value["domain"], DOMAIN);
        assert_ne!(value["domain"], ROVENT_LIFECYCLE_DIAGNOSTIC_DOMAIN);
        assert!(value["original"].is_null());
        for field in [
            "details_complete",
            "physical_completion_attested",
            "capabilities_reissued",
            "execution_authorized",
            "performance_evidence",
            "full_acceptance_issued",
        ] {
            assert_eq!(value[field], false, "{field}");
        }
        assert!(encoded.0.len() <= MAX_ROW_BYTES);
    }
    #[test]
    fn stream_limit_preserves_the_final_lifecycle_row_reservation() {
        let mut state = State {
            bytes: MAX_STREAM_BYTES - MAX_ROW_BYTES,
            ..State::default()
        };
        let mut output = Vec::new();
        assert_eq!(
            write_nnuec_row(&mut output, &mut row(), &mut state),
            Err("nnuec_diagnostic_stream_byte_limit")
        );
        assert!(output.is_empty());
        assert_eq!(state.bytes, MAX_STREAM_BYTES - MAX_ROW_BYTES);
    }
}
