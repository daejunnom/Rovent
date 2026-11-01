//! Emit the existing original CPU-slice handle, separately from owner totals.
//! All output shares lifecycle byte/row bounds and retains the final row seat.
use super::*;
use crate::rovent_bootstrap::nnuec_session::diagnostics::cpu_slices::{
    CpuSliceOwnerObservation, CpuSliceSummary, SettledSlice,
};

const DOMAIN: &str = "rz-rovent-nnuec-cpu-slice-observation/1";
mod chain;

#[derive(Serialize)]
struct SliceRow<'a> {
    domain: &'static str,
    event: &'static str,
    input_lock_sha256: String,
    implementation_sha256: String,
    observed_after_authority: AuthorityObservation,
    root_admission_sha256: Option<&'a str>,
    source_status: &'static str,
    snapshot_read_failure: Option<&'static str>,
    summary: Option<&'a CpuSliceSummary>,
    settled_slice: Option<&'a SettledSlice>,
    physical_completion_attested: bool,
    token_or_job_authority_issued: bool,
    accepted_critic_chain_observed: Option<bool>,
    conditional_scalar_value_observed: Option<bool>,
    performance_evidence: bool,
    full_acceptance_issued: bool,
}
impl RoventLifecycleDiagnostics {
    pub(in crate::rovent_bootstrap::product::diagnostics) fn observe_nnuec_cpu_slices(
        &self,
        product: &RoventNativeProductDriver,
        context: &SearchSessionContext,
    ) {
        let ActualProductDriver::Nnuec(driver) = product.inner.as_ref() else {
            return;
        };
        let snapshot = driver.nnuec_cpu_slice_observation();
        let (snapshot, status, failure) = match snapshot {
            Ok(Some(value))
                if value.summary.authority.matches(context.authority)
                    && value.summary.original_root_scope_matches == Some(false) =>
            {
                (
                    Some(value),
                    "original_handle_root_scope_mismatch",
                    Some("cpu_slice_handle_scope_mismatch"),
                )
            }
            Ok(Some(value)) if value.summary.authority.matches(context.authority) => (
                Some(value),
                "original_same_root_settled_receipt_prefix",
                None,
            ),
            Ok(Some(_)) => (
                None,
                "original_root_scope_mismatch",
                Some("cpu_slice_root_scope_mismatch"),
            ),
            Ok(None) => (None, "original_slice_scope_not_observed", None),
            Err(cause) => (None, "original_slice_snapshot_unavailable", Some(cause)),
        };
        let mut row = SliceRow {
            domain: DOMAIN,
            event: "root_summary",
            input_lock_sha256: hex(self.lock),
            implementation_sha256: hex(self.implementation),
            observed_after_authority: context.authority.into(),
            root_admission_sha256: snapshot
                .as_ref()
                .map(|s| s.summary.root_admission_sha256.as_str()),
            source_status: status,
            snapshot_read_failure: failure,
            summary: snapshot.as_ref().map(|s| &s.summary),
            settled_slice: None,
            physical_completion_attested: false,
            token_or_job_authority_issued: false,
            accepted_critic_chain_observed: None,
            conditional_scalar_value_observed: None,
            performance_evidence: false,
            full_acceptance_issued: false,
        };
        #[cfg(feature = "native-validation-diagnostics")]
        let reference = snapshot.as_ref().and_then(|s| {
            let (sequence, input) = s.reference_input.as_ref()?;
            let original = s.records.iter().find(|r| r.sequence() == *sequence)?;
            Some(ReferencePending {
                header: serde_json::json!({
                    "domain":"rz-rovent-nnuec-reference-input-observation/1",
                    "event":"optional_input_after_original_session_finish",
                    "input_lock_sha256":hex(self.lock), "implementation_sha256":hex(self.implementation),
                    "root_admission_sha256":s.summary.root_admission_sha256,
                    "source_status":status, "original_settled_slice":original,
                    "scalar":input.scalar.map(|v|serde_json::json!({
                        "binding_sha256":hex(v.binding),"full_view_sha256":hex(Digest(v.full_view)),
                        "view_seal_sha256":hex(Digest(v.view_seal)),"record_scope_sha256":hex(Digest(v.record_scope)),
                        "generation":v.generation,"raw_value_f32_bits":v.raw_value_bits,
                        "scaled_value_f32_bits":v.scaled_value_bits,"returned_frontier_score_i32":v.returned_frontier_score})),
                    "capture_maximum_input_bytes":input.maximum_bytes,
                    "unavailable":input.unavailable.map(|v|v.code()),
                    "execution_authorized":false,"physical_completion_attested":false,
                    "full_acceptance_issued":false,"performance_evidence":false
                }),
                input: Arc::clone(input),
            })
        });
        let Ok(mut state) = self.state.lock() else {
            self.poisoned.store(true, Ordering::Release);
            return;
        };
        #[cfg(feature = "native-validation-diagnostics")]
        if state.reference_input.is_none() { state.reference_input = reference; }
        if state.root_runs > MAX_ROOT_ROWS {
            state.scope_incomplete = true;
            state
                .first_issue
                .get_or_insert("nnuec_cpu_slice_root_limit");
            return;
        }
        let result = (|| {
            write_slice_row(&mut io::stderr().lock(), &row, &mut state)?;
            row.summary = None;
            row.event = "settled_slice";
            if let Some(CpuSliceOwnerObservation { records, .. }) = snapshot.as_ref() {
                // The source cap is checked before projection; no sorting,
                // coalescing, resume inference or post-failure retry occurs.
                for record in records {
                    row.settled_slice = Some(record);
                    write_slice_row(&mut io::stderr().lock(), &row, &mut state)?;
                    #[cfg(feature = "native-validation-diagnostics")]
                    if let Some(actual) = record.nnuec_scalar() {
                        // Separate raw domain; the /1 slice and ACK rows are unchanged.
                        // The same existing stream cap and reserved final seat apply.
                        let scalar = serde_json::json!({
                            "domain":"rz-rovent-nnuec-native-scalar-slice-observation/1",
                            "event":"original_settled_slice_numeric_observation",
                            "input_lock_sha256":hex(self.lock),"implementation_sha256":hex(self.implementation),
                            "root_admission_sha256":row.root_admission_sha256,"source_status":status,
                            "original_settled_slice":record,"numeric":actual,
                            "accepted_critic_chain_observed":null,"conditional_scalar_value_observed":null,
                            "physical_completion_attested":false,"execution_authorized":false,
                            "token_or_job_authority_issued":false,"performance_evidence":false,"full_acceptance_issued":false});
                        write_slice_row(&mut io::stderr().lock(), &scalar, &mut state)?;
                    }
                }
            }
            Ok::<(), &'static str>(())
        })();
        if let Err(cause) = result {
            state.scope_incomplete = true;
            state.first_issue.get_or_insert(cause);
        }
        drop(state);
        self.observe_nnuec_chain(context, status, failure, snapshot.as_ref());
    }
}
fn write_slice_row<T: Serialize>(
    output: &mut impl Write,
    row: &T,
    state: &mut State,
) -> Result<(), &'static str> {
    let mut buffer = RowBuffer(Vec::new());
    buffer
        .0
        .try_reserve_exact(MAX_ROW_BYTES)
        .map_err(|_| "nnuec_cpu_slice_allocation_failed")?;
    serde_json::to_writer(&mut buffer, row)
        .map_err(|_| "nnuec_cpu_slice_row_encoding_or_byte_limit")?;
    buffer
        .write_all(b"\n")
        .map_err(|_| "nnuec_cpu_slice_row_byte_limit")?;
    if buffer.0.len() > (MAX_STREAM_BYTES - MAX_ROW_BYTES).saturating_sub(state.bytes) {
        return Err("nnuec_cpu_slice_stream_byte_limit");
    }
    state.bytes += buffer.0.len();
    output
        .write_all(&buffer.0)
        .map_err(|_| "nnuec_cpu_slice_write_failed")?;
    output.flush().map_err(|_| "nnuec_cpu_slice_flush_failed")
}

#[cfg(feature = "native-validation-diagnostics")]
pub(super) struct ReferencePending {
    header: serde_json::Value,
    input: Arc<rz_search::cpu_value::nnuec::v2::scalar_diagnostic::ReferenceInputSnapshot>,
}
#[cfg(feature = "native-validation-diagnostics")]
impl RoventLifecycleDiagnostics {
    /// This optional output is last. Its failure cannot take a required row's
    /// seat or change the original protocol/finish/diagnostic Result.
    pub(super) fn flush_reference_input(&self, state: &mut State) {
        let Some(pending) = state.reference_input.take() else { return; };
        let raw = pending.input.input_json.as_deref().map(str::as_bytes);
        let available = MAX_STREAM_BYTES.saturating_sub(state.bytes);
        let make_rows = || -> Option<Vec<Vec<u8>>> {
            use sha2::{Digest as _, Sha256};
            let total = raw.map_or(1, |b| b.len().div_ceil(1024).max(1));
            let digest = raw.map(|b| format!("{:x}", Sha256::digest(b)));
            let mut rows = Vec::new();
            rows.try_reserve_exact(total).ok()?;
            let mut bytes = 0usize;
            for index in 0..total {
                let mut value = pending.header.clone();
                let data = raw.map(|b| &b[index * 1024..b.len().min((index + 1) * 1024)]);
                value["reference_input"] = serde_json::json!({
                    "status":if raw.is_some(){"captured"}else{"unavailable"},
                    "encoding":"original_utf8_hex_chunks", "bytes":raw.map(|v|v.len()),
                    "raw_sha256":digest,"chunk_index":index,"chunks":total,
                    "data_hex":data.map(|b|b.iter().map(|x|format!("{x:02x}")).collect::<String>())
                });
                let mut encoded = serde_json::to_vec(&value).ok()?;
                encoded.push(b'\n');
                bytes = bytes.checked_add(encoded.len())?;
                if encoded.len() > MAX_ROW_BYTES || bytes > available { return None; }
                rows.push(encoded);
            }
            Some(rows)
        };
        let rows = make_rows().unwrap_or_else(|| {
            let mut header = pending.header.clone();
            header["reference_input"] = serde_json::json!({
                "status":"unavailable","reason":"optional_output_capacity_or_encoding",
                "bytes":raw.map(|v|v.len())
            });
            serde_json::to_vec(&header).ok().and_then(|mut row| {
                row.push(b'\n');
                (row.len() <= MAX_ROW_BYTES && row.len() <= available).then_some(vec![row])
            }).unwrap_or_default()
        });
        let mut output = io::stderr().lock();
        for row in rows {
            if output.write_all(&row).is_err() { break; }
            state.bytes += row.len();
        }
        let _ = output.flush();
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn missing_original_scope_remains_unknown_and_cannot_issue_acceptance() {
        let row = SliceRow {
            domain: DOMAIN,
            event: "root_summary",
            input_lock_sha256: "1".repeat(64),
            implementation_sha256: "2".repeat(64),
            observed_after_authority: AuthorityObservation {
                epoch: 1,
                game: 2,
                root: 3,
                implementation_sha256: "3".repeat(64),
            },
            root_admission_sha256: None,
            source_status: "original_slice_scope_not_observed",
            snapshot_read_failure: None,
            summary: None,
            settled_slice: None,
            physical_completion_attested: false,
            token_or_job_authority_issued: false,
            accepted_critic_chain_observed: None,
            conditional_scalar_value_observed: None,
            performance_evidence: false,
            full_acceptance_issued: false,
        };
        let mut state = State::default();
        let mut output = Vec::new();
        write_slice_row(&mut output, &row, &mut state).unwrap();
        let raw: serde_json::Value = serde_json::from_slice(&output).unwrap();
        assert!(raw["summary"].is_null());
        assert!(raw["settled_slice"].is_null());
        assert!(raw["accepted_critic_chain_observed"].is_null());
        assert!(raw["conditional_scalar_value_observed"].is_null());
        assert_eq!(raw["physical_completion_attested"], false);
        assert_eq!(raw["token_or_job_authority_issued"], false);
        assert_eq!(raw["performance_evidence"], false);
        assert!(output.len() <= MAX_ROW_BYTES);
        state.bytes = MAX_STREAM_BYTES - MAX_ROW_BYTES;
        let mut rejected = Vec::new();
        assert_eq!(
            write_slice_row(&mut rejected, &row, &mut state),
            Err("nnuec_cpu_slice_stream_byte_limit")
        );
        assert!(rejected.is_empty());
    }
}
