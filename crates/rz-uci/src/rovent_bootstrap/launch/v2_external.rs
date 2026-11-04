//! Original /2 external researcher inputs, held for the whole UCI session.

use super::super::process_boundary::VerifiedOwnedRoventProcessBoundaryV2;
use super::*;

pub(super) struct ExternalLaunchV2 {
    pub(super) source: RoventPinnedExternalRSource<RoventInputLockV2>,
    pub(super) retirement: LaunchRetirementEvidence,
    // This is the actual provenance selection FD, not a reconstructed path.
    pub(super) selection_file: File,
}

pub(super) fn open_external_launch_v2(
    arguments: &Arguments,
    sources: &Arc<RoventPinnedNativeSources<RoventInputLockV2>>,
    process: &Arc<VerifiedOwnedRoventProcessBoundaryV2>,
    control: &RoventStartupControl,
) -> Result<Option<ExternalLaunchV2>, RoventLaunchError> {
    if !matches!(
        sources.configuration().fields().execution,
        ExecutionConfiguration::ExternalResearcher { .. }
    ) {
        if arguments.external_r_sources.is_some()
            || arguments.retirement_sink.is_some()
            || arguments.evidence_root.is_some()
        {
            return Err(error(
                "ExternalSources",
                "external source and retirement arguments require an external researcher",
            ));
        }
        return Ok(None);
    }

    sources
        .configuration()
        .external_question_limit()
        .map_err(|failure| error("ExternalQuestionPolicy", failure))?
        .ok_or_else(|| error("ExternalQuestionPolicy", "explicit external query required"))?;
    let path = arguments.external_r_sources.as_deref().ok_or_else(|| {
        error(
            "ExternalSources",
            "external researcher requires --external-r-sources",
        )
    })?;
    let registration = arguments.retirement_sink.as_deref().ok_or_else(|| {
        error(
            "RetirementEvidence",
            "external researcher requires --retirement-sink",
        )
    })?;
    let managed_root = arguments.evidence_root.as_deref().ok_or_else(|| {
        error(
            "RetirementEvidence",
            "external researcher requires --evidence-root",
        )
    })?;
    let matching: Vec<_> = process
        .input_lock()
        .manifest
        .provenance
        .iter()
        .filter(|artifact| artifact.path == path)
        .collect();
    if matching.len() != 1 {
        return Err(error(
            "ExternalSources",
            "selection is not a unique original provenance artifact",
        ));
    }
    let selection_file = matching[0]
        .open_verified_at(process.original_root(), MAX_SELECTION_BYTES)
        .map_err(|failure| error("ExternalSources", failure))?;
    let selection =
        ExternalSelectionWire::parse(&read_bounded(&selection_file, MAX_SELECTION_BYTES)?)?;
    let source = RoventPinnedExternalRSource::open(sources, &selection, control)
        .map_err(|failure| error("ExternalSources", failure))?;
    let retirement =
        LaunchRetirementEvidence::open_v2(Arc::clone(process), registration, managed_root, control)
            .map_err(|failure| error("RetirementEvidence", failure))?;
    Ok(Some(ExternalLaunchV2 {
        source,
        retirement,
        selection_file,
    }))
}
