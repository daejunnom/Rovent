//! Source-only intake for expected cohort pin preparation. No collection,
//! native runtime, resource migration or trial is started by this CLI branch.
use super::*;
use rz_uci::rovent_bootstrap::launch::{
    ExpectedUtilityCohortPreparation, prepare_expected_utility_cohort_source_pins,
};

pub(super) fn parse_preparation_cli(
    arguments: Vec<String>,
) -> Result<RoventCollectionCliParse, ArenaError> {
    let (mut path, mut sha256, mut bytes) = (None, None, None);
    for argument in arguments {
        if let Some(value) = argument.strip_prefix("--prepare-utility-cohort-pins=") {
            check(
                path.replace(value.to_string()).is_none(),
                "duplicate source preparation plan",
            )?;
        } else if let Some(value) = argument.strip_prefix("--prepare-utility-cohort-pins-sha256=") {
            check(
                sha256.replace(value.to_string()).is_none(),
                "duplicate source preparation SHA",
            )?;
        } else if let Some(value) = argument.strip_prefix("--prepare-utility-cohort-pins-bytes=") {
            let value = value
                .parse::<u64>()
                .map_err(|_| invalid("source preparation plan bytes"))?;
            check(
                bytes.replace(value).is_none(),
                "duplicate source preparation bytes",
            )?;
        } else {
            return Err(invalid(
                "source-only preparation accepts no run, cohort, UCI or resource options",
            ));
        }
    }
    let plan = RoventPublicationFile {
        path: path.ok_or_else(|| invalid("explicit source preparation plan required"))?,
        sha256: sha256.ok_or_else(|| invalid("explicit source preparation SHA required"))?,
        bytes: bytes.ok_or_else(|| invalid("explicit source preparation bytes required"))?,
    };
    plan.validate()?;
    check(
        rz_experiments::pals_private_snapshot_path_v5(&plan.path) && plan.bytes <= 64 << 10,
        "bounded source preparation input namespace required",
    )?;
    Ok(RoventCollectionCliParse::PrepareCohortPins { plan })
}

pub fn prepare_rovent_utility_cohort_pins_from_cli(
    pin: &RoventPublicationFile,
) -> Result<Value, ArenaError> {
    pin.validate()?;
    check(
        rz_experiments::pals_private_snapshot_path_v5(&pin.path) && pin.bytes <= 64 << 10,
        "bounded source preparation pin required",
    )?;
    let input = Dir::open_ambient_dir(
        std::env::current_dir().map_err(io)?,
        cap_std::ambient_authority(),
    )
    .map_err(io)?;
    let raw = read_pin(&input, pin, 64 << 10)?;
    let request: ExpectedUtilityCohortPreparation = rz_experiments::decode_json(
        std::str::from_utf8(&raw).map_err(|_| invalid("source preparation JSON UTF8"))?,
    )?;
    let mut result =
        prepare_expected_utility_cohort_source_pins(&request, Arc::new(AtomicBool::new(false)))
            .map_err(|failure| invalid(failure.to_string()))?;
    result["preparation_plan"] =
        serde_json::to_value(pin).map_err(|failure| invalid(failure.to_string()))?;
    result["preparation_plan_read_bytes"] = json!(raw.len());
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn arguments() -> Vec<String> {
        vec![
            "--prepare-utility-cohort-pins=inputs/prepare.json".into(),
            format!("--prepare-utility-cohort-pins-sha256={}", "a".repeat(64)),
            "--prepare-utility-cohort-pins-bytes=101".into(),
        ]
    }
    #[test]
    fn source_preparation_is_a_separate_explicit_cli_action() {
        let parsed = parse_rovent_collection_cli(arguments()).unwrap();
        assert!(matches!(
            parsed,
            RoventCollectionCliParse::PrepareCohortPins { .. }
        ));
        for extra in [
            "--collection-plan=inputs/run.json",
            "--utility-cohort-plan=inputs/cohort.json",
            "--process-epoch=1",
        ] {
            let mut values = arguments();
            values.push(extra.into());
            assert!(parse_rovent_collection_cli(values).is_err());
        }
    }
    #[test]
    fn duplicate_missing_oversized_and_unowned_preparation_pins_are_rejected() {
        let mut values = arguments();
        values.push(values[0].clone());
        assert!(parse_rovent_collection_cli(values).is_err());
        let mut values = arguments();
        values.pop();
        assert!(parse_rovent_collection_cli(values).is_err());
        let mut values = arguments();
        values[0] = "--prepare-utility-cohort-pins=../prepare.json".into();
        assert!(parse_rovent_collection_cli(values).is_err());
        let mut values = arguments();
        values[2] = "--prepare-utility-cohort-pins-bytes=65537".into();
        assert!(parse_rovent_collection_cli(values).is_err());
    }
}
