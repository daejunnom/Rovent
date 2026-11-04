//! One-shot, read-only /2 intervention history inspection. This does not
//! issue current execution, physical completion, or training authority.

#[cfg(all(target_os = "linux", feature = "pals-collection-onnx"))]
fn main() -> std::process::ExitCode {
    use rz_arena::rovent_collect::{
        ROVENT_UTILITY_INTERVENTION_ACK_FILE, ROVENT_UTILITY_INTERVENTION_PUBLICATION_FILE,
        RoventHistoricalUtilityInterventionFiles, RoventPublicationFile,
    };
    use rz_arena::rovent_training_process::recovery::RoventRecoveryReadBudget;
    use std::sync::{Arc, atomic::AtomicBool};
    use std::time::{Duration, Instant};

    fn run() -> Result<(), Box<dyn std::error::Error>> {
        let args: Vec<String> = std::env::args().skip(1).collect();
        if args.len() != 7 {
            return Err("usage: rovent_intervention_cold_reader ROOT PUBLICATION_SHA256 PUBLICATION_BYTES ACK_SHA256 ACK_BYTES READ_BYTES WALL_MS".into());
        }
        let root = std::path::PathBuf::from(&args[0]);
        let publication = RoventPublicationFile {
            path: ROVENT_UTILITY_INTERVENTION_PUBLICATION_FILE.into(),
            sha256: args[1].clone(),
            bytes: args[2].parse()?,
        };
        let ack = RoventPublicationFile {
            path: ROVENT_UTILITY_INTERVENTION_ACK_FILE.into(),
            sha256: args[3].clone(),
            bytes: args[4].parse()?,
        };
        let maximum_bytes: u64 = args[5].parse()?;
        let wall_ms: u64 = args[6].parse()?;
        if wall_ms == 0 {
            return Err("wall budget must be positive".into());
        }
        let until = Instant::now()
            .checked_add(Duration::from_millis(wall_ms))
            .ok_or("wall budget overflow")?;
        let budget = Arc::new(RoventRecoveryReadBudget::new(
            maximum_bytes,
            until,
            Arc::new(AtomicBool::new(false)),
        )?);
        let original = RoventHistoricalUtilityInterventionFiles::open_retaining(
            root,
            publication,
            ack,
            Arc::clone(&budget),
        )?;
        let evidence = original.evidence().ok_or("reader returned no evidence")?;
        println!(
            "{}",
            serde_json::json!({
                "schema": "rz-rovent-intervention-cold-reader-observation/1",
                "scope": "historical_only",
                "cohort_id": evidence.cohort_id,
                "usage": evidence.usage,
                "role_queries": evidence.role_queries.len(),
                "known_pairs": evidence.known_pairs.len(),
                "all_pairs": evidence.all_pair_sha256.len(),
                "read_budget": budget.observation(),
                "training_authorized": false,
                "physical_completion_reissued": false,
            })
        );
        Ok(())
    }

    match run() {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error}");
            std::process::ExitCode::FAILURE
        }
    }
}

#[cfg(not(all(target_os = "linux", feature = "pals-collection-onnx")))]
fn main() -> std::process::ExitCode {
    eprintln!("requires Linux and pals-collection-onnx");
    std::process::ExitCode::FAILURE
}
