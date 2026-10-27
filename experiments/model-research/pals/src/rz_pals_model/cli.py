"""Finite model/export commands and an explicit diagnostic optimizer smoke."""
import argparse
import json

from .config import DEFAULT_INIT_PROFILE, REGISTERED_PROFILES, ModelConfig, ROLES, matmul_flops


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init")
    init.add_argument("--output", required=True)
    init.add_argument("--seed", type=int, default=1)
    init.add_argument("--profile", choices=REGISTERED_PROFILES, default=DEFAULT_INIT_PROFILE)
    export = commands.add_parser("export")
    export.add_argument("--checkpoint", required=True)
    export.add_argument("--output", required=True)
    export.add_argument("--include-validator", action="store_true")
    export.add_argument("--layout", choices=("separate_pc", "shared_pc_if", "virtual_review_shared_pc_if_v1"), default="separate_pc")
    export.add_argument("--rules-profile-json")
    export.add_argument("--max-export-bytes", type=int,
                        help="optional serialized-write cap for virtual review export; excludes journals and logs")
    check = commands.add_parser("numeric-check")
    check.add_argument("--checkpoint", required=True)
    check.add_argument("--export", required=True)
    fixtures = commands.add_parser("rust-fixtures")
    fixtures.add_argument("--checkpoint", required=True)
    fixtures.add_argument("--export", required=True)
    fixtures.add_argument("--output", required=True)
    flops = commands.add_parser("flops")
    flops.add_argument("--role", choices=ROLES, default="proposer")
    flops.add_argument("--records", type=int, default=128)
    flops.add_argument("--candidates", type=int, default=32)
    flops.add_argument("--divergences", type=int, default=0)
    flops.add_argument("--batch", type=int, default=1)
    flops.add_argument("--profile", choices=REGISTERED_PROFILES, default="legacy_summary_v1")
    # Parser registration must remain usable on hosts without Torch/NumPy.
    smoke = commands.add_parser("train-step-smoke", help="finite diagnostic CPU optimizer/resume check")
    smoke.add_argument("--output", required=True)
    smoke.add_argument("--seed", type=int, default=17)
    # Diagnostic V1/V2 smoke stays separate from actual generation V3 training.
    from .config import PROFILES
    smoke.add_argument("--profile", choices=PROFILES, default=DEFAULT_INIT_PROFILE)
    smoke.add_argument("--coverage-collection")
    smoke.add_argument("--coverage-receipt-sha256")
    smoke.add_argument("--producer-registration-set")
    smoke.add_argument("--producer-registration-set-sha256")
    generation = commands.add_parser("train-generation", help="validate actual-generation data and recipe without updates")
    generation.add_argument("--spec", required=True)
    generation.add_argument("--dataset", required=True)
    generation.add_argument("--phase", choices=("pc", "v"), required=True)
    generation.add_argument("--frozen-pc-sha256")
    generation.add_argument("--model-epoch", type=int)
    args = parser.parse_args(argv)
    if args.command == "export" and args.max_export_bytes is not None:
        if args.max_export_bytes <= 0:
            parser.error("--max-export-bytes must be positive")
        if args.layout != "virtual_review_shared_pc_if_v1":
            parser.error("--max-export-bytes requires the virtual review layout")
    if args.command == "train-generation":
        # This branch remains usable without importing Torch or any optimizer.
        from .generation_training import main as validate_generation
        forwarded = ["--spec", args.spec, "--dataset", args.dataset, "--phase", args.phase]
        if args.frozen_pc_sha256 is not None:
            forwarded += ["--frozen-pc-sha256", args.frozen_pc_sha256]
        if args.model_epoch is not None:
            forwarded += ["--model-epoch", str(args.model_epoch)]
        return validate_generation(forwarded)
    if args.command == "flops":
        result = matmul_flops(ModelConfig.for_profile(args.profile), args.role, args.records, args.candidates, args.divergences, args.batch)
    else:
        import torch
        from .artifacts import ExportWriteBudget, export_checkpoint, initialize_checkpoint, numeric_check, rust_fixtures
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if args.command == "init": result = initialize_checkpoint(args.output, args.seed, args.profile)
        elif args.command == "export":
            options = {} if args.max_export_bytes is None else {"output_guard": ExportWriteBudget(args.max_export_bytes)}
            result = export_checkpoint(args.checkpoint, args.output, args.include_validator, args.layout, args.rules_profile_json, **options)
        elif args.command == "rust-fixtures": result = rust_fixtures(args.checkpoint, args.export, args.output)
        elif args.command == "train-step-smoke":
            from .nonzero_training import run_train_step_smoke
            result = run_train_step_smoke(args)
        else: result = numeric_check(args.checkpoint, args.export)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
