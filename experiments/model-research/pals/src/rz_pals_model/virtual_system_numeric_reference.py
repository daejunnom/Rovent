"""Explicit CPU reference for a separately registered V6 system CUDA plan.

This module never starts ORT or CUDA. A caller must provide an external finite
process supervisor; a timeout checked between tensor operations cannot preempt
a blocked native call. The result is reference math, not GPU acceptance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from . import numeric_capture as capture


def _pinned_json(path, expected_sha256):
    path = Path(path)
    if not path.is_absolute() or not capture._sha(expected_sha256):
        raise ValueError("absolute pinned JSON input required")
    raw = capture._regular_read(path, capture.MAX_FILE_BYTES)
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("JSON input digest differs")
    return capture._strict_json(raw)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan")
    plan.add_argument("--bindings", required=True)
    plan.add_argument("--bindings-sha256", required=True)
    plan.add_argument("--output", required=True)
    ref = sub.add_parser("reference")
    ref.add_argument("--plan", required=True)
    ref.add_argument("--plan-sha256", required=True)
    ref.add_argument("--checkpoint", required=True)
    ref.add_argument("--metadata")
    ref.add_argument("--export-directory", required=True)
    ref.add_argument("--output-directory", required=True)
    ref.add_argument("--max-seconds", type=int, default=600)
    args = parser.parse_args(argv)
    if args.command == "plan":
        output = Path(args.output)
        if not output.is_absolute():
            raise ValueError("absolute plan output path required")
        result = capture.write_system_case_plan(_pinned_json(args.bindings, args.bindings_sha256), output)
    else:
        paths = [args.checkpoint, args.export_directory, args.output_directory]
        if args.metadata is not None:
            paths.append(args.metadata)
        if any(not Path(path).is_absolute() for path in paths):
            raise ValueError("absolute reference input/output paths required")
        parsed = _pinned_json(args.plan, args.plan_sha256)
        capture.validate_system_case_plan(parsed)
        if not 1 <= args.max_seconds <= 3600:
            raise ValueError("finite reference window required")
        import torch
        from .artifacts import load_checkpoint
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        model, metadata = load_checkpoint(args.checkpoint, args.metadata)
        if "validator" not in parsed["bindings"]["graphs"]:
            model = model.virtual_review_pc_export_view()
        result = capture.capture_system_reference(
            model, metadata, args.export_directory, parsed, args.output_directory,
            checkpoint_path=args.checkpoint,
            expected_reference_source_sha256=parsed["bindings"]["source_components"]["model_reference"],
            max_seconds=args.max_seconds, execute=True,
        )
    # This separate CLI is outside the historical V5 source component roster.
    # Report its actual bytes independently rather than reusing a V5 pin.
    result["launcher_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
