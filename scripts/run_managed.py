#!/usr/bin/env python3
"""Run builds/tests with bounded owned cache and post-run scratch cleanup."""
from __future__ import annotations
import argparse
import json
import math
import shutil
import sys
from pathlib import Path
sys.dont_write_bytecode = True
from managed_storage import (ManagedBuild, StorageError, canonical_root, tree_size,
                             DEFAULT_OUTPUT_WARNING_BYTES)
from managed_process import run_group
from managed_outputs import sample_output_budget


def _emit_storage_failure_observation(error):
    """Additional bounded JSON only; original failure/cleanup owns the result."""
    try:
        observation = getattr(error, "managed_output_identity_observation", None)
        if (type(observation) is not dict
                or observation.get("schema") != "rovezero.managed-output-identity-observation.v1"):
            return
        raw = json.dumps(observation, sort_keys=True, separators=(",", ":"), allow_nan=False)
        if len(raw.encode("utf-8")) + 1 <= 32 << 10:
            print(raw, file=sys.stderr)
    except Exception:
        # A closed diagnostic pipe or encoding failure must not replace the
        # original StorageError, supervisor status or uncertain lease state.
        pass


def _emit_storage_cleanup_failure(error):
    """One bounded diagnostic; an unavailable pipe never replaces the first error."""
    try:
        # ensure_ascii can use twelve bytes per non-BMP character.
        observation = {"stage": "pre_dispatch_original_manager_finish",
                       "type": type(error).__name__[:128], "detail": str(error)[:1024]}
        raw = "managed_storage_cleanup_error=" + json.dumps(
            observation, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
        if len(raw.encode("utf-8")) + 1 <= 16 << 10:
            print(raw, file=sys.stderr)
    except BaseException:
        # The operation already failed. Diagnostic encoding/writing cannot
        # replace its original exception, including KeyboardInterrupt.
        pass


def _emit_custody_deadline_observation(error):
    """Report bounded timings without paths, content, or replacing the error."""
    try:
        observation = getattr(error, "custody_deadline_observation", None)
        if (type(observation) is not dict
                or set(observation) != {"schema", "stage", "elapsed_seconds", "window_seconds"}
                or observation["schema"] != "rovezero.custody-deadline-observation.v1"
                or observation["stage"] not in {"observation", "directory", "metadata_read",
                    "metadata_recheck", "mount_observation", "tree_snapshot", "observation_check",
                    "registry_inventory", "output_accounting"}
                or any(type(observation[key]) not in (int, float) or not math.isfinite(observation[key])
                       or observation[key] < 0
                       for key in ("elapsed_seconds", "window_seconds"))):
            return
        print("custody_deadline_observation=" + json.dumps(
            observation, sort_keys=True, separators=(",", ":"), allow_nan=False), file=sys.stderr)
    except BaseException:
        # Diagnostics must never alter process supervision or recovery.
        pass


def main(*, output_preservation=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument("--purpose", help="Stable diagnostic purpose slot; never a timestamp or PID")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or not 1 <= args.timeout <= 3600:
        parser.error("a command and timeout of 1..3600 seconds are required")
    # Windows CreateProcess does not perform POSIX PATH lookup for argv[0].
    program = shutil.which(command[0])
    if program is None or Path(program).suffix.lower() in {".bat", ".cmd"}:
        parser.error("command must resolve to an executable, not a shell batch file")
    command[0] = program
    budget_failure = None
    output_warning_seen = False

    def note_output_total(total):
        nonlocal output_warning_seen
        if type(total) is int and total > DEFAULT_OUTPUT_WARNING_BYTES:
            output_warning_seen = True

    try:
        manager = ManagedBuild(canonical_root(), args.source_root, purpose=args.purpose,
                               output_preservation=output_preservation)
        manager.acquire()
        try:
            environment = manager.environment()
            if output_preservation is not None:
                note_output_total(manager.sample_output_budget())
            else:
                output_scope = {"root": str(manager.root), "owner": manager._identity(),
                                "scope": "inherited_outer_process_group_or_job"}
                foreign_output_cache = {}
                manager._foreign_output_cache = foreign_output_cache
                note_output_total(sample_output_budget(supervisor_scope=output_scope,
                                                       foreign_cache=foreign_output_cache))
            # The same budget is still sampled while the child runs; the
            # earlier check is not a waiver for a later failure.
        except BaseException as error:
            # This try ends before run_group: no payload child was dispatched.
            # Keep the first failure if original-owner cleanup also fails.
            try:
                if output_preservation is None:
                    manager.finish(exit_code=125, tree_gone=True, _not_dispatched=True)
                else:
                    manager.finish(exit_code=125, tree_gone=True)
            except BaseException as cleanup_error:
                error.add_note("pre-dispatch original manager cleanup failed: "
                               + type(cleanup_error).__name__ + ": " + str(cleanup_error)[:2048])
                _emit_storage_cleanup_failure(cleanup_error)
            raise

        def check_budget():
            nonlocal budget_failure
            try:
                if output_preservation is None:
                    note_output_total(sample_output_budget(supervisor_scope=output_scope,
                                                           foreign_cache=foreign_output_cache))
                else:
                    note_output_total(manager.sample_output_budget())
            except StorageError as exc:
                # run_group preserves the original failure string while it
                # drains children. Keep this original instance for diagnostics
                # after that same cleanup; do not alter its type or message.
                budget_failure = exc
                raise
            if tree_size(manager.root / "slots", allow_disappearing=True) > manager.total_limit:
                raise StorageError("live managed build catalog exceeds 12 GiB")

        # Set before even attempting dispatch: a launcher failure with uncertain
        # child creation must never use the no-child cleanup shortcut.
        manager._dispatch_started = True
        code, gone, failure = run_group(command, manager.source, environment, args.timeout, check_budget)
        receipt = manager.finish(exit_code=code, tree_gone=gone, supervisor_error=failure)
        print("managed_storage=" + json.dumps(receipt, sort_keys=True), file=sys.stderr)
        if output_warning_seen and gone is True:
            # Run only after the original owner finalized and released its
            # guard. A later owner, uncertain physical state, or inaccessible
            # candidate is preserved by the maintenance routine. Maintenance
            # is observationally separate from the completed command.
            try:
                from managed_output_maintenance import cleanup_completed_builds
                cleanup = cleanup_completed_builds(manager, seconds=5)
                try:
                    print("managed_storage_maintenance=" + json.dumps(cleanup, sort_keys=True), file=sys.stderr)
                except (OSError, ValueError):
                    pass
            except Exception as error:
                try:
                    print("managed_storage_maintenance_warning=" + type(error).__name__, file=sys.stderr)
                except (OSError, ValueError):
                    pass
        if budget_failure is not None:
            _emit_storage_failure_observation(budget_failure)
            _emit_custody_deadline_observation(budget_failure)
        return code
    except (StorageError, OSError, RuntimeError) as exc:
        try:
            print(f"managed_storage_error={exc}; uncertain artifacts/leases are preserved", file=sys.stderr)
        except BaseException:
            # A closed diagnostic pipe must not turn the original failure into
            # an unrelated write exception or interrupt.
            pass
        if budget_failure is not None and budget_failure is not exc:
            _emit_storage_failure_observation(budget_failure)
            _emit_custody_deadline_observation(budget_failure)
        _emit_storage_failure_observation(exc)
        _emit_custody_deadline_observation(exc)
        return 125


if __name__ == "__main__":
    sys.exit(main())
