#!/usr/bin/env python3
"""Run builds/tests with bounded owned cache and post-run scratch cleanup."""
from __future__ import annotations
import argparse
from contextlib import ExitStack, contextmanager
import json
import math
import os
import shutil
import sys
from pathlib import Path
sys.dont_write_bytecode = True
from managed_storage import (DEFAULT_OUTPUT_WARNING_BYTES, ManagedBuild, StorageError,
                             canonical_root, checked)
from managed_process import run_group, supervisor_signal_scope
from managed_outputs import sample_output_budget


def _consumer_registry_options(root, registry_id, required):
    """Resolve an explicitly configured, pinned participating registry.

    Missing/invalid registry setup defers deletion; it does not cancel an
    unrelated command. The registration scope is never implied to cover legacy
    or unregistered consumers.
    """
    root = root if root is not None else os.environ.get("RZ_CONSUMER_REGISTRY_ROOT")
    registry_id = registry_id if registry_id is not None else os.environ.get("RZ_CONSUMER_REGISTRY_ID")
    inherited_required = os.environ.get("RZ_CONSUMER_REGISTRY_REQUIRED") == "1"
    required = bool(required or inherited_required or root is not None or registry_id is not None)
    if not required:
        return {}
    registry = None
    if root and registry_id:
        try:
            from storage_consumer_registry import ConsumerRegistry
            registry = ConsumerRegistry(root, registry_id=registry_id)
        except (StorageError, OSError, ValueError):
            registry = None
    if registry is None:
        try:
            print("managed_storage_warning=consumer_registry_unavailable; "
                  "participating-domain cleanup will be deferred", file=sys.stderr)
        except Exception:
            pass
    return {"dependency_registry": registry, "dependency_registry_required": True}


@contextmanager
def _participating_catalogs(registry, catalogs):
    if registry is None or not catalogs or len(catalogs) > 16:
        raise StorageError("pinned registry and participating catalogs required")
    roots = sorted({checked(path) for path in catalogs}, key=lambda p: os.path.normcase(str(p)))
    from storage_guard import CatalogGuard
    from storage_consumer_registry import require_catalog_binding
    with ExitStack() as stack:
        for root in roots:
            if not root.is_dir():
                raise StorageError("existing participating catalog required")
            guard = stack.enter_context(CatalogGuard(root / "catalog.lock"))
            require_catalog_binding(root, registry, guard)
        yield roots


def _register_cpu_validator_inputs(registry, inputs, future_inputs, catalogs, command, source_root):
    """Pin explicitly declared CPU-validator inputs before any child dispatch.

    Only explicitly named catalogs are checked. This is not proof that an
    unregistered catalog or a nonparticipating process does not exist.
    """
    if registry is None or not (inputs or future_inputs):
        raise StorageError("pinned registry, inputs and participating catalogs required")
    if len(command) < 2 or os.path.normcase(os.path.realpath(command[0])) != os.path.normcase(os.path.realpath(sys.executable)):
        raise StorageError("CPU input registration supports the original Python interpreter only")
    requested_script = Path(command[1])
    if not requested_script.is_absolute():
        requested_script = Path(source_root) / requested_script
    expected_script = checked(Path(source_root) / "scripts" / "pals_model_validate.py")
    if checked(requested_script) != expected_script or not expected_script.is_file():
        raise StorageError("CPU input registration supports the PALS CPU validator only")
    with _participating_catalogs(registry, catalogs) as roots:
        return registry.register(inputs, future_inputs=future_inputs), roots


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
    # Keep catchable stop signals as a request until the original manager has
    # drained its child and finalized. The context restores caller handlers.
    with supervisor_signal_scope() as interruption:
        return _main(output_preservation=output_preservation, interruption=interruption)


def _main(*, output_preservation=None, interruption=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument("--purpose", help="Stable diagnostic purpose slot; never a timestamp or PID")
    parser.add_argument("--consumer-registry-root", type=Path,
                        help="Existing shared input-consumer registry; use with its pinned ID")
    parser.add_argument("--consumer-registry-id", help="Pinned consumer registry identity")
    parser.add_argument("--consumer-registry-required", action="store_true",
                        help="Preserve cleanup candidates if the registry is unavailable")
    parser.add_argument("--consumer-input", type=Path, action="append", default=[],
                        help="Input protected by a supervisor lease; repeat per input")
    parser.add_argument("--consumer-future-input", type=Path, action="append", default=[],
                        help="Not-yet-created input path protected by the same lease")
    parser.add_argument("--consumer-input-catalog", type=Path, action="append", default=[],
                        help="Participating catalog bound to the same registry; repeat per catalog")
    parser.add_argument("--consumer-input-cpu-only", action="store_true",
                        help="Declare the PALS validator CPU-only for supervisor cleanup")
    parser.add_argument("--consumer-broker-reviewed-sha256",
                        help="Opt in to a reviewed reader scope bound to the actual script SHA-256")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or not 1 <= args.timeout <= 3600:
        parser.error("a command and timeout of 1..3600 seconds are required")
    if args.consumer_broker_reviewed_sha256 is not None:
        digest = args.consumer_broker_reviewed_sha256
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            parser.error("reviewed reader SHA-256 must be 64 lowercase hexadecimal digits")
        if args.consumer_input or args.consumer_future_input or args.consumer_input_cpu_only:
            parser.error("brokered readers and the separate CPU validator profile cannot be combined")
    # Windows CreateProcess does not perform POSIX PATH lookup for argv[0].
    program = shutil.which(command[0])
    if program is None or Path(program).suffix.lower() in {".bat", ".cmd"}:
        parser.error("command must resolve to an executable, not a shell batch file")
    command[0] = program
    budget_failure = None
    consumer_lease = None
    verified_catalogs = []
    consumer_dispatched = False
    consumer_closed = False
    broker_launch = None
    try:
        if interruption is not None and interruption["signal"] is not None:
            return 128 + int(interruption["signal"])
        dependency_options = _consumer_registry_options(args.consumer_registry_root,
                                                        args.consumer_registry_id,
                                                        args.consumer_registry_required)
        if args.consumer_input or args.consumer_future_input or args.consumer_input_cpu_only:
            if not args.consumer_input_cpu_only:
                raise StorageError("explicit CPU-only PALS validator profile required for input lease")
            consumer_lease, verified_catalogs = _register_cpu_validator_inputs(
                dependency_options.get("dependency_registry"), args.consumer_input,
                args.consumer_future_input, args.consumer_input_catalog, command, args.source_root)
        elif args.consumer_input_catalog:
            with _participating_catalogs(dependency_options.get("dependency_registry"),
                                         args.consumer_input_catalog) as roots:
                verified_catalogs = list(roots)
        catalog_environment = json.dumps([str(path) for path in verified_catalogs], ensure_ascii=True)
        if len(catalog_environment.encode("ascii")) > 8192:
            raise StorageError("participating catalog environment exceeds bounded size")
        manager = ManagedBuild(canonical_root(), args.source_root, purpose=args.purpose,
                               output_preservation=output_preservation, **dependency_options)
        manager.acquire()
        try:
            environment = manager.environment()
            if verified_catalogs:
                environment["RZ_CONSUMER_INPUT_CATALOGS"] = catalog_environment
            if consumer_lease is not None:
                environment["RZ_CONSUMER_LEASE_OWNER_ID"] = consumer_lease.owner_id
                environment["CUDA_VISIBLE_DEVICES"] = "-1"
            if args.consumer_broker_reviewed_sha256 is not None:
                from storage_consumer_broker_launch import ManagedBrokerLaunch
                from storage_consumer_broker_profile import ObservedReaderProfile, ReviewedReaderContract
                profile = ObservedReaderProfile(manager, command)
                try:
                    contract = ReviewedReaderContract(
                        profile, reviewed_sha256=args.consumer_broker_reviewed_sha256,
                        descriptor_content_only=True, derived_content_after_ack=True)
                    broker_launch = ManagedBrokerLaunch(manager, verified_catalogs, reader_contract=contract)
                except BaseException as error:
                    try:
                        profile.close()
                    except BaseException as close_error:
                        error.add_note("reader profile close unknown: " + type(close_error).__name__)
                    raise
                environment = broker_launch.environment()
                try:
                    print("managed_consumer_broker_profile=" + json.dumps(profile.observation(), sort_keys=True),
                          file=sys.stderr)
                except Exception:
                    pass
            if output_preservation is not None:
                manager.sample_output_budget()
            else:
                output_scope = {"root": str(manager.root), "owner": manager._identity(),
                                "scope": "inherited_outer_process_group_or_job"}
                foreign_output_cache = {}
                manager._foreign_output_cache = foreign_output_cache
                sample_output_budget(supervisor_scope=output_scope,
                                     foreign_cache=foreign_output_cache)
            # The same budget is still sampled while the child runs; the
            # earlier check is not a waiver for a later failure.
        except BaseException as error:
            # This try ends before run_group: no payload child was dispatched.
            # Keep the first failure if original-owner cleanup also fails.
            if broker_launch is not None:
                try:
                    broker_launch.finish(tree_gone=False)
                except BaseException as cleanup_error:
                    error.add_note("pre-dispatch broker cleanup unknown: " + type(cleanup_error).__name__)
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

        def sample_budget():
            nonlocal budget_failure
            # Aggregate storage is observed separately. Reading its bounded
            # cache cannot delay the child with a full build-tree traversal;
            # a missing or failed advisory observation never cancels the run.
            try:
                from rovezero_usage_snapshot import warn_cached_total_usage
                warn_cached_total_usage()
            except Exception:
                pass
            try:
                if output_preservation is None:
                    sample_output_budget(supervisor_scope=output_scope,
                                         foreign_cache=foreign_output_cache)
                else:
                    manager.sample_output_budget()
            except StorageError as exc:
                # run_group preserves the original failure string while it
                # drains children. Keep this original instance for diagnostics
                # after that same cleanup; do not alter its type or message.
                budget_failure = exc
                raise
        budget_gate = None
        if broker_launch is not None:
            from storage_consumer_broker_budget import BrokerBudgetGate
            budget_gate = BrokerBudgetGate(
                sample_budget,
                lambda: broker_launch.broker is not None and broker_launch.broker.transaction_active)

        def check_budget():
            if budget_gate is None:
                sample_budget()
            else:
                budget_gate.sample()

        def broker_tick():
            broker_launch.on_tick()
            budget_gate.after_tick()
        if interruption is not None and interruption["signal"] is not None:
            # This branch precedes any spawn attempt, so the original manager
            # can finalize its own lease without inventing child evidence.
            exit_code = 128 + int(interruption["signal"])
            if broker_launch is not None:
                broker_launch.finish(tree_gone=False)
            if output_preservation is None:
                receipt = manager.finish(exit_code=exit_code, tree_gone=True, _not_dispatched=True)
            else:
                receipt = manager.finish(exit_code=exit_code, tree_gone=True)
            try:
                print("managed_storage=" + json.dumps(receipt, sort_keys=True), file=sys.stderr)
            except BaseException:
                pass
            return exit_code
        if broker_launch is not None:
            try:
                launch_command = broker_launch.command
                broker_launch.assert_spawn_command(launch_command, manager.source)
                child_fds = broker_launch.child_fds
                broker_launch.mark_dispatch_attempted()
            except BaseException as error:
                try:
                    broker_launch.finish(tree_gone=False)
                except BaseException as cleanup_error:
                    error.add_note("pre-dispatch broker cleanup unknown: " + type(cleanup_error).__name__)
                try:
                    if output_preservation is None:
                        manager.finish(exit_code=125, tree_gone=True, _not_dispatched=True)
                    else:
                        manager.finish(exit_code=125, tree_gone=True)
                except BaseException as cleanup_error:
                    error.add_note("pre-dispatch manager cleanup unknown: " + type(cleanup_error).__name__)
                raise
        # Set before even attempting dispatch: a launcher failure with uncertain
        # child creation must never use the no-child cleanup shortcut.
        manager._dispatch_started = True
        consumer_dispatched = True
        if broker_launch is None:
            code, gone, failure = run_group(command, manager.source, environment, args.timeout, check_budget)
        else:
            from managed_process import ProcessNotSpawned
            try:
                code, gone, failure = run_group(
                    launch_command, manager.source, environment, args.timeout, check_budget,
                    pass_fds=child_fds, on_spawn=broker_launch.on_spawn, on_tick=broker_tick,
                    owned_group=broker_launch.group, on_stop=broker_launch.on_stop,
                    verify_before_spawn=True)
            except ProcessNotSpawned as error:
                try:
                    broker_launch.mark_pre_dispatch_failure(error)
                    broker_launch.finish(tree_gone=False)
                except BaseException as cleanup_error:
                    error.add_note("pre-dispatch broker cleanup unknown: " + type(cleanup_error).__name__)
                manager._dispatch_started = False
                consumer_dispatched = False
                try:
                    if output_preservation is None:
                        manager.finish(exit_code=125, tree_gone=True, _not_dispatched=True)
                    else:
                        manager.finish(exit_code=125, tree_gone=True)
                except BaseException as cleanup_error:
                    error.add_note("pre-dispatch manager cleanup unknown: " + type(cleanup_error).__name__)
                raise
            except BaseException as error:
                # No broker exists until the original group's Popen object is
                # bound. If Popen itself raised, close only this owner's unserved
                # channel/pins; child creation and manager owner remain unknown.
                if broker_launch.group._child is None:
                    try:
                        broker_launch.revoke_unconfirmed_spawn()
                        try:
                            print("managed_consumer_broker_spawn=" + json.dumps({
                                "spawn": "unknown", "unserved_channel": "revoked",
                                "manager_owner": "preserved", "physical_completion": "unknown",
                            }, sort_keys=True), file=sys.stderr)
                        except Exception:
                            pass
                    except BaseException as cleanup_error:
                        error.add_note("unconfirmed spawn broker close unknown: " + type(cleanup_error).__name__)
                        try:
                            print("managed_consumer_broker_close_warning=" + type(cleanup_error).__name__,
                                  file=sys.stderr)
                        except Exception:
                            pass
                raise
            if not broker_launch.finish(tree_gone=gone):
                raise StorageError("broker original group or reader scope remains unknown; preserve owner")
            try:
                print("managed_consumer_broker_observation=" + json.dumps({
                    "registered_scope": "participating_bound_catalogs_only",
                    "reader_source": "observed_script_only",
                    "imports": "unknown", "same_descriptor_fd": "unknown",
                    "nonparticipating_writers": "unknown", "physical_completion": "unknown",
                    "original_group_gone": gone,
                    "child_cwd_post_spawn": broker_launch.child_cwd_observation,
                    "deferred_budget_count": budget_gate.deferred_count,
                    "deferred_budget_pending_before_final_check": budget_gate.deferred,
                    "transport_failure_type": broker_launch.first_transport_failure,
                }, sort_keys=True), file=sys.stderr)
            except Exception:
                pass
            final_budget_status = "no_deferred_check"
            try:
                if budget_gate.finish():
                    final_budget_status = "deferred_check_completed"
            except Exception as error:
                final_budget_status = "deferred_check_failed_" + type(error).__name__
                if failure is None:
                    failure = "post-broker deferred budget check failed: " + type(error).__name__
                if code == 0:
                    code = 125
            try:
                print("managed_consumer_broker_budget=" + json.dumps({
                    "status": final_budget_status,
                    "group_scope": "gone" if gone else "unknown",
                    "final_peak_attested": False,
                }, sort_keys=True), file=sys.stderr)
            except Exception:
                pass
        if consumer_lease is not None and gone is True:
            # The validator can invoke a caller-provided binary. Process-tree
            # termination and CUDA_VISIBLE_DEVICES do not prove that every
            # dispatched device operation physically completed. Preserve the
            # lease until an independent original-owner/fence path exists.
            try:
                consumer_lease.retain_unknown(tree_gone=True)
                print("managed_consumer_lease_status=physical_completion_unknown_preserved", file=sys.stderr)
            except (StorageError, OSError) as error:
                # If the status transition fails, the active record remains
                # protective; do not turn completed work into a false success.
                try:
                    print("managed_consumer_lease_warning=" + type(error).__name__, file=sys.stderr)
                except Exception:
                    pass
        receipt = manager.finish(exit_code=code, tree_gone=gone, supervisor_error=failure)
        print("managed_storage=" + json.dumps(receipt, sort_keys=True), file=sys.stderr)
        # The catalog total includes foreign reservations, so it cannot
        # establish actual disk pressure. Read a fresh advisory snapshot only
        # after the owner is released, and never let monitoring fail the run.
        try:
            from rovezero_usage_snapshot import claim_total_usage_maintenance
            maintenance_due = claim_total_usage_maintenance() if gone is True else False
        except Exception:
            maintenance_due = False
        if maintenance_due and gone is True:
            # Run only after the original owner finalized and released its
            # guard. A later owner, uncertain physical state, or inaccessible
            # candidate is preserved by the maintenance routine. Maintenance
            # is observationally separate from the completed command.
            try:
                from managed_output_maintenance import cleanup_completed_builds
                cleanup = cleanup_completed_builds(manager, seconds=5)
                try:
                    print("managed_storage_maintenance=" + json.dumps(cleanup, sort_keys=True), file=sys.stderr)
                except BaseException:
                    pass
            except Exception as error:
                try:
                    print("managed_storage_maintenance_warning=" + type(error).__name__, file=sys.stderr)
                except BaseException:
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
    finally:
        if consumer_lease is not None and not consumer_dispatched and not consumer_closed:
            try:
                consumer_lease.close(not_started=True)
            except (StorageError, OSError) as error:
                try:
                    print("managed_consumer_lease_warning=" + type(error).__name__, file=sys.stderr)
                except Exception:
                    pass


if __name__ == "__main__":
    sys.exit(main())
