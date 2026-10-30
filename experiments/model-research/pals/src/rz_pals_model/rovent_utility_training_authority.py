"""Separate utility learner authority backed by its original live Rust parent.

Only Linux peer/process and bounded socket primitives are shared with the older
trainer. No P/C/V request, owner, role map or serialized permission is reused.
An acknowledgement is accepted only after its operation-specific meaning has
also been checked; any uncertainty quarantines the connection without replay.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from pathlib import PurePosixPath
import socket
import stat
import struct
import threading
import time

from .rovent_training_authority import _process_identity
from .rovent_training_bootstrap import PinnedReadBudget, _strict_json
from .generation_export import COST_FIELDS
from .rovent_utility_training_window import (_fields, _identifier, _sha, _uint, utility_window_v2_digest,
    validate_utility_window_v2, validate_utility_sampler)


STARTUP_SCHEMA = "rz-rovent-utility-trainer-startup/1"
RPC_SCHEMA = "rz-rovent-utility-trainer-rpc/1"
ACK_SCHEMA = "rz-rovent-utility-trainer-ack/1"
FRAME_BYTES = 65536
MAX_COMPLETION_RECEIPT_BYTES = 16 * 1024**2
_CONNECTED = object()
_STATE_FIELDS = ("model_state_sha256", "optimizer_state_sha256", "scheduler_state_sha256",
                 "rng_state_sha256", "sampler_state_sha256")
COST_DOMAIN = "rz-rovent-utility-checkpoint-cost/1"
COST_SCOPE = "recorded_cost_only_not_old_process_closure_or_new_window_authority"


class OriginalUtilityAuthorityRequired(ValueError):
    """Historical evidence and preparation JSON cannot issue a live lease."""


class _UtilityLease:
    __slots__ = ("authority", "owner", "intake", "spec", "admission", "closed", "initial_dispatched", "initial_completed")

    def __init__(self, authority, owner, intake, spec, admission):
        self.authority, self.owner, self.intake = authority, owner, intake
        self.spec, self.admission, self.closed = copy.deepcopy(spec), copy.deepcopy(admission), False
        self.initial_dispatched = owner.dispatched_updates
        self.initial_completed = owner.completed_updates


def require_original_utility_authority(authority, *, lease=None, owner=None, allow_unleased=False, cleanup=False):
    """Require this concrete connected owner, optionally its same opaque lease."""
    from .rovent_utility_training_family import LEGACY
    return _require_utility_authority(authority, family=LEGACY, lease=lease, owner=owner,
        allow_unleased=allow_unleased, cleanup=cleanup)


def _authority_type(family):
    from .rovent_utility_training_family import require_family
    family = require_family(family)
    if family.intervention:
        from .rovent_utility_intervention_authority import ParentSocketInterventionUtilityTrainerAuthority
        return ParentSocketInterventionUtilityTrainerAuthority
    return ParentSocketUtilityTrainerAuthority


def _require_utility_authority(authority, *, family, lease=None, owner=None, allow_unleased=False, cleanup=False):
    if (type(authority) is not _authority_type(family) or getattr(authority, "_family", None) is not family
            or getattr(authority, "_connection", None) is not _CONNECTED
            or authority._failed or authority._closed or authority._socket.fileno() < 0):
        raise OriginalUtilityAuthorityRequired("the original connected utility parent is required")
    authority._remaining(cleanup)
    current = authority._lease
    if lease is not None and (type(lease) is not _UtilityLease or lease is not current):
        raise OriginalUtilityAuthorityRequired("same original utility window lease required")
    if current is None:
        if not allow_unleased or lease is not None or owner is not None:
            raise OriginalUtilityAuthorityRequired("original utility learner lease has not begun")
    elif (current.authority is not authority or current.closed
            or owner is not None and current.owner is not owner):
        raise OriginalUtilityAuthorityRequired("original utility lease lifetime/owner differs")
    return authority


def _empty(payload):
    _fields(payload, (), "utility empty acknowledgement")
    return payload


def _pin(payload):
    _fields(payload, ("sha256", "bytes"), "utility original completion pin")
    _sha(payload["sha256"]); _uint(payload["bytes"], MAX_COMPLETION_RECEIPT_BYTES, positive=True)
    return copy.deepcopy(payload)


def _cost(value):
    _fields(value, COST_FIELDS, "utility recorded expense")
    for number in value.values():
        _uint(number)
    return value


def _cost_evidence(value):
    _fields(value, ("path", "sha256", "bytes"), "utility cost evidence reference")
    path = value["path"]
    if (type(path) is not str or not path or len(path.encode("utf-8")) > 4096 or "\\" in path
            or PurePosixPath(path).is_absolute() or str(PurePosixPath(path)) != path
            or any(part in ("", ".", "..") for part in path.split("/"))):
        raise ValueError("normalized relative original cost evidence required")
    from .rovent_training_bootstrap import _absolute
    _absolute("/recorded-utility-cost/" + path)  # Reuse the no-secret path boundary without any I/O.
    _sha(value["sha256"]); _uint(value["bytes"], positive=True)
    return value


def _tail(value, *, only_unknown=False):
    if type(value) is not dict:
        raise ValueError("tagged original utility closure-tail observation required")
    if value.get("status") == "known" and not only_unknown:
        _fields(value, ("status", "delta", "evidence"), "known utility closure tail")
        _cost(value["delta"]); _cost_evidence(value["evidence"])
    elif value.get("status") == "unknown":
        _fields(value, ("status", "reason", "evidence"), "unknown utility closure tail")
        _identifier(value["reason"])
        if value["evidence"] is not None:
            _cost_evidence(value["evidence"])
    else:
        raise ValueError("utility closure tail cannot be silently converted to known zero")
    return value


def validate_utility_checkpoint_cost(value):
    """Check recorded arithmetic, never old process closure or a new lease."""
    _fields(value, ("schema", "ancestor_cost", "snapshot_cost", "post_snapshot_delta", "closure_tail",
        "inherited_unknown_tails", "known_lower_bound", "total_status", "scope", "execution_authorized"), "utility checkpoint cost")
    if value["schema"] != COST_DOMAIN or value["scope"] != COST_SCOPE or value["execution_authorized"] is not False:
        raise ValueError("separate utility historical cost observation required")
    for name in ("ancestor_cost", "snapshot_cost", "post_snapshot_delta", "known_lower_bound"):
        _cost(value[name])
    _tail(value["closure_tail"])
    inherited = value["inherited_unknown_tails"]
    if type(inherited) is not list or len(inherited) > 262144:
        raise ValueError("bounded original inherited unknown utility tails required")
    seen, publications, windows, previous = set(), set(), set(), None
    for item in inherited:
        _fields(item, ("checkpoint_publication_sha256", "execution_id", "window_id", "tail"), "inherited unknown utility tail")
        _sha(item["checkpoint_publication_sha256"]); _identifier(item["execution_id"]); _identifier(item["window_id"])
        _tail(item["tail"], only_unknown=True)
        key = (item["checkpoint_publication_sha256"], item["execution_id"], item["window_id"])
        if (key in seen or key[0] in publications or key[1:] in windows
                or previous is not None and key <= previous):
            raise ValueError("inherited utility tails must be sorted, unique and cannot relabel a source/window")
        seen.add(key)
        publications.add(key[0]); windows.add(key[1:]); previous = key
    delta = value["closure_tail"]["delta"] if value["closure_tail"]["status"] == "known" else dict.fromkeys(COST_FIELDS, 0)
    expected = {name: _uint(value["ancestor_cost"][name] + value["snapshot_cost"][name]
        + value["post_snapshot_delta"][name] + delta[name]) for name in COST_FIELDS}
    status = "lower_bound_only" if inherited or value["closure_tail"]["status"] == "unknown" else "known"
    if value["known_lower_bound"] != expected or value["total_status"] != status:
        raise ValueError("utility recorded lower bound/status differs; unknown cost is not zero")
    return copy.deepcopy(value)


class ParentSocketUtilityTrainerAuthority:
    def __init__(self, token, stream, startup, startup_sha256, read_accounting, clock, *, family=None):
        if token is not _CONNECTED:
            raise OriginalUtilityAuthorityRequired("live utility parent peer handshake required")
        from .rovent_utility_training_family import LEGACY, require_family
        self._family = require_family(LEGACY if family is None else family)
        if type(self) is not _authority_type(self._family):
            raise OriginalUtilityAuthorityRequired("exact original utility authority source family required")
        self._connection, self._socket, self._startup = token, stream, copy.deepcopy(startup)
        self._startup_sha256, self._read_budget, self._clock = startup_sha256, read_accounting, clock
        self._sequence, self._failed, self._closed, self._lease = 0, False, False, None
        self._cancelled = False
        self._lock = threading.RLock()

    @classmethod
    def connect_registered(cls, startup, *, raw_startup, expected_sha256, read_accounting,
                           execute=False, clock=time.monotonic_ns):
        if cls is not ParentSocketUtilityTrainerAuthority or execute is not True:
            raise OriginalUtilityAuthorityRequired("explicit original utility socket connection required")
        from .rovent_utility_training_family import LEGACY
        return cls._connect_registered_core(startup, raw_startup=raw_startup, expected_sha256=expected_sha256,
            read_accounting=read_accounting, execute=execute, clock=clock, family=LEGACY)

    @classmethod
    def _connect_registered_core(cls, startup, *, raw_startup, expected_sha256, read_accounting, execute, clock, family):
        if cls is not _authority_type(family) or execute is not True:
            raise OriginalUtilityAuthorityRequired("explicit same-family original utility socket connection required")
        if (os.name != "posix" or not hasattr(socket, "SO_PEERCRED") or not hasattr(os, "getuid")
                or type(read_accounting) is not PinnedReadBudget or not read_accounting.strict_growth_credit):
            raise OriginalUtilityAuthorityRequired("Linux peer credentials and one original strict read budget required")
        if family.intervention:
            from .rovent_utility_intervention_execution import validate_intervention_startup as validate_startup
        else:
            from .rovent_utility_training_bootstrap import validate_startup
        checked = validate_startup(raw_startup, expected_sha256=expected_sha256, expected_bytes=len(raw_startup),
            execution_until_ns=startup["execution_until_ns"], whole_until_ns=startup["whole_until_ns"], now_ns=clock())
        if checked != startup or read_accounting.until_ns != startup["execution_until_ns"] or read_accounting.failed:
            raise ValueError("utility startup or original read account differs")
        control, path = startup["control"], Path(startup["control"]["socket_path"])
        if (not path.is_absolute() or len(os.fsencode(path)) >= 104 or path.parent.resolve(strict=True) != path.parent
                or path.is_symlink() or path.parent.is_symlink()):
            raise ValueError("plain original utility Unix endpoint required")
        if control["parent_pid"] != os.getppid() or _process_identity(control["parent_pid"])["start_ticks"] != control["parent_start_ticks"]:
            raise OriginalUtilityAuthorityRequired("live original utility parent PID/start differs")
        directory_fd, stream = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW), None
        try:
            directory, endpoint = os.fstat(directory_fd), path.lstat()
            if (not stat.S_ISDIR(directory.st_mode) or stat.S_IMODE(directory.st_mode) != 0o700
                    or directory.st_uid != os.getuid() or (directory.st_dev, directory.st_ino) != (control["control_device"], control["control_inode"])
                    or not stat.S_ISSOCK(endpoint.st_mode) or endpoint.st_uid != os.getuid()):
                raise OriginalUtilityAuthorityRequired("utility original private directory/socket owner differs")
            stream = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            stream.settimeout(min(30.0, max(0.001, (startup["execution_until_ns"] - clock()) / 10**9)))
            stream.connect(str(path))
            peer_pid, peer_uid, _ = struct.unpack("3i", stream.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i")))
            current = path.parent.stat(follow_symlinks=False)
            if (peer_pid != control["parent_pid"] or peer_uid != os.getuid()
                    or _process_identity(peer_pid)["start_ticks"] != control["parent_start_ticks"]
                    or (current.st_dev, current.st_ino) != (directory.st_dev, directory.st_ino)
                    or clock() >= startup["execution_until_ns"]):
                raise OriginalUtilityAuthorityRequired("connected peer is not the original utility parent")
            result = cls(_CONNECTED, stream, startup, expected_sha256, read_accounting, clock, family=family)
            stream = None
            return result
        finally:
            os.close(directory_fd)
            if stream is not None:
                stream.close()

    def _remaining(self, cleanup=False):
        delta = self._startup["whole_until_ns" if cleanup else "execution_until_ns"] - self._clock()
        if delta <= 0:
            raise TimeoutError("utility original deadline expired; renewal/reconnect forbidden")
        return delta / 10**9

    def _require(self, **kwargs):
        return _require_utility_authority(self, family=self._family, **kwargs)

    def _rpc(self, operation, payload, *, validate=_empty, cleanup=False):
        with self._lock:
            try:
                self._require(allow_unleased=True, cleanup=cleanup)
                sequence = self._sequence + 1
                request = {"schema": self._family.domain(RPC_SCHEMA), "window_id": self._startup["window_id"],
                    "startup_sha256": self._startup_sha256, "sequence": sequence, "operation": operation, "payload": payload}
                raw = (json.dumps(request, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
                if len(raw) > FRAME_BYTES:
                    raise ValueError("utility request exceeds its finite socket frame")
                self._socket.settimeout(min(30.0, self._remaining(cleanup)))
                self._socket.sendall(raw)
                received = bytearray()
                while not received.endswith(b"\n"):
                    self._socket.settimeout(min(30.0, self._remaining(cleanup)))
                    try:
                        block = self._socket.recv(min(4096, FRAME_BYTES + 1 - len(received)))
                    except socket.timeout:
                        continue  # Observe the same request against its original absolute deadline.
                    if not block:
                        raise EOFError("utility parent closed before its original acknowledgement")
                    received.extend(block)
                    if len(received) > FRAME_BYTES or b"\n" in received[:-1]:
                        raise ValueError("utility acknowledgement exceeded one bounded frame")
                ack = _strict_json(bytes(received), maximum=FRAME_BYTES)
                _fields(ack, ("schema", "window_id", "startup_sha256", "sequence", "operation", "status", "payload"), "utility acknowledgement")
                if (ack["schema"] != self._family.domain(ACK_SCHEMA) or ack["window_id"] != self._startup["window_id"]
                        or ack["startup_sha256"] != self._startup_sha256 or type(ack["sequence"]) is not int
                        or ack["sequence"] != sequence or ack["operation"] != operation or ack["status"] != "ok"):
                    raise OriginalUtilityAuthorityRequired("utility parent rejected or changed the original request binding")
                result = validate(ack["payload"])
                self._sequence = sequence
                return result
            except BaseException:
                self._failed = True
                raise

    @staticmethod
    def _state(owner):
        from .rovent_utility_tensor_training import _runtime
        runtime = _runtime(True)
        clone, seal = runtime[1], runtime[5]
        return {"model_state_sha256": seal(clone(owner.model.state_dict())),
                "optimizer_state_sha256": seal(clone(owner.optimizer.state_dict())),
                "scheduler_state_sha256": seal(clone(owner.scheduler.state_dict())),
                "rng_state_sha256": seal(clone(owner.rng)), "sampler_state_sha256": seal(owner.sampler)}

    def _derived(self, owner):
        from .generation_data import digest
        from .rovent_utility_tensor_training import STATE_DOMAIN
        return digest(self._family.domain(STATE_DOMAIN), self._family.derived(owner))

    @property
    def read_accounting(self):
        return self._read_budget

    def begin_window(self, owner, intake, spec):
        self._require(allow_unleased=True)
        if self._lease is not None:
            raise OriginalUtilityAuthorityRequired("one original utility process cannot issue a second lease")
        admission = self._family.require_dataset(intake)
        checked = self._family.validate_window(owner.preparation, spec)
        if admission["dataset_sha256"] != checked["dataset_sha256"]:
            raise ValueError("utility window differs from its original dataset observation")
        validate_utility_sampler(owner.sampler, context_ids=intake.context_ids, pool_sha256=intake.pool_sha256,
            run_id=checked["run_id"], seed=checked["training_seed"])
        payload = {"schema": self._family.domain("rz-rovent-utility-trainer-begin/1"), "preparation_sha256": owner.preparation["preparation_sha256"],
            "window_spec_sha256": self._family.window_digest(checked), "dataset_sha256": checked["dataset_sha256"],
            "completed_updates": owner.completed_updates, "ancestor_updates": owner.ancestor_updates,
            # These are the same actual owner's numerical metadata observations.
            # The parent checks raw/semantic assets separately; its serde float
            # spelling must never be substituted for the Python mapping seal.
            "initial_source_declaration": copy.deepcopy(owner.metadata["initial_training"]),
            "frozen_base_mapping_sha256": owner.metadata["frozen_base_mapping_sha256"],
            "frozen_base_weights_sha256": owner.metadata["base_weights_sha256"],
            "derived_training_state_sha256": self._derived(owner), **self._state(owner)}
        self._rpc("begin_window", payload)
        self._lease = _UtilityLease(self, owner, intake, checked, admission)
        return self._lease

    def consume_pair_context(self, lease, source_projection):
        self._require(lease=lease)
        # This is an operand, not a local authority factory. Its exact original
        # context/action/pair roster is owned and checked by the Rust parent
        # against the held verified dataset before it can return an empty ACK.
        if type(source_projection) is not dict:
            raise ValueError("explicit original utility source projection required")
        self._rpc("consume_pair_context", {"schema": self._family.domain("rz-rovent-utility-trainer-consume/1"),
                  "source_projection": copy.deepcopy(source_projection)})

    def completed_update(self, lease, event, owner):
        self._require(lease=lease, owner=owner)
        from .generation_data import digest
        from .rovent_utility_tensor_training import STATE_DOMAIN
        _fields(event, ("schema", "completed_updates", "dispatched_updates", "unique_known_pair_denominator",
                        "consumed_context_sha256", "groups", "contexts"), "utility completed update event")
        if (event["schema"] != self._family.domain("rz-rovent-utility-learner-update/1")
                or event["completed_updates"] != owner.completed_updates or event["dispatched_updates"] != owner.dispatched_updates
                or len(owner.completion_receipts) != owner.completed_updates - 1):
            raise ValueError("utility completion event differs from one unacknowledged actual step")
        for name in ("completed_updates", "dispatched_updates", "unique_known_pair_denominator", "groups", "contexts"):
            _uint(event[name], positive=True)
        if type(event["consumed_context_sha256"]) is not list or len(event["consumed_context_sha256"]) != event["contexts"]:
            raise ValueError("utility completion must retain every original consumed context in order")
        for value in event["consumed_context_sha256"]:
            _sha(value)
        return self._rpc("completed_update", {"schema": self._family.domain("rz-rovent-utility-trainer-update/1"), "event": copy.deepcopy(event),
            **self._state(owner), "cost": dict(owner.cost), "pending_microbatches": owner.pending_microbatches,
            "frozen_base_weights_sha256": owner.metadata["base_weights_sha256"],
            "frozen_base_mapping_sha256": owner.metadata["frozen_base_mapping_sha256"],
            "prior_completion_chain_sha256": digest(self._family.domain(STATE_DOMAIN), [*owner.ancestor_completion_sha256,
                *(item["sha256"] for item in owner.completion_receipts)])}, validate=_pin)

    def checkpoint_published(self, lease, checkpoint, owner):
        self._require(lease=lease, owner=owner)
        from .rovent_training_bootstrap import _pin as file_pin
        projection = file_pin(checkpoint["tensor_projection"], 16 * 1024**2)
        self._rpc("checkpoint_published", {"schema": self._family.domain("rz-rovent-utility-trainer-checkpoint/1"),
            "checkpoint": {name: checkpoint[name] for name in ("path", "sha256", "bytes")},
            "tensor_projection": copy.deepcopy(projection),
            "completed_updates": owner.completed_updates, "ancestor_updates": owner.ancestor_updates,
            **self._state(owner), "snapshot_cost": copy.deepcopy(checkpoint["snapshot_cost"]),
            "post_snapshot_cost": copy.deepcopy(checkpoint["post_snapshot_cost"]), "pending_microbatches": owner.pending_microbatches,
            "derived_training_state_sha256": self._derived(owner)})

    def window_deadline_ns(self, lease):
        self._require(lease=lease, cleanup=True)
        return self._startup["execution_until_ns"]

    def observe_cancel(self):
        """The child signal handler may revoke admission, never extend it."""
        self._cancelled = True

    def should_stop(self, lease):
        self._require(lease=lease, cleanup=True)
        return self._cancelled or self._clock() >= self._startup["execution_until_ns"]

    def end_window(self, lease, owner, status, *, physical_completion, dispatched_updates, checkpoint=None, result=None):
        self._require(lease=lease, owner=owner, cleanup=True)
        if (physical_completion != owner.physical_completion or type(dispatched_updates) is not int
                or dispatched_updates != owner.dispatched_updates - lease.initial_dispatched):
            raise ValueError("utility end must preserve actual physical/dispatched observations")
        try:
            self._rpc("end_window", {"schema": self._family.domain("rz-rovent-utility-trainer-end/1"), "status": status,
                "completed_updates": owner.completed_updates, "dispatched_updates": owner.dispatched_updates,
                "window_dispatched_updates": dispatched_updates, "ancestor_updates": owner.ancestor_updates, "cost": dict(owner.cost),
                "pending_microbatches": owner.pending_microbatches, "quarantined": owner.quarantined,
                "physical_completion": owner.physical_completion, "checkpoint": copy.deepcopy(checkpoint),
                "result": copy.deepcopy(result)}, cleanup=True)
        finally:
            lease.closed = True

    def validate_checkpoint_state(self, owner, checkpoint):
        self._require(allow_unleased=True)
        if self._lease is not None:
            raise OriginalUtilityAuthorityRequired("checkpoint state admission must precede the new utility window")
        def cost(payload):
            checked = validate_utility_checkpoint_cost(payload)
            if checked["snapshot_cost"] != owner.cost or checked["ancestor_cost"] != owner.ancestor_cost:
                raise ValueError("utility original cost sidecar differs from the exact loaded snapshot")
            return checked
        return self._rpc("validate_checkpoint_state", {"schema": self._family.domain("rz-rovent-utility-trainer-resume/1"),
            "checkpoint": {name: checkpoint[name] for name in ("path", "sha256", "bytes")},
            "preparation_sha256": owner.preparation["preparation_sha256"],
            "completed_updates": owner.completed_updates, "ancestor_updates": owner.ancestor_updates,
            "snapshot_cost": dict(owner.cost), "ancestor_cost": dict(owner.ancestor_cost),
            "source_payload_state_sha256": owner.source_payload_state_sha256,
            "source_snapshot_cost_observation": copy.deepcopy(owner.source_snapshot_cost_observation),
            **self._state(owner), "derived_training_state_sha256": self._derived(owner)}, validate=cost)

    def validate_learned_export(self, owner, kind):
        self._require(owner=owner)
        if kind not in ("collection", "deployment"):
            raise ValueError("explicit utility export kind required")
        lease = self._lease
        def learned(payload):
            _fields(payload, ("kind", "steps", "run_id", "dataset_sha256"), "utility learned export admission")
            if (payload["kind"] != "learned" or _uint(payload["steps"]) != owner.total_updates()
                    or not owner.total_updates() or payload["run_id"] != lease.spec["run_id"]
                    or payload["dataset_sha256"] != lease.spec["dataset_sha256"]):
                raise ValueError("utility learned export differs from actual current/ancestor completions")
            _identifier(payload["run_id"]); _sha(payload["dataset_sha256"])
            return copy.deepcopy(payload)
        return self._rpc("validate_learned_export", {"schema": self._family.domain("rz-rovent-utility-trainer-export/1"), "kind": kind,
            "preparation_sha256": owner.preparation["preparation_sha256"], "completed_updates": owner.completed_updates,
            "ancestor_updates": owner.ancestor_updates, **self._state(owner),
            "derived_training_state_sha256": self._derived(owner)}, validate=learned)

    def close(self):
        self._failed, self._closed = True, True
        if self._lease is not None:
            self._lease.closed = True
        self._socket.close()
