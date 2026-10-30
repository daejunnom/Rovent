"""Original Rust parent socket adapter; importing this module is stdlib-only.

A pinned startup JSON is insufficient. Construction also checks the live
parent/child kernel identities, private control directory and connected Unix
peer. Each request is bound to that single window, sequence and deadline; a
broken connection is never reconnected or converted to metadata permission.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import socket
import stat
import struct
import threading
import time

from .generation_data import fields, sha, strict_json, uint
from .rovent_training import membership_digest
from .rovent_training_window import OriginalTrainerAuthority, OriginalTrainingAuthorityRequired, retained_seal_set, training_profile_digest

RPC_SCHEMA = "rz-rovent-original-trainer-rpc/1"
ACK_SCHEMA = "rz-rovent-original-trainer-ack/1"
FRAME_BYTES = 65536
_CONNECTED = object()


def _process_identity(pid):
    uint(pid, 2**31 - 1)
    if not pid:
        raise ValueError("live positive original process id required")
    raw = Path("/proc") / str(pid) / "stat"
    with raw.open("rb") as stream:
        value = stream.read(8193)
    if len(value) > 8192 or b") " not in value:
        raise ValueError("bounded Linux process identity required")
    after = value.rsplit(b") ", 1)[1].split()
    if len(after) < 20:
        raise ValueError("incomplete original process identity")
    return {"parent_pid": int(after[1]), "start_ticks": int(after[19])}


class _WindowLease:
    __slots__ = ("authority", "closed")
    def __init__(self, authority):
        self.authority, self.closed = authority, False


class ParentSocketTrainerAuthority(OriginalTrainerAuthority):
    STARTUP_SCHEMA = "rz-rovent-original-trainer-startup/1"
    RPC_SCHEMA = RPC_SCHEMA
    ACK_SCHEMA = ACK_SCHEMA
    def __init__(self, token, stream, startup, startup_sha256, clock):
        if token is not _CONNECTED:
            raise OriginalTrainingAuthorityRequired("live original parent socket handshake required")
        self._socket, self._startup = stream, copy.deepcopy(startup)
        self._startup_sha256, self._clock = startup_sha256, clock
        self._sequence, self._failed, self._lease = 0, False, None
        self._lock = threading.RLock()

    @classmethod
    def connect_registered(cls, startup, *, raw_startup, expected_sha256, execute=False, clock=time.monotonic_ns):
        if execute is not True:
            raise OriginalTrainingAuthorityRequired("explicit original socket connection required")
        if os.name != "posix" or not hasattr(socket, "SO_PEERCRED") or not hasattr(os, "getuid"):
            raise OriginalTrainingAuthorityRequired("Linux original process peer credentials required; no transport fallback")
        sha(expected_sha256)
        if (type(raw_startup) is not bytes or not 0 < len(raw_startup) <= 16 * 1024**2
                or hashlib.sha256(raw_startup).hexdigest() != expected_sha256
                or strict_json(raw_startup.decode("utf-8", "strict")) != startup):
            raise ValueError("original socket startup byte pin differs")
        if startup.get("schema") != cls.STARTUP_SCHEMA:
            raise ValueError("new original Rovent startup domain required")
        control = fields(startup["control"], ("socket_path", "parent_pid", "parent_start_ticks", "control_device", "control_inode"), "original private control")
        path = Path(control["socket_path"])
        if (not path.is_absolute() or len(os.fsencode(path)) >= 104 or path.parent.resolve(strict=True) != path.parent
                or path.is_symlink() or path.parent.is_symlink()):
            raise ValueError("plain original Unix control endpoint required")
        until = uint(startup["execution_until_ns"])
        whole = uint(startup["whole_until_ns"])
        if not uint(startup["original_started_ns"]) <= clock() < until <= whole:
            raise ValueError("original socket window expired; deadlines cannot be renewed")
        if control["parent_pid"] != os.getppid() or _process_identity(control["parent_pid"])["start_ticks"] != control["parent_start_ticks"]:
            raise OriginalTrainingAuthorityRequired("original live parent PID/start identity differs")
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        directory_fd = os.open(path.parent, flags)
        stream = None
        try:
            directory = os.fstat(directory_fd)
            endpoint = path.lstat()
            if (not stat.S_ISDIR(directory.st_mode) or stat.S_IMODE(directory.st_mode) != 0o700
                    or directory.st_uid != os.getuid() or (directory.st_dev, directory.st_ino) != (control["control_device"], control["control_inode"])
                    or not stat.S_ISSOCK(endpoint.st_mode) or endpoint.st_uid != os.getuid()):
                raise OriginalTrainingAuthorityRequired("original private control directory/socket owner differs")
            stream = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            stream.settimeout(min(30.0, max(0.001, (until - clock()) / 10**9)))
            stream.connect(str(path))
            peer_pid, peer_uid, _peer_gid = struct.unpack("3i", stream.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i")))
            now = path.parent.stat(follow_symlinks=False)
            if (peer_pid != control["parent_pid"] or peer_uid != os.getuid()
                    or _process_identity(peer_pid)["start_ticks"] != control["parent_start_ticks"]
                    or (now.st_dev, now.st_ino) != (directory.st_dev, directory.st_ino) or clock() >= until):
                raise OriginalTrainingAuthorityRequired("connected peer is not the original registered parent")
            result = cls(_CONNECTED, stream, startup, expected_sha256, clock)
            stream = None
            return result
        finally:
            os.close(directory_fd)
            if stream is not None:
                stream.close()

    def _remaining(self, cleanup):
        until = self._startup["whole_until_ns" if cleanup else "execution_until_ns"]
        delta = until - self._clock()
        if delta <= 0:
            raise TimeoutError("original trainer deadline expired; reconnect/renewal forbidden")
        return delta / 10**9

    def _rpc(self, operation, payload, *, cleanup=False):
        with self._lock:
            if self._failed:
                raise OriginalTrainingAuthorityRequired("failed original trainer connection is quarantined")
            try:
                sequence = self._sequence + 1
                request = {"schema": self.RPC_SCHEMA, "window_id": self._startup["window_id"],
                    "startup_sha256": self._startup_sha256, "sequence": sequence, "operation": operation, "payload": payload}
                raw = (json.dumps(request, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
                if len(raw) > FRAME_BYTES:
                    raise ValueError("original trainer request exceeds bounded frame")
                self._socket.settimeout(min(30.0, self._remaining(cleanup)))
                self._socket.sendall(raw)
                received = bytearray()
                while not received.endswith(b"\n"):
                    self._socket.settimeout(min(30.0, self._remaining(cleanup)))
                    try:
                        block = self._socket.recv(min(4096, FRAME_BYTES + 1 - len(received)))
                    except socket.timeout:
                        # This is a finite observation poll, not a renewed
                        # lease or replayed request. The absolute original
                        # deadline is checked again before the next read.
                        continue
                    if not block:
                        raise EOFError("original trainer parent closed before terminal acknowledgement")
                    received.extend(block)
                    if len(received) > FRAME_BYTES or b"\n" in received[:-1]:
                        raise ValueError("original trainer ack exceeded one bounded frame")
                ack = strict_json(bytes(received).decode("utf-8", "strict"))
                fields(ack, ("schema", "window_id", "sequence", "operation", "accepted", "payload"), "original trainer acknowledgement")
                if (ack["schema"] != self.ACK_SCHEMA or ack["window_id"] != self._startup["window_id"]
                        or type(ack["sequence"]) is not int or ack["sequence"] != sequence or ack["operation"] != operation
                        or type(ack["accepted"]) is not bool or type(ack["payload"]) is not dict):
                    raise ValueError("original trainer ack changed window/order/operation")
                self._sequence = sequence
                if not ack["accepted"]:
                    raise OriginalTrainingAuthorityRequired("original parent rejected " + operation + ": " + str(ack["payload"].get("code", "rejected"))[:256])
                return ack["payload"]
            except BaseException:
                self._failed = True
                raise

    def _checked_lease(self, lease):
        if type(lease) is not _WindowLease or lease is not self._lease or lease.authority is not self or lease.closed:
            raise OriginalTrainingAuthorityRequired("same original live window lease required")

    @staticmethod
    def _state(owner):
        from .rovent_tensor_training import _runtime
        runtime = _runtime(True)
        return {"model_state_sha256": runtime[5](runtime[1](owner.model.state_dict())),
            "optimizer_state_sha256": runtime[5]({name: runtime[1](optimizer.state_dict()) for name, optimizer in owner.optimizers.items()}),
            "scheduler_state_sha256": runtime[5]({name: runtime[1](scheduler.state_dict()) for name, scheduler in owner.schedulers.items()}),
            "rng_state_sha256": runtime[5](owner.rng), "sampler_state_sha256": runtime[5](owner.samplers)}

    def begin_window(self, owner, intake, spec, admission):
        if self._lease is not None:
            raise OriginalTrainingAuthorityRequired("one original window cannot issue a second learner lease")
        payload = {"schema": "rz-rovent-trainer-begin/1", "recipe_sha256": owner.recipe.sha256, "stage_id": owner.stage_id,
            "training_profile_sha256": training_profile_digest(spec), "corpus_sha256": admission["corpus_sha256"],
            "source_sha256": spec["source_sha256"], "runtime_sha256": spec["runtime_sha256"],
            "completed_updates": dict(owner.completed), "ancestor_updates": dict(owner.ancestor_updates),
            "frozen_pc_sha256": spec["frozen_pc_sha256"]}
        if self._rpc("begin_window", payload):
            raise ValueError("original begin ack must not smuggle serialized capabilities")
        self._lease = _WindowLease(self)
        return self._lease

    def consume_sample(self, lease, sample):
        self._checked_lease(lease)
        payload = {"schema": "rz-rovent-trainer-consume/1", **{name: copy.deepcopy(sample[name]) for name in
            ("input_sha256", "origin_sha256", "record_sha256", "native_retirement_sha256", "head_raw_sha256", "role", "split")}}
        if self._rpc("consume_sample", payload):
            raise ValueError("original consume ack must be empty")

    def completed_update(self, lease, event, owner):
        self._checked_lease(lease)
        return self._rpc("completed_update", {"schema": "rz-rovent-trainer-update/1", "event": copy.deepcopy(event),
            **self._state(owner), "cost": dict(owner.cost), "pending_microbatches": owner.pending_microbatches})

    def checkpoint_published(self, lease, checkpoint, owner):
        self._checked_lease(lease)
        state = self._state(owner)
        payload = {"schema": "rz-rovent-trainer-checkpoint/1", "checkpoint": {name: checkpoint[name] for name in ("sha256", "bytes", "path")},
            "completed_updates": dict(owner.completed), "sampler_state_sha256": state["sampler_state_sha256"],
            "model_state_sha256": state["model_state_sha256"], "optimizer_state_sha256": state["optimizer_state_sha256"],
            "scheduler_state_sha256": state["scheduler_state_sha256"], "rng_state_sha256": state["rng_state_sha256"],
            "cost": dict(owner.cost), "pending_microbatches": owner.pending_microbatches}
        if self._rpc("checkpoint_published", payload):
            raise ValueError("original checkpoint ack must be empty")

    def end_window(self, lease, *, status, owner):
        self._checked_lease(lease)
        payload = {"schema": "rz-rovent-trainer-end/1", "status": status, "completed_updates": dict(owner.completed),
            "dispatched_updates": dict(owner.dispatched), "cost": dict(owner.cost),
            "pending_microbatches": owner.pending_microbatches, "quarantined": owner.quarantined,
            "physical_completion": owner.physical_completion}
        try:
            if self._rpc("end_window", payload, cleanup=True):
                raise ValueError("original end ack must be empty")
        finally:
            lease.closed = True

    def _checkpoint_request(self, owner, intake, checkpoint):
        inputs, origins = retained_seal_set(owner.inherited_input_sha256, "input"), retained_seal_set(owner.inherited_origin_sha256, "origin")
        return {"checkpoint": {name: checkpoint[name] for name in ("sha256", "bytes")},
            "recipe_sha256": owner.recipe.sha256, "stage_id": owner.stage_id,
            "membership_sha256": membership_digest(owner.membership), "completed_updates": dict(owner.completed),
            "ancestor_updates": dict(owner.ancestor_updates), "retained_input_set_sha256": inputs["sha256"],
            "retained_input_count": inputs["count"], "retained_origin_set_sha256": origins["sha256"],
            "retained_origin_count": origins["count"], **self._state(owner)}

    def validate_checkpoint_state(self, owner, intake, checkpoint):
        return self._rpc("validate_checkpoint_state", {"schema": "rz-rovent-trainer-resume/1", **self._checkpoint_request(owner, intake, checkpoint)})

    def validate_learned_export(self, owner, intake, checkpoint):
        return self._rpc("validate_learned_export", {"schema": "rz-rovent-trainer-export/1", **self._checkpoint_request(owner, intake, checkpoint)})

    def close(self):
        self._failed = True
        self._socket.close()
