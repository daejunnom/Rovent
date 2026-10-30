"""Utility-only finite windows and reproducible sampler metadata.

These standard-library validators neither load a dataset nor issue a learner
lease. The original utility parent must separately join the held publication,
actual action pairs, pinned runtime and current process before dispatch.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re

from .rovent_utility_training import MAX_CONTEXTS, validate_training_preparation
from .rovent_training_bootstrap import MAX_READ_BYTES


WINDOW_DOMAIN = "rz-rovent-utility-train-window/1"
WINDOW_V2_DOMAIN = "rz-rovent-utility-train-window/2"
SAMPLER_DOMAIN = "rz-rovent-utility-sampler/1"
MAX_UPDATES = 131072
MAX_WALL_NS = 24 * 60 * 60 * 10**9
MAX_CLEANUP_NS = 30 * 10**9
MAX_OUTPUT_BYTES = 18 * 1024**3
WINDOW_FIELDS = ("schema", "window_id", "run_id", "preparation_sha256", "dataset_sha256", "device",
                 "precision", "tf32", "deterministic", "cpu_threads", "interop_threads", "training_seed", "microbatch", "accumulation",
                 "max_optimizer_updates", "checkpoint_every_updates", "max_wall_time_ns",
                 "cleanup_time_ns", "max_read_bytes", "max_output_bytes", "max_groups", "max_contexts")
WINDOW_V2_FIELDS = (*WINDOW_FIELDS, "checkpoint_reserve_ns")
SAMPLER_FIELDS = ("schema", "run_id", "pool_sha256", "context_ids", "seed", "epoch", "permutation", "cursor")


def _fields(value, names, label):
    if type(value) is not dict or set(value) != set(names):
        raise ValueError(label + ": exact utility fields required")
    return value


def _uint(value, maximum=2**64 - 1, *, positive=False):
    if type(value) is not int or not int(positive) <= value <= maximum:
        raise ValueError("bounded exact utility unsigned integer required")
    return value


def _sha(value):
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("lowercase utility SHA-256 required")
    return value


def _identifier(value):
    if type(value) is not str or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", value) is None:
        raise ValueError("bounded utility execution identifier required")
    return value


def _json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")


def utility_window_digest(value):
    """Pin only the original V1 domain; never upgrade a new window."""
    return _window_digest(value, expected_domain=WINDOW_DOMAIN)


def utility_window_v2_digest(value):
    """Pin an explicit V2 reserve inside its distinct canonical domain."""
    return _window_digest(value, expected_domain=WINDOW_V2_DOMAIN)


def _window_digest(value, *, expected_domain):
    if type(value) is not dict or value.get("schema") != expected_domain:
        raise ValueError("explicit original utility window digest domain required")
    return hashlib.sha256(_json([expected_domain, value])).hexdigest()


def validate_window_spec(preparation, value):
    """Read only original V1 metadata; never accept or default a V2 reserve."""
    return _validate_window_spec(preparation, value, expected_domain=WINDOW_DOMAIN)


def validate_utility_window_v2(preparation, value):
    """Require an original bounded checkpoint reserve inside the hard window.

    This declaration neither extends execution/cleanup deadlines nor renews
    the original read budget. The live parent and trainer separately close
    forward/update admission before spending the reserved final-write time.
    """
    return _validate_window_spec(preparation, value, expected_domain=WINDOW_V2_DOMAIN)


def _validate_window_spec(preparation, value, *, expected_domain):
    if type(preparation) is not dict:
        raise ValueError("explicit utility preparation required")
    expected = validate_training_preparation(preparation)
    if _json(preparation) != _json(expected):
        raise ValueError("utility preparation changed from its exact declared codec")
    return _validate_window_fields(expected["preparation_sha256"], value, expected_domain=expected_domain,
        checkpoint_reserve=expected_domain == WINDOW_V2_DOMAIN)


def _validate_window_fields(preparation_sha256, value, *, expected_domain, checkpoint_reserve):
    """Shared bounded arithmetic only; callers validate their own preparation."""
    if type(checkpoint_reserve) is not bool:
        raise ValueError("explicit original checkpoint reserve policy required")
    _sha(preparation_sha256)
    _fields(value, WINDOW_V2_FIELDS if checkpoint_reserve else WINDOW_FIELDS, "utility window")
    if value["schema"] != expected_domain or value["preparation_sha256"] != preparation_sha256:
        raise ValueError("separate utility window/preparation seal required")
    _identifier(value["window_id"]); _identifier(value["run_id"])
    _sha(value["preparation_sha256"]); _sha(value["dataset_sha256"])
    if (type(value["device"]) is not str or not (value["device"] == "cpu"
            or re.fullmatch(r"cuda:(0|[1-9][0-9]?)", value["device"]) and int(value["device"][5:]) < 64)):
        raise ValueError("explicit CPU or canonical CUDA utility device required")
    _uint(value["cpu_threads"], 256, positive=True)
    _uint(value["interop_threads"], 256, positive=True)
    _uint(value["training_seed"], 2**32 - 1)
    if value["precision"] != "fp32" or value["tf32"] is not False or value["deterministic"] is not True:
        raise ValueError("deterministic FP32 utility execution with TF32 disabled required")
    if type(value["microbatch"]) is not int or value["microbatch"] != 1 or type(value["accumulation"]) is not int or value["accumulation"] != 8:
        raise ValueError("registered utility microbatch 1 and accumulation 8 required")
    _uint(value["max_optimizer_updates"], MAX_UPDATES)
    _uint(value["checkpoint_every_updates"], MAX_UPDATES, positive=True)
    _uint(value["max_wall_time_ns"], MAX_WALL_NS, positive=True)
    if checkpoint_reserve:
        _uint(value["checkpoint_reserve_ns"], value["max_wall_time_ns"] - 1, positive=True)
    _uint(value["cleanup_time_ns"], MAX_CLEANUP_NS, positive=True)
    _uint(value["max_read_bytes"], MAX_READ_BYTES, positive=True)
    _uint(value["max_output_bytes"], MAX_OUTPUT_BYTES, positive=True)
    _uint(value["max_groups"], 2**32 - 1, positive=True)
    _uint(value["max_contexts"], 2**32 - 1, positive=True)
    return copy.deepcopy(value)


def _contexts(values):
    if type(values) not in (list, tuple) or not 1 <= len(values) <= MAX_CONTEXTS:
        raise ValueError("bounded nonempty original utility context inventory required")
    checked = [_sha(value) for value in values]
    if len(set(checked)) != len(checked):
        raise ValueError("duplicate original utility context identity")
    return checked


def _permutation(context_ids, seed, epoch):
    # The exact domain and binary framing define the sampler, independently of
    # Python random's hidden state or a caller's global RNG. Context order is
    # preserved as the index namespace; the original inventory is not sorted.
    prefix = SAMPLER_DOMAIN.encode("ascii") + b"\0" + seed.to_bytes(8, "little") + epoch.to_bytes(8, "little")
    return sorted(range(len(context_ids)), key=lambda index: (
        hashlib.sha256(prefix + index.to_bytes(4, "little") + bytes.fromhex(context_ids[index])).digest(), index))


def make_utility_sampler(context_ids, pool_sha256, seed, *, run_id):
    contexts = _contexts(context_ids)
    _sha(pool_sha256); _uint(seed, 2**32 - 1); _identifier(run_id)
    return {"schema": SAMPLER_DOMAIN, "run_id": run_id, "pool_sha256": pool_sha256,
            "context_ids": contexts, "seed": seed, "epoch": 0,
            "permutation": _permutation(contexts, seed, 0), "cursor": 0}


def validate_utility_sampler(value, *, context_ids, pool_sha256, run_id, seed):
    _fields(value, SAMPLER_FIELDS, "utility sampler")
    contexts = _contexts(context_ids)
    _sha(pool_sha256); _identifier(run_id); _uint(seed, 2**32 - 1)
    if (value["schema"] != SAMPLER_DOMAIN or value["run_id"] != run_id
            or value["pool_sha256"] != pool_sha256 or type(value["context_ids"]) is not list
            or value["context_ids"] != contexts or type(value["seed"]) is not int or value["seed"] != seed):
        raise ValueError("utility sampler changed original run/pool/context order")
    epoch = _uint(value["epoch"])
    _uint(value["cursor"], len(contexts))
    permutation = value["permutation"]
    if (type(permutation) is not list or len(permutation) != len(contexts)
            or any(type(index) is not int for index in permutation)
            or permutation != _permutation(contexts, seed, epoch)):
        raise ValueError("utility sampler permutation differs from its exact seed/epoch framing")
    return copy.deepcopy(value)


def next_utility_sample(value, *, context_ids, pool_sha256, run_id, seed):
    """Return one original index and a new sampler, never mutate the caller."""
    checked = validate_utility_sampler(value, context_ids=context_ids, pool_sha256=pool_sha256, run_id=run_id, seed=seed)
    if checked["cursor"] == len(checked["context_ids"]):
        checked["epoch"] = _uint(checked["epoch"] + 1)
        checked["permutation"] = _permutation(checked["context_ids"], checked["seed"], checked["epoch"])
        checked["cursor"] = 0
    index = checked["permutation"][checked["cursor"]]
    checked["cursor"] += 1
    return index, checked
