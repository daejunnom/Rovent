"""R auxiliary input/origin ancestry across enabled and disabled stages.

Only original cumulative seal sets are projected here. NNUEC update counts
belong to the separately validated tensor-derived state and are never read,
added or inferred from these sets. This helper does not validate input/origin
pairs, Rules, a dataset, targets or past execution authority; the original
joint owner and parent checkpoint chain retain those responsibilities.
"""
from __future__ import annotations

ANCESTRY_DOMAIN = "rz-rovent-r-auxiliary-ancestry/1"
_AUXILIARY_JOURNAL_DOMAIN = "rz-rovent-r-auxiliary-journal/1"
_MAX_SEALS = 262144
_MAX_CHAIN = 256


def _seals(value):
    if type(value) is not list or len(value) > _MAX_SEALS:
        raise ValueError("bounded R ancestry seal list required")
    for item in value:
        if type(item) is not str or len(item) != 64 or any(char not in "0123456789abcdef" for char in item):
            raise ValueError("R ancestry requires original lowercase SHA-256 seals")
    if value != sorted(set(value)):
        raise ValueError("R ancestry seal sets must be sorted and unique")
    return value


def validate_ancestry(value):
    """Validate the exact passive wire object, without modifying it."""
    if (type(value) is not dict
            or set(value) != {"schema", "input_sha256", "origin_sha256", "execution_authorized"}):
        raise ValueError("R auxiliary ancestry fields mismatch")
    if value["schema"] != ANCESTRY_DOMAIN or value["execution_authorized"] is not False:
        raise ValueError("R auxiliary ancestry cannot issue execution authority")
    _seals(value["input_sha256"])
    _seals(value["origin_sha256"])
    return value


def _projection(inputs, origins):
    if len(inputs) > _MAX_SEALS or len(origins) > _MAX_SEALS:
        raise ValueError("bounded R ancestry seal list required")
    return validate_ancestry({"schema": ANCESTRY_DOMAIN, "input_sha256": sorted(inputs),
        "origin_sha256": sorted(origins), "execution_authorized": False})


def merge_ancestry(values):
    """Copy the dominant original cumulative set, never invent a fork union.

    All inputs must form a nested chain in both seal dimensions regardless of
    argument order. Equal/subset ancestors are already represented once. An
    incomparable or cross-direction pair needs the original owner's chain
    evidence and cannot be reconciled by this metadata helper.
    """
    if type(values) not in (list, tuple) or len(values) > _MAX_CHAIN:
        raise ValueError("bounded explicit R ancestry chain required")
    checked = []
    for value in values:
        checked.append(validate_ancestry(value))
    checked.sort(key=lambda value: (len(value["input_sha256"]) + len(value["origin_sha256"]),
        len(value["input_sha256"]), len(value["origin_sha256"])))
    inputs, origins = set(), set()
    for value in checked:
        next_inputs, next_origins = set(value["input_sha256"]), set(value["origin_sha256"])
        if not (inputs <= next_inputs and origins <= next_origins):
            raise ValueError("R ancestry is not a nested cumulative input/origin chain")
        inputs, origins = next_inputs, next_origins
    return _projection(inputs, origins)


def ancestry_from_journal(journal):
    """Project source-owned R sets; do not validate or read optimizer counts.

    The caller validates its full original journal before this projection.
    Here only the existing R journal domain and its four relevant seal sets
    are consumed. Other sampler/recipe/count fields remain with their owner.
    Missing optional fields mean no recorded R ancestry; explicit null is not
    a valid journal/ancestry object and is never used to erase old lineage.
    """
    if type(journal) is not dict:
        raise ValueError("original checkpoint journal mapping required for R ancestry")
    values = []
    if "r_auxiliary" in journal:
        sidecar = journal["r_auxiliary"]
        names = ("retained_input_sha256", "retained_origin_sha256", "ancestor_input_sha256", "ancestor_origin_sha256")
        if (type(sidecar) is not dict or sidecar.get("schema") != _AUXILIARY_JOURNAL_DOMAIN
                or not set(names) <= set(sidecar)):
            raise ValueError("original R auxiliary journal seal projection required")
        for name in names:
            _seals(sidecar[name])
        values.append(_projection(set(sidecar["retained_input_sha256"]) | set(sidecar["ancestor_input_sha256"]),
            set(sidecar["retained_origin_sha256"]) | set(sidecar["ancestor_origin_sha256"])))
    if "r_auxiliary_ancestry" in journal:
        values.append(validate_ancestry(journal["r_auxiliary_ancestry"]))
    return merge_ancestry(values)
