"""Explicit intervention execution/window/startup codecs; decoding runs nothing."""
from __future__ import annotations

import time

from .generation_data import digest
from .rovent_utility_intervention_numerical import DATA_FAMILY, validate_intervention_numerical_parts
from .rovent_utility_training_family import INTERVENTION
from . import rovent_utility_training_bootstrap as common
from .rovent_utility_training_window import _validate_window_fields, _window_digest


REGISTRATION_DOMAIN = "rz-rovent-utility-training-registration/3"
WINDOW_DOMAIN = "rz-rovent-utility-train-window/3"
STARTUP_DOMAIN = "rz-rovent-utility-trainer-startup/2"
RESULT_DOMAIN = "rz-rovent-utility-trainer-result/2"
REGISTRATION_FIELDS = (*common.REGISTRATION_FIELDS, "data_family")
STARTUP_FIELDS = (*common.STARTUP_FIELDS, "data_family")


def validate_intervention_registration(value):
    return common._validate_registration_core(value, expected_domain=REGISTRATION_DOMAIN, intervention=True)


def validate_intervention_ancestor_registration(value):
    if type(value) is dict and value.get("kind") == "utility_intervention":
        common._fields(value, ("kind", "registration", "preparation"), "original intervention ancestor registration")
        common._dataset_reference(value["registration"])
        common._pin(value["preparation"], 65536)
        return value["registration"]["publication"]["sha256"]
    return common._ancestor_v2(value)


def validate_intervention_window(preparation, value):
    parts = validate_intervention_numerical_parts(preparation)
    return _validate_window_fields(parts.preparation_sha256, value, expected_domain=WINDOW_DOMAIN,
        checkpoint_reserve=True)


def intervention_window_digest(value):
    return _window_digest(value, expected_domain=WINDOW_DOMAIN)


def validate_intervention_startup(raw, *, expected_sha256, expected_bytes, execution_until_ns, whole_until_ns, now_ns):
    value = common._validate_startup_core(raw, expected_sha256=expected_sha256, expected_bytes=expected_bytes,
        execution_until_ns=execution_until_ns, whole_until_ns=whole_until_ns, now_ns=now_ns, family=INTERVENTION)
    if (value["data_family"] != DATA_FAMILY
            or value["registration_sha256"] != digest(REGISTRATION_DOMAIN, value["registration"])):
        raise ValueError("intervention startup changed its exact original registration/source family")
    return value


def bootstrap_intervention_child(startup, *, until_ns, clock=time.monotonic_ns):
    return common._bootstrap_child_material(startup, until_ns=until_ns, clock=clock,
        validator=validate_intervention_startup, entry_basename="rovent_utility_intervention_child.py",
        executable_maximum=common.MAX_RUNTIME_MODULE_BYTES,
        report_domain="rz-rovent-utility-trainer-bootstrap-readback/2", recovered=True)
