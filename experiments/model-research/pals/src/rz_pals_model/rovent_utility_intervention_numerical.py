"""Sealed numerical access to an original exact-nine-field preparation.

This object validates metadata and exposes unchanged numerical declarations.
It creates no tensor, optimizer, historical dataset or current learner lease.
"""
from __future__ import annotations

import copy

from .generation_data import digest
from . import rovent_utility_training as mathematics
from .rovent_utility_intervention_preparation import PREPARATION_DOMAIN, validate_intervention_preparation


DATA_FAMILY = "intervention_v2"
_VALIDATED = object()


class ValidatedUtilityNumericalParts:
    __slots__ = ("_original", "_seal")

    def __init__(self, token, original):
        if token is not _VALIDATED:
            raise TypeError("use the strict intervention numerical-parts validator")
        self._original = copy.deepcopy(original)
        self._seal = digest(PREPARATION_DOMAIN, {"kind": "numerical_parts", "value": self._original})

    def _check(self):
        checked = validate_intervention_preparation(self._original)
        if digest(PREPARATION_DOMAIN, {"kind": "numerical_parts", "value": checked}) != self._seal:
            raise ValueError("validated original intervention numerical metadata changed")
        return checked

    @property
    def data_family(self):
        self._check()
        return DATA_FAMILY

    @property
    def preparation(self):
        return copy.deepcopy(self._check())

    @property
    def recipe(self):
        return copy.deepcopy(self._check()["recipe"])

    @property
    def registration(self):
        return copy.deepcopy(self._check()["registration"])

    @property
    def parameter_spec(self):
        return copy.deepcopy(self._check()["parameter_spec"])

    @property
    def preparation_sha256(self):
        return self._check()["preparation_sha256"]

    @property
    def recipe_sha256(self):
        return mathematics.recipe_sha256(self._check()["recipe"])

    @property
    def parameter_spec_sha256(self):
        checked = self._check()
        return mathematics.parameter_spec_sha256(checked["parameter_spec"],
            checked["registration"]["frozen_base"]["config"], checked["recipe"]["optimizer_owner"])

    @property
    def registration_sha256(self):
        # Separate source-family seal, not a PREP2 registration hash. The
        # unchanged exact9 preparation seal additionally binds source13.
        return digest(PREPARATION_DOMAIN, {"kind": "source_registration", "value": self._check()["registration"]})

    @property
    def source_registration_sha256(self):
        return self.registration_sha256

    @property
    def optimizer_owner(self):
        return self._check()["recipe"]["optimizer_owner"]


def validate_intervention_numerical_parts(preparation):
    return ValidatedUtilityNumericalParts(_VALIDATED, validate_intervention_preparation(preparation))
