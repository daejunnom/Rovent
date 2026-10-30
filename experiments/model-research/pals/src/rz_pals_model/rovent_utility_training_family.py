"""Closed source-family selection for shared numerical/OS machinery.

Public old readers remain old-only. Only explicit intervention entry points
pass INTERVENTION to private cores; no caller-supplied domain/callback table,
ambient context switch or data retagging selects another source family.
"""
from __future__ import annotations

import copy

from . import rovent_utility_training as legacy


_FAMILY = object()
_PARTS = object()


class _LegacyNumericalParts:
    __slots__ = ("_value",)

    def __init__(self, token, value):
        if token is not _PARTS:
            raise TypeError("original numerical parts validator required")
        self._value = legacy.validate_training_preparation(value)

    @property
    def preparation(self):
        return copy.deepcopy(legacy.validate_training_preparation(self._value))

    @property
    def recipe(self): return self.preparation["recipe"]
    @property
    def registration(self): return self.preparation["registration"]
    @property
    def parameter_spec(self): return self.preparation["parameter_spec"]
    @property
    def preparation_sha256(self): return self.preparation["preparation_sha256"]
    @property
    def recipe_sha256(self): return self.preparation["recipe_sha256"]
    @property
    def parameter_spec_sha256(self): return self.preparation["parameter_spec_sha256"]
    @property
    def source_registration_sha256(self): return self.preparation["registration_sha256"]
    @property
    def registration_sha256(self): return self.source_registration_sha256
    @property
    def optimizer_owner(self): return self.preparation["optimizer_owner"]


class _UtilityTrainingFamily:
    __slots__ = ("_intervention",)

    def __init__(self, token, intervention):
        if token is not _FAMILY or type(intervention) is not bool:
            raise TypeError("closed utility training family required")
        self._intervention = intervention

    def checked(self):
        if self not in (LEGACY, INTERVENTION):
            raise ValueError("unregistered utility source-family object")
        if self._intervention is not (self is INTERVENTION):
            raise ValueError("utility source-family discriminator changed")
        return self

    @property
    def intervention(self):
        self.checked()
        return self is INTERVENTION

    def domain(self, original):
        self.checked()
        if type(original) is not str or not original.startswith("rz-rovent-utility-") or not original.endswith("/1"):
            raise ValueError("declared original utility domain required")
        common = ("rz-rovent-utility-trainer-source-bundle/1", "rz-rovent-utility-sampler/1",
            "rz-rovent-utility-checkpoint-cost/1")
        return original[:-1] + "2" if self.intervention and original not in common else original

    def parts(self, preparation):
        self.checked()
        if self.intervention:
            from .rovent_utility_intervention_numerical import validate_intervention_numerical_parts
            return validate_intervention_numerical_parts(preparation)
        return _LegacyNumericalParts(_PARTS, preparation)

    def validate_window(self, preparation, value):
        self.checked()
        if self.intervention:
            from .rovent_utility_intervention_execution import validate_intervention_window
            return validate_intervention_window(preparation, value)
        from .rovent_utility_training_window import validate_utility_window_v2
        return validate_utility_window_v2(preparation, value)

    def window_digest(self, value):
        self.checked()
        if self.intervention:
            from .rovent_utility_intervention_execution import intervention_window_digest
            return intervention_window_digest(value)
        from .rovent_utility_training_window import utility_window_v2_digest
        return utility_window_v2_digest(value)

    def require_dataset(self, value):
        self.checked()
        if self.intervention:
            from .rovent_utility_intervention_dataset import HistoricalUtilityInterventionDataset
            expected = HistoricalUtilityInterventionDataset
        else:
            from .rovent_utility_route import HistoricalUtilityRouteDataset
            expected = HistoricalUtilityRouteDataset
        if type(value) is not expected:
            raise ValueError("exact original source-family dataset owner required")
        return value.observation()

    def require_authority(self, authority, **kwargs):
        self.checked()
        if self.intervention:
            from .rovent_utility_intervention_authority import require_intervention_utility_authority
            return require_intervention_utility_authority(authority, **kwargs)
        from .rovent_utility_training_authority import require_original_utility_authority
        return require_original_utility_authority(authority, **kwargs)

    def save_checkpoint(self, owner, path, *, authority, lease, **kwargs):
        self.checked()
        if self.intervention:
            from .rovent_utility_intervention_checkpoint import save_intervention_utility_checkpoint
            return save_intervention_utility_checkpoint(owner, path, authority=authority, lease=lease, **kwargs)
        from .rovent_utility_tensor_training import save_utility_checkpoint
        kwargs.pop("post_snapshot_observer", None)
        return save_utility_checkpoint(owner, path, **kwargs)

    def derived(self, owner):
        self.checked()
        if self.intervention:
            from .rovent_utility_intervention_checkpoint import derived_intervention_training_state
            return derived_intervention_training_state(owner)
        from .rovent_utility_tensor_training import derived_training_state
        return derived_training_state(owner)


LEGACY = _UtilityTrainingFamily(_FAMILY, False)
INTERVENTION = _UtilityTrainingFamily(_FAMILY, True)


def require_family(value):
    if type(value) is not _UtilityTrainingFamily:
        raise TypeError("closed source-family selection required")
    return value.checked()
