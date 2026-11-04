"""Current intervention socket/peer ownership, never reconstructed from JSON."""
from __future__ import annotations

import time

from .rovent_utility_training_family import INTERVENTION
from .rovent_utility_training_authority import (ParentSocketUtilityTrainerAuthority,
    OriginalUtilityAuthorityRequired, _require_utility_authority)


class ParentSocketInterventionUtilityTrainerAuthority(ParentSocketUtilityTrainerAuthority):
    @classmethod
    def connect_registered(cls, startup, *, raw_startup, expected_sha256, read_accounting,
                           execute=False, clock=time.monotonic_ns):
        if cls is not ParentSocketInterventionUtilityTrainerAuthority:
            raise OriginalUtilityAuthorityRequired("exact original intervention parent owner required")
        return cls._connect_registered_core(startup, raw_startup=raw_startup,
            expected_sha256=expected_sha256, read_accounting=read_accounting, execute=execute,
            clock=clock, family=INTERVENTION)


def require_intervention_utility_authority(authority, *, lease=None, owner=None, allow_unleased=False, cleanup=False):
    return _require_utility_authority(authority, family=INTERVENTION, lease=lease, owner=owner,
        allow_unleased=allow_unleased, cleanup=cleanup)
