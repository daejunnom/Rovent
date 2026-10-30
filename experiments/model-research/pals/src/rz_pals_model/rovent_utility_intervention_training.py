"""Explicit intervention learner entry over the unchanged finite math loop."""
from __future__ import annotations

import time

from .rovent_utility_training_family import INTERVENTION


def train_intervention_utility_window(owner, intake, spec, *, authority=None,
        checkpoint_directory=None, execute=False, clock=time.monotonic_ns):
    from .rovent_utility_training_run import _train_utility_window_core
    return _train_utility_window_core(owner, intake, spec, authority=authority,
        checkpoint_directory=checkpoint_directory, execute=execute, clock=clock, family=INTERVENTION)
