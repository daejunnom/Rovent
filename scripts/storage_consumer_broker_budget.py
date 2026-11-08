"""Keep a bounded broker transaction from recursively taking budget locks.

This gate is inert until an original supervisor wires it to the *same* budget
sampler and drives ``after_tick`` and ``finish``. It does not relax the budget
or itself prove child, catalog, physical completion, or timing observations.
"""
from __future__ import annotations

import os
import threading


class BrokerBudgetGate:
    def __init__(self, sampler, transaction_active):
        if not callable(sampler) or not callable(transaction_active):
            raise ValueError("original sampler and transaction observation required")
        self._sampler = sampler
        self._active = transaction_active
        self._deferred = False
        self.deferred_count = 0
        self._owner_pid = os.getpid()
        self._owner_thread = threading.get_ident()

    def _check_owner(self):
        if os.getpid() != self._owner_pid or threading.get_ident() != self._owner_thread:
            raise RuntimeError("original budget event-loop owner required")

    @property
    def deferred(self):
        return self._deferred

    def sample(self):
        self._check_owner()
        # An unavailable active observation cannot establish that sampling
        # happened; retain the obligation before asking for that observation.
        self._deferred = True
        if self._active():
            self.deferred_count += 1
            return False
        # Clear only after the original sampler actually succeeds. Any error
        # remains visible to the original supervisor and preserves dirty state.
        self._sampler()
        self._deferred = False
        return True

    def after_tick(self):
        self._check_owner()
        if self._deferred and not self._active():
            return self.sample()
        return False

    def finish(self):
        self._check_owner()
        try:
            active = self._active()
        except BaseException:
            self._deferred = True
            raise
        if active:
            raise RuntimeError("budget resample requires original transaction drain")
        if self._deferred:
            return self.sample()
        return False
