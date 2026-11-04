"""Explicit current intervention_v2 child; importing performs no execution.

Only this named entry chooses the closed INTERVENTION family. Original /1
entry points keep LEGACY and cannot decode a /2 corpus by changing a flag or
rebuilding a PREP2 envelope. Parent peer/FD/resource/deadline and terminal
machinery are shared without restoring any historical execution authority.
"""
from __future__ import annotations

import time

from .rovent_utility_training_child import _utility_child_main
from .rovent_utility_training_family import INTERVENTION


def intervention_utility_child_main(argv=None, *, stdin=None, clock=time.monotonic_ns):
    return _utility_child_main(argv, stdin=stdin, clock=clock, family=INTERVENTION)


if __name__ == "__main__":
    raise SystemExit(intervention_utility_child_main())
