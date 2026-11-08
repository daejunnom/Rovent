"""Read-only binding of a broker reply to its live direct supervisor.

The peer check is not an adversarial same-UID security boundary and does not
prove an OS lock. The supervisor must independently hold and verify its
original catalog guard and registry binding while handling each request.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


class BrokerIdentityError(RuntimeError):
    pass


def _process_birth(pid):
    # Linux /proc stat's comm field may contain spaces and parentheses.
    raw = Path("/proc").joinpath(str(pid), "stat").read_text(encoding="ascii")
    tail = raw.rsplit(")", 1)[1].split()
    if len(tail) <= 19 or not tail[19].isdecimal():
        raise BrokerIdentityError("original supervisor birth unavailable")
    return tail[19]


class OriginalSupervisorIdentity:
    """Check direct parent, nonsecret managed owner record and registry ID."""

    def __init__(self, *, root, source, token, parent_pid, registry_root, registry_id):
        if sys.platform != "linux":
            raise BrokerIdentityError("Linux direct-parent identity required")
        if type(parent_pid) is not int or parent_pid <= 1 or parent_pid != os.getppid():
            raise BrokerIdentityError("live direct supervisor required")
        if not isinstance(token, str) or not token or len(token) > 256:
            raise BrokerIdentityError("managed owner token required")
        from managed_storage import checked
        from storage_consumer_registry import ConsumerRegistry
        from storage_guard import lifetime
        self.root = checked(Path(root))
        self.source = checked(Path(source))
        self.token = token
        self.parent_pid = parent_pid
        self.uid, self.gid = os.getuid(), os.getgid()
        self.registry = ConsumerRegistry(registry_root, registry_id=registry_id)
        current = lifetime()
        self.host, self.boot = current["host"], current["boot"]
        self.birth = _process_birth(parent_pid)
        self._validate()

    def _validate(self):
        from managed_storage import SCHEMA, read_record
        if os.getppid() != self.parent_pid:
            raise BrokerIdentityError("original direct supervisor changed")
        record = read_record(self.root / "owner.json")
        root = self.root.stat()
        life = record.get("lifetime") if isinstance(record, dict) else None
        if (not isinstance(record, dict) or record.get("schema") != SCHEMA
                or type(record.get("pid")) is not int or record.get("pid") != self.parent_pid
                or record.get("token") != self.token
                or record.get("source") != str(self.source)
                or record.get("root_identity") != [root.st_dev, root.st_ino]
                or not isinstance(life, dict) or life.get("platform") != "linux"
                or life.get("host") != self.host or life.get("boot") != self.boot
                or life.get("process_birth") != self.birth
                or _process_birth(self.parent_pid) != self.birth):
            raise BrokerIdentityError("original managed supervisor identity changed")
        self.registry._check_identity()

    def check_peer(self, pid, uid, gid):
        if ((pid, uid, gid) != (self.parent_pid, self.uid, self.gid)
                or os.getppid() != self.parent_pid):
            raise BrokerIdentityError("broker reply was not from the original direct supervisor")
        self._validate()
