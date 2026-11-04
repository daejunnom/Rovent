"""Bounded in-process custody observations, never a GPU completion fence."""
from __future__ import annotations

import socket


class CustodyTransport:
    """Own a Linux seqpacket pair for one existing supervised child.

    Only the child retains GPU-related owners. This channel reports their
    process-local state; no packet or child exit establishes a device fence.
    """

    def __init__(self):
        self.parent, self.child = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        self.parent.setblocking(False)
        self.state = {"status": "unobserved", "events": [], "requests": 0,
                      "error": None, "scope": "same_child_process_only"}
        self._disabled = False
        self._unknown_seen = False

    def child_fd(self):
        return self.child.fileno()

    def child_started(self):
        # Popen has inherited its own duplicate. The supervisor must not retain
        # a peer endpoint and mistake an exited child for a live recipient.
        self.child.close()

    def poll(self):
        if self._disabled:
            return
        try:
            for _ in range(64):
                try:
                    packet, _, flags, _ = self.parent.recvmsg(64)
                except BlockingIOError:
                    break
                if not packet:
                    break
                if flags & socket.MSG_TRUNC or packet not in {
                    b"UNKNOWN", b"PENDING", b"SESSION_RELEASED", b"OWNER_RECOVERED",
                    b"NO_NATIVE_ENTRY", b"UNRESOLVED"
                } or len(self.state["events"]) >= 64:
                    self.state["error"] = "invalid_or_excess_custody_packet"
                    self._disabled = True
                    break
                previous = self.state["status"]
                if ((previous == "unobserved" and packet not in {b"PENDING", b"UNKNOWN", b"NO_NATIVE_ENTRY"})
                        or (previous in {"session_released", "owner_recovered", "no_native_entry", "unresolved"})
                        or (self._unknown_seen and packet == b"UNKNOWN")
                        or (packet == b"OWNER_RECOVERED" and not self._unknown_seen)):
                    self.state["error"] = "invalid_custody_sequence"
                    self._disabled = True
                    break
                event = packet.decode("ascii")
                self.state["events"].append(event)
                self.state["status"] = event.lower()
                if packet == b"UNKNOWN":
                    self._unknown_seen = True
                if self._unknown_seen and packet in {b"UNKNOWN", b"PENDING"} and self.state["requests"] < 32:
                    try:
                        self.parent.send(b"REOBSERVE", getattr(socket, "MSG_NOSIGNAL", 0))
                        self.state["requests"] += 1
                    except BlockingIOError:
                        self.state["error"] = "reobserve_send_would_block"
                        self._disabled = True
                        break
        except OSError as error:
            self.state["error"] = type(error).__name__
            self._disabled = True

    def close(self):
        self.child.close()
        self.parent.close()
