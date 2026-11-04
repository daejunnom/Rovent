"""Bounded in-process custody observations, never a GPU completion fence."""
from __future__ import annotations

import socket
import time


class CustodyTransport:
    """Own a Linux seqpacket pair for one existing supervised child.

    Only the child retains GPU-related owners. This channel reports their
    process-local state; no packet or child exit establishes a device fence.
    """

    def __init__(self, *, allow_native=None, reobserve_interval_ns=0, clock_ns=time.monotonic_ns):
        self.parent, self.child = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        self.parent.setblocking(False)
        self.state = {"status": "unobserved", "events": [], "requests": 0,
                      "error": None, "scope": "per_spawn_private_channel_peer_pid_unverified",
                      "native_authorization": "not_requested"}
        self._disabled = False
        self._send_disabled = False
        self._unknown_seen = False
        self._native_ready_seen = False
        self._allow_native = allow_native
        self._clock_ns = clock_ns
        self._reobserve_interval_ns = reobserve_interval_ns
        self._reobserve_after_ns = 0
        self._reobserve_requested = False

    def child_fd(self):
        return self.child.fileno()

    def child_started(self):
        # Popen has inherited its own duplicate. The supervisor must not retain
        # a peer endpoint and mistake an exited child for a live recipient.
        self.child.close()

    def stop_sending(self, error):
        self.state["error"] = error
        self._send_disabled = True

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
                    b"UNKNOWN", b"PENDING", b"NATIVE_READY", b"SESSION_RELEASED", b"OWNER_RECOVERED",
                    b"NO_NATIVE_ENTRY", b"UNRESOLVED"
                } or len(self.state["events"]) >= 64:
                    self.state["error"] = "invalid_or_excess_custody_packet"
                    self._disabled = True
                    break
                previous = self.state["status"]
                if ((previous == "unobserved" and packet not in {b"PENDING", b"UNKNOWN", b"NO_NATIVE_ENTRY"})
                        or (previous in {"session_released", "owner_recovered", "no_native_entry", "unresolved"})
                        or (self._unknown_seen and packet == b"UNKNOWN")
                        or (packet == b"OWNER_RECOVERED" and not self._unknown_seen)
                        or (packet == b"NATIVE_READY" and (previous != "pending" or self._native_ready_seen
                                                           or self._unknown_seen))
                        or (packet == b"SESSION_RELEASED" and self.state["native_authorization"] != "sent")):
                    self.state["error"] = "invalid_custody_sequence"
                    self._disabled = True
                    break
                event = packet.decode("ascii")
                self.state["events"].append(event)
                self.state["status"] = event.lower()
                if packet == b"UNKNOWN":
                    self._unknown_seen = True
                if packet == b"NATIVE_READY":
                    self._native_ready_seen = True
                    self.state["native_authorization"] = "pending"
                if self._unknown_seen and packet in {b"UNKNOWN", b"PENDING"} and self.state["requests"] < 32:
                    self._reobserve_requested = True
        except OSError as error:
            self.state["error"] = type(error).__name__
            self._disabled = True
        # Drain every already-queued terminal observation before writing. A
        # closed peer must not conceal its earlier SESSION_RELEASED packet.
        if self._disabled or self._send_disabled:
            return
        if self.state["native_authorization"] == "pending":
            if self.state["status"] != "native_ready":
                self.state["native_authorization"] = "not_sent_child_moved_on"
            else:
                try:
                    allowed = self._allow_native is not None and self._allow_native()
                except Exception as error:
                    self.state["error"] = type(error).__name__
                    allowed = False
                if not allowed:
                    self.state["native_authorization"] = "not_sent_deadline_or_policy"
                elif self._send(b"ALLOW_NATIVE"):
                    self.state["native_authorization"] = "sent"
                else:
                    self.state["native_authorization"] = "send_failed"
        if (self._reobserve_requested and self.state["status"] in {"unknown", "pending"}
                and not self._send_disabled and self._clock_ns() >= self._reobserve_after_ns):
            if self._send(b"REOBSERVE"):
                self.state["requests"] += 1
                self._reobserve_requested = False
                self._reobserve_after_ns = self._clock_ns() + self._reobserve_interval_ns

    def _send(self, packet):
        try:
            self.parent.send(packet, getattr(socket, "MSG_NOSIGNAL", 0))
            return True
        except OSError as error:
            self.state["error"] = type(error).__name__
            self._send_disabled = True
            # Reading remains enabled: a queued terminal observation may
            # still arrive after a failed control write.
            return False

    def close(self):
        self.child.close()
        self.parent.close()
