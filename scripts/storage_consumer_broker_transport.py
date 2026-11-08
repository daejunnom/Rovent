"""Bounded Linux local transport for a future consumer-input broker.

This module is not an ownership or authorization proof by itself. Callers
must compare every kernel-supplied sender identity with the original live
supervisor or direct child process object, and must separately supervise
leases, deadlines, guards and physical completion.
"""
from __future__ import annotations

import array
import json
import math
import os
import socket
import struct

MAX_PACKET_BYTES = 64 * 1024
_CREDENTIAL_BYTES = struct.calcsize("3i")
_ANCILLARY_BYTES = (socket.CMSG_SPACE(_CREDENTIAL_BYTES) + socket.CMSG_SPACE(32 * array.array("i").itemsize)
                    if hasattr(socket, "CMSG_SPACE") else 0)


class BrokerTransportError(RuntimeError):
    pass


def open_pair():
    """Create private packet endpoints; neither is inheritable by default."""
    if os.name != "posix" or not all(hasattr(socket, name) for name in
                                      ("SO_PASSCRED", "SCM_CREDENTIALS", "SOCK_SEQPACKET", "MSG_CMSG_CLOEXEC")):
        raise BrokerTransportError("Linux credentialled Unix packets required")
    left, right = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    try:
        for endpoint in (left, right):
            endpoint.set_inheritable(False)
            endpoint.setsockopt(socket.SOL_SOCKET, socket.SO_PASSCRED, 1)
            endpoint.setblocking(False)
    except BaseException:
        left.close()
        right.close()
        raise
    return left, right


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _finite_float(value):
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("nonfinite JSON number")
    return parsed


def send_packet(endpoint, message):
    """Send one complete packet, or raise without claiming delivery."""
    if type(message) is not dict:
        raise BrokerTransportError("JSON object required")
    try:
        data = json.dumps(message, ensure_ascii=True, allow_nan=False,
                          separators=(",", ":")).encode("ascii")
    except (TypeError, ValueError, OverflowError, RecursionError) as error:
        raise BrokerTransportError("invalid broker message") from error
    if not 1 <= len(data) <= MAX_PACKET_BYTES:
        raise BrokerTransportError("bounded nonempty packet required")
    try:
        written = endpoint.send(data)
    except OSError as error:
        raise BrokerTransportError("packet delivery unknown") from error
    if written != len(data):
        raise BrokerTransportError("partial packet delivery unknown")


def receive_packet(endpoint, assert_peer):
    """Return None if no packet, else a credential-checked object.

    EOF, malformed framing, ancillary descriptors and truncation are failures.
    Every delivered descriptor is closed even when its packet is rejected.
    ``assert_peer(pid, uid, gid)`` must validate the original process lifetime;
    a PID comparison alone is insufficient.
    """
    if not callable(assert_peer):
        raise ValueError("original peer validator required")
    if not hasattr(socket, "MSG_CMSG_CLOEXEC"):
        raise BrokerTransportError("atomic close-on-exec receipt is unavailable")
    try:
        data, ancillary, flags, _ = endpoint.recvmsg(MAX_PACKET_BYTES + 1, _ANCILLARY_BYTES,
                                                      socket.MSG_CMSG_CLOEXEC)
    except BlockingIOError:
        return None
    except OSError as error:
        raise BrokerTransportError("packet receive failed") from error
    credentials = []
    unexpected = False
    for level, kind, value in ancillary:
        if level == socket.SOL_SOCKET and kind == socket.SCM_RIGHTS:
            received = array.array("i")
            usable = len(value) - (len(value) % received.itemsize)
            received.frombytes(value[:usable])
            for fd in received:
                try:
                    os.close(fd)
                except OSError:
                    pass
            unexpected = True
        elif level == socket.SOL_SOCKET and kind == socket.SCM_CREDENTIALS and len(value) == _CREDENTIAL_BYTES:
            credentials.append(struct.unpack("3i", value))
        else:
            unexpected = True
    if (not data or len(data) > MAX_PACKET_BYTES or unexpected or len(credentials) != 1
            or flags & (socket.MSG_TRUNC | socket.MSG_CTRUNC)):
        raise BrokerTransportError("closed, truncated or invalid broker packet")
    pid, uid, gid = credentials[0]
    try:
        assert_peer(pid, uid, gid)
    except Exception as error:
        raise BrokerTransportError("original peer identity unavailable") from error
    try:
        message = json.loads(data.decode("ascii"), object_pairs_hook=_unique_object,
                             parse_float=_finite_float,
                             parse_constant=lambda _: (_ for _ in ()).throw(ValueError("invalid number")))
    except (UnicodeError, ValueError, RecursionError) as error:
        raise BrokerTransportError("invalid broker JSON") from error
    if type(message) is not dict:
        raise BrokerTransportError("JSON object required")
    return message
