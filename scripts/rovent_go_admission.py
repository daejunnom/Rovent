"""Optional same-owner Linux UCI go rejection channel; never a GPU fence.

The owner must be the only writer to the child's stdin. Until an integrated
caller can enforce that and retain this receiver through child completion,
the production Rust handoff must remain disabled.
"""

import json
import os
import socket
import struct

MAX_PACKET = 4096
_CRED = struct.Struct('3i')
_FIELDS = frozenset(('schema', 'process_id', 'process_epoch', 'input_lock_sha256',
                     'go_sequence', 'game_generation', 'root_generation',
                     'classification', 'prior_search', 'stage', 'code',
                     'physical_completion'))


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate packet field')
        result[key] = value
    return result


class GoAdmissionChannel:
    """Dedicated inherited datagram and exact, single outstanding go tracking.

    ``child_fd`` may be passed only to the one direct child via ``pass_fds``;
    call ``after_spawn`` immediately, even on failure. Do not pass the fd
    while the child-side handoff gate is disabled: it would remain inheritable.
    A rejection is only an observation. Caller separately verifies live parent
    admission, child state, and serializes all stdin writes before acting.
    """

    def __init__(self):
        self.receiver, self._child = socket.socketpair(socket.AF_UNIX, socket.SOCK_DGRAM)
        try:
            self.receiver.setsockopt(socket.SOL_SOCKET, socket.SO_PASSCRED, 1)
            self.receiver.setblocking(False)
        except BaseException:
            self.receiver.close()
            self._child.close()
            raise
        self._sent = 0
        self._pending = None
        self._revoked = False
        self.observation_error = None
        self.last_trusted_rejection = None

    @property
    def child_fd(self):
        if self._child is None:
            raise RuntimeError('child descriptor already released')
        return self._child.fileno()

    def after_spawn(self):
        if self._child is not None:
            self._child.close()
            self._child = None

    def begin_go(self):
        if self._revoked or self._pending is not None:
            raise RuntimeError('a go is already pending or input is revoked')
        self._sent += 1
        self._pending = self._sent
        return self._pending

    def completed_go(self):
        if self._pending is None:
            raise RuntimeError('no pending go')
        self._pending = None

    def revoke(self):
        self._revoked = True
        self._pending = None

    def receive(self, *, child_pid, child_uid, epoch, lock_sha256):
        """Return a verified matching event or None; consume at most one packet.

        Invalid or missing credentials, stale events and oversize datagrams do
        not alter the pending command. In particular they never authorize stop.
        """
        try:
            raw, ancillary, flags, _ = self.receiver.recvmsg(
                MAX_PACKET + 1, socket.CMSG_SPACE(_CRED.size), socket.MSG_DONTWAIT)
        except BlockingIOError:
            return None
        except OSError as exc:
            # This optional diagnostic must not become an engine failure or
            # grant authority when the channel is damaged.
            self.observation_error = {'type': type(exc).__name__, 'errno': exc.errno}
            self.revoke()
            return None
        if (flags & (socket.MSG_TRUNC | socket.MSG_CTRUNC)) or not raw or len(raw) > MAX_PACKET:
            return None
        creds = [data for level, kind, data in ancillary
                 if level == socket.SOL_SOCKET and kind == socket.SCM_CREDENTIALS]
        if len(creds) != 1 or len(creds[0]) != _CRED.size:
            return None
        pid, uid, _gid = _CRED.unpack(creds[0])
        if pid != child_pid or uid != child_uid:
            return None
        try:
            event = json.loads(raw.decode('utf-8'), object_pairs_hook=_unique,
                               parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite')))
        except (UnicodeError, ValueError, TypeError, RecursionError):
            return None
        if not isinstance(event, dict) or set(event) != _FIELDS:
            return None
        for key in ('process_id', 'process_epoch', 'go_sequence', 'game_generation', 'root_generation'):
            if type(event[key]) is not int or event[key] < 0:
                return None
        if (self._revoked or self._pending is None or event['go_sequence'] != self._pending
                or event['process_id'] != child_pid or event['process_epoch'] != epoch
                or event['input_lock_sha256'] != lock_sha256
                or event['schema'] != 'rz-rovent-go-admission/1'
                or event['classification'] != 'driver_rejected_before_ticket'
                or event['prior_search'] not in ('no_owner_search_or_drain_observed',
                                                 'search_drain_or_failure_pending')
                or event['physical_completion'] != 'not_inferred'
                or event['stage'] not in ('contract', 'admission', 'backend', 'output', 'backup')
                or event['code'] not in ('invalid_input', 'unsupported_contract', 'backend_unavailable',
                                         'resource_exhausted', 'numerical_failure', 'identity_mismatch',
                                         'backend_failure', 'canceled', 'expired', 'stale')):
            return None
        # Save the old-search/pending observation, without asserting physical
        # completion. Even a trusted event is accepted at most once.
        self.last_trusted_rejection = event
        self.revoke()
        if event['prior_search'] != 'no_owner_search_or_drain_observed':
            return None
        # Caller must still prove live owner and safe stdin ordering.
        return event

    def close_after_child_completion(self):
        self.after_spawn()
        self.receiver.close()
