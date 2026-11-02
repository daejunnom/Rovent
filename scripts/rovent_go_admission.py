"""Optional same-owner Linux UCI go rejection channel; never a GPU fence.

The owner must be the only writer to the child's stdin. Until an integrated
caller can enforce that and retain this receiver through child completion,
the production Rust handoff must remain disabled.
"""

import json
import os
import socket
import struct
import threading

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


class OriginalOwnerGoSupervisor:
    """Serialize this owner's stdin and handle one authenticated rejected go.

    The caller must route *every* stdin command through this object, retain the
    original Popen and admission connection, and preserve the normal shutdown
    and physical-completion evidence path. This object is not a GPU fence.
    """

    def __init__(self, process, engine_io, channel, *, epoch, lock_sha256):
        if engine_io.process is not process:
            raise ValueError('engine stream and original process differ')
        self.process, self.engine_io, self.channel = process, engine_io, channel
        self.epoch, self.lock_sha256 = epoch, lock_sha256
        self.uid = os.geteuid()
        self.pending = False
        self.frozen = False
        self.rejection = None
        self.observation_error = None
        self._shutdown_attempted = False
        self._observer_disabled = False
        self._owner_thread = threading.current_thread()

    def _assert_owner_thread(self):
        if threading.current_thread() is not self._owner_thread:
            raise RuntimeError('stdin control belongs to the original owner thread')

    def _disable_observer(self, reason):
        self.observation_error = reason
        self._observer_disabled = True
        self.channel.revoke()

    def write(self, command):
        self._assert_owner_thread()
        if self.frozen or self.pending:
            raise RuntimeError('original stdin is sealed or a go is pending')
        # Match the engine parser's ASCII, single-line and tab/space token
        # rules. In particular an embedded newline must not smuggle a go.
        if (not isinstance(command, str) or not command.isascii()
                or any((ord(ch) < 32 and ch != '\t') or ord(ch) == 127 for ch in command)):
            raise ValueError('one ASCII UCI command without control characters required')
        fields = command.split()
        is_go = bool(fields) and fields[0] == 'go'
        if is_go:
            if not self._observer_disabled:
                try:
                    self.channel.begin_go()
                except (OSError, RuntimeError) as exc:
                    self._disable_observer({'type': type(exc).__name__,
                                            'errno': getattr(exc, 'errno', None)})
            self.pending = True
        try:
            self.engine_io.write(command)
        except BaseException:
            # The child may have received a partial command; never treat it
            # as a known rejection or issue an inferred shutdown.
            self.channel.revoke()
            self.frozen = True
            raise

    def bestmove(self):
        self._assert_owner_thread()
        if not self.pending or self.frozen:
            raise RuntimeError('bestmove has no original pending go')
        # Losing an optional diagnostic cannot invalidate an otherwise valid
        # bestmove. A trusted report of a prior active search is different:
        # that result cannot safely be attributed to the rejected go.
        if self.channel.last_trusted_rejection is not None:
            raise RuntimeError('bestmove attribution is ambiguous after rejected go')
        if not self._observer_disabled:
            self.channel.completed_go()
        self.pending = False

    def _same_live_owner(self):
        admission = self.engine_io.parent_admission
        if (self.engine_io.process is not self.process or admission is None
                or admission.process is not self.process or not admission.confirmed
                or admission.epoch != self.epoch or admission.lock_sha != self.lock_sha256
                or admission.peer is None or admission.peer.fileno() < 0):
            return False
        # Never reap the direct child: the owner retains the wait relationship
        # through its process-group census and evidence-preservation sequence.
        if os.waitid(os.P_PID, self.process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is not None:
            return False
        pid, uid, _ = _CRED.unpack(admission.peer.getsockopt(
            socket.SOL_SOCKET, socket.SO_PEERCRED, _CRED.size))
        if pid != self.process.pid or uid != self.uid:
            return False
        try:
            # The admission stream carries no further application frames.
            # EOF or unexpected data makes its current liveness ambiguous.
            admission.peer.recv(1, socket.MSG_PEEK | socket.MSG_DONTWAIT)
            return False
        except BlockingIOError:
            return True

    def poll_rejection(self):
        """Return a same-owner rejection; never stop a prior/unknown search."""
        self._assert_owner_thread()
        if not self.pending or self.frozen or self._observer_disabled:
            return None
        try:
            if not self._same_live_owner():
                self._disable_observer({'type': 'OriginalPeerUnavailable', 'errno': None})
                return None
            event = self.channel.receive(child_pid=self.process.pid, child_uid=self.uid,
                                         epoch=self.epoch, lock_sha256=self.lock_sha256)
            if self.channel.observation_error is not None:
                self._disable_observer(self.channel.observation_error)
                return None
            if event is None:
                return None
            if not self._same_live_owner():
                self._disable_observer({'type': 'OriginalPeerUnavailable', 'errno': None})
                return None
        except (OSError, ValueError, struct.error) as exc:
            # A diagnostic or liveness-check failure is not a model failure.
            self._disable_observer({'type': type(exc).__name__, 'errno': getattr(exc, 'errno', None)})
            return None
        self.frozen = True
        self.pending = False
        self.rejection = event
        return event

    def stop_and_quit_after_rejection(self):
        self._assert_owner_thread()
        if not self.frozen or self.rejection is None or self._shutdown_attempted:
            raise RuntimeError('same-owner rejected go not established')
        if not self._same_live_owner():
            raise RuntimeError('original child or admission is no longer live')
        # No new stdin command can be sent through write() after freeze.
        # A write failure leaves the ordinary original-owner quarantine path.
        self._shutdown_attempted = True
        self.engine_io.write('stop')
        if not self._same_live_owner():
            raise RuntimeError('original child or admission changed after stop')
        self.engine_io.write('quit')
