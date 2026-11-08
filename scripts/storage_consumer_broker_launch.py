"""Original Linux supervisor ownership adapter for an opt-in broker.

``run_managed`` connects this module only through an explicit reviewed-reader
opt-in; automatic enrollment is disabled. The caller must ensure the reader's
declared scope and source identity have been separately checked. Process-group
completion is not a GPU fence.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import stat
import sys

from managed_process import OwnedProcessGroup, ProcessNotSpawned
from managed_storage import StorageError
from storage_consumer_broker_binding import PinnedBrokerDomain, _ordered_roots
from storage_consumer_broker_profile import ObservedReaderProfile, ReviewedReaderContract
from storage_consumer_broker_supervisor import ConsumerBrokerSupervisor
from storage_consumer_broker_transport import open_pair


def _same_process_cwd(pid, source):
    """Compare one Linux child cwd observation with the original source inode."""
    child_cwd = os.stat(Path("/proc") / str(pid) / "cwd")
    source_cwd = os.stat(source)
    return (stat.S_ISDIR(child_cwd.st_mode) and stat.S_ISDIR(source_cwd.st_mode)
            and (child_cwd.st_dev, child_cwd.st_ino) == (source_cwd.st_dev, source_cwd.st_ino))


class ManagedBrokerLaunch:
    """Keep the original manager, process-group owner and pinned domain alive."""

    def __init__(self, manager, catalogs, *, reader_contract=None, reader_profile=None):
        if sys.platform != "linux" or not hasattr(os, "waitid") or not hasattr(os, "WNOWAIT"):
            raise StorageError("Linux original process supervision required")
        self.manager = manager
        if reader_contract is not None and (type(reader_contract) is not ReviewedReaderContract
                                            or reader_contract.profile._manager is not manager):
            raise StorageError("original manager's reviewed reader contract required")
        if reader_profile is not None and (reader_contract is not None or type(reader_profile) is not ObservedReaderProfile
                                           or reader_profile._manager is not manager):
            raise StorageError("original manager's observed reader profile required")
        self.reader_contract = reader_contract
        self.reader_profile = reader_contract.profile if reader_contract is not None else reader_profile
        self._command = None if self.reader_profile is None else self.reader_profile.command
        self._owner_pid, self._uid, self._gid = os.getpid(), os.getuid(), os.getgid()
        self.roots = _ordered_roots(manager, catalogs)
        self.domain = PinnedBrokerDomain(manager, self.roots)
        self.group = OwnedProcessGroup()
        self._parent = self._child_endpoint = self._child = self.broker = None
        self._attempted = self._not_dispatched = self._closed = False
        self._spawn_armed = False
        self.budget_deferred_count = 0
        self.first_transport_failure = None
        self.child_cwd_observation = "not_observed"
        try:
            self._parent, self._child_endpoint = open_pair()
        except BaseException as error:
            try:
                self.domain.close()
            except BaseException as close_error:
                if hasattr(error, "add_note"):
                    error.add_note("broker domain close unknown: " + type(close_error).__name__)
            raise

    def environment(self):
        self._assert_supervisor()
        if self._attempted or self._child_endpoint is None:
            raise StorageError("broker child admission already attempted")
        env = self.manager.environment()
        env.update(RZ_CONSUMER_BROKER_FD=str(self._child_endpoint.fileno()),
                   RZ_CONSUMER_BROKER_PARENT_PID=str(self._owner_pid),
                   RZ_CONSUMER_INPUT_CATALOGS=json.dumps([str(root) for root in self.roots], ensure_ascii=True))
        return env

    @property
    def command(self):
        return self._command

    def assert_spawn_command(self, command, cwd):
        """Bind this exact immutable argv and original source cwd to spawn."""
        self._assert_supervisor()
        if (self.reader_profile is None or self._command is None
                or command is not self._command or self.reader_profile.command is not self._command
                or os.path.realpath(os.fspath(cwd)) != os.path.realpath(os.fspath(self.manager.source))
                or not self.reader_profile.unchanged()):
            raise StorageError("original profiled command and source cwd required")
        if self._attempted:
            raise StorageError("broker spawn already attempted")
        self._spawn_armed = True

    @property
    def child_fds(self):
        if self._attempted or self._child_endpoint is None:
            raise StorageError("original broker child FD unavailable")
        return (self._child_endpoint.fileno(),)

    def _assert_supervisor(self):
        if os.getpid() != self._owner_pid or self._closed:
            raise StorageError("original broker supervisor unavailable")
        self.manager._assert_owner()
        self.domain.verify()

    def _assert_child(self):
        self._assert_supervisor()
        if (self._child is None or self.group._owner_pid != self._owner_pid
                or self.group._child is not self._child or self.group._completed):
            raise StorageError("original broker child scope unavailable")
        status = os.waitid(os.P_PID, self._child.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
        if status is not None:
            raise StorageError("original broker child already exited")

    def _assert_peer(self, pid, uid, gid):
        self._assert_child()
        if (pid, uid, gid) != (self._child.pid, self._uid, self._gid):
            raise StorageError("broker sender is not the original direct child")

    def mark_dispatch_attempted(self):
        self._assert_supervisor()
        if self._attempted or not self._spawn_armed:
            raise StorageError("verified broker spawn required")
        self._attempted = True

    def mark_pre_dispatch_failure(self, error):
        if (os.getpid() != self._owner_pid or type(error) is not ProcessNotSpawned
                or self.group._child is not None or self.broker is not None):
            raise StorageError("original proven pre-dispatch failure required")
        self._not_dispatched = True

    def on_spawn(self):
        if self.group._child is None or self._child is not None or self.broker is not None:
            raise StorageError("original process group spawn required")
        # Keep the original child object even when subsequent validation fails;
        # the same process group must still be drained by run_group.
        self._child = self.group._child
        self._child_endpoint.close()
        self._child_endpoint = None
        self._assert_supervisor()
        if not self._attempted or not self._spawn_armed or self._child.args is not self._command:
            raise StorageError("original profiled command required")
        try:
            self.child_cwd_observation = ("matches" if _same_process_cwd(self._child.pid, self.manager.source)
                                          else "differs")
        except OSError:
            self.child_cwd_observation = "unavailable"
        self.broker = ConsumerBrokerSupervisor(self._parent, self.domain.transaction,
                                               self._assert_supervisor, self._assert_child,
                                               self._assert_peer)

    def on_tick(self):
        if self.broker is not None:
            try:
                self.broker.service()
            except BaseException as error:
                if self.first_transport_failure is None:
                    self.first_transport_failure = type(error).__name__
                raise

    def defer_budget(self):
        """Caller records and promptly resamples deferred budget observations."""
        active = self.broker is not None and self.broker.transaction_active
        if active:
            self.budget_deferred_count += 1
        return active

    def on_stop(self):
        if self.broker is not None:
            self.broker.revoke()

    def revoke_unconfirmed_spawn(self):
        """Revoke unserved admission without deciding whether Popen started.

        This is valid only before an original child was bound and before a
        broker existed, so no registration ACK or broker lease was issued.
        It does not close the manager owner or claim that the process is gone.
        """
        self._assert_supervisor()
        if (not self._attempted or self._not_dispatched or self.group._child is not None
                or self._child is not None or self.broker is not None):
            raise StorageError("unbound unconfirmed broker spawn required")
        self._close_resources()

    def _close_resources(self):
        close_errors = []
        for endpoint in (self._child_endpoint, self._parent):
            if endpoint is not None:
                try:
                    endpoint.close()
                except OSError as error:
                    close_errors.append(error)
        self._child_endpoint = self._parent = None
        if self.reader_profile is not None:
            try:
                self.reader_profile.close()
            except (OSError, StorageError) as error:
                close_errors.append(error)
        try:
            self.domain.close()
        except (OSError, StorageError) as error:
            close_errors.append(error)
        self._closed = True
        if close_errors:
            raise StorageError("broker descriptor close outcome unknown") from close_errors[0]

    def finish(self, *, tree_gone):
        """Return False while a dispatched original group remains unresolved.

        In that case the caller must keep this exact object and the original
        manager/group alive; a later bounded same-owner reobservation is needed.
        """
        self._assert_supervisor()
        if type(tree_gone) is not bool:
            raise StorageError("explicit original group observation required")
        if tree_gone and (self._child is None or self.group._child is not self._child
                          or self.group._completed is not True):
            raise StorageError("original group completion not observed")
        if self._child is not None and not self._attempted and not tree_gone:
            return False
        if self._attempted and self._child is None and not self._not_dispatched:
            return False
        if self._attempted and not self._not_dispatched and not tree_gone:
            if self.broker is not None:
                self.broker.revoke()
                if self.reader_contract is not None and self.reader_contract.allows_unknown_abandon():
                    self.broker.abandon_unknown(reader_scope_proven=True)
                else:
                    return False
            else:
                return False
        if self.broker is not None:
            self.broker.revoke()
            if tree_gone or self._not_dispatched:
                self.broker.finish(child_never_dispatched=self._not_dispatched, tree_gone=tree_gone)
        self._close_resources()
        return True
