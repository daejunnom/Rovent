"""Original Linux supervisor ownership adapter for an opt-in broker.

This module is not wired to ``run_managed``. The eventual caller must only
enable approved consumers whose descriptor readers and source identity have
been separately checked. Process-group completion is not a GPU fence.
"""
from __future__ import annotations

import json
import os
import sys

from managed_process import OwnedProcessGroup, ProcessNotSpawned
from managed_storage import StorageError
from storage_consumer_broker_binding import PinnedBrokerDomain, _ordered_roots
from storage_consumer_broker_supervisor import ConsumerBrokerSupervisor
from storage_consumer_broker_transport import open_pair


class ManagedBrokerLaunch:
    """Keep the original manager, process-group owner and pinned domain alive."""

    def __init__(self, manager, catalogs):
        if sys.platform != "linux" or not hasattr(os, "waitid") or not hasattr(os, "WNOWAIT"):
            raise StorageError("Linux original process supervision required")
        self.manager = manager
        self._owner_pid, self._uid, self._gid = os.getpid(), os.getuid(), os.getgid()
        self.roots = _ordered_roots(manager, catalogs)
        self.domain = PinnedBrokerDomain(manager, self.roots)
        self.group = OwnedProcessGroup()
        self._parent = self._child_endpoint = self._child = self.broker = None
        self._attempted = self._not_dispatched = self._closed = False
        self.budget_deferred_count = 0
        self.first_transport_failure = None
        try:
            self._parent, self._child_endpoint = open_pair()
        except BaseException:
            self.domain.close()
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
        if self._attempted:
            raise StorageError("broker child dispatch already attempted")
        self._attempted = True

    def mark_pre_dispatch_failure(self, error):
        if (os.getpid() != self._owner_pid or type(error) is not ProcessNotSpawned
                or self.group._child is not None or self.broker is not None):
            raise StorageError("original proven pre-dispatch failure required")
        self._not_dispatched = True

    def on_spawn(self):
        self._assert_supervisor()
        if not self._attempted or self.group._child is None or self.broker is not None:
            raise StorageError("original process group spawn required")
        self._child = self.group._child
        self._child_endpoint.close()
        self._child_endpoint = None
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
        if self._attempted and self._child is None and not self._not_dispatched:
            return False
        if self._attempted and not self._not_dispatched and not tree_gone:
            if self.broker is not None:
                self.broker.revoke()
            return False
        if self.broker is not None:
            self.broker.revoke()
            if self.broker.transaction_active and not tree_gone:
                return False
            self.broker.finish(child_never_dispatched=self._not_dispatched, tree_gone=tree_gone)
        if self._child_endpoint is not None:
            self._child_endpoint.close()
            self._child_endpoint = None
        if self._parent is not None:
            self._parent.close()
            self._parent = None
        self.domain.close()
        self._closed = True
        return True
