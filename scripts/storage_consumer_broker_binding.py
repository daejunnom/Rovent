"""Original-manager catalog and registry guards for an input broker.

This factory is inert until an original supervisor explicitly invokes it.
It covers only the supplied, already-bound catalogs and does not establish
the absence of legacy or nonparticipating readers.
"""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
import os
from pathlib import Path
import stat

from managed_storage import StorageError, checked
from storage_consumer_registry import require_catalog_binding
from storage_guard import CatalogGuard


def _ordered_roots(manager, catalogs):
    if not isinstance(catalogs, (list, tuple)) or len(catalogs) > 32:
        raise StorageError("bounded participating catalogs required")
    root = checked(Path(manager.root))
    roots = sorted({root, *(checked(Path(path)) for path in catalogs)},
                   key=lambda path: os.path.normcase(str(path)))
    if len(roots) > 32 or roots[0] != root:
        raise StorageError("original manager catalog must be first in lock order")
    return roots


@contextmanager
def manager_registry_transaction(manager, catalogs):
    """Hold original manager guard, additional catalog guards, then registry.

    Never reopen the manager's lock: doing so would contend with the original
    owner.  An unavailable or differently bound additional catalog is an
    error for this participating transaction, not an adoption or deletion.
    """
    roots = _ordered_roots(manager, catalogs)
    registry = manager.dependency_registry
    original_guard = manager.guard
    if registry is None:
        raise StorageError("original manager's bound registry required")
    manager._assert_owner()
    with ExitStack() as stack:
        catalog_guards = []
        for root in roots:
            guard = original_guard if root == roots[0] else stack.enter_context(CatalogGuard(root / "catalog.lock"))
            require_catalog_binding(root, registry, guard)
            catalog_guards.append((root, guard))
        if manager.guard is not original_guard or manager.dependency_registry is not registry:
            raise StorageError("original manager guard or registry changed")
        manager._assert_owner()
        guard = stack.enter_context(registry.guard())
        try:
            yield guard
        finally:
            manager._assert_owner()
            if manager.guard is not original_guard or manager.dependency_registry is not registry:
                raise StorageError("original manager guard or registry changed")
            for root, catalog_guard in catalog_guards:
                require_catalog_binding(root, registry, catalog_guard)
            guard.check()


class PinnedBrokerDomain:
    """Keep original registry and participating catalog identities open.

    A pin never adopts a catalog or permits a deletion. Caller must retain the
    original manager object throughout, and close this domain only after the
    broker has no transaction and its original child scope has been handled.
    """

    def __init__(self, manager, catalogs):
        if os.name != "posix" or not hasattr(os, "O_NOFOLLOW"):
            raise StorageError("POSIX no-follow descriptor pinning required")
        self._manager = manager
        self._registry = manager.dependency_registry
        self._roots = _ordered_roots(manager, catalogs)
        self._entries = []
        self._active = False
        self._closed = False
        self._close_unknown = False
        try:
            with manager_registry_transaction(manager, self._roots):
                registry_root = checked(Path(self._registry.root))
                self._registry_record = self._registry._check_identity()
                self._pin(registry_root, directory=True)
                self._pin(registry_root / "registry.json")
                self._pin(registry_root / "registry.lock")
                for root in self._roots:
                    self._pin(root, directory=True)
                    self._pin(root / "catalog.lock")
                    self._pin(root / "consumer-registry.binding")
                self.verify()
        except BaseException as error:
            try:
                self.close()
            except BaseException as close_error:
                if hasattr(error, "add_note"):
                    error.add_note("broker domain cleanup unknown: " + type(close_error).__name__)
            raise

    @property
    def close_unknown(self):
        """Some close failed; fd state is unknown and its number is not retried."""
        return self._close_unknown

    def _pin(self, path, *, directory=False):
        flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
        if directory:
            flags |= os.O_DIRECTORY
        fd = os.open(path, flags)
        try:
            info = os.fstat(fd)
            if ((directory and not stat.S_ISDIR(info.st_mode))
                    or (not directory and (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1))):
                raise StorageError("broker domain pin has invalid type or links")
            named = os.stat(path, follow_symlinks=False)
            if (info.st_dev, info.st_ino) != (named.st_dev, named.st_ino):
                raise StorageError("broker domain pin changed while opening")
            self._entries.append((path, fd, (info.st_dev, info.st_ino), directory))
        except BaseException:
            os.close(fd)
            raise

    def verify(self):
        if self._closed or self._manager.dependency_registry is not self._registry:
            raise StorageError("original broker domain unavailable")
        self._manager._assert_owner()
        for path, fd, identity, directory in self._entries:
            opened = os.fstat(fd)
            named = os.stat(path, follow_symlinks=False)
            if ((opened.st_dev, opened.st_ino) != identity
                    or (named.st_dev, named.st_ino) != identity
                    or (directory and not stat.S_ISDIR(opened.st_mode))
                    or (not directory and (not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1))):
                raise StorageError("pinned broker domain path replaced")
        if self._registry._check_identity() != self._registry_record:
            raise StorageError("pinned broker registry identity changed")

    @contextmanager
    def transaction(self):
        if self._active:
            raise StorageError("broker domain transaction already active")
        self.verify()
        self._active = True
        try:
            with manager_registry_transaction(self._manager, self._roots) as guard:
                self.verify()
                try:
                    yield guard
                finally:
                    self.verify()
        finally:
            self._active = False

    def close(self):
        if self._active:
            raise StorageError("cannot close an active broker domain")
        if not self._closed:
            self._closed = True
            first_error = None
            while self._entries:
                _, fd, _, _ = self._entries.pop()
                try:
                    os.close(fd)
                except OSError as error:
                    self._close_unknown = True
                    if first_error is None:
                        first_error = error
            if first_error is not None:
                raise StorageError("one or more pinned descriptor closes are unknown") from first_error
