# SPDX-License-Identifier: Apache-2.0
"""A fixed pool of interprocess device events, one per registered KV cache.

The server creates and exports the pool once per worker instance; the worker
imports it once. A retrieve then signals a layer batch by recording one pool
event, and names it by index, so no event is exported per request.
"""

# Future
from __future__ import annotations

# First Party
from lmcache import torch_dev
from lmcache.v1.platform.base.event_ipc import EventIPCBackend


class EventPool:
    """Interprocess events addressed by index.

    Args:
        backend: The event IPC backend of the device.
        device: The device the events belong to.
        size: Number of events.
    """

    def __init__(self, backend: EventIPCBackend, device: object, size: int) -> None:
        self._backend = backend
        # The server's current device may differ from the instance's.
        with torch_dev.device(device):
            self._events = [backend.create_event(device) for _ in range(size)]
            self._handles = [backend.export_event(e, device) for e in self._events]

    @classmethod
    def import_pool(
        cls, backend: EventIPCBackend, device: object, handles: list[bytes]
    ) -> EventPool:
        """Open a pool another process exported.

        Args:
            backend: The event IPC backend of the device.
            device: The device the events belong to.
            handles: The exporter's :attr:`handles`.
        """
        pool = object.__new__(cls)
        pool._backend = backend
        pool._events = [backend.import_event(h, device) for h in handles]
        pool._handles = list(handles)
        return pool

    @property
    def size(self) -> int:
        """Number of events."""
        return len(self._events)

    @property
    def handles(self) -> list[bytes]:
        """Exported IPC handle of every event, by index."""
        return self._handles

    def record(self, index: int, stream: object) -> None:
        """Record event ``index`` on ``stream``."""
        self._backend.record_event(self._events[index], stream)

    def native_event(self, index: int) -> int:
        """Raw device handle of event ``index`` (a ``cudaEvent_t``), for
        native code to record; 0 if the backend's events expose none."""
        return int(getattr(self._events[index], "cuda_event", 0))

    def wait(self, index: int, stream: object) -> None:
        """Make ``stream`` wait for event ``index``'s latest record."""
        self._backend.wait_event(self._events[index], stream)
