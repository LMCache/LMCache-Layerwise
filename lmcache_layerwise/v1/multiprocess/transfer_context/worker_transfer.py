# SPDX-License-Identifier: Apache-2.0
"""Worker side of the layer-wise retrieve.

:class:`LayerwiseTransferContext` is LMCache's LMCache-driven transfer
context with retrieves sent as ``layerwise.retrieve``. Its futures let the
engine wait for one layer at a time instead of the whole retrieve.
"""

# Standard
from collections.abc import Sequence
from typing import Any, Optional
import itertools
import threading
import time
import weakref

# Third Party
import msgspec
import torch
import zmq

# First Party
from lmcache import torch_dev
from lmcache.logging import init_logger
from lmcache.utils import EngineType
from lmcache.v1.gpu_connector.utils import LayoutHints
from lmcache.v1.mp_observability.errors import LMCacheTimeoutError
from lmcache.v1.multiprocess.futures import MessagingFuture
from lmcache.v1.multiprocess.group_view import EngineGroupInfo
from lmcache.v1.multiprocess.protocols.server_module import (
    ServerModuleCallRequest,
    ServerModuleCallResponse,
)
from lmcache.v1.multiprocess.transfer_context.worker_transfer import (
    IPCEvent,
    LMCacheDrivenTransferContext,
)
from lmcache.v1.multiprocess.transport.base import RequestClient
from lmcache.v1.platform.base.event_ipc import EventIPCBackend

# First Party
from lmcache_layerwise import debug
from lmcache_layerwise.v1.multiprocess import protocol
from lmcache_layerwise.v1.platform.base.event_pool import EventPool

logger = init_logger(__name__)

# How long wait_for_layer() waits for a layer before giving up.
LAYER_TIMEOUT_S = 60.0
# How often the frame reader checks whether it was closed.
_FRAME_POLL_MS = 100
# How long register() waits for a probe frame before probing again.
_PROBE_INTERVAL_S = 0.02
# How often a waiting wait_for_layer() checks whether the retrieve already
# answered. Frames wake it immediately; this only bounds the wait when they
# do not arrive, such as for a rejected retrieve.
_RESPONSE_CHECK_S = 0.05


class _Call:
    """Frames received so far for one retrieve.

    With ``LMCACHE_LAYERWISE_DEBUG``, ``arrivals`` holds each frame's KV cache
    count and arrival time in seconds after the submit.
    """

    __slots__ = ("__weakref__", "arrivals", "cond", "leaders", "start")

    def __init__(self) -> None:
        self.cond = threading.Condition()
        self.leaders: dict[int, int] = {}
        self.start = time.monotonic()
        self.arrivals: list[tuple[int, float]] | None = [] if debug.ENABLED else None

    def add(self, frame: protocol.Frame) -> None:
        with self.cond:
            for layer in frame.layers:
                self.leaders[layer] = frame.leader
            if self.arrivals is not None:
                self.arrivals.append((len(frame.layers), time.monotonic() - self.start))
            self.cond.notify_all()


class _WaitStats:
    """``LMCACHE_LAYERWISE_DEBUG`` counters of one retrieve's layer waits."""

    __slots__ = ("blocked", "blocked_s", "calls", "logged", "stream_waits", "total_s")

    def __init__(self) -> None:
        self.calls = 0
        self.stream_waits = 0
        self.blocked = 0
        self.blocked_s = 0.0
        self.total_s = 0.0
        self.logged = False


class LayerwiseRetrieveFuture(MessagingFuture[bool]):
    """Future of one layer-wise retrieve.

    ``query``, ``wait``, ``result`` and ``wait_on_stream`` behave as for
    LMCache's retrieve future: they cover the whole retrieve and yield
    whether it succeeded. :meth:`wait_for_layer` orders the current stream
    after one layer only.
    """

    def __init__(
        self,
        call: _Call,
        raw: MessagingFuture[ServerModuleCallResponse],
        *,
        device: torch.device,
        event_backend: EventIPCBackend,
        pool: EventPool,
        layer_timeout: float = LAYER_TIMEOUT_S,
    ) -> None:
        super().__init__()
        self._call = call
        self._raw = raw
        self._device = device
        self._event_backend = event_backend
        self._pool = pool
        self._layer_timeout = layer_timeout
        self._response: protocol.RetrieveResponse | None = None
        self._done_event: Any = None
        self._last_waited: int | None = None
        self._stats = _WaitStats() if debug.ENABLED else None

    def _resolve(self) -> protocol.RetrieveResponse:
        """Decode the answer; the raw future must be done."""
        if self._response is None:
            envelope = self._raw.result()
            if not envelope.success:
                logger.error("layerwise.retrieve failed: %s", envelope.error)
                response = protocol.RetrieveResponse(b"", False, [])
            else:
                try:
                    response = protocol.decode(
                        envelope.payload, protocol.RetrieveResponse
                    )
                except msgspec.DecodeError:
                    logger.exception("Malformed layerwise.retrieve response")
                    response = protocol.RetrieveResponse(b"", False, [])
            if response.event_handle:
                self._done_event = self._event_backend.import_event(
                    response.event_handle, self._device
                )
            self._response = response
        return self._response

    def _leader_of(self, layer: int) -> int | None:
        """The pool event index to wait on for ``layer``, from its frame or
        else from the answer; None if the retrieve submitted nothing.

        Raises:
            LMCacheTimeoutError: If neither arrives in time.
        """
        call = self._call
        deadline = time.monotonic() + self._layer_timeout
        with call.cond:
            while True:
                leader = call.leaders.get(layer)
                if leader is not None:
                    return leader
                if self._raw.query():
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise LMCacheTimeoutError(
                        f"Timed out waiting for the KV of layer {layer}"
                    )
                call.cond.wait(min(remaining, _RESPONSE_CHECK_S))
        leaders = self._resolve().leaders
        if not leaders:
            return None
        if not 0 <= layer < len(leaders) or leaders[layer] < 0:
            raise RuntimeError(
                f"The layer-wise retrieve has no event for KV cache {layer}."
            )
        return leaders[layer]

    def wait_for_layer(self, layer: int) -> None:
        """Make the current stream wait until KV cache ``layer`` has landed.

        Returns without waiting if the retrieve submitted nothing; it then
        failed, and the caller recomputes.

        Args:
            layer: Registration index of the KV cache.

        Raises:
            LMCacheTimeoutError: If the layer is not reported in time.
        """
        stats = self._stats
        if stats is None:
            self._wait_for_layer(layer)
            return
        start = time.monotonic()
        ready = layer in self._call.leaders
        waited = self._wait_for_layer(layer)
        elapsed = time.monotonic() - start
        stats.calls += 1
        stats.stream_waits += waited
        stats.total_s += elapsed
        if not ready:
            stats.blocked += 1
            stats.blocked_s += elapsed

    def _wait_for_layer(self, layer: int) -> bool:
        """:meth:`wait_for_layer`; returns whether the stream was made to
        wait."""
        leader = self._leader_of(layer)
        if leader is None or leader == self._last_waited:
            return False
        self._pool.wait(leader, torch_dev.current_stream())
        self._last_waited = leader
        return True

    def _log_stats(self, stats: _WaitStats) -> None:
        """Log the layer waits and frame arrivals of this retrieve, once."""
        stats.logged = True
        arrivals = list(self._call.arrivals or ())
        logger.info(
            "layerwise-wait: %d call(s), %d stream wait(s), %d blocked for "
            "%.2f ms, total %.2f ms; %d frame(s) for %d KV cache(s), "
            "per frame (caches@+ms): %s",
            stats.calls,
            stats.stream_waits,
            stats.blocked,
            stats.blocked_s * 1e3,
            stats.total_s * 1e3,
            len(arrivals),
            sum(n for n, _ in arrivals),
            ", ".join(f"{n}@{t * 1e3:.1f}" for n, t in arrivals) or "none",
        )

    def query(self) -> bool:
        if not self._raw.query():
            return False
        self._resolve()
        if self._done_event is None:
            return True
        return self._event_backend.query_event(self._done_event)

    def wait(self, timeout: Optional[float] = None) -> bool:
        if not self._raw.wait(timeout):
            return False
        self._resolve()
        if self._done_event is not None:
            self._event_backend.synchronize_event(self._done_event, self._device)
        if self._stats is not None and not self._stats.logged:
            self._log_stats(self._stats)
        return True

    def result(self, timeout: Optional[float] = None) -> bool:
        if not self.wait(timeout):
            raise LMCacheTimeoutError("Layer-wise retrieve result not available")
        return self._resolve().succeeded

    def wait_on_stream(self, stream: Any, timeout: Optional[float] = None) -> bool:
        if not self._raw.wait(timeout):
            raise LMCacheTimeoutError("Layer-wise retrieve result not available")
        response = self._resolve()
        if self._done_event is not None:
            self._event_backend.wait_event(self._done_event, stream)
        return response.succeeded

    def set_result(self, result: bool) -> None:
        raise NotImplementedError("LayerwiseRetrieveFuture is read-only")

    def set_exception(self, exception: BaseException) -> None:
        raise NotImplementedError("LayerwiseRetrieveFuture is read-only")


class LayerwiseTransferContext(LMCacheDrivenTransferContext):
    """LMCache-driven transfer context that retrieves layer by layer.

    Registration additionally reports the model depth of each KV cache,
    imports the server's event pool and subscribes to its frames. Retrieves go
    to ``layerwise.retrieve`` and return :class:`LayerwiseRetrieveFuture`;
    everything else is LMCache's.

    Args:
        instance_id: Worker process instance identifier.
        req_client: Transport client for this worker.
        model_depths: Model depth of each KV cache, in registration order, or
            None to order layers by registration.
        frames_host: Host of the server's frame channel; the host the
            request server is reached at.
    """

    def __init__(
        self,
        instance_id: int,
        req_client: RequestClient,
        model_depths: Sequence[int] | None = None,
        frames_host: str = "127.0.0.1",
    ) -> None:
        super().__init__(instance_id, req_client)
        self._model_depths = None if model_depths is None else list(model_depths)
        self._frames_host = frames_host
        self._pool: EventPool | None = None
        self._call_ids = itertools.count()
        self._calls: weakref.WeakValueDictionary[int, _Call] = (
            weakref.WeakValueDictionary()
        )
        self._calls_lock = threading.Lock()
        # Set to stop the current frame reader.
        self._frames_stop: threading.Event | None = None

    def _server_module_call(self, method: str, message: Any, timeout: float) -> bytes:
        request = ServerModuleCallRequest(method, protocol.encode(message))
        response = self._req_client.server_module_call(request).result(timeout)
        if not response.success:
            raise RuntimeError(f"{method} failed: {response.error}")
        return response.payload

    def register(
        self,
        kv_caches: dict[str, torch.Tensor],
        model_name: str,
        world_size: int,
        _blocks_in_chunk: int,
        mq_timeout: float,
        layout_hints: LayoutHints | None = None,
        engine_group_infos: Sequence[EngineGroupInfo] = (),
        engine_type: EngineType = EngineType.VLLM,
    ) -> None:
        """Register as LMCache does, then set up the layer-wise state.

        Raises:
            RuntimeError: If the server does not serve the layer-wise
                module, or the context is closed.
            ValueError: If the model depths do not cover ``kv_caches``.
        """
        if self._model_depths is not None:
            if len(self._model_depths) != len(kv_caches):
                raise ValueError(
                    f"Got {len(self._model_depths)} model depths for "
                    f"{len(kv_caches)} KV caches."
                )
            self._server_module_call(
                protocol.SET_MODEL_DEPTHS,
                protocol.SetModelDepthsRequest(self._instance_id, self._model_depths),
                mq_timeout,
            )
        super().register(
            kv_caches,
            model_name,
            world_size,
            _blocks_in_chunk,
            mq_timeout,
            layout_hints,
            engine_group_infos,
            engine_type,
        )
        if self._is_closed():
            return
        assert self._event_backend is not None and self._device is not None
        events = protocol.decode(
            self._server_module_call(
                protocol.REGISTER_EVENTS,
                protocol.InstanceRequest(self._instance_id),
                mq_timeout,
            ),
            protocol.RegisterEventsResponse,
        )
        self._pool = EventPool.import_pool(
            self._event_backend, self._device, events.handles
        )
        self._open_frames(events.frames_port, mq_timeout)

    def _open_frames(self, port: int, timeout: float) -> None:
        """Subscribe to the instance's frames, then probe the channel until
        the subscription is live, for up to ``timeout`` seconds."""
        self._close_frames()
        if not port:
            return
        address = f"tcp://{self._frames_host}:{port}"
        socket = zmq.Context.instance().socket(zmq.SUB)
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.SUBSCRIBE, protocol.frames_topic(self._instance_id))
        socket.connect(address)
        stop = threading.Event()
        ready = threading.Event()
        self._frames_stop = stop
        threading.Thread(
            target=self._read_frames,
            args=(socket, stop, ready),
            name=f"lmcache-layerwise-frames-{self._instance_id}",
            daemon=True,
        ).start()
        deadline = time.monotonic() + timeout
        probe = protocol.InstanceRequest(self._instance_id)
        while True:
            self._server_module_call(protocol.PROBE_FRAMES, probe, timeout)
            if ready.wait(_PROBE_INTERVAL_S):
                return
            if time.monotonic() >= deadline:
                logger.warning(
                    "The layer-wise frame channel at %s did not deliver within "
                    "%.1f s; layer waits fall back to each retrieve's answer, "
                    "without overlapping the transfer.",
                    address,
                    timeout,
                )
                return

    def _close_frames(self) -> None:
        stop, self._frames_stop = self._frames_stop, None
        if stop is not None:
            stop.set()

    def _read_frames(
        self, socket: zmq.Socket, stop: threading.Event, ready: threading.Event
    ) -> None:
        """Hand each frame to its retrieve until ``stop``, then close
        ``socket``; set ``ready`` on a probe frame. Frames of retrieves no
        longer awaited are dropped; lost frames only slow waits down, since
        every answer also carries each layer's event."""
        try:
            while not stop.is_set():
                if not socket.poll(_FRAME_POLL_MS):
                    continue
                _, payload = socket.recv_multipart(zmq.NOBLOCK)
                frame = protocol.decode(payload, protocol.Frame)
                if frame.call_id == protocol.PROBE_CALL_ID:
                    ready.set()
                    continue
                with self._calls_lock:
                    call = self._calls.get(frame.call_id)
                if call is not None:
                    call.add(frame)
        except Exception:
            if not stop.is_set():
                logger.exception("The layer-wise frame reader failed")
        finally:
            socket.close()

    def submit_retrieve(
        self,
        _request_id: str,
        key: Any,
        _kv_caches: dict[str, torch.Tensor],
        block_ids: list[list[int]],
        event: IPCEvent | None,
        _blocks_in_chunk: int,
        skip_first_n_tokens: int = 0,
    ) -> LayerwiseRetrieveFuture:
        """Submit a layer-wise retrieve ordered by ``event``.

        Raises:
            RuntimeError: If the context is not registered or ``event`` is
                None.
        """
        if self._device is None or self._event_backend is None or self._pool is None:
            raise RuntimeError(
                "Layer-wise transfer context is not registered. "
                "Call register() before submit_retrieve()."
            )
        if event is None:
            raise RuntimeError("LMCache-driven transfer requires an IPC event.")
        event_ipc_handle = self._event_backend.export_event(event, self._device)
        call = _Call()
        with self._calls_lock:
            call_id = next(self._call_ids)
            self._calls[call_id] = call
        request = protocol.RetrieveRequest(
            call_id,
            key,
            self._instance_id,
            block_ids,
            event_ipc_handle,
            skip_first_n_tokens,
        )
        raw = self._req_client.server_module_affinity_call(
            ServerModuleCallRequest(protocol.RETRIEVE, protocol.encode(request))
        )
        return LayerwiseRetrieveFuture(
            call,
            raw,
            device=self._device,
            event_backend=self._event_backend,
            pool=self._pool,
        )

    def close(self) -> None:
        """Stop the frame reader, then close as LMCache does."""
        self._close_frames()
        super().close()
