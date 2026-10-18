# SPDX-License-Identifier: Apache-2.0
"""The layer-wise transfer module and the server module serving its RPCs.

Load with::

    lmcache server ... --server-module '{
        "module_path": "lmcache_layerwise.v1.multiprocess.modules.layerwise_transfer",
        "config": {"batch_layers": 4}}'

``batch_layers`` (default 4) is how many model layers each signalled batch
of a retrieve covers. ``frames_port`` (default 0, a random free port) is the
TCP port of the frame channel, bound on the server's ``--host``; fix it when
only known ports are reachable, as with published container ports.
``enqueue_per_batch`` (default false) enqueues and signals each layer batch
as soon as it is built, instead of an object group's batches in one native
call. It pays off when that one call would enqueue many copies, as with a
small ``batch_layers``, and costs a little per extra call otherwise.
"""

# Standard
from typing import Any, Callable, Mapping, Sequence
import threading

# Third Party
import zmq

# First Party
from lmcache import torch_dev
from lmcache.logging import init_logger
from lmcache.utils import EngineType
from lmcache.v1.gpu_connector.utils import LayoutHints
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey, KVCache
from lmcache.v1.multiprocess.engine_context import MPCacheServerContext
from lmcache.v1.multiprocess.engine_module import (
    DiscoverableModule,
    ModuleBuildContext,
)
from lmcache.v1.multiprocess.ext_server_module import server_module_handler
from lmcache.v1.multiprocess.group_view import EngineGroupInfo
from lmcache.v1.multiprocess.modules.lmcache_driven_transfer import (
    LMCacheDrivenTransferModule,
)
from lmcache.v1.multiprocess.request_handler import HandlerType, request_handler
from lmcache.v1.platform.base.cache_context import BaseCacheContext
from lmcache_layerwise.v1.multiprocess import protocol
from lmcache_layerwise.v1.platform.base.event_pool import EventPool
from lmcache_layerwise.v1.platform.devices.cuda.cache_context import (
    GPULayerwiseCacheContext,
)

logger = init_logger(__name__)


def _batch_layers(config: Mapping[str, Any]) -> int:
    value = config.get("batch_layers", 4)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"batch_layers must be an integer >= 1, got {value!r}")
    return value


def _enqueue_per_batch(config: Mapping[str, Any]) -> bool:
    value = config.get("enqueue_per_batch", False)
    if not isinstance(value, bool):
        raise ValueError(f"enqueue_per_batch must be a boolean, got {value!r}")
    return value


def _frames_port(config: Mapping[str, Any]) -> int:
    value = config.get("frames_port", 0)
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 65536:
        raise ValueError(f"frames_port must be an integer in [0, 65535], got {value!r}")
    return value


class _RetrieveSignal:
    """The :class:`LayerSignal` of one retrieve.

    A batch's leader is its first KV cache, and the leader's pool event marks
    the batch. Announcing a batch notes the leader of each of its caches and
    publishes a frame.
    """

    def __init__(
        self,
        pool: EventPool,
        stream: object,
        num_layers: int,
        publish: Callable[[int, list[int]], None],
    ) -> None:
        self._pool = pool
        self._stream = stream
        self._publish = publish
        self.leaders = [-1] * num_layers
        """Leader of each KV cache; -1 until a batch covers it."""

    def event(self, layers: list[int]) -> int:
        return self._pool.native_event(layers[0])

    def __call__(self, layers: list[int], *, recorded: bool = False) -> None:
        leader = layers[0]
        if not recorded:
            self._pool.record(leader, self._stream)
        for layer in layers:
            self.leaders[layer] = leader
        self._publish(leader, layers)


class _FramePublisher:
    """The frame channel: a ZMQ PUB socket, one topic per instance.

    A retrieve sends its frames from its own thread, non-blocking, under a
    lock. A sender thread would cost a GIL handoff per frame: it waits up to
    the interpreter's switch interval to run whenever a retrieve is running
    Python. Frames never wait for the worker; PUB drops what a slow or
    absent subscriber cannot take.

    Args:
        host: Interface to bind, as the server's request port.
        port: Port to bind; 0 for a random free one.

    Raises:
        RuntimeError: If the port cannot be bound.
    """

    def __init__(self, host: str, port: int = 0) -> None:
        socket = zmq.Context.instance().socket(zmq.PUB)
        socket.setsockopt(zmq.LINGER, 0)
        address = f"tcp://{host or '*'}"
        try:
            if port:
                socket.bind(f"{address}:{port}")
            else:
                port = socket.bind_to_random_port(address)
        except zmq.ZMQError as e:
            socket.close()
            raise RuntimeError(
                f"Cannot bind the layer-wise frame channel to {address}:{port}: {e}"
            ) from e
        self.port = port
        self._socket = socket
        self._lock = threading.Lock()
        self._closed = False

    def publish(self, instance_id: int, frame: protocol.Frame) -> None:
        parts = [protocol.frames_topic(instance_id), protocol.encode(frame)]
        with self._lock:
            if self._closed:
                return
            try:
                self._socket.send_multipart(parts, flags=zmq.NOBLOCK)
            except zmq.ZMQError:
                # Frames are best effort; the answer names every event.
                logger.debug("Dropped a layer-wise frame", exc_info=True)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._socket.close()


class LayerwiseTransferModule(LMCacheDrivenTransferModule):
    """LMCache-driven transfer module with a layer-wise retrieve.

    Every store writes layer-major objects, so plain ``retrieve`` cannot read
    them and is refused; :meth:`retrieve_layerwise` runs the inherited
    retrieve with a per-batch signal instead. Everything else is LMCache's.
    It keeps the inherited ``module_name``, so the server builds it in place
    of :class:`LMCacheDrivenTransferModule`.

    Args:
        ctx: The shared server context.
        layer_batch: Model depths per signalled layer batch.
        enqueue_per_batch: Enqueue and signal each layer batch as soon as it
            is built.
        frames_host: Interface the frame channel binds; the server's host.
        frames_port: Port the frame channel binds; 0 for a random free one.

    Raises:
        ValueError: For a ``layer_batch`` below 1.
        RuntimeError: If the frame channel cannot be bound.
    """

    @classmethod
    def create(cls, build_ctx: ModuleBuildContext) -> "LayerwiseTransferModule":
        """Build from the ``--server-module`` config.

        Raises:
            ValueError: For an invalid ``batch_layers``,
                ``enqueue_per_batch`` or ``frames_port``, or
                in ``engine_driven`` transfer mode.
            RuntimeError: If ``frames_port`` cannot be bound.
        """
        if build_ctx.mp_config.supported_transfer_mode == "engine_driven":
            raise ValueError(
                "lmcache_layerwise needs --supported-transfer-mode "
                "lmcache_driven or auto"
            )
        config = build_ctx.config_for(cls)
        return cls(
            build_ctx.engine_context,
            layer_batch=_batch_layers(config),
            enqueue_per_batch=_enqueue_per_batch(config),
            frames_host=build_ctx.mp_config.host,
            frames_port=_frames_port(config),
        )

    def __init__(
        self,
        ctx: MPCacheServerContext,
        layer_batch: int = 4,
        enqueue_per_batch: bool = False,
        frames_host: str = "localhost",
        frames_port: int = 0,
    ) -> None:
        if layer_batch < 1:
            raise ValueError(f"layer_batch must be at least 1, got {layer_batch}")
        super().__init__(ctx)
        self._layer_batch = layer_batch
        self._enqueue_per_batch = enqueue_per_batch
        self._tls = threading.local()
        # Per-instance layer-wise state, guarded by its own leaf lock.
        self._layerwise_lock = threading.Lock()
        self._model_depths: dict[int, list[int]] = {}
        self._event_pools: dict[int, EventPool] = {}
        self._frames: _FramePublisher | None = _FramePublisher(frames_host, frames_port)

    @property
    def layer_batch(self) -> int:
        """Model depths per signalled layer batch."""
        return self._layer_batch

    def _create_cache_context(
        self,
        kv_caches: KVCache,
        *,
        instance_id: int,
        engine_type: EngineType,
        layout_hints: LayoutHints | None,
        engine_group_infos: Sequence[EngineGroupInfo],
    ) -> BaseCacheContext:
        with self._layerwise_lock:
            model_depths = self._model_depths.get(instance_id)
        return GPULayerwiseCacheContext(
            kv_caches,
            self._ctx.chunk_size,
            layout_hints,
            engine_group_infos,
            engine_type,
            self._ctx.separate_object_groups,
            self._ctx.full_sw_kv,
            model_depths=model_depths,
            layer_signal=lambda: getattr(self._tls, "signal", None),
            layer_batch=self._layer_batch,
            enqueue_per_batch=self._enqueue_per_batch,
        )

    @request_handler()
    def unregister_kv_cache(self, instance_id: int) -> None:
        """Unregister as LMCache does, then drop the layer-wise state."""
        super().unregister_kv_cache(instance_id)
        self.drop_instance_state(instance_id)

    def drop_instance_state(self, instance_id: int) -> None:
        """Drop a gone instance's depths and events."""
        with self._layerwise_lock:
            self._model_depths.pop(instance_id, None)
            self._event_pools.pop(instance_id, None)

    def close(self) -> None:
        """Close the frame channel, then run LMCache's teardown."""
        with self._layerwise_lock:
            frames, self._frames = self._frames, None
            self._event_pools.clear()
            self._model_depths.clear()
        if frames is not None:
            frames.close()
        super().close()

    @request_handler(HandlerType.BLOCKING, requires_client_affinity=True)
    def retrieve(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
        gpu_block_ids: list[list[int]],
        event_ipc_handle: bytes,
        skip_first_n_tokens: int = 0,
    ) -> tuple[bytes, bool]:
        """Refuse a plain retrieve: stored objects are layer-major.

        Answered rather than raised, so the worker fails fast instead of
        waiting out its timeout; no device work is submitted.

        Returns:
            ``(b"", False)``.
        """
        del gpu_block_ids, event_ipc_handle, skip_first_n_tokens
        logger.error(
            "Rejecting plain retrieve for GPU instance ID %d: this server "
            "stores layer-major objects and serves layerwise.retrieve only. "
            "Run the worker with the layer-wise connector.",
            instance_id,
        )
        self._release_locks_quietly(key, instance_id)
        return b"", False

    def set_model_depths(self, instance_id: int, depths: list[int]) -> None:
        """Record each KV cache's model depth before it is registered.

        Raises:
            ValueError: For a negative depth.
        """
        if any(d < 0 for d in depths):
            raise ValueError("model depths must be non-negative")
        with self._layerwise_lock:
            self._model_depths[instance_id] = list(depths)

    def register_events(self, instance_id: int) -> protocol.RegisterEventsResponse:
        """Create, once, the event pool of a registered instance.

        Raises:
            ValueError: If the instance is not registered.
        """
        with self._layerwise_lock:
            pool = self._event_pools.get(instance_id)
        if pool is None:
            entry = self.get_and_touch_context_entry(instance_id)
            if entry is None or entry.event_backend is None:
                raise ValueError(f"GPU instance ID {instance_id} is not registered")
            cache_context = entry.cache_context
            new_pool = EventPool(
                entry.event_backend, cache_context.device, cache_context.num_layers
            )
            with self._layerwise_lock:
                pool = self._event_pools.setdefault(instance_id, new_pool)
        frames = self._frames
        return protocol.RegisterEventsResponse(
            self._layer_batch, pool.handles, frames.port if frames else 0
        )

    def probe_frames(self, instance_id: int) -> None:
        """Publish a probe frame, which reaches the instance once its
        subscription is live."""
        self._publish_frame(instance_id, protocol.Frame(protocol.PROBE_CALL_ID, -1, []))

    def _publish_frame(self, instance_id: int, frame: protocol.Frame) -> None:
        frames = self._frames
        if frames is not None:
            frames.publish(instance_id, frame)

    def _release_locks_quietly(self, key: IPCCacheServerKey, instance_id: int) -> None:
        try:
            self._release_failed_retrieve_locks(key, instance_id)
        except Exception:
            # The answer must still be sent.
            logger.exception(
                "Failed to release retrieve locks for GPU instance ID %d",
                instance_id,
            )

    def retrieve_layerwise(
        self, request: protocol.RetrieveRequest
    ) -> protocol.RetrieveResponse:
        """Run LMCache's retrieve, signalling each layer batch as it lands.

        Each batch's leader pool event is recorded on the transfer stream
        right after the batch, and a frame is published once the batch's
        object group is enqueued. Layers no batch covered
        (aux groups, or a failure midway) are signalled once the retrieve
        returns, so a worker never waits on an event this call left stale.
        """
        instance_id = request.instance_id
        entry = self.get_and_touch_context_entry(instance_id)
        with self._layerwise_lock:
            pool = self._event_pools.get(instance_id)
        if entry is None or pool is None:
            logger.error(
                "Rejecting layerwise.retrieve for GPU instance ID %d: not "
                "registered, or layerwise.register_events was not called.",
                instance_id,
            )
            self._release_locks_quietly(request.key, instance_id)
            return protocol.RetrieveResponse(b"", False, [])

        cache_context = entry.cache_context
        signal = _RetrieveSignal(
            pool,
            cache_context.stream,
            cache_context.num_layers,
            lambda leader, layers: self._publish_frame(
                instance_id, protocol.Frame(request.call_id, leader, layers)
            ),
        )
        self._tls.signal = signal
        try:
            handle, succeeded = super().retrieve(
                request.key,
                instance_id,
                request.gpu_block_ids,
                request.event_ipc_handle,
                request.skip_first_n_tokens,
            )
        finally:
            self._tls.signal = None

        leaders = signal.leaders
        uncovered = [i for i, leader in enumerate(leaders) if leader < 0]
        if uncovered:
            with torch_dev.device(cache_context.device):
                signal(uncovered)
        return protocol.RetrieveResponse(handle, succeeded, leaders)


class LayerwiseServerModule(DiscoverableModule):
    """Serves the ``layerwise.*`` methods on a :class:`LayerwiseTransferModule`.

    Args:
        transfer: The server's transfer module.
    """

    module_name = "layerwise_server"
    module_dependencies = (LayerwiseTransferModule,)

    @classmethod
    def create(cls, build_ctx: ModuleBuildContext) -> "LayerwiseServerModule":
        """Build on the server's layer-wise transfer module."""
        return cls(build_ctx.require(LayerwiseTransferModule))

    def __init__(self, transfer: LayerwiseTransferModule) -> None:
        self._transfer = transfer

    @property
    def context(self) -> MPCacheServerContext:
        """Return the shared engine context. Exposed for testing only."""
        return self._transfer.context

    def report_status(self) -> dict[str, Any]:
        """Return the layer batch size."""
        return {"layer_batch": self._transfer.layer_batch}

    def close(self) -> None:
        """Nothing to release; the transfer module owns the state."""

    @server_module_handler(protocol.SET_MODEL_DEPTHS)
    def set_model_depths(self, payload: bytes) -> bytes:
        request = protocol.decode(payload, protocol.SetModelDepthsRequest)
        self._transfer.set_model_depths(request.instance_id, request.depths)
        return b""

    @server_module_handler(protocol.REGISTER_EVENTS)
    def register_events(self, payload: bytes) -> bytes:
        request = protocol.decode(payload, protocol.InstanceRequest)
        return protocol.encode(self._transfer.register_events(request.instance_id))

    @server_module_handler(protocol.PROBE_FRAMES)
    def probe_frames(self, payload: bytes) -> bytes:
        request = protocol.decode(payload, protocol.InstanceRequest)
        self._transfer.probe_frames(request.instance_id)
        return b""

    @server_module_handler(protocol.RETRIEVE, requires_client_affinity=True)
    def retrieve(self, payload: bytes) -> bytes:
        request = protocol.decode(payload, protocol.RetrieveRequest)
        return protocol.encode(self._transfer.retrieve_layerwise(request))
