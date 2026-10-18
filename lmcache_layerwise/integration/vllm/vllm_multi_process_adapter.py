# SPDX-License-Identifier: Apache-2.0
"""Worker adapter for the layer-wise KV retrieve.

LMCache's worker adapter with a :class:`LayerwiseTransferContext`, a
per-layer wait for vLLM, and no retrieve completion reports (the connector
loads synchronously).
"""

# Standard
from collections.abc import Iterable
from typing import Any
import re

# First Party
from lmcache.integration.vllm.vllm_multi_process_adapter import (
    LMCacheMPWorkerAdapter,
    LoadStoreOp,
    _IpcEvent,
)
from lmcache.logging import init_logger
from lmcache.v1.mp_observability.errors import LMCacheTimeoutError
from lmcache.v1.multiprocess.transfer_context import TransferContext

# First Party
from lmcache_layerwise import debug
from lmcache_layerwise.v1.multiprocess.transfer_context.worker_transfer import (
    LayerwiseRetrieveFuture,
    LayerwiseTransferContext,
)

logger = init_logger(__name__)

# The model depth in vLLM layer names such as "model.layers.5.self_attn.attn".
_LAYER_RE = re.compile(r"model\.layers\.(\d+)")


def model_depths(layer_names: Iterable[str]) -> list[int]:
    """Model depth of each KV cache, from its vLLM layer name.

    Args:
        layer_names: Names of the KV caches, in registration order.

    Raises:
        RuntimeError: If a name has no model depth. vLLM waits per layer
            name, so such a cache could not be ordered against the compute
            that reads it.
    """
    names = list(layer_names)
    depths: list[int] = []
    unmatched: list[str] = []
    for name in names:
        match = _LAYER_RE.search(name)
        if match is None:
            unmatched.append(name)
        else:
            depths.append(int(match.group(1)))
    if unmatched:
        raise RuntimeError(
            "Layer-wise KV loading is not supported for this model: "
            f"{len(unmatched)} of {len(names)} KV cache name(s) do not match "
            f"{_LAYER_RE.pattern!r}, e.g. {unmatched[:3]}. Use LMCache's "
            "LMCacheMPConnector instead."
        )
    return depths


def _log_kv_cache_layout(kv_caches: dict, depths: list[int]) -> None:
    """Log each KV cache's registration index and model depth, and which
    caches share one allocation (``LMCACHE_LAYERWISE_DEBUG``).

    The server orders each object, and so each signalled batch, by model
    depth; a depth whose caches are spread over the registration order costs
    more copy launches, not more waiting.
    """
    logger.info(
        "Layer-wise KV caches in registration order (name@depth), %d: %s",
        len(kv_caches),
        [f"{name}@{depth}" for name, depth in zip(kv_caches, depths, strict=True)],
    )
    pools: dict[int, list[str]] = {}
    for name, cache in kv_caches.items():
        try:
            data_ptr = cache.untyped_storage().data_ptr()
        except AttributeError:
            data_ptr = -1
        pools.setdefault(data_ptr, []).append(name)
    logger.info(
        "Layer-wise KV caches occupy %d allocation(s): %s",
        len(pools),
        [f"{len(names)} cache(s), e.g. {names[:2]}" for names in pools.values()],
    )


def server_host(server_url: str) -> str:
    """The host a request server URL reaches, which also serves the frame
    channel; loopback for URLs without one, such as ``ipc://``."""
    scheme, sep, target = server_url.strip().partition("://")
    if not sep:
        scheme, target = "tcp", scheme
    if scheme.lower() not in ("tcp", "grpc"):
        return "127.0.0.1"
    return target.rsplit(":", 1)[0] or "127.0.0.1"


class LMCacheLayerwiseMPWorkerAdapter(LMCacheMPWorkerAdapter):
    """Worker adapter that loads KV cache one layer batch at a time."""

    def __init__(self, server_url: str, *args: Any, **kwargs: Any) -> None:
        self._frames_host = server_host(server_url)
        super().__init__(server_url, *args, **kwargs)

    # Set once vLLM calls wait_for_layer_load() for a known layer. Until then
    # each retrieve is waited for in full before it is handed back, so KV is
    # never read in flight by a model whose attention does not call it.
    _gate_verified: bool = False
    _drain_warned: bool = False

    # Model depth -> registration indices of its KV caches. A signalled
    # batch covers whole depths of one object group only; with separate
    # object groups a depth spans several, each signalled on its own.
    _depth_wait_idx: dict[int, list[int]] | None = None

    def _create_transfer_context(self, kv_caches: dict) -> TransferContext:
        """Build a layer-wise transfer context.

        A previous context (re-registration after a server restart) is
        closed, which stops its frame reader.

        Raises:
            RuntimeError: If a KV cache name has no model depth.
        """
        depths = model_depths(kv_caches)
        if debug.ENABLED:
            _log_kv_cache_layout(kv_caches, depths)
        wait_map: dict[int, list[int]] = {}
        for idx, depth in enumerate(depths):
            wait_map.setdefault(depth, []).append(idx)
        self._depth_wait_idx = wait_map
        previous = self.transfer_ctx
        if isinstance(previous, LayerwiseTransferContext):
            previous.close()
        return LayerwiseTransferContext(
            self.instance_id,
            self.req_client,
            model_depths=depths,
            frames_host=self._frames_host,
        )

    def get_finished(
        self, finished_req_ids_from_engine: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        """Report finished stores only.

        Loads are synchronous, so vLLM never parks a request waiting for
        them, and reporting one would trip the scheduler. Failed retrieves
        are still recorded for :meth:`get_block_ids_with_load_errors`.
        """
        finished_stores, _ = super().get_finished(finished_req_ids_from_engine)
        return finished_stores, None

    def get_finished_with_lazy_offload(
        self,
    ) -> tuple[set[str] | None, set[str] | None]:
        """As :meth:`get_finished`, for lazy offload."""
        finished_stores, _ = super().get_finished_with_lazy_offload()
        return finished_stores, None

    def submit_retrieve_request(
        self,
        request_id: str,
        op: LoadStoreOp,
        event: _IpcEvent | None,
        cache_salt: str = "",
        request_configs: dict[str, Any] | None = None,
    ) -> None:
        """Submit a retrieve; wait for it in full until vLLM is seen calling
        :meth:`wait_for_layer_load`.

        Ordering against attention comes only from those calls, so a model
        that never makes them would otherwise read KV still in flight.
        """
        super().submit_retrieve_request(
            request_id, op, event, cache_salt, request_configs
        )
        if self._gate_verified:
            return
        entry = self.retrieve_futures.get(request_id)
        if entry is None or not isinstance(entry[0], LayerwiseRetrieveFuture):
            return
        if not self._drain_warned:
            self._drain_warned = True
            logger.warning(
                "Waiting for layer-wise retrieves in full until vLLM calls "
                "wait_for_layer_load(). If this model's attention never "
                "calls it, loads stay at per-chunk speed."
            )
        entry[0].wait()

    def wait_for_layer_load(self, layer_name: str) -> None:
        """Make the current stream wait for ``layer_name``'s KV in every
        pending retrieve.

        A retrieve that times out is retired and its blocks are reported for
        recomputation.

        Args:
            layer_name: vLLM layer name, e.g. ``"model.layers.5.self_attn"``.
        """
        wait_map = self._depth_wait_idx
        if wait_map is None or not isinstance(
            self.transfer_ctx, LayerwiseTransferContext
        ):
            return
        match = _LAYER_RE.search(layer_name)
        if match is None:
            return
        wait_indices = wait_map.get(int(match.group(1)))
        if wait_indices is None:
            # A layer without KV cache, such as a draft layer.
            return
        if not self._gate_verified:
            self._gate_verified = True
            logger.info("Layer-wise loads overlap with compute from %r on.", layer_name)

        for request_id in list(self.retrieve_futures):
            entry = self.retrieve_futures.get(request_id)
            if entry is None:
                continue
            future, block_ids = entry
            if not isinstance(future, LayerwiseRetrieveFuture):
                continue
            try:
                for wait_idx in wait_indices:
                    future.wait_for_layer(wait_idx)
            except LMCacheTimeoutError:
                logger.error(
                    "Layer-wise retrieve for request_id=%s timed out at %r; "
                    "recomputing %d block(s)",
                    request_id,
                    layer_name,
                    len(block_ids),
                )
                self.error_block_ids.update(block_ids)
                self.retrieve_futures.pop(request_id, None)
                self.retrieve_events.pop(request_id, None)
