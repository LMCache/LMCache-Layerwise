# SPDX-License-Identifier: Apache-2.0
"""vLLM connector for the layer-wise KV retrieve.

Select it with::

    --kv-transfer-config '{
      "kv_connector": "LMCacheLayerwiseMPConnector",
      "kv_connector_module_path":
        "lmcache_layerwise.integration.vllm.lmcache_mp_connector",
      "kv_role": "kv_both"
    }'

Everything else is LMCache's ``LMCacheMPConnector``.
"""

# Standard
from typing import TYPE_CHECKING, Any

# First Party
from lmcache.integration.vllm.lmcache_mp_connector import LMCacheMPConnector

# First Party
from lmcache_layerwise.integration.vllm.vllm_multi_process_adapter import (
    LMCacheLayerwiseMPWorkerAdapter,
)

if TYPE_CHECKING:
    # Third Party
    from vllm.v1.request import Request


class LMCacheLayerwiseMPConnector(LMCacheMPConnector):
    """LMCache MP connector that loads KV cache layer by layer."""

    def create_lmcache_mp_worker_adapter(
        self, **adapter_kwargs: Any
    ) -> LMCacheLayerwiseMPWorkerAdapter:
        """Build the layer-wise worker adapter."""
        return LMCacheLayerwiseMPWorkerAdapter(**adapter_kwargs)

    def get_num_new_matched_tokens(
        self, request: "Request", num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        """Report hits as synchronous loads.

        vLLM runs a request with an asynchronous load only after the whole
        load finished, leaving nothing for per-layer waits to overlap. A
        synchronous load runs in the same step, gated per layer by
        :meth:`wait_for_layer_load`.
        """
        num_tokens, load_async = super().get_num_new_matched_tokens(
            request, num_computed_tokens
        )
        if num_tokens is None:
            return None, load_async
        return num_tokens, False

    def wait_for_layer_load(self, layer_name: str) -> None:
        """Make the current stream wait for ``layer_name``'s KV."""
        adapter = self.worker_adapter
        assert isinstance(adapter, LMCacheLayerwiseMPWorkerAdapter)
        adapter.wait_for_layer_load(layer_name)
