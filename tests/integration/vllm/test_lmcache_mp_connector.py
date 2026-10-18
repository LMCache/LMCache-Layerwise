# SPDX-License-Identifier: Apache-2.0
"""The layer-wise connector's overrides."""

# Standard
from unittest.mock import MagicMock

# Third Party
import pytest

pytest.importorskip("vllm", reason="Requires vLLM")

# First Party
from lmcache.integration.vllm.lmcache_mp_connector import (  # noqa: E402
    LMCacheMPConnector,
)
from lmcache_layerwise.integration.vllm.lmcache_mp_connector import (  # noqa: E402
    LMCacheLayerwiseMPConnector,
)
from lmcache_layerwise.integration.vllm.vllm_multi_process_adapter import (  # noqa: E402
    LMCacheLayerwiseMPWorkerAdapter,
)


def test_the_connector_uses_the_layerwise_adapter(monkeypatch):
    # First Party
    from lmcache_layerwise.integration.vllm import (
        lmcache_mp_connector as connector_mod,
    )

    built = MagicMock()
    monkeypatch.setattr(connector_mod, "LMCacheLayerwiseMPWorkerAdapter", built)
    connector = LMCacheLayerwiseMPConnector.__new__(LMCacheLayerwiseMPConnector)
    adapter = connector.create_lmcache_mp_worker_adapter(server_url="u", x=1)
    built.assert_called_once_with(server_url="u", x=1)
    assert adapter is built.return_value


@pytest.mark.parametrize(
    "base,expected", [((None, True), (None, True)), ((32, True), (32, False))]
)
def test_hits_load_synchronously(monkeypatch, base, expected):
    monkeypatch.setattr(
        LMCacheMPConnector, "get_num_new_matched_tokens", lambda self, r, n: base
    )
    connector = LMCacheLayerwiseMPConnector.__new__(LMCacheLayerwiseMPConnector)
    assert connector.get_num_new_matched_tokens(None, 0) == expected


def test_layer_waits_reach_the_adapter():
    adapter = LMCacheLayerwiseMPWorkerAdapter.__new__(LMCacheLayerwiseMPWorkerAdapter)
    seen = []
    adapter.wait_for_layer_load = seen.append
    connector = LMCacheLayerwiseMPConnector.__new__(LMCacheLayerwiseMPConnector)
    connector.worker_adapter = adapter
    connector.wait_for_layer_load("model.layers.0.attn")
    assert seen == ["model.layers.0.attn"]
