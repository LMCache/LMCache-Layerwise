# SPDX-License-Identifier: Apache-2.0
"""The layer-wise worker adapter's own logic, on an adapter built without
vLLM or a server."""

# Third Party
import pytest

# First Party
from lmcache.integration.vllm.vllm_multi_process_adapter import (
    LMCacheMPWorkerAdapter,
)
from lmcache.v1.mp_observability.errors import LMCacheTimeoutError
from lmcache_layerwise import debug
from lmcache_layerwise.integration.vllm import vllm_multi_process_adapter as mod
from lmcache_layerwise.integration.vllm.vllm_multi_process_adapter import (
    LMCacheLayerwiseMPWorkerAdapter,
    model_depths,
)


class _Future(mod.LayerwiseRetrieveFuture):
    def __init__(self, timeout_at=None):
        self.waited = []
        self.drained = False
        self._timeout_at = timeout_at

    def wait_for_layer(self, layer):
        if layer == self._timeout_at:
            raise LMCacheTimeoutError("stalled")
        self.waited.append(layer)

    def wait(self, timeout=None):
        self.drained = True
        return True


class _Ctx(mod.LayerwiseTransferContext):
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


def _adapter(names=("model.layers.0.a", "model.layers.0.b", "model.layers.1.a")):
    adapter = LMCacheLayerwiseMPWorkerAdapter.__new__(LMCacheLayerwiseMPWorkerAdapter)
    adapter.transfer_ctx = None
    adapter.instance_id = 9
    adapter.req_client = object()
    adapter._frames_host = "server"
    adapter.retrieve_futures = {}
    adapter.retrieve_events = {}
    adapter.error_block_ids = set()
    adapter.transfer_ctx = adapter._create_transfer_context(dict.fromkeys(names))
    return adapter


def test_model_depths_come_from_the_layer_names():
    names = ["model.layers.3.self_attn.attn", "language_model.model.layers.10.x"]
    assert model_depths(names) == [3, 10]


def test_a_name_without_a_depth_is_rejected():
    with pytest.raises(RuntimeError, match="1 of 2"):
        model_depths(["model.layers.0.attn", "lm_head"])


@pytest.mark.parametrize(
    "url, host",
    [
        ("localhost:5555", "localhost"),
        ("tcp://10.0.0.2:5555", "10.0.0.2"),
        ("grpc://lmcache-server:5555", "lmcache-server"),
        ("ipc:///tmp/lmcache.sock", "127.0.0.1"),
    ],
)
def test_the_frame_channel_host_is_the_request_servers(url, host):
    assert mod.server_host(url) == host


def test_the_context_reports_the_depths_and_replaces_the_previous_one():
    adapter = _adapter()
    ctx = adapter.transfer_ctx
    assert isinstance(ctx, mod.LayerwiseTransferContext)
    assert ctx._model_depths == [0, 0, 1]
    assert ctx._instance_id == 9
    assert ctx._frames_host == "server"

    previous = _Ctx()
    adapter.transfer_ctx = previous
    adapter._create_transfer_context({"model.layers.0.a": None})
    assert previous.closed


def test_wait_for_layer_load_waits_every_pending_retrieve_on_its_depth():
    adapter = _adapter()
    a, b = _Future(), _Future()
    adapter.retrieve_futures = {"a": (a, [1]), "b": (b, [2]), "c": (object(), [3])}

    adapter.wait_for_layer_load("model.layers.0.self_attn")
    adapter.wait_for_layer_load("model.layers.1.self_attn")
    adapter.wait_for_layer_load("model.layers.7.self_attn")  # no KV cache
    adapter.wait_for_layer_load("lm_head")

    assert a.waited == b.waited == [0, 1, 2]
    assert adapter._gate_verified


def test_wait_for_layer_load_waits_every_kv_cache_of_the_depth():
    # Separate object groups signal a depth's caches in separate batches, so
    # one cache landing says nothing about the others.
    adapter = _adapter(
        ("model.layers.0.a", "model.layers.1.a", "model.layers.1.b", "model.layers.0.c")
    )
    future = _Future()
    adapter.retrieve_futures = {"r": (future, [1])}

    adapter.wait_for_layer_load("model.layers.0.attn")
    adapter.wait_for_layer_load("model.layers.1.attn")

    assert future.waited == [0, 3, 1, 2]


def test_a_stalled_retrieve_is_retired_for_recomputation():
    adapter = _adapter()
    stalled, healthy = _Future(timeout_at=2), _Future()
    adapter.retrieve_futures = {"s": (stalled, [4, 5]), "h": (healthy, [6])}
    adapter.retrieve_events = {"s": object()}

    adapter.wait_for_layer_load("model.layers.1.attn")

    assert adapter.error_block_ids == {4, 5}
    assert list(adapter.retrieve_futures) == ["h"]
    assert adapter.retrieve_events == {}
    assert healthy.waited == [2]


def test_retrieves_are_drained_until_vllm_waits_per_layer(monkeypatch):
    adapter = _adapter()
    futures = iter([_Future(), _Future()])

    def submit(self, request_id, op, event, cache_salt="", request_configs=None):
        self.retrieve_futures[request_id] = (next(futures), [])

    monkeypatch.setattr(LMCacheMPWorkerAdapter, "submit_retrieve_request", submit)

    adapter.submit_retrieve_request("a", None, None)
    assert adapter.retrieve_futures["a"][0].drained

    adapter.wait_for_layer_load("model.layers.0.attn")
    adapter.submit_retrieve_request("b", None, None)
    assert not adapter.retrieve_futures["b"][0].drained


def test_finished_retrieves_are_not_reported(monkeypatch):
    adapter = _adapter()
    monkeypatch.setattr(
        LMCacheMPWorkerAdapter, "get_finished", lambda self, ids: ({"s"}, {"r"})
    )
    monkeypatch.setattr(
        LMCacheMPWorkerAdapter,
        "get_finished_with_lazy_offload",
        lambda self: ({"s"}, {"r"}),
    )
    assert adapter.get_finished(set()) == ({"s"}, None)
    assert adapter.get_finished_with_lazy_offload() == ({"s"}, None)


def test_debug_logs_the_kv_cache_layout(monkeypatch):
    monkeypatch.setattr(debug, "ENABLED", True)
    logs = []
    monkeypatch.setattr(mod.logger, "info", lambda msg, *args: logs.append(msg % args))
    _adapter()
    assert "model.layers.0.b@0" in logs[0]
    assert "model.layers.1.a@1" in logs[0]
    assert "1 allocation(s)" in logs[1]
