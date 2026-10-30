# SPDX-License-Identifier: Apache-2.0
"""The worker's layer-wise transfer context and futures, against a fake
client on CPU."""

# Standard
import threading

# Third Party
import pytest
import torch
import zmq

# First Party
from lmcache.v1.mp_observability.errors import LMCacheTimeoutError
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.futures import MessagingFuture
from lmcache.v1.multiprocess.protocols.server_module import (
    ServerModuleCallResponse,
)
from lmcache.v1.multiprocess.token_codec import pack_token_ids
from lmcache.v1.multiprocess.transfer_context import worker_transfer as core
from lmcache_layerwise import debug
from lmcache_layerwise.v1.multiprocess import protocol
from lmcache_layerwise.v1.multiprocess.transfer_context import worker_transfer
from lmcache_layerwise.v1.multiprocess.transfer_context.worker_transfer import (
    LayerwiseRetrieveFuture,
    LayerwiseTransferContext,
    _Call,
)
from lmcache_layerwise.v1.platform.base.event_pool import EventPool

_INSTANCE = 5
_STREAM = "stream"


class _Backend:
    """Events are strings; records what the worker does with them."""

    def __init__(self):
        self.waits = []
        self.synced = []
        self.done = True

    def check_event_support(self, device):
        return None

    def export_event(self, event, device):
        return b"x-" + event.encode()

    def import_event(self, handle, device):
        return handle.decode()

    def wait_event(self, event, stream):
        self.waits.append(event)

    def query_event(self, event):
        return self.done

    def synchronize_event(self, event, device):
        self.synced.append(event)


def _ok(message=None):
    payload = b"" if message is None else protocol.encode(message)
    future = MessagingFuture()
    future.set_result(ServerModuleCallResponse(True, payload, ""))
    return future


class _Client:
    """Answers the layer-wise RPCs; retrieves stay pending until answered.

    Frames and probes go out on a real frame channel, an XPUB socket, so
    registration returns only once the worker's subscription is live.
    """

    sockets: list = []

    def __init__(self, pool_size=4):
        self.calls = []
        self.retrieves = []
        self.pool_size = pool_size
        self.fail = set()
        self.publisher = zmq.Context.instance().socket(zmq.XPUB)
        self.publisher.setsockopt(zmq.LINGER, 0)
        self.port = self.publisher.bind_to_random_port("tcp://127.0.0.1")
        self.answer_probes = True
        _Client.sockets.append(self.publisher)

    def server_module_call(self, request):
        self.calls.append(request.method)
        if request.method in self.fail:
            future = MessagingFuture()
            future.set_result(ServerModuleCallResponse(False, b"", "unknown"))
            return future
        if request.method == protocol.REGISTER_EVENTS:
            handles = [b"e%d" % i for i in range(self.pool_size)]
            return _ok(protocol.RegisterEventsResponse(2, handles, self.port))
        if request.method == protocol.PROBE_FRAMES and self.answer_probes:
            self.push_frame(protocol.PROBE_CALL_ID, -1, [])
        return _ok()

    def register_kv_cache(self, *args):
        self.calls.append("register_kv_cache")
        future = MessagingFuture()
        future.set_result(None)
        return future

    def server_module_affinity_call(self, request):
        self.calls.append(request.method)
        future = MessagingFuture()
        self.retrieves.append(
            (protocol.decode(request.payload, protocol.RetrieveRequest), future)
        )
        return future

    def push_frames(self, *frames):
        for frame in frames:
            self.publisher.send_multipart(
                [
                    protocol.frames_topic(_INSTANCE),
                    protocol.encode(protocol.Frame(*frame)),
                ]
            )

    def push_frame(self, call_id, leader, layers):
        self.push_frames((call_id, leader, layers))

    def answer(self, index, response):
        self.retrieves[index][1].set_result(
            ServerModuleCallResponse(True, protocol.encode(response), "")
        )


@pytest.fixture(autouse=True)
def _close_client_sockets():
    yield
    while _Client.sockets:
        _Client.sockets.pop().close()


@pytest.fixture
def backend(monkeypatch):
    backend = _Backend()
    monkeypatch.setattr(core, "get_event_ipc_backend", lambda device: backend)
    monkeypatch.setattr(core, "wrap_kv_caches", lambda kv_caches: [])
    return backend


def _caches(n=4):
    return {f"model.layers.{i}.attn": torch.zeros(1) for i in range(n)}


def _registered(client, depths=(0, 1, 2, 3)):
    ctx = LayerwiseTransferContext(_INSTANCE, client, model_depths=depths)
    ctx.register(_caches(len(depths)), "m", 1, 16, 5.0)
    return ctx


def _key():
    return IPCCacheServerKey("m", 1, 0, pack_token_ids([1, 2]), 0, 2, "req")


def _submit(ctx, skip=0):
    return ctx.submit_retrieve("r", _key(), {}, [[1, 2]], "prod", 16, skip)


def _wait_until(predicate):
    for _ in range(500):
        if predicate():
            return
        threading.Event().wait(0.01)
    raise AssertionError("condition not reached")


def test_register_reports_depths_first_then_sets_up_the_layerwise_state(backend):
    client = _Client()
    ctx = _registered(client)
    try:
        assert client.calls[:3] == [
            protocol.SET_MODEL_DEPTHS,
            "register_kv_cache",
            protocol.REGISTER_EVENTS,
        ]
        assert set(client.calls[3:]) == {protocol.PROBE_FRAMES}
    finally:
        ctx.close()


def test_register_without_depths_skips_set_model_depths(backend):
    client = _Client()
    ctx = LayerwiseTransferContext(_INSTANCE, client)
    ctx.register(_caches(), "m", 1, 16, 5.0)
    ctx.close()
    assert protocol.SET_MODEL_DEPTHS not in client.calls


def test_register_fails_when_the_server_lacks_the_module(backend):
    client = _Client()
    client.fail.add(protocol.SET_MODEL_DEPTHS)
    ctx = LayerwiseTransferContext(_INSTANCE, client, model_depths=[0, 1, 2, 3])
    with pytest.raises(RuntimeError, match="unknown"):
        ctx.register(_caches(), "m", 1, 16, 5.0)
    assert "register_kv_cache" not in client.calls


def test_register_rejects_depths_that_do_not_cover_the_caches(backend):
    ctx = LayerwiseTransferContext(_INSTANCE, _Client(), model_depths=[0, 1])
    with pytest.raises(ValueError, match="2 model depths for 4"):
        ctx.register(_caches(), "m", 1, 16, 5.0)


def test_submit_before_register_raises(backend):
    ctx = LayerwiseTransferContext(_INSTANCE, _Client())
    with pytest.raises(RuntimeError, match="not registered"):
        _submit(ctx)


def test_submit_sends_the_retrieve_on_the_affinity_rpc(backend):
    client = _Client()
    ctx = _registered(client)
    try:
        _submit(ctx, skip=16)
        _submit(ctx)
        (first, _), (second, _) = client.retrieves
        assert first.key == _key()
        assert first.instance_id == _INSTANCE
        assert first.gpu_block_ids == [[1, 2]]
        assert first.event_ipc_handle == b"x-prod"
        assert first.skip_first_n_tokens == 16
        assert first.call_id != second.call_id
    finally:
        ctx.close()


def test_frames_drive_the_layer_waits(backend):
    client = _Client()
    ctx = _registered(client)
    try:
        first = _submit(ctx)
        second = _submit(ctx)
        first_id = client.retrieves[0][0].call_id
        second_id = client.retrieves[1][0].call_id
        client.push_frames((second_id, 2, [2, 3]), (first_id, 0, [0, 1]))

        first.wait_for_layer(1)
        first.wait_for_layer(0)  # same event, waited once
        second.wait_for_layer(3)
        assert backend.waits == ["e0", "e2"]
        assert not first.query()
    finally:
        ctx.close()


def test_the_answer_covers_layers_whose_frames_were_lost(backend):
    client = _Client()
    ctx = _registered(client)
    try:
        future = _submit(ctx)
        client.answer(0, protocol.RetrieveResponse(b"done", True, [0, 0, 2, 2]))
        future.wait_for_layer(3)
        assert backend.waits == ["e2"]
        assert future.result() is True
        assert backend.synced == ["done"]
    finally:
        ctx.close()


def test_a_rejected_retrieve_waits_on_nothing(backend):
    client = _Client()
    ctx = _registered(client)
    try:
        future = _submit(ctx)
        client.answer(0, protocol.RetrieveResponse(b"", False, []))
        future.wait_for_layer(2)
        assert backend.waits == []
        assert future.query()
        assert future.result() is False
    finally:
        ctx.close()


def test_a_failed_call_is_a_failed_retrieve(backend):
    client = _Client()
    ctx = _registered(client)
    try:
        future = _submit(ctx)
        client.retrieves[0][1].set_result(ServerModuleCallResponse(False, b"", "x"))
        future.wait_for_layer(0)
        assert future.result() is False
    finally:
        ctx.close()


def test_query_and_wait_on_stream_follow_the_completion_event(backend):
    client = _Client()
    ctx = _registered(client)
    try:
        future = _submit(ctx)
        client.answer(0, protocol.RetrieveResponse(b"done", True, [0, 1, 2, 3]))
        backend.done = False
        assert not future.query()
        backend.done = True
        assert future.query()
        assert future.wait_on_stream(_STREAM) is True
        assert backend.waits == ["done"]
    finally:
        ctx.close()


def test_a_layer_never_reported_times_out(backend):
    pool = EventPool.import_pool(backend, torch.device("cpu"), [b"e0"])
    future = LayerwiseRetrieveFuture(
        _Call(),
        MessagingFuture(),
        device=torch.device("cpu"),
        event_backend=backend,
        pool=pool,
        layer_timeout=0.05,
    )
    with pytest.raises(LMCacheTimeoutError):
        future.wait_for_layer(0)


def test_frames_of_dropped_futures_are_ignored(backend):
    client = _Client()
    ctx = _registered(client)
    try:
        future = _submit(ctx)
        call_id = client.retrieves[0][0].call_id
        del future
        client.push_frame(call_id, 0, [0])
        later = _submit(ctx)
        later_id = client.retrieves[1][0].call_id
        client.push_frame(later_id, 1, [1])
        later.wait_for_layer(1)
        assert backend.waits == ["e1"]
        _wait_until(lambda: not ctx._calls.get(call_id))
    finally:
        ctx.close()


def test_register_warns_when_no_probe_arrives(backend, monkeypatch):
    warnings = []
    monkeypatch.setattr(
        worker_transfer.logger,
        "warning",
        lambda msg, *args: warnings.append(msg % args),
    )
    client = _Client()
    client.answer_probes = False
    ctx = LayerwiseTransferContext(_INSTANCE, client)
    try:
        ctx.register(_caches(), "m", 1, 16, 0.1)
        assert client.calls.count(protocol.PROBE_FRAMES) > 1
        assert len(warnings) == 1
        assert f"127.0.0.1:{client.port} did not deliver" in warnings[0]
        future = _submit(ctx)
        client.answer(0, protocol.RetrieveResponse(b"done", True, [0, 0, 2, 2]))
        future.wait_for_layer(3)
        assert backend.waits == ["e2"]
    finally:
        ctx.close()


def test_close_ends_the_frame_reader(backend):
    client = _Client()
    ctx = _registered(client)
    readers = [
        t for t in threading.enumerate() if t.name.startswith("lmcache-layerwise")
    ]
    assert readers
    ctx.close()
    for reader in readers:
        reader.join(timeout=5)
        assert not reader.is_alive()


def test_the_future_is_read_only(backend):
    pool = EventPool.import_pool(backend, torch.device("cpu"), [])
    future = LayerwiseRetrieveFuture(
        _Call(),
        MessagingFuture(),
        device=torch.device("cpu"),
        event_backend=backend,
        pool=pool,
    )
    with pytest.raises(NotImplementedError):
        future.set_result(True)
    assert worker_transfer.LAYER_TIMEOUT_S > 0


def test_debug_logs_the_waits_and_frames_of_a_retrieve(backend, monkeypatch):
    monkeypatch.setattr(debug, "ENABLED", True)
    logs = []
    monkeypatch.setattr(
        worker_transfer.logger, "info", lambda msg, *args: logs.append(msg % args)
    )
    client = _Client()
    ctx = _registered(client)
    try:
        future = _submit(ctx)
        client.push_frame(client.retrieves[0][0].call_id, 0, [0, 1])
        future.wait_for_layer(1)
        future.wait_for_layer(0)  # same event
        client.answer(0, protocol.RetrieveResponse(b"done", True, [0, 0, 2, 2]))
        future.wait_for_layer(3)  # from the answer
        assert future.result() is True
        assert future.wait()
    finally:
        ctx.close()
    assert len(logs) == 1
    assert logs[0].startswith("layerwise-wait: 3 call(s), 2 stream wait(s), ")
    assert "1 frame(s) for 2 KV cache(s)" in logs[0]


def test_no_debug_state_by_default(backend):
    client = _Client()
    ctx = _registered(client)
    try:
        future = _submit(ctx)
        assert future._stats is None
        assert future._call.arrivals is None
    finally:
        ctx.close()
