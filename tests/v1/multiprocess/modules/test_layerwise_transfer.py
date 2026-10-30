# SPDX-License-Identifier: Apache-2.0
"""The layer-wise module's RPCs, served through LMCache's router on CPU.

LMCache's retrieve loop is replaced by a stub that signals the batches a real
transfer would, so the tests pin this module's own logic: the event pool, the
frames, and the leader map.
"""

# Standard
from types import SimpleNamespace
import contextlib
import socket
import threading

# Third Party
import pytest
import torch
import zmq

# First Party
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.ext_server_module import (
    build_server_module_router,
    iter_server_module_handlers,
)
from lmcache.v1.multiprocess.modules.lmcache_driven_transfer import (
    LMCacheDrivenTransferModule,
)
from lmcache.v1.multiprocess.protocols.server_module import (
    ServerModuleCallRequest,
)
from lmcache.v1.multiprocess.token_codec import pack_token_ids
from lmcache_layerwise.v1.multiprocess import protocol
from lmcache_layerwise.v1.multiprocess.modules import layerwise_transfer
from lmcache_layerwise.v1.multiprocess.modules.layerwise_transfer import (
    LayerwiseServerModule,
    LayerwiseTransferModule,
)
from lmcache_layerwise.v1.platform.base import event_pool

_INSTANCE = 7
_NUM_LAYERS = 4


class _Backend:
    """Records what is done to each event; events are their own indices."""

    def __init__(self):
        self.created = 0
        self.records = []

    def create_event(self, device):
        self.created += 1
        return self.created - 1

    def export_event(self, event, device):
        return b"h%d" % event

    def record_event(self, event, stream):
        self.records.append((event, stream))


class _NativeBackend(_Backend):
    """Events also carry a raw device handle, as ``torch.cuda.Event`` does."""

    def create_event(self, device):
        return SimpleNamespace(cuda_event=0x100 + super().create_event(device))

    def export_event(self, event, device):
        return b"h%d" % event.cuda_event


def _key():
    return IPCCacheServerKey("m", 1, 0, pack_token_ids([1, 2]), 0, 2, "req")


@pytest.fixture
def module(monkeypatch):
    def init(self, ctx):
        self._ctx = ctx
        self._cache_contexts = {}
        self._lock = threading.Lock()

    monkeypatch.setattr(LMCacheDrivenTransferModule, "__init__", init)
    # The instance is on CPU, which a CUDA host's device guard refuses.
    no_guard = SimpleNamespace(device=lambda _device: contextlib.nullcontext())
    monkeypatch.setattr(event_pool, "torch_dev", no_guard)
    monkeypatch.setattr(layerwise_transfer, "torch_dev", no_guard)
    module = LayerwiseTransferModule(SimpleNamespace(), layer_batch=2)
    module.released = []
    monkeypatch.setattr(
        module,
        "_release_failed_retrieve_locks",
        lambda key, instance_id: module.released.append(instance_id),
    )
    yield module
    if module._frames is not None:
        module._frames.close()


def _register(module, backend=None):
    backend = backend or _Backend()
    cache_context = SimpleNamespace(
        stream="stream", device=torch.device("cpu"), num_layers=_NUM_LAYERS
    )
    module._cache_contexts[_INSTANCE] = SimpleNamespace(
        cache_context=cache_context, event_backend=backend, last_seen=0.0
    )
    return backend


def _router(module):
    router = build_server_module_router(
        SimpleNamespace(), [LayerwiseServerModule(module)]
    )
    assert router is not None
    return router


def _retrieve_request(call_id=3):
    return ServerModuleCallRequest(
        method=protocol.RETRIEVE,
        payload=protocol.encode(
            protocol.RetrieveRequest(call_id, _key(), _INSTANCE, [[0, 1]], b"ev", 0)
        ),
    )


def _stub_base_retrieve(monkeypatch, batches, result=(b"done", True)):
    """Replace LMCache's retrieve loop with one that signals ``batches`` as
    one object group's native call would: their events, then each batch."""

    def retrieve(self, key, instance_id, gpu_block_ids, event, skip):
        signal = self._tls.signal
        events = [signal.event(layers) for layers in batches]
        for layers, handle in zip(batches, events, strict=True):
            signal(layers, recorded=handle != 0)
        return result

    monkeypatch.setattr(LMCacheDrivenTransferModule, "retrieve", retrieve)


@contextlib.contextmanager
def _subscribed(module, instance_id=_INSTANCE):
    """A SUB socket receiving the instance's frames, once a probe through
    ``layerwise.probe_frames`` arrives."""
    sub = zmq.Context.instance().socket(zmq.SUB)
    sub.setsockopt(zmq.LINGER, 0)
    sub.setsockopt(zmq.SUBSCRIBE, protocol.frames_topic(instance_id))
    sub.connect(f"tcp://127.0.0.1:{module._frames.port}")
    try:
        router = _router(module)
        probe = ServerModuleCallRequest(
            method=protocol.PROBE_FRAMES,
            payload=protocol.encode(protocol.InstanceRequest(instance_id)),
        )
        while not sub.poll(10):
            assert router.server_module_call(probe).success
        _, payload = sub.recv_multipart()
        assert protocol.decode(payload, protocol.Frame) == protocol.Frame(
            protocol.PROBE_CALL_ID, -1, []
        )
        while sub.poll(50):
            sub.recv_multipart()
        yield sub
    finally:
        sub.close()


def _read_frames(sub, count):
    frames = []
    while len(frames) < count:
        assert sub.poll(5000)
        topic, payload = sub.recv_multipart()
        assert topic == protocol.frames_topic(_INSTANCE)
        frames.append(protocol.decode(payload, protocol.Frame))
    return frames


def test_the_server_module_serves_each_method_on_its_rpc():
    handlers = {
        h.method: h.requires_client_affinity
        for h in iter_server_module_handlers(LayerwiseServerModule)
    }

    assert handlers == {
        protocol.SET_MODEL_DEPTHS: False,
        protocol.REGISTER_EVENTS: False,
        protocol.PROBE_FRAMES: False,
        protocol.RETRIEVE: True,
    }


def test_register_events_creates_one_event_per_cache_once(module):
    backend = _register(module)
    router = _router(module)
    request = ServerModuleCallRequest(
        method=protocol.REGISTER_EVENTS,
        payload=protocol.encode(protocol.InstanceRequest(_INSTANCE)),
    )

    first = router.server_module_call(request)
    second = router.server_module_call(request)

    assert first.success and second.success
    response = protocol.decode(first.payload, protocol.RegisterEventsResponse)
    assert response.layer_batch == 2
    assert response.handles == [b"h0", b"h1", b"h2", b"h3"]
    assert response.frames_port == module._frames.port > 0
    assert second.payload == first.payload
    assert backend.created == _NUM_LAYERS


def test_register_events_refuses_an_unregistered_instance(module):
    response = _router(module).server_module_call(
        ServerModuleCallRequest(
            method=protocol.REGISTER_EVENTS,
            payload=protocol.encode(protocol.InstanceRequest(_INSTANCE)),
        )
    )

    assert not response.success
    assert "not registered" in response.error


def test_retrieve_signals_each_batch_then_every_uncovered_layer(module, monkeypatch):
    backend = _register(module)
    module.register_events(_INSTANCE)
    router = _router(module)
    _stub_base_retrieve(monkeypatch, [[0, 2]])

    with _subscribed(module) as frames:
        reply = router.server_module_affinity_call(_retrieve_request(call_id=3))

        assert reply.success
        response = protocol.decode(reply.payload, protocol.RetrieveResponse)
        assert response == protocol.RetrieveResponse(b"done", True, [0, 1, 0, 1])
        assert _read_frames(frames, 2) == [
            protocol.Frame(3, 0, [0, 2]),
            protocol.Frame(3, 1, [1, 3]),
        ]
    assert backend.records == [(0, "stream"), (1, "stream")]


def test_batches_the_native_call_recorded_are_not_recorded_again(module, monkeypatch):
    backend = _register(module, _NativeBackend())
    module.register_events(_INSTANCE)
    _stub_base_retrieve(monkeypatch, [[0, 2]])

    response = module.retrieve_layerwise(
        protocol.decode(_retrieve_request().payload, protocol.RetrieveRequest)
    )

    assert response == protocol.RetrieveResponse(b"done", True, [0, 1, 0, 1])
    # Only the uncovered layers' leader, which no native call recorded.
    assert [(e.cuda_event, s) for e, s in backend.records] == [(0x101, "stream")]


def test_a_failed_retrieve_still_signals_every_layer(module, monkeypatch):
    _register(module)
    module.register_events(_INSTANCE)
    _stub_base_retrieve(monkeypatch, [], result=(b"", False))

    reply = _router(module).server_module_affinity_call(_retrieve_request())

    response = protocol.decode(reply.payload, protocol.RetrieveResponse)
    assert response == protocol.RetrieveResponse(b"", False, [0, 0, 0, 0])


def test_retrieve_without_events_is_refused_and_releases_locks(module, monkeypatch):
    _register(module)
    _stub_base_retrieve(monkeypatch, [[0]])

    reply = _router(module).server_module_affinity_call(_retrieve_request())

    response = protocol.decode(reply.payload, protocol.RetrieveResponse)
    assert response == protocol.RetrieveResponse(b"", False, [])
    assert module.released == [_INSTANCE]


def test_the_signal_is_unbound_after_a_retrieve(module, monkeypatch):
    _register(module)
    module.register_events(_INSTANCE)
    _stub_base_retrieve(monkeypatch, [[0, 1, 2, 3]])

    module.retrieve_layerwise(
        protocol.decode(_retrieve_request().payload, protocol.RetrieveRequest)
    )

    assert module._tls.signal is None


def test_plain_retrieve_is_refused_and_releases_locks(module):
    assert module.retrieve(_key(), _INSTANCE, [[0]], b"ev") == (b"", False)
    assert module.released == [_INSTANCE]


def test_depths_reach_the_cache_context(module, monkeypatch):
    built = {}

    def context(*args, **kwargs):
        built.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(layerwise_transfer, "GPULayerwiseCacheContext", context)
    module._ctx = SimpleNamespace(
        chunk_size=16, separate_object_groups=False, full_sw_kv=False
    )
    reply = _router(module).server_module_call(
        ServerModuleCallRequest(
            method=protocol.SET_MODEL_DEPTHS,
            payload=protocol.encode(
                protocol.SetModelDepthsRequest(_INSTANCE, [0, 0, 1, 1])
            ),
        )
    )
    assert reply.success

    module._create_cache_context(
        [],
        instance_id=_INSTANCE,
        engine_type=None,
        layout_hints=None,
        engine_group_infos=(),
    )

    assert built["model_depths"] == [0, 0, 1, 1]
    assert built["layer_batch"] == 2
    assert built["layer_signal"]() is None


def test_negative_depths_are_refused(module):
    with pytest.raises(ValueError):
        module.set_model_depths(_INSTANCE, [0, -1])


def test_frames_reach_only_their_instance(module):
    _register(module)
    module.register_events(_INSTANCE)
    with _subscribed(module) as frames:
        module._publish_frame(_INSTANCE + 1, protocol.Frame(1, 0, [0]))
        module._publish_frame(_INSTANCE, protocol.Frame(2, 1, [1]))
        assert _read_frames(frames, 1) == [protocol.Frame(2, 1, [1])]
        assert not frames.poll(50)


def test_the_frame_channel_binds_a_fixed_port_once():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    publisher = layerwise_transfer._FramePublisher("0.0.0.0", port)
    try:
        assert publisher.port == port
        with pytest.raises(RuntimeError, match=f"0.0.0.0:{port}"):
            layerwise_transfer._FramePublisher("0.0.0.0", port)
    finally:
        publisher.close()
    publisher.publish(_INSTANCE, protocol.Frame(1, 0, [0]))  # closed: dropped


def test_dropping_an_instance_drops_its_state(module):
    _register(module)
    module.register_events(_INSTANCE)
    module.set_model_depths(_INSTANCE, [0, 1, 2, 3])

    module.drop_instance_state(_INSTANCE)

    assert _INSTANCE not in module._event_pools
    assert _INSTANCE not in module._model_depths


def test_layer_batch_must_be_positive(module):
    with pytest.raises(ValueError):
        LayerwiseTransferModule(SimpleNamespace(), layer_batch=0)
