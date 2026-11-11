# SPDX-License-Identifier: Apache-2.0
"""Payloads of the ``layerwise.*`` server-module methods.

Every payload is msgpack. A retrieve is one ``layerwise.retrieve`` affinity
call; while it runs, the server publishes a :class:`Frame` per layer batch,
from the retrieve's own thread, on the frame channel: a ZMQ PUB socket on
the server's host, at the port :class:`RegisterEventsResponse` names, with one
topic per instance (:func:`frames_topic`). Before its first retrieve, the
worker calls ``layerwise.probe_frames`` until the probe frame arrives, so no
frame is lost to a subscription still in flight. Frames are not acknowledged
and may still be dropped past the socket's queue limit; the call's
:class:`RetrieveResponse` repeats every layer's event index, so a worker that
missed frames can still wait on each layer.
"""

# Standard
from typing import TypeVar

# Third Party
import msgspec

# First Party
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey

SET_MODEL_DEPTHS = "layerwise.set_model_depths"
REGISTER_EVENTS = "layerwise.register_events"
RETRIEVE = "layerwise.retrieve"
PROBE_FRAMES = "layerwise.probe_frames"

# Call id of the frame ``layerwise.probe_frames`` publishes; never a retrieve's.
PROBE_CALL_ID = -1


class SetModelDepthsRequest(msgspec.Struct, frozen=True):
    """Sent before ``register_kv_cache``.

    Attributes:
        instance_id: The worker instance.
        depths: Model depth of each KV cache, in registration order.
    """

    instance_id: int
    depths: list[int]


class InstanceRequest(msgspec.Struct, frozen=True):
    """A request that names only the worker instance."""

    instance_id: int


class RegisterEventsResponse(msgspec.Struct, frozen=True):
    """The instance's event pool.

    Attributes:
        layer_batch: Model depths per signalled layer batch.
        handles: IPC handle of each pool event; event ``i`` is recorded for
            batches led by KV cache ``i``.
        frames_port: TCP port of the frame channel; 0 for none.
    """

    layer_batch: int
    handles: list[bytes]
    frames_port: int = 0


class RetrieveRequest(msgspec.Struct, frozen=True):
    """A layer-wise retrieve. See ``LMCacheDrivenTransferModule.retrieve``.

    Attributes:
        call_id: Client-chosen tag of this call's frames.
    """

    call_id: int
    key: IPCCacheServerKey
    instance_id: int
    gpu_block_ids: list[list[int]]
    event_ipc_handle: bytes
    skip_first_n_tokens: int = 0


class RetrieveResponse(msgspec.Struct, frozen=True):
    """The result of a layer-wise retrieve.

    Attributes:
        event_handle: IPC handle of the event that completes the retrieve;
            empty when no device work was submitted.
        succeeded: Whether every object was found and copied.
        leaders: Pool event index of each KV cache's batch, by registration
            index; empty when nothing was submitted.
    """

    event_handle: bytes
    succeeded: bool
    leaders: list[int]


class Frame(msgspec.Struct, frozen=True, array_like=True):
    """One layer batch is enqueued.

    Attributes:
        call_id: The retrieve's :attr:`RetrieveRequest.call_id`.
        leader: Pool event index recorded after the batch.
        layers: Registration indices of the KV caches in the batch.
    """

    call_id: int
    leader: int
    layers: list[int]


_T = TypeVar("_T")

_encoder = msgspec.msgpack.Encoder()


def frames_topic(instance_id: int) -> bytes:
    """The frame channel topic of an instance."""
    return instance_id.to_bytes(8, "big", signed=True)


def encode(message: msgspec.Struct) -> bytes:
    """Encode a payload."""
    return _encoder.encode(message)


def decode(payload: bytes, message_type: type[_T]) -> _T:
    """Decode a payload of ``message_type``.

    Raises:
        msgspec.DecodeError: For a malformed payload.
    """
    return msgspec.msgpack.decode(payload, type=message_type)
