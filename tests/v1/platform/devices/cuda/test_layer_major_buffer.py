# SPDX-License-Identifier: Apache-2.0
"""The layer-major staging buffer, carved over a real geometry on CPU."""

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.kv_layer_groups import KVLayerGroupsManager
from lmcache.v1.multiprocess.group_view import EngineGroupInfo
from lmcache.v1.platform.devices.cuda.cache_context import _TempGPUBuffer
from lmcache_layerwise.v1.multiprocess.layer_major_plan import layer_major_placement
from lmcache_layerwise.v1.platform.devices.cuda.cache_context import (
    _TempLayerMajorGPUBuffer,
)
import lmcache.lmcache_native as lmcache_native

_CHUNK = 16
_BATCH = 2


def _manager(engine_groups):
    """Two caches per model layer, one in each engine group.

    Layers of group 0 have 8 heads and those of group 1 have 4, so the two
    kernel groups have different per-layer sizes.
    """
    num_caches = sum(len(layers) for layers in engine_groups)
    heads = {}
    for group_idx, layers in enumerate(engine_groups):
        for layer in layers:
            heads[layer] = 8 if group_idx == 0 else 4
    tensors = [
        torch.zeros(2, 4, _CHUNK, heads[i], 64, dtype=torch.float16)
        for i in range(num_caches)
    ]
    return KVLayerGroupsManager(
        tensors,
        engine_kv_formats=[lmcache_native.EngineKVFormat.NL_X_TWO_NB_BS_NH_HS]
        * num_caches,
        engine_group_infos=[
            EngineGroupInfo(i, tuple(layers)) for i, layers in enumerate(engine_groups)
        ],
        lmcache_tokens_per_chunk=_CHUNK,
    )


def _buffer(model_depths, engine_groups=((0, 2), (1, 3))):
    return _TempLayerMajorGPUBuffer(
        _manager(engine_groups),
        _CHUNK,
        torch.device("cpu"),
        _BATCH,
        model_depths,
    )


def test_every_layer_is_carved_where_the_placement_puts_it():
    buf = _buffer([0, 0, 1, 1])
    placement = layer_major_placement(
        buf._kv_groups_manager, 0, buf._get_size_for_kernel_group, [0, 0, 1, 1]
    )

    offset = 0
    for kernel_group_idx, local, size in placement:
        assert buf._offset_map_layer[(0, kernel_group_idx, local)] == (offset, size)
        assert buf.get_layer_offset_in_object(0, kernel_group_idx, local) == offset
        offset += size
    assert buf._offset_map_object_group_only[(0, 0)] == (0, offset)


def test_every_batch_slot_repeats_the_same_order():
    buf = _buffer([0, 0, 1, 1])
    start, size = buf._offset_map_object_group_only[(1, 0)]

    assert start == size
    for (batch_idx, kg, local), (offset, layer_size) in buf._offset_map_layer.items():
        if batch_idx == 1:
            base = buf._offset_map_layer[(0, kg, local)]
            assert (offset, layer_size) == (base[0] + size, base[1])


def test_the_buffer_size_does_not_change():
    manager = _manager(((0, 2), (1, 3)))
    base = _TempGPUBuffer(manager, _CHUNK, torch.device("cpu"), _BATCH)
    layer_major = _buffer([0, 0, 1, 1])

    assert layer_major._temp_buffer.numel() == base._temp_buffer.numel()


def test_a_layer_view_aliases_its_slice_of_the_buffer():
    buf = _buffer([0, 0, 1, 1])
    buf._temp_buffer.zero_()
    view = buf.get_temp_layer_buffer(1, 1, 1)
    offset, size = buf._offset_map_layer[(1, 1, 1)]

    assert view.shape[1] == 1
    assert view.numel() * view.element_size() == size
    assert view.data_ptr() == buf._temp_buffer.data_ptr() + offset
    view.fill_(1)
    outside = torch.cat([buf._temp_buffer[:offset], buf._temp_buffer[offset + size :]])
    assert not outside.any()


def test_interleaved_kernel_groups_are_reported():
    assert not _buffer([0, 0, 1, 1]).kernel_groups_contiguous
    assert _buffer(
        [0, 1, 2, 3], engine_groups=((0, 1), (2, 3))
    ).kernel_groups_contiguous


def test_the_kernel_group_view_is_refused():
    with pytest.raises(NotImplementedError):
        _buffer([0, 0, 1, 1]).get_temp_kernel_group_buffer(0, 0)


def test_unknown_layers_are_refused():
    buf = _buffer([0, 0, 1, 1])

    with pytest.raises(ValueError):
        buf.get_temp_layer_buffer(_BATCH, 0, 0)
    with pytest.raises(ValueError):
        buf.get_layer_offset_in_object(0, 0, 2)
