# SPDX-License-Identifier: Apache-2.0
"""The layer-major placement rule and the run arithmetic built on it.

Driven with a stub geometry, so no GPU and no registration are needed.
"""

# Standard
from dataclasses import dataclass, field

# Third Party
import pytest

# First Party
from lmcache_layerwise.v1.multiprocess.layer_major_plan import (
    carve_layer_major_object_group,
    layer_major_placement,
    placement_keeps_kernel_groups_contiguous,
    uniform_stride_runs,
)
import lmcache.lmcache_native as lmcache_native

_PER_LAYER_FMT = lmcache_native.EngineKVFormat.NL_X_TWO_NB_BS_NH_HS


@dataclass
class _Group:
    layer_indices: list[int]
    engine_kv_format: object = _PER_LAYER_FMT

    @property
    def num_layers(self) -> int:
        return len(self.layer_indices)


@dataclass
class _ObjectGroup:
    kernel_group_indices: list[int]


@dataclass
class _Manager:
    kernel_groups: list[_Group]
    object_groups: list[_ObjectGroup] = field(default_factory=list)


def _manager(layer_indices_per_group, object_groups=None):
    groups = [_Group(list(indices)) for indices in layer_indices_per_group]
    if object_groups is None:
        object_groups = [list(range(len(groups)))]
    return _Manager(groups, [_ObjectGroup(list(og)) for og in object_groups])


def _placement(manager, per_layer, model_depths, object_group_idx=0):
    def size(kernel_group_idx):
        group = manager.kernel_groups[kernel_group_idx]
        return per_layer[kernel_group_idx] * group.num_layers

    return layer_major_placement(manager, object_group_idx, size, model_depths)


def _offsets(placement):
    offsets = {}
    running = 0
    for kernel_group_idx, local, size in placement:
        offsets[(kernel_group_idx, local)] = running
        running += size
    return offsets


def _v4_flash():
    """DeepSeek-V4-Flash-like: 167 caches in 8 type-major kernel groups.

    Registration order and model depth disagree, so the depths decide.
    """
    even = list(range(2, 43, 2))
    odd = list(range(3, 42, 2))
    depths_per_group = [even, even, odd, [0] + even, [1] + odd, even, even, odd]
    per_layer = [4096, 9344, 1168, 2336, 584, 1024, 2048, 512]
    indices = []
    model_depths = []
    for depths in depths_per_group:
        start = len(model_depths)
        indices.append(list(range(start, start + len(depths))))
        model_depths.extend(depths)
    return _manager(indices), per_layer, model_depths


# --- uniform_stride_runs -----------------------------------------------------


def test_even_spread_is_one_run():
    assert uniform_stride_runs([0, 10, 20, 30]) == [(0, 4, 10)]


def test_single_layer_run_has_no_stride():
    assert uniform_stride_runs([7]) == [(0, 1, 0)]


def test_no_layers_is_no_runs():
    assert uniform_stride_runs([]) == []


def test_run_breaks_where_the_spread_changes():
    assert uniform_stride_runs([0, 4, 14, 24, 34]) == [(0, 2, 4), (2, 3, 10)]


def test_runs_only_walk_forward():
    # The kernel's layer stride is unsigned.
    assert uniform_stride_runs([30, 20, 10, 0]) == [(i, 1, 0) for i in range(4)]
    assert uniform_stride_runs([10, 0, 30, 20]) == [(0, 1, 0), (1, 2, 30), (3, 1, 0)]


@pytest.mark.parametrize(
    "offsets",
    [
        [0, 1, 3, 6, 10, 15],
        [0, 8, 16, 17, 18, 40],
        [5, 9],
        [0, 3, 6, 100, 200],
        [20, 10, 0, 30],
    ],
)
def test_runs_cover_every_layer_once_at_their_stride(offsets):
    covered = []
    for start, length, stride in uniform_stride_runs(offsets):
        assert stride >= 0
        for step in range(length):
            assert offsets[start + step] == offsets[start] + step * stride
            covered.append(start + step)
    assert covered == list(range(len(offsets)))


# --- layer_major_placement ---------------------------------------------------


def test_one_cache_per_layer_keeps_kernel_group_major_order():
    manager = _manager([[0, 2], [1, 3]])
    placement = _placement(manager, [8, 4], [0, 1, 2, 3])

    assert [(kg, local) for kg, local, _ in placement] == [
        (0, 0),
        (1, 0),
        (0, 1),
        (1, 1),
    ]
    assert not placement_keeps_kernel_groups_contiguous(placement)

    contiguous = _placement(_manager([[0, 1], [2, 3]]), [8, 4], [0, 1, 2, 3])
    assert placement_keeps_kernel_groups_contiguous(contiguous)


def test_placement_orders_by_depth_then_kernel_group():
    manager = _manager([[0, 2], [1, 3]])
    placement = _placement(manager, [8, 4], [0, 0, 1, 1])

    assert placement == [(0, 0, 8), (1, 0, 4), (0, 1, 8), (1, 1, 4)]


def test_placement_falls_back_to_registration_order_without_depths():
    manager = _manager([[0, 2], [1, 3]])

    assert _placement(manager, [8, 4], None) == _placement(
        manager, [8, 4], [0, 1, 2, 3]
    )


def test_placement_covers_every_cache_once():
    manager, per_layer, depths = _v4_flash()
    placement = _placement(manager, per_layer, depths)

    expected = {
        (kg, local)
        for kg, group in enumerate(manager.kernel_groups)
        for local in range(group.num_layers)
    }
    assert {(kg, local) for kg, local, _ in placement} == expected
    assert len(placement) == len(depths)


def test_placement_refuses_a_size_that_does_not_split_across_layers():
    manager = _manager([[0, 1]])

    with pytest.raises(ValueError, match="whole number of staging bytes"):
        layer_major_placement(manager, 0, lambda _kg: 9, None)


def test_placement_refuses_an_empty_kernel_group():
    manager = _manager([[0], []])

    with pytest.raises(ValueError, match="whole number of staging bytes"):
        _placement(manager, [8, 8], [0])


def test_placement_refuses_the_fused_cross_layer_format():
    manager = _manager([[0, 1]])
    manager.kernel_groups[
        0
    ].engine_kv_format = lmcache_native.EngineKVFormat.NB_NL_TWO_BS_NH_HS

    with pytest.raises(ValueError, match="cross-layer"):
        _placement(manager, [8], None)


# --- runs and batches over a real geometry -----------------------------------


def test_a_group_owning_a_leading_layer_costs_one_extra_run():
    manager, per_layer, depths = _v4_flash()
    offsets = _offsets(_placement(manager, per_layer, depths))

    for kg, group in enumerate(manager.kernel_groups):
        ordered = [offsets[(kg, local)] for local in range(group.num_layers)]
        expected = 2 if kg in (3, 4) else 1
        assert len(uniform_stride_runs(ordered)) == expected, kg


def test_depth_batches_tile_a_contiguous_byte_range():
    manager, per_layer, depths = _v4_flash()
    placement = _placement(manager, per_layer, depths)
    offsets = _offsets(placement)
    size = {(kg, local): s for kg, local, s in placement}
    depth_of = {
        (kg, local): depths[index]
        for kg, group in enumerate(manager.kernel_groups)
        for local, index in enumerate(group.layer_indices)
    }

    all_depths = sorted(set(depths))
    for start in range(0, len(all_depths), 4):
        window = set(all_depths[start : start + 4])
        batch = [
            (kg, local) for kg, local, _ in placement if depth_of[(kg, local)] in window
        ]
        first = offsets[batch[0]]
        total = sum(size[e] for e in batch)
        assert offsets[batch[-1]] + size[batch[-1]] == first + total


def test_the_object_is_packed_in_placement_order_without_gaps():
    manager, per_layer, depths = _v4_flash()
    placement = _placement(manager, per_layer, depths)
    layer_offsets = {}

    end, size = carve_layer_major_object_group(placement, 1, 100, layer_offsets)

    assert size == sum(s for _, _, s in placement)
    assert end == 100 + size
    running = 100
    for kg, local, s in placement:
        assert layer_offsets[(1, kg, local)] == (running, s)
        running += s
