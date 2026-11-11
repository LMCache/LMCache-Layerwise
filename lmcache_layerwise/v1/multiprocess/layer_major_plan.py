# SPDX-License-Identifier: Apache-2.0
"""Planning for the layer-major object layout.

A layer-major object is ordered by model depth: depth 0's caches, then depth
1's, and so on, with one layer's K and V adjacent. A run of consecutive
depths is then one contiguous byte range, which is what lets the retrieve
copy and signal a batch of layers before the rest of the object has landed.
The price is that a kernel group is scattered through the object; a launch
still covers several of its layers when they sit a constant stride apart
(see :func:`uniform_stride_runs`).

Store and retrieve must agree on this layout byte for byte, so both build
their plans from here.

``MemoryLayoutDesc`` does not record the layout: it names an object's
contents per kernel group, which reads as kernel-group-major. That is only
safe while every consumer uses it to size or allocate a buffer.
"""

# Standard
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

# Third Party
import torch

# First Party
from lmcache import device_ops
from lmcache.logging import init_logger
from lmcache.v1.kv_layer_groups import KVLayerGroupsManager
from lmcache.v1.platform.base.cache_context import BaseCacheContext
import lmcache.lmcache_native as lmcache_native

logger = init_logger(__name__)

# (kernel_group_idx, local_layer_idx, size_in_bytes) per layer, in object order.
Placement = list[tuple[int, int, int]]


def kernel_group_depths(
    kv_groups_manager: KVLayerGroupsManager,
    kernel_group_idx: int,
    model_depths: Sequence[int] | None,
) -> list[int]:
    """Model depth of each layer of a kernel group, in local-layer order.

    Args:
        kv_groups_manager: The registered geometry.
        kernel_group_idx: Which kernel group.
        model_depths: Model depth of every registered KV cache, indexed by
            registration order, or None when the worker did not report
            depths. Registration order is then used as the depth, which is
            right only when each model layer registers a single cache.
    """
    layer_indices = kv_groups_manager.kernel_groups[kernel_group_idx].layer_indices
    if model_depths is None:
        return list(layer_indices)
    return [model_depths[i] for i in layer_indices]


def layer_major_placement(
    kv_groups_manager: KVLayerGroupsManager,
    object_group_idx: int,
    size_for_kernel_group: Callable[[int], int],
    model_depths: Sequence[int] | None,
) -> Placement:
    """Order one object group's per-layer slices by model depth.

    Ties between kernel groups at the same depth are broken by kernel group.

    Args:
        kv_groups_manager: The registered geometry.
        object_group_idx: Which object group to place.
        size_for_kernel_group: Byte size of one kernel group's staging
            region, as the staging buffer computes it.
        model_depths: See :func:`kernel_group_depths`.

    Returns:
        ``[(kernel_group_idx, local_layer_idx, size_in_bytes)]`` in object
        order.

    Raises:
        ValueError: For a geometry this layout cannot address: a kernel group
            whose staging bytes do not split evenly across its layers, or one
            in the fused cross-layer KV format, whose engine tensor is
            addressed by an absolute layer index and so cannot be copied one
            layer at a time.
    """
    entries: list[tuple[int, int, int, int]] = []
    for kernel_group_idx in kv_groups_manager.object_groups[
        object_group_idx
    ].kernel_group_indices:
        group = kv_groups_manager.kernel_groups[kernel_group_idx]
        num_layers = group.num_layers
        size = size_for_kernel_group(kernel_group_idx)
        if num_layers <= 0 or size % num_layers != 0:
            raise ValueError(
                f"Layer-wise transfer needs a whole number of staging bytes "
                f"per layer, but kernel group {kernel_group_idx} stages "
                f"{size} bytes across {num_layers} layers."
            )
        fmt = group.engine_kv_format
        if fmt is not None and lmcache_native.is_cross_layer(fmt):
            raise ValueError(
                f"Layer-wise transfer does not support the fused cross-layer "
                f"KV format of kernel group {kernel_group_idx}: its engine "
                f"tensor is addressed by an absolute layer index, so one layer "
                f"cannot be transferred on its own."
            )
        per_layer = size // num_layers
        depths = kernel_group_depths(kv_groups_manager, kernel_group_idx, model_depths)
        for local, depth in enumerate(depths):
            entries.append((depth, kernel_group_idx, local, per_layer))

    entries.sort()
    return [(kg, local, per_layer) for _depth, kg, local, per_layer in entries]


def placement_keeps_kernel_groups_contiguous(placement: Placement) -> bool:
    """Whether a placement leaves every kernel group one contiguous run.

    True when each model layer owns its caches in a single kernel group, the
    common case: depth order then equals kernel-group-major order. A model
    whose layers span several kernel groups interleaves them by depth.

    Args:
        placement: Per-layer slices in object order.
    """
    seen: set[int] = set()
    current = -1
    next_local = 0
    for kernel_group_idx, local, _size in placement:
        if kernel_group_idx != current:
            if kernel_group_idx in seen:
                return False
            seen.add(kernel_group_idx)
            current = kernel_group_idx
            next_local = 0
        if local != next_local:
            return False
        next_local += 1
    return True


def carve_layer_major_object_group(
    placement: Placement,
    batch_idx: int,
    offset: int,
    layer_offsets: dict[tuple[int, int, int], tuple[int, int]],
) -> tuple[int, int]:
    """Lay one object group out in placement order.

    Args:
        placement: Per-layer slices in object order.
        batch_idx: Which batch slot is being carved.
        offset: Byte offset the object group starts at.
        layer_offsets: Map to record ``(batch_idx, kernel_group_idx,
            local_layer_idx) -> (byte offset, size in bytes)`` into.

    Returns:
        The offset just past the object group, and its size in bytes.
    """
    start = offset
    for kernel_group_idx, local, per_layer in placement:
        layer_offsets[(batch_idx, kernel_group_idx, local)] = (offset, per_layer)
        offset += per_layer
    return offset, offset - start


def uniform_stride_runs(offsets: Sequence[int]) -> list[tuple[int, int, int]]:
    """Split a kernel group's layers into maximal constant-stride runs.

    One launch walks the object forward with a single layer stride, so
    consecutive layers share a launch only while their offsets increase by
    the same amount.

    Args:
        offsets: Byte offsets of the layers, in local-layer order.

    Returns:
        ``(start_index, length, stride_bytes)`` per run; ``stride_bytes`` is 0
        for a single-layer run, where no stride applies.
    """
    runs: list[tuple[int, int, int]] = []
    i = 0
    n = len(offsets)
    while i < n:
        j = i + 1
        stride = 0
        while j < n:
            delta = offsets[j] - offsets[j - 1]
            if delta <= 0 or (j > i + 1 and delta != stride):
                break
            stride = delta
            j += 1
        runs.append((i, j - i, stride if j - i > 1 else 0))
        i = j
    return runs


def build_run_shape_desc(shape_desc: Any, nl: int) -> Any:
    """Clone ``shape_desc`` for a run of ``nl`` of the group's layers."""
    run_desc = lmcache_native.PageBufferShapeDesc()
    run_desc.kv_size = shape_desc.kv_size
    run_desc.nl = nl
    run_desc.nb = shape_desc.nb
    run_desc.bs = shape_desc.bs
    run_desc.nh = shape_desc.nh
    run_desc.hs = shape_desc.hs
    run_desc.element_size = shape_desc.element_size
    run_desc.block_stride_elems = shape_desc.block_stride_elems
    return run_desc


@dataclass
class LayerMajorSpecs:
    """Kernel group specs plus the per-spec layout ``_C`` takes beside them.

    ``PageBufferShapeDesc`` has no layout fields, so each spec's interleaved
    layout and layer stride travel as parallel lists.
    """

    specs: list[Any] = field(default_factory=list)
    kv_interleaved: list[bool] = field(default_factory=list)
    layer_stride_elems: list[int] = field(default_factory=list)

    def add_run(
        self,
        shape_desc: Any,
        *,
        nl: int,
        stride_bytes: int,
        paged_buffer_ptrs: int,
        lmcache_objects_ptrs: list[int],
        lmcache_chunk_size: int,
        engine_kv_format: Any,
        block_ids: torch.Tensor,
    ) -> int:
        """Add the spec for one constant-stride run of layers.

        Returns:
            The index of the new spec.

        Raises:
            ValueError: If the stride is not a whole number of elements.
        """
        if stride_bytes % shape_desc.element_size != 0:
            raise ValueError(
                f"Layer stride {stride_bytes} B is not a whole number of "
                f"{shape_desc.element_size}-byte elements"
            )
        self.specs.append(
            device_ops.KernelGroupSpec(
                paged_buffer_ptrs,
                lmcache_objects_ptrs,
                build_run_shape_desc(shape_desc, nl),
                lmcache_chunk_size,
                engine_kv_format,
                block_ids.data_ptr(),
                block_ids.numel(),
            )
        )
        self.kv_interleaved.append(True)
        self.layer_stride_elems.append(stride_bytes // shape_desc.element_size)
        return len(self.specs) - 1


def build_layer_major_specs(
    cache_context: BaseCacheContext,
    object_group_id: int,
    block_ids_gpu: list[torch.Tensor],
    max_batch_size: int,
) -> tuple[LayerMajorSpecs, list[tuple[int, int]]]:
    """Build the specs that copy whole objects of one object group.

    Each kernel group becomes one spec per constant-stride run, addressing
    the run's layers in every batch slot of the staging buffer.

    Args:
        cache_context: A layer-wise cache context (it must provide
            ``get_temp_layer_buffer`` and ``get_layer_offset_in_object``).
        object_group_id: Which object group.
        block_ids_gpu: GPU block IDs, indexed by kernel group.
        max_batch_size: Number of staging batch slots to address.

    Returns:
        The specs, and one ``(spec_index, kernel_group_id)`` launch unit per
        spec.
    """
    kv_groups_manager = cache_context.kv_layer_groups_manager
    specs = LayerMajorSpecs()
    launch_units: list[tuple[int, int]] = []
    for kernel_group_id in kv_groups_manager.object_groups[
        object_group_id
    ].kernel_group_indices:
        num_layers = kv_groups_manager.kernel_groups[kernel_group_id].num_layers
        offsets = [
            cache_context.get_layer_offset_in_object(  # type: ignore[attr-defined]
                object_group_id, kernel_group_id, local
            )
            for local in range(num_layers)
        ]
        paged_ptrs = cache_context.get_kernel_group_kv_pointers(kernel_group_id)
        for first_local, run_len, stride_bytes in uniform_stride_runs(offsets):
            # A view of a tensor the context holds for its whole life, so the
            # sliced pointer array stays valid without a keepalive.
            run_paged_ptrs = paged_ptrs[first_local : first_local + run_len]
            spec_index = specs.add_run(
                cache_context.get_shape_desc(kernel_group_id),
                nl=run_len,
                stride_bytes=stride_bytes,
                paged_buffer_ptrs=run_paged_ptrs.data_ptr(),
                lmcache_objects_ptrs=[
                    cache_context.get_temp_layer_buffer(  # type: ignore[attr-defined]
                        slot, kernel_group_id, first_local
                    ).data_ptr()
                    for slot in range(max_batch_size)
                ],
                lmcache_chunk_size=cache_context.get_slots_per_chunk_in_sw(
                    kernel_group_id
                ),
                engine_kv_format=cache_context.get_engine_kv_format(kernel_group_id),
                block_ids=block_ids_gpu[kernel_group_id],
            )
            launch_units.append((spec_index, kernel_group_id))
    return specs, launch_units
