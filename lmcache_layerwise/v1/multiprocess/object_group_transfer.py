# SPDX-License-Identifier: Apache-2.0
"""Object-group copies in the layer-major layout.

The batch walk mirrors LMCache's ``_run_object_group_transfer_plan`` and
reuses its helpers; what differs is that a kernel group costs one spec per
constant-stride run (see :mod:`.layer_major_plan`) and that the copies go
through this plugin's ``_C``, which addresses layer-major objects.
"""

# Standard
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol
import itertools
import time

# Third Party
import torch

# First Party
from lmcache import device_ops
from lmcache.logging import init_logger
from lmcache.v1.gpu_connector.gpu_ops import build_staging_copies
from lmcache.v1.memory_allocators.lazy_memory_allocator import LazyMemoryAllocator
from lmcache.v1.memory_management import GDSMemoryObject, MemoryObj
from lmcache.v1.mp_observability.event import EventType
from lmcache.v1.mp_observability.event_bus import (
    get_event_bus,
    is_observability_enabled,
)
from lmcache.v1.multiprocess.object_group_transfer import (
    batched_iteration_with_skip,
    recalculate_blocks_to_skip,
)
from lmcache.v1.platform.base.cache_context import BaseCacheContext
from lmcache.v1.platform.ops_types import StagingCopy
import lmcache.lmcache_native as lmcache_native

# First Party
from lmcache_layerwise import debug, native
from lmcache_layerwise.v1.multiprocess.layer_major_plan import (
    build_layer_major_specs,
    build_run_shape_desc,
    kernel_group_depths,
    uniform_stride_runs,
)

logger = init_logger(__name__)


class LayerSignal(Protocol):
    """Signals each batch of layers enqueued on the cache context's stream.

    ``layers`` are the registration indices of the KV caches a batch fills.
    """

    def event(self, layers: list[int]) -> int:
        """The raw device event (``cudaEvent_t``) to record on the stream
        right after the batch filling ``layers``, or 0 for none."""
        ...

    def __call__(self, layers: list[int], *, recorded: bool = False) -> None:
        """Announce the batch filling ``layers``, once it is enqueued.

        Args:
            layers: The batch's KV caches.
            recorded: Whether the batch's :meth:`event` was recorded already;
                if not, the signal records its own.
        """
        ...


def build_staging_copy_range(
    memory_obj: MemoryObj,
    gpu_ptr: int,
    offset: int,
    nbytes: int,
    is_h2d: bool,
) -> StagingCopy:
    """Build one native ``StagingCopy`` for a byte range of a lazy object.

    Covers bytes ``[offset, offset + nbytes)`` of ``memory_obj``. The host
    pointer and the allocator offset (``meta.address``) are advanced together,
    which ``lmcache_memcpy_async`` needs to split the copy at the lazy
    allocator's pinned-chunk boundaries. Callers must ensure the object is
    lazy-allocator-backed.

    Args:
        memory_obj: Lazy-allocator memory object holding the host side.
        gpu_ptr: Device address of the GPU side of the copy.
        offset: Start of the range, in bytes from the start of the object.
        nbytes: Length of the range in bytes.
        is_h2d: True for retrieve (CPU->GPU), False for store (GPU->CPU).

    Returns:
        A ``device_ops.StagingCopy`` for the range.

    Raises:
        ValueError: If the object has not been allocated (``raw_tensor`` is
            None) or the range is empty or does not fit in the object.
    """
    if memory_obj.raw_tensor is None:
        raise ValueError(
            "memory_obj.raw_tensor is None; ensure the MemoryObj has been allocated."
        )
    mem_obj_size = memory_obj.get_size()
    if offset < 0 or nbytes <= 0 or offset + nbytes > mem_obj_size:
        raise ValueError(
            f"Invalid range: offset={offset}, nbytes={nbytes} for a "
            f"memory_obj of {mem_obj_size} bytes"
        )
    host_ptr = memory_obj.data_ptr + offset
    host_offset = memory_obj.meta.address + offset
    if is_h2d:
        return device_ops.StagingCopy(gpu_ptr, host_ptr, nbytes, host_offset)
    return device_ops.StagingCopy(host_ptr, gpu_ptr, nbytes, host_offset)


def _num_objects_to_skip(
    cache_context: BaseCacheContext,
    object_group_id: int,
    num_objects: int,
    is_h2d: bool,
) -> int:
    """Objects before a sliding window that a retrieve does not copy."""
    attn_desc = cache_context.kv_layer_groups_manager.get_attn_desc()
    if not is_h2d or attn_desc.is_full_attention(object_group_id):
        return 0
    return max(0, num_objects - attn_desc.num_chunks_in_sw[object_group_id])


def _blocks_per_chunk_and_window(
    cache_context: BaseCacheContext, kernel_group_id: int
) -> tuple[int, int]:
    chunk_size = cache_context.lmcache_tokens_per_chunk
    tokens_per_window = min(
        chunk_size,
        cache_context.kv_layer_groups_manager.get_subchunk_sw_size_tokens(
            kernel_group_id
        ),
    )
    return (
        cache_context.calculate_num_blocks(chunk_size, kernel_group_id),
        cache_context.calculate_num_blocks(tokens_per_window, kernel_group_id),
    )


def transfer_kv_layer_major(
    cache_context: BaseCacheContext,
    block_ids_gpu: list[torch.Tensor],
    memory_objs: Sequence[MemoryObj | None],
    *,
    object_group_id: int,
    batch_size: int,
    skip_first_n_tokens: int,
    direction: lmcache_native.TransferDirection,
    transfer_key: str,
) -> None:
    """Copy whole layer-major objects of one object group.

    Arguments are those of LMCache's ``transfer_kv_per_object_group``.

    Raises:
        NotImplementedError: For GDS-backed objects, which are not staged
            through the lazy allocator.
        ValueError: If an H2D batch holds a None object.
    """
    if any(isinstance(mo, GDSMemoryObject) for mo in memory_objs):
        raise NotImplementedError(
            "Layer-wise transfer does not support GDS-backed objects."
        )
    kv_groups_manager = cache_context.kv_layer_groups_manager
    chunk_size = cache_context.lmcache_tokens_per_chunk
    is_h2d = direction == lmcache_native.TransferDirection.H2D
    max_batch_size = cache_context.max_batch_size

    specs, launch_units = build_layer_major_specs(
        cache_context, object_group_id, block_ids_gpu, max_batch_size
    )
    blocks_by_kg = {
        kg: _blocks_per_chunk_and_window(cache_context, kg)
        for kg in kv_groups_manager.object_groups[object_group_id].kernel_group_indices
    }
    object_group_buffers = [
        cache_context.get_temp_object_group_buffer(slot, object_group_id)
        for slot in range(max_batch_size)
    ]

    batch_steps: list[Any] = []
    for start_object_idx, batch in batched_iteration_with_skip(
        memory_objs,
        batch_size,
        skip_count=_num_objects_to_skip(
            cache_context, object_group_id, len(memory_objs), is_h2d
        ),
    ):
        if any(mo is None for mo in batch):
            if is_h2d:
                raise ValueError(
                    "MemoryObj is None for some objects in the batch, cannot "
                    f"perform H2D copy. memory_object_batch: {batch}"
                )
            continue

        batch_len = len(batch)
        batch_start_token = start_object_idx * chunk_size
        effective_start = max(batch_start_token, skip_first_n_tokens)
        if effective_start >= batch_start_token + batch_len * chunk_size:
            continue

        launches: list[Any] = []
        for spec_index, kernel_group_id in launch_units:
            blocks_per_chunk, blocks_per_window = blocks_by_kg[kernel_group_id]
            launches.append(
                device_ops.LaunchVar(
                    spec_index,
                    start_object_idx * blocks_per_window,
                    batch_len * blocks_per_window,
                    batch_len,
                    recalculate_blocks_to_skip(
                        blocks_per_chunk,
                        blocks_per_window,
                        cache_context.calculate_num_blocks(
                            effective_start - batch_start_token, kernel_group_id
                        ),
                    ),
                )
            )
        batch_steps.append(
            device_ops.BatchStep(
                build_staging_copies(
                    batch,  # type: ignore[arg-type]
                    object_group_buffers[:batch_len],
                    is_h2d,
                ),
                launches,
            )
        )

    if not batch_steps:
        return
    native.load().execute_object_group_transfer(
        int(direction),
        cache_context.device.index,
        LazyMemoryAllocator.PIN_CHUNK_SIZE,
        specs.specs,
        batch_steps,
        specs.kv_interleaved,
        specs.layer_stride_elems,
        phase_timing_enabled=is_observability_enabled()
        and get_event_bus().has_subscribers(EventType.MP_TRANSFER_PHASE_SAMPLES),
        session_id=transfer_key,
    )


@dataclass(frozen=True)
class _KernelGroupPlan:
    """Retrieve invariants of one kernel group."""

    kernel_group_id: int
    blocks_per_chunk: int
    blocks_per_window: int
    layer_bytes: int
    layer_offsets: list[int]
    layer_indices: list[int]
    depths: list[int]


@dataclass(frozen=True)
class _RunPlan:
    """One launch of a layer batch: a constant-stride run of a kernel group.

    Everything a ``KernelGroupSpec`` needs except the request's block IDs.
    """

    kg_idx: int  # index into LayerwiseRetrievePlan.kernel_groups
    paged_buffer_ptrs: int
    object_ptrs: list[int]  # the run's start in each staging unit
    shape_desc: Any
    lmcache_chunk_size: int
    engine_kv_format: Any


@dataclass(frozen=True)
class _LayerBatchPlan:
    """Retrieve invariants of one signalled layer batch.

    Attributes:
        layer_indices: Registration indices of the KV caches it fills.
        run_offset: Byte offset of its contiguous range in each object.
        run_bytes: Byte length of that range: one staging unit.
        unit_ptrs: Device address of each staging unit; one pass copies at
            most this many objects.
        runs: Its launches, in spec order.
        kv_interleaved: Per run, the layout ``_C`` takes beside the specs.
        layer_stride_elems: Per run, the layer stride ``_C`` takes.
    """

    layer_indices: list[int]
    run_offset: int
    run_bytes: int
    unit_ptrs: list[int]
    runs: list[_RunPlan]
    kv_interleaved: list[bool]
    layer_stride_elems: list[int]


@dataclass(frozen=True)
class LayerwiseRetrievePlan:
    """Retrieve invariants of one object group, built once per context.

    Attributes:
        kernel_groups: Per kernel group of the object group.
        layers: ``(kernel_groups index, local_layer_idx)`` per layer, in
            object order.
        layer_batch: Model depths per signalled batch.
        batches: The signalled layer batches, in object order.
    """

    kernel_groups: list[_KernelGroupPlan]
    layers: list[tuple[int, int]]
    layer_batch: int
    batches: list[_LayerBatchPlan]


def build_layerwise_retrieve_plan(
    cache_context: BaseCacheContext,
    object_group_id: int,
    model_depths: Sequence[int] | None,
    layer_batch: int,
) -> LayerwiseRetrievePlan:
    """Resolve the retrieve invariants of one object group.

    Args:
        cache_context: A layer-wise cache context.
        object_group_id: Which object group.
        model_depths: See ``layer_major_plan.kernel_group_depths``.
        layer_batch: Model depths per signalled batch.

    Raises:
        ValueError: For a layer batch that is not one contiguous range or
            does not fit a staging slot, or a layer stride that is not a
            whole number of elements.
    """
    kv_groups_manager = cache_context.kv_layer_groups_manager
    kernel_groups: list[_KernelGroupPlan] = []
    for kg in kv_groups_manager.object_groups[object_group_id].kernel_group_indices:
        group = kv_groups_manager.kernel_groups[kg]
        blocks_per_chunk, blocks_per_window = _blocks_per_chunk_and_window(
            cache_context, kg
        )
        layer = cache_context.get_temp_layer_buffer(0, kg, 0)  # type: ignore[attr-defined]
        kernel_groups.append(
            _KernelGroupPlan(
                kernel_group_id=kg,
                blocks_per_chunk=blocks_per_chunk,
                blocks_per_window=blocks_per_window,
                layer_bytes=layer.numel() * layer.element_size(),
                layer_offsets=[
                    cache_context.get_layer_offset_in_object(  # type: ignore[attr-defined]
                        object_group_id, kg, local
                    )
                    for local in range(group.num_layers)
                ],
                layer_indices=list(group.layer_indices),
                depths=kernel_group_depths(kv_groups_manager, kg, model_depths),
            )
        )
    layers = [
        (i, local)
        for i, plan in enumerate(kernel_groups)
        for local in range(len(plan.layer_indices))
    ]
    # The object order: by model depth, ties broken by kernel group.
    layers.sort(key=lambda e: (kernel_groups[e[0]].depths[e[1]], e[0]))
    if debug.ENABLED:
        # A layer batch is copied per kernel group, so the kernel-group runs
        # in object order bound how few launches a batch can take.
        logger.info(
            "layerwise[group %d]: %d KV cache(s) in %d kernel group(s), "
            "%d bytes per object; kernel-group runs in object order "
            "(kernel group, caches): %s",
            object_group_id,
            len(layers),
            len(kernel_groups),
            sum(p.layer_bytes * len(p.layer_indices) for p in kernel_groups),
            [
                (kernel_groups[i].kernel_group_id, len(list(run)))
                for i, run in itertools.groupby(e[0] for e in layers)
            ],
        )
        for p in kernel_groups:
            logger.info(
                "layerwise[group %d]:   kernel group %d: %d cache(s) of %d "
                "bytes, registration indices %s, depths %s",
                object_group_id,
                p.kernel_group_id,
                len(p.layer_indices),
                p.layer_bytes,
                p.layer_indices,
                p.depths,
            )
    return LayerwiseRetrievePlan(
        kernel_groups,
        layers,
        layer_batch,
        _build_layer_batches(
            cache_context, object_group_id, kernel_groups, layers, layer_batch
        ),
    )


def _build_layer_batches(
    cache_context: BaseCacheContext,
    object_group_id: int,
    kernel_groups: list[_KernelGroupPlan],
    layers: list[tuple[int, int]],
    layer_batch: int,
) -> list[_LayerBatchPlan]:
    """Split an object group into layer batches and resolve their launches.

    Each batch covers ``layer_batch`` whole model depths: their bytes are one
    contiguous range of every object. Every object of a batch shares one
    staging unit layout: the object-group slots are used as a flat arena
    packed with as many units as fit.
    """
    max_batch_size = cache_context.max_batch_size
    probe = cache_context.get_temp_object_group_buffer(0, object_group_id)
    slot_bytes = probe.numel() * probe.element_size()
    slot_base_ptrs = [
        cache_context.get_temp_object_group_buffer(slot, object_group_id).data_ptr()
        for slot in range(max_batch_size)
    ]

    batches: list[_LayerBatchPlan] = []
    batch_start = 0
    while batch_start < len(layers):
        batch_end = batch_start
        depths: set[int] = set()
        while batch_end < len(layers):
            kg_idx, local = layers[batch_end]
            depth = kernel_groups[kg_idx].depths[local]
            if depth not in depths and len(depths) == layer_batch:
                break
            depths.add(depth)
            batch_end += 1
        entries = layers[batch_start:batch_end]

        def offset_of(entry: tuple[int, int]) -> int:
            return kernel_groups[entry[0]].layer_offsets[entry[1]]

        run_offset = offset_of(entries[0])
        run_bytes = sum(kernel_groups[kg].layer_bytes for kg, _ in entries)
        last = entries[-1]
        run_end = offset_of(last) + kernel_groups[last[0]].layer_bytes
        if run_end != run_offset + run_bytes:
            raise ValueError(
                f"Layer batch {entries[0]}..{last} of object group "
                f"{object_group_id} is not one contiguous range."
            )
        if run_bytes > slot_bytes:
            raise ValueError(
                f"A batch of {len(depths)} layer(s) needs {run_bytes} staging "
                f"bytes, more than the {slot_bytes}-byte object-group slot."
            )

        units_per_slot = slot_bytes // run_bytes
        unit_ptrs = [
            _unit_ptr(
                slot_base_ptrs,
                unit,
                unit_bytes=run_bytes,
                units_per_slot=units_per_slot,
            )
            for unit in range(units_per_slot * max_batch_size)
        ]

        runs: list[_RunPlan] = []
        layer_stride_elems: list[int] = []
        by_kg: dict[int, list[int]] = {}
        for kg_idx, local in entries:
            by_kg.setdefault(kg_idx, []).append(local)
        for kg_idx, locals_ in by_kg.items():
            kg_plan = kernel_groups[kg_idx]
            kg = kg_plan.kernel_group_id
            kv_ptrs = cache_context.get_kernel_group_kv_pointers(kg)
            shape_desc = cache_context.get_shape_desc(kg)
            # A run reads consecutive KV caches of the group, so it may only
            # span consecutive local layers.
            segments: list[list[int]] = []
            for local in sorted(locals_):
                if segments and segments[-1][-1] + 1 == local:
                    segments[-1].append(local)
                else:
                    segments.append([local])
            for segment in segments:
                offsets = [kg_plan.layer_offsets[local] for local in segment]
                for start, run_len, stride_bytes in uniform_stride_runs(offsets):
                    if stride_bytes % shape_desc.element_size != 0:
                        raise ValueError(
                            f"Layer stride {stride_bytes} B is not a whole "
                            f"number of {shape_desc.element_size}-byte elements"
                        )
                    first_local = segment[start]
                    rel = offsets[start] - run_offset
                    runs.append(
                        _RunPlan(
                            kg_idx=kg_idx,
                            paged_buffer_ptrs=kv_ptrs[
                                first_local : first_local + run_len
                            ].data_ptr(),
                            object_ptrs=[ptr + rel for ptr in unit_ptrs],
                            shape_desc=build_run_shape_desc(shape_desc, run_len),
                            lmcache_chunk_size=cache_context.get_slots_per_chunk_in_sw(
                                kg
                            ),
                            engine_kv_format=cache_context.get_engine_kv_format(kg),
                        )
                    )
                    layer_stride_elems.append(stride_bytes // shape_desc.element_size)

        batches.append(
            _LayerBatchPlan(
                layer_indices=[
                    kernel_groups[kg_idx].layer_indices[local]
                    for kg_idx, local in entries
                ],
                run_offset=run_offset,
                run_bytes=run_bytes,
                unit_ptrs=unit_ptrs,
                runs=runs,
                kv_interleaved=[True] * len(runs),
                layer_stride_elems=layer_stride_elems,
            )
        )
        batch_start = batch_end
    return batches


def _unit_ptr(
    slot_base_ptrs: Sequence[int],
    unit_idx: int,
    *,
    unit_bytes: int,
    units_per_slot: int,
) -> int:
    """Device address of one staging unit; slots are separate allocations."""
    slot, sub = divmod(unit_idx, units_per_slot)
    return slot_base_ptrs[slot] + sub * unit_bytes


def _execute_layerwise_batches(
    ops: Any, cache_context: BaseCacheContext, batch_args: list[tuple[Any, ...]]
) -> float:
    """Enqueue ``batch_args`` in one native call; return its seconds."""
    started = time.monotonic()
    ops.execute_layerwise_batches(
        int(lmcache_native.TransferDirection.H2D),
        cache_context.device.index,
        LazyMemoryAllocator.PIN_CHUNK_SIZE,
        batch_args,
    )
    return time.monotonic() - started


def transfer_kv_layerwise(
    cache_context: BaseCacheContext,
    block_ids_gpu: list[torch.Tensor],
    memory_objs: Sequence[MemoryObj | None],
    *,
    object_group_id: int,
    batch_size: int,
    skip_first_n_tokens: int,
    plan: LayerwiseRetrievePlan,
    signal: LayerSignal,
    enqueue_per_batch: bool = False,
) -> None:
    """Retrieve one object group a batch of model layers at a time.

    Each of the plan's layer batches is copied for all objects and scattered
    into the KV cache, followed on the stream by the batch's ``signal`` event.
    A consumer waiting on that event can then start a layer before the
    remaining ones have landed. All batches are enqueued in one native call,
    then ``signal`` announces each in order; with ``enqueue_per_batch`` each
    batch is enqueued and announced as soon as it is built. Objects that do
    not fit the staging arena run in further passes.

    Args:
        cache_context: A layer-wise cache context.
        block_ids_gpu: GPU block IDs, indexed by kernel group.
        memory_objs: Objects to copy; None only before a sliding window.
        object_group_id: Which object group.
        batch_size: Unused; the arena decides how many objects a pass holds.
        skip_first_n_tokens: Tokens to skip writing at the start.
        plan: This object group's :func:`build_layerwise_retrieve_plan`.
        signal: See :data:`LayerSignal`.
        enqueue_per_batch: Make one native call per batch instead of one for
            the whole object group.

    Raises:
        NotImplementedError: For objects not backed by the lazy allocator.
        ValueError: For a None object inside the window.
    """
    del batch_size
    chunk_size = cache_context.lmcache_tokens_per_chunk
    skip = _num_objects_to_skip(
        cache_context, object_group_id, len(memory_objs), is_h2d=True
    )
    chunks: list[tuple[int, MemoryObj]] = []
    for idx in range(skip, len(memory_objs)):
        mo = memory_objs[idx]
        if mo is None:
            raise ValueError(
                f"MemoryObj {idx} is None during the layer-wise H2D transfer."
            )
        if not isinstance(mo.parent(), LazyMemoryAllocator):
            raise NotImplementedError(
                "Layer-wise retrieve needs objects from the lazy allocator."
            )
        # Objects wholly before skip_first_n_tokens have nothing to write.
        if (idx + 1) * chunk_size > skip_first_n_tokens:
            chunks.append((idx, mo))
    if not chunks or not plan.batches:
        return

    base_obj_idx = chunks[0][0]
    first_start_token = base_obj_idx * chunk_size
    skip_tokens = max(first_start_token, skip_first_n_tokens) - first_start_token

    # Request-scoped: this request's staged block IDs, per kernel group.
    block_args: list[tuple[int, int]] = []
    skip_blocks: list[int] = []
    for kg_plan in plan.kernel_groups:
        kg = kg_plan.kernel_group_id
        bpw = kg_plan.blocks_per_window
        block_ids = block_ids_gpu[kg][base_obj_idx * bpw : (chunks[-1][0] + 1) * bpw]
        block_args.append((block_ids.data_ptr(), block_ids.numel()))
        skip_blocks.append(
            recalculate_blocks_to_skip(
                kg_plan.blocks_per_chunk,
                bpw,
                cache_context.calculate_num_blocks(skip_tokens, kg),
            )
        )

    ops = native.load()
    started = time.monotonic()
    num_launches = num_passes = 0
    native_seconds = 0.0
    batch_args: list[tuple[Any, ...]] = []
    for batch in plan.batches:
        max_per_pass = len(batch.unit_ptrs)
        units_needed = min(max_per_pass, len(chunks))
        specs = [
            device_ops.KernelGroupSpec(
                run.paged_buffer_ptrs,
                run.object_ptrs[:units_needed],
                run.shape_desc,
                run.lmcache_chunk_size,
                run.engine_kv_format,
                *block_args[run.kg_idx],
            )
            for run in batch.runs
        ]

        steps: list[Any] = []
        for pass_start in range(0, len(chunks), max_per_pass):
            pass_chunks = chunks[pass_start : pass_start + max_per_pass]
            first_obj_idx = pass_chunks[0][0]
            num_objs = pass_chunks[-1][0] + 1 - first_obj_idx
            launches = []
            for spec_index, run in enumerate(batch.runs):
                bpw = plan.kernel_groups[run.kg_idx].blocks_per_window
                launches.append(
                    device_ops.LaunchVar(
                        spec_index,
                        (first_obj_idx - base_obj_idx) * bpw,
                        num_objs * bpw,
                        len(pass_chunks),
                        skip_blocks[run.kg_idx] if pass_start == 0 else 0,
                    )
                )
            steps.append(
                device_ops.BatchStep(
                    [
                        build_staging_copy_range(
                            mo,
                            batch.unit_ptrs[i],
                            batch.run_offset,
                            batch.run_bytes,
                            True,
                        )
                        for i, (_, mo) in enumerate(pass_chunks)
                    ],
                    launches,
                )
            )
        batch_args.append(
            (
                specs,
                steps,
                batch.kv_interleaved,
                batch.layer_stride_elems,
                signal.event(batch.layer_indices),
            )
        )
        num_launches += len(batch.runs) * len(steps)
        num_passes += len(steps)
        if enqueue_per_batch:
            native_seconds += _execute_layerwise_batches(ops, cache_context, batch_args)
            signal(batch.layer_indices, recorded=batch_args[0][-1] != 0)
            batch_args.clear()

    if batch_args:
        native_seconds += _execute_layerwise_batches(ops, cache_context, batch_args)
        for batch, args in zip(plan.batches, batch_args, strict=True):
            signal(batch.layer_indices, recorded=args[-1] != 0)

    if debug.ENABLED:
        logger.info(
            "layerwise[group %d]: %d object(s) in %d batch(es) of up to %d "
            "depth(s), %d launch(es) over %d pass(es), enqueued in %.2f ms "
            "(%.2f ms in the native executor)",
            object_group_id,
            len(chunks),
            len(plan.batches),
            plan.layer_batch,
            num_launches,
            num_passes,
            (time.monotonic() - started) * 1e3,
            native_seconds * 1e3,
        )
