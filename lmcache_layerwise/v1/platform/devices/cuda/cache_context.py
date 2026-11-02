# SPDX-License-Identifier: Apache-2.0
"""CUDA cache context that stages KV objects in layer-major order."""

# Standard
from collections.abc import Callable, Sequence
from typing import cast

# Third Party
import torch

# First Party
from lmcache.utils import EngineType
from lmcache.v1.gpu_connector.utils import LayoutHints
from lmcache.v1.kv_layer_groups import KVLayerGroupsManager
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.multiprocess.custom_types import KVCache
from lmcache.v1.multiprocess.group_view import EngineGroupInfo
from lmcache.v1.platform.devices.cuda.cache_context import (
    GPUCacheContext,
    _TempGPUBuffer,
)
import lmcache.lmcache_native as lmcache_native

# First Party
from lmcache_layerwise.v1.multiprocess.layer_major_plan import (
    carve_layer_major_object_group,
    layer_major_placement,
    placement_keeps_kernel_groups_contiguous,
)
from lmcache_layerwise.v1.multiprocess.object_group_transfer import (
    LayerSignal,
    LayerwiseRetrievePlan,
    build_layerwise_retrieve_plan,
    transfer_kv_layer_major,
    transfer_kv_layerwise,
)


class _TempLayerMajorGPUBuffer(_TempGPUBuffer):
    """Staging buffer whose object groups are carved in model-depth order.

    The base carves kernel-group-major; this re-carves the same bytes after
    ``super().__init__``. The buffer size does not depend on the order, since
    both layouts hold the same slices.
    """

    def __init__(
        self,
        kv_layer_groups_manager: KVLayerGroupsManager,
        lmcache_tokens_per_chunk: int,
        device: torch.device,
        max_batch_size: int,
        model_depths: Sequence[int] | None,
    ) -> None:
        super().__init__(
            kv_layer_groups_manager, lmcache_tokens_per_chunk, device, max_batch_size
        )
        # (batch_idx, kernel_group_idx, local_layer_idx) -> (byte offset, size)
        self._offset_map_layer: dict[tuple[int, int, int], tuple[int, int]] = {}
        placements = {
            og: layer_major_placement(
                self._kv_groups_manager,
                og,
                self._get_size_for_kernel_group,
                model_depths,
            )
            for og in range(self._kv_groups_manager.num_object_groups)
        }
        self._kernel_groups_contiguous = all(
            placement_keeps_kernel_groups_contiguous(p) for p in placements.values()
        )

        self._offset_map.clear()
        self._offset_map_kernel_group_only.clear()
        self._offset_map_object_group_only.clear()
        offset = 0
        for batch_idx in range(self._max_batch_size):
            for og, placement in placements.items():
                start = offset
                offset, size = carve_layer_major_object_group(
                    placement, batch_idx, offset, self._offset_map_layer
                )
                self._offset_map_object_group_only[(batch_idx, og)] = (start, size)

    @property
    def kernel_groups_contiguous(self) -> bool:
        """Whether each kernel group is still one contiguous byte range."""
        return self._kernel_groups_contiguous

    def get_temp_kernel_group_buffer(
        self, batch_idx: int, kernel_group_idx: int
    ) -> torch.Tensor:
        """Not available: a group's layers are not where its shape says.

        Even a contiguous group holds ``[L, 2, T, D]`` here rather than the
        ``[2, L, T, D]`` its shape describes.
        """
        raise NotImplementedError(
            "The layer-major staging buffer has no kernel-group view; use "
            "get_temp_layer_buffer()"
        )

    def get_temp_layer_buffer(
        self, batch_idx: int, kernel_group_idx: int, local_layer_idx: int
    ) -> torch.Tensor:
        """Staging slice holding one layer's caches.

        The view keeps the kernel group's shape with the layer axis pinned to
        length one.

        Args:
            batch_idx: Batch slot.
            kernel_group_idx: Kernel group.
            local_layer_idx: Index into the group's ``layer_indices``, not a
                model depth.

        Raises:
            ValueError: If any index is out of range.
        """
        key = (batch_idx, kernel_group_idx, local_layer_idx)
        if key not in self._offset_map_layer:
            raise ValueError(
                f"Invalid batch_idx {batch_idx}, kernel_group_idx "
                f"{kernel_group_idx} or local_layer_idx {local_layer_idx}"
            )
        offset, size = self._offset_map_layer[key]
        shape, dtype = self._shape_cache_kernel_group[kernel_group_idx]
        layer_shape = torch.Size((shape[0], 1, *shape[2:]))
        return self._temp_buffer[offset : offset + size].view(dtype).view(layer_shape)

    def get_layer_offset_in_object(
        self, object_group_idx: int, kernel_group_idx: int, local_layer_idx: int
    ) -> int:
        """Byte offset of one layer's caches from the start of its object.

        Raises:
            ValueError: If the layer is not part of the object group.
        """
        layer = self._offset_map_layer.get((0, kernel_group_idx, local_layer_idx))
        group = self._offset_map_object_group_only.get((0, object_group_idx))
        if (
            layer is None
            or group is None
            or not (group[0] <= layer[0] < group[0] + group[1])
        ):
            raise ValueError(
                f"No staging offset for object_group={object_group_idx}, "
                f"kernel_group={kernel_group_idx}, local_layer={local_layer_idx}"
            )
        return layer[0] - group[0]


class GPULayerwiseCacheContext(GPUCacheContext):
    """CUDA cache context for layer-major objects.

    Stores write layer-major objects; retrieves read them a batch of layers at
    a time and signal each batch through the bound :class:`LayerSignal`.
    Everything else is inherited.

    Args:
        model_depths: Model depth of every registered KV cache, indexed by
            registration order, or None to order by registration.
        layer_signal: Returns the signal of the retrieve running on this
            thread, or None outside one.
        layer_batch: Model layers per signalled batch.
        enqueue_per_batch: Enqueue and signal each batch as soon as it is
            built, instead of an object group's batches in one native call.

    Other arguments are those of :class:`GPUCacheContext`.

    Raises:
        ValueError: For blending (``full_sw_kv``), which addresses whole
            kernel groups of the staging buffer; for ``model_depths`` that do
            not cover every registered cache; or for a geometry the
            layer-major layout cannot address.
    """

    def __init__(
        self,
        kv_caches: KVCache,
        lmcache_tokens_per_chunk: int = 256,
        layout_hints: LayoutHints | None = None,
        engine_group_infos: Sequence[EngineGroupInfo] = (),
        engine_type: EngineType = EngineType.VLLM,
        separate_object_groups: bool = False,
        full_sw_kv: bool = False,
        *,
        model_depths: Sequence[int] | None,
        layer_signal: Callable[[], LayerSignal | None],
        layer_batch: int,
        enqueue_per_batch: bool = False,
    ) -> None:
        if full_sw_kv:
            raise ValueError("Layer-wise transfer does not support blending.")
        if model_depths is not None and len(model_depths) != len(kv_caches):
            raise ValueError(
                f"Got {len(model_depths)} model depths for {len(kv_caches)} "
                f"registered KV caches."
            )
        self._model_depths = model_depths
        self._layer_signal = layer_signal
        self._layer_batch = layer_batch
        self._enqueue_per_batch = enqueue_per_batch
        self._retrieve_plans: dict[int, LayerwiseRetrievePlan] = {}
        super().__init__(
            kv_caches,
            lmcache_tokens_per_chunk,
            layout_hints,
            engine_group_infos,
            engine_type,
            separate_object_groups,
            full_sw_kv,
        )

    @property
    def _layer_buffer(self) -> _TempLayerMajorGPUBuffer:
        return cast(_TempLayerMajorGPUBuffer, self._temp_buffer)

    def _make_temp_buffer(
        self,
        kv_layer_groups_manager: KVLayerGroupsManager,
        lmcache_tokens_per_chunk: int,
        device: torch.device,
        max_batch_size: int,
        full_sw_kv: bool,
    ) -> _TempGPUBuffer:
        return _TempLayerMajorGPUBuffer(
            kv_layer_groups_manager,
            lmcache_tokens_per_chunk,
            device,
            max_batch_size,
            self._model_depths,
        )

    @property
    def model_depths(self) -> Sequence[int] | None:
        """Model depth per registered KV cache, or None if not reported."""
        return self._model_depths

    @property
    def kernel_groups_contiguous(self) -> bool:
        """Whether each kernel group is one contiguous staging range."""
        return self._layer_buffer.kernel_groups_contiguous

    def get_temp_layer_buffer(
        self, batch_idx: int, kernel_group_idx: int, local_layer_idx: int
    ) -> torch.Tensor:
        """See :meth:`_TempLayerMajorGPUBuffer.get_temp_layer_buffer`."""
        return self._layer_buffer.get_temp_layer_buffer(
            batch_idx, kernel_group_idx, local_layer_idx
        )

    def get_layer_offset_in_object(
        self, object_group_idx: int, kernel_group_idx: int, local_layer_idx: int
    ) -> int:
        """See :meth:`_TempLayerMajorGPUBuffer.get_layer_offset_in_object`."""
        return self._layer_buffer.get_layer_offset_in_object(
            object_group_idx, kernel_group_idx, local_layer_idx
        )

    def transfer_object_group(
        self,
        block_ids_gpu: list[torch.Tensor],
        memory_objs: Sequence[MemoryObj | None],
        *,
        object_group_id: int,
        batch_size: int,
        skip_first_n_tokens: int,
        direction: lmcache_native.TransferDirection,
        transfer_key: str,
        block_ids_host: Sequence[Sequence[int]] = (),
    ) -> None:
        """Copy one object group in the layer-major layout.

        Stores copy whole objects. Retrieves copy a batch of layers at a time
        and signal each batch; they must run inside a layer-wise retrieve.
        ``block_ids_host`` only feeds LMCache's direct copy, which moves
        objects in the token-major layout, so it is unused.

        Raises:
            RuntimeError: For a retrieve outside a layer-wise retrieve.
        """
        del block_ids_host
        if direction != lmcache_native.TransferDirection.H2D:
            transfer_kv_layer_major(
                self,
                block_ids_gpu,
                memory_objs,
                object_group_id=object_group_id,
                batch_size=batch_size,
                skip_first_n_tokens=skip_first_n_tokens,
                direction=direction,
                transfer_key=transfer_key,
            )
            return

        signal = self._layer_signal()
        if signal is None:
            raise RuntimeError(
                "Layer-major objects can only be retrieved by a layer-wise retrieve."
            )
        plan = self._retrieve_plans.get(object_group_id)
        if plan is None:
            plan = build_layerwise_retrieve_plan(
                self, object_group_id, self._model_depths, self._layer_batch
            )
            self._retrieve_plans[object_group_id] = plan
        transfer_kv_layerwise(
            self,
            block_ids_gpu,
            memory_objs,
            object_group_id=object_group_id,
            batch_size=batch_size,
            skip_first_n_tokens=skip_first_n_tokens,
            plan=plan,
            signal=signal,
            enqueue_per_batch=self._enqueue_per_batch,
        )
