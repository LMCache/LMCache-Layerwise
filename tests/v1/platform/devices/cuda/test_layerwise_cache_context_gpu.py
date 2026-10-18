# SPDX-License-Identifier: Apache-2.0
"""GPU round trip through :class:`GPULayerwiseCacheContext`.

A store writes layer-major objects from the paged KV cache; a layer-wise
retrieve reads them back into a zeroed cache. The pages must come back
bit-exact, each batch must be signalled once in model-depth order, and a
layer's bytes in an object must land in that layer only.
"""

# Standard
import importlib.util
import random

# Third Party
import pytest
import torch

# First Party
from lmcache import torch_device_type

pytest.importorskip("cupy", reason="GPU cache context tests require cupy")
pytest.importorskip("lmcache.cuda_ops", reason="Requires lmcache.cuda_ops")
if importlib.util.find_spec("lmcache_layerwise._C") is None:
    pytest.skip("lmcache_layerwise._C is not built", allow_module_level=True)

# First Party
from lmcache.v1.memory_allocators.lazy_memory_allocator import (  # noqa: E402
    LazyMemoryAllocator,
)
from lmcache.v1.memory_management import MemoryObj  # noqa: E402
from lmcache.v1.multiprocess.object_group_transfer import (  # noqa: E402
    downsample_and_stage_block_ids,
)
from lmcache_layerwise import debug  # noqa: E402
from lmcache_layerwise.v1.multiprocess import (  # noqa: E402
    object_group_transfer,
)
from lmcache_layerwise.v1.platform.devices.cuda.cache_context import (  # noqa: E402
    GPULayerwiseCacheContext,
)
import lmcache.lmcache_native as lmcache_native  # noqa: E402

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(torch_device_type != "cuda", reason="Requires CUDA"),
]

H2D = lmcache_native.TransferDirection.H2D
D2H = lmcache_native.TransferDirection.D2H
_DEVICE = torch.device("cuda:0")
_DTYPE = torch.bfloat16
_NUM_LAYERS = 4
_SHAPE = [2, 128, 16, 8, 128]  # NHD flash-attention page, 16-token blocks
_CHUNK = 256
_NUM_CHUNKS = 6


class _FakeIPCWrapper:
    """Hands the context a local CUDA tensor (same-process CUDA IPC cannot
    reimport its own handle)."""

    def __init__(self, tensor: torch.Tensor) -> None:
        self._tensor = tensor

    def to_tensor(self) -> torch.Tensor:
        return self._tensor

    def close(self) -> None:
        return None


class _Recorder:
    def __init__(self) -> None:
        self.batches: list[list[int]] = []

    def event(self, layers: list[int]) -> int:
        return 0

    def __call__(self, layers: list[int], *, recorded: bool = False) -> None:
        self.batches.append(list(layers))


@pytest.fixture(autouse=True)
def _seed() -> None:
    random.seed(11)
    torch.manual_seed(11)


@pytest.fixture
def allocator():
    alloc = LazyMemoryAllocator(init_size=256 << 20, final_size=256 << 20)
    yield alloc
    alloc.close()


def _make_context(
    model_depths: list[int] | None,
    layer_batch: int,
    recorder: _Recorder,
    enqueue_per_batch: bool = False,
) -> tuple[GPULayerwiseCacheContext, list[torch.Tensor], dict]:
    tensors = [
        torch.rand(_SHAPE, dtype=_DTYPE, device=_DEVICE) for _ in range(_NUM_LAYERS)
    ]
    bound: dict = {"signal": None}
    ctx = GPULayerwiseCacheContext(
        [_FakeIPCWrapper(t) for t in tensors],  # type: ignore[arg-type]
        lmcache_tokens_per_chunk=_CHUNK,
        model_depths=model_depths,
        layer_signal=lambda: bound["signal"],
        layer_batch=layer_batch,
        enqueue_per_batch=enqueue_per_batch,
    )
    bound["signal"] = recorder
    return ctx, tensors, bound


def _allocate(allocator: LazyMemoryAllocator, ctx) -> list[MemoryObj]:
    shape, dtype = ctx.get_kernel_group_shape_dtype(_CHUNK, 0)
    objs = []
    for _ in range(_NUM_CHUNKS):
        obj = allocator.allocate(shape, dtype)
        assert obj is not None
        obj.raw_tensor.view(torch.uint8)[: obj.get_size()].zero_()
        objs.append(obj)
    return objs


def _transfer(ctx, block_ids, objs, direction, skip_first_n_tokens=0) -> None:
    with torch.cuda.device(ctx.device), torch.cuda.stream(ctx.stream):
        staged = downsample_and_stage_block_ids(ctx, [list(block_ids)])
        ctx.transfer_object_group(
            staged,
            objs,
            object_group_id=0,
            batch_size=ctx.max_batch_size,
            skip_first_n_tokens=skip_first_n_tokens,
            direction=direction,
            transfer_key="layerwise-test",
        )
    ctx.stream.synchronize()


def _block_ids(ctx) -> list[int]:
    per_chunk = ctx.calculate_num_blocks(_CHUNK, 0)
    return random.sample(range(ctx.num_blocks), _NUM_CHUNKS * per_chunk)


@pytest.mark.parametrize(
    "model_depths,layer_batch,expected",
    [
        (None, 1, [[0], [1], [2], [3]]),
        (None, 3, [[0, 1, 2], [3]]),
        ([3, 2, 1, 0], 1, [[3], [2], [1], [0]]),
        ([1, 0, 3, 2], 2, [[1, 0], [3, 2]]),
    ],
)
@pytest.mark.parametrize("enqueue_per_batch", [False, True])
def test_round_trip(allocator, model_depths, layer_batch, expected, enqueue_per_batch):
    recorder = _Recorder()
    ctx, tensors, _ = _make_context(
        model_depths, layer_batch, recorder, enqueue_per_batch
    )
    try:
        block_ids = _block_ids(ctx)
        objs = _allocate(allocator, ctx)
        _transfer(ctx, block_ids, objs, D2H)
        assert recorder.batches == []

        snapshot = [t.clone() for t in tensors]
        for t in tensors:
            t.zero_()
        _transfer(ctx, block_ids, objs, H2D)

        assert [sorted(b) for b in recorder.batches] == [sorted(b) for b in expected]
        ids = torch.tensor(block_ids, device=_DEVICE)
        for t, snap in zip(tensors, snapshot, strict=True):
            assert torch.equal(
                t[:, ids].view(torch.int16), snap[:, ids].view(torch.int16)
            )
            untouched = torch.ones(t.shape[1], dtype=torch.bool, device=_DEVICE)
            untouched[ids] = False
            assert int(t[:, untouched].view(torch.int16).abs().sum()) == 0
        for obj in objs:
            allocator.free(obj)
    finally:
        ctx.close()


def test_skip_prefix(allocator):
    """Blocks before skip_first_n_tokens are left alone."""
    recorder = _Recorder()
    ctx, tensors, _ = _make_context(None, 2, recorder)
    try:
        block_ids = _block_ids(ctx)
        objs = _allocate(allocator, ctx)
        _transfer(ctx, block_ids, objs, D2H)
        snapshot = [t.clone() for t in tensors]
        for t in tensors:
            t.zero_()
        block_size = _CHUNK // ctx.calculate_num_blocks(_CHUNK, 0)
        skip_tokens = _CHUNK + block_size  # one chunk plus one block
        skip_blocks = skip_tokens // block_size
        _transfer(ctx, block_ids, objs, H2D, skip_first_n_tokens=skip_tokens)

        skipped = torch.tensor(block_ids[:skip_blocks], device=_DEVICE)
        written = torch.tensor(block_ids[skip_blocks:], device=_DEVICE)
        for t, snap in zip(tensors, snapshot, strict=True):
            assert int(t[:, skipped].view(torch.int16).abs().sum()) == 0
            assert torch.equal(
                t[:, written].view(torch.int16), snap[:, written].view(torch.int16)
            )
        for obj in objs:
            allocator.free(obj)
    finally:
        ctx.close()


def test_layer_ranges_are_isolated(allocator):
    """Zeroing one layer's range in every object clears that layer only."""
    recorder = _Recorder()
    ctx, tensors, _ = _make_context([2, 0, 3, 1], 1, recorder)
    try:
        block_ids = _block_ids(ctx)
        objs = _allocate(allocator, ctx)
        _transfer(ctx, block_ids, objs, D2H)
        snapshot = [t.clone() for t in tensors]

        target = 2  # registration index of the cleared layer
        layer_bytes = objs[0].get_size() // _NUM_LAYERS
        offset = ctx.get_layer_offset_in_object(0, 0, target)
        for obj in objs:
            raw = obj.raw_tensor.view(torch.uint8)
            raw[offset : offset + layer_bytes].zero_()

        for t in tensors:
            t.fill_(1.0)
        _transfer(ctx, block_ids, objs, H2D)

        ids = torch.tensor(block_ids, device=_DEVICE)
        for i, (t, snap) in enumerate(zip(tensors, snapshot, strict=True)):
            got = t[:, ids].view(torch.int16)
            if i == target:
                assert int(got.abs().sum()) == 0
            else:
                assert torch.equal(got, snap[:, ids].view(torch.int16))
        for obj in objs:
            allocator.free(obj)
    finally:
        ctx.close()


def test_debug_logs_the_plan_and_each_retrieve(allocator, monkeypatch):
    monkeypatch.setattr(debug, "ENABLED", True)
    logs: list[str] = []
    monkeypatch.setattr(
        object_group_transfer.logger,
        "info",
        lambda msg, *args: logs.append(msg % args),
    )
    ctx, _, _ = _make_context([1, 0, 3, 2], 2, _Recorder())
    try:
        block_ids = _block_ids(ctx)
        objs = _allocate(allocator, ctx)
        _transfer(ctx, block_ids, objs, D2H)
        assert logs == []
        _transfer(ctx, block_ids, objs, H2D)
        _transfer(ctx, block_ids, objs, H2D)
        for obj in objs:
            allocator.free(obj)
    finally:
        ctx.close()
    plan_logs = [line for line in logs if "kernel group" in line]
    retrieve_logs = [line for line in logs if "batch(es)" in line]
    assert len(plan_logs) == 2  # once per context: summary + one kernel group
    assert "4 KV cache(s) in 1 kernel group(s)" in plan_logs[0]
    assert len(retrieve_logs) == 2
    assert f"{_NUM_CHUNKS} object(s) in 2 batch(es) of up to 2" in retrieve_logs[0]
