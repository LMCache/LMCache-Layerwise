# SPDX-License-Identifier: Apache-2.0
"""GPU tests for ``lmcache_layerwise._C``.

The expected LMCache objects are computed with plain torch indexing, not with
LMCache's kernel, so the layouts are checked independently of the code under
test.
"""

# Standard
import importlib.util

# Third Party
import pytest
import torch

NL, NB, BS, NH, HS = 4, 128, 16, 8, 128
D = NH * HS
CHUNK = 256  # tokens per LMCache object = 16 blocks
NUM_OBJECTS = 6  # more than LMCache's in-register MemoryObj4 holds
BY_VALUE_OBJECTS = 3  # few enough to be passed in a MemoryObj4
TIGHT_LAYER = 2 * CHUNK * D  # elements in one layer's K and V
HOST_ALIGNMENT = 1 << 26
SENTINEL = -3.0

LAYOUTS = {
    "2ltd": (False, 0),
    "l2td": (True, 0),
    # Room for a second kernel group's layer between this group's layers.
    "l2td_strided": (True, 2 * TIGHT_LAYER),
}

pytestmark = pytest.mark.cuda


@pytest.fixture(scope="module")
def ops():
    if not torch.cuda.is_available():
        pytest.skip("Requires CUDA")
    pytest.importorskip("lmcache.cuda_ops", reason="Requires lmcache.cuda_ops")
    if importlib.util.find_spec("lmcache_layerwise._C") is None:
        pytest.skip("lmcache_layerwise._C is not built")
    # First Party
    from lmcache_layerwise import native

    return native.load()


def _directions():
    # First Party
    import lmcache.lmcache_native as lmcache_native

    return (
        int(lmcache_native.TransferDirection.D2H),
        int(lmcache_native.TransferDirection.H2D),
    )


def _group(paged_ptrs, objects, block_ids):
    # First Party
    import lmcache.lmcache_native as lmcache_native

    shape_desc = lmcache_native.PageBufferShapeDesc()
    shape_desc.kv_size = 2
    shape_desc.nl = NL
    shape_desc.nb = NB
    shape_desc.bs = BS
    shape_desc.nh = NH
    shape_desc.hs = HS
    shape_desc.element_size = 2
    shape_desc.block_stride_elems = 0
    return lmcache_native.KernelGroupSpec(
        paged_ptrs.data_ptr(),
        [o.data_ptr() for o in objects],
        shape_desc,
        CHUNK,
        lmcache_native.EngineKVFormat.NL_X_TWO_NB_BS_NH_HS,
        block_ids.data_ptr(),
        block_ids.numel(),
    )


def _launch(group_idx, block_ids, num_objects=NUM_OBJECTS):
    # First Party
    import lmcache.cuda_ops as cuda_ops

    return cuda_ops.LaunchVar(group_idx, 0, block_ids.numel(), num_objects, 0)


def _step(launches):
    # First Party
    import lmcache.cuda_ops as cuda_ops

    return cuda_ops.BatchStep([], launches)


def _paged(device):
    layers = [
        torch.randn(2, NB, BS, NH, HS, dtype=torch.bfloat16, device=device)
        for _ in range(NL)
    ]
    ptrs = torch.tensor([t.data_ptr() for t in layers], device=device)
    return layers, ptrs


def _objects(device, layer_stride, num_objects=NUM_OBJECTS):
    numel = NL * layer_stride if layer_stride else NL * TIGHT_LAYER
    return [
        torch.full((numel,), SENTINEL, dtype=torch.bfloat16, device=device)
        for _ in range(num_objects)
    ]


def _expected(paged, block_ids):
    """Objects in 2LTD order: [num_objects, 2, NL, CHUNK, D]."""
    num_objects = block_ids.numel() * BS // CHUNK
    per_layer = [t[:, block_ids].reshape(2, num_objects, CHUNK, D) for t in paged]
    return torch.stack(per_layer, dim=2).transpose(0, 1)


def _as_2ltd(buf, kv_interleaved, layer_stride):
    """Read an object back as [2, NL, CHUNK, D] for the given layout."""
    if not kv_interleaved:
        return buf.view(2, NL, CHUNK, D)
    stride = layer_stride or TIGHT_LAYER
    layers = [
        buf[layer * stride : layer * stride + TIGHT_LAYER].view(2, CHUNK, D)
        for layer in range(NL)
    ]
    return torch.stack(layers).transpose(0, 1)


def _gaps(buf, layer_stride):
    if not layer_stride:
        return buf[:0]
    return buf.view(NL, layer_stride)[:, TIGHT_LAYER:]


@pytest.mark.parametrize(
    "num_objects",
    [BY_VALUE_OBJECTS, NUM_OBJECTS],
    ids=["by_value", "uploaded"],
)
@pytest.mark.parametrize("layout", list(LAYOUTS))
def test_store_and_load_round_trip(ops, layout, num_objects):
    kv_interleaved, layer_stride = LAYOUTS[layout]
    d2h, h2d = _directions()
    device = torch.device("cuda", torch.cuda.current_device())
    torch.manual_seed(0)
    paged, paged_ptrs = _paged(device)
    block_ids = torch.randperm(NB, device=device)[: num_objects * CHUNK // BS]
    objects = _objects(device, layer_stride, num_objects)

    ops.execute_object_group_transfer(
        d2h,
        device.index,
        HOST_ALIGNMENT,
        [_group(paged_ptrs, objects, block_ids)],
        [_step([_launch(0, block_ids, num_objects)])],
        [kv_interleaved],
        [layer_stride],
    )
    torch.cuda.synchronize(device)

    expected = _expected(paged, block_ids)
    for i, obj in enumerate(objects):
        got = _as_2ltd(obj, kv_interleaved, layer_stride)
        assert torch.equal(got, expected[i]), f"object {i}"
        assert torch.all(_gaps(obj, layer_stride) == SENTINEL), f"object {i}"

    restored = [torch.zeros_like(t) for t in paged]
    restored_ptrs = torch.tensor([t.data_ptr() for t in restored], device=device)
    ops.execute_object_group_transfer(
        h2d,
        device.index,
        HOST_ALIGNMENT,
        [_group(restored_ptrs, objects, block_ids)],
        [_step([_launch(0, block_ids, num_objects)])],
        [kv_interleaved],
        [layer_stride],
    )
    torch.cuda.synchronize(device)
    for got, want in zip(restored, paged, strict=True):
        assert torch.equal(got[:, block_ids], want[:, block_ids])


def test_each_kernel_group_uses_its_own_layout(ops):
    d2h, _ = _directions()
    device = torch.device("cuda", torch.cuda.current_device())
    torch.manual_seed(1)
    paged, paged_ptrs = _paged(device)
    block_ids = torch.randperm(NB, device=device)[: NUM_OBJECTS * CHUNK // BS]
    names = ["2ltd", "l2td_strided"]
    objects = [_objects(device, LAYOUTS[name][1]) for name in names]

    ops.execute_object_group_transfer(
        d2h,
        device.index,
        HOST_ALIGNMENT,
        [_group(paged_ptrs, objs, block_ids) for objs in objects],
        [_step([_launch(0, block_ids), _launch(1, block_ids)])],
        [LAYOUTS[name][0] for name in names],
        [LAYOUTS[name][1] for name in names],
    )
    torch.cuda.synchronize(device)

    expected = _expected(paged, block_ids)
    for name, objs in zip(names, objects, strict=True):
        kv_interleaved, layer_stride = LAYOUTS[name]
        for i, obj in enumerate(objs):
            got = _as_2ltd(obj, kv_interleaved, layer_stride)
            assert torch.equal(got, expected[i]), f"{name} object {i}"


@pytest.mark.parametrize(
    ("kv_interleaved", "layer_stride", "match"),
    [
        ([False, False], [0, 0], "one entry per kernel group"),
        ([False], [TIGHT_LAYER], "only supported with kv_interleaved"),
        ([True], [-1], "must be >= 0"),
    ],
)
def test_rejects_bad_layouts(ops, kv_interleaved, layer_stride, match):
    d2h, _ = _directions()
    device = torch.device("cuda", torch.cuda.current_device())
    _, paged_ptrs = _paged(device)
    block_ids = torch.arange(NUM_OBJECTS * CHUNK // BS, device=device)
    objects = _objects(device, 0)

    with pytest.raises(RuntimeError, match=match):
        ops.execute_object_group_transfer(
            d2h,
            device.index,
            HOST_ALIGNMENT,
            [_group(paged_ptrs, objects, block_ids)],
            [_step([_launch(0, block_ids)])],
            kv_interleaved,
            layer_stride,
        )
