# SPDX-License-Identifier: Apache-2.0
"""Tests for ``object_group_transfer.build_staging_copy_range``."""

# Standard
from types import SimpleNamespace

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.gpu_connector import gpu_ops

# Local
from lmcache_layerwise.v1.multiprocess import object_group_transfer

OBJ_BYTES = 64
# Allocator offset just below a pinned-chunk boundary, so a range that is
# placed by pointer but not by offset would split at the wrong byte.
ADDRESS = (3 << 26) - 16


@pytest.fixture(autouse=True)
def _record_staging_copies(monkeypatch):
    """Replace the native ``StagingCopy`` with one that returns its fields."""
    recorder = SimpleNamespace(
        StagingCopy=lambda dest, src, nbytes, host_offset: (
            dest,
            src,
            nbytes,
            host_offset,
        )
    )
    monkeypatch.setattr(object_group_transfer, "device_ops", recorder)
    monkeypatch.setattr(gpu_ops, "device_ops", recorder)


def _memory_obj(raw_tensor=None):
    if raw_tensor is None:
        raw_tensor = torch.zeros(OBJ_BYTES, dtype=torch.uint8)
    return SimpleNamespace(
        raw_tensor=raw_tensor,
        get_size=lambda: OBJ_BYTES,
        data_ptr=raw_tensor.data_ptr(),
        meta=SimpleNamespace(address=ADDRESS),
    )


@pytest.mark.parametrize("is_h2d", [True, False])
def test_range_advances_host_pointer_and_offset_together(is_h2d):
    obj = _memory_obj()
    gpu_ptr = 0x7000_0000

    copy = object_group_transfer.build_staging_copy_range(obj, gpu_ptr, 24, 16, is_h2d)

    host = obj.data_ptr + 24
    expected = (gpu_ptr, host) if is_h2d else (host, gpu_ptr)
    assert copy == (*expected, 16, ADDRESS + 24)


@pytest.mark.parametrize("is_h2d", [True, False])
def test_full_range_matches_build_staging_copies(is_h2d):
    obj = _memory_obj()
    gpu_buffer = torch.zeros(OBJ_BYTES, dtype=torch.uint8)

    copy = object_group_transfer.build_staging_copy_range(
        obj, gpu_buffer.data_ptr(), 0, OBJ_BYTES, is_h2d
    )

    assert [copy] == gpu_ops.build_staging_copies([obj], [gpu_buffer], is_h2d)


@pytest.mark.parametrize(
    ("offset", "nbytes"),
    [(-1, 8), (0, 0), (0, -8), (60, 8), (OBJ_BYTES, 1)],
)
def test_rejects_ranges_outside_the_object(offset, nbytes):
    with pytest.raises(ValueError, match="Invalid range"):
        object_group_transfer.build_staging_copy_range(
            _memory_obj(), 0x1000, offset, nbytes, True
        )


def test_rejects_unallocated_object():
    obj = _memory_obj()
    obj.raw_tensor = None

    with pytest.raises(ValueError, match="raw_tensor is None"):
        object_group_transfer.build_staging_copy_range(obj, 0x1000, 0, 8, True)
