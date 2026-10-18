# SPDX-License-Identifier: Apache-2.0
"""Builds ``lmcache_layerwise._C`` against the headers LMCache installs.

pip install --no-build-isolation -e .
"""

# Third Party
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import lmcache

setup(
    ext_modules=[
        CUDAExtension(
            "lmcache_layerwise._C",
            sources=["csrc/layerwise_ops.cu"],
            include_dirs=[lmcache.get_include()],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
