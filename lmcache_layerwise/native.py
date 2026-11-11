# SPDX-License-Identifier: Apache-2.0
"""Loader for the compiled ``lmcache_layerwise._C`` extension."""

# Standard
from types import ModuleType
import functools
import importlib


@functools.cache
def load() -> ModuleType:
    """Import ``lmcache_layerwise._C``.

    The extension takes LMCache's plan types (``KernelGroupSpec``,
    ``BatchStep``, ...) as arguments and runs plans through the executor that
    ``lmcache.cuda_ops`` exports, so both LMCache modules are imported first:
    they register those types.

    Returns:
        The extension module.

    Raises:
        ImportError: If the extension is not built, or was built against a
            different lmcache native extension API.
    """
    # First Party
    import lmcache.cuda_ops  # noqa: F401
    import lmcache.lmcache_native  # noqa: F401

    return importlib.import_module("lmcache_layerwise._C")
