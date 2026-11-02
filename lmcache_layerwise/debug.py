# SPDX-License-Identifier: Apache-2.0
"""``LMCACHE_LAYERWISE_DEBUG=1`` logs the layer-wise retrieve's structure and
timings, on the server and the worker. Off by default."""

# Standard
import os

ENABLED = os.getenv("LMCACHE_LAYERWISE_DEBUG", "0").lower() not in (
    "0",
    "",
    "false",
    "no",
)
