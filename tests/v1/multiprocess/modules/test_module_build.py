# SPDX-License-Identifier: Apache-2.0
"""The plugin's modules, as LMCache's server loads and builds them."""

# Standard
from types import SimpleNamespace
import json

# Third Party
import pytest

# First Party
from lmcache.v1.multiprocess.engine_module import (
    ModuleBuildContext,
    discover_modules,
    order_modules,
)
from lmcache.v1.multiprocess.ext_server_module import (
    load_server_module_classes,
    parse_server_module_specs,
)
from lmcache.v1.multiprocess.modules.lmcache_driven_transfer import (
    LMCacheDrivenTransferModule,
)
from lmcache.v1.multiprocess.request_handler import iter_request_handlers
from lmcache_layerwise.v1.multiprocess.modules.layerwise_transfer import (
    LayerwiseServerModule,
    LayerwiseTransferModule,
)

SPEC = {
    "module_path": "lmcache_layerwise.v1.multiprocess.modules.layerwise_transfer",
    "config": {"batch_layers": 2},
}


def test_documented_spec_replaces_the_builtin_transfer_module():
    loaded = load_server_module_classes(parse_server_module_specs([json.dumps(SPEC)]))
    assert loaded == [
        (LayerwiseServerModule, SPEC["config"]),
        (LayerwiseTransferModule, SPEC["config"]),
    ]

    # The server drops a built-in whose module_name a configured subclass
    # keeps, and resolves dependencies on it by name.
    assert issubclass(LayerwiseTransferModule, LMCacheDrivenTransferModule)
    assert (
        LayerwiseTransferModule.module_name == LMCacheDrivenTransferModule.module_name
    )
    external = [module_type for module_type, _ in loaded]
    names = {module_type.module_name for module_type in external}
    classes = [
        *external,
        *(cls for cls in discover_modules() if cls.module_name not in names),
    ]
    assert LMCacheDrivenTransferModule not in classes
    order = order_modules([cls for cls in classes if not cls.deferred], set(classes))
    transfer = order.index(LayerwiseTransferModule)
    dependents = [
        cls
        for cls in order
        if LMCacheDrivenTransferModule in cls.module_dependencies
        or LayerwiseTransferModule in cls.module_dependencies
    ]
    assert LayerwiseServerModule in dependents
    assert len(dependents) > 1
    assert all(order.index(cls) > transfer for cls in dependents)


def _build_ctx(config, engine_context=None, mode="lmcache_driven"):
    return ModuleBuildContext(
        engine_context,  # type: ignore[arg-type]
        SimpleNamespace(  # type: ignore[arg-type]
            host="0.0.0.0", enable=[], supported_transfer_mode=mode
        ),
        object(),  # type: ignore[arg-type]
        module_configs={
            LayerwiseTransferModule: config,
            LayerwiseServerModule: config,
        },
    )


@pytest.fixture
def base_init(monkeypatch):
    seen = []
    monkeypatch.setattr(
        LMCacheDrivenTransferModule, "__init__", lambda self, ctx: seen.append(ctx)
    )
    return seen


def test_transfer_module_builds_on_the_server_context(base_init):
    engine_context = object()

    module = LayerwiseTransferModule.create(_build_ctx({}, engine_context))

    assert base_init == [engine_context]
    assert module.layer_batch == 4
    assert module._enqueue_per_batch is False
    assert module._frames.port > 0
    module._frames.close()


def test_batch_layers_comes_from_the_spec_config(base_init):
    module = LayerwiseTransferModule.create(_build_ctx({"batch_layers": 8}))

    assert module.layer_batch == 8
    module._frames.close()  # type: ignore[union-attr]


@pytest.mark.parametrize("port", [-1, 65536, "5556", True])
def test_an_invalid_frames_port_is_refused(base_init, port):
    with pytest.raises(ValueError, match="frames_port"):
        LayerwiseTransferModule.create(_build_ctx({"frames_port": port}))


@pytest.mark.parametrize("value", [0, -1, 1.5, "2", True])
def test_invalid_batch_layers_is_refused(base_init, value):
    with pytest.raises(ValueError, match="batch_layers"):
        LayerwiseTransferModule.create(_build_ctx({"batch_layers": value}))


def test_enqueue_per_batch_comes_from_the_spec_config(base_init):
    module = LayerwiseTransferModule.create(_build_ctx({"enqueue_per_batch": True}))

    assert module._enqueue_per_batch is True
    module._frames.close()  # type: ignore[union-attr]


@pytest.mark.parametrize("value", [1, "true", None])
def test_invalid_enqueue_per_batch_is_refused(base_init, value):
    with pytest.raises(ValueError, match="enqueue_per_batch"):
        LayerwiseTransferModule.create(_build_ctx({"enqueue_per_batch": value}))


def test_engine_driven_mode_is_refused(base_init):
    with pytest.raises(ValueError, match="supported-transfer-mode"):
        LayerwiseTransferModule.create(_build_ctx({}, mode="engine_driven"))


def test_server_module_serves_the_built_transfer_module(base_init):
    build_ctx = _build_ctx({})
    transfer = LayerwiseTransferModule.create(build_ctx)
    build_ctx.register(LayerwiseTransferModule, transfer)

    module = LayerwiseServerModule.create(build_ctx)

    assert module._transfer is transfer
    transfer._frames.close()  # type: ignore[union-attr]


def _handler_table(module_type: type) -> dict:
    return {h.operation: h.options for h in iter_request_handlers(module_type)}


def test_transfer_module_serves_the_same_rpcs_as_core():
    # An override loses the base method's @request_handler metadata, so an
    # undecorated override of e.g. retrieve would silently drop the RPC.
    assert _handler_table(LayerwiseTransferModule) == _handler_table(
        LMCacheDrivenTransferModule
    )
