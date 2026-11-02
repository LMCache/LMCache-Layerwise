# LMCache-Layerwise

Layer-major KV store and layer-wise KV retrieve for LMCache multiprocess (MP)
mode, packaged as an LMCache plugin (`lmcache-layerwise`, import
`lmcache_layerwise`).

A store writes each LMCache object layer-major (all of one model layer's KV
cache bytes together). A retrieve then copies a batch of model layers at a
time and signals each batch with an IPC event, so vLLM can start attention on
the first layers while later ones are still loading.

## What lives here and what LMCache provides

The plugin owns only what is specific to layer-wise:

- `csrc/layerwise_ops.cu`
  - `upload_object_ptrs`: stages the LMCache object pointers in a device
    array, so one launch can cover up to 1024 objects (LMCache's built-in
    launch holds at most 4 in registers).
  - `LayerMajorObjectOffset`: the object-offset policy for the interleaved
    `[L, 2, T, D]` layout, with an optional layer stride for layer-major
    objects that hold several kernel groups per layer.
  - a launch callback that runs LMCache's kernel with those two policies.
- `lmcache_layerwise/v1/multiprocess/`
  - `layer_major_plan.py`: where each layer sits in a layer-major object.
  - `object_group_transfer.py`: the layer-major store and the batched
    layer-wise retrieve.
  - `modules/layerwise_transfer.py`: the transfer module and the
    `layerwise.*` server-module methods, loaded by `--server-module`.
  - `transfer_context/worker_transfer.py`: the worker side and its per-layer
    futures.
- `lmcache_layerwise/v1/platform/`: the layer-major CUDA cache context and the
  IPC event pool.
- `lmcache_layerwise/integration/vllm/`: the vLLM connector and worker adapter.

Everything else is reused from LMCache:

- the transfer kernel and launcher, as header-only templates
  (`cuda/mp_mem_kernels_detail.cuh` under `lmcache.get_include()`);
- the plan executor (staging copies, bounds checks, launch ordering, phase
  timing), called through the `lmcache.cuda_ops._ext_api` function table
  (`cuda/lmcache_ext_api.h`);
- the plan types (`KernelGroupSpec`, `BatchStep`, ...) bound by
  `lmcache.lmcache_native` and `lmcache.cuda_ops`;
- the store and retrieve flow (`LMCacheDrivenTransferModule`), cache context
  (`GPUCacheContext`), worker transfer context and vLLM connector, through
  their extension hooks;
- replacing the server's LMCache-driven transfer module: a configured server
  module that subclasses a built-in one and keeps its `module_name` is built
  in its place;
- the server-module RPCs (`server_module_call`, `server_module_affinity_call`).

See [docs/design.md](docs/design.md) for the design.

## Requirements

- An LMCache build that ships the native extension API (`lmcache.get_include()`
  and `lmcache.cuda_ops._ext_api`), installed with its CUDA extensions.
- CUDA, with an `nvcc` that matches the CUDA version of the installed torch.

## Build

```bash
pip install --no-build-isolation -e .
```

The extension is built against the installed torch and lmcache, and must be
rebuilt when either changes. Loading it against a different LMCache native
extension API raises `ImportError`.

## Use

Server:

```bash
lmcache server ... --server-module '{
  "module_path": "lmcache_layerwise.v1.multiprocess.modules.layerwise_transfer",
  "config": {"batch_layers": 4}
}'
```

`batch_layers` (default 4) is how many model layers each signalled batch
covers. `frames_port` (default 0, a random free port) is the TCP port of the
frame channel (see [Protocol](#protocol)); set it, e.g. `"frames_port": 5556`,
when the worker can reach only known server ports, such as published container
ports. `enqueue_per_batch` (default `false`) enqueues and signals each layer
batch as soon as it is built, instead of an object group's batches in one
native call; it pays off when that call would enqueue many copies, as with a
small `batch_layers`. The plugin needs `--supported-transfer-mode lmcache_driven` or
`auto`. Objects are stored layer-major, so a server runs
either this plugin or LMCache's built-in transfer, not both.

vLLM:

```bash
--kv-transfer-config '{
  "kv_connector": "LMCacheLayerwiseMPConnector",
  "kv_connector_module_path":
    "lmcache_layerwise.integration.vllm.lmcache_mp_connector",
  "kv_role": "kv_both",
  "kv_load_failure_policy": "recompute"
}'
```

Every KV cache name must carry its model depth (`model.layers.<n>`).

Set `LMCACHE_LAYERWISE_DEBUG=1` on both the server and vLLM to log, per
retrieve, the server's batches, launches and enqueue time, and the worker's
layer waits and frame arrival times. vLLM also logs the KV caches with their
depths at registration, and the server each object group's kernel-group runs
on its first retrieve.

## Protocol

| Method | RPC | Purpose |
| --- | --- | --- |
| `layerwise.set_model_depths` | `server_module_call` | Model depth of each KV cache; sent before `register_kv_cache`. |
| `layerwise.register_events` | `server_module_call` | Creates the instance's IPC event pool; returns its handles and the frame channel's port. |
| `layerwise.probe_frames` | `server_module_call` | Publishes a probe frame; the worker repeats it at registration until one arrives. |
| `layerwise.retrieve` | `server_module_affinity_call` | The retrieve; the answer also carries each KV cache's event index. |

While a retrieve runs, the server publishes one frame per layer batch (call
id, event index, KV caches) from the retrieve's own thread, on a ZMQ PUB socket
bound to `frames_port` of the server's `--host`, with one topic per worker
instance. The worker subscribes at the host of its request server URL, so the
frame port must be reachable wherever the request port is. Registration
probes the channel until a probe frame arrives, so the subscription is live
before the first retrieve; if none arrives within the request timeout, the
worker logs a warning and its layer waits fall back to the answers. Frames are
not acknowledged and can still be dropped past 1000 unread; a dropped frame
only delays that layer's wait until the answer, which carries every layer's
event too.

Payloads are msgpack structs in `lmcache_layerwise/v1/multiprocess/protocol.py`.
LMCache's plain `retrieve` is refused, since it cannot read layer-major
objects.

## Tests

```bash
pytest tests
```

Everything under `tests/` runs on CPU except `tests/test_layerwise_ops.py` and
`tests/v1/platform/devices/cuda/test_layerwise_cache_context_gpu.py`, which
need a CUDA device and the built extension, and
`tests/integration/vllm/test_lmcache_mp_connector.py`, which needs vLLM. They
skip otherwise.
