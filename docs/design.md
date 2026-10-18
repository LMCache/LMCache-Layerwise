# LMCache-Layerwise: Design

## 1. Overview

LMCache-Layerwise adds a layer-major KV store and a layer-wise KV retrieve to
LMCache's multiprocess (MP) mode, as a plugin. Every store writes
an LMCache object **layer-major**: ordered by model depth, so the bytes of N
consecutive depths form one contiguous range of the object. A retrieve then
copies N depths at a time for all objects, records one IPC event per batch
and publishes a frame naming it, so vLLM can start attention on the first
layers while later ones are still loading.

`batch_layers` (N, default 4) sets how many model depths a batch covers. For
a model with D depths, each object group of a retrieve takes `ceil(D / N)`
batches, and a batch covers every KV cache those depths own, across all
kernel groups. There is no "off" value: a server that should not load layer
by layer runs LMCache's built-in transfer instead of the plugin.

### 1.1 Supported Scope

- **LMCache-driven transfer only.** `LayerwiseTransferModule.create()` refuses
  `--supported-transfer-mode engine_driven` at startup; `lmcache_driven` and
  `auto` are accepted.
- **CUDA only.** The extension `lmcache_layerwise._C` is built with `nvcc`
  against the installed torch and LMCache, and the cache context subclasses
  LMCache's CUDA `GPUCacheContext`.
- **vLLM layer names.** Every KV cache name must carry its model depth
  (`model.layers.<n>`); registration raises otherwise.
- **Refused at registration or on first use:** blending (`full_sw_kv`), the
  fused cross-layer KV format, a kernel group whose staging bytes do not split
  evenly across its layers, GDS-backed objects, and retrieves of objects not
  backed by the lazy allocator.

---

## 2. Architecture

```
+------------------------------ lmcache server ------------------------------+
|                                                                            |
|  LayerwiseTransferModule  (replaces LMCacheDrivenTransferModule)           |
|    store()               inherited; copies go through the cache context    |
|    retrieve()            refused: objects are layer-major                  |
|    retrieve_layerwise()  LMCache's retrieve + a per-batch signal           |
|    _create_cache_context -> GPULayerwiseCacheContext                       |
|                               transfer_object_group (D2H: layer-major      |
|                               store; H2D: batched retrieve)                |
|                               _make_temp_buffer -> layer-major staging     |
|    EventPool per instance, _FramePublisher (ZMQ PUB, frames_port)          |
|                                                                            |
|  LayerwiseServerModule  (depends on LayerwiseTransferModule)               |
|    layerwise.set_model_depths / register_events / probe_frames             |
|    layerwise.retrieve  (client-affinity thread)                            |
+-------------+--------------------------------------------+-----------------+
              | request transport                          | ZMQ PUB/SUB
              | (server_module_call,                       | one Frame per
              |  server_module_affinity_call)              | layer batch
+-------------+--------------------------------------------+-----------------+
|  LMCacheLayerwiseMPConnector (vLLM)                      |                 |
|    LMCacheLayerwiseMPWorkerAdapter                       |                 |
|      wait_for_layer_load(name) -> depth -> KV cache idx  |                 |
|      LayerwiseTransferContext                            |                 |
|        submit_retrieve -> LayerwiseRetrieveFuture        |                 |
|        frame reader thread  <----------------------------+                 |
|      LayerwiseRetrieveFuture.wait_for_layer(idx)                           |
|        -> EventPool.wait(leader, current stream)                           |
+------------------------------------ vLLM worker ---------------------------+
```

### 2.1 What LMCache Provides

The plugin changes no LMCache behavior; it plugs into these extension points:

| Extension point | LMCache location | Plugin use |
| --- | --- | --- |
| Built-in module override by `module_name` | `lmcache/v1/multiprocess/server.py`, `engine_module.py` | `LayerwiseTransferModule` subclasses `LMCacheDrivenTransferModule` and keeps its `module_name`, so it is built in its place and LMCache modules that depend on it get the plugin instance. The core lock, prefetch, store and retrieve flow is reused as is |
| `LMCacheDrivenTransferModule._create_cache_context` | `lmcache/v1/multiprocess/modules/lmcache_driven_transfer.py` | Returns a `GPULayerwiseCacheContext` |
| `BaseCacheContext.transfer_object_group` | `lmcache/v1/platform/base/cache_context.py` | Every store and retrieve copy goes through it; the plugin overrides both directions |
| `GPUCacheContext._make_temp_buffer` | `lmcache/v1/platform/devices/cuda/cache_context.py` | Builds the layer-major staging buffer |
| `server_module_affinity_call`, `@server_module_handler(..., requires_client_affinity=True)` | `lmcache/v1/multiprocess/transport/base.py`, `ext_server_module.py` | `layerwise.retrieve` runs on the client's affinity thread, like LMCache's own retrieve |
| `LMCacheMPConnector.create_lmcache_mp_worker_adapter`, `LMCacheMPWorkerAdapter._create_transfer_context` | `lmcache/integration/vllm/` | The connector and adapter subclasses |
| Native extension API: `lmcache.get_include()`, `lmcache.cuda_ops._ext_api` | `csrc/cuda/lmcache_ext_api.h`, `csrc/cuda/mp_mem_kernels_detail.cuh` | `_C` instantiates LMCache's transfer kernel with its own policies and runs plans through LMCache's executor |

`server_module_affinity_call` is a regular `RequestClient` RPC, so the
layer-wise retrieve works on any request transport. Frames travel on a
separate ZMQ channel (section 8).

---

## 3. Retrieve Flow

```
worker (vLLM)                               server (affinity thread of the worker)
  submit_retrieve
    call_id = next id, _Call registered
    server_module_affinity_call(
      layerwise.retrieve, RetrieveRequest) ---> LayerwiseTransferModule.retrieve_layerwise
                                                  tls.signal = signal
                                                  LMCacheDrivenTransferModule.retrieve
                                                    (locks, prefetch, wait on the
                                                     worker's event, then per object group:)
                                                    cache_context.transfer_object_group(H2D)
                                                      transfer_kv_layerwise:
                                                        one native call, per batch:
                                                          H2D copy + scatter (enqueue)
                                                          record the leader's pool event
                                                        per batch, signal(layers):
  frame reader: _Call.add(frame)  <------------------      publish Frame(call_id, leader, layers)
                                                        ...
  wait_for_layer_load("model.layers.d...")        signal any layer no batch covered
    depth d -> KV cache idx                       answer RetrieveResponse(
    future.wait_for_layer(idx)                      event_handle, succeeded, leaders)
      leader from frame (or answer)
      pool.wait(leader, current stream)
  attention(d)
```

The leader of a batch is the registration index of its first KV cache; pool
event `leader` is recorded after the batch. `wait_for_layer` skips the stream
wait when the leader equals the one it last waited on, so with N > 1 the
other caches of a batch cost nothing.

Layers no batch covered (aux groups, or a failure midway) are signalled once
LMCache's retrieve returns, so a worker never waits on an event this call
left stale. The answer repeats every KV cache's leader, so a worker that
missed frames can still wait per layer.

### 3.1 Frames Signal Enqueue, Not Completion

Each batch is enqueued, then its leader's pool event recorded on the cache
context's stream, all within one native call per object group; the frames of
the group's batches are published after that call returns. A frame therefore means "this batch is ordered on the
stream", and the worker's `pool.wait` is a stream-ordered wait that does not
block the host. Completion of the whole retrieve is carried by
`RetrieveResponse.event_handle`, LMCache's usual completion event:
`LayerwiseRetrieveFuture.wait()` synchronizes it, `wait_on_stream()` waits on
it.

### 3.2 Synchronous Loads

vLLM runs a request with an asynchronous load only after the whole load
finished, which leaves nothing for per-layer waits to overlap.
`LMCacheLayerwiseMPConnector.get_num_new_matched_tokens` therefore reports
hits as synchronous loads, gated per layer by `wait_for_layer_load`, and the
adapter's `get_finished` reports finished stores only.

Ordering against attention then comes only from `wait_for_layer_load`. Until
vLLM is first seen calling it for a known layer, the adapter waits for each
retrieve in full before handing it back (and warns once), so a model whose
attention never calls it reads no KV in flight.

---

## 4. Layer-Major Object Layout

### 4.1 Layout

LMCache's default object holds each kernel group as `[2, L, T, D]`: all K
layers, then all V layers. A layer-major object is ordered by model depth,
ties broken by kernel group, and holds each layer as `[2, T, D]`:

```
default (kernel-group-major, [2, L, T, D] per group):
  [ kg0: K0 K1 ... K(L-1) V0 V1 ... V(L-1) | kg1: K... V... | ... ]

layer-major (depth order, [L, 2, T, D] per group):
  [ depth 0: kg0 K0 V0 | kg1 K0 V0 | depth 1: kg0 K1 V1 | kg1 K1 V1 | ... ]
```

N consecutive depths are then one contiguous byte range of every object,
which is what lets a retrieve copy and signal one batch before the rest of
the object has landed. The price is that a kernel group is scattered through
the object: its consecutive layers are a whole depth extent apart (every
group's slice at that depth), not packed.

`layer_major_plan.py` is the single source of this layout; the store, the
retrieve and the staging buffer all build their plans from it, since they
must agree byte for byte.

### 4.2 Native Side

`csrc/layerwise_ops.cu` holds only what is specific to layer-wise:

- `LayerMajorObjectOffset`: the object-offset policy. With
  `kv_interleaved == false` it defers to LMCache's `TwoLTDObjectOffset`; with
  `true` it addresses `[L, 2, T, D]`, with consecutive layers
  `layer_stride_elems` elements apart, or packed when that is 0.
- `upload_object_ptrs`: stages object pointers in a device array, so one
  launch covers up to `MAX_OBJECTS_PER_LAUNCH` (1024) objects. LMCache's
  `MemoryObj4` holds 4 in registers; smaller launches still use it.
- `execute_object_group_transfer`: runs a plan through LMCache's executor
  (staging copies, bounds checks, launch ordering, phase timing) via the
  `lmcache_ext` function table, with LMCache's kernel instantiated on these
  two policies.

`PageBufferShapeDesc` has no layout fields, so `LayerMajorSpecs` passes each
spec's `kv_interleaved` and `layer_stride_elems` as parallel lists. A kernel
group costs one spec per maximal constant-stride run of its layers
(`uniform_stride_runs`): a launch walks the object with a single layer stride.

`lmcache_layerwise.native.load()` imports `lmcache.cuda_ops` and
`lmcache.lmcache_native` before `_C`, since they register the plan types `_C`
takes. Loading `_C` against a different native extension API version raises
`ImportError`.

### 4.3 Store

`transfer_kv_layer_major` copies whole objects. Its batch walk mirrors
LMCache's `_run_object_group_transfer_plan` and reuses its helpers
(`batched_iteration_with_skip`, `recalculate_blocks_to_skip`,
`build_staging_copies`); only the specs differ (section 4.2). The store
handler itself is LMCache's, reached through `transfer_object_group`.

### 4.4 Retrieve Batches

`transfer_kv_layerwise` retrieves one object group:

1. `build_layerwise_retrieve_plan` (cached per object group on the context)
   lists every `(kernel group, local layer)` in object order with its byte
   offset, size, registration index and depth.
2. A batch takes consecutive entries until one more would add an
   `N + 1`-th depth. Its bytes `[run_offset, run_offset + run_bytes)` must be
   one contiguous range and fit one object-group staging slot; otherwise the
   retrieve raises rather than copy garbage.
3. The object-group slots are used as a flat arena of `run_bytes` units:
   `slot_bytes // run_bytes` per slot, times `max_batch_size` slots per pass.
   Objects that do not fit run in further passes of the same native call.
4. Each object contributes one `StagingCopy` of the batch's range
   (`build_staging_copy_range` advances the host pointer and the lazy
   allocator offset together, so the copy can be split at pinned-chunk
   boundaries). Specs cover the batch's caches per kernel group, per run of
   consecutive local layers, per constant-stride run.
5. One `execute_layerwise_batches` call per object group enqueues every
   batch's passes, each followed by the batch's leader event
   (`signal.event`, a raw `cudaEvent_t`). `signal` then publishes the frame
   of each batch in order. The call keeps the GIL: the server runs a thread
   per rank, and each Python event record, or a native call per batch,
   would be one more point where a thread drops the GIL and waits to take
   it back.

---

## 5. Staging Buffer

`_TempLayerMajorGPUBuffer` subclasses LMCache's `_TempGPUBuffer` and re-carves
the same tensor after `super().__init__`; the size does not depend on the
order. Slots stay outermost, and inside each object group the slices follow
the layer-major placement:

```
for batch_idx in range(max_batch_size):          # one slot = one object
    for object_group_idx in ...:                 # one memory object
        for (kernel_group_idx, local_layer_idx) in depth order:
            _offset_map_layer[(batch_idx, kg, local)] = (offset, size)
```

- `get_temp_layer_buffer(batch, kg, local)` returns one layer's slice, with
  the kernel group's shape and the layer axis pinned to 1.
- `get_layer_offset_in_object(og, kg, local)` gives a layer's byte offset
  inside its object; the retrieve plan reads it.
- `get_temp_kernel_group_buffer` raises `NotImplementedError`: even a group
  whose layers stay contiguous holds `[L, 2, T, D]` here, not the
  `[2, L, T, D]` its shape describes.
- `kernel_groups_contiguous` reports whether every kernel group is still one
  run in depth order, as with a single kernel group.

A slot spans every object group, so a unit address is
`slot_base(slot) + sub * unit_bytes`, with `slot_base` taken from
`get_temp_object_group_buffer(slot, og)` (`_unit_ptr`). Stepping by unit size
from slot 0 would run into the next object group whenever an object has more
than one.

---

## 6. Two Index Spaces: Model Depth vs Registration Index

A "layer" is two different numbers, and they diverge on exactly the models
that layer-wise loading targets.

| Name | Index space |
| --- | --- |
| `wait_for_layer_load(layer_name)` (vLLM) | model depth, parsed by `_LAYER_RE` |
| `SetModelDepthsRequest.depths` | model depth, indexed by registration index |
| `kernel_groups[kg].layer_indices` | registration index |
| `Frame.layers`, `RetrieveResponse.leaders`, `wait_for_layer(layer)` | registration index |
| `Frame.leader`, event pool index | registration index of the batch's first cache |
| `num_layers` of a kernel group or context | a count of caches, not of depths |

Registration indices are positions in the flat `kv_caches` sequence. They
coincide with depth only when each model layer registers exactly one KV
cache; a model that registers several caches per layer registers them by
cache type, so registration order is not depth order. DeepSeek-V4-Flash
registers 167 caches over 43 depths; Qwen3-32B registers 64 over 64, so the
distinction is invisible there.

The worker therefore sends `layerwise.set_model_depths` (depths parsed from
the cache names) before `register_kv_cache`, and the server orders objects by
those depths. Without them, registration order is used as the depth, which is
right only for one cache per layer. On the worker, the adapter maps each depth
to all registration indices of that depth and waits on each. A batch covers
whole depths of one object group only: with `separate_object_groups=True` a
depth's caches are spread over several object groups, each retrieved and
signalled on its own, so one cache landing says nothing about the others.

---

## 7. IPC Event Pool

The worker waits on one event per layer batch. Creating, exporting and
importing those events per retrieve (`cudaEventCreate`,
`cudaIpcGetEventHandle`, `cudaIpcOpenEventHandle`) would put driver calls on
the forward path. The plugin pays that cost once, at registration, and reuses
the events for every retrieve.

At registration, after LMCache's `register_kv_cache`:

1. The worker calls `layerwise.register_events`.
2. The server creates one interprocess event per registered KV cache and
   exports all their handles (`EventPool`). The pool is sized from the
   registration, so there is no fixed cap to check. It is created once per
   instance; a repeated call returns the same pool.
3. The server returns the handles with `batch_layers` and the frame port.
4. The worker imports every handle once (`EventPool.import_pool`) and
   subscribes to the frames.

Per retrieve, the server only records the batch leader's pool event on the
transfer stream (`cudaEventRecord`) and publishes its index in a frame; the
worker only waits on the pre-imported event at that index. No pool event is
created, exported or imported, and a frame carries an integer rather than an
IPC handle.

Reuse across retrieves is conservative: a later record of the same event is
ordered after the earlier one on the same stream, so a wait that sees it
waits longer, never shorter.

The pool covers only the per-layer signals. LMCache's own `retrieve`, which
the plugin calls, still exchanges two events per retrieve, as it does
per-chunk: the worker exports its producer event and the server imports it,
and the server creates and exports a completion event that the worker
imports. Pooling those as well would change LMCache's `retrieve` contract.

Pools and depths are dropped on `unregister_kv_cache` and on module close. A
`layerwise.retrieve` for an instance without a pool is answered as failed,
after releasing the retrieve's locks.

---

## 8. Frame Channel

Frames do not use the request transport. That keeps every RPC one request,
one answer, so LMCache's transports need no multi-frame support and the
plugin works on any of them.

- **Server:** `_FramePublisher`, a ZMQ PUB socket bound on the server's
  `--host` at `frames_port` (0 picks a free port). One topic per instance
  (`frames_topic`: the instance id as 8 big-endian bytes). Frames are sent
  non-blocking from the retrieve's own thread; PUB drops what a slow or absent
  subscriber cannot take (past 1000 queued).
- **Worker:** a SUB socket connected to the host of the request server URL
  (loopback for `ipc://`) and a daemon reader thread that hands each frame to
  its `_Call` by `call_id`. Calls are held weakly, so frames for retrieves no
  longer awaited are dropped.
- **Handshake:** a SUB subscription takes effect asynchronously. At
  registration the worker calls `layerwise.probe_frames` every 20 ms until the
  probe frame (`call_id == PROBE_CALL_ID`) arrives, for up to the request
  timeout. If none does, it warns, and layer waits fall back to the answers
  without overlapping the transfer.
- **Fallback:** `wait_for_layer` waits for the layer's frame, checking every
  50 ms whether the answer arrived; once it has, it takes the leader from
  `RetrieveResponse.leaders`. A layer reported by neither within 60 s raises
  `LMCacheTimeoutError`; the adapter retires that retrieve and reports its
  blocks for recomputation.

The frame port must be reachable wherever the request port is; set
`frames_port` explicitly when only known ports are published, as with
containers.

---

## 9. Protocol

Payloads are msgpack structs in `lmcache_layerwise/v1/multiprocess/protocol.py`.

| Method | RPC | Request | Response |
| --- | --- | --- | --- |
| `layerwise.set_model_depths` | `server_module_call` | `SetModelDepthsRequest(instance_id, depths)` | empty |
| `layerwise.register_events` | `server_module_call` | `InstanceRequest(instance_id)` | `RegisterEventsResponse(layer_batch, handles, frames_port)` |
| `layerwise.probe_frames` | `server_module_call` | `InstanceRequest(instance_id)` | empty; publishes a probe frame |
| `layerwise.retrieve` | `server_module_affinity_call` | `RetrieveRequest(call_id, key, instance_id, gpu_block_ids, event_ipc_handle, skip_first_n_tokens)` | `RetrieveResponse(event_handle, succeeded, leaders)` |

Frame: `Frame(call_id, leader, layers)`, array-like msgpack, published as
`[topic, payload]`.

LMCache's plain `retrieve` is overridden to answer `(b"", False)` after
releasing its locks: it cannot read layer-major objects, and answering rather
than raising lets the worker fail fast and recompute.

---

## 10. Configuration and Pairing

Server:

```bash
lmcache server ... --supported-transfer-mode lmcache_driven \
  --server-module '{
    "module_path": "lmcache_layerwise.v1.multiprocess.modules.layerwise_transfer",
    "config": {"batch_layers": 4, "frames_port": 0}
  }'
```

`batch_layers` must be an integer >= 1, `enqueue_per_batch` a boolean
(default `false`) and `frames_port` an integer in `[0, 65535]`; all are
validated at startup. With `enqueue_per_batch` a retrieve makes one native
call per layer batch, each followed by its frame, instead of one call per
object group; the first layers are announced sooner at the cost of one GIL
release per call. The server's status reports
`layer_batch`.

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

Set `kv_load_failure_policy` to `"recompute"`. A failed or timed-out
layer-wise load reports its blocks through `get_block_ids_with_load_errors`;
under vLLM's default `"fail"` the request is aborted instead of recomputed.

The two sides must be paired; the worker's mode is fixed by the connector
vLLM imports:

| Server | Worker | Result |
| --- | --- | --- |
| plugin loaded | `LMCacheLayerwiseMPConnector` | layer-wise |
| no plugin | `LMCacheMPConnector` | LMCache per-chunk |
| plugin loaded | `LMCacheMPConnector` | every plain retrieve is answered as failed |
| no plugin | `LMCacheLayerwiseMPConnector` | registration fails: the server has no `layerwise.*` methods |

---

## 11. Layout Uniformity and Shared L2

Objects carry no layout tag. `MemoryLayoutDesc` names an object's contents
per kernel group, which reads as kernel-group-major; that is only safe while
every consumer uses it to size or allocate a buffer. A server therefore runs
either this plugin or LMCache's built-in transfer, never both, and every
object it stores is layer-major.

Servers that share persistent L2 storage must all agree. Switching a
deployment between the plugin and the built-in transfer can leave objects of
the other layout behind, and reading one produces corrupt KV; flush L2 when
switching. A future alternative is to record the layout in each object's
metadata at store time and treat a mismatch as a cache miss, so a stale entry
is never read and is overwritten by the next store.

---

## 12. Debug Logging: `LMCACHE_LAYERWISE_DEBUG`

The layer-wise path is silent by default. Set `LMCACHE_LAYERWISE_DEBUG=1` on
**both** the `lmcache server` and the vLLM processes; each side logs a
different half. Any value other than `0`, `false`, `no` or empty enables it.
The flag is read once at import, and with it off nothing is counted or timed.

| Emitted by | When | Contents |
| --- | --- | --- |
| vLLM worker | registration | KV caches in registration order as `name@depth`, and which caches share one allocation |
| server | first retrieve of each object group | caches, kernel groups and bytes per object; kernel-group runs in object order; per kernel group its registration indices and depths |
| server | every retrieve, per object group | objects, batches, launches, passes, enqueue time and native executor time |
| vLLM worker | end of every retrieve | layer waits and frame arrivals |

Examples:

```
layerwise[group 0]: 12 object(s) in 16 batch(es) of up to 4 depth(s),
16 launch(es) over 16 pass(es), enqueued in 3.10 ms (2.85 ms in the native executor)

layerwise-wait: 64 call(s), 16 stream wait(s), 3 blocked for 6.92 ms,
total 7.36 ms; 16 frame(s) for 64 KV cache(s), per frame (caches@+ms): 4@1.2, 4@4.1, ...
```

How to read the wait line:

- **`stream wait(s)` is `call(s)` divided by the batch size**, mechanically:
  the caches of one batch share a leader and repeated waits are skipped. A
  different ratio means the batching is not what was configured.
- **`blocked` is the number that carries information**: waits whose frame had
  not arrived yet, and the time spent in them. Few blocked waits mean frames
  arrive ahead of demand; as many as frames mean the worker never got ahead
  of the transfer.
- **Frames arrive per object group.** An object group's batches are
  enqueued in one native call and announced together; the overlap within a
  group is on the device, through each batch's own event. Frames late
  relative to the retrieve mean the worker waited on the server to enqueue,
  not on the copies.
- **Frame caches should add up to every KV cache.** Fewer means frames were
  dropped and the waits fell back to the answer.

---

## 13. Limitations and Follow-ups

- `separate_object_groups=True`: placement, slot addressing and plans are
  all per object group, and a batch that is not one contiguous range raises.
  Object groups are retrieved one after another, so a depth with a cache in
  a later group waits for every earlier group in full; overlap is limited to
  the last group's batches.
- CUDA only; other platforms need their own staging and native extension.
- No layout tag in object metadata (section 11).
- Frames are best effort by design; lost frames cost overlap, not
  correctness.
