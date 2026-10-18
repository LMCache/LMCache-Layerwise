// SPDX-License-Identifier: Apache-2.0
//
// Native side of the layer-wise transfer.
//
// Only what is specific to layer-wise lives here: the upload of the LMCache
// object pointers to a device array (a launch can cover more objects than
// LMCache's in-register MemoryObj4 holds; smaller launches pass them in a
// MemoryObj4 as LMCache does) and the layer-major object-offset policy. Plan
// execution (staging copies, bounds checks, launch ordering, phase timing)
// runs in LMCache's executor through the lmcache_ext API, and the kernel is
// LMCache's, instantiated with these two policies.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <array>
#include <cstring>
#include <optional>
#include <tuple>
#include <vector>

#include "cuda/lmcache_ext_api.h"
#include "cuda/mp_mem_kernels_detail.cuh"

namespace {

const lmcache_ext::Api* g_api = nullptr;

// Where layer `layer_idx` of a kernel group sits inside an LMCache object.
//
// kv_interleaved == false: LMCache's default 2LTD layout [2, L, T, D].
// kv_interleaved == true:  L2TD layout [L, 2, T, D], one layer's K and V
//   adjacent. Consecutive layers are `layer_stride_elems` source-dtype
//   elements apart, or tightly packed when it is 0. A layer-major object
//   holds every kernel group of a model layer before the next layer, so one
//   group's layers are a whole layer extent apart rather than packed.
struct LayerMajorObjectOffset {
  bool kv_interleaved = false;
  int64_t layer_stride_elems = 0;

  template <typename ScalarType, EngineKVFormat format>
  __device__ inline size_t offset(const int k_or_v, const int token_offset,
                                  const int layer_idx,
                                  const int lmcache_chunk_size,
                                  const PageBufferShapeDesc& shape_desc) const {
    if (!kv_interleaved) {
      return lmcache_mp::TwoLTDObjectOffset{}.offset<ScalarType, format>(
          k_or_v, token_offset, layer_idx, lmcache_chunk_size, shape_desc);
    }
    const size_t scalars_per_token = shape_desc.scalars_per_token<ScalarType>();
    const size_t scalars_per_layer =
        layer_stride_elems > 0
            ? static_cast<size_t>(layer_stride_elems) *
                  shape_desc.element_size / sizeof(ScalarType)
            : static_cast<size_t>(shape_desc.kv_size) * lmcache_chunk_size *
                  scalars_per_token;
    return token_offset * scalars_per_token +
           k_or_v * lmcache_chunk_size * scalars_per_token +
           layer_idx * scalars_per_layer;
  }
};

// Upper bound on LMCache objects per launch. The pointer array is too large
// for kernel parameter space, so it is staged through pinned host memory.
constexpr int kMaxObjects = 1024;

// Launches of at most this many objects pass them in LMCache's MemoryObj4.
constexpr int kByValueObjects = 4;

// Independent staging slots kept per thread. One slot is occupied per
// distinct pointer array, so this need only cover the number of kernel groups
// a thread interleaves -- in practice one or two.
constexpr int kUploadSlots = 4;

// Uploads num_objects LMCache object pointers and returns their device
// address.
//
// The host side of an in-flight cudaMemcpyAsync must not be rewritten before
// the copy runs, and nothing on this path synchronises with the stream. A
// single shared staging buffer would let one launch overwrite pointers an
// earlier launch has queued but not yet copied. Slots are handed out per
// distinct pointer array instead: a kernel group's object pointers are the
// same for every layer, so repeat launches find their array already resident
// and upload nothing, while a different kernel group gets a slot of its own.
// Wrapping past kUploadSlots distinct arrays waits on that slot's previous
// upload, so correctness never depends on the host staying behind the device.
const int64_t* upload_object_ptrs(const int64_t* object_ptrs, int num_objects,
                                  int device_index, cudaStream_t stream) {
  TORCH_CHECK(num_objects >= 1 && num_objects <= kMaxObjects,
              "layer-wise launches support 1 to ", kMaxObjects,
              " objects, got ", num_objects);

  static thread_local int64_t* pinned_host = nullptr;
  static thread_local torch::Tensor dev_buf;
  static thread_local int dev_buf_device_index = -1;
  static thread_local std::array<cudaEvent_t, kUploadSlots> slot_uploaded{};
  static thread_local std::array<cudaStream_t, kUploadSlots> slot_stream{};
  static thread_local std::array<int, kUploadSlots> slot_len{};
  static thread_local int next_slot = 0;

  const size_t nbytes = static_cast<size_t>(num_objects) * sizeof(int64_t);

  if (pinned_host == nullptr) {
    C10_CUDA_CHECK(cudaHostAlloc(
        reinterpret_cast<void**>(&pinned_host),
        static_cast<size_t>(kMaxObjects) * kUploadSlots * sizeof(int64_t),
        cudaHostAllocDefault));
    for (auto& event : slot_uploaded) {
      C10_CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    }
  }

  // Allocated once per device. Resident copies do not survive reallocation,
  // so the slot state is dropped.
  if (dev_buf_device_index != device_index) {
    dev_buf =
        torch::empty({static_cast<int64_t>(kMaxObjects) * kUploadSlots},
                     torch::TensorOptions()
                         .dtype(torch::kInt64)
                         .device(torch::Device(torch::kCUDA, device_index)));
    dev_buf_device_index = device_index;
    slot_len.fill(0);
    next_slot = 0;
  }
  int64_t* dev_base = dev_buf.data_ptr<int64_t>();

  // Already resident from an earlier launch on this stream: the earlier copy
  // is ordered ahead of the kernel about to be launched. A longer resident
  // array also serves a shorter one, since the kernel reads only the first
  // num_objects entries.
  for (int i = 0; i < kUploadSlots; ++i) {
    if (slot_len[i] >= num_objects && slot_stream[i] == stream &&
        std::memcmp(pinned_host + static_cast<size_t>(i) * kMaxObjects,
                    object_ptrs, nbytes) == 0) {
      return dev_base + static_cast<size_t>(i) * kMaxObjects;
    }
  }

  const int slot = next_slot;
  next_slot = (next_slot + 1) % kUploadSlots;
  if (slot_len[slot] != 0) {
    // Reclaiming a slot: its previous upload must have read the host side.
    // The only host wait of a plan, so the only place the GIL is let go.
    std::optional<py::gil_scoped_release> no_gil;
    if (PyGILState_Check()) {
      no_gil.emplace();
    }
    C10_CUDA_CHECK(cudaEventSynchronize(slot_uploaded[slot]));
  }
  int64_t* host_slot = pinned_host + static_cast<size_t>(slot) * kMaxObjects;
  int64_t* dev_slot = dev_base + static_cast<size_t>(slot) * kMaxObjects;
  std::memcpy(host_slot, object_ptrs, nbytes);
  C10_CUDA_CHECK(cudaMemcpyAsync(dev_slot, host_slot, nbytes,
                                 cudaMemcpyHostToDevice, stream));
  C10_CUDA_CHECK(cudaEventRecord(slot_uploaded[slot], stream));
  slot_stream[slot] = stream;
  slot_len[slot] = num_objects;
  return dev_slot;
}

struct LayerwisePlan {
  TransferDirection direction;
  int device_index;
  std::vector<LayerMajorObjectOffset> offsets;  // one per kernel group
};

void launch_group(void* user_data, const KernelGroupSpec& group,
                  const LaunchVar& launch, const int64_t* block_ids,
                  cudaStream_t stream) {
  const auto& plan = *static_cast<const LayerwisePlan*>(user_data);
  const LayerMajorObjectOffset offset = plan.offsets[launch.group_idx];
  lmcache_mp::dispatch_by_transfer_unit(
      group.shape_desc, group.engine_kv_format, [&](auto unit) {
        using T = typename decltype(unit)::type;
        const auto launch_with = [&](auto object_ptrs) {
          lmcache_mp::launch_multi_layer_block_transfer<T>(
              object_ptrs, launch.num_objects,
              reinterpret_cast<T**>(group.paged_buffer_ptrs), block_ids,
              launch.total_blocks, plan.direction, group.shape_desc,
              group.lmcache_chunk_size, group.engine_kv_format,
              launch.skip_prefix_n_blocks, stream, offset);
        };
        // Up to four objects travel by value, as in LMCache's own launcher.
        // Nothing is uploaded, so the host never waits on the stream. Store
        // batches, at most max_batch_size (4 by default) objects, take this
        // path.
        if (launch.num_objects >= 1 && launch.num_objects <= kByValueObjects) {
          lmcache_mp::MemoryObj4<T> objects{};
          objects.num_objects = launch.num_objects;
          for (int i = 0; i < launch.num_objects; ++i) {
            objects.objects[i] =
                reinterpret_cast<T*>(group.lmcache_objects_ptrs[i]);
          }
          launch_with(objects);
          return;
        }
        launch_with(reinterpret_cast<T* const*>(
            upload_object_ptrs(group.lmcache_objects_ptrs.data(),
                               launch.num_objects, plan.device_index, stream)));
      });
}

void execute_object_group_transfer(
    int direction, int device_index, size_t host_buffer_alignment,
    const std::vector<KernelGroupSpec>& kernel_group_specs,
    const std::vector<BatchStep>& batch_steps,
    const std::vector<bool>& kv_interleaved,
    const std::vector<int64_t>& layer_stride_elems, bool phase_timing_enabled,
    const std::string& session_id) {
  const size_t num_groups = kernel_group_specs.size();
  TORCH_CHECK(kv_interleaved.size() == num_groups &&
                  layer_stride_elems.size() == num_groups,
              "kv_interleaved and layer_stride_elems need one entry per "
              "kernel group (",
              num_groups, "), got ", kv_interleaved.size(), " and ",
              layer_stride_elems.size());
  LayerwisePlan plan{
      static_cast<TransferDirection>(direction), device_index, {}};
  plan.offsets.reserve(num_groups);
  for (size_t i = 0; i < num_groups; ++i) {
    TORCH_CHECK(layer_stride_elems[i] >= 0,
                "layer_stride_elems must be >= 0, got ", layer_stride_elems[i],
                " for kernel group ", i);
    TORCH_CHECK(kv_interleaved[i] || layer_stride_elems[i] == 0,
                "layer_stride_elems is only supported with kv_interleaved "
                "(kernel group ",
                i, ")");
    plan.offsets.push_back({kv_interleaved[i], layer_stride_elems[i]});
  }
  g_api->execute_object_group_transfer(
      plan.direction, device_index, host_buffer_alignment, kernel_group_specs,
      batch_steps, &launch_group, &plan, phase_timing_enabled,
      session_id.c_str());
}

// One layer batch of execute_layerwise_batches: the kernel group specs, batch
// steps, kv_interleaved and layer_stride_elems of
// execute_object_group_transfer, then the cudaEvent_t to record once the
// batch is enqueued (0 for none).
using LayerBatch =
    std::tuple<std::vector<KernelGroupSpec>, std::vector<BatchStep>,
               std::vector<bool>, std::vector<int64_t>, int64_t>;

// Enqueues the layer batches of one object group in order, recording each
// batch's event on the current stream right after the batch, so a consumer
// waiting on it can start before later batches land.
void execute_layerwise_batches(int direction, int device_index,
                               size_t host_buffer_alignment,
                               const std::vector<LayerBatch>& batches) {
  const auto device = static_cast<c10::DeviceIndex>(device_index);
  const c10::cuda::CUDAGuard guard(device);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(device).stream();
  for (const auto& [specs, steps, kv_interleaved, layer_stride_elems, event] :
       batches) {
    execute_object_group_transfer(
        direction, device_index, host_buffer_alignment, specs, steps,
        kv_interleaved, layer_stride_elems, false, "");
    if (event != 0) {
      C10_CUDA_CHECK(
          cudaEventRecord(reinterpret_cast<cudaEvent_t>(event), stream));
    }
  }
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  g_api = lmcache_ext::import_api();
  if (g_api == nullptr) {
    throw py::error_already_set();
  }
  m.attr("MAX_OBJECTS_PER_LAUNCH") = kMaxObjects;
  // Runs with the GIL held. A batch only enqueues asynchronous work, about
  // 0.1 ms, while a thread that releases the GIL waits up to the switch
  // interval (5 ms) to take it back whenever another thread runs Python, and
  // a retrieve makes one call per layer batch. upload_object_ptrs releases it
  // around its one host wait.
  m.def("execute_object_group_transfer", &execute_object_group_transfer,
        py::arg("direction"), py::arg("device_index"),
        py::arg("host_buffer_alignment"), py::arg("kernel_group_specs"),
        py::arg("batch_steps"), py::arg("kv_interleaved"),
        py::arg("layer_stride_elems"), py::arg("phase_timing_enabled") = false,
        py::arg("session_id") = "");
  // One call per object group of a retrieve, rather than one per layer batch
  // plus a Python event record each. It runs without the GIL: an object group
  // of many layers and chunks takes milliseconds to enqueue (about 2,500
  // staging copies for Qwen3-32B with one layer per batch), and holding the
  // GIL that long serializes the retrieves of every rank and request.
  m.def("execute_layerwise_batches", &execute_layerwise_batches,
        py::arg("direction"), py::arg("device_index"),
        py::arg("host_buffer_alignment"), py::arg("batches"),
        py::call_guard<py::gil_scoped_release>());
}
