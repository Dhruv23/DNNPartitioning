#!/usr/bin/env python3
"""
Sequentially execute split ResNet-50 TensorRT engines.
Implements the single-task, sequential runtime described in Section V-C of paper151.
"""

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import time
from torch.cuda import nvtx
import glob
import os

# ============================================================
# 1. Setup TensorRT runtime and load all chunk engines
# ============================================================

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
runtime = trt.Runtime(TRT_LOGGER)

ENGINE_DIR = "engines"
engine_paths = sorted(glob.glob(os.path.join(ENGINE_DIR, "*.engine")))

engines = []
contexts = []

for path in engine_paths:
    with open(path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
        ctx = engine.create_execution_context()
        engines.append(engine)
        contexts.append(ctx)
    print(f"[INFO] Loaded engine: {os.path.basename(path)}")

print(f"✅ Loaded {len(engines)} chunk engines.")

# ============================================================
# 2. Allocate reusable GPU buffers (FIXED)
# ============================================================

# Prepare dummy input (batch 1, RGB image)
input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)

# Find maximum buffer size across all chunks
max_bytes = 0
for engine in engines:
    for b in range(engine.num_bindings):
        shape = engine.get_binding_shape(b)
        # Dynamic shapes may include -1; skip those safely
        if all(dim > 0 for dim in shape):
            size_bytes = int(np.prod(shape)) * 4  # float32 = 4 bytes
            max_bytes = max(max_bytes, size_bytes)

print(f"[INFO] Allocating {max_bytes / (1024**2):.2f} MB for reusable buffers...")

# Allocate device buffers as plain Python int sizes
d_input = cuda.mem_alloc(int(max_bytes))
d_output = cuda.mem_alloc(int(max_bytes))
stream = cuda.Stream()


# ============================================================
# 3. Define helper to run one chunk
# ============================================================

def run_chunk(context, h_input, h_output, d_input, d_output, stream):
    """Run a single TensorRT engine chunk."""
    cuda.memcpy_htod_async(d_input, h_input, stream)

    start = cuda.Event()
    end = cuda.Event()
    start.record(stream)

    context.execute_v2([int(d_input), int(d_output)])

    end.record(stream)
    end.synchronize()
    cuda.memcpy_dtoh_async(h_output, d_output, stream)
    stream.synchronize()

    return start.time_till(end)  # milliseconds


# ============================================================
# 4. Execute chunks sequentially
# ============================================================

h_input = input_data
times = []

print("\n[INFO] Starting sequential inference ...\n")

for i, (engine, ctx) in enumerate(zip(engines, contexts), 1):
    nvtx.range_push(f"Chunk {i}")
    out_shape = tuple(ctx.get_binding_shape(1))
    h_output = np.empty(out_shape, dtype=np.float32)

    gpu_ms = run_chunk(ctx, h_input, h_output, d_input, d_output, stream)
    times.append(gpu_ms)

    print(f"Chunk {i}: {gpu_ms:.3f} ms, output shape = {out_shape}")

    # Output of this chunk becomes next input
    h_input = h_output.copy()
    nvtx.range_pop()

total_time = sum(times)
print(f"\n✅ Sequential inference complete.")
print(f"Per-chunk GPU times (ms): {['%.3f' % t for t in times]}")
print(f"Total GPU time: {total_time:.3f} ms")

# ============================================================
# 5. (Optional) Log simple timeline to CSV
# ============================================================

with open("chunk_timing.csv", "w") as f:
    f.write("chunk_id,gpu_time_ms\n")
    for i, t in enumerate(times, 1):
        f.write(f"{i},{t:.4f}\n")
    f.write(f"total,{total_time:.4f}\n")

print("📝 Saved per-chunk timing to chunk_timing.csv")
