#!/usr/bin/env python3
"""
Sequentially execute split ResNet-50 TensorRT engines with profiling.

This version does *full GPU chaining*:
- Upload host input to GPU once at the start.
- Each engine reads from a device buffer and writes to another device buffer.
- We DO NOT copy intermediate activations back to the CPU.
- We only download final output to host at the end of each inference.
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

num_chunks = len(contexts)
print(f"✅ Loaded {num_chunks} chunk engines.")

# ============================================================
# 2. Host input + discover tensor shapes
# ============================================================

# 1 sample dummy input
host_input = np.random.randn(1, 3, 224, 224).astype(np.float32)

# infer I/O shapes for each chunk
# we assume binding 0 is input, binding 1 is output
chunk_input_shapes = []
chunk_output_shapes = []
for ctx in contexts:
    in_shape = tuple(ctx.get_binding_shape(0))
    out_shape = tuple(ctx.get_binding_shape(1))
    chunk_input_shapes.append(in_shape)
    chunk_output_shapes.append(out_shape)

# the final output shape of the last chunk (for host download)
final_out_shape = chunk_output_shapes[-1]
print(f"[INFO] Final output shape: {final_out_shape}")

# host buffer for final result
host_final_output = np.empty(final_out_shape, dtype=np.float32)

# ============================================================
# 3. Allocate device buffers for chaining
# ============================================================

# Strategy:
# - d_activation[i] will hold the *output* of chunk i
# - For chunk 0, we upload host_input -> d_activation[-1_input]
#   We'll just call that d_input0.
#
# Memory layout we'll maintain:
#   d_input0 (input to chunk 0)
#   d_activation[0] (output of chunk 0 / input to chunk 1)
#   d_activation[1] (output of chunk 1 / input to chunk 2)
#   ...
#   d_activation[num_chunks-1] (output of last chunk)

stream = cuda.Stream()

# allocate device buffer for the first chunk's input
bytes_input0 = int(np.prod(chunk_input_shapes[0])) * 4
d_input0 = cuda.mem_alloc(bytes_input0)

# allocate device buffers for each chunk's output
d_activation = []
for out_shape in chunk_output_shapes:
    size_bytes = int(np.prod(out_shape)) * 4
    d_activation.append(cuda.mem_alloc(size_bytes))

print("[INFO] Allocated device buffers for all chunk activations on GPU.")

# ============================================================
# 4. Run a single chunk on GPU buffers only
# ============================================================

def run_chunk_device_only(context, d_in_ptr, d_out_ptr, stream):
    """
    context: TensorRT execution context for this chunk
    d_in_ptr, d_out_ptr: pycuda GPUAllocation objects
    """
    start = cuda.Event()
    end = cuda.Event()

    start.record(stream)

    # IMPORTANT: execute_v2() takes a list of bindings as raw ints (addresses)
    context.execute_v2([int(d_in_ptr), int(d_out_ptr)])

    end.record(stream)
    end.synchronize()

    return start.time_till(end)  # milliseconds

# ============================================================
# 5. Warmups + timed runs with NVTX
# ============================================================

warmups = 5
num_runs = 20
all_run_times = []  # list of tuples: (chunk_times_ms[], total_gpu_ms, total_cpu_ms)

print(f"\n[INFO] Starting sequential inference ({warmups} warmups, {num_runs} runs)...\n")

# -----------------
# Warmup passes
# -----------------
for w in range(warmups):
    nvtx.range_push(f"Warmup {w+1}")

    # upload first input ONLY once at start of chain
    cuda.memcpy_htod_async(d_input0, host_input, stream)

    # now walk chunks entirely on GPU
    for i, ctx in enumerate(contexts):
        d_src = d_input0 if i == 0 else d_activation[i - 1]
        d_dst = d_activation[i]
        _ = run_chunk_device_only(ctx, d_src, d_dst, stream)

    # download only final output
    cuda.memcpy_dtoh_async(host_final_output, d_activation[-1], stream)
    stream.synchronize()

    nvtx.range_pop()

# -----------------
# Timed passes
# -----------------
for r in range(num_runs):
    nvtx.range_push(f"Inference {r+1}")

    # copy host input -> GPU input buffer for first chunk
    cuda.memcpy_htod_async(d_input0, host_input, stream)

    chunk_times = []
    t0 = time.time()

    for i, ctx in enumerate(contexts):
        nvtx.range_push(f"Chunk {i+1}")

        # d_src: input buffer for this chunk on GPU
        d_src = d_input0 if i == 0 else d_activation[i - 1]
        # d_dst: output buffer for this chunk on GPU
        d_dst = d_activation[i]

        gpu_ms = run_chunk_device_only(ctx, d_src, d_dst, stream)
        chunk_times.append(gpu_ms)

        nvtx.range_pop()

    # after full chain, pull ONLY final result back
    cuda.memcpy_dtoh_async(host_final_output, d_activation[-1], stream)
    stream.synchronize()

    t1 = time.time()

    total_gpu_time = sum(chunk_times)  # ms, sum of kernel times
    total_cpu_time = (t1 - t0) * 1000  # ms, wall clock including copies + python

    all_run_times.append((chunk_times, total_gpu_time, total_cpu_time))

    print(
        f"Inference {r+1}: total GPU {total_gpu_time:.3f} ms | "
        f"total CPU {total_cpu_time:.3f} ms"
    )

    nvtx.range_pop()

# ============================================================
# 6. Aggregate statistics
# ============================================================

gpu_totals = [t[1] for t in all_run_times]
cpu_totals = [t[2] for t in all_run_times]

def summarize(arr):
    arr = np.array(arr)
    return (
        f"{np.mean(arr):.3f} ± {np.std(arr):.3f} "
        f"(min={np.min(arr):.3f}, max={np.max(arr):.3f})"
    )

print("\n✅ Sequential inference complete with full GPU chaining.\n")
print(f"GPU total time (ms): {summarize(gpu_totals)}")
print(f"CPU total time (ms): {summarize(cpu_totals)}")

# ============================================================
# 7. CSV dump
# ============================================================

with open("chunk_timing.csv", "w") as f:
    # Header
    header_chunks = ",".join([f"chunk{i+1}_gpu_ms" for i in range(num_chunks)])
    f.write("run_id," + header_chunks + ",total_gpu_ms,total_cpu_ms\n")

    # Rows
    for r_i, (chunk_times, gpu_total, cpu_total) in enumerate(all_run_times, 1):
        chunk_str = ",".join([f"{x:.4f}" for x in chunk_times])
        f.write(
            f"{r_i},{chunk_str},{gpu_total:.4f},{cpu_total:.4f}\n"
        )

print("📝 Saved detailed timing to chunk_timing.csv")
