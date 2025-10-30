#!/usr/bin/env python3
"""
Sequentially execute split ResNet-50 TensorRT engines with profiling.
Now uses CUDA pinned (page-locked) host memory to reduce CPU↔GPU transfer overhead.
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

engines, contexts = [], []
for path in engine_paths:
    with open(path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
        ctx = engine.create_execution_context()
        engines.append(engine)
        contexts.append(ctx)
    print(f"[INFO] Loaded engine: {os.path.basename(path)}")

print(f"✅ Loaded {len(engines)} chunk engines.")

# ============================================================
# 2. Allocate reusable GPU buffers and pinned host buffers
# ============================================================

input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)

max_bytes = 0
for engine in engines:
    for b in range(engine.num_bindings):
        shape = engine.get_binding_shape(b)
        if all(dim > 0 for dim in shape):
            size_bytes = int(np.prod(shape)) * 4
            max_bytes = max(max_bytes, size_bytes)

print(f"[INFO] Allocating {max_bytes / (1024**2):.2f} MB for reusable buffers...")

# GPU buffers
d_input = cuda.mem_alloc(int(max_bytes))
d_output = cuda.mem_alloc(int(max_bytes))
stream = cuda.Stream()

# Pinned host buffers per chunk (for input/output reuse)
pinned_inputs = []
pinned_outputs = []

for ctx in contexts:
    out_shape = tuple(ctx.get_binding_shape(1))
    h_in = cuda.pagelocked_empty(input_data.size, np.float32).reshape(input_data.shape)
    h_out = cuda.pagelocked_empty(int(np.prod(out_shape)), np.float32).reshape(out_shape)
    pinned_inputs.append(h_in)
    pinned_outputs.append(h_out)

# ============================================================
# 3. Helper to run one chunk
# ============================================================

def run_chunk(context, h_input, h_output, d_input, d_output, stream):
    cuda.memcpy_htod_async(d_input, h_input, stream)
    start = cuda.Event()
    end = cuda.Event()
    start.record(stream)
    context.execute_v2([int(d_input), int(d_output)])
    end.record(stream)
    cuda.memcpy_dtoh_async(h_output, d_output, stream)
    stream.synchronize()
    return start.time_till(end)  # ms

# ============================================================
# 4. Sequential inference with warmup and timing
# ============================================================

warmups = 5
num_runs = 20
all_run_times = []

print(f"\n[INFO] Starting sequential inference ({warmups} warmups, {num_runs} runs)...\n")

# Warmups
for w in range(warmups):
    nvtx.range_push(f"Warmup {w+1}")
    np.copyto(pinned_inputs[0], input_data)
    for i, ctx in enumerate(contexts):
        h_input = pinned_inputs[i] if i == 0 else pinned_outputs[i-1]
        h_output = pinned_outputs[i]
        _ = run_chunk(ctx, h_input, h_output, d_input, d_output, stream)
    nvtx.range_pop()

# Timed runs
for r in range(num_runs):
    nvtx.range_push(f"Inference {r+1}")
    np.copyto(pinned_inputs[0], input_data)
    chunk_times = []
    t0 = time.time()

    for i, ctx in enumerate(contexts):
        nvtx.range_push(f"Chunk {i+1}")
        h_input = pinned_inputs[i] if i == 0 else pinned_outputs[i-1]
        h_output = pinned_outputs[i]
        gpu_ms = run_chunk(ctx, h_input, h_output, d_input, d_output, stream)
        chunk_times.append(gpu_ms)
        nvtx.range_pop()

    t1 = time.time()
    total_gpu_time = sum(chunk_times)
    total_cpu_time = (t1 - t0) * 1000
    all_run_times.append((chunk_times, total_gpu_time, total_cpu_time))

    print(f"Inference {r+1}: total GPU {total_gpu_time:.3f} ms | total CPU {total_cpu_time:.3f} ms")

    nvtx.range_pop()

# ============================================================
# 5. Stats
# ============================================================

gpu_totals = [t[1] for t in all_run_times]
cpu_totals = [t[2] for t in all_run_times]

def summarize(name, arr):
    arr = np.array(arr)
    return f"{np.mean(arr):.3f} ± {np.std(arr):.3f} (min={np.min(arr):.3f}, max={np.max(arr):.3f})"

print("\n✅ Sequential inference complete.\n")
print(f"GPU total time (ms): {summarize('GPU', gpu_totals)}")
print(f"CPU total time (ms): {summarize('CPU', cpu_totals)}")

# ============================================================
# 6. CSV export
# ============================================================

with open("chunk_timing.csv", "w") as f:
    f.write("run_id," + ",".join([f"chunk{i+1}_gpu_ms" for i in range(len(engines))]) + ",total_gpu_ms,total_cpu_ms\n")
    for r, (chunk_times, gpu_total, cpu_total) in enumerate(all_run_times, 1):
        chunk_str = ",".join([f"{x:.4f}" for x in chunk_times])
        f.write(f"{r},{chunk_str},{gpu_total:.4f},{cpu_total:.4f}\n")

print("📝 Saved detailed timing to chunk_timing.csv")
