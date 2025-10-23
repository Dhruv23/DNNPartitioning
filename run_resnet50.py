#!/usr/bin/env python3
"""
Single-engine TensorRT inference with detailed timing.
Includes NVTX markers, warmup runs, per-call overhead, and stats.
"""

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import time
from torch.cuda import nvtx
import os

# ============================================================
# 1. Load TensorRT engine
# ============================================================

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
ENGINE_PATH = "resnet50.engine"

if not os.path.exists(ENGINE_PATH):
    raise FileNotFoundError(f"❌ Engine file not found: {ENGINE_PATH}")

print(f"[INFO] Loading engine: {ENGINE_PATH}")
runtime = trt.Runtime(TRT_LOGGER)
with open(ENGINE_PATH, "rb") as f:
    engine = runtime.deserialize_cuda_engine(f.read())

context = engine.create_execution_context()
print("✅ Engine loaded and execution context created.\n")

# ============================================================
# 2. Allocate buffers
# ============================================================

input_shape = tuple(engine.get_binding_shape(0))
output_shape = tuple(engine.get_binding_shape(1))

h_input = np.random.randn(*input_shape).astype(np.float32)
h_output = np.empty(output_shape, dtype=np.float32)

d_input = cuda.mem_alloc(h_input.nbytes)
d_output = cuda.mem_alloc(h_output.nbytes)
stream = cuda.Stream()

print(f"[INFO] Input shape:  {input_shape}")
print(f"[INFO] Output shape: {output_shape}")
print(f"[INFO] Allocated {h_input.nbytes / (1024 ** 2):.2f} MB total buffer memory.\n")

# ============================================================
# 3. Helper: Run inference once
# ============================================================

def run_inference(context, h_input, h_output, d_input, d_output, stream):
    """Perform one inference pass and return GPU kernel time (ms)."""
    cuda.memcpy_htod_async(d_input, h_input, stream)

    start = cuda.Event()
    end = cuda.Event()
    start.record(stream)

    context.execute_v2([int(d_input), int(d_output)])

    end.record(stream)
    end.synchronize()

    cuda.memcpy_dtoh_async(h_output, d_output, stream)
    stream.synchronize()

    return start.time_till(end)  # GPU kernel duration in ms


# ============================================================
# 4. Profiling loop: warmups + timed runs
# ============================================================

warmups = 5
num_runs = 20
gpu_times, cpu_times, overheads = [], [], []

print(f"[INFO] Starting profiling loop ({warmups} warmups, {num_runs} measured runs)...\n")

# --- Warmup ---
for i in range(warmups):
    nvtx.range_push(f"Warmup {i+1}")
    run_inference(context, h_input, h_output, d_input, d_output, stream)
    nvtx.range_pop()
    time.sleep(0.05)  # small gap to stabilize GPU clocks

# --- Timed runs ---
for i in range(num_runs):
    label = f"Inference {i+1}"
    nvtx.range_push(label)

    t0 = time.time()
    gpu_ms = run_inference(context, h_input, h_output, d_input, d_output, stream)
    t1 = time.time()
    cpu_ms = (t1 - t0) * 1000


    # Overhead between calls (time since previous end)
    if i > 0:
        overhead = (t0 - prev_t1) * 1000
        overheads.append(overhead)
    prev_t1 = t1

    nvtx.range_pop()

    gpu_times.append(gpu_ms)
    cpu_times.append(cpu_ms)

    print(f"{label}: GPU {gpu_ms:.3f} ms | CPU {cpu_ms:.3f} ms")

# ============================================================
# 5. Compute statistics
# ============================================================

def summarize(name, arr):
    if len(arr) == 0:
        return "N/A"
    return f"{np.mean(arr):.3f} ± {np.std(arr):.3f} (min={np.min(arr):.3f}, max={np.max(arr):.3f})"

print("\n✅ Inference complete.\n")
print(f"GPU Time (ms): {summarize('GPU', gpu_times)}")
print(f"CPU Time (ms): {summarize('CPU', cpu_times)}")
print(f"Overhead (ms): {summarize('Overhead', overheads)}")

# ============================================================
# 6. Save timing to CSV
# ============================================================

with open("inference_timing.csv", "w") as f:
    f.write("iteration,gpu_time_ms,cpu_time_ms,overhead_ms\n")
    for i, (g, c) in enumerate(zip(gpu_times, cpu_times), 1):
        overhead = overheads[i-2] if i > 1 else 0.0
        f.write(f"{i},{g:.4f},{c:.4f},{overhead:.4f}\n")
    f.write(f"mean,{np.mean(gpu_times):.4f},{np.mean(cpu_times):.4f},{np.mean(overheads):.4f}\n")

print("📝 Saved timing details to inference_timing.csv")
