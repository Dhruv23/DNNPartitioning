#!/usr/bin/env python3
"""
TensorRT 10.14.01 sequential chained inference (fully on-GPU)

- Loads all .engine files from ./engines (sorted)
- Supports dynamic shapes (sets batch=1,3x224x224 for first chunk; later chunks inherit previous output shapes)
- Uses CUDA events for per-chunk GPU timing
- Measures CPU wall time per run
- Warmups=10, Runs=100
- Saves per-run, per-chunk timing to chunk_timing.csv
"""

import os
import glob
import time
import csv
import numpy as np

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # Create a CUDA context

# -----------------------------
# Config
# -----------------------------
ENGINE_DIR = "engines"
BATCH, CH, H, W = 1, 3, 224, 224
INPUT_SHAPE = (BATCH, CH, H, W)

WARMUPS = 10
RUNS = 100
RNG_SEED = 42

# -----------------------------
# Helpers
# -----------------------------
def vol(shape) -> int:
    v = 1
    for d in shape:
        v *= int(d)
    return v

def is_dynamic(shape) -> bool:
    # TensorRT denotes dynamic dims with -1
    return any(int(d) < 0 for d in shape)

def np_dtype_for(engine, tensor_name):
    """Return numpy dtype for a given tensor in this engine."""
    return trt.nptype(engine.get_tensor_dtype(tensor_name))

def single_io_names(engine):
    """
    Return (input_name, output_name) for an engine that has exactly 1 input and 1 output.
    Raises if not true.
    """
    in_names, out_names = [], []
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        mode = engine.get_tensor_mode(name)
        if mode == trt.TensorIOMode.INPUT:
            in_names.append(name)
        elif mode == trt.TensorIOMode.OUTPUT:
            out_names.append(name)

    if len(in_names) != 1 or len(out_names) != 1:
        raise RuntimeError(
            f"Engine must have exactly 1 input and 1 output. Found inputs={in_names}, outputs={out_names}"
        )
    return in_names[0], out_names[0]

def bytes_for_shape_and_dtype(shape, dtype):
    return vol(shape) * np.dtype(dtype).itemsize

# -----------------------------
# 1) Load engines + contexts
# -----------------------------
logger = trt.Logger(trt.Logger.WARNING)
runtime = trt.Runtime(logger)

engine_paths = sorted(glob.glob(os.path.join(ENGINE_DIR, "*.engine")))
if not engine_paths:
    raise RuntimeError(f"No .engine files found under ./{ENGINE_DIR}")

engines = []
contexts = []
io_names = []  # list of (in_name, out_name) per chunk

for p in engine_paths:
    with open(p, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
        if engine is None:
            raise RuntimeError(f"Failed to load engine: {p}")

    ctx = engine.create_execution_context()
    if ctx is None:
        raise RuntimeError(f"Failed to create execution context for: {p}")

    in_name, out_name = single_io_names(engine)

    engines.append(engine)
    contexts.append(ctx)
    io_names.append((in_name, out_name))
    print(f"[INFO] Loaded engine: {os.path.basename(p)}")

num_chunks = len(engines)
print(f"✅ Loaded {num_chunks} chunk engines.")

# -----------------------------
# 2) Determine concrete shapes
#    and host input
# -----------------------------
rng = np.random.default_rng(RNG_SEED)
h_in = rng.normal(0.0, 1.0, size=INPUT_SHAPE).astype(np.float32)

in_shapes = []
out_shapes = []
in_dtypes = []
out_dtypes = []

for i, (eng, ctx) in enumerate(zip(engines, contexts)):
    in_name, out_name = io_names[i]

    # Engine-declared shapes (may include -1)
    eng_in_shape = tuple(eng.get_tensor_shape(in_name))

    # If dynamic, set a concrete input shape
    if is_dynamic(eng_in_shape):
        # First chunk: use the fixed input image shape
        # Later chunks: inherit previous out_shape
        desired = INPUT_SHAPE if i == 0 else out_shapes[i - 1]
        if not ctx.set_input_shape(in_name, tuple(desired)):
            raise RuntimeError(f"Failed to set input shape {desired} for chunk {i+1}")

    # Now query concrete shapes from the context
    in_shape = tuple(ctx.get_tensor_shape(in_name))
    out_shape = tuple(ctx.get_tensor_shape(out_name))
    in_shapes.append(in_shape)
    out_shapes.append(out_shape)

    # Record dtypes
    in_dtypes.append(np_dtype_for(eng, in_name))
    out_dtypes.append(np_dtype_for(eng, out_name))

print(f"[INFO] Final output dims: [{', '.join(str(x) for x in out_shapes[-1])}]")

# Allocate host output buffer that matches final chunk dtype & shape
final_dtype = out_dtypes[-1]
h_out = np.empty(out_shapes[-1], dtype=final_dtype)

# -----------------------------
# 3) Allocate device buffers
# -----------------------------
stream = cuda.Stream()

# Input buffer for first engine
bytes_in0 = bytes_for_shape_and_dtype(in_shapes[0], in_dtypes[0])
d_in0 = cuda.mem_alloc(bytes_in0)

# One device activation/output buffer per chunk
d_act = []
for shape, dtype in zip(out_shapes, out_dtypes):
    d_act.append(cuda.mem_alloc(bytes_for_shape_and_dtype(shape, dtype)))

print("[INFO] Allocated device buffers.")

# -----------------------------
# 4) Run helper (per chunk)
# -----------------------------
def run_chunk(i: int, d_src, d_dst, stream: cuda.Stream) -> float:
    eng = engines[i]
    ctx = contexts[i]
    in_name, out_name = io_names[i]

    # It is safer to set the shapes again for dynamic networks if input shapes can vary;
    # here shapes are static across runs, so we skip repeating set_input_shape.

    # Tell TRT where the device buffers are
    ctx.set_tensor_address(in_name, int(d_src))
    ctx.set_tensor_address(out_name, int(d_dst))

    start = cuda.Event()
    stop = cuda.Event()

    start.record(stream)
    # TRT 10+ uses execute_async_v3
    if not ctx.execute_async_v3(stream.handle):
        raise RuntimeError(f"execute_async_v3 failed at chunk {i+1}")
    stop.record(stream)
    stop.synchronize()

    return start.time_till(stop)  # ms

# -----------------------------
# 5) Warmups
# -----------------------------
print(f"\n[INFO] Starting ({WARMUPS} warmups, {RUNS} runs)...\n")

# Upload input once per iteration
for _ in range(WARMUPS):
    cuda.memcpy_htod_async(d_in0, h_in, stream)
    for i in range(num_chunks):
        src = d_in0 if i == 0 else d_act[i - 1]
        dst = d_act[i]
        _ = run_chunk(i, src, dst, stream)
    # Download final output
    cuda.memcpy_dtoh_async(h_out, d_act[-1], stream)
    stream.synchronize()

# -----------------------------
# 6) Timed runs
# -----------------------------
chunk_times = np.zeros((RUNS, num_chunks), dtype=np.float64)
gpu_totals = np.zeros(RUNS, dtype=np.float64)
cpu_totals = np.zeros(RUNS, dtype=np.float64)

for r in range(RUNS):
    t0 = time.perf_counter()

    cuda.memcpy_htod_async(d_in0, h_in, stream)

    gpu_ms_this_run = 0.0
    for i in range(num_chunks):
        src = d_in0 if i == 0 else d_act[i - 1]
        dst = d_act[i]
        ms = run_chunk(i, src, dst, stream)
        chunk_times[r, i] = ms
        gpu_ms_this_run += ms

    cuda.memcpy_dtoh_async(h_out, d_act[-1], stream)
    stream.synchronize()

    t1 = time.perf_counter()

    gpu_totals[r] = gpu_ms_this_run
    cpu_totals[r] = (t1 - t0) * 1000.0

    print(f"Run {r+1}: GPU {gpu_ms_this_run:.3f} ms | CPU {cpu_totals[r]:.3f} ms")

# -----------------------------
# 7) CSV output
# -----------------------------
csv_path = "chunk_timing.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    header = ["run_id"] + [f"chunk{i+1}_gpu_ms" for i in range(num_chunks)] + ["total_gpu_ms", "total_cpu_ms"]
    writer.writerow(header)
    for r in range(RUNS):
        row = [r + 1] + [f"{chunk_times[r, i]:.6f}" for i in range(num_chunks)] + \
              [f"{gpu_totals[r]:.6f}", f"{cpu_totals[r]:.6f}"]
        writer.writerow(row)

print(f"📝 Saved detailed timing to {csv_path}")
print("✅ Done.")
