import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import time
from torch.cuda import nvtx  # <-- add NVTX for timeline markers

TRT_LOGGER = trt.Logger(trt.Logger.INFO)

# Load TensorRT engine
with open("resnet50.engine", "rb") as f:
    engine = trt.Runtime(TRT_LOGGER).deserialize_cuda_engine(f.read())

context = engine.create_execution_context()
input_shape = engine.get_binding_shape(0)
output_shape = engine.get_binding_shape(1)

# Allocate host/GPU buffers
h_input = np.random.randn(*input_shape).astype(np.float32)
h_output = np.empty(output_shape, dtype=np.float32)
d_input = cuda.mem_alloc(h_input.nbytes)
d_output = cuda.mem_alloc(h_output.nbytes)
stream = cuda.Stream()

def infer_once():
    cuda.memcpy_htod_async(d_input, h_input, stream)
    start_event = cuda.Event()
    end_event = cuda.Event()
    start_event.record(stream)

    context.execute_v2([int(d_input), int(d_output)])

    end_event.record(stream)
    stream.synchronize()
    cuda.memcpy_dtoh_async(h_output, d_output, stream)
    stream.synchronize()
    return start_event.time_till(end_event)  # GPU time (ms)

# Run several iterations with NVTX markers
times = []
for i in range(50):
    label = f"Inference {i+1}"
    nvtx.range_push(label)  # Start NVTX range
    t0 = time.time()
    gpu_ms = infer_once()
    cpu_ms = (time.time() - t0) * 1000
    nvtx.range_pop()  # End NVTX range
    times.append(cpu_ms)
    print(f"{label}: CPU total {cpu_ms:.3f} ms, GPU kernel {gpu_ms:.3f} ms")

print(f"\nMean CPU latency: {np.mean(times):.3f} ± {np.std(times):.3f} ms")
