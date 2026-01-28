#!/bin/bash

# Stop script execution if any command fails
set -e

echo "========================================"
echo "🔨 Compiling Standard Inference (Contiguous)..."
echo "========================================"
nvcc -std=c++17 -O2 sequential_inference_gpu.cpp -o run_trt_gpu \
    -I/usr/include/x86_64-linux-gnu \
    -I/usr/local/cuda/include \
    -L/usr/lib/x86_64-linux-gnu \
    -L/usr/local/cuda/lib64 \
    -lnvinfer -lnvonnxparser -lcudart

echo "========================================"
echo "🔨 Compiling PagedAttention Inference (vLLM)..."
echo "========================================"
nvcc -std=c++17 -O2 sequential_inference_PagedAttention.cpp -o run_trt_PagedAttention \
    -I/usr/include/x86_64-linux-gnu \
    -I/usr/local/cuda/include \
    -L/usr/lib/x86_64-linux-gnu \
    -L/usr/local/cuda/lib64 \
    -lnvinfer -lnvonnxparser -lcudart

echo "========================================"
echo "🚀 Running Standard Inference..."
echo "========================================"
./run_trt_gpu

echo "========================================"
echo "🚀 Running PagedAttention Inference..."
echo "========================================"
./run_trt_PagedAttention

echo "========================================"
echo "📊 Opening Visualizations in Browser..."
echo "========================================"

# Detect OS and open files accordingly
if command -v xdg-open &> /dev/null; then
    # Linux
    xdg-open memory_layout_standard.html &
    xdg-open memory_layout_paged.html &
elif command -v open &> /dev/null; then
    # MacOS
    open memory_layout_standard.html &
    open memory_layout_paged.html &
else
    echo "⚠️  Could not detect browser opener. Please open:"
    echo "   - memory_layout_standard.html"
    echo "   - memory_layout_paged.html"
fi

echo "✅ Done."