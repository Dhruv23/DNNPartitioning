#!/bin/bash
# =========================================================
# Build TensorRT engines for all ResNet-50 ONNX chunks
# =========================================================

# Directory containing your ONNX files
ONNX_DIR="onnx_chunks"

# Output directory for .engine files
ENGINE_DIR="engines"
mkdir -p "$ENGINE_DIR"

# TensorRT binary path (adjust if needed)
TRTEXEC="trtexec"

# FP16 precision flag
PRECISION="--fp16"

echo "==========================================="
echo " Building TensorRT engines from ONNX chunks"
echo "==========================================="

# Iterate through each ONNX file
for ONNX_FILE in "$ONNX_DIR"/*.onnx; do
    NAME=$(basename "$ONNX_FILE" .onnx)
    ENGINE_FILE="$ENGINE_DIR/${NAME}.engine"
    echo "[INFO] Building engine for $ONNX_FILE ..."
    $TRTEXEC --onnx="$ONNX_FILE" --saveEngine="$ENGINE_FILE" $PRECISION
    echo "       Saved: $ENGINE_FILE"
    echo "-------------------------------------------"
done

echo "✅ All engines built under $ENGINE_DIR/"
