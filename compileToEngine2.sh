#!/bin/bash
# =========================================================
# Build TensorRT engines for all ResNet-50 ONNX chunks
# =========================================================

# Directory containing your ONNX files (recursively processed)
ONNX_DIR="onnx_chunks2"

# Output directory for .engine files
ENGINE_DIR="engines2"
mkdir -p "$ENGINE_DIR"

# TensorRT binary path (adjust if needed)
TRTEXEC="trtexec"

# FP16 precision flag
PRECISION="--fp16"

echo "==========================================="
echo " Building TensorRT engines from ONNX chunks"
echo "==========================================="

# Iterate through each ONNX file in the directory and its subdirectories
find "$ONNX_DIR" -type f -name "*.onnx" | while read ONNX_FILE; do
    # Extract file name without extension
    NAME=$(basename "$ONNX_FILE" .onnx)
    
    # Define engine output path
    ENGINE_FILE="$ENGINE_DIR/$(basename $(dirname "$ONNX_FILE"))_${NAME}.engine"
    
    # Ensure subdirectories exist for engines2
    mkdir -p "$(dirname "$ENGINE_FILE")"
    
    echo "[INFO] Building engine for $ONNX_FILE ..."
    
    # Run TensorRT to build engine
    $TRTEXEC --onnx="$ONNX_FILE" --saveEngine="$ENGINE_FILE" $PRECISION
    
    echo "       Saved: $ENGINE_FILE"
    echo "-------------------------------------------"
done

echo "✅ All engines built under $ENGINE_DIR/"
