#!/bin/bash
# ==============================================================
# Nsight Systems profiling script for sequential_inference.py
# (verified working configuration)
# ==============================================================

set -e  # Exit on error

# --- Create output directory ---
REPORT_DIR="reports"
mkdir -p "$REPORT_DIR"

# --- Timestamp for unique filenames ---
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
REPORT_NAME="sequential_inference_${TIMESTAMP}"
REPORT_PATH="${REPORT_DIR}/${REPORT_NAME}"

echo "=================================================="
echo " 🚀 Profiling sequential_inference.py with Nsight "
echo "--------------------------------------------------"
echo " Output: ${REPORT_PATH}.qdrep"
echo "=================================================="

# --- Run Nsight Systems profile (known working setup) ---
nsys profile \
    -t cuda,nvtx,osrt,cudnn \
    -o "${REPORT_PATH}" \
    python3 sequential_inference.py

echo ""
echo "✅ Profiling complete!"
echo "--------------------------------------------------"

# --- List generated report files ---
if ls ${REPORT_PATH}.* 1> /dev/null 2>&1; then
    echo "Files generated:"
    ls -lh ${REPORT_PATH}.*
else
    echo "⚠️ No report files found. Check nsys output above for errors."
fi

# --- Move chunk timing CSV (if exists) ---
if [ -f "chunk_timing.csv" ]; then
    mv chunk_timing.csv "${REPORT_PATH}_chunk_timing.csv"
    echo " - Timing CSV saved as ${REPORT_PATH}_chunk_timing.csv"
fi

echo "--------------------------------------------------"

# --- Launch Nsight Systems GUI automatically ---
if command -v nsight-sys &> /dev/null; then
    echo "Opening Nsight Systems GUI..."
    nsight-sys "${REPORT_PATH}.qdrep" &
    echo "💡 GUI launched. You can continue using this terminal."
else
    echo "⚠️ Nsight Systems GUI not found in PATH."
    echo "   Open manually later using:"
    echo "   nsight-sys ${REPORT_PATH}.qdrep"
fi

echo "=================================================="
echo "✅ Done — report ready at: ${REPORT_PATH}.qdrep"
echo "=================================================="
