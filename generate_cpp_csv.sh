#!/bin/bash
# ==============================================================
# profile_all.sh
# --------------------------------------------------------------
# Runs all compiled TensorRT C++ executables sequentially:
#   1. run_trt_naive      (naive CPU-controlled baseline)
#   2. run_trt_gpu        (sequential GPU chaining)
#   3. run_trt_gpu_fused  (single-buffer fused chaining)
#   4. run_trt_graph      (CUDA Graph optimized)
#   5. run_trt_pinned     (pinned host memory)
# Saves each run's CSV timing file into ./reports/
# ==============================================================

set -e  # Exit on any error

REPORT_DIR="reports"
mkdir -p "$REPORT_DIR"

# ---------------------------------------------------------------
# 🧹 Clean old reports (keep directory)
# ---------------------------------------------------------------
echo "=============================================================="
echo " 🧹 Cleaning old reports in: ${REPORT_DIR}/"
echo "=============================================================="
rm -f ${REPORT_DIR}/*.csv 2>/dev/null || true
echo "✅ Cleared old CSV files."

# ---------------------------------------------------------------
# Timestamp helper
# ---------------------------------------------------------------
timestamp() { date +"%Y-%m-%d_%H-%M-%S"; }

echo ""
echo "=============================================================="
echo " 🚀 Starting full TensorRT timing run "
echo "=============================================================="

# ---------------------------------------------------------------
# Helper to run a binary and move its CSV
# ---------------------------------------------------------------
run_and_store() {
    local BIN="$1"
    local LABEL="$2"
    local TS
    TS=$(timestamp)
    local OUT_PATH="${REPORT_DIR}/${BIN}_${TS}.csv"

    echo ""
    echo "▶ Running ${LABEL}: ./${BIN}"
    echo "--------------------------------------------------------------"
    if ! ./${BIN}; then
        echo "❌ ${BIN} failed!"
        exit 1
    fi
    echo "✅ Done: ${BIN}"

    if [ -f "chunk_timing.csv" ]; then
        mv chunk_timing.csv "${OUT_PATH}"
        echo " - Saved timing CSV: ${OUT_PATH}"
    elif [ -f "chunk_timing_min.csv" ]; then
        mv chunk_timing_min.csv "${OUT_PATH}"
        echo " - Saved timing CSV: ${OUT_PATH}"
    else
        echo "⚠️  Warning: no timing CSV found after ${BIN}"
    fi
}

# ---------------------------------------------------------------
# 1️⃣  Naive baseline
# ---------------------------------------------------------------
run_and_store "run_trt_naive" "naive baseline"

# ---------------------------------------------------------------
# 2️⃣  Sequential GPU-chained version
# ---------------------------------------------------------------
run_and_store "run_trt_gpu" "sequential GPU-chained"

# ---------------------------------------------------------------
# 3️⃣  Fused single-buffer GPU chaining
# ---------------------------------------------------------------
run_and_store "run_trt_gpu_fused" "fused GPU single-buffer chaining"

# ---------------------------------------------------------------
# 4️⃣  CUDA Graph optimized version
# ---------------------------------------------------------------
run_and_store "run_trt_graph" "CUDA Graph optimized"

# ---------------------------------------------------------------
# 5️⃣  Pinned-memory version
# ---------------------------------------------------------------
run_and_store "run_trt_pinned" "pinned-memory version"

# ---------------------------------------------------------------
# ✅ Summary
# ---------------------------------------------------------------
echo ""
echo "=============================================================="
echo " ✅ All runs complete — CSVs saved in: ${REPORT_DIR}/"
echo "=============================================================="
ls -lh "${REPORT_DIR}"/*.csv || true
