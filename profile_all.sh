#!/bin/bash
# ==============================================================
# profile_all.sh
# --------------------------------------------------------------
# Full profiling pipeline:
#   1. Deletes old reports (ensures clean run)
#   2. Profiles monolithic, GPU-chained, and pinned-memory runs
#   3. Collects Nsight Systems .qdrep + CSV outputs
#   4. Invokes compare_overhead.py for analysis
# ==============================================================

set -e  # Exit on any error

REPORT_DIR="reports"
mkdir -p "$REPORT_DIR"

# ---------------------------------------------------------------
# 🧹 Clean old reports (keep directory, remove files)
# ---------------------------------------------------------------
echo "=============================================================="
echo " 🧹 Cleaning old reports in: ${REPORT_DIR}/"
echo "=============================================================="
rm -f ${REPORT_DIR}/*.qdrep ${REPORT_DIR}/*.csv ${REPORT_DIR}/*.nsys-rep 2>/dev/null || true
echo "✅ Cleared old profiling data."

# ---------------------------------------------------------------
# Timestamp helper
# ---------------------------------------------------------------
timestamp() { date +"%Y-%m-%d_%H-%M-%S"; }

echo "=============================================================="
echo " 🚀 Starting full profiling pipeline "
echo "=============================================================="

# ---------------------------------------------------------------
# 1️⃣  Profile monolithic run (run_resnet50.py)
# ---------------------------------------------------------------
TS=$(timestamp)
BASE_NAME="resnet50_${TS}"
OUT_PATH="${REPORT_DIR}/${BASE_NAME}"

echo ""
echo "▶ Profiling monolithic baseline: run_resnet50.py"
echo "--------------------------------------------------------------"
nsys profile -t cuda,nvtx,osrt,cudnn -o "${OUT_PATH}" python3 run_resnet50.py
echo "✅ Done: ${OUT_PATH}.qdrep"

if [ -f "inference_timing.csv" ]; then
    mv inference_timing.csv "${OUT_PATH}_inference_timing.csv"
    echo " - Saved timing CSV: ${OUT_PATH}_inference_timing.csv"
fi

# ---------------------------------------------------------------
# 2️⃣  Profile sequential GPU-chained run
# ---------------------------------------------------------------
TS=$(timestamp)
BASE_NAME="sequential_inference_gpu_${TS}"
OUT_PATH="${REPORT_DIR}/${BASE_NAME}"

echo ""
echo "▶ Profiling sequential GPU-chained: sequential_inference_gpu.py"
echo "--------------------------------------------------------------"
nsys profile -t cuda,nvtx,osrt,cudnn -o "${OUT_PATH}" python3 sequential_inference_gpu.py
echo "✅ Done: ${OUT_PATH}.qdrep"

if [ -f "chunk_timing.csv" ]; then
    mv chunk_timing.csv "${OUT_PATH}_chunk_timing.csv"
    echo " - Saved timing CSV: ${OUT_PATH}_chunk_timing.csv"
fi

# ---------------------------------------------------------------
# 3️⃣  Profile sequential pinned-memory run
# ---------------------------------------------------------------
TS=$(timestamp)
BASE_NAME="sequential_inference_pinned_${TS}"
OUT_PATH="${REPORT_DIR}/${BASE_NAME}"

echo ""
echo "▶ Profiling sequential pinned-memory: sequential_inference_pinned.py"
echo "--------------------------------------------------------------"
nsys profile -t cuda,nvtx,osrt,cudnn -o "${OUT_PATH}" python3 sequential_inference_pinned.py
echo "✅ Done: ${OUT_PATH}.qdrep"

if [ -f "chunk_timing.csv" ]; then
    mv chunk_timing.csv "${OUT_PATH}_chunk_timing.csv"
    echo " - Saved timing CSV: ${OUT_PATH}_chunk_timing.csv"
fi

# ---------------------------------------------------------------
# 4️⃣  Run overhead comparison
# ---------------------------------------------------------------
echo ""
echo "=============================================================="
echo " 📊 Running segmentation overhead analysis (compare_overhead.py)"
echo "=============================================================="
python3 compare_overhead.py

echo ""
echo "=============================================================="
echo " ✅ All profiling complete — reports saved in: ${REPORT_DIR}/"
echo "=============================================================="
