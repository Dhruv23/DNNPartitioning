#!/bin/bash
# ==============================================================
# profile_all_nsys.sh
# --------------------------------------------------------------
# Profiles all compiled TensorRT C++ executables using Nsight Systems.
#
#   1. Cleans old Nsight reports
#   2. Profiles:
#        - run_trt_naive
#        - run_trt_gpu
#        - run_trt_graph
#        - run_trt_pinned
#   3. Saves .qdrep + .nsys-rep into ./nsys_reports/
#
# View results with:
#   nsys-ui ./nsys_reports/<file>.qdrep
# ==============================================================

set -e  # Exit immediately on any error

REPORT_DIR="nsys_reports"
mkdir -p "$REPORT_DIR"

# ---------------------------------------------------------------
# 🧹 Clean old Nsight reports
# ---------------------------------------------------------------
echo "=============================================================="
echo " 🧹 Cleaning old Nsight reports in: ${REPORT_DIR}/"
echo "=============================================================="
rm -f ${REPORT_DIR}/*.qdrep ${REPORT_DIR}/*.nsys-rep ${REPORT_DIR}/*.sqlite 2>/dev/null || true
echo "✅ Cleared old profiling files."

# ---------------------------------------------------------------
# Timestamp helper
# ---------------------------------------------------------------
timestamp() { date +"%Y-%m-%d_%H-%M-%S"; }

# ---------------------------------------------------------------
# Helper to profile one binary
# ---------------------------------------------------------------
profile_bin() {
    local BIN="$1"
    local LABEL="$2"
    local TS=$(timestamp)
    local OUT_PATH="${REPORT_DIR}/${BIN}_${TS}"

    echo ""
    echo "=============================================================="
    echo " ▶ Profiling ${LABEL}: ${BIN}"
    echo "=============================================================="

    if [ ! -x "./${BIN}" ]; then
        echo "❌ Executable ./${BIN} not found or not executable."
        exit 1
    fi

    # Nsight Systems command
    nsys profile \
        -t cuda,nvtx,osrt,cudnn,cublas \
        --sample=none \
        --force-overwrite=true \
        --export=sqlite \
        -o "${OUT_PATH}" \
        ./${BIN}

    echo "✅ Done: ${OUT_PATH}.qdrep"
}

# ---------------------------------------------------------------
# Profile each TensorRT executable
# ---------------------------------------------------------------
profile_bin "run_trt_naive" "Naive baseline"
profile_bin "run_trt_gpu"   "Sequential GPU-chained"
profile_bin "run_trt_graph" "CUDA Graph optimized"
profile_bin "run_trt_pinned" "Pinned-memory version"

# ---------------------------------------------------------------
# Summary
# ---------------------------------------------------------------
echo ""
echo "=============================================================="
echo " ✅ All Nsight profiles complete — view them in Nsight Systems UI"
echo "=============================================================="
ls -lh "${REPORT_DIR}"/*.qdrep || true
echo ""
echo "💡 Example: nsys-ui ${REPORT_DIR}/run_trt_graph_<timestamp>.qdrep"
