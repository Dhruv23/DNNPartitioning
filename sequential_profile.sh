#!/bin/bash
# ==============================================================
# Nsight Systems profiling script for both sequential inference modes
# - Runs both: sequential_inference_gpu.py and sequential_inference_pinned.py
# - Saves timestamped Nsight Systems reports and timing CSVs
# ==============================================================

set -e  # Exit on error

# --- Configuration ---
REPORT_DIR="reports"
mkdir -p "$REPORT_DIR"

SCRIPTS=("sequential_inference_gpu.py" "sequential_inference_pinned.py")

# --- Helper: run Nsight profiling for a given script ---
run_profile () {
    local script_name="$1"
    local base_name
    base_name=$(basename "$script_name" .py)
    local timestamp
    timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
    local report_name="${base_name}_${timestamp}"
    local report_path="${REPORT_DIR}/${report_name}"

    echo "=================================================="
    echo " 🚀 Profiling ${script_name} with Nsight Systems "
    echo "--------------------------------------------------"
    echo " Output: ${report_path}.qdrep"
    echo "=================================================="

    # --- Run Nsight Systems ---
    nsys profile \
        -t cuda,nvtx,osrt,cudnn \
        -o "${report_path}" \
        python3 "${script_name}"

    echo ""
    echo "✅ Profiling complete for ${script_name}"
    echo "--------------------------------------------------"

    # --- List generated report files ---
    if ls "${report_path}".* 1> /dev/null 2>&1; then
        echo "Files generated:"
        ls -lh "${report_path}".*
    else
        echo "⚠️ No report files found for ${script_name}. Check nsys output above."
    fi

    # --- Move chunk timing CSV if it exists ---
    if [ -f "chunk_timing.csv" ]; then
        mv chunk_timing.csv "${report_path}_chunk_timing.csv"
        echo " - Timing CSV saved as ${report_path}_chunk_timing.csv"
    fi

    echo "--------------------------------------------------"

    # --- Optionally open Nsight Systems GUI ---
    if command -v nsight-sys &> /dev/null; then
        echo "Opening Nsight Systems GUI for ${script_name}..."
        nsight-sys "${report_path}.qdrep" &
        echo "💡 GUI launched in background."
    else
        echo "⚠️ Nsight Systems GUI not found in PATH."
        echo "   Open manually later with:"
        echo "   nsight-sys ${report_path}.qdrep"
    fi

    echo "=================================================="
    echo "✅ Done — report ready at: ${report_path}.qdrep"
    echo "=================================================="
    echo ""
}

# --- Main loop over both scripts ---
for script in "${SCRIPTS[@]}"; do
    if [ -f "$script" ]; then
        run_profile "$script"
    else
        echo "⚠️ Skipping ${script}: file not found."
    fi
done

echo ""
echo "=================================================="
echo "🎯 All profiling runs complete!"
echo "Reports and CSVs saved under: ${REPORT_DIR}/"
echo "=================================================="
