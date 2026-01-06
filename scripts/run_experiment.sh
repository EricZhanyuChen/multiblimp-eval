#!/bin/bash

# ================= Configuration Area =================
# 1. Target Languages
# eng: English, heb: Hebrew, ita: Italian, 
# rus: Russian, wol: Wolof, sah: Yakut
LANGS=("eng" "heb" "ita" "rus" "wol" "sah")

# 2. Target Models
MODELS=("google/gemma-3-1b-pt" "google/gemma-3-4b-pt" "google/gemma-3-12b-pt")
# "google/gemma-3-1b-it" "google/gemma-3-4b-it" "google/gemma-3-12b-it" 

# 3. Target Methods
# Added 'native_prompt' as you updated the python code
METHODS=("direct" "prompt" "native_prompt" "meta")

# 4. Path Settings
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPTS_DIR="$BASE_DIR/scripts"
ALL_RESULTS_ROOT="$BASE_DIR/results"

# ================= Versioning Logic =================
RUN_NUM=1
while [ -d "$ALL_RESULTS_ROOT/run_$RUN_NUM" ]; do
  RUN_NUM=$((RUN_NUM + 1))
done

CURRENT_OUTPUT_DIR="$ALL_RESULTS_ROOT/run_$RUN_NUM"
mkdir -p "$CURRENT_OUTPUT_DIR"

echo "========================================================"
echo "STARTING FINAL FULL RUN"
echo "Run ID:       #$RUN_NUM"
echo "Languages:    ${LANGS[*]}"
echo "Models:       ${MODELS[*]}"
echo "Methods:      ${METHODS[*]}"
echo "Precision:    bfloat16 (Fast & Stable)"
echo "Save Path:    $CURRENT_OUTPUT_DIR"
echo "========================================================"

# ================= Execution Loop =================
for model in "${MODELS[@]}"; do
    for lang in "${LANGS[@]}"; do
        for method in "${METHODS[@]}"; do
            echo -e "\n[RUN] Model: $model | Lang: $lang | Method: $method"
            
            # Using bfloat16 for stability and speed
            # Using batch_size 16 because A100 can easily handle it in bfloat16
            python "$SCRIPTS_DIR/eval_tool_new.py" \
                --model "$model" \
                --lang "$lang" \
                --method "$method" \
                --batch_size "16" \
                --dtype "bfloat16" \
                --output_dir "$CURRENT_OUTPUT_DIR"
            
            if [ $? -ne 0 ]; then
                echo "[ERROR] Execution failed for: $model - $lang - $method"
            fi
        done
    done
done

# ================= Summary & Cleanup =================
echo -e "\n[*] Generating Final Summary Report..."
python "$SCRIPTS_DIR/generate_summary_table.py" --target_dir "$CURRENT_OUTPUT_DIR"

CSV_FILE="$CURRENT_OUTPUT_DIR/final_summary.csv"
if [ -f "$CSV_FILE" ]; then
    echo -e "\n>>> Final Run Results (Run #$RUN_NUM) <<<"
    echo "--------------------------------------------------------"
    # Try to display comfortably, fallback to cat if column command missing
    if command -v column &> /dev/null; then
        column -t -s, "$CSV_FILE"
    else
        cat "$CSV_FILE"
    fi
    echo "--------------------------------------------------------"
fi

# Compress
ZIP_NAME="full_run_${RUN_NUM}_$(date +%Y%m%d).tar.gz"
echo -e "\n[*] Compressing results to $ZIP_NAME ..."
tar -czvf "$BASE_DIR/$ZIP_NAME" -C "$ALL_RESULTS_ROOT" "run_$RUN_NUM" > /dev/null

echo "========================================================"
echo "ALL DONE! Go get a coffee ☕"
echo "========================================================"