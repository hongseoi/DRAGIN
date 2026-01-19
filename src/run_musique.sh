#!/bin/bash

BASE_DIR="/home/seoi0215/trend/DRAGIN"
SRC_DIR="$BASE_DIR/src"
CONFIG_DIR="$BASE_DIR/config/Llama-3.1-8B-Instruct/Musique"
GPU=0

METHODS=("BasicRAG" "DRAGIN" "FLARE" "SingleRAG")

cd "$SRC_DIR" || exit 1

for method in "${METHODS[@]}"; do
    config_path="$CONFIG_DIR/${method}.json"

    echo "=================================================="
    echo "🚀 Running Musique | $method on GPU $GPU"
    echo "Config: $config_path"
    echo "=================================================="

    CUDA_VISIBLE_DEVICES=$GPU \
    python main.py -c "$config_path"

    # output_dir 읽기
    out_dir=$(python - <<EOF
import json
with open("$config_path") as f:
    print(json.load(f)["output_dir"])
EOF
)

    abs_out_dir=$(realpath "$SRC_DIR/$out_dir")

    if [ ! -d "$abs_out_dir" ]; then
        echo "❌ Output dir not found: $abs_out_dir"
        continue
    fi

    # 최신 run 찾기
    latest_run=$(ls -td "$abs_out_dir"/* | head -n 1)

    echo "📊 Evaluating latest run: $latest_run"

    CUDA_VISIBLE_DEVICES=$GPU \
    python evaluation.py --dir "$latest_run"

done

echo "=================================================="
echo "✅ All Musique jobs finished"
echo "=================================================="

echo "📄 Collecting summary table..."

python collect_eval_results.py

echo "📄 Summary saved to summary_latest.csv"
