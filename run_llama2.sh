#!/bin/bash

BASE_DIR="/home/seoi0215/trend/DRAGIN"
SRC_DIR="$BASE_DIR/src"
CONFIG_DIR="$BASE_DIR/config/Llama2-7b-chat"
LOG_DIR="$BASE_DIR/logs2"

DATASETS=("HotpotQA" "2WikiMultihopQA" "Musique" "nq" "squad" "tqa")
METHODS=("BasicRAG" "SingleRAG" "DRAGIN" "FLARE" "SAETRAG")

GPU=0

run_job () {
    local dataset=$1
    local method=$2

    config_path="$CONFIG_DIR/$dataset/$method.json"

    log_path="$LOG_DIR/$dataset/$method"
    mkdir -p "$log_path"

    echo "🚀 [$dataset | $method] on GPU $GPU"

    cd $SRC_DIR || exit 1

    # main 실행 (실패 시 중단)
    CUDA_VISIBLE_DEVICES=$GPU \
        python main.py -c $config_path \
        > "$log_path/main.log" 2>&1 || exit 1

    out_dir=$(python - <<EOF
import json
with open("$config_path") as f:
    print(json.load(f)["output_dir"])
EOF
)

    abs_out_dir=$(realpath "$SRC_DIR/$out_dir")
    latest_run=$(ls -td $abs_out_dir/* | head -n 1)

    echo "📊 Evaluating: $latest_run on GPU $GPU"

    # eval 실행
    CUDA_VISIBLE_DEVICES=$GPU \
        python eval.py --dir $latest_run \
        > "$log_path/eval.log" 2>&1 || exit 1
}

for dataset in "${DATASETS[@]}"; do
    for method in "${METHODS[@]}"; do
        run_job "$dataset" "$method"
    done
done

echo "✅ All jobs finished safely on GPU 0."
