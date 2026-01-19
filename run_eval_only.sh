#!/bin/bash

BASE_DIR="/home/seoi0215/trend/DRAGIN"
SRC_DIR="$BASE_DIR/src"
CONFIG_DIR="$BASE_DIR/config/Llama-3.1-8B-Instruct"
LOG_DIR="$BASE_DIR/logs_eval_only"

DATASETS=("HotpotQA" "2WikiMultihopQA" "Musique" "nq" "squad" "tqa")
METHODS=("BasicRAG" "SingleRAG" "DRAGIN" "FLARE" "SAETRAG")
GPUS=(1 2 3)

mkdir -p "$LOG_DIR"

declare -A GPU_PIDS

get_latest_run () {
    local out_dir=$1
    ls -td "$out_dir"/* | head -n 1
}

run_eval_job () {
    local dataset=$1
    local method=$2
    local gpu=$3

    config_path="$CONFIG_DIR/$dataset/$method.json"

    log_path="$LOG_DIR/$dataset/$method"
    mkdir -p "$log_path"

    # config에서 output_dir 읽기
    out_dir=$(python - <<EOF
import json
with open("$config_path") as f:
    print(json.load(f)["output_dir"])
EOF
)

    abs_out_dir=$(realpath "$SRC_DIR/$out_dir")

    if [ ! -d "$abs_out_dir" ]; then
        echo "❌ Output dir not found: $abs_out_dir"
        return
    fi

    latest_run=$(get_latest_run "$abs_out_dir")

    if [ ! -d "$latest_run" ]; then
        echo "❌ No runs found in $abs_out_dir"
        return
    fi

    echo "📊 [EVAL] $dataset | $method | GPU $gpu"
    echo "     → $latest_run"

    cd "$SRC_DIR"

    CUDA_VISIBLE_DEVICES=$gpu \
        python evaluation.py --dir "$latest_run" \
        > "$log_path/eval.log" 2>&1
}

job_queue=()

for dataset in "${DATASETS[@]}"; do
    for method in "${METHODS[@]}"; do
        job_queue+=("$dataset|$method")
    done
done

job_index=0
total_jobs=${#job_queue[@]}

while [ $job_index -lt $total_jobs ]; do
    for gpu in "${GPUS[@]}"; do

        if [ -z "${GPU_PIDS[$gpu]}" ] || ! kill -0 "${GPU_PIDS[$gpu]}" 2>/dev/null; then

            if [ $job_index -ge $total_jobs ]; then
                break
            fi

            IFS="|" read dataset method <<< "${job_queue[$job_index]}"
            job_index=$((job_index + 1))

            run_eval_job "$dataset" "$method" "$gpu" &
            GPU_PIDS[$gpu]=$!

            sleep 2
        fi
    done
    sleep 5
done

wait
echo "✅ All EVAL jobs finished."
