import os
import pandas as pd

BASE_DIR = "../result"

rows = []

def get_latest_run_dir(exp_dir):
    subdirs = [d for d in os.listdir(exp_dir) if os.path.isdir(os.path.join(exp_dir, d))]
    if not subdirs:
        return None

    # 숫자 폴더만 필터링
    numeric_dirs = [d for d in subdirs if d.isdigit()]
    if numeric_dirs:
        return max(numeric_dirs, key=lambda x: int(x))
    
    # fallback: 수정 시간 기준
    return max(subdirs, key=lambda d: os.path.getmtime(os.path.join(exp_dir, d)))

for exp_name in os.listdir(BASE_DIR):
    exp_dir = os.path.join(BASE_DIR, exp_name)
    if not os.path.isdir(exp_dir):
        continue

    latest_run = get_latest_run_dir(exp_dir)
    if latest_run is None:
        continue

    result_path = os.path.join(exp_dir, latest_run, "result.tsv")
    if not os.path.exists(result_path):
        continue

    # exp_name 파싱
    # 예: llama-3.1-8B_musique_single-retrieval
    parts = exp_name.split("_")
    if len(parts) < 3:
        continue

    model = parts[0]
    dataset = parts[1]
    method = "_".join(parts[2:])

    # comma / tsv 자동 처리
    with open(result_path) as f:
        first_line = f.readline()
        sep = "," if "," in first_line else "\t"

    df = pd.read_csv(result_path, sep=sep, header=None, names=["metric", "value"])

    row = {
        "model": model,
        "dataset": dataset,
        "method": method,
        "run": latest_run
    }

    for _, r in df.iterrows():
        row[r["metric"]] = r["value"]

    rows.append(row)

df_all = pd.DataFrame(rows)
df_all = df_all.sort_values(by=["dataset", "method"])

df_all.to_csv("summary_latest.csv", index=False)
print(df_all)
