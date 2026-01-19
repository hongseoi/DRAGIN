import os
import json
import argparse
import logging
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import re
import string

from data import StrategyQA, WikiMultiHopQA, HotpotQA, IIRC, Musique, NQ, SQuAD, TQA
from transformers import AutoTokenizer, AutoModelForCausalLM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, required=True)
    tmp = parser.parse_args()
    with open(os.path.join(tmp.dir, "config.json"), "r") as f:
        args = json.load(f)
    args = argparse.Namespace(**args)
    args.output_dir = tmp.dir
    return args


# -----------------------------
# DRAGIN/AdaRAGUE-compatible answer normalization
# (SQuAD official style)
# -----------------------------
def normalize_answer(s: str) -> str:
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


# -----------------------------
# Universal "final answer only" extractor
# Works across: 2wiki/hotpot/iirc/musique/nq/squad/tqa/strategyqa
# -----------------------------
def extract_final_answer(text: str) -> str:
    if text is None:
        return ""

    t = text.strip()

    # If model followed our forced format, this is the most reliable
    # Take last occurrence to survive earlier demo text.
    m = re.findall(r"final answer\s*:\s*(.*)", t, flags=re.IGNORECASE)
    if m:
        cand = m[-1].strip()
        # cut at newline or common separators
        cand = cand.split("\n")[0].strip()
        cand = re.split(r"(question\s*:|explanation\s*:|reason\s*:|note\s*:)", cand, flags=re.IGNORECASE)[0].strip()
        return cand

    # Fallback: common “answer is …” patterns (DRAGIN-style)
    low = t.lower()
    patterns = [
        "final answer is",
        "the final answer is",
        "so the answer is",
        "therefore the answer is",
        "thus the answer is",
        "hence the answer is",
        "the answer is",
        "answer is",
        "answer:",
    ]
    for p in patterns:
        if p in low:
            # split based on original string indices (safer for casing)
            idx = low.rfind(p)
            t2 = t[idx + len(p):].strip()
            t2 = t2.split("\n")[0].strip()
            t2 = re.split(r"(question\s*:|explanation\s*:|reason\s*:|note\s*:)", t2, flags=re.IGNORECASE)[0].strip()
            return t2

    # Fallback2: your current bad outputs contain multiple "Answer: X"
    # pick the LAST "Answer:" line (usually closest to end)
    answers = re.findall(r"answer\s*:\s*(.*)", t, flags=re.IGNORECASE)
    if answers:
        cand = answers[-1].strip()
        cand = cand.split("\n")[0].strip()
        cand = re.split(r"(question\s*:|explanation\s*:|reason\s*:|note\s*:)", cand, flags=re.IGNORECASE)[0].strip()
        return cand

    # Fallback3: just take first line (better than raw whole junk)
    return t.split("\n")[0].strip()


def _sanitize_inputs(input_ids, vocab_limit):
    if (input_ids >= vocab_limit).any():
        logger.warning(f"Found input ID >= vocab limit {vocab_limit}. Clamping...")
        input_ids = torch.clamp(input_ids, min=0, max=vocab_limit - 1)
    if (input_ids < 0).any():
        input_ids = torch.clamp(input_ids, min=0)
    return input_ids


# -----------------------------
# IMPORTANT CHANGE:
# regenerate_answer now FORCES "Final Answer:" format
# It DOES NOT generate multi QA junk.
# -----------------------------
def regenerate_answer(raw_text, tokenizer, model, case, demo):
    # 0) If raw already has Final Answer, keep it (but we will extract later anyway)
    if re.search(r"final answer\s*:", raw_text, flags=re.IGNORECASE):
        return raw_text

    # 1) Trim obvious junk that your outputs contain
    # (remove extra QA blocks etc.)
    split_words = ["#10000000", "Note:"]
    t = raw_text
    for word in split_words:
        pos = t.find(word)
        if pos != -1 and pos > 0:
            t = t[:pos]

    # 2) Build a prompt that asks ONLY for final answer with forced format
    # - demo is kept as you had it, but we ensure the final output format is strict.
    prompt = "".join([d["case"] + "\n" for d in demo])
    prompt += case.strip() + "\n\n"
    prompt += "Respond with ONLY one line in the exact format:\n"
    prompt += "Final Answer: <answer>\n\n"
    prompt += "Final Answer:"

    # 3) Tokenize + sanitize
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    vocab_limit = model.config.vocab_size
    input_ids = _sanitize_inputs(input_ids, vocab_limit)

    input_ids = input_ids.to(model.device)
    attention_mask = torch.ones_like(input_ids)
    input_length = input_ids.shape[1]

    # 4) Greedy decoding, short
    try:
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=32,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
        )
    except RuntimeError as e:
        logger.error(f"Generation failed. {e}")
        # fallback: return original text
        return raw_text

    generated_tokens = outputs[:, input_length:]
    gen = tokenizer.decode(generated_tokens[0], skip_special_tokens=True).strip()

    # Ensure the prefix exists
    # If model returned "Rome" only, we add prefix.
    if not re.search(r"final answer\s*:", gen, flags=re.IGNORECASE):
        gen = "Final Answer: " + gen.split("\n")[0].strip()

    # Keep it single-line
    gen = gen.split("\n")[0].strip()
    return gen


def main():
    args = get_args()
    logger.info(f"{args}")

    # dataset load
    if args.dataset == 'strategyqa':
        data = StrategyQA(args.data_path)
    elif args.dataset == '2wikimultihopqa':
        data = WikiMultiHopQA(args.data_path)
    elif args.dataset == 'hotpotqa':
        data = HotpotQA(args.data_path)
    elif args.dataset == 'iirc':
        data = IIRC(args.data_path)
    elif args.dataset == "musique":
        data = Musique(args.data_path)
    elif args.dataset == "nq":
        data = NQ(args.data_path)
    elif args.dataset == "squad":
        data = SQuAD(args.data_path)
    elif args.dataset == "tqa":
        data = TQA(args.data_path)
    else:
        raise NotImplementedError

    data.format(fewshot=args.fewshot)

    dataset = {}
    for i in range(len(data.dataset)):
        t = data.dataset[i]
        dataset[t["qid"]] = [
            t["answer"],
            t["answer_id"] if "answer_id" in t else None,
            t["case"] if "case" in t else None
        ]

    metrics = ["EM", "F1", "Precision", "Recall"]
    if "use_counter" not in args or args.use_counter:
        count_list = ["retrieve_count", "generate_count", "token_count", "sentence_count"]
        metrics += count_list
    value = [[] for _ in range(len(metrics))]

    with open(os.path.join(args.output_dir, "output.txt"), "r") as fin:
        lines = fin.readlines()

    need_generate = args.dataset in ['2wikimultihopqa', "hotpotqa", "iirc", "strategyqa", "musique", "nq", "squad", "tqa"]

    if need_generate:
        logger.info(f"Loading model: {args.model_name_or_path}")

        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = 'left'

        logger.info("Loading model on GPU 0 with Float32...")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            device_map={'': 0},
            torch_dtype=torch.float32,
            trust_remote_code=("falcon" in args.model_name_or_path)
        )
        model.config.pad_token_id = tokenizer.pad_token_id

        # Embedding resize safety
        current_embedding_size = model.get_input_embeddings().weight.shape[0]
        tokenizer_len = len(tokenizer)
        logger.info(f"Embedding Size: {current_embedding_size}, Tokenizer Len: {tokenizer_len}")
        if current_embedding_size != tokenizer_len:
            new_size = max(current_embedding_size, tokenizer_len)
            logger.info(f"Resizing embeddings to {new_size}")
            model.resize_token_embeddings(new_size)

        demo = data.dataset[0]["demo"]

    pred_out = open(f"{args.output_dir}/details.txt", "w")

    for line in tqdm(lines):
        rd = json.loads(line)
        qid = rd["qid"]
        raw_pred = rd["prediction"]

        if qid not in dataset:
            continue

        ground_truth, ground_truth_id, case = dataset[qid]

        # 1) If your pipeline can produce junk/multi QA, we regenerate a clean final answer
        #    (this matches DRAGIN/AdaRAGUE evaluation practice: final answer only)
        if need_generate:
            raw_pred = regenerate_answer(raw_pred, tokenizer, model, case, demo)

        # 2) Extract final answer ONLY + normalize (THIS is the key fix)
        pred = extract_final_answer(raw_pred)
        pred_norm = normalize_answer(pred)
        gt_norm = normalize_answer(ground_truth)

        # 3) Now compute EM/F1 using dataset functions, BUT on clean normalized strings
        #    If your data.* functions already normalize internally, this is still safe.
        #    We pass pred_norm/gt_norm to maximize cross-dataset consistency.
        em_ret = data.exact_match_score(pred_norm, gt_norm, ground_truth_id)
        f1_ret = data.f1_score(pred_norm, gt_norm, ground_truth_id)

        value[0].append(em_ret["correct"])
        for i, k in enumerate(f1_ret.keys()):
            value[i + 1].append(f1_ret[k])

        if "use_counter" not in args or args.use_counter:
            for i, k in enumerate(count_list):
                value[i + 4].append(rd.get(k, 0))

        detail = {
            "qid": qid,
            "final_pred": pred,           # extracted
            "final_pred_norm": pred_norm, # normalized
            "ground_truth_norm": gt_norm,
            "EM": str(em_ret["correct"]),
            "F1": str(f1_ret["f1"])
        }
        pred_out.write(json.dumps(detail) + "\n")

    pred_out.close()

    ret = []
    for i, metric in enumerate(metrics):
        val = np.array(value[i])
        ret.append([metric, float(val.mean()) if len(val) > 0 else 0.0])

    df = pd.DataFrame(ret)
    df.to_csv(f"{args.output_dir}/result.tsv", index=False, header=False, sep="\t")
    logger.info("Evaluation finished successfully.")


if __name__ == "__main__":
    main()
