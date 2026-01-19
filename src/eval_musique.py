# output이 cot형태일경우
import os
import json
import argparse
import logging
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from data import StrategyQA, WikiMultiHopQA, HotpotQA, IIRC, Musique, NQ, SQuAD, TQA
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

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

def _sanitize_inputs(input_ids, vocab_limit):
    """입력 ID가 모델의 Vocab 크기를 벗어나지 않도록 안전장치 적용"""
    if (input_ids >= vocab_limit).any():
        logger.warning(f"Found input ID >= vocab limit {vocab_limit}. Clamping...")
        input_ids = torch.clamp(input_ids, min=0, max=vocab_limit - 1)
    if (input_ids < 0).any():
        input_ids = torch.clamp(input_ids, min=0)
    return input_ids

# [추가됨] 긴 문장에서 정답 단어만 추출하는 함수 (EM 점수 해결용)
def extract_answer_only(pred_str):
    # 1. "answer is" (대소문자 무시를 위해 낮춤 처리 가능하나, 일단 패턴대로)
    # 모델이 "So the answer is" 또는 "The answer is" 등을 사용할 때 대응
    if "answer is" in pred_str:
        pred_str = pred_str.split("answer is")[-1]
    
    # 2. 불필요한 뒷부분(마침표, 줄바꿈, Note:, Question: 등) 제거
    # 문장이 끝나는 지점이나 새로운 메타 텍스트가 나오는 지점에서 자름
    for delimiter in ["\n", ".", "Note:", "Question:", "(Note"]:
        if delimiter in pred_str:
            pred_str = pred_str.split(delimiter)[0]
    
    # 3. 앞뒤 공백 제거
    return pred_str.strip()

def regenerate_answer(cot, tokenizer, model, case, demo):
    # 전처리
    split_words = ["Question:", "#10000000", "Note:"]
    for word in split_words:
        pos = cot.find(word)
        if pos != -1 and pos > 0:
            cot = cot[:pos]
    
    # 이미 정답 문구가 있어도 바로 리턴하지 않고, 
    # 일단 여기서 리턴하되 main에서 extract_answer_only가 처리하게 됨.
    if "the answer is" in cot:
        return cot 

    cot += " So the answer is "
    prompt = "".join([d["case"]+"\n" for d in demo])
    prompt += case + " " + cot
    
    # 1. 토큰화
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    
    # 2. 입력값 안전장치
    vocab_limit = model.config.vocab_size
    input_ids = _sanitize_inputs(input_ids, vocab_limit)
    
    input_ids = input_ids.to(model.device)
    input_length = input_ids.shape[1]
    attention_mask = torch.ones_like(input_ids)
    
    # 3. 생성
    try:
        outputs = model.generate(
            input_ids = input_ids, 
            attention_mask = attention_mask, 
            max_new_tokens = 20,
            pad_token_id = tokenizer.pad_token_id,
            do_sample = False
        )
    except RuntimeError as e:
        logger.error(f"Generation failed for prompt: {prompt[:100]}...")
        return cot

    generated_tokens = outputs[:, input_length:]
    text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
    text = cot + text.strip()
    
    for word in split_words:
        pos = text.find(word)
        if pos != -1:
            text = text[:pos] 

    return text


def main():
    args = get_args()
    logger.info(f"{args}")
    
    # 데이터셋 로드
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

        logger.info("Forcing model to load on GPU 0 with Float32...")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path, 
            device_map={'': 0},
            torch_dtype=torch.float32,
            trust_remote_code = "falcon" in args.model_name_or_path
        )
        model.config.pad_token_id = tokenizer.pad_token_id

        current_embedding_size = model.get_input_embeddings().weight.shape[0]
        tokenizer_len = len(tokenizer)
        logger.info(f"Embedding Size: {current_embedding_size}, Tokenizer Len: {tokenizer_len}")
        
        if current_embedding_size != tokenizer_len:
            new_size = max(current_embedding_size, tokenizer_len)
            logger.info(f"Resizing embeddings to {new_size}")
            model.resize_token_embeddings(new_size)

        demo = data.dataset[0]["demo"]

    pred_out = open(f"{args.output_dir}/details.txt", "w")
    
    # 진행 상황 표시
    for line in tqdm(lines):
        rd = json.loads(line)
        qid = rd["qid"]
        pred = rd["prediction"]
        
        if qid not in dataset:
            continue
            
        ground_truth, ground_truth_id, case = dataset[qid]
        
        if need_generate:
            # 1. 모델을 통한 재생성 (여전히 긴 문장일 수 있음)
            pred = regenerate_answer(pred, tokenizer, model, case, demo)
            
            # 2. [핵심 수정] 긴 문장에서 정답만 추출 (EM 점수 정상화)
            pred = extract_answer_only(pred)
        
        pred = data.get_real_prediction(pred)

        em_ret = data.exact_match_score(pred, ground_truth, ground_truth_id)
        f1_ret = data.f1_score(pred, ground_truth, ground_truth_id)
        
        value[0].append(em_ret["correct"])
        for i, k in enumerate(f1_ret.keys()):
            value[i+1].append(f1_ret[k])
            
        if "use_counter" not in args or args.use_counter:
            for i, k in enumerate(count_list):
                if k in rd:
                    value[i+4].append(rd[k])
                else:
                    value[i+4].append(0) 

        detail = {
            "qid": qid, 
            "final_pred": pred,
            "EM": str(em_ret["correct"]), 
            "F1": str(f1_ret["f1"]) 
        }
        pred_out.write(json.dumps(detail)+"\n")

    pred_out.close()

    ret = []
    for i, metric in enumerate(metrics):
        val = np.array(value[i])
        if len(val) > 0:
            ret.append([metric, val.mean()])
        else:
            ret.append([metric, 0.0])
            
    df = pd.DataFrame(ret)
    df.to_csv(f"{args.output_dir}/result.tsv", index=False, header=False)
    logger.info("Evaluation finished successfully.")

if __name__ == "__main__":
    main()
