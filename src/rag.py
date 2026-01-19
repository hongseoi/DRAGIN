# =========================
# rag.py
# =========================
import logging
from math import exp
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import spacy

from generate import BasicGenerator
from retriever import BM25, SGPT

logger = logging.getLogger(__name__)
nlp = spacy.load("en_core_web_sm")


class Counter:
    def __init__(self):
        self.retrieve = 0
        self.generate = 0
        self.hallucinated = 0
        self.token = 0
        self.sentence = 0

    def add_generate(self, text, tokenizer):
        self.generate += 1
        ids = tokenizer(text, return_tensors="pt")["input_ids"][0].tolist()
        self.token += len(ids)
        self.sentence += len([s.text for s in nlp(text).sents])

    def calc(self, other: "Counter") -> Dict[str, float]:
        return {
            "retrieve_count": float(self.retrieve - other.retrieve),
            "generate_count": float(self.generate - other.generate),
            "hallucinated_count": float(self.hallucinated - other.hallucinated),
            "token_count": float(self.token - other.token),
            "sentence_count": float(self.sentence - other.sentence),
        }


class BasicRAG:
    def __init__(self, args):
        # args is argparse.Namespace
        for k, v in vars(args).items():
            setattr(self, k, v)

        self.generator = BasicGenerator(
            self.model_name_or_path,
            dtype=getattr(args, "dtype", "bf16"),
            attn_implementation=getattr(args, "attn_implementation", "eager"),
            trust_remote_code=getattr(args, "trust_remote_code", False),
        )

        self.counter = Counter()

        self.retriever_type = None
        self.retriever = None
        if getattr(self, "retriever", None) is not None:
            self.retriever_type = self.retriever
            if self.retriever_type == "BM25":
                self.retriever = BM25(
                    tokenizer=self.generator.tokenizer,
                    index_name=getattr(self, "es_index_name", "wiki"),
                    engine="elasticsearch",
                    text_field=getattr(self, "es_text_field", "text"),
                    es_url=getattr(self, "es_url", "http://localhost:9200"),
                )
            elif self.retriever_type == "SGPT":
                self.retriever = SGPT(
                    model_name_or_path=self.sgpt_model_name_or_path,
                    sgpt_encode_file_path=self.sgpt_encode_file_path,
                    passage_file=self.passage_file,
                )
            else:
                raise NotImplementedError(f"Unknown retriever={self.retriever_type}")

    def retrieve(self, query: str, topk: int = 3, max_query_length: int = 64) -> List[str]:
        if self.retriever is None:
            return []
        self.counter.retrieve += 1
        if self.retriever_type == "BM25":
            _, docs = self.retriever.retrieve([query], topk=topk, max_query_length=max_query_length)
            return docs[0]
        else:
            docs = self.retriever.retrieve([query], topk=topk)
            return docs[0]

    @staticmethod
    def get_top_sentence(text: str) -> str:
        sents = [s.text.strip() for s in nlp(text).sents]
        sents = [s for s in sents if s]
        return sents[0] if sents else ""

    @staticmethod
    def get_last_sentence(text: str) -> str:
        sents = [s.text.strip() for s in nlp(text).sents]
        sents = [s for s in sents if s]
        return sents[-1] if sents else ""

    @staticmethod
    def normalize_answer_span(text: str) -> str:
        """
        평가에서 'the answer is' 형식이 있을 수도 있고 없을 수도 있으니,
        최대한 마지막 짧은 답을 뽑아주는 안전장치.
        """
        t = text.strip()
        low = t.lower()
        key = "the answer is"
        if key in low:
            idx = low.rfind(key)
            t2 = t[idx + len(key) :].strip(" :.\n\t")
            # 너무 길면 첫 문장/첫 줄
            t2 = t2.split("\n")[0].strip()
            # 끝이 길면 첫 문장
            t2s = [s.text.strip() for s in nlp(t2).sents]
            if t2s:
                t2 = t2s[0]
            return t2.strip()
        # fallback: 첫 문장
        s = BasicRAG.get_top_sentence(t)
        return s if s else t


class NoRAG(BasicRAG):
    """
    non-retrieval
    """

    def inference(self, question: str, demo: List[Dict[str, str]], case: str) -> str:
        assert self.query_formulation == "direct"
        prompt = "".join([d["case"] + "\n" for d in demo]) + case
        out = self.generator.generate(prompt, self.generate_max_length)
        if self.use_counter:
            self.counter.add_generate(out.text, self.generator.tokenizer)
        return out.text


class SingleRAG(BasicRAG):
    """
    single-retrieval (retrieve once, then answer)
    """

    def inference(self, question: str, demo: List[Dict[str, str]], case: str) -> str:
        assert self.query_formulation == "direct"
        docs = self.retrieve(question, topk=self.retrieve_topk)

        prompt = "".join([d["case"] + "\n" for d in demo])
        prompt += "Context:\n"
        for i, doc in enumerate(docs):
            prompt += f"[{i+1}] {doc}\n"
        prompt += "Answer in the same format as before.\n"
        prompt += case

        out = self.generator.generate(prompt, self.generate_max_length)
        if self.use_counter:
            self.counter.add_generate(out.text, self.generator.tokenizer)
        return out.text


class TokenRAG(BasicRAG):
    """
    FLARE-ish: logprob 기반 hallucination 감지 후 retrieve
    """

    def modifier(self, text: str, tokens: List[str], logprobs: List[float]) -> Tuple[str, Optional[str], bool]:
        sents = [s.text.strip() for s in nlp(text).sents]
        sents = [s for s in sents if s]

        tid = 0
        for sid, sent in enumerate(sents):
            pos = 0
            tr = tid
            while tr < len(tokens):
                apr = sent[pos:].find(tokens[tr])
                if apr == -1:
                    break
                pos = apr + len(tokens[tr])
                tr += 1

            probs = [1 - exp(v) for v in logprobs[tid : tr + 1]]
            probs = np.array(probs)
            p = {
                "avg": np.mean,
                "max": np.max,
                "min": np.min,
            }.get(self.sentence_solver, np.mean)(probs)

            if p > self.hallucination_threshold:
                prev = "" if sid == 0 else " ".join(sents[:sid])
                curr = sents[sid]
                return prev, curr, True

            tid = tr + 1

        return text, None, False

    def inference(self, question: str, demo: List[Dict[str, str]], case: str) -> str:
        text = ""
        while True:
            old_len = len(text)
            prompt = "".join([d["case"] + "\n" for d in demo]) + case + " " + text

            out = self.generator.generate(prompt, self.generate_max_length, return_logprobs=True)
            if self.use_counter:
                self.counter.add_generate(out.text, self.generator.tokenizer)

            ptext, curr, halluc = self.modifier(out.text, out.tokens, out.logprobs)
            if not halluc:
                text = (text.strip() + " " + out.text.strip()).strip()
            else:
                if self.query_formulation == "direct":
                    retrieve_q = curr
                elif self.query_formulation == "forward_all":
                    retrieve_q = " ".join([x for x in [question, text, ptext] if x])
                else:
                    raise NotImplementedError(self.query_formulation)

                docs = self.retrieve(retrieve_q, topk=self.retrieve_topk)

                prompt2 = "".join([d["case"] + "\n" for d in demo])
                prompt2 += "Context:\n"
                for i, doc in enumerate(docs):
                    prompt2 += f"[{i+1}] {doc}\n"
                prompt2 += "Answer in the same format as before.\n"
                prompt2 += (case + " " + text + " " + ptext).strip()

                out2 = self.generator.generate(prompt2, self.generate_max_length)
                if self.use_counter:
                    self.counter.add_generate(out2.text, self.generator.tokenizer)
                    self.counter.hallucinated += 1

                text = " ".join([x for x in [text.strip(), ptext.strip(), out2.text.strip()] if x])

            if len(self.generator.tokenizer.encode(text)) > self.generate_max_length:
                break
            if len(text) <= old_len:
                break
            if "the answer is" in text.lower():
                break

        return text


class AttnWeightRAG(BasicRAG):
    """
    DRAGIN: entropy(불확실성) + attention 기반으로 hallucination 감지 후 retrieve
    """

    def modifier(
        self,
        text: str,
        tokens: List[str],
        attentions: List[float],
        weight: List[float],
    ) -> Tuple[bool, str, Optional[List[str]], Optional[List[int]]]:
        sents = [s.text.strip() for s in nlp(text).sents]
        sents = [s for s in sents if s]

        tid = 0
        for sid, sent in enumerate(sents):
            tl, tr = tid, tid
            if sid == len(sents) - 1:
                tl, tr = tid, len(tokens)
            else:
                # 토큰 span 근사 매칭
                for i in range(tid + 1, len(tokens) + 1):
                    seq = " ".join(tokens[tl:i])
                    if sent in seq:
                        tr = i
                        break
                tid = tr

            att = attentions[tl:tr]
            if not att:
                continue

            att = np.array(att, dtype=np.float32)
            att = att / (att.sum() + 1e-9)

            w = np.array(weight[tl:tr], dtype=np.float32)
            if len(w) != len(att):
                # 길이 불일치 방지
                m = min(len(w), len(att))
                w = w[:m]
                att = att[:m]

            value = att * w * float(len(att))
            hit = (value > self.hallucination_threshold).astype(int).tolist()
            if 1 in hit:
                prev = "" if sid == 0 else " ".join(sents[:sid])
                return True, prev, tokens[tl:tr], hit

        return False, text, None, None

    def inference(self, question: str, demo: List[Dict[str, str]], case: str) -> str:
        text = ""
        while True:
            old_len = len(text)
            prompt = "".join([d["case"] + "\n" for d in demo])
            prompt += " ".join([x for x in [case, text] if x])

            use_entropy = (self.method == "dragin")
            use_logprob = (self.method == "attn_prob")

            gen_text, toks, attns, logprobs, entropies = self.generator.generate_attn(
                prompt,
                self.generate_max_length,
                solver="last_token",
                use_entropy=use_entropy,
                use_logprob=use_logprob,
            )

            if self.use_counter:
                self.counter.add_generate(gen_text, self.generator.tokenizer)

            # DRAGIN weight: entropy (높을수록 불확실) / attn_prob: -logprob
            if use_entropy and entropies is not None:
                weight = entropies
            elif use_logprob and logprobs is not None:
                weight = [-v for v in logprobs]
            else:
                # fallback
                weight = [0.0 for _ in toks]

            halluc, ptext, curr_tokens, curr_hit = self.modifier(gen_text, toks, attns, weight)
            if not halluc:
                text = (text.strip() + " " + gen_text.strip()).strip()
            else:
                forward_all = " ".join([x for x in [question, text, ptext] if x])

                if self.query_formulation == "current":
                    retrieve_q = " ".join(curr_tokens)
                elif self.query_formulation == "forward_all":
                    retrieve_q = forward_all
                elif self.query_formulation == "last_sentence":
                    retrieve_q = self.get_last_sentence(forward_all)
                else:
                    # 가장 보수적으로 forward_all
                    retrieve_q = forward_all

                docs = self.retrieve(retrieve_q, topk=self.retrieve_topk)

                prompt2 = "".join([d["case"] + "\n" for d in demo])
                prompt2 += "Context:\n"
                for i, doc in enumerate(docs):
                    prompt2 += f"[{i+1}] {doc}\n"
                prompt2 += "Answer in the same format as before.\n"
                prompt2 += " ".join([x for x in [case, text, ptext] if x]).strip()

                out2 = self.generator.generate(prompt2, self.generate_max_length)
                if self.use_counter:
                    self.counter.add_generate(out2.text, self.generator.tokenizer)
                    self.counter.hallucinated += 1

                first = self.get_top_sentence(out2.text)
                text = " ".join([x for x in [text.strip(), ptext.strip(), first.strip()] if x]).strip()

            if len(self.generator.tokenizer.encode(text)) > self.generate_max_length:
                break
            if len(text) <= old_len:
                break
            if "the answer is" in text.lower():
                break

        return text


# =========================
# SAE (optional)
# =========================
class SAETriggerRAG(BasicRAG):
    def __init__(self, args):
        super().__init__(args)
        try:
            import joblib
            import pandas as pd
            import torch.nn as nn
            import torch.nn.functional as F
            from transformers import StoppingCriteria, StoppingCriteriaList
            from sae_lens import SAE
        except Exception as e:
            raise RuntimeError(
                "SAETriggerRAG requires: sae_lens, pandas, joblib, torch. "
                f"Import error: {e}"
            )

        self._joblib = joblib
        self._pd = pd
        self._nn = nn
        self._F = F
        self._StoppingCriteria = StoppingCriteria
        self._StoppingCriteriaList = StoppingCriteriaList
        self._SAE = SAE

        # dtype 맞추기
        self.generator.model = self.generator.model.to(torch.bfloat16)

        df = pd.read_csv(self.feature_csv_path)
        if "feature_idx" in df.columns:
            self.target_indices = df["feature_idx"].tolist()
        elif "feature_index" in df.columns:
            self.target_indices = df["feature_index"].tolist()
        else:
            raise ValueError("feature_csv_path must have feature_idx or feature_index column")

        self.sae_projector = self._build_projector(self.sae_layer_index, self.target_indices, self.generator.device)
        self.lr_model = joblib.load(self.lr_model_path)
        self.search_threshold = getattr(self, "search_threshold", 0.5)

        self.handler = self._build_handler(self.sae_projector, self.lr_model, self.search_threshold)

    def _build_projector(self, layer_index: int, feat_idxs: List[int], device):
        nn = self._nn
        F = self._F
        SAE = self._SAE

        release = getattr(self, "sae_release", "llama_scope_lxr_8x")
        sae_id = f"l{layer_index}r_8x"

        logger.info(f"[SAETriggerRAG] Loading SAE: release={release}, sae_id={sae_id}")
        sae = SAE.from_pretrained(release=release, sae_id=sae_id, device=str(device))
        if isinstance(sae, tuple):
            sae = sae[0]

        W = sae.W_enc.data.to(dtype=torch.bfloat16)  # [d_model, d_sae]
        b = sae.b_enc.data.to(dtype=torch.bfloat16)  # [d_sae]

        idx = torch.tensor(feat_idxs, dtype=torch.long, device=W.device)
        W_sel = W[:, idx]        # [d_model, k]
        b_sel = b[idx]           # [k]

        proj = nn.Linear(W_sel.shape[0], W_sel.shape[1], bias=True).to(device=device, dtype=torch.bfloat16)
        with torch.no_grad():
            proj.weight.copy_(W_sel.t())
            proj.bias.copy_(b_sel)
        proj.eval()

        # cleanup
        del sae
        torch.cuda.empty_cache()

        class Projector(nn.Module):
            def __init__(self, p):
                super().__init__()
                self.p = p
            def forward(self, hs):
                return F.relu(self.p(hs.to(self.p.weight.dtype)))

        return Projector(proj)

    def _build_handler(self, projector, lr_model, threshold):
        StoppingCriteria = self._StoppingCriteria

        class Handler(StoppingCriteria):
            def __init__(self):
                self.triggered = False
                self.trigger_prob = 0.0

            def reset(self):
                self.triggered = False
                self.trigger_prob = 0.0

            def hook_fn(self, module, inp, out):
                if self.triggered:
                    return
                hidden = out[0] if isinstance(out, tuple) else out
                last = hidden[:, -1:, :]
                with torch.no_grad():
                    acts = projector(last).squeeze(1).float().detach().cpu().numpy()
                    try:
                        prob = lr_model.predict_proba(acts)[0][1]
                        if prob > threshold:
                            self.triggered = True
                            self.trigger_prob = float(prob)
                    except Exception:
                        pass

            def __call__(self, input_ids, scores, **kwargs):
                return self.triggered

        return Handler()

    def inference(self, question: str, demo: List[Dict[str, str]], case: str) -> str:
        text = ""
        ctx = ""
        tries = 0
        max_tries = getattr(self, "max_retries", 3)

        base_prompt = "".join([d["case"] + "\n" for d in demo]) + case

        # register hook on layer
        layer = self.generator.model.model.layers[self.sae_layer_index]
        hook = layer.register_forward_hook(self.handler.hook_fn)
        try:
            while True:
                prompt = base_prompt + "\n"
                if ctx:
                    prompt += "Context:\n" + ctx + "\n"
                prompt += "Answer: " + text

                self.handler.reset()
                inputs = self.generator.tokenizer(prompt, return_tensors="pt").to(self.generator.device)
                inlen = inputs["input_ids"].shape[1]

                out_ids = self.generator.model.generate(
                    **inputs,
                    max_new_tokens=self.generate_max_length,
                    do_sample=False,
                    pad_token_id=self.generator.tokenizer.pad_token_id,
                    stopping_criteria=self._StoppingCriteriaList([self.handler]),
                )

                gen_ids = out_ids[0][inlen:]
                new_text = self.generator.tokenizer.decode(gen_ids, skip_special_tokens=True)
                if self.use_counter:
                    self.counter.add_generate(new_text, self.generator.tokenizer)

                if self.handler.triggered and tries < max_tries:
                    sent = self.get_top_sentence(new_text) or new_text
                    logger.info(f"[SAETrigger] prob={self.handler.trigger_prob:.4f} | '{sent[:80]}'")

                    docs = self.retrieve(f"{question} {sent}", topk=self.retrieve_topk)
                    for i, d in enumerate(docs):
                        ctx += f"[{tries*self.retrieve_topk + i + 1}] {d}\n"
                    tries += 1
                    continue

                # finalize one sentence
                first = self.get_top_sentence(new_text)
                if not first:
                    break
                text = (text + " " + first).strip()
                if "the answer is" in first.lower():
                    break
                if len(self.generator.tokenizer.encode(text)) > 200:
                    break
        finally:
            hook.remove()

        return text
