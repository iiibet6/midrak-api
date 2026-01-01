# tools/ranking_tool.py
# -*- coding: utf-8 -*-
import json
import time
import re
import requests
import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel

URL = "https://falak.ksaa.gov.sa/api/v1/external/concordancer"
REQUEST_TIMEOUT = 25
MAX_BACKOFF = 15

model_name = "SinaLab/ArabGlossBERT"
tokenizer = AutoTokenizer.from_pretrained(model_name)
clf_model = AutoModelForSequenceClassification.from_pretrained(model_name)
embed_model = AutoModel.from_pretrained(model_name)

device = "cuda" if torch.cuda.is_available() else "cpu"
clf_model.to(device).eval()
embed_model.to(device).eval()


def clean_text(t: Any) -> str:
    if not isinstance(t, str):
        return ""
    t = re.sub(r"<[^>]+>", " ", t)
    t = re.sub(r"[\u0617-\u061A\u064B-\u0652]", "", t)
    t = re.sub(r"[ـ،؛:«»\"'()\[\]{}؟!]", "", t)
    t = re.sub(r"[\n\r\t]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def get_embedding(text: str) -> np.ndarray:
    text = clean_text(text)
    inp = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        out = embed_model(**inp)
    return out.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()


def fetch_contexts(sess: requests.Session, lemma: str, falak_key: str, max_retries: int = 5) -> List[str]:
    payload = {
        "corpusId": "babe4d42-221a-4fc1-8ce9-03bd3fa92dc1",
        "query": clean_text(lemma),
        "prevLen": 15,
        "postLen": 15,
    }
    headers = {
        "apikey": falak_key,
        "Content-Type": "application/json",
        "accept": "application/json",
    }
    backoff = 2

    for _ in range(max_retries):
        try:
            r = sess.post(URL, json=payload, headers=headers, timeout=REQUEST_TIMEOUT)
        except Exception:
            time.sleep(backoff)
            backoff = min(backoff * 2, MAX_BACKOFF)
            continue
        if r.status_code in (401, 403):
            return []

        if r.status_code in (429, 500, 502, 503):
            time.sleep(backoff)
            backoff = min(backoff * 2, MAX_BACKOFF)
            continue

        try:
            df = pd.DataFrame(r.json()).loc["result", "data"]
            df = pd.DataFrame(df)
            sents = (df["prevContext"] + " " + df["query"] + " " + df["postContext"]).astype(str)
            uniq = []
            for s in sents:
                s = s.strip()
                if s and s not in uniq:
                    uniq.append(s)
            return uniq
        except Exception:
            return []
    return []


def classify_context(
    ctx: str,
    lemma: str,
    def_ex_pairs: List[Tuple[str, str]],
    threshold: float,
) -> List[Tuple[str, float]]:
    res = []
    ctx_clean = clean_text(ctx)
    for pair_text, def_text in def_ex_pairs:
        inp = tokenizer(
            ctx_clean,
            clean_text(pair_text),
            return_tensors="pt",
            truncation=True,
            padding=True,
        ).to(device)
        with torch.no_grad():
            logits = clf_model(**inp).logits
        prob = torch.softmax(logits, dim=1)[0][1].item()
        if prob >= threshold:
            res.append((def_text, round(prob, 3)))
    return res


def build_pairs(lemma: str, senses: List[Dict[str, Any]]):
    for s in senses:
        defs = [
            clean_text(t.get("form"))
            for t in s.get("definition", {}).get("textRepresentations", [])
            if t.get("form")
        ]
        exs = [
            clean_text(e.get("form"))
            for e in s.get("examples", [])
            if e.get("form")
        ]
        pairs = []
        if exs:
            for ex in exs:
                for d in defs:
                    pairs.append((f" {lemma} {ex}{d}", d))
        else:
            for d in defs:
                pairs.append((f" {lemma}  {d}", d))
        yield s, pairs


def _run_ranking_on_entries(
    entries: List[Dict[str, Any]],
    falak_key: str,
    threshold: float,
):
    classified: Dict[str, Dict[str, Any]] = {}
    unclassified: Dict[str, List[Dict[str, Any]]] = {}
    no_contexts = set()

    with requests.Session() as sess:
        for entry in tqdm(entries, desc=" Processing lemmas"):
            lemma = entry.get("lemma", "")
            if not lemma:
                continue

            if lemma in classified or lemma in no_contexts:
                continue

            senses = entry.get("senses", [])
            contexts = fetch_contexts(sess, lemma, falak_key)
            if not contexts:
                no_contexts.add(lemma)
                continue

            unclassified_list = []

            for ctx in contexts:
                matched = False

                for sense, pairs in build_pairs(lemma, senses):
                    res = classify_context(ctx, lemma, pairs, threshold)
                    if res:
                        matched = True
                        sense.setdefault("falakContexts", [])
                        sense["falakContexts"].append({"form": ctx, "score": res[0][1]})

                if not matched:
                    best_def = None
                    best_score = 0.0
                    for sense, pairs in build_pairs(lemma, senses):
                        for pair_text, def_text in pairs:
                            inp = tokenizer(
                                clean_text(ctx),
                                clean_text(pair_text),
                                return_tensors="pt",
                                truncation=True,
                                padding=True,
                            ).to(device)
                            with torch.no_grad():
                                logits = clf_model(**inp).logits
                            prob = torch.softmax(logits, dim=1)[0][1].item()
                            if prob < threshold and prob > best_score:
                                best_score = prob
                                best_def = def_text

                    unclassified_list.append(
                        {
                            "context": ctx,
                            "best_score": round(best_score, 3) if best_score > 0 else None,
                            "closest_definition": best_def,
                        }
                    )

            for s in senses:
                if "falakContexts" in s:
                    s["falakContexts"].sort(key=lambda x: x["score"], reverse=True)
                    s["falak_count"] = len(s["falakContexts"])

            entry["senses"] = sorted(
                senses, key=lambda s: s.get("falak_count", 0), reverse=True
            )
            entry["total_falak_contexts"] = sum(
                s.get("falak_count", 0) for s in entry["senses"]
            )
            classified[lemma] = entry
            if unclassified_list:
                unclassified[lemma] = unclassified_list

    sorted_items = sorted(
        classified.items(),
        key=lambda x: x[1].get("total_falak_contexts", 0),
        reverse=True,
    )
    final_data = [{**v, "lemma": k} for k, v in sorted_items]

    context_pos_map = {}
    for entry in classified.values():
        lemma = entry.get("lemma")
        for s in entry.get("senses", []):
            pos = s.get("pos")
            if not pos:
                continue
            for ctx in s.get("falakContexts", []):
                if ctx.get("score", 0) >= threshold:
                    c = clean_text(ctx.get("form", ""))
                    context_pos_map[(lemma, c)] = pos

    results = {}
    for lemma, contexts_data in tqdm(
        unclassified.items(), desc=" Clustering unclassified"
    ):
        contexts = [c["context"].strip() for c in contexts_data if c.get("context")]
        if len(contexts) < 2:
            continue
        emb = np.array([get_embedding(c) for c in contexts])
        emb = normalize(emb)
        k = max(2, min(len(contexts) // 3, 8))
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(emb)
        clusters = {}
        for ctx, lab in zip(contexts, labels):
            lab = str(lab)
            pos = context_pos_map.get((lemma, clean_text(ctx)))
            item = {"context": ctx}
            if pos:
                item["pos"] = pos
            clusters.setdefault(lab, []).append(item)
        results[lemma] = {"num_clusters": k, "clusters": clusters}

    no_ctx_list = sorted(list(no_contexts))

    return final_data, unclassified, results, no_ctx_list


def run_ranking_pipeline(
    input_file: str,
    falak_key: str,
    threshold: float,
    output_ranked: str,
    output_unclassified: str,
    output_clusters: str,
    output_no_contexts: str,
):
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        entries = data
    elif isinstance(data, dict) and "entries" in data:
        entries = data["entries"]
    else:
        raise ValueError("صيغة الملف غير مدعومة لـ Ranking (يجب أن يكون List أو dict فيه 'entries').")

    final_data, unclassified, clusters, no_ctx_list = _run_ranking_on_entries(
        entries, falak_key=falak_key, threshold=threshold
    )

    with open(output_ranked, "w", encoding="utf-8") as f:
        json.dump(final_data, f, ensure_ascii=False, indent=2)

    with open(output_unclassified, "w", encoding="utf-8") as f:
        json.dump(unclassified, f, ensure_ascii=False, indent=2)

    with open(output_clusters, "w", encoding="utf-8") as f:
        json.dump(clusters, f, ensure_ascii=False, indent=2)

    with open(output_no_contexts, "w", encoding="utf-8") as f:
        json.dump(no_ctx_list, f, ensure_ascii=False, indent=2)

    return (
        output_ranked,
        output_unclassified,
        output_clusters,
        output_no_contexts,)

def process_single_entry_ranking(
    entry: Dict[str, Any],
    falak_key: str,
    threshold: float,
):
    entries = [entry]
    final_data, unclassified, clusters, no_ctx_list = _run_ranking_on_entries(
        entries, falak_key=falak_key, threshold=threshold
    )
    result_entry = final_data[0] if final_data else entry
    lemma = result_entry.get("lemma")
    return {
        "entry": result_entry,
        "unclassified": unclassified.get(lemma, []),
        "clusters": clusters.get(lemma),
        "no_contexts": lemma in no_ctx_list,
    }
