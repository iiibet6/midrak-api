# tools/variants_tool.py
# -*- coding: utf-8 -*-
import re
import json
from typing import List, Dict, Any
import requests
from camel_tools.morphology.database import MorphologyDB
from camel_tools.morphology.reinflector import Reinflector
from sentence_transformers import SentenceTransformer, util
db = MorphologyDB.builtin_db(flags="r")
reinflector = Reinflector(db)
sim_model = SentenceTransformer("SinaLab/ArabGlossBERT")
def remove_diacritics(text: str) -> str:
    return re.sub(r'[\u0617-\u061A\u064B-\u0652]', '', text or "")

CORPORA = {
    "babe4d42-221a-4fc1-8ce9-03bd3fa92dc1": "مدونة المجمع",
    "1689cb91-f84b-4686-b550-5f03e657efd1": "مدونة نقش",
    "355943e7-8413-4b88-ba8d-78ef7c873f4": "أدلة الاستخدام",
    "134ae566-063c-4551-9930-dbae7b83533e": "المدونة الثقافية",
    "1959eefb-8bc8-4c68-9695-42812ae2d379": "المدونة الرياضية",
    "4ffb2ae1-0c17-444b-8653-a063fd1f14d5": "الشعر الشعبي",
    "40ece28a-c948-41a9-83b7-05c81705e29d": "الترفيه",
    "arabiccorpus": "المدونة العربية",
    "90eccba1-2056-48f4-ab51-79b9768ee7b1": "كتب العربية",
    "b7265808-6dca-4375-be02-cd72f13c1e8e": "متعلمين",
    "07661dc5-10bd-48a0-8796-757b2436307f": "الروايات",
    "e691881d-b532-40ab-9f47-d1308222990b": "القرآن الكريم",
    "9692af45-5565-4ea2-8d8f-a6c08e42734b": "الإبل",
    "3d837b1e-b0a4-4c76-a53a-6179ffa11c4f": "الأنظمة",
    "96b998af-95c8-4bf6-94a4-ca2ad8bb1581": "المال والأعمال",
    "8b17a05d-ca25-4ff4-9ec9-dd03f3ec31a1": "العمارة",
    "ffd7bf17-d04f-4820-a0ca-ae57637e0abd": "الكتب المتخصصة",
    "8704983f-8b5e-4a7b-91bf-6235c16d59e3": "الأدوية",
    "e4c3be80-c18a-4343-b206-818cb38ef784": "الإعلانات",
    "e4a66160-c5de-4aac-b95c-0ffb70d5671c": "الأصوات",
    "daf9a4f4-8d3e-47d9-965a-cb006f6be431": "الحرف اليدوية",
}
def extract_contexts_from_falak(json_res, corpus_name):
    results = json_res.get("data", {}).get("result", [])
    contexts = []
    for r in results:
        full = f"{r.get('prevContext','')} {r.get('query','')} {r.get('postContext','')}".strip()
        contexts.append({"corpus": corpus_name, "full_context": full})
    return contexts

def get_contexts_from_falak(word: str, corpus_id: str, api_key: str):
    URL = "https://falak.ksaa.gov.sa/api/v1/external/concordancer"
    headers = {"apikey": api_key, "Content-Type": "application/json", "accept": "application/json"}
    payload = {"corpusId": corpus_id, "query": word, "prevLen": 15, "postLen": 15}

    try:
        resp = requests.post(URL, json=payload, headers=headers, timeout=20)
        return extract_contexts_from_falak(resp.json(), CORPORA[corpus_id])
    except Exception:
        return []
def generate_variants(word: str) -> List[str]:
    base = remove_diacritics(word)
    variants = set()
    try:
        analyses = reinflector.reinflect(base, {})    
    except Exception:
        return []

    for a in analyses:
        undiac = remove_diacritics(a.get("diac", ""))
        if undiac and undiac != base:
            variants.add(undiac)
    return list(variants)

def filter_variants_by_similarity(lemma: str, variants: List[str], threshold: float):
    filtered = []

    orig_clean = remove_diacritics(lemma)
    orig_emb = sim_model.encode(orig_clean, convert_to_tensor=True)

    for v in variants:
        v_emb = sim_model.encode(v, convert_to_tensor=True)
        sim = util.cos_sim(orig_emb, v_emb).item()

        if sim >= threshold:
            filtered.append(v)

    return filtered

def process_single_lemma_with_variants(lemma: str, api_key: str, threshold: float):

    lemma_clean = remove_diacritics(lemma)

    raw_variants = generate_variants(lemma_clean)
    good_variants = filter_variants_by_similarity(lemma_clean, raw_variants, threshold)

    result_variants = []

    for v in good_variants:
        ctxs = []
        for corpus_id in CORPORA:
            ctxs.extend(get_contexts_from_falak(v, corpus_id, api_key))

        # unique
        unique, seen = [], set()
        for c in ctxs:
            t = c["full_context"]
            if t not in seen:
                seen.add(t)
                unique.append(c)

        result_variants.append({"variant": v, "contexts": unique})

    return {"lemma": lemma, "threshold": threshold, "variants": result_variants}

def run_variants_on_json_with_similarity(filepath: str, api_key: str, threshold: float):
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        for entry in data:
            lemma = entry.get("lemma")
            if not lemma:
                continue
            processed = process_single_lemma_with_variants(lemma, api_key, threshold)
            entry["variants"] = processed["variants"]

    elif isinstance(data, dict) and "entries" in data:
        for entry in data["entries"]:
            lemma = entry.get("lemma")
            if not lemma:
                continue
            processed = process_single_lemma_with_variants(lemma, api_key, threshold)
            entry["variants"] = processed["variants"]

    return data
