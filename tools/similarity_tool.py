# tools/similarity_tool.py
# -*- coding: utf-8 -*-
import json
import re
from typing import List, Dict, Any
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def clean_text(text: str) -> str:
    text = text or ""
    text = re.sub(r"[ـًٌٍَُِّْ،؛:«»\"'\[\]\(\){}]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

model_name = "SinaLab/ArabGlossBERT"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.eval()

def similarity_score(text1: str, text2: str) -> float:
    inputs = tokenizer(text1, text2, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        return round(probs[0][1].item(), 3)

def compute_definition_pairs(definitions: List[str], threshold: float) -> List[Dict[str, Any]]:
    defs_clean = [clean_text(d) for d in definitions]
    similar_pairs = []
    for i in range(len(defs_clean)):
        for j in range(i + 1, len(defs_clean)):
            def1 = defs_clean[i]
            def2 = defs_clean[j]
            score = similarity_score(def1, def2)
            if score >= threshold:
                similar_pairs.append(
                    {
                        "def1": def1,
                        "def2": def2,
                        "similarity": score,})
    return similar_pairs
def run_similarity_on_file(filepath: str, threshold: float):
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        return {
            "note": "البيانات ليست List، لم يتم تطبيق similarity.",
            "original_data": data,
        }

    for entry in tqdm(data, desc=" Measuring definition similarities"):
        senses = entry.get("senses", [])

        definitions = []
        for s in senses:
            text_reps = s.get("definition", {}).get("textRepresentations", [])
            for tr in text_reps:
                if tr.get("form"):
                    definitions.append(tr["form"].strip())

        if len(definitions) < 2:
            continue

        similar_pairs = compute_definition_pairs(definitions, threshold=threshold)

        if similar_pairs:
            entry["similar_definitions"] = similar_pairs

    return data
