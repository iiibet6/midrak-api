# tools/synsets_tool.py
# -*- coding: utf-8 -*-
import json
import re
from typing import List, Dict, Any, Tuple
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "SinaLab/ArabGlossBERT"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.eval()


def clean_text(text: Any) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"<[^>]+>", " ", text)  
    text = re.sub(r"[\u0617-\u061A\u064B-\u0652-\u200F]", "", text)  
    text = re.sub(r"[^ء-يA-Za-z0-9\s،؛؟!.]", " ", text) 
    text = re.sub(r"\s+", " ", text).strip()
    return text


def batched_similarity(pairs: List[Tuple[str, str]]):
    texts1 = [clean_text(t1) for t1, _ in pairs]
    texts2 = [clean_text(t2) for _, t2 in pairs]
    inputs = tokenizer(texts1, texts2, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1)[:, 1].tolist()
    return [round(p, 3) for p in probs]


def _build_definitions_list_from_lexicon(data: List[Dict[str, Any]]):
    definitions_list = []
    for entry in data:
        lemma = entry.get("lemma")
        for sense in entry.get("senses", []):
            defs = sense.get("definition", {}).get("textRepresentations", [])
            for d in defs:
                if not d or not isinstance(d, dict):
                    continue
                form = d.get("form")
                if not form or not isinstance(form, str):
                    continue
                form = form.strip()
                if not form:
                    continue
                definitions_list.append(
                    {
                        "lemma": lemma,
                        "definition": form,
                        "assigned": False,
                        "synsetId": None,
                    }
                )
    return definitions_list


def _cluster_definitions(definitions_list, threshold: float, batch_size: int):
    synsets = {}
    synset_counter = 1

    for i in tqdm(range(len(definitions_list)), desc="Clustering similar definitions"):
        if definitions_list[i]["assigned"]:
            continue

        current_group = [definitions_list[i]]
        definitions_list[i]["assigned"] = True
        synset_id = f"synset_{synset_counter:04d}"

        pairs, indices = [], []
        for j in range(i + 1, len(definitions_list)):
            if not definitions_list[j]["assigned"]:
                pairs.append(
                    (definitions_list[i]["definition"], definitions_list[j]["definition"])
                )
                indices.append(j)

            if len(pairs) >= batch_size:
                scores = batched_similarity(pairs)
                for idx, score in zip(indices, scores):
                    if score >= threshold:
                        definitions_list[idx]["assigned"] = True
                        current_group.append(definitions_list[idx])
                pairs, indices = [], []

        if pairs:
            scores = batched_similarity(pairs)
            for idx, score in zip(indices, scores):
                if score >= threshold:
                    definitions_list[idx]["assigned"] = True
                    current_group.append(definitions_list[idx])

        lemmas = sorted(list(set([d["lemma"] for d in current_group])))
        synsets[synset_id] = {
            "lemmas": lemmas,
            "definitions": [d["definition"] for d in current_group],
        }

        for d in current_group:
            d["synsetId"] = synset_id

        synset_counter += 1

    return synsets, definitions_list, synset_counter - 1


def _propagate_synset_ids_to_lexicon(data, definitions_list):
    def2syn = {}
    for d in definitions_list:
        form = d["definition"]
        sid = d["synsetId"]
        if form not in def2syn:
            def2syn[form] = sid

    for entry in data:
        for sense in entry.get("senses", []):
            defs = sense.get("definition", {}).get("textRepresentations", [])
            for d in defs:
                if not d or not isinstance(d, dict):
                    continue
                form = d.get("form")
                if not form or not isinstance(form, str):
                    continue
                form = form.strip()
                if not form:
                    continue
                sid = def2syn.get(form)
                if sid:
                    sense["synsetId"] = sid

    return data


def run_synsets_pipeline(
    input_file: str,
    threshold: float,
    batch_size: int,
    output_lexicon: str,
    output_synsets: str,
):
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Synsets تتوقع قائمة entries (List) كمدخل.")

    definitions_list = _build_definitions_list_from_lexicon(data)
    synsets, definitions_list, num_synsets = _cluster_definitions(
        definitions_list, threshold=threshold, batch_size=batch_size
    )
    data_with_synsets = _propagate_synset_ids_to_lexicon(data, definitions_list)

    with open(output_lexicon, "w", encoding="utf-8") as f:
        json.dump(data_with_synsets, f, ensure_ascii=False, indent=2)

    with open(output_synsets, "w", encoding="utf-8") as f:
        json.dump(synsets, f, ensure_ascii=False, indent=2)

    return output_lexicon, output_synsets

def build_synsets_for_entries(entries: List[Dict[str, Any]], threshold: float, batch_size: int):
    definitions_list = []
    for e_idx, entry in enumerate(entries):
        lemma = entry.get("lemma")
        for def_str in entry.get("definitions", []):
            def_str = (def_str or "").strip()
            if not def_str:
                continue
            definitions_list.append(
                {
                    "lemma": lemma,
                    "definition": def_str,
                    "assigned": False,
                    "synsetId": None,
                    "entry_index": e_idx,
                }
            )

    synsets, definitions_list, num_synsets = _cluster_definitions(
        definitions_list, threshold=threshold, batch_size=batch_size
    )

    for d in definitions_list:
        sid = d["synsetId"]
        e_idx = d["entry_index"]
        def_str = d["definition"]
        if "definitions_with_synsets" not in entries[e_idx]:
            entries[e_idx]["definitions_with_synsets"] = []
        entries[e_idx]["definitions_with_synsets"].append(
            {"definition": def_str, "synsetId": sid}
        )

    return synsets, entries
