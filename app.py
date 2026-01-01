# app.py
# -*- coding: utf-8 -*-
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import os
import shutil
import uuid
import zipfile
import json
from fastapi import Body

from tools.ranking_tool import run_ranking_pipeline, process_single_entry_ranking
from tools.similarity_tool import run_similarity_on_file, compute_definition_pairs
from tools.synsets_tool import run_synsets_pipeline, build_synsets_for_entries
from tools.variants_tool import process_single_lemma_with_variants, run_variants_on_json_with_similarity
class FalakSense(BaseModel):
    definition: Dict[str, Any] = {}
    translations: list = []
    contexts: list = []
    domainIds: list = []
    examples: list = []
    subSenses: list = []
    synsetId: Any = None
    level: float | None = None
    relations: list = []
    pos: str | None = None

class FalakEntry(BaseModel):
    lemma: str
    pos: str | None = None
    senses: list[FalakSense]
    falak_key: str
    threshold: float = 0.5

app = FastAPI(
    title="Midrak API",
    description="Arabic Lexical Processing Tools (Ranking, Similarity, Synsets, Variants)",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# Models

class DefinitionText(BaseModel):
    form: str


class SenseForText(BaseModel):
    definition: Dict[str, Any]
    examples: Optional[List[Dict[str, Any]]] = []
    contexts: Optional[List[Dict[str, Any]]] = []


class RankingTextRequest(BaseModel):
    lemma: str = Field(..., description="الكلمة المدروسة")
    senses: List[SenseForText] = Field(..., description="قائمة المعاني بصيغة مبسطة")
    falak_key: str = Field(..., description="مفتاح واجهة فلك")
    threshold: float = Field(0.5, description="عتبة التصنيف (افتراضي 0.5)")


class SimilarityTextRequest(BaseModel):
    lemma: Optional[str] = Field(None, description="الكلمة (اختياري)")
    definitions: List[str] = Field(..., description="التعاريف المراد قياس التشابه بينها")
    threshold: float = Field(0.95, description="العتبة (افتراضي 0.95)")


class SynsetEntryText(BaseModel):
    lemma: str
    definitions: List[str]


class SynsetsTextRequest(BaseModel):
    entries: List[SynsetEntryText]
    threshold: float = Field(0.90)
    batch_size: int = Field(8)


class VariantsTextRequest(BaseModel):
    lemma: str
    falak_key: str
    threshold: float = Field(0.90, description="عتبة تشابه الكلمة مع المتغيرات (BERT)")


# Helper functions

def save_uploaded_file(file: UploadFile, subdir: str = "") -> str:
    ext = os.path.splitext(file.filename)[1]
    uid = uuid.uuid4().hex
    dir_path = os.path.join(UPLOAD_DIR, subdir) if subdir else UPLOAD_DIR
    os.makedirs(dir_path, exist_ok=True)
    dest_path = os.path.join(dir_path, f"{uid}{ext}")
    with open(dest_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return dest_path


def make_zip(files: Dict[str, str], zip_name: str) -> str:
    zip_path = os.path.join(OUTPUT_DIR, zip_name)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for internal_name, real_path in files.items():
            if os.path.exists(real_path):
                zf.write(real_path, arcname=internal_name)
    return zip_path


def save_json_output(data: Any, filename: str) -> str:
    if not filename.lower().endswith(".json"):
        filename += ".json"
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return path


# RANKING 
from copy import deepcopy
@app.post("/midrak/ranking/from-text")
async def ranking_from_text(entry: FalakEntry):
    
    falak_key = entry.falak_key
    threshold = entry.threshold

    entry_dict = entry.dict()
    entry_dict.pop("falak_key", None)
    entry_dict.pop("threshold", None)

    result = process_single_entry_ranking(
        entry=entry_dict,
        falak_key=falak_key,
        threshold=threshold,
    )
    return JSONResponse(result)


@app.post("/midrak/ranking/from-file")
async def ranking_from_file(
    file: UploadFile = File(...),
    falak_key: str = Form(...),
    threshold: float = Form(0.5),
    zip_name: str = Form("ranking_results.zip"),
    ranked_name: str = Form("ranked_lexicon.json"),
    unclassified_name: str = Form("unclassified_contexts.json"),
    clusters_name: str = Form("clusters.json"),
    no_contexts_name: str = Form("no_contexts.json"),
):
    input_path = save_uploaded_file(file, subdir="ranking")

    ranked_path, unclassified_path, clusters_path, no_contexts_path = run_ranking_pipeline(
        input_file=input_path,
        falak_key=falak_key,
        threshold=threshold,
        output_ranked=os.path.join(OUTPUT_DIR, ranked_name),
        output_unclassified=os.path.join(OUTPUT_DIR, unclassified_name),
        output_clusters=os.path.join(OUTPUT_DIR, clusters_name),
        output_no_contexts=os.path.join(OUTPUT_DIR, no_contexts_name),
    )

    zip_path = make_zip(
        {
            ranked_name: ranked_path,
            unclassified_name: unclassified_path,
            clusters_name: clusters_path,
            no_contexts_name: no_contexts_path,
        },
        zip_name=zip_name,
    )

    return FileResponse(
        zip_path,
        media_type="application/zip",
        filename=os.path.basename(zip_path),
    )


# SIMILARITY

@app.post("/midrak/similarity/from-text")
async def similarity_from_text(payload: SimilarityTextRequest):
    if len(payload.definitions) < 2:
        raise HTTPException(status_code=400, detail="يجب إدخال تعريفين فأكثر.")

    similar_pairs = compute_definition_pairs(
        payload.definitions,
        threshold=payload.threshold,
    )

    result = {
        "lemma": payload.lemma,
        "threshold": payload.threshold,
        "similar_definitions": similar_pairs,
    }
    return JSONResponse(result)


@app.post("/midrak/similarity/from-file")
async def similarity_from_file(
    file: UploadFile = File(...),
    threshold: float = Form(0.95),
    output_name: str = Form("similarity_results.json"),
):
    input_path = save_uploaded_file(file, subdir="similarity")
    output_data = run_similarity_on_file(filepath=input_path, threshold=threshold)
    out_path = save_json_output(output_data, output_name)

    return FileResponse(
        out_path,
        media_type="application/json",
        filename=os.path.basename(out_path),
    )


# SYNSETS 

@app.post("/midrak/synsets/from-text")
async def synsets_from_text(payload: SynsetsTextRequest):
    entries = [e.dict() for e in payload.entries]
    synsets, updated_entries = build_synsets_for_entries(
        entries=entries,
        threshold=payload.threshold,
        batch_size=payload.batch_size,
    )
    return JSONResponse(
        {
            "threshold": payload.threshold,
            "batch_size": payload.batch_size,
            "synsets": synsets,
            "entries": updated_entries,
        }
    )


@app.post("/midrak/synsets/from-file")
async def synsets_from_file(
    file: UploadFile = File(...),
    threshold: float = Form(0.90),
    batch_size: int = Form(8),
    zip_name: str = Form("synsets_results.zip"),
    output_lexicon_name: str = Form("lexicon_with_synsets.json"),
    output_synsets_name: str = Form("synsets_index.json"),
):
    input_path = save_uploaded_file(file, subdir="synsets")

    lexicon_path, syn_index_path = run_synsets_pipeline(
        input_file=input_path,
        threshold=threshold,
        batch_size=batch_size,
        output_lexicon=os.path.join(OUTPUT_DIR, output_lexicon_name),
        output_synsets=os.path.join(OUTPUT_DIR, output_synsets_name),
    )

    zip_path = make_zip(
        {
            output_lexicon_name: lexicon_path,
            output_synsets_name: syn_index_path,
        },
        zip_name=zip_name,
    )

    return FileResponse(
        zip_path,
        media_type="application/zip",
        filename=os.path.basename(zip_path),
    )


# VARIANTS 

@app.post("/midrak/variants/from-text")
async def variants_from_text(payload: VariantsTextRequest):
    result = process_single_lemma_with_variants(
        lemma=payload.lemma,
        api_key=payload.falak_key,
        threshold=payload.threshold,
    )
    return JSONResponse(result)


@app.post("/midrak/variants/from-file")
async def variants_from_file(
    file: UploadFile = File(...),
    falak_key: str = Form(...),
    threshold: float = Form(0.90),
    output_name: str = Form("variants_output.json"),
):
    input_path = save_uploaded_file(file, subdir="variants")

    output_data = run_variants_on_json_with_similarity(
        filepath=input_path,
        api_key=falak_key,
        threshold=threshold,
    )

    out_path = save_json_output(output_data, output_name)

    return FileResponse(
        out_path,
        media_type="application/json",
        filename=os.path.basename(out_path),
    )


# Simple UI

@app.get("/midrak/ui")
def midrak_ui():
    html = """
    <html>
    <head>
      <meta charset="utf-8" />
      <title>Midrak Tools</title>
    </head>
    <body>
      <h1>Midrak Tools - Simple UI</h1>

      <h2>Ranking (from file)</h2>
      <form action="/midrak/ranking/from-file" method="post" enctype="multipart/form-data">
        Falak Key: <input type="text" name="falak_key" required /><br/>
        Threshold: <input type="number" step="0.01" name="threshold" value="0.5" /><br/>
        ZIP Name: <input type="text" name="zip_name" value="ranking_results.zip" /><br/>
        File: <input type="file" name="file" required /><br/>
        <button type="submit">Run Ranking</button>
      </form>

      <h2>Similarity (from file)</h2>
      <form action="/midrak/similarity/from-file" method="post" enctype="multipart/form-data">
        Threshold: <input type="number" step="0.01" name="threshold" value="0.95" /><br/>
        Output Name: <input type="text" name="output_name" value="similarity_results.json" /><br/>
        File: <input type="file" name="file" required /><br/>
        <button type="submit">Run Similarity</button>
      </form>

      <h2>Synsets (from file)</h2>
      <form action="/midrak/synsets/from-file" method="post" enctype="multipart/form-data">
        Threshold: <input type="number" step="0.01" name="threshold" value="0.90" /><br/>
        Batch Size: <input type="number" name="batch_size" value="8" /><br/>
        ZIP Name: <input type="text" name="zip_name" value="synsets_results.zip" /><br/>
        File: <input type="file" name="file" required /><br/>
        <button type="submit">Run Synsets</button>
      </form>

      <h2>Variants (from file)</h2>
      <form action="/midrak/variants/from-file" method="post" enctype="multipart/form-data">
        Falak Key: <input type="text" name="falak_key" required /><br/>
        Threshold: <input type="number" step="0.01" name="threshold" value="0.90" /><br/>
        Output Name: <input type="text" name="output_name" value="variants_output.json" /><br/>
        File: <input type="file" name="file" required /><br/>
        <button type="submit">Run Variants</button>
      </form>
    </body>
    </html>
    """
    return HTMLResponse(content=html)
