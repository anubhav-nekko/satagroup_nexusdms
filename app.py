# doc_rag_api.py

"""
==============

A minimal RAG back-end for your Streamlit client.

▶︎ Quick-start
pip install fastapi uvicorn boto3 sentence_transformers faiss-cpu PyMuPDF
           python-multipart tavily-python pillow pandas python-docx python-pptx

▶︎ Run
uvicorn doc_rag_api:app --host 0.0.0.0 --port 8000
"""

import os, io, json, uuid, tempfile, pickle, fitz, faiss, boto3
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from docx import Document
from pptx import Presentation
from PIL import Image
from tavily import TavilyClient              # pip install tavily-python

# ── CONFIG ──────────────────────────────────────────────────────────────────
REGION   = os.getenv("AWS_REGION", "us-east-1")
BUCKET   = os.getenv("S3_BUCKET",  "satagroup-test")
TAVILY   = os.getenv("TAVILY_API", "YOUR-TAVILY-KEY")
MODEL_ID = os.getenv("BEDROCK_MODEL_ID",
                     "anthropic.claude-3-7-sonnet-20250219-v1:0")
INDEX_F  = "faiss_index.bin"
META_F   = "metadata_store.pkl"
EMB_DIM  = 768

# ── AWS clients (re-use across calls) ───────────────────────────────────────
s3       = boto3.client("s3",      region_name=REGION)
textract = boto3.client("textract",region_name=REGION)
bedrock  = boto3.client("bedrock-runtime", region_name=REGION)

# ── Globals (in-memory cache) ───────────────────────────────────────────────
app          = FastAPI(title="Document-RAG Gateway", version="1.0.0")
mpnet_model  = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
faiss_index  : faiss.Index  # defined in _load_stores()
metadata     : List[Dict]

# ── Helpers ─────────────────────────────────────────────────────────────────
def _load_stores() -> None:
    """Load FAISS + metadata from S3 (or initialise fresh)."""
    global faiss_index, metadata
    try:
        s3.download_file(BUCKET, INDEX_F, INDEX_F)
        s3.download_file(BUCKET, META_F,  META_F)
        faiss_index = faiss.read_index(INDEX_F)
        metadata    = pickle.load(open(META_F, "rb"))
    except Exception:
        faiss_index = faiss.IndexFlatL2(EMB_DIM)         # empty
        metadata    = []

def _persist_stores() -> None:
    faiss.write_index(faiss_index, INDEX_F)
    pickle.dump(metadata, open(META_F, "wb"))
    s3.upload_file(INDEX_F, BUCKET, INDEX_F)
    s3.upload_file(META_F, BUCKET, META_F)

def _embed(txt: str) -> np.ndarray:
    return mpnet_model.encode(txt, normalize_embeddings=True)

def _ocr_bytes(blob: bytes) -> str:
    resp = textract.detect_document_text(Document={'Bytes': blob})
    return "\n".join(b["Text"] for b in resp.get("Blocks", [])
                     if b["BlockType"] == "LINE")

def _process_file(path: Path, owner: str) -> None:
    """Extract text, embed, push into FAISS & metadata."""
    ext = path.suffix.lower()
    chunks: List[Tuple[str,int]] = []   # [(text,page)]

    if ext == ".pdf":
        doc = fitz.open(path)
        for p in doc:
            text = _ocr_bytes(p.get_pixmap().tobytes("png"))
            chunks.append((text, p.number + 1))
    elif ext in {".jpg", ".jpeg", ".png"}:
        chunks.append((_ocr_bytes(path.read_bytes()), 1))
    elif ext in {".doc", ".docx"}:
        full = "\n".join(p.text for p in Document(path).paragraphs if p.text)
        chunks = [(full[i:i+1000], idx+1)
                  for idx,i in enumerate(range(0, len(full), 1000))]
    elif ext == ".pptx":
        prs = Presentation(path)
        for idx, slide in enumerate(prs.slides, 1):
            txt = "\n".join(s.text for s in slide.shapes if hasattr(s, "text"))
            chunks.append((txt, idx))
    elif ext in {".csv", ".xlsx"}:
        df = pd.read_csv(path) if ext==".csv" else pd.read_excel(path)
        for i in range(0, len(df), 50):
            chunk = df.iloc[i:i+50].to_string(index=False)
            chunks.append((chunk, i//50 + 1))
    else:
        raise ValueError("Unsupported file type")

    for txt, page in chunks:
        vec = _embed(txt)
        faiss_index.add(vec.reshape(1,-1))
        metadata.append({
            "filename": path.name,
            "page": page,
            "text": txt,
            "owner": owner,
            "uploaded": datetime.utcnow().isoformat()
        })

# ── Pydantic DTOs ───────────────────────────────────────────────────────────
class UploadAck(BaseModel):
    status: str
    filename: str
    pages_indexed: int

class QueryBody(BaseModel):
    selected_files: List[str]
    selected_page_ranges: Dict[str, Tuple[int,int]]
    prompt: str
    top_k: int = 20
    last_messages: List[str] = []
    web_search: bool = False

class Answer(BaseModel):
    answer: str
    context: List[Dict]

# ── End-points ───────────────────────────────────────────────────────────────
@app.on_event("startup")
def _warm():
    _load_stores()

@app.post("/upload_document", response_model=UploadAck)
async def upload_document(file: UploadFile = File(...),
                           owner: str = "anonymous",
                           bg: BackgroundTasks = BackgroundTasks()):
    fn = file.filename
    if not fn:
        raise HTTPException(400, "No filename")

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(fn).suffix)
    tmp.write(await file.read()); tmp.close()

    # 1) ship raw to S3 (so users can download)
    with open(tmp.name, "rb") as fh:
        s3.upload_fileobj(fh, BUCKET, fn)

    # 2) heavy processing in background
    def _job():
        try:
            _process_file(Path(tmp.name), owner)
            _persist_stores()
        finally:
            os.remove(tmp.name)
    bg.add_task(_job)

    return UploadAck(status="queued", filename=fn, pages_indexed=0)

@app.post("/query_documents_with_page_range", response_model=Answer)
def query_docs(body: QueryBody):
    if faiss_index.ntotal == 0:
        raise HTTPException(404, "Index empty")

    q_vec = _embed(body.prompt).reshape(1,-1)
    # oversample → later filter
    D,I = faiss_index.search(q_vec, min(body.top_k*5, faiss_index.ntotal))

    hits=[]
    for dist, idx in zip(D[0], I[0]):
        meta = metadata[idx]
        fname = meta["filename"]; pg = meta["page"]
        if body.selected_files and fname not in body.selected_files:
            continue
        lo,hi = body.selected_page_ranges.get(fname, (1,10**6))
        if not (lo <= pg <= hi): continue
        hits.append({"score": float(dist), **meta})
        if len(hits) >= body.top_k: break

    # optional Tavily
    tav_results = {}
    if body.web_search and TAVILY:
        cli = TavilyClient(api_key=TAVILY)
        tav_results = cli.search(body.prompt, search_depth="advanced",
                                 include_raw_content=True)

    # build LLM context
    ctx = {"hits": hits, "last": body.last_messages, "tavily": tav_results}
    prompt = json.dumps(ctx, indent=2)

    payload = {
        "anthropic_version":"bedrock-2023-05-31",
        "max_tokens":4096,
        "messages":[{"role":"user","content":[{"type":"text",
                    "text": body.prompt + "\n\n# Context:\n" + prompt}]}]
    }
    out = bedrock.invoke_model(modelId=MODEL_ID,
                               contentType="application/json",
                               accept="application/json",
                               body=json.dumps(payload))
    answer = json.loads(out["body"].read())["content"][0]["text"]
    return Answer(answer=answer, context=hits)
