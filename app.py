# app.py  ── synchronous RAG backend

import os, json, tempfile, pickle, fitz, faiss, boto3
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime
import numpy as np, pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from docx import Document
from pptx import Presentation
from tavily import TavilyClient

# ── CONFIG ──────────────────────────────────────────────────────────────────
REGION   = "us-east-1"
BUCKET   = "satagroup-test"
MODEL_ID = "anthropic.claude-3-7-sonnet-20250219-v1:0"
INDEX_F  = "faiss_index.bin"
META_F   = "metadata_store.pkl"
EMB_DIM  = 768

# ── AWS clients ─────────────────────────────────────────────────────────────
s3       = boto3.client("s3", region_name=REGION)
textract = boto3.client("textract", region_name=REGION)
bedrock  = boto3.client("bedrock-runtime", region_name=REGION)

# ── Globals ─────────────────────────────────────────────────────────────────
app         = FastAPI(title="Document-RAG Gateway", version="1.1.0")
embed_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
faiss_index : faiss.Index
metadata    : List[Dict]

# ── Utility - load / save stores ────────────────────────────────────────────
def _load_stores() -> None:
    global faiss_index, metadata
    try:
        s3.download_file(BUCKET, INDEX_F, INDEX_F)
        s3.download_file(BUCKET, META_F,  META_F)
        faiss_index = faiss.read_index(INDEX_F)
        metadata    = pickle.load(open(META_F, "rb"))
    except Exception:
        faiss_index = faiss.IndexFlatL2(EMB_DIM)
        metadata    = []

def _persist_stores() -> None:
    faiss.write_index(faiss_index, INDEX_F)
    pickle.dump(metadata, open(META_F, "wb"))
    s3.upload_file(INDEX_F, BUCKET, INDEX_F)
    s3.upload_file(META_F, BUCKET, META_F)

def _embed(text: str) -> np.ndarray:
    return embed_model.encode(text, normalize_embeddings=True)

def _ocr_png(png_bytes: bytes) -> str:
    resp = textract.detect_document_text(Document={"Bytes": png_bytes})
    return "\n".join(b["Text"] for b in resp.get("Blocks", [])
                     if b["BlockType"] == "LINE")

# ── Core extractor ─────────────────────────────────────────────────────────
def _process_file(path: Path, owner: str) -> int:
    """Return number of non-empty chunks processed."""
    ext = path.suffix.lower()
    chunks: List[Tuple[str,int]] = []

    if ext == ".pdf":
        doc = fitz.open(path)
        for pg in doc:
            # higher dpi → sharper OCR
            png = pg.get_pixmap(dpi=300).tobytes("png")
            text = _ocr_png(png)
            if text.strip():
                chunks.append((text, pg.number + 1))

    elif ext in {".jpg", ".jpeg", ".png"}:
        text = _ocr_png(path.read_bytes())
        if text.strip():
            chunks.append((text, 1))

    elif ext in {".doc", ".docx"}:
        full = "\n".join(p.text for p in Document(path).paragraphs if p.text)
        for i in range(0, len(full), 1000):
            chunk = full[i:i+1000]
            if chunk.strip():
                chunks.append((chunk, i//1000 + 1))

    elif ext == ".pptx":
        prs = Presentation(path)
        for idx, slide in enumerate(prs.slides, 1):
            slide_txt = "\n".join(
                s.text for s in slide.shapes if hasattr(s, "text")
            )
            if slide_txt.strip():
                chunks.append((slide_txt, idx))

    elif ext in {".csv", ".xlsx"}:
        df = pd.read_csv(path) if ext == ".csv" else pd.read_excel(path)
        for i in range(0, len(df), 50):
            chunk = df.iloc[i:i+50].to_string(index=False)
            if chunk.strip():
                chunks.append((chunk, i//50 + 1))
    else:
        raise HTTPException(400, f"Unsupported file type {ext}")

    # embed & store
    for txt, page in chunks:
        faiss_index.add(_embed(txt).reshape(1, -1))
        metadata.append({
            "filename": path.name,
            "page": page,
            "text": txt,
            "owner": owner,
            "uploaded": datetime.utcnow().isoformat()
        })
    return len(chunks)

# ── DTOs ────────────────────────────────────────────────────────────────────
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

# ── Startup ────────────────────────────────────────────────────────────────
@app.on_event("startup")
def _startup():
    _load_stores()

# ── Endpoints ──────────────────────────────────────────────────────────────
@app.post("/upload_document", response_model=UploadAck)
async def upload_document(file: UploadFile = File(...), owner: str = ""):
    if not owner:
        raise HTTPException(400, "Missing owner field.")

    fn = file.filename or ""
    if not fn:
        raise HTTPException(400, "No filename provided.")

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(fn).suffix)
    tmp.write(await file.read())
    tmp.close()

    # save original to S3
    with open(tmp.name, "rb") as fh:
        s3.upload_fileobj(fh, BUCKET, fn)

    # extract / embed
    try:
        pages = _process_file(Path(tmp.name), owner)
        _persist_stores()
    finally:
        os.remove(tmp.name)

    return UploadAck(status="done", filename=fn, pages_indexed=pages)

@app.post("/query_documents_with_page_range", response_model=Answer)
def query_docs(body: QueryBody):
    if faiss_index.ntotal == 0:
        raise HTTPException(404, "Index empty")

    q_vec = _embed(body.prompt).reshape(1, -1)
    D, I = faiss_index.search(q_vec, min(body.top_k * 5, faiss_index.ntotal))

    hits = []
    for dist, idx in zip(D[0], I[0]):
        meta = metadata[idx]
        f = meta["filename"]; p = meta["page"]
        if body.selected_files and f not in body.selected_files:
            continue
        lo, hi = body.selected_page_ranges.get(f, (1, 10**6))
        if not lo <= p <= hi:
            continue
        hits.append({"score": float(dist), **meta})
        if len(hits) >= body.top_k:
            break

    # build LLM prompt (no change)
    context = json.dumps({"hits": hits, "chat": body.last_messages}, indent=2)
    prompt = body.prompt + "\n\n# Context\n" + context
    payload = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 4096,
        "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
    }
    out = bedrock.invoke_model(modelId=MODEL_ID,
                               contentType="application/json",
                               accept="application/json",
                               body=json.dumps(payload))
    answer = json.loads(out["body"].read())["content"][0]["text"]
    return Answer(answer=answer, context=hits)
