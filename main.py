import os
import json
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from sklearn.neighbors import NearestNeighbors

# --------- Config ---------
DATA_DIR = "data"
EMB_PATH = os.path.join(DATA_DIR, "embeddings.npy")
META_PATH = os.path.join(DATA_DIR, "metadata.json")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")

os.makedirs(DATA_DIR, exist_ok=True)

app = FastAPI(title="English-Arabic RAG Translator")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------- In-memory state ---------
embeddings: Optional[np.ndarray] = None  # shape (N, D)
metadata: List[dict] = []
vector_dim: Optional[int] = None
nn_index: Optional[NearestNeighbors] = None


def get_openai_client(api_key: Optional[str] = None) -> OpenAI:
    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise HTTPException(status_code=400, detail="OpenAI API key not provided. Set OPENAI_API_KEY env or pass api_key in request.")
    return OpenAI(api_key=key)


def save_state():
    global embeddings, metadata
    if embeddings is None or embeddings.size == 0 or not metadata:
        return
    np.save(EMB_PATH, embeddings)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False)


def load_state():
    global embeddings, metadata, vector_dim, nn_index
    if os.path.exists(EMB_PATH) and os.path.exists(META_PATH):
        try:
            embeddings = np.load(EMB_PATH)
            with open(META_PATH, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            vector_dim = embeddings.shape[1] if embeddings.ndim == 2 else None
            if embeddings is not None and embeddings.size > 0:
                build_nn_index()
        except Exception:
            embeddings = None
            metadata = []
            vector_dim = None
            nn_index = None


def build_nn_index():
    global nn_index, embeddings
    if embeddings is None or embeddings.size == 0:
        nn_index = None
        return
    # Use cosine distance for semantic similarity
    nn = NearestNeighbors(metric="cosine")
    nn.fit(embeddings)
    nn_index = nn


@app.on_event("startup")
async def on_startup():
    load_state()


@app.get("/")
def read_root():
    return {"message": "RAG Translator API running"}


@app.get("/rag/status")
def rag_status():
    return {
        "indexed": len(metadata) if metadata else 0,
        "has_index": nn_index is not None,
        "vector_dim": vector_dim,
        "embedding_model": EMBEDDING_MODEL,
        "chat_model": CHAT_MODEL,
    }


class QueryBody(BaseModel):
    query: str
    direction: str = "en2ar"  # or "ar2en"
    top_k: int = 5
    api_key: Optional[str] = None


def embed_texts(client: OpenAI, texts: List[str]) -> np.ndarray:
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    vectors = [item.embedding for item in resp.data]
    return np.array(vectors, dtype=np.float32)


def detect_columns(df: pd.DataFrame) -> Tuple[str, str]:
    cols = [c.lower() for c in df.columns]
    eng_idx = None
    ar_idx = None
    for i, c in enumerate(cols):
        if eng_idx is None and ("english" in c or c.startswith("en") or "source" in c):
            eng_idx = i
        if ar_idx is None and ("arab" in c or c.startswith("ar") or "target" in c):
            ar_idx = i
    if eng_idx is None or ar_idx is None:
        if len(df.columns) >= 2:
            eng_idx, ar_idx = 0, 1
        else:
            raise HTTPException(status_code=400, detail="Could not auto-detect English/Arabic columns. Please ensure at least two columns exist or name them clearly.")
    return df.columns[eng_idx], df.columns[ar_idx]


@app.post("/rag/upload")
async def rag_upload(
    file: UploadFile = File(...),
    sheet_name: Optional[str] = Form(None),
    api_key: Optional[str] = Form(None),
):
    global embeddings, metadata, vector_dim
    try:
        if not file.filename.lower().endswith((".xlsx", ".xls")):
            raise HTTPException(status_code=400, detail="Please upload an Excel file (.xlsx or .xls)")
        content = await file.read()
        xlsx_path = os.path.join(DATA_DIR, "uploaded.xlsx")
        with open(xlsx_path, "wb") as f:
            f.write(content)
        if sheet_name:
            df = pd.read_excel(xlsx_path, sheet_name=sheet_name)
        else:
            df = pd.read_excel(xlsx_path)
        if df.empty:
            raise HTTPException(status_code=400, detail="The uploaded Excel contains no rows")
        en_col, ar_col = detect_columns(df)

        records = []
        for idx, row in df.iterrows():
            en = str(row.get(en_col, "")).strip()
            ar = str(row.get(ar_col, "")).strip()
            if not en and not ar:
                continue
            combined = f"EN: {en}\nAR: {ar}"
            records.append({
                "english": en,
                "arabic": ar,
                "combined": combined,
                "row": int(idx),
            })
        if not records:
            raise HTTPException(status_code=400, detail="No valid rows found in the Excel file")

        client = get_openai_client(api_key)
        texts = [r["combined"] for r in records]
        vecs = embed_texts(client, texts)

        # Initialize or extend embeddings
        if embeddings is None or embeddings.size == 0:
            embeddings = vecs
            metadata = records
        else:
            # If dimensions mismatch, reset index
            if embeddings.shape[1] != vecs.shape[1]:
                embeddings = vecs
                metadata = records
            else:
                embeddings = np.vstack([embeddings, vecs])
                metadata.extend(records)

        vector_dim = embeddings.shape[1]
        build_nn_index()
        save_state()

        return {
            "message": "Index built successfully",
            "rows_indexed": len(records),
            "columns": {"english": en_col, "arabic": ar_col},
            "vector_dim": vector_dim,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload/indexing error: {str(e)}")


@app.post("/rag/query")
async def rag_query(body: QueryBody):
    if nn_index is None or embeddings is None or embeddings.size == 0 or not metadata:
        raise HTTPException(status_code=400, detail="No index loaded. Please upload and index your Excel first.")
    if body.direction not in ("en2ar", "ar2en"):
        raise HTTPException(status_code=400, detail="direction must be 'en2ar' or 'ar2en'")

    client = get_openai_client(body.api_key)
    q_vec = embed_texts(client, [body.query])

    k = min(body.top_k, len(metadata))
    distances, indices = nn_index.kneighbors(q_vec, n_neighbors=k)
    I = indices[0].tolist()
    D = distances[0].tolist()
    retrieved = [metadata[i] for i in I if i < len(metadata)]

    direction_text = "English to Arabic" if body.direction == "en2ar" else "Arabic to English"
    examples = "\n".join([f"EN: {r['english']}\nAR: {r['arabic']}" for r in retrieved if r.get('english') or r.get('arabic')])
    user_text = body.query
    if body.direction == "ar2en":
        examples = "\n".join([f"AR: {r['arabic']}\nEN: {r['english']}" for r in retrieved if r.get('english') or r.get('arabic')])

    system_prompt = (
        "You are a professional bilingual translator. Translate between English and Arabic using the provided glossary examples as guidance. "
        "Maintain meaning, tone, and correctness. If specialized terms appear in the examples, use them consistently. "
        "Respond with only the translated text, no explanations."
    )

    context_block = f"GUIDANCE EXAMPLES (most similar):\n{examples}\n---\nTranslate ({direction_text}) the following text:"

    completion = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{context_block}\n{user_text}"},
        ],
        temperature=0.2,
    )
    translated = completion.choices[0].message.content.strip()

    return {
        "translation": translated,
        "retrieved": retrieved,
        "distances": D,
    }


@app.get("/test")
def test_endpoint():
    return {"backend": "Running", "index_loaded": nn_index is not None, "items": len(metadata) if metadata else 0}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
