import os
import requests
import faiss
import pickle
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict

from embed_index import get_embedding_model, load_faiss_index, embed_queries

app = FastAPI(title="Car Manual RAG-Based Assistant")

# Ollama server settings (defaults)
LLM_URL   = os.getenv("LLM_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")  # or your Ollama model name

class QueryRequest(BaseModel):
    question: str
    top_k: int = 3

class Source(BaseModel):
    page: int
    text: str
    score: float

class QueryResponse(BaseModel):
    answer: str
    sources: List[Source]

# Load FAISS index + metadata once at startup
index, metadata = load_faiss_index()  # metadata: list of {"page":…, "text":…}

@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    # 1) Embed user question
    q_embs = embed_queries([req.question])
    if not q_embs:
        raise HTTPException(500, "Failed to embed query")
    q_emb = q_embs[0].astype("float32")

    # 2) FAISS search
    D, I = index.search(q_emb.reshape(1, -1), req.top_k)
    D, I = D[0], I[0]

    # 3) Build prompt
    contexts = []
    for idx, score in zip(I, D):
        meta = metadata[idx]
        contexts.append(f"Page {meta['page']}: {meta['text']}")
    prompt = (
        "You are an expert car-manual assistant. Use the following excerpts "
        "from the manual to answer the user’s question.\n\n"
        + "\n".join(contexts)
        + f"\n\nUser: {req.question}\nAssistant:"
    )

    # 4) Call Ollama HTTP API
    try:
        resp = requests.post(
            f"{LLM_URL}/completions?model={LLM_MODEL}",
            json={
                "prompt": prompt,
                "max_tokens": 512,
                "temperature": 0.0
            },
            timeout=30,
        )
        resp.raise_for_status()
        completion = resp.json()["choices"][0]["text"].strip()
    except Exception as e:
        raise HTTPException(502, f"LLM request failed: {e}")

    # 5) Return answer + sources
    sources = [
        Source(page=metadata[i]["page"], text=metadata[i]["text"], score=float(score))
        for i, score in zip(I, D)
    ]
    return QueryResponse(answer=completion, sources=sources)
