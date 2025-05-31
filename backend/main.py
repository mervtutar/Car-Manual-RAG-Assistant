# backend/main.py

from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import requests

# Dosya yolları
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
INDEX_PATH = "i20_index.faiss"
CHUNKS_PATH = "chunks.pkl"

# Model ve index yükleniyor
model = SentenceTransformer(EMBEDDING_MODEL)
index = faiss.read_index(INDEX_PATH)
with open(CHUNKS_PATH, "rb") as f:
    chunks = pickle.load(f)

app = FastAPI(title="Hyundai i20 RAG API")

class QuestionRequest(BaseModel):
    question: str
    top_k: int = 4

def call_ollama(question, context, model="llama3"):
    prompt = f"Soru: {question}\nBağlam:\n{context}\nCevap:"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    response = requests.post("http://ollama:11434/api/generate", json=payload)
    return response.json().get("response", "Cevap üretilemedi.")

@app.post("/ask")
def ask(payload: QuestionRequest):
    q_emb = model.encode([payload.question])
    D, I = index.search(np.array(q_emb, dtype="float32"), payload.top_k)
    results = [chunks[i] for i in I[0]]
    context = "\n".join([r['text'] for r in results])

    # Ollama ile LLM cevabı al
    answer = call_ollama(payload.question, context)

    return {
        "question": payload.question,
        "answer": answer,
        "sources": results
    }

# Çalıştırmak için:
# uvicorn main:app --reload --host 0.0.0.0 --port 8000
