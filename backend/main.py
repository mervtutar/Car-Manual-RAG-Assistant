from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
import torch

EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
INDEX_PATH = "i20_index.faiss"
CHUNKS_PATH = "chunks.pkl"

embedding_model = SentenceTransformer(EMBEDDING_MODEL)
index = faiss.read_index(INDEX_PATH)
with open(CHUNKS_PATH, "rb") as f:
    chunks = pickle.load(f)

QA_MODEL_NAME = "dbmdz/bert-base-turkish-uncased"
tokenizer = AutoTokenizer.from_pretrained(QA_MODEL_NAME)
qa_model = AutoModelForQuestionAnswering.from_pretrained(QA_MODEL_NAME)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
qa_model.to(device)

# CORS middleware
app = FastAPI(title="Hyundai i20 RAG API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QuestionRequest(BaseModel):
    question: str
    top_k: int = 4

# Extractive QA fonksiyonu
def answer_question(question: str, context: str) -> str:
    qa_pipeline = pipeline(
        "question-answering",
        model=qa_model,
        tokenizer=tokenizer,
        device=0 if device.type == "cuda" else -1
    )
    result = qa_pipeline(question=question, context=context)
    return result['answer']

@app.post("/ask")
def ask(payload: QuestionRequest):
    q_emb = embedding_model.encode([payload.question])
    D, I = index.search(np.array(q_emb, dtype="float32"), payload.top_k)
    results = [chunks[i] for i in I[0]]
    context = "\n".join([r['text'] for r in results])

    answer = answer_question(payload.question, context)

    return {
        "question": payload.question,
        "answer": answer,
        "sources": results
    }

# Çalıştırmak için:
# uvicorn main:app --reload --host 0.0.0.0 --port 8000
