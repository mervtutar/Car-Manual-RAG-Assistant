from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dosya yolları
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
INDEX_PATH = "i20_index.faiss"
CHUNKS_PATH = "chunks.pkl"

# Embedding modeli ve index yükleniyor
embedding_model = SentenceTransformer(EMBEDDING_MODEL, device=device)
index = faiss.read_index(INDEX_PATH)
with open(CHUNKS_PATH, "rb") as f:
    chunks = pickle.load(f)

# BERT extractive QA modeli yükleniyor
QA_MODEL_NAME = "dbmdz/bert-base-turkish-cased"
tokenizer = AutoTokenizer.from_pretrained(QA_MODEL_NAME)
model = AutoModelForQuestionAnswering.from_pretrained(QA_MODEL_NAME).to(device)
qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)

app = FastAPI(title="Hyundai i20 RAG API")

class QuestionRequest(BaseModel):
    question: str
    top_k: int = 4

def answer_question(question: str, context: str) -> str:
    inputs = tokenizer(question, context, return_tensors="pt", truncation=True, max_length=512).to(device)
    outputs = model(**inputs)
    
    # Pipeline kullanmıyorsan output'tan answer span çıkarma işlemi yapmalısın
    # Ama pipeline kullanıyorsan aşağıdaki gibi devam edebilirsin:
    
    from transformers import pipeline
    qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer, device=0 if device.type == "cuda" else -1)
    result = qa_pipeline(question=question, context=context)
    return result['answer']


@app.post("/ask")
def ask(payload: QuestionRequest):
    # Soru için embedding hesapla
    q_emb = embedding_model.encode([payload.question])
    # Benzer chunkları bul
    D, I = index.search(np.array(q_emb, dtype="float32"), payload.top_k)
    results = [chunks[i] for i in I[0]]
    # Bağlamı birleştir
    context = "\n".join([r['text'] for r in results])

    # Extractive QA modeli ile cevap al
    answer = answer_question(payload.question, context)

    return {
        "question": payload.question,
        "answer": answer,
        "sources": results
    }

# Çalıştırmak için:
# uvicorn main:app --reload --host 0.0.0.0 --port 8000
