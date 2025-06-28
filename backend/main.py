from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import faiss, numpy as np, pickle, torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
QA_MODEL = "ucsahin/mT5-base-turkish-qa"

# Semantic embedding + FAISS yükle
embedding_model = SentenceTransformer(EMBED_MODEL, device=device)
index = faiss.read_index("i20_index.faiss")
with open("chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

# BM25 index
bm25 = BM25Okapi([c["text"].split() for c in chunks])

# Generative QA modeli
tokenizer = AutoTokenizer.from_pretrained(QA_MODEL)
t5_model = AutoModelForSeq2SeqLM.from_pretrained(QA_MODEL).to(device)

app = FastAPI(title="Hyundai i20 RAG – Enhanced CoT")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class QuestionRequest(BaseModel):
    question: str
    top_k: int = 4

# FAISS + BM25 hibrit retrieval
def hybrid_retrieve(question: str, top_k: int):
    q_emb = embedding_model.encode([question])
    Dv, Iv = index.search(np.array(q_emb, dtype="float32"), top_k * 2)
    bm25_scores = bm25.get_scores(question.split())
    Ib = np.argsort(bm25_scores)[-top_k * 2:]
    combined = list(dict.fromkeys(list(Iv[0]) + list(Ib)))
    return combined[:top_k]

# Geliştirilmiş CoT prompt
def answer_question(question: str, contexts: list[str]) -> str:
    prompt = f"Soru: {question}\n"
    prompt += "1) Öncelikle soruyu kendi kelimelerinle kısaca özetle.\n"
    prompt += "2) Aşağıdaki metinlerden hangi bilgilerin kullanılabileceğini belirt.\n"
    for i, txt in enumerate(contexts, 1):
        snippet = txt.replace("\n", " ")
        prompt += f"{i}) {snippet}\n"
    prompt += (
        "3) Son olarak, adım adım düşün (Düşünce), ardından net bir şekilde 'Cevap:' kısmında yaz.\n"
    )
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=800).to(device)
    out = t5_model.generate(
        inputs.input_ids,
        max_new_tokens=200,
        num_beams=4,
        early_stopping=True,
        no_repeat_ngram_size=2
    )
    return tokenizer.decode(out[0], skip_special_tokens=True).strip()

@app.post("/ask")
def ask(req: QuestionRequest):
    idxs = hybrid_retrieve(req.question, req.top_k)
    ctxs = [chunks[i]["text"] for i in idxs]
    answer = answer_question(req.question, ctxs)
    sources = [{"text": chunks[i]["text"], "page": chunks[i].get("page")} for i in idxs]
    return {"question": req.question, "answer": answer, "sources": sources}
