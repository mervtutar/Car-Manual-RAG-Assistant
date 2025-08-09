import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import faiss, numpy as np, pickle, torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline, AutoModelForCausalLM
import google.generativeai as genai
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
EXTRACTIVE_MODEL = "dbmdz/bert-base-turkish-cased"
GEN_MODEL = "ytu-ce-cosmos/turkish-gpt2-large"
INDEX_PATH = "i20_index.faiss"
CHUNKS_PATH = "chunks.pkl"

# Gemini API anahtarını ortam değişkeninden al
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_KEY)


# Modelleri yükle
embedding_model = SentenceTransformer(EMBED_MODEL, device=device)
index = faiss.read_index(INDEX_PATH)
with open(CHUNKS_PATH, "rb") as f:
    chunks = pickle.load(f)
bm25 = BM25Okapi([c["text"].split() for c in chunks])

# Extractive QA (BERT)
tokenizer_e = AutoTokenizer.from_pretrained(EXTRACTIVE_MODEL)
model_e = AutoModelForQuestionAnswering.from_pretrained(EXTRACTIVE_MODEL).to(device)
extractive_qa = pipeline("question-answering", model=model_e, tokenizer=tokenizer_e, device=0 if device.type=="cuda" else -1)

# Generative LLM (GPT2)
tokenizer_g = AutoTokenizer.from_pretrained(GEN_MODEL)
model_g = AutoModelForCausalLM.from_pretrained(GEN_MODEL).to(device)
if tokenizer_g.pad_token is None:
    tokenizer_g.pad_token = tokenizer_g.eos_token
    model_g.resize_token_embeddings(len(tokenizer_g))
model_g.config.pad_token_id = tokenizer_g.pad_token_id

app = FastAPI(title="Hyundai i20 RAG – Hybrid QA")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class QuestionRequest(BaseModel):
    question: str
    top_k: int = 5  # Daha fazla chunk ile daha iyi sonuç!

def hybrid_retrieve(question: str, top_k: int):
    q_emb = embedding_model.encode([question])
    Dv, Iv = index.search(np.array(q_emb, dtype="float32"), top_k * 2)
    candidate_idxs = list(dict.fromkeys(list(Iv[0])))
    candidate_chunks = [chunks[i]["text"] for i in candidate_idxs]
    bm25_local = BM25Okapi([c.split() for c in candidate_chunks])
    bm25_scores = bm25_local.get_scores(question.split())
    reranked = np.argsort(bm25_scores)[::-1][:top_k]
    best_idxs = [candidate_idxs[i] for i in reranked]
    return best_idxs

def best_extractive_answer(question: str, contexts: list[str]) -> str:
    optimized_q = (
        f"{question} Cevabı, metindeki teknik talimatı, prosedürü veya uyarı cümlesini aynen yaz."
    )
    best_score = -float('inf')
    best_ans = ""
    for ctx in contexts:
        try:
            res = extractive_qa(question=optimized_q, context=ctx)
            if res and res.get("answer") and (len(res["answer"]) >= 4):
                if res["score"] > best_score:
                    best_score = res["score"]
                    best_ans = res["answer"]
        except Exception:
            continue
    return best_ans if best_ans else ""

def generate_answer(system_hint: str, question: str, context: str, lang: str = "tr") -> str:
    model = genai.GenerativeModel("gemini-2.5-flash")
    prompt = (
        f"{system_hint}\n\n"
        f"Kullanıcı sorusu: {question}\n\n"
        f"Bağlam metni:\n{context}\n\n"
        "Yanıt verirken bağlam dışına çıkma. Teknik ve net ol."
    )
    resp = model.generate_content(prompt)
    return resp.text.strip() if hasattr(resp, 'text') else "Yanıt üretilemedi."

    


@app.post("/ask")
@app.post("/ask")
def ask(req: QuestionRequest):
    idxs = hybrid_retrieve(req.question, req.top_k)
    ctxs = [chunks[i]["text"] for i in idxs]
    sources = [{"text": chunks[i]["text"], "page": chunks[i].get("page")} for i in idxs]

    extractive = best_extractive_answer(req.question, ctxs)  # ← artık kullanılıyor
    generative = ""
    if GEMINI_KEY:  # güvenli çağrı
        generative = generate_answer(
            system_hint="Hyundai i20 kullanım kılavuzuna dayalı teknik asistan.",
            question=req.question,
            context="\n\n".join(ctxs),
            lang="tr",
        )

    return {
        "question": req.question,
        "answer_extractive": extractive,
        "answer_generative": generative or (ctxs[0] if ctxs else ""),
        "sources": sources,
    }