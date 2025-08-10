# build_index.py
import pickle
import numpy as np
import torch
import faiss
from pathlib import Path
import fitz
from PIL import Image
import numpy as np
import numpy as np_img
import easyocr
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ==========================
# Ayarlar
# ==========================
PDF_PATH = "i20-Kullanim-Kilavuzu.pdf"
OUTPUT_CHUNKS = "chunks.pkl"
OUTPUT_INDEX = "i20_index.faiss"

CHUNK_SIZE = 240
CHUNK_OVERLAP = 60
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================
# 1. PDF'ten metin çıkarma (OCR fallback)
# ==========================
def extract_text_with_fallback(pdf_path: str, n: int = None):
    doc = fitz.open(pdf_path)
    page_count = doc.page_count if n is None else min(n, doc.page_count)
    texts = []

    reader = easyocr.Reader(['tr'])
    for i in range(page_count):
        page = doc.load_page(i)
        text = page.get_text("text").strip()

        # Çok kısa veya bozuksa OCR'a geç
        if (len(text) < 20) or any(ch in text for ch in ["�", "♥", "Ô", "Ę"]):
            print(f"Sayfa {i+1}: OCR devrede...")
            pix = page.get_pixmap(dpi=200)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            img_np = np_img.array(img)
            result = reader.readtext(img_np, detail=0, paragraph=True)
            text = "\n".join(result)

        texts.append(text)
    doc.close()
    return texts


# ==========================
# 2. RecursiveCharacterTextSplitter ile chunking
# ==========================
def chunk_text_characters(texts, chunk_size=240, overlap=60):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = []
    for page_no, text in enumerate(texts, 1):
        for chunk in splitter.split_text(text):
            if chunk.strip():
                chunks.append({"text": chunk, "page": page_no})
    return chunks


# ==========================
# 3. Embedding + FAISS index oluşturma
# ==========================
def build_index(chunks):
    print(f"Toplam {len(chunks)} chunk üretildi.")
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", device=device)
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings, dtype="float32"))

    with open(OUTPUT_CHUNKS, "wb") as f:
        pickle.dump(chunks, f)
    faiss.write_index(index, OUTPUT_INDEX)
    print("✅ Index ve chunk'lar kaydedildi.")


# ==========================
# Ana akış
# ==========================
if __name__ == "__main__":
    print("1) PDF'ten metin çıkarılıyor...")
    texts = extract_text_with_fallback(PDF_PATH)

    print("2) Chunking başlıyor... (RecursiveCharacterTextSplitter)")
    chunks = chunk_text_characters(texts, CHUNK_SIZE, CHUNK_OVERLAP)

    print("3) Embedding ve index oluşturuluyor...")
    build_index(chunks)
