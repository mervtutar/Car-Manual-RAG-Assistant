from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import torch
from extract import extract_text_with_fallback

# Chunklama için langchain splitter (chunk başına 200-300 karakter önerilir)
from langchain.text_splitter import RecursiveCharacterTextSplitter

PDF_PATH = "i20-Kullanim-Kilavuzu.pdf"
N_PAGES = None  # None ile tüm sayfalar
CHUNK_SIZE = 240   # Karakter (idealde 200–300)
CHUNK_OVERLAP = 60 # Karakter
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    print("1. PDF'den metin çıkarılıyor...")
    page_texts = extract_text_with_fallback(PDF_PATH, n=N_PAGES)

    # Her sayfa ayrı ayrı chunklanır, böylece sayfa bilgisini koruyabilirsin!
    print("2. RecursiveCharacterTextSplitter ile chunk'lara bölünüyor...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = []
    for page_no, text in enumerate(page_texts, 1):
        for chunk in splitter.split_text(text):
            if chunk.strip():
                chunks.append({
                    "text": chunk,
                    "page": page_no  # Doğru sayfa numarası!
                })

    print(f"Toplam {len(chunks)} chunk üretildi.")
    print("3. Embedding hesaplanıyor...")
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", device=device)
    chunk_texts = [c["text"] for c in chunks]
    embeddings = model.encode(chunk_texts, show_progress_bar=True)

    print("4. FAISS index oluşturuluyor...")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings, dtype="float32"))

    # Sonuçları kaydet
    with open("chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)
    faiss.write_index(index, "i20_index.faiss")
    print("Index ve chunk'lar kaydedildi!")

if __name__ == "__main__":
    main()
