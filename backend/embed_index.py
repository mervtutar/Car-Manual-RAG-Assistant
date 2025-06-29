from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import torch
from extract import extract_text_with_fallback
from langchain.text_splitter import RecursiveCharacterTextSplitter

PDF_PATH = "i20-Kullanim-Kilavuzu.pdf"
N_PAGES = None  # Tüm sayfalar için None bırakabilirsin
CHUNK_SIZE = 500  # Karakter cinsinden
CHUNK_OVERLAP = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    print("1. PDF'den metin çıkarılıyor...")
    page_texts = extract_text_with_fallback(PDF_PATH, n=N_PAGES)
    full_text = "\n".join(page_texts)

    print("2. RecursiveCharacterTextSplitter ile chunk'lara bölünüyor...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )
    chunk_texts = splitter.split_text(full_text)
    chunks = [{"text": chunk, "page": None} for chunk in chunk_texts]

    print("3. Embedding hesaplanıyor...")
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", device=device)
    embeddings = model.encode([c["text"] for c in chunks], show_progress_bar=True)

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
