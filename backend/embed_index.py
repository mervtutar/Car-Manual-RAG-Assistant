from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

from extract import extract_text_with_fallback
from chunk import chunk_text_sentences

PDF_PATH = "i20-Kullanim-Kilavuzu.pdf"
N_PAGES = None    # Tüm sayfalar için None bırakabilirsin
SENTENCES_PER_CHUNK = 3
OVERLAP = 1

def main():
    print("1. PDF'den metin çıkarılıyor...")
    page_texts = extract_text_with_fallback(PDF_PATH, n=N_PAGES)

    print("2. Cümle bazlı chunk'lara bölünüyor...")
    chunks = chunk_text_sentences(page_texts, sentences_per_chunk=SENTENCES_PER_CHUNK, overlap=OVERLAP)

    print("3. Embedding hesaplanıyor...")
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
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
