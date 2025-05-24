# hyundai-i20-rag

```plaintext
hyundai-i20-rag/
│
├── backend/
│   ├── main.py                   # FastAPI app, RAG pipeline kodu
│   ├── extract.py                # PyMuPDF + OCR fallback metin çıkarım fonksiyonu
│   ├── chunk.py                  # chunk_text fonksiyonu
│   ├── embed_index.py            # embedding ve FAISS indeksleme
│   ├── requirements.txt          # Python bağımlılıkları
│   ├── i20-Kullanim-Kilavuzu.pdf
│   └── Dockerfile                # Backend container tanımı
│
├── frontend/
│   ├── public/
│   │   └── index.html
│   ├── src/
│   │   ├── App.jsx               # Soru-cevap UI
│   │   └── index.js
│   ├── package.json
│   └── Dockerfile                # React container tanımı
│
├── models/
│   └── i20_index.faiss           # FAISS index dosyası
│
├── docker-compose.yml            # Backend, frontend, ollama servisleri
└── README.md                     # Proje açıklaması, kurulum adımları
