
# Hyundai i20 Smart Q&A System

This project is an AI-powered assistant (RAG-based QA) that reads and understands the **Hyundai i20** user manual, and can instantly answer technical questions in natural Turkish language.

```plaintext
Car-Manual-RAG-Assistant/
│
├── backend/
│   ├── main.py                   # FastAPI app, hybrid RAG pipeline (QA backend)
│   ├── extract.py                # PDF text extraction + EasyOCR fallback
│   ├── chunk.py                  # Sentence-based chunking utility
│   ├── embed_index.py            # Embedding and FAISS indexing script
│   ├── requirements.txt          # Python dependencies
│   ├── i20_index.faiss           # FAISS vector index (generated)
│   ├── chunks.pkl                # Chunk metadata (generated)
│   ├── i20-Kullanim-Kilavuzu.pdf # Hyundai i20 user manual (PDF)
│   └── Dockerfile                # Backend Docker container definition
│
├── frontend/
│   ├── public/
│   │   └── index.html
│   ├── src/
│   │   ├── App.js                # React Q&A interface
│   │   └── index.js
│   ├── package.json              # Frontend npm dependencies & scripts
│
└── README.md                     # Project description, installation & usage instructions


```

![i20](https://github.com/user-attachments/assets/01a86ad1-77ba-472a-88ab-b608b2129bf9)


---

## Features

- **PDF Document Text Extraction**
    - Extracts text from the user manual; falls back to EasyOCR if any pages are corrupted or missing.
- **Sentence-Based Chunking**
    - Splits the text into overlapping, sentence-level chunks for better context handling.
- **Hybrid Search (FAISS Vector + BM25)**
    - Retrieves the most relevant chunks using both vector similarity and keyword-based scoring.
- **Extractive + Generative QA**
    - Attempts to extract the best answer directly; then uses a Turkish GPT2 LLM to generate a concise, technical response.
- **Transparent Sourcing**
    - Shows the related chunk and page number alongside every answer.
- **Modern Web UI (React)**
    - Users can ask questions, view history, and see chunk sources in a clean web interface.

**Models used:**

Embedding: paraphrase-multilingual-MiniLM-L12-v2

Extractive: dbmdz/bert-base-turkish-cased

Generative: ytu-ce-cosmos/turkish-gpt2-large

---

### Requirements

- Python 3.9+
- Node.js and npm (for frontend)
- CUDA-enabled GPU (recommended but not required)
- The `i20-Kullanim-Kilavuzu.pdf` file must be present

## Usage

### 1. Running the Backend

Make sure you have completed the embedding and index creation steps (see Installation).  
Then, start the backend API server:

```shell
uvicorn main:app --host 0.0.0.0 --port 8000
```

### 2. Running the Frontend
```
npm install
npm start
```
The web interface will be available at http://localhost:3000.
