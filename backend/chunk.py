import nltk
from nltk.tokenize import sent_tokenize

# NLTK tokenizer verisi yoksa indir (hem lokal, hem docker için güvenli)
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

def chunk_text_sentences(texts, sentences_per_chunk=5, overlap=1):
    """
    texts: Sayfa sayfa metinlerin listesi
    sentences_per_chunk: Her chunk'ta kaç cümle olsun?
    overlap: Chunk'lar arasında kaç cümle tekrar etsin?
    """
    chunks = []
    for page_no, text in enumerate(texts, 1):
        sentences = sent_tokenize(text)
        i = 0
        while i < len(sentences):
            chunk = " ".join(sentences[i : i + sentences_per_chunk])
            if chunk.strip():
                chunks.append({"text": chunk, "page": page_no})
            i += sentences_per_chunk - overlap
    return chunks

# Test amaçlı
if __name__ == "__main__":
    from extract import extract_text_with_fallback
    metinler = extract_text_with_fallback("i20-Kullanim-Kilavuzu.pdf", n=4)
    chunklar = chunk_text_sentences(metinler, sentences_per_chunk=5, overlap=1)
    for idx, chunk in enumerate(chunklar[:5], 1):
        print(f"--- Chunk {idx} (Sayfa {chunk['page']}) ---\n{chunk['text']}\n")
