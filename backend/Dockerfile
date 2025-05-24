FROM python:3.10-slim

# 1. Çalışma dizinini oluştur
WORKDIR /app

# 2. Gerekli dosyaları kopyala
COPY requirements.txt .

# 3. Bağımlılıkları yükle
RUN pip install --no-cache-dir -r requirements.txt

# Tesseract OCR'yi yüklemek için gerekli paketleri kur
RUN apt-get update && apt-get install -y tesseract-ocr && apt-get clean

# 4. Tüm backend kodunu kopyala
COPY . .

# 5. Giriş komutu (main.py FastAPI app için)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
