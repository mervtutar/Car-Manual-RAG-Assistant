FROM python:3.10-slim

WORKDIR /app

# 1. Sisteme bağımlı paketleri kur
RUN apt-get update && apt-get install -y tesseract-ocr && apt-get clean

# 2. Gerekli dosyaları kopyala
COPY requirements.txt .

# 3. pip’i güncelle ve requirements’ı kur
RUN pip install --upgrade pip
RUN pip install --default-timeout=300 --no-cache-dir -r requirements.txt

# 4. Uygulama kodunu kopyala
COPY . .

# 5. FastAPI server başlat
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
