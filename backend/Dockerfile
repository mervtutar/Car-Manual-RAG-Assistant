FROM python:3.10-slim

WORKDIR /app


# Gerekli dosyaları kopyala
COPY requirements.txt .

# pip’i güncelle ve requirements’ı kur
RUN pip install --upgrade pip \
    && pip install --no-cache-dir --default-timeout=300 -i https://pypi.org/simple -r requirements.txt


# Uygulama kodunu kopyala
COPY . .

# FastAPI server başlat
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
