import fitz
from PIL import Image
import numpy as np
import easyocr

def extract_text_with_fallback(pdf_path: str, n: int = None):
    doc = fitz.open(pdf_path)
    page_count = doc.page_count if n is None else min(n, doc.page_count)
    texts = []

    # EasyOCR modelini bir kere yükle, tekrar tekrar başlatma!
    reader = easyocr.Reader(['tr'])

    for i in range(page_count):
        page = doc.load_page(i)
        text = page.get_text("text").strip()
        # Bozuk/eksik sayfalarda OCR fallback
        if any(ch in text for ch in ["�", "♥", "Ô", "Ę"]):
            print(f"Sayfa {i+1} için EasyOCR kullanılıyor...")
            pix = page.get_pixmap(dpi=200)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            # EasyOCR numpy array ister:
            img_np = np.array(img)
            result = reader.readtext(img_np, detail=0, paragraph=True)
            text = "\n".join(result)
        texts.append(text)
    doc.close()
    return texts

# Test için:
if __name__ == "__main__":
    metinler = extract_text_with_fallback("i20-Kullanim-Kilavuzu.pdf", n=4)
    print(metinler)
# docker run -it --rm -v %cd%:/app hyundai-backend python extract.py