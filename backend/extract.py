import fitz
from PIL import Image
import pytesseract

def extract_text_with_fallback(pdf_path: str, n: int = None):
    doc = fitz.open(pdf_path)
    page_count = doc.page_count if n is None else min(n, doc.page_count)
    texts = []
    for i in range(page_count):
        page = doc.load_page(i)
        text = page.get_text("text").strip()
        if any(ch in text for ch in ["�", "♥", "Ô", "Ę"]):
            pix = page.get_pixmap(dpi=200)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            text = pytesseract.image_to_string(img, lang="tur")
        texts.append(text)
    doc.close()
    return texts

# Test için (isteğe bağlı):
if __name__ == "__main__":
    texts = extract_text_with_fallback("i20-Kullanim-Kilavuzu.pdf", n=2)
    print(texts[:1])
