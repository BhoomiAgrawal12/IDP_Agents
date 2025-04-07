import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io

def extract_text_from_file(file_bytes, filename):
    if filename.lower().endswith(".pdf"):
        return extract_from_pdf(file_bytes)
    elif filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        return extract_from_image(file_bytes)
    return "Unsupported file format"

def extract_from_pdf(file_bytes):
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def extract_from_image(file_bytes):
    image = Image.open(io.BytesIO(file_bytes))
    text = pytesseract.image_to_string(image)
    return text
