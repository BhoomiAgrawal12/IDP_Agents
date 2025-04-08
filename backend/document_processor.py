import fitz 
import pytesseract
from PIL import Image
import io

def extract_text_from_file(file_bytes, filename):
    try:
        if filename.lower().endswith(".pdf"):
            return extract_from_pdf(file_bytes)
        elif filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            return extract_from_image(file_bytes)
        elif filename.lower().endswith(('.txt', '.md', '.csv', '.json')):
            # For text-based files
            return file_bytes.decode('utf-8', errors='ignore')
        return f"Processed file: {filename} (Unsupported format for text extraction)"
    except Exception as e:
        return f"Error processing {filename}: {str(e)}"

def extract_from_pdf(file_bytes):
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        return f"Error extracting PDF text: {str(e)}"

def extract_from_image(file_bytes):
    try:
        image = Image.open(io.BytesIO(file_bytes))
        text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        return f"Error extracting image text: {str(e)}"
