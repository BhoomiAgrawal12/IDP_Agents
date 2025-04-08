<<<<<<< HEAD
from fastapi import FastAPI, UploadFile, Form, Depends, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordBearer
import logging
from document_processor import extract_text_from_file
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from agent_manager import run_agent
from pydantic import BaseModel
from deidentification import deidentify_text
from supabase import create_client, Client

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

async def get_current_user(token: str = Depends(oauth2_scheme)):
    # Implement your user authentication logic here
    # This is a basic example
    if token != "valid_token":
        raise HTTPException(status_code=401, detail="Invalid authentication")
    return {"username": "test_user"}
=======
from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import traceback
from typing import List, Optional
import io
import random

# No torch, transformers or numpy imports
>>>>>>> c5fc23132d7b94787782643c67a2329d1579c46a

app = FastAPI()
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


class DocInput(BaseModel):
    content: str


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
async def log_rate_limit(request: Request, call_next):
    response = await call_next(request)
    if response.status_code == 429:
        logging.warning(f"Rate limited: {request.url.path} from {request.client.host}")
    return response

<<<<<<< HEAD
@app.post("/process")
@limiter.limit("5/minute")
async def process_docs(query: str = Form(...), files: list[UploadFile] = Form(...)):
    full_text = ""
    for file in files:
        contents = await file.read()
        full_text += extract_text_from_file(contents, file.filename)

    structured_output = run_agent(query, full_text)
    return JSONResponse(content={"structured_output": structured_output})

@app.get("/secure-data")
async def secure_data(current_user: dict = Depends(get_current_user)):
    return {"message": f"Hello, {current_user['username']}"}

@app.post("/api/deidentify")
def deidentify_and_store(doc: DocInput):
    clean_text, entity_map = deidentify_text(doc.content)

    supabase.table("documents").insert({
        "original": doc.content,
        "deidentified": clean_text,
        "entity_map": entity_map  # store as json
    }).execute()

    return {
        "deidentified": clean_text,
        "entities": entity_map
    }
=======
# Flag variables for optional features
try:
    import fitz  # PyMuPDF
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

try:
    from PIL import Image
    import pytesseract
    IMAGE_OCR_SUPPORT = True
except ImportError:
    IMAGE_OCR_SUPPORT = False

# Mock LLM responses - No actual model needed
def get_ai_response(query, context=""):
    """Generate a mock AI response based on query and context"""
    # Predefined responses for common queries
    general_responses = [
        "Based on my analysis, the information you're looking for can be found in the provided documents.",
        "After reviewing the documents, I can see several relevant points related to your query.",
        "The documents contain information that directly addresses your question.",
        "Looking at the files you shared, I can provide the following insights.",
        "According to the information in the documents, there are multiple aspects to consider."
    ]
    
    # If no context, provide a general answer
    if not context.strip():
        if "what" in query.lower():
            return f"To answer your question about '{query}', I'd need more specific information. Could you provide more details or documents?"
        elif "how" in query.lower():
            return f"Regarding how to {query.split('how')[1].strip() if 'how' in query else query}, there are several approaches you could take. Without specific documents, I can only provide general guidance."
        elif "?" in query:
            return f"That's an interesting question. To give you the most accurate answer, I'd need some reference documents or more context."
        else:
            return f"I've analyzed your query: '{query}'. To provide a more tailored response, please upload relevant documents or provide additional details."
    
    # With context, craft a more specific response
    else:
        # Extract some content from the context to make it seem like it was analyzed
        context_preview = context[:500].replace("\n", " ")
        words = context_preview.split()
        
        # Select random words to highlight in the response
        if len(words) > 10:
            selected_words = random.sample([w for w in words if len(w) > 4], min(5, len([w for w in words if len(w) > 4])))
        else:
            selected_words = words[:2] if words else ["content"]
            
        # Create a more personalized response
        return f"""
Based on the documents you provided, I found information relevant to your query about '{query}'.

The documents mention key terms like {', '.join([f'"{w}"' for w in selected_words])} which relate to your question.

From analyzing the content, I can see that the documents contain approximately {len(context.split())} words and cover topics that appear relevant to your inquiry.

For a more detailed analysis, I would recommend focusing on the sections that specifically mention these key terms.
"""

# Enhanced function to extract text from various file types
def extract_text_from_file(contents, filename):
    try:
        # Text files
        if filename.lower().endswith(('.txt', '.md', '.py', '.js', '.html', '.css', '.json', '.csv')):
            return contents.decode('utf-8', errors='ignore')
        
        # PDF files
        elif filename.lower().endswith('.pdf') and PDF_SUPPORT:
            try:
                doc = fitz.open(stream=contents, filetype="pdf")
                text = ""
                for page_num, page in enumerate(doc):
                    text += f"[Page {page_num+1}]\n{page.get_text()}\n\n"
                return text
            except Exception as e:
                return f"Error extracting PDF text: {str(e)}"
        
        # Image files with OCR
        elif filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')) and IMAGE_OCR_SUPPORT:
            try:
                image = Image.open(io.BytesIO(contents))
                text = pytesseract.image_to_string(image)
                return text if text.strip() else "No text detected in image."
            except Exception as e:
                return f"Error performing OCR on image: {str(e)}"
                
        else:
            extensions = []
            if not PDF_SUPPORT and filename.lower().endswith('.pdf'):
                extensions.append("PDF (install PyMuPDF)")
            if not IMAGE_OCR_SUPPORT and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                extensions.append("Images (install PIL and pytesseract)")
            
            if extensions:
                return f"File format not supported without additional libraries: {', '.join(extensions)}"
            else:
                return f"Unsupported file format: {filename}"
    except Exception as e:
        return f"Error extracting text from {filename}: {str(e)}"

@app.post("/process")
async def process_docs(query: str = Form(...), files: Optional[List[UploadFile]] = Form(None)):
    try:
        if not files:
            files = []
            
        full_text = ""
        file_details = []
        total_size_kb = 0
        
        for file in files:
            try:
                contents = await file.read()
                size_kb = len(contents) / 1024
                total_size_kb += size_kb
                file_details.append({
                    "name": file.filename, 
                    "size_kb": round(size_kb, 1),
                    "type": file.content_type
                })
                
                extracted_text = extract_text_from_file(contents, file.filename)
                full_text += f"\n--- BEGIN CONTENT FROM {file.filename} ---\n"
                full_text += extracted_text
                full_text += f"\n--- END CONTENT FROM {file.filename} ---\n\n"
                
                # Reset file position for potential future reads
                await file.seek(0)
            except Exception as e:
                full_text += f"Error processing file {file.filename}: {str(e)}\n\n"
        
        # Generate AI response using our mock function
        ai_response = get_ai_response(query, full_text)
        
        # Format a nice response with file details
        formatted_response = ai_response
        if file_details:
            file_info = "\n\n---\n\nAnalyzed Files:\n"
            for i, fd in enumerate(file_details, 1):
                file_info += f"{i}. {fd['name']} ({fd['size_kb']} KB)\n"
            formatted_response += file_info
            
        return JSONResponse(content={"structured_output": formatted_response.strip()})
    except Exception as e:
        error_traceback = traceback.format_exc()
        return JSONResponse(
            status_code=500,
            content={
                "structured_output": f"An error occurred while processing your request: {str(e)}",
                "error": str(e)
            }
        )

@app.get("/health")
async def health_check():
    features = {
        "pdf_support": PDF_SUPPORT,
        "image_ocr": IMAGE_OCR_SUPPORT
    }
    return {
        "status": "Backend server is running!",
        "supported_features": features
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
>>>>>>> c5fc23132d7b94787782643c67a2329d1579c46a
