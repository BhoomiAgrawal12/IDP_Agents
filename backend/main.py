from fastapi import FastAPI, UploadFile, Form, Depends, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordBearer
import logging
from typing import List
from document_processor import extract_text_from_file
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from agent_manager import run_agent
from pydantic import BaseModel
from deidentification import deidentify_text
import firebase_admin
from firebase_admin import credentials, firestore
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
import magic
import pyclamd
import os
import dotenv


dotenv.load_dotenv()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

async def get_current_user(token: str = Depends(oauth2_scheme)):
    # Implement your user authentication logic here
    # This is a basic example
    if token != "valid_token":
        raise HTTPException(status_code=401, detail="Invalid authentication")
    return {"username": "test_user"}

app = FastAPI()
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

cred = credentials.Certificate(os.getenv("FIREBASE_CREDENTIALS"))  
firebase_admin.initialize_app(cred)
db = firestore.client()

analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()
file_magic = magic.Magic(mime=True)  # Initialize file type checker
class DocInput(BaseModel):
    content: str

class RefinementRequest(BaseModel):
    originalQuery: str
    feedback: str

class SecurityReport(BaseModel):
    anonymizedFields: List[str]
    totalAnonymized: int

class ProcessResult(BaseModel):
    dataset: List[dict]
    qualityIssues: List[str]
    securityReport: SecurityReport

class RefinementResponse(BaseModel):
    newQuery: str
    dataset: List[dict] | None = None
    qualityIssues: List[str] | None = None
    securityReport: SecurityReport | None = None

def validate_file(file: UploadFile) -> None:
    """Validate that the file is a safe PDF."""
    # Step 1: Check file extension
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail=f"File {file.filename} is not a PDF")

    # Step 2: Verify MIME type
    file_content = file.file.read()
    mime_type = file_magic.from_buffer(file_content)
    if mime_type != "application/pdf":
        raise HTTPException(status_code=400, detail=f"File {file.filename} is not a valid PDF (MIME: {mime_type})")

    # Step 3: Scan for malware with ClamAV
    try:
        cd = pyclamd.ClamdUnixSocket()  # Connect to local ClamAV daemon
        scan_result = cd.scan_stream(file_content)
        if scan_result and scan_result["stream"][0] == "FOUND":
            raise HTTPException(status_code=400, detail=f"File {file.filename} contains malware: {scan_result['stream'][1]}")
    except pyclamd.ConnectionError:
        # Fallback if ClamAV isn't running (optional for demo)
        print("Warning: ClamAV not available, skipping malware scan")
    finally:
        file.file.seek(0)  # Reset file pointer for further processing    


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

@app.post("/process")
@limiter.limit("5/minute")
async def process_docs(request: Request, query: str = Form(...), files: list[UploadFile] = Form(...)):
    if not files or not query:
        raise HTTPException(status_code=400, detail="Files and query are required")
    
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

    doc_ref = db.collection("processed_docs").document()
    doc_ref.set({
        "query": query,
        "dataset": dataset,
        "security_report": consolidated_report.dict(),
        "timestamp": firestore.SERVER_TIMESTAMP
    })

    return {
        "deidentified": clean_text,
        "entities": entity_map
    }

