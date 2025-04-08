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
