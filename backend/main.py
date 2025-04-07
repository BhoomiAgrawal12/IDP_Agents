from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from document_processor import extract_text_from_file
from agent_manager import run_agent

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/process")
async def process_docs(query: str = Form(...), files: list[UploadFile] = Form(...)):
    full_text = ""
    for file in files:
        contents = await file.read()
        full_text += extract_text_from_file(contents, file.filename)

    structured_output = run_agent(query, full_text)
    return JSONResponse(content={"structured_output": structured_output})
