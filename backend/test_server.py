from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

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
    try:
        # Simple echo response to verify connection
        file_names = [file.filename for file in files]
        
        return JSONResponse(content={
            "structured_output": f"Received your query: '{query}' and {len(files)} files: {', '.join(file_names)}"
        })
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "structured_output": f"An error occurred: {str(e)}",
                "error": str(e)
            }
        )

@app.get("/health")
async def health_check():
    return {"status": "Backend server is running!"} 

