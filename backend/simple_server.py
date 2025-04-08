from fastapi import FastAPI, UploadFile, Form, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from typing import List, Optional

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/process")
async def process_docs(query: str = Form(...), files: Optional[List[UploadFile]] = Form(None)):
    try:
        # Handle case when no files are sent
        if not files:
            files = []
            
        # Extract basic info from files
        file_info = []
        total_size = 0
        
        for file in files:
            contents = await file.read()
            size_kb = len(contents) / 1024
            file_info.append(f"{file.filename} ({size_kb:.1f} KB)")
            total_size += size_kb
            
            # Reset file position for potential future reads
            await file.seek(0)
        
        # Create a simple response
        response_text = f"""
I processed your query: "{query}"

Files analyzed ({len(files)}):
{", ".join(file_info) if file_info else "No files uploaded"}

Total data processed: {total_size:.1f} KB

Analysis result:
These files appear to contain information relevant to your query. 
I'd recommend focusing on the key points mentioned in the documents 
and considering how they relate to your specific question.
"""
        
        return JSONResponse(content={"structured_output": response_text.strip()})
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

if __name__ == "__main__":
    uvicorn.run("simple_server:app", host="0.0.0.0", port=8000, reload=True) 