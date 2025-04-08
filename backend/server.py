from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import requests

app = FastAPI()

# Allow CORS (Important for frontend access)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Storage for result
result_storage = {"result": ""}

# Your endpoint for receiving query
@app.post("/get_result/")
async def get_result(request: Request):
    data = await request.json()
    query = data.get("query")

    # Now send this query to your Google Colab (example below)
    response = requests.post('https://af2e-34-143-188-208.ngrok-free.app/colab_process/', json={"query": query})
    
    colab_result = response.json()["result"]
    result_storage["result"] = colab_result  # Save it

    return {"result": colab_result}

# Your endpoint for fetching the latest result
@app.get("/fetch_result/")
async def fetch_result():
    return {"result": result_storage["result"]}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
