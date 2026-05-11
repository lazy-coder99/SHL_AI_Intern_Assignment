from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from agent import process_chat

app = FastAPI(title="SHL Conversational Assessment Recommender")

class ChatRequest(BaseModel):
    messages: List[Dict[str, str]]

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    if not request.messages:
        raise HTTPException(status_code=400, detail="Messages list cannot be empty")
    
    try:
        # Agent processes the stateless conversation history
        response = process_chat(request.messages)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
