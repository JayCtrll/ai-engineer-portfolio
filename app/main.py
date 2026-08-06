from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Optional

app = FastAPI(
    title="AI Portfolio API",
    version="0.1.0",
    openapi_tags=[
        {"name": "System", "description": "System health and status endpoints"},
    ],
)
class EchoRequest(BaseModel):
    message: str
    sender: Optional[str] = None
    priority: int = Field(default=1, ge=1, le=5)

@app.post("/echo", tags=["Echo"])
async def echo(request: EchoRequest):
    return {
        "received_message": request.message,
        "from": request.sender,
        "priority": request.priority,
        "echo": f"Echo: {request.message}"
    }

@app.get("/health", tags=["System"])
def health():
    return {"status": "ok", "message": "Service is running"}
