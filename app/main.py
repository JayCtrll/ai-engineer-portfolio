from fastapi import FastAPI,Depends,Query
from pydantic import BaseModel, Field,EmailStr 
from typing import Optional
from app.middleware import log_middleware
from app.auth import verify_token
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
    priority: int = Field(default=1, ge=1, le=5) ###字符串正则用pattern
    ###email:EmailStr 邮箱校验

@app.post("/echo", tags=["Echo"])
async def echo(request: EchoRequest):
    return {
        "received_message": request.message,
        "from": request.sender,
        "priority": request.priority,
        "echo": f"Echo: {request.message}"
    }

@app.get("/echo", tags=["Echo"])
async def get_echo(request: EchoRequest = Depends()):
    return {
        "received_message": request.message,
        "from": request.sender,
        "priority": request.priority,
        "echo": f"Echo: {request.message}"
    }

@app.get("/secure-data", tags=["Auth"])
async def get_secure_data(token: str = Depends(verify_token)):
    return {"secret": "This is protected data", "token_used": token}


@app.get("/health", tags=["System"])
def health():
    return {"status": "ok", "message": "Service is running"}

app.middleware("http")(log_middleware)
