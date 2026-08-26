from fastapi import Header, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from dotenv import load_dotenv
import os
import logging
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
API_TOKEN = os.getenv("API_TOKEN", default="my-secret-token-123")
security_scheme = HTTPBearer(auto_error=False)

async def verify_token(creds: HTTPAuthorizationCredentials = Depends(security_scheme)):
    if creds is None:
        raise HTTPException(status_code=401, detail="Authorization header missing")
    if creds.credentials != API_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid or missing token")
    return creds.credentials