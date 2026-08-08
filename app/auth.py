from fastapi import Header, HTTPException, Depends

API_TOKEN = "my-secret-token-123"

async def verify_token(authorization: str = Header(None)):
    if authorization is None:
        raise HTTPException(status_code=401, detail="Authorization header missing")
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or token != API_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid or missing token")
    return token