from fastapi import FastAPI

app = FastAPI(
    title="AI Portfolio API",
    version="0.1.0",
    openapi_tags=[
        {"name": "System", "description": "System health and status endpoints"},
    ],
)


@app.get("/health", tags=["System"])
def health():
    return {"status": "ok", "message": "Service is running"}
