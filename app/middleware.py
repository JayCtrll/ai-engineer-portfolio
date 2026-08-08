import time
import logging
from fastapi import Request

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

async def log_middleware(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = (time.time() - start_time) * 1000
    logger.info(
        f"{request.method} {request.url.path} - "
        f"Status: {response.status_code} - "
        f"Time: {process_time:.2f}ms"
    )
    return response