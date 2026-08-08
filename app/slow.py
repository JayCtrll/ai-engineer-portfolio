import time
import asyncio

def sync_task(seconds: int = 2):
    time.sleep(seconds)
    return f"Sync task finished after {seconds}s"

async def async_task(seconds: int = 2):
    await asyncio.sleep(seconds)
    return f"Async task finished after {seconds}s"