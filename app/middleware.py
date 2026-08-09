# 标准库导入
import time
import logging
from typing import Awaitable, Callable

# 第三方库导入
from fastapi import Request
from fastapi.responses import Response

# 日志初始化配置
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


async def log_middleware(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """
    FastAPI 全局请求日志中间件
    记录每个HTTP请求的请求方法、路由路径、响应状态码、接口耗时（毫秒）

    Args:
        request: FastAPI 原生请求对象，包含请求方法、URL、请求头、参数等信息
        call_next: 接收Request并返回异步Response的回调函数，用于执行后续路由逻辑

    Returns:
        Response: 接口处理完成后的原生响应对象，包含状态码、响应体等数据
    """
    start_time = time.time()
    # 执行后续接口逻辑，获取响应
    response = await call_next(request)
    # 计算接口耗时，转换为毫秒并保留2位小数
    process_time = (time.time() - start_time) * 1000

    logger.info(
        f"{request.method} {request.url.path} - "
        f"Status: {response.status_code} - "
        f"Time: {process_time:.2f}ms"
    )
    return response