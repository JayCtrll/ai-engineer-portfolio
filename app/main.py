# ====================== 标准库导入 ======================
from typing import Optional

# ====================== 第三方库导入 ======================
from fastapi import FastAPI, Depends
from pydantic import BaseModel, Field, EmailStr

# ====================== 本地项目模块导入 ======================
from app.middleware import log_middleware
from app.auth import verify_token
from app.slow import sync_task, async_task

# 初始化FastAPI应用实例
app = FastAPI(
    title="AI Portfolio API",
    version="0.1.0",
    openapi_tags=[
        {"name": "System", "description": "系统健康检测、服务状态相关接口"},
        {"name": "Echo", "description": "消息回声测试接口，支持GET/POST两种传参方式"},
        {"name": "Auth", "description": "Token鉴权、受保护资源访问接口"},
        {"name": "Performance", "description": "同步/异步耗时性能测试接口"},
    ],
)


class EchoRequest(BaseModel):
    """
    Echo接口统一请求体模型（POST请求体 / GET查询参数复用）

    Attributes:
        message: 输入消息文本，必填字段
        sender: 发送者标识，可选，不传为None
        priority: 消息优先级，取值范围1~5，默认值1
    """
    message: str
    sender: Optional[str] = None
    priority: int = Field(default=1, ge=1, le=5)


@app.post(
    "/echo",
    tags=["Echo"],
    summary="POST 消息回声接口",
    description="接收JSON格式请求体，读取消息、发送人、优先级参数，返回带Echo前缀的回显消息，适用于复杂参数提交场景。"
)
async def echo(request: EchoRequest) -> dict:
    """
    POST 回声接口，接收JSON请求体并原样回传消息，附带回声字符串

    Args:
        request: EchoRequest 模型，包含message、sender、priority请求参数

    Returns:
        dict: 包含原始消息、发送人、优先级、拼接后的回声文本
    """
    return {
        "received_message": request.message,
        "from": request.sender,
        "priority": request.priority,
        "echo": f"Echo: {request.message}"
    }


@app.get(
    "/echo",
    tags=["Echo"],
    summary="GET 消息回声接口",
    description="通过URL Query查询参数传递消息、发送人、优先级，服务端直接回显内容，适合快速简单调试。"
)
async def get_echo(request: EchoRequest = Depends()) -> dict:
    """
    GET 回声接口，通过URL查询参数接收数据并返回回声结果

    Args:
        request: 由Depends自动解析URL查询参数映射为EchoRequest模型

    Returns:
        dict: 包含原始消息、发送人、优先级、拼接后的回声文本
    """
    return {
        "received_message": request.message,
        "from": request.sender,
        "priority": request.priority,
        "echo": f"Echo: {request.message}"
    }


@app.get(
    "/secure-data",
    tags=["Auth"],
    summary="获取鉴权保护数据",
    description="接口强制校验请求携带的Token，Token合法才可访问内部私密测试数据；依赖verify_token完成统一身份校验逻辑。"
)
async def get_secure_data(token: str = Depends(verify_token)) -> dict:
    """
    鉴权保护接口，仅携带合法Token可访问，返回加密测试数据

    Args:
        token: 依赖verify_token校验函数，自动从请求头提取并校验token，校验通过返回token字符串

    Returns:
        dict: 受保护的密钥文本、当前校验通过的token
    """
    return {"secret": "This is protected data", "token_used": token}


@app.get(
    "/sync-slow",
    tags=["Performance"],
    summary="同步阻塞耗时测试",
    description="调用同步阻塞任务模拟CPU/IO长耗时操作，会阻塞FastAPI事件循环，并发访问性能较差，用于对比异步接口性能差异。"
)
def sync_endpoint() -> dict:
    """
    同步耗时测试接口，调用阻塞式同步任务模拟长耗时操作

    Returns:
        dict: 同步任务执行完成后的返回结果
    """
    result = sync_task(2)
    return {"result": result}


@app.get(
    "/async-slow",
    tags=["Performance"],
    summary="异步非阻塞耗时测试",
    description="调用异步IO任务模拟长耗时操作，不阻塞服务事件循环，高并发场景吞吐量远高于同步接口，用于性能基准对比。"
)
async def async_endpoint() -> dict:
    """
    异步耗时测试接口，调用非阻塞异步任务模拟长耗时操作，不阻塞事件循环

    Returns:
        dict: 异步任务执行完成后的返回结果
    """
    result = await async_task(2)
    return {"result": result}


@app.get(
    "/health",
    tags=["System"],
    summary="服务健康探活检测",
    description="基础存活检测接口，无业务逻辑，负载均衡、监控系统定时调用判断服务是否正常运行。"
)
def health() -> dict:
    """
    服务健康检测接口，用于监控、负载均衡探活

    Returns:
        dict: 服务运行状态标识与提示信息
    """
    return {"status": "ok", "message": "Service is running"}


# 注册全局HTTP日志中间件
app.middleware("http")(log_middleware)