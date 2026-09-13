# ====================== 标准库导入 ======================
from typing import Optional
import os
import logging
import time
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# ====================== 第三方库导入 ======================
from fastapi import FastAPI, Depends, UploadFile, File, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# ====================== 本地项目模块导入 ======================
from app.middleware import log_middleware
from app.auth import verify_token
from app.slow import sync_task, async_task
from app.rag_service import load_or_create_vectorstore, build_vectorstore_from_directory, answer_question

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 初始化FastAPI应用实例
app = FastAPI(
    title="AI Portfolio API",
    version="0.2.0",
    openapi_tags=[
        {"name": "System", "description": "系统健康检测、服务状态相关接口"},
        {"name": "Echo", "description": "消息回声测试接口，支持GET/POST两种传参方式"},
        {"name": "Auth", "description": "Token鉴权、受保护资源访问接口"},
        {"name": "Performance", "description": "同步/异步耗时性能测试接口"},
        {"name": "News RAG", "description": "新闻RAG问答：基于MinIO清洗后的新闻向量库检索并生成答案"},
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

class QuestionRequest(BaseModel):
    query: str
    session_id: str

class NewsAskRequest(BaseModel):
    """
    新闻RAG问答请求体模型

    Attributes:
        query: 用户提问文本，必填
        top_k: 返回引用的新闻条数，默认3，范围1~10
        use_llm: 是否使用LLM生成答案，默认True；设为False时仅返回检索到的相关新闻
    """
    query: str
    top_k: int = Field(default=3, ge=1, le=10)
    use_llm: bool = True

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

# 确保data目录存在
DATA_DIR = "./data"
os.makedirs(DATA_DIR, exist_ok=True)

# 请求体Pydantic校验模型
class AskRequest(BaseModel):
    question: str

@app.post("/upload", tags=["RAG"])
async def upload_pdf(file: UploadFile = File(...)):
    """
    PDF文件上传接口，支持PDF文件上传并存储到本地data目录
    
    Args:
        file: 上传的PDF文件
        
    Returns:
        dict: 上传成功后的提示信息
    """
        # 只允许pdf
    if not file.filename.lower().endswith(".pdf"):
        return {"error": "only pdf file allowed"}

    # ✅ 新增：读取文件内容并校验非空
    file_content = await file.read()
    if len(file_content) == 0:
        return {"error": "uploaded file is empty, please check the pdf file"}
        
    save_path = os.path.join(DATA_DIR, file.filename)
    # 保存文件到磁盘
    with open(save_path, "wb") as f:
        f.write(file_content)

    # 加载PDF文档
    loader = PyPDFLoader(save_path)
    docs = loader.load()

    # 文档切分
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )
    split_docs = text_splitter.split_documents(docs)

    # 获取向量库对象，追加文档
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_chroma import Chroma
    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    vector_db = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embedding
    )
    vector_db.add_documents(split_docs)
    return {
        "msg": "upload and vector update success",
        "filename": file.filename,
        "chunk_count": len(split_docs)
    }

@app.post("/reset-db",tags=["RAG"])
def reset_vector_db():
    import shutil
    db_path = "./chroma_db"
    if os.path.exists(db_path):
        shutil.rmtree(db_path)
    return {"msg":"vector database cleared"}

@app.post("/rebuild-index", tags=["RAG"])
async def rebuild_index():
    try:
        build_vectorstore_from_directory()
        return {"status": "success", "message": "Vector index rebuilt"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# @app.post("/ask", tags=["RAG"])
# async def ask_rag(req: AskRequest):
#     """
#     RAG问答接口，接收 {"question":"xxx"} 返回答案
#     """
#     from app.rag_service import answer_question
#     answer = answer_question(req.question)
#     return {
#         "question": req.question,
#         "answer": answer
#     }

@app.post(
    "/rag/ask",
    tags=["RAG Chat"],
    summary="多轮RAG问答接口",
    description="""
基于知识库执行RAG问答，支持多轮对话。
1. session_id 作为会话标识，内存保存聊天历史，服务重启后会话丢失；
2. 内部流程：query改写 → 向量库检索top6文档 → CrossEncoder重排序取top3 → LLM结合上下文与历史生成答案；
3. 严格遵循知识库约束：上下文找不到答案固定返回 `I don't know`；
""",
)
def rag_ask(req: QuestionRequest):
    """
    :param req: 请求体包含用户query与session_id会话标识
    :return: 返回会话ID、原始提问、RAG生成回答
    """
    answer = answer_question(query=req.query, session_id=req.session_id)
    return {
        "session_id": req.session_id,
        "query": req.query,
        "answer": answer
    }

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {exc}", exc_info=True)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})

@app.delete(
    "/rag/session/{session_id}",
    tags=["RAG Chat"],
    summary="清除指定会话历史",
    description="删除内存中指定session_id的全部对话记录。不存在会话不会报错。内存存储，服务重启自动清空全部会话。",
)
def delete_session(session_id: str):
    if session_id in store:
        del store[session_id]
        logger.info(f"[Session] deleted session: {session_id}")
        return {"session_id": session_id, "status": "deleted"}
    logger.warning(f"[Session] delete session not found: {session_id}")
    return {"session_id": session_id, "status": "not_found"}

@app.post(
    "/news/ask",
    tags=["News RAG"],
    summary="新闻RAG问答接口",
    description="""
基于新闻向量库执行RAG问答，数据来自MinIO中spark_cleaner清洗后的news_cleaned.parquet。
1. 内部流程：新闻向量库相似性检索 top_k 条 → Ollama LLM 基于检索结果生成答案；
2. 返回 answer（生成答案）与 sources（引用的新闻列表，含标题/来源/链接/发布时间/情感/相关度分）；
3. use_llm=False 时仅返回检索到的相关新闻列表，不调用LLM；
4. 依赖本地 MinIO + Ollama，部署环境未安装对应依赖时调用将返回500；
""",
)
def news_ask(req: NewsAskRequest) -> dict:
    """
    新闻RAG问答接口

    Args:
        req: NewsAskRequest，包含query提问、top_k引用条数、use_llm是否生成答案

    Returns:
        dict: 包含query、answer（LLM生成答案或降级提示）和sources（引用的新闻列表）
    """
    # 懒加载：news_rag 依赖 pyspark/MinIO，仅在本地开发环境安装，
    # 避免在 Docker 部署环境因缺少依赖导致服务启动失败
    from app.news_rag import get_news_answer
    return get_news_answer(query=req.query, top_k=req.top_k, use_llm=req.use_llm)

# 注册全局HTTP日志中间件
app.middleware("http")(log_middleware)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
