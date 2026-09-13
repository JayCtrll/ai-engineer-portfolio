"""
app/news_rag.py — 新闻 RAG 可调用模块

数据链路:
  s3a://spark-bucket/news/cleaned          (news_cleaned.parquet，由 spark_cleaner.py 生成)
    → PySpark 读取 → Pandas DataFrame
    → LangChain Document 包装（title + summary 作为 page_content，
      metadata: source / link / published / sentiment）
    → HuggingFaceEmbeddings (all-MiniLM-L6-v2) 向量化
    → 存入 Chroma 向量库 ./news_chroma_db
    → get_news_answer() 检索 + LLM 生成答案

对外 API:
  get_news_answer(query, top_k=3, use_llm=True) -> dict   # 供 FastAPI 直接调用
  get_vector_store() -> Chroma                             # 加载/构建向量库（惰性单例）
  rebuild_store() -> Chroma                                # 强制从 MinIO 重建向量库

FastAPI 调用示例:
  from app.news_rag import get_news_answer
  @app.post("/news/ask")
  async def ask(req: QuestionRequest):
      return get_news_answer(req.query)

运行（项目根目录，使用 fastapi-week1 环境）:
  /home/cjc/.pyenv/versions/fastapi-week1/bin/python app/news_rag.py
"""
import logging
import os
import re
import sys
from pathlib import Path

import pandas as pd
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from pyspark.sql import SparkSession

logger = logging.getLogger(__name__)

# ---------- MinIO 配置 ----------
MINIO_ENDPOINT = "http://127.0.0.1:9000"
MINIO_ACCESS_KEY = "minioadmin"
MINIO_SECRET_KEY = "minioadmin"
# spark_cleaner.py 清洗后的新闻 parquet（Spark 分区目录）
INPUT_PARQUET = "s3a://spark-bucket/news/cleaned"

# ---------- RAG 配置 ----------
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
PERSIST_DIR = "./news_chroma_db"
COLLECTION_NAME = "news_collection"
TOP_K = 3
# 相关性阈值：similarity_search_with_score 返回余弦距离（越小越相关，0~2）。
# 超过该阈值视为检索命中无关内容（如中文查询撞英文库），过滤后不送入 LLM。
# 实测：真相关新闻 < 1.0，半相关 ~1.2，无关内容 >= 1.3，故取 1.2 较平衡。
RELEVANCE_THRESHOLD = 1.2

# ---------- LLM 配置 ----------
OLLAMA_BASE_URL = "http://127.0.0.1:11434"
# 用 qwen3:4b：中文回答更稳定（qwen2:1.5b 对"答案语言跟随提问"指令执行不稳定）
LLM_MODEL = "qwen2:1.5b"

# hadoop-aws + aws-sdk 两个 jar 提供 s3a:// 协议支持
SPARK_JARS = (
    "/home/cjc/spark_jars/hadoop-aws-3.3.4.jar,"
    "/home/cjc/spark_jars/aws-java-sdk-bundle-1.12.262.jar"
)

# 新闻问答 Prompt：正向措辞引导直接作答，避免小模型过度触发拒答
# （经验：strict 写法 "reply exactly with I don't know" 会让 1.5B 级模型误拒答）
_NEWS_PROMPT = PromptTemplate(
    template=(
        "You are a helpful news assistant. Answer the question using ONLY the news "
        "context below.\n"
        "Start your answer with the direct facts from the context. "
        "If the context truly does not contain the answer, then say 'I don't know'.\n"
        "Do not make up information. When you use a news item, mention its title and "
        "source link.\n"
        "Answer in the same language as the question (e.g. if the question is in "
        "Chinese, answer in Chinese).\n\n"
        "News Context:\n{context}\n\n"
        "Question: {question}\n"
        "Answer:"
    ),
    input_variables=["context", "question"],
)

# 模块级单例：向量库 / 嵌入模型 / LLM，服务启动后只初始化一次
_vector_store: Chroma | None = None
_embeddings: HuggingFaceEmbeddings | None = None
_llm = None


# ============================ 数据读取（MinIO → Documents） ============================

def create_spark() -> SparkSession:
    """创建带 MinIO(S3A) 连接的 SparkSession。"""
    # 保证 Spark worker 与 driver 使用同一 Python 解释器，
    # 否则 createDataFrame 等操作会报 PYTHON_VERSION_MISMATCH
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
    return (
        SparkSession.builder
        .appName("NewsRAG-MinIO")
        .master("local[2]")
        .config("spark.jars", SPARK_JARS)
        .config("spark.hadoop.fs.s3a.endpoint", MINIO_ENDPOINT)
        .config("spark.hadoop.fs.s3a.access.key", MINIO_ACCESS_KEY)
        .config("spark.hadoop.fs.s3a.secret.key", MINIO_SECRET_KEY)
        # MinIO 要求 path-style 访问，不能用 virtual-host 方式
        .config("spark.hadoop.fs.s3a.path.style.access", "true")
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .getOrCreate()
    )


def load_news(spark: SparkSession) -> pd.DataFrame:
    """从 MinIO 读取新闻 parquet，转成 Pandas DataFrame。"""
    df = spark.read.parquet(INPUT_PARQUET)
    count = df.count()
    logger.info("[RAG] 从 MinIO 读取 %s 条新闻: %s", count, INPUT_PARQUET)
    return df.toPandas()


def to_documents(news: pd.DataFrame) -> list[Document]:
    """每条新闻包装为 LangChain Document。

    page_content = title + summary
    metadata     = source / link / published / sentiment
    """
    docs = []
    for _, row in news.iterrows():
        title = str(row.get("title") or "").strip()
        summary = str(row.get("summary") or "").strip()
        page_content = f"{title}\n{summary}" if title and summary else (title or summary)

        metadata = {
            "source": str(row.get("source") or ""),
            "link": str(row.get("link") or ""),
            "published": str(row.get("published") or ""),
            "sentiment": str(row.get("sentiment") or ""),
        }
        docs.append(Document(page_content=page_content, metadata=metadata))
    return docs


def _load_docs_from_minio() -> list[Document]:
    """从 MinIO 拉取新闻并包装为 Document 列表。"""
    spark = create_spark()
    try:
        news = load_news(spark)
        docs = to_documents(news)
        logger.info("[RAG] 已包装 %s 个 Document", len(docs))
        return docs
    finally:
        spark.stop()


# ============================ 向量库（构建 / 加载） ============================

def _get_embeddings() -> HuggingFaceEmbeddings:
    """惰性初始化嵌入模型（全局只加载一次）。"""
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings


def build_vector_store(docs: list[Document]) -> Chroma:
    """向量化并存入 Chroma；重复调用会重建库，避免旧数据残留。"""
    global _vector_store
    persist_path = Path(PERSIST_DIR)
    if persist_path.exists():
        logger.info("[RAG] 检测到已有向量库 %s，重建前先清理", PERSIST_DIR)
        import shutil
        shutil.rmtree(persist_path)

    _vector_store = Chroma.from_documents(
        documents=docs,
        embedding=_get_embeddings(),
        collection_name=COLLECTION_NAME,
        persist_directory=PERSIST_DIR,
    )
    logger.info("[RAG] 已向量化并存入 %s 条新闻 -> %s", len(docs), PERSIST_DIR)
    return _vector_store


def rebuild_store() -> Chroma:
    """强制从 MinIO 重新构建向量库（数据更新后调用）。"""
    docs = _load_docs_from_minio()
    return build_vector_store(docs)


def get_vector_store() -> Chroma:
    """获取新闻向量库（惰性单例）：已有磁盘库直接加载，否则从 MinIO 构建。"""
    global _vector_store
    if _vector_store is not None:
        return _vector_store

    if Path(PERSIST_DIR).exists():
        _vector_store = Chroma(
            persist_directory=PERSIST_DIR,
            collection_name=COLLECTION_NAME,
            embedding_function=_get_embeddings(),
        )
        logger.info("[RAG] 加载已有向量库: %s", PERSIST_DIR)
    else:
        _vector_store = rebuild_store()
    return _vector_store


# ============================ 问答入口 ============================

def _get_llm():
    """惰性初始化 Ollama LLM；不可用时返回 None（调用方降级为纯检索）。"""
    global _llm
    if _llm is None:
        try:
            from langchain_ollama import OllamaLLM
            _llm = OllamaLLM(
                model=LLM_MODEL,
                base_url=OLLAMA_BASE_URL,
                temperature=0.0,   # 关闭随机性，防止幻觉
                num_ctx=8192,
                timeout=120,
            )
        except Exception as exc:  # 未安装 langchain-ollama 或配置错误
            logger.warning("[RAG] LLM 初始化失败，降级为纯检索: %s", exc)
            _llm = None
    return _llm


def _format_context(hits: list[tuple[Document, float]]) -> str:
    """把检索结果拼成带序号和来源信息的上下文字符串。"""
    blocks = []
    for i, (doc, _score) in enumerate(hits, start=1):
        title, _, summary = doc.page_content.partition("\n")
        meta = doc.metadata
        blocks.append(
            f"[{i}] {title}\n"
            f"{summary.strip() if summary else ''}\n"
            f"Source: {meta.get('source', '')} | Link: {meta.get('link', '')} | "
            f"Published: {meta.get('published', '')} | Sentiment: {meta.get('sentiment', '')}"
        )
    return "\n\n".join(blocks)


def _contains_chinese(text: str) -> bool:
    """判断文本是否包含中文字符。"""
    return bool(re.search(r"[\u4e00-\u9fff]", text))


def _translate_query(query: str, llm) -> str:
    """把中文查询翻译成英文，供英文新闻向量库检索。

    all-MiniLM-L6-v2 是纯英文嵌入模型，中文查询的向量与英文文档不匹配，
    必须先翻译再检索。翻译失败时返回原查询。
    """
    prompt = (
        "Translate the following Chinese question into English for a news search. "
        "Output ONLY the English translation, without quotes or any other text.\n\n"
        f"Chinese: {query}\nEnglish:"
    )
    translated = llm.invoke(prompt).strip()
    # 清理模型可能输出的前缀/引号
    for prefix in ("English:", "Translation:", "EN:"):
        if translated.upper().startswith(prefix.upper()):
            translated = translated[len(prefix):].strip()
    translated = translated.strip('"\'')
    return translated if translated else query


def get_news_answer(query: str, top_k: int = TOP_K, use_llm: bool = True) -> dict:
    """新闻 RAG 问答入口函数，供 FastAPI 直接调用。

    流程：中文查询先翻译为英文 → 向量库相似性检索 top_k → 相关性阈值过滤
          → （可选）Ollama LLM 基于过滤后的检索结果生成答案
    :param query: 用户提问字符串（支持中文，自动翻译后检索）
    :param top_k: 返回的相关新闻条数
    :param use_llm: 是否用 LLM 生成答案；False 或 LLM 不可用时返回纯检索结果
    :return: {"query", "answer", "sources"}，sources 每项含
             title/link/source/published/sentiment/score（score 为余弦距离，越小越相关）
    """
    store = get_vector_store()

    # 1) 中文查询翻译成英文（纯英文嵌入模型的跨语言检索方案）
    search_query = query
    llm = _get_llm() if use_llm else None
    if llm is not None and _contains_chinese(query):
        try:
            search_query = _translate_query(query, llm)
            logger.info("[RAG] 中文查询翻译: '%s' -> '%s'", query, search_query)
        except Exception as exc:
            logger.warning("[RAG] 查询翻译失败，改用原始查询: %s", exc)
            search_query = query

    # 2) 相似性检索 + 相关性阈值过滤（过滤噪声，避免无关上下文送入 LLM）
    hits = store.similarity_search_with_score(search_query, k=top_k)
    relevant = [(doc, score) for doc, score in hits if score <= RELEVANCE_THRESHOLD]

    if not relevant:
        logger.info("[RAG] 未检索到相关新闻: query='%s' search='%s'", query, search_query)
        return {
            "query": query,
            "answer": "未在新闻库中找到与该问题相关的报道，请换一种问法或尝试英文关键词。",
            "sources": [],
        }

    sources = []
    for doc, score in relevant:
        title = doc.page_content.partition("\n")[0]
        meta = doc.metadata
        sources.append({
            "title": title,
            "link": meta.get("link", ""),
            "source": meta.get("source", ""),
            "published": meta.get("published", ""),
            "sentiment": meta.get("sentiment", ""),
            "score": round(float(score), 4),
        })

    # 3) LLM 基于相关新闻生成答案（question 用原始 query，答案语言跟随提问）
    answer = None
    if llm is not None:
        try:
            chain = _NEWS_PROMPT | llm | StrOutputParser()
            answer = chain.invoke({
                "context": _format_context(relevant),
                "question": query,
            })
            logger.info("[RAG] query='%s' -> answer len=%s", query, len(answer))
        except Exception as exc:
            logger.warning("[RAG] LLM 调用失败，降级为纯检索: %s", exc)
            answer = None

    if answer is None:
        if use_llm:
            answer = "（LLM 暂不可用，以下为检索到的相关新闻，可直接参考 sources 字段）"
        else:
            answer = "（纯检索模式：以下为检索到的相关新闻，可直接参考 sources 字段）"

    return {"query": query, "answer": answer, "sources": sources}


def main():
    """CLI 入口：从 MinIO 重建向量库并自测 get_news_answer。"""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    print("[RAG] 重建向量库（从 MinIO）...")
    rebuild_store()

    print("\n[RAG] 自测：get_news_answer()")
    result = get_news_answer("AI companies accused of model distillation", top_k=TOP_K)
    print(f"Q: {result['query']}\nA: {result['answer']}\n")
    for i, src in enumerate(result["sources"], start=1):
        print(f"  [{i}] {src['title']} (score={src['score']}) | {src['link']}")
    print("\n[RAG] 全部完成")


if __name__ == "__main__":
    main()
