"""
app/news_rag.py — 新闻 RAG 向量库构建脚本

数据链路:
  s3a://spark-bucket/news/cleaned          (news_cleaned.parquet，由 spark_cleaner.py 生成)
    → PySpark 读取 → Pandas DataFrame
    → LangChain Document 包装（title + summary 作为 page_content，
      metadata: source / link / published / sentiment）
    → HuggingFaceEmbeddings (all-MiniLM-L6-v2) 向量化
    → 存入 Chroma 向量库 ./news_chroma_db
    → 相似性搜索测试

运行（项目根目录，使用 fastapi-week1 环境）:
  /home/cjc/.pyenv/versions/fastapi-week1/bin/python app/news_rag.py
"""
from pathlib import Path

import pandas as pd
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from pyspark.sql import SparkSession

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

# hadoop-aws + aws-sdk 两个 jar 提供 s3a:// 协议支持
SPARK_JARS = (
    "/home/cjc/spark_jars/hadoop-aws-3.3.4.jar,"
    "/home/cjc/spark_jars/aws-java-sdk-bundle-1.12.262.jar"
)


def create_spark() -> SparkSession:
    """创建带 MinIO(S3A) 连接的 SparkSession。"""
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
    print(f"[RAG] 从 MinIO 读取 {count} 条新闻: {INPUT_PARQUET}")
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


def build_vector_store(docs: list[Document]):
    """向量化并存入 Chroma；重复运行时重建库，避免旧数据残留。"""
    persist_path = Path(PERSIST_DIR)
    if persist_path.exists():
        print(f"[RAG] 检测到已有向量库 {PERSIST_DIR}，重新构建前先清理")
        import shutil
        shutil.rmtree(persist_path)

    embedding = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    db = Chroma.from_documents(
        documents=docs,
        embedding=embedding,
        collection_name=COLLECTION_NAME,
        persist_directory=PERSIST_DIR,
    )
    print(f"[RAG] 已向量化并存入 {len(docs)} 条新闻 -> {PERSIST_DIR}")
    return db


def test_similarity_search(db: Chroma):
    """相似性搜索测试：从磁盘重载向量库再查询，验证可持久化检索。"""
    print(f"\n[RAG] 相似性搜索测试（top-{TOP_K}）")
    query = "AI companies accused of model distillation and competition"
    results = db.similarity_search_with_score(query, k=TOP_K)
    for i, (doc, score) in enumerate(results, start=1):
        print(f"\n===== Result {i} (score={score:.4f}) =====")
        print("Content:", doc.page_content[:200].replace("\n", " | "))
        print("Metadata:", doc.metadata)

    # 持久化验证：从磁盘重载后再次查询
    print("\n[RAG] 持久化验证：从磁盘重载向量库再查询一次")
    embedding = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    reloaded = Chroma(
        persist_directory=PERSIST_DIR,
        collection_name=COLLECTION_NAME,
        embedding_function=embedding,
    )
    docs = reloaded.similarity_search(query, k=1)
    hit = docs[0].page_content[:60].replace("\n", " | ") if docs else "无"
    print(f"[RAG] 重载成功，命中: {hit}")
    return reloaded


def main():
    spark = create_spark()
    try:
        news = load_news(spark)
        docs = to_documents(news)
        print(f"[RAG] 已包装 {len(docs)} 个 Document")

        db = build_vector_store(docs)
        test_similarity_search(db)
        print("\n[RAG] 全部完成")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
