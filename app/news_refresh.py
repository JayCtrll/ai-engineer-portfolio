"""
app/news_refresh.py — 新闻定时检查与入库脚本（配合每小时 cron 任务）

流程:
  1. 抓取 BBC / Reuters / TechCrunch 三个 RSS 源的最新新闻
  2. 读取 MinIO news/cleaned 中已有的新闻 link 集合
  3. 对比找出新增新闻（新抓取链接 - 已有链接）
  4. 有新增: 合并去重 + 情感分析 → 写回 MinIO → 重建 news 向量库（news_chroma_db）
  5. 无新增: 打印统计后直接退出，不触发向量库重建

输出约定（供定时任务解析）:
  NEW_NEWS=0 total=55                      无新增
  NEW_NEWS=3 total=58                      新增3条，库内共58条
  FETCH_FAILED                             所有RSS源抓取失败
  ERROR: <原因>                             异常

运行（项目根目录，使用 fastapi-week1 环境）:
  /home/cjc/.pyenv/versions/fastapi-week1/bin/python app/news_refresh.py
"""
import logging
import sys
from pathlib import Path

# 保证 `python app/news_refresh.py` 或 `python -m app.news_refresh` 都能 import app.*
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from app.news_crawler import RSS_FEEDS, fetch_rss_feed
from app.news_rag import INPUT_PARQUET, create_spark, rebuild_store
from app.spark_cleaner import build_sentiment_expr

logger = logging.getLogger(__name__)

# 写回时保留的列（与 spark_cleaner 输出对齐，text_combined 会重新生成）
STORE_COLS = ["link", "published", "source", "summary", "title", "sentiment"]


def fetch_fresh_news() -> list[dict]:
    """抓取全部 RSS 源，返回 [{source,title,link,published,summary}]。"""
    all_items = []
    failed = 0
    for feed in RSS_FEEDS:
        try:
            items = fetch_rss_feed(feed)
            all_items.extend(items)
            logger.info("[Refresh] %s: %s 条", feed["source"], len(items))
        except Exception as exc:
            failed += 1
            logger.warning("[Refresh] %s 抓取失败: %s", feed["source"], exc)
    if not all_items:
        raise RuntimeError("所有 RSS 源抓取失败")
    logger.info("[Refresh] RSS 共抓取 %s 条（%s 个源失败）", len(all_items), failed)
    return all_items


def load_existing(spark: SparkSession) -> tuple:
    """读取 MinIO 已有新闻，返回 (现有DataFrame[5列], link集合)。"""
    existing = spark.read.parquet(INPUT_PARQUET)
    links = {row.link for row in existing.select("link").collect()}
    df = existing.select("link", "published", "source", "summary", "title")
    logger.info("[Refresh] MinIO 现有 %s 条新闻", len(links))
    return df, links


def enrich_and_write(spark: SparkSession, merged_df, total: int) -> None:
    """情感分析 + 过滤空标题 + 写回 MinIO。"""
    merged = merged_df.withColumn(
        "text_combined", F.lower(F.concat_ws(" ", F.col("title"), F.col("summary")))
    )
    merged = merged.withColumn("sentiment", build_sentiment_expr("text_combined"))
    merged = merged.filter(F.col("title").isNotNull() & (F.col("title") != ""))
    merged = merged.dropDuplicates(["link"])
    merged.write.mode("overwrite").parquet(INPUT_PARQUET)
    logger.info("[Refresh] 已写回 MinIO: %s 条 -> %s", total, INPUT_PARQUET)


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    # 用 news_rag.create_spark()：带 MinIO(S3A) 配置，且 rebuild_store 内部的
    # getOrCreate 会复用同一个 session，避免 s3a:// 协议缺失
    spark = create_spark()
    try:
        fresh = fetch_fresh_news()
        existing_df, existing_links = load_existing(spark)

        new_items = [it for it in fresh if it["link"] not in existing_links]
        if not new_items:
            print(f"NEW_NEWS=0 total={len(existing_links)}")
            return

        new_df = spark.createDataFrame(new_items)
        merged = existing_df.unionByName(new_df, allowMissingColumns=True)
        merged = merged.dropDuplicates(["link"])
        total = merged.count()
        enrich_and_write(spark, merged, total)
        logger.info("[Refresh] 新增 %s 条，重建向量库...", len(new_items))
        rebuild_store()
        print(f"NEW_NEWS={len(new_items)} total={total}")
    except Exception as exc:
        print(f"ERROR: {exc}")
        raise
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
