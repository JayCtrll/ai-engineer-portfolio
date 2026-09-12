from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, lower, concat_ws, expr

POSITIVE_WORDS = ["good", "great", "success", "win", "improve", "breakthrough", "growth", "benefit"]
NEGATIVE_WORDS = ["bad", "fail", "crisis", "war", "disaster", "decline", "loss", "risk"]

def build_sentiment_expr(text_col):
    # 构建SQL LIKE条件字符串
    positive_expr = " or ".join([f"lower({text_col}) like '%{w}%'" for w in POSITIVE_WORDS])
    negative_expr = " or ".join([f"lower({text_col}) like '%{w}%'" for w in NEGATIVE_WORDS])
    # 用expr()把字符串转为Column表达式.
    return when(expr(positive_expr), "positive")\
        .when(expr(negative_expr), "negative")\
        .otherwise("neutral")

def main():
    spark = SparkSession.builder \
        .appName("NewsCleaner") \
        .master("local[2]") \
        .getOrCreate()

    # 读取所有 JSON 文件
    raw_path = "data/raw_news/*.json"
    df = spark.read.option("multiline", "true").json(raw_path)
    print(f"Raw count: {df.count()}")

    # 合并 title 和 summary 用于情感分析
    df = df.withColumn("text_combined", lower(concat_ws(" ", col("title"), col("summary"))))

    # 去重
    df = df.dropDuplicates(["link"])

    # 过滤空值
    df = df.filter(col("title").isNotNull() & (col("title") != ""))

    # 添加情感列
    df = df.withColumn("sentiment", build_sentiment_expr("text_combined"))

    # 输出
    df.write.mode("overwrite").parquet("output/news_cleaned.parquet")

    # 统计
    print(f"Cleaned count: {df.count()}")
    df.groupBy("sentiment").count().show()

    spark.stop()

if __name__ == "__main__":
    main()