"""
app/etl_minio.py — 基于 PySpark + MinIO 的 ETL 示例

数据链路：
  s3a://spark-bucket/input/raw_sales.csv      (原始 CSV，含脏数据)
    → 清洗：去空格、统一大小写、标准化日期、去空、去重、过滤业务无效行
    → 聚合：按 品类 × 地区 统计订单数、营收、客单价、销量
    → 写出：s3a://spark-bucket/output/etl/ 下的三个结果

运行（项目根目录）：
  python app/etl_minio.py
"""
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

INPUT_CSV = "s3a://spark-bucket/input/raw_sales.csv"
OUTPUT_CLEANED = "s3a://spark-bucket/output/etl/cleaned_data"
OUTPUT_AGG = "s3a://spark-bucket/output/etl/aggregated"
OUTPUT_AGG_CSV = "s3a://spark-bucket/output/etl/aggregated_report.csv"

# 业务规则：只统计有效状态的订单
VALID_STATUS = ("completed", "pending")


def create_spark() -> SparkSession:
    """创建带 MinIO(S3A) 连接的 SparkSession。"""
    return (
        SparkSession.builder
        .appName("ETL-MinIO")
        .master("local[2]")
        # hadoop-aws + aws-sdk 两个 jar 提供 s3a:// 协议支持
        .config(
            "spark.jars",
            "/home/cjc/spark_jars/hadoop-aws-3.3.4.jar,"
            "/home/cjc/spark_jars/aws-java-sdk-bundle-1.12.262.jar",
        )
        .config("spark.hadoop.fs.s3a.endpoint", "http://127.0.0.1:9000")
        .config("spark.hadoop.fs.s3a.access.key", "minioadmin")
        .config("spark.hadoop.fs.s3a.secret.key", "minioadmin")
        # MinIO 要求 path-style 访问，不能用 virtual-host 方式
        .config("spark.hadoop.fs.s3a.path.style.access", "true")
        .getOrCreate()
    )


def read_raw(spark: SparkSession):
    """从 MinIO 读取原始 CSV（首行为表头，自动推断类型）。"""
    return (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .csv(INPUT_CSV)
    )


def clean(df):
    """清洗：逐条返回经过各规则处理后的 DataFrame。"""
    # 1) 字符串列去掉首尾空格（例如 "Singapore " → "Singapore"）
    for col in ["customer", "region", "category", "status"]:
        df = df.withColumn(col, F.trim(F.col(col)))

    # 2) 品类统一小写，避免 "Electronics" 和 "electronics" 被当成两类
    df = df.withColumn("category", F.lower(F.col("category")))

    # 3) 日期标准化为 DateType，非法日期（如 2026-02-30）会变成 NULL
    df = df.withColumn("order_date", F.to_date(F.col("order_date"), "yyyy-MM-dd"))

    # 4) 关键字段为空的行直接丢弃（金额/数量/日期缺失）
    df = df.dropna(subset=["order_id", "order_date", "amount", "quantity"])

    # 5) 按 order_id 去重，保留第一条
    df = df.dropDuplicates(["order_id"])

    # 6) 业务规则过滤：状态有效，且金额 > 0（排除退款/负数等无效行）
    #    注意：PySpark 3.5 的 isin 需用 * 解包元组
    df = df.filter(F.col("status").isin(*VALID_STATUS) & (F.col("amount") > 0))

    # 7) 衍生列：营收 = 单价 × 数量
    df = df.withColumn("revenue", F.round(F.col("amount") * F.col("quantity"), 2))
    return df


def aggregate(df):
    """按 品类 × 地区 聚合。"""
    return (
        df.groupBy("category", "region")
        .agg(
            F.count("*").alias("order_count"),
            F.round(F.sum("revenue"), 2).alias("total_revenue"),
            F.round(F.avg("amount"), 2).alias("avg_amount"),
            F.sum("quantity").alias("total_quantity"),
        )
        .orderBy(F.desc("total_revenue"))
    )


def main():
    spark = create_spark()
    try:
        raw = read_raw(spark)
        raw_count = raw.count()

        cleaned = clean(raw)
        cleaned_count = cleaned.count()
        print(f"[ETL] 原始行数: {raw_count} -> 清洗后行数: {cleaned_count}")

        # 结果 1：清洗后的明细数据（Parquet）
        cleaned.write.mode("overwrite").parquet(OUTPUT_CLEANED)
        print(f"[ETL] 已写出清洗明细 -> {OUTPUT_CLEANED}")

        # 聚合
        result = aggregate(cleaned)
        print("[ETL] 聚合结果（品类 × 地区）:")
        result.show(truncate=False)

        # 结果 2：聚合结果（Parquet）
        result.write.mode("overwrite").parquet(OUTPUT_AGG)
        print(f"[ETL] 已写出聚合结果 -> {OUTPUT_AGG}")

        # 结果 3：聚合结果导出为单文件 CSV，方便人工查看
        result.coalesce(1).write.mode("overwrite").option("header", True).csv(OUTPUT_AGG_CSV)
        print(f"[ETL] 已写出聚合 CSV -> {OUTPUT_AGG_CSV}")

        print("[ETL] 全部完成")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
