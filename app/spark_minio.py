from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("MinIOIntegration") \
    .master("local[2]") \
    .config("spark.jars", "/home/cjc/spark_jars/hadoop-aws-3.3.4.jar,/home/cjc/spark_jars/aws-java-sdk-bundle-1.12.262.jar") \
    .config("spark.hadoop.fs.s3a.endpoint", "http://127.0.0.1:9000") \
    .config("spark.hadoop.fs.s3a.access.key", "minioadmin") \
    .config("spark.hadoop.fs.s3a.secret.key", "minioadmin") \
    .config("spark.hadoop.fs.s3a.path.style.access", "true") \
    .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
    .getOrCreate()

# 读取本地 Parquet
df = spark.read.parquet("output/cleaned_data.parquet")

# 写入 MinIO
df.write.mode("overwrite").parquet("s3a://spark-bucket/output/cleaned_data")

# 从 MinIO 读取
df_read = spark.read.parquet("s3a://spark-bucket/output/cleaned_data")
df_read.show()

spark.stop()