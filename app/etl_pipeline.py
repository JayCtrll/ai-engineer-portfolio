from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when

spark = SparkSession.builder \
    .appName("ETLPipeline") \
    .master("local[2]") \
    .getOrCreate()

df = spark.read.option("header", "true").csv("data/dirty_sample.csv")

# 清洗
cleaned = df.filter(col("age").isNotNull() & (col("age") >= 0)) \
    .withColumn("city", when(col("city").isNull(), "Unknown").otherwise(col("city")))

cleaned.show()
print(f"Cleaned rows: {cleaned.count()}")

cleaned.write.mode("overwrite").parquet("output/cleaned_data.parquet")

spark.stop()

