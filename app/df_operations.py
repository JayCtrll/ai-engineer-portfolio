from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count

spark = SparkSession.builder \
    .appName("DataFrameOps") \
    .master("local[2]") \
    .getOrCreate()

df = spark.read.option("header", "true").csv("data/sample.csv")
df.printSchema()
df.show()

# 选择与过滤
filtered = df.select("name", "age").filter(col("age") > 28)
filtered.show()

# 分组统计
result = df.groupBy("city").agg(count("*").alias("count")).orderBy("count", ascending=False)
result.show()

spark.stop()