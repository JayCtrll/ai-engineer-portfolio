from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("FirstSparkApp") \
    .master("local[2]") \
    .getOrCreate()

df = spark.range(1, 6)
df.show()

spark.stop()