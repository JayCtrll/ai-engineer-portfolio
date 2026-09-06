from pyspark.sql import SparkSession
from pyspark.sql.window import Window
from pyspark.sql.functions import row_number, col

spark = SparkSession.builder \
    .appName("WindowFunc") \
    .master("local[2]") \
    .getOrCreate()

df = spark.read.option("header", "true").csv("data/sample.csv")
window_spec = Window.partitionBy("city").orderBy(col("age").desc())
ranked = df.withColumn("rank", row_number().over(window_spec))
top2 = ranked.filter(col("rank") <= 2)
top2.show()

spark.stop()