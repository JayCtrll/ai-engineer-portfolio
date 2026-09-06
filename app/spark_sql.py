from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("SparkSQL") \
    .master("local[2]") \
    .getOrCreate()

df = spark.read.option("header", "true").csv("data/sample.csv")
df.createOrReplaceTempView("people")

result = spark.sql("""
    SELECT name, age
    FROM people
    WHERE city = 'Singapore'
    ORDER BY age DESC
""")
result.show()

spark.stop()