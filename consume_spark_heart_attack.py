from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.sql.types import StructType, StructField, StringType
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.sql.functions import from_json, col

# Load the saved logistic regression model
model_save_path = "/home/ayemon/KafkaProjects/kafkaspark07_90/model"  # Replace with the path where you saved the model
loaded_lrModel = PipelineModel.load(model_save_path)

# Kafka configuration
kafka_broker = "localhost:9092"
kafka_topic = "my_json_topic"  # The same topic you used in the producer code

# Create a SparkSession
spark = SparkSession.builder \
    .appName("Kafka Spark Model Inference") \
    .getOrCreate()

# Set the log level to ERROR to suppress unnecessary logs
spark.sparkContext.setLogLevel("ERROR")

# Define the schema for the incoming data from Kafka
schema = StructType([
    StructField("age", StringType(), True),
    StructField("sex", StringType(), True),
    StructField("cp", StringType(), True),
    StructField('trtbps', StringType(), True),
    StructField("chol", StringType(), True),
    StructField("fbs", StringType(), True),
    StructField("restecg", StringType(), True),
    StructField("thalachh", StringType(), True),
    StructField("exng", StringType(), True),
    StructField("oldpeak", StringType(), True),
    StructField("slp", StringType(), True),
    StructField("caa", StringType(), True),
    StructField("thall", StringType(), True),
    StructField("output", StringType(), True),
])

# Read data from Kafka topic
df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", kafka_broker) \
    .option("subscribe", kafka_topic) \
    .load()

# Parse the JSON data from Kafka into individual columns
df = df.selectExpr("CAST(value AS STRING)")

# Apply the schema to the data
df = df.select(from_json("value", schema).alias("data")).select("data.*")

# Convert the necessary columns to appropriate data types

df = df.withColumn("age",col("age").cast('long'))\
    .withColumn("sex",col("sex").cast('long'))\
    .withColumn("cp",col("cp").cast('long'))\
    .withColumn("trtbps",col("trtbps").cast('long'))\
    .withColumn("chol",col("chol").cast('long'))\
    .withColumn("fbs", col("fbs").cast('long'))\
    .withColumn("restecg", col("restecg").cast('long'))\
    .withColumn("thalachh", col("thalachh").cast('long'))\
    .withColumn("exng", col("exng").cast('long'))\
    .withColumn("oldpeak", col("oldpeak").cast('double'))\
    .withColumn("slp", col("slp").cast('long'))\
    .withColumn("caa", col("caa").cast('long'))\
    .withColumn("thall", col("thall").cast('long'))\
    .withColumn("output", col("output").cast('long'))
df = df.withColumnRenamed("output","label")
#df.columns.remove('output')
# Make predictions using the loaded logistic regression model
predictions = loaded_lrModel.transform(df)

# Select the original columns and the prediction column
predictions = predictions.select('label','prediction')

# Start the streaming query to continuously read from Kafka and make predictions
query = predictions.writeStream \
    .outputMode("append") \
    .format("console") \
    .start()

query.awaitTermination()
