from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.sql.types import StructType,StructField,LongType, StringType,DoubleType,TimestampType
from pyspark.ml.feature import OneHotEncoder, MinMaxScaler, StringIndexer, VectorAssembler, OneHotEncoder
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


# Create a SparkSession
spark = SparkSession.builder \
    .appName("HA_Model") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

schema = StructType( \
                     [StructField("age", LongType(),True), \
                      StructField("sex", LongType(), True), \
                      StructField("cp", LongType(), True), \
                      StructField('trtbps', LongType(), True), \
                      StructField("chol", LongType(), True), \
                      StructField("fbs", LongType(), True), \
                      StructField("restecg", LongType(), True), \
                      StructField("thalachh", LongType(), True),\
                      StructField("exng", LongType(), True), \
                      StructField("oldpeak", DoubleType(), True), \
                      StructField("slp", LongType(),True), \
                      StructField("caa", LongType(), True), \
                      StructField("thall", LongType(), True), \
                      StructField("output", LongType(), True), \
                        ])

# Load the CSV files into DataFrames
df = spark.read.format('csv').option("header", "true").schema(schema).load("/home/ayemon/KafkaProjects/kafkaspark07_90/heart_80.csv")
df = df.withColumnRenamed("output","label")
df.printSchema()
#df.show()

trainDF, testDF = df.randomSplit([0.75, 0.25], seed=42)

feature_cols = df.columns
feature_cols.remove('label')
lr = LogisticRegression(maxIter=10, regParam= 0.01)


inputs = ['age','trtbps','chol','thalachh','oldpeak']

assembler1 = VectorAssembler(inputCols=inputs, outputCol="features_scaled1")
scaler = MinMaxScaler(inputCol="features_scaled1", outputCol="features_scaled")
assembler2 = VectorAssembler(inputCols=['features_scaled'], outputCol="features")

# Create stages list
myStages = [assembler1, scaler, assembler2,lr]

# Set up the pipeline
pipeline = Pipeline(stages= myStages)

# We fit the model using the training data.
pModel = pipeline.fit(trainDF)# We transform the data.
trainingPred = pModel.transform(testDF)# # We select the actual label, probability and predictions
trainingPred.select('label','probability','prediction').show()

# Evaluate the model's performance
evaluator = MulticlassClassificationEvaluator(labelCol='label', predictionCol='prediction', metricName='accuracy')
accuracy = evaluator.evaluate(trainingPred)
print("Accuracy: ", accuracy)

trainingPred.crosstab('label','prediction').show()

# Save the trained model
model_save_path = "/home/ayemon/KafkaProjects/kafkaspark07_90/model"
pModel.write().overwrite().save(model_save_path)

