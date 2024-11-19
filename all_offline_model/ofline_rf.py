from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.sql.types import StructType, StructField, LongType, DoubleType
from pyspark.ml.feature import OneHotEncoder, MinMaxScaler, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics, BinaryClassificationMetrics
from pyspark.sql.functions import col

# Create a SparkSession
spark = SparkSession.builder \
    .appName("ECG Random Forest Model") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

# Define schema for input data
schema = StructType([
    StructField("age", LongType(), True),
    StructField("sex", LongType(), True),
    StructField("cp", LongType(), True),
    StructField('trtbps', LongType(), True),
    StructField("chol", LongType(), True),
    StructField("fbs", LongType(), True),
    StructField("restecg", LongType(), True),
    StructField("thalachh", LongType(), True),
    StructField("exng", LongType(), True),
    StructField("oldpeak", DoubleType(), True),
    StructField("slp", LongType(), True),
    StructField("caa", LongType(), True),
    StructField("thall", LongType(), True),
    StructField("output", LongType(), True),
])

# Load the CSV file into a DataFrame
df = spark.read.format('csv').option("header", "true").schema(schema).load("/home/ayemon/KafkaProjects/kafkaspark07_90/heart_80.csv")
df = df.withColumnRenamed("output", "label")
df.printSchema()

# Split data into training and testing datasets
trainDF, testDF = df.randomSplit([0.75, 0.25], seed=42)

# One-hot encoding
ohe = OneHotEncoder(inputCols=['sex', 'cp', 'fbs', 'restecg', 'slp', 'exng', 'caa', 'thall'],
                    outputCols=['sex_ohe', 'cp_ohe', 'fbs_ohe', 'restecg_ohe', 'slp_ohe', 'exng_ohe', 'caa_ohe', 'thall_ohe'])

# Define feature scaling
inputs = ['age', 'trtbps', 'chol', 'thalachh', 'oldpeak']
assembler1 = VectorAssembler(inputCols=inputs, outputCol="features_scaled1")
scaler = MinMaxScaler(inputCol="features_scaled1", outputCol="features_scaled")

# Combine all features into a single vector
assembler2 = VectorAssembler(inputCols=['sex_ohe', 'cp_ohe', 'fbs_ohe', 'restecg_ohe', 'slp_ohe', 'exng_ohe', 'caa_ohe', 'thall_ohe', 'features_scaled'],
                             outputCol="features")

# Define the Random Forest model
rf = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=20, maxDepth=8, seed=42)

# Create a pipeline with all stages
pipeline = Pipeline(stages=[assembler1, scaler, ohe, assembler2, rf])


# Train the model
pModel = pipeline.fit(trainDF)

# Predict on the training data
trainingPred = pModel.transform(trainDF)
trainingPred.select('label', 'probability', 'prediction').show()

# Evaluate on the training data
evaluator = MulticlassClassificationEvaluator(labelCol='label', predictionCol='prediction', metricName='accuracy')
training_accuracy = evaluator.evaluate(trainingPred)
print(f"Training Accuracy: {training_accuracy}")

# Predict on the test data
testPred = pModel.transform(testDF)

# Evaluate the model's performance on the test data
test_accuracy = evaluator.evaluate(testPred)
print(f"Test Accuracy: {test_accuracy}")

# Calculate additional metrics
predictions_and_labels = testPred.select(col("prediction"), col("label")).rdd.map(lambda x: (float(x[0]), float(x[1])))
metrics = MulticlassMetrics(predictions_and_labels)

# Confusion matrix
confusion_matrix = metrics.confusionMatrix().toArray()
true_positive = confusion_matrix[1, 1]
false_positive = confusion_matrix[0, 1]
true_negative = confusion_matrix[0, 0]
false_negative = confusion_matrix[1, 0]

# Sensitivity (Recall)
sensitivity = true_positive / (true_positive + false_negative)

# Specificity
specificity = true_negative / (true_negative + false_positive)

# Precision
precision = true_positive / (true_positive + false_positive)

# F1 Score
f1_score = metrics.fMeasure(1.0)

# ROC AUC
binary_metrics = BinaryClassificationMetrics(testPred.select(col("probability").alias("score"), col("label")).rdd.map(lambda x: (float(x[0][1]), float(x[1]))))
roc_auc = binary_metrics.areaUnderROC

# Print metrics
print("Confusion Matrix:")
print(confusion_matrix)

print(f"Test Accuracy: {test_accuracy}")
print(f"Sensitivity (Recall): {sensitivity}")
print(f"Specificity: {specificity}")
print(f"Precision: {precision}")
print(f"F1 Score: {f1_score}")
print(f"ROC AUC: {roc_auc}")

# Save the trained model
model_save_path = "/home/ayemon/KafkaProjects/kafkaspark07_90/rf_model"
pModel.write().overwrite().save(model_save_path)
