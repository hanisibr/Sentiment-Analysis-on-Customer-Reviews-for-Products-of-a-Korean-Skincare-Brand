import pandas as pd
from pyspark.sql import SparkSession
import pyspark.sql.types as tp


# Create a Spark Session

spark = SparkSession.builder.master("local").config('spark.executor.cores', 4).appName("LogReg").getOrCreate()


# define the schema
my_schema = tp.StructType([
  tp.StructField(name= 'Comments', dataType= tp.StringType(),  nullable= True),
  tp.StructField(name= 'Product_Name', dataType= tp.StringType(),  nullable= True),
  tp.StructField(name= 'Rating_Score', dataType= tp.FloatType(),   nullable= True)
])


# Read dataset

df = spark.read.csv("product_reviews_clean.csv", schema=my_schema, header=True)

  
# Creating a tokenizer object for spliting the Comments column and creating a new output column

from pyspark.ml.feature import RegexTokenizer

tokens = RegexTokenizer().setGaps(False).setPattern("\\p{L}+").setInputCol("Comments").setOutputCol("Words")

from pyspark.ml import Pipeline


# Apply TF-IDF

from pyspark.ml.feature import HashingTF

tf = HashingTF().setInputCol("Words").setOutputCol("TF")

from pyspark.ml.feature import IDF

idf = IDF().setInputCol('TF').setOutputCol('Features')


# Split training and test set

train_df, test_df = df.randomSplit([0.7, 0.3], seed=123)


# Logistic Regression Model

from pyspark.ml.classification import LogisticRegression

LR = LogisticRegression(maxIter=100, featuresCol="Features", labelCol="Rating_Score")

Pipeline_LR = Pipeline(stages=[tokens, tf, idf, LR])

model_LR = Pipeline_LR.fit(train_df)

test_LR = model_LR.transform(test_df)

test_LR.select('Product_Name','Comments','Rating_Score', 'Prediction').show(10)


# Logistic Regression model ROC

from pyspark.ml.evaluation import BinaryClassificationEvaluator

ROC_eval = BinaryClassificationEvaluator(rawPredictionCol="prediction", labelCol="Rating_Score")

ROC_LR = ROC_eval.evaluate(test_LR)


# Logistic Regression model accuracy

from pyspark.ml.evaluation import MulticlassClassificationEvaluator

acc_eval = MulticlassClassificationEvaluator(metricName='accuracy', labelCol="Rating_Score")

acc_LR = acc_eval.evaluate(test_LR)

print("Accuracy of the model: {}".format(acc_LR))

print("ROC of the model: {}".format(ROC_LR))
