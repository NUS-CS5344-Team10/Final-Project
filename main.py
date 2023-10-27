import pyspark
from pyspark.sql.functions import col, count, when, isnull, udf
from pyspark.sql import SparkSession
from pyspark.sql.types import StringType, FloatType, DoubleType
from pyspark.ml.feature import Tokenizer, StopWordsRemover, Word2Vec
from pyspark.ml import Pipeline
from pyspark.ml.classification import MultilayerPerceptronClassifier

spark = SparkSession.builder.config('spark.driver.memory', '4g').appName('TwitterSentimentAnalysis').getOrCreate()
# import data
metadf = spark.read.csv("training.1600000.processed.noemoticon.csv", inferSchema=True)
# data preprocess
metadf = (((((metadf.withColumnRenamed("_c0", "Polarity")
              .withColumnRenamed("_c1", "TweetID"))
             .withColumnRenamed("_c2", "Date"))
            .withColumnRenamed("_c3", "QueryFlag"))
           .withColumnRenamed("_c4", "User"))
          .withColumnRenamed("_c5", "TweetText"))
missing_count = metadf.select([count(when(isnull(c), c)).alias(c) for c in metadf.columns]).collect()
print(missing_count)
duplicates = metadf.groupBy(metadf.columns).count().filter("count > 1")
duplicates.show()

# text preprocess
tokenizer = Tokenizer(inputCol="TweetText", outputCol="words")
metadf = tokenizer.transform(metadf)

remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
metadf = remover.transform(metadf)

# word2vec process
word2Vec = Word2Vec(vectorSize=100, minCount=5, inputCol="filtered_words", outputCol="features")


# turn text polarity into double type

def polarity_map(value):
    if value == 4:
        return 2.0  # Positive
    elif value == 2:
        return 1.0  # Neutral
    else:
        return 0.0  # Negative


polarity_udf = udf(polarity_map, DoubleType())
metadf = metadf.withColumn("label", polarity_udf(metadf["Polarity"]))

# multi layer perceptron classifier
layers = [100, 64, 32, 3]
mlp = MultilayerPerceptronClassifier(layers=layers, blockSize=128, seed=1234)
pipeline = Pipeline(stages=[word2Vec, mlp])
seed = 24
(train, test) = metadf.randomSplit([0.7, 0.3], seed)
model = pipeline.fit(train)

predictions = model.transform(test)
predictions.select("TweetText", "probability", "prediction").show()

