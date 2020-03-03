from pyspark_classifier import classifier
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.sql import SparkSession
import pandas as pd


if __name__ == "__main__":
    # create spark session
    spark = SparkSession \
        .builder \
        .appName("tree test") \
        .getOrCreate()

    source_data = pd.read_csv('./samples/Averaged_BearingTest_Dataset.csv')

    # labelling
    X_train = source_data[source_data.Date <= '2004-02-16 03:02:39']
    X_test = source_data[source_data.Date >= '2004-02-16 03:12:39']
    X_train['label'] = 0
    X_test['label'] = 1
    X = pd.concat([X_train, X_test])

    # preprocessing
    df = spark.createDataFrame(X, list(X.columns))
    assembler = VectorAssembler(inputCols=['Bearing 1', 'Bearing 2', 'Bearing 3', 'Bearing 4'], outputCol="features")
    stages = [assembler]

    pipeline = Pipeline(stages=stages)
    pipelineModel = pipeline.fit(df)
    df = pipelineModel.transform(df)
    selectedCols = df.columns
    df = df.select(selectedCols)
    df.printSchema()

    # create dataset
    train, test = df.randomSplit([0.7, 0.3], seed=2018)
    print("Training Dataset Count: " + str(train.count()))
    print("Test Dataset Count: " + str(test.count()))

    # decision tree classifier
    dtModel = classifier.decision_tree(train)
    predictions = dtModel.transform(test)
    predictions.show(10)
    accuracy = classifier.multi_classification_evaluator(predictions)
    roc = classifier.binary_classification_evaluator(predictions)
    print("Test Error = %g " % (1.0 - accuracy))
    print("Test Area Under ROC: " + roc)

    # random forest classifier
    rfModel = classifier.random_forest(train)
    predictions = rfModel.transform(test)
    predictions.show(10)
    accuracy = classifier.multi_classification_evaluator(predictions)
    roc = classifier.binary_classification_evaluator(predictions)
    print("Test Error = %g " % (1.0 - accuracy))
    print("Test Area Under ROC: " + roc)

    # gradient-boosted tree classifier
    gbtModel = classifier.gradient_boosted_Tree(train)
    predictions = gbtModel.transform(test)
    predictions.show(10)
    accuracy = classifier.multi_classification_evaluator(predictions)
    roc = classifier.binary_classification_evaluator(predictions)
    print("Test Error = %g " % (1.0 - accuracy))
    print("Test Area Under ROC: " + roc)
