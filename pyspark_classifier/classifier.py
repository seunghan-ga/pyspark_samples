from pyspark.ml.classification import DecisionTreeClassifier, RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator


def decision_tree(data, featuresCol='features', labelCol='label', maxDepth=3):
    dt = DecisionTreeClassifier(featuresCol=featuresCol, labelCol=labelCol, maxDepth=maxDepth)
    dtModel = dt.fit(data)

    return dtModel


def random_forest(data, featuresCol='features', labelCol='label'):
    rf = RandomForestClassifier(featuresCol=featuresCol, labelCol=labelCol)
    rfModel = rf.fit(data)

    return rfModel


def gradient_boosted_Tree(data, maxIter=10):
    gbt = GBTClassifier(maxIter=maxIter)
    gbtModel = gbt.fit(data)

    return gbtModel


def binary_classification_evaluator(predictions):
    evaluator = BinaryClassificationEvaluator()
    roc = str(evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"}))
    return roc


def multi_classification_evaluator(predictions, labelCol="label", predictionCol="prediction", metricName="accuracy"):
    evaluator = MulticlassClassificationEvaluator(labelCol=labelCol, predictionCol=predictionCol, metricName=metricName)
    accuracy = evaluator.evaluate(predictions)

    return accuracy


if __name__ == "__main__":
    pass