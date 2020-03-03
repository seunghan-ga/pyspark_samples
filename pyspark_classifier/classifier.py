from pyspark.ml.classification import DecisionTreeClassifier, RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator


def decision_tree(data):
    dt = DecisionTreeClassifier(featuresCol='features', labelCol='label', maxDepth=3)
    dtModel = dt.fit(data)

    return dtModel


def random_forest(data):
    rf = RandomForestClassifier(featuresCol='features', labelCol='label')
    rfModel = rf.fit(data)

    return rfModel


def gradient_boosted_Tree(data):
    gbt = GBTClassifier(maxIter=10)
    gbtModel = gbt.fit(data)

    return gbtModel


def binary_classification_evaluator(predictions):
    evaluator = BinaryClassificationEvaluator()
    roc = str(evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"}))
    return roc


def multi_classification_evaluator(predictions):
    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)

    return accuracy


if __name__ == "__main__":
    pass