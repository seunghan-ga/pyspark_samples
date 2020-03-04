from pyspark.ml.regression import DecisionTreeRegressor, RandomForestRegressor, GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator


def decision_tree(data, featuresCol="indexedFeatures"):
    dt = DecisionTreeRegressor(featuresCol=featuresCol)
    dtModel = dt.fit(data)

    return dtModel


def random_forest(data, featuresCol="indexedFeatures"):
    rf = RandomForestRegressor(featuresCol=featuresCol)
    rfModel = rf.fit(data)

    return rfModel


def gradient_boosted_Tree(data, featuresCol="indexedFeatures", maxIter=10):
    gbt = GBTRegressor(featuresCol=featuresCol, maxIter=maxIter)
    gbtModel = gbt.fit(data)

    return gbtModel


def regression_evaluator(predictions, labelCol="label", predictionCol="prediction", metricName="rmse"):
    evaluator = RegressionEvaluator(labelCol=labelCol, predictionCol=predictionCol, metricName=metricName)
    rmse = evaluator.evaluate(predictions)

    return rmse