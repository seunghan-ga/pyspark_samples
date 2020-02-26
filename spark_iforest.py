from sklearn.ensemble import IsolationForest

from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.types import Row
from pyspark.ml.feature import MinMaxScaler
from pyspark.ml.linalg import Vectors

from pyspark_iforest.ml.iforest import *  # https://github.com/titicaca/spark-iforest/tree/master/python

import pandas as pd

if __name__ == "__main__":
    # Data Load
    SparkContext.setSystemProperty("hive.metastore.uris", "thrift://192.168.1.103:9083")
    spark = (SparkSession
             .builder
             .appName("test app")
             .enableHiveSupport()
             .getOrCreate())

    start_date = "2004-02-10 03:12:39"
    end_date = '2004-02-20 03:12:39'

    all_data = spark.sql("select * from demo.bearing where idx_date >= '%s' and idx_date < '%s'" \
                         % (start_date, end_date))
    columns = all_data.columns

    # create scaled data
    tmp_data = all_data.rdd.map(lambda x: (x[0], Vectors.dense(x[1:]))).collect()
    scale_df = spark.createDataFrame(tmp_data, ['idx_date', '_features'])

    scaler = MinMaxScaler(inputCol="_features", outputCol="features")
    scalerModel = scaler.fit(scale_df)
    scaledData = scalerModel.transform(scale_df)

    train_data = scaledData.select("idx_date", "features").filter("idx_date <= '2004-02-15 12:52:39'")
    test_data = scaledData.select("idx_date", "features").filter("idx_date >= '%s'" % start_date) \
        .filter("idx_date <= '%s'" % end_date)

    iforest = IForest(contamination=0.1, maxFeatures=1.0, maxSamples=256, bootstrap=True)
    model = iforest.fit(train_data)
    model.hasSummary
    summary = model.summary
    summary.numAnomalies
    transformed = model.transform(test_data)


    def f(x):
        row = [x[0]]
        for i in x[1].values:
            row.append(float(i))
        row.append(x[2])
        row.append(x[3])
        return tuple(row)


    tmp_data = transformed.rdd.map(lambda x: f(x)).collect()
scaled_df = spark.createDataFrame(tmp_data, columns + ['anomalyScore', 'prediction']).show()