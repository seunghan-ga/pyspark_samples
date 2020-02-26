import preprocessing.normalization as norm

from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors


if __name__ == "__main__":
    spark = (SparkSession
             .builder
             .appName("pyspark normalization test")
             .getOrCreate())

    dataFrame = spark.createDataFrame([
        (0, Vectors.dense([1.0, 0.5, -1.0]),),
        (1, Vectors.dense([2.0, 1.0, 1.0]),),
        (2, Vectors.dense([4.0, 10.0, 2.0]),)
    ], ["id", "features"])

    norm_result = norm.norm_scaler(dataFrame)
    norm_result.show()

    std_resuilt = norm.std_scaler(dataFrame)
    std_resuilt.show()

    min_max_resuilt = norm.min_max_scaler(dataFrame)
    min_max_resuilt.show()

    max_abs_resuilt = norm.max_abs_scaler(dataFrame)
    max_abs_resuilt.show()