import pyspark.ml.feature as normalizer


def norm_scaler(X, inputCol="features", outputCol="resFeatures", p=1.0):
    scaler = normalizer.Normalizer(inputCol=inputCol, outputCol=outputCol, p=p)
    l1NormData = scaler.transform(X)
    # lInfNormData = scaler.transform(X, {normalizer.p: float("inf")})

    return l1NormData


def std_scaler(X, inputCol="features", outputCol="resFeatures"):
    scaler = normalizer.StandardScaler(inputCol=inputCol, outputCol=outputCol,
                                       withStd=True, withMean=False)
    scalerModel = scaler.fit(X)
    scaledData = scalerModel.transform(X)

    return scaledData


def min_max_scaler(X, inputCol="features", outputCol="resFeatures"):
    scaler = normalizer.MinMaxScaler(inputCol=inputCol, outputCol=outputCol)
    scalerModel = scaler.fit(X)
    scaledData = scalerModel.transform(X)

    return scaledData


def max_abs_scaler(X, inputCol="features", outputCol="resFeatures"):
    scaler = normalizer.MaxAbsScaler(inputCol=inputCol, outputCol=outputCol)
    scalerModel = scaler.fit(X)
    scaledData = scalerModel.transform(X)

    return scaledData
