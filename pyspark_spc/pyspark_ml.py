# -*- coding: utf-8 -*-
from pyspark.mllib.linalg.distributed import RowMatrix
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import PCA
from pyspark.ml.stat import Correlation
from pyspark.sql import Row
import numpy as np


class Statistics(object):
    def __init__(self):
        pass

    def mean(self, df):
        res = df.rdd.map(lambda x: Vectors.dense(x)).mean()

        return res

    def std(self, df):
        res = df.rdd.map(lambda x: Vectors.dense(x)).stdev()

        return res

    def cor(self, df):
        features = df.rdd.map(lambda x: Row(features=Vectors.dense(x))).toDF()
        res = Correlation.corr(features, "features").head()

        return res[0].toArray()

    def cov(self, df):
        res = RowMatrix(df.rdd.map(lambda x: list(x))).computeCovariance()

        return res.toArray()

    def pca(self, df, k=1):
        cov = RowMatrix(df.rdd.map(lambda x: list(x))).computeCovariance().toArray()
        col = cov.shape[1]
        eigVals, eigVecs = np.linalg.eigh(cov)
        inds = np.argsort(eigVals)
        eigVecs = eigVecs.T[inds[-1:-(col + 1):-1]]
        eigVals = eigVals[inds[-1:-(col + 1):-1]]
        components = RowMatrix(df.rdd.map(lambda x: list(x))).computePrincipalComponents(k)

        train_data = df.rdd.map(lambda x: Row(features=Vectors.dense(x))).toDF()

        pca = PCA(k=k, inputCol="features", outputCol="pcaFeatures")
        model = pca.fit(train_data)
        score = model.transform(train_data)

        res = {
            "components": components.toArray(),
            "score": np.array(score.select("pcaFeatures").rdd.map(lambda x: list(x[0])).collect()),
            "eigVectors": eigVecs,
            "eigValues": eigVals
        }

        return res

    def t2(self, df):
        df_colmean = df.rdd.map(lambda x: Row(features=Vectors.dense(x))).toDF()
        colmean = df_colmean.rdd.map(lambda x: x[0]).mean()
        matcov = self.cov(df)
        matinv = np.linalg.inv(matcov)

        t2 = df.rdd.map(lambda x: list(x - colmean))\
            .map(lambda x: matinv.dot(np.array(x).T).dot(np.array(x)))\
            .map(lambda x: Row(t2=float(x))).toDF()

        return t2


if __name__ == "__main__":
    pass
