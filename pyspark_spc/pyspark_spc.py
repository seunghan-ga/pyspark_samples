# -*- coding: utf-8 -*-
from pyspark.sql import functions as f
from pyspark.sql import window
from pyspark_spc.tables import A2, A3, B3, B4, d2, D3, D4
import numpy as np


class pysparkSPC(object):
    def __init__(self):
        pass

    def c(self, dataframe, reverse=False):
        if reverse is True:
            n, x = 1, 0
        else:
            n, x = 0, 1

        assert np.mean(dataframe.rdd.map(lambda d: d[n]).collect()) \
               == dataframe.select(dataframe[n]).limit(1).collect()[0][0]

        data = dataframe.select(dataframe[x]).rdd.map(lambda d: int(d[0])).collect()
        cbar = np.mean(data)

        lcl = cbar - 3 * np.sqrt(cbar)
        ucl = cbar + 3 * np.sqrt(cbar)

        return data, cbar, lcl, ucl

    def u(self, dataframe, reverse=False):
        if reverse is True:
            n, x = 1, 0
        else:
            n, x = 0, 1

        data = dataframe.rdd.map(lambda d: d[x] / d[n]).map(lambda d: float(d)).collect()
        Ubar = dataframe.select((f.sum(dataframe[x]) / f.sum(dataframe[n]))).collect()[0][0]
        lcl = dataframe.rdd.map(lambda d: Ubar - 3 * np.sqrt(Ubar / d[n])).map(lambda d: float(d)).collect()
        ucl = dataframe.rdd.map(lambda d: Ubar + 3 * np.sqrt(Ubar / d[n])).map(lambda d: float(d)).collect()

        return data, float(Ubar), lcl, ucl

    def p(self, dataframe, reverse=False):
        if reverse is True:
            n, x = 1, 0
        else:
            n, x = 0, 1

        tmp = dataframe.withColumn('values', f.lit(dataframe[x] / dataframe[n]))
        data = tmp.select('values').rdd.map(lambda d: float(d[0])).collect()
        pbar = float(tmp.select(f.mean('values')).collect()[0][0])

        assert tmp.select(tmp[n]).filter(tmp[n] * pbar < 5).count() == 0
        assert tmp.select(tmp[n]).filter(tmp[n] * (1 - pbar) < 5).count() == 0

        if tmp.select(f.mean(tmp[n])).collect()[0][0] == tmp.select(tmp[n]).collect()[0][0]:
            size = float(tmp.select(tmp[n]).collect()[0][0])
            lcl = pbar - 3 * np.sqrt((pbar * (1 - pbar)) / size)
            ucl = pbar + 3 * np.sqrt((pbar * (1 - pbar)) / size)

            if lcl < 0:
                lcl = 0
            if ucl > 1:
                ucl = 1

        else:
            lcl = tmp.withColumn('lcl', f.lit(pbar - 3 * f.sqrt(pbar / tmp[n]))) \
                .select('lcl').rdd.map(lambda d: float(d[0])).collect()
            ucl = tmp.withColumn('ucl', f.lit(pbar + 3 * f.sqrt(pbar / tmp[n]))) \
                .select('ucl').rdd.map(lambda d: float(d[0])).collect()

        return data, pbar, lcl, ucl

    def np(self, dataframe, reverse=False):
        if reverse is True:
            n, x = 1, 0
        else:
            n, x = 0, 1

        assert dataframe.select(f.mean(dataframe[n])).collect()[0][0] \
               == dataframe.select(dataframe[n]).collect()[0][0]

        tmp = dataframe.withColumn('values', f.lit(dataframe[x] / dataframe[n]))
        data = tmp.select(tmp[x]).rdd.map(lambda d: int(d[0])).collect()
        p = float(tmp.select(f.mean(tmp['values'])).collect()[0][0])
        pbar = float(tmp.select(f.mean(tmp[x])).collect()[0][0])

        lcl = pbar - 3 * np.sqrt(pbar * (1 - p))
        ucl = pbar + 3 * np.sqrt(pbar * (1 - p))

        return data, pbar, lcl, ucl

    def mr(self, dataframe):

        tmp = dataframe.withColumn("num", f.monotonically_increasing_id()) \
            .withColumn("next", f.lead(dataframe[0], 1, 0) \
                        .over(window.Window.orderBy("num"))).drop("num")

        R = tmp.rdd.map(lambda x: abs(x[0] - x[1])).map(lambda x: float(x)).collect()[:-1]
        Rbar = np.mean(R)

        lclr = D3[2] * Rbar
        uclr = D4[2] * Rbar

        return R, Rbar, lclr, uclr

    def i(self, dataframe):
        tmp = dataframe.withColumn("num", f.monotonically_increasing_id()) \
            .withColumn("next", f.lead(dataframe[0], 1, 0) \
                        .over(window.Window.orderBy("num"))).drop("num")

        R = tmp.rdd.map(lambda x: abs(x[0] - x[1])).map(lambda x: float(x)).collect()[:-1]
        Rbar = np.mean(R)

        X = dataframe.rdd.map(lambda x: float(x[0])).collect()
        Xbar = np.mean(X)

        lcl = Xbar - 3 * (Rbar / d2[2])
        ucl = Xbar + 3 * (Rbar / d2[2])

        return X, Xbar, lcl, ucl

    def xbar_r(self, dataframe, size):
        assert size >= 2
        assert size <= 8
        assert size == len(dataframe.columns)

        R = dataframe.rdd.map(lambda x: max(x) - min(x)).map(lambda x: float(x)).collect()
        X = dataframe.rdd.map(lambda x: sum(x)).map(lambda x: float(x) / size).collect()
        Rbar = np.mean(R)
        Xbar = np.mean(X)

        lcl = Xbar - A2[size] * Rbar
        ucl = Xbar + A2[size] * Rbar

        return X, Xbar, lcl, ucl

    def r(self, dataframe, size):
        assert size >= 2
        assert size <= 8
        assert size == len(dataframe.columns)

        R = dataframe.rdd.map(lambda x: max(x) - min(x)).map(lambda x: float(x)).collect()
        Rbar = np.mean(R)

        lcl = D3[size] * Rbar
        ucl = D4[size] * Rbar

        return R, Rbar, lcl, ucl

    def xbar_s(self, dataframe, size):
        assert size >= 2
        assert size == len(dataframe.columns)

        S = dataframe.rdd.map(lambda x: np.std(x, ddof=1)).map(lambda x: float(x)).collect()
        X = dataframe.rdd.map(lambda x: np.mean(x)).map(lambda x: float(x)).collect()
        Sbar = np.mean(S)
        Xbar = np.mean(X)

        lclx = Xbar - A3[size] * Sbar
        uclx = Xbar + A3[size] * Sbar

        return X, Xbar, lclx, uclx

    def std(self, dataframe, size):
        assert size >= 2
        assert size == len(dataframe.columns)

        S = dataframe.rdd.map(lambda x: np.std(x, ddof=1)).map(lambda x: float(x)).collect()
        Sbar = np.mean(S)

        lclx = B3[size] * Sbar
        uclx = B4[size] * Sbar

        return S, Sbar, lclx, uclx

    def Cp(self, mylist, usl, lsl):
        arr = np.array(mylist)
        arr = arr.ravel()
        sigma = np.std(arr)
        cp = float(usl - lsl) / (6 * sigma)

        return cp

    def Cpk(self, mylist, usl, lsl):
        arr = np.array(mylist)
        arr = arr.ravel()
        sigma = np.std(arr)
        m = np.mean(arr)

        cpu = float(usl - m) / (3 * sigma)
        cpl = float(m - lsl) / (3 * sigma)
        cpk = np.min([cpu, cpl])

        return cpk
