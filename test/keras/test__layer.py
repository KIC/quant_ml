import os
from unittest import TestCase

import numpy as np
import pandas as pd
from keras import backend as K, Sequential
from keras.callbacks import EarlyStopping
from keras.optimizers import SGD

from pandas_ml_utils.utils.classes import ReScaler
from talib_ml.keras.layers import LinearRegressionLayer, LPPLLayer
from sklearn.preprocessing import MinMaxScaler

import pandas_ml_utils as pmu
df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "SPY.csv"), index_col='Date').dropna()
os.environ["CUDA_VISIBLE_DEVICES"] = ""
print(pmu.__version__)


class TestKerasLayer(TestCase):

    def __test__LinearRegressionLayer(self):
        """given"""
        model = Sequential()
        model.add(LinearRegressionLayer())
        model.compile(loss='mse', optimizer='nadam')

        x = df["Close"].values.reshape(1, -1)

        """when"""
        model.fit(x, x, epochs=500, verbose=0)

        "then"
        res = pd.DataFrame({"close": df["Close"], "reg": model.predict_on_batch(x)}, index=df.index)
        print(res.head())

    def test__LPPLLayer(self):
        """given"""
        model = Sequential([LPPLLayer()])
        model.compile(loss='mse', optimizer=SGD(0.2, 0.01))
        #model.compile(loss='mse', optimizer='adam')

        x = np.log(df["Close"].values)
        x = ReScaler((x.min(), x.max()), (1, 2))(x)
        x = x.reshape(1, -1)

        """when"""
        model.fit(x, x, epochs=5000, verbose=0, callbacks=[EarlyStopping('loss')])

        """then"""
        print(model.predict_on_batch(x))
        res = pd.DataFrame({"close": x[0], "lppl": model.predict_on_batch(x)}, index=df.index)
        res.to_csv('/tmp/lppl.csv')
        print(res.head())
        print(model.layers[0].get_weights())