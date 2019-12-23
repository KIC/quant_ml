from unittest import TestCase
from keras import backend as K, Sequential

from talib_ml import one_hot
import pandas_ml_utils as pmu
import numpy as np
import pandas as pd
import os

from talib_ml.keras.layers import LinearRegressionLayer, LPPLLayer

df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "SPY.csv"), index_col='Date')
os.environ["CUDA_VISIBLE_DEVICES"] = ""
print(pmu.__version__)


class TestKerasLayer(TestCase):

    def test__LinearRegressionLayer(self):
        """given"""
        model = Sequential()
        model.add(LinearRegressionLayer())
        model.compile(loss='mse', optimizer='adam')

        x = df["Close"].values.reshape(1, -1)
        model.fit(x, x, epochs=500)

        model.predict_on_batch(x)

    def test__LPPLLayer(self):
        """given"""
        model = Sequential()
        model.add(LPPLLayer())
        model.compile(loss='mse', optimizer='adam')

        x = df["Close"].values.reshape(1, -1)
        model.fit(x, x, epochs=500)

        model.predict_on_batch(x)