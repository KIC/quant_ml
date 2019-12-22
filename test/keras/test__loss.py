import os
from unittest import TestCase

import matplotlib.pyplot as plt
import pandas as pd
from keras import Sequential

import talib_ml as tml
from talib_ml.keras.layers import LinearRegressionLayer

df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "SPY.csv"), index_col='Date')
os.environ["CUDA_VISIBLE_DEVICES"] = ""
print(tml.__version__)


class TestKerasLoss(TestCase):

    def test__(self):
        """given"""
        model = Sequential()
        model.add(LinearRegressionLayer())
        model.compile(loss='mse', optimizer='adam')

        """ and"""
        x = df["Adj Close"].values.reshape(1, -1)
        y = x

        """when"""
        history = model.fit(x, y, batch_size=1, epochs=300)

        """then"""
        regression_line = model.predict_on_batch(x)
        rdf = pd.DataFrame({"price": df["Adj Close"], "regression": regression_line}, index=df.index)
        rdf.plot()
        plt.savefig('/tmp/figure.png')

        self.assertTrue(True)