import os
from unittest import TestCase

import pandas as pd
from sklearn.neural_network import MLPClassifier

import pandas_ml_utils as pmu
import talib_ml as tml
from talib_ml.encoders import IntervalIndexEncoder

df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "SPY.csv"), index_col='Date')
print(tml.__version__)


class TestEncoder(TestCase):

    def test__inderval_index_encoder(self):
        """given features and labels"""
        df["sma_ratio"] = df["Close"].rolling(20).ta_future_pct_of_mean(0)
        df["forward_sma_ratio"] = df["Close"].rolling(20).ta_future_pct_of_mean(3)

        """and an IntervalIndex"""
        buckets = pd.IntervalIndex.from_breaks([-float("inf"), -0.05, 0.0, 0.05, float("inf")])

        """and a model"""
        model = pmu.SkitModel(
            MLPClassifier(activation='tanh', hidden_layer_sizes=(60, 50), random_state=42),
            pmu.FeaturesAndLabels(features=['sma_ratio'],
                                  labels=IntervalIndexEncoder("forward_sma_ratio", buckets)))

        """when"""
        fit = df.fit(model, test_size=0.4, test_validate_split_seed=42,)
        predicted = df.predict(fit.model)

        """then"""
        print(predicted)
        self.assertEqual(df.predict(fit.model), False) # FIXME

