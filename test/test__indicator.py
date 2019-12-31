import os
import numpy as np
import pandas as pd
from unittest import TestCase

import talib_ml as tml


df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "SPY.csv"), index_col='Date')
print(tml.__version__)


class TestIndicators(TestCase):

    def test__bbands(self):
        res = df.rolling(20).bbands()
        print(res)

        idx = pd.IndexSlice
        self.assertTrue(True)

    def test__future_pct_of_mean(self):
        """given"""
        x = df[["Close"]]

        """when"""
        x["sma"] = x.rolling(2).mean()
        x["fpm"] = x["Close"].rolling(2).ta_future_pct_of_mean(1)

        """then"""
        print(f"\n{x.tail()}")
        self.assertAlmostEqual(x["sma"].iloc[-2], 310.504990, 5)
        self.assertAlmostEqual(x["Close"].iloc[-1], 312.089996, 5)
        self.assertAlmostEqual(x["fpm"].iloc[-2], 312.089996 / 310.504990 - 1, 5)
        self.assertAlmostEqual(x["sma"].iloc[-2] * (312.089996 / 310.504990), x["Close"].iloc[-1], 5)
        self.assertTrue(np.isnan(x["fpm"].iloc[-1]))

