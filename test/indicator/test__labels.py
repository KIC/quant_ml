from unittest import TestCase

import numpy as np
import pandas as pd
import talib

from talib_ml.indicator.labels import *
from test import DF_TEST, DF_DEBUG


class TestIndicators(TestCase):

    def test__future_pct_of_mean(self):
        """given"""
        x = DF_TEST[["Close"]]

        """when"""
        x["sma"] = x.rolling(2).mean()
        x["fpm"] = ta_future_pct_of_mean(x["Close"], 2, 1)

        """then"""
        print(f"\n{x.tail()}")
        self.assertAlmostEqual(x["sma"].iloc[-2], 310.504990, 5)
        self.assertAlmostEqual(x["Close"].iloc[-1], 312.089996, 5)
        self.assertAlmostEqual(x["fpm"].iloc[-2], 312.089996 / 310.504990 - 1, 5)
        self.assertAlmostEqual(x["sma"].iloc[-2] * (312.089996 / 310.504990), x["Close"].iloc[-1], 5)
        self.assertTrue(np.isnan(x["fpm"].iloc[-1]))

