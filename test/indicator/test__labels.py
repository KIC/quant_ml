from unittest import TestCase

import numpy as np
import pandas as pd
import talib

from quant_ml.indicator.features import *
from quant_ml.indicator.labels import *
from test import DF_TEST, DF_DEBUG


class TestIndicators(TestCase):

    def test__future_pct_of_mean(self):
        """given"""
        x = DF_TEST[["Close"]]

        """when"""
        x["sma"] = x.rolling(2).mean()
        x["fpm"] = ta_future_pct_of_mean(x["Close"], 1, 2)

        """then"""
        print(f"\n{x.tail()}")
        self.assertAlmostEqual(x["sma"].iloc[-2], 310.504990, 5)
        self.assertAlmostEqual(x["Close"].iloc[-1], 312.089996, 5)
        self.assertAlmostEqual(x["fpm"].iloc[-2], 312.089996 / 310.504990 - 1, 5)
        self.assertAlmostEqual(x["sma"].iloc[-2] * (312.089996 / 310.504990), x["Close"].iloc[-1], 5)
        self.assertTrue(np.isnan(x["fpm"].iloc[-1]))

    def test__future_sma_cross(self):
        fast = ta_sma(DF_TEST[["Close"]], 3)
        slow = ta_sma(DF_TEST[["Close"]], 5)
        cross = ta_cross_under(None, fast, slow)[-10:] # -3 == True
        future_cross = ta_future_sma_cross(DF_TEST["Close"], 2, 3, 5)

        self.assertTrue(cross.iloc[-3].values[0])
        self.assertTrue(future_cross.iloc[-5])

    def test__future_macd_cross(self):
        cross = ta_macd(DF_TEST["Close"], 3, 5, 2)
        fcross = ta_future_macd_cross(DF_TEST["Close"], 2, 3, 5, 2)

        self.assertTrue(fcross.iloc[-3])

    def test__future_bband_quantile(self):
        q = ta_future_bband_quantile(DF_TEST["Close"], 2)

        print(q[-10:])
        self.assertEqual(-1, q.iloc[-5])
        self.assertEqual(0, q.iloc[-6])
        self.assertEqual(0, q.iloc[-7])
        self.assertEqual(1, q.iloc[-8])

    def test__future_multiband_bucket(self):
        bucket = ta_future_multiband_bucket(DF_TEST["Close"], 2)
        print(bucket[-10:])

        self.assertEqual(4, bucket.iloc[-3])
        self.assertEqual(1, bucket.iloc[-4])
        self.assertEqual(0, bucket.iloc[-5])
