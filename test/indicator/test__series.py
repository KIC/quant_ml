from unittest import TestCase

import numpy as np
import pandas as pd
import talib

from talib_ml.indicator.series import *
from test import DF_TEST, DF_DEBUG


class TestSeriesIndicator(TestCase):

    def test__ema(self):
        me = ta_ema(DF_TEST["Close"], 20)[-100:]
        ta = talib.EMA(DF_TEST["Close"], 20)[-100:]

        np.testing.assert_array_almost_equal(me, ta)

    def test__macd(self):
        my_macd = ta_macd(DF_TEST["Close"], relative=False)
        talib_macd = pd.DataFrame(talib.MACD(DF_TEST["Close"])).T

        np.testing.assert_array_almost_equal(talib_macd[-100:].values[0], my_macd[-100:].values[0])

    def test__mom(self):
        me = ta_mom(DF_TEST["Close"])[-100:]
        ta = talib.MOM(DF_TEST["Close"])[-100:]

        np.testing.assert_array_almost_equal(me, ta)

    def test__apo(self):
        me = ta_apo(DF_TEST["Close"])[-100:]
        ta = talib.APO(DF_TEST["Close"])[-100:]

        np.testing.assert_array_almost_equal(me, ta)

    def test__trix(self):
        me = ta_trix(DF_TEST["Close"])[-100:]
        ta = talib.TRIX(DF_TEST["Close"])[-100:]

        np.testing.assert_array_almost_equal(me, ta)

    def test__tr(self):
        me = ta_tr(DF_TEST, relative=False)[-100:]
        ta = talib.TRANGE(DF_TEST["High"], DF_TEST["Low"], DF_TEST["Close"])[-100:]

        np.testing.assert_array_almost_equal(me, ta)

    def test__atr(self):
        me = ta_atr(DF_TEST, relative=False)[-100:]
        ta = talib.ATR(DF_TEST["High"], DF_TEST["Low"], DF_TEST["Close"])[-100:]

        np.testing.assert_array_almost_equal(me, ta)

