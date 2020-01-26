from unittest import TestCase

from quant_ml.discrete.candlestick import *
from test import DF_TEST, DF_DEBUG


class TestEncoder(TestCase):

    def test__candlestick(self):
        categories = ta_candle(DF_TEST, flatten=True)
        print(categories.index[-2])
        self.assertEqual(21, categories.iloc[-2])
