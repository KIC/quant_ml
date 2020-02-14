from unittest import TestCase
import pandas as pd

from quant_ml.strategy.optimized import ta_markowitz
from test import DF_TEST_MULTI


class TestOptimizedStrategies(TestCase):

    def test__markowitz(self):
        """given"""
        df = DF_TEST_MULTI
        df_price = df.loc[:, (slice(None), 'Close')].swaplevel(0, 1, axis=1)

        """when"""
        portfolios = ta_markowitz(df_price, period=20)

        """then"""
        print(portfolios)
        pass
