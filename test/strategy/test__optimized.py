import logging
from unittest import TestCase

import numpy as np

from pandas_ml_utils.pandas_utils_extension import inner_join, cloc2
from quant_ml.indicator.features import ta_zscore, ta_ewma_covariance
from quant_ml.strategy.optimized import ta_markowitz
from test import DF_TEST_MULTI

logging.basicConfig(level=logging.DEBUG)


class TestOptimizedStrategies(TestCase):

    def test__ewma_markowitz(self):
        """given"""
        df = DF_TEST_MULTI

        """when"""
        portfolios = ta_markowitz(df, return_period=20, result='weights')

        """then"""
        print(portfolios)
        np.testing.assert_array_almost_equal(np.array([0.683908, 3.160920e-01]), portfolios.iloc[-4].values, 0.00001)

    def test__given_expected_returns(self):
        # _default_returns_estimator
        pass

    def test__markowitz_strategy(self):
        """given"""
        df = DF_TEST_MULTI
        df_price = cloc2(df, ['Close'])
        df_expected_returns = cloc2(df_price["Close"].ta_macd(), "histogram")
        df_trigger = ta_zscore(df_price['Close']).abs() > 2.0
        df_data = inner_join(df_price, df_expected_returns, prefix='expectation')
        df_data = inner_join(df_data, df_trigger, prefix='trigger')

        """when"""
        portfolios = ta_markowitz(df_data,
                                  period=20,
                                  prices='Close',
                                  expected_returns='expectation',
                                  rebalance_trigger='trigger',
                                  result='weights')

        """then"""
        print(portfolios)
        # np.testing.assert_array_almost_equal(np.array([0.683908, 3.160920e-01]), portfolios.iloc[-4].values, 0.00001)
