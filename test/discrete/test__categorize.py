from unittest import TestCase

import numpy as np

from quant_ml.discrete.categorize import *
from quant_ml.indicator.features import ta_sma
from test import DF_TEST


class TestCategorize(TestCase):

    def test__convolution(self):
        ta_convolution(DF_TEST["Close"])
        pass

    def test__one_hot_categories(self):
        """given"""
        osc = DF_TEST["Close"] / ta_sma(DF_TEST["Close"], 20)

        """when"""
        cat = ta_bucketize(osc, 3)
        cat = ta_one_hot_categories(cat)

        """then"""
        np.testing.assert_array_equal(cat[-1:].values, np.array([[0., 0., 1.]]))
