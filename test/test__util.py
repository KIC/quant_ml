from unittest import TestCase
import numpy as np
import talib
from quant_ml.util import wilders_smoothing


class TestUtil(TestCase):

    def test__wilders_smoothing(self):
        a = np.arange(1.0, 21)
        res = np.full(a.shape, np.nan, dtype=float)
        wilders_smoothing(a, 2, res)

        np.testing.assert_array_almost_equal(res[-3:], talib.EMA(a, 2 * 2 - 1)[-3:], 0.001)