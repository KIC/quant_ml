import os
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

    def test__xxx(self):
        cat = df.rolling(20).future_pct_of_mean(1).bucketize(10).tail()
        oht = cat.one_hot_categories()
        print(oht.tail())

        y = oht[:1]

        # decode back to percentage
        y.one_hot_to_categories([cat[col].cat.categories for col in cat.columns])
        self.assertTrue(True)

