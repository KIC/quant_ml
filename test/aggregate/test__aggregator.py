import os
import numpy as np
import pandas as pd
from unittest import TestCase

import quant_ml as qml
from quant_ml.aggregate.aggregator import max_draw_down

#df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "SPY.csv"), index_col='Date')
print(qml.__version__)


class TestAggregator(TestCase):

    def test__max_draw_down(self):
        """given"""
        df = pd.DataFrame({"a": [1.0, 1.2, 1.1, 0.9, 0.7, 0.9]})

        """when"""
        dd = max_draw_down(df["a"])

        """then"""
        self.assertAlmostEqual(dd, -0.3)
