from unittest import TestCase

import numpy as np
import pandas as pd
import talib

from quant_ml.discrete.categorize import *
from test import DF_TEST, DF_DEBUG


class TestCategorize(TestCase):

    def test__convolution(self):
        ta_convolution(DF_TEST["Close"])
        pass

