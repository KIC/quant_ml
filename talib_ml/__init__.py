"""Augment pandas DataFrame with methods for machine learning"""
__version__ = '0.0.1'

import talib_ml.indicator.series as indicators
from pandas.core.base import PandasObject
from talib_ml.indicator.rolling import *
#from talib_ml.indicator.series import *
from talib_ml.discrete.categorize import *

for indicator in dir(indicators):
    if indicator.startswith("ta_"):
        setattr(PandasObject, indicator, getattr(indicators, indicator))

# FIXME move to series
PandasObject.ta_future_pct_of_mean = ta_future_pct_of_mean
PandasObject.ta_cross_over = ta_cross_over
PandasObject.ta_cross_under = ta_cross_under

PandasObject.ta_bucketize = ta_bucketize

# todo better prefix
PandasObject.ta_index_of_categories = index_of_categories
PandasObject.ta_one_hot_categories = one_hot_categories
PandasObject.ta_one_hot_to_categories = one_hot_to_categories