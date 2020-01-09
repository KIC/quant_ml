"""Augment pandas DataFrame with methods for machine learning"""
__version__ = '0.0.1'

from pandas.core.base import PandasObject
from talib_ml.indicator import *
from talib_ml.categorize import *

PandasObject.ta_bbands = bbands
PandasObject.ta_future_pct_of_mean = future_pct_of_mean
PandasObject.ta_cross_over = cross_over
PandasObject.ta_cross_under = cross_under
PandasObject.ta_bucketize = bucketize
PandasObject.ta_index_of_categories = index_of_categories
PandasObject.ta_one_hot_categories = one_hot_categories
PandasObject.ta_one_hot_to_categories = one_hot_to_categories