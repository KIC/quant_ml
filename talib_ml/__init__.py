"""Augment pandas DataFrame with methods for machine learning"""
__version__ = '0.0.1'

import talib_ml.indicator.features as indicators
import talib_ml.indicator.labels as forward_indicators
import talib_ml.discrete.categorize as categorize
from pandas.core.base import PandasObject

for indicator_functions in [indicators, forward_indicators, categorize]:
    for indicator_function in dir(indicator_functions):
        if indicator_function.startswith("ta_"):
            setattr(PandasObject, indicator_function, getattr(indicator_functions, indicator_function))
