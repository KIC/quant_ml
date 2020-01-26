"""Augment pandas DataFrame with quant methods for machine learning"""
__version__ = '0.0.1'

import quant_ml.indicator.features as indicators
import quant_ml.indicator.labels as forward_indicators
import quant_ml.indicator.forecast as forecast
import quant_ml.discrete.categorize as categorize
from pandas.core.base import PandasObject

for indicator_functions in [indicators, forward_indicators, categorize, forecast]:
    for indicator_function in dir(indicator_functions):
        if indicator_function.startswith("ta_"):
            setattr(PandasObject, indicator_function, getattr(indicator_functions, indicator_function))
