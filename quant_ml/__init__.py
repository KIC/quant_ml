"""Augment pandas DataFrame with quant methods for machine learning"""
__version__ = '0.0.1'


from pandas.plotting import register_matplotlib_converters

import pandas_ml_utils as pmu
import quant_ml.discrete.candlestick as candlesticks
import quant_ml.discrete.categorize as categorize
import quant_ml.indicator.features as indicators
import quant_ml.indicator.forecast as forecast
import quant_ml.indicator.labels as forward_indicators
import quant_ml.plots.candlestick as candlestick
import quant_ml.strategy.optimized as optimized_strategies
from pandas_ml_utils import *
from pandas_ml_utils import _PandasObject

# just call version to allow "optimization of imports"
pmu.__version__
np.__version__


for indicator_functions in [indicators, forward_indicators, categorize, forecast, candlesticks, optimized_strategies]:
    for indicator_function in dir(indicator_functions):
        if indicator_function.startswith("ta_"):
            setattr(_PandasObject, indicator_function, getattr(indicator_functions, indicator_function))


for plot_functions in [candlestick]:
    for plot_function in dir(plot_functions):
        if plot_function.startswith("ta_"):
            setattr(pd.DataFrame.plot, plot_function, getattr(plot_functions, plot_function))


# register matplotlib converters
register_matplotlib_converters()