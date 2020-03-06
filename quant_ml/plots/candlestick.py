import matplotlib.dates as mdates
import pandas as pd
from mpl_finance import candlestick_ohlc

from quant_ml.plots.utils import new_fig_axis


def ta_candlestick(self, open="Open", high="High", low="Low", close="Close", ax=None, figsize=None, **kwargs):
    df = self if isinstance(self, pd.DataFrame) else self._parent

    if ax is None:
        fig, ax = new_fig_axis(figsize)

    # Plot candlestick chart
    data = pd.DataFrame({
        "Date": mdates.date2num(df.index),
        "open": df[open],
        "high": df[high],
        "low": df[low],
        "close": df[close],
    })

    candlestick_ohlc(ax, data.values, width=0.6, colorup='g', colordown='r')
    return ax