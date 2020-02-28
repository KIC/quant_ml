import datetime as dt

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from mpl_finance import candlestick_ohlc, volume_overlay
import pandas as pd

def ta_candlestick(self, open="Open", high="High", low="Low", close="Close", volume=None, ax=None, **kwargs):
    df = self._parent

    if ax is None:
        fig, ax = plt.subplots()

    data = pd.DataFrame({
        "Date": mdates.date2num(df.index),
        "open": df[open],
        "high": df[high],
        "low": df[low],
        "close": df[close],
    })

    if volume is not None:
        data["volume"] = df[volume]

    # Plot candlestick chart
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
    plt.xticks(rotation=45)

    if volume is not None:
        candlestick(ax, data.values, width=0.6, colorup='g', colordown='r')
    else:
        candlestick_ohlc(ax, data.values, width=0.6, colorup='g', colordown='r')

    return ax