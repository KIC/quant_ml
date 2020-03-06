import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from mpl_finance import candlestick_ohlc


def ta_candlestick(self, open="Open", high="High", low="Low", close="Close", ax=None, **kwargs):
    df = self if isinstance(self, pd.DataFrame) else self._parent

    if ax is None:
        fig, ax = plt.subplots()
        ax.xaxis_date()
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
        plt.xticks(rotation=45)

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