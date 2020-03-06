import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import gridspec

from quant_ml.plots.bar import ta_stacked_bar
from quant_ml.plots.candlestick import ta_candlestick


def ta_plot(df: pd.DataFrame, figsize=(16, 14), rows=2, cols=1):
    return TaPlot(df, figsize, rows, cols)


class TaPlot(object):

    def __init__(self, df: pd.DataFrame, figsize=(12, 8), rows=2, cols=1, main_height_ratio=4):
        fig = plt.figure('r-', figsize=figsize)
        grid = gridspec.GridSpec(rows, cols, height_ratios=[main_height_ratio, *[1 for _ in range(1, rows)]])
        axis = []

        for i, gs in enumerate(grid):
            ax = fig.add_subplot(gs, sharex=axis[0] if i > 0 else None)
            ax.xaxis_date()
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))

            if i < rows - 1:
                ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
            else:
                ax.tick_params(axis='x', labelrotation=45)

            axis.append(ax)

        plt.xticks(rotation=45)

        self.df = df
        self.x = mdates.date2num(df.index)
        self.fig = fig
        self.axis = axis
        self.grid = grid

    def candlestick(self, open="Open", high="High", low="Low", close="Close", panel=0):
        self.axis[panel] = ta_candlestick(self.df, open, high, low, close, ax=self.axis[panel])
        return self._return()

    def stacked_bar(self, columns, padding=0.02, panel=1, **kwargs ):
        self.axis[panel] = ta_stacked_bar(self.df, columns, ax=self.axis[panel], padding=padding, **kwargs)
        return self._return()

    def bar(self, fields="Volume", panel=1, **kwargs):
        self.axis[panel].bar(self.x, height=self.df[fields].values, **kwargs)
        return self._return()

    def line(self, fields="Close", panel=0, **kwargs):
        self.axis[panel].plot(self.x, self.df[fields].values, **kwargs)
        return self._return()

    def __call__(self, *args, **kwargs):
        if "lines" in kwargs:
            self.line(kwargs.pop('lines', None), **kwargs)
        else:
            self.line()

        if "bars" in kwargs:
            self.bar(kwargs.pop('bars', None), **kwargs)
        else:
            self.bar()

    def _return(self):
        self.grid.tight_layout(self.fig)

