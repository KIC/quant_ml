import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib import gridspec


def ta_stacked_bar(self, columns, figsize=None, ax=None, padding=0.02, **kwargs):
    df = self if isinstance(self, pd.DataFrame) else self._parent

    if ax is None:
        fig = plt.figure('r-', figsize=figsize)
        grid = gridspec.GridSpec(1, 1)
        ax = fig.add_subplot(grid[0])
        ax.xaxis_date()
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
        plt.xticks(rotation=45)

    if padding is not None:
        b, t = ax.get_ylim()

        if b == 0 and t == 1:
            b = np.inf
            t = -np.inf

        ax.set_ylim(min(df[columns].values.min(), b) * (1 - padding), max(df[columns].values.max(), t) * (1 + padding))

    bottom = None
    for column in columns:
        if bottom is not None:
            kwargs["bottom"] = bottom
            height = df[column] - bottom
        else:
            height = df[column]

        bottom = height if bottom is None else bottom + height
        ax.bar(mdates.date2num(df.index), height, **kwargs)

    return ax
