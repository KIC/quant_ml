import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from typing import List

from quant_ml.plots.utils import new_fig_axis
from quant_ml.util import pandas_data


def ta_stacked_bar(self, columns, figsize=None, ax=None, padding=0.02, **kwargs):
    df = self if isinstance(self, pd.DataFrame) else self._parent

    if ax is None:
        fig, ax = new_fig_axis(figsize)

    if padding is not None:
        b, t = ax.get_ylim()

        if b == 0 and t == 1:
            b = np.inf
            t = -np.inf

        ax.set_ylim(min(df[columns].values.min(), b) * (1 - padding), max(df[columns].values.max(), t) * (1 + padding))

    bottom = None
    for column in columns:
        data = pandas_data(df, column)

        if bottom is not None:
            kwargs["bottom"] = bottom
            height = data - bottom
        else:
            height = data

        bottom = height if bottom is None else bottom + height
        ax.bar(mdates.date2num(df.index), height, **kwargs)

    return ax


def ta_bars(self, columns, figsize=None, ax=None, padding=0.02, **kwargs):
    df = self if isinstance(self, pd.DataFrame) else self._parent
    # FIXME allow to pass pandas objects as columns
    columns = columns if isinstance(columns, List) else list(columns)

    if ax is None:
        fig, ax = new_fig_axis(figsize)

    if padding is not None:
        b, t = ax.get_ylim()

        if b == 0 and t == 1:
            b = np.inf
            t = -np.inf

        ax.set_ylim(min(df[columns].values.min(), b) * (1 - padding), max(df[columns].values.max(), t) * (1 + padding))

    width = 1 * (1-padding) / len(columns)
    for i, column in enumerate(columns):
        ax.bar(mdates.date2num(df.index) + width * i, df[column].values, width, **kwargs)

    return ax
