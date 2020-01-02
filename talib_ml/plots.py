from typing import Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axis import Axis
from matplotlib.figure import Figure


def plot_heated_stacked_area(df: pd.DataFrame,
                             lines: str,
                             heat: str,
                             backtest: str = None,
                             reset_y_lim: bool = False,
                             figsize: Tuple[int, int] = (16, 9),
                             color_map: str = 'afmhot') -> Tuple[Figure, Axis]:
    color_function = plt.get_cmap(color_map)
    x = df.index
    y = df[lines].values
    c = df[heat].values
    b = df[backtest].values if backtest is not None else None

    # make sure we have one more line as heats
    assert len(y.shape) > 1 and len(c.shape) > 1 and y.shape[1] - 1 == c.shape[1], \
        f'unexpeced shapes: {len(y.shape)} > 1 and {len(c.shape)} > 1 and {y.shape[1] - 1} == {c.shape[1]}'

    fig, ax = plt.subplots(figsize=figsize)
    plt.plot(x, y, color='k', alpha=0.0)

    for ci in range(c.shape[1]):
        for xi in range(len(x)):
            plt.fill_between(x[xi-1:xi+1], y[xi-1:xi+1, ci], y[xi-1:xi+1, ci+1],
                             facecolors=color_function(c[xi-1:xi+1, ci]))

        plt.annotate(f'{y[-1, ci]:.2f}', xy=(mdates.date2num(x[-1]), y[-1, ci]),
                     xytext=(4, -4), textcoords='offset pixels')

    # reset limits
    if reset_y_lim:
        plt.ylim(bottom=y[:, 1].min(), top=y[:, -1].max())

    # backtest
    if backtest:
        plt.plot(x, b)

    return fig, ax