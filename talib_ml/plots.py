from typing import Tuple, List

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import gridspec
from matplotlib.axis import Axis
from matplotlib.figure import Figure
from mlxtend.evaluate import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import roc_curve, auc


def plot_heat_bar(df: pd.DataFrame, prediction_columns, target_columns) -> Tuple[Figure, Axis]:
    fig, ax = plt.subplots(figsize=(2, 9))
    probabilities = df[prediction_columns].values[-1]
    targets = df[target_columns].values[-1, :len(probabilities)].round(2)
    date = df.index[-1]

    return fig, sns.heatmap(pd.DataFrame({f"probability\n{date}": probabilities}, index=targets).iloc[::-1])


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

        if ci > 0:
            # todo annotate all first last and only convert date if it is actually a date
            plt.annotate(f'{y[-1, ci]:.2f}', xy=(mdates.date2num(x[-1]), y[-1, ci]),
                         xytext=(4, -4), textcoords='offset pixels')

    # reset limits
    if reset_y_lim:
        plt.ylim(bottom=y[:, 1].min(), top=y[:, -1].max())

    # backtest
    if backtest:
        plt.plot(x, b)

    return fig, ax


def plot_one_hot_encoded_confusion_matrix(df: pd.DataFrame, true_columns, prediction_columns) -> Tuple[Figure, Axis]:
    y_hat = df[prediction_columns].apply(lambda row: np.argmax(row), raw=True, axis=1)
    y = df[true_columns].apply(lambda row: np.argmax(row), raw=True, axis=1)

    cm = confusion_matrix(y.values, y_hat.values)
    return plot_confusion_matrix(cm, figsize=(12,12))


def plot_ROC(df: pd.DataFrame, true_columns, predicted_columns) -> Tuple[Figure, Axis]:
    dft = df[true_columns]
    dfp = df[predicted_columns]

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(len(dfp.columns)):
        # FIXME we want to count a to big if > 0 or a too small if < 0 as a true as well
        fpr[i], tpr[i], _ = roc_curve(dft.values[:, i], dfp.values[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # plot ROC curves
    fig, axis = plt.subplots(figsize=(25, 10))

    for i in fpr.keys():
        plt.plot(fpr[i], tpr[i], label=f"{dfp.columns[i]} auc:{roc_auc[i] * 100:.2f}")
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    return fig, axis


def ts_bar(df: pd.DataFrame, figsize=(16,9), width=2) -> Tuple[Figure, Axis]:
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(df.index, df, width=width)
    ax.xaxis_date()

    return fig, ax


def plot_strategy_statistics(df: pd.DataFrame, figsize=(21, 9), bins=10) -> Tuple[Figure, List[Axis]]:
    # "count", max_draw_down, profit_and_loss, max_peak, pct_std
    # i want 2 plots the distribution (histogram) of the pnl as well as each trade pnl as bar plot along with its max
    # draw down

    fig = plt.figure(figsize=figsize)
    axis = []
    grid = (1, 3)
    gridspec.GridSpec(*grid)

    ax = plt.subplot2grid(grid, (0, 0))
    df.loc[:, (slice(None), "profit_and_loss")].hist(ax=ax, bins=bins)
    axis.append(ax)

    ax = plt.subplot2grid(grid, (0, 1), colspan=2)
    ax.bar(df.index, df.loc[:, (slice(None), "max_draw_down")].values[:, 0], color='C1', label="max draw down")
    ax.bar(df.index, df.loc[:, (slice(None), "profit_and_loss")].values[:, 0], color='C0', label="PnL")
    ax.legend()
    ax.autoscale(tight=True)
    axis.append(ax)

    return fig, axis
