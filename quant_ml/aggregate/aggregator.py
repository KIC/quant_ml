import numpy as np


def max_draw_down(x):
    return (np.cumprod(np.diff(x) / x[:-1] + 1.0)).min() - 1.0


def max_peak(x):
    return (np.cumprod(np.diff(x) / x[:-1] + 1.0)).max() - 1.0


def profit_and_loss(x):
    return x[-1] / x[0] - 1.0


def range_label(x):
    return f'{x[0]} - {x[-1]}'


def pct_std(x):
    return (np.diff(x) / x[:-1]).std()