import numpy as np


def max_draw_down(x):
    return (np.cumprod(np.diff(x) / x[:-1] + 1.0)).min() - 1.0

