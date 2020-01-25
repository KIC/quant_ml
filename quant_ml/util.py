from collections import OrderedDict

import numpy as np


def one_hot(index, len):
    arr = np.zeros(len)

    if isinstance(index, int):
        arr[index] = 1.0
    elif index.values >= 0:
        arr[index] = 1.0
    else:
        arr += np.NAN

    return arr


def unique(items):
    return list(OrderedDict.fromkeys(items))


def arange_open(start, stop, step, round=None):
    arr = np.arange(start, stop + step / 10, step)

    if len(arr) > 0:
        arr[0] = -float('inf')
        arr[-1] = float('inf')

    return arr if round is None else arr.round(round)


class LazyInit(object):

    def __init__(self, supplier):
        self.supplier = supplier
        self.init = True

    def __call__(self, *args, **kwargs):
        if self.init:
            self.supplier = self.supplier()
            self.init = False

        return self.supplier

