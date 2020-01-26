from minisom import MiniSom as _MiniSom
from quant_ml.util import one_hot as _one_hot
import pandas as _pd
import numpy as _np
import os as _os

# init
_som = _MiniSom(7, 6, 4, random_seed=2)
_som._weights = _np.load(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "candlestick_minisom_weights_7x6.npy"))

"""
Script to reproduce the result
%matplotlib inline

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from minisom import MiniSom
from pylab import bone, pcolor, colorbar


path = "../../data/private/stock-prices/"
candles = None
for r, d, f in os.walk(path):
    for file in f:
        if file.endswith('.csv'):
            print(f'{file}')
            filename = os.path.join(r, file)
            raw = pd.read_csv(filename, parse_dates=True, index_col='date', skiprows=14)

            relative = pd.DataFrame(index=raw.index)
            relative["open"] = (np.log(raw["open"]) - np.log(raw["close"].shift(1)))
            relative["close"] = (np.log(raw["close"]) - np.log(raw["close"].shift(1)))
            relative["high"] = (np.log(raw["high"]) - np.log(raw["close"].shift(1)))
            relative["low"] = (np.log(raw["low"]) - np.log(raw["close"].shift(1)))

            # relative["volume"] = (np.log(raw["volume"]) - np.log(raw["volume"].shift(1)))
            relative = relative.dropna()
            candles = relative.values if candles is None else np.vstack([candles, relative.values])
            
candles.shape



from mpl_finance import candlestick_ochl


# SOM clustering
size = (7, 6)
som = MiniSom(size[0], size[1], candles.shape[1], random_seed=2)
#som.random_weights_init(candles)
som.pca_weights_init(candles)
som.train_random(candles, 1000)
weights = som.get_weights()

bone()
pcolor(som.distance_map().T)
colorbar()

f, ax = plt.subplots(nrows=size[0], ncols=1, sharey=True, figsize=(10,10))
for i in range(weights.shape[0]):
    candlestick_ochl(ax[i],
                     [np.append([j], weights[i,j] + 1) for j in range(weights.shape[1])])
"""


def ta_realative_candles(df: _pd.DataFrame, open="Open", high="High", low="Low", close="Close"):
    relative = _pd.DataFrame(index=df.index)
    relative[open] = (_np.log(df[open]) - _np.log(df[close].shift(1)))
    relative[close] = (_np.log(df[close]) - _np.log(df[close].shift(1)))
    relative[high] = (_np.log(df[high]) - _np.log(df[close].shift(1)))
    relative[low] = (_np.log(df[low]) - _np.log(df[close].shift(1)))
    return relative


def ta_candle(df: _pd.DataFrame, open="Open", high="High", low="Low", close="Close", make_relative=True, flatten=False, one_hot_encoded=False):
    relative = ta_realative_candles(df, open, high, low, close) if make_relative else df[[open, close, high, low]]
    winners = relative.apply(lambda x: _np.nan if _np.isnan(x).any() else _som.winner(x), raw=True, axis=1)

    if flatten or one_hot_encoded:
        winners = winners.apply(lambda x: -1 if _np.isnan(x).any() else (x[0] * 6 + x[1])).astype(int)
        if one_hot_encoded:
            winners = winners.apply(lambda x: _one_hot(x, 6))

    return winners


def fit(df: _pd.DataFrame, size=(7, 6), open="Open", high="High", low="Low", close="Close", make_relative=True, iter=1000, random_seed=2):
    relative = ta_realative_candles(df, open, high, low, close) if make_relative else df[[open, close, high, low]]
    candles = relative.dropna().values
    som = _MiniSom(size[0], size[1], candles.shape[1], random_seed=random_seed)
    som.pca_weights_init(candles)
    som.train_random(candles, iter)
    return som


def plot_distances(som_model):
    from pylab import bone, pcolor, colorbar
    bone()
    pcolor(som_model.distance_map().T)
    colorbar()


def plot_candles(som_model):
    import matplotlib.pyplot as plt
    from mpl_finance import candlestick_ochl

    weights = som_model.get_weights()

    f, ax = plt.subplots(nrows=weights[0], ncols=1, sharey=True, figsize=(10, 10))
    for i in range(weights.shape[0]):
        candlestick_ochl(ax[i],
                         [_np.append([j], weights[i, j] + 1) for j in range(weights.shape[1])])
