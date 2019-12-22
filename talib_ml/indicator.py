import pandas as _pd
import numpy as _np
from typing import Iterable, Union

from talib_ml.util import one_hot


def bbands(df: _pd.DataFrame, stddev: float = 2.0):
    mean = df.mean()
    std = df.std()

    upper = mean + (std * stddev)
    lower = mean - (std * stddev)
    most_recent = df.apply(lambda x: x[-1])
    z_score = (most_recent - mean) / std

    if isinstance(df, _pd.Series):
        upper.name = "UPPER"
        mean.name = "MEAN"
        lower.name = "LOWER"
        z_score.name = "Z"
    else:
        upper.columns = _pd.MultiIndex.from_product([["UPPER"], upper.columns])
        mean.columns = _pd.MultiIndex.from_product([["MEAN"], mean.columns])
        lower.columns = _pd.MultiIndex.from_product([["LOWER"], lower.columns])
        z_score.columns = _pd.MultiIndex.from_product([["Z"], z_score.columns])

    return _pd.DataFrame(upper) \
        .join(mean) \
        .join(lower) \
        .join(z_score) \
        .swaplevel(i=0, j=1, axis=1) \
        .sort_index(axis=1)


def future_pct_of_mean(df: _pd.DataFrame, lag: int = 1):
    # (price / mean[t-x]) - 1
    most_recent = df.apply(lambda x: x[-1])
    mean = df.mean().shift(-lag)

    return (most_recent / mean) - 1


# TODO implement this indicators and test them against ta-lib
#                                macd            = lambda df: talib.MACD(df["Close"])[0] / 10,
#                                macd_signal     = lambda df: talib.MACD(df["Close"])[1] / 10,
#                                plus_di         = lambda df: talib.PLUS_DI(df["High"], df["Low"], df["Close"]) / 100,
#                                mom             = lambda df: talib.MOM(df["Close"]) / 100,
#                                apo             = lambda df: talib.APO(df["Close"]) / 10,
#                                atr             = lambda df: talib.ATR(df["High"], df["Low"], df["Close"]) / 10,
#                                minus_dm        = lambda df: talib.MINUS_DM(df["High"], df["Low"]) / 100,
#                                trix            = lambda df: talib.TRIX(df["Close"]),
#                                plus_dm         = lambda df: talib.PLUS_DM(df["High"], df["Low"]) / 100,