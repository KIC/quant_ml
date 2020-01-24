import math

import pandas as _pd
import numpy as _np
from typing import Iterable, Union

from talib_ml.util import one_hot, unique


def ta_convolution(df: _pd.DataFrame, period=90, buckets=10):
    # TODO take all columns of the data frame and transform them into a 2D array where the columns are one hot encoded
    #   trading days
    pass


def ta_bucketize(df: _pd.DataFrame,
                 rrange: Union[_pd.IntervalIndex, Iterable, int],
                 closed: bool = False) -> _pd.DataFrame:
    if isinstance(rrange, _pd.IntervalIndex):
        buckets = rrange
    elif isinstance(rrange, Iterable):
        borders = list(rrange)

        if closed:
            buckets = _pd.IntervalIndex.from_tuples([(borders[r], borders[r + 1]) for r in range(len(borders) - 1)])
        else:
            buckets = _pd.IntervalIndex.from_tuples(
                [(-float("inf") if r == 0 else borders[r], float("inf") if r == len(borders) - 2 else borders[r + 1])
                 for r in range(len(borders) - 1)])
    else:
        buckets = rrange

    # cut each column and return the index
    if isinstance(df, _pd.DataFrame):
        return _pd.DataFrame({col: _pd.cut(df[col], buckets) for col in df.columns}, index=df.index)
    elif isinstance(df, _pd.Series):
        return _pd.DataFrame({df.name: _pd.cut(df, buckets)}, index=df.index)
    else:
        raise ValueError(f"unsupported type {type(df)}")


def percent_bucket_to_target(b, price):
    if isinstance(b, Iterable):
        return [percent_bucket_to_target(b_, price) for b_ in b]
    else:
        if b.right < 0:
            return price * (1 + (b.left if not math.isinf(b.left) else b.right))
        elif b.left > 0:
            return price * (1 + (b.right if not math.isinf(b.right) else b.left))
        else:
            return price


def index_of_categories(df: _pd.DataFrame):
    if isinstance(df, _pd.DataFrame):
        return _pd.DataFrame({col: df[col].cat.codes.values for col in df.columns}, index=df.index)
    elif isinstance(df, _pd.Series):
        return df.cat.codes.values
    else:
        raise ValueError(f"unsupported type {type(df)}")


def ta_one_hot_categories(df: _pd.DataFrame):
    indexes = index_of_categories(df)
    df = df.to_frame() if isinstance(df, _pd.Series) else df
    res = None

    for col in df.columns:
        categories = [str(cat) for cat in df[col].cat.categories]
        l = len(categories)
        ohdf = indexes[[col]].apply(lambda c: one_hot(c, l), axis=1, result_type='expand')
        ohdf.columns = _pd.MultiIndex.from_product([[col], categories])

        res = ohdf if res is None else res.join(ohdf)

    return res


def one_hot_to_categories(df: _pd.DataFrame, categories):
    res = None

    for l in range(len(categories)):
        if df.columns.nlevels > 1:
            cat = categories[l]
            name = unique(df.columns.get_level_values(0))[l]
            dat = df[name]
        else:
            cat = categories
            name = l
            dat = df

        idx = dat.apply(lambda x: _np.argmax(x) if not _pd.isna(x).any() else _np.nan, axis=1, raw=True)
        cat = idx.apply(lambda row: cat[row] if not _pd.isna(row) else _np.nan).rename(name)
        res = cat.to_frame() if res is None else res.join(cat)

    return res

