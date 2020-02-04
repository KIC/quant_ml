import logging
import math

import pandas as _pd
import numpy as _np
from typing import Iterable, Union

import quant_ml.util as util

_log = logging.getLogger(__name__)


def ta_convolution(df: _pd.DataFrame, period=90, buckets=10):
    # TODO take all columns of the data frame and transform them into a 2D array where the columns are one hot encoded
    #   trading days
    # per column:
    #   take min/max
    #   then construct an IntervalIndex using [min, ..., max]
    #   then take each value in the column and bucketize it
    def convoluted(df: _pd.Series) -> _np.ndarray:
        min = df.min()
        max = df.max()
        interval_index = _np.linspace(min, max, buckets)
        indexes = _np.digitize(df, interval_index) - 1
        _np.array([util.one_hot(index, buckets) for index in indexes])

    df.rolling(period).apply(convoluted)

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
        return _pd.DataFrame(_pd.cut(df, buckets))
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


def ta_one_hot_categories(df: _pd.DataFrame):
    """
    Take a category column or a column of integers and turn them into a one hot encoded data frame
    :param df: a series or a data frame which has category columns or integer columns
    :return: a multi index data frame with one hot encoded integer columns
             note: can be empty
    """

    df = df.to_frame() if isinstance(df, _pd.Series) else df
    res = None

    for col in df.columns:
        if hasattr(df[col], "cat"):
            categories = [str(cat) for cat in df[col].cat.categories]
            df_of_categories = index_of_categories(df[col])

        elif df[col].dtype.kind in 'iu':
            categories = sorted(set(df[col].values))
            df_of_categories = df

        else:
            continue

        number_of_categories = len(categories)
        ohdf = df_of_categories[[col]].apply(lambda r: util.one_hot(r, number_of_categories), axis=1, result_type='expand')
        ohdf.columns = _pd.MultiIndex.from_product([[col], categories])
        res = ohdf if res is None else res.join(ohdf)

    if res is None:
        _log.warning(f'non of the {df.columns} are of type category index or integer value!\n'
                     f'You might want to call df.ta_one_hot_categories(df.ta_bucketize(3))')

    return res


def index_of_categories(df: _pd.DataFrame):
    """
    Convert a pandas category index into the integer (index) values of the categories

    :param df: data frame or series with a one category column
    :return: a data frame or series of integer values corresponding to the indexes in the category index.
             note: can be empty
    """
    if isinstance(df, _pd.DataFrame):
        return _pd.DataFrame({col: df[col].cat.codes.values for col in df.columns}, index=df.index)
    elif isinstance(df, _pd.Series):
        return _pd.DataFrame(df.cat.codes.values, index=df.index, columns=[df.name])
    else:
        raise ValueError(f"unsupported type {type(df)}")


def one_hot_to_categories(df: _pd.DataFrame, categories):
    res = None

    for l in range(len(categories)):
        if df.columns.nlevels > 1:
            cat = categories[l]
            name = util.unique(df.columns.get_level_values(0))[l]
            dat = df[name]
        else:
            cat = categories
            name = l
            dat = df

        idx = dat.apply(lambda x: _np.argmax(x) if not _pd.isna(x).any() else _np.nan, axis=1, raw=True)
        cat = idx.apply(lambda row: cat[row] if not _pd.isna(row) else _np.nan).rename(name)
        res = cat.to_frame() if res is None else res.join(cat)

    return res

