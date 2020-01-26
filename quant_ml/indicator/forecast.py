import pandas as _pd
from typing import Union as _Union

# create convenient type hint
from pandas.core.base import PandasObject
import numpy as _np
from arch import arch_model

_PANDAS = _Union[_pd.DataFrame, _pd.Series]


def ta_arch(df: _PANDAS, returns="pct", **kwargs) -> _PANDAS:
    """
    Volatility modelling of time series data

    :param df: pandas data frame or series
    :param returns: used to transform variables into returns can be None, "pct" or "log"
    :param kwargs: same as :func:`arch.arch_model`
    :return: conditional volatility
    """
    x = df.index.astype(float).values

    if returns == "pct":
        df = df.pct_change()
    elif returns == "log":
        df = _np.log(df) - _np.log(df.shift(1))

    if isinstance(df, _pd.DataFrame):
        res = _pd.DataFrame({}, index=df.index)

        for col in df.columns:
            m = arch_model(df[col].values, x, **kwargs)
            m = m.fit()
            res[col] = m.conditional_volatility
    else:
        m = arch_model(df.values, x, **kwargs)
        m = m.fit()
        res = m.conditional_volatility

    return res
