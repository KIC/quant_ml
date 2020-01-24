import pandas as _pd
from typing import Union as _Union

# create convenient type hint
from pandas.core.base import PandasObject

_PANDAS = _Union[_pd.DataFrame, _pd.Series]


def ta_future_pct_of_mean(df: _pd.DataFrame, period=14, lag: int = 1):
    # (price / mean[t-x]) - 1
    most_recent = df.rolling(period).apply(lambda x: x[-1]).shift(-lag)
    mean = df.rolling(period).mean()

    return (most_recent / mean) - 1


