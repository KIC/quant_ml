import pandas as _pd
from typing import Union as _Union

# create convenient type hint
from pandas.core.base import PandasObject
import numpy as _np

from quant_ml.util import wilders_smoothing

_PANDAS = _Union[_pd.DataFrame, _pd.Series]


def ta_sma(df: _PANDAS, period=12) -> _PANDAS:
    return df.rolling(period).mean()


def ta_ema(df: _PANDAS, period=12) -> _PANDAS:
    return df.ewm(span=period, adjust=False, min_periods=period-1).mean()


def ta_wilders(df: _PANDAS, period=12) -> _PANDAS:
    if isinstance(df, _pd.Series):
        return ta_wilders(df.to_frame(), period).iloc[:, 0]
    else:
        resdf = _pd.DataFrame({}, index=df.index)
        for col in df.columns:
            s = df[col].dropna()
            res = _np.zeros(s.shape)
            wilders_smoothing(s.values, period, res)
            resdf = resdf.join(_pd.DataFrame({col: res}, index=s.index))

        return resdf


def ta_macd(df: _PANDAS, fast_period=12, slow_period=26, signal_period=9, relative=True) -> _PANDAS:
    fast = ta_ema(df, fast_period)
    slow = ta_ema(df, slow_period)
    macd = (fast / slow - 1) if relative else (fast - slow)
    signal = ta_ema(macd, signal_period)
    hist = macd - signal

    for label, frame in {"macd": macd, "signal": signal, "histogram": hist}.items():
        if isinstance(frame, _pd.DataFrame) and len(df.columns) > 1:
            frame.columns = _pd.MultiIndex.from_product([frame.columns, [label]])
        else:
            frame.name = label

    macd = macd.to_frame() if isinstance(macd, _pd.Series) else macd
    return macd.join(signal).join(hist)


def ta_mom(df: _PANDAS, period=10) -> _PANDAS:
    return df.diff(period)


def ta_roc(df: _PANDAS, period=10) -> _PANDAS:
    return df.pct_change(period)


def ta_stddev(df: _PANDAS, period=5, nbdev=1, ddof=1) -> _PANDAS:
    return df.rolling(period).std(ddof=ddof) * nbdev


def ta_rsi(df: _PANDAS, period=14):
    returns = df.diff()

    pos = ta_wilders(returns.clip(lower=0), period)
    neg = ta_wilders(_np.abs(returns.clip(upper=0)), period)

    return pos / (pos + neg)


def ta_apo(df: _PANDAS, fast_period=12, slow_period=26, exponential=False) -> _PANDAS:
    fast = ta_ema(df, fast_period) if exponential else ta_sma(df, fast_period)
    slow = ta_ema(df, slow_period) if exponential else ta_sma(df, slow_period)
    return fast - slow


def ta_trix(df: _PANDAS, period=30) -> _PANDAS:
    return ta_ema(ta_ema(ta_ema(df, period), period), period).pct_change() * 100


def ta_tr(df: _PANDAS, high="High", low="Low", close="Close", relative=False) -> _PANDAS:
    h = df[high]
    l = df[low]
    c = df[close].shift(1)

    if relative:
        ranges = (h / l - 1).rename("a").to_frame() \
            .join((h / c - 1).rename("b").abs()) \
            .join((l / c - 1).rename("c").abs())
    else:
        ranges = (h - l).rename("a").to_frame()\
            .join((h - c).rename("b").abs())\
            .join((l - c).rename("c").abs())

    return ranges.max(axis=1).rename("true_range")


def ta_atr(df: _PANDAS, period=14, high="High", low="Low", close="Close", relative=False, exponential='wilder') -> _PANDAS:
    if exponential is True:
        return ta_ema(ta_tr(df, high, low, close, relative), period)
    if exponential == 'wilder':
        return ta_wilders(ta_tr(df, high, low, close, relative), period)
    else:
        return ta_sma(ta_tr(df, high, low, close, relative), period)


def ta_adx(df: _PANDAS, period=14, high="High", low="Low", close="Close") -> _PANDAS:
    temp = _pd.DataFrame({
        "up": df[high] - df[high].shift(1),
        "down": df[low].shift(1) - df[low]
    }, index=df.index)

    atr = ta_atr(df, period, high, low, close, relative=False)
    pdm = ta_wilders(temp.apply(lambda r: r[0] if r["up"] > r["down"] and r["up"] > 0 else 0, raw=False, axis=1), period)
    ndm = ta_wilders(temp.apply(lambda r: r[1] if r["down"] > r["up"] and r["down"] > 0 else 0, raw=False, axis=1), period)

    pdi = pdm / atr
    ndi = ndm / atr
    adx = ta_wilders((pdi - ndi).abs() / (pdi + ndi).abs(), period)

    return _pd.DataFrame({"+DM": pdm, "-DM": ndm, "+DI": pdi, "-DI": ndi, "ADX": adx}, index=df.index)


def ta_bbands(df: _PANDAS, period=5, stddev=2.0, ddof=1) -> _PANDAS:
    mean = df.rolling(period).mean()
    std = df.rolling(period).std(ddof=ddof)
    most_recent = df.rolling(period).apply(lambda x: x[-1])

    upper = mean + (std * stddev)
    lower = mean - (std * stddev)
    z_score = (most_recent - mean) / std

    if isinstance(mean, _pd.Series):
        upper.name = "upper"
        mean.name = "mean"
        lower.name = "lower"
        z_score.name = "z"
    else:
        upper.columns = _pd.MultiIndex.from_product([upper.columns, ["uppen"]])
        mean.columns = _pd.MultiIndex.from_product([mean.columns, ["mean"]])
        lower.columns = _pd.MultiIndex.from_product([lower.columns, ["lower"]])
        z_score.columns = _pd.MultiIndex.from_product([z_score.columns, ["z"]])

    return _pd.DataFrame(upper) \
        .join(mean) \
        .join(lower) \
        .join(z_score) \
        .sort_index(axis=1)


def ta_cross_over(df: _pd.DataFrame, a, b, period=1) -> _PANDAS:
    old_a = (a if isinstance(a, PandasObject) else df[a]).shift(period)
    old_b = (b if isinstance(b, PandasObject) else df[b]).shift(period)
    young_a = (a if isinstance(a, PandasObject) else df[a])
    young_b = (b if isinstance(b, PandasObject) else df[b])
    return (old_a <= old_b) & (young_a > young_b)


def ta_cross_under(df: _pd.DataFrame, a, b) -> _PANDAS:
    return ta_cross_over(df, b, a)


