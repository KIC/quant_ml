"""
this module basically re-implements all oscillators from TA-Lib:
  https://mrjbq7.github.io/ta-lib/func_groups/momentum_indicators.html
"""

import pandas as _pd
import numpy as _np
from typing import Union as _Union

# create convenient type hint
from pandas.core.base import PandasObject
import numpy as _np
from pyts.image import GramianAngularField

from quant_ml.util import wilders_smoothing
from scipy.stats import zscore

_PANDAS = _Union[_pd.DataFrame, _pd.Series]


def ta_inverse(df: _PANDAS) -> _PANDAS:
    return df.apply(lambda col: col * -1 + col.min() + col.max(), raw=True)


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


def ta_adx(df: _PANDAS, period=14, high="High", low="Low", close="Close", relative=True) -> _PANDAS:
    temp = _pd.DataFrame({
        "up": (df[high] / df[high].shift(1) - 1) if relative else (df[high] - df[high].shift(1)),
        "down": (df[low].shift(1) / df[low] - 1) if relative else (df[low].shift(1) - df[low])
    }, index=df.index)

    atr = ta_atr(df, period, high, low, close, relative=False)
    pdm = ta_wilders(temp.apply(lambda r: r[0] if r["up"] > r["down"] and r["up"] > 0 else 0, raw=False, axis=1), period)
    ndm = ta_wilders(temp.apply(lambda r: r[1] if r["down"] > r["up"] and r["down"] > 0 else 0, raw=False, axis=1), period)

    pdi = pdm / atr
    ndi = ndm / atr
    adx = ta_wilders((pdi - ndi).abs() / (pdi + ndi).abs(), period)

    return _pd.DataFrame({"+DM": pdm, "-DM": ndm, "+DI": pdi, "-DI": ndi, "ADX": adx}, index=df.index)


def ta_multi_bbands(s: _pd.Series, period=5, stddevs=[1.0, 1.5, 2.0], ddof=1) -> _PANDAS:
    assert isinstance(s, _pd.Series)
    mean = s.rolling(period).mean().rename("mean")
    std = s.rolling(period).std(ddof=ddof)
    df = mean.to_frame()

    for stddev in stddevs:
        df[f'upper-{stddev}'] = mean + (std * stddev)
        df[f'lower-{stddev}'] = mean - (std * stddev)

    return df


def ta_bbands(df: _PANDAS, period=5, stddev=2.0, ddof=1) -> _PANDAS:
    mean = df.rolling(period).mean()
    std = df.rolling(period).std(ddof=ddof)
    most_recent = df.rolling(period).apply(lambda x: x[-1], raw=True)

    upper = mean + (std * stddev)
    lower = mean - (std * stddev)
    z_score = (most_recent - mean) / std
    quantile = (most_recent > upper).astype(int) - (most_recent < lower).astype(int)

    if isinstance(mean, _pd.Series):
        upper.name = "upper"
        mean.name = "mean"
        lower.name = "lower"
        z_score.name = "z"
        quantile.name = "quantile"
    else:
        upper.columns = _pd.MultiIndex.from_product([upper.columns, ["upper"]])
        mean.columns = _pd.MultiIndex.from_product([mean.columns, ["mean"]])
        lower.columns = _pd.MultiIndex.from_product([lower.columns, ["lower"]])
        z_score.columns = _pd.MultiIndex.from_product([z_score.columns, ["z"]])
        quantile.columns = _pd.MultiIndex.from_product([z_score.columns, ["quantile"]])

    return _pd.DataFrame(upper) \
        .join(mean) \
        .join(lower) \
        .join(z_score) \
        .join(quantile) \
        .sort_index(axis=1)


def ta_cross_over(df: _pd.DataFrame, a, b=None, period=1) -> _PANDAS:
    if isinstance(a, int):
        if isinstance(df, _pd.Series):
            a = _pd.Series(_np.ones(len(df)) * a, name=df.name, index=df.index)
        else:
            a = _pd.DataFrame({c: _np.ones(len(df)) * a for c in df.columns}, index=df.index)

    if b is None:
        b = a
        a = df

    old_a = (a if isinstance(a, PandasObject) else df[a]).shift(period)
    young_a = (a if isinstance(a, PandasObject) else df[a])

    if isinstance(b, int):
        if isinstance(old_a, _pd.Series):
            b = _pd.Series(_np.ones(len(df)) * b, name=old_a.name, index=old_a.index)
        else:
            b = _pd.DataFrame({c : _np.ones(len(df)) * b for c in old_a.columns}, index=old_a.index)

    old_b = (b if isinstance(b, PandasObject) else df[b]).shift(period)
    young_b = (b if isinstance(b, PandasObject) else df[b])

    return (old_a <= old_b) & (young_a > young_b)


def ta_cross_under(df: _pd.DataFrame, a, b=None, period=1) -> _PANDAS:
    if b is None:
        b = a
        a = df

    return ta_cross_over(df, b, a, period)


def ta_williams_R(df: _pd.DataFrame, period=14, close="Close", high="High", low="Low") -> _pd.Series:
    temp = df[[close]]
    temp = temp.join(df[high if high is not None else close].rolling(period).max().rename("highest_high"))
    temp = temp.join(df[low if low is not None else close].rolling(period).min().rename("lowest_low"))
    return (temp["highest_high"] - temp[close]) / (temp["highest_high"] - temp["lowest_low"])


def ta_ultimate_osc(df: _pd.DataFrame, period1=7, period2=14, period3=28, close="Close", high="High", low="Low") -> _pd.Series:
    # BP = Close - Minimum(Low or Prior Close).
    # TR = Maximum(High or Prior Close)  -  Minimum(Low or Prior Close)
    prev_close = df[close].shift(1)
    downs = (df[[low if low is not None else close]].join(prev_close)).min(axis=1)
    ups = (df[[high if high is not None else close]].join(prev_close)).max(axis=1)
    temp = _pd.DataFrame({
        "bp": df[close] - downs,
        "tr": ups - downs
    }, index=df.index)

    periods = [period1, period2, period3]
    avs = []
    for period in periods:
        # Average7 = (7 - period BP Sum) / (7 - period TR Sum)
        av = temp.rolling(period).sum()
        avs.append(av["bp"] / av["tr"])

    # UO = [(4 x Average7) + (2 x Average14) + Average28] / (4 + 2 + 1)
    return (4 * avs[0] + 2 * avs[1] + avs[2]) / 7


def ta_ppo(df: _pd.DataFrame, fast_period=12, slow_period=26, exponential=True) -> _PANDAS:
    fast = ta_ema(df, period=fast_period) if exponential else ta_sma(df, period=fast_period)
    slow = ta_ema(df, period=slow_period) if exponential else ta_sma(df, period=slow_period)
    return (fast - slow) / slow


def ta_bop(df: _pd.DataFrame, open="Open", high="High", low="Low", close="Close") -> _PANDAS:
    # (CLOSE – OPEN) / (HIGH – LOW)
    return (df[close] - df[open]) / (df[high] - df[low])


def ta_cci(df: _pd.DataFrame, period=14, ddof=1, high="High", low="Low", close="Close", alpha=0.015) -> _PANDAS:
    tp = (df[high] + df[low] + df[close]) / 3
    tp_sma = ta_sma(tp, period)
    md = tp.rolling(period).apply(lambda x: _np.abs(x - x.mean()).sum() / period)
    return (1 / alpha) * (tp - tp_sma) / md / 100


def ta_up_down_volatility_ratio(df: _PANDAS, period=60, normalize=True, setof_date=False):
    if isinstance(df, _pd.DataFrame):
        return df.apply(lambda col: ta_up_down_volatility_ratio(col, period, normalize))

    returns = df.pct_change() if normalize else df
    std = _pd.DataFrame({
        "std+": returns[returns > 0].rolling(period).std(),
        "std-": returns[returns < 0].rolling(period).std()
    }, index=returns.index).fillna(method="ffill")

    ratio = (std["std+"] / std["std-"] - 1).rename("std +/-")

    # eventually we can off set the date such that we can fake one continuous data frame
    if setof_date:
        # +7 -1 binds us approximately to the same week day
        ratio.index = ratio.index - _pd.DateOffset(days=(ratio.index[-1] - ratio.index[0]).days + 7 - 1)

    return ratio


def ta_zscore(df: _PANDAS, period=20, ddof=1):
    return df.rolling(period).apply(lambda c: zscore(c, ddof=ddof)[-1])


def ta_ewma_covariance(df: _PANDAS, convert_to='returns', alpha=0.97):
    data = df

    if convert_to == 'returns':
        data = df.pct_change()
    if convert_to == 'log-returns':
        data = _np.log(df) - _np.log(df.shift(1))

    return data.ewm(com=alpha).cov()


def ta_gaf(df: _PANDAS,
          period=20,
          image_size=24,
          sample_range=(-1, 1),
          method='summation',
          flatten=False,
          overlapping=False):

    def to_gaf(df):
        gaf = GramianAngularField(image_size=image_size, sample_range=sample_range, method=method,
                                  overlapping=overlapping, flatten=flatten)
        return gaf.fit_transform(df.values)

    return _pd.Series([to_gaf(df.iloc[i-period, period]) for i in range(period, len(df))], index=df.index, name="GAF")



"""
TODO add this missing indicators

AROONOSC - Aroon Oscillator
real = AROONOSC(high, low, timeperiod=14)
Learn more about the Aroon Oscillator at tadoc.org.

CMO - Chande Momentum Oscillator
real = CMO(close, timeperiod=14)
Learn more about the Chande Momentum Oscillator at tadoc.org.

MFI - Money Flow Index
real = MFI(high, low, close, volume, timeperiod=14)
Learn more about the Money Flow Index at tadoc.org.

STOCH - Stochastic
slowk, slowd = STOCH(high, low, close, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
Learn more about the Stochastic at tadoc.org.

"""