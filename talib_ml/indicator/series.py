import pandas as _pd
from typing import Union

# create convenient type hint
PANDAS = Union[_pd.DataFrame, _pd.Series]


def ta_sma(df: PANDAS, period=12):
    return df.rolling(period).mean()


def ta_ema(df: PANDAS, period=12):
    return df.ewm(span=period, adjust=False, min_periods=period-1).mean()


def ta_macd(df: PANDAS, fast_period=12, slow_period=26, signal_period=9):
    fast = ta_ema(df, fast_period)
    slow = ta_ema(df, slow_period)
    macd = (fast - slow)
    signal = ta_ema(macd, signal_period)
    hist = macd - signal

    for l, f in {"macd": macd, "signal": signal, "histogram": hist}.items():
        if isinstance(f, _pd.DataFrame) and len(df.columns) > 1:
            f.columns = _pd.MultiIndex.from_product([f.columns, [l]])
        else:
            f.name = l

    macd = macd.to_frame() if isinstance(macd, _pd.Series) else macd
    return macd.join(signal).join(hist)


def ta_mom(df: PANDAS, period=10):
    return df.diff(period)


def ta_roc(df: PANDAS, period=10):
    return df.pct_change(period)

#                                plus_di         = lambda df: talib.PLUS_DI(df["High"], df["Low"], df["Close"]) / 100,
#                                apo             = lambda df: talib.APO(df["Close"]) / 10,
#                                atr             = lambda df: talib.ATR(df["High"], df["Low"], df["Close"]) / 10,
#                                minus_dm        = lambda df: talib.MINUS_DM(df["High"], df["Low"]) / 100,
#                                trix            = lambda df: talib.TRIX(df["Close"]),
#                                plus_dm         = lambda df: talib.PLUS_DM(df["High"], df["Low"]) / 100,