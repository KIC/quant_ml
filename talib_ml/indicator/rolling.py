import pandas as _pd


def ta_bbands(df: _pd.DataFrame, stddev: float = 2.0):
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


def ta_future_pct_of_mean(df: _pd.DataFrame, lag: int = 1):
    # (price / mean[t-x]) - 1
    most_recent = df.apply(lambda x: x[-1]).shift(-lag)
    mean = df.mean()

    return (most_recent / mean) - 1


def ta_cross_over(df: _pd.DataFrame, a, b):
    old = df.apply(lambda x: x[0])
    young = df.apply(lambda x: x[-1])
    return (old[a] <= old[b]) & (young[a] > young[b])


def ta_cross_under(df: _pd.DataFrame, a, b):
    return ta_cross_over(df, b, a)

