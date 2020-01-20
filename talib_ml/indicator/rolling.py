import pandas as _pd




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

