import pandas as _pd




def ta_future_pct_of_mean(df: _pd.DataFrame, lag: int = 1):
    # (price / mean[t-x]) - 1
    most_recent = df.apply(lambda x: x[-1]).shift(-lag)
    mean = df.mean()

    return (most_recent / mean) - 1


