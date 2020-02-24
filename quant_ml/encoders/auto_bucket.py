import pandas as pd

from quant_ml.encoders.interval_index_encoder import IntervalIndexEncoder
from quant_ml.util import arange_open


def estimate_option_chain(label_column: str,
                          s: pd.Series,
                          nr_of_buckets: int = 19,
                          strike_interval: float = 1.0) -> IntervalIndexEncoder:
    last = s[-1:]
    percentage = float(strike_interval / last)
    fact = round(nr_of_buckets / 2, 1)
    decimals = 3 if float(last) < 1000 else 2
    bucket_range = arange_open(percentage * -fact, percentage * fact, percentage, decimals)

    # return Interval Encoder
    return IntervalIndexEncoder(label_column,  pd.IntervalIndex.from_breaks(bucket_range))
