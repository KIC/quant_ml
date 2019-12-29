import math
import operator
from typing import List, Any, Callable

import numpy as np
import pandas as pd

import talib_ml as tml
from pandas_ml_utils.model.features_and_labels.target_encoder import TargetLabelEncoder

tml.__version__


class IntervalIndexEncoder(TargetLabelEncoder):

    def __init__(self,
                 label_column: str,
                 buckets: pd.IntervalIndex,
                 target_operator:Callable[[Any, Any], Any] = lambda a, b: operator.mul(a + 1, b)):
        self.label_column = label_column
        self.buckets = buckets
        self.target_operator = target_operator

    @property
    def labels_source_columns(self) -> List[str]:
        return [self.label_column]

    @property
    def encoded_labels_columns(self) -> List[str]:
        return [f"{self.label_column}, {b}" for b in self.buckets]

    def encode(self, df: pd.DataFrame) -> pd.DataFrame:
        cat = df[self.label_column].ta_bucketize(self.buckets)
        one_hot_categories = cat.ta_one_hot_categories()
        one_hot_categories.columns = one_hot_categories.columns.to_flat_index()

        return one_hot_categories

    def calculate_targets(self, s: pd.Series):
        # define a function how to unroll the classes back into its borders
        def unfold(value: np.ndarray):
            value = np.asscalar(value)
            result = []

            for i, b in enumerate(self.buckets):
                left_border = b.left
                result.append(self.target_operator(left_border, value))
                if i + 1 >= len(self.buckets):
                    right_border = b.right
                    result.append(self.target_operator(right_border, value))

            return result

        # return a new data frame with all targets such that we can plot it as a stacked bar chart
        frame = s.to_frame().apply(unfold, axis=1, result_type='expand')

        # finally fix eventual inf columns
        raw = frame.values[np.isfinite(frame.values)]
        return frame.replace([-np.inf, np.inf], [raw.min(), raw.max()])

    def decode(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.ta_one_hot_to_categories([self.buckets])

    def __len__(self):
        return len(self.buckets)