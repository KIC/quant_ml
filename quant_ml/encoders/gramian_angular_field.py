import operator
from typing import List, Any, Callable
from pyts.image import GramianAngularField
import numpy as np
import pandas as pd

import quant_ml as qml
from pandas_ml_utils.model.features_and_labels.target_encoder import TargetLabelEncoder

qml.__version__


class GramianAngularField(TargetLabelEncoder):

    def __init__(self,
                 source_columns,
                 image_size=1.,
                 sample_range=(-1, 1),
                 method='summation',
                 overlapping=False,
                 flatten=False):
        super().__init__()
        self.labels_source = source_columns if isinstance(source_columns, List) else [source_columns]
        self.encoded_labels = ["+".join(self.labels_source_columns)]
        self.gaf = GramianAngularField(image_size=image_size, sample_range=sample_range, method=method,
                                       overlapping=overlapping, flatten=flatten)

    @property
    def labels_source_columns(self) -> List[str]:
        return self.labels_source

    @property
    def encoded_labels_columns(self) -> List[str]:
        return self.encoded_labels

    def encode(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return super().encode(df, **kwargs)

    def decode(self, df: pd.DataFrame) -> pd.DataFrame:
        raise ValueError("decoding not supported :-(")

