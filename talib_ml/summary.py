import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mlxtend.evaluate import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix

from pandas_ml_utils.constants import *
from pandas_ml_utils.summary.summary import Summary


class TradingSummary(Summary):

    # TODO implement a nice trading summary like max draw down etc. we might use zipline or similars
    #  i.e. http://pmorissette.github.io/bt/

    def __init__(self, df: pd.DataFrame, **kwargs):
        super().__init__(df, **kwargs)

    def plot_ts_heatmap(self, loss_alpha=0.1, loss_color="white", loss_width=0.25, figsize=(34, 8)) -> plt.Figure:
        fig = plt.figure(figsize=figsize)
        plt.pcolormesh(self.df.index, self.df[PREDICTION_COLUMN_NAME].columns, self.df[PREDICTION_COLUMN_NAME].T)
        if LOSS_COLUMN_NAME in self.df.columns:
            ax2 = self.df[LOSS_COLUMN_NAME].plot(color=loss_color, alpha=loss_alpha, linewidth=loss_width,
                                                 secondary_y=True)
            # fixme find range from targets ..
            ax2.set_ylim(-0.09, 0.09)

        plt.colorbar()
        plt.close()
        return fig

    def plot_confusion_matrix(self, figsize=(12, 12)) -> plt.Figure:
        y = self.df[LABEL_COLUMN_NAME].apply(lambda row: np.argmax(row), raw=True, axis=1)
        y_hat = self.df[PREDICTION_COLUMN_NAME].apply(lambda row: np.argmax(row), raw=True, axis=1)

        cm = confusion_matrix(y.values, y_hat.values)
        return plot_confusion_matrix(cm, figsize=figsize)[0]

