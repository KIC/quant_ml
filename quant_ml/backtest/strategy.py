from typing import Callable

import pandas as pd
import numpy as np

from quant_ml.aggregate.aggregator import max_draw_down, profit_and_loss, max_peak, range_label, pct_std


class Signal(object):
    pass


class Open(Signal):

    def __init__(self, quantity=1, stop_loss=None):
        self.quantity = quantity


class Close(Signal):
    pass


# other signals may be Swing,


class Position(object):

    def __init__(self, signal: Callable[[pd.DataFrame], Signal]):
        self.signal = signal
        self.trade_count = 0
        self.position_id = np.nan
        self.quantity = 0

    def __call__(self, *args, **kwargs):
        signal = self.signal(*args)
        current_pos_id = self.position_id
        current_qty = self.quantity

        if isinstance(signal, Signal):
            if isinstance(signal, Open):
                if np.isnan(self.position_id):
                    self.trade_count += 1
                    self.position_id = self.trade_count
                    self.quantity += signal.quantity
                    current_pos_id = self.position_id
                    current_qty = self.quantity
            if isinstance(signal, Close):
                self.quantity = 0
                self.position_id = np.nan

        # in case we missed something
        return current_pos_id, current_qty


class Strategy(object):

    def __init__(self, df: pd.DataFrame, signal: Callable[[pd.DataFrame], Signal]):
        self.df = df
        self.signal = Position(signal)

    def backtest(self, price_column) -> pd.DataFrame:
        trades = self.df.apply(self.signal, axis=1, raw=False, result_type='expand')
        trades.columns = ["position_id", "quantity"]
        data = self.df.join(trades).dropna()
        data["date"] = data.index

        return data.groupby("position_id").agg({
            "date": range_label,
            price_column: ["count", max_draw_down, profit_and_loss, max_peak, pct_std]
        })
