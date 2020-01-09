from unittest import TestCase
import pandas_ml_utils as pmu
import talib_ml as tml
import pandas as pd
import os

# read testdata
from talib_ml.backtest.strategy import Strategy, Open, Close

df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "SPY.csv"), index_col='Date')
print(tml.__version__)


class TestStrategy(TestCase):

    def test_lala(self):
        """given"""
        backtest_df = pmu.LazyDataFrame(df[["Close"]],
                                        sma1=lambda df: df["Close"].rolling(20).mean(),
                                        sma2=lambda df: df["Close"].rolling(50).mean(),
                                        buy=lambda df: df.rolling(2).ta_cross_over("sma1", "sma2"),
                                        sell=lambda df: df.rolling(2).ta_cross_under("sma1", "sma2"))

        def signal(row):
            if row["buy"]:
                return Open()
            elif row["sell"]:
                return Close()

            return None

        """when"""
        strategy = Strategy(backtest_df, signal)

        """then"""
        print()
        print(strategy.backtest())
        self.assertEqual(strategy.backtest(), None)