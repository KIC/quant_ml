from pandas.core.base import PandasObject
import matplotlib.dates as mdates


def matplot_dates(df: PandasObject):
    return mdates.date2num(df.index)