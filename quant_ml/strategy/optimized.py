import logging
import traceback

import numpy as np
import pandas as pd
from qpsolvers import solve_qp
from statsmodels.stats.correlation_tools import cov_nearest


def ta_markowitz(df: pd.DataFrame,
                 period=90,
                 risk_aversion=5,
                 prices='Close',
                 expected_returns=None,
                 rebalance_trigger=None,
                 solver='cvxopt'):
    assert isinstance(df.columns, pd.MultiIndex), \
        "expect multi index columns 'prices', 'expected returns' and rebalance trigger"
    assert period > 1, "we need a positiv period"

    # calculate log returns
    log_returns = np.log(df[prices]) - np.log(df[prices].shift(1))
    log_returns.columns = pd.MultiIndex.from_tuples([('returns', col) for col in log_returns.columns])

    # if expected returns is None simply use the moving average of the returns
    expected = df[prices].pct_change() if expected_returns is None else df[expected_returns]
    expected.columns = pd.MultiIndex.from_tuples([('expected', col) for col in expected.columns])

    # if re-balance trigger is none re-balance every row
    trigger = pd.DataFrame(np.ones(len(df)), index=df.index) if rebalance_trigger is None else df[rebalance_trigger]
    trigger.columns = pd.MultiIndex.from_tuples([('trigger', col) for col in trigger.columns])

    # non negative weight constraint and weights sum to 1
    h = np.zeros(len(log_returns.columns)).reshape((-1, 1))
    G = -np.eye(len(h))
    A = np.ones(len(h)).reshape((1, -1))
    b = np.ones(1)

    # no solution
    keep_solution = (np.empty(len(h)) * np.nan)
    uninvest = np.zeros(len(h))

    # define optimization function
    def optimize_portfolios(row):
        # only optimize if we have a re-balance trigger
        if row['trigger'][-1:].sum(axis=1).any() < 1:
            return keep_solution

        # calculate covariance matrix # TODO use ewma covariance
        # r = row['returns'].values
        # cov = r.T @ r * (1 / period)
        cov = row['returns'].cov().values

        # make sure covariance matrix is positive definite
        cov = cov_nearest(cov)

        # calculate expected returns
        er = (row['expected'].mean() if expected_returns is None else row['expected'][-1:]).values

        # we perform an un-constraint optimization and then rescale the weights back
        if len(er[er < 0]) == len(er):
            return uninvest
        else:
            try:
                sol = solve_qp(risk_aversion * cov, -er, G=G, h=h, A=A, b=b, solver=solver)
                if sol is None:
                    logging.error("no solution found")
                    return uninvest
                else:
                    return sol
            except Exception as e:
                logging.error(traceback.format_exc())
                return uninvest

    # create a unified data frame
    data = log_returns.join(expected).join(trigger).dropna()

    # optimize portfolios and fill nan weights with previous weights
    weights = [optimize_portfolios(data.iloc[i-period:i]) for i in range(period, len(data))]
    weights = np.stack(weights, axis=0)
    weights = pd.DataFrame(weights, index=data.index[period:], columns=log_returns.columns)
    return weights.fillna(method="ffill")
