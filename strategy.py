# -*- encode: utf-8 -*-

from dateutil.relativedelta import relativedelta
from functools import partial

import numpy as np
import pandas as pd

import bt
from bt import algos
from ffn import fmtp, fmtn
from hrp import WeightHRP, SaveWeights, GapWeights

class TestStrategy(object):

    def __init__(self) -> None:
        super().__init__()

        self.init_balance = 0.0

        self.prc_fee = 0.0
        self.fix_fee = 0.0

        self.reb_gap = 0.0
        self.weight_round = 0
        self.robust = False

        self.est_plen = 0
        self.est_ptype = 'n'

        self.roll_plen = 0
        self.roll_ptype = 'n'

    def __str__(self) -> str:
        return 'Estimation {:5} {:5} - Roll {:5} {:5} | Fee {:.4f} + {:.4f} | Gap {:.4f} | Cov {:6} | Balance {:.2f}'.format(
            self.est_plen, self.est_ptype,
            self.roll_plen, self.roll_ptype,
            self.fix_fee, self.prc_fee,
            self.reb_gap, 'OAS' if self.robust else 'Simple',
            self.init_balance)
        # return 'Estimation {} {} | Roll {} {}'.format(self.est_plen, self.est_ptype, self.roll_plen, self.roll_ptype)

    def name(self):
        return 'e{}{}r{}{}ff{}pf{}g{}c{}b{}'.format(
            self.est_plen, self.est_ptype[0],
            self.roll_plen, self.roll_ptype[0],
            self.fix_fee, self.prc_fee,
            self.reb_gap, 'R' if self.robust else 'S',
            int(self.init_balance))

    def __eq__(self, o: object) -> bool:
        if isinstance(o, TestStrategy):
            return self.est_plen == o.est_plen \
                   and self.est_ptype == o.est_ptype \
                   and self.roll_plen == o.roll_plen \
                   and self.roll_ptype == o.roll_ptype \
                   and self.init_balance == o.init_balance \
                   and self.reb_gap == o.reb_gap \
                   and self.prc_fee == o.prc_fee \
                   and self.fix_fee == o.fix_fee \
                   and self.robust == o.robust
        return False

    def __hash__(self) -> int:
        hsh = super().__hash__()
        p = 31
        hsh = p * hsh + hash(self.est_plen)
        hsh = p * hsh + hash(self.est_ptype)
        hsh = p * hsh + hash(self.roll_plen)
        hsh = p * hsh + hash(self.roll_ptype)
        hsh = p * hsh + hash(self.init_balance)
        hsh = p * hsh + hash(self.reb_gap)
        hsh = p * hsh + hash(self.prc_fee)
        hsh = p * hsh + hash(self.fix_fee)
        hsh = p * hsh + hash(self.robust)
        return hsh


    def bt_strategy(self, data):

        rsmpl = {
            'days': 'B',
            'weeks': 'W-Fri',
            'months': 'BM',
            'years': 'BY'
        }[self.roll_ptype]

        first_date = data.index[0] + relativedelta(**{self.est_ptype: self.est_plen})
        run_dates = data.resample(rsmpl).last()
        run_dates = run_dates.loc[run_dates.index > first_date]
        run_dates = run_dates.iloc[:-1]
        run_dates.loc[data.index[-1]] = data.iloc[-1]

        # algo_stack = [
        #     algos.RunOnDate(*run_dates.index.tolist()),
        #     algos.SelectAll()
        # ]
        # if self.reb_gap != 0.0:
        #     algo_stack.extend([
        #         SaveWeights(),
        #         WeightHRP(plen=self.est_plen, ptype=self.est_ptype, robust=self.robust),
        #         GapWeights(self.reb_gap)
        #     ])
        # else:
        #     algo_stack.append(WeightHRP(plen=self.est_plen, ptype=self.est_ptype, robust=self.robust))
        # algo_stack.append(algos.Rebalance())

        strategy = bt.Strategy(self.name(), [
            algos.RunOnDate(*run_dates.index.tolist()),
            algos.SelectAll(),
            SaveWeights(),
            WeightHRP(plen=self.est_plen, ptype=self.est_ptype, robust=self.robust),
            GapWeights(self.reb_gap),
            algos.Rebalance()
        ])

        fee_func = partial(lambda q, p, ff, pf: np.max([ff, pf*p*q]), ff=self.fix_fee, pf=self.prc_fee)

        return bt.Backtest(strategy, data.copy(), initial_capital=self.init_balance,
                           commissions=fee_func)


def make_stats(res):
    names = res.backtests.keys()

    stats = [('start', 'Start', 'dt'),
             ('end', 'End', 'dt'),
             ('rf', 'Risk-free rate', 'p'),
             (None, None, None),
             ('total_return', 'Total Return', 'p'),
             ('daily_sharpe', 'Daily Sharpe', 'n'),
             ('daily_sortino', 'Daily Sortino', 'n'),
             ('cagr', 'CAGR', 'p'),
             ('max_drawdown', 'Max Drawdown', 'p'),
             ('calmar', 'Calmar Ratio', 'n'),
             (None, None, None),
             ('mtd', 'MTD', 'p'),
             ('three_month', '3m', 'p'),
             ('six_month', '6m', 'p'),
             ('ytd', 'YTD', 'p'),
             ('one_year', '1Y', 'p'),
             ('three_year', '3Y (ann.)', 'p'),
             ('five_year', '5Y (ann.)', 'p'),
             ('ten_year', '10Y (ann.)', 'p'),
             ('incep', 'Since Incep. (ann.)', 'p'),
             (None, None, None),
             ('daily_sharpe', 'Daily Sharpe', 'n'),
             ('daily_sortino', 'Daily Sortino', 'n'),
             ('daily_mean', 'Daily Mean (ann.)', 'p'),
             ('daily_vol', 'Daily Vol (ann.)', 'p'),
             ('daily_skew', 'Daily Skew', 'n'),
             ('daily_kurt', 'Daily Kurt', 'n'),
             ('best_day', 'Best Day', 'p'),
             ('worst_day', 'Worst Day', 'p'),
             (None, None, None),
             ('monthly_sharpe', 'Monthly Sharpe', 'n'),
             ('monthly_sortino', 'Monthly Sortino', 'n'),
             ('monthly_mean', 'Monthly Mean (ann.)', 'p'),
             ('monthly_vol', 'Monthly Vol (ann.)', 'p'),
             ('monthly_skew', 'Monthly Skew', 'n'),
             ('monthly_kurt', 'Monthly Kurt', 'n'),
             ('best_month', 'Best Month', 'p'),
             ('worst_month', 'Worst Month', 'p'),
             (None, None, None),
             ('yearly_sharpe', 'Yearly Sharpe', 'n'),
             ('yearly_sortino', 'Yearly Sortino', 'n'),
             ('yearly_mean', 'Yearly Mean', 'p'),
             ('yearly_vol', 'Yearly Vol', 'p'),
             ('yearly_skew', 'Yearly Skew', 'n'),
             ('yearly_kurt', 'Yearly Kurt', 'n'),
             ('best_year', 'Best Year', 'p'),
             ('worst_year', 'Worst Year', 'p'),
             (None, None, None),
             ('avg_drawdown', 'Avg. Drawdown', 'p'),
             ('avg_drawdown_days', 'Avg. Drawdown Days', 'n'),
             ('avg_up_month', 'Avg. Up Month', 'p'),
             ('avg_down_month', 'Avg. Down Month', 'p'),
             ('win_year_perc', 'Win Year %', 'p'),
             ('twelve_month_win_perc', 'Win 12m %', 'p')]

    data = pd.DataFrame(columns=names)
    empty = pd.Series(np.full(len(names), ''), index=names)
    for stat in stats:
        k, n, f = stat
        # blank row
        if k is None:
            data.loc[str(len(data))] = empty
            continue

        raw = res.stats.loc[k]
        if f is None:
            raw = empty
        elif f == 'p':
            raw = raw.map(fmtp)
        elif f == 'n':
            raw = raw.map(fmtn)
        elif f == 'dt':
            raw = raw.map(lambda val: val.strftime('%Y-%m-%d'))
        else:
            raise NotImplementedError('unsupported format %s' % f)

        data.loc[n] = raw.values

    return data