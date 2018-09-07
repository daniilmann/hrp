# -*- encode: utf-8 -*-

import bt
from bt import algos
from hrp import WeightHRP


class TestStrategy(object):

    def __init__(self) -> None:
        super().__init__()

        self.est_plen = None
        self.est_ptype = 'n'

        self.roll_plen = None
        self.roll_ptype = 'n'

        self.fee = None

    def __str__(self) -> str:
        return 'e{}{}-r{}{}'.format(self.est_plen, self.est_ptype[0], self.roll_plen, self.roll_ptype[0])
        # return 'Estimation {} {} | Roll {} {}'.format(self.est_plen, self.est_ptype, self.roll_plen, self.roll_ptype)

    def __eq__(self, o: object) -> bool:
        if isinstance(o, TestStrategy):
            return self.est_plen == o.est_plen and self.est_ptype == o.est_ptype and self.roll_plen == o.roll_plen and self.roll_ptype == o.roll_ptype
        return False

    def __hash__(self) -> int:
        hsh = super().__hash__()
        p = 31
        hsh = p * hsh + hash(self.est_plen)
        hsh = p * hsh + hash(self.est_ptype)
        hsh = p * hsh + hash(self.roll_plen)
        hsh = p * hsh + hash(self.roll_ptype)
        return hsh

    @property
    def name(self):
        try:
            return '{}{}{}{}'.format(self.est_plen, self.est_ptype[0], self.roll_plen, self.roll_ptype[0])
        except:
            return None


def bt_strategy(strat, data, capital):

    rsmpl = {
        'days': 'B',
        'weeks': 'W-Fri',
        'months': 'BM',
        'years': 'BY'
    }[strat.roll_ptype]

    first_date = data.index[0] + relativedelta(**{strat.est_ptype: strat.est_plen})
    run_dates = data.resample(rsmpl).last()
    run_dates = run_dates.loc[run_dates.index > first_date]
    run_dates = run_dates.iloc[:-1]
    run_dates.loc[data.index[-1]] = data.iloc[-1]

    strategy = bt.Strategy(str(strat), [
        algos.RunOnDate(*run_dates.index.tolist()),
        algos.SelectAll(),
        WeightHRP(plen=strat.est_plen, ptype=strat.est_ptype),
        algos.Rebalance()
    ])

    return bt.Backtest(strategy, data.copy(), initial_capital=capital)


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