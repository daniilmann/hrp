# -*- encode: utf-8 -*-

import sys
from os.path import dirname, exists, join
from os import makedirs

import numpy as np
import pandas as pd
pd.core.common.is_list_like = pd.api.types.is_list_like

import yaml
from dateutil.relativedelta import relativedelta
import bt
from bt import algos
from hrp import WeightHRP
from ffn import fmtn, fmtp

import argparse
import re

from PyQt5 import QtWidgets as qtw
from app import HRPApp


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


def load_xlsx(fl_path, date_column=None, start_date=None, end_date=None, date_format='%Y-%m-%d'):

    date_column = date_column if date_column else 'Date'

    data = pd.read_excel(fl_path)
    data = data.set_index(pd.DatetimeIndex(pd.to_datetime(data[date_column], format=date_format)))
    data = data.drop(date_column, axis=1)

    data = data.sort_index()
    data = data.fillna(method='pad')

    idxs = None
    if start_date is not None:
        idxs = data.index >= start_date
    if end_date is not None:
        idxs = data.index <= end_date if idxs is None else (idxs) & (data.index <= end_date)
    if idxs is not None:
        data = data.loc[idxs]

    return data


def test(data_path, date_column, date_format, start_date, end_date, strategies):
    pass


def parse_config(conf_path):
    strategies = []
    with open(conf_path, 'r') as stream:
        try:
            config = yaml.load(stream)

            for s in config['strategies']:
                s = s.split(',')
                strategies.append(TestStrategy())
                strategies[-1].est_plen = int(s[0])
                strategies[-1].est_ptype = s[1]
                strategies[-1].roll_plen = int(s[2])
                strategies[-1].roll_ptype = s[3]
                if len(s) ==5:
                    strategies[-1].fee = float(s[4])

            config['strategies'] = strategies

            return config
        except yaml.YAMLError as exc:
            print(exc)


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


def args_parser():

    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--config', help='path to yaml config')
    parser.add_argument('-d', help='path to data file')
    parser.add_argument('-cn', help='date column name')
    parser.add_argument('-df', help='date format')
    parser.add_argument('-sd', help='start date in format yyyy-mm-dd')
    parser.add_argument('-ed', help='end date in format yyyy-mm-dd')
    parser.add_argument('-et', help='estimation period type')
    parser.add_argument('-el', help='estimation period len', type=int)
    parser.add_argument('-rt', help='roll period type')
    parser.add_argument('-rl', help='roll period len', type=int)
    parser.add_argument('-ic', help='intial capital; default 1 000 000.0', type=float)
    parser.add_argument('-f', help='fee in percents; default 0', type=float)
    parser.add_argument('-o', help='path to output folder')

    return parser


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


if __name__ == '__main__':

    # app = qtw.QApplication(sys.argv)
    # window = HRPApp()
    # window.show()
    # app.exec_()

    parser = args_parser()
    args = parser.parse_args()

    if args.config:
        config = parse_config(args.config)
    else:
        config = dict()

        data = dict()
        data['path'] = args.d
        data['column'] = args.d
        data['format'] = args.d
        config['data'] = data.copy()

        config['start_date'] = args.sd
        config['end_date'] = args.sd

        strat = TestStrategy()
        strat.est_plen = args.el
        strat.est_ptype = args.et
        strat.roll_plen = args.rl
        strat.roll_ptype = args.rt
        strat.fee = args.f
        config['strategies'] = [strat]

        config['capital'] = args.ic if args.ic else 1000000.0
        config['output'] = args.o


    data = load_xlsx(config['data']['path'], start_date=config.get('start_date'), end_date=config.get('end_date'))

    backtests = []
    for s in config['strategies']:
        backtests.append(bt_strategy(s, data, config['capital']))

    res = bt.run(*backtests)
    stats = make_stats(res)
    bdf = {b.name: pd.concat((b.weights, b.positions, b.turnover), axis=1) for b in backtests}
    pattern = re.compile('.*>')
    columns = ['W_' + pattern.sub('', c) for c in backtests[0].weights.columns]
    columns.extend(['POS_' + pattern.sub('', c) for c in backtests[0].positions.columns])
    columns.append('Turnover')

    config['output'] = config.get('output') if config.get('output') else dirname(__file__)
    if not exists(config['output']):
        makedirs(config['output'])

    writer = pd.ExcelWriter(join(config['output'], 'hrp_results.xlsx'))
    stats.to_excel(writer, 'Stats')
    res.prices.to_excel(writer, 'Prices')
    res.lookback_returns.applymap(fmtp).to_excel(writer, 'Lookback')

    for name, df in bdf.items():
        df.columns = columns
        df.to_excel(writer, name)

    writer.save()