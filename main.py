# -*- encode: utf-8 -*-

import numpy as np
import pandas as pd
pd.core.common.is_list_like = pd.api.types.is_list_like

import yaml
from dateutil.relativedelta import relativedelta
import bt
from bt import algos
from hrp import WeightHRP

import argparse


class TestStrategy(object):

    def __init__(self) -> None:
        super().__init__()

        self.est_plen = None
        self.est_ptype = None

        self.roll_plen = None
        self.roll_ptype = None

    def __str__(self) -> str:
        return 'Estimation {} {} | Roll {} {}'.format(self.est_plen, self.est_ptype, self.roll_plen, self.roll_ptype)

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

            config['strategies'] = strategies

            return config
        except yaml.YAMLError as exc:
            print(exc)


def bt_strategy(strat, data):

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

    return bt.Backtest(strategy, data.copy())


def args_parser():

    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--config', help='path to yaml config')
    parser.add_argument('-d', help='path to data file')
    parser.add_argument('-sd', help='start date')
    parser.add_argument('-ed', help='end date')
    parser.add_argument('-et', help='estimation period type')
    parser.add_argument('-el', help='estimation period len')
    parser.add_argument('-rt', help='roll period type')
    parser.add_argument('-rl', help='roll period len')
    parser.add_argument('-o', help='path to output folder')

    return parser


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
        pass

    exit()

    data = load_xlsx(config['data']['path'], start_date=config.get('start_date'), end_date=config.get('end_date'))

    backtests = []
    for s in config['strategies']:
        backtests.append(bt_strategy(s, data))

    res = bt.run(*backtests)
    res.display()