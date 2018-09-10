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
from strategy import TestStrategy


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
                    strategies[-1].prc_fee = float(s[4])

            config['strategies'] = strategies

            return config
        except yaml.YAMLError as exc:
            print(exc)


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


if __name__ == '__main__':

    app = qtw.QApplication(sys.argv)
    window = HRPApp()
    window.show()
    app.exec_()

    # parser = args_parser()
    # args = parser.parse_args()
    #
    # if args.config:
    #     config = parse_config(args.config)
    # else:
    #     config = dict()
    #
    #     data = dict()
    #     data['path'] = args.d
    #     data['column'] = args.d
    #     data['format'] = args.d
    #     config['data'] = data.copy()
    #
    #     config['start_date'] = args.sd
    #     config['end_date'] = args.sd
    #
    #     strat = TestStrategy()
    #     strat.est_plen = args.el
    #     strat.est_ptype = args.et
    #     strat.roll_plen = args.rl
    #     strat.roll_ptype = args.rt
    #     strat.fee = args.f
    #     config['strategies'] = [strat]
    #
    #     config['capital'] = args.ic if args.ic else 1000000.0
    #     config['output'] = args.o
    #
    #
    # data = load_xlsx(config['data']['path'], start_date=config.get('start_date'), end_date=config.get('end_date'))
    #
    # backtests = []
    # for s in config['strategies']:
    #     backtests.append(bt_strategy(s, data, config['capital']))
    #
    # res = bt.run(*backtests)
    # stats = make_stats(res)
    # bdf = {b.name: pd.concat((b.weights, b.positions, b.turnover), axis=1) for b in backtests}
    # pattern = re.compile('.*>')
    # columns = ['W_' + pattern.sub('', c) for c in backtests[0].weights.columns]
    # columns.extend(['POS_' + pattern.sub('', c) for c in backtests[0].positions.columns])
    # columns.append('Turnover')
    #
    # config['output'] = config.get('output') if config.get('output') else dirname(__file__)
    # if not exists(config['output']):
    #     makedirs(config['output'])