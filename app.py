# -*- encode: utf-8 -*-

from PyQt5 import QtWidgets as qtw, QtCore as qtc, QtGui as qtgui
import design

from strategy import TestStrategy, make_stats
import bt
from ffn import fmtp, PerformanceStats

import pandas as pd
import numpy as np
import re
import shutil
from os import makedirs
from os.path import dirname, exists, join
import sys
import traceback
from matplotlib import colors
from dateutil.relativedelta import relativedelta


class ERPeriod(object):

    def __init__(self, plen, ptype):

        self.plen = plen
        self.ptype = ptype


class HRPApp(qtw.QMainWindow, design.Ui_mainWindow):

    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.startDatePicker.setEnabled(False)
        self.endDatePicker.setEnabled(False)

        self.startDatePicker.setDate(qtc.QDate.currentDate())
        self.endDatePicker.setDate(qtc.QDate.currentDate())

        self.startDateCBox.stateChanged.connect(lambda: self.startDatePicker.setEnabled(self.startDateCBox.checkState()))
        self.endDateCBox.stateChanged.connect(lambda: self.endDatePicker.setEnabled(self.endDateCBox.checkState()))

        self.estPeriodSpin.setMinimum(1)
        self.rollPeriodSpin.setMinimum(1)

        ptypes = ['days', 'weeks', 'months', 'years']
        self.estTypeCombo.addItems(ptypes)
        self.estTypeCombo.setCurrentIndex(0)
        self.rollTypeCombo.addItems(ptypes)
        self.rollTypeCombo.setCurrentIndex(0)

        self._strategies = []
        self.strategyListModel = qtgui.QStandardItemModel()
        self.strategyList.setModel(self.strategyListModel)

        self.dataFileButton.clicked.connect(self.select_data_path)
        self.rateFileButton.clicked.connect(self.select_rate_path)
        self.addStrategyBtn.clicked.connect(self.add_strategy)
        self.removeStrategyBtn.clicked.connect(self.remove_strategy)
        self.reportFileBtn.clicked.connect(self.select_report_dir)
        self.tvolCheck.stateChanged.connect(lambda: self.tvolSpin.setEnabled(self.tvolCheck.checkState()))

        self.runTestButton.clicked.connect(self.run_test)

    @property
    def data_path(self):
        return self.dataFileEdit.text()

    @property
    def initial_balance(self):
        return self.moneyBox.value()

    @property
    def fix_fee(self):
        return self.fixfeeSpin.value()

    @property
    def prc_fee(self):
        return np.round(self.prcfeeSpin.value() / 100.0, 6)

    @property
    def est_period(self):
        return ERPeriod(self.estPeriodSpin.value(), self.estTypeCombo.currentText())

    @property
    def roll_period(self):
        return ERPeriod(self.rollPeriodSpin.value(), self.rollTypeCombo.currentText())

    def select_data_path(self):
        dataPath = qtw.QFileDialog.getOpenFileUrl(self, caption='Select Data File')[0]

        if not dataPath.isEmpty():
            self.dataFileEdit.setText(dataPath.path()[1:])

    def select_rate_path(self):
        dataPath = qtw.QFileDialog.getOpenFileUrl(self, caption='Select Rate File')[0]

        if not dataPath.isEmpty():
            self.rateFileEdit.setText(dataPath.path()[1:])

    def select_report_dir(self):
        dataPath = qtw.QFileDialog.getExistingDirectoryUrl(self, caption='Select directory for report')

        if not dataPath.isEmpty():
            self.reportFileEdit.setText(dataPath.path()[1:])

    def add_strategy(self):
        strategy = TestStrategy()

        strategy.init_balance = self.initial_balance
        strategy.leverage = self.leverageSpin.value()
        strategy.is_tvol = self.tvolCheck.isChecked()
        strategy.tvol = np.round(self.tvolSpin.value() / 100, 6)
        strategy.fix_fee = self.fix_fee
        strategy.prc_fee = self.prc_fee
        strategy.reb_gap = np.round(self.rebgapSpin.value() / 100, 6)
        strategy.ddev = self.ddCB.isChecked()
        strategy.weight_round = int(self.decimalSpin.value())
        if self.rateFileEdit.text() and exists(self.rateFileEdit.text()):
            strategy.risk_free = self.rateFileEdit.text()
        else:
            strategy.risk_free = np.round(self.rateSpin.value() / 100, 6)
        strategy.robust = self.covCheckBox.isChecked()
        strategy.int_pos = self.integerCheckBox.isChecked()
        strategy.est_plen = self.estPeriodSpin.value()
        strategy.est_ptype = self.estTypeCombo.currentText()
        strategy.roll_plen = self.rollPeriodSpin.value()
        strategy.roll_ptype = self.rollTypeCombo.currentText()

        strategy.min_weight = np.round(self.minWeightSpin.value() / 100, strategy.weight_round)
        strategy.max_weight = np.round(self.maxWeightSpin.value() / 100, strategy.weight_round)

        if strategy not in self._strategies:
            self._strategies.append(strategy)
            self.strategyListModel.appendRow(qtgui.QStandardItem(str(strategy)))

    def remove_strategy(self):

        if len(self.strategyList.selectedIndexes()) and self._strategies:
            idx = self.strategyList.selectedIndexes()[0].row()
            del self._strategies[idx]
            self.strategyListModel.removeRow(idx)

    def run_test(self):

        try:
            data = None
            if self.dataFileEdit.text():

                try:
                    cats = pd.read_excel(self.dataFileEdit.text(), sheet_name=1)

                    sc = np.unique(cats.loc['Strategy'])
                    if len(sc) <= len(colors.BASE_COLORS):
                        scc = list(colors.BASE_COLORS.keys())[:len(sc)]
                    else:
                        scc = list(colors.CSS4_COLORS.keys())[:len(sc)]
                    scd = {s: c for s, c in zip(sc, scc)}
                    cats.loc['Colors'] = pd.Series({i: scd[cats.loc['Strategy'][i]] for i in cats.loc['Strategy'].index})

                    am = np.unique(cats.loc['Asset'])
                    ams = list('so^>v<dph8')[:len(am)]
                    amd = {a: s for a, s in zip(am, ams)}
                    cats.loc['Shapes'] = pd.Series(
                        {i: amd[cats.loc['Asset'][i]] for i in cats.loc['Asset'].index})
                except:
                    cats = None

                date_column = self.dateColumnEdit.text() if self.dateColumnEdit.text() else 'Date'
                date_format = self.dateFormatEdit.text() if self.dateFormatEdit.text() else '%Y-%m-%d'

                data = pd.read_excel(self.dataFileEdit.text())
                data = data.set_index(pd.DatetimeIndex(pd.to_datetime(data[date_column], format=date_format)))
                data = data.sort_index()
                data = data.drop(date_column, axis=1)

                data = data.sort_index()
                data = data.fillna(method='pad')

                idxs = None
                if self.startDateCBox.isChecked():
                    idxs = data.index >= self.startDatePicker.text()
                if self.endDateCBox.isChecked():
                    idxs = data.index <= self.endDatePicker.text() if idxs is None else (idxs) & (data.index <= self.endDatePicker.text())
                if idxs is not None:
                    data = data.loc[idxs]

            if len(self._strategies) and data is not None:
                out_dir = self.reportFileEdit.text() if self.reportFileEdit.text() else dirname(__file__)
                if not exists(out_dir):
                    makedirs(out_dir)

                backtests = []
                rfs = dict()
                for s in self._strategies:
                    if s.min_weight * data.shape[1] > s.leverage:
                        raise Exception('Sum of min weights ({:.4f}) more than leverage ({:.4f})\n{}'
                                        .format(s.min_weight, s.leverage, str(s)))
                    graph_dir = join(out_dir, s.name())
                    if exists(graph_dir):
                        shutil.rmtree(graph_dir)
                    makedirs(graph_dir)
                    backtests.append(s.bt_strategy(data, cats, graph_dir))
                    rfs[s.name()] = s.risk_free
                res = bt.run(*backtests)
                perfs = dict()
                for s in self._strategies:
                    k, v = s.name(), rfs[s.name()]
                    fdate = data.index[0] + relativedelta(**{s.est_ptype: s.est_plen})
                    prices = res.backtests[k].strategy.prices
                    if isinstance(v, str):
                        v = pd.read_excel(v)
                        v = v.set_index(pd.DatetimeIndex(pd.to_datetime(v[date_column], format=date_format)))
                        v = v.sort_index()
                        column = v.columns[1]
                        v['Price'] = prices
                        v = v.fillna(method='pad').fillna(0.0)[column]
                    else:
                        v = float(v)
                    perfs[k] = PerformanceStats(prices.loc[prices.index >= fdate], v)
                stats = make_stats(perfs)
                bdf = {b.name: pd.concat((b.strategy.data, b.weights, b.positions, b.turnover), axis=1) for b in backtests}
                pattern = re.compile('.*>')
                columns = backtests[0].strategy.data.columns.tolist()
                columns.extend(['W_' + pattern.sub('', c) for c in backtests[0].weights.columns])
                columns.extend(['POS_' + pattern.sub('', c) for c in backtests[0].positions.columns])
                columns.append('Turnover')

                writer = pd.ExcelWriter(join(out_dir, 'hrp_results.xlsx'))
                stats.to_excel(writer, 'Stats')
                res.prices.to_excel(writer, 'Prices')
                res.lookback_returns.applymap(fmtp).to_excel(writer, 'Lookback')
                for name, df in bdf.items():
                    df.columns = columns
                    df.to_excel(writer, name)

                writer.save()
                qtw.QMessageBox.information(self, "I'm ready", "I'm ready")
            else:
                qtw.QMessageBox.critical(self, "No strategies", "Add strategies")
        except:
            qtw.QMessageBox.critical(self, 'Something wrong T_T', str(sys.exc_info()))
            traceback.print_tb(sys.exc_info()[2])