# -*- encode: utf-8 -*-

from PyQt5 import QtWidgets as qtw, QtCore as qtc, QtGui as qtgui
import design


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
        self.addStrategyBtn.clicked.connect(self.add_strategy)
        self.removeStrategyBtn.clicked.connect(self.remoe_strategy)
        self.reportFileBtn.clicked.connect(self.select_report_dir)

    def select_data_path(self):
        dataPath = qtw.QFileDialog.getOpenFileUrl(self, caption='Select Data File')[0]

        if not dataPath.isEmpty():
            self.dataFileEdit.setText(dataPath.path()[1:])

    def select_report_dir(self):
        dataPath = qtw.QFileDialog.getExistingDirectoryUrl(self, caption='Select directory for report')

        if not dataPath.isEmpty():
            self.reportFileEdit.setText(dataPath.path()[1:])

    def add_strategy(self):
        strategy = TestStrategy()

        strategy.est_plen = self.estPeriodSpin.value()
        strategy.est_ptype = self.estTypeCombo.currentText()
        strategy.roll_plen = self.estPeriodSpin.value()
        strategy.roll_ptype = self.rollTypeCombo.currentText()

        if strategy not in self._strategies:
            self._strategies.append(strategy)
            self.strategyListModel.appendRow(qtgui.QStandardItem(str(strategy)))

    def remoe_strategy(self):

        if self._strategies:
            idx = self.strategyList.selectedIndexes()[0].row()
            del self._strategies[idx]
            self.strategyListModel.removeRow(idx)
