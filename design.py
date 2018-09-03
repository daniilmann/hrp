# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'hrp.ui'
#
# Created by: PyQt5 UI code generator 5.11.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_mainWindow(object):
    def setupUi(self, mainWindow):
        mainWindow.setObjectName("mainWindow")
        mainWindow.resize(333, 578)
        self.centralwidget = QtWidgets.QWidget(mainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.dataFileEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.dataFileEdit.setGeometry(QtCore.QRect(10, 40, 311, 20))
        self.dataFileEdit.setObjectName("dataFileEdit")
        self.dataFileLabel = QtWidgets.QLabel(self.centralwidget)
        self.dataFileLabel.setGeometry(QtCore.QRect(20, 20, 91, 16))
        self.dataFileLabel.setObjectName("dataFileLabel")
        self.dataFileButton = QtWidgets.QToolButton(self.centralwidget)
        self.dataFileButton.setGeometry(QtCore.QRect(240, 10, 71, 19))
        self.dataFileButton.setObjectName("dataFileButton")
        self.startDatePicker = QtWidgets.QDateEdit(self.centralwidget)
        self.startDatePicker.setGeometry(QtCore.QRect(40, 160, 91, 22))
        self.startDatePicker.setObjectName("startDatePicker")
        self.endDatePicker = QtWidgets.QDateEdit(self.centralwidget)
        self.endDatePicker.setGeometry(QtCore.QRect(210, 160, 91, 22))
        self.endDatePicker.setObjectName("endDatePicker")
        self.startDateLabel = QtWidgets.QLabel(self.centralwidget)
        self.startDateLabel.setGeometry(QtCore.QRect(50, 140, 61, 16))
        self.startDateLabel.setObjectName("startDateLabel")
        self.endDateLabel = QtWidgets.QLabel(self.centralwidget)
        self.endDateLabel.setGeometry(QtCore.QRect(220, 140, 47, 13))
        self.endDateLabel.setObjectName("endDateLabel")
        self.dateColumnLabel = QtWidgets.QLabel(self.centralwidget)
        self.dateColumnLabel.setGeometry(QtCore.QRect(20, 80, 151, 16))
        self.dateColumnLabel.setObjectName("dateColumnLabel")
        self.dateColumnEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.dateColumnEdit.setGeometry(QtCore.QRect(10, 100, 141, 20))
        self.dateColumnEdit.setObjectName("dateColumnEdit")
        self.dateFormatLabel = QtWidgets.QLabel(self.centralwidget)
        self.dateFormatLabel.setGeometry(QtCore.QRect(190, 80, 91, 16))
        self.dateFormatLabel.setObjectName("dateFormatLabel")
        self.dateFormatEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.dateFormatEdit.setGeometry(QtCore.QRect(180, 100, 141, 20))
        self.dateFormatEdit.setObjectName("dateFormatEdit")
        self.estPeriodLabel = QtWidgets.QLabel(self.centralwidget)
        self.estPeriodLabel.setGeometry(QtCore.QRect(20, 200, 101, 16))
        self.estPeriodLabel.setObjectName("estPeriodLabel")
        self.estTypeCombo = QtWidgets.QComboBox(self.centralwidget)
        self.estTypeCombo.setGeometry(QtCore.QRect(70, 220, 69, 22))
        self.estTypeCombo.setEditable(False)
        self.estTypeCombo.setCurrentText("")
        self.estTypeCombo.setObjectName("estTypeCombo")
        self.estPeriodSpin = QtWidgets.QSpinBox(self.centralwidget)
        self.estPeriodSpin.setGeometry(QtCore.QRect(10, 220, 51, 22))
        self.estPeriodSpin.setMinimum(1)
        self.estPeriodSpin.setObjectName("estPeriodSpin")
        self.rollPeriodSpin = QtWidgets.QSpinBox(self.centralwidget)
        self.rollPeriodSpin.setGeometry(QtCore.QRect(170, 220, 51, 22))
        self.rollPeriodSpin.setMinimum(1)
        self.rollPeriodSpin.setObjectName("rollPeriodSpin")
        self.rollTypeCombo = QtWidgets.QComboBox(self.centralwidget)
        self.rollTypeCombo.setGeometry(QtCore.QRect(230, 220, 69, 22))
        self.rollTypeCombo.setObjectName("rollTypeCombo")
        self.rollPeriodLabel = QtWidgets.QLabel(self.centralwidget)
        self.rollPeriodLabel.setGeometry(QtCore.QRect(180, 200, 81, 16))
        self.rollPeriodLabel.setObjectName("rollPeriodLabel")
        self.addStrategyBtn = QtWidgets.QPushButton(self.centralwidget)
        self.addStrategyBtn.setGeometry(QtCore.QRect(10, 260, 101, 23))
        self.addStrategyBtn.setObjectName("addStrategyBtn")
        self.removeStrategyBtn = QtWidgets.QPushButton(self.centralwidget)
        self.removeStrategyBtn.setGeometry(QtCore.QRect(130, 260, 101, 23))
        self.removeStrategyBtn.setObjectName("removeStrategyBtn")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(10, 540, 311, 23))
        self.pushButton.setObjectName("pushButton")
        self.reportFileLabel = QtWidgets.QLabel(self.centralwidget)
        self.reportFileLabel.setGeometry(QtCore.QRect(20, 490, 91, 16))
        self.reportFileLabel.setObjectName("reportFileLabel")
        self.reportFileEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.reportFileEdit.setGeometry(QtCore.QRect(10, 510, 311, 20))
        self.reportFileEdit.setObjectName("reportFileEdit")
        self.reportFileBtn = QtWidgets.QToolButton(self.centralwidget)
        self.reportFileBtn.setGeometry(QtCore.QRect(210, 480, 101, 20))
        self.reportFileBtn.setObjectName("reportFileBtn")
        self.startDateCBox = QtWidgets.QCheckBox(self.centralwidget)
        self.startDateCBox.setGeometry(QtCore.QRect(10, 160, 20, 20))
        self.startDateCBox.setText("")
        self.startDateCBox.setObjectName("startDateCBox")
        self.endDateCBox = QtWidgets.QCheckBox(self.centralwidget)
        self.endDateCBox.setGeometry(QtCore.QRect(180, 160, 16, 21))
        self.endDateCBox.setText("")
        self.endDateCBox.setObjectName("endDateCBox")
        self.strategyList = QtWidgets.QListView(self.centralwidget)
        self.strategyList.setGeometry(QtCore.QRect(10, 300, 311, 171))
        self.strategyList.setObjectName("strategyList")
        mainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(mainWindow)
        QtCore.QMetaObject.connectSlotsByName(mainWindow)
        mainWindow.setTabOrder(self.dataFileButton, self.dataFileEdit)
        mainWindow.setTabOrder(self.dataFileEdit, self.dateColumnEdit)
        mainWindow.setTabOrder(self.dateColumnEdit, self.dateFormatEdit)
        mainWindow.setTabOrder(self.dateFormatEdit, self.startDatePicker)
        mainWindow.setTabOrder(self.startDatePicker, self.endDatePicker)
        mainWindow.setTabOrder(self.endDatePicker, self.estPeriodSpin)
        mainWindow.setTabOrder(self.estPeriodSpin, self.estTypeCombo)
        mainWindow.setTabOrder(self.estTypeCombo, self.rollPeriodSpin)
        mainWindow.setTabOrder(self.rollPeriodSpin, self.rollTypeCombo)
        mainWindow.setTabOrder(self.rollTypeCombo, self.addStrategyBtn)
        mainWindow.setTabOrder(self.addStrategyBtn, self.removeStrategyBtn)
        mainWindow.setTabOrder(self.removeStrategyBtn, self.reportFileBtn)
        mainWindow.setTabOrder(self.reportFileBtn, self.reportFileEdit)
        mainWindow.setTabOrder(self.reportFileEdit, self.pushButton)

    def retranslateUi(self, mainWindow):
        _translate = QtCore.QCoreApplication.translate
        mainWindow.setWindowTitle(_translate("mainWindow", "HRP Tester"))
        self.dataFileLabel.setText(_translate("mainWindow", "Data File Path"))
        self.dataFileButton.setText(_translate("mainWindow", "Open File"))
        self.startDateLabel.setText(_translate("mainWindow", "Start Date"))
        self.endDateLabel.setText(_translate("mainWindow", "End Date"))
        self.dateColumnLabel.setText(_translate("mainWindow", "Column name with dates"))
        self.dateColumnEdit.setText(_translate("mainWindow", "Date"))
        self.dateFormatLabel.setText(_translate("mainWindow", "Date format"))
        self.dateFormatEdit.setText(_translate("mainWindow", "%Y-%m-%d"))
        self.estPeriodLabel.setText(_translate("mainWindow", "Estimation period"))
        self.rollPeriodLabel.setText(_translate("mainWindow", "Roll period"))
        self.addStrategyBtn.setText(_translate("mainWindow", "Add Strategy"))
        self.removeStrategyBtn.setText(_translate("mainWindow", "Remove Strategy"))
        self.pushButton.setText(_translate("mainWindow", "Test!"))
        self.reportFileLabel.setText(_translate("mainWindow", "Report File Path"))
        self.reportFileBtn.setText(_translate("mainWindow", "Report Folder"))

