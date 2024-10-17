# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'main_pagesVvzoPn.ui'
##
## Created by: Qt User Interface Compiler version 6.4.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QCheckBox, QFrame, QGridLayout,
    QHBoxLayout, QLabel, QLayout, QLineEdit,
    QPushButton, QRadioButton, QSizePolicy, QSlider,
    QSpacerItem, QStackedWidget, QVBoxLayout, QWidget)

class Ui_MainPages(object):
    def setupUi(self, MainPages):
        if not MainPages.objectName():
            MainPages.setObjectName(u"MainPages")
        MainPages.resize(1145, 820)
        self.verticalLayout_13 = QVBoxLayout(MainPages)
        self.verticalLayout_13.setObjectName(u"verticalLayout_13")
        self.pages = QStackedWidget(MainPages)
        self.pages.setObjectName(u"pages")
        font = QFont()
        font.setPointSize(12)
        self.pages.setFont(font)
        self.page_1 = QWidget()
        self.page_1.setObjectName(u"page_1")
        self.page_1.setFont(font)
        self.page_1.setStyleSheet(u"font-size: 12pt")
        self.verticalLayout = QVBoxLayout(self.page_1)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout_6 = QVBoxLayout()
        self.verticalLayout_6.setObjectName(u"verticalLayout_6")
        self.label_27 = QLabel(self.page_1)
        self.label_27.setObjectName(u"label_27")
        self.label_27.setAlignment(Qt.AlignCenter)

        self.verticalLayout_6.addWidget(self.label_27)

        self.horizontalLayout_14 = QHBoxLayout()
        self.horizontalLayout_14.setObjectName(u"horizontalLayout_14")
        self.verticalLayout_7 = QVBoxLayout()
        self.verticalLayout_7.setSpacing(0)
        self.verticalLayout_7.setObjectName(u"verticalLayout_7")
        self.line_9 = QFrame(self.page_1)
        self.line_9.setObjectName(u"line_9")
        self.line_9.setFrameShape(QFrame.HLine)
        self.line_9.setFrameShadow(QFrame.Sunken)

        self.verticalLayout_7.addWidget(self.line_9)

        self.label_14 = QLabel(self.page_1)
        self.label_14.setObjectName(u"label_14")
        sizePolicy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_14.sizePolicy().hasHeightForWidth())
        self.label_14.setSizePolicy(sizePolicy)
        self.label_14.setAlignment(Qt.AlignCenter)

        self.verticalLayout_7.addWidget(self.label_14)

        self.line_10 = QFrame(self.page_1)
        self.line_10.setObjectName(u"line_10")
        self.line_10.setFrameShape(QFrame.HLine)
        self.line_10.setFrameShadow(QFrame.Sunken)

        self.verticalLayout_7.addWidget(self.line_10)

        self.gridLayout = QGridLayout()
        self.gridLayout.setObjectName(u"gridLayout")
        self.label_15 = QLabel(self.page_1)
        self.label_15.setObjectName(u"label_15")

        self.gridLayout.addWidget(self.label_15, 0, 0, 1, 1)

        self.lineEdit_Low_Freq = QLineEdit(self.page_1)
        self.lineEdit_Low_Freq.setObjectName(u"lineEdit_Low_Freq")
        sizePolicy1 = QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.lineEdit_Low_Freq.sizePolicy().hasHeightForWidth())
        self.lineEdit_Low_Freq.setSizePolicy(sizePolicy1)

        self.gridLayout.addWidget(self.lineEdit_Low_Freq, 2, 2, 1, 1)

        self.label_18 = QLabel(self.page_1)
        self.label_18.setObjectName(u"label_18")

        self.gridLayout.addWidget(self.label_18, 3, 0, 1, 1)

        self.label_16 = QLabel(self.page_1)
        self.label_16.setObjectName(u"label_16")

        self.gridLayout.addWidget(self.label_16, 1, 0, 1, 1)

        self.lineEdit_High_Freq = QLineEdit(self.page_1)
        self.lineEdit_High_Freq.setObjectName(u"lineEdit_High_Freq")
        sizePolicy1.setHeightForWidth(self.lineEdit_High_Freq.sizePolicy().hasHeightForWidth())
        self.lineEdit_High_Freq.setSizePolicy(sizePolicy1)

        self.gridLayout.addWidget(self.lineEdit_High_Freq, 3, 2, 1, 1)

        self.label_17 = QLabel(self.page_1)
        self.label_17.setObjectName(u"label_17")

        self.gridLayout.addWidget(self.label_17, 2, 0, 1, 1)

        self.lineEdit_Laser = QLineEdit(self.page_1)
        self.lineEdit_Laser.setObjectName(u"lineEdit_Laser")
        sizePolicy1.setHeightForWidth(self.lineEdit_Laser.sizePolicy().hasHeightForWidth())
        self.lineEdit_Laser.setSizePolicy(sizePolicy1)
        font1 = QFont()
        font1.setFamilies([u"Segoe UI"])
        font1.setPointSize(12)
        self.lineEdit_Laser.setFont(font1)

        self.gridLayout.addWidget(self.lineEdit_Laser, 0, 2, 1, 1)

        self.lineEdit_number_samples = QLineEdit(self.page_1)
        self.lineEdit_number_samples.setObjectName(u"lineEdit_number_samples")
        sizePolicy1.setHeightForWidth(self.lineEdit_number_samples.sizePolicy().hasHeightForWidth())
        self.lineEdit_number_samples.setSizePolicy(sizePolicy1)

        self.gridLayout.addWidget(self.lineEdit_number_samples, 1, 2, 1, 1)

        self.label_19 = QLabel(self.page_1)
        self.label_19.setObjectName(u"label_19")

        self.gridLayout.addWidget(self.label_19, 4, 0, 1, 1)

        self.lineEdit_Avg_FFT = QLineEdit(self.page_1)
        self.lineEdit_Avg_FFT.setObjectName(u"lineEdit_Avg_FFT")
        sizePolicy1.setHeightForWidth(self.lineEdit_Avg_FFT.sizePolicy().hasHeightForWidth())
        self.lineEdit_Avg_FFT.setSizePolicy(sizePolicy1)

        self.gridLayout.addWidget(self.lineEdit_Avg_FFT, 4, 2, 1, 1)


        self.verticalLayout_7.addLayout(self.gridLayout)

        self.horizontalLayout_18 = QHBoxLayout()
        self.horizontalLayout_18.setObjectName(u"horizontalLayout_18")
        self.DAQ_connect_but = QPushButton(self.page_1)
        self.DAQ_connect_but.setObjectName(u"DAQ_connect_but")
        sizePolicy1.setHeightForWidth(self.DAQ_connect_but.sizePolicy().hasHeightForWidth())
        self.DAQ_connect_but.setSizePolicy(sizePolicy1)

        self.horizontalLayout_18.addWidget(self.DAQ_connect_but)

        self.Stop_DAQ_but = QPushButton(self.page_1)
        self.Stop_DAQ_but.setObjectName(u"Stop_DAQ_but")
        sizePolicy1.setHeightForWidth(self.Stop_DAQ_but.sizePolicy().hasHeightForWidth())
        self.Stop_DAQ_but.setSizePolicy(sizePolicy1)

        self.horizontalLayout_18.addWidget(self.Stop_DAQ_but)


        self.verticalLayout_7.addLayout(self.horizontalLayout_18)

        self.horizontalLayout_20 = QHBoxLayout()
        self.horizontalLayout_20.setObjectName(u"horizontalLayout_20")
        self.horizontalLayout_20.setContentsMargins(0, 0, -1, -1)
        self.label_42 = QLabel(self.page_1)
        self.label_42.setObjectName(u"label_42")

        self.horizontalLayout_20.addWidget(self.label_42)

        self.Moment_val = QLabel(self.page_1)
        self.Moment_val.setObjectName(u"Moment_val")

        self.horizontalLayout_20.addWidget(self.Moment_val)


        self.verticalLayout_7.addLayout(self.horizontalLayout_20)

        self.horizontalLayout_21 = QHBoxLayout()
        self.horizontalLayout_21.setObjectName(u"horizontalLayout_21")
        self.horizontalLayout_21.setContentsMargins(0, 0, 0, 0)
        self.label_43 = QLabel(self.page_1)
        self.label_43.setObjectName(u"label_43")

        self.horizontalLayout_21.addWidget(self.label_43)

        self.Max_Peak_val = QLabel(self.page_1)
        self.Max_Peak_val.setObjectName(u"Max_Peak_val")

        self.horizontalLayout_21.addWidget(self.Max_Peak_val)


        self.verticalLayout_7.addLayout(self.horizontalLayout_21)

        self.verticalSpacer_4 = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.verticalLayout_7.addItem(self.verticalSpacer_4)

        self.line_12 = QFrame(self.page_1)
        self.line_12.setObjectName(u"line_12")
        self.line_12.setFrameShape(QFrame.HLine)
        self.line_12.setFrameShadow(QFrame.Sunken)

        self.verticalLayout_7.addWidget(self.line_12)

        self.label_22 = QLabel(self.page_1)
        self.label_22.setObjectName(u"label_22")
        sizePolicy.setHeightForWidth(self.label_22.sizePolicy().hasHeightForWidth())
        self.label_22.setSizePolicy(sizePolicy)
        self.label_22.setAlignment(Qt.AlignCenter)

        self.verticalLayout_7.addWidget(self.label_22)

        self.line_11 = QFrame(self.page_1)
        self.line_11.setObjectName(u"line_11")
        self.line_11.setFrameShape(QFrame.HLine)
        self.line_11.setFrameShadow(QFrame.Sunken)

        self.verticalLayout_7.addWidget(self.line_11)

        self.gridLayout_2 = QGridLayout()
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.lineEdit_Directory = QLineEdit(self.page_1)
        self.lineEdit_Directory.setObjectName(u"lineEdit_Directory")
        sizePolicy.setHeightForWidth(self.lineEdit_Directory.sizePolicy().hasHeightForWidth())
        self.lineEdit_Directory.setSizePolicy(sizePolicy)

        self.gridLayout_2.addWidget(self.lineEdit_Directory, 0, 1, 1, 1)

        self.label_20 = QLabel(self.page_1)
        self.label_20.setObjectName(u"label_20")
        sizePolicy2 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.label_20.sizePolicy().hasHeightForWidth())
        self.label_20.setSizePolicy(sizePolicy2)

        self.gridLayout_2.addWidget(self.label_20, 0, 0, 1, 1)

        self.lineEdit_Name_Files = QLineEdit(self.page_1)
        self.lineEdit_Name_Files.setObjectName(u"lineEdit_Name_Files")
        sizePolicy1.setHeightForWidth(self.lineEdit_Name_Files.sizePolicy().hasHeightForWidth())
        self.lineEdit_Name_Files.setSizePolicy(sizePolicy1)

        self.gridLayout_2.addWidget(self.lineEdit_Name_Files, 1, 1, 1, 1)

        self.label_21 = QLabel(self.page_1)
        self.label_21.setObjectName(u"label_21")
        sizePolicy2.setHeightForWidth(self.label_21.sizePolicy().hasHeightForWidth())
        self.label_21.setSizePolicy(sizePolicy2)

        self.gridLayout_2.addWidget(self.label_21, 1, 0, 1, 1)

        self.label_29 = QLabel(self.page_1)
        self.label_29.setObjectName(u"label_29")

        self.gridLayout_2.addWidget(self.label_29, 2, 0, 1, 1)

        self.lineEdit_amp_peak = QLineEdit(self.page_1)
        self.lineEdit_amp_peak.setObjectName(u"lineEdit_amp_peak")
        sizePolicy1.setHeightForWidth(self.lineEdit_amp_peak.sizePolicy().hasHeightForWidth())
        self.lineEdit_amp_peak.setSizePolicy(sizePolicy1)

        self.gridLayout_2.addWidget(self.lineEdit_amp_peak, 2, 1, 1, 1)


        self.verticalLayout_7.addLayout(self.gridLayout_2)

        self.horizontalLayout_19 = QHBoxLayout()
        self.horizontalLayout_19.setObjectName(u"horizontalLayout_19")
        self.Search_Particles_but = QPushButton(self.page_1)
        self.Search_Particles_but.setObjectName(u"Search_Particles_but")

        self.horizontalLayout_19.addWidget(self.Search_Particles_but)

        self.Stop_Search_P_but = QPushButton(self.page_1)
        self.Stop_Search_P_but.setObjectName(u"Stop_Search_P_but")

        self.horizontalLayout_19.addWidget(self.Stop_Search_P_but)


        self.verticalLayout_7.addLayout(self.horizontalLayout_19)

        self.verticalSpacer_5 = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.verticalLayout_7.addItem(self.verticalSpacer_5)

        self.line_13 = QFrame(self.page_1)
        self.line_13.setObjectName(u"line_13")
        self.line_13.setFrameShape(QFrame.HLine)
        self.line_13.setFrameShadow(QFrame.Sunken)

        self.verticalLayout_7.addWidget(self.line_13)

        self.label_23 = QLabel(self.page_1)
        self.label_23.setObjectName(u"label_23")
        sizePolicy.setHeightForWidth(self.label_23.sizePolicy().hasHeightForWidth())
        self.label_23.setSizePolicy(sizePolicy)
        self.label_23.setAlignment(Qt.AlignCenter)

        self.verticalLayout_7.addWidget(self.label_23)

        self.line_14 = QFrame(self.page_1)
        self.line_14.setObjectName(u"line_14")
        self.line_14.setFrameShape(QFrame.HLine)
        self.line_14.setFrameShadow(QFrame.Sunken)

        self.verticalLayout_7.addWidget(self.line_14)

        self.gridLayout_3 = QGridLayout()
        self.gridLayout_3.setObjectName(u"gridLayout_3")
        self.label_24 = QLabel(self.page_1)
        self.label_24.setObjectName(u"label_24")

        self.gridLayout_3.addWidget(self.label_24, 0, 0, 1, 1)

        self.lineEdit_steps_zab = QLineEdit(self.page_1)
        self.lineEdit_steps_zab.setObjectName(u"lineEdit_steps_zab")
        sizePolicy1.setHeightForWidth(self.lineEdit_steps_zab.sizePolicy().hasHeightForWidth())
        self.lineEdit_steps_zab.setSizePolicy(sizePolicy1)

        self.gridLayout_3.addWidget(self.lineEdit_steps_zab, 2, 1, 1, 1)

        self.label_26 = QLabel(self.page_1)
        self.label_26.setObjectName(u"label_26")

        self.gridLayout_3.addWidget(self.label_26, 2, 0, 1, 1)

        self.label_25 = QLabel(self.page_1)
        self.label_25.setObjectName(u"label_25")

        self.gridLayout_3.addWidget(self.label_25, 1, 0, 1, 1)

        self.lineEdit_start_zab_pos = QLineEdit(self.page_1)
        self.lineEdit_start_zab_pos.setObjectName(u"lineEdit_start_zab_pos")
        sizePolicy1.setHeightForWidth(self.lineEdit_start_zab_pos.sizePolicy().hasHeightForWidth())
        self.lineEdit_start_zab_pos.setSizePolicy(sizePolicy1)

        self.gridLayout_3.addWidget(self.lineEdit_start_zab_pos, 0, 1, 1, 1)

        self.lineEdit_end_zab_pos = QLineEdit(self.page_1)
        self.lineEdit_end_zab_pos.setObjectName(u"lineEdit_end_zab_pos")
        sizePolicy1.setHeightForWidth(self.lineEdit_end_zab_pos.sizePolicy().hasHeightForWidth())
        self.lineEdit_end_zab_pos.setSizePolicy(sizePolicy1)

        self.gridLayout_3.addWidget(self.lineEdit_end_zab_pos, 1, 1, 1, 1)

        self.label_28 = QLabel(self.page_1)
        self.label_28.setObjectName(u"label_28")

        self.gridLayout_3.addWidget(self.label_28, 3, 0, 1, 1)

        self.lineEdit_zab_samplesCheck = QLineEdit(self.page_1)
        self.lineEdit_zab_samplesCheck.setObjectName(u"lineEdit_zab_samplesCheck")
        sizePolicy1.setHeightForWidth(self.lineEdit_zab_samplesCheck.sizePolicy().hasHeightForWidth())
        self.lineEdit_zab_samplesCheck.setSizePolicy(sizePolicy1)

        self.gridLayout_3.addWidget(self.lineEdit_zab_samplesCheck, 3, 1, 1, 1)


        self.verticalLayout_7.addLayout(self.gridLayout_3)

        self.Start_Zaber_Calib_but = QPushButton(self.page_1)
        self.Start_Zaber_Calib_but.setObjectName(u"Start_Zaber_Calib_but")
        sizePolicy.setHeightForWidth(self.Start_Zaber_Calib_but.sizePolicy().hasHeightForWidth())
        self.Start_Zaber_Calib_but.setSizePolicy(sizePolicy)
        self.Start_Zaber_Calib_but.setFont(font)

        self.verticalLayout_7.addWidget(self.Start_Zaber_Calib_but)

        self.verticalSpacer_3 = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.verticalLayout_7.addItem(self.verticalSpacer_3)


        self.horizontalLayout_14.addLayout(self.verticalLayout_7)

        self.LayoutGraphics = QVBoxLayout()
        self.LayoutGraphics.setObjectName(u"LayoutGraphics")
        self.LayoutGraphics.setContentsMargins(10, 10, 10, 10)
        self.horizontalSpacer_8 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.LayoutGraphics.addItem(self.horizontalSpacer_8)


        self.horizontalLayout_14.addLayout(self.LayoutGraphics)


        self.verticalLayout_6.addLayout(self.horizontalLayout_14)


        self.verticalLayout.addLayout(self.verticalLayout_6)

        self.pages.addWidget(self.page_1)
        self.page_2 = QWidget()
        self.page_2.setObjectName(u"page_2")
        self.page_2_layout = QVBoxLayout(self.page_2)
        self.page_2_layout.setSpacing(5)
        self.page_2_layout.setObjectName(u"page_2_layout")
        self.page_2_layout.setContentsMargins(5, 5, 5, 5)
        self.Server_Layout = QVBoxLayout()
        self.Server_Layout.setObjectName(u"Server_Layout")
        self.label_50 = QLabel(self.page_2)
        self.label_50.setObjectName(u"label_50")
        sizePolicy3 = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        sizePolicy3.setHorizontalStretch(0)
        sizePolicy3.setVerticalStretch(0)
        sizePolicy3.setHeightForWidth(self.label_50.sizePolicy().hasHeightForWidth())
        self.label_50.setSizePolicy(sizePolicy3)
        self.label_50.setFocusPolicy(Qt.NoFocus)
        self.label_50.setLayoutDirection(Qt.LeftToRight)
        self.label_50.setAlignment(Qt.AlignCenter)

        self.Server_Layout.addWidget(self.label_50)

        self.Update_but_Server = QPushButton(self.page_2)
        self.Update_but_Server.setObjectName(u"Update_but_Server")

        self.Server_Layout.addWidget(self.Update_but_Server)


        self.page_2_layout.addLayout(self.Server_Layout)

        self.pages.addWidget(self.page_2)
        self.page_3 = QWidget()
        self.page_3.setObjectName(u"page_3")
        self.page_3.setStyleSheet(u"font-size: 12pt;\n"
"")
        self.page_3_layout = QVBoxLayout(self.page_3)
        self.page_3_layout.setObjectName(u"page_3_layout")
        self.verticalLayout_2 = QVBoxLayout()
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.label_p3 = QLabel(self.page_3)
        self.label_p3.setObjectName(u"label_p3")

        self.verticalLayout_2.addWidget(self.label_p3)

        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.zabercon_p3 = QPushButton(self.page_3)
        self.zabercon_p3.setObjectName(u"zabercon_p3")

        self.horizontalLayout.addWidget(self.zabercon_p3)

        self.n_devices_p3 = QLabel(self.page_3)
        self.n_devices_p3.setObjectName(u"n_devices_p3")
        self.n_devices_p3.setFont(font)

        self.horizontalLayout.addWidget(self.n_devices_p3)

        self.check_p3 = QRadioButton(self.page_3)
        self.check_p3.setObjectName(u"check_p3")
        self.check_p3.setCheckable(False)
        self.check_p3.setAutoExclusive(False)

        self.horizontalLayout.addWidget(self.check_p3)

        self.horizontalSpacer = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout.addItem(self.horizontalSpacer)


        self.verticalLayout_2.addLayout(self.horizontalLayout)

        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.globalhome_p3 = QPushButton(self.page_3)
        self.globalhome_p3.setObjectName(u"globalhome_p3")
        sizePolicy4 = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Preferred)
        sizePolicy4.setHorizontalStretch(0)
        sizePolicy4.setVerticalStretch(0)
        sizePolicy4.setHeightForWidth(self.globalhome_p3.sizePolicy().hasHeightForWidth())
        self.globalhome_p3.setSizePolicy(sizePolicy4)

        self.horizontalLayout_2.addWidget(self.globalhome_p3)

        self.home1_p3 = QPushButton(self.page_3)
        self.home1_p3.setObjectName(u"home1_p3")
        sizePolicy4.setHeightForWidth(self.home1_p3.sizePolicy().hasHeightForWidth())
        self.home1_p3.setSizePolicy(sizePolicy4)

        self.horizontalLayout_2.addWidget(self.home1_p3)

        self.home2_p3 = QPushButton(self.page_3)
        self.home2_p3.setObjectName(u"home2_p3")
        sizePolicy4.setHeightForWidth(self.home2_p3.sizePolicy().hasHeightForWidth())
        self.home2_p3.setSizePolicy(sizePolicy4)

        self.horizontalLayout_2.addWidget(self.home2_p3)

        self.home3_p3 = QPushButton(self.page_3)
        self.home3_p3.setObjectName(u"home3_p3")
        sizePolicy4.setHeightForWidth(self.home3_p3.sizePolicy().hasHeightForWidth())
        self.home3_p3.setSizePolicy(sizePolicy4)

        self.horizontalLayout_2.addWidget(self.home3_p3)


        self.verticalLayout_2.addLayout(self.horizontalLayout_2)

        self.horizontalLayout_3 = QHBoxLayout()
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.verticalLayout_3 = QVBoxLayout()
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.label = QLabel(self.page_3)
        self.label.setObjectName(u"label")

        self.verticalLayout_3.addWidget(self.label)

        self.horizontalLayout_10 = QHBoxLayout()
        self.horizontalLayout_10.setObjectName(u"horizontalLayout_10")
        self.z1_Relative = QCheckBox(self.page_3)
        self.z1_Relative.setObjectName(u"z1_Relative")

        self.horizontalLayout_10.addWidget(self.z1_Relative)

        self.z1_Absolute = QCheckBox(self.page_3)
        self.z1_Absolute.setObjectName(u"z1_Absolute")

        self.horizontalLayout_10.addWidget(self.z1_Absolute)


        self.verticalLayout_3.addLayout(self.horizontalLayout_10)

        self.horizontalLayout_4 = QHBoxLayout()
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.z1_cm = QCheckBox(self.page_3)
        self.z1_cm.setObjectName(u"z1_cm")

        self.horizontalLayout_4.addWidget(self.z1_cm)

        self.z1_mm = QCheckBox(self.page_3)
        self.z1_mm.setObjectName(u"z1_mm")

        self.horizontalLayout_4.addWidget(self.z1_mm)

        self.z1_um = QCheckBox(self.page_3)
        self.z1_um.setObjectName(u"z1_um")

        self.horizontalLayout_4.addWidget(self.z1_um)


        self.verticalLayout_3.addLayout(self.horizontalLayout_4)

        self.horizontalLayout_7 = QHBoxLayout()
        self.horizontalLayout_7.setObjectName(u"horizontalLayout_7")
        self.label_4 = QLabel(self.page_3)
        self.label_4.setObjectName(u"label_4")

        self.horizontalLayout_7.addWidget(self.label_4)

        self.z1_data = QLineEdit(self.page_3)
        self.z1_data.setObjectName(u"z1_data")
        sizePolicy1.setHeightForWidth(self.z1_data.sizePolicy().hasHeightForWidth())
        self.z1_data.setSizePolicy(sizePolicy1)

        self.horizontalLayout_7.addWidget(self.z1_data)

        self.z1_set = QPushButton(self.page_3)
        self.z1_set.setObjectName(u"z1_set")

        self.horizontalLayout_7.addWidget(self.z1_set)


        self.verticalLayout_3.addLayout(self.horizontalLayout_7)

        self.z1_slider = QSlider(self.page_3)
        self.z1_slider.setObjectName(u"z1_slider")
        self.z1_slider.setValue(50)
        self.z1_slider.setOrientation(Qt.Horizontal)

        self.verticalLayout_3.addWidget(self.z1_slider)

        self.line_3 = QFrame(self.page_3)
        self.line_3.setObjectName(u"line_3")
        self.line_3.setFrameShape(QFrame.HLine)
        self.line_3.setFrameShadow(QFrame.Sunken)

        self.verticalLayout_3.addWidget(self.line_3)

        self.label_7 = QLabel(self.page_3)
        self.label_7.setObjectName(u"label_7")
        self.label_7.setFont(font)

        self.verticalLayout_3.addWidget(self.label_7)

        self.line_6 = QFrame(self.page_3)
        self.line_6.setObjectName(u"line_6")
        self.line_6.setFrameShape(QFrame.HLine)
        self.line_6.setFrameShadow(QFrame.Sunken)

        self.verticalLayout_3.addWidget(self.line_6)

        self.horizontalLayout_13 = QHBoxLayout()
        self.horizontalLayout_13.setObjectName(u"horizontalLayout_13")
        self.horizontalLayout_13.setContentsMargins(-1, 0, -1, -1)
        self.z1speed = QLineEdit(self.page_3)
        self.z1speed.setObjectName(u"z1speed")
        sizePolicy5 = QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Minimum)
        sizePolicy5.setHorizontalStretch(0)
        sizePolicy5.setVerticalStretch(0)
        sizePolicy5.setHeightForWidth(self.z1speed.sizePolicy().hasHeightForWidth())
        self.z1speed.setSizePolicy(sizePolicy5)
        self.z1speed.setMaxLength(200)

        self.horizontalLayout_13.addWidget(self.z1speed)

        self.label_10 = QLabel(self.page_3)
        self.label_10.setObjectName(u"label_10")

        self.horizontalLayout_13.addWidget(self.label_10)

        self.horizontalSpacer_5 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_13.addItem(self.horizontalSpacer_5)

        self.z1_left = QPushButton(self.page_3)
        self.z1_left.setObjectName(u"z1_left")

        self.horizontalLayout_13.addWidget(self.z1_left)

        self.horizontalSpacer_2 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_13.addItem(self.horizontalSpacer_2)

        self.z1_right = QPushButton(self.page_3)
        self.z1_right.setObjectName(u"z1_right")

        self.horizontalLayout_13.addWidget(self.z1_right)


        self.verticalLayout_3.addLayout(self.horizontalLayout_13)

        self.z1_Stop = QPushButton(self.page_3)
        self.z1_Stop.setObjectName(u"z1_Stop")

        self.verticalLayout_3.addWidget(self.z1_Stop)


        self.horizontalLayout_3.addLayout(self.verticalLayout_3)

        self.line = QFrame(self.page_3)
        self.line.setObjectName(u"line")
        self.line.setFrameShape(QFrame.VLine)
        self.line.setFrameShadow(QFrame.Sunken)

        self.horizontalLayout_3.addWidget(self.line)

        self.verticalLayout_4 = QVBoxLayout()
        self.verticalLayout_4.setObjectName(u"verticalLayout_4")
        self.label_2 = QLabel(self.page_3)
        self.label_2.setObjectName(u"label_2")

        self.verticalLayout_4.addWidget(self.label_2)

        self.horizontalLayout_11 = QHBoxLayout()
        self.horizontalLayout_11.setObjectName(u"horizontalLayout_11")
        self.z2_Relative = QCheckBox(self.page_3)
        self.z2_Relative.setObjectName(u"z2_Relative")

        self.horizontalLayout_11.addWidget(self.z2_Relative)

        self.z2_Absolute = QCheckBox(self.page_3)
        self.z2_Absolute.setObjectName(u"z2_Absolute")

        self.horizontalLayout_11.addWidget(self.z2_Absolute)


        self.verticalLayout_4.addLayout(self.horizontalLayout_11)

        self.horizontalLayout_5 = QHBoxLayout()
        self.horizontalLayout_5.setObjectName(u"horizontalLayout_5")
        self.z2_cm = QCheckBox(self.page_3)
        self.z2_cm.setObjectName(u"z2_cm")

        self.horizontalLayout_5.addWidget(self.z2_cm)

        self.z2_mm = QCheckBox(self.page_3)
        self.z2_mm.setObjectName(u"z2_mm")

        self.horizontalLayout_5.addWidget(self.z2_mm)

        self.z2_um = QCheckBox(self.page_3)
        self.z2_um.setObjectName(u"z2_um")

        self.horizontalLayout_5.addWidget(self.z2_um)


        self.verticalLayout_4.addLayout(self.horizontalLayout_5)

        self.horizontalLayout_8 = QHBoxLayout()
        self.horizontalLayout_8.setObjectName(u"horizontalLayout_8")
        self.label_5 = QLabel(self.page_3)
        self.label_5.setObjectName(u"label_5")

        self.horizontalLayout_8.addWidget(self.label_5)

        self.z2_data = QLineEdit(self.page_3)
        self.z2_data.setObjectName(u"z2_data")
        sizePolicy1.setHeightForWidth(self.z2_data.sizePolicy().hasHeightForWidth())
        self.z2_data.setSizePolicy(sizePolicy1)

        self.horizontalLayout_8.addWidget(self.z2_data)

        self.z2_Set = QPushButton(self.page_3)
        self.z2_Set.setObjectName(u"z2_Set")

        self.horizontalLayout_8.addWidget(self.z2_Set)


        self.verticalLayout_4.addLayout(self.horizontalLayout_8)

        self.z2_slider = QSlider(self.page_3)
        self.z2_slider.setObjectName(u"z2_slider")
        self.z2_slider.setValue(50)
        self.z2_slider.setOrientation(Qt.Horizontal)

        self.verticalLayout_4.addWidget(self.z2_slider)

        self.line_4 = QFrame(self.page_3)
        self.line_4.setObjectName(u"line_4")
        self.line_4.setFrameShape(QFrame.HLine)
        self.line_4.setFrameShadow(QFrame.Sunken)

        self.verticalLayout_4.addWidget(self.line_4)

        self.label_8 = QLabel(self.page_3)
        self.label_8.setObjectName(u"label_8")

        self.verticalLayout_4.addWidget(self.label_8)

        self.line_7 = QFrame(self.page_3)
        self.line_7.setObjectName(u"line_7")
        self.line_7.setFrameShape(QFrame.HLine)
        self.line_7.setFrameShadow(QFrame.Sunken)

        self.verticalLayout_4.addWidget(self.line_7)

        self.horizontalLayout_15 = QHBoxLayout()
        self.horizontalLayout_15.setObjectName(u"horizontalLayout_15")
        self.z2speed = QLineEdit(self.page_3)
        self.z2speed.setObjectName(u"z2speed")

        self.horizontalLayout_15.addWidget(self.z2speed)

        self.label_11 = QLabel(self.page_3)
        self.label_11.setObjectName(u"label_11")

        self.horizontalLayout_15.addWidget(self.label_11)

        self.horizontalSpacer_6 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_15.addItem(self.horizontalSpacer_6)

        self.z2_left = QPushButton(self.page_3)
        self.z2_left.setObjectName(u"z2_left")

        self.horizontalLayout_15.addWidget(self.z2_left)

        self.horizontalSpacer_3 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_15.addItem(self.horizontalSpacer_3)

        self.z2_right = QPushButton(self.page_3)
        self.z2_right.setObjectName(u"z2_right")

        self.horizontalLayout_15.addWidget(self.z2_right)


        self.verticalLayout_4.addLayout(self.horizontalLayout_15)

        self.z2_Stop = QPushButton(self.page_3)
        self.z2_Stop.setObjectName(u"z2_Stop")

        self.verticalLayout_4.addWidget(self.z2_Stop)


        self.horizontalLayout_3.addLayout(self.verticalLayout_4)

        self.line_2 = QFrame(self.page_3)
        self.line_2.setObjectName(u"line_2")
        self.line_2.setFrameShape(QFrame.VLine)
        self.line_2.setFrameShadow(QFrame.Sunken)

        self.horizontalLayout_3.addWidget(self.line_2)

        self.verticalLayout_5 = QVBoxLayout()
        self.verticalLayout_5.setObjectName(u"verticalLayout_5")
        self.label_3 = QLabel(self.page_3)
        self.label_3.setObjectName(u"label_3")

        self.verticalLayout_5.addWidget(self.label_3)

        self.horizontalLayout_12 = QHBoxLayout()
        self.horizontalLayout_12.setObjectName(u"horizontalLayout_12")
        self.z3_Relative = QCheckBox(self.page_3)
        self.z3_Relative.setObjectName(u"z3_Relative")

        self.horizontalLayout_12.addWidget(self.z3_Relative)

        self.z3_Absolute = QCheckBox(self.page_3)
        self.z3_Absolute.setObjectName(u"z3_Absolute")

        self.horizontalLayout_12.addWidget(self.z3_Absolute)


        self.verticalLayout_5.addLayout(self.horizontalLayout_12)

        self.horizontalLayout_6 = QHBoxLayout()
        self.horizontalLayout_6.setObjectName(u"horizontalLayout_6")
        self.z3_cm = QCheckBox(self.page_3)
        self.z3_cm.setObjectName(u"z3_cm")

        self.horizontalLayout_6.addWidget(self.z3_cm)

        self.z3_mm = QCheckBox(self.page_3)
        self.z3_mm.setObjectName(u"z3_mm")

        self.horizontalLayout_6.addWidget(self.z3_mm)

        self.z3_um = QCheckBox(self.page_3)
        self.z3_um.setObjectName(u"z3_um")

        self.horizontalLayout_6.addWidget(self.z3_um)


        self.verticalLayout_5.addLayout(self.horizontalLayout_6)

        self.horizontalLayout_9 = QHBoxLayout()
        self.horizontalLayout_9.setObjectName(u"horizontalLayout_9")
        self.label_6 = QLabel(self.page_3)
        self.label_6.setObjectName(u"label_6")

        self.horizontalLayout_9.addWidget(self.label_6)

        self.z3_data = QLineEdit(self.page_3)
        self.z3_data.setObjectName(u"z3_data")
        sizePolicy1.setHeightForWidth(self.z3_data.sizePolicy().hasHeightForWidth())
        self.z3_data.setSizePolicy(sizePolicy1)

        self.horizontalLayout_9.addWidget(self.z3_data)

        self.z3_Set = QPushButton(self.page_3)
        self.z3_Set.setObjectName(u"z3_Set")

        self.horizontalLayout_9.addWidget(self.z3_Set)


        self.verticalLayout_5.addLayout(self.horizontalLayout_9)

        self.z3_slider = QSlider(self.page_3)
        self.z3_slider.setObjectName(u"z3_slider")
        self.z3_slider.setValue(50)
        self.z3_slider.setOrientation(Qt.Horizontal)

        self.verticalLayout_5.addWidget(self.z3_slider)

        self.line_5 = QFrame(self.page_3)
        self.line_5.setObjectName(u"line_5")
        self.line_5.setFrameShape(QFrame.HLine)
        self.line_5.setFrameShadow(QFrame.Sunken)

        self.verticalLayout_5.addWidget(self.line_5)

        self.label_9 = QLabel(self.page_3)
        self.label_9.setObjectName(u"label_9")

        self.verticalLayout_5.addWidget(self.label_9)

        self.line_8 = QFrame(self.page_3)
        self.line_8.setObjectName(u"line_8")
        self.line_8.setFrameShape(QFrame.HLine)
        self.line_8.setFrameShadow(QFrame.Sunken)

        self.verticalLayout_5.addWidget(self.line_8)

        self.horizontalLayout_16 = QHBoxLayout()
        self.horizontalLayout_16.setObjectName(u"horizontalLayout_16")
        self.z3speed = QLineEdit(self.page_3)
        self.z3speed.setObjectName(u"z3speed")

        self.horizontalLayout_16.addWidget(self.z3speed)

        self.label_12 = QLabel(self.page_3)
        self.label_12.setObjectName(u"label_12")

        self.horizontalLayout_16.addWidget(self.label_12)

        self.horizontalSpacer_7 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_16.addItem(self.horizontalSpacer_7)

        self.z3_left = QPushButton(self.page_3)
        self.z3_left.setObjectName(u"z3_left")

        self.horizontalLayout_16.addWidget(self.z3_left)

        self.horizontalSpacer_4 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_16.addItem(self.horizontalSpacer_4)

        self.z3_right = QPushButton(self.page_3)
        self.z3_right.setObjectName(u"z3_right")

        self.horizontalLayout_16.addWidget(self.z3_right)


        self.verticalLayout_5.addLayout(self.horizontalLayout_16)

        self.z3_Stop = QPushButton(self.page_3)
        self.z3_Stop.setObjectName(u"z3_Stop")

        self.verticalLayout_5.addWidget(self.z3_Stop)


        self.horizontalLayout_3.addLayout(self.verticalLayout_5)


        self.verticalLayout_2.addLayout(self.horizontalLayout_3)

        self.Daq_Signal = QHBoxLayout()
        self.Daq_Signal.setObjectName(u"Daq_Signal")

        self.verticalLayout_2.addLayout(self.Daq_Signal)

        self.Keyboard_p3 = QPushButton(self.page_3)
        self.Keyboard_p3.setObjectName(u"Keyboard_p3")

        self.verticalLayout_2.addWidget(self.Keyboard_p3)

        self.verticalSpacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.verticalLayout_2.addItem(self.verticalSpacer)


        self.page_3_layout.addLayout(self.verticalLayout_2)

        self.pages.addWidget(self.page_3)
        self.page_calib = QWidget()
        self.page_calib.setObjectName(u"page_calib")
        self.page_calib.setStyleSheet(u"font: 12pt \"Segoe UI\";")
        self.verticalLayout_11 = QVBoxLayout(self.page_calib)
        self.verticalLayout_11.setObjectName(u"verticalLayout_11")
        self.verticalLayout_10 = QVBoxLayout()
        self.verticalLayout_10.setSpacing(0)
        self.verticalLayout_10.setObjectName(u"verticalLayout_10")
        self.label_44 = QLabel(self.page_calib)
        self.label_44.setObjectName(u"label_44")
        sizePolicy.setHeightForWidth(self.label_44.sizePolicy().hasHeightForWidth())
        self.label_44.setSizePolicy(sizePolicy)

        self.verticalLayout_10.addWidget(self.label_44)

        self.horizontalLayout_17 = QHBoxLayout()
        self.horizontalLayout_17.setSpacing(10)
        self.horizontalLayout_17.setObjectName(u"horizontalLayout_17")
        self.horizontalLayout_17.setSizeConstraint(QLayout.SetFixedSize)
        self.horizontalLayout_17.setContentsMargins(10, 10, 10, 10)
        self.verticalLayout_8 = QVBoxLayout()
        self.verticalLayout_8.setSpacing(0)
        self.verticalLayout_8.setObjectName(u"verticalLayout_8")
        self.verticalLayout_8.setSizeConstraint(QLayout.SetMinimumSize)
        self.verticalSpacer_10 = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.verticalLayout_8.addItem(self.verticalSpacer_10)

        self.line_15 = QFrame(self.page_calib)
        self.line_15.setObjectName(u"line_15")
        self.line_15.setFrameShape(QFrame.HLine)
        self.line_15.setFrameShadow(QFrame.Sunken)

        self.verticalLayout_8.addWidget(self.line_15)

        self.label_45 = QLabel(self.page_calib)
        self.label_45.setObjectName(u"label_45")
        sizePolicy6 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        sizePolicy6.setHorizontalStretch(0)
        sizePolicy6.setVerticalStretch(0)
        sizePolicy6.setHeightForWidth(self.label_45.sizePolicy().hasHeightForWidth())
        self.label_45.setSizePolicy(sizePolicy6)
        self.label_45.setAlignment(Qt.AlignCenter)

        self.verticalLayout_8.addWidget(self.label_45)

        self.line_16 = QFrame(self.page_calib)
        self.line_16.setObjectName(u"line_16")
        self.line_16.setFrameShape(QFrame.HLine)
        self.line_16.setFrameShadow(QFrame.Sunken)

        self.verticalLayout_8.addWidget(self.line_16)

        self.Vel_Start_Calib = QPushButton(self.page_calib)
        self.Vel_Start_Calib.setObjectName(u"Vel_Start_Calib")
        sizePolicy3.setHeightForWidth(self.Vel_Start_Calib.sizePolicy().hasHeightForWidth())
        self.Vel_Start_Calib.setSizePolicy(sizePolicy3)

        self.verticalLayout_8.addWidget(self.Vel_Start_Calib)

        self.gridLayout_5 = QGridLayout()
        self.gridLayout_5.setObjectName(u"gridLayout_5")
        self.gridLayout_5.setSizeConstraint(QLayout.SetFixedSize)
        self.Vel_Steps_Calib = QLineEdit(self.page_calib)
        self.Vel_Steps_Calib.setObjectName(u"Vel_Steps_Calib")
        sizePolicy3.setHeightForWidth(self.Vel_Steps_Calib.sizePolicy().hasHeightForWidth())
        self.Vel_Steps_Calib.setSizePolicy(sizePolicy3)

        self.gridLayout_5.addWidget(self.Vel_Steps_Calib, 2, 1, 1, 1)

        self.Vel_Limit_X1 = QLabel(self.page_calib)
        self.Vel_Limit_X1.setObjectName(u"Vel_Limit_X1")
        sizePolicy4.setHeightForWidth(self.Vel_Limit_X1.sizePolicy().hasHeightForWidth())
        self.Vel_Limit_X1.setSizePolicy(sizePolicy4)

        self.gridLayout_5.addWidget(self.Vel_Limit_X1, 3, 1, 1, 1)

        self.label_31 = QLabel(self.page_calib)
        self.label_31.setObjectName(u"label_31")
        sizePolicy4.setHeightForWidth(self.label_31.sizePolicy().hasHeightForWidth())
        self.label_31.setSizePolicy(sizePolicy4)

        self.gridLayout_5.addWidget(self.label_31, 2, 0, 1, 1)

        self.Vel_Manual_X1 = QLineEdit(self.page_calib)
        self.Vel_Manual_X1.setObjectName(u"Vel_Manual_X1")
        sizePolicy3.setHeightForWidth(self.Vel_Manual_X1.sizePolicy().hasHeightForWidth())
        self.Vel_Manual_X1.setSizePolicy(sizePolicy3)

        self.gridLayout_5.addWidget(self.Vel_Manual_X1, 1, 1, 1, 1)

        self.label_30 = QLabel(self.page_calib)
        self.label_30.setObjectName(u"label_30")
        sizePolicy3.setHeightForWidth(self.label_30.sizePolicy().hasHeightForWidth())
        self.label_30.setSizePolicy(sizePolicy3)

        self.gridLayout_5.addWidget(self.label_30, 1, 0, 1, 1)

        self.label_47 = QLabel(self.page_calib)
        self.label_47.setObjectName(u"label_47")
        sizePolicy3.setHeightForWidth(self.label_47.sizePolicy().hasHeightForWidth())
        self.label_47.setSizePolicy(sizePolicy3)

        self.gridLayout_5.addWidget(self.label_47, 2, 2, 1, 1)

        self.label_48 = QLabel(self.page_calib)
        self.label_48.setObjectName(u"label_48")
        sizePolicy3.setHeightForWidth(self.label_48.sizePolicy().hasHeightForWidth())
        self.label_48.setSizePolicy(sizePolicy3)

        self.gridLayout_5.addWidget(self.label_48, 3, 2, 1, 1)

        self.label_46 = QLabel(self.page_calib)
        self.label_46.setObjectName(u"label_46")
        sizePolicy3.setHeightForWidth(self.label_46.sizePolicy().hasHeightForWidth())
        self.label_46.setSizePolicy(sizePolicy3)

        self.gridLayout_5.addWidget(self.label_46, 1, 2, 1, 1)

        self.label_32 = QLabel(self.page_calib)
        self.label_32.setObjectName(u"label_32")
        sizePolicy4.setHeightForWidth(self.label_32.sizePolicy().hasHeightForWidth())
        self.label_32.setSizePolicy(sizePolicy4)

        self.gridLayout_5.addWidget(self.label_32, 3, 0, 1, 1)

        self.Vel_Manual_X2 = QLineEdit(self.page_calib)
        self.Vel_Manual_X2.setObjectName(u"Vel_Manual_X2")
        sizePolicy3.setHeightForWidth(self.Vel_Manual_X2.sizePolicy().hasHeightForWidth())
        self.Vel_Manual_X2.setSizePolicy(sizePolicy3)

        self.gridLayout_5.addWidget(self.Vel_Manual_X2, 1, 3, 1, 1)

        self.Vel_Threshold_Peak = QLineEdit(self.page_calib)
        self.Vel_Threshold_Peak.setObjectName(u"Vel_Threshold_Peak")
        sizePolicy3.setHeightForWidth(self.Vel_Threshold_Peak.sizePolicy().hasHeightForWidth())
        self.Vel_Threshold_Peak.setSizePolicy(sizePolicy3)

        self.gridLayout_5.addWidget(self.Vel_Threshold_Peak, 2, 3, 1, 1)

        self.Vel_Limit_X2 = QLabel(self.page_calib)
        self.Vel_Limit_X2.setObjectName(u"Vel_Limit_X2")
        sizePolicy4.setHeightForWidth(self.Vel_Limit_X2.sizePolicy().hasHeightForWidth())
        self.Vel_Limit_X2.setSizePolicy(sizePolicy4)

        self.gridLayout_5.addWidget(self.Vel_Limit_X2, 3, 3, 1, 1)


        self.verticalLayout_8.addLayout(self.gridLayout_5)

        self.verticalSpacer_6 = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.verticalLayout_8.addItem(self.verticalSpacer_6)

        self.line_17 = QFrame(self.page_calib)
        self.line_17.setObjectName(u"line_17")
        self.line_17.setFrameShape(QFrame.HLine)
        self.line_17.setFrameShadow(QFrame.Sunken)

        self.verticalLayout_8.addWidget(self.line_17)

        self.label_49 = QLabel(self.page_calib)
        self.label_49.setObjectName(u"label_49")
        sizePolicy6.setHeightForWidth(self.label_49.sizePolicy().hasHeightForWidth())
        self.label_49.setSizePolicy(sizePolicy6)
        self.label_49.setAlignment(Qt.AlignCenter)

        self.verticalLayout_8.addWidget(self.label_49)

        self.line_18 = QFrame(self.page_calib)
        self.line_18.setObjectName(u"line_18")
        self.line_18.setFrameShape(QFrame.HLine)
        self.line_18.setFrameShadow(QFrame.Sunken)

        self.verticalLayout_8.addWidget(self.line_18)

        self.gridLayout_7 = QGridLayout()
        self.gridLayout_7.setSpacing(10)
        self.gridLayout_7.setObjectName(u"gridLayout_7")
        self.gridLayout_7.setSizeConstraint(QLayout.SetFixedSize)
        self.gridLayout_7.setContentsMargins(10, 10, 10, 10)
        self.label_41 = QLabel(self.page_calib)
        self.label_41.setObjectName(u"label_41")

        self.gridLayout_7.addWidget(self.label_41, 2, 2, 1, 1)

        self.lineEdit_speed_ums = QLineEdit(self.page_calib)
        self.lineEdit_speed_ums.setObjectName(u"lineEdit_speed_ums")
        sizePolicy3.setHeightForWidth(self.lineEdit_speed_ums.sizePolicy().hasHeightForWidth())
        self.lineEdit_speed_ums.setSizePolicy(sizePolicy3)

        self.gridLayout_7.addWidget(self.lineEdit_speed_ums, 2, 3, 1, 1)

        self.lineEdit_y2_scan = QLineEdit(self.page_calib)
        self.lineEdit_y2_scan.setObjectName(u"lineEdit_y2_scan")
        sizePolicy3.setHeightForWidth(self.lineEdit_y2_scan.sizePolicy().hasHeightForWidth())
        self.lineEdit_y2_scan.setSizePolicy(sizePolicy3)

        self.gridLayout_7.addWidget(self.lineEdit_y2_scan, 1, 3, 1, 1)

        self.label_35 = QLabel(self.page_calib)
        self.label_35.setObjectName(u"label_35")

        self.gridLayout_7.addWidget(self.label_35, 0, 2, 1, 1)

        self.label_34 = QLabel(self.page_calib)
        self.label_34.setObjectName(u"label_34")

        self.gridLayout_7.addWidget(self.label_34, 1, 0, 1, 1)

        self.lineEdit_x1_scan = QLineEdit(self.page_calib)
        self.lineEdit_x1_scan.setObjectName(u"lineEdit_x1_scan")
        sizePolicy3.setHeightForWidth(self.lineEdit_x1_scan.sizePolicy().hasHeightForWidth())
        self.lineEdit_x1_scan.setSizePolicy(sizePolicy3)

        self.gridLayout_7.addWidget(self.lineEdit_x1_scan, 0, 1, 1, 1)

        self.lineEdit_y1_scan = QLineEdit(self.page_calib)
        self.lineEdit_y1_scan.setObjectName(u"lineEdit_y1_scan")
        sizePolicy3.setHeightForWidth(self.lineEdit_y1_scan.sizePolicy().hasHeightForWidth())
        self.lineEdit_y1_scan.setSizePolicy(sizePolicy3)

        self.gridLayout_7.addWidget(self.lineEdit_y1_scan, 0, 3, 1, 1)

        self.label_33 = QLabel(self.page_calib)
        self.label_33.setObjectName(u"label_33")

        self.gridLayout_7.addWidget(self.label_33, 0, 0, 1, 1)

        self.label_38 = QLabel(self.page_calib)
        self.label_38.setObjectName(u"label_38")

        self.gridLayout_7.addWidget(self.label_38, 2, 0, 1, 1)

        self.label_36 = QLabel(self.page_calib)
        self.label_36.setObjectName(u"label_36")

        self.gridLayout_7.addWidget(self.label_36, 1, 2, 1, 1)

        self.lineEdit_Pixel_size = QLineEdit(self.page_calib)
        self.lineEdit_Pixel_size.setObjectName(u"lineEdit_Pixel_size")
        sizePolicy3.setHeightForWidth(self.lineEdit_Pixel_size.sizePolicy().hasHeightForWidth())
        self.lineEdit_Pixel_size.setSizePolicy(sizePolicy3)

        self.gridLayout_7.addWidget(self.lineEdit_Pixel_size, 2, 1, 1, 1)

        self.lineEdit_x2_scan = QLineEdit(self.page_calib)
        self.lineEdit_x2_scan.setObjectName(u"lineEdit_x2_scan")
        sizePolicy3.setHeightForWidth(self.lineEdit_x2_scan.sizePolicy().hasHeightForWidth())
        self.lineEdit_x2_scan.setSizePolicy(sizePolicy3)

        self.gridLayout_7.addWidget(self.lineEdit_x2_scan, 1, 1, 1, 1)


        self.verticalLayout_8.addLayout(self.gridLayout_7)

        self.gridLayout_4 = QGridLayout()
        self.gridLayout_4.setObjectName(u"gridLayout_4")
        self.gridLayout_4.setContentsMargins(10, 10, 10, -1)
        self.Stop_Y_but = QPushButton(self.page_calib)
        self.Stop_Y_but.setObjectName(u"Stop_Y_but")
        sizePolicy3.setHeightForWidth(self.Stop_Y_but.sizePolicy().hasHeightForWidth())
        self.Stop_Y_but.setSizePolicy(sizePolicy3)

        self.gridLayout_4.addWidget(self.Stop_Y_but, 1, 1, 1, 1)

        self.Stop_x_but = QPushButton(self.page_calib)
        self.Stop_x_but.setObjectName(u"Stop_x_but")
        sizePolicy3.setHeightForWidth(self.Stop_x_but.sizePolicy().hasHeightForWidth())
        self.Stop_x_but.setSizePolicy(sizePolicy3)

        self.gridLayout_4.addWidget(self.Stop_x_but, 1, 0, 1, 1)

        self.continuous_scanY_but = QPushButton(self.page_calib)
        self.continuous_scanY_but.setObjectName(u"continuous_scanY_but")
        sizePolicy3.setHeightForWidth(self.continuous_scanY_but.sizePolicy().hasHeightForWidth())
        self.continuous_scanY_but.setSizePolicy(sizePolicy3)

        self.gridLayout_4.addWidget(self.continuous_scanY_but, 0, 1, 1, 1)

        self.continuous_scanX_but = QPushButton(self.page_calib)
        self.continuous_scanX_but.setObjectName(u"continuous_scanX_but")
        sizePolicy3.setHeightForWidth(self.continuous_scanX_but.sizePolicy().hasHeightForWidth())
        self.continuous_scanX_but.setSizePolicy(sizePolicy3)

        self.gridLayout_4.addWidget(self.continuous_scanX_but, 0, 0, 1, 1)

        self.Step_Step_but = QPushButton(self.page_calib)
        self.Step_Step_but.setObjectName(u"Step_Step_but")
        sizePolicy3.setHeightForWidth(self.Step_Step_but.sizePolicy().hasHeightForWidth())
        self.Step_Step_but.setSizePolicy(sizePolicy3)

        self.gridLayout_4.addWidget(self.Step_Step_but, 2, 0, 1, 1)

        self.Calib_Reset_Scan_but = QPushButton(self.page_calib)
        self.Calib_Reset_Scan_but.setObjectName(u"Calib_Reset_Scan_but")
        sizePolicy3.setHeightForWidth(self.Calib_Reset_Scan_but.sizePolicy().hasHeightForWidth())
        self.Calib_Reset_Scan_but.setSizePolicy(sizePolicy3)

        self.gridLayout_4.addWidget(self.Calib_Reset_Scan_but, 2, 1, 1, 1)


        self.verticalLayout_8.addLayout(self.gridLayout_4)

        self.Vel_Report = QPushButton(self.page_calib)
        self.Vel_Report.setObjectName(u"Vel_Report")
        sizePolicy3.setHeightForWidth(self.Vel_Report.sizePolicy().hasHeightForWidth())
        self.Vel_Report.setSizePolicy(sizePolicy3)

        self.verticalLayout_8.addWidget(self.Vel_Report)

        self.verticalSpacer_9 = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.verticalLayout_8.addItem(self.verticalSpacer_9)


        self.horizontalLayout_17.addLayout(self.verticalLayout_8)

        self.horizontalSpacer_9 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_17.addItem(self.horizontalSpacer_9)

        self.verticalLayout_9 = QVBoxLayout()
        self.verticalLayout_9.setSpacing(10)
        self.verticalLayout_9.setObjectName(u"verticalLayout_9")
        self.verticalLayout_9.setContentsMargins(10, 10, 10, 10)
        self.verticalSpacer_8 = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.verticalLayout_9.addItem(self.verticalSpacer_8)

        self.Layout_table_Scan = QVBoxLayout()
        self.Layout_table_Scan.setSpacing(5)
        self.Layout_table_Scan.setObjectName(u"Layout_table_Scan")
        self.Layout_table_Scan.setSizeConstraint(QLayout.SetMaximumSize)
        self.Layout_table_Scan.setContentsMargins(5, 5, 5, 5)

        self.verticalLayout_9.addLayout(self.Layout_table_Scan)

        self.verticalSpacer_7 = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.verticalLayout_9.addItem(self.verticalSpacer_7)


        self.horizontalLayout_17.addLayout(self.verticalLayout_9)

        self.horizontalSpacer_10 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_17.addItem(self.horizontalSpacer_10)


        self.verticalLayout_10.addLayout(self.horizontalLayout_17)


        self.verticalLayout_11.addLayout(self.verticalLayout_10)

        self.pages.addWidget(self.page_calib)
        self.page_cam = QWidget()
        self.page_cam.setObjectName(u"page_cam")
        self.page_cam.setStyleSheet(u"font: 12pt")
        self.verticalLayout_12 = QVBoxLayout(self.page_cam)
        self.verticalLayout_12.setObjectName(u"verticalLayout_12")
        self.verticalLayout_26 = QVBoxLayout()
        self.verticalLayout_26.setObjectName(u"verticalLayout_26")
        self.label_98 = QLabel(self.page_cam)
        self.label_98.setObjectName(u"label_98")
        sizePolicy.setHeightForWidth(self.label_98.sizePolicy().hasHeightForWidth())
        self.label_98.setSizePolicy(sizePolicy)

        self.verticalLayout_26.addWidget(self.label_98)

        self.horizontalLayout_43 = QHBoxLayout()
        self.horizontalLayout_43.setObjectName(u"horizontalLayout_43")
        self.verticalLayout_27 = QVBoxLayout()
        self.verticalLayout_27.setObjectName(u"verticalLayout_27")
        self.verticalLayout_27.setContentsMargins(10, 10, 10, 10)
        self.line_35 = QFrame(self.page_cam)
        self.line_35.setObjectName(u"line_35")
        self.line_35.setFrameShape(QFrame.HLine)
        self.line_35.setFrameShadow(QFrame.Sunken)

        self.verticalLayout_27.addWidget(self.line_35)

        self.label_99 = QLabel(self.page_cam)
        self.label_99.setObjectName(u"label_99")
        sizePolicy3.setHeightForWidth(self.label_99.sizePolicy().hasHeightForWidth())
        self.label_99.setSizePolicy(sizePolicy3)

        self.verticalLayout_27.addWidget(self.label_99)

        self.line_36 = QFrame(self.page_cam)
        self.line_36.setObjectName(u"line_36")
        self.line_36.setFrameShape(QFrame.HLine)
        self.line_36.setFrameShadow(QFrame.Sunken)

        self.verticalLayout_27.addWidget(self.line_36)

        self.gridLayout_13 = QGridLayout()
        self.gridLayout_13.setObjectName(u"gridLayout_13")
        self.gridLayout_13.setContentsMargins(20, 20, 20, 10)
        self.but_start_cam_2 = QPushButton(self.page_cam)
        self.but_start_cam_2.setObjectName(u"but_start_cam_2")

        self.gridLayout_13.addWidget(self.but_start_cam_2, 7, 0, 1, 1)

        self.line_resolution_cam_h_2 = QLineEdit(self.page_cam)
        self.line_resolution_cam_h_2.setObjectName(u"line_resolution_cam_h_2")
        sizePolicy3.setHeightForWidth(self.line_resolution_cam_h_2.sizePolicy().hasHeightForWidth())
        self.line_resolution_cam_h_2.setSizePolicy(sizePolicy3)

        self.gridLayout_13.addWidget(self.line_resolution_cam_h_2, 1, 1, 1, 1)

        self.but_stop_cam_2 = QPushButton(self.page_cam)
        self.but_stop_cam_2.setObjectName(u"but_stop_cam_2")

        self.gridLayout_13.addWidget(self.but_stop_cam_2, 8, 0, 1, 1)

        self.label_100 = QLabel(self.page_cam)
        self.label_100.setObjectName(u"label_100")

        self.gridLayout_13.addWidget(self.label_100, 1, 0, 1, 1)

        self.line_exposure_cam_2 = QLineEdit(self.page_cam)
        self.line_exposure_cam_2.setObjectName(u"line_exposure_cam_2")
        sizePolicy3.setHeightForWidth(self.line_exposure_cam_2.sizePolicy().hasHeightForWidth())
        self.line_exposure_cam_2.setSizePolicy(sizePolicy3)

        self.gridLayout_13.addWidget(self.line_exposure_cam_2, 3, 1, 1, 1)

        self.line_trigger_frame_cam_2 = QLineEdit(self.page_cam)
        self.line_trigger_frame_cam_2.setObjectName(u"line_trigger_frame_cam_2")
        sizePolicy3.setHeightForWidth(self.line_trigger_frame_cam_2.sizePolicy().hasHeightForWidth())
        self.line_trigger_frame_cam_2.setSizePolicy(sizePolicy3)

        self.gridLayout_13.addWidget(self.line_trigger_frame_cam_2, 4, 1, 1, 1)

        self.line_frame_rate_cam_2 = QLineEdit(self.page_cam)
        self.line_frame_rate_cam_2.setObjectName(u"line_frame_rate_cam_2")
        sizePolicy3.setHeightForWidth(self.line_frame_rate_cam_2.sizePolicy().hasHeightForWidth())
        self.line_frame_rate_cam_2.setSizePolicy(sizePolicy3)

        self.gridLayout_13.addWidget(self.line_frame_rate_cam_2, 2, 1, 1, 1)

        self.label_101 = QLabel(self.page_cam)
        self.label_101.setObjectName(u"label_101")

        self.gridLayout_13.addWidget(self.label_101, 0, 0, 1, 1)

        self.label_102 = QLabel(self.page_cam)
        self.label_102.setObjectName(u"label_102")

        self.gridLayout_13.addWidget(self.label_102, 2, 0, 1, 1)

        self.label_103 = QLabel(self.page_cam)
        self.label_103.setObjectName(u"label_103")

        self.gridLayout_13.addWidget(self.label_103, 4, 0, 1, 1)

        self.line_resolution_cam_w_2 = QLineEdit(self.page_cam)
        self.line_resolution_cam_w_2.setObjectName(u"line_resolution_cam_w_2")
        sizePolicy3.setHeightForWidth(self.line_resolution_cam_w_2.sizePolicy().hasHeightForWidth())
        self.line_resolution_cam_w_2.setSizePolicy(sizePolicy3)

        self.gridLayout_13.addWidget(self.line_resolution_cam_w_2, 0, 1, 1, 1)

        self.label_104 = QLabel(self.page_cam)
        self.label_104.setObjectName(u"label_104")

        self.gridLayout_13.addWidget(self.label_104, 3, 0, 1, 1)

        self.label_105 = QLabel(self.page_cam)
        self.label_105.setObjectName(u"label_105")

        self.gridLayout_13.addWidget(self.label_105, 5, 0, 1, 1)

        self.line_directory_cam_2 = QLineEdit(self.page_cam)
        self.line_directory_cam_2.setObjectName(u"line_directory_cam_2")
        sizePolicy3.setHeightForWidth(self.line_directory_cam_2.sizePolicy().hasHeightForWidth())
        self.line_directory_cam_2.setSizePolicy(sizePolicy3)

        self.gridLayout_13.addWidget(self.line_directory_cam_2, 5, 1, 1, 1)

        self.label_106 = QLabel(self.page_cam)
        self.label_106.setObjectName(u"label_106")

        self.gridLayout_13.addWidget(self.label_106, 6, 0, 1, 1)

        self.line_partition_count_cam_2 = QLineEdit(self.page_cam)
        self.line_partition_count_cam_2.setObjectName(u"line_partition_count_cam_2")
        sizePolicy3.setHeightForWidth(self.line_partition_count_cam_2.sizePolicy().hasHeightForWidth())
        self.line_partition_count_cam_2.setSizePolicy(sizePolicy3)

        self.gridLayout_13.addWidget(self.line_partition_count_cam_2, 6, 1, 1, 1)

        self.but_record_cam = QPushButton(self.page_cam)
        self.but_record_cam.setObjectName(u"but_record_cam")

        self.gridLayout_13.addWidget(self.but_record_cam, 7, 1, 1, 1)

        self.but_trigger_cam = QPushButton(self.page_cam)
        self.but_trigger_cam.setObjectName(u"but_trigger_cam")

        self.gridLayout_13.addWidget(self.but_trigger_cam, 8, 1, 1, 1)


        self.verticalLayout_27.addLayout(self.gridLayout_13)

        self.verticalSpacer_20 = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.verticalLayout_27.addItem(self.verticalSpacer_20)


        self.horizontalLayout_43.addLayout(self.verticalLayout_27)

        self.verticalLayout_28 = QVBoxLayout()
        self.verticalLayout_28.setObjectName(u"verticalLayout_28")
        self.horizontalSpacer_21 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.verticalLayout_28.addItem(self.horizontalSpacer_21)

        self.Layout_Cam = QHBoxLayout()
        self.Layout_Cam.setObjectName(u"Layout_Cam")
        self.horizontalSpacer_22 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.Layout_Cam.addItem(self.horizontalSpacer_22)

        self.verticalLayout_29 = QVBoxLayout()
        self.verticalLayout_29.setObjectName(u"verticalLayout_29")
        self.verticalLayout_29.setContentsMargins(10, 10, 10, 10)
        self.image_label = QLabel(self.page_cam)
        self.image_label.setObjectName(u"image_label")

        self.verticalLayout_29.addWidget(self.image_label)


        self.Layout_Cam.addLayout(self.verticalLayout_29)

        self.horizontalSpacer_23 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.Layout_Cam.addItem(self.horizontalSpacer_23)


        self.verticalLayout_28.addLayout(self.Layout_Cam)

        self.horizontalSpacer_24 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.verticalLayout_28.addItem(self.horizontalSpacer_24)


        self.horizontalLayout_43.addLayout(self.verticalLayout_28)


        self.verticalLayout_26.addLayout(self.horizontalLayout_43)


        self.verticalLayout_12.addLayout(self.verticalLayout_26)

        self.pages.addWidget(self.page_cam)

        self.verticalLayout_13.addWidget(self.pages)


        self.retranslateUi(MainPages)

        self.pages.setCurrentIndex(3)


        QMetaObject.connectSlotsByName(MainPages)
    # setupUi

    def retranslateUi(self, MainPages):
        MainPages.setWindowTitle(QCoreApplication.translate("MainPages", u"Form", None))
        self.label_27.setText(QCoreApplication.translate("MainPages", u"<html><head/><body><p align=\"center\"><span style=\" font-size:16pt; font-weight:700;\">Signal Acquisition</span></p></body></html>", None))
        self.label_14.setText(QCoreApplication.translate("MainPages", u"<html><head/><body><p align=\"center\"><span style=\" font-size:14pt; font-weight:700;\">Parameters</span></p></body></html>", None))
        self.label_15.setText(QCoreApplication.translate("MainPages", u"<html><head/><body><p><span style=\" font-size:12pt;\">Acquisition Frequency [Hz]:</span></p></body></html>", None))
        self.lineEdit_Low_Freq.setText(QCoreApplication.translate("MainPages", u"7000", None))
        self.label_18.setText(QCoreApplication.translate("MainPages", u"<html><head/><body><p><span style=\" font-size:12pt;\">Bandpass Filter H [Hz]:</span></p></body></html>", None))
        self.label_16.setText(QCoreApplication.translate("MainPages", u"<html><head/><body><p><span style=\" font-size:12pt;\">Number of Samples:</span></p></body></html>", None))
        self.lineEdit_High_Freq.setText(QCoreApplication.translate("MainPages", u"100000", None))
        self.label_17.setText(QCoreApplication.translate("MainPages", u"<html><head/><body><p><span style=\" font-size:12pt;\">Bandpass Filter L [Hz]:</span></p></body></html>", None))
        self.lineEdit_Laser.setText(QCoreApplication.translate("MainPages", u"2000000", None))
        self.lineEdit_number_samples.setText(QCoreApplication.translate("MainPages", u"16384", None))
        self.label_19.setText(QCoreApplication.translate("MainPages", u"<html><head/><body><p><span style=\" font-size:12pt;\">Average FFT</span></p></body></html>", None))
        self.lineEdit_Avg_FFT.setText(QCoreApplication.translate("MainPages", u"100", None))
        self.DAQ_connect_but.setText(QCoreApplication.translate("MainPages", u"Start Acquisition", None))
        self.Stop_DAQ_but.setText(QCoreApplication.translate("MainPages", u"Stop Acquisition", None))
        self.label_42.setText(QCoreApplication.translate("MainPages", u"<html><head/><body><p><span style=\" font-size:12pt;\">Moment:</span></p></body></html>", None))
        self.Moment_val.setText(QCoreApplication.translate("MainPages", u"00.00", None))
        self.label_43.setText(QCoreApplication.translate("MainPages", u"<html><head/><body><p><span style=\" font-size:12pt;\">Max Amp Peak:</span></p></body></html>", None))
        self.Max_Peak_val.setText(QCoreApplication.translate("MainPages", u"00.00", None))
        self.label_22.setText(QCoreApplication.translate("MainPages", u"<html><head/><body><p align=\"center\"><span style=\" font-size:14pt; font-weight:700;\">Particle Search</span></p></body></html>", None))
        self.lineEdit_Directory.setText(QCoreApplication.translate("MainPages", u"./Temporal_Files/Particles/", None))
        self.label_20.setText(QCoreApplication.translate("MainPages", u"<html><head/><body><p><span style=\" font-size:12pt;\">Directory:</span></p></body></html>", None))
        self.lineEdit_Name_Files.setText(QCoreApplication.translate("MainPages", u"HFocusing", None))
        self.label_21.setText(QCoreApplication.translate("MainPages", u"<html><head/><body><p><span style=\" font-size:12pt;\">Name Files:</span></p></body></html>", None))
        self.label_29.setText(QCoreApplication.translate("MainPages", u"<html><head/><body><p><span style=\" font-size:12pt;\">Amp Peak:</span></p></body></html>", None))
        self.lineEdit_amp_peak.setText(QCoreApplication.translate("MainPages", u"0.155", None))
        self.Search_Particles_but.setText(QCoreApplication.translate("MainPages", u"Start search", None))
        self.Stop_Search_P_but.setText(QCoreApplication.translate("MainPages", u"Stop Search", None))
        self.label_23.setText(QCoreApplication.translate("MainPages", u"<html><head/><body><p align=\"center\"><span style=\" font-size:14pt; font-weight:700;\">Zaber Calibration</span></p></body></html>", None))
        self.label_24.setText(QCoreApplication.translate("MainPages", u"<html><head/><body><p><span style=\" font-size:12pt;\">Start Pos [mm]:</span></p></body></html>", None))
        self.lineEdit_steps_zab.setText(QCoreApplication.translate("MainPages", u"15", None))
        self.label_26.setText(QCoreApplication.translate("MainPages", u"<html><head/><body><p><span style=\" font-size:12pt;\">Steps:</span></p></body></html>", None))
        self.label_25.setText(QCoreApplication.translate("MainPages", u"<html><head/><body><p><span style=\" font-size:12pt;\">End Pos [mm]:</span></p></body></html>", None))
        self.lineEdit_start_zab_pos.setText(QCoreApplication.translate("MainPages", u"37500", None))
        self.lineEdit_end_zab_pos.setText(QCoreApplication.translate("MainPages", u"39500", None))
        self.label_28.setText(QCoreApplication.translate("MainPages", u"<html><head/><body><p><span style=\" font-size:12pt;\"> Samples to check:</span></p></body></html>", None))
        self.lineEdit_zab_samplesCheck.setText(QCoreApplication.translate("MainPages", u"100", None))
        self.Start_Zaber_Calib_but.setText(QCoreApplication.translate("MainPages", u"Start Calibration", None))
        self.label_50.setText(QCoreApplication.translate("MainPages", u"<html><head/><body><p align=\"center\"><span style=\" font-size:16pt; font-weight:700;\">Data Analysis</span></p></body></html>", None))
        self.Update_but_Server.setText(QCoreApplication.translate("MainPages", u"PushButton", None))
        self.label_p3.setText(QCoreApplication.translate("MainPages", u"<html><head/><body><p align=\"center\"><span style=\" font-size:16pt; font-weight:700;\">Zaber connection</span></p></body></html>", None))
        self.zabercon_p3.setText(QCoreApplication.translate("MainPages", u"Detect", None))
        self.n_devices_p3.setText(QCoreApplication.translate("MainPages", u"<html><head/><body><p># Devices</p></body></html>", None))
        self.check_p3.setText("")
        self.globalhome_p3.setText(QCoreApplication.translate("MainPages", u"Reset Connection", None))
        self.home1_p3.setText(QCoreApplication.translate("MainPages", u"Home 1", None))
        self.home2_p3.setText(QCoreApplication.translate("MainPages", u"Home 2", None))
        self.home3_p3.setText(QCoreApplication.translate("MainPages", u"Home 3", None))
        self.label.setText(QCoreApplication.translate("MainPages", u"<html><head/><body><p align=\"center\"><span style=\" font-size:14pt; font-weight:700;\">Zaber 1</span></p></body></html>", None))
        self.z1_Relative.setText(QCoreApplication.translate("MainPages", u"Relative", None))
        self.z1_Absolute.setText(QCoreApplication.translate("MainPages", u"Absolute", None))
        self.z1_cm.setText(QCoreApplication.translate("MainPages", u"cm", None))
        self.z1_mm.setText(QCoreApplication.translate("MainPages", u"mm", None))
        self.z1_um.setText(QCoreApplication.translate("MainPages", u"um", None))
        self.label_4.setText(QCoreApplication.translate("MainPages", u"<html><head/><body><p>Distance: </p></body></html>", None))
        self.z1_set.setText(QCoreApplication.translate("MainPages", u"Set", None))
        self.label_7.setText(QCoreApplication.translate("MainPages", u"<html><head/><body><p align=\"center\">Move at constant speed</p></body></html>", None))
        self.z1speed.setText(QCoreApplication.translate("MainPages", u"0.2", None))
        self.label_10.setText(QCoreApplication.translate("MainPages", u"<html><head/><body><p>mm/s</p></body></html>", None))
        self.z1_left.setText(QCoreApplication.translate("MainPages", u"<<", None))
        self.z1_right.setText(QCoreApplication.translate("MainPages", u">>", None))
        self.z1_Stop.setText(QCoreApplication.translate("MainPages", u"STOP", None))
        self.label_2.setText(QCoreApplication.translate("MainPages", u"<html><head/><body><p align=\"center\"><span style=\" font-size:14pt; font-weight:700;\">Zaber 2</span></p></body></html>", None))
        self.z2_Relative.setText(QCoreApplication.translate("MainPages", u"Relative", None))
        self.z2_Absolute.setText(QCoreApplication.translate("MainPages", u"Absolute", None))
        self.z2_cm.setText(QCoreApplication.translate("MainPages", u"cm", None))
        self.z2_mm.setText(QCoreApplication.translate("MainPages", u"mm", None))
        self.z2_um.setText(QCoreApplication.translate("MainPages", u"um", None))
        self.label_5.setText(QCoreApplication.translate("MainPages", u"<html><head/><body><p>Distance</p></body></html>", None))
        self.z2_Set.setText(QCoreApplication.translate("MainPages", u"Set", None))
        self.label_8.setText(QCoreApplication.translate("MainPages", u"<html><head/><body><p align=\"center\">Move at constant speed</p></body></html>", None))
        self.z2speed.setText(QCoreApplication.translate("MainPages", u"0.2", None))
        self.label_11.setText(QCoreApplication.translate("MainPages", u"<html><head/><body><p>mm/s</p></body></html>", None))
        self.z2_left.setText(QCoreApplication.translate("MainPages", u"<<", None))
        self.z2_right.setText(QCoreApplication.translate("MainPages", u">>", None))
        self.z2_Stop.setText(QCoreApplication.translate("MainPages", u"STOP", None))
        self.label_3.setText(QCoreApplication.translate("MainPages", u"<html><head/><body><p align=\"center\"><span style=\" font-size:14pt; font-weight:700;\">Zaber 3</span></p></body></html>", None))
        self.z3_Relative.setText(QCoreApplication.translate("MainPages", u"Relative", None))
        self.z3_Absolute.setText(QCoreApplication.translate("MainPages", u"Absolute", None))
        self.z3_cm.setText(QCoreApplication.translate("MainPages", u"cm", None))
        self.z3_mm.setText(QCoreApplication.translate("MainPages", u"mm", None))
        self.z3_um.setText(QCoreApplication.translate("MainPages", u"um", None))
        self.label_6.setText(QCoreApplication.translate("MainPages", u"<html><head/><body><p>Distance</p></body></html>", None))
        self.z3_Set.setText(QCoreApplication.translate("MainPages", u"Set", None))
        self.label_9.setText(QCoreApplication.translate("MainPages", u"<html><head/><body><p align=\"center\">Move at constant speed</p></body></html>", None))
        self.z3speed.setText(QCoreApplication.translate("MainPages", u"0.2", None))
        self.label_12.setText(QCoreApplication.translate("MainPages", u"<html><head/><body><p>mm/s</p></body></html>", None))
        self.z3_left.setText(QCoreApplication.translate("MainPages", u"<<", None))
        self.z3_right.setText(QCoreApplication.translate("MainPages", u">>", None))
        self.z3_Stop.setText(QCoreApplication.translate("MainPages", u"STOP", None))
        self.Keyboard_p3.setText(QCoreApplication.translate("MainPages", u"Keyboard", None))
        self.label_44.setText(QCoreApplication.translate("MainPages", u"<html><head/><body><p align=\"center\"><span style=\" font-size:16pt; font-weight:700;\">Flow Velocity Profile</span></p></body></html>", None))
        self.label_45.setText(QCoreApplication.translate("MainPages", u"<html><head/><body><p align=\"center\"><span style=\" font-size:14pt; font-weight:700;\">Calibration</span></p></body></html>", None))
        self.Vel_Start_Calib.setText(QCoreApplication.translate("MainPages", u"START", None))
        self.Vel_Steps_Calib.setText(QCoreApplication.translate("MainPages", u"10", None))
        self.Vel_Limit_X1.setText(QCoreApplication.translate("MainPages", u"90000", None))
        self.label_31.setText(QCoreApplication.translate("MainPages", u"# Steps:", None))
        self.Vel_Manual_X1.setText(QCoreApplication.translate("MainPages", u"2000", None))
        self.label_30.setText(QCoreApplication.translate("MainPages", u"Manual X1 [um]:", None))
        self.label_47.setText(QCoreApplication.translate("MainPages", u"Threshold Amp Peak:", None))
        self.label_48.setText(QCoreApplication.translate("MainPages", u"Limit X2:", None))
        self.label_46.setText(QCoreApplication.translate("MainPages", u"Manual X2 [um]:", None))
        self.label_32.setText(QCoreApplication.translate("MainPages", u"Limit X1:", None))
        self.Vel_Manual_X2.setText(QCoreApplication.translate("MainPages", u"7000", None))
        self.Vel_Threshold_Peak.setText(QCoreApplication.translate("MainPages", u"30", None))
        self.Vel_Limit_X2.setText(QCoreApplication.translate("MainPages", u"90000", None))
        self.label_49.setText(QCoreApplication.translate("MainPages", u"<html><head/><body><p align=\"center\"><span style=\" font-size:14pt; font-weight:700;\">Scanning</span></p></body></html>", None))
        self.label_41.setText(QCoreApplication.translate("MainPages", u"Speed [um/s]: ", None))
        self.lineEdit_speed_ums.setText(QCoreApplication.translate("MainPages", u"100", None))
        self.lineEdit_y2_scan.setText(QCoreApplication.translate("MainPages", u"200", None))
        self.label_35.setText(QCoreApplication.translate("MainPages", u"Y1 [um]:", None))
        self.label_34.setText(QCoreApplication.translate("MainPages", u"X2 [um]:", None))
        self.lineEdit_x1_scan.setText(QCoreApplication.translate("MainPages", u"2900", None))
        self.lineEdit_y1_scan.setText(QCoreApplication.translate("MainPages", u"0", None))
        self.label_33.setText(QCoreApplication.translate("MainPages", u"X1 [um]:", None))
        self.label_38.setText(QCoreApplication.translate("MainPages", u"Size Pixels [um]:", None))
        self.label_36.setText(QCoreApplication.translate("MainPages", u"Y2 [um]:", None))
        self.lineEdit_Pixel_size.setText(QCoreApplication.translate("MainPages", u"50", None))
        self.lineEdit_x2_scan.setText(QCoreApplication.translate("MainPages", u"4100", None))
        self.Stop_Y_but.setText(QCoreApplication.translate("MainPages", u"Stop Y", None))
        self.Stop_x_but.setText(QCoreApplication.translate("MainPages", u"Stop X", None))
        self.continuous_scanY_but.setText(QCoreApplication.translate("MainPages", u"Continuous Scan Y", None))
        self.continuous_scanX_but.setText(QCoreApplication.translate("MainPages", u"Continuos Scan X", None))
        self.Step_Step_but.setText(QCoreApplication.translate("MainPages", u"Stop DAQ", None))
        self.Calib_Reset_Scan_but.setText(QCoreApplication.translate("MainPages", u"Reset Grid", None))
        self.Vel_Report.setText(QCoreApplication.translate("MainPages", u"Velocity Profile Report", None))
        self.label_98.setText(QCoreApplication.translate("MainPages", u"<html><head/><body><p align=\"center\"><span style=\" font-size:16pt; font-weight:700;\">Camera Settings</span></p></body></html>", None))
        self.label_99.setText(QCoreApplication.translate("MainPages", u"<html><head/><body><p align=\"center\"><span style=\" font-size:14pt; font-weight:700;\">Parameters</span></p></body></html>", None))
        self.but_start_cam_2.setText(QCoreApplication.translate("MainPages", u"Live Image", None))
        self.line_resolution_cam_h_2.setText(QCoreApplication.translate("MainPages", u"128", None))
        self.but_stop_cam_2.setText(QCoreApplication.translate("MainPages", u"Stop", None))
        self.label_100.setText(QCoreApplication.translate("MainPages", u"Resolution H [px]", None))
        self.line_exposure_cam_2.setText(QCoreApplication.translate("MainPages", u"70", None))
        self.line_trigger_frame_cam_2.setText(QCoreApplication.translate("MainPages", u"50", None))
        self.line_frame_rate_cam_2.setText(QCoreApplication.translate("MainPages", u"8000", None))
        self.label_101.setText(QCoreApplication.translate("MainPages", u"Resolution W [px]:", None))
        self.label_102.setText(QCoreApplication.translate("MainPages", u"Frame Rate [fps]:", None))
        self.label_103.setText(QCoreApplication.translate("MainPages", u"Trigger Frames:", None))
        self.line_resolution_cam_w_2.setText(QCoreApplication.translate("MainPages", u"640", None))
        self.label_104.setText(QCoreApplication.translate("MainPages", u"Exposure Time [us]", None))
        self.label_105.setText(QCoreApplication.translate("MainPages", u"Directory:", None))
        self.line_directory_cam_2.setText(QCoreApplication.translate("MainPages", u"./Temporal_Files/Images/", None))
        self.label_106.setText(QCoreApplication.translate("MainPages", u"Partition Count:", None))
        self.line_partition_count_cam_2.setText(QCoreApplication.translate("MainPages", u"1", None))
        self.but_record_cam.setText(QCoreApplication.translate("MainPages", u"Record/Abort", None))
        self.but_trigger_cam.setText(QCoreApplication.translate("MainPages", u"Trigger", None))
        self.image_label.setText("")
    # retranslateUi

