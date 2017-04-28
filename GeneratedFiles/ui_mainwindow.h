/********************************************************************************
** Form generated from reading UI file 'mainwindow.ui'
**
** Created by: Qt User Interface Compiler version 5.7.0
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_MAINWINDOW_H
#define UI_MAINWINDOW_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QTabWidget>
#include <QtWidgets/QToolBar>
#include <QtWidgets/QWidget>
#include <src/gui/InFlowWidget.hpp>
#include "src/gui/fluidglwidget.h"

QT_BEGIN_NAMESPACE

class Ui_MainWindow
{
public:
    QWidget *centralWidget;
    FluidGLWidget *widget;
    QPushButton *pause_button;
    QPushButton *start_button;
    QPushButton *stop_button;
    QTabWidget *tabWidget;
    InFlowWidget *inflows_tab;
    QWidget *tab_2;
    QMenuBar *menuBar;
    QToolBar *mainToolBar;
    QStatusBar *statusBar;

    void setupUi(QMainWindow *MainWindow)
    {
        if (MainWindow->objectName().isEmpty())
            MainWindow->setObjectName(QStringLiteral("MainWindow"));
        MainWindow->resize(1163, 788);
        centralWidget = new QWidget(MainWindow);
        centralWidget->setObjectName(QStringLiteral("centralWidget"));
        widget = new FluidGLWidget(centralWidget);
        widget->setObjectName(QStringLiteral("widget"));
        widget->setGeometry(QRect(0, 0, 951, 691));
        pause_button = new QPushButton(centralWidget);
        pause_button->setObjectName(QStringLiteral("pause_button"));
        pause_button->setGeometry(QRect(1040, 630, 91, 28));
        start_button = new QPushButton(centralWidget);
        start_button->setObjectName(QStringLiteral("start_button"));
        start_button->setGeometry(QRect(960, 630, 81, 28));
        stop_button = new QPushButton(centralWidget);
        stop_button->setObjectName(QStringLiteral("stop_button"));
        stop_button->setGeometry(QRect(960, 660, 81, 28));
        tabWidget = new QTabWidget(centralWidget);
        tabWidget->setObjectName(QStringLiteral("tabWidget"));
        tabWidget->setGeometry(QRect(960, 0, 201, 631));
        tabWidget->setStyleSheet(QStringLiteral(""));
        inflows_tab = new InFlowWidget();
        inflows_tab->setObjectName(QStringLiteral("inflows_tab"));
        inflows_tab->setStyleSheet(QStringLiteral(""));
        tabWidget->addTab(inflows_tab, QString());
        tab_2 = new QWidget();
        tab_2->setObjectName(QStringLiteral("tab_2"));
        tabWidget->addTab(tab_2, QString());
        MainWindow->setCentralWidget(centralWidget);
        menuBar = new QMenuBar(MainWindow);
        menuBar->setObjectName(QStringLiteral("menuBar"));
        menuBar->setGeometry(QRect(0, 0, 1163, 26));
        MainWindow->setMenuBar(menuBar);
        mainToolBar = new QToolBar(MainWindow);
        mainToolBar->setObjectName(QStringLiteral("mainToolBar"));
        MainWindow->addToolBar(Qt::TopToolBarArea, mainToolBar);
        statusBar = new QStatusBar(MainWindow);
        statusBar->setObjectName(QStringLiteral("statusBar"));
        MainWindow->setStatusBar(statusBar);

        retranslateUi(MainWindow);

        tabWidget->setCurrentIndex(0);


        QMetaObject::connectSlotsByName(MainWindow);
    } // setupUi

    void retranslateUi(QMainWindow *MainWindow)
    {
        MainWindow->setWindowTitle(QApplication::translate("MainWindow", "MainWindow", 0));
        pause_button->setText(QApplication::translate("MainWindow", "Pause", 0));
        start_button->setText(QApplication::translate("MainWindow", "Start", 0));
        stop_button->setText(QApplication::translate("MainWindow", "Stop", 0));
        tabWidget->setTabText(tabWidget->indexOf(inflows_tab), QApplication::translate("MainWindow", "Inflows", 0));
        tabWidget->setTabText(tabWidget->indexOf(tab_2), QApplication::translate("MainWindow", "Tab 2", 0));
    } // retranslateUi

};

namespace Ui {
    class MainWindow: public Ui_MainWindow {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MAINWINDOW_H
