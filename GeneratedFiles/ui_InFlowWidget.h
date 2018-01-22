/********************************************************************************
** Form generated from reading UI file 'InFlowWidget.ui'
**
** Created by: Qt User Interface Compiler version 5.7.0
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_INFLOWWIDGET_H
#define UI_INFLOWWIDGET_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QDoubleSpinBox>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLabel>
#include <QtWidgets/QSpinBox>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_InFlowWidget
{
public:
    QDoubleSpinBox *u_velocity_combo_box;
    QDoubleSpinBox *v_velocity_combo_box;
    QDoubleSpinBox *density_combo_box;
    QDoubleSpinBox *red_combo_box;
    QDoubleSpinBox *green_combo_box;
    QDoubleSpinBox *blue_combo_box;
    QLabel *label;
    QLabel *label_2;
    QLabel *label_3;
    QLabel *label_4;
    QLabel *label_5;
    QLabel *label_6;
    QLabel *label_7;
    QDoubleSpinBox *y_origin_combo_box;
    QDoubleSpinBox *x_origin_combo_box;
    QDoubleSpinBox *radius_combo_box;
    QLabel *label_9;
    QLabel *label_10;
    QLabel *label_11;
    QLabel *label_12;
    QLabel *label_13;
    QSpinBox *inflow_selector;

    void setupUi(QWidget *InFlowWidget)
    {
        if (InFlowWidget->objectName().isEmpty())
            InFlowWidget->setObjectName(QStringLiteral("InFlowWidget"));
        InFlowWidget->resize(280, 574);
        InFlowWidget->setWindowOpacity(0);
        u_velocity_combo_box = new QDoubleSpinBox(InFlowWidget);
        u_velocity_combo_box->setObjectName(QStringLiteral("u_velocity_combo_box"));
        u_velocity_combo_box->setGeometry(QRect(10, 130, 71, 20));
        u_velocity_combo_box->setMinimum(-1000);
        u_velocity_combo_box->setMaximum(1000);
        v_velocity_combo_box = new QDoubleSpinBox(InFlowWidget);
        v_velocity_combo_box->setObjectName(QStringLiteral("v_velocity_combo_box"));
        v_velocity_combo_box->setGeometry(QRect(10, 170, 71, 20));
        v_velocity_combo_box->setMinimum(-1000);
        v_velocity_combo_box->setMaximum(1000);
        density_combo_box = new QDoubleSpinBox(InFlowWidget);
        density_combo_box->setObjectName(QStringLiteral("density_combo_box"));
        density_combo_box->setGeometry(QRect(10, 210, 71, 20));
        density_combo_box->setMaximum(1);
        density_combo_box->setSingleStep(0.01);
        red_combo_box = new QDoubleSpinBox(InFlowWidget);
        red_combo_box->setObjectName(QStringLiteral("red_combo_box"));
        red_combo_box->setGeometry(QRect(10, 250, 71, 20));
        red_combo_box->setMaximum(255);
        green_combo_box = new QDoubleSpinBox(InFlowWidget);
        green_combo_box->setObjectName(QStringLiteral("green_combo_box"));
        green_combo_box->setGeometry(QRect(10, 290, 71, 20));
        green_combo_box->setMaximum(255);
        blue_combo_box = new QDoubleSpinBox(InFlowWidget);
        blue_combo_box->setObjectName(QStringLiteral("blue_combo_box"));
        blue_combo_box->setGeometry(QRect(10, 330, 71, 20));
        blue_combo_box->setMaximum(255);
        label = new QLabel(InFlowWidget);
        label->setObjectName(QStringLiteral("label"));
        label->setGeometry(QRect(100, 130, 101, 16));
        label_2 = new QLabel(InFlowWidget);
        label_2->setObjectName(QStringLiteral("label_2"));
        label_2->setGeometry(QRect(100, 170, 91, 16));
        label_3 = new QLabel(InFlowWidget);
        label_3->setObjectName(QStringLiteral("label_3"));
        label_3->setGeometry(QRect(100, 250, 55, 16));
        label_4 = new QLabel(InFlowWidget);
        label_4->setObjectName(QStringLiteral("label_4"));
        label_4->setGeometry(QRect(100, 210, 55, 16));
        label_5 = new QLabel(InFlowWidget);
        label_5->setObjectName(QStringLiteral("label_5"));
        label_5->setGeometry(QRect(100, 290, 55, 16));
        label_6 = new QLabel(InFlowWidget);
        label_6->setObjectName(QStringLiteral("label_6"));
        label_6->setGeometry(QRect(100, 330, 55, 16));
        label_7 = new QLabel(InFlowWidget);
        label_7->setObjectName(QStringLiteral("label_7"));
        label_7->setGeometry(QRect(100, 490, 55, 16));
        y_origin_combo_box = new QDoubleSpinBox(InFlowWidget);
        y_origin_combo_box->setObjectName(QStringLiteral("y_origin_combo_box"));
        y_origin_combo_box->setGeometry(QRect(10, 450, 71, 20));
        y_origin_combo_box->setMaximum(1);
        y_origin_combo_box->setSingleStep(0.01);
        x_origin_combo_box = new QDoubleSpinBox(InFlowWidget);
        x_origin_combo_box->setObjectName(QStringLiteral("x_origin_combo_box"));
        x_origin_combo_box->setGeometry(QRect(10, 410, 71, 20));
        x_origin_combo_box->setMaximum(1);
        x_origin_combo_box->setSingleStep(0.01);
        radius_combo_box = new QDoubleSpinBox(InFlowWidget);
        radius_combo_box->setObjectName(QStringLiteral("radius_combo_box"));
        radius_combo_box->setGeometry(QRect(10, 490, 71, 20));
        radius_combo_box->setMaximum(1);
        radius_combo_box->setSingleStep(0.01);
        label_9 = new QLabel(InFlowWidget);
        label_9->setObjectName(QStringLiteral("label_9"));
        label_9->setGeometry(QRect(100, 410, 55, 16));
        label_10 = new QLabel(InFlowWidget);
        label_10->setObjectName(QStringLiteral("label_10"));
        label_10->setGeometry(QRect(100, 450, 55, 16));
        label_11 = new QLabel(InFlowWidget);
        label_11->setObjectName(QStringLiteral("label_11"));
        label_11->setGeometry(QRect(10, 90, 161, 20));
        QFont font;
        font.setPointSize(12);
        label_11->setFont(font);
        label_12 = new QLabel(InFlowWidget);
        label_12->setObjectName(QStringLiteral("label_12"));
        label_12->setGeometry(QRect(10, 370, 161, 20));
        label_12->setFont(font);
        label_13 = new QLabel(InFlowWidget);
        label_13->setObjectName(QStringLiteral("label_13"));
        label_13->setGeometry(QRect(10, 20, 161, 20));
        label_13->setFont(font);
        inflow_selector = new QSpinBox(InFlowWidget);
        inflow_selector->setObjectName(QStringLiteral("inflow_selector"));
        inflow_selector->setGeometry(QRect(10, 50, 42, 22));

        retranslateUi(InFlowWidget);

        QMetaObject::connectSlotsByName(InFlowWidget);
    } // setupUi

    void retranslateUi(QWidget *InFlowWidget)
    {
        InFlowWidget->setWindowTitle(QApplication::translate("InFlowWidget", "InFlowWidget", 0));
#ifndef QT_NO_ACCESSIBILITY
        InFlowWidget->setAccessibleName(QApplication::translate("InFlowWidget", "S", 0));
#endif // QT_NO_ACCESSIBILITY
        label->setText(QApplication::translate("InFlowWidget", "X Velocity", 0));
        label_2->setText(QApplication::translate("InFlowWidget", "Y Velocity", 0));
        label_3->setText(QApplication::translate("InFlowWidget", "Red", 0));
        label_4->setText(QApplication::translate("InFlowWidget", "Density", 0));
        label_5->setText(QApplication::translate("InFlowWidget", "Green", 0));
        label_6->setText(QApplication::translate("InFlowWidget", "Blue", 0));
        label_7->setText(QApplication::translate("InFlowWidget", "Radius", 0));
        label_9->setText(QApplication::translate("InFlowWidget", "x0", 0));
        label_10->setText(QApplication::translate("InFlowWidget", "y0", 0));
        label_11->setText(QApplication::translate("InFlowWidget", "Values", 0));
        label_12->setText(QApplication::translate("InFlowWidget", "Size and Position", 0));
        label_13->setText(QApplication::translate("InFlowWidget", "Current InFlow", 0));
    } // retranslateUi

};

namespace Ui {
    class InFlowWidget: public Ui_InFlowWidget {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_INFLOWWIDGET_H
