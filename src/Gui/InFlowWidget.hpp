#pragma once
#ifndef INFLOWWIDGET_HPP
#define INFLOWWIDGET_HPP

#include "src/Gui/InFlowWidget.hpp"
#include "src/Fluid/InFlowData.h"
#include "ui_inflowwidget.h"

class FluidSolver2D;

#include <QWidget>
class InFlowWidget : public QWidget {
	Q_OBJECT

public:
	InFlowWidget(QWidget * parent = Q_NULLPTR);
	~InFlowWidget();

	void SetFluidSolver( FluidSolver2D* solver );

signals:
	void InflowChanged(); // used to trigger a reupload of  the inflow parameters

private slots:
	void on_inflow_selector_valueChanged(int arg1);

	void on_u_velocity_combo_box_valueChanged(double arg1);
	void on_u_velocity_combo_box_valueChanged(const QString &arg1);

	void on_v_velocity_combo_box_valueChanged( double arg1 );
	void on_v_velocity_combo_box_valueChanged(const QString &arg1);

	void on_density_combo_box_valueChanged(double arg1);
	void on_density_combo_box_valueChanged(const QString &arg1);

	void on_red_combo_box_valueChanged(const QString &arg1);
	void on_red_combo_box_valueChanged(double arg1);

	void on_green_combo_box_valueChanged(const QString &arg1);
	void on_green_combo_box_valueChanged(double arg1);

	void on_blue_combo_box_valueChanged(const QString &arg1);
	void on_blue_combo_box_valueChanged(double arg1);

	void on_x_origin_combo_box_valueChanged(double arg1);
	void on_x_origin_combo_box_valueChanged(const QString &arg1);

	void on_y_origin_combo_box_valueChanged( double arg1 );
	void on_y_origin_combo_box_valueChanged(const QString &arg1);

	void on_radius_combo_box_valueChanged(double arg1);
	void on_radius_combo_box_valueChanged(const QString &arg1);

	void on_addInFlowButton_pressed();
	void on_removeInFlowButton_pressed();

private:
	void SetDisplayValues_();

private:
	Ui::InFlowWidget ui;
	FluidSolver2D* solver_ = 0;
	unsigned current_flow_ = 0;
};

#endif // INFLOWWIDGET_HPP