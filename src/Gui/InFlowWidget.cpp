#include "InFlowWidget.hpp"

#include <iostream>

InFlowWidget::InFlowWidget(QWidget * parent) : QWidget(parent) 
{
	ui.setupUi(this);
}

InFlowWidget::~InFlowWidget()
{
	
}

void InFlowWidget::SetInFlows(InFlowData* flows)
{
	flows_ = flows; 
	current_flow_ = 0;
	SetDisplayValues_();
}

void InFlowWidget::SetInflowSize(unsigned size)
{
	 num_flows_ = size; 
	 ui.inflow_selector->setMaximum( size - 1 );
}

void InFlowWidget::on_inflow_selector_valueChanged(int arg1)
{ 
	current_flow_ = arg1;
	SetDisplayValues_();
}

void InFlowWidget::on_u_velocity_combo_box_valueChanged(double arg1)
{
	InFlowData* flows = flows_ + current_flow_;
	if (!flows) return;
	flows->u_vel = arg1;
	emit InflowChanged();
}

void InFlowWidget::on_u_velocity_combo_box_valueChanged(const QString &arg1)
{
	InFlowData* flows = flows_ + current_flow_;
	if (!flows) return;
	flows->u_vel = arg1.toDouble();
	emit InflowChanged();
}

void InFlowWidget::on_v_velocity_combo_box_valueChanged(double arg1)
{
	InFlowData* flows = flows_ + current_flow_;
	if (!flows) return;
	flows->v_vel = arg1;
	emit InflowChanged();
}

void InFlowWidget::on_v_velocity_combo_box_valueChanged(const QString &arg1)
{
	InFlowData* flows = flows_ + current_flow_;
	if (!flows) return;
	flows->v_vel = arg1.toDouble();
	emit InflowChanged();
}

void InFlowWidget::on_density_combo_box_valueChanged(double arg1)
{
	InFlowData* flows = flows_ + current_flow_;
	if (!flows) return;
	flows->density = arg1;
	emit InflowChanged();
}

void InFlowWidget::on_density_combo_box_valueChanged(const QString &arg1)
{
	InFlowData* flows = flows_ + current_flow_;
	if (!flows) return;
	flows->density = arg1.toDouble();
	emit InflowChanged();
}

void InFlowWidget::on_red_combo_box_valueChanged(const QString &arg1)
{
	InFlowData* flows = flows_ + current_flow_;
	if (!flows) return;
	flows->red = arg1.toDouble();
	emit InflowChanged();
}

void InFlowWidget::on_red_combo_box_valueChanged(double arg1) 
{
	InFlowData* flows = flows_ + current_flow_;
	if (!flows) return;
	flows->red = arg1;
	emit InflowChanged();
}

void InFlowWidget::on_green_combo_box_valueChanged(const QString &arg1)
{
	InFlowData* flows = flows_ + current_flow_;
	if (!flows) return;
	flows->green = arg1.toDouble();
	emit InflowChanged();
}

void InFlowWidget::on_green_combo_box_valueChanged(double arg1)
{
	InFlowData* flows = flows_ + current_flow_;
	if (!flows) return;
	flows->green = arg1;
	emit InflowChanged();
}

void InFlowWidget::on_blue_combo_box_valueChanged(const QString &arg1)
{
	InFlowData* flows = flows_ + current_flow_;
	if (!flows) return;
	flows->blue = arg1.toDouble();
	emit InflowChanged();
}

void InFlowWidget::on_blue_combo_box_valueChanged(double arg1)
{
	InFlowData* flows = flows_ + current_flow_;
	if (!flows) return;
	flows->blue = arg1;
	emit InflowChanged();
}

void InFlowWidget::on_x_origin_combo_box_valueChanged(double arg1)
{
	InFlowData* flows = flows_ + current_flow_;
	if (!flows) return;
	flows->x = arg1;
	emit InflowChanged();
}

void InFlowWidget::on_x_origin_combo_box_valueChanged(const QString &arg1)
{
	InFlowData* flows = flows_ + current_flow_;
	if (!flows) return;
	flows->x = arg1.toDouble();
	emit InflowChanged();
}

void InFlowWidget::on_y_origin_combo_box_valueChanged(double arg1)
{
	InFlowData* flows = flows_ + current_flow_;
	if (!flows) return;
	flows->y = arg1;
	emit InflowChanged();
}

void InFlowWidget::on_y_origin_combo_box_valueChanged(const QString &arg1)
{
	InFlowData* flows = flows_ + current_flow_;
	if (!flows) return;
	flows->y = arg1.toDouble();
	emit InflowChanged();
}

void InFlowWidget::on_width_combo_box_valueChanged(double arg1)
{
	InFlowData* flows = flows_ + current_flow_;
	if (!flows) return;
	flows->width = arg1;
	emit InflowChanged();
}

void InFlowWidget::on_width_combo_box_valueChanged(const QString &arg1)
{
	InFlowData* flows = flows_ + current_flow_;
	if (!flows) return;
	flows->width = arg1.toDouble();
	emit InflowChanged();
}

void InFlowWidget::on_height_combo_box_valueChanged(double arg1)
{
	InFlowData* flows = flows_ + current_flow_;
	if (!flows) return;
	flows->height= arg1;
	emit InflowChanged();
}

void InFlowWidget::on_height_combo_box_valueChanged(const QString &arg1)
{
	InFlowData* flows = flows_ + current_flow_;
	if (!flows) return;
	flows->height = arg1.toDouble();
	emit InflowChanged();
}

void InFlowWidget::SetDisplayValues_()
{
	if (!flows_) return;
	//dont go off firing signals for display changes
	this->blockSignals(true);

	InFlowData* flows = flows_ + current_flow_;

	if (num_flows_)
	{
		ui.u_velocity_combo_box->setValue((*flows).u_vel);
		ui.v_velocity_combo_box->setValue((*flows).v_vel);
		ui.density_combo_box->setValue((*flows).density);

		ui.red_combo_box->setValue((*flows).red);
		ui.blue_combo_box->setValue((*flows).blue);
		ui.green_combo_box->setValue((*flows).green);

		ui.x_origin_combo_box->setValue((*flows).x);
		ui.y_origin_combo_box->setValue((*flows).y);
		ui.width_combo_box->setValue((*flows).width);
		ui.height_combo_box->setValue((*flows).height);
	}

	this->blockSignals(false);
}
