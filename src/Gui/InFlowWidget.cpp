#include "InFlowWidget.hpp"
#include "../Fluid/fluidsolver2D.h"

#include <iostream>

InFlowWidget::InFlowWidget(QWidget * parent) : QWidget(parent) 
{
	ui.setupUi(this);
}

InFlowWidget::~InFlowWidget()
{
	
}

void InFlowWidget::SetFluidSolver( FluidSolver2D * solver )
{
	solver_ = solver;
	current_flow_ = 0;
	SetDisplayValues_();
	ui.inflow_selector->setMaximum( solver->GetInFlows().size() - 1 );
}

void InFlowWidget::on_inflow_selector_valueChanged(int arg1)
{ 
	current_flow_ = arg1;
	SetDisplayValues_();
}

void InFlowWidget::on_u_velocity_combo_box_valueChanged(double arg1)
{
	InFlowData* flows = solver_->GetInFlows().data() + current_flow_;
	if (!flows) return;
	flows->u_vel = arg1;
	emit InflowChanged();
}

void InFlowWidget::on_u_velocity_combo_box_valueChanged(const QString &arg1)
{
	InFlowData* flows = solver_->GetInFlows().data() + current_flow_;
	if (!flows) return;
	flows->u_vel = arg1.toDouble();
	emit InflowChanged();
}

void InFlowWidget::on_v_velocity_combo_box_valueChanged(double arg1)
{
	InFlowData* flows = solver_->GetInFlows().data() + current_flow_;
	if (!flows) return;
	flows->v_vel = arg1;
	emit InflowChanged();
}

void InFlowWidget::on_v_velocity_combo_box_valueChanged(const QString &arg1)
{
	InFlowData* flows = solver_->GetInFlows().data() + current_flow_;
	if (!flows) return;
	flows->v_vel = arg1.toDouble();
	emit InflowChanged();
}

void InFlowWidget::on_density_combo_box_valueChanged(double arg1)
{
	InFlowData* flows = solver_->GetInFlows().data() + current_flow_;
	if (!flows) return;
	flows->density = arg1;
	emit InflowChanged();
}

void InFlowWidget::on_density_combo_box_valueChanged(const QString &arg1)
{
	InFlowData* flows = solver_->GetInFlows().data() + current_flow_;
	if (!flows) return;
	flows->density = arg1.toDouble();
	emit InflowChanged();
}

void InFlowWidget::on_red_combo_box_valueChanged(const QString &arg1)
{
	InFlowData* flows = solver_->GetInFlows().data() + current_flow_;
	if (!flows) return;
	flows->red = arg1.toDouble();
	emit InflowChanged();
}

void InFlowWidget::on_red_combo_box_valueChanged(double arg1) 
{
	InFlowData* flows = solver_->GetInFlows().data() + current_flow_;
	if (!flows) return;
	flows->red = arg1;
	emit InflowChanged();
}

void InFlowWidget::on_green_combo_box_valueChanged(const QString &arg1)
{
	InFlowData* flows = solver_->GetInFlows().data() + current_flow_;
	if (!flows) return;
	flows->green = arg1.toDouble();
	emit InflowChanged();
}

void InFlowWidget::on_green_combo_box_valueChanged(double arg1)
{
	InFlowData* flows = solver_->GetInFlows().data() + current_flow_;
	if (!flows) return;
	flows->green = arg1;
	emit InflowChanged();
}

void InFlowWidget::on_blue_combo_box_valueChanged(const QString &arg1)
{
	InFlowData* flows = solver_->GetInFlows().data() + current_flow_;
	if (!flows) return;
	flows->blue = arg1.toDouble();
	emit InflowChanged();
}

void InFlowWidget::on_blue_combo_box_valueChanged(double arg1)
{
	InFlowData* flows = solver_->GetInFlows().data() + current_flow_;
	if (!flows) return;
	flows->blue = arg1;
	emit InflowChanged();
}

void InFlowWidget::on_x_origin_combo_box_valueChanged(double arg1)
{
	InFlowData* flows = solver_->GetInFlows().data() + current_flow_;
	if (!flows) return;
	flows->x = arg1;
	emit InflowChanged();
}

void InFlowWidget::on_x_origin_combo_box_valueChanged(const QString &arg1)
{
	InFlowData* flows = solver_->GetInFlows().data() + current_flow_;
	if (!flows) return;
	flows->x = arg1.toDouble();
	emit InflowChanged();
}

void InFlowWidget::on_y_origin_combo_box_valueChanged(double arg1)
{
	InFlowData* flows = solver_->GetInFlows().data() + current_flow_;
	if (!flows) return;
	flows->y = arg1;
	emit InflowChanged();
}

void InFlowWidget::on_y_origin_combo_box_valueChanged(const QString &arg1)
{
	InFlowData* flows = solver_->GetInFlows().data() + current_flow_;
	if (!flows) return;
	flows->y = arg1.toDouble();
	emit InflowChanged();
}

void InFlowWidget::on_radius_combo_box_valueChanged(double arg1)
{
	InFlowData* flows = solver_->GetInFlows().data() + current_flow_;
	if (!flows) return;
	flows->radius = arg1;
	emit InflowChanged();
}

void InFlowWidget::on_radius_combo_box_valueChanged(const QString &arg1)
{
	InFlowData* flows = solver_->GetInFlows().data() + current_flow_;
	if (!flows) return;
	flows->radius = arg1.toDouble();
	emit InflowChanged();
}

void InFlowWidget::on_addInFlowButton_pressed()
{
	if (solver_->AddInflowObject( 1.0, 0.0, 300.0, 0.0, 0.0, 255.0, 0.5, 0.1, 0.01 ) )
	{
		current_flow_ = solver_->GetInFlows().size() - 1;
		ui.inflow_selector->setMaximum( solver_->GetInFlows().size() - 1 );
		ui.inflow_selector->setValue( current_flow_ );
	}
}

void InFlowWidget::on_removeInFlowButton_pressed()
{
	if (solver_->removeInflowObject(current_flow_))
	{
		if (current_flow_ > 0)
		{
			current_flow_--;
			ui.inflow_selector->setMaximum( solver_->GetInFlows().size() - 1 );
			ui.inflow_selector->setValue( current_flow_ );
		}
	}
}

void InFlowWidget::SetDisplayValues_()
{
	if (!solver_->GetInFlows().data()) return;
	//dont go off firing signals for display changes
	this->blockSignals(true);

	InFlowData* flows = solver_->GetInFlows().data() + current_flow_;

	if (solver_->GetInFlows().size())
	{
		ui.u_velocity_combo_box->setValue((*flows).u_vel);
		ui.v_velocity_combo_box->setValue((*flows).v_vel);
		ui.density_combo_box->setValue((*flows).density);

		ui.red_combo_box->setValue((*flows).red);
		ui.blue_combo_box->setValue((*flows).blue);
		ui.green_combo_box->setValue((*flows).green);

		ui.x_origin_combo_box->setValue((*flows).x);
		ui.y_origin_combo_box->setValue((*flows).y);
		ui.radius_combo_box->setValue((*flows).radius);
	}

	this->blockSignals(false);
}
