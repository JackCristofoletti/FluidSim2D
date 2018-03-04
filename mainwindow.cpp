#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "InFlowWidget.hpp"

#include <iostream>

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
	ui->setupUi(this);
	// Inflow Widget will finish it's setup once InitGL is called on the fluid widget
	ui->widget->SetInflowFluidSolverCallback( ( std::bind( &InFlowWidget::SetFluidSolver, ui->inflows_tab, std::placeholders::_1 ) ) );
	ui->widget->SetSolidBodyFluidSolverCallback( ( std::bind( &SolidBodyTab::SetFluidSolver, ui->solidBodyTab, std::placeholders::_1 )) );

	QObject::connect( ui->inflows_tab, SIGNAL( InflowChanged() ), ui->widget, SLOT( ReUploadInFlows() ) );
	QObject::connect( ui->solidBodyTab, SIGNAL( SolidBodyChanged() ), ui->widget, SLOT( ReUploadSolidBodies() ) );
}

void MainWindow::on_start_button_clicked()
{
	ui->widget->Start();
	ui->solidBodyTab->setDisabled( true );
}

void MainWindow::on_pause_button_clicked()
{
	ui->widget->Pause();
}

void MainWindow::on_stop_button_clicked()
{
	ui->widget->Stop();
	ui->solidBodyTab->setEnabled( true );
}

MainWindow::~MainWindow()
{
    delete ui;
}
