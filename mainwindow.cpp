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
	ui->widget->SetInflowCallback( ( std::bind( &InFlowWidget::SetInFlows, ui->inflows_tab, std::placeholders::_1 ) ) );
	ui->widget->SetNumInFlowsCallback( ( std::bind( &InFlowWidget::SetInflowSize, ui->inflows_tab, std::placeholders::_1 ) ) );

	QObject::connect( ui->inflows_tab, SIGNAL( InflowChanged() ), ui->widget, SLOT( ReUploadInFlows() ) );

	QObject::connect( ui->start_button, SIGNAL( clicked() ), ui->widget, SLOT( Start() ) );
	QObject::connect( ui->pause_button, SIGNAL( clicked() ), ui->widget, SLOT( Pause() ) );
	QObject::connect( ui->stop_button, SIGNAL( clicked() ), ui->widget, SLOT( Stop() ) );
}

MainWindow::~MainWindow()
{
    delete ui;
}
