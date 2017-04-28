#include "fluidglwidget.h"
#include <iostream>

FluidGLWidget::FluidGLWidget(QWidget *parent )
    : QOpenGLWidget(parent)
{}

void FluidGLWidget::Start()
{
	running_ = true;
	update();
}

void FluidGLWidget::Pause()
{
	running_ = false;
	update();
}

void FluidGLWidget::Stop()
{
	fluid_simulation_->ResetSimulation();
	running_ = false;
	update();
}

void FluidGLWidget::initializeGL()
{
   initializeOpenGLFunctions();

    /* Play with these constants, if you want */
   const int sizeX = 256;
   const int sizeY = 256;

   const double density = 0.1;

   fluid_simulation_.reset(  new FluidSolver2D( sizeX, sizeY, density ) );
   graphics_renderer_.reset( new Renderer() ); 
   InitFluidTexture_( sizeX, sizeY );

   fluid_simulation_->AddInflowObject(1.0, 0.0, 3.0, 0.0, 45.0, 255.0, 0.5, 0.5, 0.01, 0.05);
   fluid_simulation_->AddInflowObject(1.0, 0.0, -6.0, 255.0, 70.0, 0.0, 0.5, 0.8, 0.01, 0.05);

   //call this first so we can display the pointer data in the widget
   if (num_inflows_callback_)
	   num_inflows_callback_(fluid_simulation_->GetInFlows().size());

   if (inflow_data_callback_)
	   inflow_data_callback_(fluid_simulation_->GetInFlows().data());
 
 
   graphics_renderer_->InitGL();
}

void FluidGLWidget::paintGL()
{
    UpdateSimulation_();
    glClear(GL_COLOR_BUFFER_BIT);
    graphics_renderer_->RenderTexture( fluid_texture_ );
    update();
}

void FluidGLWidget::resizeGL(int width, int height)
{
    glViewport(0, 0, width, height);
    update();
}

void FluidGLWidget::UpdateSimulation_()
{
	if (running_)
	{
		const double timestep = 0.0016;
		fluid_simulation_->Update(timestep);
	}
}

void FluidGLWidget::InitFluidTexture_(unsigned width, unsigned height)
{
	glGenTextures(1, &fluid_texture_);
	glBindTexture(GL_TEXTURE_2D, fluid_texture_);
	// set basic parameters
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	//CUDA Opengl Interop only works with a few texture types, the internal format below is one of the following
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8UI_EXT, width, height, 0, GL_RGBA_INTEGER_EXT, GL_UNSIGNED_BYTE, 0);
	glBindTexture(GL_TEXTURE_2D, 0);

	fluid_simulation_->RegisterGLTexture(fluid_texture_);
}

void FluidGLWidget::ReUploadInFlows()
{
	fluid_simulation_->ReUploadInFlows();
}