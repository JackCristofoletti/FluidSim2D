#include "fluidglwidget.h"
#include <iostream>
#include <QElapsedTimer>

FluidGLWidget::FluidGLWidget(QWidget *parent )
    : QOpenGLWidget(parent)
{}

void FluidGLWidget::ReUploadSolidBodies()
{
	fluid_simulation_->ReUploadSolidBodies();
}

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
   const int sizeX = 512;
   const int sizeY = 512;

   const double density = 0.1;

   fluid_simulation_.reset(  new FluidSolver2D( sizeX, sizeY, density ) );
   graphics_renderer_.reset( new Renderer() ); 
   InitFluidTexture_( sizeX, sizeY );

   //two example emitters
   fluid_simulation_->AddInflowObject(1.0, 0.0, 300.0, 0.0, 0.0, 255.0, 0.5, 0.1, 0.01);
   fluid_simulation_->AddInflowObject(1.0, 0.0, -600.0, 255.0, 0.0, 0.0, 0.5, 0.8, 0.01);

   //example solidBody
   SolidBody body( 0.5f, 0.3f, 0.1f, 0.1f );
   SolidBody circle( 0.5f, 0.6f, 0.05f );
   fluid_simulation_->AddSolidBody( body );
   fluid_simulation_->AddSolidBody( circle );

   //call this first so we can display the pointer data in the widget
   if (inflowFluidSolverCallback_)
	   inflowFluidSolverCallback_(fluid_simulation_.get());

   if (solidBodyFluidSolverCallback_)
	   solidBodyFluidSolverCallback_( fluid_simulation_.get() );

   graphics_renderer_->InitGL();
}

void FluidGLWidget::paintGL()
{
	QElapsedTimer timer;
	timer.start();
    UpdateSimulation_();
    glClear(GL_COLOR_BUFFER_BIT);
    graphics_renderer_->RenderTexture( fluid_texture_ );
	graphics_renderer_->RenderSolidBodies( fluid_simulation_->GetSolidBodies() );
	std::cout << "Frames Per Second: " << 1.0f / float( timer.elapsed() / 1000.0f ) << '\r';
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