#ifndef FLUIDGLWIDGET_H
#define FLUIDGLWIDGET_H

#include "../Fluid/fluidsolver2D.h"
#include "../Graphics/renderer.h"

#include <QWidget>
#include <QOpenGLWidget>
#include <QOpenGLFunctions_4_5_Compatibility>

#include <functional>


class FluidGLWidget : public QOpenGLWidget, public QOpenGLFunctions_4_5_Compatibility
{
    Q_OBJECT

public:
    FluidGLWidget(QWidget *parent = 0);
    ~FluidGLWidget() = default;

	void SetInflowCallback(std::function< void(InFlowData*) > callback) { inflow_data_callback_ = callback; }
	void SetNumInFlowsCallback(std::function< void(unsigned) > callback) { num_inflows_callback_ = callback; }

public slots:
    void cleanup() {}
	void ReUploadInFlows();
	void Start();
	void Pause();
	void Stop();

protected:
    void initializeGL();
    void paintGL();
    void resizeGL(int width, int height);
private:
    void UpdateSimulation_();
	// Create and Register texture for opengl - cuda interop
	void InitFluidTexture_(unsigned width, unsigned height);

private:
    std::unique_ptr<FluidSolver2D> fluid_simulation_;
    std::unique_ptr<Renderer> graphics_renderer_;
    unsigned int fluid_texture_; //fluid color 

	//callback functions for widget, sim comunication
	std::function< void(InFlowData*) > inflow_data_callback_;
	std::function< void(unsigned) > num_inflows_callback_;
	bool running_ = false;
	
};

#endif // FLUIDGLWIDGET_H
