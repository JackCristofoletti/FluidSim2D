#ifndef FLUID_SOLVER_2D_H
#define FLUID_SOLVER_2D_H

#include "InFlowData.h"
#include <memory>
#include <vector>

// This class handles writing to opengl textures through cuda
class FluidSolverGLCudaInteropHelper;
class FluidSolver2DPrivate;

#define TX 16 // number of threads per block along x-axis
#define TY 16 // number of threads per block along y-axis

/* Fluid solver class. Sets up the fluid quantities, forces incompressibility
* performs advection and adds inflows.
*/
class FluidSolver2D
{
public:
	FluidSolver2D(unsigned int w, unsigned int h, double density);

	~FluidSolver2D();

	void Update(double timestep);

	//add a fluid emmsion object to the scene
	void AddInflowObject(float d, float u, float v, float r, float g, float b, float x, float y, float rad );

	//returns density data
	const double* ToImage() const;

	unsigned GetWidth() const;
	unsigned GetHeight() const;

	/* This function will register a texture with a cuda resource, allowing us to write to it using a cuda kernel
	*  The assumption here is that the texture was created with a valid type for cuda-opengl texture interop and has the same width, height
	*  as the fluid solver
	*/
	void RegisterGLTexture(unsigned tex_id);

	std::vector<InFlowData>& GetInFlows();

	//call this function when fluid flow params have changed and we need to trigger a reupload
	void ReUploadInFlows();

	//zeros out simulation
	void ResetSimulation();

private:
	//Binds our cuda arrays to textures
	void BindTextures_();
	void UnBindTextures_();

	void CreateObstacleBorder_(); //fills the obstacle texture along the grid border
	void AddInFlows_();           //Adds speed, color and density to simulation
	void AdvectCuda_( float timestep); //advect whatever you want
	void DivergenceCuda_();
	void JacobiStepCuda_();
	void ProjectionCuda_();

	void Setup2DTexturesCuda_(); //sets fitler modes on our texture resources

private:
	std::unique_ptr<FluidSolver2DPrivate> d_;
};

#endif
