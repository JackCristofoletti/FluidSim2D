#ifndef FLUID_SOLVER_2D_H
#define FLUID_SOLVER__2DH

#include "fluidquantity2D.h"

#include <memory>
#include <vector>

// This class handles writing to opengl textures through cuda
class FluidSolverGLCudaInteropHelper;

/* Fluid solver class. Sets up the fluid quantities, forces incompressibility
 * performs advection and adds inflows.
 */
class FluidSolver2D
{
public:
    FluidSolver2D( unsigned int w, unsigned int h, double density);

    ~FluidSolver2D();

    void Update( double timestep );

    /* Set density and x/y velocity in given rectangle to d/u/v, respectively */
    void AddInflow( double x, double y, double w, double h, double d, double u, double v );
	void AddInflowObject(double d, double u, double v, double r, double g, double b, double x, double y, double w, double h );

    /* Returns the maximum allowed timestep. Note that the actual timestep
     * taken should usually be much below this to ensure accurate
     * simulation - just never above.
	 * Not currently implemented
     */
    double CalcMaxTimeStep();

	//returns density data
	const double* ToImage() const; 

	unsigned GetWidth() const { return width_; }
	unsigned GetHeight() const { return height_; }

	/* This function will register a texture with a cuda resource, allowing us to write to it using a cuda kernel
	*  The assumption here is that the texture was created with a valid type for cuda-opengl texture interop and has the same width, height
	*  as the fluid solver
	*/
	void RegisterGLTexture(unsigned tex_id);

	std::vector<InFlowData>& GetInFlows() { return inflow_objects_; }

	//call this function when fluid flow params have changed and we need to trigger a reupload
	void ReUploadInFlows();

	//zeros out simulation
	void ResetSimulation();

private:
    /* Builds the pressure right hand side as the negative divergence */
    void BuildRhsPressure_();

    /* 
     * The Project operation enforces boundary conditions on solid walls and forces fluid incompressibility ( fluid volume does not change )
     * The operation will run untill either the iteration limit is reached
     * Advection algorithms can only be run in divergence free vector fields so this operation must occur first.
     */
    void Project_( int limit, double timestep );

    /* Applies the computed pressure to the velocity field */
    void ApplyPressure_( double timestep );

private:
    /* Fluid quantities */
    std::unique_ptr<FluidQuantity2D> fluid_concentration_;
    std::unique_ptr<FluidQuantity2D> u_velocity_;
	std::unique_ptr<FluidQuantity2D> v_velocity_;
	std::unique_ptr<FluidQuantity2D> red_, green_, blue_;
	std::vector<InFlowData> inflow_objects_;
	

    unsigned int width_;
    unsigned int height_;
    double cell_size_;
    double fluid_density_;

	//gpu buffers and helpers
	double* rhs_pressure_gpu_; /*Right hand side of pressure solve */
	double* pressure_gpu_; /* Pressure solution */
	double* projected_pressure_gpu_; /* tmp buffer used during projection*/
	InFlowData* inflow_objects_gpu_;
	bool inflow_objects_dirty_; //flag flowers for reupload
	std::unique_ptr<FluidSolverGLCudaInteropHelper> texture_writer_;
};

#endif;
