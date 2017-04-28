#ifndef FLUID_SOLVER_2D_GPU
#define FLUID_SOLVER_2D_GPU

#include "FluidQuantity2DGpu.h"
#include <vector>
#include <memory>

class FluidSolver2DGpu
{
public:
	FluidSolver2DGpu( unsigned w, unsigned h, double density );

	/*
	* Updates the fluid simulation and writes to a 2D GL texture specified by the id
	*/ 
	void Update( double time_step, unsigned texture_id );

	/* Set density and x/y velocity in given rectangle to d/u/v, respectively */
	void AddInflow(double x, double y, double w, double h, double d, double u, double v);

	/* Enforces bundary condtions at simulation extents and solid cells*/

	std::vector<double>& GetImage() { return cpu_density_; }

	unsigned GetWidth() { return width_; }
	unsigned GetHeight() { return height_; }

private:
	/* Builds the pressure right hand side as the negative divergence */
	void BuildRhsPressure_();
	/*
	* The Project operation enforces boundary conditions on solid walls and forces fluid incompressibility ( fluid volume does not change )
	* The operation will run untill either the iteration limit is reached or the error is below a certain threshold
	* Advection algorithms can only be run in divergence free vector fields so this operation must occur first.
	* Implementation can be found on pg 78
	*/
	void Project_(int limit, double timestep);

	/* Applies the computed pressure to the velocity field */

	void ApplyPressure_(double timestep);

private:
	/* Device Arrays for: */
	double* rhs_pressure_; /* Right hand side of pressure solve */
	double* pressure_;	   /* Pressure solution */
	double* new_pressure_;
	double* delta_buffer_; /* Projection Delta*/
	
	std::vector<double> cpu_density_; // cpu side buffer for density test code

	/* Grid cell size and fluid density */
	double cell_size_;
	double fluid_density_;

	unsigned width_, height_;

	std::unique_ptr<FluidQuantity2DGpu> u_, v_, concentration_;
};

#endif