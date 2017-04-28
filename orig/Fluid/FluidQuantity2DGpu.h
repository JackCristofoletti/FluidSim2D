#ifndef FLUID_QUANTITY_2D_GPU
#define FLUID_QUANTITY_2D_GPU


/*
* A class for representing a fluid quantity (this could be color, velocity, pressure etc. )
* Note that x,y coordinates refer to x,y in the cartesian sense and not matrix row/column
*/

struct FluidQuantityParams
{
	/* Width and height */
	unsigned width;
	unsigned height;
	/* X and Y offset from top left grid cell.
	* This is (0.5,0.5) for centered quantities such as density,
	* and (0.0, 0.5) or (0.5, 0.0) for jittered quantities like the velocity.
	*/
	double offset_x;
	double offset_y;
	/* Grid cell size */
	double cell_size;
};

class FluidQuantity2DGpu
{
public:
	FluidQuantity2DGpu(unsigned w, unsigned h, double ox, double oy, double cell_size );

	FluidQuantity2DGpu(const FluidQuantityParams& params);

	~FluidQuantity2DGpu();

	//make advection values visible
	void Flip();
	//value buffer
	const double *Src() const;
	//Params Buffer
	const FluidQuantityParams* GetParamsGPUBuffer() const { return gpu_params_; }
	const FluidQuantityParams& GetParams() { return params_; }

    double* GetSrcBuffer() const { return src_; }

	/* Read-only and read-write access to grid cells */
	double At(int x, int y) const;
	double &At(int x, int y);

	/* Linear intERPolate on grid at coordinates (x, y).
	* Coordinates will be clamped to lie in simulation domain
	*/
	double GridValue(double x, double y) const;

	/* Advect grid in velocity field u, v with given timestep */
	void Advect(double timestep, const FluidQuantity2DGpu &u, const FluidQuantity2DGpu &v);

	/* Sets fluid quantity inside the given rect to value `v' */
	void AddInflow( double x0, double y0, double x1, double y1, double v );

private:
	/* Device Memory buffers for fluid quantity and parameters */
	double* src_;
	double* dst_;
	FluidQuantityParams* gpu_params_;

	FluidQuantityParams params_;

};

#endif
